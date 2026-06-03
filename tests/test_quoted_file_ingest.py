"""End-to-end: user replies to a previous file message → bot reads the file.

This verifies the path that ties together:
  - message_parser.parse_event picking up the [CQ:reply] segment
  - main._extract_quoted carrying [CQ:file] out of the quoted payload
  - Handler.handle copying quoted.files into msg.files
  - _ingest_files downloading the file and prepending extracted text
"""
from __future__ import annotations

import asyncio
import types
from typing import List, Optional, Tuple

import httpx

import config as cfg
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import AttachedFile, QuotedMessage, parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _reply_event(text: str, *, reply_id: str = "QUOTE-1") -> dict:
    """A group message that quotes some other message + adds new text."""
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": 42,
        "raw_message": text,
        "message": [
            {"type": "reply", "data": {"id": reply_id}},
            {"type": "at", "data": {"qq": "10000"}},
            {"type": "text", "data": {"text": " " + text}},
        ],
        "sender": {"user_id": 42, "nickname": "x"},
    }


def _make_handler(monkeypatch, fetch_reply, file_bytes: bytes
                  ) -> Tuple[Handler, List[Tuple[int, str]], List[List]]:
    sent: List[Tuple[int, str]] = []
    captured_messages: List[List] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
        captured_messages.append(list(messages))
        return TextReply(text="ok", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image=False, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "file_ingest_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    handler = Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        fetch_reply=fetch_reply,
    )
    # Stub the http client so _download() returns our fake bytes without
    # touching the network.
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=file_bytes)
    )
    handler._http = httpx.AsyncClient(transport=transport)
    return handler, sent, captured_messages


def test_user_replies_to_text_file_message(monkeypatch):
    """The classic case: a group member sent foo.txt earlier; another user
    replies to that message + asks '这文件讲啥'. Bot should ingest the file
    and answer based on its content."""
    file_body = "这份周报的核心结论是利润涨了 12%，比上个季度好。".encode("utf-8")

    async def fetch_reply(msg_id):
        assert msg_id == "QUOTE-1"
        return QuotedMessage(
            text="",  # original message was just the file upload
            image_urls=[],
            files=[AttachedFile(
                name="周报.txt",
                url="https://example.com/zhoubao.txt",
                file_id="abc",
                size=len(file_body),
            )],
        )

    handler, sent, captured = _make_handler(
        monkeypatch, fetch_reply=fetch_reply, file_bytes=file_body,
    )
    asyncio.run(handler.handle(parse_event(_reply_event("这文件讲啥"))))
    asyncio.run(handler.aclose())

    # Bot replied via the chat route.
    assert sent and sent[0][1] == "ok"
    # The chat call saw a user message that includes the file content.
    assert captured, "deepseek.chat should have been invoked"
    user_msgs = [m for m in captured[-1] if m.role == "user"]
    assert user_msgs, "must have a user message"
    body = user_msgs[-1].content
    assert "周报.txt" in body, "filename must be visible to the LLM"
    assert "利润涨了 12%" in body, "extracted text must reach the prompt"
    assert "这文件讲啥" in body, "user's question must be preserved"


def test_user_replies_to_code_file(monkeypatch):
    """Code files get wrapped in a markdown fence so the LLM recognises them."""
    code = b"def add(a, b):\n    return a + b\n"

    async def fetch_reply(msg_id):
        return QuotedMessage(
            text="",
            image_urls=[],
            files=[AttachedFile(
                name="util.py", url="https://example.com/util.py",
                file_id="", size=len(code),
            )],
        )

    handler, sent, captured = _make_handler(
        monkeypatch, fetch_reply=fetch_reply, file_bytes=code,
    )
    asyncio.run(handler.handle(parse_event(_reply_event("解释一下"))))
    asyncio.run(handler.aclose())

    body = "\n".join(m.content for m in captured[-1] if m.role == "user")
    assert "```py" in body, "code file should be fenced"
    assert "def add" in body


def test_quoted_file_does_not_double_ingest_when_already_attached(monkeypatch):
    """If the user attaches a file AND replies to one, only the user's own
    attachment is ingested (the quote is treated as just text context)."""
    user_file_body = b"USER'S OWN FILE"
    quoted_file_body = b"QUOTED FILE (should be ignored)"

    async def fetch_reply(msg_id):
        return QuotedMessage(
            text="原始文件",
            image_urls=[],
            files=[AttachedFile(
                name="quoted.txt", url="https://example.com/quoted.txt",
                file_id="", size=len(quoted_file_body),
            )],
        )

    handler, sent, captured = _make_handler(
        monkeypatch, fetch_reply=fetch_reply, file_bytes=user_file_body,
    )
    # Build an event where the user attaches their OWN file in this message
    # AND replies to the previous one.
    event = _reply_event("看看")
    event["message"].append({
        "type": "file",
        "data": {
            "file": "mine.txt", "url": "https://example.com/mine.txt",
            "file_id": "u1",
        },
    })
    asyncio.run(handler.handle(parse_event(event)))
    asyncio.run(handler.aclose())

    body = "\n".join(m.content for m in captured[-1] if m.role == "user")
    assert "mine.txt" in body
    # quoted.txt must NOT be ingested because msg.files was already non-empty.
    assert "quoted.txt" not in body


def test_fetch_reply_failure_falls_back_to_text(monkeypatch):
    """If fetch_reply explodes, the bot still tries to answer based on the
    user's text alone — it shouldn't crash."""
    async def fetch_reply(msg_id):
        raise RuntimeError("OneBot API down")

    handler, sent, _ = _make_handler(
        monkeypatch, fetch_reply=fetch_reply, file_bytes=b"",
    )
    asyncio.run(handler.handle(parse_event(_reply_event("帮我看看"))))
    asyncio.run(handler.aclose())
    assert sent and sent[0][1] == "ok"


def test_quoted_file_without_url_reports_failure(monkeypatch):
    """If the quoted payload has a file_id but no URL and we have no
    fetch_file_url callback, the bot reports the issue rather than crashing."""
    async def fetch_reply(msg_id):
        return QuotedMessage(
            text="",
            image_urls=[],
            files=[AttachedFile(
                name="orphan.pdf", url="",     # no URL
                file_id="opaque-id", size=0,    # only a file_id
            )],
        )

    handler, sent, captured = _make_handler(
        monkeypatch, fetch_reply=fetch_reply, file_bytes=b"",
    )
    # No fetch_file_url callback registered.
    assert handler.fetch_file_url is None
    asyncio.run(handler.handle(parse_event(_reply_event("这啥?"))))
    asyncio.run(handler.aclose())
    # The chat call still happens — the bot reports that the file's
    # download link couldn't be resolved.
    body = "\n".join(m.content for m in captured[-1] if m.role == "user")
    assert "orphan.pdf" in body
    assert "拿不到下载链接" in body or "下载失败" in body
