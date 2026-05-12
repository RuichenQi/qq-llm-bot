"""Group-wide memory + /recap + human-send splitter."""
from __future__ import annotations

import asyncio
import time
import types
from typing import List, Tuple

import pytest

import config as cfg
from bot.command_handler import Handler
from bot.group_memory import GroupMemory
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, group_id: int = 1,
           nickname: str = "tester"):
    segs: list = []
    if text.startswith("/"):
        segs.append({"type": "at", "data": {"qq": "10000"}})
    if text:
        segs.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": nickname, "card": nickname},
    }


def _make_handler(monkeypatch, *, chat_reply: str = "ok"):
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    captured_messages: list[list] = []
    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **kw):
        from providers.base import TextReply
        captured_messages.append(list(messages))
        return TextReply(text=chat_reply, model="stub", usage={})

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)

    handler = Handler(
        deepseek=stub,
        openai=None,
        router=router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
        group_memory=GroupMemory(),
    )
    return handler, sent, captured_messages


# ---------- GroupMemory direct tests ----------
def test_group_memory_records_messages():
    async def run():
        gm = GroupMemory()
        await gm.append(1, 100, "Alice", "今天好热啊")
        await gm.append(1, 101, "Bob", "是啊 35度")
        await gm.append(2, 100, "Alice", "另一个群的消息")
        rows = await gm.recent(1, limit=10)
        assert [r.text for r in rows] == ["今天好热啊", "是啊 35度"]
        assert [r.nickname for r in rows] == ["Alice", "Bob"]

    asyncio.run(run())


def test_group_memory_since_filter():
    async def run():
        gm = GroupMemory()
        # All real timestamps are now-ish; just verify the filter works.
        await gm.append(1, 100, "A", "old")
        cutoff = time.time() - 0.001
        await gm.append(1, 100, "A", "fresh")
        recent = await gm.since(1, cutoff)
        assert any(r.text == "fresh" for r in recent)

    asyncio.run(run())


# ---------- Handler logs every msg to GroupMemory ----------
def test_handler_logs_every_msg_to_group_memory(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)
    # Send a router-skipped message — bot stays silent but should still record.

    async def decide_skip(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("skip", 1.0, "skip", text)

    handler.router.decide = decide_skip

    async def run():
        await handler.handle(parse_event(_event("纯水群里聊", nickname="Alice")))
        await asyncio.sleep(0.05)  # let the create_task settle
        rows = await handler.group_memory.recent(1, limit=10)
        return rows

    rows = asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any(r.text == "纯水群里聊" for r in rows)


# ---------- Group context is injected into chat ----------
def test_group_context_appears_in_chat_messages(monkeypatch):
    handler, _, captured = _make_handler(monkeypatch, chat_reply="嗯")
    monkeypatch.setattr(cfg.CONFIG, "group_context_turns", 5, raising=False)

    async def run():
        # Pre-populate group memory
        await handler.group_memory.append(1, 200, "Alice", "晚上吃啥")
        await handler.group_memory.append(1, 201, "Bob", "麻辣烫吧")
        # Now trigger a chat
        await handler.handle(parse_event(_event("你呢", nickname="C")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured, "chat() should have been called"
    system_content = "\n".join(
        m.content for m in captured[-1] if m.role == "system"
    )
    assert "晚上吃啥" in system_content
    assert "Alice" in system_content
    assert "麻辣烫吧" in system_content


# ---------- _human_send chunking ----------
def test_split_human_chunks_one_sentence():
    chunks = Handler._split_human_chunks("hello", max_chunks=3)
    assert chunks == ["hello"]


def test_split_human_chunks_chinese_sentences():
    text = "嗯。让我想想。也许是天气太热了。"
    chunks = Handler._split_human_chunks(text, max_chunks=3)
    assert len(chunks) <= 3
    assert "".join(c.replace(" ", "") for c in chunks).startswith("嗯")


def test_split_human_chunks_groups_when_over_limit():
    text = "一。二。三。四。五。六。"
    chunks = Handler._split_human_chunks(text, max_chunks=2)
    assert len(chunks) == 2


def test_human_send_disabled_falls_back_to_one_message(monkeypatch):
    # conftest disables by default; just confirm
    monkeypatch.setattr(cfg.CONFIG, "human_send_enabled", False, raising=False)
    handler, sent, _ = _make_handler(monkeypatch)
    asyncio.run(handler._human_send(1, "嗯。让我想想。也许是天气太热了。"))
    asyncio.run(handler.aclose())
    # When disabled we expect one combined send (chunk_text only splits at
    # MAX_REPLY_CHARS, which is huge).
    assert len(sent) == 1


# ---------- /recap ----------
def test_recap_with_no_messages(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)

    async def run():
        await handler.handle(parse_event(_event("/recap 1h")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any("没啥消息" in t for _, t in sent)


def test_recap_calls_deepseek_with_transcript(monkeypatch):
    handler, sent, captured = _make_handler(monkeypatch, chat_reply="今天大家在聊吃")

    async def run():
        # seed group memory
        await handler.group_memory.append(1, 100, "Alice", "晚上吃啥")
        await handler.group_memory.append(1, 101, "Bob", "麻辣烫吧")
        await handler.group_memory.append(1, 102, "Carol", "我想吃日料")
        await handler.handle(parse_event(_event("/recap 1h")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured, "deepseek.chat should have been called"
    user_msgs = [m.content for m in captured[-1] if m.role == "user"]
    assert user_msgs and "麻辣烫" in user_msgs[0]
    assert any("今天大家在聊吃" in t for _, t in sent)


def test_recap_bad_period(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/recap garbageblob"))))
    asyncio.run(handler.aclose())
    assert any("用法" in t for _, t in sent)
