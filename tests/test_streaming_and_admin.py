"""Streaming text + /admin commands + reply segments + image disk cache."""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import pytest

import config as cfg
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, reply_id: str | None = None,
           image_url: str | None = None):
    segs: list = []
    if reply_id is not None:
        segs.append({"type": "reply", "data": {"id": reply_id}})
    if text:
        segs.append({"type": "text", "data": {"text": text}})
    if image_url:
        segs.append({"type": "image", "data": {"url": image_url}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch, *, stream_chunks=None, fetch_reply=None) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid: int, text: str) -> None:
        sent.append((gid, text))

    async def send_image(gid: int, img: str) -> None:
        sent.append((gid, f"[image:{img[:40]}]"))

    stub_provider = types.SimpleNamespace(name="stub")

    async def stub_chat(messages, **kw):
        from providers.base import TextReply
        return TextReply(text="non-streamed-ok", model="stub")

    stub_provider.chat = stub_chat

    async def stub_chat_stream(messages, **kw):
        for c in (stream_chunks or ["hel", "lo ", "world"]):
            yield c

    stub_provider.chat_stream = stub_chat_stream

    async def stub_aclose():
        return None

    stub_provider.aclose = stub_aclose

    stub_router = types.SimpleNamespace()

    async def decide(text, *, has_image):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    stub_router.decide = decide

    handler = Handler(
        deepseek=stub_provider,
        openai=None,
        router=stub_router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
        fetch_reply_text=fetch_reply,
    )
    return handler, sent


@pytest.fixture(autouse=True)
def patch_allowed(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)


def test_streaming_flushes_at_threshold(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_flush_chars", 3, raising=False)
    # 9 chars total → expect 3 flushes of size 3 each
    handler, sent = _make_handler(monkeypatch, stream_chunks=["123", "456", "789"])
    parsed = parse_event(_event("hi"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    texts = [t for _, t in sent]
    assert texts == ["123", "456", "789"]


def test_streaming_flushes_on_paragraph(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_flush_chars", 9999, raising=False)
    handler, sent = _make_handler(monkeypatch, stream_chunks=["para1", "\n\n", "para2"])
    parsed = parse_event(_event("hi"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    texts = [t for _, t in sent]
    assert texts[0] == "para1"
    assert "para2" in texts[1]


def test_streaming_disabled_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("hi"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent == [(1, "non-streamed-ok")]


def test_reply_segment_prepends_quoted_text(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)

    seen_messages: list[str] = []

    async def stub_chat(messages, **kw):
        from providers.base import TextReply
        seen_messages.append(messages[-1].content)  # last is user
        return TextReply(text="ok", model="stub")

    handler, _ = _make_handler(monkeypatch)
    # swap chat to capture
    handler.deepseek.chat = stub_chat

    async def fetch(mid):
        assert mid == "xyz"
        return "之前那个柴犬的图"

    handler.fetch_reply_text = fetch
    parsed = parse_event(_event("更可爱一点", reply_id="xyz"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert seen_messages, "expected chat() to be called"
    assert "之前那个柴犬的图" in seen_messages[0]
    assert "更可爱一点" in seen_messages[0]


def test_admin_blocks_non_superuser(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "superusers", set(), raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("/admin status", user_id=42))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent and "管理员" in sent[0][1]


def test_admin_allow_group_round_trip(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    handler, sent = _make_handler(monkeypatch)
    # add a group
    parsed = parse_event(_event("/admin allow_group 12345", user_id=42))
    asyncio.run(handler.handle(parsed))
    # list it
    parsed = parse_event(_event("/admin list_groups", user_id=42))
    asyncio.run(handler.handle(parsed))
    # remove it
    parsed = parse_event(_event("/admin disallow_group 12345", user_id=42))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    flat = "\n".join(t for _, t in sent)
    assert "已添加" in flat
    assert "12345" in flat
    assert "已移除" in flat


def test_admin_disallow_env_pinned_refuses(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 999}, raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("/admin disallow_group 999", user_id=42))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert any("固定" in t for _, t in sent)


def test_message_parser_reply_segment():
    ev = _event("回复看看", reply_id="abc-123")
    p = parse_event(ev)
    assert p is not None
    assert p.reply_to_msg_id == "abc-123"


def test_image_disk_cache_saves_file(monkeypatch):
    """When a message carries an image, bytes should land in IMAGE_DIR."""
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    handler, sent = _make_handler(monkeypatch)

    async def fake_download(url):
        return b"\x89PNG\x00fake-bytes"

    handler._download = fake_download
    parsed = parse_event(_event("看图", image_url="https://example.com/x.png"))

    async def run():
        await handler.handle(parsed)
        # give the create_task scheduling a chance
        await asyncio.sleep(0)
        # wait for the cache task to finish
        for _ in range(20):
            if any((cfg.IMAGE_DIR).glob("*.dat")):
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())
    asyncio.run(handler.aclose())
    files = list(cfg.IMAGE_DIR.glob("*.dat"))
    assert files, "expected at least one cached image file"
    assert files[0].read_bytes() == b"\x89PNG\x00fake-bytes"
