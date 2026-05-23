"""Per-group reply cooldown."""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import pytest

import config as cfg
from bot import command_handler as ch_mod
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, user_id: int = 42, group_id: int = 1):
    # Commands now require @bot, so auto-add it for test convenience.
    segs: list = []
    if text.startswith("/"):
        segs.append({"type": "at", "data": {"qq": "10000"}})
    segs.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(*a, **k):
        from providers.base import TextReply
        return TextReply(text="ok", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
    ), sent


def test_cooldown_blocks_second_message(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 30, raising=False)
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("hello"))))
    asyncio.run(handler.handle(parse_event(_event("hello again"))))
    asyncio.run(handler.aclose())
    assert len(sent) == 1, f"second message should be silenced, got {sent}"
    assert sent[0][1] == "ok"


def test_cooldown_zero_means_disabled(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)
    handler, sent = _make_handler(monkeypatch)
    for _ in range(3):
        asyncio.run(handler.handle(parse_event(_event("hi"))))
    asyncio.run(handler.aclose())
    assert len(sent) == 3


def test_cooldown_does_not_block_commands(monkeypatch):
    """Explicit /commands should bypass cooldown — users invoked them."""
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 30, raising=False)
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("free chat"))))
    asyncio.run(handler.handle(parse_event(_event("/help"))))
    asyncio.run(handler.handle(parse_event(_event("/balance"))))
    asyncio.run(handler.aclose())
    # 1 natural + 1 /help + 1 /balance = 3 replies
    assert len(sent) == 3
    assert any("/ask" in t for _, t in sent)  # /help fired
    assert any("额度" in t for _, t in sent)   # /balance fired


def test_cooldown_is_per_group(monkeypatch):
    """A reply in group 1 should not silence group 2."""
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 30, raising=False)
    handler, sent = _make_handler(monkeypatch)
    # _make_handler resets allowed_groups → override AFTER it runs.
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 2}, raising=False)
    asyncio.run(handler.handle(parse_event(_event("hi", group_id=1))))
    asyncio.run(handler.handle(parse_event(_event("hi", group_id=2))))
    asyncio.run(handler.aclose())
    assert len(sent) == 2
    assert {gid for gid, _ in sent} == {1, 2}


def test_cooldown_releases_after_window(monkeypatch):
    """Once the cooldown window elapses, replies resume."""
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 5, raising=False)
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("first"))))
    # Simulate clock advance by rewinding the stamp.
    handler._last_dispatch_at[1] -= 10  # pretend last reply was 10s ago
    asyncio.run(handler.handle(parse_event(_event("second"))))
    asyncio.run(handler.aclose())
    assert len(sent) == 2


def test_router_skip_does_not_consume_cooldown(monkeypatch):
    """If the router says skip, the bot stays silent AND the cooldown stamp
    is rolled back, so the next genuine message can still get a reply."""
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 30, raising=False)
    handler, sent = _make_handler(monkeypatch)

    # Both handle calls must share one event loop because aiosqlite binds
    # its worker thread to whichever loop opened the connection.
    async def both_calls():
        async def decide_skip(text, *, has_image, was_at_bot=False, **_kw):
            from bot.router import RouteDecision
            return RouteDecision("skip", 1.0, "off_topic", text)

        handler.router.decide = decide_skip
        await handler.handle(parse_event(_event("they were talking about food")))
        assert sent == [], f"skip should be silent, got {sent}"
        assert 1 not in handler._last_dispatch_at

        async def decide_chat(text, *, has_image, was_at_bot=False, **_kw):
            from bot.router import RouteDecision
            return RouteDecision("deepseek_chat", 1.0, "chat", text)

        handler.router.decide = decide_chat
        await handler.handle(parse_event(_event("hi bot")))
        await handler.aclose()

    asyncio.run(both_calls())
    assert len(sent) == 1
    assert sent[0][1] == "ok"
