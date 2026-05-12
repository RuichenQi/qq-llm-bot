"""Trigger-mode gate (no network)."""
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


def _event(text: str, *, at: list[int] | None = None, image_url: str | None = None):
    segs: list = []
    if at:
        for q in at:
            segs.append({"type": "at", "data": {"qq": str(q)}})
    if text:
        segs.append({"type": "text", "data": {"text": text}})
    if image_url:
        segs.append({"type": "image", "data": {"url": image_url}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": 42,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": 42, "nickname": "x"},
    }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)


def _make_handler(monkeypatch) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid: int, text: str) -> None:
        sent.append((gid, text))

    async def send_image(gid: int, img: str) -> None:
        sent.append((gid, f"[image:{img[:40]}]"))

    stub_provider = types.SimpleNamespace(name="stub")

    async def stub_chat(messages, **kw):
        from providers.base import TextReply
        return TextReply(text="ok", model="stub")

    stub_provider.chat = stub_chat

    async def stub_aclose():
        return None

    stub_provider.aclose = stub_aclose

    stub_router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    stub_router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)

    handler = Handler(
        deepseek=stub_provider,
        openai=None,
        router=stub_router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
    )
    return handler, sent


def test_trigger_always(monkeypatch, patched):
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("hello"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent and sent[0][1] == "ok"


def test_trigger_mention_blocks_without_at(monkeypatch, patched):
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "mention", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("hello"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent == []


def test_trigger_mention_passes_with_at(monkeypatch, patched):
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "mention", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("hello", at=[10000]))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent and sent[0][1] == "ok"


def test_trigger_prefix_strips(monkeypatch, patched):
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "prefix", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_prefix", "#", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("#  ping"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent and sent[0][1] == "ok"


def test_trigger_prefix_blocks_unprefixed(monkeypatch, patched):
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "prefix", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_prefix", "#", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("hello"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent == []


def test_command_without_at_is_ignored(monkeypatch, patched):
    """Commands now require @bot too — bare /help in the group is silenced."""
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("/help"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent == []


def test_command_with_at_bot_passes(monkeypatch, patched):
    """`@bot /help` should fire the command."""
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    handler, sent = _make_handler(monkeypatch)
    parsed = parse_event(_event("/help", at=[10000]))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert sent and "/ask" in sent[0][1]
