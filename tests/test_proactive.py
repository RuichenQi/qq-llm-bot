"""Proactive interjection: hard pre-filter + LLM judgment."""
from __future__ import annotations

import asyncio
import time
import types
from typing import List, Tuple
from unittest.mock import patch

import pytest

import config as cfg
from bot.command_handler import Handler
from bot.group_memory import GroupMemory
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, group_id: int = 1, nickname: str = "Alice"):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": [{"type": "text", "data": {"text": text}}],
        "sender": {"user_id": user_id, "nickname": nickname, "card": nickname},
    }


def _make_handler(monkeypatch, *, judge_reply: str = '{"r":"skip"}'):
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    captured_calls: list[list] = []
    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **kw):
        from providers.base import TextReply
        captured_calls.append(list(messages))
        return TextReply(text=judge_reply, model="stub", usage={})

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("skip", 1.0, "skip", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    # trigger gate: bot has nickname-only; messages without @ won't match
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "mention", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_enabled", True, raising=False)
    # Default to never triggering randomly — tests override per-case.
    monkeypatch.setattr(cfg.CONFIG, "proactive_probability", 0.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_min_seconds", 60, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_min_new_messages", 3, raising=False)

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
    return handler, sent, captured_calls


# ---------- gate tests ----------
def test_zero_probability_never_fires(monkeypatch):
    handler, sent, captured = _make_handler(monkeypatch)

    async def run():
        for _ in range(50):
            await handler.handle(parse_event(_event("hi")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured == [], "judge should never have been called"
    assert sent == []


def test_full_probability_fires_once_conditions_met(monkeypatch):
    handler, sent, captured = _make_handler(monkeypatch, judge_reply='{"r":"skip"}')
    monkeypatch.setattr(cfg.CONFIG, "proactive_probability", 1.0, raising=False)
    # Set bot speech "long ago" so seconds gate passes.
    handler._last_bot_speech_at[1] = time.monotonic() - 10_000

    async def run():
        # Send 3 messages → meets min_new_messages threshold.
        for i in range(3):
            await handler.handle(parse_event(_event(f"hello{i}")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    # Judge should have been called at least once; it said skip so no message.
    assert captured, "judge should have been invoked"
    assert sent == []


def test_judge_say_sends_to_group(monkeypatch):
    handler, sent, captured = _make_handler(
        monkeypatch, judge_reply='{"r":"say","t":"嘿，吃啥呢"}'
    )
    monkeypatch.setattr(cfg.CONFIG, "proactive_probability", 1.0, raising=False)
    handler._last_bot_speech_at[1] = time.monotonic() - 10_000

    async def run():
        for i in range(3):
            await handler.handle(parse_event(_event(f"food talk {i}")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any("吃啥" in t for _, t in sent), f"expected proactive say, got {sent}"


def test_recent_speech_blocks_proactive(monkeypatch):
    handler, sent, captured = _make_handler(
        monkeypatch, judge_reply='{"r":"say","t":"hi"}'
    )
    monkeypatch.setattr(cfg.CONFIG, "proactive_probability", 1.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_min_seconds", 60, raising=False)
    # Stamp speech as if bot just spoke
    handler._last_bot_speech_at[1] = time.monotonic()

    async def run():
        for i in range(5):
            await handler.handle(parse_event(_event(f"x{i}")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured == [], "judge should be gated by recent-speech check"


def test_too_few_messages_blocks(monkeypatch):
    handler, sent, captured = _make_handler(
        monkeypatch, judge_reply='{"r":"say","t":"hi"}'
    )
    monkeypatch.setattr(cfg.CONFIG, "proactive_probability", 1.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_min_new_messages", 10, raising=False)
    handler._last_bot_speech_at[1] = time.monotonic() - 10_000

    async def run():
        # Only 2 messages; min_new_messages is 10
        for i in range(2):
            await handler.handle(parse_event(_event(f"x{i}")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured == [], "should not call judge below message threshold"


# ---------- coerce parsing ----------
def test_coerce_say_returns_text():
    out = Handler._coerce_proactive_decision('{"r":"say","t":"嘿"}')
    assert out == "嘿"


def test_coerce_skip_returns_none():
    assert Handler._coerce_proactive_decision('{"r":"skip"}') is None


def test_coerce_garbage_returns_none():
    assert Handler._coerce_proactive_decision("not json") is None


def test_coerce_overlong_text_returns_none():
    long_text = "x" * 200
    out = Handler._coerce_proactive_decision(
        '{"r":"say","t":"' + long_text + '"}'
    )
    assert out is None
