"""Hard-rule pre-filter for bare-image messages."""
from __future__ import annotations

import asyncio
import types

import config as cfg
from bot.router import Router


def _stub_deepseek(captured):
    """A DeepSeek stub that records every chat() call so we can prove
    the router never called it."""
    ds = types.SimpleNamespace(name="deepseek")

    async def chat(messages, **kw):
        captured.append(messages)
        from providers.base import TextReply
        return TextReply(text='{"r":"chat"}', model="stub", usage={})

    ds.chat = chat
    return ds


def test_bare_image_no_text_skips_without_llm(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    calls = []
    router = Router(_stub_deepseek(calls))
    d = asyncio.run(router.decide("", has_image=True, was_at_bot=False))
    assert d.route == "skip"
    assert d.reason == "bare_image"
    assert calls == [], "LLM should NOT have been called for a bare image"


def test_bare_image_short_unrelated_text_skips(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    calls = []
    router = Router(_stub_deepseek(calls))
    d = asyncio.run(router.decide("?", has_image=True, was_at_bot=False))
    assert d.route == "skip"
    assert calls == []


def test_bare_image_with_nickname_goes_to_llm(monkeypatch):
    """If the message names the bot, defer to the LLM — could be a vision request."""
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    calls = []
    router = Router(_stub_deepseek(calls))
    d = asyncio.run(router.decide("小笨蛋看看", has_image=True, was_at_bot=False))
    assert d.route == "deepseek_chat"  # whatever the stub returned
    assert len(calls) == 1


def test_image_with_at_bot_goes_to_llm(monkeypatch):
    """An @ on a bare image still goes to the LLM — user might want vision."""
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    calls = []
    router = Router(_stub_deepseek(calls))
    d = asyncio.run(router.decide("", has_image=True, was_at_bot=True))
    assert len(calls) == 1, "LLM should be called when @-bot, even on a bare image"


def test_no_image_pre_filter_does_not_fire(monkeypatch):
    """Short text alone (no image) should still hit the LLM."""
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    calls = []
    router = Router(_stub_deepseek(calls))
    asyncio.run(router.decide("hi", has_image=False, was_at_bot=False))
    assert len(calls) == 1
