"""Bot dreams: the prompt round-trip + the maybe_send_dream picker."""
from __future__ import annotations

import asyncio
import types

import pytest

import config as cfg
from bot import allowlist
from bot.command_handler import Handler
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from providers.base import TextReply


def _make_handler(*, captured):
    """Build a minimal Handler whose deepseek.chat is a stub recording the
    prompt + returning a fixed dream string."""
    deepseek = types.SimpleNamespace()
    deepseek.name = "stub"

    async def chat(messages, **kw):
        captured["last_messages"] = messages
        captured["last_kw"] = kw
        return TextReply(text="我刚做了个梦 群里全是会说话的猫", usage={}, model="stub")

    deepseek.chat = chat

    async def chat_stream(messages, **kw):  # not used here
        return
        yield ""  # pragma: no cover

    deepseek.chat_stream = chat_stream

    async def aclose():
        return None

    deepseek.aclose = aclose

    sent: list[tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        pass  # pragma: no cover

    handler = Handler(
        deepseek=deepseek,
        openai=None,
        router=Router(deepseek),
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
    )
    return handler, sent


def test_send_dream_posts_to_group(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "human_send_enabled", False, raising=False)
    captured: dict = {}
    handler, sent = _make_handler(captured=captured)

    async def go():
        recaps = [
            ("2026-05-21", "群里在讨论早饭吃啥，最后没结论"),
            ("2026-05-20", "有人推荐了一家拉面店"),
        ]
        await handler.send_dream(group_id=42, recaps=recaps)
        await handler.aclose()

    asyncio.run(go())
    # The bot's dream message must reach the group.
    assert sent, "dream should be sent via send_text"
    assert sent[0][0] == 42
    assert "梦" in sent[0][1]
    # The classifier prompt must include both recap days as context.
    msgs = captured["last_messages"]
    user_msg = next(m for m in msgs if m.role == "user")
    assert "拉面" in user_msg.content
    assert "早饭" in user_msg.content


def test_maybe_send_dream_picks_group_with_recaps(monkeypatch, tmp_path):
    """No recaps anywhere → no dream sent. With recaps → exactly one group
    receives a dream."""
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "human_send_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 2}, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    captured: dict = {}
    handler, sent = _make_handler(captured=captured)

    async def go():
        # No recaps yet → no dream.
        n0 = await handler.maybe_send_dream()
        assert n0 == 0
        assert sent == []
        # Seed a recap for group 1; dream should land there.
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-21", "群里聊了早饭")
        n1 = await handler.maybe_send_dream()
        assert n1 == 1
        assert len(sent) == 1
        assert sent[0][0] == 1
        await handler.aclose()

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None
