"""Long-term memory: daily recap save + retrieve + /recall."""
from __future__ import annotations

import asyncio
import time
import types
from datetime import datetime, timedelta
from typing import List, Tuple

import pytest

import config as cfg
from bot.command_handler import Handler
from bot.group_memory import GroupMemory
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.storage import Storage


def _event(text: str, *, user_id: int = 42, group_id: int = 1, nickname: str = "Alice"):
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


def _make_handler(monkeypatch, *, deepseek_reply: str = "今天群里聊了吃的"):
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    deepseek = types.SimpleNamespace(name="deepseek")
    captured: list[list] = []

    async def chat(messages, **kw):
        from providers.base import TextReply
        captured.append(list(messages))
        return TextReply(text=deepseek_reply, model="stub", usage={})

    deepseek.chat = chat

    async def aclose():
        return None

    deepseek.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "chat", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "auto_vision_group_images", False, raising=False)

    handler = Handler(
        deepseek=deepseek,
        openai=None,
        router=router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
        group_memory=GroupMemory(),
    )
    return handler, sent, captured


# ---------- LongMemory direct ----------
def test_save_day_writes_summary(monkeypatch):
    handler, _, captured = _make_handler(monkeypatch, deepseek_reply="今天大家聊了麻辣烫")

    async def run():
        # Seed group memory at a specific yesterday timestamp.
        yest = (datetime.now() - timedelta(days=1)).replace(hour=12, minute=0, second=0)
        store = await Storage.get()
        for n, (uid, nick, txt) in enumerate([
            (100, "Alice", "晚上吃啥"),
            (101, "Bob", "麻辣烫吧"),
            (102, "Carol", "我想吃日料"),
        ]):
            await store.group_memory_append(
                1, yest.timestamp() + n, uid, nick, txt, 500,
            )
        day = yest.strftime("%Y-%m-%d")
        summary = await handler.long_memory.save_day(1, day)
        assert summary == "今天大家聊了麻辣烫"
        # Verify persisted
        got = await handler.long_memory.get(1, day)
        assert got == "今天大家聊了麻辣烫"

    asyncio.run(run())
    asyncio.run(handler.aclose())


def test_save_day_skipped_with_no_messages(monkeypatch):
    handler, _, _ = _make_handler(monkeypatch)

    async def run():
        out = await handler.long_memory.save_day(1, "2026-01-01")
        assert out is None

    asyncio.run(run())
    asyncio.run(handler.aclose())


def test_recent_returns_most_recent_first(monkeypatch):
    handler, _, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-01", "聊了 A")
        await store.daily_recap_upsert(1, "2026-05-02", "聊了 B")
        await store.daily_recap_upsert(1, "2026-05-03", "聊了 C")
        rows = await handler.long_memory.recent(1, days=3)
        assert [d for d, _ in rows] == ["2026-05-03", "2026-05-02", "2026-05-01"]

    asyncio.run(run())
    asyncio.run(handler.aclose())


def test_search_filters_by_keyword(monkeypatch):
    handler, _, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-01", "聊了麻辣烫和日料")
        await store.daily_recap_upsert(1, "2026-05-02", "聊了电影")
        rows = await handler.long_memory.search(1, "麻辣")
        assert len(rows) == 1
        assert rows[0][0] == "2026-05-01"

    asyncio.run(run())
    asyncio.run(handler.aclose())


# ---------- Injection into chat ----------
def test_long_memory_injected_into_chat(monkeypatch):
    handler, _, captured = _make_handler(monkeypatch)
    monkeypatch.setattr(cfg.CONFIG, "long_memory_inject_days", 3, raising=False)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-09", "聊了一些游戏")
        await store.daily_recap_upsert(1, "2026-05-10", "约了周末出去")
        # Trigger a chat
        await handler.handle(parse_event(_event("hi", group_id=1)))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert captured, "chat() should have been called"
    sys_content = "\n".join(m.content for m in captured[-1] if m.role == "system")
    assert "长时记忆" in sys_content
    assert "聊了一些游戏" in sys_content
    assert "约了周末出去" in sys_content


def test_long_memory_zero_days_skips_injection(monkeypatch):
    handler, _, captured = _make_handler(monkeypatch)
    monkeypatch.setattr(cfg.CONFIG, "long_memory_inject_days", 0, raising=False)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-10", "X")
        await handler.handle(parse_event(_event("hi")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    sys_content = "\n".join(m.content for m in captured[-1] if m.role == "system")
    assert "长时记忆" not in sys_content


# ---------- /recall ----------
def test_recall_lists_recent(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-09", "聊了 A")
        await store.daily_recap_upsert(1, "2026-05-10", "聊了 B")
        await handler.handle(parse_event(_event("/recall")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    flat = "\n".join(t for _, t in sent)
    assert "2026-05-09" in flat or "2026-05-10" in flat
    assert "聊了" in flat


def test_recall_by_date(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-04-01", "april fools")
        await handler.handle(parse_event(_event("/recall 2026-04-01")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any("april fools" in t for _, t in sent)


def test_recall_by_keyword(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        await store.daily_recap_upsert(1, "2026-05-01", "聊了麻辣烫")
        await store.daily_recap_upsert(1, "2026-05-02", "聊了电影")
        await handler.handle(parse_event(_event("/recall 麻辣")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any("麻辣烫" in t for _, t in sent)
    assert not any("电影" in t for _, t in sent)


def test_recall_empty_state(monkeypatch):
    handler, sent, _ = _make_handler(monkeypatch)

    async def run():
        await handler.handle(parse_event(_event("/recall")))

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert any("没攒下" in t for _, t in sent)


# ---------- prune ----------
def test_prune_drops_old_recaps(monkeypatch):
    handler, _, _ = _make_handler(monkeypatch)

    async def run():
        store = await Storage.get()
        # Insert one very old day, one fresh
        await store.daily_recap_upsert(1, "2020-01-01", "old")
        await store.daily_recap_upsert(1, datetime.now().strftime("%Y-%m-%d"), "new")
        # Prune anything older than 30 days
        monkeypatch.setattr(cfg.CONFIG, "daily_recap_keep_days", 30, raising=False)
        deleted = await handler.long_memory.prune()
        assert deleted >= 1
        recent = await handler.long_memory.recent(1, days=10)
        assert all(day != "2020-01-01" for day, _ in recent)

    asyncio.run(run())
    asyncio.run(handler.aclose())
