"""`/admin reset confirm` — full per-group state wipe + `/clear` per-user shortcut."""
from __future__ import annotations

import asyncio
import time
import types
from typing import List, Tuple

import config as cfg
from bot.command_handler import Handler
from bot.lessons import Lessons
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.storage import Storage


def _event(text: str, *, user_id: int = 42, group_id: int = 1) -> dict:
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": [
            {"type": "at", "data": {"qq": "10000"}},
            {"type": "text", "data": {"text": text}},
        ],
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
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

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 2}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        lessons=Lessons(stub),  # type: ignore[arg-type]
    ), sent


# ---------- /reset is no longer a top-level command ----------
def test_top_level_reset_is_gone(monkeypatch):
    """`/reset` (top-level) should no longer dispatch to the wipe handler.
    Falls through to the unknown-command response."""
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/reset confirm"))))
    asyncio.run(handler.aclose())
    assert sent
    body = sent[0][1]
    assert "未知指令" in body or "/help" in body
    assert "已重置" not in body


# ---------- /admin reset (no confirm) ----------
def test_admin_reset_without_confirm_shows_warning(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/admin reset"))))
    asyncio.run(handler.aclose())
    assert sent
    body = sent[0][1]
    assert "confirm" in body or "确认" in body
    assert "/admin reset" in body
    assert "/clear" in body


def test_admin_reset_rejects_non_superuser(monkeypatch):
    """The /admin gate already runs; a non-superuser doesn't even reach
    `/admin reset`."""
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(
        _event("/admin reset confirm", user_id=9999),
    )))
    asyncio.run(handler.aclose())
    body = sent[0][1]
    assert "超级用户" in body


# ---------- happy path ----------
def test_admin_reset_confirm_wipes_every_per_group_table(monkeypatch):
    """Plant memory + group_memory + daily_recap + lessons rows for group 1
    AND group 2, run /admin reset confirm in group 1, verify group 1 is
    empty and group 2 is untouched."""
    handler, sent = _make_handler(monkeypatch)

    async def go():
        store = await Storage.get()
        # Per-user conversation memory in group 1 (two users) and group 2.
        await handler.memory.append(1, 100, "user", "hi from user 100 in g1")
        await handler.memory.append(1, 100, "assistant", "ack 100")
        await handler.memory.append(1, 200, "user", "hi from user 200 in g1")
        await handler.memory.append(2, 100, "user", "hi from user 100 in g2")
        # Rolling group chat log.
        await handler.group_memory.append(1, 100, "u100", "g1 message")
        await handler.group_memory.append(1, 200, "u200", "g1 another")
        await handler.group_memory.append(2, 100, "u100", "g2 message")
        # Daily recap (long memory).
        await store.daily_recap_upsert(1, "2026-05-01", "g1 recap A")
        await store.daily_recap_upsert(1, "2026-05-02", "g1 recap B")
        await store.daily_recap_upsert(2, "2026-05-01", "g2 recap")
        # Lessons (mix of teach + auto in g1; one in g2).
        await handler.lessons.teach_raw(group_id=1, user_id=42, text="g1 rule")
        await store.lesson_insert(
            group_id=1, kind="fact", subject_user_id=100,
            content="g1 fact", importance=0.6, tags="", trigger_at=None,
            recurrence=None, expires_at=None, source_user_id=100,
            source_text="x", created_at=time.time(),
        )
        await handler.lessons.teach_raw(group_id=2, user_id=42, text="g2 rule")

        # Sanity: everything is in place pre-reset.
        assert await handler.memory.get(1, 100)
        assert await handler.group_memory.recent(1)
        assert await store.daily_recap_get(1, "2026-05-01") == "g1 recap A"
        assert await handler.lessons.active_for_user(1, 42, limit=10)

        # Fire /admin reset confirm in group 1.
        await handler.handle(parse_event(
            _event("/admin reset confirm", group_id=1),
        ))
        await handler.aclose()

        # Post-reset: group 1 is empty across every table.
        results = {
            "memory_g1_u100": await handler.memory.get(1, 100),
            "memory_g1_u200": await handler.memory.get(1, 200),
            "group_memory_g1": await handler.group_memory.recent(1),
            "recap_g1_a": await store.daily_recap_get(1, "2026-05-01"),
            "recap_g1_b": await store.daily_recap_get(1, "2026-05-02"),
            "lessons_g1": await handler.lessons.active_for_user(1, 42, limit=10),
        }
        # Group 2 survives untouched.
        survivors = {
            "memory_g2_u100": await handler.memory.get(2, 100),
            "group_memory_g2": await handler.group_memory.recent(2),
            "recap_g2": await store.daily_recap_get(2, "2026-05-01"),
            "lessons_g2": await handler.lessons.active_for_user(2, 42, limit=10),
        }
        return results, survivors

    results, survivors = asyncio.run(go())

    assert results["memory_g1_u100"] == []
    assert results["memory_g1_u200"] == []
    assert results["group_memory_g1"] == []
    assert results["recap_g1_a"] is None
    assert results["recap_g1_b"] is None
    assert results["lessons_g1"] == []

    assert survivors["memory_g2_u100"], "group 2 conversation memory wiped"
    assert survivors["group_memory_g2"], "group 2 chat log wiped"
    assert survivors["recap_g2"] == "g2 recap"
    assert any(a.content == "g2 rule" for a in survivors["lessons_g2"])

    # The bot replies with a counts summary.
    body = sent[-1][1]
    assert "已重置本群全部记录" in body
    for label in ("对话记忆", "群聊日志", "长时记忆", "lessons"):
        assert label in body


def test_admin_reset_accepts_chinese_confirm(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.lessons.teach_raw(group_id=1, user_id=42, text="rule")
        await handler.handle(parse_event(_event("/admin reset 确认")))
        await handler.aclose()

    asyncio.run(go())
    assert sent and "已重置本群全部记录" in sent[-1][1]


def test_admin_reset_preserves_allowlist_and_pause(monkeypatch):
    """/admin reset is for state, not control. Allow-list + pause state must
    survive a wipe so the bot stays usable in this group."""
    handler, _ = _make_handler(monkeypatch)
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", set(), raising=False)

    async def go():
        from bot import allowlist
        await allowlist.add(1)
        await allowlist.pause(1, by_user_id=42)
        await handler.handle(parse_event(_event("/admin reset confirm")))
        await handler.aclose()
        return (
            await allowlist.is_allowed(1),
            await allowlist.is_paused(1),
        )

    allowed, paused = asyncio.run(go())
    assert allowed, "allow-list must survive /admin reset"
    assert paused, "pause state must survive /admin reset"


# ---------- /admin clear ----------
def test_admin_clear_by_user_id(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.memory.append(1, 100, "user", "100 said hi")
        await handler.memory.append(1, 100, "assistant", "ack")
        await handler.memory.append(1, 200, "user", "200 said hi")
        await handler.handle(parse_event(_event("/admin clear 100")))
        await handler.aclose()
        return (
            await handler.memory.get(1, 100),
            await handler.memory.get(1, 200),
        )

    target, untouched = asyncio.run(go())
    assert target == [], "targeted user's memory must be wiped"
    assert untouched, "other users' memory must not be touched"
    assert sent and "100" in sent[-1][1] and "已清空" in sent[-1][1]


def test_admin_clear_by_at_mention(monkeypatch):
    """`/admin clear @user` picks up the at-target that isn't the bot."""
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.memory.append(1, 555, "user", "555 chats")
        # Build a message with TWO at-targets: the bot (10000) and user 555.
        event = {
            "post_type": "message",
            "message_type": "group",
            "self_id": 10000,
            "group_id": 1,
            "user_id": 42,
            "raw_message": "/admin clear",
            "message": [
                {"type": "at", "data": {"qq": "10000"}},   # @bot
                {"type": "text", "data": {"text": "/admin clear "}},
                {"type": "at", "data": {"qq": "555"}},     # @target
            ],
            "sender": {"user_id": 42, "nickname": "x"},
        }
        await handler.handle(parse_event(event))
        await handler.aclose()
        return await handler.memory.get(1, 555)

    leftover = asyncio.run(go())
    assert leftover == []
    assert sent and "555" in sent[-1][1]


def test_admin_clear_without_target_shows_usage(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/admin clear"))))
    asyncio.run(handler.aclose())
    assert sent and "用法" in sent[-1][1]


def test_admin_clear_refuses_to_target_bot_itself(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/admin clear 10000"))))
    asyncio.run(handler.aclose())
    assert sent and "我自己" in sent[-1][1]


def test_admin_clear_rejects_non_superuser(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(
        _event("/admin clear 100", user_id=9999),
    )))
    asyncio.run(handler.aclose())
    assert sent and "超级用户" in sent[0][1]


# ---------- /clear (self-wipe, unchanged behavior) ----------
def test_clear_wipes_only_caller_memory(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.memory.append(1, 42, "user", "my msg")
        await handler.memory.append(1, 42, "assistant", "ack")
        await handler.memory.append(1, 100, "user", "other's msg")
        await handler.handle(parse_event(_event("/clear")))
        await handler.aclose()
        return (
            await handler.memory.get(1, 42),
            await handler.memory.get(1, 100),
        )

    mine, other = asyncio.run(go())
    assert mine == []
    assert other, "other users' memory must not be touched by /clear"
    assert sent and "已清空你的对话记忆" in sent[-1][1]


# ---------- help text ----------
def test_help_mentions_clear_and_admin_paths(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/help"))))
    asyncio.run(handler.aclose())
    body = sent[-1][1]
    assert "/clear" in body
    assert "/admin clear" in body
    assert "/admin reset" in body


def test_admin_help_lists_new_subcommands(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/admin help"))))
    asyncio.run(handler.aclose())
    body = sent[-1][1]
    assert "/admin clear" in body
    assert "/admin reset" in body
    # The old reset_memory subcommand is gone.
    assert "reset_memory" not in body
