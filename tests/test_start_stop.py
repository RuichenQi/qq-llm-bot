"""Per-group /start and /stop pause toggle."""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import config as cfg
from bot import allowlist
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, group_id: int = 1, at_bot: bool = True):
    segs = []
    if at_bot or text.startswith("/"):
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

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
    ), sent


# ---------- happy path ----------
def test_stop_silences_subsequent_messages(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        await handler.handle(parse_event(_event("hello", at_bot=False)))
        await handler.handle(parse_event(_event("hi again", at_bot=False)))
        await handler.aclose()

    asyncio.run(go())
    # Exactly one ack from /stop; the two follow-ups are silent.
    assert len(sent) == 1
    assert "安静" in sent[0][1] or "下线" in sent[0][1] or "好" in sent[0][1]


def test_start_resumes_replies(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        await handler.handle(parse_event(_event("hello", at_bot=False)))
        sent.clear()
        await handler.handle(parse_event(_event("/start")))
        await handler.handle(parse_event(_event("hi again", at_bot=False)))
        await handler.aclose()

    asyncio.run(go())
    # /start acknowledged + chat now flows again.
    assert len(sent) == 2
    assert any("回来" in t for _, t in sent)
    assert sent[-1][1] == "ok"


# ---------- permissions ----------
def test_stop_rejects_non_superuser(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/stop", user_id=9999))))
    asyncio.run(handler.aclose())
    assert sent and "超级用户" in sent[0][1]


def test_start_rejects_non_superuser_when_unpaused(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/start", user_id=9999))))
    asyncio.run(handler.aclose())
    assert sent and "超级用户" in sent[0][1]


def test_paused_blocks_non_superuser_start(monkeypatch):
    """When paused, a non-superuser /start gets silently ignored (the gate
    fires before _run_start can refuse them)."""
    handler, sent = _make_handler(monkeypatch)

    async def go():
        # Superuser pauses.
        await handler.handle(parse_event(_event("/stop")))
        sent.clear()
        # Non-superuser tries to start — silent skip.
        await handler.handle(parse_event(_event("/start", user_id=9999)))
        await handler.aclose()

    asyncio.run(go())
    assert sent == []


# ---------- idempotency ----------
def test_stop_twice_is_noop(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        sent.clear()
        await handler.handle(parse_event(_event("/stop")))
        await handler.aclose()

    asyncio.run(go())
    assert len(sent) == 1
    assert "已经" in sent[0][1] or "本来" in sent[0][1] or "休息" in sent[0][1]


def test_start_when_already_running_is_noop(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/start"))))
    asyncio.run(handler.aclose())
    assert sent and ("本来" in sent[0][1] or "在群里" in sent[0][1])


# ---------- escape hatches ----------
def test_admin_still_works_while_paused(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        sent.clear()
        await handler.handle(parse_event(_event("/admin list_groups")))
        await handler.aclose()

    asyncio.run(go())
    assert sent, "admin commands must work while paused"
    assert "允许的群" in sent[0][1]


def test_group_memory_still_records_while_paused(monkeypatch):
    """Even when silenced, group_memory keeps growing so /start has context."""
    handler, _ = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        await handler.handle(parse_event(_event("something happened", at_bot=False)))
        await handler.handle(parse_event(_event("then something else", at_bot=False)))
        await handler.aclose()
        rows = await handler.group_memory.recent(1, limit=10)
        return [r.text for r in rows]

    texts = asyncio.run(go())
    # The user's two muted messages should still be on file.
    assert any("something happened" in t for t in texts)
    assert any("then something else" in t for t in texts)


def test_list_groups_marks_paused(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/stop")))
        sent.clear()
        await handler.handle(parse_event(_event("/admin list_groups")))
        await handler.aclose()

    asyncio.run(go())
    body = sent[0][1]
    assert "⏸" in body
