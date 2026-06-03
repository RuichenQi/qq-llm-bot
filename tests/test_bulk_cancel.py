"""Bulk-cancel / hard-delete behavior for /remember, /forget, /admin.

Covers:
  - parsing multiple ids (space, comma, mixed)
  - special tokens: `all`, kind aliases (`rules`/`facts`/Chinese)
  - cancel = hard delete (no residue in any read path)
  - cancel is per-group (no cross-group leak)
"""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import config as cfg
from bot.command_handler import Handler
from bot.lessons import Lessons
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


# ---------- argument parser ----------
def test_parse_single_id():
    assert Handler._parse_cancel_targets("7") == ([7], None, False)


def test_parse_space_separated_ids():
    assert Handler._parse_cancel_targets("5 7 9") == ([5, 7, 9], None, False)


def test_parse_comma_separated_ids():
    assert Handler._parse_cancel_targets("5,7,9") == ([5, 7, 9], None, False)


def test_parse_mixed_separators():
    assert Handler._parse_cancel_targets("5, 7  9,11") == ([5, 7, 9, 11], None, False)


def test_parse_all_token():
    assert Handler._parse_cancel_targets("all") == ([], None, True)
    assert Handler._parse_cancel_targets("ALL") == ([], None, True)
    assert Handler._parse_cancel_targets("全部") == ([], None, True)
    assert Handler._parse_cancel_targets("所有") == ([], None, True)


def test_parse_kind_aliases():
    assert Handler._parse_cancel_targets("rules") == ([], "rule", False)
    assert Handler._parse_cancel_targets("规则") == ([], "rule", False)
    assert Handler._parse_cancel_targets("facts") == ([], "fact", False)
    assert Handler._parse_cancel_targets("agreements") == ([], "agreement", False)
    assert Handler._parse_cancel_targets("提醒") == ([], "reminder", False)


def test_parse_garbage_returns_empty():
    assert Handler._parse_cancel_targets("") == ([], None, False)
    assert Handler._parse_cancel_targets("hello world") == ([], None, False)
    # Garbled but contains some digits — we pick those out.
    assert Handler._parse_cancel_targets("foo 3 bar 7") == ([3, 7], None, False)


# ---------- end-to-end via the command handler ----------
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


def test_forget_multiple_ids_in_one_command(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        ids = []
        for i, text in enumerate(("规则一", "规则二", "规则三")):
            row_id = await handler.lessons.teach_raw(
                group_id=1, user_id=42, text=text,
            )
            assert row_id > 0
            ids.append(row_id)
        await handler.handle(parse_event(_event(
            f"/forget {ids[0]} {ids[2]}"  # cancel first + third
        )))
        await handler.aclose()
        return ids

    ids = asyncio.run(ids_run := go())

    assert sent and "已忘记 2 条" in sent[-1][1]

    # Confirm via storage: only the middle rule remains.
    async def check():
        active = await handler.lessons.active_for_user(1, 42, limit=10)
        return {a.id: a.content for a in active}

    remaining = asyncio.run(check())
    assert ids[0] not in remaining
    assert ids[1] in remaining
    assert ids[2] not in remaining


def test_forget_all_wipes_group(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        for text in ("一", "二", "三", "四"):
            await handler.lessons.teach_raw(
                group_id=1, user_id=42, text=text,
            )
        await handler.handle(parse_event(_event("/forget all")))
        await handler.aclose()

    asyncio.run(go())
    assert sent and "已忘记本群所有 4 条" in sent[-1][1]

    async def check():
        return await handler.lessons.active_for_user(1, 42, limit=10)

    assert asyncio.run(check()) == []


def test_forget_rules_keeps_other_kinds(monkeypatch):
    """`/forget rules` removes only rule rows, leaves fact/agreement/reminder
    rows alone."""
    handler, sent = _make_handler(monkeypatch)

    async def go():
        # Plant one of each kind directly via storage so we don't have to
        # mock the classifier.
        import time as _time
        from bot.storage import Storage
        store = await Storage.get()
        for kind, content in (
            ("rule", "你说话简短"),
            ("fact", "Alice 是程序员"),
            ("agreement", "周五开会"),
        ):
            await store.lesson_insert(
                group_id=1, kind=kind, subject_user_id=None,
                content=content, importance=0.7, tags="",
                trigger_at=None, recurrence=None, expires_at=None,
                source_user_id=42, source_text=content,
                created_at=_time.time(),
            )
        await handler.handle(parse_event(_event("/forget rules")))
        await handler.aclose()

    asyncio.run(go())
    assert sent and "已忘记本群所有 1 条规则" in sent[-1][1]

    async def check():
        return await handler.lessons.active_for_user(1, 42, limit=20)

    remaining = asyncio.run(check())
    kinds = {a.kind for a in remaining}
    assert "rule" not in kinds
    assert kinds == {"fact", "agreement"}


def test_forget_is_per_group(monkeypatch):
    """`/forget all` in group 1 must not touch group 2's rules."""
    handler, _ = _make_handler(monkeypatch)

    async def go():
        await handler.lessons.teach_raw(group_id=1, user_id=42, text="A")
        await handler.lessons.teach_raw(group_id=2, user_id=42, text="B")
        await handler.handle(parse_event(_event("/forget all", group_id=1)))
        await handler.aclose()
        a = await handler.lessons.active_for_user(1, 42, limit=10)
        b = await handler.lessons.active_for_user(2, 42, limit=10)
        return a, b

    a, b = asyncio.run(go())
    assert a == [], "group 1 should be empty"
    assert any(x.content == "B" for x in b), "group 2 must be untouched"


def test_cancel_is_hard_delete_no_residue(monkeypatch):
    """After cancel, the row must be physically gone — not just status-flipped.

    Verifies that ALL read paths (active_for_user, list_pending, list_all admin
    view, dedup_candidates) stop returning the cancelled row.
    """
    handler, _ = _make_handler(monkeypatch)

    async def go():
        row_id = await handler.lessons.teach_raw(
            group_id=1, user_id=42, text="测试规则",
        )
        assert row_id > 0
        # Pre-cancel: every read path sees the row.
        active = await handler.lessons.active_for_user(1, 42)
        assert any(a.id == row_id for a in active)
        admin_list = await handler.lessons.list_all(1, limit=50)
        assert any(r[0] == row_id for r in admin_list)
        # Now cancel.
        ok = await handler.lessons.cancel(row_id, 1)
        assert ok is True
        # Post-cancel: every read path no longer sees the row.
        active2 = await handler.lessons.active_for_user(1, 42)
        assert not any(a.id == row_id for a in active2)
        admin_list2 = await handler.lessons.list_all(1, limit=50)
        assert not any(r[0] == row_id for r in admin_list2)
        # Direct DB check — row is GONE, not just hidden by a status filter.
        from bot.storage import Storage
        store = await Storage.get()
        async with store._conn.execute(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM lessons WHERE id=?", (row_id,),
        ) as cur:
            (count,) = await cur.fetchone()
        assert count == 0, "row must be physically deleted"
        # Cancelling the same id again returns False — there's nothing there.
        again = await handler.lessons.cancel(row_id, 1)
        assert again is False

    asyncio.run(go())


def test_remember_cancel_accepts_same_syntax_as_forget(monkeypatch):
    handler, sent = _make_handler(monkeypatch)

    async def go():
        ids = []
        for text in ("X", "Y", "Z"):
            ids.append(await handler.lessons.teach_raw(
                group_id=1, user_id=42, text=text,
            ))
        await handler.handle(parse_event(_event(
            f"/remember cancel {ids[0]},{ids[1]}"
        )))
        await handler.aclose()

    asyncio.run(go())
    assert sent and "已忘记 2 条" in sent[-1][1]


def test_forget_unknown_ids_report_friendly(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/forget 9999"))))
    asyncio.run(handler.aclose())
    assert sent and ("没找到" in sent[-1][1] or "已经被忘了" in sent[-1][1])
