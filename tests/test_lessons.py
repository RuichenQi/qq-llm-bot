"""Unified 功能注入 (Lessons) module: classifier parsing, storage round-trip,
prompt formatting, and reminder firing flow."""
from __future__ import annotations

import asyncio
import json
import time
import types
from datetime import datetime, timedelta

import pytest

from bot.lessons import ActiveLesson, ClassifyResult, Lessons, next_recurrence


# ---------- classifier parsing ----------
def test_parse_rule_with_importance():
    raw = json.dumps({
        "kind": "rule", "content": "你说话简短点",
        "subject_user_id": None, "importance": 0.8,
    })
    r = Lessons._parse_classifier_reply(raw, default_user_id=42)
    assert isinstance(r, ClassifyResult)
    assert r.kind == "rule"
    assert r.content == "你说话简短点"
    assert 0.79 < r.importance < 0.81


def test_parse_fact_keeps_subject():
    raw = json.dumps({
        "kind": "fact", "content": "对花生过敏",
        "subject_user_id": 42, "importance": 0.7,
    })
    r = Lessons._parse_classifier_reply(raw, default_user_id=42)
    assert r is not None and r.kind == "fact"
    assert r.subject_user_id == 42


def test_parse_invalid_subject_falls_back_for_fact():
    """Bad subject value → for kind=fact, fall back to speaker; otherwise None."""
    raw = json.dumps({
        "kind": "fact", "content": "对花生过敏",
        "subject_user_id": "not-an-int", "importance": 0.7,
    })
    r = Lessons._parse_classifier_reply(raw, default_user_id=42)
    assert r is not None and r.subject_user_id == 42


def test_parse_reminder_future_trigger_kept():
    fut = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    raw = json.dumps({
        "kind": "reminder", "content": "晚上9点叫张三",
        "subject_user_id": 42, "trigger_at": fut, "importance": 0.8,
    })
    r = Lessons._parse_classifier_reply(raw, default_user_id=42)
    assert r is not None
    assert r.kind == "reminder"
    assert r.trigger_at is not None and r.trigger_at > time.time()


def test_parse_reminder_past_trigger_dropped():
    past = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    raw = json.dumps({
        "kind": "reminder", "content": "x", "trigger_at": past,
    })
    r = Lessons._parse_classifier_reply(raw, default_user_id=1)
    assert r is not None and r.trigger_at is None


def test_parse_none_returns_kind_none():
    r = Lessons._parse_classifier_reply('{"kind": "none"}', default_user_id=1)
    assert r is not None and r.kind == "none"


def test_parse_malformed_returns_none():
    assert Lessons._parse_classifier_reply("not json", default_user_id=1) is None


def test_next_recurrence_rolls_forward():
    base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    nxt = next_recurrence("daily 07:00", base.timestamp())
    assert nxt is not None
    nxt_dt = datetime.fromtimestamp(nxt)
    assert nxt_dt.hour == 7 and nxt_dt.minute == 0
    assert nxt_dt > base


# ---------- prefilter ----------
def test_prefilter_blocks_short_text():
    assert not Lessons._passes_prefilter("hi", force=False)


def test_prefilter_blocks_irrelevant():
    assert not Lessons._passes_prefilter("haha 666", force=False)


def test_prefilter_allows_temporal_words():
    assert Lessons._passes_prefilter("晚上9点叫我", force=False)
    assert Lessons._passes_prefilter("我对花生过敏", force=False)


def test_prefilter_force_bypasses_keyword_gate():
    # @-mentioned messages should always reach the classifier even without
    # any of the keyword cues.
    assert Lessons._passes_prefilter("你说话简短点", force=True)


# ---------- format_for_prompt ----------
def test_format_for_prompt_empty():
    assert Lessons.format_for_prompt([], speaker_user_id=1) is None


def test_format_for_prompt_groups_by_kind():
    rows = [
        ActiveLesson(1, "rule", None, "你说话简短点", 0.8, "", None),
        ActiveLesson(2, "fact", 42, "对花生过敏", 0.7, "", None),
        ActiveLesson(3, "agreement", None, "周五开会", 0.6, "", None),
    ]
    block = Lessons.format_for_prompt(rows, speaker_user_id=42)
    assert block is not None
    assert "你说话简短点" in block
    assert "对花生过敏" in block
    assert "周五开会" in block
    # Speaker-personal facts use "你（说话人）" label.
    assert "（说话人）" in block


# ---------- round-trip via storage ----------
def test_maybe_learn_persists_rule(monkeypatch, tmp_path):
    import config as cfg
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    stub = types.SimpleNamespace()

    async def chat(messages, **kw):
        from providers.base import TextReply
        payload = json.dumps({
            "kind": "rule", "content": "你说话简短点",
            "subject_user_id": None, "importance": 0.8,
        })
        return TextReply(text=payload, usage={}, model="stub")

    stub.chat = chat
    lessons = Lessons(stub)  # type: ignore[arg-type]

    async def go():
        # addressed=True so the prefilter doesn't gate (text has no keyword).
        row_id = await lessons.maybe_learn(
            group_id=42, user_id=99, text="你说话简短点", addressed=True,
        )
        assert row_id is not None and row_id > 0
        # Recall via the prompt-injection API.
        active = await lessons.active_for_user(42, 99, limit=10)
        assert any(a.content == "你说话简短点" and a.kind == "rule" for a in active)
        # Cancel and confirm gone.
        assert await lessons.cancel(row_id, 42) is True
        active2 = await lessons.active_for_user(42, 99, limit=10)
        assert not any(a.id == row_id for a in active2)

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None


def test_teach_raw_is_per_group_and_skips_classifier(monkeypatch, tmp_path):
    """teach_raw stores literal text scoped to one group, with no LLM call.

    Per-group prompt injection: a rule taught in group A must not leak into
    group B's chat-time recall."""
    import config as cfg
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    classifier_calls = []
    stub = types.SimpleNamespace()

    async def chat(messages, **kw):
        classifier_calls.append(messages)
        from providers.base import TextReply
        return TextReply(text="{}", usage={}, model="stub")

    stub.chat = chat
    lessons = Lessons(stub)  # type: ignore[arg-type]

    async def go():
        # Two different groups; each teaches its own rule.
        a_id = await lessons.teach_raw(
            group_id=1, user_id=42,
            text="本群讨论 ML 论文，回复时多引用 paper",
        )
        b_id = await lessons.teach_raw(
            group_id=2, user_id=99, text="本群是英文群，请用英文回复",
        )
        assert a_id > 0 and b_id > 0
        # Classifier was never called.
        assert classifier_calls == []
        # Group 1 sees A's rule but NOT B's.
        active_a = await lessons.active_for_user(1, 42, limit=10)
        contents_a = {a.content for a in active_a}
        assert "本群讨论 ML 论文，回复时多引用 paper" in contents_a
        assert "本群是英文群，请用英文回复" not in contents_a
        # Group 2 sees B's rule but NOT A's.
        active_b = await lessons.active_for_user(2, 99, limit=10)
        contents_b = {a.content for a in active_b}
        assert "本群是英文群，请用英文回复" in contents_b
        assert "本群讨论 ML 论文，回复时多引用 paper" not in contents_b
        # Both rows are kind=rule, subject=None (group-wide).
        assert all(a.kind == "rule" for a in active_a + active_b)
        assert all(a.subject_user_id is None for a in active_a + active_b)

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None


def test_teach_raw_rejects_empty_and_oversize(monkeypatch, tmp_path):
    import config as cfg
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    stub = types.SimpleNamespace()

    async def chat(messages, **kw):
        from providers.base import TextReply
        return TextReply(text="{}", usage={}, model="stub")

    stub.chat = chat
    lessons = Lessons(stub)  # type: ignore[arg-type]

    async def go():
        assert await lessons.teach_raw(group_id=1, user_id=1, text="") == 0
        assert await lessons.teach_raw(
            group_id=1, user_id=1, text="x" * 600,
        ) == 0
        # Borderline: exactly 500 chars is accepted.
        ok_id = await lessons.teach_raw(
            group_id=1, user_id=1, text="x" * 500,
        )
        assert ok_id > 0

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None


def test_maybe_learn_skips_when_none(monkeypatch, tmp_path):
    import config as cfg
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    stub = types.SimpleNamespace()

    async def chat(messages, **kw):
        from providers.base import TextReply
        return TextReply(text='{"kind": "none"}', usage={}, model="stub")

    stub.chat = chat
    lessons = Lessons(stub)  # type: ignore[arg-type]

    async def go():
        row_id = await lessons.maybe_learn(
            group_id=1, user_id=2, text="今天天气真好", addressed=True,
        )
        assert row_id is None
        active = await lessons.active_for_user(1, 2)
        assert active == []

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None


def test_due_reminders_and_mark_fired(monkeypatch, tmp_path):
    """Insert a past-trigger row directly via storage; ensure due_reminders
    picks it up and mark_fired flips status."""
    import config as cfg
    from bot import storage as storage_mod
    from bot.storage import Storage

    db = tmp_path / "state.db"
    monkeypatch.setattr(cfg, "DB_FILE", db, raising=False)
    monkeypatch.setattr(storage_mod, "DB_FILE", db, raising=False)
    Storage._instance = None
    Storage._init_lock = None

    stub = types.SimpleNamespace()

    async def chat(messages, **kw):
        from providers.base import TextReply
        return TextReply(text="{}", usage={}, model="stub")

    stub.chat = chat
    lessons = Lessons(stub)  # type: ignore[arg-type]

    past = (datetime.now() - timedelta(minutes=5)).timestamp()

    async def go():
        store = await Storage.get()
        mid = await store.lesson_insert(
            group_id=1, kind="reminder", subject_user_id=42,
            content="叫张三起床", importance=0.8, tags="提醒",
            trigger_at=past, recurrence=None, expires_at=None,
            source_user_id=42, source_text="晚上9点叫我",
            created_at=time.time(),
        )
        due = await lessons.due_reminders()
        assert any(d[0] == mid for d in due)
        await lessons.mark_fired(mid, recurrence=None)
        due2 = await lessons.due_reminders()
        assert not any(d[0] == mid for d in due2)

    asyncio.run(go())
    Storage._instance = None
    Storage._init_lock = None
