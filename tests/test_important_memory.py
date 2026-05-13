"""Tests for the important-memory layer."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta

import pytest

import config as cfg
from bot.important_memory import (
    ClassifyResult,
    ImportantMemory,
    next_recurrence,
)
from bot.storage import Storage


class _StubDeepSeek:
    """Returns a queued JSON payload from `chat`. One queued reply per call."""

    def __init__(self, replies):
        self._queue = list(replies)
        self.calls: list[dict] = []

    async def chat(self, messages, **kw):
        from providers.base import TextReply
        self.calls.append({"messages": messages, "kw": kw})
        payload = self._queue.pop(0) if self._queue else "{}"
        return TextReply(text=payload, model="stub")


@pytest.fixture(autouse=True)
def _enable_important_memory(monkeypatch):
    monkeypatch.setattr(
        cfg.CONFIG, "important_memory_enabled", True, raising=False,
    )


def test_prefilter_short_circuits_irrelevant_text():
    assert not ImportantMemory._passes_prefilter("hi")
    assert not ImportantMemory._passes_prefilter("haha 666")
    assert ImportantMemory._passes_prefilter("晚上9点叫我起床")
    assert ImportantMemory._passes_prefilter("我对花生过敏")
    assert ImportantMemory._passes_prefilter("remind me tomorrow")


def test_parse_classifier_reply_remembers_with_trigger():
    fut = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    raw = json.dumps({
        "remember": True,
        "content": "晚上9点叫张三起床",
        "subject_user_id": 42,
        "trigger_at": fut,
        "importance": 0.8,
        "tags": ["睡眠", "提醒"],
    })
    r = ImportantMemory._parse_classifier_reply(raw, default_user_id=42)
    assert isinstance(r, ClassifyResult)
    assert r.remember is True
    assert r.subject_user_id == 42
    assert r.trigger_at is not None and r.trigger_at > time.time()
    assert r.importance == pytest.approx(0.8)
    assert "睡眠" in r.tags


def test_parse_classifier_rejects_past_trigger():
    past = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    raw = json.dumps({
        "remember": True,
        "content": "x", "trigger_at": past, "importance": 0.5,
    })
    r = ImportantMemory._parse_classifier_reply(raw, default_user_id=1)
    assert r is not None and r.trigger_at is None


def test_parse_classifier_recurrence_seeds_first_trigger():
    raw = json.dumps({
        "remember": True, "content": "每天早上7点叫我",
        "recurrence": "daily 07:00", "importance": 0.7,
    })
    r = ImportantMemory._parse_classifier_reply(raw, default_user_id=1)
    assert r is not None
    assert r.recurrence == "daily 07:00"
    assert r.trigger_at is not None and r.trigger_at > time.time()


def test_parse_classifier_handles_garbage():
    assert ImportantMemory._parse_classifier_reply(
        "not json at all", default_user_id=1,
    ) is None


def test_next_recurrence_rolls_to_next_day_when_past():
    base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    # 'daily 07:00' from a time already past 7am → next day at 7am
    nxt = next_recurrence("daily 07:00", base.timestamp())
    assert nxt is not None
    nxt_dt = datetime.fromtimestamp(nxt)
    assert nxt_dt.hour == 7 and nxt_dt.minute == 0
    assert nxt_dt > base


def test_maybe_extract_skips_prefilter_miss():
    im = ImportantMemory(_StubDeepSeek([]))  # no replies queued
    inserted = asyncio.run(im.maybe_extract(
        group_id=1, user_id=42, nickname="nick", text="haha 666",
    ))
    assert inserted is None
    assert im._deepseek.calls == []  # LLM never invoked


def test_maybe_extract_saves_and_recalls(monkeypatch):
    fut = (datetime.now() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    stub = _StubDeepSeek([
        json.dumps({
            "remember": True,
            "content": "晚上9点叫张三起床",
            "subject_user_id": 42,
            "trigger_at": fut,
            "importance": 0.9,
            "tags": ["提醒"],
        }),
    ])
    im = ImportantMemory(stub)
    mem_id = asyncio.run(im.maybe_extract(
        group_id=1, user_id=42, nickname="张三",
        text="晚上9点叫我",  # triggers prefilter via 叫我 + 点
    ))
    assert mem_id is not None and mem_id > 0
    # Recall it for the same user.
    rows = asyncio.run(im.recall_for_user(1, 42))
    assert any(r[2] == "晚上9点叫张三起床" for r in rows)
    # Recall for a different user — personal memory shouldn't surface.
    other_rows = asyncio.run(im.recall_for_user(1, 99))
    assert not any(r[2] == "晚上9点叫张三起床" for r in other_rows)


def test_due_reminders_and_mark_fired():
    past = (datetime.now() - timedelta(minutes=5))  # already past
    # Insert directly via storage (bypass LLM).
    async def setup_and_run():
        store = await Storage.get()
        # We need a non-past trigger to slip past the parser's clamp, so insert
        # the row directly with trigger_at in the past.
        mid = await store.memory_item_insert(
            group_id=1, subject_user_id=42,
            content="叫张三起床", importance=0.8, tags="提醒",
            trigger_at=past.timestamp(), recurrence=None, expires_at=None,
            created_at=time.time(), source_text="晚上9点叫我",
            source_nickname="张三",
        )
        im = ImportantMemory(_StubDeepSeek([]))
        due = await im.due_reminders()
        assert any(d[0] == mid for d in due)
        await im.mark_fired(mid, recurrence=None)
        # Now should be empty.
        due2 = await im.due_reminders()
        assert not any(d[0] == mid for d in due2)

    asyncio.run(setup_and_run())


def test_recurrence_reschedules_after_firing():
    async def setup_and_run():
        store = await Storage.get()
        past = datetime.now() - timedelta(minutes=2)
        mid = await store.memory_item_insert(
            group_id=1, subject_user_id=42,
            content="每天早上7点叫张三", importance=0.7, tags="",
            trigger_at=past.timestamp(), recurrence="daily 07:00",
            expires_at=None, created_at=time.time(),
            source_text="每天7点叫我", source_nickname="张三",
        )
        im = ImportantMemory(_StubDeepSeek([]))
        await im.mark_fired(mid, recurrence="daily 07:00")
        # Should be rescheduled, not marked 'fired' — so it still shows up in
        # list_pending.
        pending = await im.list_pending(1)
        assert any(p[0] == mid for p in pending)

    asyncio.run(setup_and_run())


def test_cancel_marks_status():
    async def run():
        store = await Storage.get()
        fut = datetime.now() + timedelta(hours=1)
        mid = await store.memory_item_insert(
            group_id=1, subject_user_id=42, content="x",
            importance=0.5, tags="", trigger_at=fut.timestamp(),
            recurrence=None, expires_at=None,
            created_at=time.time(), source_text="x", source_nickname="x",
        )
        im = ImportantMemory(_StubDeepSeek([]))
        assert await im.cancel(mid, 1) is True
        # Second cancel is a no-op.
        assert await im.cancel(mid, 1) is False
        pending = await im.list_pending(1)
        assert not any(p[0] == mid for p in pending)

    asyncio.run(run())
