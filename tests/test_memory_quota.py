"""SQLite-backed memory + quota tests."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import config as cfg
from bot import memory as memory_mod
from bot import quota as quota_mod


def test_memory_append_get_reset():
    async def run():
        m = memory_mod.Memory(max_turns=3)
        await m.append(1, 99, "user", "hi")
        await m.append(1, 99, "assistant", "hello")
        got = await m.get(1, 99)
        assert got == [("user", "hi"), ("assistant", "hello")]
        await m.reset(1, 99)
        assert await m.get(1, 99) == []
    asyncio.run(run())


def test_memory_trim_to_max_turns():
    async def run():
        m = memory_mod.Memory(max_turns=2)  # keep last 4 rows
        for i in range(10):
            await m.append(1, 99, "user", f"u{i}")
            await m.append(1, 99, "assistant", f"a{i}")
        rows = await m.get(1, 99)
        assert len(rows) == 4
        assert rows[0] == ("user", "u8")
        assert rows[-1] == ("assistant", "a9")
    asyncio.run(run())


def test_memory_admin_reset_group():
    async def run():
        m = memory_mod.Memory()
        await m.append(1, 99, "user", "a")
        await m.append(1, 88, "user", "b")
        await m.append(2, 77, "user", "c")
        removed = await m.admin_reset_group(1)
        assert removed >= 2
        assert await m.get(1, 99) == []
        assert await m.get(2, 77) == [("user", "c")]
    asyncio.run(run())


def test_quota_check_and_consume(monkeypatch):
    tiny = replace(
        cfg.CONFIG.limits,
        openai_text_group=2,
        openai_text_user=1,
        openai_image_group=0,
        openai_image_user=0,
        openai_image_edit_group=0,
        openai_image_edit_user=0,
        openai_vision_group=0,
        openai_vision_user=0,
    )
    monkeypatch.setattr(cfg.CONFIG, "limits", tiny)

    async def run():
        q = quota_mod.Quota()
        ok, _ = await q.check("openai_text", 1, 99)
        assert ok
        await q.consume("openai_text", 1, 99)
        ok, reason = await q.check("openai_text", 1, 99)
        assert not ok and "user" in reason
        ok, _ = await q.check("openai_text", 1, 100)
        assert ok
        await q.consume("openai_text", 1, 100)
        ok, reason = await q.check("openai_text", 1, 101)
        assert not ok and "group" in reason

    asyncio.run(run())


def test_quota_admin_reset(monkeypatch):
    big = replace(cfg.CONFIG.limits, openai_text_group=10, openai_text_user=10)
    monkeypatch.setattr(cfg.CONFIG, "limits", big)

    async def run():
        q = quota_mod.Quota()
        await q.consume("openai_text", 1, 99)
        snap = await q.snapshot(1, 99)
        assert snap["openai_text"]["user"].startswith("1/")
        await q.admin_reset()
        snap = await q.snapshot(1, 99)
        assert snap["openai_text"]["user"].startswith("0/")

    asyncio.run(run())


def test_quota_dump_today():
    async def run():
        q = quota_mod.Quota()
        # consume something for two different groups
        await q.consume("openai_text", 1, 99)
        await q.consume("openai_text", 1, 99)
        await q.consume("openai_image", 2, 77)
        dump = await q.dump_today()
        assert dump["group"]["1"]["openai_text"] == 2
        assert dump["group"]["2"]["openai_image"] == 1
        assert dump["user"]["1:99"]["openai_text"] == 2
    asyncio.run(run())
