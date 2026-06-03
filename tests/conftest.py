"""Shared test fixtures: every test gets a fresh SQLite DB and image dir."""
from __future__ import annotations

import pytest

import config as cfg
from bot.storage import Storage


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    db_file = tmp_path / "state.db"
    mem_file = tmp_path / "memory.json"
    quo_file = tmp_path / "quota.json"

    monkeypatch.setattr(cfg, "DB_FILE", db_file)
    monkeypatch.setattr(cfg, "MEMORY_FILE", mem_file)
    monkeypatch.setattr(cfg, "QUOTA_FILE", quo_file)
    monkeypatch.setattr(cfg, "IMAGE_DIR", image_dir)

    from bot import storage as storage_mod
    from bot import command_handler as ch_mod
    monkeypatch.setattr(storage_mod, "DB_FILE", db_file)
    monkeypatch.setattr(storage_mod, "MEMORY_FILE", mem_file)
    monkeypatch.setattr(storage_mod, "QUOTA_FILE", quo_file)
    monkeypatch.setattr(ch_mod, "IMAGE_DIR", image_dir)

    # Disable human-send delays in unit tests by default — tests that care
    # can re-enable it explicitly.
    monkeypatch.setattr(cfg.CONFIG, "human_send_enabled", False, raising=False)
    # Lessons classifier fires an extra background LLM call per addressed
    # message; off by default so stub call logs stay clean.
    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", False, raising=False)
    # Ambient gate uses random.random(); pin to deterministic always-pass for
    # tests that exercise non-addressed messages. Tests that want to verify
    # gating override this explicitly.
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_high", 1.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_question", 1.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_low", 1.0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_min_seconds", 0, raising=False)
    # Quoted-image intent gate runs a vision call — switch it off for unit tests
    # that aren't exercising it. Tests that care can flip back on.
    monkeypatch.setattr(cfg.CONFIG, "quoted_image_intent_gate", False, raising=False)

    Storage._instance = None  # type: ignore[attr-defined]
    Storage._init_lock = None  # type: ignore[attr-defined]
    yield
    Storage._instance = None  # type: ignore[attr-defined]
    Storage._init_lock = None  # type: ignore[attr-defined]
