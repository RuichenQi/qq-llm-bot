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
    # The important-memory classifier fires an extra LLM call per message;
    # tests that don't opt in shouldn't see that behaviour leak into their
    # stub provider's call log.
    monkeypatch.setattr(cfg.CONFIG, "important_memory_enabled", False, raising=False)

    Storage._instance = None  # type: ignore[attr-defined]
    Storage._init_lock = None  # type: ignore[attr-defined]
    yield
    Storage._instance = None  # type: ignore[attr-defined]
    Storage._init_lock = None  # type: ignore[attr-defined]
