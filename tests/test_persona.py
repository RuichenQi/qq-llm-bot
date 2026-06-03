"""Persona loader: default + custom file + nickname interpolation."""
from __future__ import annotations

from pathlib import Path

import config as cfg
from bot.persona import DEFAULT_PERSONA, load_persona


def test_default_persona_substitutes_nickname(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_persona_file", "", raising=False)
    p = load_persona()
    assert "{nickname}" not in p
    assert "小笨蛋" in p
    assert "温柔" in p  # default persona content


def test_default_persona_includes_owner_identity(monkeypatch):
    """The persona must teach the bot who R is, so group members asking
    "你主人是谁" get a correct answer."""
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_persona_file", "", raising=False)
    p = load_persona()
    # The owner's identifiers — handle + QQ — are in the persona prompt.
    assert "1424403605" in p
    assert "R" in p
    # And the persona explicitly tells the bot how to describe him —
    # 温柔可爱 is the wording the bot should produce when answering.
    assert "主人" in p
    assert "00 后" in p or "00后" in p
    assert "温柔可爱" in p
    # The bot is told to only mention this when asked, not unprompted.
    assert "被问到" in p or "问到" in p


def test_persona_from_file(monkeypatch, tmp_path):
    persona_path = tmp_path / "p.txt"
    persona_path.write_text("你是 {nickname}，今天是英语角的小老师。", encoding="utf-8")
    monkeypatch.setattr(cfg.CONFIG, "bot_persona_file", str(persona_path), raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "瓜瓜", raising=False)
    p = load_persona()
    assert p == "你是 瓜瓜，今天是英语角的小老师。"


def test_missing_persona_file_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg.CONFIG, "bot_persona_file", str(tmp_path / "nope.txt"), raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    p = load_persona()
    assert "温柔" in p  # default is in effect


def test_default_persona_constant_uses_placeholder():
    """Sanity: the constant itself must have the placeholder, not a hardcoded name."""
    assert "{nickname}" in DEFAULT_PERSONA
