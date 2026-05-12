"""Strict-JSON router fallback logic (no network)."""
from __future__ import annotations

from bot.router import _coerce


def test_coerce_ok():
    d = _coerce('{"r":"image"}', fallback_prompt="画只猫")
    assert d.route == "openai_image"
    assert d.normalized_prompt == "画只猫"


def test_coerce_each_short_code():
    cases = {
        "chat": "deepseek_chat",
        "think": "deepseek_think",
        "gpt": "openai_text",
        "vision": "openai_vision",
        "image": "openai_image",
        "edit": "openai_image_edit",
        "skip": "skip",
        "no": "reject",
    }
    for short, internal in cases.items():
        d = _coerce(f'{{"r":"{short}"}}', "x")
        assert d.route == internal, f"{short} → {d.route}"


def test_coerce_extracts_embedded_json():
    payload = 'Sure: {"r":"chat"} trailing'
    assert _coerce(payload, "hi").route == "deepseek_chat"


def test_coerce_unknown_code_falls_back_to_skip():
    d = _coerce('{"r":"hyperdrive"}', fallback_prompt="x")
    assert d.route == "skip"


def test_coerce_garbage_falls_back_to_skip():
    d = _coerce("not json at all", fallback_prompt="orig")
    assert d.route == "skip"
    assert d.normalized_prompt == "orig"


def test_coerce_case_insensitive():
    assert _coerce('{"r":"CHAT"}', "x").route == "deepseek_chat"
    assert _coerce('{"r":" Skip  "}', "x").route == "skip"
