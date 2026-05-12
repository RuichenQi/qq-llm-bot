"""Tests for the strict-JSON router fallback logic (no network)."""
from __future__ import annotations

from bot.router import _coerce


def test_coerce_ok():
    payload = '{"route":"openai_image","confidence":0.9,"reason":"画图","normalized_prompt":"画只猫"}'
    d = _coerce(payload, fallback_prompt="画只猫")
    assert d.route == "openai_image"
    assert 0.0 <= d.confidence <= 1.0
    assert d.normalized_prompt == "画只猫"


def test_coerce_extracts_embedded_json():
    payload = "Sure! Here it is: {\"route\":\"deepseek_chat\",\"confidence\":0.5,\"reason\":\"x\",\"normalized_prompt\":\"hi\"} trailing"
    d = _coerce(payload, fallback_prompt="hi")
    assert d.route == "deepseek_chat"


def test_coerce_unknown_route_falls_back():
    payload = '{"route":"hyperdrive","confidence":1.0,"reason":"","normalized_prompt":"x"}'
    d = _coerce(payload, fallback_prompt="x")
    assert d.route == "deepseek_chat"


def test_coerce_garbage_falls_back():
    d = _coerce("not json at all", fallback_prompt="orig")
    assert d.route == "deepseek_chat"
    assert d.normalized_prompt == "orig"
