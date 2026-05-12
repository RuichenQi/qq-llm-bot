"""Probabilistic emoji stripper."""
from __future__ import annotations

import random

from bot.emoji_filter import filter_emoji


def test_keep_zero_strips_all():
    random.seed(0)
    cases = {
        "嘿嘿😊": ["😊"],
        "好喜欢 ✨🌸": ["✨", "🌸"],
        "❤️ 想吃🍜": ["❤", "🍜"],
        "👌🏻👌🏻👌🏻ok": ["👌"],
        "诶呀 🥺 别这样": ["🥺"],
        "纯文本无 emoji": [],
    }
    for src, forbidden in cases.items():
        out = filter_emoji(src, keep_probability=0.0)
        for em in forbidden:
            assert em not in out, f"{em!r} leaked from {src!r} → {out!r}"


def test_keep_one_passes_all():
    text = "嘿嘿😊 好开心 ✨🌸"
    assert filter_emoji(text, keep_probability=1.0) == text


def test_empty_input_safe():
    assert filter_emoji("", 0.5) == ""


def test_plain_text_untouched():
    text = "你好啊，今天去吃啥呢"
    out = filter_emoji(text, 0.0)
    assert out == text


def test_zwj_sequence_stripped():
    """Multi-codepoint ZWJ emoji like 👨‍💻 should go away as one unit."""
    out = filter_emoji("hi 👨‍💻 nice", keep_probability=0.0)
    assert "👨" not in out and "💻" not in out
    assert "hi" in out and "nice" in out


def test_keep_probability_statistical():
    """At keep=0.5, ~half survive over many trials."""
    random.seed(42)
    survived = 0
    trials = 400
    for _ in range(trials):
        out = filter_emoji("😊", keep_probability=0.5)
        if "😊" in out:
            survived += 1
    ratio = survived / trials
    assert 0.40 < ratio < 0.60, f"keep ratio {ratio:.2f} not near 0.5"
