"""Sentence splitter: decimal-aware, trailing-period-stripped."""
from __future__ import annotations

from bot.command_handler import Handler


def test_decimal_stays_joined():
    chunks = Handler._split_human_chunks("Pi is 3.14. It is irrational.", 3)
    assert any("3.14" in c for c in chunks), chunks
    # The "3.14" chunk should not be split into "3" / "14".
    for c in chunks:
        assert not c.endswith("3"), chunks


def test_version_string_stays_joined():
    chunks = Handler._split_human_chunks("v1.2.3 is the new version.", 3)
    assert any("v1.2.3" in c for c in chunks), chunks


def test_trailing_period_stripped_english():
    chunks = Handler._split_human_chunks("Hello world.", 3)
    assert chunks == ["Hello world"], chunks


def test_trailing_period_stripped_chinese():
    chunks = Handler._split_human_chunks("好的。", 3)
    assert chunks == ["好的"], chunks


def test_exclamation_preserved():
    chunks = Handler._split_human_chunks("太好啦！", 3)
    assert chunks == ["太好啦！"], chunks


def test_question_mark_preserved():
    chunks = Handler._split_human_chunks("真的吗?", 3)
    assert chunks == ["真的吗?"], chunks


def test_multiple_sentences_split():
    chunks = Handler._split_human_chunks(
        "First sentence. Second sentence. Third sentence.", 3,
    )
    assert len(chunks) == 3
    # None should end with a period.
    for c in chunks:
        assert not c.endswith(".")


def test_paragraph_break_splits():
    text = "段落一。\n\n段落二。"
    chunks = Handler._split_human_chunks(text, 3)
    assert len(chunks) == 2
    assert "段落一" in chunks[0]
    assert "段落二" in chunks[1]


def test_ellipsis_preserved():
    # "..." or "…" shouldn't be stripped as trailing period.
    chunks = Handler._split_human_chunks("怎么说呢…", 3)
    assert chunks == ["怎么说呢…"], chunks
