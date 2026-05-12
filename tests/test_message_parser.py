"""Pure-logic tests for the OneBot message parser."""
from __future__ import annotations

from bot.message_parser import chunk_text, parse_event


def _wrap_array(segments, **overrides):
    base = {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": 42,
        "raw_message": "",
        "message": segments,
        "sender": {"user_id": 42, "nickname": "x"},
    }
    base.update(overrides)
    return base


def test_parse_array_text_command():
    ev = _wrap_array([{"type": "text", "data": {"text": "/help me out"}}])
    p = parse_event(ev)
    assert p is not None
    assert p.is_command and p.command == "help" and p.command_args == "me out"


def test_parse_array_image_and_at():
    ev = _wrap_array(
        [
            {"type": "at", "data": {"qq": "10000"}},
            {"type": "text", "data": {"text": " 这张图说啥"}},
            {"type": "image", "data": {"url": "https://example.com/a.png"}},
        ]
    )
    p = parse_event(ev)
    assert p is not None
    assert p.image_urls == ["https://example.com/a.png"]
    assert p.at_targets == [10000]
    assert p.mentions(10000)
    assert "这张图说啥" in p.text


def test_parse_string_cq_format():
    raw = "/vision 看看 [CQ:image,url=https://example.com/b.png,file=b.png]"
    ev = _wrap_array(raw)
    ev["message"] = raw
    ev["raw_message"] = raw
    p = parse_event(ev)
    assert p is not None
    assert p.is_command and p.command == "vision"
    assert p.image_urls == ["https://example.com/b.png"]


def test_parse_rejects_non_group():
    p = parse_event(_wrap_array("hi", message_type="private"))
    assert p is None


def test_chunk_text_short():
    assert chunk_text("hello", 100) == ["hello"]


def test_chunk_text_long_prefers_newlines():
    body = ("a" * 50 + "\n") * 5
    parts = chunk_text(body, 80)
    assert all(len(p) <= 80 for p in parts)
    assert "".join(p + ("" if p.endswith("\n") else "\n") for p in parts).strip() == body.strip()
