"""Message parser: file segments + group_upload notice events."""
from __future__ import annotations

from bot.message_parser import parse_event


def _group_event(message, **overrides):
    base = {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": 42,
        "raw_message": "",
        "message": message,
        "sender": {"user_id": 42, "nickname": "x"},
    }
    base.update(overrides)
    return base


def test_parse_file_segment_array():
    ev = _group_event([
        {"type": "text", "data": {"text": "看这个"}},
        {"type": "file", "data": {
            "file": "report.pdf", "url": "https://x/report.pdf",
            "file_id": "abc123", "size": "12345",
        }},
    ])
    p = parse_event(ev)
    assert p is not None
    assert p.has_file
    assert p.files[0].name == "report.pdf"
    assert p.files[0].url == "https://x/report.pdf"
    assert p.files[0].file_id == "abc123"
    assert p.files[0].size == 12345
    assert "看这个" in p.text


def test_parse_file_segment_string():
    raw = "/file 总结一下 [CQ:file,file=notes.txt,url=https://x/n.txt,size=200]"
    ev = _group_event(raw)
    ev["message"] = raw
    ev["raw_message"] = raw
    p = parse_event(ev)
    assert p is not None
    assert p.has_file
    assert p.files[0].name == "notes.txt"
    assert p.is_command and p.command == "file"


def test_group_upload_notice_becomes_parsed_message():
    ev = {
        "post_type": "notice",
        "notice_type": "group_upload",
        "self_id": 10000,
        "group_id": 1,
        "user_id": 42,
        "file": {
            "id": "ZZZZZ",
            "name": "deck.pdf",
            "size": 99999,
            "url": "https://x/deck.pdf",
        },
    }
    p = parse_event(ev)
    assert p is not None
    assert p.has_file
    assert p.files[0].name == "deck.pdf"
    assert p.files[0].url == "https://x/deck.pdf"
    assert p.files[0].file_id == "ZZZZZ"
    assert p.text == ""


def test_non_message_non_upload_returns_none():
    ev = {"post_type": "notice", "notice_type": "group_increase"}
    assert parse_event(ev) is None
