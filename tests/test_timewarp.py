"""`/timewarp` past-period parser: shorthand phrases + ISO formats."""
from __future__ import annotations

from datetime import datetime, timedelta

from bot.command_handler import Handler


def _label(s):
    parsed = Handler._parse_timewarp(s)
    assert parsed is not None, f"parser rejected {s!r}"
    return parsed[2]


def _days_ago_center(s):
    parsed = Handler._parse_timewarp(s)
    assert parsed is not None, f"parser rejected {s!r}"
    start_ts, end_ts, _ = parsed
    center = (start_ts + end_ts) / 2
    delta = datetime.now().timestamp() - center
    return delta / 86400  # days


def test_empty_defaults_to_a_year_ago():
    assert _label("") == "一年前"
    assert 355 < _days_ago_center("") < 375


def test_yi_nian_qian():
    assert _label("一年前") == "一年前"


def test_ban_nian_qian():
    assert _label("半年前") == "半年前"
    assert 170 < _days_ago_center("半年前") < 190


def test_san_ge_yue_qian():
    assert _label("三个月前") == "三个月前"
    assert 80 < _days_ago_center("三个月前") < 100


def test_shang_ge_yue():
    assert _label("上个月") == "一个月前"
    assert 20 < _days_ago_center("上个月") < 40


def test_shang_zhou():
    assert _label("上周") == "上周"
    assert 1 < _days_ago_center("上周") < 14


def test_n_days_ago_chinese():
    parsed = Handler._parse_timewarp("100 天前")
    assert parsed is not None
    assert "100" in parsed[2]
    assert 95 < _days_ago_center("100 天前") < 105


def test_iso_month_window_is_full_month():
    parsed = Handler._parse_timewarp("2025-03")
    assert parsed is not None
    start_ts, end_ts, label = parsed
    assert label == "2025-03"
    start_d = datetime.fromtimestamp(start_ts)
    end_d = datetime.fromtimestamp(end_ts)
    assert start_d.year == 2025 and start_d.month == 3 and start_d.day == 1
    assert end_d.year == 2025 and end_d.month == 4 and end_d.day == 1


def test_iso_date_centers_on_date():
    parsed = Handler._parse_timewarp("2025-05-15")
    assert parsed is not None
    start_ts, end_ts, _ = parsed
    target = datetime.strptime("2025-05-15", "%Y-%m-%d").timestamp()
    # Window should include the target.
    assert start_ts < target < end_ts
    # ±~3 day window.
    assert (end_ts - start_ts) / 86400 < 10


def test_unrecognised_returns_none():
    assert Handler._parse_timewarp("noodle soup") is None
    assert Handler._parse_timewarp("2099/01/01") is None
