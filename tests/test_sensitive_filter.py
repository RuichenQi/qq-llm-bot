"""Mainland-China politically-sensitive query gate on web_search."""
from __future__ import annotations

from bot.command_handler import _query_is_blocked


def test_blocks_june_fourth():
    assert _query_is_blocked("六四 真相")
    assert _query_is_blocked("天安门事件")
    assert _query_is_blocked("8964")


def test_blocks_leaders():
    assert _query_is_blocked("习近平 评价")
    assert _query_is_blocked("江泽民 八九")


def test_blocks_party_terms():
    assert _query_is_blocked("中共 历史")
    assert _query_is_blocked("政治局 决策")


def test_blocks_minority_political():
    assert _query_is_blocked("新疆 集中营 报告")
    assert _query_is_blocked("达赖 流亡")
    assert _query_is_blocked("法轮功 history")


def test_blocks_protest_terms():
    assert _query_is_blocked("香港 抗议 2019")
    assert _query_is_blocked("白纸运动 时间线")
    assert _query_is_blocked("台湾独立 民调")


def test_blocks_circumvention():
    assert _query_is_blocked("VPN 中国 翻墙")
    assert _query_is_blocked("GFW 工作原理")


def test_allows_general_china_topics():
    """Generic China topics that aren't politically sensitive must pass."""
    assert not _query_is_blocked("北京 旅游 推荐")
    assert not _query_is_blocked("上海 美食 攻略")
    assert not _query_is_blocked("唐朝 历史")
    assert not _query_is_blocked("故宫 开放时间")
    assert not _query_is_blocked("中国 GDP 2025")
    assert not _query_is_blocked("中文 学习 方法")


def test_allows_unrelated():
    assert not _query_is_blocked("python asyncio tutorial")
    assert not _query_is_blocked("天气 北京 今天")
    assert not _query_is_blocked("")
    assert not _query_is_blocked("怎么做番茄炒蛋")
