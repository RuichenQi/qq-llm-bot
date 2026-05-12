"""/admin ping + WS status formatter."""
from __future__ import annotations

import asyncio
import time
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest

import config as cfg
from bot.command_handler import Handler, format_ws_status
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


@dataclass
class _FakeStatus:
    mode: str = "reverse"
    connected: bool = True
    connected_at: Optional[float] = None
    last_event_at: Optional[float] = None
    last_heartbeat_at: Optional[float] = None
    disconnect_count: int = 0
    last_disconnect_at: Optional[float] = None
    last_disconnect_reason: str = ""


def test_fmt_ago_buckets():
    now = 1_000_000.0
    out = format_ws_status(
        _FakeStatus(
            connected_at=now - 45,
            last_event_at=now - 10,
            last_heartbeat_at=now - 3700,
        ),
        now=now,
    )
    assert "已连接" in out
    assert "10 秒前" in out
    assert "1 小时" in out


def test_format_disconnected():
    now = 2_000_000.0
    s = _FakeStatus(
        connected=False,
        last_event_at=now - 500,
        disconnect_count=3,
        last_disconnect_at=now - 120,
        last_disconnect_reason="ConnectionClosed: 1006 abnormal",
    )
    out = format_ws_status(s, now=now)
    assert "未连接" in out
    assert "3" in out  # disconnect_count
    assert "1006" in out


def test_format_never_connected():
    out = format_ws_status(_FakeStatus(connected=False), now=1.0)
    assert "未连接" in out
    assert "从未" in out  # connected_at + last_event_at both None


def _event(text: str, *, user_id: int = 42):
    segs: list = []
    if text.startswith("/"):
        segs.append({"type": "at", "data": {"qq": "10000"}})
    segs.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": 1,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch, *, health=None) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid: int, text: str) -> None:
        sent.append((gid, text))

    async def send_image(gid: int, img: str) -> None:
        sent.append((gid, f"[image:{img[:40]}]"))

    stub_provider = types.SimpleNamespace(name="stub")

    async def stub_chat(*a, **kw):
        from providers.base import TextReply
        return TextReply(text="ok", model="stub")

    stub_provider.chat = stub_chat
    stub_provider.aclose = (lambda: asyncio.sleep(0))

    stub_router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    stub_router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)

    handler = Handler(
        deepseek=stub_provider,
        openai=None,
        router=stub_router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
        health_status=health,
    )
    return handler, sent


def test_admin_ping_uses_status(monkeypatch):
    now = time.time()
    status = _FakeStatus(
        mode="reverse",
        connected=True,
        connected_at=now - 60,
        last_event_at=now - 5,
        last_heartbeat_at=now - 5,
    )
    handler, sent = _make_handler(monkeypatch, health=lambda: status)
    parsed = parse_event(_event("/admin ping"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    flat = "\n".join(t for _, t in sent)
    assert "已连接" in flat
    assert "reverse" in flat


def test_admin_ping_without_callback(monkeypatch):
    handler, sent = _make_handler(monkeypatch, health=None)
    parsed = parse_event(_event("/admin ping"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert any("未注入" in t for _, t in sent)


def test_admin_ping_blocked_for_non_superuser(monkeypatch):
    handler, sent = _make_handler(monkeypatch, health=lambda: _FakeStatus())
    monkeypatch.setattr(cfg.CONFIG, "superusers", set(), raising=False)
    parsed = parse_event(_event("/admin ping"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())
    assert any("管理员" in t for _, t in sent)


def test_onebot_client_status_initial_state():
    """Without any connection, status() should be all-zero / None."""
    from bot.onebot_client import OneBotClient

    async def _noop(_evt):
        return None

    c = OneBotClient(_noop)
    s = c.status()
    assert s.connected is False
    assert s.connected_at is None
    assert s.disconnect_count == 0


def test_onebot_client_heartbeat_updates_stat(monkeypatch):
    """Feeding a heartbeat frame through _read_loop should bump last_heartbeat_at."""
    from bot.onebot_client import OneBotClient

    async def _noop(_evt):
        return None

    c = OneBotClient(_noop)

    class _FakeWs:
        def __init__(self, frames):
            self._frames = frames

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._frames:
                raise StopAsyncIteration
            return self._frames.pop(0)

    import json
    heartbeat = json.dumps({"post_type": "meta_event", "meta_event_type": "heartbeat"})

    async def run():
        await c._read_loop(_FakeWs([heartbeat]))

    asyncio.run(run())
    assert c._last_heartbeat_at is not None
    assert c._last_event_at is not None
