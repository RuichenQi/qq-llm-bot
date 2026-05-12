"""End-to-end: /vision pipeline should download → downscale → send data URI."""
from __future__ import annotations

import asyncio
import types
from io import BytesIO
from typing import List, Tuple

import pytest
from PIL import Image

import config as cfg
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from providers.base import TextReply


def _png(size, colour=(0, 200, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _event(text: str, image_url: str | None = None, user_id: int = 42):
    segs: list = []
    # Commands now require @bot.
    if text.startswith("/"):
        segs.append({"type": "at", "data": {"qq": "10000"}})
    if image_url:
        segs.append({"type": "image", "data": {"url": image_url}})
    if text:
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


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "max_vision_input_size", 128, raising=False)


def test_vision_resizes_input_to_max_size():
    captured: List[Tuple[str, List[str]]] = []
    deepseek = types.SimpleNamespace(name="deepseek")

    async def chat(*a, **kw):
        return TextReply(text="ok", model="stub")

    deepseek.chat = chat

    async def aclose():
        return None

    deepseek.aclose = aclose

    openai = types.SimpleNamespace(name="openai")

    async def vision(q, urls, **kw):
        captured.append((q, urls))
        return TextReply(text="(看到了个绿方块)", model="stub-vision")

    openai.vision = vision

    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    handler = Handler(
        deepseek=deepseek,
        openai=openai,
        router=types.SimpleNamespace(decide=lambda *a, **k: None),
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=lambda *a, **k: None,
    )

    big_png = _png((640, 480))

    async def fake_download(url):
        return big_png

    handler._download = fake_download

    parsed = parse_event(_event("/vision 这是啥", image_url="https://x/big.png"))
    asyncio.run(handler.handle(parsed))
    asyncio.run(handler.aclose())

    assert captured, "vision provider was not called"
    _q, urls = captured[0]
    assert len(urls) == 1
    uri = urls[0]
    assert uri.startswith("data:image/png;base64,"), "expected data URI"
    # decode and confirm size <= MAX
    import base64
    raw = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(BytesIO(raw)) as im:
        assert max(im.size) == 128
        assert im.size == (128, 96)  # 4:3 preserved
