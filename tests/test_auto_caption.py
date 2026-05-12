"""Background auto-captioning of group images."""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import pytest

import config as cfg
from bot.command_handler import Handler
from bot.group_memory import GroupMemory
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event_with_image(text: str = "", *, image_url: str = "https://example.com/x.png",
                      user_id: int = 200, group_id: int = 1,
                      nickname: str = "Alice"):
    segs: list = []
    if text:
        segs.append({"type": "text", "data": {"text": text}})
    segs.append({"type": "image", "data": {"url": image_url}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": nickname, "card": nickname},
    }


def _make_handler(monkeypatch, *, vision_text: str = "一碗拉面"):
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    deepseek = types.SimpleNamespace(name="deepseek")

    async def chat(*a, **k):
        from providers.base import TextReply
        return TextReply(text='{"r":"skip"}', model="stub", usage={})

    deepseek.chat = chat

    async def aclose():
        return None

    deepseek.aclose = aclose

    openai = types.SimpleNamespace(name="openai")
    vision_calls: list = []

    async def vision(question, image_urls, **kw):
        from providers.base import TextReply
        vision_calls.append((question, image_urls))
        return TextReply(text=vision_text, model="stub-v", usage={})

    openai.vision = vision

    router = types.SimpleNamespace()

    async def decide(text, *, has_image, was_at_bot=False):
        from bot.router import RouteDecision
        return RouteDecision("skip", 1.0, "skip", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "mention", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "bot_nickname", "小笨蛋", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "auto_vision_group_images", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "proactive_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "max_vision_input_size", 256, raising=False)

    handler = Handler(
        deepseek=deepseek,
        openai=openai,
        router=router,
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(per_minute=999),
        send_text=send_text,
        send_image=send_image,
        group_memory=GroupMemory(),
    )

    # Stub the network download with a tiny valid PNG.
    from io import BytesIO
    from PIL import Image
    png_buf = BytesIO()
    Image.new("RGB", (80, 80), (1, 2, 3)).save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    async def fake_download(url):
        return png_bytes

    handler._download = fake_download

    return handler, sent, vision_calls


def test_auto_caption_updates_group_memory_row(monkeypatch):
    handler, sent, vision_calls = _make_handler(monkeypatch, vision_text="一碗热腾腾的拉面")

    async def run():
        # bare image
        await handler.handle(parse_event(_event_with_image()))
        # wait for background caption task
        for _ in range(50):
            rows = await handler.group_memory.recent(1, limit=5)
            if rows and "拉面" in rows[-1].text:
                return rows
            await asyncio.sleep(0.02)
        return await handler.group_memory.recent(1, limit=5)

    rows = asyncio.run(run())
    asyncio.run(handler.aclose())
    assert vision_calls, "vision should have been called"
    assert rows, "row should have been written"
    assert "拉面" in rows[-1].text, f"caption not in row: {rows[-1].text}"


def test_auto_caption_preserves_user_text(monkeypatch):
    handler, sent, vision_calls = _make_handler(monkeypatch, vision_text="蓝天白云")

    async def run():
        await handler.handle(parse_event(_event_with_image(text="你们看这个")))
        for _ in range(50):
            rows = await handler.group_memory.recent(1, limit=5)
            if rows and "蓝天" in rows[-1].text:
                return rows
            await asyncio.sleep(0.02)
        return await handler.group_memory.recent(1, limit=5)

    rows = asyncio.run(run())
    asyncio.run(handler.aclose())
    assert rows
    assert "你们看这个" in rows[-1].text
    assert "蓝天" in rows[-1].text


def test_auto_caption_disabled_by_config(monkeypatch):
    handler, sent, vision_calls = _make_handler(monkeypatch)
    monkeypatch.setattr(cfg.CONFIG, "auto_vision_group_images", False, raising=False)

    async def run():
        await handler.handle(parse_event(_event_with_image()))
        await asyncio.sleep(0.1)
        return await handler.group_memory.recent(1, limit=5)

    rows = asyncio.run(run())
    asyncio.run(handler.aclose())
    assert vision_calls == []
    # Row exists with placeholder, no caption
    assert rows and rows[-1].text == "[图片]"


def test_auto_caption_respects_daily_cap(monkeypatch):
    from dataclasses import replace
    new_limits = replace(cfg.CONFIG.limits, auto_vision_group=0)
    monkeypatch.setattr(cfg.CONFIG, "limits", new_limits, raising=False)
    handler, sent, vision_calls = _make_handler(monkeypatch)

    async def run():
        await handler.handle(parse_event(_event_with_image()))
        await asyncio.sleep(0.1)

    asyncio.run(run())
    asyncio.run(handler.aclose())
    assert vision_calls == [], "should not call vision when daily cap is 0"
