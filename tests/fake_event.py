"""Fake OneBot event runner — feed a text (and optional image URL) into the
Handler without needing the WebSocket adapter.

Usage:
    python -m tests.fake_event "/help"
    python -m tests.fake_event "帮我画一只柴犬"
    python -m tests.fake_event "这张图里有什么" --image https://example.com/x.png

It re-uses the real providers + router (so DEEPSEEK_API_KEY must be set).
Set ALLOWED_GROUPS=1 in .env (or pass --group 1) so the message isn't filtered.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from bot.command_handler import Handler
from bot.logger import setup_logging, get_logger
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from bot.storage import Storage
from config import CONFIG
from providers.base import ProviderError
from providers.deepseek import DeepSeekProvider
from providers.openai_provider import OpenAIProvider

log = get_logger("fake_event")


def build_event(text: str, group_id: int, user_id: int, image_url: Optional[str]) -> dict:
    message: list = []
    if text:
        message.append({"type": "text", "data": {"text": text}})
    if image_url:
        message.append({"type": "image", "data": {"url": image_url, "file": image_url}})
    raw = text + (f" [CQ:image,url={image_url}]" if image_url else "")
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": raw,
        "message": message,
        "sender": {"user_id": user_id, "nickname": "tester", "card": "tester"},
    }


async def amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="message text (with or without /command)")
    parser.add_argument("--image", default=None, help="attach an image URL")
    parser.add_argument("--group", type=int, default=None,
                        help="group id (default: first ALLOWED_GROUPS entry, or 1)")
    parser.add_argument("--user", type=int, default=42, help="user id")
    args = parser.parse_args()

    setup_logging()

    group_id = args.group
    if group_id is None:
        group_id = next(iter(CONFIG.allowed_groups), 1)

    # Ensure the test group is allowed even if .env omits it.
    if group_id not in CONFIG.allowed_groups:
        CONFIG.allowed_groups.add(group_id)  # type: ignore[attr-defined]

    deepseek = DeepSeekProvider()
    openai_provider: Optional[OpenAIProvider] = None
    try:
        openai_provider = OpenAIProvider()
    except ProviderError as e:
        log.warning("OpenAI disabled: %s", e)

    sent_texts: list[tuple[int, str]] = []
    sent_images: list[tuple[int, str]] = []

    async def send_text(gid: int, text: str) -> None:
        sent_texts.append((gid, text))
        print(f"\n=== bot → group {gid} (text) ===\n{text}\n")

    async def send_image(gid: int, image: str) -> None:
        sent_images.append((gid, image))
        preview = image if len(image) < 160 else image[:160] + f"...({len(image)} chars)"
        print(f"\n=== bot → group {gid} (image) ===\n{preview}\n")

    handler = Handler(
        deepseek=deepseek,
        openai=openai_provider,
        router=Router(deepseek),
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(),
        send_text=send_text,
        send_image=send_image,
    )

    event = build_event(args.text, group_id, args.user, args.image)
    parsed = parse_event(event)
    if parsed is None:
        print("parser rejected the event", file=sys.stderr)
        return 1

    await handler.handle(parsed)

    await handler.aclose()
    await deepseek.aclose()
    if openai_provider is not None:
        await openai_provider.aclose()
    store = await Storage.get()
    await store.close()

    if not sent_texts and not sent_images:
        print("(no reply produced)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
