"""Ad-hoc end-to-end test: run /image through the real handler, save the PNG."""
from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path

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

log = get_logger("manual")


async def amain(prompt: str) -> int:
    setup_logging()

    group_id = next(iter(CONFIG.allowed_groups), 1)
    if group_id not in CONFIG.allowed_groups:
        CONFIG.allowed_groups.add(group_id)

    deepseek = DeepSeekProvider()
    try:
        openai = OpenAIProvider()
    except ProviderError as e:
        print("OpenAI not configured:", e)
        return 1

    saved: list[Path] = []
    sent_texts: list[str] = []

    async def send_text(gid, text):
        sent_texts.append(text)
        print(f"[bot text → {gid}] {text[:200]}")

    async def send_image(gid, payload):
        if payload.startswith("base64://"):
            data = base64.b64decode(payload[len("base64://"):])
            out = Path(f"f:/qqbot/data/test_image_{int(time.time())}.png")
            out.write_bytes(data)
            saved.append(out)
            print(f"[bot image → {gid}] saved {len(data)} bytes → {out}")
        else:
            print(f"[bot image → {gid}] (url) {payload[:120]}")

    handler = Handler(
        deepseek=deepseek,
        openai=openai,
        router=Router(deepseek),
        memory=Memory(),
        quota=Quota(),
        rate=RateLimiter(),
        send_text=send_text,
        send_image=send_image,
    )

    ev = {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": next(iter(CONFIG.superusers), 1),  # superuser → bypass quota
        "raw_message": f"/image {prompt}",
        "message": [{"type": "text", "data": {"text": f"/image {prompt}"}}],
        "sender": {"user_id": 42, "nickname": "tester"},
    }
    parsed = parse_event(ev)
    t0 = time.time()
    await handler.handle(parsed)
    print(f"[elapsed] {time.time() - t0:.1f}s")

    await handler.aclose()
    await deepseek.aclose()
    await openai.aclose()
    store = await Storage.get()
    await store.close()

    return 0 if saved else (0 if sent_texts else 1)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "a shiba inu wearing tiny sunglasses, vector art"
    sys.exit(asyncio.run(amain(prompt)))
