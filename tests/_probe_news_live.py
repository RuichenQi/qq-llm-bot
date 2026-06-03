"""Live probe: actually call send_news with different topics and print the
bot's reply. Uses the real Tavily + DeepSeek backends, so it costs a small
amount of quota each run. NOT a unit test — opt-in only.

Writes to data/news_probe.log as it goes (so you can watch progress).

Run: python tests/_probe_news_live.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Windows: force UTF-8 stdout so Chinese characters in log() don't crash on cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.command_handler import Handler
from bot.memory import Memory
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from bot.storage import Storage
from providers.deepseek import DeepSeekProvider
from providers.web_search import build_provider


TOPICS = [
    None,                       # default CONFIG.news_query
    "AI 大模型 最新发布",
    "太空 天文 最新发现",
    "新游戏 新硬件",
    "奇怪有趣的小事 动物",
    "六四 三十周年",             # sensitive — should be silently bailed
]


OUT_PATH = ROOT / "data" / "news_probe.log"


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with OUT_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


async def amain() -> None:
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("", encoding="utf-8")
    await Storage.get()
    deepseek = DeepSeekProvider()
    router = Router(deepseek)
    web_search = build_provider()
    if web_search is None:
        log("ERROR: TAVILY_API_KEY not set or WEB_SEARCH_ENABLED=0")
        return

    posted: list[tuple[int, str]] = []

    async def send_text(gid: int, text: str) -> None:
        posted.append((gid, text))

    async def send_image(gid: int, img: str) -> None:
        posted.append((gid, "[image]"))

    handler = Handler(
        deepseek=deepseek, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        web_search=web_search,
    )
    # Disable the bot's "human send" pacing — it splits replies into chunks
    # with random sleeps which makes the probe slow and harder to read.
    import config as cfg
    cfg.CONFIG.human_send_enabled = False

    for i, topic in enumerate(TOPICS, 1):
        label = topic if topic else "(默认 CONFIG.news_query)"
        log(f"\n{'=' * 72}")
        log(f"[{i}/{len(TOPICS)}] /news {label}")
        log("-" * 72)
        posted.clear()
        try:
            ok = await handler.send_news(group_id=999, query=topic)
        except Exception as e:
            log(f"  EXCEPTION: {type(e).__name__}: {e}")
            continue
        if not posted:
            log(f"  (silent bail; send_news returned {ok})")
            continue
        for _, text in posted:
            log("")
            for line in text.split("\n"):
                log(f"  → {line}")

    await handler.aclose()
    await deepseek.aclose()
    if hasattr(web_search, "aclose"):
        await web_search.aclose()
    store = await Storage.get()
    await store.close()
    log("\n=== done ===")


if __name__ == "__main__":
    asyncio.run(amain())
