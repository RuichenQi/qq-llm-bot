"""QQ Group LLM Bot — entry point."""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from bot import allowlist
from bot.command_handler import Handler
from bot.logger import get_logger, setup_logging
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.message_parser import QuotedMessage, parse_event
from bot.onebot_client import OneBotClient
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from bot.storage import Storage
from config import CONFIG
from providers.base import ProviderError
from providers.deepseek import DeepSeekProvider
from providers.openai_provider import OpenAIProvider

log = get_logger("main")


def _extract_quoted(payload: Dict[str, Any]) -> Optional[QuotedMessage]:
    """Pull text AND image URLs out of a get_msg response."""
    msg = payload.get("message")
    if isinstance(msg, str):
        text = msg.strip()
        return QuotedMessage(text=text) if text else None
    if isinstance(msg, list):
        parts: list[str] = []
        urls: list[str] = []
        for seg in msg:
            t = seg.get("type")
            data = seg.get("data") or {}
            if t == "text":
                parts.append(str(data.get("text", "")))
            elif t == "image":
                u = data.get("url") or data.get("file")
                if u:
                    urls.append(str(u))
        text = "".join(parts).strip()
        if not text and not urls:
            return None
        return QuotedMessage(text=text, image_urls=urls)
    raw = payload.get("raw_message")
    if isinstance(raw, str) and raw.strip():
        return QuotedMessage(text=raw.strip())
    return None


async def _seconds_until(target_hhmm: str) -> float:
    try:
        hh, mm = (int(x) for x in target_hhmm.split(":", 1))
    except Exception:
        hh, mm = 23, 55
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _daily_report_loop(handler: Handler) -> None:
    if CONFIG.daily_report_group <= 0:
        return
    log.info("daily report scheduled for %s (group %s)",
             CONFIG.daily_report_time, CONFIG.daily_report_group)
    while True:
        try:
            wait_s = await _seconds_until(CONFIG.daily_report_time)
            await asyncio.sleep(wait_s)
            await handler._send_daily_report()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("daily report loop iteration failed")
            await asyncio.sleep(60)


async def _image_sweeper(period_s: int = 600) -> None:
    while True:
        try:
            await asyncio.sleep(period_s)
            await Handler.sweep_image_cache()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("image sweeper iteration failed")


async def _daily_recap_loop(handler: Handler) -> None:
    """Once a day, save yesterday's recap for every allowed group + prune old."""
    if not CONFIG.daily_recap_enabled:
        return
    log.info("daily recap scheduled for %s", CONFIG.daily_recap_time)
    while True:
        try:
            wait_s = await _seconds_until(CONFIG.daily_recap_time)
            await asyncio.sleep(wait_s)
            if handler.long_memory is None:
                continue
            groups = sorted(await allowlist.all_allowed_groups())
            for gid in groups:
                try:
                    summary = await handler.long_memory.save_yesterday(gid)
                    if summary:
                        log.info("daily recap saved for group %s", gid)
                except Exception:
                    log.exception("daily recap failed for group %s", gid)
            try:
                pruned = await handler.long_memory.prune()
                if pruned:
                    log.info("daily recap pruned %d old rows", pruned)
            except Exception:
                log.exception("daily recap prune failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("daily recap loop iteration failed")
            await asyncio.sleep(60)


async def amain() -> int:
    setup_logging()
    if not CONFIG.deepseek_api_key:
        log.error("DEEPSEEK_API_KEY missing — cannot start.")
        return 2
    if not CONFIG.allowed_groups:
        log.warning(
            "ALLOWED_GROUPS env is empty — only groups added via /admin allow_group will work."
        )

    # Initialize SQLite + run JSON migration.
    await Storage.get()

    deepseek = DeepSeekProvider()
    openai: Optional[OpenAIProvider] = None
    try:
        openai = OpenAIProvider()
    except ProviderError as e:
        log.warning("OpenAI provider disabled: %s", e)

    router = Router(deepseek)
    memory = Memory()
    quota = Quota()
    rate = RateLimiter()
    client: Optional[OneBotClient] = None

    async def send_text(group_id: int, text: str) -> None:
        if client is None:
            return
        await client.send_group_msg(group_id, text)

    async def send_image(group_id: int, image: str) -> None:
        if client is None:
            return
        await client.send_group_image(group_id, image)

    async def fetch_reply(msg_id: str) -> Optional[QuotedMessage]:
        if client is None:
            return None
        payload = await client.get_msg(msg_id)
        if payload is None:
            return None
        return _extract_quoted(payload)

    handler = Handler(
        deepseek=deepseek,
        openai=openai,
        router=router,
        memory=memory,
        quota=quota,
        rate=rate,
        send_text=send_text,
        send_image=send_image,
        fetch_reply=fetch_reply,
        health_status=lambda: client.status() if client is not None else None,
    )

    async def on_event(event):
        parsed = parse_event(event)
        if parsed is None:
            return
        await handler.handle(parsed)

    client = OneBotClient(on_event)

    stop_event = asyncio.Event()

    def _request_stop(*_):
        log.info("stop requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, _request_stop)

    run_task = asyncio.create_task(client.run(), name="ws_run")
    report_task = asyncio.create_task(_daily_report_loop(handler), name="daily_report")
    sweeper_task = asyncio.create_task(_image_sweeper(), name="image_sweeper")
    recap_task = asyncio.create_task(_daily_recap_loop(handler), name="daily_recap")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop_wait")

    try:
        await asyncio.wait(
            {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        await client.stop()
        for t in (run_task, stop_task, report_task, sweeper_task, recap_task):
            if not t.done():
                t.cancel()
        await handler.aclose()
        await deepseek.aclose()
        if openai is not None:
            await openai.aclose()
        store = await Storage.get()
        await store.close()

    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
