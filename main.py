"""QQ Group LLM Bot — entry point."""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bot import allowlist
from bot.command_handler import Handler
from bot.logger import get_logger, setup_logging
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.message_parser import AttachedFile, QuotedMessage, parse_event
from bot.onebot_client import OneBotClient
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from bot.storage import Storage
from config import CONFIG
from providers.base import ProviderError
from providers.deepseek import DeepSeekProvider
from providers.openai_provider import OpenAIProvider
from providers.web_search import build_provider as build_web_search

log = get_logger("main")


def _extract_quoted(payload: Dict[str, Any]) -> Optional[QuotedMessage]:
    """Pull text, image URLs, and file segments out of a get_msg response."""
    msg = payload.get("message")
    if isinstance(msg, str):
        text = msg.strip()
        return QuotedMessage(text=text) if text else None
    if isinstance(msg, list):
        parts: list[str] = []
        urls: list[str] = []
        files: list[AttachedFile] = []
        for seg in msg:
            t = seg.get("type")
            data = seg.get("data") or {}
            if t == "text":
                parts.append(str(data.get("text", "")))
            elif t == "image":
                u = data.get("url") or data.get("file")
                if u:
                    urls.append(str(u))
            elif t == "file":
                try:
                    size = int(data.get("size") or 0)
                except (TypeError, ValueError):
                    size = 0
                files.append(AttachedFile(
                    name=str(data.get("file") or data.get("name") or "file"),
                    url=str(data.get("url") or ""),
                    file_id=str(data.get("file_id") or data.get("id") or ""),
                    size=size,
                ))
        text = "".join(parts).strip()
        if not text and not urls and not files:
            return None
        return QuotedMessage(text=text, image_urls=urls, files=files)
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


async def _maintenance_loop(handler: Handler) -> None:
    """Periodic housekeeping: rolling daily-recap refresh + memories dedup
    + expiry. Replaces the once-a-day recap loop. Cadence from config."""
    period = max(60, CONFIG.maintenance_tick_seconds)
    log.info("maintenance loop tick every %ds", period)
    while True:
        try:
            await asyncio.sleep(period)
            await handler.run_maintenance()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("maintenance loop iteration failed")
            await asyncio.sleep(60)


def _seconds_until_news_time() -> float:
    """Seconds from now until the next NEWS_TIME in NEWS_TIME_TZ. If the
    target hour:minute has already passed today, returns the gap to the
    same time tomorrow. Falls back to UTC if the configured timezone name
    can't be loaded (rare on Windows without tzdata)."""
    try:
        hh, mm = (int(x) for x in CONFIG.news_time.split(":", 1))
    except Exception:
        hh, mm = 9, 0
    try:
        tz = ZoneInfo(CONFIG.news_time_tz)
    except ZoneInfoNotFoundError:
        log.warning(
            "news: timezone %r not available, falling back to UTC",
            CONFIG.news_time_tz,
        )
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _news_loop(handler: Handler) -> None:
    """Daily news drop. Sleeps until next NEWS_TIME (default 09:00 Asia/
    Shanghai), fires `handler.send_news_to_all_groups()`, repeats.

    Per-group min-interval guards against double-posting from a quick restart.
    Per-group `/stop` pause skips silenced groups. The actual content selection
    happens inside the handler — we just trigger the daily fire here.
    """
    if not CONFIG.news_enabled:
        return
    log.info(
        "news loop: daily at %s %s",
        CONFIG.news_time, CONFIG.news_time_tz,
    )
    while True:
        try:
            wait_s = _seconds_until_news_time()
            log.info("news loop: next post in %.0fs (%.1fh)",
                     wait_s, wait_s / 3600.0)
            await asyncio.sleep(wait_s)
            fired = await handler.send_news_to_all_groups()
            log.info("news loop: fired in %d group(s)", fired)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("news loop iteration failed")
            # Don't burn CPU if the loop body keeps throwing — back off a
            # few minutes before retrying the schedule calculation.
            await asyncio.sleep(300)


async def _reminder_loop(handler: Handler) -> None:
    """Fire due reminders from the unified lessons layer."""
    if not CONFIG.lessons_enabled:
        return
    period = max(10, CONFIG.reminder_tick_seconds)
    log.info("reminder loop tick every %ds", period)
    while True:
        try:
            await asyncio.sleep(period)
            await handler.fire_due_reminders()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reminder loop iteration failed")
            await asyncio.sleep(30)


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

    async def fetch_file_url(group_id: int, file_id: str) -> Optional[str]:
        if client is None or not file_id:
            return None
        return await client.get_group_file_url(group_id, file_id)

    web_search = build_web_search()
    if web_search is not None:
        log.info("web search backend enabled: %s", web_search.name)

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
        fetch_file_url=fetch_file_url,
        web_search=web_search,
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
    maint_task = asyncio.create_task(_maintenance_loop(handler), name="maintenance")
    reminder_task = asyncio.create_task(_reminder_loop(handler), name="reminders")
    news_task = asyncio.create_task(_news_loop(handler), name="news")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop_wait")

    try:
        await asyncio.wait(
            {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        await client.stop()
        for t in (run_task, stop_task, report_task, sweeper_task,
                  maint_task, reminder_task, news_task):
            if not t.done():
                t.cancel()
        await handler.aclose()
        await deepseek.aclose()
        if openai is not None:
            await openai.aclose()
        if web_search is not None and hasattr(web_search, "aclose"):
            try:
                await web_search.aclose()
            except Exception:
                log.exception("web_search aclose failed")
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
