"""Long-term memory: daily recaps of each allowed group.

A background task summarises yesterday's group_memory rows into a short
plain-text recap and saves it to the `daily_recaps` table. The handler then
injects the most recent few days' recaps into chat calls so the bot has
'a year of memory' it can vaguely reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from bot.group_memory import GroupMemory, GroupMsg
from bot.logger import get_logger
from bot.storage import Storage
from config import CONFIG
from providers.base import ChatMessage, ProviderError
from providers.deepseek import DeepSeekProvider

log = get_logger(__name__)


_SUMMARY_SYSTEM_PROMPT = """你是一个忠实的群聊归档员。把这一天群里的聊天浓缩成一段长时记忆。
要求：
- 100 字以内
- 用平淡的第三人称事实陈述
- 概括当天主要话题、谁参与最多、有意思的事 / 决定 / 计划 / 笑话
- 称呼 bot 为「我」（指未来读这段记忆的 bot 自己）
- 不要 emoji，不要 markdown，不要分点列表
- 不要复述具体每句话，只抓核心"""


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _day_bounds(day: str) -> Tuple[float, float]:
    """Return (start_ts, end_ts) for a local date YYYY-MM-DD."""
    d = datetime.strptime(day, "%Y-%m-%d")
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


class LongMemory:
    """Long-term memory wrapper around the daily_recaps table."""

    def __init__(self, group_memory: GroupMemory, deepseek: DeepSeekProvider) -> None:
        self._gm = group_memory
        self._deepseek = deepseek

    async def save_day(self, group_id: int, day: str) -> Optional[str]:
        """Build + save a recap for the given (group, day). Returns the
        summary, or None if nothing to summarise / LLM failed."""
        start, end = _day_bounds(day)
        rows = await self._gm.since(group_id, start)
        rows = [r for r in rows if r.ts < end]
        if not rows:
            log.info("long_memory: no rows for group=%s day=%s", group_id, day)
            return None
        # Feed every row from the day to the LLM — truncating loses morning
        # context for evening-heavy groups. Hard cap is a safety belt; storage
        # already tops out at CONFIG.group_memory_max so this rarely trips.
        if len(rows) > 3000:
            rows = rows[-3000:]
        transcript = self._format_transcript(rows)
        try:
            reply = await self._deepseek.chat(
                [
                    ChatMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=transcript),
                ],
                temperature=0.2,
                max_tokens=300,
            )
        except ProviderError as e:
            log.warning("long_memory summary failed for %s/%s: %s", group_id, day, e)
            return None
        summary = (reply.text or "").strip()
        if not summary:
            return None
        store = await Storage.get()
        await store.daily_recap_upsert(group_id, day, summary)
        log.info("long_memory saved group=%s day=%s chars=%d",
                 group_id, day, len(summary))
        return summary

    async def save_yesterday(self, group_id: int) -> Optional[str]:
        return await self.save_day(group_id, _yesterday())

    async def recent(self, group_id: int, days: Optional[int] = None
                     ) -> List[Tuple[str, str]]:
        days = days if days is not None else CONFIG.long_memory_inject_days
        if days <= 0:
            return []
        store = await Storage.get()
        return await store.daily_recap_recent(group_id, days)

    async def get(self, group_id: int, day: str) -> Optional[str]:
        store = await Storage.get()
        return await store.daily_recap_get(group_id, day)

    async def search(self, group_id: int, keyword: str, limit: int = 5
                     ) -> List[Tuple[str, str]]:
        store = await Storage.get()
        return await store.daily_recap_search(group_id, keyword, limit)

    async def prune(self) -> int:
        store = await Storage.get()
        return await store.daily_recap_prune(CONFIG.daily_recap_keep_days)

    @staticmethod
    def _format_transcript(rows: List[GroupMsg]) -> str:
        lines = []
        for r in rows:
            t = datetime.fromtimestamp(r.ts).strftime("%H:%M")
            lines.append(f"[{t} {r.nickname}] {r.text}")
        return "\n".join(lines)
