"""Important-memory layer.

A unified, LLM-judged memory system that sits next to `group_memory` (the raw
rolling log) and `long_memory` (daily recaps). The LLM decides what's worth
keeping — and how — for each incoming message. Two consumption paths share
the same `memories` table:

  1. Reminder firing: rows with `trigger_at <= now` are fired by a periodic
     loop (see main._reminder_loop).
  2. Context injection: relevant rows are pulled into the system prompt on
     every chat turn (see command_handler._run_text).

Cost control: a cheap regex pre-filter gates the LLM classifier, so the vast
majority of group messages don't pay the round-trip.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from bot.logger import get_logger
from bot.storage import Storage
from providers.base import ChatMessage, ProviderError
from providers.deepseek import DeepSeekProvider

log = get_logger(__name__)


# Pre-filter: a message must contain at least one of these to be considered for
# LLM classification. Tuned for Chinese + English. False negatives are fine —
# the LLM still summarises everything via daily_recap.
_KEYWORD_RE = re.compile(
    r"提醒|记得|别忘|不要忘|叫我|喊我|催我|"
    r"(\d{1,2}|两|一|二|三|四|五|六|七|八|九|十)\s*(点|分钟|小时|个小时)|"
    r"今晚|明天|后天|下周|周[一二三四五六日天]|"
    r"早上|早晨|中午|下午|晚上|凌晨|"
    r"我喜欢|我讨厌|我是|我在|我家|"
    r"过敏|生日|纪念日|约|说好|决定|"
    r"remind|remember|don'?t forget|tomorrow|tonight|"
    r"(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?",
    re.IGNORECASE,
)


_CLASSIFIER_SYSTEM = """你是一个 QQ 群聊机器人的「记忆判断员」。
你的任务：读一条消息，判断这条消息里有没有值得长期记住的事情。

何为「值得记」（有一项即可）：
- 定时提醒：用户希望某时刻被叫一下（"晚上9点叫我"、"半小时后提醒我"）
- 个人事实：能影响今后对话的偏好/状况（"我对花生过敏"、"我家在北京"、"我是程序员"）
- 群体约定/决定：大家达成的事（"周五开会"、"今天决定用方案A"）
- 模糊期限任务：有截止但没具体时间（"这周内交报告"）

何为「不值得记」（绝大多数消息）：
- 普通闲聊、玩笑、表情、附和
- 已经结束的、一次性的提问
- 关于天气、新闻、八卦的讨论
- 转瞬即逝的情绪表达

输出严格 JSON（一行）：
{
  "remember": true | false,
  "content": "用一句中文写下要记的事，含主语",
  "subject_user_id": <int or null, null=群体共享>,
  "trigger_at": "YYYY-MM-DD HH:MM" or null,
  "recurrence": "daily HH:MM" or null,
  "expires_at": "YYYY-MM-DD HH:MM" or null,
  "importance": 0.0..1.0,
  "tags": ["关键词1", "关键词2"]
}

判断规则：
- 默认 remember=false，不确定就 false
- "我..." 开头的偏好/事实 → subject_user_id 就是说话人
- "周五" 这种没说年份的 → 解析为下一个最近的周五
- 已经过去的时间不要设 trigger_at
- importance: 涉及健康/约定/明确的提醒 0.7+；一般偏好 0.4~0.6；模糊的 < 0.3
- 不要把现成的 group_memory 行重复存进来；只存有"新信息量"的事
"""


@dataclass
class ClassifyResult:
    remember: bool
    content: str
    subject_user_id: Optional[int]
    trigger_at: Optional[float]   # unix ts
    recurrence: Optional[str]
    expires_at: Optional[float]
    importance: float
    tags: List[str]


def _parse_iso_local(s: Optional[str]) -> Optional[float]:
    """Accept 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DDTHH:MM' as local-time."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


_DAILY_RE = re.compile(r"^\s*daily\s+(\d{1,2}):(\d{2})\s*$", re.IGNORECASE)


def next_recurrence(recurrence: str, after_ts: float) -> Optional[float]:
    """Compute next firing time strictly after `after_ts`. Only 'daily HH:MM'
    supported in v1. Returns None for unknown formats."""
    m = _DAILY_RE.match(recurrence)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    base = datetime.fromtimestamp(after_ts)
    cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= base:
        cand += timedelta(days=1)
    return cand.timestamp()


class ImportantMemory:
    """LLM-driven importance classifier + retrieval, backed by the memories table."""

    def __init__(self, deepseek: DeepSeekProvider) -> None:
        self._deepseek = deepseek

    # ---------- ingest ----------
    @staticmethod
    def _passes_prefilter(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        return _KEYWORD_RE.search(text) is not None

    async def maybe_extract(
        self,
        *,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
    ) -> Optional[int]:
        """Fire-and-forget entry point. Returns inserted memory id or None.

        Silently swallows errors — this is best-effort enrichment, never
        user-facing."""
        if not self._passes_prefilter(text):
            return None
        result = await self._classify(group_id=group_id, user_id=user_id, text=text)
        if result is None or not result.remember:
            return None
        # Sanity floor: a memory with no trigger AND low importance isn't worth
        # the disk. Tuned empirically; we may need to revisit.
        if result.trigger_at is None and result.importance < 0.25:
            log.info("memory dropped (low value): %r", result.content[:60])
            return None
        store = await Storage.get()
        mem_id = await store.memory_item_insert(
            group_id=group_id,
            subject_user_id=result.subject_user_id,
            content=result.content,
            importance=result.importance,
            tags=" ".join(result.tags)[:200],
            trigger_at=result.trigger_at,
            recurrence=result.recurrence,
            expires_at=result.expires_at,
            created_at=time.time(),
            source_text=text,
            source_nickname=nickname,
        )
        log.info(
            "memory saved id=%s subject=%s trigger=%s imp=%.2f content=%r",
            mem_id, result.subject_user_id,
            datetime.fromtimestamp(result.trigger_at).isoformat()
            if result.trigger_at else "-",
            result.importance, result.content[:60],
        )
        return mem_id

    async def _classify(
        self, *, group_id: int, user_id: int, text: str,
    ) -> Optional[ClassifyResult]:
        now = datetime.now()
        user_prompt = (
            f"当前时间：{now.strftime('%Y-%m-%d %H:%M')} ({now.strftime('%A')})\n"
            f"群 id：{group_id}\n"
            f"说话人 user_id：{user_id}\n"
            f"消息：{text}\n\n"
            "请输出 JSON。"
        )
        try:
            reply = await self._deepseek.chat(
                [
                    ChatMessage(role="system", content=_CLASSIFIER_SYSTEM),
                    ChatMessage(role="user", content=user_prompt),
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
        except ProviderError as e:
            log.warning("memory classifier call failed: %s", e)
            return None
        return self._parse_classifier_reply(reply.text, default_user_id=user_id)

    @staticmethod
    def _parse_classifier_reply(
        raw: str, *, default_user_id: int,
    ) -> Optional[ClassifyResult]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = m.group(0) if m else raw
        try:
            obj = json.loads(payload)
        except Exception:
            log.info("classifier reply not valid JSON: %r", raw[:200])
            return None
        remember = bool(obj.get("remember", False))
        if not remember:
            return ClassifyResult(
                remember=False, content="", subject_user_id=None,
                trigger_at=None, recurrence=None, expires_at=None,
                importance=0.0, tags=[],
            )
        content = str(obj.get("content", "")).strip()[:500]
        if not content:
            return None
        subj_raw = obj.get("subject_user_id")
        if isinstance(subj_raw, bool):
            subj_raw = None
        try:
            subject_user_id: Optional[int] = (
                int(subj_raw) if subj_raw is not None else None
            )
        except (TypeError, ValueError):
            subject_user_id = default_user_id
        trigger_at = _parse_iso_local(obj.get("trigger_at"))
        # Don't fire reminders in the past.
        if trigger_at is not None and trigger_at <= time.time():
            trigger_at = None
        recurrence_raw = obj.get("recurrence")
        recurrence = (
            str(recurrence_raw).strip()
            if isinstance(recurrence_raw, str) and recurrence_raw.strip()
            else None
        )
        # If recurrence given but no first trigger, compute it from now.
        if recurrence and trigger_at is None:
            trigger_at = next_recurrence(recurrence, time.time())
        expires_at = _parse_iso_local(obj.get("expires_at"))
        try:
            importance = float(obj.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        importance = max(0.0, min(1.0, importance))
        tags_raw = obj.get("tags") or []
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in re.split(r"[,\s]+", tags_raw) if t.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tags = []
        return ClassifyResult(
            remember=True, content=content, subject_user_id=subject_user_id,
            trigger_at=trigger_at, recurrence=recurrence, expires_at=expires_at,
            importance=importance, tags=tags[:8],
        )

    # ---------- read ----------
    async def recall_for_user(
        self, group_id: int, user_id: int, limit: int = 6,
    ) -> List[Tuple[int, Optional[int], str, float, str, Optional[float]]]:
        """Top memories to inject into a chat turn's system prompt."""
        store = await Storage.get()
        return await store.memory_item_recall(group_id, user_id, time.time(), limit)

    @staticmethod
    def format_for_prompt(
        rows: List[Tuple[int, Optional[int], str, float, str, Optional[float]]],
        *, speaker_user_id: int,
    ) -> Optional[str]:
        """Render recall rows into a system-prompt snippet, or None if empty."""
        if not rows:
            return None
        lines: List[str] = []
        for _id, subj, content, _imp, _tags, trigger in rows:
            who = "（关于你）" if subj == speaker_user_id else "（群）"
            t_part = ""
            if trigger:
                t_part = (
                    f" [{datetime.fromtimestamp(trigger).strftime('%m-%d %H:%M')} 提醒]"
                )
            lines.append(f"- {who} {content}{t_part}")
        return (
            "你记得的相关事项（被问到或顺其自然提到时可参考，不要主动复读）：\n"
            + "\n".join(lines)
        )

    # ---------- reminders ----------
    async def due_reminders(
        self, now_ts: Optional[float] = None, limit: int = 50,
    ) -> List[Tuple[int, int, Optional[int], str, str, Optional[str]]]:
        store = await Storage.get()
        return await store.memory_item_due(now_ts or time.time(), limit)

    async def mark_fired(self, item_id: int, recurrence: Optional[str]) -> None:
        now = time.time()
        next_t: Optional[float] = None
        if recurrence:
            next_t = next_recurrence(recurrence, now)
        store = await Storage.get()
        await store.memory_item_mark_fired(item_id, now, next_t)

    async def cancel(self, item_id: int, group_id: int) -> bool:
        store = await Storage.get()
        return await store.memory_item_cancel(item_id, group_id)

    async def list_pending(
        self, group_id: int, user_id: Optional[int] = None, limit: int = 20,
    ) -> List[Tuple[int, Optional[int], str, Optional[float], Optional[str]]]:
        store = await Storage.get()
        return await store.memory_item_list_pending(group_id, user_id, limit)

    # ---------- maintenance ----------
    async def maintenance_pass(self, group_ids: List[int]) -> Tuple[int, int]:
        """Run per-tick housekeeping. Returns (expired_count, deduped_count)."""
        store = await Storage.get()
        expired = await store.memory_item_expire(time.time())
        deduped_total = 0
        for gid in group_ids:
            try:
                deduped_total += await self._dedup_group(gid)
            except Exception:
                log.exception("dedup pass failed for group=%s", gid)
        # Hard-delete tombstones older than a year so the table doesn't bloat.
        await store.memory_item_prune(365)
        if expired or deduped_total:
            log.info("memory maintenance: expired=%d deduped=%d",
                     expired, deduped_total)
        return expired, deduped_total

    async def _dedup_group(self, group_id: int) -> int:
        """Ask the LLM to point at duplicates among recent memories. Conservative:
        only delete what the LLM explicitly flags."""
        store = await Storage.get()
        rows = await store.memory_item_dedup_candidates(group_id, limit=50)
        if len(rows) < 4:
            return 0
        catalogue_lines = []
        for _id, subj, content, tags, _ts in rows:
            subj_s = "群" if subj is None else f"u{subj}"
            tag_s = f" [tags: {tags}]" if tags else ""
            catalogue_lines.append(f"#{_id} ({subj_s}) {content}{tag_s}")
        catalogue = "\n".join(catalogue_lines)
        system = (
            "你在帮 QQ 群聊机器人整理它的长期记忆。"
            "下面是一组记忆条目。请找出语义重复或互相矛盾的条目，"
            "**保留最新/最完整的一条**，把要删的条目 id 输出来。\n"
            "标准：\n"
            "- 同一个人同一个事实重复多条 → 留最新\n"
            "- 同一个提醒重复 → 留最早的一条 trigger\n"
            "- 已被新事实推翻的旧偏好（比如先说喜欢，后说不喜欢）→ 删旧的\n"
            "- 拿不准就保留\n\n"
            "输出 JSON：{\"delete\":[id1, id2, ...]}。"
            "如果没有重复，输出 {\"delete\":[]}。"
        )
        try:
            reply = await self._deepseek.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=catalogue),
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        except ProviderError as e:
            log.warning("dedup LLM call failed: %s", e)
            return 0
        m = re.search(r"\{.*\}", reply.text, re.DOTALL)
        try:
            obj = json.loads(m.group(0) if m else reply.text)
        except Exception:
            return 0
        ids_raw = obj.get("delete") or []
        if not isinstance(ids_raw, list):
            return 0
        valid_ids = {r[0] for r in rows}
        to_delete = [int(x) for x in ids_raw if isinstance(x, int) and x in valid_ids]
        if not to_delete:
            return 0
        return await store.memory_item_delete_many(to_delete)
