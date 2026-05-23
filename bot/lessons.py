"""Unified 功能注入 (function injection) layer.

This is the single durable home for everything the group has *taught* the
bot — behavior rules, personal facts, group agreements, and scheduled
reminders. It replaces the previous split between `lessons` (rules only)
and `important_memory` (reminders/facts), consolidating both ingestion
and retrieval through one LLM classifier and one SQLite table.

Two consumption paths:
  1. Reminder firing — rows with `trigger_at <= now` are fired by the
     periodic loop in main.py.
  2. Prompt injection — `active_for_user()` pulls the top-N items into
     the system prompt of every chat turn so the bot follows them.

Why one classifier:
- A single LLM call decides the *kind* of memory (rule/fact/agreement/
  reminder/none) and fills in every column the storage layer needs.
- Cheaper than chaining two classifiers (lesson + memory) the way the
  old split required.

Cost control:
- A cheap regex pre-filter gates the classifier; messages with no
  teaching/temporal/preference keywords skip the round-trip entirely.
- @-mentioned or nickname-addressed messages always go through (since
  the user is talking to the bot directly).
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


_VALID_KINDS = {"rule", "fact", "agreement", "reminder"}


# Pre-filter for non-addressed messages: must contain at least one keyword
# associated with rule-teaching, time, or personal preference. Addressed
# messages (@bot or nickname-in-text) bypass this gate.
_KEYWORD_RE = re.compile(
    r"提醒|记得|别忘|不要忘|叫我|喊我|催我|"
    r"以后|从今天起|从现在起|今后|从此|"
    r"(\d{1,2}|两|一|二|三|四|五|六|七|八|九|十)\s*(点|分钟|小时|个小时)|"
    r"今晚|明天|后天|下周|周[一二三四五六日天]|"
    r"早上|早晨|中午|下午|晚上|凌晨|"
    r"我喜欢|我讨厌|我是|我在|我家|"
    r"过敏|生日|纪念日|约|说好|决定|定下|约好|"
    r"你要|你不要|你别|你得|你应该|你应当|你需要|"
    r"remind|remember|don'?t forget|tomorrow|tonight|from now on|"
    r"(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?",
    re.IGNORECASE,
)


_CLASSIFIER_SYSTEM = """你是 QQ 群聊机器人的「功能注入判断员」。
群里有人发了一条消息。判断这条消息里有没有值得机器人**长期记住**的东西，并给它分类。

四种值得记的类型：
1. rule       — 教机器人一条今后要遵守的行为规则
   例: "以后群里有人重复一句话，你也跟着重复一遍"
       "看到 R 发消息要 @ 他"
       "你说话简短点"
2. fact       — 关于说话人或某成员的持久事实/偏好
   例: "我对花生过敏"
       "我家在北京"
       "我是程序员"
3. agreement  — 群体达成的约定 / 决定 / 计划
   例: "周五开会"
       "今天决定用方案 A"
       "周末聚餐定在新荣记"
4. reminder   — 用户希望在某时刻或某条件下被叫一下
   例: "晚上9点叫我喝水"           (一次性)
       "每天早上 8 点提醒大家打卡"  (重复)
       "下周一记得交报告"
   设置 trigger_at；如果是重复，再设 recurrence。

如果以上都不像，输出 kind="none"。绝大多数消息都是 "none"。

输出严格 JSON（一行）：
{
  "kind": "none" | "rule" | "fact" | "agreement" | "reminder",
  "content": "用一句中文写下要记的事，主语清楚（'你'指机器人；'我'保留为说话人）",
  "subject_user_id": <int or null>,        // null = 全群；规则永远为 null
  "trigger_at": "YYYY-MM-DD HH:MM" or null, // 仅 reminder 用，且必须为未来时间
  "recurrence": "daily HH:MM" or null,
  "expires_at": "YYYY-MM-DD HH:MM" or null,
  "importance": 0.0..1.0,
  "tags": ["关键词"]
}

判断规则：
- 拿不准就 kind="none"
- "我..." 开头的偏好/事实 → kind="fact"，subject_user_id=说话人
- "你..." 让机器人做/不做某事 → kind="rule"，subject_user_id=null
- 涉及时间词 → 多半是 reminder；解析为最近一次未来时间
- 已经过去的时间不要设 trigger_at
- 不要把现成的群消息（"今天天气好"这类闲聊）当成要记的事
- importance: 健康/约定/明确提醒 0.7+；行为规则 0.6+；一般偏好 0.4~0.6；模糊 < 0.3
"""


@dataclass
class ClassifyResult:
    kind: str                       # "none" if nothing to remember
    content: str
    subject_user_id: Optional[int]
    trigger_at: Optional[float]
    recurrence: Optional[str]
    expires_at: Optional[float]
    importance: float
    tags: List[str]


@dataclass
class ActiveLesson:
    """A row pulled for prompt injection."""
    id: int
    kind: str
    subject_user_id: Optional[int]
    content: str
    importance: float
    tags: str
    trigger_at: Optional[float]


def _parse_iso_local(s: Optional[str]) -> Optional[float]:
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
    supported. Returns None for unknown formats."""
    m = _DAILY_RE.match(recurrence)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    base = datetime.fromtimestamp(after_ts)
    cand = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= base:
        cand += timedelta(days=1)
    return cand.timestamp()


class Lessons:
    """Unified ingestion + retrieval for rules / facts / agreements / reminders."""

    def __init__(self, deepseek: DeepSeekProvider) -> None:
        self._deepseek = deepseek

    # ---------- ingest ----------
    @staticmethod
    def _passes_prefilter(text: str, *, force: bool) -> bool:
        if not text or len(text) < 3:
            return False
        if force:
            return True
        return _KEYWORD_RE.search(text) is not None

    async def maybe_learn(
        self,
        *,
        group_id: int,
        user_id: int,
        text: str,
        addressed: bool = False,
    ) -> Optional[int]:
        """Fire-and-forget ingestion. Returns inserted row id or None.

        `addressed=True` means the user @-mentioned the bot or used its
        nickname — bypass the keyword pre-filter so rules/teaching commands
        always go through, even if their wording dodges the keyword regex."""
        if not self._passes_prefilter(text, force=addressed):
            return None
        try:
            result = await self._classify(
                group_id=group_id, user_id=user_id, text=text,
            )
        except Exception:
            log.exception("lesson classifier crashed")
            return None
        if result is None or result.kind not in _VALID_KINDS:
            return None
        content = result.content.strip()
        if not content or len(content) > 500:
            return None
        # Sanity floor: kind=reminder without a trigger is just a passive
        # note; if it also lacks importance, skip.
        if result.kind == "reminder" and result.trigger_at is None and result.importance < 0.4:
            log.info("lesson dropped (low-value reminder): %r", content[:60])
            return None
        store = await Storage.get()
        # Rules are never personal. Force subject_user_id=None.
        subject = None if result.kind == "rule" else result.subject_user_id
        row_id = await store.lesson_insert(
            group_id=group_id,
            kind=result.kind,
            subject_user_id=subject,
            content=content,
            importance=result.importance,
            tags=" ".join(result.tags)[:200],
            trigger_at=result.trigger_at,
            recurrence=result.recurrence,
            expires_at=result.expires_at,
            source_user_id=user_id,
            source_text=text,
            created_at=time.time(),
        )
        log.info(
            "lesson saved id=%s kind=%s group=%s imp=%.2f trigger=%s content=%r",
            row_id, result.kind, group_id, result.importance,
            datetime.fromtimestamp(result.trigger_at).isoformat()
            if result.trigger_at else "-",
            content[:80],
        )
        return row_id

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
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
        except ProviderError as e:
            log.warning("lesson classifier call failed: %s", e)
            return None
        return self._parse_classifier_reply(reply.text, default_user_id=user_id)

    @staticmethod
    def _parse_classifier_reply(
        raw: str, *, default_user_id: int,
    ) -> Optional[ClassifyResult]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            obj = json.loads(m.group(0) if m else raw)
        except Exception:
            log.info("classifier reply not valid JSON: %r", raw[:200])
            return None
        kind = str(obj.get("kind", "none")).strip().lower()
        if kind == "none":
            return ClassifyResult(
                kind="none", content="", subject_user_id=None,
                trigger_at=None, recurrence=None, expires_at=None,
                importance=0.0, tags=[],
            )
        if kind not in _VALID_KINDS:
            log.info("classifier returned unknown kind=%r — drop", kind)
            return None
        content = str(obj.get("content", "")).strip()[:500]
        if not content:
            return None
        subj_raw = obj.get("subject_user_id")
        if isinstance(subj_raw, bool):
            subj_raw = None
        subject_user_id: Optional[int]
        if subj_raw is None:
            # Facts default to the speaker — they nearly always describe "me".
            # Rules / agreements stay group-wide.
            subject_user_id = default_user_id if kind == "fact" else None
        else:
            try:
                subject_user_id = int(subj_raw)
            except (TypeError, ValueError):
                subject_user_id = default_user_id if kind == "fact" else None
        trigger_at = _parse_iso_local(obj.get("trigger_at"))
        if trigger_at is not None and trigger_at <= time.time():
            trigger_at = None
        recurrence_raw = obj.get("recurrence")
        recurrence = (
            str(recurrence_raw).strip()
            if isinstance(recurrence_raw, str) and recurrence_raw.strip()
            else None
        )
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
            kind=kind, content=content, subject_user_id=subject_user_id,
            trigger_at=trigger_at, recurrence=recurrence, expires_at=expires_at,
            importance=importance, tags=tags[:8],
        )

    # ---------- retrieval ----------
    async def active_for_user(
        self, group_id: int, subject_user_id: int, limit: int = 12,
    ) -> List[ActiveLesson]:
        """Top items to inject into a chat turn's system prompt."""
        store = await Storage.get()
        rows = await store.lesson_active_for_user(
            group_id, subject_user_id, time.time(), limit,
        )
        return [
            ActiveLesson(
                id=r[0], kind=r[1], subject_user_id=r[2], content=r[3],
                importance=r[4], tags=r[5], trigger_at=r[6],
            )
            for r in rows
        ]

    async def list_pending(
        self, group_id: int, subject_user_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Tuple[int, str, Optional[int], str, Optional[float], Optional[str]]]:
        """Active rows for /remember. Returns (id, kind, subject_user_id,
        content, trigger_at, recurrence)."""
        store = await Storage.get()
        return await store.lesson_list_pending(group_id, subject_user_id, limit)

    async def list_all(
        self, group_id: int, limit: int = 50,
    ) -> List[Tuple[int, str, str, float, str, Optional[float], Optional[str]]]:
        """All rows for /admin lessons."""
        store = await Storage.get()
        return await store.lesson_list(group_id, limit)

    @staticmethod
    def format_for_prompt(
        rows: List[ActiveLesson], *, speaker_user_id: int,
    ) -> Optional[str]:
        """Render active rows as a system-prompt block grouped by kind."""
        if not rows:
            return None
        buckets: dict[str, List[str]] = {
            "rule": [], "fact": [], "agreement": [], "reminder": [],
        }
        for r in rows:
            if r.kind == "fact":
                who = "你（说话人）" if r.subject_user_id == speaker_user_id else \
                      (f"u{r.subject_user_id}" if r.subject_user_id else "群")
                buckets["fact"].append(f"- ({who}) {r.content}")
            elif r.kind == "agreement":
                buckets["agreement"].append(f"- {r.content}")
            elif r.kind == "reminder":
                t_part = ""
                if r.trigger_at:
                    t_part = (
                        f" [{datetime.fromtimestamp(r.trigger_at).strftime('%m-%d %H:%M')}]"
                    )
                buckets["reminder"].append(f"- {r.content}{t_part}")
            else:
                buckets.setdefault(r.kind, []).append(f"- {r.content}")
        sections: List[str] = []
        labels = [
            ("rule", "行为规则（必须遵守）"),
            ("fact", "关于群成员的事实/偏好"),
            ("agreement", "群体约定/决定"),
            ("reminder", "正在等的提醒"),
        ]
        for key, label in labels:
            items = buckets.get(key) or []
            if items:
                sections.append(f"【{label}】\n" + "\n".join(items))
        if not sections:
            return None
        return "群里教给你的事项（按重要度排序，被问到或自然提到时参考）：\n" + "\n\n".join(sections)

    # ---------- reminders ----------
    async def due_reminders(
        self, now_ts: Optional[float] = None, limit: int = 50,
    ) -> List[Tuple[int, int, Optional[int], str, str, Optional[str]]]:
        """Rows whose trigger_at <= now. Returns (id, group_id,
        subject_user_id, content, kind, recurrence)."""
        store = await Storage.get()
        return await store.lesson_due(now_ts or time.time(), limit)

    async def mark_fired(
        self, lesson_id: int, recurrence: Optional[str],
    ) -> None:
        now = time.time()
        next_t: Optional[float] = None
        if recurrence:
            next_t = next_recurrence(recurrence, now)
        store = await Storage.get()
        await store.lesson_mark_fired(lesson_id, now, next_t)

    async def cancel(self, lesson_id: int, group_id: int) -> bool:
        store = await Storage.get()
        return await store.lesson_revoke(lesson_id, group_id)

    # ---------- maintenance ----------
    async def maintenance_pass(
        self, group_ids: List[int], keep_days: int = 365,
    ) -> Tuple[int, int]:
        """Periodic dedup + expiry. Returns (expired_count, deduped_count)."""
        store = await Storage.get()
        expired = await store.lesson_expire(time.time())
        deduped_total = 0
        for gid in group_ids:
            try:
                deduped_total += await self._dedup_group(gid)
            except Exception:
                log.exception("dedup pass failed for group=%s", gid)
        await store.lesson_prune(keep_days)
        if expired or deduped_total:
            log.info("lesson maintenance: expired=%d deduped=%d",
                     expired, deduped_total)
        return expired, deduped_total

    async def _dedup_group(self, group_id: int) -> int:
        """LLM-judged dedup. Conservative: only revoke flagged ids."""
        store = await Storage.get()
        rows = await store.lesson_dedup_candidates(group_id, limit=50)
        if len(rows) < 4:
            return 0
        lines = []
        for _id, kind, subj, content, tags, _ts in rows:
            subj_s = "群" if subj is None else f"u{subj}"
            tag_s = f" [tags: {tags}]" if tags else ""
            lines.append(f"#{_id} ({kind}/{subj_s}) {content}{tag_s}")
        catalogue = "\n".join(lines)
        system = (
            "下面是 QQ 群机器人学到的长期事项（规则/事实/约定/提醒）。"
            "请找出语义重复或互相矛盾的条目，"
            "**保留最新最完整的一条**，把要删的条目 id 输出来。\n"
            "标准：\n"
            "- 同一行为/事实重复 → 留最新\n"
            "- 新规则推翻旧规则 → 删旧的\n"
            "- 拿不准就保留\n\n"
            "输出 JSON：{\"delete\":[id1, id2]}。没重复就 {\"delete\":[]}。"
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
        except ProviderError:
            return 0
        m = re.search(r"\{.*\}", reply.text, re.DOTALL)
        try:
            obj = json.loads(m.group(0) if m else reply.text)
        except Exception:
            return 0
        ids_raw = obj.get("delete") or []
        if not isinstance(ids_raw, list):
            return 0
        valid = {r[0] for r in rows}
        targets = [int(x) for x in ids_raw if isinstance(x, int) and x in valid]
        if not targets:
            return 0
        return await store.lesson_revoke_many(targets, group_id)
