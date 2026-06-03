"""SQLite-backed durable state.

One file (data/state.db) holds:
- conversation memory (one row per turn)
- daily quota counters
- runtime group allow-list

aiosqlite is used in single-connection mode — fine for our throughput (a few
messages per second at most) and survives `kill -9` on Termux thanks to WAL.

A one-time JSON migration runs on first init if `data/memory.json` or
`data/quota.json` exist.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiosqlite

from bot.logger import get_logger
from config import DB_FILE, MEMORY_FILE, QUOTA_FILE

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    group_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    seq      INTEGER NOT NULL,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    PRIMARY KEY (group_id, user_id, seq)
);
CREATE INDEX IF NOT EXISTS memory_by_pair ON memory(group_id, user_id, seq);

CREATE TABLE IF NOT EXISTS quota (
    day   TEXT NOT NULL,
    scope TEXT NOT NULL,            -- 'group' or 'user'
    key   TEXT NOT NULL,            -- '<gid>' or '<gid>:<uid>'
    route TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, scope, key, route)
);
CREATE INDEX IF NOT EXISTS quota_by_day ON quota(day);

CREATE TABLE IF NOT EXISTS groups (
    group_id  INTEGER PRIMARY KEY,
    added_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_report (
    day TEXT PRIMARY KEY
);

-- Per-group pause flag. Presence of a row means the bot is silenced in that
-- group (still records group_memory so context survives a /start, but won't
-- reply or run LLM routes). Removed by /start. Allow-list state is separate:
-- a group can be allow-listed AND paused at the same time.
CREATE TABLE IF NOT EXISTS group_pause (
    group_id  INTEGER PRIMARY KEY,
    paused_at TEXT NOT NULL,
    paused_by INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS group_memory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id  INTEGER NOT NULL,
    ts        REAL NOT NULL,
    user_id   INTEGER NOT NULL,
    nickname  TEXT NOT NULL,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS group_memory_by_group_ts
    ON group_memory(group_id, ts);

-- Long-term memory: one row per (group, day) holding an LLM-summarised
-- recap of that day's conversation. Kept for up to a year.
CREATE TABLE IF NOT EXISTS daily_recaps (
    group_id   INTEGER NOT NULL,
    day        TEXT NOT NULL,            -- "YYYY-MM-DD" local date
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL,            -- ISO timestamp
    PRIMARY KEY (group_id, day)
);
CREATE INDEX IF NOT EXISTS daily_recaps_by_group_day
    ON daily_recaps(group_id, day);

-- (The pre-unify `memories` table is no longer created here. The migration
-- in `_migrate_memories_into_lessons` reads from it if present on legacy
-- DBs, then `_init` drops the empty husk so it stops appearing in tooling.)

-- Lessons: unified durable knowledge the group has shared with the bot —
-- behavior rules, personal facts, group agreements, and scheduled reminders.
-- Reminder-firing + context-injection both read from this one table.
--
-- kind ∈ {rule, fact, agreement, reminder}.
--   rule       behavioral instruction the bot should follow
--   fact       persistent attribute of a person or the group
--   agreement  decision / plan the group reached
--   reminder   passive "remember this" without a trigger time
--   (rows with trigger_at set act as scheduled reminders regardless of kind)
CREATE TABLE IF NOT EXISTS lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id        INTEGER NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'rule',
    subject_user_id INTEGER,                       -- NULL = group-wide
    content         TEXT NOT NULL,                 -- 'rule' column kept for legacy reads
    rule            TEXT NOT NULL DEFAULT '',      -- legacy mirror of content
    importance      REAL NOT NULL DEFAULT 0.6,
    tags            TEXT NOT NULL DEFAULT '',
    trigger_at      REAL,                          -- unix ts; reminder firing
    recurrence      TEXT,                          -- 'daily HH:MM' | NULL
    expires_at      REAL,
    source_user_id  INTEGER NOT NULL,
    source_text     TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    last_used_at    REAL,
    status          TEXT NOT NULL DEFAULT 'active',  -- active | revoked | fired
    fired_at        REAL
);
CREATE INDEX IF NOT EXISTS lessons_by_group
    ON lessons(group_id, status, importance DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS lessons_due
    ON lessons(status, trigger_at);
CREATE INDEX IF NOT EXISTS lessons_group_subject
    ON lessons(group_id, subject_user_id, expires_at);
"""

# Columns we add to lessons after the table was first created (v1) — applied
# idempotently on every startup via _add_missing_columns(). Kept in code (not
# SQL) so we can run the same logic against pre-existing databases.
_LESSONS_NEW_COLUMNS = (
    ("kind", "TEXT NOT NULL DEFAULT 'rule'"),
    ("subject_user_id", "INTEGER"),
    ("content", "TEXT NOT NULL DEFAULT ''"),
    ("tags", "TEXT NOT NULL DEFAULT ''"),
    ("trigger_at", "REAL"),
    ("recurrence", "TEXT"),
    ("expires_at", "REAL"),
    ("fired_at", "REAL"),
)


class Storage:
    """Singleton-like async storage wrapper."""

    _instance: "Storage | None" = None
    # Lazy asyncio.Lock — created on first `get()` call so it's bound to the
    # current event loop. Reset alongside `_instance` between tests.
    _init_lock: Optional[asyncio.Lock] = None

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    @classmethod
    async def get(cls) -> "Storage":
        # Fast path: fully initialized.
        if cls._instance is not None and cls._instance._conn is not None:
            return cls._instance
        # Slow path: serialize concurrent first-callers so we don't open two
        # connections or hand back a half-initialised instance.
        if cls._init_lock is None:
            cls._init_lock = asyncio.Lock()
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = Storage(DB_FILE)
            if cls._instance._conn is None:
                await cls._instance._init()
        return cls._instance

    @classmethod
    async def reset_for_tests(cls, path: Path) -> "Storage":
        if cls._instance is not None:
            await cls._instance.close()
        cls._instance = Storage(path)
        cls._init_lock = None
        await cls._instance._init(migrate=False)
        return cls._instance

    async def _init(self, *, migrate: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        await self._add_missing_lesson_columns()
        if migrate:
            await self._migrate_legacy_json()
            await self._migrate_memories_into_lessons()
            # Now that everything is unified into `lessons`, drop the
            # `memories` table so it stops showing up as residue in any
            # ad-hoc SQL someone runs against the DB.
            try:
                await self._conn.execute("DROP TABLE IF EXISTS memories")
                await self._conn.commit()
            except aiosqlite.OperationalError as e:
                log.warning("dropping legacy memories table failed: %s", e)

    async def _add_missing_lesson_columns(self) -> None:
        """Idempotent ALTER TABLE for any lesson columns added after the
        initial v1 schema. Safe to run repeatedly."""
        assert self._conn is not None
        async with self._conn.execute("PRAGMA table_info(lessons)") as cur:
            rows = await cur.fetchall()
        existing = {str(r[1]) for r in rows}
        for col, decl in _LESSONS_NEW_COLUMNS:
            if col in existing:
                continue
            try:
                await self._conn.execute(
                    f"ALTER TABLE lessons ADD COLUMN {col} {decl}"
                )
            except aiosqlite.OperationalError as e:
                log.warning("lessons ALTER skipped (%s): %s", col, e)
        # Backfill: copy rule → content for any pre-unify rows that have one
        # but not the other. Two-direction so legacy reads against `rule`
        # still work.
        try:
            await self._conn.execute(
                "UPDATE lessons SET content=rule "
                "WHERE (content='' OR content IS NULL) AND rule<>''"
            )
            await self._conn.execute(
                "UPDATE lessons SET rule=content "
                "WHERE (rule='' OR rule IS NULL) AND content<>''"
            )
        except aiosqlite.OperationalError:
            pass
        await self._conn.commit()

    async def _migrate_memories_into_lessons(self) -> None:
        """One-time copy of pre-unify `memories` rows into the unified
        `lessons` table. Idempotent — uses a sentinel tag to avoid re-copy."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ) as cur:
            if not await cur.fetchone():
                return
        async with self._conn.execute(
            "SELECT id,group_id,subject_user_id,content,importance,tags,"
            " trigger_at,recurrence,expires_at,created_at,status,fired_at,"
            " source_text,source_nickname "
            "FROM memories"
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return
        copied = 0
        for r in rows:
            (
                old_id, group_id, subject_user_id, content, importance, tags,
                trigger_at, recurrence, expires_at, created_at, status,
                fired_at, source_text, source_nickname,
            ) = r
            tag_str = (tags or "")
            sentinel = f"_migrated_from_memories:{old_id}"
            if sentinel in tag_str:
                continue
            tag_str = (tag_str + " " + sentinel).strip()
            kind = "reminder" if trigger_at is not None else (
                "fact" if subject_user_id is not None else "agreement"
            )
            # Best-effort: source_user_id is unknown for legacy rows; use 0.
            try:
                await self._conn.execute(
                    "INSERT INTO lessons("
                    "  group_id,kind,subject_user_id,content,rule,importance,"
                    "  tags,trigger_at,recurrence,expires_at,source_user_id,"
                    "  source_text,created_at,status,fired_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        group_id, kind, subject_user_id, content, content,
                        float(importance or 0.6), tag_str, trigger_at,
                        recurrence, expires_at, 0,
                        (source_text or "")[:500], float(created_at or 0.0),
                        str(status or "active"), fired_at,
                    ),
                )
                copied += 1
            except aiosqlite.OperationalError as e:
                log.warning("memories→lessons skip id=%s: %s", old_id, e)
        await self._conn.commit()
        if copied:
            log.info("migrated %d rows from memories → lessons", copied)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ---------- migration ----------
    async def _migrate_legacy_json(self) -> None:
        # memory.json → memory table
        if MEMORY_FILE.exists():
            try:
                raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("memory.json migrate skip: %s", e)
            else:
                async with self._lock:
                    assert self._conn is not None
                    n = 0
                    for key, items in raw.items():
                        try:
                            gid_s, uid_s = key.split(":", 1)
                            gid, uid = int(gid_s), int(uid_s)
                        except Exception:
                            continue
                        for seq, (role, content) in enumerate(items):
                            await self._conn.execute(
                                "INSERT OR IGNORE INTO memory(group_id,user_id,seq,role,content)"
                                " VALUES(?,?,?,?,?)",
                                (gid, uid, seq, role, content),
                            )
                            n += 1
                    await self._conn.commit()
                if n:
                    log.info("migrated %d memory rows from memory.json", n)
                MEMORY_FILE.rename(MEMORY_FILE.with_suffix(".json.migrated"))

        # quota.json → quota table
        if QUOTA_FILE.exists():
            try:
                raw = json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("quota.json migrate skip: %s", e)
            else:
                day = raw.get("date") or date.today().isoformat()
                async with self._lock:
                    assert self._conn is not None
                    for scope_key, table in (("group", raw.get("group", {})),
                                              ("user", raw.get("user", {}))):
                        for k, routes in table.items():
                            for route, count in routes.items():
                                await self._conn.execute(
                                    "INSERT OR REPLACE INTO quota(day,scope,key,route,count)"
                                    " VALUES(?,?,?,?,?)",
                                    (day, scope_key, k, route, int(count)),
                                )
                    await self._conn.commit()
                log.info("migrated quota.json (date=%s)", day)
                QUOTA_FILE.rename(QUOTA_FILE.with_suffix(".json.migrated"))

    # ---------- memory ----------
    async def memory_get(self, group_id: int, user_id: int, limit: int) -> List[Tuple[str, str]]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT role,content FROM memory WHERE group_id=? AND user_id=? "
            "ORDER BY seq DESC LIMIT ?",
            (group_id, user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [(r[0], r[1]) for r in reversed(rows)]

    async def memory_append(
        self,
        group_id: int,
        user_id: int,
        role: str,
        content: str,
        max_rows: int,
    ) -> None:
        assert self._conn is not None
        async with self._lock:
            async with self._conn.execute(
                "SELECT COALESCE(MAX(seq),-1)+1 FROM memory WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            ) as cur:
                row = await cur.fetchone()
                seq = int(row[0]) if row else 0
            await self._conn.execute(
                "INSERT INTO memory(group_id,user_id,seq,role,content) VALUES(?,?,?,?,?)",
                (group_id, user_id, seq, role, content),
            )
            # Trim to last max_rows turns.
            await self._conn.execute(
                "DELETE FROM memory WHERE group_id=? AND user_id=? AND seq <= ?",
                (group_id, user_id, seq - max_rows),
            )
            await self._conn.commit()

    async def memory_reset(self, group_id: int, user_id: int) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "DELETE FROM memory WHERE group_id=? AND user_id=?",
                (group_id, user_id),
            )
            await self._conn.commit()

    async def memory_reset_group(self, group_id: int) -> int:
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM memory WHERE group_id=?", (group_id,)
            )
            await self._conn.commit()
            return cur.rowcount or 0

    # ---------- quota ----------
    async def quota_count(self, day: str, scope: str, key: str, route: str) -> int:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT count FROM quota WHERE day=? AND scope=? AND key=? AND route=?",
            (day, scope, key, route),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def quota_bump(self, day: str, scope: str, key: str, route: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO quota(day,scope,key,route,count) VALUES(?,?,?,?,1) "
                "ON CONFLICT(day,scope,key,route) DO UPDATE SET count=count+1",
                (day, scope, key, route),
            )
            await self._conn.commit()

    async def quota_dump(self, day: str) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return {scope: {key: {route: count}}} for one day."""
        assert self._conn is not None
        out: Dict[str, Dict[str, Dict[str, int]]] = {"group": {}, "user": {}}
        async with self._conn.execute(
            "SELECT scope,key,route,count FROM quota WHERE day=?", (day,)
        ) as cur:
            async for scope, key, route, count in cur:
                out.setdefault(scope, {}).setdefault(key, {})[route] = int(count)
        return out

    async def quota_reset_day(self, day: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute("DELETE FROM quota WHERE day=?", (day,))
            await self._conn.commit()

    # ---------- allow-list ----------
    async def groups_list(self) -> List[int]:
        assert self._conn is not None
        async with self._conn.execute("SELECT group_id FROM groups ORDER BY group_id") as cur:
            rows = await cur.fetchall()
        return [int(r[0]) for r in rows]

    async def groups_add(self, group_id: int) -> bool:
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "INSERT OR IGNORE INTO groups(group_id, added_at) VALUES(?, datetime('now'))",
                (group_id,),
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0

    async def groups_remove(self, group_id: int) -> bool:
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM groups WHERE group_id=?", (group_id,)
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0

    # ---------- per-group pause ----------
    async def group_pause_is_set(self, group_id: int) -> bool:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT 1 FROM group_pause WHERE group_id=?", (group_id,),
        ) as cur:
            return (await cur.fetchone()) is not None

    async def group_pause_set(self, group_id: int, by_user_id: int) -> bool:
        """Pause this group. Returns True if a new row was inserted, False if
        the group was already paused."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "INSERT OR IGNORE INTO group_pause(group_id, paused_at, paused_by)"
                " VALUES(?, datetime('now'), ?)",
                (group_id, by_user_id),
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0

    async def group_pause_clear(self, group_id: int) -> bool:
        """Resume this group. Returns True if a row was actually removed."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM group_pause WHERE group_id=?", (group_id,),
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0

    async def group_pause_list(self) -> List[int]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT group_id FROM group_pause ORDER BY group_id",
        ) as cur:
            rows = await cur.fetchall()
        return [int(r[0]) for r in rows]

    # ---------- group memory ----------
    async def group_memory_append(
        self,
        group_id: int,
        ts: float,
        user_id: int,
        nickname: str,
        text: str,
        max_rows: int,
    ) -> int:
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "INSERT INTO group_memory(group_id,ts,user_id,nickname,text)"
                " VALUES(?,?,?,?,?)",
                (group_id, ts, user_id, nickname, text),
            )
            rowid = cur.lastrowid
            # Prune oldest rows over the per-group cap.
            await self._conn.execute(
                "DELETE FROM group_memory WHERE group_id=? AND id NOT IN ("
                "  SELECT id FROM group_memory WHERE group_id=? "
                "  ORDER BY ts DESC LIMIT ?)",
                (group_id, group_id, max_rows),
            )
            await self._conn.commit()
            return int(rowid or 0)

    async def group_memory_update_text(self, row_id: int, new_text: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "UPDATE group_memory SET text=? WHERE id=?", (new_text, row_id),
            )
            await self._conn.commit()

    async def group_memory_recent(
        self, group_id: int, limit: int
    ) -> List[Tuple[float, int, str, str]]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT ts,user_id,nickname,text FROM group_memory "
            "WHERE group_id=? ORDER BY ts DESC LIMIT ?",
            (group_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [(float(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in reversed(rows)]

    async def group_memory_since(
        self, group_id: int, since_ts: float
    ) -> List[Tuple[float, int, str, str]]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT ts,user_id,nickname,text FROM group_memory "
            "WHERE group_id=? AND ts>=? ORDER BY ts ASC",
            (group_id, since_ts),
        ) as cur:
            rows = await cur.fetchall()
        return [(float(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]

    async def group_memory_reset(self, group_id: int) -> int:
        """Wipe the rolling chat log for one group. Returns rows deleted."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM group_memory WHERE group_id=?", (group_id,),
            )
            await self._conn.commit()
            return cur.rowcount or 0

    # ---------- daily recaps (long-term memory) ----------
    async def daily_recap_upsert(
        self, group_id: int, day: str, summary: str
    ) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO daily_recaps(group_id,day,summary,created_at) "
                "VALUES(?,?,?, datetime('now')) "
                "ON CONFLICT(group_id, day) DO UPDATE SET "
                "  summary=excluded.summary, created_at=excluded.created_at",
                (group_id, day, summary[:2000]),
            )
            await self._conn.commit()

    async def daily_recap_get(self, group_id: int, day: str) -> Optional[str]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT summary FROM daily_recaps WHERE group_id=? AND day=?",
            (group_id, day),
        ) as cur:
            row = await cur.fetchone()
        return str(row[0]) if row else None

    async def daily_recap_recent(
        self, group_id: int, limit: int
    ) -> List[Tuple[str, str]]:
        """Return [(day, summary), ...] most-recent first."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT day, summary FROM daily_recaps WHERE group_id=? "
            "ORDER BY day DESC LIMIT ?",
            (group_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    async def daily_recap_search(
        self, group_id: int, keyword: str, limit: int = 5
    ) -> List[Tuple[str, str]]:
        assert self._conn is not None
        pattern = f"%{keyword}%"
        async with self._conn.execute(
            "SELECT day, summary FROM daily_recaps WHERE group_id=? "
            "AND summary LIKE ? ORDER BY day DESC LIMIT ?",
            (group_id, pattern, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    async def daily_recap_prune(self, keep_days: int) -> int:
        """Drop recaps older than `keep_days`. Returns rows deleted."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM daily_recaps "
                "WHERE day < date('now', ?)",
                (f"-{keep_days} days",),
            )
            await self._conn.commit()
            return cur.rowcount or 0

    async def daily_recap_reset(self, group_id: int) -> int:
        """Wipe every daily recap for one group. Returns rows deleted."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM daily_recaps WHERE group_id=?", (group_id,),
            )
            await self._conn.commit()
            return cur.rowcount or 0

    # ---------- lessons (unified rules + facts + agreements + reminders) ----------
    async def lesson_insert(
        self,
        *,
        group_id: int,
        kind: str,
        subject_user_id: Optional[int],
        content: str,
        importance: float,
        tags: str,
        trigger_at: Optional[float],
        recurrence: Optional[str],
        expires_at: Optional[float],
        source_user_id: int,
        source_text: str,
        created_at: float,
    ) -> int:
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "INSERT INTO lessons("
                "  group_id,kind,subject_user_id,content,rule,importance,tags,"
                "  trigger_at,recurrence,expires_at,source_user_id,source_text,"
                "  created_at,status"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, 'active')",
                (
                    group_id, kind, subject_user_id, content[:500], content[:500],
                    float(importance), tags[:200], trigger_at, recurrence,
                    expires_at, source_user_id, source_text[:500], created_at,
                ),
            )
            await self._conn.commit()
            return int(cur.lastrowid or 0)

    async def lesson_active_for_user(
        self, group_id: int, subject_user_id: int, now_ts: float, limit: int = 12,
    ) -> List[Tuple[int, str, Optional[int], str, float, str, Optional[float]]]:
        """Active rows scoped to a user (personal first, then group-wide).
        Returns (id, kind, subject_user_id, content, importance, tags, trigger_at)."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT id,kind,subject_user_id,content,importance,tags,trigger_at "
            "FROM lessons WHERE group_id=? "
            "AND status='active' "
            "AND (subject_user_id IS NULL OR subject_user_id=?) "
            "AND (expires_at IS NULL OR expires_at>?) "
            "ORDER BY "
            "  CASE kind WHEN 'rule' THEN 0 ELSE 1 END ASC, "
            "  (subject_user_id IS NULL) ASC, "
            "  importance DESC, "
            "  created_at DESC "
            "LIMIT ?",
            (group_id, subject_user_id, now_ts, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            (int(r[0]), str(r[1]),
             int(r[2]) if r[2] is not None else None,
             str(r[3]), float(r[4]), str(r[5]),
             float(r[6]) if r[6] is not None else None)
            for r in rows
        ]

    async def lesson_list(
        self, group_id: int, limit: int = 50,
    ) -> List[Tuple[int, str, str, float, str, Optional[float], Optional[str]]]:
        """All ACTIVE lessons in this group, for /admin.

        Returns (id, kind, content, importance, status, trigger_at, recurrence).
        Filters out stale 'revoked'/'fired'/'cancelled' rows that may linger
        from a pre-hard-delete DB; cancel now physically removes rows, so this
        filter is just belt-and-suspenders.
        """
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT id,kind,content,importance,status,trigger_at,recurrence "
            "FROM lessons WHERE group_id=? AND status='active' "
            "ORDER BY created_at DESC LIMIT ?",
            (group_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            (int(r[0]), str(r[1]), str(r[2]), float(r[3]), str(r[4]),
             float(r[5]) if r[5] is not None else None,
             str(r[6]) if r[6] is not None else None)
            for r in rows
        ]

    async def lesson_list_pending(
        self, group_id: int, subject_user_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Tuple[int, str, Optional[int], str, Optional[float], Optional[str]]]:
        """Active reminders for /remember. Returns
        (id, kind, subject_user_id, content, trigger_at, recurrence)."""
        assert self._conn is not None
        if subject_user_id is None:
            sql = (
                "SELECT id,kind,subject_user_id,content,trigger_at,recurrence "
                "FROM lessons WHERE group_id=? AND status='active' "
                "ORDER BY trigger_at IS NULL, trigger_at ASC LIMIT ?"
            )
            params: Tuple = (group_id, limit)
        else:
            sql = (
                "SELECT id,kind,subject_user_id,content,trigger_at,recurrence "
                "FROM lessons WHERE group_id=? AND status='active' "
                "AND (subject_user_id IS NULL OR subject_user_id=?) "
                "ORDER BY trigger_at IS NULL, trigger_at ASC LIMIT ?"
            )
            params = (group_id, subject_user_id, limit)
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            (int(r[0]), str(r[1]),
             int(r[2]) if r[2] is not None else None,
             str(r[3]),
             float(r[4]) if r[4] is not None else None,
             str(r[5]) if r[5] is not None else None)
            for r in rows
        ]

    async def lesson_due(
        self, now_ts: float, limit: int = 50,
    ) -> List[Tuple[int, int, Optional[int], str, str, Optional[str]]]:
        """Active items whose trigger_at <= now. Returns
        (id, group_id, subject_user_id, content, kind, recurrence)."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT id,group_id,subject_user_id,content,kind,recurrence "
            "FROM lessons WHERE status='active' AND trigger_at IS NOT NULL "
            "AND trigger_at<=? ORDER BY trigger_at ASC LIMIT ?",
            (now_ts, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            (int(r[0]), int(r[1]),
             int(r[2]) if r[2] is not None else None,
             str(r[3]), str(r[4]),
             str(r[5]) if r[5] is not None else None)
            for r in rows
        ]

    async def lesson_mark_fired(
        self, lesson_id: int, fired_at: float,
        next_trigger: Optional[float] = None,
    ) -> None:
        """Recurring → reschedule with the next trigger.
        One-shot → hard-delete (no residue once it fires)."""
        assert self._conn is not None
        async with self._lock:
            if next_trigger is not None:
                await self._conn.execute(
                    "UPDATE lessons SET trigger_at=?, fired_at=? WHERE id=?",
                    (next_trigger, fired_at, lesson_id),
                )
            else:
                # One-shot reminder has done its job — drop the row entirely.
                # Rules/facts/agreements have trigger_at=NULL so they're
                # filtered out by the due-reminder query and never come here.
                await self._conn.execute(
                    "DELETE FROM lessons "
                    "WHERE id=? AND trigger_at IS NOT NULL",
                    (lesson_id,),
                )
            await self._conn.commit()

    # Cancel = HARD DELETE. The old design used `status='revoked'` so admin
    # views could show an audit trail, but in practice it read as "the bot
    # still kinda remembers what I told it to forget". Cancelled rows are
    # now physically gone — no surface, no residue, no risk that a downstream
    # query forgot to filter the status column.

    async def lesson_delete(self, lesson_id: int, group_id: int) -> bool:
        """Hard-delete one row. Returns True if a row was actually removed."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM lessons WHERE id=? AND group_id=?",
                (lesson_id, group_id),
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0

    async def lesson_delete_many(self, ids: List[int], group_id: int) -> int:
        """Hard-delete a list of rows scoped to one group. Returns count."""
        if not ids:
            return 0
        assert self._conn is not None
        qs = ",".join("?" * len(ids))
        async with self._lock:
            cur = await self._conn.execute(
                f"DELETE FROM lessons WHERE group_id=? AND id IN ({qs})",
                (group_id, *ids),
            )
            await self._conn.commit()
            return cur.rowcount or 0

    async def lesson_delete_all(
        self, group_id: int, kind: Optional[str] = None,
    ) -> int:
        """Wipe every lesson in this group. If `kind` is given (e.g. 'rule'),
        only that kind is deleted."""
        assert self._conn is not None
        async with self._lock:
            if kind is None:
                cur = await self._conn.execute(
                    "DELETE FROM lessons WHERE group_id=?",
                    (group_id,),
                )
            else:
                cur = await self._conn.execute(
                    "DELETE FROM lessons WHERE group_id=? AND kind=?",
                    (group_id, kind),
                )
            await self._conn.commit()
            return cur.rowcount or 0

    # Legacy aliases (kept so any direct callers / tests don't break). All
    # routes through to the hard-delete path now.
    async def lesson_revoke(self, lesson_id: int, group_id: int) -> bool:
        return await self.lesson_delete(lesson_id, group_id)

    async def lesson_revoke_many(self, ids: List[int], group_id: int) -> int:
        return await self.lesson_delete_many(ids, group_id)

    async def lesson_expire(self, now_ts: float) -> int:
        """Hard-delete rows past expires_at. Returns count."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "DELETE FROM lessons "
                "WHERE expires_at IS NOT NULL AND expires_at<=?",
                (now_ts,),
            )
            await self._conn.commit()
            return cur.rowcount or 0

    async def lesson_prune(self, keep_days: int) -> int:
        """No-op since cancel/fire/expire all hard-delete. Kept on the
        interface so the maintenance loop can call it unchanged."""
        return 0

    async def lesson_dedup_candidates(
        self, group_id: int, limit: int = 50,
    ) -> List[Tuple[int, str, Optional[int], str, str, float]]:
        """Recent items in this group, for the LLM dedup pass.
        Returns (id, kind, subject_user_id, content, tags, created_at)."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT id,kind,subject_user_id,content,tags,created_at "
            "FROM lessons WHERE group_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (group_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            (int(r[0]), str(r[1]),
             int(r[2]) if r[2] is not None else None,
             str(r[3]), str(r[4]), float(r[5]))
            for r in rows
        ]

    # ---------- daily report bookkeeping ----------
    async def report_mark_sent(self, day: str) -> bool:
        """Return True if we successfully claimed this day (i.e. report not yet sent)."""
        assert self._conn is not None
        async with self._lock:
            cur = await self._conn.execute(
                "INSERT OR IGNORE INTO daily_report(day) VALUES(?)", (day,)
            )
            await self._conn.commit()
            return (cur.rowcount or 0) > 0
