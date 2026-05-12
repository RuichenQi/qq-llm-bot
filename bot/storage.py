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
"""


class Storage:
    """Singleton-like async storage wrapper."""

    _instance: "Storage | None" = None

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    @classmethod
    async def get(cls) -> "Storage":
        if cls._instance is None:
            cls._instance = Storage(DB_FILE)
            await cls._instance._init()
        return cls._instance

    @classmethod
    async def reset_for_tests(cls, path: Path) -> "Storage":
        if cls._instance is not None:
            await cls._instance.close()
        cls._instance = Storage(path)
        await cls._instance._init(migrate=False)
        return cls._instance

    async def _init(self, *, migrate: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        if migrate:
            await self._migrate_legacy_json()

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
