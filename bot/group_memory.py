"""Group-wide message memory.

Unlike `Memory` (which is per-`(group, user)` conversation), this keeps a
single rolling log of *all* messages in a group — what everyone said, in
order. Used to give the bot context when it replies (so it can react to the
broader chat, not just one user's conversation with it) and for `/recap`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from bot.storage import Storage
from config import CONFIG


@dataclass
class GroupMsg:
    ts: float
    user_id: int
    nickname: str
    text: str


class GroupMemory:
    async def append(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
    ) -> int:
        """Insert a row and return its sqlite rowid (0 if skipped)."""
        text = (text or "").strip()
        if not text:
            return 0
        store = await Storage.get()
        return await store.group_memory_append(
            group_id,
            time.time(),
            user_id,
            (nickname or f"u{user_id}")[:32],
            text[:500],
            CONFIG.group_memory_max,
        )

    async def update_text(self, row_id: int, new_text: str) -> None:
        if row_id <= 0:
            return
        store = await Storage.get()
        await store.group_memory_update_text(row_id, new_text[:500])

    async def recent(self, group_id: int, limit: Optional[int] = None) -> List[GroupMsg]:
        store = await Storage.get()
        rows = await store.group_memory_recent(
            group_id, limit or CONFIG.group_context_turns,
        )
        return [GroupMsg(ts=r[0], user_id=r[1], nickname=r[2], text=r[3]) for r in rows]

    async def since(self, group_id: int, since_ts: float) -> List[GroupMsg]:
        store = await Storage.get()
        rows = await store.group_memory_since(group_id, since_ts)
        return [GroupMsg(ts=r[0], user_id=r[1], nickname=r[2], text=r[3]) for r in rows]
