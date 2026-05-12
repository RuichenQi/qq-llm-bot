"""Per-(group, user) short conversation memory, backed by SQLite (bot.storage).

API is identical to the v1 JSON-backed implementation.
"""
from __future__ import annotations

from typing import List, Tuple

from bot.logger import get_logger
from bot.storage import Storage
from config import CONFIG

log = get_logger(__name__)


class Memory:
    def __init__(self, max_turns: int | None = None) -> None:
        # max_turns counts USER+ASSISTANT pairs, so we store 2*max_turns rows.
        self._max_rows = (max_turns or CONFIG.limits.max_history_turns) * 2

    async def get(self, group_id: int, user_id: int) -> List[Tuple[str, str]]:
        store = await Storage.get()
        return await store.memory_get(group_id, user_id, self._max_rows)

    async def append(
        self, group_id: int, user_id: int, role: str, content: str
    ) -> None:
        store = await Storage.get()
        await store.memory_append(group_id, user_id, role, content, self._max_rows)

    async def reset(self, group_id: int, user_id: int) -> None:
        store = await Storage.get()
        await store.memory_reset(group_id, user_id)

    async def admin_reset_group(self, group_id: int) -> int:
        store = await Storage.get()
        return await store.memory_reset_group(group_id)
