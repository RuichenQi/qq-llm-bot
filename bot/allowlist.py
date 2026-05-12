"""Runtime allow-list = env ALLOWED_GROUPS ∪ persisted SQLite rows.

env entries are immutable from a running session; the SQLite layer is the
admin-mutable surface.
"""
from __future__ import annotations

from typing import Set

from bot.storage import Storage
from config import CONFIG


async def is_allowed(group_id: int) -> bool:
    if group_id in CONFIG.allowed_groups:
        return True
    store = await Storage.get()
    return group_id in set(await store.groups_list())


async def all_allowed_groups() -> Set[int]:
    store = await Storage.get()
    return set(CONFIG.allowed_groups) | set(await store.groups_list())


async def add(group_id: int) -> bool:
    store = await Storage.get()
    return await store.groups_add(group_id)


async def remove(group_id: int) -> bool:
    if group_id in CONFIG.allowed_groups:
        return False  # env-pinned, can't remove at runtime
    store = await Storage.get()
    return await store.groups_remove(group_id)
