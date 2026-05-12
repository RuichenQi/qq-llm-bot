"""Daily quota tracking, backed by SQLite (bot.storage)."""
from __future__ import annotations

from datetime import date
from typing import Dict, Tuple

from bot.logger import get_logger
from bot.storage import Storage
from config import CONFIG

log = get_logger(__name__)

ROUTE_GROUP_LIMITS = {
    "openai_text": "openai_text_group",
    "openai_image": "openai_image_group",
    "openai_image_edit": "openai_image_edit_group",
    "openai_vision": "openai_vision_group",
}
ROUTE_USER_LIMITS = {
    "openai_text": "openai_text_user",
    "openai_image": "openai_image_user",
    "openai_image_edit": "openai_image_edit_user",
    "openai_vision": "openai_vision_user",
}


def _today() -> str:
    return date.today().isoformat()


class Quota:
    def _limit_for(self, route: str, scope: str) -> int:
        attr = (ROUTE_GROUP_LIMITS if scope == "group" else ROUTE_USER_LIMITS).get(route)
        if not attr:
            return 10**9
        return int(getattr(CONFIG.limits, attr))

    async def check(self, route: str, group_id: int, user_id: int) -> Tuple[bool, str]:
        if user_id in CONFIG.superusers:
            return True, ""
        store = await Storage.get()
        day = _today()
        g_used = await store.quota_count(day, "group", str(group_id), route)
        u_used = await store.quota_count(day, "user", f"{group_id}:{user_id}", route)
        g_lim = self._limit_for(route, "group")
        u_lim = self._limit_for(route, "user")
        if g_used >= g_lim:
            return False, f"group quota for {route} reached ({g_used}/{g_lim})"
        if u_used >= u_lim:
            return False, f"user quota for {route} reached ({u_used}/{u_lim})"
        return True, ""

    async def consume(self, route: str, group_id: int, user_id: int) -> None:
        if user_id in CONFIG.superusers:
            return
        store = await Storage.get()
        day = _today()
        await store.quota_bump(day, "group", str(group_id), route)
        await store.quota_bump(day, "user", f"{group_id}:{user_id}", route)

    async def admin_reset(self) -> None:
        store = await Storage.get()
        await store.quota_reset_day(_today())

    async def snapshot(self, group_id: int, user_id: int) -> Dict[str, Dict[str, str]]:
        store = await Storage.get()
        day = _today()
        out: Dict[str, Dict[str, str]] = {}
        for route in ROUTE_GROUP_LIMITS:
            g_used = await store.quota_count(day, "group", str(group_id), route)
            u_used = await store.quota_count(day, "user", f"{group_id}:{user_id}", route)
            out[route] = {
                "group": f"{g_used}/{self._limit_for(route, 'group')}",
                "user":  f"{u_used}/{self._limit_for(route, 'user')}",
            }
        return out

    async def dump_today(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        store = await Storage.get()
        return await store.quota_dump(_today())
