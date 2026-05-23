"""OneBot v11 WebSocket client — supports forward AND reverse modes.

forward  : we *connect to* the adapter at CONFIG.onebot_ws_url.
reverse  : we run a small WS server; the adapter connects to us. NapCat default.

Both modes share the same RPC echo protocol and `send_group_msg` helpers.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from bot.logger import get_logger
from config import CONFIG

log = get_logger(__name__)

EventHandler = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class WsStatus:
    mode: str
    connected: bool
    connected_at: Optional[float]
    last_event_at: Optional[float]
    last_heartbeat_at: Optional[float]
    disconnect_count: int
    last_disconnect_at: Optional[float]
    last_disconnect_reason: str


class OneBotClient:
    """Bidirectional OneBot v11 transport.

    Holds a *single* active WS at a time (whichever side connected last). For
    reverse mode that's the adapter; for forward mode that's our outbound conn.
    """

    def __init__(self, on_event: EventHandler) -> None:
        self._on_event = on_event
        self._ws: Optional[Any] = None
        self._send_lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._stop = asyncio.Event()
        self._server: Optional[Any] = None
        # Health stats (unix timestamps, time.time()).
        self._connected_at: Optional[float] = None
        self._last_event_at: Optional[float] = None
        self._last_heartbeat_at: Optional[float] = None
        self._disconnect_count: int = 0
        self._last_disconnect_at: Optional[float] = None
        self._last_disconnect_reason: str = ""

    def status(self) -> WsStatus:
        return WsStatus(
            mode=CONFIG.onebot_mode,
            connected=self._ws is not None,
            connected_at=self._connected_at,
            last_event_at=self._last_event_at,
            last_heartbeat_at=self._last_heartbeat_at,
            disconnect_count=self._disconnect_count,
            last_disconnect_at=self._last_disconnect_at,
            last_disconnect_reason=self._last_disconnect_reason,
        )

    def _mark_connected(self) -> None:
        now = time.time()
        self._connected_at = now
        self._last_event_at = now

    def _mark_disconnected(self, reason: str) -> None:
        self._disconnect_count += 1
        self._last_disconnect_at = time.time()
        self._last_disconnect_reason = reason[:200]

    # ---------- public ----------
    async def run(self) -> None:
        if CONFIG.onebot_mode == "reverse":
            await self._run_reverse()
        else:
            await self._run_forward()

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

    async def send_group_msg(self, group_id: int, message: str) -> Dict[str, Any]:
        return await self._call_api(
            "send_group_msg",
            {"group_id": int(group_id), "message": message, "auto_escape": False},
        )

    async def send_group_image(self, group_id: int, image: str) -> Dict[str, Any]:
        cq = f"[CQ:image,file={image}]"
        return await self.send_group_msg(group_id, cq)

    async def get_msg(self, msg_id: str) -> Optional[Dict[str, Any]]:
        """OneBot get_msg → {message, sender, ...} or None on failure."""
        resp = await self._call_api("get_msg", {"message_id": msg_id})
        if (resp.get("status") or "").lower() not in ("ok", "async"):
            return None
        return resp.get("data") or None

    async def get_group_file_url(
        self, group_id: int, file_id: str, busid: int = 0,
    ) -> Optional[str]:
        """Resolve a group-file id to a fetchable URL. Tries common OneBot v11
        APIs (NapCat / go-cqhttp variants) in turn — first hit wins."""
        for action, params in (
            ("get_group_file_url", {
                "group_id": int(group_id), "file_id": file_id, "busid": busid,
            }),
            ("get_file", {"file_id": file_id}),
        ):
            resp = await self._call_api(action, params)
            status = (resp.get("status") or "").lower()
            if status not in ("ok", "async"):
                continue
            data = resp.get("data") or {}
            url = data.get("url") or data.get("file_url") or data.get("file")
            if url:
                return str(url)
        return None

    # ---------- forward (we dial out) ----------
    async def _run_forward(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                headers = {}
                if CONFIG.onebot_access_token:
                    headers["Authorization"] = f"Bearer {CONFIG.onebot_access_token}"
                log.info("connecting (forward) to OneBot WS at %s", CONFIG.onebot_ws_url)
                async with websockets.connect(
                    CONFIG.onebot_ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    self._mark_connected()
                    log.info("OneBot WS connected (forward)")
                    await self._read_loop(ws)
                    backoff = 1.0
                    self._mark_disconnected("clean_close")
            except (OSError, ConnectionClosed) as e:
                log.warning("forward WS lost: %s — reconnect in %.1fs", e, backoff)
                self._mark_disconnected(f"{type(e).__name__}: {e}")
            except Exception as e:
                log.exception("unexpected forward WS error — reconnect in %.1fs", backoff)
                self._mark_disconnected(f"{type(e).__name__}: {e}")
            finally:
                self._ws = None
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    # ---------- reverse (adapter dials us) ----------
    async def _run_reverse(self) -> None:
        host = CONFIG.onebot_reverse_host
        port = CONFIG.onebot_reverse_port
        log.info("starting reverse WS server on ws://%s:%d%s",
                 host, port, CONFIG.onebot_reverse_path)
        self._server = await websockets.serve(
            self._reverse_handler,
            host=host,
            port=port,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        )
        try:
            await self._stop.wait()
        finally:
            self._server.close()
            await self._server.wait_closed()

    async def _reverse_handler(self, ws) -> None:
        # path-prefix check (be lenient — some adapters append /api/, /event/)
        path = getattr(ws, "request", None)
        path_str = getattr(path, "path", "") if path is not None else ""
        if CONFIG.onebot_reverse_path and not path_str.startswith(CONFIG.onebot_reverse_path):
            log.warning("rejecting reverse-WS connection on unexpected path %s", path_str)
            await ws.close(code=1008, reason="bad path")
            return
        if CONFIG.onebot_access_token:
            auth = ws.request.headers.get("Authorization", "") if ws.request else ""
            expected = f"Bearer {CONFIG.onebot_access_token}"
            if auth != expected:
                log.warning("rejecting reverse-WS connection: bad/missing access token")
                await ws.close(code=1008, reason="unauthorized")
                return

        log.info("reverse WS adapter connected")
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = ws
        self._mark_connected()
        try:
            await self._read_loop(ws)
            self._mark_disconnected("clean_close")
        except ConnectionClosed as e:
            log.info("reverse WS adapter disconnected: %s", e)
            self._mark_disconnected(f"ConnectionClosed: {e.code} {e.reason}")
        except Exception as e:
            log.exception("reverse WS read loop error")
            self._mark_disconnected(f"{type(e).__name__}: {e}")
        finally:
            if self._ws is ws:
                self._ws = None

    # ---------- shared ----------
    async def _read_loop(self, ws) -> None:
        async for raw in ws:
            self._last_event_at = time.time()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("non-JSON WS frame ignored")
                continue
            if "echo" in data and data["echo"] in self._pending:
                fut = self._pending.pop(data["echo"])
                if not fut.done():
                    fut.set_result(data)
                continue
            # OneBot v11: meta_event/heartbeat — track so /admin ping can show it.
            if data.get("post_type") == "meta_event" and data.get("meta_event_type") == "heartbeat":
                self._last_heartbeat_at = time.time()
                continue
            if data.get("post_type"):
                asyncio.create_task(self._safe_dispatch(data))

    async def _safe_dispatch(self, event: Dict[str, Any]) -> None:
        try:
            await self._on_event(event)
        except Exception:
            log.exception("handler raised")

    async def _call_api(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ws = self._ws
        if ws is None:
            log.warning("API call %s dropped: no WS connected", action)
            return {"status": "failed", "retcode": -1, "msg": "no_ws"}
        echo = uuid.uuid4().hex
        fut: asyncio.Future[Dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[echo] = fut
        payload = json.dumps({"action": action, "params": params, "echo": echo})
        async with self._send_lock:
            await ws.send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(echo, None)
            log.warning("API call %s timed out", action)
            return {"status": "failed", "retcode": -1, "msg": "timeout"}
