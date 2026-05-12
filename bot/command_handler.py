"""Orchestrates incoming messages: filter -> route -> provider -> reply."""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from bot import allowlist
from bot.image_utils import downscale_to_max, to_data_uri
from bot.logger import get_logger
from bot.memory import Memory
from bot.message_parser import ParsedMessage, chunk_text
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from config import CONFIG, IMAGE_DIR
from providers.base import ChatMessage, ImageReply, ProviderError
from providers.deepseek import DeepSeekProvider
from providers.openai_provider import OpenAIProvider

log = get_logger(__name__)

SendText = Callable[[int, str], Awaitable[None]]
SendImage = Callable[[int, str], Awaitable[None]]
# Fetch the *text* of the message referenced by [CQ:reply,id=...]. None if unavailable.
FetchReplyText = Callable[[str], Awaitable[Optional[str]]]
# Returns a `bot.onebot_client.WsStatus` (kept loose-typed to avoid the import cycle).
HealthFn = Callable[[], object]

QUOTA_EXCEEDED_MSG = "今天这个功能的额度用完了，请明天再试吧~"
RATE_LIMITED_MSG = "你发得太快啦，先休息一下吧~"
REJECT_MSG = "这个请求我没法处理~"
ERROR_MSG = "出了点小问题，请稍后再试 ({err})"

def _fmt_ago(ts: Optional[float], *, now: Optional[float] = None) -> str:
    if ts is None:
        return "从未"
    now = now if now is not None else time.time()
    delta = max(0, int(now - ts))
    if delta < 60:
        return f"{delta} 秒前"
    if delta < 3600:
        return f"{delta // 60} 分 {delta % 60} 秒前"
    if delta < 86400:
        return f"{delta // 3600} 小时 {(delta % 3600) // 60} 分前"
    return f"{delta // 86400} 天前"


def format_ws_status(status: object, *, now: Optional[float] = None) -> str:
    """Render an onebot_client.WsStatus into a Chinese summary block.

    Loose-typed for testing — accepts any object with the expected attributes.
    """
    mode = getattr(status, "mode", "?")
    connected = bool(getattr(status, "connected", False))
    lines: list[str] = []
    lines.append(f"OneBot 连接状态：{'🟢 已连接' if connected else '🔴 未连接'}")
    lines.append(f"模式：{mode}")
    lines.append(f"已连接时长：{_fmt_ago(getattr(status, 'connected_at', None), now=now)}")
    lines.append(f"最近事件：{_fmt_ago(getattr(status, 'last_event_at', None), now=now)}")
    lines.append(f"最近心跳：{_fmt_ago(getattr(status, 'last_heartbeat_at', None), now=now)}")
    lines.append(f"累计断开次数：{int(getattr(status, 'disconnect_count', 0))}")
    last_dc = getattr(status, "last_disconnect_at", None)
    if last_dc is not None:
        reason = getattr(status, "last_disconnect_reason", "") or "(无)"
        lines.append(f"上次断开：{_fmt_ago(last_dc, now=now)} — {reason}")
    return "\n".join(lines)


HELP_MSG = (
    "QQ 小助手指令：\n"
    "/ask <问题>      普通对话 (DeepSeek)\n"
    "/think <问题>    深度推理 (DeepSeek Reasoner)\n"
    "/gpt <问题>      使用 GPT (OpenAI，受额度限制)\n"
    "/image <描述>    生成图片 (OpenAI，受额度限制)\n"
    "/vision <问题>   分析最近一张图片 (OpenAI)\n"
    "/edit <修改指令> 编辑最近一张图片 (OpenAI)\n"
    "/reset           清空当前对话上下文\n"
    "/balance         查看今日额度\n"
    "/help            显示帮助\n"
    "直接发消息我会自动决定用哪个模型~"
)


@dataclass
class _ImageMemo:
    """Most recent image per (group, user). Bytes are kept on disk; we hold a
    path + monotonic timestamp + url so we can re-fetch if needed."""
    url: Optional[str] = None
    disk_path: Optional[Path] = None
    cached_at: float = 0.0  # monotonic seconds

    def is_fresh(self, ttl: int) -> bool:
        return (
            self.disk_path is not None
            and self.disk_path.exists()
            and (time.monotonic() - self.cached_at) < ttl
        )


@dataclass
class Handler:
    deepseek: DeepSeekProvider
    openai: Optional[OpenAIProvider]
    router: Router
    memory: Memory
    quota: Quota
    rate: RateLimiter
    send_text: SendText
    send_image: SendImage
    fetch_reply_text: Optional[FetchReplyText] = None  # injected from main
    health_status: Optional[HealthFn] = None           # injected from main
    _http: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=60.0))
    _last_image: Dict[Tuple[int, int], _ImageMemo] = field(default_factory=dict)

    # ---------- entry ----------
    async def handle(self, msg: ParsedMessage) -> None:
        if not await allowlist.is_allowed(msg.group_id):
            log.debug("ignoring message from non-allowed group %s", msg.group_id)
            return
        if msg.self_id == msg.user_id:
            return
        if not msg.text and not msg.has_image:
            return

        if msg.has_image:
            asyncio.create_task(
                self._cache_image_to_disk(msg.group_id, msg.user_id, msg.image_urls[0])
            )

        if not self._trigger_allows(msg):
            log.debug("trigger mode %s skipped: %r", CONFIG.trigger_mode, msg.text[:60])
            return
        msg = self._strip_trigger(msg)

        if not self.rate.check(msg.user_id):
            await self._reply(msg.group_id, RATE_LIMITED_MSG)
            return

        log.info(
            "msg from group=%s user=%s cmd=%s has_image=%s reply_to=%s text=%r",
            msg.group_id, msg.user_id, msg.command if msg.is_command else "-",
            msg.has_image, msg.reply_to_msg_id, msg.text[:120],
        )

        # Reply-segment: prepend the quoted message to the prompt.
        if msg.reply_to_msg_id and self.fetch_reply_text is not None:
            try:
                quoted = await self.fetch_reply_text(msg.reply_to_msg_id)
            except Exception as e:
                log.warning("fetch_reply_text failed: %s", e)
                quoted = None
            if quoted:
                msg.text = f"[被引用的消息]\n{quoted}\n\n[我的问题]\n{msg.text}".strip()

        try:
            if msg.is_command:
                await self._dispatch_command(msg)
            else:
                await self._dispatch_llm_route(msg)
        except ProviderError as e:
            log.exception("provider error")
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))
        except Exception as e:  # noqa: BLE001
            log.exception("unhandled error")
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))

    # ---------- trigger gate ----------
    def _trigger_allows(self, msg: ParsedMessage) -> bool:
        if msg.is_command:
            return True
        mode = CONFIG.trigger_mode
        if mode == "always":
            return True
        if mode == "mention":
            return msg.mentions(msg.self_id)
        if mode == "prefix":
            return msg.text.startswith(CONFIG.trigger_prefix)
        return True

    def _strip_trigger(self, msg: ParsedMessage) -> ParsedMessage:
        if msg.is_command:
            return msg
        if CONFIG.trigger_mode == "prefix" and msg.text.startswith(CONFIG.trigger_prefix):
            msg.text = msg.text[len(CONFIG.trigger_prefix):].strip()
        return msg

    # ---------- commands ----------
    async def _dispatch_command(self, msg: ParsedMessage) -> None:
        c = msg.command
        args = msg.command_args
        if c == "help":
            await self._reply(msg.group_id, HELP_MSG)
        elif c == "reset":
            await self.memory.reset(msg.group_id, msg.user_id)
            await self._reply(msg.group_id, "已清空你的对话记忆~")
        elif c == "balance":
            await self._reply(msg.group_id, await self._format_balance(msg))
        elif c == "ask":
            await self._run_deepseek_chat(msg, args or msg.text)
        elif c == "think":
            await self._run_deepseek_think(msg, args or msg.text)
        elif c == "gpt":
            await self._run_openai_text(msg, args or msg.text)
        elif c == "image":
            await self._run_openai_image(msg, args or msg.text)
        elif c == "vision":
            await self._run_openai_vision(msg, args or msg.text)
        elif c == "edit":
            await self._run_openai_image_edit(msg, args or msg.text)
        elif c == "admin":
            await self._run_admin(msg, args)
        else:
            await self._reply(msg.group_id, "未知指令，发送 /help 查看帮助")

    async def _run_admin(self, msg: ParsedMessage, args: str) -> None:
        if msg.user_id not in CONFIG.superusers:
            await self._reply(msg.group_id, "管理员命令仅限超级用户使用~")
            return
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        if sub in ("", "help"):
            await self._reply(
                msg.group_id,
                "/admin status               本群路由用量\n"
                "/admin usage                所有群今日用量\n"
                "/admin reset_quota          清空今日额度\n"
                "/admin reset_memory <uid>   清空指定用户对话记忆\n"
                "/admin reset_memory all     清空本群所有人记忆\n"
                "/admin allow_group <gid>    允许新的群\n"
                "/admin disallow_group <gid> 禁用某群 (env 中的群无法移除)\n"
                "/admin list_groups          显示所有允许的群\n"
                "/admin report               立即推送一次日报\n"
                "/admin ping                 OneBot 连接状态与最近心跳",
            )
            return
        if sub == "status":
            snap = await self.quota.snapshot(msg.group_id, msg.user_id)
            lines = [f"群 {msg.group_id} 今日用量:"]
            for route, cells in snap.items():
                lines.append(f"  {route}: 群 {cells['group']}")
            await self._reply(msg.group_id, "\n".join(lines))
            return
        if sub == "usage":
            await self._reply(msg.group_id, await self._format_global_usage())
            return
        if sub == "reset_quota":
            await self.quota.admin_reset()
            await self._reply(msg.group_id, "已清空今日额度~")
            return
        if sub == "reset_memory":
            if rest == "all":
                await self.memory.admin_reset_group(msg.group_id)
                await self._reply(msg.group_id, f"已清空群 {msg.group_id} 的所有对话记忆")
            elif rest.isdigit():
                await self.memory.reset(msg.group_id, int(rest))
                await self._reply(msg.group_id, f"已清空用户 {rest} 的对话记忆")
            else:
                await self._reply(msg.group_id, "用法: /admin reset_memory <user_id|all>")
            return
        if sub == "allow_group":
            if not rest.isdigit():
                await self._reply(msg.group_id, "用法: /admin allow_group <group_id>")
                return
            added = await allowlist.add(int(rest))
            await self._reply(msg.group_id, f"群 {rest} {'已添加' if added else '本来就在白名单里了'}")
            return
        if sub == "disallow_group":
            if not rest.isdigit():
                await self._reply(msg.group_id, "用法: /admin disallow_group <group_id>")
                return
            removed = await allowlist.remove(int(rest))
            if removed:
                await self._reply(msg.group_id, f"群 {rest} 已移除")
            elif int(rest) in CONFIG.allowed_groups:
                await self._reply(msg.group_id, "该群在 .env 里固定允许，无法运行时移除")
            else:
                await self._reply(msg.group_id, "该群不在白名单中")
            return
        if sub == "list_groups":
            groups = sorted(await allowlist.all_allowed_groups())
            lines = ["允许的群（* 表示固定在 env 中）："]
            for g in groups:
                marker = " *" if g in CONFIG.allowed_groups else ""
                lines.append(f"  {g}{marker}")
            await self._reply(msg.group_id, "\n".join(lines) if groups else "白名单是空的")
            return
        if sub == "report":
            await self._send_daily_report(force=True)
            await self._reply(msg.group_id, "日报已推送~")
            return
        if sub == "ping":
            await self._reply(msg.group_id, self._format_ping())
            return
        await self._reply(msg.group_id, "未知子命令，发送 /admin help 查看帮助")

    def _format_ping(self) -> str:
        if self.health_status is None:
            return "（未注入健康检查回调）"
        return format_ws_status(self.health_status())

    async def _format_global_usage(self) -> str:
        dump = await self.quota.dump_today()
        # group-level counters, by group id
        groups = dump.get("group", {})
        if not groups:
            return "今天还没人用过付费功能~"
        lines = ["今日全局用量 (按群):"]
        for gid in sorted(groups, key=lambda x: int(x) if x.isdigit() else 0):
            routes = groups[gid]
            parts = [f"{r}={n}" for r, n in sorted(routes.items())]
            lines.append(f"  群 {gid}: " + ", ".join(parts))
        return "\n".join(lines)

    async def _dispatch_llm_route(self, msg: ParsedMessage) -> None:
        decision = await self.router.decide(msg.text, has_image=msg.has_image)
        prompt = decision.normalized_prompt or msg.text

        route = decision.route
        if route == "deepseek_chat":
            await self._run_deepseek_chat(msg, prompt)
        elif route == "deepseek_think":
            await self._run_deepseek_think(msg, prompt)
        elif route == "openai_text":
            await self._run_openai_text(msg, prompt)
        elif route == "openai_vision":
            await self._run_openai_vision(msg, prompt)
        elif route == "openai_image":
            await self._run_openai_image(msg, prompt)
        elif route == "openai_image_edit":
            await self._run_openai_image_edit(msg, prompt)
        else:
            await self._reply(msg.group_id, REJECT_MSG)

    # ---------- route runners ----------
    async def _run_deepseek_chat(self, msg: ParsedMessage, prompt: str) -> None:
        await self._run_text(
            msg, prompt, provider=self.deepseek, route="deepseek_chat", supports_stream=True,
        )

    async def _run_deepseek_think(self, msg: ParsedMessage, prompt: str) -> None:
        await self._run_text(
            msg, prompt, provider=self.deepseek, route="deepseek_think",
            model=CONFIG.deepseek_reasoner_model, supports_stream=True,
        )

    async def _run_openai_text(self, msg: ParsedMessage, prompt: str) -> None:
        if self.openai is None:
            await self._reply(msg.group_id, "OpenAI 未配置~")
            return
        if not await self._check_quota("openai_text", msg):
            return
        await self._run_text(
            msg, prompt, provider=self.openai, route="openai_text", supports_stream=False,
        )
        await self.quota.consume("openai_text", msg.group_id, msg.user_id)

    async def _run_openai_vision(self, msg: ParsedMessage, prompt: str) -> None:
        if self.openai is None:
            await self._reply(msg.group_id, "OpenAI 未配置~")
            return
        memo = self._last_image.get((msg.group_id, msg.user_id))
        image_urls = msg.image_urls or ([memo.url] if memo and memo.url else [])
        if not image_urls:
            await self._reply(msg.group_id, "请先发一张图片再用 /vision 哦")
            return
        if not await self._check_quota("openai_vision", msg):
            return
        # Downscale each input to MAX_VISION_INPUT_SIZE and send as data URI
        # instead of the original URL. Cuts vision token cost to near zero.
        try:
            data_uris: list[str] = []
            for url in image_urls:
                raw = await self._download(url)
                small = downscale_to_max(raw, CONFIG.max_vision_input_size)
                data_uris.append(to_data_uri(small))
        except (httpx.HTTPError, OSError, ValueError) as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=f"图片处理失败: {e}"))
            return
        try:
            reply = await self.openai.vision(prompt, data_uris, max_tokens=600)
        except ProviderError as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))
            return
        await self.quota.consume("openai_vision", msg.group_id, msg.user_id)
        await self._reply(msg.group_id, reply.text)

    async def _run_openai_image(self, msg: ParsedMessage, prompt: str) -> None:
        if self.openai is None:
            await self._reply(msg.group_id, "OpenAI 未配置~")
            return
        if not prompt.strip():
            await self._reply(msg.group_id, "请告诉我你想画什么呀~")
            return
        if not await self._check_quota("openai_image", msg):
            return
        try:
            img = await self.openai.generate(prompt, size=CONFIG.openai_image_size)
        except ProviderError as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))
            return
        await self.quota.consume("openai_image", msg.group_id, msg.user_id)
        await self._send_image_reply(msg.group_id, img)

    async def _run_openai_image_edit(self, msg: ParsedMessage, prompt: str) -> None:
        if self.openai is None:
            await self._reply(msg.group_id, "OpenAI 未配置~")
            return
        try:
            image_bytes = await self._fetch_target_image(msg)
        except LookupError:
            await self._reply(msg.group_id, "找不到要编辑的图片，请先发一张图")
            return
        except httpx.HTTPError as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=f"下载图片失败: {e}"))
            return
        if not await self._check_quota("openai_image_edit", msg):
            return
        # dall-e-2 edit needs input==output size; gpt-image-* doesn't care.
        if CONFIG.openai_image_model.startswith("dall-e"):
            try:
                target_w = int(CONFIG.openai_image_size.split("x")[0])
                image_bytes = downscale_to_max(image_bytes, target_w)
            except (ValueError, OSError) as e:
                await self._reply(msg.group_id, ERROR_MSG.format(err=f"图片预处理失败: {e}"))
                return
        try:
            img = await self.openai.edit(prompt, image_bytes, size=CONFIG.openai_image_size)
        except ProviderError as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))
            return
        await self.quota.consume("openai_image_edit", msg.group_id, msg.user_id)
        await self._send_image_reply(msg.group_id, img)

    # ---------- image cache ----------
    async def _cache_image_to_disk(self, group_id: int, user_id: int, url: str) -> None:
        """Eagerly fetch + write image bytes to data/images/<sha>.dat."""
        try:
            data = await self._download(url)
        except httpx.HTTPError as e:
            log.warning("image prefetch failed: %s", e)
            return
        digest = hashlib.sha256(data).hexdigest()[:32]
        path = IMAGE_DIR / f"{digest}.dat"
        try:
            path.write_bytes(data)
        except OSError as e:
            log.warning("image disk-cache write failed: %s", e)
            return
        self._last_image[(group_id, user_id)] = _ImageMemo(
            url=url, disk_path=path, cached_at=time.monotonic()
        )

    async def _fetch_target_image(self, msg: ParsedMessage) -> bytes:
        key = (msg.group_id, msg.user_id)
        memo = self._last_image.get(key) or _ImageMemo()
        if msg.image_urls:
            await self._cache_image_to_disk(msg.group_id, msg.user_id, msg.image_urls[0])
            memo = self._last_image.get(key) or memo
        if memo.is_fresh(CONFIG.image_cache_ttl) and memo.disk_path is not None:
            return memo.disk_path.read_bytes()
        if not memo.url:
            raise LookupError("no image in context")
        data = await self._download(memo.url)
        digest = hashlib.sha256(data).hexdigest()[:32]
        path = IMAGE_DIR / f"{digest}.dat"
        path.write_bytes(data)
        memo.disk_path = path
        memo.cached_at = time.monotonic()
        self._last_image[key] = memo
        return data

    @staticmethod
    async def sweep_image_cache() -> None:
        """Drop image files older than 2× IMAGE_CACHE_TTL. Safe to call repeatedly."""
        cutoff = time.time() - 2 * CONFIG.image_cache_ttl
        removed = 0
        for p in IMAGE_DIR.glob("*.dat"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            log.info("image cache sweep: removed %d old files", removed)

    # ---------- text core (streaming-aware) ----------
    async def _run_text(
        self,
        msg: ParsedMessage,
        prompt: str,
        *,
        provider,
        route: str,
        model: Optional[str] = None,
        supports_stream: bool = False,
    ) -> None:
        if not prompt.strip():
            await self._reply(msg.group_id, "请把你的问题说清楚一点~")
            return
        history = await self.memory.get(msg.group_id, msg.user_id)
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "你是一个友好的 QQ 群助手。请用简洁、口语化的中文回答，"
                    "除非用户明确用英文提问。回答不要超过 1500 字。"
                ),
            )
        ]
        for role, content in history:
            messages.append(ChatMessage(role=role, content=content))
        messages.append(ChatMessage(role="user", content=prompt))

        use_stream = supports_stream and CONFIG.stream_replies and hasattr(provider, "chat_stream")
        try:
            if use_stream:
                text = await self._stream_text(msg.group_id, provider, messages, model)
                log.info("route=%s provider=%s model=%s streamed=true", route, provider.name, model)
            else:
                reply = await provider.chat(messages, model=model, max_tokens=1200)
                text = reply.text or "(空回复)"
                log.info(
                    "route=%s provider=%s model=%s tokens=%s",
                    route, provider.name, reply.model, reply.usage,
                )
                await self._reply(msg.group_id, text)
        except ProviderError as e:
            await self._reply(msg.group_id, ERROR_MSG.format(err=str(e)[:120]))
            return

        await self.memory.append(msg.group_id, msg.user_id, "user", prompt)
        await self.memory.append(msg.group_id, msg.user_id, "assistant", text)

    async def _stream_text(
        self,
        group_id: int,
        provider,
        messages: List[ChatMessage],
        model: Optional[str],
    ) -> str:
        """Consume the streaming iterator; flush a QQ message every paragraph or
        every CONFIG.stream_flush_chars characters."""
        buf: list[str] = []
        pending = ""
        full = ""
        flush_at = CONFIG.stream_flush_chars
        async for chunk in provider.chat_stream(messages, model=model, max_tokens=1200):
            pending += chunk
            full += chunk
            # flush at paragraph breaks, else when buffer crosses threshold
            while True:
                nl = pending.find("\n\n")
                if nl >= 0 and nl + 2 <= len(pending):
                    piece, pending = pending[: nl + 2], pending[nl + 2 :]
                    buf.append(piece)
                    await self._reply(group_id, piece.strip())
                    continue
                if len(pending) >= flush_at:
                    piece, pending = pending[:flush_at], pending[flush_at:]
                    buf.append(piece)
                    await self._reply(group_id, piece.strip())
                    continue
                break
        if pending.strip():
            await self._reply(group_id, pending.strip())
        return full.strip() or "(空回复)"

    async def _check_quota(self, route: str, msg: ParsedMessage) -> bool:
        ok, reason = await self.quota.check(route, msg.group_id, msg.user_id)
        if not ok:
            log.info("quota blocked route=%s reason=%s", route, reason)
            await self._reply(msg.group_id, QUOTA_EXCEEDED_MSG)
        return ok

    async def _format_balance(self, msg: ParsedMessage) -> str:
        snap = await self.quota.snapshot(msg.group_id, msg.user_id)
        lines = ["今日额度：(已用 / 上限)"]
        labels = {
            "openai_text": "GPT 文本",
            "openai_image": "图片生成",
            "openai_image_edit": "图片编辑",
            "openai_vision": "图片理解",
        }
        for route, label in labels.items():
            cells = snap.get(route, {"user": "?", "group": "?"})
            lines.append(f"  {label}: 你 {cells['user']} / 群 {cells['group']}")
        return "\n".join(lines)

    async def _reply(self, group_id: int, text: str) -> None:
        for chunk in chunk_text(text, CONFIG.limits.max_reply_chars):
            await self.send_text(group_id, chunk)

    async def _send_image_reply(self, group_id: int, img: ImageReply) -> None:
        if img.b64_png:
            payload = f"base64://{img.b64_png}"
        elif img.url:
            payload = img.url
        else:
            await self._reply(group_id, ERROR_MSG.format(err="image_empty"))
            return
        await self.send_image(group_id, payload)

    async def _download(self, url: str) -> bytes:
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.content

    # ---------- daily report ----------
    async def _send_daily_report(self, *, force: bool = False) -> None:
        gid = CONFIG.daily_report_group
        if gid <= 0:
            log.debug("no DAILY_REPORT_GROUP configured, skipping")
            return
        if not force:
            from datetime import date as _date
            from bot.storage import Storage
            store = await Storage.get()
            if not await store.report_mark_sent(_date.today().isoformat()):
                return  # someone (or a previous run) already sent today's
        text = await self._format_global_usage()
        await self._reply(gid, "📊 今日 LLM 用量日报\n" + text)

    async def aclose(self) -> None:
        await self._http.aclose()
