"""Orchestrates incoming messages: filter -> route -> provider -> reply."""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from bot import allowlist
from bot.group_memory import GroupMemory, GroupMsg
from bot.emoji_filter import filter_emoji
from bot.interjection_filter import filter_interjections
from bot.image_utils import downscale_to_max, to_data_uri
from bot.important_memory import ImportantMemory
from bot.logger import get_logger
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.message_parser import ParsedMessage, QuotedMessage, chunk_text
from bot.persona import load_persona
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
# Fetch the message referenced by [CQ:reply,id=...] (text + image URLs).
FetchReply = Callable[[str], Awaitable[Optional[QuotedMessage]]]
# Returns a `bot.onebot_client.WsStatus` (kept loose-typed to avoid the import cycle).
HealthFn = Callable[[], object]

QUOTA_EXCEEDED_MSG = "今天这个功能的额度用完了，请明天再试吧~"
RATE_LIMITED_MSG = "你发得太快啦，先休息一下吧~"
REJECT_MSG = "这个请求我没法处理~"
ERROR_MSG = "Can someone tell R there is a problem with my AI."

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
    "指令（都要 @我 才生效）：\n"
    "/ask <问题>      普通对话\n"
    "/think <问题>    深度推理\n"
    "/gpt <问题>      用 GPT 回答（受额度）\n"
    "/image <描述>    生成图片（受额度）\n"
    "/vision <问题>   分析最近一张图\n"
    "/edit <修改指令> 编辑最近一张图\n"
    "/recap [今天|昨天|1h|1d|一周]  总结群里活动\n"
    "/recall [YYYY-MM-DD|关键词]  查长时记忆\n"
    "/remember [list|cancel <id>]  看/取消我帮你记的事\n"
    "/reset           清空我和你的对话记忆\n"
    "/balance         查看今日额度\n"
    "/help            显示帮助\n"
    "（直接 @我 也行～）"
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
    fetch_reply: Optional[FetchReply] = None  # injected from main
    health_status: Optional[HealthFn] = None  # injected from main
    group_memory: GroupMemory = field(default_factory=GroupMemory)
    long_memory: Optional[LongMemory] = None
    important_memory: Optional[ImportantMemory] = None
    _http: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=60.0))
    _last_image: Dict[Tuple[int, int], _ImageMemo] = field(default_factory=dict)
    _last_dispatch_at: Dict[int, float] = field(default_factory=dict)
    # Proactive-interjection bookkeeping (per group).
    _last_bot_speech_at: Dict[int, float] = field(default_factory=dict)
    _msgs_since_bot_spoke: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.long_memory is None:
            self.long_memory = LongMemory(self.group_memory, self.deepseek)
        if self.important_memory is None:
            self.important_memory = ImportantMemory(self.deepseek)

    # ---------- entry ----------
    async def handle(self, msg: ParsedMessage) -> None:
        if not await allowlist.is_allowed(msg.group_id):
            log.debug("ignoring message from non-allowed group %s", msg.group_id)
            return
        if msg.self_id == msg.user_id:
            return
        if not msg.text and not msg.has_image:
            return

        # Log into group memory BEFORE any filtering — even messages we'll
        # ignore become part of the bot's awareness of the group. Awaited (not
        # background task) so downstream reads (proactive judge, /recap, chat
        # context injection) always see this row. Skip /commands; they're
        # bot-control plumbing, not conversation.
        record_text = msg.text or ("[图片]" if msg.has_image else "")
        row_id = 0
        if record_text and not msg.is_command:
            row_id = await self.group_memory.append(
                msg.group_id, msg.user_id, msg.nickname or f"u{msg.user_id}", record_text,
            )
            # If image floated through and auto-vision is on, caption it in
            # the background so the row text becomes useful for recap / context.
            if (
                CONFIG.auto_vision_group_images
                and msg.has_image
                and row_id > 0
                and self.openai is not None
            ):
                asyncio.create_task(self._caption_image_for_memory(
                    row_id, msg.group_id, msg.image_urls[0], msg.text or "",
                ))
            # Important-memory classifier: text-only messages run through the
            # LLM judge so reminders / facts / decisions get persisted. Cheap
            # regex pre-filter inside maybe_extract gates the LLM call.
            if (
                CONFIG.important_memory_enabled
                and self.important_memory is not None
                and msg.text
            ):
                asyncio.create_task(self._safe_extract_memory(
                    msg.group_id, msg.user_id,
                    msg.nickname or f"u{msg.user_id}", msg.text,
                ))
        # Count toward "messages since bot spoke" for proactive interjection.
        self._msgs_since_bot_spoke[msg.group_id] = (
            self._msgs_since_bot_spoke.get(msg.group_id, 0) + 1
        )

        if msg.has_image:
            asyncio.create_task(
                self._cache_image_to_disk(msg.group_id, msg.user_id, msg.image_urls[0])
            )

        if not self._trigger_allows(msg):
            log.debug("trigger mode %s skipped: %r", CONFIG.trigger_mode, msg.text[:60])
            if CONFIG.proactive_enabled:
                await self._maybe_proactive(msg)
            return
        msg = self._strip_trigger(msg)

        # Cooldown: silently skip non-command chatter within N seconds of last
        # dispatch in this group. /commands are exempt (explicit user actions).
        if not msg.is_command and CONFIG.reply_cooldown_seconds > 0:
            last = self._last_dispatch_at.get(msg.group_id, 0.0)
            now = time.monotonic()
            elapsed = now - last
            if elapsed < CONFIG.reply_cooldown_seconds:
                log.debug(
                    "cooldown skip group=%s elapsed=%.1fs cap=%ds",
                    msg.group_id, elapsed, CONFIG.reply_cooldown_seconds,
                )
                return
        # Stamp BEFORE dispatch so concurrent messages don't all squeeze through.
        if not msg.is_command:
            self._last_dispatch_at[msg.group_id] = time.monotonic()

        if not self.rate.check(msg.user_id):
            await self._reply(msg.group_id, RATE_LIMITED_MSG)
            return

        log.info(
            "msg from group=%s user=%s cmd=%s has_image=%s reply_to=%s text=%r",
            msg.group_id, msg.user_id, msg.command if msg.is_command else "-",
            msg.has_image, msg.reply_to_msg_id, msg.text[:120],
        )

        # Reply-segment: bring the quoted message's content into scope so the
        # bot can act on it (text becomes prompt context; images become input).
        if msg.reply_to_msg_id and self.fetch_reply is not None:
            try:
                quoted = await self.fetch_reply(msg.reply_to_msg_id)
            except Exception as e:
                log.warning("fetch_reply failed: %s", e)
                quoted = None
            if quoted:
                if quoted.text:
                    msg.text = (
                        f"[被引用的消息]\n{quoted.text}\n\n[我的问题]\n{msg.text}"
                    ).strip()
                # If the quote carried an image and this message didn't, treat
                # the quoted image as the user's own attachment.
                if quoted.image_urls and not msg.image_urls:
                    msg.image_urls = list(quoted.image_urls)
                    log.info(
                        "reply-segment provided image(s); using %d from quoted msg",
                        len(quoted.image_urls),
                    )
                    asyncio.create_task(
                        self._cache_image_to_disk(
                            msg.group_id, msg.user_id, msg.image_urls[0]
                        )
                    )

        try:
            if msg.is_command:
                await self._dispatch_command(msg)
            else:
                await self._dispatch_llm_route(msg)
        except ProviderError:
            log.exception("provider error")
            await self._reply(msg.group_id, ERROR_MSG)
        except Exception:  # noqa: BLE001
            log.exception("unhandled error")
            await self._reply(msg.group_id, ERROR_MSG)

    # ---------- trigger gate ----------
    def _trigger_allows(self, msg: ParsedMessage) -> bool:
        # /commands now require @bot too — the bot must be explicitly addressed.
        if msg.is_command:
            return msg.mentions(msg.self_id)
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
        elif c == "recap":
            await self._run_recap(msg, args)
        elif c == "recall":
            await self._run_recall(msg, args)
        elif c == "remember":
            await self._run_remember(msg, args)
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
                "/admin ping                 OneBot 连接状态与最近心跳\n"
                "/admin save_recap [day]     手动写入某天的长时记忆",
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
        if sub == "save_recap":
            # /admin save_recap [yesterday|today|YYYY-MM-DD]
            if self.long_memory is None:
                await self._reply(msg.group_id, "长时记忆未启用")
                return
            target = rest.strip().lower() or "yesterday"
            if target in ("yesterday", "昨天"):
                day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            elif target in ("today", "今天"):
                day = datetime.now().strftime("%Y-%m-%d")
            elif self._DATE_RE.match(target):
                day = target
            else:
                await self._reply(
                    msg.group_id,
                    "用法: /admin save_recap [today|yesterday|YYYY-MM-DD]",
                )
                return
            summary = await self.long_memory.save_day(msg.group_id, day)
            if summary is None:
                await self._reply(msg.group_id, f"{day} 没消息可总结")
            else:
                await self._reply(msg.group_id, f"已存档 {day}:\n{summary}")
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
        was_at_bot = msg.mentions(msg.self_id)
        decision = await self.router.decide(
            msg.text,
            has_image=msg.has_image,
            was_at_bot=was_at_bot,
        )
        prompt = decision.normalized_prompt or msg.text

        route = decision.route
        if route == "skip":
            # Router decided this message isn't addressed at us. Stay silent.
            # Also undo the cooldown stamp so the next message in this group
            # isn't unfairly blocked by a non-reply.
            self._last_dispatch_at.pop(msg.group_id, None)
            return

        # Ambient gate: when the router approves a non-skip route but the
        # message wasn't directly addressed (no @, no nickname in text), throttle
        # with tier-specific probability + per-group cooldown so the bot doesn't
        # pile onto every conversation. Directly-addressed messages bypass entirely.
        addressed = was_at_bot or (
            CONFIG.bot_nickname and CONFIG.bot_nickname in msg.text
        )
        if not addressed:
            tier = decision.tier
            p = (CONFIG.ambient_reply_probability_high if tier == "high"
                 else CONFIG.ambient_reply_probability_low)
            elapsed = time.monotonic() - self._last_bot_speech_at.get(msg.group_id, 0.0)
            if elapsed < CONFIG.ambient_reply_min_seconds:
                log.info(
                    "ambient gate: cooldown %.1fs < %ds (tier=%s) — skip",
                    elapsed, CONFIG.ambient_reply_min_seconds, tier,
                )
                self._last_dispatch_at.pop(msg.group_id, None)
                return
            if random.random() >= p:
                log.info(
                    "ambient gate: dice skip (tier=%s, p=%.2f)", tier, p,
                )
                self._last_dispatch_at.pop(msg.group_id, None)
                return
            log.info(
                "ambient gate: passed (tier=%s, p=%.2f, elapsed=%.1fs)",
                tier, p, elapsed,
            )

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
        except (httpx.HTTPError, OSError, ValueError):
            log.exception("vision image preprocessing failed")
            await self._reply(msg.group_id, ERROR_MSG)
            return
        try:
            reply = await self.openai.vision(prompt, data_uris, max_tokens=600)
        except ProviderError:
            log.exception("openai vision call failed")
            await self._reply(msg.group_id, ERROR_MSG)
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
        except ProviderError:
            log.exception("openai image generation failed")
            await self._reply(msg.group_id, ERROR_MSG)
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
        except httpx.HTTPError:
            log.exception("image download for /edit failed")
            await self._reply(msg.group_id, ERROR_MSG)
            return
        if not await self._check_quota("openai_image_edit", msg):
            return
        # dall-e-2 edit needs input==output size; gpt-image-* doesn't care.
        if CONFIG.openai_image_model.startswith("dall-e"):
            try:
                target_w = int(CONFIG.openai_image_size.split("x")[0])
                image_bytes = downscale_to_max(image_bytes, target_w)
            except (ValueError, OSError):
                log.exception("image preprocessing for /edit failed")
                await self._reply(msg.group_id, ERROR_MSG)
                return
        try:
            img = await self.openai.edit(prompt, image_bytes, size=CONFIG.openai_image_size)
        except ProviderError:
            log.exception("openai image edit failed")
            await self._reply(msg.group_id, ERROR_MSG)
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

    async def _caption_image_for_memory(
        self,
        row_id: int,
        group_id: int,
        image_url: str,
        user_text: str,
    ) -> None:
        """Background: caption the image and rewrite the row in group_memory.

        Silently bails on quota / network / decoding errors — this is best-effort
        ambient context, never user-facing.
        """
        if self.openai is None:
            return
        ok, reason = await self.quota.check("auto_vision", group_id, 0)
        if not ok:
            log.info("auto-caption skipped: %s", reason)
            return
        try:
            raw = await self._download(image_url)
            small = downscale_to_max(raw, CONFIG.max_vision_input_size)
            data_uri = to_data_uri(small)
            reply = await self.openai.vision(
                "用中文一句话概括这张图（15字以内，只输出概括本身，不加标点）",
                [data_uri],
                max_tokens=60,
            )
        except Exception:
            log.exception("auto-caption call failed")
            return
        caption = (reply.text or "").strip().split("\n")[0][:40]
        if not caption:
            return
        await self.quota.consume("auto_vision", group_id, 0)
        # Rebuild the row's text: keep any user text, replace placeholder.
        if user_text.strip():
            new_text = f"{user_text.strip()} [图：{caption}]"
        else:
            new_text = f"[图：{caption}]"
        try:
            await self.group_memory.update_text(row_id, new_text)
        except Exception:
            log.exception("auto-caption row update failed")
            return
        log.info("auto-caption ok group=%s caption=%r", group_id, caption)

    async def _safe_extract_memory(
        self, group_id: int, user_id: int, nickname: str, text: str,
    ) -> None:
        """Background classifier wrapper that never raises."""
        if self.important_memory is None:
            return
        try:
            await self.important_memory.maybe_extract(
                group_id=group_id, user_id=user_id, nickname=nickname, text=text,
            )
        except Exception:
            log.exception("important-memory extract failed")

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
            ChatMessage(role="system", content=load_persona()),
        ]
        # Long-term memory: most recent N daily recaps (cheap because each is
        # ~100 chars). Lets the bot vaguely reference "前几天" / "上周".
        if self.long_memory is not None and CONFIG.long_memory_inject_days > 0:
            recaps = await self.long_memory.recent(msg.group_id)
            if recaps:
                lines = ["你的长时记忆（最近几天群里的事，被问到才参考）："]
                for day, summary in reversed(recaps):  # oldest first
                    lines.append(f"- {day}: {summary}")
                messages.append(ChatMessage(role="system", content="\n".join(lines)))
        # Inject recent group activity so the bot has shared context with
        # everyone in the group, not just the user it's replying to.
        group_ctx = await self.group_memory.recent(msg.group_id)
        if group_ctx:
            messages.append(ChatMessage(
                role="system",
                content=self._format_group_context(
                    group_ctx, bot_nickname=CONFIG.bot_nickname,
                ),
            ))
        # Inject important-memory recall: free-form facts/preferences/decisions
        # the LLM classifier saved earlier. Personal items rank above group ones.
        if (
            CONFIG.important_memory_enabled
            and self.important_memory is not None
            and CONFIG.important_memory_recall_limit > 0
        ):
            mem_rows = await self.important_memory.recall_for_user(
                msg.group_id, msg.user_id,
                limit=CONFIG.important_memory_recall_limit,
            )
            mem_block = ImportantMemory.format_for_prompt(
                mem_rows, speaker_user_id=msg.user_id,
            )
            if mem_block:
                messages.append(ChatMessage(role="system", content=mem_block))
        for role, content in history:
            messages.append(ChatMessage(role=role, content=content))
        messages.append(ChatMessage(role="user", content=prompt))

        try:
            reply = await provider.chat(messages, model=model, max_tokens=600)
        except ProviderError:
            log.exception("text route provider error")
            await self._reply(msg.group_id, ERROR_MSG)
            return
        text = reply.text or "(空回复)"
        log.info(
            "route=%s provider=%s model=%s tokens=%s",
            route, provider.name, reply.model, reply.usage,
        )
        await self._human_send(msg.group_id, text)

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

        def _clean(piece: str) -> str:
            piece = filter_emoji(piece, CONFIG.emoji_keep_probability)
            piece = filter_interjections(piece)
            return piece.strip()

        async for chunk in provider.chat_stream(messages, model=model, max_tokens=1200):
            pending += chunk
            full += chunk
            # flush at paragraph breaks, else when buffer crosses threshold
            while True:
                nl = pending.find("\n\n")
                if nl >= 0 and nl + 2 <= len(pending):
                    piece, pending = pending[: nl + 2], pending[nl + 2 :]
                    buf.append(piece)
                    cleaned = _clean(piece)
                    if cleaned:
                        await self._reply(group_id, cleaned)
                    continue
                if len(pending) >= flush_at:
                    piece, pending = pending[:flush_at], pending[flush_at:]
                    buf.append(piece)
                    cleaned = _clean(piece)
                    if cleaned:
                        await self._reply(group_id, cleaned)
                    continue
                break
        if pending.strip():
            cleaned = _clean(pending)
            if cleaned:
                await self._reply(group_id, cleaned)
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

    # ---------- human-paced reply splitter ----------
    _SENT_SPLIT_RE = re.compile(r"(?<=[。？！；.!?])\s*|(?:\n\s*\n)+")

    @classmethod
    def _split_human_chunks(cls, text: str, max_chunks: int) -> List[str]:
        """Split into 1..max_chunks short messages, splitting on sentence-end
        punctuation or paragraph breaks. Never splits mid-sentence."""
        text = text.strip()
        if not text:
            return []
        parts = [p.strip() for p in cls._SENT_SPLIT_RE.split(text) if p and p.strip()]
        if not parts:
            return [text]
        if len(parts) <= max_chunks:
            return parts
        # More sentences than chunks — distribute evenly.
        per = -(-len(parts) // max_chunks)  # ceil div
        return ["".join(parts[i:i + per]) for i in range(0, len(parts), per)]

    async def _human_send(self, group_id: int, text: str, *, log_to_memory: bool = True) -> None:
        """Send `text` as 1..N short messages with random delays between them.

        Also records each chunk to group_memory so /recap and future context
        injection see the bot's voice as part of the conversation."""
        # Strip most emojis the LLM leaked through — they're the #1 AI tell.
        text = filter_emoji(text, CONFIG.emoji_keep_probability)
        # Strip AI-ish 嘿嘿 / 诶呀 / 嘶 / 嗯嗯 — group history reinforces them and
        # the persona alone can't suppress reliably.
        text = filter_interjections(text)
        if not text.strip():
            return

        # Stamp proactive-interjection state at the moment of speech.
        self._last_bot_speech_at[group_id] = time.monotonic()
        self._msgs_since_bot_spoke[group_id] = 0

        if not CONFIG.human_send_enabled:
            await self._reply(group_id, text)
            if log_to_memory:
                asyncio.create_task(self.group_memory.append(
                    group_id, 0, CONFIG.bot_nickname, text,
                ))
            return

        chunks = self._split_human_chunks(text, CONFIG.human_send_max_chunks)
        if not chunks:
            return
        for i, chunk in enumerate(chunks):
            if i > 0:
                base = random.uniform(
                    CONFIG.human_send_delay_min, CONFIG.human_send_delay_max,
                )
                # Longer chunks take longer to "type".
                base += min(2.5, len(chunk) * 0.05)
                await asyncio.sleep(min(base, 5.0))
            # Honor MAX_REPLY_CHARS chunking as a safety belt too.
            for piece in chunk_text(chunk, CONFIG.limits.max_reply_chars):
                await self.send_text(group_id, piece)
            if log_to_memory:
                asyncio.create_task(self.group_memory.append(
                    group_id, 0, CONFIG.bot_nickname, chunk,
                ))

    # ---------- group-context formatting ----------
    @staticmethod
    def _format_group_context(rows: List[GroupMsg], *, bot_nickname: str) -> str:
        """Render recent group rows into a compact system-prompt snippet."""
        lines: list[str] = []
        for r in rows:
            who = "你" if r.nickname == bot_nickname else r.nickname
            t = datetime.fromtimestamp(r.ts).strftime("%H:%M")
            text = r.text.replace("\n", " ")
            lines.append(f"[{t} {who}] {text}")
        return "最近群里的对话（按时间顺序，最新在底部，你可以参考但不必每条都接）：\n" + "\n".join(lines)

    # ---------- /recap ----------
    _PERIOD_RE = re.compile(r"^\s*(\d+)\s*([hd])\s*$", re.IGNORECASE)

    @staticmethod
    def _parse_period(args: str) -> Optional[Tuple[float, str]]:
        """Return (since_ts_unix, label) or None for unrecognised input."""
        a = args.strip().lower()
        now = datetime.now()
        if a in ("", "今天", "today"):
            t = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return t.timestamp(), "今天"
        if a in ("昨天", "yesterday"):
            today0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
            yest0 = today0 - timedelta(days=1)
            # Note: callers should fetch since yest0 and trim to before today0.
            # For our use we cheat — return yest0 and rely on natural row count.
            return yest0.timestamp(), "昨天"
        m = Handler._PERIOD_RE.match(a)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            seconds = n * (3600 if unit == "h" else 86400)
            return (now - timedelta(seconds=seconds)).timestamp(), f"最近 {n}{unit}"
        if a in ("一小时", "一小時"):
            return (now - timedelta(hours=1)).timestamp(), "最近 1 小时"
        if a in ("一天",):
            return (now - timedelta(days=1)).timestamp(), "最近 24 小时"
        if a in ("一周", "一週"):
            return (now - timedelta(days=7)).timestamp(), "最近 7 天"
        return None

    # ---------- /recall (long-term memory query) ----------
    _DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

    async def _run_recall(self, msg: ParsedMessage, args: str) -> None:
        if self.long_memory is None:
            await self._reply(msg.group_id, "长时记忆未启用")
            return
        a = args.strip()
        if not a:
            # No arg → list last few days
            rows = await self.long_memory.recent(msg.group_id, days=7)
            if not rows:
                await self._human_send(msg.group_id, "我还没攒下啥长时记忆呢")
                return
            lines = ["最近几天的记忆："]
            for day, summary in rows:
                lines.append(f"{day}\n{summary}")
            await self._reply(msg.group_id, "\n\n".join(lines))
            return
        # Exact date
        if self._DATE_RE.match(a):
            summary = await self.long_memory.get(msg.group_id, a)
            if not summary:
                await self._human_send(msg.group_id, f"{a} 那天没存档呢")
                return
            await self._reply(msg.group_id, f"{a}\n{summary}")
            return
        # Keyword search
        rows = await self.long_memory.search(msg.group_id, a, limit=5)
        if not rows:
            await self._human_send(msg.group_id, f"没找到跟 '{a}' 有关的记忆")
            return
        lines = [f"找到 {len(rows)} 条相关记忆："]
        for day, summary in rows:
            lines.append(f"{day}\n{summary}")
        await self._reply(msg.group_id, "\n\n".join(lines))

    async def _run_remember(self, msg: ParsedMessage, args: str) -> None:
        """List / cancel important memories. Defaults to listing for caller."""
        if not CONFIG.important_memory_enabled or self.important_memory is None:
            await self._reply(msg.group_id, "重要记忆功能没开~")
            return
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "cancel":
            if not rest.isdigit():
                await self._reply(msg.group_id, "用法: /remember cancel <id>")
                return
            ok = await self.important_memory.cancel(int(rest), msg.group_id)
            await self._reply(
                msg.group_id,
                "已取消~" if ok else "找不到这条，或者已经取消/触发过了",
            )
            return
        if sub in ("", "list"):
            rows = await self.important_memory.list_pending(
                msg.group_id, msg.user_id, limit=15,
            )
            if not rows:
                await self._human_send(msg.group_id, "我现在没记着什么待办呢")
                return
            lines = ["我记着的事："]
            now_ts = time.time()
            for item_id, subj, content, trigger, recurrence in rows:
                who = "（你）" if subj == msg.user_id else (
                    "（群）" if subj is None else f"（u{subj}）"
                )
                t_part = ""
                if trigger:
                    delta = trigger - now_ts
                    if delta < 0:
                        t_part = " [应触发未发]"
                    elif delta < 3600:
                        t_part = f" [{int(delta // 60)} 分钟后]"
                    elif delta < 86400:
                        t_part = (
                            f" [{datetime.fromtimestamp(trigger).strftime('%H:%M')}]"
                        )
                    else:
                        t_part = (
                            f" [{datetime.fromtimestamp(trigger).strftime('%m-%d %H:%M')}]"
                        )
                if recurrence:
                    t_part += f" ({recurrence})"
                lines.append(f"#{item_id} {who} {content}{t_part}")
            lines.append("\n取消: /remember cancel <id>")
            await self._reply(msg.group_id, "\n".join(lines))
            return
        await self._reply(msg.group_id, "用法: /remember [list|cancel <id>]")

    async def _run_recap(self, msg: ParsedMessage, args: str) -> None:
        parsed = self._parse_period(args)
        if parsed is None:
            await self._reply(
                msg.group_id,
                "想看哪段？用法：/recap 今天 | 昨天 | 1h | 3h | 1d | 一周",
            )
            return
        since_ts, label = parsed
        rows = await self.group_memory.since(msg.group_id, since_ts)
        if not rows:
            await self._human_send(msg.group_id, f"{label}群里没啥消息呢")
            return

        # Feed the whole period to the LLM — recap quality drops sharply when
        # we throw out the beginning. The hard cap (well above GROUP_MEMORY_MAX)
        # is just a runaway-token safety belt; in normal operation we never hit
        # it because storage tops out at CONFIG.group_memory_max rows.
        if len(rows) > 3000:
            rows = rows[-3000:]
        transcript = "\n".join(
            f"[{datetime.fromtimestamp(r.ts).strftime('%H:%M')} {r.nickname}] {r.text}"
            for r in rows
        )
        prompt = (
            f"以下是{label}的群聊片段，按时间顺序。请用你的人设语气，"
            "在 80 字以内总结主要话题、有意思的对话和谁参与最多。"
            "不要列要点、不要 markdown。\n\n" + transcript
        )
        try:
            reply = await self.deepseek.chat(
                [
                    ChatMessage(role="system", content=load_persona()),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.5,
                max_tokens=400,
            )
        except ProviderError:
            log.exception("recap deepseek failed")
            await self._reply(msg.group_id, ERROR_MSG)
            return
        await self._human_send(msg.group_id, reply.text)

    # ---------- proactive interjection ----------
    _PROACTIVE_SYSTEM_PROMPT_TEMPLATE = (
        "You decide whether bot {nickname} should proactively jump into a QQ "
        "group chat WITHOUT being addressed.\n\n"
        "Default: skip. Only speak if it really fits the flow.\n\n"
        "Speak when (rare):\n"
        "- Clear hook for a short joke / one-liner that fits your persona\n"
        "- Group restarted talk after a quiet period\n"
        "- Someone said something that begs an obvious quick reaction\n"
        "- Conversation touches a topic you recently mentioned\n\n"
        "Skip when (most cases):\n"
        "- Serious discussion / personal talk\n"
        "- People are clearly talking to each other\n"
        "- Reaction-only messages (嗯/哈/666/emoji)\n"
        "- Nothing genuinely interesting to add\n"
        "- It would feel forced or noisy\n\n"
        "Persona reminder: {persona_blurb}\n\n"
        "Output STRICT JSON, one field only:\n"
        '- skip:  {{"r":"skip"}}\n'
        '- speak: {{"r":"say","t":"你要说的话(最多 20 字，符合人设，'
        "不要 markdown，不要 @ 人)\"}}\n"
    )

    _PROACTIVE_PERSONA_BLURB = (
        "短句、口语化、可不用标点；不要塞嘿嘿/诶呀/嘶/啦这类语气词凑可爱；"
        "不要自称 AI；不要列要点"
    )

    def _proactive_gate_open(self, msg: ParsedMessage) -> bool:
        """All the cheap, no-LLM checks."""
        if random.random() >= CONFIG.proactive_probability:
            return False
        last = self._last_bot_speech_at.get(msg.group_id, 0.0)
        if time.monotonic() - last < CONFIG.proactive_min_seconds:
            return False
        if (self._msgs_since_bot_spoke.get(msg.group_id, 0)
                < CONFIG.proactive_min_new_messages):
            return False
        return True

    async def _maybe_proactive(self, msg: ParsedMessage) -> None:
        if not self._proactive_gate_open(msg):
            return
        # Pull recent group context.
        ctx = await self.group_memory.recent(msg.group_id, limit=12)
        if len(ctx) < 3:
            return  # nothing to work with

        transcript = "\n".join(
            f"[{r.nickname}] {r.text}" for r in ctx
        )
        system = self._PROACTIVE_SYSTEM_PROMPT_TEMPLATE.format(
            nickname=CONFIG.bot_nickname,
            persona_blurb=self._PROACTIVE_PERSONA_BLURB,
        )
        try:
            reply = await self.deepseek.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=transcript),
                ],
                temperature=0.7,
                max_tokens=60,
                response_format={"type": "json_object"},
            )
        except ProviderError:
            log.exception("proactive judge call failed")
            return

        say_text = self._coerce_proactive_decision(reply.text)
        if say_text is None:
            log.info("proactive: judge said skip")
            return
        log.info("proactive: speaking %r", say_text[:60])
        await self._human_send(msg.group_id, say_text)

    @staticmethod
    def _coerce_proactive_decision(text: str) -> Optional[str]:
        """Return the utterance to say, or None to skip."""
        import json
        m = re.search(r"\{.*\}", text, re.DOTALL)
        payload = m.group(0) if m else text
        try:
            obj = json.loads(payload)
        except Exception:
            return None
        r = str(obj.get("r", "")).strip().lower()
        if r != "say":
            return None
        t = str(obj.get("t", "")).strip()
        if not t or len(t) > 60:  # sanity cap; persona allows ≤20 but be lenient
            return None
        return t

    async def _reply(self, group_id: int, text: str) -> None:
        for chunk in chunk_text(text, CONFIG.limits.max_reply_chars):
            await self.send_text(group_id, chunk)

    async def _send_image_reply(self, group_id: int, img: ImageReply) -> None:
        if img.b64_png:
            payload = f"base64://{img.b64_png}"
        elif img.url:
            payload = img.url
        else:
            log.warning("image provider returned empty payload")
            await self._reply(group_id, ERROR_MSG)
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

    # ---------- reminder firing ----------
    async def fire_due_reminders(self) -> int:
        """Send any due reminders. Called by the periodic loop in main.py.
        Returns number of reminders fired."""
        if not CONFIG.important_memory_enabled or self.important_memory is None:
            return 0
        try:
            due = await self.important_memory.due_reminders()
        except Exception:
            log.exception("due_reminders query failed")
            return 0
        fired = 0
        for item_id, group_id, subject_user_id, content, _src_nick, recurrence in due:
            if not await allowlist.is_allowed(group_id):
                # Group de-allowed since the memory was created — drop the row.
                await self.important_memory.cancel(item_id, group_id)
                continue
            try:
                # Compose the reminder body. If we know who it's for, @-mention.
                if subject_user_id and subject_user_id > 0:
                    body = f"[CQ:at,qq={subject_user_id}] {content}"
                else:
                    body = content
                await self.send_text(group_id, body)
                await self.important_memory.mark_fired(item_id, recurrence)
                fired += 1
                log.info(
                    "reminder fired id=%s group=%s subject=%s recur=%s",
                    item_id, group_id, subject_user_id, recurrence,
                )
            except Exception:
                log.exception("reminder send failed id=%s group=%s",
                              item_id, group_id)
        return fired

    # ---------- maintenance pass ----------
    async def run_maintenance(self) -> None:
        """Periodic housekeeping: rolling recap refresh + memories dedup/expiry.
        Idempotent and safe to call frequently."""
        groups: List[int] = []
        try:
            groups = sorted(await allowlist.all_allowed_groups())
        except Exception:
            log.exception("maintenance: failed to enumerate groups")
            return
        # Rolling daily-recap refresh for today + yesterday (idempotent upsert).
        if (
            CONFIG.daily_recap_enabled
            and self.long_memory is not None
            and groups
        ):
            today = datetime.now().strftime("%Y-%m-%d")
            yest = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            for gid in groups:
                for day in (today, yest):
                    try:
                        await self.long_memory.save_day(gid, day)
                    except Exception:
                        log.exception(
                            "maintenance: save_day failed group=%s day=%s",
                            gid, day,
                        )
            try:
                pruned = await self.long_memory.prune()
                if pruned:
                    log.info("maintenance: pruned %d old recap rows", pruned)
            except Exception:
                log.exception("maintenance: recap prune failed")
        # Memories dedup + expiry.
        if (
            CONFIG.important_memory_enabled
            and self.important_memory is not None
            and groups
        ):
            try:
                await self.important_memory.maintenance_pass(groups)
            except Exception:
                log.exception("maintenance: memory pass failed")

    async def aclose(self) -> None:
        await self._http.aclose()
