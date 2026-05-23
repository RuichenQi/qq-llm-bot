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
from bot import file_utils
from bot.group_memory import GroupMemory, GroupMsg
from bot.emoji_filter import filter_emoji
from bot.interjection_filter import filter_interjections
from bot.image_utils import downscale_to_max, to_data_uri
from bot.lessons import Lessons
from bot.logger import get_logger
from bot.long_memory import LongMemory
from bot.memory import Memory
from bot.message_parser import AttachedFile, ParsedMessage, QuotedMessage, chunk_text
from bot.persona import load_persona
from bot.quota import Quota
from bot.rate_limit import RateLimiter
from bot.router import Router
from bot.tools import ToolContext, ToolRegistry, run_with_tools
from config import CONFIG, IMAGE_DIR
from providers.base import ChatMessage, ImageReply, ProviderError
from providers.deepseek import DeepSeekProvider
from providers.openai_provider import OpenAIProvider
from providers.web_search import WebSearchProvider

log = get_logger(__name__)

SendText = Callable[[int, str], Awaitable[None]]
SendImage = Callable[[int, str], Awaitable[None]]
# Fetch the message referenced by [CQ:reply,id=...] (text + image URLs).
FetchReply = Callable[[str], Awaitable[Optional[QuotedMessage]]]
# Resolve a group-file id to a downloadable URL (adapter-specific).
FetchFileUrl = Callable[[int, str], Awaitable[Optional[str]]]
# Returns a `bot.onebot_client.WsStatus` (kept loose-typed to avoid the import cycle).
HealthFn = Callable[[], object]

QUOTA_EXCEEDED_MSG = "今天这个功能的额度用完了，请明天再试吧~"
RATE_LIMITED_MSG = "你发得太快啦，先休息一下吧~"
REJECT_MSG = "这个请求我没法处理~"
ERROR_MSG = "Can someone tell R there is a problem with my AI."


# Mainland-China politically sensitive keyword set used as a tool-level gate on
# `web_search`. Persona handles the conversational dodge; this just stops the
# bot from quietly pulling fresh content about these topics off the open web.
# Tuned for terms that are unambiguously political — common everyday words
# (中国 / 北京 / 上海) are NOT in here, since they're fine to search for in a
# non-political context.
_SENSITIVE_QUERY_RE = re.compile(
    r"六四|天安门|8964|六4|"
    r"习近平|李克强|李强|温家宝|江泽民|胡锦涛|"
    r"中共|共产党|党中央|政治局|中央政府|"
    r"文革|文化大革命|大跃进|反右|"
    r"达赖|西藏独立|藏独|tibet\s*independ|"
    r"维吾尔|新疆\s*(集中营|再教育营|压迫|种族)|xinjiang\s*camp|"
    r"法轮功|falun\s*gong|"
    r"刘晓波|艾未未|许志永|"
    r"709\s*律师|维权人士|"
    r"香港\s*(抗议|国安法|占中)|hong\s*kong\s*protest|"
    r"台独|台湾独立|两岸统独|"
    r"翻墙|vpn\s*(中国|大陆|墙)|gfw|防火长城|"
    r"白纸\s*(运动|革命)|"
    r"活摘\s*器官",
    re.IGNORECASE,
)


def _query_is_blocked(query: str) -> bool:
    """Belt-and-suspenders filter: refuse politically-sensitive web searches.

    Keeps the bot's tool layer aligned with the persona's "don't discuss"
    list even if the LLM tries to route around the system prompt.
    """
    return bool(_SENSITIVE_QUERY_RE.search(query or ""))

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
    "—— 对话 ——\n"
    "/ask <问题>           普通对话（DeepSeek）\n"
    "/think <问题>         深度推理\n"
    "/gpt <问题>           GPT 回答（受额度）\n"
    "/search <关键词>      联网搜索 + 回答\n"
    "—— 图片 ——\n"
    "/image <描述>         生成图片（受额度）\n"
    "/vision [问题]        分析最近一张图\n"
    "/edit <修改指令>      编辑最近一张图\n"
    "—— 文件 ——\n"
    "/file [问题]          回答关于刚上传文件的问题\n"
    "                       （支持 txt/pdf/docx/code/音频/视频）\n"
    "—— 群记忆 ——\n"
    "/recap [今天|昨天|1h|1d|一周]   总结群里活动\n"
    "/recall [YYYY-MM-DD|关键词]    查长时记忆\n"
    "/timewarp [一年前|半年前|YYYY-MM]  怀旧短文\n"
    "/dream                让我现在做个梦（仅管理员）\n"
    "—— 功能注入（规则/事实/约定/提醒）——\n"
    "/teach <规则>         教我一条今后要遵守的规则\n"
    "/remember [list|cancel <id>]   看/取消我记着的事\n"
    "/forget <id>          忘掉一条（=/remember cancel <id>）\n"
    "—— 杂项 ——\n"
    "/start                让我在本群上线（仅管理员）\n"
    "/stop                 让我在本群下线（仅管理员）\n"
    "/reset                清空我和你的对话记忆\n"
    "/balance              查看今日额度\n"
    "/help                 显示帮助\n"
    "（直接 @我 聊天也行～）"
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
    fetch_file_url: Optional[FetchFileUrl] = None  # injected from main
    health_status: Optional[HealthFn] = None  # injected from main
    group_memory: GroupMemory = field(default_factory=GroupMemory)
    long_memory: Optional[LongMemory] = None
    lessons: Optional[Lessons] = None
    # Web search backend (Tavily etc.); None when not configured. Tools that
    # want to search go through this; the LLM never sees the provider directly.
    web_search: Optional[WebSearchProvider] = None
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    _http: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=60.0))
    _last_image: Dict[Tuple[int, int], _ImageMemo] = field(default_factory=dict)
    # Cache: image-URL → short caption (for the quoted-image intent gate).
    _image_caption_cache: Dict[str, str] = field(default_factory=dict)
    _last_dispatch_at: Dict[int, float] = field(default_factory=dict)
    # Proactive-interjection bookkeeping (per group).
    _last_bot_speech_at: Dict[int, float] = field(default_factory=dict)
    _msgs_since_bot_spoke: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.long_memory is None:
            self.long_memory = LongMemory(self.group_memory, self.deepseek)
        if self.lessons is None:
            self.lessons = Lessons(self.deepseek)
        if CONFIG.tool_use_enabled:
            self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register the bot's default tool palette. The web_search tool is
        always registered (so the LLM can decide to use it) but gracefully
        reports unavailable when the backend isn't configured."""
        from bot.tools import Tool  # local import avoids cycle at module load

        async def _web_search_handler(args, ctx):
            query = str(args.get("query") or "").strip()
            if not query:
                return "Error: missing required parameter 'query'"
            # Belt-and-suspenders content gate: even if the LLM tries to dodge
            # the persona's "don't discuss" rule by going through the search
            # tool, refuse here. The returned message is itself benign so the
            # LLM can integrate it into a graceful dodge.
            if _query_is_blocked(query):
                log.info("web_search blocked sensitive query: %r", query[:120])
                return (
                    "（这条搜索被规则拦截了，不要把它转述给用户；"
                    "请用礼貌的话岔开话题，例如 '这个我不太懂呢，我们聊点别的吧'）"
                )
            max_results = args.get("max_results")
            try:
                n = int(max_results) if max_results is not None else CONFIG.web_search_max_results
            except (TypeError, ValueError):
                n = CONFIG.web_search_max_results
            n = max(1, min(n, 10))
            if self.web_search is None:
                return (
                    "Web search is not configured on this bot "
                    "(set TAVILY_API_KEY or WEB_SEARCH_ENABLED=0)."
                )
            try:
                results = await self.web_search.search(query, max_results=n)
            except Exception as e:  # noqa: BLE001
                log.warning("web_search tool failed: %s", e)
                return f"Web search failed: {e}"
            from providers.web_search import format_results
            return format_results(results)

        self.tools.register(Tool(
            name="web_search",
            description=(
                "Search the web for current or specific information. Call this "
                "whenever you need facts the model may not know — recent events, "
                "real-time data, niche/specific named entities (people, brands, "
                "places, products), or before generating/editing an image of a "
                "particular real-world subject so you can ground the description. "
                "Do NOT call for general knowledge, casual chat, or things you "
                "already know well. "
                "STRICTLY DO NOT search for mainland-China politically sensitive "
                "topics (CCP leadership, June 4, Xinjiang/Tibet/Taiwan/Hong Kong "
                "political issues, Falun Gong, dissidents, GFW/VPN, etc.) — those "
                "are forbidden by the bot's persona rules and the tool will refuse. "
                "Returns top results as title + URL + snippet."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Be specific — include disambiguating "
                            "context (e.g. '雷军 外貌 风格 公开形象', not just '雷军')."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10). Default 5.",
                    },
                },
                "required": ["query"],
            },
            handler=_web_search_handler,
        ))

    # ---------- entry ----------
    async def handle(self, msg: ParsedMessage) -> None:
        if not await allowlist.is_allowed(msg.group_id):
            log.debug("ignoring message from non-allowed group %s", msg.group_id)
            return
        if msg.self_id == msg.user_id:
            return
        if not msg.text and not msg.has_image and not msg.has_file:
            return

        # Log into group memory BEFORE any filtering — even messages we'll
        # ignore become part of the bot's awareness of the group. Awaited (not
        # background task) so downstream reads (proactive judge, /recap, chat
        # context injection) always see this row. Skip /commands; they're
        # bot-control plumbing, not conversation.
        if msg.text:
            record_text = msg.text
        elif msg.has_image:
            record_text = "[图片]"
        elif msg.has_file:
            record_text = f"[文件:{msg.files[0].name}]"
        else:
            record_text = ""
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
        # Count toward "messages since bot spoke" for proactive interjection.
        self._msgs_since_bot_spoke[msg.group_id] = (
            self._msgs_since_bot_spoke.get(msg.group_id, 0) + 1
        )

        if msg.has_image:
            asyncio.create_task(
                self._cache_image_to_disk(msg.group_id, msg.user_id, msg.image_urls[0])
            )

        # Per-group pause gate. When the group is paused, the bot stays silent
        # in all routes — including lesson learning and the LLM router —
        # except for three superuser-only escape hatches: `/start` (un-pause),
        # `/stop` (so repeat-stops still get an ack), and `/admin` (ops). Group
        # memory recording above already happened, so context survives the pause.
        if await allowlist.is_paused(msg.group_id):
            is_superuser = msg.user_id in CONFIG.superusers
            allowed_while_paused = (
                msg.is_command
                and msg.command in ("start", "stop", "admin")
                and is_superuser
                and msg.mentions(msg.self_id)  # @bot still required
            )
            if not allowed_while_paused:
                log.debug(
                    "group %s paused — skipping (cmd=%s user=%s)",
                    msg.group_id,
                    msg.command if msg.is_command else "-",
                    msg.user_id,
                )
                return

        # Unified lesson-learning: one LLM classifier decides whether the
        # message holds a behavior rule, a personal fact, a group agreement,
        # or a scheduled reminder — and persists it accordingly. Addressed
        # messages (@bot or nickname) bypass the keyword pre-filter so a
        # rule like "你说话简短点" still gets caught even without time/
        # preference vocabulary.
        if (
            CONFIG.lessons_enabled
            and self.lessons is not None
            and msg.text
            and not msg.is_command
        ):
            addressed = self._is_addressed(msg)
            asyncio.create_task(self._safe_learn_lesson(
                msg.group_id, msg.user_id, msg.text, addressed,
            ))

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
                # If the quote carried an image and this message didn't, decide
                # whether the user actually wants vision-level analysis of that
                # image. Cheap caption first, then router judges based on text
                # + caption. Stops the bot from auto-describing every quoted pic.
                if quoted.image_urls and not msg.image_urls:
                    await self._handle_quoted_image(msg, quoted.image_urls)
                # Pull files from quote so /file works for "reply + ask about
                # that previous upload". Don't double-add if already attached.
                if quoted.files and not msg.files:
                    msg.files = list(quoted.files)
                    log.info(
                        "reply-segment provided file(s); using %d from quoted msg",
                        len(quoted.files),
                    )

        # File ingestion: extract / transcribe and inject as system context.
        if CONFIG.file_ingest_enabled and msg.has_file:
            try:
                await self._ingest_files(msg)
            except Exception:
                log.exception("file ingest crashed")

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
        elif c == "file":
            await self._run_file_qa(msg, args or msg.text)
        elif c == "recap":
            await self._run_recap(msg, args)
        elif c == "recall":
            await self._run_recall(msg, args)
        elif c == "timewarp":
            await self._run_timewarp(msg, args)
        elif c == "remember":
            await self._run_remember(msg, args)
        elif c == "forget":
            # Alias for `/remember cancel <id>` so users can drop a lesson
            # without remembering the longer form.
            await self._run_remember(msg, f"cancel {args}".strip())
        elif c == "search":
            # Use args explicitly — falling back to msg.text would pass "/search"
            # as the query when the user types the bare command.
            await self._run_search(msg, args)
        elif c == "teach":
            await self._run_teach(msg, args)
        elif c == "dream":
            await self._run_dream(msg)
        elif c == "start":
            await self._run_start(msg)
        elif c == "stop":
            await self._run_stop(msg)
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
                "/admin status               本群今日路由用量\n"
                "/admin usage                所有群今日用量\n"
                "/admin reset_quota          清空今日额度\n"
                "/admin reset_memory <uid>   清空指定用户对话记忆\n"
                "/admin reset_memory all     清空本群所有人记忆\n"
                "/admin allow_group <gid>    允许新的群\n"
                "/admin disallow_group <gid> 禁用某群 (env 中的群无法移除)\n"
                "/admin list_groups          显示所有允许的群\n"
                "/admin report               立即推送一次日报\n"
                "/admin ping                 OneBot 连接状态与最近心跳\n"
                "/admin save_recap [day]     手动写入某天的长时记忆\n"
                "/admin lessons              查看本群学到的行为规则/事实/约定\n"
                "/admin forget_lesson <id>   忘掉一条 lesson\n"
                "（普通指令：/help；任何超级用户也可以 /dream 强制做梦）",
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
            paused = await allowlist.all_paused_groups()
            lines = ["允许的群（* = env 固定；⏸ = 被 /stop 暂停）："]
            for g in groups:
                marker = " *" if g in CONFIG.allowed_groups else ""
                marker += " ⏸" if g in paused else ""
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
        if sub == "lessons":
            if self.lessons is None:
                await self._reply(msg.group_id, "lessons 模块未启用")
                return
            rows = await self.lessons.list_all(msg.group_id, limit=50)
            if not rows:
                await self._reply(msg.group_id, "本群还没教过我什么~")
                return
            kind_label = {
                "rule": "规则", "fact": "事实",
                "agreement": "约定", "reminder": "提醒",
            }
            lines = [f"群 {msg.group_id} 学到的事项："]
            for _id, kind, content, imp, status, trigger_at, recurrence in rows:
                tag = "" if status == "active" else f" [{status}]"
                t_part = ""
                if trigger_at:
                    t_part = (
                        f" @{datetime.fromtimestamp(trigger_at).strftime('%m-%d %H:%M')}"
                    )
                if recurrence:
                    t_part += f" ({recurrence})"
                label = kind_label.get(kind, kind)
                lines.append(
                    f"#{_id} [{label}] ({imp:.2f}){tag}{t_part} {content}"
                )
            lines.append("\n忘掉: /admin forget_lesson <id>")
            await self._reply(msg.group_id, "\n".join(lines))
            return
        if sub == "forget_lesson":
            if self.lessons is None:
                await self._reply(msg.group_id, "lessons 模块未启用")
                return
            if not rest.isdigit():
                await self._reply(msg.group_id, "用法: /admin forget_lesson <id>")
                return
            ok = await self.lessons.cancel(int(rest), msg.group_id)
            await self._reply(
                msg.group_id,
                f"已忘记 #{rest}" if ok else "没找到这条（或已经忘了）",
            )
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
            has_file=msg.has_file,
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

    async def _run_search(self, msg: ParsedMessage, query: str) -> None:
        """`/search <query>` — force a web search and present grounded results.

        Pipes the user's query through the same web_search tool the LLM uses
        internally, then sends the formatted query+results into deepseek_chat
        so the persona / lessons / sensitive-content rules all still apply.
        """
        query = (query or "").strip()
        if not query:
            await self._reply(msg.group_id, "搜啥呀，告诉我关键词~")
            return
        if _query_is_blocked(query):
            log.info("/search blocked sensitive query: %r", query[:120])
            await self._human_send(msg.group_id, "这个我不太懂呢，我们聊点别的吧")
            return
        if self.web_search is None:
            await self._human_send(
                msg.group_id,
                "联网搜索还没配置呢（管理员要去填 TAVILY_API_KEY）",
            )
            return
        try:
            results = await self.web_search.search(
                query, max_results=CONFIG.web_search_max_results,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("/search call failed: %s", e)
            await self._reply(msg.group_id, "搜索失败了，过会儿再试~")
            return
        from providers.web_search import format_results
        block = format_results(results, max_chars=1800)
        # Hand the results to the chat path so the bot picks them up in
        # character. Strip URLs from the spoken summary — persona handles that.
        prompt = (
            f"用户让我联网搜索 '{query}'。下面是搜到的内容：\n\n{block}\n\n"
            "请基于上述资料用一两句话回答用户。不要把所有 URL 都念出来；"
            "如果资料里没有有用信息，就直说没查到。"
        )
        await self._run_deepseek_chat(msg, prompt)

    async def _run_teach(self, msg: ParsedMessage, rule: str) -> None:
        """`/teach <rule>` — explicitly save a behavioral rule / fact /
        agreement / reminder. Bypasses the keyword pre-filter (since the user
        is intentionally teaching), but still runs the LLM classifier so the
        right `kind` and trigger_at get filled in."""
        rule = (rule or "").strip()
        if not rule:
            await self._reply(
                msg.group_id,
                "用法: /teach <规则/事实/约定>，比如 /teach 群里有人发666你也跟一个",
            )
            return
        if not CONFIG.lessons_enabled or self.lessons is None:
            await self._reply(msg.group_id, "功能注入没开~")
            return
        try:
            row_id = await self.lessons.maybe_learn(
                group_id=msg.group_id, user_id=msg.user_id,
                text=rule, addressed=True,
            )
        except Exception:
            log.exception("/teach lesson save crashed")
            await self._reply(msg.group_id, "记不住，过会再说~")
            return
        if row_id:
            await self._human_send(msg.group_id, f"好的，记下了（#{row_id}）")
        else:
            await self._human_send(
                msg.group_id,
                "这条没存下来（看起来不像一条要长期遵守的规则）",
            )

    async def _run_start(self, msg: ParsedMessage) -> None:
        """`/start` — un-pause the bot in this group. Super-user only."""
        if msg.user_id not in CONFIG.superusers:
            await self._reply(msg.group_id, "/start 仅限超级用户使用~")
            return
        changed = await allowlist.resume(msg.group_id)
        if changed:
            await self._human_send(msg.group_id, "好的，我回来啦~")
            log.info("group %s un-paused by user %s", msg.group_id, msg.user_id)
        else:
            await self._human_send(msg.group_id, "我本来就在群里呀")

    async def _run_stop(self, msg: ParsedMessage) -> None:
        """`/stop` — pause the bot in this group. Super-user only.

        Paused groups keep recording group_memory (so context survives) but
        the bot won't reply or run any LLM route until a `/start` from a
        super-user resumes it.
        """
        if msg.user_id not in CONFIG.superusers:
            await self._reply(msg.group_id, "/stop 仅限超级用户使用~")
            return
        changed = await allowlist.pause(msg.group_id, msg.user_id)
        if changed:
            await self._human_send(
                msg.group_id,
                "好，我先安静一会儿，要叫我回来用 /start",
            )
            log.info("group %s paused by user %s", msg.group_id, msg.user_id)
        else:
            await self._human_send(msg.group_id, "我已经在休息啦")

    async def _run_dream(self, msg: ParsedMessage) -> None:
        """`/dream` — force the overnight dream routine to run now. Super-user
        only since it consumes a deepseek call and is mainly a debugging aid."""
        if msg.user_id not in CONFIG.superusers:
            await self._reply(msg.group_id, "/dream 仅限超级用户使用~")
            return
        if self.long_memory is None:
            await self._reply(msg.group_id, "长时记忆没开，没法编梦~")
            return
        recaps = await self.long_memory.recent(msg.group_id, days=5)
        if not recaps:
            await self._human_send(
                msg.group_id, "本群最近没什么记忆，做不出梦来",
            )
            return
        await self.send_dream(msg.group_id, recaps)

    async def _run_file_qa(self, msg: ParsedMessage, prompt: str) -> None:
        """/file <question> — answer based on file content already injected
        into msg.text by _ingest_files. If no file was attached, ask for one."""
        if not msg.has_file:
            await self._reply(msg.group_id, "请先发文件，再用 /file 提问哦")
            return
        # _ingest_files already ran upstream and prepended file content to
        # msg.text. Just route through the standard text path so persona,
        # lessons, group context, etc. all apply.
        question = prompt.strip() or "请总结这份文件的主要内容"
        await self._run_deepseek_chat(msg, question)

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

    async def _handle_quoted_image(
        self, msg: ParsedMessage, image_urls: List[str],
    ) -> None:
        """Decide what to do with an image attached to a quoted message.

        Default: caption it (cheap vision call) and inject the caption as text
        context. Then strip the image from msg.image_urls so the router judges
        intent from text alone — preventing the bot from auto-describing every
        image someone happens to reply to.

        Bypass cases (image is kept as a true vision input):
          * /vision and /edit commands — explicit ask.
          * QUOTED_IMAGE_INTENT_GATE disabled by config.
        """
        if not image_urls:
            return
        # Always cache; downstream (/vision /edit) needs the file on disk.
        asyncio.create_task(
            self._cache_image_to_disk(msg.group_id, msg.user_id, image_urls[0])
        )
        bypass = (
            not CONFIG.quoted_image_intent_gate
            or (msg.is_command and msg.command in ("vision", "edit"))
        )
        if bypass:
            msg.image_urls = list(image_urls)
            log.info(
                "quoted image: bypass gate (cmd=%s gate=%s) — passing to vision",
                msg.command if msg.is_command else "-",
                CONFIG.quoted_image_intent_gate,
            )
            return
        if self.openai is None:
            # No vision provider — fall back to treating the image as input.
            msg.image_urls = list(image_urls)
            return
        caption = await self._caption_quoted_image(msg.group_id, image_urls[0])
        if caption:
            msg.text = (
                f"[被引用的图片大致内容: {caption}]\n{msg.text}"
                if msg.text else
                f"[被引用的图片大致内容: {caption}]"
            ).strip()
            log.info(
                "quoted image: captioned (%d chars), routing on text+caption",
                len(caption),
            )
        else:
            # Caption failed — let routing see the image so vision can still
            # answer if the text genuinely needs it.
            msg.image_urls = list(image_urls)

    async def _caption_quoted_image(
        self, group_id: int, image_url: str,
    ) -> Optional[str]:
        """Low-cost one-line caption. Returns the caption or None on failure."""
        if self.openai is None:
            return None
        ok, reason = await self.quota.check("auto_vision", group_id, 0)
        if not ok:
            log.info("quoted-image caption skipped: %s", reason)
            return None
        try:
            raw = await self._download(image_url)
            small = downscale_to_max(raw, CONFIG.max_vision_input_size)
            data_uri = to_data_uri(small)
            reply = await self.openai.vision(
                "用中文一句话概括这张图（20字以内，只输出概括本身，不加标点）",
                [data_uri],
                max_tokens=60,
            )
        except Exception:
            log.exception("quoted-image caption call failed")
            return None
        text = (reply.text or "").strip().split("\n")[0][:40]
        if not text:
            return None
        await self.quota.consume("auto_vision", group_id, 0)
        return text

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

    # ---------- file ingestion ----------
    async def _ingest_files(self, msg: ParsedMessage) -> None:
        """Fetch each attached file, extract / transcribe / sample, and prepend
        the result to msg.text so downstream routing treats it as context.

        Mirrors the image flow: cheap, best-effort, gracefully degrades if
        optional deps (pypdf, python-docx, ffmpeg) are missing.
        """
        if not msg.files:
            return
        # Cap to a few files per message to avoid runaway processing.
        files = msg.files[:3]
        blocks: List[str] = []
        for f in files:
            try:
                block = await self._ingest_one_file(msg.group_id, msg.user_id, f)
            except Exception:
                log.exception("ingest_one_file crashed for %s", f.name)
                block = f"[文件:{f.name}] 读取时出错了"
            if block:
                blocks.append(block)
        if not blocks:
            return
        prefix = "\n\n".join(blocks)
        msg.text = (prefix + "\n\n" + msg.text).strip() if msg.text else prefix

    async def _ingest_one_file(
        self, group_id: int, user_id: int, f: AttachedFile,
    ) -> Optional[str]:
        url = f.url
        if not url and f.file_id and self.fetch_file_url is not None:
            try:
                url = await self.fetch_file_url(group_id, f.file_id) or ""
            except Exception:
                log.exception("fetch_file_url failed for %s", f.file_id)
                url = ""
        if not url:
            return f"[文件:{f.name}]（拿不到下载链接）"

        # Size pre-check via Content-Length (best-effort).
        max_bytes = CONFIG.file_ingest_max_mb * 1024 * 1024
        if 0 < max_bytes and f.size and f.size > max_bytes:
            return f"[文件:{f.name}] 太大了（{f.size // (1024*1024)}MB > {CONFIG.file_ingest_max_mb}MB）"
        try:
            data = await self._download(url)
        except httpx.HTTPError as e:
            log.warning("file download failed %s: %s", f.name, e)
            return f"[文件:{f.name}]（下载失败）"
        if max_bytes and len(data) > max_bytes:
            return f"[文件:{f.name}] 太大了，超过 {CONFIG.file_ingest_max_mb}MB 限制"
        kind = file_utils.classify(f.name)
        log.info(
            "file ingest: name=%r kind=%s bytes=%d group=%s",
            f.name, kind, len(data), group_id,
        )
        if kind == "audio":
            return await self._ingest_audio(f, data)
        if kind == "video":
            return await self._ingest_video(group_id, f, data)
        # text / code / pdf / docx / unsupported
        extr = file_utils.extract_text(
            f.name, data, max_chars=CONFIG.file_ingest_max_chars,
        )
        return file_utils.format_for_prompt(extr)

    async def _ingest_audio(self, f: AttachedFile, data: bytes) -> str:
        if not CONFIG.file_audio_transcribe or self.openai is None:
            return f"[音频文件:{f.name}]（未启用语音转录）"
        ext = Path(f.name).suffix.lower() or ".bin"
        down = file_utils.downsample_audio(data, ext)
        payload = down if down else data
        try:
            reply = await self.openai.transcribe(
                payload, filename=file_utils.safe_name(f.name),
            )
        except ProviderError as e:
            log.warning("audio transcribe failed: %s", e)
            return f"[音频文件:{f.name}]（转录失败）"
        text = (reply.text or "").strip()
        if not text:
            return f"[音频文件:{f.name}]（没识别出内容）"
        if len(text) > CONFIG.file_ingest_max_chars:
            text = text[: CONFIG.file_ingest_max_chars] + "…（已截断）"
        return f"[音频文件 {f.name} 的语音内容]\n{text}"

    async def _ingest_video(
        self, group_id: int, f: AttachedFile, data: bytes,
    ) -> str:
        if self.openai is None:
            return f"[视频文件:{f.name}]（未启用视觉/转录）"
        ext = Path(f.name).suffix.lower() or ".mp4"
        sections: List[str] = []
        # Audio track → Whisper.
        if CONFIG.file_audio_transcribe:
            audio_bytes = file_utils.downsample_audio(data, ext)
            if audio_bytes:
                try:
                    reply = await self.openai.transcribe(
                        audio_bytes, filename="audio.mp3",
                    )
                    if reply.text.strip():
                        snippet = reply.text.strip()
                        if len(snippet) > 4000:
                            snippet = snippet[:4000] + "…（已截断）"
                        sections.append(f"语音内容：\n{snippet}")
                except ProviderError as e:
                    log.warning("video audio transcribe failed: %s", e)
        # Frames → vision captions.
        frame_count = CONFIG.file_video_frame_count
        if frame_count > 0:
            frames = file_utils.extract_video_frames(data, ext, frame_count)
            captions: List[str] = []
            for i, frame in enumerate(frames):
                ok, _r = await self.quota.check("auto_vision", group_id, 0)
                if not ok:
                    break
                try:
                    small = downscale_to_max(frame, CONFIG.max_vision_input_size)
                    data_uri = to_data_uri(small)
                    cap = await self.openai.vision(
                        "用中文一句话描述这张视频帧（20字以内）",
                        [data_uri],
                        max_tokens=60,
                    )
                except Exception:
                    log.exception("video frame caption failed")
                    continue
                await self.quota.consume("auto_vision", group_id, 0)
                text = (cap.text or "").strip().split("\n")[0][:40]
                if text:
                    captions.append(f"第{i + 1}帧：{text}")
            if captions:
                sections.append("画面要点：\n" + "\n".join(captions))
        if not sections:
            return f"[视频文件:{f.name}]（无法提取内容；可能缺少 ffmpeg）"
        return f"[视频文件 {f.name}]\n" + "\n\n".join(sections)

    def _is_addressed(self, msg: ParsedMessage) -> bool:
        """True if the message clearly targets the bot — @-mention OR the bot's
        nickname appears in the text. Used to gate lesson-learning."""
        if msg.mentions(msg.self_id):
            return True
        nick = CONFIG.bot_nickname
        return bool(nick) and nick in (msg.text or "")

    async def _safe_learn_lesson(
        self, group_id: int, user_id: int, text: str, addressed: bool,
    ) -> None:
        if self.lessons is None:
            return
        try:
            await self.lessons.maybe_learn(
                group_id=group_id, user_id=user_id, text=text,
                addressed=addressed,
            )
        except Exception:
            log.exception("lesson learn failed")

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
        # Unified lessons: rules + facts + agreements + reminders injected
        # right after persona so they shape every response. Personal items
        # (subject_user_id=speaker) rank above group-wide ones.
        if CONFIG.lessons_enabled and self.lessons is not None:
            try:
                active = await self.lessons.active_for_user(
                    msg.group_id, msg.user_id,
                    limit=CONFIG.lessons_inject_limit,
                )
            except Exception:
                log.exception("lessons recall failed")
                active = []
            block = Lessons.format_for_prompt(active, speaker_user_id=msg.user_id)
            if block:
                messages.append(ChatMessage(role="system", content=block))
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
        for role, content in history:
            messages.append(ChatMessage(role=role, content=content))
        messages.append(ChatMessage(role="user", content=prompt))

        try:
            if CONFIG.tool_use_enabled and not self.tools.is_empty():
                ctx = ToolContext(group_id=msg.group_id, user_id=msg.user_id)
                reply = await run_with_tools(
                    provider=provider, messages=messages, registry=self.tools,
                    ctx=ctx, model=model, max_tokens=600,
                    max_hops=CONFIG.tool_use_max_hops,
                )
            else:
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
    # Split on:
    #   1. Chinese sentence-end punct (。？！；) — split happens after the punct.
    #   2. English "!" or "?" followed by whitespace.
    #   3. English "." followed by whitespace, but NOT when the preceding char
    #      is a digit (keeps decimals like "3.14" intact) and NOT before more
    #      digits (handles "v1.2.3", "192.168.0.1"). Common abbreviations
    #      ("Mr.", "Dr.") may still split — acceptable for chat output.
    #   4. Paragraph breaks (one or more blank lines).
    _SENT_SPLIT_RE = re.compile(
        r"(?<=[。？！；])"
        r"|(?<=[!?])\s+"
        r"|(?<=[^\d]\.)\s+(?!\d)"
        r"|(?:\n\s*\n)+"
    )
    # Drop trailing English periods and Chinese 句号 from each chunk per
    # product spec ("句末不要加句号"). Other terminals (!?？！；…) are kept
    # so emotional punctuation survives.
    _TRAILING_PERIOD_RE = re.compile(r"[.。]+\s*$")

    @classmethod
    def _split_human_chunks(cls, text: str, max_chunks: int) -> List[str]:
        """Split into 1..max_chunks short messages on sentence boundaries.

        Decimals ("3.14") and version strings ("v1.2") are never broken.
        Trailing periods on each chunk are stripped (English '.' and 句号)."""
        text = text.strip()
        if not text:
            return []
        raw_parts = cls._SENT_SPLIT_RE.split(text)
        parts: List[str] = []
        for p in raw_parts:
            if not p:
                continue
            stripped = p.strip()
            if not stripped:
                continue
            stripped = cls._TRAILING_PERIOD_RE.sub("", stripped).rstrip()
            if stripped:
                parts.append(stripped)
        if not parts:
            return [cls._TRAILING_PERIOD_RE.sub("", text).rstrip()]
        if len(parts) <= max_chunks:
            return parts
        # More sentences than chunks — distribute evenly, re-stripping the
        # joined boundary so we don't reintroduce trailing periods.
        per = -(-len(parts) // max_chunks)  # ceil div
        grouped: List[str] = []
        for i in range(0, len(parts), per):
            joined = "".join(parts[i:i + per])
            joined = cls._TRAILING_PERIOD_RE.sub("", joined).rstrip()
            if joined:
                grouped.append(joined)
        return grouped

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
            single = self._TRAILING_PERIOD_RE.sub("", text).rstrip()
            await self._reply(group_id, single)
            if log_to_memory:
                asyncio.create_task(self.group_memory.append(
                    group_id, 0, CONFIG.bot_nickname, single,
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

    # ---------- /timewarp (nostalgic recall) ----------
    @staticmethod
    def _parse_timewarp(args: str) -> Optional[Tuple[float, float, str]]:
        """Parse a past-time expression into (start_ts, end_ts, label).

        Returns None for unrecognised input. Window is a few days around the
        target date so the LLM has more than one recap to riff on.
        """
        a = args.strip().lower()
        now = datetime.now()
        # YYYY-MM-DD → ±3 days.
        if re.match(r"^\d{4}-\d{2}-\d{2}$", a):
            try:
                d = datetime.strptime(a, "%Y-%m-%d")
            except ValueError:
                return None
            return (
                (d - timedelta(days=3)).timestamp(),
                (d + timedelta(days=4)).timestamp(),
                a,
            )
        # YYYY-MM → whole month.
        if re.match(r"^\d{4}-\d{2}$", a):
            try:
                d = datetime.strptime(a + "-01", "%Y-%m-%d")
            except ValueError:
                return None
            nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
            return d.timestamp(), nxt.timestamp(), a
        # Shorthand phrases. Default (empty) = a year ago.
        days: Optional[int] = None
        label = ""
        if not a or a in ("一年前", "去年", "1 year ago"):
            days, label = 365, "一年前"
        elif "半年" in a:
            days, label = 180, "半年前"
        elif "三个月" in a or "3个月" in a:
            days, label = 90, "三个月前"
        elif "两个月" in a or "2个月" in a:
            days, label = 60, "两个月前"
        elif "一个月" in a or a in ("上月", "上个月", "last month"):
            days, label = 30, "一个月前"
        elif "一周" in a or a in ("上周", "上礼拜", "last week"):
            days, label = 7, "上周"
        else:
            m = re.match(r"^(\d+)\s*天前?$", a)
            if m:
                days = int(m.group(1))
                label = f"{days} 天前"
            else:
                m = re.match(r"^(\d+)\s*年前?$", a)
                if m:
                    days = int(m.group(1)) * 365
                    label = f"{m.group(1)} 年前"
                else:
                    m = re.match(r"^(\d+)\s*个?月前?$", a)
                    if m:
                        days = int(m.group(1)) * 30
                        label = f"{m.group(1)} 个月前"
        if days is None:
            return None
        target = now - timedelta(days=days)
        # Use a 7-day window centered on the target so we capture more recaps.
        start = target - timedelta(days=3)
        end = target + timedelta(days=4)
        return start.timestamp(), end.timestamp(), label

    async def _run_timewarp(self, msg: ParsedMessage, args: str) -> None:
        """`/timewarp <时间>` — bot writes a short nostalgic riff based on
        the daily_recaps stored for that period."""
        if self.long_memory is None:
            await self._reply(msg.group_id, "长时记忆没开~")
            return
        parsed = self._parse_timewarp(args)
        if parsed is None:
            await self._reply(
                msg.group_id,
                "用法: /timewarp 一年前 | 半年前 | 三个月前 | 上个月 | 上周 | "
                "YYYY-MM | YYYY-MM-DD",
            )
            return
        start_ts, end_ts, label = parsed
        store = await Storage.get()
        # Pull a wide window of recaps and filter client-side. Recaps are
        # short (~100 chars each) so even fetching the recent 400 is cheap.
        all_rows = await store.daily_recap_recent(msg.group_id, 400)
        in_range: List[Tuple[str, str]] = []
        for day, summary in all_rows:
            try:
                day_ts = datetime.strptime(day, "%Y-%m-%d").timestamp()
            except ValueError:
                continue
            if start_ts <= day_ts < end_ts:
                in_range.append((day, summary))
        if not in_range:
            await self._human_send(
                msg.group_id, f"{label}那段时间…我好像没什么记忆"
            )
            return
        in_range.sort(key=lambda r: r[0])
        bullet = "\n".join(f"- {d}: {s}" for d, s in in_range[:14])
        prompt = (
            f"你是 {CONFIG.bot_nickname}，正在怀念群里 {label} 的日子。\n"
            f"那段时间群里聊过这些（按天序）：\n{bullet}\n\n"
            "请以第一人称写一段 100-150 字的怀旧短文：\n"
            "- 开头用'我记得那时候…'或类似的口吻\n"
            "- 提一些具体的话题/场景，但不要点名说谁\n"
            "- 语气温柔、略带感慨\n"
            "- 不要 markdown，不要 emoji，不要分点列表，不要署名"
        )
        try:
            reply = await self.deepseek.chat(
                [
                    ChatMessage(role="system", content=load_persona()),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.7,
                max_tokens=400,
            )
        except ProviderError:
            log.exception("timewarp deepseek failed")
            await self._reply(msg.group_id, ERROR_MSG)
            return
        await self._human_send(msg.group_id, reply.text)

    # ---------- bot dreams (overnight ambient post) ----------
    async def maybe_send_dream(self) -> int:
        """Pick one allowed group with recent activity and post a dream.
        Returns 1 if a dream was sent, else 0."""
        if self.long_memory is None:
            return 0
        groups = sorted(await allowlist.all_allowed_groups())
        if not groups:
            return 0
        random.shuffle(groups)
        for gid in groups:
            recaps = await self.long_memory.recent(gid, days=5)
            if recaps:
                await self.send_dream(gid, recaps)
                return 1
        return 0

    async def send_dream(
        self, group_id: int, recaps: List[Tuple[str, str]],
    ) -> None:
        """Generate a 1-2 sentence "I just had a dream..." message rooted
        in recent group themes, and post it via the normal human-send path
        so it lands in group_memory."""
        bullet = "\n".join(f"- {d}: {s}" for d, s in reversed(recaps))
        now = datetime.now()
        system = (
            f"你叫 {CONFIG.bot_nickname}。现在是凌晨 {now.strftime('%H:%M')}，"
            "你刚做了个梦，想发到 QQ 群里。"
        )
        user = (
            "群里最近几天的话题（按天序，旧 → 新）：\n" + bullet + "\n\n"
            "请用你自己的口吻写一条群消息描述这个梦。要求：\n"
            "- 1-2 句话，不超过 60 字\n"
            "- 梦的内容超现实、有点荒诞，但能隐约看到群里最近聊过的某个东西\n"
            "- 用'我刚做了个梦…'或'刚梦到…'之类的开头\n"
            "- 不要解释、不要 emoji、不要 markdown\n"
            "- 不要点名说谁，用'有人'替代"
        )
        try:
            reply = await self.deepseek.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                temperature=1.0,
                max_tokens=120,
            )
        except ProviderError:
            log.warning("dream generation call failed")
            return
        text = (reply.text or "").strip()
        if not text:
            return
        log.info("dream group=%s text=%r", group_id, text[:120])
        await self._human_send(group_id, text)

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
        """List / cancel pending lessons (reminders + facts + agreements +
        rules). Now backed by the unified `lessons` table."""
        if not CONFIG.lessons_enabled or self.lessons is None:
            await self._reply(msg.group_id, "功能注入未启用~")
            return
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""
        if sub == "cancel":
            if not rest.isdigit():
                await self._reply(msg.group_id, "用法: /remember cancel <id>")
                return
            ok = await self.lessons.cancel(int(rest), msg.group_id)
            await self._reply(
                msg.group_id,
                "已取消~" if ok else "找不到这条，或者已经取消/触发过了",
            )
            return
        if sub in ("", "list"):
            rows = await self.lessons.list_pending(
                msg.group_id, msg.user_id, limit=15,
            )
            if not rows:
                await self._human_send(msg.group_id, "我现在没记着什么待办呢")
                return
            lines = ["我记着的事："]
            now_ts = time.time()
            for item_id, kind, subj, content, trigger, recurrence in rows:
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
                lines.append(f"#{item_id} [{kind}]{who} {content}{t_part}")
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
        Reads from the unified `lessons` table; rows with `trigger_at <= now`
        and status='active' are fired (and rescheduled if recurring)."""
        if not CONFIG.lessons_enabled or self.lessons is None:
            return 0
        try:
            due = await self.lessons.due_reminders()
        except Exception:
            log.exception("lesson due_reminders query failed")
            return 0
        fired = 0
        for item_id, group_id, subject_user_id, content, _kind, recurrence in due:
            if not await allowlist.is_allowed(group_id):
                # Group de-allowed since the lesson was created — drop the row.
                await self.lessons.cancel(item_id, group_id)
                continue
            try:
                # Compose the reminder body. If we know who it's for, @-mention.
                if subject_user_id and subject_user_id > 0:
                    body = f"[CQ:at,qq={subject_user_id}] {content}"
                else:
                    body = content
                await self.send_text(group_id, body)
                await self.lessons.mark_fired(item_id, recurrence)
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
        # Unified lessons maintenance: expire stale rows + LLM-driven dedup
        # of duplicates / contradictions. Replaces the old important_memory
        # maintenance pass entirely (and any lessons-v1 dedup that lived here
        # before the merge).
        if (
            CONFIG.lessons_enabled
            and self.lessons is not None
            and groups
        ):
            try:
                await self.lessons.maintenance_pass(groups)
            except Exception:
                log.exception("maintenance: lessons pass failed")

    async def aclose(self) -> None:
        await self._http.aclose()
