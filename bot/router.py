"""LLM router.

DeepSeek classifies each non-command group message into one of a few
backends. Output is **strict JSON** with a single short field `r`:

    {"r":"chat" | "think" | "gpt" | "vision" | "image" | "edit" | "skip" | "no"}

We map the short codes to the internal route names used elsewhere in the bot.
On any failure (network / malformed JSON / unknown code) we default to `skip`
so the bot stays quiet rather than hallucinating a reply.

Why so terse:
- The system prompt is sent on EVERY group message → tokens add up fast.
  Keeping it stable also lets DeepSeek's prefix cache kick in.
- The output is one field → output tokens stay near zero (~5-8 each call).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from bot.logger import get_logger
from config import CONFIG
from providers.base import ChatMessage
from providers.deepseek import DeepSeekProvider

log = get_logger(__name__)

# Short-code → internal route name. The handler dispatches on the internal name.
_ROUTE_MAP = {
    "chat":   "deepseek_chat",
    "think":  "deepseek_think",
    "gpt":    "openai_text",
    "vision": "openai_vision",
    "image":  "openai_image",
    "edit":   "openai_image_edit",
    "skip":   "skip",
    "no":     "reject",
}

ROUTES = set(_ROUTE_MAP.values())

# Built once at import. Stable string → DeepSeek prefix cache stays warm.
def _build_system_prompt() -> str:
    return f"""You classify a single QQ group message. Output STRICT JSON only:
{{"r":"<route>","t":"<tier>"}}

The bot's nickname is: {CONFIG.bot_nickname}

Routes:
- chat: normal chat / translation / casual answer
- think: harder reasoning or code
- gpt: user explicitly wants GPT / top quality
- vision: explicit request to look at / describe / OCR / answer-based-on an image
- image: explicit request to GENERATE an image
- edit: explicit request to MODIFY an image
- skip: bot definitely shouldn't speak (use sparingly — see list below)
- no: unsafe / illegal / impossible

Tier (downstream code uses this to throttle non-addressed replies):
- high: the bot SHOULD plausibly engage. Use when AT LEAST ONE holds:
   1. Text contains "[at_bot=true]" (user @-mentioned the bot)
   2. Text contains the bot's nickname "{CONFIG.bot_nickname}"
   3. Message is clearly addressed to the bot (greets it, asks for opinion/help by role)
   4. Sincere question or "ask the group" call that a friend would naturally answer
      ("早饭吃啥", "今天天气咋样", "有人懂xxx吗", "你们觉得xxx好吃吗")
   5. Natural opening to riff on: interesting topic, story hook, casual
      confession that invites a reaction ("我今天好累啊", "刚发现一家新店")
- low: borderline / probably not engaged with — pure reactions ("666"/"嗯"/"哈"/
   "23333"), banter between two specific users (含 @他人、紧接对话链), pure
   statements / complaints with no question and no riff hook, generic chit-chat
   the bot has no reason to interrupt. Downstream code will reply to these
   very rarely; classify here rather than skipping so the bot can still occasionally
   chime in like a real groupmate.
- omit (or "high") for skip / no.

skip ONLY for:
- Explicit NSFW / illegal content
- Ads, spam, copypasta promoting external products
- Deeply personal / political / argumentative content the bot has no business in
- Messages that are clearly addressed to a specific OTHER user (e.g. "@张三 ...")
  with no overlap to the wider group

CRITICAL — image / vision rules:
- A bare image with empty or trivial text → skip (just sharing a picture).
- "[image attached]" alone is NOT a vision request. Vision requires an explicit
  ask ("看看这张图"/"这是啥"/"翻译一下图里的字").
- Conceptual talk about images ("你能看图吗", "你是怎么识别的") → chat, NOT vision.

Examples:
- "[at_bot=true] 翻译一下 hello" → {{"r":"chat","t":"high"}}
- "{CONFIG.bot_nickname}你早饭吃啥" → {{"r":"chat","t":"high"}}
- "有人知道这个咋调吗" → {{"r":"chat","t":"high"}}
- "今天好累" → {{"r":"chat","t":"high"}}
- "666" → {{"r":"chat","t":"low"}}
- "@张三 你说的那个" → {{"r":"chat","t":"low"}}
- "[image attached]" → {{"r":"skip"}}"""


ROUTER_SYSTEM_PROMPT = _build_system_prompt()


@dataclass
class RouteDecision:
    route: str           # internal name (e.g., "deepseek_chat", "skip")
    confidence: float
    reason: str
    normalized_prompt: str
    tier: str = "high"   # "high" | "low" — used to pick the ambient gate probability


_JSON_RE = re.compile(r"\{[^}]*\}", re.DOTALL)
_VALID_TIERS = {"high", "low"}


def _coerce(text: str, fallback_prompt: str) -> RouteDecision:
    """Best-effort parse. Any failure → skip."""
    m = _JSON_RE.search(text)
    payload = m.group(0) if m else text
    try:
        obj = json.loads(payload)
    except Exception as e:
        log.warning("router parse failed → skip: %s", e)
        return RouteDecision("skip", 0.0, "parse_fail", fallback_prompt)

    short = str(obj.get("r", "")).strip().lower()
    route = _ROUTE_MAP.get(short)
    if route is None:
        log.warning("router unknown code %r → skip", short)
        return RouteDecision("skip", 0.0, "unknown_code", fallback_prompt)

    tier_raw = str(obj.get("t", "")).strip().lower()
    tier = tier_raw if tier_raw in _VALID_TIERS else "low"  # default conservative

    return RouteDecision(
        route=route, confidence=1.0, reason=short,
        normalized_prompt=fallback_prompt, tier=tier,
    )


class Router:
    def __init__(self, deepseek: DeepSeekProvider) -> None:
        self._deepseek = deepseek

    async def decide(
        self,
        user_text: str,
        *,
        has_image: bool,
        was_at_bot: bool = False,
        has_file: bool = False,
    ) -> RouteDecision:
        # Cheap pre-filter: a literal empty message can never be addressed at us.
        if not user_text.strip() and not has_image and not has_file:
            return RouteDecision("skip", 1.0, "empty", user_text)

        # Hard rule: a bare image (no text, no @) is just someone sharing a
        # picture — never auto-trigger vision. Saves a router call AND the
        # vision call that the LLM might otherwise vote for.
        if has_image and not was_at_bot:
            stripped = user_text.strip()
            if (
                not stripped
                or (len(stripped) < 6 and CONFIG.bot_nickname not in stripped)
            ):
                log.info("router pre-skip: bare image without bot-direction")
                return RouteDecision("skip", 1.0, "bare_image", user_text)

        # Same rule for files: bare uploads (no text, no @) are just sharing.
        if has_file and not was_at_bot:
            stripped = user_text.strip()
            if (
                not stripped
                or (len(stripped) < 6 and CONFIG.bot_nickname not in stripped)
            ):
                log.info("router pre-skip: bare file without bot-direction")
                return RouteDecision("skip", 1.0, "bare_file", user_text)

        # Build the user-side prompt with optional system tags. The system
        # prompt stays byte-stable for cache hits; only this user line changes.
        tags = []
        if was_at_bot:
            tags.append("[at_bot=true]")
        if has_image:
            tags.append("[image attached]")
        prompt = (" ".join(tags) + " " + user_text).strip() if tags else user_text

        messages = [
            ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            reply = await self._deepseek.chat(
                messages,
                temperature=0.0,
                max_tokens=30,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            log.warning("router call failed → skip: %s", e)
            return RouteDecision("skip", 0.0, "router_error", user_text)

        decision = _coerce(reply.text, user_text)
        log.info("router %s tier=%s (reason=%s, in_tokens=%s, out_tokens=%s)",
                 decision.route, decision.tier, decision.reason,
                 reply.usage.get("prompt_tokens") if reply.usage else "?",
                 reply.usage.get("completion_tokens") if reply.usage else "?")
        return decision
