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
{{"r":"<route>"}}

The bot's nickname is: {CONFIG.bot_nickname}

Routes:
- chat: directed at bot, normal chat / translation / casual answer
- think: directed at bot, harder reasoning or code
- gpt: user explicitly wants GPT / top quality
- vision: directed at bot AND explicitly asks to look at / describe / OCR / answer-based-on an image
- image: directed at bot AND asks to GENERATE an image
- edit: directed at bot AND asks to MODIFY an image
- skip: NOT addressed to bot — the default for almost anything
- no: unsafe / illegal / impossible

A message is "directed at bot" ONLY IF AT LEAST ONE of these holds:
1. The text contains the tag "[at_bot=true]" (system-injected when the user @-mentions the bot)
2. The text contains the bot's nickname "{CONFIG.bot_nickname}"
3. The message is unambiguously a request to the bot (e.g. greets the bot by role)

If none of those hold → skip.

CRITICAL — image / vision rules:
- A bare image with empty or trivial text → skip (the user just shared a picture, not asked the bot anything).
- "[image attached]" alone is NOT a vision request. Vision requires the user to ASK to look at the image ("看看这张图"/"这是啥"/"翻译一下图里的字").
- Conceptual talk about images ("你能看图吗", "你是怎么识别的") → chat, NOT vision.

Other → skip rules:
- Pure reactions (嗯/哈/啊/666/[emoji-only]/[punctuation-only]) → skip
- Statements, observations, banter between other users → skip
- Generic questions thrown into the group without addressing the bot → skip

When in doubt → skip."""


ROUTER_SYSTEM_PROMPT = _build_system_prompt()


@dataclass
class RouteDecision:
    route: str           # internal name (e.g., "deepseek_chat", "skip")
    confidence: float
    reason: str
    normalized_prompt: str


_JSON_RE = re.compile(r"\{[^}]*\}", re.DOTALL)


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

    return RouteDecision(
        route=route, confidence=1.0, reason=short, normalized_prompt=fallback_prompt
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
    ) -> RouteDecision:
        # Cheap pre-filter: a literal empty message can never be addressed at us.
        if not user_text.strip() and not has_image:
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
        log.info("router %s (reason=%s, in_tokens=%s, out_tokens=%s)",
                 decision.route, decision.reason,
                 reply.usage.get("prompt_tokens") if reply.usage else "?",
                 reply.usage.get("completion_tokens") if reply.usage else "?")
        return decision
