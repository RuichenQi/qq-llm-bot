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
- chat: bot should respond, normal chat / translation / casual answer
- think: bot should respond, harder reasoning or code
- gpt: user explicitly wants GPT / top quality
- vision: explicit request to look at / describe / OCR / answer-based-on an image
- image: explicit request to GENERATE an image
- edit: explicit request to MODIFY an image
- skip: bot shouldn't speak
- no: unsafe / illegal / impossible

The bot may respond (any non-skip route) ONLY IF AT LEAST ONE holds:
1. The text contains the tag "[at_bot=true]" (the user @-mentioned the bot)
2. The text contains the bot's nickname "{CONFIG.bot_nickname}"
3. The message is clearly addressed to the bot (greets it, asks for its opinion / help by role)
4. The message is a sincere question or "ask the group" call that a friend in
   the group would naturally answer ("早饭吃啥", "今天天气咋样", "有人懂xxx吗",
   "你们觉得xxx好吃吗") — even without naming the bot
5. A natural opening to riff on: interesting topic, story hook, casual
   confession that invites a reaction ("我今天好累啊", "刚发现一家新店")

Skip otherwise. In particular, ALWAYS skip:
- Pure reactions / acknowledgements (嗯/哈/啊/666/23333/[emoji-only]/[punctuation-only])
- Clearly a back-and-forth between two specific users (含 @他人、紧接对话链、显然在回别人的话)
- Statements / observations / complaints with no question and no riff hook
- Off-topic spam, copypasta, ads
- Political / NSFW / heavy personal content the bot has no business jumping into

CRITICAL — image / vision rules:
- A bare image with empty or trivial text → skip (the user just shared a picture, not asked the bot anything).
- "[image attached]" alone is NOT a vision request. Vision requires the user to ASK to look at the image ("看看这张图"/"这是啥"/"翻译一下图里的字").
- Conceptual talk about images ("你能看图吗", "你是怎么识别的") → chat, NOT vision.

When uncertain between chat and skip → skip."""


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
