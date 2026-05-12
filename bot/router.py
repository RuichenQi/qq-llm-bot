"""LLM router. DeepSeek decides which backend should handle a request.

Strict JSON contract. We never trust the model's text output blindly; the
result is validated against an allowed enum, with a `deepseek_chat` fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from bot.logger import get_logger
from providers.base import ChatMessage
from providers.deepseek import DeepSeekProvider

log = get_logger(__name__)

ROUTES = {
    "deepseek_chat",
    "deepseek_think",
    "openai_text",
    "openai_vision",
    "openai_image",
    "openai_image_edit",
    "reject",
}

ROUTER_SYSTEM_PROMPT = """You are a routing module for a QQ group LLM bot. You must not answer the user. You only decide which backend should handle the request.

Return strict JSON only. No markdown. No explanation.

Available routes:
- deepseek_chat: normal text chat, translation, rewriting, simple explanation, casual conversation
- deepseek_think: moderately complex reasoning, technical explanation, code explanation
- openai_text: difficult research reasoning, complex code debugging, high-quality answer requested, or user explicitly wants GPT/OpenAI
- openai_vision: user uploaded an image and asks to analyze, explain, OCR, or answer based on the image
- openai_image: user asks to generate an image
- openai_image_edit: user asks to modify, edit, restyle, or transform an image
- reject: illegal, unsafe, unsupported, or impossible request

Output schema:
{
  "route": "deepseek_chat | deepseek_think | openai_text | openai_vision | openai_image | openai_image_edit | reject",
  "confidence": 0.0,
  "reason": "short reason within 20 Chinese characters",
  "normalized_prompt": "cleaned user request"
}

Routing policy:
- Use deepseek_chat for most normal messages.
- Use openai_text only when clearly necessary.
- Use image routes whenever the user asks for image generation, image editing, or image understanding.
- If uncertain, choose deepseek_chat.
- Never call OpenAI just because the question is long.
- OpenAI is expensive; be conservative.
"""


@dataclass
class RouteDecision:
    route: str
    confidence: float
    reason: str
    normalized_prompt: str


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce(text: str, fallback_prompt: str) -> RouteDecision:
    m = _JSON_OBJECT_RE.search(text)
    payload = m.group(0) if m else text
    try:
        obj = json.loads(payload)
    except Exception as e:
        log.warning("router returned non-JSON, falling back: %s", e)
        return RouteDecision("deepseek_chat", 0.0, "router_parse_fail", fallback_prompt)

    route = str(obj.get("route", "")).strip()
    if route not in ROUTES:
        log.warning("router returned unknown route %r, falling back", route)
        route = "deepseek_chat"

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return RouteDecision(
        route=route,
        confidence=confidence,
        reason=str(obj.get("reason", ""))[:80],
        normalized_prompt=str(obj.get("normalized_prompt") or fallback_prompt)[:4000],
    )


class Router:
    def __init__(self, deepseek: DeepSeekProvider) -> None:
        self._deepseek = deepseek

    async def decide(self, user_text: str, *, has_image: bool) -> RouteDecision:
        if has_image and "图" not in user_text and "image" not in user_text.lower():
            # nudge so the router knows there's an image attached
            user_text = f"[user attached an image] {user_text}"

        messages = [
            ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_text),
        ]
        try:
            reply = await self._deepseek.chat(
                messages,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            log.warning("router call failed, falling back to deepseek_chat: %s", e)
            return RouteDecision("deepseek_chat", 0.0, "router_error", user_text)
        decision = _coerce(reply.text, user_text)
        log.info(
            "router decision route=%s conf=%.2f reason=%s",
            decision.route, decision.confidence, decision.reason,
        )
        return decision
