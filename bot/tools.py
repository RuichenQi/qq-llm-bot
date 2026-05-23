"""Tool-use framework.

Lets the LLM decide at chat time whether to invoke external capabilities
(web search, URL fetch, calculator, …) before composing its final reply.

Design:

  * `Tool` describes one capability — name, human-readable description, JSON
    Schema for arguments, and an async handler that produces a string result.
  * `ToolRegistry` holds the bot's installed tools and serialises them to the
    OpenAI tool-spec format. DeepSeek and OpenAI both consume that format.
  * `run_with_tools()` is the executor loop: call provider.chat() with tools,
    handle any returned tool_calls, append results as `role="tool"` messages,
    repeat — capped at `max_hops` so a misbehaving LLM can't infinite-loop.

Adding a new tool is just `registry.register(Tool(...))`. No provider or
handler changes needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from bot.logger import get_logger
from providers.base import ChatMessage, TextReply, ToolCall

log = get_logger(__name__)


@dataclass
class ToolContext:
    """Side-channel passed to every tool handler. Lets a tool know who's
    asking + access shared services (HTTP client, quota, etc.) without going
    through global state."""
    group_id: int
    user_id: int
    extras: Dict[str, Any] = field(default_factory=dict)


# A tool handler takes parsed args + context and returns a string the LLM
# will see as the tool's "output". Keep results compact (a few hundred chars
# typically) — they're concatenated into the next chat call.
ToolHandler = Callable[[Dict[str, Any], ToolContext], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (OpenAI tool spec)
    handler: ToolHandler


class ToolRegistry:
    """Holds the installed tools. One instance per Handler is plenty."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("tool %r already registered; overwriting", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return not self._tools

    def for_openai(self) -> List[Dict[str, Any]]:
        """Serialise to the OpenAI / DeepSeek tools[] format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]


async def _execute_one(
    registry: ToolRegistry, call: ToolCall, ctx: ToolContext,
) -> str:
    """Run a single tool call. Errors become string error messages — never
    raise, since the LLM still needs *something* back to continue."""
    tool = registry.get(call.name)
    if tool is None:
        log.warning("LLM tried to call unknown tool %r", call.name)
        return f"Error: tool {call.name!r} not available"
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as e:
        log.warning("tool %s got bad JSON args: %s", call.name, e)
        return f"Error: arguments were not valid JSON ({e})"
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object"
    try:
        result = await tool.handler(args, ctx)
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s handler crashed", call.name)
        return f"Error: tool {call.name} failed: {e}"
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    log.info("tool %s ok: out_chars=%d", call.name, len(result))
    return result


async def run_with_tools(
    *,
    provider,
    messages: List[ChatMessage],
    registry: ToolRegistry,
    ctx: ToolContext,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    max_hops: int = 3,
) -> TextReply:
    """Chat with tool-use enabled. Loops until the LLM gives a plain text
    response, an error, or we hit `max_hops`.

    Caller is responsible for the initial messages including any system
    prompts (persona, lessons, group context, …). We append tool-roundtrip
    messages onto a local copy so the caller's list isn't mutated.
    """
    work: List[ChatMessage] = list(messages)
    tools_spec = registry.for_openai() if not registry.is_empty() else None
    last_reply: Optional[TextReply] = None
    for hop in range(max_hops + 1):
        # On the last hop, drop tools — force the model to commit to a
        # final natural-language answer rather than calling more tools.
        active_tools = tools_spec if hop < max_hops else None
        reply = await provider.chat(
            work,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=active_tools,
        )
        last_reply = reply
        if not reply.tool_calls:
            return reply
        log.info(
            "tool hop %d/%d: %d call(s) — %s",
            hop + 1, max_hops, len(reply.tool_calls),
            ", ".join(c.name for c in reply.tool_calls),
        )
        # Echo the assistant turn (with its tool_calls) so subsequent
        # tool messages have something to attach to.
        work.append(ChatMessage(
            role="assistant",
            content=reply.text or "",
            tool_calls=list(reply.tool_calls),
        ))
        for call in reply.tool_calls:
            result = await _execute_one(registry, call, ctx)
            work.append(ChatMessage(
                role="tool", content=result,
                tool_call_id=call.id, name=call.name,
            ))
    # Hop budget exhausted but the model still wants more tools — return
    # whatever final text we got (may be empty).
    log.warning("tool loop hit max_hops=%d without final answer", max_hops)
    return last_reply or TextReply(text="(tool loop exhausted)")
