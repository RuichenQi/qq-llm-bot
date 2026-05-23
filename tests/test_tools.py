"""Tool framework: registry, OpenAI serialization, executor hop loop."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from bot.tools import Tool, ToolContext, ToolRegistry, run_with_tools
from providers.base import ChatMessage, TextReply, ToolCall


# ---------- registry ----------
def _make_dummy_tool(name: str = "echo") -> Tool:
    async def _h(args, ctx):
        return f"echo: {args.get('text', '')}"
    return Tool(
        name=name,
        description="echoes back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=_h,
    )


def test_registry_register_get_names():
    reg = ToolRegistry()
    assert reg.is_empty()
    t = _make_dummy_tool()
    reg.register(t)
    assert not reg.is_empty()
    assert reg.get("echo") is t
    assert reg.get("missing") is None
    assert reg.names() == ["echo"]


def test_registry_register_overwrites():
    reg = ToolRegistry()
    reg.register(_make_dummy_tool("echo"))
    reg.register(_make_dummy_tool("echo"))  # second wins, no crash
    assert reg.names() == ["echo"]


def test_for_openai_serialization():
    reg = ToolRegistry()
    reg.register(_make_dummy_tool("search"))
    spec = reg.for_openai()
    assert len(spec) == 1
    item = spec[0]
    assert item["type"] == "function"
    assert item["function"]["name"] == "search"
    assert item["function"]["description"] == "echoes back"
    assert item["function"]["parameters"]["required"] == ["text"]


# ---------- executor loop ----------
class _ScriptedProvider:
    """Provider stub that returns canned replies in sequence."""

    def __init__(self, replies: List[TextReply]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, messages, *, model=None, max_tokens=None,
                   temperature=0.7, response_format=None, tools=None):
        self.calls.append({
            "messages": [
                {"role": m.role, "content": m.content,
                 "tool_call_id": m.tool_call_id, "name": m.name,
                 "tool_calls": [(tc.name, tc.arguments) for tc in (m.tool_calls or [])]}
                for m in messages
            ],
            "tools": tools,
            "model": model,
        })
        if not self.replies:
            return TextReply(text="(no more scripted replies)")
        return self.replies.pop(0)


def test_run_with_tools_no_tools_path_through():
    """When the very first reply has no tool_calls, we return it as-is."""
    provider = _ScriptedProvider([TextReply(text="hello")])
    reg = ToolRegistry()
    reg.register(_make_dummy_tool())
    ctx = ToolContext(group_id=1, user_id=2)

    async def go():
        out = await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ctx, max_hops=3,
        )
        return out

    out = asyncio.run(go())
    assert out.text == "hello"
    assert len(provider.calls) == 1
    # Tools were exposed even though the model didn't use them.
    assert provider.calls[0]["tools"] is not None


def test_run_with_tools_executes_one_call_then_returns():
    provider = _ScriptedProvider([
        TextReply(text="", tool_calls=[
            ToolCall(id="c1", name="echo", arguments='{"text": "world"}'),
        ]),
        TextReply(text="final answer"),
    ])
    reg = ToolRegistry()
    reg.register(_make_dummy_tool())
    ctx = ToolContext(group_id=1, user_id=2)

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ctx, max_hops=3,
        )

    out = asyncio.run(go())
    assert out.text == "final answer"
    assert len(provider.calls) == 2
    # Second call should contain the original user msg + assistant tool_calls
    # turn + tool result.
    second = provider.calls[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert second["messages"][-1]["content"] == "echo: world"
    assert second["messages"][-1]["tool_call_id"] == "c1"


def test_run_with_tools_unknown_tool_returns_error():
    provider = _ScriptedProvider([
        TextReply(text="", tool_calls=[
            ToolCall(id="c1", name="nope", arguments="{}"),
        ]),
        TextReply(text="ok"),
    ])
    reg = ToolRegistry()
    reg.register(_make_dummy_tool())

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ToolContext(1, 2),
        )

    out = asyncio.run(go())
    assert out.text == "ok"
    tool_msg = provider.calls[1]["messages"][-1]
    assert "not available" in tool_msg["content"]


def test_run_with_tools_bad_arguments_returns_error():
    provider = _ScriptedProvider([
        TextReply(text="", tool_calls=[
            ToolCall(id="c1", name="echo", arguments="not-json"),
        ]),
        TextReply(text="recovered"),
    ])
    reg = ToolRegistry()
    reg.register(_make_dummy_tool())

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ToolContext(1, 2),
        )

    out = asyncio.run(go())
    assert out.text == "recovered"
    tool_msg = provider.calls[1]["messages"][-1]
    assert "valid JSON" in tool_msg["content"]


def test_run_with_tools_handler_exception_is_swallowed():
    async def _crashing(args, ctx):
        raise RuntimeError("boom")

    tool = Tool(
        name="bad", description="crashes",
        parameters={"type": "object", "properties": {}},
        handler=_crashing,
    )
    provider = _ScriptedProvider([
        TextReply(text="", tool_calls=[
            ToolCall(id="c1", name="bad", arguments="{}"),
        ]),
        TextReply(text="moved on"),
    ])
    reg = ToolRegistry()
    reg.register(tool)

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ToolContext(1, 2),
        )

    out = asyncio.run(go())
    assert out.text == "moved on"
    tool_msg = provider.calls[1]["messages"][-1]
    assert "boom" in tool_msg["content"]


def test_run_with_tools_hop_cap_drops_tools_on_last_hop():
    """If the model keeps calling tools, we cut off and force a final answer
    on the last hop by stripping tools from the request."""
    # Model wants 4 calls in a row; max_hops=2 allows hops 0 and 1 with tools,
    # then hop 2 (the final) goes out WITHOUT tools.
    def _tool_call_reply():
        return TextReply(text="", tool_calls=[
            ToolCall(id="c", name="echo", arguments='{"text": "x"}'),
        ])

    provider = _ScriptedProvider([
        _tool_call_reply(),
        _tool_call_reply(),
        TextReply(text="committed"),  # forced final answer
    ])
    reg = ToolRegistry()
    reg.register(_make_dummy_tool())

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ToolContext(1, 2), max_hops=2,
        )

    out = asyncio.run(go())
    assert out.text == "committed"
    assert len(provider.calls) == 3
    # The first two requests had tools; the last one MUST NOT have them.
    assert provider.calls[0]["tools"] is not None
    assert provider.calls[1]["tools"] is not None
    assert provider.calls[2]["tools"] is None


def test_run_with_tools_empty_registry_is_fine():
    """No tools registered → just a plain chat call, no tools field."""
    provider = _ScriptedProvider([TextReply(text="ok")])
    reg = ToolRegistry()

    async def go():
        return await run_with_tools(
            provider=provider, messages=[ChatMessage("user", "hi")],
            registry=reg, ctx=ToolContext(1, 2),
        )

    asyncio.run(go())
    assert provider.calls[0]["tools"] is None
