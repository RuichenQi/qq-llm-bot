"""Provider tool-call wire format: serialize ChatMessage(role=tool), parse
tool_calls out of chat-completion responses, pass `tools=[]` through."""
from __future__ import annotations

import json

import httpx
import pytest

from providers.base import ChatMessage, ToolCall
from providers.deepseek import (
    DeepSeekProvider, _message_to_wire, _parse_tool_calls,
)


# ---------- _message_to_wire ----------
def test_wire_simple_user_message():
    out = _message_to_wire(ChatMessage("user", "hi"))
    assert out == {"role": "user", "content": "hi"}


def test_wire_assistant_with_tool_calls():
    msg = ChatMessage(
        role="assistant", content="",
        tool_calls=[ToolCall(id="abc", name="search", arguments='{"q":"x"}')],
    )
    out = _message_to_wire(msg)
    assert out["role"] == "assistant"
    assert out["tool_calls"][0]["id"] == "abc"
    assert out["tool_calls"][0]["function"]["name"] == "search"
    assert out["tool_calls"][0]["function"]["arguments"] == '{"q":"x"}'
    assert out["tool_calls"][0]["type"] == "function"


def test_wire_tool_result():
    msg = ChatMessage(
        role="tool", content="result here",
        tool_call_id="abc", name="search",
    )
    out = _message_to_wire(msg)
    assert out["role"] == "tool"
    assert out["content"] == "result here"
    assert out["tool_call_id"] == "abc"
    assert out["name"] == "search"


# ---------- _parse_tool_calls ----------
def test_parse_tool_calls_none():
    assert _parse_tool_calls(None) == []
    assert _parse_tool_calls([]) == []


def test_parse_tool_calls_basic():
    raw = [
        {"id": "c1", "type": "function",
         "function": {"name": "search", "arguments": '{"q":"hi"}'}},
        {"id": "c2", "type": "function",
         "function": {"name": "echo", "arguments": ""}},
    ]
    out = _parse_tool_calls(raw)
    assert len(out) == 2
    assert out[0].id == "c1" and out[0].name == "search"
    assert out[0].arguments == '{"q":"hi"}'
    assert out[1].name == "echo" and out[1].arguments == ""


def test_parse_tool_calls_skips_nameless():
    raw = [
        {"id": "c1", "function": {"name": "", "arguments": "{}"}},  # skip
        {"id": "c2", "function": {"name": "good", "arguments": "{}"}},
    ]
    out = _parse_tool_calls(raw)
    assert len(out) == 1
    assert out[0].name == "good"


# ---------- end-to-end DeepSeek.chat() with tools ----------
def test_chat_passes_tools_and_parses_tool_calls(monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg.CONFIG, "deepseek_api_key", "k", raising=False)

    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured["body"] = body
        return httpx.Response(200, json={
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "雷军"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    provider = DeepSeekProvider(client=client)
    tools_spec = [{
        "type": "function",
        "function": {"name": "web_search", "description": "", "parameters": {}},
    }]

    import asyncio
    async def go():
        reply = await provider.chat(
            [ChatMessage("user", "hi")],
            tools=tools_spec,
        )
        await provider.aclose()
        return reply

    reply = asyncio.run(go())
    assert reply.text == ""
    assert reply.finish_reason == "tool_calls"
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "web_search"
    assert json.loads(reply.tool_calls[0].arguments)["query"] == "雷军"
    # Verify the tools spec was actually sent on the wire.
    assert captured["body"]["tools"] == tools_spec


def test_chat_without_tools_omits_field(monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg.CONFIG, "deepseek_api_key", "k", raising=False)

    captured = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}],
            "usage": {},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    provider = DeepSeekProvider(client=client)

    import asyncio
    asyncio.run(provider.chat([ChatMessage("user", "hi")]))
    asyncio.run(provider.aclose())
    assert "tools" not in captured["body"]
