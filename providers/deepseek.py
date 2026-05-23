"""DeepSeek provider (OpenAI-compatible chat completions, with streaming)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from bot.logger import get_logger
from config import CONFIG
from providers.base import ChatMessage, ProviderError, TextReply, ToolCall

log = get_logger(__name__)


def _message_to_wire(m: ChatMessage) -> Dict[str, Any]:
    """Serialize a ChatMessage into the OpenAI-compatible wire format,
    threading through tool_calls and tool result fields when present."""
    out: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
            }
            for tc in m.tool_calls
        ]
        # Assistant messages that solely request tool calls usually have empty
        # content. OpenAI lets us send it as null; "" also works, leave as is.
    if m.role == "tool":
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        if m.name:
            out["name"] = m.name
    return out


def _parse_tool_calls(raw: Any) -> List[ToolCall]:
    """Pull tool_calls out of a chat-completion response message."""
    if not isinstance(raw, list):
        return []
    calls: List[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        calls.append(ToolCall(
            id=str(entry.get("id") or f"call_{len(calls)}"),
            name=name,
            arguments=str(fn.get("arguments") or ""),
        ))
    return calls


class DeepSeekProvider:
    name = "deepseek"

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        if not CONFIG.deepseek_api_key:
            raise ProviderError("DEEPSEEK_API_KEY is not configured")
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._headers = {
            "Authorization": f"Bearer {CONFIG.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> TextReply:
        body: Dict[str, Any] = {
            "model": model or CONFIG.deepseek_chat_model,
            "messages": [_message_to_wire(m) for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools

        url = f"{CONFIG.deepseek_base_url.rstrip('/')}/chat/completions"
        try:
            resp = await self._client.post(url, headers=self._headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"DeepSeek network error: {e}") from e
        if resp.status_code >= 400:
            raise ProviderError(
                f"DeepSeek HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"DeepSeek malformed response: {e}: {data}") from e
        tool_calls = _parse_tool_calls(msg.get("tool_calls"))
        usage = data.get("usage", {}) or {}
        finish_reason = str(choice.get("finish_reason") or "")
        log.info(
            "deepseek call ok model=%s tokens_in=%s tokens_out=%s tools=%d finish=%s",
            body["model"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            len(tool_calls), finish_reason,
        )
        return TextReply(
            text=text.strip(), usage=usage, model=body["model"],
            tool_calls=tool_calls, finish_reason=finish_reason,
        )

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive (SSE). The caller decides how
        often to flush them into QQ."""
        body: Dict[str, Any] = {
            "model": model or CONFIG.deepseek_chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens

        url = f"{CONFIG.deepseek_base_url.rstrip('/')}/chat/completions"
        try:
            async with self._client.stream(
                "POST", url, headers=self._headers, json=body
            ) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    raise ProviderError(f"DeepSeek stream HTTP {resp.status_code}: {text}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = evt["choices"][0].get("delta", {})
                    except (KeyError, IndexError):
                        continue
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            raise ProviderError(f"DeepSeek stream network error: {e}") from e
