"""DeepSeek provider (OpenAI-compatible chat completions, with streaming)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from bot.logger import get_logger
from config import CONFIG
from providers.base import ChatMessage, ProviderError, TextReply

log = get_logger(__name__)


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
    ) -> TextReply:
        body: Dict[str, Any] = {
            "model": model or CONFIG.deepseek_chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format

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
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"DeepSeek malformed response: {e}: {data}") from e
        usage = data.get("usage", {}) or {}
        log.info(
            "deepseek call ok model=%s tokens_in=%s tokens_out=%s",
            body["model"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        return TextReply(text=text.strip(), usage=usage, model=body["model"])

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
