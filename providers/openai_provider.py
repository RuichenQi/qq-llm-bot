"""OpenAI provider: text, vision, image generation, image editing.

Kept SDK-free to minimise install footprint on Termux — uses httpx directly
against the public OpenAI REST surface.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import httpx

from bot.logger import get_logger
from config import CONFIG
from providers.base import ChatMessage, ImageReply, ProviderError, TextReply

log = get_logger(__name__)


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        if not CONFIG.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not configured")
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._headers = {
            "Authorization": f"Bearer {CONFIG.openai_api_key}",
            "Content-Type": "application/json",
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- text ----------
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
            "model": model or CONFIG.openai_text_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format

        url = f"{CONFIG.openai_base_url.rstrip('/')}/chat/completions"
        try:
            resp = await self._client.post(url, headers=self._headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI network error: {e}") from e
        if resp.status_code >= 400:
            raise ProviderError(f"OpenAI HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"OpenAI malformed response: {e}") from e
        usage = data.get("usage", {}) or {}
        log.info(
            "openai chat ok model=%s tokens_in=%s tokens_out=%s",
            body["model"], usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        return TextReply(text=text.strip(), usage=usage, model=body["model"])

    # ---------- vision ----------
    async def vision(
        self,
        question: str,
        image_urls: List[str],
        *,
        max_tokens: Optional[int] = None,
    ) -> TextReply:
        content: List[Dict[str, Any]] = [{"type": "text", "text": question or "请描述这张图片"}]
        for u in image_urls:
            # detail=low locks the flat 85-token price. With our 512px max
            # input that's enough resolution for OCR + general description.
            content.append({
                "type": "image_url",
                "image_url": {"url": u, "detail": "low"},
            })

        body: Dict[str, Any] = {
            "model": CONFIG.openai_vision_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "stream": False,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens

        url = f"{CONFIG.openai_base_url.rstrip('/')}/chat/completions"
        try:
            resp = await self._client.post(url, headers=self._headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI vision network error: {e}") from e
        if resp.status_code >= 400:
            raise ProviderError(f"OpenAI vision HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"OpenAI vision malformed: {e}") from e
        usage = data.get("usage", {}) or {}
        log.info("openai vision ok images=%d tokens_out=%s",
                 len(image_urls), usage.get("completion_tokens"))
        return TextReply(text=text.strip(), usage=usage, model=body["model"])

    # ---------- image generate ----------
    async def generate(self, prompt: str, *, size: str = "1024x1024") -> ImageReply:
        model = CONFIG.openai_image_model
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
        # gpt-image-* takes a quality knob and always returns b64_json (so
        # response_format would be a 400). dall-e-* needs response_format set
        # explicitly to avoid the default URL (which expires in ~1h).
        if model.startswith("gpt-image"):
            body["quality"] = CONFIG.openai_image_quality
        elif model.startswith("dall-e"):
            body["response_format"] = "b64_json"
        url = f"{CONFIG.openai_base_url.rstrip('/')}/images/generations"
        try:
            resp = await self._client.post(url, headers=self._headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI image network error: {e}") from e
        if resp.status_code >= 400:
            raise ProviderError(f"OpenAI image HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            item = data["data"][0]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"OpenAI image malformed: {e}") from e
        log.info("openai image ok model=%s size=%s", body["model"], size)
        return ImageReply(
            b64_png=item.get("b64_json"),
            url=item.get("url"),
            model=body["model"],
        )

    # ---------- image edit ----------
    async def edit(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        size: str = "1024x1024",
    ) -> ImageReply:
        url = f"{CONFIG.openai_base_url.rstrip('/')}/images/edits"
        files = {"image": ("input.png", image_bytes, "image/png")}
        model = CONFIG.openai_image_model
        data: Dict[str, str] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": "1",
        }
        if model.startswith("gpt-image"):
            data["quality"] = CONFIG.openai_image_quality
        elif model.startswith("dall-e"):
            data["response_format"] = "b64_json"
        headers = {"Authorization": f"Bearer {CONFIG.openai_api_key}"}
        try:
            resp = await self._client.post(url, headers=headers, files=files, data=data)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI image-edit network error: {e}") from e
        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI image-edit HTTP {resp.status_code}: {resp.text[:300]}"
            )
        payload = resp.json()
        try:
            item = payload["data"][0]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"OpenAI image-edit malformed: {e}") from e
        log.info("openai image-edit ok bytes_in=%d", len(image_bytes))
        return ImageReply(
            b64_png=item.get("b64_json"),
            url=item.get("url"),
            model=CONFIG.openai_image_model,
        )

    @staticmethod
    def b64_to_bytes(b64: str) -> bytes:
        return base64.b64decode(b64)
