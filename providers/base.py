"""Provider abstractions. Concrete providers live in sibling modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class TextReply:
    text: str
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""


@dataclass
class ImageReply:
    """Either a base64 PNG or a remote URL.

    OneBot v11 accepts both `file=base64://...` and `file=http(s)://...`.
    """

    b64_png: Optional[str] = None
    url: Optional[str] = None
    model: str = ""


class TextProvider(Protocol):
    name: str

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> TextReply: ...


class VisionProvider(Protocol):
    name: str

    async def vision(
        self,
        question: str,
        image_urls: List[str],
        *,
        max_tokens: Optional[int] = None,
    ) -> TextReply: ...


class ImageProvider(Protocol):
    name: str

    async def generate(self, prompt: str, *, size: str = "1024x1024") -> ImageReply: ...

    async def edit(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        size: str = "1024x1024",
    ) -> ImageReply: ...


class ProviderError(RuntimeError):
    """Raised for any backend provider failure."""
