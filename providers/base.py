"""Provider abstractions. Concrete providers live in sibling modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    """One function call the assistant wants us to execute on its behalf.

    Mirrors the OpenAI tool_calls schema:
        {"id": "...", "type": "function",
         "function": {"name": "...", "arguments": "<json string>"}}
    """
    id: str
    name: str
    arguments: str = ""  # JSON-encoded args, per OpenAI spec


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    # Optional fields used only on tool-use turns. Kept as Optional so all
    # existing call sites that construct `ChatMessage(role, content)` keep
    # working unchanged.
    name: Optional[str] = None              # tool name (for role=="tool")
    tool_call_id: Optional[str] = None      # links tool result → assistant tool_call
    tool_calls: Optional[List[ToolCall]] = None  # assistant-side: calls being requested


@dataclass
class TextReply:
    text: str
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    # OpenAI finish_reason; "tool_calls" means the model asked us to run tools
    # rather than producing a final answer.
    finish_reason: str = ""


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
        tools: Optional[List[Dict[str, Any]]] = None,
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
