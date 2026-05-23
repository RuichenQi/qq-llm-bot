"""Parse OneBot v11 group_message events into something the handler can use."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AttachedFile:
    """A file attached to a QQ message (CQ:file segment)."""
    name: str
    url: str = ""
    file_id: str = ""    # adapter-specific identifier (NapCat: get_file by id)
    size: int = 0


@dataclass
class QuotedMessage:
    """A fetched OneBot get_msg payload — text + any image URLs and file
    segments from the quote."""
    text: str
    image_urls: List[str] = field(default_factory=list)
    files: List["AttachedFile"] = field(default_factory=list)


@dataclass
class ParsedMessage:
    self_id: int
    group_id: int
    user_id: int
    nickname: str
    raw_text: str
    text: str
    image_urls: List[str] = field(default_factory=list)
    files: List[AttachedFile] = field(default_factory=list)
    is_command: bool = False
    command: str = ""
    command_args: str = ""
    at_targets: List[int] = field(default_factory=list)
    reply_to_msg_id: Optional[str] = None

    @property
    def has_image(self) -> bool:
        return bool(self.image_urls)

    @property
    def has_file(self) -> bool:
        return bool(self.files)

    def mentions(self, qq_id: int) -> bool:
        return qq_id in self.at_targets


_CQ_RE = re.compile(r"\[CQ:(?P<type>[^,\]]+)(?P<params>(,[^\]]*)?)\]")


def _strip_cq(raw: str) -> str:
    return _CQ_RE.sub("", raw).strip()


def _extract_image_urls_from_string(raw: str) -> List[str]:
    """For string-format OneBot events: [CQ:image,file=...,url=https://...]"""
    urls: List[str] = []
    for m in _CQ_RE.finditer(raw):
        if m.group("type") != "image":
            continue
        params = m.group("params") or ""
        parts = dict(
            p.split("=", 1) for p in params.lstrip(",").split(",") if "=" in p
        )
        url = parts.get("url") or parts.get("file")
        if url:
            urls.append(url)
    return urls


def _extract_files_from_string(raw: str) -> List[AttachedFile]:
    """For string-format OneBot events: [CQ:file,name=..,url=..,file_id=..]"""
    out: List[AttachedFile] = []
    for m in _CQ_RE.finditer(raw):
        if m.group("type") != "file":
            continue
        params = m.group("params") or ""
        parts = dict(
            p.split("=", 1) for p in params.lstrip(",").split(",") if "=" in p
        )
        name = parts.get("file") or parts.get("name") or "file"
        url = parts.get("url") or ""
        file_id = parts.get("file_id") or parts.get("id") or ""
        try:
            size = int(parts.get("size") or 0)
        except ValueError:
            size = 0
        out.append(AttachedFile(name=name, url=url, file_id=file_id, size=size))
    return out


def _extract_from_array(
    segments: List[Dict[str, Any]],
) -> tuple[str, List[str], List[int], Optional[str], List[AttachedFile]]:
    text_parts: List[str] = []
    urls: List[str] = []
    ats: List[int] = []
    reply_id: Optional[str] = None
    files: List[AttachedFile] = []
    for seg in segments:
        t = seg.get("type")
        data = seg.get("data") or {}
        if t == "text":
            text_parts.append(str(data.get("text", "")))
        elif t == "image":
            url = data.get("url") or data.get("file")
            if url:
                urls.append(str(url))
        elif t == "at":
            qq = data.get("qq")
            try:
                if qq is not None and str(qq) != "all":
                    ats.append(int(qq))
            except (TypeError, ValueError):
                pass
        elif t == "reply":
            mid = data.get("id")
            if mid is not None:
                reply_id = str(mid)
        elif t == "file":
            files.append(AttachedFile(
                name=str(data.get("file") or data.get("name") or "file"),
                url=str(data.get("url") or ""),
                file_id=str(data.get("file_id") or data.get("id") or ""),
                size=int(data.get("size") or 0) if str(data.get("size") or "0").isdigit() else 0,
            ))
    return ("".join(text_parts).strip(), urls, ats, reply_id, files)


_AT_RE = re.compile(r"\[CQ:at,qq=(\d+)[^\]]*\]")
_REPLY_RE = re.compile(r"\[CQ:reply,id=([^,\]]+)[^\]]*\]")


_COMMAND_RE = re.compile(r"^/(\w+)\s*(.*)$", re.DOTALL)


def parse_event(event: Dict[str, Any]) -> Optional[ParsedMessage]:
    """Return a ParsedMessage for a group message or group_upload notice;
    None otherwise."""
    post_type = event.get("post_type")
    if post_type == "notice" and event.get("notice_type") == "group_upload":
        return _parse_group_upload(event)
    if post_type != "message":
        return None
    if event.get("message_type") != "group":
        return None

    self_id = int(event.get("self_id") or 0)
    group_id = int(event.get("group_id") or 0)
    sender = event.get("sender") or {}
    user_id = int(event.get("user_id") or sender.get("user_id") or 0)
    nickname = str(sender.get("card") or sender.get("nickname") or "")

    message = event.get("message")
    raw_text = str(event.get("raw_message") or "")

    reply_id: Optional[str] = None
    files: List[AttachedFile] = []
    if isinstance(message, list):
        text, urls, ats, reply_id, files = _extract_from_array(message)
    elif isinstance(message, str):
        text = _strip_cq(message)
        urls = _extract_image_urls_from_string(message)
        ats = [int(m.group(1)) for m in _AT_RE.finditer(message)]
        rm = _REPLY_RE.search(message)
        if rm:
            reply_id = rm.group(1)
        files = _extract_files_from_string(message)
    else:
        text, urls, ats = "", [], []

    is_cmd = False
    cmd, args = "", ""
    m = _COMMAND_RE.match(text)
    if m:
        is_cmd = True
        cmd = m.group(1).lower()
        args = m.group(2).strip()

    return ParsedMessage(
        self_id=self_id,
        group_id=group_id,
        user_id=user_id,
        nickname=nickname,
        raw_text=raw_text,
        text=text,
        image_urls=urls,
        files=files,
        is_command=is_cmd,
        command=cmd,
        command_args=args,
        at_targets=ats,
        reply_to_msg_id=reply_id,
    )


def _parse_group_upload(event: Dict[str, Any]) -> Optional[ParsedMessage]:
    """Coerce a notice/group_upload event into a ParsedMessage so the file
    flows through the same handler as message-segment files. OneBot v11
    group_upload events carry a `file` object: {id, name, size, url, busid}."""
    finfo = event.get("file") or {}
    name = str(finfo.get("name") or "file")
    url = str(finfo.get("url") or "")
    file_id = str(finfo.get("id") or "")
    try:
        size = int(finfo.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if not name and not url and not file_id:
        return None
    self_id = int(event.get("self_id") or 0)
    group_id = int(event.get("group_id") or 0)
    user_id = int(event.get("user_id") or 0)
    return ParsedMessage(
        self_id=self_id,
        group_id=group_id,
        user_id=user_id,
        nickname=str(event.get("sender", {}).get("nickname") or f"u{user_id}"),
        raw_text=f"[group_upload {name}]",
        text="",
        image_urls=[],
        files=[AttachedFile(name=name, url=url, file_id=file_id, size=size)],
        is_command=False,
        command="",
        command_args="",
        at_targets=[],
        reply_to_msg_id=None,
    )


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split a long string into <= max_chars chunks, preferring newlines."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_chars:
        slice_ = remaining[:max_chars]
        cut = slice_.rfind("\n")
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks
