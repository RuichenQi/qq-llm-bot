"""Strip AI-ish Chinese interjections that leak through despite the persona.

Even with strict prompt instructions, the LLM emits 嘿嘿 / 诶呀 / 嘶... because
prior chat history (group memory) is full of them — the model picks up its own
style and keeps repeating it. This is a deterministic strip on outgoing chunks,
mirroring how `emoji_filter` enforces the no-emoji rule.

Conservative on purpose: only catches the most blatant AI tells. Single 嘿 / 嗯 /
啦 / 呀 stay (they can be natural in real speech).
"""
from __future__ import annotations

import re

_TRAILING = r"[.。\-—~,，!！?？…\s]*"

_PATTERNS = [
    re.compile(r"嘿+嘿+" + _TRAILING),     # 嘿嘿, 嘿嘿嘿, 嘿嘿嘿嘿
    re.compile(r"诶+呀+" + _TRAILING),     # 诶呀
    re.compile(r"哎+呀+" + _TRAILING),     # 哎呀
    re.compile(r"嘶+" + _TRAILING),        # 嘶, 嘶~, 嘶..., 嘶——
    re.compile(r"嗯+嗯+" + _TRAILING),     # 嗯嗯, 嗯嗯嗯
]


def filter_interjections(text: str) -> str:
    """Remove AI-ish interjections (嘿嘿/诶呀/嘶/嗯嗯/...) from `text`."""
    if not text:
        return text
    out = text
    for pat in _PATTERNS:
        out = pat.sub("", out)
    # Tidy: drop punctuation orphaned at the start, collapse double spaces.
    out = re.sub(r"^[，,。！?？~…\s]+", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()
