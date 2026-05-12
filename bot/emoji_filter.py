"""Probabilistic emoji stripper.

Even with strict persona instructions, LLMs leak emojis (😊 🥺 etc.) into
chat replies — it's their #1 AI tell. This filter runs on every outgoing
chunk and removes emoji with `1 - keep_probability` chance per match.
"""
from __future__ import annotations

import random
import re

# Covers the bulk of Unicode emoji + dingbats + arrows + sundry symbols that
# show up as colorful icons in QQ. Variation selector + ZWJ stay attached so
# ZWJ sequences (eg 👨‍💻) are removed as one unit.
_EMOJI_RE = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"   # most modern emoji blocks
    r"\U0001F600-\U0001F64F"   # faces
    r"\U0001F680-\U0001F6FF"   # transport / map
    r"☀-⛿"           # misc symbols (☀ ⭐ etc.)
    r"✀-➿"           # dingbats (✨ ❤ etc.)
    r"⌀-⏿"           # misc technical (⌚ ⏰)
    r"⬀-⯿"           # misc symbols + arrows (⭐ ⬆)
    r"]"
    r"[️‍\U0001F3FB-\U0001F3FF⃣]*"  # variation / skin tone / ZWJ
    r"(?:[\U0001F300-\U0001FAFF][️‍\U0001F3FB-\U0001F3FF⃣]*)*",
    flags=re.UNICODE,
)


def filter_emoji(text: str, keep_probability: float) -> str:
    """Remove emoji from `text` with `1 - keep_probability` chance each."""
    if not text or keep_probability >= 1.0:
        return text

    def _maybe(m: re.Match) -> str:
        return m.group(0) if random.random() < keep_probability else ""

    out = _EMOJI_RE.sub(_maybe, text)
    # Tidy up double spaces / leading punctuation orphaned by removals.
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()
