"""Tiny helpers for downscaling images before sending them to a vision API.

We only depend on Pillow. PNG everywhere — JPEG would save bytes but OpenAI's
vision endpoint costs the same per-tile regardless of format.
"""
from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def downscale_to_max(data: bytes, max_side: int) -> bytes:
    """Resize so max(width, height) <= max_side, re-encode as PNG."""
    with Image.open(BytesIO(data)) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()


def to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
