"""Pillow-backed image helpers."""
from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from bot.image_utils import downscale_to_max, to_data_uri


def _png(size: tuple[int, int], colour=(200, 30, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_downscale_shrinks_large_image():
    src = _png((1024, 768))
    out = downscale_to_max(src, 128)
    with Image.open(BytesIO(out)) as im:
        assert max(im.size) == 128
        # aspect preserved (1024:768 == 4:3)
        assert im.size == (128, 96)


def test_downscale_leaves_small_image_alone_sizewise():
    src = _png((100, 100))
    out = downscale_to_max(src, 128)
    with Image.open(BytesIO(out)) as im:
        assert im.size == (100, 100)


def test_downscale_handles_non_rgb_modes():
    # Palette-mode PNG should still round-trip.
    pal = Image.new("P", (300, 300))
    buf = BytesIO()
    pal.save(buf, format="PNG")
    out = downscale_to_max(buf.getvalue(), 128)
    with Image.open(BytesIO(out)) as im:
        assert max(im.size) == 128


def test_to_data_uri_round_trip():
    src = _png((4, 4))
    uri = to_data_uri(src)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    assert decoded == src
