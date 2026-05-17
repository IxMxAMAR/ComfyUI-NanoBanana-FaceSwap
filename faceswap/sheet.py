"""Compose multiple identity reference images into a single labeled grid."""

from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def _resolve_layout(layout: str, n: int) -> Tuple[int, int]:
    if layout == "auto":
        if n <= 1: return (1, 1)
        if n == 2: return (2, 1)
        if n == 3: return (3, 1)
        if n == 4: return (2, 2)
        if n <= 6: return (3, 2)
        return (4, ((n + 3) // 4))
    if layout == "1xN":
        return (n, 1)
    static = {
        "2x1": (2, 1),
        "1x2": (1, 2),
        "2x2": (2, 2),
        "3x2": (3, 2),
        "2x3": (2, 3),
    }
    if layout in static:
        return static[layout]
    raise ValueError(f"unknown layout {layout!r}")


def _fit_tile(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    tile.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return tile


def _label(tile: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, tile.width // 12))
    except (OSError, IOError):
        font = ImageFont.load_default()
    pad = 4
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((pad + dx, pad + dy), text, fill=(0, 0, 0), font=font)
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return tile


def compose(refs: List[Image.Image], tile_size: int = 512,
            layout: str = "auto") -> Image.Image:
    if not refs:
        raise ValueError("compose() requires at least one reference image")
    cols, rows = _resolve_layout(layout, len(refs))
    grid_w, grid_h = cols * tile_size, rows * tile_size
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, ref in enumerate(refs[:cols * rows]):
        tile = _fit_tile(ref, tile_size)
        tile = _label(tile, f"REF {i + 1}")
        r, c = divmod(i, cols)
        canvas.paste(tile, (c * tile_size, r * tile_size))
    return canvas
