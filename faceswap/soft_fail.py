"""Render red/yellow placeholder images when a swap is refused or errors."""

from __future__ import annotations
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def _tint(img: Image.Image, color: Tuple[int, int, int], strength: float = 0.4,
          region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    base = img.convert("RGB").copy()
    if region is None:
        overlay = Image.new("RGB", base.size, color)
        return Image.blend(base, overlay, strength)
    x1, y1, x2, y2 = region
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(base.width, x2); y2 = min(base.height, y2)
    if x2 <= x1 or y2 <= y1:
        return base
    crop = base.crop((x1, y1, x2, y2))
    overlay = Image.new("RGB", crop.size, color)
    blended = Image.blend(crop, overlay, strength)
    base.paste(blended, (x1, y1))
    return base


def _draw_text(img: Image.Image, text: str, xy: Tuple[int, int]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 40))
    except (OSError, IOError):
        font = ImageFont.load_default()
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def render_refused(target: Image.Image, reason: str,
                   region_bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    tinted = _tint(target, color=(220, 20, 20), strength=0.45, region=region_bbox)
    text = f"REFUSED: {reason}"
    if region_bbox is not None:
        x = region_bbox[0]
        y = min(region_bbox[3] + 4, target.height - 24)
    else:
        x, y = 12, max(8, target.height // 20)
    return _draw_text(tinted, text, (x, y))


def render_error(target: Image.Image, reason: str,
                 region_bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    tinted = _tint(target, color=(240, 200, 30), strength=0.5, region=region_bbox)
    text = f"ERROR: {reason}"
    if region_bbox is not None:
        x = region_bbox[0]
        y = min(region_bbox[3] + 4, target.height - 24)
    else:
        x, y = 12, max(8, target.height // 20)
    return _draw_text(tinted, text, (x, y))
