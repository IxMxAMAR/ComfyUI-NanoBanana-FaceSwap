"""Aspect-ratio handling for Gemini image gen.

Gemini conforms strictly to its supported aspect ratios. If we send an
arbitrary-ratio image and request a standard ratio, the model generates at the
standard ratio and any resize-back squashes the result. The remedy is to pad
the input symmetrically to the closest supported ratio before sending, then
center-crop the padding off after receiving.
"""

from __future__ import annotations
from typing import Tuple
from PIL import Image


ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "5:4": (5, 4),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
}


def closest_aspect(w: int, h: int) -> str:
    """Return the Gemini-supported aspect ratio nearest to w/h."""
    target = w / max(1, h)
    return min(ASPECT_RATIOS.items(),
               key=lambda kv: abs((kv[1][0] / kv[1][1]) - target))[0]


def pad_to_aspect(image: Image.Image,
                  aspect_str: str) -> Tuple[Image.Image, int, int]:
    """Symmetrically pad `image` (with black) so its dimensions exactly match
    `aspect_str`. Returns (padded_image, pad_x, pad_y).
    """
    w, h = image.size
    if aspect_str not in ASPECT_RATIOS:
        return image, 0, 0
    ar_w, ar_h = ASPECT_RATIOS[aspect_str]
    target = ar_w / ar_h
    current = w / h
    if abs(current - target) < 0.005:
        return image, 0, 0

    if current > target:
        new_w = w
        new_h = int(round(w * ar_h / ar_w))
    else:
        new_w = int(round(h * ar_w / ar_h))
        new_h = h

    pad_x = (new_w - w) // 2
    pad_y = (new_h - h) // 2
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (pad_x, pad_y))
    return padded, pad_x, pad_y


def unpad_after_gemini(gemini_out: Image.Image,
                      padded_size: Tuple[int, int],
                      pad_x: int, pad_y: int,
                      orig_size: Tuple[int, int]) -> Image.Image:
    """Inverse of pad_to_aspect.

    PERF: rather than naively upsampling the whole Gemini output to padded_size
    (potentially millions of pixels of LANCZOS work just to discard the
    border), compute the relative crop on the actual returned image, crop
    first, then resize the smaller patch. Visually identical, but for a
    Gemini 1024x1024 → 4000x4000 unpad this is ~16x faster.
    """
    pw, ph = padded_size
    ow, oh = orig_size
    if pw <= 0 or ph <= 0:
        return gemini_out.resize((ow, oh), Image.LANCZOS)
    gw, gh = gemini_out.size
    sx = gw / pw
    sy = gh / ph
    left = max(0, int(round(pad_x * sx)))
    top = max(0, int(round(pad_y * sy)))
    right = min(gw, int(round((pad_x + ow) * sx)))
    bottom = min(gh, int(round((pad_y + oh) * sy)))
    if right <= left or bottom <= top:
        # Degenerate; fall back to the legacy path so we never return empty.
        at_padded = gemini_out.resize((pw, ph), Image.LANCZOS)
        return at_padded.crop((pad_x, pad_y, pad_x + ow, pad_y + oh))
    cropped = gemini_out.crop((left, top, right, bottom))
    if cropped.size != (ow, oh):
        cropped = cropped.resize((ow, oh), Image.LANCZOS)
    return cropped
