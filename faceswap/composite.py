"""Histogram matching + feathered alpha composite for Pathway B."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter


def match_histograms(src: Image.Image, ref: Image.Image) -> Image.Image:
    src_rgb = src.convert("RGB")
    ref_rgb = ref.convert("RGB")
    if ref_rgb.size != src_rgb.size:
        ref_rgb = ref_rgb.resize(src_rgb.size, Image.LANCZOS)

    s = np.asarray(src_rgb, dtype=np.uint8)
    r = np.asarray(ref_rgb, dtype=np.uint8)
    out = np.empty_like(s)
    for ch in range(3):
        s_hist, _ = np.histogram(s[..., ch], bins=256, range=(0, 256))
        r_hist, _ = np.histogram(r[..., ch], bins=256, range=(0, 256))
        s_cdf = np.cumsum(s_hist).astype(np.float64)
        s_cdf /= max(s_cdf[-1], 1.0)
        r_cdf = np.cumsum(r_hist).astype(np.float64)
        r_cdf /= max(r_cdf[-1], 1.0)
        lut = np.searchsorted(r_cdf, s_cdf).clip(0, 255).astype(np.uint8)
        out[..., ch] = lut[s[..., ch]]
    return Image.fromarray(out, mode="RGB")


def make_feather_mask(size: int, feather_px: int) -> Image.Image:
    if feather_px < 0:
        feather_px = 0
    inset = min(feather_px, max(1, size // 4))
    arr = np.zeros((size, size), dtype=np.uint8)
    if size - 2 * inset > 0:
        arr[inset:size - inset, inset:size - inset] = 255
    img = Image.fromarray(arr, mode="L")
    if feather_px > 0:
        img = img.filter(ImageFilter.GaussianBlur(max(1, feather_px // 2)))
    return img


def alpha_composite_at(base: Image.Image, top: Image.Image, mask: Image.Image,
                       origin: Tuple[int, int]) -> Image.Image:
    out = base.copy().convert("RGB")
    out.paste(top.convert("RGB"), origin, mask.convert("L"))
    return out
