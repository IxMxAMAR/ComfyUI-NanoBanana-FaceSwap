"""Pathway-B crop math: expand bbox per scope, square around face center,
extract with edge-replicate padding for out-of-bounds regions.

CRITICAL: do NOT clamp the expanded bbox to image bounds before squaring.
Clamping at that stage shifts the face center away from the actual face center,
which causes the swapped head to land off-center when composited back.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
from PIL import Image


class ExpandRatios:
    _MAP = {
        "face":         (0.25, 0.25, 0.25, 0.25),
        "head":         (0.60, 0.40, 0.40, 0.40),
        "head+styling": (1.00, 0.80, 0.70, 0.70),
    }

    @classmethod
    def for_scope(cls, scope: str) -> Tuple[float, float, float, float]:
        if scope not in cls._MAP:
            raise ValueError(f"unknown scope {scope!r}; expected one of {list(cls._MAP)}")
        return cls._MAP[scope]


def expand_bbox(rect: Tuple[int, int, int, int], scope: str,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    up, down, left, right = ExpandRatios.for_scope(scope)
    x1, y1, x2, y2 = rect
    w = x2 - x1; h = y2 - y1
    new_x1 = int(round(x1 - left * w))
    new_x2 = int(round(x2 + right * w))
    new_y1 = int(round(y1 - up * h))
    new_y2 = int(round(y2 + down * h))
    return (new_x1, new_y1, new_x2, new_y2)


def square_around_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    w = x2 - x1; h = y2 - y1
    s = max(w, h)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - s / 2.0))
    ny1 = int(round(cy - s / 2.0))
    nx2 = nx1 + s
    ny2 = ny1 + s
    return (nx1, ny1, nx2, ny2)


def extract_replicate(img: Image.Image,
                      sq: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = sq
    s = x2 - x1
    if s <= 0 or (y2 - y1) != s:
        raise ValueError(f"sq must be a positive square: got {sq}")

    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        arr = cv2.copyMakeBorder(arr, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_REPLICATE)
        x1 += pad_left; x2 += pad_left
        y1 += pad_top;  y2 += pad_top

    crop = arr[y1:y2, x1:x2]
    return Image.fromarray(crop, mode="RGB")
