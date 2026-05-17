"""Shared helpers used by both ComfyUI swap node wrappers.

Lives under `faceswap/` (not `nodes/`) so node files can import via the
unambiguous `faceswap.node_utils` path. Importing across `nodes/X` would
collide with ComfyUI's own top-level `nodes.py`.
"""

from __future__ import annotations
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image


def resolve_api_key(api_key_input: str) -> str:
    """Resolve a Gemini API key from (in order):
    1. Direct node input
    2. GEMINI_API_KEY env var
    3. <pack_root>/settings.json `nanobanana_key` or `gemini_api_key` field
    """
    key = (api_key_input or "").strip()
    if key:
        return key
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env

    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(pack_root, "settings.json"),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for field in ("nanobanana_key", "gemini_api_key", "api_key"):
                v = (data.get(field) or "").strip()
                if v:
                    return v
        except (OSError, json.JSONDecodeError):
            continue

    raise ValueError(
        "Gemini API key required. Provide via the node's api_key input, "
        "the GEMINI_API_KEY environment variable, or a settings.json file "
        "in the pack root with a 'nanobanana_key' field."
    )


def stack_masks(masks: List[Image.Image]) -> torch.Tensor:
    """Stack a list of L-mode PIL masks into a ComfyUI MASK tensor (B,H,W)."""
    arrs = [np.asarray(m.convert("L"), dtype=np.float32) / 255.0 for m in masks]
    max_h = max(a.shape[0] for a in arrs)
    max_w = max(a.shape[1] for a in arrs)
    padded = []
    for a in arrs:
        if a.shape != (max_h, max_w):
            buf = np.zeros((max_h, max_w), dtype=np.float32)
            buf[:a.shape[0], :a.shape[1]] = a
            a = buf
        padded.append(a)
    return torch.from_numpy(np.stack(padded, axis=0))
