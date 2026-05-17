"""ComfyUI IMAGE/MASK tensor <-> PIL conversions.

ComfyUI conventions:
- IMAGE: torch.Tensor float32 in [0,1], shape (B, H, W, C) with C=3 (RGB)
- MASK:  torch.Tensor float32 in [0,1], shape (B, H, W) or (H, W)
"""

from __future__ import annotations
import logging
from typing import List, Union
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def image_tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]


def pil_to_image_tensor(pil: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    pils = [pil] if isinstance(pil, Image.Image) else list(pil)
    arrays = []
    for p in pils:
        if p.mode != "RGB":
            p = p.convert("RGB")
        arr = np.asarray(p, dtype=np.float32) / 255.0
        arrays.append(arr)
    # If sizes differ, pad to max H, W with zeros so torch.stack succeeds.
    # Warn so users can spot why a downstream node (e.g. VAE encode) sees
    # black bars on the edges of some batch elements.
    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)
    sizes = {a.shape[:2] for a in arrays}
    if len(sizes) > 1:
        logger.warning(
            "[faceswap] pil_to_image_tensor: batch contains mixed sizes %s; "
            "smaller frames will be zero-padded to (%d, %d)",
            sorted(sizes), max_h, max_w,
        )
    padded = []
    for a in arrays:
        if a.shape[:2] != (max_h, max_w):
            buf = np.zeros((max_h, max_w, 3), dtype=np.float32)
            buf[:a.shape[0], :a.shape[1]] = a
            a = buf
        padded.append(a)
    return torch.from_numpy(np.stack(padded, axis=0))


def mask_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 3:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pil_to_mask_tensor(pil: Image.Image) -> torch.Tensor:
    if pil.mode != "L":
        pil = pil.convert("L")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
