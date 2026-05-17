"""LoadImagePaint - LoadImage with inline mask painting on the node body.

The mask is painted directly on a canvas embedded in the node (no popup
MaskEditor). Frontend JS handles the painting; backend decodes the mask
data string and returns IMAGE + MASK tensors.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

# folder_paths is provided by ComfyUI's runtime
try:
    import folder_paths
except ImportError:
    folder_paths = None


def _get_image_path(image_name):
    """Resolve a LoadImage filename to a full path.

    Uses folder_paths.get_annotated_filepath when available (ComfyUI mid-2024
    onward); falls back to get_full_path("input", name) for older installs.
    """
    if folder_paths is None:
        return image_name
    if hasattr(folder_paths, "get_annotated_filepath"):
        return folder_paths.get_annotated_filepath(image_name)
    if hasattr(folder_paths, "get_full_path"):
        return folder_paths.get_full_path("input", image_name)
    # Last-resort fallback: assume input directory.
    # SECURITY: explicitly reject path traversal here because ancient
    # ComfyUI builds without `get_annotated_filepath` don't pre-sanitize.
    # A malicious workflow JSON could otherwise pass "../../etc/passwd".
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    full = os.path.abspath(os.path.join(input_dir, image_name))
    # Use normcase to match Windows case-insensitivity (C:\ vs c:\).
    if (os.path.normcase(full) != os.path.normcase(input_dir)
            and not os.path.normcase(full).startswith(
                os.path.normcase(input_dir) + os.sep)):
        raise ValueError(
            f"[LoadImagePaint] path traversal rejected: {image_name!r}")
    return full


def _annotated_exists(image_name):
    if folder_paths is None:
        return False
    if hasattr(folder_paths, "exists_annotated_filepath"):
        return folder_paths.exists_annotated_filepath(image_name)
    return os.path.isfile(_get_image_path(image_name))


def _list_input_images():
    """Mirror ComfyUI's LoadImage file enumeration."""
    if folder_paths is None:
        return []
    input_dir = folder_paths.get_input_directory()
    files = []
    if os.path.isdir(input_dir):
        for f in os.listdir(input_dir):
            full = os.path.join(input_dir, f)
            if os.path.isfile(full):
                files.append(f)
    return sorted(files)


class LoadImagePaint:
    """LoadImage with inline mask painting.

    Right-click → "Open in MaskEditor" not required: paint directly on the
    canvas embedded in this node. The painted mask is output as the MASK
    socket. The IMAGE socket is the loaded image (unchanged).

    Frontend JS file: web/load_image_paint.js
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_images() or ["(no files)"]
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                # mask_data is set by the JS frontend on each paint stroke.
                # Stored as base64-encoded PNG of an L-mode mask matching the
                # image's dimensions. Empty string = no mask painted.
                "mask_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "lazy": True,
                }),
                "brush_size": ("INT", {"default": 20, "min": 1, "max": 200,
                                        "step": 1,
                                        "tooltip": "Brush radius in image pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, image, mask_data, brush_size):
        path = _get_image_path(image)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return f"{path}:{mtime}:{len(mask_data)}:{hash(mask_data) if mask_data else 0}"

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if folder_paths is None:
            return True
        if not _annotated_exists(image):
            return f"Invalid image file: {image}"
        return True

    def load(self, image, mask_data, brush_size):
        # Load image - mirror ComfyUI's LoadImage behavior
        image_path = _get_image_path(image)
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGBA" and "A" in pil.getbands():
            pil = pil.convert("RGBA")
        image_rgb = pil.convert("RGB")

        img_arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(img_arr).unsqueeze(0)  # [1,H,W,3]

        # Decode mask_data
        H, W = img_arr.shape[:2]
        mask_arr = np.zeros((H, W), dtype=np.float32)
        if mask_data and mask_data.strip():
            try:
                # SECURITY: cap the payload size before decoding to defend
                # against a malicious workflow shipping a 100 MB mask_data
                # string that would OOM during base64 decode + PIL parse.
                # Typical real masks are < 200 KB; 50 MB is a generous ceiling.
                MAX_MASK_DATA_LEN = 50_000_000
                if len(mask_data) > MAX_MASK_DATA_LEN:
                    raise ValueError(
                        f"mask_data length {len(mask_data)} exceeds cap "
                        f"{MAX_MASK_DATA_LEN}; rejecting to prevent OOM"
                    )
                # mask_data may be a data URL ("data:image/png;base64,XXX") or raw base64
                payload = mask_data
                if "," in payload[:64] and "base64" in payload[:64]:
                    payload = payload.split(",", 1)[1]
                raw = base64.b64decode(payload)
                mask_pil = Image.open(io.BytesIO(raw))
                # Catch PIL decompression bombs by clamping pixel count.
                # PIL itself triggers DecompressionBombWarning past ~89.5MP
                # but a malicious mask should be rejected hard.
                if mask_pil.width * mask_pil.height > 25_000_000:  # 25MP cap
                    raise ValueError(
                        f"mask image {mask_pil.size} exceeds 25MP cap"
                    )
                mask_pil = mask_pil.convert("L")
                if mask_pil.size != (W, H):
                    mask_pil = mask_pil.resize((W, H), Image.NEAREST)
                mask_arr = np.asarray(mask_pil, dtype=np.float32) / 255.0
            except Exception as e:
                # If decoding fails, just return an empty mask. The frontend
                # may have sent corrupt data on first paint; don't crash the
                # workflow.
                print(f"[LoadImagePaint] mask decode failed: {e}; using empty mask")
                mask_arr = np.zeros((H, W), dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # [1,H,W]
        return (image_tensor, mask_tensor)


NODE_CLASS_MAPPINGS = {"LoadImagePaint": LoadImagePaint}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImagePaint": "Load Image (Paint Mask)"}
