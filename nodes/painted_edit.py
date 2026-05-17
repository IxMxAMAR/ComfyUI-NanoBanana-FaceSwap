"""NanoBanana Painted Edit - localized edit of a user-painted region.

Wire a LoadImage (paint a mask via right-click -> Open in MaskEditor) into
target_image+mask, then add an edit prompt and optional reference images.
Crops around the painted region, sends to Gemini, composites back.
"""

from __future__ import annotations

import os, sys
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

import numpy as np
import torch
from PIL import Image

from faceswap.backend import FaceSwapBackend
from faceswap import tensor_utils
from faceswap.node_utils import resolve_api_key


IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]
IMAGE_SIZES = ["1K", "2K", "4K"]
THINKING_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
COMPOSITE_METHODS = ["laplacian", "feather"]
OBSCURE_MODES = ["off", "blur", "mosaic", "neutral"]
EDIT_MODES = ["blend", "additive"]


class NanoBananaPaintedEdit:
    """Edit a user-painted region of an image with a text prompt + optional refs.

    Workflow:
      LoadImage (paint mask via right-click -> Open in MaskEditor)
       -> IMAGE+MASK -> NanoBananaPaintedEdit -> PreviewImage
                       ^                ^
                       |          edit_prompt = "small rose tattoo"
                       |
              optional identity_1..6 = reference images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": (IMAGE_MODELS, {"default": IMAGE_MODELS[0]}),
                "target_image": ("IMAGE",),
                "mask": ("MASK",),
                "edit_prompt": ("STRING", {"multiline": True,
                    "default": "Add a small rose tattoo to the painted region."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "thinking_level": (THINKING_LEVELS, {"default": "NONE"}),
                "edit_mode": (EDIT_MODES, {"default": "blend",
                    "tooltip": "blend = tuned for face-swap / refining a "
                               "region to look native (LAB+sharpness on, "
                               "Laplacian composite). additive = tuned for "
                               "tattoos / logos / objects you WANT to stand "
                               "out crisply (LAB off, sharpness off, feather "
                               "composite, grain still on)."}),
                "crop_tightness_pct": ("INT", {"default": 100, "min": 0, "max": 200,
                    "tooltip": "Padding around the painted mask bbox before "
                               "cropping. 0 = mask bbox only (tightest, best "
                               "for evading safety filters on revealing "
                               "content). 100% = bbox-sized padding per side "
                               "(more scene context for the model)."}),
                "obscure_outside_mask": (OBSCURE_MODES, {"default": "off",
                    "tooltip": "Fill the crop's non-painted area before "
                               "sending. 'blur' = heavy Gaussian (model gets "
                               "skin tone + lighting but no NSFW shapes). "
                               "'mosaic' = pixelate. 'neutral' = solid skin "
                               "tone. Use when editing near revealing "
                               "content that trips IMAGE_SAFETY."}),
                "obscure_blur_px": ("INT", {"default": 40, "min": 4, "max": 200,
                    "step": 2,
                    "tooltip": "Blur radius (or mosaic block size) for "
                               "obscure_outside_mask. Bigger = more hidden."}),
                "composite_dilate_px": ("INT", {"default": 0, "min": 0, "max": 40,
                    "tooltip": "Expand the painted mask outward before feather. "
                               "0 = composite exactly inside the painted region."}),
                "composite_feather_px": ("INT", {"default": 24, "min": 0, "max": 128}),
                "image_size": (IMAGE_SIZES, {"default": "2K"}),
                "color_match": ("BOOLEAN", {"default": True}),
                "grain_match": ("BOOLEAN", {"default": True}),
                "sharpness_match": ("BOOLEAN", {"default": True}),
                "composite_method": (COMPOSITE_METHODS, {"default": "laplacian"}),
                "lab_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "identity_1": ("IMAGE",),
                "identity_2": ("IMAGE",),
                "identity_3": ("IMAGE",),
                "identity_4": ("IMAGE",),
                "identity_5": ("IMAGE",),
                "identity_6": ("IMAGE",),
                "dry_run": ("BOOLEAN", {"default": False}),
                "timeout_ms": ("INT", {"default": 180000, "min": 5000, "max": 600000, "step": 1000}),
                "ref_cap_px": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64}),
                "auto_relax_on_refused": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "status", "mask", "debug_sheet")
    FUNCTION = "run"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, api_key, model, target_image, mask, edit_prompt, seed,
            thinking_level, edit_mode, crop_tightness_pct,
            obscure_outside_mask, obscure_blur_px,
            composite_dilate_px, composite_feather_px, image_size,
            color_match, grain_match, sharpness_match, composite_method,
            lab_strength, grain_strength,
            identity_1=None, identity_2=None, identity_3=None,
            identity_4=None, identity_5=None, identity_6=None,
            dry_run=False, timeout_ms=180000, ref_cap_px=1024,
            auto_relax_on_refused=False):
        key = resolve_api_key(api_key)
        backend = FaceSwapBackend(
            api_key=key, timeout_ms=timeout_ms, dry_run=dry_run,
            ref_cap_px=ref_cap_px,
            auto_relax_on_refused=auto_relax_on_refused,
        )

        # ComfyUI IMAGE: [B,H,W,C]. ComfyUI MASK: [B,H,W] or [H,W].
        # v1 enforces batch_size == 1 (per Gemini Pro consult).
        if target_image.dim() == 4 and target_image.shape[0] > 1:
            raise ValueError("NanoBananaPaintedEdit: batch_size>1 not supported in v1. "
                             "Process one image at a time.")

        target_pil = tensor_utils.image_tensor_to_pil_list(target_image)[0]

        # MASK tensor -> PIL L
        mask_t = mask
        if mask_t.dim() == 3:
            mask_t = mask_t[0]
        mask_arr = (mask_t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_arr, mode="L")

        # Resize mask to match target image if shapes differ (common when
        # the mask comes from a node that resamples).
        if mask_pil.size != target_pil.size:
            mask_pil = mask_pil.resize(target_pil.size, Image.NEAREST)

        refs = []
        for r in (identity_1, identity_2, identity_3, identity_4, identity_5, identity_6):
            if r is None:
                continue
            refs.append(tensor_utils.image_tensor_to_pil_list(r)[0])

        res = backend.swap_painted_edit(
            target=target_pil, mask=mask_pil, edit_prompt=edit_prompt,
            refs=refs, model=model, seed=seed,
            crop_tightness_pct=crop_tightness_pct,
            composite_dilate_px=composite_dilate_px,
            composite_feather_px=composite_feather_px,
            image_size=image_size, thinking_level=thinking_level,
            color_match=color_match, grain_match=grain_match,
            sharpness_match=sharpness_match,
            composite_method=composite_method,
            lab_strength=lab_strength, grain_strength=grain_strength,
            obscure_outside_mask=obscure_outside_mask,
            obscure_blur_px=obscure_blur_px,
            edit_mode=edit_mode,
        )

        image_t = tensor_utils.pil_to_image_tensor([res.image])
        # Build MASK output tensor [1,H,W]
        out_mask_arr = np.asarray(res.mask.convert("L"), dtype=np.float32) / 255.0
        mask_out = torch.from_numpy(out_mask_arr).unsqueeze(0)
        debug_t = tensor_utils.pil_to_image_tensor([res.debug_sheet])
        return (image_t, res.status, mask_out, debug_t)


NODE_CLASS_MAPPINGS = {"NanoBananaPaintedEdit": NanoBananaPaintedEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaPaintedEdit": "Nano Banana - Painted Edit"}
