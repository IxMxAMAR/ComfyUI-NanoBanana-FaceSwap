"""NanoBanana Face Swap (Crop & Composite) - Pathway B node."""

from __future__ import annotations

import os, sys
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from faceswap.backend import FaceSwapBackend
from faceswap import tensor_utils
from faceswap.node_utils import resolve_api_key, stack_masks


IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]
SCOPES = ["face", "head", "head+styling"]
GRID_MODES = ["separate_refs", "auto_sheet"]
BATCH_AXES = ["target", "identity"]
SAFETY_LEVELS = ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"]
DETECTORS = ["auto", "mediapipe", "opencv_yunet", "gemini_bbox"]
IMAGE_SIZES = ["1K", "2K", "4K"]
COMPOSITE_METHODS = ["laplacian", "feather"]


class NanoBananaCropSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": (IMAGE_MODELS, {"default": IMAGE_MODELS[0]}),
                "target_image": ("IMAGE",),
                "identity_1": ("IMAGE",),
                "scope": (SCOPES, {"default": "head"}),
                "grid_mode": (GRID_MODES, {"default": "separate_refs"}),
                "custom_hint": ("STRING", {"multiline": True, "default": ""}),
                "safety_threshold": (SAFETY_LEVELS, {"default": "BLOCK_NONE"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "batch_axis": (BATCH_AXES, {"default": "target"}),
                "detector": (DETECTORS, {"default": "auto"}),
                "crop_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "histogram_match": ("BOOLEAN", {"default": False,
                    "tooltip": "Legacy RGB histogram match. Superseded by color_match below; keep off unless debugging."}),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 256}),
                "image_size": (IMAGE_SIZES, {"default": "2K"}),
                "color_match": ("BOOLEAN", {"default": True,
                    "tooltip": "LAB color match with face-polygon stats. Fixes skin-tone cast."}),
                "grain_match": ("BOOLEAN", {"default": True,
                    "tooltip": "Add synthetic Gaussian noise matching the target's cheek variance."}),
                "sharpness_match": ("BOOLEAN", {"default": True,
                    "tooltip": "Blur Gemini's face if it's sharper than the body. Never sharpens."}),
                "composite_method": (COMPOSITE_METHODS, {"default": "laplacian"}),
                "match_directional_lighting": ("BOOLEAN", {"default": False,
                    "tooltip": "Same-character mode. Imposes the original face's lighting envelope on the new face. Corrupts identity for cross-identity swaps."}),
                "lab_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "identity_2": ("IMAGE",),
                "identity_3": ("IMAGE",),
                "identity_4": ("IMAGE",),
                "identity_5": ("IMAGE",),
                "identity_6": ("IMAGE",),
                "dry_run": ("BOOLEAN", {"default": False,
                    "tooltip": "Skip API call; return a structured preview of the prompt + parts."}),
                "timeout_ms": ("INT", {"default": 180000, "min": 5000, "max": 600000, "step": 1000}),
                "ref_cap_px": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Downscale ref longest-edge to this px. 0=off."}),
                "auto_relax_on_refused": ("BOOLEAN", {"default": False,
                    "tooltip": "On refusal, retry once with no safety_settings."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "status", "mask", "debug_sheet")
    FUNCTION = "run"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, api_key, model, target_image, identity_1, scope, grid_mode,
            custom_hint, safety_threshold, seed, batch_axis,
            detector, crop_size, histogram_match, feather_px,
            image_size="2K",
            color_match=True, grain_match=True, sharpness_match=True,
            composite_method="laplacian", match_directional_lighting=False,
            lab_strength=0.6, grain_strength=1.0,
            identity_2=None, identity_3=None, identity_4=None,
            identity_5=None, identity_6=None,
            dry_run=False, timeout_ms=180000, ref_cap_px=1024,
            auto_relax_on_refused=False):
        key = resolve_api_key(api_key)
        backend = FaceSwapBackend(
            api_key=key, timeout_ms=timeout_ms, dry_run=dry_run,
            ref_cap_px=ref_cap_px,
            auto_relax_on_refused=auto_relax_on_refused,
        )

        target_pils = tensor_utils.image_tensor_to_pil_list(target_image)
        ident1_pils = tensor_utils.image_tensor_to_pil_list(identity_1)
        identity_tensors = [t for t in (identity_2, identity_3, identity_4,
                                        identity_5, identity_6) if t is not None]
        extra_refs = [tensor_utils.image_tensor_to_pil_list(t)[0] for t in identity_tensors]

        from faceswap import helpers as _h
        iter_count, iter_get = _h.build_batch_iter(
            batch_axis, target_pils, ident1_pils, extra_refs)

        out_images, out_masks, statuses, debug_imgs = [], [], [], []
        for i in range(iter_count):
            tgt, refs = iter_get(i)
            res = backend.swap_crop(target=tgt, refs=refs, scope=scope,
                                    custom_hint=custom_hint, model=model, seed=seed,
                                    grid_mode=grid_mode, safety_threshold=safety_threshold,
                                    detector=detector, crop_size=crop_size,
                                    histogram_match=histogram_match, feather_px=feather_px,
                                    image_size=image_size,
                                    color_match=color_match, grain_match=grain_match,
                                    sharpness_match=sharpness_match,
                                    composite_method=composite_method,
                                    match_directional_lighting=match_directional_lighting,
                                    lab_strength=lab_strength, grain_strength=grain_strength)
            out_images.append(res.image)
            out_masks.append(res.mask)
            statuses.append(res.status)
            debug_imgs.append(res.debug_sheet)

        image_t = tensor_utils.pil_to_image_tensor(out_images)
        mask_t = stack_masks(out_masks)
        debug_t = tensor_utils.pil_to_image_tensor(debug_imgs)
        status = ";".join(statuses) if len(statuses) > 1 else statuses[0]
        if not dry_run:
            status = status + _h.format_cost_suffix(model, n_calls=iter_count)
        return (image_t, status, mask_t, debug_t)


NODE_CLASS_MAPPINGS = {"NanoBananaCropSwap": NanoBananaCropSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaCropSwap": "Nano Banana - Face Swap (Crop & Composite)"}
