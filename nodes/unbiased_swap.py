"""NanoBanana Unbiased Face Swap - Pathway D node.

Mirrors the exact API request shape of the FashionGUI try_face_match.py
pipeline that empirically passes Gemini's celebrity classifier:
- system_instruction with "biometric compositing engine" persona
- JSON-structured user prompt
- response_modalities=["IMAGE"] only (no TEXT escape hatch)
- NO explicit safety_settings (uses Gemini defaults)
- Flat parts list (no Content(role=) wrapper)
- NO text labels between image parts
"""

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
BATCH_AXES = ["target", "identity"]
DETECTORS = ["auto", "mediapipe", "opencv_yunet", "gemini_bbox"]
IMAGE_SIZES = ["1K", "2K", "4K"]
THINKING_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
COMPOSITE_METHODS = ["feather", "laplacian"]


class NanoBananaUnbiasedFaceSwap:
    """Mask-inpaint with the request shape that bypasses Gemini's celebrity
    classifier in practice. Use when standard Pathway C is refused on
    public-figure references.

    Post-process stages (LAB / grain / sharpness / Laplacian / lighting) are
    OFF by default so the first test isolates whether the request-shape change
    alone got past safety. Enable `apply_integrate` to layer them on once you
    confirm the call succeeds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": (IMAGE_MODELS, {"default": IMAGE_MODELS[0]}),
                "target_image": ("IMAGE",),
                "identity_1": ("IMAGE",),
                "scope": (SCOPES, {"default": "face"}),
                "custom_hint": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "batch_axis": (BATCH_AXES, {"default": "target"}),
                "detector": (DETECTORS, {"default": "auto"}),
                "mask_dilate_px": ("INT", {"default": 25, "min": 0, "max": 128}),
                "mask_feather_px": ("INT", {"default": 16, "min": 0, "max": 128}),
                "image_size": (IMAGE_SIZES, {"default": "2K"}),
                "thinking_level": (THINKING_LEVELS, {"default": "NONE"}),
                "apply_integrate": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable post-process integration stages "
                               "(LAB color match, grain match, sharpness "
                               "match, optional lighting transfer). "
                               "Defaults OFF so the first call isolates the "
                               "request-shape change."}),
                "lab_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "composite_method": (COMPOSITE_METHODS, {"default": "feather"}),
                "match_directional_lighting": ("BOOLEAN", {"default": False,
                    "tooltip": "Same-character refinement only. Steals the "
                               "original face's broad lighting envelope and "
                               "applies it to Gemini's face. Imposes the "
                               "original person's skull/brow geometry on the "
                               "new identity in cross-identity swaps."}),
            },
            "optional": {
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

    def run(self, api_key, model, target_image, identity_1, scope, custom_hint,
            seed, batch_axis, detector, mask_dilate_px, mask_feather_px,
            image_size, thinking_level, apply_integrate, lab_strength,
            grain_strength, composite_method, match_directional_lighting,
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
        extras = [t for t in (identity_2, identity_3, identity_4,
                              identity_5, identity_6) if t is not None]
        extra_refs = [tensor_utils.image_tensor_to_pil_list(t)[0] for t in extras]

        from faceswap import helpers as _h
        iter_count, iter_get = _h.build_batch_iter(
            batch_axis, target_pils, ident1_pils, extra_refs)

        out_images, out_masks, statuses, debug_imgs = [], [], [], []
        for i in range(iter_count):
            tgt, refs = iter_get(i)
            res = backend.swap_unbiased(
                target=tgt, refs=refs, scope=scope, custom_hint=custom_hint,
                model=model, seed=seed,
                detector=detector,
                mask_dilate_px=mask_dilate_px, mask_feather_px=mask_feather_px,
                image_size=image_size, thinking_level=thinking_level,
                apply_integrate=apply_integrate,
                lab_strength=lab_strength, grain_strength=grain_strength,
                composite_method=composite_method,
                match_directional_lighting=match_directional_lighting,
            )
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


NODE_CLASS_MAPPINGS = {"NanoBananaUnbiasedFaceSwap": NanoBananaUnbiasedFaceSwap}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaUnbiasedFaceSwap": "Nano Banana - Unbiased Face Swap",
}
