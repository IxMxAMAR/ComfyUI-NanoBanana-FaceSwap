"""NanoBanana Face Swap (Whole Image) - Pathway A node."""

from __future__ import annotations

# Pack root must already be on sys.path for `faceswap.*` imports to resolve.
# ComfyUI's loader does NOT add custom_node dirs to sys.path automatically, so
# we ensure it here without using `from nodes import ...` (would collide with
# ComfyUI's own top-level nodes.py).
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
IMAGE_SIZES = ["1K", "2K", "4K"]


class NanoBananaWholeImageSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": (IMAGE_MODELS, {"default": IMAGE_MODELS[0]}),
                "target_image": ("IMAGE",),
                "identity_1": ("IMAGE",),
                "scope": (SCOPES, {"default": "face"}),
                "grid_mode": (GRID_MODES, {"default": "separate_refs"}),
                "custom_hint": ("STRING", {"multiline": True, "default": ""}),
                "safety_threshold": (SAFETY_LEVELS, {"default": "BLOCK_NONE"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "batch_axis": (BATCH_AXES, {"default": "target"}),
                "image_size": (IMAGE_SIZES, {"default": "2K"}),
            },
            "optional": {
                "identity_2": ("IMAGE",),
                "identity_3": ("IMAGE",),
                "identity_4": ("IMAGE",),
                "identity_5": ("IMAGE",),
                "identity_6": ("IMAGE",),
                "dry_run": ("BOOLEAN", {"default": False,
                    "tooltip": "Skip the API call and return a structured "
                               "preview of the prompt + parts that would have "
                               "been sent. Burns no quota; useful for "
                               "debugging custom_hint and prompt builders."}),
                "timeout_ms": ("INT", {"default": 180000, "min": 5000, "max": 600000,
                    "step": 1000,
                    "tooltip": "Per-call API timeout in milliseconds. Lower "
                               "(e.g. 30000) lets you fail fast on stuck "
                               "calls; higher tolerates slow Pro-tier renders."}),
                "ref_cap_px": ("INT", {"default": 1024, "min": 0, "max": 4096,
                    "step": 64,
                    "tooltip": "Downscale identity reference images so their "
                               "longest edge is at most this many pixels "
                               "before send. 0 = disabled. 1024 saves "
                               "bandwidth and empirically improves identity "
                               "transfer (4K pore detail confuses the model)."}),
                "auto_relax_on_refused": ("BOOLEAN", {"default": False,
                    "tooltip": "If the model refuses, retry once with no "
                               "safety_settings (model defaults). Some "
                               "BLOCK_NONE refusals relax under defaults."}),
                            "network": ("NB_NETWORK", {"tooltip": "Optional. Wire a NanoBanana - Network Route node here to route this swap's API call through that proxy (e.g. US egress)."}),
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
            image_size="2K",
            identity_2=None, identity_3=None, identity_4=None,
            identity_5=None, identity_6=None,
            dry_run=False, timeout_ms=180000, ref_cap_px=1024,
            auto_relax_on_refused=False, network=None):
        key = resolve_api_key(api_key)
        backend = FaceSwapBackend(
            api_key=key, timeout_ms=timeout_ms, dry_run=dry_run,
            ref_cap_px=ref_cap_px,
            auto_relax_on_refused=auto_relax_on_refused,
            network=network,
        )

        identity_tensors = [t for t in (identity_1, identity_2, identity_3, identity_4,
                                        identity_5, identity_6) if t is not None]

        target_pils = tensor_utils.image_tensor_to_pil_list(target_image)
        ident1_pils = tensor_utils.image_tensor_to_pil_list(identity_1)
        extra_refs = [tensor_utils.image_tensor_to_pil_list(t)[0] for t in identity_tensors[1:]]

        from faceswap import helpers as _h
        iter_count, iter_get = _h.build_batch_iter(
            batch_axis, target_pils, ident1_pils, extra_refs)

        out_images, out_masks, statuses, debug_imgs = [], [], [], []
        for i in range(iter_count):
            tgt, refs = iter_get(i)
            res = backend.swap_whole(target=tgt, refs=refs, scope=scope,
                                     custom_hint=custom_hint, model=model, seed=seed,
                                     grid_mode=grid_mode, safety_threshold=safety_threshold,
                                     image_size=image_size)
            out_images.append(res.image)
            out_masks.append(res.mask)
            statuses.append(res.status)
            debug_imgs.append(res.debug_sheet)

        image_t = tensor_utils.pil_to_image_tensor(out_images)
        mask_t = stack_masks(out_masks)
        debug_t = tensor_utils.pil_to_image_tensor(debug_imgs)
        status = ";".join(statuses) if len(statuses) > 1 else statuses[0]
        # Append a cost-estimate suffix unless this is a dry run.
        if not dry_run:
            status = status + _h.format_cost_suffix(model, n_calls=iter_count)
        return (image_t, status, mask_t, debug_t)


NODE_CLASS_MAPPINGS = {"NanoBananaWholeImageSwap": NanoBananaWholeImageSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaWholeImageSwap": "Nano Banana - Face Swap (Whole Image)"}
