"""IdentityRefObfuscator - obfuscate identity refs to defeat Gemini's
celebrity-recognition classifier while preserving identity signal.

Wire between a LoadImage and a swap node's identity_N input.
"""

from __future__ import annotations
import os, sys

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from faceswap import ref_obfuscate as _obf, tensor_utils


class IdentityRefObfuscator:
    """Apply blur + small perspective warp + LAB shift to identity refs.

    Use when Gemini refuses the request because the reference image is a
    recognized public figure. The output is intentionally a slight
    distortion of the input - enough to fall below the celebrity-recognition
    classifier's confidence threshold, but still recognizably the same
    person to Gemini's image-edit pathway.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "Overall distortion strength. "
                                                   "0.3-0.5 typically defeats the "
                                                   "celebrity classifier without "
                                                   "destroying identity. >0.7 may "
                                                   "lose recognizable features."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                  "tooltip": "0 = random each run; non-zero "
                                             "produces deterministic output."}),
                "apply_blur": ("BOOLEAN", {"default": True}),
                "apply_warp": ("BOOLEAN", {"default": True}),
                "apply_color": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("identity_obfuscated",)
    FUNCTION = "run"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, identity_image, strength, seed, apply_blur, apply_warp, apply_color):
        pils = tensor_utils.image_tensor_to_pil_list(identity_image)
        seed_arg = seed if seed and seed > 0 else None
        outs = [_obf.obfuscate(p, strength=strength, seed=seed_arg,
                                apply_blur=apply_blur, apply_warp=apply_warp,
                                apply_color=apply_color) for p in pils]
        return (tensor_utils.pil_to_image_tensor(outs),)


NODE_CLASS_MAPPINGS = {"IdentityRefObfuscator": IdentityRefObfuscator}
NODE_DISPLAY_NAME_MAPPINGS = {"IdentityRefObfuscator": "Identity Ref Obfuscator"}
