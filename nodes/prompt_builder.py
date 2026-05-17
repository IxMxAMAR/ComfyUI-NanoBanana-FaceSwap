"""FaceSwapPromptBuilder - preview/override the prompt sent to Gemini."""

import os, sys
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from faceswap import prompts


class FaceSwapPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (list(prompts.SCOPES), {"default": "face"}),
                "pathway": (list(prompts.PATHWAYS), {"default": "whole"}),
                "custom_hint": ("STRING", {"multiline": True, "default": ""}),
                "n_refs": ("INT", {"default": 1, "min": 0, "max": 16,
                    "tooltip": "Number of reference images you plan to pass "
                               "to the swap node. Drives the ref-count-"
                               "conditional usage instruction (1 ref vs "
                               "multi-angle triangulation)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, scope, pathway, custom_hint, n_refs):
        return (prompts.build(scope=scope, custom_hint=custom_hint,
                              pathway=pathway, n_refs=n_refs),)


NODE_CLASS_MAPPINGS = {"FaceSwapPromptBuilder": FaceSwapPromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceSwapPromptBuilder": "Face Swap Prompt Builder"}
