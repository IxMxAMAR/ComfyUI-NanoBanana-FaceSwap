"""IdentitySheetComposer - compose up to 6 IMAGE inputs into a single grid."""

import os, sys
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from faceswap import sheet, tensor_utils


LAYOUTS = ["auto", "2x1", "1x2", "2x2", "3x2", "2x3", "1xN"]


class IdentitySheetComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_1": ("IMAGE",),
                "layout": (LAYOUTS, {"default": "auto"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
            "optional": {
                "identity_2": ("IMAGE",),
                "identity_3": ("IMAGE",),
                "identity_4": ("IMAGE",),
                "identity_5": ("IMAGE",),
                "identity_6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "layout_info")
    FUNCTION = "run"
    CATEGORY = "NanoBanana FaceSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, identity_1, layout, tile_size,
            identity_2=None, identity_3=None, identity_4=None,
            identity_5=None, identity_6=None):
        refs = [identity_1, identity_2, identity_3, identity_4, identity_5, identity_6]
        pils = []
        for r in refs:
            if r is None:
                continue
            pils.append(tensor_utils.image_tensor_to_pil_list(r)[0])
        out = sheet.compose(pils, tile_size=tile_size, layout=layout)
        cols, rows = sheet._resolve_layout(layout, len(pils))
        info = f"{cols}x{rows} | {len(pils)} refs | tile={tile_size}"
        return (tensor_utils.pil_to_image_tensor(out), info)


NODE_CLASS_MAPPINGS = {"IdentitySheetComposer": IdentitySheetComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"IdentitySheetComposer": "Identity Sheet Composer"}
