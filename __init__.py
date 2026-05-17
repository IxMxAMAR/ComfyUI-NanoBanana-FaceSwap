"""ComfyUI-NanoBanana-FaceSwap - face/head replacement via Nano Banana 2."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ComfyUI loads custom-node packages with `spec_from_file_location(... submodule_search_locations=[pack_dir])`,
# which makes the `from .nodes import ...` relative import resolve correctly.
# The fallback covers direct script invocation (tests run from a separate conftest path).
try:
    from .nodes import (
        whole_image_swap, crop_composite_swap, inpaint_swap, unbiased_swap,
        identity_sheet, prompt_builder, ref_obfuscate as ref_obfuscate_node,
        painted_edit, load_image_paint,
    )
except ImportError:
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import importlib
    whole_image_swap = importlib.import_module("nodes.whole_image_swap")
    crop_composite_swap = importlib.import_module("nodes.crop_composite_swap")
    inpaint_swap = importlib.import_module("nodes.inpaint_swap")
    unbiased_swap = importlib.import_module("nodes.unbiased_swap")
    identity_sheet = importlib.import_module("nodes.identity_sheet")
    prompt_builder = importlib.import_module("nodes.prompt_builder")
    ref_obfuscate_node = importlib.import_module("nodes.ref_obfuscate")
    painted_edit = importlib.import_module("nodes.painted_edit")
    load_image_paint = importlib.import_module("nodes.load_image_paint")

for _mod in (whole_image_swap, crop_composite_swap, inpaint_swap, unbiased_swap,
             identity_sheet, prompt_builder, ref_obfuscate_node, painted_edit,
             load_image_paint):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

# Tell ComfyUI's web server to serve frontend JS from the pack's web/ folder.
# Must be set at the package level (this file).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
