import os
import sys

# Add pack root to sys.path so `from faceswap...` and `from nodes...` resolve,
# but do NOT import the pack `__init__.py` (which uses relative imports that only
# work when ComfyUI loads it as a real package).
_PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)
