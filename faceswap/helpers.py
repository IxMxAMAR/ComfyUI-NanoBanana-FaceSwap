"""Cross-node helpers shared by the swap-node wrappers.

Extracted from per-node duplication identified in v0.2.0 review. Keeps the
node files thin and concentrates batch/cost/cap logic in one place so future
changes don't drift across N files.
"""

from __future__ import annotations
import logging
from typing import Callable, Iterable, List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def build_batch_iter(batch_axis: str, target_pils: List[Image.Image],
                     ident1_pils: List[Image.Image],
                     extra_refs: List[Image.Image]
                     ) -> Tuple[int, Callable[[int], Tuple[Image.Image, List[Image.Image]]]]:
    """Resolve the (iter_count, iter_get) pair from the batch axis selector.

    `batch_axis="target"` iterates each frame of the target tensor against
    a fixed set of identity refs. `batch_axis="identity"` iterates each
    frame of identity_1 against the (fixed) first target frame.

    Returns (iter_count, iter_get(i) -> (target_pil, refs_list)).
    """
    if batch_axis == "target":
        iter_count = len(target_pils)
        refs = [ident1_pils[0]] + extra_refs

        def iter_get(i: int):
            return target_pils[i], refs

        return iter_count, iter_get
    if batch_axis == "identity":
        if not target_pils:
            raise ValueError("target_pils must be non-empty")
        fixed_target = target_pils[0]
        iter_count = len(ident1_pils)

        def iter_get(i: int):
            return fixed_target, [ident1_pils[i]] + extra_refs

        return iter_count, iter_get
    raise ValueError(f"unknown batch_axis {batch_axis!r}")


# Max longest-edge for an identity reference before we downscale before send.
# Empirically: identity transfer doesn't improve with 4K-pore refs over 1024px
# refs, while bandwidth + base64 cost scales quadratically. 1024 is a sweet
# spot per the Gemini Pro review 2026-05-17.
DEFAULT_REF_CAP_PX = 1024


def cap_reference_size(image: Image.Image, max_edge: int = DEFAULT_REF_CAP_PX
                       ) -> Image.Image:
    """Downscale an identity reference if its longest edge exceeds `max_edge`.
    No-op if already small enough. Uses LANCZOS for the downsample (high
    quality) since this is one-time per call, not per-frame.
    """
    if max_edge <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)


def cap_reference_list(refs: Iterable[Image.Image], max_edge: int = DEFAULT_REF_CAP_PX
                        ) -> List[Image.Image]:
    return [cap_reference_size(r, max_edge) for r in refs]


# ----- Cost estimation -----
# Indicative USD per output-image as of 2026-05. Pricing actually depends on
# input/output token count for image-edit models too, but the dominant cost
# component for these workflows is the generated image. Numbers are
# best-effort and meant for UX (informational), not billing.
_COST_PER_OUTPUT_USD = {
    # Flash image-preview: input+output ~ free tier today; informational only
    "gemini-3.1-flash-image-preview": 0.039,    # nominal
    "gemini-3-pro-image-preview":     0.12,     # higher quality, higher tier
    "gemini-2.5-flash-image":         0.039,
}


def estimate_cost_usd(model: str, n_calls: int = 1) -> float:
    """Best-effort per-call cost estimate for status-line UX."""
    return _COST_PER_OUTPUT_USD.get(model, 0.04) * max(0, int(n_calls))


def format_cost_suffix(model: str, n_calls: int = 1) -> str:
    cost = estimate_cost_usd(model, n_calls)
    if cost <= 0:
        return ""
    return f" | ~${cost:.3f}"
