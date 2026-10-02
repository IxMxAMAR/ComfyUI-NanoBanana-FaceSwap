"""Gemini model / image-size options (GA model IDs, per-model size limits)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from faceswap import backend as _be
from faceswap import detect as _detect
from faceswap import helpers as _h
from nodes import (whole_image_swap, crop_composite_swap, inpaint_swap,
                   unbiased_swap, painted_edit)

SWAP_NODES = [
    whole_image_swap.NanoBananaWholeImageSwap,
    crop_composite_swap.NanoBananaCropSwap,
    inpaint_swap.NanoBananaInpaintSwap,
    unbiased_swap.NanoBananaUnbiasedFaceSwap,
    painted_edit.NanoBananaPaintedEdit,
]


@pytest.mark.parametrize("node", SWAP_NODES)
def test_swap_nodes_offer_ga_models_and_512(node):
    required = node.INPUT_TYPES()["required"]
    models, opts = required["model"]
    assert models == ["gemini-3.1-flash-image", "gemini-3-pro-image",
                      "gemini-3.1-flash-lite-image"]
    assert opts["default"] == "gemini-3.1-flash-image"
    sizes, size_opts = required["image_size"]
    assert sizes == ["512", "1K", "2K", "4K"]
    assert size_opts["default"] == "2K"


def test_supported_image_size_clamps_per_model():
    f = _be._supported_image_size
    assert f("gemini-3.1-flash-lite-image", "4K") == "1K"
    assert f("gemini-3.1-flash-lite-image", "1K") == "1K"
    assert f("gemini-3-pro-image", "512") == "1K"
    assert f("gemini-3-pro-image", "4K") == "4K"
    assert f("gemini-3.1-flash-image", "512") == "512"
    assert f("gemini-3.1-flash-image", "4K") == "4K"


def _sent_image_size(model, image_size):
    backend = _be.FaceSwapBackend(api_key="FAKE")
    target = Image.new("RGB", (64, 64), (100, 100, 100))
    refs = [Image.new("RGB", (64, 64), (200, 50, 50))]
    resp = MagicMock()
    resp.candidates = []
    resp.prompt_feedback.block_reason = "OTHER"
    with patch.object(backend, "_call_generate", return_value=resp) as call:
        backend.swap_whole(target=target, refs=refs, scope="face",
                           custom_hint="", model=model, seed=0,
                           grid_mode="separate_refs",
                           safety_threshold="BLOCK_NONE", image_size=image_size)
    return call.call_args.kwargs["config"].image_config.image_size


def test_swap_whole_sends_supported_size_for_lite_model():
    assert _sent_image_size("gemini-3.1-flash-lite-image", "4K") == "1K"


def test_swap_whole_sends_512_for_flash_model():
    assert _sent_image_size("gemini-3.1-flash-image", "512") == "512"


def test_cost_estimate_follows_model_and_size():
    assert _h.estimate_cost_usd("gemini-3.1-flash-image", 1, "2K") == pytest.approx(0.101)
    assert _h.estimate_cost_usd("gemini-3-pro-image", 2, "4K") == pytest.approx(0.48)
    assert _h.estimate_cost_usd("gemini-3.1-flash-lite-image", 1, "4K") == pytest.approx(0.0336)
    assert _h.format_cost_suffix("gemini-3.1-flash-image", 1, "512") == " | ~$0.045"


def test_gemini_bbox_does_not_set_temperature():
    img = Image.new("RGB", (200, 200), (0, 0, 0))

    class _Resp:
        text = "[0.1, 0.2, 0.9, 0.95]"
        candidates = []

    client_mock = MagicMock()
    client_mock.models.generate_content.return_value = _Resp()
    with patch("google.genai.Client", return_value=client_mock):
        _detect._detect_gemini_bbox(img, api_key="FAKE")
    config = client_mock.models.generate_content.call_args.kwargs["config"]
    assert config.temperature is None
