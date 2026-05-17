import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from PIL import Image
import pytest

from faceswap.backend import FaceSwapBackend, SwapResult
from nodes.painted_edit import NanoBananaPaintedEdit


def _img(size=(256, 256), color=(128, 128, 128)):
    return Image.new("RGB", size, color)


def _fake_response_with_image(pil):
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    part = MagicMock(); part.inline_data.data = buf.getvalue(); part.inline_data.mime_type = "image/png"
    cand = MagicMock(); cand.content.parts = [part]; cand.finish_reason = None
    resp = MagicMock(); resp.candidates = [cand]; resp.prompt_feedback = None
    return resp


@pytest.fixture
def backend():
    return FaceSwapBackend(api_key="FAKE")


def _mask_with_dot(size, x, y, r):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    yy, xx = np.ogrid[:size[1], :size[0]]
    arr[(yy - y) ** 2 + (xx - x) ** 2 < r ** 2] = 255
    return Image.fromarray(arr, mode="L")


def test_empty_mask_bypasses_api(backend):
    target = _img((256, 256))
    mask = Image.new("L", (256, 256), 0)  # all zero
    with patch.object(backend, "_call_generate") as mocked_call:
        res = backend.swap_painted_edit(
            target=target, mask=mask, edit_prompt="x", refs=[],
            model="x", seed=0, crop_tightness_pct=100,
            composite_dilate_px=0, composite_feather_px=24,
        )
        mocked_call.assert_not_called()
    assert res.status == "ERROR:empty_mask"


def test_painted_edit_ok_path(backend):
    target = _img((512, 512), color=(100, 150, 200))
    mask = _mask_with_dot((512, 512), 256, 256, 50)
    edited_crop = _img((1024, 1024), color=(200, 50, 50))
    with patch.object(backend, "_call_generate",
                      return_value=_fake_response_with_image(edited_crop)):
        res = backend.swap_painted_edit(
            target=target, mask=mask, edit_prompt="red blob",
            refs=[], model="x", seed=0, crop_tightness_pct=50,
            composite_dilate_px=0, composite_feather_px=8,
        )
    assert res.status == "OK"
    assert res.image.size == target.size
    out_arr = np.asarray(res.image)
    # Painted region should now be redder than surroundings
    center_r = out_arr[256, 256, 0]
    corner_r = out_arr[10, 10, 0]
    assert center_r > corner_r + 30


def test_painted_edit_bounds_safe_when_mask_near_edge(backend):
    """Mask near the corner -> squared crop extends past image bounds.
    Must not crash. Composite must respect actual image bounds."""
    target = _img((256, 256), color=(50, 50, 50))
    mask = _mask_with_dot((256, 256), 8, 8, 30)  # corner-ish
    edited_crop = _img((512, 512), color=(255, 255, 255))
    with patch.object(backend, "_call_generate",
                      return_value=_fake_response_with_image(edited_crop)):
        res = backend.swap_painted_edit(
            target=target, mask=mask, edit_prompt="white",
            refs=[], model="x", seed=0, crop_tightness_pct=200,
            composite_dilate_px=0, composite_feather_px=4,
        )
    assert res.status == "OK"
    assert res.image.size == target.size


def test_node_input_types_schema():
    schema = NanoBananaPaintedEdit.INPUT_TYPES()
    req = schema["required"]
    for k in ("api_key", "model", "target_image", "mask", "edit_prompt",
              "crop_tightness_pct", "composite_dilate_px",
              "composite_feather_px", "color_match", "grain_match",
              "sharpness_match", "composite_method",
              "lab_strength", "grain_strength"):
        assert k in req, f"missing required input: {k}"
    opt = schema["optional"]
    for k in ("identity_1", "identity_2", "identity_3", "identity_4", "identity_5", "identity_6"):
        assert k in opt


def test_node_rejects_batch_size_greater_than_one():
    n = NanoBananaPaintedEdit()
    big = torch.zeros((2, 32, 32, 3), dtype=torch.float32)
    mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch_size"):
        n.run(api_key="FAKE", model="gemini-3.1-flash-image-preview",
              target_image=big, mask=mask, edit_prompt="x",
              seed=0, thinking_level="NONE", edit_mode="blend",
              crop_tightness_pct=100,
              obscure_outside_mask="off", obscure_blur_px=40,
              composite_dilate_px=0, composite_feather_px=24,
              image_size="2K", color_match=False, grain_match=False,
              sharpness_match=False, composite_method="feather",
              lab_strength=0.6, grain_strength=1.0)


def test_node_resizes_mask_if_different_size():
    """Mask resizing to match image is required when MaskEditor outputs
    a differently-sized mask vs the image (rare but happens)."""
    n = NanoBananaPaintedEdit()
    target_t = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
    # Mask at half resolution
    mask_t = torch.zeros((1, 32, 32), dtype=torch.float32)
    mask_t[0, 12:20, 12:20] = 1.0
    fake = SwapResult(image=Image.new("RGB", (64, 64), (100, 100, 100)),
                      status="OK",
                      mask=Image.new("L", (64, 64), 128),
                      debug_sheet=Image.new("RGB", (256, 256), (0, 0, 0)))
    with patch("faceswap.backend.FaceSwapBackend.swap_painted_edit", return_value=fake):
        out = n.run(api_key="FAKE", model="gemini-3.1-flash-image-preview",
                    target_image=target_t, mask=mask_t, edit_prompt="x",
                    seed=0, thinking_level="NONE", edit_mode="blend",
                    crop_tightness_pct=100,
                    obscure_outside_mask="off", obscure_blur_px=40,
                    composite_dilate_px=0, composite_feather_px=24,
                    image_size="2K", color_match=True, grain_match=True,
                    sharpness_match=True, composite_method="laplacian",
                    lab_strength=0.6, grain_strength=1.0)
    img_t, status, mask_out, debug_t = out
    assert img_t.shape == (1, 64, 64, 3)
    assert mask_out.shape == (1, 64, 64)
