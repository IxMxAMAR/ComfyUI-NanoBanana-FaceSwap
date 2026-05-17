import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from nodes.whole_image_swap import NanoBananaWholeImageSwap
from faceswap.backend import SwapResult


def _img_tensor(size=(16, 16), color=(0.5, 0.5, 0.5), batch=1):
    arr = np.zeros((batch, size[1], size[0], 3), dtype=np.float32)
    arr[..., 0] = color[0]; arr[..., 1] = color[1]; arr[..., 2] = color[2]
    return torch.from_numpy(arr)


def test_node_returns_correct_tuple_shape():
    n = NanoBananaWholeImageSwap()
    target = _img_tensor(color=(0.5, 0.5, 0.5))
    ident = _img_tensor(color=(1.0, 0.0, 0.0))
    fake_result = SwapResult(
        image=Image.new("RGB", (16, 16), (0, 200, 0)),
        status="OK",
        mask=Image.new("L", (16, 16), 128),
        debug_sheet=Image.new("RGB", (32, 16), (0, 0, 0)),
    )
    with patch("faceswap.backend.FaceSwapBackend.swap_whole", return_value=fake_result):
        out = n.run(api_key="FAKE_KEY_FOR_TEST", model="gemini-3.1-flash-image-preview",
                    target_image=target, identity_1=ident,
                    identity_2=None, identity_3=None, identity_4=None,
                    identity_5=None, identity_6=None,
                    scope="face", grid_mode="separate_refs", custom_hint="",
                    safety_threshold="BLOCK_NONE", seed=0, batch_axis="target")
    img_t, status, mask_t, debug_t = out
    assert isinstance(img_t, torch.Tensor) and img_t.shape == (1, 16, 16, 3)
    # status now carries an appended cost-estimate suffix
    assert status.startswith("OK")
    assert mask_t.shape == (1, 16, 16)
    assert isinstance(debug_t, torch.Tensor)


def test_node_input_types_schema_has_required_fields():
    schema = NanoBananaWholeImageSwap.INPUT_TYPES()
    req = schema["required"]
    assert "api_key" in req and "model" in req and "target_image" in req
    assert "identity_1" in req
    assert "scope" in req
    opt = schema["optional"]
    for k in ("identity_2", "identity_3", "identity_4", "identity_5", "identity_6"):
        assert k in opt


def test_node_is_changed_returns_nan():
    val = NanoBananaWholeImageSwap.IS_CHANGED()
    assert val != val  # NaN check
