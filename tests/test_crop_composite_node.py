import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from nodes.crop_composite_swap import NanoBananaCropSwap
from faceswap.backend import SwapResult


def _img_tensor(size=(64, 64), color=(0.5, 0.5, 0.5), batch=1):
    arr = np.zeros((batch, size[1], size[0], 3), dtype=np.float32)
    arr[..., 0] = color[0]; arr[..., 1] = color[1]; arr[..., 2] = color[2]
    return torch.from_numpy(arr)


def test_node_returns_correct_shape_and_calls_swap_crop():
    n = NanoBananaCropSwap()
    fake = SwapResult(image=Image.new("RGB", (64, 64), (0, 200, 0)),
                      status="OK",
                      mask=Image.new("L", (64, 64), 200),
                      debug_sheet=Image.new("RGB", (512, 512), (0, 0, 0)))
    with patch("faceswap.backend.FaceSwapBackend.swap_crop", return_value=fake) as sc:
        out = n.run(api_key="FAKE_KEY", model="gemini-3.1-flash-image-preview",
                    target_image=_img_tensor(), identity_1=_img_tensor(color=(1, 0, 0)),
                    identity_2=None, identity_3=None, identity_4=None,
                    identity_5=None, identity_6=None,
                    scope="head", grid_mode="separate_refs", custom_hint="",
                    safety_threshold="BLOCK_NONE", seed=0, batch_axis="target",
                    detector="auto", crop_size=1024,
                    histogram_match=True, feather_px=24)
        sc.assert_called_once()
    img_t, status, mask_t, debug_t = out
    assert img_t.shape == (1, 64, 64, 3)
    # Status now carries an appended cost suffix; the base status remains "OK".
    assert status.startswith("OK")
    assert mask_t.shape == (1, 64, 64)


def test_input_types_has_pathway_b_specific_fields():
    schema = NanoBananaCropSwap.INPUT_TYPES()
    req = schema["required"]
    for k in ("detector", "crop_size", "histogram_match", "feather_px"):
        assert k in req
