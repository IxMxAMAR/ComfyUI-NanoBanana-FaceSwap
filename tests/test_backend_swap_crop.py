import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import pytest
from faceswap.backend import FaceSwapBackend, SwapResult
from faceswap.detect import BBox


def _fake_response_with_image(pil):
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    part = MagicMock(); part.inline_data.data = buf.getvalue(); part.inline_data.mime_type = "image/png"
    cand = MagicMock(); cand.content.parts = [part]; cand.finish_reason = None
    resp = MagicMock(); resp.candidates = [cand]; resp.prompt_feedback = None
    return resp


@pytest.fixture
def backend():
    return FaceSwapBackend(api_key="FAKE")


def test_swap_crop_ok_path(backend):
    target = Image.new("RGB", (512, 512), (128, 128, 128))
    refs = [Image.new("RGB", (256, 256), (255, 0, 0))]
    swapped = Image.new("RGB", (1024, 1024), (50, 200, 50))

    fake_box = BBox(180, 180, 320, 320, source="mediapipe", confidence=0.9)
    with patch("faceswap.detect.detect_head_bbox", return_value=fake_box):
        with patch.object(backend, "_call_generate", return_value=_fake_response_with_image(swapped)):
            res = backend.swap_crop(target=target, refs=refs, scope="head",
                                    custom_hint="", model="x", seed=0,
                                    grid_mode="separate_refs", safety_threshold="BLOCK_NONE",
                                    detector="auto", crop_size=1024,
                                    histogram_match=True, feather_px=24)
    assert isinstance(res, SwapResult)
    assert res.status == "OK"
    assert res.image.size == target.size
    mask_arr = np.asarray(res.mask)
    assert mask_arr.shape == (512, 512)
    assert mask_arr[250, 250] > 100


def test_swap_crop_no_face_returns_error(backend):
    target = Image.new("RGB", (256, 256), (50, 50, 50))
    refs = [Image.new("RGB", (64, 64), (255, 255, 255))]
    with patch("faceswap.detect.detect_head_bbox", return_value=None):
        res = backend.swap_crop(target=target, refs=refs, scope="head",
                                custom_hint="", model="x", seed=0,
                                grid_mode="separate_refs", safety_threshold="BLOCK_NONE",
                                detector="auto", crop_size=1024,
                                histogram_match=True, feather_px=24)
    assert res.status.startswith("ERROR:no_face")


def test_swap_crop_refused_returns_region_tinted_placeholder(backend):
    target = Image.new("RGB", (512, 512), (128, 128, 128))
    refs = [Image.new("RGB", (64, 64), (255, 0, 0))]
    fake_box = BBox(180, 180, 320, 320, source="mediapipe", confidence=0.9)
    refused = MagicMock()
    refused.candidates = []
    refused.prompt_feedback.block_reason = "HARM_CATEGORY_HARASSMENT"
    with patch("faceswap.detect.detect_head_bbox", return_value=fake_box):
        with patch.object(backend, "_call_generate", return_value=refused):
            res = backend.swap_crop(target=target, refs=refs, scope="head",
                                    custom_hint="", model="x", seed=0,
                                    grid_mode="separate_refs", safety_threshold="BLOCK_NONE",
                                    detector="auto", crop_size=1024,
                                    histogram_match=True, feather_px=24)
    assert res.status.startswith("REFUSED:")
    arr = np.asarray(res.image)
    outside = arr[10, 10]
    assert abs(int(outside[0]) - int(outside[1])) < 10


def test_swap_crop_explicit_detector_with_no_face(backend):
    target = Image.new("RGB", (256, 256), (50, 50, 50))
    refs = [Image.new("RGB", (64, 64), (255, 255, 255))]
    with patch("faceswap.detect.detect_head_bbox", return_value=None):
        res = backend.swap_crop(target=target, refs=refs, scope="face",
                                custom_hint="", model="x", seed=0,
                                grid_mode="separate_refs", safety_threshold="BLOCK_NONE",
                                detector="mediapipe", crop_size=1024,
                                histogram_match=False, feather_px=12)
    assert res.status.startswith("ERROR:")
