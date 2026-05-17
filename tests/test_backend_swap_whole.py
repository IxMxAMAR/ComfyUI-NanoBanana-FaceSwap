import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import pytest
from faceswap.backend import FaceSwapBackend, SwapResult


def _fake_response_with_image(pil: Image.Image):
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    part = MagicMock()
    part.inline_data.data = buf.getvalue()
    part.inline_data.mime_type = "image/png"
    cand = MagicMock()
    cand.content.parts = [part]
    cand.finish_reason = None
    resp = MagicMock()
    resp.candidates = [cand]
    resp.prompt_feedback = None
    return resp


def _fake_response_refused(category="HARM_CATEGORY_DANGEROUS_CONTENT"):
    resp = MagicMock()
    resp.candidates = []
    resp.prompt_feedback.block_reason = category
    return resp


@pytest.fixture
def backend():
    return FaceSwapBackend(api_key="FAKE_KEY")


def test_swap_whole_ok(backend):
    target = Image.new("RGB", (64, 64), (100, 100, 100))
    refs = [Image.new("RGB", (64, 64), (200, 50, 50))]
    swapped = Image.new("RGB", (64, 64), (50, 200, 50))
    with patch.object(backend, "_call_generate", return_value=_fake_response_with_image(swapped)):
        res = backend.swap_whole(target=target, refs=refs, scope="face",
                                 custom_hint="", model="x", seed=0,
                                 grid_mode="separate_refs", safety_threshold="BLOCK_NONE")
    assert isinstance(res, SwapResult)
    assert res.status == "OK"
    assert res.image.size == (64, 64)
    arr = np.asarray(res.image)
    assert arr[..., 1].mean() > arr[..., 0].mean()


def test_swap_whole_refused_soft_fail(backend):
    target = Image.new("RGB", (64, 64), (100, 100, 100))
    refs = [Image.new("RGB", (64, 64), (200, 50, 50))]
    with patch.object(backend, "_call_generate",
                      return_value=_fake_response_refused("HARM_CATEGORY_DANGEROUS_CONTENT")):
        res = backend.swap_whole(target=target, refs=refs, scope="face",
                                 custom_hint="", model="x", seed=0,
                                 grid_mode="separate_refs", safety_threshold="BLOCK_NONE")
    assert res.status.startswith("REFUSED:")
    assert "DANGEROUS" in res.status
    arr = np.asarray(res.image)
    assert arr[..., 0].mean() > arr[..., 1].mean() + 20


def test_swap_whole_requires_at_least_one_ref(backend):
    target = Image.new("RGB", (64, 64), (100, 100, 100))
    with pytest.raises(ValueError, match="at least one"):
        backend.swap_whole(target=target, refs=[], scope="face",
                           custom_hint="", model="x", seed=0,
                           grid_mode="separate_refs", safety_threshold="BLOCK_NONE")


def test_diff_mask_computed_when_ok(backend):
    target = Image.new("RGB", (64, 64), (100, 100, 100))
    refs = [Image.new("RGB", (64, 64), (200, 50, 50))]
    swapped = Image.new("RGB", (64, 64), (200, 50, 50))
    with patch.object(backend, "_call_generate", return_value=_fake_response_with_image(swapped)):
        res = backend.swap_whole(target=target, refs=refs, scope="face",
                                 custom_hint="", model="x", seed=0,
                                 grid_mode="separate_refs", safety_threshold="BLOCK_NONE")
    mask_arr = np.asarray(res.mask)
    assert mask_arr.mean() > 50
