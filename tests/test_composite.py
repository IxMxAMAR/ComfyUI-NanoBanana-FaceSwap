import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from faceswap.composite import match_histograms, make_feather_mask, alpha_composite_at


def test_match_histograms_shifts_distribution():
    src = Image.new("RGB", (64, 64), (50, 50, 50))
    ref = Image.new("RGB", (64, 64), (200, 200, 200))
    out = match_histograms(src, ref)
    out_arr = np.asarray(out)
    assert out_arr.mean() > 150


def test_match_histograms_preserves_shape():
    src = Image.new("RGB", (37, 53), (100, 100, 100))
    ref = Image.new("RGB", (37, 53), (200, 100, 50))
    out = match_histograms(src, ref)
    assert out.size == (37, 53)


def test_match_histograms_mismatched_size():
    src = Image.new("RGB", (32, 32), (50, 50, 50))
    ref = Image.new("RGB", (64, 64), (200, 200, 200))
    out = match_histograms(src, ref)
    assert out.size == (32, 32)


def test_feather_mask_smooth_edges():
    m = make_feather_mask(size=64, feather_px=8)
    arr = np.asarray(m, dtype=np.float32)
    assert arr.shape == (64, 64)
    assert arr[32, 32] > 240
    assert arr[0, 0] < 40


def test_feather_mask_size_validation():
    m = make_feather_mask(size=128, feather_px=24)
    assert m.size == (128, 128)


def test_alpha_composite_at_negative_origin():
    base = Image.new("RGB", (100, 100), (0, 0, 0))
    top = Image.new("RGB", (50, 50), (255, 255, 255))
    mask = Image.new("L", (50, 50), 255)
    out = alpha_composite_at(base, top, mask, origin=(-20, -20))
    arr = np.asarray(out)
    assert arr[0, 0].max() == 255
    assert arr[40, 40].max() == 0


def test_alpha_composite_at_partial_off_right_bottom():
    base = Image.new("RGB", (100, 100), (50, 50, 50))
    top = Image.new("RGB", (50, 50), (200, 200, 200))
    mask = Image.new("L", (50, 50), 255)
    out = alpha_composite_at(base, top, mask, origin=(80, 80))
    arr = np.asarray(out)
    assert arr[99, 99, 0] > 100
