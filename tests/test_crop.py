import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from PIL import Image
import numpy as np
from faceswap.crop import expand_bbox, square_around_center, extract_replicate, ExpandRatios


def test_expand_face_uniform_25pct():
    rect = (150, 150, 250, 250)
    out = expand_bbox(rect, "face", img_w=400, img_h=400)
    assert out == (125, 125, 275, 275)


def test_expand_head_asymmetric():
    rect = (100, 100, 200, 200)
    out = expand_bbox(rect, "head", img_w=1000, img_h=1000)
    assert out == (60, 40, 240, 240)


def test_expand_head_styling():
    rect = (100, 100, 200, 200)
    out = expand_bbox(rect, "head+styling", img_w=1000, img_h=1000)
    assert out == (30, 0, 270, 280)


def test_expand_does_not_clamp_to_image_bounds():
    rect = (10, 10, 60, 60)
    out = expand_bbox(rect, "head+styling", img_w=200, img_h=200)
    assert out == (-25, -40, 95, 100)


def test_square_around_center_basic():
    rect = (100, 50, 300, 250)
    sq = square_around_center(rect)
    assert sq == (100, 50, 300, 250)


def test_square_around_center_wider():
    rect = (100, 100, 300, 200)
    sq = square_around_center(rect)
    assert sq == (100, 50, 300, 250)


def test_square_around_center_taller():
    rect = (100, 100, 200, 300)
    sq = square_around_center(rect)
    assert sq == (50, 100, 250, 300)


def test_extract_replicate_returns_square():
    img = Image.new("RGB", (300, 300), (128, 128, 128))
    arr = np.asarray(img).copy()
    arr[80:120, 80:120] = [200, 50, 50]
    img = Image.fromarray(arr)
    sq = (50, 50, 150, 150)
    extracted = extract_replicate(img, sq)
    assert extracted.size == (100, 100)


def test_extract_replicate_handles_negative_coords():
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    sq = (-20, -20, 80, 80)
    extracted = extract_replicate(img, sq)
    assert extracted.size == (100, 100)
    arr = np.asarray(extracted)
    assert tuple(arr[0, 0]) == (100, 150, 200)


def test_extract_replicate_handles_beyond_right_bottom():
    img = Image.new("RGB", (200, 200), (50, 60, 70))
    sq = (150, 150, 250, 250)
    extracted = extract_replicate(img, sq)
    assert extracted.size == (100, 100)


def test_expand_ratios_known():
    assert ExpandRatios.for_scope("face") == (0.25, 0.25, 0.25, 0.25)
    assert ExpandRatios.for_scope("head") == (0.6, 0.4, 0.4, 0.4)
    assert ExpandRatios.for_scope("head+styling") == (1.0, 0.8, 0.7, 0.7)


def test_expand_invalid_scope():
    with pytest.raises(ValueError):
        expand_bbox((0, 0, 10, 10), "bogus", img_w=100, img_h=100)
