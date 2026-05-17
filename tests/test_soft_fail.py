import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from faceswap.soft_fail import render_refused, render_error


def test_refused_full_image_red_tint():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    out = render_refused(img, reason="HARM_CATEGORY_DANGEROUS_CONTENT")
    arr = np.asarray(out)
    assert arr[..., 0].mean() > arr[..., 1].mean()
    assert arr[..., 0].mean() > arr[..., 2].mean()
    assert out.size == (256, 256)


def test_refused_region_only_red_tint():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    bbox = (64, 64, 192, 192)
    out = render_refused(img, reason="X", region_bbox=bbox)
    arr = np.asarray(out)
    outside_r = arr[10, 10, 0]
    outside_g = arr[10, 10, 1]
    assert abs(int(outside_r) - int(outside_g)) < 5
    inside_r = arr[128, 128, 0]
    inside_g = arr[128, 128, 1]
    assert int(inside_r) > int(inside_g)


def test_error_yellow_tint():
    img = Image.new("RGB", (128, 128), (100, 100, 100))
    out = render_error(img, reason="no_face_detected")
    arr = np.asarray(out)
    assert arr[..., 0].mean() > arr[..., 2].mean()
    assert arr[..., 1].mean() > arr[..., 2].mean()
