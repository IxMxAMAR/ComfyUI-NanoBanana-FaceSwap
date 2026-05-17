import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
from PIL import Image
from faceswap.detect import BBox, detect_head_bbox, _normalize_box


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "face.jpg")


def test_bbox_normalize_clamps_and_returns_pixels():
    box = _normalize_box([-0.1, -0.1, 1.2, 1.2], img_w=100, img_h=200)
    assert box.x1 == 0 and box.y1 == 0
    assert box.x2 == 100 and box.y2 == 200


def test_cascade_returns_first_detector_hit():
    img = Image.open(FIXTURE)
    fake_box = BBox(10, 20, 90, 180, source="mediapipe", confidence=0.9)
    with patch("faceswap.detect._detect_mediapipe", return_value=fake_box):
        with patch("faceswap.detect._detect_yunet") as yn:
            with patch("faceswap.detect._detect_gemini_bbox") as gem:
                out = detect_head_bbox(img, detector="auto", api_key=None)
                assert out == fake_box
                yn.assert_not_called()
                gem.assert_not_called()


def test_cascade_falls_through_to_yunet_then_gemini():
    img = Image.open(FIXTURE)
    fake_yunet = BBox(5, 5, 50, 50, source="opencv_yunet", confidence=0.8)
    with patch("faceswap.detect._detect_mediapipe", return_value=None):
        with patch("faceswap.detect._detect_yunet", return_value=fake_yunet):
            with patch("faceswap.detect._detect_gemini_bbox") as gem:
                out = detect_head_bbox(img, detector="auto", api_key="x")
                assert out is fake_yunet
                gem.assert_not_called()


def test_cascade_uses_gemini_when_others_miss():
    img = Image.open(FIXTURE)
    fake_gemini = BBox(0, 0, 200, 200, source="gemini_bbox", confidence=1.0)
    with patch("faceswap.detect._detect_mediapipe", return_value=None):
        with patch("faceswap.detect._detect_yunet", return_value=None):
            with patch("faceswap.detect._detect_gemini_bbox", return_value=fake_gemini):
                out = detect_head_bbox(img, detector="auto", api_key="x")
                assert out.source == "gemini_bbox"


def test_explicit_detector_unavailable_returns_none():
    img = Image.open(FIXTURE)
    with patch("faceswap.detect._detect_mediapipe", side_effect=ImportError("no mp")):
        out = detect_head_bbox(img, detector="mediapipe", api_key=None)
        assert out is None
