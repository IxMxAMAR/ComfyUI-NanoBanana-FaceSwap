import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from PIL import Image
from faceswap.sheet import compose, _resolve_layout


def test_compose_two_images_returns_2x1():
    refs = [Image.new("RGB", (256, 256), (255, 0, 0)),
            Image.new("RGB", (256, 256), (0, 255, 0))]
    out = compose(refs, tile_size=128, layout="auto")
    assert out.size == (256, 128)


def test_compose_four_images_2x2():
    refs = [Image.new("RGB", (256, 256), (c, c, c)) for c in (50, 100, 150, 200)]
    out = compose(refs, tile_size=128, layout="auto")
    assert out.size == (256, 256)


def test_compose_six_images_3x2():
    refs = [Image.new("RGB", (128, 128), (50, 50, 50)) for _ in range(6)]
    out = compose(refs, tile_size=128, layout="auto")
    assert out.size == (384, 256)


def test_compose_empty_raises():
    with pytest.raises(ValueError):
        compose([], tile_size=128)


def test_compose_pads_non_square():
    refs = [Image.new("RGB", (100, 200), (255, 255, 255)),
            Image.new("RGB", (300, 50), (0, 0, 0))]
    out = compose(refs, tile_size=64, layout="2x1")
    assert out.size == (128, 64)


def test_resolve_layout_auto():
    assert _resolve_layout("auto", 1) == (1, 1)
    assert _resolve_layout("auto", 2) == (2, 1)
    assert _resolve_layout("auto", 3) == (3, 1)
    assert _resolve_layout("auto", 4) == (2, 2)
    assert _resolve_layout("auto", 5) == (3, 2)
    assert _resolve_layout("auto", 6) == (3, 2)
