import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from faceswap import ref_obfuscate


def _gradient(size=(256, 256)):
    """A non-uniform test image so we can detect actual changes."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = (x % 256, y % 256, (x + y) % 256)
    return Image.fromarray(arr)


def test_obfuscate_strength_zero_returns_input():
    src = _gradient()
    out = ref_obfuscate.obfuscate(src, strength=0.0, seed=42)
    assert np.array_equal(np.asarray(out), np.asarray(src))


def test_obfuscate_strength_changes_pixels():
    src = _gradient()
    out = ref_obfuscate.obfuscate(src, strength=0.5, seed=42)
    diff = np.abs(np.asarray(out).astype(np.int16) - np.asarray(src).astype(np.int16))
    assert diff.mean() > 1.0  # something was changed


def test_obfuscate_deterministic_with_seed():
    src = _gradient()
    a = ref_obfuscate.obfuscate(src, strength=0.5, seed=7)
    b = ref_obfuscate.obfuscate(src, strength=0.5, seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_obfuscate_different_seeds_differ():
    src = _gradient()
    a = ref_obfuscate.obfuscate(src, strength=0.5, seed=1)
    b = ref_obfuscate.obfuscate(src, strength=0.5, seed=2)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_obfuscate_preserves_size():
    src = _gradient((177, 211))
    out = ref_obfuscate.obfuscate(src, strength=0.7, seed=0)
    assert out.size == (177, 211)


def test_blur_only_no_warp_or_color():
    # Use random noise (high-freq content) so blur has something to smooth
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    src = Image.fromarray(arr)
    out = ref_obfuscate.obfuscate(src, strength=1.0, seed=0,
                                  apply_blur=True, apply_warp=False, apply_color=False)
    src_arr = np.asarray(src).astype(np.float32)
    out_arr = np.asarray(out).astype(np.float32)
    assert np.abs(out_arr - src_arr).mean() > 1.0


def test_obfuscate_strength_scales_change():
    src = _gradient()
    low = ref_obfuscate.obfuscate(src, strength=0.2, seed=5)
    high = ref_obfuscate.obfuscate(src, strength=1.0, seed=5)
    low_diff = np.abs(np.asarray(low).astype(np.int16) - np.asarray(src).astype(np.int16)).mean()
    high_diff = np.abs(np.asarray(high).astype(np.int16) - np.asarray(src).astype(np.int16)).mean()
    assert high_diff > low_diff
