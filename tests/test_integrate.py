import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from faceswap import integrate


def _solid(size, color):
    if isinstance(color, int):
        color = (color, color, color)
    return Image.new("RGB", size, color)


# ---------- LAB color match ----------

def test_lab_color_match_shifts_dark_source_toward_bright_target():
    src = _solid((64, 64), (50, 50, 50))
    tgt = _solid((64, 64), (200, 200, 200))
    out = integrate.lab_color_match(src, tgt, strength=1.0)
    out_mean = np.asarray(out).mean()
    assert out_mean > 100  # shifted clearly toward target


def test_lab_color_match_strength_zero_returns_source():
    src = _solid((64, 64), (50, 50, 50))
    tgt = _solid((64, 64), (200, 200, 200))
    out = integrate.lab_color_match(src, tgt, strength=0.0)
    src_arr = np.asarray(src).astype(np.int16)
    out_arr = np.asarray(out).astype(np.int16)
    assert np.abs(src_arr - out_arr).mean() < 2.0


def test_lab_color_match_respects_mask():
    # Source: bg dark, "face" patch bright. Target: bg bright, "face" patch dark.
    src_arr = np.full((64, 64, 3), 30, dtype=np.uint8)
    src_arr[20:44, 20:44] = 220
    tgt_arr = np.full((64, 64, 3), 220, dtype=np.uint8)
    tgt_arr[20:44, 20:44] = 30
    src = Image.fromarray(src_arr)
    tgt = Image.fromarray(tgt_arr)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    msk = Image.fromarray(mask, mode="L")
    # With matching masks, source's bright patch (mean ~220) should shift toward
    # target's masked region (mean ~30) -> output's patch should darken.
    out = integrate.lab_color_match(src, tgt, source_mask=msk, target_mask=msk, strength=1.0)
    out_patch_mean = np.asarray(out)[20:44, 20:44].mean()
    src_patch_mean = src_arr[20:44, 20:44].mean()
    assert out_patch_mean < src_patch_mean - 20


# ---------- Grain match ----------

class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x; self.y = y


def _fake_landmarks(image_w, image_h, face_bbox):
    """Build a sparse mediapipe-like landmark list with cheek points inside bbox."""
    fx1, fy1, fx2, fy2 = face_bbox
    fw = fx2 - fx1; fh = fy2 - fy1
    # Sparse list - we only need the cheek indices to resolve correctly.
    # Use a dict keyed by index, accessed via __getitem__.
    class _LM:
        def __init__(self):
            self._d = {}
        def __getitem__(self, i):
            if i not in self._d:
                # Default to a point in the bbox center
                self._d[i] = _FakeLandmark((fx1 + fw/2) / image_w, (fy1 + fh/2) / image_h)
            return self._d[i]
    lm = _LM()
    # Spread cheek landmarks across the bbox so the union has a meaningful area
    for i, idx in enumerate(integrate.LEFT_CHEEK_LANDMARKS):
        # left cheek = left half of bbox
        px = fx1 + fw * 0.2 + (i % 3) * fw * 0.1
        py = fy1 + fh * 0.4 + (i // 3) * fh * 0.2
        lm._d[idx] = _FakeLandmark(px / image_w, py / image_h)
    for i, idx in enumerate(integrate.RIGHT_CHEEK_LANDMARKS):
        px = fx1 + fw * 0.6 + (i % 3) * fw * 0.1
        py = fy1 + fh * 0.4 + (i // 3) * fh * 0.2
        lm._d[idx] = _FakeLandmark(px / image_w, py / image_h)
    return lm


def test_grain_match_adds_noise_when_target_has_noise():
    # Source: clean. Target: clean + Gaussian noise.
    src = _solid((128, 128), (128, 128, 128))
    rng = np.random.default_rng(0)
    noisy = np.clip(128 + rng.normal(0, 8.0, (128, 128, 3)), 0, 255).astype(np.uint8)
    tgt = Image.fromarray(noisy)
    lm = _fake_landmarks(128, 128, (10, 10, 118, 118))

    out = integrate.grain_match(src, tgt, target_landmarks=lm,
                                strength=1.0, rng_seed=42)
    out_var = np.asarray(out, dtype=np.float32).var()
    src_var = np.asarray(src, dtype=np.float32).var()
    assert out_var > src_var + 10


def test_grain_match_skips_without_landmarks():
    src = _solid((128, 128), (128, 128, 128))
    tgt = _solid((128, 128), (128, 128, 128))
    out = integrate.grain_match(src, tgt, target_landmarks=None)
    assert np.array_equal(np.asarray(out), np.asarray(src))


def test_grain_match_strength_zero_returns_source():
    src = _solid((128, 128), (128, 128, 128))
    tgt = _solid((128, 128), (128, 128, 128))
    out = integrate.grain_match(src, tgt, target_landmarks=None, strength=0.0)
    assert np.array_equal(np.asarray(out), np.asarray(src))


# ---------- Sharpness match ----------

def test_sharpness_match_blurs_sharper_source():
    # Sharp source: random noise (realistic high-frequency content).
    rng = np.random.default_rng(0)
    src_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    src = Image.fromarray(src_arr)
    # Soft target: blurred version.
    from PIL import ImageFilter
    tgt = src.filter(ImageFilter.GaussianBlur(2.0))
    out = integrate.sharpness_match(src, tgt)
    src_vol = integrate._variance_of_laplacian(np.asarray(src.convert("L")))
    out_vol = integrate._variance_of_laplacian(np.asarray(out.convert("L")))
    assert out_vol < src_vol


def test_sharpness_match_does_not_sharpen_softer_source():
    """If source is softer than target, return source unchanged (never sharpen)."""
    from PIL import ImageFilter
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[::2, ::2] = 255
    sharp = Image.fromarray(base)
    soft = sharp.filter(ImageFilter.GaussianBlur(2.0))
    out = integrate.sharpness_match(source=soft, target=sharp)
    assert np.array_equal(np.asarray(out), np.asarray(soft))


# ---------- Laplacian composite ----------

def test_laplacian_composite_handles_arbitrary_dims():
    # Non-power-of-two dims - composite must pad internally and crop back.
    tgt = _solid((137, 91), (50, 50, 50))
    src = _solid((137, 91), (200, 200, 200))
    mask_arr = np.zeros((91, 137), dtype=np.uint8)
    mask_arr[20:70, 30:100] = 255
    mask = Image.fromarray(mask_arr, mode="L")
    out = integrate.laplacian_pyramid_composite(tgt, src, mask, origin=(0, 0))
    assert out.size == (137, 91)
    out_arr = np.asarray(out)
    # Inside mask area should be brighter than the target's 50
    assert out_arr[45, 65].mean() > 100
    # Outside mask: should still be near target's 50
    assert out_arr[5, 5].mean() < 80


def test_laplacian_composite_zero_levels_picks_dynamic():
    tgt = _solid((256, 256), (100, 100, 100))
    src = _solid((256, 256), (200, 200, 200))
    mask = _solid((256, 256), 255).convert("L")
    # Should not crash and should produce a result close to source under full mask
    out = integrate.laplacian_pyramid_composite(tgt, src, mask, num_levels=0)
    out_mean = np.asarray(out).mean()
    assert out_mean > 150  # mostly source through full mask


def test_dynamic_num_levels_scales_with_size():
    assert integrate._dynamic_num_levels(256, 256) >= 5
    assert integrate._dynamic_num_levels(1024, 1024) >= 7
    assert integrate._dynamic_num_levels(64, 64) >= 3


# ---------- Lighting transfer ----------

def test_lighting_transfer_preserves_source_highfreq():
    """Source's high-frequency texture should dominate output's high-frequency band."""
    from PIL import ImageFilter
    # Source: high-frequency stripes
    src_arr = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8).reshape(1, 4),
                       (128, 32))
    src_arr = np.stack([src_arr]*3, axis=-1)
    src = Image.fromarray(src_arr)
    # Target: solid color (no high-freq content)
    tgt = _solid((128, 128), (128, 128, 128))
    out = integrate.transfer_lowfreq_lighting(src, tgt)
    # Output high-pass should still have stripe content
    out_gray = np.asarray(out.convert("L"), dtype=np.float32)
    blurred = np.asarray(out.filter(ImageFilter.GaussianBlur(10)).convert("L"),
                          dtype=np.float32)
    high = out_gray - blurred
    assert high.var() > 100


def test_lighting_transfer_imposes_target_lowfreq():
    """Output's low-frequency content should resemble target's, not source's."""
    src = _solid((128, 128), (50, 50, 50))   # dark broad
    tgt = _solid((128, 128), (220, 220, 220))  # bright broad
    out = integrate.transfer_lowfreq_lighting(src, tgt)
    out_mean = np.asarray(out).mean()
    src_mean = np.asarray(src).mean()
    tgt_mean = np.asarray(tgt).mean()
    # Output should be much closer to target's mean than source's
    assert abs(out_mean - tgt_mean) < abs(out_mean - src_mean)


# ---------- Small-face gate ----------

def test_too_small_for_integrate():
    assert integrate.too_small_for_integrate(_solid((64, 64), (0, 0, 0)))
    assert not integrate.too_small_for_integrate(_solid((256, 256), (0, 0, 0)))
