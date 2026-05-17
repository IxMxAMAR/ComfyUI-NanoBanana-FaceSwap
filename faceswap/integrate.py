"""Post-process integration stages for face-swap composites.

Five stages, all pure functions, all operate inside an existing edit mask.
None of them move pixels around (no warpAffine, no alignment) - those
operations caused ghosting in earlier pipelines per user testing.

Stages (in compositing order):
1. lab_color_match     - Reinhard LAB shift, face-polygon stats, strength control
2. grain_match         - Additive Gaussian noise matched to original's cheek variance
3. sharpness_match     - Blur Gemini face if sharper than original; never sharpen
4. laplacian_composite - Multi-band frequency blend on the edit mask
5. transfer_lowfreq_lighting - Opt-in same-character lighting transfer

Per Gemini Pro pass-2 consult 2026-05-14.
"""

from __future__ import annotations
from typing import Optional, Tuple
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Minimum crop dimension below which integrate stages are unreliable.
# FaceMesh degrades, stats noisy, Laplacian washes out.
SMALL_FACE_GATE_PX = 128

# Cheek landmark indices for grain-variance sampling (MediaPipe FaceMesh).
# Cheeks are the largest contiguous flat skin area with minimal geometric
# curvature; forehead is often specular, nose bridge too small.
LEFT_CHEEK_LANDMARKS = (116, 117, 118, 119, 100, 126)
RIGHT_CHEEK_LANDMARKS = (345, 346, 347, 348, 329, 355)
MIN_CHEEK_PIXELS = 400  # below this, sigma estimate is unstable


# ---------------------------------------------------------------------------
# Stage 1: LAB color match (Reinhard mean/std transfer, face-polygon stats)
# ---------------------------------------------------------------------------

def lab_color_match(source: Image.Image, target: Image.Image,
                    source_mask: Optional[Image.Image] = None,
                    target_mask: Optional[Image.Image] = None,
                    strength: float = 0.6) -> Image.Image:
    """Reinhard-style LAB color transfer.

    Stats are computed from the MASKED region of each side (face polygon for
    Pathway C, swapped-face polygon for Pathway B). Apply the per-channel
    shift to the whole `source` so the resulting face blends with the target's
    surroundings. `strength` lerps between identity (0.0) and full transfer (1.0).

    Falls back to RGB space if cv2 is unavailable.
    """
    src_arr = np.asarray(source.convert("RGB"), dtype=np.uint8)
    tgt_arr = np.asarray(target.convert("RGB"), dtype=np.uint8)

    try:
        import cv2
        src_lab = cv2.cvtColor(src_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        space = "LAB"
    except ImportError:
        src_lab = src_arr.astype(np.float32)
        tgt_lab = tgt_arr.astype(np.float32)
        space = "RGB"

    # Build per-side masks (default to all-true if not provided).
    if source_mask is not None:
        s_mask = np.asarray(source_mask.convert("L"), dtype=np.uint8) > 127
    else:
        s_mask = np.ones(src_lab.shape[:2], dtype=bool)
    if target_mask is not None:
        t_mask = np.asarray(target_mask.convert("L"), dtype=np.uint8) > 127
    else:
        t_mask = np.ones(tgt_lab.shape[:2], dtype=bool)

    if s_mask.sum() < 100 or t_mask.sum() < 100:
        logger.info("[integrate] color_match: too few masked pixels, returning source unchanged")
        return source

    out = src_lab.copy()
    for c in range(3):
        s_vals = src_lab[..., c][s_mask]
        t_vals = tgt_lab[..., c][t_mask]
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std())
        t_mean, t_std = float(t_vals.mean()), float(t_vals.std())
        if s_std < 1e-3:
            # Zero-variance source: scale term undefined, just shift the mean.
            shifted = src_lab[..., c] + (t_mean - s_mean)
        else:
            # Reinhard: (x - mean_s) * (std_t / std_s) + mean_t
            shifted = (src_lab[..., c] - s_mean) * (t_std / s_std) + t_mean
        # Strength lerp
        out[..., c] = src_lab[..., c] + (shifted - src_lab[..., c]) * strength

    out = np.clip(out, 0, 255).astype(np.uint8)
    if space == "LAB":
        import cv2
        rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    else:
        rgb = out
    return Image.fromarray(rgb, mode="RGB")


# ---------------------------------------------------------------------------
# Stage 2: Synthetic noise grain match (variance-only, never pixel copy)
# ---------------------------------------------------------------------------

def _cheek_bbox_from_landmarks(landmarks, image_size: Tuple[int, int]
                                ) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the union of cheek landmark points, or None."""
    if landmarks is None:
        return None
    w, h = image_size
    pts = []
    for i in LEFT_CHEEK_LANDMARKS + RIGHT_CHEEK_LANDMARKS:
        try:
            lm = landmarks[i]
            pts.append((int(lm.x * w), int(lm.y * h)))
        except (IndexError, AttributeError):
            return None
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(0, min(xs)), max(0, min(ys)),
            min(w, max(xs)), min(h, max(ys)))


def _high_pass_variance(gray_patch: np.ndarray) -> float:
    """Estimate noise variance via Gaussian high-pass residual."""
    try:
        import cv2
        blurred = cv2.GaussianBlur(gray_patch, (5, 5), 0)
    except ImportError:
        # numpy fallback - separable box blur, good enough
        from scipy.ndimage import uniform_filter
        blurred = uniform_filter(gray_patch.astype(np.float32), size=5)
    residual = gray_patch.astype(np.float32) - blurred.astype(np.float32)
    return float(residual.var())


def grain_match(source: Image.Image, target: Image.Image,
                target_landmarks=None,
                strength: float = 1.0,
                rng_seed: Optional[int] = None) -> Image.Image:
    """Add synthetic Gaussian noise to `source` matching the noise variance of
    `target` (sampled from cheek landmarks).

    Pure additive. Never copies pixel values from target - that's the trap
    that painted original shadows onto new faces in earlier pipelines.

    Skips (returns source unchanged) if landmarks unavailable, cheek area
    too small (<400 px), or strength == 0.
    """
    if strength <= 0:
        return source
    if target_landmarks is None:
        logger.debug("[integrate] grain_match: no target landmarks; skipping")
        return source

    tgt_arr = np.asarray(target.convert("RGB"))
    cheek_bbox = _cheek_bbox_from_landmarks(target_landmarks, target.size)
    if cheek_bbox is None:
        return source
    x1, y1, x2, y2 = cheek_bbox
    if (x2 - x1) * (y2 - y1) < MIN_CHEEK_PIXELS:
        logger.debug("[integrate] grain_match: cheek area %d px < %d; skipping",
                     (x2 - x1) * (y2 - y1), MIN_CHEEK_PIXELS)
        return source

    cheek_rgb = tgt_arr[y1:y2, x1:x2]
    # Use the green channel for noise estimation - matches human eye response
    # and is the channel most cameras prioritize for noise reduction.
    cheek_gray = cheek_rgb[..., 1]
    variance = _high_pass_variance(cheek_gray)
    sigma = float(np.sqrt(max(variance, 0.0)))
    if sigma < 0.5:
        logger.debug("[integrate] grain_match: target is essentially noise-free "
                     "(sigma=%.2f); skipping", sigma)
        return source

    sigma *= strength
    rng = np.random.default_rng(rng_seed)
    src_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    # CORRECTNESS: generate monochrome (luminance) noise and broadcast to RGB
    # rather than independent per-channel noise. Real digital sensor noise in
    # well-lit areas is predominantly luminance; per-channel noise looks like
    # "color confetti" and is visibly wrong on skin.
    noise2d = rng.normal(0.0, sigma, size=src_arr.shape[:2]).astype(np.float32)
    noise = np.broadcast_to(noise2d[..., None], src_arr.shape)
    out = src_arr + noise
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def grain_match_from_bbox(source: Image.Image, target: Image.Image,
                          bbox: Tuple[int, int, int, int],
                          strength: float = 1.0,
                          rng_seed: Optional[int] = None) -> Image.Image:
    """Grain match using an explicit bbox into target for variance sampling.

    Used by the painted-edit pathway where FaceMesh landmarks aren't
    available - the caller hands us a known-flat patch (typically the
    largest unmasked region of the target crop).
    """
    if strength <= 0:
        return source
    x1, y1, x2, y2 = bbox
    if (x2 - x1) * (y2 - y1) < MIN_CHEEK_PIXELS:
        logger.debug("[integrate] grain_match_from_bbox: patch area %d px < %d; skipping",
                     (x2 - x1) * (y2 - y1), MIN_CHEEK_PIXELS)
        return source
    tgt_arr = np.asarray(target.convert("RGB"))
    patch_rgb = tgt_arr[y1:y2, x1:x2]
    if patch_rgb.size == 0:
        return source
    patch_gray = patch_rgb[..., 1]  # green channel
    variance = _high_pass_variance(patch_gray)
    sigma = float(np.sqrt(max(variance, 0.0)))
    if sigma < 0.5:
        return source
    sigma *= strength
    rng = np.random.default_rng(rng_seed)
    src_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    # Luminance-only noise (broadcast to RGB) — see grain_match for rationale.
    noise2d = rng.normal(0.0, sigma, size=src_arr.shape[:2]).astype(np.float32)
    noise = np.broadcast_to(noise2d[..., None], src_arr.shape)
    out = src_arr + noise
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Stage 3: Sharpness match (blur source if too sharp; never sharpen)
# ---------------------------------------------------------------------------

def _variance_of_laplacian(gray: np.ndarray) -> float:
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        # numpy 4-neighbor laplacian
        L = (-4 * gray.astype(np.float64)
             + np.roll(gray, 1, axis=0)
             + np.roll(gray, -1, axis=0)
             + np.roll(gray, 1, axis=1)
             + np.roll(gray, -1, axis=1))
        return float(L.var())


def sharpness_match(source: Image.Image, target: Image.Image,
                    threshold_ratio: float = 1.5,
                    target_ratio: float = 1.2,
                    max_sigma: float = 1.5) -> Image.Image:
    """If `source` is significantly sharper than `target`, blur until the ratio
    falls below `target_ratio`. Never sharpens - sharpening produces ringing
    artifacts that are worse than the mismatch.
    """
    try:
        import cv2
        src_gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
        tgt_gray = cv2.cvtColor(np.asarray(target.convert("RGB")), cv2.COLOR_RGB2GRAY)
    except ImportError:
        # PIL grayscale fallback
        src_gray = np.asarray(source.convert("L"))
        tgt_gray = np.asarray(target.convert("L"))

    s_var = _variance_of_laplacian(src_gray)
    t_var = _variance_of_laplacian(tgt_gray)
    if t_var < 1e-3:
        return source  # target is degenerate; no reliable comparison

    ratio = s_var / t_var
    if ratio <= threshold_ratio:
        return source  # source is already as soft as or softer than target

    # Binary search a Gaussian sigma that brings the ratio below target_ratio.
    # Cap at max_sigma so we never crush detail. If the binary search can't
    # converge within the cap (degenerate over-sharp source like alternating
    # pixels), fall back to blurring at max_sigma rather than giving up.
    lo, hi = 0.0, max_sigma
    best_sigma: Optional[float] = None
    for _ in range(8):
        mid = (lo + hi) / 2.0
        blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(mid)).convert("L"))
        new_ratio = _variance_of_laplacian(blurred) / t_var
        if new_ratio <= target_ratio:
            best_sigma = mid
            hi = mid
        else:
            lo = mid
    if best_sigma is None:
        best_sigma = max_sigma  # best-effort blur at the cap
    if best_sigma <= 0.0:
        return source
    return source.filter(ImageFilter.GaussianBlur(best_sigma))


# ---------------------------------------------------------------------------
# Stage 4: Laplacian pyramid composite
# ---------------------------------------------------------------------------

def _laplacian_pyramid_blend(A: np.ndarray, B: np.ndarray, m: np.ndarray,
                              num_levels: int) -> np.ndarray:
    """Multi-band blend. A = source (foreground), B = target (background),
    m = mask in [0,1]. All same shape.
    """
    import cv2
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    m = m.astype(np.float32)
    if np.max(m) > 1.0:
        m = m / 255.0

    gpA, gpB, gpM = [A], [B], [m]
    for _ in range(num_levels):
        A = cv2.pyrDown(A)
        B = cv2.pyrDown(B)
        m = cv2.pyrDown(m)
        gpA.append(A); gpB.append(B); gpM.append(m)

    lpA = [gpA[-1]]
    lpB = [gpB[-1]]
    gpMr = [gpM[-1]]
    for i in range(num_levels, 0, -1):
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
        lpA.append(cv2.subtract(gpA[i - 1], cv2.pyrUp(gpA[i], dstsize=size)))
        lpB.append(cv2.subtract(gpB[i - 1], cv2.pyrUp(gpB[i], dstsize=size)))
        gpMr.append(gpM[i - 1])

    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        if la.ndim == 3 and gm.ndim == 2:
            gm = gm[:, :, np.newaxis]
        LS.append(la * gm + lb * (1.0 - gm))

    ls_ = LS[0]
    for i in range(1, num_levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])
    return np.clip(ls_, 0, 255).astype(np.uint8)


def _dynamic_num_levels(w: int, h: int) -> int:
    """Choose pyramid depth based on the smaller dimension."""
    n = max(3, int(np.log2(min(w, h))) - 3)
    return min(n, 8)


def laplacian_pyramid_composite(target: Image.Image, source: Image.Image,
                                mask: Image.Image,
                                origin: Tuple[int, int] = (0, 0),
                                num_levels: int = 0) -> Image.Image:
    """Composite `source` into `target` at `origin` using multi-band blending
    along `mask`. Same-size mask as source. `num_levels=0` picks dynamically.
    Falls back to feathered alpha if cv2 is unavailable.
    """
    try:
        import cv2
    except ImportError:
        out = target.copy().convert("RGB")
        out.paste(source.convert("RGB"), origin, mask.convert("L"))
        return out

    ox, oy = origin
    sw, sh = source.size
    # Sample the target window where the source will go. CRITICAL: when the
    # origin extends past image bounds (negative or beyond W/H), PIL's crop
    # pads with BLACK pixels by default - those black pixels then bleed into
    # the result through the pyramid blur, producing a visible dark halo at
    # the image edge. Mirror crop.extract_replicate's BORDER_REPLICATE behavior.
    target_arr_full = np.asarray(target.convert("RGB"))
    th, tw = target_arr_full.shape[:2]
    pad_left = max(0, -ox)
    pad_top = max(0, -oy)
    pad_right = max(0, (ox + sw) - tw)
    pad_bottom = max(0, (oy + sh) - th)
    if pad_left or pad_top or pad_right or pad_bottom:
        import cv2 as _cv2
        target_arr_full = _cv2.copyMakeBorder(target_arr_full, pad_top, pad_bottom,
                                              pad_left, pad_right, _cv2.BORDER_REPLICATE)
        x1 = ox + pad_left
        y1 = oy + pad_top
    else:
        x1, y1 = ox, oy
    tgt_arr = target_arr_full[y1:y1 + sh, x1:x1 + sw]
    src_arr = np.asarray(source.convert("RGB"))
    mask_arr = np.asarray(mask.convert("L"))

    n = num_levels if num_levels > 0 else _dynamic_num_levels(sw, sh)
    pad_unit = 2 ** n
    pad_w = (pad_unit - sw % pad_unit) % pad_unit
    pad_h = (pad_unit - sh % pad_unit) % pad_unit
    if pad_w or pad_h:
        src_arr = np.pad(src_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        tgt_arr = np.pad(tgt_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        mask_arr = np.pad(mask_arr, ((0, pad_h), (0, pad_w)), mode="constant")

    blended = _laplacian_pyramid_blend(src_arr, tgt_arr, mask_arr, n)
    if pad_w or pad_h:
        blended = blended[:sh, :sw]

    out = target.copy().convert("RGB")
    out.paste(Image.fromarray(blended, mode="RGB"), (ox, oy))
    return out


# ---------------------------------------------------------------------------
# Stage 5: Low-frequency lighting transfer (same-character mode, opt-in)
# ---------------------------------------------------------------------------

def transfer_lowfreq_lighting(source: Image.Image, target: Image.Image,
                              blur_sigma_pct: float = 0.12) -> Image.Image:
    """Replace source's broad lighting envelope with target's.

    Heavy-blur both faces (sigma ~12% of face width) to isolate the lighting
    gradient. source_high = source - source_low. Result = target_low + source_high.

    Source keeps its identity-defining high-frequency texture; target's broad
    lighting cast and shadow gradient are imposed.

    WARNING: only safe for same-character refinement. For cross-identity swaps
    the source person's high-freq texture will get pasted onto the target
    person's low-freq broad geometry, producing a Frankenstein - the skull
    shape and shadow positions will be the target's, the facial details will
    be the source's. Use only when you're refining the same person.
    """
    if source.size != target.size:
        target = target.resize(source.size, Image.LANCZOS)

    w, _ = source.size
    sigma = max(1.0, blur_sigma_pct * w)

    s_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    t_arr = np.asarray(target.convert("RGB"), dtype=np.float32)

    s_low = np.asarray(source.filter(ImageFilter.GaussianBlur(sigma)).convert("RGB"),
                       dtype=np.float32)
    t_low = np.asarray(target.filter(ImageFilter.GaussianBlur(sigma)).convert("RGB"),
                       dtype=np.float32)

    # Additive frequency split: high = source - source_low. Recombine with target's low.
    s_high = s_arr - s_low
    out = t_low + s_high
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Convenience: run the full stack
# ---------------------------------------------------------------------------

def too_small_for_integrate(image: Image.Image) -> bool:
    """Stages produce unreliable output below this size; caller should
    fall back to a simpler composite path.
    """
    w, h = image.size
    return min(w, h) < SMALL_FACE_GATE_PX
