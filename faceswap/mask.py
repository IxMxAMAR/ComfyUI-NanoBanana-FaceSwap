"""Face polygon mask generation + obscuration for Pathway C (mask-inpaint).

Lifted from C:/ComfyUI/RD/gemini_makeup/try_face_match.py (the proven pipeline
that survives celebrity-recognition refusals). Uses mediapipe FaceMesh to get a
precise face-skin polygon, fills internal holes via convex hull, dilates
outward, and paints the masked region solid red so Gemini sees no original
facial features to recognize.
"""

from __future__ import annotations
from typing import Optional, Tuple
import logging
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# FaceMesh face-oval edge pairs (from mediapipe public schema). Pinned here
# because the modern FaceLandmarker API doesn't always re-export this constant
# but the 468-landmark topology is stable.
_FACE_OVAL_EDGES = frozenset({
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
})


def _ordered_face_oval_indices() -> list[int]:
    """Walk the FACEMESH_FACE_OVAL edge set into a closed polygon order."""
    pairs = list(_FACE_OVAL_EDGES)
    adj: dict[int, list[int]] = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = min(adj.keys())
    ordered = [start]
    prev = None
    while True:
        nxt = next((n for n in adj[ordered[-1]] if n != prev), None)
        if nxt is None or nxt == start:
            break
        ordered.append(nxt)
        prev = ordered[-2]
    return ordered


_FACE_OVAL_ORDER: list[int] | None = None


def _face_oval_order() -> list[int]:
    global _FACE_OVAL_ORDER
    if _FACE_OVAL_ORDER is None:
        _FACE_OVAL_ORDER = _ordered_face_oval_indices()
    return _FACE_OVAL_ORDER


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_FILENAME = "face_landmarker.task"
_FACE_LANDMARKER_LOCAL_FALLBACKS = (
    # Reuse the existing copy from the gemini_makeup project if present
    r"C:/ComfyUI/RD/gemini_makeup/face_landmarker.task",
    r"C:/ComfyUI/models/mediapipe/face_landmarker.task",
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nanobanana_faceswap")


def _ensure_face_landmarker_model() -> Optional[str]:
    """Return a filesystem path to face_landmarker.task. Tries known local
    locations first; falls back to downloading to the cache dir."""
    for path in _FACE_LANDMARKER_LOCAL_FALLBACKS:
        if os.path.isfile(path):
            return path
    os.makedirs(_CACHE_DIR, exist_ok=True)
    final = os.path.join(_CACHE_DIR, FACE_LANDMARKER_FILENAME)
    if os.path.isfile(final):
        # Catch obvious truncations (real model is ~3.5MB).
        try:
            if os.path.getsize(final) > 500_000:
                return final
        except OSError:
            pass
    # PID + ns timestamp suffix prevents concurrent-worker TOCTOU corruption.
    import time as _t
    tmp = final + f".tmp.{os.getpid()}.{int(_t.time_ns())}"
    try:
        import urllib.request
        logger.info("[faceswap] downloading face_landmarker.task to %s", final)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
        try:
            os.replace(tmp, final)
        except OSError:
            try: os.remove(tmp)
            except OSError: pass
        return final
    except Exception as e:
        logger.warning("[faceswap] face_landmarker.task download failed: %s", e)
        try: os.remove(tmp)
        except OSError: pass
        return None


class _LandmarkAdapter:
    """Mimic the legacy `landmarks[i].x/.y/.z` interface that the rest of the
    pack relies on. Wraps the new FaceLandmarker output."""
    __slots__ = ("_lm",)
    def __init__(self, lm):
        self._lm = lm
    def __getitem__(self, i):
        return self._lm[i]
    def __len__(self):
        return len(self._lm)
    def __iter__(self):
        return iter(self._lm)


def _facemesh_landmarks(image: Image.Image):
    """Return per-face landmarks for the first face in `image`, or None.

    Tries the legacy `mediapipe.solutions.face_mesh` API first (still works
    on older mediapipe installs). Falls back to the modern
    `mediapipe.tasks.vision.FaceLandmarker` API (mediapipe 0.10.x patch
    versions removed `solutions` entirely).
    """
    arr = np.asarray(image.convert("RGB"))
    # Legacy API path
    try:
        from mediapipe.solutions import face_mesh as mp_fm  # type: ignore
        with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.4) as fm:
            results = fm.process(arr)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark
    except ImportError:
        pass

    # Modern Tasks API path
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None
    model_path = _ensure_face_landmarker_model()
    if model_path is None:
        return None
    try:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        # Tasks API requires mp.Image wrapper
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        try:
            result = landmarker.detect(mp_image)
        finally:
            landmarker.close()
        if not result.face_landmarks:
            return None
        return _LandmarkAdapter(result.face_landmarks[0])
    except Exception as e:
        logger.warning("[faceswap] FaceLandmarker (tasks API) failed: %s", e)
        return None


def face_mask_from_landmarks(image: Image.Image) -> Optional[Image.Image]:
    """L-mode mask where 255 = face skin region (FaceMesh face-oval polygon)."""
    landmarks = _facemesh_landmarks(image)
    if landmarks is None:
        return None
    w, h = image.size
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in _face_oval_order()]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return mask


def face_mask_ellipse_fallback(image_size: Tuple[int, int],
                               face_bbox: Tuple[int, int, int, int],
                               aspect: float = 1.15,
                               pad: float = 1.05) -> Image.Image:
    """Ellipse mask aligned to the detected face bbox. Used when FaceMesh fails."""
    w, h = image_size
    x1, y1, x2, y2 = face_bbox
    fw = x2 - x1
    fh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ex = (fw * pad) / 2.0
    ey = (fh * pad * aspect) / 2.0
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse(
        [cx - ex, cy - ey, cx + ex, cy + ey], fill=255,
    )
    return mask


def fill_mask_holes(mask: Image.Image) -> Image.Image:
    """Convex-hull fill so the polygon is a single solid silhouette with no
    interior gaps (mouth, nostrils). Without this, the original mouth shows
    through and Gemini produces double-jaw ghosting.
    """
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    pts = cv2.findNonZero(arr)
    if pts is None:
        return mask
    hull = cv2.convexHull(pts)
    filled = np.zeros_like(arr)
    cv2.fillConvexPoly(filled, hull, 255)
    return Image.fromarray(filled, mode="L")


def dilate(mask: Image.Image, px: int) -> Image.Image:
    """Dilate the mask outward by `px` pixels (elliptical kernel)."""
    if px <= 0:
        return mask
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated, mode="L")


def feather(mask: Image.Image, px: int) -> Image.Image:
    """Gaussian-blur the mask edges for a smooth alpha falloff."""
    if px <= 0:
        return mask
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=max(0.5, px / 2.0))
    return Image.fromarray(blurred, mode="L")


def paint_red(image: Image.Image, mask: Image.Image,
              color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Paint `image` with solid `color` everywhere `mask` is non-zero. Returns
    an RGB image. The default red (#FF0000) is what survives moderation best
    per the FashionGUI face-match results.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    m = m[..., None]
    fill = np.zeros_like(base)
    fill[..., 0] = color[0]
    fill[..., 1] = color[1]
    fill[..., 2] = color[2]
    out = base * (1.0 - m) + fill * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
