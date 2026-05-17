"""Head bounding-box detection via mediapipe -> opencv YuNet -> Gemini cascade."""

from __future__ import annotations
import os
import io
import json
import urllib.request
import logging
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nanobanana_faceswap")


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str
    confidence: float

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)


def _normalize_box(coords_4: List[float], img_w: int, img_h: int) -> BBox:
    ymin, xmin, ymax, xmax = coords_4
    x1 = max(0, int(round(xmin * img_w)))
    y1 = max(0, int(round(ymin * img_h)))
    x2 = min(img_w, int(round(xmax * img_w)))
    y2 = min(img_h, int(round(ymax * img_h)))
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2, source="normalized", confidence=1.0)


def detect_head_bbox(img: Image.Image, detector: str = "auto",
                     api_key: Optional[str] = None) -> Optional[BBox]:
    img = img.convert("RGB")
    if detector == "auto":
        cascade = ["mediapipe", "opencv_yunet", "gemini_bbox"]
    else:
        cascade = [detector]

    for name in cascade:
        try:
            if name == "mediapipe":
                box = _detect_mediapipe(img)
            elif name == "opencv_yunet":
                box = _detect_yunet(img)
            elif name == "gemini_bbox":
                if not api_key:
                    logger.warning("[faceswap] gemini_bbox detector requires api_key; skipping")
                    box = None
                else:
                    box = _detect_gemini_bbox(img, api_key=api_key)
            else:
                logger.warning("[faceswap] unknown detector %r", name)
                return None
        except ImportError as e:
            if detector == "auto":
                logger.info("[faceswap] %s unavailable (%s); cascading", name, e)
                continue
            return None
        except Exception as e:
            logger.warning("[faceswap] %s raised %s; %s", name,
                           type(e).__name__,
                           "cascading" if detector == "auto" else "returning None")
            if detector == "auto":
                continue
            return None
        if box is not None:
            return box
    return None


def _detect_mediapipe(img: Image.Image) -> Optional[BBox]:
    """MediaPipe face detection. Tries the legacy solutions API first;
    derives a bbox from FaceLandmarker (tasks API) as a fallback when
    `solutions` was removed in newer mediapipe builds. If neither works,
    raise ImportError so the cascade falls through to OpenCV YuNet.
    """
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    # Legacy API
    try:
        from mediapipe.solutions import face_detection as mp_fd  # type: ignore
        with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
            results = fd.process(arr)
            if not results.detections:
                return None
            det = max(results.detections, key=lambda d: d.score[0] if d.score else 0)
            rb = det.location_data.relative_bounding_box
            x1 = int(max(0, rb.xmin) * w)
            y1 = int(max(0, rb.ymin) * h)
            x2 = int(min(1, rb.xmin + rb.width) * w)
            y2 = int(min(1, rb.ymin + rb.height) * h)
            if x2 <= x1 or y2 <= y1:
                return None
            score = float(det.score[0]) if det.score else 0.0
            return BBox(x1, y1, x2, y2, source="mediapipe", confidence=score)
    except ImportError:
        pass

    # Tasks API fallback: derive a bbox from FaceLandmarker landmarks.
    # Slightly heavier than dedicated face detection, but it works on the
    # same model file we already need for the integrate stages.
    from . import mask as _mask
    landmarks = _mask._facemesh_landmarks(img)
    if landmarks is None or len(landmarks) == 0:
        return None
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(max(0.0, min(xs)) * w)
    y1 = int(max(0.0, min(ys)) * h)
    x2 = int(min(1.0, max(xs)) * w)
    y2 = int(min(1.0, max(ys)) * h)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2, source="mediapipe_landmarker", confidence=1.0)


def _ensure_yunet_model() -> str:
    """Download the YuNet ONNX model into the cache dir. Uses a process+PID-
    unique temp file then atomic rename so two ComfyUI workers in the same
    pack can't TOCTOU-corrupt the model.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    final = os.path.join(CACHE_DIR, YUNET_FILENAME)
    if os.path.exists(final):
        # Defensive: catch obviously-truncated downloads (real model is ~232KB).
        # If anything below 50KB sneaks in we re-download.
        try:
            if os.path.getsize(final) > 50_000:
                return final
        except OSError:
            pass
    # PID + nanosecond-resolution timestamp guards against concurrent download.
    tmp = final + f".tmp.{os.getpid()}.{int(__import__('time').time_ns())}"
    logger.info("[faceswap] downloading YuNet model to %s", final)
    urllib.request.urlretrieve(YUNET_URL, tmp)
    try:
        os.replace(tmp, final)
    except OSError:
        # Another worker just won the race; tmp can be discarded.
        try: os.remove(tmp)
        except OSError: pass
    return final


# Cache the YuNet detector so we don't pay the ONNX-load tax on every call.
# `setInputSize` is cheap and lets us reuse one instance across varying sizes.
_YUNET_DETECTOR = None
_YUNET_LOCK = None  # late-init to avoid importing threading at module level


def _get_yunet_detector(model_path: str, w: int, h: int):
    """Cached YuNet detector. Loading the ONNX from disk on every detect call
    causes severe I/O thrashing on long video batches."""
    global _YUNET_DETECTOR, _YUNET_LOCK
    import cv2
    if _YUNET_LOCK is None:
        import threading
        _YUNET_LOCK = threading.Lock()
    with _YUNET_LOCK:
        if _YUNET_DETECTOR is None:
            _YUNET_DETECTOR = cv2.FaceDetectorYN_create(
                model=model_path, config="",
                input_size=(w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=10,
            )
        _YUNET_DETECTOR.setInputSize((w, h))
        return _YUNET_DETECTOR


def _detect_yunet(img: Image.Image) -> Optional[BBox]:
    import cv2
    model_path = _ensure_yunet_model()
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    det = _get_yunet_detector(model_path, w, h)
    retval, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    faces = faces.tolist()
    best = max(faces, key=lambda f: f[-1])
    x, y, fw, fh = best[0], best[1], best[2], best[3]
    x1 = max(0, int(x)); y1 = max(0, int(y))
    x2 = min(w, int(x + fw)); y2 = min(h, int(y + fh))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2, source="opencv_yunet", confidence=float(best[-1]))


def _detect_gemini_bbox(img: Image.Image, api_key: str,
                        model: str = "gemini-flash-latest") -> Optional[BBox]:
    from google import genai
    from google.genai import types
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)

    client = genai.Client(api_key=api_key, http_options={"timeout": 10_000})
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[float],
        temperature=0.0,
    )
    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=(
            "Return the bounding box of the largest visible human head in the image "
            "as [ymin, xmin, ymax, xmax] with normalized coordinates 0.0-1.0."
        )),
    ]
    contents = [types.Content(role="user", parts=parts)]
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    raw = resp.text if hasattr(resp, "text") and resp.text else None
    if not raw:
        for cand in (getattr(resp, "candidates", None) or []):
            for part in (cand.content.parts or []):
                if getattr(part, "text", None):
                    raw = part.text
                    break
            if raw:
                break
    if not raw:
        return None
    # Gemini frequently wraps JSON in ```json ... ``` fences even with
    # response_mime_type=application/json. Strip them defensively so the
    # detector doesn't silently fall through on every call.
    raw = raw.strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        # Drop a leading 'json' word if present after the fence
        if raw.lower().startswith("json\n") or raw.lower().startswith("json "):
            raw = raw[4:].lstrip()
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[faceswap] gemini_bbox returned unparseable JSON: %r", raw[:120])
        return None
    if not (isinstance(arr, list) and len(arr) == 4 and all(isinstance(x, (int, float)) for x in arr)):
        return None
    if not all(0.0 <= float(x) <= 1.0 for x in arr):
        if all(0 <= float(x) <= 1000 for x in arr):
            arr = [float(x) / 1000.0 for x in arr]
        else:
            return None
    box = _normalize_box(list(map(float, arr)), img_w=img.width, img_h=img.height)
    return BBox(box.x1, box.y1, box.x2, box.y2, source="gemini_bbox", confidence=1.0)
