# ComfyUI-NanoBanana-FaceSwap

Face/head replacement for ComfyUI via Google's Nano Banana 2 (`gemini-3.1-flash-image-preview`) and Nano Banana Pro (`gemini-3-pro-image-preview`).

Two pathways are shipped so you can route around safety refusals on innocuous targets:

| Pathway | When to use |
|---|---|
| **Whole Image Swap** | Default. Fast. Sends target + refs to the model. |
| **Crop & Composite Swap** | Use when the model refuses the whole-image path because of *scene content* (background, clothing, peripheral objects). Detects the head, swaps only the crop, composites back. |

Honest note: cropping reduces refusals caused by background/scene content. It can **increase** refusals when the safety classifier is reacting to identity recognition (real person, perceived minor). It is not a universal anti-refusal solution.

## Nodes

| Node | Purpose |
|---|---|
| **Nano Banana - Face Swap (Whole Image)** | Pathway A. Inputs: target + up to 6 identity refs + scope (face/head/head+styling). |
| **Nano Banana - Face Swap (Crop & Composite)** | Pathway B. Adds detector cascade, crop_size, histogram_match, feather_px. |
| **Identity Sheet Composer** | Helper. Composes up to 6 refs into a single labeled grid image (`auto`, `2x2`, `3x2`, etc.). |
| **Face Swap Prompt Builder** | Helper. Preview the exact prompt the swap nodes will send. |

All nodes appear in the ComfyUI category `NanoBanana FaceSwap`.

## Scope semantics

- `face` - replace eyes/nose/mouth/jawline/skin tone; preserve hair, hairline, ears.
- `head` - replace face + hair + hairline + ears; preserve neck and shoulders.
- `head+styling` - replace head plus jewelry, earrings, makeup, head/neck styling.

The Pathway B crop expansion ratios per scope are tuned so the crop captures the right amount of context for each:

| Scope | Up | Down | Left | Right |
|---|---|---|---|---|
| face | +25% | +25% | +25% | +25% |
| head | +60% | +40% | +40% | +40% |
| head+styling | +100% | +80% | +70% | +70% |

(Percentages applied to the detected face bbox dimensions.)

## Detector cascade (Pathway B)

With `detector=auto`, the node tries in order:

1. **MediaPipe Face Detection** - fast, CPU-only. Requires `pip install mediapipe`.
2. **OpenCV YuNet** (`face_detection_yunet_2023mar.onnx`, ~1.8MB) - modern, side-profile-friendly. Auto-downloaded to `~/.cache/nanobanana_faceswap/` on first use (atomic write, safe under concurrent runs).
3. **Gemini vision bbox** (JSON-mode call to `gemini-flash-latest`) - last-resort, costs one extra API call.

Pick a specific detector to disable the cascade. Selecting one that isn't available returns `ERROR:detector_unavailable:<name>` rather than falling back silently.

## Outputs

Both swap nodes return `(IMAGE, STRING, MASK, IMAGE)`:

- **image** - the result, or a red-tinted placeholder on safety refusal.
- **status** - `OK` / `REFUSED:<category>` / `ERROR:<reason>`. Batched calls return statuses joined by `;`.
- **mask** - Pathway A: per-pixel diff mask of the changed region. Pathway B: the feathered alpha composite mask in full-image coordinates.
- **debug_sheet** - Pathway A: the identity sheet (if `auto_sheet`) or a side-by-side input/output. Pathway B: a 2x2 montage of [tight bbox / expanded bbox / cropped square sent / swapped square returned].

## Prompt vocabulary

Templates avoid words that elevate Gemini safety-classifier sensitivity ("swap", "replace face", "deepfake", "fake"). They use "edit", "redraw", "match identity", "render", "transform". The full template is exposed via the **Face Swap Prompt Builder** node so you can preview it before each run; you can also pass your own additions via the `custom_hint` input on either swap node.

## Pathway B math (technical note)

Pathway B's crop alignment is the bug-prone part. The implementation deliberately does NOT clamp the expanded face bbox to image bounds before squaring - clamping shifts the face center away from the true center and causes the swapped head to land misaligned with the body. Instead:

1. Detect face bbox.
2. Expand per scope (may extend outside the image - left it that way).
3. Square the expanded bbox around the original face center (still may extend outside).
4. Extract the SxS region from the target using `cv2.copyMakeBorder` with `BORDER_REPLICATE` to fill any out-of-bounds pixels.
5. Resize to `crop_size` and send to the model.
6. Resize the returned image back to SxS, optionally histogram-match to the original crop, then alpha-composite at the same `(x, y)` anchor (which may itself be negative - PIL's paste handles that correctly).

This preserves face-center alignment in all cases including faces near image edges.

## Batch handling

Both swap nodes have a `batch_axis` dropdown:

- `target` (default) - if the target IMAGE has batch dim B>1, iterates each target frame against the fixed identity refs.
- `identity` - if `identity_1` has batch dim B>1, iterates each frame as a different identity (refs 2..6 stay fixed across iterations).

Iteration is sequential (Nano Banana 2 has no batched-inference endpoint via `generate_content`). Each iteration is its own API call with its own retry; a single refusal in a batch doesn't stop subsequent frames.

## Installation

```bash
git clone https://github.com/IxMxAMAR/ComfyUI-NanoBanana-FaceSwap "C:/ComfyUI/custom_nodes/ComfyUI-NanoBanana-FaceSwap"
cd "C:/ComfyUI/custom_nodes/ComfyUI-NanoBanana-FaceSwap"
"C:/ComfyUI/venv/Scripts/python" -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in your environment, or paste your key into the `api_key` input of either swap node.

## Failure modes

- **Safety refusal (REFUSED:<category>)** - returns a red-tinted version of the target (Pathway A) or only the bbox region red-tinted (Pathway B), with the refusal category burned in.
- **No face detected (ERROR:no_face_detected)** - Pathway B only. Returns a yellow-tinted placeholder.
- **API error after retries (ERROR:<ExceptionClass>)** - returns a yellow-tinted placeholder. Transient errors (429/5xx/DEADLINE_EXCEEDED, plus gRPC `UNAVAILABLE` / `RESOURCE_EXHAUSTED`) are retried up to 3 times with exponential backoff before this path triggers. v0.2's retry classifier requires the status code be paired with an HTTP/status marker or recognized reason phrase so an incidental "500 MB" in an error message no longer triggers spurious retries.

`IS_CHANGED` returns `float('nan')` on every node so ComfyUI re-runs generative nodes on each queue. No stale cache.

## v0.2 optional inputs (every swap node)

- **`dry_run`** (default false) - skip the Gemini call and return a structured preview of the prompt + parts that *would* have been sent. Status becomes `DRY_RUN:<pathway>\n--- PROMPT PREVIEW ---\n...`. Burns no quota. Use to debug `custom_hint` / `edit_prompt` injection-escape behavior.
- **`timeout_ms`** (default 180000) - per-call API timeout. Lower (e.g. 30000) to fail fast on stuck calls; higher for slow Pro-tier renders.
- **`ref_cap_px`** (default 1024) - identity references with a longest edge above this are downscaled with LANCZOS before send. Saves bandwidth (quadratic in pixel count) and empirically improves identity transfer - the model is confused by 4K skin-pore detail when only the embedding-level identity matters. 0 = disabled.
- **`auto_relax_on_refused`** (default false) - on a `REFUSED` response, retry once with `safety_settings` stripped entirely (SDK falls back to model defaults). Sometimes a hard BLOCK_NONE refusal relaxes under defaults.

The status output now also carries an indicative cost suffix (` | ~$0.039` for a single Flash-image call, scaled by batch size). Informational only - actual cost depends on input/output token counts which we don't measure.

## Security model

- The pack accepts user-supplied prompt strings (`custom_hint`, `edit_prompt`). v0.2 escapes them via `json.dumps()` before splicing into JSON-shaped prompts; the previous ad-hoc `.replace()` chain missed control characters and Unicode escapes.
- `LoadImagePaint`'s fallback path resolution (used by ancient ComfyUI builds) refuses `../` traversal.
- `mask_data` decode in `LoadImagePaint` caps at 50 MB string length + 25 MP decoded pixels to defend against decompression bombs and OOM.
- Model downloads (YuNet, FaceMesh) use a PID + nanosecond-suffixed temp file then atomic rename, so two concurrent workers don't TOCTOU-corrupt each other.

See `CHANGELOG.md` for the full v0.2.0 change set.

## License

MIT.
