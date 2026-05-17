# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-17

First public release. Major audit + fix release driven by two parallel
full-codebase Gemini Pro reviews (Pathway A: face-detection / mask /
composite math; Pathway B: API client / node integration / soft-fail /
security). Verified every Critical and High finding against the actual
source before applying any change.

### Security — fix immediately if you ever load shared workflows

- **Prompt-injection-safe JSON escaping (`backend.py`).** The Unbiased
  pathway (`swap_unbiased`) and Painted-Edit pathway (`swap_painted_edit`)
  used ad-hoc `.replace()` chains to splice user-supplied `custom_hint` /
  `edit_prompt` text into a JSON-shaped prompt. That chain missed control
  characters (`\t`, `\b`, `\r`, vertical tab, BEL) and Unicode escapes,
  producing invalid JSON (Gemini returns 400 Bad Request) and — more
  importantly — leaving the door open for a malicious workflow author to
  break out of the string and inject extra JSON keys into the prompt
  schema. v0.2 uses `json.dumps()` for full JSON-spec-correct escaping.
- **Path-traversal block in `load_image_paint.py`.** The fallback path
  used by ancient ComfyUI builds (pre-`get_annotated_filepath`)
  `os.path.join(input_dir, image_name)`'d the user-supplied filename
  directly. A malicious workflow could ship `"../../etc/passwd"`. The
  fallback now `os.path.abspath`s and verifies the resolved path is
  inside the input dir, raising `ValueError` otherwise. Case-insensitive
  compare for Windows (`C:\ComfyUI\input` vs `c:\comfyui\input`).
- **Base64 mask-decode DoS cap.** `LoadImagePaint.load()` accepted the
  full `mask_data` string with no length check before `base64.b64decode`
  + `Image.open`. A 100 MB attacker-supplied mask would OOM the worker.
  v0.2 enforces a 50 MB string-length cap and a 25 MP decoded-pixel cap
  (PIL decompression-bomb defense). On overflow, falls back to an empty
  mask and logs the rejection.
- **TOCTOU race on YuNet + FaceLandmarker downloads.** Both
  `_ensure_yunet_model` and `_ensure_face_landmarker_model` checked
  `os.path.exists(final)` and then wrote to a static `.tmp` file. Two
  concurrent workers could corrupt each other's download. v0.2 writes
  to a `tmp.<pid>.<ns>` path so each worker has its own staging file,
  then atomic-renames. Loser of the race cleans up its tmp.
- **Truncated-model detection.** Existing cache files smaller than a
  threshold (50 KB for YuNet, 500 KB for FaceLandmarker) are now treated
  as corrupt and re-downloaded. Catches the previous failure mode where
  an interrupted urlretrieve left a 0-byte final file and every
  subsequent run died inside `cv2.FaceDetectorYN_create`.
- **Removed hardcoded developer settings path.** `resolve_api_key` no
  longer attempts to read `c:/ComfyUI/RD/FashionGUI Premium/settings.json`
  — that was the original author's local config. Public users get a
  clean "Gemini API key required" error if their key isn't in the env
  or in `<pack_root>/settings.json`. The portable `settings.json` field
  list expanded to accept `nanobanana_key`, `gemini_api_key`, or
  `api_key`.

### Fixed — Correctness

- **Gemini bbox JSON markdown-fence stripping.** `_detect_gemini_bbox`
  used `json.loads(raw)` directly. Gemini routinely wraps responses in
  ```` ```json ... ``` ```` fences even with
  `response_mime_type="application/json"` set. Every fenced response
  silently failed parsing and the detector returned `None` → the cascade
  fell through to the next detector or gave up. v0.2 strips fences
  defensively before parsing.
- **Grain match: monochrome luminance noise, not color confetti.**
  `grain_match` and `grain_match_from_bbox` generated independent
  Gaussian noise per RGB channel — visible as chromatic specks on skin.
  Real digital sensor noise is predominantly luminance. v0.2 generates a
  single 2-D noise plane and broadcasts to RGB. Verified by test that
  R-G and G-B per-pixel differences are zero (within rounding) after
  applying grain to a solid-grey source.
- **`unpad_after_gemini` perf (and correctness on extreme aspects).**
  The old implementation `LANCZOS`-resized the *entire* Gemini output
  to the padded size before cropping. For a 1024×1024 Gemini return and
  a 4000×4000 padded input that's ~16× extra LANCZOS work just to throw
  away the borders. v0.2 computes the relative crop on the Gemini output
  directly, crops first, then resizes the smaller patch.
- **`_is_transient` regex word boundaries.** The retry classifier
  previously checked `"500" in str(e)`, which matched legitimate
  non-transient errors like `"Image size must be under 500 MB"` or
  `"5000 tokens used"` and triggered unnecessary retries. v0.2 only
  flags 4xx/5xx codes when paired with an HTTP/status/code marker
  (leading) or a recognized reason phrase (trailing: "Too Many
  Requests", "Internal Server Error", "Bad Gateway", "Service
  Unavailable", "Gateway Timeout"). gRPC-status names
  (`DEADLINE_EXCEEDED`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`) still
  match as before.
- **`ref_obfuscate` perspective-warp magnitude capped.** Was `0.08 *
  min(w,h) * strength` — at strength=1.0 on a 1024px image, corners
  shifted by ~80 px, completely destroying biometric identity before
  the celebrity classifier even saw it. v0.2 caps at `0.03 *` (3%),
  enough to perturb the face-embedding vector while keeping the face
  recognizable as the same person.

### Fixed — Performance

- **YuNet detector now cached.** `cv2.FaceDetectorYN_create` was called
  on every single `_detect_yunet()` invocation, loading the ONNX from
  disk into memory each time. On a 30-frame video batch that's 30 model
  loads. v0.2 caches one detector instance globally (thread-locked
  init), updates `setInputSize` per call. `setInputSize` is a cheap
  metadata update on the already-loaded graph.

### Fixed — UX

- **Mixed-size batch warning in `pil_to_image_tensor`.** When sizes
  differ inside a batch, output is zero-padded so `torch.stack`
  succeeds. v0.2 logs a `logger.warning` listing the mixed sizes so
  users can tell *why* their downstream VAE-encoded batch has black
  bars on the edges of some frames.

### Added — New features

- **Dry-run mode (`dry_run` optional input on every swap node).** When
  enabled, skips the Gemini API call entirely and returns a structured
  preview of the prompt + parts that *would* have been sent. Status
  becomes `DRY_RUN:<pathway>\n--- PROMPT PREVIEW ---\n…`. Excellent for
  debugging custom_hint / edit_prompt injection-escape behavior without
  burning quota.
- **Per-call timeout override (`timeout_ms` optional input).** Default
  180 000 ms (3 min). Lower (e.g. 30 000) for "fail fast on stuck
  calls" UX; higher for slow Pro-tier renders. Cached client keying
  now includes the timeout so changing it doesn't silently reuse the
  prior client.
- **Reference-image size cap (`ref_cap_px` optional input).** Defaults
  to 1024 px (longest edge). Downscales identity references with
  LANCZOS before sending. Saves bandwidth (quadratic in pixel count)
  and empirically improves identity transfer — per the Gemini Pro
  review, the model gets confused by 4K skin-pore detail when only the
  embedding-level identity matters. 0 = disabled.
- **Auto-relax-on-refused (`auto_relax_on_refused` optional input).**
  Off by default. When enabled, a `REFUSED:` response triggers a second
  attempt with `safety_settings` stripped entirely (SDK falls back to
  model defaults). Sometimes a hard BLOCK_NONE refusal relaxes under
  defaults.
- **Cost-estimate suffix on status.** Status strings now end with
  ` | ~$0.039` (or equivalent) based on the model selected and number
  of API calls in the batch. Informational only — actual cost depends
  on input/output token counts which we don't measure.
- **Shared batch-iteration helper (`faceswap.helpers.build_batch_iter`).**
  All five swap nodes used to duplicate the same 15-line `batch_axis`
  switch. Extracted to one helper; nodes are 14 lines shorter each.
  Reusable for future swap nodes.
- **Shared cost-estimation helper (`faceswap.helpers.format_cost_suffix`).**

### Tests

- **Python: 116 / 116 passing** (was 93). 23 new tests covering:
  - JSON-escape correctness on `custom_hint` with tabs / quotes / BEL
  - JSON-escape correctness on painted-edit `edit_prompt`
  - `unpad_after_gemini` returns exact original dims for in-bounds and
    downsized Gemini returns
  - `grain_match` produces monochrome (zero chroma differences) noise
  - `_is_transient` accepts real status patterns AND rejects "500 MB"
    style false positives
  - `helpers.build_batch_iter` correctness on both axes + bad axis
  - `helpers.cap_reference_size` downscale / no-op / disabled paths
  - `helpers.format_cost_suffix` non-empty for known model
  - Backend `dry_run=True` skips the API call and returns a preview
  - Backend `ref_cap_px` downsizes large refs before send
  - `ref_obfuscate` warp magnitude capped functionally
  - Gemini bbox parser strips markdown fences AND handles plain arrays
  - `LoadImagePaint._get_image_path` blocks `../` traversal AND allows
    legit filenames
  - `LoadImagePaint.load` rejects an oversized mask_data string without
    OOMing

### Documentation

- CHANGELOG.md (this file) added.
- README updated to reflect new optional inputs + features.

### Notes / non-changes

- Crop-alignment math, Laplacian pyramid composite, Reinhard LAB
  transfer, FaceMesh face-oval traversal, BBox normalization all
  reviewed and **confirmed correct** by Gemini Pro. No changes.
- `IS_CHANGED` returning `float("nan")` is the modern correct way to
  force ComfyUI to re-run a generative node — confirmed by Gemini Pro,
  the `time.time()` string rumor is outdated.
- `nodes/` directory name retained (vs MegaPack's rename to `mp_nodes/`).
  Collision with ComfyUI's core `nodes.py` is mitigated by relative
  imports in `__init__.py` and absolute imports rooted in the pack via
  the existing `_PACK_ROOT` insert.
