# Template workflows

Load any of these in ComfyUI (drag the JSON file onto the canvas, or use **Load** menu). Replace the placeholder image filenames with your own. Each workflow has a single coloured group titled with the pathway name and brief usage notes in `extra.info`.

## When to use which workflow

| # | File | Pathway | Use when |
|---|---|---|---|
| 01 | `01_face_swap_whole_image.json` | A — Whole image | Default starting point. Lets Gemini see the full target and refine it. Best for same-character refinement style work (Face Detailer mode). |
| 02 | `02_face_swap_crop_composite.json` | B — Crop + composite | Target has a face you want swapped to a different identity, integration stages on. |
| 03 | `03_face_swap_mask_inpaint.json` | C — Mask inpaint (red obscure) | Whole-image refuses due to scene content (clothing, background). Paints the face red before sending so Gemini sees no original face. |
| 04 | `04_face_swap_unbiased_celeb.json` | D — Unbiased | Celebrity / public-figure refs. Uses `try_face_match.py` request shape (system_instruction + JSON prompt + no safety_settings). **Requires US/permissive-region VPN if your billing/IP is in EU.** |
| 05 | `05_celeb_with_obfuscation.json` | D + Obfuscator | Even with VPN, Gemini still refuses the specific celeb. Routes each reference through `IdentityRefObfuscator` (blur + perspective warp + LAB shift) to drop the recognition classifier's confidence below threshold. |
| 06 | `06_tattoo_additive_edit.json` | E — Painted edit, `edit_mode=additive` | Tattoo / logo / sticker / object insertion. Paint the region directly in the **LoadImagePaint** canvas, supply a text prompt. `edit_mode=additive` force-disables LAB color match + sharpness match + Laplacian (which all desaturate / blur additive content). |
| 07 | `07_painted_edit_safety_obscure.json` | E — Painted edit + `obscure_outside_mask=blur` | Edit near revealing content (cleavage, etc.) that trips `IMAGE_SAFETY`. Combines tight crop (`crop_tightness_pct=0`) with heavy blur on the non-painted area of the crop so the safety classifier has no anatomical shapes to fire on. |
| 08 | `08_identity_sheet_multi_angle.json` | D + Identity Sheet | You have 4+ reference angles of the same person. Composes them into one labeled grid that Gemini can read holistically; often gives better identity lock than separate refs. |

`faceswap_reference.json` is the older two-pathway example (Pathway A + B side by side); kept for reference.

## Per-workflow notes

### 01 — Whole Image Swap
Simplest possible workflow. Set the model to Nano Banana Pro (`gemini-3-pro-image-preview`) if the Flash output is too smoothed. `image_size=2K` is the default; bump to `4K` if you want max resolution.

### 02 — Crop + Composite
The integrate stack defaults are `color_match=True, grain_match=True, sharpness_match=True, composite_method=laplacian, lab_strength=0.6, grain_strength=1.0`. For cross-identity work (different person on a different body), leave `match_directional_lighting=False`. For same-character refinement (refining a face you've already roughly placed) flip it to True.

### 03 — Mask Inpaint (Pathway C)
Critical knob: `mask_dilate_px`. Default 12. If you see "oval boundary" sticker effects, lower it (e.g., to 6). If you see jaw double-edges, raise it (e.g., to 25 — same as v1 default). `mask_feather_px` is the soft edge of the composite; 16–32 is typical.

### 04 — Unbiased (celeb path)
**Pre-flight:** connect a VPN exit in a permissive region (US works well) before queuing. EU IPs (especially Germany) reliably block celebrity refs regardless of request shape. `apply_integrate=False` for the first test (proves the call succeeds); flip to True afterwards for full LAB+grain+sharpness post-processing.

### 05 — Celeb with Obfuscation
For specifically-hard-blocked identities that refuse even with VPN. The obfuscator does:
- Slight Gaussian blur (strips high-freq detail used by recognition embeddings)
- Mild perspective warp (~8% corner displacement at strength 1.0)
- LAB color shift (drops skin-tone hash matches)

Start with `strength=0.5`. Bump to 0.7–0.85 if still blocked. Above 0.85 identity starts to degrade for Gemini's pattern matching.

### 06 — Tattoo (additive edit)
Paint directly on the **LoadImagePaint** canvas. Left mouse paint, right mouse erase, brush size widget. `edit_mode=additive` is what makes tattoos come out crisp and contrasty. Default text prompt asks for a "small detailed black-ink rose tattoo" — replace with whatever you want.

For text-style tattoos (script names, dates, etc.): include exact text in the prompt with `"in [font style]"` — e.g., `"Zalim ki Begum in cursive script, dark grey ink"`.

### 07 — Safety-Obscured Painted Edit
For NSFW-adjacent photos (chest/cleavage/etc.) where the model refuses with `IMAGE_SAFETY`. Combines:
- `crop_tightness_pct=0` — crop is exactly the painted bbox (no surrounding context)
- `obscure_outside_mask=blur` — heavy Gaussian on everything in the crop outside the painted region
- `obscure_blur_px=60` — pretty heavy blur; bump higher if still blocked

The model sees: a small crop with a small painted region surrounded by indistinct blurry skin/fabric. Safety classifier has nothing to fire on.

### 08 — Identity Sheet
Compose 2–6 reference angles into a single labeled grid via `IdentitySheetComposer`. Plug that single sheet into `identity_1` on the Unbiased node. Often produces better identity adherence than supplying refs separately because the model sees all angles at once.

## Common knobs across all swap workflows

| Knob | Default | When to change |
|---|---|---|
| `model` | `gemini-3.1-flash-image-preview` | Switch to `gemini-3-pro-image-preview` (Nano Banana Pro) for higher quality at higher cost; sometimes different safety behavior. |
| `image_size` | `2K` | `4K` for max detail at higher cost; `1K` for fast iteration. |
| `seed` | 0 (random) | Set non-zero to reproduce a specific result. |
| `safety_threshold` (A/B/C) | `BLOCK_NONE` | Server-side safety enforcement may ignore this; doesn't bypass identity-recognition blocks. |
| `crop_tightness_pct` (E) | 100 | 0–20 for tight safety-evading crops; 100–200 for edits that need scene context. |
| `composite_feather_px` | 24 | Smaller (~8) for crisp boundaries (tattoos, logos); larger (~40) for smooth blends (face skin). |

## Region routing reminder

If a workflow refuses with `REFUSED:BlockedReason.OTHER` on a real-person reference: Gemini's safety policies are partitioned by request IP. EU egress (especially Germany) applies the strictest celebrity-recognition policy. Route your face-swap traffic through a permissive-region VPN exit (US works). Billing country of the GCP project does not appear to override request IP for this endpoint.

## API key

All workflow files have an empty `api_key` field. You can:
1. Paste your key directly into the `api_key` field on each swap node, or
2. Set the `GEMINI_API_KEY` environment variable (preferred — survives workflow JSON edits), or
3. Drop the key into `c:/ComfyUI/RD/FashionGUI Premium/settings.json` under `nanobanana_key` (the global tool fallback path).
