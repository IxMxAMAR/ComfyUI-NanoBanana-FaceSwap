"""Prompt templates for face-swap operations (Pathways A/B/C - prose).

Mirrored from gemini_makeup/try_face_match.py - the proven-working prompts
that the user validated empirically. Kept simple by design. Custom_hint is
the user's lever for case-specific adjustments (anti-beautify, expression
direction, identity description for private subjects, etc.) - the default
prompt should not pre-bake opinionated language.

Templates avoid vocabulary that elevates Gemini safety classifier sensitivity:
"swap", "replace face", "deepfake", "fake". Preferred vocabulary: "edit",
"redraw", "match identity", "render", "transform".
"""

from __future__ import annotations

SCOPES = ("face", "head", "head+styling")
PATHWAYS = ("whole", "crop")

FORBIDDEN_VOCAB = (
    "swap", "face swap", "faceswap",
    "replace face", "face replacement",
    "deepfake", "deep fake", "fake",
)

_REGION = {
    "face": (
        "face - including the eyes, nose, mouth, jawline, and skin tone - "
        "while preserving the existing hair, hairline, and ears"
    ),
    "head": (
        "head - including face, hair, hairline, and ears - "
        "while preserving the neck and shoulders"
    ),
    "head+styling": (
        "head, including all visible jewelry, earrings, makeup, and personal "
        "styling features around the head and neck"
    ),
}

# Prose template for Pathways A/B/C. Mirrors try_face_match.py's spirit:
# identity transfer, scene preservation, photographic literalism. No
# editorializing about celebrities or spatial preservation - those live
# in custom_hint when the user needs them.
_BASE = (
    "Edit the base image to redraw the subject's {region} so that it matches "
    "the identity and facial features of the person shown in the identity "
    "reference images. Preserve the base image's lighting direction, head "
    "pose, expression, body, clothing, framing, and background exactly. "
    "Only alter the {region} identity. Match the exact film grain, noise "
    "level, sharpness, and lighting conditions of the base image so the "
    "edited region blends naturally.\n\n"
    "The image above is the base image to be edited. The images below are "
    "identity references for the person whose appearance should appear in "
    "the edited result."
)

_CROP_CLARIFIER = (
    "\n\nThe base image is a tight cropped close-up of the {region_short} region; "
    "render the edited result at the same framing without adding shoulders, "
    "background, or scene elements."
)

_REGION_SHORT = {
    "face": "face",
    "head": "head",
    "head+styling": "head and styling",
}


def build(scope: str, custom_hint: str, pathway: str, n_refs: int = 1) -> str:
    """Build the user prompt for Pathways A/B/C.

    `n_refs` is accepted for backward-compat with callers but is currently
    unused - the default template doesn't vary by reference count. Add
    explicit ref-handling guidance in custom_hint if a specific case
    benefits from it.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")
    if pathway not in PATHWAYS:
        raise ValueError(f"unknown pathway {pathway!r}; expected one of {PATHWAYS}")

    parts = [_BASE.format(region=_REGION[scope])]
    if pathway == "crop":
        parts.append(_CROP_CLARIFIER.format(region_short=_REGION_SHORT[scope]))
    hint = (custom_hint or "").strip()
    if hint:
        parts.append("\n\n" + hint)
    return "".join(parts)


def ref_instruction_for_count(n: int) -> str:
    """Kept for backward-compat with older test/node imports.

    Returns an empty string - the active prompts no longer vary by ref count.
    Callers that imported this function will get a harmless no-op.
    """
    return ""
