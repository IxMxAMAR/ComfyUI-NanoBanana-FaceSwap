import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from faceswap.prompts import build, SCOPES, FORBIDDEN_VOCAB, ref_instruction_for_count


@pytest.mark.parametrize("scope", SCOPES)
def test_build_returns_string_for_each_scope(scope):
    p = build(scope=scope, custom_hint="", pathway="whole")
    assert isinstance(p, str) and len(p) > 50


def test_no_forbidden_vocab_in_any_default_template():
    for scope in SCOPES:
        for pathway in ("whole", "crop"):
            p = build(scope=scope, custom_hint="", pathway=pathway).lower()
            for word in FORBIDDEN_VOCAB:
                assert word.lower() not in p, f"{word!r} appeared in {scope}/{pathway}"


def test_custom_hint_is_appended():
    p = build(scope="face", custom_hint="HELLO_HINT_123", pathway="whole")
    assert "HELLO_HINT_123" in p


def test_crop_pathway_includes_crop_clarifier():
    p = build(scope="head", custom_hint="", pathway="crop")
    assert "cropped close-up" in p.lower() or "tight crop" in p.lower()
    assert "without adding" in p.lower()


def test_invalid_scope_raises():
    with pytest.raises(ValueError):
        build(scope="bogus", custom_hint="", pathway="whole")


def test_invalid_pathway_raises():
    with pytest.raises(ValueError):
        build(scope="face", custom_hint="", pathway="bogus")


def test_region_phrase_present_in_template():
    p = build(scope="face", custom_hint="", pathway="whole")
    assert "eyes, nose, mouth" in p
    p = build(scope="head", custom_hint="", pathway="whole")
    assert "hair" in p and "ears" in p
    p = build(scope="head+styling", custom_hint="", pathway="whole")
    assert "jewelry" in p.lower() or "earrings" in p.lower()


def test_ref_instruction_backcompat_returns_string():
    # Helper kept for backward-compat; current default returns empty string.
    s = ref_instruction_for_count(1)
    assert isinstance(s, str)


def test_n_refs_accepted_but_ignored_by_default():
    # build() accepts n_refs for backward-compat; result must not vary by it
    # in the current simple template.
    a = build(scope="face", custom_hint="", pathway="whole", n_refs=1)
    b = build(scope="face", custom_hint="", pathway="whole", n_refs=5)
    assert a == b
