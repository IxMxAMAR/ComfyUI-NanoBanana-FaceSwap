import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nodes.prompt_builder import FaceSwapPromptBuilder


def test_returns_string_prompt():
    n = FaceSwapPromptBuilder()
    (out,) = n.run(scope="head", pathway="whole", custom_hint="", n_refs=1)
    assert isinstance(out, str)
    assert "hair" in out


def test_custom_hint_appended():
    n = FaceSwapPromptBuilder()
    (out,) = n.run(scope="face", pathway="crop", custom_hint="MY_EXTRA", n_refs=2)
    assert "MY_EXTRA" in out
