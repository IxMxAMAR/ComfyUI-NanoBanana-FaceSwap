import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from nodes.identity_sheet import IdentitySheetComposer


def _img(h, w, c=(0.5, 0.5, 0.5)):
    arr = np.zeros((1, h, w, 3), dtype=np.float32)
    arr[..., 0] = c[0]; arr[..., 1] = c[1]; arr[..., 2] = c[2]
    return torch.from_numpy(arr)


def test_compose_two_refs():
    n = IdentitySheetComposer()
    img_t, info = n.run(layout="auto", tile_size=128,
                        identity_1=_img(64, 64, (1, 0, 0)),
                        identity_2=_img(64, 64, (0, 1, 0)))
    assert img_t.shape == (1, 128, 256, 3)
    assert "2x1" in info


def test_compose_one_ref_returns_tile():
    n = IdentitySheetComposer()
    img_t, _ = n.run(layout="auto", tile_size=64,
                     identity_1=_img(32, 32, (1, 1, 1)))
    assert img_t.shape == (1, 64, 64, 3)
