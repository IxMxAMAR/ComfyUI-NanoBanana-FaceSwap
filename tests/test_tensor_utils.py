import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from PIL import Image
from faceswap.tensor_utils import (
    image_tensor_to_pil_list,
    pil_to_image_tensor,
    mask_tensor_to_pil,
    pil_to_mask_tensor,
)


def test_image_tensor_to_pil_roundtrip_single():
    arr = np.full((1, 32, 48, 3), 0.5, dtype=np.float32)
    tensor = torch.from_numpy(arr)
    pils = image_tensor_to_pil_list(tensor)
    assert len(pils) == 1
    assert pils[0].size == (48, 32)
    assert pils[0].mode == "RGB"


def test_image_tensor_to_pil_roundtrip_batch():
    arr = np.zeros((3, 16, 16, 3), dtype=np.float32)
    arr[0] = 1.0
    tensor = torch.from_numpy(arr)
    pils = image_tensor_to_pil_list(tensor)
    assert len(pils) == 3
    assert np.array(pils[0]).max() == 255


def test_pil_to_image_tensor_shape_and_range():
    pil = Image.new("RGB", (10, 20), (128, 128, 128))
    tensor = pil_to_image_tensor(pil)
    assert tensor.shape == (1, 20, 10, 3)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_pil_list_to_image_tensor_batches():
    pils = [Image.new("RGB", (8, 8), (0, 0, 0)),
            Image.new("RGB", (8, 8), (255, 255, 255))]
    tensor = pil_to_image_tensor(pils)
    assert tensor.shape == (2, 8, 8, 3)
    assert tensor[0].max() == 0.0
    assert tensor[1].min() == 1.0


def test_mask_roundtrip():
    pil = Image.new("L", (12, 24), 200)
    tensor = pil_to_mask_tensor(pil)
    assert tensor.shape == (1, 24, 12)
    out = mask_tensor_to_pil(tensor)
    assert out.size == (12, 24)
    assert out.mode == "L"
