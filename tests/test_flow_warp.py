"""Tests for the torch flow-warp (backward sampling, vanilla remap semantics)."""
import torch

from cadence_core import flow_warp


def _img():
    # 1x5x5x3 ramp so horizontal shifts are visible
    x = torch.arange(5, dtype=torch.float32).repeat(5, 1)  # cols vary 0..4
    return (x / 4.0)[None, :, :, None].repeat(1, 1, 1, 3)


def test_zero_flow_is_identity():
    img = _img()
    flow = torch.zeros(1, 5, 5, 2)
    out = flow_warp(img, flow)
    assert torch.allclose(out, img, atol=1e-5)


def test_constant_shift_moves_content_left():
    # flow_warp samples img at (x - f); f=+1 in x => out(x)=img(x-1) => content shifts RIGHT by 1
    img = _img()
    flow = torch.zeros(1, 5, 5, 2)
    flow[..., 0] = 1.0
    out = flow_warp(img, flow)
    # interior column 2 should now equal original column 1
    assert torch.allclose(out[0, 2, 2], img[0, 2, 1], atol=1e-5)


def test_preserves_shape_and_dtype():
    img = _img()
    out = flow_warp(img, torch.zeros(1, 5, 5, 2))
    assert out.shape == img.shape and out.dtype == img.dtype
