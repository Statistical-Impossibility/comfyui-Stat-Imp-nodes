"""Tests for ComputeFlow node + estimate_flow dispatch (IMAGE<->core conversion, FLOW out)."""
import cv2
import numpy as np
import torch

from flow_nodes import ComputeFlow, estimate_flow


def _img_pair(dx=4):
    rng = np.random.default_rng(2)
    base = (rng.random((48, 48)).astype(np.float32) * 255)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0) / 255.0
    base = base[None, :, :, None].repeat(3, axis=-1)         # ComfyUI IMAGE BHWC RGB 0..1
    a = torch.from_numpy(base.copy())
    b = torch.from_numpy(np.roll(base, dx, axis=2).copy())   # shift right in x
    return a, b


def test_none_returns_zero_flow_matching_size():
    a, b = _img_pair()
    flow = estimate_flow(a, b, "None", "cpu")
    assert flow.shape == (1, 48, 48, 2)
    assert torch.count_nonzero(flow) == 0


def test_dis_medium_returns_flow_tensor_with_motion():
    a, b = _img_pair(4)
    flow = estimate_flow(a, b, "DIS Medium", "cpu")
    assert flow.shape == (1, 48, 48, 2)
    assert flow[0, 16:32, 16:32, 0].mean().item() > 1.5


def test_node_run_returns_flow_tuple():
    a, b = _img_pair(4)
    (flow,) = ComputeFlow().run(a, b, "None", "AUTO")
    assert isinstance(flow, torch.Tensor) and flow.shape[-1] == 2


def test_node_return_type_is_flow():
    assert ComputeFlow.RETURN_TYPES == ("FLOW",)


def test_method_choices_match_vanilla():
    choices = ComputeFlow.INPUT_TYPES()["required"]["method"][0]
    assert choices == ["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"]


def test_mismatched_sizes_do_not_crash():
    # bootstrap/first-span frames can feed a smaller old-anchor than the corrected image;
    # estimate_flow must resize image_from -> image_to and return flow sized to image_to.
    a = torch.rand(1, 24, 24, 3)
    b = torch.rand(1, 48, 48, 3)
    for method in ("None", "DIS Medium", "Farneback"):
        flow = estimate_flow(a, b, method, "cpu")
        assert flow.shape == (1, 48, 48, 2), method


def test_pack_registers_computeflow():
    import __init__ as pack
    assert "ComputeFlow| Deforum_Stat-Imp" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_CLASS_MAPPINGS["ComputeFlow| Deforum_Stat-Imp"] is ComputeFlow
