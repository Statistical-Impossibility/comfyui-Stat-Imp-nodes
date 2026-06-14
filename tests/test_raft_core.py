"""RAFT smoke test (torchvision, CPU). Downloads weights on first run."""
import cv2
import numpy as np
import pytest

raft = pytest.importorskip("raft_core")


@pytest.mark.slow
def test_raft_recovers_horizontal_shift():
    # RAFT downsamples by 8 and needs feature maps >= 16 -> images must be >= 128px
    rng = np.random.default_rng(1)
    base = (rng.random((128, 128)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=4.0)
    base = np.stack([base] * 3, axis=-1)
    shifted = np.roll(base, 4, axis=1)
    flow = raft.raft_flow(base, shifted, device="cpu")    # a->b
    assert flow.shape == (128, 128, 2)
    assert flow[40:88, 40:88, 0].mean() > 1.5
