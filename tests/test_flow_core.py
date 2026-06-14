"""Tests for OpenCV optical-flow estimators (DIS/Farneback) — numpy BGR uint8 in, (H,W,2) out."""
import cv2
import numpy as np

from flow_core import compute_flow_dis, compute_flow_farneback


def _shifted_pair(dx=4):
    # blurred noise = spatially-coherent texture that DIS/Farneback can actually track
    # (per-pixel white noise is pathological for optical flow — aperture problem everywhere)
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0)
    base = np.stack([base] * 3, axis=-1)  # HxWx3 BGR
    shifted = np.roll(base, dx, axis=1)   # content moves right by dx
    return base, shifted


def test_dis_medium_recovers_horizontal_shift():
    a, b = _shifted_pair(4)
    flow = compute_flow_dis(a, b, "DIS Medium")          # flow a->b
    cx = flow[20:44, 20:44, 0].mean()                    # central region avoids wrap edges
    assert cx > 2.0                                       # detects rightward motion of several px


def test_dis_fine_runs_and_shapes():
    a, b = _shifted_pair(4)
    flow = compute_flow_dis(a, b, "DIS Fine")
    assert flow.shape == (64, 64, 2) and flow.dtype == np.float32


def test_farneback_recovers_shift():
    a, b = _shifted_pair(4)
    flow = compute_flow_farneback(a, b)
    assert flow[20:44, 20:44, 0].mean() > 2.0
