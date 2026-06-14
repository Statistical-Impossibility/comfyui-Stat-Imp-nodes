"""Optical-flow primitives for cadence (Legacy = OpenCV DIS / Farneback).

Pure functions, no ComfyUI imports, so they run headless under pytest.
Clean-room reimplementation of the flow compute + flow-warp primitives
(mirrors the MIT seanlynch/comfyui-optical-flow approach). All images are
HxWx3 uint8 BGR; flow is (H,W,2) float32 absolute pixel displacement.
"""
import cv2
import numpy as np

def compute_flow_dis(i1, i2, method="DIS Medium"):
    """Dense optical flow i1->i2 via OpenCV DIS. Returns (H,W,2) float32.

    Vanilla's two cadence presets (hybrid_video.py:362-381): DIS Medium = the stock
    MEDIUM preset; DIS Fine = custom (finest scale 0, 192 grad-descent iters, patch 8/stride 4).
    """
    g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    if method == "DIS Medium":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    elif method == "DIS Fine":
        dis = cv2.DISOpticalFlow_create(None)
        dis.setFinestScale(0)
        dis.setGradientDescentIterations(192)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    else:
        raise ValueError(f"unknown DIS method: {method}")
    return dis.calc(g1, g2, None)


def compute_flow_farneback(i1, i2):
    """Dense optical flow i1->i2 via Farneback (the other original-Deforum method)."""
    g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _extend_flow(flow, w, h):
    """Pad a flow field up to (h,w), filling the border with identity coords."""
    fh, fw = flow.shape[:2]
    ox, oy = (w - fw) // 2, (h - fh) // 2
    out = np.zeros((h, w, 2), np.float32)
    out[:, :, 0] = np.arange(w)[None, :]
    out[:, :, 1] = np.arange(h)[:, None]
    out[oy:oy + fh, ox:ox + fw] = flow + np.array([ox, oy], np.float32)
    return out


def _remap(img, flow):
    """Resample img by an absolute-coordinate map, with reflect padding to avoid black edges."""
    border = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    dy, dx = int(h * 0.25), int(w * 0.25)  # 25% reflect pad stops black edges
    big = cv2.copyMakeBorder(img, dy, dy, dx, dx, border)
    lh, lw = big.shape[:2]
    big_flow = _extend_flow(flow, lw, lh)
    out = cv2.remap(big, big_flow, None, cv2.INTER_LINEAR, borderMode=border)
    return out[dy:dy + h, dx:dx + w]  # center-crop back


def image_transform_optical_flow(img, flow, flow_factor=1.0):
    """Warp img along `flow` (scaled by flow_factor). img HxWx3 uint8, flow (H,W,2)."""
    flow = flow * flow_factor if flow_factor != 1.0 else flow.copy()
    flow = -flow  # reversed for remap (sample from source)
    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    return _remap(img, flow)
