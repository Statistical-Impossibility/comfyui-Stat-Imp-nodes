"""RAFT optical flow (Improved variant) — torchvision (BSD-3), opt-in.

Not used by the Legacy DIS/Farneback path. Requires `torchvision`; the weights
(~20 MB) auto-download on first use, so this stays opt-in. The model is lazily
constructed and cached. Returns (H,W,2) float32 absolute pixel flow.
"""
import numpy as np

_RAFT = None


def _model(device):
    global _RAFT
    if _RAFT is None:
        import torch  # noqa: F401
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval().to(device)
    return _RAFT


def raft_flow(i1, i2, device="cuda"):
    """i1,i2: HxWx3 uint8 BGR. Returns (H,W,2) float32 absolute pixel flow."""
    import torch
    import torch.nn.functional as F

    def prep(img):
        t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0  # BGR->RGB BCHW
        h, w = t.shape[-2:]
        t = F.pad(t, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        return (t * 2 - 1).to(device), (h, w)

    a, (h, w) = prep(i1)
    b, _ = prep(i2)
    with torch.inference_mode():
        flow = _model(device)(a, b)[-1]  # (1,2,H,W), last = finest
    flow = flow[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()  # (H,W,2)
    return np.ascontiguousarray(flow, dtype=np.float32)
