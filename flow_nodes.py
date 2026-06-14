"""ComfyUI node: ComputeFlow — optical-flow estimator for cadence (A2 v2).

Estimator lives OUTSIDE the Cadence node (transform-agnostic boundary). Outputs FLOW =
torch (1,H,W,2) abs-pixel displacement image_from->image_to; "None" => zero flow (identity).
Backends: RAFT (torchvision, GPU/CPU), DIS Medium/Fine + Farneback (OpenCV, CPU).
"""
import numpy as np
import torch

try:
    from .flow_core import compute_flow_dis, compute_flow_farneback
except ImportError:
    from flow_core import compute_flow_dis, compute_flow_farneback


def _to_bgr_uint8(image):
    """ComfyUI IMAGE (1,H,W,3) RGB float 0..1 -> HxWx3 uint8 BGR for OpenCV/RAFT cores."""
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)  # HWC RGB
    return arr[:, :, ::-1].copy()                                                # -> BGR


def _pick_device(device):
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def estimate_flow(image_from, image_to, method, device):
    """Return FLOW (1,H,W,2) float32 abs-pixel, image_from->image_to, sized to image_to.

    image_from is resized to image_to's resolution first: on bootstrap/first-span frames the
    old-anchor lineage can be a different size than the corrected image, and the flow backends
    (RAFT/DIS/Farneback) all assert equal-size inputs. On real tween frames both are already the
    render resolution, so the resize is a no-op. (Cadence ignores the result on guarded frames.)
    """
    import torch.nn.functional as F
    _, h, w, _ = image_to.shape
    if image_from.shape[1:3] != image_to.shape[1:3]:
        chw = image_from.permute(0, 3, 1, 2)
        chw = F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)
        image_from = chw.permute(0, 2, 3, 1).contiguous()
    if method == "None":
        return torch.zeros(1, h, w, 2, dtype=torch.float32)
    a_bgr = _to_bgr_uint8(image_from)
    b_bgr = _to_bgr_uint8(image_to)
    if method in ("DIS Medium", "DIS Fine"):
        f = compute_flow_dis(a_bgr, b_bgr, method)
    elif method == "Farneback":
        f = compute_flow_farneback(a_bgr, b_bgr)
    elif method == "RAFT":
        try:
            from .raft_core import raft_flow
        except ImportError:
            from raft_core import raft_flow
        f = raft_flow(a_bgr, b_bgr, device=_pick_device(device))
    else:
        raise ValueError(f"unknown flow method: {method}")
    return torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32))[None]      # (1,H,W,2)


class ComputeFlow:
    CATEGORY = "Stat-Imp/Deforum"
    RETURN_TYPES = ("FLOW",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_from": ("IMAGE",),
            "image_to": ("IMAGE",),
            "method": (["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"],),
            "device": (["AUTO", "cuda", "cpu"], {"default": "AUTO"}),
        }}

    def run(self, image_from, image_to, method, device):
        return (estimate_flow(image_from, image_to, method, device),)


NODE_CLASS_MAPPINGS = {"ComputeFlow| Deforum_Stat-Imp": ComputeFlow}
NODE_DISPLAY_NAME_MAPPINGS = {"ComputeFlow| Deforum_Stat-Imp": "Compute Optical Flow"}
