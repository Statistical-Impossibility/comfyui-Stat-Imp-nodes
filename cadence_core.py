"""Cadence core (brick A2): pure math + emit for vanilla-semantics cadence.

The dissolve pairs the two most recent PAST diffusion lineages (content arrives
<=N frames after its schedule index, same as vanilla Deforum). Tail divisor
D = min(N, F - p) makes the final fade complete exactly at frame F-1.
"""
import torch


def cadence_math(loop_index, cadence, num_frames):
    """Per-iteration scalars. Returns (is_gen, w, first_span).

    is_gen     -- this iteration diffuses (i % N == 0); drives the sampler gate.
    w          -- dissolve weight ((i%N)+1)/D with tail divisor D = min(N, F-p).
    first_span -- i < N: no old lineage exists yet, emit the corrected image as-is.
    """
    n = max(1, int(cadence))
    i = int(loop_index)
    m = i % n
    p = i - m
    d = min(n, int(num_frames) - p)
    return m == 0, (m + 1) / d, i < n


def cadence_emit(is_start, first_span, is_gen, w, warped_old, corrected, fresh,
                 flow=None, flow_factor=1.0):
    """Emit (frame, next_anchor). Unused image args may be None (lazy inputs).

    frame: fresh at frame 0; corrected during the first span; otherwise the
    dissolve OLD*(1-w) + NEW*w where OLD/NEW are the two most recent lineages.
    next_anchor: the value5 roll — corrected on gen iterations, warped_old on tweens.

    Optical-flow morph (A2 v2): on TWEEN frames only (is_gen False), when `flow` is given
    and flow_factor != 0, the two blended lineages (warped_old, corrected) are flow-aligned
    before the dissolve —
      OLD_aligned = flow_warp(warped_old, flow * ( w        * flow_factor))
      NEW_aligned = flow_warp(corrected,  flow * (-(1-w)    * flow_factor))
    Gen frames are left as a plain blend (their pair is corrected->fresh, a different field;
    vanilla likewise flows only the in-between frames). flow=None / flow_factor=0 / zero-field
    / gen-frame => bit-identical to the plain dissolve.
    """
    next_anchor = corrected if is_gen else warped_old
    if is_start:
        return fresh, next_anchor
    if first_span:
        return corrected, next_anchor
    old = corrected if is_gen else warped_old
    new = fresh if is_gen else corrected
    if not is_gen and flow is not None and float(flow_factor) != 0.0:
        old = flow_warp(old, flow * (w * float(flow_factor)))
        new = flow_warp(new, flow * (-(1.0 - w) * float(flow_factor)))
    # clamp mirrors ComfyUI ImageBlend's post-blend clamp (no-op for in-range inputs)
    frame = (old * (1.0 - w) + new * w).clamp(0.0, 1.0)
    return frame, next_anchor


def flow_warp(image, flow):
    """Backward-warp image (B,H,W,C) by flow (B,H,W,2) abs-pixel: out(x) = image(x - flow(x)).

    Mirrors vanilla image_transform_optical_flow (cv2.remap of x-flow) in torch grid_sample,
    bilinear + reflection padding (no black edges). Sampling-only; no estimation.
    """
    import torch.nn.functional as F
    b, h, w, c = image.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                            torch.arange(w, device=image.device, dtype=image.dtype),
                            indexing="ij")
    src_x = xs[None] - flow[..., 0]          # sample location = x - flow
    src_y = ys[None] - flow[..., 1]
    gx = 2.0 * src_x / max(w - 1, 1) - 1.0   # -> normalized [-1,1] for grid_sample
    gy = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)     # (B,H,W,2)
    chw = image.permute(0, 3, 1, 2)
    out = F.grid_sample(chw, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return out.permute(0, 2, 3, 1).contiguous()
