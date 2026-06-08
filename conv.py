"""Tensor layout helpers (no ComfyUI imports)."""
import torch


def bhwc_to_bchw(t):
    return t.permute(0, 3, 1, 2).contiguous()


def bchw_to_bhwc(t):
    return t.permute(0, 2, 3, 1).contiguous()


def to_single_channel(depth):
    """Accept IMAGE (B,H,W,C) or MASK (B,H,W)/(H,W); return (H,W)."""
    if depth.dim() == 4:                # BHWC image -> first item, mean over channels
        return depth[0].mean(dim=-1)
    if depth.dim() == 3:                # B,H,W mask -> first
        return depth[0]
    return depth                        # H,W
