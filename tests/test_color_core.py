import torch
from color_core import rgb_to_lab, lab_to_rgb
from color_core import rgb_to_hsv, hsv_to_rgb
from color_core import match_histograms_torch
from color_core import color_match
from color_core import ema_update


def test_lab_roundtrip_identity():
    torch.manual_seed(0)
    rgb = torch.rand(2, 16, 16, 3)            # (B,H,W,C) [0,1] RGB
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert back.shape == rgb.shape
    assert torch.max(torch.abs(back - rgb)) < 1e-3   # invertible


def test_lab_known_white():
    rgb = torch.ones(1, 2, 2, 3)             # white
    lab = rgb_to_lab(rgb)
    assert torch.abs(lab[..., 0] - 100.0).max() < 0.5  # L≈100
    assert torch.abs(lab[..., 1:]).max() < 1.0         # a,b≈0


def test_hsv_roundtrip_identity():
    torch.manual_seed(1)
    rgb = torch.rand(2, 8, 8, 3)
    back = hsv_to_rgb(rgb_to_hsv(rgb))
    assert torch.max(torch.abs(back - rgb)) < 1e-4


def test_histogram_match_pulls_to_reference():
    torch.manual_seed(2)
    ref = torch.rand(1, 32, 32, 3)
    src = (ref * 0.4 + 0.3).clamp(0, 1)          # washed-out, shifted version
    matched = match_histograms_torch(src, ref)
    assert matched.shape == src.shape
    # matched should be far closer to ref's per-channel mean than src was
    err_src = torch.abs(src.mean(dim=(1, 2)) - ref.mean(dim=(1, 2))).sum()
    err_out = torch.abs(matched.mean(dim=(1, 2)) - ref.mean(dim=(1, 2))).sum()
    assert err_out < err_src * 0.1


def test_histogram_match_handles_size_mismatch():
    src = torch.rand(1, 16, 16, 3)
    ref = torch.rand(1, 24, 24, 3)               # different resolution
    out = match_histograms_torch(src, ref)
    assert out.shape == src.shape                # quantile-based, size-independent


def test_color_match_strength_zero_is_identity():
    img = torch.rand(1, 8, 8, 3); ref = torch.rand(1, 8, 8, 3)
    out = color_match(img, ref, color_space="LAB", strength=0.0, preserve_luminance=False)
    assert torch.max(torch.abs(out - img)) < 1e-6


def test_color_match_preserve_luminance_keeps_brightness():
    torch.manual_seed(3)
    img = torch.rand(1, 16, 16, 3); ref = torch.rand(1, 16, 16, 3)
    out = color_match(img, ref, color_space="LAB", strength=1.0, preserve_luminance=True)
    from color_core import rgb_to_lab
    assert torch.abs(rgb_to_lab(out)[..., 0] - rgb_to_lab(img)[..., 0]).mean() < 2.0  # L≈unchanged


def test_color_match_lab_reduces_color_drift():
    torch.manual_seed(4)
    ref = torch.rand(1, 32, 32, 3)
    img = (ref * 0.5 + torch.tensor([0.2, 0.0, 0.0])).clamp(0, 1)   # reddish drift
    out = color_match(img, ref, color_space="LAB", strength=1.0, preserve_luminance=False)
    d_in = torch.abs(img.mean((1, 2)) - ref.mean((1, 2))).sum()
    d_out = torch.abs(out.mean((1, 2)) - ref.mean((1, 2))).sum()
    assert d_out < d_in


def test_node_runs_and_returns_image():
    import importlib
    cn = importlib.import_module("color_nodes")
    node = cn.ColorMatchLAB()
    img = torch.rand(1, 8, 8, 3); ref = torch.rand(1, 8, 8, 3)
    (out,) = node.run(img, ref, "LAB", 1.0, True)
    assert out.shape == img.shape and out.dtype == img.dtype


def test_init_registers_node():
    import importlib
    m = importlib.import_module("__init__")
    assert "ColorMatchLAB| Deforum_Stat-Imp" in m.NODE_CLASS_MAPPINGS


# --- D1.1 RollingColorMatch (EMA color anchor) ---

def test_ema_alpha_zero_keeps_anchor():
    torch.manual_seed(10)
    anchor = torch.rand(1, 8, 8, 3); image = torch.rand(1, 8, 8, 3)
    out = ema_update(anchor, image, 0.0)
    assert torch.max(torch.abs(out - anchor)) < 1e-6      # frozen anchor (= D1 behavior)


def test_ema_alpha_one_becomes_image():
    torch.manual_seed(11)
    anchor = torch.rand(1, 8, 8, 3); image = torch.rand(1, 8, 8, 3)
    out = ema_update(anchor, image, 1.0)
    assert torch.max(torch.abs(out - image)) < 1e-6      # tracks previous frame


def test_ema_alpha_half_is_midpoint():
    anchor = torch.zeros(1, 4, 4, 3); image = torch.ones(1, 4, 4, 3)
    out = ema_update(anchor, image, 0.5)
    assert torch.max(torch.abs(out - 0.5)) < 1e-6


def test_ema_resizes_mismatched_anchor():
    anchor = torch.rand(1, 24, 24, 3)                     # bootstrap at different resolution
    image = torch.rand(1, 16, 16, 3)
    out = ema_update(anchor, image, 0.5)
    assert out.shape == image.shape                       # output always image-shaped


def test_ema_alpha_zero_does_not_resize_mismatched_anchor():
    """Regression: at alpha=0 a differently-sized bootstrap anchor must pass through UNCHANGED
    (no bilinear resize), else the frozen anchor drifts from the native-res reference and the
    EMA stops being bit-identical to D1 (the 'similar but not identical' bug)."""
    anchor = torch.rand(1, 24, 24, 3)                     # reference at native (non-frame) resolution
    image = torch.rand(1, 16, 16, 3)
    out = ema_update(anchor, image, 0.0)
    assert out.shape == anchor.shape                      # original shape kept
    assert torch.equal(out, anchor)                       # byte-identical


def test_loop_alpha_zero_native_res_anchor_matches_d1():
    """End-to-end: with alpha=0 and a native-res (non-frame) reference, every frame is matched to
    the ORIGINAL reference (== old ColorMatchLAB), not a resized copy."""
    import importlib
    cn = importlib.import_module("color_nodes")
    torch.manual_seed(22)
    node = cn.RollingColorMatch()
    ref = torch.rand(1, 24, 24, 3)                        # reference bigger than frames (e.g. 768 vs 512)
    frames = [torch.rand(1, 16, 16, 3) for _ in range(4)]
    anchor = ref.clone()                                  # bootstrap = native-res reference
    for f in frames:
        matched, anchor = node.run(f, anchor, 0.0, "LAB", 0.05, True)
        expected = color_match(f, ref, "LAB", 0.05, True)
        assert torch.max(torch.abs(matched - expected)) < 1e-6
        assert torch.equal(anchor, ref)                  # anchor stays the native-res reference


def test_rolling_node_returns_matched_and_anchor():
    import importlib
    cn = importlib.import_module("color_nodes")
    node = cn.RollingColorMatch()
    img = torch.rand(1, 8, 8, 3); anchor = torch.rand(1, 8, 8, 3)
    matched, anchor_out = node.run(img, anchor, 0.1, "LAB", 1.0, True)
    assert matched.shape == img.shape and anchor_out.shape == img.shape
    assert matched.dtype == img.dtype


def test_rolling_node_strength_zero_passes_image():
    import importlib
    cn = importlib.import_module("color_nodes")
    node = cn.RollingColorMatch()
    img = torch.rand(1, 8, 8, 3); anchor = torch.rand(1, 8, 8, 3)
    matched, _ = node.run(img, anchor, 0.3, "LAB", 0.0, False)
    assert torch.max(torch.abs(matched - img)) < 1e-6        # strength=0 -> identity


def test_rolling_node_alpha_zero_freezes_anchor():
    import importlib
    cn = importlib.import_module("color_nodes")
    node = cn.RollingColorMatch()
    img = torch.rand(1, 8, 8, 3); anchor = torch.rand(1, 8, 8, 3)
    _, anchor_out = node.run(img, anchor, 0.0, "LAB", 1.0, True)
    assert torch.max(torch.abs(anchor_out - anchor)) < 1e-6  # alpha=0 -> frozen (= D1)


def test_rolling_node_matched_equals_color_match():
    import importlib
    cn = importlib.import_module("color_nodes")
    node = cn.RollingColorMatch()
    img = torch.rand(1, 8, 8, 3); anchor = torch.rand(1, 8, 8, 3)
    matched, _ = node.run(img, anchor, 0.5, "LAB", 1.0, True)
    ref = color_match(img, anchor, "LAB", 1.0, True)         # reuses D1 math, anchor as reference
    assert torch.max(torch.abs(matched - ref)) < 1e-6


def test_rolling_init_registers_node():
    import importlib
    m = importlib.import_module("__init__")
    assert "RollingColorMatch| Deforum_Stat-Imp" in m.NODE_CLASS_MAPPINGS
    assert "ColorMatchLAB| Deforum_Stat-Imp" in m.NODE_CLASS_MAPPINGS   # old node kept


def test_loop_alpha_zero_equals_frozen_frame0():
    """alpha=0: anchor never moves -> every frame matched to frame-0 == D1 ColorMatchLAB."""
    import importlib
    cn = importlib.import_module("color_nodes")
    torch.manual_seed(20)
    node = cn.RollingColorMatch()
    frames = [torch.rand(1, 16, 16, 3) for _ in range(4)]
    anchor = frames[0].clone()                       # bootstrap = frame 0
    for f in frames:
        matched, anchor = node.run(f, anchor, 0.0, "LAB", 1.0, False)
        expected = color_match(f, frames[0], "LAB", 1.0, False)
        assert torch.max(torch.abs(matched - expected)) < 1e-6
        assert torch.max(torch.abs(anchor - frames[0])) < 1e-6   # anchor stayed frame-0


def test_loop_alpha_one_tracks_previous_frame():
    """alpha=1: anchor == previous frame -> frame N matched to frame N-1."""
    import importlib
    cn = importlib.import_module("color_nodes")
    torch.manual_seed(21)
    node = cn.RollingColorMatch()
    frames = [torch.rand(1, 16, 16, 3) for _ in range(3)]
    anchor = frames[0].clone()
    prev = frames[0]
    for f in frames[1:]:
        matched, anchor = node.run(f, anchor, 1.0, "LAB", 1.0, False)
        expected = color_match(f, prev, "LAB", 1.0, False)       # ref was the previous frame
        assert torch.max(torch.abs(matched - expected)) < 1e-6
        assert torch.max(torch.abs(anchor - f)) < 1e-6           # anchor became this frame
        prev = f
