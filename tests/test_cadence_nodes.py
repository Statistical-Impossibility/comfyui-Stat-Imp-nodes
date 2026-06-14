"""Tests for the Cadence node class (lazy-input contract + run)."""
import torch

from cadence_nodes import Cadence


def _img(val):
    return torch.full((1, 4, 4, 3), float(val))


def _lazy(i, n, f, start):
    """check_lazy_status with no image inputs resolved yet."""
    return set(Cadence().check_lazy_status(i, n, f, start))


def test_lazy_gen_iteration_never_requests_warped_old():
    # gen (i%N==0): needs corrected+fresh; warp branch must NOT be pulled
    assert _lazy(4, 4, 12, False) == {"corrected", "fresh"}


def test_lazy_tween_iteration_never_requests_fresh():
    # tween: needs corrected+warped_old(+flow for v2); the sampler branch must NOT be pulled
    needed = _lazy(5, 4, 12, False)
    assert {"corrected", "warped_old"} <= needed
    assert "fresh" not in needed


def test_lazy_frame0_requests_fresh_not_warp():
    assert _lazy(0, 4, 12, True) == {"corrected", "fresh"}


def test_lazy_cadence1_always_gen():
    assert _lazy(3, 1, 8, False) == {"corrected", "fresh"}


def test_lazy_requests_only_missing_inputs():
    # corrected already supplied + a zero flow supplied => only warped_old still missing
    flow = torch.zeros(1, 4, 4, 2)
    needed = Cadence().check_lazy_status(5, 4, 12, False, corrected=_img(0.5), flow=flow)
    assert set(needed) == {"warped_old"}


def test_run_tween_returns_blend_anchor_and_is_gen():
    old, corrected = _img(0.2), _img(0.6)
    frame, anchor, is_gen = Cadence().run(6, 4, 12, False, warped_old=old, corrected=corrected)
    assert is_gen is False
    expected = (old * 0.25 + corrected * 0.75).clamp(0, 1)  # w=3/4 at i=6,N=4
    assert torch.allclose(frame, expected)
    assert torch.equal(anchor, old)


def test_run_gen_returns_is_gen_true():
    corrected, fresh = _img(0.4), _img(0.8)
    frame, anchor, is_gen = Cadence().run(4, 4, 12, False, corrected=corrected, fresh=fresh)
    assert is_gen is True
    assert torch.equal(anchor, corrected)


def test_node_has_no_widgets():
    # serialization-trap guard: every input is forceInput or link-only
    # -> the frontend derives ZERO widgets -> widgets_values=[] always aligns
    for name, spec in Cadence.INPUT_TYPES()["required"].items():
        opts = spec[1] if len(spec) > 1 else {}
        assert spec[0] == "IMAGE" or opts.get("forceInput") is True, name


def test_lazy_flags_on_all_image_inputs():
    req = Cadence.INPUT_TYPES()["required"]
    for name in ("warped_old", "corrected", "fresh"):
        assert req[name][1].get("lazy") is True, name


def test_pack_registers_cadence():
    import __init__ as pack
    assert "Cadence| Deforum_Stat-Imp" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_CLASS_MAPPINGS["Cadence| Deforum_Stat-Imp"] is Cadence


def test_flow_optional_present_and_non_lazy():
    # flow is NON-lazy on purpose so an UNCONNECTED flow doesn't crash v1 workflows
    opt = Cadence.INPUT_TYPES().get("optional", {})
    assert opt["flow"][0] == "FLOW"
    assert (len(opt["flow"]) < 2) or (opt["flow"][1].get("lazy") is not True)
    assert opt["flow_factor"][0] == "FLOAT"


def test_lazy_never_requests_flow():
    # flow is non-lazy -> check_lazy_status must never list it (would crash if unconnected)
    for args in [(5, 4, 12, False), (4, 4, 12, False), (0, 4, 12, True)]:
        assert "flow" not in Cadence().check_lazy_status(*args)


def test_gen_frame_ignores_flow():
    import torch
    ramp = (torch.arange(8, dtype=torch.float32).repeat(8, 1) / 7.0)[None, :, :, None].repeat(1, 1, 1, 3)
    flow = torch.zeros(1, 8, 8, 2); flow[..., 0] = 2.0
    # gen iteration (i=4,N=4): flow must be ignored -> equals plain blend
    plain = Cadence().run(4, 4, 12, False, corrected=ramp, fresh=ramp)[0]
    withflow = Cadence().run(4, 4, 12, False, corrected=ramp, fresh=ramp, flow=flow, flow_factor=1.0)[0]
    assert torch.equal(plain, withflow)


def test_run_with_flow_none_matches_v1():
    old, corrected = _img(0.2), _img(0.6)
    a = Cadence().run(6, 4, 12, False, warped_old=old, corrected=corrected)[0]
    b = Cadence().run(6, 4, 12, False, warped_old=old, corrected=corrected,
                      flow=None, flow_factor=1.0)[0]
    assert torch.equal(a, b)


def test_run_with_flow_morphs():
    # ramp content so a horizontal warp changes pixels
    ramp = (torch.arange(8, dtype=torch.float32).repeat(8, 1) / 7.0)[None, :, :, None].repeat(1, 1, 1, 3)
    flow = torch.zeros(1, 8, 8, 2); flow[..., 0] = 2.0
    plain = Cadence().run(6, 4, 12, False, warped_old=ramp, corrected=ramp)[0]
    morph = Cadence().run(6, 4, 12, False, warped_old=ramp, corrected=ramp,
                          flow=flow, flow_factor=1.0)[0]
    assert not torch.allclose(plain, morph, atol=1e-3)
