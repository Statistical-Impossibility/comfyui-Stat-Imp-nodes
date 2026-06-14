"""Tests for cadence_core (brick A2 consolidation)."""
import pytest
import torch

from cadence_core import cadence_math, cadence_emit


# (i, N, F) -> (is_gen, w, first_span)
MATH_TABLE = [
    # cadence=1 degenerates: always gen, w=1, only i=0 in first span
    (0, 1, 8, True, 1.0, True),
    (3, 1, 8, True, 1.0, False),
    (7, 1, 8, True, 1.0, False),
    # N=4, F=12 (no tail truncation: D=N=4 mid-video)
    (0, 4, 12, True, 1 / 4, True),
    (1, 4, 12, False, 2 / 4, True),
    (3, 4, 12, False, 4 / 4, True),
    (4, 4, 12, True, 1 / 4, False),
    (6, 4, 12, False, 3 / 4, False),
    (7, 4, 12, False, 4 / 4, False),
    (11, 4, 12, False, 4 / 4, False),
    # tail divisor: F=9, N=6 -> last span p=6, D=min(6, 9-6)=3, fade completes at F-1
    (6, 6, 9, True, 1 / 3, False),
    (7, 6, 9, False, 2 / 3, False),
    (8, 6, 9, False, 3 / 3, False),
    # cadence=0 guard: treated as 1 (no div-by-zero)
    (2, 0, 8, True, 1.0, False),
]


@pytest.mark.parametrize("i,n,f,is_gen,w,first_span", MATH_TABLE)
def test_cadence_math_table(i, n, f, is_gen, w, first_span):
    g, ww, fs = cadence_math(i, n, f)
    assert g is is_gen
    assert ww == pytest.approx(w)
    assert fs is first_span


def _img(val, h=4, w=4):
    return torch.full((1, h, w, 3), float(val))


def test_emit_frame0_is_fresh():
    fresh, corrected = _img(0.9), _img(0.5)
    frame, anchor = cadence_emit(True, True, True, 1.0, None, corrected, fresh)
    assert torch.equal(frame, fresh)
    assert torch.equal(anchor, corrected)  # is_gen at i=0 -> anchor rolls to corrected


def test_emit_first_span_tween_is_corrected():
    corrected, old = _img(0.5), _img(0.2)
    # i=1, N=4: first_span tween -> frame = corrected, anchor = warped_old
    frame, anchor = cadence_emit(False, True, False, 0.5, old, corrected, None)
    assert torch.equal(frame, corrected)
    assert torch.equal(anchor, old)


def test_emit_gen_blends_corrected_to_fresh():
    # gen iteration mid-video: OLD=corrected, NEW=fresh, w=1/4
    corrected, fresh = _img(0.4), _img(0.8)
    frame, anchor = cadence_emit(False, False, True, 0.25, None, corrected, fresh)
    expected = corrected * 0.75 + fresh * 0.25
    assert torch.allclose(frame, expected)
    assert torch.equal(anchor, corrected)


def test_emit_tween_blends_old_to_corrected():
    # tween mid-video: OLD=warped_old, NEW=corrected, w=3/4
    old, corrected = _img(0.2), _img(0.6)
    frame, anchor = cadence_emit(False, False, False, 0.75, old, corrected, None)
    expected = old * 0.25 + corrected * 0.75
    assert torch.allclose(frame, expected)
    assert torch.equal(anchor, old)


def test_emit_cadence1_every_frame_fresh():
    # N=1: is_gen always, w=1 -> frame == fresh exactly (no-cadence degeneration)
    corrected, fresh = _img(0.4), _img(0.8)
    frame, _ = cadence_emit(False, False, True, 1.0, None, corrected, fresh)
    assert torch.equal(frame, fresh)


def test_emit_output_clamped():
    old, corrected = _img(0.0), _img(1.0)
    frame, _ = cadence_emit(False, False, False, 0.5, old, corrected, None)
    assert frame.min() >= 0.0 and frame.max() <= 1.0


def test_emit_flow_none_equals_v1():
    # flow=None must reproduce the plain-dissolve path exactly
    old, corrected = _img(0.2), _img(0.6)
    a = cadence_emit(False, False, False, 0.75, old, corrected, None)
    b = cadence_emit(False, False, False, 0.75, old, corrected, None, flow=None, flow_factor=1.0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_emit_flow_factor_zero_equals_v1():
    old, corrected = _img(0.2), _img(0.6)
    flow = torch.ones(1, 4, 4, 2)  # non-trivial flow, but factor 0 disables morph
    a = cadence_emit(False, False, False, 0.5, old, corrected, None)
    b = cadence_emit(False, False, False, 0.5, old, corrected, None, flow=flow, flow_factor=0.0)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def test_emit_flow_zero_field_equals_v1():
    old, corrected = _img(0.2), _img(0.6)
    flow = torch.zeros(1, 4, 4, 2)
    a = cadence_emit(False, False, False, 0.5, old, corrected, None)
    b = cadence_emit(False, False, False, 0.5, old, corrected, None, flow=flow, flow_factor=1.0)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def _ramp(h=8, w=8):
    # horizontal ramp so a horizontal warp visibly changes pixels
    x = torch.arange(w, dtype=torch.float32).repeat(h, 1) / (w - 1)
    return x[None, :, :, None].repeat(1, 1, 1, 3)


def test_emit_flow_morph_differs_from_plain_blend():
    old, corrected = _ramp(), _ramp()
    flow = torch.zeros(1, 8, 8, 2)
    flow[..., 0] = 2.0  # real horizontal motion -> morph shifts content, plain blend does not
    plain = cadence_emit(False, False, False, 0.5, old, corrected, None)[0]
    morph = cadence_emit(False, False, False, 0.5, old, corrected, None, flow=flow, flow_factor=1.0)[0]
    assert not torch.allclose(plain, morph, atol=1e-3)


def test_emit_flow_ignored_on_first_span_and_frame0():
    corrected, fresh, old = _img(0.5), _img(0.9), _img(0.2)
    flow = torch.ones(1, 4, 4, 2)
    # frame 0 -> fresh regardless of flow
    assert torch.equal(cadence_emit(True, True, True, 1.0, old, corrected, fresh, flow=flow)[0], fresh)
    # first span -> corrected regardless of flow
    assert torch.equal(cadence_emit(False, True, False, 0.5, old, corrected, None, flow=flow)[0], corrected)
