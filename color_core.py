"""Pure-torch color-space conversions + histogram matching for color coherence (brick D1).
Clean-room reimplementation of vanilla Deforum maintain_colors (helpers/colors.py).
All images are (B,H,W,C) float[0,1] RGB torch tensors (ComfyUI IMAGE)."""
import torch
import torch.nn.functional as F

# sRGB <-> CIELAB (D65). Matrices in row form for `img @ M.T`.
_RGB2XYZ = torch.tensor([[0.4124, 0.3576, 0.1805],
                         [0.2126, 0.7152, 0.0722],
                         [0.0193, 0.1192, 0.9505]])
_XYZ2RGB = torch.inverse(_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_EPS = 0.008856
_KAPPA = 903.3


def _srgb_to_linear(c):
    return torch.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _linear_to_srgb(c):
    c = c.clamp(min=0.0)
    return torch.where(c > 0.0031308, 1.055 * (c ** (1 / 2.4)) - 0.055, 12.92 * c)


def rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb.clamp(0, 1))
    xyz = lin @ _RGB2XYZ.t().to(rgb)
    xyz = xyz / _WHITE.to(rgb)
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-8) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def _inv(t):
        t3 = t ** 3
        return torch.where(t3 > _EPS, t3, (116 * t - 16) / _KAPPA)

    xyz = torch.stack([_inv(fx), _inv(fy), _inv(fz)], dim=-1) * _WHITE.to(lab)
    lin = xyz @ _XYZ2RGB.t().to(lab)
    return _linear_to_srgb(lin).clamp(0, 1)


def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, _ = rgb.max(dim=-1)
    mn, _ = rgb.min(dim=-1)
    diff = mx - mn
    eps = 1e-8
    h = torch.zeros_like(mx)
    mask = diff > eps
    rc = (mx - r) / (diff + eps)
    gc = (mx - g) / (diff + eps)
    bc = (mx - b) / (diff + eps)
    hr = (bc - gc)
    hg = (2.0 + rc - bc)
    hb = (4.0 + gc - rc)
    h = torch.where(mx == r, hr, torch.where(mx == g, hg, hb))
    h = (h / 6.0) % 1.0
    h = torch.where(mask, h, torch.zeros_like(h))
    s = torch.where(mx > eps, diff / (mx + eps), torch.zeros_like(mx))
    v = mx
    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.long() % 6
    conds = [i == k for k in range(6)]
    r = torch.where(conds[0], v, torch.where(conds[1], q, torch.where(conds[2], p,
        torch.where(conds[3], p, torch.where(conds[4], t, v)))))
    g = torch.where(conds[0], t, torch.where(conds[1], v, torch.where(conds[2], v,
        torch.where(conds[3], q, torch.where(conds[4], p, p)))))
    b = torch.where(conds[0], p, torch.where(conds[1], p, torch.where(conds[2], t,
        torch.where(conds[3], v, torch.where(conds[4], v, q)))))
    return torch.stack([r, g, b], dim=-1).clamp(0, 1)


def _match_1d(s, r):
    """Map values of 1D tensor s onto the distribution of 1D tensor r (quantile match)."""
    n = s.numel()
    # rank of each src element in [0, n-1] -> normalized quantile
    rank = torch.argsort(torch.argsort(s)).float()
    q = rank / max(n - 1, 1)
    r_sorted, _ = torch.sort(r)
    idx = (q * (r_sorted.numel() - 1)).round().long().clamp(0, r_sorted.numel() - 1)
    return r_sorted[idx]


def match_histograms_torch(src, ref):
    """Per-channel histogram match. src,ref: (B,H,W,C). ref may differ in H,W. Returns src-shaped."""
    out = src.clone()
    B, H, W, C = src.shape
    rB = ref.shape[0]
    for b in range(B):
        rb = ref[b if rB > 1 else 0]
        for c in range(C):
            s = src[b, :, :, c].reshape(-1)
            r = rb[:, :, c].reshape(-1)
            out[b, :, :, c] = _match_1d(s, r).reshape(H, W)
    return out


_SPACES = {
    "LAB": (rgb_to_lab, lab_to_rgb, 0),   # luminance channel index
    "HSV": (rgb_to_hsv, hsv_to_rgb, 2),
    "RGB": (None, None, None),
}


def color_match(image, reference, color_space="LAB", strength=1.0, preserve_luminance=False):
    """Histogram-match `image` to `reference` in `color_space`, blend by `strength`.
    image,reference: (B,H,W,C) [0,1] RGB. Returns (B,H,W,C) [0,1] RGB."""
    if strength <= 0.0:
        return image
    fwd, inv, lum = _SPACES[color_space]
    src = fwd(image) if fwd else image.clone()
    ref = fwd(reference) if fwd else reference
    matched = match_histograms_torch(src, ref)
    if preserve_luminance and lum is not None:
        matched[..., lum] = src[..., lum]
    out = inv(matched) if inv else matched
    out = (image * (1.0 - strength) + out * strength).clamp(0, 1)
    return out


def ema_update(anchor, image, alpha):
    """EMA anchor update for the rolling color anchor (brick D1.1):
    anchor_out = (1 - alpha) * anchor + alpha * image.
    anchor, image: (B,H,W,C) [0,1] RGB. A differently-sized or differently-batched bootstrap
    `anchor` is resized/broadcast to `image` so the blend (and every later frame) stays image-shaped.
    At alpha=0 the anchor is returned untouched (no resize) so the EMA collapses *exactly* to the
    frozen-anchor D1 behavior, bit-for-bit, regardless of reference resolution."""
    if alpha <= 0.0:
        return anchor
    if anchor.shape[1:3] != image.shape[1:3]:
        a = anchor.permute(0, 3, 1, 2)
        a = F.interpolate(a, size=tuple(image.shape[1:3]), mode="bilinear", align_corners=False)
        anchor = a.permute(0, 2, 3, 1)
    if anchor.shape[0] != image.shape[0]:
        anchor = anchor[:1].expand(image.shape[0], -1, -1, -1)
    return (anchor * (1.0 - alpha) + image * alpha).clamp(0, 1)
