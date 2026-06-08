"""Exact port of original Deforum's 3D camera warp (forward formulation).

Reproduces `transform_image_3d` from deforum-stable-diffusion
(helpers/animation.py): project a world grid (x,y in [-1,1], z = depth) through
an OLD camera (identity) and a NEW camera (R,T), take the screen-space offset,
and resample the image by that offset. Pure torch -- no pytorch3d at runtime.

Conventions copied verbatim from py3d_tools (BSD), so motion matches 1:1:
  * row-vector transform:        X_view = X_world @ R + T
  * symmetric FoV projection:    ndc_x = (1/(tan(fov/2)*aspect)) * x_view / z_view
                                 ndc_y = (1/ tan(fov/2))          * y_view / z_view
    (znear/zfar cancel out of the xy offset -- they only scale the z channel,
     which the original discards via [:, 0:2], so we don't need them here)
  * euler_angles_to_matrix(XYZ): R = Rx @ Ry @ Rz, used directly (NOT transposed)
  * translate vector:            T = [-tx, +ty, -tz] * (1/200)
  * sampling:                    grid = identity_grid - offset, align_corners=False

This is geometry only. Turning a depth IMAGE into the z values fed here is the
node wrapper's job (and matching the original's exact depth scale is a separate,
later task).
"""
import math
import torch
import torch.nn.functional as F

TRANSLATION_SCALE = 1.0 / 200.0   # matches original Deforum (inherited from Disco)


def euler_xyz_to_matrix(rx, ry, rz, device, dtype):
    """R = Rx @ Ry @ Rz, identical to py3d euler_angles_to_matrix(..., 'XYZ').
    Angles in radians. Returns (3, 3)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=dtype)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=dtype)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=dtype)
    return Rx @ Ry @ Rz


def _project(xyz, R, T, inv_tan, aspect, eps):
    """World points (N,3) -> NDC xy (N,2). X_view = xyz @ R + T, then perspective divide."""
    Xv = xyz @ R + T
    z = Xv[:, 2:3]
    z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)   # guard div-by-zero only
    ndc_x = (inv_tan / aspect) * Xv[:, 0:1] / z
    ndc_y = inv_tan * Xv[:, 1:2] / z
    return torch.cat([ndc_x, ndc_y], dim=1)


def forward_warp(image_bchw, z_hw, tx, ty, tz, rx, ry, rz, fov,
                 translation_scale=TRANSLATION_SCALE, aspect=1.0,
                 padding_mode="border", sampling_mode="bilinear", eps=1e-9):
    """Warp (1,C,H,W) image through depth z_hw by a 6-DOF camera move + fov.
    Rotations in DEGREES. Returns (warped (1,C,H,W), hole_mask (1,H,W)) where
    hole_mask = 1 marks pixels that sampled from outside the source frame
    (newly revealed). fp32 projection."""
    dev = image_bchw.device
    dt = torch.float32
    img = image_bchw.float()
    B, C, H, W = img.shape

    z = z_hw.float().to(dev)
    if z.shape[-2:] != (H, W):
        z = F.interpolate(z.view(1, 1, *z.shape[-2:]), size=(H, W),
                          mode="bilinear", align_corners=False).view(H, W)

    # world grid: y,x = meshgrid(linspace(-1,1,H), linspace(-1,1,W)) -- order matches original
    ys = torch.linspace(-1.0, 1.0, H, device=dev, dtype=dt)
    xs = torch.linspace(-1.0, 1.0, W, device=dev, dtype=dt)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    xyz = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)   # (HW,3)

    inv_tan = 1.0 / math.tan(math.radians(fov) / 2.0)
    R_old = torch.eye(3, device=dev, dtype=dt)
    T_old = torch.zeros(3, device=dev, dtype=dt)
    R_new = euler_xyz_to_matrix(math.radians(rx), math.radians(ry), math.radians(rz), dev, dt)
    T_new = torch.tensor([-tx, ty, -tz], device=dev, dtype=dt) * translation_scale

    ndc_old = _project(xyz, R_old, T_old, inv_tan, aspect, eps)
    ndc_new = _project(xyz, R_new, T_new, inv_tan, aspect, eps)
    offset = (ndc_new - ndc_old).view(1, H, W, 2)

    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=dev, dtype=dt)
    base = F.affine_grid(identity, [1, 1, H, W], align_corners=False)   # (1,H,W,2)
    grid = base - offset

    warped = F.grid_sample(img, grid, mode=sampling_mode,
                           padding_mode=padding_mode, align_corners=False)
    inside = ((grid[..., 0] >= -1) & (grid[..., 0] <= 1) &
              (grid[..., 1] >= -1) & (grid[..., 1] <= 1)).to(dt)         # (1,H,W)
    return warped.clamp(0, 1), (1.0 - inside)


def normalize_depth(depth_hw, invert=False, equalize=False):
    """Map a depth map to z in [1,2] -- identical to the original node's handling
    (depth offset 1, range 1). invert = MiDaS inverse depth; equalize = histogram-eq."""
    d = depth_hw.float()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)                # min-max -> [0,1]
    if equalize:
        flat = d.flatten()
        hist = torch.histc(flat, bins=1024, min=0.0, max=1.0)
        cdf = torch.cumsum(hist, 0)
        cdf = cdf / cdf[-1]
        idx = (flat * 1023).clamp(0, 1023).long()
        d = cdf[idx].view_as(d)
    if invert:
        d = 1.0 - d
    return d * 1.0 + 1.0                                          # z in [1,2]


def depthwarp3d(image_bchw, depth_hw, tx, ty, tz, rx, ry, rz, fov,
                translation_scale=TRANSLATION_SCALE, invert=False, equalize=False,
                padding_mode="border", sampling_mode="bilinear"):
    """Drop-in replacement for the old node's warp: same signature + depth handling,
    but the camera math is now the exact-Deforum forward warp. Returns (warped, hole)."""
    z = normalize_depth(depth_hw, invert=invert, equalize=equalize)
    return forward_warp(image_bchw, z, tx, ty, tz, rx, ry, rz, fov,
                        translation_scale=translation_scale,
                        padding_mode=padding_mode, sampling_mode=sampling_mode)
