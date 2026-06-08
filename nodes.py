"""ComfyUI node wrapper for the exact-Deforum 3D camera warp.

Drop-in replacement for the old DepthWarp3D: SAME inputs, SAME outputs, SAME
depth handling. The ONLY change is the camera math -- warp_core now uses the
exact-Deforum forward projection (proven pixel-identical to the original in
tests/), instead of the old mirrored OpenCV inverse warp.
"""
try:  # inside ComfyUI the folder is imported as a package (relative imports)
    from .conv import bhwc_to_bchw, bchw_to_bhwc, to_single_channel
    from .warp_core import depthwarp3d
except ImportError:  # standalone (pytest)
    from conv import bhwc_to_bchw, bchw_to_bhwc, to_single_channel
    from warp_core import depthwarp3d


class DepthWarp3D_StatImp:
    CATEGORY = "Stat-Imp/Deforum/Depth3D"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "reveal_mask")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d: ("FLOAT", {"default": d, "min": -10000.0, "max": 10000.0, "step": 0.01})
        return {"required": {
            "image": ("IMAGE",), "depth": ("IMAGE",),
            "translation_x": f(0.0), "translation_y": f(0.0), "translation_z": f(0.0),
            "rotation_3d_x": f(0.0), "rotation_3d_y": f(0.0), "rotation_3d_z": f(0.0),
            "fov": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 179.0, "step": 0.5}),
            "translation_scale": ("FLOAT", {"default": 0.005, "min": 0.0001, "max": 1.0, "step": 0.0001}),
            "depth_invert": ("BOOLEAN", {"default": True}),     # MiDaS = inverse depth (default backend)
            "depth_equalize": ("BOOLEAN", {"default": True}),
            "padding_mode": (["border", "reflection", "zeros"],),
            "sampling_mode": (["bilinear", "nearest", "bicubic"],),
        }}

    def run(self, image, depth, translation_x, translation_y, translation_z,
            rotation_3d_x, rotation_3d_y, rotation_3d_z, fov, translation_scale,
            depth_invert, depth_equalize, padding_mode, sampling_mode):
        img = bhwc_to_bchw(image)
        d = to_single_channel(depth)
        warped, hole = depthwarp3d(img, d, translation_x, translation_y, translation_z,
                                   rotation_3d_x, rotation_3d_y, rotation_3d_z, fov,
                                   translation_scale, depth_invert, depth_equalize,
                                   padding_mode, sampling_mode)
        return (bchw_to_bhwc(warped), hole)   # hole: (1,H,W) MASK, 1 = newly-revealed
