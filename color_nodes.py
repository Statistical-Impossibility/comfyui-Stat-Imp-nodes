"""ComfyUI node: ColorMatchLAB — histogram color coherence (brick D1).
Stateless drop-in replacement for `ImageColorMatch| Deforum` (which is mean/std-only, no LAB)."""
try:
    from .color_core import color_match, ema_update
except ImportError:
    from color_core import color_match, ema_update


class ColorMatchLAB:
    CATEGORY = "Stat-Imp/Deforum/Color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "reference": ("IMAGE",),
            "color_space": (["LAB", "HSV", "RGB"],),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "preserve_luminance": ("BOOLEAN", {"default": False}),
        }}

    def run(self, image, reference, color_space, strength, preserve_luminance):
        return (color_match(image, reference, color_space, float(strength), bool(preserve_luminance)),)


class RollingColorMatch:
    """EMA-anchor color coherence (brick D1.1). Stateless: the anchor is threaded by the workflow
    (ForLoop value3), not held in the node. matched = color_match(image, anchor_in);
    anchor_out = (1-alpha)*anchor_in + alpha*image. alpha=0 => frozen anchor (== ColorMatchLAB);
    alpha=1 => match the previous frame; middle => drifting weighted average."""
    CATEGORY = "Stat-Imp/Deforum/Color"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("matched", "anchor_out")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "anchor_in": ("IMAGE",),
            "alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "color_space": (["LAB", "HSV", "RGB"],),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "preserve_luminance": ("BOOLEAN", {"default": False}),
        }}

    def run(self, image, anchor_in, alpha, color_space, strength, preserve_luminance):
        matched = color_match(image, anchor_in, color_space, float(strength), bool(preserve_luminance))
        anchor_out = ema_update(anchor_in, image, float(alpha))
        return (matched, anchor_out)


NODE_CLASS_MAPPINGS = {
    "ColorMatchLAB| Deforum_Stat-Imp": ColorMatchLAB,
    "RollingColorMatch| Deforum_Stat-Imp": RollingColorMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatchLAB| Deforum_Stat-Imp": "Color Match LAB",
    "RollingColorMatch| Deforum_Stat-Imp": "Rolling Color Match (EMA)",
}
