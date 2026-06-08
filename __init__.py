try:
    from .nodes import DepthWarp3D_StatImp
    from .color_nodes import NODE_CLASS_MAPPINGS as COLOR_CLASS, NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY
except ImportError:
    from nodes import DepthWarp3D_StatImp
    from color_nodes import NODE_CLASS_MAPPINGS as COLOR_CLASS, NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY

# Distinct node keys (the type IDs saved in workflows) so this pack coexists with
# the old deforum-comfy-nodes -- install both, swap a node for the other, no collision.
NODE_CLASS_MAPPINGS = {"DepthWarp3D| Deforum_Stat-Imp": DepthWarp3D_StatImp}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthWarp3D| Deforum_Stat-Imp": "Depth Warp 3D"}

# Color-coherence nodes: ColorMatchLAB (fixed anchor) + RollingColorMatch (EMA anchor).
NODE_CLASS_MAPPINGS.update(COLOR_CLASS)
NODE_DISPLAY_NAME_MAPPINGS.update(COLOR_DISPLAY)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
