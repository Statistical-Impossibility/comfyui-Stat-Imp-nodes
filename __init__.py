try:
    from .nodes import DepthWarp3D_StatImp
    from .color_nodes import NODE_CLASS_MAPPINGS as COLOR_CLASS, NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY
    from .cadence_nodes import NODE_CLASS_MAPPINGS as CADENCE_CLASS, NODE_DISPLAY_NAME_MAPPINGS as CADENCE_DISPLAY
    from .flow_nodes import NODE_CLASS_MAPPINGS as FLOW_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FLOW_DISPLAY
except ImportError:
    from nodes import DepthWarp3D_StatImp
    from color_nodes import NODE_CLASS_MAPPINGS as COLOR_CLASS, NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY
    from cadence_nodes import NODE_CLASS_MAPPINGS as CADENCE_CLASS, NODE_DISPLAY_NAME_MAPPINGS as CADENCE_DISPLAY
    from flow_nodes import NODE_CLASS_MAPPINGS as FLOW_CLASS, NODE_DISPLAY_NAME_MAPPINGS as FLOW_DISPLAY

# Distinct node keys (the type IDs saved in workflows) so this pack coexists with
# the old deforum-comfy-nodes -- install both, swap a node for the other, no collision.
NODE_CLASS_MAPPINGS = {"DepthWarp3D| Deforum_Stat-Imp": DepthWarp3D_StatImp}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthWarp3D| Deforum_Stat-Imp": "Depth Warp 3D"}

# Color-coherence nodes: ColorMatchLAB (fixed anchor) + RollingColorMatch (EMA anchor).
NODE_CLASS_MAPPINGS.update(COLOR_CLASS)
NODE_DISPLAY_NAME_MAPPINGS.update(COLOR_DISPLAY)

# Cadence (brick A2): vanilla-semantics dissolve, lazy inputs gate sampler/warp pulls.
NODE_CLASS_MAPPINGS.update(CADENCE_CLASS)
NODE_DISPLAY_NAME_MAPPINGS.update(CADENCE_DISPLAY)

# Optical-flow estimator (A2 v2): feeds the Cadence node's flow socket.
NODE_CLASS_MAPPINGS.update(FLOW_CLASS)
NODE_DISPLAY_NAME_MAPPINGS.update(FLOW_DISPLAY)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
