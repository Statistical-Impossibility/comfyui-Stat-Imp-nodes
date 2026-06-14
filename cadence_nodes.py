"""ComfyUI node: Cadence — vanilla-semantics cadence dissolve (brick A2).

Consolidates the CADENCE group's scalar math + switch/blend logic. The 2nd
warp (old-anchor) stays OUTSIDE the node so any transform can feed warped_old
(3D depth warp today, optical flow later). The sampler gate also stays
graph-level (LazySwitch on the LATENT path); this node only emits its boolean.

Lazy contract: tween iterations never request `fresh` (so the sampler branch
is never executed); gen iterations never request `warped_old` (warp skipped).
"""
try:
    from .cadence_core import cadence_math, cadence_emit
except ImportError:
    from cadence_core import cadence_math, cadence_emit


class Cadence:
    CATEGORY = "Stat-Imp/Deforum"
    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("frame", "next_anchor", "is_gen")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # all scalars forceInput: the node has ZERO widgets, so saved
            # widgets_values=[] can never mis-align in the frontend (FINDINGS trap)
            "loop_index": ("INT", {"default": 0, "forceInput": True}),
            "cadence": ("INT", {"default": 1, "forceInput": True}),
            "num_frames": ("INT", {"default": 1, "forceInput": True}),
            "is_start": ("BOOLEAN", {"default": False, "forceInput": True}),
            "warped_old": ("IMAGE", {"lazy": True}),
            "corrected": ("IMAGE", {"lazy": True}),
            "fresh": ("IMAGE", {"lazy": True}),
        }, "optional": {
            # A2 v2 optical-flow morph (omit both => bit-identical v1 dissolve).
            # NON-lazy on purpose: a lazy optional input that is left UNCONNECTED makes
            # ComfyUI raise "needs input flow, but there is no input" — which would break
            # every v1 workflow that has no ComputeFlow wired. Non-lazy => unconnected is
            # simply None. flow is link-only + flow_factor is forceInput => still ZERO widgets.
            "flow": ("FLOW",),
            "flow_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "forceInput": True}),
        }}

    def check_lazy_status(self, loop_index, cadence, num_frames, is_start,
                          warped_old=None, corrected=None, fresh=None,
                          flow=None, flow_factor=1.0):
        is_gen, _, _ = cadence_math(loop_index, cadence, num_frames)
        needed = []
        if corrected is None:
            needed.append("corrected")
        if is_gen and fresh is None:
            needed.append("fresh")
        if not is_gen and warped_old is None:
            needed.append("warped_old")
        return needed

    def run(self, loop_index, cadence, num_frames, is_start,
            warped_old=None, corrected=None, fresh=None,
            flow=None, flow_factor=1.0):
        is_gen, w, first_span = cadence_math(loop_index, cadence, num_frames)
        frame, next_anchor = cadence_emit(bool(is_start), first_span, is_gen, w,
                                          warped_old, corrected, fresh,
                                          flow=flow, flow_factor=float(flow_factor))
        return (frame, next_anchor, is_gen)


NODE_CLASS_MAPPINGS = {"Cadence| Deforum_Stat-Imp": Cadence}
NODE_DISPLAY_NAME_MAPPINGS = {"Cadence| Deforum_Stat-Imp": "Cadence"}
