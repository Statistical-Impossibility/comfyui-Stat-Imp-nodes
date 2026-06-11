# ComfyUI Stat-Imp Nodes

Clean, MIT-licensed nodes for Deforum-style animation loops in ComfyUI.
This release ships three nodes: a **depth-aware 3D camera warp** and two
**color-coherence** nodes (a fixed-anchor matcher and a rolling/EMA matcher)
that stop the feedback loop's color from drifting into neon "acid".

Pure torch — no extra Python dependencies. See **[NODES.md](NODES.md)** for the
full per-node reference (inputs, outputs, motion directions, tuning guidance).

## Install

**Manual / git:** clone into your ComfyUI custom-nodes folder, then restart ComfyUI
and hard-refresh the browser:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Statistical-Impossibility/comfyui-Stat-Imp-nodes
```

The nodes appear under the **Stat-Imp / Deforum** menu group.


## How to use

These nodes are building blocks for a Deforum-style 3D animation loop. A ready-to-run
example workflow is included:

➡️ **[workflows/deforum_SDXL_Distilled.json](workflows/deforum_SDXL_Distilled.json)** —
a full depth-warp + color-coherence animation loop for **SDXL Turbo / Lightning / Hyper**.

Open it in ComfyUI, pick a distilled SDXL checkpoint, and queue. The embedded note
explains the inputs and settings.

## Credits

This pack stands on the original **Deforum** work — full credit to its authors:

- **[Deforum (Stable Diffusion)](https://github.com/deforum/deforum-stable-diffusion)**
  — the original animation engine. The camera math here is a forward port of its
  `transform_image_3d`, and the color matcher is a clean-room reimplementation of
  its `maintain_colors` (`helpers/colors.py`).
- **[Deforum-Comfy-Nodes](https://github.com/deforum/deforum-comfy-nodes)**
  — the main ComfyUI Deforum harness and node pack (ForLoop, schedules, Set/Get bus).
