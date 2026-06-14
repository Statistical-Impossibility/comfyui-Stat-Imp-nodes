# ComfyUI Stat-Imp Nodes

Clean, MIT-licensed nodes for Deforum-style animation loops in ComfyUI.
This release ships following nodes:

- a **depth-aware 3D camera warp** (exact forward port of Deforum's `transform_image_3d`),
- two **color-coherence** nodes — a fixed-anchor matcher and a rolling/EMA matcher —
  that stop the feedback loop's color from drifting into neon "acid",
- a **cadence** node (diffuse every Nth frame, dissolve the in-betweens — vanilla Deforum semantics), and
- an **optical-flow** estimator that feeds the cadence node for motion-aware in-betweens.

The core warp + color nodes are **pure torch** — no extra Python dependencies. The optical-flow
node additionally uses **OpenCV** (DIS / Farneback backends, usually already bundled with ComfyUI)
and, only for its optional RAFT backend, **torchvision**. See **[NODES.md](NODES.md)** for the full
per-node reference (inputs, outputs, motion directions, tuning guidance).

## Requirements

1. **A working ComfyUI install.** If you don't have one yet, get it from the official site /
   repo first: **https://github.com/comfy-org/comfyui** (follow its install guide).
2. **The Deforum harness — our fork.** The example workflow runs inside a Deforum
   `ForLoop` animation loop. It needs the harness nodes (ForLoop, ValueSchedule, Set/Get bus),
   and specifically **our fork** of `deforum-comfy-nodes`, which widens the loop's carry state
   (10 `value` sockets instead of 1) and fixes the value-schedule formula parser so `sin/cos`
   and multi-argument functions work. Install the fork:
   **https://github.com/Statistical-Impossibility/deforum-comfy-nodes**
   (See *"Why a fork"* below.) The five nodes in *this* pack work standalone, but the shipped
   example workflow assumes the fork.

## Install

Clone into your ComfyUI custom-nodes folder, then restart ComfyUI and hard-refresh the browser:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Statistical-Impossibility/comfyui-Stat-Imp-nodes
# and the harness fork the example workflow needs:
git clone https://github.com/Statistical-Impossibility/deforum-comfy-nodes
```

The nodes appear under the **Stat-Imp / Deforum** menu group.

## How to use

These nodes are building blocks for a Deforum-style 3D animation loop. A ready-to-run
example workflow is included:

➡️ **[workflows/deforum_SDXL_Distilled.json](workflows/deforum_SDXL_Distilled.json)** —
a full depth-warp + color-coherence animation loop for **SDXL Turbo / Lightning / Hyper**.

Open it in ComfyUI, pick a distilled SDXL checkpoint, and queue. The embedded note
explains the inputs and settings.

## Why a fork (of deforum-comfy-nodes)

The upstream [deforum-comfy-nodes](https://github.com/deforum/deforum-comfy-nodes) harness is great
but two things blocked this pack:

1. **Loop carry width.** Cadence needs to thread several extra images through the animation loop
   (the old anchor lineage, the captured frame-0, …). Upstream `ForLoop` exposes a single
   carry socket; our fork widens it to **10** (`value1`…`value10`).
2. **Value-schedule formula parser.** Upstream `_sanitize_value` stripped *all* parentheses, so
   `0:(cos(t))` collapsed to `cost` → `NameError`, and a comma inside a value (`pow(t,2)`) was
   misread as a keyframe separator. Our fork keeps inner parens and splits keyframes only at
   paren-depth 0, so `sin/cos/abs/pow/min/max/...` work in value schedules.

Both are clean, additive changes; the fork stays drop-in compatible with upstream workflows.

## Credits

This pack stands on the original **Deforum** work — full credit to its authors:

- **[Deforum (Stable Diffusion)](https://github.com/deforum/deforum-stable-diffusion)**
  — the original animation engine. The camera math here is a forward port of its
  `transform_image_3d`, and the color matcher is a clean-room reimplementation of
  its `maintain_colors` (`helpers/colors.py`).
- **[Deforum-Comfy-Nodes](https://github.com/deforum/deforum-comfy-nodes)**
  — the main ComfyUI Deforum harness and node pack (ForLoop, schedules, Set/Get bus);
  our fork above extends it.
