# Node reference — ComfyUI Stat-Imp Nodes

Detailed description of every node in this pack. For a quick overview and install
instructions, see the [README](README.md).

---

## `Depth Warp 3D`

Menu group: **Stat-Imp / Deforum / Depth3D**.

An **exact forward port** of original Deforum's `transform_image_3d`
(`deforum-stable-diffusion`, `helpers/animation.py`), reimplemented in **pure
torch** — no pytorch3d, no kornia at runtime. It is verified **pixel-identical**
to the real original on 9 camera axes by an oracle test (run in the development
repo against the genuine Deforum/py3d code; not shipped here because it needs the
original Deforum source to compare against).

> Why it exists: older ComfyUI Deforum 3D nodes mirrored every in-plane axis
> (tx, ty, rx, ry, rz). This node reproduces the original's motion exactly.

### Inputs
- `image` (IMAGE) — frame to warp.
- `depth` (IMAGE) — a depth map for that frame (see **Depth** below).
- `translation_x` / `translation_y` / `translation_z` — camera translation.
- `rotation_3d_x` / `rotation_3d_y` / `rotation_3d_z` — camera rotation (degrees).
- `fov` — field of view (degrees).
- `translation_scale` — global translation gain.
- `depth_invert` — True for MiDaS-style inverse depth (default True).
- `depth_equalize` — histogram-equalize the depth before use.
- `padding_mode` / `sampling_mode` — grid-sample edge / interpolation behaviour.

### Outputs
- `image` (IMAGE) — the warped frame.
- `reveal_mask` (MASK) — *optional.* Marks newly-revealed (disoccluded) pixels
  that have no source data. Wire into `Set Latent Noise Mask` to inpaint only
  those regions. Safe to leave unconnected.

### Motion directions (positive value)
| Axis | On screen |
|---|---|
| `translation_x` | content moves **right** |
| `translation_y` | content moves **up** |
| `translation_z` | **zoom in** / approach |
| `rotation_3d_x` | tilt **up** |
| `rotation_3d_y` | pan **right** |
| `rotation_3d_z` | roll **clockwise** |

### Depth is external (bring your own)

This node does **not** estimate depth — it consumes a depth-map image from any
upstream estimator. Recommended:
[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux):
- `MiDaS-DepthMapPreprocessor` (auto-downloads `Intel/dpt-hybrid-midas`), or
- a **Depth Anything V2** preprocessor (recommended — generally better than the
  MiDaS + AdaBins pipeline the 2023 original used).

To switch estimators at runtime, wire two preprocessors into an `AnySwitch` node
and select the active input.

> **Fidelity note:** the *camera geometry* matches original Deforum exactly. The
> original blended MiDaS **+ AdaBins** for depth; this pack leaves depth to
> external nodes, so depth *values* differ from the 2023 original. Geometry is
> exact; the depth source is your choice.

---

## `Color Match LAB`

Menu group: **Stat-Imp / Deforum / Color**.

Per-frame **histogram color match** to a reference image, in LAB / HSV / RGB.
Unlike mean/std matchers, it remaps the *full* per-channel distribution, so it
holds the whole palette instead of just the average — killing the slow color
drift (the neon "acid" degeneration) of low-denoise img2img feedback loops.
Pure torch, **stateless**.

### Inputs
- `image` (IMAGE) — frame to recolor.
- `reference` (IMAGE) — palette source (e.g. the frame-0 init image for a locked palette, or any image).
- `color_space` — `LAB` (default; match color, keep detail natural), `HSV`, or `RGB`.
- `strength` (0–1) — how strongly to pull toward the reference palette (0 = off, 1 = full match). Schedulable.
- `preserve_luminance` — keep the frame's own brightness; match color only (LAB/HSV).

### Outputs
- `image` (IMAGE) — the recolored frame.

> It is a color **stabilizer**, not an injector: matching to a fixed reference
> *holds* a palette but cannot *create* new prompt colors. For an evolving
> palette, lower `strength` or use `Rolling Color Match` below.

---

## `Rolling Color Match (EMA)`

Menu group: **Stat-Imp / Deforum / Color**.

Same histogram match, but the reference is a **moving anchor** — an exponential
moving average of past frames — instead of a fixed image. This lets the palette
**drift with the animation** while staying coherent (no frame-0 jail, no acid).
Stateless: the anchor is carried through the loop, not stored in the node.

### Inputs
- `image` (IMAGE) — frame to recolor.
- `anchor_in` (IMAGE) — current anchor; thread from the loop's carry slot (`ForLoopOpen` value), bootstrapped at frame 0 with a reference image.
- `alpha` (0–1) — anchor update speed: `0` = frozen anchor (≡ `Color Match LAB`), `1` = follow the previous frame, middle = exponential moving average. Schedulable.
- `color_space` / `strength` / `preserve_luminance` — as in `Color Match LAB`.

### Outputs
- `matched` (IMAGE) — the recolored frame.
- `anchor_out` (IMAGE) — updated anchor; wire to `ForLoopClose` to carry it to the next frame.

### `alpha` × `strength`
`strength` = how much recolor (0 = none, 1 = full). `alpha` = what to match toward
(0 = fixed origin, 1 = previous frame, middle = fading memory). They are
independent; at `strength 0` the node is a pass-through.

**Practical guidance:**
- To **kill acid while keeping the original palette**, use a **very low `alpha`**
  (≈0.02–0.1). Higher `alpha` lets the anchor chase recent frames, so a dark/dull
  source degenerates quickly. `alpha` of one frame is a fast memory — small is strong.
- Schedule **`strength`** (e.g. ramp 0→0.7 over the first frames) so early frames
  aren't slammed to the anchor before the scene forms.
- A **poor/dark source palette** can't be enriched by matching alone — feed a
  richer image as the bootstrap/reference and the whole loop inherits its colors.
- `preserve_luminance` keeps brightness, so this node controls **color only**; it
  will not stop a loop that is darkening on its own (that's a denoise/luminance issue).
