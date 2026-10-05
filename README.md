# ComfyUI_Meridian_for_Gaussian_Splatting

**Depth Anything v3 replaces VGGT.** The geometry node reconstructs the depth of a single photo
*in-process* with **Depth Anything v3** (fast depth) and reprojects the camera path itself — there
is **no VGGT checkout, no `vggt_omega` checkpoint and no separate python environment** any more.

This Repo utilizes the Meridian project and Depth Anything v3 to create custom camera paths for
static images. Minimax H3 is used to create plausible completions of missing parts of the picture.
Afterwards GLOMAP/COLMAP is used to prepare a dataset for Gaussian Splatting. A custom node is able
to automatically call Lichtfeld for the splatting process.

---

## What the flow does

```
 Load & Resize Image   Meridian Parameters and Camera    Meridian Geometry             MiniMax H3 (core)
 ┌──────────────────┐  ┌───────────────────────────────┐ ┌───────────────────────┐  ┌──────────────────────────┐
 │ one still photo  │─►│ geometry args + camera path   │─►│ Depth Anything v3:    │─►│ re-camera pass with the  │
 │ (+ background)   │  │ manual styles or automatic    │  │ depth + reprojection  │  │ Meridian LoRAs           │
 └──────────────────┘  │ pivot estimate + path         │  │ (pseudo-views)        │  └──────────────────────────┘
                       └───────────────────────────────┘ └───────────────────────┘
                                             GLOMAP Lichtfeld Tracker ──► Lichtfeld Headless Trainer
                                             (dataset + COLMAP model)      (Gaussian-splat training)
```

1. **One photo** (optionally background-removed) is the only input.
2. The **Meridian Parameters and Camera** node builds the arguments *and* the camera path: either
   hand-authored styles (O orbits at named stations, an alternating-height pendulum, a spiral
   sweep) or an **automatic estimate** that locates the subject's (or the whole scene's) geometric
   pivot from the still's depth profile and flies a speed-capped, collision-guarded path around it.
   `Auto Orbit View Angle` picks *which side* you stand on, `Auto Orbit Coverage` picks *how much*
   you see (`Front only` or `Front and Back`: front circle → shortest level connection → the far
   orbit's full loop 9 → 12 → 3 → 6 → 8 o'clock).
3. The **Meridian Geometry** node runs **Depth Anything v3** on the still, unprojects the point
   cloud and re-renders that path into *pseudo-views* (synthetic reprojections, not observed
   geometry). In-process — no VGGT, no external environment, no `sample.py`.
4. **MiniMax H3** with the Meridian teacher + turbo LoRAs turns the rough reprojection into a
   plausible, temporally stable clip.
5. Optionally the clip feeds the **GLOMAP Lichtfeld Tracker** and the **Lichtfeld Headless Trainer**
   to produce a Gaussian Splat.

---

## What is in this repository

| Path | Content |
|---|---|
| `nodes/enndee_meridian_parameters.py` | **Meridian Parameters and Camera (Enndee)** — the args string *and* the camera path (manual styles + automatic estimate) |
| `nodes/enndee_meridian_auto_camera.py` | the automatic camera: pivot estimate, 45° front circle, view angle / coverage, speed cap, visibility guarantee |
| `nodes/enndee_meridian_fast_depth.py` | fast-depth backend (Depth Anything v3 / V2) and the in-process reprojection renderer |
| `nodes/enndee_meridian_camera_path.py` | path math: O orbits, alternating-height pendulum, spiral sweep |
| `nodes/enndee_meridian_geometry.py` | **Meridian Geometry (Enndee)** — the fast-depth reprojection node |
| `nodes/glomap_lichtfeld_node.py`, `lichtfeld_training_node.py` | GLOMAP Lichtfeld Tracker, Lichtfeld Headless Trainer |
| `nodes/enndee_image_loader.py`, `enndee_resolution_selector.py`, `enndee_resize_modes.py`, `enndee_standby_signal.py`, `enndee_unique_filenames.py` | the pack's supporting nodes |
| `nodes/meridian_prompt_composer.py` | **Meridian Prompt Composer** — the conditional per-picture prompt blocks |
| `nodes/enndee_bin.py`, `enndee_colmap/` | binary installer (COLMAP/GLOMAP), vendored COLMAP helpers |
| `web/js/*.js` | adaptive widget panels (parameters node, image loader, resolution) |
| `install.py`, `requirements.txt` | ComfyUI-Manager install: COLMAP/GLOMAP binaries on demand + optional python deps |
| `tests/*.py` | the unit-test suite (no model downloads, no GPU) |
| `examples/Meridian_Splatting_1.0.json` | **the reference workflow for the whole pipeline** |
| `setup.bat`, `setup/meridian_setup.py` | one-shot setup for a fresh ComfyUI (see below) |

The node code is MIT licensed. **Meridian, the MiniMax H3 weights and the LoRAs are *not*
redistributed here** — they are separate, licence-gated downloads (see *Requirements*).

---

## Requirements

### 1. ComfyUI

A recent ComfyUI (portable Windows build recommended) with the **MiniMax H3 core nodes**
(`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `MiniMaxH3ImageToVideo` inside
`comfy_extras/nodes_minimax_h3.py`). Update ComfyUI if those nodes are missing.

### 2. Depth Anything v3 (the fast depth backend)

One package, installed **without** dependency resolution — its wheel pins `numpy<2` and would drag
xformers/open3d/pycolmap/moviepy/gsplat/evo into ComfyUI; the backend stubs the two optional
sub-packages itself:

```
python -m pip install --no-deps depth-anything-3
```

`setup.bat` runs this as step 3/7 and the report then says
`Depth Anything v3  installed [ok]`. The weights download into the Hugging Face cache on the first
run (a few hundred MB). The V2 family (`Depth-Anything-V2-*`) stays available as a fallback and
only needs `transformers`.

### 3. Meridian release (licence-gated)

Only the **LoRAs** are needed — the geometry no longer runs `inference/sample.py`:
`meridian_teacher_lora.safetensors` and `meridian_turbo_lora.safetensors` from the MiniMax H3
"Meridian" release (MiniMax H3 Community License) belong into `ComfyUI\models\loras`.

```
setup.bat "D:\ComfyUI_windows_portable" --meridian-zip "X:\path\to\Meridian_release.zip"
```

### 4. MiniMax H3 model files

`models\diffusion_models\H3\minimax_h3_fl2va_pruned_int8_convrot.safetensors`,
`models\text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors`,
`models\vae\minimax_h3_video_vae_fp16.safetensors`,
`models\vae\minimax_h3_audio_vae_fp32.safetensors` (your own H3 download).
Step 5/7 of the setup lists whatever is missing.

### 5. Community node packs (optional)

The example workflow uses **VideoHelperSuite**, **ComfyUI-Easy-Use** and **comfyui-various** —
`setup.bat` offers to clone them (or pass `--install-community-nodes`).

---

## Setup (the tutorial)

### 1. Get the code

Clone into `ComfyUI\custom_nodes\`, or just run the setup from anywhere:

```
git clone https://github.com/Enndee/ComfyUI_Meridian_for_Gaussian_Splatting.git
ComfyUI_Meridian_for_Gaussian_Splatting\setup.bat
```

`setup.bat` asks for the ComfyUI folder (the one that contains `ComfyUI\` and `python_embeded\`,
or its parent), offers the community packs, and calls `setup\meridian_setup.py`.

### 2. What the setup does (7 steps, idempotent)

| Step | What it does |
|---|---|
| 1/7 | copies this repository into `ComfyUI\custom_nodes\` |
| 2/7 | checks `av` / `numpy` / `torch` in ComfyUI's python |
| 3/7 | installs **Depth Anything v3** (`pip install --no-deps depth-anything-3`) |
| 4/7 | unpacks `--meridian-zip` and copies the Meridian LoRAs into `models\loras` |
| 5/7 | reports the H3 model files the example workflow needs |
| 6/7 | copies the example workflows to `ComfyUI\user\default\workflows` and rewrites machine-specific `...\Tools\...` paths |
| 7/7 | (with `--install-community-nodes`) clones VHS / easy-use / various |

Useful flags (forwarded by `setup.bat` after the folder argument):

```
setup.bat "D:\ComfyUI" --check                          # read-only report, change nothing
setup.bat "D:\ComfyUI" --dry-run                        # print every command, run none
setup.bat "D:\ComfyUI" --meridian-zip "X:\Meridian.zip" # unpack the release + copy its LoRAs
setup.bat "D:\ComfyUI" --install-community-nodes        # clone the public example packs
setup.bat "D:\ComfyUI" --no-workflow                    # do not copy the example workflows
```

### 3. Restart and go

Restart ComfyUI, open **`Meridian_Splatting_1.0`** (Workflow menu) and queue it. The first run
downloads the DA3 weights into the Hugging Face cache; everything else is user-provided (models
above). The node defaults in that example are also the *node defaults*:
**158 frames**, `camera_mode = Automatic`, `auto_max_speed 12`, `auto_subject_fill 40`,
`model_size = Depth-Anything-3-Mono-Large`, geometry `canvas_mode = custom 832x480`,
`depth_res = 1920`, `edge_threshold 0.1`, `back_face_cull` on.

---

## Nodes

### Meridian Parameters and Camera (Enndee) — `Enndee_MeridianParametersAndCamera`

One node for the geometry arguments **and** the camera path. `Camera Mode` picks the source:
`Manual` flies the path widgets (O orbits at named stations / alternating-height pendulum / spiral
sweep) around the absolute look pivot; `Automatic` estimates the pivot from the connected reference
image and emits the estimated path (the `args` output and the `custom_camera` signal stay in sync
with the Geometry node).

The interesting automatic-mode widgets (defaults = the `Meridian_Splatting_1.0` values):

| Widget | Default | Meaning |
|---|---|---|
| `output_frames` | `158` | path length; Meridian ships prompt assets for 73/90/107/124/141/158/175/243 |
| `camera_mode` | `Automatic` | `Manual` / `Automatic` |
| `auto_target` | `subject` | orbit the subject (mask or near depth layer) or the whole scene |
| `auto_max_speed` | `12` | % of the content radius the camera may travel per frame — the hard cap |
| `auto_subject_fill` | `40` | fill of the subject in the picture (bigger = closer) |
| `auto_orbit_view_angle` | `0` | azimuth the front circle is centred on: 0 frontal, +90 viewer-left, 180 back, 270 viewer-right |
| `auto_orbit_coverage` | `Front and Back` | `Front only` = the circle alone; `Front and Back` = circle + level connection + the far orbit's loop (9 → 12 → 3 → 6 → 8 o'clock) |
| `auto_orbit_direction` | `counter-clockwise` | mirrors the whole path |
| `model_size` | `Depth-Anything-3-Mono-Large` | depth model of the estimate — must match the Geometry node, or the keys are in the wrong units |
| `auto_pivot_x/y/z` | `0` | final aim offset in content radii (applied last, cannot be cancelled by the solves) |

The estimate is **speed-capped and visibility-guaranteed**: it never exceeds `auto_max_speed`, the
back part gives way first when frames cannot pay for it, and the whole subject box stays inside
every frame (the console prints the achieved swing, drift and clearance instead of hiding it).
`auto_orbit_distance`, `auto_orbit_size` and `auto_orbit_end` are **deprecated** — they stay in the
graph so old workflows keep loading, but they are hidden and ignored.

### Meridian Geometry (Enndee) — `Enndee_MeridianGeometry`

Fast-depth reprojection node: it reads the still (or an IMAGE batch), runs the depth model, builds
the point cloud and renders the camera path from it — all inside ComfyUI's python.

| Widget | Default | Meaning |
|---|---|---|
| `video` | `clip.mp4` | source video path; ignored when an IMAGE is connected (frame 0 repeats) |
| `args` | *(empty)* | extra Meridian flags; `custom_camera` wins over frame-count/motion flags |
| `model_size` | `Depth-Anything-3-Mono-Large` | render depth model (V2 trio is the fallback) |
| `canvas_mode` | `custom` (`832x480`) | `auto_meridian480` = the trained canvas ladder, `custom` = the two fields below |
| `depth_res` | `1920` | working-resolution cap on the still's longest side; also DA3's `process_res` (`0` = keep the still's own resolution) |
| `edge_cull` / `edge_threshold` | `on` / `0.1` | drop points on steep depth edges so silhouettes cannot smear into spikes |
| `back_face_cull` | `on` | Meridian's `--cull`: splats the target camera sees from behind are dropped (no mirrored front) |
| `cloud_scale` / `point_size` | `2` / `1` | unprojection-grid upscale / point footprint |
| `image` (optional) | — | the still or frame batch to reproject |
| `custom_camera` (optional) | — | `custom_camera` from the Parameters node; its frame count sets the flight length |

### The rest of the pack

**GLOMAP Lichtfeld Tracker (Enndee)** turns a frame sequence into a ready-to-train Lichtfeld Studio
dataset (global SfM with the vendored COLMAP/GLOMAP builds that `install.py` downloads);
**Lichtfeld Headless Trainer (Enndee)** runs the training and exports `.ply`/`.sog`/`.spz`;
**Load & Resize Image** / **Resolution Selector** / **Standby On Signal** are the workflow
helpers. Their widget
tables live in the development README
([Enndee/Enndees_Nodepack](https://github.com/Enndee/Enndees_Nodepack)).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'depth_anything_3'` | `python -m pip install --no-deps depth-anything-3` (step 3/7 of the setup does it) |
| DA3 weights not found / first run slow | they download into the Hugging Face cache on first use; re-run and they are local afterwards |
| Red nodes "Meridian Parameters / Geometry missing" | install this pack (`setup.bat`), restart ComfyUI, and install the community packs with `--install-community-nodes` |
| The estimate's depth model differs from the render | set `model_size` on **both** nodes to the same value (default: `Depth-Anything-3-Mono-Large`) — the emitted keys are in the estimating model's median units, a mismatch puts the pivot at the wrong depth (the Geometry node warns) |
| "the max camera speed ends the path at … deg" | that is the speed cap working: more Output Frames, a higher `auto_max_speed`, or `Auto Orbit Coverage = Front only` buy the back visit back |
| Pseudo-views show holes where the camera sees behind the subject | `back_face_cull` is on by design (no mirrored front); a *tight* `depth_res` and `edge_threshold 0.1` keep silhouettes clean |
| OOM during the H3 pass | lower the resolution / super-resolution factor or free VRAM (the example uses three `easy cleanGpuUsed` nodes) |
| `sample.py --help`-style Meridian setup errors | not needed any more: nothing runs `inference/sample.py` — only the LoRAs are read from the Meridian release |

---

## Development

The full unit-test suite ships with the package (no GPU, no model downloads, about 20 seconds):

```
python -m unittest discover -s tests -p "test_*.py"
```

The suite covers the camera estimator (path shapes, speed cap, visibility guarantee, view angle /
coverage), the geometry modes, the fast-depth backend, the path math and the workflow widgets.

---

## Licences & credits

- Code in this repository: **MIT** (see `LICENSE`).
- The Meridian release (MiniMax H3 re-camera model, LoRAs, assets): MiniMax H3 Community License —
  not redistributed here.
- Depth Anything v3 and the Depth-Anything family: **Apache-2.0**, distributed via
  PyPI/Hugging Face — installed on demand, never vendored.
- ComfyUI and its MiniMax H3 core nodes: ComfyUI / MiniMax.
- COLMAP 3.11 / GLOMAP 1.2 binaries: their own licences, downloaded by `install.py`.
- The community packs referenced above keep their own licences.

