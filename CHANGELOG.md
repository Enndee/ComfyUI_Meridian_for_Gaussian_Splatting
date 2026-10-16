# Changelog

## v1.3.0 — Meridian_Splatting_1.4 becomes the reference and the defaults (2026-10-16)

**The node defaults now match `examples/Meridian_Splatting_1.4.json`** — drop a fresh node and it
starts where the reference workflow is tuned.

* **New reference workflow** `examples/Meridian_Splatting_1.4.json`: 175 frames, banded-orbit
  automatic camera (2 bands x 120 deg, -10 -> 45 deg, start yaw 60), SAM3 face orbit
  (`Auto (SAM3)`, face fill 10), sharp-frame selection on, subject-only alignment, JPEG dataset
  export (quality 97) and the MiniMax H3 int8/nvfp4 model set.
* **Node defaults updated from the workflow** (in the Enndees Nodepack, single source of truth):
  `output_frames 175`, `auto_subject_fill 30`, `auto_orbit_coverage Banded Orbits` plus the five
  band widgets, `face_orbit Auto (SAM3)` / `face_fill 10`; Geometry `canvas_mode auto_meridian480`,
  custom canvas `1920x1088`, `cloud_scale 1`, `edge_threshold 0.05`, `moge_refine_steps 10`;
  GLOMAP tracker `matcher exhaustive`, `frame_step 1`, `offset_splat 0`, `image_format JPEG`
  (`jpeg_quality 97`), face-detail masks **on** (`dilate 5`), `subject_focus subject only`,
  `sharp_selection` **on** (`sharp_num_frames 1`); Load & Resize Image `resize` **on**
  (`scale total pixels`, `1000x1000`, `divisible_by 32`, `scale_method area`, `no_upscale` **on**);
  the Meridian Prompt Composer ships the 1.4 prompt texts; the Lichtfeld trainer's tuned
  iterations/strategy/loss schedule. Machine-specific paths (the Studio exe) and the trainer's
  `eval_steps` were deliberately **not** copied - they are not sane defaults.
* **setup step 7 additionally clones `comfyui-rmbg` (RMBG + SAM3Segment) and
  `veda-sparse-attention`** - the packs the 1.4 workflow needs beyond the 1.1 list. Lichtfeld
  Studio stays user-provided and is never auto-installed.
* **The model report follows the 1.4 workflow** (qwen3vl nvfp4_awq clip, int8_convrot video VAE,
  the H3 Turbo LoRA) and lists the pack-shipped weights (latent upscaler, Veda).
* README: `Meridian_Splatting_1.4` is the reference workflow, new default tables, and a corrected
  pack table - the `RMBG` / `SAM3Segment` nodes come from ComfyUI-RMBG (BRIA stays for the older
  examples).

## v1.2.0 — the nodes move to the Enndees Nodepack (2026-10-05)

**One place owns the node code.** This repository is now the installer, the example workflows and
the documentation.

* **Removed the duplicated node code** (`nodes/`, `web/`, `tests/`, `enndee_colmap/`,
  `enndee_bin.py`, `install.py`, `requirements.txt`). The nodes are the **Enndees Nodepack**:
  <https://github.com/Enndee/Enndees_Nodepack>. `__init__.py` here deliberately registers nothing,
  so an existing `custom_nodes` clone of this repository stays harmless.
* **Setup step 1** clones (or `git pull`s) the Enndees Nodepack into `custom_nodes` instead of
  copying this repository.
* **Setup step 7 installs every community pack the example workflow needs** — now by default:
  Sharp-Selector, rgthree, KJNodes, various, custom-scripts, VHS, easy-use, Pixaroma, BRIA RMBG,
  RTX nodes, UniBlockSwap, Memory-Cleanup, H3 MotionCache, H3 Turbo, H3 latent upscaler.
  `--skip-community-nodes` turns it off; `--install-community-nodes` is deprecated (it is the
  default now).
* **New example:** `examples/Meridian_Splatting_1.1.json` — the current reference workflow
  (automatic camera with the new **O Orbit Angle**, 158 frames, sharp-frame selection).
  `Meridian_Splatting_1.0.json` is kept for comparison.
* **Nodes that ship with the pack for this workflow:** the `auto_orbit_angle` ("O Orbit Angle")
  widget on *Meridian Parameters and Camera*, the `Enndee_SharpFrameSelector` /
  `Enndee_SharpnessAnalyzer` pair and `MeridianPromptComposer`.
* README rewritten for the split (nodes → pack, installer → here).

## v1.1.0 — Depth Anything v3 replaces VGGT (2026-10-03)

* The geometry node reconstructs depth **in-process** with Depth Anything v3 (fast depth) and
  reprojects the camera path itself: no VGGT checkout, no `vggt_omega` checkpoint, no separate
  `meridian-vggt` environment.
* `setup.bat` installs `depth-anything-3` (`--no-deps`) into ComfyUI's own python, unpacks the
  Meridian release LoRAs and patches the example workflow's machine-specific paths.
* Meridian auto camera rework: geometric pivot, constant orbit distance, speed fit, collision
  guard, view angle / coverage (`Front only` / `Front and Back`).

## v1.0.0 — ComfyUI Meridian install package (2026-09-26)

* First packaged release: one-shot setup (VGGT era), example workflow, node code in this
  repository.
