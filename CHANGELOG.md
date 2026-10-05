# Changelog

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
