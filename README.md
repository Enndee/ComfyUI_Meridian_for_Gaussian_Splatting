# ComfyUI_Meridian_for_Gaussian_Splatting

This Repo utilizes the Meridian project and VGGT to create custom camera paths for static images. Minimax H3 is used to create plausible completions of missing parts of the picture. Afterwards GLOMAP/COLMAP is used to prepare a dataset for Gaussian Splatting. A custom node is able to automatically call Lichtfeld for the splatting process.

---

## What the flow does

```
 Load & Resize Image            Meridian Parameter Picker            Meridian Geometry            MiniMax H3 (core nodes)
 ┌──────────────────┐   args   ┌───────────────────────────┐  path   ┌──────────────────┐  render  ┌──────────────────────┐
 │ one still photo  │ ───────► │ geometry args + a custom  │ ──────► │ VGGT reconstruction│ ──────► │ re-camera pass with  │
 │ (+ background)   │          │ camera path (3 styles)    │         │ + pseudo-views     │         │ the Meridian LoRAs   │
 └──────────────────┘          └───────────────────────────┘         └──────────────────┘         └──────────────────────┘
                                                                                                              │
                                                            GLOMAP Lichtfeld Tracker ──► Lichtfeld Headless Trainer
                                                            (dataset + COLMAP model)     (Gaussian-splat training)
```

1. **One photo** (optionally background-removed) is the only input.
2. The **Meridian Geometry** node runs Meta's VGGT-Omega once, then renders the requested camera path from that single depth reconstruction as *pseudo-views* (they are synthetic, not observed geometry).
3. The **Meridian Parameter Picker** designs that path: O-orbit loops at named stations, an alternating-height pendulum, or a monotone spiral sweep - including zoom (path dolly), look pivot and output length, all with context-aware widgets.
4. **MiniMax H3** with the Meridian teacher + turbo LoRAs turns the rough re-projection into a plausible, temporally stable clip.
5. Optionally the clip feeds the **GLOMAP Lichtfeld Tracker** and the **Lichtfeld Headless Trainer** to produce a Gaussian Splat.

---

## What is in this repository

| Path | Content |
|---|---|
| `nodes/enndee_meridian_parameter_picker.py` | Unified arg + camera-path builder (three path styles, adaptive widgets) |
| `nodes/enndee_meridian_camera_path.py` | Path math: O orbits, alternating-height pendulum, spiral sweep |
| `nodes/enndee_meridian_geometry.py` | Runs Meridian `inference/sample.py` from ComfyUI, repeats the first frame for the path length |
| `examples/meridian_customcampath2.json` | The reference workflow for the whole pipeline |
| `setup.bat`, `setup/meridian_setup.py` | One-shot setup for a fresh ComfyUI (see below) |

The node code is MIT licensed. **Meridian, VGGT-Omega and the H3 weights are *not* redistributed here** - they are separate, licence-gated downloads (see *Requirements*). The adaptive widget extension of the picker and the unit-test suite live in the development checkout ([Enndee/Enndees_Nodepack](https://github.com/Enndee/Enndees_Nodepack)); this install package ships the nodes and the setup only.

---

## Requirements

### 1. ComfyUI

A recent ComfyUI (portable Windows build recommended) with the **MiniMax H3 core nodes** (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `MiniMaxH3ImageToVideo` inside `comfy_extras/nodes_minimax_h3.py`). Update ComfyUI if those nodes are missing.

### 2. Meridian release (licence-gated)

The `recam`/`inference` code plus the Meridian LoRAs and frozen-embedding assets come from the MiniMax H3 "Meridian" release (MiniMax H3 Community License). Download it yourself and place it so the folder contains:

```
<tools>\Meridian\
├── inference\sample.py            <- executed by Meridian Geometry
├── recam\                         <- runtime library
├── assets\                        <- fixed_embed_*.pt / silence_audio_*.pt
├── LICENSE / LICENSE-CODE / MODIFICATIONS.md
```

The release also ships a small ComfyUI node file (`MeridianFrozenPrompt`) - install it into `custom_nodes` (or keep the release's own instructions).

### 3. VGGT-Omega + fp16 fork + checkpoint

| Piece | Where it goes | Source |
|---|---|---|
| fp16 fork (installable package) | `<tools>\vggt-omega-fp16-version` | `https://github.com/venlyrina/vggt-omega-fp16-version.git` |
| Meta's upstream checkout (checkpoint folder) | `<tools>\vggt-omega` | `https://github.com/facebookresearch/vggt-omega.git` |
| `vggt_omega_1b_512.pt` (~4.6 GB) | `<tools>\vggt-omega\checkpoints\` | Hugging Face `wincentIsMe/VGGT-Omega` |

`<tools>` is the folder next to the portable ComfyUI root (`...\ComfyUI_windows_portable\Tools` on the reference machine). With that exact layout the Meridian Geometry node finds both VGGT paths **automatically** - no `--vggt-repo`/`--vggt` needed.

### 4. Meridian python environment

An isolated environment (conda env `meridian-vggt`, Python 3.12, **torch 2.12.1 + cu130**, plus `numpy<2`, `Pillow`, `einops`, `safetensors`, `opencv-python`, `av==16.1.0`, `transformers>=4.57`, `peft==0.18.0`, `diffusers@d6726f3`, `huggingface_hub`). `setup.bat` creates it for you; the exact pip recipe lives in `setup/meridian_setup.py` and mirrors the validated `Tools\Meridian\setup_geometry_environment.py` of the reference machine.

### 5. The rest of the Enndees pack (optional)

The example workflow also uses the **GLOMAP Lichtfeld Tracker**, the **Lichtfeld Headless Trainer** and the **Load & Resize Image** node from [Enndee/Enndees_Nodepack](https://github.com/Enndee/Enndees_Nodepack) (that pack also bundles COLMAP/GLOMAP downloads). Install it additionally if you want the dataset/training half of the flow.

### 6. Community node packs used by the example workflow

| Node(s) | Pack | Install |
|---|---|---|
| `VHS_VideoCombine` | VideoHelperSuite | `git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite` |
| `easy cleanGpuUsed` | ComfyUI-Easy-Use | `git clone https://github.com/yamatazen/ComfyUI-Easy-Use` |
| `JWDatetimeString`, `JWStringConcat` | comfyui-various | `git clone https://github.com/jamesWalker55/comfyui-various` |
| `RTXVideoSuperResolution` | comfyui_nvidia_rtx_nodes | via ComfyUI-Manager |
| `PixaromaGroupSwitch` | ComfyUI-Pixaroma | via ComfyUI-Manager |

`setup.bat --install-community-nodes` clones the first three automatically.

### 7. Models (user-provided)

| File | Folder |
|---|---|
| `H3\minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models\diffusion_models` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models\text_encoders` |
| `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` | `models\vae` |
| `meridian_teacher_lora.safetensors`, `meridian_turbo_lora.safetensors` | `models\loras` (from the Meridian release) |

---

## Installing on a fresh ComfyUI

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/Enndee/ComfyUI_Meridian_for_Gaussian_Splatting
cd ComfyUI_Meridian_for_Gaussian_Splatting
setup.bat
```

`setup.bat` uses the portable python of the surrounding ComfyUI install and runs `setup/meridian_setup.py`, which:

1. installs this repository into `custom_nodes` when run from a standalone clone,
2. checks that ComfyUI's python can import `av`, `numpy` and `torch` (they ship with ComfyUI - this repository adds no pip packages),
3. creates the isolated Meridian environment at `<tools>\.conda\meridian-vggt` (conda when available, venv otherwise) and installs torch cu130 + the Meridian requirements,
4. clones the VGGT fp16 fork and Meta's checkout when missing, and downloads `vggt_omega_1b_512.pt` from Hugging Face when the checkpoint is absent,
5. verifies the install by running `inference/sample.py --help` inside that environment,
6. copies `examples/meridian_customcampath2.json` into `ComfyUI\user\default\workflows` and rewrites every machine-specific path (tools folder, Meridian repo, Meridian python, VGGT paths) to your machine, then lists the values that remain yours (LichtFeld-Studio path, input image).

Everything is idempotent - run it again after a partial install or an update.

Useful flags (pass through `setup.bat`):

| Flag | Effect |
|---|---|
| `--check` | read-only report of every location; changes nothing |
| `--dry-run` | print all commands without running them |
| `--comfy-root PATH` | portable root when auto-detection fails |
| `--tools-dir PATH` | where Meridian/VGGT live (default `<root>\..\Tools`) |
| `--meridian-dir PATH` | your Meridian release folder (if not `<tools>\Meridian`) |
| `--python PATH` | use an existing Meridian environment |
| `--workflows-dir PATH` | e.g. a OneDrive user folder instead of the ComfyUI default |
| `--torch-index URL` | another CUDA wheel index (default cu130) |
| `--skip-env` | keep the python environment untouched |
| `--no-workflow` | do not touch the example workflow |
| `--install-community-nodes` | also clone VideoHelperSuite, Easy-Use and comfyui-various |

### Manual equivalent (if you prefer doing it yourself)

```bat
git clone https://github.com/venlyrina/vggt-omega-fp16-version.git "%TOOLS%\vggt-omega-fp16-version"
git clone https://github.com/facebookresearch/vggt-omega.git "%TOOLS%\vggt-omega"
conda create -p "%TOOLS%\.conda\meridian-vggt" python=3.12 -y
"%TOOLS%\.conda\meridian-vggt\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
"%TOOLS%\.conda\meridian-vggt\python.exe" -m pip install "numpy<2" Pillow einops safetensors opencv-python av==16.1.0 "transformers>=4.57" peft==0.18.0 huggingface_hub "git+https://github.com/huggingface/diffusers@d6726f3"
"%TOOLS%\.conda\meridian-vggt\python.exe" -m pip install --no-deps -e "%TOOLS%\vggt-omega-fp16-version"
"%TOOLS%\.conda\meridian-vggt\python.exe" "%TOOLS%\Meridian\inference\sample.py" --help
```

---

## The example workflow

`examples/meridian_customcampath2.json` is the full pipeline of the diagram above: one image -> parameter picker -> Meridian Geometry (`--cull --dolly 1.5`) -> dual H3 LoRA pass -> RTX super resolution -> VHS output, plus the GLOMAP tracker and the Lichtfeld trainer on a bypassable group.

`setup.bat` patches these values to your machine:

| Node | Widget | Patched value |
|---|---|---|
| `Enndee_MeridianGeometry` | `repo` | `<tools>\Meridian` |
| `Enndee_MeridianGeometry` | `python` | `<tools>\.conda\meridian-vggt\python.exe` |
| `Enndee_MeridianParameterPicker` | `vggt_repo` / `vggt_checkpoint` | `<tools>\vggt-omega-fp16-version` / `<tools>\vggt-omega\checkpoints\vggt_omega_1b_512.pt` |
| `MeridianFrozenPrompt` | assets folder | `<tools>\Meridian\assets` |

Still yours to set: the input image (`Load & Resize Image`), the LichtFeld-Studio executable in the trainer, and the model files from section 7.

---

## Nodes

### Meridian Parameter Picker (Enndee) - `Enndee_MeridianParameterPicker`

Builds the Meridian `sample.py` arguments and, with **Use Custom Camera** enabled, a camera path. The widget list adapts to the configuration (the frontend extension shows only what matters):

| Master | Reveals / hides |
|---|---|
| `use_custom_camera` | shows the path group, hides the freeze/author-move options that Meridian Geometry overrides |
| `path_camera_mode` | `O Orbits` -> station checkboxes + start station + orbit diameter; `Alternating Height` -> yaw pair, low/high arc, switch count, first arc; `Spiral Sweep` -> yaw pair + spiral start/end elevation |
| `freeze_source`, `pivot_enabled`, `aim`, `follow`, `canvas_enabled`, `full_enabled` | their detail widgets |

Path styles:

- **O Orbits** - closed vertical O loops at named stations (Front, Left, Right, Back, Up, Down, LeftBack, RightBack), the 120-degree rear stations included.
- **Alternating Height** - sweeps the azimuth from Start Yaw to Target Yaw while the elevation pendulum-swings between the Low and High Arc (odd switch counts end on the opposite arc).
- **Spiral Sweep** - monotone azimuth + elevation: the shortest route across the full envelope, therefore the lowest, steadiest camera speed (fewest artefacts) and the most new surface per frame. Recommended for maximum coverage.

Shared by every style: **Path Dolly** (zoom in/out along the view axis in median-depth units), the look **Pivot** and **Output Frames** (the path length always follows `--frames`). Yaw accepts +/-360 degrees (unwrapped, so 270 = -90). The exported `args` and `custom_camera` keep the picker and the Geometry node in sync.

### Meridian Geometry (Enndee) - `Enndee_MeridianGeometry`

Runs `inference/sample.py` of the Meridian release from ComfyUI. `repo` = Meridian folder, `python` = the isolated environment. Accepts a still or an IMAGE batch (encoded as a temporary lossless video, deleted afterwards); with `custom_camera` connected it repeats frame 0 to the path's length and removes motion/freeze/start flags from the args. Adjacent `vggt-omega-fp16-version` and `vggt-omega\checkpoints` folders are detected automatically, so the VGGT widgets can stay empty.

### Meridian Camera Path Configurator (Enndee) - `Enndee_MeridianCameraPath`

Standalone O-orbit path node for path-only graphs; the picker contains the same builder plus the other two styles.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sample.py --help` fails in setup step 5 | Usually torch/VGGT: reinstall torch cu130 in the environment, then `pip install --no-deps -e <tools>\vggt-omega-fp16-version` |
| Checkpoint download refused (gated) | Download `vggt_omega_1b_512.pt` manually from Hugging Face and drop it into `<tools>\vggt-omega\checkpoints` |
| Workflow shows red nodes | `setup.bat --install-community-nodes`, add Pixaroma/RTX packs via ComfyUI-Manager, install Enndees_Nodepack for tracker/trainer |
| Picker shows every widget at once | That is the fallback without the adaptive web extension; everything still works - install [Enndees_Nodepack](https://github.com/Enndee/Enndees_Nodepack) for the adaptive widgets |
| `Pivot X/Z must place the subject in front...` | Raise `path_pivot_z` above 0.15 |
| OOM during the H3 pass | Lower the resolution / super-resolution factor or free VRAM (the example uses three `easy cleanGpuUsed` nodes) |

---

## Development

The adaptive widget extension (the picker's show/hide logic) and the
49-unit-test suite live in the development checkout,
[Enndee/Enndees_Nodepack](https://github.com/Enndee/Enndees_Nodepack). This
install package intentionally ships only what the workflow needs.

---

## Licences & credits

- Code in this repository: **MIT** (see `LICENSE`).
- The Meridian release (MiniMax H3 re-camera model, `recam`/`inference` code, LoRAs, assets): MiniMax H3 Community License - not redistributed here.
- VGGT-Omega: Meta FAIR Noncommercial Research License with gated weights - not redistributed here; the fp16 fork lives at `venlyrina/vggt-omega-fp16-version`.
- ComfyUI and its MiniMax H3 core nodes: ComfyUI / MiniMax.
- The community packs referenced above keep their own licences.




