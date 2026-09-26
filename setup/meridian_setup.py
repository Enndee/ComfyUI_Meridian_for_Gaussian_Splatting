"""Set up everything the example Meridian workflow needs, step by step.

This script is idempotent: run it again after updates or a partial install and
it only fills the gaps. Run it with the portable ComfyUI python
(``setup.bat`` does that for you) or any Python 3.10+.

What it does:

1. makes sure this repository is installed into ``custom_nodes``,
2. checks the ComfyUI-side dependencies (av / numpy / torch are shipped),
3. prepares the isolated Meridian python environment
   (conda env or venv, torch cu130 + the Meridian requirements),
4. clones VGGT-Omega + the fp16 fork and downloads the
   ``vggt_omega_1b_512.pt`` checkpoint when they are missing,
5. verifies ``inference/sample.py --help`` inside the Meridian checkout,
6. copies the example workflow into the ComfyUI user workflows folder and
   rewrites every machine-specific path to this machine.

Use ``--check`` for a read-only report (no installs, no downloads). Pass
``--meridian-zip <file>`` to unpack a Meridian release zip; its LoRAs are
copied into ``models\\loras`` automatically.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Enndee/ComfyUI_Meridian_for_Gaussian_Splatting"
REPO_NAME = "ComfyUI_Meridian_for_Gaussian_Splatting"
VGGT_FORK_URL = "https://github.com/venlyrina/vggt-omega-fp16-version.git"
VGGT_META_URL = "https://github.com/facebookresearch/vggt-omega.git"
CHECKPOINT_REPO = "wincentIsMe/VGGT-Omega"
CHECKPOINT_FILE = "vggt_omega_1b_512.pt"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
ENV_NAME = "meridian-vggt"
# Mirrors Tools\Meridian\setup_geometry_environment.py, the recipe that was
# verified on the reference machine (Python 3.12, torch 2.12.1+cu130).
MERIDIAN_REQUIREMENTS = (
    "numpy<2",
    "Pillow",
    "einops",
    "safetensors",
    "opencv-python",
    "av==16.1.0",
    "transformers>=4.57",
    "peft==0.18.0",
    "huggingface_hub",
    "git+https://github.com/huggingface/diffusers@d6726f3",
)
# Packs used by examples/meridian_customcampath2.json that are public and safe
# to clone. The remaining third-party nodes are listed by the README instead.
COMMUNITY_PACKS = (
    ("comfyui-videohelpersuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("comfyui-easy-use", "https://github.com/yamatazen/ComfyUI-Easy-Use"),
    ("comfyui-various", "https://github.com/jamesWalker55/comfyui-various"),
)

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[96m"
RESET = "\033[0m"


def say(message, color=BLUE):
    print(f"{color}{message}{RESET}", flush=True)


def ok(message):
    print(f"  {GREEN}OK{RESET}   {message}", flush=True)


def note(message):
    print(f"  {YELLOW}NOTE{RESET} {message}", flush=True)


def fail(message):
    print(f"  {RED}FAIL{RESET} {message}", flush=True)


def run(command, dry=False, check=True, cwd=None, env=None):
    printable = " ".join(str(part) for part in command)
    say(f"  $ {printable}", BLUE)
    if dry:
        return 0
    result = subprocess.run(
        [str(part) for part in command], cwd=str(cwd) if cwd else None, env=env)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {printable}")
    return result.returncode


def find_on_path(name):
    return shutil.which(name)


def has_git():
    return find_on_path("git") is not None


class Paths:
    """Every location the setup needs, resolved once and printed afterwards."""

    def __init__(self, args):
        script = Path(__file__).resolve()
        self.repo_dir = script.parents[1]
        self.selection_dir = Path(args.comfy_root).resolve() if args.comfy_root else self._guess_selection()
        self.comfy_root = self._resolve_comfy_root(self.selection_dir)
        self.comfy_dir = self.comfy_root / "ComfyUI"
        self.python_embeded = self.comfy_root / "python_embeded" / "python.exe"
        self.custom_nodes = self.comfy_dir / "custom_nodes"
        self.tools_dir = self._resolve_tools_dir(args)
        self.meridian_dir = Path(args.meridian_dir).resolve() if args.meridian_dir else self.tools_dir / "Meridian"
        self.vggt_dir = self.tools_dir / "vggt-omega"
        self.vggt_fork_dir = self.tools_dir / "vggt-omega-fp16-version"
        self.checkpoint = self.vggt_dir / "checkpoints" / CHECKPOINT_FILE
        self.env_dir = self.tools_dir / ".conda" / ENV_NAME
        self.env_python = Path(args.python).resolve() if args.python else self.env_dir / "python.exe"
        self.workflows_dir = (
            Path(args.workflows_dir).resolve() if args.workflows_dir
            else self.comfy_dir / "user" / "default" / "workflows"
        )

    def _guess_selection(self):
        # Inside custom_nodes\<repo>? Then the portable root is three levels up.
        candidate = self.repo_dir.parents[1] if len(self.repo_dir.parents) > 1 else None
        if candidate and (candidate / "ComfyUI" / "main.py").is_file():
            return candidate
        for parent in self.repo_dir.parents:
            if (parent / "ComfyUI" / "main.py").is_file():
                return parent
        return Path.cwd()

    @staticmethod
    def _resolve_comfy_root(selection):
        """Accept the portable root itself, its parent, or any wrapper folder."""
        if (selection / "ComfyUI" / "main.py").is_file():
            return selection
        for candidate in sorted(selection.glob("ComfyUI*")):
            if (candidate / "ComfyUI" / "main.py").is_file():
                return candidate
        for candidate in sorted(selection.glob("*/ComfyUI*")):
            if (candidate / "ComfyUI" / "main.py").is_file():
                return candidate
        return selection

    def _resolve_tools_dir(self, args):
        """Tools live in a 'Tools' folder of the selected ComfyUI folder.

        An existing install is reused in place; otherwise '<selected>\\Tools' is
        the target, exactly as the README describes.
        """
        if args.tools_dir:
            return Path(args.tools_dir).resolve()
        for candidate in (
            self.selection_dir / "Tools",
            self.comfy_root / "Tools",
            self.comfy_root.parent / "Tools",
        ):
            if (candidate / "Meridian").is_dir() or (candidate / "vggt-omega").is_dir():
                return candidate
        return self.selection_dir / "Tools"


def install_nodes(paths, dry=False):
    say("[1/8] ComfyUI custom node installation", BLUE)
    target = paths.custom_nodes / REPO_NAME
    if target.resolve() == paths.repo_dir.resolve():
        ok(f"already installed at {target}")
        return
    if target.exists():
        ok(f"already installed at {target} (this copy: {paths.repo_dir})")
        return
    say(f"  copying {paths.repo_dir} -> {target}")
    if dry:
        return
    shutil.copytree(
        paths.repo_dir, target,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    ok(f"installed into {target} - restart ComfyUI to load the nodes")


def check_comfy_dependencies(paths):
    say("[2/8] ComfyUI-side dependencies", BLUE)
    if not paths.python_embeded.is_file():
        note(f"portable python not found at {paths.python_embeded}; skipping the import check")
        return
    probe = "import av, numpy, torch; print('deps-ok', torch.__version__)"
    result = subprocess.run(
        [str(paths.python_embeded), "-c", probe], capture_output=True, text=True)
    if result.returncode == 0 and "deps-ok" in result.stdout:
        ok(result.stdout.strip())
    else:
        fail("ComfyUI's python cannot import av/numpy/torch:")
        print(f"       {result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown error'}")
    note("the Meridian nodes themselves need no extra pip packages in ComfyUI")


def conda_executable():
    for name in ("conda", "mamba"):
        found = find_on_path(name)
        if found:
            return found
    return None


def ensure_environment(paths, args, dry=False):
    say("[3/8] Meridian python environment", BLUE)
    if paths.env_python.is_file():
        ok(f"using {paths.env_python}")
        return
    if args.skip_env:
        note("--skip-env: leaving the environment alone (point --python at one later)")
        return
    conda = conda_executable()
    if conda:
        say(f"  creating conda environment {paths.env_dir} (python 3.12)")
        run([conda, "create", "-p", str(paths.env_dir), "python=3.12", "-y"], dry=dry)
    else:
        base_python = find_on_path("python")
        if not base_python:
            raise SystemExit(
                "Neither conda nor python found. Install Miniconda (recommended) or "
                "Python 3.12, then run this script again.")
        say(f"  creating venv {paths.env_dir} with {base_python}")
        run([base_python, "-m", "venv", str(paths.env_dir)], dry=dry)
    say("  installing torch/torchvision")
    run([paths.env_python, "-m", "pip", "install", "--upgrade", "pip"], dry=dry)
    run([
        paths.env_python, "-m", "pip", "install",
        "torch", "torchvision", "--index-url", args.torch_index,
    ], dry=dry)
    say("  installing Meridian requirements (this downloads a few hundred MB)")
    run([paths.env_python, "-m", "pip", "install", *MERIDIAN_REQUIREMENTS], dry=dry)
    ok("environment ready")


def ensure_vggt(paths, dry=False):
    say("[4/8] VGGT-Omega code and checkpoint", BLUE)
    if not has_git():
        note("git not found on PATH; clone these manually and rerun:")
        print(f"       git clone {VGGT_FORK_URL} \"{paths.vggt_fork_dir}\"")
        print(f"       git clone {VGGT_META_URL} \"{paths.vggt_dir}\"")
    else:
        if (paths.vggt_fork_dir / "vggt_omega" / "models" / "vggt_omega.py").is_file():
            ok(f"fp16 fork present: {paths.vggt_fork_dir}")
        elif paths.vggt_fork_dir.is_dir():
            note(f"{paths.vggt_fork_dir} looks incomplete; pulling")
            run(["git", "-C", paths.vggt_fork_dir, "pull", "--ff-only"], dry=dry, check=False)
        else:
            run(["git", "clone", VGGT_FORK_URL, paths.vggt_fork_dir], dry=dry, check=False)
        if (paths.vggt_dir / ".git").is_dir():
            ok(f"meta checkout present: {paths.vggt_dir}")
        else:
            run(["git", "clone", VGGT_META_URL, paths.vggt_dir], dry=dry, check=False)

    if paths.checkpoint.is_file():
        ok(f"checkpoint present: {paths.checkpoint}")
        return
    if not paths.env_python.is_file():
        note("no Meridian python yet; download the checkpoint later from "
             f"https://huggingface.co/{CHECKPOINT_REPO} to {paths.checkpoint}")
        return
    huggingface_check = subprocess.run(
        [str(paths.env_python), "-c", "import huggingface_hub"], capture_output=True)
    if huggingface_check.returncode != 0:
        run([paths.env_python, "-m", "pip", "install", "huggingface_hub"], dry=dry)
    say(f"  downloading {CHECKPOINT_FILE} (~4.6 GB) from huggingface.co/{CHECKPOINT_REPO}")
    download = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download('{CHECKPOINT_REPO}', '{CHECKPOINT_FILE}', "
        f"local_dir=r'{paths.checkpoint.parent}'))"
    )
    exit_code = run([paths.env_python, "-c", download], dry=dry, check=False)
    if not dry and (exit_code != 0 or not paths.checkpoint.is_file()):
        note("automatic download failed - grab the file manually from "
             f"https://huggingface.co/{CHECKPOINT_REPO} and place it at {paths.checkpoint}")
    elif not dry:
        ok(f"checkpoint downloaded: {paths.checkpoint}")


def ensure_meridian(paths, dry=False):
    say("[5/8] Meridian checkout and environment check", BLUE)
    extract_meridian_zip(paths, args, dry)
    sample = paths.meridian_dir / "inference" / "sample.py"
    if not sample.is_file():
        fail(f"Meridian not found at {paths.meridian_dir}")
        print("       Meridian is a separate, licence-gated release (MiniMax H3 Community License)")
        print("       that this repository does not redistribute. Point the setup at your copy:")
        print("         setup.bat --meridian-dir \"X:\\path\\to\\Meridian\"")
        print("       The folder must contain inference\\sample.py, assets\\ and the release's")
        print("       ComfyUI nodes (MeridianFrozenPrompt).")
        return False
    ok(f"Meridian checkout found: {paths.meridian_dir}")
    if not paths.env_python.is_file():
        note(f"Meridian python not found at {paths.env_python}; skipping the --help check")
        return True
    say("  running  inference/sample.py --help  inside the Meridian environment")
    exit_code = run([paths.env_python, sample, "--help"], dry=dry, check=False)
    if dry or exit_code == 0:
        ok("Meridian CLI answered - the environment is functional")
        return True
    fail("sample.py --help failed; read the traceback above (usually a torch/VGGT mismatch)")
    return False


def _rewrite_tools_path(value, paths):
    """Point any '...\\Tools\\<tail>' string at this machine's tools folder."""
    if not isinstance(value, str):
        return value, False
    normalized = value.replace("/", "\\")
    marker = None
    for candidate in ("\\Tools\\", "\\tools\\"):
        index = normalized.rfind(candidate)
        if index != -1 and (marker is None or index > marker):
            marker = index
    if marker is None:
        return value, False
    tail = normalized[marker + len("\\Tools\\"):]
    if not tail:
        return value, False
    return str(paths.tools_dir / tail), True


def _node_widget_order(node_type):
    """name -> index map from the freshly imported node definitions (best effort)."""
    nodes_dir = Path(__file__).resolve().parents[1] / "nodes"
    if str(nodes_dir) not in sys.path:
        sys.path.insert(0, str(nodes_dir))
    try:
        import enndee_meridian_camera_path
        import enndee_meridian_geometry
        import enndee_meridian_parameter_picker
    except Exception:
        return {}
    classes = {
        "Enndee_MeridianGeometry": enndee_meridian_geometry.EnndeeMeridianGeometry,
        "Enndee_MeridianParameterPicker": enndee_meridian_parameter_picker.MeridianParameterPickerEnndee,
        "Enndee_MeridianCameraPath": enndee_meridian_camera_path.MeridianCameraPathConfigurator,
    }
    selected = classes.get(node_type)
    if selected is None:
        return {}
    required = selected.INPUT_TYPES()["required"]
    return {name: index for index, name in enumerate(required)}


def patch_workflow(paths, args, dry=False):
    say("[6/8] Example workflow", BLUE)
    if args.no_workflow:
        note("--no-workflow: the workflow was not touched")
        return
    source = paths.repo_dir / "examples" / "meridian_customcampath2.json"
    if not source.is_file():
        note("examples/meridian_customcampath2.json missing; skipping")
        return
    data = json.loads(source.read_text(encoding="utf-8"))

    explicit = {
        "Enndee_MeridianGeometry": {"repo": str(paths.meridian_dir), "python": str(paths.env_python)},
        "Enndee_MeridianParameterPicker": {
            "vggt_repo": str(paths.vggt_fork_dir),
            "vggt_checkpoint": str(paths.checkpoint),
        },
    }
    rewritten = 0
    explicit_applied = 0
    for node in data.get("nodes", []):
        values = node.get("widgets_values")
        named = node.get("widgets_values_named")
        if isinstance(values, list):
            for index, value in enumerate(values):
                new_value, changed = _rewrite_tools_path(value, paths)
                if changed and new_value != value:
                    values[index] = new_value
                    rewritten += 1
        if isinstance(named, dict):
            for key, value in list(named.items()):
                new_value, changed = _rewrite_tools_path(value, paths)
                if changed and new_value != value:
                    named[key] = new_value
                    rewritten += 1
        targets = explicit.get(node.get("type"))
        if not targets:
            continue
        order = _node_widget_order(node.get("type"))
        for name, new_value in targets.items():
            if isinstance(named, dict) and name in named:
                named[name] = new_value
                explicit_applied += 1
            if isinstance(values, list) and name in order and order[name] < len(values):
                values[order[name]] = new_value
                explicit_applied += 1
    if rewritten:
        ok(f"rewrote {rewritten} '...\\Tools\\...' path(s) to {paths.tools_dir}")
    if explicit_applied:
        ok(f"set {explicit_applied} Meridian widget value(s) (repo / python / VGGT paths)")
    if not rewritten and not explicit_applied:
        note("no Meridian paths were found in the example workflow")
    if dry:
        return
    paths.workflows_dir.mkdir(parents=True, exist_ok=True)
    target = paths.workflows_dir / source.name
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    ok(f"workflow written to {target}")
    report_user_specific_paths(data, paths)


def report_user_specific_paths(data, paths):
    tools_prefix = str(paths.tools_dir).replace("/", "\\").lower()
    leftovers = set()
    pattern = re.compile(r"^[A-Za-z]:[\\/]")
    for node in data.get("nodes", []):
        named = node.get("widgets_values_named")
        candidates = list(node.get("widgets_values") or [])
        if isinstance(named, dict):
            candidates.extend(named.values())
        for value in candidates:
            if not isinstance(value, str) or not pattern.match(value):
                continue
            if value.replace("/", "\\").lower().startswith(tools_prefix):
                continue
            leftovers.add(value)
    if leftovers:
        note("review these machine-specific values in the workflow:")
        for value in sorted(leftovers):
            print(f"         {value}")


def install_community_packs(paths, args, dry=False):
    if not args.install_community_nodes:
        return
    say("[extra] Community node packs used by the example workflow", BLUE)
    for name, url in COMMUNITY_PACKS:
        target = paths.custom_nodes / name
        if target.is_dir():
            ok(f"{name} already present")
            continue
        if not has_git():
            note(f"git missing; clone {url} into {target} manually")
            continue
        run(["git", "clone", "--depth", "1", url, target], dry=dry, check=False)


def print_report(paths):
    say("Resolved locations", BLUE)
    rows = (
        ("Selected folder", paths.selection_dir),
        ("ComfyUI root", paths.comfy_root),
        ("custom_nodes", paths.custom_nodes),
        ("tools folder", paths.tools_dir),
        ("Meridian checkout", paths.meridian_dir),
        ("VGGT fp16 fork", paths.vggt_fork_dir),
        ("VGGT checkpoint", paths.checkpoint),
        ("Meridian python", paths.env_python),
        ("workflows folder", paths.workflows_dir),
    )
    for label, value in rows:
        state = " [present]" if Path(value).exists() else " [missing]"
        print(f"  {label:<18} {value}{state}")


def _read_extra_model_paths(paths):
    """Collect model roots from extra_model_paths.yaml files, if any."""
    roots = []
    for candidate in (paths.comfy_root / "extra_model_paths.yaml", paths.comfy_dir / "extra_model_paths.yaml"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("base_path:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if value:
                    roots.append(Path(value))
    return roots


def _model_roots(paths):
    roots = [paths.comfy_dir / "models"]
    roots.extend(_read_extra_model_paths(paths))
    return [root for root in roots if root.is_dir()]


REQUIRED_MODELS = (
    ("diffusion_models", "H3/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "your MiniMax H3 download"),
    ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "your MiniMax H3 download"),
    ("vae", "minimax_h3_video_vae_fp16.safetensors", "your MiniMax H3 download"),
    ("vae", "minimax_h3_audio_vae_fp32.safetensors", "your MiniMax H3 download"),
    ("loras", "meridian_teacher_lora.safetensors", "ships in the Meridian release"),
    ("loras", "meridian_turbo_lora.safetensors", "ships in the Meridian release"),
)


def ensure_models(paths, args, dry=False):
    say("[7/8] Model files the example workflow needs", BLUE)
    roots = _model_roots(paths)
    if not roots:
        note("no models folder found yet; expected under ComfyUI\\models")
        return
    missing = []
    for folder, name, source in REQUIRED_MODELS:
        if any((root / folder / name).is_file() for root in roots):
            ok(f"{folder}\\{name}")
        else:
            missing.append((folder, name, source))
    if missing:
        note("missing (copy them into ComfyUI\\models\\<folder>):")
        for folder, name, source in missing:
            print(f"         {folder}\\{name}  <- {source}")
    else:
        ok("all workflow models found")


def extract_meridian_zip(paths, args, dry=False):
    """Unpack a Meridian release zip when the user provided one."""
    zip_option = getattr(args, "meridian_zip", None)
    if not zip_option:
        return
    zip_path = Path(zip_option).resolve()
    if not zip_path.is_file():
        note(f"--meridian-zip not found: {zip_path}")
        return
    if (paths.meridian_dir / "inference" / "sample.py").is_file():
        ok("Meridian is already extracted; keeping it")
        copy_meridian_loras(paths, dry=dry)
        return
    import zipfile
    say(f"  extracting {zip_path.name} into {paths.meridian_dir}")
    if dry:
        return
    paths.meridian_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(paths.meridian_dir)
    entries = list(paths.meridian_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not (paths.meridian_dir / "inference").is_dir():
        inner = entries[0]
        for item in inner.iterdir():
            item.rename(paths.meridian_dir / item.name)
        inner.rmdir()
    ok(f"Meridian extracted to {paths.meridian_dir}")
    copy_meridian_loras(paths, dry=dry)


def copy_meridian_loras(paths, dry=False):
    """Copy the release's Meridian LoRAs into the ComfyUI loras folder."""
    matches = list(paths.meridian_dir.rglob("meridian_*lora.safetensors"))
    if not matches:
        return
    target = paths.comfy_dir / "models" / "loras"
    if not dry:
        target.mkdir(parents=True, exist_ok=True)
    for lora in matches:
        destination = target / lora.name
        if destination.is_file():
            ok(f"lora already present: {destination}")
            continue
        if not dry:
            shutil.copy2(lora, destination)
        ok(f"lora copied: {destination}")
    if _read_extra_model_paths(paths):
        note("extra_model_paths.yaml detected: if ComfyUI reads loras from another "
             "folder, copy the meridian_*lora.safetensors files there as well")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a fresh ComfyUI for the Meridian camera-path workflow.")
    parser.add_argument("--comfy-root", help="portable ComfyUI folder (contains ComfyUI\\ and python_embeded\\)")
    parser.add_argument("--tools-dir", help="folder for Meridian/VGGT (default: <comfy root>\\..\\Tools)")
    parser.add_argument("--meridian-dir", help="your Meridian release folder (contains inference\\sample.py)")
    parser.add_argument("--python", help="Meridian python (default: <tools>\\.conda\\meridian-vggt\\python.exe)")
    parser.add_argument("--workflows-dir", help="where to write the patched example workflow")
    parser.add_argument("--torch-index", default=DEFAULT_TORCH_INDEX,
                        help="pip index for torch (default: the CUDA 13.0 wheels)")
    parser.add_argument("--check", action="store_true", help="report only; change nothing")
    parser.add_argument("--dry-run", action="store_true", help="print every command without running it")
    parser.add_argument("--skip-env", action="store_true", help="do not create/install the Meridian environment")
    parser.add_argument("--no-workflow", action="store_true", help="do not copy/patch the example workflow")
    parser.add_argument("--meridian-zip",
                        help="a Meridian release .zip to extract into <tools>\\Meridian")
    parser.add_argument("--install-community-nodes", action="store_true",
                        help="also clone the public node packs the example workflow uses")
    args = parser.parse_args()

    say("ComfyUI Meridian for Gaussian Splatting - setup", BLUE)
    paths = Paths(args)
    print_report(paths)
    if args.check:
        say("--check finished (nothing was modified)", GREEN)
        return 0

    install_nodes(paths, dry=args.dry_run)
    check_comfy_dependencies(paths)
    ensure_environment(paths, args, dry=args.dry_run)
    ensure_vggt(paths, dry=args.dry_run)
    meridian_ok = ensure_meridian(paths, dry=args.dry_run)
    ensure_models(paths, args, dry=args.dry_run)
    patch_workflow(paths, args, dry=args.dry_run)
    install_community_packs(paths, args, dry=args.dry_run)

    say("Summary", BLUE)
    if meridian_ok:
        ok("Meridian environment verified")
    else:
        note("Meridian still needs attention (see [5/8] above)")
    ok(f"Meridian Geometry values: repo = {paths.meridian_dir}")
    ok(f"                           python = {paths.env_python}")
    say("Restart ComfyUI, open the copied workflow and queue it. The first run "
        "downloads nothing else; models stay user-provided.", GREEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




