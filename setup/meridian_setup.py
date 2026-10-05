"""Set up everything the example Meridian workflow needs, step by step.

This script is idempotent: run it again after updates or a partial install and
it only fills the gaps. Run it with the portable ComfyUI python
(``setup.bat`` does that for you) or any Python 3.10+.

The nodes themselves live in the **Enndees Nodepack**
(https://github.com/Enndee/Enndees_Nodepack) - this repository is only the
installer, so there is exactly one place that owns the node code.

What it does:

1. clones (or updates) the **Enndees Nodepack** into ``custom_nodes``,
2. checks the ComfyUI-side dependencies (av / numpy / torch are shipped),
3. installs **Depth Anything v3** into ComfyUI's python (``--no-deps``) - the
   fast-depth backend the Meridian Geometry node reprojections with.
4. unpacks a Meridian release zip and copies its LoRAs into ``models\\loras``
   (licence-gated, they are not redistributed here),
5. reports the model files the workflow needs,
6. copies the example workflows into the ComfyUI user workflows folder and
   rewrites every machine-specific path to this machine,
7. clones the community node packs the example workflow uses (Sharp-Selector,
   rgthree, KJNodes, various, custom-scripts, VHS, easy-use, Pixaroma, BRIA
   RMBG, RTX nodes, UniBlockSwap, Memory-Cleanup, H3 MotionCache, H3 Turbo,
   H3 latent upscaler). ``--skip-community-nodes`` turns this off.

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
# The node code lives in the Enndees Nodepack (single source of truth) - this
# repository only ships the installer, the example workflows and the docs.
PACK_URL = "https://github.com/Enndee/Enndees_Nodepack"
PACK_NAME = "Enndees_Nodepack"
PACK_MARKER = ("nodes", "enndee_meridian_parameters.py")
DA3_PACKAGE = "depth-anything-3"
# Every community pack examples/Meridian_Splatting_1.1.json needs (all Enndee_* /
# Meridian* nodes come from the pack above). Public and safe to clone; installed
# by default, `--skip-community-nodes` turns it off.
COMMUNITY_PACKS = (
    ("ComfyUI-Sharp-Selector", "https://github.com/ethanfel/ComfyUI-Sharp-Selector"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy"),
    ("comfyui-kjnodes", "https://github.com/kijai/ComfyUI-KJNodes"),
    ("comfyui-various", "https://github.com/jamesWalker55/comfyui-various"),
    ("comfyui-custom-scripts", "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"),
    ("comfyui-videohelpersuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("comfyui-easy-use", "https://github.com/yamatazen/ComfyUI-Easy-Use"),
    ("ComfyUI-Pixaroma", "https://github.com/pixaroma/ComfyUI-Pixaroma"),
    ("ComfyUI-BRIA_AI-RMBG", "https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG"),
    ("comfyui_nvidia_rtx_nodes", "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI"),
    ("uniblockswap", "https://github.com/smthemex/ComfyUI_UniBlockSwap"),
    ("comfyui_memory_cleanup", "https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup"),
    ("minimax-h3-motioncache", "https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache"),
    ("comfyui-minimax-h3-turbo", "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo"),
    ("comfyui-minimax-h3-latent-upscaler",
     "https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus"),
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
            if (candidate / "Meridian").is_dir():
                return candidate
        return self.selection_dir / "Tools"


def install_pack(paths, dry=False):
    """Clone (or update) the Enndees Nodepack - the only place the nodes live."""
    say("[1/7] ComfyUI custom node installation (Enndees Nodepack)", BLUE)
    target = paths.custom_nodes / PACK_NAME
    marker = target.joinpath(*PACK_MARKER)
    if marker.is_file():
        if (target / ".git").is_dir() and has_git():
            run(["git", "-C", str(target), "pull", "--ff-only"], dry=dry, check=False)
        ok(f"{PACK_NAME} already installed at {target}")
        return
    if not has_git():
        note(f"git missing; clone {PACK_URL} into {target} manually")
        return
    say(f"  cloning {PACK_URL} -> {target}")
    run(["git", "clone", PACK_URL, str(target)], dry=dry, check=False)
    if not dry and not marker.is_file():
        fail(f"could not clone {PACK_URL}")
        print(f"       download it manually into {target}")
        return
    ok(f"cloned {PACK_NAME} into {target} - restart ComfyUI to load the nodes")


def check_comfy_dependencies(paths):
    say("[2/7] ComfyUI-side dependencies", BLUE)
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


def ensure_fast_depth(paths, dry=False):
    """Depth Anything v3 into ComfyUI's own python - the geometry node's fast-depth backend.

    Installed with ``--no-deps``: the wheel pins ``numpy<2`` and would drag xformers/open3d/
    pycolmap/moviepy/gsplat/evo into ComfyUI. The backend stubs the two optional sub-packages
    itself (export dispatcher / moviepy, evo-based pose alignment), so this one package is enough.

    The probe imports the *package* (``depth_anything_3``), never ``depth_anything_3.api`` - the
    latter needs moviepy, which the backend stubs on purpose.
    """
    say("[3/7] Depth Anything v3 (fast depth backend)", BLUE)
    probe = ["import depth_anything_3; print('da3-ok')"]
    if not paths.python_embeded.is_file():
        note(f"portable python not found at {paths.python_embeded}; install manually:")
        print(f"       {sys.executable} -m pip install --no-deps {DA3_PACKAGE}")
        return
    result = subprocess.run(
        [str(paths.python_embeded), "-c", probe[0]], capture_output=True, text=True)
    if result.returncode == 0 and "da3-ok" in result.stdout:
        ok("depth_anything_3 importable - Meridian Geometry can reproject in-process")
        return
    say(f"  pip install --no-deps {DA3_PACKAGE}  (single package, no dependency resolution)")
    run([str(paths.python_embeded), "-m", "pip", "install", "--no-deps", DA3_PACKAGE],
        dry=dry, check=False)
    if dry:
        return
    result = subprocess.run(
        [str(paths.python_embeded), "-c", probe[0]], capture_output=True, text=True)
    if result.returncode == 0 and "da3-ok" in result.stdout:
        ok("depth_anything_3 installed - its model weights download into the Hugging Face cache "
           "on first use")
    else:
        fail("depth_anything_3 did not import")
        print("       run it by hand: python -m pip install --no-deps depth-anything-3")


def ensure_meridian(paths, args, dry=False):
    """LoRAs of the licence-gated Meridian release - the H3 re-camera pass loads them."""
    say("[4/7] Meridian release assets (LoRAs)", BLUE)
    extract_meridian_zip(paths, args, dry)
    if paths.meridian_dir.is_dir():
        copy_meridian_loras(paths, dry=dry)
        ok(f"Meridian release at {paths.meridian_dir} (source of meridian_*lora.safetensors)")
        return
    note("no Meridian release found - its LoRAs are licence-gated (MiniMax H3 Community License)")
    print("       and are not redistributed here. Unpack yours with:")
    print("         setup.bat --meridian-zip \"X:\\path\\to\\Meridian_release.zip\"")
    print("       or copy meridian_teacher_lora.safetensors / meridian_turbo_lora.safetensors")
    print("       into ComfyUI\\models\\loras by hand.")


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


def patch_workflow(paths, args, dry=False):
    say("[6/7] Example workflows", BLUE)
    if args.no_workflow:
        note("--no-workflow: the workflows were not touched")
        return
    sources = sorted((paths.repo_dir / "examples").glob("*.json"))
    if not sources:
        note("examples/*.json missing; skipping")
        return
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        rewritten = 0
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
        if rewritten:
            ok(f"{source.name}: rewrote {rewritten} '...\\Tools\\...' path(s) to {paths.tools_dir}")
        else:
            note(f"{source.name}: no machine-specific Meridian paths found")
        if dry:
            continue
        paths.workflows_dir.mkdir(parents=True, exist_ok=True)
        target = paths.workflows_dir / source.name
        target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                          encoding="utf-8")
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
    if getattr(args, "skip_community_nodes", False):
        note("--skip-community-nodes: the community packs were not touched")
        return
    say("[7/7] Community node packs used by the example workflow", BLUE)
    if not has_git():
        note("git missing; clone these manually:")
        for name, url in COMMUNITY_PACKS:
            print(f"         {name:<40} {url}")
        return
    for name, url in COMMUNITY_PACKS:
        target = paths.custom_nodes / name
        if target.is_dir():
            ok(f"{name} already present")
            continue
        run(["git", "clone", "--depth", "1", url, str(target)], dry=dry, check=False)


def print_report(paths):
    say("Resolved locations", BLUE)
    da3 = subprocess.run(
        [str(paths.python_embeded), "-c", "import depth_anything_3"],
        capture_output=True, text=True) if paths.python_embeded.is_file() else None
    rows = (
        ("Selected folder", paths.selection_dir),
        ("ComfyUI root", paths.comfy_root),
        ("custom_nodes", paths.custom_nodes),
        ("tools folder", paths.tools_dir),
        ("Meridian release", paths.meridian_dir),
        ("Depth Anything v3", "installed" if da3 and da3.returncode == 0 else
         f"missing (pip install --no-deps {DA3_PACKAGE})"),
        ("workflows folder", paths.workflows_dir),
    )
    for label, value in rows:
        if label == "Depth Anything v3":
            state = " [ok]" if str(value).startswith("installed") else " [todo]"
            print(f"  {label:<18} {value}{state}")
            continue
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
    say("[5/7] Model files the example workflow needs", BLUE)
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
    parser.add_argument("--tools-dir", help="folder for the Meridian release (default: <comfy root>\\Tools)")
    parser.add_argument("--meridian-dir", help="your Meridian release folder (source of the meridian_* LoRAs)")
    parser.add_argument("--workflows-dir", help="where to write the patched example workflows")
    parser.add_argument("--check", action="store_true", help="report only; change nothing")
    parser.add_argument("--dry-run", action="store_true", help="print every command without running it")
    parser.add_argument("--no-workflow", action="store_true", help="do not copy/patch the example workflows")
    parser.add_argument("--meridian-zip",
                        help="a Meridian release .zip to extract into <tools>\\Meridian")
    parser.add_argument("--install-community-nodes", action="store_true",
                        help="deprecated: the community packs are installed by default now")
    parser.add_argument("--skip-community-nodes", action="store_true",
                        help="do not clone the community node packs the example workflow uses")
    args = parser.parse_args()

    say("ComfyUI Meridian for Gaussian Splatting - setup", BLUE)
    paths = Paths(args)
    print_report(paths)
    if args.check:
        say("--check finished (nothing was modified)", GREEN)
        return 0

    install_pack(paths, dry=args.dry_run)
    check_comfy_dependencies(paths)
    ensure_fast_depth(paths, dry=args.dry_run)
    ensure_meridian(paths, args, dry=args.dry_run)
    ensure_models(paths, args, dry=args.dry_run)
    patch_workflow(paths, args, dry=args.dry_run)
    install_community_packs(paths, args, dry=args.dry_run)

    say("Summary", BLUE)
    ok(f"nodes installed in {paths.custom_nodes / PACK_NAME}")
    ok("Depth Anything v3 reprojection: no VGGT, no separate environment")
    if not args.skip_community_nodes:
        ok(f"{len(COMMUNITY_PACKS)} community packs ensured "
           "(Sharp-Selector, rgthree, KJNodes, various, custom-scripts, VHS, easy-use, ...)")
    say("Restart ComfyUI, open the copied workflow and queue it. The DA3 model weights "
        "download into the Hugging Face cache on first use; everything else is "
        "user-provided (see the README).", GREEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




