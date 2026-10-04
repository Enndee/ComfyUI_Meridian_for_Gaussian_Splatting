# Copyright 2026 Viggle AI. Licensed under the Apache License, Version 2.0 (see LICENSE-CODE).
# SPDX-License-Identifier: Apache-2.0
"""A ComfyUI node that produces Meridian's two reference videos from a clip and a camera move.

Drop this file into `ComfyUI/custom_nodes/` and **Meridian Geometry** appears under `video`. It shells
out to this repo's `inference/sample.py --preview-only`, which runs VGGT-Omega, unprojects frame
`--start` into a point cloud, moves the camera and rasterises it -- then stops before the VAE. What
comes back is what the sampler wants: `<Video 1>` the source crop, `<Video 2>` the warp.

The optional ComfyUI IMAGE input accepts either one still (written as a temporary PNG) or an image
batch (encoded as a temporary video). Connect a multi-frame video loader's IMAGE output when using
`--camera-path`; connecting it must not silently collapse the clip to its first frame.

A subprocess, not an import, because the geometry side pulls in VGGT-Omega, which is gated and
FAIR-NC-licensed: you accept that licence and install it yourself, and nothing of it ends up inside
ComfyUI's process. `python` must therefore be an interpreter that can import it -- not necessarily the
one ComfyUI runs on.

`args` is passed to `sample.py` verbatim, so the camera vocabulary is the one the README documents
(`--yaw --truck --boom --dolly --zoom --pivot --pivot-lock --aim --sweep --ease --bounce --freeze
--follow --cull --start --frames ...`). Nothing is re-declared here, so nothing can drift out of sync --
which is also where `--vggt` and `--vggt-repo` go if you have not exported `$VGGT_OMEGA_CKPT`/`$VGGT_OMEGA_DIR`.
An optional `args_override` STRING socket accepts generated arguments (for example from the Enndees
Nodepack's Meridian Parameter Picker) and takes precedence over the editable `args` widget when linked.

What is read back is the `cond_*.mp4` pair, which `--preview-only` writes at the condition canvas --
the model is conditioned at the 480 class, so the sampler is handed exactly what `inference/sample.py`
feeds its own, with no second resampling in between.
`width`/`height` are the target canvas and belong on `MiniMaxH3ReferenceToVideo`: do not let it size the
references itself, its `adapt_canvas` would land on the 768 class and condition the model off-distribution.
"""
import os
import re
import shlex
import subprocess
import tempfile

import av
import numpy as np
import torch


def _add_default_vggt_paths(args, repo):
    """Append this machine's isolated VGGT checkout/checkpoint if none were supplied."""
    has_repo = any(token == "--vggt-repo" or token.startswith("--vggt-repo=") for token in args)
    has_checkpoint = any(token == "--vggt" or token.startswith("--vggt=") for token in args)
    if has_repo and has_checkpoint:
        return args

    candidate_roots = [os.path.dirname(repo)]
    configured_repo = next(
        (os.path.abspath(token.split("=", 1)[1]) for token in args if token.startswith("--vggt-repo=")),
        None,
    )
    if configured_repo:
        candidate_roots.insert(0, os.path.dirname(configured_repo))
    if "--vggt-repo" in args:
        option_index = args.index("--vggt-repo")
        if option_index + 1 < len(args):
            candidate_roots.insert(0, os.path.dirname(os.path.abspath(args[option_index + 1])))

    for root in candidate_roots:
        candidate_repo = os.path.join(root, "vggt-omega-fp16-version")
        candidate_checkpoint = os.path.join(root, "vggt-omega", "checkpoints", "vggt_omega_1b_512.pt")
        if os.path.isfile(os.path.join(candidate_repo, "vggt_omega", "models", "vggt_omega.py")):
            if not has_repo:
                args.extend(["--vggt-repo", candidate_repo])
            if not has_checkpoint:
                args.extend(["--vggt", candidate_checkpoint])
            break
    return args


def _frames(path):
    with av.open(path) as c:
        a = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    return torch.from_numpy(a).float().div_(255.0)


def _write_image_batch_video(frames, path, fps=24):
    """Encode a BHWC image batch as a temporary lossless H.264 RGB video."""
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError("Meridian image batches must have shape (frames, height, width, RGB[A]).")

    height, width = frames.shape[1:3]
    if height < 1 or width < 1:
        raise ValueError("Meridian image batches must have non-empty frames.")

    if isinstance(frames, torch.Tensor):
        frame_iterator = (
            frame.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            for frame in frames
        )
    else:
        frame_iterator = iter(np.asarray(frames))

    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "fast"}

        for pixels in frame_iterator:
            rgb = np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


class MeridianGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("STRING", {"default": "clip.mp4"}),
                             "args": ("STRING", {"default": "--boom 0.35 --pivot 0.5,0.55 --ease --sweep",
                                                 "multiline": True}),
                             "repo": ("STRING", {"default": "/path/to/release_recam"}),
                             "python": ("STRING", {"default": "python"})},
                "optional": {"image": ("IMAGE",),
                             "args_override": ("STRING", {"forceInput": True,
                                                           "tooltip": "Optional argument string from a parameter-picker node. When connected, it replaces the args text above."})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("source", "render", "width", "height", "length")
    FUNCTION = "build"
    CATEGORY = "video"

    def build(self, video, args, repo, python, image=None, args_override=None):
        out = tempfile.mkdtemp(prefix="meridian_")
        image_input_path = None
        try:
            if image is not None:
                if image.shape[0] > 1:
                    handle, image_input_path = tempfile.mkstemp(prefix="meridian_batch_", suffix=".mp4")
                    os.close(handle)
                    _write_image_batch_video(image, image_input_path)
                else:
                    from PIL import Image

                    handle, image_input_path = tempfile.mkstemp(prefix="meridian_still_", suffix=".png")
                    os.close(handle)
                    pixels = image[0].detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
                    Image.fromarray(pixels[..., :3]).save(image_input_path)
                video = image_input_path

            cmd = [python, f"{repo}/inference/sample.py", "--video", video, "--out", out, "--preview-only"]
            effective_args = args_override if args_override is not None else args
            cmd += shlex.split(effective_args)
            cmd = _add_default_vggt_paths(cmd, repo)
            print("Meridian geometry:", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(r.stdout, flush=True)
            r.check_returncode()

            # "1280x960 -> content ... canvas (1184, 864), cond (736, 544)"
            w, h = (int(g) for g in re.search(r"canvas \((\d+), (\d+)\)", r.stdout).groups())
            source = _frames(f"{out}/cond_source.mp4")
            render = _frames(f"{out}/cond_render.mp4")
            return (source, render, w, h, source.shape[0])
        finally:
            if image_input_path and os.path.exists(image_input_path):
                os.remove(image_input_path)
            if os.path.isdir(out):
                for name in os.listdir(out):
                    path = os.path.join(out, name)
                    if os.path.isfile(path):
                        os.remove(path)
                os.rmdir(out)


NODE_CLASS_MAPPINGS = {"MeridianGeometry": MeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianGeometry": "Meridian Geometry"}
