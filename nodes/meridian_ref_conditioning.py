# SPDX-License-Identifier: MIT
"""Clip-free MiniMax H3 reference conditioning for Meridian frozen embeds.

`MiniMaxH3ReferenceToVideo` always tokenizes and encodes the prompt through the huge
Qwen3-VL text encoder - work `MeridianFrozenPromptFile` then throws away. This node
replaces it when you use baked embeds: it builds exactly the same `minimax_refs` payload
(image + video reference latents, resized and 17k+5-aligned like the core node) and emits
a placeholder text tensor that `MeridianFrozenPromptFile` completes. The CLIPLoader can be
bypassed - the text encoder is never loaded and never encodes.

`ref_canvas`:
- ``h3``          - the core node's canvas (768 short edge, ~1.0 MP): unchanged behaviour.
- ``meridian480`` - Meridian's own 480-class ladder for the reference videos (~0.4 MP):
                    ~2.5x fewer reference tokens per step, and the canvas the adapters
                    were trained with. The rotary grid is normalised by sqrt(area), so
                    only the sampling density changes (see recam/h3.py's LADDERS).

Reference order matches the baked presentation: images first (`<Picture i>`), then videos
(`<Video k>`). The placeholder conditioning only carries `minimax_refs`; wire
`MeridianFrozenPromptFile` after it (conditioning + latent) or the model gets a 1-token
text tensor.
"""

import math

import torch
import torch.nn.functional as F

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
LADDER_480 = [(416, 960), (448, 896), (480, 832), (544, 736), (640, 640),
              (736, 544), (832, 480), (896, 448), (960, 416)]


def _stretch(samples, width, height, crop):
    if crop == "center":
        b, c, h, w = samples.shape
        scale = max(height / h, width / w)
        nh, nw = max(height, round(h * scale)), max(width, round(w * scale))
        samples = F.interpolate(samples, size=(nh, nw), mode="bilinear", align_corners=False)
        top, left = (nh - height) // 2, (nw - width) // 2
        return samples[:, :, top:top + height, left:left + width]
    return F.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)


def _resize(image, width, height, crop="disabled"):
    samples = image[..., :3].movedim(-1, 1)
    try:
        import comfy.utils
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    except Exception:
        samples = _stretch(samples, width, height, crop)
    return samples.movedim(1, -1)


def _adapt_canvas(width, height):
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _bucket_480(width, height):
    """Meridian's 480-class ladder entry nearest in log-aspect (recam/h3.py bucket())."""
    i = min(range(len(LADDER_480)),
            key=lambda i: abs(math.log(LADDER_480[i][0] / LADDER_480[i][1] * height / width)))
    return LADDER_480[i]


def _video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = max(5, length)
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count, _video_latent_t(frame_count), round(frame_count / FPS * AUDIO_LATENT_FPS)


class MeridianRefConditioning:
    """`MiniMaxH3ReferenceToVideo` minus the text encoder: builds only minimax_refs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Frame count at 24 fps (17k+5 grid, 124 = ~5s)"}),
                "ref_canvas": (["h3", "meridian480"], {"default": "h3",
                               "tooltip": "Reference video canvas: 'h3' = the core node's "
                                          "768-class (~1.0 MP); 'meridian480' = Meridian's own "
                                          "480-class ladder (~0.4 MP, ~2.5x fewer ref tokens, "
                                          "the trained canvas)"}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                   "tooltip": "'match' scales refs to the generation's pixel "
                                              "area; 'max' uses the 2048px short edge"}),
            },
            "optional": {
                "ref_image_0": ("IMAGE",), "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",),
                "ref_video_0": ("IMAGE",), "ref_video_1": ("IMAGE",), "ref_video_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Builds MiniMax H3 reference conditioning (image + video ref latents) with NO "
        "text encoder, for use with Meridian baked embeds: wire MeridianFrozenPromptFile "
        "after it (conditioning + latent). Bypass the CLIPLoader."
    )

    def _video_canvas(self, vw, vh, ref_canvas):
        if ref_canvas == "meridian480":
            cw, ch = _bucket_480(vw, vh)
        else:
            cw, ch = _adapt_canvas(vw, vh)
        if vw * vh < cw * ch:  # never upscale beyond the render's native size
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        return cw, ch

    def build(self, vae, width, height, length, ref_canvas, ref_image_size,
              ref_image_0=None, ref_image_1=None, ref_image_2=None,
              ref_video_0=None, ref_video_1=None, ref_video_2=None):
        import comfy.model_management
        import comfy.nested_tensor

        frame_count, latent_t, audio_t = temporal_shape(length)
        device = comfy.model_management.intermediate_device()
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros([1, 24, latent_t, height // 16, width // 16], device=device),
            torch.zeros([1, 32, 2, audio_t], device=device)))}

        ref_blocks = []
        for image in (ref_image_0, ref_image_1, ref_image_2):
            if image is None:
                continue
            h, w = image.shape[1], image.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, (width * height / (w * h)) ** 0.5)
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(image[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                               "latent": z})

        for video_frames in (ref_video_0, ref_video_1, ref_video_2):
            if video_frames is None:
                continue
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = self._video_canvas(vw, vh, ref_canvas)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            ref_blocks.append({"kind": "video", "latent_t": z.shape[2], "latent_h": ch // 16,
                               "latent_w": cw // 16, "ref_audio_t": 0, "latent": z,
                               "audio_latent": None})

        print(f"MeridianRefConditioning: {len(ref_blocks)} ref block(s) at "
              f"{'meridian480' if ref_canvas == 'meridian480' else 'h3'} canvas "
              f"({width}x{height}, {length} frames) - no text encoder used")
        for block in ref_blocks:
            per_row = (block["latent_h"] // 2) * (block["latent_w"] // 2)
            rows = block.get("latent_t", 1)
            print(f"    {block['kind']}: canvas {block['latent_w'] * 16}x{block['latent_h'] * 16}, "
                  f"{rows} latent rows, {per_row} tokens/row = {per_row * rows} ref tokens")
        if ref_blocks and ref_canvas == "meridian480":
            print("    (ref_canvas only changes anything when the render is LARGER than the "
                  "480-class ladder entry; cond_render.mp4 already arrives at that size)")
        cond = [[torch.zeros([1, 1, 5120]), {"minimax_refs": ref_blocks}]]
        return (cond, latent)


NODE_CLASS_MAPPINGS = {"MeridianRefConditioning": MeridianRefConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianRefConditioning": "Meridian Ref Conditioning (no CLIP)"}
