# SPDX-License-Identifier: MIT
"""Bake a Meridian-compatible frozen text embedding from any H3 reference graph.

Meridian's adapters were trained with a text-only presentation: the reference
videos reach the model as DiT condition rows and the text side is a precomputed
``fixed_embed_{n}.pt`` (the last-layer Qwen3-VL hidden states of the ``<Video 1>``
/ ``<Video 2>`` prompt, with timestamp markers). Change the prompt (e.g. to an
image2video prompt that refers to ``<Picture 1>``) and the shipped embeds no
longer match - this node saves a matching replacement so later runs skip the
25 GB text encoder entirely.

How to use it (see the i2v variant workflow):

1. bypass ``MeridianFrozenPrompt``, wire this node after
   ``MiniMaxH3ReferenceToVideo`` (conditioning + latent), set ``assets_dir`` to
   the Meridian ``assets/`` folder and ``tag`` to something like ``i2v``,
2. queue once - the tensor is written to
   ``assets_dir/fixed_embed_<frames>_<tag>.pt``,
3. bypass this node and use ``MeridianFrozenPromptFile`` with that file name.
"""
import os

import torch


class MeridianBakeEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "latent": ("LATENT",),
            "assets_dir": ("STRING", {"default": "assets"}),
            "tag": ("STRING", {"default": "i2v"}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "bake"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Saves the text tensor (+ minimax token tags) of H3 reference "
        "conditioning as a Meridian fixed_embed file. Pass-through, so it can "
        "sit inline after MiniMaxH3ReferenceToVideo while the frozen prompt "
        "node is bypassed."
    )

    def bake(self, conditioning, latent, assets_dir, tag, overwrite):
        samples = latent["samples"]
        video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        frames = (video.shape[2] - 2) // 5 * 17 + 5  # inverse of H3's video_latent_t

        tensor, tags = None, None
        for entry in conditioning:
            if not isinstance(entry[0], torch.Tensor):
                continue
            extra = entry[1] if isinstance(entry[1], dict) else {}
            if extra.get("minimax_token_tags") is not None:
                tensor, tags = entry[0], extra["minimax_token_tags"]
                break
        if tensor is None or tags is None:
            raise ValueError(
                "MeridianBakeEmbed found no tensor/minimax_token_tags in the conditioning; "
                "is the clip a MiniMax H3 text encoder?")

        name = f"fixed_embed_{frames}_{tag}.pt"
        target = os.path.join(assets_dir, name)
        if os.path.isfile(target) and not overwrite:
            print(f"MeridianBakeEmbed: {target} already exists (enable overwrite to replace it)")
            return (conditioning,)
        payload = {"prompt_embeds": tensor.detach().to("cpu"), "text_token_tags": tags}
        torch.save(payload, target)
        tag_len = len(tags) if hasattr(tags, "__len__") else "?"
        print(f"MeridianBakeEmbed: saved {target} (tensor {tuple(payload['prompt_embeds'].shape)}, "
              f"tags {tag_len})")
        return (conditioning,)


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24


def _stretch(samples, width, height, crop):
    """Torch-only copy of comfy.utils.common_upscale's stretch/center-crop geometry."""
    if crop == "center":
        b, c, h, w = samples.shape
        scale = max(height / h, width / w)
        nh, nw = max(height, round(h * scale)), max(width, round(w * scale))
        samples = torch.nn.functional.interpolate(samples, size=(nh, nw), mode="bilinear",
                                                  align_corners=False)
        top, left = (nh - height) // 2, (nw - width) // 2
        return samples[:, :, top:top + height, left:left + width]
    return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear",
                                           align_corners=False)


def _resize(image, width, height, crop="disabled"):
    """The H3 reference node's resize: comfy.utils.common_upscale when running inside
    ComfyUI, a torch fallback outside it (tests)."""
    samples = image[..., :3].movedim(-1, 1)
    try:
        import comfy.utils
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    except Exception:
        samples = _stretch(samples, width, height, crop)
    return samples.movedim(1, -1)


def _adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32 (core node's)."""
    import math
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


class MeridianBakeAllEmbeds:
    """Bake `fixed_embed_<frames>_<tag>.pt` for every supported output length in one run.

    Replicates `MiniMaxH3ReferenceToVideo`'s presentation exactly: the image ref as
    `<Picture 1>` (resized like `ref_image_size`), the video ref as `<Video 1>` cropped to
    the output length, aligned to the 17k+5 grid, sampled at 2 fps with timestamps. Only
    the CLIP model + this node run - no VAE, no transformer, no sampler.

    The video ref source can be a single render frame (repeated) or the full render batch;
    what the text embed needs from it is the frame COUNT and the rough content.
    """

    LENGTHS = "73,90,107,124,141,158,175,243"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "assets_dir": ("STRING", {"default": "assets"}),
                "tag": ("STRING", {"default": "i2v"}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "lengths": ("STRING", {"default": cls.LENGTHS}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {"video": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "bake"
    OUTPUT_NODE = True  # runs even with nothing wired after it: the file is the product
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Bakes Meridian frozen embeds for all output lengths in one run (CLIP only): "
        "image -> <Picture 1>, video -> <Video 1>, sampled exactly like the H3 "
        "reference node, saved as assets/fixed_embed_<frames>_<tag>.pt."
    )

    def _image_item(self, image, width, height, ref_image_size):
        h, w = image.shape[1], image.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, (width * height / (w * h)) ** 0.5)
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        return {"type": "image", "data": _resize(image[:1], tw, th, "disabled")}

    def _video_item(self, source, length):
        """Crop/tile the render source to `length`, align to 17k+5, sample at 2 fps."""
        frames = source
        if frames.shape[0] < length:
            repeat = -(-length // frames.shape[0])
            frames = frames.repeat(repeat, 1, 1, 1)
        frames = frames[:length]
        vh, vw = frames.shape[1], frames.shape[2]
        cw, ch = _adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(frames, cw, ch, "disabled")
        n = frames.shape[0]
        if n < 5:
            raise ValueError("the video reference needs at least 5 frames")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        return {"type": "video", "data": frames[sample_idx],
                "timestamps": [i / 2.0 for i in range(len(sample_idx))]}

    def bake(self, clip, image, prompt, assets_dir, tag, width, height, ref_image_size,
             lengths, overwrite, video=None):
        import os
        wanted = [int(value) for value in str(lengths).replace(";", ",").split(",") if value.strip()]
        report = []
        for length in wanted:
            target = os.path.join(assets_dir, f"fixed_embed_{length}_{tag}.pt")
            if os.path.isfile(target) and not overwrite:
                report.append(f"{length}: exists (skipped)")
                continue
            ref_items = [self._image_item(image, width, height, ref_image_size),
                         self._video_item(video if video is not None else image, length)]
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            tensor, extra = cond[0][0], cond[0][1]
            tags = extra.get("minimax_token_tags") if isinstance(extra, dict) else None
            if tags is None:
                raise ValueError("the clip did not return minimax_token_tags; "
                                 "is it a MiniMax H3 text encoder?")
            torch.save({"prompt_embeds": tensor.detach().to("cpu"),
                        "text_token_tags": tags}, target)
            report.append(f"{length}: {tensor.shape[1]} tokens -> {os.path.basename(target)}")
            print(f"MeridianBakeAllEmbeds: {length} frames -> {tensor.shape[1]} text tokens "
                  f"({os.path.basename(target)})")
        return ("\n".join(report),)


class MeridianBakeTextEmbeds:
    """Bake Viggle-style *text-only* frozen embeds for every output length in one run.

    Decoded from the shipped `assets/fixed_embed_*.pt`: Viggle's presentation is plain
    text - `<Video k>: ` labels followed by `<X.Y seconds>` markers (one per 2-fps frame
    pair) and then the prompt - tokenized with NO vision entries, so `text_token_tags`
    is all 1s and the reference pixels reach the model only as DiT condition rows.
    This node reproduces that presentation for any prompt/label layout, e.g.
    `Picture 1, Video 1` for the image2video variant, and saves the same five keys
    (`num_frames`, `presentation`, `prompt_embeds`, `text_token_tags`, `token_ids`).
    """

    LENGTHS = "73,90,107,124,141,158,175,243"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "refs": ("STRING", {"default": "Picture 1, Video 1"}),
                "assets_dir": ("STRING", {"default": "assets"}),
                "tag": ("STRING", {"default": "i2v"}),
                "lengths": ("STRING", {"default": cls.LENGTHS}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "bake"
    OUTPUT_NODE = True  # runs even with nothing wired after it: the files are the product
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Bakes text-only Meridian frozen embeds for all output lengths (CLIP only, no "
        "vision tokens), like Viggle's shipped assets. `refs` lists the labels in "
        "reference order; labels starting with 'Video' get 2-fps timestamp markers."
    )

    @staticmethod
    def presentation(refs, length, prompt):
        labels = [label.strip() for label in refs.split(",") if label.strip()]
        samples = -(-length // (FPS // 2))
        head = []
        for label in labels:
            text = f"{label}: "
            if label.lower().startswith("video"):
                for start in range(0, samples, 2):
                    first = start / 2.0
                    second = (start + 1) / 2.0 if start + 1 < samples else first
                    text += f"<{(first + second) / 2.0:.1f} seconds>"
            head.append(text)
        return "".join(head) + prompt

    def bake(self, clip, prompt, refs, assets_dir, tag, lengths, overwrite):
        import os
        wanted = [int(value) for value in str(lengths).replace(";", ",").split(",") if value.strip()]
        report = []
        for length in wanted:
            target = os.path.join(assets_dir, f"fixed_embed_{length}_{tag}.pt")
            if os.path.isfile(target) and not overwrite:
                report.append(f"{length}: exists (skipped)")
                continue
            presentation = self.presentation(refs, length, prompt)
            tokens = clip.tokenize(presentation)
            cond = clip.encode_from_tokens_scheduled(tokens)
            tensor, extra = cond[0][0], cond[0][1]
            tags = extra.get("minimax_token_tags") if isinstance(extra, dict) else None
            if tags is None:
                raise ValueError("the clip did not return minimax_token_tags; "
                                 "is it a MiniMax H3 text encoder?")
            entries = tokens["qwen3vl_32b"][0]
            ids = [token for token, _ in entries if isinstance(token, int)]
            torch.save({"num_frames": length, "presentation": presentation,
                        "prompt_embeds": tensor.detach().to("cpu"),
                        "text_token_tags": tags,
                        "token_ids": torch.tensor(ids, dtype=torch.long)}, target)
            blocks = presentation.count(" seconds>")
            report.append(f"{length}: {tensor.shape[1]} tokens, {blocks} timestamp blocks "
                          f"-> {os.path.basename(target)}")
            print(f"MeridianBakeTextEmbeds: {length} frames -> {tensor.shape[1]} text tokens, "
                  f"{blocks} blocks ({os.path.basename(target)})")
        return ("\n".join(report),)


NODE_CLASS_MAPPINGS = {
    "MeridianBakeEmbed": MeridianBakeEmbed,
    "MeridianBakeAllEmbeds": MeridianBakeAllEmbeds,
    "MeridianBakeTextEmbeds": MeridianBakeTextEmbeds,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeridianBakeEmbed": "Meridian Bake Frozen Embed",
    "MeridianBakeAllEmbeds": "Meridian Bake Frozen Embeds (all lengths)",
    "MeridianBakeTextEmbeds": "Meridian Bake Frozen Embeds (text-only, all lengths)",
}
