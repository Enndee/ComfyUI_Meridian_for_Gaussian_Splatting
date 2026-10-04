# Copyright 2026 Viggle AI. Licensed under the Apache License, Version 2.0 (see LICENSE-CODE).
# SPDX-License-Identifier: Apache-2.0
"""A ComfyUI node that swaps in Meridian's frozen text conditioning.

Drop this file into `ComfyUI/custom_nodes/` and **Meridian Frozen Prompt** appears under `conditioning`.
Wire it between `MiniMaxH3ReferenceToVideo` and the sampler; leave the rest of the graph alone.

Why it is needed. Meridian was trained on a text-only presentation: `assets/fixed_embed_{n}.pt` holds
`hidden_states[50]` of Qwen3-VL for the `<Video 1>` / `<Video 2>` text with timestamp markers and no
pixels, and the two reference videos reach the model only as DiT condition rows. ComfyUI's reference
node instead samples both videos at 2 fps and feeds those frames to Qwen3-VL, so its text conditioning
carries vision tokens these adapters never saw. This node replaces that tensor and its
`minimax_token_tags` with the frozen pair and keeps everything else in the conditioning, `minimax_refs`
included.

The two sides meet at the same tensor: `layer="last"` on ComfyUI's text encoder, whose stack is
truncated to 50 layers, with `layer_norm_hidden_state=False`, is `hidden_states[50]` unnormalised --
which is what `recam/make_embed.py` saved. Both paths then run the model's own `condition_proj` and
token refiner, so the substitution happens one stage before anything H3-specific.

`assets_dir` is this repo's `assets/` (an absolute path if ComfyUI does not run from here). The clip
length is read off the latent rather than typed again, so the embedding cannot disagree with what is
being sampled.
"""
import os

import torch


class MeridianFrozenPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "latent": ("LATENT",),
                             "assets_dir": ("STRING", {"default": "assets"})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning"

    def apply(self, conditioning, latent, assets_dir):
        samples = latent["samples"]
        video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        frames = (video.shape[2] - 2) // 5 * 17 + 5  # the inverse of H3's video_latent_t
        embed = torch.load(os.path.join(assets_dir, f"fixed_embed_{frames}.pt"),
                           map_location="cpu", weights_only=True)
        print(f"Meridian: {frames} frames -> {embed['prompt_embeds'].shape[1]} frozen text tokens")
        return ([[embed["prompt_embeds"], {**d, "minimax_token_tags": embed["text_token_tags"]}]
                 for _, d in conditioning],)


NODE_CLASS_MAPPINGS = {"MeridianFrozenPrompt": MeridianFrozenPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianFrozenPrompt": "Meridian Frozen Prompt"}
