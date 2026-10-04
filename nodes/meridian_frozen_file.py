# SPDX-License-Identifier: MIT
"""Meridian frozen prompt with an explicit file - use a baked embed.

Same substitution as Meridian's ``MeridianFrozenPrompt``, but the file name comes
from a widget instead of being derived from the output length. Pair it with
``MeridianBakeEmbed`` to keep the Meridian adapters on their trained text-only
presentation while using a NEW prompt (e.g. an image2video prompt that refers to
``<Picture 1>`` instead of ``<Video 1>``).

Wire it between ``MiniMaxH3ReferenceToVideo`` and the sampler, exactly like the
original frozen prompt node.
"""
import os

import torch


class MeridianFrozenPromptFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "assets_dir": ("STRING", {"default": "assets"}),
            "file": ("STRING", {"default": "fixed_embed_{frames}_i2v.pt"}),
        }, "optional": {
            "latent": ("LATENT",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Replaces the text tensor and minimax_token_tags of H3 reference "
        "conditioning with a specific baked embed file (see MeridianBakeEmbed). "
        "`{frames}` in the file name is filled from the latent's frame count, so "
        "one node follows the workflow's length setting."
    )

    def apply(self, conditioning, assets_dir, file, latent=None):
        if "{frames}" in file:
            if latent is None:
                raise ValueError("MeridianFrozenPromptFile: the file name uses {frames}; "
                                 "connect the latent input")
            samples = latent["samples"]
            video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
            file = file.format(frames=(video.shape[2] - 2) // 5 * 17 + 5)
        path = file if os.path.isabs(file) else os.path.join(assets_dir, file)
        embed = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Meridian: {os.path.basename(path)} -> "
              f"{embed['prompt_embeds'].shape[1]} frozen text tokens")
        return ([[embed["prompt_embeds"], {**d, "minimax_token_tags": embed["text_token_tags"]}]
                 for _, d in conditioning],)


NODE_CLASS_MAPPINGS = {"MeridianFrozenPromptFile": MeridianFrozenPromptFile}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianFrozenPromptFile": "Meridian Frozen Prompt (file)"}
