# SPDX-License-Identifier: MIT
"""Meridian's exact H3 sigma grid for ComfyUI (for the DMD student's few-step regime).

Meridian's pipeline samples on a uniform-in-timestep grid that ComfyUI's stock schedulers
can only approximate at a shifted point count: diffusers' MiniMaxH3Scheduler.set_timesteps
uses ``linspace(1, 0, steps)`` (steps - 1 model evaluations, terminal 0 collapsed), while
ComfyUI's "simple"/"normal" schedules use ``linspace(1, 0, KSampler steps + 1)``. So

    Meridian --steps 4  ==  ComfyUI KSampler steps 3   (3 evaluations, the DMD student's grid)
    Meridian --steps 10 ==  ComfyUI KSampler steps 9
    Meridian --steps 50 ==  ComfyUI KSampler steps 49  (the teacher's grid)

Grids (sigma values) the student was DMD-distilled for, i.e. Meridian's:

    steps=4,  shift=3   -> [1.0, 0.8571, 0.6, 0.0]      (3 model evaluations)
    steps=4,  shift=12  -> [1.0, 0.96,   0.8571, 0.0]
    steps=10, shift=12  -> [1.0, .9897, .9767, .96, .9375, .9057, .8571, .7742, .6, 0.0]
    steps=50, shift=12  -> the teacher's fine grid

This node emits those exact sigmas for SamplerCustomAdvanced (+ KSamplerSelect "euler",
CFGGuider cfg=1, RandomNoise) - useful to run a grid the stock schedulers cannot reach, or
to verify the KSampler-steps - 1 mapping above.
"""

import torch


class MeridianH3Sigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "steps": ("INT", {"default": 4, "min": 2, "max": 200, "step": 1,
                              "tooltip": "Grid points including the terminal 0; the student's "
                                         "3-forward grid is steps=4, teacher 50"}),
            "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01,
                                "tooltip": "3.0 for the DMD student, 12.0 for the recam teacher"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/sigmas"
    DESCRIPTION = (
        "Meridian's exact MiniMax-H3 sigma grid (uniform in timestep, shifted, duplicates "
        "collapsed) for SamplerCustomAdvanced - use it to run the Meridian turbo (DMD) "
        "student on the grid it was distilled for."
    )

    def build(self, steps, shift):
        base = torch.linspace(1.0, 0.0, steps, dtype=torch.float32)
        sigmas = shift * base / (1.0 + (shift - 1.0) * base)
        sigmas = torch.unique_consecutive(sigmas)
        print(f"MeridianH3Sigmas: steps={steps} shift={shift} -> {len(sigmas)} model "
              f"evaluations, sigmas {[round(float(value), 4) for value in sigmas][:12]}"
              f"{' ...' if len(sigmas) > 12 else ''}")
        return (sigmas,)


NODE_CLASS_MAPPINGS = {"MeridianH3Sigmas": MeridianH3Sigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianH3Sigmas": "Meridian H3 Sigmas (student/teacher grid)"}
