"""
Enndees Nodepack - Custom Nodes for ComfyUI
GLOMAP Camera Tracker with advanced mask handling for Lichtfeld Studio.
"""

import sys
import os

print("\033[96m[Enndee] Enndees Nodepack loaded\033[0m")

# Add this directory and nodes/ to path for imports.
# The COLMAP/GLOMAP helpers are vendored (enndee_colmap/) so the pack is
# fully self contained - no external comfyui_colmap is needed.
_pack_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _pack_dir)
sys.path.insert(0, os.path.join(_pack_dir, "nodes"))

try:
    from glomap_lichtfeld_node import GLOMAPLichtfeldTracker
except Exception as _e:
    print(f"\033[31m[Enndee] GLOMAPLichtfeldTracker unavailable: {_e}\033[0m")
    GLOMAPLichtfeldTracker = None

try:
    from enndee_video_frame_extractor import Enndee_VideoFrameExtractorWithAudio
except Exception as _e:
    print(f"\033[31m[Enndee] VideoFrameExtractorWithAudio unavailable: {_e}\033[0m")
    Enndee_VideoFrameExtractorWithAudio = None

try:
    from minimax_h3_promptor import H3_Multimodal_Promptor_Enndee
except Exception as _e:
    print(f"\033[31m[Enndee] H3_Multimodal_Promptor_Enndee unavailable: {_e}\033[0m")
    H3_Multimodal_Promptor_Enndee = None

try:
    from enndee_resolution_selector import ResolutionSelectorEnndee
except Exception as _e:
    print(f"\033[31m[Enndee] ResolutionSelectorEnndee unavailable: {_e}\033[0m")
    ResolutionSelectorEnndee = None

try:
    from enndee_image_loader import ImageLoaderResizeEnndee
except Exception as _e:
    print(f"\033[31m[Enndee] ImageLoaderResizeEnndee unavailable: {_e}\033[0m")
    ImageLoaderResizeEnndee = None

try:
    from enndee_meridian_parameters import MeridianParametersAndCamera
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianParametersAndCamera unavailable: {_e}\033[0m")
    MeridianParametersAndCamera = None

try:
    from enndee_meridian_geometry import EnndeeMeridianGeometry
except Exception as _e:
    print(f"\033[31m[Enndee] EnndeeMeridianGeometry unavailable: {_e}\033[0m")
    EnndeeMeridianGeometry = None

try:
    from lichtfeld_training_node import LichtfeldHeadlessTrainer
except Exception as _e:
    print(f"\033[31m[Enndee] LichtfeldHeadlessTrainer unavailable: {_e}\033[0m")
    LichtfeldHeadlessTrainer = None

try:
    from enndee_standby_signal import Enndee_StandbyOnSignal
except Exception as _e:
    print(f"\033[31m[Enndee] Enndee_StandbyOnSignal unavailable: {_e}\033[0m")
    Enndee_StandbyOnSignal = None

# --- Meridian-specific nodes (conditional prompt, frozen-embed tooling) ---
try:
    from meridian_prompt_composer import MeridianPromptComposer
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianPromptComposer unavailable: {_e}\033[0m")
    MeridianPromptComposer = None

try:
    from meridian_geometry import MeridianGeometry
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianGeometry unavailable: {_e}\033[0m")
    MeridianGeometry = None

try:
    from meridian_ref_conditioning import MeridianRefConditioning
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianRefConditioning unavailable: {_e}\033[0m")
    MeridianRefConditioning = None

try:
    from meridian_h3_sigmas import MeridianH3Sigmas
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianH3Sigmas unavailable: {_e}\033[0m")
    MeridianH3Sigmas = None

try:
    from meridian_embed import MeridianFrozenPrompt
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianFrozenPrompt unavailable: {_e}\033[0m")
    MeridianFrozenPrompt = None

try:
    from meridian_frozen_file import MeridianFrozenPromptFile
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianFrozenPromptFile unavailable: {_e}\033[0m")
    MeridianFrozenPromptFile = None

try:
    from meridian_embed_baker import (
        MeridianBakeEmbed,
        MeridianBakeAllEmbeds,
        MeridianBakeTextEmbeds,
    )
except Exception as _e:
    print(f"\033[31m[Enndee] Meridian embed bakers unavailable: {_e}\033[0m")
    MeridianBakeEmbed = None
    MeridianBakeAllEmbeds = None
    MeridianBakeTextEmbeds = None

# Global save-behavior hook (no node): strip ComfyUI's running counter and
# number files only when the target name is already taken. Opt out with the
# environment variable ENNDEE_KEEP_FILE_COUNTER=1.
try:
    from enndee_unique_filenames import install as install_unique_filenames

    install_unique_filenames()
except Exception as _e:
    print(f"\033[31m[Enndee] unique filenames unavailable: {_e}\033[0m")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if GLOMAPLichtfeldTracker is not None:
    NODE_CLASS_MAPPINGS["Enndee_GLOMAPLichtfeldTracker"] = GLOMAPLichtfeldTracker
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_GLOMAPLichtfeldTracker"] = "GLOMAP Lichtfeld Tracker (Enndee)"

# WEB_DIRECTORY lets ComfyUI serve static files (web/js/...).
# The timeline widget for Enndee_VideoFrameExtractorWithAudio lives there.
if Enndee_VideoFrameExtractorWithAudio is not None:
    NODE_CLASS_MAPPINGS["Enndee_VideoFrameExtractorWithAudio"] = Enndee_VideoFrameExtractorWithAudio
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_VideoFrameExtractorWithAudio"] = "Video Frame Extractor + Audio (Enndee)"

if H3_Multimodal_Promptor_Enndee is not None:
    NODE_CLASS_MAPPINGS["H3_Multimodal_Promptor_Enndee"] = H3_Multimodal_Promptor_Enndee
    NODE_DISPLAY_NAME_MAPPINGS["H3_Multimodal_Promptor_Enndee"] = "MiniMax H3 Direct Promptor (Enndee)"

if ResolutionSelectorEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_ResolutionSelector"] = ResolutionSelectorEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_ResolutionSelector"] = "Resolution Selector (Enndee)"

if ImageLoaderResizeEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_ImageLoaderResize"] = ImageLoaderResizeEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_ImageLoaderResize"] = "Load & Resize Image (Enndee)"

if MeridianParametersAndCamera is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianParametersAndCamera"] = MeridianParametersAndCamera
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianParametersAndCamera"] = "Meridian Parameters and Camera (Enndee)"

if EnndeeMeridianGeometry is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianGeometry"] = EnndeeMeridianGeometry
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianGeometry"] = "Meridian Geometry (Enndee)"

if LichtfeldHeadlessTrainer is not None:
    NODE_CLASS_MAPPINGS["Enndee_LichtfeldHeadlessTrainer"] = LichtfeldHeadlessTrainer
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_LichtfeldHeadlessTrainer"] = "Lichtfeld Headless Trainer (Enndee)"

if Enndee_StandbyOnSignal is not None:
    NODE_CLASS_MAPPINGS["Enndee_StandbyOnSignal"] = Enndee_StandbyOnSignal
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_StandbyOnSignal"] = "Standby On Signal (Enndee)"

if MeridianPromptComposer is not None:
    NODE_CLASS_MAPPINGS["MeridianPromptComposer"] = MeridianPromptComposer
    NODE_DISPLAY_NAME_MAPPINGS["MeridianPromptComposer"] = "Meridian Prompt Composer (conditional pictures)"

if MeridianGeometry is not None:
    NODE_CLASS_MAPPINGS["MeridianGeometry"] = MeridianGeometry
    NODE_DISPLAY_NAME_MAPPINGS["MeridianGeometry"] = "Meridian Geometry"

if MeridianRefConditioning is not None:
    NODE_CLASS_MAPPINGS["MeridianRefConditioning"] = MeridianRefConditioning
    NODE_DISPLAY_NAME_MAPPINGS["MeridianRefConditioning"] = "Meridian Ref Conditioning (no CLIP)"

if MeridianH3Sigmas is not None:
    NODE_CLASS_MAPPINGS["MeridianH3Sigmas"] = MeridianH3Sigmas
    NODE_DISPLAY_NAME_MAPPINGS["MeridianH3Sigmas"] = "Meridian H3 Sigmas (student/teacher grid)"

if MeridianFrozenPrompt is not None:
    NODE_CLASS_MAPPINGS["MeridianFrozenPrompt"] = MeridianFrozenPrompt
    NODE_DISPLAY_NAME_MAPPINGS["MeridianFrozenPrompt"] = "Meridian Frozen Prompt"

if MeridianFrozenPromptFile is not None:
    NODE_CLASS_MAPPINGS["MeridianFrozenPromptFile"] = MeridianFrozenPromptFile
    NODE_DISPLAY_NAME_MAPPINGS["MeridianFrozenPromptFile"] = "Meridian Frozen Prompt (file)"

if MeridianBakeEmbed is not None:
    NODE_CLASS_MAPPINGS["MeridianBakeEmbed"] = MeridianBakeEmbed
    NODE_DISPLAY_NAME_MAPPINGS["MeridianBakeEmbed"] = "Meridian Bake Frozen Embed"

if MeridianBakeAllEmbeds is not None:
    NODE_CLASS_MAPPINGS["MeridianBakeAllEmbeds"] = MeridianBakeAllEmbeds
    NODE_DISPLAY_NAME_MAPPINGS["MeridianBakeAllEmbeds"] = "Meridian Bake Frozen Embeds (all lengths)"

if MeridianBakeTextEmbeds is not None:
    NODE_CLASS_MAPPINGS["MeridianBakeTextEmbeds"] = MeridianBakeTextEmbeds
    NODE_DISPLAY_NAME_MAPPINGS["MeridianBakeTextEmbeds"] = "Meridian Bake Frozen Embeds (text-only, all lengths)"

WEB_DIRECTORY = os.path.join(_pack_dir, "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]