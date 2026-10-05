"""ComfyUI Meridian for Gaussian Splatting - installer / distribution repository.

The node code lives in the **Enndees Nodepack** (single source of truth):

    https://github.com/Enndee/Enndees_Nodepack

Nothing is registered here on purpose: this repository only ships the one-shot
setup (``setup.bat`` / ``setup/meridian_setup.py``), the example workflows and
the documentation, so the nodes exist in exactly one place.

If you cloned this repository into ``ComfyUI\\custom_nodes\\`` that is harmless
(this module registers no nodes) - but the actual nodes come from the pack: run
``setup.bat`` (it clones the pack next to this folder) or clone
https://github.com/Enndee/Enndees_Nodepack into ``custom_nodes`` yourself.
"""

print("\033[96m[Enndee] ComfyUI_Meridian_for_Gaussian_Splatting: the nodes live in the "
      "Enndees Nodepack (https://github.com/Enndee/Enndees_Nodepack) - run setup.bat "
      "to install them\033[0m")

# Present but empty: the custom-node loader then does not warn about a missing
# NODE_CLASS_MAPPINGS, and no node ID is registered twice next to the pack.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

try:
    from glomap_lichtfeld_node import GLOMAPLichtfeldTracker
except Exception as _e:
    print(f"\033[31m[Enndee] GLOMAPLichtfeldTracker unavailable: {_e}\033[0m")
    GLOMAPLichtfeldTracker = None

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

# --- Meridian-specific node: the conditional prompt composer ---
try:
    from meridian_prompt_composer import MeridianPromptComposer
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianPromptComposer unavailable: {_e}\033[0m")
    MeridianPromptComposer = None

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

# WEB_DIRECTORY lets ComfyUI serve static files (web/js/...) - the adaptive
# widget panels for the parameters, image-loader and resolution nodes.
WEB_DIRECTORY = os.path.join(_pack_dir, "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]