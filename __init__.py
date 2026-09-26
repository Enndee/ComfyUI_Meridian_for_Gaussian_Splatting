"""
ComfyUI_Meridian_for_Gaussian_Splatting - Meridian camera-path nodes (Enndee).

Registers the three Meridian nodes and serves web/js (the adaptive widget
extension of the picker):

- Meridian Parameter Picker (Enndee)      Enndee_MeridianParameterPicker
- Meridian Camera Path Configurator (Enndee) Enndee_MeridianCameraPath
- Meridian Geometry (Enndee)              Enndee_MeridianGeometry
"""

import os
import sys

print("\033[96m[Enndee Meridian] ComfyUI_Meridian_for_Gaussian_Splatting loaded\033[0m")

# Make nodes/ importable so the node files can use absolute sibling imports.
_pack_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _pack_dir)
sys.path.insert(0, os.path.join(_pack_dir, "nodes"))

try:
    from enndee_meridian_parameter_picker import MeridianParameterPickerEnndee
except Exception as _e:
    print(f"\033[31m[Enndee Meridian] MeridianParameterPickerEnndee unavailable: {_e}\033[0m")
    MeridianParameterPickerEnndee = None

try:
    from enndee_meridian_camera_path import MeridianCameraPathConfigurator
except Exception as _e:
    print(f"\033[31m[Enndee Meridian] MeridianCameraPathConfigurator unavailable: {_e}\033[0m")
    MeridianCameraPathConfigurator = None

try:
    from enndee_meridian_geometry import EnndeeMeridianGeometry
except Exception as _e:
    print(f"\033[31m[Enndee Meridian] EnndeeMeridianGeometry unavailable: {_e}\033[0m")
    EnndeeMeridianGeometry = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if MeridianParameterPickerEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianParameterPicker"] = MeridianParameterPickerEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianParameterPicker"] = "Meridian Parameter Picker (Enndee)"

if MeridianCameraPathConfigurator is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianCameraPath"] = MeridianCameraPathConfigurator
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianCameraPath"] = "Meridian Camera Path Configurator (Enndee)"

if EnndeeMeridianGeometry is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianGeometry"] = EnndeeMeridianGeometry
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianGeometry"] = "Meridian Geometry (Enndee)"

# WEB_DIRECTORY lets ComfyUI serve web/js/... (the picker's adaptive widgets).
WEB_DIRECTORY = os.path.join(_pack_dir, "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
