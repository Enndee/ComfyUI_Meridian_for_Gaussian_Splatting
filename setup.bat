@echo off
setlocal EnableDelayedExpansion
title ComfyUI Meridian for Gaussian Splatting - Setup
set "REPO_DIR=%~dp0"

echo ============================================================
echo   ComfyUI Meridian for Gaussian Splatting - Setup
echo ============================================================
echo.
echo   Geometry runs through fast depth (Depth Anything v3, installed into
echo   ComfyUI's python) - no VGGT checkout, no checkpoint, no extra env.
echo.
echo   Which ComfyUI folder should be prepared? Both answers work:
echo     - the folder containing "ComfyUI" and "python_embeded"
echo       e.g.  D:\ComfyUI_windows_portable\ComfyUI_windows_portable
echo     - its parent folder, e.g.  D:\ComfyUI_windows_portable
echo.
set "GUESS=%REPO_DIR%..\..\.."
if not "%~1"=="" (
  set "COMFY_IN=%~1"
) else (
  set /p "COMFY_IN=ComfyUI folder [%GUESS%]: "
  if "!COMFY_IN!"=="" set "COMFY_IN=%GUESS%"
)

rem Resolve the portable root (the folder that has ComfyUI\main.py).
set "ROOT=!COMFY_IN!"
if exist "!ROOT!\ComfyUI\main.py" goto root_ok
if exist "!ROOT!\ComfyUI_windows_portable\ComfyUI\main.py" (
  set "ROOT=!ROOT!\ComfyUI_windows_portable"
  goto root_ok
)
for /d %%D in ("!COMFY_IN!\ComfyUI*") do (
  if exist "%%D\ComfyUI\main.py" set "ROOT=%%D"
)
if exist "!ROOT!\ComfyUI\main.py" goto root_ok

echo.
echo   Could not find "ComfyUI\main.py" below "!COMFY_IN!".
echo   Please run this file again and give the folder that contains the
echo   ComfyUI and python_embeded folders.
echo.
pause
exit /b 1

:root_ok
set "PY=!ROOT!\python_embeded\python.exe"
if not exist "!PY!" set "PY=python"

echo.
echo   ComfyUI root : !ROOT!
echo   Python       : !PY!
echo.

rem Collect every argument after the first (the first one is the ComfyUI
rem folder) and forward them to the setup script - e.g.
rem   setup.bat "D:\ComfyUI" --meridian-zip "X:\Meridian_release.zip" --check
set "FWD="
shift
:collect_args
if "%~1"=="" goto args_done
set FWD=!FWD! "%~1"
shift
goto collect_args
:args_done

set "EXTRA="
if not "%FWD%"=="" goto run
echo.
echo   Step 1 clones the Enndees Nodepack - that is where the nodes live.
echo   Step 7 clones the community packs the example workflow uses:
echo     Sharp-Selector, rgthree, KJNodes, various, custom-scripts, VHS,
echo     easy-use, Pixaroma, BRIA RMBG, RTX nodes, UniBlockSwap, Memory-Cleanup,
echo     H3 MotionCache, H3 Turbo, H3 latent upscaler.
set /p "COMM=Skip the community node packs (step 7)? [y/N]: "
if /i "!COMM!"=="y" set "EXTRA=--skip-community-nodes"

:run
"!PY!" "%REPO_DIR%setup\meridian_setup.py" --comfy-root "!ROOT!" !FWD! !EXTRA!
echo.
pause
