@echo off
setlocal EnableDelayedExpansion
title ComfyUI Meridian for Gaussian Splatting - Setup
set "REPO_DIR=%~dp0"

echo ============================================================
echo   ComfyUI Meridian for Gaussian Splatting - Setup
echo ============================================================
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

set "EXTRA="
if not "%~1"=="" goto run
set /p "COMM=Install the public community node packs used by the example workflow (VHS / easy-use / various)? [y/N]: "
if /i "!COMM!"=="y" set "EXTRA=--install-community-nodes"

:run
"!PY!" "%REPO_DIR%setup\meridian_setup.py" --comfy-root "!ROOT!" !EXTRA!
echo.
pause
