@echo off
setlocal
title ComfyUI Meridian for Gaussian Splatting - Setup
set "REPO_DIR=%~dp0"

rem Prefer the portable ComfyUI python next to this repository
rem (custom_nodes\<repo>\ -> ..\..\..\python_embeded\python.exe).
set "PY=%REPO_DIR%..\..\..\python_embeded\python.exe"
if exist "%PY%" goto run

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Portable ComfyUI was not found next to this repository and there is no
  echo python on PATH. Run this file from
  echo    ...\ComfyUI\custom_nodes\ComfyUI_Meridian_for_Gaussian_Splatting
  echo or pass the root manually, for example:
  echo    setup.bat --comfy-root "D:\ComfyUI_Windows_portable\ComfyUI_windows_portable"
  echo.
  pause
  exit /b 1
)
set "PY=python"

:run
echo Using %PY%
echo.
"%PY%" "%REPO_DIR%setup\meridian_setup.py" %*
echo.
pause
