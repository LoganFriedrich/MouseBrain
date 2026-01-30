@echo off
REM Install all SCI-Connectome napari plugins
REM Run this after activating brainglobe-env

echo ============================================================
echo Installing SCI-Connectome napari plugins...
echo ============================================================

cd /d "%~dp0"

echo.
echo Installing Pipeline Dashboard...
pip install -e sci_pipeline

echo.
echo Installing Tuning Tools...
pip install -e sci_tuning

echo.
echo Installing Manual Crop (from original location)...
pip install -e ..\napari_manual_crop

echo.
echo ============================================================
echo Done! Restart napari to see the plugins.
echo ============================================================
pause
