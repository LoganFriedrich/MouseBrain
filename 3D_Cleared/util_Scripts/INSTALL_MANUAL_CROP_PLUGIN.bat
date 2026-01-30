@echo off
echo ============================================================
echo Installing napari-manual-crop plugin
echo ============================================================
echo.

call conda activate brainglobe-env
cd /d "%~dp0napari_manual_crop"

echo Uninstalling old version (if exists)...
pip uninstall -y napari-manual-crop 2>nul

echo.
echo Installing plugin...
pip install -e .

echo.
echo ============================================================
echo Testing installation...
echo ============================================================
python test_install.py

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo You can now use the plugin:
echo   1. CLI: python util_manual_crop.py --brain BRAIN_NAME
echo   2. GUI: Open napari, then Plugins -^> Manual Crop Tool
echo.
pause
