@echo off
REM Thumbs.db Destroyer - Double-click to nuke thumbs.db files
REM Will scan the folder where this batch file is located

cd /d "%~dp0"
echo ============================================================
echo THUMBS.DB DESTROYER
echo ============================================================
echo.
echo This will destroy all thumbs.db files in this folder and subfolders.
echo.
pause

python util_thumbs_destroyer.py "%~dp0" --prevent

echo.
echo ============================================================
echo Done! Decoy files have been created to prevent recreation.
echo ============================================================
pause
