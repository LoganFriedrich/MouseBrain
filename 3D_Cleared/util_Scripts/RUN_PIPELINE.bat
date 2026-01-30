@echo off
REM ============================================================================
REM  BRAINGLOBE PIPELINE LAUNCHER
REM
REM  Just double-click this file to start.
REM  No need to open command prompt or remember any commands.
REM ============================================================================

title BrainGlobe Pipeline

echo ============================================================
echo   BRAINGLOBE CELL DETECTION PIPELINE
echo ============================================================
echo.
echo Activating conda environment...

REM Try different conda locations
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call C:\ProgramData\anaconda3\Scripts\activate.bat brainglobe-env
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call %USERPROFILE%\anaconda3\Scripts\activate.bat brainglobe-env
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call %USERPROFILE%\miniconda3\Scripts\activate.bat brainglobe-env
) else (
    echo.
    echo ERROR: Could not find conda!
    echo.
    echo Please:
    echo   1. Open Anaconda Prompt manually
    echo   2. Type: conda activate brainglobe-env
    echo   3. Type: cd %~dp0
    echo   4. Type: python RUN_PIPELINE.py
    echo.
    pause
    exit /b 1
)

echo Environment activated!
echo.

REM Run the pipeline
cd /d "%~dp0"
python RUN_PIPELINE.py

echo.
echo ============================================================
echo Pipeline session ended.
echo ============================================================
pause
