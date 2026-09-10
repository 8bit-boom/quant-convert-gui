@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Quant Convert GUI - updater
echo ============================================================
echo.

where git >nul 2>nul
if %errorlevel%==0 (
    if exist "%~dp0.git" (
        echo Pulling latest changes from git...
        git pull
        echo.
    ) else (
        echo (This folder isn't a git checkout - skipping source update.
        echo  Download the latest release/zip to get code updates.)
        echo.
    )
) else (
    echo (git not found on PATH - skipping source update.)
    echo.
)

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo No virtual environment found - run install.bat first.
    pause
    exit /b 1
)

echo Upgrading GUI dependencies (gradio, huggingface_hub, convert-to-quant, scipy)...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade -r "%~dp0requirements.txt"
if not %errorlevel%==0 (
    echo.
    echo ============================================================
    echo  Dependency upgrade failed - see the error above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Update complete.
echo ============================================================
echo.
echo PyTorch and Triton are NOT auto-upgraded here (they're large, GPU-
echo specific downloads). Re-run install.bat if you want to pick up a
echo newer CUDA build, or upgrade them yourself.
echo.
pause
