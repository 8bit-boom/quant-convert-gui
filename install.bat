@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Quant Convert GUI - installer
echo ============================================================
echo.

set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel%==0 (
    python --version >nul 2>nul
    if !errorlevel!==0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if !errorlevel!==0 (
        py -3 --version >nul 2>nul
        if !errorlevel!==0 set "PYTHON_CMD=py -3"
    )
)

if not defined PYTHON_CMD (
    echo Python was not found on PATH.
    echo.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
    echo During setup, make sure to check "Add python.exe to PATH".
    echo Then re-run this installer.
    echo.
    start "" "https://www.python.org/downloads/windows/"
    pause
    exit /b 1
)

echo Using:
%PYTHON_CMD% --version
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYTHON_CMD% -m venv "%~dp0.venv"
    if not exist "%~dp0.venv\Scripts\python.exe" (
        echo.
        echo Failed to create the virtual environment. See the error above.
        pause
        exit /b 1
    )
) else (
    echo Reusing existing virtual environment in .venv
)
echo.

echo Running automated setup (installs GUI deps, detects your GPU, installs
echo a matching PyTorch build and Triton)...
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\setup_env.py"
if not %errorlevel%==0 (
    echo.
    echo ============================================================
    echo  Setup failed - see the error above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete. Launching Quant Convert GUI...
echo ============================================================
echo.
call "%~dp0run.bat"
