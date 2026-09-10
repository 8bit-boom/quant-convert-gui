@echo off
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo No virtual environment found - run install.bat first.
    pause
    exit /b 1
)

echo Starting Quant Convert GUI at http://127.0.0.1:7860 ...
echo Press Ctrl+C to stop.
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
if not %errorlevel%==0 (
    echo.
    echo The app exited with an error - see above.
    pause
)
