#!/usr/bin/env bash
# Linux/Mac counterpart to install.bat - same automated setup.
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " Quant Convert GUI - installer"
echo "============================================================"
echo

PYTHON_CMD=""
for cmd in python3 python; do
  if command -v "$cmd" >/dev/null 2>&1; then
    PYTHON_CMD="$cmd"
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "Python 3.10+ was not found on PATH."
  echo "Install it from https://www.python.org/downloads/ and re-run this script."
  exit 1
fi

echo "Using: $("$PYTHON_CMD" --version)"
echo

if [ ! -f ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_CMD" -m venv .venv
else
  echo "Reusing existing virtual environment in .venv"
fi
echo

echo "Running automated setup (installs GUI deps, detects your GPU, installs"
echo "a matching PyTorch build and Triton)..."
echo
.venv/bin/python scripts/setup_env.py

echo
echo "============================================================"
echo " Setup complete. Launching Quant Convert GUI..."
echo "============================================================"
echo
exec "$(dirname "$0")/run.sh"
