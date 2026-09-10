#!/usr/bin/env bash
# Linux/Mac counterpart to update.bat.
cd "$(dirname "$0")"

echo "============================================================"
echo " Quant Convert GUI - updater"
echo "============================================================"
echo

if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
  echo "Pulling latest changes from git..."
  git pull
  echo
elif [ -d ".git" ]; then
  echo "(git not found on PATH - skipping source update.)"
  echo
else
  echo "(This folder isn't a git checkout - skipping source update."
  echo " Download the latest release/zip to get code updates.)"
  echo
fi

if [ ! -f ".venv/bin/python" ]; then
  echo "No virtual environment found - run install.sh first."
  exit 1
fi

echo "Upgrading GUI dependencies (gradio, huggingface_hub, convert-to-quant, scipy)..."
.venv/bin/python -m pip install --upgrade -r requirements.txt

echo
echo "============================================================"
echo " Update complete."
echo "============================================================"
echo
echo "PyTorch and Triton are NOT auto-upgraded here (they're large, GPU-"
echo "specific downloads). Re-run install.sh if you want to pick up a"
echo "newer CUDA build, or upgrade them yourself."
