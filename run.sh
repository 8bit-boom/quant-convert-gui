#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
  echo "No virtual environment found - run install.sh first."
  exit 1
fi

echo "Starting Quant Convert GUI at http://127.0.0.1:7860 ..."
echo "Press Ctrl+C to stop."
echo
.venv/bin/python app.py
