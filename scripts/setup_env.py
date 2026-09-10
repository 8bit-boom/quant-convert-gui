"""Automated environment setup for Quant Convert GUI.

Run inside the project's venv (install.bat / install.sh do this for you):
    python scripts/setup_env.py

Installs GUI dependencies, then auto-detects the GPU and installs a matching
PyTorch build and (on Windows) triton-windows, so nobody has to hand-pick a
CUDA wheel URL. Safe to re-run: already-satisfied installs are skipped.
"""

from __future__ import annotations

import importlib.util
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (min CUDA version the driver must support, wheel tag), newest first.
# A driver's "CUDA Version" (from nvidia-smi) is the highest CUDA runtime it
# can run - so we pick the newest wheel tag at or below that number.
CUDA_WHEEL_TAGS = [
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, **kwargs)


def pip_install(args: list[str]) -> bool:
    result = run([sys.executable, "-m", "pip", "install", *args])
    return result.returncode == 0


def already_importable(module_name: str) -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec(module_name) is not None


def detect_cuda_driver_version() -> tuple[int, int] | None:
    """Best-effort read of the NVIDIA driver's max supported CUDA version
    via `nvidia-smi`. Returns None if no NVIDIA GPU/driver is found."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    if not m:
        return None
    parts = m.group(1).split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None


def pick_wheel_tag(driver_cuda: tuple[int, int]) -> str:
    for min_version, tag in CUDA_WHEEL_TAGS:
        if driver_cuda >= min_version:
            return tag
    return "cu118"  # oldest supported tag as a last resort


def install_torch() -> None:
    if already_importable("torch"):
        print("PyTorch already installed - skipping (delete .venv to force a reinstall).")
        return

    driver_cuda = detect_cuda_driver_version()
    if driver_cuda is None:
        print("No NVIDIA GPU/driver detected via nvidia-smi - installing CPU-only PyTorch.")
        print("Conversion will work but be slow. Install NVIDIA drivers and re-run this script for GPU support.")
        index_url = "https://download.pytorch.org/whl/cpu"
    else:
        tag = pick_wheel_tag(driver_cuda)
        print(f"Detected NVIDIA driver supporting up to CUDA {driver_cuda[0]}.{driver_cuda[1]} -> using PyTorch build '{tag}'.")
        index_url = f"https://download.pytorch.org/whl/{tag}"

    ok = pip_install(["torch", "--index-url", index_url])
    if not ok:
        print("!! PyTorch install failed. See the error above - you may need to install it manually")
        print("   (see https://pytorch.org/get-started/locally/).")


def get_torch_version() -> tuple[int, int] | None:
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    m = re.match(r"(\d+)\.(\d+)", result.stdout.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def install_triton() -> None:
    if already_importable("triton"):
        print("Triton already installed - skipping.")
        return

    system = platform.system()
    torch_version = get_torch_version()

    if system == "Windows":
        # Mirrors convert_to_quant's own README guidance: triton-windows'
        # supported ceiling tracks the installed torch minor version.
        constraint = "triton-windows"
        if torch_version:
            major, minor = torch_version
            if (major, minor) >= (2, 12):
                constraint = "triton-windows<3.8"
            elif (major, minor) >= (2, 10):
                constraint = "triton-windows<3.7"
        print(f"Installing Triton for Windows (optional, speeds up INT8 kernels): {constraint}")
        ok = pip_install(["-U", constraint])
    elif system == "Linux":
        print("Installing Triton (optional, speeds up INT8 kernels).")
        ok = pip_install(["-U", "triton"])
    else:
        print(f"Skipping Triton on {system} - not published for this platform.")
        return

    if not ok:
        print("!! Triton install failed - this is optional (only speeds up INT8 kernels), continuing without it.")


def install_requirements() -> bool:
    req_file = REPO_ROOT / "requirements.txt"
    print("Installing GUI + ctq requirements...")
    return pip_install(["-r", str(req_file)])


def print_final_report() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from quant_gui.env_check import check_environment, report_markdown

        print("\n" + "=" * 60)
        print("Environment summary")
        print("=" * 60)
        print(report_markdown(check_environment()).replace("**", "").replace("`", ""))
        print("=" * 60)
    except Exception as exc:  # noqa: BLE001
        print(f"(Could not run the final environment check: {exc})")


def main() -> int:
    print("Quant Convert GUI - automated setup\n")

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    install_torch()
    install_triton()

    if not install_requirements():
        print("\n!! Installing requirements.txt failed - see the error above.")
        return 1

    print_final_report()
    print("\nSetup complete. Run run.bat (Windows) or run.sh (Linux/Mac) to start the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
