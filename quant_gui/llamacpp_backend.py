"""LLM -> GGUF conversion, orchestrated around a real llama.cpp checkout -
unlike the diffusion GGUF backend, this one is NOT a self-contained pure
Python module.

Why: a text-model GGUF file needs proper tokenizer/vocab conversion plus
per-architecture hyperparameter mapping (attention head count, rope
settings, etc.), which llama.cpp implements in its own `conversion/`
package - as of this writing, 90+ files, pinned to an exact `transformers`
version, actively updated for new model releases. There's no equivalent
lightweight pip package, and reimplementing it here would mean chasing
every new architecture by hand. So this module manages a dedicated
llama.cpp clone (kept in its own venv, since its pinned deps shouldn't leak
into this app's own) and shells out to its real `convert_hf_to_gguf.py` and
(once compiled) its `llama-quantize` binary - the same arrangement this
app already uses for `ctq`, just for a second external tool.

Direct output from convert_hf_to_gguf.py is limited to F32/F16/BF16/Q8_0
(the same pure-Python-quantizable types as quant_gui/gguf_backend.py, for
the same reason). Real K-quants (Q4_K_M, Q6_K, etc. - what most LLM GGUFs
actually use) need `llama-quantize`, a compiled C++ binary: this module
can build it (needs cmake + a C/C++ toolchain already on the machine) but
never fakes having it when it doesn't.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LLAMACPP_REPO_URL = "https://github.com/ggerganov/llama.cpp.git"

# What convert_hf_to_gguf.py can write directly, no llama-quantize needed.
DIRECT_OUTTYPE_CHOICES = ["auto", "f16", "bf16", "q8_0", "f32"]

# The real, complete list `llama-quantize --help` reports (checked against
# an actual build of the binary, not guessed/copied from docs). Aliases
# (Q3_K/Q4_K/Q5_K, which just mean their own _M variant) are omitted since
# their _M name is already in this list.
QUANT_TYPE_CHOICES = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K", "Q2_K_S",
    "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
    "TQ2_0", "TQ1_0", "Q1_0", "Q2_0", "MXFP4_MOE",
    "BF16", "F16", "F32", "COPY",
]

REQUIRED_MODULES = ("gguf", "transformers", "sentencepiece")


class LlamaCppBackendError(RuntimeError):
    pass


def default_llamacpp_dir(app_dir: Path) -> Path:
    return app_dir / "llama.cpp"


def _venv_python(llamacpp_dir: Path) -> Path:
    if platform.system() == "Windows":
        return llamacpp_dir / ".venv" / "Scripts" / "python.exe"
    return llamacpp_dir / ".venv" / "bin" / "python"


def _quantize_binary(llamacpp_dir: Path) -> Path | None:
    for candidate in (
        llamacpp_dir / "build" / "bin" / "llama-quantize",
        llamacpp_dir / "build" / "bin" / "llama-quantize.exe",
        llamacpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def is_cloned(llamacpp_dir: Path) -> bool:
    return (llamacpp_dir / "convert_hf_to_gguf.py").is_file()


def is_venv_ready(llamacpp_dir: Path) -> bool:
    py = _venv_python(llamacpp_dir)
    if not py.is_file():
        return False
    try:
        result = subprocess.run(
            [str(py), "-c", f"import {', '.join(REQUIRED_MODULES)}"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def is_quantize_built(llamacpp_dir: Path) -> bool:
    return _quantize_binary(llamacpp_dir) is not None


@dataclass
class _ProcResult:
    returncode: int | None = None


def _run_streamed(cmd: list[str], cwd: str | None = None, result: _ProcResult | None = None):
    yield f"$ {' '.join(cmd)}\n"
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            # llama-quantize's vocab/tensor-name dump can include raw BPE byte
            # pieces that aren't valid UTF-8 on their own; never let that crash
            # log streaming (confirmed against a real llama-quantize run).
        )
    except OSError as exc:
        yield f"Could not launch: {exc}\n"
        if result is not None:
            result.returncode = 127
        return
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    code = proc.wait()
    if result is not None:
        result.returncode = code


def stream_clone_or_update(llamacpp_dir: Path, repo_url: str = LLAMACPP_REPO_URL):
    """Yields log lines, then a final "__OK__" or "__FAIL__:<code>"."""
    if is_cloned(llamacpp_dir):
        cmd = ["git", "-C", str(llamacpp_dir), "pull", "--ff-only"]
    else:
        llamacpp_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", repo_url, str(llamacpp_dir)]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"


def stream_setup_venv(llamacpp_dir: Path):
    if not is_cloned(llamacpp_dir):
        yield "llama.cpp isn't cloned yet - clone it first.\n"
        yield "__FAIL__:1"
        return

    venv_dir = llamacpp_dir / ".venv"
    if not (venv_dir / ("Scripts" if platform.system() == "Windows" else "bin")).is_dir():
        r = _ProcResult()
        yield from _run_streamed([sys.executable, "-m", "venv", str(venv_dir)], result=r)
        if r.returncode != 0:
            yield f"\n❌ Creating the venv failed (exit {r.returncode}).\n"
            yield f"__FAIL__:{r.returncode}"
            return

    py = _venv_python(llamacpp_dir)
    req_file = llamacpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    r2 = _ProcResult()
    yield from _run_streamed([str(py), "-m", "pip", "install", "-r", str(req_file)], result=r2)
    if r2.returncode != 0:
        yield f"\n❌ Installing dependencies failed (exit {r2.returncode}).\n"
        yield f"__FAIL__:{r2.returncode}"
        return
    yield "\n✅ llama.cpp's Python environment is ready.\n"
    yield "__OK__"


def stream_build_quantize(llamacpp_dir: Path, jobs: int | None = None):
    if not is_cloned(llamacpp_dir):
        yield "llama.cpp isn't cloned yet - clone it first.\n"
        yield "__FAIL__:1"
        return

    build_dir = llamacpp_dir / "build"
    jobs = jobs or (os.cpu_count() or 4)

    r = _ProcResult()
    yield from _run_streamed(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"],
        cwd=str(llamacpp_dir), result=r,
    )
    if r.returncode != 0:
        yield (
            f"\n❌ cmake configure failed (exit {r.returncode}). Needs cmake and a C/C++ compiler "
            "installed (Visual Studio Build Tools on Windows, build-essential on Linux, Xcode "
            "command line tools on Mac).\n"
        )
        yield f"__FAIL__:{r.returncode}"
        return

    r2 = _ProcResult()
    yield from _run_streamed(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs), "--target", "llama-quantize"],
        cwd=str(llamacpp_dir), result=r2,
    )
    if r2.returncode != 0:
        yield f"\n❌ Build failed (exit {r2.returncode}).\n"
        yield f"__FAIL__:{r2.returncode}"
        return
    yield "\n✅ llama-quantize built.\n"
    yield "__OK__"


def stream_convert_to_gguf(llamacpp_dir: Path, model_dir: str, output_path: str, outtype: str = "auto"):
    if not is_venv_ready(llamacpp_dir):
        yield "llama.cpp's Python environment isn't set up yet.\n"
        yield "__FAIL__:1"
        return

    py = _venv_python(llamacpp_dir)
    script = llamacpp_dir / "convert_hf_to_gguf.py"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), str(script), "--outfile", output_path, "--outtype", outtype, model_dir]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"


def stream_quantize(llamacpp_dir: Path, input_gguf: str, output_gguf: str, quant_type: str):
    binary = _quantize_binary(llamacpp_dir)
    if binary is None:
        yield "llama-quantize isn't built yet.\n"
        yield "__FAIL__:1"
        return
    if quant_type not in QUANT_TYPE_CHOICES:
        yield f"Unknown quant type {quant_type!r}.\n"
        yield "__FAIL__:1"
        return

    Path(output_gguf).parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), input_gguf, output_gguf, quant_type]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"
