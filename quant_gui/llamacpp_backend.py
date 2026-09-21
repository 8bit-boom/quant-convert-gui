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

Also manages `llama-imatrix`, the same real tool behind most "imatrix"
GGUF quants on Hugging Face (and the foundation Unsloth's own "Dynamic"
quants are built on, per their own docs): it runs calibration text through
the full-precision model and records which weights actually matter, so
`llama-quantize --imatrix ...` can round more carefully on the layers that
need it. `llama-quantize` also exposes `--tensor-type`/`--tensor-type-file`
for manually assigning a different GGML type per tensor - the real
mechanism behind per-layer "dynamic" mixing, exposed here as a power-user
override rather than an automatic reproduction of any specific published
recipe, since exact per-model layer choices (Unsloth's included) aren't
published in a form this module can just consume.
"""

from __future__ import annotations

import os
import platform
import shutil
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

# llama.cpp's requirements-convert_hf_to_gguf.txt pins transformers==4.57.6,
# which crashes on Gemma 3/4 tokenizers (their tokenizer_config ships
# extra_special_tokens as a LIST, and 4.x's GemmaFastTokenizer does
# `special_tokens.keys()` on it -> AttributeError: 'list' object has no
# attribute 'keys'). transformers 5.x handles both forms, so setup
# force-upgrades past llama.cpp's pin afterwards.
TRANSFORMERS_MIN_VERSION = "5.0.0"
TRANSFORMERS_MIN_SPEC = f"transformers>={TRANSFORMERS_MIN_VERSION}"

DEFAULT_CALIBRATION_FILE = Path(__file__).resolve().parent / "data" / "default_calibration.txt"


class LlamaCppBackendError(RuntimeError):
    pass


def default_llamacpp_dir(app_dir: Path) -> Path:
    return app_dir / "llama.cpp"


def _venv_python(llamacpp_dir: Path) -> Path:
    if platform.system() == "Windows":
        return llamacpp_dir / ".venv" / "Scripts" / "python.exe"
    return llamacpp_dir / ".venv" / "bin" / "python"


def _find_binary(llamacpp_dir: Path, name: str) -> Path | None:
    for candidate in (
        llamacpp_dir / "build" / "bin" / name,
        llamacpp_dir / "build" / "bin" / f"{name}.exe",
        llamacpp_dir / "build" / "bin" / "Release" / f"{name}.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def _quantize_binary(llamacpp_dir: Path) -> Path | None:
    return _find_binary(llamacpp_dir, "llama-quantize")


def _imatrix_binary(llamacpp_dir: Path) -> Path | None:
    return _find_binary(llamacpp_dir, "llama-imatrix")


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


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """'5.17.0' -> (5, 17, 0); anything unparseable -> None."""
    parts = (text or "").strip().split(".")
    if not parts or not parts[0].isdigit():
        return None
    out = []
    for p in parts:
        digits = "".join(ch for ch in p if ch.isdigit())
        if digits == "":
            break
        out.append(int(digits))
    return tuple(out) if out else None


def transformers_too_old(version: str | None) -> bool:
    """True when `version` (or unparseable/None) is below TRANSFORMERS_MIN_VERSION."""
    got = _version_tuple(version)
    want = _version_tuple(TRANSFORMERS_MIN_VERSION)
    if got is None or want is None:
        return True
    length = max(len(got), len(want))
    return got + (0,) * (length - len(got)) < want + (0,) * (length - len(want))


def venv_transformers_version(llamacpp_dir: Path) -> str | None:
    """The venv's installed transformers version, or None when unreadable."""
    py = _venv_python(llamacpp_dir)
    if not py.is_file():
        return None
    try:
        result = subprocess.run(
            [str(py), "-c", "import transformers; print(transformers.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def is_quantize_built(llamacpp_dir: Path) -> bool:
    return _quantize_binary(llamacpp_dir) is not None


def is_imatrix_built(llamacpp_dir: Path) -> bool:
    return _imatrix_binary(llamacpp_dir) is not None


def _perplexity_binary(llamacpp_dir: Path) -> Path | None:
    return _find_binary(llamacpp_dir, "llama-perplexity")


def is_perplexity_built(llamacpp_dir: Path) -> bool:
    return _perplexity_binary(llamacpp_dir) is not None


PPL_FINAL_RE = None  # compiled lazily to keep import cost at zero


def parse_final_ppl(output: str) -> float | None:
    """Extract the 'Final estimate: PPL = X' value from llama-perplexity output."""
    import re

    global PPL_FINAL_RE
    if PPL_FINAL_RE is None:
        PPL_FINAL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    matches = PPL_FINAL_RE.findall(output or "")
    return float(matches[-1]) if matches else None


def stream_perplexity(llamacpp_dir: Path, model_gguf: str, text_file: str, ngl: int | None = None):
    """Run llama-perplexity on `text_file` against `model_gguf`, yielding log
    lines followed by '__OK__' or '__FAIL__:<code>'. The final estimate line
    is in the log; pull it with parse_final_ppl."""
    binary = _perplexity_binary(Path(llamacpp_dir))
    if binary is None:
        yield "llama-perplexity isn't available - re-download the prebuilt binaries (Setup step 3), the release zip ships it.\n"
        yield "__FAIL__:1"
        return
    if not Path(model_gguf).is_file():
        yield f"Model GGUF not found: {model_gguf}\n"
        yield "__FAIL__:1"
        return
    if not Path(text_file).is_file():
        yield f"Validation text not found: {text_file}\n"
        yield "__FAIL__:1"
        return
    cmd = [str(binary), "-m", model_gguf, "-f", str(text_file)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"


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

    # llama.cpp pins transformers==4.57.6, which crashes on Gemma 3/4
    # tokenizers (extra_special_tokens list vs dict) - force >=5.
    yield (
        f"\nllama.cpp pins transformers 4.x, which crashes on Gemma 3/4 tokenizers - "
        f"upgrading to {TRANSFORMERS_MIN_SPEC}.\n"
    )
    r3 = _ProcResult()
    yield from _run_streamed(
        [str(py), "-m", "pip", "install", "--upgrade", TRANSFORMERS_MIN_SPEC], result=r3,
    )
    if r3.returncode != 0:
        yield (
            f"\n⚠️ Couldn't upgrade transformers to {TRANSFORMERS_MIN_SPEC} (exit "
            f"{r3.returncode}) - Gemma 3/4 conversions may fail on the tokenizer step.\n"
        )
    version = venv_transformers_version(llamacpp_dir)
    yield f"\n✅ llama.cpp's Python environment is ready (transformers {version or 'unknown'}).\n"
    yield "__OK__"


def stream_build_quantize(llamacpp_dir: Path, jobs: int | None = None, prefer: str | None = None):
    """Make llama-quantize + llama-imatrix available, by the best means the
    machine supports:

    1. Already present (built or previously downloaded) - done.
    2. ``prefer='source'`` or a working cmake+compiler toolchain - build
       from source (needs cmake and a C/C++ toolchain installed).
    3. Otherwise - download the official prebuilt Windows binaries for the
       cloned llama.cpp's exact release tag (plus the matching cudart
       package when an NVIDIA GPU is present) and unpack them where the
       rest of this module looks for built binaries.

    Yields log lines, then "__OK__" or "__FAIL__:<code>".
    """
    llamacpp_dir = Path(llamacpp_dir)
    if not is_cloned(llamacpp_dir):
        yield "llama.cpp isn't cloned yet - clone it first.\n"
        yield "__FAIL__:1"
        return
    if _quantize_binary(llamacpp_dir) is not None and _imatrix_binary(llamacpp_dir) is not None:
        yield "llama-quantize and llama-imatrix are already present - nothing to do.\n"
        yield "__OK__"
        return

    prefer = (prefer or "auto").lower()
    if prefer == "source" or (prefer == "auto" and _toolchain_available()):
        yield from _stream_build_from_source(llamacpp_dir, jobs=jobs)
        return
    if platform.system() == "Windows":
        yield "No cmake/C++ toolchain found - downloading official prebuilt binaries instead.\n"
        yield "(Set one up and use prefer='source' to compile from source.)\n"
        yield from _stream_download_prebuilt(llamacpp_dir)
        return
    yield (
        "No cmake/C++ toolchain found, and prebuilt downloads are only wired up for Windows. "
        "Install cmake + a C/C++ compiler (build-essential / Xcode CLT / VS Build Tools) and retry.\n"
    )
    yield "__FAIL__:1"


def _toolchain_available() -> bool:
    return shutil.which("cmake") is not None


def local_build_tag(llamacpp_dir: Path) -> str | None:
    """The llama.cpp release tag of this clone (e.g. 'b11070'), via git describe."""
    try:
        r = subprocess.run(
            ["git", "-C", str(llamacpp_dir), "describe", "--tags", "--always"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    tag = (r.stdout or "").strip()
    return tag or None


def choose_prebuilt_assets(assets: list[str], use_cuda: bool) -> dict | None:
    """Pick the download URLs-to-be from a release's asset names.

    Returns {"main": name, "cudart": name|None} or None when nothing fits.
    Prefers the newest CUDA flavour when use_cuda and a matching cudart
    package exists; falls back to the CPU build.
    """
    win = [a for a in assets if a.endswith(".zip") and "-bin-win-" in a and "arm64" not in a]
    cudavers = sorted(
        {a.split("-cuda-")[1].split("-")[0] for a in win if "-cuda-" in a},
        key=lambda v: [int(x) for x in v.split(".")],
        reverse=True,
    )
    if use_cuda:
        for cv in cudavers:
            main = next(
                (a for a in win if f"-bin-win-cuda-{cv}-x64" in a and not a.startswith("cudart-")),
                None,
            )
            cudart = next((a for a in win if a.startswith("cudart-") and f"-cuda-{cv}-x64" in a), None)
            if main and cudart:
                return {"main": main, "cudart": cudart}
    main = next((a for a in win if "-bin-win-cpu-x64" in a), None)
    if main:
        return {"main": main, "cudart": None}
    return None


def _nvidia_gpu_present() -> bool:
    return shutil.which("nvidia-smi") is not None


def _stream_download_prebuilt(llamacpp_dir: Path, use_cuda: bool | None = None):
    """Download + unpack official prebuilt llama.cpp binaries for this clone's tag."""
    import json
    import urllib.request
    import zipfile

    if use_cuda is None:
        use_cuda = _nvidia_gpu_present()

    tag = local_build_tag(llamacpp_dir)
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
    release = None
    if tag:
        try:
            with urllib.request.urlopen(f"{api}/tags/{tag}", timeout=30) as r:
                release = json.loads(r.read().decode("utf-8"))
            yield f"Found GitHub release {tag} matching this clone.\n"
        except Exception:  # noqa: BLE001 - fall through to latest release
            release = None
    if release is None:
        yield (
            f"No GitHub release for tag {tag!r} (or unreachable) - using the latest release; "
            "binaries may be a bit newer than the cloned scripts.\n"
        )
        try:
            with urllib.request.urlopen(f"{api}/latest", timeout=30) as r:
                release = json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            yield f"Couldn't reach GitHub releases: {exc}\n"
            yield "__FAIL__:1"
            return

    assets = [a["name"] for a in release.get("assets", [])]
    picked = choose_prebuilt_assets(assets, use_cuda=use_cuda)
    if picked is None:
        yield f"Release {release.get('tag_name')} has no Windows x64 binaries to download.\n"
        yield "__FAIL__:1"
        return
    yield f"Release {release.get('tag_name')}: downloading {picked['main']}"
    yield f" (GPU detected: CUDA build + cudart package)\n" if picked["cudart"] else " (CPU build)\n"

    dest = llamacpp_dir / "build" / "bin" / "Release"
    dest.mkdir(parents=True, exist_ok=True)
    tmp_dir = llamacpp_dir / "build" / ".prebuilt-download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in [picked["main"], picked["cudart"]]:
            if not name:
                continue
            url = next(a["browser_download_url"] for a in release["assets"] if a["name"] == name)
            target = tmp_dir / name
            yield f"  {name} ...\n"
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(target, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    got, since = 0, 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        since += len(chunk)
                        if since >= 25 * (1 << 20):
                            since = 0
                            pct = f" {got/1e6:.0f}/{total/1e6:.0f} MB" if total else f" {got/1e6:.0f} MB"
                            yield f"    ...{pct}\n"
            except Exception as exc:  # noqa: BLE001
                yield f"  download failed: {exc}\n"
                yield "__FAIL__:1"
                return
            yield "    unpacking...\n"
            try:
                with zipfile.ZipFile(target) as zf:
                    zf.extractall(dest)
            except zipfile.BadZipFile as exc:
                yield f"  corrupt download: {exc}\n"
                yield "__FAIL__:1"
                return
            target.unlink()
    finally:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    q = _quantize_binary(llamacpp_dir)
    i = _imatrix_binary(llamacpp_dir)
    if q is None or i is None:
        yield "Unpack finished but llama-quantize/llama-imatrix still not found.\n"
        yield "__FAIL__:1"
        return
    try:
        # --version isn't a real flag: exit 0/1 with usage text still proves
        # the binary and its DLLs load; a missing cudart DLL would give a
        # large Windows status code or fail to launch at all.
        r = subprocess.run([str(q), "--version"], capture_output=True, timeout=60)
        launched_ok = r.returncode in (0, 1)
    except (OSError, subprocess.TimeoutExpired):
        launched_ok = False
    if not launched_ok:
        yield "llama-quantize is present but failed to launch - a runtime DLL is probably missing.\n"
        yield "__FAIL__:1"
        return
    yield "✅ Prebuilt binaries installed and verified runnable (llama-quantize + llama-imatrix).\n"
    yield "__OK__"


def _stream_build_from_source(llamacpp_dir: Path, jobs: int | None = None):
    """Compile llama-quantize + llama-imatrix from source (cmake path)."""
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
        [
            "cmake", "--build", str(build_dir), "--config", "Release", "-j", str(jobs),
            "--target", "llama-quantize", "--target", "llama-imatrix",
        ],
        cwd=str(llamacpp_dir), result=r2,
    )
    if r2.returncode != 0:
        yield f"\n❌ Build failed (exit {r2.returncode}).\n"
        yield f"__FAIL__:{r2.returncode}"
        return
    yield "\n✅ llama-quantize and llama-imatrix built.\n"
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


def stream_generate_imatrix(
    llamacpp_dir: Path, model_gguf: str, output_imatrix: str,
    calibration_file: str | None = None, chunks: int | None = None,
):
    """Runs llama-imatrix: the real tool behind most "imatrix" GGUF quants
    (and the foundation Unsloth's own Dynamic quants are built on) - records
    which weights actually matter by running calibration text through the
    full-precision model, so a later llama-quantize --imatrix run can round
    more carefully on the layers that need it."""
    binary = _imatrix_binary(llamacpp_dir)
    if binary is None:
        yield "llama-imatrix isn't built yet.\n"
        yield "__FAIL__:1"
        return
    if not Path(model_gguf).is_file():
        yield f"Model GGUF not found: {model_gguf}\n"
        yield "__FAIL__:1"
        return

    calib_path = Path(calibration_file) if (calibration_file or "").strip() else DEFAULT_CALIBRATION_FILE
    if not calib_path.is_file():
        yield f"Calibration file not found: {calib_path}\n"
        yield "__FAIL__:1"
        return

    Path(output_imatrix).parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-m", model_gguf, "-f", str(calib_path), "-o", output_imatrix]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"


def stream_quantize(
    llamacpp_dir: Path, input_gguf: str, output_gguf: str, quant_type: str,
    imatrix_file: str | None = None, tensor_type_file: str | None = None,
):
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
    cmd = [str(binary)]
    if (imatrix_file or "").strip():
        cmd += ["--imatrix", imatrix_file.strip()]
    if (tensor_type_file or "").strip():
        cmd += ["--tensor-type-file", tensor_type_file.strip()]
    cmd += [input_gguf, output_gguf, quant_type]
    r = _ProcResult()
    yield from _run_streamed(cmd, result=r)
    yield "__OK__" if r.returncode == 0 else f"__FAIL__:{r.returncode}"
