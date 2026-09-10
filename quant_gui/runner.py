"""Run `ctq` as a subprocess and stream its output line by line."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Iterator

_INLINE_ENTRYPOINT = "from convert_to_quant.cli import main; main()"


def _has_convert_to_quant(python_executable: str | None) -> bool:
    if python_executable is None:
        # The interpreter running this code right now - check in-process,
        # no subprocess needed.
        return importlib.util.find_spec("convert_to_quant") is not None
    try:
        result = subprocess.run(
            [python_executable, "-c", "import convert_to_quant"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_command(args: list[str], python_executable: str | None = None) -> list[str]:
    """Pick the best way to invoke ctq.

    Prefers running `python -c "from convert_to_quant.cli import main;
    main()"` through the given (or current) interpreter *whenever that
    interpreter actually has ctq installed* - this guarantees we use the
    project's own correctly-provisioned .venv (installed by install.bat/sh)
    rather than a stray/stale `ctq` console script that happens to be
    first on PATH from some other, possibly incomplete, install. Only
    falls back to a PATH-found `ctq` when the chosen interpreter doesn't
    have ctq at all.
    """
    py = python_executable or sys.executable

    if _has_convert_to_quant(python_executable):
        return [py, "-c", _INLINE_ENTRYPOINT, *args]

    ctq_path = shutil.which("ctq")
    if ctq_path:
        return [ctq_path, *args]

    return [py, "-c", _INLINE_ENTRYPOINT, *args]


def stream_conversion(args: list[str], python_executable: str | None = None) -> Iterator[str]:
    """Yield stdout/stderr lines from the ctq process as they arrive.

    The final yielded line is one of:
      "__CTQ_OK__"           on success (return code 0)
      "__CTQ_FAIL__:<code>"  on non-zero exit
    """
    cmd = resolve_command(args, python_executable)
    yield f"$ {' '.join(cmd)}\n"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        yield f"Could not launch ctq: {exc}\n"
        yield "__CTQ_FAIL__:127"
        return

    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    code = proc.wait()

    if code == 0:
        yield "__CTQ_OK__"
    else:
        yield f"__CTQ_FAIL__:{code}"
