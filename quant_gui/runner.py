"""Run `ctq` as a subprocess and stream its output line by line."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator

_INLINE_ENTRYPOINT = "from convert_to_quant.cli import main; main()"


def resolve_command(args: list[str], python_executable: str | None = None) -> list[str]:
    """Pick the best way to invoke ctq: the installed console script if
    present on PATH, otherwise `python -c "from convert_to_quant.cli import
    main; main()"` using the given (or current) interpreter.
    """
    ctq_path = shutil.which("ctq")
    if ctq_path and not python_executable:
        return [ctq_path, *args]

    py = python_executable or sys.executable
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
