"""Run `ctq` as a subprocess and stream its output line by line."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
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


def _suspend_process(pid: int) -> None:
    """Freeze every thread of the process (true pause, no state lost).

    POSIX: SIGSTOP. Windows has no signal equivalent, so enumerate the
    process's threads and SuspendThread() each one via ctypes.
    Raises OSError if suspension isn't possible on this platform.
    """
    if os.name == "posix":
        os.kill(pid, signal.SIGSTOP)
        return
    if os.name == "nt":
        _windows_suspend(pid, suspend=True)
        return
    raise OSError(f"process suspension not supported on {os.name}")


def _resume_process(pid: int) -> None:
    if os.name == "posix":
        os.kill(pid, signal.SIGCONT)
        return
    if os.name == "nt":
        _windows_suspend(pid, suspend=False)
        return
    raise OSError(f"process suspension not supported on {os.name}")


def _windows_suspend(pid: int, suspend: bool) -> None:
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPTHREAD = 0x00000004
    THREAD_SUSPEND_RESUME = 0x0002
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise OSError(f"CreateToolhelp32Snapshot failed (error {ctypes.get_last_error()})")

    failures = 0
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)
        if not kernel32.Thread32First(snapshot, ctypes.byref(entry)):
            raise OSError(f"Thread32First failed (error {ctypes.get_last_error()})")
        while True:
            if entry.th32OwnerProcessID == pid:
                thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if thread:
                    try:
                        result = kernel32.SuspendThread(thread) if suspend else kernel32.ResumeThread(thread)
                        if result == 0xFFFFFFFF:
                            failures += 1
                    finally:
                        kernel32.CloseHandle(thread)
                else:
                    failures += 1
            if not kernel32.Thread32Next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)

    if failures:
        action = "suspend" if suspend else "resume"
        raise OSError(f"failed to {action} {failures} thread(s) of pid {pid}")


_checkpoint_support_cache: dict[str | None, bool] = {}


def ctq_supports_checkpoints(python_executable: str | None = None) -> bool:
    """True when the ctq we would launch has native stop/resume checkpoints.

    Probes for the ``convert_to_quant.checkpoint`` module added in the
    checkpoint-resume release, without importing torch in this process.
    Uses the same resolution rule as ``resolve_command``: the given (or
    current) interpreter first, then a ``ctq --help`` scan for the
    PATH-fallback case. Results are cached per interpreter.
    """
    key = (python_executable or "").strip() or None
    if key in _checkpoint_support_cache:
        return _checkpoint_support_cache[key]

    ok = False
    if _has_convert_to_quant(python_executable):
        if key is None:
            ok = importlib.util.find_spec("convert_to_quant.checkpoint") is not None
        else:
            try:
                result = subprocess.run(
                    [key, "-c", "import convert_to_quant.checkpoint"],
                    capture_output=True, timeout=30,
                )
                ok = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                ok = False
    else:
        ctq_path = shutil.which("ctq")
        if ctq_path:
            try:
                result = subprocess.run(
                    [ctq_path, "--help"], capture_output=True, text=True, timeout=180,
                )
                ok = "--checkpoint-dir" in (result.stdout or "")
            except (OSError, subprocess.TimeoutExpired):
                ok = False

    _checkpoint_support_cache[key] = ok
    return ok


_perf_flag_support_cache: dict[str | None, bool] = {}


def ctq_supports_perf_flags(python_executable: str | None = None) -> bool:
    """True when the ctq we would launch accepts the GPU perf flags.

    Probes for the ``convert_to_quant.utils.debounced_gc`` module added in
    the perf release (--fast_math / --loss_sync_batch / --snapshot_interval /
    --compile_loop), without importing torch in this process. Uses the same
    resolution rule as ``ctq_supports_checkpoints``; cached per interpreter.
    """
    key = (python_executable or "").strip() or None
    if key in _perf_flag_support_cache:
        return _perf_flag_support_cache[key]

    ok = False
    if _has_convert_to_quant(python_executable):
        if key is None:
            ok = importlib.util.find_spec("convert_to_quant.utils.debounced_gc") is not None
        else:
            try:
                result = subprocess.run(
                    [key, "-c", "import convert_to_quant.utils.debounced_gc"],
                    capture_output=True, timeout=30,
                )
                ok = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                ok = False
    else:
        ctq_path = shutil.which("ctq")
        if ctq_path:
            try:
                result = subprocess.run(
                    [ctq_path, "--help"], capture_output=True, text=True, timeout=180,
                )
                ok = "--fast_math" in (result.stdout or "")
            except (OSError, subprocess.TimeoutExpired):
                ok = False

    _perf_flag_support_cache[key] = ok
    return ok


def _extract_output_path(args: list[str]) -> str | None:
    """Find the ctq output file (-o/--output) in the CLI arg list."""
    best: str | None = None
    best_index: int | None = None
    for flag in ("-o", "--output"):
        for i, arg in enumerate(args):
            value: str | None = None
            if arg == flag and i + 1 < len(args):
                value = args[i + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            if value is not None and (best_index is None or i < best_index):
                best, best_index = value, i
    return best


def _preserve_partial_output(args: list[str]) -> str | None:
    """Rename ctq's partially-written output to a timestamped .partial file.

    ctq (FP8/INT8/NVFP4/MXFP8) writes its output incrementally; when a run
    is stopped, the partial file is kept rather than left in place to be
    overwritten by the next resume attempt. Returns the backup path, or
    None when there is nothing worth keeping (or the rename failed).
    """
    import datetime

    out = _extract_output_path(args)
    if not out:
        return None
    try:
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            return None
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{out}.partial-{stamp}"
        os.replace(out, backup)
        return backup
    except OSError:
        return None


def stream_conversion(
    args: list[str],
    python_executable: str | None = None,
    control=None,
    stop_file: str | None = None,
) -> Iterator[str]:
    """Yield stdout/stderr lines from the ctq process as they arrive.

    The final yielded line is one of:
      "__CTQ_OK__"           on success (return code 0)
      "__CTQ_FAIL__:<code>"  on non-zero exit
      "__CTQ_CANCELLED__"    the run was stopped via `control` (Stop & save)
                             in legacy mode (no `stop_file`): ctq was
                             terminated and any partial output preserved
      "__CTQ_STOPPED__"      a checkpointed stop: ctq saved per-tensor
                             progress and exited (code 3) - resume continues
                             from the last completed tensor

    `control` (a run_control.RunControl) enables the GUI's pause/stop
    buttons: pause freezes the whole ctq process between output lines
    (SIGSTOP/SIGCONT, or thread suspension on Windows). With `stop_file`
    set, stop writes that file and ctq (checkpoint-aware build) exits
    cleanly once the current tensor is done; without it, stop terminates
    ctq immediately (legacy builds) and the partial output file is kept
    under a `.partial-<timestamp>` name.
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
    suspended = False
    cancelled = False
    stop_written = False
    try:
        for line in proc.stdout:
            if control is not None and control.is_cancelled:
                cancelled = True
                if stop_file is not None:
                    # Checkpoint-aware ctq: ask it to stop between tensors
                    # instead of killing it mid-write.
                    if not stop_written:
                        stop_written = True
                        try:
                            with open(stop_file, "w"):
                                pass
                            yield "(stop requested - ctq finishes the current tensor, saves progress, then exits)\n"
                        except OSError as exc:
                            yield f"(couldn't write the stop file: {exc} - falling back to terminating ctq)\n"
                            try:
                                proc.terminate()
                            except OSError:
                                pass
                else:
                    # TerminateProcess works even on a suspended process; after
                    # SIGSTOP the pipe stops producing lines, so terminate first
                    # and let the loop drain to EOF below.
                    try:
                        proc.terminate()
                    except OSError:
                        pass
            elif control is not None and control.is_paused and not suspended:
                try:
                    _suspend_process(proc.pid)
                    suspended = True
                    yield "(ctq process suspended - click Resume to continue)\n"
                except OSError as exc:
                    yield f"(couldn't suspend the ctq process: {exc} - it keeps running)\n"
            elif control is not None and not control.is_paused and suspended:
                try:
                    _resume_process(proc.pid)
                except OSError as exc:
                    yield f"(couldn't resume the ctq process: {exc})\n"
                suspended = False
            yield line
    finally:
        if suspended:
            try:
                _resume_process(proc.pid)
            except OSError:
                pass
        # Consume the stop request once the process is gone, so a resume
        # doesn't stop instantly on a stale file.
        if stop_file is not None:
            try:
                if os.path.exists(stop_file):
                    os.unlink(stop_file)
            except OSError:
                pass

    code = proc.wait()

    if stop_file is not None and (code == 3 or cancelled):
        yield "__CTQ_STOPPED__"
    elif cancelled:
        backup = _preserve_partial_output(args)
        if backup:
            yield f"(partial output preserved: {backup} - resume restarts ctq from the beginning)\n"
        yield "__CTQ_CANCELLED__"
    elif code == 0:
        yield "__CTQ_OK__"
    else:
        yield f"__CTQ_FAIL__:{code}"
