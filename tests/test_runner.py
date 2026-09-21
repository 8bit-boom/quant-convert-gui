import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

from quant_gui import runner


def test_prefers_current_interpreter_when_it_has_ctq(monkeypatch):
    # Simulate running inside a venv that has convert_to_quant installed -
    # this is the normal case (app.py launched via run.bat/run.sh).
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(runner.shutil, "which", lambda name: r"C:\Some\Other\Python\Scripts\ctq.exe")

    cmd = runner.resolve_command(["-i", "model.safetensors"])

    assert cmd[0] == sys.executable
    assert "-c" in cmd
    # Must NOT pick the stray global ctq.exe even though it's on PATH -
    # this is the bug that surfaced for real: a stale global install
    # missing a dependency shadowed the correctly-provisioned venv.
    assert r"C:\Some\Other\Python\Scripts\ctq.exe" not in cmd


def test_falls_back_to_path_ctq_when_current_interpreter_lacks_it(monkeypatch):
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/local/bin/ctq")

    cmd = runner.resolve_command(["-i", "model.safetensors"])

    assert cmd[0] == "/usr/local/bin/ctq"


def test_falls_back_to_inline_entrypoint_when_nothing_found(monkeypatch):
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)

    cmd = runner.resolve_command(["-i", "model.safetensors"])

    assert cmd[0] == sys.executable
    assert "-c" in cmd


def test_stream_conversion_can_be_cancelled(monkeypatch):
    import time as _time

    from quant_gui import run_control

    script = "import time\nwhile True:\n    print('tick', flush=True)\n    time.sleep(0.05)"
    monkeypatch.setattr(
        runner, "resolve_command", lambda args, python_executable=None: [sys.executable, "-c", script],
    )
    ctl = run_control.RunControl()

    lines = []
    for line in runner.stream_conversion(["-i", "model.safetensors"], control=ctl):
        lines.append(line)
        if "tick" in line:
            ctl.cancel()

    assert lines[-1] == "__CTQ_CANCELLED__"
    _time.sleep(0.1)  # let the OS reap the terminated child


def test_suspend_and_resume_process():
    import signal
    import subprocess
    import time

    if os.name != "nt" and not hasattr(signal, "SIGSTOP"):
        import pytest

        pytest.skip("process suspension not supported on this platform")

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        runner._suspend_process(proc.pid)
        time.sleep(0.5)
        assert proc.poll() is None  # still alive while suspended
        runner._resume_process(proc.pid)
        proc.terminate()
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
