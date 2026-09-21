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


def test_extract_output_path_variants():
    assert runner._extract_output_path(["-i", "in.st", "-o", "out.st"]) == "out.st"
    assert runner._extract_output_path(["--output", "out.st"]) == "out.st"
    assert runner._extract_output_path(["--output=out.st"]) == "out.st"
    assert runner._extract_output_path(["-i", "in.st", "--output", "a.st", "-o", "b.st"]) == "a.st"
    assert runner._extract_output_path(["-i", "in.st"]) is None


def test_preserve_partial_output_renames_existing_file(tmp_path):
    out = tmp_path / "model.safetensors"
    out.write_bytes(b"partial-bytes")

    backup = runner._preserve_partial_output(["-o", str(out)])

    assert backup is not None
    assert not out.exists()
    assert Path(backup).read_bytes() == b"partial-bytes"


def test_preserve_partial_output_keeps_nothing_when_absent_or_empty(tmp_path):
    missing = tmp_path / "nope.safetensors"
    assert runner._preserve_partial_output(["-o", str(missing)]) is None

    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    assert runner._preserve_partial_output(["-o", str(empty)]) is None
    assert empty.exists()  # untouched


def test_cancelled_run_preserves_partial_output(monkeypatch, tmp_path):
    import time as _time

    from quant_gui import run_control

    out = tmp_path / "out.st"
    script = (
        f"open({str(out)!r}, 'wb').write(b'partial')\n"
        "import time\n"
        "time.sleep(0.2)\n"  # let the file handle close before we get cancelled
        "while True:\n    print('tick', flush=True)\n    time.sleep(0.05)"
    )
    monkeypatch.setattr(
        runner, "resolve_command", lambda args, python_executable=None: [sys.executable, "-c", script],
    )
    ctl = run_control.RunControl()

    notices = []
    for line in runner.stream_conversion(["-o", str(out)], control=ctl):
        if "partial output preserved" in line:
            notices.append(line)
        if "tick" in line:
            ctl.cancel()

    assert notices, "expected a partial-output preservation notice"
    assert not out.exists()  # renamed away, not left to be overwritten
    _time.sleep(0.1)  # let the OS reap the terminated child
