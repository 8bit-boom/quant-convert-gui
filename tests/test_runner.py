import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
