"""Batch queue: sequential conversion, auto naming, stop propagation."""
from pathlib import Path

import app
from quant_gui import run_control


def _fake_convert_factory(exists):
    """Fake run_convert yielding a short log + a result file when 'converted'."""
    calls = []

    def fake(input_local, input_hf, source, *rest):
        calls.append((input_local, source, rest[1]))  # record auto_output
        if input_local in exists:
            yield f"log for {input_local}", str(Path(input_local).with_suffix(".out")), ""
        else:
            yield f"failed {input_local}", None, ""
    fake.calls = calls
    return fake


def test_batch_empty_gives_guidance():
    out = list(app.run_convert_batch("", "", "", "Local file path"))
    assert "batch box" in out[-1][0].lower()


def test_batch_converts_each_path_and_forces_auto_output(monkeypatch, tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_text("x")
    b.write_text("y")
    fake = _fake_convert_factory(exists={str(a), str(b)})
    monkeypatch.setattr(app, "run_convert", fake)
    out = list(app.run_convert_batch(f"{a}\n{b}\n", "", "", "Local file path",
                                     "name", False, "rest1", "rest2"))
    last_log, _, _ = out[-1]
    assert "Batch finished: 2 ok, 0 failed, 2 total." in last_log
    assert "[1/2]" in last_log and "[2/2]" in last_log
    assert all(call[2] is True for call in fake.calls)  # auto_output forced


def test_batch_skips_missing_files(monkeypatch, tmp_path):
    a = tmp_path / "a.safetensors"
    a.write_text("x")
    fake = _fake_convert_factory(exists={str(a)})
    monkeypatch.setattr(app, "run_convert", fake)
    out = list(app.run_convert_batch(f"{a}\n{tmp_path}/missing.safetensors", "", "",
                                     "Local file path", "n", False))
    assert "Batch finished: 1 ok, 1 failed, 2 total." in out[-1][0]
    assert "not found on disk" in out[-1][0]


def test_batch_stops_when_user_stops(monkeypatch, tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_text("x")
    b.write_text("y")
    fake = _fake_convert_factory(exists={str(a), str(b)})
    monkeypatch.setattr(app, "run_convert", fake)
    ctl = run_control.RunControl()
    ctl.cancel()
    run_control.register(ctl)
    try:
        out = list(app.run_convert_batch(f"{a}\n{b}", "", "", "Local file path",
                                         "n", False))
    finally:
        run_control.register(None)
    # First item's cancel is visible to the loop only at the next boundary;
    # either way the batch must not run to a 'finished' summary.
    assert "Batch finished" not in out[-1][0] or "0 ok" in out[-1][0]
    assert len(fake.calls) <= 1
