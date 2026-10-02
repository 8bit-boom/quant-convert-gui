import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui import run_control
from quant_gui.run_control import RunCancelled, RunControl


def test_starts_running_and_not_cancelled():
    ctl = RunControl()
    assert not ctl.is_paused
    assert not ctl.is_cancelled


def test_pause_resume_toggle():
    ctl = RunControl()
    assert ctl.toggle_pause() is True
    assert ctl.is_paused
    assert ctl.toggle_pause() is False
    assert not ctl.is_paused


def test_wait_if_paused_returns_false_when_running():
    ctl = RunControl()
    assert ctl.wait_if_paused() is False


def test_cancel_while_paused_raises_in_waiter():
    ctl = RunControl()
    ctl.pause()

    seen = []

    def worker():
        try:
            ctl.wait_if_paused(poll_seconds=0.05)
        except RunCancelled:
            seen.append("cancelled")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.2)  # let the worker actually block in the pause loop
    ctl.cancel()
    t.join(timeout=5)

    assert seen == ["cancelled"]


def test_cancel_unpauses_a_paused_worker():
    ctl = RunControl()
    ctl.pause()
    ctl.cancel()
    # cancel() also clears the pause, so a *new* waiter doesn't block forever
    assert not ctl.is_paused


def test_raise_if_cancelled():
    ctl = RunControl()
    ctl.cancel()
    with pytest.raises(RunCancelled):
        ctl.raise_if_cancelled()


def test_register_and_current():
    ctl = RunControl()
    run_control.register(ctl)
    try:
        assert run_control.current() is ctl
    finally:
        run_control.register(None)
    assert run_control.current() is None
