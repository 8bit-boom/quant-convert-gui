"""Cooperative pause / cancel controls shared by every conversion backend.

A single `RunControl` instance is created per conversion run and threaded
through the backend (in-process INT4/GGUF loops) or the subprocess runner
(ctq). The GUI's Pause / Stop buttons flip the events on the *currently
registered* control, so handlers don't need to know which run is active.

Backends cooperate at tensor boundaries: between tensors they call
`wait_if_paused()` (blocks while paused) and `raise_if_cancelled()`
(abandons the run, keeping whatever checkpoint shards are already on disk).
That keeps pause/cancel safe - a tensor that has started always finishes,
so checkpoint state stays consistent.
"""

from __future__ import annotations

import threading
import time


class RunCancelled(Exception):
    """Raised inside a backend when the user asked to stop & save progress.

    Not an error: the checkpoint on disk is the intended outcome, so the UI
    treats this as a successful 'paused run', not a failure.
    """


class RunControl:
    """Pause/cancel switches for one conversion run.

    - `pause_event` set  -> running; clear -> paused.
    - `cancel_event` set -> stop at the next tensor boundary.
    """

    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.pause_event.set()  # start un-paused
        self.cancel_event = threading.Event()

    # -- user-facing ops (called from Gradio button handlers) --------------

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def toggle_pause(self) -> bool:
        """Flip pause state; returns True if now paused."""
        if self.pause_event.is_set():
            self.pause()
            return True
        self.resume()
        return False

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.set()  # wake a paused worker so it can see the cancel

    @property
    def is_paused(self) -> bool:
        return not self.pause_event.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    # -- backend-facing ops (called from worker threads) -------------------

    def wait_if_paused(self, poll_seconds: float = 0.5) -> bool:
        """Block while paused. Returns True if it actually waited; raises
        RunCancelled if a stop was requested (even if it arrived together
        with the unpause)."""
        waited = False
        while not self.pause_event.is_set():
            waited = True
            if self.cancel_event.is_set():
                raise RunCancelled()
            time.sleep(poll_seconds)
        if self.cancel_event.is_set():
            raise RunCancelled()
        return waited

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled()


_current_lock = threading.Lock()
_current: RunControl | None = None


def register(run: RunControl | None) -> None:
    """Make `run` the control the Pause/Stop buttons act on (or None)."""
    global _current
    with _current_lock:
        _current = run


def current() -> RunControl | None:
    with _current_lock:
        return _current
