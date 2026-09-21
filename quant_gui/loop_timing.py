"""Track per-tensor optimizer-loop phase time from ctq's streamed log.

ctq doesn't log per-phase timestamps, but its optimizer loops are wrapped in
tqdm bars whose frames carry an elapsed-time counter, e.g.:

    Optimizing INT8 (Prodigy-plateau):  73%|██▎  | 1460/2000 [00:11<00:03, 8.1it/s]

Feeding the raw stream through LoopPhaseTimer yields a summary line whenever
one tensor's optimizer phase closes (i.e. the next tensor starts) plus an
overall summary at the end. This is the phase the GPU speed flags
(--fast_math / --loss_sync_batch / --snapshot_interval / --compile_loop)
affect, so it makes their speedup visible per conversion instead of burying
it in total wall time.
"""

from __future__ import annotations

import re

# tqdm frame: "Optimizing ... 1460/2000 [00:11<00:03, 8.1it/s]" — tolerate
# percentage/bar noise between the label and the counters. Elapsed is either
# MM:SS or H:MM:SS inside the brackets; the trailing rate token (8.1it/s or
# 1.2s/it) is optional and used to refine timing when elapsed rounds to 0s
# (tqdm's bracket only has 1-second granularity).
_TQDM_FRAME_RE = re.compile(
    r"Optimizing\b.*?(\d+)\s*/\s*(\d+)\s*\[(\d+):(\d{2})(?::(\d{2}))?<"
    r"(?:(?:[^,]*,\s*)(\d+(?:\.\d+)?)(it/s|s/it))?"
)

# Tensor header: "(12/264) Processing (INT8): blocks.0.mlp.up.weight"
_TENSOR_HEADER_RE = re.compile(r"^\((\d+)/(\d+)\)\s+Processing \(([^)]+)\):\s*(\S+)")


def _elapsed_seconds(m: re.Match) -> float:
    if m.group(5) is not None:  # H:MM:SS
        return int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
    return int(m.group(3)) * 60 + int(m.group(4))  # MM:SS


class LoopPhaseTimer:
    """Stateful per-tensor optimizer-phase tracker. Not thread-safe."""

    def __init__(self) -> None:
        self._current_name: str | None = None
        # elapsed, done, total, ms/iter derived from the rate token (or None)
        self._last_frame: tuple[float, int, int, float | None] | None = None
        self._tensors: list[tuple[str, float, int, int]] = []  # name, elapsed, done, total
        self._skipped = 0

    def feed(self, line: str) -> str | None:
        """Consume one log line; return a summary line when a phase closes."""
        out = None
        header = _TENSOR_HEADER_RE.search(line)
        if header:
            out = self._close_current()
            self._current_name = header.group(4)
            self._last_frame = None
        elif "Skipping tensor:" in line and self._current_name is not None:
            # A skipped tensor closes the previous one without an optimizer phase.
            out = self._close_current()
            self._current_name = None
            self._skipped += 1
        frame = _TQDM_FRAME_RE.search(line)
        if frame and self._current_name is not None:
            ms_from_rate: float | None = None
            if frame.group(6) is not None:
                rate = float(frame.group(6))
                if rate > 0:
                    ms_from_rate = 1000.0 / rate if frame.group(7) == "it/s" else rate * 1000.0
            self._last_frame = (
                _elapsed_seconds(frame),
                int(frame.group(1)),
                int(frame.group(2)),
                ms_from_rate,
            )
        return out

    def finish(self) -> str:
        """Close the last open tensor and return the overall summary."""
        tail = self._close_current()
        self._current_name = None
        lines = [tail] if tail else []
        if self._tensors:
            total_s = sum(t[1] for t in self._tensors)
            avg_ms = 1000.0 * total_s / max(1, sum(max(1, t[2]) for t in self._tensors))
            lines.append(
                f"[loop] Optimizer phase total: {total_s:.1f}s across "
                f"{len(self._tensors)} tensor(s) (~{avg_ms:.1f} ms/iter) — "
                "the GPU speed flags affect exactly this phase."
            )
        else:
            lines.append(
                "[loop] No optimizer-loop timing found (simple mode, or ctq build "
                "without tqdm progress output)."
            )
        return "\n".join(l for l in lines if l)

    def _close_current(self) -> str | None:
        if self._current_name is None:
            return None
        name = self._current_name
        self._current_name = None
        if self._last_frame is None:
            return None
        elapsed, done, total, ms_from_rate = self._last_frame
        # tqdm's bracket has 1s granularity: a phase that took e.g. 0.4s shows
        # [00:00<...]. Estimate it as <0.5s and take per-iter pace from the rate
        # token instead (never derive elapsed from rate — tqdm's rate is
        # smoothed and can be wildly off for short phases).
        sub_second = elapsed == 0
        effective = 0.5 if sub_second else elapsed
        self._tensors.append((name, effective, done, total))
        if sub_second and ms_from_rate is not None:
            ms_per_iter = ms_from_rate
            elapsed_txt = "<0.5s"
        elif sub_second:
            ms_per_iter = 0.0
            elapsed_txt = "<0.5s"
        else:
            ms_per_iter = 1000.0 * elapsed / max(1, done)
            elapsed_txt = f"~{elapsed:.1f}s"
        short = name if len(name) <= 48 else "…" + name[-47:]
        return (
            f"[loop] {short}: optimizer {elapsed_txt} "
            f"({done}/{total} iters, ~{ms_per_iter:.1f} ms/iter)"
        )
