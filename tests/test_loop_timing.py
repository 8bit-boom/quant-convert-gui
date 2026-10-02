import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui.loop_timing import LoopPhaseTimer, _TQDM_FRAME_RE, _elapsed_seconds


def test_tqdm_frame_regex_mmss():
    m = _TQDM_FRAME_RE.search(
        "    Optimizing INT8 (Prodigy-plateau):  73%|██▎  | 1460/2000 [00:11<00:03, 8.1it/s]"
    )
    assert m is not None
    assert int(m.group(1)) == 1460 and int(m.group(2)) == 2000
    assert _elapsed_seconds(m) == 11.0


def test_tqdm_frame_regex_hhmmss():
    m = _TQDM_FRAME_RE.search(
        "Optimizing (AdamW-plateau):  10%|█  | 500/5000 [1:23:45<12:00:00, 1.2s/it]"
    )
    assert m is not None
    assert _elapsed_seconds(m) == 1 * 3600 + 23 * 60 + 45


def test_frame_without_optimizing_label_ignored():
    assert _TQDM_FRAME_RE.search("Loading tensors: 100%|...| 41/41 [00:00, ...]") is None


def test_per_tensor_summary_on_next_header():
    timer = LoopPhaseTimer()
    assert timer.feed("(1/3) Processing (INT8): blocks.0.mlp.up.weight") is None
    note = timer.feed(
        "Optimizing INT8 (Prodigy-plateau):  73%|██ | 1460/2000 [00:11<00:03, 8.1it/s]"
    )
    assert note is None  # frames don't emit; the next header closes the phase
    note = timer.feed("(2/3) Processing (INT8): blocks.1.mlp.up.weight")
    assert note is not None
    assert "blocks.0.mlp.up.weight" in note
    assert "~11.0s" in note and "1460/2000" in note and "7.5 ms/iter" in note


def test_finish_emits_total_and_phase_note():
    timer = LoopPhaseTimer()
    timer.feed("(1/2) Processing (INT8): a.weight")
    timer.feed("Optimizing INT8 (Prodigy-plateau):  50%|██ | 100/200 [00:05<00:05, 20it/s]")
    timer.feed("(2/2) Processing (INT8): b.weight")
    timer.feed("Optimizing INT8 (Prodigy-plateau):  25%|██ | 50/200 [00:03<00:09, 16it/s]")
    summary = timer.finish()
    assert "Optimizer phase total: 8.0s across 2 tensor(s)" in summary
    assert "GPU speed flags affect exactly this phase" in summary


def test_simple_mode_reports_no_loop():
    timer = LoopPhaseTimer()
    timer.feed("(1/1) Processing (INT8): a.weight")
    summary = timer.finish()
    assert "No optimizer-loop timing found" in summary


def test_skipped_tensor_closes_phase_without_summary():
    timer = LoopPhaseTimer()
    timer.feed("(1/2) Processing (INT8): a.weight")
    timer.feed("Optimizing INT8 (Prodigy-plateau):  50%|██ | 100/200 [00:05<00:05, 20it/s]")
    note = timer.feed("(2/2) Skipping tensor: b.weight (Reason: exclude)")
    assert note is not None and "a.weight" in note  # a's phase closed by the skip line
    summary = timer.finish()
    assert "across 1 tensor(s)" in summary


def test_carriage_return_chunks_split_correctly():
    # Real stream: tqdm frames separated by \r inside one chunk.
    timer = LoopPhaseTimer()
    timer.feed("(1/1) Processing (INT8): a.weight")
    chunk = "Optimizing INT8 (Prodigy):   0%|  | 0/100 [00:00<?, ?it/s]\rOptimizing INT8 (Prodigy):  50%|██ | 50/100 [00:04<00:04, 12it/s]"
    for line in chunk.splitlines():
        timer.feed(line)
    summary = timer.finish()
    assert "~4.0s" in summary


def test_sub_second_phase_uses_rate_for_ms_per_iter():
    # tqdm's [MM:SS bracket has 1s granularity; a fast phase shows [00:00...
    timer = LoopPhaseTimer()
    timer.feed("(1/1) Processing (INT8): a.weight")
    timer.feed(
        "Optimizing INT8 (Prodigy-plateau):  14%|█▎ | 289/2000 [00:00<00:03, 8.1it/s]"
    )
    summary = timer.finish()
    assert "<0.5s" in summary
    assert "~123.5 ms/iter" in summary  # 1000 / 8.1


def test_sub_second_phase_without_rate_token():
    timer = LoopPhaseTimer()
    timer.feed("(1/1) Processing (INT8): a.weight")
    timer.feed("Optimizing INT8 (Prodigy-plateau):  14%|█▎ | 289/2000 [00:00<00:03, ?it/s]")
    summary = timer.finish()
    assert "<0.5s" in summary
