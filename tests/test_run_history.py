import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_gui import run_history as rh
from quant_gui.loop_timing import LoopPhaseTimer


def _timer(total_s=4.0, tensors=None):
    timer = LoopPhaseTimer()
    tensors = tensors or [("a.weight", 2.0, 100, 200), ("b.weight", 2.0, 100, 200)]
    for name, elapsed, done, total in tensors:
        timer.feed(f"(1/2) Processing (INT8): {name}")
        timer.feed(
            f"Optimizing INT8 (Prodigy-plateau):  50%|██ | {done}/{total} "
            f"[00:0{int(elapsed)}<00:0{int(elapsed)}, 10it/s]"
        )
    timer.finish()
    return timer


def test_load_missing_file_is_empty(tmp_path):
    assert rh.load(tmp_path / "nope.json") == []


def test_load_corrupt_file_is_empty(tmp_path):
    p = tmp_path / "h.json"
    p.write_text("{not json", encoding="utf-8")
    assert rh.load(p) == []


def test_record_appends_and_adds_timestamp(tmp_path):
    p = tmp_path / "h.json"
    rh.record(p, {"key": "a", "loop": {"total_seconds": 4.0}})
    entries = rh.load(p)
    assert len(entries) == 1
    assert entries[0]["timestamp"]
    assert entries[0]["loop"]["total_seconds"] == 4.0


def test_record_trims_to_keep(tmp_path):
    p = tmp_path / "h.json"
    for i in range(10):
        rh.record(p, {"key": f"k{i}"}, keep=3)
    entries = rh.load(p)
    assert len(entries) == 3
    assert entries[-1]["key"] == "k9"


def test_previous_returns_latest_for_key_excluding_self(tmp_path):
    p = tmp_path / "h.json"
    rh.record(p, {"key": "out.safetensors", "loop": {"total_seconds": 4.0}, "timestamp": "t1"})
    rh.record(p, {"key": "other", "loop": {"total_seconds": 9.0}, "timestamp": "t2"})
    entries = rh.load(p)
    prev = rh.previous(entries, "out.safetensors", before="t3")
    assert prev is not None and prev["timestamp"] == "t1"
    # Same timestamp → excluded (don't compare a run against itself).
    assert rh.previous(entries, "out.safetensors", before="t1") is None
    assert rh.previous(entries, "missing-key", before="tX") is None


def test_comparison_line_faster_slower_same():
    loop = {"tensor_count": 2, "total_seconds": 2.5}
    fast = rh.comparison_line({"timestamp": "t", "loop": {"total_seconds": 4.0}}, loop)
    assert "2.5s vs 4.0s" in fast and "faster" in fast
    slow = rh.comparison_line({"timestamp": "t", "loop": {"total_seconds": 2.0}}, loop)
    assert "slower" in slow
    same = rh.comparison_line({"timestamp": "t", "loop": {"total_seconds": 2.55}}, loop)
    assert "about the same" in same


def test_comparison_line_none_when_no_data():
    assert rh.comparison_line({"timestamp": "t"}, {"tensor_count": 2, "total_seconds": 2.5}) is None
    assert rh.comparison_line({"loop": {"total_seconds": 4.0}}, {"tensor_count": 0}) is None


def test_stats_breakdown_matches_summary():
    timer = _timer()
    stats = timer.stats()
    assert stats["tensor_count"] == 2
    assert stats["total_seconds"] == 4.0
    assert stats["total_iters"] == 200
    assert stats["avg_ms_per_iter"] == 20.0
    assert stats["tensors"][0]["name"] == "a.weight"
    assert stats["tensors"][0]["ms_per_iter"] == 20.0


def test_stats_empty_when_no_loop():
    assert LoopPhaseTimer().stats()["tensor_count"] == 0


def test_roundtrip_entry_is_json_serializable(tmp_path):
    p = tmp_path / "h.json"
    entry = {"key": "out", "loop": _timer().stats(), "resumed": False}
    rh.record(p, entry)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded[0]["loop"]["tensors"][1]["name"] == "b.weight"


def test_rows_render_newest_first_with_metrics_and_comparison(tmp_path):
    p = tmp_path / "h.json"
    rh.record(p, {
        "key": "/m/out.safetensors", "input": "/m/in.safetensors",
        "output": "/m/out.safetensors", "duration_s": 60.0,
        "output_bytes": 4_000_000_000,
        "timestamp": "2026-01-01 00:00:00",
        "loop": {"total_seconds": 40.0, "tensor_count": 336},
    })
    rh.record(p, {
        "key": "/m/out.safetensors", "input": "/m/in.safetensors",
        "output": "/m/out.safetensors", "duration_s": 30.0,
        "output_bytes": 4_000_000_000,
        "timestamp": "2026-01-01 01:00:00",
        "loop": {"total_seconds": 20.0, "tensor_count": 336},
    })
    table = rh.rows(rh.load(p))
    assert len(table) == 2
    newest, oldest = table  # newest first
    assert newest[3] == 4.0            # size GB
    assert newest[4] == 30.0           # wall s
    assert newest[5] == 20.0           # loop s
    assert newest[6] == 336            # tensors
    assert "faster" in newest[8]       # 40s -> 20s vs previous
    assert oldest[8] == ""             # no previous run to compare


def test_rows_tolerate_partial_entries():
    table = rh.rows([{"timestamp": "t", "key": "k", "input": "", "output": ""}])
    assert len(table) == 1
    assert table[0][3] == "" and table[0][8] == ""
