"""Auto pipeline: sweep -> tune -> validate chaining and stage gating."""
import app


def test_auto_pipeline_stops_when_sweep_fails(monkeypatch):
    monkeypatch.setattr(app, "run_llm_find_best",
                        lambda *a, **k: iter([("sweep log ❌ Sweep failed", None)]))
    out = list(app.run_llm_auto_pipeline("", "4.0", "", "", ""))
    assert "Pipeline stopped" in out[-1][0]
    assert out[-1][1] is None


def test_auto_pipeline_stops_when_no_winner(monkeypatch):
    monkeypatch.setattr(app, "_LAST_SWEEP", {})  # sweep "ran" but picked nothing
    monkeypatch.setattr(app, "run_llm_find_best", lambda *a, **k: iter([("sweep log", None)]))
    out = list(app.run_llm_auto_pipeline("", "4.0", "", "", ""))
    assert "Pipeline stopped" in out[-1][0]


def test_auto_pipeline_chains_all_stages(monkeypatch, tmp_path):
    tuned = str(tmp_path / "tuned.gguf")
    monkeypatch.setattr(app, "_LAST_SWEEP", {"winner_quant": "Q4_K_M"})
    monkeypatch.setattr(app, "run_llm_find_best", lambda *a, **k: iter([("sweep ok", None)]))
    monkeypatch.setattr(app, "run_llm_tune_winner", lambda name: iter([("tune ok", tuned)]))
    monkeypatch.setattr(app, "run_llm_validate", lambda *a: iter([("val ok", None)]))
    out = list(app.run_llm_auto_pipeline("", "4.0", "", "", ""))
    last_log, last_file = out[-1]
    assert "Auto pipeline complete" in last_log
    assert "Stage 1/3" in last_log and "Stage 2/3" in last_log and "Stage 3/3" in last_log
    assert last_file == tuned  # validate yielded no file -> falls back to the tuned one


def test_auto_pipeline_surfaces_tune_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_LAST_SWEEP", {"winner_quant": "Q4_K_M"})
    monkeypatch.setattr(app, "run_llm_find_best", lambda *a, **k: iter([("sweep ok", None)]))
    monkeypatch.setattr(app, "run_llm_tune_winner",
                        lambda name: iter([("tune ❌ failed", None)]))
    out = list(app.run_llm_auto_pipeline("", "4.0", "", "", ""))
    assert "Pipeline stopped" in out[-1][0]
    assert "Stage 3/3" not in out[-1][0]  # validate never ran
