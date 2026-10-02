import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui import hf


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the network call with one that records requests and 'downloads' tiny files."""
    calls = []
    absent = set()

    def fake(target, dest_dir, token, progress_cb):
        calls.append(target.filename)
        if target.filename in absent:
            raise RuntimeError("404")
        out = Path(dest_dir) / target.filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if target.filename.endswith(".index.json"):
            out.write_text(json.dumps({"weight_map": {"a": "m-00001-of-00002.safetensors", "b": "m-00002-of-00002.safetensors"}}))
        else:
            out.write_bytes(b"x")
        return str(out)

    monkeypatch.setattr(hf, "_download_one", fake)
    return calls, absent


URL = "https://huggingface.co/o/r/blob/main/sub/{}"


def test_single_file_url_is_unchanged(fake_hub, tmp_path):
    calls, _ = fake_hub
    path = hf.download(URL.format("plain.safetensors"), str(tmp_path))
    assert calls == ["sub/plain.safetensors"] and path.endswith("plain.safetensors")


def test_any_shard_url_downloads_the_whole_set_and_returns_the_index(fake_hub, tmp_path):
    calls, _ = fake_hub
    seen = []
    path = hf.download(URL.format("m-00002-of-00002.safetensors"), str(tmp_path), file_cb=lambda i, n, name: seen.append((i, n, name)))
    assert "sub/m-00001-of-00002.safetensors" in calls and "sub/m-00002-of-00002.safetensors" in calls
    assert path.endswith("m.safetensors.index.json")
    assert [s[:2] for s in seen] == [(1, 2), (2, 2)]


def test_repo_without_an_index_returns_the_first_shard(fake_hub, tmp_path):
    calls, absent = fake_hub
    absent.add("sub/m.safetensors.index.json")
    path = hf.download(URL.format("m-00001-of-00002.safetensors"), str(tmp_path))
    assert path.endswith("m-00001-of-00002.safetensors")
    assert sum(c.endswith(".safetensors") for c in calls) == 2


def test_index_url_downloads_every_shard_it_lists(fake_hub, tmp_path):
    calls, _ = fake_hub
    path = hf.download(URL.format("m.safetensors.index.json"), str(tmp_path))
    assert path.endswith("m.safetensors.index.json")
    assert {"sub/m-00001-of-00002.safetensors", "sub/m-00002-of-00002.safetensors"} <= set(calls)


def test_a_failed_shard_surfaces_instead_of_returning_a_partial_set(fake_hub, tmp_path):
    _, absent = fake_hub
    absent.add("sub/m-00002-of-00002.safetensors")
    with pytest.raises(RuntimeError):
        hf.download(URL.format("m-00001-of-00002.safetensors"), str(tmp_path))
