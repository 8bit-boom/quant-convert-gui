import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui import imatrix_hub as hub
from quant_gui import llamacpp_backend as lcpp

PAYLOAD = b"GGUF-fake-matrix" * 100


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/models/unsloth/Good-GGUF"):
            body = json.dumps({"siblings": [{"rfilename": "x.gguf"}, {"rfilename": "imatrix_unsloth.gguf_file", "size": len(PAYLOAD)}]}).encode()
        elif self.path.startswith("/api/models/unsloth/Empty-GGUF"):
            body = json.dumps({"siblings": [{"rfilename": "x.gguf"}]}).encode()
        elif self.path.startswith("/unsloth/Good-GGUF/resolve/main/imatrix_unsloth.gguf_file"):
            body = PAYLOAD
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield srv
    srv.shutdown()


def test_repo_derivation(tmp_path):
    assert hub.published_repo("Qwen/Qwen3-4B") == "unsloth/Qwen3-4B-GGUF"
    assert hub.published_repo("unsloth/Foo-bnb-4bit") == "unsloth/Foo-GGUF"
    assert hub.published_repo("unsloth/Foo-GGUF") == "unsloth/Foo-GGUF"
    assert hub.published_repo(str(tmp_path)) is None


def test_lookup_and_download(server, tmp_path):
    info = hub.lookup("someone/Good", tmp_path)
    assert info["file"] == "imatrix_unsloth.gguf_file" and info["local_name"] == "imatrix_unsloth.gguf"
    assert info["size"] == len(PAYLOAD) and not info["cached"]
    seen = []
    f = hub.download(info, progress=lambda a, b: seen.append((a, b)))
    assert f.read_bytes() == PAYLOAD and seen[-1][0] == len(PAYLOAD)
    assert not list(tmp_path.rglob("*.part"))
    assert hub.lookup("someone/Good", tmp_path)["cached"]


@pytest.mark.parametrize("base,kind", [("someone/Empty", "no_file"), ("someone/Missing", "not_found")])
def test_lookup_plain_errors(server, tmp_path, base, kind):
    with pytest.raises(hub.HubError) as e:
        hub.lookup(base, tmp_path)
    assert e.value.kind == kind


def test_local_folder_has_no_published_matrix(tmp_path):
    with pytest.raises(hub.HubError) as e:
        hub.lookup(str(tmp_path), tmp_path)
    assert e.value.kind == "local"


def test_stop_cancels_and_leaves_no_partial(server, tmp_path):
    info = hub.lookup("someone/Good", tmp_path)
    with pytest.raises(hub.HubError) as e:
        hub.download(info, should_stop=lambda: True)
    assert e.value.kind == "cancelled" and not list(tmp_path.rglob("*.part")) and not Path(info["cache_file"]).exists()


def test_merge_imatrix_command_line(tmp_path, monkeypatch):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"x"), b.write_bytes(b"y")
    cmds = []
    monkeypatch.setattr(lcpp, "_imatrix_binary", lambda d: Path("/bin/llama-imatrix"))

    def fake_run(cmd, cwd=None, result=None):
        cmds.append(cmd)
        result.returncode = 0
        yield "ok\n"
    monkeypatch.setattr(lcpp, "_run_streamed", fake_run)
    out = list(lcpp.stream_merge_imatrix(tmp_path, [str(a), str(b)], str(tmp_path / "o" / "m.gguf")))
    assert out[-1] == "__OK__"
    assert cmds[0] == [str(Path("/bin/llama-imatrix")), "--in-file", str(a), "--in-file", str(b), "-o", str(tmp_path / "o" / "m.gguf")]
    assert list(lcpp.stream_merge_imatrix(tmp_path, [str(a)], "x"))[-1].startswith("__FAIL__")


def test_quantize_passes_extra_args_before_files(tmp_path, monkeypatch):
    monkeypatch.setattr(lcpp, "_quantize_binary", lambda d: Path("/bin/llama-quantize"))
    cmds = []

    def fake_run(cmd, cwd=None, result=None):
        cmds.append(cmd)
        result.returncode = 0
        yield ""
    monkeypatch.setattr(lcpp, "_run_streamed", fake_run)
    list(lcpp.stream_quantize(tmp_path, "in.gguf", str(tmp_path / "o.gguf"), "Q4_K_M", imatrix_file="im.gguf",
                              extra_args=["--output-tensor-type", "Q6_K", "--tensor-type", "a=Q5_K"]))
    assert cmds[0] == [str(Path("/bin/llama-quantize")), "--imatrix", "im.gguf", "--output-tensor-type", "Q6_K", "--tensor-type", "a=Q5_K",
                       "in.gguf", str(tmp_path / "o.gguf"), "Q4_K_M"]
