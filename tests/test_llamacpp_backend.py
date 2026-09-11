import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quant_gui.llamacpp_backend import (
    DEFAULT_CALIBRATION_FILE,
    QUANT_TYPE_CHOICES,
    _imatrix_binary,
    _quantize_binary,
    _venv_python,
    is_cloned,
    is_imatrix_built,
    is_quantize_built,
    is_venv_ready,
    stream_build_quantize,
    stream_clone_or_update,
    stream_convert_to_gguf,
    stream_generate_imatrix,
    stream_quantize,
    stream_setup_venv,
)


def test_is_cloned_false_for_empty_dir(tmp_path):
    assert is_cloned(tmp_path) is False


def test_is_cloned_true_when_convert_script_present(tmp_path):
    (tmp_path / "convert_hf_to_gguf.py").write_text("# stub")
    assert is_cloned(tmp_path) is True


def test_is_venv_ready_false_when_no_venv(tmp_path):
    assert is_venv_ready(tmp_path) is False


def test_is_quantize_built_false_when_missing(tmp_path):
    assert is_quantize_built(tmp_path) is False


def test_quantize_binary_finds_unix_binary(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-quantize"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\n")
    binpath.chmod(0o755)
    assert _quantize_binary(tmp_path) == binpath
    assert is_quantize_built(tmp_path) is True


def test_venv_python_path_matches_platform(tmp_path):
    py = _venv_python(tmp_path)
    if platform.system() == "Windows":
        assert py == tmp_path / ".venv" / "Scripts" / "python.exe"
    else:
        assert py == tmp_path / ".venv" / "bin" / "python"


def test_quant_type_choices_are_unique_and_nonempty():
    assert len(QUANT_TYPE_CHOICES) == len(set(QUANT_TYPE_CHOICES))
    assert "Q4_K_M" in QUANT_TYPE_CHOICES
    assert "Q8_0" in QUANT_TYPE_CHOICES


def _make_local_git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "convert_hf_to_gguf.py").write_text("# stub llama.cpp checkout\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


def test_stream_clone_or_update_clones_from_local_repo(tmp_path):
    source = _make_local_git_repo(tmp_path / "source_repo")
    dest = tmp_path / "cloned"

    events = list(stream_clone_or_update(dest, repo_url=str(source)))

    assert events[-1] == "__OK__"
    assert is_cloned(dest)


def test_stream_clone_or_update_pulls_when_already_cloned(tmp_path):
    source = _make_local_git_repo(tmp_path / "source_repo")
    dest = tmp_path / "cloned"
    list(stream_clone_or_update(dest, repo_url=str(source)))  # initial clone

    events = list(stream_clone_or_update(dest, repo_url=str(source)))

    assert events[-1] == "__OK__"
    assert any("pull" in e for e in events)


def test_stream_setup_venv_fails_fast_when_not_cloned(tmp_path):
    events = list(stream_setup_venv(tmp_path))
    assert events[-1].startswith("__FAIL__")


def test_stream_build_quantize_fails_fast_when_not_cloned(tmp_path):
    events = list(stream_build_quantize(tmp_path))
    assert events[-1].startswith("__FAIL__")


def test_stream_convert_to_gguf_fails_fast_when_venv_not_ready(tmp_path):
    events = list(stream_convert_to_gguf(tmp_path, "/some/model/dir", str(tmp_path / "out.gguf")))
    assert events[-1].startswith("__FAIL__")


def test_stream_quantize_fails_fast_when_binary_missing(tmp_path):
    events = list(stream_quantize(tmp_path, "/in.gguf", "/out.gguf", "Q4_K_M"))
    assert events[-1].startswith("__FAIL__")


def test_stream_quantize_rejects_unknown_type(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-quantize"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\n")
    binpath.chmod(0o755)

    events = list(stream_quantize(tmp_path, "/in.gguf", "/out.gguf", "NOT_A_REAL_TYPE"))
    assert events[-1].startswith("__FAIL__")


def test_default_calibration_file_exists_and_is_nonempty():
    assert DEFAULT_CALIBRATION_FILE.is_file()
    assert DEFAULT_CALIBRATION_FILE.stat().st_size > 1000


def test_is_imatrix_built_false_when_missing(tmp_path):
    assert is_imatrix_built(tmp_path) is False


def test_imatrix_binary_finds_unix_binary(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-imatrix"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\n")
    binpath.chmod(0o755)
    assert _imatrix_binary(tmp_path) == binpath
    assert is_imatrix_built(tmp_path) is True


def test_stream_generate_imatrix_fails_fast_when_binary_missing(tmp_path):
    events = list(stream_generate_imatrix(tmp_path, "/model.gguf", str(tmp_path / "imatrix.gguf")))
    assert events[-1].startswith("__FAIL__")


def test_stream_generate_imatrix_fails_fast_when_model_missing(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-imatrix"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\n")
    binpath.chmod(0o755)

    events = list(stream_generate_imatrix(tmp_path, "/no/such/model.gguf", str(tmp_path / "imatrix.gguf")))
    assert events[-1].startswith("__FAIL__")


def test_stream_generate_imatrix_fails_fast_when_calibration_file_missing(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-imatrix"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\n")
    binpath.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_text("fake")

    events = list(stream_generate_imatrix(
        tmp_path, str(model), str(tmp_path / "imatrix.gguf"), calibration_file=str(tmp_path / "missing.txt"),
    ))
    assert events[-1].startswith("__FAIL__")


def test_stream_quantize_includes_imatrix_flag_in_command(tmp_path):
    binpath = tmp_path / "build" / "bin" / "llama-quantize"
    binpath.parent.mkdir(parents=True)
    binpath.write_text("#!/bin/sh\nexit 1\n")
    binpath.chmod(0o755)

    events = list(stream_quantize(
        tmp_path, "/in.gguf", "/out.gguf", "Q4_K_M", imatrix_file="/some/imatrix.gguf",
    ))
    assert "--imatrix /some/imatrix.gguf" in events[0]
