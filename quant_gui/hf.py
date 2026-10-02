"""Download a single safetensors file straight from a Hugging Face URL.

Accepts the URLs users copy out of their browser, e.g.:
  https://huggingface.co/silveroxides/Kroma-Quant/blob/main/foo.safetensors
  https://huggingface.co/silveroxides/Kroma-Quant/resolve/main/foo.safetensors
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from .sharded import INDEX_SUFFIX, SHARD_RE

_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo_id>[^/]+/[^/]+)/(?:blob|resolve)/(?P<revision>[^/]+)/(?P<filename>.+)$"
)


class HFUrlError(ValueError):
    pass


@dataclass
class HFTarget:
    repo_id: str
    revision: str
    filename: str

    @property
    def resolve_url(self) -> str:
        from urllib.parse import quote

        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{quote(self.filename)}"


def parse_hf_url(url: str) -> HFTarget:
    url = url.strip()
    m = _HF_URL_RE.match(url)
    if not m:
        raise HFUrlError(
            "That doesn't look like a Hugging Face file URL. Expected something like "
            "https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors"
        )
    return HFTarget(repo_id=m.group("repo_id"), revision=m.group("revision"), filename=m.group("filename"))


def download(url: str, dest_dir: str, token: str | None = None, progress_cb=None, file_cb=None) -> str:
    """Download the file to dest_dir, returning the local path.

    A URL to one shard of a sharded checkpoint (`model-00001-of-00005.safetensors`)
    or to its `.safetensors.index.json` downloads the *whole set* and returns
    the index path (or the first shard if the repo has no index) - a single
    shard is useless on its own. `file_cb(i, n, name)` reports shard progress.

    Prefers huggingface_hub (handles resume, caching, auth) and falls back
    to a plain streamed HTTP GET when it isn't installed.
    """
    target = parse_hf_url(url)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    base = posixpath.basename(target.filename)
    if SHARD_RE.match(base) or base.endswith(INDEX_SUFFIX):
        return _download_shard_set(target, dest_dir, token, progress_cb, file_cb)
    return _download_one(target, dest_dir, token, progress_cb)


def _download_one(target: HFTarget, dest_dir: str, token: str | None, progress_cb) -> str:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=target.repo_id,
            filename=target.filename,
            revision=target.revision,
            local_dir=dest_dir,
            token=token,
        )
    except ImportError:
        return _plain_download(target, dest_dir, token, progress_cb)


def _sibling(target: HFTarget, name: str) -> HFTarget:
    folder = posixpath.dirname(target.filename)
    return HFTarget(target.repo_id, target.revision, posixpath.join(folder, name) if folder else name)


def _download_shard_set(target: HFTarget, dest_dir: str, token: str | None, progress_cb, file_cb) -> str:
    base = posixpath.basename(target.filename)
    index_local: str | None = None

    if base.endswith(INDEX_SUFFIX):
        index_local = _download_one(target, dest_dir, token, progress_cb)
        try:
            weight_map = json.loads(Path(index_local).read_text(encoding="utf-8"))["weight_map"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Couldn't read the downloaded shard index {base}: {exc}") from exc
        names = sorted(set(weight_map.values()))
    else:
        m = SHARD_RE.match(base)
        prefix, suffix = m.group("prefix"), m.group("suffix")
        width, total = len(m.group("idx")), int(m.group("total"))
        names = [f"{prefix}-{i:0{width}d}-of-{total:0{width}d}{suffix}" for i in range(1, total + 1)]
        try:  # the index is optional - plenty of repos ship shards without one
            index_local = _download_one(_sibling(target, f"{prefix}{INDEX_SUFFIX}"), dest_dir, token, None)
        except Exception:  # noqa: BLE001 - absent index is fine, anything else surfaces via the shards
            index_local = None

    first_local: str | None = None
    for i, name in enumerate(names, 1):
        if file_cb:
            file_cb(i, len(names), name)
        local = _download_one(_sibling(target, name), dest_dir, token, progress_cb)
        first_local = first_local or local
    return index_local or first_local


def download_repo(repo_id: str, dest_dir: str, token: str | None = None, revision: str = "main") -> str:
    """Download a whole model repo (config, tokenizer, safetensors shards) -
    needed for LLM -> GGUF conversion, which reads far more than one file.
    Skips redundant/large weight formats (.bin, .onnx, etc.) since the repo
    almost always also ships safetensors."""
    from huggingface_hub import snapshot_download

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest_dir,
        token=token,
        ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.h5", "*.pth", "*.ckpt", "*.gguf", "*.tflite", "original/*"],
    )


def _plain_download(target: HFTarget, dest_dir: str, token: str | None, progress_cb) -> str:
    import urllib.request

    dest_path = Path(dest_dir) / Path(target.filename).name
    req = urllib.request.Request(target.resolve_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        chunk = 1024 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            read += len(buf)
            if progress_cb:
                progress_cb(read, total)

    return str(dest_path)
