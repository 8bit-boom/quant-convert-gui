"""Download a single safetensors file straight from a Hugging Face URL.

Accepts the URLs users copy out of their browser, e.g.:
  https://huggingface.co/silveroxides/Kroma-Quant/blob/main/foo.safetensors
  https://huggingface.co/silveroxides/Kroma-Quant/resolve/main/foo.safetensors
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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


def download(url: str, dest_dir: str, token: str | None = None, progress_cb=None) -> str:
    """Download the file to dest_dir, returning the local path.

    Prefers huggingface_hub (handles resume, caching, auth) and falls back
    to a plain streamed HTTP GET when it isn't installed.
    """
    target = parse_hf_url(url)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=target.repo_id,
            filename=target.filename,
            revision=target.revision,
            local_dir=dest_dir,
            token=token,
        )
        return local_path
    except ImportError:
        return _plain_download(target, dest_dir, token, progress_cb)


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
