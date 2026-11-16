"""Unsloth's published importance matrix for a base model, looked up on the Hugging Face Hub.

Unsloth's own code (`unsloth/save.py`) looks for `imatrix_unsloth.dat` /
`imatrix_unsloth.gguf_file` / `imatrix_unsloth.gguf` in the repo
`unsloth/<model name>-GGUF`; this module derives the repo the same way and
downloads that file (urllib only, no extra dependency). Ported from the owner's
8bit-boom/Llm-training worker (imatrixmerge.py), minus its job protocol.

The published matrix describes the UNMODIFIED base model: it is a reasonable
start for a light fine-tune of that model and a poor one for a heavily changed
model. HF_TOKEN / HF_ENDPOINT are read from the environment (the endpoint for
mirrors and tests); the token is only ever sent to the Hub host itself.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

# unsloth_zoo.llama_cpp.IMATRIX_UPSTREAM_NAMES, in its order; `.gguf_file` is renamed to `.gguf` locally
UPSTREAM_NAMES = ("imatrix_unsloth.dat", "imatrix_unsloth.gguf_file", "imatrix_unsloth.gguf")
ENDPOINT = "https://huggingface.co"
TIMEOUT = 30
MAX_PUBLISHED = 4 << 30


class HubError(RuntimeError):
    """A Hub problem in plain words. `kind`: offline | not_found | no_file | gated | local | http | cancelled."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or ENDPOINT).rstrip("/")


def token() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()


def published_repo(base: str) -> str | None:
    """`owner/name` -> `unsloth/<name>-GGUF` (a name already ending in -GGUF is kept; Unsloth's
    `-unsloth-bnb-4bit` / `-bnb-4bit` suffixes are undone; a local folder has none)."""
    name = str(base or "").strip()
    if not name or os.path.isdir(name):
        return None
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name if name.endswith("-GGUF") else f"unsloth/{name.split('/')[-1]}-GGUF"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _open(url: str, depth: int = 0):
    """GET with redirects followed by hand so the token only goes to the Hub, never to its CDN."""
    headers = {"User-Agent": "quant-convert-gui"}
    tok = token()
    if tok and urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(endpoint()).netloc:
        headers["Authorization"] = "Bearer " + tok
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location") and depth < 5:
            return _open(urllib.parse.urljoin(url, exc.headers["Location"]), depth + 1)
        raise


def _hub_error(exc: BaseException, repo: str, what: str) -> HubError:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return HubError("gated", f"The Hub refused {what}: {repo} needs you to accept its terms on its Hugging Face page "
                                     f"and set HF_TOKEN. (HTTP {exc.code})")
        if exc.code == 404:
            return HubError("not_found", f"The Hub has no repository {repo}.")
        return HubError("http", f"The Hub answered {what} with an error (HTTP {exc.code}). Try again later.")
    return HubError("offline", f"huggingface.co could not be reached ({type(exc).__name__}). Only this download needs the internet.")


def lookup(base: str, cache: str | Path | None) -> dict:
    """{"base","repo","file","local_name","size","url","page","gated","cached","cache_file"}; raises HubError."""
    repo = published_repo(base)
    if repo is None:
        raise HubError("local", f"{base or 'This model'} is a folder on this computer, so it has no published matrix on the Hub. "
                                "Use the original model's Hugging Face id (e.g. google/gemma-3-4b-it).")
    url = f"{endpoint()}/api/models/{urllib.parse.quote(repo, safe='/')}?blobs=true"
    try:
        with _open(url) as r:
            info = json.loads(r.read(1 << 24).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404 or (exc.code == 401 and not token()):
            raise HubError("not_found", f"There is no published matrix for {base}: the Hub has no repository {repo} (or it is private). "
                                        "Measure a matrix yourself with llama-imatrix instead.")
        raise _hub_error(exc, repo, "the lookup")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise _hub_error(exc, repo, "the lookup")
    files = {s.get("rfilename"): s for s in info.get("siblings") or [] if isinstance(s, dict)}
    name = next((n for n in UPSTREAM_NAMES if n in files), None)
    if name is None:
        raise HubError("no_file", f"{repo} exists but holds none of {', '.join(UPSTREAM_NAMES)}: Unsloth publishes no matrix for {base}.")
    size = files[name].get("size")
    size = size if isinstance(size, int) else None
    local_name = name[: -len(".gguf_file")] + ".gguf" if name.endswith(".gguf_file") else name
    cache_file = Path(cache) / repo.replace("/", "--") / local_name if cache else None
    cached = bool(cache_file and cache_file.is_file() and (size is None or cache_file.stat().st_size == size))
    return {"base": base, "repo": repo, "file": name, "local_name": local_name, "size": size, "page": f"{endpoint()}/{repo}",
            "url": f"{endpoint()}/{urllib.parse.quote(repo, safe='/')}/resolve/main/{name}", "gated": bool(info.get("gated")),
            "cached": cached, "cache_file": str(cache_file) if cache_file else None}


def download(info: dict, progress: Callable[[int, int], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> Path:
    """Downloads into `info["cache_file"]` (skipped when already cached at the Hub's size). Returns the file."""
    import shutil

    target = Path(info["cache_file"])
    if info["cached"]:
        return target
    if info.get("size") and info["size"] > MAX_PUBLISHED:
        raise HubError("http", f"The published matrix is {info['size'] / 1e9:.1f} GB, far more than a matrix should be; not downloaded.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if info.get("size") and shutil.disk_usage(target.parent).free < info["size"] * 1.1:
        raise HubError("http", f"Not enough free disk space in {target.parent} for the {info['size'] / 1e6:.0f} MB matrix.")
    part = target.with_name(target.name + ".part")
    try:
        with _open(info["url"]) as r, open(part, "wb") as out:
            total = int(r.headers.get("Content-Length") or info.get("size") or 0)
            got = 0
            while True:
                if should_stop and should_stop():
                    raise HubError("cancelled", "Download stopped.")
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                if progress:
                    progress(got, total)
        if info.get("size") and part.stat().st_size != info["size"]:
            raise HubError("offline", "The download was cut off. Try again.")
        os.replace(part, target)
    except HubError:
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise _hub_error(exc, info["repo"], "the download")
    return target
