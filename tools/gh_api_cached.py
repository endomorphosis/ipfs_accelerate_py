#!/usr/bin/env python3

"""Cache-aware wrapper for `gh api`.

Goal: ensure CI/CD scripts go through GitHubAPICache instead of repeatedly
hitting GitHub REST endpoints.

This is intentionally minimal (supports the patterns used in this repo):
- `--jq <expr>`: apply jq to cached JSON
- `-i/--include`: pass-through (no caching)
- `--ttl <seconds>`: override cache TTL

Examples:
  tools/gh_api_cached.py user --jq '.login'
  tools/gh_api_cached.py rate_limit
  tools/gh_api_cached.py repos/OWNER/REPO/commits --jq '.[0].commit.committer.date'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure local imports work even if the package isn't installed.
_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run(cmd: Sequence[str], *, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        input=stdin,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )


def _apply_jq(jq_expr: str, json_text: str) -> str:
    proc = _run(["jq", "-r", jq_expr], stdin=json_text)
    if proc.returncode != 0:
        raise RuntimeError(f"jq failed: {proc.stderr.strip()}")
    return proc.stdout


class _SimpleDiskCache:
    def __init__(self, cache_dir: Path, default_ttl: int = 300):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"gh_api_{digest}.json"

    def _load(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        payload = self._load(path)
        if not payload:
            return None
        stored_at = float(payload.get("stored_at", 0))
        ttl = int(payload.get("ttl", self.default_ttl))
        if time.time() - stored_at > ttl:
            return None
        return payload.get("data")

    def get_stale(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        payload = self._load(path)
        if not payload:
            return None
        return payload.get("data")

    def put(self, key: str, data: Any, ttl: Optional[int]) -> None:
        path = self._key_to_path(key)
        effective_ttl = int(ttl if ttl is not None else self.default_ttl)
        payload = {"stored_at": time.time(), "ttl": effective_ttl, "data": data}
        path.write_text(json.dumps(payload))


def _get_cache_backend() -> Tuple[str, Any]:
    """Return ('ipfs', cache) if repo cache is importable, otherwise ('simple', cache)."""
    cache_dir_env = os.environ.get("CACHE_DIR") or os.environ.get("IPFS_ACCELERATE_CACHE_DIR")
    cache_dir = Path(cache_dir_env) if cache_dir_env else (Path.home() / ".cache" / "github_cli")
    default_ttl = int(os.environ.get("CACHE_DEFAULT_TTL", "300"))

    try:
        from ipfs_accelerate_py.github_cli.cache import get_global_cache

        # Ensure the global cache respects our chosen dir, even if the library
        # defaults differ.
        os.environ.setdefault("CACHE_DIR", str(cache_dir))
        cache = get_global_cache()
        return "ipfs", cache
    except Exception:
        return "simple", _SimpleDiskCache(cache_dir=cache_dir, default_ttl=default_ttl)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("endpoint", help="GitHub API endpoint, like 'user' or '/user'")
    parser.add_argument("--jq", dest="jq", default=None, help="jq expression to apply")
    parser.add_argument("-i", "--include", action="store_true", help="Include HTTP response headers (pass-through)")
    parser.add_argument("--ttl", type=int, default=None, help="Cache TTL override in seconds")
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to `gh api` (excluding --jq)",
    )

    args = parser.parse_args(argv)

    # `argparse.REMAINDER` keeps a leading `--` if present; strip it.
    forward_args: List[str] = list(args.rest)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    # `-i/--include` output includes headers and/or redirects; keep it pass-through.
    if args.include:
        proc = _run(["gh", "api", "-i", args.endpoint] + forward_args)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    backend_name, cache = _get_cache_backend()

    cache_key = json.dumps({"endpoint": args.endpoint, "args": forward_args}, sort_keys=True)

    if backend_name == "ipfs":
        cached = cache.get("gh_api", args.endpoint, tuple(forward_args))
    else:
        cached = cache.get(cache_key)
    if cached is None:
        proc = _run(["gh", "api", args.endpoint] + forward_args)
        if proc.returncode != 0:
            # If we're rate-limited or otherwise failing, fall back to stale cache if available.
            if backend_name == "ipfs":
                stale = cache.get_stale("gh_api", args.endpoint, tuple(forward_args))
            else:
                stale = cache.get_stale(cache_key)
            if stale is not None:
                cached = stale
            else:
                sys.stderr.write(proc.stderr)
                return proc.returncode
        else:
            try:
                cached = json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Failed to parse gh api JSON output: {e}\n")
                sys.stderr.write(proc.stdout)
                return 2

            if backend_name == "ipfs":
                cache.put("gh_api", cached, args.ttl, args.endpoint, tuple(forward_args))
            else:
                cache.put(cache_key, cached, args.ttl)

    if args.jq:
        json_text = json.dumps(cached)
        sys.stdout.write(_apply_jq(args.jq, json_text))
        return 0

    sys.stdout.write(json.dumps(cached) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
