#!/usr/bin/env python3
"""Audit remaining AnyIO usage.

This repo is migrating to AnyIO. This script helps track remaining
imports/usages and can optionally fail CI once an allowlist is in place.

Usage:
  .venv/bin/python tools/audit_asyncio_usage.py
  .venv/bin/python tools/audit_asyncio_usage.py --fail --allowlist .asyncio_allowlist
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ASYNCIO_PATTERNS = [
    re.compile(r"^\s*import\s+anyio\b", re.MULTILINE),
    re.compile(r"^\s*from\s+anyio\b", re.MULTILINE),
    re.compile(r"\banyio\.", re.MULTILINE),
]


@dataclass(frozen=True)
class Hit:
    path: Path
    count: int


def load_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        return set()
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        lines.append(raw)
    return set(lines)


def is_allowed(rel: str, allowlist: set[str]) -> bool:
    for pattern in allowlist:
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--allowlist", default=None)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[
            ".venv/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/node_modules/**",
            ".migrate_backups/**",
            "bin/**",
            "build/**",
            "dist/**",
        ],
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    allowlist = load_allowlist(Path(args.allowlist).resolve() if args.allowlist else None)

    hits: list[Hit] = []

    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()

        excluded = any(fnmatch.fnmatch(rel, pat) for pat in args.exclude)
        if excluded:
            continue

        if is_allowed(rel, allowlist):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        count = 0
        for pat in ASYNCIO_PATTERNS:
            count += len(pat.findall(text))

        if count:
            hits.append(Hit(path=path, count=count))

    hits.sort(key=lambda h: (-h.count, str(h.path)))

    if hits:
        print("Remaining anyio usage (non-allowlisted):")
        for hit in hits:
            rel = hit.path.relative_to(root).as_posix()
            print(f"- {rel}: {hit.count}")
        print(f"Total files: {len(hits)}")
    else:
        print("No non-allowlisted anyio usage found.")

    if args.fail and hits:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
