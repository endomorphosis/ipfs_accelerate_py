#!/usr/bin/env python3
"""Initialize declared git submodules, treating local pins as success.

Board validation often runs::

    git submodule update --init --depth 1 external/foo external/bar

When a monorepo pin is already checked out to a local-only commit (or is
already an initialized worktree), a fresh ``--depth 1`` fetch against origin
fails with exit 128 even though the workspace is usable. This helper:

1. Skips paths that are already initialized git worktrees with a resolvable HEAD
2. Otherwise runs ``git submodule update --init --depth 1 -- <path>``
3. After a failed fetch, succeeds if the path became a usable worktree anyway
   (object present locally) so sealed validation can continue to tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _is_initialized_worktree(path: Path) -> bool:
    if not path.exists() or path.is_symlink():
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _init_one(repo_root: Path, relative: str, *, depth: int) -> int:
    relative = relative.strip().strip("/")
    if not relative or ".." in Path(relative).parts:
        print(f"invalid submodule path: {relative!r}", file=sys.stderr)
        return 2
    target = repo_root / relative
    if _is_initialized_worktree(target):
        print(f"already initialized: {relative}")
        return 0
    cmd = [
        "git",
        "submodule",
        "update",
        "--init",
        f"--depth={depth}",
        "--",
        relative,
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        print(f"initialized: {relative}")
        return 0
    # Fetch may fail for local-only pins; accept if the worktree is usable.
    if _is_initialized_worktree(target):
        print(
            f"init fetch failed but worktree usable: {relative} "
            f"(git rc={result.returncode})",
            file=sys.stderr,
        )
        return 0
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    print(
        f"submodule init failed: {relative} (git rc={result.returncode})",
        file=sys.stderr,
    )
    return int(result.returncode or 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Superproject root (default: cwd)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Shallow clone depth for missing submodules (default: 1)",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Submodule paths relative to the superproject",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / ".git").exists() and not (repo_root / ".git").is_file():
        # worktrees use .git file
        if not (repo_root / ".git").exists():
            print(f"not a git repository: {repo_root}", file=sys.stderr)
            return 2
    worst = 0
    for relative in args.paths:
        code = _init_one(repo_root, relative, depth=max(1, int(args.depth)))
        if code != 0:
            worst = code
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
