"""Check candidate whitespace without staging into any live repository index.

This is a validation result, never a completion receipt. The caller still owns
the source/lifecycle fence and must verify its post-validation candidate binding.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


def check_candidate_diff(
    workspace: Path, baseline: str, *, timeout: float = 600.0,
) -> dict[str, Any]:
    """Include committed, staged, working and untracked candidate changes.

Each initialized gitlink is checked against its parent's baseline gitlink,
including nested submodules. Temporary indexes are initialized from HEAD so
live assume-unchanged/skip-worktree flags cannot conceal candidate bytes.
"""
    deadline = time.monotonic() + timeout
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    resolved = ""
    messages: list[str] = []
    result_code = 0

    def git(root: Path, *args: str, index: Path | None = None) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("candidate diff check", timeout)
        env = dict(environment)
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        result = subprocess.run(
            ["git", *args], cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="surrogateescape", timeout=remaining, check=False,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, result.args,
                output=result.stdout, stderr=result.stderr,
            )
        return result.stdout

    def gitlinks(tree: str) -> dict[str, str]:
        links = {}
        for entry in tree.split("\0"):
            if not entry:
                continue
            metadata, path = entry.split("\t", 1)
            mode, _kind, oid = metadata.split()
            if mode == "160000":
                links[path] = oid
        return links

    def check(root: Path, base: str, label: str = "") -> None:
        nonlocal result_code
        if Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
            raise ValueError(f"not an initialized repository: {label or root}")
        base_links = gitlinks(git(root, "ls-tree", "-rz", base))
        with tempfile.TemporaryDirectory(prefix="candidate-diff-index-") as directory:
            index = Path(directory) / "index"
            git(root, "read-tree", "HEAD", index=index)
            git(root, "add", "--all", "--", ".", index=index)
            tree = git(root, "write-tree", index=index).strip()
            candidate_links = gitlinks(git(root, "ls-tree", "-rz", tree))
            try:
                git(root, "diff", "--no-ext-diff", "--no-textconv", "--check", base, tree, "--")
            except subprocess.CalledProcessError as exc:
                result_code = int(exc.returncode)
                messages.extend(
                    (f"[{label}] " if label else "") + line
                    for line in ((exc.output or "") + (exc.stderr or "")).splitlines()
                )
            for relative, oid in candidate_links.items():
                child = root / relative
                child_label = f"{label}/{relative}" if label else relative
                if child.is_symlink() or not child.resolve().is_relative_to(root):
                    raise ValueError(f"gitlink escapes repository: {child_label}")
                if not (child / ".git").exists():
                    if base_links.get(relative) != oid:
                        raise ValueError(f"changed gitlink is not initialized: {child_label}")
                    continue
                child_base = base_links.get(relative)
                if child_base is None:
                    # The empty tree is independent of the repository hash format.
                    child_base = git(child, "hash-object", "-t", "tree", "-w", "--stdin").strip()
                check(child.resolve(), child_base, child_label)

    try:
        root = workspace.resolve(strict=True)
        resolved = git(root, "rev-parse", "--verify", "--end-of-options", f"{baseline}^{{commit}}").strip()
        check(root, resolved)
    except subprocess.TimeoutExpired:
        result_code = 124
        messages.append("candidate diff invariant timed out")
    except subprocess.CalledProcessError as exc:
        result_code = int(exc.returncode or 1)
        messages.append((exc.output or "") + (exc.stderr or ""))
    except (OSError, ValueError) as exc:
        result_code = 1
        messages.append(f"{type(exc).__name__}: {exc}")
    return {"baseline": resolved, "returncode": result_code, "output": "\n".join(messages)}
