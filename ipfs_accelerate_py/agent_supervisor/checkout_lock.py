"""Reusable checkout mutation lock helpers for autonomous agent supervisors."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHECKOUT_MUTATION_LOCK_NAME = "implementation-main-merge.lock"


def git_common_dir(repo_root: Path) -> Path:
    """Return the repository common git directory for a checkout."""

    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return repo_root / ".git"
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else repo_root / path


def checkout_mutation_lock_path(
    repo_root: Path,
    *,
    lock_name: str = DEFAULT_CHECKOUT_MUTATION_LOCK_NAME,
) -> Path:
    """Return a repo-wide lock path for parent checkout mutations."""

    return git_common_dir(repo_root) / lock_name


def checkout_lock_metadata(
    *,
    kind: str,
    repo_root: Path,
    task_id: str = "",
    branch: str = "",
    attempt: int = 0,
    owner_script: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable metadata for a checkout mutation lock."""

    payload: dict[str, Any] = {
        "kind": kind,
        "pid": os.getpid(),
        "owner_script": owner_script if owner_script is not None else Path(sys.argv[0]).name,
        "repo_root": str(repo_root.resolve()),
        "task_id": task_id,
        "attempt": int(attempt or 0),
        "branch": branch,
    }
    if extra:
        payload.update(extra)
    return payload


def checkout_lock_owner_is_active(
    metadata: dict[str, Any],
    *,
    expected_kind: str,
    expected_repo_root: Path | None = None,
    process_command_line: Any,
    process_is_running: Any,
) -> bool:
    """Return whether lock metadata still belongs to a live compatible process."""

    kind = str(metadata.get("kind") or "")
    if kind and kind != expected_kind:
        return False
    repo_root = str(metadata.get("repo_root") or "")
    if expected_repo_root is not None and repo_root:
        try:
            if Path(repo_root).resolve() != expected_repo_root.resolve():
                return False
        except OSError:
            return False
    try:
        pid = int(metadata.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if not process_is_running(pid):
        return False
    owner_script = str(metadata.get("owner_script") or "")
    command_line = process_command_line(pid)
    if owner_script and owner_script not in command_line:
        return False
    return True
