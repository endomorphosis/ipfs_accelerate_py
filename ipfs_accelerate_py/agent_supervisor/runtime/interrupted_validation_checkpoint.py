"""Persist implementation worktrees across exclusive-owner SIGKILL.

Grok often finishes ASEH-061, then pytest/auto-rescue is cut by an external
SIGTERM followed ~30s later by SIGKILL. The worktree is ephemeral, so the
next owner re-runs Grok from HEAD. Snapshot dirty files when Grok returns 0
(and again on ignored SIGTERM). Restore them into a fresh worktree so the
daemon can validate without another provider call.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_TASK_BRANCH_RE = re.compile(
    r"^implementation/(aseh-\d+)",
    re.IGNORECASE,
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_CHECKPOINT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/interrupted-validation-checkpoint@1"
)


def interrupted_validation_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "aseh" / "state" / "interrupted-validation"


def _run_git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or "").strip()


def task_id_from_workspace(workspace: Path) -> str | None:
    """Return ASEH-NNN from the implementation branch, if present."""

    branch = _run_git(workspace, "rev-parse", "--abbrev-ref", "HEAD")
    match = _TASK_BRANCH_RE.match(branch)
    if match is None:
        return None
    return match.group(1).upper()


def workspace_is_fresh(workspace: Path) -> bool:
    """Return whether the worktree has no local edits to restore onto."""

    return not _run_git(workspace, "status", "--porcelain=v1")


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    payload = completed.stdout or b""
    paths: list[str] = []
    for entry in payload.split(b"\0"):
        if len(entry) < 4:
            continue
        raw = entry[3:].decode("utf-8", errors="replace").strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        if raw and raw != ".git" and not raw.startswith(".git/"):
            paths.append(raw)
    return tuple(dict.fromkeys(paths))


def _copy_file(source: Path, dest: Path) -> bool:
    try:
        if not source.is_file() or source.is_symlink():
            return False
        size = source.stat().st_size
        if size > _MAX_FILE_BYTES:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return True
    except OSError:
        return False


def snapshot_implementation_workspace(
    workspace: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """Copy dirty implementation files to a durable checkpoint directory."""

    root = Path(repo_root) if repo_root is not None else _infer_repo_root(workspace)
    if root is None:
        return None
    task_id = task_id_from_workspace(workspace)
    if task_id is None:
        return None
    paths = _changed_paths(workspace)
    if not paths:
        return None
    dest_root = interrupted_validation_dir(root) / task_id
    files_root = dest_root / "files"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    copied: list[str] = []
    for relative in paths:
        if _copy_file(workspace / relative, files_root / relative):
            copied.append(relative)
    if not copied:
        return None
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "task_id": task_id,
        "workspace": str(workspace),
        "files": copied,
        "file_count": len(copied),
        "pid": os.getpid(),
    }
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        from .pytest_item_ledger import (
            WORKSPACE_LEDGER_NAME,
            pytest_item_ledger_dir,
            workspace_records_path,
            write_task_id_marker,
        )

        write_task_id_marker(workspace, task_id)
        local_ledger = workspace_records_path(workspace)
        if local_ledger.is_file():
            _copy_file(local_ledger, dest_root / WORKSPACE_LEDGER_NAME)
            durable = pytest_item_ledger_dir(root, "aseh", task_id)
            _copy_file(local_ledger, durable / "items.jsonl")
    except Exception:
        pass
    return payload


def snapshot_dirty_worktrees(repo_root: Path, worktree_root: Path) -> int:
    """Snapshot every dirty leased worktree under ``worktree_root``."""

    root = Path(worktree_root)
    if not root.is_dir():
        return 0
    count = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / ".git").exists():
            continue
        if snapshot_implementation_workspace(child, repo_root=repo_root):
            count += 1
    return count


def restore_interrupted_validation(
    workspace: Path,
    *,
    repo_root: Path | None = None,
) -> bool:
    """Restore a prior snapshot into a fresh worktree. Return True if skipped Grok."""

    if not workspace_is_fresh(workspace):
        return False
    root = Path(repo_root) if repo_root is not None else _infer_repo_root(workspace)
    if root is None:
        return False
    task_id = task_id_from_workspace(workspace)
    if task_id is None:
        return False
    dest_root = interrupted_validation_dir(root) / task_id
    manifest_path = dest_root / "manifest.json"
    files_root = dest_root / "files"
    if not manifest_path.is_file() or not files_root.is_dir():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema") != _CHECKPOINT_SCHEMA:
        return False
    restored = 0
    for relative in payload.get("files") or ():
        text = str(relative).strip()
        if not text or text.startswith("/") or ".." in Path(text).parts:
            continue
        if _copy_file(files_root / text, workspace / text):
            restored += 1
    try:
        from .pytest_item_ledger import (
            WORKSPACE_LEDGER_NAME,
            pytest_item_ledger_dir,
            workspace_records_path,
            write_task_id_marker,
        )

        local_ledger = workspace_records_path(workspace)
        checkpoint_ledger = dest_root / WORKSPACE_LEDGER_NAME
        durable_ledger = pytest_item_ledger_dir(root, "aseh", task_id) / "items.jsonl"
        if checkpoint_ledger.is_file():
            _copy_file(checkpoint_ledger, local_ledger)
        elif durable_ledger.is_file():
            _copy_file(durable_ledger, local_ledger)
        write_task_id_marker(workspace, task_id)
    except Exception:
        pass
    return restored > 0


def _infer_repo_root(workspace: Path) -> Path | None:
    """Worktrees live at <repo>/data/aseh/worktrees/<name>."""

    try:
        resolved = workspace.resolve()
    except OSError:
        return None
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "worktrees" and index >= 2 and parts[index - 1] == "aseh":
            return Path(*parts[: index - 2]) if index - 2 > 0 else Path("/")
    return None
