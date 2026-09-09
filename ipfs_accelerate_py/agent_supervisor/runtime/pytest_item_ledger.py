"""Durable pytest-item pass ledger outside ephemeral task worktrees.

Exclusive-owner SIGKILL destroys the leased worktree and its ``.pytest_cache``,
so the next generation re-runs every already-green item. This ledger lives next
to interrupted-validation checkpoints under ``data/<board>/state/`` and records
each call-phase outcome as it happens. A later pytest on a restored worktree
skips a prior pass only when the test file hash, dirty-source fingerprint, and
validation command fingerprint still match. Hash mismatch, corruption, or a
workspace that is not a board worktree fail closed and run the item.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

LEDGER_SCHEMA = "ipfs_accelerate_py/agent-supervisor/pytest-item-ledger@1"
SKIP_REASON = "pytest-item-ledger reuse"
_TASK_BRANCH_RE = re.compile(
    r"^implementation/([a-z][a-z0-9]*-\d+)",
    re.IGNORECASE,
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_JSONL_BYTES = 8 * 1024 * 1024
_PASSED = "passed"


def pytest_item_ledger_dir(repo_root: Path, board: str, task_id: str) -> Path:
    return Path(repo_root) / "data" / board / "state" / "pytest-item-ledger" / task_id


def infer_board_workspace(workspace: Path) -> tuple[Path, str] | None:
    """Return ``(repo_root, board)`` when *workspace* is ``data/<board>/worktrees/*``."""

    try:
        resolved = workspace.resolve()
    except OSError:
        return None
    parts = resolved.parts
    for index, part in enumerate(parts):
        if (
            part == "worktrees"
            and index >= 2
            and parts[index - 2] == "data"
        ):
            board = parts[index - 1]
            if not board or board in {".", ".."}:
                return None
            root = Path(*parts[: index - 2]) if index - 2 > 0 else Path("/")
            return root, board
    return None


def task_id_from_workspace(workspace: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = (completed.stdout or "").strip()
    match = _TASK_BRANCH_RE.match(branch)
    if match is None:
        return None
    return match.group(1).upper()


def command_fingerprint(args: Iterable[str]) -> str:
    payload = json.dumps([str(part) for part in args], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(65536)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def dirty_source_paths(workspace: Path) -> tuple[str, ...]:
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
        if not raw or raw == ".git" or raw.startswith(".git/"):
            continue
        if raw.endswith(".pyc") or "/__pycache__/" in raw or raw.endswith(".pyo"):
            continue
        if raw.startswith(".pytest_cache/") or "/.pytest_cache/" in raw:
            continue
        paths.append(raw)
    return tuple(dict.fromkeys(paths))


def workspace_fingerprint(workspace: Path) -> str:
    entries: list[list[str]] = []
    for relative in dirty_source_paths(workspace):
        digest = file_sha256(workspace / relative)
        if digest is None:
            continue
        entries.append([relative, digest])
    payload = json.dumps(entries, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records_path(dest: Path) -> Path:
    return dest / "items.jsonl"


def load_records(dest: Path) -> dict[str, dict[str, Any]]:
    path = _records_path(dest)
    if not path.is_file():
        return {}
    try:
        if path.stat().st_size > _MAX_JSONL_BYTES:
            return {}
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict) or payload.get("schema") != LEDGER_SCHEMA:
            return {}
        nodeid = str(payload.get("nodeid") or "").strip()
        if not nodeid:
            return {}
        records[nodeid] = payload
    return records


def record_item(
    dest: Path,
    *,
    nodeid: str,
    outcome: str,
    test_file: str,
    test_file_sha256: str,
    workspace_fingerprint_value: str,
    command_sha256: str,
    task_id: str,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LEDGER_SCHEMA,
        "task_id": task_id,
        "nodeid": nodeid,
        "outcome": outcome,
        "test_file": test_file,
        "test_file_sha256": test_file_sha256,
        "workspace_fingerprint": workspace_fingerprint_value,
        "command_sha256": command_sha256,
    }
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    path = _records_path(dest)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def reusable_nodeids(
    records: Mapping[str, Mapping[str, Any]],
    *,
    command_sha256: str,
    workspace_fingerprint_value: str,
    test_file_hashes: Mapping[str, str],
) -> frozenset[str]:
    reusable: set[str] = set()
    for nodeid, payload in records.items():
        if str(payload.get("outcome") or "") != _PASSED:
            continue
        if str(payload.get("command_sha256") or "") != command_sha256:
            continue
        if str(payload.get("workspace_fingerprint") or "") != workspace_fingerprint_value:
            continue
        test_file = str(payload.get("test_file") or "")
        expected = str(payload.get("test_file_sha256") or "")
        current = test_file_hashes.get(test_file)
        if not test_file or not expected or current != expected:
            continue
        reusable.add(nodeid)
    return frozenset(reusable)


def ledger_context(workspace: Path) -> dict[str, Any] | None:
    located = infer_board_workspace(workspace)
    if located is None:
        return None
    repo_root, board = located
    task_id = task_id_from_workspace(workspace)
    if task_id is None:
        return None
    return {
        "repo_root": repo_root,
        "board": board,
        "task_id": task_id,
        "dest": pytest_item_ledger_dir(repo_root, board, task_id),
        "workspace_fingerprint": workspace_fingerprint(workspace),
    }
