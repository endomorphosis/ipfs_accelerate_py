"""Backlog-refinery helpers for autonomous agent supervisors.

This module ports the repo-local supervisor feed logic into ``ipfs_accelerate_py``
without depending on the ``ipfs_datasets_py`` implementation package.  It keeps
the reusable pieces close to the accelerator daemon runtime:

* refill low todo queues from an objective heap,
* scan tracked code for small bug/improvement findings,
* turn repeated implementation, validation, or merge failures into
  evidence-backed follow-up tasks instead of allowing indefinite retry loops.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .event_log import read_jsonl_events
from .objective_graph import (
    DEFAULT_DISCOVERY_OUTPUT_PATH,
    DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    bundle_path,
    generate_objective_todos,
    repo_relative_path,
    safe_bundle_key,
)
from .todo_daemon.implementation_daemon import (
    is_retry_budget_repair_task,
    parse_task_file,
    retry_budget_repair_source,
)
from .validation_commands import split_validation_commands
from .wrapper_utils import AgentSupervisorNamespacePaths


logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.backlog_refinery")

DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_CODEBASE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_VALIDATION_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_VALIDATION_RETRY_BUDGET", "3"))
DEFAULT_MERGE_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_MERGE_RETRY_BUDGET", "3"))
DEFAULT_IMPLEMENTATION_RETRY_BUDGET = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_RETRY_BUDGET", "3")
)
DEFAULT_STALE_GIT_LOCK_SECONDS = float(
    os.environ.get("IPFS_ACCELERATE_AGENT_STALE_GIT_LOCK_SECONDS", "300")
)
DEFAULT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS", "5")
)
DEFAULT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS", "3")
)
DEFAULT_TASK_ID_PREFIX = "AUTO-"
DEFAULT_TASK_HEADER_PREFIX = "## AUTO-"
CODEBASE_SCAN_MAX_FILE_BYTES = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FILE_BYTES", "262144"))
CODEBASE_SCAN_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
CODEBASE_SCAN_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}
CODEBASE_SCAN_SKIP_PREFIXES = (
    "archive/",
    "backup/",
    "cleanup-archive/",
    "data/agent_supervisor/discovery/",
    "data/agent_supervisor/objective_bundles/",
    "data/agent_supervisor/objective_datasets/",
    "data/agent_supervisor/state/",
    "data/agent_supervisor/worktrees/",
    "external/ipfs_accelerate/test/duckdb_api/",
    "external/ipfs_accelerate/test/generators/",
    "external/ipfs_accelerate/test/huggingface_transformers/",
    "external/ipfs_accelerate/test/skills/",
    "external/ipfs_kit/archive/",
    "external/ipfs_kit/backup/",
)
ANNOTATION_FOLLOWUP_RE = re.compile(
    r"""
    (?:
        ^\s*(?:[-*]\s*)?(?P<line_marker>todo|fixme|hack|xxx)\b\s*(?::|\(|-)
        |
        (?P<comment_prefix>\#|//|/\*|<!--|--)\s*(?P<comment_marker>todo|fixme|hack|xxx)\b\s*(?::|\(|-)
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class CodebaseFinding:
    """One static codebase finding that can be converted to a todo task."""

    fingerprint: str
    kind: str
    priority: str
    track: str
    root_relative_path: str
    line_number: int
    snippet: str
    summary: str
    validation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_id_prefix(value: str) -> str:
    value = str(value or DEFAULT_TASK_ID_PREFIX).strip()
    if value.startswith("## "):
        value = value[3:].strip()
    return value or DEFAULT_TASK_ID_PREFIX


def task_header_prefix(value: str) -> str:
    value = str(value or DEFAULT_TASK_HEADER_PREFIX).strip()
    if value.startswith("## "):
        return value
    return f"## {value}"


def split_csv(values: Iterable[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    items: list[str] = []
    for value in raw_values:
        for raw in str(value).split(","):
            item = " ".join(raw.strip().split())
            if item and item.lower() not in {"none", "n/a"}:
                items.append(item)
    return items


def task_ids_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> list[str]:
    prefix = task_id_prefix(task_prefix)
    ids: list[str] = []
    for line in todo_text.splitlines():
        if not line.startswith(f"## {prefix}"):
            continue
        parts = line[3:].strip().split(" ", 1)
        if parts:
            ids.append(parts[0])
    return ids


def task_block_is_present(todo_text: str, task_id: str) -> bool:
    """Return whether a markdown task block for ``task_id`` is already present."""

    escaped_task_id = re.escape(str(task_id).strip())
    if not escaped_task_id:
        return False
    return re.search(rf"^##\s+{escaped_task_id}(?:\s|$)", todo_text, flags=re.MULTILINE) is not None


def ensure_task_blocks_present(
    todo_path: Path,
    task_blocks: Mapping[str, str] | Sequence[tuple[str, str]],
) -> bool:
    """Append missing markdown task blocks to a todo board in caller-provided order."""

    if not todo_path.exists():
        return False
    todo_text = todo_path.read_text(encoding="utf-8")
    entries = task_blocks.items() if isinstance(task_blocks, Mapping) else task_blocks
    additions = [
        block.strip()
        for task_id, block in entries
        if block.strip() and not task_block_is_present(todo_text, task_id)
    ]
    if not additions:
        return False
    todo_path.write_text(todo_text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n", encoding="utf-8")
    return True


def build_task_blocks_ensurer(
    task_blocks: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    default_todo_path: Path | None = None,
) -> Callable[[Path | None], bool]:
    """Build a callback that appends configured task blocks to a todo board."""

    configured_blocks = dict(task_blocks.items() if isinstance(task_blocks, Mapping) else task_blocks)

    def ensurer(todo_path: Path | None = None) -> bool:
        path = todo_path or default_todo_path
        if path is None:
            raise ValueError("todo_path is required when no default todo path is configured")
        return ensure_task_blocks_present(path, configured_blocks)

    return ensurer


def next_task_id(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> str:
    prefix = task_id_prefix(task_prefix)
    highest = 0
    for current in task_ids_from_todo_text(todo_text, task_prefix=prefix):
        try:
            highest = max(highest, int(current.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"{prefix}{highest + 1:03d}"


def task_statuses_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> dict[str, str]:
    prefix = task_id_prefix(task_prefix)
    statuses: dict[str, str] = {}
    current_task_id = ""
    for line in todo_text.splitlines():
        if line.startswith(f"## {prefix}"):
            parts = line[3:].strip().split(" ", 1)
            current_task_id = parts[0] if parts else ""
            continue
        if current_task_id and line.startswith("- Status:"):
            statuses[current_task_id] = line.split(":", 1)[1].strip().lower()
            current_task_id = ""
    return statuses


def mark_task_statuses_in_todo_text(
    todo_text: str,
    task_ids: Sequence[str],
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    status: str = "completed",
) -> tuple[str, list[str]]:
    """Return todo text with selected task status lines rewritten."""

    prefix = task_id_prefix(task_prefix)
    target_task_ids = {
        str(task_id).strip()
        for task_id in task_ids
        if str(task_id).strip()
    }
    if not target_task_ids:
        return todo_text, []

    lines = todo_text.splitlines(keepends=True)
    current_task_id = ""
    updated_task_ids: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith(f"## {prefix}"):
            parts = line[3:].strip().split(" ", 1)
            current_task_id = parts[0] if parts else ""
            continue
        if current_task_id not in target_task_ids or not line.startswith("- Status:"):
            continue
        current_status = line.split(":", 1)[1].strip().lower()
        if current_status == status.lower():
            current_task_id = ""
            continue
        newline = "\n" if line.endswith("\n") else ""
        lines[index] = f"- Status: {status}{newline}"
        updated_task_ids.append(current_task_id)
        current_task_id = ""
    if not updated_task_ids:
        return todo_text, []
    return "".join(lines), updated_task_ids


def open_task_count(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> int:
    statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    return sum(1 for status in statuses.values() if status not in {"completed", "blocked"})


def load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy(path: Path) -> dict[str, Any]:
    repair_reason = ""
    if not path.exists():
        strategy: dict[str, Any] = {}
        repair_reason = "missing_strategy_file"
    else:
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
        except OSError:
            strategy = {}
            repair_reason = "unreadable_strategy_file"
        else:
            if not raw_text:
                strategy = {}
                repair_reason = "empty_strategy_file"
            else:
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    strategy = {}
                    repair_reason = "invalid_strategy_json"
                else:
                    if isinstance(payload, dict):
                        strategy = dict(payload)
                    else:
                        strategy = {}
                        repair_reason = "non_object_strategy_json"
    if not strategy:
        strategy = {"blocked_tasks": []}
    blocked = strategy.get("blocked_tasks")
    strategy["blocked_tasks"] = [str(item) for item in blocked] if isinstance(blocked, list) else []
    if repair_reason:
        strategy["last_strategy_repair_at"] = utc_now()
        strategy["last_strategy_repair_reason"] = repair_reason
        write_json(path, strategy)
    return strategy


def effective_open_task_count(
    todo_text: str,
    *,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> int:
    if state_path is None or not state_path.exists():
        return open_task_count(todo_text, task_prefix=task_prefix)
    payload = load_json_dict(state_path)
    statuses = payload.get("task_statuses")
    if not isinstance(statuses, dict):
        return open_task_count(todo_text, task_prefix=task_prefix)
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    normalized = {str(task_id): str(status).lower() for task_id, status in statuses.items()}
    if set(normalized) != task_ids:
        return open_task_count(todo_text, task_prefix=task_prefix)
    try:
        state_task_count = int(payload.get("task_count") or 0)
    except (TypeError, ValueError):
        return open_task_count(todo_text, task_prefix=task_prefix)
    if state_task_count != len(task_ids):
        return open_task_count(todo_text, task_prefix=task_prefix)
    return sum(1 for status in normalized.values() if status not in {"completed", "blocked"})


def refill_state_counts(
    todo_text: str,
    *,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> dict[str, int]:
    if state_path is None or not state_path.exists():
        return {}
    payload = load_json_dict(state_path)
    statuses = payload.get("task_statuses")
    if not isinstance(statuses, dict):
        return {}
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    normalized = {str(task_id): str(status).lower() for task_id, status in statuses.items()}
    if set(normalized) != task_ids:
        return {}
    try:
        state_task_count = int(payload.get("task_count") or 0)
    except (TypeError, ValueError):
        return {}
    if state_task_count != len(task_ids):
        return {}

    def count(name: str, fallback: int) -> int:
        try:
            return int(payload.get(name))
        except (TypeError, ValueError):
            return fallback

    completed = sum(1 for status in normalized.values() if status == "completed")
    blocked = sum(1 for status in normalized.values() if status == "blocked")
    ready = sum(1 for status in normalized.values() if status == "todo")
    waiting = sum(1 for status in normalized.values() if status == "waiting")
    return {
        "task_count": state_task_count,
        "completed_count": count("completed_count", completed),
        "blocked_count": count("blocked_count", blocked),
        "ready_count": count("ready_count", ready),
        "selectable_ready_count": count("selectable_ready_count", ready),
        "eligible_ready_count": count("eligible_ready_count", ready),
        "strict_deprioritized_ready_count": count("strict_deprioritized_ready_count", 0),
        "waiting_count": count("waiting_count", waiting),
    }


def parse_iso_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def should_refill_backlog(
    *,
    todo_text: str,
    state_path: Path | None,
    strategy: Mapping[str, Any],
    last_scan_key: str,
    last_drained_scan_task_count_key: str,
    task_prefix: str,
    min_open_tasks: int,
    cooldown_seconds: int,
    force: bool = False,
) -> tuple[bool, str, int, int]:
    current_open = effective_open_task_count(todo_text, state_path=state_path, task_prefix=task_prefix)
    task_count = len(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    state_counts = refill_state_counts(todo_text, state_path=state_path, task_prefix=task_prefix)
    ready_for_refill = int(state_counts.get("eligible_ready_count", state_counts.get("ready_count") or 0) or 0)
    no_ready_existing_work = (
        bool(state_counts)
        and ready_for_refill == 0
        and int(state_counts.get("completed_count") or 0) > 0
        and (int(state_counts.get("waiting_count") or 0) > 0 or int(state_counts.get("blocked_count") or 0) > 0)
    )
    if force:
        return True, "force", current_open, task_count
    if current_open > min_open_tasks and not no_ready_existing_work:
        return False, "open_task_threshold", current_open, task_count
    drained = current_open == 0
    try:
        last_drained_count = int(strategy.get(last_drained_scan_task_count_key) or -1)
    except (TypeError, ValueError):
        last_drained_count = -1
    if drained and last_drained_count != task_count:
        return True, "drained_exhaustive", current_open, task_count
    if no_ready_existing_work and last_drained_count != task_count:
        return True, "runnable_drained_exhaustive", current_open, task_count
    last_scan_at = parse_iso_timestamp(str(strategy.get(last_scan_key) or ""))
    if last_scan_at is None:
        return True, "runnable_drained_low_backlog" if no_ready_existing_work else "low_backlog", current_open, task_count
    elapsed = (datetime.now(timezone.utc) - last_scan_at).total_seconds()
    if elapsed >= cooldown_seconds:
        return True, "runnable_drained_low_backlog" if no_ready_existing_work else "low_backlog", current_open, task_count
    return False, "cooldown", current_open, task_count


def git_toplevel_for_path(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def path_status(repo: Path, relative: str) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def unmerged_worktree_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def commit_specific_path(repo: Path, relative: str, *, subject: str) -> dict[str, Any]:
    if not repo_relative_path_safe(relative):
        return {"committed": False, "reason": "unsafe_path", "repo": str(repo), "path": relative}
    unmerged = unmerged_worktree_paths(repo)
    if unmerged and relative not in unmerged:
        return {
            "committed": False,
            "reason": "repo_has_unrelated_unmerged_paths",
            "repo": str(repo),
            "path": relative,
            "unmerged_paths": sorted(unmerged),
        }
    status = path_status(repo, relative)
    if not status:
        return {"committed": False, "reason": "no_changes", "repo": str(repo), "path": relative}
    add = subprocess.run(["git", "add", "--", relative], cwd=repo, text=True, capture_output=True, check=False)
    if add.returncode != 0:
        return {
            "committed": False,
            "reason": "git_add_failed",
            "repo": str(repo),
            "path": relative,
            "returncode": add.returncode,
            "stdout": add.stdout[-4000:],
            "stderr": add.stderr[-4000:],
        }
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo, check=False)
    if staged.returncode == 0:
        return {"committed": False, "reason": "no_staged_changes", "repo": str(repo), "path": relative}
    commit = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Accelerator Backlog Refinery",
            "-c",
            "user.email=accelerator-backlog-refinery@example.invalid",
            "commit",
            "-m",
            subject,
            "--",
            relative,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {
            "committed": False,
            "reason": "git_commit_failed",
            "repo": str(repo),
            "path": relative,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }
    ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return {"committed": True, "repo": str(repo), "path": relative, "commit": ref.stdout.strip(), "status": status}


def parent_git_toplevel_for_repo(repo: Path) -> Path | None:
    parent = git_toplevel_for_path(repo.resolve().parent)
    if parent is None or parent.resolve() == repo.resolve():
        return None
    try:
        repo.resolve().relative_to(parent.resolve())
    except ValueError:
        return None
    return parent


def commit_parent_gitlink_updates(child_repo: Path, *, repo_root: Path, subject: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    current = child_repo.resolve()
    root = repo_root.resolve()
    while current != root:
        parent = parent_git_toplevel_for_repo(current)
        if parent is None:
            break
        relative = repo_relative_path(parent, current)
        if not relative:
            break
        results.append(commit_specific_path(parent, relative, subject=subject))
        current = parent.resolve()
    return results


def commit_generated_outputs(paths: Sequence[Path], *, repo_root: Path, subject: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        repo = git_toplevel_for_path(path.parent)
        if repo is None:
            results.append({"committed": False, "reason": "not_in_git_repo", "path": str(path)})
            continue
        relative = repo_relative_path(repo, path)
        if not relative:
            results.append({"committed": False, "reason": "path_outside_repo", "path": str(path), "repo": str(repo)})
            continue
        result = commit_specific_path(repo, relative, subject=subject)
        if result.get("committed"):
            parent_results = commit_parent_gitlink_updates(repo, repo_root=repo_root, subject=subject)
            if parent_results:
                result["parent_gitlink_commits"] = parent_results
        results.append(result)
    return results


def git_status_porcelain(repo: Path) -> list[str]:
    """Return short porcelain status lines, including untracked files."""

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def git_dir_for_repo(repo: Path) -> Path | None:
    """Return the resolved git metadata directory for a worktree."""

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    return git_dir.resolve()


def git_index_lock_path(repo: Path) -> Path | None:
    git_dir = git_dir_for_repo(repo)
    if git_dir is None:
        return None
    return git_dir / "index.lock"


def git_merge_head_path(repo: Path) -> Path | None:
    git_dir = git_dir_for_repo(repo)
    if git_dir is None:
        return None
    return git_dir / "MERGE_HEAD"


def generated_dirty_commit_blocker(repo: Path) -> dict[str, Any] | None:
    """Return a checkout state that should defer generated-output commits."""

    merge_head = git_merge_head_path(repo)
    if merge_head is not None and merge_head.exists():
        return {
            "repo": str(repo),
            "reason": "repo_merge_in_progress",
            "merge_head_path": str(merge_head),
        }

    try:
        from ipfs_accelerate_py.agent_supervisor.checkout_lock import (
            checkout_mutation_lock_path,
        )
    except ImportError:
        return None

    lock_path = checkout_mutation_lock_path(repo)
    if lock_path.exists():
        return {
            "repo": str(repo),
            "reason": "checkout_mutation_lock_exists",
            "lock_path": str(lock_path),
        }
    return None


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _cmdline_has_git_process(cmdline: str) -> bool:
    if not cmdline:
        return False
    for token in cmdline.replace("\x00", " ").split():
        name = Path(token).name
        if name == "git" or name.startswith("git-"):
            return True
    return False


def active_git_processes_for_repo(repo: Path, git_dir: Path) -> list[dict[str, Any]]:
    """Best-effort check for active git processes that could own a lock."""

    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    repo = repo.resolve()
    git_dir = git_dir.resolve()
    active: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw_cmdline.decode("utf-8", errors="replace").replace("\x00", " ").strip()
        if not _cmdline_has_git_process(cmdline):
            continue
        cwd_text = ""
        try:
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            cwd_text = str(cwd)
        except OSError:
            cwd = None
        mentions_repo = str(repo) in cmdline or str(git_dir) in cmdline
        cwd_matches = bool(cwd and (_path_inside(cwd, repo) or _path_inside(cwd, git_dir)))
        if mentions_repo or cwd_matches:
            active.append({"pid": pid, "cwd": cwd_text, "cmdline": cmdline[:500]})
    return active


def repair_stale_git_index_lock(
    repo: Path,
    *,
    stale_seconds: float = DEFAULT_STALE_GIT_LOCK_SECONDS,
) -> dict[str, Any]:
    """Remove an inactive stale ``index.lock`` for one git worktree.

    Git leaves ``index.lock`` behind when an add/commit process crashes. The
    supervisor can safely remove it only when the lock is old enough and there
    is no active git process associated with the same worktree/git directory.
    """

    repo = repo.resolve()
    lock_path = git_index_lock_path(repo)
    if lock_path is None:
        return {"attempted": False, "repo": str(repo), "reason": "not_git_repo"}
    if not lock_path.exists():
        return {"attempted": False, "repo": str(repo), "lock_path": str(lock_path), "reason": "no_lock"}
    try:
        stat = lock_path.stat()
    except OSError as exc:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_stat_failed",
            "error": str(exc),
        }
    age_seconds = max(0.0, time.time() - stat.st_mtime)
    git_dir = lock_path.parent
    active_processes = active_git_processes_for_repo(repo, git_dir)
    if active_processes:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "active_git_process",
            "age_seconds": age_seconds,
            "active_processes": active_processes[:10],
        }
    if age_seconds < float(stale_seconds):
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_not_stale",
            "age_seconds": age_seconds,
            "stale_seconds": stale_seconds,
        }
    try:
        lock_path.unlink()
    except OSError as exc:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_unlink_failed",
            "age_seconds": age_seconds,
            "error": str(exc),
        }
    return {
        "attempted": True,
        "repo": str(repo),
        "lock_path": str(lock_path),
        "removed": True,
        "reason": "stale_lock_removed",
        "age_seconds": age_seconds,
    }


def _resolve_existing_path_for_git_root(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _relative_filter_for_git_root(
    relative: str,
    *,
    repo_root: Path,
    git_root: Path,
) -> str:
    path_text = normalize_status_path(relative)
    if not path_text:
        return ""
    try:
        full_path = (repo_root / path_text).resolve()
        root = git_root.resolve()
    except OSError:
        return ""
    if full_path == root:
        return ""
    try:
        return full_path.relative_to(root).as_posix()
    except ValueError:
        return ""


def generated_status_filters_for_git_root(
    *,
    repo_root: Path,
    git_root: Path,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Convert repo-root-relative generated filters to one git root."""

    if git_root.resolve() == repo_root.resolve():
        return (
            [normalize_status_path(path) for path in generated_paths if normalize_status_path(path)],
            [normalize_status_path(path) for path in generated_prefixes if normalize_status_path(path)],
        )
    return (
        list(
            dict.fromkeys(
                rel
                for rel in (
                    _relative_filter_for_git_root(path, repo_root=repo_root, git_root=git_root)
                    for path in generated_paths
                )
                if rel
            )
        ),
        list(
            dict.fromkeys(
                rel
                for rel in (
                    _relative_filter_for_git_root(path, repo_root=repo_root, git_root=git_root)
                    for path in generated_prefixes
                )
                if rel
            )
        ),
    )


def _git_root_candidates_for_dirty_generated_outputs(
    *,
    repo_root: Path,
    generated_paths: Sequence[str],
    generated_prefixes: Sequence[str],
    candidate_git_roots: Sequence[Path | str],
) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        top = git_toplevel_for_path(_resolve_existing_path_for_git_root(candidate))
        if top is None:
            return
        try:
            top.resolve().relative_to(repo_root.resolve())
        except ValueError:
            return
        key = str(top.resolve())
        if key not in seen:
            seen.add(key)
            roots.append(top.resolve())

    add(repo_root)
    for candidate in candidate_git_roots:
        add(repo_root / candidate if not Path(candidate).is_absolute() else Path(candidate))
    for relative in [*generated_paths, *generated_prefixes]:
        path_text = normalize_status_path(relative)
        if path_text:
            add(repo_root / path_text)
    for submodule_root in _initialized_submodule_git_roots(repo_root):
        add(submodule_root)
    return sorted(roots, key=lambda path: len(path.resolve().parts), reverse=True)


def _initialized_submodule_git_roots(repo_root: Path) -> list[Path]:
    """Return initialized submodule worktree roots under ``repo_root``."""

    result = subprocess.run(
        ["git", "submodule", "foreach", "--quiet", "--recursive", "pwd"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    roots: list[Path] = []
    for line in result.stdout.splitlines():
        path_text = line.strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = repo_root / path
        roots.append(path)
    return roots


def _path_is_gitlink(repo: Path, relative: str) -> bool:
    if not repo_relative_path_safe(relative):
        return False
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", relative],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(line.startswith("160000 ") for line in result.stdout.splitlines())


def _clean_child_git_root(repo: Path, relative: str) -> str:
    child = repo / relative
    child_root = git_toplevel_for_path(child)
    if child_root is None:
        return ""
    if git_status_porcelain(child_root):
        return ""
    return str(child_root)


def _status_line_is_clean_gitlink_update(repo: Path, line: str) -> tuple[bool, str]:
    code = line[:2]
    relative = status_line_path(line)
    if not relative or "U" in code or "R" in code or "C" in code:
        return False, ""
    if code == "??" or not _path_is_gitlink(repo, relative):
        return False, ""
    child_root = _clean_child_git_root(repo, relative)
    return bool(child_root), child_root


def _commit_selected_dirty_paths(repo: Path, paths: Sequence[str], *, subject: str) -> dict[str, Any]:
    selected_paths = [path for path in dict.fromkeys(paths) if repo_relative_path_safe(path)]
    if not selected_paths:
        return {
            "committed": False,
            "reason": "no_safe_paths",
            "repo": str(repo),
            "selected_paths": [],
        }
    add = subprocess.run(
        ["git", "add", "--", *selected_paths],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if add.returncode != 0:
        return {
            "committed": False,
            "reason": "git_add_failed",
            "repo": str(repo),
            "selected_paths": selected_paths,
            "returncode": add.returncode,
            "stdout": add.stdout[-4000:],
            "stderr": add.stderr[-4000:],
        }
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *selected_paths],
        cwd=repo,
        check=False,
    )
    if staged.returncode == 0:
        return {
            "committed": False,
            "reason": "no_staged_changes",
            "repo": str(repo),
            "selected_paths": selected_paths,
        }
    commit = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Agent Supervisor",
            "-c",
            "user.email=agent-supervisor@example.invalid",
            "commit",
            "-m",
            subject,
            "--",
            *selected_paths,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {
            "committed": False,
            "reason": "git_commit_failed",
            "repo": str(repo),
            "selected_paths": selected_paths,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }
    ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return {
        "committed": True,
        "repo": str(repo),
        "selected_paths": selected_paths,
        "commit": ref.stdout.strip(),
        "stdout": commit.stdout[-4000:],
    }


def commit_generated_dirty_outputs(
    *,
    repo_root: Path,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
    candidate_git_roots: Sequence[Path | str] = (),
    subject: str = "Agent: commit generated supervisor outputs",
    include_clean_submodule_gitlinks: bool = True,
    max_paths: int = 200,
    stale_git_lock_seconds: float = DEFAULT_STALE_GIT_LOCK_SECONDS,
) -> dict[str, Any]:
    """Commit safe supervisor-generated dirt across nested git roots.

    The repair is deliberately conservative: it stages only paths matching the
    generated-output filters, plus clean submodule gitlink pointer updates when
    requested. Unknown dirty files are reported but left untouched.
    """

    repo_root = repo_root.resolve()
    roots = _git_root_candidates_for_dirty_generated_outputs(
        repo_root=repo_root,
        generated_paths=generated_paths,
        generated_prefixes=generated_prefixes,
        candidate_git_roots=candidate_git_roots,
    )
    remaining_budget = max(0, int(max_paths))
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    lock_repairs: list[dict[str, Any]] = []
    selected_path_count = 0
    for git_root in roots:
        lock_repair = repair_stale_git_index_lock(
            git_root,
            stale_seconds=stale_git_lock_seconds,
        )
        if lock_repair.get("attempted"):
            lock_repairs.append(lock_repair)
        if lock_repair.get("attempted") and not lock_repair.get("removed"):
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": str(lock_repair.get("reason") or "git_index_lock_blocked"),
                    "lock_repair": lock_repair,
                }
            )
            continue
        commit_blocker = generated_dirty_commit_blocker(git_root)
        if commit_blocker is not None:
            skipped.append(commit_blocker)
            continue
        status = git_status_porcelain(git_root)
        if not status:
            continue
        unmerged = sorted(unmerged_worktree_paths(git_root))
        if unmerged:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "repo_has_unmerged_paths",
                    "unmerged_paths": unmerged[:50],
                }
            )
            continue
        repo_generated_paths, repo_generated_prefixes = generated_status_filters_for_git_root(
            repo_root=repo_root,
            git_root=git_root,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        )
        selected: list[str] = []
        selected_reasons: dict[str, str] = {}
        for line in status:
            if remaining_budget <= 0:
                break
            code = line[:2]
            relative = status_line_path(line)
            if not relative or not repo_relative_path_safe(relative):
                continue
            if "U" in code or "R" in code or "C" in code:
                continue
            if path_is_generated_status_output(
                relative,
                generated_paths=repo_generated_paths,
                generated_prefixes=repo_generated_prefixes,
            ):
                selected.append(relative)
                selected_reasons[relative] = "generated_output"
                remaining_budget -= 1
                continue
            if include_clean_submodule_gitlinks:
                gitlink, child_root = _status_line_is_clean_gitlink_update(git_root, line)
                if gitlink:
                    selected.append(relative)
                    selected_reasons[relative] = f"clean_submodule_gitlink:{child_root}"
                    remaining_budget -= 1
        if not selected:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "no_safe_dirty_paths",
                    "status_short": status[:50],
                }
            )
            continue
        result = _commit_selected_dirty_paths(git_root, selected, subject=subject)
        result["selected_reasons"] = selected_reasons
        result["status_short_before"] = status[:50]
        selected_path_count += len(selected)
        results.append(result)

    final_status = git_status_porcelain(repo_root)
    return {
        "attempted": True,
        "repo_root": str(repo_root),
        "git_root_count": len(roots),
        "selected_path_count": selected_path_count,
        "committed_count": sum(1 for item in results if item.get("committed")),
        "results": results,
        "lock_repairs": lock_repairs,
        "skipped": skipped[:50],
        "remaining_status_short": final_status[:50],
        "remaining_status_count": len(final_status),
        "max_paths": max_paths,
    }


def repo_relative_path_safe(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\0" in relative:
        return False
    return ".." not in Path(relative).parts


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def discovery_output_path_for(
    repo_root: Path,
    discovery_dir: Path,
    *,
    default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    """Return a repo-relative discovery output path, or a stable fallback."""

    try:
        return discovery_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return default


def task_dependencies_if_present(
    todo_path: Path,
    *,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    dependency_ids: Sequence[str] = (),
) -> list[str]:
    """Return dependency ids that are already declared in a todo board."""

    if not dependency_ids or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    task_prefix = task_id_prefix(task_header_prefix_value)
    declared_task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    return [dependency_id for dependency_id in dependency_ids if dependency_id in declared_task_ids]


def codebase_scan_path_skipped(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative = path.as_posix()
    if any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in skip_prefixes):
        return True
    return any(part in CODEBASE_SCAN_SKIP_PARTS for part in path.parts)


def discover_git_worktrees(
    repo_root: Path,
    *,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add_if_worktree(candidate: Path) -> None:
        top = git_toplevel_for_path(candidate)
        if top is None:
            return
        resolved = top.resolve()
        if not path_is_under(resolved, repo_root):
            return
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    add_if_worktree(repo_root)
    for current, dirnames, _filenames in os.walk(repo_root):
        current_path = Path(current)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in CODEBASE_SCAN_SKIP_PARTS
            and not codebase_scan_path_skipped(current_path / dirname, repo_root=repo_root, skip_prefixes=skip_prefixes)
        ]
        if current_path != repo_root and (current_path / ".git").exists():
            add_if_worktree(current_path)
            dirnames[:] = []
    return roots


def tracked_files(repo: Path) -> list[Path]:
    if not repo.is_dir():
        return []
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=False)
    except (FileNotFoundError, OSError):
        logger.debug("Skipping vanished git root during codebase scan: %s", repo)
        return []
    if result.returncode != 0:
        return []
    files: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if not repo_relative_path_safe(relative):
            continue
        path = repo / relative
        if path.is_file():
            files.append(path)
    return files


def root_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_is_scan_candidate(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> bool:
    if codebase_scan_path_skipped(path, repo_root=repo_root, skip_prefixes=skip_prefixes):
        return False
    if "-codebase-scan-" in path.name or "retry-budget" in path.name:
        return False
    if path.name == "todo.md" or path.name.endswith(".todo.md"):
        return False
    if path.suffix.lower() not in CODEBASE_SCAN_SUFFIXES:
        return False
    try:
        if path.stat().st_size > CODEBASE_SCAN_MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return True


def scan_fingerprint(*, kind: str, root_relative_path: str, line_number: int, snippet: str) -> str:
    normalized = " ".join(snippet.strip().split())
    payload = f"{kind}\0{root_relative_path}\0{line_number}\0{normalized}"
    return sha1(payload.encode("utf-8")).hexdigest()


def scan_track_for_path(path: str) -> str:
    lowered = path.lower()
    if "/test/" in lowered or lowered.startswith("tests/") or "test_" in Path(lowered).name:
        return "quality"
    if "ui" in Path(lowered).parts or "frontend" in Path(lowered).parts:
        return "ui"
    if lowered.endswith((".md", ".rst")):
        return "docs"
    if lowered.endswith((".py", ".rs", ".sh")):
        return "runtime"
    return "ops"


def scan_validation_for_path(root_relative: str) -> str:
    quoted = shlex.quote(root_relative)
    suffix = Path(root_relative).suffix.lower()
    if suffix == ".py":
        return f"python3 -m py_compile {quoted}"
    if suffix == ".json":
        return f"python3 -m json.tool {quoted} >/dev/null"
    if suffix in {".yaml", ".yml"}:
        source = 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        return f"python3 -c {shlex.quote(source)} {quoted}"
    return f"test -f {quoted}"


def annotation_scan_text(line: str) -> str:
    """Remove path-like tokens that should not count as TODO annotations."""

    return re.sub(r"(?i)[A-Za-z0-9_./-]*\.todo\.md\b", "", line)


def _position_in_simple_quoted_string(text: str, index: int) -> bool:
    quote = ""
    escaped = False
    for char in text[:index]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
    return bool(quote)


def annotation_followup_marker(line: str) -> str:
    """Return the TODO-like marker when the line looks like a real annotation."""

    text = annotation_scan_text(line)
    for match in ANNOTATION_FOLLOWUP_RE.finditer(text):
        marker = str(match.group("line_marker") or match.group("comment_marker") or "").lower()
        start = match.start("line_marker") if match.group("line_marker") else match.start("comment_prefix")
        if _position_in_simple_quoted_string(text, start):
            continue
        return marker
    return ""


def scan_findings_in_file(path: Path, *, repo_root: Path) -> list[CodebaseFinding]:
    root_relative = root_relative_path(repo_root, path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    findings: list[CodebaseFinding] = []
    in_fenced_block = False
    scan_fences = path.suffix.lower() in {".md", ".rst"}
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if scan_fences and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block or not stripped:
            continue
        lowered = stripped.lower()
        kind = ""
        priority = "P2"
        summary = ""
        annotation_marker = annotation_followup_marker(stripped)
        if annotation_marker:
            kind = "annotated_followup"
            priority = "P2" if annotation_marker in {"fixme", "hack", "xxx"} else "P3"
            summary = f"Resolve code annotation in {root_relative}:{index}"
        elif re.search(r"\bexcept\s*:\s*$", stripped) or re.search(r"\bexcept\s+Exception\b", stripped):
            window = "\n".join(lines[index : min(len(lines), index + 3)]).lower()
            if "pass" in window or "return none" in window:
                kind = "swallowed_exception"
                priority = "P1"
                summary = f"Review swallowed exception path in {root_relative}:{index}"
        elif "assert false" in lowered or "raise notimplementederror" in lowered:
            kind = "placeholder_runtime_path"
            priority = "P1"
            summary = f"Replace placeholder runtime path in {root_relative}:{index}"
        if not kind:
            continue
        fingerprint = scan_fingerprint(
            kind=kind,
            root_relative_path=root_relative,
            line_number=index,
            snippet=stripped,
        )
        findings.append(
            CodebaseFinding(
                fingerprint=fingerprint,
                kind=kind,
                priority=priority,
                track=scan_track_for_path(root_relative),
                root_relative_path=root_relative,
                line_number=index,
                snippet=stripped[:240],
                summary=summary,
                validation=scan_validation_for_path(root_relative),
            )
        )
    return findings


def scan_codebase_findings(
    repo_root: Path,
    *,
    max_findings: int,
    seen_fingerprints: Iterable[str] = (),
    exhaustive: bool = False,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> list[CodebaseFinding]:
    findings: list[CodebaseFinding] = []
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    for git_root in discover_git_worktrees(repo_root, skip_prefixes=skip_prefixes):
        for path in tracked_files(git_root):
            if not file_is_scan_candidate(path, repo_root=repo_root, skip_prefixes=skip_prefixes):
                continue
            for finding in scan_findings_in_file(path, repo_root=repo_root):
                if finding.fingerprint in seen:
                    continue
                if len(findings) < max_findings:
                    findings.append(finding)
                    seen.add(finding.fingerprint)
                    continue
                if not exhaustive:
                    return findings
    return findings


def write_codebase_scan_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    finding: CodebaseFinding,
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-codebase-scan-{finding.fingerprint[:12]}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# {task_id} Codebase Scan Finding

Date: {date}
Fingerprint: {finding.fingerprint}
Kind: {finding.kind}
Source: {finding.root_relative_path}:{finding.line_number}
Priority: {finding.priority}
Track: {finding.track}

## Evidence

```text
{finding.snippet}
```

## Suggested Handling

Review the finding in context, decide whether it represents a bug, missing test,
maintenance risk, or false positive, and land a small fix with validation. If the
finding is a false positive, document why in the changed code or discovery notes
so the supervisor does not keep re-adding the same work.
"""
    path.write_text(content, encoding="utf-8")
    return path


def codebase_scan_task_block(
    *,
    task_id: str,
    finding: CodebaseFinding,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = [discovery_output_path, finding.root_relative_path]
    return f"""## {task_id} {finding.summary}

- Status: todo
- Completion: manual
- Priority: {finding.priority}
- Track: {finding.track}
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {finding.validation}
- Acceptance: Codebase scan filed this finding from {finding.root_relative_path}:{finding.line_number}. Use evidence in {discovery_path}, fix the bug or improvement, add or update focused validation when appropriate, and keep the supervisor-fed backlog parseable.
"""


def duplicate_task_id_records(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    """Return todo-board records for task ids that appear more than once."""

    task_groups: dict[str, list[Any]] = {}
    for task in tasks:
        task_id = str(getattr(task, "task_id", "") or "").strip()
        if not task_id:
            continue
        task_groups.setdefault(task_id, []).append(task)

    records: list[dict[str, Any]] = []
    for task_id, duplicates in sorted(task_groups.items()):
        if len(duplicates) < 2:
            continue
        titles = [str(getattr(task, "title", "") or "") for task in duplicates]
        source_lines: list[int] = []
        for task in duplicates:
            try:
                source_line = int(getattr(task, "source_line", 0) or 0)
            except (TypeError, ValueError):
                continue
            if source_line > 0:
                source_lines.append(source_line)
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "duplicate_task_id",
                    "task_id": task_id,
                    "titles": sorted(title for title in titles if title),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_task_id": task_id,
                "source_title": "Duplicate task id",
                "missing_dependencies": [],
                "self_references": [],
                "dependency_cycle": [],
                "duplicate_task_id": task_id,
                "duplicate_task_lines": source_lines,
                "duplicate_task_titles": titles,
                "fingerprint": fingerprint,
            }
        )
    return records


def dependency_guardrail_records(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    """Return todo-board records that can keep tasks from becoming ready."""

    task_ids = {str(task.task_id) for task in tasks}
    open_task_ids = {
        str(task.task_id)
        for task in tasks
        if str(task.status).lower() not in {"completed", "blocked"}
    }
    dependency_graph = {
        str(task.task_id): [
            str(dep)
            for dep in task.depends_on
            if str(dep).strip() and str(dep) in open_task_ids
        ]
        for task in tasks
        if str(task.task_id) in open_task_ids
    }
    records: list[dict[str, Any]] = duplicate_task_id_records(tasks)

    def reachable_cycle(start: str) -> list[str]:
        path: list[str] = []

        def visit(node: str) -> list[str]:
            if node in path:
                index = path.index(node)
                return [*path[index:], node]
            path.append(node)
            for dependency in dependency_graph.get(node, []):
                cycle = visit(dependency)
                if cycle:
                    return cycle
            path.pop()
            return []

        return visit(start)

    for task in tasks:
        if str(task.status).lower() in {"completed", "blocked"}:
            continue
        dependencies = [str(dep) for dep in task.depends_on if str(dep).strip()]
        missing = sorted(dep for dep in dependencies if dep not in task_ids)
        self_references = sorted(dep for dep in dependencies if dep == task.task_id)
        dependency_cycle = reachable_cycle(task.task_id)
        if not missing and not self_references and not dependency_cycle:
            continue
        fingerprint = sha1(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "missing_dependencies": missing,
                    "self_references": self_references,
                    "dependency_cycle": dependency_cycle,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_task_id": task.task_id,
                "source_title": task.title,
                "missing_dependencies": missing,
                "self_references": self_references,
                "dependency_cycle": dependency_cycle,
                "fingerprint": fingerprint,
            }
        )
    return records


def write_dependency_guardrail_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    record: Mapping[str, Any],
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-dependency-guardrail.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = ", ".join(str(item) for item in record.get("missing_dependencies", []) or []) or "none"
    self_references = ", ".join(str(item) for item in record.get("self_references", []) or []) or "none"
    dependency_cycle = " -> ".join(str(item) for item in record.get("dependency_cycle", []) or []) or "none"
    duplicate_task_id = str(record.get("duplicate_task_id") or "") or "none"
    duplicate_lines = ", ".join(str(item) for item in record.get("duplicate_task_lines", []) or []) or "none"
    duplicate_titles = "\n".join(
        f"- {title}" for title in record.get("duplicate_task_titles", []) or [] if str(title).strip()
    )
    duplicate_titles = duplicate_titles or "- none"
    content = f"""# Dependency Guardrail: {record.get("source_task_id")}

Created: {utc_now()}
Fingerprint: {record.get("fingerprint")}
Source task: {record.get("source_task_id")} {record.get("source_title") or ""}
Missing dependencies: {missing}
Self-referential dependencies: {self_references}
Dependency cycle: {dependency_cycle}
Duplicate task id: {duplicate_task_id}
Duplicate source lines: {duplicate_lines}

## Duplicate Task Titles

{duplicate_titles}

## Why This Blocks Progress

The implementation daemon only selects tasks whose dependencies are completed.
When an open task depends on a task id that is not present on the board, or on
itself, or participates in a dependency cycle, the task can remain waiting
indefinitely while the supervisor reports no ready work. Duplicate task ids are
also ambiguous because status maps, dependency resolution, and guardrail
releases all key by task id.

## Suggested Repair

Inspect the source task metadata and either add the missing prerequisite task,
remove the stale dependency, break the dependency cycle, rename duplicate task
ids so each task is unique, or replace stale references with the correct existing
task id. Keep the todo board parseable after the repair.
"""
    path.write_text(content, encoding="utf-8")
    return path


def dependency_guardrail_task_block(
    *,
    task_id: str,
    source_task_id: str,
    discovery_path: Path,
    todo_output_path: str,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    return f"""## {task_id} Resolve dependency guardrail for {source_task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: {discovery_output_path}, {todo_output_path}
- Validation: test -f {shlex.quote(str(discovery_path))}
- Acceptance: Dependency guardrail filed this because {source_task_id} has missing, self-referential, cyclic, or duplicate task-id metadata. Use the evidence in {discovery_path} to repair the todo board metadata or add the missing prerequisite task, then verify the original task can become ready once its real dependencies complete.
"""


def reconciliation_guardrail_records(
    *,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    generated_status_paths: Sequence[str] = (),
    generated_status_prefixes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return grouped cleanup/reconciliation blockers that need deliberate repair."""

    records: list[dict[str, Any]] = []
    reconciliation = dict(reconciliation_result or {})
    cleanup = dict(cleanup_result or {})

    if reconciliation.get("attempted") and reconciliation.get("main_checkout_dirty"):
        candidate_count = int(reconciliation.get("candidate_count") or 0)
        if candidate_count > 0:
            status_short = [str(item) for item in reconciliation.get("main_status_short", []) if str(item).strip()]
            main_dirty_evidence = (
                dict(reconciliation.get("main_dirty_evidence") or {})
                if isinstance(reconciliation.get("main_dirty_evidence"), Mapping)
                else {}
            )
            status_short, main_dirty_evidence = filter_generated_main_checkout_evidence(
                status_short=status_short,
                evidence=main_dirty_evidence,
                generated_paths=generated_status_paths,
                generated_prefixes=generated_status_prefixes,
            )
            if status_short:
                candidates = [
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "target_ref": str(item.get("target_ref") or reconciliation.get("target_ref") or ""),
                    }
                    for item in reconciliation.get("candidates", [])
                    if isinstance(item, Mapping)
                ]
                fingerprint = sha1(
                    json.dumps(
                        {
                            "kind": "main_checkout_dirty",
                            "status_short": status_short,
                            "candidate_branches": [item["branch"] for item in candidates],
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "kind": "main_checkout_dirty",
                        "priority": "P1",
                        "track": "ops",
                        "summary": f"Resolve dirty main checkout blocking {candidate_count} worktree merges",
                        "fingerprint": fingerprint,
                        "candidate_count": candidate_count,
                        "status_short": status_short,
                        "main_dirty_evidence": main_dirty_evidence,
                        "samples": candidates[:20],
                        "reason": "main_checkout_dirty",
                        "dedupe_key": "reconciliation_guardrail:main_checkout_dirty",
                    }
                )

    preflight_samples: list[dict[str, Any]] = []
    conflict_path_counts: dict[str, int] = {}
    for item in reconciliation.get("processed", []) or []:
        if not isinstance(item, Mapping):
            continue
        preflight_result = item.get("preflight_result") or {}
        if not isinstance(preflight_result, Mapping):
            continue
        if preflight_result.get("mergeable") is not False:
            continue
        conflict_paths = [
            str(path).strip()
            for path in preflight_result.get("conflict_paths", []) or []
            if str(path).strip()
        ]
        for path in conflict_paths:
            conflict_path_counts[path] = conflict_path_counts.get(path, 0) + 1
        preflight_samples.append(
            {
                "branch": str(item.get("branch") or preflight_result.get("branch") or ""),
                "path": str(item.get("path") or ""),
                "target_ref": str(item.get("target_ref") or preflight_result.get("target_ref") or ""),
                "conflict_paths": conflict_paths[:20],
                "reason": str(preflight_result.get("reason") or "preflight_merge_conflict"),
            }
        )
    if preflight_samples:
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "preflight_merge_conflict",
                    "branches": [item["branch"] for item in preflight_samples],
                    "conflict_path_counts": conflict_path_counts,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "kind": "preflight_merge_conflict",
                "priority": "P1",
                "track": "ops",
                "summary": (
                    f"Resolve {len(preflight_samples)} preflight-conflicting "
                    "backlogged worktree merges"
                ),
                "fingerprint": fingerprint,
                "candidate_count": len(preflight_samples),
                "status_short": [],
                "samples": preflight_samples[:20],
                "reason": "preflight_merge_conflict",
                "conflict_path_counts": conflict_path_counts,
                "dedupe_key": "reconciliation_guardrail:preflight_merge_conflict",
            }
        )

    dirty_groups: dict[str, dict[str, Any]] = {}
    grouped_payload = cleanup.get("dirty_worktree_groups")
    if isinstance(grouped_payload, Mapping) and grouped_payload:
        for dirty_reason, payload in grouped_payload.items():
            if not isinstance(payload, Mapping):
                continue
            dirty_groups[str(dirty_reason)] = {
                "count": int(payload.get("count") or 0),
                "samples": [dict(item) for item in payload.get("samples", []) if isinstance(item, Mapping)],
            }
    else:
        for item in cleanup.get("skipped", []):
            if not isinstance(item, Mapping) or str(item.get("reason") or "") != "dirty_worktree":
                continue
            dirty_redundancy = item.get("dirty_redundancy") or {}
            dirty_reason = (
                str(dirty_redundancy.get("reason") or "dirty_worktree")
                if isinstance(dirty_redundancy, Mapping)
                else "dirty_worktree"
            )
            group = dirty_groups.setdefault(dirty_reason, {"count": 0, "samples": []})
            group["count"] += 1
            if len(group["samples"]) < 20:
                group["samples"].append(
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "status_short": [str(line) for line in item.get("status_short", []) if str(line).strip()],
                        "dirty_reason": dirty_reason,
                        "dirty_evidence": dict(item.get("dirty_evidence") or {}),
                    }
                )

    for dirty_reason, group in sorted(dirty_groups.items()):
        samples = list(group.get("samples") or [])
        count = int(group.get("count") or len(samples))
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "dirty_backlogged_worktree",
                    "dirty_reason": dirty_reason,
                    "branches": [item["branch"] for item in samples],
                    "paths": [item["path"] for item in samples],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "kind": "dirty_backlogged_worktree",
                "priority": "P1" if dirty_reason == "unsupported_status" else "P2",
                "track": "ops",
                "summary": f"Resolve {count} dirty backlogged worktrees blocked by {dirty_reason}",
                "fingerprint": fingerprint,
                "candidate_count": count,
                "status_short": [],
                "samples": samples[:20],
                "reason": dirty_reason,
                "dedupe_key": f"reconciliation_guardrail:dirty_backlogged_worktree:{dirty_reason}",
            }
        )

    return records


def status_line_path(line: str) -> str:
    path_text = line[3:].strip() if len(line) > 3 else line.strip()
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[-1].strip()
    return path_text.rstrip("/")


def status_line_category(line: str) -> str:
    code = line[:2]
    if code == "??":
        return "untracked"
    if "U" in code:
        return "unmerged"
    if "D" in code:
        return "deleted"
    if "R" in code:
        return "renamed"
    if "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    if code.strip():
        return "other_dirty"
    return "clean"


def normalize_status_path(path: str) -> str:
    path_text = str(path).strip()
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[-1].strip()
    return path_text.rstrip("/")


def name_status_path(line: str) -> str:
    parts = str(line).split("\t")
    if len(parts) > 1:
        return normalize_status_path(parts[-1])
    return normalize_status_path(str(line).split(maxsplit=1)[-1] if str(line).split() else "")


def path_is_generated_status_output(
    path: str,
    *,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> bool:
    path_text = normalize_status_path(path)
    if not path_text:
        return False
    exact = {normalize_status_path(item) for item in generated_paths if normalize_status_path(item)}
    if path_text in exact:
        return True
    for prefix in generated_prefixes:
        prefix_text = normalize_status_path(str(prefix))
        if not prefix_text:
            continue
        if path_text == prefix_text or path_text.startswith(prefix_text + "/"):
            return True
    return False


def filter_generated_main_checkout_evidence(
    *,
    status_short: Sequence[str],
    evidence: Mapping[str, Any],
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Remove supervisor-generated todo/discovery output paths from dirty-main evidence."""

    filtered_status: list[str] = []
    filtered_paths: list[str] = []
    removed_paths: list[str] = []
    for line in status_short:
        line_text = str(line)
        path = status_line_path(line_text)
        if path_is_generated_status_output(
            path,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        ):
            if path and path not in removed_paths:
                removed_paths.append(path)
            continue
        filtered_status.append(line_text)
        if path and path not in filtered_paths:
            filtered_paths.append(path)

    filtered_evidence: dict[str, Any] = dict(evidence or {})
    filtered_evidence["status_short"] = filtered_status[:50]
    filtered_evidence["status_paths"] = filtered_paths[:50]
    path_categories: dict[str, int] = {}
    for line in filtered_status:
        category = status_line_category(line)
        path_categories[category] = path_categories.get(category, 0) + 1
    filtered_evidence["path_categories"] = path_categories
    for key in ("untracked_paths",):
        values = []
        for item in filtered_evidence.get(key, []) or []:
            path = normalize_status_path(str(item))
            if path and not path_is_generated_status_output(
                path,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                values.append(path)
            elif path and path not in removed_paths:
                removed_paths.append(path)
        if values:
            filtered_evidence[key] = values[:50]
        else:
            filtered_evidence.pop(key, None)
    for key in ("name_status", "staged_name_status"):
        lines = []
        for line in str(filtered_evidence.get(key) or "").splitlines():
            path = name_status_path(line)
            if path and path_is_generated_status_output(
                path,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                if path not in removed_paths:
                    removed_paths.append(path)
                continue
            if line.strip():
                lines.append(line)
        if lines:
            filtered_evidence[key] = "\n".join(lines)
        else:
            filtered_evidence.pop(key, None)
    if removed_paths:
        filtered_evidence["filtered_generated_status_paths"] = removed_paths[:50]
        filtered_evidence.pop("diff_stat", None)
        filtered_evidence.pop("submodule_summary", None)
    return filtered_status, filtered_evidence


def relative_status_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def generated_status_filter_path(path: Path | str, *, repo_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return relative_status_path(candidate, repo_root=repo_root)
    return normalize_status_path(str(path))


def generated_guardrail_status_filters(
    *,
    todo_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    additional_generated_paths: Sequence[Path | str] = (),
    additional_generated_prefixes: Sequence[Path | str] = (),
) -> tuple[list[str], list[str]]:
    generated_paths: list[str] = []
    generated_prefixes: list[str] = []
    todo_relative = relative_status_path(todo_path, repo_root=repo_root)
    if todo_relative:
        generated_paths.append(todo_relative)
    discovery_relative = relative_status_path(discovery_dir, repo_root=repo_root)
    if discovery_relative:
        generated_prefixes.append(discovery_relative)
    for path in additional_generated_paths:
        relative = generated_status_filter_path(path, repo_root=repo_root)
        if relative:
            generated_paths.append(relative)
    for prefix in additional_generated_prefixes:
        relative = generated_status_filter_path(prefix, repo_root=repo_root)
        if relative:
            generated_prefixes.append(relative)

    parts = Path(todo_relative).parts
    for end in range(1, len(parts)):
        ancestor = Path(*parts[:end])
        if ancestor.as_posix() in {".", ""}:
            continue
        if (repo_root / ancestor / ".git").exists():
            generated_paths.append(ancestor.as_posix())
    return list(dict.fromkeys(generated_paths)), list(dict.fromkeys(generated_prefixes))


def reconciliation_guardrail_plan(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a bounded reconciliation plan for a cleanup blocker record."""

    kind = str(record.get("kind") or "")
    reason = str(record.get("reason") or "")
    samples = [dict(item) for item in record.get("samples", []) or [] if isinstance(item, Mapping)]
    main_dirty_evidence = (
        dict(record.get("main_dirty_evidence") or {})
        if isinstance(record.get("main_dirty_evidence"), Mapping)
        else {}
    )
    sample_status_paths: list[str] = []
    for line in main_dirty_evidence.get("status_short", []) or []:
        path = status_line_path(str(line))
        if path and path not in sample_status_paths:
            sample_status_paths.append(path)
    for path in main_dirty_evidence.get("status_paths", []) or []:
        path_text = str(path).strip()
        if path_text and path_text not in sample_status_paths:
            sample_status_paths.append(path_text)
    for sample in samples:
        for path in sample.get("conflict_paths", []) or []:
            path_text = str(path).strip()
            if path_text and path_text not in sample_status_paths:
                sample_status_paths.append(path_text)
        for line in sample.get("status_short", []) or []:
            path = status_line_path(str(line))
            if path and path not in sample_status_paths:
                sample_status_paths.append(path)
        evidence = sample.get("dirty_evidence") or {}
        if isinstance(evidence, Mapping):
            for line in str(evidence.get("name_status") or "").splitlines():
                parts = line.split("\t")
                path = parts[-1].strip() if parts else ""
                if path and path not in sample_status_paths:
                    sample_status_paths.append(path)
            for path in evidence.get("untracked_paths", []) or []:
                path_text = str(path).strip()
                if path_text and path_text not in sample_status_paths:
                    sample_status_paths.append(path_text)

    conflict_path_counts: dict[str, int] = {}
    top_conflict_paths: list[str] = []
    safety_constraints = [
        "Do not discard dirty or untracked content unless it is proven redundant with the target ref.",
        "Prefer commits, merges, or explicit follow-up tasks over destructive cleanup.",
        "Keep todo, objective, discovery, and strategy files parseable after reconciliation.",
    ]
    success_signals = [
        "candidate_count_decreases",
        "dirty_worktree_group_count_decreases",
        "main_checkout_dirty_becomes_false",
        "cleanup_or_reconciliation_pass_processes_candidates",
    ]

    if kind == "preflight_merge_conflict":
        conflict_path_counts = {
            str(path): int(count)
            for path, count in (
                record.get("conflict_path_counts") or {}
                if isinstance(record.get("conflict_path_counts"), Mapping)
                else {}
            ).items()
        }
        top_conflict_paths = [
            path
            for path, _count in sorted(
                conflict_path_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        actions = [
            {
                "action": "bundle_preflight_conflicts_by_path",
                "scope": "backlogged_worktrees",
                "automation": "group blocked branches by shared conflict paths before resolving individual branches",
            },
            {
                "action": "resolve_markdown_and_discovery_conflicts_deterministically",
                "scope": "append_only_docs",
                "automation": "use deterministic append-only markdown/objective/todo merge repair where conflict paths are documentation or discovery files",
            },
            {
                "action": "resolve_code_or_submodule_conflicts_in_isolated_worktree",
                "scope": "code_and_gitlinks",
                "automation": "stage conflicts in a temporary reconciliation worktree or invoke the configured LLM resolver before mutating main",
            },
            {
                "action": "rerun_worktree_reconciliation",
                "scope": "backlogged_worktrees",
                "automation": "rerun reconcile_backlogged_worktrees and confirm preflight_blocked_count decreases",
            },
        ]
        safety_constraints = [
            "Do not run conflict-producing merges directly in main without a preflight or isolated resolver plan.",
            "Preserve submodule gitlink intent explicitly; never pick a gitlink side without recording why.",
            "Keep todo, objective, discovery, and strategy files parseable after reconciliation.",
        ]
        success_signals = [
            "preflight_blocked_count_decreases",
            "conflict_path_count_decreases",
            "reconciled_count_increases",
            "main_checkout_dirty_becomes_false",
        ]
    elif kind == "main_checkout_dirty":
        actions = [
            {
                "action": "classify_main_checkout_changes",
                "scope": "repo_root",
                "automation": "inspect git status, diff stats, submodule status, and generated artifacts before merges",
            },
            {
                "action": "preserve_or_split_main_checkout_work",
                "scope": "repo_root",
                "automation": "commit intentional changes or convert unresolved changes into follow-up tasks; never discard unknown work",
            },
            {
                "action": "rerun_worktree_reconciliation",
                "scope": "backlogged_worktrees",
                "automation": "rerun reconcile_backlogged_worktrees once the main checkout is clean enough to mutate",
            },
        ]
    else:
        actions = [
            {
                "action": "classify_dirty_worktree_group",
                "scope": "sampled_worktrees",
                "automation": "inspect sampled dirty statuses and compare against the target ref",
            },
            {
                "action": "preserve_or_merge_backlogged_work",
                "scope": "dirty_worktrees",
                "automation": "merge valuable branch work, commit preserved changes, or file follow-up tasks for unresolved work",
            },
            {
                "action": "rerun_cleanup_pass",
                "scope": "worktree_root",
                "automation": "rerun cleanup_backlogged_worktrees after preserving or merging dirty worktree content",
            },
        ]
        if reason == "content_not_in_target":
            actions.insert(
                1,
                {
                    "action": "compare_dirty_content_to_target",
                    "scope": "dirty_worktrees",
                    "automation": "separate real unmerged content from generated duplicates before deleting worktrees",
                },
            )
        elif reason == "unsupported_status":
            actions.insert(
                1,
                {
                    "action": "resolve_unsupported_statuses",
                    "scope": "dirty_worktrees",
                    "automation": "handle deletes, renames, unmerged paths, or unusual index states with an explicit resolver pass",
                },
            )

    return {
        "kind": kind,
        "reason": reason,
        "dedupe_key": str(record.get("dedupe_key") or ""),
        "fingerprint": str(record.get("fingerprint") or ""),
        "candidate_count": int(record.get("candidate_count") or 0),
        "sample_count": len(samples),
        "sample_branches": [str(item.get("branch") or "") for item in samples[:20] if str(item.get("branch") or "")],
        "sample_worktrees": [str(item.get("path") or "") for item in samples[:20] if str(item.get("path") or "")],
        "sample_status_paths": sample_status_paths[:40],
        "conflict_path_counts": conflict_path_counts,
        "top_conflict_paths": top_conflict_paths[:20],
        "main_dirty_evidence": main_dirty_evidence,
        "actions": actions,
        "safety_constraints": safety_constraints,
        "success_signals": success_signals,
    }


def reconciliation_guardrail_plan_markdown(record: Mapping[str, Any]) -> str:
    plan = reconciliation_guardrail_plan(record)
    action_lines = [
        f"- `{item['action']}`: {item['automation']}"
        for item in plan.get("actions", [])
        if isinstance(item, Mapping)
    ]
    constraint_lines = [f"- {item}" for item in plan.get("safety_constraints", [])]
    signal_lines = [f"- `{item}`" for item in plan.get("success_signals", [])]
    manifest = json.dumps(plan, indent=2, sort_keys=True)
    return f"""## Reconciliation Plan

Work surface: `{plan["candidate_count"]}` candidates, `{plan["sample_count"]}` sampled records.

### Suggested Actions

{chr(10).join(action_lines) or "- none"}

### Safety Constraints

{chr(10).join(constraint_lines) or "- none"}

### Success Signals

{chr(10).join(signal_lines) or "- none"}

## Machine Readable Manifest

```json
{manifest}
```
"""


def reconciliation_evidence_markdown(evidence: Mapping[str, Any] | None) -> str:
    if not isinstance(evidence, Mapping) or not evidence:
        return "- none"
    lines: list[str] = []
    path_categories = evidence.get("path_categories") or {}
    if isinstance(path_categories, Mapping) and path_categories:
        category_text = ", ".join(
            f"{key}={value}" for key, value in sorted(path_categories.items())
        )
        lines.append(f"- Path categories: `{category_text}`")
    status_paths = [str(item) for item in evidence.get("status_paths", []) if str(item).strip()]
    if status_paths:
        lines.append("- Status paths:")
        lines.extend(f"  - `{item}`" for item in status_paths[:20])
    for key, label in (
        ("name_status", "Name status"),
        ("staged_name_status", "Staged name status"),
        ("diff_stat", "Diff stat"),
        ("submodule_summary", "Submodule summary"),
    ):
        value = str(evidence.get(key) or "").strip()
        if not value:
            continue
        lines.append(f"- {label}:")
        lines.extend(f"  - `{line}`" for line in value.splitlines()[:20])
    untracked_paths = [str(item) for item in evidence.get("untracked_paths", []) if str(item).strip()]
    if untracked_paths:
        lines.append("- Untracked paths:")
        lines.extend(f"  - `{item}`" for item in untracked_paths[:20])
    return "\n".join(lines) or "- none"


def write_reconciliation_guardrail_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    record: Mapping[str, Any],
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    fingerprint = str(record.get("fingerprint") or "")
    path = discovery_dir / f"{date}-{task_id.lower()}-reconciliation-{fingerprint[:12]}.md"
    write_reconciliation_guardrail_discovery_path(path=path, task_id=task_id, record=record, date=date)
    return path


def preserved_reconciliation_discovery_sections(existing_text: str) -> list[str]:
    """Return manual resolution sections to carry across guardrail refreshes."""

    preserved: list[str] = []
    for match in re.finditer(r"^##\s+([^\n]+)\n.*?(?=^##\s+|\Z)", existing_text, flags=re.MULTILINE | re.DOTALL):
        title = " ".join(match.group(1).strip().lower().split())
        if title == "resolution" or title.startswith("resolution "):
            section = match.group(0).strip()
            if section:
                preserved.append(section)
    return preserved


def write_reconciliation_guardrail_discovery_path(
    *,
    path: Path,
    task_id: str,
    record: Mapping[str, Any],
    date: str | None = None,
) -> Path:
    date = date or datetime.now(timezone.utc).date().isoformat()
    fingerprint = str(record.get("fingerprint") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_text = path.read_text(encoding="utf-8")
    except OSError:
        existing_text = ""
    preserved_sections = preserved_reconciliation_discovery_sections(existing_text)
    status_lines = "\n".join(f"- `{line}`" for line in record.get("status_short", []) or []) or "- none"
    main_checkout_evidence = reconciliation_evidence_markdown(
        record.get("main_dirty_evidence")
        if isinstance(record.get("main_dirty_evidence"), Mapping)
        else None
    )
    sample_lines = []
    for sample in record.get("samples", []) or []:
        if not isinstance(sample, Mapping):
            continue
        branch = str(sample.get("branch") or "unknown-branch")
        path_text = str(sample.get("path") or "unknown-path")
        status = "; ".join(str(line) for line in sample.get("status_short", []) or [])
        suffix = f" status: `{status}`" if status else ""
        sample_lines.append(f"- `{branch}` at `{path_text}`{suffix}")
        conflict_paths = [str(path).strip() for path in sample.get("conflict_paths", []) or [] if str(path).strip()]
        if conflict_paths:
            sample_lines.append("  - Conflict paths:")
            sample_lines.extend(f"    - `{path}`" for path in conflict_paths[:12])
        evidence = sample.get("dirty_evidence") or {}
        if isinstance(evidence, Mapping):
            diff_stat = str(evidence.get("diff_stat") or "").strip()
            name_status = str(evidence.get("name_status") or "").strip()
            untracked_paths = [str(item) for item in evidence.get("untracked_paths", []) if str(item).strip()]
            if name_status:
                sample_lines.append("  - Name status:")
                sample_lines.extend(f"    - `{line}`" for line in name_status.splitlines()[:12])
            if diff_stat:
                sample_lines.append("  - Diff stat:")
                sample_lines.extend(f"    - `{line}`" for line in diff_stat.splitlines()[:12])
            if untracked_paths:
                sample_lines.append("  - Untracked paths:")
                sample_lines.extend(f"    - `{path}`" for path in untracked_paths[:12])
    samples = "\n".join(sample_lines) or "- none"
    plan_markdown = reconciliation_guardrail_plan_markdown(record)
    content = f"""# {task_id} Reconciliation Guardrail

Date: {date}
Fingerprint: {fingerprint}
Kind: {record.get("kind")}
Reason: {record.get("reason")}
Candidate count: {record.get("candidate_count")}
Priority: {record.get("priority")}
Track: {record.get("track")}

## Main Checkout Status

{status_lines}

## Main Checkout Evidence

{main_checkout_evidence}

## Sample Branches Or Worktrees

{samples}

## Why This Blocks Progress

The implementation supervisor can only merge clean inactive implementation
worktrees when the main checkout is safe to mutate. Dirty main checkouts and
dirty backlogged worktrees are preserved until a deliberate reconciliation task
decides whether to commit, merge, discard generated duplicates, or split
unresolved work into follow-up tasks.

## Suggested Repair

Inspect the dirty paths and sampled worktrees, resolve any real work into
reviewable commits or follow-up tasks, rerun the supervisor reconciliation pass,
and verify that either the candidate merge count decreases or the dirty
worktree cleanup skip count decreases.

{plan_markdown.rstrip()}
"""
    if preserved_sections:
        content = content.rstrip() + "\n\n" + "\n\n".join(preserved_sections).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def reconciliation_guardrail_task_block(
    *,
    task_id: str,
    record: Mapping[str, Any],
    discovery_path: Path,
    todo_output_path: str,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = [discovery_output_path, todo_output_path]
    return f"""## {task_id} {record.get("summary")}

- Status: todo
- Completion: manual
- Priority: {record.get("priority") or "P1"}
- Track: {record.get("track") or "ops"}
- Fingerprint: {record.get("fingerprint") or ""}
- Dedupe key: {record.get("dedupe_key") or ""}
- Depends on:
- Outputs: {", ".join(outputs)}
- Validation: test -f {shlex.quote(str(discovery_path))}
- Acceptance: Reconciliation guardrail filed this because {record.get("candidate_count")} branch or worktree cleanup candidates are blocked by {record.get("reason")}. Use evidence and the machine-readable reconciliation plan in {discovery_path}, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
"""


def reconciliation_record_matches_block(block: str, record: Mapping[str, Any]) -> bool:
    fingerprint = str(record.get("fingerprint") or "")
    dedupe_key = str(record.get("dedupe_key") or "")
    kind = str(record.get("kind") or "")
    reason = str(record.get("reason") or "")
    if fingerprint and fingerprint in block:
        return True
    if dedupe_key and dedupe_key in block:
        return True
    if kind == "main_checkout_dirty" and re.search(
        r"^##\s+\S+\s+Resolve dirty main checkout blocking \d+ worktree merges",
        block,
        flags=re.MULTILINE,
    ):
        return True
    if kind == "dirty_backlogged_worktree" and re.search(
        rf"^##\s+\S+\s+Resolve \d+ dirty backlogged worktrees blocked by {re.escape(reason)}",
        block,
        flags=re.MULTILINE,
    ):
        return True
    if kind == "preflight_merge_conflict" and re.search(
        r"^##\s+\S+\s+Resolve \d+ preflight-conflicting backlogged worktree merges",
        block,
        flags=re.MULTILINE,
    ):
        return True
    return False


def task_blocks_with_spans(todo_text: str) -> list[tuple[int, int, str]]:
    starts = [match.start() for match in re.finditer(r"^##\s+\S+", todo_text, flags=re.MULTILINE)]
    blocks: list[tuple[int, int, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(todo_text)
        blocks.append((start, end, todo_text[start:end]))
    return blocks


def reconciliation_task_validation_path(block: str) -> Path | None:
    match = re.search(r"^- Validation:\s+test -f\s+(.+?)\s*$", block, flags=re.MULTILINE)
    if not match:
        return None
    raw_path = match.group(1).strip()
    try:
        parts = shlex.split(raw_path)
    except ValueError:
        parts = [raw_path]
    if not parts:
        return None
    return Path(parts[0])


def refresh_reconciliation_guardrail_block(
    block: str,
    record: Mapping[str, Any],
) -> tuple[str, str, Path | None, bool]:
    heading_match = re.match(r"^##\s+(\S+)\s+[^\n]*", block)
    if not heading_match:
        return block, "", None, False
    task_id = heading_match.group(1)
    changed = False
    updated = re.sub(
        r"^##\s+\S+\s+.*$",
        f"## {task_id} {record.get('summary')}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
    if updated != block:
        changed = True
    block = updated
    fingerprint = str(record.get("fingerprint") or "")
    dedupe_key = str(record.get("dedupe_key") or "")
    if fingerprint and re.search(r"^- Fingerprint:", block, flags=re.MULTILINE):
        updated = re.sub(r"^- Fingerprint:.*$", f"- Fingerprint: {fingerprint}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    elif fingerprint:
        updated = re.sub(r"^- Track:.*$", lambda match: f"{match.group(0)}\n- Fingerprint: {fingerprint}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    if dedupe_key and re.search(r"^- Dedupe key:", block, flags=re.MULTILINE):
        updated = re.sub(r"^- Dedupe key:.*$", f"- Dedupe key: {dedupe_key}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    elif dedupe_key:
        updated = re.sub(
            r"^- Fingerprint:.*$",
            lambda match: f"{match.group(0)}\n- Dedupe key: {dedupe_key}",
            block,
            count=1,
            flags=re.MULTILINE,
        )
        changed = changed or updated != block
        block = updated
    validation_path = reconciliation_task_validation_path(block)
    if validation_path is not None:
        replacement = (
            f"- Acceptance: Reconciliation guardrail filed this because {record.get('candidate_count')} "
            f"branch or worktree cleanup candidates are blocked by {record.get('reason')}. "
            f"Use evidence and the machine-readable reconciliation plan in {validation_path}, "
            "reconcile the dirty checkout or dirty worktree group deliberately, "
            "then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases."
        )
        updated = re.sub(r"^- Acceptance:.*$", replacement, block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    return block, task_id, validation_path, changed


def refresh_existing_reconciliation_guardrails(
    *,
    todo_text: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    blocks = task_blocks_with_spans(todo_text)
    if not blocks:
        return todo_text, []
    replacements: dict[tuple[int, int], str] = {}
    refreshes: list[dict[str, Any]] = []
    for record in records:
        for start, end, block in blocks:
            if (start, end) in replacements:
                block = replacements[(start, end)]
            if not reconciliation_record_matches_block(block, record):
                continue
            refreshed_block, task_id, validation_path, changed = refresh_reconciliation_guardrail_block(block, record)
            discovery_changed = False
            if validation_path is not None and task_id:
                try:
                    before_discovery = validation_path.read_text(encoding="utf-8")
                except OSError:
                    before_discovery = ""
                write_reconciliation_guardrail_discovery_path(
                    path=validation_path,
                    task_id=task_id,
                    record=record,
                )
                try:
                    after_discovery = validation_path.read_text(encoding="utf-8")
                except OSError:
                    after_discovery = ""
                discovery_changed = before_discovery != after_discovery
            if changed:
                replacements[(start, end)] = refreshed_block
            if changed or discovery_changed:
                refreshes.append(
                    {
                        "follow_up_task_id": task_id,
                        "fingerprint": str(record.get("fingerprint") or ""),
                        "kind": str(record.get("kind") or ""),
                        "reason": str(record.get("reason") or ""),
                        "candidate_count": int(record.get("candidate_count") or 0),
                        "discovery_path": str(validation_path or ""),
                        "refreshed": True,
                    }
                )
            break
    if not replacements:
        return todo_text, refreshes
    pieces: list[str] = []
    cursor = 0
    for start, end, block in blocks:
        pieces.append(todo_text[cursor:start])
        pieces.append(replacements.get((start, end), block))
        cursor = end
    pieces.append(todo_text[cursor:])
    return "".join(pieces), refreshes


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_events(path, repair=True)


def event_merge_result(event: Mapping[str, Any]) -> dict[str, Any]:
    merge_result = event.get("merge_result") or {}
    if isinstance(merge_result, Mapping) and merge_result:
        return dict(merge_result)
    event_type = str(event.get("type") or "")
    if event_type in {"merge_reconcile_skipped", "merge_reconcile_exception"}:
        return {
            "attempted": True,
            "merged": False,
            "reason": str(event.get("reason") or event_type),
            "branch": str(event.get("branch") or ""),
        }
    return {}


def consecutive_validation_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        if str(event.get("type") or "") != "implementation_finished":
            continue
        if str(event.get("task_id") or "") != task_id:
            continue
        validation = event.get("validation_result") or {}
        if not isinstance(validation, Mapping) or not validation.get("attempted"):
            break
        if validation.get("passed", False):
            break
        failures.append(dict(event))
    failures.reverse()
    return failures


def consecutive_merge_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        event_type = str(event.get("type") or "")
        if event_type not in {
            "implementation_finished",
            "merge_reconciled",
            "merge_reconcile_skipped",
            "merge_reconcile_exception",
        }:
            continue
        if str(event.get("task_id") or "") != task_id:
            continue

        merge_result = event_merge_result(event)
        if event_type == "merge_reconciled" and event.get("resolved", False):
            break
        if event_type == "implementation_finished":
            validation = event.get("validation_result") or {}
            if isinstance(validation, Mapping) and validation.get("attempted") and not validation.get("passed", False):
                break
            if merge_result.get("merged", False):
                break

        if not merge_result.get("attempted", False):
            continue
        if merge_result.get("merged", False):
            break
        if str(merge_result.get("reason") or "") == "not_attempted":
            continue
        failures.append(dict(event))

    failures.reverse()
    return failures


def implementation_failure_label(event: Mapping[str, Any]) -> str:
    exception = event.get("exception_result") or {}
    if isinstance(exception, Mapping) and exception:
        exception_type = str(exception.get("exception_type") or "unknown")
        return f"implementation_exception:{exception_type}"
    try:
        returncode = int(event.get("returncode"))
    except (TypeError, ValueError):
        returncode = 1
    if returncode == 124:
        return "implementation_timeout"
    return f"implementation_command_returncode:{returncode}"


def consecutive_implementation_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        if str(event.get("type") or "") != "implementation_finished":
            continue
        if str(event.get("task_id") or "") != task_id:
            continue

        validation = event.get("validation_result") or {}
        if isinstance(validation, Mapping) and validation.get("attempted") and not validation.get("passed", False):
            break

        merge_result = event_merge_result(event)
        if merge_result.get("attempted", False):
            if merge_result.get("merged", False):
                break
            if str(merge_result.get("reason") or "") != "not_attempted":
                break

        try:
            returncode = int(event.get("returncode"))
        except (TypeError, ValueError):
            returncode = 1 if event.get("exception_result") else 0
        if returncode == 0:
            break
        failures.append(dict(event))

    failures.reverse()
    return failures


def write_retry_budget_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    source_task_id: str,
    failed_command: str,
    failures: Sequence[Mapping[str, Any]],
    retry_budget: int,
    failure_kind: str = "validation",
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    suffix = (
        "merge-retry-budget"
        if failure_kind == "merge"
        else "implementation-retry-budget"
        if failure_kind == "implementation"
        else "retry-budget"
    )
    path = discovery_dir / f"{date}-{task_id.lower()}-{source_task_id.lower()}-{suffix}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [str(event.get("log_path") or "") for event in failures if event.get("log_path")]
    attempt_numbers = [str(event.get("attempt") or "") for event in failures if event.get("attempt")]
    merge_result = event_merge_result(failures[-1]) if failures and failure_kind == "merge" else {}
    merge_evidence = ""
    if merge_result:
        dirty_paths = merge_result.get("dirty_paths") or []
        dirty_paths_text = ", ".join(str(path) for path in dirty_paths) if isinstance(dirty_paths, list) else str(dirty_paths)
        merge_evidence = "\n".join(
            [
                f"- Merge reason: `{str(merge_result.get('reason') or 'not recorded')}`",
                f"- Dirty paths: {dirty_paths_text or 'not recorded'}",
                f"- Branch: `{str(merge_result.get('branch') or 'not recorded')}`",
                f"- Main worktree: `{str(merge_result.get('main_worktree_path') or 'not recorded')}`",
            ]
        )
    implementation_evidence = ""
    if failures and failure_kind == "implementation":
        latest = failures[-1]
        exception = latest.get("exception_result") or {}
        exception_text = ""
        if isinstance(exception, Mapping) and exception:
            exception_text = "\n".join(
                [
                    f"- Exception type: `{str(exception.get('exception_type') or 'not recorded')}`",
                    f"- Exception phase: `{str(exception.get('phase') or 'not recorded')}`",
                    f"- Exception message: {str(exception.get('message') or 'not recorded')}",
                ]
            )
        implementation_evidence = "\n".join(
            [
                f"- Return code: `{str(latest.get('returncode') or 'not recorded')}`",
                f"- Branch: `{str(latest.get('branch') or 'not recorded')}`",
                f"- Worktree: `{str(latest.get('worktree_path') or 'not recorded')}`",
                exception_text,
            ]
        ).strip()
    content = f"""# {task_id} {failure_kind.title()} Retry-Budget Finding: {source_task_id}

Date: {date}
Source task: {source_task_id}
Follow-up task: {task_id}
Retry budget: {retry_budget}
Observed consecutive {failure_kind} failures: {len(failures)}

## Evidence

- Failed command: `{failed_command}`
- Attempts: {", ".join(attempt_numbers) or "not recorded"}
- Logs: {", ".join(log_paths) or "not recorded"}
{merge_evidence}
{implementation_evidence}

## Guardrail Result

The accelerator backlog refinery classified this as backlog work instead of
allowing another implementation attempt to loop on the same failure. The source
task is added to the strategy `blocked_tasks` list and the follow-up task below
is appended for normal daemon parsing.
"""
    path.write_text(content, encoding="utf-8")
    return path


def validation_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    failed_command: str,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    if discovery_output_path not in outputs:
        outputs.append(discovery_output_path)
    validation_command = safe_retry_validation_command(failed_command, discovery_path=discovery_path)
    return f"""## {task_id} Resolve validation retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the validation blocker, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.
"""


def safe_retry_validation_command(command: str, *, discovery_path: Path) -> str:
    """Return a parseable validation command for a retry-budget follow-up task."""

    stripped = str(command or "").strip()
    if stripped:
        commands = split_validation_commands(stripped)
        try:
            for parsed_command in commands:
                shlex.split(parsed_command)
        except ValueError:
            commands = []
        if commands:
            return stripped
    return f"test -f {shlex.quote(str(discovery_path))}"


def implementation_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    discovery_path: Path,
    strategy_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    if discovery_output_path not in outputs:
        outputs.append(discovery_output_path)
    validation_command = f"test -f {shlex.quote(str(discovery_path))}"
    return f"""## {task_id} Resolve implementation retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.
"""


def merge_command_label(merge_result: Mapping[str, Any]) -> str:
    command = merge_result.get("command")
    if isinstance(command, list) and command:
        return shlex.join(str(part) for part in command)
    if command:
        return str(command)
    return f"git merge ({str(merge_result.get('reason') or 'merge_failed')})"


def merge_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    discovery_path: Path,
    strategy_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    if discovery_output_path not in outputs:
        outputs.append(discovery_output_path)
    validation_command = f"test -f {shlex.quote(str(discovery_path))}"
    return f"""## {task_id} Resolve merge retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
- Acceptance: Merge retry-budget guardrail filed this from repeated merge failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the merge blocker, verify the intended implementation changes are committed in their owning repository or submodule, run `ipfs-accelerate-agent-merge-resolver --events-path ... --apply` when the conflict is semantic, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.
"""


def record_retry_budget_findings(
    *,
    todo_path: Path,
    events_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
) -> list[dict[str, Any]]:
    """Append follow-up tasks for repeated implementation, validation, or merge failures."""

    if not todo_path.exists():
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_header_prefix_value))
    if not tasks:
        return []

    todo_text = todo_path.read_text(encoding="utf-8")
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    completed_task_ids = {task.task_id for task in tasks if task.status == "completed"}
    retry_budget_repair_task_ids = {
        task.task_id
        for task in tasks
        if is_retry_budget_repair_task(task)
    }
    events = iter_jsonl(events_path)
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []

    if implementation_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"implementation retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_implementation_failures(events, task.task_id)
            if len(failures) < implementation_retry_budget:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            failed_command = implementation_failure_label(failures[-1])
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=implementation_retry_budget,
                failure_kind="implementation",
            )
            generated_paths.append(discovery_path)
            task_block = implementation_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                discovery_path=discovery_path,
                strategy_path=strategy_path,
                depends_on=task.depends_on,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "implementation",
                }
            )

    if validation_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_validation_failures(events, task.task_id)
            if len(failures) < validation_retry_budget:
                continue
            latest_validation = failures[-1].get("validation_result") or {}
            failed_command = str(latest_validation.get("failed_command") or "")
            if not failed_command:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=validation_retry_budget,
            )
            generated_paths.append(discovery_path)
            depends_on = list(validation_depends_on) if validation_depends_on else list(task.depends_on)
            validation_command = (
                validation_task_command_transform(failed_command)
                if validation_task_command_transform is not None
                else failed_command
            )
            task_block = validation_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                failed_command=validation_command,
                discovery_path=discovery_path,
                depends_on=depends_on,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "validation",
                }
            )

    if merge_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"merge retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_merge_failures(events, task.task_id)
            if len(failures) < merge_retry_budget:
                continue
            latest_merge_result = event_merge_result(failures[-1])
            if not latest_merge_result:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            failed_command = merge_command_label(latest_merge_result)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=merge_retry_budget,
                failure_kind="merge",
            )
            generated_paths.append(discovery_path)
            task_block = merge_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                discovery_path=discovery_path,
                strategy_path=strategy_path,
                depends_on=task.depends_on,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "merge",
                }
            )

    if not findings:
        return []

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["blocked_tasks"] = blocked_tasks
    strategy["last_retry_budget_guardrail_at"] = utc_now()
    strategy["retry_budget_findings"] = findings
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_retry_budget_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return findings


def record_dependency_guardrail_findings(
    *,
    todo_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    max_findings: int = DEFAULT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record dependency guardrail outputs",
) -> list[dict[str, Any]]:
    """Append ready repair tasks for missing or self-referential dependencies."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_header_prefix_value))
    if not tasks:
        return []

    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    seen = {str(item) for item in strategy.get("dependency_guardrail_seen_fingerprints", []) if str(item).strip()}
    records = [
        record
        for record in dependency_guardrail_records(tasks)
        if str(record.get("fingerprint") or "") not in seen
        and f"dependency guardrail for {record.get('source_task_id')}" not in todo_text
    ][:max_findings]
    if not records:
        return []

    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    try:
        todo_output_path = todo_path.resolve().relative_to((repo_root or todo_path.parent).resolve()).as_posix()
    except ValueError:
        todo_output_path = todo_path.as_posix()
    for record in records:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_dependency_guardrail_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            record=record,
        )
        generated_paths.append(discovery_path)
        source_task_id = str(record.get("source_task_id") or "")
        task_block = dependency_guardrail_task_block(
            task_id=follow_up_task_id,
            source_task_id=source_task_id,
            discovery_path=discovery_path,
            todo_output_path=todo_output_path,
            discovery_output_path=discovery_output_path,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        if source_task_id and source_task_id not in blocked_tasks:
            blocked_tasks.append(source_task_id)
        findings.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": follow_up_task_id,
                "missing_dependencies": list(record.get("missing_dependencies", []) or []),
                "self_references": list(record.get("self_references", []) or []),
                "dependency_cycle": list(record.get("dependency_cycle", []) or []),
                "duplicate_task_id": str(record.get("duplicate_task_id") or ""),
                "duplicate_task_lines": list(record.get("duplicate_task_lines", []) or []),
                "discovery_path": str(discovery_path),
                "fingerprint": str(record.get("fingerprint") or ""),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["blocked_tasks"] = blocked_tasks
    strategy["dependency_guardrail_seen_fingerprints"] = sorted(
        seen | {str(record.get("fingerprint") or "") for record in records if record.get("fingerprint")}
    )
    strategy["last_dependency_guardrail_at"] = utc_now()
    strategy["dependency_guardrail_findings"] = findings
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_dependency_guardrail_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return findings


def record_reconciliation_guardrail_findings(
    *,
    todo_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    max_findings: int = DEFAULT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record reconciliation guardrail outputs",
    additional_generated_status_paths: Sequence[Path | str] = (),
    additional_generated_status_prefixes: Sequence[Path | str] = (),
) -> list[dict[str, Any]]:
    """Append deliberate cleanup tasks for blocked worktree reconciliation."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    seen = {
        str(item)
        for item in strategy.get("reconciliation_guardrail_seen_fingerprints", [])
        if str(item).strip()
    }

    def already_present(record: Mapping[str, Any]) -> bool:
        fingerprint = str(record.get("fingerprint") or "")
        dedupe_key = str(record.get("dedupe_key") or "")
        if fingerprint and fingerprint in todo_text:
            return True
        if dedupe_key and dedupe_key in todo_text:
            return True
        kind = str(record.get("kind") or "")
        reason = str(record.get("reason") or "")
        if kind == "main_checkout_dirty" and "Resolve dirty main checkout blocking" in todo_text:
            return True
        if kind == "dirty_backlogged_worktree" and f"dirty backlogged worktrees blocked by {reason}" in todo_text:
            return True
        if kind == "preflight_merge_conflict" and "preflight-conflicting backlogged worktree merges" in todo_text:
            return True
        return False

    filter_repo_root = (repo_root or todo_path.parent).resolve()
    generated_paths, generated_prefixes = generated_guardrail_status_filters(
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        repo_root=filter_repo_root,
        additional_generated_paths=additional_generated_status_paths,
        additional_generated_prefixes=additional_generated_status_prefixes,
    )
    all_records = reconciliation_guardrail_records(
        reconciliation_result=reconciliation_result,
        cleanup_result=cleanup_result,
        generated_status_paths=generated_paths,
        generated_status_prefixes=generated_prefixes,
    )
    refreshed_todo_text, refreshes = refresh_existing_reconciliation_guardrails(
        todo_text=todo_text,
        records=all_records,
    )
    if refreshes:
        todo_text = refreshed_todo_text

    records = [
        record
        for record in all_records
        if str(record.get("fingerprint") or "") not in seen
        and not already_present(record)
    ][:max_findings]
    if not records and not refreshes:
        return []

    try:
        todo_output_path = todo_path.resolve().relative_to((repo_root or todo_path.parent).resolve()).as_posix()
    except ValueError:
        todo_output_path = todo_path.as_posix()
    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    for record in records:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_reconciliation_guardrail_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            record=record,
        )
        generated_paths.append(discovery_path)
        task_block = reconciliation_guardrail_task_block(
            task_id=follow_up_task_id,
            record=record,
            discovery_path=discovery_path,
            todo_output_path=todo_output_path,
            discovery_output_path=discovery_output_path,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        findings.append(
            {
                "follow_up_task_id": follow_up_task_id,
                "fingerprint": str(record.get("fingerprint") or ""),
                "kind": str(record.get("kind") or ""),
                "reason": str(record.get("reason") or ""),
                "candidate_count": int(record.get("candidate_count") or 0),
                "discovery_path": str(discovery_path),
                "sample_count": len(record.get("samples", []) or []),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["reconciliation_guardrail_seen_fingerprints"] = sorted(
        seen | {str(record.get("fingerprint") or "") for record in records if record.get("fingerprint")}
    )
    strategy["last_reconciliation_guardrail_at"] = utc_now()
    strategy["reconciliation_guardrail_findings"] = [*refreshes, *findings]
    write_json(strategy_path, strategy)
    if commit_outputs and (generated_paths or refreshes):
        generated_paths.insert(0, todo_path)
        generated_paths.extend(
            Path(item["discovery_path"])
            for item in refreshes
            if str(item.get("discovery_path") or "").strip()
        )
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_reconciliation_guardrail_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return [*refreshes, *findings]


def completed_retry_budget_repairs_by_source(tasks: Sequence[Any]) -> dict[str, dict[str, str]]:
    """Map source task ids to completed retry-budget repair task metadata."""

    repairs: dict[str, dict[str, str]] = {}
    for task in tasks:
        if str(getattr(task, "status", "") or "").lower() != "completed":
            continue
        source_task_id, failure_kind = retry_budget_repair_source(task)
        if not source_task_id:
            continue
        repairs[source_task_id] = {
            "follow_up_task_id": str(getattr(task, "task_id", "") or ""),
            "failure_kind": failure_kind,
        }
    return repairs


def release_completed_guardrail_blocks(
    *,
    todo_path: Path,
    strategy_path: Path,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> list[dict[str, Any]]:
    """Unblock source tasks after guardrail repair or stale strategy state clears."""

    if not todo_path.exists() or not strategy_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    if not statuses:
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_prefix))
    completed_retry_repairs = completed_retry_budget_repairs_by_source(tasks)
    retry_budget_repair_sources_by_task_id = {
        str(getattr(task, "task_id", "") or ""): retry_budget_repair_source(task)
        for task in tasks
    }
    retry_budget_repair_task_ids = {
        task_id
        for task_id, (source_task_id, _failure_kind) in retry_budget_repair_sources_by_task_id.items()
        if source_task_id
    }
    pending_retry_repair_sources = {
        source_task_id
        for task in tasks
        if str(getattr(task, "status", "") or "").lower() != "completed"
        for source_task_id, _failure_kind in (retry_budget_repair_source(task),)
        if source_task_id
    }
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]

    releases: list[dict[str, Any]] = []
    deduplicated_blocked_tasks = list(dict.fromkeys(blocked_tasks))
    if len(deduplicated_blocked_tasks) != len(blocked_tasks):
        duplicate_ids = sorted(
            {
                task_id
                for task_id in blocked_tasks
                if blocked_tasks.count(task_id) > 1
            }
        )
        releases.extend(
            {
                "source_task_id": task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "reason": "duplicate_strategy_block",
            }
            for task_id in duplicate_ids
        )
        blocked_tasks = deduplicated_blocked_tasks

    active_dependency_records = dependency_guardrail_records(
        parse_task_file(todo_path, task_header_prefix(task_prefix))
    )
    active_dependency_fingerprints = {
        str(record.get("fingerprint") or "")
        for record in active_dependency_records
        if str(record.get("fingerprint") or "").strip()
    }
    active_dependency_sources = {
        str(record.get("source_task_id") or "")
        for record in active_dependency_records
        if str(record.get("source_task_id") or "").strip()
    }
    raw_dependency_findings = strategy.get("dependency_guardrail_findings")
    if isinstance(raw_dependency_findings, list):
        retained_dependency_findings: list[Any] = []
        pruned_dependency_findings = False
        for raw_record in raw_dependency_findings:
            if not isinstance(raw_record, Mapping):
                pruned_dependency_findings = True
                continue
            source_task_id = str(raw_record.get("source_task_id") or "")
            follow_up_task_id = str(raw_record.get("follow_up_task_id") or "")
            fingerprint = str(raw_record.get("fingerprint") or "")
            if fingerprint and fingerprint not in active_dependency_fingerprints:
                if source_task_id in active_dependency_sources:
                    retained_dependency_findings.append(raw_record)
                    continue
                pruned_dependency_findings = True
                if source_task_id:
                    if source_task_id in blocked_tasks:
                        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
                    releases.append(
                        {
                            "source_task_id": source_task_id,
                            "follow_up_task_id": follow_up_task_id,
                            "guardrail_kind": "dependency_guardrail",
                            "reason": "dependency_metadata_resolved",
                        }
                    )
                continue
            retained_dependency_findings.append(raw_record)
        if pruned_dependency_findings:
            strategy["dependency_guardrail_findings"] = retained_dependency_findings

    guardrail_groups = (
        ("retry_budget", strategy.get("retry_budget_findings")),
        ("dependency_guardrail", strategy.get("dependency_guardrail_findings")),
    )
    active_guardrail_sources: set[str] = set()
    for guardrail_kind, raw_records in guardrail_groups:
        if not isinstance(raw_records, list):
            continue
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                continue
            source_task_id = str(raw_record.get("source_task_id") or "")
            follow_up_task_id = str(raw_record.get("follow_up_task_id") or "")
            if not source_task_id or not follow_up_task_id:
                continue
            active_guardrail_sources.add(source_task_id)
            if source_task_id not in blocked_tasks:
                continue
            if statuses.get(follow_up_task_id) != "completed":
                continue
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "guardrail_kind": guardrail_kind,
                }
            )

    for source_task_id in list(blocked_tasks):
        repair = completed_retry_repairs.get(source_task_id)
        if not repair:
            continue
        follow_up_task_id = str(repair.get("follow_up_task_id") or "")
        if not follow_up_task_id:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": follow_up_task_id,
                "guardrail_kind": "retry_budget",
                "failure_kind": str(repair.get("failure_kind") or ""),
                "reason": "historical_retry_repair_completed",
            }
        )

    for source_task_id in list(blocked_tasks):
        original_source_task_id, failure_kind = retry_budget_repair_sources_by_task_id.get(
            source_task_id,
            ("", ""),
        )
        if not original_source_task_id:
            continue
        if source_task_id not in pending_retry_repair_sources:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "failure_kind": failure_kind,
                "reason": "recursive_retry_repair_block",
                "original_source_task_id": original_source_task_id,
            }
        )

    for source_task_id in list(blocked_tasks):
        status = statuses.get(source_task_id)
        if status is None or status == "completed":
            continue
        if source_task_id not in retry_budget_repair_task_ids:
            continue
        if source_task_id in active_guardrail_sources or source_task_id in active_dependency_sources:
            continue
        if source_task_id in pending_retry_repair_sources:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "reason": "no_guardrail_repair_path",
            }
        )

    for source_task_id in list(blocked_tasks):
        status = statuses.get(source_task_id)
        if status is None:
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": "",
                    "guardrail_kind": "stale_strategy_block",
                    "reason": "missing_task",
                }
            )
            continue
        if status == "completed":
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": "",
                    "guardrail_kind": "stale_strategy_block",
                    "reason": "source_completed",
                }
            )

    recursive_retry_repair_task_ids: list[str] = []
    for task_id, (source_task_id, failure_kind) in retry_budget_repair_sources_by_task_id.items():
        if not task_id or not source_task_id:
            continue
        if task_id not in statuses:
            continue
        if statuses.get(task_id) == "completed":
            continue
        if source_task_id not in retry_budget_repair_task_ids:
            continue
        original_source_task_id, _original_failure_kind = retry_budget_repair_sources_by_task_id.get(
            source_task_id,
            ("", ""),
        )
        recursive_retry_repair_task_ids.append(task_id)
        releases.append(
            {
                "source_task_id": task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "retry_budget",
                "failure_kind": failure_kind,
                "reason": "recursive_retry_repair_task_retired",
                "parent_repair_task_id": source_task_id,
                "original_source_task_id": original_source_task_id,
            }
        )

    if recursive_retry_repair_task_ids:
        todo_text, retired_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            recursive_retry_repair_task_ids,
            task_prefix=task_prefix,
            status="completed",
        )
        if retired_task_ids:
            todo_path.write_text(todo_text, encoding="utf-8")
            statuses.update({task_id: "completed" for task_id in retired_task_ids})
            strategy["last_recursive_retry_repair_retired_task_ids"] = retired_task_ids

    if not releases:
        return []
    strategy["blocked_tasks"] = blocked_tasks
    strategy["last_guardrail_unblock_at"] = utc_now()
    strategy["guardrail_unblock_releases"] = releases
    write_json(strategy_path, strategy)
    return releases


def record_codebase_scan_findings(
    *,
    todo_path: Path,
    state_path: Path | None,
    strategy_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
) -> list[dict[str, Any]]:
    """Feed a todo board with static codebase findings when backlog runs low."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=strategy,
        last_scan_key="last_codebase_scan_at",
        last_drained_scan_task_count_key="last_drained_codebase_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        return []

    seen = {str(item) for item in strategy.get("codebase_scan_seen_fingerprints", []) if str(item).strip()}
    findings = scan_codebase_findings(
        repo_root,
        max_findings=max_findings,
        seen_fingerprints=seen,
        exhaustive=mode.endswith("drained_exhaustive"),
        skip_prefixes=skip_prefixes,
    )
    strategy["last_codebase_scan_at"] = utc_now()
    strategy["last_codebase_scan_mode"] = mode
    if current_open == 0 or mode.endswith("drained_exhaustive"):
        strategy["last_drained_codebase_scan_task_count"] = task_count
    strategy["codebase_scan_seen_fingerprints"] = sorted(seen | {finding.fingerprint for finding in findings})
    if not findings:
        strategy["last_codebase_scan_findings"] = []
        write_json(strategy_path, strategy)
        return []

    appended: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    for finding in findings:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_codebase_scan_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            finding=finding,
        )
        generated_paths.append(discovery_path)
        task_block = codebase_scan_task_block(
            task_id=follow_up_task_id,
            finding=finding,
            discovery_path=discovery_path,
            depends_on=depends_on,
            discovery_output_path=discovery_output_path,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        appended.append(
            {
                "follow_up_task_id": follow_up_task_id,
                "fingerprint": finding.fingerprint,
                "kind": finding.kind,
                "source": f"{finding.root_relative_path}:{finding.line_number}",
                "discovery_path": str(discovery_path),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["last_codebase_scan_findings"] = appended
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_codebase_scan_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return appended


def record_objective_backlog_findings(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    strategy_path: Path,
    state_path: Path | None = None,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    todo_vector_index_path: Path | None = None,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
) -> list[dict[str, Any]]:
    """Feed a todo board from objective-heap gaps when backlog runs low."""

    if max_findings <= 0 or not todo_path.exists() or not objective_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=strategy,
        last_scan_key="last_objective_goal_scan_at",
        last_drained_scan_task_count_key="last_drained_objective_goal_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        return []

    seen = {str(item) for item in strategy.get("objective_goal_seen_fingerprints", []) if str(item).strip()}
    records = generate_objective_todos(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        task_prefix=task_prefix,
        depends_on=depends_on,
        max_findings=max_findings,
        seen_fingerprints=seen,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        todo_vector_index_path=todo_vector_index_path,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path,
    )
    strategy["last_objective_goal_scan_at"] = utc_now()
    strategy["last_objective_goal_scan_mode"] = mode
    if current_open == 0 or mode.endswith("drained_exhaustive"):
        strategy["last_drained_objective_goal_scan_task_count"] = task_count
    strategy["objective_goal_seen_fingerprints"] = sorted(
        seen | {record.finding.fingerprint for record in records}
    )

    appended = [
        {
            "follow_up_task_id": record.task_id,
            "fingerprint": record.finding.fingerprint,
            "kind": "objective_goal_gap",
            "goal_id": record.finding.goal_id,
            "missing_evidence": record.finding.missing_evidence,
            "bundle_key": record.finding.bundle_key,
            "bundle_shard": repo_relative_path(repo_root, bundle_dir / f"{safe_bundle_key(record.finding.bundle_key)}.todo.md"),
            "bundle_strategy": record.finding.bundle_strategy,
            "graph_depth": record.finding.graph_depth,
            "parent_goal_ids": record.finding.parent_goal_ids,
            "candidate_kind": record.finding.candidate_kind,
            "merge_key": record.finding.merge_key,
            "merge_family": record.finding.merge_family or record.finding.surplus_group,
            "merge_role": record.finding.merge_role or record.finding.candidate_kind,
            "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
            "work_scope": record.finding.work_scope,
            "goal_packet_key": record.finding.goal_packet_key,
            "goal_packet_role": record.finding.goal_packet_role,
            "goal_packet_goal_ids": record.finding.goal_packet_goal_ids,
            "goal_packet_task_count": record.finding.goal_packet_task_count,
            "goal_packet_work_item_count": record.finding.goal_packet_work_item_count,
            "todo_vector_key": record.finding.todo_vector_key,
            "discovery_path": str(record.discovery_path),
        }
        for record in records
    ]
    strategy["last_objective_todo_vector_index_path"] = str(
        todo_vector_index_path or bundle_dir / "todo_vector_index.json"
    )
    strategy["last_objective_surplus_findings_per_goal"] = surplus_findings_per_goal
    strategy["last_objective_surplus_min_terms_per_todo"] = surplus_min_terms_per_todo
    strategy["last_objective_goal_scan_findings"] = appended
    write_json(strategy_path, strategy)
    if commit_outputs and records:
        generated_paths = [todo_path]
        generated_paths.extend(record.discovery_path for record in records)
        generated_paths.append(bundle_dir / "index.json")
        generated_paths.extend(bundle_path(bundle_dir, record.finding.bundle_key) for record in records)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_objective_goal_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return appended


def record_configured_objective_backlog_findings(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    strategy_path: Path,
    state_path: Path | None = None,
    bundle_dir: Path | None = None,
    dataset_dir: Path | None = None,
    default_bundle_dir: Path | None = None,
    default_dataset_dir: Path | None = None,
    todo_vector_index_path: Path | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
) -> list[dict[str, Any]]:
    """Run objective backlog refill with common wrapper-level defaults."""

    resolved_bundle_dir = (
        bundle_dir
        or default_bundle_dir
        or repo_root / "data" / "agent_supervisor" / "objective_bundles"
    )
    resolved_dataset_dir = dataset_dir if dataset_dir is not None else default_dataset_dir
    return record_objective_backlog_findings(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=resolved_bundle_dir,
        dataset_dir=resolved_dataset_dir,
        strategy_path=strategy_path,
        state_path=state_path,
        task_prefix=task_id_prefix(task_header_prefix_value),
        depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=depends_on_if_present,
        ),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        todo_vector_index_path=todo_vector_index_path,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(repo_root, discovery_dir, default=discovery_output_path_default),
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
    )


def record_configured_codebase_scan_findings(
    *,
    todo_path: Path,
    state_path: Path | None,
    strategy_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
) -> list[dict[str, Any]]:
    """Run codebase backlog refill with common wrapper-level defaults."""

    return record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        repo_root=repo_root,
        task_prefix=task_id_prefix(task_header_prefix_value),
        depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=depends_on_if_present,
        ),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(repo_root, discovery_dir, default=discovery_output_path_default),
        skip_prefixes=skip_prefixes,
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
    )


def record_configured_retry_budget_findings(
    *,
    todo_path: Path,
    events_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on_if_present: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    strip_validation_failure_kind: bool = False,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
) -> list[dict[str, Any]]:
    """Run retry-budget guardrails with common wrapper-level defaults."""

    if not todo_path.exists():
        return []
    resolved_repo_root = repo_root or git_toplevel_for_path(todo_path.parent) or todo_path.parent
    task_prefix_value = task_id_prefix(task_header_prefix_value)
    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value=task_header_prefix_value,
        task_prefix=task_prefix_value,
        validation_retry_budget=validation_retry_budget,
        merge_retry_budget=merge_retry_budget,
        implementation_retry_budget=implementation_retry_budget,
        validation_depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=validation_depends_on_if_present,
        ),
        validation_task_command_transform=validation_task_command_transform,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(
            resolved_repo_root,
            discovery_dir,
            default=discovery_output_path_default,
        ),
        commit_outputs=commit_outputs,
        repo_root=resolved_repo_root,
        commit_subject=commit_subject,
    )
    if strip_validation_failure_kind:
        for finding in findings:
            if finding.get("failure_kind") == "validation":
                finding.pop("failure_kind", None)
    return findings


def _configured_recorder_kwargs(
    recorder: object,
    overrides: Mapping[str, Any],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    params = {
        field.name: getattr(recorder, field.name)
        for field in fields(recorder)
        if field.name != "prepare_environment"
    }
    translated = dict(overrides)
    translated.pop("prepare_environment", None)
    for source, target in (aliases or {}).items():
        if source not in translated:
            continue
        if target in translated:
            raise TypeError(f"received both {source!r} and {target!r}")
        translated[target] = translated.pop(source)
    params.update(translated)
    return params


def _prepare_configured_recorder(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


@dataclass(frozen=True)
class ConfiguredObjectiveBacklogRecorder:
    """Callable objective-refill recorder with wrapper-specific defaults."""

    repo_root: Path
    objective_path: Path
    todo_path: Path
    discovery_dir: Path
    strategy_path: Path
    state_path: Path | None = None
    bundle_dir: Path | None = None
    dataset_dir: Path | None = None
    default_bundle_dir: Path | None = None
    default_dataset_dir: Path | None = None
    todo_vector_index_path: Path | None = None
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    depends_on_if_present: Sequence[str] = ()
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS
    force: bool = False
    persist_ast_dataset: bool = True
    write_todo_vector_index: bool = True
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    commit_outputs: bool = False
    commit_subject: str = "Agent: record objective backlog findings"
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> list[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_objective_backlog_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={"task_header_prefix": "task_header_prefix_value"},
            )
        )


@dataclass(frozen=True)
class ConfiguredCodebaseScanRecorder:
    """Callable codebase-scan recorder with wrapper-specific defaults."""

    todo_path: Path
    state_path: Path | None
    strategy_path: Path
    discovery_dir: Path
    repo_root: Path
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    depends_on_if_present: Sequence[str] = ()
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS
    force: bool = False
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES
    commit_outputs: bool = False
    commit_subject: str = "Agent: record codebase scan backlog findings"
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> list[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_codebase_scan_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={"task_header_prefix": "task_header_prefix_value"},
            )
        )


@dataclass(frozen=True)
class ConfiguredRetryBudgetRecorder:
    """Callable retry-budget recorder with wrapper-specific defaults."""

    todo_path: Path
    events_path: Path
    strategy_path: Path
    discovery_dir: Path
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET
    validation_depends_on_if_present: Sequence[str] = ()
    validation_task_command_transform: Callable[[str], str] | None = None
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    strip_validation_failure_kind: bool = False
    commit_outputs: bool = False
    repo_root: Path | None = None
    commit_subject: str = "Agent: record retry-budget guardrail outputs"
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> list[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_retry_budget_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={
                    "retry_budget": "validation_retry_budget",
                    "task_header_prefix": "task_header_prefix_value",
                },
            )
        )


ConfiguredBacklogRecordCallback = Callable[..., list[dict[str, Any]]]
ConfiguredBootstrapExtraKwargsFactory = Callable[[Mapping[str, Path | str]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class ConfiguredBacklogRecorderBundle:
    """Reusable bridge from configured backlog recorders to runtime hook factories."""

    objective_recorder: ConfiguredBacklogRecordCallback | None = None
    codebase_scan_recorder: ConfiguredBacklogRecordCallback | None = None
    retry_budget_recorder: ConfiguredBacklogRecordCallback | None = None

    def daemon_refill_hooks_factory(
        self,
        *,
        discovery_dir_key: str | None = None,
        discovery_dir: Path | str | None = None,
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        repo_root: Path | None = None,
        objective_extra_kwargs: Mapping[str, Any] | None = None,
        objective_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
        codebase_scan_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
        retry_budget_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        scope_label: str = "",
        before: bool = True,
        after: bool = True,
        after_order: Sequence[str] | None = None,
        log_level: int = logging.WARNING,
    ) -> Callable[[Mapping[str, Path | str]], tuple[Any, ...]]:
        """Build daemon refill hooks from this bundle without repo-local wiring."""

        from .implementation_daemon_runner import build_daemon_refill_hooks_factory_from_recorders

        return build_daemon_refill_hooks_factory_from_recorders(
            discovery_dir_key=discovery_dir_key,
            discovery_dir=discovery_dir,
            objective_recorder=self.objective_recorder,
            codebase_scan_recorder=self.codebase_scan_recorder,
            retry_budget_recorder=self.retry_budget_recorder,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            repo_root=repo_root,
            objective_extra_kwargs=objective_extra_kwargs,
            objective_extra_kwargs_factory=objective_extra_kwargs_factory,
            codebase_scan_extra_kwargs=codebase_scan_extra_kwargs,
            codebase_scan_extra_kwargs_factory=codebase_scan_extra_kwargs_factory,
            retry_budget_extra_kwargs=retry_budget_extra_kwargs,
            retry_budget_extra_kwargs_factory=retry_budget_extra_kwargs_factory,
            scope_label=scope_label,
            before=before,
            after=after,
            after_order=after_order,
            log_level=log_level,
        )

    def supervisor_refill_hooks_factory(
        self,
        *,
        discovery_dir_key: str | None = None,
        discovery_dir: Path | str | None = None,
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        repo_root: Path | None = None,
        objective_extra_kwargs: Mapping[str, Any] | None = None,
        objective_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
        codebase_scan_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
        retry_budget_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        scope_label: str = "",
        before: bool = True,
        after_once: bool = True,
        after_once_order: Sequence[str] | None = None,
        log_level: int = logging.WARNING,
    ) -> Callable[[Mapping[str, Path | str]], tuple[Any, ...]]:
        """Build supervisor refill hooks from this bundle without repo-local wiring."""

        from .implementation_supervisor_runner import build_supervisor_refill_hooks_factory_from_recorders

        return build_supervisor_refill_hooks_factory_from_recorders(
            discovery_dir_key=discovery_dir_key,
            discovery_dir=discovery_dir,
            objective_recorder=self.objective_recorder,
            codebase_scan_recorder=self.codebase_scan_recorder,
            retry_budget_recorder=self.retry_budget_recorder,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            repo_root=repo_root,
            objective_extra_kwargs=objective_extra_kwargs,
            objective_extra_kwargs_factory=objective_extra_kwargs_factory,
            codebase_scan_extra_kwargs=codebase_scan_extra_kwargs,
            codebase_scan_extra_kwargs_factory=codebase_scan_extra_kwargs_factory,
            retry_budget_extra_kwargs=retry_budget_extra_kwargs,
            retry_budget_extra_kwargs_factory=retry_budget_extra_kwargs_factory,
            scope_label=scope_label,
            before=before,
            after_once=after_once,
            after_once_order=after_once_order,
            log_level=log_level,
        )


def build_configured_backlog_recorder_bundle(
    *,
    objective_recorder: ConfiguredBacklogRecordCallback | None = None,
    codebase_scan_recorder: ConfiguredBacklogRecordCallback | None = None,
    retry_budget_recorder: ConfiguredBacklogRecordCallback | None = None,
) -> ConfiguredBacklogRecorderBundle:
    """Collect configured backlog recorders for daemon/supervisor reuse."""

    return ConfiguredBacklogRecorderBundle(
        objective_recorder=objective_recorder,
        codebase_scan_recorder=codebase_scan_recorder,
        retry_budget_recorder=retry_budget_recorder,
    )


def build_namespace_objective_backlog_recorder(
    *,
    repo_root: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    objective_path: Path | str,
    todo_path: Path | str,
    strategy_path: Path | str,
    state_path: Path | str | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredObjectiveBacklogRecorder:
    """Build an objective recorder using standard namespace artifact paths."""

    return ConfiguredObjectiveBacklogRecorder(
        repo_root=Path(repo_root),
        objective_path=Path(objective_path),
        todo_path=Path(todo_path),
        discovery_dir=namespace_paths.discovery_dir,
        default_bundle_dir=namespace_paths.objective_bundle_dir,
        default_dataset_dir=namespace_paths.objective_dataset_dir,
        todo_vector_index_path=namespace_paths.objective_todo_vector_index_path,
        strategy_path=Path(strategy_path),
        state_path=Path(state_path) if state_path is not None else None,
        task_header_prefix_value=task_header_prefix_value,
        depends_on_if_present=tuple(depends_on_if_present),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_namespace_codebase_scan_recorder(
    *,
    repo_root: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    todo_path: Path | str,
    strategy_path: Path | str,
    state_path: Path | str | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredCodebaseScanRecorder:
    """Build a codebase-scan recorder using a standard namespace discovery path."""

    return ConfiguredCodebaseScanRecorder(
        repo_root=Path(repo_root),
        todo_path=Path(todo_path),
        state_path=Path(state_path) if state_path is not None else None,
        strategy_path=Path(strategy_path),
        discovery_dir=namespace_paths.discovery_dir,
        task_header_prefix_value=task_header_prefix_value,
        depends_on_if_present=tuple(depends_on_if_present),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        skip_prefixes=tuple(skip_prefixes),
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_namespace_retry_budget_recorder(
    *,
    namespace_paths: AgentSupervisorNamespacePaths,
    todo_path: Path | str,
    events_path: Path | str,
    strategy_path: Path | str,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on_if_present: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    strip_validation_failure_kind: bool = False,
    commit_outputs: bool = False,
    repo_root: Path | str | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredRetryBudgetRecorder:
    """Build a retry-budget recorder using a standard namespace discovery path."""

    return ConfiguredRetryBudgetRecorder(
        todo_path=Path(todo_path),
        events_path=Path(events_path),
        strategy_path=Path(strategy_path),
        discovery_dir=namespace_paths.discovery_dir,
        task_header_prefix_value=task_header_prefix_value,
        validation_retry_budget=validation_retry_budget,
        merge_retry_budget=merge_retry_budget,
        implementation_retry_budget=implementation_retry_budget,
        validation_depends_on_if_present=tuple(validation_depends_on_if_present),
        validation_task_command_transform=validation_task_command_transform,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        strip_validation_failure_kind=strip_validation_failure_kind,
        commit_outputs=commit_outputs,
        repo_root=Path(repo_root) if repo_root is not None else None,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refill and guard an accelerator todo backlog")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--todo-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--strategy-path", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--discovery-output-path", default=DEFAULT_DISCOVERY_OUTPUT_PATH)
    parser.add_argument("--objective-path", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_ID_PREFIX)
    parser.add_argument("--task-header-prefix", default=DEFAULT_TASK_HEADER_PREFIX)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--validation-depends-on", action="append", default=[])
    parser.add_argument("--skip-prefix", action="append", default=[])
    parser.add_argument("--objective-scan", action="store_true")
    parser.add_argument("--codebase-scan", action="store_true")
    parser.add_argument("--retry-budget", action="store_true")
    parser.add_argument("--dependency-guardrail", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-open-tasks", type=int, default=DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS)
    parser.add_argument("--max-findings", type=int, default=DEFAULT_CODEBASE_SCAN_MAX_FINDINGS)
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS)
    parser.add_argument("--validation-retry-budget", type=int, default=DEFAULT_VALIDATION_RETRY_BUDGET)
    parser.add_argument("--merge-retry-budget", type=int, default=DEFAULT_MERGE_RETRY_BUDGET)
    parser.add_argument("--implementation-retry-budget", type=int, default=DEFAULT_IMPLEMENTATION_RETRY_BUDGET)
    parser.add_argument("--no-persist-ast-dataset", action="store_true")
    parser.add_argument("--no-objective-todo-vector-index", action="store_true")
    parser.add_argument("--objective-todo-vector-index-path", type=Path, default=None)
    parser.add_argument(
        "--objective-surplus-findings-per-goal",
        type=int,
        default=DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    )
    parser.add_argument(
        "--objective-surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    )
    parser.add_argument("--objective-summary-prefix", default=DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX)
    parser.add_argument("--commit-generated-outputs", action="store_true")
    return parser


def run_backlog_refinery(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    state_root = repo_root / "data" / "agent_supervisor"
    strategy_path = (args.strategy_path or state_root / "strategy.json").resolve()
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    state_path = args.state_path.resolve() if args.state_path else None
    events_path = args.events_path.resolve() if args.events_path else state_root / "events.jsonl"
    depends_on = split_csv(args.depends_on)
    validation_depends_on = split_csv(args.validation_depends_on)
    skip_prefixes = tuple(args.skip_prefix) if args.skip_prefix else CODEBASE_SCAN_SKIP_PREFIXES

    run_all = not (args.objective_scan or args.codebase_scan or args.retry_budget or args.dependency_guardrail)
    objective_findings: list[dict[str, Any]] = []
    codebase_findings: list[dict[str, Any]] = []
    retry_findings: list[dict[str, Any]] = []
    dependency_findings: list[dict[str, Any]] = []

    if (args.objective_scan or run_all) and args.objective_path:
        objective_findings = record_objective_backlog_findings(
            repo_root=repo_root,
            objective_path=args.objective_path.resolve(),
            todo_path=args.todo_path.resolve(),
            discovery_dir=discovery_dir,
            bundle_dir=bundle_dir,
            dataset_dir=args.dataset_dir.resolve() if args.dataset_dir else None,
            strategy_path=strategy_path,
            state_path=state_path,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
            persist_ast_dataset=not args.no_persist_ast_dataset,
            write_todo_vector_index=not args.no_objective_todo_vector_index,
            todo_vector_index_path=args.objective_todo_vector_index_path,
            surplus_findings_per_goal=args.objective_surplus_findings_per_goal,
            surplus_min_terms_per_todo=args.objective_surplus_min_terms_per_todo,
            summary_prefix=args.objective_summary_prefix,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
        )
    if args.codebase_scan or run_all:
        codebase_findings = record_codebase_scan_findings(
            todo_path=args.todo_path.resolve(),
            state_path=state_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            repo_root=repo_root,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
            discovery_output_path=args.discovery_output_path,
            skip_prefixes=skip_prefixes,
            commit_outputs=args.commit_generated_outputs,
        )
    if args.retry_budget or run_all:
        retry_findings = record_retry_budget_findings(
            todo_path=args.todo_path.resolve(),
            events_path=events_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=args.task_header_prefix,
            task_prefix=args.task_prefix,
            validation_retry_budget=args.validation_retry_budget,
            merge_retry_budget=args.merge_retry_budget,
            implementation_retry_budget=args.implementation_retry_budget,
            validation_depends_on=validation_depends_on,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
            repo_root=repo_root,
        )
    if args.dependency_guardrail or run_all:
        dependency_findings = record_dependency_guardrail_findings(
            todo_path=args.todo_path.resolve(),
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=args.task_header_prefix,
            task_prefix=args.task_prefix,
            max_findings=args.max_findings,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
            repo_root=repo_root,
        )

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.backlog_refinery",
        "repo_root": str(repo_root),
        "todo_path": str(args.todo_path.resolve()),
        "strategy_path": str(strategy_path),
        "objective_generated_count": len(objective_findings),
        "codebase_generated_count": len(codebase_findings),
        "retry_budget_generated_count": len(retry_findings),
        "dependency_guardrail_generated_count": len(dependency_findings),
        "objective_findings": objective_findings,
        "codebase_findings": codebase_findings,
        "retry_budget_findings": retry_findings,
        "dependency_guardrail_findings": dependency_findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_backlog_refinery(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
