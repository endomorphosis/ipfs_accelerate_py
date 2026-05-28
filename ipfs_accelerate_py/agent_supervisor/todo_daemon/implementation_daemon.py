from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import pid_alive as _shared_pid_alive
from .core import process_args as _shared_process_args
from .engine import atomic_write_json as _shared_atomic_write_json
from ..event_log import append_jsonl_event, read_jsonl_events, repair_jsonl_event_log, unique_backup_path
from .runner import TodoDaemonHooks, TodoDaemonRunner

REPO_ROOT = Path.cwd()

logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")

TASK_HEADER_PREFIX = "## PORTAL-"
DEFAULT_TRACKS = [
    "platform",
    "agent",
    "graphrag",
    "data",
    "ui",
    "mobile",
    "wallet",
    "privacy",
    "runtime",
    "quality",
    "collab",
    "pwa",
    "ops",
]
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS = 1800.0
LLM_MERGE_RESOLVER_COMMAND_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND"
LLM_MERGE_RESOLVER_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS"
RECENT_NO_CHANGE_COOLDOWN_SECONDS = 1800.0
NO_CHANGE_SELECTION_PENALTY = 50
UNRESOLVED_MERGE_SELECTION_PENALTY = 1000
SHARED_WORKTREE_PATHS = ("wallet_interface/ui/node_modules",)
DEFAULT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET", "260")
)


def normalize_relative_path_list(values: Any) -> tuple[str, ...]:
    """Normalize comma/list configured repo-relative paths."""

    if values is None:
        raw_values: list[Any] = []
    elif isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    paths: list[str] = []
    for value in raw_values:
        for raw_path in str(value).split(","):
            path = raw_path.strip().strip("/")
            if not path or path.startswith("/") or "\0" in path:
                continue
            if ".." in Path(path).parts:
                continue
            if path not in paths:
                paths.append(path)
    return tuple(paths)


DEFAULT_WORKTREE_SUBMODULE_PATHS = normalize_relative_path_list(
    os.environ.get("IPFS_ACCELERATE_AGENT_WORKTREE_SUBMODULE_PATHS", "")
)
WORKTREE_SUBMODULE_PATHS = DEFAULT_WORKTREE_SUBMODULE_PATHS
EPHEMERAL_WORKTREE_PATHS = (
    *SHARED_WORKTREE_PATHS,
    ".pytest_cache",
    "test-results",
    "wallet_interface/__pycache__",
    "wallet_interface/ui/dist",
    "wallet_interface/ui/playwright-report",
    "wallet_interface/ui/test-results",
    "wallet_interface/ui/artifacts/ui-iterations/latest",
    "wallet_interface/ui/artifacts/ui-review",
    "wallet_interface/ui/artifacts/ui-screenshots",
    "wallet_interface/ui/artifacts/ui-screenshots/latest",
)
GENERATED_WORKTREE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "playwright-report",
    "test-results",
}
GENERATED_WORKTREE_SUFFIXES = (".pyc", ".pyo")
UNTRACKED_WORKTREE_CONTEXT_PREFIXES = (
    ".gitmodules",
    "docs/",
    "implementation_plan/",
    "scripts/",
    "scraper/",
    "tests/",
    "wallet_interface/",
)
GENERATED_ADD_ADD_CONFLICT_PREFIXES = (
    "data/",
    "docs/",
    "implementation_plan/",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copilot_fallback_command(*, codex: str | None, copilot: str, workspace_path: Path) -> list[str]:
        return [
                "bash",
                "-lc",
                """
prompt_file=$(mktemp)
trap 'rm -f "$prompt_file"' EXIT
cat > "$prompt_file"
codex_bin="$1"
copilot_bin="$2"
workspace="$3"
if [[ -n "$codex_bin" ]]; then
    if "$codex_bin" exec --dangerously-bypass-approvals-and-sandbox -C "$workspace" - < "$prompt_file"; then
        exit 0
    else
        rc=$?
        printf 'codex exec failed with exit %s; falling back to copilot\n' "$rc" >&2
    fi
fi
exec "$copilot_bin" --silent --allow-all-tools --allow-all-paths --no-ask-user --autopilot --prompt "$(cat "$prompt_file")"
""",
                "bash",
                codex or "",
                copilot,
                str(workspace_path),
        ]


def split_csv(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",")]
    return [item for item in raw if item and item.lower() not in {"none", "n/a"}]


def normalize_status(value: str) -> str:
    lowered = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in {"done", "complete", "completed"}:
        return "completed"
    if lowered in {"blocked", "on_hold"}:
        return "blocked"
    if lowered in {"active", "in_progress"}:
        return "in_progress"
    if lowered in {"ready", "todo", "queued", ""}:
        return "todo"
    return lowered


def normalize_task_header_prefix(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("## "):
        return stripped
    return f"## {stripped}"


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        backup_path = unique_backup_path(path, "directory-backup")
        path.rename(backup_path)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(temp_path, path)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


def write_json_atomic(path: Path, payload: Any) -> None:
    if isinstance(payload, dict):
        _shared_atomic_write_json(path, payload)
        return
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    return _shared_pid_alive(pid)


def process_command_line(pid: int) -> str:
    return _shared_process_args(pid)


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class PortalTask:
    task_id: str
    title: str
    status: str
    completion: str
    priority: str
    track: str
    depends_on: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    acceptance: str = ""
    source_line: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PortalTaskState:
    heartbeat_at: str = ""
    last_progress_at: str = ""
    active_task_id: str = ""
    active_task_title: str = ""
    active_task_track: str = ""
    active_task_started_at: str = ""
    active_attempt: int = 0
    active_phase: str = ""
    active_phase_started_at: str = ""
    active_phase_detail: str = ""
    active_log_path: str = ""
    active_worktree_path: str = ""
    active_branch: str = ""
    implementation_in_progress: bool = False
    recommended_task_id: str = ""
    recommended_actions: list[str] = field(default_factory=list)
    completed_task_ids: list[str] = field(default_factory=list)
    ready_task_ids: list[str] = field(default_factory=list)
    waiting_task_ids: list[str] = field(default_factory=list)
    blocked_task_ids: list[str] = field(default_factory=list)
    task_statuses: dict[str, str] = field(default_factory=dict)
    task_artifacts: dict[str, list[str]] = field(default_factory=dict)
    task_validation: dict[str, list[str]] = field(default_factory=dict)
    implementation_attempts: dict[str, int] = field(default_factory=dict)
    last_implementation_task_id: str = ""
    last_implementation_started_at: str = ""
    last_implementation_finished_at: str = ""
    last_implementation_returncode: int | None = None
    last_implementation_log_path: str = ""
    last_implementation_worktree_path: str = ""
    last_implementation_branch: str = ""
    last_implementation_commit: str = ""
    last_merge_started_at: str = ""
    last_merge_finished_at: str = ""
    last_merge_branch: str = ""
    last_merge_commit: str = ""
    last_merge_returncode: int | None = None
    last_merge_error: str = ""
    completed_count: int = 0
    ready_count: int = 0
    waiting_count: int = 0
    blocked_count: int = 0
    task_count: int = 0
    strategy_generation: int = 0

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "PortalTaskState":
        if not path.exists():
            return cls()
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return cls()
        if not text:
            return cls()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(payload, dict):
            return cls()
        try:
            return cls(
                heartbeat_at=str(payload.get("heartbeat_at") or ""),
                last_progress_at=str(payload.get("last_progress_at") or ""),
                active_task_id=str(payload.get("active_task_id") or ""),
                active_task_title=str(payload.get("active_task_title") or ""),
                active_task_track=str(payload.get("active_task_track") or ""),
                active_task_started_at=str(payload.get("active_task_started_at") or ""),
                active_attempt=int(payload.get("active_attempt") or 0),
                active_phase=str(payload.get("active_phase") or ""),
                active_phase_started_at=str(payload.get("active_phase_started_at") or ""),
                active_phase_detail=str(payload.get("active_phase_detail") or ""),
                active_log_path=str(payload.get("active_log_path") or ""),
                active_worktree_path=str(payload.get("active_worktree_path") or ""),
                active_branch=str(payload.get("active_branch") or ""),
                implementation_in_progress=bool(payload.get("implementation_in_progress")),
                recommended_task_id=str(payload.get("recommended_task_id") or ""),
                recommended_actions=[str(item) for item in payload.get("recommended_actions", []) or []],
                completed_task_ids=[str(item) for item in payload.get("completed_task_ids", []) or []],
                ready_task_ids=[str(item) for item in payload.get("ready_task_ids", []) or []],
                waiting_task_ids=[str(item) for item in payload.get("waiting_task_ids", []) or []],
                blocked_task_ids=[str(item) for item in payload.get("blocked_task_ids", []) or []],
                task_statuses={str(key): str(value) for key, value in (payload.get("task_statuses") or {}).items()},
                task_artifacts={
                    str(key): [str(item) for item in value]
                    for key, value in (payload.get("task_artifacts") or {}).items()
                    if isinstance(value, list)
                },
                task_validation={
                    str(key): [str(item) for item in value]
                    for key, value in (payload.get("task_validation") or {}).items()
                    if isinstance(value, list)
                },
                implementation_attempts={
                    str(key): int(value)
                    for key, value in (payload.get("implementation_attempts") or {}).items()
                    if str(value).isdigit()
                },
                last_implementation_task_id=str(payload.get("last_implementation_task_id") or ""),
                last_implementation_started_at=str(payload.get("last_implementation_started_at") or ""),
                last_implementation_finished_at=str(payload.get("last_implementation_finished_at") or ""),
                last_implementation_returncode=(
                    int(payload["last_implementation_returncode"])
                    if payload.get("last_implementation_returncode") is not None
                    else None
                ),
                last_implementation_log_path=str(payload.get("last_implementation_log_path") or ""),
                last_implementation_worktree_path=str(payload.get("last_implementation_worktree_path") or ""),
                last_implementation_branch=str(payload.get("last_implementation_branch") or ""),
                last_implementation_commit=str(payload.get("last_implementation_commit") or ""),
                last_merge_started_at=str(payload.get("last_merge_started_at") or ""),
                last_merge_finished_at=str(payload.get("last_merge_finished_at") or ""),
                last_merge_branch=str(payload.get("last_merge_branch") or ""),
                last_merge_commit=str(payload.get("last_merge_commit") or ""),
                last_merge_returncode=(
                    int(payload["last_merge_returncode"])
                    if payload.get("last_merge_returncode") is not None
                    else None
                ),
                last_merge_error=str(payload.get("last_merge_error") or ""),
                completed_count=int(payload.get("completed_count") or 0),
                ready_count=int(payload.get("ready_count") or 0),
                waiting_count=int(payload.get("waiting_count") or 0),
                blocked_count=int(payload.get("blocked_count") or 0),
                task_count=int(payload.get("task_count") or 0),
                strategy_generation=int(payload.get("strategy_generation") or 0),
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


def state_file_repair_reason(path: Path) -> str:
    if not path.exists():
        return "missing_state_file"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unreadable_state_file"
    if not text:
        return "empty_state_file"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "invalid_state_json"
    if not isinstance(payload, dict):
        return "non_object_state_json"
    int_fields = (
        "active_attempt",
        "completed_count",
        "ready_count",
        "waiting_count",
        "blocked_count",
        "task_count",
        "strategy_generation",
    )
    optional_int_fields = ("last_implementation_returncode", "last_merge_returncode")
    try:
        for field_name in int_fields:
            int(payload.get(field_name) or 0)
        for field_name in optional_int_fields:
            if payload.get(field_name) is not None:
                int(payload[field_name])
    except (TypeError, ValueError):
        return "malformed_state_metadata"
    for field_name in ("task_statuses", "task_artifacts", "task_validation", "implementation_attempts"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, dict):
            return "malformed_state_metadata"
    return ""


def parse_task_file(path: Path, task_header_prefix: str = TASK_HEADER_PREFIX) -> list[PortalTask]:
    task_header_prefix = normalize_task_header_prefix(task_header_prefix)
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks: list[PortalTask] = []
    current_id = ""
    current_title = ""
    current_line = 0
    block: list[str] = []

    def flush() -> None:
        nonlocal block, current_id, current_title, current_line
        if not current_id:
            return
        metadata: dict[str, str] = {}
        for line in block:
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            metadata[key.strip().lower()] = value.strip()
        tasks.append(
            PortalTask(
                task_id=current_id,
                title=current_title,
                status=normalize_status(metadata.get("status", "todo")),
                completion=str(metadata.get("completion", "manual")).strip().lower(),
                priority=str(metadata.get("priority", "P2")).strip().upper(),
                track=str(metadata.get("track", "ops")).strip().lower(),
                depends_on=split_csv(metadata.get("depends on", "")),
                outputs=split_csv(metadata.get("outputs", "")),
                validation=[item.strip() for item in metadata.get("validation", "").split(";") if item.strip()],
                acceptance=str(metadata.get("acceptance", "")).strip(),
                source_line=current_line,
                metadata=dict(metadata),
            )
        )
        current_id = ""
        current_title = ""
        current_line = 0
        block = []

    for index, line in enumerate(lines, start=1):
        if line.startswith(task_header_prefix):
            flush()
            header = line[3:].strip()
            parts = header.split(" ", 1)
            if len(parts) == 1:
                current_id = parts[0]
                current_title = ""
            else:
                current_id, current_title = parts[0], parts[1].strip()
            current_line = index
            block = []
            continue
        if current_id:
            block.append(line)

    flush()
    return tasks


class PortalImplementationDaemon:
    shared_todo_runner_class = TodoDaemonRunner
    shared_todo_hooks_class = TodoDaemonHooks

    def __init__(
        self,
        *,
        todo_path: Path,
        state_path: Path,
        strategy_path: Path,
        events_path: Path,
        repo_root: Path | None = None,
        task_header_prefix: str = TASK_HEADER_PREFIX,
        implement: bool = False,
        implementation_command: str | None = None,
        implementation_timeout: float = DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS,
        implementation_log_dir: Path | None = None,
        use_ephemeral_worktree: bool = False,
        worktree_root: Path | None = None,
        worktree_submodule_paths: Any = None,
        llm_merge_resolver_command: str | None = None,
        llm_merge_resolver_timeout_seconds: float | None = None,
    ) -> None:
        self.todo_path = todo_path
        self.state_path = state_path
        self.strategy_path = strategy_path
        self.events_path = events_path
        self.repo_root = repo_root or REPO_ROOT
        self.task_header_prefix = normalize_task_header_prefix(task_header_prefix)
        self.implement = implement
        self.implementation_command = implementation_command
        self.implementation_timeout = implementation_timeout
        self.implementation_log_dir = implementation_log_dir or self.state_path.parent / "implementation_logs"
        self.use_ephemeral_worktree = use_ephemeral_worktree
        self.worktree_root = worktree_root or Path(tempfile.gettempdir()) / "211-ai-implementation-worktrees"
        self.llm_merge_resolver_command = (
            llm_merge_resolver_command
            if llm_merge_resolver_command is not None
            else os.environ.get(LLM_MERGE_RESOLVER_COMMAND_ENV, "")
        ).strip()
        self.llm_merge_resolver_timeout_seconds = llm_merge_resolver_timeout_seconds
        configured_submodules = (
            DEFAULT_WORKTREE_SUBMODULE_PATHS
            if worktree_submodule_paths is None
            else normalize_relative_path_list(worktree_submodule_paths)
        )
        self.worktree_submodule_paths = configured_submodules

    def load_strategy(self) -> dict[str, Any]:
        defaults = {
            "generation": 0,
            "focus_tracks": DEFAULT_TRACKS,
            "blocked_tasks": [],
            "deprioritized_tasks": [],
            "last_rewrite_at": "",
            "last_rewrite_reason": "",
        }
        if not self.strategy_path.exists():
            write_json_atomic(self.strategy_path, defaults)
            return defaults
        payload = load_json_dict(self.strategy_path)
        if payload is None:
            logger.warning("Strategy file is missing or invalid JSON; using defaults: %s", self.strategy_path)
            repaired = {
                **defaults,
                "last_strategy_repair_at": utc_now(),
                "last_strategy_repair_reason": "invalid_or_unreadable_strategy_file",
            }
            write_json_atomic(self.strategy_path, repaired)
            self._record_event(
                "strategy_file_repaired",
                {
                    "strategy_path": str(self.strategy_path),
                    "reason": "invalid_or_unreadable_strategy_file",
                },
            )
            return repaired
        merged = {**defaults, **payload}
        merged["focus_tracks"] = [str(item).lower() for item in merged.get("focus_tracks", DEFAULT_TRACKS)]
        merged["blocked_tasks"] = [str(item) for item in merged.get("blocked_tasks", [])]
        merged["deprioritized_tasks"] = [str(item) for item in merged.get("deprioritized_tasks", [])]
        return merged

    def _mark_long_running_phase(self, *, task_id: str, phase: str, detail: str = "") -> None:
        state = PortalTaskState.load(self.state_path)
        now = utc_now()
        if task_id:
            state.active_task_id = task_id
        if state.active_phase != phase or state.active_phase_detail != detail:
            state.active_phase_started_at = now
        state.active_phase = phase
        state.active_phase_detail = detail
        state.heartbeat_at = now
        state.save(self.state_path)
        self._record_event(
            "daemon_phase_heartbeat",
            {
                "task_id": task_id,
                "phase": phase,
                "detail": detail,
            },
        )

    def _record_empty_backlog_state(self, *, reason: str, error: str = "") -> dict[str, Any]:
        previous = PortalTaskState.load(self.state_path)
        strategy = self.load_strategy()
        live_inflight_implementation = self._find_live_inflight_implementation()
        now = utc_now()
        state = PortalTaskState.load(self.state_path)
        state.heartbeat_at = now
        if not state.last_progress_at:
            state.last_progress_at = now
        state.completed_task_ids = []
        state.ready_task_ids = []
        state.waiting_task_ids = []
        state.blocked_task_ids = []
        state.completed_count = 0
        state.ready_count = 0
        state.waiting_count = 0
        state.blocked_count = 0
        state.task_count = 0
        state.task_statuses = {}
        state.task_artifacts = {}
        state.task_validation = {}
        state.strategy_generation = int(strategy.get("generation", 0))
        if not (previous.implementation_in_progress and live_inflight_implementation is not None):
            state.active_task_id = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            self._clear_active_execution_state(state)
            state.recommended_task_id = ""
            state.recommended_actions = []
        state.save(self.state_path)
        payload = {
            "reason": reason,
            "todo_path": str(self.todo_path),
            "task_count": 0,
            "active_task_id": state.active_task_id,
        }
        if error:
            payload["error"] = error[-4000:]
        self._record_event("daemon_no_tasks", payload)
        return {
            "task_count": 0,
            "completed_count": 0,
            "ready_count": 0,
            "waiting_count": 0,
            "blocked_count": 0,
            "active_task_id": state.active_task_id,
            "state_path": str(self.state_path),
            "strategy_path": str(self.strategy_path),
            "events_path": str(self.events_path),
            "implementation_result": None,
            "merge_reconciliation": [],
            "reason": reason,
        }

    def ensure_state_file(self) -> dict[str, Any]:
        """Repair malformed durable state before this pass reads it."""

        reason = state_file_repair_reason(self.state_path)
        if not reason or reason == "missing_state_file":
            return {"repaired": False, "reason": reason or "valid", "path": str(self.state_path)}
        state = PortalTaskState()
        state.save(self.state_path)
        result = {"repaired": True, "reason": reason, "path": str(self.state_path)}
        self._record_event("state_file_repaired", result)
        return result

    def ensure_event_log_file(self) -> dict[str, Any]:
        """Repair malformed event-log storage before this pass reads or writes it."""

        result = repair_jsonl_event_log(self.events_path)
        if result.get("repaired"):
            append_jsonl_event(self.events_path, "event_log_repaired", result)
        return result

    def run_once(self) -> dict[str, Any]:
        event_log_repair = self.ensure_event_log_file()
        state_file_repair = self.ensure_state_file()
        try:
            tasks = parse_task_file(self.todo_path, self.task_header_prefix)
        except (OSError, UnicodeDecodeError) as exc:
            return self._record_empty_backlog_state(reason="todo_read_failed", error=str(exc))
        if not tasks:
            return self._record_empty_backlog_state(reason="no_tasks_found")
        previous = PortalTaskState.load(self.state_path)
        strategy = self.load_strategy()
        now = utc_now()
        status_completed_task_ids = {task.task_id for task in tasks if task.status == "completed"}
        strategy_blocked_task_ids = {str(task_id) for task_id in strategy.get("blocked_tasks", [])}
        merge_skip_task_ids = status_completed_task_ids | strategy_blocked_task_ids
        merge_reconciliation = self._reconcile_failed_merges(skip_task_ids=merge_skip_task_ids)
        unresolved_merge_failures = self._unresolved_merge_failures_by_task(skip_task_ids=merge_skip_task_ids)
        recent_outcomes = self._latest_implementation_finished_by_task()
        successfully_merged_task_ids = self._successfully_merged_task_ids()
        live_inflight_implementation = self._find_live_inflight_implementation()

        previous_completed = set(previous.completed_task_ids)
        completed_set: set[str] = set()
        newly_completed: list[str] = []
        resolved_statuses: dict[str, str] = {}
        task_artifacts: dict[str, list[str]] = {}

        for task in tasks:
            existing_outputs = [item for item in task.outputs if (self.repo_root / item).exists()]
            task_artifacts[task.task_id] = existing_outputs
            unresolved_merge_failure = (
                task.task_id in unresolved_merge_failures
                or self._has_unresolved_merge_failure(task, previous)
            )
            artifact_complete = (
                task.completion == "artifact"
                and bool(task.outputs)
                and len(existing_outputs) == len(task.outputs)
                and not unresolved_merge_failure
            )
            merged_complete = task.task_id in successfully_merged_task_ids and not unresolved_merge_failure
            if task.status == "completed" or artifact_complete or merged_complete:
                completed_set.add(task.task_id)

        for task in tasks:
            if task.task_id in completed_set:
                resolved_statuses[task.task_id] = "completed"
                if task.task_id not in previous_completed:
                    newly_completed.append(task.task_id)
                continue
            if task.task_id in strategy.get("blocked_tasks", []) or task.status == "blocked":
                resolved_statuses[task.task_id] = "blocked"
                continue
            unresolved_deps = [dep for dep in task.depends_on if dep not in completed_set]
            if unresolved_deps:
                resolved_statuses[task.task_id] = "waiting"
                continue
            resolved_statuses[task.task_id] = "ready"

        selected = self._select_next_task(tasks, resolved_statuses, strategy, unresolved_merge_failures, recent_outcomes)
        state = PortalTaskState.load(self.state_path)
        state.heartbeat_at = now
        if newly_completed or not state.last_progress_at:
            state.last_progress_at = now
        state.completed_task_ids = sorted(completed_set)
        state.completed_count = len(state.completed_task_ids)
        state.ready_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "ready"]
        state.waiting_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "waiting"]
        state.blocked_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "blocked"]
        state.ready_count = len(state.ready_task_ids)
        state.waiting_count = len(state.waiting_task_ids)
        state.blocked_count = len(state.blocked_task_ids)
        state.task_count = len(tasks)
        state.task_statuses = resolved_statuses
        state.task_artifacts = task_artifacts
        state.task_validation = {task.task_id: task.validation for task in tasks if task.validation}
        state.strategy_generation = int(strategy.get("generation", 0))
        state.implementation_attempts = previous.implementation_attempts
        state.active_attempt = previous.active_attempt
        state.active_phase = previous.active_phase
        state.active_phase_started_at = previous.active_phase_started_at
        state.active_phase_detail = previous.active_phase_detail
        state.active_log_path = previous.active_log_path
        state.active_worktree_path = previous.active_worktree_path
        state.active_branch = previous.active_branch
        state.implementation_in_progress = previous.implementation_in_progress
        state.last_implementation_task_id = previous.last_implementation_task_id
        state.last_implementation_started_at = previous.last_implementation_started_at
        state.last_implementation_finished_at = previous.last_implementation_finished_at
        state.last_implementation_returncode = previous.last_implementation_returncode
        state.last_implementation_log_path = previous.last_implementation_log_path
        state.last_implementation_worktree_path = previous.last_implementation_worktree_path
        state.last_implementation_branch = previous.last_implementation_branch
        state.last_implementation_commit = previous.last_implementation_commit
        state.last_merge_started_at = previous.last_merge_started_at
        state.last_merge_finished_at = previous.last_merge_finished_at
        state.last_merge_branch = previous.last_merge_branch
        state.last_merge_commit = previous.last_merge_commit
        state.last_merge_returncode = previous.last_merge_returncode
        state.last_merge_error = previous.last_merge_error
        if previous.implementation_in_progress and live_inflight_implementation is None:
            self._clear_active_execution_state(state)
            self._record_event(
                "implementation_state_recovered",
                {
                    "task_id": previous.active_task_id or previous.last_implementation_task_id,
                    "attempt": previous.active_attempt,
                    "reason": "inflight_process_missing",
                    "worktree_path": previous.active_worktree_path,
                    "branch": previous.active_branch,
                },
            )

        if selected is not None:
            if state.active_task_id != selected.task_id:
                state.active_task_started_at = now
                state.last_progress_at = now
                self._clear_active_execution_state(state)
                self._record_event(
                    "task_selected",
                    {
                        "task_id": selected.task_id,
                        "title": selected.title,
                        "track": selected.track,
                    },
                )
            state.active_task_id = selected.task_id
            state.active_task_title = selected.title
            state.active_task_track = selected.track
            state.recommended_task_id = selected.task_id
            state.recommended_actions = self._build_recommended_actions(selected)
        else:
            state.active_task_id = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            self._clear_active_execution_state(state)
            state.recommended_task_id = ""
            state.recommended_actions = []

        state.save(self.state_path)
        for task_id in newly_completed:
            self._record_event("task_completed", {"task_id": task_id})
        implementation_result: dict[str, Any] | None = None
        if self.implement and selected is not None and resolved_statuses.get(selected.task_id) == "ready":
            unresolved_for_selected = unresolved_merge_failures.get(selected.task_id)
            if unresolved_for_selected is not None:
                implementation_result = {
                    "skipped": True,
                    "reason": "unresolved_merge_failure",
                    "task_id": selected.task_id,
                    "branch": str(unresolved_for_selected.get("branch") or ""),
                    "implementation_commit": str(unresolved_for_selected.get("implementation_commit") or ""),
                }
                self._record_event("implementation_skipped", implementation_result)
            elif self._task_has_recent_no_change_outcome(selected.task_id, recent_outcomes):
                implementation_result = {
                    "skipped": True,
                    "reason": "recent_no_change",
                    "task_id": selected.task_id,
                    "last_attempt": int((recent_outcomes.get(selected.task_id) or {}).get("attempt") or 0),
                }
                self._record_event("implementation_skipped", implementation_result)
            else:
                implementation_result = self._run_implementation(selected, state)
        self._record_event(
            "daemon_pass",
            {
                "completed_count": state.completed_count,
                "ready_count": state.ready_count,
                "waiting_count": state.waiting_count,
                "blocked_count": state.blocked_count,
                "active_task_id": state.active_task_id,
            },
        )
        return {
            "task_count": state.task_count,
            "completed_count": state.completed_count,
            "ready_count": state.ready_count,
            "waiting_count": state.waiting_count,
            "blocked_count": state.blocked_count,
            "active_task_id": state.active_task_id,
            "state_path": str(self.state_path),
            "strategy_path": str(self.strategy_path),
            "events_path": str(self.events_path),
            "implementation_result": implementation_result,
            "merge_reconciliation": merge_reconciliation,
            "event_log_repair": event_log_repair,
            "state_file_repair": state_file_repair,
        }

    def _run_implementation(self, task: PortalTask, state: PortalTaskState) -> dict[str, Any]:
        inflight = self._find_live_inflight_implementation()
        if inflight is not None:
            result = {
                "skipped": True,
                "reason": "inflight_process",
                "task_id": str(inflight.get("task_id") or task.task_id),
                "attempt": int(inflight.get("attempt") or 0),
                "worktree_path": str(inflight.get("worktree_path") or ""),
            }
            self._record_event("implementation_skipped", result)
            return result

        started_at = utc_now()
        attempt = state.implementation_attempts.get(task.task_id, 0) + 1
        lock_path = self._implementation_lock_path()
        lock_metadata = self._build_implementation_lock_metadata(task, attempt, started_at)
        lock_fd, lock_reason, existing_lock = self._try_acquire_lock(
            lock_path,
            lock_kind="implementation",
            owner_active=self._implementation_lock_owner_is_active,
        )
        if lock_fd is None:
            result = {
                "skipped": True,
                "reason": lock_reason,
                "task_id": task.task_id,
                "attempt": attempt,
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_task_id"] = str(existing_lock.get("task_id") or "")
            self._record_event("implementation_skipped", result)
            return result

        acquired_lock = True
        log_path = self.implementation_log_dir / f"{task.task_id.lower()}-attempt-{attempt}.log"
        prompt = self._build_implementation_prompt(task, attempt)
        workspace_path = self.repo_root
        command: list[str] = []
        result: dict[str, Any]
        validation_result: dict[str, Any] = {
            "attempted": False,
            "passed": True,
            "returncode": 0,
            "results": [],
            "reason": "not_run",
        }
        todo_update_result: dict[str, Any] = {}

        try:
            self._write_lock_metadata(lock_fd, lock_metadata)
            if self.use_ephemeral_worktree:
                return self._run_implementation_in_ephemeral_worktree(
                    task=task,
                    state=state,
                    attempt=attempt,
                    started_at=started_at,
                    log_path=log_path,
                    prompt=prompt,
                )
            command = self._build_implementation_command(workspace_path)
            self.implementation_log_dir.mkdir(parents=True, exist_ok=True)
            self._mark_implementation_started(
                state,
                task=task,
                attempt=attempt,
                started_at=started_at,
                log_path=log_path,
            )
            self._record_event(
                "implementation_started",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "command": command,
                    "log_path": str(log_path),
                },
            )
            with log_path.open("w", encoding="utf-8") as log_fh:
                log_fh.write(f"Task: {task.task_id} {task.title}\n")
                log_fh.write(f"Started: {started_at}\n")
                log_fh.write(f"Command: {' '.join(shlex.quote(item) for item in command)}\n\n")
                log_fh.flush()
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=workspace_path,
                    timeout=self.implementation_timeout,
                    check=False,
                )
            effective_returncode = completed.returncode
            if completed.returncode == 0:
                self._mark_active_phase(
                    state,
                    phase="validating",
                    phase_detail="; ".join(task.validation) if task.validation else "",
                )
                validation_result = self._run_validation_commands(workspace_path, task, log_path)
                if not validation_result.get("passed", False):
                    effective_returncode = int(validation_result.get("returncode") or 1)
            if effective_returncode == 0:
                todo_update_result = self._mark_task_completed_in_todo(task.task_id)
            finished_at = utc_now()
            state.implementation_attempts[task.task_id] = attempt
            state.last_implementation_task_id = task.task_id
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = effective_returncode
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": effective_returncode,
                "log_path": str(log_path),
                "validation_result": validation_result,
            }
            if todo_update_result:
                result["todo_update_result"] = todo_update_result
            self._record_event("implementation_finished", result)
            return result
        except subprocess.TimeoutExpired:
            finished_at = utc_now()
            state.implementation_attempts[task.task_id] = attempt
            state.last_implementation_task_id = task.task_id
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = 124
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": 124,
                "log_path": str(log_path),
                "error": "timeout",
            }
            self._record_event("implementation_finished", result)
            return result
        except Exception as exc:
            finished_at = utc_now()
            failed_phase = state.active_phase or "implementation_setup"
            state.implementation_attempts[task.task_id] = attempt
            state.last_implementation_task_id = task.task_id
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = 1
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            exception_result = {
                "exception_type": type(exc).__name__,
                "message": str(exc)[-4000:],
                "phase": failed_phase,
                "command": command,
            }
            result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": 1,
                "log_path": str(log_path),
                "validation_result": validation_result,
                "exception_result": exception_result,
            }
            self._record_event(
                "implementation_exception",
                {"task_id": task.task_id, "attempt": attempt, **exception_result},
            )
            self._record_event("implementation_finished", result)
            return result
        finally:
            try:
                if acquired_lock and lock_path.exists():
                    lock_path.unlink()
            except OSError:
                logger.warning("Failed to remove implementation lock %s", lock_path)

    def _mark_task_completed_in_todo(self, task_id: str) -> dict[str, Any]:
        todo_path = self.todo_path
        try:
            lines = todo_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            result = {"updated": False, "task_id": task_id, "reason": "read_failed", "error": str(exc)}
            self._record_event("todo_status_update_failed", result)
            return result

        heading = f"## {task_id}"
        in_task = False
        status_index: int | None = None
        for index, line in enumerate(lines):
            if line.startswith(self.task_header_prefix):
                if in_task:
                    break
                in_task = line.startswith(heading)
                continue
            if in_task and line.startswith("- Status:"):
                status_index = index
                break

        if status_index is None:
            result = {"updated": False, "task_id": task_id, "reason": "status_line_missing"}
            self._record_event("todo_status_update_failed", result)
            return result

        current = lines[status_index].split(":", 1)[1].strip()
        if normalize_status(current) == "completed":
            return {"updated": False, "task_id": task_id, "reason": "already_completed"}

        newline = "\n" if lines[status_index].endswith("\n") else ""
        lines[status_index] = "- Status: completed" + newline
        tmp_path = todo_path.with_name(f".{todo_path.name}.tmp")
        try:
            tmp_path.write_text("".join(lines), encoding="utf-8")
            os.replace(tmp_path, todo_path)
        except OSError as exc:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            result = {"updated": False, "task_id": task_id, "reason": "write_failed", "error": str(exc)}
            self._record_event("todo_status_update_failed", result)
            return result

        commit_result = self._commit_generated_file_update(
            todo_path,
            task_id=task_id,
            subject=f"{task_id}: mark todo completed",
        )
        result = {"updated": True, "task_id": task_id, "path": str(todo_path)}
        if commit_result:
            result["commit_result"] = commit_result
        self._record_event("todo_status_updated", result)
        return result

    def _commit_generated_file_update(self, path: Path, *, task_id: str, subject: str) -> dict[str, Any]:
        """Commit a daemon-owned generated file and any parent gitlink updates."""

        repo = self._git_toplevel_for_path(path.parent)
        if repo is None:
            return {"committed": False, "reason": "not_in_git_repo", "path": str(path)}
        relative = self._relative_to_repo(repo, path)
        if not relative:
            return {"committed": False, "reason": "path_outside_repo", "path": str(path), "repo": str(repo)}

        result = self._commit_specific_path(repo, relative, subject=subject)
        parent_results: list[dict[str, Any]] = []
        if result.get("committed"):
            parent_results = self._commit_parent_gitlink_updates(repo, task_id=task_id)
        if parent_results:
            result["parent_gitlink_commits"] = parent_results
        return result

    def _commit_parent_gitlink_updates(self, child_repo: Path, *, task_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        current = child_repo.resolve()
        repo_root = self.repo_root.resolve()
        while current != repo_root:
            parent = self._parent_git_toplevel_for_repo(current)
            if parent is None:
                break
            relative = self._relative_to_repo(parent, current)
            if not relative:
                break
            result = self._commit_specific_path(
                parent,
                relative,
                subject=f"{task_id}: update generated submodule pointer",
            )
            results.append(result)
            current = parent.resolve()
        return results

    def _commit_specific_path(self, repo: Path, relative: str, *, subject: str) -> dict[str, Any]:
        if not self._repo_relative_path_safe(relative):
            return {"committed": False, "reason": "unsafe_path", "repo": str(repo), "path": relative}
        unmerged = self._unmerged_worktree_paths(repo)
        if unmerged and relative not in unmerged:
            return {
                "committed": False,
                "reason": "repo_has_unrelated_unmerged_paths",
                "repo": str(repo),
                "path": relative,
                "unmerged_paths": sorted(unmerged),
            }
        status = self._path_status(repo, relative)
        if not status:
            return {"committed": False, "reason": "no_changes", "repo": str(repo), "path": relative}
        add = subprocess.run(
            ["git", "add", "--", relative],
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
                "path": relative,
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
            }
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 0:
            return {"committed": False, "reason": "no_staged_changes", "repo": str(repo), "path": relative}
        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
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
        commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
        return {
            "committed": True,
            "repo": str(repo),
            "path": relative,
            "commit": commit_ref,
            "status": status,
        }

    def _git_toplevel_for_path(self, cwd: Path) -> Path | None:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return Path(result.stdout.strip()).resolve()

    def _parent_git_toplevel_for_repo(self, repo: Path) -> Path | None:
        parent_dir = repo.resolve().parent
        parent = self._git_toplevel_for_path(parent_dir)
        if parent is None or parent.resolve() == repo.resolve():
            return None
        try:
            repo.resolve().relative_to(parent.resolve())
        except ValueError:
            return None
        return parent

    @staticmethod
    def _relative_to_repo(repo: Path, path: Path) -> str:
        try:
            return path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            return ""

    def _path_status(self, repo: Path, relative: str) -> str:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _unmerged_worktree_paths(self, repo: Path) -> set[str]:
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

    def _run_implementation_in_ephemeral_worktree(
        self,
        *,
        task: PortalTask,
        state: PortalTaskState,
        attempt: int,
        started_at: str,
        log_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        self.implementation_log_dir.mkdir(parents=True, exist_ok=True)
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        safe_task_id = task.task_id.lower().replace("/", "-")
        attempt_stamp = int(time.time())
        worktree_path = self.worktree_root / f"{safe_task_id}-attempt-{attempt}-{attempt_stamp}"
        branch_name = f"implementation/{safe_task_id}-attempt-{attempt}-{attempt_stamp}"
        baseline_ref = ""
        implementation_commit = ""
        merge_result: dict[str, Any] = {"merged": False, "reason": "not_attempted"}
        validation_result: dict[str, Any] = {
            "attempted": False,
            "passed": True,
            "returncode": 0,
            "results": [],
            "reason": "not_run",
        }
        cleanup_result: dict[str, Any] = {"cleaned": False, "reason": "not_attempted"}
        command: list[str] = []
        returncode = 1
        commit_result: dict[str, Any] = {"committed": False}
        failed_preservation_result: dict[str, Any] = {}
        todo_update_result: dict[str, Any] = {}
        exception_result: dict[str, Any] = {}

        try:
            baseline_ref = self._create_seeded_worktree(worktree_path, branch_name, task=task)
            command = self._build_implementation_command(worktree_path)
            self._mark_implementation_started(
                state,
                task=task,
                attempt=attempt,
                started_at=started_at,
                log_path=log_path,
                worktree_path=worktree_path,
                branch_name=branch_name,
            )
            self._record_event(
                "implementation_started",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "command": command,
                    "log_path": str(log_path),
                    "worktree_path": str(worktree_path),
                    "branch": branch_name,
                    "baseline_ref": baseline_ref,
                },
            )
            with log_path.open("w", encoding="utf-8") as log_fh:
                log_fh.write(f"Task: {task.task_id} {task.title}\n")
                log_fh.write(f"Started: {started_at}\n")
                log_fh.write(f"Workspace: {worktree_path}\n")
                log_fh.write(f"Branch: {branch_name}\n")
                log_fh.write(f"Baseline: {baseline_ref}\n")
                log_fh.write(f"Command: {' '.join(shlex.quote(item) for item in command)}\n\n")
                log_fh.flush()
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=worktree_path,
                    timeout=self.implementation_timeout,
                    check=False,
            )
            returncode = completed.returncode
            if returncode == 0:
                self._mark_active_phase(
                    state,
                    phase="validating",
                    phase_detail="; ".join(task.validation) if task.validation else "",
                    worktree_path=worktree_path,
                    branch_name=branch_name,
                )
                self._prepare_worktree_for_validation(worktree_path, task=task, branch_name=branch_name)
                validation_result = self._run_validation_commands(worktree_path, task, log_path)
                if validation_result.get("passed", False):
                    commit_result = self._commit_worktree_changes(worktree_path, task, attempt)
                    implementation_commit = str(commit_result.get("commit", ""))
                    if implementation_commit:
                        self._mark_active_phase(
                            state,
                            phase="merging",
                            phase_detail=branch_name,
                            worktree_path=worktree_path,
                            branch_name=branch_name,
                        )
                        merge_result = self._merge_branch_to_main(
                            branch_name,
                            task,
                            attempt,
                            baseline_ref=baseline_ref,
                        )
                        if merge_result.get("merged"):
                            cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
                        else:
                            returncode = int(merge_result.get("returncode") or 1)
                    elif commit_result.get("reason") == "no_changes":
                        cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
                else:
                    returncode = int(validation_result.get("returncode") or 1)
                    failed_preservation_result = self._preserve_failed_validation_worktree(
                        worktree_path,
                        branch_name,
                        task,
                        attempt,
                        validation_result,
                    )
                    commit_result = dict(failed_preservation_result.get("commit_result") or commit_result)
                    implementation_commit = str(commit_result.get("commit", ""))
                    cleanup_result = dict(failed_preservation_result.get("cleanup_result") or cleanup_result)
        except subprocess.TimeoutExpired:
            returncode = 124
            self._record_event(
                "implementation_timeout",
                {"task_id": task.task_id, "attempt": attempt, "worktree_path": str(worktree_path)},
            )
        except Exception as exc:
            returncode = 1
            exception_result = {
                "exception_type": type(exc).__name__,
                "message": str(exc)[-4000:],
                "worktree_path": str(worktree_path),
                "branch": branch_name,
                "phase": state.active_phase or "worktree_setup",
            }
            self._record_event(
                "implementation_exception",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    **exception_result,
                },
            )
        finished_at = utc_now()
        state.implementation_attempts[task.task_id] = attempt
        state.last_implementation_task_id = task.task_id
        state.last_implementation_started_at = started_at
        state.last_implementation_finished_at = finished_at
        state.last_implementation_returncode = returncode
        state.last_implementation_log_path = str(log_path)
        state.last_implementation_worktree_path = str(worktree_path)
        state.last_implementation_branch = branch_name
        state.last_implementation_commit = implementation_commit
        state.last_merge_started_at = str(merge_result.get("started_at") or "")
        state.last_merge_finished_at = str(merge_result.get("finished_at") or "")
        state.last_merge_branch = branch_name if merge_result.get("merged") or merge_result.get("attempted") else ""
        state.last_merge_commit = str(merge_result.get("merge_commit") or "")
        state.last_merge_returncode = (
            int(merge_result["returncode"]) if merge_result.get("returncode") is not None else None
        )
        state.last_merge_error = str(merge_result.get("stderr") or merge_result.get("reason") or "")
        if returncode == 0:
            todo_update_result = self._mark_task_completed_in_todo(task.task_id)
        self._mark_implementation_finished(state, finished_at=finished_at)
        state.save(self.state_path)
        result = {
            "task_id": task.task_id,
            "attempt": attempt,
            "returncode": returncode,
            "log_path": str(log_path),
            "worktree_path": str(worktree_path),
            "branch": branch_name,
            "baseline_ref": baseline_ref,
            "commit_result": commit_result,
            "implementation_commit": implementation_commit,
            "merge_result": merge_result,
            "validation_result": validation_result,
            "cleanup_result": cleanup_result,
            "failed_preservation_result": failed_preservation_result,
        }
        if exception_result:
            result["exception_result"] = exception_result
        if todo_update_result:
            result["todo_update_result"] = todo_update_result
        self._record_event("implementation_finished", result)
        return result

    def _clear_active_execution_state(self, state: PortalTaskState) -> None:
        state.active_attempt = 0
        state.active_phase = ""
        state.active_phase_started_at = ""
        state.active_phase_detail = ""
        state.active_log_path = ""
        state.active_worktree_path = ""
        state.active_branch = ""
        state.implementation_in_progress = False

    def _mark_implementation_started(
        self,
        state: PortalTaskState,
        *,
        task: PortalTask,
        attempt: int,
        started_at: str,
        log_path: Path,
        worktree_path: Path | None = None,
        branch_name: str = "",
    ) -> None:
        state.active_task_id = task.task_id
        state.active_task_title = task.title
        state.active_task_track = task.track
        if not state.active_task_started_at:
            state.active_task_started_at = started_at
        state.active_attempt = attempt
        state.active_phase = "implementing"
        state.active_phase_started_at = started_at
        state.active_phase_detail = ""
        state.active_log_path = str(log_path)
        state.active_worktree_path = str(worktree_path) if worktree_path is not None else ""
        state.active_branch = branch_name
        state.implementation_in_progress = True
        state.last_implementation_task_id = task.task_id
        state.last_implementation_started_at = started_at
        state.last_implementation_finished_at = ""
        state.last_implementation_returncode = None
        state.last_implementation_log_path = str(log_path)
        state.last_implementation_worktree_path = str(worktree_path) if worktree_path is not None else ""
        state.last_implementation_branch = branch_name
        state.last_implementation_commit = ""
        state.heartbeat_at = started_at
        state.last_progress_at = started_at
        state.save(self.state_path)

    def _mark_active_phase(
        self,
        state: PortalTaskState,
        *,
        phase: str,
        phase_detail: str = "",
        worktree_path: Path | None = None,
        branch_name: str | None = None,
        at: str | None = None,
    ) -> None:
        timestamp = at or utc_now()
        if state.active_phase != phase:
            state.active_phase_started_at = timestamp
        elif not state.active_phase_started_at:
            state.active_phase_started_at = timestamp
        state.active_phase = phase
        state.active_phase_detail = phase_detail
        if worktree_path is not None:
            state.active_worktree_path = str(worktree_path)
        if branch_name is not None:
            state.active_branch = branch_name
        state.implementation_in_progress = True
        state.heartbeat_at = timestamp
        state.last_progress_at = timestamp
        state.save(self.state_path)

    def _mark_implementation_finished(self, state: PortalTaskState, *, finished_at: str) -> None:
        state.implementation_in_progress = False
        state.heartbeat_at = finished_at
        state.last_progress_at = finished_at
        self._clear_active_execution_state(state)

    def _create_seeded_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        *,
        task: PortalTask | None = None,
    ) -> str:
        self._run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), self._main_branch_name()],
            cwd=self.repo_root,
        )
        baseline_ref = self._run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
        self._initialize_worktree_submodules(worktree_path, branch_name=branch_name)
        self._link_shared_worktree_paths(worktree_path)
        self._seed_untracked_worktree_context(worktree_path, task=task, overwrite_existing=True)
        return baseline_ref

    def _initialize_worktree_submodules(self, worktree_path: Path, *, branch_name: str = "") -> None:
        for relative in self.worktree_submodule_paths:
            if self._create_local_submodule_worktree(worktree_path, relative, branch_name=branch_name):
                target = worktree_path / relative
                if self._is_git_worktree(target):
                    self._initialize_nested_worktree_submodules(
                        target,
                        branch_name=branch_name,
                        parent_relative=relative,
                    )
                continue
            if self._worktree_declares_submodule(worktree_path, relative):
                self._run_git(["submodule", "update", "--init", "--recursive", "--", relative], cwd=worktree_path)
                target = worktree_path / relative
                if self._is_git_worktree(target):
                    self._initialize_nested_worktree_submodules(
                        target,
                        branch_name=branch_name,
                        parent_relative=relative,
                    )

    def _initialize_nested_worktree_submodules(
        self,
        worktree_path: Path,
        *,
        branch_name: str,
        parent_relative: str,
    ) -> None:
        for relative in self._declared_submodule_paths(worktree_path):
            full_relative = f"{parent_relative.rstrip('/')}/{relative}"
            if self._create_local_submodule_worktree(
                worktree_path,
                relative,
                branch_name=branch_name,
                source_relative=full_relative,
            ):
                target = worktree_path / relative
                if self._is_git_worktree(target):
                    self._initialize_nested_worktree_submodules(
                        target,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                    )
                continue

    def _create_local_submodule_worktree(
        self,
        worktree_path: Path,
        relative: str,
        *,
        branch_name: str = "",
        source_relative: str | None = None,
    ) -> bool:
        source_key = source_relative or relative
        source = (self.repo_root / source_key).resolve()
        if not source.exists() or not self._is_git_worktree(source):
            return False
        base_ref = self._submodule_gitlink_ref(worktree_path, relative) or "HEAD"
        target = worktree_path / relative
        if self._is_git_worktree(target) and not target.is_symlink():
            if branch_name:
                expected_branch = self._submodule_worktree_branch_name(branch_name, source_key)
                current_branch = self._git_current_branch(target)
                if current_branch and current_branch != expected_branch:
                    return False
            return True
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if branch_name:
            submodule_branch = self._submodule_worktree_branch_name(branch_name, source_key)
            if self._git_ref_exists_in_repo(source, submodule_branch):
                self._run_git(["worktree", "add", str(target), submodule_branch], cwd=source)
                return True
            self._run_git(["worktree", "add", "-b", submodule_branch, str(target), base_ref], cwd=source)
            return True
        self._run_git(["worktree", "add", "--detach", str(target), base_ref], cwd=source)
        return True

    def _submodule_gitlink_ref(self, worktree_path: Path, relative: str) -> str:
        if not self._repo_relative_path_safe(relative):
            return ""
        result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _submodule_worktree_branch_name(branch_name: str, relative: str) -> str:
        safe_relative = relative.strip("/").replace("/", "-")
        return f"{branch_name}-submodule-{safe_relative}"

    def _is_git_worktree(self, path: Path) -> bool:
        if not path.exists() or path.is_symlink():
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        try:
            return Path(result.stdout.strip()).resolve() == path.resolve()
        except OSError:
            return False

    def _git_current_branch(self, cwd: Path) -> str:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _git_ref_exists_in_repo(self, cwd: Path, ref: str) -> bool:
        if not ref:
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _worktree_declares_submodule(self, worktree_path: Path, relative: str) -> bool:
        return relative in self._declared_submodule_paths(worktree_path)

    def _declared_submodule_paths(self, worktree_path: Path) -> list[str]:
        gitmodules = worktree_path / ".gitmodules"
        if not gitmodules.exists():
            return []
        result = subprocess.run(
            ["git", "config", "--file", str(gitmodules), "--get-regexp", r"^submodule\..*\.path$"],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        paths: list[str] = []
        for line in result.stdout.splitlines():
            path = line.split(maxsplit=1)[-1].strip()
            if path and self._repo_relative_path_safe(path):
                paths.append(path)
        return paths

    def _link_shared_worktree_paths(self, worktree_path: Path) -> None:
        for relative in SHARED_WORKTREE_PATHS:
            source = (self.repo_root / relative).resolve()
            if not source.exists():
                continue
            target = worktree_path / relative
            if target.is_symlink():
                if target.resolve() == source:
                    continue
                target.unlink()
            elif target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())

    def _prepare_worktree_for_validation(
        self,
        worktree_path: Path,
        *,
        task: PortalTask | None = None,
        branch_name: str = "",
    ) -> None:
        self._initialize_worktree_submodules(worktree_path, branch_name=branch_name)
        self._link_shared_worktree_paths(worktree_path)
        self._seed_untracked_worktree_context(worktree_path, task=task)

    def _seed_untracked_worktree_context(
        self,
        worktree_path: Path,
        *,
        task: PortalTask | None = None,
        overwrite_existing: bool = False,
    ) -> list[str]:
        """Copy relevant dirty source context into an ephemeral worktree."""

        seeded: list[str] = []
        for relative in self._untracked_worktree_context_paths():
            if not self._untracked_context_path_allowed(relative):
                continue
            source = self.repo_root / relative
            if not source.exists() or source.is_dir():
                continue
            target = worktree_path / relative
            if target.exists() or target.is_symlink():
                if not overwrite_existing:
                    continue
                if target.is_dir():
                    continue
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            else:
                shutil.copy2(source, target)
            seeded.append(relative)

        if seeded:
            payload: dict[str, Any] = {
                "worktree_path": str(worktree_path),
                "seeded_paths": seeded,
                "seeded_count": len(seeded),
            }
            if task is not None:
                payload["task_id"] = task.task_id
            self._record_event("worktree_context_seeded", payload)
        return seeded

    def _untracked_worktree_context_paths(self) -> list[str]:
        candidates: set[str] = set()
        commands = (
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            ["git", "diff", "--name-only", "-z"],
            ["git", "diff", "--cached", "--name-only", "-z"],
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                continue
            paths = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
            candidates.update(path for path in paths if path)
        return sorted(candidates)

    def _untracked_context_path_allowed(self, relative: str) -> bool:
        if not relative or relative.startswith("/") or "\0" in relative:
            return False
        if ".." in Path(relative).parts:
            return False
        if any(self._path_matches_prefix(relative, prefix) for prefix in EPHEMERAL_WORKTREE_PATHS):
            return False
        if any(self._path_matches_prefix(relative, prefix) for prefix in self.worktree_submodule_paths):
            return False
        return any(self._path_matches_prefix(relative, prefix) for prefix in UNTRACKED_WORKTREE_CONTEXT_PREFIXES)

    @staticmethod
    def _path_matches_prefix(relative: str, prefix: str) -> bool:
        normalized = prefix.rstrip("/")
        return relative == normalized or relative.startswith(f"{normalized}/")

    def _commit_worktree_changes(self, worktree_path: Path, task: PortalTask, attempt: int) -> dict[str, Any]:
        submodule_results = self._commit_worktree_submodule_changes(worktree_path, task, attempt)
        self._restore_ephemeral_worktree_paths_for_commit(worktree_path)
        self._restore_uncommitted_submodule_pointers(worktree_path, submodule_results)
        self._run_git(["add", "-A"], cwd=worktree_path)
        self._remove_generated_paths_from_index(worktree_path)
        self._restore_uncommitted_submodule_pointers(worktree_path, submodule_results)
        status = self._run_git(["status", "--porcelain"], cwd=worktree_path).stdout.strip()
        staged_status = self._staged_worktree_status(worktree_path)
        if not staged_status:
            result: dict[str, Any] = {"committed": False, "reason": "no_changes"}
            if status:
                result["status"] = status
            if submodule_results:
                result["submodule_results"] = submodule_results
            return result
        self._run_git(
            [
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "commit",
                "-m",
                f"{task.task_id}: {task.title or 'implementation attempt'}",
                "-m",
                f"Attempt: {attempt}",
            ],
            cwd=worktree_path,
        )
        commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
        result = {
            "committed": True,
            "commit": commit_ref,
            "status": status,
        }
        if submodule_results:
            result["submodule_results"] = submodule_results
        return result

    def _commit_worktree_submodule_changes(
        self,
        worktree_path: Path,
        task: PortalTask,
        attempt: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative in self.worktree_submodule_paths:
            target = worktree_path / relative
            if not self._is_git_worktree(target):
                continue
            nested_results = self._commit_nested_submodule_changes(
                target,
                task,
                attempt,
                parent_relative=relative,
            )
            self._restore_ephemeral_worktree_paths_for_commit(target)
            self._run_git(["add", "-A"], cwd=target)
            self._remove_generated_paths_from_index(target)
            status = self._run_git(["status", "--porcelain"], cwd=target).stdout.strip()
            staged_status = self._staged_worktree_status(target)
            if not staged_status:
                result: dict[str, Any] = {"path": relative, "committed": False, "reason": "no_changes"}
                if status:
                    result["status"] = status
                if nested_results:
                    result["nested_submodule_results"] = nested_results
                results.append(result)
                continue
            self._run_git(
                [
                    "-c",
                    "user.name=Implementation Daemon",
                    "-c",
                    "user.email=implementation-daemon@example.invalid",
                    "commit",
                    "-m",
                    f"{task.task_id}: {task.title or 'implementation attempt'}",
                    "-m",
                    f"Attempt: {attempt}",
                    "-m",
                    f"Submodule: {relative}",
                ],
                cwd=target,
            )
            commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=target).stdout.strip()
            result = {"path": relative, "committed": True, "commit": commit_ref, "status": status}
            if nested_results:
                result["nested_submodule_results"] = nested_results
            results.append(result)
        return results

    def _commit_nested_submodule_changes(
        self,
        worktree_path: Path,
        task: PortalTask,
        attempt: int,
        *,
        parent_relative: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative in self._declared_submodule_paths(worktree_path):
            full_relative = f"{parent_relative.rstrip('/')}/{relative}"
            target = worktree_path / relative
            if not self._is_git_worktree(target):
                continue
            nested_results = self._commit_nested_submodule_changes(
                target,
                task,
                attempt,
                parent_relative=full_relative,
            )
            self._restore_ephemeral_worktree_paths_for_commit(target)
            self._run_git(["add", "-A"], cwd=target)
            self._remove_generated_paths_from_index(target)
            status = self._run_git(["status", "--porcelain"], cwd=target).stdout.strip()
            staged_status = self._staged_worktree_status(target)
            if not staged_status:
                result: dict[str, Any] = {
                    "path": full_relative,
                    "committed": False,
                    "reason": "no_changes",
                }
                if status:
                    result["status"] = status
                if nested_results:
                    result["nested_submodule_results"] = nested_results
                results.append(result)
                continue
            self._run_git(
                [
                    "-c",
                    "user.name=Implementation Daemon",
                    "-c",
                    "user.email=implementation-daemon@example.invalid",
                    "commit",
                    "-m",
                    f"{task.task_id}: {task.title or 'implementation attempt'}",
                    "-m",
                    f"Attempt: {attempt}",
                    "-m",
                    f"Submodule: {full_relative}",
                ],
                cwd=target,
            )
            commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=target).stdout.strip()
            result = {"path": full_relative, "committed": True, "commit": commit_ref, "status": status}
            if nested_results:
                result["nested_submodule_results"] = nested_results
            results.append(result)
        return results

    def _restore_uncommitted_submodule_pointers(
        self,
        worktree_path: Path,
        submodule_results: list[dict[str, Any]],
    ) -> None:
        for result in submodule_results:
            if result.get("committed", False):
                continue
            relative = str(result.get("path") or "")
            if relative not in self.worktree_submodule_paths or not self._repo_relative_path_safe(relative):
                continue
            subprocess.run(
                ["git", "restore", "--source=HEAD", "--staged", "--worktree", "--", relative],
                cwd=worktree_path,
                text=True,
                capture_output=True,
                check=False,
            )

    def _preserve_failed_validation_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        started_at = utc_now()
        commit_result = self._commit_worktree_changes(worktree_path, task, attempt)
        rescue_branch = ""
        implementation_commit = str(commit_result.get("commit", ""))
        if implementation_commit:
            rescue_branch = self._failed_validation_rescue_branch_name(branch_name)
            self._run_git(["branch", "-f", rescue_branch, implementation_commit], cwd=self.repo_root)
        cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
        result = {
            "task_id": task.task_id,
            "attempt": attempt,
            "branch": branch_name,
            "worktree_path": str(worktree_path),
            "started_at": started_at,
            "finished_at": utc_now(),
            "preserved": bool(implementation_commit),
            "rescue_branch": rescue_branch,
            "implementation_commit": implementation_commit,
            "commit_result": commit_result,
            "cleanup_result": cleanup_result,
            "validation_result": validation_result,
        }
        self._record_event("failed_validation_worktree_preserved", result)
        return result

    @staticmethod
    def _failed_validation_rescue_branch_name(branch_name: str) -> str:
        safe_name = branch_name.removeprefix("implementation/").strip("/").replace(" ", "-")
        return f"rescue/{safe_name or 'implementation-attempt'}-failed-validation"

    def _restore_ephemeral_worktree_paths_for_commit(self, worktree_path: Path) -> None:
        for relative in EPHEMERAL_WORKTREE_PATHS:
            self._restore_or_remove_generated_path_for_commit(worktree_path, relative)
        for relative in sorted(self._dirty_worktree_paths(worktree_path)):
            if self._path_is_generated_worktree_artifact(relative):
                self._restore_or_remove_generated_path_for_commit(worktree_path, relative)

    def _remove_generated_paths_from_index(self, worktree_path: Path) -> None:
        for relative in self._staged_worktree_paths(worktree_path):
            if self._path_is_generated_worktree_artifact(relative):
                self._restore_or_remove_generated_path_for_commit(worktree_path, relative)

    def _restore_or_remove_generated_path_for_commit(self, worktree_path: Path, relative: str) -> None:
        if not self._repo_relative_path_safe(relative):
            return
        target = worktree_path / relative
        if relative in self.worktree_submodule_paths and target.is_symlink():
            target.unlink()
        if self._path_tracked_in_head(worktree_path, relative) or self._path_tracked_in_repo(worktree_path, relative):
            restore = subprocess.run(
                ["git", "restore", "--source=HEAD", "--staged", "--worktree", "--", relative],
                cwd=worktree_path,
                text=True,
                capture_output=True,
                check=False,
            )
            if restore.returncode == 0:
                return
        subprocess.run(
            ["git", "restore", "--staged", "--", relative],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    def _path_is_generated_worktree_artifact(self, relative: str) -> bool:
        if not self._repo_relative_path_safe(relative):
            return False
        normalized = relative.strip("/")
        parts = Path(normalized).parts
        if any(part in GENERATED_WORKTREE_DIR_NAMES for part in parts):
            return True
        if normalized.endswith(GENERATED_WORKTREE_SUFFIXES):
            return True
        return any(self._path_matches_prefix(normalized, prefix) for prefix in EPHEMERAL_WORKTREE_PATHS)

    def _staged_worktree_paths(self, cwd: Path) -> list[str]:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        paths = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        return [path for path in paths if path]

    def _staged_worktree_status(self, cwd: Path) -> str:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-status"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _path_tracked_in_repo(self, cwd: Path, relative: str) -> bool:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _path_tracked_in_head(self, cwd: Path, relative: str) -> bool:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", relative],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        return any(line == relative or line.startswith(f"{relative.rstrip('/')}/") for line in result.stdout.splitlines())

    def _run_validation_commands(self, workspace_path: Path, task: PortalTask, log_path: Path) -> dict[str, Any]:
        if not task.validation:
            return {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": "no_commands",
            }

        results: list[dict[str, Any]] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write("\nValidation:\n")
            for command in task.validation:
                started_at = utc_now()
                log_fh.write(f"$ {command}\n")
                log_fh.flush()
                try:
                    completed = subprocess.run(
                        ["/bin/bash", "-lc", command],
                        cwd=workspace_path,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        timeout=self.implementation_timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    results.append(
                        {
                            "command": command,
                            "started_at": started_at,
                            "finished_at": utc_now(),
                            "returncode": 124,
                            "timed_out": True,
                        }
                    )
                    log_fh.write(f"[validation timed out] timeout={self.implementation_timeout}\n")
                    log_fh.flush()
                    return {
                        "attempted": True,
                        "passed": False,
                        "returncode": 124,
                        "results": results,
                        "failed_command": command,
                        "error": "timeout",
                    }
                result = {
                    "command": command,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "returncode": completed.returncode,
                }
                results.append(result)
                if completed.returncode != 0:
                    log_fh.write(f"[validation failed] returncode={completed.returncode}\n")
                    log_fh.flush()
                    return {
                        "attempted": True,
                        "passed": False,
                        "returncode": completed.returncode,
                        "results": results,
                        "failed_command": command,
                    }
            log_fh.write("[validation passed]\n")
            log_fh.flush()
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": results,
        }

    def _main_branch_name(self) -> str:
        for candidate in ("main", "master"):
            if self._git_ref_exists(candidate):
                return candidate
        current_branch = self._git_current_branch(self.repo_root)
        return current_branch or "HEAD"

    def _main_merge_worktree_root(self) -> Path:
        return self.worktree_root / ".main-merge-worktrees"

    @staticmethod
    def _safe_ref_path_fragment(ref: str) -> str:
        safe = "".join(character if character.isalnum() or character in "-._" else "-" for character in ref)
        return safe.strip("-") or "main"

    def _git_worktree_entries(self) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        entries: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current:
                    entries.append(current)
                current = {"worktree": line.split(" ", 1)[1]}
            elif line.startswith("branch "):
                branch = line.split(" ", 1)[1]
                current["branch"] = branch.removeprefix("refs/heads/")
        if current:
            entries.append(current)
        return entries

    def _branch_checked_out_worktree_paths(self, branch_name: str) -> list[Path]:
        paths: list[Path] = []
        for entry in self._git_worktree_entries():
            if entry.get("branch") != branch_name:
                continue
            worktree = entry.get("worktree")
            if worktree:
                paths.append(Path(worktree))
        return paths

    @staticmethod
    def _path_is_under(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            return False
        return True

    def _prepare_main_merge_workspace(self, target_branch: str, branch_name: str) -> dict[str, Any]:
        if self._git_current_branch(self.repo_root) == target_branch:
            return {
                "available": True,
                "path": str(self.repo_root),
                "ephemeral": False,
                "target_branch": target_branch,
            }

        merge_root = self._main_merge_worktree_root()
        checked_out_paths = self._branch_checked_out_worktree_paths(target_branch)
        for checked_out_path in checked_out_paths:
            if checked_out_path.resolve() == self.repo_root.resolve():
                return {
                    "available": True,
                    "path": str(self.repo_root),
                    "ephemeral": False,
                    "target_branch": target_branch,
                }
            if self._path_is_under(checked_out_path, merge_root):
                dirty_paths = sorted(self._dirty_worktree_paths(checked_out_path))
                if dirty_paths:
                    return {
                        "available": False,
                        "reason": "main_merge_worktree_dirty",
                        "target_branch": target_branch,
                        "worktree_path": str(checked_out_path),
                        "dirty_paths": dirty_paths,
                    }
                self._run_git(["worktree", "remove", "--force", str(checked_out_path)], cwd=self.repo_root)
                continue
            return {
                "available": False,
                "reason": "main_branch_checked_out_elsewhere",
                "target_branch": target_branch,
                "worktree_path": str(checked_out_path),
            }

        merge_root.mkdir(parents=True, exist_ok=True)
        safe_target = self._safe_ref_path_fragment(target_branch)
        safe_branch = self._safe_ref_path_fragment(branch_name)
        workspace = merge_root / f"{safe_target}-{safe_branch}-{os.getpid()}-{int(time.time())}"
        self._run_git(["worktree", "add", str(workspace), target_branch], cwd=self.repo_root)
        return {
            "available": True,
            "path": str(workspace),
            "ephemeral": True,
            "target_branch": target_branch,
        }

    def _cleanup_main_merge_workspace(self, workspace_path: Path, *, ephemeral: bool) -> dict[str, Any]:
        if not ephemeral:
            return {"cleaned": True, "removed": False, "worktree_path": str(workspace_path)}
        if not workspace_path.exists():
            return {"cleaned": True, "removed": False, "worktree_path": str(workspace_path)}
        remove = subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace_path)],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "cleaned": remove.returncode == 0,
            "removed": remove.returncode == 0,
            "worktree_path": str(workspace_path),
            "returncode": remove.returncode,
            "stdout": remove.stdout[-4000:],
            "stderr": remove.stderr[-4000:],
        }

    def _merge_branch_to_main(
        self,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        *,
        baseline_ref: str = "",
    ) -> dict[str, Any]:
        started_at = utc_now()
        target_branch = self._main_branch_name()
        if baseline_ref and not self._git_ref_is_ancestor(baseline_ref, target_branch):
            result = {
                "attempted": False,
                "merged": False,
                "returncode": 2,
                "branch": branch_name,
                "target_branch": target_branch,
                "baseline_ref": baseline_ref,
                "started_at": started_at,
                "finished_at": utc_now(),
                "merge_commit": "",
                "stdout": "",
                "stderr": "",
                "reason": "baseline_not_ancestor_of_target",
                "identical_untracked_paths": [],
                "submodule_merge_results": [],
            }
            self._record_event("merge_finished", result)
            return result
        merge_lock = self._repo_merge_lock_path()
        lock_metadata = self._build_merge_lock_metadata(branch_name, task, attempt, started_at)
        lock_fd, lock_reason, existing_lock = self._try_acquire_lock(
            merge_lock,
            lock_kind="merge",
            owner_active=self._merge_lock_owner_is_active,
        )
        if lock_fd is None:
            result = {
                "attempted": False,
                "merged": False,
                "reason": lock_reason,
                "branch": branch_name,
                "target_branch": target_branch,
                "started_at": started_at,
                "identical_untracked_paths": [],
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_branch"] = str(existing_lock.get("branch") or "")
            return result

        merge_workspace: Path | None = None
        merge_workspace_ephemeral = False
        removed_untracked: dict[str, bytes] = {}
        try:
            self._write_lock_metadata(lock_fd, lock_metadata)
            workspace_result = self._prepare_main_merge_workspace(target_branch, branch_name)
            llm_workspace_resolver: dict[str, Any] = {}
            if not workspace_result.get("available", False):
                workspace_reason = str(workspace_result.get("reason") or "main_merge_workspace_unavailable")
                workspace_path = str(workspace_result.get("worktree_path") or "")
                if workspace_reason == "main_merge_worktree_dirty" and workspace_path:
                    llm_workspace_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                        workspace=Path(workspace_path),
                        task=task,
                        attempt=attempt,
                        branch_name=branch_name,
                        target_branch=target_branch,
                        merge_command=[],
                        merge_stdout="",
                        merge_stderr="",
                        reason=workspace_reason,
                        dirty_paths=[str(item) for item in workspace_result.get("dirty_paths", [])],
                    )
                    if llm_workspace_resolver.get("applied", False):
                        workspace_result = self._prepare_main_merge_workspace(target_branch, branch_name)
                    if workspace_result.get("available", False):
                        self._record_event(
                            "main_merge_workspace_blocker_resolved",
                            {
                                "task_id": task.task_id,
                                "attempt": attempt,
                                "branch": branch_name,
                                "target_branch": target_branch,
                                "llm_merge_resolver": llm_workspace_resolver,
                            },
                        )
            if not workspace_result.get("available", False):
                result = {
                    "attempted": True,
                    "merged": False,
                    "returncode": 2,
                    "branch": branch_name,
                    "target_branch": target_branch,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "merge_commit": "",
                    "stdout": "",
                    "stderr": "",
                    "reason": str(workspace_result.get("reason") or "main_merge_workspace_unavailable"),
                    "dirty_paths": workspace_result.get("dirty_paths", []),
                    "main_worktree_path": str(workspace_result.get("worktree_path") or ""),
                    "identical_untracked_paths": [],
                    "submodule_merge_results": [],
                }
                if llm_workspace_resolver:
                    result["llm_merge_resolver"] = llm_workspace_resolver
                self._record_event("merge_finished", result)
                return result

            merge_workspace = Path(str(workspace_result["path"]))
            merge_workspace_ephemeral = bool(workspace_result.get("ephemeral", False))
            resolved_add_add_conflicts = self._resolve_generated_add_add_conflicts(cwd=merge_workspace)
            identical_untracked_paths = self._identical_untracked_merge_paths(branch_name, cwd=merge_workspace)
            dirty_overlap = self._dirty_merge_conflict_paths(
                branch_name,
                cwd=merge_workspace,
                ignore_paths=set(identical_untracked_paths),
            )
            if dirty_overlap:
                llm_merge_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                    workspace=merge_workspace,
                    task=task,
                    attempt=attempt,
                    branch_name=branch_name,
                    target_branch=target_branch,
                    merge_command=[],
                    merge_stdout="",
                    merge_stderr="",
                    reason="main_checkout_dirty_conflict",
                    dirty_paths=dirty_overlap,
                )
                if llm_merge_resolver.get("applied", False):
                    dirty_overlap = self._dirty_merge_conflict_paths(
                        branch_name,
                        cwd=merge_workspace,
                        ignore_paths=set(identical_untracked_paths),
                    )
                if not dirty_overlap:
                    self._record_event(
                        "dirty_checkout_merge_blocker_resolved",
                        {
                            "task_id": task.task_id,
                            "attempt": attempt,
                            "branch": branch_name,
                            "target_branch": target_branch,
                            "llm_merge_resolver": llm_merge_resolver,
                        },
                    )
                result = {
                    "attempted": True,
                    "merged": False,
                    "returncode": 2,
                    "branch": branch_name,
                    "target_branch": target_branch,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "merge_commit": "",
                    "stdout": "",
                    "stderr": "",
                    "reason": "main_checkout_dirty_conflict",
                    "dirty_paths": dirty_overlap,
                    "main_worktree_path": str(merge_workspace),
                    "used_ephemeral_main_worktree": merge_workspace_ephemeral,
                    "identical_untracked_paths": identical_untracked_paths,
                    "resolved_generated_conflicts": resolved_add_add_conflicts,
                    "submodule_merge_results": [],
                }
                if dirty_overlap:
                    if llm_merge_resolver:
                        result["llm_merge_resolver"] = llm_merge_resolver
                    self._record_event("merge_finished", result)
                    return result

            removed_untracked = self._remove_untracked_paths_for_merge(identical_untracked_paths, cwd=merge_workspace)
            self._record_event(
                "merge_started",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "branch": branch_name,
                    "target_branch": target_branch,
                    "main_worktree_path": str(merge_workspace),
                    "used_ephemeral_main_worktree": merge_workspace_ephemeral,
                    "started_at": started_at,
                    "resolved_generated_conflicts": resolved_add_add_conflicts,
                },
            )
            command = [
                "git",
                "merge",
                "--no-ff",
                "--no-edit",
                branch_name,
            ]
            merge = subprocess.run(
                command,
                cwd=merge_workspace,
                text=True,
                capture_output=True,
                check=False,
            )
            finished_at = utc_now()
            merge_commit = ""
            submodule_merge_results: list[dict[str, Any]] = []
            submodule_conflict_repair: dict[str, Any] = {}
            merge_abort_result: dict[str, Any] = {}
            llm_merge_resolver: dict[str, Any] = {}
            llm_merge_commit_result: dict[str, Any] = {}
            merge_returncode = merge.returncode
            if merge_returncode != 0:
                submodule_conflict_repair = self._repair_submodule_gitlink_merge_conflicts(
                    merge_workspace,
                    task=task,
                )
                if submodule_conflict_repair.get("repaired", False):
                    merge_returncode = 0
                else:
                    llm_merge_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                        workspace=merge_workspace,
                        task=task,
                        attempt=attempt,
                        branch_name=branch_name,
                        target_branch=target_branch,
                        merge_command=command,
                        merge_stdout=merge.stdout,
                        merge_stderr=merge.stderr,
                    )
                    if llm_merge_resolver.get("applied", False):
                        llm_merge_commit_result = self._commit_llm_resolved_merge(merge_workspace)
                        if llm_merge_commit_result.get("completed", False):
                            merge_returncode = 0
                        elif (
                            llm_merge_commit_result.get("reason") == "no_merge_in_progress"
                            and self._branch_merged_in_workspace(merge_workspace, branch_name)
                        ):
                            llm_merge_commit_result = {
                                **llm_merge_commit_result,
                                "completed": True,
                                "reason": "resolver_committed_merge",
                                "commit": self._run_git(["rev-parse", "HEAD"], cwd=merge_workspace).stdout.strip(),
                            }
                            merge_returncode = 0
                        else:
                            merge_abort_result = self._abort_failed_merge(merge_workspace)
                    else:
                        merge_abort_result = self._abort_failed_merge(merge_workspace)
            if merge_returncode == 0:
                merge_commit = self._run_git(["rev-parse", "HEAD"], cwd=merge_workspace).stdout.strip()
                submodule_merge_results = self._merge_submodule_branches_to_main(
                    branch_name,
                    task=task,
                    attempt=attempt,
                )
            elif removed_untracked:
                self._restore_removed_untracked_paths(removed_untracked, cwd=merge_workspace)
            failed_submodules = [item for item in submodule_merge_results if not item.get("merged", False)]
            effective_returncode = merge_returncode
            effective_merged = merge_returncode == 0 and not failed_submodules
            result = {
                "attempted": True,
                "merged": effective_merged,
                "returncode": 2 if failed_submodules else effective_returncode,
                "branch": branch_name,
                "target_branch": target_branch,
                "command": command,
                "started_at": started_at,
                "finished_at": finished_at,
                "merge_commit": merge_commit,
                "stdout": merge.stdout[-4000:],
                "stderr": merge.stderr[-4000:],
                "main_worktree_path": str(merge_workspace),
                "used_ephemeral_main_worktree": merge_workspace_ephemeral,
                "identical_untracked_paths": identical_untracked_paths,
                "resolved_generated_conflicts": resolved_add_add_conflicts,
                "submodule_merge_results": submodule_merge_results,
            }
            if submodule_conflict_repair:
                result["submodule_conflict_repair"] = submodule_conflict_repair
            if llm_workspace_resolver:
                result["llm_workspace_resolver"] = llm_workspace_resolver
            if merge_abort_result:
                result["merge_abort_result"] = merge_abort_result
            if llm_merge_resolver:
                result["llm_merge_resolver"] = llm_merge_resolver
            if llm_merge_commit_result:
                result["llm_merge_commit_result"] = llm_merge_commit_result
            if failed_submodules:
                result["submodule_merge_failed"] = True
                result["reason"] = "submodule_merge_failed"
            self._record_event("merge_finished", result)
            return result
        finally:
            if merge_workspace is not None:
                merge_workspace_cleanup = self._cleanup_main_merge_workspace(
                    merge_workspace,
                    ephemeral=merge_workspace_ephemeral,
                )
                if not merge_workspace_cleanup.get("cleaned", False):
                    self._record_event("main_merge_worktree_cleanup_failed", merge_workspace_cleanup)
            try:
                if merge_lock.exists():
                    merge_lock.unlink()
            except OSError:
                logger.warning("Failed to remove merge lock %s", merge_lock)

    def _abort_failed_merge(self, cwd: Path) -> dict[str, Any]:
        merge_head = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if merge_head.returncode != 0:
            return {"attempted": False, "reason": "no_merge_in_progress"}
        abort = subprocess.run(
            ["git", "merge", "--abort"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        result = {
            "attempted": True,
            "aborted": abort.returncode == 0,
            "returncode": abort.returncode,
            "stdout": abort.stdout[-4000:],
            "stderr": abort.stderr[-4000:],
        }
        self._record_event("failed_merge_aborted", {"worktree_path": str(cwd), **result})
        return result

    def _invoke_llm_merge_resolver_for_failed_merge(
        self,
        *,
        workspace: Path,
        task: PortalTask,
        attempt: int,
        branch_name: str,
        target_branch: str,
        merge_command: list[str],
        merge_stdout: str,
        merge_stderr: str,
        reason: str = "merge_conflict",
        dirty_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        command_template = self.llm_merge_resolver_command
        if not command_template:
            return {"attempted": False, "reason": "resolver_command_not_configured"}
        from ipfs_accelerate_py.agent_supervisor.merge_resolver import build_merge_prompt, invoke_llm_resolver

        merge_result = {
            "attempted": True,
            "merged": False,
            "returncode": 1,
            "branch": branch_name,
            "target_branch": target_branch,
            "command": merge_command,
            "reason": reason,
            "stdout": merge_stdout[-4000:],
            "stderr": merge_stderr[-4000:],
            "main_worktree_path": str(workspace),
            "dirty_paths": dirty_paths or [],
        }
        event = {
            "type": "merge_finished",
            "task_id": task.task_id,
            "attempt": attempt,
            "merge_result": merge_result,
        }
        payload = {
            "found": True,
            "task_id": task.task_id,
            "attempt": attempt,
            "events_path": str(self.events_path),
            "repo_root": str(workspace),
            "branch": branch_name,
            "target_branch": target_branch,
            "command": merge_command,
            "reason": reason,
            "dirty_paths": dirty_paths or [],
            "unmerged_paths": sorted(self._unmerged_worktree_paths(workspace)),
            "prompt": build_merge_prompt(event=event, repo_root=workspace),
        }
        self._mark_long_running_phase(
            task_id=task.task_id,
            phase="merge_resolver",
            detail=reason,
        )
        result = invoke_llm_resolver(
            payload,
            command_template=command_template,
            timeout_seconds=self.llm_merge_resolver_timeout_seconds,
        )
        compact_result = dict(result)
        if "prompt" in compact_result:
            compact_result["prompt_chars"] = len(str(compact_result.pop("prompt") or ""))
        self._record_event("llm_merge_resolver_invoked", compact_result)
        return compact_result

    def _branch_merged_in_workspace(self, workspace: Path, branch_name: str) -> bool:
        if self._unmerged_worktree_paths(workspace):
            return False
        if not branch_name:
            return False
        return self._git_ref_is_ancestor_in_repo(workspace, branch_name, "HEAD")

    def _commit_llm_resolved_merge(self, workspace: Path) -> dict[str, Any]:
        unresolved = sorted(self._unmerged_worktree_paths(workspace))
        if unresolved:
            return {
                "attempted": True,
                "completed": False,
                "reason": "unresolved_paths_remain",
                "unresolved_paths": unresolved,
            }
        merge_head = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if merge_head.returncode != 0:
            return {"attempted": False, "completed": False, "reason": "no_merge_in_progress"}
        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "commit",
                "--no-edit",
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "attempted": True,
            "completed": commit.returncode == 0,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }

    def _repair_submodule_gitlink_merge_conflicts(
        self,
        workspace: Path,
        *,
        task: PortalTask,
    ) -> dict[str, Any]:
        conflicts = self._unmerged_gitlink_conflicts(workspace)
        if not conflicts:
            return {"repaired": False, "reason": "no_gitlink_conflicts"}
        repairs: list[dict[str, Any]] = []
        for relative, stages in conflicts.items():
            selected_commit = self._select_submodule_gitlink_resolution(relative, stages, task=task)
            if not selected_commit:
                repairs.append(
                    {
                        "path": relative,
                        "repaired": False,
                        "reason": "no_safe_resolution",
                        "stages": stages,
                    }
                )
                continue
            update = subprocess.run(
                ["git", "update-index", "--add", "--cacheinfo", f"160000,{selected_commit},{relative}"],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=False,
            )
            repairs.append(
                {
                    "path": relative,
                    "repaired": update.returncode == 0,
                    "reason": "selected_current_equivalent_submodule_head"
                    if update.returncode == 0
                    else "update_index_failed",
                    "selected_commit": selected_commit,
                    "stages": stages,
                    "returncode": update.returncode,
                    "stdout": update.stdout[-4000:],
                    "stderr": update.stderr[-4000:],
                }
            )
        unresolved = self._unmerged_worktree_paths(workspace)
        if unresolved:
            result = {
                "repaired": False,
                "reason": "unresolved_paths_remain",
                "repairs": repairs,
                "unresolved_paths": sorted(unresolved),
            }
            self._record_event("submodule_gitlink_conflict_repair", result)
            return result
        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "commit",
                "--no-edit",
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if commit.returncode != 0:
            result = {
                "repaired": False,
                "reason": "merge_commit_failed",
                "repairs": repairs,
                "returncode": commit.returncode,
                "stdout": commit.stdout[-4000:],
                "stderr": commit.stderr[-4000:],
            }
            self._record_event("submodule_gitlink_conflict_repair", result)
            return result
        merge_commit = self._run_git(["rev-parse", "HEAD"], cwd=workspace).stdout.strip()
        result = {
            "repaired": True,
            "reason": "committed_resolved_gitlinks",
            "repairs": repairs,
            "merge_commit": merge_commit,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }
        self._record_event("submodule_gitlink_conflict_repair", result)
        return result

    def _unmerged_gitlink_conflicts(self, cwd: Path) -> dict[str, dict[str, str]]:
        result = subprocess.run(
            ["git", "ls-files", "-u", "-z"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return {}
        conflicts: dict[str, dict[str, str]] = {}
        for raw_entry in result.stdout.split(b"\0"):
            if not raw_entry:
                continue
            try:
                metadata, raw_path = raw_entry.split(b"\t", 1)
                mode, object_id, stage = metadata.decode("ascii").split()
            except ValueError:
                continue
            if mode != "160000":
                continue
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            if not self._repo_relative_path_safe(relative):
                continue
            conflicts.setdefault(relative, {})[stage] = object_id
        return conflicts

    def _select_submodule_gitlink_resolution(
        self,
        relative: str,
        stages: dict[str, str],
        *,
        task: PortalTask,
    ) -> str:
        source = (self.repo_root / relative).resolve()
        if not self._is_git_worktree(source):
            return ""
        if self._run_git(["status", "--porcelain"], cwd=source).stdout.strip():
            return ""
        head = self._run_git(["rev-parse", "HEAD"], cwd=source).stdout.strip()
        theirs = stages.get("3", "")
        if theirs and self._git_ref_is_ancestor_in_repo(source, theirs, head):
            return head
        if task.task_id and self._submodule_head_has_task_commit(source, task.task_id):
            return head
        return ""

    def _submodule_head_has_task_commit(self, source: Path, task_id: str) -> bool:
        result = subprocess.run(
            ["git", "log", "--format=%H", "--fixed-strings", f"--grep={task_id}:", "HEAD"],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    def _merge_submodule_branches_to_main(
        self,
        branch_name: str,
        *,
        task: PortalTask,
        attempt: int,
    ) -> list[dict[str, Any]]:
        return self._merge_submodule_branches_to_main_in_repo(
            repo_path=self.repo_root,
            branch_name=branch_name,
            parent_relative="",
            task=task,
            attempt=attempt,
        )

    def _merge_submodule_branches_to_main_in_repo(
        self,
        *,
        repo_path: Path,
        branch_name: str,
        parent_relative: str,
        task: PortalTask,
        attempt: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        relatives = self.worktree_submodule_paths if not parent_relative else tuple(self._declared_submodule_paths(repo_path))
        for relative in relatives:
            full_relative = f"{parent_relative.rstrip('/')}/{relative}" if parent_relative else relative
            source = (self.repo_root / full_relative).resolve()
            submodule_branch = self._submodule_worktree_branch_name(branch_name, full_relative)
            if not self._is_git_worktree(source):
                continue
            if not self._git_ref_exists_in_repo(source, submodule_branch):
                continue
            default_branch = self._submodule_default_branch(relative, source)
            dirty = self._run_git(["status", "--porcelain"], cwd=source).stdout.strip()
            if dirty:
                llm_merge_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                    workspace=source,
                    task=task,
                    attempt=attempt,
                    branch_name=submodule_branch,
                    target_branch=default_branch,
                    merge_command=[],
                    merge_stdout="",
                    merge_stderr="",
                    reason="submodule_checkout_dirty",
                    dirty_paths=self._dirty_status_paths(dirty),
                )
                if llm_merge_resolver.get("applied", False):
                    dirty = self._run_git(["status", "--porcelain"], cwd=source).stdout.strip()
                if not dirty:
                    self._record_event(
                        "submodule_checkout_blocker_resolved",
                        {
                            "task_id": task.task_id,
                            "attempt": attempt,
                            "path": full_relative,
                            "branch": submodule_branch,
                            "default_branch": default_branch,
                            "llm_merge_resolver": llm_merge_resolver,
                        },
                    )
                else:
                    results.append(
                        {
                            "path": full_relative,
                            "branch": submodule_branch,
                            "default_branch": default_branch,
                            "merged": False,
                            "reason": "submodule_checkout_dirty",
                            "status": dirty,
                            "dirty_paths": self._dirty_status_paths(dirty),
                            "llm_merge_resolver": llm_merge_resolver,
                        }
                    )
                    continue
            if self._git_ref_is_ancestor_in_repo(source, submodule_branch, default_branch):
                results.append(
                    {
                        "path": full_relative,
                        "branch": submodule_branch,
                        "default_branch": default_branch,
                        "merged": True,
                        "reason": "already_merged",
                    }
                )
                continue
            if self._git_current_branch(source) != default_branch:
                checkout = subprocess.run(
                    ["git", "checkout", default_branch],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if checkout.returncode != 0:
                    llm_merge_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                        workspace=source,
                        task=task,
                        attempt=attempt,
                        branch_name=submodule_branch,
                        target_branch=default_branch,
                        merge_command=["git", "checkout", default_branch],
                        merge_stdout=checkout.stdout,
                        merge_stderr=checkout.stderr,
                        reason="submodule_default_branch_checkout_failed",
                    )
                    if llm_merge_resolver.get("applied", False) and self._git_current_branch(source) != default_branch:
                        checkout = subprocess.run(
                            ["git", "checkout", default_branch],
                            cwd=source,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                    if self._git_current_branch(source) == default_branch:
                        self._record_event(
                            "submodule_checkout_blocker_resolved",
                            {
                                "task_id": task.task_id,
                                "attempt": attempt,
                                "path": full_relative,
                                "branch": submodule_branch,
                                "default_branch": default_branch,
                                "llm_merge_resolver": llm_merge_resolver,
                            },
                        )
                    else:
                        results.append(
                            {
                                "path": full_relative,
                                "branch": submodule_branch,
                                "default_branch": default_branch,
                                "merged": False,
                                "returncode": checkout.returncode,
                                "reason": "default_branch_checkout_failed",
                                "stdout": checkout.stdout[-4000:],
                                "stderr": checkout.stderr[-4000:],
                                "llm_merge_resolver": llm_merge_resolver,
                            }
                        )
                        continue
            merge_command = ["git", "merge", "--ff-only", submodule_branch]
            merge = subprocess.run(
                merge_command,
                cwd=source,
                text=True,
                capture_output=True,
                check=False,
            )
            merge_abort_result: dict[str, Any] = {}
            llm_merge_resolver: dict[str, Any] = {}
            llm_merge_commit_result: dict[str, Any] = {}
            ff_only_result = {
                "returncode": merge.returncode,
                "stdout": merge.stdout[-4000:],
                "stderr": merge.stderr[-4000:],
            }
            if merge.returncode != 0:
                merge_command = ["git", "merge", "--no-ff", "--no-edit", submodule_branch]
                merge = subprocess.run(
                    merge_command,
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if merge.returncode != 0:
                    llm_merge_resolver = self._invoke_llm_merge_resolver_for_failed_merge(
                        workspace=source,
                        task=task,
                        attempt=attempt,
                        branch_name=submodule_branch,
                        target_branch=default_branch,
                        merge_command=merge_command,
                        merge_stdout=merge.stdout,
                        merge_stderr=merge.stderr,
                        reason="submodule_merge_conflict",
                    )
                    if llm_merge_resolver.get("applied", False):
                        llm_merge_commit_result = self._commit_llm_resolved_merge(source)
                        if llm_merge_commit_result.get("completed", False):
                            merge = subprocess.CompletedProcess(merge_command, 0, merge.stdout, merge.stderr)
                        elif (
                            llm_merge_commit_result.get("reason") == "no_merge_in_progress"
                            and self._branch_merged_in_workspace(source, submodule_branch)
                        ):
                            llm_merge_commit_result = {
                                **llm_merge_commit_result,
                                "completed": True,
                                "reason": "resolver_committed_merge",
                                "commit": self._run_git(["rev-parse", "HEAD"], cwd=source).stdout.strip(),
                            }
                            merge = subprocess.CompletedProcess(merge_command, 0, merge.stdout, merge.stderr)
                        else:
                            merge_abort_result = self._abort_failed_merge(source)
                    else:
                        merge_abort_result = self._abort_failed_merge(source)
            result = {
                "path": full_relative,
                "branch": submodule_branch,
                "default_branch": default_branch,
                "merged": merge.returncode == 0,
                "returncode": merge.returncode,
                "command": merge_command,
                "stdout": merge.stdout[-4000:],
                "stderr": merge.stderr[-4000:],
                "commit": "",
                "ff_only_result": ff_only_result,
            }
            if merge_abort_result:
                result["merge_abort_result"] = merge_abort_result
            if llm_merge_resolver:
                result["llm_merge_resolver"] = llm_merge_resolver
            if llm_merge_commit_result:
                result["llm_merge_commit_result"] = llm_merge_commit_result
            if merge.returncode == 0:
                result["commit"] = self._run_git(["rev-parse", "HEAD"], cwd=source).stdout.strip()
            results.append(result)
            if merge.returncode == 0:
                results.extend(
                    self._merge_submodule_branches_to_main_in_repo(
                        repo_path=source,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                        task=task,
                        attempt=attempt,
                    )
                )
        return results

    @staticmethod
    def _dirty_status_paths(status: str) -> list[str]:
        paths: list[str] = []
        for line in status.splitlines():
            if len(line) < 3:
                continue
            if len(line) >= 3 and line[2] == " ":
                path = line[3:].strip()
            elif len(line) >= 2 and line[1] == " ":
                path = line[2:].strip()
            else:
                path = line.strip()
            if " -> " in path:
                path = path.rsplit(" -> ", 1)[1].strip()
            if path:
                paths.append(path)
        return paths

    def _submodule_default_branch(self, relative: str, source: Path) -> str:
        result = subprocess.run(
            ["git", "config", "--file", str(self.repo_root / ".gitmodules"), "--get-regexp", r"^submodule\..*\.path$"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                key, _, path_value = line.partition(" ")
                if path_value.strip() != relative:
                    continue
                module_key = key.rsplit(".", 1)[0]
                branch = subprocess.run(
                    ["git", "config", "--file", str(self.repo_root / ".gitmodules"), "--get", f"{module_key}.branch"],
                    cwd=self.repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if branch.returncode == 0 and branch.stdout.strip():
                    return branch.stdout.strip()
        current = self._git_current_branch(source)
        return current or "main"

    def _git_ref_is_ancestor_in_repo(self, cwd: Path, ancestor: str, descendant: str) -> bool:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _cleanup_merged_worktree(self, worktree_path: Path | None, branch_name: str) -> dict[str, Any]:
        started_at = utc_now()
        removed_worktree = False
        deleted_branch = False
        submodule_cleanup: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            if worktree_path is not None:
                submodule_cleanup = self._cleanup_worktree_submodules(worktree_path, branch_name)
            if worktree_path is not None and worktree_path.exists():
                self._run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=self.repo_root)
                removed_worktree = True
            if self._git_ref_exists(branch_name):
                self._run_git(["branch", "-D", branch_name], cwd=self.repo_root)
                deleted_branch = True
        except RuntimeError as exc:
            errors.append(str(exc))

        if errors:
            result = {
                "cleaned": False,
                "branch": branch_name,
                "worktree_path": str(worktree_path or ""),
                "started_at": started_at,
                "finished_at": utc_now(),
                "removed_worktree": removed_worktree,
                "deleted_branch": deleted_branch,
                "submodule_cleanup": submodule_cleanup,
                "error": "\n".join(errors),
            }
            self._record_event("cleanup_finished", result)
            return result

        result = {
            "cleaned": True,
            "branch": branch_name,
            "worktree_path": str(worktree_path or ""),
            "started_at": started_at,
            "finished_at": utc_now(),
            "removed_worktree": removed_worktree,
            "deleted_branch": deleted_branch,
            "submodule_cleanup": submodule_cleanup,
        }
        self._record_event("cleanup_finished", result)
        return result

    def _cleanup_worktree_submodules(
        self,
        worktree_path: Path,
        branch_name: str,
        *,
        parent_relative: str = "",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        relatives = self.worktree_submodule_paths if not parent_relative else tuple(self._declared_submodule_paths(worktree_path))
        for relative in relatives:
            full_relative = f"{parent_relative.rstrip('/')}/{relative}" if parent_relative else relative
            source = (self.repo_root / full_relative).resolve()
            target = worktree_path / relative
            submodule_branch = self._submodule_worktree_branch_name(branch_name, full_relative)
            if not self._is_git_worktree(source):
                continue
            removed_worktree = False
            deleted_branch = False
            nested_cleanup: list[dict[str, Any]] = []
            errors: list[str] = []
            if self._is_git_worktree(target):
                nested_cleanup = self._cleanup_worktree_submodules(
                    target,
                    branch_name,
                    parent_relative=full_relative,
                )
                remove = subprocess.run(
                    ["git", "worktree", "remove", "--force", str(target)],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if remove.returncode == 0:
                    removed_worktree = True
                else:
                    errors.append((remove.stderr or remove.stdout).strip())
            default_branch = self._submodule_default_branch(relative, source)
            if self._git_ref_exists_in_repo(source, submodule_branch) and self._git_ref_is_ancestor_in_repo(
                source, submodule_branch, default_branch
            ):
                delete = subprocess.run(
                    ["git", "branch", "-D", submodule_branch],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if delete.returncode == 0:
                    deleted_branch = True
                else:
                    errors.append((delete.stderr or delete.stdout).strip())
            results.append(
                {
                    "path": full_relative,
                    "branch": submodule_branch,
                    "removed_worktree": removed_worktree,
                    "deleted_branch": deleted_branch,
                    "cleaned": not errors,
                    "errors": errors,
                    "nested_submodule_cleanup": nested_cleanup,
                }
            )
        return results

    def _dirty_merge_conflict_paths(
        self,
        branch_name: str,
        *,
        cwd: Path | None = None,
        ignore_paths: set[str] | None = None,
    ) -> list[str]:
        workspace = cwd or self.repo_root
        dirty_paths = self._dirty_worktree_paths(workspace)
        if not dirty_paths:
            return []
        branch_paths = self._branch_changed_paths(branch_name)
        overlap = dirty_paths & branch_paths
        if ignore_paths:
            overlap -= ignore_paths
        return sorted(overlap)

    def _resolve_generated_add_add_conflicts(self, *, cwd: Path | None = None) -> list[dict[str, Any]]:
        workspace = cwd or self.repo_root
        results: list[dict[str, Any]] = []
        for relative in self._unmerged_add_add_paths(workspace):
            if not self._generated_add_add_conflict_path_allowed(relative):
                continue
            ours = self._conflict_stage_blob(workspace, relative, stage=2)
            theirs = self._conflict_stage_blob(workspace, relative, stage=3)
            selected = self._select_generated_conflict_blob(ours, theirs)
            if selected is None:
                results.append(
                    {
                        "path": relative,
                        "resolved": False,
                        "reason": "contents_not_equivalent_or_contained",
                    }
                )
                continue
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(selected)
            add = subprocess.run(
                ["git", "add", "--", relative],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=False,
            )
            results.append(
                {
                    "path": relative,
                    "resolved": add.returncode == 0,
                    "reason": "selected_equivalent_generated_content" if add.returncode == 0 else "git_add_failed",
                    "returncode": add.returncode,
                    "stdout": add.stdout[-4000:],
                    "stderr": add.stderr[-4000:],
                }
            )
        if results:
            self._record_event(
                "generated_add_add_conflict_repair",
                {"main_worktree_path": str(workspace), "results": results},
            )
        return results

    def _unmerged_add_add_paths(self, cwd: Path) -> list[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        paths: list[str] = []
        for raw_entry in result.stdout.split(b"\0"):
            if not raw_entry or len(raw_entry) < 4:
                continue
            status = raw_entry[:2].decode("ascii", errors="ignore")
            if status != "AA":
                continue
            relative = raw_entry[3:].decode("utf-8", errors="surrogateescape")
            if relative:
                paths.append(relative)
        return paths

    def _generated_add_add_conflict_path_allowed(self, relative: str) -> bool:
        if not self._repo_relative_path_safe(relative):
            return False
        normalized = relative.strip("/")
        return any(self._path_matches_prefix(normalized, prefix) for prefix in GENERATED_ADD_ADD_CONFLICT_PREFIXES)

    def _conflict_stage_blob(self, cwd: Path, relative: str, *, stage: int) -> bytes | None:
        result = subprocess.run(
            ["git", "show", f":{stage}:{relative}"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    @staticmethod
    def _select_generated_conflict_blob(ours: bytes | None, theirs: bytes | None) -> bytes | None:
        if ours is None or theirs is None:
            return None
        if ours == theirs:
            return ours
        if ours and ours in theirs:
            return theirs
        if theirs and theirs in ours:
            return ours
        return None

    def _identical_untracked_merge_paths(self, branch_name: str, *, cwd: Path | None = None) -> list[str]:
        workspace = cwd or self.repo_root
        branch_paths = self._branch_changed_paths(branch_name)
        if not branch_paths:
            return []
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        untracked_paths = {
            path
            for path in result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
            if path and path in branch_paths
        }
        identical: list[str] = []
        for relative in sorted(untracked_paths):
            if not self._repo_relative_path_safe(relative):
                continue
            source = workspace / relative
            if not source.is_file() or source.is_symlink():
                continue
            branch_blob = subprocess.run(
                ["git", "show", f"{branch_name}:{relative}"],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
            if branch_blob.returncode == 0 and source.read_bytes() == branch_blob.stdout:
                identical.append(relative)
        return identical

    def _remove_untracked_paths_for_merge(self, paths: list[str], *, cwd: Path | None = None) -> dict[str, bytes]:
        workspace = cwd or self.repo_root
        removed: dict[str, bytes] = {}
        for relative in paths:
            if not self._repo_relative_path_safe(relative):
                continue
            source = workspace / relative
            if not source.is_file() or source.is_symlink():
                continue
            removed[relative] = source.read_bytes()
            source.unlink()
        return removed

    def _restore_removed_untracked_paths(self, removed: dict[str, bytes], *, cwd: Path | None = None) -> None:
        workspace = cwd or self.repo_root
        for relative, content in removed.items():
            if not self._repo_relative_path_safe(relative):
                continue
            target = workspace / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    @staticmethod
    def _repo_relative_path_safe(relative: str) -> bool:
        if not relative or relative.startswith("/") or "\0" in relative:
            return False
        return ".." not in Path(relative).parts

    def _dirty_worktree_paths(self, cwd: Path) -> set[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return set()
        paths: set[str] = set()
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            path_text = line[3:].strip()
            if " -> " in path_text:
                original, renamed = path_text.split(" -> ", 1)
                if original:
                    paths.add(original.strip())
                if renamed:
                    paths.add(renamed.strip())
                continue
            if path_text:
                paths.add(path_text)
        return paths

    def _branch_changed_paths(self, branch_name: str, *, base_ref: str | None = None) -> set[str]:
        base = base_ref or self._branch_merge_base(branch_name, self._main_branch_name())
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base}..{branch_name}"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return set()
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _branch_merge_base(self, branch_name: str, target_branch: str) -> str:
        result = subprocess.run(
            ["git", "merge-base", target_branch, branch_name],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return target_branch

    def _reconcile_failed_merges(self, *, skip_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        target_branch = self._main_branch_name()
        for event in self._failed_merge_candidates(skip_task_ids=skip_task_ids):
            task_id = str(event.get("task_id") or "")
            attempt = int(event.get("attempt") or 0)
            branch = str(event.get("branch") or "")
            worktree_path_text = str(event.get("worktree_path") or "")
            worktree_path = Path(worktree_path_text) if worktree_path_text else None
            implementation_commit = str(event.get("implementation_commit") or "")
            if not task_id or not implementation_commit:
                continue
            if self._git_ref_is_ancestor(implementation_commit, target_branch):
                cleanup_result = self._cleanup_merged_worktree(worktree_path, branch) if branch else {}
                cleanup_cleaned = bool(cleanup_result.get("cleaned", False)) if cleanup_result else True
                result = {
                    "task_id": task_id,
                    "attempt": attempt,
                    "branch": branch,
                    "implementation_commit": implementation_commit,
                    "resolved": cleanup_cleaned,
                    "reason": "implementation_commit_already_merged" if cleanup_cleaned else "cleanup_retry_failed",
                    "cleanup_result": cleanup_result,
                }
                self._record_event("merge_reconciled", result)
                results.append(result)
                continue
            if not branch or not self._git_ref_exists(branch):
                result = {
                    "task_id": task_id,
                    "attempt": attempt,
                    "branch": branch,
                    "implementation_commit": implementation_commit,
                    "resolved": False,
                    "reason": "implementation_branch_missing",
                }
                self._record_event("merge_reconcile_skipped", result)
                results.append(result)
                continue

            task = PortalTask(
                task_id=task_id,
                title=str(event.get("title") or "failed implementation merge"),
                status="todo",
                completion="manual",
                priority="P2",
                track="ops",
            )
            self._mark_long_running_phase(
                task_id=task_id,
                phase="merge_reconciliation",
                detail=branch,
            )
            try:
                merge_result = self._merge_branch_to_main(
                    branch,
                    task,
                    attempt,
                    baseline_ref=str(event.get("baseline_ref") or ""),
                )
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "attempt": attempt,
                    "branch": branch,
                    "implementation_commit": implementation_commit,
                    "resolved": False,
                    "reason": "merge_reconcile_exception",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[-4000:],
                }
                self._record_event("merge_reconcile_exception", result)
                results.append(result)
                continue
            cleanup_result = {}
            if merge_result.get("merged"):
                cleanup_result = self._cleanup_merged_worktree(worktree_path, branch)
            result = {
                "task_id": task_id,
                "attempt": attempt,
                "branch": branch,
                "implementation_commit": implementation_commit,
                "resolved": bool(merge_result.get("merged")),
                "reason": "merge_retried",
                "merge_result": merge_result,
                "cleanup_result": cleanup_result,
            }
            self._record_event("merge_reconciled", result)
            results.append(result)
        return results

    def _failed_merge_candidates(self, *, skip_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
        skip_task_ids = skip_task_ids or set()
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        reconciled_commits: set[str] = set()
        abandoned_commits: set[str] = set()
        target_branch = self._main_branch_name()
        for event in self._iter_events():
            if str(event.get("type") or "") == "merge_reconciled":
                implementation_commit = str(event.get("implementation_commit") or "")
                merge_result = event.get("merge_result") or {}
                merge_reason = merge_result.get("reason") if isinstance(merge_result, dict) else ""
                if implementation_commit and event.get("resolved"):
                    reconciled_commits.add(implementation_commit)
                elif implementation_commit and merge_reason == "baseline_not_ancestor_of_target":
                    abandoned_commits.add(implementation_commit)
                continue
            if str(event.get("type") or "") != "implementation_finished":
                continue
            task_id = str(event.get("task_id") or "")
            if task_id in skip_task_ids:
                continue
            implementation_commit = str(event.get("implementation_commit") or "")
            if (
                not implementation_commit
                or implementation_commit in reconciled_commits
                or implementation_commit in abandoned_commits
            ):
                continue
            validation = event.get("validation_result") or {}
            if isinstance(validation, dict) and validation.get("attempted") and not validation.get("passed", False):
                continue
            merge_result = event.get("merge_result") or {}
            if not isinstance(merge_result, dict):
                continue
            if not self._merge_result_needs_reconciliation(merge_result):
                continue
            key = (task_id, implementation_commit)
            candidates[key] = event

        unresolved: list[dict[str, Any]] = []
        for event in candidates.values():
            implementation_commit = str(event.get("implementation_commit") or "")
            if implementation_commit in reconciled_commits or implementation_commit in abandoned_commits:
                continue
            if implementation_commit and not self._git_ref_is_ancestor(implementation_commit, target_branch):
                unresolved.append(event)
                continue
            cleanup = event.get("cleanup_result") or {}
            if isinstance(cleanup, dict) and not cleanup.get("cleaned", False):
                unresolved.append(event)
        return unresolved

    @staticmethod
    def _merge_result_needs_reconciliation(merge_result: dict[str, Any]) -> bool:
        if not isinstance(merge_result, dict) or merge_result.get("merged"):
            return False
        if merge_result.get("attempted"):
            return True
        return str(merge_result.get("reason") or "") in {
            "lock_exists",
            "lock_unavailable",
            "lock_cleanup_failed",
        }

    def _unresolved_merge_failures_by_task(self, *, skip_task_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
        skip_task_ids = skip_task_ids or set()
        failures: dict[str, dict[str, Any]] = {}
        target_branch = self._main_branch_name()
        for event in self._failed_merge_candidates(skip_task_ids=skip_task_ids):
            task_id = str(event.get("task_id") or "")
            if task_id in skip_task_ids:
                continue
            implementation_commit = str(event.get("implementation_commit") or "")
            if task_id and implementation_commit and not self._git_ref_is_ancestor(implementation_commit, target_branch):
                failures[task_id] = event
        return failures

    def _has_unresolved_merge_failure(self, task: PortalTask, previous: PortalTaskState) -> bool:
        if previous.last_implementation_task_id != task.task_id:
            return False
        if not previous.last_implementation_commit:
            return False
        if previous.last_merge_returncode in (None, 0):
            return False
        if previous.last_merge_commit:
            return False
        return not self._git_ref_is_ancestor(previous.last_implementation_commit, self._main_branch_name())

    def _git_ref_is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _git_ref_exists(self, ref: str) -> bool:
        if not ref:
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _implementation_lock_path(self) -> Path:
        return self.state_path.parent / "implementation.lock"

    def _build_implementation_lock_metadata(self, task: PortalTask, attempt: int, started_at: str) -> dict[str, Any]:
        return {
            "kind": "implementation",
            "pid": os.getpid(),
            "owner_script": Path(sys.argv[0]).name,
            "repo_root": str(self.repo_root.resolve()),
            "state_dir": str(self.state_path.parent.resolve()),
            "task_id": task.task_id,
            "attempt": attempt,
            "started_at": started_at,
        }

    def _build_merge_lock_metadata(
        self,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        started_at: str,
    ) -> dict[str, Any]:
        return {
            "kind": "merge",
            "pid": os.getpid(),
            "owner_script": Path(sys.argv[0]).name,
            "repo_root": str(self.repo_root.resolve()),
            "task_id": task.task_id,
            "attempt": attempt,
            "branch": branch_name,
            "started_at": started_at,
        }

    def _implementation_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        state_dir = str(metadata.get("state_dir") or "")
        if state_dir and Path(state_dir).resolve() != self.state_path.parent.resolve():
            return False
        return self._lock_owner_is_active(metadata, expected_kind="implementation")

    def _merge_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        repo_root = str(metadata.get("repo_root") or "")
        if repo_root and Path(repo_root).resolve() != self.repo_root.resolve():
            return False
        return self._lock_owner_is_active(metadata, expected_kind="merge")

    def _lock_owner_is_active(self, metadata: dict[str, Any], *, expected_kind: str) -> bool:
        kind = str(metadata.get("kind") or "")
        if kind and kind != expected_kind:
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

    def _try_acquire_lock(
        self,
        lock_path: Path,
        *,
        lock_kind: str,
        owner_active: Any,
    ) -> tuple[int | None, str, dict[str, Any] | None]:
        for _ in range(2):
            try:
                return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "acquired", None
            except FileExistsError:
                existing = load_json_dict(lock_path)
                if existing is not None and owner_active(existing):
                    return None, "lock_exists", existing
                if not self._clear_stale_lock(lock_path, lock_kind=lock_kind, metadata=existing):
                    return None, "lock_cleanup_failed", existing
        existing = load_json_dict(lock_path)
        if existing is not None and owner_active(existing):
            return None, "lock_exists", existing
        return None, "lock_unavailable", existing

    def _write_lock_metadata(self, lock_fd: int, metadata: dict[str, Any]) -> None:
        try:
            os.write(lock_fd, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"))
        finally:
            os.close(lock_fd)

    def _clear_stale_lock(self, lock_path: Path, *, lock_kind: str, metadata: dict[str, Any] | None) -> bool:
        moved_directory_path = ""
        try:
            if lock_path.is_dir():
                backup_path = unique_backup_path(lock_path, "directory-backup")
                lock_path.rename(backup_path)
                moved_directory_path = str(backup_path)
            else:
                lock_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            logger.warning("Failed to remove stale %s lock %s", lock_kind, lock_path)
            return False
        event = {
            "lock_path": str(lock_path),
            "lock_owner_pid": int(metadata.get("pid") or 0) if metadata else 0,
            "task_id": str(metadata.get("task_id") or "") if metadata else "",
            "branch": str(metadata.get("branch") or "") if metadata else "",
        }
        if moved_directory_path:
            event["moved_directory_path"] = moved_directory_path
        self._record_event(f"{lock_kind}_lock_cleared", event)
        return True

    def _find_live_inflight_implementation(self) -> dict[str, Any] | None:
        inflight_events = self._inflight_implementation_events()
        for event in reversed(inflight_events):
            if self._implementation_process_active(event):
                return event
        return None

    def _inflight_implementation_events(self) -> list[dict[str, Any]]:
        inflight: dict[tuple[str, int], dict[str, Any]] = {}
        for event in self._iter_events():
            event_type = str(event.get("type") or "")
            task_id = str(event.get("task_id") or "")
            attempt = int(event.get("attempt") or 0)
            if not task_id or attempt <= 0:
                continue
            key = (task_id, attempt)
            if event_type == "implementation_started":
                inflight[key] = event
            elif event_type == "implementation_finished":
                inflight.pop(key, None)

        return list(inflight.values())

    def _latest_implementation_finished_by_task(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._iter_events():
            if str(event.get("type") or "") != "implementation_finished":
                continue
            task_id = str(event.get("task_id") or "")
            if task_id:
                latest[task_id] = event
        return latest

    def _successfully_merged_task_ids(self) -> set[str]:
        task_ids: set[str] = set()
        target_branch = self._main_branch_name()
        for event in self._iter_events():
            event_type = str(event.get("type") or "")
            task_id = str(event.get("task_id") or "")
            if not task_id:
                continue
            implementation_commit = str(event.get("implementation_commit") or "")
            if event_type == "implementation_finished":
                merge_result = event.get("merge_result") or {}
                if not isinstance(merge_result, dict) or not merge_result.get("merged"):
                    continue
            elif event_type == "merge_reconciled":
                if not event.get("resolved"):
                    continue
            else:
                continue
            if implementation_commit and not self._git_ref_is_ancestor(implementation_commit, target_branch):
                continue
            task_ids.add(task_id)
        return task_ids

    def _task_has_recent_no_change_outcome(
        self,
        task_id: str,
        latest_results: dict[str, dict[str, Any]],
        *,
        now_ts: float | None = None,
    ) -> bool:
        latest = latest_results.get(task_id)
        if not latest:
            return False
        commit_result = latest.get("commit_result") or {}
        if not isinstance(commit_result, dict):
            return False
        if commit_result.get("reason") != "no_changes":
            return False
        if int(latest.get("returncode") or 0) != 0:
            return False
        event_timestamp = parse_timestamp(str(latest.get("timestamp") or ""))
        if event_timestamp is None:
            return False
        age = (now_ts or time.time()) - event_timestamp.timestamp()
        return max(0.0, age) < RECENT_NO_CHANGE_COOLDOWN_SECONDS

    def _iter_events(self) -> list[dict[str, Any]]:
        return read_jsonl_events(self.events_path, repair=True)

    def _implementation_process_active(self, event: dict[str, Any]) -> bool:
        worktree_path = str(event.get("worktree_path") or "")
        command = event.get("command") or []
        process_lines = self._list_process_commands()
        if worktree_path:
            return any(worktree_path in line for line in process_lines)
        if isinstance(command, list):
            command_text = " ".join(str(item) for item in command if item)
            if command_text:
                return any(command_text in line for line in process_lines)
        return False

    def _list_process_commands(self) -> list[str]:
        result = subprocess.run(
            ["ps", "-eo", "args="],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _repo_merge_lock_path(self) -> Path:
        git_common_dir = self._run_git(["rev-parse", "--git-common-dir"], cwd=self.repo_root).stdout.strip()
        path = Path(git_common_dir)
        if not path.is_absolute():
            path = self.repo_root / path
        return path / "implementation-main-merge.lock"

    def _run_git(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def _build_implementation_command(self, workspace_path: Path) -> list[str]:
        if self.implementation_command:
            return shlex.split(self.implementation_command)
        env_command = os.environ.get("IMPLEMENTATION_DAEMON_COMMAND", "").strip()
        if env_command:
            return shlex.split(env_command)
        codex = shutil.which("codex")
        copilot = shutil.which("copilot")
        if copilot:
            return _copilot_fallback_command(codex=codex, copilot=copilot, workspace_path=workspace_path)
        if codex:
            return [
                codex,
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C",
                str(workspace_path),
                "-",
            ]
        raise RuntimeError(
            "No implementation command configured. Install codex or copilot, or set IMPLEMENTATION_DAEMON_COMMAND."
        )

    def _task_metadata_value(self, task: PortalTask, *keys: str) -> str:
        normalized = {
            str(key).strip().lower().replace("_", " "): str(value).strip()
            for key, value in task.metadata.items()
        }
        for key in keys:
            value = normalized.get(str(key).strip().lower().replace("_", " "))
            if value:
                return value
        return ""

    def _resolve_context_path(self, value: Any) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    def _todo_vector_index_candidate_paths(self, task: PortalTask) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()

        def add(path: Path | None) -> None:
            if path is None:
                return
            key = str(path)
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        strategy = load_json_dict(self.strategy_path) or {}
        for key in (
            "last_objective_todo_vector_index_path",
            "objective_todo_vector_index_path",
            "todo_vector_index_path",
        ):
            add(self._resolve_context_path(strategy.get(key)))

        for key in ("todo vector index path", "todo_vector_index_path", "todo vector index"):
            add(self._resolve_context_path(self._task_metadata_value(task, key)))

        bundle_shard = self._task_metadata_value(task, "bundle shard", "bundle_shard")
        bundle_shard_path = self._resolve_context_path(bundle_shard)
        if bundle_shard_path is not None:
            add(bundle_shard_path.parent / "todo_vector_index.json")

        add(self.todo_path.parent / "todo_vector_index.json")
        add(self.repo_root / "todo_vector_index.json")
        return candidates

    def _load_todo_vector_context(self, task: PortalTask) -> dict[str, Any] | None:
        for index_path in self._todo_vector_index_candidate_paths(task):
            payload = load_json_dict(index_path)
            if payload is None:
                continue
            records = payload.get("records")
            if not isinstance(records, list):
                continue
            by_task = {
                str(record.get("task_id") or ""): record
                for record in records
                if isinstance(record, dict) and str(record.get("task_id") or "")
            }
            record = by_task.get(task.task_id)
            if not isinstance(record, dict):
                continue
            cluster: dict[str, Any] | None = None
            for item in payload.get("clusters", []) if isinstance(payload.get("clusters"), list) else []:
                if not isinstance(item, dict):
                    continue
                task_ids = item.get("task_ids")
                if isinstance(task_ids, list) and task.task_id in {str(task_id) for task_id in task_ids}:
                    cluster = item
                    break

            related_ids: list[str] = []
            for raw_task_id in record.get("related_task_ids", []) if isinstance(record.get("related_task_ids"), list) else []:
                related_task_id = str(raw_task_id)
                if related_task_id and related_task_id != task.task_id and related_task_id not in related_ids:
                    related_ids.append(related_task_id)
            if cluster is not None:
                for raw_task_id in cluster.get("task_ids", []) if isinstance(cluster.get("task_ids"), list) else []:
                    related_task_id = str(raw_task_id)
                    if related_task_id and related_task_id != task.task_id and related_task_id not in related_ids:
                        related_ids.append(related_task_id)
            merge_candidates: list[dict[str, Any]] = []
            for candidate in payload.get("merge_candidates", []) if isinstance(payload.get("merge_candidates"), list) else []:
                if not isinstance(candidate, dict):
                    continue
                candidate_task_ids = candidate.get("task_ids")
                if isinstance(candidate_task_ids, list) and task.task_id in {str(task_id) for task_id in candidate_task_ids}:
                    merge_candidates.append(candidate)
            bundle_contexts: list[dict[str, Any]] = []
            for bundle_context in payload.get("bundle_contexts", []) if isinstance(payload.get("bundle_contexts"), list) else []:
                if not isinstance(bundle_context, dict):
                    continue
                context_task_ids = bundle_context.get("task_ids")
                if isinstance(context_task_ids, list) and task.task_id in {str(task_id) for task_id in context_task_ids}:
                    bundle_contexts.append(bundle_context)
            execution_packets: list[dict[str, Any]] = []
            for execution_packet in payload.get("execution_packets", []) if isinstance(payload.get("execution_packets"), list) else []:
                if not isinstance(execution_packet, dict):
                    continue
                packet_task_ids = execution_packet.get("active_task_ids") or execution_packet.get("task_ids")
                if isinstance(packet_task_ids, list) and task.task_id in {str(task_id) for task_id in packet_task_ids}:
                    execution_packets.append(execution_packet)
            aggregate_primary = (
                str(record.get("candidate_kind") or "").strip().lower() == "goal_packet_aggregate"
                or str(record.get("goal_packet_role") or "").strip().lower() == "packet_aggregate"
                or str(record.get("merge_role") or "").strip().lower() == "packet_aggregate"
            )
            covered_packet_task_ids: list[str] = []
            if aggregate_primary:
                for execution_packet in execution_packets:
                    packet_task_ids = execution_packet.get("active_task_ids") or execution_packet.get("task_ids")
                    if not isinstance(packet_task_ids, list):
                        continue
                    primary_task_id = str(execution_packet.get("primary_task_id") or "")
                    if primary_task_id and primary_task_id != task.task_id:
                        continue
                    for packet_task_id in packet_task_ids:
                        normalized = str(packet_task_id)
                        if normalized and normalized != task.task_id and normalized not in covered_packet_task_ids:
                            covered_packet_task_ids.append(normalized)
            related_record_limit = 0 if aggregate_primary and covered_packet_task_ids else 5

            return {
                "index_path": index_path,
                "record": record,
                "cluster": cluster or {},
                "related_records": [
                    by_task[task_id]
                    for task_id in related_ids
                    if task_id in by_task and task_id not in set(covered_packet_task_ids)
                ][:related_record_limit],
                "merge_candidates": merge_candidates[:3],
                "bundle_contexts": bundle_contexts[:2],
                "execution_packets": execution_packets[:2],
                "aggregate_primary": aggregate_primary,
                "covered_packet_task_ids": covered_packet_task_ids,
            }
        return None

    def _display_context_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def _compact_value_list(self, value: Any, *, limit: int = 8) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = split_csv(str(value or ""))
        return items[:limit]

    def _estimate_prompt_tokens(self, value: str) -> int:
        return len(re.findall(r"[A-Za-z0-9_./:-]+", str(value or "")))

    def _budgeted_todo_vector_context(
        self,
        required_lines: list[str],
        optional_lines: list[str],
        *,
        token_budget: int = DEFAULT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET,
    ) -> str:
        """Render compact vector context without letting optional details bloat prompts."""

        budget = max(80, int(token_budget or DEFAULT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET))
        lines: list[str] = []
        current_tokens = 0
        for line in required_lines:
            if not line:
                continue
            lines.append(line)
            current_tokens += self._estimate_prompt_tokens(line)

        skipped = 0
        for line in optional_lines:
            if not line:
                continue
            line_tokens = self._estimate_prompt_tokens(line)
            if current_tokens + line_tokens <= budget:
                lines.append(line)
                current_tokens += line_tokens
            else:
                skipped += 1
        if skipped:
            summary = f"- Context budget: kept {current_tokens}/{budget} estimated tokens; skipped {skipped} lower-priority vector details"
            if current_tokens + self._estimate_prompt_tokens(summary) <= budget + 24:
                lines.append(summary)
        return "\n".join(lines)

    def _render_todo_vector_context(self, task: PortalTask) -> str:
        context = self._load_todo_vector_context(task)
        if context is None:
            return ""
        record = context["record"]
        cluster = context["cluster"]
        index_path = context["index_path"]

        fields = [
            ("Todo vector key", record.get("vector_key") or record.get("todo_vector_key")),
            ("Merge key", record.get("merge_key")),
            ("Merge family", record.get("merge_family")),
            ("Merge role", record.get("merge_role")),
            ("Work item count", record.get("work_item_count")),
            ("Work scope", record.get("work_scope")),
            ("Goal packet", record.get("goal_packet_key")),
            ("Goal packet role", record.get("goal_packet_role")),
            ("Goal packet task count", record.get("goal_packet_task_count")),
            ("Goal packet work item count", record.get("goal_packet_work_item_count")),
            ("Surplus group", record.get("surplus_group")),
            ("Candidate kind", record.get("candidate_kind")),
            ("Goal id", record.get("goal_id")),
            ("Graph depth", record.get("graph_depth")),
            ("Bundle", record.get("bundle_key")),
            ("Cluster", cluster.get("cluster_key") if isinstance(cluster, dict) else ""),
        ]
        required_lines = [f"- Index: {self._display_context_path(index_path)}"]
        optional_lines: list[str] = []
        required_field_labels = {
            "Todo vector key",
            "Merge key",
            "Merge family",
            "Work item count",
            "Work scope",
            "Goal packet",
            "Goal packet role",
            "Goal packet task count",
            "Goal packet work item count",
            "Goal id",
            "Bundle",
        }
        for label, value in fields:
            text = str(value or "").strip()
            if text:
                target = required_lines if label in required_field_labels else optional_lines
                target.append(f"- {label}: {text}")

        missing_evidence = self._compact_value_list(record.get("missing_evidence"), limit=8)
        if missing_evidence:
            required_lines.append(f"- Missing evidence: {', '.join(missing_evidence)}")

        packet_goals = self._compact_value_list(record.get("goal_packet_goal_ids"), limit=8)
        if packet_goals:
            required_lines.append(f"- Goal packet goals: {', '.join(packet_goals)}")

        cluster_task_ids = self._compact_value_list(cluster.get("task_ids") if isinstance(cluster, dict) else [], limit=10)
        if cluster_task_ids:
            optional_lines.append(f"- Cluster task ids: {', '.join(cluster_task_ids)}")

        symbol_candidates = [
            *self._compact_value_list(record.get("ast_symbols"), limit=24),
            *self._compact_value_list(cluster.get("ast_symbols") if isinstance(cluster, dict) else [], limit=24),
        ]
        ast_symbols = sorted({item for item in symbol_candidates if item})[:24]
        if ast_symbols:
            optional_lines.append(f"- AST symbols: {', '.join(ast_symbols)}")

        execution_packet_entries: list[str] = []
        for packet in context.get("execution_packets", []):
            if not isinstance(packet, dict):
                continue
            compact = str(packet.get("compact_packet") or "").strip()
            if compact:
                execution_packet_entries.append(compact)
                continue
            packet_key = str(packet.get("packet_key") or "").strip()
            active_ids = ", ".join(self._compact_value_list(packet.get("active_task_ids"), limit=6))
            work_items = str(packet.get("work_item_count_total") or "").strip()
            outputs = ", ".join(
                self._compact_value_list(
                    packet.get("shared_outputs") or packet.get("all_outputs"),
                    limit=5,
                )
            )
            details = [
                part
                for part in (
                    packet_key,
                    f"active={active_ids}" if active_ids else "",
                    f"work_items={work_items}" if work_items else "",
                    f"outputs={outputs}" if outputs else "",
                )
                if part
            ]
            if details:
                execution_packet_entries.append("; ".join(details))
        if execution_packet_entries:
            required_lines.insert(1, f"- Execution packets: {' | '.join(execution_packet_entries)}")

        covered_packet_task_ids = self._compact_value_list(context.get("covered_packet_task_ids"), limit=12)
        if covered_packet_task_ids:
            required_lines.append(f"- Packet sibling tasks covered by primary: {', '.join(covered_packet_task_ids)}")

        bundle_context_entries: list[str] = []
        for bundle_context in context.get("bundle_contexts", []):
            if not isinstance(bundle_context, dict):
                continue
            compact = str(bundle_context.get("compact_context") or "").strip()
            if compact:
                bundle_context_entries.append(compact)
                continue
            context_key = str(bundle_context.get("context_key") or "").strip()
            confidence = str(bundle_context.get("confidence") or "").strip()
            active_ids = ", ".join(self._compact_value_list(bundle_context.get("active_task_ids"), limit=6))
            merge_ready = "true" if bundle_context.get("merge_ready") else "false"
            outputs = ", ".join(
                self._compact_value_list(
                    bundle_context.get("shared_outputs") or bundle_context.get("all_outputs"),
                    limit=4,
                )
            )
            details = [
                part
                for part in (
                    context_key,
                    confidence,
                    f"merge_ready={merge_ready}",
                    f"active={active_ids}" if active_ids else "",
                    f"outputs={outputs}" if outputs else "",
                )
                if part
            ]
            if details:
                bundle_context_entries.append("; ".join(details))
        if bundle_context_entries:
            optional_lines.append(f"- Bundle contexts: {' | '.join(bundle_context_entries)}")

        merge_candidate_entries: list[str] = []
        for candidate in context.get("merge_candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_key = str(candidate.get("candidate_key") or "").strip()
            confidence = str(candidate.get("confidence") or "").strip()
            active_ids = ", ".join(self._compact_value_list(candidate.get("active_task_ids"), limit=5))
            evidence = ", ".join(self._compact_value_list(candidate.get("missing_evidence"), limit=5))
            outputs = ", ".join(self._compact_value_list(candidate.get("shared_outputs") or candidate.get("all_outputs"), limit=4))
            details = [
                part
                for part in (
                    candidate_key,
                    confidence,
                    f"active={active_ids}" if active_ids else "",
                    f"missing={evidence}" if evidence else "",
                    f"outputs={outputs}" if outputs else "",
                )
                if part
            ]
            if details:
                merge_candidate_entries.append("; ".join(details))
        if merge_candidate_entries:
            optional_lines.append(f"- Merge candidates: {' | '.join(merge_candidate_entries)}")

        related_entries: list[str] = []
        for related in context["related_records"]:
            related_id = str(related.get("task_id") or "").strip()
            if not related_id:
                continue
            title = str(related.get("title") or "").strip()
            status = str(related.get("status") or "").strip()
            evidence = ", ".join(self._compact_value_list(related.get("missing_evidence"), limit=3))
            outputs = ", ".join(self._compact_value_list(related.get("outputs"), limit=3))
            details = [part for part in (status, title, f"missing={evidence}" if evidence else "", f"outputs={outputs}" if outputs else "") if part]
            related_entries.append(f"{related_id} ({'; '.join(details)})")
        if related_entries:
            optional_lines.append(f"- Related tasks: {' | '.join(related_entries)}")

        return self._budgeted_todo_vector_context(required_lines, optional_lines)

    def _build_implementation_prompt(self, task: PortalTask, attempt: int) -> str:
        todo_vector_context = self._render_todo_vector_context(task)
        todo_vector_context_section = (
            f"""
Compact todo vector context:
{todo_vector_context}
"""
            if todo_vector_context
            else ""
        )
        return f"""You are an autonomous implementation agent working in this repository.

Implement exactly this backlog task and keep changes scoped.

Task:
- ID: {task.task_id}
- Title: {task.title}
- Priority: {task.priority}
- Track: {task.track}
- Attempt: {attempt}
- Todo file: {self.todo_path}
- Source line: {task.source_line}
- Depends on: {", ".join(task.depends_on) or "none"}
- Expected outputs: {", ".join(task.outputs) or "none listed"}
- Validation commands: {"; ".join(task.validation) or "none listed"}
- Acceptance: {task.acceptance or "none listed"}
{todo_vector_context_section}

Primary plan document:
- docs/AI_AGENT_CHAT_IMPLEMENTATION_PLAN.md when the task ID starts with AGENT-
- docs/211_SERVICE_NAVIGATION_PORTAL_PLAN.md when the task ID starts with PORTAL-

Rules:
- Read the relevant plan and nearby code before editing.
- Do not revert unrelated local changes.
- Prefer existing repo patterns and small, reviewable changes.
- Implement the expected outputs for this task.
- If a compact execution packet or goal packet is shown, prefer one cohesive implementation that advances the shared packet evidence without making unrelated edits.
- Run the listed validation commands when practical.
- The daemon will run the listed validation commands and will only commit and merge the worktree if they pass.
- Leave generated artifacts and shared dependency paths alone; the daemon restores dist, screenshot artifacts, and linked node_modules before commit.
- If validation cannot be run, record why in your final response.
- Do not mark the backlog task completed manually unless the task explicitly asks for TODO metadata changes.
- Final response should list changed files and validation results.
"""

    def _build_recommended_actions(self, task: PortalTask) -> list[str]:
        actions = [f"Implement outputs for {task.task_id}: {', '.join(task.outputs)}"]
        for command in task.validation:
            actions.append(f"Validate with: {command}")
        if task.acceptance:
            actions.append(f"Acceptance: {task.acceptance}")
        return actions

    def _load_todo_vector_payload_for_tasks(self, tasks: list[PortalTask]) -> dict[str, Any] | None:
        for task in tasks:
            for index_path in self._todo_vector_index_candidate_paths(task):
                payload = load_json_dict(index_path)
                if payload is None:
                    continue
                records = payload.get("records")
                if not isinstance(records, list):
                    continue
                task_ids = {
                    str(record.get("task_id") or "")
                    for record in records
                    if isinstance(record, dict) and str(record.get("task_id") or "")
                }
                if any(task.task_id in task_ids for task in tasks):
                    return payload
        return None

    def _todo_vector_selection_context(
        self,
        tasks: list[PortalTask],
        ready_task_ids: set[str],
    ) -> dict[str, Any]:
        payload = self._load_todo_vector_payload_for_tasks(tasks)
        if payload is None:
            return {}

        records = payload.get("records")
        if not isinstance(records, list):
            return {}
        record_by_task = {
            str(record.get("task_id") or ""): record
            for record in records
            if isinstance(record, dict) and str(record.get("task_id") or "")
        }

        cluster_by_task: dict[str, str] = {}
        ready_cluster_sizes: dict[str, int] = {}
        clusters = payload.get("clusters")
        if isinstance(clusters, list):
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue
                cluster_key = str(cluster.get("cluster_key") or "")
                if not cluster_key:
                    continue
                task_ids = cluster.get("task_ids")
                if not isinstance(task_ids, list):
                    continue
                normalized_ids = [str(task_id) for task_id in task_ids if str(task_id)]
                for task_id in normalized_ids:
                    cluster_by_task[task_id] = cluster_key
                ready_count = sum(1 for task_id in normalized_ids if task_id in ready_task_ids)
                if ready_count:
                    ready_cluster_sizes[cluster_key] = ready_count

        bundle_context_by_task: dict[str, str] = {}
        ready_bundle_context_sizes: dict[str, int] = {}
        bundle_contexts = payload.get("bundle_contexts")
        if isinstance(bundle_contexts, list):
            for bundle_context in bundle_contexts:
                if not isinstance(bundle_context, dict):
                    continue
                context_key = str(bundle_context.get("context_key") or "")
                if not context_key:
                    continue
                task_ids = bundle_context.get("task_ids")
                if not isinstance(task_ids, list):
                    continue
                normalized_ids = [str(task_id) for task_id in task_ids if str(task_id)]
                for task_id in normalized_ids:
                    bundle_context_by_task.setdefault(task_id, context_key)
                ready_count = sum(1 for task_id in normalized_ids if task_id in ready_task_ids)
                if ready_count:
                    ready_bundle_context_sizes[context_key] = ready_count
        execution_packet_by_task: dict[str, str] = {}
        ready_execution_packet_sizes: dict[str, int] = {}
        execution_packet_primary_by_task: dict[str, str] = {}
        execution_packets = payload.get("execution_packets")
        if isinstance(execution_packets, list):
            for packet in execution_packets:
                if not isinstance(packet, dict):
                    continue
                packet_key = str(packet.get("packet_key") or "")
                if not packet_key:
                    continue
                primary_task_id = str(packet.get("primary_task_id") or "")
                task_ids = packet.get("active_task_ids") or packet.get("task_ids")
                if not isinstance(task_ids, list):
                    continue
                normalized_ids = [str(task_id) for task_id in task_ids if str(task_id)]
                for task_id in normalized_ids:
                    execution_packet_by_task.setdefault(task_id, packet_key)
                    if primary_task_id:
                        execution_packet_primary_by_task.setdefault(task_id, primary_task_id)
                ready_count = sum(1 for task_id in normalized_ids if task_id in ready_task_ids)
                if ready_count:
                    ready_execution_packet_sizes[packet_key] = ready_count

        state = PortalTaskState.load(self.state_path)
        anchor_task_id = state.last_implementation_task_id or state.active_task_id
        anchor_record = record_by_task.get(anchor_task_id) if anchor_task_id else None
        return {
            "record_by_task": record_by_task,
            "cluster_by_task": cluster_by_task,
            "ready_cluster_sizes": ready_cluster_sizes,
            "bundle_context_by_task": bundle_context_by_task,
            "ready_bundle_context_sizes": ready_bundle_context_sizes,
            "execution_packet_by_task": execution_packet_by_task,
            "ready_execution_packet_sizes": ready_execution_packet_sizes,
            "execution_packet_primary_by_task": execution_packet_primary_by_task,
            "anchor_task_id": anchor_task_id,
            "anchor_record": anchor_record if isinstance(anchor_record, dict) else None,
            "anchor_cluster_key": cluster_by_task.get(anchor_task_id, "") if anchor_task_id else "",
            "anchor_bundle_context_key": bundle_context_by_task.get(anchor_task_id, "") if anchor_task_id else "",
            "anchor_execution_packet_key": execution_packet_by_task.get(anchor_task_id, "") if anchor_task_id else "",
        }

    @staticmethod
    def _todo_vector_record_int(record: dict[str, Any], key: str) -> int:
        try:
            return int(record.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _todo_vector_record_primary_rank(task_id: str, record: dict[str, Any], context: dict[str, Any]) -> int:
        execution_packet_primary_by_task = context.get("execution_packet_primary_by_task")
        primary_task_id = (
            str(execution_packet_primary_by_task.get(task_id) or "")
            if isinstance(execution_packet_primary_by_task, dict)
            else ""
        )
        candidate_kind = str(record.get("candidate_kind") or "").strip().lower()
        packet_role = str(record.get("goal_packet_role") or "").strip().lower()
        merge_role = str(record.get("merge_role") or "").strip().lower()
        if primary_task_id and task_id == primary_task_id:
            return 0
        if candidate_kind == "goal_packet_aggregate" or packet_role == "packet_aggregate" or merge_role == "packet_aggregate":
            return 1
        if packet_role == "packet_anchor":
            return 2
        if candidate_kind == "aggregate":
            return 3
        if candidate_kind == "evidence_cluster":
            return 4
        if packet_role == "packet_member":
            return 5
        return 6

    def _todo_vector_selection_rank(self, task: PortalTask, context: dict[str, Any]) -> tuple[int, ...]:
        record_by_task = context.get("record_by_task")
        if not isinstance(record_by_task, dict):
            return (9, 9, 0, 0, 0, 0, 0, 0)
        record = record_by_task.get(task.task_id)
        if not isinstance(record, dict):
            return (9, 9, 0, 0, 0, 0, 0, 0)

        cluster_by_task = context.get("cluster_by_task")
        ready_cluster_sizes = context.get("ready_cluster_sizes")
        cluster_key = (
            str(cluster_by_task.get(task.task_id) or "")
            if isinstance(cluster_by_task, dict)
            else ""
        )
        ready_cluster_size = (
            int(ready_cluster_sizes.get(cluster_key) or 0)
            if cluster_key and isinstance(ready_cluster_sizes, dict)
            else 0
        )
        bundle_context_by_task = context.get("bundle_context_by_task")
        ready_bundle_context_sizes = context.get("ready_bundle_context_sizes")
        bundle_context_key = (
            str(bundle_context_by_task.get(task.task_id) or "")
            if isinstance(bundle_context_by_task, dict)
            else ""
        )
        ready_bundle_context_size = (
            int(ready_bundle_context_sizes.get(bundle_context_key) or 0)
            if bundle_context_key and isinstance(ready_bundle_context_sizes, dict)
            else 0
        )
        execution_packet_by_task = context.get("execution_packet_by_task")
        ready_execution_packet_sizes = context.get("ready_execution_packet_sizes")
        execution_packet_key = (
            str(execution_packet_by_task.get(task.task_id) or "")
            if isinstance(execution_packet_by_task, dict)
            else ""
        )
        ready_execution_packet_size = (
            int(ready_execution_packet_sizes.get(execution_packet_key) or 0)
            if execution_packet_key and isinstance(ready_execution_packet_sizes, dict)
            else 0
        )
        primary_rank = self._todo_vector_record_primary_rank(task.task_id, record, context)
        work_item_count = self._todo_vector_record_int(record, "work_item_count")
        packet_work_item_count = self._todo_vector_record_int(record, "goal_packet_work_item_count")
        token_count = int(record.get("token_count") or 0)

        anchor = context.get("anchor_record")
        if not isinstance(anchor, dict):
            return (
                5,
                primary_rank,
                -work_item_count,
                -packet_work_item_count,
                -ready_execution_packet_size,
                -ready_bundle_context_size,
                -ready_cluster_size,
                token_count,
            )

        anchor_related = {
            str(task_id)
            for task_id in anchor.get("related_task_ids", [])
            if str(task_id)
        } if isinstance(anchor.get("related_task_ids"), list) else set()
        record_related = {
            str(task_id)
            for task_id in record.get("related_task_ids", [])
            if str(task_id)
        } if isinstance(record.get("related_task_ids"), list) else set()
        anchor_task_id = str(context.get("anchor_task_id") or "")
        anchor_cluster_key = str(context.get("anchor_cluster_key") or "")
        anchor_bundle_context_key = str(context.get("anchor_bundle_context_key") or "")
        anchor_execution_packet_key = str(context.get("anchor_execution_packet_key") or "")

        if record.get("merge_key") and record.get("merge_key") == anchor.get("merge_key"):
            relation_rank = 0
        elif record.get("goal_packet_key") and record.get("goal_packet_key") == anchor.get("goal_packet_key"):
            relation_rank = 1
        elif execution_packet_key and anchor_execution_packet_key and execution_packet_key == anchor_execution_packet_key:
            relation_rank = 2
        elif bundle_context_key and anchor_bundle_context_key and bundle_context_key == anchor_bundle_context_key:
            relation_rank = 3
        elif cluster_key and anchor_cluster_key and cluster_key == anchor_cluster_key:
            relation_rank = 4
        elif task.task_id in anchor_related or (anchor_task_id and anchor_task_id in record_related):
            relation_rank = 5
        elif record.get("merge_family") and record.get("merge_family") == anchor.get("merge_family"):
            relation_rank = 6
        elif record.get("surplus_group") and record.get("surplus_group") == anchor.get("surplus_group"):
            relation_rank = 7
        elif record.get("goal_id") and record.get("goal_id") == anchor.get("goal_id"):
            relation_rank = 8
        else:
            relation_rank = 9
        return (
            relation_rank,
            primary_rank,
            -work_item_count,
            -packet_work_item_count,
            -ready_execution_packet_size,
            -ready_bundle_context_size,
            -ready_cluster_size,
            token_count,
        )

    def _select_next_task(
        self,
        tasks: list[PortalTask],
        resolved_statuses: dict[str, str],
        strategy: dict[str, Any],
        unresolved_merge_failures: dict[str, dict[str, Any]],
        recent_outcomes: dict[str, dict[str, Any]],
    ) -> PortalTask | None:
        ready = [task for task in tasks if resolved_statuses.get(task.task_id) == "ready"]
        if not ready:
            return None
        ready_task_ids = {task.task_id for task in ready}
        vector_context = self._todo_vector_selection_context(tasks, ready_task_ids)
        focus_order = {
            track: index
            for index, track in enumerate(
                [str(item).lower() for item in strategy.get("focus_tracks", DEFAULT_TRACKS)]
            )
        }
        deprioritized = {str(item) for item in strategy.get("deprioritized_tasks", [])}

        def sort_key(task: PortalTask) -> tuple[Any, ...]:
            selection_penalty = 0
            if task.task_id in unresolved_merge_failures:
                selection_penalty += UNRESOLVED_MERGE_SELECTION_PENALTY
            if self._task_has_recent_no_change_outcome(task.task_id, recent_outcomes):
                selection_penalty += NO_CHANGE_SELECTION_PENALTY
            vector_rank = self._todo_vector_selection_rank(task, vector_context)
            return (
                selection_penalty,
                PRIORITY_ORDER.get(task.priority, 99),
                1 if task.task_id in deprioritized else 0,
                focus_order.get(task.track, len(focus_order)),
                *vector_rank,
                len(task.depends_on),
                task.task_id,
            )

        return sorted(ready, key=sort_key)[0]

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl_event(self.events_path, event_type, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the portal implementation backlog daemon")
    parser.add_argument("--once", action="store_true", help="Run one backlog pass and exit")
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between backlog passes")
    parser.add_argument(
        "--todo-path",
        type=Path,
        default=Path("docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md"),
        help="Machine-readable markdown backlog",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("data/portal_implementation/state"),
        help="Portal daemon state directory",
    )
    parser.add_argument(
        "--task-prefix",
        default=TASK_HEADER_PREFIX,
        help="Markdown heading prefix for tasks, for example '## PORTAL-' or '## AGENT-'",
    )
    parser.add_argument(
        "--state-prefix",
        default="portal",
        help="State file prefix inside --state-dir",
    )
    parser.add_argument("--implement", action="store_true", help="Invoke an autonomous implementation agent for the ready task")
    parser.add_argument(
        "--implementation-command",
        default="",
        help="Command used for implementation. Defaults to codex exec with local Copilot CLI fallback when available.",
    )
    parser.add_argument(
        "--llm-merge-resolver-command",
        default=os.environ.get(LLM_MERGE_RESOLVER_COMMAND_ENV, ""),
        help=(
            "Command invoked with merge-conflict repair prompts on stdin. "
            f"Defaults to {LLM_MERGE_RESOLVER_COMMAND_ENV}."
        ),
    )
    parser.add_argument(
        "--llm-merge-resolver-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Timeout for the merge resolver subprocess. "
            f"Defaults to {LLM_MERGE_RESOLVER_TIMEOUT_ENV} or 600 seconds; <=0 disables."
        ),
    )
    parser.add_argument("--implementation-timeout", type=float, default=DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS)
    parser.add_argument(
        "--no-ephemeral-worktree",
        action="store_true",
        help="Run the implementation command in the main checkout instead of an isolated temporary git worktree",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=None,
        help="Directory for temporary implementation worktrees. Defaults to the system temp directory.",
    )
    parser.add_argument(
        "--worktree-submodule-path",
        action="append",
        default=[],
        help=(
            "Repo-relative submodule path to initialize and commit inside implementation worktrees. "
            "May be repeated or comma-separated. Defaults to IPFS_ACCELERATE_AGENT_WORKTREE_SUBMODULE_PATHS."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


TodoTask = PortalTask
TodoTaskState = PortalTaskState
TodoImplementationDaemon = PortalImplementationDaemon


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.llm_merge_resolver_command:
        os.environ[LLM_MERGE_RESOLVER_COMMAND_ENV] = args.llm_merge_resolver_command
    if args.llm_merge_resolver_timeout_seconds is not None:
        os.environ[LLM_MERGE_RESOLVER_TIMEOUT_ENV] = str(args.llm_merge_resolver_timeout_seconds)
    daemon = PortalImplementationDaemon(
        todo_path=args.todo_path,
        state_path=args.state_dir / f"{args.state_prefix}_task_state.json",
        strategy_path=args.state_dir / f"{args.state_prefix}_strategy.json",
        events_path=args.state_dir / f"{args.state_prefix}_events.jsonl",
        repo_root=REPO_ROOT,
        task_header_prefix=args.task_prefix,
        implement=args.implement,
        implementation_command=args.implementation_command or None,
        implementation_timeout=args.implementation_timeout,
        use_ephemeral_worktree=args.implement and not args.no_ephemeral_worktree,
        worktree_root=args.worktree_root,
        worktree_submodule_paths=args.worktree_submodule_path or None,
        llm_merge_resolver_command=args.llm_merge_resolver_command or None,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
    )
    while True:
        result = daemon.run_once()
        logger.info("Portal implementation daemon pass complete: %s", result)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
