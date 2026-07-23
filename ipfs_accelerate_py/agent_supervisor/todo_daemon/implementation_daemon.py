from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import pid_alive as _shared_pid_alive
from .core import process_args as _shared_process_args
from .engine import atomic_write_json as _shared_atomic_write_json
from ..checkout_lock import checkout_lock_metadata, checkout_mutation_lock_path
from ..event_log import append_jsonl_event, read_jsonl_events, repair_jsonl_event_log, unique_backup_path
from ..merge_conflict_repair import (
    resolve_append_only_markdown_conflicts,
    resolve_launch_readiness_conflicts,
    resolve_reconciliation_guardrail_todo_conflicts,
)
from ..submodule_degradation import DegradationState
from ..persistent_task_queue import PersistentTaskQueue
from ..task_identity import TaskIdentity, canonical_task_identity
from ..git_gc import GitGarbageCollector
from ..llm_merge_resolver_fallback import llm_merge_resolver_fallback_command
from ..merge_checkpoint import MergeCheckpoint
from ..merge_queue import MergeQueue
from ..validation_commands import normalize_validation_command_text, split_validation_commands
from ..validation_scheduler import ValidationScheduler
from .runner import TodoDaemonHooks, TodoDaemonRunner
from .supervisor_runtime import run_process_group_stream
from .worktrees import WorktreeLease, WorktreePool

REPO_ROOT = Path.cwd()

logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")

TASK_HEADER_PREFIX = "## PORTAL-"
DEFAULT_TRACKS = [
    "launch",
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
WORKTREE_POOL_ENABLED_ENV = "IPFS_ACCELERATE_AGENT_WORKTREE_POOL_ENABLED"
WORKTREE_POOL_MAX_ENTRIES_ENV = "IPFS_ACCELERATE_AGENT_WORKTREE_POOL_MAX_ENTRIES"
DEFAULT_WORKTREE_POOL_MAX_ENTRIES = 4
SHARED_WORKTREE_SOURCE_ROOT_ENV = "IPFS_ACCELERATE_AGENT_SHARED_WORKTREE_SOURCE_ROOT"
LLM_MERGE_RESOLVER_COMMAND_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND"
LLM_MERGE_RESOLVER_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS"
DAEMON_MERGE_RECONCILIATION_MAX_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_MERGE_RECONCILIATION_MAX"
DEFAULT_DAEMON_MERGE_RECONCILIATION_MAX = 3
DAEMON_MERGED_WORKTREE_CLEANUP_MAX_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_MERGED_WORKTREE_CLEANUP_MAX"
DEFAULT_DAEMON_MERGED_WORKTREE_CLEANUP_MAX = 25
DAEMON_HOOK_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_HOOK_TIMEOUT_SECONDS"
DEFAULT_DAEMON_HOOK_TIMEOUT_SECONDS = 60.0
DAEMON_MAINTENANCE_INTERVAL_ENV = "IPFS_ACCELERATE_AGENT_DAEMON_MAINTENANCE_INTERVAL_SECONDS"
DEFAULT_DAEMON_MAINTENANCE_INTERVAL_SECONDS = 300.0
MERGE_RECONCILIATION_MAX_AGE_ENV = "IPFS_ACCELERATE_AGENT_MERGE_RECONCILIATION_MAX_AGE_SECONDS"
DEFAULT_MERGE_RECONCILIATION_MAX_AGE_SECONDS = 86400
UNSUPPORTED_TYPESCRIPT_VALIDATION_FLAGS = ("--ignoreConfig",)
RECENT_NO_CHANGE_COOLDOWN_SECONDS = 1800.0
NO_CHANGE_SELECTION_PENALTY = 50
UNRESOLVED_MERGE_SELECTION_PENALTY = 1000
TRANSIENT_MERGE_LOCK_REASONS = frozenset(
    {
        "lock_exists",
        "lock_unavailable",
        "lock_cleanup_failed",
    }
)
TRANSIENT_MERGE_RETRY_BUDGET_WHEN_DISABLED = 1
IMPLEMENTATION_TASK_CLAIM_LOCK_KIND = "implementation_task_claim"
IMPLEMENTATION_TASK_CLAIM_LOCK_DIRNAME = "implementation-task-claims"
VALIDATION_MAX_WORKERS_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_MAX_WORKERS"
VALIDATION_RESOURCE_BUDGET_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_RESOURCE_BUDGET"
DEFAULT_VALIDATION_MAX_WORKERS = 2
MAX_MERGE_PROOF_METADATA_ITEMS = 256
MAX_MERGE_PROOF_METADATA_DEPTH = 8
MAX_MERGE_PROOF_METADATA_TEXT = 4096
TRANSIENT_MERGE_RETRY_MAX_AGE_WHEN_DISABLED_SECONDS = 900.0
IMPLEMENTATION_RUNNER_PROCESS_PATTERN = re.compile(r"(?:^|[\s/])(codex|copilot)(?:\s|$)")
PROVIDER_CAPACITY_BACKOFF_ENV = "IPFS_ACCELERATE_AGENT_PROVIDER_CAPACITY_BACKOFF_SECONDS"
DEFAULT_PROVIDER_CAPACITY_BACKOFF_SECONDS = 300.0
PROVIDER_CAPACITY_LOG_TAIL_BYTES = 128 * 1024
PROVIDER_CAPACITY_PATTERNS = (
    ("codex", re.compile(r"you(?:'|\u2019)?ve hit your usage limit", re.IGNORECASE)),
    ("copilot", re.compile(r"you(?:'|\u2019)?ve reached your additional usage limit", re.IGNORECASE)),
    ("provider", re.compile(r"(?:insufficient_quota|quota[_ ]exceeded|rate_limit_exceeded)", re.IGNORECASE)),
)
SHARED_WORKTREE_PATHS = (
    "wallet_interface/ui/node_modules",
    "mobile/node_modules",
    "swissknife/node_modules",
    "swissknife/web/node_modules",
    "swissknife/ipfs_accelerate_js/node_modules",
    "hallucinate_app/node_modules",
    "hallucinate_app/swissknife/node_modules",
)
DEFAULT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_TODO_VECTOR_CONTEXT_TOKEN_BUDGET", "600")
)


def default_llm_merge_resolver_command() -> str:
    """Return the configured resolver or the packaged agent fallback.

    The fallback starts Codex in the conflicted workspace and uses Copilot only
    when Codex cannot complete the repair. Keeping this as the daemon default
    means semantic merge conflicts are actively repaired instead of merely
    recorded for a later manual retry.
    """

    configured = os.environ.get(LLM_MERGE_RESOLVER_COMMAND_ENV, "").strip()
    if configured:
        return configured
    return llm_merge_resolver_fallback_command(python_executable=sys.executable)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _bounded_merge_proof_value(
    value: Any,
    *,
    depth: int = 0,
    field_name: str = "",
) -> Any:
    """Project proof metadata without persisting raw outputs or witnesses.

    Merge queue records are long-lived coordination artifacts, not proof
    artifact storage.  Preserve the bounded identities and verdicts needed to
    reproduce a gate decision while leaving potentially large or sensitive
    provider material in its dedicated artifact store.
    """

    if depth >= MAX_MERGE_PROOF_METADATA_DEPTH:
        return None
    normalized_name = str(field_name or "").strip().lower().replace("-", "_")
    if (
        normalized_name in {
            "output",
            "outputs",
            "stdout",
            "stderr",
            "witness",
            "witnesses",
            "counterexample",
            "counterexamples",
            "prompt",
            "prompts",
            "model_output",
            "provider_response",
            "proof_blob",
            "proof_bytes",
            "proof_text",
            "raw_context",
            "raw_output",
            "raw_response",
        }
        or normalized_name.endswith("_witness")
        or normalized_name.endswith("_output")
    ):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_MERGE_PROOF_METADATA_TEXT]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return _bounded_merge_proof_value(
            enum_value,
            depth=depth + 1,
            field_name=field_name,
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item))[
            :MAX_MERGE_PROOF_METADATA_ITEMS
        ]:
            key = str(raw_key)[:MAX_MERGE_PROOF_METADATA_TEXT]
            projected_value = _bounded_merge_proof_value(
                value[raw_key],
                depth=depth + 1,
                field_name=key,
            )
            if projected_value is not None:
                projected[key] = projected_value
        return projected
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        projected_items = []
        for item in value[:MAX_MERGE_PROOF_METADATA_ITEMS]:
            projected_item = _bounded_merge_proof_value(
                item,
                depth=depth + 1,
                field_name=field_name,
            )
            if projected_item is not None:
                projected_items.append(projected_item)
        return projected_items
    return str(value)[:MAX_MERGE_PROOF_METADATA_TEXT]


def shared_worktree_source_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return checkout roots that may provide untracked shared dependencies.

    An ephemeral worktree normally links dependencies from ``repo_root``. A
    branch checkout can omit untracked directories such as
    ``swissknife/node_modules`` though, so an operator may configure a stable
    checkout containing those installed dependencies. Multiple roots use the
    platform path separator.
    """

    candidates = [repo_root]
    configured = os.environ.get(SHARED_WORKTREE_SOURCE_ROOT_ENV, "")
    for raw_root in configured.split(os.pathsep):
        value = raw_root.strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidates.append(candidate)

    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


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


def normalize_focus_tracks(values: Any) -> list[str]:
    """Normalize scheduler focus tracks while keeping launch-readiness first."""

    if values is None:
        raw_values: list[Any] = list(DEFAULT_TRACKS)
    elif isinstance(values, str):
        raw_values = values.split(",")
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [values]

    configured = [str(item).strip().lower() for item in raw_values if str(item).strip()]
    return list(dict.fromkeys(["launch", *configured, *DEFAULT_TRACKS]))


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
GENERATED_NESTED_WORKTREE_DIR_NAMES = frozenset({"tmp"})
SUBMODULE_MERGE_DIAGNOSTICS_FILENAME = "submodule-merge-diagnostics.json"
SUBMODULE_MERGE_DIAGNOSTICS_MAX_ATTEMPTS = 200
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


def _copilot_has_auth() -> bool:
    """Return whether the local Copilot CLI has non-interactive auth available."""

    if any(os.environ.get(name) for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")):
        return True
    gh = shutil.which("gh")
    if not gh:
        return False
    completed = subprocess.run([gh, "auth", "status"], text=True, capture_output=True, check=False)
    return completed.returncode == 0


# Environment variable configuration for CLI tool capabilities.
# These allow tuning without code changes.
_CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
_CODEX_CONTEXT_WINDOW_ENV = "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW"
_CODEX_REASONING_EFFORT_ENV = "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
_CODEX_MAX_THREADS_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MAX_THREADS"
_CODEX_MAX_DEPTH_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MAX_DEPTH"
_COPILOT_MODEL_ENV = "IPFS_ACCELERATE_AGENT_COPILOT_MODEL"
_COPILOT_EFFORT_ENV = "IPFS_ACCELERATE_AGENT_COPILOT_EFFORT"
_COPILOT_CONTEXT_TIER_ENV = "IPFS_ACCELERATE_AGENT_COPILOT_CONTEXT_TIER"
_COPILOT_MAX_CONTINUES_ENV = "IPFS_ACCELERATE_AGENT_COPILOT_MAX_CONTINUES"


def _copilot_fallback_command(*, codex: str | None, copilot: str, workspace_path: Path) -> list[str]:
    """Build a bash command that tries Codex first, falls back to Copilot CLI.

    Both tools are invoked with full capability flags:
    - Codex: model selection, reasoning effort, context window, sub-agent threading
    - Copilot: model selection, reasoning effort, long context, autopilot with continuation limit
    """
    # Codex configuration
    codex_model = os.environ.get(_CODEX_MODEL_ENV, "").strip()
    codex_context = os.environ.get(_CODEX_CONTEXT_WINDOW_ENV, "200000").strip()
    codex_reasoning = os.environ.get(_CODEX_REASONING_EFFORT_ENV, "high").strip()
    codex_max_threads = os.environ.get(_CODEX_MAX_THREADS_ENV, "10").strip()
    codex_max_depth = os.environ.get(_CODEX_MAX_DEPTH_ENV, "2").strip()

    # Copilot configuration
    copilot_model = os.environ.get(_COPILOT_MODEL_ENV, "").strip()
    copilot_effort = os.environ.get(_COPILOT_EFFORT_ENV, "high").strip()
    copilot_context = os.environ.get(_COPILOT_CONTEXT_TIER_ENV, "long_context").strip()
    copilot_max_continues = os.environ.get(_COPILOT_MAX_CONTINUES_ENV, "30").strip()

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
codex_model="$4"
codex_context="$5"
codex_reasoning="$6"
codex_max_threads="$7"
codex_max_depth="$8"
copilot_model="$9"
copilot_effort="${10}"
copilot_context="${11}"
copilot_max_continues="${12}"

if [[ -n "$codex_bin" ]]; then
    codex_args=(exec --dangerously-bypass-approvals-and-sandbox -C "$workspace")
    # Model selection
    [[ -n "$codex_model" ]] && codex_args+=(-m "$codex_model")
    # Context window, reasoning, and sub-agent configuration via -c overrides
    [[ -n "$codex_context" ]] && codex_args+=(-c "model_context_window=$codex_context")
    [[ -n "$codex_reasoning" ]] && codex_args+=(-c "model_reasoning_effort=\"$codex_reasoning\"")
    [[ -n "$codex_max_threads" ]] && codex_args+=(-c "agents.max_threads=$codex_max_threads")
    [[ -n "$codex_max_depth" ]] && codex_args+=(-c "agents.max_depth=$codex_max_depth")
    codex_args+=(-)

    if "${codex_bin}" "${codex_args[@]}" < "$prompt_file"; then
        exit 0
    else
        rc=$?
        printf 'codex exec failed with exit %s; falling back to copilot\\n' "$rc" >&2
    fi
fi

copilot_args=(--silent --allow-all-tools --allow-all-paths --no-ask-user --autopilot)
# Model selection
[[ -n "$copilot_model" ]] && copilot_args+=(--model="$copilot_model")
# Reasoning effort
[[ -n "$copilot_effort" ]] && copilot_args+=(--effort="$copilot_effort")
# Long context tier (1M tokens)
[[ -n "$copilot_context" ]] && copilot_args+=(--context "$copilot_context")
# Autopilot continuation limit (safety cap)
[[ -n "$copilot_max_continues" ]] && copilot_args+=(--max-autopilot-continues "$copilot_max_continues")
copilot_args+=(--prompt "$(cat "$prompt_file")")

exec "$copilot_bin" "${copilot_args[@]}"
""",
        "bash",
        codex or "",
        copilot,
        str(workspace_path),
        codex_model,
        codex_context,
        codex_reasoning,
        codex_max_threads,
        codex_max_depth,
        copilot_model,
        copilot_effort,
        copilot_context,
        copilot_max_continues,
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


RETRY_BUDGET_REPAIR_TITLE_RE = re.compile(
    r"^Resolve\s+(?P<kind>validation|implementation|merge)\s+retry-budget\s+failure\s+for\s+(?P<source>[A-Z][A-Z0-9]*-\d+)\b",
    re.IGNORECASE,
)
RETRY_BUDGET_REPAIR_ACCEPTANCE_RE = re.compile(
    r"\b(?:release|remove)\s+(?P<source>[A-Z][A-Z0-9]*-\d+)\s+from\s+(?:the\s+)?strategy\s+blocked_tasks\b",
    re.IGNORECASE,
)


def retry_budget_repair_source(task: Any) -> tuple[str, str]:
    """Return ``(source_task_id, failure_kind)`` for generated retry repairs."""

    title_match = RETRY_BUDGET_REPAIR_TITLE_RE.search(str(getattr(task, "title", "") or ""))
    acceptance_match = RETRY_BUDGET_REPAIR_ACCEPTANCE_RE.search(str(getattr(task, "acceptance", "") or ""))
    if not title_match or not acceptance_match:
        return "", ""
    source_task_id = str(title_match.group("source") or "").strip()
    acceptance_source = str(acceptance_match.group("source") or "").strip()
    if source_task_id != acceptance_source:
        return "", ""
    return source_task_id, str(title_match.group("kind") or "retry").strip().lower()


def is_retry_budget_repair_task(task: Any) -> bool:
    """Return whether a task is itself a generated retry-budget repair."""

    return bool(retry_budget_repair_source(task)[0])


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


def classify_provider_capacity_failure(text: str) -> dict[str, Any]:
    """Classify provider quota/capacity failures without treating them as code failures."""

    providers = [provider for provider, pattern in PROVIDER_CAPACITY_PATTERNS if pattern.search(text)]
    unique_providers = list(dict.fromkeys(providers))
    return {
        "exhausted": bool(unique_providers),
        "providers": unique_providers,
        "reason": "provider_capacity_exhausted" if unique_providers else "",
    }


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
    canonical_task_key: str = ""
    canonical_task_cid: str = ""
    board_namespace: str = "default"


@dataclass(frozen=True)
class BundleWorkOrder:
    """A packet aggregate that can close multiple todo records after validation."""

    primary_task_id: str
    covered_task_ids: list[str]
    packet_key: str
    goal_ids: list[str]
    work_item_count: int
    index_path: str

    @property
    def task_ids(self) -> list[str]:
        return [self.primary_task_id, *self.covered_task_ids]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortalTaskState:
    heartbeat_at: str = ""
    last_progress_at: str = ""
    active_task_id: str = ""
    active_task_key: str = ""
    active_task_cid: str = ""
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
    selectable_ready_task_ids: list[str] = field(default_factory=list)
    external_reserved_task_ids: list[str] = field(default_factory=list)
    assumed_completed_task_ids: list[str] = field(default_factory=list)
    eligible_ready_task_ids: list[str] = field(default_factory=list)
    strict_deprioritized_ready_task_ids: list[str] = field(default_factory=list)
    waiting_task_ids: list[str] = field(default_factory=list)
    blocked_task_ids: list[str] = field(default_factory=list)
    task_statuses: dict[str, str] = field(default_factory=dict)
    task_artifacts: dict[str, list[str]] = field(default_factory=dict)
    task_validation: dict[str, list[str]] = field(default_factory=dict)
    task_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    implementation_attempts: dict[str, int] = field(default_factory=dict)
    implementation_attempts_by_cid: dict[str, int] = field(default_factory=dict)
    last_implementation_task_id: str = ""
    last_implementation_task_key: str = ""
    last_implementation_task_cid: str = ""
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
    selectable_ready_count: int = 0
    external_reserved_count: int = 0
    assumed_completed_count: int = 0
    eligible_ready_count: int = 0
    strict_deprioritized_ready_count: int = 0
    waiting_count: int = 0
    blocked_count: int = 0
    task_count: int = 0
    strategy_generation: int = 0
    selection_idle_reason: str = ""

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
                active_task_key=str(payload.get("active_task_key") or ""),
                active_task_cid=str(payload.get("active_task_cid") or ""),
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
                selectable_ready_task_ids=[
                    str(item) for item in payload.get("selectable_ready_task_ids", []) or []
                ],
                external_reserved_task_ids=[
                    str(item) for item in payload.get("external_reserved_task_ids", []) or []
                ],
                assumed_completed_task_ids=[
                    str(item) for item in payload.get("assumed_completed_task_ids", []) or []
                ],
                eligible_ready_task_ids=[str(item) for item in payload.get("eligible_ready_task_ids", []) or []],
                strict_deprioritized_ready_task_ids=[
                    str(item) for item in payload.get("strict_deprioritized_ready_task_ids", []) or []
                ],
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
                task_identities={
                    str(key): {str(item_key): str(item_value) for item_key, item_value in value.items()}
                    for key, value in (payload.get("task_identities") or {}).items()
                    if isinstance(value, dict)
                },
                implementation_attempts={
                    str(key): int(value)
                    for key, value in (payload.get("implementation_attempts") or {}).items()
                    if str(value).isdigit()
                },
                implementation_attempts_by_cid={
                    str(key): int(value)
                    for key, value in (payload.get("implementation_attempts_by_cid") or {}).items()
                    if str(value).isdigit()
                },
                last_implementation_task_id=str(payload.get("last_implementation_task_id") or ""),
                last_implementation_task_key=str(payload.get("last_implementation_task_key") or ""),
                last_implementation_task_cid=str(payload.get("last_implementation_task_cid") or ""),
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
                selectable_ready_count=int(payload.get("selectable_ready_count") or 0),
                external_reserved_count=int(payload.get("external_reserved_count") or 0),
                assumed_completed_count=int(payload.get("assumed_completed_count") or 0),
                eligible_ready_count=int(payload.get("eligible_ready_count") or 0),
                strict_deprioritized_ready_count=int(payload.get("strict_deprioritized_ready_count") or 0),
                waiting_count=int(payload.get("waiting_count") or 0),
                blocked_count=int(payload.get("blocked_count") or 0),
                task_count=int(payload.get("task_count") or 0),
                strategy_generation=int(payload.get("strategy_generation") or 0),
                selection_idle_reason=str(payload.get("selection_idle_reason") or ""),
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
        "selectable_ready_count",
        "eligible_ready_count",
        "strict_deprioritized_ready_count",
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
    for field_name in (
        "task_statuses",
        "task_artifacts",
        "task_validation",
        "task_identities",
        "implementation_attempts",
        "implementation_attempts_by_cid",
    ):
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
        if not metadata:
            metadata["blocked reason"] = "empty task metadata"
        default_status = "blocked" if metadata.get("blocked reason") == "empty task metadata" else "todo"
        identity = canonical_task_identity(
            {
                "task_id": current_id,
                "title": current_title,
                "outputs": split_csv(metadata.get("outputs", "")),
                "acceptance": str(metadata.get("acceptance", "")).strip(),
                "metadata": metadata,
            },
            board_namespace=metadata.get("board namespace", "") or path.name,
            source_path=path,
        )
        tasks.append(
            PortalTask(
                task_id=current_id,
                title=current_title,
                status=normalize_status(metadata.get("status", default_status)),
                completion=str(metadata.get("completion", "manual")).strip().lower(),
                priority=str(metadata.get("priority", "P2")).strip().upper(),
                track=str(metadata.get("track", "ops")).strip().lower(),
                depends_on=split_csv(metadata.get("depends on", "")),
                outputs=split_csv(metadata.get("outputs", "")),
                validation=split_validation_commands(metadata.get("validation", "")),
                acceptance=str(metadata.get("acceptance", "")).strip(),
                source_line=current_line,
                metadata=dict(metadata),
                canonical_task_key=identity.canonical_task_key,
                canonical_task_cid=identity.canonical_task_cid,
                board_namespace=identity.board_namespace,
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
        merge_target_branch: str | None = None,
        worktree_submodule_paths: Any = None,
        objective_path: Path | None = None,
        objective_bundle_dir: Path | None = None,
        generated_status_paths: Sequence[Path | str] = (),
        external_reservation_manifest_paths: Sequence[Path | str] = (),
        assumed_completed_task_ids: Sequence[str] = (),
        llm_merge_resolver_command: str | None = None,
        llm_merge_resolver_timeout_seconds: float | None = None,
        merge_reconciliation_max_merges: int | None = None,
        merge_reconciliation_max_age_seconds: int | None = None,
        merged_worktree_cleanup_max: int | None = None,
        task_shard_count: int = 1,
        task_shard_index: int = 0,
        merge_queue: MergeQueue | None = None,
        merge_queue_dir: Path | None = None,
        validation_scheduler: ValidationScheduler | None = None,
        validation_cache_dir: Path | None = None,
        validation_max_workers: int | None = None,
        validation_resource_budget: int | None = None,
        formal_verification_policy: Any = None,
        proof_gate: Any = None,
        proof_cache_dir: Path | None = None,
        worktree_pool_enabled: bool | None = None,
        worktree_pool_max_entries: int | None = None,
        worktree_pool: WorktreePool | None = None,
        maintenance_interval_seconds: float | None = None,
    ) -> None:
        self.todo_path = todo_path
        self.state_path = state_path
        self.strategy_path = strategy_path
        self.events_path = events_path
        self.repo_root = (repo_root or REPO_ROOT).resolve()
        self.shared_worktree_source_roots = shared_worktree_source_roots(self.repo_root)
        self.task_header_prefix = normalize_task_header_prefix(task_header_prefix)
        self.implement = implement
        self.implementation_command = implementation_command
        self.implementation_timeout = implementation_timeout
        self.implementation_log_dir = implementation_log_dir or self.state_path.parent / "implementation_logs"
        self.use_ephemeral_worktree = use_ephemeral_worktree
        configured_worktree_root = worktree_root or Path(tempfile.gettempdir()) / "211-ai-implementation-worktrees"
        # The implementation runner executes with the ephemeral worktree as
        # its cwd.  Keep the path supplied to Codex/Copilot absolute so a
        # relative --worktree-root cannot be resolved a second time below it.
        if not configured_worktree_root.is_absolute():
            configured_worktree_root = self.repo_root / configured_worktree_root
        self.worktree_root = configured_worktree_root.resolve()
        self.worktree_pool_enabled = (
            _env_bool(WORKTREE_POOL_ENABLED_ENV, True)
            if worktree_pool_enabled is None
            else bool(worktree_pool_enabled)
        )
        configured_pool_size = (
            _env_int(WORKTREE_POOL_MAX_ENTRIES_ENV, DEFAULT_WORKTREE_POOL_MAX_ENTRIES)
            if worktree_pool_max_entries is None
            else int(worktree_pool_max_entries)
        )
        self.worktree_pool = worktree_pool
        if self.worktree_pool is None and self.worktree_pool_enabled:
            self.worktree_pool = WorktreePool(
                repo_root=self.repo_root,
                worktree_root=self.worktree_root,
                max_entries=max(1, configured_pool_size),
            )
        self._worktree_pool_leases: dict[Path, WorktreeLease] = {}
        self._worktree_pool_effective_paths: dict[Path, Path] = {}
        self._worktree_setup_metrics: dict[Path, dict[str, Any]] = {}
        self.merge_target_branch = str(merge_target_branch or "").strip()
        self.objective_path = objective_path
        self.objective_bundle_dir = objective_bundle_dir
        self.generated_status_paths = tuple(Path(path) for path in generated_status_paths)
        self.external_reservation_manifest_paths = tuple(
            Path(path).resolve() for path in external_reservation_manifest_paths
        )
        self.assumed_completed_task_ids = frozenset(
            str(task_id).strip()
            for task_id in assumed_completed_task_ids
            if str(task_id).strip()
        )
        self.llm_merge_resolver_command = (
            default_llm_merge_resolver_command()
            if llm_merge_resolver_command is None
            else llm_merge_resolver_command
        ).strip()
        self.llm_merge_resolver_timeout_seconds = llm_merge_resolver_timeout_seconds
        self.merge_reconciliation_max_merges = (
            _env_int(DAEMON_MERGE_RECONCILIATION_MAX_ENV, DEFAULT_DAEMON_MERGE_RECONCILIATION_MAX)
            if merge_reconciliation_max_merges is None
            else int(merge_reconciliation_max_merges)
        )
        self.merge_reconciliation_max_age_seconds = (
            _env_int(MERGE_RECONCILIATION_MAX_AGE_ENV, DEFAULT_MERGE_RECONCILIATION_MAX_AGE_SECONDS)
            if merge_reconciliation_max_age_seconds is None
            else int(merge_reconciliation_max_age_seconds)
        )
        self.merged_worktree_cleanup_max = (
            _env_int(DAEMON_MERGED_WORKTREE_CLEANUP_MAX_ENV, DEFAULT_DAEMON_MERGED_WORKTREE_CLEANUP_MAX)
            if merged_worktree_cleanup_max is None
            else int(merged_worktree_cleanup_max)
        )
        self.task_shard_count = max(1, int(task_shard_count))
        self.task_shard_index = int(task_shard_index)
        if self.task_shard_index < 0 or self.task_shard_index >= self.task_shard_count:
            raise ValueError("task_shard_index must be in range [0, task_shard_count)")
        self.maintenance_interval_seconds = max(
            0.0,
            _env_float(
                DAEMON_MAINTENANCE_INTERVAL_ENV,
                DEFAULT_DAEMON_MAINTENANCE_INTERVAL_SECONDS,
            )
            if maintenance_interval_seconds is None
            else float(maintenance_interval_seconds),
        )
        self._last_periodic_maintenance_monotonic: float | None = None
        # Lane state directories are intentionally isolated, so the merge train
        # cannot live next to ``state_path``.  The git common directory is shared
        # by every worktree and supervisor lane for this repository.
        default_merge_queue_dir = checkout_mutation_lock_path(self.repo_root).parent / "agent-merge-train"
        self.merge_queue_dir = merge_queue_dir or default_merge_queue_dir
        self.merge_queue = merge_queue or MergeQueue(self.merge_queue_dir)
        configured_validation_workers = (
            _env_int(VALIDATION_MAX_WORKERS_ENV, DEFAULT_VALIDATION_MAX_WORKERS)
            if validation_max_workers is None
            else int(validation_max_workers)
        )
        configured_validation_budget = (
            _env_int(VALIDATION_RESOURCE_BUDGET_ENV, configured_validation_workers)
            if validation_resource_budget is None
            else int(validation_resource_budget)
        )
        # This directory lives beside the shared checkout mutation lock so all
        # isolated lanes and merge consumers reuse the same content-addressed
        # successes without sharing mutable scheduler state.
        default_validation_cache_dir = (
            checkout_mutation_lock_path(self.repo_root).parent / "agent-validation-cache"
        )
        self.validation_cache_dir = validation_cache_dir or default_validation_cache_dir
        self.validation_scheduler = validation_scheduler or ValidationScheduler(
            cache_dir=self.validation_cache_dir,
            max_workers=max(1, configured_validation_workers),
            resource_budget=max(1, configured_validation_budget),
            default_timeout_seconds=self.implementation_timeout,
        )
        self.formal_verification_policy = formal_verification_policy
        self.proof_gate = proof_gate
        self.proof_cache_dir = (
            Path(proof_cache_dir) if proof_cache_dir is not None else None
        )
        configured_submodules = (
            DEFAULT_WORKTREE_SUBMODULE_PATHS
            if worktree_submodule_paths is None
            else normalize_relative_path_list(worktree_submodule_paths)
        )
        self.worktree_submodule_paths = configured_submodules
        self.submodule_merge_diagnostics_path = (
            self.state_path.parent / SUBMODULE_MERGE_DIAGNOSTICS_FILENAME
        )
        self.degradation_state = DegradationState.load(
            self.state_path.parent / "submodule_degradation.json"
        )
        self.task_queue = PersistentTaskQueue.load(
            self.state_path.parent / "task_queue.json"
        )
        self._task_identity_by_display_id: dict[str, TaskIdentity] = {}
        self._active_canonical_task_cids: set[str] = set()
        self.git_gc = GitGarbageCollector(
            repo_root=self.repo_root,
            worktree_root=self.worktree_root if hasattr(self, "worktree_root") else None,
        )

    def _identity_for_task(self, task: PortalTask) -> TaskIdentity:
        metadata = dict(task.metadata)
        if task.canonical_task_key:
            metadata["canonical task key"] = task.canonical_task_key
        if task.canonical_task_cid:
            metadata["canonical task cid"] = task.canonical_task_cid
        return canonical_task_identity(
            {
                "task_id": task.task_id,
                "title": task.title,
                "outputs": task.outputs,
                "acceptance": task.acceptance,
                "metadata": metadata,
            },
            board_namespace=task.board_namespace or self.todo_path.name,
            source_path=self.todo_path,
        )

    def _register_task_identities(self, tasks: Sequence[PortalTask]) -> dict[str, list[str]]:
        aliases_by_cid: dict[str, list[str]] = {}
        self._task_identity_by_display_id = {}
        for task in tasks:
            identity = self._identity_for_task(task)
            self._task_identity_by_display_id[task.task_id] = identity
            aliases_by_cid.setdefault(identity.canonical_task_cid, []).append(task.task_id)
            self.task_queue.register_task(identity, priority=task.priority, track=task.track)
        self._active_canonical_task_cids = set(aliases_by_cid)
        self.task_queue.compact(self._active_canonical_task_cids)
        if self.task_queue.dirty:
            self.task_queue.save()
        duplicates = {
            canonical_task_cid: task_ids
            for canonical_task_cid, task_ids in aliases_by_cid.items()
            if len(task_ids) > 1
        }
        if duplicates:
            self._record_event("canonical_task_aliases_coalesced", {"aliases_by_cid": duplicates})
        return aliases_by_cid

    def _canonical_ref(self, task: PortalTask) -> str:
        return self._identity_for_task(task).canonical_task_cid

    def _task_attempt(self, state: PortalTaskState, task: PortalTask) -> int:
        canonical_task_cid = self._canonical_ref(task)
        prior = max(
            state.implementation_attempts.get(task.task_id, 0),
            state.implementation_attempts_by_cid.get(canonical_task_cid, 0),
        )
        return prior + 1

    def _record_task_attempt(self, state: PortalTaskState, task: PortalTask, attempt: int) -> None:
        identity = self._identity_for_task(task)
        state.implementation_attempts[task.task_id] = attempt
        state.implementation_attempts_by_cid[identity.canonical_task_cid] = attempt
        state.last_implementation_task_id = task.task_id
        state.last_implementation_task_key = identity.canonical_task_key
        state.last_implementation_task_cid = identity.canonical_task_cid

    def _record_task_queue_outcome(self, task: PortalTask, returncode: int, reason: str = "") -> None:
        canonical_task_cid = self._canonical_ref(task)
        if returncode == 0:
            self.task_queue.record_success(canonical_task_cid)
        else:
            self.task_queue.record_failure(canonical_task_cid, reason=reason)
        self.task_queue.save()

    def _shared_merge_queue_task_cids(self, method_name: str) -> set[str]:
        method = getattr(self.merge_queue, method_name, None)
        if not callable(method):
            return set()
        try:
            return {str(item) for item in method() if str(item)}
        except Exception as exc:
            self._record_event(
                "shared_merge_receipts_unavailable",
                {
                    "query": method_name,
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[-4000:],
                },
            )
            return set()

    @staticmethod
    def _canonical_representative_task_ids(
        tasks: Sequence[PortalTask],
        resolved_statuses: dict[str, str],
    ) -> set[str]:
        status_rank = {"ready": 0, "waiting": 1, "blocked": 2, "completed": 3}
        representative: dict[str, PortalTask] = {}
        for task in tasks:
            key = task.canonical_task_cid or task.canonical_task_key or task.task_id
            current = representative.get(key)
            if current is None:
                representative[key] = task
                continue
            candidate_rank = (status_rank.get(resolved_statuses.get(task.task_id, ""), 9), task.source_line)
            current_rank = (status_rank.get(resolved_statuses.get(current.task_id, ""), 9), current.source_line)
            if candidate_rank < current_rank:
                representative[key] = task
        return {task.task_id for task in representative.values()}

    def _task_belongs_to_shard(self, task_id: str) -> bool:
        """Return whether this daemon lane should implement ``task_id``."""

        if self.task_shard_count <= 1:
            return True
        match = re.search(r"(\d+)$", task_id)
        if match is None:
            return self.task_shard_index == 0
        return int(match.group(1)) % self.task_shard_count == self.task_shard_index

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
        merged["focus_tracks"] = normalize_focus_tracks(merged.get("focus_tracks", DEFAULT_TRACKS))
        merged["blocked_tasks"] = [str(item) for item in merged.get("blocked_tasks", [])]
        merged["deprioritized_tasks"] = [str(item) for item in merged.get("deprioritized_tasks", [])]
        return merged

    def _mark_long_running_phase(self, *, task_id: str, phase: str, detail: str = "") -> None:
        state = PortalTaskState.load(self.state_path)
        now = utc_now()
        if task_id:
            state.active_task_id = task_id
            identity = self._task_identity_by_display_id.get(task_id)
            if identity is not None:
                state.active_task_key = identity.canonical_task_key
                state.active_task_cid = identity.canonical_task_cid
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
        state.selectable_ready_task_ids = []
        state.eligible_ready_task_ids = []
        state.strict_deprioritized_ready_task_ids = []
        state.waiting_task_ids = []
        state.blocked_task_ids = []
        state.completed_count = 0
        state.ready_count = 0
        state.selectable_ready_count = 0
        state.eligible_ready_count = 0
        state.strict_deprioritized_ready_count = 0
        state.waiting_count = 0
        state.blocked_count = 0
        state.task_count = 0
        state.task_statuses = {}
        state.task_artifacts = {}
        state.task_validation = {}
        state.task_identities = {}
        state.strategy_generation = int(strategy.get("generation", 0))
        if not (previous.implementation_in_progress and live_inflight_implementation is not None):
            state.active_task_id = ""
            state.active_task_key = ""
            state.active_task_cid = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            self._clear_active_execution_state(state)
            state.recommended_task_id = ""
            state.recommended_actions = []
        state.selection_idle_reason = reason
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
            "selectable_ready_count": 0,
            "eligible_ready_count": 0,
            "strict_deprioritized_ready_count": 0,
            "waiting_count": 0,
            "blocked_count": 0,
            "active_task_id": state.active_task_id,
            "selection_idle_reason": reason,
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
        aliases_by_cid = self._register_task_identities(tasks)
        merge_train_progress: dict[str, Any] | None = None
        try:
            merge_train_progress = self._consume_one_merge_candidate()
        except Exception as exc:
            self._record_event(
                "merge_train_consumer_deferred",
                {
                    "reason": "merge_train_consumer_unavailable",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[-4000:],
                },
            )
        shared_active_merge_cids = self._shared_merge_queue_task_cids(
            "active_canonical_task_ids"
        )
        shared_completed_merge_cids = self._shared_merge_queue_task_cids(
            "completed_canonical_task_ids"
        )
        shared_active_merge_cids.difference_update(shared_completed_merge_cids)
        shared_completed_task_ids = {
            task.task_id
            for task in tasks
            if self._canonical_ref(task) in shared_completed_merge_cids
        }
        shared_active_merge_task_ids = {
            task.task_id
            for task in tasks
            if self._canonical_ref(task) in shared_active_merge_cids
        }
        previous = PortalTaskState.load(self.state_path)
        strategy = self.load_strategy()
        now = utc_now()
        board_completed_task_ids = {
            task.task_id for task in tasks if task.status == "completed"
        }
        status_completed_task_ids = board_completed_task_ids | shared_completed_task_ids
        strategy_blocked_task_ids = {str(task_id) for task_id in strategy.get("blocked_tasks", [])}
        # A historical deprioritization is only a scheduling hint.  Failed
        # implementation merges must still be reconciled unless the janitor
        # explicitly retired the task as off-mission.
        strategy_deprioritized_task_ids = self._strict_off_mission_deprioritized_task_ids(strategy)
        merge_skip_task_ids = status_completed_task_ids | strategy_blocked_task_ids
        live_inflight_implementation = self._find_live_inflight_implementation()
        if previous.implementation_in_progress and live_inflight_implementation is None:
            recovered_state = PortalTaskState.load(self.state_path)
            self._clear_active_execution_state(recovered_state)
            recovered_state.save(self.state_path)
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
            previous = recovered_state
        merge_reconciliation = self._reconcile_failed_merges(
            skip_task_ids=merge_skip_task_ids,
            deprioritized_task_ids=strategy_deprioritized_task_ids,
        )
        merged_worktree_cleanup = self._cleanup_already_merged_worktrees()
        self._periodic_maintenance()
        unresolved_merge_failures = self._unresolved_merge_failures_by_task(skip_task_ids=merge_skip_task_ids)
        unresolved_merge_failure_task_ids = set(unresolved_merge_failures)
        transient_merge_deferrals = self._transient_merge_deferrals_by_task(skip_task_ids=merge_skip_task_ids)
        transient_merge_deferral_task_ids = set(transient_merge_deferrals)
        recent_outcomes = self._latest_implementation_finished_by_task()
        queued_merge_task_ids = self._pending_queued_merge_task_ids(recent_outcomes)
        quarantined_merge_task_ids = self._quarantined_queued_merge_task_ids(recent_outcomes)
        successfully_merged_task_ids = self._successfully_merged_task_ids()
        completion_receipt_task_ids = successfully_merged_task_ids | shared_completed_task_ids
        merged_status_repair: dict[str, Any] = {}
        stale_merged_completed_task_ids = [
            task.task_id
            for task in tasks
            if task.task_id in completion_receipt_task_ids and task.task_id not in board_completed_task_ids
        ]
        if stale_merged_completed_task_ids:
            merged_status_repair = self._mark_tasks_completed_in_todo(
                stale_merged_completed_task_ids,
                primary_task_id=stale_merged_completed_task_ids[0],
                completion_reason="merged_status_repair",
            )
            repaired_task_ids = set(merged_status_repair.get("updated_task_ids") or [])
            repaired_task_ids.update(merged_status_repair.get("already_completed_task_ids") or [])
            board_completed_task_ids.update(repaired_task_ids)
            status_completed_task_ids.update(repaired_task_ids)

        previous_completed = set(previous.completed_task_ids)
        completed_set: set[str] = set()
        newly_completed: list[str] = []
        resolved_statuses: dict[str, str] = {}
        task_artifacts: dict[str, list[str]] = {}

        for task in tasks:
            existing_outputs = [item for item in task.outputs if (self.repo_root / item).exists()]
            task_artifacts[task.task_id] = existing_outputs
            if self._has_unresolved_merge_failure(task, previous):
                unresolved_merge_failure_task_ids.add(task.task_id)
            unresolved_merge_failure = task.task_id in unresolved_merge_failure_task_ids
            transient_merge_deferral = task.task_id in transient_merge_deferral_task_ids
            artifact_complete = (
                task.completion == "artifact"
                and bool(task.outputs)
                and len(existing_outputs) == len(task.outputs)
                and not unresolved_merge_failure
                and not transient_merge_deferral
            )
            merged_complete = (
                task.task_id in successfully_merged_task_ids
                and not unresolved_merge_failure
                and not transient_merge_deferral
            )
            if task.task_id in status_completed_task_ids or artifact_complete or merged_complete:
                completed_set.add(task.task_id)

        completed_cids = {
            self._canonical_ref(task)
            for task in tasks
            if task.task_id in completed_set
        }
        completed_set.update(
            task.task_id
            for task in tasks
            if self._canonical_ref(task) in completed_cids
        )
        dependency_satisfied_task_ids = completed_set | self.assumed_completed_task_ids

        for task in tasks:
            if task.task_id in completed_set:
                resolved_statuses[task.task_id] = "completed"
                if task.task_id not in previous_completed:
                    newly_completed.append(task.task_id)
                continue
            if task.task_id in strategy.get("blocked_tasks", []) or task.status == "blocked":
                resolved_statuses[task.task_id] = "blocked"
                continue
            if task.task_id in shared_active_merge_task_ids:
                resolved_statuses[task.task_id] = "waiting"
                continue
            if task.task_id in transient_merge_deferral_task_ids:
                resolved_statuses[task.task_id] = "waiting"
                continue
            if task.task_id in queued_merge_task_ids:
                resolved_statuses[task.task_id] = "waiting"
                continue
            if task.task_id in quarantined_merge_task_ids:
                resolved_statuses[task.task_id] = "blocked"
                continue
            if task.task_id in unresolved_merge_failure_task_ids:
                resolved_statuses[task.task_id] = "blocked"
                continue
            unresolved_deps = [
                dep for dep in task.depends_on if dep not in dependency_satisfied_task_ids
            ]
            if unresolved_deps:
                resolved_statuses[task.task_id] = "waiting"
                continue
            resolved_statuses[task.task_id] = "ready"

        representative_task_ids = self._canonical_representative_task_ids(tasks, resolved_statuses)
        active_task_claims = self._active_implementation_task_claims(tasks)
        external_task_reservations = self._external_task_reservations(tasks)
        for task_id, reservation in external_task_reservations.items():
            active_task_claims.setdefault(task_id, reservation)
        selectable_tasks = [
            task
            for task in tasks
            if (
                task.task_id in representative_task_ids
                and self._task_belongs_to_shard(task.task_id)
                and task.task_id not in active_task_claims
            )
        ]
        if self.task_shard_count > 1 and not any(
            resolved_statuses.get(task.task_id) == "ready" for task in selectable_tasks
        ):
            fallback_tasks = [
                task
                for task in tasks
                if (
                    task.task_id in representative_task_ids
                    and task.task_id not in active_task_claims
                    and resolved_statuses.get(task.task_id) == "ready"
                )
            ]
            if fallback_tasks:
                self._record_event(
                    "task_shard_ready_fallback",
                    {
                        "task_shard_count": self.task_shard_count,
                        "task_shard_index": self.task_shard_index,
                        "fallback_task_ids": [task.task_id for task in fallback_tasks[:20]],
                    },
                )
                selectable_tasks = fallback_tasks
        selected = self._select_next_task(
            selectable_tasks,
            resolved_statuses,
            strategy,
            unresolved_merge_failures,
            recent_outcomes,
        )
        selection_scope = self._selection_scope(selectable_tasks, resolved_statuses, strategy)
        state = PortalTaskState.load(self.state_path)
        state.heartbeat_at = now
        if newly_completed or not state.last_progress_at:
            state.last_progress_at = now
        state.completed_task_ids = sorted(completed_set)
        state.completed_count = len(state.completed_task_ids)
        state.ready_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "ready"]
        state.selectable_ready_task_ids = list(selection_scope["selectable_ready_task_ids"])
        state.external_reserved_task_ids = sorted(external_task_reservations)
        state.assumed_completed_task_ids = sorted(self.assumed_completed_task_ids)
        state.eligible_ready_task_ids = list(selection_scope["eligible_ready_task_ids"])
        state.strict_deprioritized_ready_task_ids = list(selection_scope["strict_deprioritized_ready_task_ids"])
        state.waiting_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "waiting"]
        state.blocked_task_ids = [task.task_id for task in tasks if resolved_statuses[task.task_id] == "blocked"]
        state.ready_count = len(state.ready_task_ids)
        state.selectable_ready_count = len(state.selectable_ready_task_ids)
        state.external_reserved_count = len(state.external_reserved_task_ids)
        state.assumed_completed_count = len(state.assumed_completed_task_ids)
        state.eligible_ready_count = len(state.eligible_ready_task_ids)
        state.strict_deprioritized_ready_count = len(state.strict_deprioritized_ready_task_ids)
        state.waiting_count = len(state.waiting_task_ids)
        state.blocked_count = len(state.blocked_task_ids)
        state.task_count = len(tasks)
        state.task_statuses = resolved_statuses
        state.task_artifacts = task_artifacts
        state.task_validation = {task.task_id: task.validation for task in tasks if task.validation}
        state.task_identities = {
            task.task_id: self._identity_for_task(task).to_dict()
            for task in tasks
        }
        state.strategy_generation = int(strategy.get("generation", 0))
        state.implementation_attempts = previous.implementation_attempts
        state.implementation_attempts_by_cid = previous.implementation_attempts_by_cid
        state.active_attempt = previous.active_attempt
        state.active_phase = previous.active_phase
        state.active_phase_started_at = previous.active_phase_started_at
        state.active_phase_detail = previous.active_phase_detail
        state.active_log_path = previous.active_log_path
        state.active_worktree_path = previous.active_worktree_path
        state.active_branch = previous.active_branch
        state.implementation_in_progress = previous.implementation_in_progress
        state.last_implementation_task_id = previous.last_implementation_task_id
        state.last_implementation_task_key = previous.last_implementation_task_key
        state.last_implementation_task_cid = previous.last_implementation_task_cid
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
            selected_identity = self._identity_for_task(selected)
            state.active_task_key = selected_identity.canonical_task_key
            state.active_task_cid = selected_identity.canonical_task_cid
            state.active_task_title = selected.title
            state.active_task_track = selected.track
            state.recommended_task_id = selected.task_id
            state.recommended_actions = self._build_recommended_actions(selected)
            state.selection_idle_reason = ""
        else:
            state.active_task_id = ""
            state.active_task_key = ""
            state.active_task_cid = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            self._clear_active_execution_state(state)
            state.recommended_task_id = ""
            state.recommended_actions = []
            state.selection_idle_reason = str(selection_scope["selection_idle_reason"])

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
                "selectable_ready_count": state.selectable_ready_count,
                "eligible_ready_count": state.eligible_ready_count,
                "strict_deprioritized_ready_count": state.strict_deprioritized_ready_count,
                "selection_idle_reason": state.selection_idle_reason,
                "shared_active_merge_task_ids": sorted(shared_active_merge_task_ids),
                "shared_completed_task_ids": sorted(shared_completed_task_ids),
            },
        )
        return {
            "task_count": state.task_count,
            "completed_count": state.completed_count,
            "ready_count": state.ready_count,
            "selectable_ready_count": state.selectable_ready_count,
            "eligible_ready_count": state.eligible_ready_count,
            "strict_deprioritized_ready_count": state.strict_deprioritized_ready_count,
            "waiting_count": state.waiting_count,
            "blocked_count": state.blocked_count,
            "active_task_id": state.active_task_id,
            "selection_idle_reason": state.selection_idle_reason,
            "state_path": str(self.state_path),
            "strategy_path": str(self.strategy_path),
            "events_path": str(self.events_path),
            "implementation_result": implementation_result,
            "merge_reconciliation": merge_reconciliation,
            "merged_worktree_cleanup": merged_worktree_cleanup,
            "event_log_repair": event_log_repair,
            "state_file_repair": state_file_repair,
            "merged_status_repair": merged_status_repair,
            "active_task_claims": sorted(active_task_claims),
            "external_reserved_task_ids": sorted(external_task_reservations),
            "assumed_completed_task_ids": sorted(self.assumed_completed_task_ids),
            "shared_active_merge_task_ids": sorted(shared_active_merge_task_ids),
            "shared_completed_task_ids": sorted(shared_completed_task_ids),
            "canonical_task_count": len(aliases_by_cid),
            "merge_train_progress": merge_train_progress,
        }

    @staticmethod
    def _provider_capacity_backoff_seconds() -> float:
        raw = os.environ.get(
            PROVIDER_CAPACITY_BACKOFF_ENV,
            str(DEFAULT_PROVIDER_CAPACITY_BACKOFF_SECONDS),
        ).strip()
        try:
            return max(10.0, float(raw))
        except ValueError:
            return DEFAULT_PROVIDER_CAPACITY_BACKOFF_SECONDS

    def _provider_capacity_failure_from_log(self, log_path: Path) -> dict[str, Any]:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - PROVIDER_CAPACITY_LOG_TAIL_BYTES))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return {"exhausted": False, "providers": [], "reason": ""}
        classified = classify_provider_capacity_failure(text)
        if not classified["exhausted"]:
            return classified
        evidence = [
            line.strip()
            for line in text.splitlines()
            if any(pattern.search(line) for _provider, pattern in PROVIDER_CAPACITY_PATTERNS)
        ]
        classified["evidence"] = evidence[-4:]
        return classified

    def _active_provider_capacity_backoff(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        for event in reversed(self._iter_events()):
            event_type = str(event.get("type") or "")
            if event_type == "implementation_provider_exhausted":
                retry_at = parse_timestamp(str(event.get("retry_at") or ""))
                if retry_at is not None and retry_at > now:
                    return {
                        "active": True,
                        "retry_at": retry_at.isoformat(),
                        "retry_after_seconds": max(0.0, (retry_at - now).total_seconds()),
                        "providers": list(event.get("providers") or []),
                    }
                return {}
            if event_type == "implementation_finished" and int(event.get("returncode") or 0) == 0:
                return {}
        return {}

    def _record_provider_capacity_deferral(
        self,
        *,
        task: PortalTask,
        state: PortalTaskState,
        attempt: int,
        started_at: str,
        returncode: int,
        log_path: Path,
        failure: dict[str, Any],
        worktree_path: Path | None = None,
        branch_name: str = "",
        cleanup_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finished_at = utc_now()
        retry_at = datetime.fromtimestamp(
            time.time() + self._provider_capacity_backoff_seconds(),
            tz=timezone.utc,
        ).isoformat()
        state.last_implementation_started_at = started_at
        state.last_implementation_finished_at = finished_at
        state.last_implementation_returncode = returncode
        state.last_implementation_log_path = str(log_path)
        if worktree_path is not None:
            state.last_implementation_worktree_path = str(worktree_path)
        state.last_implementation_branch = branch_name
        self._mark_implementation_finished(state, finished_at=finished_at)
        state.save(self.state_path)
        result = {
            "task_id": task.task_id,
            "attempt": attempt,
            "returncode": returncode,
            "log_path": str(log_path),
            "deferred": True,
            "reason": "provider_capacity_exhausted",
            "providers": list(failure.get("providers") or []),
            "evidence": list(failure.get("evidence") or []),
            "retry_at": retry_at,
            "attempt_consumed": False,
        }
        if worktree_path is not None:
            result["worktree_path"] = str(worktree_path)
        if branch_name:
            result["branch"] = branch_name
        if cleanup_result:
            result["cleanup_result"] = cleanup_result
        self._record_event("implementation_provider_exhausted", result)
        return result

    def _run_implementation(self, task: PortalTask, state: PortalTaskState) -> dict[str, Any]:
        provider_backoff = self._active_provider_capacity_backoff()
        if provider_backoff:
            result = {
                "skipped": True,
                "reason": "provider_capacity_backoff",
                "task_id": task.task_id,
                "attempt": self._task_attempt(state, task),
                **provider_backoff,
            }
            self._record_event("implementation_skipped", result)
            return result
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
        attempt = self._task_attempt(state, task)
        task_claim_path = self._implementation_task_claim_path(
            task.task_id,
            canonical_task_cid=self._canonical_ref(task),
        )
        legacy_task_claim_path = self._implementation_task_claim_path(task.task_id)
        if legacy_task_claim_path != task_claim_path and legacy_task_claim_path.exists():
            legacy_claim = load_json_dict(legacy_task_claim_path)
            if legacy_claim is not None and self._implementation_task_claim_owner_is_active(legacy_claim):
                result = {
                    "skipped": True,
                    "reason": "task_claim_lock_exists",
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "lock_owner_pid": int(legacy_claim.get("pid") or 0),
                    "lock_owner_task_id": str(legacy_claim.get("task_id") or ""),
                    "lock_owner_state_dir": str(legacy_claim.get("state_dir") or ""),
                }
                self._record_event("implementation_skipped", result)
                return result
        task_claim_metadata = self._build_implementation_task_claim_metadata(task, attempt, started_at)
        lock_path = self._implementation_lock_path()
        lock_metadata = self._build_implementation_lock_metadata(task, attempt, started_at)
        task_claim_fd, task_claim_reason, existing_task_claim = self._try_acquire_lock(
            task_claim_path,
            lock_kind=IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
            owner_active=self._implementation_task_claim_owner_is_active,
        )
        if task_claim_fd is None:
            result = {
                "skipped": True,
                "reason": f"task_claim_{task_claim_reason}",
                "task_id": task.task_id,
                "attempt": attempt,
            }
            if existing_task_claim:
                result["lock_owner_pid"] = int(existing_task_claim.get("pid") or 0)
                result["lock_owner_task_id"] = str(existing_task_claim.get("task_id") or "")
                result["lock_owner_state_dir"] = str(existing_task_claim.get("state_dir") or "")
            self._record_event("implementation_skipped", result)
            return result

        acquired_task_claim = True
        lock_fd: int | None = None
        acquired_lock = False
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
            self._write_lock_metadata(task_claim_fd, task_claim_metadata)
            task_claim_fd = None
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
            self._write_lock_metadata(lock_fd, lock_metadata)
            lock_fd = None
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
                completed = run_process_group_stream(
                    command,
                    cwd=workspace_path,
                    stdout=log_fh,
                    input_text=prompt,
                    timeout_seconds=self.implementation_timeout,
                )
            effective_returncode = completed.returncode
            if completed.returncode != 0:
                provider_failure = self._provider_capacity_failure_from_log(log_path)
                if provider_failure.get("exhausted", False):
                    return self._record_provider_capacity_deferral(
                        task=task,
                        state=state,
                        attempt=attempt,
                        started_at=started_at,
                        returncode=completed.returncode,
                        log_path=log_path,
                        failure=provider_failure,
                    )
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
                todo_update_result = self._mark_task_or_bundle_completed_in_todo(task)
            finished_at = utc_now()
            self._record_task_attempt(state, task, attempt)
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = effective_returncode
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            self._record_task_queue_outcome(task, effective_returncode, reason="validation_or_implementation_failed")
            result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": effective_returncode,
                "log_path": str(log_path),
                "validation_result": validation_result,
            }
            termination_result = self._implementation_returncode_detail(effective_returncode)
            if termination_result:
                result["termination_result"] = termination_result
                self._record_implementation_termination(task, attempt, termination_result)
            if todo_update_result:
                result["todo_update_result"] = todo_update_result
            self._record_event("implementation_finished", result)
            return result
        except subprocess.TimeoutExpired:
            finished_at = utc_now()
            self._record_task_attempt(state, task, attempt)
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = 124
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            self._record_task_queue_outcome(task, 124, reason="implementation_timeout")
            result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": 124,
                "log_path": str(log_path),
                "error": "timeout",
                "termination_result": self._implementation_returncode_detail(124),
            }
            self._record_implementation_termination(task, attempt, result["termination_result"])
            self._record_event("implementation_finished", result)
            return result
        except Exception as exc:
            finished_at = utc_now()
            failed_phase = state.active_phase or "implementation_setup"
            self._record_task_attempt(state, task, attempt)
            state.last_implementation_started_at = started_at
            state.last_implementation_finished_at = finished_at
            state.last_implementation_returncode = 1
            state.last_implementation_log_path = str(log_path)
            self._mark_implementation_finished(state, finished_at=finished_at)
            state.save(self.state_path)
            self._record_task_queue_outcome(task, 1, reason=f"{type(exc).__name__}: {exc}"[-1000:])
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
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            if task_claim_fd is not None:
                try:
                    os.close(task_claim_fd)
                except OSError:
                    pass
            try:
                if acquired_lock and lock_path.exists():
                    lock_path.unlink()
            except OSError:
                logger.warning("Failed to remove implementation lock %s", lock_path)
            try:
                if acquired_task_claim and task_claim_path.exists():
                    task_claim_path.unlink()
            except OSError:
                logger.warning("Failed to remove implementation task claim lock %s", task_claim_path)

    def _mark_task_completed_in_todo(self, task_id: str) -> dict[str, Any]:
        return self._mark_tasks_completed_in_todo(
            [task_id],
            primary_task_id=task_id,
            completion_reason="single_task",
        )

    def _mark_task_or_bundle_completed_in_todo(self, task: PortalTask) -> dict[str, Any]:
        work_order = self._bundle_work_order_for_task(task)
        if work_order is None:
            return self._mark_task_completed_in_todo(task.task_id)
        return self._mark_tasks_completed_in_todo(
            work_order.task_ids,
            primary_task_id=work_order.primary_task_id,
            completion_reason="bundle_work_order",
            bundle_work_order=work_order.to_dict(),
        )

    def _mark_tasks_completed_in_todo(
        self,
        task_ids: Sequence[str],
        *,
        primary_task_id: str,
        completion_reason: str,
        bundle_work_order: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        todo_path = self.todo_path
        try:
            lines = todo_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            result = {"updated": False, "task_id": primary_task_id, "reason": "read_failed", "error": str(exc)}
            self._record_event("todo_status_update_failed", result)
            return result

        target_task_ids = [
            str(task_id).strip()
            for task_id in dict.fromkeys(task_ids)
            if str(task_id).strip()
        ]
        target_set = set(target_task_ids)
        current_task_id = ""
        header_indices: dict[str, int] = {}
        status_indices: dict[str, int] = {}
        checkbox_indices: dict[str, int] = {}
        checkbox_pattern = re.compile(
            r"^(?P<prefix>\s*[-*]\s+\[)(?P<mark>[^\]])"
            r"(?P<suffix>\]\s+Task checkbox-\d+:\s+(?P<task_id>\S+)\b.*)$"
        )
        for index, line in enumerate(lines):
            checkbox = checkbox_pattern.match(line.rstrip("\r\n"))
            if checkbox and checkbox.group("task_id") in target_set:
                checkbox_indices[checkbox.group("task_id")] = index
            if line.startswith(self.task_header_prefix):
                header = line[3:].strip()
                current_task_id = header.split(" ", 1)[0] if header else ""
                if current_task_id in target_set:
                    header_indices[current_task_id] = index
                continue
            if current_task_id in target_set and line.startswith("- Status:"):
                status_indices[current_task_id] = index
                current_task_id = ""

        missing_status_task_ids = [
            task_id
            for task_id in target_task_ids
            if task_id in header_indices and task_id not in status_indices
        ]
        missing_task_ids = [
            task_id
            for task_id in target_task_ids
            if task_id not in header_indices and task_id not in status_indices
        ]
        if primary_task_id in missing_task_ids:
            result = {
                "updated": False,
                "task_id": primary_task_id,
                "reason": "status_line_missing",
                "missing_task_ids": missing_task_ids,
                "missing_status_task_ids": missing_status_task_ids,
            }
            self._record_event("todo_status_update_failed", result)
            return result

        updated_task_ids: list[str] = []
        already_completed_task_ids: list[str] = []
        updated_checkbox_task_ids: list[str] = []
        for task_id in target_task_ids:
            status_index = status_indices.get(task_id)
            if status_index is None:
                continue
            current = lines[status_index].split(":", 1)[1].strip()
            task_updated = False
            if normalize_status(current) != "completed":
                newline = "\n" if lines[status_index].endswith("\n") else ""
                lines[status_index] = "- Status: completed" + newline
                task_updated = True
            checkbox_index = checkbox_indices.get(task_id)
            if checkbox_index is not None:
                raw_checkbox = lines[checkbox_index]
                newline = "\n" if raw_checkbox.endswith("\n") else ""
                checkbox = checkbox_pattern.match(raw_checkbox.rstrip("\r\n"))
                if checkbox and checkbox.group("mark").lower() != "x":
                    lines[checkbox_index] = (
                        f"{checkbox.group('prefix')}x{checkbox.group('suffix')}{newline}"
                    )
                    updated_checkbox_task_ids.append(task_id)
                    task_updated = True
            if task_updated:
                updated_task_ids.append(task_id)
            else:
                already_completed_task_ids.append(task_id)

        inserted_status_task_ids: list[str] = []
        for task_id in sorted(missing_status_task_ids, key=lambda value: header_indices[value], reverse=True):
            header_index = header_indices[task_id]
            insert_at = header_index + 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            insertion: list[str] = []
            if insert_at == header_index + 1:
                insertion.append("\n")
            insertion.append("- Status: completed\n")
            if insert_at >= len(lines) or lines[insert_at].startswith(self.task_header_prefix):
                insertion.append("\n")
            lines[insert_at:insert_at] = insertion
            inserted_status_task_ids.append(task_id)
            updated_task_ids.append(task_id)
        inserted_status_task_ids.reverse()

        if not updated_task_ids:
            result = {
                "updated": False,
                "task_id": primary_task_id,
                "reason": "already_completed",
                "path": str(todo_path),
                "completion_reason": completion_reason,
                "updated_task_ids": [],
                "already_completed_task_ids": already_completed_task_ids,
                "missing_task_ids": missing_task_ids,
                "missing_status_task_ids": missing_status_task_ids,
                "inserted_status_task_ids": inserted_status_task_ids,
                "updated_checkbox_task_ids": updated_checkbox_task_ids,
            }
            if bundle_work_order is not None:
                result["bundle_work_order"] = bundle_work_order
            commit_result = self._commit_generated_file_update(
                todo_path,
                task_id=primary_task_id,
                subject=f"{primary_task_id}: mark todo completed",
            )
            if commit_result and commit_result.get("reason") != "no_changes":
                result["commit_result"] = commit_result
                self._record_event("todo_status_reconciled", result)
            return result
        tmp_path = todo_path.with_name(f".{todo_path.name}.tmp")
        try:
            tmp_path.write_text("".join(lines), encoding="utf-8")
            os.replace(tmp_path, todo_path)
        except OSError as exc:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            result = {"updated": False, "task_id": primary_task_id, "reason": "write_failed", "error": str(exc)}
            self._record_event("todo_status_update_failed", result)
            return result

        commit_result = self._commit_generated_file_update(
            todo_path,
            task_id=primary_task_id,
            subject=f"{primary_task_id}: mark todo completed",
        )
        result = {
            "updated": True,
            "task_id": primary_task_id,
            "path": str(todo_path),
            "completion_reason": completion_reason,
            "updated_task_ids": updated_task_ids,
            "already_completed_task_ids": already_completed_task_ids,
            "missing_task_ids": missing_task_ids,
            "missing_status_task_ids": missing_status_task_ids,
            "inserted_status_task_ids": inserted_status_task_ids,
            "updated_checkbox_task_ids": updated_checkbox_task_ids,
        }
        if bundle_work_order is not None:
            result["bundle_work_order"] = bundle_work_order
        if commit_result:
            result["commit_result"] = commit_result
        self._record_event("todo_status_updated", result)
        return result

    def _commit_generated_file_update(self, path: Path, *, task_id: str, subject: str) -> dict[str, Any]:
        """Commit a daemon-owned generated file and any parent gitlink updates."""

        started_at = utc_now()
        lock_path = self._repo_merge_lock_path()
        lock_fd, lock_reason, existing_lock = self._try_acquire_lock(
            lock_path,
            lock_kind="merge",
            owner_active=self._merge_lock_owner_is_active,
        )
        if lock_fd is None:
            result: dict[str, Any] = {
                "committed": False,
                "reason": f"checkout_mutation_{lock_reason}",
                "path": str(path),
                "lock_path": str(lock_path),
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_task_id"] = str(existing_lock.get("task_id") or "")
                result["lock_owner_branch"] = str(existing_lock.get("branch") or "")
            return result

        try:
            self._write_lock_metadata(
                lock_fd,
                checkout_lock_metadata(
                    kind="merge",
                    repo_root=self.repo_root,
                    task_id=task_id,
                    branch="generated-file-update",
                    extra={
                        "operation": "commit_generated_file_update",
                        "path": str(path),
                        "started_at": started_at,
                        "state_dir": str(self.state_path.parent.resolve()),
                        "state_path": str(self.state_path.resolve()),
                    },
                ),
            )
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
        finally:
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except OSError:
                logger.warning("Failed to remove checkout mutation lock %s", lock_path)

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
        merge_head = self._git_merge_head_in_repo(repo)
        if merge_head:
            return {
                "committed": False,
                "reason": "repo_merge_in_progress",
                "repo": str(repo),
                "path": relative,
                "merge_head": merge_head,
                "unmerged_paths": sorted(self._unmerged_worktree_paths(repo)),
            }
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

    def _candidate_repository_tree(self, implementation_commit: str) -> str:
        """Return the immutable Git tree promoted by a candidate commit."""

        if not implementation_commit:
            return ""
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--verify",
                f"{implementation_commit}^{{tree}}",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def _proof_risk_for_task(task: PortalTask) -> str:
        """Map scheduler priority to the conservative proof-policy risk."""

        return {
            "P0": "critical",
            "P1": "high",
            "P2": "medium",
            "P3": "low",
        }.get(str(task.priority or "").strip().upper(), "high")

    @staticmethod
    def _proof_scope_hints(task: PortalTask) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Extract bounded AST and modeled-invariant hints from task metadata."""

        normalized_metadata = {
            str(key).strip().lower().replace("_", " "): str(value or "")
            for key, value in task.metadata.items()
        }
        ast_scope_ids = tuple(
            sorted(
                set(
                    split_csv(
                        normalized_metadata.get("ast symbols", "")
                        or normalized_metadata.get("ast scopes", "")
                    )
                )
            )[:MAX_MERGE_PROOF_METADATA_ITEMS]
        )
        invariant_classes = set(
            split_csv(
                normalized_metadata.get("invariant classes", "")
                or normalized_metadata.get("proof invariant classes", "")
                or normalized_metadata.get("proof invariants", "")
            )
        )
        # Task records predate typed invariant metadata in many boards.  AST
        # symbols are still deterministic semantic hints, so project only
        # reviewed, high-signal keywords into the common policy vocabulary.
        hint_text = " ".join(ast_scope_ids).lower()
        keyword_classes = (
            (("transition", "state", "status"), "state_transition"),
            (("lease", "fenc", "lock owner"), "lease_safety"),
            (("acyclic", "dag", "dependency graph"), "dag_acyclicity"),
            (("merge", "dedupe", "idempot"), "merge_idempotence"),
            (("cache key", "cache_key", "fingerprint"), "cache_key_completeness"),
            (("fresh", "stale", "expir"), "evidence_freshness"),
            (("projection", "alias", "canonical"), "projection_equivalence"),
            (("authoriz", "permission", "override"), "authorization"),
            (("integrity", "receipt", "evidence", "proof"), "data_integrity"),
            (("resource", "budget", "isolation"), "resource_isolation"),
        )
        for keywords, invariant_class in keyword_classes:
            if any(keyword in hint_text for keyword in keywords):
                invariant_classes.add(invariant_class)
        return ast_scope_ids, tuple(
            sorted(invariant_classes)[:MAX_MERGE_PROOF_METADATA_ITEMS]
        )

    def _proof_changed_scopes(
        self,
        *,
        baseline_ref: str,
        implementation_commit: str,
        task: PortalTask,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Compile the immutable baseline diff into deterministic proof scopes."""

        if not baseline_ref or not implementation_commit:
            return [], False
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-status",
                "--find-renames",
                baseline_ref,
                implementation_commit,
                "--",
            ],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return [], False
        ast_scope_ids, invariant_classes = self._proof_scope_hints(task)
        risk = self._proof_risk_for_task(task)
        path_kinds: dict[str, str] = {}
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            status = fields[0].strip().upper()
            status_kind = status[:1]
            if status_kind in {"R", "C"} and len(fields) >= 3:
                path_kinds[fields[1].strip("/")] = "deleted"
                path_kinds[fields[2].strip("/")] = "added"
                continue
            change_kind = {
                "A": "added",
                "D": "deleted",
                "M": "modified",
                "T": "modified",
                "U": "modified",
                "X": "modified",
                "B": "modified",
            }.get(status_kind, "modified")
            path_kinds[fields[1].strip("/")] = change_kind
        valid_paths = [
            path
            for path in sorted(path_kinds)
            if path and not path.startswith("../") and not Path(path).is_absolute()
        ]
        complete = len(valid_paths) <= MAX_MERGE_PROOF_METADATA_ITEMS
        scopes: list[dict[str, Any]] = []
        from ..formal_verification_policy import ChangedScope

        for path in valid_paths[:MAX_MERGE_PROOF_METADATA_ITEMS]:
            scope = ChangedScope(
                path=path,
                ast_scope_ids=ast_scope_ids,
                risk=risk,
                invariant_classes=invariant_classes,
                change_kind=path_kinds[path],
                metadata={
                    "task_id": task.task_id,
                    "task_priority": str(task.priority or "").strip().upper(),
                },
            )
            scopes.append({**scope.to_dict(), "scope_id": scope.scope_id})
        return scopes, complete

    def _enqueue_merge_candidate(
        self,
        *,
        branch_name: str,
        implementation_commit: str,
        baseline_ref: str,
        worktree_path: Path | None,
        task: PortalTask,
        attempt: int,
        changed_submodule_paths: Sequence[str] | None = None,
        validation_result: dict[str, Any] | None = None,
        worktree_pool_handoff: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """Durably hand a validated implementation to the repo-wide merge train.

        This is the only merge handoff used by implementation lanes.  The
        canonical task identity and immutable implementation commit form the
        queue's deduplication key; the remaining metadata is sufficient for a
        train consumer to reproduce the daemon's historical merge lifecycle.
        """

        identity = self._identity_for_task(task)
        work_order = self._bundle_work_order_for_task(task)
        candidate_tree = self._candidate_repository_tree(implementation_commit)
        repository_tree_id = f"git-tree:{candidate_tree}" if candidate_tree else ""
        proof_changed_scopes, proof_changed_scopes_complete = (
            self._proof_changed_scopes(
                baseline_ref=baseline_ref,
                implementation_commit=implementation_commit,
                task=task,
            )
        )
        metadata = {
            "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@1",
            "baseline_ref": baseline_ref,
            "implementation_commit": implementation_commit,
            "candidate_tree": candidate_tree,
            "repository_tree_id": repository_tree_id,
            "proof_changed_scopes": proof_changed_scopes,
            "proof_changed_scopes_complete": proof_changed_scopes_complete,
            "worktree_path": str(worktree_path or ""),
            "todo_path": str(self.todo_path),
            "state_path": str(self.state_path),
            "strategy_path": str(self.strategy_path),
            "events_path": str(self.events_path),
            "repo_root": str(self.repo_root),
            "task_header_prefix": self.task_header_prefix,
            "task": asdict(task),
        }
        if changed_submodule_paths is not None:
            metadata["changed_submodule_paths"] = sorted(
                {str(path).strip("/") for path in changed_submodule_paths if str(path).strip("/")}
            )
        if validation_result is not None:
            result_records = [
                {
                    key: item.get(key)
                    for key in (
                        "command",
                        "validation_id",
                        "returncode",
                        "passed",
                        "verdict",
                        "reason",
                        "timed_out",
                        "cache_hit",
                        "cache_key",
                        "cache_evidence_id",
                        "stage",
                        "ordinal",
                        "started_at",
                        "finished_at",
                    )
                    if key in item
                }
                for item in validation_result.get("results", [])[
                    :MAX_MERGE_PROOF_METADATA_ITEMS
                ]
                if isinstance(item, dict)
            ]
            validation_proof = {
                "attempted": bool(validation_result.get("attempted")),
                "passed": bool(validation_result.get("passed")),
                "returncode": int(validation_result.get("returncode") or 0),
                # Validation runs before the daemon creates its commit.  The
                # merge gate must bind the evidence to the immutable commit
                # and Git tree that were actually enqueued, not the earlier
                # workspace HEAD reported by the validation scheduler.
                "target_commit": implementation_commit,
                "target_tree": candidate_tree,
                "repository_tree_id": repository_tree_id,
                "selection": _bounded_merge_proof_value(
                    validation_result.get("selection") or {},
                    field_name="selection",
                ),
                "stages": _bounded_merge_proof_value(
                    validation_result.get("stages") or [],
                    field_name="stages",
                ),
                "results": _bounded_merge_proof_value(
                    result_records,
                    field_name="results",
                ),
                "verdicts": _bounded_merge_proof_value(
                    validation_result.get("verdicts") or {},
                    field_name="verdicts",
                ),
                "proof": _bounded_merge_proof_value(
                    validation_result.get("proof") or {},
                    field_name="proof",
                ),
                "fallbacks": _bounded_merge_proof_value(
                    validation_result.get("fallbacks") or [],
                    field_name="fallbacks",
                ),
                "cache_hits": int(validation_result.get("cache_hits") or 0),
                "cache_misses": int(validation_result.get("cache_misses") or 0),
            }
            raw_proof_gate = validation_result.get(
                "proof_gate",
                validation_result.get("proof_gate_packet"),
            )
            if raw_proof_gate is not None:
                proof_gate_packet = _bounded_merge_proof_value(
                    raw_proof_gate,
                    field_name="proof_gate",
                )
                validation_proof["proof_gate"] = proof_gate_packet
                metadata["proof_gate"] = proof_gate_packet
            metadata["validation_proof"] = validation_proof
        if self.formal_verification_policy is not None:
            metadata["formal_verification_policy"] = _bounded_merge_proof_value(
                self.formal_verification_policy,
                field_name="formal_verification_policy",
            )
        if work_order is not None:
            metadata["bundle_work_order"] = work_order.to_dict()
        if worktree_pool_handoff:
            metadata["worktree_pool_handoff"] = True
        request = self.merge_queue.enqueue(
            branch_name=branch_name,
            task_id=task.task_id,
            priority=task.priority,
            lane_id=f"{os.getpid()}:{self.task_shard_index}",
            attempt=attempt,
            metadata=metadata,
            commit_sha=implementation_commit,
            canonical_task_id=identity.canonical_task_cid,
            canonical_task_key=identity.canonical_task_key,
        )
        result = {
            "attempted": False,
            "merged": False,
            "queued": True,
            "reason": "merge_queued",
            "request_id": str(request.request_id),
            "branch": branch_name,
            "implementation_commit": implementation_commit,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": identity.canonical_task_cid,
            "queue_dir": str(self.merge_queue_dir),
        }
        self._record_event(
            "merge_candidate_enqueued",
            {
                "task_id": task.task_id,
                "attempt": attempt,
                "branch": branch_name,
                "baseline_ref": baseline_ref,
                "implementation_commit": implementation_commit,
                "worktree_path": str(worktree_path or ""),
                **result,
            },
        )
        return request, result

    @staticmethod
    def _portal_task_from_merge_request(request: Any) -> PortalTask:
        task_payload = request.metadata.get("task") if isinstance(request.metadata, dict) else {}
        if not isinstance(task_payload, dict):
            task_payload = {}
        field_names = set(PortalTask.__dataclass_fields__)
        values = {key: value for key, value in task_payload.items() if key in field_names}
        values.setdefault("task_id", str(request.task_id or ""))
        values.setdefault("title", "queued implementation merge")
        values.setdefault("status", "todo")
        values.setdefault("completion", "manual")
        values.setdefault("priority", str(request.priority or "P2"))
        values.setdefault("track", "ops")
        return PortalTask(**values)

    def _merge_train_callback(self, request: Any) -> dict[str, Any]:
        """Adapt one durable queue request to the daemon's mature merge path."""

        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        task = self._portal_task_from_merge_request(request)
        branch_name = str(request.branch_name or "")
        implementation_commit = str(
            request.commit_sha or metadata.get("implementation_commit") or ""
        )
        branch_rehydration = self._rehydrate_merge_request_branch(
            branch_name=branch_name,
            commit_sha=implementation_commit,
            task=task,
            attempt=int(request.attempt or 0),
        )
        if not branch_rehydration.get("ready", False):
            return {
                "attempted": False,
                "merged": False,
                "returncode": 2,
                "reason": str(
                    branch_rehydration.get("reason") or "merge_branch_rehydration_failed"
                ),
                "branch": branch_name,
                "branch_rehydration": branch_rehydration,
            }
        validation_proof = metadata.get("validation_proof")
        if isinstance(validation_proof, dict):
            selection = validation_proof.get("selection")
            selection_scope = (
                str(selection.get("scope") or "") if isinstance(selection, dict) else ""
            )
            if not validation_proof.get("passed") or selection_scope != "pre_merge":
                return {
                    "attempted": False,
                    "merged": False,
                    "returncode": int(validation_proof.get("returncode") or 1),
                    "reason": "validation_failed",
                    "branch": branch_name,
                    "validation_result": validation_proof,
                    "validation_gate_reason": (
                        "validation_did_not_pass"
                        if not validation_proof.get("passed")
                        else "broad_pre_merge_scope_missing"
                    ),
                }
        raw_changed_submodule_paths = metadata.get("changed_submodule_paths")
        changed_submodule_paths = (
            {
                str(path).strip("/")
                for path in raw_changed_submodule_paths
                if str(path).strip("/")
            }
            if isinstance(raw_changed_submodule_paths, list)
            else None
        )
        result = self._merge_branch_to_main(
            branch_name,
            task,
            int(request.attempt or 0),
            baseline_ref=str(metadata.get("baseline_ref") or ""),
            changed_submodule_paths=changed_submodule_paths,
        )
        raw_submodule_merge_results = result.get("submodule_merge_results", [])
        submodule_merge_results = (
            raw_submodule_merge_results
            if isinstance(raw_submodule_merge_results, list)
            else []
        )
        reported_submodule_paths = {
            str(item.get("path") or "").strip("/")
            for item in submodule_merge_results
            if isinstance(item, dict) and str(item.get("path") or "").strip("/")
        }
        missing_changed_submodule_paths = sorted(
            set(changed_submodule_paths or ()) - reported_submodule_paths
        )
        failed_submodules = [
            item
            for item in submodule_merge_results
            if isinstance(item, dict) and not item.get("merged", False)
        ]
        raw_gitlink_recording = result.get("merged_gitlink_recording")
        gitlink_recording_failed = bool(
            isinstance(raw_gitlink_recording, dict)
            and raw_gitlink_recording.get("ok") is False
        )
        if missing_changed_submodule_paths:
            previous_reason = str(result.get("reason") or "submodule_merge_results_missing")
            result.update(
                {
                    "merged": False,
                    "returncode": 2,
                    "reason": "changed_submodule_merge_unverified",
                    "missing_changed_submodule_paths": missing_changed_submodule_paths,
                    "submodule_verification": {
                        "verified": False,
                        "expected_paths": sorted(changed_submodule_paths or ()),
                        "reported_paths": sorted(reported_submodule_paths),
                        "previous_reason": previous_reason,
                    },
                }
            )
        target_branch = self._main_branch_name()
        if (
            not result.get("merged", False)
            and implementation_commit
            and not result.get("submodule_merge_failed", False)
            and not failed_submodules
            and not missing_changed_submodule_paths
            and not gitlink_recording_failed
            and self._git_ref_is_ancestor(implementation_commit, target_branch)
        ):
            previous_reason = str(result.get("reason") or "merge_callback_failed")
            target_commit = self._run_git(
                ["rev-parse", target_branch],
                cwd=self.repo_root,
            ).stdout.strip()
            result.update(
                {
                    "merged": True,
                    "returncode": 0,
                    "reason": "implementation_commit_already_merged",
                    "merge_commit": target_commit,
                    "post_callback_ancestry_reconciliation": {
                        "promoted": True,
                        "implementation_commit": implementation_commit,
                        "target_branch": target_branch,
                        "target_commit": target_commit,
                        "previous_reason": previous_reason,
                    },
                }
            )
        if branch_rehydration.get("rehydrated", False):
            result["branch_rehydration"] = branch_rehydration
        if result.get("merged"):
            worktree_path_text = str(metadata.get("worktree_path") or "")
            cleanup_result = (
                self._cleanup_merged_worktree(
                    Path(worktree_path_text) if worktree_path_text else None,
                    branch_name,
                )
                if worktree_path_text or metadata.get("worktree_pool_handoff") is True
                else {}
            )
            result["cleanup_result"] = cleanup_result
            if cleanup_result and not cleanup_result.get("cleaned", False):
                result["merged"] = False
                result["reason"] = "merge_cleanup_failed"
                result["returncode"] = 1
            else:
                completion_daemon = self
                request_todo_path = Path(str(metadata.get("todo_path") or self.todo_path))
                if request_todo_path != self.todo_path:
                    request_state_path = Path(str(metadata.get("state_path") or self.state_path))
                    completion_daemon = PortalImplementationDaemon(
                        todo_path=request_todo_path,
                        state_path=request_state_path,
                        strategy_path=Path(str(metadata.get("strategy_path") or request_state_path.parent / "strategy.json")),
                        events_path=Path(str(metadata.get("events_path") or request_state_path.parent / "events.jsonl")),
                        repo_root=self.repo_root,
                        task_header_prefix=str(metadata.get("task_header_prefix") or self.task_header_prefix),
                        implement=False,
                        worktree_root=self.worktree_root,
                        worktree_submodule_paths=self.worktree_submodule_paths,
                        merge_queue=self.merge_queue,
                        merge_queue_dir=self.merge_queue_dir,
                    )
                bundle_payload = metadata.get("bundle_work_order")
                if isinstance(bundle_payload, dict):
                    task_ids = [
                        str(item)
                        for item in [
                            bundle_payload.get("primary_task_id"),
                            *(bundle_payload.get("covered_task_ids") or []),
                        ]
                        if str(item or "")
                    ]
                    todo_update_result = completion_daemon._mark_tasks_completed_in_todo(
                        task_ids,
                        primary_task_id=str(bundle_payload.get("primary_task_id") or task.task_id),
                        completion_reason="bundle_work_order",
                        bundle_work_order=bundle_payload,
                    )
                else:
                    todo_update_result = completion_daemon._mark_task_completed_in_todo(task.task_id)
                completion_daemon._record_task_queue_outcome(task, 0)
                result["todo_update_result"] = todo_update_result
        return result

    def _rehydrate_merge_request_branch(
        self,
        *,
        branch_name: str,
        commit_sha: str,
        task: PortalTask,
        attempt: int,
    ) -> dict[str, Any]:
        """Restore a cleaned-up queue branch from its immutable candidate commit."""

        branch_name = str(branch_name or "").strip()
        branch_ref = f"refs/heads/{branch_name}"
        if not branch_name or not commit_sha:
            return {
                "ready": False,
                "rehydrated": False,
                "reason": "merge_branch_rehydration_metadata_missing",
                "branch": branch_name,
                "commit_sha": commit_sha,
            }
        check_ref = subprocess.run(
            ["git", "check-ref-format", "--branch", branch_name],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        candidate = subprocess.run(
            ["git", "rev-parse", "--verify", f"{commit_sha}^{{commit}}"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if check_ref.returncode != 0 or candidate.returncode != 0:
            return {
                "ready": False,
                "rehydrated": False,
                "reason": (
                    "merge_branch_name_invalid"
                    if check_ref.returncode != 0
                    else "merge_candidate_commit_missing"
                ),
                "branch": branch_name,
                "commit_sha": commit_sha,
                "stderr": (check_ref.stderr or candidate.stderr)[-2000:],
            }
        candidate_commit = candidate.stdout.strip()
        current = subprocess.run(
            ["git", "rev-parse", "--verify", f"{branch_ref}^{{commit}}"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if current.returncode == 0:
            current_commit = current.stdout.strip()
            if current_commit == candidate_commit:
                return {
                    "ready": True,
                    "rehydrated": False,
                    "branch": branch_name,
                    "commit_sha": candidate_commit,
                }
            return {
                "ready": False,
                "rehydrated": False,
                "reason": "merge_branch_candidate_mismatch",
                "branch": branch_name,
                "commit_sha": candidate_commit,
                "branch_commit": current_commit,
            }
        create = subprocess.run(
            ["git", "update-ref", branch_ref, candidate_commit, "0" * 40],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        current = subprocess.run(
            ["git", "rev-parse", "--verify", f"{branch_ref}^{{commit}}"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if current.returncode != 0 or current.stdout.strip() != candidate_commit:
            return {
                "ready": False,
                "rehydrated": False,
                "reason": "merge_branch_rehydration_failed",
                "branch": branch_name,
                "commit_sha": candidate_commit,
                "stderr": (create.stderr or current.stderr)[-2000:],
            }
        result = {
            "ready": True,
            "rehydrated": True,
            "branch": branch_name,
            "commit_sha": candidate_commit,
        }
        self._record_event(
            "merge_branch_rehydrated",
            {
                **result,
                "task_id": task.task_id,
                "attempt": attempt,
            },
        )
        return result

    @staticmethod
    def _merge_train_result_request_id(result: dict[str, Any]) -> str:
        request = result.get("request")
        request_id = result.get("request_id")
        if not request_id and isinstance(request, dict):
            request_id = request.get("request_id")
        return str(request_id or "")

    def _consume_one_merge_candidate(self) -> dict[str, Any] | None:
        """Opportunistically advance one item while respecting the train lease."""

        from ..merge_train import MergeTrain

        train = MergeTrain(
            repo_root=self.repo_root,
            queue=self.merge_queue,
            target_branch=self._main_branch_name(),
            max_attempts=int(getattr(self.merge_queue, "max_attempts", 3)),
            merge_callback=self._merge_train_callback,
            formal_verification_policy=self.formal_verification_policy,
            proof_gate=self.proof_gate,
            proof_cache_dir=self.proof_cache_dir,
        )
        return train.run_once()

    @staticmethod
    def _merge_train_result_is_integrated(result: dict[str, Any]) -> bool:
        status = str(result.get("status") or result.get("reason") or "").strip().lower()
        merge_result = result.get("merge_result")
        return (
            status in {"merged", "already_merged", "deduplicated", "completed"}
            or bool(result.get("merged"))
            or (isinstance(merge_result, dict) and bool(merge_result.get("merged")))
        )

    def _enqueue_validated_worktree(
        self,
        *,
        state: PortalTaskState,
        task: PortalTask,
        attempt: int,
        branch_name: str,
        baseline_ref: str,
        worktree_path: Path,
        implementation_commit: str,
        commit_result: Mapping[str, Any],
        validation_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Hand a validated implementation commit to the durable merge train."""

        self._mark_active_phase(
            state,
            phase="merge_queue",
            phase_detail=branch_name,
            worktree_path=worktree_path,
            branch_name=branch_name,
        )
        pool_handoff = self._release_pooled_worktree_lease(
            worktree_path,
            reason="merge_queue_handoff",
        )
        request, merge_result = self._enqueue_merge_candidate(
            branch_name=branch_name,
            implementation_commit=implementation_commit,
            baseline_ref=baseline_ref,
            worktree_path=(
                None if pool_handoff.get("released", False) else worktree_path
            ),
            task=task,
            attempt=attempt,
            changed_submodule_paths=self._committed_submodule_paths(
                commit_result.get("submodule_results") or []
            ),
            validation_result=dict(validation_result),
            worktree_pool_handoff=bool(pool_handoff.get("released", False)),
        )
        if pool_handoff.get("attempted", False):
            merge_result["worktree_pool_handoff"] = pool_handoff
        try:
            train_result = self._consume_one_merge_candidate()
        except Exception as exc:
            # Enqueue has already committed the durable handoff; a busy
            # consumer must not turn the lane into a merge polling loop.
            train_result = {
                "status": "deferred",
                "reason": "merge_train_consumer_unavailable",
                "exception_type": type(exc).__name__,
                "error": str(exc)[-4000:],
            }
            self._record_event(
                "merge_train_consumer_deferred",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "request_id": str(request.request_id),
                    **train_result,
                },
            )
        if train_result is not None:
            merge_result["train_result"] = train_result
            consumed_request_id = self._merge_train_result_request_id(train_result)
            if (
                consumed_request_id == str(request.request_id)
                and self._merge_train_result_is_integrated(train_result)
            ):
                callback_result = train_result.get("merge_result")
                if isinstance(callback_result, dict):
                    merge_result.update(callback_result)
                merge_result.update(
                    {
                        "queued": False,
                        "merged": True,
                        "reason": str(train_result.get("status") or "merged"),
                        "request_id": str(request.request_id),
                    }
                )
        return merge_result

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
        identity_suffix = self._identity_for_task(task).short_id
        execution_id = f"{safe_task_id}-{identity_suffix}"
        attempt_stamp = int(time.time())
        worktree_path = self.worktree_root / f"{execution_id}-attempt-{attempt}-{attempt_stamp}"
        branch_name = f"implementation/{execution_id}-attempt-{attempt}-{attempt_stamp}"
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
        provider_failure: dict[str, Any] = {}
        timeout_result: dict[str, Any] = {}

        try:
            baseline_ref = self._create_seeded_worktree(worktree_path, branch_name, task=task)
            # A pooled checkout keeps a stable physical path so Git does not
            # have to relocate populated submodule worktrees.  Resolve the
            # task's provisional timestamp path before any command, state, or
            # merge metadata is built from it.
            worktree_path = self._effective_pooled_worktree_path(worktree_path)
            workspace_setup = self._worktree_setup_result(worktree_path)
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
                    "workspace_setup": workspace_setup,
                    "cache_hit": workspace_setup["cache_hit"],
                    "setup_duration_seconds": workspace_setup["setup_duration_seconds"],
                    "saved_duration_seconds": workspace_setup["saved_duration_seconds"],
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
                completed = run_process_group_stream(
                    command,
                    cwd=worktree_path,
                    stdout=log_fh,
                    input_text=prompt,
                    timeout_seconds=self.implementation_timeout,
                )
            returncode = completed.returncode
            if returncode != 0:
                provider_failure = self._provider_capacity_failure_from_log(log_path)
            if returncode == 0:
                self._mark_active_phase(
                    state,
                    phase="validating",
                    phase_detail="; ".join(task.validation) if task.validation else "",
                    worktree_path=worktree_path,
                    branch_name=branch_name,
                )
                if not worktree_path.exists():
                    returncode = 1
                    validation_result = self._missing_validation_workspace_result(
                        worktree_path,
                        task=task,
                        log_path=log_path,
                    )
                    cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
                else:
                    self._prepare_worktree_for_validation(worktree_path, task=task, branch_name=branch_name)
                    validation_result = self._run_validation_commands(worktree_path, task, log_path)
                if validation_result.get("passed", False):
                    commit_result = self._commit_worktree_changes(worktree_path, task, attempt)
                    implementation_commit = str(commit_result.get("commit", ""))
                    if implementation_commit:
                        merge_result = self._enqueue_validated_worktree(
                            state=state,
                            task=task,
                            attempt=attempt,
                            branch_name=branch_name,
                            baseline_ref=baseline_ref,
                            worktree_path=worktree_path,
                            implementation_commit=implementation_commit,
                            commit_result=commit_result,
                            validation_result=validation_result,
                        )
                    elif commit_result.get("reason") == "no_changes":
                        cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
                else:
                    returncode = int(validation_result.get("returncode") or 1)
                    if worktree_path.exists():
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
            else:
                pool_failure_release = self._release_pooled_worktree_lease(
                    worktree_path,
                    reason="implementation_command_failed",
                )
                if pool_failure_release.get("attempted", False):
                    cleanup_result = {
                        "cleaned": bool(pool_failure_release.get("released", False)),
                        "reason": "failed_implementation_pool_lease_released",
                        "pooled": bool(pool_failure_release.get("pooled", False)),
                        "pool_release": pool_failure_release,
                    }
        except subprocess.TimeoutExpired:
            returncode = 124
            timeout_result = {
                "task_id": task.task_id,
                "attempt": attempt,
                "worktree_path": str(worktree_path),
                "branch": branch_name,
                "timeout_seconds": float(self.implementation_timeout),
                "salvaged": False,
            }
            self._record_event("implementation_timeout", timeout_result)
            if worktree_path.exists():
                try:
                    self._mark_active_phase(
                        state,
                        phase="validating",
                        phase_detail="timeout salvage: " + "; ".join(task.validation),
                        worktree_path=worktree_path,
                        branch_name=branch_name,
                    )
                    self._prepare_worktree_for_validation(
                        worktree_path,
                        task=task,
                        branch_name=branch_name,
                    )
                    validation_result = self._run_validation_commands(
                        worktree_path,
                        task,
                        log_path,
                    )
                    can_promote = bool(
                        validation_result.get("attempted")
                        and validation_result.get("passed")
                    )
                    if can_promote:
                        commit_result = self._commit_worktree_changes(
                            worktree_path,
                            task,
                            attempt,
                        )
                        implementation_commit = str(commit_result.get("commit", ""))
                        if implementation_commit:
                            merge_result = self._enqueue_validated_worktree(
                                state=state,
                                task=task,
                                attempt=attempt,
                                branch_name=branch_name,
                                baseline_ref=baseline_ref,
                                worktree_path=worktree_path,
                                implementation_commit=implementation_commit,
                                commit_result=commit_result,
                                validation_result=validation_result,
                            )
                        elif commit_result.get("reason") == "no_changes":
                            cleanup_result = self._cleanup_merged_worktree(
                                worktree_path,
                                branch_name,
                            )
                        returncode = 0
                        timeout_result.update(
                            {
                                "salvaged": True,
                                "implementation_commit": implementation_commit,
                                "validation_result": validation_result,
                            }
                        )
                        self._record_event(
                            "implementation_timeout_salvaged",
                            timeout_result,
                        )
                    else:
                        failed_preservation_result = self._preserve_timed_out_worktree(
                            worktree_path,
                            branch_name,
                            task,
                            attempt,
                            validation_result,
                        )
                        commit_result = dict(
                            failed_preservation_result.get("commit_result")
                            or commit_result
                        )
                        implementation_commit = str(commit_result.get("commit", ""))
                        cleanup_result = dict(
                            failed_preservation_result.get("cleanup_result")
                            or cleanup_result
                        )
                        timeout_result["preservation_result"] = failed_preservation_result
                except Exception as timeout_exc:
                    timeout_result["salvage_error"] = str(timeout_exc)[-4000:]
                    timeout_result["salvage_error_type"] = type(timeout_exc).__name__
                    try:
                        cleanup_result = self._cleanup_failed_setup_worktree(
                            worktree_path,
                            branch_name,
                            task=task,
                            attempt=attempt,
                            exception_result=timeout_result,
                        )
                    except Exception as cleanup_exc:
                        cleanup_result = {
                            "cleaned": False,
                            "reason": "cleanup_after_timeout_failed",
                            "error": str(cleanup_exc)[-1000:],
                        }
                    self._record_event(
                        "implementation_timeout_salvage_failed",
                        timeout_result,
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
            # Clean up worktree on any exception, not just setup failures
            if worktree_path.exists():
                try:
                    cleanup_result = self._cleanup_failed_setup_worktree(
                        worktree_path,
                        branch_name,
                        task=task,
                        attempt=attempt,
                        exception_result=exception_result,
                    )
                    exception_result["cleanup_result"] = cleanup_result
                except Exception as cleanup_exc:
                    exception_result["cleanup_error"] = str(cleanup_exc)[-1000:]
            self._record_event(
                "implementation_exception",
                {
                    "task_id": task.task_id,
                    "attempt": attempt,
                    **exception_result,
                },
            )
        if provider_failure.get("exhausted", False):
            return self._record_provider_capacity_deferral(
                task=task,
                state=state,
                attempt=attempt,
                started_at=started_at,
                returncode=returncode,
                log_path=log_path,
                failure=provider_failure,
                worktree_path=worktree_path,
                branch_name=branch_name,
                cleanup_result=cleanup_result,
            )

        finished_at = utc_now()
        self._record_task_attempt(state, task, attempt)
        state.last_implementation_started_at = started_at
        state.last_implementation_finished_at = finished_at
        state.last_implementation_returncode = returncode
        state.last_implementation_log_path = str(log_path)
        state.last_implementation_worktree_path = str(worktree_path)
        state.last_implementation_branch = branch_name
        state.last_implementation_commit = implementation_commit
        state.last_merge_started_at = str(merge_result.get("started_at") or "")
        state.last_merge_finished_at = str(merge_result.get("finished_at") or "")
        state.last_merge_branch = (
            branch_name
            if merge_result.get("merged") or merge_result.get("attempted") or merge_result.get("queued")
            else ""
        )
        state.last_merge_commit = str(merge_result.get("merge_commit") or "")
        state.last_merge_returncode = (
            int(merge_result["returncode"]) if merge_result.get("returncode") is not None else None
        )
        state.last_merge_error = (
            ""
            if merge_result.get("queued")
            else str(merge_result.get("stderr") or merge_result.get("reason") or "")
        )
        if returncode == 0 and (not implementation_commit or merge_result.get("merged")):
            todo_update_result = self._mark_task_or_bundle_completed_in_todo(task)
        self._mark_implementation_finished(state, finished_at=finished_at)
        state.save(self.state_path)
        # Queueing is a successful implementation handoff, but not task
        # completion.  The train consumer records the terminal merge outcome.
        if not merge_result.get("queued"):
            self._record_task_queue_outcome(
                task,
                returncode,
                reason=str(exception_result.get("message") or merge_result.get("reason") or "implementation_failed"),
            )
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
            "workspace_setup": self._worktree_setup_result(worktree_path),
        }
        result["cache_hit"] = result["workspace_setup"]["cache_hit"]
        result["setup_duration_seconds"] = result["workspace_setup"]["setup_duration_seconds"]
        result["saved_duration_seconds"] = result["workspace_setup"]["saved_duration_seconds"]
        termination_result = self._implementation_returncode_detail(returncode)
        if termination_result:
            result["termination_result"] = termination_result
            self._record_implementation_termination(task, attempt, termination_result)
        if exception_result:
            result["exception_result"] = exception_result
        if timeout_result:
            result["timeout_result"] = timeout_result
        if todo_update_result:
            result["todo_update_result"] = todo_update_result
        self._record_event("implementation_finished", result)
        return result

    def _missing_validation_workspace_result(
        self,
        worktree_path: Path,
        *,
        task: PortalTask,
        log_path: Path,
    ) -> dict[str, Any]:
        result = {
            "attempted": False,
            "passed": False,
            "returncode": 1,
            "results": [],
            "reason": "validation_workspace_missing",
            "error": f"validation workspace missing: {worktree_path}",
            "worktree_path": str(worktree_path),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write("\nValidation:\n")
            log_fh.write(f"[validation workspace missing] {worktree_path}\n")
        self._record_event(
            "validation_workspace_missing",
            {"task_id": task.task_id, "worktree_path": str(worktree_path)},
        )
        return result

    def _cleanup_failed_setup_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        *,
        task: PortalTask,
        attempt: int,
        exception_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Remove partial worktrees when setup fails before the implementation command starts."""

        cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
        self._record_event(
            "failed_setup_worktree_cleanup",
            {
                "task_id": task.task_id,
                "attempt": attempt,
                "worktree_path": str(worktree_path),
                "branch": branch_name,
                "cleanup_result": cleanup_result,
                "exception_result": exception_result,
            },
        )
        return cleanup_result

    @staticmethod
    def _implementation_returncode_detail(returncode: int | None) -> dict[str, Any]:
        if returncode is None:
            return {}
        if int(returncode) == 124:
            return {"termination_reason": "timeout", "timed_out": True}
        if int(returncode) < 0:
            signum = -int(returncode)
            try:
                signal_name = signal.Signals(signum).name
            except ValueError:
                signal_name = ""
            return {
                "termination_reason": "signal",
                "terminated_by_signal": True,
                "signal": signum,
                "signal_name": signal_name,
            }
        return {}

    def _record_implementation_termination(
        self,
        task: PortalTask,
        attempt: int,
        termination_result: dict[str, Any],
    ) -> None:
        self._record_event(
            "implementation_terminated",
            {
                "task_id": task.task_id,
                "attempt": attempt,
                **termination_result,
            },
        )

    def _clear_active_execution_state(self, state: PortalTaskState, *, clear_task: bool = False) -> None:
        if clear_task:
            state.active_task_id = ""
            state.active_task_key = ""
            state.active_task_cid = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            state.recommended_task_id = ""
            state.recommended_actions = []
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
        identity = self._identity_for_task(task)
        state.active_task_key = identity.canonical_task_key
        state.active_task_cid = identity.canonical_task_cid
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
        state.last_implementation_task_key = identity.canonical_task_key
        state.last_implementation_task_cid = identity.canonical_task_cid
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
        self._clear_active_execution_state(state, clear_task=True)

    def _create_seeded_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        *,
        task: PortalTask | None = None,
    ) -> str:
        if self.worktree_pool is not None:
            base_ref = self._main_branch_name()
            cache_key = self._implementation_worktree_cache_key()

            def activate(candidate: Path) -> None:
                self._initialize_worktree_submodules(candidate, branch_name=branch_name)

            lease = self.worktree_pool.acquire(
                cache_key=cache_key,
                base_ref=base_ref,
                branch_name=branch_name,
                dependency_paths=self.worktree_submodule_paths,
                activate=activate,
            )
            lease_path = Path(lease.path).resolve()
            try:
                requested_path = worktree_path.resolve()
            except OSError:
                requested_path = worktree_path
            self._worktree_pool_effective_paths[requested_path] = lease_path
            self._worktree_pool_leases[lease_path] = lease
            setup_metrics = self._pool_lease_metrics(lease)
            self._worktree_setup_metrics[lease_path] = setup_metrics
            # Shared dependency links and dirty, untracked planning context are
            # deliberately per lease.  Neither is allowed to become part of a
            # clean pooled image or leak from one task into the next.
            try:
                self._link_shared_worktree_paths(lease_path)
                self._seed_untracked_worktree_context(lease_path, task=task, overwrite_existing=True)
            except BaseException:
                self._worktree_pool_effective_paths.pop(requested_path, None)
                self._worktree_pool_leases.pop(lease_path, None)
                self._worktree_setup_metrics.pop(lease_path, None)
                lease.release(reusable=False)
                raise
            baseline_ref = str(getattr(lease, "base_commit", "") or "")
            if not baseline_ref:
                baseline_ref = self._run_git(["rev-parse", "HEAD"], cwd=lease_path).stdout.strip()
            return baseline_ref

        self._run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), self._main_branch_name()],
            cwd=self.repo_root,
        )
        baseline_ref = self._run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
        self._initialize_worktree_submodules(worktree_path, branch_name=branch_name)
        self._link_shared_worktree_paths(worktree_path)
        self._seed_untracked_worktree_context(worktree_path, task=task, overwrite_existing=True)
        return baseline_ref

    def _effective_pooled_worktree_path(self, requested_path: Path) -> Path:
        """Return the stable leased checkout behind a provisional task path."""

        try:
            key = requested_path.resolve()
        except OSError:
            key = requested_path
        return self._worktree_pool_effective_paths.pop(key, requested_path)

    def _release_pooled_worktree_lease(
        self,
        worktree_path: Path,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Release a pooled checkout while retaining its durable task branch."""

        try:
            lease_key = worktree_path.resolve()
        except OSError:
            lease_key = worktree_path
        lease = self._worktree_pool_leases.pop(lease_key, None)
        if lease is None:
            return {
                "attempted": False,
                "released": False,
                "reason": "worktree_not_pooled",
                "worktree_path": str(worktree_path),
            }
        release_result = lease.release(reusable=True)
        result = {
            "attempted": True,
            "handoff_reason": reason,
            "worktree_path": str(worktree_path),
            **release_result,
        }
        self._record_event("worktree_pool_lease_released", result)
        return result

    def _implementation_worktree_cache_key(self) -> str:
        """Return the stable dependency-setup identity for implementation leases."""

        payload = {
            "schema": "implementation-worktree-setup-v1",
            "submodule_paths": list(self.worktree_submodule_paths),
            "shared_paths": list(SHARED_WORKTREE_PATHS),
            "shared_source_roots": [str(path) for path in self.shared_worktree_source_roots],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _pool_lease_metrics(lease: WorktreeLease) -> dict[str, Any]:
        """Normalize pool metrics for durable events and implementation results."""

        raw = getattr(lease, "metadata", None)
        if not isinstance(raw, dict):
            raw = getattr(lease, "metrics", None)
        metadata = dict(raw) if isinstance(raw, dict) else {}
        cache_hit = bool(metadata.get("cache_hit", metadata.get("reused", getattr(lease, "reused", False))))
        setup_seconds = metadata.get("setup_duration_seconds", metadata.get("setup_seconds", 0.0))
        saved_seconds = metadata.get(
            "saved_duration_seconds",
            metadata.get("saved_seconds", metadata.get("estimated_seconds_saved", 0.0)),
        )
        try:
            normalized_setup = max(0.0, float(setup_seconds or 0.0))
        except (TypeError, ValueError):
            normalized_setup = 0.0
        try:
            normalized_saved = max(0.0, float(saved_seconds or 0.0))
        except (TypeError, ValueError):
            normalized_saved = 0.0
        return {
            **metadata,
            "cache_hit": cache_hit,
            "reused": cache_hit,
            "setup_duration_seconds": normalized_setup,
            "saved_duration_seconds": normalized_saved,
        }

    def _worktree_setup_result(self, worktree_path: Path) -> dict[str, Any]:
        try:
            key = worktree_path.resolve()
        except OSError:
            key = worktree_path
        metrics = self._worktree_setup_metrics.get(key)
        if metrics is not None:
            return dict(metrics)
        return {
            "cache_hit": False,
            "reused": False,
            "setup_duration_seconds": 0.0,
            "saved_duration_seconds": 0.0,
            "pool_enabled": self.worktree_pool is not None,
        }

    def _initialize_worktree_submodules(self, worktree_path: Path, *, branch_name: str = "") -> None:
        init_failures: list[dict[str, Any]] = []
        for relative in self.worktree_submodule_paths:
            if self._create_local_submodule_worktree(worktree_path, relative, branch_name=branch_name):
                target = worktree_path / relative
                if self._is_git_worktree(target):
                    self._initialize_nested_worktree_submodules(
                        target,
                        branch_name=branch_name,
                        parent_relative=relative,
                    )
                    # Validate submodule initialization
                    validation = self._validate_submodule_init(target, relative)
                    if not validation.get("valid"):
                        init_failures.append(validation)
                continue
            if self._worktree_declares_submodule(worktree_path, relative):
                result = self._run_git(["submodule", "update", "--init", "--recursive", "--", relative], cwd=worktree_path)
                target = worktree_path / relative
                if self._is_git_worktree(target):
                    self._initialize_nested_worktree_submodules(
                        target,
                        branch_name=branch_name,
                        parent_relative=relative,
                    )
                    # Validate submodule initialization
                    validation = self._validate_submodule_init(target, relative)
                    if not validation.get("valid"):
                        init_failures.append(validation)
                elif result.returncode != 0:
                    init_failures.append({
                        "valid": False,
                        "path": relative,
                        "reason": "submodule_update_failed",
                        "stderr": result.stderr[-1000:] if hasattr(result, "stderr") else "",
                    })
        if init_failures:
            self._record_event("worktree_submodule_init_failures", {
                "worktree_path": str(worktree_path),
                "branch_name": branch_name,
                "failures": init_failures,
                "failure_count": len(init_failures),
            })

    def _validate_submodule_init(self, target: Path, relative: str) -> dict[str, Any]:
        """Validate that a submodule was properly initialized in a worktree."""
        if not target.exists():
            return {"valid": False, "path": relative, "reason": "target_missing"}
        if not self._is_git_worktree(target):
            return {"valid": False, "path": relative, "reason": "not_git_repo"}
        # Check that HEAD resolves (objects exist)
        head_check = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target,
            text=True,
            capture_output=True,
            check=False,
        )
        if head_check.returncode != 0:
            return {"valid": False, "path": relative, "reason": "head_unresolvable"}
        # Check for detached HEAD with no commits
        branch = self._git_current_branch(target)
        return {
            "valid": True,
            "path": relative,
            "head": head_check.stdout.strip()[:12],
            "branch": branch or "(detached)",
        }

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
        gitlink_ref = self._submodule_gitlink_ref(worktree_path, relative)
        base_ref = self._resolve_submodule_worktree_base_ref(
            source,
            gitlink_ref or "HEAD",
            source_key=source_key,
            worktree_path=worktree_path,
            fallback_when_missing=source_relative is None,
        )
        target = worktree_path / relative
        if base_ref is None:
            # A recursive dependency whose recorded gitlink is unavailable
            # cannot be replaced with an unrelated local HEAD without making
            # its parent checkout dirty. Leave it uninitialized instead.
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(target)],
                cwd=source,
                text=True,
                capture_output=True,
                check=False,
            )
            if target.exists() or target.is_symlink():
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.mkdir(parents=True, exist_ok=True)
            self._record_event(
                "nested_submodule_worktree_deferred",
                {
                    "source": str(source),
                    "source_key": source_key,
                    "worktree_path": str(worktree_path),
                    "target": str(target),
                    "missing_ref": gitlink_ref,
                    "reason": "gitlink_ref_unavailable",
                },
            )
            return False
        if self._is_git_worktree(target) and not target.is_symlink():
            if branch_name:
                expected_branch = self._submodule_worktree_branch_name(branch_name, source_key)
                current_branch = self._git_current_branch(target)
                if current_branch and current_branch != expected_branch:
                    return False
                if not current_branch:
                    # Pooled dependencies are deliberately restored detached
                    # at their recorded prepared revision.  Reattach the clean
                    # checkout to this lease's task-local branch before an
                    # implementation can commit into it.
                    self._run_git(["switch", "-C", expected_branch, base_ref], cwd=target)
            return True
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        # A task may update the parent gitlink or run ``submodule update`` and
        # remove this checkout while Git still records it as a worktree. Drop
        # only this stale registration before recreating the managed path.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(target)],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if branch_name:
            submodule_branch = self._submodule_worktree_branch_name(branch_name, source_key)
            if self._git_ref_exists_in_repo(source, submodule_branch):
                self._run_git(["worktree", "add", str(target), submodule_branch], cwd=source)
                return True
            try:
                self._run_git(["worktree", "add", "-b", submodule_branch, str(target), base_ref], cwd=source)
            except RuntimeError:
                fallback_ref = self._fallback_submodule_worktree_ref(
                    source,
                    bad_ref=base_ref,
                    source_key=source_key,
                    worktree_path=worktree_path,
                )
                self._run_git(["worktree", "add", "-b", submodule_branch, str(target), fallback_ref], cwd=source)
            return True
        self._run_git(["worktree", "add", "--detach", str(target), base_ref], cwd=source)
        return True

    def _resolve_submodule_worktree_base_ref(
        self,
        source: Path,
        base_ref: str,
        *,
        source_key: str,
        worktree_path: Path,
        fallback_when_missing: bool = True,
    ) -> str | None:
        if not base_ref or base_ref == "HEAD" or self._git_ref_exists_in_repo(source, base_ref):
            return base_ref or "HEAD"

        fetch_result = subprocess.run(
            ["git", "fetch", "--quiet", "origin"],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        if fetch_result.returncode == 0 and self._git_ref_exists_in_repo(source, base_ref):
            return base_ref

        fallback_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        fallback_ref = fallback_result.stdout.strip() if fallback_result.returncode == 0 else "HEAD"
        self._record_event(
            "submodule_gitlink_ref_missing",
            {
                "source": str(source),
                "source_key": source_key,
                "worktree_path": str(worktree_path),
                "missing_ref": base_ref,
                "fallback_ref": fallback_ref,
                "fallback_used": fallback_when_missing,
                "fetch_attempted": True,
                "fetch_returncode": fetch_result.returncode,
                "fetch_error": fetch_result.stderr.strip()[:1000],
            },
        )
        if not fallback_when_missing:
            return None
        return fallback_ref or "HEAD"

    def _fallback_submodule_worktree_ref(
        self,
        source: Path,
        *,
        bad_ref: str,
        source_key: str,
        worktree_path: Path,
    ) -> str:
        fallback_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        fallback_ref = fallback_result.stdout.strip() if fallback_result.returncode == 0 else "HEAD"
        self._record_event(
            "submodule_worktree_base_ref_retried",
            {
                "source": str(source),
                "source_key": source_key,
                "worktree_path": str(worktree_path),
                "bad_ref": bad_ref,
                "fallback_ref": fallback_ref,
                "fallback_returncode": fallback_result.returncode,
                "fallback_error": fallback_result.stderr.strip()[:1000],
            },
        )
        return fallback_ref or "HEAD"

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

    def _git_absolute_dir(self, repo: Path) -> Path | None:
        result = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return Path(result.stdout.strip()).resolve()

    @staticmethod
    def _submodule_relative_from_config(modules_dir: Path, config_path: Path) -> Path | None:
        try:
            relative_parent = config_path.parent.relative_to(modules_dir)
        except ValueError:
            return None
        parts = [part for part in relative_parent.parts if part != "modules"]
        if not parts:
            return None
        return Path(*parts)

    def _repair_stale_submodule_worktree_configs(self, repo_path: Path) -> dict[str, Any]:
        git_dir = self._git_absolute_dir(repo_path)
        if git_dir is None:
            return {"attempted": False, "reason": "git_dir_unavailable", "repo": str(repo_path)}
        modules_dir = git_dir / "modules"
        if not modules_dir.is_dir():
            return {"attempted": False, "reason": "modules_dir_missing", "repo": str(repo_path)}

        repairs: list[dict[str, Any]] = []
        for config_path in sorted(modules_dir.rglob("config")):
            module_relative = self._submodule_relative_from_config(modules_dir, config_path)
            if module_relative is None:
                continue
            checkout_path = (repo_path / module_relative).resolve()
            if not checkout_path.exists():
                continue
            current = subprocess.run(
                ["git", "config", "--file", str(config_path), "--get", "core.worktree"],
                cwd=repo_path,
                text=True,
                capture_output=True,
                check=False,
            )
            if current.returncode != 0 or not current.stdout.strip():
                continue
            current_value = current.stdout.strip()
            current_path = Path(current_value)
            current_target = current_path if current_path.is_absolute() else (config_path.parent / current_path)
            try:
                if current_target.resolve().exists():
                    continue
            except OSError:
                pass

            new_value = os.path.relpath(checkout_path, config_path.parent.resolve())
            update = subprocess.run(
                ["git", "config", "--file", str(config_path), "core.worktree", new_value],
                cwd=repo_path,
                text=True,
                capture_output=True,
                check=False,
            )
            repairs.append(
                {
                    "config_path": str(config_path),
                    "module_path": str(module_relative),
                    "old_worktree": current_value,
                    "new_worktree": new_value,
                    "repaired": update.returncode == 0,
                    "returncode": update.returncode,
                    "stdout": update.stdout[-4000:],
                    "stderr": update.stderr[-4000:],
                }
            )

        result = {
            "attempted": True,
            "repo": str(repo_path),
            "repaired_count": sum(1 for item in repairs if item.get("repaired", False)),
            "repairs": repairs,
        }
        if repairs:
            self._record_event("stale_submodule_worktree_config_repair", result)
        return result

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
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
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
        try:
            worktree_path.resolve().relative_to(self.worktree_root.resolve())
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Refusing to link shared dependencies outside managed worktree root: %s",
                worktree_path,
            )
            return

        for relative in SHARED_WORKTREE_PATHS:
            source: Path | None = None
            for source_root in self.shared_worktree_source_roots:
                try:
                    candidate = (source_root / relative).resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if candidate.is_dir():
                    source = candidate
                    break
            if source is None:
                continue
            target = worktree_path / relative
            try:
                target_parent = target.parent.resolve()
                target_resolved = target.resolve(strict=False)
            except (OSError, RuntimeError):
                logger.warning("Cannot resolve shared dependency link parent: %s", target.parent)
                continue
            if target_resolved == source or target_parent == source:
                logger.warning(
                    "Refusing to link shared dependency onto or inside its source: %s",
                    target,
                )
                continue
            if target.is_symlink():
                try:
                    if target.resolve(strict=True) == source:
                        continue
                except (OSError, RuntimeError):
                    pass
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
        """Copy relevant new source context into an ephemeral worktree.

        Modified tracked files belong to the source checkout and must not leak
        into an isolated implementation branch. Otherwise an unrelated local
        edit can be committed by the agent and later block the merge train.
        """

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
            ["git", "diff", "--cached", "--name-only", "--diff-filter=A", "-z"],
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
            if self._submodule_results_have_commits(submodule_results):
                self._run_git(
                    [
                        "-c",
                        "user.name=Implementation Daemon",
                        "-c",
                        "user.email=implementation-daemon@example.invalid",
                        "commit",
                        "--allow-empty",
                        "-m",
                        f"{task.task_id}: {task.title or 'implementation attempt'}",
                        "-m",
                        f"Attempt: {attempt}",
                        "-m",
                        "Handoff: committed submodule changes",
                    ],
                    cwd=worktree_path,
                )
                commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=worktree_path).stdout.strip()
                return {
                    "committed": True,
                    "commit": commit_ref,
                    "reason": "submodule_only",
                    "status": status,
                    "submodule_results": submodule_results,
                }
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

    @classmethod
    def _submodule_results_have_commits(cls, results: Sequence[dict[str, Any]]) -> bool:
        for result in results:
            if result.get("committed", False):
                return True
            nested = result.get("nested_submodule_results")
            if isinstance(nested, list) and cls._submodule_results_have_commits(nested):
                return True
        return False

    @classmethod
    def _committed_submodule_paths(cls, results: Sequence[dict[str, Any]]) -> list[str]:
        """Return exact repositories that produced task-owned commits."""

        paths: set[str] = set()
        for result in results:
            if result.get("committed", False):
                path = str(result.get("path") or "").strip("/")
                if path:
                    paths.add(path)
            nested = result.get("nested_submodule_results")
            if isinstance(nested, list):
                paths.update(cls._committed_submodule_paths(nested))
        return sorted(paths)

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
        return self._preserve_interrupted_worktree(
            worktree_path,
            branch_name,
            task,
            attempt,
            evidence=validation_result,
            rescue_suffix="failed-validation",
            event_type="failed_validation_worktree_preserved",
            evidence_field="validation_result",
        )

    def _preserve_timed_out_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        return self._preserve_interrupted_worktree(
            worktree_path,
            branch_name,
            task,
            attempt,
            evidence=validation_result,
            rescue_suffix="timed-out",
            event_type="timed_out_worktree_preserved",
            evidence_field="validation_result",
        )

    def _preserve_interrupted_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        *,
        evidence: Mapping[str, Any],
        rescue_suffix: str,
        event_type: str,
        evidence_field: str,
    ) -> dict[str, Any]:
        started_at = utc_now()
        commit_result = self._commit_worktree_changes(worktree_path, task, attempt)
        rescue_branch = ""
        implementation_commit = str(commit_result.get("commit", ""))
        if implementation_commit:
            rescue_branch = self._interrupted_worktree_rescue_branch_name(
                branch_name,
                rescue_suffix,
            )
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
            evidence_field: dict(evidence),
        }
        self._record_event(event_type, result)
        return result

    @staticmethod
    def _failed_validation_rescue_branch_name(branch_name: str) -> str:
        return PortalImplementationDaemon._interrupted_worktree_rescue_branch_name(
            branch_name,
            "failed-validation",
        )

    @staticmethod
    def _interrupted_worktree_rescue_branch_name(
        branch_name: str,
        suffix: str,
    ) -> str:
        safe_name = branch_name.removeprefix("implementation/").strip("/").replace(" ", "-")
        safe_suffix = suffix.strip("/").replace(" ", "-") or "interrupted"
        return f"rescue/{safe_name or 'implementation-attempt'}-{safe_suffix}"

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
        if not workspace_path.exists():
            return self._missing_validation_workspace_result(workspace_path, task=task, log_path=log_path)

        if not task.validation:
            return {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": "no_commands",
            }

        commands: list[str] = []
        normalization_notes: list[str] = []
        for raw_command in task.validation:
            command, notes = self._normalize_validation_command(raw_command)
            commands.append(command)
            normalization_notes.extend(notes)

        # Validation is the last gate before a candidate is committed/enqueued
        # (or before an in-place task is marked complete).  Impact selection is
        # still recorded and used for staging, but every unrelated targeted
        # check is escalated into the broad pre-merge stage here.
        result = self.validation_scheduler.run(
            commands,
            workspace_path=workspace_path,
            require_full_validation=True,
            scope="pre_merge",
            runner=self._validation_command_runner,
        )

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write("\nValidation:\n")
            for note in normalization_notes:
                log_fh.write(f"[validation normalized] {note}\n")
            selection = result.get("selection")
            if isinstance(selection, dict):
                log_fh.write(
                    "[validation selection] "
                    f"scope={selection.get('scope')} changed={selection.get('changed_files', [])} "
                    f"escalated={selection.get('escalated', False)}\n"
                )
            for command_result in result.get("results", []):
                if not isinstance(command_result, dict):
                    continue
                cache_label = " cache-hit" if command_result.get("cache_hit") else ""
                stage = str(command_result.get("stage") or "validation")
                log_fh.write(f"$ {command_result.get('command', '')} [{stage}{cache_label}]\n")
                output = command_result.get("output")
                if output:
                    log_fh.write(str(output))
                    if not str(output).endswith("\n"):
                        log_fh.write("\n")
                if command_result.get("timed_out"):
                    log_fh.write(f"[validation timed out] timeout={self.implementation_timeout}\n")
                elif int(command_result.get("returncode") or 0) != 0:
                    log_fh.write(
                        f"[validation failed] returncode={command_result.get('returncode')}\n"
                    )
                # Command output belongs in the attempt log, not the durable
                # daemon state/event stream or merge-queue receipt.
                command_result.pop("output", None)
            log_fh.write("[validation passed]\n" if result.get("passed") else "[validation stopped]\n")
            log_fh.flush()
        return result

    @staticmethod
    def _validation_command_runner(
        *,
        spec: Any,
        workspace_path: Path,
        timeout_seconds: float,
        environment: dict[str, str],
    ) -> dict[str, Any]:
        """Run one command through the daemon's patchable subprocess seam."""

        started_at = utc_now()
        completed = subprocess.run(
            ["/bin/bash", "-lc", str(spec.command)],
            cwd=workspace_path,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            env=environment,
        )
        return {
            "command": str(spec.command),
            "raw_command": str(spec.raw_command or spec.command),
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": int(completed.returncode),
            "output": completed.stdout or "",
        }

    @staticmethod
    def _normalize_validation_command(command: str) -> tuple[str, list[str]]:
        """Return a shell validation command with known stale tool flags removed."""

        normalized = normalize_validation_command_text(command)
        notes: list[str] = []
        if normalized != command:
            notes.append("removed markdown inline-code wrapper from validation command")
        if re.search(r"\b(?:tsc|typescript)\b", normalized):
            for flag in UNSUPPORTED_TYPESCRIPT_VALIDATION_FLAGS:
                updated = re.sub(rf"(^|[\s;&|]){re.escape(flag)}(?=$|[\s;&|])", r"\1", normalized)
                if updated != normalized:
                    normalized = updated
                    notes.append(f"removed unsupported TypeScript flag {flag}")
        return normalized, notes

    def _main_branch_name(self) -> str:
        if self.merge_target_branch:
            if not self._git_ref_exists(self.merge_target_branch):
                raise RuntimeError(
                    f"Configured merge target branch does not exist: {self.merge_target_branch}"
                )
            return self.merge_target_branch
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

    def _git_worktree_entries_for_repo(self, cwd: Path) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=cwd,
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

    def _git_worktree_entries(self) -> list[dict[str, str]]:
        return self._git_worktree_entries_for_repo(self.repo_root)

    @staticmethod
    def _path_compare_key(path: Path) -> Path:
        try:
            return path.resolve(strict=False)
        except OSError:
            return path.absolute()

    def _worktree_path_registered_in_repo(self, cwd: Path, worktree_path: Path) -> bool:
        expected = self._path_compare_key(worktree_path)
        for entry in self._git_worktree_entries_for_repo(cwd):
            registered = entry.get("worktree")
            if not registered:
                continue
            if self._path_compare_key(Path(registered)) == expected:
                return True
        return False

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
                generated_restore = self._restore_generated_dirty_paths(
                    checked_out_path,
                    dirty_paths,
                    reason="main_merge_worktree_dirty",
                )
                if generated_restore:
                    dirty_paths = sorted(self._dirty_worktree_paths(checked_out_path))
                if dirty_paths:
                    result = {
                        "available": False,
                        "reason": "main_merge_worktree_dirty",
                        "target_branch": target_branch,
                        "worktree_path": str(checked_out_path),
                        "dirty_paths": dirty_paths,
                    }
                    if generated_restore:
                        result["generated_dirty_restore"] = generated_restore
                    return result
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

    def _rebase_stale_submodule_pointers(
        self,
        branch_name: str,
        target_branch: str,
    ) -> dict[str, Any]:
        """Rebase a branch's submodule pointers when they've drifted from target.

        If a branch was created days ago and the target branch has since updated
        submodule pointers, this rebases the branch onto the current target to
        pick up the new submodule state. This prevents merge conflicts caused
        solely by outdated submodule pointer commits.
        """
        results: list[dict[str, Any]] = []

        # Check if branch is behind target on submodule paths
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", f"{branch_name}...{target_branch}"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if diff_result.returncode != 0:
            return {"attempted": False, "reason": "diff_failed"}

        changed_paths = set(diff_result.stdout.strip().splitlines())
        submodule_paths = set(self.worktree_submodule_paths)
        stale_submodules = changed_paths & submodule_paths

        if not stale_submodules:
            return {"attempted": False, "reason": "no_stale_submodules"}

        # Check if the branch can be cleanly rebased
        # First try a dry-run merge to see if submodule-only conflicts exist
        merge_base = subprocess.run(
            ["git", "merge-base", branch_name, target_branch],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if merge_base.returncode != 0:
            return {"attempted": False, "reason": "no_merge_base"}

        base_commit = merge_base.stdout.strip()

        # Check which stale submodules only have pointer changes (not content conflicts)
        for sm_path in stale_submodules:
            # Get the submodule commit on branch vs target
            branch_sm = subprocess.run(
                ["git", "rev-parse", f"{branch_name}:{sm_path}"],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            target_sm = subprocess.run(
                ["git", "rev-parse", f"{target_branch}:{sm_path}"],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

            if branch_sm.returncode != 0 or target_sm.returncode != 0:
                results.append({"path": sm_path, "action": "skip", "reason": "rev_parse_failed"})
                continue

            branch_commit = branch_sm.stdout.strip()
            target_commit = target_sm.stdout.strip()

            if branch_commit == target_commit:
                results.append({"path": sm_path, "action": "skip", "reason": "already_current"})
                continue

            # Check if target commit is descendant of branch commit (fast-forward possible)
            is_ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", branch_commit, target_commit],
                cwd=self.repo_root / sm_path if (self.repo_root / sm_path).exists() else self.repo_root,
                capture_output=True,
                check=False,
            )

            results.append({
                "path": sm_path,
                "branch_commit": branch_commit[:12],
                "target_commit": target_commit[:12],
                "fast_forward_possible": is_ancestor.returncode == 0,
                "action": "rebase_candidate",
            })

        # If all stale submodules can fast-forward, attempt rebase
        rebase_candidates = [r for r in results if r.get("fast_forward_possible")]
        if rebase_candidates and len(rebase_candidates) == len([r for r in results if r.get("action") == "rebase_candidate"]):
            # Safe to rebase - all submodule changes are fast-forwardable
            rebase = subprocess.run(
                ["git", "rebase", "--onto", target_branch, base_commit, branch_name],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if rebase.returncode == 0:
                return {
                    "attempted": True,
                    "rebased": True,
                    "stale_submodules": list(stale_submodules),
                    "results": results,
                }
            else:
                # Abort failed rebase
                subprocess.run(
                    ["git", "rebase", "--abort"],
                    cwd=self.repo_root,
                    capture_output=True,
                    check=False,
                )
                return {
                    "attempted": True,
                    "rebased": False,
                    "reason": "rebase_failed",
                    "stderr": rebase.stderr[:2000],
                    "results": results,
                }

        return {
            "attempted": True,
            "rebased": False,
            "reason": "not_all_fast_forwardable",
            "stale_submodules": list(stale_submodules),
            "results": results,
        }

    def _merge_branch_to_main(
        self,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        *,
        baseline_ref: str = "",
        changed_submodule_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now()
        self._preserve_generated_nested_worktree_directories()
        stale_submodule_worktree_config_repair = self._repair_stale_submodule_worktree_configs(self.repo_root)
        target_branch = self._main_branch_name()
        # Attempt to rebase stale submodule pointers before merge
        submodule_rebase = self._rebase_stale_submodule_pointers(branch_name, target_branch)
        if submodule_rebase.get("rebased"):
            self._record_event("submodule_pointer_rebase", submodule_rebase)
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
            if stale_submodule_worktree_config_repair.get("repairs"):
                result["stale_submodule_worktree_config_repair"] = stale_submodule_worktree_config_repair
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
                if stale_submodule_worktree_config_repair.get("repairs"):
                    result["stale_submodule_worktree_config_repair"] = stale_submodule_worktree_config_repair
                if llm_workspace_resolver:
                    result["llm_merge_resolver"] = llm_workspace_resolver
                self._record_event("merge_finished", result)
                return result

            merge_workspace = Path(str(workspace_result["path"]))
            merge_workspace_ephemeral = bool(workspace_result.get("ephemeral", False))
            resolved_add_add_conflicts = self._resolve_generated_add_add_conflicts(cwd=merge_workspace)
            identical_untracked_paths = self._identical_untracked_merge_paths(branch_name, cwd=merge_workspace)
            restored_generated_dirty_overlap = self._restore_generated_dirty_merge_overlap(
                branch_name,
                cwd=merge_workspace,
                ignore_paths=set(identical_untracked_paths),
            )
            dirty_overlap = self._dirty_merge_conflict_paths(
                branch_name,
                cwd=merge_workspace,
                ignore_paths=set(identical_untracked_paths),
            )
            generated_submodule_reconciliation = self._reconcile_generated_dirty_submodule_overlap(
                merge_workspace,
                dirty_overlap,
                branch_name=branch_name,
                task=task,
            )
            if generated_submodule_reconciliation:
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
                    "restored_generated_dirty_overlap": restored_generated_dirty_overlap,
                    "generated_submodule_reconciliation": generated_submodule_reconciliation,
                    "submodule_merge_results": [],
                }
                if dirty_overlap:
                    if stale_submodule_worktree_config_repair.get("repairs"):
                        result["stale_submodule_worktree_config_repair"] = stale_submodule_worktree_config_repair
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
                    "restored_generated_dirty_overlap": restored_generated_dirty_overlap,
                },
            )
            command = [
                "git",
                "merge",
                "--no-ff",
                "--no-edit",
                branch_name,
            ]
            pre_merge_commit = self._run_git(["rev-parse", "HEAD"], cwd=merge_workspace).stdout.strip()
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
            deterministic_conflict_repair: list[dict[str, object]] = []
            shared_worktree_path_scrub: dict[str, Any] = {}
            merge_returncode = merge.returncode
            merged_gitlink_recording: dict[str, Any] = {}
            if merge_returncode != 0:
                deterministic_conflict_repair = [
                    *self._resolve_generated_markdown_conflicts(merge_workspace),
                    *self._resolve_reconciliation_guardrail_todo_conflicts(merge_workspace),
                    *self._resolve_launch_readiness_conflicts(merge_workspace),
                ]
                if deterministic_conflict_repair and not self._unmerged_worktree_paths(merge_workspace):
                    llm_merge_commit_result = self._commit_llm_resolved_merge(merge_workspace)
                    if llm_merge_commit_result.get("completed", False):
                        merge_returncode = 0
                    else:
                        merge_abort_result = self._abort_failed_merge(merge_workspace)
                if merge_returncode != 0 and not merge_abort_result:
                    submodule_conflict_repair = self._repair_submodule_gitlink_merge_conflicts(
                        merge_workspace,
                        task=task,
                        attempt=attempt,
                    )
                if merge_returncode != 0 and submodule_conflict_repair.get("repaired", False):
                    merge_returncode = 0
                elif (
                    merge_returncode != 0
                    and submodule_conflict_repair
                    and submodule_conflict_repair.get("reason") != "no_gitlink_conflicts"
                    and not merge_abort_result
                ):
                    # An unresolved gitlink must never be handed to a resolver
                    # that could blindly stage ours or theirs.  Abort this merge;
                    # the failed implementation event remains eligible for retry.
                    merge_abort_result = self._abort_failed_merge(merge_workspace)
                elif merge_returncode != 0 and not merge_abort_result:
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
                shared_worktree_path_scrub = self._scrub_tracked_shared_worktree_paths(
                    merge_workspace,
                    task=task,
                )
                if shared_worktree_path_scrub.get("committed", False):
                    merge_commit = str(shared_worktree_path_scrub.get("commit") or merge_commit)
                if not shared_worktree_path_scrub.get("ok", True):
                    merge_returncode = 2
                submodule_merge_results = self._merge_submodule_branches_to_main(
                    branch_name,
                    task=task,
                    attempt=attempt,
                    baseline_ref=baseline_ref,
                    changed_submodule_paths=changed_submodule_paths,
                )
                merged_gitlink_recording = self._record_merged_submodule_gitlinks(
                    merge_workspace,
                    submodule_merge_results,
                    task=task,
                )
                if merged_gitlink_recording.get("committed", False):
                    merge_commit = str(merged_gitlink_recording["commit"])
                if not merged_gitlink_recording.get("ok", True):
                    merge_returncode = 2
            elif removed_untracked:
                self._restore_removed_untracked_paths(removed_untracked, cwd=merge_workspace)
            failed_submodules = [item for item in submodule_merge_results if not item.get("merged", False)]
            submodule_failure_rollback: dict[str, Any] = {}
            if failed_submodules and merge_returncode == 0:
                submodule_failure_rollback = self._rollback_parent_merge_after_submodule_failure(
                    merge_workspace,
                    pre_merge_commit=pre_merge_commit,
                    failed_submodules=failed_submodules,
                )
                if submodule_failure_rollback.get("rolled_back", False):
                    merge_commit = ""
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
                    "restored_generated_dirty_overlap": restored_generated_dirty_overlap,
                    "generated_submodule_reconciliation": generated_submodule_reconciliation,
                    "deterministic_conflict_repair": deterministic_conflict_repair,
                    "shared_worktree_path_scrub": shared_worktree_path_scrub,
                    "merged_gitlink_recording": merged_gitlink_recording,
                    "submodule_failure_rollback": submodule_failure_rollback,
                    "submodule_merge_results": submodule_merge_results,
                }
            if stale_submodule_worktree_config_repair.get("repairs"):
                result["stale_submodule_worktree_config_repair"] = stale_submodule_worktree_config_repair
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
            elif not merged_gitlink_recording.get("ok", True):
                result["reason"] = "submodule_gitlink_recording_failed"
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

    def _scrub_tracked_shared_worktree_paths(self, cwd: Path, *, task: PortalTask) -> dict[str, Any]:
        removed: list[dict[str, Any]] = []
        for relative in SHARED_WORKTREE_PATHS:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
            )
            if tracked.returncode != 0:
                continue
            remove = subprocess.run(
                ["git", "rm", "-r", "--ignore-unmatch", "--", relative],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
            )
            removed.append(
                {
                    "path": relative,
                    "removed": remove.returncode == 0,
                    "returncode": remove.returncode,
                    "stdout": remove.stdout[-1000:],
                    "stderr": remove.stderr[-1000:],
                }
            )
        if not removed:
            return {"ok": True, "scrubbed": False, "paths": []}
        failed = [item for item in removed if not item.get("removed", False)]
        if failed:
            return {"ok": False, "scrubbed": True, "paths": removed, "reason": "git_rm_failed"}
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if status.returncode != 0:
            return {
                "ok": False,
                "scrubbed": True,
                "paths": removed,
                "reason": "status_failed",
                "stderr": status.stderr[-1000:],
            }
        if not status.stdout.strip():
            return {"ok": True, "scrubbed": True, "committed": False, "paths": removed, "reason": "no_changes"}
        commit = subprocess.run(
            ["git", "commit", "-m", f"{task.task_id}: scrub shared dependency paths"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        result: dict[str, Any] = {
            "ok": commit.returncode == 0,
            "scrubbed": True,
            "committed": commit.returncode == 0,
            "paths": removed,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-1000:],
            "stderr": commit.stderr[-1000:],
        }
        if commit.returncode == 0:
            result["commit"] = self._run_git(["rev-parse", "HEAD"], cwd=cwd).stdout.strip()
        else:
            result["reason"] = "commit_failed"
        return result

    def _record_merged_submodule_gitlinks(
        self,
        workspace: Path,
        submodule_merge_results: Sequence[dict[str, Any]],
        *,
        task: PortalTask,
    ) -> dict[str, Any]:
        """Record verified merged revisions through every parent gitlink.

        Managed paths may identify a nested repository such as
        ``outer/parent/inner/child``.  After the child branch merges, each
        containing repository must commit its direct child gitlink before the
        root can record the new parent revision.  Only the expected gitlink is
        staged at each level; unrelated unstaged work remains untouched.
        """

        managed_roots = tuple(
            path.strip().strip("/")
            for path in self.worktree_submodule_paths
            if path.strip().strip("/")
        )
        expected_commits: dict[str, str] = {}
        for item in submodule_merge_results:
            relative = str(item.get("path") or "").strip().strip("/")
            commit = str(item.get("commit") or "").strip()
            if (
                not item.get("merged", False)
                or not relative
                or not commit
                or not self._repo_relative_path_safe(relative)
                or not any(
                    relative == root or relative.startswith(f"{root}/")
                    for root in managed_roots
                )
            ):
                continue
            checkout = workspace / relative
            if not self._is_git_worktree(checkout):
                continue
            current = self._run_git(["rev-parse", "HEAD"], cwd=checkout).stdout.strip()
            if current == commit:
                expected_commits[relative] = commit

        # A descendant chain records every containing gitlink up to the root.
        # Its parent result therefore names the pre-propagation commit and must
        # not be replayed after the descendant has advanced that parent.
        expected_commits = {
            relative: commit
            for relative, commit in expected_commits.items()
            if not any(
                other.startswith(f"{relative}/")
                for other in expected_commits
                if other != relative
            )
        }

        if not expected_commits:
            return {
                "attempted": False,
                "ok": True,
                "committed": False,
                "reason": "no_matching_merged_root_gitlinks",
            }

        workspace = workspace.resolve()
        chain: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        committed = False

        for relative, leaf_commit in sorted(
            expected_commits.items(),
            key=lambda item: (-len(Path(item[0]).parts), item[0]),
        ):
            current_checkout = (workspace / relative).resolve()
            expected_commit = leaf_commit
            if not self._is_git_worktree(current_checkout):
                failures.append(
                    {
                        "path": relative,
                        "reason": "merged_submodule_checkout_missing",
                        "expected_commit": expected_commit,
                    }
                )
                continue

            while current_checkout != workspace:
                parent_repo = current_checkout.parent
                while parent_repo != workspace and not self._is_git_worktree(parent_repo):
                    parent_repo = parent_repo.parent
                if not self._is_git_worktree(parent_repo):
                    failures.append(
                        {
                            "path": relative,
                            "reason": "parent_git_worktree_missing",
                            "checkout": str(current_checkout),
                        }
                    )
                    break

                try:
                    child_relative = current_checkout.relative_to(parent_repo).as_posix()
                    full_relative = current_checkout.relative_to(workspace).as_posix()
                except ValueError:
                    failures.append(
                        {
                            "path": relative,
                            "reason": "submodule_outside_merge_workspace",
                            "checkout": str(current_checkout),
                            "parent_repo": str(parent_repo),
                        }
                    )
                    break

                current_commit = self._run_git(
                    ["rev-parse", "HEAD"], cwd=current_checkout
                ).stdout.strip()
                if current_commit != expected_commit:
                    failures.append(
                        {
                            "path": full_relative,
                            "reason": "merged_submodule_revision_mismatch",
                            "expected_commit": expected_commit,
                            "current_commit": current_commit,
                        }
                    )
                    break

                tracked = self._run_git(
                    ["ls-files", "--stage", "--", child_relative], cwd=parent_repo
                )
                tracked_line = tracked.stdout.strip().splitlines()
                if (
                    tracked.returncode != 0
                    or len(tracked_line) != 1
                    or not tracked_line[0].startswith("160000 ")
                ):
                    failures.append(
                        {
                            "path": full_relative,
                            "reason": "parent_path_is_not_gitlink",
                            "parent_repo": str(parent_repo),
                            "tracked": tracked.stdout[-1000:],
                        }
                    )
                    break

                recorded_commit = self._submodule_gitlink_ref(parent_repo, child_relative)
                entry: dict[str, Any] = {
                    "path": full_relative,
                    "parent_repo": str(parent_repo),
                    "expected_commit": expected_commit,
                    "recorded_commit": recorded_commit,
                    "committed": False,
                }
                if recorded_commit != expected_commit:
                    staged = self._run_git(
                        ["diff", "--cached", "--name-only"], cwd=parent_repo
                    )
                    staged_paths = {
                        line.strip() for line in staged.stdout.splitlines() if line.strip()
                    }
                    unrelated_staged = sorted(staged_paths - {child_relative})
                    if staged.returncode != 0 or unrelated_staged:
                        entry.update(
                            {
                                "reason": "parent_has_unrelated_staged_changes",
                                "staged_paths": sorted(staged_paths),
                                "unrelated_staged_paths": unrelated_staged,
                            }
                        )
                        failures.append(entry)
                        break

                    stage = self._run_git(
                        ["add", "--", child_relative], cwd=parent_repo
                    )
                    if stage.returncode != 0:
                        entry.update(
                            {
                                "reason": "parent_gitlink_stage_failed",
                                "returncode": stage.returncode,
                                "stdout": stage.stdout[-1000:],
                                "stderr": stage.stderr[-1000:],
                            }
                        )
                        failures.append(entry)
                        break

                    commit_result = self._run_git(
                        [
                            "commit",
                            "-m",
                            f"{task.task_id}: record merged submodule revision {full_relative}",
                            "--",
                            child_relative,
                        ],
                        cwd=parent_repo,
                    )
                    recorded_commit = self._submodule_gitlink_ref(parent_repo, child_relative)
                    if commit_result.returncode != 0 or recorded_commit != expected_commit:
                        entry.update(
                            {
                                "reason": "parent_gitlink_commit_failed",
                                "returncode": commit_result.returncode,
                                "stdout": commit_result.stdout[-1000:],
                                "stderr": commit_result.stderr[-1000:],
                                "recorded_commit": recorded_commit,
                            }
                        )
                        failures.append(entry)
                        break
                    committed = True
                    entry.update(
                        {
                            "committed": True,
                            "recorded_commit": recorded_commit,
                            "commit": self._run_git(
                                ["rev-parse", "HEAD"], cwd=parent_repo
                            ).stdout.strip(),
                        }
                    )
                else:
                    entry["reason"] = "gitlink_already_recorded"

                chain.append(entry)
                expected_commit = self._run_git(
                    ["rev-parse", "HEAD"], cwd=parent_repo
                ).stdout.strip()
                current_checkout = parent_repo

        result: dict[str, Any] = {
            "attempted": True,
            "ok": not failures,
            "committed": committed,
            "paths": sorted(expected_commits),
            "expected_commits": expected_commits,
            "chain": chain,
            "failures": failures,
        }
        if failures:
            result["reason"] = "parent_gitlink_chain_failed"
        else:
            result["commit"] = self._run_git(
                ["rev-parse", "HEAD"], cwd=workspace
            ).stdout.strip()
            if not committed:
                result["reason"] = "parent_gitlinks_already_recorded"
        return result

    def _rollback_parent_merge_after_submodule_failure(
        self,
        workspace: Path,
        *,
        pre_merge_commit: str,
        failed_submodules: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Abort a parent merge when one of its configured child merges fails.

        Parent and configured submodule branches form one logical transaction.
        Git commits the parent merge first, so a child conflict used to leave a
        parent gitlink pointing at an unmerged child revision. Restore the
        parent to its pre-merge commit only after checking out the parent-index
        child revisions and confirming there are no unrelated tracked changes.
        ``git reset --merge`` preserves untracked files while removing the
        unpublishable merge ancestry, allowing the failed task to retry.
        """

        paths = sorted(
            {
                str(item.get("path") or "").strip().strip("/")
                for item in failed_submodules
                if str(item.get("path") or "").strip().strip("/") in self.worktree_submodule_paths
            }
        )
        if not pre_merge_commit or not paths:
            return {
                "attempted": False,
                "rolled_back": False,
                "reason": "no_failed_configured_root_submodules",
                "paths": paths,
            }

        current = self._run_git(["rev-parse", "HEAD"], cwd=workspace).stdout.strip()
        if current == pre_merge_commit:
            return {
                "attempted": True,
                "rolled_back": True,
                "reason": "parent_already_at_pre_merge_commit",
                "paths": paths,
            }

        changed_paths: list[str] = []
        unchanged_paths: list[str] = []
        for path in paths:
            diff = subprocess.run(
                ["git", "diff", "--quiet", pre_merge_commit, current, "--", path],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=False,
            )
            if diff.returncode == 0:
                unchanged_paths.append(path)
            else:
                # Return code 1 means the gitlink changed. Operational errors
                # remain conservative and retain the transactional rollback.
                changed_paths.append(path)
        if not changed_paths:
            return {
                "attempted": False,
                "rolled_back": False,
                "reason": "parent_gitlinks_unchanged",
                "paths": paths,
                "unchanged_paths": unchanged_paths,
            }
        paths = changed_paths

        align_before = subprocess.run(
            ["git", "submodule", "update", "--init", "--checkout", "--", *paths],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if align_before.returncode != 0:
            return {
                "attempted": True,
                "rolled_back": False,
                "reason": "failed_submodule_alignment_failed",
                "paths": paths,
                "returncode": align_before.returncode,
                "stdout": align_before.stdout[-1000:],
                "stderr": align_before.stderr[-1000:],
            }
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked_status.returncode != 0 or tracked_status.stdout.strip():
            return {
                "attempted": True,
                "rolled_back": False,
                "reason": "parent_has_tracked_changes_after_submodule_alignment",
                "paths": paths,
                "status": tracked_status.stdout[-1000:],
                "stderr": tracked_status.stderr[-1000:],
            }

        reset = subprocess.run(
            ["git", "reset", "--merge", pre_merge_commit],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if reset.returncode != 0:
            return {
                "attempted": True,
                "rolled_back": False,
                "reason": "parent_merge_reset_failed",
                "paths": paths,
                "returncode": reset.returncode,
                "stdout": reset.stdout[-1000:],
                "stderr": reset.stderr[-1000:],
            }
        align_after = subprocess.run(
            ["git", "submodule", "update", "--init", "--checkout", "--", *paths],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        result: dict[str, Any] = {
            "attempted": True,
            "rolled_back": align_after.returncode == 0,
            "paths": paths,
            "unchanged_paths": unchanged_paths,
            "pre_merge_commit": pre_merge_commit,
            "returncode": align_after.returncode,
            "stdout": align_after.stdout[-1000:],
            "stderr": align_after.stderr[-1000:],
        }
        if align_after.returncode != 0:
            result["reason"] = "post_reset_submodule_alignment_failed"
        return result

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
        if abort.returncode != 0 and (
            self._git_merge_head_in_repo(cwd) or self._unmerged_worktree_paths(cwd)
        ):
            reset = subprocess.run(
                ["git", "reset", "--merge"],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
            )
            fallback = {
                "attempted": True,
                "reset": reset.returncode == 0,
                "returncode": reset.returncode,
                "stdout": reset.stdout[-4000:],
                "stderr": reset.stderr[-4000:],
            }
            result["reset_merge_fallback"] = fallback
            if reset.returncode == 0:
                result["aborted"] = True
                result["reason"] = "reset_merge_fallback"
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
        attempt: int = 0,
        parent_relative: str = "",
    ) -> dict[str, Any]:
        conflicts = self._unmerged_gitlink_conflicts(workspace)
        if not conflicts:
            return {"repaired": False, "reason": "no_gitlink_conflicts"}
        repairs: list[dict[str, Any]] = []
        for relative, stages in sorted(conflicts.items()):
            full_relative = f"{parent_relative.rstrip('/')}/{relative}" if parent_relative else relative
            resolution = self._submodule_gitlink_resolution(
                full_relative,
                stages,
                task=task,
                attempt=attempt,
            )
            selected_commit = str(resolution.get("selected_commit") or "")
            if not selected_commit:
                repairs.append(
                    {
                        **resolution,
                        "repaired": False,
                        "reason": str(resolution.get("reason") or "no_safe_resolution"),
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
                    **resolution,
                    "repaired": update.returncode == 0,
                    "reason": str(resolution.get("selection_reason") or "selected_verified_descendant")
                    if update.returncode == 0
                    else "update_index_failed",
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
            return self._record_submodule_gitlink_repair_result(
                result, workspace=workspace, task=task, attempt=attempt
            )
        alignment_repairs = self._unmanaged_gitlink_checkout_alignment_repairs(workspace, repairs)
        checkout_alignment = self._preflight_resolved_gitlink_checkout_alignment(workspace, alignment_repairs)
        if not checkout_alignment["ready"]:
            result = {
                "repaired": False,
                "reason": "resolved_gitlink_checkout_alignment_unavailable",
                "repairs": repairs,
                "checkout_alignment": checkout_alignment,
            }
            return self._record_submodule_gitlink_repair_result(
                result, workspace=workspace, task=task, attempt=attempt
            )
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
            return self._record_submodule_gitlink_repair_result(
                result, workspace=workspace, task=task, attempt=attempt
            )
        merge_commit = self._run_git(["rev-parse", "HEAD"], cwd=workspace).stdout.strip()
        checkout_alignment = self._align_resolved_gitlink_checkouts(workspace, checkout_alignment)
        result = {
            "repaired": True,
            "reason": "committed_resolved_gitlinks",
            "repairs": repairs,
            "merge_commit": merge_commit,
            "checkout_alignment": checkout_alignment,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }
        return self._record_submodule_gitlink_repair_result(
            result, workspace=workspace, task=task, attempt=attempt
        )

    def _unmanaged_gitlink_checkout_alignment_repairs(
        self,
        workspace: Path,
        repairs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only embedded gitlinks that lack normal submodule ownership.

        Declared or daemon-managed submodules are reconciled by
        ``_merge_submodule_branches_to_main_in_repo`` and must remain on their
        active branch while that work completes.  An embedded repository without
        either declaration has no such owner; leaving it at the old commit makes
        the just-committed parent worktree dirty.
        """

        if workspace.resolve() != self.repo_root.resolve():
            return []
        managed = set(self.worktree_submodule_paths)
        managed.update(self._declared_submodule_paths(workspace))
        return [
            repair
            for repair in repairs
            if str(repair.get("path") or "") not in managed
        ]

    def _preflight_resolved_gitlink_checkout_alignment(
        self,
        workspace: Path,
        repairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Verify that resolved gitlinks can safely match their parent index.

        A parent merge that records a new unmanaged gitlink while leaving its
        embedded checkout on the old commit reports the parent as dirty
        immediately. Only align clean, available checkouts, and validate every
        target before committing the parent merge so an unsafe checkout leaves
        the merge abortable.
        """

        entries: list[dict[str, Any]] = []
        for repair in repairs:
            relative = str(repair.get("path") or "")
            selected_commit = str(repair.get("selected_commit") or "")
            entry: dict[str, Any] = {
                "path": relative,
                "selected_commit": selected_commit,
                "ready": False,
            }
            if not relative or not selected_commit or not self._repo_relative_path_safe(relative):
                entry["reason"] = "invalid_gitlink_checkout_target"
                entries.append(entry)
                continue
            checkout = (workspace / relative).resolve()
            if not self._is_git_worktree(checkout):
                entry["reason"] = "submodule_checkout_unavailable"
                entries.append(entry)
                continue
            if not self._git_commit_exists_in_repo(checkout, selected_commit):
                entry["reason"] = "selected_commit_unavailable"
                entries.append(entry)
                continue
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=checkout,
                text=True,
                capture_output=True,
                check=False,
            )
            if status.returncode != 0:
                entry["reason"] = "submodule_status_failed"
                entry["stderr"] = status.stderr[-4000:]
                entries.append(entry)
                continue
            if status.stdout.strip():
                entry["reason"] = "submodule_checkout_dirty"
                entry["status"] = status.stdout[-4000:]
                entries.append(entry)
                continue
            entry["ready"] = True
            entries.append(entry)
        return {"ready": all(entry["ready"] for entry in entries), "entries": entries}

    @staticmethod
    def _align_resolved_gitlink_checkouts(
        workspace: Path,
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        """Detach clean nested checkouts at the gitlinks committed by a merge."""

        entries: list[dict[str, Any]] = []
        for planned in preflight.get("entries", []):
            relative = str(planned.get("path") or "")
            selected_commit = str(planned.get("selected_commit") or "")
            checkout = (workspace / relative).resolve()
            result = subprocess.run(
                ["git", "checkout", "--detach", selected_commit],
                cwd=checkout,
                text=True,
                capture_output=True,
                check=False,
            )
            entries.append(
                {
                    "path": relative,
                    "selected_commit": selected_commit,
                    "aligned": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                }
            )
        return {"aligned": all(entry["aligned"] for entry in entries), "entries": entries}

    def _record_submodule_gitlink_repair_result(
        self,
        result: dict[str, Any],
        *,
        workspace: Path,
        task: PortalTask,
        attempt: int,
    ) -> dict[str, Any]:
        """Persist a bounded, retry-oriented audit trail for every gitlink conflict."""

        self._record_event("submodule_gitlink_conflict_repair", result)
        existing = load_json_dict(self.submodule_merge_diagnostics_path) or {}
        attempts = existing.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
        entry = {
            "timestamp": utc_now(),
            "task_id": task.task_id,
            "attempt": int(attempt),
            "workspace": str(workspace),
            "repaired": bool(result.get("repaired", False)),
            "retryable": not bool(result.get("repaired", False)),
            "reason": str(result.get("reason") or ""),
            "conflicts": list(result.get("repairs") or []),
        }
        attempts.append(entry)
        payload = {
            "schema_version": 1,
            "updated_at": entry["timestamp"],
            "latest": entry,
            "attempts": attempts[-SUBMODULE_MERGE_DIAGNOSTICS_MAX_ATTEMPTS:],
        }
        try:
            write_json_atomic(self.submodule_merge_diagnostics_path, payload)
        except OSError as exc:
            result["diagnostic_write_error"] = str(exc)[-1000:]
        else:
            result["diagnostic_path"] = str(self.submodule_merge_diagnostics_path)
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
        """Compatibility wrapper returning only a safely selected commit."""

        resolution = self._submodule_gitlink_resolution(relative, stages, task=task)
        return str(resolution.get("selected_commit") or "")

    def _submodule_gitlink_resolution(
        self,
        relative: str,
        stages: dict[str, str],
        *,
        task: PortalTask,
        attempt: int = 0,
    ) -> dict[str, Any]:
        """Select only a commit proven to contain both gitlink candidates.

        A divergent pair is merged in an isolated worktree and published under a
        deterministic recovery ref. The active submodule checkout is never used
        as an implicit third candidate. After a resolved parent merge commits,
        a separate clean-only alignment step may detach an unmanaged embedded
        checkout at the selected commit so the parent worktree remains clean.
        """

        source = (self.repo_root / relative).resolve()
        ours = str(stages.get("2") or "")
        theirs = str(stages.get("3") or "")
        base = str(stages.get("1") or "")
        diagnostic: dict[str, Any] = {
            "path": relative,
            "stages": dict(stages),
            "base_candidate": base,
            "ours_candidate": ours,
            "theirs_candidate": theirs,
            "reachable_merge_bases": [],
            "selected_commit": "",
            "selection_reason": "",
            "recovery_ref": "",
        }
        if not self._is_git_worktree(source):
            diagnostic["reason"] = "submodule_checkout_unavailable"
            return diagnostic
        if not ours or not theirs:
            diagnostic["reason"] = "missing_gitlink_candidate"
            return diagnostic
        missing_candidates = [
            candidate
            for candidate in (ours, theirs)
            if not self._git_commit_exists_in_repo(source, candidate)
        ]
        if missing_candidates:
            diagnostic["missing_candidates"] = missing_candidates
            available_candidates = [
                candidate
                for candidate in (ours, theirs)
                if candidate and candidate not in missing_candidates
            ]
            if len(available_candidates) == 1:
                diagnostic.update(
                    {
                        "selected_commit": available_candidates[0],
                        "selection_reason": "only_verified_available_candidate",
                        "reason": "selected_verified_available_candidate",
                    }
                )
                return diagnostic
            diagnostic["reason"] = "gitlink_candidate_unavailable"
            return diagnostic

        diagnostic["reachable_merge_bases"] = self._git_merge_bases_in_repo(source, ours, theirs)
        if ours == theirs:
            diagnostic.update(
                {
                    "selected_commit": ours,
                    "selection_reason": "identical_candidates",
                    "reason": "selected_verified_descendant",
                }
            )
            return diagnostic
        if self._git_ref_is_ancestor_in_repo(source, ours, theirs):
            diagnostic.update(
                {
                    "selected_commit": theirs,
                    "selection_reason": "theirs_verified_descendant_of_ours",
                    "reason": "selected_verified_descendant",
                }
            )
            return diagnostic
        if self._git_ref_is_ancestor_in_repo(source, theirs, ours):
            diagnostic.update(
                {
                    "selected_commit": ours,
                    "selection_reason": "ours_verified_descendant_of_theirs",
                    "reason": "selected_verified_descendant",
                }
            )
            return diagnostic
        if self._git_commit_subject_mentions_task(source, ours, task.task_id) and self._git_commit_subject_mentions_task(
            source, theirs, task.task_id
        ):
            # The target branch already contains an independently committed
            # outcome for this task. Preserve that current result instead of
            # creating a synthetic merge that combines two task attempts.
            diagnostic.update(
                {
                    "selected_commit": ours,
                    "selection_reason": "ours_equivalent_task_head",
                    "reason": "selected_equivalent_task_head",
                }
            )
            return diagnostic

        recovery_ref = self._submodule_recovery_ref(relative, ours, theirs)
        diagnostic["recovery_ref"] = recovery_ref
        recovered = self._resolve_git_commit_in_repo(source, recovery_ref)
        if recovered and self._commit_descends_from_both(source, recovered, ours, theirs):
            diagnostic.update(
                {
                    "selected_commit": recovered,
                    "selection_reason": "validated_deterministic_recovery_ref",
                    "reason": "selected_deterministic_recovery_ref",
                }
            )
            return diagnostic

        recovery = self._create_submodule_recovery_ref(
            source=source,
            relative=relative,
            ours=ours,
            theirs=theirs,
            recovery_ref=recovery_ref,
            task=task,
            attempt=attempt,
        )
        diagnostic["recovery"] = recovery
        selected = str(recovery.get("commit") or "")
        if selected and self._commit_descends_from_both(source, selected, ours, theirs):
            diagnostic.update(
                {
                    "selected_commit": selected,
                    "selection_reason": "created_deterministic_recovery_ref",
                    "reason": "selected_deterministic_recovery_ref",
                }
            )
            return diagnostic
        diagnostic["reason"] = str(recovery.get("reason") or "no_safe_resolution")
        return diagnostic

    def _git_commit_exists_in_repo(self, repo: Path, ref: str) -> bool:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _resolve_git_commit_in_repo(self, repo: Path, ref: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def _git_commit_subject_mentions_task(repo: Path, commit: str, task_id: str) -> bool:
        if not commit or not task_id:
            return False
        result = subprocess.run(
            ["git", "show", "-s", "--format=%s", commit],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        pattern = rf"(?<![A-Za-z0-9]){re.escape(task_id)}(?![A-Za-z0-9])"
        return bool(re.search(pattern, result.stdout.strip()))

    def _git_merge_bases_in_repo(self, repo: Path, ours: str, theirs: str) -> list[str]:
        result = subprocess.run(
            ["git", "merge-base", "--all", ours, theirs],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})

    def _commit_descends_from_both(self, repo: Path, commit: str, ours: str, theirs: str) -> bool:
        return bool(commit) and all(
            self._git_ref_is_ancestor_in_repo(repo, candidate, commit)
            for candidate in (ours, theirs)
        )

    @staticmethod
    def _submodule_recovery_ref(relative: str, ours: str, theirs: str) -> str:
        digest = hashlib.sha256(
            f"{relative}\0{ours}\0{theirs}".encode("utf-8", errors="surrogateescape")
        ).hexdigest()[:24]
        safe_path = re.sub(r"[^A-Za-z0-9._/-]+", "-", relative).strip("/.-") or "submodule"
        return f"refs/agent-supervisor/submodule-merge-recovery/{safe_path}/{digest}"

    def _submodule_recovery_worktree_root(self) -> Path:
        """Return an absolute recovery root outside every submodule checkout."""

        state_dir = self.state_path.parent
        if not state_dir.is_absolute():
            state_dir = self.repo_root / state_dir
        return (state_dir / "submodule-merge-recovery-worktrees").resolve()

    def _create_submodule_recovery_ref(
        self,
        *,
        source: Path,
        relative: str,
        ours: str,
        theirs: str,
        recovery_ref: str,
        task: PortalTask,
        attempt: int = 0,
    ) -> dict[str, Any]:
        """Create a clean merge commit without changing the active checkout."""

        if not source.exists() or not self._is_git_worktree(source):
            return {
                "created": False,
                "reason": "recovery_source_unavailable",
                "source": str(source),
            }
        recovery_root = self._submodule_recovery_worktree_root()
        recovery_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(f"{relative}\0{ours}\0{theirs}".encode()).hexdigest()[:16]
        workspace = recovery_root / f"{digest}-{os.getpid()}-{time.time_ns()}"
        add = subprocess.run(
            ["git", "worktree", "add", "--detach", str(workspace), ours],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            return {
                "created": False,
                "reason": "recovery_worktree_add_failed",
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
            }
        if not workspace.exists() or not self._is_git_worktree(workspace):
            return {
                "created": False,
                "reason": "recovery_worktree_unavailable",
                "workspace": str(workspace),
            }
        merge = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Daemon",
                "-c",
                "user.email=implementation-daemon@example.invalid",
                "merge",
                "--no-ff",
                "--no-edit",
                theirs,
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        nested_gitlink_repair: dict[str, Any] = {}
        effective_merge_returncode = merge.returncode
        if merge.returncode != 0:
            # A gitlink conflict can occur at every nesting level.  Reconcile it
            # in this detached recovery worktree, then continue validating the
            # recovery commit against both candidates.  This never mutates the
            # active submodule checkout.
            nested_gitlink_repair = self._repair_submodule_gitlink_merge_conflicts(
                workspace,
                task=task,
                attempt=attempt,
                parent_relative=relative,
            )
            if nested_gitlink_repair.get("repaired", False):
                effective_merge_returncode = 0
        commit = ""
        update_ref = subprocess.CompletedProcess([], 1, "", "merge did not complete")
        if effective_merge_returncode == 0:
            commit = self._run_git(["rev-parse", "HEAD"], cwd=workspace).stdout.strip()
            if self._commit_descends_from_both(source, commit, ours, theirs):
                update_ref = subprocess.run(
                    ["git", "update-ref", recovery_ref, commit],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
        remove = subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace)],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        created = bool(commit and effective_merge_returncode == 0 and update_ref.returncode == 0)
        result = {
            "created": created,
            "reason": "recovery_ref_created" if created else "recovery_merge_failed",
            "recovery_ref": recovery_ref,
            "commit": commit if created else "",
            "task_id": task.task_id,
            "merge_returncode": merge.returncode,
            "effective_merge_returncode": effective_merge_returncode,
            "merge_stdout": merge.stdout[-4000:],
            "merge_stderr": merge.stderr[-4000:],
            "update_ref_returncode": update_ref.returncode,
            "update_ref_stdout": str(update_ref.stdout)[-4000:],
            "update_ref_stderr": str(update_ref.stderr)[-4000:],
            "cleanup_returncode": remove.returncode,
            "cleanup_stderr": remove.stderr[-4000:],
        }
        if nested_gitlink_repair:
            result["nested_gitlink_repair"] = nested_gitlink_repair
        if remove.returncode != 0:
            result["preserved_recovery_workspace"] = str(workspace)
        return result

    def _merge_submodule_branches_to_main(
        self,
        branch_name: str,
        *,
        task: PortalTask,
        attempt: int,
        baseline_ref: str = "",
        changed_submodule_paths: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._merge_submodule_branches_to_main_in_repo(
            repo_path=self.repo_root,
            branch_name=branch_name,
            parent_relative="",
            task=task,
            attempt=attempt,
            baseline_ref=baseline_ref,
            changed_submodule_paths=changed_submodule_paths,
        )

    def _root_submodule_changed_in_task(
        self,
        branch_name: str,
        baseline_ref: str,
        relative: str,
    ) -> bool:
        """Return whether a task changed a configured top-level gitlink.

        A task worktree creates branches for every configured submodule. Those
        branches may contain unrelated prior work, so reconciling them merely
        because they exist can turn a successful parent task into a false
        failure. When the task baseline is available, reconcile only gitlinks
        changed by that task. An inconclusive diff preserves the existing
        conservative behavior.
        """

        if not baseline_ref:
            return True
        tree_entry = subprocess.run(
            ["git", "ls-tree", baseline_ref, "--", relative],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if tree_entry.returncode != 0:
            return True
        exact_gitlink = any(
            line.startswith("160000 ") and line.partition("\t")[2] == relative
            for line in tree_entry.stdout.splitlines()
        )
        # A configured path may point through a parent submodule to a deeper
        # repository. The root tree cannot express that nested branch change,
        # so it must be reconciled conservatively instead of being skipped.
        if not exact_gitlink:
            return True
        # ``git diff --quiet`` returns 1 for a normal, positive finding. Do
        # not route it through ``_run_git``, which correctly treats any
        # non-zero status as an operational failure for mutating commands.
        diff = subprocess.run(
            ["git", "diff", "--quiet", baseline_ref, branch_name, "--", relative],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if diff.returncode in (0, 1):
            return diff.returncode == 1
        return True

    def _merge_submodule_branches_to_main_in_repo(
        self,
        *,
        repo_path: Path,
        branch_name: str,
        parent_relative: str,
        task: PortalTask,
        attempt: int,
        baseline_ref: str = "",
        changed_submodule_paths: set[str] | None = None,
        checkpoint: MergeCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        stale_config_repair = self._repair_stale_submodule_worktree_configs(repo_path)
        relatives = self.worktree_submodule_paths if not parent_relative else tuple(self._declared_submodule_paths(repo_path))
        # Sort submodules by dependency order: leaf submodules merge first
        relatives = self._topological_sort_submodules(relatives, repo_path)
        # Resume from checkpoint if one exists (crash recovery)
        owns_checkpoint = checkpoint is None
        if checkpoint is None:
            checkpoint_dir = self.state_path.parent / "merge_checkpoints"
            checkpoint = MergeCheckpoint.resume(checkpoint_dir, branch_name)
            if checkpoint:
                self._record_event("merge_checkpoint_resumed", {
                    "branch_name": branch_name,
                    "task_id": task.task_id,
                    "previously_merged": list(checkpoint.merged_submodules.keys()),
                    "previously_failed": list(checkpoint.failed_submodules.keys()),
                })
            else:
                checkpoint = MergeCheckpoint.create(
                    checkpoint_dir=checkpoint_dir,
                    branch_name=branch_name,
                    task_id=task.task_id,
                    attempt=attempt,
                )
        for relative in relatives:
            full_relative = f"{parent_relative.rstrip('/')}/{relative}" if parent_relative else relative
            source = (self.repo_root / full_relative).resolve()
            merge_current_submodule = (
                changed_submodule_paths is None or full_relative in changed_submodule_paths
            )
            has_changed_descendant = bool(
                changed_submodule_paths
                and any(
                    path.startswith(f"{full_relative.rstrip('/')}/")
                    for path in changed_submodule_paths
                )
            )
            if changed_submodule_paths is not None and not (
                merge_current_submodule or has_changed_descendant
            ):
                continue
            # Skip submodules already merged in a previous checkpoint
            if checkpoint.is_already_merged(full_relative):
                results.append(checkpoint.merged_submodules[full_relative])
                if self._is_git_worktree(source):
                    results.extend(
                        self._merge_submodule_branches_to_main_in_repo(
                            repo_path=source,
                            branch_name=branch_name,
                            parent_relative=full_relative,
                            task=task,
                            attempt=attempt,
                            baseline_ref=baseline_ref,
                            changed_submodule_paths=changed_submodule_paths,
                            checkpoint=checkpoint,
                        )
                    )
                continue
            submodule_branch = self._submodule_worktree_branch_name(branch_name, full_relative)
            if not self._is_git_worktree(source):
                continue
            if not merge_current_submodule:
                results.extend(
                    self._merge_submodule_branches_to_main_in_repo(
                        repo_path=source,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                        task=task,
                        attempt=attempt,
                        baseline_ref=baseline_ref,
                        changed_submodule_paths=changed_submodule_paths,
                        checkpoint=checkpoint,
                    )
                )
                continue
            if (
                not parent_relative
                and baseline_ref
                and not self._root_submodule_changed_in_task(
                    branch_name,
                    baseline_ref,
                    relative,
                )
            ):
                result = {
                    "path": full_relative,
                    "branch": submodule_branch,
                    "default_branch": self._submodule_default_branch(relative, source),
                    "merged": True,
                    "reason": "unchanged_gitlink_in_task",
                }
                results.append(result)
                checkpoint.record_submodule(full_relative, result)
                continue
            if not self._git_ref_exists_in_repo(source, submodule_branch):
                # A parent branch may be unchanged or already cleaned while a
                # deeper daemon-owned branch still needs reconciliation.
                results.extend(
                    self._merge_submodule_branches_to_main_in_repo(
                        repo_path=source,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                        task=task,
                        attempt=attempt,
                        baseline_ref=baseline_ref,
                        changed_submodule_paths=changed_submodule_paths,
                        checkpoint=checkpoint,
                    )
                )
                continue
            default_branch = self._submodule_default_branch(relative, source)
            if self._git_ref_is_ancestor_in_repo(source, submodule_branch, default_branch):
                result = {
                    "path": full_relative,
                    "branch": submodule_branch,
                    "default_branch": default_branch,
                    "merged": True,
                    "reason": "already_merged",
                }
                if stale_config_repair.get("repairs"):
                    result["stale_submodule_worktree_config_repair"] = stale_config_repair
                results.append(result)
                checkpoint.record_submodule(full_relative, result)
                results.extend(
                    self._merge_submodule_branches_to_main_in_repo(
                        repo_path=source,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                        task=task,
                        attempt=attempt,
                        baseline_ref=baseline_ref,
                        changed_submodule_paths=changed_submodule_paths,
                        checkpoint=checkpoint,
                    )
                )
                continue
            dirty = self._run_git(["status", "--porcelain"], cwd=source).stdout.strip()
            dirty_paths = self._dirty_status_paths(dirty)
            changed_paths = self._branch_changed_paths_in_repo(
                source,
                submodule_branch,
                base_ref=default_branch,
            )
            dirty_overlap = self._overlapping_paths(dirty_paths, changed_paths)
            preserved_dirty_paths = sorted(set(dirty_paths) - set(dirty_overlap))
            if dirty_overlap:
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
                    dirty_paths=dirty_overlap,
                )
                if llm_merge_resolver.get("applied", False):
                    dirty = self._run_git(["status", "--porcelain"], cwd=source).stdout.strip()
                    dirty_paths = self._dirty_status_paths(dirty)
                    changed_paths = self._branch_changed_paths_in_repo(
                        source,
                        submodule_branch,
                        base_ref=default_branch,
                    )
                    dirty_overlap = self._overlapping_paths(dirty_paths, changed_paths)
                    preserved_dirty_paths = sorted(set(dirty_paths) - set(dirty_overlap))
                if not dirty_overlap:
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
                    result = {
                        "path": full_relative,
                        "branch": submodule_branch,
                        "default_branch": default_branch,
                        "merged": False,
                        "reason": "submodule_checkout_dirty",
                        "status": dirty,
                        "dirty_paths": dirty_overlap,
                        "preserved_dirty_paths": preserved_dirty_paths,
                        "llm_merge_resolver": llm_merge_resolver,
                    }
                    results.append(result)
                    checkpoint.record_submodule(full_relative, result)
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
                        result = {
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
                        results.append(result)
                        checkpoint.record_submodule(full_relative, result)
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
            nested_gitlink_repair: dict[str, Any] = {}
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
                    nested_gitlink_repair = self._repair_submodule_gitlink_merge_conflicts(
                        source,
                        task=task,
                        attempt=attempt,
                        parent_relative=full_relative,
                    )
                    if nested_gitlink_repair.get("repaired", False):
                        merge = subprocess.CompletedProcess(merge_command, 0, merge.stdout, merge.stderr)
                    elif nested_gitlink_repair.get("reason") != "no_gitlink_conflicts":
                        merge_abort_result = self._abort_failed_merge(source)
                    else:
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
            if nested_gitlink_repair:
                result["nested_gitlink_repair"] = nested_gitlink_repair
            if preserved_dirty_paths:
                result["preserved_dirty_paths"] = preserved_dirty_paths
            if merge.returncode == 0:
                result["commit"] = self._run_git(["rev-parse", "HEAD"], cwd=source).stdout.strip()
                # Post-merge validation: ensure submodule is in a healthy state
                validation = self._validate_merged_submodule_state(source, full_relative)
                if not validation.get("valid"):
                    result["post_merge_validation"] = validation
                    self._record_event("submodule_post_merge_validation_failed", {
                        "task_id": task.task_id,
                        "path": full_relative,
                        "validation": validation,
                    })
            results.append(result)
            # Record in checkpoint for crash recovery
            checkpoint.record_submodule(full_relative, result)
            if merge.returncode == 0:
                results.extend(
                    self._merge_submodule_branches_to_main_in_repo(
                        repo_path=source,
                        branch_name=branch_name,
                        parent_relative=full_relative,
                        task=task,
                        attempt=attempt,
                        baseline_ref=baseline_ref,
                        changed_submodule_paths=changed_submodule_paths,
                        checkpoint=checkpoint,
                    )
                )
        # Keep failed checkpoints durable.  A later reconciliation pass resumes
        # at the failed nested path while retaining proof of successful siblings.
        if owns_checkpoint:
            if checkpoint.failed_submodules:
                self._record_event(
                    "merge_checkpoint_pending",
                    {
                        **checkpoint.summary(),
                        "failed_paths": sorted(checkpoint.failed_submodules),
                        "retryable": True,
                    },
                )
            else:
                checkpoint.complete()
        return results

    def _topological_sort_submodules(
        self,
        relatives: tuple[str, ...] | list[str],
        repo_path: Path,
    ) -> list[str]:
        """Sort submodules so that dependencies are merged before dependents.

        Leaf submodules (those that don't contain other submodules) merge first.
        This prevents merge failures from stale pointers when submodule A
        depends on submodule B at the git level.
        """
        if len(relatives) <= 1:
            return list(relatives)

        # Build simple dependency graph: a submodule that contains other submodules
        # depends on those nested submodules being merged first
        depth: dict[str, int] = {}
        for relative in relatives:
            source = (repo_path / relative).resolve()
            if source.exists():
                nested = self._declared_submodule_paths(source)
                # Depth = number of nested submodules (more nested = merge later)
                depth[relative] = len(nested)
            else:
                depth[relative] = 0

        # Also check if any submodule path is a prefix of another (nested relationship)
        for rel_a in relatives:
            for rel_b in relatives:
                if rel_a != rel_b and rel_b.startswith(rel_a + "/"):
                    # rel_b is nested inside rel_a, so rel_b should merge first
                    depth[rel_a] = max(depth.get(rel_a, 0), depth.get(rel_b, 0) + 1)

        # Sort by depth ascending (leaves first)
        return sorted(relatives, key=lambda r: (depth.get(r, 0), r))

    def _validate_merged_submodule_state(self, source: Path, submodule_path: str) -> dict[str, Any]:
        """Validate a submodule is in a healthy state after merge.

        Checks:
        - Not in detached HEAD state
        - Working tree is clean (no dirty files)
        - Nested submodules are initialized
        """
        validation: dict[str, Any] = {"valid": True, "path": submodule_path, "checks": {}}

        # Check 1: Has a branch (not detached HEAD)
        branch = self._git_current_branch(source)
        validation["checks"]["has_branch"] = bool(branch)
        if not branch:
            validation["valid"] = False

        # Check 2: Working tree is clean
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            text=True,
            capture_output=True,
            check=False,
        )
        is_clean = status.returncode == 0 and not status.stdout.strip()
        validation["checks"]["clean"] = is_clean
        if not is_clean:
            validation["valid"] = False
            validation["dirty_paths"] = status.stdout.strip().splitlines()[:10]

        # Check 3: Nested submodules exist (if declared)
        try:
            nested = self._declared_submodule_paths(source)
            all_nested_valid = True
            for nested_path in nested[:5]:  # Limit to avoid long checks
                nested_target = source / nested_path
                if not nested_target.exists():
                    all_nested_valid = False
                    break
            validation["checks"]["nested_initialized"] = all_nested_valid
            if not all_nested_valid:
                validation["valid"] = False
        except Exception:
            validation["checks"]["nested_initialized"] = True  # Skip on error

        return validation

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

    @staticmethod
    def _submodule_cleanup_failures(cleanup: list[dict[str, Any]]) -> list[str]:
        failures: list[str] = []
        for item in cleanup:
            before = len(failures)
            path = str(item.get("path") or "")
            prefix = f"{path}: " if path else ""
            for error in item.get("errors") or []:
                error_text = str(error).strip()
                if error_text:
                    failures.append(f"{prefix}{error_text}")
            nested_cleanup = item.get("nested_submodule_cleanup") or []
            if isinstance(nested_cleanup, list):
                failures.extend(PortalImplementationDaemon._submodule_cleanup_failures(nested_cleanup))
            if item.get("cleaned") is False and len(failures) == before:
                failures.append(f"{prefix}cleanup incomplete")
        return failures

    @staticmethod
    def _managed_cleanup_branch(branch_name: str) -> bool:
        return branch_name.startswith("implementation/") or branch_name.startswith("rescue/worktree/")

    def _cleanup_already_merged_worktrees(self) -> dict[str, Any]:
        """Continuously drain inactive worktrees whose branches are already merged."""

        max_cleanups = max(0, int(self.merged_worktree_cleanup_max))
        if max_cleanups <= 0:
            return {"attempted": False, "reason": "merged_worktree_cleanup_disabled"}

        prune = subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            root_resolved = self.worktree_root.resolve()
        except OSError:
            root_resolved = self.worktree_root
        try:
            active_worktree = PortalTaskState.load(self.state_path).active_worktree_path
        except Exception:
            active_worktree = ""
        active_resolved: Path | None = None
        if active_worktree:
            try:
                active_resolved = Path(active_worktree).resolve()
            except OSError:
                active_resolved = Path(active_worktree)

        process_lines = self._list_process_commands()
        target_branch = self._main_branch_name()
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for entry in self._git_worktree_entries():
            if len(removed) >= max_cleanups:
                break
            path_text = str(entry.get("worktree") or "")
            if not path_text:
                continue
            worktree_path = Path(path_text)
            try:
                worktree_resolved = worktree_path.resolve()
                worktree_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue

            branch_name = str(entry.get("branch") or "").removeprefix("refs/heads/")
            detail = {"worktree_path": str(worktree_path), "branch": branch_name}
            if active_resolved is not None and worktree_resolved == active_resolved:
                skipped.append({**detail, "reason": "active_state_worktree"})
                continue
            if any(str(worktree_resolved) in line for line in process_lines):
                skipped.append({**detail, "reason": "active_process"})
                continue
            if not self._managed_cleanup_branch(branch_name):
                skipped.append({**detail, "reason": "unmanaged_branch"})
                continue
            if not self._git_ref_exists(branch_name):
                skipped.append({**detail, "reason": "branch_missing"})
                continue
            if not self._git_ref_is_ancestor(branch_name, target_branch):
                skipped.append({**detail, "reason": "branch_not_merged"})
                continue

            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=worktree_path,
                text=True,
                capture_output=True,
                check=False,
            )
            if status.returncode != 0:
                skipped.append(
                    {
                        **detail,
                        "reason": "status_failed",
                        "returncode": status.returncode,
                        "stderr": status.stderr[-4000:],
                    }
                )
                continue
            if status.stdout.strip():
                skipped.append(
                    {
                        **detail,
                        "reason": "dirty_worktree",
                        "status_short": status.stdout.splitlines()[:20],
                    }
                )
                continue

            cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
            removed.append({**detail, "cleanup_result": cleanup_result})

        result = {
            "attempted": True,
            "worktree_root": str(self.worktree_root),
            "target_branch": target_branch,
            "max_cleanups": max_cleanups,
            "prune_returncode": prune.returncode,
            "prune_stdout": prune.stdout[-4000:],
            "prune_stderr": prune.stderr[-4000:],
            "removed_count": sum(1 for item in removed if item["cleanup_result"].get("cleaned", False)),
            "skipped_count": len(skipped),
            "removed": removed,
            "skipped": skipped[:50],
        }
        if removed:
            self._record_event("merged_worktree_cleanup", result)
        return result

    def _cleanup_merged_worktree(self, worktree_path: Path | None, branch_name: str) -> dict[str, Any]:
        started_at = utc_now()
        lease: WorktreeLease | None = None
        lease_key: Path | None = None
        if worktree_path is not None:
            try:
                lease_key = worktree_path.resolve()
            except OSError:
                lease_key = worktree_path
            lease = self._worktree_pool_leases.pop(lease_key, None)
        if lease is not None:
            pool_release = lease.release(reusable=True)
            if pool_release.get("released", False):
                deleted_branch = False
                branch_error = ""
                try:
                    if self._git_ref_exists(branch_name):
                        self._run_git(["branch", "-D", branch_name], cwd=self.repo_root)
                        deleted_branch = True
                except RuntimeError as exc:
                    branch_error = str(exc)
                result = {
                    "cleaned": not branch_error,
                    "branch": branch_name,
                    "worktree_path": str(worktree_path or ""),
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "removed_worktree": not bool(pool_release.get("pooled", False)),
                    "deleted_branch": deleted_branch,
                    "submodule_cleanup": [],
                    "pooled": bool(pool_release.get("pooled", False)),
                    "pool_release": pool_release,
                }
                if branch_error:
                    result["error"] = branch_error
                self._record_event("cleanup_finished", result)
                return result

        removed_worktree = False
        deleted_branch = False
        submodule_cleanup: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            if worktree_path is not None:
                submodule_cleanup = self._cleanup_worktree_submodules(worktree_path, branch_name)
            if worktree_path is not None and (
                worktree_path.exists() or self._worktree_path_registered_in_repo(self.repo_root, worktree_path)
            ):
                self._run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=self.repo_root)
                removed_worktree = True
            if self._git_ref_exists(branch_name):
                self._run_git(["branch", "-D", branch_name], cwd=self.repo_root)
                deleted_branch = True
        except RuntimeError as exc:
            errors.append(str(exc))
        errors.extend(self._submodule_cleanup_failures(submodule_cleanup))

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
        _depth: int = 0,
    ) -> list[dict[str, Any]]:
        # Guard against infinite recursion in deeply nested submodules
        max_depth = int(os.environ.get("IPFS_ACCELERATE_AGENT_SUBMODULE_MAX_DEPTH", "10"))
        if _depth >= max_depth:
            return [{"error": f"max_recursion_depth_{max_depth}", "path": parent_relative}]
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
            target_is_registered_worktree = self._worktree_path_registered_in_repo(source, target)
            target_is_independent_checkout = (
                not target_is_registered_worktree and self._is_git_worktree(target)
            )
            if target_is_registered_worktree:
                if target.exists():
                    nested_cleanup = self._cleanup_worktree_submodules(
                        target,
                        branch_name,
                        parent_relative=full_relative,
                        _depth=_depth + 1,
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
            nested_failures = self._submodule_cleanup_failures(nested_cleanup)
            results.append(
                {
                    "path": full_relative,
                    "branch": submodule_branch,
                    "removed_worktree": removed_worktree,
                    "deleted_branch": deleted_branch,
                    "cleaned": not errors and not nested_failures,
                    "errors": errors,
                    "nested_submodule_cleanup": nested_cleanup,
                    "independent_checkout": target_is_independent_checkout,
                }
            )
        return results

    def _cleanup_stale_worktrees(self, *, max_age_seconds: float = 0) -> dict[str, Any]:
        """Remove worktrees whose branches are unmerged but have been inactive too long.

        This prevents orphaned worktrees from accumulating after failed or
        abandoned implementations. Only removes worktrees under our managed root
        that have no active process and whose branch hasn't been updated within
        ``max_age_seconds`` (default from env: 6 hours).
        """
        if max_age_seconds <= 0:
            max_age_seconds = float(
                os.environ.get("IPFS_ACCELERATE_AGENT_STALE_WORKTREE_SECONDS", "21600")
            )
        if max_age_seconds <= 0:
            return {"attempted": False, "reason": "stale_cleanup_disabled"}

        try:
            root_resolved = self.worktree_root.resolve()
        except OSError:
            return {"attempted": False, "reason": "worktree_root_unresolvable"}

        try:
            active_worktree = PortalTaskState.load(self.state_path).active_worktree_path
        except Exception:
            active_worktree = ""
        active_resolved: Path | None = None
        if active_worktree:
            try:
                active_resolved = Path(active_worktree).resolve()
            except OSError:
                active_resolved = Path(active_worktree)

        process_lines = self._list_process_commands()
        now_mono = time.time()
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for entry in self._git_worktree_entries():
            path_text = str(entry.get("worktree") or "")
            if not path_text:
                continue
            worktree_path = Path(path_text)
            try:
                worktree_resolved = worktree_path.resolve()
                worktree_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue

            branch_name = str(entry.get("branch") or "").removeprefix("refs/heads/")
            detail = {"worktree_path": str(worktree_path), "branch": branch_name}

            if active_resolved is not None and worktree_resolved == active_resolved:
                skipped.append({**detail, "reason": "active_state_worktree"})
                continue
            if any(str(worktree_resolved) in line for line in process_lines):
                skipped.append({**detail, "reason": "active_process"})
                continue
            if not self._managed_cleanup_branch(branch_name):
                skipped.append({**detail, "reason": "unmanaged_branch"})
                continue

            # Check age by looking at the most recent commit timestamp on the branch
            try:
                age_result = subprocess.run(
                    ["git", "log", "-1", "--format=%ct", branch_name],
                    cwd=self.repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if age_result.returncode != 0 or not age_result.stdout.strip():
                    skipped.append({**detail, "reason": "cannot_determine_age"})
                    continue
                last_commit_time = float(age_result.stdout.strip())
                age_seconds = now_mono - last_commit_time
            except (ValueError, OSError):
                skipped.append({**detail, "reason": "age_parse_error"})
                continue

            if age_seconds < max_age_seconds:
                skipped.append({**detail, "reason": "not_stale_yet", "age_seconds": age_seconds})
                continue

            # Stale: remove it
            cleanup_result = self._cleanup_merged_worktree(worktree_path, branch_name)
            removed.append({**detail, "age_seconds": age_seconds, "cleanup_result": cleanup_result})

        result = {
            "attempted": True,
            "max_age_seconds": max_age_seconds,
            "removed_count": len(removed),
            "skipped_count": len(skipped),
            "removed": removed,
            "skipped": skipped[:30],
        }
        if removed:
            self._record_event("stale_worktree_cleanup", result)
        return result

    def _cleanup_stale_locks(self, *, max_age_seconds: float = 0) -> dict[str, Any]:
        """Remove stale .lock files that persist after crashes/SIGKILL.

        Lock files older than ``max_age_seconds`` (default 30 minutes) are
        force-removed to prevent deadlocks in long-running supervisors.
        """
        if max_age_seconds <= 0:
            max_age_seconds = float(
                os.environ.get("IPFS_ACCELERATE_AGENT_STALE_LOCK_SECONDS", "1800")
            )
        if max_age_seconds <= 0:
            return {"attempted": False, "reason": "lock_cleanup_disabled"}

        lock_patterns = [
            self.repo_root / ".git" / "*.lock",
            self.repo_root / ".git" / "refs" / "**" / "*.lock",
        ]
        # Also check state directory for merge resolver locks
        state_dir = self.state_path.parent if self.state_path else None

        now_mono = time.time()
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        import glob as glob_mod
        lock_files: list[Path] = []
        for pattern in lock_patterns:
            lock_files.extend(Path(p) for p in glob_mod.glob(str(pattern), recursive=True))
        if state_dir and state_dir.exists():
            lock_files.extend(state_dir.glob("*.lock"))

        for lock_path in lock_files:
            try:
                stat = lock_path.stat()
                age_seconds = now_mono - stat.st_mtime
            except OSError:
                continue

            detail = {"lock_path": str(lock_path), "age_seconds": age_seconds}
            if age_seconds < max_age_seconds:
                skipped.append({**detail, "reason": "not_stale"})
                continue

            try:
                lock_path.unlink()
                removed.append(detail)
            except OSError as exc:
                skipped.append({**detail, "reason": "unlink_failed", "error": str(exc)})

        result = {
            "attempted": True,
            "max_age_seconds": max_age_seconds,
            "removed_count": len(removed),
            "skipped_count": len(skipped),
            "removed": removed,
            "skipped": skipped[:20],
        }
        if removed:
            self._record_event("stale_lock_cleanup", result)
        return result

    def _periodic_maintenance(self) -> dict[str, Any]:
        """Run periodic maintenance tasks to keep the supervisor healthy for 24/7 operation.

        This includes:
        - Cleaning up stale worktrees (abandoned after failures)
        - Removing stale lock files (orphaned after crashes)
        - Preserving generated submodule artifacts and reporting user-dirty state
        - Git garbage collection (loose objects, reflogs, repacking)
        - Task queue compaction (removing stale entries)
        """
        now_monotonic = time.monotonic()
        if (
            self._last_periodic_maintenance_monotonic is not None
            and self.maintenance_interval_seconds > 0
        ):
            elapsed_seconds = max(
                0.0,
                now_monotonic - self._last_periodic_maintenance_monotonic,
            )
            if elapsed_seconds < self.maintenance_interval_seconds:
                return {
                    "ran": False,
                    "reason": "cooldown",
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "interval_seconds": self.maintenance_interval_seconds,
                }
        self._last_periodic_maintenance_monotonic = now_monotonic
        results: dict[str, Any] = {
            "ran": True,
            "interval_seconds": self.maintenance_interval_seconds,
        }
        try:
            results["stale_worktrees"] = self._cleanup_stale_worktrees()
        except Exception as exc:
            results["stale_worktrees"] = {"error": str(exc)}
        try:
            results["stale_locks"] = self._cleanup_stale_locks()
        except Exception as exc:
            results["stale_locks"] = {"error": str(exc)}
        try:
            results["dirty_submodule_reset"] = self._reset_persistently_dirty_submodules()
        except Exception as exc:
            results["dirty_submodule_reset"] = {"error": str(exc)}
        # Every shard shares the repository object database, but each shard
        # keeps separate runtime state. Letting a new secondary shard see an
        # empty GC state triggers an immediate aggressive repack and blocks
        # all lanes. The primary shard owns repository-wide garbage collection.
        if self.task_shard_count > 1 and self.task_shard_index != 0:
            results["git_gc"] = {"ran": False, "reason": "non_primary_shard"}
        else:
            try:
                results["git_gc"] = self.git_gc.run_if_needed()
            except Exception as exc:
                results["git_gc"] = {"error": str(exc)}
        try:
            # Compact queue by canonical work identity, not board-local display id.
            removed = self.task_queue.compact(self._active_canonical_task_cids)
            results["queue_compact"] = {"removed": removed}
        except Exception as exc:
            results["queue_compact"] = {"error": str(exc)}
        return results

    def _reset_persistently_dirty_submodules(self) -> dict[str, Any]:
        """Preserve generated dirt and report remaining submodules without resetting.

        A previous maintenance path used ``checkout --force`` and ``submodule
        update --force`` here.  That could destroy user work merely because a
        checkout stayed dirty across daemon passes.  Reconciliation now decides
        whether a gitlink is actually relevant to a merge, so destructive reset
        is neither necessary nor safe.
        """
        max_dirty_attempts = int(os.environ.get("IPFS_ACCELERATE_AGENT_MAX_DIRTY_ATTEMPTS", "3"))
        if max_dirty_attempts <= 0:
            return {"attempted": False, "reason": "disabled"}
        preservation = self._preserve_generated_nested_worktree_directories()
        dirty_submodules: list[dict[str, Any]] = []
        for relative, source in self._initialized_submodule_worktrees():
            status = self._run_git(
                ["status", "--porcelain", "--untracked-files=all"],
                cwd=source,
            ).stdout.strip()
            if status:
                dirty_submodules.append(
                    {
                        "path": relative,
                        "reset_ok": False,
                        "update_ok": False,
                        "preserved": True,
                        "reason": "non_destructive_reconciliation",
                        "dirty_paths": self._dirty_status_paths(status),
                    }
                )
        result = {
            "attempted": True,
            "dirty_count": len(dirty_submodules),
            "reset": dirty_submodules,
            "generated_artifact_preservation": preservation,
        }
        if dirty_submodules or preservation:
            self._record_event("dirty_submodule_reset_deferred", result)
        return result

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

    def _restore_generated_dirty_merge_overlap(
        self,
        branch_name: str,
        *,
        cwd: Path | None = None,
        ignore_paths: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        workspace = cwd or self.repo_root
        dirty_paths = self._dirty_worktree_paths(workspace)
        if not dirty_paths:
            return []
        overlap = dirty_paths & self._branch_changed_paths(branch_name)
        if ignore_paths:
            overlap -= ignore_paths
        return self._restore_generated_dirty_paths(
            workspace,
            sorted(overlap),
            reason="generated_dirty_merge_overlap",
        )

    def _reconcile_generated_dirty_submodule_overlap(
        self,
        workspace: Path,
        dirty_paths: Sequence[str],
        *,
        branch_name: str,
        task: PortalTask,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative in dirty_paths:
            if not self._repo_relative_path_safe(relative):
                continue
            if not self._path_is_generated_status_output(relative):
                continue
            source = (workspace / relative).resolve()
            if not self._is_git_worktree(source):
                continue
            status = self._run_git(["status", "--porcelain", "--untracked-files=all"], cwd=source).stdout.strip()
            if not status:
                continue
            submodule_dirty_paths = self._dirty_status_paths(status)
            generated_dirty_paths = [
                path
                for path in submodule_dirty_paths
                if self._submodule_dirty_path_is_generated_status(relative, path)
            ]
            if set(generated_dirty_paths) != set(submodule_dirty_paths):
                results.append(
                    {
                        "path": relative,
                        "reconciled": False,
                        "reason": "submodule_has_non_generated_dirty_paths",
                        "dirty_paths": submodule_dirty_paths,
                        "generated_dirty_paths": generated_dirty_paths,
                    }
                )
                continue
            current_branch = self._git_current_branch(source)
            default_branch = self._submodule_default_branch(relative, source)
            if current_branch and current_branch != default_branch:
                results.append(
                    {
                        "path": relative,
                        "reconciled": False,
                        "reason": "submodule_not_on_default_branch",
                        "current_branch": current_branch,
                        "default_branch": default_branch,
                    }
                )
                continue

            generated_commit = self._commit_generated_submodule_paths(
                source,
                generated_dirty_paths,
                subject=f"{task.task_id}: reconcile generated submodule status",
            )
            submodule_branch = self._submodule_worktree_branch_name(branch_name, relative)
            submodule_merge: dict[str, Any] = {}
            if self._git_ref_exists_in_repo(source, submodule_branch) and not self._git_ref_is_ancestor_in_repo(
                source,
                submodule_branch,
                default_branch,
            ):
                submodule_merge = self._merge_generated_submodule_branch(
                    source,
                    submodule_branch,
                    default_branch=default_branch,
                )
            parent_commit = self._commit_specific_path(
                workspace,
                relative,
                subject=f"{task.task_id}: update generated submodule pointer",
            )
            reconciled = bool(parent_commit.get("committed") or parent_commit.get("reason") == "no_changes")
            if submodule_merge and not submodule_merge.get("merged", False):
                reconciled = False
            result = {
                "path": relative,
                "reconciled": reconciled,
                "reason": "generated_submodule_status_committed" if reconciled else "generated_submodule_commit_failed",
                "dirty_paths": submodule_dirty_paths,
                "generated_dirty_paths": generated_dirty_paths,
                "generated_commit": generated_commit,
                "submodule_branch": submodule_branch,
                "submodule_merge": submodule_merge,
                "parent_commit": parent_commit,
            }
            results.append(result)
        if results:
            self._record_event(
                "generated_dirty_submodule_reconciliation",
                {"main_worktree_path": str(workspace), "branch": branch_name, "results": results},
            )
        return results

    def _submodule_dirty_path_is_generated_status(self, submodule_relative: str, dirty_path: str) -> bool:
        if not self._repo_relative_path_safe(dirty_path):
            return False
        parent_relative = f"{submodule_relative.rstrip('/')}/{dirty_path.lstrip('/')}"
        return (
            self._path_is_generated_status_output(parent_relative)
            or self._path_is_generated_status_output(dirty_path)
            or self._path_is_generated_worktree_artifact(parent_relative)
            or self._path_is_generated_worktree_artifact(dirty_path)
        )

    def _commit_generated_submodule_paths(self, repo: Path, relative_paths: Sequence[str], *, subject: str) -> dict[str, Any]:
        safe_paths = [path for path in dict.fromkeys(relative_paths) if self._repo_relative_path_safe(path)]
        if not safe_paths:
            return {"committed": False, "reason": "no_safe_generated_paths", "repo": str(repo)}
        add = subprocess.run(
            ["git", "add", "--", *safe_paths],
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
                "paths": safe_paths,
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
            }
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *safe_paths],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 0:
            return {"committed": False, "reason": "no_staged_changes", "repo": str(repo), "paths": safe_paths}
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
                *safe_paths,
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
                "paths": safe_paths,
                "returncode": commit.returncode,
                "stdout": commit.stdout[-4000:],
                "stderr": commit.stderr[-4000:],
            }
        commit_ref = self._run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
        return {"committed": True, "repo": str(repo), "paths": safe_paths, "commit": commit_ref}

    def _merge_generated_submodule_branch(
        self,
        repo: Path,
        submodule_branch: str,
        *,
        default_branch: str,
    ) -> dict[str, Any]:
        merge_command = ["git", "merge", "--ff-only", submodule_branch]
        merge = subprocess.run(
            merge_command,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        ff_only_result = {
            "returncode": merge.returncode,
            "stdout": merge.stdout[-4000:],
            "stderr": merge.stderr[-4000:],
        }
        if merge.returncode != 0:
            merge_command = ["git", "merge", "--no-ff", "--no-edit", submodule_branch]
            merge = subprocess.run(
                merge_command,
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
        result = {
            "branch": submodule_branch,
            "default_branch": default_branch,
            "merged": merge.returncode == 0,
            "returncode": merge.returncode,
            "command": merge_command,
            "stdout": merge.stdout[-4000:],
            "stderr": merge.stderr[-4000:],
            "ff_only_result": ff_only_result,
            "commit": "",
        }
        if merge.returncode == 0:
            result["commit"] = self._run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
        else:
            result["merge_abort_result"] = self._abort_failed_merge(repo)
        return result

    def _restore_generated_dirty_paths(
        self,
        workspace: Path,
        dirty_paths: Sequence[str],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative in dirty_paths:
            if not self._path_is_generated_status_output(relative):
                continue
            was_dirty = relative in self._dirty_worktree_paths(workspace)
            self._restore_or_remove_generated_path_for_commit(workspace, relative)
            still_dirty = relative in self._dirty_worktree_paths(workspace)
            results.append(
                {
                    "path": relative,
                    "restored": was_dirty and not still_dirty,
                    "reason": reason,
                }
            )
        if results:
            self._record_event(
                "generated_dirty_path_restore",
                {"main_worktree_path": str(workspace), "reason": reason, "results": results},
            )
        return results

    def _path_is_generated_status_output(self, relative: str) -> bool:
        if self._path_is_generated_worktree_artifact(relative):
            return True
        from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
            generated_guardrail_status_filters,
            path_is_generated_status_output,
        )

        discovery_dir = self.state_path.parent.parent / "discovery"
        additional_paths = [
            path
            for path in (
                self.objective_path,
                *self.generated_status_paths,
            )
            if path is not None
        ]
        additional_prefixes = [
            path
            for path in (
                self.objective_bundle_dir,
                self.state_path.parent,
            )
            if path is not None
        ]
        generated_paths, generated_prefixes = generated_guardrail_status_filters(
            todo_path=self.todo_path,
            discovery_dir=discovery_dir,
            repo_root=self.repo_root,
            additional_generated_paths=additional_paths,
            additional_generated_prefixes=additional_prefixes,
        )
        return path_is_generated_status_output(
            relative,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        )

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

    def _resolve_generated_markdown_conflicts(self, cwd: Path) -> list[dict[str, object]]:
        allowed_paths: list[Path] = []
        allowed_dirs: list[Path] = []
        if self.objective_path is not None:
            allowed_paths.append(self.objective_path)
        if self.objective_bundle_dir is not None:
            allowed_dirs.append(self.objective_bundle_dir)
        if not allowed_paths and not allowed_dirs:
            return []
        results = resolve_append_only_markdown_conflicts(
            repo_root=cwd,
            allowed_paths=allowed_paths,
            allowed_dirs=allowed_dirs,
        )
        if results:
            self._record_event(
                "generated_markdown_conflict_repair",
                {"main_worktree_path": str(cwd), "results": results},
            )
        return results

    def _resolve_launch_readiness_conflicts(self, cwd: Path) -> list[dict[str, object]]:
        results = resolve_launch_readiness_conflicts(repo_root=cwd)
        if results:
            self._record_event(
                "launch_readiness_conflict_repair",
                {"main_worktree_path": str(cwd), "results": results},
            )
        return results

    def _resolve_reconciliation_guardrail_todo_conflicts(self, cwd: Path) -> list[dict[str, object]]:
        results = resolve_reconciliation_guardrail_todo_conflicts(repo_root=cwd)
        if results:
            self._record_event(
                "reconciliation_guardrail_todo_conflict_repair",
                {"main_worktree_path": str(cwd), "results": results},
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

    @staticmethod
    def _branch_changed_paths_in_repo(
        repo_path: Path,
        branch_name: str,
        *,
        base_ref: str,
    ) -> set[str] | None:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...{branch_name}"],
            cwd=repo_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    @staticmethod
    def _paths_overlap(left: str, right: str) -> bool:
        left = left.rstrip("/")
        right = right.rstrip("/")
        return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")

    @classmethod
    def _overlapping_paths(
        cls,
        dirty_paths: Sequence[str],
        changed_paths: set[str] | None,
    ) -> list[str]:
        if changed_paths is None:
            return sorted(set(dirty_paths))
        return sorted(
            {
                dirty
                for dirty in dirty_paths
                if any(cls._paths_overlap(dirty, changed) for changed in changed_paths)
            }
        )

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

    def _initialized_submodule_worktrees(self) -> list[tuple[str, Path]]:
        """Return initialized submodules at every declared nesting level."""

        discovered: list[tuple[str, Path]] = []
        seen: set[str] = set()
        queue: list[tuple[Path, str, str]] = []
        top_level = list(dict.fromkeys([*self.worktree_submodule_paths, *self._declared_submodule_paths(self.repo_root)]))
        for relative in top_level:
            queue.append((self.repo_root, "", relative))
        while queue:
            parent_repo, parent_relative, relative = queue.pop(0)
            full_relative = f"{parent_relative}/{relative}" if parent_relative else relative
            if full_relative in seen or not self._repo_relative_path_safe(full_relative):
                continue
            seen.add(full_relative)
            source = (parent_repo / relative).resolve()
            if not self._is_git_worktree(source):
                continue
            discovered.append((full_relative, source))
            for nested in self._declared_submodule_paths(source):
                queue.append((source, full_relative, nested))
        return discovered

    def _preserve_generated_nested_worktree_directories(self) -> list[dict[str, Any]]:
        """Move untracked nested ``tmp`` trees into durable supervisor state.

        These directories are generated by nested worktree tooling, but may hold
        logs or recovery material.  Moving instead of deleting them makes the
        checkout mergeable without losing evidence.
        """

        preserved: list[dict[str, Any]] = []
        preservation_root = self.state_path.parent / "preserved-nested-worktree-artifacts"
        for submodule_relative, source in self._initialized_submodule_worktrees():
            for directory_name in sorted(GENERATED_NESTED_WORKTREE_DIR_NAMES):
                generated_dir = source / directory_name
                if not generated_dir.is_dir() or generated_dir.is_symlink():
                    continue
                tracked = subprocess.run(
                    ["git", "ls-files", "--", directory_name],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if tracked.returncode != 0 or tracked.stdout.strip():
                    continue
                status = subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=all", "--", directory_name],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                status_lines = [line for line in status.stdout.splitlines() if line.strip()]
                if status.returncode != 0 or not status_lines or any(not line.startswith("??") for line in status_lines):
                    continue
                safe_submodule = self._safe_ref_path_fragment(submodule_relative.replace("/", "-"))
                content_fingerprint = hashlib.sha256(
                    "\n".join(sorted(status_lines)).encode("utf-8", errors="surrogateescape")
                ).hexdigest()[:12]
                destination_dir = preservation_root / safe_submodule
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{directory_name}-{content_fingerprint}-{time.time_ns()}"
                try:
                    shutil.move(str(generated_dir), str(destination))
                except OSError as exc:
                    preserved.append(
                        {
                            "path": f"{submodule_relative}/{directory_name}",
                            "preserved": False,
                            "reason": "preservation_move_failed",
                            "error": str(exc)[-1000:],
                        }
                    )
                    continue
                preserved.append(
                    {
                        "path": f"{submodule_relative}/{directory_name}",
                        "preserved": True,
                        "reason": "generated_nested_worktree_directory_moved",
                        "destination": str(destination),
                    }
                )
        if preserved:
            self._record_event(
                "generated_nested_worktree_artifacts_preserved",
                {"results": preserved},
            )
        return preserved

    def _gitlink_commit_at_ref(self, ref: str, relative: str) -> str:
        if not ref or not self._repo_relative_path_safe(relative):
            return ""
        result = subprocess.run(
            ["git", "ls-tree", ref, "--", relative],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        for line in result.stdout.splitlines():
            metadata, separator, path = line.partition("\t")
            fields = metadata.split()
            if separator and path == relative and len(fields) >= 3 and fields[0] == "160000":
                return fields[2]
        return ""

    def _dirty_gitlink_is_unchanged_for_candidates(
        self,
        relative: str,
        candidates: Sequence[dict[str, Any]],
        *,
        target_branch: str,
    ) -> bool:
        """Prove that a dirty gitlink is unrelated to every pending merge."""

        target_commit = self._gitlink_commit_at_ref(target_branch, relative)
        if not target_commit:
            return False
        compared = 0
        for event in candidates:
            branch = str(event.get("branch") or "")
            implementation_commit = str(event.get("implementation_commit") or "")
            merge_ref = branch if branch and self._git_ref_exists(branch) else implementation_commit
            candidate_commit = self._gitlink_commit_at_ref(merge_ref, relative)
            if not candidate_commit:
                return False
            compared += 1
            if candidate_commit != target_commit:
                return False
        return compared > 0

    def _reconciliation_blocking_dirty_paths(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        target_branch: str,
    ) -> tuple[list[str], list[str]]:
        blocking: list[str] = []
        nonblocking: list[str] = []
        for relative in sorted(self._dirty_worktree_paths(self.repo_root)):
            state_relative = ""
            try:
                state_relative = self.state_path.parent.resolve().relative_to(self.repo_root.resolve()).as_posix()
            except (OSError, ValueError):
                pass
            if state_relative and self._path_matches_prefix(relative, state_relative):
                nonblocking.append(relative)
                continue
            if self._dirty_gitlink_is_unchanged_for_candidates(
                relative,
                candidates,
                target_branch=target_branch,
            ):
                nonblocking.append(relative)
            else:
                blocking.append(relative)
        return blocking, nonblocking

    def _reconcile_failed_merges(
        self,
        *,
        skip_task_ids: set[str] | None = None,
        deprioritized_task_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        target_branch = self._main_branch_name()
        deprioritized_task_ids = {str(task_id) for task_id in (deprioritized_task_ids or set()) if str(task_id)}
        candidates = self._failed_merge_candidates(skip_task_ids=skip_task_ids)
        fresh_candidates, stale_candidates = self._partition_stale_failed_merge_candidates(candidates)
        for event in stale_candidates:
            result = self._stale_failed_merge_candidate_result(event)
            self._record_event("merge_reconciled", result)
            results.append(result)
        candidates = fresh_candidates
        if candidates:
            nested_artifact_preservation = self._preserve_generated_nested_worktree_directories()
            main_checkout_dirty_paths, nonblocking_dirty_paths = self._reconciliation_blocking_dirty_paths(
                candidates,
                target_branch=target_branch,
            )
            if main_checkout_dirty_paths:
                result = {
                    "resolved": False,
                    "reason": "main_checkout_dirty",
                    "candidate_count": len(candidates),
                    "processed_count": 0,
                    "dirty_paths": main_checkout_dirty_paths,
                }
                if nested_artifact_preservation:
                    result["nested_artifact_preservation"] = nested_artifact_preservation
                if nonblocking_dirty_paths:
                    result["nonblocking_dirty_paths"] = nonblocking_dirty_paths
                self._record_event("merge_reconciliation_deferred", result)
                results.append(result)
                return results
            if nonblocking_dirty_paths or nested_artifact_preservation:
                self._record_event(
                    "merge_reconciliation_nonblocking_checkout_state",
                    {
                        "nonblocking_dirty_paths": nonblocking_dirty_paths,
                        "nested_artifact_preservation": nested_artifact_preservation,
                        "candidate_count": len(candidates),
                    },
                )
        max_merges = int(self.merge_reconciliation_max_merges)
        selected_candidates = self._select_failed_merge_candidates_for_reconciliation(
            candidates,
            max_merges,
            deprioritized_task_ids=deprioritized_task_ids,
        )
        deferred_by_strategy = [
            str(event.get("task_id") or "")
            for event in candidates
            if str(event.get("task_id") or "") in deprioritized_task_ids
        ]
        if deferred_by_strategy:
            self._record_event(
                "merge_reconciliation_deferred",
                {
                    "candidate_count": len(candidates),
                    "processed_count": len(selected_candidates),
                    "deferred_count": len(deferred_by_strategy),
                    "deferred_task_ids": sorted(set(deferred_by_strategy)),
                    "max_merges": max_merges,
                    "reason": "strategy_deprioritized_task",
                },
            )
        for event in selected_candidates:
            task_id = str(event.get("task_id") or "")
            attempt = int(event.get("attempt") or 0)
            branch = str(event.get("branch") or "")
            worktree_path_text = str(event.get("worktree_path") or "")
            worktree_path = Path(worktree_path_text) if worktree_path_text else None
            implementation_commit = str(event.get("implementation_commit") or "")
            if not task_id or not implementation_commit:
                continue
            task = PortalTask(
                task_id=task_id,
                title=str(event.get("title") or "failed implementation merge"),
                status="todo",
                completion="manual",
                priority="P2",
                track="ops",
            )
            if self._git_ref_is_ancestor(implementation_commit, target_branch):
                # The parent commit can land before its daemon-owned submodule
                # branches finish merging.  Do not interpret parent ancestry as
                # proof that nested work is complete: resume the durable
                # submodule checkpoint first.
                submodule_merge_results = self._merge_submodule_branches_to_main(
                    branch,
                    task=task,
                    attempt=attempt,
                    baseline_ref=str(event.get("baseline_ref") or ""),
                ) if branch else []
                failed_submodules = [
                    item for item in submodule_merge_results if not item.get("merged", False)
                ]
                cleanup_result = (
                    self._cleanup_merged_worktree(worktree_path, branch)
                    if branch and not failed_submodules
                    else {}
                )
                cleanup_cleaned = bool(cleanup_result.get("cleaned", False)) if cleanup_result else True
                resolved = not failed_submodules and cleanup_cleaned
                todo_update_result = self._mark_task_completed_in_todo(task_id) if resolved else {}
                result = {
                    "task_id": task_id,
                    "attempt": attempt,
                    "branch": branch,
                    "implementation_commit": implementation_commit,
                    "resolved": resolved,
                    "reason": (
                        "submodule_merge_retry_failed"
                        if failed_submodules
                        else "implementation_commit_already_merged"
                        if cleanup_cleaned
                        else "cleanup_retry_failed"
                    ),
                    "submodule_merge_results": submodule_merge_results,
                    "cleanup_result": cleanup_result,
                }
                if todo_update_result:
                    result["todo_update_result"] = todo_update_result
                self._record_event("merge_reconciled", result)
                results.append(result)
                continue
            branch_exists = bool(branch and self._git_ref_exists(branch))
            merge_ref = branch if branch_exists else ""
            merge_ref_source = "branch" if branch_exists else ""
            if not merge_ref and self._git_ref_exists(implementation_commit):
                merge_ref = implementation_commit
                merge_ref_source = "implementation_commit"
                self._record_event(
                    "merge_reconcile_ref_recovered",
                    {
                        "task_id": task_id,
                        "attempt": attempt,
                        "branch": branch,
                        "implementation_commit": implementation_commit,
                        "merge_ref": merge_ref,
                        "merge_ref_source": merge_ref_source,
                        "reason": "implementation_branch_missing",
                    },
                )
            if not merge_ref:
                result = {
                    "task_id": task_id,
                    "attempt": attempt,
                    "branch": branch,
                    "implementation_commit": implementation_commit,
                    "merge_ref": "",
                    "merge_ref_source": "",
                    "resolved": False,
                    "reason": (
                        "implementation_branch_missing"
                        if branch or not implementation_commit
                        else "implementation_ref_missing"
                    ),
                }
                self._record_event("merge_reconcile_skipped", result)
                results.append(result)
                continue

            self._mark_long_running_phase(
                task_id=task_id,
                phase="merge_reconciliation",
                detail=merge_ref,
            )
            try:
                merge_result = self._merge_branch_to_main(
                    merge_ref,
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
                    "merge_ref": merge_ref,
                    "merge_ref_source": merge_ref_source,
                    "resolved": False,
                    "reason": "merge_reconcile_exception",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[-4000:],
                }
                self._record_event("merge_reconcile_exception", result)
                results.append(result)
                continue
            cleanup_result = {}
            cleanup_cleaned = True
            if merge_result.get("merged"):
                cleanup_result = self._cleanup_merged_worktree(worktree_path, branch)
                cleanup_cleaned = bool(cleanup_result.get("cleaned", False))
            resolved = bool(merge_result.get("merged")) and cleanup_cleaned
            reason = "merge_retried" if resolved else "merge_retry_failed"
            if merge_result.get("merged") and not cleanup_cleaned:
                reason = "cleanup_retry_failed"
            todo_update_result = self._mark_task_completed_in_todo(task_id) if resolved else {}
            result = {
                "task_id": task_id,
                "attempt": attempt,
                "branch": branch,
                "implementation_commit": implementation_commit,
                "merge_ref": merge_ref,
                "merge_ref_source": merge_ref_source,
                "resolved": resolved,
                "reason": reason,
                "merge_result": merge_result,
                "cleanup_result": cleanup_result,
            }
            if todo_update_result:
                result["todo_update_result"] = todo_update_result
            self._record_event("merge_reconciled", result)
            results.append(result)
        if max_merges > 0 and len(candidates) > len(selected_candidates):
            self._record_event(
                "merge_reconciliation_deferred",
                {
                    "candidate_count": len(candidates),
                    "processed_count": len(selected_candidates),
                    "deferred_count": len(candidates) - len(selected_candidates),
                    "max_merges": max_merges,
                },
            )
        return results

    def _partition_stale_failed_merge_candidates(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        max_age_seconds = int(self.merge_reconciliation_max_age_seconds)
        if max_age_seconds <= 0:
            return list(candidates), []
        fresh: list[dict[str, Any]] = []
        stale: list[dict[str, Any]] = []
        for event in candidates:
            if not str(event.get("timestamp") or ""):
                fresh.append(event)
            elif self._event_age_seconds(event) > max_age_seconds:
                stale.append(event)
            else:
                fresh.append(event)
        return fresh, stale

    def _stale_failed_merge_candidate_result(self, event: dict[str, Any]) -> dict[str, Any]:
        task_id = str(event.get("task_id") or "")
        attempt = int(event.get("attempt") or 0)
        branch = str(event.get("branch") or "")
        implementation_commit = str(event.get("implementation_commit") or "")
        return {
            "task_id": task_id,
            "attempt": attempt,
            "branch": branch,
            "implementation_commit": implementation_commit,
            "merge_ref": branch if branch else implementation_commit,
            "merge_ref_source": "branch" if branch else "implementation_commit",
            "resolved": False,
            "reason": "stale_failed_merge_candidate",
            "max_age_seconds": int(self.merge_reconciliation_max_age_seconds),
            "candidate_timestamp": str(event.get("timestamp") or ""),
            "merge_result": {
                "attempted": False,
                "merged": False,
                "reason": "stale_failed_merge_candidate",
            },
            "cleanup_result": {},
        }

    @classmethod
    def _select_failed_merge_candidates_for_reconciliation(
        cls,
        candidates: Sequence[dict[str, Any]],
        max_merges: int,
        *,
        blocked_task_ids: set[str] | None = None,
        deprioritized_task_ids: set[str] | None = None,
        now_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        blocked_task_ids = {str(task_id) for task_id in (blocked_task_ids or set()) if str(task_id)}
        deprioritized_task_ids = {
            str(task_id) for task_id in (deprioritized_task_ids or set()) if str(task_id)
        }
        filtered_candidates = [
            event
            for event in candidates
            if str(event.get("task_id") or "") not in blocked_task_ids
            and str(event.get("task_id") or "") not in deprioritized_task_ids
        ]
        transient = [
            event
            for event in filtered_candidates
            if cls._event_has_transient_merge_lock_deferral(event)
        ]
        if max_merges <= 0:
            recent_transient = [
                event
                for event in transient
                if cls._event_age_seconds(event, now_ts=now_ts)
                <= TRANSIENT_MERGE_RETRY_MAX_AGE_WHEN_DISABLED_SECONDS
            ]
            return recent_transient[:TRANSIENT_MERGE_RETRY_BUDGET_WHEN_DISABLED]

        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for event in (*transient, *filtered_candidates):
            identity = id(event)
            if identity in seen:
                continue
            selected.append(event)
            seen.add(identity)
            if len(selected) >= max_merges:
                break
        return selected

    def _failed_merge_candidates(self, *, skip_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
        skip_task_ids = skip_task_ids or set()
        current_task_ids = self._current_todo_task_ids_for_reconciliation()
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        reconciled_commits: set[str] = set()
        abandoned_commits: set[str] = set()
        target_branch = self._main_branch_name()
        for event in self._iter_events():
            if str(event.get("type") or "") == "merge_reconciled":
                implementation_commit = str(event.get("implementation_commit") or "")
                merge_result = event.get("merge_result") or {}
                merge_reason = merge_result.get("reason") if isinstance(merge_result, dict) else ""
                reconcile_reason = str(event.get("reason") or "")
                if implementation_commit and event.get("resolved"):
                    reconciled_commits.add(implementation_commit)
                elif implementation_commit and merge_reason == "baseline_not_ancestor_of_target":
                    abandoned_commits.add(implementation_commit)
                elif implementation_commit and reconcile_reason == "stale_failed_merge_candidate":
                    abandoned_commits.add(implementation_commit)
                continue
            if str(event.get("type") or "") != "implementation_finished":
                continue
            task_id = str(event.get("task_id") or "")
            if task_id in skip_task_ids:
                continue
            if current_task_ids is not None and task_id not in current_task_ids:
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
            cleanup = event.get("cleanup_result") or {}
            cleanup_failed = isinstance(cleanup, dict) and bool(cleanup) and not cleanup.get("cleaned", False)
            if not cleanup_failed and not self._merge_result_needs_reconciliation(merge_result):
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

    def _current_todo_task_ids_for_reconciliation(self) -> set[str] | None:
        try:
            tasks = parse_task_file(self.todo_path, self.task_header_prefix)
        except (OSError, UnicodeDecodeError):
            return None
        return {
            task.task_id
            for task in tasks
            if normalize_status(task.status) not in {"blocked", "completed"}
        }

    @staticmethod
    def _merge_result_needs_reconciliation(merge_result: dict[str, Any]) -> bool:
        if not isinstance(merge_result, dict) or merge_result.get("merged"):
            return False
        if merge_result.get("attempted"):
            return True
        return str(merge_result.get("reason") or "") in TRANSIENT_MERGE_LOCK_REASONS

    @staticmethod
    def _merge_result_is_transient_lock_deferral(merge_result: dict[str, Any]) -> bool:
        if not isinstance(merge_result, dict) or merge_result.get("merged"):
            return False
        if merge_result.get("attempted"):
            return False
        return str(merge_result.get("reason") or "") in TRANSIENT_MERGE_LOCK_REASONS

    @classmethod
    def _event_has_transient_merge_lock_deferral(cls, event: dict[str, Any]) -> bool:
        merge_result = event.get("merge_result") or {}
        return cls._merge_result_is_transient_lock_deferral(merge_result)

    @staticmethod
    def _event_age_seconds(event: dict[str, Any], *, now_ts: float | None = None) -> float:
        event_timestamp = parse_timestamp(str(event.get("timestamp") or ""))
        if event_timestamp is None:
            return float("inf")
        return max(0.0, (now_ts or time.time()) - event_timestamp.timestamp())

    def _transient_merge_deferrals_by_task(self, *, skip_task_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
        skip_task_ids = skip_task_ids or set()
        failures: dict[str, dict[str, Any]] = {}
        target_branch = self._main_branch_name()
        for event in self._failed_merge_candidates(skip_task_ids=skip_task_ids):
            if not self._event_has_transient_merge_lock_deferral(event):
                continue
            task_id = str(event.get("task_id") or "")
            if task_id in skip_task_ids:
                continue
            implementation_commit = str(event.get("implementation_commit") or "")
            if task_id and implementation_commit and not self._git_ref_is_ancestor(implementation_commit, target_branch):
                failures[task_id] = event
        return failures

    def _unresolved_merge_failures_by_task(self, *, skip_task_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
        skip_task_ids = skip_task_ids or set()
        failures: dict[str, dict[str, Any]] = {}
        target_branch = self._main_branch_name()
        for event in self._failed_merge_candidates(skip_task_ids=skip_task_ids):
            if self._event_has_transient_merge_lock_deferral(event):
                continue
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

    def _implementation_task_claim_path(self, task_id: str, *, canonical_task_cid: str = "") -> Path:
        lock_identity = canonical_task_cid or task_id
        safe_task_id = (
            "canonical-task"
            if canonical_task_cid
            else re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._-") or "task"
        )
        digest = hashlib.sha1(lock_identity.encode("utf-8")).hexdigest()[:20]
        lock_filename = f"{safe_task_id[:96]}-{digest}.lock"
        return (
            checkout_mutation_lock_path(self.repo_root, lock_name=IMPLEMENTATION_TASK_CLAIM_LOCK_DIRNAME)
            / lock_filename
        )

    def _build_implementation_lock_metadata(self, task: PortalTask, attempt: int, started_at: str) -> dict[str, Any]:
        identity = self._identity_for_task(task)
        return {
            "kind": "implementation",
            "pid": os.getpid(),
            "owner_script": Path(sys.argv[0]).name,
            "repo_root": str(self.repo_root.resolve()),
            "state_dir": str(self.state_path.parent.resolve()),
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": identity.canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": attempt,
            "started_at": started_at,
        }

    def _build_implementation_task_claim_metadata(
        self,
        task: PortalTask,
        attempt: int,
        started_at: str,
    ) -> dict[str, Any]:
        identity = self._identity_for_task(task)
        return checkout_lock_metadata(
            kind=IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
            repo_root=self.repo_root,
            task_id=task.task_id,
            attempt=attempt,
            owner_script=Path(sys.argv[0]).name,
            extra={
                "state_dir": str(self.state_path.parent.resolve()),
                "state_path": str(self.state_path.resolve()),
                "started_at": started_at,
                "canonical_task_key": identity.canonical_task_key,
                "canonical_task_cid": identity.canonical_task_cid,
                "board_namespace": identity.board_namespace,
                "task_shard_count": self.task_shard_count,
                "task_shard_index": self.task_shard_index,
            },
        )

    def _build_merge_lock_metadata(
        self,
        branch_name: str,
        task: PortalTask,
        attempt: int,
        started_at: str,
    ) -> dict[str, Any]:
        identity = self._identity_for_task(task)
        return {
            "kind": "merge",
            "pid": os.getpid(),
            "owner_script": Path(sys.argv[0]).name,
            "repo_root": str(self.repo_root.resolve()),
            "state_dir": str(self.state_path.parent.resolve()),
            "state_path": str(self.state_path.resolve()),
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": identity.canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": attempt,
            "branch": branch_name,
            "started_at": started_at,
        }

    def _implementation_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        state_dir = str(metadata.get("state_dir") or "")
        if state_dir and Path(state_dir).resolve() != self.state_path.parent.resolve():
            return False
        return self._lock_owner_is_active(metadata, expected_kind="implementation")

    def _implementation_task_claim_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        repo_root = str(metadata.get("repo_root") or "")
        if repo_root:
            try:
                if Path(repo_root).resolve() != self.repo_root.resolve():
                    return False
            except OSError:
                return False
        return self._lock_owner_is_active(metadata, expected_kind=IMPLEMENTATION_TASK_CLAIM_LOCK_KIND)

    def _external_task_reservations(
        self,
        tasks: Sequence[PortalTask | str],
    ) -> dict[str, dict[str, Any]]:
        task_ids = {
            item.task_id if isinstance(item, PortalTask) else str(item)
            for item in tasks
        }
        reservations: dict[str, dict[str, Any]] = {}
        if not task_ids:
            return reservations
        from ..artifact_store import read_artifact_fields

        for manifest_path in self.external_reservation_manifest_paths:
            try:
                payload = read_artifact_fields(manifest_path, ("lanes",))
            except (OSError, RuntimeError, TypeError, ValueError):
                payload = {}
            if not isinstance(payload.get("lanes"), list):
                payload = load_json_dict(manifest_path) or {}
            for lane in payload.get("lanes", []) or []:
                if not isinstance(lane, dict):
                    continue
                lane_state = str(lane.get("state") or "running").strip().lower()
                if lane_state not in {"accepted", "active", "running"}:
                    continue
                queue_payload = (
                    lane.get("queue_payload")
                    if isinstance(lane.get("queue_payload"), dict)
                    else {}
                )
                reserved_ids = queue_payload.get("execution_slice_task_ids")
                if not isinstance(reserved_ids, list):
                    reserved_ids = lane.get("task_ids")
                if not isinstance(reserved_ids, list):
                    continue
                for raw_task_id in reserved_ids:
                    task_id = str(raw_task_id)
                    if task_id not in task_ids:
                        continue
                    reservations[task_id] = {
                        "kind": "external_bundle_reservation",
                        "manifest_path": str(manifest_path),
                        "bundle_key": str(lane.get("bundle_key") or ""),
                        "pid": int(lane.get("pid") or 0),
                    }
        return reservations

    def _active_implementation_task_claims(
        self,
        tasks: Sequence[PortalTask | str],
    ) -> dict[str, dict[str, Any]]:
        active_claims: dict[str, dict[str, Any]] = {}
        active_claims_by_cid: dict[str, dict[str, Any]] = {}
        for item in tasks:
            if isinstance(item, PortalTask):
                task_id = item.task_id
                canonical_task_cid = self._canonical_ref(item)
                paths = [
                    self._implementation_task_claim_path(
                        task_id,
                        canonical_task_cid=canonical_task_cid,
                    ),
                    self._implementation_task_claim_path(task_id),
                ]
            else:
                task_id = str(item)
                canonical_task_cid = ""
                paths = [self._implementation_task_claim_path(task_id)]
            for claim_path in dict.fromkeys(paths):
                if not claim_path.exists():
                    continue
                metadata = load_json_dict(claim_path)
                if metadata is not None and self._implementation_task_claim_owner_is_active(metadata):
                    active_claims[task_id] = metadata
                    if canonical_task_cid:
                        active_claims_by_cid[canonical_task_cid] = metadata
                    break
        for item in tasks:
            if not isinstance(item, PortalTask):
                continue
            metadata = active_claims_by_cid.get(self._canonical_ref(item))
            if metadata is not None:
                active_claims.setdefault(item.task_id, metadata)
        return active_claims

    def _merge_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        repo_root = str(metadata.get("repo_root") or "")
        if repo_root and Path(repo_root).resolve() != self.repo_root.resolve():
            return False
        if not self._lock_owner_is_active(metadata, expected_kind="merge"):
            return False
        if self._lock_targets_current_daemon_state(metadata):
            return self._lock_task_is_active(metadata)
        return True

    def _lock_targets_current_daemon_state(self, metadata: dict[str, Any]) -> bool:
        state_path = str(metadata.get("state_path") or "")
        if state_path:
            try:
                return Path(state_path).resolve() == self.state_path.resolve()
            except OSError:
                return False
        state_dir = str(metadata.get("state_dir") or "")
        if state_dir:
            try:
                return Path(state_dir).resolve() == self.state_path.parent.resolve()
            except OSError:
                return False
        owner_script = str(metadata.get("owner_script") or "")
        return bool(owner_script and owner_script == Path(sys.argv[0]).name)

    def _lock_task_is_active(self, metadata: dict[str, Any]) -> bool:
        task_id = str(metadata.get("task_id") or "")
        canonical_task_cid = str(metadata.get("canonical_task_cid") or "")
        if not task_id and not canonical_task_cid:
            return True
        try:
            state = PortalTaskState.load(self.state_path)
        except Exception:
            return True
        if canonical_task_cid and state.active_task_cid:
            if state.active_task_cid != canonical_task_cid:
                return False
        elif state.active_task_id != task_id:
            return False
        branch = str(metadata.get("branch") or "")
        return not branch or not state.active_branch or state.active_branch == branch

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
            # ``python -m package.implementation_daemon`` does not retain the
            # ``.py`` filename in argv. Accept the module stem as well so a
            # live daemon never has its task claim stolen by another shard.
            owner_module_stem = Path(owner_script).stem
            if not owner_module_stem or owner_module_stem not in command_line:
                return False
        return True

    def _try_acquire_lock(
        self,
        lock_path: Path,
        *,
        lock_kind: str,
        owner_active: Any,
    ) -> tuple[int | None, str, dict[str, Any] | None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
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
            elif event_type in {"implementation_finished", "implementation_provider_exhausted"}:
                inflight.pop(key, None)

        return list(inflight.values())

    def _inflight_submodule_paths(self) -> set[str]:
        """Return submodule paths currently being modified by in-flight implementations.

        Used to prevent two lanes from simultaneously modifying the same
        submodule pointer, which would cause silent data loss on merge.
        """
        inflight_paths: set[str] = set()
        for event in self._inflight_implementation_events():
            # Check task outputs for submodule paths
            outputs = event.get("outputs") or []
            for output in outputs:
                for sm_path in self.worktree_submodule_paths:
                    if output.startswith(sm_path + "/") or output == sm_path:
                        inflight_paths.add(sm_path)
            # Also check recently merged submodules (within last 5 minutes)
            # to avoid race conditions during merge
        recent_merge_events = [
            e for e in self._iter_events()
            if str(e.get("type") or "") == "merge_finished"
            and e.get("submodule_merge_results")
        ]
        now = time.time()
        for event in recent_merge_events[-5:]:  # Check last 5 merges
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(str(event.get("started_at", "")))
                age = now - ts.timestamp()
                if age < 300:  # Within 5 minutes
                    for sm_result in event.get("submodule_merge_results", []):
                        if sm_result.get("merged") and sm_result.get("path"):
                            inflight_paths.add(str(sm_result["path"]))
            except (ValueError, TypeError):
                pass
        return inflight_paths

    def _task_conflicts_with_inflight_submodules(
        self,
        task: PortalTask,
        inflight_submodules: set[str],
    ) -> str | None:
        """Check if a task's outputs overlap with in-flight submodule work.

        Returns the conflicting submodule path, or None if no conflict.
        """
        if not inflight_submodules:
            return None
        for output in task.outputs:
            for sm_path in inflight_submodules:
                if output.startswith(sm_path + "/") or output == sm_path:
                    return sm_path
        return None

    def _latest_implementation_finished_by_task(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._iter_events():
            if str(event.get("type") or "") != "implementation_finished":
                continue
            task_id = str(event.get("task_id") or "")
            if task_id:
                latest[task_id] = event
        return latest

    def _pending_queued_merge_task_ids(
        self,
        latest_results: dict[str, dict[str, Any]] | None = None,
    ) -> set[str]:
        """Return tasks handed to the train whose commit is not integrated yet."""

        target_branch = self._main_branch_name()
        pending: set[str] = set()
        for task_id, event in (latest_results or self._latest_implementation_finished_by_task()).items():
            merge_result = event.get("merge_result") or {}
            if not isinstance(merge_result, dict) or not merge_result.get("queued"):
                continue
            request_id = str(merge_result.get("request_id") or "")
            request = self.merge_queue.get(request_id) if request_id and hasattr(self.merge_queue, "get") else None
            if request is not None and str(getattr(request, "status", "")) == "quarantined":
                continue
            implementation_commit = str(event.get("implementation_commit") or "")
            if implementation_commit and not self._git_ref_is_ancestor(implementation_commit, target_branch):
                pending.add(task_id)
        return pending

    def _quarantined_queued_merge_task_ids(
        self,
        latest_results: dict[str, dict[str, Any]] | None = None,
    ) -> set[str]:
        """Return queued tasks whose bounded train attempts are exhausted."""

        quarantined: set[str] = set()
        if not hasattr(self.merge_queue, "get"):
            return quarantined
        for task_id, event in (latest_results or self._latest_implementation_finished_by_task()).items():
            merge_result = event.get("merge_result") or {}
            if not isinstance(merge_result, dict) or not merge_result.get("queued"):
                continue
            request_id = str(merge_result.get("request_id") or "")
            request = self.merge_queue.get(request_id) if request_id else None
            if request is not None and str(getattr(request, "status", "")) == "quarantined":
                quarantined.add(task_id)
        return quarantined

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
                if not isinstance(merge_result, dict):
                    continue
                # A lane can finish while its request remains queued.  Commit
                # ancestry is the durable proof a later train consumer landed
                # it, even when that consumer wrote to another lane's event log.
                if not merge_result.get("merged") and not merge_result.get("queued"):
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
        """Read all events, with per-iteration file-stat caching.

        Avoids re-reading the entire file multiple times within the same
        run_once() call if the file hasn't changed (same size + mtime).
        """
        try:
            stat = self.events_path.stat()
            cache_key = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            cache_key = None

        if cache_key is not None and hasattr(self, "_events_cache_key") and self._events_cache_key == cache_key:
            return self._events_cache_data

        data = read_jsonl_events(self.events_path, repair=True)
        self._events_cache_key = cache_key
        self._events_cache_data = data
        return data

    def _invalidate_event_cache(self) -> None:
        """Invalidate the event read cache (call after appending events)."""
        self._events_cache_key = None
        self._events_cache_data = []

    def _implementation_process_active(self, event: dict[str, Any]) -> bool:
        worktree_path = str(event.get("worktree_path") or "")
        command = event.get("command") or []
        process_lines = self._list_process_commands()
        if worktree_path:
            # Task validation can leave MCP bridge servers in its worktree. Only
            # the configured Codex/Copilot runner proves implementation is live.
            return any(
                worktree_path in line and IMPLEMENTATION_RUNNER_PROCESS_PATTERN.search(line)
                for line in process_lines
            )
        # Shared-checkout implementations deliberately do not have a task
        # worktree path.  Their serialized wrapper command contains a
        # heredoc, so matching the complete command line is not reliable.
        # The repository path plus the configured runner is the stable
        # identity, and still excludes MCP bridge/service processes.
        repo_path = str(self.repo_root.resolve())
        if repo_path:
            return any(
                repo_path in line and IMPLEMENTATION_RUNNER_PROCESS_PATTERN.search(line)
                for line in process_lines
            )
        if isinstance(command, list):
            command_text = " ".join(str(item) for item in command if item)
            if command_text:
                return any(
                    command_text in line and IMPLEMENTATION_RUNNER_PROCESS_PATTERN.search(line)
                    for line in process_lines
                )
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
        return checkout_mutation_lock_path(self.repo_root)

    @staticmethod
    def _git_merge_head_in_repo(repo: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _run_git(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def _build_implementation_command(self, workspace_path: Path) -> list[str]:
        workspace_path = workspace_path.resolve()
        if self.implementation_command:
            return shlex.split(self.implementation_command)
        env_command = os.environ.get("IMPLEMENTATION_DAEMON_COMMAND", "").strip()
        if env_command:
            return shlex.split(env_command)
        codex = shutil.which("codex")
        copilot = shutil.which("copilot")
        if copilot and _copilot_has_auth():
            return _copilot_fallback_command(codex=codex, copilot=copilot, workspace_path=workspace_path)
        if codex:
            # Build codex command with full capability flags
            codex_model = os.environ.get(_CODEX_MODEL_ENV, "").strip()
            codex_context = os.environ.get(_CODEX_CONTEXT_WINDOW_ENV, "200000").strip()
            codex_reasoning = os.environ.get(_CODEX_REASONING_EFFORT_ENV, "high").strip()
            codex_max_threads = os.environ.get(_CODEX_MAX_THREADS_ENV, "10").strip()
            codex_max_depth = os.environ.get(_CODEX_MAX_DEPTH_ENV, "2").strip()

            cmd = [
                codex,
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C",
                str(workspace_path),
            ]
            if codex_model:
                cmd.extend(["-m", codex_model])
            if codex_context:
                cmd.extend(["-c", f"model_context_window={codex_context}"])
            if codex_reasoning:
                cmd.extend(["-c", f'model_reasoning_effort="{codex_reasoning}"'])
            if codex_max_threads:
                cmd.extend(["-c", f"agents.max_threads={codex_max_threads}"])
            if codex_max_depth:
                cmd.extend(["-c", f"agents.max_depth={codex_max_depth}"])
            cmd.append("-")
            return cmd
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
                "covered_packet_records": {
                    task_id: by_task[task_id]
                    for task_id in covered_packet_task_ids
                    if task_id in by_task
                },
            }
        return None

    def _bundle_work_order_for_task(self, task: PortalTask) -> BundleWorkOrder | None:
        """Return bundle completion metadata for a packet aggregate task."""

        context = self._load_todo_vector_context(task)
        if context is None or not context.get("aggregate_primary"):
            return None
        candidate_task_ids = [
            str(task_id).strip()
            for task_id in context.get("covered_packet_task_ids", [])
            if str(task_id).strip() and str(task_id).strip() != task.task_id
        ]
        if not candidate_task_ids:
            return None
        record = context.get("record")
        if not isinstance(record, dict):
            return None
        covered_records = context.get("covered_packet_records")
        if not isinstance(covered_records, dict):
            covered_records = {}
        try:
            shard_tasks = {
                shard_task.task_id: shard_task
                for shard_task in parse_task_file(self.todo_path, self.task_header_prefix)
            }
        except OSError:
            return None
        primary_bundle_key = (
            self._task_metadata_value(task, "bundle")
            or str(record.get("bundle_key") or "").strip()
        )
        covered_task_ids: list[str] = []
        for task_id in candidate_task_ids:
            shard_task = shard_tasks.get(task_id)
            candidate_record = covered_records.get(task_id)
            if shard_task is None or not isinstance(candidate_record, dict):
                continue
            candidate_bundle_key = (
                self._task_metadata_value(shard_task, "bundle")
                or str(candidate_record.get("bundle_key") or "").strip()
            )
            if primary_bundle_key and candidate_bundle_key != primary_bundle_key:
                continue
            covered_task_ids.append(task_id)
        if not covered_task_ids:
            return None
        packet_key = str(record.get("goal_packet_key") or record.get("merge_family") or "").strip()
        goal_ids = self._compact_value_list(record.get("goal_packet_goal_ids"), limit=24)
        work_item_count = self._todo_vector_record_int(record, "goal_packet_work_item_count")
        if work_item_count <= 0:
            work_item_count = self._todo_vector_record_int(record, "work_item_count")
        index_path = context.get("index_path")
        display_index_path = self._display_context_path(index_path) if isinstance(index_path, Path) else ""
        return BundleWorkOrder(
            primary_task_id=task.task_id,
            covered_task_ids=list(dict.fromkeys(covered_task_ids)),
            packet_key=packet_key,
            goal_ids=goal_ids,
            work_item_count=work_item_count,
            index_path=display_index_path,
        )

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
            required_lines.append(
                f"- Bundle work order: primary={task.task_id}; covers={', '.join(covered_packet_task_ids)}; completion_propagates=true"
            )
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
        # Build sub-agent guidance when multiple outputs suggest parallelizable work
        subagent_guidance = ""
        if len(task.outputs) > 3:
            subagent_guidance = """
- This task has many expected outputs. Use sub-agents or parallel execution when possible:
  decompose into independent file/module implementations and work on them concurrently.
"""
        return f"""You are an autonomous implementation agent working in this repository.

Implement this backlog task completely and thoroughly. Produce a full, production-ready implementation
that covers all expected outputs. Do not artificially limit scope or break the work into smaller pieces —
deliver the entire task in one pass, touching as many files as needed.

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
- Prefer existing repo patterns. Implement comprehensively — create all necessary files, classes, functions, tests, and integrations that the task requires.
- Implement ALL expected outputs for this task in full. Do not leave stubs, placeholders, or TODOs.
- If a compact execution packet or goal packet is shown, implement a single cohesive change that advances all the shared packet evidence together without making unrelated edits.
- You may create new files and modify multiple existing files. Larger, complete implementations are preferred over minimal patches.
{subagent_guidance}- Run the listed validation commands when practical.
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

    @staticmethod
    def _task_metadata_int(task: PortalTask, key: str) -> int:
        try:
            return int(str(task.metadata.get(key, "0")).strip() or "0")
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _task_candidate_rank(task: PortalTask) -> int:
        candidate_kind = str(task.metadata.get("candidate kind", "")).strip().lower()
        goal_packet_role = str(task.metadata.get("goal packet role", "")).strip().lower()
        merge_role = str(task.metadata.get("merge role", "")).strip().lower()
        if candidate_kind == "goal_packet_aggregate" or goal_packet_role == "packet_aggregate" or merge_role == "packet_aggregate":
            return 0
        if goal_packet_role == "packet_anchor":
            return 1
        if candidate_kind == "aggregate":
            return 2
        if candidate_kind == "evidence_cluster":
            return 3
        if goal_packet_role == "packet_member":
            return 4
        return 5

    def _task_work_surface_rank(self, task: PortalTask) -> tuple[int, int, int]:
        packet_work_items = self._task_metadata_int(task, "goal packet work item count")
        work_items = self._task_metadata_int(task, "work item count")
        return (self._task_candidate_rank(task), -packet_work_items, -work_items)

    @staticmethod
    def _strict_off_mission_deprioritized_task_ids(strategy: dict[str, Any]) -> set[str]:
        return {
            str(receipt.get("task_id") or "")
            for receipt in strategy.get("objective_task_janitor_receipts", [])
            if isinstance(receipt, dict)
            and str(receipt.get("action") or "") == "deprioritize"
            and str(receipt.get("retired_task_reason") or "").startswith("off_mission_")
        }

    def _selection_scope(
        self,
        tasks: list[PortalTask],
        resolved_statuses: dict[str, str],
        strategy: dict[str, Any],
    ) -> dict[str, Any]:
        selectable_ready = [task for task in tasks if resolved_statuses.get(task.task_id) == "ready"]
        strict_deprioritized = self._strict_off_mission_deprioritized_task_ids(strategy)
        strict_ready = [task.task_id for task in selectable_ready if task.task_id in strict_deprioritized]
        eligible_ready = [task.task_id for task in selectable_ready if task.task_id not in strict_deprioritized]
        reason = ""
        if not eligible_ready:
            if strict_ready:
                reason = "all_selectable_ready_tasks_deprioritized_as_off_mission"
            elif selectable_ready:
                reason = "no_eligible_ready_tasks_after_selection_filters"
            else:
                reason = "no_shard_selectable_ready_tasks"
        return {
            "selectable_ready_task_ids": [task.task_id for task in selectable_ready],
            "eligible_ready_task_ids": eligible_ready,
            "strict_deprioritized_ready_task_ids": strict_ready,
            "selection_idle_reason": reason,
        }

    def _select_next_task(
        self,
        tasks: list[PortalTask],
        resolved_statuses: dict[str, str],
        strategy: dict[str, Any],
        unresolved_merge_failures: dict[str, dict[str, Any]],
        recent_outcomes: dict[str, dict[str, Any]],
    ) -> PortalTask | None:
        ready = [task for task in tasks if resolved_statuses.get(task.task_id) == "ready"]
        # The durable queue is authoritative across isolated lane state dirs.
        # Consult both canonical and display identities for compatibility with
        # queue records written before canonical task ids were introduced.
        ready = [
            task
            for task in ready
            if not self.merge_queue.has_pending_for_task(self._canonical_ref(task))
            and not self.merge_queue.has_pending_for_task(task.task_id)
        ]
        strict_deprioritized = self._strict_off_mission_deprioritized_task_ids(strategy)
        if strict_deprioritized:
            ready = [task for task in ready if task.task_id not in strict_deprioritized]
        # Graceful degradation: skip tasks that depend on degraded submodules
        degraded_skipped: list[str] = []
        if self.degradation_state.degraded_submodules():
            filtered_ready = []
            for task in ready:
                degraded_sub = self.degradation_state.should_skip_task(
                    task.outputs, getattr(task, "inputs", None)
                )
                if degraded_sub:
                    degraded_skipped.append(task.task_id)
                else:
                    filtered_ready.append(task)
            if degraded_skipped:
                self._record_event("tasks_skipped_degraded_submodule", {
                    "skipped_task_ids": degraded_skipped[:20],
                    "degraded_submodules": self.degradation_state.degraded_submodules(),
                })
            ready = filtered_ready
        if not ready:
            return None
        # Concurrent submodule protection: skip tasks that modify submodules
        # already being worked on by in-flight implementations
        inflight_submodules = self._inflight_submodule_paths()
        if inflight_submodules:
            conflict_skipped: list[str] = []
            safe_ready = []
            for task in ready:
                conflicting = self._task_conflicts_with_inflight_submodules(task, inflight_submodules)
                if conflicting:
                    conflict_skipped.append(task.task_id)
                else:
                    safe_ready.append(task)
            if conflict_skipped and safe_ready:
                self._record_event("tasks_skipped_submodule_conflict", {
                    "skipped_task_ids": conflict_skipped[:20],
                    "inflight_submodules": sorted(inflight_submodules),
                })
                ready = safe_ready
            # If ALL tasks conflict, proceed anyway (don't deadlock)
        if not ready:
            return None
        # Filter out tasks in cooldown from persistent queue
        cooled_ready = [t for t in ready if not self.task_queue.is_cooled_down(self._canonical_ref(t))]
        if not cooled_ready:
            # All ready tasks are in cooldown - use the one with shortest remaining cooldown
            cooled_ready = ready
        ready = cooled_ready
        ready_task_ids = {task.task_id for task in ready}
        vector_context = self._todo_vector_selection_context(tasks, ready_task_ids)
        focus_order = {
            track: index
            for index, track in enumerate(normalize_focus_tracks(strategy.get("focus_tracks", DEFAULT_TRACKS)))
        }
        deprioritized = {str(item) for item in strategy.get("deprioritized_tasks", [])}
        blocked_strategy_task_ids = {str(item) for item in strategy.get("blocked_tasks", [])}

        def sort_key(task: PortalTask) -> tuple[Any, ...]:
            selection_penalty = self.task_queue.get_penalty(self._canonical_ref(task))
            if task.task_id in unresolved_merge_failures:
                selection_penalty += UNRESOLVED_MERGE_SELECTION_PENALTY
            if self._task_has_recent_no_change_outcome(task.task_id, recent_outcomes):
                selection_penalty += NO_CHANGE_SELECTION_PENALTY
            retry_repair_source_id, _failure_kind = retry_budget_repair_source(task)
            vector_rank = self._todo_vector_selection_rank(task, vector_context)
            work_surface_rank = self._task_work_surface_rank(task)
            return (
                selection_penalty,
                PRIORITY_ORDER.get(task.priority, 99),
                0 if retry_repair_source_id in blocked_strategy_task_ids else 1,
                1 if task.task_id in deprioritized else 0,
                focus_order.get(task.track, len(focus_order)),
                *vector_rank,
                *work_surface_rank,
                len(task.depends_on),
                task.task_id,
            )

        selected = sorted(ready, key=sort_key)[0]
        # Record selection in persistent queue
        self.task_queue.record_selection(self._canonical_ref(selected))
        return selected

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        enriched = dict(payload)
        task_id = str(enriched.get("task_id") or "")
        identity = self._task_identity_by_display_id.get(task_id)
        if identity is not None:
            enriched.setdefault("canonical_task_key", identity.canonical_task_key)
            enriched.setdefault("canonical_task_cid", identity.canonical_task_cid)
            enriched.setdefault("board_namespace", identity.board_namespace)
        append_jsonl_event(self.events_path, event_type, enriched)
        self._invalidate_event_cache()


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
        default=default_llm_merge_resolver_command(),
        help=(
            "Command invoked with merge-conflict repair prompts on stdin. "
            "Defaults to the packaged Codex/Copilot resolver, unless "
            f"{LLM_MERGE_RESOLVER_COMMAND_ENV} is set."
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
    parser.add_argument(
        "--merge-reconciliation-max-merges",
        type=int,
        default=None,
        help=(
            "Maximum failed-merge reconciliation candidates to process per daemon pass. "
            f"Defaults to {DAEMON_MERGE_RECONCILIATION_MAX_ENV} or "
            f"{DEFAULT_DAEMON_MERGE_RECONCILIATION_MAX}; <=0 disables the cap."
        ),
    )
    parser.add_argument(
        "--merged-worktree-cleanup-max",
        type=int,
        default=None,
        help=(
            "Maximum already-merged implementation worktrees to remove per daemon pass. "
            f"Defaults to {DAEMON_MERGED_WORKTREE_CLEANUP_MAX_ENV} or "
            f"{DEFAULT_DAEMON_MERGED_WORKTREE_CLEANUP_MAX}; <=0 disables daemon-side cleanup."
        ),
    )
    parser.add_argument(
        "--task-shard-count",
        type=int,
        default=1,
        help="Number of deterministic task-ID shards for parallel daemon lanes. Defaults to 1.",
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index implemented by this daemon lane. Defaults to 0.",
    )
    parser.add_argument(
        "--daemon-hook-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Maximum seconds for each configured before/after daemon hook. "
            f"Defaults to {DAEMON_HOOK_TIMEOUT_ENV} or {DEFAULT_DAEMON_HOOK_TIMEOUT_SECONDS}; "
            "<=0 disables hook timeouts."
        ),
    )
    parser.add_argument(
        "--maintenance-interval-seconds",
        type=float,
        default=None,
        help=(
            "Minimum seconds between repository-wide maintenance scans. "
            f"Defaults to {DAEMON_MAINTENANCE_INTERVAL_ENV} or "
            f"{DEFAULT_DAEMON_MAINTENANCE_INTERVAL_SECONDS}; <=0 runs every pass."
        ),
    )
    parser.add_argument(
        "--objective-scan-min-open-tasks",
        type=int,
        default=None,
        help="Override daemon objective-refill minimum open backlog threshold.",
    )
    parser.add_argument(
        "--objective-scan-max-findings",
        type=int,
        default=None,
        help="Override daemon objective-refill maximum generated findings; <=0 disables findings.",
    )
    parser.add_argument(
        "--objective-scan-cooldown-seconds",
        type=int,
        default=None,
        help="Override daemon objective-refill cooldown seconds.",
    )
    parser.add_argument(
        "--objective-surplus-findings-per-goal",
        type=int,
        default=None,
        help="Override daemon objective-refill surplus findings per goal.",
    )
    parser.add_argument(
        "--objective-surplus-min-terms-per-todo",
        type=int,
        default=None,
        help="Override daemon objective-refill minimum evidence terms per generated todo.",
    )
    parser.add_argument(
        "--codebase-scan-min-open-tasks",
        type=int,
        default=None,
        help="Override daemon codebase-scan minimum open backlog threshold.",
    )
    parser.add_argument(
        "--codebase-scan-max-findings",
        type=int,
        default=None,
        help="Override daemon codebase-scan maximum generated findings; <=0 disables findings.",
    )
    parser.add_argument(
        "--codebase-scan-cooldown-seconds",
        type=int,
        default=None,
        help="Override daemon codebase-scan cooldown seconds.",
    )
    parser.add_argument("--implementation-timeout", type=float, default=DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS)
    parser.add_argument(
        "--validation-max-workers",
        type=int,
        default=None,
        help=(
            "Maximum validation subprocesses per stage. Defaults to "
            f"${VALIDATION_MAX_WORKERS_ENV} or {DEFAULT_VALIDATION_MAX_WORKERS}."
        ),
    )
    parser.add_argument(
        "--validation-resource-budget",
        type=int,
        default=None,
        help=(
            "Weighted validation resource budget. Defaults to "
            f"${VALIDATION_RESOURCE_BUDGET_ENV} or validation-max-workers."
        ),
    )
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
        "--merge-target-branch",
        default="",
        help=(
            "Branch that receives isolated implementation merges. Defaults to main/master, then the "
            "current branch. A configured branch must exist."
        ),
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
        "--objective-path",
        type=Path,
        default=None,
        help="Configured objective/goal markdown path that may be deterministically repaired during merges.",
    )
    parser.add_argument(
        "--objective-bundle-dir",
        type=Path,
        default=None,
        help="Directory of generated objective bundle markdown files that may be deterministically repaired.",
    )
    parser.add_argument(
        "--generated-status-path",
        type=Path,
        action="append",
        default=[],
        help="Repeatable generated file path that may be reconciled without blocking the checkout.",
    )
    parser.add_argument(
        "--external-reservation-manifest-path",
        type=Path,
        action="append",
        default=[],
        help="Repeatable bundle scheduler manifest whose running execution slices reserve tasks.",
    )
    parser.add_argument(
        "--assume-completed-task-id",
        action="append",
        default=[],
        help="Repeatable external dependency task ID already proven complete by the planner.",
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
        merge_target_branch=args.merge_target_branch,
        worktree_submodule_paths=args.worktree_submodule_path or None,
        objective_path=args.objective_path,
        objective_bundle_dir=args.objective_bundle_dir,
        generated_status_paths=args.generated_status_path,
        external_reservation_manifest_paths=args.external_reservation_manifest_path,
        assumed_completed_task_ids=args.assume_completed_task_id,
        llm_merge_resolver_command=args.llm_merge_resolver_command or None,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
        merge_reconciliation_max_merges=args.merge_reconciliation_max_merges,
        merged_worktree_cleanup_max=args.merged_worktree_cleanup_max,
        task_shard_count=args.task_shard_count,
        task_shard_index=args.task_shard_index,
        validation_max_workers=args.validation_max_workers,
        validation_resource_budget=args.validation_resource_budget,
        maintenance_interval_seconds=args.maintenance_interval_seconds,
    )
    while True:
        result = daemon.run_once()
        logger.info("Portal implementation daemon pass complete: %s", result)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
