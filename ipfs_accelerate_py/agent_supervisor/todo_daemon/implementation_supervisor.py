from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import logging
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...llm_router import (
    AgentImplementationControlPlanePin,
    AgentImplementationSealedControlPlane,
    verify_agent_implementation_sealed_control_plane,
)
from ..control.manual_completion_seal import (
    ManualCompletionSealError,
    verify_manual_completion_seal,
)
from ..entrypoints.execution_plan import (
    MAX_PLAN_BOUND_WAVE_TRANSFERS,
    PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA,
    PLAN_BOUND_MERGE_ENQUEUE_INTENT_SCHEMA,
    PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA,
    PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA,
    PLAN_BOUND_MERGE_TERMINAL_FAILURE_SCHEMA,
    PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA,
    ConfiguredBoardExecutionSlices,
    PlanBoundExecutionLease,
    PlanBoundProcessBirth,
    PlanBoundProposalDisposition,
    ProductionParallelPlanAdapter,
    _load_plan_bound_execution_lease_locked,
    _load_plan_bound_merge_terminal_failure_locked,
    _load_plan_bound_process_birth_chain_locked,
    _load_plan_bound_proposal_disposition_locked,
    _load_plan_revision_store_binding_locked,
    _publish_plan_bound_execution_lease_locked,
    _publish_plan_bound_merge_terminal_failure_locked,
    _publish_plan_bound_proposal_disposition_locked,
    _secure_store_active,
    _secure_store_cas,
    _secure_store_continuation,
)
from ..merge.checkout_lock import (
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    GENERATED_PROTECTED_BOARD_COMMIT_MARKER,
    PROTECTED_PATH_MAINTENANCE_LOCK_NAME,
    CheckoutMutationLease,
    adopt_inactive_checkout_mutation_lease,
    checkout_lock_metadata,
    checkout_lock_owner_is_active,
    checkout_mutation_lock_path,
    generated_protected_board_commit_subject,
    read_checkout_mutation_lease,
    release_checkout_mutation_lease,
    serialized_lock_update,
    update_checkout_mutation_lease,
)
from ..merge.checkout_lock import (
    acquire_checkout_mutation_lease as acquire_atomic_checkout_mutation_lease,
)
from ..merge.merge_conflict_repair import resolve_append_only_markdown_conflicts
from ..objectives.scan_receipts import (
    RefillScanResult,
    ScanTerminalReason,
    adapt_legacy_scan_result,
    build_scan_result,
    scan_identity,
)
from ..prompt.prompt_workflow import RescueOperation, prompt_workflow_cid
from ..proof.formal_verification_contracts import content_identity
from ..rescue.rescue_planner import (
    RescuePlanner,
    RescuePlannerPolicy,
    RescuePlanningRequest,
)
from ..rescue.supervisor_watchdog import (
    AUTONOMOUS_UNSTALL_STATE_SCHEMA,
    AutonomousUnstallCoordinator,
    AutonomousUnstallPolicy,
)
from ..runtime.event_log import (
    append_jsonl_event,
    repair_jsonl_event_log,
    unique_backup_path,
)
from ..runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_LEGACY_MARKDOWN,
    DATABASE_PROGRAM_JSON_ENV,
    FAILOVER_FAIL_CLOSED,
    TASK_SOURCE_LEGACY_MARKDOWN,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
    provider_subprocess_environment,
)
from ..runtime.resource_scheduler import evaluate_capacity_drift
from ..task_sources.plan_revision_store import PlanRevisionStore
from .core import ManagedDaemonSpec, terminate_pid_tree
from .implementation_daemon import (
    DEFAULT_TRACKS,
    IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME,
    IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME,
    IMPLEMENTATION_RUNNER_PROCESS_PATTERN,
    IMPLEMENTATION_TASK_CLAIM_LOCK_DIRNAME,
    IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
    TASK_HEADER_PREFIX,
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    consume_stale_active_attempt,
    implementation_task_claim_protected_fence_paths,
    load_json_dict,
    normalize_focus_tracks,
    normalize_implementation_protected_paths,
    normalize_llm_merge_resolver_command,
    normalize_relative_path_list,
    parse_task_file,
    parse_timestamp,
    process_command_line,
    process_is_running,
    state_file_repair_reason,
    utc_now,
    write_json_atomic,
    write_text_atomic,
)
from .implementation_supervisor_runner import (
    persist_goal_completion_projection,
    persist_supervisor_scan_receipt,
)
from .supervisor import (
    SupervisorStatusContext,
    active_codex_exec_workers,
    descendant_processes,
    worktree_phase_worker_status,
)
from .supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
    SupervisorLoopDecision,
)
from .supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
    OwnerLiveness,
    RestartPolicy,
    load_supervised_child_identity,
    read_process_birth,
    read_process_command_argv,
    supervised_child_identity_liveness,
    supervised_child_identity_path,
    terminate_direct_child_process,
    write_supervised_child_identity,
)
from .worktrees import WORKTREE_POOL_SCHEMA, pid_is_alive

REPO_ROOT = Path.cwd()

logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor")

RECOVERABLE_SUPERVISOR_LOOP_STATUSES = {"child_exited", "launch_failed", "max_restarts_reached"}
DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "3")
)
DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "3")
)
DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS = float(
    os.environ.get("IPFS_ACCELERATE_AGENT_WORKTREE_SCAN_CACHE_TTL_SECONDS", "900")
)
MAX_MANAGED_SUBMODULE_WORKTREE_PRUNES_PER_PASS = 32
MANAGED_SUBMODULE_WORKTREE_PRUNE_TIMEOUT_SECONDS = 30.0
SCHEDULER_CONFIG_SCHEMA_PATTERN = re.compile(
    r"^ipfs_accelerate_py\.agent_supervisor\."
    r"[a-z0-9_.-]+\.scheduler_config@1$"
)

# ---------------------------------------------------------------------------
# WPD-040 / SelectionDispositionProjection@1
# ---------------------------------------------------------------------------
# Minimal projection of planner/doctor implementation dispositions into
# supervisor selection status.  Closed disposition classes appear as typed
# selection_idle_reason codes; provider capacity backoff remains a distinct
# non-disposition idle class so operators never confuse model quota with
# doctor/planner outcomes.
SELECTION_DISPOSITION_PROJECTION_INTERFACE = "SelectionDispositionProjection@1"
SELECTION_DISPOSITION_PROJECTION_VERSION = 1
SELECTION_DISPOSITION_PROJECTION_EVIDENCE = "wpd/selection-disposition@1"
SELECTION_DISPOSITION_IDLE_REASON_PREFIX = "disposition_idle:"
PROVIDER_CAPACITY_BACKOFF_IDLE_REASON = "provider_capacity_backoff"
IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX = "implementation_retry_deferred:"

# Lower ranks are preferred when ranking ready work under policy.  Doctor /
# planner closed_deterministic readiness always outranks residual LLM work.
_DISPOSITION_SELECTION_PRIORITY: dict[str, int] = {
    "closed_deterministic": 0,
    "residual_llm_authorized": 1,
    "abstain_review": 2,
    "defer_capability": 3,
}

# Dispositions that leave ready work idle (no autonomous start without further
# authority).  residual_llm_authorized is runnable when capacity admits it;
# closed_deterministic is preferred runnable work, not an idle class.
_DISPOSITION_IDLE_CLASSES: frozenset[str] = frozenset(
    {
        "abstain_review",
        "defer_capability",
    }
)

# Heartbeat-fallback idle reasons that prove the content-addressed projection
# is intentionally idle (no active claim) without masking real work.
_QUIESCENT_EMPTY_BACKLOG_IDLE_REASONS: frozenset[str] = frozenset(
    {
        "no_shard_selectable_ready_tasks",
        "no_tasks_found",
    }
)
_QUIESCENT_POLICY_IDLE_REASONS: frozenset[str] = frozenset(
    {
        "all_selectable_ready_tasks_reached_max_task_attempts",
        "all_selectable_ready_tasks_deferred_by_resource_claim",
        "all_selectable_ready_tasks_deprioritized_as_off_mission",
        "no_eligible_ready_tasks_after_selection_filters",
        PROVIDER_CAPACITY_BACKOFF_IDLE_REASON,
    }
)


# Atomic checkout leases describe complete, bounded mutation transactions
# rather than projected task ownership.  A live owner of one of these
# recognized operations remains authoritative even when the supervisor's task
# state advances before the transaction releases its lease.
ATOMIC_CHECKOUT_MUTATION_LEASE_OPERATIONS = frozenset(
    {
        "cleanup_backlogged_worktrees",
        "commit_generated_file_update",
        "generated_board_update",
        "generated_dirty_repair",
        "implementation_protected_path_verification",
        "mark_tasks_completed",
        "merge_branch_to_main",
        "reopen_dependency_blocked_tasks",
        "repair_main_checkout_merge_state",
    }
)


class SupervisorSchedulerConfigError(ValueError):
    """Raised when a scheduler profile cannot safely configure the supervisor."""


class PlanBoundDispatchError(RuntimeError):
    """A slice no longer matches the canonical active plan or source fence."""


class PlanBoundReplanRequired(PlanBoundDispatchError):
    """A typed plan-bound proposal result requires a fenced wave replan."""


PLAN_BOUND_REPLAN_RETURN_CODE = 75


def _canonical_plan_bound_repo_root(path: Path | str) -> Path:
    """Return one lexical, real-directory repository root without following links."""

    root = Path(path)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise PlanBoundDispatchError(
            "plan-bound repository root must be lexical absolute"
        )
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise PlanBoundDispatchError(
                f"cannot lstat plan-bound repository component: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(
            observed.st_mode
        ):
            raise PlanBoundDispatchError(
                "plan-bound repository component is not a real directory: "
                f"{current}"
            )
    # Every component was inspected without following links.  This final
    # equality rejects lexical aliases such as mount/path spellings that do
    # not name the exact accepted root.
    if root.resolve(strict=True) != root:
        raise PlanBoundDispatchError(
            "plan-bound repository root is not canonical"
        )
    return root


def _plan_bound_contained_path(
    repo_root: Path,
    value: Path | str,
    *,
    field_name: str,
    require_existing: bool = False,
    require_regular: bool = False,
    require_directory: bool = False,
) -> Path:
    """Validate one lexical repo path before any normalizing resolution/write."""

    raw = Path(value)
    if raw.is_absolute():
        candidate = raw
        if Path(os.path.abspath(candidate)) != candidate:
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} must be lexical absolute"
            )
    else:
        if not str(raw) or str(raw) in {".", ".."} or ".." in raw.parts:
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} must be a safe repository path"
            )
        candidate = repo_root / raw
    try:
        relative = candidate.relative_to(repo_root)
    except ValueError as exc:
        raise PlanBoundDispatchError(
            f"plan-bound {field_name} escapes the accepted repository"
        ) from exc
    if not relative.parts:
        raise PlanBoundDispatchError(
            f"plan-bound {field_name} cannot be the repository root"
        )

    current = repo_root
    missing = False
    for index, part in enumerate(relative.parts):
        current /= part
        final = index == len(relative.parts) - 1
        if missing:
            continue
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            missing = True
            if require_existing:
                raise PlanBoundDispatchError(
                    f"plan-bound {field_name} is absent: {current}"
                )
            continue
        except OSError as exc:
            raise PlanBoundDispatchError(
                f"cannot lstat plan-bound {field_name}: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} contains a symbolic link: {current}"
            )
        if not final and not stat.S_ISDIR(observed.st_mode):
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} parent is not a directory: {current}"
            )
        if final and require_regular and (
            not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1
        ):
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} is not a single-link regular file"
            )
        if final and require_directory and not stat.S_ISDIR(observed.st_mode):
            raise PlanBoundDispatchError(
                f"plan-bound {field_name} is not a real directory"
            )
    return candidate


def _validated_plan_bound_authority_paths(
    *,
    repo_root: Path | str,
    accepted_tree_root: Path | str,
    state_dir: Path | str,
    plan_revision_store_path: Path | str,
    scheduler_config_path: Path | str | None = None,
    todo_path: Path | str | None = None,
    require_live_module_root: bool,
) -> tuple[Path, Path, Path, Path | None, Path | None]:
    """Bind plan state/store/config paths to one accepted lexical tree.

    The store constructor creates directories, so callers must invoke this
    helper before constructing ``PlanRevisionStore``.  Missing state/store
    leaves are allowed, but every existing parent is inspected with ``lstat``
    and the common runtime authority root must already be unambiguous.
    """

    root = _canonical_plan_bound_repo_root(repo_root)
    accepted = _canonical_plan_bound_repo_root(accepted_tree_root)
    if accepted != root:
        raise PlanBoundDispatchError(
            "plan-bound accepted tree differs from the repository root"
        )
    if require_live_module_root:
        module_root = _canonical_plan_bound_repo_root(
            Path(__file__).absolute().parents[3]
        )
        if accepted != module_root:
            raise PlanBoundDispatchError(
                "plan-bound accepted tree is not the live module root"
            )

    state = _plan_bound_contained_path(
        root,
        state_dir,
        field_name="state directory",
        require_directory=True,
    )
    store = _plan_bound_contained_path(
        root,
        plan_revision_store_path,
        field_name="plan revision store",
        require_directory=True,
    )
    if state.parent != store.parent or store.name != "plan-revision-store":
        raise PlanBoundDispatchError(
            "plan-bound state and store do not share the exact runtime state root"
        )
    config = (
        _plan_bound_contained_path(
            root,
            scheduler_config_path,
            field_name="scheduler config",
            require_existing=True,
            require_regular=True,
        )
        if scheduler_config_path is not None
        else None
    )
    todo = (
        _plan_bound_contained_path(
            root,
            todo_path,
            field_name="task board",
            require_existing=True,
            require_regular=True,
        )
        if todo_path is not None
        else None
    )
    return root, state, store, config, todo


PLAN_BOUND_DAEMON_CHILD_MARKER = "--run-plan-bound-daemon-child"
PLAN_BOUND_DAEMON_ENTRYPOINT = (
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
)


class _PlanBoundRevisionStoreView:
    """Read-through view that preserves and fences the store revision CID.

    ``PlanRevision`` intentionally excludes its CAS key from ``to_dict()``.
    The daemon's canonical binding loader consumes mapping-shaped values, so
    this view joins that key back to the typed revision without persisting a
    second projection or becoming mutable authority.
    """

    def __init__(
        self,
        store: PlanRevisionStore,
        expected_revision_cid: str,
        *,
        slice_manifest_cid: str,
        slice_id: str,
        lane_id: str,
        reassignment_cid: str,
    ) -> None:
        self._store = store
        self._expected_revision_cid = str(expected_revision_cid).strip()
        self._slice_manifest_cid = str(slice_manifest_cid).strip()
        self._slice_id = str(slice_id).strip()
        self._lane_id = str(lane_id).strip()
        self._reassignment_cid = str(reassignment_cid or "").strip()
        self._adapter = ProductionParallelPlanAdapter(store)

    def is_quarantined(self) -> bool:
        with self._store._thread_lock:  # noqa: SLF001
            with self._store._guard():  # noqa: SLF001
                active = _secure_store_active(self._store)
                return active is not None and bool(active.quarantined)

    def get_active(self) -> Any:
        with self._store._thread_lock:  # noqa: SLF001
            with self._store._guard():  # noqa: SLF001
                active = _secure_store_active(self._store)
                observed = str(
                    getattr(active, "revision_cid", "") or ""
                ).strip()
                if active is None or observed != self._expected_revision_cid:
                    raise PlanBoundDispatchError(
                        "active plan revision crossed the plan-bound child fence"
                    )
                try:
                    self._adapter._validate_slice_owner_locked(  # noqa: SLF001
                        revision_cid=self._expected_revision_cid,
                        slice_manifest_cid=self._slice_manifest_cid,
                        slice_id=self._slice_id,
                        lane_id=self._lane_id,
                        reassignment_cid=self._reassignment_cid,
                    )
                except Exception as exc:
                    raise PlanBoundDispatchError(
                        "plan-bound child lost its canonical slice ownership"
                    ) from exc
        return active

    def load_revision(self, revision_cid: str) -> Mapping[str, Any]:
        if str(revision_cid).strip() != self._expected_revision_cid:
            raise PlanBoundDispatchError(
                "plan-bound child requested a foreign plan revision"
            )
        with self._store._thread_lock:  # noqa: SLF001
            with self._store._guard():  # noqa: SLF001
                from ..planning.plan_revision_contracts import PlanRevision

                stored = _secure_store_cas(self._store, revision_cid)
                revision = PlanRevision.from_dict(stored)
                if revision.to_dict() != stored:
                    raise PlanBoundDispatchError(
                        "plan revision changed during typed decode"
                    )
        payload = revision.to_dict()
        payload["revision_cid"] = revision.revision_cid
        return payload

    def get_cas(self, cid: str) -> Mapping[str, Any]:
        with self._store._thread_lock:  # noqa: SLF001
            with self._store._guard():  # noqa: SLF001
                return _secure_store_cas(self._store, cid)


def _scheduler_config_sequence(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SupervisorSchedulerConfigError(
            f"{field_name} must be a sequence of strings"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SupervisorSchedulerConfigError(
                f"{field_name} must contain non-empty strings"
            )
        normalized = item.strip()
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _scheduler_config_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SupervisorSchedulerConfigError(f"{field_name} must be an object")
    return value


def _scheduler_config_relative_path(
    value: Any,
    *,
    field_name: str,
    repo_root: Path,
    must_exist: bool,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupervisorSchedulerConfigError(
            f"{field_name} must be a non-empty repo-relative path"
        )
    raw = value.strip()
    candidate = Path(raw)
    if (
        raw in {".", ".."}
        or raw.startswith(("/", "\\"))
        or raw.endswith(("/", "\\"))
        or "\\" in raw
        or "\0" in raw
        or "://" in raw
        or re.match(r"^[A-Za-z]:", raw)
        or candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise SupervisorSchedulerConfigError(
            f"{field_name} must be a safe repo-relative path: {raw!r}"
        )
    normalized = candidate.as_posix()
    try:
        resolved = (repo_root / normalized).resolve(strict=False)
        resolved.relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SupervisorSchedulerConfigError(
            f"{field_name} escapes the repository: {raw!r}"
        ) from exc
    if must_exist and not resolved.exists():
        raise SupervisorSchedulerConfigError(
            f"{field_name} does not exist: {raw!r}"
        )
    return normalized


def authority_epoch_seal_projection(
    manual_seals: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project seal configs into the authority-epoch preimage.

    ``expected_receipt_id`` is a mechanical pin rewrite after delegated
    completion verifies a seal.  Including it in the epoch preimage reopens
    the entire revalidation closure whenever a pin is updated even when the
    durable seal shape and verified receipt set are unchanged.
    """

    projected: dict[str, dict[str, Any]] = {}
    for task_id, seal_body in sorted(manual_seals.items()):
        if not isinstance(seal_body, Mapping):
            continue
        projected[str(task_id)] = {
            key: value
            for key, value in dict(seal_body).items()
            if key != "expected_receipt_id"
        }
    return projected


def load_supervisor_scheduler_config(
    path: Path | str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate a sealed scheduler profile without enabling effects.

    The profile is configuration input only.  It cannot turn on implementation,
    refill, Doctor mutation, or rollout; those remain explicit runtime actions.
    """

    root = (repo_root or REPO_ROOT).resolve()
    raw_path = Path(path)
    config_path = raw_path if raw_path.is_absolute() else root / raw_path
    if config_path.is_symlink():
        raise SupervisorSchedulerConfigError(
            "scheduler config must be a regular non-symlink file"
        )
    try:
        resolved_config_path = config_path.resolve(strict=True)
        resolved_config_path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SupervisorSchedulerConfigError(
            "scheduler config must be an existing file inside the repository"
        ) from exc
    if not resolved_config_path.is_file():
        raise SupervisorSchedulerConfigError(
            "scheduler config must be a regular non-symlink file"
        )
    try:
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorSchedulerConfigError(
            f"scheduler config is not valid JSON: {resolved_config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SupervisorSchedulerConfigError(
            "scheduler config root must be an object"
        )
    schema = payload.get("schema")
    if not isinstance(schema, str) or not SCHEDULER_CONFIG_SCHEMA_PATTERN.fullmatch(
        schema
    ):
        raise SupervisorSchedulerConfigError(
            "scheduler config schema must be a supported scheduler_config@1"
        )

    normalized = dict(payload)
    normalized["taskboard_path"] = _scheduler_config_relative_path(
        payload.get("taskboard_path"),
        field_name="taskboard_path",
        repo_root=root,
        must_exist=True,
    )
    normalized["objectives_path"] = _scheduler_config_relative_path(
        payload.get("objectives_path"),
        field_name="objectives_path",
        repo_root=root,
        must_exist=True,
    )
    task_prefix = payload.get("task_prefix")
    if (
        not isinstance(task_prefix, str)
        or not re.fullmatch(r"## [A-Z][A-Z0-9]*-", task_prefix.strip())
    ):
        raise SupervisorSchedulerConfigError(
            "task_prefix must be a canonical heading prefix such as '## PDR-'"
        )
    normalized["task_prefix"] = task_prefix.strip()
    namespace = payload.get("board_namespace")
    if (
        not isinstance(namespace, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", namespace.strip())
    ):
        raise SupervisorSchedulerConfigError(
            "board_namespace must be a canonical lowercase identifier"
        )
    normalized["board_namespace"] = namespace.strip()

    integer_fields = {
        "max_lanes": (1, 64),
        "max_restarts": (0, 10_000),
        "max_task_attempts": (0, 10_000),
        "implementation_timeout_seconds": (1, 7 * 24 * 60 * 60),
        "validation_max_workers": (1, 256),
    }
    for field_name, (minimum, maximum) in integer_fields.items():
        value = payload.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise SupervisorSchedulerConfigError(
                f"{field_name} must be an integer in [{minimum}, {maximum}]"
            )
        normalized[field_name] = value
    number_fields = {
        "poll_interval_seconds": (0.05, 86_400.0),
        "daemon_interval_seconds": (0.05, 86_400.0),
        "check_interval_seconds": (0.05, 86_400.0),
        "stale_seconds": (1.0, 30 * 24 * 60 * 60.0),
    }
    for field_name, (minimum, maximum) in number_fields.items():
        value = payload.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < minimum
            or float(value) > maximum
        ):
            raise SupervisorSchedulerConfigError(
                f"{field_name} must be a finite number in [{minimum}, {maximum}]"
            )
        normalized[field_name] = float(value)

    merge_target = payload.get("merge_target_branch")
    if (
        not isinstance(merge_target, str)
        or not merge_target.strip()
        or merge_target.startswith(("/", "-"))
        or merge_target.endswith("/")
        or ".." in merge_target
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", merge_target)
    ):
        raise SupervisorSchedulerConfigError(
            "merge_target_branch is not a safe branch name"
        )
    normalized["merge_target_branch"] = merge_target

    authority_switches = (
        ("derived_refill", "enabled_at_bootstrap"),
        ("doctor", "enabled_at_bootstrap"),
        ("doctor", "mutation_authorized"),
        ("doctor", "narrow_autonomous_mutation_enabled"),
        ("rollout", "automatic_enabled"),
    )
    for section_name, switch_name in authority_switches:
        section = _scheduler_config_mapping(
            payload.get(section_name, {}),
            field_name=section_name,
        )
        switch = section.get(switch_name, False)
        if not isinstance(switch, bool):
            raise SupervisorSchedulerConfigError(
                f"{section_name}.{switch_name} must be a boolean"
            )
        if switch:
            raise SupervisorSchedulerConfigError(
                f"{section_name}.{switch_name} cannot be enabled by a "
                "scheduler bootstrap profile"
            )

    submodules = _scheduler_config_sequence(
        payload.get("worktree_submodule_paths", ()),
        field_name="worktree_submodule_paths",
    )
    normalized["worktree_submodule_paths"] = tuple(
        _scheduler_config_relative_path(
            item,
            field_name="worktree_submodule_paths",
            repo_root=root,
            must_exist=True,
        )
        for item in submodules
    )
    protected_paths = _scheduler_config_sequence(
        payload.get("protected_paths", ()),
        field_name="protected_paths",
    )
    try:
        normalized_protected_paths = normalize_implementation_protected_paths(
            protected_paths,
            repo_root=root,
        )
    except ValueError as exc:
        raise SupervisorSchedulerConfigError(str(exc)) from exc

    staged_raw = _scheduler_config_mapping(
        payload.get("protected_after_manual_completion", {}),
        field_name="protected_after_manual_completion",
    )
    tasks = parse_task_file(
        root / normalized["taskboard_path"],
        normalized["task_prefix"],
    )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise SupervisorSchedulerConfigError(
            "taskboard contains duplicate task IDs"
        )
    task_by_id = {task.task_id: task for task in tasks}
    seal_config_fields = {
        "artifact_paths",
        "grant_action",
        "grant_claims",
        "grant_type",
        "interface",
        "policy_revision",
        "expected_receipt_id",
        "receipt_path",
        "reviewed_base_claims",
        "schema",
    }
    manual_seal_raw = _scheduler_config_mapping(
        payload.get("manual_completion_seals", {}),
        field_name="manual_completion_seals",
    )
    manual_seals: dict[str, dict[str, Any]] = {}
    for task_id, raw_seal in manual_seal_raw.items():
        if not isinstance(task_id, str) or task_id not in task_by_id:
            raise SupervisorSchedulerConfigError(
                "manual_completion_seals keys must name declared tasks"
            )
        if task_by_id[task_id].completion != "manual":
            raise SupervisorSchedulerConfigError(
                "manual_completion_seals tasks must use manual completion"
            )
        seal = _scheduler_config_mapping(
            raw_seal,
            field_name=f"manual_completion_seals.{task_id}",
        )
        if set(seal) != seal_config_fields:
            raise SupervisorSchedulerConfigError(
                f"manual_completion_seals.{task_id} fields do not match "
                "the closed schema"
            )
        strings: dict[str, str] = {}
        for field_name in (
            "grant_action",
            "grant_type",
            "interface",
            "policy_revision",
            "schema",
        ):
            value = seal.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise SupervisorSchedulerConfigError(
                    f"manual_completion_seals.{task_id}.{field_name} "
                    "must be a non-empty string"
                )
            strings[field_name] = value.strip()
        expected_receipt_id = seal.get("expected_receipt_id")
        if (
            not isinstance(expected_receipt_id, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_receipt_id)
        ):
            raise SupervisorSchedulerConfigError(
                f"manual_completion_seals.{task_id}.expected_receipt_id "
                "must be a canonical SHA-256 identity"
            )
        receipt_path = _scheduler_config_relative_path(
            seal.get("receipt_path"),
            field_name=f"manual_completion_seals.{task_id}.receipt_path",
            repo_root=root,
            must_exist=False,
        )
        raw_artifacts = _scheduler_config_mapping(
            seal.get("artifact_paths"),
            field_name=f"manual_completion_seals.{task_id}.artifact_paths",
        )
        artifact_paths: dict[str, str] = {}
        for role, raw_artifact_path in raw_artifacts.items():
            if (
                not isinstance(role, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", role)
            ):
                raise SupervisorSchedulerConfigError(
                    f"manual_completion_seals.{task_id} artifact roles "
                    "must be canonical identifiers"
                )
            artifact_paths[role] = _scheduler_config_relative_path(
                raw_artifact_path,
                field_name=(
                    f"manual_completion_seals.{task_id}.artifact_paths.{role}"
                ),
                repo_root=root,
                must_exist=False,
            )
        if not artifact_paths or len(set(artifact_paths.values())) != len(
            artifact_paths
        ):
            raise SupervisorSchedulerConfigError(
                f"manual_completion_seals.{task_id} artifact paths must "
                "be non-empty and unique"
            )
        normalized_claims: dict[str, dict[str, Any]] = {}
        for field_name in ("grant_claims", "reviewed_base_claims"):
            raw_claims = _scheduler_config_mapping(
                seal.get(field_name),
                field_name=f"manual_completion_seals.{task_id}.{field_name}",
            )
            claims: dict[str, Any] = {}
            for claim_name, claim_value in raw_claims.items():
                if (
                    not isinstance(claim_name, str)
                    or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", claim_name)
                    or isinstance(claim_value, float)
                    or not isinstance(claim_value, (str, int, bool))
                ):
                    raise SupervisorSchedulerConfigError(
                        f"manual_completion_seals.{task_id}.{field_name} "
                        "must contain canonical scalar claims"
                    )
                claims[claim_name] = claim_value
            normalized_claims[field_name] = claims
        manual_seals[task_id] = {
            **strings,
            **normalized_claims,
            "expected_receipt_id": expected_receipt_id,
            "receipt_path": receipt_path,
            "artifact_paths": artifact_paths,
        }

    staged_protected_paths: dict[str, tuple[str, ...]] = {}
    activated_task_ids: list[str] = []
    verified_manual_seals: dict[str, str] = {}
    active_paths = list(normalized_protected_paths)
    for task_id, raw_paths in staged_raw.items():
        if not isinstance(task_id, str) or task_id not in task_by_id:
            raise SupervisorSchedulerConfigError(
                "protected_after_manual_completion keys must name declared tasks"
            )
        task = task_by_id[task_id]
        if task.completion != "manual":
            raise SupervisorSchedulerConfigError(
                "protected_after_manual_completion tasks must use manual completion"
            )
        staged_values = _scheduler_config_sequence(
            raw_paths,
            field_name=f"protected_after_manual_completion.{task_id}",
        )
        try:
            staged_paths = normalize_implementation_protected_paths(
                staged_values,
                repo_root=root,
            )
        except ValueError as exc:
            raise SupervisorSchedulerConfigError(str(exc)) from exc
        if not staged_paths:
            raise SupervisorSchedulerConfigError(
                f"protected_after_manual_completion.{task_id} cannot be empty"
            )
        try:
            declared_outputs = set(
                normalize_implementation_protected_paths(
                    task.outputs,
                    repo_root=root,
                )
            )
        except ValueError as exc:
            raise SupervisorSchedulerConfigError(
                f"{task_id} has an unsafe declared output: {exc}"
            ) from exc
        undeclared_paths = set(staged_paths) - declared_outputs
        if undeclared_paths:
            raise SupervisorSchedulerConfigError(
                "protected_after_manual_completion paths must be declared "
                f"task outputs: {sorted(undeclared_paths)!r}"
            )
        omitted_paths = declared_outputs - set(staged_paths)
        if omitted_paths:
            raise SupervisorSchedulerConfigError(
                "protected_after_manual_completion must protect every "
                f"declared task output: {sorted(omitted_paths)!r}"
            )
        seal = manual_seals.get(task_id)
        if seal is not None:
            receipt_path = str(seal["receipt_path"])
            artifact_path_set = set(seal["artifact_paths"].values())
            if receipt_path not in staged_paths:
                raise SupervisorSchedulerConfigError(
                    f"{task_id} manual seal receipt must become protected"
                )
            if artifact_path_set != declared_outputs - {receipt_path}:
                raise SupervisorSchedulerConfigError(
                    f"{task_id} manual seal must bind every non-receipt output"
                )
        staged_protected_paths[task_id] = staged_paths
        if task.status != "completed":
            continue
        if seal is None:
            raise SupervisorSchedulerConfigError(
                f"completed manual protection task {task_id} has no "
                "operator seal configuration"
            )
        try:
            from ..control.delegated_operator_completion import (
                DelegatedOperatorCompletionPolicy,
            )

            delegated_policy = DelegatedOperatorCompletionPolicy.from_mapping(
                payload.get("delegated_operator_completion")
                if isinstance(payload, Mapping)
                else None
            )
        except Exception:
            delegated_policy = None
        allow_delegated = bool(
            delegated_policy is not None and delegated_policy.allows(task_id)
        )
        try:
            verified = verify_manual_completion_seal(
                str(seal["receipt_path"]),
                repo_root=root,
                task_id=task_id,
                board_namespace=normalized["board_namespace"],
                schema=str(seal["schema"]),
                interface=str(seal["interface"]),
                policy_revision=str(seal["policy_revision"]),
                expected_receipt_id=str(seal["expected_receipt_id"]),
                artifact_paths=seal["artifact_paths"],
                grant_type=str(seal["grant_type"]),
                grant_action=str(seal["grant_action"]),
                reviewed_base_claims=seal["reviewed_base_claims"],
                grant_claims=seal["grant_claims"],
                allow_delegated_operator=allow_delegated,
            )
        except ManualCompletionSealError as exc:
            raise SupervisorSchedulerConfigError(
                f"manual completion seal verification failed for {task_id}: {exc}"
            ) from exc
        verified_manual_seals[task_id] = str(verified["receipt_id"])
        for relative in staged_paths:
            candidate = root / relative
            if not candidate.is_file():
                raise SupervisorSchedulerConfigError(
                    "completed manual protection task references a missing "
                    f"or non-file artifact: {relative!r}"
                )
            if relative not in active_paths:
                active_paths.append(relative)
        activated_task_ids.append(task_id)

    orphaned_seal_configs = set(manual_seals) - set(staged_protected_paths)
    if orphaned_seal_configs:
        raise SupervisorSchedulerConfigError(
            "manual_completion_seals tasks must also declare staged protection: "
            f"{sorted(orphaned_seal_configs)!r}"
        )

    try:
        from ..control.delegated_operator_completion import (
            DelegatedOperatorCompletionPolicy,
        )

        delegated_policy = DelegatedOperatorCompletionPolicy.from_mapping(
            payload.get("delegated_operator_completion")
            if isinstance(payload, Mapping)
            else None
        )
    except Exception as exc:
        raise SupervisorSchedulerConfigError(
            f"delegated_operator_completion is invalid: {exc}"
        ) from exc
    normalized["delegated_operator_completion"] = {
        "enabled": delegated_policy.enabled,
        "allowed_task_ids": sorted(delegated_policy.allowed_task_ids),
        "require_validation": delegated_policy.require_validation,
        "validation_timeout_seconds": (
            delegated_policy.validation_timeout_seconds
        ),
    }

    normalized["protected_after_manual_completion"] = staged_protected_paths
    normalized["manual_completion_seals"] = manual_seals
    normalized["verified_manual_completion_seals"] = verified_manual_seals
    normalized["activated_protected_task_ids"] = tuple(activated_task_ids)
    normalized["manual_completion_authority_task_ids"] = tuple(
        sorted(staged_protected_paths)
    )
    normalized["manual_completion_authority_required_task_ids"] = tuple(
        sorted(set(staged_protected_paths) - set(activated_task_ids))
    )
    authority_epoch = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "manual-completion-authority-epoch@1"
        ),
        "board_namespace": str(normalized["board_namespace"]),
        "taskboard_path": str(normalized["taskboard_path"]),
        "protected_after_manual_completion": staged_protected_paths,
        "manual_completion_seals": authority_epoch_seal_projection(
            manual_seals
        ),
        "verified_manual_completion_seals": verified_manual_seals,
        "task_ids": list(normalized["manual_completion_authority_task_ids"]),
        "required_task_ids": list(
            normalized["manual_completion_authority_required_task_ids"]
        ),
    }
    normalized["manual_completion_authority_epoch_id"] = (
        content_identity(authority_epoch) if staged_protected_paths else ""
    )
    normalized["protected_paths"] = tuple(active_paths)
    normalized["_config_path"] = str(resolved_config_path)
    return normalized


def supervisor_scheduler_config_cli_defaults(
    profile: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """Translate a validated profile into conservative existing CLI options."""

    root = (repo_root or REPO_ROOT).resolve()
    state_prefix = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(profile["board_namespace"]).lower(),
    ).strip("_")[:80]
    options = [
        "--todo-path",
        str(root / str(profile["taskboard_path"])),
        "--task-prefix",
        str(profile["task_prefix"]),
        "--state-prefix",
        state_prefix,
        "--stale-seconds",
        str(profile["stale_seconds"]),
        "--check-interval",
        str(profile["check_interval_seconds"]),
        "--max-restarts",
        str(profile["max_restarts"]),
        "--max-task-attempts",
        str(profile["max_task_attempts"]),
        "--daemon-interval",
        str(profile["daemon_interval_seconds"]),
        "--implementation-timeout",
        str(profile["implementation_timeout_seconds"]),
        "--validation-max-workers",
        str(profile["validation_max_workers"]),
        "--merge-target-branch",
        str(profile["merge_target_branch"]),
        "--objective-path",
        str(root / str(profile["objectives_path"])),
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
    ]
    for path in profile["worktree_submodule_paths"]:
        options.extend(("--worktree-submodule-path", str(path)))
    for path in profile["protected_paths"]:
        options.extend(("--implementation-protected-path", str(path)))
    for task_id in profile["manual_completion_authority_task_ids"]:
        options.extend(("--manual-completion-authority-task-id", str(task_id)))
    for task_id in profile["manual_completion_authority_required_task_ids"]:
        options.extend(
            ("--manual-completion-authority-required-task-id", str(task_id))
        )
    authority_epoch_id = str(
        profile.get("manual_completion_authority_epoch_id") or ""
    ).strip()
    if authority_epoch_id:
        options.extend(
            ("--manual-completion-authority-epoch-id", authority_epoch_id)
        )
    return options


def expand_supervisor_scheduler_config_args(
    argv: Sequence[str],
    *,
    repo_root: Path | None = None,
) -> tuple[list[str], Path | None]:
    """Prepend scheduler defaults while preserving later explicit overrides."""

    raw = [str(item) for item in argv]
    if "-h" in raw or "--help" in raw:
        return raw, None
    config_values: list[str] = []
    remaining: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "--scheduler-config":
            if index + 1 >= len(raw):
                raise SupervisorSchedulerConfigError(
                    "--scheduler-config requires a path"
                )
            config_values.append(raw[index + 1])
            index += 2
            continue
        if token.startswith("--scheduler-config="):
            config_values.append(token.split("=", 1)[1])
            index += 1
            continue
        remaining.append(token)
        index += 1
    if not config_values:
        return remaining, None
    if len(config_values) != 1 or not config_values[0].strip():
        raise SupervisorSchedulerConfigError(
            "--scheduler-config may be supplied exactly once"
        )
    root = (repo_root or REPO_ROOT).resolve()
    profile = load_supervisor_scheduler_config(config_values[0], repo_root=root)
    defaults = supervisor_scheduler_config_cli_defaults(profile, repo_root=root)
    return [*defaults, *remaining], Path(str(profile["_config_path"]))


def database_program_from_cli_namespace(
    args: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> DatabaseProgramConfig | None:
    """Build a database program selection from parsed supervisor CLI args/env."""

    environment = os.environ if environ is None else environ
    env_payload = str(environment.get(DATABASE_PROGRAM_JSON_ENV, "") or "").strip()
    env_program: DatabaseProgramConfig | None = None
    if env_payload:
        try:
            parsed = json.loads(env_payload)
        except json.JSONDecodeError as exc:
            raise DatabaseProgramConfigError(
                "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON is not valid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise DatabaseProgramConfigError(
                "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON must be an object"
            )
        env_program = DatabaseProgramConfig.from_mapping(parsed)

    authority_mode = str(getattr(args, "authority_mode", "") or "").strip()
    task_source_kind = str(getattr(args, "task_source_kind", "") or "").strip()
    explicit_legacy = bool(getattr(args, "explicit_legacy_task_source", False))
    if not authority_mode and not task_source_kind and env_program is None:
        return None
    if env_program is not None and not authority_mode and not task_source_kind:
        return env_program

    if not task_source_kind and env_program is not None:
        task_source_kind = env_program.task_source_kind
    if not authority_mode and env_program is not None:
        authority_mode = env_program.authority_mode
    if not task_source_kind:
        if authority_mode or explicit_legacy:
            raise DatabaseProgramConfigError(
                "task_source_kind is required when authority options are set; "
                "the implicit legacy-Markdown default is deprecated"
            )
        return env_program
    if not authority_mode:
        if task_source_kind in {TASK_SOURCE_LEGACY_MARKDOWN, "markdown"}:
            authority_mode = AUTHORITY_MODE_LEGACY_MARKDOWN
            explicit_legacy = True
        elif task_source_kind == "duckdb":
            authority_mode = "embedded"
        else:
            raise DatabaseProgramConfigError(
                "cannot infer authority_mode for task_source_kind "
                f"{task_source_kind!r}"
            )

    payload = {
        "authority_mode": authority_mode,
        "task_source_kind": task_source_kind,
        "endpoint_secret_handle": str(
            getattr(args, "endpoint_secret_handle", "") or ""
        ).strip()
        or (env_program.endpoint_secret_handle if env_program else ""),
        "store_id": str(getattr(args, "state_store_id", "") or "").strip()
        or (env_program.store_id if env_program else ""),
        "store_generation": str(
            getattr(args, "state_store_generation", "") or ""
        ).strip()
        or (env_program.store_generation if env_program else ""),
        "schema_revision": str(
            getattr(args, "state_schema_revision", "") or ""
        ).strip()
        or (env_program.schema_revision if env_program else ""),
        "event_store_path": str(
            getattr(args, "event_store_path", "") or ""
        ).strip()
        or (env_program.event_store_path if env_program else ""),
        "runtime_registry_path": str(
            getattr(args, "runtime_registry_path", "") or ""
        ).strip()
        or (env_program.runtime_registry_path if env_program else ""),
        # --worktree-root is already resolved by the supervisor and may be
        # absolute. It is not the database program's repository-relative root.
        "worktree_root": "",
        "export_profile": str(getattr(args, "export_profile", "") or "").strip()
        or (env_program.export_profile if env_program else ""),
        "failover_policy": str(
            getattr(args, "state_failover_policy", "") or ""
        ).strip()
        or (env_program.failover_policy if env_program else FAILOVER_FAIL_CLOSED),
        "explicit_legacy": explicit_legacy
        or bool(env_program.explicit_legacy if env_program else False)
        or authority_mode == AUTHORITY_MODE_LEGACY_MARKDOWN,
    }
    return DatabaseProgramConfig.from_mapping(payload)


def _managed_daemon_child_environment(
    *,
    database_program: DatabaseProgramConfig | None = None,
) -> dict[str, str]:
    """Keep source code and explicit state authority bound in daemon children."""

    entries: list[str] = []
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "ipfs_accelerate_py").is_dir():
        entries.append(str(source_root))
    for raw_entry in sys.path:
        if not raw_entry:
            continue
        try:
            candidate = Path(raw_entry).resolve()
        except OSError:
            continue
        if (candidate / "ipfs_accelerate_py").is_dir():
            entries.append(str(candidate))
    entries.extend(
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    )
    pythonpath = os.pathsep.join(dict.fromkeys(entries))
    environment = (
        dict(database_program.environment())
        if database_program is not None
        else {}
    )
    if pythonpath:
        environment["PYTHONPATH"] = pythonpath
    return environment


def provider_environment_without_state_credentials(
    environment: Mapping[str, str] | None = None,
    *,
    database_program: DatabaseProgramConfig | None = None,
) -> dict[str, str]:
    """Return an implementation-provider environment without state secrets."""

    return provider_subprocess_environment(
        environment,
        program=database_program,
    )


def _normalize_disposition_token(value: Any) -> str:
    """Return a closed disposition wire value or raise ValueError."""

    from .implementation_disposition import (
        ImplementationDisposition,
        parse_implementation_disposition,
    )

    if isinstance(value, ImplementationDisposition):
        return value.value
    if isinstance(value, Mapping):
        raw = value.get("disposition")
        if raw is None:
            raise ValueError("disposition mapping requires a disposition field")
        return parse_implementation_disposition(raw).value
    return parse_implementation_disposition(value).value


def disposition_selection_idle_reason(disposition: Any) -> str:
    """Return the selection_idle_reason code for a doctor/planner disposition.

    Disposition idle classes are distinct from
    :data:`PROVIDER_CAPACITY_BACKOFF_IDLE_REASON`.  Every closed disposition
    value has a stable idle-reason code so status consumers can attribute
    idle loops to planner/doctor outcomes rather than model capacity.
    """

    token = _normalize_disposition_token(disposition)
    return f"{SELECTION_DISPOSITION_IDLE_REASON_PREFIX}{token}"


def closed_disposition_selection_idle_reasons() -> frozenset[str]:
    """Return the closed set of disposition-class selection_idle_reason codes."""

    from .implementation_disposition import closed_disposition_values

    return frozenset(
        disposition_selection_idle_reason(value)
        for value in closed_disposition_values()
    )


def is_disposition_selection_idle_reason(reason: Any) -> bool:
    """Return whether ``reason`` is a typed doctor/planner disposition idle code."""

    if not isinstance(reason, str) or not reason:
        return False
    if not reason.startswith(SELECTION_DISPOSITION_IDLE_REASON_PREFIX):
        return False
    token = reason[len(SELECTION_DISPOSITION_IDLE_REASON_PREFIX) :]
    if not token or ":" in token or any(char.isspace() for char in token):
        return False
    try:
        _normalize_disposition_token(token)
    except Exception:
        return False
    return True


def is_provider_capacity_backoff_idle_reason(reason: Any) -> bool:
    """Return whether ``reason`` is provider capacity backoff (not a disposition).

    Capacity backoff remains a first-class idle class so residual LLM work
    deferred for quota is never re-labeled as a planner/doctor disposition.
    """

    if not isinstance(reason, str) or not reason:
        return False
    if reason == PROVIDER_CAPACITY_BACKOFF_IDLE_REASON:
        return True
    if reason == (
        f"{IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX}"
        f"{PROVIDER_CAPACITY_BACKOFF_IDLE_REASON}"
    ):
        return True
    return False


def disposition_selection_priority_hint(
    disposition: Any,
    *,
    prefer_closed_deterministic: bool = True,
) -> int:
    """Return a lower-is-better selection rank for a disposition class.

    Under the default policy, ``closed_deterministic`` readiness ranks ahead
    of residual LLM work so the scheduler prefers doctor/planner closes over
    model-heavy residuals when both are ready.
    """

    token = _normalize_disposition_token(disposition)
    rank = _DISPOSITION_SELECTION_PRIORITY.get(token)
    if rank is None:
        raise ValueError(f"unknown disposition for selection priority: {token!r}")
    if not prefer_closed_deterministic and token == "closed_deterministic":
        # Policy may opt out of the deterministic preference; residual then
        # shares the primary rank without inventing a new disposition class.
        return _DISPOSITION_SELECTION_PRIORITY["residual_llm_authorized"]
    return rank


def compare_disposition_selection_priority(
    left: Any,
    right: Any,
    *,
    prefer_closed_deterministic: bool = True,
) -> int:
    """Compare two dispositions for selection preference.

    Returns ``-1`` when ``left`` should run first, ``1`` when ``right`` should
    run first, and ``0`` when ranks tie.
    """

    left_rank = disposition_selection_priority_hint(
        left, prefer_closed_deterministic=prefer_closed_deterministic
    )
    right_rank = disposition_selection_priority_hint(
        right, prefer_closed_deterministic=prefer_closed_deterministic
    )
    if left_rank < right_rank:
        return -1
    if left_rank > right_rank:
        return 1
    return 0


def rank_tasks_by_disposition_priority(
    task_dispositions: Mapping[str, Any],
    *,
    prefer_closed_deterministic: bool = True,
) -> list[str]:
    """Order task ids so closed_deterministic readiness precedes residual LLM.

    Unknown or missing dispositions fail closed by sorting after every known
    class.  Tie-breaks are stable by task id.
    """

    ranked: list[tuple[int, str]] = []
    for task_id, disposition in task_dispositions.items():
        key = str(task_id)
        if not key:
            continue
        try:
            rank = disposition_selection_priority_hint(
                disposition,
                prefer_closed_deterministic=prefer_closed_deterministic,
            )
        except Exception:
            rank = max(_DISPOSITION_SELECTION_PRIORITY.values()) + 1
        ranked.append((rank, key))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [task_id for _, task_id in ranked]


def project_selection_disposition(
    status: Mapping[str, Any] | None = None,
    *,
    ready_task_dispositions: Mapping[str, Any] | None = None,
    prefer_closed_deterministic: bool = True,
    provider_capacity_backoff: bool = False,
    selected_task_id: str = "",
) -> dict[str, Any]:
    """Project disposition classes into status / selection_idle_reason.

    Interface: :data:`SELECTION_DISPOSITION_PROJECTION_INTERFACE`.

    Rules (fail-closed, minimal projection):

    1. When provider capacity backoff is active, ``selection_idle_reason`` is
       exactly :data:`PROVIDER_CAPACITY_BACKOFF_IDLE_REASON` — never a
       disposition class.
    2. When no task is selected and every ready disposition is an idle class
       (``abstain_review`` / ``defer_capability``), project the dominant
       disposition idle reason.
    3. Always attach ordered ``selection_disposition_priority_hints`` so
       schedulers can prefer ``closed_deterministic`` over residual LLM work.
    4. Existing non-disposition idle reasons on the input status are preserved
       unless disposition projection replaces them under rules 1–2.
    """

    base = dict(status or {})
    dispositions = {
        str(task_id): _normalize_disposition_token(value)
        for task_id, value in dict(ready_task_dispositions or {}).items()
        if str(task_id)
    }
    ordered_ids = rank_tasks_by_disposition_priority(
        dispositions,
        prefer_closed_deterministic=prefer_closed_deterministic,
    )
    priority_hints = [
        {
            "task_id": task_id,
            "disposition": dispositions[task_id],
            "priority_hint": disposition_selection_priority_hint(
                dispositions[task_id],
                prefer_closed_deterministic=prefer_closed_deterministic,
            ),
            "prefer_closed_deterministic": bool(prefer_closed_deterministic),
        }
        for task_id in ordered_ids
    ]

    selected = str(
        selected_task_id
        or base.get("active_task_id")
        or base.get("recommended_task_id")
        or ""
    ).strip()
    existing_idle = str(base.get("selection_idle_reason") or "")
    has_closed_deterministic = any(
        token == "closed_deterministic" for token in dispositions.values()
    )
    residual_ready_count = sum(
        1
        for value in dispositions.values()
        if value == "residual_llm_authorized"
    )
    # Capacity backoff only idles residual LLM work.  Prefer closed_deterministic
    # readiness over residual when both are present under policy.
    residual_blocked_by_capacity = bool(
        provider_capacity_backoff
        and not selected
        and residual_ready_count > 0
        and not (prefer_closed_deterministic and has_closed_deterministic)
    )

    if selected:
        idle_reason = ""
    elif residual_blocked_by_capacity:
        idle_reason = PROVIDER_CAPACITY_BACKOFF_IDLE_REASON
    elif dispositions and all(
        token in _DISPOSITION_IDLE_CLASSES for token in dispositions.values()
    ):
        # Dominant idle disposition: lowest priority_hint among present classes
        # (stable, closed vocabulary).
        dominant = min(
            dispositions.values(),
            key=lambda token: (
                disposition_selection_priority_hint(
                    token,
                    prefer_closed_deterministic=prefer_closed_deterministic,
                ),
                token,
            ),
        )
        idle_reason = disposition_selection_idle_reason(dominant)
    elif existing_idle and (
        is_provider_capacity_backoff_idle_reason(existing_idle)
        or is_disposition_selection_idle_reason(existing_idle)
        or existing_idle in _QUIESCENT_EMPTY_BACKLOG_IDLE_REASONS
        or existing_idle in _QUIESCENT_POLICY_IDLE_REASONS
        or existing_idle.startswith("resource_claim_deferred:")
    ):
        idle_reason = existing_idle
    else:
        idle_reason = existing_idle

    # When residual is capacity-blocked but closed work remains, prefer that
    # closed task id for selection hints without clearing residual visibility.
    if (
        provider_capacity_backoff
        and prefer_closed_deterministic
        and has_closed_deterministic
        and not selected
    ):
        preferred_task_id = next(
            (
                task_id
                for task_id in ordered_ids
                if dispositions.get(task_id) == "closed_deterministic"
            ),
            ordered_ids[0] if ordered_ids else "",
        )
        idle_reason = ""
    elif ordered_ids and not selected:
        preferred_task_id = ordered_ids[0]
    else:
        preferred_task_id = selected
    preferred_disposition = (
        dispositions.get(preferred_task_id, "") if preferred_task_id else ""
    )

    projected = dict(base)
    projected["selection_disposition_projection"] = {
        "interface": SELECTION_DISPOSITION_PROJECTION_INTERFACE,
        "contract_version": SELECTION_DISPOSITION_PROJECTION_VERSION,
        "evidence": SELECTION_DISPOSITION_PROJECTION_EVIDENCE,
        "prefer_closed_deterministic": bool(prefer_closed_deterministic),
        "provider_capacity_backoff": bool(provider_capacity_backoff),
        "ready_disposition_counts": {
            token: sum(1 for value in dispositions.values() if value == token)
            for token in sorted(_DISPOSITION_SELECTION_PRIORITY)
        },
        "preferred_task_id": preferred_task_id,
        "preferred_disposition": preferred_disposition,
        "residual_deferred_by_provider_capacity": (
            residual_ready_count if provider_capacity_backoff else 0
        ),
    }
    projected["selection_disposition_priority_hints"] = priority_hints
    projected["selection_idle_reason"] = idle_reason
    return projected


def _selection_idle_reason_is_quiescent(reason: Any) -> bool:
    """Return whether an idle reason is a known intentional idle class."""

    if not isinstance(reason, str) or not reason:
        return False
    if reason in _QUIESCENT_EMPTY_BACKLOG_IDLE_REASONS:
        return True
    if reason in _QUIESCENT_POLICY_IDLE_REASONS:
        return True
    if is_provider_capacity_backoff_idle_reason(reason):
        return True
    if is_disposition_selection_idle_reason(reason):
        return True
    if reason.startswith("resource_claim_deferred:") and len(reason) > len(
        "resource_claim_deferred:"
    ):
        return True
    if reason.startswith(IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX):
        suffix = reason[len(IMPLEMENTATION_RETRY_DEFERRED_IDLE_PREFIX) :]
        return bool(suffix) and (
            is_provider_capacity_backoff_idle_reason(suffix)
            or is_disposition_selection_idle_reason(suffix)
            or suffix in _QUIESCENT_POLICY_IDLE_REASONS
        )
    return False


def _projection_is_quiescent_for_heartbeat_fallback(
    status: Mapping[str, Any],
) -> bool:
    """Recognize an idle content-addressed task projection without masking work."""

    required_fields = {
        "active_task_id",
        "implementation_in_progress",
        "ready_count",
        "selectable_ready_count",
        "eligible_ready_count",
        "blocked_count",
        "selection_idle_reason",
    }
    if not required_fields.issubset(status):
        return False
    active_task_id = status["active_task_id"]
    if not isinstance(active_task_id, str) or active_task_id:
        return False
    if status["implementation_in_progress"] is not False:
        return False
    idle_reason = status["selection_idle_reason"]
    if not _selection_idle_reason_is_quiescent(idle_reason):
        return False
    for field_name in (
        "ready_count",
        "selectable_ready_count",
        "eligible_ready_count",
        "blocked_count",
    ):
        value = status[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    # Empty-backlog idle reasons must not report phantom ready work.
    if idle_reason in _QUIESCENT_EMPTY_BACKLOG_IDLE_REASONS:
        for field_name in (
            "ready_count",
            "selectable_ready_count",
            "eligible_ready_count",
        ):
            if status[field_name] != 0:
                return False
    return True


class ObjectiveRefillTimeoutError(TimeoutError):
    """Raised when supervisor-owned objective refill exceeds its local budget."""


class CodebaseRefillTimeoutError(TimeoutError):
    """Raised when supervisor-owned codebase refill exceeds its local budget."""


class ObjectiveCompletionArtifactRefreshError(RuntimeError):
    """Raised when the configured completion-artifact producer cannot refresh."""


OBJECTIVE_REFILL_ANALYZER_VERSION = "objective-daemon-v1"
CODEBASE_REFILL_ANALYZER_VERSION = "codebase-scan-v1"

# Fields derived exclusively from a validated ``ProofRolloutStatus``.  Keeping
# the set explicit lets a long-running supervisor replace the whole projection
# when a durable policy configuration changes, instead of accidentally mixing
# fields from two policy identities.
PROOF_ROLLOUT_PROJECTION_FIELDS = frozenset(
    {
        "proof_rollout",
        "proof_rollout_snapshot_id",
        "proof_policy_id",
        "proof_rollout_mode",
        "proof_rollout_blocking",
        "proof_provider_health_can_change_mode",
        "proof_capability_healthy",
        "proof_protected_scope_count",
        "proof_active_plan_count",
        "proof_override_count",
        "proof_active_override_count",
        "proof_failure_count",
        "proof_fallback_count",
        "proof_assurance_counts",
    }
)


def apply_proof_rollout_projection(
    payload: Mapping[str, Any],
    rollout_status: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Attach the bounded proof-rollout view to a supervisor status payload.

    The detailed snapshot remains nested so future fields do not collide with
    the daemon heartbeat schema.  A few stable operator fields are projected
    at the top level for health checks and older status consumers.
    """

    from ..proof.formal_verification_policy import (
        PROOF_ROLLOUT_STATUS_SCHEMA,
        ProofRolloutStatus,
    )

    converter = getattr(rollout_status, "to_dict", None)
    raw = converter() if callable(converter) else rollout_status
    if not isinstance(raw, Mapping):
        raise TypeError("rollout_status must be a mapping or expose to_dict()")
    normalized = (
        rollout_status
        if isinstance(rollout_status, ProofRolloutStatus)
        else ProofRolloutStatus(raw)
    )
    projected = normalized.to_dict()
    if projected.get("schema") != PROOF_ROLLOUT_STATUS_SCHEMA:
        raise ValueError("unsupported proof rollout status schema")
    result = dict(payload)
    result["proof_rollout"] = projected
    result["proof_rollout_snapshot_id"] = str(
        projected.get("snapshot_id") or ""
    )
    result["proof_policy_id"] = str(projected.get("policy_id") or "")
    result["proof_rollout_mode"] = str(projected.get("rollout_mode") or "")
    result["proof_rollout_blocking"] = bool(projected.get("blocking"))
    # This is intentionally copied from the validated snapshot rather than
    # inferred from provider health.  An outage is diagnostic input and never
    # an authority to weaken an enforcement policy.
    result["proof_provider_health_can_change_mode"] = bool(
        projected.get("provider_health_can_change_mode")
    )
    capabilities = [
        item
        for item in projected.get("capability_health", ())
        if isinstance(item, Mapping)
    ]
    result["proof_capability_healthy"] = bool(capabilities) and all(
        bool(item.get("healthy"))
        for item in capabilities
    )
    result["proof_protected_scope_count"] = len(
        projected.get("protected_scopes") or ()
    )
    result["proof_active_plan_count"] = len(projected.get("active_plans") or ())
    overrides = [
        item
        for item in projected.get("overrides", ())
        if isinstance(item, Mapping)
    ]
    result["proof_override_count"] = len(overrides)
    result["proof_active_override_count"] = sum(
        str(item.get("state") or "") == "active"
        and item.get("applicable_to_policy_mode", True) is True
        for item in overrides
    )
    result["proof_failure_count"] = len(projected.get("failures") or ())
    result["proof_fallback_count"] = len(projected.get("fallbacks") or ())
    result["proof_assurance_counts"] = dict(
        projected.get("assurance_counts") or {}
    )
    return result


def _scan_skip_reason(mode: str) -> ScanTerminalReason:
    """Translate the backlog threshold decision into an explicit terminal reason."""

    if mode == "cooldown":
        return ScanTerminalReason.COOLDOWN
    return ScanTerminalReason.THRESHOLD_SATISFIED


def split_csv_values(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        for raw_item in str(value).split(","):
            item = " ".join(raw_item.strip().split())
            if item and item.lower() not in {"none", "n/a"} and item not in items:
                items.append(item)
    return tuple(items)


@dataclass
class PortalSupervisorConfig:
    todo_path: Path
    state_path: Path
    strategy_path: Path
    events_path: Path
    state_dir: Path
    stale_seconds: float = 1800.0
    check_interval: float = 60.0
    watchdog_startup_grace_seconds: float | None = None
    max_restarts: int = 10
    max_task_attempts: int = 0
    daemon_interval: float = 300.0
    task_prefix: str = TASK_HEADER_PREFIX
    state_prefix: str = "portal"
    database_program: DatabaseProgramConfig | None = None
    reconciliation_only: bool = False
    implement: bool = False
    implementation_command: str = ""
    llm_merge_resolver_command: str = ""
    llm_merge_resolver_timeout_seconds: float | None = None
    implementation_timeout: float = 1800.0
    implementation_max_timeout: float | None = None
    implementation_log_stall_seconds: float = 300.0
    validation_max_workers: int | None = None
    use_ephemeral_worktree: bool = True
    worktree_root: Path | None = None
    merge_target_branch: str = ""
    merge_queue_dir: Path | None = None
    worktree_submodule_paths: tuple[str, ...] = field(default_factory=tuple)
    implementation_protected_paths: tuple[str, ...] = field(default_factory=tuple)
    manual_completion_authority_task_ids: tuple[str, ...] = field(
        default_factory=tuple
    )
    manual_completion_authority_required_task_ids: tuple[str, ...] = field(
        default_factory=tuple
    )
    manual_completion_authority_epoch_id: str = ""
    manual_completion_authority_revalidation_only: bool = False
    # Optional sealed scheduler profile path.  When set, each supervisor
    # pass may run delegated operator completion for seal-gated manuals.
    scheduler_config_path: Path | None = None
    worktree_reconciliation_enabled: bool = True
    worktree_reconciliation_max_merges: int = 1
    worktree_reconciliation_dry_run: bool = False
    worktree_reconciliation_preflight_enabled: bool = True
    worktree_scan_cache_enabled: bool = True
    worktree_scan_cache_ttl_seconds: float = DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS
    worktree_scan_cache_path: Path | None = None
    merge_reconciliation_max_merges: int | None = None
    daemon_merged_worktree_cleanup_max: int | None = None
    task_shard_count: int = 1
    task_shard_index: int = 0
    strict_task_sharding: bool = False
    retry_budget_guardrail_enabled: bool = True
    retry_budget_discovery_dir: Path | None = None
    retry_budget_discovery_output_path: str = ""
    validation_retry_budget: int = 3
    merge_retry_budget: int = 3
    implementation_retry_budget: int = 3
    retry_budget_commit_outputs: bool = False
    retry_budget_commit_subject: str = "Agent: record retry-budget guardrail outputs"
    dependency_guardrail_enabled: bool = True
    dependency_guardrail_discovery_dir: Path | None = None
    dependency_guardrail_discovery_output_path: str = ""
    dependency_guardrail_max_findings: int = 5
    dependency_guardrail_commit_outputs: bool = False
    dependency_guardrail_commit_subject: str = "Agent: record dependency guardrail outputs"
    reconciliation_guardrail_enabled: bool = True
    reconciliation_guardrail_discovery_dir: Path | None = None
    reconciliation_guardrail_discovery_output_path: str = ""
    reconciliation_guardrail_max_findings: int = 3
    reconciliation_guardrail_commit_outputs: bool = False
    reconciliation_guardrail_commit_subject: str = "Agent: record reconciliation guardrail outputs"
    generated_dirty_repair_enabled: bool = False
    generated_dirty_repair_commit_subject: str = "Agent: commit generated supervisor outputs"
    generated_dirty_repair_include_submodule_gitlinks: bool = False
    generated_dirty_repair_max_paths: int = 200
    generated_dirty_repair_stale_lock_seconds: float = 300.0
    generated_dirty_repair_paths: tuple[Path, ...] = field(default_factory=tuple)
    external_reservation_manifest_paths: tuple[Path, ...] = field(default_factory=tuple)
    assumed_completed_task_ids: tuple[str, ...] = field(default_factory=tuple)
    execution_slice_task_ids: tuple[str, ...] = field(default_factory=tuple)
    execution_slice_task_cids: tuple[str, ...] = field(default_factory=tuple)
    plan_bound_dispatch: bool = False
    plan_revision_store_path: Path | None = None
    plan_bound_revision_cid: str = ""
    plan_bound_plan_root_cid: str = ""
    plan_bound_execution_plan_cid: str = ""
    plan_bound_capacity_snapshot_id: str = ""
    plan_bound_slice_manifest_cid: str = ""
    plan_bound_slice_id: str = ""
    plan_bound_lane_id: str = ""
    plan_bound_reassignment_cid: str = ""
    plan_bound_source_head: str = ""
    plan_bound_source_tree: str = ""
    plan_bound_task_source_revision: str = ""
    plan_bound_configuration_root: str = ""
    plan_bound_accepted_tree_root: Path | None = None
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None
    accepted_control_plane_descriptor: int = -1
    codebase_refill_enabled: bool = False
    codebase_scan_discovery_dir: Path | None = None
    codebase_scan_discovery_output_path: str = ""
    codebase_scan_min_open_tasks: int = 0
    codebase_scan_max_findings: int = 5
    codebase_scan_cooldown_seconds: int = 21600
    codebase_refill_timeout_seconds: float = 0.0
    codebase_scan_depends_on: tuple[str, ...] = field(default_factory=tuple)
    codebase_scan_skip_prefixes: tuple[str, ...] = field(default_factory=tuple)
    allow_unscoped_codebase_refill: bool = False
    codebase_defer_when_objective_refills: bool = True
    codebase_scan_commit_outputs: bool = False
    codebase_scan_commit_subject: str = "Agent: record supervisor codebase scan findings"
    objective_refill_enabled: bool = False
    objective_task_janitor_enabled: bool = True
    objective_task_janitor_max_blocked_tasks: int = 50
    objective_task_janitor_max_deprioritized_tasks: int = 50
    objective_task_janitor_max_reopened_goals: int = 12
    objective_task_janitor_mission_terms: tuple[str, ...] = field(default_factory=tuple)
    objective_path: Path | None = None
    objective_graph_path: Path | None = None
    objective_bundle_dir: Path | None = None
    objective_dataset_dir: Path | None = None
    objective_discovery_dir: Path | None = None
    objective_discovery_output_path: str = ""
    objective_summary_prefix: str = ""
    objective_refine_goals: bool = True
    objective_reconcile_goal_completion: bool = True
    objective_goal_completion_todo_boards: tuple[str, ...] = field(default_factory=tuple)
    objective_goal_completion_gate_path: Path | None = None
    objective_goal_completion_evidence_path: Path | None = None
    objective_goal_completion_artifact_refresh_command: str = ""
    objective_goal_completion_artifact_refresh_timeout_seconds: float = 300.0
    objective_goal_migration_enabled: bool = True
    objective_goal_migration_preview: bool = False
    objective_goal_migration_batch_size: int = 100
    objective_seed_interoperability_goals: bool = False
    objective_seed_launch_readiness_goals: bool = False
    objective_interoperability_focus: tuple[str, ...] = field(default_factory=tuple)
    objective_interoperability_component_paths: tuple[str, ...] = field(default_factory=tuple)
    objective_max_interoperability_goals: int = 12
    objective_max_launch_readiness_goals: int = 8
    objective_ensure_tracking_document: bool = False
    objective_ultimate_goal: str = ""
    objective_root_evidence: tuple[str, ...] = field(default_factory=tuple)
    objective_goal_prefix: str | None = None
    objective_root_goal_id: str | None = None
    objective_root_goal_title: str = ""
    objective_tracking_document_title: str = ""
    objective_scan_min_open_tasks: int = 0
    objective_scan_max_findings: int = 5
    objective_scan_cooldown_seconds: int = 21600
    objective_scan_exclude_paths: tuple[str, ...] = field(default_factory=tuple)
    objective_refill_timeout_seconds: float = 0.0
    objective_scan_depends_on: tuple[str, ...] = field(default_factory=tuple)
    objective_max_refinement_children: int = 3
    objective_max_refinement_depth: int = 4
    objective_persist_ast_dataset: bool = True
    objective_write_todo_vector_index: bool = True
    objective_todo_vector_index_path: Path | None = None
    objective_surplus_findings_per_goal: int = DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL
    objective_surplus_min_terms_per_todo: int = DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO
    repo_root: Path = field(default_factory=Path.cwd)
    daemon_script_path: Path | None = None
    supervisor_script_path: Path | None = None

    def __post_init__(self) -> None:
        if self.plan_bound_dispatch:
            if (
                self.plan_bound_accepted_tree_root is None
                or self.plan_revision_store_path is None
                or self.scheduler_config_path is None
            ):
                raise PlanBoundDispatchError(
                    "plan-bound supervisor config is missing path authority"
                )
            (
                root,
                state_dir,
                store_path,
                scheduler_config,
                todo_path,
            ) = _validated_plan_bound_authority_paths(
                repo_root=self.repo_root,
                accepted_tree_root=self.plan_bound_accepted_tree_root,
                state_dir=self.state_dir,
                plan_revision_store_path=self.plan_revision_store_path,
                scheduler_config_path=self.scheduler_config_path,
                todo_path=self.todo_path,
                require_live_module_root=False,
            )
            assert scheduler_config is not None
            assert todo_path is not None
            for field_name in ("state_path", "strategy_path", "events_path"):
                authority_path = _plan_bound_contained_path(
                    root,
                    getattr(self, field_name),
                    field_name=field_name,
                )
                if authority_path.parent != state_dir:
                    raise PlanBoundDispatchError(
                        f"plan-bound {field_name} escapes its lane state directory"
                    )
                setattr(self, field_name, authority_path)
            self.repo_root = root
            self.state_dir = state_dir
            self.plan_revision_store_path = store_path
            self.scheduler_config_path = scheduler_config
            self.todo_path = todo_path
            self.plan_bound_accepted_tree_root = root
            if self.accepted_control_plane_pin is None:
                raise PlanBoundDispatchError(
                    "plan-bound supervisor lacks a sealed accepted control plane"
                )
            try:
                verified_path = verify_agent_implementation_sealed_control_plane(
                    self.accepted_control_plane_pin,
                    self.accepted_control_plane_descriptor,
                )
            except ValueError as exc:
                raise PlanBoundDispatchError(
                    "plan-bound accepted control plane is invalid"
                ) from exc
            if (
                verified_path
                != f"/proc/self/fd/{self.accepted_control_plane_descriptor}"
                or self.accepted_control_plane_pin.source_head
                != self.plan_bound_source_head
                or self.accepted_control_plane_pin.source_tree
                != self.plan_bound_source_tree
            ):
                raise PlanBoundDispatchError(
                    "plan-bound accepted control-plane generation drifted"
                )
        if (
            self.manual_completion_authority_revalidation_only
            and not self.manual_completion_authority_task_ids
        ):
            raise ValueError(
                "manual completion authority revalidation-only mode requires "
                "at least one authority task ID"
            )
        if (
            self.manual_completion_authority_revalidation_only
            and not self.implement
        ):
            raise ValueError(
                "manual completion authority revalidation-only mode requires "
                "implementation execution to be enabled"
            )


class AdoptedManagedDaemonProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if process_is_running(self.pid):
            return None
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        raise RuntimeError(
            "adopted managed daemons must be terminated through the "
            "supervisor ownership fence"
        )

    def kill(self) -> None:
        raise RuntimeError(
            "adopted managed daemons must be killed through the "
            "supervisor ownership fence"
        )

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            polled = self.poll()
            if polled is not None:
                return polled
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired(cmd=["pid", str(self.pid)], timeout=timeout)
            time.sleep(0.2)


class PortalImplementationSupervisor:
    shared_supervisor_loop_class = SupervisorLoop
    shared_supervisor_loop_config_class = SupervisorLoopConfig
    shared_managed_daemon_spec_class = ManagedDaemonSpec
    autonomous_unstall_coordinator_class = AutonomousUnstallCoordinator
    autonomous_unstall_rescue_planner_factory: Any = None
    autonomous_unstall_rescue_orchestrator: Any = None
    autonomous_unstall_rescue_execution_request_factory: Any = None

    def __init__(self, config: PortalSupervisorConfig) -> None:
        self.config = config
        self.restart_count = 0
        self.last_start_at: float | None = None
        self._last_supervisor_maintenance_at: float = 0.0
        self._worktree_worker_phase = ""
        self._last_worktree_worker_seen_monotonic: float | None = None
        self._checkout_mutation_context = threading.local()

    def _autonomous_unstall_state_path(self) -> Path:
        return (
            self.config.state_dir
            / f"{self.config.state_prefix}_autonomous_unstall"
            / "autonomous-unstall-state.json"
        )

    @staticmethod
    def _autonomous_unstall_policy(
        strategy: Mapping[str, Any],
    ) -> AutonomousUnstallPolicy:
        raw = strategy.get("autonomous_unstall_policy")
        if not isinstance(raw, Mapping):
            # Deterministic repair is the production default.  Provider access
            # and rescue execution remain off until an identified operating
            # policy explicitly enables them.
            return AutonomousUnstallPolicy()
        allowed = set(AutonomousUnstallPolicy.__dataclass_fields__)
        values = {key: value for key, value in raw.items() if key in allowed}
        return AutonomousUnstallPolicy(**values)

    def _autonomous_unstall_status(self) -> dict[str, Any]:
        path = self._autonomous_unstall_state_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != AUTONOMOUS_UNSTALL_STATE_SCHEMA
            or not isinstance(payload.get("incidents"), dict)
        ):
            return {}
        incidents = [
            item
            for item in payload["incidents"].values()
            if isinstance(item, Mapping)
        ]
        def updated_at_ms(item: Mapping[str, Any]) -> int:
            value = item.get("updated_at_ms")
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool)
                else 0
            )

        incidents.sort(key=updated_at_ms, reverse=True)
        latest = dict(incidents[0]) if incidents else {}
        result = {
            "schema": AUTONOMOUS_UNSTALL_STATE_SCHEMA,
            "state_path": str(path),
            "incident_count": len(incidents),
            "latest": latest,
            "completion_authority": False,
        }
        runtime = payload.get("rescue_runtime")
        if isinstance(runtime, Mapping):
            result["rescue_runtime"] = {
                key: runtime[key]
                for key in (
                    "circuit_open",
                    "consecutive_failures",
                    "executions",
                    "last_provider_call_ms",
                    "provider_calls",
                    "reason",
                )
                if key in runtime
            }
        repair = payload.get("state_repair")
        if isinstance(repair, Mapping):
            result["state_repair"] = dict(repair)
        return result

    def _supervisor_status_path(self) -> Path:
        return self.config.state_dir / f"{self.config.state_prefix}_supervisor_status.json"

    def _write_signal_shutdown_status(
        self,
        *,
        stop_signal: int,
        cleanup: Mapping[str, Any],
        interrupted_reconciliation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Publish a terminal projection after the owned child tree is fenced.

        The inner ``SupervisorLoop`` normally writes this projection.  A
        process signal raises ``SystemExit`` in the outer loop, so its normal
        return path is bypassed.  Persisting the terminal state here prevents
        an orderly window expiry from looking like a live supervisor with a
        dead PID.  This projection is diagnostic only and grants no task or
        completion authority.
        """

        loop_config = self.build_supervisor_loop_config()
        previous = load_json_dict(self._supervisor_status_path()) or {}
        context = SupervisorStatusContext(
            loop_config.spec,
            static_fields={
                "restart_backoff_seconds": (
                    loop_config.restart_policy.restart_backoff_seconds
                ),
                "fast_restart_backoff_seconds": (
                    loop_config.restart_policy.fast_restart_backoff_seconds
                ),
                "supervisor_heartbeat_seconds": loop_config.heartbeat_seconds,
                "supervisor_poll_seconds": loop_config.poll_seconds,
                "watchdog_stale_after_seconds": (
                    loop_config.watchdog_stale_after_seconds
                ),
                "watchdog_startup_grace_seconds": (
                    loop_config.watchdog_startup_grace_seconds
                ),
                "stop_grace_seconds": loop_config.stop_grace_seconds,
                **dict(loop_config.status_static_fields),
                **dict(loop_config.status_extra_fields),
            },
        )
        return context.write(
            "stopped",
            run_id=str(previous.get("run_id") or ""),
            log_path=str(previous.get("log_path") or ""),
            daemon_pid=None,
            restart_count=int(previous.get("restart_count") or 0),
            last_exit_code=128 + int(stop_signal),
            extra={
                "active_worker_count": 0,
                "active_worker_pids": [],
                "worker_descendant_count": 0,
                "stalled_without_active_worker": False,
                "shutdown_signal": int(stop_signal),
                "shutdown_signal_name": signal.Signals(stop_signal).name,
                "stop_signal": int(stop_signal),
                "last_recycle_reason": "supervisor_signal_shutdown",
                "managed_daemon_cleanup": dict(cleanup),
                "interrupted_implementation_reconciliation": dict(
                    interrupted_reconciliation
                ),
                "daemon_pid_alive": False,
                "supervisor_pid_alive": False,
                "completion_authority": False,
            },
        )

    def _supervisor_maintenance_timeout_seconds(self) -> float:
        return max(
            float(self.config.stale_seconds),
            self._implementation_watchdog_timeout_seconds(),
            float(self.config.check_interval) * 4.0,
            300.0,
        )

    def _implementation_watchdog_timeout_seconds(self) -> float:
        """Return the lane envelope without weakening per-task idle limits."""

        configured = float(self.config.implementation_timeout)
        maximum = self.config.implementation_max_timeout
        if maximum is None:
            return configured
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or float(maximum) <= 0
        ):
            raise ValueError(
                "implementation_max_timeout must be finite and positive"
            )
        return max(configured, float(maximum))

    def _watchdog_startup_grace_seconds(self) -> float:
        configured = self.config.watchdog_startup_grace_seconds
        if configured is not None:
            return max(0.0, float(configured))
        return max(300.0, float(self.config.check_interval) * 2.0)

    def _proof_rollout_status_fields(
        self,
        strategy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load and validate the durable proof-rollout status projection.

        Strategy state is the supervisor-owned configuration/query artifact.
        Returning an empty mapping is reserved for a strategy which has never
        configured proof rollout.  A present but malformed snapshot raises
        instead of silently presenting enforcement work as advisory or absent.
        """

        current = (
            dict(strategy)
            if strategy is not None
            else (load_json_dict(self.config.strategy_path) or {})
        )
        if "proof_rollout" not in current:
            return {}
        return apply_proof_rollout_projection({}, current["proof_rollout"])

    def _refresh_loop_proof_rollout_status(
        self,
        loop: SupervisorLoop | None,
    ) -> None:
        """Refresh rollout diagnostics used by ordinary loop heartbeats."""

        loop_config = getattr(loop, "config", None)
        fields = getattr(loop_config, "status_extra_fields", None)
        if not isinstance(fields, dict):
            # ``build_supervisor_loop_config`` always supplies a mutable dict,
            # but test hooks and custom loop implementations may not.
            return
        projected = self._proof_rollout_status_fields()
        for key in PROOF_ROLLOUT_PROJECTION_FIELDS:
            fields.pop(key, None)
        fields.update(projected)
        autonomous_unstall = self._autonomous_unstall_status()
        if autonomous_unstall:
            fields["autonomous_unstall"] = autonomous_unstall
        else:
            fields.pop("autonomous_unstall", None)

    def _write_supervisor_maintenance_status(
        self,
        phase: str,
        *,
        status: str,
        started_at: str,
        error: str = "",
        daemon_pid: int | None = None,
    ) -> None:
        """Refresh supervisor status while recovery/refill work is running."""

        status_path = self._supervisor_status_path()
        payload = load_json_dict(status_path) or {}
        now = utc_now()
        timeout_seconds = self._supervisor_maintenance_timeout_seconds()
        active = status == "running"
        daemon_alive = bool(daemon_pid and process_is_running(int(daemon_pid)))
        payload.update(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor.supervisor",
                "status": "agentic_maintenance_started" if active else f"agentic_maintenance_{status}",
                "updated_at": now,
                "supervisor_pid": os.getpid(),
                "supervisor_pid_alive": True,
                "daemon_pid": int(daemon_pid) if daemon_pid else None,
                "daemon_pid_alive": daemon_alive,
                "repo_root": str(self.config.repo_root),
                "current_status_path": str(self.config.state_path),
                "progress_path": str(self.config.state_path),
                "state_path": str(self.config.state_path),
                "child_pid_path": str(self._managed_daemon_pid_path()),
                "supervisor_lock_path": str(
                    self.config.state_dir / f"{self.config.state_prefix}_supervisor.lock"
                ),
                "task_prefix": self.config.task_prefix,
                "state_prefix": self.config.state_prefix,
                "last_agentic_maintenance_status": status,
                "last_agentic_maintenance_phase": phase,
                "last_agentic_maintenance_reason": f"recovery_phase:{phase}",
                "active_agentic_maintenance_started_at": started_at if active else "",
                "active_agentic_maintenance_timeout_seconds": timeout_seconds,
                "active_agentic_maintenance_has_daemon": bool(daemon_pid),
                "agentic_timeout_seconds": timeout_seconds,
                "agentic_stuck_maintenance_timeout_seconds": timeout_seconds,
                "watchdog_stale_after_seconds": float(self.config.stale_seconds),
                "watchdog_startup_grace_seconds": self._watchdog_startup_grace_seconds(),
                "supervisor_heartbeat_seconds": max(0.01, float(self.config.check_interval)),
            }
        )
        if error:
            payload["last_agentic_maintenance_error"] = error[-1000:]
        else:
            payload.pop("last_agentic_maintenance_error", None)

        # Scan state lives in the strategy file and is copied into the status
        # heartbeat as a compact projection.  Full generated items and parser
        # diagnostics remain in the content-addressed receipt artifact.
        strategy = load_json_dict(self.config.strategy_path) or {}
        for key in (
            "latest_attempted_scan",
            "latest_successful_scan",
            "scan_terminal_reason",
            "scan_freshness",
            "scan_health",
            "candidate_funnel",
            "scan_receipts",
            "goal_completion",
            "goal_completion_diagnostics",
            "goal_completion_by_goal_id",
            "goal_lifecycle_state_counts",
            "goal_completion_migration",
        ):
            if key in strategy:
                payload[key] = strategy[key]
        rollout_fields = self._proof_rollout_status_fields(strategy)
        for key in PROOF_ROLLOUT_PROJECTION_FIELDS:
            payload.pop(key, None)
        payload.update(rollout_fields)
        autonomous_unstall = self._autonomous_unstall_status()
        if autonomous_unstall:
            payload["autonomous_unstall"] = autonomous_unstall
            latest_unstall = autonomous_unstall.get("latest")
            if isinstance(latest_unstall, Mapping):
                phase = str(latest_unstall.get("phase") or "")
                if phase in {"quarantined", "rescue_previewed"}:
                    reasons = list(payload.get("backpressure_reasons") or ())
                    if "autonomous_unstall_quarantine" not in reasons:
                        reasons.append("autonomous_unstall_quarantine")
                    payload["backpressure"] = True
                    payload["backpressure_reasons"] = reasons[:256]
        write_json_atomic(status_path, payload)

    def _begin_supervisor_maintenance_heartbeat(self, phase: str, *, daemon_pid: int | None = None):
        """Return phase-update and finish callbacks for long supervisor recovery passes."""

        started_at = utc_now()
        current = {"phase": phase}
        stop_event = threading.Event()
        interval = max(5.0, min(30.0, float(self.config.check_interval) / 2.0))

        def write(status: str = "running", error: str = "") -> None:
            try:
                self._write_supervisor_maintenance_status(
                    current["phase"],
                    status=status,
                    started_at=started_at,
                    error=error,
                    daemon_pid=daemon_pid,
                )
            except Exception:
                logger.warning("Failed to update supervisor maintenance heartbeat", exc_info=True)

        def heartbeat() -> None:
            while not stop_event.wait(interval):
                write()

        thread = threading.Thread(
            target=heartbeat,
            name=f"{self.config.state_prefix}-supervisor-maintenance-heartbeat",
            daemon=True,
        )
        write()
        thread.start()

        def update(next_phase: str) -> None:
            current["phase"] = next_phase
            write()

        def finish(status: str = "completed", error: str = "") -> None:
            write(status=status, error=error)
            stop_event.set()
            thread.join(timeout=1.0)

        return update, finish

    def run_once(self, *, include_refill: bool = True) -> dict[str, Any]:
        update_maintenance_phase, finish_maintenance = self._begin_supervisor_maintenance_heartbeat(
            "run_once"
        )
        failed = False
        try:
            return self._run_once_with_maintenance(
                update_maintenance_phase,
                include_refill=include_refill,
            )
        except Exception as exc:
            failed = True
            finish_maintenance("failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not failed:
                finish_maintenance("completed")

    def _maybe_reload_scheduler_authority_profile(self) -> dict[str, Any]:
        """Hot-reload seal/epoch authority fields from the scheduler profile.

        The managed daemon is started with a frozen ``--manual-completion-
        authority-epoch-id``.  When delegated completion verifies new seals or
        the profile's protected package changes, the live epoch diverges from
        the child command line and durable receipts stop matching.  Reloading
        the profile and recycling the child keeps autonomous drains unblocked.
        """

        scheduler_path = self.config.scheduler_config_path
        if scheduler_path is None:
            return {"reloaded": False, "reason": "scheduler_config_path_unset"}
        try:
            profile = load_supervisor_scheduler_config(
                scheduler_path,
                repo_root=REPO_ROOT,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed without recycle
            return {
                "reloaded": False,
                "reason": "scheduler_profile_load_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        new_epoch = str(profile.get("manual_completion_authority_epoch_id") or "")
        new_task_ids = tuple(
            str(item)
            for item in (profile.get("manual_completion_authority_task_ids") or ())
        )
        new_required = tuple(
            str(item)
            for item in (
                profile.get("manual_completion_authority_required_task_ids") or ()
            )
        )
        new_protected = tuple(
            str(item) for item in (profile.get("protected_paths") or ())
        )

        before = {
            "epoch_id": self.config.manual_completion_authority_epoch_id,
            "task_ids": self.config.manual_completion_authority_task_ids,
            "required_task_ids": (
                self.config.manual_completion_authority_required_task_ids
            ),
            "protected_paths": self.config.implementation_protected_paths,
        }
        after = {
            "epoch_id": new_epoch,
            "task_ids": new_task_ids,
            "required_task_ids": new_required,
            "protected_paths": new_protected,
        }
        if before == after:
            return {"reloaded": False, "reason": "authority_profile_unchanged"}

        self.config.manual_completion_authority_epoch_id = new_epoch
        self.config.manual_completion_authority_task_ids = new_task_ids
        self.config.manual_completion_authority_required_task_ids = new_required
        self.config.implementation_protected_paths = new_protected
        payload = {
            "reloaded": True,
            "reason": "authority_profile_changed",
            "before": before,
            "after": after,
            "recycle_required": True,
        }
        self._record_event("scheduler_authority_profile_reloaded", payload)
        return payload

    def _maybe_run_delegated_operator_completion(self) -> dict[str, Any]:
        """Complete seal-gated manuals when the scheduler policy allows it.

        Fail-closed and no-op when the scheduler profile is absent, the policy
        is disabled, or no seal-configured pending tasks are eligible.

        Also no-ops while an implementation protected-path fence is active or
        latched: delegated completion mutates the scheduler pin and taskboard,
        which are themselves protected paths.  Running under an active fence
        latches a false-positive incident and freezes the lane.
        """

        scheduler_path = self.config.scheduler_config_path
        if scheduler_path is None:
            return {"attempted": False, "reason": "scheduler_config_path_unset"}
        try:
            from ..control.delegated_operator_completion import (
                DelegatedOperatorCompletionPolicy,
                complete_ready_sealed_manual_tasks,
            )

            # Quiet fence probe: do not emit maintenance-blocked events for a
            # deferred delegated-completion attempt.
            implementation_state_dir = self.config.state_path.parent
            fence_active = (
                implementation_state_dir
                / IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME
            ).exists()
            fence_incident = (
                implementation_state_dir / IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME
            ).exists()
            if fence_active or fence_incident:
                return {
                    "attempted": False,
                    "reason": "implementation_protected_path_fence_active",
                    "active_snapshot_exists": fence_active,
                    "incident_exists": fence_incident,
                }

            profile = load_supervisor_scheduler_config(
                scheduler_path,
                repo_root=REPO_ROOT,
            )
            policy = DelegatedOperatorCompletionPolicy.from_mapping(
                profile.get("delegated_operator_completion")
            )
            if not policy.enabled:
                return {"attempted": False, "reason": "policy_disabled"}

            todo_path = self.config.todo_path
            text = todo_path.read_text(encoding="utf-8")
            import re as _re

            seal_configs = profile.get("manual_completion_seals") or {}
            seal_task_ids = set(seal_configs)
            board_tasks: list[dict[str, Any]] = []
            for block in _re.split(r"(?=^## )", text, flags=_re.M):
                header = _re.match(r"^## (\S+)", block)
                if header is None:
                    continue
                task_id = header.group(1)
                # Track seal-gated manuals plus their dependency status.
                status_m = _re.search(r"(?m)^- Status:\s*(\S+)", block)
                depends_m = _re.search(r"(?m)^- Depends on:\s*(.+)$", block)
                validation_m = _re.search(r"(?m)^- Validation:\s*(.+)$", block)
                depends = []
                if depends_m is not None:
                    depends = [
                        part.strip()
                        for part in depends_m.group(1).split(",")
                        if part.strip()
                    ]
                board_tasks.append(
                    {
                        "task_id": task_id,
                        "status": (
                            status_m.group(1) if status_m is not None else "pending"
                        ),
                        "depends_on": depends,
                        "validation": (
                            validation_m.group(1).strip()
                            if validation_m is not None
                            else ""
                        ),
                    }
                )

            completed = [
                task["task_id"]
                for task in board_tasks
                if task["status"] == "completed"
            ]
            pending = [
                task["task_id"]
                for task in board_tasks
                if task["status"] != "completed" and task["task_id"] in seal_task_ids
            ]
            if not pending:
                return {
                    "attempted": False,
                    "reason": "no_pending_sealed_manual_tasks",
                    "completed_task_ids": completed,
                }
            result = complete_ready_sealed_manual_tasks(
                repo_root=REPO_ROOT,
                todo_path=todo_path,
                scheduler_path=Path(scheduler_path),
                board_namespace=str(profile.get("board_namespace") or ""),
                seal_configs=seal_configs,
                validation_commands={
                    task["task_id"]: task["validation"] for task in board_tasks
                },
                completed_task_ids=completed,
                pending_task_ids=pending,
                depends_on={
                    task["task_id"]: task["depends_on"] for task in board_tasks
                },
                policy=policy,
            )
            self._record_event(
                "delegated_operator_completion_pass",
                {
                    "completed_count": len(result.get("completed") or []),
                    "attempted": list(result.get("attempted") or []),
                    "error_count": len(result.get("errors") or []),
                },
            )
            return result
        except Exception as exc:
            payload = {
                "attempted": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._record_event("delegated_operator_completion_failed", payload)
            return payload

    def _implementation_protected_maintenance_guard(self) -> dict[str, Any]:
        """Block supervisor mutations while an agent fence is active/latched."""

        implementation_state_dir = self.config.state_path.parent
        active_path = (
            implementation_state_dir
            / IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME
        )
        incident_path = (
            implementation_state_dir
            / IMPLEMENTATION_PROTECTED_INCIDENT_FILENAME
        )
        active_exists = active_path.exists()
        incident_exists = incident_path.exists()
        if not active_exists and not incident_exists:
            return {"blocked": False, "reason": "no_protected_path_guard"}
        payload = {
            "blocked": True,
            "reason": (
                "implementation_protected_path_incident_latched"
                if incident_exists
                else "implementation_protected_path_attempt_active"
            ),
            "active_snapshot_path": str(active_path),
            "active_snapshot_exists": active_exists,
            "incident_path": str(incident_path),
            "incident_exists": incident_exists,
        }
        self._record_event(
            "supervisor_maintenance_protected_path_blocked",
            payload,
        )
        return payload

    def _quarantine_autonomous_unstall_scope(
        self,
        target_ids: Sequence[str],
        incident_cid: str,
        reason: str,
    ) -> dict[str, Any]:
        """Fence only the affected task while leaving other ready work usable."""

        state = PortalTaskState.load(self.config.state_path)
        active_task_id = state.active_task_id.strip()
        normalized_targets = {
            str(item).strip() for item in target_ids if str(item).strip()
        }
        strategy = self._load_strategy()
        existing_quarantine = next(
            (
                dict(item)
                for item in strategy.get(
                    "autonomous_unstall_quarantines", ()
                )
                if isinstance(item, Mapping)
                and item.get("incident_cid") == incident_cid
            ),
            None,
        )
        explicit_task_targets = {
            item.removeprefix("task:")
            for item in normalized_targets
            if item.startswith("task:")
        }
        lane_scoped = any(
            item == "lane:implementation"
            or item.startswith("lane:implementation:")
            for item in normalized_targets
        )
        exact_task = active_task_id if (
            active_task_id
            and (
                active_task_id in normalized_targets
                or active_task_id in explicit_task_targets
                or lane_scoped
                or (
                    existing_quarantine is not None
                    and existing_quarantine.get("task_id") == active_task_id
                )
            )
        ) else ""
        if existing_quarantine is not None and not exact_task:
            return {
                "scope": "task",
                "task_id": str(existing_quarantine.get("task_id") or ""),
                "target_ids": list(
                    existing_quarantine.get("target_ids") or ()
                ),
                "incident_cid": incident_cid,
                "reason": str(existing_quarantine.get("reason") or reason),
                "attempt_recovery": {
                    "consumed": False,
                    "reason": "incident_already_quarantined",
                },
                "deduplicated": True,
                "independent_work_preserved": True,
                "completion_authority": False,
            }
        blocked_tasks = [
            str(item)
            for item in strategy.get("blocked_tasks", ())
            if str(item).strip()
        ]
        if exact_task and exact_task not in blocked_tasks:
            blocked_tasks.append(exact_task)
        quarantines = [
            dict(item)
            for item in strategy.get("autonomous_unstall_quarantines", ())
            if isinstance(item, Mapping)
            and item.get("incident_cid") != incident_cid
        ]
        if existing_quarantine is not None:
            quarantines.append(existing_quarantine)
        else:
            quarantines.append(
                {
                    "incident_cid": incident_cid,
                    "target_ids": sorted(normalized_targets),
                    "task_id": exact_task,
                    "reason": reason,
                    "quarantined_at": utc_now(),
                }
            )
        strategy.update(
            {
                "blocked_tasks": blocked_tasks,
                "autonomous_unstall_quarantines": quarantines[-128:],
                "last_rewrite_at": utc_now(),
                "last_rewrite_reason": (
                    f"autonomous unstall quarantine: {reason}"
                ),
            }
        )
        write_json_atomic(self.config.strategy_path, strategy)

        attempt_recovery: dict[str, Any] = {
            "consumed": False,
            "reason": "no_exact_active_task",
        }
        if exact_task:
            if state.implementation_in_progress:
                attempt_recovery = consume_stale_active_attempt(state)
            state.active_task_id = ""
            state.active_task_key = ""
            state.active_task_cid = ""
            state.active_task_title = ""
            state.active_task_track = ""
            state.active_task_started_at = ""
            state.active_attempt = 0
            state.active_phase = ""
            state.active_phase_started_at = ""
            state.active_phase_detail = ""
            state.active_log_path = ""
            state.active_worktree_path = ""
            state.active_branch = ""
            state.implementation_in_progress = False
            state.recommended_task_id = ""
            state.recommended_actions = []
            state.heartbeat_at = utc_now()
            state.last_progress_at = state.heartbeat_at
            state.save(self.config.state_path)
        result = {
            "scope": "task",
            "task_id": exact_task,
            "target_ids": sorted(normalized_targets),
            "incident_cid": incident_cid,
            "reason": reason,
            "attempt_recovery": attempt_recovery,
            "deduplicated": existing_quarantine is not None,
            "independent_work_preserved": True,
            "completion_authority": False,
        }
        self._record_event("autonomous_unstall_scope_quarantined", result)
        return result

    @staticmethod
    def _autonomous_unstall_evidence(
        state: PortalTaskState,
        reason: str,
    ) -> dict[str, Any]:
        target = state.active_task_id or "lane:implementation"
        common = {"task_id": target, "lane_id": "lane:implementation"}
        lowered = reason.lower()
        if "merge" in lowered:
            return {
                "task": {**common, "failed": True},
                "merge": {
                    **common,
                    "status": "failed",
                    "reason": reason[:1000],
                },
            }
        if "validation" in lowered:
            return {
                "task": {**common, "failed": True},
                "validation": {
                    **common,
                    "status": "failed",
                    "reason": reason[:1000],
                },
            }
        if "dirty" in lowered and "worktree" in lowered:
            return {
                "worktree": {
                    **common,
                    "dirty": True,
                    "worktree_id": state.active_worktree_path or target,
                }
            }
        if "attempt" in lowered and (
            "consumed" in lowered or "stale" in lowered
        ):
            return {
                "attempt": {
                    **common,
                    "attempt_id": f"{target}:{state.active_attempt}",
                    "consumed": True,
                }
            }
        return {
            "task": {**common, "failed": True},
            "heartbeat": {
                **common,
                "stale": True,
                "reason": reason[:1000],
            },
        }

    def _run_autonomous_unstall(
        self,
        state: PortalTaskState,
        reason: str,
    ) -> dict[str, Any]:
        strategy = self._load_strategy()
        affected_task_id = state.active_task_id.strip()
        policy = self._autonomous_unstall_policy(strategy)
        policy_config = strategy.get("autonomous_unstall_policy")
        policy_mapping = (
            dict(policy_config) if isinstance(policy_config, Mapping) else {}
        )
        identity = {
            "state_prefix": self.config.state_prefix,
            "state_dir": str(self.config.state_dir.resolve()),
        }
        def current_roots(
            current_strategy: Mapping[str, Any],
        ) -> dict[str, str]:
            current_policy = current_strategy.get(
                "autonomous_unstall_policy"
            )
            current_policy_mapping = (
                dict(current_policy)
                if isinstance(current_policy, Mapping)
                else {}
            )
            return {
                "repository_root_cid": str(
                    current_strategy.get("repository_root_cid")
                    or prompt_workflow_cid(
                        {
                            "implementation-supervisor-repository": str(
                                self.config.repo_root.resolve()
                            )
                        }
                    )
                ),
                "policy_root": str(
                    current_strategy.get("policy_root")
                    or prompt_workflow_cid(
                        {
                            "autonomous-unstall-policy": (
                                current_policy_mapping
                                or {"deterministic_only": True}
                            )
                        }
                    )
                ),
                "run_cid": str(
                    current_strategy.get("run_cid")
                    or prompt_workflow_cid(
                        {"implementation-supervisor-run": identity}
                    )
                ),
            }

        roots = current_roots(strategy)

        def probe_roots() -> Mapping[str, str]:
            current = load_json_dict(self.config.strategy_path)
            if not isinstance(current, Mapping):
                return {}
            return current_roots(current)
        action_details: dict[str, Any] = {}

        def health() -> Mapping[str, Any]:
            current = PortalTaskState.load(self.config.state_path)
            isolated = bool(
                affected_task_id
                and current.active_task_id != affected_task_id
                and not current.implementation_in_progress
            )
            return {
                "healthy": isolated,
                "status": "healthy" if isolated else "degraded",
                "affected_task_isolated": isolated,
                "active_task_id": current.active_task_id,
                "ready_count": current.ready_count,
                "work_complete": False,
                "completion_authority": False,
            }

        def retry(context: Any) -> Mapping[str, Any]:
            self.rewrite_strategy(state, reason)
            repair = self.repair_blocked_progress_state(
                state, reason, now_ts=time.time()
            )
            action_details["state_repair"] = dict(repair)
            return {
                "succeeded": bool(repair.get("repaired")),
                "observed_effects": (
                    context.action.expected_effects
                    if repair.get("repaired")
                    else ()
                ),
                "reason": str(
                    repair.get("reason") or "task_retry_not_applied"
                ),
            }

        def quarantine(context: Any) -> Mapping[str, Any]:
            result = self._quarantine_autonomous_unstall_scope(
                context.incident.target_ids,
                context.incident.incident_cid,
                "deterministic_recovery_selected_quarantine",
            )
            action_details["state_repair"] = {
                "repaired": True,
                "reason": "affected_scope_quarantined",
                "quarantined": True,
                "active_task_id": affected_task_id,
                **dict(result),
            }
            return {
                "succeeded": True,
                "observed_effects": context.action.expected_effects,
                "reason": str(result.get("reason") or "scope_quarantined"),
            }

        planner = None
        planner_factory = self.autonomous_unstall_rescue_planner_factory
        if (
            policy.rescue_preview_enabled
            and policy.allow_provider_calls
        ):
            if callable(planner_factory):
                planner = planner_factory(policy_mapping)
            else:
                provider = str(policy_mapping.get("provider") or "llm_router")
                model = str(
                    policy_mapping.get("model")
                    or RescuePlannerPolicy().model
                )
                planner = RescuePlanner(
                    RescuePlannerPolicy.permit(
                        provider=provider,
                        model=model,
                        cooldown_ms=max(policy.cooldown_ms, 1),
                    )
                )

        def rescue_request(
            diagnosis: Any,
            exhaustion: Any,
            current_roots: Mapping[str, str],
        ) -> RescuePlanningRequest:
            return RescuePlanningRequest(
                incident=diagnosis.incident,
                exhaustion_receipt=exhaustion,
                diagnostics={
                    "incident_kind": diagnosis.kind.value,
                    "reason_codes": list(diagnosis.reason_codes),
                    "health": dict(diagnosis.health),
                },
                evidence_redacted=True,
                current_repository_root_cid=current_roots[
                    "repository_root_cid"
                ],
                current_run_cid=current_roots["run_cid"],
                current_policy_root=current_roots["policy_root"],
                evidence_reference_cids=diagnosis.incident.evidence_cids,
                now_ms=int(time.time() * 1000),
            )

        coordinator = self.autonomous_unstall_coordinator_class(
            state_dir=self._autonomous_unstall_state_path().parent,
            repository_root=self.config.repo_root,
            repository_root_cid=roots["repository_root_cid"],
            policy_root=roots["policy_root"],
            run_cid=roots["run_cid"],
            policy=policy,
            recovery_handlers={
                RescueOperation.RETRY: retry,
                RescueOperation.QUARANTINE: quarantine,
            },
            health_probe=health,
            root_probe=probe_roots,
            quarantine_scope=self._quarantine_autonomous_unstall_scope,
            event_publisher=lambda event_type, payload: self._record_event(
                event_type, dict(payload)
            ),
            rescue_planner=planner,
            rescue_request_factory=rescue_request if planner is not None else None,
            rescue_orchestrator=self.autonomous_unstall_rescue_orchestrator,
            rescue_execution_request_factory=(
                self.autonomous_unstall_rescue_execution_request_factory
            ),
        )
        result = coordinator.unstall(
            evidence=self._autonomous_unstall_evidence(state, reason)
        )
        if "state_repair" in action_details:
            result["state_repair"] = action_details["state_repair"]
        self._record_event("autonomous_unstall_result", result)
        return result

    def _implementation_maintenance_lock_path(self) -> Path:
        return self.config.state_path.parent / "implementation.lock"

    def _protected_path_maintenance_lock_path(self) -> Path:
        return checkout_mutation_lock_path(
            self.config.repo_root,
            lock_name=PROTECTED_PATH_MAINTENANCE_LOCK_NAME,
        )

    def _protected_path_maintenance_lease_metadata(self) -> dict[str, Any]:
        metadata = self._implementation_maintenance_lease_metadata()
        metadata["kind"] = "implementation-protected-maintenance"
        metadata["lease_role"] = "shared_protected_path_maintenance"
        return metadata

    def _protected_path_maintenance_owner_is_active(
        self,
        metadata: Mapping[str, Any],
    ) -> bool:
        return checkout_lock_owner_is_active(
            dict(metadata),
            expected_kind="implementation-protected-maintenance",
            expected_repo_root=self.config.repo_root,
            process_command_line=process_command_line,
            process_is_running=process_is_running,
        )

    def _active_implementation_task_claims_for_maintenance(
        self,
    ) -> list[dict[str, Any]]:
        claim_dir = checkout_mutation_lock_path(
            self.config.repo_root,
            lock_name=IMPLEMENTATION_TASK_CLAIM_LOCK_DIRNAME,
        )
        try:
            claim_paths = sorted(claim_dir.glob("*.lock"))
        except OSError:
            return [{"claim_path": str(claim_dir), "reason": "claim_scan_failed"}]
        active: list[dict[str, Any]] = []
        for claim_path in claim_paths:
            if claim_path.name.startswith("."):
                continue
            metadata = load_json_dict(claim_path)
            if metadata is None:
                active.append(
                    {
                        "claim_path": str(claim_path),
                        "reason": "claim_metadata_unreadable",
                    }
                )
                continue
            kind = str(metadata.get("kind") or "")
            repo_root = str(metadata.get("repo_root") or "")
            try:
                same_repository = (
                    not repo_root
                    or Path(repo_root).resolve()
                    == self.config.repo_root.resolve()
                )
                pid = int(metadata.get("pid") or 0)
            except (OSError, TypeError, ValueError):
                same_repository = False
                pid = 0
            protected_fence_paths = (
                implementation_task_claim_protected_fence_paths(metadata)
            )
            owner_live = process_is_running(pid)
            # Task claims may be owned through pytest, systemd, or another
            # wrapper whose argv does not contain the daemon filename. A live
            # PID on a compatible claim is sufficient to keep maintenance out.
            # A crash-surviving snapshot or incident must do the same even
            # after its process exits.
            if (
                (not kind or kind == IMPLEMENTATION_TASK_CLAIM_LOCK_KIND)
                and same_repository
                and (owner_live or protected_fence_paths)
            ):
                active.append(
                    {
                        "claim_path": str(claim_path),
                        "task_id": str(metadata.get("task_id") or ""),
                        "pid": pid,
                        "owner_live": owner_live,
                        "state_dir": str(metadata.get("state_dir") or ""),
                        "protected_fence_paths": list(
                            protected_fence_paths
                        ),
                    }
                )
        return active

    def _acquire_protected_path_maintenance_lease(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        lock_path = self._protected_path_maintenance_lock_path()
        metadata = self._protected_path_maintenance_lease_metadata()
        lease_published = False
        try:
            with serialized_lock_update(lock_path):
                for _ in range(2):
                    if self._publish_implementation_maintenance_lease(
                        lock_path,
                        metadata,
                    ):
                        lease_published = True
                        break
                    existing = load_json_dict(lock_path)
                    if existing is not None and (
                        self._protected_path_maintenance_owner_is_active(
                            existing
                        )
                    ):
                        return None, {
                            "blocked": True,
                            "reason": "protected_path_maintenance_active",
                            "lock_path": str(lock_path),
                            "lock_owner_pid": int(existing.get("pid") or 0),
                            "lock_owner_state_dir": str(
                                existing.get("state_dir") or ""
                            ),
                        }
                    lock_path.unlink(missing_ok=True)
                else:
                    return None, {
                        "blocked": True,
                        "reason": "protected_path_maintenance_unavailable",
                        "lock_path": str(lock_path),
                    }
            active_claims = (
                self._active_implementation_task_claims_for_maintenance()
            )
            if active_claims:
                self._release_protected_path_maintenance_lease(metadata)
                lease_published = False
                return None, {
                    "blocked": True,
                    "reason": "shared_implementation_task_claim_active",
                    "lock_path": str(lock_path),
                    "active_claims": active_claims,
                }
            return metadata, {
                "blocked": False,
                "reason": "protected_path_maintenance_lease_acquired",
                "lock_path": str(lock_path),
                "lease_id": str(metadata["lease_id"]),
            }
        except (OSError, RuntimeError) as exc:
            if lease_published:
                self._release_protected_path_maintenance_lease(metadata)
            return None, {
                "blocked": True,
                "reason": "protected_path_maintenance_coordination_failed",
                "lock_path": str(lock_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _release_protected_path_maintenance_lease(
        self,
        metadata: Mapping[str, Any],
    ) -> None:
        lock_path = self._protected_path_maintenance_lock_path()
        try:
            with serialized_lock_update(lock_path):
                existing = load_json_dict(lock_path)
                if existing is None:
                    return
                if str(existing.get("lease_id") or "") != str(
                    metadata.get("lease_id") or ""
                ):
                    logger.warning(
                        "Refusing to remove shared protected-path lease no "
                        "longer owned by this supervisor pass: %s",
                        lock_path,
                    )
                    return
                lock_path.unlink(missing_ok=True)
        except (OSError, RuntimeError):
            logger.warning(
                "Failed to release shared protected-path maintenance lease %s",
                lock_path,
                exc_info=True,
            )

    def _implementation_maintenance_lease_metadata(self) -> dict[str, Any]:
        lease_seed = (
            f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}:{id(self)}"
        )
        owner_script = Path(sys.argv[0]).name
        owner_stem = Path(owner_script).stem
        command_line = process_command_line(os.getpid())
        if owner_script not in command_line and (
            not owner_stem or owner_stem not in command_line
        ):
            # ``python -m`` entrypoints may expose only the requested module
            # name in procfs while ``sys.argv[0]`` points at ``__main__.py``.
            # An empty marker deliberately falls back to the existing
            # implementation-lock PID check instead of making a live
            # supervisor lease look stale to the managed daemon.
            owner_script = ""
        return {
            "kind": "implementation",
            "lease_role": "supervisor_maintenance",
            "lease_id": sha1(lease_seed.encode("utf-8")).hexdigest(),
            "pid": os.getpid(),
            "owner_script": owner_script,
            "repo_root": str(self.config.repo_root.resolve()),
            "state_dir": str(self.config.state_path.parent.resolve()),
            "state_path": str(self.config.state_path.resolve()),
            "started_at": utc_now(),
        }

    def _implementation_lease_owner_is_active(
        self,
        metadata: Mapping[str, Any],
    ) -> bool:
        kind = str(metadata.get("kind") or "")
        if kind and kind != "implementation":
            return False
        state_dir = str(metadata.get("state_dir") or "")
        if state_dir:
            try:
                if (
                    Path(state_dir).resolve()
                    != self.config.state_path.parent.resolve()
                ):
                    return False
            except OSError:
                return False
        try:
            pid = int(metadata.get("pid") or 0)
        except (TypeError, ValueError):
            return False
        # A live PID in this state directory is sufficient to fail closed.
        # ``owner_script`` remains useful diagnostics, but a wrapper command or
        # unreadable procfs entry must never let maintenance steal a daemon's
        # active implementation lease.
        return process_is_running(pid)

    def _publish_implementation_maintenance_lease(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any],
    ) -> bool:
        """Atomically publish a complete lease without an empty-file window."""

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lease_id = str(metadata.get("lease_id") or "")
        temporary_path = lock_path.with_name(
            f".{lock_path.name}.{lease_id}.tmp"
        )
        data = (
            json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        fd: int | None = None
        try:
            fd = os.open(
                temporary_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(fd, "wb") as stream:
                fd = None
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, lock_path)
            except FileExistsError:
                return False
            return True
        finally:
            if fd is not None:
                os.close(fd)
            temporary_path.unlink(missing_ok=True)

    def _acquire_implementation_maintenance_lease(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        lock_path = self._implementation_maintenance_lock_path()
        metadata = self._implementation_maintenance_lease_metadata()
        try:
            with serialized_lock_update(lock_path):
                return self._acquire_implementation_maintenance_lease_serialized(
                    lock_path,
                    metadata,
                )
        except (OSError, RuntimeError) as exc:
            return None, {
                "blocked": True,
                "reason": "implementation_maintenance_lease_coordination_failed",
                "lock_path": str(lock_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _acquire_implementation_maintenance_lease_serialized(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Acquire the durable lease while its update guard is held."""

        for _ in range(2):
            if self._publish_implementation_maintenance_lease(
                lock_path,
                metadata,
            ):
                return metadata, {
                    "blocked": False,
                    "reason": "implementation_maintenance_lease_acquired",
                    "lock_path": str(lock_path),
                    "lease_id": str(metadata["lease_id"]),
                }
            existing = load_json_dict(lock_path)
            if existing is None:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    return None, {
                        "blocked": True,
                        "reason": "implementation_maintenance_lease_cleanup_failed",
                        "lock_path": str(lock_path),
                    }
                continue
            if self._implementation_lease_owner_is_active(existing):
                return None, {
                    "blocked": True,
                    "reason": "implementation_protected_path_attempt_active",
                    "lock_path": str(lock_path),
                    "lock_owner_pid": int(existing.get("pid") or 0),
                    "lock_owner_task_id": str(existing.get("task_id") or ""),
                    "lock_owner_lease_role": str(
                        existing.get("lease_role") or "implementation_attempt"
                    ),
                }
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return None, {
                    "blocked": True,
                    "reason": "implementation_maintenance_lease_cleanup_failed",
                    "lock_path": str(lock_path),
                }
        return None, {
            "blocked": True,
            "reason": "implementation_maintenance_lease_unavailable",
            "lock_path": str(lock_path),
        }

    def _release_implementation_maintenance_lease(
        self,
        metadata: Mapping[str, Any],
    ) -> None:
        lock_path = self._implementation_maintenance_lock_path()
        try:
            with serialized_lock_update(lock_path):
                self._release_implementation_maintenance_lease_serialized(
                    lock_path,
                    metadata,
                )
        except (OSError, RuntimeError):
            logger.warning(
                "Failed to coordinate release of supervisor implementation "
                "lease %s",
                lock_path,
                exc_info=True,
            )

    def _release_implementation_maintenance_lease_serialized(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        existing = load_json_dict(lock_path)
        if existing is None:
            return
        if str(existing.get("lease_id") or "") != str(
            metadata.get("lease_id") or ""
        ):
            logger.warning(
                "Refusing to remove implementation lease no longer owned by "
                "this supervisor pass: %s",
                lock_path,
            )
            return
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            logger.warning(
                "Failed to remove supervisor implementation lease %s",
                lock_path,
                exc_info=True,
            )

    def _run_once_with_maintenance(
        self,
        update_maintenance_phase,
        *,
        include_refill: bool = True,
    ) -> dict[str, Any]:
        if self.config.manual_completion_authority_revalidation_only:
            update_maintenance_phase(
                "manual_completion_authority_revalidation_only"
            )
            return {
                "stuck": False,
                "maintenance_blocked": False,
                "reason": "manual_completion_authority_revalidation_only",
                "manual_completion_authority_revalidation_only": True,
                "ordinary_provider_dispatch_allowed": False,
            }
        if not self.config.implementation_protected_paths:
            return self._run_once_with_maintenance_under_lease(
                update_maintenance_phase,
                include_refill=include_refill,
                implementation_maintenance_lease=None,
            )
        lease, lease_guard = self._acquire_implementation_maintenance_lease()
        if lease is None:
            return {
                "stuck": False,
                "maintenance_blocked": True,
                "reason": str(lease_guard.get("reason") or ""),
                "protected_path_guard": lease_guard,
            }
        shared_lease: dict[str, Any] | None = None
        try:
            update_maintenance_phase("implementation_maintenance_lease")
            shared_lease, shared_guard = (
                self._acquire_protected_path_maintenance_lease()
            )
            if shared_lease is None:
                return {
                    "stuck": False,
                    "maintenance_blocked": True,
                    "reason": str(shared_guard.get("reason") or ""),
                    "protected_path_guard": shared_guard,
                }
            update_maintenance_phase(
                "shared_protected_path_maintenance_lease"
            )
            return self._run_once_with_maintenance_under_lease(
                update_maintenance_phase,
                include_refill=include_refill,
                implementation_maintenance_lease=lease,
            )
        finally:
            if shared_lease is not None:
                self._release_protected_path_maintenance_lease(shared_lease)
            self._release_implementation_maintenance_lease(lease)

    def _run_once_with_maintenance_under_lease(
        self,
        update_maintenance_phase,
        *,
        include_refill: bool = True,
        implementation_maintenance_lease: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # A producer can retain the checkout lease only when its protected
        # outputs could not be proven clean.  Resolve that state before any
        # other maintenance callback is allowed to mutate repository state.
        update_maintenance_phase("retained_generated_checkout_recovery")
        retained_generated_checkout_recovery = (
            self._recover_retained_generated_checkout_lease()
        )
        if retained_generated_checkout_recovery.get("retained_lease"):
            return {
                "stuck": False,
                "maintenance_blocked": True,
                "reason": "checkout_mutation_protected_recovery_required",
                "retained_generated_checkout_recovery": (
                    retained_generated_checkout_recovery
                ),
            }
        update_maintenance_phase("event_log_repair")
        event_log_repair = self.ensure_event_log_file()
        update_maintenance_phase("state_file_repair")
        state_file_repair = self.ensure_state_file()
        update_maintenance_phase("implementation_protected_path_guard")
        protected_path_guard = self._implementation_protected_maintenance_guard()
        if protected_path_guard.get("blocked", False):
            return {
                "stuck": False,
                "maintenance_blocked": True,
                "reason": str(protected_path_guard.get("reason") or ""),
                "event_log_repair": event_log_repair,
                "state_file_repair": state_file_repair,
                "protected_path_guard": protected_path_guard,
                "retained_generated_checkout_recovery": (
                    retained_generated_checkout_recovery
                ),
            }
        update_maintenance_phase("stale_worktree_detection")
        stale_worktree_detection = self.detect_stale_worktrees()
        update_maintenance_phase("stale_active_state_repair")
        stale_active_state_repair = self.repair_stale_active_execution_state()
        update_maintenance_phase("main_checkout_repair")
        main_checkout_repair = self.repair_main_checkout_merge_state()
        update_maintenance_phase("generated_dirty_repair")
        generated_dirty_repair = self.repair_generated_dirty_checkouts()
        update_maintenance_phase("worktree_reconciliation")
        worktree_reconciliation = self.reconcile_backlogged_worktrees(
            preacquired_implementation_lock=(
                implementation_maintenance_lease
            ),
        )
        update_maintenance_phase("worktree_reconciliation_replay")
        worktree_reconciliation_replay = (
            self.recover_already_merged_reconciliation_candidates(
                preacquired_implementation_lock=(
                    implementation_maintenance_lease
                ),
            )
        )
        update_maintenance_phase("worktree_cleanup")
        worktree_cleanup = self.cleanup_backlogged_worktrees()
        update_maintenance_phase("strategy_state_repair")
        strategy_file_repair = self.ensure_strategy_file()
        todo_board_repair = self.ensure_todo_board_for_refill()
        update_maintenance_phase("objective_goal_migration")
        objective_goal_migration = self.migrate_legacy_objective_goal_completion()
        update_maintenance_phase("objective_task_janitor")
        objective_task_janitor = self.reconcile_objective_task_janitor()
        update_maintenance_phase("reconciliation_guardrails")
        reconciliation_findings = self.record_reconciliation_guardrails(
            worktree_reconciliation,
            worktree_cleanup,
        )
        update_maintenance_phase("guardrail_releases")
        guardrail_releases = self.release_completed_guardrail_blocks(
            reconciliation_result=worktree_reconciliation,
            cleanup_result=worktree_cleanup,
            replay_result=worktree_reconciliation_replay,
        )
        state = PortalTaskState.load(self.config.state_path)
        now_ts = time.time()
        stuck, reason = self.is_stuck(state, now_ts=now_ts)
        if stuck:
            update_maintenance_phase("stuck_recovery")
            try:
                autonomous_unstall = self._run_autonomous_unstall(
                    state, reason
                )
            except Exception as exc:
                logger.warning(
                    "Bounded autonomous unstall failed closed",
                    exc_info=True,
                )
                quarantine = self._quarantine_autonomous_unstall_scope(
                    (state.active_task_id or "lane:implementation",),
                    prompt_workflow_cid(
                        {
                            "autonomous-unstall-failure": type(exc).__name__,
                            "reason": reason,
                            "task_id": state.active_task_id,
                        }
                    ),
                    f"autonomous_unstall_internal_error:{type(exc).__name__}",
                )
                autonomous_unstall = {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "autonomous-unstall-result@1"
                    ),
                    "status": "quarantined",
                    "reason": "autonomous_unstall_internal_error",
                    "recovered": False,
                    "quarantined": True,
                    "quarantine": quarantine,
                    "independent_work_preserved": True,
                    "completion_authority": False,
                    "work_complete": False,
                }
                self._record_event(
                    "autonomous_unstall_failed_closed",
                    {
                        **autonomous_unstall,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
            retry_budget_findings = self.record_retry_budget_guardrails()
            dependency_findings = self.record_dependency_guardrails()
            if autonomous_unstall.get("status") == "disabled":
                strategy = self.rewrite_strategy(state, reason)
                state_repair = self.repair_blocked_progress_state(
                    state, reason, now_ts=now_ts
                )
            else:
                strategy = self._load_strategy()
                projected_repair = autonomous_unstall.get("state_repair")
                state_repair = (
                    dict(projected_repair)
                    if isinstance(projected_repair, Mapping)
                    else {
                        "repaired": bool(
                            autonomous_unstall.get("recovered")
                            or autonomous_unstall.get("quarantined")
                        ),
                        "reason": str(
                            autonomous_unstall.get("reason")
                            or autonomous_unstall.get("status")
                            or "autonomous_unstall_terminal"
                        ),
                        "quarantined": bool(
                            autonomous_unstall.get("quarantined")
                        ),
                    }
                )
                state_repair["completion_authority"] = False
            update_maintenance_phase("post_stuck_generated_dirty_repair")
            post_stuck_generated_dirty_repair = self.repair_generated_dirty_checkouts()
            return {
                "stuck": True,
                "reason": reason,
                "autonomous_unstall": autonomous_unstall,
                "retry_budget_count": len(retry_budget_findings),
                "dependency_guardrail_count": len(dependency_findings),
                "reconciliation_guardrail_count": len(reconciliation_findings),
                "strategy_generation": int(strategy.get("generation", 0)),
                "active_task_id": state.active_task_id,
                "state_repair": state_repair,
                "event_log_repair": event_log_repair,
                "strategy_file_repair": strategy_file_repair,
                "state_file_repair": state_file_repair,
                "stale_active_state_repair": stale_active_state_repair,
                "stale_worktree_detection": stale_worktree_detection,
                "todo_board_repair": todo_board_repair,
                "objective_task_janitor": objective_task_janitor,
                "objective_goal_migration": objective_goal_migration,
                "main_checkout_repair": main_checkout_repair,
                "generated_dirty_repair": generated_dirty_repair,
                "post_stuck_generated_dirty_repair": post_stuck_generated_dirty_repair,
                "worktree_reconciliation": worktree_reconciliation,
                "worktree_reconciliation_replay": (
                    worktree_reconciliation_replay
                ),
                "worktree_cleanup": worktree_cleanup,
                "guardrail_unblock_count": len(guardrail_releases),
                "retained_generated_checkout_recovery": (
                    retained_generated_checkout_recovery
                ),
            }
        update_maintenance_phase("retry_dependency_guardrails")
        retry_budget_findings = self.record_retry_budget_guardrails()
        dependency_findings = self.record_dependency_guardrails()
        if include_refill:
            update_maintenance_phase("objective_refill")
            objective_started_at = datetime.now(timezone.utc)
            try:
                objective_result = self._adapt_legacy_objective_result(
                    self._run_protected_refill_mutation(
                        scan_kind="objective",
                        scan_mode="supervisor_callback",
                        analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                        started_at=objective_started_at,
                        output_paths=self._objective_refill_output_paths(),
                        callback=self.refill_objective_backlog,
                    )
                    if self.config.objective_refill_enabled
                    else self.refill_objective_backlog(),
                    scan_mode="supervisor_callback",
                    started_at=objective_started_at,
                )
            except Exception as exc:
                logger.warning(
                    "Objective backlog refill failed; leaving supervisor alive",
                    exc_info=True,
                )
                objective_result = self._terminal_refill_result(
                    ScanTerminalReason.FAILED,
                    scan_mode="supervisor_callback",
                    analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                    started_at=objective_started_at,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"error_type": type(exc).__name__},
                )
            objective_scan = self._persist_refill_result("objective", objective_result)
            objective_payload = dict(objective_result.metadata)
            objective_generated_count = int(
                objective_payload.get("generated_count")
                or len(objective_payload.get("task_ids") or [])
                or objective_result.generated_count
            )
            objective_refined_goal_count = len(objective_payload.get("refined_goal_ids") or [])
            objective_seeded_goal_count = len(objective_payload.get("seeded_interoperability_goal_ids") or [])
            objective_seeded_launch_goal_count = len(
                objective_payload.get("seeded_launch_readiness_goal_ids") or []
            )
            codebase_deferred_reason = ""
            if (
                self.config.codebase_defer_when_objective_refills
                and self.config.objective_refill_enabled
                and objective_generated_count > 0
            ):
                codebase_findings = []
                codebase_result = self._terminal_refill_result(
                    ScanTerminalReason.DISABLED,
                    scan_mode="deferred",
                    analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                    started_at=datetime.now(timezone.utc),
                    metadata={"deferred_reason": "objective_refill_generated_todos"},
                )
                codebase_deferred_reason = "objective_refill_generated_todos"
            else:
                update_maintenance_phase("codebase_refill")
                codebase_started_at = datetime.now(timezone.utc)
                try:
                    codebase_result = self._adapt_legacy_codebase_result(
                        self._run_protected_refill_mutation(
                            scan_kind="codebase",
                            scan_mode="supervisor_callback",
                            analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                            started_at=codebase_started_at,
                            output_paths=(self.config.todo_path,),
                            callback=self.refill_codebase_backlog,
                        )
                        if self.config.codebase_refill_enabled
                        else self.refill_codebase_backlog(),
                        scan_mode="supervisor_callback",
                        started_at=codebase_started_at,
                    )
                except Exception as exc:
                    logger.warning(
                        "Codebase backlog refill failed; leaving supervisor alive",
                        exc_info=True,
                    )
                    codebase_result = self._terminal_refill_result(
                        ScanTerminalReason.FAILED,
                        scan_mode="supervisor_callback",
                        analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                        started_at=codebase_started_at,
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={"error_type": type(exc).__name__},
                    )
                codebase_findings = list(codebase_result.findings)
            codebase_scan = self._persist_refill_result("codebase", codebase_result)
            mapped_contradictions = self._mapped_finding_contradictions(
                codebase_findings,
                source_receipt=codebase_scan,
                goals=self._objective_goals_for_finding_mapping(),
            )
            if mapped_contradictions:
                update_maintenance_phase("post_refill_goal_contradictions")
                objective_contradiction_reconciliation = self.reconcile_objective_task_janitor(
                    contradictions=mapped_contradictions,
                )
            else:
                objective_contradiction_reconciliation = {
                    "changed": False,
                    "reason": "no_mapped_contradictions",
                }
        else:
            update_maintenance_phase("preflight_refill_deferred")
            objective_payload = {}
            objective_generated_count = 0
            objective_refined_goal_count = 0
            objective_seeded_goal_count = 0
            objective_seeded_launch_goal_count = 0
            codebase_findings = []
            codebase_result = self._terminal_refill_result(
                ScanTerminalReason.DISABLED,
                scan_mode="preflight_deferred",
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=datetime.now(timezone.utc),
                metadata={"deferred_reason": "preflight_refill_deferred_until_daemon_loop"},
            )
            codebase_deferred_reason = "preflight_refill_deferred_until_daemon_loop"
            objective_scan = {}
            codebase_scan = {}
            mapped_contradictions = ()
            objective_contradiction_reconciliation = {
                "changed": False,
                "reason": "refill_deferred",
            }
        update_maintenance_phase("post_refill_generated_dirty_repair")
        post_refill_generated_dirty_repair = self.repair_generated_dirty_checkouts()
        update_maintenance_phase("supervisor_check_event")
        self._record_event(
            "supervisor_check",
            {
                "stuck": False,
                "active_task_id": state.active_task_id,
                "completed_count": state.completed_count,
                "worktree_reconciliation_candidate_count": int(
                    worktree_reconciliation.get("candidate_count") or 0
                ),
                "worktree_reconciliation_processed_count": int(
                    worktree_reconciliation.get("processed_count") or 0
                ),
                "worktree_reconciliation_reconciled_count": int(
                    worktree_reconciliation.get("reconciled_count") or 0
                ),
                "worktree_reconciliation_preflight_blocked_count": int(
                    worktree_reconciliation.get("preflight_blocked_count") or 0
                ),
                "worktree_reconciliation_replay_completed_count": int(
                    worktree_reconciliation_replay.get("completed_count")
                    or 0
                ),
                "worktree_reconciliation_replay_failed_count": int(
                    worktree_reconciliation_replay.get("failed_count")
                    or 0
                ),
                "worktree_reconciliation_replay_deferred_count": int(
                    worktree_reconciliation_replay.get("deferred_count")
                    or 0
                ),
                "stale_worktree_detected_count": int(stale_worktree_detection.get("stale_count") or 0),
                "stale_worktree_remedy_count": int(stale_worktree_detection.get("remedy_count") or 0),
                "worktree_cleanup_removed_count": int(worktree_cleanup.get("removed_count") or 0),
                "worktree_cleanup_dirty_group_count": len(
                    worktree_cleanup.get("dirty_worktree_groups") or {}
                ),
                "retry_budget_count": len(retry_budget_findings),
                "dependency_guardrail_count": len(dependency_findings),
                "reconciliation_guardrail_count": len(reconciliation_findings),
                "guardrail_unblock_count": len(guardrail_releases),
                "objective_refill_count": objective_generated_count,
                "objective_refined_goal_count": objective_refined_goal_count,
                "objective_seeded_interoperability_goal_count": objective_seeded_goal_count,
                "objective_seeded_launch_readiness_goal_count": objective_seeded_launch_goal_count,
                "objective_task_janitor_blocked_count": len(
                    objective_task_janitor.get("blocked_task_ids") or []
                ),
                "objective_task_janitor_deprioritized_count": len(
                    objective_task_janitor.get("deprioritized_task_ids") or []
                ),
                "objective_task_janitor_reopened_goal_count": len(
                    objective_task_janitor.get("reopened_goal_ids") or []
                ),
                "objective_goal_migration_preview": bool(
                    objective_goal_migration.get("preview")
                ),
                "objective_goal_migrated_count": len(
                    objective_goal_migration.get("migrated_goal_ids") or []
                ),
                "objective_goal_migration_remaining_count": len(
                    objective_goal_migration.get("remaining_goal_ids") or []
                ),
                "mapped_contradiction_count": len(mapped_contradictions),
                "contradiction_reopened_goal_count": len(
                    objective_contradiction_reconciliation.get(
                        "contradiction_reopened_goal_ids"
                    )
                    or []
                ),
                "objective_contradiction_reconciliation": (
                    objective_contradiction_reconciliation
                ),
                "codebase_refill_count": codebase_result.generated_count,
                "codebase_deferred_reason": codebase_deferred_reason,
                "objective_scan": objective_scan,
                "codebase_scan": codebase_scan,
                "generated_dirty_repair_committed_count": int(
                    generated_dirty_repair.get("committed_count") or 0
                ),
                "post_refill_generated_dirty_repair_committed_count": int(
                    post_refill_generated_dirty_repair.get("committed_count") or 0
                ),
            },
        )
        update_maintenance_phase("scheduler_authority_profile_reload")
        scheduler_authority_profile_reload = (
            self._maybe_reload_scheduler_authority_profile()
        )
        update_maintenance_phase("delegated_operator_completion")
        delegated_operator_completion = (
            self._maybe_run_delegated_operator_completion()
        )
        return {
            "stuck": False,
            "active_task_id": state.active_task_id,
            "completed_count": state.completed_count,
            "retry_budget_count": len(retry_budget_findings),
            "dependency_guardrail_count": len(dependency_findings),
            "reconciliation_guardrail_count": len(reconciliation_findings),
            "guardrail_unblock_count": len(guardrail_releases),
            "objective_refill_count": objective_generated_count,
            "objective_refined_goal_count": objective_refined_goal_count,
            "objective_seeded_interoperability_goal_count": objective_seeded_goal_count,
            "objective_seeded_launch_readiness_goal_count": objective_seeded_launch_goal_count,
            "objective_task_janitor": objective_task_janitor,
            "objective_goal_migration": objective_goal_migration,
            "mapped_contradiction_count": len(mapped_contradictions),
            "objective_contradiction_reconciliation": objective_contradiction_reconciliation,
            "codebase_refill_count": codebase_result.generated_count,
            "codebase_deferred_reason": codebase_deferred_reason,
            "objective_scan": objective_scan,
            "codebase_scan": codebase_scan,
            "scheduler_authority_profile_reload": (
                scheduler_authority_profile_reload
            ),
            "delegated_operator_completion": delegated_operator_completion,
            "event_log_repair": event_log_repair,
            "strategy_file_repair": strategy_file_repair,
            "state_file_repair": state_file_repair,
            "stale_active_state_repair": stale_active_state_repair,
            "stale_worktree_detection": stale_worktree_detection,
            "todo_board_repair": todo_board_repair,
            "main_checkout_repair": main_checkout_repair,
            "generated_dirty_repair": generated_dirty_repair,
            "post_refill_generated_dirty_repair": post_refill_generated_dirty_repair,
            "worktree_reconciliation": worktree_reconciliation,
            "worktree_reconciliation_replay": (
                worktree_reconciliation_replay
            ),
            "worktree_cleanup": worktree_cleanup,
            "retained_generated_checkout_recovery": (
                retained_generated_checkout_recovery
            ),
        }

    def run_forever(self) -> int:
        """Run continuously and fence the managed daemon on process signals."""

        stop_signal: int | None = None

        def request_stop(signum: int, _frame: object) -> None:
            nonlocal stop_signal
            stop_signal = signum
            raise SystemExit(128 + signum)

        handlers_installed = threading.current_thread() is threading.main_thread()
        previous_term: Any = None
        previous_int: Any = None
        if handlers_installed:
            previous_term = signal.signal(signal.SIGTERM, request_stop)
            previous_int = signal.signal(signal.SIGINT, request_stop)
        try:
            return self._run_forever_loop()
        finally:
            if stop_signal is not None:
                cleanup = self._terminate_managed_daemon_tree()
                interrupted_reconciliation = (
                    self._reconcile_interrupted_implementation_after_shutdown()
                )
                try:
                    self._record_event(
                        "supervisor_signal_shutdown",
                        {
                            "signal": stop_signal,
                            "managed_daemon_cleanup": cleanup,
                            "interrupted_implementation_reconciliation": (
                                interrupted_reconciliation
                            ),
                        },
                    )
                except OSError:
                    logger.exception("Could not record supervisor signal shutdown")
                try:
                    self._write_signal_shutdown_status(
                        stop_signal=stop_signal,
                        cleanup=cleanup,
                        interrupted_reconciliation=interrupted_reconciliation,
                    )
                except Exception:
                    logger.exception(
                        "Could not write terminal supervisor signal status"
                    )
            if handlers_installed:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)

    def _run_forever_loop(self) -> int:
        self.ensure_event_log_file()
        if (
            self.config.plan_bound_dispatch
            and not self.config.execution_slice_task_ids
            and not self.config.execution_slice_task_cids
        ):
            self._record_event(
                "plan_bound_empty_slice",
                {"daemon_started": False},
            )
            return 0
        if self.config.plan_bound_dispatch:
            self._validated_plan_bound_slice()
        managed_daemon_guard = self.ensure_managed_daemon_pid_file()
        if managed_daemon_guard.get("blocked", False):
            self._record_event(
                "managed_daemon_start_blocked",
                managed_daemon_guard,
            )
            raise RuntimeError(
                str(
                    managed_daemon_guard.get("reason")
                    or "managed_daemon_ownership_unproven"
                )
            )
        if self.config.plan_bound_dispatch:
            # The canonical plan and task source were re-observed above.  Do
            # not run broad supervisor maintenance between that fence and the
            # exact daemon slice.
            self._record_event(
                "plan_bound_preflight_pass",
                {
                    "revision_cid": self.config.plan_bound_revision_cid,
                    "slice_id": self.config.plan_bound_slice_id,
                },
            )
        else:
            try:
                preflight = self.run_once(include_refill=False)
            except Exception as exc:
                self._record_event(
                    "supervisor_preflight_maintenance_failed",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            self._record_event("supervisor_preflight_maintenance_pass", preflight)
        self._last_supervisor_maintenance_at = time.monotonic()
        while True:
            loop = self.shared_supervisor_loop_class(
                self.build_supervisor_loop_config(),
                watchdog_hook=self._supervisor_loop_watchdog_decision,
            )
            result = loop.run()
            self.restart_count = result.restart_count
            self._worktree_worker_phase = ""
            self._last_worktree_worker_seen_monotonic = None
            result_payload = {
                "status": result.status,
                "restart_count": result.restart_count,
                "last_exit_code": result.last_exit_code,
                "last_recycle_reason": result.last_recycle_reason,
                "last_run_id": result.last_run_id,
                "last_log_path": result.last_log_path,
            }
            self._record_event("supervisor_loop_finished", result_payload)
            if self.config.plan_bound_dispatch:
                exit_code = result.last_exit_code
                if exit_code is None:
                    return 0
                if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                    return 1
                return exit_code if 0 <= exit_code <= 255 else 1
            if result.status not in RECOVERABLE_SUPERVISOR_LOOP_STATUSES:
                return 0

            try:
                recovery = self.run_once()
            except Exception as exc:
                recovery = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                logger.warning("Supervisor recovery pass failed; restarting child loop anyway", exc_info=True)
                self._record_event(
                    "supervisor_loop_recovery_failed",
                    {
                        "loop_result": result_payload,
                        "recovery": recovery,
                    },
                )
            else:
                self._record_event(
                    "supervisor_loop_recovery_pass",
                    {
                        "loop_result": result_payload,
                        "recovery": recovery,
                    },
                )

            delay_seconds = self._supervisor_loop_recovery_delay_seconds()
            self._record_event(
                "supervisor_loop_restarting_after_recovery",
                {
                    "loop_result": result_payload,
                    "delay_seconds": delay_seconds,
                },
            )
            time.sleep(delay_seconds)

    def _supervisor_loop_recovery_delay_seconds(self) -> float:
        """Back off between outer loop recovery attempts without exceeding one check interval."""

        return max(5.0, min(float(self.config.check_interval), 60.0))

    def build_supervisor_loop_config(self) -> SupervisorLoopConfig:
        command = tuple(self._build_daemon_command())
        prefix = self.config.state_prefix
        child_env = _managed_daemon_child_environment(
            database_program=self.config.database_program,
        )
        child_env.update(
            {
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(
                    self._managed_daemon_identity_path()
                ),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                    self._managed_daemon_owner_scope(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        proof_rollout_status_fields = self._proof_rollout_status_fields()
        autonomous_unstall_status = self._autonomous_unstall_status()
        if autonomous_unstall_status:
            proof_rollout_status_fields["autonomous_unstall"] = (
                autonomous_unstall_status
            )
        # The managed daemon blocks while an implementation command is active,
        # so its task-state heartbeat may legitimately remain unchanged for the
        # full command timeout. Let the implementation-aware watchdog below
        # decide whether the live worker or its log has actually stalled.
        watchdog_stale_after_seconds = max(
            0.0,
            float(self.config.stale_seconds),
            self._implementation_watchdog_timeout_seconds()
            + max(30.0, float(self.config.check_interval) * 2.0),
        )
        spec = ManagedDaemonSpec(
            name=f"{prefix}-implementation-daemon",
            schema="ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor",
            repo_root=self.config.repo_root,
            daemon_dir=self.config.state_dir,
            runner=command,
            status_path=self.config.state_path,
            progress_path=self.config.state_path,
            result_log_path=self.config.events_path,
            task_board_path=self.config.todo_path,
            supervisor_status_path=self.config.state_dir / f"{prefix}_supervisor_status.json",
            supervisor_pid_path=self.config.state_dir / f"{prefix}_supervisor.pid",
            child_pid_path=self._managed_daemon_pid_path(),
            supervisor_out_path=self.config.state_dir / f"{prefix}_supervisor.out",
            ensure_status_path=self.config.state_dir / f"{prefix}_ensure_status.json",
            ensure_check_path=self.config.state_dir / f"{prefix}_ensure_check.json",
            supervisor_lock_path=self.config.state_dir / f"{prefix}_supervisor.lock",
            latest_log_path=self.config.state_dir / f"{prefix}_managed_daemon.latest.log",
            daemon_process_match_all=command,
            worktree_root=self.config.worktree_root,
        )
        return SupervisorLoopConfig(
            spec=spec,
            command=command,
            child_env=child_env,
            log_prefix=f"{prefix}_implementation_daemon",
            restart_policy=RestartPolicy(
                restart_backoff_seconds=max(0.0, float(self.config.check_interval)),
                fast_restart_backoff_seconds=min(2.0, max(0.0, float(self.config.check_interval))),
            ),
            heartbeat_seconds=max(0.01, float(self.config.check_interval)),
            poll_seconds=min(1.0, max(0.01, float(self.config.check_interval))),
            watchdog_stale_after_seconds=watchdog_stale_after_seconds,
            # Delta-only task state intentionally remains byte-stable during
            # idle observation windows. The managed daemon log is updated by
            # each pass and therefore supplies independent child liveness.
            watchdog_log_heartbeat_fallback=True,
            watchdog_startup_grace_seconds=self._watchdog_startup_grace_seconds(),
            watchdog_quiescent_status_predicate=(
                _projection_is_quiescent_for_heartbeat_fallback
            ),
            watchdog_accept_fresh_child_log=True,
            stop_grace_seconds=15.0,
            max_restarts=(
                0
                if self.config.plan_bound_dispatch
                else max(0, int(self.config.max_restarts))
            ),
            status_static_fields={
                "todo_path": str(self.config.todo_path),
                "state_path": str(self.config.state_path),
                "task_prefix": self.config.task_prefix,
                "state_prefix": self.config.state_prefix,
                "max_task_attempts": max(0, int(self.config.max_task_attempts)),
                "worktree_no_child_stall_seconds": max(
                    0.0,
                    float(self.config.implementation_log_stall_seconds),
                ),
            },
            # The watchdog refreshes this mutable projection from durable
            # strategy state.  SupervisorLoop applies extra fields after its
            # static heartbeat fields, so policy transitions become visible
            # without restarting the managed daemon.
            status_extra_fields=dict(proof_rollout_status_fields),
        )

    def _supervisor_loop_watchdog_decision(
        self,
        _loop: SupervisorLoop,
        _child: Any,
        _current_status: dict[str, Any],
    ) -> SupervisorLoopDecision:
        self._refresh_loop_proof_rollout_status(_loop)
        now_monotonic = time.monotonic()
        min_interval = max(1.0, float(self.config.check_interval))
        if now_monotonic - self._last_supervisor_maintenance_at < min_interval:
            return SupervisorLoopDecision.keep_running()

        state = PortalTaskState.load(self.config.state_path)
        stuck, reason = self.is_stuck(state, now_ts=time.time())
        if state.active_task_id and not stuck:
            return SupervisorLoopDecision.keep_running()
        if (
            not stuck
            and (
                state.selectable_ready_count > 0
                or bool(state.selectable_ready_task_ids)
            )
        ):
            # Give the managed daemon first claim on runnable work.  Without
            # this handoff, the watchdog can win the brief gap after one task
            # finishes, hold the global implementation lease for a long
            # objective-refill scan, and make the daemon skip ready tasks for
            # the duration of that scan.
            return SupervisorLoopDecision.keep_running()

        self._last_supervisor_maintenance_at = now_monotonic
        daemon_pid = int(getattr(_child, "pid", 0) or 0) or None
        update_maintenance_phase, finish_maintenance = self._begin_supervisor_maintenance_heartbeat(
            "watchdog",
            daemon_pid=daemon_pid,
        )
        failed = False
        try:
            result = self._run_once_with_maintenance(update_maintenance_phase)
        except Exception as exc:
            failed = True
            message = f"{type(exc).__name__}: {exc}"
            finish_maintenance("failed", message)
            logger.warning("Supervisor maintenance hook failed; leaving child alive", exc_info=True)
            self._record_event(
                "supervisor_maintenance_failed",
                {
                    "phase": "watchdog",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return SupervisorLoopDecision.keep_running()
        finally:
            self._last_supervisor_maintenance_at = time.monotonic()
            if not failed:
                finish_maintenance("completed")

        main_checkout_repair = dict(result.get("main_checkout_repair") or {})
        if main_checkout_repair.get("repaired"):
            return SupervisorLoopDecision.recycle(
                "main_checkout_merge_state_repaired",
                detail=main_checkout_repair,
            )
        if result.get("stuck"):
            return SupervisorLoopDecision.recycle(
                str(result.get("reason") or "stuck_progress"),
                detail={"active_task_id": result.get("active_task_id") or ""},
            )
        authority_reload = dict(
            result.get("scheduler_authority_profile_reload") or {}
        )
        if authority_reload.get("recycle_required"):
            return SupervisorLoopDecision.recycle(
                "scheduler_authority_profile_changed",
                detail=authority_reload,
            )
        return SupervisorLoopDecision.keep_running()

    def repair_main_checkout_merge_state(self) -> dict[str, Any]:
        """Resolve or abort an interrupted merge in the shared repository checkout."""

        repo_root = self.config.repo_root
        merge_head_query = self._git_merge_head_query(repo_root)
        unmerged_paths_query = self._git_unmerged_paths_query(repo_root)
        merge_head = str(merge_head_query.get("merge_head") or "")
        unmerged_paths = list(
            unmerged_paths_query.get("unmerged_paths") or ()
        )
        if (
            merge_head_query.get("ok")
            and unmerged_paths_query.get("ok")
            and not merge_head
            and not unmerged_paths
        ):
            return {
                "attempted": False,
                "repaired": False,
                "reason": "clean",
                "path": str(repo_root),
            }

        lock_path = self._repo_merge_lock_path()
        lock_metadata = self._supervisor_checkout_lock_metadata(
            operation="repair_main_checkout_merge_state",
        )
        lease, lock_reason, existing_lock = (
            self._acquire_supervisor_checkout_lease(
                lock_path,
                lock_metadata,
            )
        )
        if lease is None:
            result: dict[str, Any] = {
                "attempted": True,
                "repaired": False,
                "path": str(repo_root),
                "merge_in_progress": bool(merge_head),
                "merge_head": merge_head,
                "initial_unmerged_paths": unmerged_paths,
                "initial_merge_head_query": merge_head_query,
                "initial_unmerged_paths_query": unmerged_paths_query,
                "status_short": self._git_status_short(repo_root),
                "reason": f"checkout_mutation_{lock_reason}",
                "lock_path": str(lock_path),
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_task_id"] = str(existing_lock.get("task_id") or "")
                result["lock_owner_branch"] = str(existing_lock.get("branch") or "")
            self._record_event("main_checkout_merge_state_repair_deferred", result)
            return result

        try:
            # The pre-lock observation is admission evidence only.  A peer
            # may finish or change the merge before this supervisor acquires
            # the checkout lease, so all repair decisions must use a fresh
            # state sampled while the checkout is exclusively owned.
            merge_head_query = self._git_merge_head_query(repo_root)
            unmerged_paths_query = self._git_unmerged_paths_query(repo_root)
            if not merge_head_query.get("ok") or not unmerged_paths_query.get(
                "ok"
            ):
                result = {
                    "attempted": True,
                    "repaired": False,
                    "reason": "main_checkout_merge_state_refresh_failed",
                    "path": str(repo_root),
                    "merge_head_query": merge_head_query,
                    "unmerged_paths_query": unmerged_paths_query,
                }
                self._record_event(
                    "main_checkout_merge_state_repair_deferred",
                    result,
                )
                return result
            locked_merge_head = str(merge_head_query.get("merge_head") or "")
            locked_unmerged_paths = list(
                unmerged_paths_query.get("unmerged_paths") or ()
            )
            if not locked_merge_head and not locked_unmerged_paths:
                return {
                    "attempted": False,
                    "repaired": False,
                    "reason": "clean",
                    "path": str(repo_root),
                }
            return self._repair_main_checkout_merge_state_locked(
                repo_root,
                merge_head=locked_merge_head,
                unmerged_paths=locked_unmerged_paths,
            )
        finally:
            self._release_supervisor_checkout_lease(
                lease,
                operation="repair_main_checkout_merge_state",
            )

    def _repair_main_checkout_merge_state_locked(
        self,
        repo_root: Path,
        *,
        merge_head: str,
        unmerged_paths: list[str],
    ) -> dict[str, Any]:
        """Resolve or abort an interrupted merge after acquiring the checkout lock."""

        result: dict[str, Any] = {
            "attempted": True,
            "repaired": False,
            "path": str(repo_root),
            "merge_in_progress": bool(merge_head),
            "merge_head": merge_head,
            "initial_unmerged_paths": unmerged_paths,
            "status_short": self._git_status_short(repo_root),
        }
        if not merge_head:
            result["reason"] = "unmerged_paths_without_merge_head"
            self._record_event("main_checkout_merge_state_repair", result)
            return result

        deterministic_repair = self.repair_generated_main_checkout_conflicts(repo_root)
        if deterministic_repair:
            result["deterministic_conflict_repair"] = deterministic_repair
            if not self._git_unmerged_paths(repo_root):
                commit_result = self._commit_supervisor_resolved_merge(repo_root)
                result["commit_result"] = commit_result
                if commit_result.get("completed") or commit_result.get("reason") == "resolver_committed_merge":
                    result.update(
                        {
                            "repaired": True,
                            "reason": "deterministic_generated_markdown_conflict_repair",
                            "final_unmerged_paths": [],
                            "merge_in_progress_after": bool(self._git_merge_head(repo_root)),
                        }
                    )
                    self._record_event("main_checkout_merge_state_repair", result)
                    return result

        if self.config.llm_merge_resolver_command:
            llm_result = self._invoke_main_checkout_merge_resolver(
                repo_root,
                merge_head=merge_head,
                unmerged_paths=unmerged_paths,
            )
            result["llm_merge_resolver"] = self._compact_resolver_result(llm_result)
            if self._git_merge_head(repo_root):
                commit_result = self._commit_supervisor_resolved_merge(repo_root)
                result["commit_result"] = commit_result
                if commit_result.get("completed") or commit_result.get("reason") == "resolver_committed_merge":
                    result.update(
                        {
                            "repaired": True,
                            "reason": "llm_resolved_merge",
                            "final_unmerged_paths": self._git_unmerged_paths(repo_root),
                            "merge_in_progress_after": bool(self._git_merge_head(repo_root)),
                        }
                    )
                    self._record_event("main_checkout_merge_state_repair", result)
                    return result
            elif not self._git_unmerged_paths(repo_root):
                result.update(
                    {
                        "repaired": True,
                        "reason": "llm_resolver_completed_merge",
                        "final_unmerged_paths": [],
                        "merge_in_progress_after": False,
                    }
                )
                self._record_event("main_checkout_merge_state_repair", result)
                return result

        post_unmerged_paths = self._git_unmerged_paths(repo_root)
        post_merge_head = self._git_merge_head(repo_root)
        if post_merge_head:
            abort_result = self._abort_main_checkout_merge(repo_root)
            result["abort_result"] = abort_result
            result["repaired"] = bool(abort_result.get("aborted"))
            result["reason"] = (
                "merge_aborted_after_resolver_failed"
                if self.config.llm_merge_resolver_command
                else "merge_aborted_without_resolver"
            )
        else:
            result["reason"] = "merge_no_longer_in_progress"
            result["repaired"] = not post_unmerged_paths
        result["final_unmerged_paths"] = self._git_unmerged_paths(repo_root)
        result["merge_in_progress_after"] = bool(self._git_merge_head(repo_root))
        self._record_event("main_checkout_merge_state_repair", result)
        return result

    def repair_stale_active_execution_state(self, *, now_ts: float | None = None) -> dict[str, Any]:
        """Clear dead active execution markers before worktree repair passes."""

        state = PortalTaskState.load(self.config.state_path)
        active_fields = {
            "active_task_id": state.active_task_id,
            "active_task_title": state.active_task_title,
            "active_task_track": state.active_task_track,
            "active_task_started_at": state.active_task_started_at,
            "active_attempt": state.active_attempt,
            "active_phase": state.active_phase,
            "active_phase_started_at": state.active_phase_started_at,
            "active_phase_detail": state.active_phase_detail,
            "active_log_path": state.active_log_path,
            "active_worktree_path": state.active_worktree_path,
            "active_branch": state.active_branch,
            "implementation_in_progress": state.implementation_in_progress,
        }
        if not state.implementation_in_progress or not state.active_worktree_path:
            return {
                "repaired": False,
                "reason": "no_active_worktree_execution_state",
                "active_task_id": state.active_task_id,
            }

        daemon_pid = self._read_managed_daemon_pid()
        if daemon_pid and process_is_running(daemon_pid):
            command_line = process_command_line(daemon_pid)
            if self._managed_daemon_matches_command_line(command_line):
                return {
                    "repaired": False,
                    "reason": "managed_daemon_running",
                    "daemon_pid": daemon_pid,
                    "active_task_id": state.active_task_id,
                }

        process_lines = self._list_process_commands()
        active_worktree = state.active_worktree_path.strip()
        # Validation can leave an MCP compatibility adapter in a task worktree
        # after the implementation runner exits.  Only Codex/Copilot proves
        # that an implementation attempt is still live; a local service must
        # not prevent stale-state recovery indefinitely.
        if active_worktree and any(
            active_worktree in line and IMPLEMENTATION_RUNNER_PROCESS_PATTERN.search(line)
            for line in process_lines
        ):
            return {
                "repaired": False,
                "reason": "active_worktree_process_running",
                "active_worktree_path": active_worktree,
                "active_task_id": state.active_task_id,
            }
        active_branch = state.active_branch.strip()
        if active_branch and any(
            active_branch in line and IMPLEMENTATION_RUNNER_PROCESS_PATTERN.search(line)
            for line in process_lines
        ):
            return {
                "repaired": False,
                "reason": "active_branch_process_running",
                "active_branch": active_branch,
                "active_task_id": state.active_task_id,
            }

        # Preserve stalled branch work before dropping active markers. Without
        # this, a dead provider leaves dirty implementation worktrees that
        # block later merges and require manual rescue.
        rescue_result: dict[str, Any] = {}
        worktree_path = Path(active_worktree) if active_worktree else None
        if worktree_path is not None and worktree_path.exists():
            dirty = self._git_status_short(worktree_path)
            target_ref = self._git_current_branch(self.config.repo_root) or "HEAD"
            if dirty:
                rescue_result = self._rescue_dirty_worktree(
                    worktree_path,
                    branch=active_branch,
                    head=self._git_ref_commit(worktree_path, "HEAD"),
                    target_ref=target_ref,
                    status_lines=dirty,
                    reason="stale_active_execution_auto_rescue",
                )

        repaired_at = utc_now()
        recovered_attempt = consume_stale_active_attempt(state)
        state.active_attempt = 0
        state.active_phase = ""
        state.active_phase_started_at = ""
        state.active_phase_detail = ""
        state.active_log_path = ""
        state.active_worktree_path = ""
        state.active_branch = ""
        state.implementation_in_progress = False
        state.heartbeat_at = repaired_at
        state.last_progress_at = repaired_at
        state.save(self.config.state_path)
        result = {
            "repaired": True,
            "reason": "managed_daemon_process_missing",
            "daemon_pid": daemon_pid or 0,
            "repaired_at": repaired_at,
            "attempt_recovery": recovered_attempt,
            "rescue_result": rescue_result,
            **active_fields,
        }
        self._record_event("stale_active_execution_state_repaired", result)
        return result

    def _repo_merge_lock_path(self) -> Path:
        return checkout_mutation_lock_path(self.config.repo_root)

    def _todo_board_is_implementation_protected(self) -> bool:
        try:
            relative = (
                self.config.todo_path.resolve()
                .relative_to(self.config.repo_root.resolve())
                .as_posix()
            )
        except (OSError, ValueError):
            return False
        return relative in set(self.config.implementation_protected_paths)

    def _generated_board_commit_policy(
        self,
        *,
        configured_commit_outputs: bool,
        configured_subject: str,
    ) -> tuple[bool, str]:
        protected = self._todo_board_is_implementation_protected()
        commit_outputs = bool(configured_commit_outputs or protected)
        subject = (
            generated_protected_board_commit_subject(configured_subject)
            if protected
            else configured_subject
        )
        return commit_outputs, subject

    def _run_generated_board_producer(
        self,
        *,
        producer: str,
        commit_outputs: bool,
        operation: str = "generated_board_update",
        callback,
        deferred_result=None,
    ):
        """Serialize a committed generated-board update with checkout mutations."""

        if not commit_outputs:
            return callback()
        current_lease = self._current_supervisor_checkout_lease()
        if current_lease is not None:
            depth = self._supervisor_checkout_transaction_depth()
            retained = bool(
                getattr(
                    self._checkout_mutation_context,
                    "retain_until_protected_clean",
                    False,
                )
            )
            if depth <= 0:
                retained_producer = str(
                    getattr(
                        self._checkout_mutation_context,
                        "retained_producer",
                        "",
                    )
                    or ""
                )
                recovery_allowed = bool(
                    retained
                    and (
                        operation == "generated_dirty_repair"
                        or (retained_producer and producer == retained_producer)
                    )
                )
                if not recovery_allowed:
                    raise RuntimeError(
                        "checkout_mutation_protected_recovery_required"
                    )
                return self._run_retained_generated_checkout_recovery(
                    current_lease,
                    operation=operation,
                    producer=producer,
                    callback=callback,
                )

            # True nesting is permitted only while the owning transaction is
            # still on this thread's callback stack.  A retained transaction
            # resets depth to zero and cannot admit unrelated producers.
            self._checkout_mutation_context.transaction_depth = depth + 1
            try:
                return callback()
            finally:
                self._checkout_mutation_context.transaction_depth = depth
        lock_path = self._repo_merge_lock_path()
        lock_metadata = self._supervisor_checkout_lock_metadata(
            operation=operation,
            extra={"producer": producer},
        )
        lease, lock_reason, existing_lock = (
            self._acquire_supervisor_checkout_lease(
                lock_path,
                lock_metadata,
            )
        )
        if lease is None:
            payload: dict[str, Any] = {
                "producer": producer,
                "reason": f"checkout_mutation_{lock_reason}",
                "lock_path": str(lock_path),
            }
            if existing_lock:
                payload["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                payload["lock_owner_task_id"] = str(
                    existing_lock.get("task_id") or ""
                )
                payload["lock_owner_branch"] = str(
                    existing_lock.get("branch") or ""
                )
            self._record_event("generated_board_update_deferred", payload)
            return deferred_result(payload) if deferred_result is not None else []

        self._checkout_mutation_context.lease = lease
        self._checkout_mutation_context.retain_until_protected_clean = False
        self._checkout_mutation_context.transaction_depth = 0
        release_guard: dict[str, Any] | None = None
        try:
            release_guard = self._generated_protected_release_guard_snapshot()
            if release_guard:
                release_guard = self._content_addressed_supervisor_release_guard(
                    release_guard
                )
                initial_verdict = (
                    self._safe_generated_protected_release_guard(
                        release_guard
                    )
                )
                dirty_repair_preflight = bool(
                    operation == "generated_dirty_repair"
                    and self._generated_dirty_repair_preflight_allowed(
                        initial_verdict
                    )
                )
                if (
                    not initial_verdict.get("release_allowed")
                    and not dirty_repair_preflight
                ):
                    raise RuntimeError(
                        "protected generated outputs are unsafe before "
                        f"mutation: {initial_verdict.get('reason') or 'unknown'}"
                    )
                journaled_lease = (
                    self._publish_supervisor_protected_recovery_journal(
                        lease,
                        operation=operation,
                        producer=producer,
                        release_guard=release_guard,
                    )
                )
                if journaled_lease is None:
                    raise RuntimeError(
                        "supervisor protected recovery journal publication "
                        "failed"
                    )
                lease = journaled_lease
        except BaseException:
            release_error = (
                self._clear_and_release_supervisor_checkout_lease(
                    lease,
                    operation=operation,
                )
            )
            if release_error:
                self._record_generated_checkout_retention(
                    lease,
                    operation=operation,
                    producer=producer,
                    release_guard=release_guard,
                    release_verdict={
                        "release_allowed": False,
                        "reason": "protected_generated_snapshot_failed",
                        "error": release_error,
                    },
                )
            else:
                self._checkout_mutation_context.transaction_depth = 0
            raise
        self._checkout_mutation_context.generated_protected_release_guard = (
            release_guard
        )
        self._checkout_mutation_context.transaction_depth = 1
        try:
            result = callback()
        except BaseException:
            self._checkout_mutation_context.transaction_depth = 0
            self._finalize_generated_board_lease(
                lease,
                operation=operation,
                producer=producer,
                release_guard=release_guard,
            )
            raise
        self._checkout_mutation_context.transaction_depth = 0
        release_verdict = self._finalize_generated_board_lease(
            lease,
            operation=operation,
            producer=producer,
            release_guard=release_guard,
        )
        if not release_verdict.get("release_allowed"):
            raise RuntimeError(
                "generated-board producer left protected outputs unsafe for "
                f"lease release: {release_verdict.get('reason') or 'unknown'}"
            )
        return result

    def _finalize_generated_board_lease(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
        producer: str,
        release_guard: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Finalize without ever replacing the producer callback exception."""

        release_verdict = self._safe_generated_protected_release_guard(
            release_guard
        )
        retain_requested = bool(
            getattr(
                self._checkout_mutation_context,
                "retain_until_protected_clean",
                False,
            )
        )
        if retain_requested:
            release_verdict = {
                **release_verdict,
                "release_allowed": False,
                "reason": "protected_generated_release_retention_requested",
            }
        if not release_verdict.get("release_allowed"):
            self._record_generated_checkout_retention(
                lease,
                operation=operation,
                producer=producer,
                release_guard=release_guard,
                release_verdict=release_verdict,
            )
            return release_verdict

        release_error = self._clear_and_release_supervisor_checkout_lease(
            lease,
            operation=operation,
        )
        if release_error:
            release_verdict = {
                "release_allowed": False,
                "reason": "checkout_mutation_lease_release_failed",
                "error": release_error,
            }
            self._record_generated_checkout_retention(
                lease,
                operation=operation,
                producer=producer,
                release_guard=release_guard,
                release_verdict=release_verdict,
            )
        return release_verdict

    def _safe_generated_protected_release_guard(
        self,
        release_guard: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            return self._generated_protected_release_guard(release_guard)
        except BaseException as exc:
            return {
                "release_allowed": False,
                "reason": "protected_generated_release_guard_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _generated_dirty_repair_preflight_allowed(
        verdict: Mapping[str, Any],
    ) -> bool:
        """Admit a repair only when existing protected dirt is the sole fault."""

        if verdict.get("release_allowed"):
            return True
        if verdict.get("reason") != "protected_generated_outputs_dirty":
            return False
        scope_results = [
            item
            for item in verdict.get("scope_results", ())
            if isinstance(item, Mapping)
        ]
        failed_scopes = [
            item
            for item in scope_results
            if not item.get("release_allowed")
        ]
        return bool(failed_scopes) and all(
            item.get("reason") == "protected_generated_outputs_dirty"
            for item in failed_scopes
        )

    def _record_generated_checkout_retention(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
        producer: str,
        release_guard: Mapping[str, Any] | None,
        release_verdict: Mapping[str, Any],
    ) -> None:
        self._checkout_mutation_context.transaction_depth = 0
        self._checkout_mutation_context.retain_until_protected_clean = True
        if not str(
            getattr(
                self._checkout_mutation_context,
                "retained_operation",
                "",
            )
            or ""
        ):
            self._checkout_mutation_context.retained_operation = operation
        if not str(
            getattr(
                self._checkout_mutation_context,
                "retained_producer",
                "",
            )
            or ""
        ):
            self._checkout_mutation_context.retained_producer = producer
        self._checkout_mutation_context.generated_protected_release_guard = (
            dict(release_guard or {})
        )
        try:
            self._record_event(
                "checkout_mutation_lease_retained",
                {
                    "operation": operation,
                    "producer": producer,
                    "lock_path": str(lease.lock_path),
                    "lease_id": lease.lease_id,
                    "reason": str(
                        release_verdict.get("reason")
                        or "protected_generated_outputs_remain_dirty"
                    ),
                    "release_guard": dict(release_verdict),
                },
            )
        except BaseException:
            logger.warning(
                "Failed to record retained generated checkout lease %s",
                lease.lock_path,
                exc_info=True,
            )

    def _clear_and_release_supervisor_checkout_lease(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
    ) -> str:
        try:
            released = self._release_supervisor_checkout_lease(
                lease,
                operation=operation,
            )
        except BaseException as exc:
            return f"{type(exc).__name__}: {exc}"
        if not released:
            self._checkout_mutation_context.transaction_depth = 0
            self._checkout_mutation_context.retain_until_protected_clean = True
            return "checkout mutation lease was replaced before release"
        self._checkout_mutation_context.transaction_depth = 0
        self._checkout_mutation_context.retain_until_protected_clean = False
        self._checkout_mutation_context.retained_operation = ""
        self._checkout_mutation_context.retained_producer = ""
        self._checkout_mutation_context.generated_protected_release_guard = None
        self._checkout_mutation_context.lease = None
        return ""

    def _current_supervisor_checkout_lease(
        self,
    ) -> CheckoutMutationLease | None:
        context = getattr(self, "_checkout_mutation_context", None)
        lease = getattr(context, "lease", None)
        return lease if isinstance(lease, CheckoutMutationLease) else None

    def _implementation_protected_output_paths(
        self,
        paths: Sequence[Path | None],
    ) -> tuple[Path, ...]:
        repo_root = self.config.repo_root.resolve()
        protected = set(self.config.implementation_protected_paths)
        matches: list[Path] = []
        for configured_path in paths:
            if configured_path is None:
                continue
            path = Path(configured_path)
            if not path.is_absolute():
                path = repo_root / path
            try:
                relative = path.resolve().relative_to(repo_root).as_posix()
            except (OSError, RuntimeError, ValueError):
                continue
            if relative in protected and path not in matches:
                matches.append(path)
        return tuple(matches)

    def _dirty_implementation_protected_paths(
        self,
        paths: Sequence[Path],
    ) -> tuple[str, ...]:
        repo_root = self.config.repo_root.resolve()
        relative_paths: list[str] = []
        for path in paths:
            candidate = path if path.is_absolute() else repo_root / path
            try:
                relative = candidate.resolve().relative_to(repo_root).as_posix()
            except (OSError, RuntimeError, ValueError):
                continue
            if relative not in relative_paths:
                relative_paths.append(relative)
        if not relative_paths:
            return ()
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *relative_paths,
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            # Fail closed when cleanliness cannot be established.
            return tuple(relative_paths)
        dirty: list[str] = []
        for line in result.stdout.splitlines():
            relative = self._status_line_path(line)
            if relative and relative not in dirty:
                dirty.append(relative)
        return tuple(dirty)

    @staticmethod
    def _content_addressed_supervisor_release_guard(
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        guard = dict(snapshot)
        guard.pop("guard_id", None)
        guard["guard_id"] = content_identity(guard)
        return guard

    def _publish_supervisor_protected_recovery_journal(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
        producer: str,
        release_guard: Mapping[str, Any],
    ) -> CheckoutMutationLease | None:
        """CAS-journal exact recovery authority before protected writes."""

        protected_paths = [
            str(path)
            for path in release_guard.get("protected_paths", ())
            if str(path)
        ]
        journaled_guard = json.loads(
            json.dumps(dict(release_guard), sort_keys=True)
        )
        guard_id = str(release_guard.get("guard_id") or "")
        intent: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "supervisor-protected-recovery-intent@1"
            ),
            "operation": operation,
            "producer": producer,
            "protected_paths": protected_paths,
            "guard_id": guard_id,
        }
        intent["intent_id"] = content_identity(intent)
        updated = update_checkout_mutation_lease(
            lease,
            {
                **dict(lease.metadata),
                "protected_recovery_required": True,
                "protected_recovery_owner": "implementation_supervisor",
                "protected_paths": protected_paths,
                "protected_release_guard": journaled_guard,
                "protected_recovery_intent": intent,
                "protected_recovery_started_at": utc_now(),
            },
        )
        if updated is not None:
            self._checkout_mutation_context.lease = updated
        return updated

    def _generated_protected_release_guard_snapshot(
        self,
    ) -> dict[str, Any]:
        protected_paths = tuple(self.config.implementation_protected_paths)
        if not protected_paths:
            return {}
        scope_paths: dict[Path, set[str]] = {}
        discovery_errors: list[dict[str, str]] = []
        repo_root = self.config.repo_root.resolve()
        for protected_path in protected_paths:
            target = repo_root / protected_path
            containing_root = self._containing_git_root(target)
            if containing_root is None:
                discovery_errors.append(
                    {
                        "path": protected_path,
                        "reason": "containing_git_root_unavailable",
                    }
                )
                continue
            try:
                relative = target.resolve(strict=False).relative_to(
                    containing_root
                ).as_posix()
            except (OSError, RuntimeError, ValueError):
                discovery_errors.append(
                    {
                        "path": protected_path,
                        "reason": "protected_path_outside_containing_git_root",
                    }
                )
                continue
            scope_paths.setdefault(containing_root, set()).add(relative)

            child_root = containing_root
            visited = {child_root}
            while child_root != repo_root:
                parent_root = self._parent_git_root(child_root, repo_root)
                if parent_root is None or parent_root in visited:
                    discovery_errors.append(
                        {
                            "path": protected_path,
                            "reason": "parent_git_root_unavailable",
                            "git_root": str(child_root),
                        }
                    )
                    break
                visited.add(parent_root)
                try:
                    gitlink = child_root.relative_to(parent_root).as_posix()
                except ValueError:
                    discovery_errors.append(
                        {
                            "path": protected_path,
                            "reason": "child_git_root_outside_parent",
                            "git_root": str(child_root),
                            "parent_git_root": str(parent_root),
                        }
                    )
                    break
                scope_paths.setdefault(parent_root, set()).add(gitlink)
                child_root = parent_root

        scopes: list[dict[str, Any]] = []
        for git_root, paths in sorted(
            scope_paths.items(),
            key=lambda item: str(item[0]),
        ):
            head_state = self._git_head_state(git_root)
            scopes.append(
                {
                    "git_root": str(git_root),
                    "paths": sorted(paths),
                    "before_head": str(head_state.get("head") or ""),
                    "before_head_query": head_state,
                }
            )
        return {
            "protected_paths": protected_paths,
            "scopes": scopes,
            "discovery_errors": discovery_errors,
        }

    @staticmethod
    def _git_toplevel(path: Path) -> Path | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            return Path(result.stdout.strip()).resolve()
        except (OSError, RuntimeError):
            return None

    def _containing_git_root(self, target: Path) -> Path | None:
        repo_root = self.config.repo_root.resolve()
        probe = target if target.is_dir() else target.parent
        while not probe.exists() and probe != repo_root:
            parent = probe.parent
            if parent == probe:
                break
            probe = parent
        containing = self._git_toplevel(probe)
        if containing is None:
            return None
        try:
            containing.relative_to(repo_root)
        except ValueError:
            return None
        return containing

    def _parent_git_root(
        self,
        child_root: Path,
        repo_root: Path,
    ) -> Path | None:
        try:
            superproject = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--show-superproject-working-tree",
                ],
                cwd=child_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            superproject = None
        if (
            superproject is not None
            and superproject.returncode == 0
            and superproject.stdout.strip()
        ):
            parent_root = Path(superproject.stdout.strip()).resolve()
        else:
            parent_root = self._git_toplevel(child_root.parent)
        if parent_root is None or parent_root == child_root:
            return None
        try:
            child_root.relative_to(parent_root)
            parent_root.relative_to(repo_root)
        except ValueError:
            return None
        return parent_root

    @staticmethod
    def _git_head_state(git_root: Path) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                cwd=git_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "head": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if result.returncode == 0 and result.stdout.strip():
            return {"ok": True, "head": result.stdout.strip(), "unborn": False}
        try:
            symbolic = subprocess.run(
                ["git", "symbolic-ref", "-q", "HEAD"],
                cwd=git_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "head": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if symbolic.returncode != 0 or not symbolic.stdout.strip():
            return {
                "ok": False,
                "head": "",
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
            }
        try:
            referenced = subprocess.run(
                [
                    "git",
                    "show-ref",
                    "--verify",
                    "--quiet",
                    symbolic.stdout.strip(),
                ],
                cwd=git_root,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "head": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if referenced.returncode == 1:
            return {"ok": True, "head": "", "unborn": True}
        return {
            "ok": False,
            "head": "",
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
        }

    @staticmethod
    def _trusted_generated_protected_commit(
        author_email: str,
        subject: str,
    ) -> bool:
        return bool(
            author_email == BACKLOG_REFINERY_AUTHOR_EMAIL
            and subject.endswith(GENERATED_PROTECTED_BOARD_COMMIT_MARKER)
        )

    def _generated_protected_release_guard(
        self,
        snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Prove protected generated outputs are clean and trusted."""

        if not snapshot:
            return {"release_allowed": True, "reason": "no_protected_paths"}
        discovery_errors = [
            dict(item)
            for item in snapshot.get("discovery_errors", ())
            if isinstance(item, Mapping)
        ]
        if discovery_errors:
            return {
                "release_allowed": False,
                "reason": "protected_generated_scope_discovery_failed",
                "discovery_errors": discovery_errors,
            }
        scopes = [
            dict(item)
            for item in snapshot.get("scopes", ())
            if isinstance(item, Mapping)
        ]
        if not scopes:
            return {"release_allowed": True, "reason": "no_protected_paths"}
        scope_results = [
            self._generated_protected_scope_release_guard(scope)
            for scope in scopes
        ]
        failed_scope = next(
            (
                item
                for item in scope_results
                if not item.get("release_allowed")
            ),
            None,
        )
        if failed_scope is not None:
            return {
                "release_allowed": False,
                "reason": str(
                    failed_scope.get("reason")
                    or "protected_generated_scope_untrusted"
                ),
                "failed_git_root": str(failed_scope.get("git_root") or ""),
                "scope_results": scope_results,
            }
        return {
            "release_allowed": True,
            "reason": (
                "protected_generated_history_trusted"
                if any(item.get("commits") for item in scope_results)
                else "protected_outputs_clean_history_unchanged"
            ),
            "scope_results": scope_results,
        }

    def _generated_protected_scope_release_guard(
        self,
        scope: Mapping[str, Any],
    ) -> dict[str, Any]:
        git_root = Path(str(scope.get("git_root") or "")).resolve()
        paths = tuple(
            str(path).strip()
            for path in scope.get("paths", ())
            if str(path).strip()
        )
        result_base: dict[str, Any] = {
            "git_root": str(git_root),
            "paths": list(paths),
        }
        before_query = scope.get("before_head_query")
        if not isinstance(before_query, Mapping) or not before_query.get("ok"):
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_history_snapshot_failed",
                "before_head_query": dict(before_query or {}),
            }
        dirty_query = self._git_scope_dirty_paths(git_root, paths)
        if not dirty_query.get("ok"):
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_status_query_failed",
                "status_query": dirty_query,
            }
        dirty_paths = list(dirty_query.get("dirty_paths") or ())
        if dirty_paths:
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_outputs_dirty",
                "dirty_paths": dirty_paths,
            }

        before_head = str(scope.get("before_head") or "")
        after_query = self._git_head_state(git_root)
        if not after_query.get("ok"):
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_history_unavailable",
                "before_head": before_head,
                "after_head_query": after_query,
            }
        after_head = str(after_query.get("head") or "")
        commits: list[dict[str, Any]] = []
        if before_head:
            if not after_head:
                return {
                    **result_base,
                    "release_allowed": False,
                    "reason": "protected_generated_history_rewritten",
                    "before_head": before_head,
                    "after_head": after_head,
                }
            if before_head != after_head:
                try:
                    ancestry = subprocess.run(
                        [
                            "git",
                            "merge-base",
                            "--is-ancestor",
                            before_head,
                            after_head,
                        ],
                        cwd=git_root,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                except OSError as exc:
                    return {
                        **result_base,
                        "release_allowed": False,
                        "reason": "protected_generated_history_query_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if ancestry.returncode != 0:
                    return {
                        **result_base,
                        "release_allowed": False,
                        "reason": "protected_generated_history_rewritten",
                        "before_head": before_head,
                        "after_head": after_head,
                    }
                history_result = self._git_protected_history(
                    git_root,
                    f"{before_head}..{after_head}",
                    paths,
                )
                if not history_result.get("ok"):
                    return {
                        **result_base,
                        "release_allowed": False,
                        "reason": "protected_generated_history_query_failed",
                        "history_query": history_result,
                    }
                commits = list(history_result.get("commits") or ())
                if not commits:
                    try:
                        changed = subprocess.run(
                            [
                                "git",
                                "diff",
                                "--quiet",
                                before_head,
                                after_head,
                                "--",
                                *paths,
                            ],
                            cwd=git_root,
                            check=False,
                        )
                    except OSError as exc:
                        return {
                            **result_base,
                            "release_allowed": False,
                            "reason": "protected_generated_history_query_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    if changed.returncode != 0:
                        return {
                            **result_base,
                            "release_allowed": False,
                            "reason": "protected_generated_history_missing_commit",
                            "before_head": before_head,
                            "after_head": after_head,
                        }
        elif after_head:
            history_result = self._git_protected_history(
                git_root,
                after_head,
                paths,
            )
            if not history_result.get("ok"):
                return {
                    **result_base,
                    "release_allowed": False,
                    "reason": "protected_generated_history_query_failed",
                    "history_query": history_result,
                }
            commits = list(history_result.get("commits") or ())

        untrusted_commits = [
            str(item.get("commit") or "")
            for item in commits
            if not item.get("trusted_generator")
        ]
        if untrusted_commits:
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_history_untrusted",
                "before_head": before_head,
                "after_head": after_head,
                "commits": commits,
                "untrusted_commits": untrusted_commits,
            }

        confirmed_head = self._git_head_state(git_root)
        confirmed_status = self._git_scope_dirty_paths(git_root, paths)
        if (
            not confirmed_head.get("ok")
            or str(confirmed_head.get("head") or "") != after_head
            or not confirmed_status.get("ok")
            or confirmed_status.get("dirty_paths")
        ):
            return {
                **result_base,
                "release_allowed": False,
                "reason": "protected_generated_release_state_changed",
                "before_head": before_head,
                "after_head": after_head,
                "confirmed_head": confirmed_head,
                "confirmed_status": confirmed_status,
            }
        return {
            **result_base,
            "release_allowed": True,
            "reason": (
                "protected_generated_history_trusted"
                if commits
                else "protected_outputs_clean_unrelated_history"
            ),
            "before_head": before_head,
            "after_head": after_head,
            "commits": commits,
        }

    @staticmethod
    def _git_scope_dirty_paths(
        git_root: Path,
        paths: Sequence[str],
    ) -> dict[str, Any]:
        try:
            status = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    *paths,
                ],
                cwd=git_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "dirty_paths": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        if status.returncode != 0:
            return {
                "ok": False,
                "dirty_paths": [],
                "returncode": status.returncode,
                "stderr": status.stderr[-4000:],
            }
        dirty_paths = [
            PortalImplementationSupervisor._status_line_path(line)
            for line in status.stdout.splitlines()
            if PortalImplementationSupervisor._status_line_path(line)
        ]
        return {
            "ok": True,
            "dirty_paths": list(dict.fromkeys(dirty_paths)),
        }

    def _git_protected_history(
        self,
        git_root: Path,
        revision: str,
        paths: Sequence[str],
    ) -> dict[str, Any]:
        try:
            history = subprocess.run(
                [
                    "git",
                    "log",
                    "--format=%H%x09%ae%x09%s",
                    revision,
                    "--",
                    *paths,
                ],
                cwd=git_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "commits": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        if history.returncode != 0:
            return {
                "ok": False,
                "commits": [],
                "returncode": history.returncode,
                "stderr": history.stderr[-4000:],
            }
        commits: list[dict[str, Any]] = []
        for line in history.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                return {
                    "ok": False,
                    "commits": [],
                    "reason": "history_malformed",
                }
            commit, author_email, subject = parts
            commits.append(
                {
                    "commit": commit,
                    "author_email": author_email,
                    "subject": subject,
                    "trusted_generator": (
                        self._trusted_generated_protected_commit(
                            author_email,
                            subject,
                        )
                    ),
                }
            )
        return {"ok": True, "commits": commits}

    def _run_protected_refill_mutation(
        self,
        *,
        scan_kind: str,
        scan_mode: str,
        analyzer_version: str,
        started_at: datetime,
        output_paths: Sequence[Path | None],
        callback,
    ):
        """Fence protected refill writes through their trusted generated commit."""

        protected_outputs = self._implementation_protected_output_paths(
            output_paths
        )
        if not protected_outputs:
            return callback()

        def deferred(payload: Mapping[str, Any]) -> RefillScanResult:
            return self._terminal_refill_result(
                ScanTerminalReason.PARTIAL,
                scan_mode=f"{scan_mode}_checkout_mutation_deferred",
                analyzer_version=analyzer_version,
                started_at=started_at,
                metadata={
                    "deferred_reason": str(
                        payload.get("reason")
                        or "checkout_mutation_lock_unavailable"
                    ),
                    "checkout_mutation": dict(payload),
                    "protected_output_paths": [
                        str(path) for path in protected_outputs
                    ],
                },
            )

        def run_and_commit():
            try:
                result = callback()
            except Exception:
                try:
                    self.repair_generated_dirty_checkouts(
                        force=True,
                        additional_paths=protected_outputs,
                    )
                except Exception:
                    logger.exception(
                        "Protected %s refill cleanup failed after callback "
                        "failure; retaining the checkout mutation lease",
                        scan_kind,
                    )
                    self._checkout_mutation_context.retain_until_protected_clean = (
                        True
                    )
                if self._dirty_implementation_protected_paths(
                    protected_outputs
                ):
                    self._checkout_mutation_context.retain_until_protected_clean = (
                        True
                    )
                raise

            self.repair_generated_dirty_checkouts(
                force=True,
                additional_paths=protected_outputs,
            )
            dirty_paths = self._dirty_implementation_protected_paths(
                protected_outputs
            )
            if dirty_paths:
                self._checkout_mutation_context.retain_until_protected_clean = (
                    True
                )
                raise RuntimeError(
                    "protected refill outputs remain dirty after generated "
                    f"commit: {', '.join(dirty_paths)}"
                )
            return result

        return self._run_generated_board_producer(
            producer=f"{scan_kind}-refill",
            commit_outputs=True,
            # The generated-output committer recognizes this operation as its
            # own same-process transaction and therefore does not deadlock on
            # the outer checkout lease.
            operation="generated_dirty_repair",
            callback=run_and_commit,
            deferred_result=deferred,
        )

    def _objective_refill_output_paths(self) -> tuple[Path, ...]:
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            default_objective_path,
        )

        state_root = self.config.state_dir.parent
        return tuple(
            dict.fromkeys(
                path
                for path in (
                    self.config.todo_path,
                    self.config.objective_path
                    or default_objective_path(self.config.repo_root),
                    self.config.objective_graph_path
                    or state_root / "objective_graph.json",
                    state_root / "objective_generation.json",
                    self.config.objective_todo_vector_index_path,
                    self.config.objective_goal_completion_gate_path,
                    self.config.objective_goal_completion_evidence_path,
                )
                if path is not None
            )
        )

    def _checkout_lock_owner_is_active(self, metadata: dict[str, Any]) -> bool:
        if not checkout_lock_owner_is_active(
            metadata,
            expected_kind="merge",
            expected_repo_root=self.config.repo_root,
            process_command_line=process_command_line,
            process_is_running=process_is_running,
        ):
            return False
        operation = str(metadata.get("operation") or "")
        if (
            str(metadata.get("lease_id") or "")
            and operation in ATOMIC_CHECKOUT_MUTATION_LEASE_OPERATIONS
        ):
            return True
        if self._checkout_lock_targets_current_supervisor_state(metadata):
            return self._checkout_lock_task_is_active(metadata)
        return True

    def _checkout_lock_targets_current_supervisor_state(self, metadata: dict[str, Any]) -> bool:
        state_path = str(metadata.get("state_path") or "")
        if state_path:
            try:
                return Path(state_path).resolve() == self.config.state_path.resolve()
            except OSError:
                return False
        state_dir = str(metadata.get("state_dir") or "")
        if state_dir:
            try:
                return Path(state_dir).resolve() == self.config.state_dir.resolve()
            except OSError:
                return False
        owner_script = str(metadata.get("owner_script") or "")
        owner_names = {
            Path(path).name
            for path in (self.config.daemon_script_path, self.config.supervisor_script_path)
            if path is not None
        }
        return bool(owner_script and owner_script in owner_names)

    def _checkout_lock_task_is_active(self, metadata: dict[str, Any]) -> bool:
        task_id = str(metadata.get("task_id") or "")
        if not task_id:
            return True
        try:
            state = PortalTaskState.load(self.config.state_path)
        except Exception:
            return True
        if state.active_task_id != task_id:
            return False
        branch = str(metadata.get("branch") or "")
        return not branch or not state.active_branch or state.active_branch == branch

    def _supervisor_checkout_lock_metadata(
        self,
        *,
        operation: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return checkout_lock_metadata(
            kind="merge",
            repo_root=self.config.repo_root,
            task_id="",
            branch="",
            owner_script=Path(sys.argv[0]).name,
            extra={
                "operation": operation,
                "state_dir": str(self.config.state_dir.resolve()),
                "state_path": str(self.config.state_path.resolve()),
                "started_at": utc_now(),
                **dict(extra or {}),
            },
        )

    def _acquire_supervisor_checkout_lease(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any],
    ) -> tuple[
        CheckoutMutationLease | None,
        str,
        dict[str, Any] | None,
    ]:
        """Acquire a fully published lease, retaining legacy test-hook support."""

        acquire = self._try_acquire_checkout_lock
        try:
            parameter_count = len(inspect.signature(acquire).parameters)
        except (TypeError, ValueError):
            parameter_count = 2
        if parameter_count == 1:
            # Older integrations monkeypatch the original one-argument helper.
            # Preserve that narrow deferral hook while production uses complete
            # metadata and the atomic lease implementation below.
            return acquire(lock_path)
        return acquire(lock_path, metadata)

    def _try_acquire_checkout_lock(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[
        CheckoutMutationLease | None,
        str,
        dict[str, Any] | None,
    ]:
        normalized_metadata = (
            dict(metadata)
            if metadata is not None
            else self._supervisor_checkout_lock_metadata(
                operation="supervisor_checkout_mutation",
            )
        )
        lease, reason, existing_or_cleared, _waited = (
            acquire_atomic_checkout_mutation_lease(
                lock_path,
                normalized_metadata,
                owner_active=self._checkout_lock_owner_is_active,
                timeout_seconds=0.0,
            )
        )
        if lease is not None and existing_or_cleared:
            self._record_checkout_mutation_lock_cleared(
                lock_path,
                existing_or_cleared,
            )
        return lease, reason, existing_or_cleared

    def _record_checkout_mutation_lock_cleared(
        self,
        lock_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        self._record_event(
            "checkout_mutation_lock_cleared",
            {
                "lock_path": str(lock_path),
                "lock_owner_pid": int(metadata.get("pid") or 0),
                "task_id": str(metadata.get("task_id") or ""),
                "branch": str(metadata.get("branch") or ""),
            },
        )

    def _release_supervisor_checkout_lease(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
    ) -> bool:
        released = release_checkout_mutation_lease(lease)
        if not released:
            logger.warning(
                "Supervisor checkout mutation lease for %s was replaced "
                "before release: %s",
                operation,
                lease.lock_path,
            )
        return released

    def repair_generated_main_checkout_conflicts(self, repo_root: Path) -> list[dict[str, object]]:
        """Resolve configured append-only generated markdown conflicts without LLM calls."""

        allowed_paths, allowed_dirs = self._append_only_markdown_conflict_targets()
        if not allowed_paths and not allowed_dirs:
            return []
        repairs = resolve_append_only_markdown_conflicts(
            repo_root=repo_root,
            allowed_paths=allowed_paths,
            allowed_dirs=allowed_dirs,
        )
        if repairs:
            self._record_event(
                "generated_markdown_conflict_repair",
                {
                    "repo_root": str(repo_root),
                    "results": repairs,
                },
            )
        return repairs

    def _append_only_markdown_conflict_targets(self) -> tuple[list[Path], list[Path]]:
        allowed_paths: list[Path] = []
        allowed_dirs: list[Path] = []
        if self.config.objective_path is not None:
            allowed_paths.append(self.config.objective_path)
        if self.config.objective_bundle_dir is not None:
            allowed_dirs.append(self.config.objective_bundle_dir)
        return allowed_paths, allowed_dirs

    def _invoke_main_checkout_merge_resolver(
        self,
        repo_root: Path,
        *,
        merge_head: str,
        unmerged_paths: list[str],
    ) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
            build_merge_prompt,
            invoke_llm_resolver,
        )

        target_branch = self._git_current_branch(repo_root) or "HEAD"
        active_task_id = ""
        try:
            active_task_id = PortalTaskState.load(self.config.state_path).active_task_id
        except Exception:
            active_task_id = ""
        merge_result = {
            "attempted": True,
            "merged": False,
            "returncode": 1,
            "branch": merge_head,
            "target_branch": target_branch,
            "command": ["git", "status", "--short"],
            "reason": "supervisor_main_checkout_merge_in_progress",
            "stdout": "\n".join(self._git_status_short(repo_root)),
            "stderr": "",
            "main_worktree_path": str(repo_root),
            "dirty_paths": unmerged_paths,
        }
        event = {
            "type": "supervisor_main_checkout_merge_repair",
            "task_id": active_task_id or self.config.state_prefix,
            "attempt": 0,
            "merge_result": merge_result,
        }
        payload = {
            "found": True,
            "task_id": active_task_id,
            "attempt": 0,
            "events_path": str(self.config.events_path),
            "repo_root": str(repo_root),
            "branch": merge_head,
            "target_branch": target_branch,
            "command": merge_result["command"],
            "reason": merge_result["reason"],
            "dirty_paths": unmerged_paths,
            "unmerged_paths": unmerged_paths,
            "prompt": build_merge_prompt(event=event, repo_root=repo_root),
        }
        return invoke_llm_resolver(
            payload,
            command_template=self.config.llm_merge_resolver_command,
            timeout_seconds=self.config.llm_merge_resolver_timeout_seconds,
        )

    @staticmethod
    def _compact_resolver_result(result: dict[str, Any]) -> dict[str, Any]:
        compact = dict(result)
        if "prompt" in compact:
            compact["prompt_chars"] = len(str(compact.pop("prompt") or ""))
        return compact

    def _commit_supervisor_resolved_merge(self, repo_root: Path) -> dict[str, Any]:
        unresolved = self._git_unmerged_paths(repo_root)
        if unresolved:
            return {
                "attempted": True,
                "completed": False,
                "reason": "unresolved_paths_remain",
                "unresolved_paths": unresolved,
            }
        if not self._git_merge_head(repo_root):
            return {
                "attempted": False,
                "completed": True,
                "reason": "resolver_committed_merge",
            }
        add = subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            return {
                "attempted": True,
                "completed": False,
                "reason": "stage_resolved_merge_failed",
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
            }
        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Supervisor",
                "-c",
                "user.email=implementation-supervisor@example.invalid",
                "commit",
                "--no-edit",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "attempted": True,
            "completed": commit.returncode == 0,
            "reason": "committed" if commit.returncode == 0 else "commit_failed",
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }

    def _abort_main_checkout_merge(self, repo_root: Path) -> dict[str, Any]:
        if not self._git_merge_head(repo_root):
            return {"attempted": False, "aborted": False, "reason": "no_merge_in_progress"}
        abort = subprocess.run(
            ["git", "merge", "--abort"],
            cwd=repo_root,
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
        if abort.returncode != 0 and (self._git_merge_head(repo_root) or self._git_unmerged_paths(repo_root)):
            reset = subprocess.run(
                ["git", "reset", "--merge"],
                cwd=repo_root,
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
        return result

    @staticmethod
    def _git_merge_head(repo_root: Path) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _git_merge_head_query(repo_root: Path) -> dict[str, Any]:
        """Return a tri-state MERGE_HEAD observation without conflating errors."""

        try:
            git_path = subprocess.run(
                ["git", "rev-parse", "--git-path", "MERGE_HEAD"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "merge_head": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if git_path.returncode != 0 or not git_path.stdout.strip():
            return {
                "ok": False,
                "merge_head": "",
                "returncode": git_path.returncode,
                "stderr": git_path.stderr[-4000:],
            }
        merge_head_path = Path(git_path.stdout.strip())
        if not merge_head_path.is_absolute():
            merge_head_path = repo_root / merge_head_path
        try:
            merge_head_path.stat()
        except FileNotFoundError:
            return {
                "ok": True,
                "merge_head": "",
                "merge_head_path": str(merge_head_path),
            }
        except OSError as exc:
            return {
                "ok": False,
                "merge_head": "",
                "merge_head_path": str(merge_head_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "MERGE_HEAD^{commit}"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "merge_head": "",
                "merge_head_path": str(merge_head_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        if result.returncode != 0 or not result.stdout.strip():
            return {
                "ok": False,
                "merge_head": "",
                "merge_head_path": str(merge_head_path),
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
            }
        return {
            "ok": True,
            "merge_head": result.stdout.strip(),
            "merge_head_path": str(merge_head_path),
        }

    @staticmethod
    def _git_unmerged_paths(repo_root: Path) -> list[str]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())

    @staticmethod
    def _git_unmerged_paths_query(repo_root: Path) -> dict[str, Any]:
        """Return unmerged paths only when Git successfully completed the query."""

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return {
                "ok": False,
                "unmerged_paths": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        if result.returncode != 0:
            return {
                "ok": False,
                "unmerged_paths": [],
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
            }
        return {
            "ok": True,
            "unmerged_paths": sorted(
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ),
        }

    @staticmethod
    def _git_status_short(repo_root: Path) -> list[str]:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
                check=False,
            )
        except OSError:
            return []
        if result.returncode != 0:
            return []
        return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]

    @staticmethod
    def _git_status_short_strict(repo_root: Path) -> list[str]:
        """Return color-free short status or fail when Git cannot certify it.

        Reconciliation mutates the shared checkout.  The ordinary status
        helper intentionally treats an unavailable repository as empty for
        best-effort diagnostics, but that behavior is unsafe at this gate:
        an unavailable or truncated status must never be interpreted as a
        clean main checkout.  ``--short`` is deliberate here: porcelain-v1
        collapses a submodule's lowercase content-only ``m`` marker to
        uppercase ``M`` and would erase the distinction this proof requires.
        """

        try:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "color.status=false",
                    "status",
                    "--short",
                    "--untracked-files=all",
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(
                "main checkout status unavailable"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                "main checkout status unavailable"
                f" (returncode={result.returncode})"
                + (f": {detail[-1000:]}" if detail else "")
            )
        return [
            line
            for line in result.stdout.splitlines()
            if line
        ]

    @staticmethod
    def _git_current_branch(repo_root: Path) -> str:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _git_ref_commit(repo_root: Path, ref: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _worktree_scan_cache_path(self) -> Path:
        return self.config.worktree_scan_cache_path or (
            self.config.state_dir / f"{self.config.state_prefix}_worktree_scan_cache.json"
        )

    def _load_worktree_scan_cache(self) -> dict[str, Any]:
        if (
            not self.config.worktree_scan_cache_enabled
            or self.config.worktree_scan_cache_ttl_seconds <= 0
        ):
            return {"enabled": False, "entries": {}}
        payload = load_json_dict(self._worktree_scan_cache_path())
        entries = payload.get("entries") if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            entries = {}
        return {
            "enabled": True,
            "version": 1,
            "entries": {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, dict)
            },
        }

    def _write_worktree_scan_cache(self, cache: dict[str, Any]) -> bool:
        if not cache.get("enabled") or not cache.get("_changed"):
            return False
        entries = cache.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        now = time.time()
        ttl = max(float(self.config.worktree_scan_cache_ttl_seconds), 0.0)
        max_age = max(ttl * 4, ttl + 3600.0)
        pruned_entries = {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, dict)
            and now - float(value.get("updated_at_epoch") or 0.0) <= max_age
        }
        path = self._worktree_scan_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            path,
            {
                "version": 1,
                "updated_at_epoch": now,
                "updated_at": utc_now(),
                "entries": pruned_entries,
            },
        )
        return True

    @staticmethod
    def _worktree_scan_cache_key(
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
    ) -> str:
        return sha1(
            json.dumps(
                {
                    "phase": phase,
                    "path": str(path),
                    "branch": branch,
                    "head": head,
                    "target_signature": target_signature,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _worktree_scan_cache_entry(
        self,
        cache: dict[str, Any],
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
    ) -> dict[str, Any] | None:
        if not cache.get("enabled"):
            return None
        key = self._worktree_scan_cache_key(
            phase=phase,
            path=path,
            branch=branch,
            head=head,
            target_signature=target_signature,
        )
        entries = cache.get("entries")
        entry = entries.get(key) if isinstance(entries, dict) else None
        if not isinstance(entry, dict):
            return None
        if time.time() - float(entry.get("updated_at_epoch") or 0.0) > float(
            self.config.worktree_scan_cache_ttl_seconds
        ):
            return None
        return dict(entry)

    def _store_worktree_scan_cache_entry(
        self,
        cache: dict[str, Any],
        *,
        phase: str,
        path: Path,
        branch: str,
        head: str,
        target_signature: str,
        classification: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not cache.get("enabled"):
            return
        key = self._worktree_scan_cache_key(
            phase=phase,
            path=path,
            branch=branch,
            head=head,
            target_signature=target_signature,
        )
        entries = cache.setdefault("entries", {})
        if not isinstance(entries, dict):
            cache["entries"] = entries = {}
        now = time.time()
        entries[key] = {
            "phase": phase,
            "path": str(path),
            "branch": branch,
            "head": head,
            "target_signature": target_signature,
            "classification": classification,
            "payload": dict(payload),
            "updated_at_epoch": now,
            "updated_at": utc_now(),
        }
        cache["_changed"] = True

    @staticmethod
    def _path_age_seconds(path: Path, *, now_ts: float) -> float | None:
        try:
            return max(0.0, now_ts - path.stat().st_mtime)
        except OSError:
            return None

    @staticmethod
    def _timestamp_age_seconds(value: str, *, now_ts: float) -> float | None:
        parsed = parse_timestamp(value)
        if parsed is None:
            return None
        return max(0.0, now_ts - parsed.timestamp())

    @staticmethod
    def _git_ahead_behind(repo_root: Path, left_ref: str, right_ref: str) -> dict[str, Any]:
        if not left_ref or not right_ref:
            return {"available": False, "ahead": 0, "behind": 0, "reason": "missing_ref"}
        result = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", f"{left_ref}...{right_ref}"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return {
                "available": False,
                "ahead": 0,
                "behind": 0,
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
            }
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            return {"available": False, "ahead": 0, "behind": 0, "reason": "unexpected_output"}
        try:
            left_only = int(parts[0])
            right_only = int(parts[1])
        except ValueError:
            return {"available": False, "ahead": 0, "behind": 0, "reason": "non_integer_output"}
        return {"available": True, "ahead": right_only, "behind": left_only}

    def _active_worktree_stale_signal(
        self,
        state: PortalTaskState,
        *,
        target_ref: str,
        now_ts: float,
        process_lines: list[str],
    ) -> dict[str, Any] | None:
        active_worktree = state.active_worktree_path.strip()
        if not state.implementation_in_progress or not active_worktree:
            return None
        active_branch = state.active_branch.strip()
        log_path = Path(state.active_log_path) if state.active_log_path else None
        log_age_seconds = self._path_age_seconds(log_path, now_ts=now_ts) if log_path is not None else None
        phase_age_seconds = self._timestamp_age_seconds(state.active_phase_started_at, now_ts=now_ts)
        heartbeat_age_seconds = self._timestamp_age_seconds(state.heartbeat_at, now_ts=now_ts)
        path_owned_by_process = any(active_worktree in line for line in process_lines)
        branch_owned_by_process = bool(active_branch) and any(active_branch in line for line in process_lines)
        daemon_pid = self._read_managed_daemon_pid()
        daemon_running = bool(daemon_pid and process_is_running(daemon_pid))
        daemon_matches = False
        if daemon_running and daemon_pid:
            daemon_matches = self._managed_daemon_matches_command_line(process_command_line(daemon_pid))
        owner_running = daemon_matches or path_owned_by_process or branch_owned_by_process
        stalled_log = (
            log_age_seconds is None
            or log_age_seconds > max(float(self.config.implementation_log_stall_seconds), 0.0)
        )
        state_old_enough = (
            (phase_age_seconds is not None and phase_age_seconds > max(float(self.config.stale_seconds), 0.0))
            or (heartbeat_age_seconds is not None and heartbeat_age_seconds > max(float(self.config.stale_seconds), 0.0))
        )
        reasons: list[str] = []
        if not owner_running:
            reasons.append("active_worktree_owner_missing")
        if stalled_log and state_old_enough:
            reasons.append("active_log_stalled")
        ahead_behind = (
            self._git_ahead_behind(self.config.repo_root, target_ref, active_branch)
            if active_branch
            else {"available": False, "ahead": 0, "behind": 0, "reason": "missing_active_branch"}
        )
        git_stale_context = not owner_running or (stalled_log and state_old_enough)
        if git_stale_context and int(ahead_behind.get("behind") or 0) > 0:
            reasons.append("active_branch_behind_target")
        if git_stale_context and int(ahead_behind.get("ahead") or 0) > 0:
            reasons.append("active_branch_has_unmerged_commits")
        if not reasons:
            return None
        return {
            "path": active_worktree,
            "branch": active_branch,
            "head": state.active_task_id,
            "kind": "active_state",
            "reasons": reasons,
            "remedy": "repair_stale_active_execution_state_then_reconcile",
            "owner_running": owner_running,
            "daemon_pid": daemon_pid or 0,
            "daemon_running": daemon_running,
            "daemon_matches": daemon_matches,
            "active_log_path": state.active_log_path,
            "active_log_age_seconds": log_age_seconds,
            "active_phase_age_seconds": phase_age_seconds,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "ahead_behind": ahead_behind,
        }

    def detect_stale_worktrees(self, *, now_ts: float | None = None) -> dict[str, Any]:
        """Detect remedyable worktrees using git, process, and log movement signals."""

        worktree_root = self.config.worktree_root
        if worktree_root is None:
            return {"attempted": False, "reason": "worktree_root_not_configured"}
        now = time.time() if now_ts is None else float(now_ts)
        repo_root = self.config.repo_root
        records = self._git_worktree_records(repo_root)
        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root
        process_lines = self._list_process_commands()
        state = PortalTaskState.load(self.config.state_path)
        active_worktree_owners = self._shared_active_worktree_owners(
            worktree_root
        )
        target_ref = (
            self.config.merge_target_branch
            or self._git_current_branch(repo_root)
            or "HEAD"
        )
        target_signature = self._git_ref_commit(repo_root, target_ref) or target_ref
        stale_items: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        active_signal = self._active_worktree_stale_signal(
            state,
            target_ref=target_ref,
            now_ts=now,
            process_lines=process_lines,
        )
        if active_signal:
            stale_items.append(active_signal)

        for record in records:
            path_text = str(record.get("worktree") or "")
            if not path_text:
                continue
            path = Path(path_text)
            try:
                path_resolved = path.resolve()
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            branch = str(record.get("branch") or "").removeprefix("refs/heads/")
            head = str(record.get("HEAD") or "")
            detail: dict[str, Any] = {
                "path": str(path),
                "branch": branch,
                "head": head,
                "kind": "worktree",
            }
            active_skip = self._active_worktree_skip_detail(
                path_resolved,
                active_worktree_owners,
            )
            if active_skip is not None:
                skipped.append({**detail, **active_skip})
                continue
            if any(str(path_resolved) in line for line in process_lines):
                skipped.append({**detail, "reason": "active_process"})
                continue
            if not self._worktree_branch_is_reconcilable(branch):
                skipped.append({**detail, "reason": "non_reconcilable_branch"})
                continue

            reasons: list[str] = []
            branch_exists = self._git_ref_exists(repo_root, branch)
            branch_merged = branch_exists and self._git_ref_is_ancestor(repo_root, branch, target_ref)
            head_merged = bool(head) and self._git_ref_is_ancestor(repo_root, head, target_ref)
            if branch_merged:
                reasons.append("branch_already_merged")
            elif head_merged:
                reasons.append("head_already_merged")
            ahead_behind = (
                self._git_ahead_behind(repo_root, target_ref, branch)
                if branch_exists
                else {"available": False, "ahead": 0, "behind": 0, "reason": "branch_missing"}
            )
            ahead = int(ahead_behind.get("ahead") or 0)
            behind = int(ahead_behind.get("behind") or 0)
            if ahead > 0:
                reasons.append("branch_has_unmerged_commits")
            if behind > 0:
                reasons.append("branch_behind_target")
            dirty = self._git_status_short(path) if path.exists() else []
            if dirty:
                reasons.append("dirty_inactive_worktree")
            worktree_age_seconds = self._path_age_seconds(path, now_ts=now)
            if worktree_age_seconds is not None and worktree_age_seconds > max(float(self.config.stale_seconds), 0.0):
                if ahead > 0 or behind > 0 or dirty or branch_merged or head_merged:
                    reasons.append("calendar_age_supports_git_staleness")
            if not reasons:
                skipped.append({**detail, "reason": "no_stale_signal"})
                continue
            remedy = "reconcile_backlogged_worktrees"
            if branch_merged or head_merged:
                remedy = "cleanup_backlogged_worktrees"
            if dirty:
                remedy = "rescue_dirty_worktree_then_reconcile"
            stale_items.append(
                {
                    **detail,
                    "reasons": sorted(set(reasons)),
                    "remedy": remedy,
                    "branch_exists": branch_exists,
                    "branch_merged": branch_merged,
                    "head_merged": head_merged,
                    "ahead_behind": ahead_behind,
                    "dirty": bool(dirty),
                    "status_short": dirty[:20],
                    "worktree_age_seconds": worktree_age_seconds,
                }
            )

        reason_counts: dict[str, int] = {}
        for item in stale_items:
            for reason in item.get("reasons") or []:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        result = {
            "attempted": True,
            "worktree_root": str(worktree_root),
            "target_ref": target_ref,
            "target_signature": target_signature,
            "stale_count": len(stale_items),
            "remedy_count": sum(1 for item in stale_items if item.get("remedy")),
            "reason_counts": reason_counts,
            "stale": stale_items[:50],
            "skipped_count": len(skipped),
            "skipped": skipped[:50],
        }
        if stale_items:
            self._record_event("stale_worktree_detection", result)
        return result

    def reconcile_backlogged_worktrees(
        self,
        *,
        preacquired_implementation_lock: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retry clean inactive implementation worktrees before cleanup."""

        if not self.config.worktree_reconciliation_enabled:
            return {"attempted": False, "reason": "worktree_reconciliation_disabled"}
        worktree_root = self.config.worktree_root
        if worktree_root is None:
            return {"attempted": False, "reason": "worktree_root_not_configured"}

        repo_root = self.config.repo_root
        records = self._git_worktree_records(repo_root)
        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root
        process_lines = self._list_process_commands()
        active_worktree_owners = self._shared_active_worktree_owners(
            worktree_root
        )
        current_branch = self._git_current_branch(repo_root)
        target_ref = self.config.merge_target_branch or current_branch or "HEAD"
        target_signature = self._git_ref_commit(repo_root, target_ref) or target_ref
        main_status_available = True
        main_status_error = ""
        try:
            raw_main_status = (
                self._main_status_for_worktree_reconciliation(
                    repo_root,
                    worktree_root,
                )
            )
        except (OSError, RuntimeError) as exc:
            main_status_available = False
            main_status_error = f"{type(exc).__name__}: {exc}"
            raw_main_status = []
        if main_status_available:
            raw_main_dirty_evidence = (
                self._main_checkout_dirty_evidence(
                    repo_root,
                    raw_main_status,
                )
                if raw_main_status
                else {}
            )
            main_status, main_dirty_evidence = (
                self._filter_generated_main_checkout_status(
                    raw_main_status,
                    raw_main_dirty_evidence,
                )
            )
        else:
            raw_main_dirty_evidence = {
                "reason": "main_checkout_status_unavailable",
                "error": main_status_error[-2000:],
            }
            main_status = []
            main_dirty_evidence = dict(raw_main_dirty_evidence)
        current_checkout_status = list(main_status)
        main_checkout_is_merge_target = (
            not self.config.merge_target_branch
            or current_branch == target_ref
        )
        if main_status and not main_checkout_is_merge_target:
            main_dirty_evidence = {
                **main_dirty_evidence,
                "ignored_for_reconciliation": True,
                "current_branch": current_branch or "HEAD",
                "configured_merge_target": target_ref,
            }
            # Reconciliation mutates a detached target worktree. Dirt in an
            # unrelated checkout is reported below but does not authorize or
            # block mutation of the configured target branch.
            main_status = []
        max_merges = max(0, int(self.config.worktree_reconciliation_max_merges))
        dry_run = bool(self.config.worktree_reconciliation_dry_run)
        scan_cache = self._load_worktree_scan_cache()
        scan_cache_hit_count = 0
        candidates: list[dict[str, Any]] = []
        processed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        blocking_main_status: list[str] = []
        nonblocking_main_gitlinks: list[dict[str, Any]] = []
        candidate_main_status_cache: dict[
            tuple[str, str],
            tuple[list[str], list[dict[str, Any]]],
        ] = {}
        reconciliation_daemon: PortalImplementationDaemon | None = None
        reconciliation_tasks_by_id: dict[str, PortalTask] = {}
        reconciliation_task_ids_by_branch: dict[str, str] = {}
        reconciliation_outcome_keys: set[str] = set()
        reconciliation_provenance_by_branch: dict[
            str, dict[str, Any]
        ] = {}

        def candidate_main_status(
            branch: str,
            head: str,
        ) -> tuple[list[str], list[dict[str, Any]]]:
            key = (branch, head)
            cached = candidate_main_status_cache.get(key)
            if cached is not None:
                return cached
            if not main_status_available or not main_status:
                classified = (list(main_status), [])
            else:
                try:
                    classified = self._candidate_main_checkout_status(
                        repo_root,
                        main_status,
                        target_ref=target_ref,
                        branch=branch,
                        candidate_head=head,
                    )
                except (OSError, RuntimeError, ValueError):
                    # This is an authorization proof, not a liveness hint.
                    # Any failed identity/status query keeps every line
                    # blocking without mutating the nested checkout.
                    classified = (list(main_status), [])
            candidate_main_status_cache[key] = classified
            return classified

        def record_main_status_classification(
            blocking: Sequence[str],
            nonblocking: Sequence[dict[str, Any]],
        ) -> None:
            for line in blocking:
                if line not in blocking_main_status:
                    blocking_main_status.append(line)
            known = {
                (
                    str(item.get("path") or ""),
                    str(item.get("candidate_commit") or ""),
                )
                for item in nonblocking_main_gitlinks
            }
            for proof in nonblocking:
                identity = (
                    str(proof.get("path") or ""),
                    str(proof.get("candidate_commit") or ""),
                )
                if identity not in known:
                    nonblocking_main_gitlinks.append(dict(proof))
                    known.add(identity)

        for record in records:
            path_text = str(record.get("worktree") or "")
            if not path_text:
                continue
            path = Path(path_text)
            try:
                path_resolved = path.resolve()
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue

            branch = str(record.get("branch") or "").removeprefix("refs/heads/")
            head = str(record.get("HEAD") or "")
            detail: dict[str, Any] = {"path": str(path), "branch": branch, "head": head}
            active_skip = self._active_worktree_skip_detail(
                path_resolved,
                active_worktree_owners,
            )
            if active_skip is not None:
                skipped.append({**detail, **active_skip})
                continue
            if any(str(path_resolved) in line for line in process_lines):
                skipped.append({**detail, "reason": "active_process"})
                continue
            cached_entry = self._worktree_scan_cache_entry(
                scan_cache,
                phase="reconciliation",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
            )
            if cached_entry:
                classification = str(cached_entry.get("classification") or "")
                payload = dict(cached_entry.get("payload") or {})
                if classification == "skip":
                    if payload.get("reason") == "dirty_worktree":
                        pass
                    else:
                        skipped.append({**payload, "cached": True})
                        scan_cache_hit_count += 1
                        continue
                elif classification == "candidate":
                    cached_blocking, cached_nonblocking = (
                        candidate_main_status(branch, head)
                    )
                    record_main_status_classification(
                        cached_blocking,
                        cached_nonblocking,
                    )
                    if (
                        dry_run
                        or not main_status_available
                        or cached_blocking
                    ):
                        candidate = {**payload, "cached": True}
                        if cached_nonblocking:
                            candidate[
                                "nonblocking_main_gitlinks"
                            ] = cached_nonblocking
                        candidates.append(candidate)
                        scan_cache_hit_count += 1
                        if not dry_run:
                            skipped.append(
                                {
                                    **candidate,
                                    "reason": (
                                        "main_checkout_dirty"
                                        if main_status_available
                                        else "main_checkout_status_unavailable"
                                    ),
                                    "status_short": cached_blocking[:20],
                                }
                            )
                        continue
                    # A clean candidate can have been deferred by a transient
                    # claim or lane lease, and its task CID can change while
                    # its Git identity stays fixed.  Re-evaluate it instead
                    # of turning the scan cache into a permanent tombstone.
                else:
                    skipped.append({**payload, "cached": True})
                    scan_cache_hit_count += 1
                    continue
            if not self._worktree_branch_is_reconcilable(branch):
                skip = {**detail, "reason": "non_implementation_branch"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            if not self._git_ref_exists(repo_root, branch):
                skip = {**detail, "reason": "implementation_branch_missing"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue

            branch_merged = self._git_ref_is_ancestor(repo_root, branch, target_ref)
            head_merged = bool(head) and self._git_ref_is_ancestor(repo_root, head, target_ref)
            if branch_merged or head_merged:
                skip = {**detail, "reason": "already_merged_cleanup_pass"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="reconciliation",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            dirty = self._git_status_short(path) if path.exists() else []
            if dirty:
                rescue_result = self._rescue_dirty_worktree(
                    path,
                    branch=branch,
                    head=head,
                    target_ref=target_ref,
                    status_lines=dirty,
                    reason="reconciliation_dirty_worktree",
                )
                if rescue_result.get("preserved"):
                    branch = str(rescue_result.get("rescue_branch") or branch)
                    head = str(rescue_result.get("rescue_commit") or head)
                    detail = {
                        **detail,
                        "branch": branch,
                        "head": head,
                        "rescued_from_branch": record.get("branch", ""),
                        "rescue_result": rescue_result,
                    }
                    path_resolved = path.resolve()
                else:
                    skip = {
                        **detail,
                        "reason": "dirty_worktree",
                        "status_short": dirty[:20],
                        "rescue_result": rescue_result,
                    }
                    skipped.append(skip)
                    self._store_worktree_scan_cache_entry(
                        scan_cache,
                        phase="reconciliation",
                        path=path_resolved,
                        branch=branch,
                        head=head,
                        target_signature=target_signature,
                        classification="skip",
                        payload=skip,
                    )
                    continue

            candidate_blocking, candidate_nonblocking = (
                candidate_main_status(branch, head)
            )
            record_main_status_classification(
                candidate_blocking,
                candidate_nonblocking,
            )
            candidate = {**detail, "target_ref": target_ref}
            if candidate_nonblocking:
                candidate[
                    "nonblocking_main_gitlinks"
                ] = candidate_nonblocking
            candidates.append(candidate)
            self._store_worktree_scan_cache_entry(
                scan_cache,
                phase="reconciliation",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
                classification="candidate",
                payload=candidate,
            )
            if dry_run:
                continue
            if not main_status_available:
                skipped.append(
                    {
                        **candidate,
                        "reason": "main_checkout_status_unavailable",
                        "status_short": [],
                    }
                )
                continue
            if candidate_blocking:
                skipped.append(
                    {
                        **candidate,
                        "reason": "main_checkout_dirty",
                        "status_short": candidate_blocking[:20],
                    }
                )
                continue
            if sum(1 for item in processed if item.get("merged")) >= max_merges:
                skipped.append({**candidate, "reason": "reconciliation_limit_reached"})
                continue

            preflight_result: dict[str, Any] = {}
            preflight_resolver_escalated = False
            if self.config.worktree_reconciliation_preflight_enabled:
                preflight_result = self._preflight_worktree_reconciliation_merge(
                    repo_root,
                    target_ref=target_ref,
                    branch=branch,
                )
                if not preflight_result.get("mergeable", False):
                    if not self.config.llm_merge_resolver_command:
                        processed.append(
                            {
                                **candidate,
                                "merged": False,
                                "preflight_result": preflight_result,
                                "preflight_resolver_escalated": False,
                                "merge_result": {
                                    "attempted": False,
                                    "merged": False,
                                    "returncode": preflight_result.get("returncode"),
                                    "branch": branch,
                                    "target_ref": target_ref,
                                    "reason": "preflight_merge_conflict",
                                    "stdout": preflight_result.get("stdout", ""),
                                    "stderr": preflight_result.get("stderr", ""),
                                },
                            }
                        )
                        continue
                    preflight_resolver_escalated = True

            if reconciliation_daemon is None:
                reconciliation_daemon = self._build_worktree_reconciliation_daemon()
                (
                    reconciliation_tasks_by_id,
                    reconciliation_task_ids_by_branch,
                    reconciliation_outcome_keys,
                    reconciliation_provenance_by_branch,
                ) = self._reconciliation_task_context(
                    reconciliation_daemon
                )
            current_task = self._current_reconciliation_task(
                branch=branch,
                rescued_from_branch=str(
                    detail.get("rescued_from_branch") or ""
                ),
                tasks_by_id=reconciliation_tasks_by_id,
                task_ids_by_branch=reconciliation_task_ids_by_branch,
            )
            if current_task is None:
                unresolved_reason = (
                    "task_identity_unresolved"
                    if reconciliation_tasks_by_id
                    else "task_board_unavailable"
                )
                processed.append(
                    {
                        **candidate,
                        "merged": False,
                        "preflight_result": preflight_result,
                        "preflight_resolver_escalated": (
                            preflight_resolver_escalated
                        ),
                        "merge_result": {
                            "attempted": False,
                            "merged": False,
                            "reason": (
                                "reconciliation_candidate_"
                                f"{unresolved_reason}"
                            ),
                        },
                    }
                )
                continue
            if (
                current_task is not None
                and str(current_task.status).strip().lower()
                == "completed"
            ):
                processed.append(
                    {
                        **candidate,
                        "merged": False,
                        "preflight_result": preflight_result,
                        "preflight_resolver_escalated": (
                            preflight_resolver_escalated
                        ),
                        "merge_result": {
                            "attempted": False,
                            "merged": False,
                            "reason": (
                                "reconciliation_candidate_"
                                "task_already_completed"
                            ),
                        },
                    }
                )
                continue
            if current_task is not None:
                task_identity = reconciliation_daemon._identity_for_task(
                    current_task
                )
                baseline_ref = self._git_merge_base(
                    repo_root,
                    target_ref,
                    head or branch,
                )
                recovery_key = (
                    self._worktree_reconciliation_recovery_key(
                        task_cid=task_identity.canonical_task_cid,
                        baseline_ref=baseline_ref,
                        candidate_commit=head,
                        target_commit=target_signature,
                        mode="pre_merge",
                    )
                    if baseline_ref and head
                    else ""
                )
                if not recovery_key:
                    processed.append(
                        {
                            **candidate,
                            "merged": False,
                            "preflight_result": preflight_result,
                            "preflight_resolver_escalated": (
                                preflight_resolver_escalated
                            ),
                            "merge_result": {
                                "attempted": False,
                                "merged": False,
                                "reason": (
                                    "reconciliation_candidate_"
                                    "baseline_unavailable"
                                ),
                            },
                        }
                    )
                    continue
                if recovery_key in reconciliation_outcome_keys:
                    processed.append(
                        {
                            **candidate,
                            "merged": False,
                            "preflight_result": preflight_result,
                            "preflight_resolver_escalated": (
                                preflight_resolver_escalated
                            ),
                            "merge_result": {
                                "attempted": False,
                                "merged": False,
                                "reason": (
                                    "reconciliation_candidate_"
                                    "validation_already_settled"
                                ),
                            },
                            "recovery_key": recovery_key,
                        }
                    )
                    continue
                recovery_result = (
                    reconciliation_daemon.reconcile_validated_worktree_candidate(
                        worktree_path=path,
                        branch_name=branch,
                        task=current_task,
                        baseline_ref=baseline_ref,
                        candidate_commit=head,
                        recovery_key=recovery_key,
                        preacquired_implementation_lock=(
                            preacquired_implementation_lock
                        ),
                    )
                )
                merge_result = dict(
                    recovery_result.get("merge_result") or {}
                )
                cleanup_result = self._reconciliation_cleanup_result(
                    merge_result
                )
                processed.append(
                    {
                        **candidate,
                        "merged": bool(merge_result.get("merged")),
                        "preflight_result": preflight_result,
                        "preflight_resolver_escalated": (
                            preflight_resolver_escalated
                        ),
                        "merge_result": merge_result,
                        "cleanup_result": cleanup_result,
                        "recovery_result": recovery_result,
                        "recovery_key": recovery_key,
                        "validated_before_merge": True,
                    }
                )
                reconciliation_outcome_keys.add(recovery_key)
                continue

        effective_main_status = (
            list(main_status)
            if not candidates
            else list(blocking_main_status)
        )
        if nonblocking_main_gitlinks:
            main_dirty_evidence = {
                **main_dirty_evidence,
                "nonblocking_submodule_content_status": (
                    nonblocking_main_gitlinks[:50]
                ),
                "filtered_nonblocking_status_paths": sorted(
                    {
                        str(item.get("path") or "")
                        for item in nonblocking_main_gitlinks
                        if str(item.get("path") or "")
                    }
                )[:50],
            }
        if effective_main_status:
            main_dirty_evidence = {
                **main_dirty_evidence,
                "status_short": effective_main_status[:50],
                "status_paths": [
                    self._status_line_path(line)
                    for line in effective_main_status[:50]
                ],
            }
        elif (
            main_status_available
            and not main_dirty_evidence.get("ignored_for_reconciliation")
        ):
            main_dirty_evidence = {
                **main_dirty_evidence,
                "status_short": [],
                "status_paths": [],
            }

        result = {
            "attempted": True,
            "worktree_root": str(worktree_root),
            "target_ref": target_ref,
            "target_signature": target_signature,
            "dry_run": dry_run,
            "max_merges": max_merges,
            "main_checkout_dirty": (
                not main_status_available
                or bool(effective_main_status)
            ),
            "main_checkout_status_available": main_status_available,
            "main_checkout_status_error": main_status_error[-2000:],
            "main_status_short": effective_main_status[:20],
            "main_dirty_evidence": main_dirty_evidence,
            "raw_main_checkout_dirty": (
                not main_status_available
                or bool(raw_main_status)
            ),
            "main_checkout_is_merge_target": main_checkout_is_merge_target,
            "current_checkout_dirty": bool(current_checkout_status),
            "current_checkout_status_short": current_checkout_status[:20],
            "raw_main_status_short": raw_main_status[:20],
            "raw_main_dirty_evidence": raw_main_dirty_evidence,
            "candidate_count": len(candidates),
            "processed_count": len(processed),
            "reconciled_count": sum(1 for item in processed if item.get("merged")),
            "preflight_blocked_count": sum(
                1
                for item in processed
                if isinstance(item.get("preflight_result"), dict)
                and not item["preflight_result"].get("mergeable", False)
                and not item.get("merged", False)
            ),
            "preflight_resolver_escalation_count": sum(
                1 for item in processed if item.get("preflight_resolver_escalated", False)
            ),
            "cleanup_count": sum(
                1
                for item in processed
                if isinstance(item.get("cleanup_result"), dict)
                and item["cleanup_result"].get("cleaned", False)
            ),
            "skipped_count": len(skipped),
            "candidates": candidates[:50],
            "processed": processed,
            "skipped": skipped[:50],
            "scan_cache_hit_count": scan_cache_hit_count,
            "scan_cache_written": self._write_worktree_scan_cache(scan_cache),
        }
        if processed:
            self._record_event("worktree_reconciliation", result)
        return result

    def recover_already_merged_reconciliation_candidates(
        self,
        *,
        preacquired_implementation_lock: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replay legacy raw merges through current proposal/completion gates.

        Older supervisors could merge and delete a clean orphan worktree while
        recording only maintenance telemetry.  Recovery is permitted only
        when that event proves the exact two-parent merge, preflight tree, and
        cleanup.  The exact managed ``implementation_started`` provenance is
        then required so a disposable branch can recreate the immutable
        original candidate, validate its non-empty proposal against its
        original baseline using the *current* task CID, and submit that
        already-integrated candidate to the normal merge train.  The train
        remains the sole completion and task-board authority.
        """

        if not self.config.worktree_reconciliation_enabled:
            return {
                "attempted": False,
                "reason": "worktree_reconciliation_disabled",
            }
        max_replays = max(
            0,
            int(self.config.worktree_reconciliation_max_merges),
        )
        if max_replays <= 0:
            return {
                "attempted": False,
                "reason": "worktree_reconciliation_replay_disabled",
            }

        daemon = self._build_worktree_reconciliation_daemon()
        (
            tasks_by_id,
            task_ids_by_branch,
            managed_outcome_keys,
            implementation_provenance_by_branch,
        ) = self._reconciliation_task_context(daemon)
        if not tasks_by_id:
            return {
                "attempted": False,
                "reason": "task_board_unavailable",
            }

        supervisor_event_paths = {
            self.config.events_path,
            *self.config.state_dir.parent.glob(
                "*/*_supervisor_events.jsonl"
            ),
        }
        supervisor_events: list[dict[str, Any]] = []
        for event_path in sorted(
            supervisor_event_paths,
            key=lambda path: str(path),
        ):
            for event in self._read_jsonl_events(event_path):
                supervisor_events.append(
                    {
                        **event,
                        "_recovery_source_events_path": str(event_path),
                    }
                )
        supervisor_outcome_keys = {
            str(event.get("recovery_key") or "")
            for event in supervisor_events
            if str(event.get("type") or "")
            == "worktree_reconciliation_replay_finished"
            and event.get("settled") is True
            and str(event.get("recovery_key") or "")
        }
        settled_keys = managed_outcome_keys | supervisor_outcome_keys
        target_ref = (
            self.config.merge_target_branch
            or self._git_current_branch(self.config.repo_root)
            or "HEAD"
        )
        target_commit = self._git_ref_commit(
            self.config.repo_root,
            target_ref,
        )
        if not target_commit:
            return {
                "attempted": False,
                "reason": "reconciliation_target_missing",
                "target_ref": target_ref,
            }

        pending: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for event in reversed(supervisor_events):
            if str(event.get("type") or "") != "worktree_reconciliation":
                continue
            integration_baseline_ref = str(
                event.get("target_signature") or ""
            )
            processed = event.get("processed")
            if (
                not integration_baseline_ref
                or not isinstance(processed, list)
            ):
                continue
            for item in reversed(processed):
                if not isinstance(item, Mapping) or not item.get("merged"):
                    continue
                cleanup_result = item.get("cleanup_result")
                merge_result = item.get("merge_result")
                preflight_result = item.get("preflight_result")
                if (
                    not isinstance(cleanup_result, Mapping)
                    or cleanup_result.get("cleaned") is not True
                    or not isinstance(merge_result, Mapping)
                    or not isinstance(preflight_result, Mapping)
                ):
                    continue
                branch = str(item.get("branch") or "")
                item_path = str(item.get("path") or "")
                candidate_commit = str(item.get("head") or "")
                merge_commit = str(
                    merge_result.get("merge_commit") or ""
                )
                preflight_tree = str(
                    preflight_result.get("tree") or ""
                )
                merge_tree = self._git_commit_tree(
                    self.config.repo_root,
                    merge_commit,
                )
                parents = self._git_commit_parents(
                    self.config.repo_root,
                    merge_commit,
                )
                source_target_ref = str(
                    event.get("target_ref") or ""
                )
                if (
                    not branch
                    or not item_path
                    or not candidate_commit
                    or not merge_commit
                    or source_target_ref != target_ref
                    or str(item.get("target_ref") or "")
                    != source_target_ref
                    or preflight_result.get("attempted") is not True
                    or preflight_result.get("mergeable") is not True
                    or preflight_result.get("returncode") is None
                    or int(preflight_result.get("returncode")) != 0
                    or str(preflight_result.get("branch") or "")
                    != branch
                    or str(preflight_result.get("target_ref") or "")
                    != source_target_ref
                    or merge_result.get("attempted") is not True
                    or merge_result.get("merged") is not True
                    or merge_result.get("returncode") is None
                    or int(merge_result.get("returncode")) != 0
                    or str(merge_result.get("branch") or "")
                    != branch
                    or str(merge_result.get("target_branch") or "")
                    != source_target_ref
                    or str(cleanup_result.get("branch") or "")
                    != branch
                    or str(cleanup_result.get("worktree_path") or "")
                    != item_path
                    or cleanup_result.get("removed_worktree") is not True
                    or cleanup_result.get("deleted_branch") is not True
                    or parents
                    != [integration_baseline_ref, candidate_commit]
                    or not preflight_tree
                    or preflight_tree != merge_tree
                    or not self._git_ref_is_ancestor(
                        self.config.repo_root,
                        candidate_commit,
                        merge_commit,
                    )
                    or not self._git_ref_is_ancestor(
                        self.config.repo_root,
                        merge_commit,
                        target_ref,
                    )
                ):
                    continue
                task = self._current_reconciliation_task(
                    branch=branch,
                    rescued_from_branch=str(
                        item.get("rescued_from_branch") or ""
                    ),
                    tasks_by_id=tasks_by_id,
                    task_ids_by_branch=task_ids_by_branch,
                )
                if (
                    task is None
                    or str(task.status).strip().lower() == "completed"
                ):
                    continue
                provenance_branches = (
                    str(branch).removeprefix("refs/heads/"),
                    str(
                        item.get("rescued_from_branch") or ""
                    ).removeprefix("refs/heads/"),
                )
                implementation_provenance = next(
                    (
                        implementation_provenance_by_branch[
                            provenance_branch
                        ]
                        for provenance_branch in provenance_branches
                        if provenance_branch
                        in implementation_provenance_by_branch
                    ),
                    None,
                )
                if (
                    not isinstance(implementation_provenance, Mapping)
                    or str(
                        implementation_provenance.get("task_id") or ""
                    )
                    != task.task_id
                ):
                    continue
                source_task_key = str(
                    implementation_provenance.get(
                        "canonical_task_key"
                    )
                    or ""
                )
                source_board_namespace = str(
                    implementation_provenance.get(
                        "board_namespace"
                    )
                    or ""
                )
                workspace_setup = implementation_provenance.get(
                    "workspace_setup"
                )
                branch_fingerprint = self._implementation_branch_fingerprint(
                    branch
                )
                proposal_baseline_ref = str(
                    implementation_provenance.get("baseline_ref") or ""
                )
                if (
                    not proposal_baseline_ref
                    or not source_task_key
                    or not source_board_namespace
                    or not branch_fingerprint
                    or not source_task_key.removeprefix(
                        "task/v1/"
                    ).startswith(
                        branch_fingerprint
                    )
                    or str(
                        implementation_provenance.get(
                            "worktree_path"
                        )
                        or ""
                    )
                    != item_path
                    or not isinstance(workspace_setup, Mapping)
                    or str(workspace_setup.get("branch") or "")
                    != branch
                    or str(workspace_setup.get("worktree_path") or "")
                    != item_path
                    or str(workspace_setup.get("base_commit") or "")
                    != proposal_baseline_ref
                    or proposal_baseline_ref == candidate_commit
                    or not self._git_ref_is_ancestor(
                        self.config.repo_root,
                        proposal_baseline_ref,
                        candidate_commit,
                    )
                ):
                    continue
                representation_proof = (
                    self._changed_path_representation_proof(
                        self.config.repo_root,
                        baseline_ref=proposal_baseline_ref,
                        candidate_commit=candidate_commit,
                        integrated_commit=merge_commit,
                    )
                )
                if representation_proof.get("verified") is not True:
                    continue
                identity = daemon._identity_for_task(task)
                recovery_key = (
                    self._worktree_reconciliation_recovery_key(
                        task_cid=identity.canonical_task_cid,
                        baseline_ref=proposal_baseline_ref,
                        candidate_commit=candidate_commit,
                        target_commit=merge_commit,
                        mode="already_merged_replay",
                    )
                )
                if recovery_key in settled_keys or recovery_key in seen_keys:
                    continue
                seen_keys.add(recovery_key)
                pending.append(
                    {
                        "task": task,
                        "task_id": task.task_id,
                        "task_cid": identity.canonical_task_cid,
                        "historical_branch": branch,
                        "historical_candidate_commit": candidate_commit,
                        "baseline_ref": proposal_baseline_ref,
                        "integration_baseline_ref": (
                            integration_baseline_ref
                        ),
                        "merge_commit": merge_commit,
                        "merge_tree": merge_tree,
                        "preflight_tree": preflight_tree,
                        "target_ref": target_ref,
                        "target_commit": target_commit,
                        "source_event_id": str(
                            event.get("event_id") or ""
                        ),
                        "source_events_path": str(
                            event.get(
                                "_recovery_source_events_path"
                            )
                            or ""
                        ),
                        "source_implementation_event_id": str(
                            implementation_provenance.get(
                                "event_id"
                            )
                            or ""
                        ),
                        "source_implementation_events_path": str(
                            implementation_provenance.get(
                                "_reconciliation_source_events_path"
                            )
                            or ""
                        ),
                        "source_implementation_task_cid": str(
                            implementation_provenance.get(
                                "canonical_task_cid"
                            )
                            or implementation_provenance.get(
                                "task_cid"
                            )
                            or ""
                        ),
                        "source_implementation_task_key": (
                            source_task_key
                        ),
                        "source_implementation_board_namespace": (
                            source_board_namespace
                        ),
                        "candidate_representation_proof": (
                            representation_proof
                        ),
                        "recovery_key": recovery_key,
                    }
                )

        results: list[dict[str, Any]] = []
        for candidate in pending[:max_replays]:
            task = candidate["task"]
            task_id = str(candidate["task_id"])
            recovery_key = str(candidate["recovery_key"])
            safe_task_id = "".join(
                character.lower()
                if character.isalnum() or character in {"-", "_"}
                else "-"
                for character in task_id
            ).strip("-") or "reconciled-task"
            stamp = int(time.time())
            replay_branch = (
                "implementation/"
                f"{safe_task_id}-{recovery_key[:12]}-attempt-0-{stamp}"
            )
            replay_worktree = daemon.worktree_root / (
                f"replay-{safe_task_id}-{recovery_key[:12]}-{stamp}"
            )
            claim_path = daemon._implementation_task_claim_path(
                task_id,
                canonical_task_cid=str(candidate["task_cid"]),
            )
            claim_metadata = (
                daemon._build_implementation_task_claim_metadata(
                    task,
                    1,
                    utc_now(),
                )
            )
            candidate_payload = {
                key: value
                for key, value in candidate.items()
                if key != "task"
            }
            acquired = False
            retain_replay_worktree = False
            try:
                acquired, claim_reason, existing_claim = (
                    daemon._try_acquire_implementation_task_claim(
                        claim_path,
                        claim_metadata,
                    )
                )
                if not acquired:
                    deferred = {
                        **candidate_payload,
                        "attempted": False,
                        "completed": False,
                        "settled": False,
                        "reason": f"task_claim_{claim_reason}",
                        "provider_dispatched": False,
                        "attempt_consumed": False,
                        "lock_owner_pid": int(
                            (existing_claim or {}).get("pid") or 0
                        ),
                    }
                    self._record_event(
                        "worktree_reconciliation_replay_deferred",
                        deferred,
                    )
                    results.append(deferred)
                    continue

                replay_worktree.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                add_result = subprocess.run(
                    [
                        "git",
                        "worktree",
                        "add",
                        "-b",
                        replay_branch,
                        str(replay_worktree),
                        str(candidate["historical_candidate_commit"]),
                    ],
                    cwd=self.config.repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if add_result.returncode != 0:
                    deferred = {
                        **candidate_payload,
                        "attempted": False,
                        "completed": False,
                        "settled": False,
                        "reason": "replay_worktree_create_failed",
                        "returncode": add_result.returncode,
                        "stderr": add_result.stderr[-2000:],
                        "provider_dispatched": False,
                        "attempt_consumed": False,
                    }
                    self._record_event(
                        "worktree_reconciliation_replay_deferred",
                        deferred,
                    )
                    results.append(deferred)
                    continue

                self._record_event(
                    "worktree_reconciliation_replay_started",
                    {
                        **candidate_payload,
                        "replay_branch": replay_branch,
                        "replay_worktree": str(replay_worktree),
                        "provider_dispatched": False,
                        "attempt_consumed": False,
                    },
                )
                recovery_result = (
                    daemon.reconcile_validated_worktree_candidate(
                        worktree_path=replay_worktree,
                        branch_name=replay_branch,
                        task=task,
                        baseline_ref=str(candidate["baseline_ref"]),
                        candidate_commit=str(
                            candidate[
                                "historical_candidate_commit"
                            ]
                        ),
                        changed_submodule_paths=(),
                        recovery_key=recovery_key,
                        preacquired_task_claim=claim_metadata,
                        preacquired_implementation_lock=(
                            preacquired_implementation_lock
                        ),
                    )
                )
                recovery_returncode = recovery_result.get("returncode")
                recovery_merge_result = (
                    recovery_result.get("merge_result") or {}
                )
                completed = bool(
                    recovery_returncode is not None
                    and int(recovery_returncode) == 0
                    and recovery_merge_result.get("merged") is True
                )
                queued = bool(
                    recovery_merge_result.get("queued") is True
                    and str(
                        recovery_merge_result.get("request_id") or ""
                    )
                )
                settled = completed or queued
                retain_replay_worktree = queued
                result = {
                    **candidate_payload,
                    "attempted": True,
                    "completed": completed,
                    "queued": queued,
                    "settled": settled,
                    "provider_dispatched": False,
                    "attempt_consumed": False,
                    "replay_branch": replay_branch,
                    "replay_worktree": str(replay_worktree),
                    "recovery_result": recovery_result,
                }
                self._record_event(
                    (
                        "worktree_reconciliation_replay_finished"
                        if settled
                        else "worktree_reconciliation_replay_deferred"
                    ),
                    result,
                )
                results.append(result)
            except Exception as exc:
                deferred = {
                    **candidate_payload,
                    "attempted": False,
                    "completed": False,
                    "settled": False,
                    "reason": "reconciliation_replay_exception",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[-2000:],
                    "provider_dispatched": False,
                    "attempt_consumed": False,
                    "replay_branch": replay_branch,
                    "replay_worktree": str(replay_worktree),
                }
                self._record_event(
                    "worktree_reconciliation_replay_deferred",
                    deferred,
                )
                results.append(deferred)
            finally:
                if acquired and not retain_replay_worktree:
                    daemon._cleanup_merged_worktree(
                        replay_worktree,
                        replay_branch,
                        reusable=False,
                    )
                daemon._release_implementation_task_claim(
                    claim_path,
                    claim_metadata,
                )

        return {
            "attempted": any(
                result.get("attempted") for result in results
            ),
            "reason": (
                "reconciliation_replays_processed"
                if results
                else "no_pending_reconciliation_replays"
            ),
            "target_ref": target_ref,
            "target_commit": target_commit,
            "pending_count": len(pending),
            "processed_count": sum(
                1 for result in results if result.get("attempted")
            ),
            "completed_count": sum(
                1 for result in results if result.get("completed")
            ),
            "failed_count": sum(
                1
                for result in results
                if result.get("attempted")
                and not result.get("completed")
                and not result.get("settled")
                and not (
                    (result.get("recovery_result") or {}).get("skipped")
                )
            ),
            "deferred_count": sum(
                1
                for result in results
                if not result.get("attempted")
                or not result.get("settled")
                or (result.get("recovery_result") or {}).get("skipped")
            ),
            "results": results,
        }

    def _preflight_worktree_reconciliation_merge(
        self,
        repo_root: Path,
        *,
        target_ref: str,
        branch: str,
    ) -> dict[str, Any]:
        """Check branch mergeability without mutating the main checkout."""

        command = ["git", "merge-tree", "--write-tree", target_ref, branch]
        started_at = utc_now()
        result = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        payload: dict[str, Any] = {
            "attempted": True,
            "mergeable": result.returncode == 0,
            "returncode": result.returncode,
            "target_ref": target_ref,
            "branch": branch,
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
        if result.returncode == 0:
            payload["tree"] = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            return payload
        payload["reason"] = "preflight_merge_conflict"
        payload["conflict_paths"] = self._merge_tree_conflict_paths(output)
        return payload

    @staticmethod
    def _merge_tree_conflict_paths(output: str) -> list[str]:
        paths: list[str] = []
        for line in output.splitlines():
            path = ""
            if "Merge conflict in " in line:
                path = line.rsplit("Merge conflict in ", 1)[-1].strip()
            elif line.startswith("CONFLICT ") and " in " in line:
                path = line.rsplit(" in ", 1)[-1].strip()
            if path and path not in paths:
                paths.append(path)
        return paths

    def _main_checkout_dirty_evidence(self, repo_root: Path, status_lines: list[str]) -> dict[str, Any]:
        """Return bounded evidence for dirty main-checkout reconciliation blockers."""

        path_categories: dict[str, int] = {}
        status_paths: list[str] = []
        for line in status_lines:
            path = self._status_line_path(line)
            if path and path not in status_paths:
                status_paths.append(path)
            category = self._status_line_category(line)
            path_categories[category] = path_categories.get(category, 0) + 1
        evidence: dict[str, Any] = {
            "status_short": status_lines[:50],
            "status_paths": status_paths[:50],
            "path_categories": path_categories,
        }
        diff_stat = self._git_output(repo_root, ["diff", "--stat"], max_chars=4000)
        if diff_stat:
            evidence["diff_stat"] = diff_stat
        name_status = self._git_output(repo_root, ["diff", "--name-status"], max_chars=4000)
        if name_status:
            evidence["name_status"] = name_status
        staged_name_status = self._git_output(repo_root, ["diff", "--cached", "--name-status"], max_chars=4000)
        if staged_name_status:
            evidence["staged_name_status"] = staged_name_status
        submodule_summary = self._git_output(repo_root, ["submodule", "summary", "--files"], max_chars=4000)
        if submodule_summary:
            evidence["submodule_summary"] = submodule_summary
        untracked_paths = [
            self._status_line_path(line)
            for line in status_lines
            if line[:2] == "??" and self._status_line_path(line)
        ][:50]
        if untracked_paths:
            evidence["untracked_paths"] = untracked_paths
        return evidence

    def _generated_main_checkout_status_filters(
        self,
        *,
        additional_paths: Sequence[Path] = (),
    ) -> tuple[list[str], list[str]]:
        """Return supervisor-generated dirty paths that should not block reconciliation."""

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            generated_guardrail_status_filters,
        )

        discovery_dir = self._reconciliation_guardrail_discovery_dir()
        additional_paths = [
            path
            for path in (
                self.config.objective_path,
                self.config.objective_graph_path,
                self.config.objective_todo_vector_index_path,
                *self.config.generated_dirty_repair_paths,
                *additional_paths,
            )
            if path is not None
        ]
        additional_prefixes = [
            path
            for path in (
                self.config.retry_budget_discovery_dir,
                self.config.dependency_guardrail_discovery_dir,
                self.config.reconciliation_guardrail_discovery_dir,
                self.config.codebase_scan_discovery_dir,
                self.config.objective_bundle_dir,
                self.config.objective_dataset_dir,
                self.config.objective_discovery_dir,
                self.config.state_dir,
            )
            if path is not None
        ]
        return generated_guardrail_status_filters(
            todo_path=self.config.todo_path,
            discovery_dir=discovery_dir,
            repo_root=self.config.repo_root,
            additional_generated_paths=additional_paths,
            additional_generated_prefixes=additional_prefixes,
        )

    def repair_generated_dirty_checkouts(
        self,
        *,
        force: bool = False,
        additional_paths: Sequence[Path] = (),
    ) -> dict[str, Any]:
        """Commit safe generated supervisor outputs so reconciliation can proceed."""

        retained_recovery = self._retained_generated_checkout_lease()
        if (
            not self.config.generated_dirty_repair_enabled
            and not force
            and not retained_recovery
        ):
            return {"attempted": False, "reason": "generated_dirty_repair_disabled"}
        generated_paths, generated_prefixes = (
            self._generated_main_checkout_status_filters(
                additional_paths=additional_paths,
            )
        )
        candidate_git_roots = [
            self.config.repo_root / relative
            for relative in self.config.worktree_submodule_paths
            if str(relative).strip()
        ]
        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            commit_generated_dirty_outputs,
        )

        commit_subject = generated_protected_board_commit_subject(
            self.config.generated_dirty_repair_commit_subject
        )
        result = self._run_generated_board_producer(
            producer="generated-dirty-repair",
            commit_outputs=True,
            operation="generated_dirty_repair",
            callback=lambda: commit_generated_dirty_outputs(
                repo_root=self.config.repo_root,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
                protected_paths=self.config.implementation_protected_paths,
                candidate_git_roots=candidate_git_roots,
                subject=commit_subject,
                include_clean_submodule_gitlinks=(
                    self.config.generated_dirty_repair_include_submodule_gitlinks
                ),
                max_paths=self.config.generated_dirty_repair_max_paths,
                stale_git_lock_seconds=(
                    self.config.generated_dirty_repair_stale_lock_seconds
                ),
            ),
        )
        if not isinstance(result, Mapping):
            return {
                "attempted": False,
                "reason": "checkout_mutation_lock_unavailable",
                "lock_path": str(self._repo_merge_lock_path()),
            }
        result = dict(result)
        if result.get("committed_count") or result.get("selected_path_count"):
            self._record_event("generated_dirty_checkout_repair", result)
        return result

    def _filter_generated_main_checkout_status(
        self,
        status_lines: list[str],
        evidence: Mapping[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        """Filter deterministic supervisor output from main dirty evidence."""

        if not status_lines:
            return [], dict(evidence or {})
        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            filter_generated_main_checkout_evidence,
        )

        generated_paths, generated_prefixes = self._generated_main_checkout_status_filters()
        return filter_generated_main_checkout_evidence(
            status_short=status_lines,
            evidence=evidence,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        )

    def _generated_only_dirty_worktree_status(
        self,
        worktree_path: Path,
        status_lines: list[str],
    ) -> dict[str, Any]:
        """Classify stale worktree dirt that only touches supervisor-generated outputs."""

        if not status_lines:
            return {}
        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            path_is_generated_status_output,
        )

        generated_paths, generated_prefixes = self._generated_main_checkout_status_filters()
        checked: list[dict[str, Any]] = []
        for line in status_lines:
            code = line[:2]
            relative = self._status_line_path(line)
            detail = {"status": code, "path": relative}
            if not relative:
                return {}
            if "U" in code or "R" in code or "C" in code:
                return {}
            if not (
                code == "??"
                or "M" in code
                or "A" in code
                or "D" in code
            ):
                return {}
            if path_is_generated_status_output(
                relative,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                checked.append({**detail, "generated_status_output": True})
                continue
            expanded_untracked_paths = self._expand_untracked_generated_status_dir(
                worktree_path,
                relative,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            )
            if not expanded_untracked_paths:
                return {}
            checked.append(
                {
                    **detail,
                    "generated_status_output": True,
                    "expanded_untracked_paths": expanded_untracked_paths[:50],
                }
            )
        if not checked:
            return {}
        return {
            "redundant": True,
            "reason": "generated_only_status_paths_dropped",
            "checked": checked,
        }

    @staticmethod
    def _expand_untracked_generated_status_dir(
        worktree_path: Path,
        relative: str,
        *,
        generated_paths: list[str],
        generated_prefixes: list[str],
    ) -> list[str]:
        """Expand a collapsed untracked directory if all contained files are generated outputs."""

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            path_is_generated_status_output,
        )

        candidate = worktree_path / relative
        if not candidate.is_dir():
            return []
        expanded: list[str] = []
        try:
            children = sorted(candidate.rglob("*"))
        except OSError:
            return []
        for child in children:
            if child.is_dir():
                continue
            if not child.is_file():
                return []
            try:
                child_relative = child.relative_to(worktree_path).as_posix()
            except ValueError:
                return []
            if not path_is_generated_status_output(
                child_relative,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                return []
            expanded.append(child_relative)
        return expanded

    @staticmethod
    def _status_line_category(line: str) -> str:
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

    def _build_worktree_reconciliation_daemon(self) -> PortalImplementationDaemon:
        return PortalImplementationDaemon(
            todo_path=self.config.todo_path,
            state_path=self.config.state_path,
            strategy_path=self.config.strategy_path,
            # Recovery must emit proposal, validation, queue, merge, and
            # completion receipts into the managed daemon stream consumed by
            # scheduling.  The supervisor stream is maintenance telemetry.
            events_path=(
                self.config.state_dir
                / f"{self.config.state_prefix}_events.jsonl"
            ),
            repo_root=self.config.repo_root,
            task_header_prefix=self.config.task_prefix,
            # Revalidation-only construction requires implement=True, while
            # its daemon policy still forbids every model/provider seam. An
            # ordinary reconciliation helper remains non-implementing.
            implement=bool(
                self.config.manual_completion_authority_revalidation_only
            ),
            implementation_command=self.config.implementation_command,
            implementation_timeout=self.config.implementation_timeout,
            max_task_attempts=self.config.max_task_attempts,
            use_ephemeral_worktree=False,
            worktree_root=self.config.worktree_root,
            merge_target_branch=self.config.merge_target_branch,
            merge_queue_dir=self.config.merge_queue_dir,
            worktree_submodule_paths=self.config.worktree_submodule_paths,
            implementation_protected_paths=(
                self.config.implementation_protected_paths
            ),
            manual_completion_authority_task_ids=(
                self.config.manual_completion_authority_task_ids
            ),
            manual_completion_authority_required_task_ids=(
                self.config.manual_completion_authority_required_task_ids
            ),
            manual_completion_authority_epoch_id=(
                self.config.manual_completion_authority_epoch_id
            ),
            manual_completion_authority_revalidation_only=(
                self.config.manual_completion_authority_revalidation_only
            ),
            objective_path=self.config.objective_path,
            objective_bundle_dir=self.config.objective_bundle_dir,
            generated_status_paths=self.config.generated_dirty_repair_paths,
            llm_merge_resolver_command=self.config.llm_merge_resolver_command,
            llm_merge_resolver_timeout_seconds=self.config.llm_merge_resolver_timeout_seconds,
        )

    @staticmethod
    def _worktree_reconciliation_recovery_key(
        *,
        task_cid: str,
        baseline_ref: str,
        candidate_commit: str,
        target_commit: str,
        mode: str,
    ) -> str:
        payload = {
            "task_cid": str(task_cid),
            "baseline_ref": str(baseline_ref),
            "candidate_commit": str(candidate_commit),
            "target_commit": str(target_commit),
            "mode": str(mode),
        }
        return sha1(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _git_merge_base(
        repo_root: Path,
        left: str,
        right: str,
    ) -> str:
        result = subprocess.run(
            ["git", "merge-base", left, right],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def _git_commit_parents(repo_root: Path, commit: str) -> list[str]:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%P", commit],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        return result.stdout.strip().split()

    @staticmethod
    def _git_commit_tree(repo_root: Path, commit: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", f"{commit}^{{tree}}"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def _git_tree_entry(
        repo_root: Path,
        ref: str,
        path: str,
    ) -> str:
        result = subprocess.run(
            ["git", "ls-tree", "-z", ref, "--", path],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.decode(
            "utf-8",
            errors="surrogateescape",
        ).rstrip("\0")

    @classmethod
    def _changed_path_representation_proof(
        cls,
        repo_root: Path,
        *,
        baseline_ref: str,
        candidate_commit: str,
        integrated_commit: str,
    ) -> dict[str, Any]:
        """Prove the candidate's entire path projection survived integration."""

        diff = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                baseline_ref,
                candidate_commit,
                "--",
            ],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if diff.returncode != 0:
            return {
                "verified": False,
                "reason": "candidate_changed_paths_unavailable",
                "returncode": diff.returncode,
            }
        paths = [
            path
            for path in diff.stdout.decode(
                "utf-8",
                errors="surrogateescape",
            ).split("\0")
            if path
        ]
        if not paths:
            return {
                "verified": False,
                "reason": "candidate_proposal_empty",
                "changed_path_count": 0,
            }
        candidate_entries = {
            path: cls._git_tree_entry(
                repo_root,
                candidate_commit,
                path,
            )
            for path in paths
        }
        integrated_entries = {
            path: cls._git_tree_entry(
                repo_root,
                integrated_commit,
                path,
            )
            for path in paths
        }
        mismatched_paths = [
            path
            for path in paths
            if candidate_entries[path] != integrated_entries[path]
        ]
        fingerprint_material = json.dumps(
            candidate_entries,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "verified": not mismatched_paths,
            "reason": (
                "candidate_paths_preserved"
                if not mismatched_paths
                else "candidate_paths_changed_during_integration"
            ),
            "changed_path_count": len(paths),
            "changed_paths": paths[:100],
            "mismatched_paths": mismatched_paths[:100],
            "representation_digest": sha1(
                fingerprint_material.encode(
                    "utf-8",
                    errors="surrogateescape",
                )
            ).hexdigest(),
        }

    @staticmethod
    def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _reconciliation_task_context(
        self,
        daemon: PortalImplementationDaemon,
    ) -> tuple[
        dict[str, PortalTask],
        dict[str, str],
        set[str],
        dict[str, dict[str, Any]],
    ]:
        try:
            tasks = daemon._load_tasks()
        except Exception:
            return {}, {}, set(), {}
        daemon._register_task_identities(tasks)
        tasks_by_id = {task.task_id: task for task in tasks}
        task_ids_by_branch: dict[str, str] = {}
        outcome_keys: set[str] = set()
        provenance_by_branch: dict[str, dict[str, Any]] = {}
        managed_events = list(daemon._iter_events())
        current_events_path = (
            self.config.state_dir
            / f"{self.config.state_prefix}_events.jsonl"
        )
        sibling_event_paths = {
            path
            for path in self.config.state_dir.parent.glob(
                "*/*_events.jsonl"
            )
            if not path.name.endswith("_supervisor_events.jsonl")
            and path != current_events_path
        }
        for event_path in sorted(
            sibling_event_paths,
            key=lambda path: str(path),
        ):
            managed_events.extend(
                {
                    **event,
                    "_reconciliation_source_events_path": str(event_path),
                }
                for event in self._read_jsonl_events(event_path)
            )
        for event in managed_events:
            event_type = str(event.get("type") or "")
            task_id = str(event.get("task_id") or "")
            branch = str(event.get("branch") or "").removeprefix(
                "refs/heads/"
            )
            if (
                task_id in tasks_by_id
                and branch
                and event_type
                in {
                    "implementation_started",
                    "implementation_finished",
                    "worktree_reconciliation_validation_started",
                    "worktree_reconciliation_validation_finished",
                    "worktree_reconciliation_candidate_queued",
                }
            ):
                task_ids_by_branch[branch] = task_id
            if (
                event_type == "implementation_started"
                and task_id in tasks_by_id
                and branch
                and str(event.get("baseline_ref") or "")
            ):
                existing = provenance_by_branch.get(branch)
                if (
                    existing is None
                    or str(event.get("timestamp") or "")
                    >= str(existing.get("timestamp") or "")
                ):
                    provenance_by_branch[branch] = dict(event)
            recovery_key = str(event.get("recovery_key") or "")
            merge_result = event.get("merge_result")
            event_returncode = event.get("returncode")
            completed_reconciliation = bool(
                event_type == "implementation_finished"
                and event_returncode is not None
                and int(event_returncode) == 0
                and isinstance(merge_result, Mapping)
                and merge_result.get("merged") is True
            )
            durable_queue_handoff = bool(
                event_type
                == "worktree_reconciliation_candidate_queued"
                and isinstance(merge_result, Mapping)
                and merge_result.get("queued") is True
                and str(merge_result.get("request_id") or "")
            )
            validation_result = event.get("validation_result")
            validation_reason = (
                str(validation_result.get("reason") or "")
                if isinstance(validation_result, Mapping)
                else ""
            )
            proposal_gate = (
                validation_result.get("proposal_gate")
                if isinstance(validation_result, Mapping)
                else None
            )
            proposal_rejected = bool(
                isinstance(proposal_gate, Mapping)
                and proposal_gate.get("attempted") is True
                and proposal_gate.get("accepted") is False
            )
            proposal_reason_codes = {
                str(code).strip()
                for code in (
                    proposal_gate.get("reason_codes") or ()
                    if isinstance(proposal_gate, Mapping)
                    else ()
                )
                if str(code).strip()
            }
            replay_control_rejection = bool(
                proposal_reason_codes == {"stale_proposal_replay"}
            )
            retryable_event_failure = getattr(
                daemon,
                "_retryable_reconciliation_event_failure",
                None,
            )
            if callable(retryable_event_failure):
                retryable_environment_failure = bool(
                    retryable_event_failure(event)
                )
            else:
                retryable_environment_failure = bool(
                    PortalImplementationDaemon
                    ._retryable_reconciliation_validation_failure(
                        validation_result
                        if isinstance(validation_result, Mapping)
                        else {}
                    )
                )
            terminal_semantic_rejection = bool(
                event_type
                == "worktree_reconciliation_validation_finished"
                and isinstance(validation_result, Mapping)
                and (
                    validation_result.get("attempted") is True
                    or proposal_rejected
                )
                and validation_result.get("passed") is False
                and validation_reason
                not in {
                    "reconciliation_validation_exception",
                    "reconciled_candidate_handoff_failed",
                    "reconciled_candidate_task_revision_changed",
                    "merge_train_consumer_unavailable",
                }
                and not validation_result.get("error_type")
                and not replay_control_rejection
                and not retryable_environment_failure
            )
            if recovery_key and (
                completed_reconciliation
                or durable_queue_handoff
                or terminal_semantic_rejection
            ):
                outcome_keys.add(recovery_key)
        return (
            tasks_by_id,
            task_ids_by_branch,
            outcome_keys,
            provenance_by_branch,
        )

    def _current_reconciliation_task(
        self,
        *,
        branch: str,
        rescued_from_branch: str = "",
        tasks_by_id: Mapping[str, PortalTask],
        task_ids_by_branch: Mapping[str, str],
    ) -> PortalTask | None:
        normalized_branches = tuple(
            candidate_branch.removeprefix("refs/heads/")
            for candidate_branch in (branch, rescued_from_branch)
            if candidate_branch
        )
        for candidate_branch in normalized_branches:
            task_id = str(task_ids_by_branch.get(candidate_branch) or "")
            task = tasks_by_id.get(task_id)
            if task is not None:
                return task
        fallback = self._worktree_reconciliation_task(
            normalized_branches[-1]
            if rescued_from_branch and normalized_branches
            else (
                normalized_branches[0]
                if normalized_branches
                else branch
            ),
            known_task_ids=tuple(tasks_by_id),
        )
        return tasks_by_id.get(fallback.task_id)

    @staticmethod
    def _reconciliation_cleanup_result(
        merge_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        direct = merge_result.get("cleanup_result")
        if isinstance(direct, Mapping):
            return dict(direct)
        train_result = merge_result.get("train_result")
        if not isinstance(train_result, Mapping):
            return {}
        callback_result = train_result.get("merge_result")
        if not isinstance(callback_result, Mapping):
            return {}
        nested = callback_result.get("cleanup_result")
        return dict(nested) if isinstance(nested, Mapping) else {}

    def _reconcile_interrupted_implementation_after_shutdown(
        self,
    ) -> dict[str, Any]:
        """Close an interrupted attempt only after proving it is quiescent."""

        try:
            daemon = self._build_worktree_reconciliation_daemon()
            return daemon.reconcile_quiesced_active_attempt()
        except Exception as exc:
            logger.exception(
                "Could not reconcile interrupted implementation during "
                "supervisor shutdown"
            )
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "shutdown_attempt_reconciliation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def _reconciliation_guardrail_discovery_dir(self) -> Path:
        return (
            self.config.reconciliation_guardrail_discovery_dir
            or self.config.dependency_guardrail_discovery_dir
            or self.config.retry_budget_discovery_dir
            or self.config.state_dir.parent / "discovery"
        )

    def _main_status_for_worktree_reconciliation(self, repo_root: Path, worktree_root: Path) -> list[str]:
        status = self._git_status_short_strict(repo_root)
        try:
            root_relative = worktree_root.resolve().relative_to(repo_root.resolve()).as_posix().rstrip("/")
        except (OSError, ValueError):
            return status
        if not root_relative:
            return status
        return [
            line
            for line in status
            if not self._status_line_targets_prefix(line, root_relative)
        ]

    @staticmethod
    def _stage_zero_gitlink(
        repo_root: Path,
        relative: str,
    ) -> str:
        """Return one exact stage-zero gitlink object, or an empty proof."""

        result = subprocess.run(
            ["git", "ls-files", "--stage", "-z", "--", relative],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return ""
        raw = bytes(result.stdout or b"")
        if not raw.endswith(b"\0"):
            return ""
        records = [record for record in raw[:-1].split(b"\0") if record]
        if len(records) != 1:
            return ""
        metadata, separator, raw_path = records[0].partition(b"\t")
        try:
            fields = metadata.decode("ascii", errors="strict").split()
            path = raw_path.decode("utf-8", errors="surrogateescape")
        except UnicodeError:
            return ""
        if (
            not separator
            or len(fields) != 3
            or fields[0] != "160000"
            or fields[2] != "0"
            or path != relative
        ):
            return ""
        return fields[1]

    @staticmethod
    def _gitlink_at_ref(
        repo_root: Path,
        ref: str,
        relative: str,
    ) -> str:
        """Return an exact gitlink object at ``ref``, or an empty proof."""

        if not ref:
            return ""
        result = subprocess.run(
            ["git", "ls-tree", "-z", ref, "--", relative],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return ""
        raw = bytes(result.stdout or b"")
        if not raw.endswith(b"\0"):
            return ""
        records = [record for record in raw[:-1].split(b"\0") if record]
        if len(records) != 1:
            return ""
        metadata, separator, raw_path = records[0].partition(b"\t")
        try:
            fields = metadata.decode("ascii", errors="strict").split()
            path = raw_path.decode("utf-8", errors="surrogateescape")
        except UnicodeError:
            return ""
        if (
            not separator
            or len(fields) != 3
            or fields[0] != "160000"
            or fields[1] != "commit"
            or path != relative
        ):
            return ""
        return fields[2]

    @staticmethod
    def _unique_git_merge_base(
        repo_root: Path,
        left: str,
        right: str,
    ) -> str:
        """Return the sole merge base, failing closed for criss-cross bases."""

        result = subprocess.run(
            ["git", "merge-base", "--all", left, right],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            return ""
        merge_bases = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        if len(merge_bases) != 1:
            return ""
        resolved = PortalImplementationSupervisor._git_ref_commit(
            repo_root,
            merge_bases[0],
        )
        return resolved if resolved == merge_bases[0] else ""

    def _candidate_submodule_content_status_proof(
        self,
        repo_root: Path,
        status_line: str,
        *,
        target_ref: str,
        branch: str,
        candidate_head: str,
    ) -> dict[str, Any]:
        """Prove one lowercase submodule-content status is merge-independent.

        Only porcelain ``" m"`` is eligible: the superproject index and
        gitlink are unchanged while files below the nested checkout are dirty.
        The candidate must preserve that gitlink relative to its unique merge
        base.  Every missing or ambiguous identity leaves the line blocking.
        """

        proof: dict[str, Any] = {
            "status": status_line,
            "path": self._status_line_path(status_line),
            "nonblocking": False,
        }
        if len(status_line) < 4 or status_line[:3] != " m ":
            proof["reason"] = "status_not_submodule_content_only"
            return proof
        relative = status_line[3:]
        proof["path"] = relative
        if (
            not relative
            or relative != relative.strip()
            or relative.startswith("/")
            or "\0" in relative
            or ".." in Path(relative).parts
            or " -> " in relative
        ):
            proof["reason"] = "status_path_ambiguous"
            return proof

        target_commit = self._git_ref_commit(repo_root, target_ref)
        checkout_commit = self._git_ref_commit(repo_root, "HEAD")
        branch_commit = self._git_ref_commit(repo_root, branch)
        candidate_commit = self._git_ref_commit(
            repo_root,
            candidate_head,
        )
        proof.update(
            {
                "target_commit": target_commit,
                "checkout_commit": checkout_commit,
                "branch_commit": branch_commit,
                "candidate_commit": candidate_commit,
            }
        )
        if not all(
            (
                target_commit,
                checkout_commit,
                branch_commit,
                candidate_commit,
            )
        ):
            proof["reason"] = "commit_identity_unavailable"
            return proof
        if target_commit != checkout_commit:
            proof["reason"] = "target_checkout_identity_mismatch"
            return proof
        if branch_commit != candidate_commit:
            proof["reason"] = "candidate_branch_identity_mismatch"
            return proof

        merge_base = self._unique_git_merge_base(
            repo_root,
            target_commit,
            candidate_commit,
        )
        proof["merge_base"] = merge_base
        if not merge_base:
            proof["reason"] = "unique_merge_base_unavailable"
            return proof

        index_gitlink = self._stage_zero_gitlink(repo_root, relative)
        target_gitlink = self._gitlink_at_ref(
            repo_root,
            target_commit,
            relative,
        )
        baseline_gitlink = self._gitlink_at_ref(
            repo_root,
            merge_base,
            relative,
        )
        candidate_gitlink = self._gitlink_at_ref(
            repo_root,
            candidate_commit,
            relative,
        )
        proof.update(
            {
                "index_gitlink": index_gitlink,
                "target_gitlink": target_gitlink,
                "baseline_gitlink": baseline_gitlink,
                "candidate_gitlink": candidate_gitlink,
            }
        )
        if not all(
            (
                index_gitlink,
                target_gitlink,
                baseline_gitlink,
                candidate_gitlink,
            )
        ):
            proof["reason"] = "gitlink_identity_unavailable"
            return proof
        if index_gitlink != target_gitlink:
            proof["reason"] = "index_gitlink_staged_or_mismatched"
            return proof

        staged = subprocess.run(
            [
                "git",
                "diff",
                "--cached",
                "--quiet",
                target_commit,
                "--",
                relative,
            ],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if staged.returncode != 0:
            proof["reason"] = (
                "index_gitlink_staged"
                if staged.returncode == 1
                else "index_gitlink_status_unavailable"
            )
            return proof

        nested_root = repo_root / relative
        nested_head = self._git_ref_commit(nested_root, "HEAD")
        proof["nested_checkout_commit"] = nested_head
        if not nested_head or nested_head != index_gitlink:
            proof["reason"] = "nested_checkout_gitlink_mismatch"
            return proof
        if candidate_gitlink != baseline_gitlink:
            proof["reason"] = "candidate_changes_gitlink"
            return proof

        proof["nonblocking"] = True
        proof["reason"] = "candidate_preserves_content_dirty_gitlink"
        return proof

    def _candidate_main_checkout_status(
        self,
        repo_root: Path,
        status_lines: Sequence[str],
        *,
        target_ref: str,
        branch: str,
        candidate_head: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Split shared-checkout status for one immutable candidate."""

        blocking: list[str] = []
        nonblocking: list[dict[str, Any]] = []
        for status_line in status_lines:
            proof = self._candidate_submodule_content_status_proof(
                repo_root,
                status_line,
                target_ref=target_ref,
                branch=branch,
                candidate_head=candidate_head,
            )
            if proof.get("nonblocking") is True:
                nonblocking.append(proof)
            else:
                blocking.append(status_line)
        return blocking, nonblocking

    @staticmethod
    def _status_line_targets_prefix(line: str, relative_prefix: str) -> bool:
        path_text = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[-1].strip()
        path_text = path_text.rstrip("/")
        return path_text == relative_prefix or path_text.startswith(f"{relative_prefix}/")

    @staticmethod
    def _implementation_branch_fingerprint(branch: str) -> str:
        normalized = branch.removeprefix("refs/heads/")
        task_fragment = normalized.removeprefix(
            "implementation/"
        ).split("-attempt-", 1)[0]
        fingerprint = task_fragment.rsplit("-", 1)[-1].lower()
        if (
            len(fingerprint) == 12
            and all(
                character in "0123456789abcdef"
                for character in fingerprint
            )
        ):
            return fingerprint
        return ""

    @staticmethod
    def _worktree_branch_source_task_id(
        branch: str,
        *,
        known_task_ids: Sequence[str] = (),
    ) -> str:
        """Resolve a branch to the longest task ID known by the active board."""

        branch_text = str(branch or "").strip()
        for task_id in sorted(
            (str(item).strip() for item in known_task_ids if str(item).strip()),
            key=lambda item: (-len(item), item),
        ):
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(task_id)}(?![A-Za-z0-9])",
                branch_text,
                flags=re.IGNORECASE,
            ):
                return task_id
        # Preserve useful behavior for legacy boards that cannot be read while
        # avoiding rescue prefixes and attempt counters as synthetic task IDs.
        fallback = re.search(
            r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_]*-\d+)(?![A-Za-z0-9])",
            branch_text,
        )
        return fallback.group(1).upper() if fallback else "WORKTREE-RECONCILE"

    @staticmethod
    def _worktree_reconciliation_task(
        branch: str,
        *,
        known_task_ids: Sequence[str] = (),
    ) -> PortalTask:
        task_id = PortalImplementationSupervisor._worktree_branch_source_task_id(
            branch,
            known_task_ids=known_task_ids,
        )
        return PortalTask(
            task_id=task_id,
            title=f"Reconcile backlogged implementation branch {branch}",
            status="todo",
            completion="manual",
            priority="P2",
            track="ops",
        )

    @staticmethod
    def _worktree_branch_is_reconcilable(branch: str) -> bool:
        return branch.startswith("implementation/") or branch.startswith("rescue/worktree/")

    @staticmethod
    def _worktree_branch_can_delete_after_merge(branch: str) -> bool:
        return PortalImplementationSupervisor._worktree_branch_is_reconcilable(branch)

    def _shared_active_worktree_owners(
        self,
        worktree_root: Path,
    ) -> dict[Path, dict[str, str]]:
        """Return durable worktree claims from every sibling lane and pool.

        A provider process can exit a few seconds before its daemon validates
        and commits the candidate. Process inspection alone therefore has a
        destructive false-negative window. The task state and protected-path
        snapshot remain durable throughout that handoff and are authoritative
        reasons for every supervisor sharing the worktree root to stand down.
        Idle pool entries are also durable prepared assets: generic supervisor
        cleanup must not remove them behind the pool's state machine.
        """

        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root

        owners: dict[Path, dict[str, str]] = {}

        def register(raw_path: object, **metadata: object) -> None:
            path_text = str(raw_path or "").strip()
            if not path_text:
                return
            try:
                resolved = Path(path_text).resolve()
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                return
            owners[resolved] = {
                key: str(value or "")
                for key, value in metadata.items()
            }

        state_paths = {self.config.state_path}
        namespace_root = self.config.state_path.parent.parent
        try:
            sibling_dirs = [
                path
                for path in namespace_root.iterdir()
                if path.is_dir()
            ]
        except OSError:
            sibling_dirs = [self.config.state_path.parent]

        for state_dir in sibling_dirs:
            try:
                state_paths.update(state_dir.glob("*task_state.json"))
            except OSError:
                continue

        for state_path in sorted(state_paths):
            try:
                state = PortalTaskState.load(state_path)
            except Exception:
                continue
            if not state.implementation_in_progress:
                continue
            register(
                state.active_worktree_path,
                source="task_state",
                state_path=state_path,
                task_id=state.active_task_id,
                branch=state.active_branch,
            )

        for state_dir in sibling_dirs:
            snapshot_path = (
                state_dir / IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME
            )
            payload = load_json_dict(snapshot_path)
            if not payload:
                continue
            register(
                payload.get("workspace_path"),
                source="protected_path_snapshot",
                snapshot_path=snapshot_path,
                task_id=payload.get("task_id"),
            )

        pool_state_root = root_resolved / ".pool-state"
        try:
            pool_state_paths = sorted(pool_state_root.glob("*.json"))
        except OSError:
            pool_state_paths = []
        for pool_state_path in pool_state_paths:
            payload = load_json_dict(pool_state_path)
            if not payload or payload.get("schema") != WORKTREE_POOL_SCHEMA:
                continue
            lease_state = str(payload.get("state") or "")
            try:
                lease_pid = int(payload.get("lease_pid") or 0)
            except (TypeError, ValueError):
                lease_pid = 0
            lock_path = pool_state_path.with_suffix(".lock")
            lock_payload = load_json_dict(lock_path)
            try:
                lock_pid = int((lock_payload or {}).get("pid") or 0)
            except (TypeError, ValueError):
                lock_pid = 0
            live_owner_pid = 0
            if lease_state == "idle":
                # An idle entry intentionally has no owner PID or lock. Its
                # pool-state record, rather than process liveness, owns the
                # detached checkout until WorktreePool reuses or invalidates it.
                pass
            elif lease_state in {"initializing", "leased"} and pid_is_alive(lease_pid):
                live_owner_pid = lease_pid
            elif pid_is_alive(lock_pid):
                live_owner_pid = lock_pid
            if lease_state != "idle" and not live_owner_pid:
                continue
            register(
                payload.get("path"),
                source="worktree_pool_lease",
                pool_state_path=pool_state_path,
                lease_state=lease_state,
                lease_pid=live_owner_pid,
                branch=payload.get("branch"),
            )

        return owners

    def _active_worktree_skip_detail(
        self,
        path: Path,
        owners: Mapping[Path, Mapping[str, str]],
    ) -> dict[str, str] | None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        owner = owners.get(resolved)
        if owner is None:
            return None
        own_state_path = str(self.config.state_path.resolve())
        owner_state_path = str(owner.get("state_path") or "")
        owner_snapshot_path = str(owner.get("snapshot_path") or "")
        own_snapshot_path = str(
            (
                self.config.state_path.parent
                / IMPLEMENTATION_PROTECTED_ACTIVE_SNAPSHOT_FILENAME
            ).resolve()
        )
        own_lane = (
            owner_state_path == own_state_path
            or owner_snapshot_path == own_snapshot_path
        )
        owner_source = str(owner.get("source") or "")
        owner_lease_state = str(owner.get("lease_state") or "")
        return {
            "reason": (
                (
                    "idle_worktree_pool_entry"
                    if owner_lease_state == "idle"
                    else "active_worktree_pool_lease"
                )
                if owner_source == "worktree_pool_lease"
                else (
                    "active_state_worktree"
                    if own_lane
                    else "active_peer_state_worktree"
                )
            ),
            "owner_source": owner_source,
            "owner_state_path": owner_state_path,
            "owner_snapshot_path": owner_snapshot_path,
            "owner_pool_state_path": str(owner.get("pool_state_path") or ""),
            "owner_task_id": str(owner.get("task_id") or ""),
            "owner_branch": str(owner.get("branch") or ""),
            "owner_lease_state": owner_lease_state,
            "owner_lease_pid": str(owner.get("lease_pid") or ""),
        }

    @staticmethod
    def _safe_rescue_branch_fragment(value: str) -> str:
        normalized = []
        for char in value.strip().strip("/").replace("\\", "/"):
            if char.isalnum() or char in {".", "_", "-"}:
                normalized.append(char)
            elif char == "/":
                normalized.append("-")
            else:
                normalized.append("-")
        fragment = "".join(normalized).strip(".-")
        while "--" in fragment:
            fragment = fragment.replace("--", "-")
        return fragment[:96] or "worktree"

    def _rescue_dirty_worktree(
        self,
        worktree_path: Path,
        *,
        branch: str,
        head: str,
        target_ref: str,
        status_lines: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Commit dirty inactive worktree content to a rescue branch for later merge."""

        started_at = utc_now()
        if not worktree_path.exists():
            return {
                "attempted": True,
                "preserved": False,
                "reason": "worktree_path_missing",
                "path": str(worktree_path),
                "branch": branch,
                "started_at": started_at,
                "finished_at": utc_now(),
            }
        if not branch and not head:
            return {
                "attempted": True,
                "preserved": False,
                "reason": "missing_branch_and_head",
                "path": str(worktree_path),
                "started_at": started_at,
                "finished_at": utc_now(),
            }

        # Nested submodule dirt (untracked/modified content inside configured
        # gitlinks) is invisible to monorepo ``git add -A`` and to
        # ``--ignore-submodules=dirty`` stageability proofs.  Materialize it
        # into nested commits + parent gitlink updates so rescue can finish
        # instead of looping forever on
        # existing_rescue_branch_nested_state_requires_reconciliation.
        nested_materialization = self._materialize_nested_configured_submodule_dirt(
            worktree_path,
            status_lines=status_lines,
            reason=reason,
        )
        if nested_materialization.get("committed_count"):
            status_lines = self._git_status_short(worktree_path)

        stageability = self._existing_rescue_branch_stageability(
            worktree_path,
            branch=branch,
        )
        if stageability.get("no_stageable_delta"):
            rescue_commit = self._git_ref_commit(worktree_path, "HEAD")
            result = {
                "attempted": True,
                "preserved": False,
                "reason": (
                    "existing_rescue_branch_nested_state_requires_reconciliation"
                ),
                "path": str(worktree_path),
                "branch": branch,
                "head": head,
                "target_ref": target_ref,
                "rescue_branch": branch,
                "rescue_commit": rescue_commit,
                "status_short": status_lines[:20],
                "stageability_proof": stageability,
                "nested_materialization": nested_materialization,
                "started_at": started_at,
                "finished_at": utc_now(),
            }
            self._record_event("dirty_worktree_rescue_deferred", result)
            return result

        fingerprint = sha1(
            json.dumps(
                {
                    "branch": branch,
                    "head": head,
                    "path": str(worktree_path),
                    "status": status_lines,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        rescue_branch = (
            branch
            if branch.startswith("rescue/worktree/")
            else (
                f"rescue/worktree/"
                f"{self._safe_rescue_branch_fragment(branch or worktree_path.name)}-{fingerprint}"
            )
        )

        current_branch = self._git_current_branch(worktree_path)
        checkout = None
        if current_branch != rescue_branch:
            checkout = subprocess.run(
                ["git", "checkout", "-B", rescue_branch],
                cwd=worktree_path,
                text=True,
                capture_output=True,
                check=False,
            )
        if checkout is not None and checkout.returncode != 0:
            result = {
                "attempted": True,
                "preserved": False,
                "reason": "checkout_rescue_branch_failed",
                "path": str(worktree_path),
                "branch": branch,
                "head": head,
                "target_ref": target_ref,
                "rescue_branch": rescue_branch,
                "returncode": checkout.returncode,
                "stdout": checkout.stdout[-4000:],
                "stderr": checkout.stderr[-4000:],
                "started_at": started_at,
                "finished_at": utc_now(),
            }
            self._record_event("dirty_worktree_rescue_failed", result)
            return result

        add = subprocess.run(
            ["git", "add", "-A"],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            result = {
                "attempted": True,
                "preserved": False,
                "reason": "stage_rescue_changes_failed",
                "path": str(worktree_path),
                "branch": branch,
                "head": head,
                "target_ref": target_ref,
                "rescue_branch": rescue_branch,
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
                "nested_materialization": nested_materialization,
                "started_at": started_at,
                "finished_at": utc_now(),
            }
            self._record_event("dirty_worktree_rescue_failed", result)
            return result

        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if staged.returncode == 0:
            # Parent index may still look empty when only nested dirt remained
            # and the first materialization pass found nothing (race) or new
            # nested dirt appeared after checkout.  Retry once.
            retry_materialization = self._materialize_nested_configured_submodule_dirt(
                worktree_path,
                status_lines=self._git_status_short(worktree_path),
                reason=f"{reason}:retry_after_empty_stage",
            )
            if retry_materialization.get("committed_count"):
                nested_materialization = {
                    **nested_materialization,
                    "retry": retry_materialization,
                }
                add = subprocess.run(
                    ["git", "add", "-A"],
                    cwd=worktree_path,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                staged = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=worktree_path,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            if staged.returncode == 0:
                rescue_commit = self._git_ref_commit(worktree_path, "HEAD")
                result = {
                    "attempted": True,
                    "preserved": False,
                    "reason": "no_staged_rescue_delta_requires_reconciliation",
                    "path": str(worktree_path),
                    "branch": branch,
                    "head": head,
                    "target_ref": target_ref,
                    "rescue_branch": rescue_branch,
                    "rescue_commit": rescue_commit,
                    "status_short": status_lines[:20],
                    "nested_materialization": nested_materialization,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                }
                self._record_event("dirty_worktree_rescue_deferred", result)
                return result

        commit = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Implementation Supervisor",
                "-c",
                "user.email=implementation-supervisor@example.invalid",
                "commit",
                "-m",
                f"Rescue dirty worktree {branch or worktree_path.name}",
                "-m",
                f"Original branch: {branch or '(detached)'}",
                "-m",
                f"Original HEAD: {head or '(unknown)'}",
                "-m",
                f"Cleanup reason: {reason}",
            ],
            cwd=worktree_path,
            text=True,
            capture_output=True,
            check=False,
        )
        if commit.returncode != 0:
            result = {
                "attempted": True,
                "preserved": False,
                "reason": "commit_rescue_changes_failed",
                "path": str(worktree_path),
                "branch": branch,
                "head": head,
                "target_ref": target_ref,
                "rescue_branch": rescue_branch,
                "returncode": commit.returncode,
                "stdout": commit.stdout[-4000:],
                "stderr": commit.stderr[-4000:],
                "nested_materialization": nested_materialization,
                "started_at": started_at,
                "finished_at": utc_now(),
            }
            self._record_event("dirty_worktree_rescue_failed", result)
            return result

        rescue_commit = self._git_ref_commit(worktree_path, "HEAD")
        result = {
            "attempted": True,
            "preserved": bool(rescue_commit),
            "reason": "dirty_worktree_committed_to_rescue_branch",
            "path": str(worktree_path),
            "branch": branch,
            "head": head,
            "target_ref": target_ref,
            "rescue_branch": rescue_branch,
            "rescue_commit": rescue_commit,
            "status_short": status_lines[:20],
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
            "nested_materialization": nested_materialization,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        self._record_event("dirty_worktree_rescued", result)
        return result

    def _materialize_nested_configured_submodule_dirt(
        self,
        worktree_path: Path,
        *,
        status_lines: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Commit nested dirt inside configured submodules so parent gitlinks stage.

        Monorepo ``git add -A`` never captures untracked/modified files that live
        only inside a submodule worktree.  Without a nested commit the parent
        status stays `` ? path`` / `` m path`` forever, rescue defers forever,
        and multi-lane supervisors burn cycles on the same worktree.
        """

        candidates: list[str] = []
        seen: set[str] = set()
        for line in status_lines:
            code = line[:2]
            relative = self._status_line_path(line)
            if not relative:
                continue
            normalized = relative.rstrip("/")
            if normalized in seen:
                continue
            if not self._is_configured_worktree_submodule_path(normalized):
                continue
            # Nested dirt codes plus ordinary gitlink modifications.
            if code in {" m", " ?", "? ", "M ", "MM", "AM", "A ", "??"} or (
                "M" in code or "?" in code or "A" in code
            ):
                seen.add(normalized)
                candidates.append(normalized)

        if not candidates:
            return {
                "attempted": False,
                "reason": "no_configured_submodule_dirt",
                "committed_count": 0,
                "paths": [],
            }

        nested_results: list[dict[str, Any]] = []
        committed_count = 0
        for relative in candidates:
            nested_root = worktree_path / relative
            entry: dict[str, Any] = {"path": relative, "nested_root": str(nested_root)}
            if not nested_root.exists():
                entry["reason"] = "nested_path_missing"
                nested_results.append(entry)
                continue
            if not (nested_root / ".git").exists() and not (nested_root / ".git").is_file():
                # Submodule checkouts use a .git file; bare dirs are not repos.
                git_dir_probe = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=nested_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if git_dir_probe.returncode != 0:
                    entry["reason"] = "not_a_git_worktree"
                    nested_results.append(entry)
                    continue

            nested_status = self._git_status_short(nested_root)
            entry["nested_status_short"] = nested_status[:20]
            if not nested_status:
                entry["reason"] = "nested_already_clean"
                nested_results.append(entry)
                continue

            add = subprocess.run(
                ["git", "add", "-A"],
                cwd=nested_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if add.returncode != 0:
                entry["reason"] = "nested_stage_failed"
                entry["returncode"] = add.returncode
                entry["stderr"] = add.stderr[-2000:]
                nested_results.append(entry)
                continue

            staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=nested_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if staged.returncode == 0:
                entry["reason"] = "nested_no_stageable_delta"
                nested_results.append(entry)
                continue

            nested_head_before = self._git_ref_commit(nested_root, "HEAD")
            commit = subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Implementation Supervisor",
                    "-c",
                    "user.email=implementation-supervisor@example.invalid",
                    "commit",
                    "-m",
                    f"Rescue nested submodule dirt in {relative}",
                    "-m",
                    f"Parent cleanup reason: {reason}",
                ],
                cwd=nested_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if commit.returncode != 0:
                entry["reason"] = "nested_commit_failed"
                entry["returncode"] = commit.returncode
                entry["stderr"] = commit.stderr[-2000:]
                nested_results.append(entry)
                continue

            nested_head_after = self._git_ref_commit(nested_root, "HEAD")
            entry["reason"] = "nested_dirt_committed"
            entry["nested_head_before"] = nested_head_before
            entry["nested_head_after"] = nested_head_after
            entry["committed"] = True
            committed_count += 1
            nested_results.append(entry)

            # Stage the updated gitlink on the parent so rescue can commit it.
            stage_link = subprocess.run(
                ["git", "add", "--", relative],
                cwd=worktree_path,
                text=True,
                capture_output=True,
                check=False,
            )
            entry["parent_gitlink_stage_returncode"] = stage_link.returncode
            if stage_link.returncode != 0:
                entry["parent_gitlink_stage_stderr"] = stage_link.stderr[-2000:]

        return {
            "attempted": True,
            "reason": "nested_configured_submodule_materialization",
            "committed_count": committed_count,
            "paths": candidates,
            "nested_results": nested_results,
        }

    def _existing_rescue_branch_stageability(
        self,
        worktree_path: Path,
        *,
        branch: str,
    ) -> dict[str, Any]:
        """Prove whether an existing rescue branch has anything Git can stage.

        Nested-only submodule dirt is intentionally ignored. Gitlink commit
        changes, ordinary staged/unstaged changes, and untracked paths remain
        observable and prevent the idempotent short circuit.
        """

        proof: dict[str, Any] = {
            "already_rescue_branch": branch.startswith("rescue/worktree/"),
        }
        if not proof["already_rescue_branch"]:
            proof["no_stageable_delta"] = False
            return proof

        current_branch = self._git_current_branch(worktree_path)
        proof["current_branch_matches"] = current_branch == branch
        if current_branch != branch:
            proof["no_stageable_delta"] = False
            return proof

        git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        commands = {
            "staged_diff_returncode": [
                "git",
                "diff",
                "--cached",
                "--quiet",
                "--ignore-submodules=dirty",
                "--",
            ],
            "unstaged_diff_returncode": [
                "git",
                "diff",
                "--quiet",
                "--ignore-submodules=dirty",
                "--",
            ],
        }
        for field, command in commands.items():
            result = subprocess.run(
                command,
                cwd=worktree_path,
                capture_output=True,
                env=git_env,
                check=False,
            )
            proof[field] = result.returncode

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=worktree_path,
            capture_output=True,
            env=git_env,
            check=False,
        )
        proof["untracked_query_returncode"] = untracked.returncode
        proof["has_untracked_paths"] = bool(untracked.stdout)
        proof["no_stageable_delta"] = (
            proof["staged_diff_returncode"] == 0
            and proof["unstaged_diff_returncode"] == 0
            and untracked.returncode == 0
            and not untracked.stdout
        )
        return proof

    def cleanup_backlogged_worktrees(self) -> dict[str, Any]:
        """Remove inactive implementation worktrees whose branches are already merged."""

        lock_path = self._repo_merge_lock_path()
        lock_metadata = self._supervisor_checkout_lock_metadata(
            operation="cleanup_backlogged_worktrees",
        )
        lease, lock_reason, existing_lock = (
            self._acquire_supervisor_checkout_lease(
                lock_path,
                lock_metadata,
            )
        )
        if lease is None:
            result: dict[str, Any] = {
                "attempted": True,
                "removed_count": 0,
                "skipped_count": 0,
                "reason": f"checkout_mutation_{lock_reason}",
                "lock_path": str(lock_path),
            }
            if existing_lock:
                result["lock_owner_pid"] = int(existing_lock.get("pid") or 0)
                result["lock_owner_task_id"] = str(
                    existing_lock.get("task_id") or ""
                )
                result["lock_owner_branch"] = str(
                    existing_lock.get("branch") or ""
                )
            self._record_event("merged_worktree_cleanup_deferred", result)
            return result

        try:
            return self._cleanup_backlogged_worktrees_locked()
        finally:
            self._release_supervisor_checkout_lease(
                lease,
                operation="cleanup_backlogged_worktrees",
            )

    def _cleanup_backlogged_worktrees_locked(self) -> dict[str, Any]:
        """Clean merged worktrees while holding the checkout mutation lock."""

        worktree_root = self.config.worktree_root
        if worktree_root is None:
            return {"attempted": False, "reason": "worktree_root_not_configured"}
        repo_root = self.config.repo_root
        prune = subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        records = self._git_worktree_records(repo_root)
        try:
            root_resolved = worktree_root.resolve()
        except OSError:
            root_resolved = worktree_root
        process_lines = self._list_process_commands()
        active_worktree_owners = self._shared_active_worktree_owners(
            worktree_root
        )
        target_ref = (
            self.config.merge_target_branch
            or self._git_current_branch(repo_root)
            or "HEAD"
        )
        target_signature = self._git_ref_commit(repo_root, target_ref) or target_ref
        scan_cache = self._load_worktree_scan_cache()
        scan_cache_hit_count = 0
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        dirty_evidence_sample_counts: dict[str, int] = {}

        for record in records:
            path_text = str(record.get("worktree") or "")
            if not path_text:
                continue
            path = Path(path_text)
            try:
                path_resolved = path.resolve()
                path_resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            if not path_resolved.exists():
                skipped.append(
                    {
                        "path": str(path),
                        "reason": "worktree_removed_concurrently",
                    }
                )
                continue
            active_skip = self._active_worktree_skip_detail(
                path_resolved,
                active_worktree_owners,
            )
            if active_skip is not None:
                skipped.append({"path": str(path), **active_skip})
                continue
            if any(str(path_resolved) in line for line in process_lines):
                skipped.append({"path": str(path), "reason": "active_process"})
                continue

            branch = str(record.get("branch") or "").removeprefix("refs/heads/")
            head = str(record.get("HEAD") or "")
            cached_entry = self._worktree_scan_cache_entry(
                scan_cache,
                phase="cleanup",
                path=path_resolved,
                branch=branch,
                head=head,
                target_signature=target_signature,
            )
            if cached_entry:
                classification = str(cached_entry.get("classification") or "")
                payload = dict(cached_entry.get("payload") or {})
                if classification == "skip":
                    if payload.get("reason") == "dirty_worktree":
                        pass
                    else:
                        skipped.append({**payload, "cached": True})
                        scan_cache_hit_count += 1
                        continue
                else:
                    skipped.append({**payload, "cached": True})
                    scan_cache_hit_count += 1
                    continue
            branch_merged = bool(branch) and self._git_ref_is_ancestor(repo_root, branch, target_ref)
            head_merged = bool(head) and self._git_ref_is_ancestor(repo_root, head, target_ref)
            if not (branch_merged or head_merged):
                skip = {"path": str(path), "branch": branch, "reason": "not_merged"}
                skipped.append(skip)
                self._store_worktree_scan_cache_entry(
                    scan_cache,
                    phase="cleanup",
                    path=path_resolved,
                    branch=branch,
                    head=head,
                    target_signature=target_signature,
                    classification="skip",
                    payload=skip,
                )
                continue
            dirty = self._git_status_short(path) if path.exists() else []
            if not path.exists():
                skipped.append(
                    {
                        "path": str(path),
                        "branch": branch,
                        "reason": "worktree_removed_concurrently",
                    }
                )
                continue
            dirty_redundancy: dict[str, Any] = {}
            if dirty:
                redundant_dirty = self._redundant_dirty_worktree_status(path, dirty, target_ref)
                if redundant_dirty.get("redundant"):
                    dirty_redundancy = redundant_dirty
                    dirty = []
                else:
                    dirty_reason = self._dirty_redundancy_reason(redundant_dirty)
                    evidence: dict[str, Any] = {}
                    if dirty_evidence_sample_counts.get(dirty_reason, 0) < 20:
                        evidence = self._dirty_worktree_evidence(path, dirty)
                        dirty_evidence_sample_counts[dirty_reason] = (
                            dirty_evidence_sample_counts.get(dirty_reason, 0) + 1
                        )
                    rescue_result = self._rescue_dirty_worktree(
                        path,
                        branch=branch,
                        head=head,
                        target_ref=target_ref,
                        status_lines=dirty,
                        reason=f"cleanup_dirty_worktree:{dirty_reason}",
                    )
                    if rescue_result.get("preserved"):
                        skipped.append(
                            {
                                "path": str(path),
                                "branch": branch,
                                "reason": "dirty_worktree_rescued",
                                "status_short": dirty[:20],
                                "dirty_redundancy": redundant_dirty,
                                "dirty_evidence": evidence,
                                "rescue_result": rescue_result,
                            }
                        )
                        continue
                    skip = {
                        "path": str(path),
                        "branch": branch,
                        "reason": "dirty_worktree",
                        "status_short": dirty[:20],
                        "dirty_redundancy": redundant_dirty,
                        "dirty_evidence": evidence,
                        "rescue_result": rescue_result,
                    }
                    skipped.append(skip)
                    self._store_worktree_scan_cache_entry(
                        scan_cache,
                        phase="cleanup",
                        path=path_resolved,
                        branch=branch,
                        head=head,
                        target_signature=target_signature,
                        classification="skip",
                        payload=skip,
                    )
                    continue

            remove = subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            branch_delete: dict[str, Any] = {}
            if (
                remove.returncode == 0
                and self._worktree_branch_can_delete_after_merge(branch)
                and branch_merged
            ):
                delete = subprocess.run(
                    ["git", "branch", "-D", branch],
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                branch_delete = {
                    "attempted": True,
                    "deleted": delete.returncode == 0,
                    "returncode": delete.returncode,
                    "stdout": delete.stdout[-4000:],
                    "stderr": delete.stderr[-4000:],
                }
            removed.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "head": head,
                    "removed": remove.returncode == 0,
                    "returncode": remove.returncode,
                    "stdout": remove.stdout[-4000:],
                    "stderr": remove.stderr[-4000:],
                    "branch_delete": branch_delete,
                    "dirty_redundancy": dirty_redundancy,
                }
            )

        managed_submodule_prune = self._prune_managed_submodule_worktrees()
        skip_summary = self._cleanup_skip_summary(skipped)
        result = {
            "attempted": True,
            "worktree_root": str(worktree_root),
            "target_ref": target_ref,
            "target_signature": target_signature,
            "prune_returncode": prune.returncode,
            "prune_stdout": prune.stdout[-4000:],
            "prune_stderr": prune.stderr[-4000:],
            "removed_count": sum(1 for item in removed if item.get("removed")),
            "skipped_count": len(skipped),
            "skipped_reason_counts": skip_summary["reason_counts"],
            "dirty_worktree_groups": skip_summary["dirty_worktree_groups"],
            "removed": removed,
            "skipped": skipped[:50],
            "scan_cache_hit_count": scan_cache_hit_count,
            "scan_cache_written": self._write_worktree_scan_cache(scan_cache),
            "managed_submodule_worktree_prune": managed_submodule_prune,
        }
        if (
            removed
            or skip_summary["dirty_worktree_groups"]
            or managed_submodule_prune.get("failed_count")
        ):
            self._record_event("merged_worktree_cleanup", result)
        return result

    def _prune_managed_submodule_worktrees(self) -> dict[str, Any]:
        """Prune stale registrations in explicitly managed submodule repositories.

        Removing a parent worktree also removes nested submodule worktree
        directories, but Git does not remove those paths from each submodule's
        own worktree registry. Limit this follow-up to a bounded set of exact
        configured paths which Git identifies as submodule worktrees.
        """

        configured = tuple(
            dict.fromkeys(
                str(value).strip().rstrip("/")
                for value in self.config.worktree_submodule_paths
                if str(value).strip().rstrip("/")
            )
        )
        limit = MAX_MANAGED_SUBMODULE_WORKTREE_PRUNES_PER_PASS
        selected = configured[:limit]
        result: dict[str, Any] = {
            "attempted": bool(configured),
            "configured_count": len(configured),
            "considered_count": len(selected),
            "max_repositories_per_pass": limit,
            "truncated_count": max(0, len(configured) - len(selected)),
            "successful_repository_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "repositories": [],
            "skipped": [],
        }
        if not configured:
            result["reason"] = "no_managed_submodules_configured"
            return result

        repo_root = self.config.repo_root
        try:
            root_resolved = repo_root.resolve(strict=True)
        except (OSError, RuntimeError):
            result["failed_count"] = 1
            result["reason"] = "repo_root_unresolvable"
            return result

        for relative in selected:
            detail = {"path": relative}
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or "\0" in relative
                or ".." in relative_path.parts
                or not relative_path.parts
            ):
                result["skipped"].append({**detail, "reason": "unsafe_relative_path"})
                continue

            candidate = repo_root.joinpath(*relative_path.parts)
            cursor = repo_root
            symlinked = False
            for part in relative_path.parts:
                cursor /= part
                if cursor.is_symlink():
                    symlinked = True
                    break
            if symlinked:
                result["skipped"].append({**detail, "reason": "symlinked_path"})
                continue

            try:
                candidate_resolved = candidate.resolve(strict=True)
                candidate_resolved.relative_to(root_resolved)
            except FileNotFoundError:
                result["skipped"].append({**detail, "reason": "submodule_not_initialized"})
                continue
            except (OSError, RuntimeError, ValueError):
                result["skipped"].append({**detail, "reason": "path_outside_repo"})
                continue
            if not candidate_resolved.is_dir():
                result["skipped"].append({**detail, "reason": "submodule_not_directory"})
                continue

            try:
                identity = subprocess.run(
                    ["git", "rev-parse", "--show-superproject-working-tree"],
                    cwd=candidate_resolved,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=MANAGED_SUBMODULE_WORKTREE_PRUNE_TIMEOUT_SECONDS,
                )
                if identity.returncode != 0 or not identity.stdout.strip():
                    raise ValueError("not a submodule worktree")
                superproject = Path(identity.stdout.strip()).resolve(strict=True)
                superproject.relative_to(root_resolved)
            except (
                FileNotFoundError,
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ):
                result["skipped"].append({**detail, "reason": "unmanaged_repository"})
                continue

            try:
                prune = subprocess.run(
                    ["git", "worktree", "prune", "--expire", "now"],
                    cwd=candidate_resolved,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=MANAGED_SUBMODULE_WORKTREE_PRUNE_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                result["repositories"].append(
                    {
                        **detail,
                        "repo_path": str(candidate_resolved),
                        "pruned": False,
                        "reason": "prune_failed",
                        "error": str(exc)[-1000:],
                    }
                )
                continue
            result["repositories"].append(
                {
                    **detail,
                    "repo_path": str(candidate_resolved),
                    "pruned": prune.returncode == 0,
                    "returncode": prune.returncode,
                    "stdout": prune.stdout[-4000:],
                    "stderr": prune.stderr[-4000:],
                }
            )

        result["successful_repository_count"] = sum(
            bool(item.get("pruned")) for item in result["repositories"]
        )
        result["failed_count"] = sum(
            not bool(item.get("pruned")) for item in result["repositories"]
        )
        result["skipped_count"] = len(result["skipped"])
        return result

    @staticmethod
    def _dirty_redundancy_reason(dirty_redundancy: dict[str, Any]) -> str:
        return str(dirty_redundancy.get("reason") or "dirty_worktree")

    def _dirty_worktree_evidence(self, worktree_path: Path, status_lines: list[str]) -> dict[str, Any]:
        """Return bounded evidence for dirty cleanup blockers without storing full patches."""

        evidence: dict[str, Any] = {
            "status_short": status_lines[:20],
        }
        diff_stat = self._git_output(worktree_path, ["diff", "--stat"], max_chars=4000)
        if diff_stat:
            evidence["diff_stat"] = diff_stat
        name_status = self._git_output(worktree_path, ["diff", "--name-status"], max_chars=4000)
        if name_status:
            evidence["name_status"] = name_status
        untracked_paths = [
            self._status_line_path(line)
            for line in status_lines
            if line[:2] == "??" and self._status_line_path(line)
        ][:20]
        if untracked_paths:
            evidence["untracked_paths"] = untracked_paths
        return evidence

    @staticmethod
    def _cleanup_skip_summary(skipped: list[dict[str, Any]]) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        dirty_worktree_groups: dict[str, dict[str, Any]] = {}
        for item in skipped:
            reason = str(item.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason != "dirty_worktree":
                continue
            dirty_redundancy = item.get("dirty_redundancy") or {}
            dirty_reason = (
                str(dirty_redundancy.get("reason") or "dirty_worktree")
                if isinstance(dirty_redundancy, dict)
                else "dirty_worktree"
            )
            reason_key = f"dirty_worktree:{dirty_reason}"
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
            group = dirty_worktree_groups.setdefault(
                dirty_reason,
                {
                    "count": 0,
                    "samples": [],
                },
            )
            group["count"] += 1
            if len(group["samples"]) < 20:
                group["samples"].append(
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "status_short": list(item.get("status_short") or []),
                        "dirty_reason": dirty_reason,
                        "dirty_evidence": dict(item.get("dirty_evidence") or {}),
                    }
                )
        return {
            "reason_counts": reason_counts,
            "dirty_worktree_groups": dirty_worktree_groups,
        }

    @staticmethod
    def _git_output(cwd: Path, args: list[str], *, max_chars: int = 4000) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout[-max_chars:].strip()

    def _redundant_dirty_worktree_status(
        self,
        worktree_path: Path,
        status_lines: list[str],
        target_ref: str,
    ) -> dict[str, Any]:
        generated_only = self._generated_only_dirty_worktree_status(worktree_path, status_lines)
        if generated_only:
            return generated_only

        checked: list[dict[str, Any]] = []
        configured_submodule_deletion = False
        configured_submodule_unstaged_deletion = False
        for line in status_lines:
            code = line[:2]
            relative = self._status_line_path(line)
            detail = {"status": code, "path": relative}
            if not relative:
                return {"redundant": False, "reason": "empty_status_path", "checked": checked}
            if self._status_line_is_configured_submodule_deletion(code, relative, target_ref):
                checked.append({**detail, "matches_target": True, "configured_submodule_deletion": True})
                configured_submodule_deletion = True
                continue
            # Submodule dirt: " m" (modified content), " ?" / "? " (untracked
            # content inside submodule).  These are common residual states after
            # nested work and must not permanently stall cleanup as
            # unsupported_status (WPD-071 unblock).
            if code in {" m", " ?", "? "} and self._is_configured_worktree_submodule_path(
                relative
            ):
                if code == " m":
                    verdict = self._configured_submodule_unstaged_deletion_proof(
                        worktree_path,
                        relative=relative,
                        target_ref=target_ref,
                    )
                    checked.append(
                        {
                            **detail,
                            "configured_submodule_unstaged_deletion": bool(
                                verdict.get("redundant")
                            ),
                            "proof_reason": str(verdict.get("reason") or ""),
                            "proof": dict(verdict.get("proof") or {}),
                        }
                    )
                    if not verdict.get("redundant"):
                        # Nested content differs but is still a known submodule
                        # dirt class — not an exotic index state.  Report as
                        # content_not_in_target so operators/automation treat it
                        # as ordinary dirty content rather than unsupported_status.
                        return {
                            "redundant": False,
                            "reason": "content_not_in_target",
                            "checked": checked,
                        }
                    configured_submodule_unstaged_deletion = True
                    continue
                # Untracked content inside a configured submodule: treat as
                # ordinary non-matching dirty content, not unsupported_status.
                checked.append(
                    {
                        **detail,
                        "configured_submodule_untracked_content": True,
                        "matches_target": False,
                    }
                )
                return {
                    "redundant": False,
                    "reason": "content_not_in_target",
                    "checked": checked,
                }
            if "D" in code or ("?" in code and code not in {"??", " ?", "? "}):
                return {
                    "redundant": False,
                    "reason": "unsupported_status",
                    "checked": [*checked, detail],
                }
            if code == "??" or "M" in code or "A" in code or code in {" ?", "? "}:
                if not self._worktree_file_matches_ref(worktree_path, relative, target_ref):
                    return {
                        "redundant": False,
                        "reason": "content_not_in_target",
                        "checked": [*checked, detail],
                    }
                checked.append({**detail, "matches_target": True})
                continue
            return {"redundant": False, "reason": "unsupported_status", "checked": [*checked, detail]}
        reason = (
            "configured_submodule_unstaged_deletions_match_target"
            if configured_submodule_unstaged_deletion
            else (
                "configured_submodule_deletions_match_target"
                if configured_submodule_deletion
                else "all_dirty_paths_match_target"
            )
        )
        return {"redundant": True, "reason": reason, "checked": checked}

    def _is_configured_worktree_submodule_path(self, relative: str) -> bool:
        normalized = relative.rstrip("/")
        return any(
            normalized == path.rstrip("/")
            for path in self.config.worktree_submodule_paths
        )

    def _status_line_is_configured_submodule_deletion(
        self,
        code: str,
        relative: str,
        target_ref: str,
    ) -> bool:
        if code not in {" D", "D "}:
            return False
        normalized = relative.rstrip("/")
        if not self._is_configured_worktree_submodule_path(normalized):
            return False
        # An uppercase deletion is the disappearance of the configured
        # gitlink itself. It is redundant only when the integration target
        # still owns that exact path. Lowercase nested-submodule dirt follows
        # the stronger gitlink/head proof in
        # ``_configured_submodule_unstaged_deletion_proof``.
        return self._target_ref_has_path(normalized, target_ref)

    @staticmethod
    def _gitlink_tree_entry(
        cwd: Path,
        *,
        treeish: str,
        relative: str,
    ) -> dict[str, str] | None:
        result = subprocess.run(
            ["git", "ls-tree", "-z", treeish, "--", relative],
            cwd=cwd,
            capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            check=False,
        )
        if result.returncode != 0:
            return None
        records = result.stdout.split(b"\0")
        if records and records[-1] == b"":
            records.pop()
        if len(records) != 1:
            return None
        metadata, separator, raw_path = records[0].partition(b"\t")
        fields = metadata.split()
        if (
            separator != b"\t"
            or len(fields) != 3
            or fields[0] != b"160000"
            or fields[1] != b"commit"
            or raw_path != os.fsencode(relative)
        ):
            return None
        return {
            "mode": fields[0].decode("ascii"),
            "commit": fields[2].decode("ascii"),
        }

    def _configured_submodule_unstaged_deletion_proof(
        self,
        worktree_path: Path,
        *,
        relative: str,
        target_ref: str,
    ) -> dict[str, Any]:
        """Prove lowercase configured-submodule dirt is deletion-only."""

        head_gitlink = self._gitlink_tree_entry(
            worktree_path,
            treeish="HEAD",
            relative=relative,
        )
        target_gitlink = self._gitlink_tree_entry(
            self.config.repo_root,
            treeish=target_ref,
            relative=relative,
        )
        proof: dict[str, Any] = {
            "head_gitlink": head_gitlink or {},
            "target_gitlink": target_gitlink or {},
        }
        if head_gitlink is None or target_gitlink is None:
            return {
                "redundant": False,
                "reason": "configured_submodule_gitlink_unavailable",
                "proof": proof,
            }
        if head_gitlink != target_gitlink:
            return {
                "redundant": False,
                "reason": "configured_submodule_gitlink_mismatch",
                "proof": proof,
            }

        nested_path = worktree_path / relative
        try:
            worktree_root = worktree_path.resolve(strict=True)
            if nested_path.is_symlink():
                raise ValueError("nested path is a symlink")
            nested_root = nested_path.resolve(strict=True)
            nested_root.relative_to(worktree_root)
        except (OSError, ValueError):
            proof["nested_repo_root_matches"] = False
            return {
                "redundant": False,
                "reason": "configured_submodule_nested_repo_unsafe",
                "proof": proof,
            }

        git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=nested_root,
            text=True,
            capture_output=True,
            env=git_env,
            check=False,
        )
        try:
            reported_root = Path(top_level.stdout.strip()).resolve(strict=True)
        except OSError:
            reported_root = Path()
        proof["nested_repo_root_matches"] = (
            top_level.returncode == 0 and reported_root == nested_root
        )
        if not proof["nested_repo_root_matches"]:
            return {
                "redundant": False,
                "reason": "configured_submodule_nested_repo_mismatch",
                "proof": proof,
            }

        nested_head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=nested_root,
            text=True,
            capture_output=True,
            env=git_env,
            check=False,
        )
        proof["nested_head"] = (
            nested_head.stdout.strip() if nested_head.returncode == 0 else ""
        )
        if proof["nested_head"] != head_gitlink["commit"]:
            return {
                "redundant": False,
                "reason": "configured_submodule_nested_head_mismatch",
                "proof": proof,
            }

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            cwd=nested_root,
            capture_output=True,
            env=git_env,
            check=False,
        )
        proof["nested_status_returncode"] = status.returncode
        if status.returncode != 0:
            return {
                "redundant": False,
                "reason": "configured_submodule_nested_status_unavailable",
                "proof": proof,
            }

        records = status.stdout.split(b"\0")
        if records and records[-1] == b"":
            records.pop()
        status_codes: dict[str, int] = {}
        all_unstaged_tracked_deletions = bool(records)
        for record in records:
            code = record[:2].decode("ascii", errors="backslashreplace")
            status_codes[code] = status_codes.get(code, 0) + 1
            if (
                len(record) < 4
                or record[:2] != b" D"
                or record[2:3] != b" "
                or not record[3:]
            ):
                all_unstaged_tracked_deletions = False
        proof["nested_status_entry_count"] = len(records)
        proof["nested_status_codes"] = dict(sorted(status_codes.items()))
        proof["all_unstaged_tracked_deletions"] = all_unstaged_tracked_deletions
        if not all_unstaged_tracked_deletions:
            return {
                "redundant": False,
                "reason": (
                    "configured_submodule_nested_status_not_unstaged_deletions"
                ),
                "proof": proof,
            }
        proof["mechanically_restorable_from_gitlink"] = True
        return {
            "redundant": False,
            "reason": (
                "configured_submodule_unstaged_deletions_require_reconciliation"
            ),
            "proof": proof,
        }

    def _target_ref_has_path(self, relative: str, target_ref: str) -> bool:
        result = subprocess.run(
            ["git", "ls-tree", target_ref, "--", relative],
            cwd=self.config.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    @staticmethod
    def _status_line_path(line: str) -> str:
        path_text = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[-1].strip()
        return path_text.rstrip("/")

    def _worktree_file_matches_ref(self, worktree_path: Path, relative: str, target_ref: str) -> bool:
        candidate = worktree_path / relative
        if not candidate.is_file():
            return False
        result = subprocess.run(
            ["git", "show", f"{target_ref}:{relative}"],
            cwd=self.config.repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        try:
            return candidate.read_bytes() == result.stdout
        except OSError:
            return False

    @staticmethod
    def _git_worktree_records(repo_root: Path) -> list[dict[str, str]]:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        records: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        if current:
            records.append(current)
        return records

    @staticmethod
    def _git_ref_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
        if not ancestor or not descendant:
            return False
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _git_ref_exists(repo_root: Path, ref: str) -> bool:
        if not ref:
            return False
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    @staticmethod
    def _list_process_commands() -> list[str]:
        return [command for _pid, command in PortalImplementationSupervisor._list_process_details()]

    @staticmethod
    def _list_process_details() -> list[tuple[int, str]]:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        details: list[tuple[int, str]] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            pid_text, _separator, command = stripped.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            command = command.strip()
            if command:
                details.append((pid, command))
        return details

    def ensure_todo_board_for_refill(self) -> dict[str, Any]:
        """Create an empty todo board when refill machinery is expected to populate it."""

        if self.config.todo_path.exists():
            if self.config.todo_path.is_dir():
                if not (
                    self.config.objective_refill_enabled
                    or self.config.codebase_refill_enabled
                    or self.config.reconciliation_guardrail_enabled
                ):
                    return {"created": False, "reason": "todo_path_is_directory", "path": str(self.config.todo_path)}
                backup_path = unique_backup_path(self.config.todo_path, "directory-backup")
                self.config.todo_path.rename(backup_path)
                self.config.todo_path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(self.config.todo_path, "# Agent Todos\n")
                result = {
                    "created": True,
                    "repaired": True,
                    "reason": "todo_path_was_directory",
                    "path": str(self.config.todo_path),
                    "backup_path": str(backup_path),
                }
                self._record_event("todo_board_repaired", result)
                return result
            try:
                self.config.todo_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                if not (
                    self.config.objective_refill_enabled
                    or self.config.codebase_refill_enabled
                    or self.config.reconciliation_guardrail_enabled
                ):
                    return {"created": False, "reason": "todo_text_decode_failed", "path": str(self.config.todo_path)}
                backup_path = unique_backup_path(self.config.todo_path, "invalid-text")
                self.config.todo_path.rename(backup_path)
                write_text_atomic(self.config.todo_path, "# Agent Todos\n")
                result = {
                    "created": True,
                    "repaired": True,
                    "reason": "todo_text_decode_failed",
                    "path": str(self.config.todo_path),
                    "backup_path": str(backup_path),
                }
                self._record_event("todo_board_repaired", result)
                return result
            except OSError as exc:
                return {
                    "created": False,
                    "reason": "todo_read_failed",
                    "path": str(self.config.todo_path),
                    "error": str(exc),
                }
            return {"created": False, "reason": "exists", "path": str(self.config.todo_path)}
        if not (
            self.config.objective_refill_enabled
            or self.config.codebase_refill_enabled
            or self.config.reconciliation_guardrail_enabled
        ):
            return {"created": False, "reason": "refill_disabled", "path": str(self.config.todo_path)}
        self.config.todo_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.config.todo_path, "# Agent Todos\n")
        result = {"created": True, "reason": "refill_enabled", "path": str(self.config.todo_path)}
        self._record_event("todo_board_created", result)
        return result

    def ensure_event_log_file(self) -> dict[str, Any]:
        """Repair malformed supervisor event-log storage before guardrails run."""

        result = repair_jsonl_event_log(self.config.events_path)
        if result.get("repaired"):
            append_jsonl_event(self.config.events_path, "event_log_repaired", result)
        return result

    def ensure_state_file(self) -> dict[str, Any]:
        """Repair malformed durable daemon state before supervisor checks it."""

        reason = state_file_repair_reason(self.config.state_path)
        if not reason or reason == "missing_state_file":
            return {"repaired": False, "reason": reason or "valid", "path": str(self.config.state_path)}
        PortalTaskState().save(self.config.state_path)
        result = {"repaired": True, "reason": reason, "path": str(self.config.state_path)}
        self._record_event("state_file_repaired", result)
        return result

    def ensure_strategy_file(self) -> dict[str, Any]:
        """Persist a valid strategy file before guardrail/refill work starts."""

        defaults = {
            "generation": 0,
            "focus_tracks": DEFAULT_TRACKS,
            "blocked_tasks": [],
            "deprioritized_tasks": [],
            "last_rewrite_at": "",
            "last_rewrite_reason": "",
        }
        reason = ""
        if not self.config.strategy_path.exists():
            strategy = defaults.copy()
            reason = "missing_strategy_file"
        else:
            payload = load_json_dict(self.config.strategy_path)
            if payload is None:
                strategy = defaults.copy()
                reason = "invalid_or_unreadable_strategy_file"
            else:
                strategy = {**defaults, **payload}
                normalized_blocked = (
                    [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
                    if isinstance(strategy.get("blocked_tasks"), list)
                    else []
                )
                normalized_deprioritized = (
                    [str(item) for item in strategy.get("deprioritized_tasks", []) if str(item).strip()]
                    if isinstance(strategy.get("deprioritized_tasks"), list)
                    else []
                )
                normalized_focus = normalize_focus_tracks(strategy.get("focus_tracks", DEFAULT_TRACKS))
                if (
                    normalized_blocked != strategy.get("blocked_tasks")
                    or normalized_deprioritized != strategy.get("deprioritized_tasks")
                    or normalized_focus != strategy.get("focus_tracks")
                ):
                    reason = "normalized_strategy_metadata"
                strategy["blocked_tasks"] = normalized_blocked
                strategy["deprioritized_tasks"] = normalized_deprioritized
                strategy["focus_tracks"] = normalized_focus or DEFAULT_TRACKS
        if not reason:
            return {"repaired": False, "reason": "valid", "path": str(self.config.strategy_path)}
        strategy["last_strategy_repair_at"] = utc_now()
        strategy["last_strategy_repair_reason"] = reason
        write_json_atomic(self.config.strategy_path, strategy)
        result = {"repaired": True, "reason": reason, "path": str(self.config.strategy_path)}
        self._record_event("strategy_file_repaired", result)
        return result

    def release_completed_guardrail_blocks(
        self,
        reconciliation_result: Mapping[str, Any] | None = None,
        cleanup_result: Mapping[str, Any] | None = None,
        replay_result: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Remove strategy blocks once their generated repair task is completed."""

        if not self.config.todo_path.exists() or not self.config.strategy_path.exists():
            return []
        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            release_completed_guardrail_blocks,
            task_id_prefix,
        )

        commit_outputs, commit_subject = self._generated_board_commit_policy(
            configured_commit_outputs=False,
            configured_subject="Agent: retire resolved guardrail tasks",
        )
        releases = self._run_generated_board_producer(
            producer="guardrail-release",
            commit_outputs=commit_outputs,
            callback=lambda: release_completed_guardrail_blocks(
                todo_path=self.config.todo_path,
                strategy_path=self.config.strategy_path,
                reconciliation_result=reconciliation_result,
                cleanup_result=cleanup_result,
                replay_result=replay_result,
                task_prefix=task_id_prefix(self.config.task_prefix),
                commit_outputs=commit_outputs,
                repo_root=self.config.repo_root,
                commit_subject=commit_subject,
            ),
        )
        if releases:
            self._record_event(
                "guardrail_blocks_released",
                {
                    "released_count": len(releases),
                    "todo_path": str(self.config.todo_path),
                    "strategy_path": str(self.config.strategy_path),
                    "releases": releases,
                },
            )
        return releases

    def record_dependency_guardrails(self) -> list[dict[str, Any]]:
        """Convert impossible dependency metadata into ready repair tasks."""

        if not self.config.dependency_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            record_dependency_guardrail_findings,
            task_id_prefix,
        )

        discovery_dir = (
            self.config.dependency_guardrail_discovery_dir
            or self.config.retry_budget_discovery_dir
            or self.config.state_dir.parent / "discovery"
        )
        discovery_output_path = self.config.dependency_guardrail_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        commit_outputs, commit_subject = self._generated_board_commit_policy(
            configured_commit_outputs=self.config.dependency_guardrail_commit_outputs,
            configured_subject=self.config.dependency_guardrail_commit_subject,
        )
        findings = self._run_generated_board_producer(
            producer="dependency-guardrail",
            commit_outputs=commit_outputs,
            callback=lambda: record_dependency_guardrail_findings(
                todo_path=self.config.todo_path,
                strategy_path=self.config.strategy_path,
                discovery_dir=discovery_dir,
                task_header_prefix_value=self.config.task_prefix,
                task_prefix=task_id_prefix(self.config.task_prefix),
                max_findings=self.config.dependency_guardrail_max_findings,
                discovery_output_path=discovery_output_path,
                commit_outputs=commit_outputs,
                repo_root=self.config.repo_root,
                commit_subject=commit_subject,
            ),
        )
        if findings:
            self._record_event(
                "dependency_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def record_reconciliation_guardrails(
        self,
        worktree_reconciliation: Mapping[str, Any],
        worktree_cleanup: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert blocked checkout/worktree cleanup into deliberate repair tasks."""

        if not self.config.reconciliation_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            record_reconciliation_guardrail_findings,
            task_id_prefix,
        )

        discovery_dir = self._reconciliation_guardrail_discovery_dir()
        discovery_output_path = self.config.reconciliation_guardrail_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        generated_paths, generated_prefixes = self._generated_main_checkout_status_filters()
        commit_outputs, commit_subject = self._generated_board_commit_policy(
            configured_commit_outputs=self.config.reconciliation_guardrail_commit_outputs,
            configured_subject=self.config.reconciliation_guardrail_commit_subject,
        )
        findings = self._run_generated_board_producer(
            producer="reconciliation-guardrail",
            commit_outputs=commit_outputs,
            callback=lambda: record_reconciliation_guardrail_findings(
                todo_path=self.config.todo_path,
                strategy_path=self.config.strategy_path,
                discovery_dir=discovery_dir,
                reconciliation_result=worktree_reconciliation,
                cleanup_result=worktree_cleanup,
                task_prefix=task_id_prefix(self.config.task_prefix),
                max_findings=self.config.reconciliation_guardrail_max_findings,
                discovery_output_path=discovery_output_path,
                commit_outputs=commit_outputs,
                repo_root=self.config.repo_root,
                commit_subject=commit_subject,
                additional_generated_status_paths=generated_paths,
                additional_generated_status_prefixes=generated_prefixes,
            ),
        )
        if findings:
            self._record_event(
                "reconciliation_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def record_retry_budget_guardrails(self) -> list[dict[str, Any]]:
        """Convert repeated daemon blockers into follow-up work before another retry loop."""

        if not self.config.retry_budget_guardrail_enabled:
            return []
        if not self.config.todo_path.exists():
            return []

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            record_retry_budget_findings,
            task_id_prefix,
        )

        discovery_dir = self.config.retry_budget_discovery_dir or self.config.state_dir.parent / "discovery"
        discovery_output_path = self.config.retry_budget_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        commit_outputs, commit_subject = self._generated_board_commit_policy(
            configured_commit_outputs=self.config.retry_budget_commit_outputs,
            configured_subject=self.config.retry_budget_commit_subject,
        )
        findings = self._run_generated_board_producer(
            producer="retry-budget",
            commit_outputs=commit_outputs,
            callback=lambda: record_retry_budget_findings(
                todo_path=self.config.todo_path,
                events_path=self.config.state_dir
                / f"{self.config.state_prefix}_events.jsonl",
                strategy_path=self.config.strategy_path,
                discovery_dir=discovery_dir,
                task_header_prefix_value=self.config.task_prefix,
                task_prefix=task_id_prefix(self.config.task_prefix),
                validation_retry_budget=self.config.validation_retry_budget,
                merge_retry_budget=self.config.merge_retry_budget,
                implementation_retry_budget=self.config.implementation_retry_budget,
                discovery_output_path=discovery_output_path,
                commit_outputs=commit_outputs,
                repo_root=self.config.repo_root,
                commit_subject=commit_subject,
            ),
        )
        if findings:
            self._record_event(
                "retry_budget_guardrail",
                {
                    "generated_count": len(findings),
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "findings": findings,
                },
            )
        return findings

    def _run_supervisor_call_with_timeout(
        self,
        *,
        phase: str,
        timeout_seconds: float,
        timeout_error: type[TimeoutError],
        callback,
    ):
        if timeout_seconds <= 0.0:
            return callback()
        if threading.current_thread() is not threading.main_thread():
            return callback()
        if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
            return callback()
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            return callback()

        def _handle_timeout(_signum, _frame):
            raise timeout_error(f"{phase} exceeded {timeout_seconds:.3f}s")

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            return callback()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _run_objective_refill_with_timeout(self, run_objective_daemon, args: Any) -> dict[str, Any]:
        return self._run_supervisor_call_with_timeout(
            phase="objective refill",
            timeout_seconds=float(self.config.objective_refill_timeout_seconds or 0.0),
            timeout_error=ObjectiveRefillTimeoutError,
            callback=lambda: run_objective_daemon(args),
        )

    def _refresh_objective_goal_completion_artifacts(self) -> dict[str, Any]:
        """Run an explicitly configured artifact producer as bounded argv.

        The command is operator configuration, never data loaded from either
        artifact.  ``shell=False`` prevents artifact text or shell metacharacters
        from becoming executable input.
        """

        protected_path_guard = self._implementation_protected_maintenance_guard()
        if protected_path_guard.get("blocked", False):
            raise ObjectiveCompletionArtifactRefreshError(
                "completion-artifact refresh blocked by active or latched "
                "implementation protected-path fence"
            )
        command_text = str(
            self.config.objective_goal_completion_artifact_refresh_command or ""
        ).strip()
        if not command_text:
            return {"attempted": False, "reason": "not_configured"}
        repo_root = self.config.repo_root.resolve()

        def resolve_from_repo(path: Path) -> Path:
            return (
                path.resolve()
                if path.is_absolute()
                else (repo_root / path).resolve()
            )

        gate_path = (
            resolve_from_repo(self.config.objective_goal_completion_gate_path)
            if self.config.objective_goal_completion_gate_path is not None
            else None
        )
        evidence_path = (
            resolve_from_repo(self.config.objective_goal_completion_evidence_path)
            if self.config.objective_goal_completion_evidence_path is not None
            else None
        )
        artifact_paths = [
            path for path in (gate_path, evidence_path) if path is not None
        ]
        if not artifact_paths:
            raise ObjectiveCompletionArtifactRefreshError(
                "completion-artifact refresh requires a configured gate or evidence path"
            )
        try:
            command = shlex.split(command_text)
        except ValueError as exc:
            raise ObjectiveCompletionArtifactRefreshError(
                f"invalid completion-artifact refresh argv: {exc}"
            ) from exc
        if not command:
            raise ObjectiveCompletionArtifactRefreshError(
                "completion-artifact refresh command is empty"
            )
        timeout_seconds = float(
            self.config.objective_goal_completion_artifact_refresh_timeout_seconds
        )
        if timeout_seconds <= 0.0:
            raise ObjectiveCompletionArtifactRefreshError(
                "completion-artifact refresh timeout must be greater than zero"
            )
        environment = os.environ.copy()
        environment.update(
            {
                "IPFS_ACCELERATE_COMPLETION_REPO_ROOT": str(repo_root),
                "IPFS_ACCELERATE_COMPLETION_OBJECTIVE_PATH": str(
                    resolve_from_repo(self.config.objective_path)
                    if self.config.objective_path
                    else ""
                ),
                "IPFS_ACCELERATE_COMPLETION_GATE_PATH": str(
                    gate_path if gate_path is not None else ""
                ),
                "IPFS_ACCELERATE_COMPLETION_EVIDENCE_PATH": str(
                    evidence_path if evidence_path is not None else ""
                ),
                "IPFS_ACCELERATE_COMPLETION_SCAN_EXCLUDE_PATHS": json.dumps(
                    list(self.config.objective_scan_exclude_paths),
                    separators=(",", ":"),
                ),
            }
        )
        started_at = utc_now()
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                shell=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ObjectiveCompletionArtifactRefreshError(
                f"completion-artifact refresh timed out after {timeout_seconds:.3f}s"
            ) from exc
        except OSError as exc:
            raise ObjectiveCompletionArtifactRefreshError(
                f"completion-artifact refresh could not start: {exc}"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
            raise ObjectiveCompletionArtifactRefreshError(
                "completion-artifact refresh failed with exit code "
                f"{result.returncode}" + (f": {detail}" if detail else "")
            )
        payload = {
            "attempted": True,
            "passed": True,
            "started_at": started_at,
            "finished_at": utc_now(),
            "command": command,
            "timeout_seconds": timeout_seconds,
            "artifact_paths": [str(path) for path in artifact_paths],
        }
        self._record_event("objective_completion_artifacts_refreshed", payload)
        return payload

    def _reconcile_objective_goal_completion_artifacts(
        self,
        *,
        objective_path: Path,
    ) -> dict[str, Any]:
        """Reconcile completion state without authorizing a refill scan."""

        if not self.config.objective_reconcile_goal_completion:
            return {"attempted": False, "reason": "disabled"}
        if not objective_path.is_file():
            return {"attempted": False, "reason": "objective_path_missing"}

        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            completion_evidence_records_from_gate_records,
            load_goal_completion_evidence_records,
            load_goal_completion_gate_records,
            parse_goal_completion_todo_boards,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
            reconcile_objective_goal_completion,
        )

        repo_root = self.config.repo_root.resolve()

        def resolve(path: Path | None) -> Path | None:
            if path is None:
                return None
            return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

        gate_path = resolve(self.config.objective_goal_completion_gate_path)
        evidence_path = resolve(self.config.objective_goal_completion_evidence_path)
        gate_records = load_goal_completion_gate_records(
            gate_path,
            repo_root=repo_root,
        )
        embedded_evidence = completion_evidence_records_from_gate_records(gate_records)
        separate_evidence = load_goal_completion_evidence_records(evidence_path)
        duplicate_goal_ids = sorted(set(embedded_evidence) & set(separate_evidence))
        if duplicate_goal_ids:
            raise ValueError(
                "completion evidence is supplied by both gate and evidence artifacts "
                "for goals: " + ", ".join(duplicate_goal_ids)
            )
        evidence_records = {**embedded_evidence, **separate_evidence}
        todo_boards = parse_goal_completion_todo_boards(
            self.config.objective_goal_completion_todo_boards,
            repo_root=repo_root,
            default_task_prefix=self.config.task_prefix,
        )
        control_paths = [
            path for path in (gate_path, evidence_path) if path is not None
        ]
        result = reconcile_objective_goal_completion(
            repo_root=repo_root,
            objective_path=objective_path.resolve(),
            todo_path=self.config.todo_path.resolve(),
            task_header_prefix=self.config.task_prefix,
            todo_boards=todo_boards,
            completion_evidence_records=evidence_records,
            completion_gate_records=gate_records,
            completion_control_paths=control_paths,
            scan_exclude_paths=self.config.objective_scan_exclude_paths,
            require_artifact_binding=bool(control_paths),
        )
        return {
            "attempted": True,
            "completed_goal_ids": list(result.completed_goal_ids),
            "completed_goal_count": int(result.completed_goal_count),
            "validation_results": dict(result.validation_results),
            "decisions": dict(result.decisions),
        }

    def _run_codebase_refill_with_timeout(self, callback) -> Any:
        return self._run_supervisor_call_with_timeout(
            phase="codebase refill",
            timeout_seconds=float(self.config.codebase_refill_timeout_seconds or 0.0),
            timeout_error=CodebaseRefillTimeoutError,
            callback=callback,
        )

    def _cached_disabled_scan_identity(
        self,
        *,
        scan_mode: str,
        analyzer_version: str,
    ) -> Mapping[str, str] | None:
        """Reuse a prior non-evidentiary identity for an unchanged disabled scan.

        A disabled scanner reports configuration state and is never safe for
        completion reasoning.  Recomputing the dirty repository identity for
        that same report can be expensive in long-running supervisors with
        large generated-state directories.  The persisted projection already
        supplies the exact identity whose receipt will be reused by
        ``persist_supervisor_scan_receipt``.
        """

        if scan_mode != "disabled":
            return None
        strategy = load_json_dict(self.config.strategy_path) or {}
        per_kind = strategy.get("scan_receipts")
        if not isinstance(per_kind, Mapping):
            return None
        for kind_state in per_kind.values():
            if not isinstance(kind_state, Mapping):
                continue
            projection = kind_state.get("latest_attempted_scan")
            if not isinstance(projection, Mapping):
                continue
            if (
                str(projection.get("terminal_reason") or "") != "disabled"
                or str(projection.get("scan_mode") or "") != scan_mode
                or str(projection.get("analyzer_version") or "")
                != analyzer_version
                or projection.get("safe_for_completion_reasoning") is not False
            ):
                continue
            repository_id = str(
                projection.get("repository_id")
                or projection.get("repository_identity")
                or ""
            ).strip()
            tree_id = str(
                projection.get("tree_id")
                or projection.get("tree_identity")
                or ""
            ).strip()
            if repository_id and tree_id:
                return {
                    "repository_id": repository_id,
                    "tree_id": tree_id,
                }
        return None

    def _terminal_refill_result(
        self,
        reason: ScanTerminalReason,
        *,
        scan_mode: str,
        analyzer_version: str,
        started_at: datetime,
        findings: Any = (),
        safe_for_completion_reasoning: bool = False,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RefillScanResult:
        """Build a repository-bound refill receipt for supervisor-owned scans."""

        identity = None
        reason_value = reason.value if isinstance(reason, ScanTerminalReason) else str(reason)
        if reason_value == ScanTerminalReason.DISABLED.value:
            identity = self._cached_disabled_scan_identity(
                scan_mode=scan_mode,
                analyzer_version=analyzer_version,
            )
        return build_scan_result(
            reason,
            scan_mode,
            analyzer_version,
            self.config.repo_root,
            started_at,
            findings,
            safe_for_completion_reasoning=safe_for_completion_reasoning,
            error=error,
            metadata=metadata,
            identity=identity,
        )

    def _persist_refill_result(
        self,
        scan_kind: str,
        result: RefillScanResult,
    ) -> dict[str, Any]:
        """Persist the canonical receipt/event/state record for one attempt."""

        return persist_supervisor_scan_receipt(
            result,
            scan_kind=scan_kind,
            state_dir=self.config.state_dir,
            state_prefix=self.config.state_prefix,
            strategy_path=self.config.strategy_path,
            events_path=self.config.events_path,
        )

    def _adapt_legacy_objective_result(
        self,
        value: Any,
        *,
        scan_mode: str,
        started_at: datetime,
    ) -> RefillScanResult:
        """Explicitly adapt the objective daemon's historical mapping payload."""

        if isinstance(value, RefillScanResult):
            return value
        payload = dict(value) if isinstance(value, Mapping) else {}
        task_ids = list(payload.get("task_ids") or [])
        has_non_task_changes = any(
            payload.get(key)
            for key in (
                "completed_goal_ids",
                "refined_goal_ids",
                "seeded_interoperability_goal_ids",
                "seeded_launch_readiness_goal_ids",
            )
        )
        if not task_ids and has_non_task_changes:
            return self._terminal_refill_result(
                ScanTerminalReason.PARTIAL,
                scan_mode=scan_mode,
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                metadata=payload,
            )
        identity = scan_identity(self.config.repo_root)
        return adapt_legacy_scan_result(
            task_ids,
            empty_reason=ScanTerminalReason.PARTIAL,
            scan_mode=scan_mode,
            analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
            repository_id=identity.repository_id,
            tree_id=identity.tree_id,
            started_at=started_at,
            # The legacy payload does not prove why it is empty.  Even when
            # this call was requested in exhaustive mode, the adapter cannot
            # promote absence into completion evidence.
            safe_for_completion_reasoning=False,
            metadata=payload,
        )

    def _adapt_legacy_codebase_result(
        self,
        value: Any,
        *,
        scan_mode: str,
        started_at: datetime,
    ) -> RefillScanResult:
        """Explicitly adapt list-returning codebase refill callbacks."""

        if isinstance(value, RefillScanResult):
            return value
        findings = list(value) if isinstance(value, (list, tuple)) else []
        identity = scan_identity(self.config.repo_root)
        return adapt_legacy_scan_result(
            findings,
            empty_reason=ScanTerminalReason.PARTIAL,
            scan_mode=scan_mode,
            analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
            repository_id=identity.repository_id,
            tree_id=identity.tree_id,
            started_at=started_at,
            safe_for_completion_reasoning=False,
        )

    def migrate_legacy_objective_goal_completion(self) -> dict[str, Any]:
        """Migrate one bounded batch of ambiguous legacy completion claims.

        The objective markdown is the durable checkpoint: migrated goals carry
        their canonical lifecycle state and migration identity, so the next
        pass naturally resumes at the first remaining legacy goal.  Preview
        runs execute the same classifier without changing that document.
        """

        if not self.config.objective_goal_migration_enabled:
            disabled = {
                "schema": "ipfs_accelerate_py.agent_supervisor.objective_goal_migration@1",
                "schema_version": 1,
                "enabled": False,
                "preview": bool(self.config.objective_goal_migration_preview),
                "changed": False,
                "reason": "disabled",
            }
            strategy = load_json_dict(self.config.strategy_path) or {}
            persist_goal_completion_projection(
                strategy.get("goal_completion_by_goal_id") or {},
                state_dir=self.config.state_dir,
                state_prefix=self.config.state_prefix,
                strategy_path=self.config.strategy_path,
                migration=disabled,
            )
            return disabled

        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            default_objective_path,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
            migrate_legacy_objective_goals,
        )

        objective_path = self.config.objective_path or default_objective_path(self.config.repo_root)
        if not objective_path.exists():
            missing = {
                "schema": "ipfs_accelerate_py.agent_supervisor.objective_goal_migration@1",
                "schema_version": 1,
                "enabled": True,
                "preview": bool(self.config.objective_goal_migration_preview),
                "changed": False,
                "reason": "objective_path_missing",
                "objective_path": str(objective_path),
            }
            strategy = load_json_dict(self.config.strategy_path) or {}
            persist_goal_completion_projection(
                strategy.get("goal_completion_by_goal_id") or {},
                state_dir=self.config.state_dir,
                state_prefix=self.config.state_prefix,
                strategy_path=self.config.strategy_path,
                migration=missing,
            )
            return missing

        batch_size = max(1, int(self.config.objective_goal_migration_batch_size))
        try:
            result = migrate_legacy_objective_goals(
                repo_root=self.config.repo_root,
                objective_path=objective_path,
                todo_path=self.config.todo_path if self.config.todo_path.exists() else None,
                task_header_prefix=self.config.task_prefix,
                preview=bool(self.config.objective_goal_migration_preview),
                max_goals=batch_size,
            )
        except Exception as exc:
            failure = {
                "schema": "ipfs_accelerate_py.agent_supervisor.objective_goal_migration@1",
                "schema_version": 1,
                "enabled": True,
                "preview": bool(self.config.objective_goal_migration_preview),
                "changed": False,
                "reason": "migration_failed",
                "objective_path": str(objective_path),
                "error": f"{type(exc).__name__}: {exc}",
                "analyzer_health": {"healthy": False, "status": "failed"},
            }
            strategy = load_json_dict(self.config.strategy_path) or {}
            persist_goal_completion_projection(
                strategy.get("goal_completion_by_goal_id") or {},
                state_dir=self.config.state_dir,
                state_prefix=self.config.state_prefix,
                strategy_path=self.config.strategy_path,
                migration=failure,
            )
            self._record_event("objective_goal_migration_failed", failure)
            return failure
        payload = result.to_dict()
        payload["enabled"] = True
        payload["batch_size"] = batch_size
        payload["resumable"] = bool(payload.get("remaining_goal_ids"))

        strategy = load_json_dict(self.config.strategy_path) or {}
        prior = strategy.get("goal_completion_by_goal_id")
        diagnostics: dict[str, Any] = dict(prior) if isinstance(prior, Mapping) else {}
        for record in payload.get("records") or ():
            if not isinstance(record, Mapping):
                continue
            goal_id = str(record.get("goal_id") or "").strip()
            if goal_id:
                diagnostics[goal_id] = dict(record)
        projection = persist_goal_completion_projection(
            diagnostics,
            state_dir=self.config.state_dir,
            state_prefix=self.config.state_prefix,
            strategy_path=self.config.strategy_path,
            migration=payload,
        )
        payload["diagnostics"] = projection
        self._record_event(
            "objective_goal_migration_preview"
            if self.config.objective_goal_migration_preview
            else "objective_goal_migration",
            {
                "schema_version": payload.get("schema_version", 1),
                "objective_path": str(objective_path),
                "preview": bool(payload.get("preview")),
                "changed": bool(payload.get("changed")),
                "candidate_goal_ids": list(payload.get("candidate_goal_ids") or []),
                "migrated_goal_ids": list(payload.get("migrated_goal_ids") or []),
                "provisional_goal_ids": list(payload.get("provisional_goal_ids") or []),
                "verified_goal_ids": list(payload.get("verified_goal_ids") or []),
                "remaining_goal_ids": list(payload.get("remaining_goal_ids") or []),
                "resumable": bool(payload["resumable"]),
                "batch_size": batch_size,
            },
        )
        return payload

    def reconcile_objective_task_janitor(
        self,
        *,
        contradictions: tuple[Mapping[str, Any], ...] = (),
    ) -> dict[str, Any]:
        """Keep strategy blocks and objective refills aligned with the goal heap."""

        if not self.config.objective_task_janitor_enabled:
            return {}

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            load_strategy,
            mark_task_statuses_in_todo_text,
            task_id_prefix,
            write_json,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            default_objective_path,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
            parse_goal_heap,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_task_janitor import (
            DEFAULT_MISSION_TERMS,
            reconcile_objective_task_strategy,
            registered_goal_ids_from_bundle_index,
        )
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            parse_task_file,
        )

        objective_path = self.config.objective_path or default_objective_path(self.config.repo_root)
        if not objective_path.exists() or not self.config.todo_path.exists():
            return {}

        try:
            goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
            tasks = parse_task_file(self.config.todo_path, task_id_prefix(self.config.task_prefix))
        except (OSError, UnicodeDecodeError) as exc:
            result = {"changed": False, "reason": "read_failed", "error": str(exc)}
            self._record_event("objective_task_janitor_failed", result)
            return result

        strategy = load_strategy(self.config.strategy_path)
        registered_goal_ids: list[str] = []
        if self.config.objective_bundle_dir is not None:
            bundle_index_path = self.config.objective_bundle_dir / "index.json"
            try:
                bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                bundle_index = {}
            if isinstance(bundle_index, Mapping):
                registered_goal_ids = registered_goal_ids_from_bundle_index(bundle_index)
        mission_terms = tuple(
            dict.fromkeys([*DEFAULT_MISSION_TERMS, *self.config.objective_task_janitor_mission_terms])
        )
        result = reconcile_objective_task_strategy(
            goals=goals,
            tasks=tasks,
            strategy=strategy,
            now=utc_now(),
            mission_terms=mission_terms,
            registered_goal_ids=registered_goal_ids,
            max_blocked_tasks=self.config.objective_task_janitor_max_blocked_tasks,
            max_deprioritized_tasks=self.config.objective_task_janitor_max_deprioritized_tasks,
            max_reopened_goals=self.config.objective_task_janitor_max_reopened_goals,
            # Missing-work reopening relies on completion reconciliation to
            # retire goals after their finite task has passed.  When an
            # operator explicitly disables that reconciliation, forcing every
            # active goal without an open task regenerates already-completed
            # work forever.  Keep contradiction-driven reopening enabled, but
            # let the ordinary low-backlog scan discover genuinely new work.
            reopen_missing_work_goals=self.config.objective_reconcile_goal_completion,
            contradictions=contradictions,
        )
        if result.get("changed"):
            write_json(self.config.strategy_path, result["strategy"])
        materialized_reopenings = self._materialize_objective_goal_reopenings(
            objective_path,
            result,
        )
        materialized = self._materialize_objective_task_janitor_retirements(
            result,
            mark_task_statuses_in_todo_text=mark_task_statuses_in_todo_text,
            task_id_prefix=task_id_prefix,
        )
        event_payload = {
            "changed": bool(result.get("changed")),
            "blocked_task_ids": list(result.get("blocked_task_ids") or []),
            "deprioritized_task_ids": list(result.get("deprioritized_task_ids") or []),
            "materialized_blocked_task_ids": list(materialized.get("blocked_task_ids") or []),
            "materialized_reason_task_ids": list(materialized.get("reason_task_ids") or []),
            "reopened_goal_ids": list(result.get("reopened_goal_ids") or []),
            "missing_work_reopen_enabled": bool(
                result.get("missing_work_reopen_enabled")
            ),
            "contradiction_reopened_goal_ids": list(
                result.get("contradiction_reopened_goal_ids") or []
            ),
            "recalculated_goal_ids": list(result.get("recalculated_goal_ids") or []),
            "newly_scheduled_task_ids": list(result.get("newly_scheduled_task_ids") or []),
            "goal_reopening_receipts": list(result.get("goal_reopening_receipts") or []),
            "materialized_reopened_goal_ids": list(
                materialized_reopenings.get("goal_ids") or []
            ),
            "mission_terms": list(mission_terms),
            "critical_goal_count": len(result.get("critical_goal_ids") or []),
            "active_goal_count": len(result.get("active_goal_ids") or []),
            "scheduled_goal_count": len(result.get("scheduled_goal_ids") or []),
            "registered_goal_count": len(result.get("registered_goal_ids") or []),
        }
        self._record_event("objective_task_janitor", event_payload)
        result.pop("strategy", None)
        result["materialized"] = materialized
        result["materialized_reopenings"] = materialized_reopenings
        return result

    def _materialize_objective_goal_reopenings(
        self,
        objective_path: Path,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist effective reopenings without deleting completion history.

        The janitor's strategy ledger is authoritative for idempotency, while
        the objective heap remains authoritative for schedulability.  Writing
        the effective state back to markdown ensures forced gap generation
        sees the reopened goal in the same supervisor cycle.  Only lifecycle
        and contradiction fields are updated; historical completion evidence
        and validation receipts remain untouched.
        """

        effective_goal_ids = {
            str(item).strip()
            for item in result.get(
                "effective_reopened_goal_ids",
                result.get("contradiction_reopened_goal_ids", ()),
            )
            if str(item).strip()
        }
        receipts = [
            dict(item)
            for item in result.get("goal_reopening_receipts", ())
            if isinstance(item, Mapping)
            and str(item.get("goal_id") or "").strip() in effective_goal_ids
        ]
        if not effective_goal_ids or not receipts:
            return {"changed": False, "goal_ids": [], "reason": "no_effective_reopenings"}

        receipts_by_goal: dict[str, list[dict[str, Any]]] = {}
        for receipt in receipts:
            receipts_by_goal.setdefault(str(receipt.get("goal_id") or "").strip(), []).append(
                receipt
            )
        try:
            text = objective_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "changed": False,
                "goal_ids": [],
                "reason": "objective_read_failed",
                "error": str(exc),
            }

        from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
            rewrite_goal_fields,
        )

        updates: dict[str, dict[str, str]] = {}
        for goal_id in sorted(effective_goal_ids):
            goal_receipts = sorted(
                receipts_by_goal.get(goal_id, ()),
                key=lambda item: (
                    str(item.get("reopened_at") or ""),
                    str(item.get("receipt_id") or ""),
                ),
            )
            if not goal_receipts:
                continue
            latest = goal_receipts[-1]
            contradiction_ids = sorted(
                {
                    str(item).strip()
                    for receipt in goal_receipts
                    for item in receipt.get("contradiction_ids", ())
                    if str(item).strip()
                }
            )
            impacted_criteria = sorted(
                {
                    str(item).strip()
                    for receipt in goal_receipts
                    for item in receipt.get("impacted_criteria", ())
                    if str(item).strip()
                }
            )
            invalidated_evidence = sorted(
                {
                    str(item).strip()
                    for receipt in goal_receipts
                    for item in receipt.get("invalidated_evidence", ())
                    if str(item).strip()
                }
            )
            source_receipts = [
                source
                for receipt in goal_receipts
                for source in receipt.get("source_receipts", ())
                if isinstance(source, Mapping)
            ]
            scheduled_work = [
                work
                for receipt in goal_receipts
                for work in receipt.get("newly_scheduled_work", ())
                if isinstance(work, Mapping)
            ]
            updates[goal_id] = {
                "Status": "reopened",
                "State transitioned at": str(latest.get("reopened_at") or utc_now()),
                "State transition reason": (
                    "completion evidence contradicted; scheduled repair work and "
                    "recalculate parent/dependent goal proof"
                ),
                "Goal reopening receipts": json.dumps(
                    goal_receipts,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "Contradiction ids": json.dumps(contradiction_ids, separators=(",", ":")),
                "Contradiction impacted criteria": json.dumps(
                    impacted_criteria,
                    separators=(",", ":"),
                ),
                "Contradiction invalidated evidence": json.dumps(
                    invalidated_evidence,
                    separators=(",", ":"),
                ),
                "Contradiction source receipts": json.dumps(
                    source_receipts,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "Newly scheduled work": json.dumps(
                    scheduled_work,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        rewritten = rewrite_goal_fields(text, updates)
        if rewritten == text:
            return {"changed": False, "goal_ids": sorted(updates), "reason": "already_materialized"}
        try:
            write_text_atomic(objective_path, rewritten)
        except OSError as exc:
            return {
                "changed": False,
                "goal_ids": [],
                "reason": "objective_write_failed",
                "error": str(exc),
            }
        return {"changed": True, "goal_ids": sorted(updates)}

    @staticmethod
    def _mapped_finding_contradictions(
        findings: tuple[Any, ...] | list[Any],
        *,
        source_receipt: Mapping[str, Any],
        goals: tuple[Any, ...] | list[Any] = (),
    ) -> tuple[dict[str, Any], ...]:
        """Project explicitly goal-mapped findings into janitor contradictions.

        Codebase refill records are intentionally allowed to remain unmapped.
        Such findings still create backlog work, but they must not churn a
        completed objective.  A goal identifier in the finding is therefore
        the boundary between ordinary discovery and completion-invalidating
        evidence.
        """

        inferred_mapping: dict[str, dict[str, Any]] = {}
        criteria_by_goal: dict[str, list[str]] = {}
        if goals and findings:
            from ipfs_accelerate_py.agent_supervisor.objectives.goal_coverage import (
                UNMAPPED_GOAL_ID,
                acceptance_criteria_for_goal,
                attach_findings_to_goals,
            )

            enriched_findings: list[dict[str, Any]] = []
            for raw_finding in findings:
                if not isinstance(raw_finding, Mapping):
                    continue
                enriched = dict(raw_finding)
                source = str(enriched.get("source") or "").strip()
                if source and not any(
                    enriched.get(key)
                    for key in ("outputs", "predicted_files", "changed_files")
                ):
                    path, separator, line = source.rpartition(":")
                    enriched["predicted_files"] = [
                        path if separator and line.isdigit() else source
                    ]
                enriched_findings.append(enriched)
            for assignment in attach_findings_to_goals(goals, enriched_findings):
                if assignment.goal_id != UNMAPPED_GOAL_ID:
                    inferred_mapping[assignment.finding_id] = assignment.to_dict()
            for goal in goals:
                goal_id = str(
                    getattr(goal, "goal_id", "")
                    or (goal.get("goal_id") if isinstance(goal, Mapping) else "")
                ).strip()
                if goal_id:
                    criteria_by_goal[goal_id] = acceptance_criteria_for_goal(goal)

        projected: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw_finding in findings:
            if not isinstance(raw_finding, Mapping):
                continue
            finding = dict(raw_finding)
            goal_ids: list[str] = []
            for key in ("goal_id", "mapped_goal_id"):
                value = str(finding.get(key) or "").strip()
                if value and value not in goal_ids:
                    goal_ids.append(value)
            for key in ("goal_ids", "affected_goal_ids", "goal_packet_goal_ids"):
                values = finding.get(key)
                if isinstance(values, str):
                    candidates = values.split(",")
                elif isinstance(values, (list, tuple, set, frozenset)):
                    candidates = values
                else:
                    candidates = ()
                for candidate in candidates:
                    value = str(candidate or "").strip()
                    if value and value not in goal_ids:
                        goal_ids.append(value)
            impacted_criteria = finding.get(
                "impacted_criteria",
                finding.get("acceptance_criteria", finding.get("criterion", ())),
            )
            if isinstance(impacted_criteria, str):
                impacted_criteria = [impacted_criteria] if impacted_criteria.strip() else []
            elif not isinstance(impacted_criteria, (list, tuple, set, frozenset)):
                impacted_criteria = []
            criteria = [
                str(item).strip() for item in impacted_criteria if str(item).strip()
            ]

            scheduled_work = finding.get(
                "scheduled_work",
                finding.get("newly_scheduled_work", finding.get("follow_up_task_id", ())),
            )
            if isinstance(scheduled_work, str):
                scheduled_work = [scheduled_work] if scheduled_work.strip() else []
            elif not isinstance(scheduled_work, (list, tuple, set, frozenset)):
                scheduled_work = []
            scheduled: list[dict[str, Any]] = []
            for item in scheduled_work:
                if isinstance(item, Mapping):
                    record = dict(item)
                    if record and record not in scheduled:
                        scheduled.append(record)
                    continue
                task_id = str(item).strip()
                record = {"task_id": task_id}
                if task_id and record not in scheduled:
                    scheduled.append(record)
            finding_id = str(
                finding.get("finding_id") or finding.get("fingerprint") or finding.get("source") or ""
            ).strip()
            mapping_evidence = inferred_mapping.get(finding_id, {})
            inferred_goal_id = str(mapping_evidence.get("goal_id") or "").strip()
            if not goal_ids and inferred_goal_id:
                goal_ids.append(inferred_goal_id)
            if not criteria and len(goal_ids) == 1:
                criteria = list(criteria_by_goal.get(goal_ids[0], ()))
            if not goal_ids:
                continue
            description = str(
                finding.get("contradiction")
                or finding.get("description")
                or finding.get("summary")
                or finding.get("kind")
                or "novel mapped finding invalidates prior completion evidence"
            ).strip()
            invalidated_evidence = finding.get("invalidated_evidence") or []
            if isinstance(invalidated_evidence, str):
                invalidated_evidence = (
                    [invalidated_evidence] if invalidated_evidence.strip() else []
                )
            elif not isinstance(
                invalidated_evidence,
                (list, tuple, set, frozenset),
            ):
                invalidated_evidence = []
            for goal_id in goal_ids:
                finding_identity = finding_id or sha1(
                    json.dumps(
                        finding,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                contradiction_id = "contradiction-" + sha1(
                    "\0".join(["mapped_finding", goal_id, finding_identity]).encode(
                        "utf-8"
                    )
                ).hexdigest()
                # The scan receipt is retained as provenance, but it is not
                # part of contradiction identity.  Re-observing the same
                # stable finding in a later scan must replay the original
                # contradiction instead of churning a completed goal.
                dedupe_key = (goal_id, finding_identity, "mapped_finding")
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                projected.append(
                    {
                        "contradiction_id": contradiction_id,
                        "fingerprint": contradiction_id,
                        "kind": "mapped_finding",
                        "goal_id": goal_id,
                        "finding_id": finding_id,
                        "summary": description,
                        "impacted_criteria": criteria,
                        "invalidated_evidence": [
                            str(item).strip()
                            for item in invalidated_evidence
                            if str(item).strip()
                        ],
                        "source_receipt": {
                            **dict(source_receipt),
                            **(
                                {"finding_mapping": mapping_evidence}
                                if mapping_evidence
                                else {}
                            ),
                        },
                        "scheduled_work": scheduled,
                        "newly_scheduled_work": scheduled,
                    }
                )
        return tuple(
            sorted(
                projected,
                key=lambda item: (
                    str(item.get("goal_id") or ""),
                    str(item.get("finding_id") or ""),
                ),
            )
        )

    def _objective_goals_for_finding_mapping(self) -> list[Any]:
        """Read the current heap for deterministic dynamic-finding assignment."""

        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            default_objective_path,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
            parse_goal_heap,
        )

        objective_path = self.config.objective_path or default_objective_path(
            self.config.repo_root
        )
        try:
            return parse_goal_heap(objective_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return []

    def _materialize_objective_task_janitor_retirements(
        self,
        result: Mapping[str, Any],
        *,
        mark_task_statuses_in_todo_text,
        task_id_prefix,
    ) -> dict[str, Any]:
        """Persist janitor retirements into the markdown board so stale work stays out."""

        receipts = result.get("receipts") or []
        if not isinstance(receipts, list) or not receipts:
            return {"changed": False, "reason": "no_receipts"}

        reasons_by_task_id: dict[str, str] = {}
        unblock_task_ids: list[str] = []
        remove_task_ids: list[str] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            task_id = str(receipt.get("task_id") or "").strip()
            action = str(receipt.get("action") or "").strip()
            retired_reason = str(receipt.get("retired_task_reason") or "").strip()
            if not task_id:
                continue
            if action == "unblock":
                unblock_task_ids.append(task_id)
                continue
            if action == "remove":
                remove_task_ids.append(task_id)
                continue
            if action == "block":
                reasons_by_task_id[task_id] = (
                    "Retired by objective-task janitor during launch steering"
                    f" because {retired_reason or 'the referenced goal is no longer active'}."
                )
                continue
            if action == "deprioritize" and retired_reason.startswith("off_mission_"):
                reasons_by_task_id[task_id] = (
                    "Deferred by objective-task janitor during launch steering"
                    f" because {retired_reason}; this keeps lanes focused on Swissknife,"
                    " Hallucinate App, MCP++, Meta glasses, and Playwright launch readiness."
                )

        if not reasons_by_task_id and not unblock_task_ids and not remove_task_ids:
            return {"changed": False, "reason": "no_materializable_receipts"}
        try:
            todo_text = self.config.todo_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"changed": False, "reason": "todo_read_failed", "error": str(exc)}

        task_prefix = task_id_prefix(self.config.task_prefix)
        updated_text = todo_text
        updated_text, unblocked_task_ids = mark_task_statuses_in_todo_text(
            updated_text,
            unblock_task_ids,
            task_prefix=task_prefix,
            status="todo",
        )
        updated_text, removed_task_ids = mark_task_statuses_in_todo_text(
            updated_text,
            remove_task_ids,
            task_prefix=task_prefix,
            status="completed",
        )
        updated_text, removed_reason_task_ids = self._remove_blocked_reason_lines(
            updated_text,
            [*unblocked_task_ids, *removed_task_ids],
            task_prefix=task_prefix,
        )
        updated_text, blocked_task_ids = mark_task_statuses_in_todo_text(
            updated_text,
            list(reasons_by_task_id),
            task_prefix=task_prefix,
            status="blocked",
        )
        updated_text, reason_task_ids = self._ensure_blocked_reason_lines(
            updated_text,
            reasons_by_task_id,
            task_prefix=task_prefix,
        )
        if (
            not blocked_task_ids
            and not reason_task_ids
            and not unblocked_task_ids
            and not removed_task_ids
            and not removed_reason_task_ids
        ):
            return {"changed": False, "reason": "todo_already_materialized"}
        write_text_atomic(self.config.todo_path, updated_text)
        return {
            "changed": True,
            "blocked_task_ids": blocked_task_ids,
            "reason_task_ids": reason_task_ids,
            "unblocked_task_ids": unblocked_task_ids,
            "removed_task_ids": removed_task_ids,
            "removed_reason_task_ids": removed_reason_task_ids,
        }

    @staticmethod
    def _remove_blocked_reason_lines(
        todo_text: str,
        task_ids: Sequence[str],
        *,
        task_prefix: str,
    ) -> tuple[str, list[str]]:
        """Remove stale blocked reason lines from selected task blocks."""

        target_task_ids = {
            str(task_id).strip()
            for task_id in task_ids
            if str(task_id).strip()
        }
        if not target_task_ids:
            return todo_text, []

        lines = todo_text.splitlines(keepends=True)
        output: list[str] = []
        current_task_id = ""
        removed: list[str] = []
        for line in lines:
            if line.startswith(f"## {task_prefix}"):
                parts = line[3:].strip().split(" ", 1)
                current_task_id = parts[0] if parts else ""
                output.append(line)
                continue
            if current_task_id in target_task_ids and line.startswith("- Blocked reason:"):
                removed.append(current_task_id)
                continue
            output.append(line)
        if not removed:
            return todo_text, []
        return "".join(output), sorted(set(removed), key=removed.index)

    @staticmethod
    def _ensure_blocked_reason_lines(
        todo_text: str,
        reasons_by_task_id: Mapping[str, str],
        *,
        task_prefix: str,
    ) -> tuple[str, list[str]]:
        """Add a blocked reason line to each retired task block when missing."""

        target_reasons = {
            str(task_id).strip(): str(reason).strip()
            for task_id, reason in reasons_by_task_id.items()
            if str(task_id).strip() and str(reason).strip()
        }
        if not target_reasons:
            return todo_text, []

        lines = todo_text.splitlines(keepends=True)
        output: list[str] = []
        current_task_id = ""
        status_seen = False
        reason_seen = False
        inserted: list[str] = []

        def flush_reason() -> None:
            nonlocal status_seen, reason_seen
            if current_task_id in target_reasons and status_seen and not reason_seen:
                output.append(f"- Blocked reason: {target_reasons[current_task_id]}\n")
                inserted.append(current_task_id)
                reason_seen = True

        for line in lines:
            if line.startswith(f"## {task_prefix}"):
                flush_reason()
                parts = line[3:].strip().split(" ", 1)
                current_task_id = parts[0] if parts else ""
                status_seen = False
                reason_seen = False
                output.append(line)
                continue
            if current_task_id in target_reasons:
                if line.startswith("- Status:"):
                    status_seen = True
                elif line.startswith("- Blocked reason:"):
                    reason_seen = True
            output.append(line)
        flush_reason()

        if not inserted:
            return todo_text, []
        return "".join(output), inserted

    def refill_objective_backlog(self) -> RefillScanResult:
        """Refine the objective heap and feed todos when the backlog is low or drained."""

        started_at = datetime.now(timezone.utc)
        if not self.config.objective_refill_enabled:
            return self._terminal_refill_result(
                ScanTerminalReason.DISABLED,
                scan_mode="disabled",
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
            )

        from argparse import Namespace

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            load_strategy,
            should_refill_backlog,
            task_id_prefix,
            write_json,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
            default_objective_path,
            discovery_fingerprints,
            run_objective_daemon,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
            DEFAULT_DISCOVERY_OUTPUT_PATH,
            DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
        )
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
            DEFAULT_ROOT_GOAL_TITLE,
            DEFAULT_TRACKING_DOCUMENT_TITLE,
            DEFAULT_ULTIMATE_GOAL,
        )

        objective_path = self.config.objective_path or default_objective_path(self.config.repo_root)
        if not objective_path.exists() and not self.config.objective_ensure_tracking_document:
            return self._terminal_refill_result(
                ScanTerminalReason.FAILED,
                scan_mode="prerequisite_check",
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=f"objective path does not exist: {objective_path}",
            )
        if not self.config.todo_path.exists():
            return self._terminal_refill_result(
                ScanTerminalReason.FAILED,
                scan_mode="prerequisite_check",
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=f"todo path does not exist: {self.config.todo_path}",
            )

        todo_text = self.config.todo_path.read_text(encoding="utf-8")
        strategy = load_strategy(self.config.strategy_path)
        task_prefix = task_id_prefix(self.config.task_prefix)
        force_goal_ids = (
            [
                str(item)
                for item in strategy.get(
                    "objective_task_janitor_force_goal_ids",
                    [],
                )
                if str(item).strip()
            ]
            if (
                self.config.objective_task_janitor_enabled
                and isinstance(
                    strategy.get("objective_task_janitor_force_goal_ids"),
                    list,
                )
            )
            else []
        )
        should_scan, mode, current_open, task_count = should_refill_backlog(
            todo_text=todo_text,
            state_path=self.config.state_path,
            strategy=strategy,
            last_scan_key="last_objective_goal_scan_at",
            last_drained_scan_task_count_key="last_drained_objective_goal_scan_task_count",
            task_prefix=task_prefix,
            min_open_tasks=self.config.objective_scan_min_open_tasks,
            cooldown_seconds=self.config.objective_scan_cooldown_seconds,
            force=bool(force_goal_ids),
        )
        if not should_scan:
            try:
                artifact_refresh = (
                    self._refresh_objective_goal_completion_artifacts()
                )
                completion_reconciliation = (
                    self._reconcile_objective_goal_completion_artifacts(
                        objective_path=objective_path,
                    )
                )
            except (
                ObjectiveCompletionArtifactRefreshError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                self._record_event(
                    "objective_completion_reconciliation_failed",
                    {
                        "error": str(exc),
                        "scan_mode": mode,
                        "objective_path": str(objective_path),
                    },
                )
                return self._terminal_refill_result(
                    ScanTerminalReason.FAILED,
                    scan_mode=f"{mode}_completion_reconciliation",
                    analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                    started_at=started_at,
                    error=str(exc),
                    metadata={
                        "current_open": current_open,
                        "task_count": task_count,
                    },
                )
            strategy["last_objective_completed_goal_ids"] = list(
                completion_reconciliation.get("completed_goal_ids") or []
            )
            strategy["last_objective_completion_validation_results"] = dict(
                completion_reconciliation.get("validation_results") or {}
            )
            strategy["last_objective_completion_decisions"] = dict(
                completion_reconciliation.get("decisions") or {}
            )
            write_json(self.config.strategy_path, strategy)
            return self._terminal_refill_result(
                _scan_skip_reason(mode),
                scan_mode=mode,
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                metadata={
                    "current_open": current_open,
                    "task_count": task_count,
                    "completion_artifact_refresh": artifact_refresh,
                    "completion_reconciliation": completion_reconciliation,
                },
            )

        state_root = self.config.state_dir.parent
        discovery_dir = self.config.objective_discovery_dir or state_root / "discovery"
        bundle_dir = self.config.objective_bundle_dir or state_root / "objective_bundles"
        dataset_dir = self.config.objective_dataset_dir or state_root / "objective_datasets"
        graph_path = self.config.objective_graph_path or state_root / "objective_graph.json"
        discovery_output_path = self.config.objective_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = DEFAULT_DISCOVERY_OUTPUT_PATH

        seen_fingerprints = {
            str(item)
            for item in strategy.get("objective_goal_seen_fingerprints", [])
            if str(item).strip()
        }
        seen_fingerprints.update(discovery_fingerprints(discovery_dir))
        objective_args = Namespace(
            repo_root=self.config.repo_root,
            objective_path=objective_path,
            todo_path=self.config.todo_path,
            protected_output_paths=list(
                self.config.implementation_protected_paths
            ),
            discovery_dir=discovery_dir,
            bundle_dir=bundle_dir,
            dataset_dir=dataset_dir,
            graph_path=graph_path,
            objective_generation_path=state_root / "objective_generation.json",
            task_prefix=task_prefix,
            objective_summary_prefix=(
                self.config.objective_summary_prefix or DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
            ),
            discovery_output_path=discovery_output_path,
            scan_exclude_path=list(self.config.objective_scan_exclude_paths),
            depends_on=list(self.config.objective_scan_depends_on),
            seen_fingerprint=sorted(seen_fingerprints),
            force_goal_id=sorted(set(force_goal_ids)),
            repeat_existing=False,
            max_findings=self.config.objective_scan_max_findings,
            objective_generation_max_new_work=(
                self.config.objective_scan_max_findings
            ),
            ensure_tracking_document=self.config.objective_ensure_tracking_document,
            ultimate_goal=self.config.objective_ultimate_goal or DEFAULT_ULTIMATE_GOAL,
            root_evidence=list(self.config.objective_root_evidence),
            goal_prefix=self.config.objective_goal_prefix,
            root_goal_id=self.config.objective_root_goal_id,
            root_goal_title=self.config.objective_root_goal_title or DEFAULT_ROOT_GOAL_TITLE,
            tracking_document_title=(
                self.config.objective_tracking_document_title or DEFAULT_TRACKING_DOCUMENT_TITLE
            ),
            refine_objective_heap=self.config.objective_refine_goals,
            no_reconcile_goal_completion=not self.config.objective_reconcile_goal_completion,
            objective_goal_completion_todo_board=list(
                self.config.objective_goal_completion_todo_boards
            ),
            objective_goal_completion_gate_path=(
                self.config.objective_goal_completion_gate_path
            ),
            objective_goal_completion_evidence_path=(
                self.config.objective_goal_completion_evidence_path
            ),
            seed_interoperability_goals=self.config.objective_seed_interoperability_goals,
            seed_launch_readiness_goals=self.config.objective_seed_launch_readiness_goals,
            interoperability_focus=list(self.config.objective_interoperability_focus),
            interoperability_component_path=list(
                self.config.objective_interoperability_component_paths
                or self.config.worktree_submodule_paths
            ),
            max_interoperability_goals=self.config.objective_max_interoperability_goals,
            max_launch_readiness_goals=self.config.objective_max_launch_readiness_goals,
            max_refinement_children=self.config.objective_max_refinement_children,
            max_refinement_depth=self.config.objective_max_refinement_depth,
            no_persist_ast_dataset=not self.config.objective_persist_ast_dataset,
            no_todo_vector_index=not self.config.objective_write_todo_vector_index,
            todo_vector_index_path=self.config.objective_todo_vector_index_path,
            surplus_findings_per_goal=self.config.objective_surplus_findings_per_goal,
            surplus_min_terms_per_todo=self.config.objective_surplus_min_terms_per_todo,
            submit_bundles=False,
            queue_path=None,
            queue_task_type="codex.todo_bundle",
            queue_model_name="codex",
            log_level="INFO",
        )
        try:
            artifact_refresh = (
                self._refresh_objective_goal_completion_artifacts()
            )
        except ObjectiveCompletionArtifactRefreshError as exc:
            self._record_event(
                "objective_completion_artifact_refresh_failed",
                {
                    "error": str(exc),
                    "gate_path": str(
                        self.config.objective_goal_completion_gate_path or ""
                    ),
                    "evidence_path": str(
                        self.config.objective_goal_completion_evidence_path or ""
                    ),
                },
            )
            return self._terminal_refill_result(
                ScanTerminalReason.FAILED,
                scan_mode=mode,
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=str(exc),
                metadata={
                    "completion_artifact_refresh": {
                        "attempted": True,
                        "passed": False,
                        "error": str(exc),
                    }
                },
            )
        try:
            payload = self._run_objective_refill_with_timeout(run_objective_daemon, objective_args)
        except ObjectiveRefillTimeoutError as exc:
            strategy = load_strategy(self.config.strategy_path)
            strategy["last_objective_goal_scan_at"] = utc_now()
            strategy["last_objective_goal_scan_mode"] = f"{mode}_timeout"
            strategy["last_objective_refill_timeout_at"] = utc_now()
            strategy["last_objective_refill_timeout_seconds"] = float(
                self.config.objective_refill_timeout_seconds or 0.0
            )
            strategy["last_objective_refill_timeout_error"] = str(exc)
            write_json(self.config.strategy_path, strategy)
            payload = {
                "generated_count": 0,
                "task_ids": [],
                "refined_goal_ids": [],
                "completed_goal_ids": [],
                "seeded_interoperability_goal_ids": [],
                "seeded_launch_readiness_goal_ids": [],
                "objective_refill_timed_out": True,
                "objective_refill_timeout_seconds": float(
                    self.config.objective_refill_timeout_seconds or 0.0
                ),
            }
            self._record_event(
                "objective_refill_timeout",
                {
                    "mode": mode,
                    "objective_path": str(objective_path),
                    "timeout_seconds": payload["objective_refill_timeout_seconds"],
                    "error": str(exc),
                },
            )
            return self._terminal_refill_result(
                ScanTerminalReason.TIMED_OUT,
                scan_mode=mode,
                analyzer_version=OBJECTIVE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=str(exc),
                metadata=payload,
            )

        payload["completion_artifact_refresh"] = artifact_refresh
        result = self._adapt_legacy_objective_result(
            payload,
            scan_mode=mode,
            started_at=started_at,
        )
        payload = dict(result.metadata)

        strategy = load_strategy(self.config.strategy_path)
        strategy["last_objective_goal_scan_at"] = utc_now()
        strategy["last_objective_goal_scan_mode"] = mode
        if current_open == 0 or mode.endswith("drained_exhaustive"):
            strategy["last_drained_objective_goal_scan_task_count"] = task_count
        strategy["objective_goal_seen_fingerprints"] = sorted(discovery_fingerprints(discovery_dir))
        strategy["last_objective_refined_goal_ids"] = list(payload.get("refined_goal_ids") or [])
        strategy["last_objective_completed_goal_ids"] = list(payload.get("completed_goal_ids") or [])
        strategy["last_objective_completion_validation_results"] = dict(
            payload.get("objective_completion_validation_results") or {}
        )
        completion_decisions = dict(payload.get("objective_completion_decisions") or {})
        strategy["last_objective_completion_decisions"] = completion_decisions
        strategy["last_objective_seeded_interoperability_goal_ids"] = list(
            payload.get("seeded_interoperability_goal_ids") or []
        )
        strategy["last_objective_seeded_launch_readiness_goal_ids"] = list(
            payload.get("seeded_launch_readiness_goal_ids") or []
        )
        strategy["last_objective_generated_task_ids"] = list(payload.get("task_ids") or [])
        strategy["last_objective_todo_vector_index_path"] = str(payload.get("todo_vector_index_path") or "")
        strategy["last_objective_surplus_findings_per_goal"] = int(
            payload.get("surplus_findings_per_goal") or DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL
        )
        strategy["last_objective_surplus_min_terms_per_todo"] = int(
            payload.get("surplus_min_terms_per_todo") or DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO
        )
        strategy["last_objective_goal_count"] = int(payload.get("objective_goal_count") or 0)
        strategy["last_objective_active_goal_count"] = int(payload.get("objective_active_goal_count") or 0)
        strategy["last_objective_completed_goal_count"] = int(payload.get("objective_completed_goal_count") or 0)
        strategy["last_objective_heap_schedule_count"] = int(payload.get("objective_heap_schedule_count") or 0)
        strategy["last_objective_task_janitor_force_goal_ids"] = sorted(set(force_goal_ids))
        write_json(self.config.strategy_path, strategy)

        prior_diagnostics = strategy.get("goal_completion_by_goal_id")
        combined_diagnostics = (
            dict(prior_diagnostics) if isinstance(prior_diagnostics, Mapping) else {}
        )
        combined_diagnostics.update(completion_decisions)
        if combined_diagnostics:
            persist_goal_completion_projection(
                combined_diagnostics,
                state_dir=self.config.state_dir,
                state_prefix=self.config.state_prefix,
                strategy_path=self.config.strategy_path,
                migration=strategy.get("goal_completion_migration"),
            )

        return result

    def refill_codebase_backlog(self) -> RefillScanResult:
        """Feed low or drained todo boards from a codebase/submodule scan."""

        started_at = datetime.now(timezone.utc)
        if not self.config.codebase_refill_enabled:
            return self._terminal_refill_result(
                ScanTerminalReason.DISABLED,
                scan_mode="disabled",
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
            )

        from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
            CODEBASE_SCAN_SKIP_PREFIXES,
            load_strategy,
            record_codebase_scan_findings,
            should_refill_backlog,
            task_id_prefix,
            write_json,
        )

        if not self.config.todo_path.exists():
            return self._terminal_refill_result(
                ScanTerminalReason.FAILED,
                scan_mode="prerequisite_check",
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=f"todo path does not exist: {self.config.todo_path}",
            )
        discovery_dir = self.config.codebase_scan_discovery_dir or self.config.state_dir.parent / "discovery"
        discovery_output_path = self.config.codebase_scan_discovery_output_path
        if not discovery_output_path:
            try:
                discovery_output_path = discovery_dir.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                discovery_output_path = str(discovery_dir)
        task_prefix = task_id_prefix(self.config.task_prefix)
        todo_text = self.config.todo_path.read_text(encoding="utf-8")
        strategy = load_strategy(self.config.strategy_path)
        should_scan, mode, current_open, task_count = should_refill_backlog(
            todo_text=todo_text,
            state_path=self.config.state_path,
            strategy=strategy,
            last_scan_key="last_codebase_scan_at",
            last_drained_scan_task_count_key="last_drained_codebase_scan_task_count",
            task_prefix=task_prefix,
            min_open_tasks=self.config.codebase_scan_min_open_tasks,
            cooldown_seconds=self.config.codebase_scan_cooldown_seconds,
        )
        if not should_scan:
            return self._terminal_refill_result(
                _scan_skip_reason(mode),
                scan_mode=mode,
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                metadata={"current_open": current_open, "task_count": task_count},
            )

        def run_refill() -> RefillScanResult:
            return record_codebase_scan_findings(
                todo_path=self.config.todo_path,
                state_path=self.config.state_path,
                strategy_path=self.config.strategy_path,
                discovery_dir=discovery_dir,
                repo_root=self.config.repo_root,
                bundle_dir=self.config.objective_bundle_dir
                or self.config.state_dir.parent / "objective_bundles",
                task_prefix=task_prefix,
                depends_on=self.config.codebase_scan_depends_on,
                min_open_tasks=self.config.codebase_scan_min_open_tasks,
                max_findings=self.config.codebase_scan_max_findings,
                cooldown_seconds=self.config.codebase_scan_cooldown_seconds,
                discovery_output_path=discovery_output_path,
                skip_prefixes=self.config.codebase_scan_skip_prefixes or CODEBASE_SCAN_SKIP_PREFIXES,
                objective_path=self.config.objective_path,
                mission_terms=self.config.objective_task_janitor_mission_terms,
                allow_unscoped_codebase_refill=self.config.allow_unscoped_codebase_refill,
                commit_outputs=self.config.codebase_scan_commit_outputs,
                commit_subject=self.config.codebase_scan_commit_subject,
            )

        try:
            callback_result = self._run_codebase_refill_with_timeout(run_refill)
        except CodebaseRefillTimeoutError as exc:
            strategy = load_strategy(self.config.strategy_path)
            strategy["last_codebase_scan_at"] = utc_now()
            strategy["last_codebase_scan_mode"] = f"{mode}_timeout"
            strategy["last_codebase_refill_timeout_at"] = utc_now()
            strategy["last_codebase_refill_timeout_seconds"] = float(
                self.config.codebase_refill_timeout_seconds or 0.0
            )
            strategy["last_codebase_refill_timeout_error"] = str(exc)
            if current_open == 0 or mode.endswith("drained_exhaustive"):
                strategy["last_drained_codebase_scan_task_count"] = task_count
            write_json(self.config.strategy_path, strategy)
            self._record_event(
                "codebase_refill_timeout",
                {
                    "mode": mode,
                    "todo_path": str(self.config.todo_path),
                    "discovery_dir": str(discovery_dir),
                    "repo_root": str(self.config.repo_root),
                    "timeout_seconds": float(self.config.codebase_refill_timeout_seconds or 0.0),
                    "error": str(exc),
                },
            )
            return self._terminal_refill_result(
                ScanTerminalReason.TIMED_OUT,
                scan_mode=mode,
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=str(exc),
                metadata={"timeout_seconds": float(self.config.codebase_refill_timeout_seconds or 0.0)},
            )
        except Exception as exc:
            failure = {
                "todo_path": str(self.config.todo_path),
                "discovery_dir": str(discovery_dir),
                "repo_root": str(self.config.repo_root),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            logger.warning("Codebase backlog refill failed; leaving supervisor alive", exc_info=True)
            self._record_event("codebase_refill_failed", failure)
            return self._terminal_refill_result(
                ScanTerminalReason.FAILED,
                scan_mode=mode,
                analyzer_version=CODEBASE_REFILL_ANALYZER_VERSION,
                started_at=started_at,
                error=str(exc),
                metadata=failure,
            )
        result = self._adapt_legacy_codebase_result(
            callback_result,
            scan_mode=mode,
            started_at=started_at,
        )
        return result

    def _supervisor_checkout_transaction_depth(self) -> int:
        try:
            return max(
                0,
                int(
                    getattr(
                        self._checkout_mutation_context,
                        "transaction_depth",
                        0,
                    )
                    or 0
                ),
            )
        except (TypeError, ValueError):
            return 0

    def _supervisor_recovery_owner_is_active(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        candidate = dict(metadata)
        candidate.pop("protected_recovery_required", None)
        return checkout_lock_owner_is_active(
            candidate,
            expected_kind="merge",
            expected_repo_root=self.config.repo_root,
            process_command_line=process_command_line,
            process_is_running=process_is_running,
        )

    def _supervisor_recovery_journal_error(
        self,
        metadata: Mapping[str, Any],
    ) -> str:
        if str(metadata.get("kind") or "") != "merge":
            return "kind_mismatch"
        try:
            if Path(str(metadata.get("repo_root") or "")).resolve() != (
                self.config.repo_root.resolve()
            ):
                return "repository_mismatch"
        except (OSError, RuntimeError, ValueError):
            return "repository_invalid"
        protected_paths = metadata.get("protected_paths")
        expected_paths = list(self.config.implementation_protected_paths)
        if (
            not isinstance(protected_paths, list)
            or [str(path) for path in protected_paths] != expected_paths
        ):
            return "protected_paths_mismatch"

        guard = metadata.get("protected_release_guard")
        if not isinstance(guard, Mapping):
            return "guard_missing"
        normalized_guard = dict(guard)
        guard_id = str(normalized_guard.pop("guard_id", "") or "")
        if not guard_id or content_identity(normalized_guard) != guard_id:
            return "guard_identity_mismatch"
        if [
            str(path)
            for path in normalized_guard.get("protected_paths", ())
        ] != expected_paths:
            return "guard_paths_mismatch"

        intent = metadata.get("protected_recovery_intent")
        if not isinstance(intent, Mapping):
            return "intent_missing"
        normalized_intent = dict(intent)
        intent_id = str(normalized_intent.pop("intent_id", "") or "")
        if not intent_id or content_identity(normalized_intent) != intent_id:
            return "intent_identity_mismatch"
        if [
            str(path) for path in intent.get("protected_paths", ())
        ] != expected_paths:
            return "intent_paths_mismatch"
        if str(intent.get("guard_id") or "") != guard_id:
            return "intent_guard_mismatch"
        if not str(intent.get("operation") or "") or not str(
            intent.get("producer") or ""
        ):
            return "intent_operation_missing"
        return ""

    def _attach_supervisor_protected_recovery(
        self,
        lease: CheckoutMutationLease,
    ) -> None:
        intent = lease.metadata["protected_recovery_intent"]
        self._checkout_mutation_context.lease = lease
        self._checkout_mutation_context.transaction_depth = 0
        self._checkout_mutation_context.retain_until_protected_clean = True
        self._checkout_mutation_context.retained_operation = str(
            intent.get("operation") or ""
        )
        self._checkout_mutation_context.retained_producer = str(
            intent.get("producer") or ""
        )
        self._checkout_mutation_context.generated_protected_release_guard = (
            dict(lease.metadata["protected_release_guard"])
        )

    def _adopt_supervisor_protected_recovery(
        self,
    ) -> dict[str, Any]:
        existing = read_checkout_mutation_lease(
            self._repo_merge_lock_path()
        )
        if existing is None or (
            existing.metadata.get("protected_recovery_required") is not True
        ):
            return {"required": False, "adopted": False}
        if str(
            existing.metadata.get("protected_recovery_owner") or ""
        ) != "implementation_supervisor":
            return {
                "required": True,
                "adopted": False,
                "blocked": True,
                "reason": "external_protected_checkout_recovery_required",
                "lock_path": str(existing.lock_path),
            }

        journal_error = self._supervisor_recovery_journal_error(
            existing.metadata
        )
        if journal_error:
            return {
                "required": True,
                "adopted": False,
                "blocked": True,
                "reason": "supervisor_protected_recovery_journal_invalid",
                "journal_error": journal_error,
                "lock_path": str(existing.lock_path),
            }
        try:
            owner_pid = int(existing.metadata.get("pid") or 0)
        except (TypeError, ValueError):
            owner_pid = 0
        if owner_pid == os.getpid():
            self._attach_supervisor_protected_recovery(existing)
            return {
                "required": True,
                "adopted": False,
                "attached": True,
                "lease": existing,
            }
        if self._supervisor_recovery_owner_is_active(
            dict(existing.metadata)
        ):
            return {
                "required": True,
                "adopted": False,
                "blocked": True,
                "reason": "supervisor_protected_recovery_owner_active",
                "lock_path": str(existing.lock_path),
                "lock_owner_pid": owner_pid,
            }

        intent = existing.metadata["protected_recovery_intent"]
        adopted_metadata = {
            **dict(existing.metadata),
            "pid": os.getpid(),
            "owner_script": Path(sys.argv[0]).name,
            "adopted_at": utc_now(),
            "adopted_from_lease_id": existing.lease_id,
        }
        adopted_metadata["lease_id"] = content_identity(
            {
                "kind": "adopted-supervisor-protected-recovery",
                "prior_lease_id": existing.lease_id,
                "intent_id": str(intent.get("intent_id") or ""),
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "issued_ns": time.time_ns(),
            }
        )
        adopted = adopt_inactive_checkout_mutation_lease(
            existing,
            adopted_metadata,
            owner_active=self._supervisor_recovery_owner_is_active,
        )
        if adopted is None:
            return {
                "required": True,
                "adopted": False,
                "blocked": True,
                "reason": "supervisor_protected_recovery_adoption_raced",
                "lock_path": str(existing.lock_path),
            }
        self._attach_supervisor_protected_recovery(adopted)
        return {
            "required": True,
            "adopted": True,
            "lease": adopted,
        }

    def _retained_generated_checkout_lease(self) -> bool:
        return bool(
            self._current_supervisor_checkout_lease() is not None
            and self._supervisor_checkout_transaction_depth() == 0
            and getattr(
                self._checkout_mutation_context,
                "retain_until_protected_clean",
                False,
            )
        )

    def _recover_retained_generated_checkout_lease(self) -> dict[str, Any]:
        """Autonomously clean a retained generated-output transaction."""

        if not self._retained_generated_checkout_lease():
            adoption = self._adopt_supervisor_protected_recovery()
            if adoption.get("required") and adoption.get("blocked"):
                return {
                    **adoption,
                    "attempted": False,
                    "recovered": False,
                    "retained_lease": True,
                }
            if not self._retained_generated_checkout_lease():
                return {
                    "attempted": False,
                    "recovered": False,
                    "retained_lease": False,
                    "reason": "no_retained_generated_checkout_lease",
                }
        else:
            adoption = {"required": True, "adopted": False}
        try:
            repair = self.repair_generated_dirty_checkouts(force=True)
        except Exception as exc:
            result = {
                "attempted": True,
                "recovered": False,
                "retained_lease": self._retained_generated_checkout_lease(),
                "reason": "retained_generated_checkout_recovery_failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            try:
                self._record_event(
                    "retained_generated_checkout_recovery_failed",
                    result,
                )
            except Exception:
                logger.warning(
                    "Failed to record retained checkout recovery failure",
                    exc_info=True,
                )
            return result

        retained = self._retained_generated_checkout_lease()
        result = {
            "attempted": True,
            "recovered": not retained,
            "retained_lease": retained,
            "reason": (
                "retained_generated_checkout_recovered"
                if not retained
                else "retained_generated_checkout_recovery_incomplete"
            ),
            "repair": dict(repair),
            "adoption": {
                key: value
                for key, value in adoption.items()
                if key != "lease"
            },
        }
        try:
            self._record_event(
                (
                    "retained_generated_checkout_recovered"
                    if not retained
                    else "retained_generated_checkout_recovery_failed"
                ),
                result,
            )
        except Exception:
            logger.warning(
                "Failed to record retained checkout recovery result",
                exc_info=True,
            )
        return result

    def _run_retained_generated_checkout_recovery(
        self,
        lease: CheckoutMutationLease,
        *,
        operation: str,
        producer: str,
        callback,
    ):
        if (
            operation == "generated_dirty_repair"
            and str(lease.metadata.get("operation") or "")
            != "generated_dirty_repair"
        ):
            recovery_metadata = {
                **dict(lease.metadata),
                "operation": "generated_dirty_repair",
                "retained_operation": str(
                    getattr(
                        self._checkout_mutation_context,
                        "retained_operation",
                        "",
                    )
                    or ""
                ),
                "retained_producer": str(
                    getattr(
                        self._checkout_mutation_context,
                        "retained_producer",
                        "",
                    )
                    or ""
                ),
            }
            updated_lease = update_checkout_mutation_lease(
                lease,
                recovery_metadata,
            )
            if updated_lease is None:
                raise RuntimeError(
                    "checkout_mutation_protected_recovery_incomplete: "
                    "checkout_mutation_lease_update_failed"
                )
            lease = updated_lease
            self._checkout_mutation_context.lease = lease
        self._checkout_mutation_context.transaction_depth = 1
        try:
            result = callback()
        except BaseException:
            self._checkout_mutation_context.transaction_depth = 0
            raise
        self._checkout_mutation_context.transaction_depth = 0
        release_guard = getattr(
            self._checkout_mutation_context,
            "generated_protected_release_guard",
            None,
        )
        release_verdict = self._safe_generated_protected_release_guard(
            release_guard
        )
        if not release_verdict.get("release_allowed"):
            self._record_generated_checkout_retention(
                lease,
                operation=operation,
                producer=producer,
                release_guard=release_guard,
                release_verdict=release_verdict,
            )
            raise RuntimeError(
                "checkout_mutation_protected_recovery_incomplete: "
                f"{release_verdict.get('reason') or 'unknown'}"
            )
        release_error = self._clear_and_release_supervisor_checkout_lease(
            lease,
            operation=operation,
        )
        if release_error:
            release_verdict = {
                "release_allowed": False,
                "reason": "checkout_mutation_lease_release_failed",
                "error": release_error,
            }
            self._record_generated_checkout_retention(
                lease,
                operation=operation,
                producer=producer,
                release_guard=release_guard,
                release_verdict=release_verdict,
            )
            raise RuntimeError(
                "checkout_mutation_protected_recovery_incomplete: "
                "checkout_mutation_lease_release_failed"
            )
        return result

    def _implementation_attempt_is_active(self, state: PortalTaskState, *, now_ts: float) -> bool:
        if not state.active_task_id or not state.implementation_in_progress:
            return False
        if state.last_implementation_task_id and state.last_implementation_task_id != state.active_task_id:
            return False
        started_at = parse_timestamp(state.last_implementation_started_at or state.active_phase_started_at)
        if started_at is None:
            return False
        finished_at = parse_timestamp(state.last_implementation_finished_at)
        if finished_at is not None and finished_at >= started_at:
            return False
        grace_seconds = max(30.0, float(self.config.check_interval) * 2.0)
        max_age_seconds = max(
            float(self.config.stale_seconds),
            self._implementation_watchdog_timeout_seconds(),
        )
        return max(0.0, now_ts - started_at.timestamp()) <= max_age_seconds + grace_seconds

    def _implementation_log_stall_reason(self, state: PortalTaskState, *, now_ts: float) -> str:
        if not state.active_task_id or not state.implementation_in_progress:
            return ""
        # Agent and validation subprocesses can remain quiet while making
        # progress. Their implementation timeout is the authoritative bound.
        if self._implementation_attempt_is_active(state, now_ts=now_ts) and (
            self._active_agent_worker_processes()
            or state.active_phase in {"validating", "merge_reconciliation", "merge_resolver"}
            or self._active_validation_subprocess_exists()
        ):
            return ""
        threshold = max(0.0, float(self.config.implementation_log_stall_seconds))
        if threshold <= 0.0:
            return ""
        log_text = state.last_implementation_log_path or state.active_log_path
        if not log_text:
            return ""
        log_path = Path(log_text)
        if not log_path.is_absolute():
            log_path = self.config.repo_root / log_path
        try:
            stat = log_path.stat()
        except OSError:
            return ""
        age_seconds = max(0.0, now_ts - stat.st_mtime)
        if age_seconds <= threshold:
            return ""
        return (
            f"implementation log stalled for active task {state.active_task_id}: "
            f"{age_seconds:.0f}s without output in {log_path}"
        )

    def _active_agent_worker_processes(self) -> list[dict[str, Any]]:
        daemon_pid = self._read_managed_daemon_pid()
        if not daemon_pid:
            return []
        return active_codex_exec_workers(daemon_pid)

    def _active_validation_subprocess_exists(self) -> bool:
        """Return whether a managed agent is currently running a bounded test command."""

        daemon_pid = self._read_managed_daemon_pid()
        if not daemon_pid:
            return False
        markers = (
            "playwright",
            "run_playwright_test.mjs",
            "pytest",
            "vitest",
            "npm run test",
            "npm run evidence",
            "release-readiness-gate",
            "audit-release-evidence-freshness",
            "build-virtual-desktop-release-evidence",
            "tsc --noemit",
            "run_legal_ir_10m_smoke.sh",
            "run_legal_ir_8h_canary.sh",
            "run_hammer_leanstral_smoke.sh",
            "run_hammer_leanstral_hparam.sh",
            "uscode_modal_daemon_runner",
        )
        return any(
            any(marker in " ".join(item.get("cmdline") or ()).lower() for marker in markers)
            for item in descendant_processes(daemon_pid)
        )

    def is_stuck(
        self,
        state: PortalTaskState,
        *,
        now_ts: float,
        ignore_progress_until_ts: float | None = None,
    ) -> tuple[bool, str]:
        worktree_phase_stall_reason = self._worktree_phase_without_worker_reason(state, now_ts=now_ts)
        if worktree_phase_stall_reason:
            return True, worktree_phase_stall_reason
        log_stall_reason = self._implementation_log_stall_reason(state, now_ts=now_ts)
        if log_stall_reason:
            return True, log_stall_reason
        if self._implementation_attempt_is_active(state, now_ts=now_ts):
            return False, ""
        heartbeat_age = self._age_seconds(state.heartbeat_at, now_ts)
        progress_age = self._age_seconds(state.last_progress_at, now_ts)
        stale = self.config.stale_seconds
        if state.active_task_id and heartbeat_age > stale:
            return True, f"heartbeat stale for active task {state.active_task_id}"
        if (
            state.active_task_id
            and state.active_phase in {"merge_reconciliation", "merge_resolver"}
            and heartbeat_age <= stale
        ):
            return False, ""
        if ignore_progress_until_ts is not None and now_ts < ignore_progress_until_ts:
            return False, ""
        if state.active_task_id and state.ready_count > 0 and progress_age > stale:
            return True, f"no progress on active task {state.active_task_id}"
        if (
            state.active_task_id
            and state.last_implementation_task_id == state.active_task_id
            and state.last_implementation_commit
            and state.last_merge_returncode not in (None, 0)
            and not state.last_merge_commit
        ):
            detail = state.last_merge_error or "merge failed without a merge commit"
            return True, f"unresolved merge failure on active task {state.active_task_id}: {detail}"
        return False, ""

    def _worktree_phase_without_worker_reason(self, state: PortalTaskState, *, now_ts: float) -> str:
        if not state.active_task_id:
            return ""
        threshold = max(30.0, float(self.config.implementation_log_stall_seconds))
        worker_status = worktree_phase_worker_status(
            {
                "active_phase": state.active_phase,
                "active_phase_started_at": state.active_phase_started_at,
            },
            self._read_managed_daemon_pid(),
            threshold,
            now=datetime.fromtimestamp(now_ts, tz=timezone.utc),
        )
        phase = str(worker_status.get("phase") or "")
        if not worker_status.get("required"):
            self._worktree_worker_phase = ""
            self._last_worktree_worker_seen_monotonic = None
        elif phase != self._worktree_worker_phase:
            self._worktree_worker_phase = phase
            self._last_worktree_worker_seen_monotonic = None

        now_monotonic = time.monotonic()
        if int(worker_status.get("active_worker_count") or 0) > 0:
            self._last_worktree_worker_seen_monotonic = now_monotonic
            worker_status["worker_absence_age_seconds"] = 0.0
            worker_status["stalled_without_active_worker"] = False
        elif self._last_worktree_worker_seen_monotonic is not None:
            absence_age = max(
                0.0,
                now_monotonic - self._last_worktree_worker_seen_monotonic,
            )
            worker_status["worker_absence_age_seconds"] = round(absence_age, 3)
            worker_status["stalled_without_active_worker"] = bool(
                threshold > 0 and absence_age >= threshold
            )
        else:
            worker_status["worker_absence_age_seconds"] = None
        if not worker_status.get("stalled_without_active_worker"):
            return ""
        self._record_event(
            "worktree_phase_without_worker",
            {
                "active_task_id": state.active_task_id,
                "active_phase": state.active_phase,
                "active_phase_detail": state.active_phase_detail,
                "worker_status": worker_status,
            },
        )
        stall_age = worker_status.get("worker_absence_age_seconds")
        if stall_age is None:
            stall_age = worker_status.get("phase_age_seconds")
        return (
            f"{state.active_phase} stalled for active task {state.active_task_id}: "
            f"no active worker for {stall_age}s"
        )

    def rewrite_strategy(self, state: PortalTaskState, reason: str) -> dict[str, Any]:
        strategy = self._load_strategy()
        active_task_id = state.active_task_id.strip()
        active_track = state.active_task_track.strip().lower()
        focus_tracks = normalize_focus_tracks(strategy.get("focus_tracks", DEFAULT_TRACKS))
        generation = int(strategy.get("generation", 0)) + 1
        deprioritized_tasks = list(dict.fromkeys([*strategy.get("deprioritized_tasks", []), active_task_id]))
        blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
        reason_lower = reason.lower()
        should_block_active_task = bool(active_task_id) and (
            state.active_phase in {"merge_reconciliation", "merge_resolver"}
            or "merge_reconciliation" in reason_lower
            or "merge_resolver" in reason_lower
            or "merge conflict" in reason_lower
            or "merge_retry" in reason_lower
            or "unresolved merge failure" in reason_lower
        )
        blocked_active_task = False
        if should_block_active_task and active_task_id not in blocked_tasks:
            blocked_tasks.append(active_task_id)
            blocked_active_task = True

        if active_track and active_track in focus_tracks:
            focus_tracks = [track for track in focus_tracks if track != active_track] + [active_track]
            focus_tracks = normalize_focus_tracks(focus_tracks)

        strategy.update(
            {
                "generation": generation,
                "focus_tracks": focus_tracks or DEFAULT_TRACKS,
                "blocked_tasks": blocked_tasks,
                "deprioritized_tasks": [task_id for task_id in deprioritized_tasks if task_id],
                "last_rewrite_at": utc_now(),
                "last_rewrite_reason": reason,
            }
        )
        write_json_atomic(self.config.strategy_path, strategy)
        self._record_event(
            "strategy_rewrite",
            {
                "reason": reason,
                "generation": generation,
                "active_task_id": active_task_id,
                "active_track": active_track,
                "blocked_active_task": blocked_active_task,
            },
        )
        return strategy

    def repair_blocked_progress_state(
        self,
        state: PortalTaskState,
        reason: str,
        *,
        now_ts: float,
    ) -> dict[str, Any]:
        """Clear stale active-task state after strategy has recorded the blocker."""

        if not state.active_task_id:
            return {"repaired": False, "reason": "no_active_task"}
        if self._implementation_attempt_is_active(state, now_ts=now_ts) and "no active worker" not in reason:
            return {"repaired": False, "reason": "implementation_attempt_active"}
        if reason.startswith("implementation log stalled"):
            return {"repaired": False, "reason": "implementation_log_stalled"}

        previous = {
            "active_task_id": state.active_task_id,
            "active_task_title": state.active_task_title,
            "active_task_track": state.active_task_track,
            "active_task_started_at": state.active_task_started_at,
            "active_attempt": state.active_attempt,
            "active_phase": state.active_phase,
            "active_phase_detail": state.active_phase_detail,
            "active_log_path": state.active_log_path,
            "active_worktree_path": state.active_worktree_path,
            "active_branch": state.active_branch,
            "implementation_in_progress": state.implementation_in_progress,
        }
        repaired_at = utc_now()
        recovered_attempt = (
            consume_stale_active_attempt(state)
            if state.implementation_in_progress
            else {"consumed": False, "reason": "no_inflight_attempt"}
        )
        state.active_task_id = ""
        state.active_task_key = ""
        state.active_task_cid = ""
        state.active_task_title = ""
        state.active_task_track = ""
        state.active_task_started_at = ""
        state.active_attempt = 0
        state.active_phase = ""
        state.active_phase_started_at = ""
        state.active_phase_detail = ""
        state.active_log_path = ""
        state.active_worktree_path = ""
        state.active_branch = ""
        state.implementation_in_progress = False
        state.recommended_task_id = ""
        state.recommended_actions = []
        state.heartbeat_at = repaired_at
        state.last_progress_at = repaired_at
        state.save(self.config.state_path)
        result = {
            "repaired": True,
            "reason": "stale_active_state",
            "stuck_reason": reason,
            "repaired_at": repaired_at,
            "attempt_recovery": recovered_attempt,
            **previous,
        }
        self._record_event("blocked_progress_state_repaired", result)
        return result

    def _load_strategy(self) -> dict[str, Any]:
        defaults = {
            "generation": 0,
            "focus_tracks": DEFAULT_TRACKS,
            "blocked_tasks": [],
            "deprioritized_tasks": [],
            "last_rewrite_at": "",
            "last_rewrite_reason": "",
        }
        if not self.config.strategy_path.exists():
            write_json_atomic(self.config.strategy_path, defaults)
            return defaults
        payload = load_json_dict(self.config.strategy_path)
        if payload is None:
            logger.warning("Strategy file is missing or invalid JSON; using defaults: %s", self.config.strategy_path)
            repaired = {
                **defaults,
                "last_strategy_repair_at": utc_now(),
                "last_strategy_repair_reason": "invalid_or_unreadable_strategy_file",
            }
            write_json_atomic(self.config.strategy_path, repaired)
            self._record_event(
                "strategy_file_repaired",
                {
                    "repaired": True,
                    "reason": "invalid_or_unreadable_strategy_file",
                    "path": str(self.config.strategy_path),
                },
            )
            return repaired
        merged = {**defaults, **payload}
        merged["focus_tracks"] = (
            [str(item).strip().lower() for item in merged.get("focus_tracks", []) if str(item).strip()]
            if isinstance(merged.get("focus_tracks"), list)
            else DEFAULT_TRACKS
        )
        merged["blocked_tasks"] = (
            [str(item) for item in merged.get("blocked_tasks", []) if str(item).strip()]
            if isinstance(merged.get("blocked_tasks"), list)
            else []
        )
        merged["deprioritized_tasks"] = (
            [str(item) for item in merged.get("deprioritized_tasks", []) if str(item).strip()]
            if isinstance(merged.get("deprioritized_tasks"), list)
            else []
        )
        return merged

    def _start_daemon(self) -> subprocess.Popen[str]:
        managed_daemon_guard = self.ensure_managed_daemon_pid_file()
        if managed_daemon_guard.get("blocked", False):
            raise RuntimeError(
                str(
                    managed_daemon_guard.get("reason")
                    or "managed_daemon_ownership_unproven"
                )
            )
        command = self._build_daemon_command()
        child_env = dict(os.environ)
        pass_fds: tuple[int, ...] = ()
        if self.config.plan_bound_dispatch:
            child_env = {
                name: value
                for name, value in child_env.items()
                if name in {"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
            }
            child_env["PATH"] = "/usr/bin:/bin"
            pass_fds = (self.config.accepted_control_plane_descriptor,)
        else:
            child_env.update(
                _managed_daemon_child_environment(
                    database_program=self.config.database_program,
                )
            )
        process = subprocess.Popen(
            command,
            cwd=self.config.repo_root,
            env=child_env,
            text=True,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        try:
            self._write_managed_daemon_identity(
                pid=int(process.pid),
                command=command,
            )
        except Exception:
            direct_child_stopped = terminate_direct_child_process(
                process,
                grace_seconds=1.0,
            )
            launched_birth = None
            if not direct_child_stopped:
                try:
                    launched_birth = read_process_birth(int(process.pid))
                except OSError:
                    launched_birth = None
            if (
                not direct_child_stopped
                and launched_birth is not None
                and launched_birth.parent_pid == os.getpid()
            ):
                terminate_pid_tree(
                    int(process.pid),
                    grace_seconds=1.0,
                    freeze_first=True,
                    require_gone=True,
                    owned_process_group_id=int(process.pid),
                    expected_root_start_time_ticks=(
                        launched_birth.start_time_ticks
                    ),
                )
            raise
        write_text_atomic(self._managed_daemon_pid_path(), f"{process.pid}\n")
        return process

    def _terminate(self, process: subprocess.Popen[str] | AdoptedManagedDaemonProcess) -> None:
        cleanup = self._terminate_managed_daemon_tree(grace_seconds=15.0)
        if not cleanup.get("quiesced", False):
            self._record_event("daemon_stop_blocked", cleanup)
            raise RuntimeError(
                str(
                    cleanup.get("fence", {}).get("reason")
                    or "managed_daemon_termination_unproven"
                )
            )
        returncode = process.poll()
        if isinstance(process, AdoptedManagedDaemonProcess):
            process.returncode = 0 if returncode is None else returncode
        self._record_event(
            "daemon_stop",
            {
                "returncode": (
                    process.returncode
                    if process.returncode is not None
                    else 0
                ),
                "managed_daemon_cleanup": cleanup,
            },
        )

    def _validated_plan_bound_slice(self) -> None:
        """Re-adopt one exact active store revision before every child start."""

        if not self.config.plan_bound_dispatch:
            return
        task_ids = tuple(self.config.execution_slice_task_ids)
        task_cids = tuple(self.config.execution_slice_task_cids)
        if (
            len(task_ids) != 1
            or len(task_cids) != 1
            or len(set(task_ids)) != len(task_ids)
            or len(set(task_cids)) != len(task_cids)
        ):
            raise PlanBoundDispatchError(
                "plan-bound task ID/CID populations must be exact and unique"
            )
        if (
            int(self.config.task_shard_count) != 1
            or int(self.config.task_shard_index) != 0
            or bool(self.config.strict_task_sharding)
        ):
            raise PlanBoundDispatchError(
                "plan-bound slices disable hash sharding and strict fallback"
            )
        store_path = self.config.plan_revision_store_path
        expected = {
            "revision_cid": self.config.plan_bound_revision_cid,
            "plan_root_cid": self.config.plan_bound_plan_root_cid,
            "execution_plan_cid": self.config.plan_bound_execution_plan_cid,
            "capacity_snapshot_id": self.config.plan_bound_capacity_snapshot_id,
            "slice_manifest_cid": self.config.plan_bound_slice_manifest_cid,
            "slice_id": self.config.plan_bound_slice_id,
            "lane_id": self.config.plan_bound_lane_id,
            "source_head": self.config.plan_bound_source_head,
            "source_tree": self.config.plan_bound_source_tree,
            "task_source_revision": self.config.plan_bound_task_source_revision,
            "configuration_root": self.config.plan_bound_configuration_root,
            "accepted_tree_root": self.config.plan_bound_accepted_tree_root,
        }
        missing = [
            name
            for name, value in (("plan_revision_store_path", store_path), *expected.items())
            if not str(value or "").strip()
        ]
        if missing:
            raise PlanBoundDispatchError(
                "plan-bound dispatch is partial: " + ", ".join(missing)
            )

        (
            accepted_tree_root,
            _state_dir,
            validated_store_path,
            _scheduler_config,
            _todo_path,
        ) = _validated_plan_bound_authority_paths(
            repo_root=self.config.repo_root,
            accepted_tree_root=expected["accepted_tree_root"],
            state_dir=self.config.state_dir,
            plan_revision_store_path=store_path,
            scheduler_config_path=self.config.scheduler_config_path,
            todo_path=self.config.todo_path,
            require_live_module_root=True,
        )
        store = PlanRevisionStore(validated_store_path)
        plan_adapter = ProductionParallelPlanAdapter(store)
        with store._thread_lock:  # noqa: SLF001 - canonical store transaction
            with store._guard():  # noqa: SLF001 - canonical process guard
                binding = _load_plan_revision_store_binding_locked(
                    store,
                    execution_slice_task_ids=task_ids,
                    execution_slice_task_cids=task_cids,
                )
                from ..planning.plan_revision_contracts import PlanRevision

                revision_payload = _secure_store_cas(
                    store,
                    binding.revision_cid,
                )
                revision = PlanRevision.from_dict(revision_payload)
                if revision.to_dict() != revision_payload:
                    raise PlanBoundDispatchError(
                        "active revision changed during typed decode"
                    )
                manifest = ConfiguredBoardExecutionSlices.from_dict(
                    _secure_store_cas(store, expected["slice_manifest_cid"])
                )
                try:
                    execution_slice = plan_adapter._validate_slice_owner_locked(  # noqa: SLF001
                        revision_cid=expected["revision_cid"],
                        slice_manifest_cid=expected["slice_manifest_cid"],
                        slice_id=expected["slice_id"],
                        lane_id=expected["lane_id"],
                        reassignment_cid=(
                            self.config.plan_bound_reassignment_cid
                        ),
                    )
                except Exception as exc:
                    raise PlanBoundDispatchError(
                        "configured child revision fence does not own the canonical slice"
                    ) from exc
                birth_binding = _load_plan_bound_process_birth_chain_locked(
                    store,
                    revision_cid=expected["revision_cid"],
                    slice_id=expected["slice_id"],
                    lane_id=expected["lane_id"],
                )
        observed_binding = {
            "revision_cid": binding.revision_cid,
            "plan_root_cid": binding.plan_root_cid,
            "execution_plan_cid": binding.execution_plan_cid,
            "capacity_snapshot_id": binding.capacity_snapshot_id,
        }
        for name, observed in observed_binding.items():
            if observed != expected[name]:
                raise PlanBoundDispatchError(
                    f"plan-bound {name} fence changed: {observed!r}"
                )

        if revision.materialization_transaction_cid != expected["slice_manifest_cid"]:
            raise PlanBoundDispatchError(
                "active revision does not own the configured slice manifest"
            )
        manifest_binding = {
            "plan_root_cid": manifest.plan_root_cid,
            "capacity_snapshot_id": manifest.capacity_snapshot_id,
            "source_head": manifest.source_head,
            "source_tree": manifest.repository_tree_id,
            "task_source_revision": manifest.task_source_revision,
            "configuration_root": manifest.configuration_root,
        }
        for name, observed in manifest_binding.items():
            if observed != expected[name]:
                raise PlanBoundDispatchError(
                    f"slice manifest {name} does not match the launch fence"
                )
        if manifest.compiler_plan_id != binding.plan_id:
            raise PlanBoundDispatchError(
                "slice manifest and active compiled plan identities disagree"
            )
        if revision.roots.configuration_root != expected["configuration_root"]:
            raise PlanBoundDispatchError(
                "active revision configuration root crossed the launch fence"
            )
        if execution_slice.task_ids != task_ids or execution_slice.task_cids != task_cids:
            raise PlanBoundDispatchError(
                "configured child population differs from its CAS slice"
            )
        if birth_binding is None:
            raise PlanBoundDispatchError(
                "durable plan-bound process birth record is absent"
            )
        _birth_cid, typed_birth, _birth_chain = birth_binding
        birth_record = typed_birth.to_dict()
        birth_expected = {
            "revision_cid": expected["revision_cid"],
            "plan_root_cid": expected["plan_root_cid"],
            "execution_plan_cid": expected["execution_plan_cid"],
            "capacity_snapshot_id": expected["capacity_snapshot_id"],
            "slice_manifest_cid": expected["slice_manifest_cid"],
            "slice_id": expected["slice_id"],
            "lane_id": expected["lane_id"],
            "configuration_root": expected["configuration_root"],
            "accepted_tree_root": str(accepted_tree_root),
        }
        if (
            any(
                birth_record.get(name) != value
                for name, value in birth_expected.items()
            )
            or birth_record.get("task_ids") != list(task_ids)
            or birth_record.get("task_cids") != list(task_cids)
        ):
            raise PlanBoundDispatchError(
                "durable plan-bound process birth differs from the slice"
            )
        try:
            from ..control.lifecycle_orchestrator import (
                LifecycleProfile,
                LinuxProcessAdapter,
                ProcessIdentity,
            )

            birth_profile = LifecycleProfile.from_dict(birth_record["profile"])
            birth_identity = ProcessIdentity.from_dict(
                birth_record["process_birth"]
            )
            if (
                birth_profile.to_dict() != birth_record["profile"]
                or birth_identity.to_dict() != birth_record["process_birth"]
            ):
                raise ValueError("birth lifecycle records normalized")
            current_identity = LinuxProcessAdapter()._identity(  # noqa: SLF001
                os.getpid(),
                birth_profile,
            )
        except Exception as exc:
            raise PlanBoundDispatchError(
                "cannot revalidate durable plan-bound process birth"
            ) from exc
        stable_birth_fields = (
            "pid", "start_time_ticks", "process_group_id", "session_id",
            "boot_id", "run_id", "profile_id", "target_id",
            "repository_root", "state_root", "run_root", "fencing_epoch",
            "configuration_root",
        )
        if any(
            getattr(current_identity, name) != getattr(birth_identity, name)
            for name in stable_birth_fields
        ):
            raise PlanBoundDispatchError(
                "current supervisor is not the persisted gated process birth"
            )

        try:
            from ..runtime.configured_board_scheduler import (
                _git_identity as configured_board_git_identity,
            )
            from ..runtime.configured_board_scheduler import (
                _tracked_head_snapshot,
                load_configured_board,
            )

            observed_head, observed_tree = configured_board_git_identity(
                accepted_tree_root
            )
            current_board = load_configured_board(
                self.config.scheduler_config_path,
                repo_root=accepted_tree_root,
            )
            config_bytes, _config_revision = _tracked_head_snapshot(
                repo_root=accepted_tree_root,
                path=self.config.scheduler_config_path,
                source_head=expected["source_head"],
            )
            _board_bytes, task_source_revision = _tracked_head_snapshot(
                repo_root=accepted_tree_root,
                path=self.config.todo_path,
                source_head=expected["source_head"],
            )
            configured_state_root = _plan_bound_contained_path(
                accepted_tree_root,
                current_board.path(current_board.runtime_paths["state"]),
                field_name="configured runtime state root",
                require_directory=True,
            )
        except Exception as exc:
            raise PlanBoundDispatchError(
                "cannot re-observe tracked config/task authority"
            ) from exc
        if (
            observed_head != expected["source_head"]
            or observed_tree != expected["source_tree"]
        ):
            raise PlanBoundDispatchError(
                "repository identity crossed the plan fence"
            )
        if (
            self.config.state_dir.parent != configured_state_root
            or validated_store_path
            != configured_state_root / "plan-revision-store"
        ):
            raise PlanBoundDispatchError(
                "plan-bound paths differ from configured runtime state authority"
            )
        if current_board.configuration_root != expected["configuration_root"]:
            raise PlanBoundDispatchError(
                "scheduler configuration crossed the plan fence"
            )
        if content_identity(
            {"bytes_sha256": sha256(config_bytes).hexdigest()}
        ) != expected["configuration_root"]:
            raise PlanBoundDispatchError(
                "tracked scheduler bytes differ from the configuration root"
            )
        if task_source_revision != expected["task_source_revision"]:
            raise PlanBoundDispatchError("task source crossed the plan fence")

    def _build_daemon_command(self) -> list[str]:
        self._validated_plan_bound_slice()
        daemon_script_path = self.config.daemon_script_path
        if self.config.plan_bound_dispatch:
            if daemon_script_path is not None:
                raise PlanBoundDispatchError(
                    "plan-bound dispatch forbids an uninspectable daemon script"
                )
            command = [
                PLAN_BOUND_DAEMON_CHILD_MARKER,
                "--daemon-entrypoint",
                PLAN_BOUND_DAEMON_ENTRYPOINT,
                "--scheduler-config",
                str(self.config.scheduler_config_path or ""),
                "--plan-revision-store-path",
                str(self.config.plan_revision_store_path),
                "--plan-bound-revision-cid",
                self.config.plan_bound_revision_cid,
                "--plan-bound-plan-root-cid",
                self.config.plan_bound_plan_root_cid,
                "--plan-bound-execution-plan-cid",
                self.config.plan_bound_execution_plan_cid,
                "--plan-bound-capacity-snapshot-id",
                self.config.plan_bound_capacity_snapshot_id,
                "--plan-bound-slice-manifest-cid",
                self.config.plan_bound_slice_manifest_cid,
                "--plan-bound-slice-id",
                self.config.plan_bound_slice_id,
                "--plan-bound-lane-id",
                self.config.plan_bound_lane_id,
                "--plan-bound-source-head",
                self.config.plan_bound_source_head,
                "--plan-bound-source-tree",
                self.config.plan_bound_source_tree,
                "--plan-bound-task-source-revision",
                self.config.plan_bound_task_source_revision,
                "--plan-bound-configuration-root",
                self.config.plan_bound_configuration_root,
                "--plan-bound-accepted-tree-root",
                str(self.config.plan_bound_accepted_tree_root or ""),
                "--accepted-control-plane-pin-json",
                json.dumps(
                    self.config.accepted_control_plane_pin.as_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "--accepted-control-plane-fd",
                str(self.config.accepted_control_plane_descriptor),
            ]
            if self.config.plan_bound_reassignment_cid:
                command.extend(
                    [
                        "--plan-bound-reassignment-cid",
                        self.config.plan_bound_reassignment_cid,
                    ]
                )
            for task_id in self.config.execution_slice_task_ids:
                command.extend(["--plan-bound-task-id", str(task_id)])
            for task_cid in self.config.execution_slice_task_cids:
                command.extend(["--plan-bound-task-cid", str(task_cid)])
            command.append("--")
        elif daemon_script_path is None:
            # Safe-path mode prevents a stale nested checkout in the working
            # directory from shadowing the supervisor's configured package.
            command = [
                sys.executable,
                "-P",
                "-m",
                "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
            ]
        else:
            command = [sys.executable, str(daemon_script_path)]
        command.extend(
            [
                "--interval",
                str(self.config.daemon_interval),
                "--todo-path",
                str(self.config.todo_path),
                "--state-dir",
                str(self.config.state_dir),
                "--task-prefix",
                self.config.task_prefix,
                "--state-prefix",
                self.config.state_prefix,
                "--max-task-attempts",
                str(max(0, int(self.config.max_task_attempts))),
            ]
        )
        if self.config.database_program is not None:
            program = self.config.database_program
            program.assert_quack_not_demoted(
                candidate_mode=program.authority_mode,
            )
            command.extend(program.daemon_cli_args())
        if self.config.validation_max_workers is not None:
            command.extend(
                [
                    "--validation-max-workers",
                    str(max(1, int(self.config.validation_max_workers))),
                ]
            )
        for path in self.config.generated_dirty_repair_paths:
            command.extend(["--generated-status-path", str(path)])
        for relative in self.config.implementation_protected_paths:
            command.extend(["--implementation-protected-path", relative])
        if self.config.merge_target_branch:
            command.extend(
                ["--merge-target-branch", self.config.merge_target_branch]
            )
        if self.config.merge_queue_dir is not None:
            command.extend(
                ["--merge-queue-dir", str(self.config.merge_queue_dir)]
            )
        if self.config.implement:
            command.append("--implement")
            command.extend(["--implementation-timeout", str(self.config.implementation_timeout)])
            if self.config.implementation_command:
                command.extend(["--implementation-command", self.config.implementation_command])
            if self.config.llm_merge_resolver_command:
                command.extend(["--llm-merge-resolver-command", self.config.llm_merge_resolver_command])
            if self.config.llm_merge_resolver_timeout_seconds is not None:
                command.extend(
                    [
                        "--llm-merge-resolver-timeout-seconds",
                        str(self.config.llm_merge_resolver_timeout_seconds),
                    ]
                )
            if not self.config.use_ephemeral_worktree:
                command.append("--no-ephemeral-worktree")
            if self.config.worktree_root is not None:
                command.extend(["--worktree-root", str(self.config.worktree_root)])
            for relative in self.config.worktree_submodule_paths:
                command.extend(["--worktree-submodule-path", relative])
            if self.config.objective_path is not None:
                command.extend(["--objective-path", str(self.config.objective_path)])
            if self.config.objective_bundle_dir is not None:
                command.extend(["--objective-bundle-dir", str(self.config.objective_bundle_dir)])
        if self.config.objective_refill_enabled:
            command.extend(
                [
                    "--objective-scan-min-open-tasks",
                    str(self.config.objective_scan_min_open_tasks),
                    "--objective-scan-max-findings",
                    str(self.config.objective_scan_max_findings),
                    "--objective-scan-cooldown-seconds",
                    str(self.config.objective_scan_cooldown_seconds),
                    "--objective-surplus-findings-per-goal",
                    str(self.config.objective_surplus_findings_per_goal),
                    "--objective-surplus-min-terms-per-todo",
                    str(self.config.objective_surplus_min_terms_per_todo),
                ]
            )
        if self.config.codebase_refill_enabled:
            command.extend(
                [
                    "--codebase-scan-min-open-tasks",
                    str(self.config.codebase_scan_min_open_tasks),
                    "--codebase-scan-max-findings",
                    str(self.config.codebase_scan_max_findings),
                    "--codebase-scan-cooldown-seconds",
                    str(self.config.codebase_scan_cooldown_seconds),
                ]
            )
        if self.config.merge_reconciliation_max_merges is not None:
            command.extend(
                [
                    "--merge-reconciliation-max-merges",
                    str(self.config.merge_reconciliation_max_merges),
                ]
            )
        if self.config.daemon_merged_worktree_cleanup_max is not None:
            command.extend(
                [
                    "--merged-worktree-cleanup-max",
                    str(self.config.daemon_merged_worktree_cleanup_max),
                ]
            )
        command.extend(
            [
                "--task-shard-count",
                str(1 if self.config.plan_bound_dispatch else max(1, int(self.config.task_shard_count))),
                "--task-shard-index",
                str(0 if self.config.plan_bound_dispatch else int(self.config.task_shard_index)),
            ]
        )
        if self.config.strict_task_sharding and not self.config.plan_bound_dispatch:
            command.append("--strict-task-sharding")
        for path in self.config.external_reservation_manifest_paths:
            command.extend(["--external-reservation-manifest-path", str(path)])
        for task_id in self.config.assumed_completed_task_ids:
            command.extend(["--assume-completed-task-id", str(task_id)])
        for task_id in self.config.manual_completion_authority_task_ids:
            command.extend(
                ["--manual-completion-authority-task-id", str(task_id)]
            )
        for task_id in self.config.manual_completion_authority_required_task_ids:
            command.extend(
                ["--manual-completion-authority-required-task-id", str(task_id)]
            )
        if self.config.manual_completion_authority_epoch_id:
            command.extend(
                [
                    "--manual-completion-authority-epoch-id",
                    self.config.manual_completion_authority_epoch_id,
                ]
            )
        if self.config.manual_completion_authority_revalidation_only:
            command.append(
                "--manual-completion-authority-revalidation-only"
            )
        if not self.config.plan_bound_dispatch:
            for task_id in self.config.execution_slice_task_ids:
                command.extend(["--execution-slice-task-id", str(task_id)])
        for task_cid in self.config.execution_slice_task_cids:
            command.extend(["--execution-slice-task-cid", str(task_cid)])
        if self.config.plan_bound_dispatch:
            command.append("--once")
            from ..runtime.multi_supervisor_runner import (
                build_sealed_control_plane_module_command,
            )

            if self.config.accepted_control_plane_pin is None:
                raise PlanBoundDispatchError(
                    "plan-bound daemon launch lacks its sealed control plane"
                )
            command = build_sealed_control_plane_module_command(
                python_executable=sys.executable,
                pin=self.config.accepted_control_plane_pin,
                descriptor=self.config.accepted_control_plane_descriptor,
                module_name=(
                    "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                    "implementation_supervisor"
                ),
                argv=command,
            )
        return command

    def provider_subprocess_environment(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a provider environment without state-authority credentials."""

        return provider_environment_without_state_credentials(
            environment,
            database_program=self.config.database_program,
        )

    def _managed_daemon_pid_path(self) -> Path:
        return self.config.state_dir / f"{self.config.state_prefix}_managed_daemon.pid"

    def _managed_daemon_identity_path(self) -> Path:
        return supervised_child_identity_path(
            self._managed_daemon_pid_path()
        )

    def _managed_daemon_owner_scope(self) -> dict[str, str]:
        daemon_script_path = self.config.daemon_script_path
        daemon_entrypoint = (
            str(Path(daemon_script_path).resolve(strict=False))
            if daemon_script_path is not None
            else (
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_daemon"
            )
        )
        return {
            "repo_root": str(self.config.repo_root.resolve(strict=False)),
            "state_dir": str(self.config.state_dir.resolve(strict=False)),
            "state_prefix": str(self.config.state_prefix),
            "todo_path": str(self.config.todo_path.resolve(strict=False)),
            "daemon_entrypoint": daemon_entrypoint,
        }

    def _managed_daemon_command_belongs_to_scope(
        self,
        command: Sequence[str],
    ) -> bool:
        tokens = tuple(str(part) for part in command)
        daemon_script_path = self.config.daemon_script_path
        daemon_token = (
            str(daemon_script_path)
            if daemon_script_path is not None
            else (
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_daemon"
            )
        )
        if daemon_token not in tokens:
            return False

        def exact_option(option: str, expected: str) -> bool:
            values = [
                tokens[index + 1]
                for index, token in enumerate(tokens[:-1])
                if token == option
            ]
            return values == [expected]

        return (
            exact_option("--state-dir", str(self.config.state_dir))
            and exact_option("--state-prefix", self.config.state_prefix)
            and exact_option("--todo-path", str(self.config.todo_path))
        )

    def _remove_managed_daemon_identity_markers(
        self,
        *,
        expected_pid: int | None = None,
    ) -> bool:
        pid_path = self._managed_daemon_pid_path()
        identity_path = self._managed_daemon_identity_path()
        if expected_pid is not None:
            try:
                recorded_pid = int(
                    pid_path.read_text(encoding="utf-8").strip()
                )
            except (OSError, ValueError):
                return False
            if recorded_pid != int(expected_pid):
                return False
        for path in (pid_path, identity_path):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
            except OSError:
                return False
        return True

    def _quarantine_managed_daemon_identity_markers(
        self,
        *,
        reason: str,
    ) -> dict[str, str]:
        quarantined: dict[str, str] = {}
        for label, path in (
            ("pid", self._managed_daemon_pid_path()),
            ("identity", self._managed_daemon_identity_path()),
        ):
            if not path.exists() and not path.is_symlink():
                continue
            try:
                backup = unique_backup_path(path, reason)
                path.rename(backup)
            except OSError:
                continue
            quarantined[label] = str(backup)
        return quarantined
    def _write_managed_daemon_identity(
        self,
        *,
        pid: int,
        command: Sequence[str],
        require_direct_child: bool = True,
    ) -> None:
        write_supervised_child_identity(
            self._managed_daemon_identity_path(),
            pid=int(pid),
            command=command,
            owner_scope=self._managed_daemon_owner_scope(),
            require_direct_child=require_direct_child,
        )

    def _fence_recorded_managed_daemon(
        self,
        *,
        pid: int,
        grace_seconds: float = 1.0,
    ) -> dict[str, Any]:
        identity_path = self._managed_daemon_identity_path()
        identity = load_supervised_child_identity(identity_path)
        if identity is None:
            return {
                "fenced": False,
                "reason": "managed_daemon_ownership_unproven",
            }
        if (
            identity.process_birth.pid != int(pid)
            or dict(identity.owner_scope)
            != self._managed_daemon_owner_scope()
            or not self._managed_daemon_command_belongs_to_scope(
                identity.command
            )
        ):
            return {
                "fenced": False,
                "reason": "managed_daemon_ownership_scope_mismatch",
            }
        liveness = supervised_child_identity_liveness(identity)
        if liveness is OwnerLiveness.DEAD:
            return {
                "fenced": False,
                "pid_reused": bool(process_is_running(pid)),
                "reason": "managed_daemon_recorded_process_dead",
            }
        if liveness is not OwnerLiveness.ALIVE:
            return {
                "fenced": False,
                "reason": "managed_daemon_ownership_liveness_unknown",
            }
        observed_argv = read_process_command_argv(pid)
        if observed_argv is None or observed_argv != identity.command:
            return {
                "fenced": False,
                "reason": "managed_daemon_command_identity_mismatch",
            }
        # Re-read birth identity immediately before entering the existing
        # freeze/rescan/kill fence. A reused numeric PID is never signalled.
        if supervised_child_identity_liveness(identity) is not OwnerLiveness.ALIVE:
            return {
                "fenced": False,
                "reason": "managed_daemon_process_birth_changed",
            }
        fenced = terminate_pid_tree(
            int(pid),
            grace_seconds=max(0.0, float(grace_seconds)),
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=int(pid),
            expected_root_start_time_ticks=(
                identity.process_birth.start_time_ticks
            ),
        )
        gone = (
            supervised_child_identity_liveness(identity)
            is OwnerLiveness.DEAD
        )
        return {
            "fenced": bool(fenced and gone),
            "reason": (
                "managed_daemon_owned_process_fenced"
                if fenced and gone
                else "managed_daemon_owned_process_fence_failed"
            ),
        }

    def _terminate_managed_daemon_tree(self, *, grace_seconds: float = 1.0) -> dict[str, Any]:
        """Stop the daemon this supervisor owns, including late-spawned workers."""

        pid_path = self._managed_daemon_pid_path()
        repair = self.ensure_managed_daemon_pid_file()
        if repair.get("blocked", False):
            return {
                "pid": self._read_managed_daemon_pid(),
                "terminated": False,
                "quiesced": False,
                "remaining_pid": self._find_matching_managed_daemon_pid(),
                "pid_path": str(pid_path),
                "identity_path": str(
                    self._managed_daemon_identity_path()
                ),
                "fence": {
                    "fenced": False,
                    "reason": str(
                        repair.get("reason")
                        or "managed_daemon_ownership_unproven"
                    ),
                },
                "repair": repair,
            }
        pid = self._read_managed_daemon_pid()
        if pid is None:
            remaining_pid = self._find_matching_managed_daemon_pid()
            return {
                "pid": None,
                "terminated": False,
                "quiesced": remaining_pid is None,
                "remaining_pid": remaining_pid,
                "pid_path": str(pid_path),
                "identity_path": str(
                    self._managed_daemon_identity_path()
                ),
                "fence": {
                    "fenced": False,
                    "reason": (
                        "managed_daemon_not_found"
                        if remaining_pid is None
                        else "managed_daemon_ownership_unproven"
                    ),
                },
                "repair": repair,
            }

        fence = (
            self._fence_recorded_managed_daemon(
                pid=pid,
                grace_seconds=grace_seconds,
            )
            if pid is not None
            else {
                "fenced": False,
                "reason": "managed_daemon_not_found",
            }
        )
        terminated = bool(fence.get("fenced", False))
        remaining_pid = self._find_matching_managed_daemon_pid()
        if terminated:
            self._remove_managed_daemon_identity_markers(
                expected_pid=pid,
            )
        return {
            "pid": pid,
            "terminated": terminated,
            "quiesced": (pid is None or terminated) and remaining_pid is None,
            "remaining_pid": remaining_pid,
            "pid_path": str(pid_path),
            "identity_path": str(self._managed_daemon_identity_path()),
            "fence": fence,
        }

    def _read_managed_daemon_pid(self) -> int | None:
        try:
            raw_pid = self._managed_daemon_pid_path().read_text(encoding="utf-8").strip()
            return int(raw_pid)
        except (OSError, ValueError):
            return None

    def _find_matching_managed_daemon_pid(self, *, exclude_pids: set[int] | None = None) -> int | None:
        excluded = set(exclude_pids or set())
        excluded.add(os.getpid())
        for pid, command_line in self._list_process_details():
            if pid in excluded:
                continue
            if not process_is_running(pid):
                continue
            if self._managed_daemon_matches_command_line(command_line):
                return int(pid)
        return None

    def ensure_managed_daemon_pid_file(self) -> dict[str, Any]:
        """Remove stale or malformed managed-daemon PID state before adoption."""

        pid_path = self._managed_daemon_pid_path()
        identity_path = self._managed_daemon_identity_path()
        if not pid_path.exists():
            identity_exists = identity_path.exists() or identity_path.is_symlink()
            if identity_exists:
                identity = load_supervised_child_identity(identity_path)
                if identity is None:
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "orphaned_managed_daemon_identity_invalid",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                    }
                if (
                    dict(identity.owner_scope)
                    != self._managed_daemon_owner_scope()
                    or not self._managed_daemon_command_belongs_to_scope(
                        identity.command
                    )
                ):
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "orphaned_managed_daemon_identity_scope_mismatch",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": identity.process_birth.pid,
                    }
                identity_liveness = supervised_child_identity_liveness(
                    identity
                )
                if identity_liveness is OwnerLiveness.UNKNOWN:
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "orphaned_managed_daemon_identity_liveness_unknown",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": identity.process_birth.pid,
                    }
                if identity_liveness is OwnerLiveness.DEAD:
                    matching_pid = self._find_matching_managed_daemon_pid()
                    if matching_pid is not None:
                        return {
                            "repaired": False,
                            "blocked": True,
                            "reason": "matching_managed_daemon_ownership_unproven",
                            "path": str(pid_path),
                            "identity_path": str(identity_path),
                            "pid": int(matching_pid),
                        }
                    quarantined = (
                        self._quarantine_managed_daemon_identity_markers(
                            reason="stale-orphaned-managed-daemon"
                        )
                    )
                    result = {
                        "repaired": True,
                        "reason": "stale_orphaned_managed_daemon_identity",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": identity.process_birth.pid,
                        "quarantined": quarantined,
                    }
                    self._record_event(
                        "managed_daemon_pid_file_repaired",
                        result,
                    )
                    return result
                observed_argv = read_process_command_argv(
                    identity.process_birth.pid
                )
                if observed_argv != identity.command:
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "orphaned_managed_daemon_command_identity_mismatch",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": identity.process_birth.pid,
                    }
                write_text_atomic(
                    pid_path,
                    f"{identity.process_birth.pid}\n",
                )
                recovery = self.ensure_managed_daemon_pid_file()
                recovery["repaired"] = True
                recovery["orphaned_identity_recovered"] = True
                if recovery.get("reason") == "active":
                    recovery["reason"] = (
                        "orphaned_live_managed_daemon_pid_reconstructed"
                    )
                self._record_event(
                    "managed_daemon_pid_file_repaired",
                    recovery,
                )
                return recovery
            return {"repaired": False, "reason": "missing", "path": str(pid_path)}
        if pid_path.is_dir():
            backup_path = unique_backup_path(pid_path, "directory-backup")
            pid_path.rename(backup_path)
            result = {
                "repaired": True,
                "reason": "managed_pid_path_was_directory",
                "path": str(pid_path),
                "backup_path": str(backup_path),
            }
            self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        try:
            raw_pid = pid_path.read_text(encoding="utf-8").strip()
            pid = int(raw_pid)
        except (OSError, UnicodeDecodeError, ValueError):
            try:
                backup_path = unique_backup_path(pid_path, "invalid-pid")
                pid_path.rename(backup_path)
                result = {
                    "repaired": True,
                    "reason": "invalid_managed_pid_file",
                    "path": str(pid_path),
                    "backup_path": str(backup_path),
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "invalid_managed_pid_file_unrepairable",
                    "path": str(pid_path),
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        if pid <= 0:
            try:
                backup_path = unique_backup_path(pid_path, "invalid-pid")
                pid_path.rename(backup_path)
                result = {
                    "repaired": True,
                    "reason": "invalid_managed_pid",
                    "path": str(pid_path),
                    "pid": pid,
                    "backup_path": str(backup_path),
                }
            except OSError as exc:
                result = {
                    "repaired": False,
                    "reason": "invalid_managed_pid_unrepairable",
                    "path": str(pid_path),
                    "pid": pid,
                    "error": str(exc),
                }
            if result.get("repaired"):
                self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        identity_exists = identity_path.exists() or identity_path.is_symlink()
        identity = load_supervised_child_identity(identity_path)
        recorded_pid_running = process_is_running(pid)
        if identity_exists:
            if identity is None:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "managed_daemon_ownership_unproven",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": pid,
                }
            identity_liveness = supervised_child_identity_liveness(identity)
            if identity_liveness is OwnerLiveness.UNKNOWN:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "managed_daemon_ownership_liveness_unknown",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": pid,
                }
            if identity_liveness is OwnerLiveness.ALIVE:
                observed_identity_argv = read_process_command_argv(
                    identity.process_birth.pid
                )
                if (
                    dict(identity.owner_scope)
                    != self._managed_daemon_owner_scope()
                    or not self._managed_daemon_command_belongs_to_scope(
                        identity.command
                    )
                    or observed_identity_argv != identity.command
                ):
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "managed_daemon_ownership_scope_mismatch",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": pid,
                        "identity_pid": identity.process_birth.pid,
                    }
                if identity.process_birth.pid != pid:
                    write_text_atomic(
                        pid_path,
                        f"{identity.process_birth.pid}\n",
                    )
                    recovery = self.ensure_managed_daemon_pid_file()
                    recovery["repaired"] = True
                    recovery["recorded_pid_reconciled"] = pid
                    if recovery.get("reason") == "active":
                        recovery["reason"] = (
                            "managed_daemon_pid_reconciled_from_live_identity"
                        )
                    self._record_event(
                        "managed_daemon_pid_file_repaired",
                        recovery,
                    )
                    return recovery
                if not recorded_pid_running:
                    return {
                        "repaired": False,
                        "blocked": True,
                        "reason": "managed_daemon_process_liveness_inconsistent",
                        "path": str(pid_path),
                        "identity_path": str(identity_path),
                        "pid": pid,
                    }
        if not recorded_pid_running:
            matching_pid = self._find_matching_managed_daemon_pid()
            if matching_pid is not None:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "matching_managed_daemon_ownership_unproven",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": int(matching_pid),
                }
            quarantined = self._quarantine_managed_daemon_identity_markers(
                reason="stale-managed-daemon"
            )
            result = {
                "repaired": True,
                "reason": "stale_managed_pid",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
                "quarantined": quarantined,
            }
            self._record_event("managed_daemon_pid_file_repaired", result)
            return result

        command_line = process_command_line(pid)
        desired_command = tuple(self._build_daemon_command())
        desired_command_matches = self._managed_daemon_matches_command_line(
            command_line
        )
        if not identity_exists and desired_command_matches:
            # One-time migration for a live legacy daemon is safe only while
            # it already has the exact desired supervisor configuration.
            observed_argv = read_process_command_argv(pid)
            if observed_argv != desired_command:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "managed_daemon_ownership_unproven",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": pid,
                }
            try:
                self._write_managed_daemon_identity(
                    pid=pid,
                    command=desired_command,
                    require_direct_child=False,
                )
            except Exception as exc:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "managed_daemon_identity_migration_failed",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": pid,
                    "error": str(exc),
                }
            result = {
                "repaired": True,
                "reason": "active_legacy_managed_daemon_identity_migrated",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
            }
            self._record_event("managed_daemon_pid_file_repaired", result)
            return result

        if identity is None:
            return {
                "repaired": False,
                "blocked": True,
                "reason": "managed_daemon_ownership_unproven",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
            }

        identity_liveness = supervised_child_identity_liveness(identity)
        if identity.process_birth.pid != pid or identity_liveness is OwnerLiveness.DEAD:
            # The numeric PID now belongs to another process. Never signal it;
            # If any process already has the desired daemon command, its
            # ownership is not proved by this stale identity.  Fail closed so
            # startup cannot launch a duplicate matching daemon.
            matching_pid = self._find_matching_managed_daemon_pid()
            if matching_pid is not None:
                return {
                    "repaired": False,
                    "blocked": True,
                    "reason": "matching_managed_daemon_ownership_unproven",
                    "path": str(pid_path),
                    "identity_path": str(identity_path),
                    "pid": int(matching_pid),
                }
            quarantined = self._quarantine_managed_daemon_identity_markers(
                reason="pid-reuse"
            )
            result = {
                "repaired": True,
                "reason": "managed_daemon_pid_reused",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
                "quarantined": quarantined,
            }
            self._record_event("managed_daemon_pid_file_repaired", result)
            return result
        if identity_liveness is not OwnerLiveness.ALIVE:
            return {
                "repaired": False,
                "blocked": True,
                "reason": "managed_daemon_ownership_liveness_unknown",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
            }
        observed_argv = read_process_command_argv(pid)
        if (
            dict(identity.owner_scope) != self._managed_daemon_owner_scope()
            or not self._managed_daemon_command_belongs_to_scope(
                identity.command
            )
            or observed_argv != identity.command
        ):
            return {
                "repaired": False,
                "blocked": True,
                "reason": "managed_daemon_ownership_scope_mismatch",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
            }

        if identity.command == desired_command:
            return {
                "repaired": False,
                "reason": "active",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
            }

        fence = self._fence_recorded_managed_daemon(pid=pid)
        if not fence.get("fenced", False):
            return {
                "repaired": False,
                "blocked": True,
                "reason": "managed_daemon_obsolete_fence_failed",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
                "fence": fence,
            }
        if not self._remove_managed_daemon_identity_markers(
            expected_pid=pid
        ):
            return {
                "repaired": False,
                "blocked": True,
                "reason": "managed_daemon_obsolete_marker_cleanup_failed",
                "path": str(pid_path),
                "identity_path": str(identity_path),
                "pid": pid,
                "fence": fence,
            }
        result = {
            "repaired": True,
            "reason": "obsolete_owned_managed_daemon_fenced",
            "path": str(pid_path),
            "identity_path": str(identity_path),
            "pid": pid,
            "fence": fence,
        }
        self._record_event("managed_daemon_pid_file_repaired", result)
        return result

    def _adopt_existing_daemon(self) -> AdoptedManagedDaemonProcess | None:
        pid_path = self._managed_daemon_pid_path()
        repair = self.ensure_managed_daemon_pid_file()
        if repair.get("blocked", False):
            raise RuntimeError(
                str(
                    repair.get("reason")
                    or "managed_daemon_ownership_unproven"
                )
            )
        if (
            repair.get("repaired")
            and repair.get("reason")
            != "active_legacy_managed_daemon_identity_migrated"
        ) or not pid_path.exists() or pid_path.is_dir():
            return None
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            try:
                pid_path.unlink()
            except OSError:
                pass
            return None
        if not process_is_running(pid):
            try:
                pid_path.unlink()
            except OSError:
                pass
            return None
        command_line = process_command_line(pid)
        if not self._managed_daemon_matches_command_line(command_line):
            raise RuntimeError("managed_daemon_command_identity_mismatch")
        identity = load_supervised_child_identity(
            self._managed_daemon_identity_path()
        )
        if (
            identity is None
            or identity.process_birth.pid != pid
            or identity.command != tuple(self._build_daemon_command())
            or supervised_child_identity_liveness(identity)
            is not OwnerLiveness.ALIVE
        ):
            raise RuntimeError("managed_daemon_ownership_identity_mismatch")
        return AdoptedManagedDaemonProcess(pid)

    def _managed_daemon_matches_command_line(self, command_line: str) -> bool:
        daemon_script_path = self.config.daemon_script_path
        daemon_fragment = (
            Path(daemon_script_path).name
            if daemon_script_path is not None
            else "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        )
        required_fragments = [
            daemon_fragment,
            "--state-dir",
            str(self.config.state_dir),
            "--state-prefix",
            self.config.state_prefix,
            "--todo-path",
            str(self.config.todo_path),
        ]
        if not all(fragment in command_line for fragment in required_fragments):
            return False
        tokens = command_line.split()
        has_implement_flag = "--implement" in tokens
        if self.config.implement != has_implement_flag:
            return False
        has_strict_task_sharding_flag = "--strict-task-sharding" in tokens
        if (
            bool(
                self.config.strict_task_sharding
                and not self.config.plan_bound_dispatch
            )
            != has_strict_task_sharding_flag
        ):
            return False

        def option_values(option: str) -> set[str]:
            return {
                tokens[index + 1]
                for index, token in enumerate(tokens[:-1])
                if token == option
            }

        if option_values("--task-shard-count") != {
            str(
                1
                if self.config.plan_bound_dispatch
                else max(1, int(self.config.task_shard_count))
            )
        }:
            return False
        if option_values("--task-shard-index") != {
            str(0 if self.config.plan_bound_dispatch else int(self.config.task_shard_index))
        }:
            return False
        expected_slice_ids = (
            set()
            if self.config.plan_bound_dispatch
            else set(self.config.execution_slice_task_ids)
        )
        if option_values("--execution-slice-task-id") != expected_slice_ids:
            return False
        if option_values("--manual-completion-authority-task-id") != set(
            self.config.manual_completion_authority_task_ids
        ):
            return False
        if option_values(
            "--manual-completion-authority-required-task-id"
        ) != set(self.config.manual_completion_authority_required_task_ids):
            return False
        expected_authority_epoch_ids = (
            {self.config.manual_completion_authority_epoch_id}
            if self.config.manual_completion_authority_epoch_id
            else set()
        )
        if option_values(
            "--manual-completion-authority-epoch-id"
        ) != expected_authority_epoch_ids:
            return False
        if (
            "--manual-completion-authority-revalidation-only" in tokens
        ) != bool(
            self.config.manual_completion_authority_revalidation_only
        ):
            return False
        if option_values("--execution-slice-task-cid") != set(
            self.config.execution_slice_task_cids
        ):
            return False
        if ("--once" in tokens) != bool(self.config.plan_bound_dispatch):
            return False
        expected_merge_targets = (
            {self.config.merge_target_branch}
            if self.config.merge_target_branch
            else set()
        )
        if option_values("--merge-target-branch") != expected_merge_targets:
            return False
        expected_merge_queue_dirs = (
            {str(self.config.merge_queue_dir)}
            if self.config.merge_queue_dir is not None
            else set()
        )
        if option_values("--merge-queue-dir") != expected_merge_queue_dirs:
            return False
        return True

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl_event(self.config.events_path, event_type, payload)

    @staticmethod
    def _age_seconds(timestamp: str, now_ts: float) -> float:
        if not timestamp:
            return float("inf")
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, now_ts - parsed.timestamp())
        except ValueError:
            return float("inf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    expanded_argv, scheduler_config_path = (
        expand_supervisor_scheduler_config_args(
            sys.argv[1:] if argv is None else argv,
            repo_root=REPO_ROOT,
        )
    )
    parser = argparse.ArgumentParser(description="Supervise the portal implementation backlog daemon")
    parser.add_argument(
        "--scheduler-config",
        type=Path,
        default=scheduler_config_path,
        help=(
            "Sealed scheduler_config@1 JSON profile. Safe profile values become "
            "defaults; explicit scalar CLI options take precedence. The profile "
            "never enables implementation, refill, Doctor mutation, or rollout."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Run one supervisor check and exit")
    parser.add_argument(
        "--todo-path",
        type=Path,
        default=Path("docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md"),
        help="Machine-readable markdown backlog",
    )
    parser.add_argument(
        "--task-source-kind",
        choices=("", "legacy-markdown", "markdown", "duckdb"),
        default="",
        help="Explicit task-source storage contract forwarded to the managed daemon.",
    )
    parser.add_argument(
        "--authority-mode",
        choices=("", "legacy_markdown", "embedded", "embedded_exclusive", "quack"),
        default="",
        help="Explicit state-authority mode forwarded to the managed daemon.",
    )
    parser.add_argument(
        "--endpoint-secret-handle",
        default="",
        help="Opaque state endpoint secret handle; raw credentials are forbidden.",
    )
    parser.add_argument("--state-store-id", default="")
    parser.add_argument("--state-store-generation", default="")
    parser.add_argument("--state-schema-revision", default="")
    parser.add_argument(
        "--state-failover-policy",
        choices=("fail_closed", "require_explicit_operator"),
        default="fail_closed",
    )
    parser.add_argument("--event-store-path", default="")
    parser.add_argument("--runtime-registry-path", default="")
    parser.add_argument("--export-profile", default="")
    parser.add_argument(
        "--explicit-legacy-task-source",
        action="store_true",
        help="Acknowledge explicit legacy-Markdown authority.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("data/portal_implementation/state"),
        help="Portal daemon state directory",
    )
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--check-interval", type=float, default=60.0)
    parser.add_argument(
        "--watchdog-startup-grace-seconds",
        type=float,
        default=None,
        help=(
            "Delay stale-heartbeat enforcement while a new daemon performs startup maintenance. "
            "Defaults to at least 300 seconds; set explicitly to override."
        ),
    )
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        default=0,
        help=(
            "Maximum implementation attempts per canonical task identity. "
            "Zero disables the limit."
        ),
    )
    parser.add_argument("--daemon-interval", type=float, default=300.0)
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
    implement_group = parser.add_mutually_exclusive_group()
    implement_group.add_argument(
        "--implement",
        dest="implement",
        action="store_true",
        help="Allow the managed daemon to invoke the implementation agent",
    )
    implement_group.add_argument(
        "--no-implement",
        dest="implement",
        action="store_false",
        help="Only supervise backlog state; do not let the managed daemon invoke the implementation agent",
    )
    parser.set_defaults(implement=False)
    parser.add_argument(
        "--implementation-command",
        default="",
        help=(
            "Command used by the daemon for implementation. By default, "
            "automatic routing selects authenticated Grok 4.5. Only a "
            "typed durable Grok hard-quota latch authorizes a later "
            "gpt-5.6-terra Codex attempt with medium reasoning; other "
            "Grok failures remain fail closed."
        ),
    )
    parser.add_argument(
        "--implementation-protected-path",
        action="append",
        default=[],
        help=(
            "Exact repo-relative file that managed implementation agents must treat as "
            "read-only. May be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--manual-completion-authority-task-id",
        action="append",
        default=[],
        help=(
            "Repeatable staged task ID governed by operator-sealed manual "
            "completion. Pending descendants must be freshly revalidated "
            "after its seal becomes active."
        ),
    )
    parser.add_argument(
        "--manual-completion-authority-required-task-id",
        action="append",
        default=[],
        help=(
            "Repeatable staged task ID whose status cannot be selected or "
            "accepted as complete until a fresh scheduler load verifies its "
            "operator seal."
        ),
    )
    parser.add_argument(
        "--manual-completion-authority-epoch-id",
        default="",
        help=(
            "Content-addressed identity of the verified manual-completion "
            "seal and policy set used for descendant revalidation."
        ),
    )
    parser.add_argument(
        "--manual-completion-authority-revalidation-only",
        action="store_true",
        help=(
            "Supervise only zero-provider manual-completion authority "
            "revalidation and suppress ordinary supervisor maintenance, "
            "refill, merge repair, and implementation dispatch."
        ),
    )
    parser.add_argument(
        "--llm-merge-resolver-command",
        default=os.environ.get("IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND", ""),
        help=(
            "Command invoked with merge-conflict repair prompts on stdin. "
            "Passed to the managed daemon as IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND."
        ),
    )
    parser.add_argument(
        "--llm-merge-resolver-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Timeout for the merge resolver subprocess. Passed to the managed daemon as "
            "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS; defaults to that env var "
            "or 1800 seconds; <=0 disables."
        ),
    )
    parser.add_argument(
        "--allow-reconciliation-only-llm-resolver",
        action="store_true",
        help=(
            "Allow --reconciliation-only passes to invoke the configured LLM merge resolver. "
            "By default reconciliation-only disables this to keep cleanup probes non-interactive."
        ),
    )
    parser.add_argument("--implementation-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--implementation-max-timeout",
        type=float,
        default=None,
        help=(
            "Maximum task-specific implementation hard timeout in this lane. "
            "This extends only the parent watchdog; --implementation-timeout "
            "remains the daemon's ordinary and no-progress policy."
        ),
    )
    parser.add_argument(
        "--implementation-log-stall-seconds",
        type=float,
        default=300.0,
        help="Recycle an active implementation attempt after this many seconds without log output; <=0 disables.",
    )
    parser.add_argument(
        "--validation-max-workers",
        type=int,
        default=None,
        help=(
            "Maximum validation subprocesses used by the managed daemon. "
            "Defaults to the daemon policy when omitted."
        ),
    )
    parser.add_argument(
        "--no-ephemeral-worktree",
        action="store_true",
        help="Run implementation commands in the main checkout instead of isolated temporary git worktrees",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=None,
        help="Directory for temporary implementation worktrees",
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
        "--merge-queue-dir",
        type=Path,
        default=None,
        help=(
            "Explicit merge-queue namespace propagated to every managed "
            "daemon. Requests are still bound to the repository and target."
        ),
    )
    parser.add_argument(
        "--daemon-script-path",
        type=Path,
        default=None,
        help="Python script used to launch the managed daemon instead of the package module.",
    )
    parser.add_argument(
        "--supervisor-script-path",
        type=Path,
        default=None,
        help="Python script used to relaunch this supervisor from external wrappers.",
    )
    parser.add_argument(
        "--worktree-submodule-path",
        action="append",
        default=[],
        help=(
            "Repo-relative submodule path to initialize and commit inside implementation worktrees. "
            "May be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--reconciliation-only",
        action="store_true",
        help=(
            "Run only supervisor reconciliation/cleanup checks. This disables implementation, "
            "retry/dependency/reconciliation guardrail writes, and objective/codebase refill scans."
        ),
    )
    parser.add_argument(
        "--fail-on-reconciliation-error",
        action="store_true",
        help=(
            "With --once, return a non-zero exit status unless historical "
            "reconciliation replay is fully settled. This lets launchers "
            "fail closed before starting implementation providers."
        ),
    )
    parser.add_argument(
        "--no-worktree-reconciliation",
        dest="worktree_reconciliation_enabled",
        action="store_false",
        help="Disable supervisor retry/cleanup reconciliation for clean inactive implementation worktrees.",
    )
    parser.set_defaults(worktree_reconciliation_enabled=True)
    parser.add_argument(
        "--worktree-reconciliation-max-merges",
        type=int,
        default=1,
        help="Maximum clean backlogged implementation branches to merge per supervisor pass.",
    )
    parser.add_argument(
        "--worktree-reconciliation-dry-run",
        action="store_true",
        help="Classify clean backlogged implementation worktrees without merging or removing them.",
    )
    parser.add_argument(
        "--no-worktree-reconciliation-preflight",
        dest="worktree_reconciliation_preflight_enabled",
        action="store_false",
        help=(
            "Disable non-mutating merge-tree preflight before merging a backlogged "
            "implementation branch into the main checkout."
        ),
    )
    parser.set_defaults(worktree_reconciliation_preflight_enabled=True)
    parser.add_argument(
        "--no-worktree-scan-cache",
        dest="worktree_scan_cache_enabled",
        action="store_false",
        help="Disable cached non-mutating worktree reconciliation/cleanup classifications.",
    )
    parser.set_defaults(worktree_scan_cache_enabled=True)
    parser.add_argument(
        "--worktree-scan-cache-ttl-seconds",
        type=float,
        default=DEFAULT_WORKTREE_SCAN_CACHE_TTL_SECONDS,
        help="Seconds to reuse cached non-mutating worktree scan classifications; <=0 disables the cache.",
    )
    parser.add_argument(
        "--worktree-scan-cache-path",
        type=Path,
        default=None,
        help="JSON cache path for non-mutating worktree scan classifications. Defaults to the supervisor state dir.",
    )
    parser.add_argument(
        "--merge-reconciliation-max-merges",
        type=int,
        default=None,
        help=(
            "Maximum failed merge-reconciliation repairs for the managed implementation daemon "
            "per pass. Defaults to the daemon setting."
        ),
    )
    parser.add_argument(
        "--daemon-merged-worktree-cleanup-max",
        type=int,
        default=None,
        help=(
            "Maximum already-merged implementation worktrees for the managed implementation daemon "
            "to remove per pass. Defaults to the daemon setting."
        ),
    )
    parser.add_argument(
        "--task-shard-count",
        type=int,
        default=1,
        help="Total deterministic task-selection shards for this supervisor lane.",
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic task-selection shard index for this supervisor lane.",
    )
    parser.add_argument(
        "--strict-task-sharding",
        action="store_true",
        help=(
            "Keep the managed daemon within its deterministic task shard when that "
            "shard has no ready work; disables cross-shard ready-task fallback."
        ),
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
        "--execution-slice-task-id",
        action="append",
        default=[],
        help="Repeatable task ID this leased bundle lane is authorized to execute.",
    )
    parser.add_argument(
        "--execution-slice-task-cid",
        action="append",
        default=[],
        help="Repeatable canonical task CID this leased bundle lane is authorized to execute.",
    )
    parser.add_argument(
        "--plan-bound-dispatch",
        action="store_true",
        help="Use exact compiled lane slices and disable legacy hash sharding.",
    )
    parser.add_argument("--plan-revision-store-path", type=Path, default=None)
    parser.add_argument("--plan-bound-revision-cid", default="")
    parser.add_argument("--plan-bound-plan-root-cid", default="")
    parser.add_argument("--plan-bound-execution-plan-cid", default="")
    parser.add_argument("--plan-bound-capacity-snapshot-id", default="")
    parser.add_argument("--plan-bound-slice-manifest-cid", default="")
    parser.add_argument("--plan-bound-slice-id", default="")
    parser.add_argument("--plan-bound-lane-id", default="")
    parser.add_argument("--plan-bound-reassignment-cid", default="")
    parser.add_argument("--plan-bound-source-head", default="")
    parser.add_argument("--plan-bound-source-tree", default="")
    parser.add_argument("--plan-bound-task-source-revision", default="")
    parser.add_argument("--plan-bound-configuration-root", default="")
    parser.add_argument("--plan-bound-accepted-tree-root", type=Path, default=None)
    parser.add_argument(
        "--accepted-control-plane-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-retry-budget-guardrail",
        dest="retry_budget_guardrail_enabled",
        action="store_false",
        help="Disable conversion of repeated implementation, validation, or merge failures into follow-up tasks.",
    )
    parser.set_defaults(retry_budget_guardrail_enabled=True)
    parser.add_argument(
        "--retry-budget-discovery-dir",
        type=Path,
        default=None,
        help="Directory for retry-budget discovery reports. Defaults to a sibling discovery directory near state.",
    )
    parser.add_argument("--retry-budget-discovery-output-path", default="")
    parser.add_argument("--validation-retry-budget", type=int, default=3)
    parser.add_argument("--merge-retry-budget", type=int, default=3)
    parser.add_argument("--implementation-retry-budget", type=int, default=3)
    parser.add_argument(
        "--retry-budget-commit-outputs",
        action="store_true",
        help="Commit generated retry-budget todo/discovery outputs.",
    )
    parser.add_argument(
        "--retry-budget-commit-subject",
        default="Agent: record retry-budget guardrail outputs",
    )
    parser.add_argument(
        "--no-dependency-guardrail",
        dest="dependency_guardrail_enabled",
        action="store_false",
        help="Disable conversion of missing/self-referential dependencies into ready repair tasks.",
    )
    parser.set_defaults(dependency_guardrail_enabled=True)
    parser.add_argument("--dependency-guardrail-discovery-dir", type=Path, default=None)
    parser.add_argument("--dependency-guardrail-discovery-output-path", default="")
    parser.add_argument("--dependency-guardrail-max-findings", type=int, default=5)
    parser.add_argument(
        "--dependency-guardrail-commit-outputs",
        action="store_true",
        help="Commit generated dependency-guardrail todo/discovery outputs.",
    )
    parser.add_argument(
        "--dependency-guardrail-commit-subject",
        default="Agent: record dependency guardrail outputs",
    )
    parser.add_argument(
        "--no-reconciliation-guardrail",
        dest="reconciliation_guardrail_enabled",
        action="store_false",
        help="Disable conversion of blocked checkout/worktree reconciliation into ready cleanup tasks.",
    )
    parser.set_defaults(reconciliation_guardrail_enabled=True)
    parser.add_argument("--reconciliation-guardrail-discovery-dir", type=Path, default=None)
    parser.add_argument("--reconciliation-guardrail-discovery-output-path", default="")
    parser.add_argument("--reconciliation-guardrail-max-findings", type=int, default=3)
    parser.add_argument(
        "--reconciliation-guardrail-commit-outputs",
        action="store_true",
        help="Commit generated reconciliation-guardrail todo/discovery outputs.",
    )
    parser.add_argument(
        "--reconciliation-guardrail-commit-subject",
        default="Agent: record reconciliation guardrail outputs",
    )
    parser.add_argument(
        "--auto-commit-generated-dirty",
        dest="generated_dirty_repair_enabled",
        action="store_true",
        help=(
            "Commit safe supervisor-generated dirty todo/discovery/objective outputs before "
            "worktree reconciliation and after refill generation."
        ),
    )
    parser.set_defaults(generated_dirty_repair_enabled=False)
    parser.add_argument(
        "--generated-dirty-commit-subject",
        default="Agent: commit generated supervisor outputs",
    )
    parser.add_argument(
        "--no-generated-dirty-submodule-gitlinks",
        dest="generated_dirty_repair_include_submodule_gitlinks",
        action="store_false",
        help="Do not commit clean submodule gitlink updates during generated dirty repair.",
    )
    parser.set_defaults(generated_dirty_repair_include_submodule_gitlinks=False)
    parser.add_argument(
        "--generated-dirty-max-paths",
        type=int,
        default=200,
        help="Maximum dirty generated paths to stage per repair pass.",
    )
    parser.add_argument(
        "--generated-dirty-stale-lock-seconds",
        type=float,
        default=300.0,
        help=(
            "Minimum age before generated-dirty repair may remove an inactive "
            "Git index.lock in a candidate repository."
        ),
    )
    parser.add_argument(
        "--generated-dirty-path",
        dest="generated_dirty_repair_paths",
        type=Path,
        action="append",
        default=[],
        help="Repeatable generated file path that dirty-checkout repair may safely manage.",
    )
    parser.add_argument(
        "--codebase-refill-scan",
        action="store_true",
        help="Append codebase-scan follow-up tasks when the supervised backlog is low or drained.",
    )
    parser.add_argument(
        "--codebase-scan-discovery-dir",
        type=Path,
        default=None,
        help="Directory for codebase-scan discovery reports. Defaults to a sibling discovery directory near state.",
    )
    parser.add_argument(
        "--codebase-scan-discovery-output-path",
        default="",
        help="Todo Outputs path used for generated codebase-scan tasks.",
    )
    parser.add_argument(
        "--codebase-scan-min-open-tasks",
        type=int,
        default=0,
        help="Run the refill scan when open tasks are at or below this count.",
    )
    parser.add_argument("--codebase-scan-max-findings", type=int, default=5)
    parser.add_argument("--codebase-scan-cooldown-seconds", type=int, default=21600)
    parser.add_argument(
        "--codebase-refill-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Abort supervisor-owned codebase refill after this many seconds. "
            "A timed-out codebase pass yields no todos and records a cooldown marker."
        ),
    )
    parser.add_argument(
        "--codebase-scan-depends-on",
        action="append",
        default=[],
        help="Task id dependency for generated codebase-scan tasks. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--codebase-scan-skip-prefix",
        action="append",
        default=[],
        help="Repo-relative path prefix to skip during codebase scans. May be repeated.",
    )
    parser.add_argument(
        "--allow-unscoped-codebase-refill",
        action="store_true",
        help=(
            "Allow codebase findings without goal/subgoal lineage to become tasks. "
            "This compatibility escape hatch is unsafe for goal-backed boards."
        ),
    )
    parser.add_argument(
        "--codebase-scan-commit-outputs",
        action="store_true",
        help="Commit generated todo/discovery outputs after a supervisor codebase scan.",
    )
    parser.add_argument(
        "--allow-codebase-refill-with-objective-work",
        dest="codebase_defer_when_objective_refills",
        action="store_false",
        help="Allow codebase-scan refill in the same supervisor pass that objective refill creates goal work.",
    )
    parser.set_defaults(codebase_defer_when_objective_refills=True)
    parser.add_argument(
        "--codebase-scan-commit-subject",
        default="Agent: record supervisor codebase scan findings",
    )
    parser.add_argument(
        "--objective-refill-scan",
        action="store_true",
        help="Refine the objective heap and append objective-gap todos when the supervised backlog is low or drained.",
    )
    parser.add_argument(
        "--no-objective-task-janitor",
        dest="objective_task_janitor_enabled",
        action="store_false",
        help="Disable strategy reconciliation that blocks orphaned objective tasks and reopens launch-critical goals.",
    )
    parser.set_defaults(objective_task_janitor_enabled=True)
    parser.add_argument("--objective-task-janitor-max-blocked-tasks", type=int, default=50)
    parser.add_argument("--objective-task-janitor-max-deprioritized-tasks", type=int, default=50)
    parser.add_argument("--objective-task-janitor-max-reopened-goals", type=int, default=12)
    parser.add_argument(
        "--objective-mission-term",
        action="append",
        default=[],
        help=(
            "Mission term that marks active goals/tasks as launch-critical for supervisor steering. "
            "May be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--objective-path",
        type=Path,
        default=None,
        help="Objective goal heap markdown document. Defaults to implementation_plan/docs/23-virtual-ai-os-objective-goal-heap.md.",
    )
    parser.add_argument("--objective-graph-path", type=Path, default=None)
    parser.add_argument("--objective-bundle-dir", type=Path, default=None)
    parser.add_argument("--objective-dataset-dir", type=Path, default=None)
    parser.add_argument("--objective-discovery-dir", type=Path, default=None)
    parser.add_argument("--objective-discovery-output-path", default="")
    parser.add_argument("--objective-summary-prefix", default="")
    parser.add_argument(
        "--no-objective-goal-refinement",
        dest="objective_refine_goals",
        action="store_false",
        help="Generate todos from the objective heap without appending new subgoals.",
    )
    parser.set_defaults(objective_refine_goals=True)
    parser.add_argument(
        "--no-objective-goal-completion-reconcile",
        dest="objective_reconcile_goal_completion",
        action="store_false",
        help="Do not mark objective goals completed when their required evidence is already present.",
    )
    parser.set_defaults(objective_reconcile_goal_completion=True)
    parser.add_argument(
        "--objective-goal-completion-todo-board",
        action="append",
        default=[],
        help=(
            "Extra todo board that can keep shared objective goals open while referenced tasks remain pending. "
            "Use 'path::TASK-' or 'path::## TASK-' and repeat for cross-track boards."
        ),
    )
    parser.add_argument(
        "--objective-goal-completion-gate-path",
        type=Path,
        default=None,
        help=(
            "External per-goal completion-gate artifact forwarded to the "
            "objective reconciler."
        ),
    )
    parser.add_argument(
        "--objective-goal-completion-evidence-path",
        type=Path,
        default=None,
        help=(
            "External canonical per-goal CompletionEvidence artifact forwarded "
            "to the objective reconciler."
        ),
    )
    parser.add_argument(
        "--objective-goal-completion-artifact-refresh-command",
        default="",
        help=(
            "Explicit argv command run with shell disabled immediately before "
            "completion reconciliation to refresh configured proof artifacts."
        ),
    )
    parser.add_argument(
        "--objective-goal-completion-artifact-refresh-timeout-seconds",
        type=float,
        default=300.0,
        help="Positive timeout for the configured completion-artifact refresh command.",
    )
    parser.add_argument(
        "--no-objective-goal-migration",
        dest="objective_goal_migration_enabled",
        action="store_false",
        help="Disable idempotent migration of legacy completed objective goals.",
    )
    parser.set_defaults(objective_goal_migration_enabled=True)
    parser.add_argument(
        "--objective-goal-migration-preview",
        action="store_true",
        help=(
            "Classify the next legacy-completion batch and publish diagnostics without "
            "rewriting the objective document."
        ),
    )
    parser.add_argument(
        "--objective-goal-migration-batch-size",
        type=int,
        default=100,
        help="Maximum legacy completed goals to migrate per supervisor pass (minimum 1).",
    )
    parser.add_argument(
        "--objective-seed-interoperability-goals",
        action="store_true",
        help="Seed objective goals for cross-submodule interoperability and integration tests.",
    )
    parser.add_argument(
        "--objective-seed-launch-readiness-goals",
        action="store_true",
        help=(
            "Seed launch-readiness goals for Swissknife virtual desktop, Hallucinate App MCP "
            "dashboards, backend MCP servers, and Meta glasses control-plane integration."
        ),
    )
    parser.add_argument(
        "--objective-interoperability-focus",
        action="append",
        default=[],
        help=(
            "Submodule path to pair with other submodules for interoperability goal seeding. "
            "If omitted, all submodule pairs are eligible."
        ),
    )
    parser.add_argument(
        "--objective-interoperability-component-path",
        action="append",
        default=[],
        help=(
            "Repo-relative component path to include when seeding interoperability goals. "
            "Defaults to configured worktree submodule paths when omitted."
        ),
    )
    parser.add_argument("--objective-max-interoperability-goals", type=int, default=12)
    parser.add_argument("--objective-max-launch-readiness-goals", type=int, default=8)
    parser.add_argument(
        "--objective-ensure-tracking-document",
        action="store_true",
        help="Create the objective heap with a root goal if it does not exist.",
    )
    parser.add_argument("--objective-ultimate-goal", default="")
    parser.add_argument("--objective-root-evidence", action="append", default=[])
    parser.add_argument("--objective-goal-prefix", default=None)
    parser.add_argument("--objective-root-goal-id", default=None)
    parser.add_argument("--objective-root-goal-title", default="")
    parser.add_argument("--objective-tracking-document-title", default="")
    parser.add_argument("--objective-scan-min-open-tasks", type=int, default=0)
    parser.add_argument("--objective-scan-max-findings", type=int, default=5)
    parser.add_argument("--objective-scan-cooldown-seconds", type=int, default=21600)
    parser.add_argument(
        "--objective-scan-exclude-path",
        action="append",
        default=[],
        help=(
            "Repo-relative operational or control path excluded from objective "
            "evidence scans and completion-tree identity. May be repeated or "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--objective-refill-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Abort supervisor-owned objective refill after this many seconds. "
            "A timed-out objective pass yields no todos so codebase refill can still run."
        ),
    )
    parser.add_argument(
        "--objective-scan-depends-on",
        action="append",
        default=[],
        help="Task id dependency for generated objective tasks. May be repeated or comma-separated.",
    )
    parser.add_argument("--objective-max-refinement-children", type=int, default=3)
    parser.add_argument("--objective-max-refinement-depth", type=int, default=4)
    parser.add_argument(
        "--objective-surplus-findings-per-goal",
        type=int,
        default=DEFAULT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL,
        help=(
            "Generate surplus structured objective todos per missing goal. "
            "Additional candidates are vector-indexed and bundled with related work."
        ),
    )
    parser.add_argument(
        "--objective-surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO,
        help="Minimum missing-evidence terms for non-aggregate objective surplus todos.",
    )
    parser.add_argument(
        "--objective-todo-vector-index-path",
        type=Path,
        default=None,
        help="Path for the objective todo vector/AST index. Defaults to <objective-bundle-dir>/todo_vector_index.json.",
    )
    parser.add_argument(
        "--no-objective-ast-dataset",
        dest="objective_persist_ast_dataset",
        action="store_false",
        help="Skip persisting the objective AST/evidence dataset while refilling.",
    )
    parser.set_defaults(objective_persist_ast_dataset=True)
    parser.add_argument(
        "--no-objective-todo-vector-index",
        dest="objective_write_todo_vector_index",
        action="store_false",
        help="Skip writing the objective todo vector/AST index while refilling.",
    )
    parser.set_defaults(objective_write_todo_vector_index=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parsed = parser.parse_args(expanded_argv)
    parsed.scheduler_config = scheduler_config_path
    return parsed


def supervisor_config_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    daemon_script_path: Path | None = None,
    supervisor_script_path: Path | None = None,
    worktree_submodule_paths: Any = None,
    implementation_protected_paths: Any = None,
    state_path: Path | None = None,
    strategy_path: Path | None = None,
    events_path: Path | None = None,
) -> PortalSupervisorConfig:
    """Build a supervisor config from parsed CLI args with optional embedding overrides."""

    resolved_worktree_submodule_paths = (
        args.worktree_submodule_path if worktree_submodule_paths is None else worktree_submodule_paths
    )
    resolved_implementation_protected_paths = (
        args.implementation_protected_path
        if implementation_protected_paths is None
        else implementation_protected_paths
    )
    plan_bound_dispatch = bool(getattr(args, "plan_bound_dispatch", False))
    raw_repo_root = Path(repo_root or REPO_ROOT)
    if plan_bound_dispatch:
        accepted_tree = getattr(args, "plan_bound_accepted_tree_root", None)
        store_input = getattr(args, "plan_revision_store_path", None)
        scheduler_input = getattr(args, "scheduler_config", None)
        if accepted_tree is None or store_input is None or scheduler_input is None:
            raise PlanBoundDispatchError(
                "plan-bound configuration is missing accepted-tree/store/config authority"
            )
        (
            effective_repo_root,
            effective_state_dir,
            effective_plan_store,
            effective_scheduler_config,
            effective_todo_path,
        ) = _validated_plan_bound_authority_paths(
            repo_root=raw_repo_root,
            accepted_tree_root=accepted_tree,
            state_dir=args.state_dir,
            plan_revision_store_path=store_input,
            scheduler_config_path=scheduler_input,
            todo_path=args.todo_path,
            require_live_module_root=True,
        )

        def state_artifact(
            override: Path | None,
            filename: str,
            field_name: str,
        ) -> Path:
            candidate = override or effective_state_dir / filename
            validated = _plan_bound_contained_path(
                effective_repo_root,
                candidate,
                field_name=field_name,
            )
            if validated.parent != effective_state_dir:
                raise PlanBoundDispatchError(
                    f"plan-bound {field_name} escapes its lane state directory"
                )
            return validated

        effective_state_path = state_artifact(
            state_path,
            f"{args.state_prefix}_task_state.json",
            "task state",
        )
        effective_strategy_path = state_artifact(
            strategy_path,
            f"{args.state_prefix}_strategy.json",
            "strategy state",
        )
        effective_events_path = state_artifact(
            events_path,
            f"{args.state_prefix}_supervisor_events.jsonl",
            "event log",
        )
        assert effective_scheduler_config is not None
        assert effective_todo_path is not None
        from ..runtime.multi_supervisor_runner import (
            parse_accepted_control_plane_pin,
        )

        try:
            effective_control_plane_pin = parse_accepted_control_plane_pin(
                getattr(args, "accepted_control_plane_pin_json", "")
            )
            effective_control_plane_descriptor = int(
                getattr(args, "accepted_control_plane_fd", -1)
            )
            verify_agent_implementation_sealed_control_plane(
                effective_control_plane_pin,
                effective_control_plane_descriptor,
            )
        except (OSError, ValueError) as exc:
            raise PlanBoundDispatchError(
                "plan-bound control-plane launch binding is invalid"
            ) from exc
    else:
        effective_repo_root = raw_repo_root.resolve()
        effective_state_dir = args.state_dir
        effective_plan_store = (
            Path(args.plan_revision_store_path).resolve()
            if getattr(args, "plan_revision_store_path", None) is not None
            else None
        )
        effective_scheduler_config = (
            Path(args.scheduler_config).resolve()
            if getattr(args, "scheduler_config", None)
            else None
        )
        effective_todo_path = args.todo_path
        effective_state_path = (
            state_path or args.state_dir / f"{args.state_prefix}_task_state.json"
        )
        effective_strategy_path = (
            strategy_path or args.state_dir / f"{args.state_prefix}_strategy.json"
        )
        effective_events_path = (
            events_path
            or args.state_dir / f"{args.state_prefix}_supervisor_events.jsonl"
        )
        effective_control_plane_pin = None
        effective_control_plane_descriptor = -1
    reconciliation_only = bool(args.reconciliation_only)
    implement = bool(args.implement and not reconciliation_only)
    llm_merge_resolver_command = normalize_llm_merge_resolver_command(
        args.llm_merge_resolver_command
    )
    if reconciliation_only and not args.allow_reconciliation_only_llm_resolver:
        llm_merge_resolver_command = ""
    database_program = database_program_from_cli_namespace(args)
    return PortalSupervisorConfig(
        todo_path=effective_todo_path,
        state_path=effective_state_path,
        strategy_path=effective_strategy_path,
        events_path=effective_events_path,
        state_dir=effective_state_dir,
        stale_seconds=args.stale_seconds,
        check_interval=args.check_interval,
        watchdog_startup_grace_seconds=args.watchdog_startup_grace_seconds,
        max_restarts=args.max_restarts,
        max_task_attempts=max(0, int(getattr(args, "max_task_attempts", 0))),
        daemon_interval=args.daemon_interval,
        task_prefix=args.task_prefix,
        state_prefix=args.state_prefix,
        database_program=database_program,
        reconciliation_only=reconciliation_only,
        implement=implement,
        implementation_command=args.implementation_command,
        llm_merge_resolver_command=llm_merge_resolver_command,
        llm_merge_resolver_timeout_seconds=args.llm_merge_resolver_timeout_seconds,
        implementation_timeout=args.implementation_timeout,
        implementation_max_timeout=args.implementation_max_timeout,
        implementation_log_stall_seconds=args.implementation_log_stall_seconds,
        validation_max_workers=args.validation_max_workers,
        use_ephemeral_worktree=implement and not args.no_ephemeral_worktree,
        worktree_root=args.worktree_root,
        merge_target_branch=args.merge_target_branch,
        merge_queue_dir=args.merge_queue_dir,
        worktree_submodule_paths=normalize_relative_path_list(resolved_worktree_submodule_paths),
        implementation_protected_paths=normalize_implementation_protected_paths(
            resolved_implementation_protected_paths,
            repo_root=effective_repo_root,
        ),
        manual_completion_authority_task_ids=tuple(
            dict.fromkeys(
                str(task_id).strip()
                for task_id in (
                    args.manual_completion_authority_task_id or ()
                )
                if str(task_id).strip()
            )
        ),
        manual_completion_authority_required_task_ids=tuple(
            dict.fromkeys(
                str(task_id).strip()
                for task_id in (
                    args.manual_completion_authority_required_task_id or ()
                )
                if str(task_id).strip()
            )
        ),
        manual_completion_authority_epoch_id=str(
            getattr(args, "manual_completion_authority_epoch_id", "") or ""
        ).strip(),
        manual_completion_authority_revalidation_only=bool(
            getattr(
                args,
                "manual_completion_authority_revalidation_only",
                False,
            )
        ),
        scheduler_config_path=effective_scheduler_config,
        worktree_reconciliation_enabled=args.worktree_reconciliation_enabled,
        worktree_reconciliation_max_merges=args.worktree_reconciliation_max_merges,
        worktree_reconciliation_dry_run=args.worktree_reconciliation_dry_run,
        worktree_reconciliation_preflight_enabled=args.worktree_reconciliation_preflight_enabled,
        worktree_scan_cache_enabled=args.worktree_scan_cache_enabled,
        worktree_scan_cache_ttl_seconds=args.worktree_scan_cache_ttl_seconds,
        worktree_scan_cache_path=args.worktree_scan_cache_path,
        merge_reconciliation_max_merges=args.merge_reconciliation_max_merges,
        daemon_merged_worktree_cleanup_max=args.daemon_merged_worktree_cleanup_max,
        task_shard_count=args.task_shard_count,
        task_shard_index=args.task_shard_index,
        strict_task_sharding=bool(
            getattr(args, "strict_task_sharding", False)
        ),
        external_reservation_manifest_paths=tuple(
            args.external_reservation_manifest_path or ()
        ),
        assumed_completed_task_ids=tuple(args.assume_completed_task_id or ()),
        execution_slice_task_ids=tuple(args.execution_slice_task_id or ()),
        execution_slice_task_cids=tuple(args.execution_slice_task_cid or ()),
        plan_bound_dispatch=plan_bound_dispatch,
        plan_revision_store_path=effective_plan_store,
        plan_bound_revision_cid=str(
            getattr(args, "plan_bound_revision_cid", "") or ""
        ).strip(),
        plan_bound_plan_root_cid=str(
            getattr(args, "plan_bound_plan_root_cid", "") or ""
        ).strip(),
        plan_bound_execution_plan_cid=str(
            getattr(args, "plan_bound_execution_plan_cid", "") or ""
        ).strip(),
        plan_bound_capacity_snapshot_id=str(
            getattr(args, "plan_bound_capacity_snapshot_id", "") or ""
        ).strip(),
        plan_bound_slice_manifest_cid=str(
            getattr(args, "plan_bound_slice_manifest_cid", "") or ""
        ).strip(),
        plan_bound_slice_id=str(
            getattr(args, "plan_bound_slice_id", "") or ""
        ).strip(),
        plan_bound_lane_id=str(
            getattr(args, "plan_bound_lane_id", "") or ""
        ).strip(),
        plan_bound_reassignment_cid=str(
            getattr(args, "plan_bound_reassignment_cid", "") or ""
        ).strip(),
        plan_bound_source_head=str(
            getattr(args, "plan_bound_source_head", "") or ""
        ).strip(),
        plan_bound_source_tree=str(
            getattr(args, "plan_bound_source_tree", "") or ""
        ).strip(),
        plan_bound_task_source_revision=str(
            getattr(args, "plan_bound_task_source_revision", "") or ""
        ).strip(),
        plan_bound_configuration_root=str(
            getattr(args, "plan_bound_configuration_root", "") or ""
        ).strip(),
        plan_bound_accepted_tree_root=(
            effective_repo_root if plan_bound_dispatch else None
        ),
        accepted_control_plane_pin=effective_control_plane_pin,
        accepted_control_plane_descriptor=(
            effective_control_plane_descriptor
        ),
        retry_budget_guardrail_enabled=args.retry_budget_guardrail_enabled and not reconciliation_only,
        retry_budget_discovery_dir=args.retry_budget_discovery_dir,
        retry_budget_discovery_output_path=args.retry_budget_discovery_output_path,
        validation_retry_budget=args.validation_retry_budget,
        merge_retry_budget=args.merge_retry_budget,
        implementation_retry_budget=args.implementation_retry_budget,
        retry_budget_commit_outputs=args.retry_budget_commit_outputs,
        retry_budget_commit_subject=args.retry_budget_commit_subject,
        dependency_guardrail_enabled=args.dependency_guardrail_enabled and not reconciliation_only,
        dependency_guardrail_discovery_dir=args.dependency_guardrail_discovery_dir,
        dependency_guardrail_discovery_output_path=args.dependency_guardrail_discovery_output_path,
        dependency_guardrail_max_findings=args.dependency_guardrail_max_findings,
        dependency_guardrail_commit_outputs=args.dependency_guardrail_commit_outputs,
        dependency_guardrail_commit_subject=args.dependency_guardrail_commit_subject,
        reconciliation_guardrail_enabled=args.reconciliation_guardrail_enabled,
        reconciliation_guardrail_discovery_dir=args.reconciliation_guardrail_discovery_dir,
        reconciliation_guardrail_discovery_output_path=args.reconciliation_guardrail_discovery_output_path,
        reconciliation_guardrail_max_findings=args.reconciliation_guardrail_max_findings,
        reconciliation_guardrail_commit_outputs=args.reconciliation_guardrail_commit_outputs,
        reconciliation_guardrail_commit_subject=args.reconciliation_guardrail_commit_subject,
        generated_dirty_repair_enabled=args.generated_dirty_repair_enabled,
        generated_dirty_repair_commit_subject=args.generated_dirty_commit_subject,
        generated_dirty_repair_include_submodule_gitlinks=(
            args.generated_dirty_repair_include_submodule_gitlinks
        ),
        generated_dirty_repair_max_paths=args.generated_dirty_max_paths,
        generated_dirty_repair_stale_lock_seconds=args.generated_dirty_stale_lock_seconds,
        generated_dirty_repair_paths=tuple(args.generated_dirty_repair_paths),
        codebase_refill_enabled=args.codebase_refill_scan and not reconciliation_only,
        codebase_scan_discovery_dir=args.codebase_scan_discovery_dir,
        codebase_scan_discovery_output_path=args.codebase_scan_discovery_output_path,
        codebase_scan_min_open_tasks=args.codebase_scan_min_open_tasks,
        codebase_scan_max_findings=args.codebase_scan_max_findings,
        codebase_scan_cooldown_seconds=args.codebase_scan_cooldown_seconds,
        codebase_refill_timeout_seconds=args.codebase_refill_timeout_seconds,
        codebase_scan_depends_on=split_csv_values(args.codebase_scan_depends_on),
        codebase_scan_skip_prefixes=tuple(args.codebase_scan_skip_prefix),
        allow_unscoped_codebase_refill=args.allow_unscoped_codebase_refill,
        codebase_defer_when_objective_refills=args.codebase_defer_when_objective_refills,
        codebase_scan_commit_outputs=args.codebase_scan_commit_outputs,
        codebase_scan_commit_subject=args.codebase_scan_commit_subject,
        objective_refill_enabled=args.objective_refill_scan and not reconciliation_only,
        objective_task_janitor_enabled=args.objective_task_janitor_enabled and not reconciliation_only,
        objective_task_janitor_max_blocked_tasks=args.objective_task_janitor_max_blocked_tasks,
        objective_task_janitor_max_deprioritized_tasks=args.objective_task_janitor_max_deprioritized_tasks,
        objective_task_janitor_max_reopened_goals=args.objective_task_janitor_max_reopened_goals,
        objective_task_janitor_mission_terms=split_csv_values(args.objective_mission_term),
        objective_path=args.objective_path,
        objective_graph_path=args.objective_graph_path,
        objective_bundle_dir=args.objective_bundle_dir,
        objective_dataset_dir=args.objective_dataset_dir,
        objective_discovery_dir=args.objective_discovery_dir,
        objective_discovery_output_path=args.objective_discovery_output_path,
        objective_summary_prefix=args.objective_summary_prefix,
        objective_refine_goals=args.objective_refine_goals,
        objective_reconcile_goal_completion=args.objective_reconcile_goal_completion,
        objective_goal_completion_todo_boards=tuple(args.objective_goal_completion_todo_board),
        objective_goal_completion_gate_path=args.objective_goal_completion_gate_path,
        objective_goal_completion_evidence_path=(
            args.objective_goal_completion_evidence_path
        ),
        objective_goal_completion_artifact_refresh_command=(
            args.objective_goal_completion_artifact_refresh_command
        ),
        objective_goal_completion_artifact_refresh_timeout_seconds=(
            args.objective_goal_completion_artifact_refresh_timeout_seconds
        ),
        objective_goal_migration_enabled=(
            args.objective_goal_migration_enabled and not reconciliation_only
        ),
        objective_goal_migration_preview=args.objective_goal_migration_preview,
        objective_goal_migration_batch_size=max(1, args.objective_goal_migration_batch_size),
        objective_seed_interoperability_goals=args.objective_seed_interoperability_goals,
        objective_seed_launch_readiness_goals=args.objective_seed_launch_readiness_goals,
        objective_interoperability_focus=split_csv_values(args.objective_interoperability_focus),
        objective_interoperability_component_paths=split_csv_values(
            args.objective_interoperability_component_path
        ),
        objective_max_interoperability_goals=args.objective_max_interoperability_goals,
        objective_max_launch_readiness_goals=args.objective_max_launch_readiness_goals,
        objective_ensure_tracking_document=args.objective_ensure_tracking_document,
        objective_ultimate_goal=args.objective_ultimate_goal,
        objective_root_evidence=split_csv_values(args.objective_root_evidence),
        objective_goal_prefix=args.objective_goal_prefix,
        objective_root_goal_id=args.objective_root_goal_id,
        objective_root_goal_title=args.objective_root_goal_title,
        objective_tracking_document_title=args.objective_tracking_document_title,
        objective_scan_min_open_tasks=args.objective_scan_min_open_tasks,
        objective_scan_max_findings=args.objective_scan_max_findings,
        objective_scan_cooldown_seconds=args.objective_scan_cooldown_seconds,
        objective_scan_exclude_paths=split_csv_values(
            args.objective_scan_exclude_path
        ),
        objective_refill_timeout_seconds=args.objective_refill_timeout_seconds,
        objective_scan_depends_on=split_csv_values(args.objective_scan_depends_on),
        objective_max_refinement_children=args.objective_max_refinement_children,
        objective_max_refinement_depth=args.objective_max_refinement_depth,
        objective_persist_ast_dataset=args.objective_persist_ast_dataset,
        objective_write_todo_vector_index=args.objective_write_todo_vector_index,
        objective_todo_vector_index_path=args.objective_todo_vector_index_path,
        objective_surplus_findings_per_goal=args.objective_surplus_findings_per_goal,
        objective_surplus_min_terms_per_todo=args.objective_surplus_min_terms_per_todo,
        repo_root=effective_repo_root,
        daemon_script_path=daemon_script_path if daemon_script_path is not None else args.daemon_script_path,
        supervisor_script_path=supervisor_script_path
        if supervisor_script_path is not None
        else args.supervisor_script_path,
    )


def _reconciliation_preflight_failure_reason(
    result: Mapping[str, Any],
) -> str:
    """Return why a strict one-shot reconciliation pass is not settled."""

    if result.get("maintenance_blocked") is True:
        return str(result.get("reason") or "maintenance_blocked")

    replay = result.get("worktree_reconciliation_replay")
    if not isinstance(replay, Mapping):
        return "reconciliation_replay_result_missing"

    reason = str(replay.get("reason") or "")
    allowed_reasons = {
        "no_pending_reconciliation_replays",
        "reconciliation_replays_processed",
    }
    if reason not in allowed_reasons:
        return f"reconciliation_replay_unverified:{reason or 'missing_reason'}"

    counts: dict[str, int] = {}
    for field_name in (
        "pending_count",
        "processed_count",
        "completed_count",
        "failed_count",
        "deferred_count",
    ):
        value = replay.get(field_name)
        try:
            count = int(value)
        except (TypeError, ValueError):
            return f"reconciliation_replay_invalid_{field_name}"
        if count < 0:
            return f"reconciliation_replay_invalid_{field_name}"
        counts[field_name] = count

    results = replay.get("results")
    if not isinstance(results, list):
        return "reconciliation_replay_results_missing"
    if counts["failed_count"] > 0:
        return "reconciliation_replay_failed"
    if counts["deferred_count"] > 0:
        return "reconciliation_replay_deferred"
    if counts["pending_count"] != len(results):
        return "reconciliation_replay_pending"
    if counts["processed_count"] != len(results):
        return "reconciliation_replay_unprocessed"

    completed_results = 0
    for item in results:
        if not isinstance(item, Mapping) or item.get("settled") is not True:
            return "reconciliation_replay_unsettled"
        completed = item.get("completed") is True
        queued = item.get("queued") is True
        if not completed and not queued:
            return "reconciliation_replay_settlement_unproven"
        if completed:
            completed_results += 1
    if counts["completed_count"] != completed_results:
        return "reconciliation_replay_completion_count_mismatch"

    if reason == "no_pending_reconciliation_replays":
        if results or any(counts.values()):
            return "reconciliation_replay_no_pending_count_mismatch"
        return ""
    if not results:
        return "reconciliation_replay_processed_without_results"
    return ""


def _run_plan_bound_daemon_child(argv: Sequence[str]) -> int:
    """Run the existing daemon with one canonical active-plan store binding.

    The public daemon CLI predates its production ``plan_revision_store``
    constructor contract.  This narrow bootstrap is used only by a validated
    plan-bound supervisor child.  It injects the real store, immutable plan,
    and original capacity evidence into that existing constructor, then
    delegates the complete daemon lifecycle to its owning module.
    """

    try:
        separator = tuple(argv).index("--")
    except ValueError as exc:
        raise PlanBoundDispatchError(
            "plan-bound daemon child is missing its argument boundary"
        ) from exc
    binding_argv = list(argv[:separator])
    daemon_argv = list(argv[separator + 1 :])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--daemon-entrypoint", required=True)
    parser.add_argument("--scheduler-config", type=Path, required=True)
    parser.add_argument("--plan-revision-store-path", type=Path, required=True)
    parser.add_argument("--plan-bound-revision-cid", required=True)
    parser.add_argument("--plan-bound-plan-root-cid", required=True)
    parser.add_argument("--plan-bound-execution-plan-cid", required=True)
    parser.add_argument("--plan-bound-capacity-snapshot-id", required=True)
    parser.add_argument("--plan-bound-slice-manifest-cid", required=True)
    parser.add_argument("--plan-bound-slice-id", required=True)
    parser.add_argument("--plan-bound-lane-id", required=True)
    parser.add_argument("--plan-bound-reassignment-cid", default="")
    parser.add_argument("--plan-bound-source-head", required=True)
    parser.add_argument("--plan-bound-source-tree", required=True)
    parser.add_argument("--plan-bound-task-source-revision", required=True)
    parser.add_argument("--plan-bound-configuration-root", required=True)
    parser.add_argument("--plan-bound-accepted-tree-root", type=Path, required=True)
    parser.add_argument("--accepted-control-plane-pin-json", required=True)
    parser.add_argument("--accepted-control-plane-fd", type=int, required=True)
    parser.add_argument("--plan-bound-task-id", action="append", default=[])
    parser.add_argument("--plan-bound-task-cid", action="append", default=[])
    pinned = parser.parse_args(binding_argv)
    if pinned.daemon_entrypoint != PLAN_BOUND_DAEMON_ENTRYPOINT:
        raise PlanBoundDispatchError("plan-bound daemon entrypoint is foreign")
    from ..runtime.multi_supervisor_runner import (
        parse_accepted_control_plane_pin,
    )

    try:
        accepted_control_plane_pin = parse_accepted_control_plane_pin(
            pinned.accepted_control_plane_pin_json
        )
        sealed_executable = verify_agent_implementation_sealed_control_plane(
            accepted_control_plane_pin,
            pinned.accepted_control_plane_fd,
        )
        accepted_control_plane_launch = AgentImplementationSealedControlPlane(
            descriptor=pinned.accepted_control_plane_fd,
            executable_path=sealed_executable,
            archive_sha256=accepted_control_plane_pin.archive_sha256,
            seals=int(
                fcntl.fcntl(
                    pinned.accepted_control_plane_fd,
                    fcntl.F_GET_SEALS,
                )
            ),
            capsule_id=accepted_control_plane_pin.capsule_id,
        )
    except (OSError, ValueError) as exc:
        raise PlanBoundDispatchError(
            "plan-bound daemon sealed control plane is invalid"
        ) from exc
    if (
        accepted_control_plane_pin.source_head
        != pinned.plan_bound_source_head
        or accepted_control_plane_pin.source_tree
        != pinned.plan_bound_source_tree
    ):
        raise PlanBoundDispatchError(
            "plan-bound daemon control-plane generation drifted"
        )
    task_ids = tuple(str(value).strip() for value in pinned.plan_bound_task_id)
    task_cids = tuple(str(value).strip() for value in pinned.plan_bound_task_cid)
    if (
        len(task_ids) != 1
        or len(task_cids) != 1
        or len(set(task_ids)) != len(task_ids)
        or len(set(task_cids)) != len(task_cids)
        or any(not value for value in (*task_ids, *task_cids))
    ):
        raise PlanBoundDispatchError(
            "plan-bound daemon child requires one exact nonempty ID/CID slice"
        )

    from . import implementation_daemon as daemon_module

    daemon_args = daemon_module.parse_args(daemon_argv)
    if not daemon_args.once:
        raise PlanBoundDispatchError("plan-bound daemon child must be bounded")
    if (
        int(daemon_args.task_shard_count) != 1
        or int(daemon_args.task_shard_index) != 0
        or bool(daemon_args.strict_task_sharding)
        or tuple(daemon_args.execution_slice_task_id)
        or tuple(daemon_args.execution_slice_task_cid) != task_cids
    ):
        raise PlanBoundDispatchError(
            "daemon selection arguments differ from the plan-bound slice"
        )

    (
        accepted_tree_root,
        daemon_state_dir,
        validated_store_path,
        scheduler_config_path,
        taskboard_path,
    ) = _validated_plan_bound_authority_paths(
        repo_root=daemon_module.REPO_ROOT,
        accepted_tree_root=pinned.plan_bound_accepted_tree_root,
        state_dir=daemon_args.state_dir,
        plan_revision_store_path=pinned.plan_revision_store_path,
        scheduler_config_path=pinned.scheduler_config,
        todo_path=daemon_args.todo_path,
        # The verified control-plane module is executing from its sealed
        # archive, so its ``__file__`` is intentionally not beneath the
        # mutable repository.  Repository authority is instead bound by the
        # exact CLI paths plus the active PlanRevisionStore transaction below.
        require_live_module_root=False,
    )
    assert scheduler_config_path is not None
    assert taskboard_path is not None

    store = PlanRevisionStore(validated_store_path)
    plan_adapter = ProductionParallelPlanAdapter(store)
    with store._thread_lock:  # noqa: SLF001 - canonical store transaction
        with store._guard():  # noqa: SLF001 - canonical process guard
            binding = _load_plan_revision_store_binding_locked(
                store,
                execution_slice_task_ids=task_ids,
                execution_slice_task_cids=task_cids,
            )
            from ..planning.plan_revision_contracts import PlanRevision

            revision_payload = _secure_store_cas(
                store,
                binding.revision_cid,
            )
            revision = PlanRevision.from_dict(revision_payload)
            if revision.to_dict() != revision_payload:
                raise PlanBoundDispatchError(
                    "active revision changed during typed decode"
                )
            manifest = ConfiguredBoardExecutionSlices.from_dict(
                _secure_store_cas(
                    store,
                    pinned.plan_bound_slice_manifest_cid,
                )
            )
            try:
                owned_slice = plan_adapter._validate_slice_owner_locked(  # noqa: SLF001
                    revision_cid=pinned.plan_bound_revision_cid,
                    slice_manifest_cid=pinned.plan_bound_slice_manifest_cid,
                    slice_id=pinned.plan_bound_slice_id,
                    lane_id=pinned.plan_bound_lane_id,
                    reassignment_cid=pinned.plan_bound_reassignment_cid,
                )
            except Exception as exc:
                raise PlanBoundDispatchError(
                    "plan-bound daemon child does not own the canonical slice"
                ) from exc
            initial_execution = _load_plan_bound_execution_lease_locked(
                store,
                revision_cid=pinned.plan_bound_revision_cid,
                slice_id=pinned.plan_bound_slice_id,
                lane_id=pinned.plan_bound_lane_id,
            )
    recovery_phases = {
        "proposal_ready",
        "merge_enqueue_prepared",
        "merge_enqueue_confirmed",
        "merge_completed",
    }
    recovery_only = (
        initial_execution is not None
        and initial_execution[1].phase in recovery_phases
    )
    configured_board = None
    _board_bytes = b""
    if recovery_only:
        assert initial_execution is not None
        recovery_lease = initial_execution[1]
        handoff = _secure_store_cas(
            store,
            recovery_lease.proposal_handoff_cid,
        )
        enqueue_fields = handoff.get("enqueue_fields")
        enqueue_metadata = (
            enqueue_fields.get("metadata")
            if isinstance(enqueue_fields, Mapping)
            else None
        )
        if not isinstance(enqueue_metadata, Mapping):
            raise PlanBoundDispatchError(
                "recoverable proposal handoff lacks canonical enqueue metadata"
            )
        recovery_paths = {
            "repo_root": accepted_tree_root,
            "todo_path": taskboard_path,
            "state_path": (
                daemon_state_dir
                / f"{daemon_args.state_prefix}_task_state.json"
            ),
            "strategy_path": (
                daemon_state_dir
                / f"{daemon_args.state_prefix}_strategy.json"
            ),
            "events_path": (
                daemon_state_dir
                / f"{daemon_args.state_prefix}_events.jsonl"
            ),
        }
        for name, expected_path in recovery_paths.items():
            observed_path = enqueue_metadata.get(name)
            if (
                not isinstance(observed_path, str)
                or Path(observed_path) != Path(expected_path)
            ):
                raise PlanBoundDispatchError(
                    "recoverable proposal handoff path authority is mixed"
                )
        if (
            daemon_state_dir.parent != validated_store_path.parent
            or manifest.configuration_root
            != pinned.plan_bound_configuration_root
            or manifest.task_source_revision
            != pinned.plan_bound_task_source_revision
            or manifest.source_head != pinned.plan_bound_source_head
            or manifest.repository_tree_id != pinned.plan_bound_source_tree
        ):
            raise PlanBoundDispatchError(
                "recoverable proposal handoff lost its immutable plan roots"
            )
    else:
        try:
            from ..runtime.configured_board_scheduler import (
                _git_identity as configured_board_git_identity,
            )
            from ..runtime.configured_board_scheduler import (
                _tracked_head_snapshot,
                load_configured_board,
            )

            observed_head, observed_tree = configured_board_git_identity(
                accepted_tree_root
            )
            configured_board = load_configured_board(
                scheduler_config_path,
                repo_root=accepted_tree_root,
            )
            config_bytes, _config_revision = _tracked_head_snapshot(
                repo_root=accepted_tree_root,
                path=scheduler_config_path,
                source_head=pinned.plan_bound_source_head,
            )
            _board_bytes, observed_task_source_revision = _tracked_head_snapshot(
                repo_root=accepted_tree_root,
                path=taskboard_path,
                source_head=pinned.plan_bound_source_head,
            )
            configured_state_root = _plan_bound_contained_path(
                accepted_tree_root,
                configured_board.path(configured_board.runtime_paths["state"]),
                field_name="configured runtime state root",
                require_directory=True,
            )
            configured_taskboard = _plan_bound_contained_path(
                accepted_tree_root,
                configured_board.path(configured_board.taskboard_path),
                field_name="configured task board",
                require_existing=True,
                require_regular=True,
            )
        except Exception as exc:
            raise PlanBoundDispatchError(
                "cannot re-observe tracked plan-bound config/task authority"
            ) from exc
        if (
            observed_head != pinned.plan_bound_source_head
            or observed_tree != pinned.plan_bound_source_tree
        ):
            raise PlanBoundDispatchError(
                "repository identity crossed the plan-bound child fence"
            )
        if (
            daemon_state_dir.parent != configured_state_root
            or validated_store_path
            != configured_state_root / "plan-revision-store"
            or taskboard_path != configured_taskboard
        ):
            raise PlanBoundDispatchError(
                "plan-bound child paths differ from the configured runtime authority"
            )
        if (
            configured_board.configuration_root
            != pinned.plan_bound_configuration_root
        ):
            raise PlanBoundDispatchError(
                "scheduler configuration crossed the plan-bound child fence"
            )
        if content_identity(
            {"bytes_sha256": sha256(config_bytes).hexdigest()}
        ) != pinned.plan_bound_configuration_root:
            raise PlanBoundDispatchError(
                "tracked scheduler bytes differ from the child configuration root"
            )
        if observed_task_source_revision != pinned.plan_bound_task_source_revision:
            raise PlanBoundDispatchError(
                "task source crossed the plan-bound child fence"
            )

    expected_binding = {
        "revision_cid": pinned.plan_bound_revision_cid,
        "plan_root_cid": pinned.plan_bound_plan_root_cid,
        "execution_plan_cid": pinned.plan_bound_execution_plan_cid,
        "capacity_snapshot_id": pinned.plan_bound_capacity_snapshot_id,
    }
    observed_binding = {
        "revision_cid": binding.revision_cid,
        "plan_root_cid": binding.plan_root_cid,
        "execution_plan_cid": binding.execution_plan_cid,
        "capacity_snapshot_id": binding.capacity_snapshot_id,
    }
    if observed_binding != expected_binding:
        raise PlanBoundDispatchError(
            "plan-bound daemon child observed a mixed active revision"
        )
    if (
        revision.materialization_transaction_cid
        != pinned.plan_bound_slice_manifest_cid
    ):
        raise PlanBoundDispatchError(
            "active revision no longer owns the pinned slice manifest"
        )
    if (
        owned_slice.task_ids != task_ids
        or owned_slice.task_cids != task_cids
        or manifest.configuration_root != pinned.plan_bound_configuration_root
    ):
        raise PlanBoundDispatchError(
            "plan-bound daemon child slice differs from its CAS manifest"
        )

    plan_payload = dict(binding.execution_plan)
    replay_request = plan_payload.get("replay_request")
    capacity = (
        replay_request.get("capacity_snapshot")
        if isinstance(replay_request, Mapping)
        else None
    )
    if not isinstance(capacity, Mapping):
        raise PlanBoundDispatchError(
            "active execution plan lacks its capacity observation"
        )
    host = capacity.get("host")
    providers = capacity.get("providers")
    if not isinstance(host, Mapping) or not isinstance(providers, Sequence):
        raise PlanBoundDispatchError(
            "active execution plan capacity observation is partial"
        )
    provider_records = tuple(
        dict(item) for item in providers if isinstance(item, Mapping)
    )
    planned_profile_id = capacity.get("route_capacity_profile_id")
    if (
        len(provider_records) != len(providers)
        or len(provider_records) != 1
        or provider_records[0].get("schema")
        != "ipfs_accelerate_py.agent_supervisor.implementation-route-capacity@2"
        or not isinstance(planned_profile_id, str)
        or not planned_profile_id
        or provider_records[0].get("profile_id") != planned_profile_id
        or provider_records[0].get("provider_id")
        != revision.provider_contract.provider_requirement
    ):
        raise PlanBoundDispatchError(
            "active execution plan logical route capacity is partial"
        )

    if recovery_only:
        # Recovery may only consume the immutable proposal/queue handoff.  It
        # must neither require fresh provider headroom nor reinterpret a later
        # board/HEAD generation, because it cannot dispatch a provider and the
        # canonical merge train owns rebase/revalidation against current HEAD.
        live_host = dict(host)
        relevant_live_providers = provider_records
    else:
        # Re-observe capacity at the final accepted-tree daemon boundary.  The
        # stored snapshot proves what was compiled; it is not live headroom and
        # cannot by itself authorize a new provider spawn after capacity changes.
        from ..runtime.configured_board_scheduler import (
            configured_board_capacity_observation,
            configured_board_route_capacity_projection,
        )

        try:
            live_host, live_provider_observations, live_now_ms = (
                configured_board_capacity_observation(configured_board)
            )
            live_route_capacity, live_route = (
                configured_board_route_capacity_projection(
                    configured_board,
                    provider_capacity_snapshots=live_provider_observations,
                    now_ms=live_now_ms,
                )
            )
        except Exception as exc:
            raise PlanBoundDispatchError(
                "fresh live capacity evidence is unavailable"
            ) from exc
        if (
            configured_board is None
            or configured_board.board_namespace != manifest.board_namespace
        ):
            raise PlanBoundDispatchError(
                "scheduler profile and slice manifest namespaces disagree"
            )

        widths = plan_payload.get("widths")
        widths = widths if isinstance(widths, Mapping) else {}
        planned_width = int(
            widths.get("admitted")
            or widths.get("resource")
            or widths.get("conflict")
            or widths.get("graph")
            or len(binding.ready_wave_task_ids)
            or 1
        )
        candidate_task_ids = tuple(binding.ready_wave_task_ids) or tuple(
            task_id for item in manifest.nonempty for task_id in item.task_ids
        )
        provider_requirement = str(
            revision.provider_contract.provider_requirement or ""
        ).strip()
        if live_route.route_id != provider_requirement:
            raise PlanBoundDispatchError(
                "live router route identity differs from the active revision"
            )
        relevant_live_providers = (live_route_capacity,)
        stale_live_capacity = bool(
            live_route_capacity.get("healthy") is not True
            or live_route_capacity.get("schedulable") is not True
            or not isinstance(live_route_capacity.get("fresh_until_ms"), int)
            or live_now_ms >= int(live_route_capacity.get("fresh_until_ms") or 0)
        )
        live_capacity_id = content_identity(
            {
                "host": live_host,
                "providers": [live_route_capacity],
                "provider_observations": [
                    dict(item) for item in live_provider_observations
                ],
                "route_capacity_profile_id": live_route_capacity["profile_id"],
            }
        )
        capacity_decision = evaluate_capacity_drift(
            planned_width=planned_width,
            planned_capacity_snapshot_id=binding.capacity_snapshot_id,
            planned_capacity=host,
            live_host=live_host,
            live_providers=relevant_live_providers,
            live_capacity_snapshot_id=live_capacity_id,
            candidate_task_ids=candidate_task_ids,
            provider_id=provider_requirement,
            require_provider=True,
            stale_capacity=stale_live_capacity,
            current_time_ms=live_now_ms,
        )
        if (
            not capacity_decision.may_dispatch
            or not set(task_ids).issubset(capacity_decision.admitted_task_ids)
        ):
            raise PlanBoundDispatchError(
                "live capacity drift fenced this slice for coordinator replan: "
                + json.dumps(capacity_decision.to_dict(), sort_keys=True)
            )

    canonical_daemon_class = daemon_module.PortalImplementationDaemon
    canonical_parse_task_file = daemon_module.parse_task_file
    cid_by_id = dict(zip(task_ids, task_cids, strict=True))

    def plan_bound_parse_task_file(*args: Any, **kwargs: Any) -> list[Any]:
        """Attach the immutable ID/CID pairs after exact board-byte validation."""

        if recovery_only:
            requested_path = Path(
                args[0] if args else kwargs.get("path", taskboard_path)
            )
            requested_prefix = str(
                args[1]
                if len(args) > 1
                else kwargs.get(
                    "task_header_prefix",
                    daemon_module.TASK_HEADER_PREFIX,
                )
            )
            if requested_path != taskboard_path or len(args) > 2:
                raise PlanBoundDispatchError(
                    "merge-only recovery requested a foreign task board"
                )
            try:
                from ..runtime.configured_board_scheduler import (
                    _git_identity as configured_board_git_identity,
                )
                from ..runtime.configured_board_scheduler import (
                    _tracked_head_snapshot,
                )

                recovery_head, _recovery_tree = (
                    configured_board_git_identity(accepted_tree_root)
                )
                recovery_board_bytes, _recovery_board_revision = (
                    _tracked_head_snapshot(
                        repo_root=accepted_tree_root,
                        path=taskboard_path,
                        source_head=recovery_head,
                    )
                )
                tasks = daemon_module.parse_task_text(
                    recovery_board_bytes.decode("utf-8"),
                    path=taskboard_path,
                    task_header_prefix=requested_prefix,
                )
            except Exception as exc:
                raise PlanBoundDispatchError(
                    "merge-only recovery lacks a stable current task board"
                ) from exc
        else:
            tasks = canonical_parse_task_file(*args, **kwargs)
        observed_ids = {task.task_id for task in tasks}
        if not set(cid_by_id).issubset(observed_ids):
            raise PlanBoundDispatchError(
                "plan-bound daemon task IDs disappeared from the tracked board"
            )
        observed_pairs = {
            task.task_id: str(task.canonical_task_cid or "").strip()
            for task in tasks
            if task.task_id in cid_by_id
        }
        if observed_pairs != cid_by_id:
            raise PlanBoundDispatchError(
                "plan-bound daemon task identity drifted from the immutable slice"
            )
        return [
            replace(
                task,
                board_namespace=manifest.board_namespace,
            )
            if task.task_id in cid_by_id
            else task
            for task in tasks
        ]
    store_view = _PlanBoundRevisionStoreView(
        store,
        pinned.plan_bound_revision_cid,
        slice_manifest_cid=pinned.plan_bound_slice_manifest_cid,
        slice_id=pinned.plan_bound_slice_id,
        lane_id=pinned.plan_bound_lane_id,
        reassignment_cid=pinned.plan_bound_reassignment_cid,
    )

    def stable_effect_record(path: Path) -> tuple[dict[str, Any], str]:
        """Read a canonical effect guard and bind its exact file bytes."""

        from ..runtime.multi_supervisor_runner import (
            _read_stable_regular_json,
        )

        try:
            payload, evidence = _read_stable_regular_json(path)
        except Exception as exc:
            raise PlanBoundDispatchError(
                f"cannot securely read plan-bound effect guard: {path}"
            ) from exc
        mode = stat.S_IMODE(int(evidence.get("mode") or 0))
        if (
            payload is None
            or int(evidence.get("uid", -1)) != os.geteuid()
            or mode != 0o600
            or not isinstance(evidence.get("content_sha256"), str)
            or not str(evidence["content_sha256"]).startswith("sha256:")
        ):
            raise PlanBoundDispatchError(
                "plan-bound effect guard is absent, foreign-owned, or not "
                f"exactly mode 0600: {path}"
            )
        return payload, str(evidence["content_sha256"])

    def stable_effect_json(path: Path) -> dict[str, Any]:
        """Read a canonical effect guard without following mutable names."""

        return stable_effect_record(path)[0]

    def seal_owned_effect_guard(path: Path) -> None:
        """Tighten a newly acquired canonical claim before trusting it."""

        artifact = Path(path)
        try:
            before = os.lstat(artifact)
        except OSError as exc:
            raise PlanBoundDispatchError(
                "cannot inspect newly acquired canonical task claim"
            ) from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or int(before.st_uid) != os.geteuid()
        ):
            raise PlanBoundDispatchError(
                "newly acquired canonical task claim has unsafe ownership"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(artifact, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
                or int(opened.st_nlink) != 1
                or int(opened.st_uid) != os.geteuid()
            ):
                raise PlanBoundDispatchError(
                    "canonical task claim changed before permission seal"
                )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(artifact)
        if (
            (after.st_dev, after.st_ino) != (sealed.st_dev, sealed.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o600
            or int(after.st_nlink) != 1
            or int(after.st_uid) != os.geteuid()
        ):
            raise PlanBoundDispatchError(
                "canonical task claim permission seal was not stable"
            )

    class PlanBoundImplementationDaemon(canonical_daemon_class):
        """Delegate daemon policy while joining its real fenced resources."""

        def _restore_out_of_scope_workspace_paths(
            self,
            workspace_path: Path,
            task: Any,
            *,
            baseline_ref: str,
        ) -> list[str]:
            """Preserve actual drift for the unchanged typed proposal gate.

            The legacy daemon may discard incidental out-of-scope dirt before
            proposal collection.  A plan-bound wave instead needs the existing
            validator to observe every actual endpoint so a rejected diff can
            fence the optimistic plan.  This grants no new path authority:
            preserved drift can only make the canonical validator fail closed.
            """

            del workspace_path, task, baseline_ref
            return []

        def _load_execution_lease(
            self,
            *,
            phases: Sequence[str],
        ) -> tuple[str, PlanBoundExecutionLease]:
            with store._thread_lock:  # noqa: SLF001 - canonical store order
                with store._guard():  # noqa: SLF001 - canonical process guard
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed is None or observed[1].phase not in set(phases):
                        raise PlanBoundDispatchError(
                            "plan-bound execution lease is absent or in the wrong phase"
                        )
                    lease_cid, execution_lease = observed
                    try:
                        from ..control.lifecycle_orchestrator import (
                            LinuxProcessAdapter as SupervisorProcessAdapter,
                        )
                        from ..control.lifecycle_orchestrator import (
                            ProcessIdentity as SupervisorProcessIdentity,
                        )

                        supervisor_birth = SupervisorProcessIdentity.from_dict(
                            execution_lease.process_birth
                        )
                        if (
                            supervisor_birth.to_dict()
                            != execution_lease.process_birth
                            or not SupervisorProcessAdapter().identity_alive(
                                supervisor_birth
                            )
                        ):
                            raise ValueError("supervisor process birth is not live")
                    except Exception as exc:
                        raise PlanBoundDispatchError(
                            "plan-bound execution lost its gated supervisor birth"
                        ) from exc
                    return lease_cid, execution_lease

        @staticmethod
        def _current_daemon_birth() -> dict[str, Any]:
            from ..merge.worktree_lifecycle import current_process_birth

            birth = current_process_birth()
            if birth.pid != os.getpid() or birth.start_time_ticks <= 0:
                raise PlanBoundDispatchError(
                    "cannot capture exact plan-bound daemon process birth"
                )
            return birth.to_dict()

        @staticmethod
        def _require_compiled_claim_metadata(
            execution_lease: PlanBoundExecutionLease,
            metadata: Mapping[str, Any],
        ) -> tuple[str, str, Mapping[str, Any]]:
            task_id = str(metadata.get("task_id") or "")
            task_cid = str(metadata.get("canonical_task_cid") or "")
            assignment = execution_lease.assignment_for(task_id, task_cid)
            expected = {
                "plan_revision_cid": execution_lease.revision_cid,
                "execution_plan_id": str(plan_payload.get("plan_id") or ""),
                "compiled_lease_id": assignment["lease_id"],
                "compiled_lease_scope": assignment["lease_scope"],
                "compiled_worktree_id": assignment["worktree_id"],
                "compiled_worktree_path": assignment["worktree_path"],
                "compiled_fence_epoch": assignment["fence_epoch"],
                "compiled_fence_token": assignment["fence_token"],
                "compiled_affinity_key": assignment["affinity_key"],
                "compiled_exclusive_group": assignment["exclusive_group"],
                "compiled_exclusive_paths": assignment["exclusive_paths"],
                "compiled_provider_id": assignment["provider_id"],
                "compiled_resource_class": assignment["resource_class"],
            }
            if any(metadata.get(name) != value for name, value in expected.items()):
                raise PlanBoundDispatchError(
                    "canonical task claim differs from its compiled assignment"
                )
            if (
                metadata.get("compiled_claim_acquired_before_publish") is not True
                or metadata.get("lease_id") != assignment["lease_id"]
                or metadata.get("pid") != os.getpid()
            ):
                raise PlanBoundDispatchError(
                    "canonical task claim is not bound to the reserved lease"
                )
            return task_id, task_cid, assignment

        @staticmethod
        def _require_exact_claim(
            *,
            path: Path,
            expected: Mapping[str, Any],
        ) -> dict[str, Any]:
            observed = stable_effect_json(path)
            if observed != dict(expected):
                raise PlanBoundDispatchError(
                    "canonical task claim changed at the execution boundary"
                )
            return observed

        def _try_acquire_implementation_lock(
            self,
            lock_path: Path,
            metadata: dict[str, Any],
        ) -> tuple[bool, str, dict[str, Any] | None]:
            """Seal the canonical global attempt lease at acquisition time."""

            acquired, reason, existing = super()._try_acquire_implementation_lock(
                lock_path,
                metadata,
            )
            if acquired:
                # The legacy primitive creates this owner-held JSON with an
                # executable mode.  A plan-bound launch tightens its own newly
                # acquired lease before task claim, provider, or recovery
                # authority can observe it.  Recovery itself never repairs a
                # pre-existing executable artifact.
                seal_owned_effect_guard(lock_path)
            return acquired, reason, existing

        def _try_acquire_implementation_task_claim(
            self,
            lock_path: Path,
            metadata: dict[str, Any],
        ) -> tuple[bool, str, dict[str, Any] | None]:
            before_cid, before = self._load_execution_lease(
                phases=("reserved",)
            )
            task_id, task_cid, assignment = self._require_compiled_claim_metadata(
                before,
                metadata,
            )
            expected_claim_path = self._implementation_task_claim_path(
                task_id,
                canonical_task_cid=task_cid,
            )
            if Path(lock_path) != expected_claim_path:
                raise PlanBoundDispatchError(
                    "daemon requested a foreign canonical task claim path"
                )
            acquired, reason, existing = super()._try_acquire_implementation_task_claim(
                lock_path,
                metadata,
            )
            if not acquired:
                return acquired, reason, existing
            try:
                daemon_birth = self._current_daemon_birth()
                with store._thread_lock:  # noqa: SLF001
                    with store._guard():  # noqa: SLF001
                        current = _load_plan_bound_execution_lease_locked(
                            store,
                            revision_cid=pinned.plan_bound_revision_cid,
                            slice_id=pinned.plan_bound_slice_id,
                            lane_id=pinned.plan_bound_lane_id,
                        )
                        if (
                            current is None
                            or current[0] != before_cid
                            or current[1] != before
                        ):
                            raise PlanBoundDispatchError(
                                "execution lease changed during task claim acquisition"
                            )
                        with serialized_lock_update(lock_path):
                            seal_owned_effect_guard(lock_path)
                            claim = self._require_exact_claim(
                                path=lock_path,
                                expected=metadata,
                            )
                            self._require_compiled_claim_metadata(before, claim)
                            claim_cid = content_identity(claim)
                            claimed = replace(
                                before,
                                generation=before.generation + 1,
                                phase="claimed",
                                prior_execution_lease_cid=before_cid,
                                active_task_id=task_id,
                                active_task_cid=task_cid,
                                daemon_process_birth=daemon_birth,
                                canonical_claim_path=str(lock_path),
                                canonical_claim_cid=claim_cid,
                                canonical_claim_lease_id=str(assignment["lease_id"]),
                            )
                            _publish_plan_bound_execution_lease_locked(
                                store,
                                claimed,
                                expected_current_cid=before_cid,
                            )
                            if content_identity(
                                self._require_exact_claim(
                                    path=lock_path,
                                    expected=metadata,
                                )
                            ) != claim_cid:
                                raise PlanBoundDispatchError(
                                    "canonical task claim identity changed after binding"
                                )
            except BaseException:
                super()._release_implementation_task_claim(lock_path, metadata)
                raise
            return acquired, reason, existing

        def _require_bound_claim(
            self,
            execution_lease: PlanBoundExecutionLease,
        ) -> dict[str, Any]:
            claim_path = Path(execution_lease.canonical_claim_path)
            claim = stable_effect_json(claim_path)
            if (
                content_identity(claim) != execution_lease.canonical_claim_cid
                or claim.get("lease_id")
                != execution_lease.canonical_claim_lease_id
                or claim.get("task_id") != execution_lease.active_task_id
                or claim.get("canonical_task_cid")
                != execution_lease.active_task_cid
                or claim.get("pid") != os.getpid()
                or claim.get("compiled_fence_token")
                != execution_lease.assignment_for(
                    execution_lease.active_task_id,
                    execution_lease.active_task_cid,
                )["fence_token"]
            ):
                raise PlanBoundDispatchError(
                    "canonical task claim no longer matches the execution lease"
                )
            return claim

        def _read_exact_worktree_lifecycle(
            self,
            *,
            execution_lease: PlanBoundExecutionLease,
            workspace_path: Path,
            required_state: str | Sequence[str],
        ) -> tuple[dict[str, Any], Path, str]:
            from ..merge.worktree_lifecycle import WorkspaceLifecycleRecord

            lifecycle_path = self.worktree_lifecycle.workspace_path_for(
                workspace_path
            )
            raw, raw_cid = stable_effect_record(lifecycle_path)
            try:
                lifecycle = WorkspaceLifecycleRecord.from_dict(raw)
            except Exception as exc:
                raise PlanBoundDispatchError(
                    "worktree lifecycle record is malformed"
                ) from exc
            daemon_birth = self._current_daemon_birth()
            allowed_states = (
                {required_state}
                if isinstance(required_state, str)
                else set(required_state)
            )
            if not allowed_states or any(
                not isinstance(value, str) or not value
                for value in allowed_states
            ):
                raise PlanBoundDispatchError(
                    "worktree lifecycle state requirement is invalid"
                )
            if (
                lifecycle.to_dict() != raw
                or lifecycle.state.value not in allowed_states
                or lifecycle.task_id != execution_lease.active_task_id
                or lifecycle.canonical_task_cid
                != execution_lease.active_task_cid
                or lifecycle.owner.to_dict() != daemon_birth
                or lifecycle.workspace_path
                != str(Path(workspace_path).resolve(strict=False))
                or lifecycle.state_dir
                != str(self.state_path.parent.resolve(strict=False))
                or lifecycle.fence < 1
                or not lifecycle.lease_id
            ):
                raise PlanBoundDispatchError(
                    "worktree lifecycle differs from the canonical execution claim"
                )
            return raw, lifecycle_path, raw_cid

        @staticmethod
        def _seal_owned_directory(path: Path) -> None:
            """Seal one freshly created runtime directory to its exact owner."""

            try:
                before = os.lstat(path)
            except OSError as exc:
                raise PlanBoundDispatchError(
                    "cannot inspect plan-bound workspace custody"
                ) from exc
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISDIR(before.st_mode)
                or int(before.st_uid) != os.geteuid()
                or bool(stat.S_IMODE(before.st_mode) & 0o7000)
            ):
                raise PlanBoundDispatchError(
                    "plan-bound workspace directory custody is unsafe"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                    or not stat.S_ISDIR(opened.st_mode)
                    or int(opened.st_uid) != os.geteuid()
                ):
                    raise PlanBoundDispatchError(
                        "plan-bound workspace directory changed before sealing"
                    )
                os.fchmod(descriptor, 0o700)
                sealed = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = os.lstat(path)
            if (
                (after.st_dev, after.st_ino) != (sealed.st_dev, sealed.st_ino)
                or stat.S_IMODE(after.st_mode) != 0o700
                or int(after.st_uid) != os.geteuid()
            ):
                raise PlanBoundDispatchError(
                    "plan-bound workspace directory seal was not stable"
                )

        @staticmethod
        def _seal_worktree_marker(path: Path) -> bytes:
            """Read and seal one newly created Git worktree marker."""

            try:
                before = os.lstat(path)
            except OSError as exc:
                raise PlanBoundDispatchError(
                    "cannot inspect plan-bound Git worktree marker"
                ) from exc
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or int(before.st_uid) != os.geteuid()
                or int(before.st_nlink) != 1
                or bool(stat.S_IMODE(before.st_mode) & 0o111)
            ):
                raise PlanBoundDispatchError(
                    "plan-bound Git worktree marker custody is unsafe"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                    or not stat.S_ISREG(opened.st_mode)
                    or int(opened.st_uid) != os.geteuid()
                    or int(opened.st_nlink) != 1
                ):
                    raise PlanBoundDispatchError(
                        "plan-bound Git worktree marker changed before sealing"
                    )
                payload = os.read(descriptor, 16_385)
                if len(payload) > 16_384:
                    raise PlanBoundDispatchError(
                        "plan-bound Git worktree marker is oversized"
                    )
                os.fchmod(descriptor, 0o600)
                sealed = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = os.lstat(path)
            if (
                (after.st_dev, after.st_ino) != (sealed.st_dev, sealed.st_ino)
                or stat.S_IMODE(after.st_mode) != 0o600
                or int(after.st_uid) != os.geteuid()
                or int(after.st_nlink) != 1
            ):
                raise PlanBoundDispatchError(
                    "plan-bound Git worktree marker seal was not stable"
                )
            return payload

        def _seal_plan_bound_workspace_custody(
            self,
            workspace_path: Path,
        ) -> None:
            """Seal the exact fresh worktree and its canonical Git custody."""

            workspace = Path(os.path.abspath(workspace_path))
            worktree_root = Path(os.path.abspath(self.worktree_root))
            repository_root = Path(os.path.abspath(self.repo_root))
            if workspace.parent != worktree_root or workspace == repository_root:
                raise PlanBoundDispatchError(
                    "plan-bound workspace escapes its configured runtime root"
                )
            self._seal_owned_directory(worktree_root)
            self._seal_owned_directory(workspace)
            marker = workspace / ".git"
            marker_payload = self._seal_worktree_marker(marker)
            try:
                marker_text = marker_payload.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise PlanBoundDispatchError(
                    "plan-bound Git worktree marker is not text"
                ) from exc
            if not marker_text.startswith("gitdir: "):
                raise PlanBoundDispatchError(
                    "plan-bound Git worktree marker is malformed"
                )
            raw_git_dir = Path(marker_text[8:])
            git_dir = (
                raw_git_dir
                if raw_git_dir.is_absolute()
                else workspace / raw_git_dir
            )
            git_dir = Path(os.path.abspath(git_dir))
            git_root = repository_root / ".git"
            worktree_git_root = git_root / "worktrees"
            if git_dir.parent != worktree_git_root:
                raise PlanBoundDispatchError(
                    "plan-bound Git worktree custody escapes the repository"
                )
            for directory in (git_root, worktree_git_root, git_dir):
                self._seal_owned_directory(directory)

        def _build_implementation_command(
            self,
            workspace_path: Path,
            *,
            task: Any | None = None,
            prompt: str = "",
            attempt: int = 0,
            state: Any | None = None,
        ) -> list[str]:
            current_cid, current = self._load_execution_lease(
                phases=("claimed",)
            )
            task_id = str(getattr(task, "task_id", "") or "")
            task_cid = str(self._canonical_ref(task) if task is not None else "")
            if (task_id, task_cid) != (
                current.active_task_id,
                current.active_task_cid,
            ):
                raise PlanBoundDispatchError(
                    "worktree preparation crossed the canonical task pair"
                )
            self._seal_plan_bound_workspace_custody(workspace_path)
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed is None or observed[0] != current_cid:
                        raise PlanBoundDispatchError(
                            "execution lease changed before worktree binding"
                        )
                    claim_path = Path(current.canonical_claim_path)
                    lifecycle_path = self.worktree_lifecycle.workspace_path_for(
                        workspace_path
                    )
                    with serialized_lock_update(claim_path):
                        self._require_bound_claim(current)
                        with serialized_lock_update(lifecycle_path):
                            lifecycle_raw, exact_lifecycle_path, lifecycle_cid = (
                                self._read_exact_worktree_lifecycle(
                                    execution_lease=current,
                                    workspace_path=workspace_path,
                                    required_state="preparing",
                                )
                            )
                            prepared = replace(
                                current,
                                generation=current.generation + 1,
                                phase="workspace_prepared",
                                prior_execution_lease_cid=current_cid,
                                workspace_lifecycle_path=str(exact_lifecycle_path),
                                workspace_lifecycle_cid=lifecycle_cid,
                                workspace_record_id=str(
                                    lifecycle_raw.get("record_id") or ""
                                ),
                                workspace_path=str(
                                    lifecycle_raw.get("workspace_path") or ""
                                ),
                                workspace_lease_id=str(
                                    lifecycle_raw.get("lease_id") or ""
                                ),
                                workspace_fence=int(
                                    lifecycle_raw.get("fence") or 0
                                ),
                            )
                            _publish_plan_bound_execution_lease_locked(
                                store,
                                prepared,
                                expected_current_cid=current_cid,
                            )
            return super()._build_implementation_command(
                workspace_path,
                task=task,
                prompt=prompt,
                attempt=attempt,
                state=state,
            )

        def _arm_provider_effect(self, payload: Mapping[str, Any]) -> None:
            current_cid, current = self._load_execution_lease(
                phases=("workspace_prepared",)
            )
            if (
                payload.get("operation") != "implementation_provider"
                or payload.get("task_id") != current.active_task_id
                or payload.get("workspace_path") != current.workspace_path
            ):
                raise PlanBoundDispatchError(
                    "provider mutation payload differs from the execution lease"
                )
            claim_path = Path(current.canonical_claim_path)
            lifecycle_path = Path(current.workspace_lifecycle_path)
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed is None or observed[0] != current_cid:
                        raise PlanBoundDispatchError(
                            "execution lease changed before provider effect"
                        )
                    with serialized_lock_update(claim_path):
                        claim = self._require_bound_claim(current)
                        with serialized_lock_update(lifecycle_path):
                            lifecycle_raw, observed_path, lifecycle_cid = (
                                self._read_exact_worktree_lifecycle(
                                    execution_lease=current,
                                    workspace_path=Path(current.workspace_path),
                                    required_state="active",
                                )
                            )
                            if observed_path != lifecycle_path:
                                raise PlanBoundDispatchError(
                                    "worktree lifecycle path changed before provider"
                                )
                            if (
                                str(lifecycle_raw.get("record_id") or "")
                                != current.workspace_record_id
                                or str(lifecycle_raw.get("lease_id") or "")
                                != current.workspace_lease_id
                                or int(lifecycle_raw.get("fence") or 0)
                                != current.workspace_fence + 1
                            ):
                                raise PlanBoundDispatchError(
                                    "worktree lease or fence changed before provider"
                                )
                            armed = replace(
                                current,
                                generation=current.generation + 1,
                                phase="provider_ready",
                                prior_execution_lease_cid=current_cid,
                                workspace_lifecycle_cid=lifecycle_cid,
                                workspace_record_id=str(
                                    lifecycle_raw.get("record_id") or ""
                                ),
                                workspace_lease_id=str(
                                    lifecycle_raw.get("lease_id") or ""
                                ),
                                workspace_fence=int(
                                    lifecycle_raw.get("fence") or 0
                                ),
                                provider_ready=True,
                            )
                            armed_cid = _publish_plan_bound_execution_lease_locked(
                                store,
                                armed,
                                expected_current_cid=current_cid,
                            )
                            if (
                                content_identity(self._require_bound_claim(armed))
                                != content_identity(claim)
                                or self._read_exact_worktree_lifecycle(
                                    execution_lease=armed,
                                    workspace_path=Path(armed.workspace_path),
                                    required_state="active",
                                )[2]
                                != armed.workspace_lifecycle_cid
                            ):
                                raise PlanBoundDispatchError(
                                    "effect guards changed while arming provider"
                                )
                            final = _load_plan_bound_execution_lease_locked(
                                store,
                                revision_cid=pinned.plan_bound_revision_cid,
                                slice_id=pinned.plan_bound_slice_id,
                                lane_id=pinned.plan_bound_lane_id,
                            )
                            if final is None or final[0] != armed_cid:
                                raise PlanBoundDispatchError(
                                    "provider-ready execution lease did not persist"
                                )

        def _decision_runtime_mutation(
            self,
            boundary: str,
            payload: Mapping[str, Any],
            callback: Any,
        ) -> Any:
            if (
                boundary != "command_invocation"
                or payload.get("operation") != "implementation_provider"
            ):
                return super()._decision_runtime_mutation(
                    boundary,
                    payload,
                    callback,
                )

            def guarded_provider_effect() -> Any:
                self._arm_provider_effect(payload)
                return callback()

            return super()._decision_runtime_mutation(
                boundary,
                payload,
                guarded_provider_effect,
            )

        def _resolved_git_commit(
            self,
            workspace_path: Path,
            value: str,
        ) -> str:
            result = self._run_git(
                ["rev-parse", "--verify", f"{value}^{{commit}}"],
                cwd=workspace_path,
            )
            commit = result.stdout.strip()
            if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise PlanBoundDispatchError(
                    "proposal barrier requires a resolved Git commit"
                )
            return commit

        def _bind_settling_execution_lease(
            self,
        ) -> tuple[str, PlanBoundExecutionLease]:
            """Adopt the daemon's exact ACTIVE->SETTLING lifecycle transition."""

            current_cid, current = self._load_execution_lease(
                phases=("provider_ready",)
            )
            claim_path = Path(current.canonical_claim_path)
            lifecycle_path = Path(current.workspace_lifecycle_path)
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed is None or observed[0] != current_cid:
                        raise PlanBoundDispatchError(
                            "execution lease changed before settling bind"
                        )
                    with serialized_lock_update(claim_path):
                        self._require_bound_claim(current)
                        with serialized_lock_update(lifecycle_path):
                            lifecycle_raw, observed_path, lifecycle_cid = (
                                self._read_exact_worktree_lifecycle(
                                    execution_lease=current,
                                    workspace_path=Path(current.workspace_path),
                                    required_state="settling",
                                )
                            )
                            if (
                                observed_path != lifecycle_path
                                or str(lifecycle_raw.get("record_id") or "")
                                != current.workspace_record_id
                                or str(lifecycle_raw.get("lease_id") or "")
                                != current.workspace_lease_id
                            ):
                                raise PlanBoundDispatchError(
                                    "worktree identity changed while settling"
                                )
                            observed_fence = int(lifecycle_raw.get("fence") or 0)
                            if (
                                observed_fence == current.workspace_fence
                                and lifecycle_cid == current.workspace_lifecycle_cid
                            ):
                                return current_cid, current
                            if observed_fence != current.workspace_fence + 1:
                                raise PlanBoundDispatchError(
                                    "worktree fence skipped before proposal barrier"
                                )
                            settled = replace(
                                current,
                                generation=current.generation + 1,
                                prior_execution_lease_cid=current_cid,
                                workspace_lifecycle_cid=lifecycle_cid,
                                workspace_fence=observed_fence,
                            )
                            settled_cid = _publish_plan_bound_execution_lease_locked(
                                store,
                                settled,
                                expected_current_cid=current_cid,
                            )
                            return settled_cid, settled

        @staticmethod
        def _proposal_barrier_timeout_ms(
            execution_lease: PlanBoundExecutionLease,
        ) -> int:
            assignment = execution_lease.assignment_for(
                execution_lease.active_task_id,
                execution_lease.active_task_cid,
            )
            lease_ms = assignment.get("lease_duration_ms")
            if (
                isinstance(lease_ms, bool)
                or not isinstance(lease_ms, int)
                or not 50 <= lease_ms <= 86_400_000
            ):
                raise PlanBoundDispatchError(
                    "compiled proposal barrier timing is invalid"
                )
            # This is the compiler's identity-bound execution limit, not a
            # heartbeat heuristic.  Verified same-revision transfers may add
            # one CAS-linked lease window per finite owner generation.
            return lease_ms

        def _publish_proposal_disposition_and_wait(
            self,
            *,
            outcome: str,
            baseline_ref: str,
            implementation_commit: str,
            proposal_id: str,
            proposal_receipt_id: str,
            reason_codes: Sequence[str],
            actual_changed_paths: Sequence[str],
            enqueue_fields: Mapping[str, Any] | None = None,
            attempt: int = 0,
            branch_name: str = "",
        ) -> tuple[str, Any]:
            """Publish one final lane result and wait for whole-wave authority."""

            current_cid, current = self._bind_settling_execution_lease()
            phase = (
                "scope_drift"
                if outcome == "rejected" and "path_outside_scope" in reason_codes
                else "proposal_rejected"
                if outcome == "rejected"
                else "proposal_ready"
            )
            claim_path = Path(current.canonical_claim_path)
            lifecycle_path = Path(current.workspace_lifecycle_path)
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed is None or observed != (current_cid, current):
                        raise PlanBoundDispatchError(
                            "execution lease changed before proposal publication"
                        )
                    with serialized_lock_update(claim_path):
                        self._require_bound_claim(current)
                        with serialized_lock_update(lifecycle_path):
                            lifecycle_raw, observed_path, lifecycle_cid = (
                                self._read_exact_worktree_lifecycle(
                                    execution_lease=current,
                                    workspace_path=Path(current.workspace_path),
                                    required_state="settling",
                                )
                            )
                            if (
                                observed_path != lifecycle_path
                                or lifecycle_cid != current.workspace_lifecycle_cid
                                or int(lifecycle_raw.get("fence") or 0)
                                != current.workspace_fence
                                or str(lifecycle_raw.get("lease_id") or "")
                                != current.workspace_lease_id
                            ):
                                raise PlanBoundDispatchError(
                                    "effect guards changed before proposal publication"
                                )
                            proposal_handoff_cid = ""
                            if outcome in {"changed", "no_change"}:
                                if (
                                    not isinstance(enqueue_fields, Mapping)
                                    or not enqueue_fields
                                    or isinstance(attempt, bool)
                                    or not isinstance(attempt, int)
                                    or attempt < 1
                                    or not branch_name
                                ):
                                    raise PlanBoundDispatchError(
                                        "accepted proposal lacks a restart-stable handoff"
                                    )
                                handoff = {
                                    "schema": PLAN_BOUND_PROPOSAL_HANDOFF_SCHEMA,
                                    "revision_cid": current.revision_cid,
                                    "plan_root_cid": current.plan_root_cid,
                                    "execution_plan_cid": current.execution_plan_cid,
                                    "capacity_snapshot_id": current.capacity_snapshot_id,
                                    "slice_manifest_cid": current.slice_manifest_cid,
                                    "slice_id": current.slice_id,
                                    "lane_id": current.lane_id,
                                    "reassignment_cid": current.reassignment_cid,
                                    "task_id": current.active_task_id,
                                    "task_cid": current.active_task_cid,
                                    "source_execution_lease_cid": current_cid,
                                    "process_birth_cid": current.process_birth_cid,
                                    "canonical_claim_cid": current.canonical_claim_cid,
                                    "canonical_claim_lease_id": (
                                        current.canonical_claim_lease_id
                                    ),
                                    "workspace_lifecycle_cid": (
                                        current.workspace_lifecycle_cid
                                    ),
                                    "workspace_record_id": current.workspace_record_id,
                                    "workspace_path": current.workspace_path,
                                    "workspace_lease_id": current.workspace_lease_id,
                                    "workspace_fence": current.workspace_fence,
                                    "attempt": attempt,
                                    "branch_name": branch_name,
                                    "baseline_ref": baseline_ref,
                                    "implementation_commit": implementation_commit,
                                    "actual_changed_paths": list(
                                        actual_changed_paths
                                    ),
                                    "outcome": outcome,
                                    "enqueue_fields": dict(enqueue_fields),
                                    "enqueue_fields_cid": content_identity(
                                        dict(enqueue_fields)
                                    ),
                                    "created_at_ms": int(time.time() * 1000),
                                }
                                proposal_handoff_cid = store.put_cas(handoff)
                                if (
                                    _secure_store_cas(
                                        store,
                                        proposal_handoff_cid,
                                    )
                                    != handoff
                                ):
                                    raise PlanBoundDispatchError(
                                        "proposal handoff failed CAS round trip"
                                    )
                            proposal_ready = replace(
                                current,
                                generation=current.generation + 1,
                                phase=phase,
                                prior_execution_lease_cid=current_cid,
                                proposal_id=proposal_id,
                                proposal_receipt_id=proposal_receipt_id,
                                proposal_reason_codes=tuple(reason_codes),
                                actual_changed_paths=tuple(actual_changed_paths),
                                merge_enqueue_reached=False,
                                proposal_handoff_cid=proposal_handoff_cid,
                            )
                            ready_cid = _publish_plan_bound_execution_lease_locked(
                                store,
                                proposal_ready,
                                expected_current_cid=current_cid,
                            )
                            disposition = PlanBoundProposalDisposition(
                                revision_cid=current.revision_cid,
                                plan_root_cid=current.plan_root_cid,
                                execution_plan_cid=current.execution_plan_cid,
                                capacity_snapshot_id=current.capacity_snapshot_id,
                                slice_manifest_cid=current.slice_manifest_cid,
                                slice_id=current.slice_id,
                                lane_id=current.lane_id,
                                reassignment_cid=current.reassignment_cid,
                                task_id=current.active_task_id,
                                task_cid=current.active_task_cid,
                                execution_lease_cid=ready_cid,
                                process_birth_cid=current.process_birth_cid,
                                proposal_id=proposal_id,
                                proposal_receipt_id=proposal_receipt_id,
                                outcome=outcome,
                                reason_codes=tuple(reason_codes),
                                actual_changed_paths=tuple(actual_changed_paths),
                                baseline_ref=baseline_ref,
                                implementation_commit=implementation_commit,
                            )
                            disposition_cid = (
                                _publish_plan_bound_proposal_disposition_locked(
                                    store,
                                    disposition,
                                )
                            )
            barrier_cid, barrier = ProductionParallelPlanAdapter(
                store
            ).await_wave_diff_barrier(
                revision_cid=current.revision_cid,
                slice_manifest_cid=current.slice_manifest_cid,
                timeout_ms=self._proposal_barrier_timeout_ms(proposal_ready),
            )
            if barrier.decision != "released":
                raise PlanBoundReplanRequired(
                    "whole-wave proposal barrier denied merge admission: "
                    f"barrier_cid={barrier_cid} decision={barrier.decision}"
                )
            # Re-read every authority after the wait and immediately before
            # the caller may enter the existing merge enqueue boundary.
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    exact_barrier = ProductionParallelPlanAdapter(
                        store
                    )._evaluate_wave_diff_barrier_locked(  # noqa: SLF001
                        revision_cid=current.revision_cid,
                        slice_manifest_cid=current.slice_manifest_cid,
                        timeout_ms=self._proposal_barrier_timeout_ms(proposal_ready),
                        now_ms=int(time.time() * 1000),
                    )
                    exact_disposition = (
                        _publish_plan_bound_proposal_disposition_locked(
                            store,
                            disposition,
                        )
                    )
                    exact_execution = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if (
                        exact_barrier is None
                        or exact_barrier[0] != barrier_cid
                        or exact_barrier[1] != barrier
                        or exact_disposition != disposition_cid
                        or exact_execution != (ready_cid, proposal_ready)
                    ):
                        raise PlanBoundDispatchError(
                            "proposal barrier changed before merge admission"
                        )
                    with serialized_lock_update(claim_path):
                        self._require_bound_claim(proposal_ready)
                        with serialized_lock_update(lifecycle_path):
                            final_lifecycle, final_path, final_cid = (
                                self._read_exact_worktree_lifecycle(
                                    execution_lease=proposal_ready,
                                    workspace_path=Path(
                                        proposal_ready.workspace_path
                                    ),
                                    required_state="settling",
                                )
                            )
                            if (
                                final_path != lifecycle_path
                                or final_cid
                                != proposal_ready.workspace_lifecycle_cid
                                or int(final_lifecycle.get("fence") or 0)
                                != proposal_ready.workspace_fence
                                or str(
                                    final_lifecycle.get("lease_id") or ""
                                )
                                != proposal_ready.workspace_lease_id
                            ):
                                raise PlanBoundDispatchError(
                                    "effect guards changed after barrier release"
                                )
            return barrier_cid, barrier

        @staticmethod
        def _name_status_paths(payload: str) -> tuple[str, ...]:
            """Decode every endpoint from one ``git --name-status -z`` result."""

            tokens = payload.split("\0")
            if tokens and tokens[-1] == "":
                tokens.pop()
            paths: list[str] = []
            index = 0
            while index < len(tokens):
                status_code = tokens[index]
                index += 1
                path_count = 2 if status_code[:1] in {"R", "C"} else 1
                if not status_code or index + path_count > len(tokens):
                    raise PlanBoundDispatchError(
                        "full proposal effect enumeration is malformed"
                    )
                for value in tokens[index : index + path_count]:
                    normalized = value.replace("\\", "/").strip("/")
                    if (
                        not normalized
                        or normalized != value
                        or Path(normalized).as_posix() != normalized
                        or "\x00" in normalized
                        or ".." in Path(normalized).parts
                    ):
                        raise PlanBoundDispatchError(
                            "full proposal effect path is unsafe"
                        )
                    paths.append(normalized)
                index += path_count
            return tuple(paths)

        def _full_plan_bound_effect_paths(
            self,
            workspace_path: Path,
            *,
            baseline_ref: str,
        ) -> tuple[str, ...]:
            """Enumerate the unfiltered root and submodule effect endpoints.

            The canonical proposal collector intentionally applies task-owned
            scope policy.  Barrier evidence has the opposite job: observe all
            actual effects before policy or merge, including both rename
            endpoints, untracked files, dirty submodule children, and the
            corresponding root gitlink.
            """

            def repository_paths(
                repository: Path,
                baseline: str,
                *,
                prefix: str = "",
            ) -> tuple[str, ...]:
                changed = self._run_git(
                    [
                        "diff",
                        "--name-status",
                        "-z",
                        "--find-renames",
                        "--ignore-submodules=none",
                        baseline,
                        "--",
                    ],
                    cwd=repository,
                )
                untracked = self._run_git(
                    ["ls-files", "--others", "--exclude-standard", "-z"],
                    cwd=repository,
                )
                if changed.returncode != 0 or untracked.returncode != 0:
                    raise PlanBoundDispatchError(
                        "cannot enumerate the full proposal effect set"
                    )
                values = [
                    *self._name_status_paths(changed.stdout),
                    *tuple(
                        item
                        for item in untracked.stdout.split("\0")
                        if item
                    ),
                ]
                result: list[str] = []
                for value in values:
                    normalized = value.replace("\\", "/").strip("/")
                    if (
                        not normalized
                        or normalized != value
                        or Path(normalized).as_posix() != normalized
                        or "\x00" in normalized
                        or ".." in Path(normalized).parts
                    ):
                        raise PlanBoundDispatchError(
                            "full proposal effect path is unsafe"
                        )
                    result.append(
                        f"{prefix}/{normalized}" if prefix else normalized
                    )
                return tuple(result)

            try:
                workspace_root = workspace_path.resolve(strict=True)
                workspace_stat = os.lstat(workspace_path)
            except (OSError, RuntimeError) as exc:
                raise PlanBoundDispatchError(
                    "cannot bind the proposal workspace for full effect enumeration"
                ) from exc
            if (
                stat.S_ISLNK(workspace_stat.st_mode)
                or not stat.S_ISDIR(workspace_stat.st_mode)
            ):
                raise PlanBoundDispatchError(
                    "proposal workspace is not a lexical directory"
                )
            root_probe = self._run_git(
                ["rev-parse", "--show-toplevel"],
                cwd=workspace_root,
            )
            try:
                observed_root = Path(root_probe.stdout.strip()).resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise PlanBoundDispatchError(
                    "proposal workspace is not an exact Git worktree"
                ) from exc
            if root_probe.returncode != 0 or observed_root != workspace_root:
                raise PlanBoundDispatchError(
                    "proposal workspace is not an exact Git worktree"
                )

            paths = set(repository_paths(workspace_root, baseline_ref))
            for raw_relative in tuple(self.worktree_submodule_paths):
                raw_text = str(raw_relative)
                relative = raw_text.replace("\\", "/").strip("/")
                relative_path = Path(relative)
                if (
                    not relative
                    or raw_text != relative
                    or relative_path.as_posix() != relative
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or "." in relative_path.parts
                ):
                    raise PlanBoundDispatchError(
                        "configured submodule effect path is unsafe"
                    )
                submodule = workspace_root
                try:
                    for part in relative_path.parts:
                        submodule = submodule / part
                        component = os.lstat(submodule)
                        if (
                            stat.S_ISLNK(component.st_mode)
                            or not stat.S_ISDIR(component.st_mode)
                        ):
                            raise PlanBoundDispatchError(
                                "configured submodule traverses a non-directory or symlink"
                            )
                    resolved_submodule = submodule.resolve(strict=True)
                    resolved_submodule.relative_to(workspace_root)
                except PlanBoundDispatchError:
                    raise
                except (OSError, RuntimeError, ValueError) as exc:
                    raise PlanBoundDispatchError(
                        "configured submodule escapes or is absent"
                    ) from exc
                child_root = self._run_git(
                    ["rev-parse", "--show-toplevel"],
                    cwd=resolved_submodule,
                )
                child_inside = self._run_git(
                    ["rev-parse", "--is-inside-work-tree"],
                    cwd=resolved_submodule,
                )
                try:
                    observed_child_root = Path(
                        child_root.stdout.strip()
                    ).resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise PlanBoundDispatchError(
                        "configured submodule is not an exact Git worktree"
                    ) from exc
                if (
                    child_root.returncode != 0
                    or child_inside.returncode != 0
                    or child_inside.stdout.strip() != "true"
                    or observed_child_root != resolved_submodule
                ):
                    raise PlanBoundDispatchError(
                        "configured submodule is not an exact Git worktree"
                    )
                baseline_gitlink = self._run_git(
                    ["rev-parse", "--verify", f"{baseline_ref}:{relative}"],
                    cwd=workspace_root,
                )
                baseline_child = baseline_gitlink.stdout.strip()
                if (
                    baseline_gitlink.returncode != 0
                    or len(baseline_child) != 40
                    or any(character not in "0123456789abcdef" for character in baseline_child)
                ):
                    raise PlanBoundDispatchError(
                        "cannot bind submodule effects to the proposal baseline"
                    )
                child_paths = repository_paths(
                    resolved_submodule,
                    baseline_child,
                    prefix=relative,
                )
                if child_paths:
                    paths.add(relative)
                    paths.update(child_paths)
            return tuple(sorted(paths))

        def _commit_worktree_changes(
            self,
            worktree_path: Path,
            task: Any,
            attempt: int,
            *,
            baseline_ref: str = "",
        ) -> dict[str, Any]:
            """Carry no-change context to the canonical post-commit guard."""

            result = super()._commit_worktree_changes(
                worktree_path,
                task,
                attempt,
                baseline_ref=baseline_ref,
            )
            if result.get("reason") != "no_changes":
                return result
            resolved_baseline = self._resolved_git_commit(
                worktree_path,
                baseline_ref,
            )
            current_head = self._resolved_git_commit(
                worktree_path,
                "HEAD",
            )
            actual_paths = self._full_plan_bound_effect_paths(
                worktree_path,
                baseline_ref=resolved_baseline,
            )
            if current_head != resolved_baseline or actual_paths:
                raise PlanBoundDispatchError(
                    "no-change proposal barrier observed a changed candidate"
                )
            self._plan_bound_pending_no_change = {
                "workspace_path": worktree_path,
                "task": task,
                "attempt": attempt,
                "baseline_ref": resolved_baseline,
            }
            return result

        def _validated_no_change_completion_guard(
            self,
            *,
            baseline_ref: str,
            current_head: str,
            expected_branch: str,
            current_branch: str,
            validation_result: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Publish no-change only after the canonical final guard allows it."""

            result = super()._validated_no_change_completion_guard(
                baseline_ref=baseline_ref,
                current_head=current_head,
                expected_branch=expected_branch,
                current_branch=current_branch,
                validation_result=validation_result,
            )
            pending = getattr(self, "_plan_bound_pending_no_change", None)
            self._plan_bound_pending_no_change = None
            if not isinstance(pending, Mapping):
                raise PlanBoundDispatchError(
                    "no-change guard lacks its exact commit context"
                )
            if not result.get("allowed"):
                return result
            workspace_path = pending.get("workspace_path")
            task = pending.get("task")
            resolved_baseline = str(pending.get("baseline_ref") or "")
            if (
                not isinstance(workspace_path, Path)
                or task is None
                or baseline_ref != resolved_baseline
                or current_head != resolved_baseline
                or self._resolved_git_commit(workspace_path, "HEAD")
                != resolved_baseline
                or self._git_current_branch(workspace_path) != current_branch
            ):
                raise PlanBoundDispatchError(
                    "no-change candidate changed after its canonical guard"
                )
            actual_paths = self._full_plan_bound_effect_paths(
                workspace_path,
                baseline_ref=resolved_baseline,
            )
            if actual_paths:
                raise PlanBoundDispatchError(
                    "no-change guard observed a nonempty final candidate"
                )
            attempt = pending.get("attempt")
            if isinstance(attempt, bool) or not isinstance(attempt, int):
                raise PlanBoundDispatchError(
                    "no-change guard attempt is malformed"
                )
            enqueue_fields = self._capture_plan_bound_enqueue_fields(
                branch_name=current_branch,
                implementation_commit=resolved_baseline,
                baseline_ref=resolved_baseline,
                worktree_path=workspace_path,
                task=task,
                attempt=attempt,
                changed_submodule_paths=(),
                validation_result=validation_result,
            )
            barrier_cid, _barrier = self._publish_proposal_disposition_and_wait(
                outcome="no_change",
                baseline_ref=resolved_baseline,
                implementation_commit=resolved_baseline,
                proposal_id="",
                proposal_receipt_id="",
                reason_codes=(),
                actual_changed_paths=(),
                enqueue_fields=enqueue_fields,
                attempt=attempt,
                branch_name=current_branch,
            )
            _turn_cid, turn_lease = self._load_execution_lease(
                phases=("proposal_ready",)
            )
            self._await_plan_bound_merge_turn(turn_lease)
            if (
                self._resolved_git_commit(workspace_path, "HEAD")
                != resolved_baseline
                or self._git_current_branch(workspace_path) != current_branch
                or self._full_plan_bound_effect_paths(
                    workspace_path,
                    baseline_ref=resolved_baseline,
                )
            ):
                raise PlanBoundDispatchError(
                    "no-change candidate changed after barrier release"
                )
            prepared_cid, prepared = self._prepare_plan_bound_merge_enqueue(
                enqueue_fields=enqueue_fields,
                barrier_cid=barrier_cid,
                worktree_path=workspace_path,
                baseline_ref=resolved_baseline,
                implementation_commit=resolved_baseline,
                actual_paths=(),
                branch_name=current_branch,
                attempt=attempt,
            )
            request = self.merge_queue.enqueue(**enqueue_fields)
            confirmed_cid, confirmed = self._confirm_plan_bound_merge_enqueue(
                prepared_cid=prepared_cid,
                prepared=prepared,
                request=request,
                enqueue_fields=enqueue_fields,
            )
            completed = self._drain_plan_bound_merge_request(
                request_id=str(request.request_id),
                execution_lease=confirmed,
                enqueue_fields=enqueue_fields,
            )
            completed_cid, _completed_lease = (
                self._mark_plan_bound_merge_completed(
                    lease_cid=confirmed_cid,
                    execution_lease=confirmed,
                    request=completed,
                )
            )
            self._plan_bound_no_change_queue_completion = {
                "workspace_path": str(workspace_path),
                "request_id": str(request.request_id),
                "execution_lease_cid": completed_cid,
                "status": completed.status,
            }
            return result

        def _cleanup_merged_worktree(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Deny no-change paths that bypass the canonical final guard."""

            if getattr(self, "_plan_bound_pending_no_change", None) is not None:
                self._plan_bound_pending_no_change = None
                raise PlanBoundReplanRequired(
                    "no-change cleanup bypassed the canonical completion guard"
                )
            no_change_completion = getattr(
                self,
                "_plan_bound_no_change_queue_completion",
                None,
            )
            if isinstance(no_change_completion, Mapping):
                self._plan_bound_no_change_queue_completion = None
                worktree_path = args[0] if args else kwargs.get("worktree_path")
                if (
                    str(worktree_path or "")
                    != no_change_completion.get("workspace_path")
                    or no_change_completion.get("status") != "completed"
                ):
                    raise PlanBoundReplanRequired(
                        "no-change queue completion changed before cleanup"
                    )
                return {
                    "cleaned": True,
                    "reason": "plan_bound_no_change_merge_completed",
                    "worktree_path": str(worktree_path),
                    "request_id": no_change_completion["request_id"],
                    "execution_lease_cid": no_change_completion[
                        "execution_lease_cid"
                    ],
                }
            return super()._cleanup_merged_worktree(*args, **kwargs)

        @staticmethod
        def _canonical_merge_enqueue_fields(
            positional: Sequence[Any],
            keyword: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Capture the complete canonical queue call before its effect."""

            if positional:
                raise PlanBoundDispatchError(
                    "canonical merge enqueue used positional authority"
                )
            allowed = {
                "branch_name",
                "task_id",
                "priority",
                "lane_id",
                "attempt",
                "metadata",
                "commit_sha",
                "canonical_task_id",
                "canonical_task_key",
                "canonical_task_cid",
                "target_repository_id",
                "target_branch",
            }
            if not set(keyword).issubset(allowed):
                raise PlanBoundDispatchError(
                    "canonical merge enqueue fields changed"
                )
            fields = {
                "branch_name": keyword.get("branch_name", ""),
                "task_id": keyword.get("task_id", ""),
                "priority": keyword.get("priority", "P2"),
                "lane_id": keyword.get("lane_id", ""),
                "attempt": keyword.get("attempt", 1),
                "metadata": keyword.get("metadata", {}),
                "commit_sha": keyword.get("commit_sha", ""),
                "canonical_task_id": keyword.get("canonical_task_id", ""),
                "canonical_task_key": keyword.get("canonical_task_key", ""),
                "canonical_task_cid": keyword.get("canonical_task_cid", ""),
                "target_repository_id": keyword.get(
                    "target_repository_id", ""
                ),
                "target_branch": keyword.get("target_branch", ""),
            }
            text_names = allowed - {"attempt", "metadata"}
            if (
                any(not isinstance(fields[name], str) for name in text_names)
                or isinstance(fields["attempt"], bool)
                or not isinstance(fields["attempt"], int)
                or fields["attempt"] < 1
                or not isinstance(fields["metadata"], Mapping)
            ):
                raise PlanBoundDispatchError(
                    "canonical merge enqueue field types changed"
                )
            fields["metadata"] = dict(fields["metadata"])
            return fields

        def _capture_plan_bound_enqueue_fields(
            self,
            *,
            branch_name: str,
            implementation_commit: str,
            baseline_ref: str,
            worktree_path: Path,
            task: Any,
            attempt: int,
            changed_submodule_paths: Sequence[str],
            validation_result: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Use the canonical candidate builder without crossing enqueue."""

            captured: list[dict[str, Any]] = []
            original_enqueue = self.merge_queue.enqueue
            original_record_event = self._record_event

            class CapturedRequest:
                request_id = "plan-bound-prebarrier-capture"

            def capture(*positional: Any, **keyword: Any) -> Any:
                fields = self._canonical_merge_enqueue_fields(
                    positional,
                    keyword,
                )
                captured.append(fields)
                return CapturedRequest()

            def record_event(kind: str, payload: Mapping[str, Any]) -> Any:
                if kind == "merge_candidate_enqueued":
                    return None
                return original_record_event(kind, payload)

            self.merge_queue.enqueue = capture
            self._record_event = record_event
            try:
                super()._enqueue_merge_candidate(
                    branch_name=branch_name,
                    implementation_commit=implementation_commit,
                    baseline_ref=baseline_ref,
                    worktree_path=worktree_path,
                    task=task,
                    attempt=attempt,
                    changed_submodule_paths=changed_submodule_paths,
                    validation_result=dict(validation_result),
                    worktree_pool_handoff=False,
                )
            finally:
                self._record_event = original_record_event
                self.merge_queue.enqueue = original_enqueue
            if len(captured) != 1:
                raise PlanBoundDispatchError(
                    "canonical candidate builder did not yield one enqueue"
                )
            return captured[0]

        @staticmethod
        def _require_queue_request_matches_intent(
            request: Any,
            enqueue_fields: Mapping[str, Any],
        ) -> None:
            """Require a deduplicated queue row to equal the stored intent."""

            canonical_task_id = str(
                enqueue_fields.get("canonical_task_id")
                or enqueue_fields.get("canonical_task_cid")
                or ""
            )
            expected = {
                "branch_name": enqueue_fields["branch_name"],
                "task_id": enqueue_fields["task_id"],
                "priority": enqueue_fields["priority"],
                "lane_id": enqueue_fields["lane_id"],
                "commit_sha": enqueue_fields["commit_sha"],
                "canonical_task_id": canonical_task_id,
                "canonical_task_key": enqueue_fields["canonical_task_key"],
            }
            mismatched_fields = tuple(
                name
                for name, value in expected.items()
                if getattr(request, name, None) != value
            )
            if mismatched_fields:
                raise PlanBoundReplanRequired(
                    "canonical merge queue dedupe row differs from its intent: "
                    + ",".join(mismatched_fields)
                )
            request_metadata = getattr(request, "metadata", None)
            expected_metadata = dict(enqueue_fields["metadata"])
            mutable_queue_metadata = {
                "completion",
                "failure_metadata",
                "deferrals",
                "quarantine",
            }
            if (
                not isinstance(request_metadata, Mapping)
                or any(
                    request_metadata.get(name) != value
                    for name, value in expected_metadata.items()
                )
                or set(request_metadata)
                - set(expected_metadata)
                - mutable_queue_metadata
            ):
                raise PlanBoundReplanRequired(
                    "canonical merge queue metadata differs from its intent"
                )
            # MergeQueue.attempt is mutable retry state, not immutable enqueue
            # authority.  It advances once with each retry failure.  Bind it
            # to the original implementation attempt carried by the intent
            # and the queue's durable failure counter instead of falsely
            # requiring the initial value after another canonical train has
            # already retried the row.
            request_attempt = getattr(request, "attempt", None)
            failure_count = getattr(request, "failure_count", None)
            original_attempt = enqueue_fields["attempt"]
            if (
                isinstance(request_attempt, bool)
                or not isinstance(request_attempt, int)
                or isinstance(failure_count, bool)
                or not isinstance(failure_count, int)
                or failure_count < 0
                or request_attempt < original_attempt
                or request_attempt > original_attempt + failure_count
                or original_attempt + failure_count - request_attempt > 1
            ):
                raise PlanBoundReplanRequired(
                    "canonical merge queue retry state differs from its intent "
                    f"({request_attempt!r}!={original_attempt!r}+"
                    f"{failure_count!r})"
                )
            identity = str(
                expected["canonical_task_key"]
                or expected["canonical_task_id"]
                or expected["task_id"]
            ).strip().casefold()
            commit = str(expected["commit_sha"]).strip().casefold()
            dedupe_parts = [identity, commit]
            target_repository_id = str(
                enqueue_fields["target_repository_id"]
            ).strip()
            target_branch = str(enqueue_fields["target_branch"]).strip()
            if target_repository_id or target_branch:
                if not target_repository_id or not target_branch:
                    raise PlanBoundReplanRequired(
                        "merge queue intent target binding is partial"
                    )
                dedupe_parts.extend((target_repository_id, target_branch))
            expected_dedupe = sha256(
                "\0".join(dedupe_parts).encode("utf-8")
            ).hexdigest()
            if (
                not commit
                or str(getattr(request, "dedupe_key", "")) != expected_dedupe
            ):
                raise PlanBoundReplanRequired(
                    "canonical merge queue dedupe identity changed"
                )

        def _quarantine_merge_request_mismatch(
            self,
            request: Any,
            *,
            prepared: PlanBoundExecutionLease,
            reason: str,
        ) -> None:
            """Fence a mismatched canonical row before another train uses it."""

            status = str(getattr(request, "status", "") or "")
            if status not in {"pending", "processing"}:
                raise PlanBoundReplanRequired(
                    "irreconcilable merge row is already terminal: " + reason
                )
            try:
                receipt_path = self.merge_queue.quarantine(
                    request,
                    reason="plan_bound_merge_enqueue_mismatch",
                    metadata={
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "plan-bound-merge-enqueue-quarantine@1"
                        ),
                        "revision_cid": prepared.revision_cid,
                        "slice_manifest_cid": prepared.slice_manifest_cid,
                        "slice_id": prepared.slice_id,
                        "lane_id": prepared.lane_id,
                        "merge_authorization_cid": (
                            prepared.merge_authorization_cid
                        ),
                        "merge_enqueue_intent_cid": (
                            prepared.merge_enqueue_intent_cid
                        ),
                        "reason": reason,
                    },
                )
                observed = self.merge_queue.get(str(request.request_id))
            except Exception as exc:
                raise PlanBoundReplanRequired(
                    "irreconcilable merge row could not be quarantined"
                ) from exc
            if (
                receipt_path is None
                or observed is None
                or observed.status != "quarantined"
            ):
                raise PlanBoundReplanRequired(
                    "irreconcilable merge row quarantine is not durable"
                )
            self._publish_plan_bound_merge_terminal_failure(
                execution_lease=prepared,
                request=observed,
                reason_codes=(
                    "merge_queue_intent_mismatch",
                    reason,
                ),
            )

        def _publish_plan_bound_merge_terminal_failure(
            self,
            *,
            execution_lease: PlanBoundExecutionLease,
            request: Any,
            reason_codes: Sequence[str],
        ) -> str:
            """Bind one canonical failed/quarantined row to the wave slice."""

            status = str(getattr(request, "status", "") or "")
            if status not in {"failed", "quarantined"}:
                raise PlanBoundReplanRequired(
                    "merge terminal evidence is not a terminal queue row"
                )
            if not hasattr(request, "to_dict"):
                raise PlanBoundReplanRequired(
                    "merge terminal evidence lacks canonical serialization"
                )
            raw_reason_codes = tuple(reason_codes)
            canonical_reasons = sorted(
                {
                    str(value).strip()
                    for value in raw_reason_codes
                    if isinstance(value, str) and value.strip()
                }
            )
            failure_reason = str(
                getattr(request, "failure_reason", "") or ""
            ).strip()
            if failure_reason:
                canonical_reasons.append(failure_reason)
                canonical_reasons = sorted(set(canonical_reasons))
            if not canonical_reasons:
                canonical_reasons = ["canonical_merge_request_failed"]
            queue_payload = request.to_dict()
            if not isinstance(queue_payload, Mapping):
                raise PlanBoundReplanRequired(
                    "merge terminal queue serialization is not an object"
                )
            queue_json = json.dumps(
                dict(queue_payload),
                sort_keys=True,
                separators=(",", ":"),
            )
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    current = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=execution_lease.revision_cid,
                        slice_id=execution_lease.slice_id,
                        lane_id=execution_lease.lane_id,
                    )
                    if current is None or current[1] != execution_lease:
                        raise PlanBoundReplanRequired(
                            "merge terminal outcome lost its execution lease"
                        )
                    intent = _secure_store_cas(
                        store,
                        execution_lease.merge_enqueue_intent_cid,
                    )
                    observed_at_ms = max(
                        int(time.time() * 1000),
                        int(intent["prepared_at_ms"]),
                    )
                    record = {
                        "schema": PLAN_BOUND_MERGE_TERMINAL_FAILURE_SCHEMA,
                        "revision_cid": execution_lease.revision_cid,
                        "plan_root_cid": execution_lease.plan_root_cid,
                        "execution_plan_cid": (
                            execution_lease.execution_plan_cid
                        ),
                        "capacity_snapshot_id": (
                            execution_lease.capacity_snapshot_id
                        ),
                        "slice_manifest_cid": (
                            execution_lease.slice_manifest_cid
                        ),
                        "slice_id": execution_lease.slice_id,
                        "lane_id": execution_lease.lane_id,
                        "reassignment_cid": execution_lease.reassignment_cid,
                        "task_id": execution_lease.active_task_id,
                        "task_cid": execution_lease.active_task_cid,
                        "execution_lease_cid": current[0],
                        "proposal_handoff_cid": (
                            execution_lease.proposal_handoff_cid
                        ),
                        "merge_authorization_cid": (
                            execution_lease.merge_authorization_cid
                        ),
                        "merge_enqueue_intent_cid": (
                            execution_lease.merge_enqueue_intent_cid
                        ),
                        "enqueue_fields_cid": intent["enqueue_fields_cid"],
                        "request_id": str(request.request_id),
                        "queue_status": status,
                        "queue_dedupe_key": str(request.dedupe_key),
                        "queue_request_json": queue_json,
                        "queue_request_sha256": (
                            "sha256:"
                            + sha256(queue_json.encode("utf-8")).hexdigest()
                        ),
                        "reason_codes": canonical_reasons,
                        "observed_at_ms": observed_at_ms,
                    }
                    return _publish_plan_bound_merge_terminal_failure_locked(
                        store,
                        record,
                    )

        def _prepare_plan_bound_merge_enqueue(
            self,
            *,
            enqueue_fields: Mapping[str, Any],
            barrier_cid: str,
            worktree_path: Path,
            baseline_ref: str,
            implementation_commit: str,
            actual_paths: tuple[str, ...],
            branch_name: str,
            attempt: int,
            recovery_birth_cid: str = "",
        ) -> tuple[str, PlanBoundExecutionLease]:
            """Persist exact authorization and queue intent before enqueue."""

            metadata = enqueue_fields.get("metadata")
            task_payload = (
                metadata.get("task") if isinstance(metadata, Mapping) else None
            )
            if (
                enqueue_fields.get("branch_name") != branch_name
                or enqueue_fields.get("task_id") != task_ids[0]
                or enqueue_fields.get("attempt") != attempt
                or enqueue_fields.get("commit_sha") != implementation_commit
                or enqueue_fields.get("canonical_task_id") != task_cids[0]
                or enqueue_fields.get("target_repository_id")
                != self.merge_target_repository_id
                or enqueue_fields.get("target_branch")
                != self.resolved_merge_target_branch
                or not isinstance(metadata, Mapping)
                or metadata.get("baseline_ref") != baseline_ref
                or metadata.get("implementation_commit")
                != implementation_commit
                or not isinstance(task_payload, Mapping)
                or task_payload.get("task_id") != task_ids[0]
            ):
                raise PlanBoundDispatchError(
                    "canonical merge enqueue differs from final authorization"
                )
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    disposition = _load_plan_bound_proposal_disposition_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                    )
                    if (
                        observed is None
                        or observed[1].phase != "proposal_ready"
                        or disposition is None
                        or disposition[1].execution_lease_cid != observed[0]
                        or disposition[1].outcome
                        not in {"changed", "no_change"}
                    ):
                        raise PlanBoundDispatchError(
                            "merge authorization lost its proposal disposition"
                        )
                    current_cid, current = observed
                    recovery_birth: Mapping[str, Any] | None = None
                    if recovery_birth_cid:
                        recovery_birth = _secure_store_cas(
                            store,
                            recovery_birth_cid,
                        )
                        recovery_fields = {
                            "schema",
                            "revision_cid",
                            "slice_manifest_cid",
                            "slice_id",
                            "lane_id",
                            "generation",
                            "execution_lease_cid",
                            "proposal_handoff_cid",
                            "merge_authorization_cid",
                            "merge_enqueue_intent_cid",
                            "supervisor_process_birth_cid",
                            "prior_supervisor_process_birth_cid",
                            "canonical_claim_cid",
                            "canonical_claim_lease_id",
                            "custody_kind",
                            "authorized_workspace_lifecycle_cid",
                            "lifecycle_owner_process_birth",
                            "prior_recovery_daemon_process_birth",
                            "daemon_process_birth",
                            "workspace_lifecycle_path",
                            "workspace_lifecycle_cid",
                            "workspace_lifecycle_json",
                            "prior_recovery_birth_cid",
                            "observed_at_ms",
                        }
                        recovery_generation = recovery_birth.get("generation")
                        if (
                            set(recovery_birth) != recovery_fields
                            or recovery_birth.get("schema")
                            != PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA
                            or isinstance(recovery_generation, bool)
                            or not isinstance(recovery_generation, int)
                            or not 1
                            <= recovery_generation
                            <= MAX_PLAN_BOUND_WAVE_TRANSFERS
                            or recovery_birth.get("revision_cid")
                            != current.revision_cid
                            or recovery_birth.get("slice_id")
                            != current.slice_id
                            or recovery_birth.get("lane_id") != current.lane_id
                            or recovery_birth.get("execution_lease_cid")
                            != current_cid
                            or recovery_birth.get("proposal_handoff_cid")
                            != current.proposal_handoff_cid
                            or recovery_birth.get("canonical_claim_cid")
                            != current.canonical_claim_cid
                            or recovery_birth.get("canonical_claim_lease_id")
                            != current.canonical_claim_lease_id
                            or recovery_birth.get("custody_kind")
                            != "settling_candidate"
                            or recovery_birth.get(
                                "authorized_workspace_lifecycle_cid"
                            )
                            != current.workspace_lifecycle_cid
                            or recovery_birth.get("workspace_lifecycle_path")
                            != current.workspace_lifecycle_path
                            or recovery_birth.get("workspace_lifecycle_cid")
                            != current.workspace_lifecycle_cid
                            or recovery_birth.get("daemon_process_birth")
                            != self._current_daemon_birth()
                        ):
                            raise PlanBoundDispatchError(
                                "proposal recovery birth is mixed"
                            )
                    proposal_handoff = _secure_store_cas(
                        store,
                        current.proposal_handoff_cid,
                    )
                    if (
                        proposal_handoff.get("outcome")
                        != disposition[1].outcome
                        or proposal_handoff.get("enqueue_fields")
                        != dict(enqueue_fields)
                        or proposal_handoff.get("branch_name") != branch_name
                        or proposal_handoff.get("baseline_ref") != baseline_ref
                        or proposal_handoff.get("implementation_commit")
                        != implementation_commit
                        or proposal_handoff.get("actual_changed_paths")
                        != list(actual_paths)
                    ):
                        raise PlanBoundDispatchError(
                            "proposal handoff changed before merge authorization"
                        )
                    barrier_payload = _secure_store_cas(store, barrier_cid)
                    if (
                        barrier_payload.get("revision_cid")
                        != current.revision_cid
                        or barrier_payload.get("slice_manifest_cid")
                        != current.slice_manifest_cid
                        or barrier_payload.get("decision") != "released"
                    ):
                        raise PlanBoundDispatchError(
                            "merge authorization lost its released wave barrier"
                        )
                    claim_path = Path(current.canonical_claim_path)
                    lifecycle_path = Path(current.workspace_lifecycle_path)
                    with serialized_lock_update(claim_path):
                        if recovery_birth is None:
                            self._require_bound_claim(current)
                        else:
                            recovery_claim = stable_effect_json(claim_path)
                            if (
                                content_identity(recovery_claim)
                                != current.canonical_claim_cid
                                or recovery_claim.get("lease_id")
                                != current.canonical_claim_lease_id
                                or recovery_claim.get("task_id")
                                != current.active_task_id
                                or recovery_claim.get("canonical_task_cid")
                                != current.active_task_cid
                                or recovery_claim.get("pid")
                                != current.daemon_process_birth.get("pid")
                            ):
                                raise PlanBoundDispatchError(
                                    "proposal recovery claim authority changed"
                                )
                        with serialized_lock_update(lifecycle_path):
                            if recovery_birth is None:
                                lifecycle_raw, observed_path, lifecycle_cid = (
                                    self._read_exact_worktree_lifecycle(
                                        execution_lease=current,
                                        workspace_path=worktree_path,
                                        required_state="settling",
                                    )
                                )
                            else:
                                from ..merge.worktree_lifecycle import (
                                    WorkspaceLifecycleRecord,
                                )

                                lifecycle_raw, lifecycle_cid = (
                                    stable_effect_record(lifecycle_path)
                                )
                                lifecycle = WorkspaceLifecycleRecord.from_dict(
                                    lifecycle_raw
                                )
                                observed_path = (
                                    self.worktree_lifecycle.workspace_path_for(
                                        worktree_path
                                    )
                                )
                                if (
                                    lifecycle.state.value != "settling"
                                    or lifecycle.owner.to_dict()
                                    != current.daemon_process_birth
                                    or lifecycle.workspace_path
                                    != str(worktree_path)
                                    or recovery_birth.get(
                                        "workspace_lifecycle_json"
                                    )
                                    != (
                                        json.dumps(
                                            lifecycle_raw,
                                            indent=2,
                                            sort_keys=True,
                                        )
                                        + "\n"
                                    )
                                    or recovery_birth.get(
                                        "workspace_lifecycle_cid"
                                    )
                                    != lifecycle_cid
                                ):
                                    raise PlanBoundDispatchError(
                                        "proposal recovery lifecycle is not an "
                                        "exact dead-owner settling record"
                                    )
                            final_head = self._resolved_git_commit(
                                worktree_path,
                                "HEAD",
                            )
                            final_paths = self._full_plan_bound_effect_paths(
                                worktree_path,
                                baseline_ref=baseline_ref,
                            )
                            if (
                                observed_path != lifecycle_path
                                or lifecycle_cid
                                != current.workspace_lifecycle_cid
                                or int(lifecycle_raw.get("fence") or 0)
                                != current.workspace_fence
                                or str(lifecycle_raw.get("lease_id") or "")
                                != current.workspace_lease_id
                                or final_head != implementation_commit
                                or final_paths != actual_paths
                            ):
                                raise PlanBoundDispatchError(
                                    "candidate or effect guards changed at merge enqueue"
                                )
                            authorized_at_ms = int(time.time() * 1000)
                            authorization = {
                                "schema": PLAN_BOUND_MERGE_AUTHORIZATION_SCHEMA,
                                "revision_cid": current.revision_cid,
                                "plan_root_cid": current.plan_root_cid,
                                "execution_plan_cid": current.execution_plan_cid,
                                "capacity_snapshot_id": current.capacity_snapshot_id,
                                "slice_manifest_cid": current.slice_manifest_cid,
                                "slice_id": current.slice_id,
                                "lane_id": current.lane_id,
                                "reassignment_cid": current.reassignment_cid,
                                "task_id": current.active_task_id,
                                "task_cid": current.active_task_cid,
                                "process_birth_cid": current.process_birth_cid,
                                "execution_lease_cid": current_cid,
                                "proposal_handoff_cid": (
                                    current.proposal_handoff_cid
                                ),
                                "recovery_birth_cid": recovery_birth_cid,
                                "disposition_cid": disposition[0],
                                "barrier_cid": barrier_cid,
                                "outcome": disposition[1].outcome,
                                "canonical_claim_cid": current.canonical_claim_cid,
                                "workspace_lifecycle_cid": (
                                    current.workspace_lifecycle_cid
                                ),
                                "workspace_fence": current.workspace_fence,
                                "workspace_lease_id": current.workspace_lease_id,
                                "workspace_path": current.workspace_path,
                                "attempt": attempt,
                                "branch_name": branch_name,
                                "baseline_ref": baseline_ref,
                                "implementation_commit": implementation_commit,
                                "actual_changed_paths": list(actual_paths),
                                "authorized_at_ms": authorized_at_ms,
                            }
                            authorization_cid = store.put_cas(authorization)
                            if _secure_store_cas(store, authorization_cid) != authorization:
                                raise PlanBoundDispatchError(
                                    "merge authorization failed CAS round trip"
                                )
                            enqueue_fields_dict = dict(enqueue_fields)
                            intent = {
                                "schema": PLAN_BOUND_MERGE_ENQUEUE_INTENT_SCHEMA,
                                "authorization_cid": authorization_cid,
                                "enqueue_fields": enqueue_fields_dict,
                                "enqueue_fields_cid": content_identity(
                                    enqueue_fields_dict
                                ),
                                "prepared_at_ms": authorized_at_ms,
                            }
                            intent_cid = store.put_cas(intent)
                            if _secure_store_cas(store, intent_cid) != intent:
                                raise PlanBoundDispatchError(
                                    "merge enqueue intent failed CAS round trip"
                                )
                            prepared = replace(
                                current,
                                generation=current.generation + 1,
                                phase="merge_enqueue_prepared",
                                prior_execution_lease_cid=current_cid,
                                merge_enqueue_reached=True,
                                merge_authorization_cid=authorization_cid,
                                merge_enqueue_intent_cid=intent_cid,
                            )
                            prepared_cid = (
                                _publish_plan_bound_execution_lease_locked(
                                    store,
                                    prepared,
                                    expected_current_cid=current_cid,
                                )
                            )
                            return prepared_cid, prepared

        def _await_plan_bound_merge_turn(
            self,
            execution_lease: PlanBoundExecutionLease,
        ) -> None:
            """Serialize canonical queue handoff in immutable manifest order.

            Provider work and proposal validation remain parallel.  The merge
            train is globally serialized, but it can otherwise dequeue a dead
            sibling's newly committed row before that sibling's CAS-bound
            recovery child adopts its SETTLING lifecycle.  Requiring every
            earlier manifest slice to reach authoritative ``merge_completed``
            keeps the canonical train on the row whose exact owner is alive;
            a crashed predecessor is restarted by the outer runner while later
            lanes wait outside every store/claim/lifecycle lock.
            """

            timeout_ms = self._proposal_barrier_timeout_ms(execution_lease)
            deadline = time.monotonic() + timeout_ms / 1000.0
            adapter = ProductionParallelPlanAdapter(store)
            while True:
                pending: list[str] = []
                with store._thread_lock:  # noqa: SLF001
                    with store._guard():  # noqa: SLF001
                        manifest_payload = _secure_store_cas(
                            store,
                            execution_lease.slice_manifest_cid,
                        )
                        manifest = ConfiguredBoardExecutionSlices.from_dict(
                            manifest_payload
                        )
                        ordered = tuple(
                            sorted(
                                manifest.nonempty,
                                key=lambda item: (item.lane_index, item.slice_id),
                            )
                        )
                        terminal_failures = tuple(
                            item.slice_id
                            for item in ordered
                            if _load_plan_bound_merge_terminal_failure_locked(
                                store,
                                revision_cid=execution_lease.revision_cid,
                                slice_id=item.slice_id,
                            )
                            is not None
                        )
                        if terminal_failures:
                            raise PlanBoundReplanRequired(
                                "wave contains terminal canonical merge failure: "
                                f"{sorted(terminal_failures)!r}"
                            )
                        current_indexes = tuple(
                            index
                            for index, item in enumerate(ordered)
                            if item.slice_id == execution_lease.slice_id
                        )
                        if len(current_indexes) != 1:
                            raise PlanBoundReplanRequired(
                                "merge turn lost its immutable manifest slice"
                            )
                        for prior_slice in ordered[: current_indexes[0]]:
                            reassignment = adapter._load_slice_reassignment_locked(  # noqa: SLF001
                                revision_cid=execution_lease.revision_cid,
                                slice_id=prior_slice.slice_id,
                            )
                            owner_lane = (
                                prior_slice.lane_id
                                if reassignment is None
                                else reassignment[1].recipient_lane_id
                            )
                            prior = _load_plan_bound_execution_lease_locked(
                                store,
                                revision_cid=execution_lease.revision_cid,
                                slice_id=prior_slice.slice_id,
                                lane_id=owner_lane,
                            )
                            if prior is None or prior[1].phase != "merge_completed":
                                pending.append(prior_slice.slice_id)
                if not pending:
                    return
                if time.monotonic() >= deadline:
                    raise PlanBoundReplanRequired(
                        "canonical merge turn exceeded its compiled execution "
                        f"bound waiting for slices {sorted(pending)!r}"
                    )
                time.sleep(0.05)

        def _confirm_plan_bound_merge_enqueue(
            self,
            *,
            prepared_cid: str,
            prepared: PlanBoundExecutionLease,
            request: Any,
            enqueue_fields: Mapping[str, Any],
        ) -> tuple[str, PlanBoundExecutionLease]:
            """Persist the exact canonical queue receipt after enqueue."""

            try:
                self._require_queue_request_matches_intent(
                    request,
                    enqueue_fields,
                )
            except PlanBoundReplanRequired as exc:
                self._quarantine_merge_request_mismatch(
                    request,
                    prepared=prepared,
                    reason=f"enqueue_return_differs_from_intent:{exc}",
                )
                raise
            durable_request = self.merge_queue.get(str(request.request_id))
            if durable_request is None:
                raise PlanBoundReplanRequired(
                    "canonical merge queue receipt is absent after enqueue"
                )
            try:
                self._require_queue_request_matches_intent(
                    durable_request,
                    enqueue_fields,
                )
            except PlanBoundReplanRequired as exc:
                self._quarantine_merge_request_mismatch(
                    durable_request,
                    prepared=prepared,
                    reason=f"durable_queue_row_differs_from_intent:{exc}",
                )
                raise
            if durable_request.status not in {
                "pending",
                "processing",
                "completed",
            }:
                raise PlanBoundReplanRequired(
                    "canonical merge queue receipt is terminally unusable"
                )
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    observed = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
                    if observed != (prepared_cid, prepared):
                        if (
                            observed is not None
                            and observed[1].phase == "merge_enqueue_confirmed"
                            and observed[1].merge_request_id
                            == str(durable_request.request_id)
                        ):
                            return observed
                        raise PlanBoundReplanRequired(
                            "merge enqueue confirmation lost its execution CAS"
                        )
                    intent = _secure_store_cas(
                        store,
                        prepared.merge_enqueue_intent_cid,
                    )
                    confirmed_at_ms = max(
                        int(time.time() * 1000),
                        int(intent["prepared_at_ms"]),
                    )
                    receipt = {
                        "schema": PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA,
                        "authorization_cid": prepared.merge_authorization_cid,
                        "intent_cid": prepared.merge_enqueue_intent_cid,
                        "enqueue_fields_cid": intent["enqueue_fields_cid"],
                        "request_id": str(durable_request.request_id),
                        "dedupe_key": str(durable_request.dedupe_key),
                        "observed_status": str(durable_request.status),
                        "confirmed_at_ms": confirmed_at_ms,
                    }
                    receipt_cid = store.put_cas(receipt)
                    if _secure_store_cas(store, receipt_cid) != receipt:
                        raise PlanBoundReplanRequired(
                            "merge queue receipt failed CAS round trip"
                        )
                    confirmed = replace(
                        prepared,
                        generation=prepared.generation + 1,
                        phase="merge_enqueue_confirmed",
                        prior_execution_lease_cid=prepared_cid,
                        merge_request_id=str(durable_request.request_id),
                        merge_queue_receipt_cid=receipt_cid,
                    )
                    confirmed_cid = _publish_plan_bound_execution_lease_locked(
                        store,
                        confirmed,
                        expected_current_cid=prepared_cid,
                    )
                    return confirmed_cid, confirmed

        def _bind_current_merge_recovery_birth_locked(
            self,
            *,
            execution_lease_cid: str,
            execution_lease: PlanBoundExecutionLease,
        ) -> str:
            """CAS-bind merge-only cleanup custody without replaying provider work.

            The canonical lifecycle store has no process-identity adoption API.
            Its cleanup capability is the exact lease/fence pair, so a recovery
            child may use that pair only after this canonical-store record proves
            the prior recovery process dead, the original claimant dead, and the
            claim plus SETTLING lifecycle byte-identical under store -> claim ->
            lifecycle guards.  The lifecycle itself is not rewritten here; the
            existing canonical merge cleanup consumes its fence exactly once.
            """

            from ..control.lifecycle_orchestrator import (
                ProcessIdentity as SupervisorProcessIdentity,
            )
            from ..merge.worktree_lifecycle import (
                OwnerLiveness as WorktreeOwnerLiveness,
            )
            from ..merge.worktree_lifecycle import (
                ProcessBirthIdentity as WorktreeProcessBirthIdentity,
            )
            from ..merge.worktree_lifecycle import (
                WorkspaceLifecycleRecord,
                current_process_birth,
                owner_liveness,
            )
            from ..runtime.multi_supervisor_runner import (
                LifecycleProfile,
                _strict_plan_bound_process_fence_observation,
            )

            birth_binding = _load_plan_bound_process_birth_chain_locked(
                store,
                revision_cid=pinned.plan_bound_revision_cid,
                slice_id=pinned.plan_bound_slice_id,
                lane_id=pinned.plan_bound_lane_id,
            )
            if birth_binding is None:
                raise PlanBoundReplanRequired(
                    "merge recovery lacks a current accepted process birth"
                )
            current_birth_cid, typed_birth, birth_chain = birth_binding
            birth = typed_birth.to_dict()
            birth_chain_by_cid = dict(birth_chain)

            def dead_supervisor_birth_chain_reaches(
                start_cid: object,
                expected_cid: str,
            ) -> bool:
                """Accept only bounded, exact, effectless supervisor restarts."""

                current_cid = start_cid
                seen: set[str] = set()
                if expected_cid not in birth_chain_by_cid:
                    return False
                for _generation in range(MAX_PLAN_BOUND_WAVE_TRANSFERS + 1):
                    if current_cid == expected_cid:
                        return True
                    if (
                        not isinstance(current_cid, str)
                        or not current_cid
                        or current_cid in seen
                    ):
                        return False
                    seen.add(current_cid)
                    try:
                        intermediate = birth_chain_by_cid.get(current_cid)
                        if intermediate is None:
                            return False
                        intermediate_profile = LifecycleProfile.from_dict(
                            intermediate.profile
                        )
                        intermediate_birth = (
                            SupervisorProcessIdentity.from_dict(
                                intermediate.process_birth
                            )
                        )
                        intermediate_state, intermediate_tree = (
                            _strict_plan_bound_process_fence_observation(
                                intermediate_profile,
                                intermediate_birth,
                            )
                        )
                    except Exception:
                        return False
                    if (
                        intermediate_state != "dead"
                        or intermediate_tree is None
                        or intermediate_tree.members
                    ):
                        return False
                    current_cid = intermediate.prior_process_birth_cid
                return current_cid == expected_cid

            if (
                birth.get("revision_cid") != pinned.plan_bound_revision_cid
                or birth.get("slice_manifest_cid")
                != pinned.plan_bound_slice_manifest_cid
                or birth.get("slice_id") != pinned.plan_bound_slice_id
                or birth.get("lane_id") != pinned.plan_bound_lane_id
                or birth.get("task_ids") != list(task_ids)
                or birth.get("task_cids") != list(task_cids)
            ):
                raise PlanBoundReplanRequired(
                    "merge recovery process birth is mixed"
                )
            try:
                supervisor_profile = LifecycleProfile.from_dict(
                    birth["profile"]
                )
                supervisor_birth = SupervisorProcessIdentity.from_dict(
                    birth["process_birth"]
                )
                original_daemon_birth = (
                    WorktreeProcessBirthIdentity.from_dict(
                        execution_lease.daemon_process_birth
                    )
                )
            except Exception as exc:
                raise PlanBoundReplanRequired(
                    "merge recovery supervisor birth is malformed"
                ) from exc
            daemon_birth = current_process_birth()
            recovery_key = (
                "plan-bound-merge-recovery-birth:"
                f"{pinned.plan_bound_revision_cid}:"
                f"{pinned.plan_bound_slice_id}:"
                f"{pinned.plan_bound_lane_id}"
            )
            previous = _secure_store_continuation(store, recovery_key)
            prior_recovery_birth_cid = ""
            prior_recovery: Mapping[str, Any] | None = None
            prior_recovery_daemon = original_daemon_birth
            prior_supervisor_process_birth_cid = execution_lease.process_birth_cid
            recovery_generation = 1
            custody_kind = (
                "canonical_queue"
                if execution_lease.phase == "merge_enqueue_confirmed"
                else "settling_candidate"
            )
            expected_recovery_fields = {
                "schema",
                "revision_cid",
                "slice_manifest_cid",
                "slice_id",
                "lane_id",
                "generation",
                "execution_lease_cid",
                "proposal_handoff_cid",
                "merge_authorization_cid",
                "merge_enqueue_intent_cid",
                "supervisor_process_birth_cid",
                "prior_supervisor_process_birth_cid",
                "canonical_claim_cid",
                "canonical_claim_lease_id",
                "custody_kind",
                "authorized_workspace_lifecycle_cid",
                "lifecycle_owner_process_birth",
                "prior_recovery_daemon_process_birth",
                "daemon_process_birth",
                "workspace_lifecycle_path",
                "workspace_lifecycle_cid",
                "workspace_lifecycle_json",
                "prior_recovery_birth_cid",
                "observed_at_ms",
            }
            if previous is not None:
                if set(previous) != {
                    "phase",
                    "operation",
                    "revision_cid",
                    "slice_id",
                    "lane_id",
                    "recovery_birth_cid",
                } or (
                    previous.get("phase") != "committed"
                    or previous.get("operation")
                    != "plan_bound_merge_recovery_birth"
                    or previous.get("revision_cid")
                    != pinned.plan_bound_revision_cid
                    or previous.get("slice_id")
                    != pinned.plan_bound_slice_id
                    or previous.get("lane_id")
                    != pinned.plan_bound_lane_id
                ):
                    raise PlanBoundReplanRequired(
                        "merge recovery birth pointer is malformed"
                    )
                prior_recovery_birth_cid = str(
                    previous.get("recovery_birth_cid") or ""
                )
                prior = _secure_store_cas(store, prior_recovery_birth_cid)
                if (
                    set(prior) != expected_recovery_fields
                    or prior.get("schema")
                    != PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA
                    or prior.get("revision_cid")
                    != pinned.plan_bound_revision_cid
                    or prior.get("slice_manifest_cid")
                    != pinned.plan_bound_slice_manifest_cid
                    or prior.get("slice_id") != pinned.plan_bound_slice_id
                    or prior.get("lane_id") != pinned.plan_bound_lane_id
                    or prior.get("proposal_handoff_cid")
                    != execution_lease.proposal_handoff_cid
                    or prior.get("canonical_claim_cid")
                    != execution_lease.canonical_claim_cid
                    or prior.get("canonical_claim_lease_id")
                    != execution_lease.canonical_claim_lease_id
                    or prior.get("custody_kind")
                    not in {"settling_candidate", "canonical_queue"}
                    or prior.get("authorized_workspace_lifecycle_cid")
                    != execution_lease.workspace_lifecycle_cid
                    or prior.get("workspace_lifecycle_path")
                    != execution_lease.workspace_lifecycle_path
                    or not isinstance(prior.get("workspace_lifecycle_json"), str)
                    or not isinstance(prior.get("daemon_process_birth"), Mapping)
                    or (
                        prior.get("custody_kind") == "settling_candidate"
                        and (
                            prior.get("workspace_lifecycle_cid")
                            != execution_lease.workspace_lifecycle_cid
                            or not prior.get("workspace_lifecycle_json")
                        )
                    )
                    or (
                        prior.get("custody_kind") == "canonical_queue"
                        and (
                            prior.get("workspace_lifecycle_cid") != ""
                            or prior.get("workspace_lifecycle_json") != ""
                        )
                    )
                ):
                    raise PlanBoundReplanRequired(
                        "prior merge recovery birth is malformed or mixed"
                    )
                prior_generation = prior.get("generation")
                if (
                    isinstance(prior_generation, bool)
                    or not isinstance(prior_generation, int)
                    or not 1 <= prior_generation < MAX_PLAN_BOUND_WAVE_TRANSFERS
                ):
                    raise PlanBoundReplanRequired(
                        "prior merge recovery generation exhausted its bound"
                    )
                recovery_generation = prior_generation + 1
                prior_recovery = prior
                prior_recovery_daemon = WorktreeProcessBirthIdentity.from_dict(
                    prior["daemon_process_birth"]
                )
                prior_supervisor_process_birth_cid = str(
                    prior["supervisor_process_birth_cid"]
                )

            supervisor_state, _tree = (
                _strict_plan_bound_process_fence_observation(
                    supervisor_profile,
                    supervisor_birth,
                )
            )
            same_recovery_process = bool(
                prior_recovery is not None
                and prior_recovery_daemon.to_dict() == daemon_birth.to_dict()
                and prior_supervisor_process_birth_cid
                == current_birth_cid
                and prior_recovery.get("execution_lease_cid")
                == execution_lease_cid
                and prior_recovery.get("custody_kind") == custody_kind
            )
            dead_supervisor_predecessors = (
                same_recovery_process
                or dead_supervisor_birth_chain_reaches(
                    birth.get("prior_process_birth_cid"),
                    prior_supervisor_process_birth_cid,
                )
            )
            if (
                supervisor_state != "alive"
                or supervisor_birth.to_dict() != birth["process_birth"]
                or (
                    not same_recovery_process
                    and not dead_supervisor_predecessors
                )
                or daemon_birth.pid != os.getpid()
                or daemon_birth.parent_pid != supervisor_birth.pid
                or daemon_birth.start_time_ticks <= 0
                or (
                    not same_recovery_process
                    and owner_liveness(prior_recovery_daemon)
                    is not WorktreeOwnerLiveness.DEAD
                )
                or owner_liveness(original_daemon_birth)
                is not WorktreeOwnerLiveness.DEAD
            ):
                raise PlanBoundReplanRequired(
                    "merge recovery lacks an exact dead predecessor and live "
                    "gated process tree"
                )

            lifecycle_path = Path(execution_lease.workspace_lifecycle_path)
            lifecycle: WorkspaceLifecycleRecord | None = None
            lifecycle_cid = ""
            lifecycle_json = ""
            if custody_kind == "settling_candidate":
                claim_path = Path(execution_lease.canonical_claim_path)
                with serialized_lock_update(claim_path):
                    claim = stable_effect_json(claim_path)
                    if (
                        content_identity(claim)
                        != execution_lease.canonical_claim_cid
                        or claim.get("lease_id")
                        != execution_lease.canonical_claim_lease_id
                        or claim.get("task_id")
                        != execution_lease.active_task_id
                        or claim.get("canonical_task_cid")
                        != execution_lease.active_task_cid
                        or claim.get("pid") != original_daemon_birth.pid
                    ):
                        raise PlanBoundReplanRequired(
                            "merge recovery canonical task claim changed"
                        )
                    with serialized_lock_update(lifecycle_path):
                        lifecycle_raw, lifecycle_cid = stable_effect_record(
                            lifecycle_path
                        )
                        lifecycle_json = (
                            json.dumps(
                                lifecycle_raw,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        try:
                            lifecycle = WorkspaceLifecycleRecord.from_dict(
                                lifecycle_raw
                            )
                        except Exception as exc:
                            raise PlanBoundReplanRequired(
                                "merge recovery lifecycle is malformed"
                            ) from exc
                        if (
                            lifecycle.to_dict() != lifecycle_raw
                            or lifecycle.state.value != "settling"
                            or lifecycle.task_id
                            != execution_lease.active_task_id
                            or lifecycle.canonical_task_cid
                            != execution_lease.active_task_cid
                            or lifecycle.owner.to_dict()
                            != original_daemon_birth.to_dict()
                            or lifecycle_path
                            != self.worktree_lifecycle.workspace_path_for(
                                execution_lease.workspace_path
                            )
                            or lifecycle.workspace_path
                            != execution_lease.workspace_path
                            or lifecycle.record_id
                            != execution_lease.workspace_record_id
                            or lifecycle.lease_id
                            != execution_lease.workspace_lease_id
                            or lifecycle.fence
                            != execution_lease.workspace_fence
                            or lifecycle_cid
                            != execution_lease.workspace_lifecycle_cid
                            or lifecycle.state_dir
                            != str(
                                self.state_path.parent.resolve(strict=False)
                            )
                            or lifecycle.repo_root
                            != str(self.repo_root.resolve(strict=False))
                        ):
                            raise PlanBoundReplanRequired(
                                "merge recovery lifecycle changed before custody"
                            )

            if same_recovery_process:
                assert prior_recovery is not None
                if (
                    prior_recovery.get("workspace_lifecycle_json")
                    != lifecycle_json
                ):
                    raise PlanBoundReplanRequired(
                        "current merge recovery lifecycle evidence changed"
                    )
                if lifecycle is not None:
                    self._active_worktree_lifecycle = lifecycle
                return prior_recovery_birth_cid

            observed_at_ms = int(time.time() * 1000)
            recovery = {
                "schema": PLAN_BOUND_MERGE_RECOVERY_BIRTH_SCHEMA,
                "revision_cid": pinned.plan_bound_revision_cid,
                "slice_manifest_cid": pinned.plan_bound_slice_manifest_cid,
                "slice_id": pinned.plan_bound_slice_id,
                "lane_id": pinned.plan_bound_lane_id,
                "generation": recovery_generation,
                "execution_lease_cid": execution_lease_cid,
                "proposal_handoff_cid": execution_lease.proposal_handoff_cid,
                "merge_authorization_cid": (
                    execution_lease.merge_authorization_cid
                ),
                "merge_enqueue_intent_cid": (
                    execution_lease.merge_enqueue_intent_cid
                ),
                "supervisor_process_birth_cid": current_birth_cid,
                "prior_supervisor_process_birth_cid": (
                    prior_supervisor_process_birth_cid
                ),
                "canonical_claim_cid": execution_lease.canonical_claim_cid,
                "canonical_claim_lease_id": (
                    execution_lease.canonical_claim_lease_id
                ),
                "custody_kind": custody_kind,
                "authorized_workspace_lifecycle_cid": (
                    execution_lease.workspace_lifecycle_cid
                ),
                "lifecycle_owner_process_birth": (
                    original_daemon_birth.to_dict()
                ),
                "prior_recovery_daemon_process_birth": (
                    prior_recovery_daemon.to_dict()
                ),
                "daemon_process_birth": daemon_birth.to_dict(),
                "workspace_lifecycle_path": str(lifecycle_path),
                "workspace_lifecycle_cid": lifecycle_cid,
                "workspace_lifecycle_json": lifecycle_json,
                "prior_recovery_birth_cid": prior_recovery_birth_cid,
                "observed_at_ms": observed_at_ms,
            }
            recovery_birth_cid = store.put_cas(recovery)
            if _secure_store_cas(store, recovery_birth_cid) != recovery:
                raise PlanBoundReplanRequired(
                    "merge recovery birth failed CAS round trip"
                )
            continuation = {
                "phase": "committed",
                "operation": "plan_bound_merge_recovery_birth",
                "revision_cid": pinned.plan_bound_revision_cid,
                "slice_id": pinned.plan_bound_slice_id,
                "lane_id": pinned.plan_bound_lane_id,
                "recovery_birth_cid": recovery_birth_cid,
            }
            store.put_continuation(recovery_key, continuation)
            if _secure_store_continuation(store, recovery_key) != continuation:
                raise PlanBoundReplanRequired(
                    "merge recovery birth pointer failed durable round trip"
                )
            # Only the process named by the durable recovery record receives
            # the canonical lifecycle capability in memory.  No provider or
            # task-claim authority is transferred.
            if lifecycle is not None:
                self._active_worktree_lifecycle = lifecycle
            return recovery_birth_cid

        def _drain_plan_bound_merge_request(
            self,
            *,
            request_id: str,
            execution_lease: PlanBoundExecutionLease,
            enqueue_fields: Mapping[str, Any],
        ) -> Any:
            """Drive one confirmed request through the canonical merge train.

            Whole-wave release allows all disjoint lanes to enqueue together.
            A train lease may make an individual lane's first opportunistic
            consume defer, so that lane must remain alive (or be restart-
            adoptable) until its own durable row is terminal.  Every attempt
            below delegates to the existing serialized train, which performs
            the target rebase, validation, and completion callback.
            """

            timeout_ms = self._proposal_barrier_timeout_ms(execution_lease)
            deadline = time.monotonic() + timeout_ms / 1000.0
            last_exception: Exception | None = None
            last_result: Mapping[str, Any] = {}
            while True:
                request = self.merge_queue.get(request_id)
                if request is None:
                    raise PlanBoundReplanRequired(
                        "confirmed canonical merge request disappeared"
                    )
                try:
                    self._require_queue_request_matches_intent(
                        request,
                        enqueue_fields,
                    )
                except PlanBoundReplanRequired as exc:
                    self._quarantine_merge_request_mismatch(
                        request,
                        prepared=execution_lease,
                        reason=f"merge_drain_row_differs_from_intent:{exc}",
                    )
                    raise
                if request.status == "completed":
                    return request
                if request.status not in {"pending", "processing"}:
                    result_reason = str(
                        last_result.get("reason")
                        or (
                            last_result.get("merge_result", {}).get("reason")
                            if isinstance(
                                last_result.get("merge_result"), Mapping
                            )
                            else ""
                        )
                        or getattr(request, "failure_reason", "")
                        or request.status
                    )
                    self._publish_plan_bound_merge_terminal_failure(
                        execution_lease=execution_lease,
                        request=request,
                        reason_codes=(
                            "canonical_merge_request_terminal",
                            result_reason,
                        ),
                    )
                    raise PlanBoundReplanRequired(
                        "canonical merge request became terminally unusable: "
                        + result_reason
                    )
                if time.monotonic() >= deadline:
                    detail = (
                        ""
                        if last_exception is None
                        else f" ({type(last_exception).__name__})"
                    )
                    raise PlanBoundReplanRequired(
                        "canonical merge request exceeded its compiled "
                        f"execution bound{detail}"
                    )
                try:
                    consume_result = self._consume_one_merge_candidate()
                    if isinstance(consume_result, Mapping):
                        last_result = dict(consume_result)
                    last_exception = None
                except Exception as exc:
                    # Another canonical train may own the lease.  Poll the
                    # durable row outside all queue/store/lifecycle locks and
                    # retry within the compiler-owned execution bound.
                    last_exception = exc
                time.sleep(0.05)

        def _mark_plan_bound_merge_completed(
            self,
            *,
            lease_cid: str,
            execution_lease: PlanBoundExecutionLease,
            request: Any,
        ) -> tuple[str, PlanBoundExecutionLease]:
            """Bind queue completion and canonical task acceptance in one CAS."""

            if str(getattr(request, "status", "")) != "completed":
                raise PlanBoundReplanRequired(
                    "merge completion transition lacks a completed queue row"
                )
            # Recovery deliberately disables the daemon's ordinary task
            # loader so it cannot reselect/replay provider work.  Completion
            # still needs authoritative evidence that the canonical merge
            # callback accepted the exact task.  Read one clean, tracked board
            # blob from the *current* HEAD and parse those very bytes; a peer
            # may legitimately have advanced HEAD after the immutable wave
            # was compiled.
            try:
                from ..runtime.configured_board_scheduler import (
                    _git_identity as configured_board_git_identity,
                )
                from ..runtime.configured_board_scheduler import (
                    _tracked_head_snapshot,
                )

                completion_head, _completion_tree = (
                    configured_board_git_identity(accepted_tree_root)
                )
                completion_board_bytes, _completion_board_revision = (
                    _tracked_head_snapshot(
                        repo_root=accepted_tree_root,
                        path=taskboard_path,
                        source_head=completion_head,
                    )
                )
                tasks = daemon_module.parse_task_text(
                    completion_board_bytes.decode("utf-8"),
                    path=taskboard_path,
                    task_header_prefix=self.task_header_prefix,
                )
            except Exception as exc:
                raise PlanBoundReplanRequired(
                    "completed merge row lacks a stable current board"
                ) from exc
            matching = [
                task
                for task in tasks
                if task.task_id == execution_lease.active_task_id
                and self._canonical_ref(task)
                == execution_lease.active_task_cid
            ]
            if len(matching) != 1 or str(matching[0].status) != "completed":
                raise PlanBoundReplanRequired(
                    "completed merge row lacks canonical task acceptance"
                )
            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    current = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=execution_lease.revision_cid,
                        slice_id=execution_lease.slice_id,
                        lane_id=execution_lease.lane_id,
                    )
                    if current is not None and current[1].phase == "merge_completed":
                        return current
                    if current != (lease_cid, execution_lease):
                        raise PlanBoundReplanRequired(
                            "merge completion lost its execution lease CAS"
                        )
                    intent = _secure_store_cas(
                        store,
                        execution_lease.merge_enqueue_intent_cid,
                    )
                    completed_at_ms = max(
                        int(time.time() * 1000),
                        int(intent["prepared_at_ms"]),
                    )
                    receipt = {
                        "schema": PLAN_BOUND_MERGE_QUEUE_RECEIPT_SCHEMA,
                        "authorization_cid": (
                            execution_lease.merge_authorization_cid
                        ),
                        "intent_cid": execution_lease.merge_enqueue_intent_cid,
                        "enqueue_fields_cid": intent["enqueue_fields_cid"],
                        "request_id": execution_lease.merge_request_id,
                        "dedupe_key": str(request.dedupe_key),
                        "observed_status": "completed",
                        "confirmed_at_ms": completed_at_ms,
                    }
                    receipt_cid = store.put_cas(receipt)
                    if _secure_store_cas(store, receipt_cid) != receipt:
                        raise PlanBoundReplanRequired(
                            "merge completion receipt failed CAS round trip"
                        )
                    completed = replace(
                        execution_lease,
                        generation=execution_lease.generation + 1,
                        phase="merge_completed",
                        prior_execution_lease_cid=lease_cid,
                        merge_queue_receipt_cid=receipt_cid,
                    )
                    completed_cid = _publish_plan_bound_execution_lease_locked(
                        store,
                        completed,
                        expected_current_cid=lease_cid,
                    )
                    return completed_cid, completed

        def _recover_plan_bound_merge_enqueue(
            self,
            lease_cid: str,
            lease: PlanBoundExecutionLease,
        ) -> dict[str, Any]:
            """Adopt a prepared/confirmed queue handoff without provider replay."""

            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    recovery_birth_cid = (
                        self._bind_current_merge_recovery_birth_locked(
                            execution_lease_cid=lease_cid,
                            execution_lease=lease,
                        )
                    )
                    intent = _secure_store_cas(
                        store,
                        lease.merge_enqueue_intent_cid,
                    )
                    enqueue_fields = dict(intent["enqueue_fields"])
            if lease.phase == "merge_enqueue_prepared":
                try:
                    request = self.merge_queue.enqueue(**enqueue_fields)
                except Exception as exc:
                    raise PlanBoundReplanRequired(
                        "canonical merge enqueue recovery failed"
                    ) from exc
                try:
                    self._require_queue_request_matches_intent(
                        request,
                        enqueue_fields,
                    )
                except PlanBoundReplanRequired as exc:
                    self._quarantine_merge_request_mismatch(
                        request,
                        prepared=lease,
                        reason=(
                            "deduplicated_queue_row_differs_from_intent:"
                            f"{exc}"
                        ),
                    )
                    raise
                lease_cid, lease = self._confirm_plan_bound_merge_enqueue(
                    prepared_cid=lease_cid,
                    prepared=lease,
                    request=request,
                    enqueue_fields=enqueue_fields,
                )
            else:
                request = self.merge_queue.get(lease.merge_request_id)
                if request is None:
                    raise PlanBoundReplanRequired(
                        "confirmed canonical merge request disappeared"
                    )
                try:
                    self._require_queue_request_matches_intent(
                        request,
                        enqueue_fields,
                    )
                except PlanBoundReplanRequired as exc:
                    self._quarantine_merge_request_mismatch(
                        request,
                        prepared=lease,
                        reason=f"confirmed_queue_row_differs_from_intent:{exc}",
                    )
                    raise
            completed_request = self._drain_plan_bound_merge_request(
                request_id=str(request.request_id),
                execution_lease=lease,
                enqueue_fields=enqueue_fields,
            )
            completed_cid, _completed_lease = (
                self._mark_plan_bound_merge_completed(
                    lease_cid=lease_cid,
                    execution_lease=lease,
                    request=completed_request,
                )
            )
            return {
                "reason": "plan_bound_merge_enqueue_recovered",
                "request_id": str(request.request_id),
                "execution_lease_cid": completed_cid,
                "merge_recovery_birth_cid": recovery_birth_cid,
                "provider_dispatched": False,
                "attempt_consumed": False,
                "merge_request_status": completed_request.status,
            }

        def _recover_plan_bound_proposal_ready(
            self,
            lease_cid: str,
            lease: PlanBoundExecutionLease,
        ) -> dict[str, Any]:
            """Resume an accepted pre-barrier candidate without provider replay."""

            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    recovery_birth_cid = (
                        self._bind_current_merge_recovery_birth_locked(
                            execution_lease_cid=lease_cid,
                            execution_lease=lease,
                        )
                    )
                    handoff = _secure_store_cas(
                        store,
                        lease.proposal_handoff_cid,
                    )
                    disposition = (
                        _load_plan_bound_proposal_disposition_locked(
                            store,
                            revision_cid=lease.revision_cid,
                            slice_id=lease.slice_id,
                        )
                    )
                    if (
                        disposition is None
                        or disposition[1].execution_lease_cid != lease_cid
                        or disposition[1].outcome != handoff.get("outcome")
                    ):
                        raise PlanBoundReplanRequired(
                            "proposal recovery lost its immutable disposition"
                        )
                    enqueue_fields = dict(handoff["enqueue_fields"])
                    timeout_ms = self._proposal_barrier_timeout_ms(lease)
            barrier_cid, barrier = ProductionParallelPlanAdapter(
                store
            ).await_wave_diff_barrier(
                revision_cid=lease.revision_cid,
                slice_manifest_cid=lease.slice_manifest_cid,
                timeout_ms=timeout_ms,
            )
            if barrier.decision != "released":
                raise PlanBoundReplanRequired(
                    "recovered proposal barrier denied merge admission"
                )
            self._await_plan_bound_merge_turn(lease)
            prepared_cid, prepared = self._prepare_plan_bound_merge_enqueue(
                enqueue_fields=enqueue_fields,
                barrier_cid=barrier_cid,
                worktree_path=Path(str(handoff["workspace_path"])),
                baseline_ref=str(handoff["baseline_ref"]),
                implementation_commit=str(handoff["implementation_commit"]),
                actual_paths=tuple(handoff["actual_changed_paths"]),
                branch_name=str(handoff["branch_name"]),
                attempt=int(handoff["attempt"]),
                recovery_birth_cid=recovery_birth_cid,
            )
            try:
                request = self.merge_queue.enqueue(**enqueue_fields)
            except Exception as exc:
                raise PlanBoundReplanRequired(
                    "recovered proposal canonical enqueue failed"
                ) from exc
            confirmed_cid, confirmed = self._confirm_plan_bound_merge_enqueue(
                prepared_cid=prepared_cid,
                prepared=prepared,
                request=request,
                enqueue_fields=enqueue_fields,
            )
            completed_request = self._drain_plan_bound_merge_request(
                request_id=str(request.request_id),
                execution_lease=confirmed,
                enqueue_fields=enqueue_fields,
            )
            completed_cid, _completed = self._mark_plan_bound_merge_completed(
                lease_cid=confirmed_cid,
                execution_lease=confirmed,
                request=completed_request,
            )
            return {
                "reason": "plan_bound_proposal_recovered",
                "request_id": str(request.request_id),
                "execution_lease_cid": completed_cid,
                "merge_recovery_birth_cid": recovery_birth_cid,
                "provider_dispatched": False,
                "attempt_consumed": False,
                "merge_request_status": completed_request.status,
            }

        def run_once(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Resume only a durable merge handoff before ordinary selection."""

            with store._thread_lock:  # noqa: SLF001
                with store._guard():  # noqa: SLF001
                    current = _load_plan_bound_execution_lease_locked(
                        store,
                        revision_cid=pinned.plan_bound_revision_cid,
                        slice_id=pinned.plan_bound_slice_id,
                        lane_id=pinned.plan_bound_lane_id,
                    )
            if current is not None and current[1].phase in {
                "merge_enqueue_prepared",
                "merge_enqueue_confirmed",
            }:
                return self._recover_plan_bound_merge_enqueue(*current)
            if current is not None and current[1].phase == "proposal_ready":
                return self._recover_plan_bound_proposal_ready(*current)
            if current is not None and current[1].phase == "merge_completed":
                return {
                    "reason": "plan_bound_merge_already_completed",
                    "execution_lease_cid": current[0],
                    "request_id": current[1].merge_request_id,
                    "provider_dispatched": False,
                    "attempt_consumed": False,
                    "merge_request_status": "completed",
                }
            return super().run_once(*args, **kwargs)

        def _enqueue_validated_worktree(
            self,
            *,
            state: Any,
            task: Any,
            attempt: int,
            branch_name: str,
            baseline_ref: str,
            worktree_path: Path,
            implementation_commit: str,
            commit_result: Mapping[str, Any],
            validation_result: Mapping[str, Any],
            changed_submodule_paths: Sequence[str] | None = None,
        ) -> dict[str, Any]:
            """Release the existing merge enqueue only after whole-wave ALLOW."""

            resolved_baseline = self._resolved_git_commit(
                worktree_path,
                baseline_ref,
            )
            resolved_implementation = self._resolved_git_commit(
                worktree_path,
                implementation_commit,
            )
            current_head = self._resolved_git_commit(
                worktree_path,
                "HEAD",
            )
            if current_head != resolved_implementation:
                raise PlanBoundDispatchError(
                    "candidate HEAD changed before proposal barrier"
                )
            actual_paths = self._full_plan_bound_effect_paths(
                worktree_path,
                baseline_ref=resolved_baseline,
            )
            proposal_gate = validation_result.get("proposal_gate")
            if not isinstance(proposal_gate, Mapping):
                raise PlanBoundDispatchError(
                    "validated candidate lacks canonical proposal evidence"
                )
            proposal_id = str(proposal_gate.get("proposal_id") or "")
            proposal_receipt_id = str(proposal_gate.get("receipt_id") or "")
            compact_paths = {
                str(path).strip()
                for path in tuple(proposal_gate.get("changed_paths") or ())
                if str(path).strip()
            }
            owned_scope = self._proposal_scope_paths(task)
            full_paths_owned = all(
                any(
                    self._path_matches_scope(path, pattern)
                    for pattern in owned_scope
                )
                for path in actual_paths
            )
            if (
                proposal_gate.get("accepted") is True
                and proposal_id
                and proposal_receipt_id
                and actual_paths
                and not full_paths_owned
            ):
                self._publish_proposal_disposition_and_wait(
                    outcome="rejected",
                    baseline_ref=resolved_baseline,
                    implementation_commit=resolved_implementation,
                    proposal_id=proposal_id,
                    proposal_receipt_id=proposal_receipt_id,
                    reason_codes=(
                        "full_diff_path_outside_scope",
                        "path_outside_scope",
                    ),
                    actual_changed_paths=actual_paths,
                )
                raise PlanBoundReplanRequired(
                    "independent full diff crossed the plan-bound task scope"
                )
            if (
                proposal_gate.get("accepted") is not True
                or not proposal_id
                or not proposal_receipt_id
                or not actual_paths
                or not compact_paths.issubset(set(actual_paths))
            ):
                raise PlanBoundDispatchError(
                    "validated candidate proposal evidence is partial"
                )
            canonical_submodule_paths = tuple(
                changed_submodule_paths
                if changed_submodule_paths is not None
                else self._committed_submodule_paths(
                    commit_result.get("submodule_results") or []
                )
            )
            proposal_enqueue_fields = (
                self._capture_plan_bound_enqueue_fields(
                    branch_name=branch_name,
                    implementation_commit=resolved_implementation,
                    baseline_ref=resolved_baseline,
                    worktree_path=worktree_path,
                    task=task,
                    attempt=attempt,
                    changed_submodule_paths=canonical_submodule_paths,
                    validation_result=validation_result,
                )
            )
            barrier_cid, _barrier = self._publish_proposal_disposition_and_wait(
                outcome="changed",
                baseline_ref=resolved_baseline,
                implementation_commit=resolved_implementation,
                proposal_id=proposal_id,
                proposal_receipt_id=proposal_receipt_id,
                reason_codes=(),
                actual_changed_paths=actual_paths,
                enqueue_fields=proposal_enqueue_fields,
                attempt=attempt,
                branch_name=branch_name,
            )
            _turn_cid, turn_lease = self._load_execution_lease(
                phases=("proposal_ready",)
            )
            self._await_plan_bound_merge_turn(turn_lease)
            original_enqueue = self.merge_queue.enqueue
            captured_enqueue_fields: dict[str, Any] = {}

            def guarded_enqueue(*positional: Any, **keyword: Any) -> Any:
                enqueue_fields = self._canonical_merge_enqueue_fields(
                    positional,
                    keyword,
                )
                if enqueue_fields != proposal_enqueue_fields:
                    raise PlanBoundReplanRequired(
                        "canonical enqueue fields changed after wave release"
                    )
                captured_enqueue_fields.update(enqueue_fields)
                prepared_cid, prepared = self._prepare_plan_bound_merge_enqueue(
                    enqueue_fields=enqueue_fields,
                    barrier_cid=barrier_cid,
                    worktree_path=worktree_path,
                    baseline_ref=resolved_baseline,
                    implementation_commit=resolved_implementation,
                    actual_paths=actual_paths,
                    branch_name=branch_name,
                    attempt=attempt,
                )
                request = original_enqueue(**enqueue_fields)
                self._confirm_plan_bound_merge_enqueue(
                    prepared_cid=prepared_cid,
                    prepared=prepared,
                    request=request,
                    enqueue_fields=enqueue_fields,
                )
                return request

            self.merge_queue.enqueue = guarded_enqueue
            try:
                result = super()._enqueue_validated_worktree(
                    state=state,
                    task=task,
                    attempt=attempt,
                    branch_name=branch_name,
                    baseline_ref=resolved_baseline,
                    worktree_path=worktree_path,
                    implementation_commit=resolved_implementation,
                    commit_result=commit_result,
                    validation_result=validation_result,
                    changed_submodule_paths=changed_submodule_paths,
                )
            finally:
                self.merge_queue.enqueue = original_enqueue
            confirmed = self._load_execution_lease(
                phases=("merge_enqueue_confirmed",)
            )
            if str(result.get("request_id") or "") != confirmed[1].merge_request_id:
                raise PlanBoundReplanRequired(
                    "canonical merge result differs from its durable receipt"
                )
            completed_request = self._drain_plan_bound_merge_request(
                request_id=confirmed[1].merge_request_id,
                execution_lease=confirmed[1],
                enqueue_fields=captured_enqueue_fields,
            )
            self._mark_plan_bound_merge_completed(
                lease_cid=confirmed[0],
                execution_lease=confirmed[1],
                request=completed_request,
            )
            return result

        def _release_pooled_worktree_lease(
            self,
            worktree_path: Path,
            *,
            reason: str,
            reusable: bool = True,
            finalize_lifecycle: bool = True,
        ) -> dict[str, Any]:
            """Keep the exact candidate lease alive through queue admission.

            The canonical daemon ordinarily releases a pooled checkout before
            it constructs the merge request.  A plan-bound child must still
            revalidate the exact SETTLING lifecycle and complete diff at that
            last pre-enqueue boundary, so its checkout stays owned until the
            canonical post-enqueue lifecycle handoff/merge cleanup.
            """

            if reason == "merge_queue_handoff" and not finalize_lifecycle:
                return {
                    "attempted": False,
                    "released": False,
                    "reason": "plan_bound_queue_authorization_pending",
                    "worktree_path": str(worktree_path),
                }
            return super()._release_pooled_worktree_lease(
                worktree_path,
                reason=reason,
                reusable=reusable,
                finalize_lifecycle=finalize_lifecycle,
            )

        def _validate_implementation_patch(
            self,
            workspace_path: Path,
            task: Any,
            **kwargs: Any,
        ) -> Any:
            """Project a real proposal scope rejection before merge admission."""

            result = super()._validate_implementation_patch(
                workspace_path,
                task,
                **kwargs,
            )
            reason_codes = tuple(
                sorted(
                    {
                        str(getattr(finding.code, "value", finding.code))
                        for finding in tuple(getattr(result, "findings", ()) or ())
                    }
                )
            )
            if bool(getattr(result, "accepted", False)):
                return result

            proposal = getattr(result, "proposal", None)
            receipt = getattr(result, "receipt", None)
            proposal_id = str(getattr(proposal, "proposal_id", "") or "")
            receipt_id = str(getattr(receipt, "receipt_id", "") or "")
            if not proposal_id or not receipt_id or not reason_codes:
                raise PlanBoundDispatchError(
                    "typed proposal rejection is missing canonical evidence"
                )
            resolved_baseline = self._resolved_git_commit(
                workspace_path,
                str(kwargs.get("baseline_ref") or "HEAD"),
            )
            changed_paths = self._full_plan_bound_effect_paths(
                workspace_path,
                baseline_ref=resolved_baseline,
            )
            self._publish_proposal_disposition_and_wait(
                outcome="rejected",
                baseline_ref=resolved_baseline,
                implementation_commit="",
                proposal_id=proposal_id,
                proposal_receipt_id=receipt_id,
                reason_codes=reason_codes,
                actual_changed_paths=changed_paths,
            )
            raise PlanBoundDispatchError(
                "rejected proposal unexpectedly received wave release"
            )

    def plan_bound_daemon_factory(**kwargs: Any) -> Any:
        if (
            tuple(kwargs.get("execution_slice_task_ids") or ())
            or tuple(kwargs.get("execution_slice_task_cids") or ()) != task_cids
        ):
            raise PlanBoundDispatchError(
                "daemon constructor slice differs from the plan-bound slice"
            )
        reclaim_env_name = (
            daemon_module.WORKTREE_LIFECYCLE_RECLAIM_DEAD_ON_STARTUP_ENV
        )
        prior_reclaim = os.environ.get(reclaim_env_name)
        if recovery_only:
            # The generic daemon startup recovery cannot distinguish an
            # abandoned candidate from this plan-bound, CAS-authorized merge
            # handoff.  Keep it from terminalizing the exact SETTLING record
            # before the adapter proves the dead predecessor and binds the
            # new process birth in `_bind_current_merge_recovery_birth_locked`.
            # This dedicated child restores the process environment
            # immediately after construction.
            os.environ[reclaim_env_name] = "0"
        try:
            daemon = PlanBoundImplementationDaemon(
                **kwargs,
                plan_revision_store=store_view,
                parallel_execution_plan=plan_payload,
                require_active_plan_revision=True,
                plan_capacity_snapshot=dict(live_host),
                plan_provider_snapshots=relevant_live_providers,
            )
        finally:
            if recovery_only:
                if prior_reclaim is None:
                    os.environ.pop(reclaim_env_name, None)
                else:
                    os.environ[reclaim_env_name] = prior_reclaim
        canonical_ref = daemon._canonical_ref

        def plan_bound_canonical_ref(task: Any) -> str:
            """Keep daemon admission on the immutable board ID/CID pair."""

            observed = str(canonical_ref(task) or "").strip()
            task_id = str(getattr(task, "task_id", "") or "").strip()
            expected = cid_by_id.get(task_id)
            if expected is None:
                return observed
            if observed != expected:
                raise PlanBoundDispatchError(
                    "plan-bound daemon canonical task identity drifted from "
                    "the immutable slice"
                )
            return expected

        daemon._canonical_ref = plan_bound_canonical_ref
        load_active_plan_binding = daemon._load_active_plan_binding

        def plan_bound_load_active_binding(*, refresh: bool = False) -> Any:
            """Bind compiled claims to IDs only after exact CID admission."""

            binding = load_active_plan_binding(refresh=refresh)
            if binding is None:
                return None
            if tuple(binding.execution_slice_task_cids) != task_cids:
                raise PlanBoundDispatchError(
                    "daemon runtime binding changed the immutable CID slice"
                )
            bound = replace(
                binding,
                execution_slice_task_ids=task_ids,
                execution_slice_task_cids=task_cids,
            )
            daemon._active_plan_binding = bound
            return bound

        daemon._load_active_plan_binding = plan_bound_load_active_binding
        return daemon

    original_imported_capsule = daemon_module._IMPORTED_CONTROL_PLANE_CAPSULE
    original_imported_launch = daemon_module._IMPORTED_CONTROL_PLANE_LAUNCH
    daemon_module._IMPORTED_CONTROL_PLANE_CAPSULE = accepted_control_plane_pin
    daemon_module._IMPORTED_CONTROL_PLANE_LAUNCH = accepted_control_plane_launch
    daemon_module.PortalImplementationDaemon = plan_bound_daemon_factory
    daemon_module.parse_task_file = plan_bound_parse_task_file
    try:
        try:
            result = daemon_module.main(daemon_argv)
        except PlanBoundReplanRequired as exc:
            logger.warning(
                "Plan-bound daemon requested a fenced replan: %s",
                exc,
            )
            return PLAN_BOUND_REPLAN_RETURN_CODE
    finally:
        daemon_module.parse_task_file = canonical_parse_task_file
        daemon_module.PortalImplementationDaemon = canonical_daemon_class
        daemon_module._IMPORTED_CONTROL_PLANE_CAPSULE = original_imported_capsule
        daemon_module._IMPORTED_CONTROL_PLANE_LAUNCH = original_imported_launch
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            terminal_merge_failure = any(
                _load_plan_bound_merge_terminal_failure_locked(
                    store,
                    revision_cid=pinned.plan_bound_revision_cid,
                    slice_id=execution_slice.slice_id,
                )
                is not None
                for execution_slice in manifest.nonempty
            )
    if terminal_merge_failure:
        return PLAN_BOUND_REPLAN_RETURN_CODE
    barrier = ProductionParallelPlanAdapter(store).load_wave_diff_barrier(
        revision_cid=pinned.plan_bound_revision_cid,
        slice_manifest_cid=pinned.plan_bound_slice_manifest_cid,
    )
    if barrier is not None and barrier[1].decision != "released":
        return PLAN_BOUND_REPLAN_RETURN_CODE
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == PLAN_BOUND_DAEMON_CHILD_MARKER:
        return _run_plan_bound_daemon_child(raw_argv[1:])
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    supervisor = PortalImplementationSupervisor(supervisor_config_from_args(args, repo_root=REPO_ROOT))
    if args.once:
        result = supervisor.run_once()
        logger.info("Portal implementation supervisor check complete: %s", result)
        if args.fail_on_reconciliation_error:
            failure_reason = _reconciliation_preflight_failure_reason(
                result
            )
            if failure_reason:
                logger.error(
                    "Strict reconciliation preflight did not settle: %s",
                    failure_reason,
                )
                return 1
        return 0
    return supervisor.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())


TodoSupervisorConfig = PortalSupervisorConfig
TodoImplementationSupervisor = PortalImplementationSupervisor
