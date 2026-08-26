"""Transactional intent repository for objectives, goals, plans, tasks, queues.

DQP-012 / IntentRepository@1 / PlanRevisionRepository@1
=======================================================

Migrates objectives, goals, plans, tasks, queue backoff, attempts, blocks, and
completion evidence into the control-plane DuckDB schema. Every mutation advances
normalized projections and appends a domain event in the **same** transaction —
no cross-file saga is required.

Invariants
----------
* Canonical identities (``objective_id``, ``goal_cid``, ``plan_cid``,
  ``task_cid``) are stable; display aliases never serve as durable keys.
* Task/goal completion requires **current** required evidence (validation
  results or evidence nodes bound to the live acceptance criteria). Exported
  status strings are never completion authority.
* Rebuilding projections from admitted domain events matches the live rows.
* CAS heads protect concurrent writers on objectives, plans, and tasks.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    ControlPlaneBoundsError,
    ControlPlaneContractError,
    ControlPlaneIdentityError,
    canonical_json_bytes,
    content_identity,
)
from .control_plane_migrations import duckdb_available
from .control_plane_schema import install_control_plane_schema
from .duckdb_state import (
    DuckDBConnectionPolicyError,
    DuckDBQuackMutationConflictError,
    DuckDBQuackMutationTransitionError,
    DuckDBQuackMutationUnknownOutcomeError,
    STALE_IN_PROGRESS_UNSTALL_SECONDS,
    _is_quack_session_dead,
    exclusive_file_lock,
    is_quack_transport_target,
    open_duckdb_connection,
    quack_owner_mutation_write_lock_path,
    quack_session_is_live,
    quack_transport_uri,
    unstall_stale_in_progress_tasks as apply_stale_in_progress_unstall,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

INTENT_REPOSITORY_INTERFACE: Final[str] = "IntentRepository@1"
PLAN_REVISION_REPOSITORY_INTERFACE: Final[str] = "PlanRevisionRepository@1"

INTENT_REPOSITORY_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-repository@1"
PLAN_REVISION_REPOSITORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-repository@1"
)
INTENT_EVENT_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-event@1"
INTENT_SNAPSHOT_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-snapshot@1"
INTENT_RECEIPT_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-receipt@1"
QUEUE_ENTRY_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-queue-entry@1"
COMPLETION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
)
INTENT_PLAN_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-plan-projection@1"
)
INTENT_COMPLETION_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-completion-projection@1"
)
GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-completion-authority-spec@1"
)
GOAL_COMPLETION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-completion-receipt@1"
)
GOAL_ROOT_COMPLETION_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-root-completion-gate@1"
)
GOAL_RUNTIME_SETTLEMENT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vrif-runtime-settlement-binding@1"
)
GOAL_AUTHORITY_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-authority-projection@1"
)
GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-terminal-report-contract@1"
)
GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-terminal-report-evidence@2"
)
_DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-completion-binding@1"
)
_MERGE_TARGET_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
)
_GOAL_TERMINAL_PRODUCER_ARTIFACTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-terminal-producer-artifacts@1"
)
_GOAL_TERMINAL_PRODUCER_RECEIPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/goal-terminal-producer-receipt-binding@1"
)
TASK_PROJECTION_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-projection-spec@1"
)
TASK_AUTHORITY_SPEC_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/task-authority-spec@1"
DATABASE_VIRGIN_TASK_TRANSFER_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "database-virgin-task-transfer-request@1"
)
DATABASE_VIRGIN_TASK_TRANSFER_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "database-virgin-task-transfer-binding@1"
)
DATABASE_VIRGIN_TASK_TRANSFER_CURSOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "database-virgin-task-transfer-claim-cursor@1"
)
DATABASE_VIRGIN_TASK_TRANSFER_MODE: Final[str] = "virgin-transfer"
DATABASE_CLAIM_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-claim-policy@1"
)
TASK_REVISION_HISTORY_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1"
)
PLAN_HEAD_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/intent-plan-head@1"

INTENT_STREAM_ID: Final[str] = "stream:intent"
DEFAULT_OWNER_ID: Final[str] = "intent-repository:local"
DEFAULT_SESSION_ID: Final[str] = "session:intent"

MAX_ID_BYTES: Final[int] = 512
MAX_BODY_BYTES: Final[int] = 262_144
MAX_PAGE_LIMIT: Final[int] = 1_000
DEFAULT_PAGE_LIMIT: Final[int] = 100
MAX_EVENTS: Final[int] = 1_000_000
MAX_ACCEPTANCE: Final[int] = 256
MAX_VALIDATIONS: Final[int] = 256
MAX_OUTPUTS: Final[int] = 256
MAX_DEPENDENCIES: Final[int] = 1_024
MAX_EVIDENCE: Final[int] = 4_096
MAX_PROJECTION_RECORDS: Final[int] = 10_000
MAX_TASK_PROJECTION_BYTES: Final[int] = 1_048_576
MAX_PLAN_PROJECTION_BYTES: Final[int] = 16_777_216
MAX_COMPLETION_PROJECTION_BYTES: Final[int] = 16_777_216
MAX_GOAL_AUTHORITY_PROJECTION_BYTES: Final[int] = 4_194_304
DEFAULT_EVIDENCE_FRESHNESS_SECONDS: Final[int] = 3_600

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._:@+-]{0,255}$")

_READY_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "proposed",
        "admitted",
        "pending",
        "ready",
        "todo",
        "queued",
        "retrying",
    }
)
_COMPLETED_STATUSES: Final[frozenset[str]] = frozenset({"completed", "skipped", "complete", "done"})
_SUCCESSFUL_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_COMPLETED_STATUSES,
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
    }
)
_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_READY_STATUSES,
        *_TERMINAL_STATUSES,
        "claimed",
        "in_progress",
        "running",
        "blocked",
    }
)
_ACTIVE_ATTEMPT_STATUSES: Final[frozenset[str]] = frozenset(
    {"started", "running", "in_progress"}
)
_TERMINAL_ATTEMPT_STATUSES: Final[frozenset[str]] = frozenset(
    {"succeeded", "completed", "failed", "cancelled", "released", "expired"}
)
_ATTEMPT_STATUSES: Final[frozenset[str]] = frozenset(
    {*_ACTIVE_ATTEMPT_STATUSES, *_TERMINAL_ATTEMPT_STATUSES}
)
_GOAL_OPEN_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "open",
        "active",
        "reopened",
        "provisionally_complete",
        "analysis_inconclusive",
        "waiting",
    }
)
_GOAL_COMPLETED_STATUSES: Final[frozenset[str]] = frozenset(
    {"verified_complete", "completed", "complete", "done"}
)
_GOAL_CLOSED_STATUSES: Final[frozenset[str]] = frozenset(
    {*_GOAL_COMPLETED_STATUSES, "blocked"}
)
_GOAL_STATUSES: Final[frozenset[str]] = frozenset(
    {*_GOAL_OPEN_STATUSES, *_GOAL_CLOSED_STATUSES}
)

_ROOT_COMPLETION_POLICY_FIELDS: Final[tuple[str, ...]] = (
    "all_task_dependencies_terminal_required",
    "goal_completion_contracts_required",
    "current_tree_required",
    "active_mutating_claims_empty_required",
    "merge_queue_settled_required",
    "blocking_obligations_empty_required",
    "required_receipts_and_seals_verify",
    "non_success_terminals_never_report_success",
    "ducklake_outage_cannot_block_core_completion",
    "final_report_required",
)
_ROOT_TERMINAL_TASK_POLICY_FIELD: Final[str] = "terminal_task_id"
_TERMINAL_REPORT_VALIDATOR: Final[str] = "DatabasePortalExecutionBridge@1"
_TERMINAL_REPORT_VALIDATION_ARGV: Final[tuple[str, ...]] = (
    "portal-supervisor-gates",
)

# Intent-owned projection tables fully rebuilt from admitted intent events.
_PROJECTION_TABLES: Final[tuple[str, ...]] = (
    "objectives",
    "objective_revisions",
    "goals",
    "goal_edges",
    "plans",
    "plan_revisions",
    "planning_decisions",
    "plan_candidates",
    "tasks",
    "task_revisions",
    "task_dependencies",
    "task_outputs",
    "task_acceptance",
    "task_validations",
    "task_blocks",
    "task_attempts",
    "completion_receipts",
    "evidence_nodes",
)

# Shared tables: only intent-owned rows are cleared (not the whole table).
_SHARED_QUEUE_LEASE_SCHEMA: Final[str] = QUEUE_ENTRY_SCHEMA


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntentRepositoryError(RuntimeError):
    """Base fail-closed error for intent repository operations."""


class IntentRepositoryConflictError(IntentRepositoryError):
    """CAS head, fence, or expected-revision conflict."""


class IntentRepositoryTransitionError(IntentRepositoryError):
    """Owner rejected a status transition outside the closed matrix."""


class IntentRepositoryUnknownOutcomeError(IntentRepositoryError):
    """A remote owner effect committed without fresh projection settlement."""


class IntentRepositoryIntegrityError(IntentRepositoryError):
    """Schema, identity, or projection integrity failure."""


class IntentRepositoryBoundsError(IntentRepositoryError, ValueError):
    """A count, byte, or page bound was exceeded."""


class IntentRepositoryNotOpenError(IntentRepositoryError):
    """Operation requires an open repository session."""


class IntentCompletionError(IntentRepositoryError):
    """Completion refused because required current evidence is missing."""


class IntentEvidenceError(IntentRepositoryError):
    """Evidence material is stale, foreign, or incomplete."""


class DuckDBUnavailableError(IntentRepositoryError):
    """DuckDB is required but missing from the environment."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class IntentEventType(str, Enum):
    """Closed set of admitted intent domain-event types."""

    OBJECTIVE_UPSERTED = "intent.objective_upserted"
    OBJECTIVE_REVISED = "intent.objective_revised"
    GOAL_UPSERTED = "intent.goal_upserted"
    GOAL_EDGE_LINKED = "intent.goal_edge_linked"
    GOAL_REOPENED = "intent.goal_reopened"
    PLAN_UPSERTED = "intent.plan_upserted"
    PLAN_REVISION_APPENDED = "intent.plan_revision_appended"
    PLAN_SUPERSEDED = "intent.plan_superseded"
    PLAN_CONTINUED = "intent.plan_continued"
    PLAN_HEAD_SET = "intent.plan_head_set"
    TASK_UPSERTED = "intent.task_upserted"
    TASK_DEPENDENCIES_SET = "intent.task_dependencies_set"
    TASK_OUTPUTS_SET = "intent.task_outputs_set"
    TASK_ACCEPTANCE_SET = "intent.task_acceptance_set"
    TASK_VALIDATIONS_SET = "intent.task_validations_set"
    TASK_STATUS_CHANGED = "intent.task_status_changed"
    TASK_BLOCKED = "intent.task_blocked"
    TASK_UNBLOCKED = "intent.task_unblocked"
    ATTEMPT_RECORDED = "intent.attempt_recorded"
    QUEUE_BACKOFF = "intent.queue_backoff"
    QUEUE_RETRY = "intent.queue_retry"
    EVIDENCE_RECORDED = "intent.evidence_recorded"
    VALIDATION_RECORDED = "intent.validation_recorded"
    COMPLETION_RECORDED = "intent.completion_recorded"
    RECOVERY_APPLIED = "intent.recovery_applied"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_duckdb() -> Any:
    if not duckdb_available():
        raise DuckDBUnavailableError(
            "DuckDB is required for IntentRepository; install the optional duckdb dependency"
        )
    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise DuckDBUnavailableError("DuckDB is required for IntentRepository") from exc
    return duckdb


def _utc_iso(moment: datetime | None = None) -> str:
    value = moment or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event_timestamp(value: Any, *, noun: str) -> str:
    """Return one producer-canonical UTC event timestamp, or fail closed."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise IntentRepositoryIntegrityError(f"{noun} is not a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise IntentRepositoryIntegrityError(
            f"{noun} is not a canonical UTC timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise IntentRepositoryIntegrityError(f"{noun} is not a canonical UTC timestamp")
    canonical = (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    if value != canonical:
        raise IntentRepositoryIntegrityError(f"{noun} is not a canonical UTC timestamp")
    return value


def _now_ms() -> int:
    return int(time.time() * 1000)


def _identifier(value: Any, *, noun: str) -> str:
    if not isinstance(value, str):
        raise ControlPlaneIdentityError(f"{noun} must be a string")
    text = value.strip()
    if not text:
        raise ControlPlaneIdentityError(f"{noun} must not be empty")
    if len(text.encode("utf-8")) > MAX_ID_BYTES:
        raise ControlPlaneBoundsError(f"{noun} exceeds its byte bound")
    if "\x00" in text or not _SAFE_ID.match(text):
        raise ControlPlaneIdentityError(f"{noun} is not a safe identifier")
    return text


def _optional_identifier(value: Any, *, noun: str) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value, noun=noun)


def _output_path(value: Any, *, noun: str) -> str:
    """Accept a repo-relative output path, including dotfiles such as ``.gitignore``."""

    if not isinstance(value, str):
        raise ControlPlaneIdentityError(f"{noun} must be a string")
    text = value.strip()
    if not text:
        raise ControlPlaneIdentityError(f"{noun} must not be empty")
    if len(text.encode("utf-8")) > MAX_ID_BYTES:
        raise ControlPlaneBoundsError(f"{noun} exceeds its byte bound")
    if "\x00" in text or text.startswith("/") or "\\" in text:
        raise ControlPlaneIdentityError(f"{noun} is not a safe identifier")
    # Board manifests historically use a trailing delimiter to declare a
    # repository-relative directory.  Store the canonical path identity while
    # retaining the same absolute, traversal, and empty-segment checks below.
    text = text.rstrip("/")
    if not text:
        raise ControlPlaneIdentityError(f"{noun} must not be empty")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ControlPlaneIdentityError(f"{noun} is not a safe identifier")
    if not all(_SAFE_PATH_PART.match(part) for part in parts):
        raise ControlPlaneIdentityError(f"{noun} is not a safe identifier")
    return text


def _status(value: Any, *, allowed: frozenset[str], noun: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text not in allowed:
        raise IntentRepositoryError(f"{noun} status {value!r} is not in the closed set")
    return text


def _jsonable(value: Any) -> Any:
    """Coerce nested values into canonical-JSON-safe Python structures."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise IntentRepositoryError("float values are not allowed in intent JSON")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: str(item))
    raise IntentRepositoryError(f"unsupported intent JSON value type: {type(value).__name__}")


def _canonical(value: Any, *, noun: str = "payload") -> str:
    try:
        payload = canonical_json_bytes(_jsonable(value))
    except (ControlPlaneContractError, ControlPlaneBoundsError) as exc:
        raise IntentRepositoryError(f"{noun} is not canonical JSON") from exc
    if len(payload) > MAX_BODY_BYTES:
        raise IntentRepositoryBoundsError(f"{noun} exceeds body byte bound")
    return payload.decode("utf-8")


def _decode_json(value: Any, *, noun: str = "json") -> Any:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    text = str(value)

    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        return json.loads(text, object_pairs_hook=closed_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntentRepositoryIntegrityError(f"{noun} is not valid unambiguous JSON") from exc


def _receipt_with_preserved_reopen_count(
    receipt: Mapping[str, Any],
    previous_receipt: Any,
) -> dict[str, Any]:
    """Keep unknown-callback reopen count across later claim receipts."""

    stored = dict(receipt)
    if "unknown_callback_reopen_count" in stored:
        return stored
    previous_count = None
    if isinstance(previous_receipt, Mapping):
        previous_count = previous_receipt.get("unknown_callback_reopen_count")
    if previous_count is None:
        return stored
    try:
        stored["unknown_callback_reopen_count"] = max(0, int(previous_count))
    except (TypeError, ValueError):
        return dict(receipt)
    return stored


def database_task_alias_home_shard_index(task_alias: str, shard_count: int) -> int:
    """Return the shared deterministic alias-hash home lane."""

    if shard_count <= 1:
        return 0
    digest = hashlib.sha256(str(task_alias).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shard_count


def _trusted_database_claim_policy() -> Mapping[str, Any] | None:
    raw = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", "") or ""
    ).strip()
    if not raw:
        return None
    try:
        program = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IntentRepositoryTransitionError(
            "trusted database program JSON is malformed"
        ) from exc
    policy = program.get("claim_policy") if isinstance(program, Mapping) else None
    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise IntentRepositoryTransitionError(
            "trusted database claim policy is malformed"
        )
    normalized = dict(policy)
    shard_count = normalized.get("task_shard_count")
    if (
        set(normalized)
        != {
            "schema",
            "task_prefix",
            "task_shard_count",
            "strict_task_sharding",
            "idle_lane_work_stealing",
        }
        or normalized.get("schema") != DATABASE_CLAIM_POLICY_SCHEMA
        or not str(normalized.get("task_prefix") or "").strip()
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 1
        or normalized.get("strict_task_sharding") is not True
        or normalized.get("idle_lane_work_stealing")
        != DATABASE_VIRGIN_TASK_TRANSFER_MODE
    ):
        raise IntentRepositoryTransitionError(
            "trusted database claim policy is invalid"
        )
    normalized["task_prefix"] = str(normalized["task_prefix"]).strip()
    return MappingProxyType(normalized)


def _database_virgin_transfer_binding(
    *,
    task_cid: str,
    task_alias: str,
    receipt: Mapping[str, Any],
    shard_count: int,
) -> Mapping[str, Any] | None:
    raw = receipt.get("virgin_task_transfer")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise IntentRepositoryTransitionError(
            "database virgin-transfer binding is malformed"
        )
    binding = dict(raw)
    binding_id = str(binding.pop("binding_id", "") or "")
    recipient = binding.get("recipient_shard_index")
    source_revision = binding.get("source_task_revision")
    fencing_token = binding.get("fencing_token")
    fence_epoch = binding.get("fence_epoch")
    task_prefix = str(binding.get("task_prefix") or "")
    claim_policy_id = str(binding.get("claim_policy_id") or "")
    store_generation = str(binding.get("store_generation") or "")
    cohort_id = content_identity(
        {
            "kind": "database-virgin-task-transfer-cohort",
            "task_prefix": task_prefix,
            "task_shard_count": shard_count,
            "claim_policy_id": claim_policy_id,
            "store_generation": store_generation,
        }
    )
    trusted_policy = _trusted_database_claim_policy()
    trusted_policy_id = (
        content_identity(dict(trusted_policy))
        if trusted_policy is not None
        else ""
    )
    trusted_generation = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "") or ""
    ).strip()
    home_lane = database_task_alias_home_shard_index(task_alias, shard_count)
    valid = bool(
        binding_id
        and binding_id == content_identity(binding)
        and binding.get("schema") == DATABASE_VIRGIN_TASK_TRANSFER_BINDING_SCHEMA
        and binding.get("mode") == DATABASE_VIRGIN_TASK_TRANSFER_MODE
        and binding.get("task_cid") == task_cid
        and binding.get("task_alias") == task_alias
        and task_prefix
        and task_alias.startswith(task_prefix)
        and binding.get("task_shard_count") == shard_count
        and binding.get("home_shard_index") == home_lane
        and binding.get("cohort_id") == cohort_id
        and (trusted_policy is None or claim_policy_id == trusted_policy_id)
        and (
            trusted_policy is None
            or task_prefix == str(trusted_policy["task_prefix"])
        )
        and (
            trusted_policy is None
            or shard_count == int(trusted_policy["task_shard_count"])
        )
        and (not trusted_generation or store_generation == trusted_generation)
        and isinstance(recipient, int)
        and not isinstance(recipient, bool)
        and 0 <= int(recipient) < shard_count
        and int(recipient) != home_lane
        and isinstance(source_revision, int)
        and not isinstance(source_revision, bool)
        and int(source_revision) >= 1
        and str(binding.get("claim_id") or "")
        and str(binding.get("attempt_id") or "")
        and str(binding.get("owner_session_id") or "")
        and str(binding.get("lease_id") or "")
        and isinstance(fencing_token, int)
        and not isinstance(fencing_token, bool)
        and int(fencing_token) >= 1
        and isinstance(fence_epoch, int)
        and not isinstance(fence_epoch, bool)
        and int(fence_epoch) >= 1
    )
    if not valid:
        raise IntentRepositoryTransitionError(
            "database virgin-transfer binding does not match the task shard"
        )
    if receipt.get("operation") == "database_claim":
        claimed_from_revision = receipt.get("claimed_from_revision")
        valid = bool(
            receipt.get("task_shard_count") == shard_count
            and receipt.get("task_shard_index") == int(recipient)
            and receipt.get("owner_session_id") == binding.get("owner_session_id")
            and isinstance(claimed_from_revision, int)
            and not isinstance(claimed_from_revision, bool)
            and int(claimed_from_revision) >= int(source_revision)
            and str(receipt.get("claim_id") or "")
            and str(receipt.get("attempt_id") or "")
            and str(receipt.get("lease_id") or "")
            and isinstance(receipt.get("fencing_token"), int)
            and not isinstance(receipt.get("fencing_token"), bool)
            and int(receipt.get("fencing_token") or 0) >= int(fencing_token)
            and isinstance(receipt.get("fence_epoch"), int)
            and not isinstance(receipt.get("fence_epoch"), bool)
            and int(receipt.get("fence_epoch") or 0) >= int(fence_epoch)
        )
        if valid and claimed_from_revision == source_revision:
            valid = all(
                receipt.get(name) == binding.get(name)
                for name in (
                    "claim_id",
                    "attempt_id",
                    "lease_id",
                    "fencing_token",
                    "fence_epoch",
                )
            )
        elif valid:
            valid = bool(
                int(claimed_from_revision) > int(source_revision)
                and all(
                    receipt.get(name) != binding.get(name)
                    for name in ("claim_id", "attempt_id", "lease_id")
                )
                and int(receipt.get("fencing_token") or 0)
                > int(fencing_token)
            )
        if not valid:
            raise IntentRepositoryTransitionError(
                "database virgin-transfer claim does not match its binding"
            )
    return MappingProxyType({**binding, "binding_id": binding_id})


def _database_virgin_transfer_claim_cursor(
    *,
    receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = receipt.get("virgin_task_transfer_claim_cursor")
    if not isinstance(raw, Mapping):
        raise IntentRepositoryTransitionError(
            "database virgin-transfer claim cursor is missing"
        )
    cursor = dict(raw)
    cursor_id = str(cursor.pop("cursor_id", "") or "")
    claimed_from_revision = cursor.get("claimed_from_revision")
    fencing_token = cursor.get("fencing_token")
    fence_epoch = cursor.get("fence_epoch")
    valid = bool(
        cursor_id
        and cursor_id == content_identity(cursor)
        and cursor.get("schema") == DATABASE_VIRGIN_TASK_TRANSFER_CURSOR_SCHEMA
        and cursor.get("binding_id") == binding.get("binding_id")
        and cursor.get("owner_session_id") == binding.get("owner_session_id")
        and str(cursor.get("claim_id") or "")
        and str(cursor.get("attempt_id") or "")
        and str(cursor.get("lease_id") or "")
        and isinstance(claimed_from_revision, int)
        and not isinstance(claimed_from_revision, bool)
        and int(claimed_from_revision) >= int(binding["source_task_revision"])
        and isinstance(fencing_token, int)
        and not isinstance(fencing_token, bool)
        and int(fencing_token) >= int(binding["fencing_token"])
        and isinstance(fence_epoch, int)
        and not isinstance(fence_epoch, bool)
        and int(fence_epoch) >= int(binding["fence_epoch"])
    )
    if not valid:
        raise IntentRepositoryTransitionError(
            "database virgin-transfer claim cursor is invalid"
        )
    return MappingProxyType({**cursor, "cursor_id": cursor_id})


def _database_virgin_transfer_claim_cursor_body(
    *,
    binding: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": DATABASE_VIRGIN_TASK_TRANSFER_CURSOR_SCHEMA,
        "binding_id": str(binding["binding_id"]),
        "claim_id": str(receipt["claim_id"]),
        "attempt_id": str(receipt["attempt_id"]),
        "owner_session_id": str(receipt["owner_session_id"]),
        "lease_id": str(receipt["lease_id"]),
        "fencing_token": int(receipt["fencing_token"]),
        "fence_epoch": int(receipt["fence_epoch"]),
        "claimed_from_revision": int(receipt["claimed_from_revision"]),
    }
    return {**body, "cursor_id": content_identity(body)}


def database_virgin_transfer_binding_for_task(
    task: Any,
    *,
    shard_count: int,
) -> Mapping[str, Any] | None:
    """Validate and return the owner-stamped transfer assignment for a task."""

    field = (
        (lambda name, default="": task.get(name, default))
        if isinstance(task, Mapping)
        else (lambda name, default="": getattr(task, name, default))
    )
    body = field("body", {})
    receipt = body.get("completion_receipt") if isinstance(body, Mapping) else None
    if not isinstance(receipt, Mapping):
        return None
    binding = _database_virgin_transfer_binding(
        task_cid=str(field("task_cid") or ""),
        task_alias=str(field("task_alias") or ""),
        receipt=receipt,
        shard_count=shard_count,
    )
    if binding is not None:
        _database_virgin_transfer_claim_cursor(
            receipt=receipt,
            binding=binding,
        )
    return binding


def _database_task_field(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _database_task_status_receipt(task: Any) -> Mapping[str, Any]:
    body = _database_task_field(task, "body", {})
    receipt = body.get("completion_receipt") if isinstance(body, Mapping) else None
    return receipt if isinstance(receipt, Mapping) else {}


def _database_task_forbids_automatic_claim(task: Any) -> bool:
    body = _database_task_field(task, "body", {})
    if not isinstance(body, Mapping):
        return False
    completion = body.get("completion")
    if isinstance(completion, Mapping):
        completion = completion.get("mode") or completion.get("kind")
    review = body.get("review_only")
    return bool(
        str(completion or "").strip().lower() == "manual"
        or review is True
        or str(review or "").strip().lower() in {"1", "true", "yes"}
    )


def database_virgin_transfer_routes(
    tasks: Sequence[Any],
    ready_cids: Iterable[str],
    *,
    shard_count: int,
    task_prefix: str,
) -> Mapping[str, int]:
    """Return the shared deterministic ready-task lane projection."""

    if shard_count <= 1:
        return MappingProxyType({})
    prefix = str(task_prefix or "").strip()
    ready_set = {str(item) for item in ready_cids}
    routes: dict[str, int] = {}
    occupied: set[int] = set()
    ready_tasks: list[Any] = []
    for task in tasks:
        task_cid = str(_database_task_field(task, "task_cid", "") or "")
        task_alias = str(_database_task_field(task, "task_alias", "") or "").strip()
        if not task_cid or not task_alias:
            continue
        binding = database_virgin_transfer_binding_for_task(
            task,
            shard_count=shard_count,
        )
        status = str(_database_task_field(task, "status", "") or "").strip().lower()
        if status == "in_progress" and task_alias.startswith(prefix):
            if binding is not None:
                occupied.add(int(binding["recipient_shard_index"]))
            else:
                receipt = _database_task_status_receipt(task)
                lane = receipt.get("task_shard_index")
                if (
                    receipt.get("operation") == "database_claim"
                    and receipt.get("task_shard_count") == shard_count
                    and isinstance(lane, int)
                    and not isinstance(lane, bool)
                    and 0 <= lane < shard_count
                ):
                    occupied.add(lane)
                else:
                    occupied.add(
                        database_task_alias_home_shard_index(task_alias, shard_count)
                    )
        if (
            task_cid in ready_set
            and task_alias.startswith(prefix)
            and not _database_task_forbids_automatic_claim(task)
        ):
            ready_tasks.append(task)

    def order(task: Any) -> tuple[int, int, str, str]:
        priority = str(_database_task_field(task, "priority", "") or "").upper()
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)
        ordinal = _database_task_field(task, "ordinal", 0)
        ordinal = ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else 0
        return (
            priority_rank,
            int(ordinal),
            str(_database_task_field(task, "task_alias", "") or ""),
            str(_database_task_field(task, "task_cid", "") or ""),
        )

    assigned = set(occupied)
    virgin_by_home: dict[int, list[Any]] = {
        lane: [] for lane in range(shard_count)
    }
    for task in sorted(ready_tasks, key=order):
        task_cid = str(_database_task_field(task, "task_cid", "") or "")
        binding = database_virgin_transfer_binding_for_task(
            task,
            shard_count=shard_count,
        )
        if binding is not None:
            lane = int(binding["recipient_shard_index"])
            routes[task_cid] = lane
            assigned.add(lane)
            continue
        task_alias = str(_database_task_field(task, "task_alias", "") or "")
        home = database_task_alias_home_shard_index(task_alias, shard_count)
        if _database_task_status_receipt(task):
            routes[task_cid] = home
            assigned.add(home)
            continue
        virgin_by_home[home].append(task)

    surplus: list[Any] = []
    for home in range(shard_count):
        candidates = virgin_by_home[home]
        if candidates and home not in assigned:
            retained = candidates.pop(0)
            routes[str(_database_task_field(retained, "task_cid", ""))] = home
            assigned.add(home)
        surplus.extend(candidates)
    free_lanes = [lane for lane in range(shard_count) if lane not in assigned]
    surplus.sort(
        key=lambda task: (
            database_task_alias_home_shard_index(
                str(_database_task_field(task, "task_alias", "") or ""),
                shard_count,
            ),
            *order(task),
        )
    )
    transfer_count = min(len(free_lanes), len(surplus))
    for lane, task in zip(free_lanes, surplus[:transfer_count]):
        routes[str(_database_task_field(task, "task_cid", ""))] = lane
    for task in surplus[transfer_count:]:
        task_alias = str(_database_task_field(task, "task_alias", "") or "")
        routes[str(_database_task_field(task, "task_cid", ""))] = (
            database_task_alias_home_shard_index(task_alias, shard_count)
        )
    return MappingProxyType(routes)


def _database_ready_task_projection_on(
    connection: Any,
    *,
    now_ms: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows = connection.execute(
        """
        SELECT task_cid, task_alias, ordinal, status, revision, priority, body_json
        FROM tasks ORDER BY ordinal, task_cid
        """
    ).fetchall()
    dependencies: dict[str, set[str]] = {}
    for row in connection.execute(
        "SELECT task_cid, dependency_task_cid FROM task_dependencies"
    ).fetchall():
        dependencies.setdefault(str(row[0]), set()).add(str(row[1]))
    completed = {
        str(row[0])
        for row in connection.execute(
            "SELECT task_cid FROM tasks WHERE status IN ("
            + ", ".join("?" for _ in _COMPLETED_STATUSES)
            + ")",
            list(_COMPLETED_STATUSES),
        ).fetchall()
    }
    cooldown = {
        str(row[0]): int(row[1] or 0)
        for row in connection.execute(
            "SELECT task_cid, retry_not_before_ms FROM leases"
        ).fetchall()
    }
    blocked = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT task_cid FROM task_blocks WHERE state = 'active'"
        ).fetchall()
    }
    tasks: list[dict[str, Any]] = []
    ready: set[str] = set()
    for row in rows:
        task = {
            "task_cid": str(row[0]),
            "task_alias": str(row[1]),
            "ordinal": int(row[2]),
            "status": str(row[3]),
            "revision": int(row[4]),
            "priority": str(row[5] or ""),
            "body": _decode_json(row[6], noun="task body"),
        }
        tasks.append(task)
        task_cid = task["task_cid"]
        if (
            task["status"] in _READY_STATUSES
            and task_cid not in blocked
            and cooldown.get(task_cid, 0) <= now_ms
            and dependencies.get(task_cid, set()).issubset(completed)
        ):
            ready.add(task_cid)
    return tasks, ready


def _database_claim_lane(task: Mapping[str, Any], shard_count: int) -> int:
    binding = database_virgin_transfer_binding_for_task(
        task,
        shard_count=shard_count,
    )
    if binding is not None:
        return int(binding["recipient_shard_index"])
    receipt = (
        task.get("body", {}).get("completion_receipt")
        if isinstance(task.get("body"), Mapping)
        else None
    )
    lane = receipt.get("task_shard_index") if isinstance(receipt, Mapping) else None
    count = receipt.get("task_shard_count") if isinstance(receipt, Mapping) else None
    if (
        task.get("status") == "in_progress"
        and isinstance(lane, int)
        and not isinstance(lane, bool)
        and count == shard_count
        and 0 <= int(lane) < shard_count
    ):
        return int(lane)
    return database_task_alias_home_shard_index(
        str(task.get("task_alias") or ""),
        shard_count,
    )


def _prepare_database_virgin_transfer_receipt_on(
    connection: Any,
    *,
    task: Mapping[str, Any],
    previous_status: str,
    current_revision: int,
    new_status: str,
    receipt: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    """Validate/stamp DB virgin transfer inside the owner status transaction."""

    prepared = dict(receipt)
    task_cid = str(task.get("task_cid") or "")
    task_alias = str(task.get("task_alias") or "")
    body = task.get("body")
    prior_receipt = (
        body.get("completion_receipt") if isinstance(body, Mapping) else None
    )
    prior_receipt = prior_receipt if isinstance(prior_receipt, Mapping) else {}
    prior_raw = prior_receipt.get("virgin_task_transfer")
    supplied = prepared.get("virgin_task_transfer")
    supplied_cursor = prepared.get("virgin_task_transfer_claim_cursor")
    if supplied is not None and prior_raw is None:
        raise IntentRepositoryTransitionError(
            "virgin_task_transfer is owner-reserved"
        )
    if supplied_cursor is not None and prior_raw is None:
        raise IntentRepositoryTransitionError(
            "virgin_task_transfer_claim_cursor is owner-reserved"
        )

    prior_binding: Mapping[str, Any] | None = None
    prior_cursor: Mapping[str, Any] | None = None
    prior_count = (
        prior_raw.get("task_shard_count")
        if isinstance(prior_raw, Mapping)
        else None
    )
    if prior_raw is not None:
        if (
            isinstance(prior_count, bool)
            or not isinstance(prior_count, int)
            or prior_count <= 1
        ):
            raise IntentRepositoryTransitionError(
                "stored database virgin-transfer shard count is invalid"
            )
        prior_binding = _database_virgin_transfer_binding(
            task_cid=task_cid,
            task_alias=task_alias,
            receipt=prior_receipt,
            shard_count=int(prior_count),
        )
        if prior_binding is None:
            raise IntentRepositoryTransitionError(
                "stored database virgin-transfer binding is missing"
            )
        prior_cursor = _database_virgin_transfer_claim_cursor(
            receipt=prior_receipt,
            binding=prior_binding,
        )
        if supplied is not None and (
            not isinstance(supplied, Mapping)
            or dict(supplied) != dict(prior_binding or {})
        ):
            raise IntentRepositoryTransitionError(
                "database status CAS would replace a virgin-transfer binding"
            )
        if supplied_cursor is not None and (
            not isinstance(supplied_cursor, Mapping)
            or dict(supplied_cursor) != dict(prior_cursor)
        ):
            raise IntentRepositoryTransitionError(
                "database status CAS would replace a virgin-transfer claim cursor"
            )
        prepared["virgin_task_transfer"] = dict(prior_binding or {})
        prepared["virgin_task_transfer_claim_cursor"] = dict(prior_cursor)

    database_claim = bool(
        new_status == "in_progress"
        and prepared.get("operation") == "database_claim"
    )
    trusted_policy = _trusted_database_claim_policy()
    if (
        trusted_policy is not None
        and new_status == "in_progress"
        and not database_claim
    ):
        raise IntentRepositoryTransitionError(
            "trusted database claim policy requires database_claim"
        )
    if trusted_policy is not None and database_claim:
        if any(
            prepared.get(name) != trusted_policy.get(name)
            for name in (
                "task_prefix",
                "task_shard_count",
                "strict_task_sharding",
                "idle_lane_work_stealing",
            )
        ):
            raise IntentRepositoryTransitionError(
                "database claim disagrees with the trusted store policy"
            )
    transfer_claim = bool(
        database_claim
        and prepared.get("idle_lane_work_stealing")
        == DATABASE_VIRGIN_TASK_TRANSFER_MODE
    )
    if prior_binding is not None and new_status == "in_progress" and not transfer_claim:
        raise IntentRepositoryTransitionError(
            "database virgin-transfer retry requires its bound claim policy"
        )
    if not transfer_claim:
        if database_claim and prepared.get("strict_task_sharding") is True:
            shard_count = prepared.get("task_shard_count")
            lane_index = prepared.get("task_shard_index")
            task_prefix = str(prepared.get("task_prefix") or "")
            if (
                prepared.get("idle_lane_work_stealing") not in {None, ""}
                or isinstance(shard_count, bool)
                or not isinstance(shard_count, int)
                or shard_count < 1
                or isinstance(lane_index, bool)
                or not isinstance(lane_index, int)
                or not 0 <= lane_index < shard_count
                or (task_prefix and not task_alias.startswith(task_prefix))
                or lane_index
                != database_task_alias_home_shard_index(task_alias, shard_count)
            ):
                raise IntentRepositoryTransitionError(
                    "strict database claim does not match its home shard"
                )
        prepared.pop("virgin_task_transfer_request", None)
        return prepared

    shard_count = prepared.get("task_shard_count")
    lane_index = prepared.get("task_shard_index")
    task_prefix = str(prepared.get("task_prefix") or "")
    if (
        prepared.get("strict_task_sharding") is not True
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 1
        or isinstance(lane_index, bool)
        or not isinstance(lane_index, int)
        or not 0 <= lane_index < shard_count
        or not task_prefix
        or not task_alias.startswith(task_prefix)
        or prepared.get("claimed_from_revision") != current_revision
        or not str(prepared.get("claim_id") or "")
        or not str(prepared.get("attempt_id") or "")
        or not str(prepared.get("owner_session_id") or "")
        or not str(prepared.get("lease_id") or "")
        or isinstance(prepared.get("fencing_token"), bool)
        or not isinstance(prepared.get("fencing_token"), int)
        or int(prepared.get("fencing_token") or 0) < 1
        or isinstance(prepared.get("fence_epoch"), bool)
        or not isinstance(prepared.get("fence_epoch"), int)
        or int(prepared.get("fence_epoch") or 0) < 1
    ):
        raise IntentRepositoryTransitionError(
            "database virgin-transfer claim metadata is invalid"
        )
    # The typed owner can rotate an expired in-progress claim only after it
    # has independently admitted a newer live fence.  That task is no longer
    # part of the ready projection, so validate a home-lane retry directly or
    # retain its already owner-stamped foreign lane instead of trying to
    # authorize a second transfer.  The binding and cursor checks below still
    # require the exact lane, owner, and monotone claim/fence tuple.  The
    # ordinary repository never reaches this branch: its same-status CAS is a
    # no-op before receipt preparation.
    bound_in_progress_retry = bool(
        previous_status == "in_progress" and new_status == "in_progress"
    )
    tasks: list[dict[str, Any]] = []
    ready_cids: set[str] = set()
    if not bound_in_progress_retry:
        tasks, ready_cids = _database_ready_task_projection_on(
            connection,
            now_ms=now_ms,
        )
        by_cid = {str(item["task_cid"]): item for item in tasks}
        if task_cid not in ready_cids:
            raise IntentRepositoryConflictError(
                "virgin-transfer target left the authoritative ready frontier"
            )
        routes = database_virgin_transfer_routes(
            tasks,
            ready_cids,
            shard_count=shard_count,
            task_prefix=task_prefix,
        )
        if routes.get(task_cid) != lane_index:
            raise IntentRepositoryConflictError(
                "virgin-transfer request disagrees with the authoritative route"
            )
    if prior_binding is not None:
        if (
            int(prior_binding["task_shard_count"]) != shard_count
            or int(prior_binding["recipient_shard_index"]) != lane_index
            or prior_binding["owner_session_id"]
            != prepared["owner_session_id"]
        ):
            raise IntentRepositoryTransitionError(
                "database virgin-transfer retry changed its assigned lane"
            )
        prepared.pop("virgin_task_transfer_request", None)
        _database_virgin_transfer_binding(
            task_cid=task_cid,
            task_alias=task_alias,
            receipt=prepared,
            shard_count=shard_count,
        )
        if prior_cursor is None or not (
            int(prepared["claimed_from_revision"])
            > int(prior_cursor["claimed_from_revision"])
            and all(
                prepared.get(name) != prior_cursor.get(name)
                for name in ("claim_id", "attempt_id", "lease_id")
            )
            and int(prepared["fencing_token"])
            > int(prior_cursor["fencing_token"])
            and int(prepared["fence_epoch"])
            >= int(prior_cursor["fence_epoch"])
        ):
            raise IntentRepositoryTransitionError(
                "database virgin-transfer retry did not advance its claim cursor"
            )
        prepared["virgin_task_transfer_claim_cursor"] = (
            _database_virgin_transfer_claim_cursor_body(
                binding=prior_binding,
                receipt=prepared,
            )
        )
        return prepared

    home_lane = database_task_alias_home_shard_index(task_alias, shard_count)
    request = prepared.pop("virgin_task_transfer_request", None)
    if lane_index == home_lane:
        if request is not None:
            raise IntentRepositoryTransitionError(
                "home-shard claim cannot request virgin transfer"
            )
        return prepared
    if not isinstance(request, Mapping) or dict(request) != {
        "schema": DATABASE_VIRGIN_TASK_TRANSFER_REQUEST_SCHEMA,
        "mode": DATABASE_VIRGIN_TASK_TRANSFER_MODE,
        "task_shard_count": shard_count,
        "recipient_shard_index": lane_index,
        "task_prefix": task_prefix,
    }:
        raise IntentRepositoryTransitionError(
            "foreign database claim lacks an exact transfer request"
        )
    if prior_receipt or previous_status not in _READY_STATUSES:
        raise IntentRepositoryTransitionError(
            "only a virgin ready task may transfer lanes"
        )
    completion = body.get("completion") if isinstance(body, Mapping) else None
    if isinstance(completion, Mapping):
        completion = completion.get("mode") or completion.get("kind")
    review_only = body.get("review_only") if isinstance(body, Mapping) else None
    if (
        str(completion or "").strip().lower() == "manual"
        or review_only is True
        or str(review_only or "").strip().lower() in {"1", "true", "yes"}
    ):
        raise IntentRepositoryTransitionError(
            "manual or review-only task cannot transfer lanes"
        )

    active_lanes = {
        _database_claim_lane(item, shard_count)
        for item in tasks
        if item.get("status") == "in_progress"
        and str(item.get("task_alias") or "").startswith(task_prefix)
    }
    ready_home_counts: dict[int, int] = {}
    ready_transfer_lanes: set[int] = set()
    for ready_cid in ready_cids:
        item = by_cid[ready_cid]
        alias = str(item.get("task_alias") or "")
        if (
            not alias.startswith(task_prefix)
            or _database_task_forbids_automatic_claim(item)
        ):
            continue
        binding = database_virgin_transfer_binding_for_task(
            item,
            shard_count=shard_count,
        )
        if binding is not None:
            ready_transfer_lanes.add(int(binding["recipient_shard_index"]))
            continue
        lane = database_task_alias_home_shard_index(alias, shard_count)
        ready_home_counts[lane] = ready_home_counts.get(lane, 0) + 1
    if (
        lane_index in active_lanes
        or lane_index in ready_transfer_lanes
        or ready_home_counts.get(lane_index, 0)
    ):
        raise IntentRepositoryConflictError(
            "virgin-transfer recipient is not idle"
        )
    donor_active = home_lane in active_lanes
    donor_ready_count = ready_home_counts.get(home_lane, 0)
    if not donor_active and donor_ready_count < 2:
        raise IntentRepositoryConflictError(
            "virgin-transfer donor has no surplus task"
        )

    projection_id = content_identity(
        {
            "schema": "database-virgin-task-transfer-frontier@1",
            "task_shard_count": shard_count,
            "tasks": [
                {
                    "task_cid": item["task_cid"],
                    "task_alias": item["task_alias"],
                    "status": item["status"],
                    "revision": item["revision"],
                }
                for item in tasks
            ],
            "ready_task_cids": sorted(ready_cids),
        }
    )
    claim_policy_id = (
        content_identity(dict(trusted_policy))
        if trusted_policy is not None
        else ""
    )
    store_generation = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "") or ""
    ).strip()
    binding_body = {
        "schema": DATABASE_VIRGIN_TASK_TRANSFER_BINDING_SCHEMA,
        "mode": DATABASE_VIRGIN_TASK_TRANSFER_MODE,
        "cohort_id": content_identity(
            {
                "kind": "database-virgin-task-transfer-cohort",
                "task_prefix": task_prefix,
                "task_shard_count": shard_count,
                "claim_policy_id": claim_policy_id,
                "store_generation": store_generation,
            }
        ),
        "claim_policy_id": claim_policy_id,
        "store_generation": store_generation,
        "task_cid": task_cid,
        "task_alias": task_alias,
        "task_prefix": task_prefix,
        "task_shard_count": shard_count,
        "home_shard_index": home_lane,
        "recipient_shard_index": lane_index,
        "source_task_revision": current_revision,
        "claim_id": str(prepared["claim_id"]),
        "attempt_id": str(prepared["attempt_id"]),
        "owner_session_id": str(prepared["owner_session_id"]),
        "lease_id": str(prepared["lease_id"]),
        "fencing_token": int(prepared["fencing_token"]),
        "fence_epoch": int(prepared["fence_epoch"]),
        "preclaim_projection_id": projection_id,
        "donor_active": donor_active,
        "donor_ready_count": donor_ready_count,
    }
    prepared["virgin_task_transfer"] = {
        **binding_body,
        "binding_id": content_identity(binding_body),
    }
    prepared["virgin_task_transfer_claim_cursor"] = (
        _database_virgin_transfer_claim_cursor_body(
            binding=prepared["virgin_task_transfer"],
            receipt=prepared,
        )
    )
    return prepared


def _mapping(value: Any, *, noun: str = "mapping") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise IntentRepositoryError(f"{noun} must be a mapping")
    return {str(key): member for key, member in value.items()}


def _bounded_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > MAX_PAGE_LIMIT:
        raise IntentRepositoryBoundsError(f"limit must be in [1, {MAX_PAGE_LIMIT}]")
    return limit


def _nonneg_int(value: Any, *, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntentRepositoryBoundsError(f"{noun} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IntentRepositoryBoundsError(f"{noun} must be a positive integer")
    return value


def _projection_sequence(
    value: Any,
    *,
    noun: str,
    maximum: int,
) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise IntentRepositoryError(f"{noun} must be a sequence")
    items = list(value)
    if len(items) > maximum:
        raise IntentRepositoryBoundsError(f"{noun} count exceeds bound")
    return items


def _projection_task_cids(task_cids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(task_cids, (str, bytes, bytearray)) or not isinstance(task_cids, Sequence):
        raise IntentRepositoryError("task_cids must be a sequence")
    if len(task_cids) > MAX_PAGE_LIMIT:
        raise IntentRepositoryBoundsError("projection task count exceeds bound")
    resolved = tuple(_identifier(task_cid, noun="task_cid") for task_cid in task_cids)
    if len(set(resolved)) != len(resolved):
        raise IntentRepositoryIntegrityError("projection task_cids must not contain duplicates")
    return tuple(sorted(resolved))


def _task_projection_spec(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the semantic/operational specification of one task.

    Lifecycle state, revision counters, timestamps, and the containing plan are
    intentionally absent.  This makes the resulting identity suitable for CAS
    checks that distinguish a task-specification edit from a mere claim or
    completion transition.  The containing full plan projection still binds
    all of those lifecycle fields independently.
    """

    task = _mapping(record, noun="task projection record")
    task_cid = _identifier(task.get("task_cid"), noun="task_cid")
    task_alias = _identifier(task.get("task_alias") or task.get("task_id"), noun="task_alias")
    goal_cid = _identifier(task.get("goal_cid"), noun="goal_cid")
    objective_id = _optional_identifier(task.get("objective_id"), noun="objective_id")

    dependencies: list[dict[str, str]] = []
    for raw in _projection_sequence(
        task.get("dependencies"),
        noun="task dependencies",
        maximum=MAX_DEPENDENCIES,
    ):
        if isinstance(raw, Mapping):
            dependency = _mapping(raw, noun="task dependency")
            dependency_cid = _identifier(
                dependency.get("dependency_task_cid") or dependency.get("task_cid"),
                noun="dependency_task_cid",
            )
            kind = _identifier(dependency.get("kind") or "depends_on", noun="dependency kind")
        else:
            dependency_cid = _identifier(raw, noun="dependency_task_cid")
            kind = "depends_on"
        dependencies.append({"dependency_task_cid": dependency_cid, "kind": kind})
    dependencies.sort(key=lambda item: (item["dependency_task_cid"], item["kind"]))

    outputs: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _projection_sequence(task.get("outputs"), noun="task outputs", maximum=MAX_OUTPUTS)
    ):
        output = _mapping(raw, noun="task output")
        outputs.append(
            {
                "ordinal": _nonneg_int(output.get("ordinal", index), noun="output ordinal"),
                "path": _identifier(output.get("path"), noun="output path"),
                "effect": _jsonable(output.get("effect", {})),
            }
        )
    outputs.sort(key=lambda item: (item["ordinal"], item["path"]))

    acceptance: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _projection_sequence(
            task.get("acceptance"),
            noun="task acceptance",
            maximum=MAX_ACCEPTANCE,
        )
    ):
        item = _mapping(raw, noun="task acceptance entry")
        criterion = str(item.get("criterion") or "").strip()
        if not criterion:
            raise IntentRepositoryError("acceptance criterion must not be empty")
        acceptance.append(
            {
                "ordinal": _nonneg_int(item.get("ordinal", index), noun="acceptance ordinal"),
                "criterion": criterion,
                "evidence_policy": _jsonable(item.get("evidence_policy", {})),
            }
        )
    acceptance.sort(key=lambda item: item["ordinal"])

    validations: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _projection_sequence(
            task.get("validations"),
            noun="task validations",
            maximum=MAX_VALIDATIONS,
        )
    ):
        item = _mapping(raw, noun="task validation entry")
        argv = _projection_sequence(
            item.get("argv"), noun="validation argv", maximum=MAX_BODY_BYTES
        )
        validations.append(
            {
                "ordinal": _nonneg_int(item.get("ordinal", index), noun="validation ordinal"),
                "argv": [str(part) for part in argv],
                "policy": _jsonable(item.get("policy", {})),
            }
        )
    validations.sort(key=lambda item: item["ordinal"])

    return {
        "task_cid": task_cid,
        "task_alias": task_alias,
        "goal_cid": goal_cid,
        "objective_id": objective_id,
        "ordinal": _nonneg_int(task.get("ordinal", 0), noun="task ordinal"),
        "priority": str(task.get("priority") or ""),
        "identity": _jsonable(task.get("identity", {})),
        "body": _jsonable(task.get("body", {})),
        "extension_schema": str(task.get("extension_schema") or ""),
        "extension": _jsonable(task.get("extension", {})),
        "dependencies": dependencies,
        "outputs": outputs,
        "acceptance": acceptance,
        "validations": validations,
    }


def task_projection_spec_cid(record: Mapping[str, Any]) -> str:
    """Return the stable CID of a task's complete non-lifecycle specification."""

    material = {
        "schema": TASK_PROJECTION_SPEC_SCHEMA,
        "task": _task_projection_spec(record),
    }
    encoded = canonical_json_bytes(material)
    if len(encoded) > MAX_TASK_PROJECTION_BYTES:
        raise IntentRepositoryBoundsError("task projection spec exceeds byte bound")
    return content_identity(material)


def task_authority_spec_cid(record: Mapping[str, Any]) -> str:
    """Return the CID of the immutable, authority-bearing task specification.

    ``IntentRepository@1`` historically stores the latest status-transition
    receipt in ``body.completion_receipt`` so retry workers can recover an
    exact seed.  That receipt is operational lifecycle evidence: replacing it
    through an admitted status CAS must not look like a plan amendment.  Every
    other body field remains authority-bearing.  The legacy
    :func:`task_projection_spec_cid` is intentionally unchanged because its
    CIDs are already persisted in plan-revision receipts.
    """

    normalized = _task_projection_spec(record)
    body = normalized.get("body")
    if isinstance(body, dict):
        body = dict(body)
        body.pop("completion_receipt", None)
        normalized["body"] = body
    material = {
        "schema": TASK_AUTHORITY_SPEC_SCHEMA,
        "task": normalized,
    }
    encoded = canonical_json_bytes(material)
    if len(encoded) > MAX_TASK_PROJECTION_BYTES:
        raise IntentRepositoryBoundsError("task authority spec exceeds byte bound")
    return content_identity(material)


def _content_addressed_projection(
    material: Mapping[str, Any],
    *,
    maximum_bytes: int,
    noun: str,
) -> Mapping[str, Any]:
    normalized = _jsonable(material)
    encoded = canonical_json_bytes(normalized)
    if len(encoded) > maximum_bytes:
        raise IntentRepositoryBoundsError(f"{noun} exceeds byte bound")
    return MappingProxyType({**normalized, "projection_cid": content_identity(normalized)})


def _goal_completion_authority_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one closed, content-addressed goal-completion population.

    The caller that constructs this specification is the state owner.  This
    validator deliberately accepts no inferred goals, edges, or task aliases:
    a population mismatch is an integrity error, not an invitation to expand
    completion authority.
    """

    raw = _mapping(value, noun="goal completion authority specification")
    expected_fields = {
        "schema",
        "board_namespace",
        "goal_count",
        "task_count",
        "root_goal_cid",
        "root_goal_alias",
        "goals",
        "goal_edges",
        "tasks",
        "task_dependencies",
        "terminal_report_contract",
        "completion_policy",
        "receipt_backfill_goal_cids",
        "authority_spec_id",
    }
    if set(raw) != expected_fields or raw.get("schema") != GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA:
        raise IntentRepositoryIntegrityError(
            "goal completion authority specification has a non-closed schema"
        )
    authority_spec_id = str(raw.get("authority_spec_id") or "")
    identity_body = dict(raw)
    identity_body.pop("authority_spec_id", None)
    if authority_spec_id != content_identity(identity_body):
        raise IntentRepositoryIntegrityError(
            "goal completion authority specification identity is invalid"
        )
    board_namespace = _identifier(raw.get("board_namespace"), noun="board_namespace")
    goal_count = _positive_int(raw.get("goal_count"), noun="goal_count")
    task_count = _positive_int(raw.get("task_count"), noun="task_count")
    if goal_count > MAX_PAGE_LIMIT or task_count > MAX_PAGE_LIMIT:
        raise IntentRepositoryBoundsError("goal completion authority population exceeds bound")

    goal_items = _projection_sequence(
        raw.get("goals"), noun="goal authority goals", maximum=MAX_PAGE_LIMIT
    )
    if len(goal_items) != goal_count:
        raise IntentRepositoryIntegrityError("goal authority goal count is not exact")
    goals: list[dict[str, Any]] = []
    goal_cids: set[str] = set()
    goal_aliases: set[str] = set()
    for item in goal_items:
        goal = _mapping(item, noun="goal authority goal")
        if set(goal) != {
            "goal_cid",
            "goal_alias",
            "parent_goal_cid",
            "ordinal",
        }:
            raise IntentRepositoryIntegrityError("goal authority goal schema is not closed")
        normalized = {
            "goal_cid": _identifier(goal.get("goal_cid"), noun="goal_cid"),
            "goal_alias": _identifier(goal.get("goal_alias"), noun="goal_alias"),
            "parent_goal_cid": _optional_identifier(
                goal.get("parent_goal_cid"), noun="parent_goal_cid"
            ),
            "ordinal": _nonneg_int(goal.get("ordinal"), noun="goal ordinal"),
        }
        if (
            normalized["goal_cid"] in goal_cids
            or normalized["goal_alias"] in goal_aliases
        ):
            raise IntentRepositoryIntegrityError(
                "goal authority identities or aliases are duplicated"
            )
        goal_cids.add(normalized["goal_cid"])
        goal_aliases.add(normalized["goal_alias"])
        goals.append(normalized)
    goals.sort(key=lambda item: (item["ordinal"], item["goal_alias"], item["goal_cid"]))

    root_goal_cid = _identifier(raw.get("root_goal_cid"), noun="root_goal_cid")
    root_goal_alias = _identifier(raw.get("root_goal_alias"), noun="root_goal_alias")
    roots = [item for item in goals if not item["parent_goal_cid"]]
    if (
        len(roots) != 1
        or roots[0]["goal_cid"] != root_goal_cid
        or roots[0]["goal_alias"] != root_goal_alias
    ):
        raise IntentRepositoryIntegrityError("goal authority root identity is not exact")
    for goal in goals:
        parent = goal["parent_goal_cid"]
        if parent and parent not in goal_cids:
            raise IntentRepositoryIntegrityError("goal authority parent is unknown")

    edge_items = _projection_sequence(
        raw.get("goal_edges"), noun="goal authority edges", maximum=MAX_DEPENDENCIES
    )
    edges: list[dict[str, str]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    for item in edge_items:
        edge = _mapping(item, noun="goal authority edge")
        if set(edge) != {"parent_goal_cid", "child_goal_cid", "edge_kind"}:
            raise IntentRepositoryIntegrityError("goal authority edge schema is not closed")
        normalized_edge = {
            "parent_goal_cid": _identifier(
                edge.get("parent_goal_cid"), noun="edge parent_goal_cid"
            ),
            "child_goal_cid": _identifier(
                edge.get("child_goal_cid"), noun="edge child_goal_cid"
            ),
            "edge_kind": _identifier(edge.get("edge_kind"), noun="edge_kind"),
        }
        if normalized_edge["edge_kind"] not in {"goal_parent", "goal_dependency"}:
            raise IntentRepositoryIntegrityError("goal authority edge kind is not admitted")
        if (
            normalized_edge["parent_goal_cid"] not in goal_cids
            or normalized_edge["child_goal_cid"] not in goal_cids
            or normalized_edge["parent_goal_cid"] == normalized_edge["child_goal_cid"]
        ):
            raise IntentRepositoryIntegrityError("goal authority edge endpoints are invalid")
        key = (
            normalized_edge["parent_goal_cid"],
            normalized_edge["child_goal_cid"],
            normalized_edge["edge_kind"],
        )
        if key in edge_keys:
            raise IntentRepositoryIntegrityError("goal authority edges are duplicated")
        edge_keys.add(key)
        edges.append(normalized_edge)
    edges.sort(
        key=lambda item: (
            item["edge_kind"],
            item["parent_goal_cid"],
            item["child_goal_cid"],
        )
    )
    declared_parent_edges = {
        (item["parent_goal_cid"], item["goal_cid"], "goal_parent")
        for item in goals
        if item["parent_goal_cid"]
    }
    observed_parent_edges = {
        key for key in edge_keys if key[2] == "goal_parent"
    }
    if declared_parent_edges != observed_parent_edges:
        raise IntentRepositoryIntegrityError(
            "goal authority parent fields and edges disagree"
        )

    task_items = _projection_sequence(
        raw.get("tasks"), noun="goal authority tasks", maximum=MAX_PAGE_LIMIT
    )
    if len(task_items) != task_count:
        raise IntentRepositoryIntegrityError("goal authority task count is not exact")
    tasks: list[dict[str, str]] = []
    task_cids: set[str] = set()
    task_aliases: set[str] = set()
    for item in task_items:
        task = _mapping(item, noun="goal authority task")
        if set(task) != {"task_cid", "task_alias", "goal_cid"}:
            raise IntentRepositoryIntegrityError("goal authority task schema is not closed")
        normalized_task = {
            "task_cid": _identifier(task.get("task_cid"), noun="task_cid"),
            "task_alias": _identifier(task.get("task_alias"), noun="task_alias"),
            "goal_cid": _identifier(task.get("goal_cid"), noun="task goal_cid"),
        }
        if normalized_task["goal_cid"] not in goal_cids:
            raise IntentRepositoryIntegrityError("goal authority task owns an unknown goal")
        if (
            normalized_task["task_cid"] in task_cids
            or normalized_task["task_alias"] in task_aliases
        ):
            raise IntentRepositoryIntegrityError(
                "goal authority task identities or aliases are duplicated"
            )
        task_cids.add(normalized_task["task_cid"])
        task_aliases.add(normalized_task["task_alias"])
        tasks.append(normalized_task)
    tasks.sort(key=lambda item: (item["task_alias"], item["task_cid"]))

    dependency_items = _projection_sequence(
        raw.get("task_dependencies"),
        noun="goal authority task dependencies",
        maximum=MAX_DEPENDENCIES,
    )
    task_dependencies: list[dict[str, str]] = []
    task_dependency_keys: set[tuple[str, str, str]] = set()
    for item in dependency_items:
        dependency = _mapping(item, noun="goal authority task dependency")
        if set(dependency) != {"task_cid", "dependency_task_cid", "kind"}:
            raise IntentRepositoryIntegrityError(
                "goal authority task dependency schema is not closed"
            )
        normalized_dependency = {
            "task_cid": _identifier(dependency.get("task_cid"), noun="task_cid"),
            "dependency_task_cid": _identifier(
                dependency.get("dependency_task_cid"),
                noun="dependency_task_cid",
            ),
            "kind": _identifier(dependency.get("kind"), noun="task dependency kind"),
        }
        if normalized_dependency["kind"] != "depends_on":
            raise IntentRepositoryIntegrityError(
                "goal authority task dependency kind is not admitted"
            )
        if (
            normalized_dependency["task_cid"] not in task_cids
            or normalized_dependency["dependency_task_cid"] not in task_cids
            or normalized_dependency["task_cid"]
            == normalized_dependency["dependency_task_cid"]
        ):
            raise IntentRepositoryIntegrityError(
                "goal authority task dependency endpoints are invalid"
            )
        dependency_key = (
            normalized_dependency["task_cid"],
            normalized_dependency["dependency_task_cid"],
            normalized_dependency["kind"],
        )
        if dependency_key in task_dependency_keys:
            raise IntentRepositoryIntegrityError(
                "goal authority task dependencies are duplicated"
            )
        task_dependency_keys.add(dependency_key)
        task_dependencies.append(normalized_dependency)
    task_dependencies.sort(
        key=lambda item: (
            item["task_cid"],
            item["dependency_task_cid"],
            item["kind"],
        )
    )

    task_prerequisites: dict[str, set[str]] = {
        task_cid: set() for task_cid in task_cids
    }
    for dependency in task_dependencies:
        task_prerequisites[dependency["task_cid"]].add(
            dependency["dependency_task_cid"]
        )
    remaining_tasks = {
        task_cid: set(prerequisites)
        for task_cid, prerequisites in task_prerequisites.items()
    }
    while remaining_tasks:
        ready_tasks = sorted(
            task_cid
            for task_cid, prerequisites in remaining_tasks.items()
            if not prerequisites
        )
        if not ready_tasks:
            raise IntentRepositoryIntegrityError(
                "goal authority task dependency graph contains a cycle"
            )
        for task_cid in ready_tasks:
            remaining_tasks.pop(task_cid)
        for prerequisites in remaining_tasks.values():
            prerequisites.difference_update(ready_tasks)

    child_goals = {edge["parent_goal_cid"] for edge in edges if edge["edge_kind"] == "goal_parent"}
    tasks_by_goal = {goal_cid: 0 for goal_cid in goal_cids}
    for task in tasks:
        tasks_by_goal[task["goal_cid"]] += 1
    for goal_cid, population in tasks_by_goal.items():
        if goal_cid in child_goals and population:
            raise IntentRepositoryIntegrityError(
                "goal authority parent goals cannot also own direct tasks"
            )
        if goal_cid not in child_goals and population < 1:
            raise IntentRepositoryIntegrityError(
                "goal authority leaf goals require direct task evidence"
            )

    completion_policy = _mapping(raw.get("completion_policy"), noun="completion_policy")
    if set(completion_policy) != {
        *_ROOT_COMPLETION_POLICY_FIELDS,
        _ROOT_TERMINAL_TASK_POLICY_FIELD,
    } or any(
        completion_policy.get(field) is not True for field in _ROOT_COMPLETION_POLICY_FIELDS
    ):
        raise IntentRepositoryIntegrityError(
            "goal authority completion policy is not the exact fail-closed policy"
        )
    terminal_task_alias = _identifier(
        completion_policy.get(_ROOT_TERMINAL_TASK_POLICY_FIELD),
        noun="completion_policy.terminal_task_id",
    )
    terminal_tasks = [item for item in tasks if item["task_alias"] == terminal_task_alias]
    if len(terminal_tasks) != 1:
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report task is not an exact task binding"
        )
    terminal_task = terminal_tasks[0]
    raw_terminal_contract = _mapping(
        raw.get("terminal_report_contract"), noun="terminal report contract"
    )
    terminal_contract_fields = {
        "schema",
        "task_cid",
        "task_alias",
        "declared_output_paths",
        "declared_symbols",
        "required_report_paths",
        "producer_output_paths",
        "producer_validation_commands",
        "acceptance_criteria",
        "validation_commands",
        "contract_id",
    }
    if (
        set(raw_terminal_contract) != terminal_contract_fields
        or raw_terminal_contract.get("schema")
        != GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA
        or raw_terminal_contract.get("task_cid") != terminal_task["task_cid"]
        or raw_terminal_contract.get("task_alias") != terminal_task_alias
    ):
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report contract identity is not exact"
        )
    declared_output_paths = [
        _identifier(item, noun="terminal declared output path")
        for item in _projection_sequence(
            raw_terminal_contract.get("declared_output_paths"),
            noun="terminal declared output paths",
            maximum=MAX_OUTPUTS,
        )
    ]
    required_report_paths = [
        _identifier(item, noun="terminal required report path")
        for item in _projection_sequence(
            raw_terminal_contract.get("required_report_paths"),
            noun="terminal required report paths",
            maximum=MAX_OUTPUTS,
        )
    ]
    declared_symbols = [
        _identifier(item, noun="terminal declared symbol")
        for item in _projection_sequence(
            raw_terminal_contract.get("declared_symbols"),
            noun="terminal declared symbols",
            maximum=MAX_OUTPUTS,
        )
    ]
    if (
        len(set(declared_output_paths)) != len(declared_output_paths)
        or not declared_symbols
        or len(set(declared_symbols)) != len(declared_symbols)
        or len(set(required_report_paths)) != len(required_report_paths)
        or len(required_report_paths) != 2
        or not set(required_report_paths).issubset(declared_output_paths)
        or {str(Path(item).suffix).lower() for item in required_report_paths}
        != {".json", ".md"}
    ):
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report output contract is invalid"
        )
    task_alias_by_cid = {item["task_cid"]: item["task_alias"] for item in tasks}
    terminal_producer_cids = sorted(
        {
            edge["dependency_task_cid"]
            for edge in task_dependencies
            if edge["task_cid"] == terminal_task["task_cid"]
        },
        key=lambda task_cid: task_alias_by_cid[task_cid],
    )
    terminal_producer_aliases = [
        task_alias_by_cid[task_cid] for task_cid in terminal_producer_cids
    ]
    raw_producer_outputs = _mapping(
        raw_terminal_contract.get("producer_output_paths"),
        noun="terminal report producer output paths",
    )
    if len(terminal_producer_aliases) != 4 or set(raw_producer_outputs) != set(
        terminal_producer_aliases
    ):
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report producer population is not exact"
        )
    producer_output_paths: dict[str, list[str]] = {}
    all_producer_paths: set[str] = set()
    for task_alias in terminal_producer_aliases:
        paths = [
            _identifier(item, noun="terminal report producer output path")
            for item in _projection_sequence(
                raw_producer_outputs.get(task_alias),
                noun="terminal report producer output paths",
                maximum=MAX_OUTPUTS,
            )
        ]
        if (
            not paths
            or len(paths) != len(set(paths))
            or any(path in all_producer_paths for path in paths)
            or any(path in declared_output_paths for path in paths)
        ):
            raise IntentRepositoryIntegrityError(
                "goal authority terminal report producer output ownership is invalid"
            )
        all_producer_paths.update(paths)
        producer_output_paths[task_alias] = paths
    raw_producer_validations = _mapping(
        raw_terminal_contract.get("producer_validation_commands"),
        noun="terminal report producer validation commands",
    )
    if set(raw_producer_validations) != set(terminal_producer_aliases):
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report producer validation population is not exact"
        )
    producer_validation_commands: dict[str, list[list[str]]] = {}
    for task_alias in terminal_producer_aliases:
        commands: list[list[str]] = []
        for raw_command in _projection_sequence(
            raw_producer_validations.get(task_alias),
            noun="terminal report producer validation commands",
            maximum=MAX_VALIDATIONS,
        ):
            command = [
                str(part)
                for part in _projection_sequence(
                    raw_command,
                    noun="terminal report producer validation argv",
                    maximum=MAX_BODY_BYTES,
                )
            ]
            if not command or any(not part for part in command):
                raise IntentRepositoryIntegrityError(
                    "terminal report producer validation command is empty"
                )
            commands.append(command)
        if not commands:
            raise IntentRepositoryIntegrityError(
                "terminal report producer validation contract is absent"
            )
        producer_validation_commands[task_alias] = commands
    acceptance_criteria: list[str] = []
    for item in _projection_sequence(
        raw_terminal_contract.get("acceptance_criteria"),
        noun="terminal acceptance criteria",
        maximum=MAX_ACCEPTANCE,
    ):
        criterion = str(item or "").strip()
        if not criterion:
            raise IntentRepositoryIntegrityError(
                "goal authority terminal acceptance criterion is empty"
            )
        acceptance_criteria.append(criterion)
    if not acceptance_criteria:
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report acceptance contract is absent"
        )
    validation_commands: list[list[str]] = []
    for item in _projection_sequence(
        raw_terminal_contract.get("validation_commands"),
        noun="terminal validation commands",
        maximum=MAX_VALIDATIONS,
    ):
        command = [
            str(part)
            for part in _projection_sequence(
                item,
                noun="terminal validation command argv",
                maximum=MAX_BODY_BYTES,
            )
        ]
        if not command or any(not part for part in command):
            raise IntentRepositoryIntegrityError(
                "goal authority terminal validation command is empty"
            )
        validation_commands.append(command)
    if not validation_commands:
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report validation contract is absent"
        )
    terminal_contract = {
        "schema": GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA,
        "task_cid": terminal_task["task_cid"],
        "task_alias": terminal_task_alias,
        "declared_output_paths": declared_output_paths,
        "declared_symbols": declared_symbols,
        "required_report_paths": required_report_paths,
        "producer_output_paths": producer_output_paths,
        "producer_validation_commands": producer_validation_commands,
        "acceptance_criteria": acceptance_criteria,
        "validation_commands": validation_commands,
    }
    terminal_contract["contract_id"] = content_identity(terminal_contract)
    if raw_terminal_contract.get("contract_id") != terminal_contract["contract_id"]:
        raise IntentRepositoryIntegrityError(
            "goal authority terminal report contract identity is invalid"
        )
    backfill_items = _projection_sequence(
        raw.get("receipt_backfill_goal_cids"),
        noun="goal receipt backfill identities",
        maximum=MAX_PAGE_LIMIT,
    )
    backfills = sorted(
        {_identifier(item, noun="receipt backfill goal_cid") for item in backfill_items}
    )
    if len(backfills) != len(backfill_items) or not set(backfills).issubset(goal_cids):
        raise IntentRepositoryIntegrityError("goal receipt backfill allowlist is invalid")

    prerequisites: dict[str, set[str]] = {goal_cid: set() for goal_cid in goal_cids}
    for edge in edges:
        if edge["edge_kind"] == "goal_parent":
            prerequisites[edge["parent_goal_cid"]].add(edge["child_goal_cid"])
        else:
            prerequisites[edge["child_goal_cid"]].add(edge["parent_goal_cid"])
    remaining = {key: set(value) for key, value in prerequisites.items()}
    ordered: list[str] = []
    ordinal_by_cid = {item["goal_cid"]: item["ordinal"] for item in goals}
    alias_by_cid = {item["goal_cid"]: item["goal_alias"] for item in goals}
    while remaining:
        ready = sorted(
            (goal_cid for goal_cid, dependencies in remaining.items() if not dependencies),
            key=lambda goal_cid: (
                ordinal_by_cid[goal_cid],
                alias_by_cid[goal_cid],
                goal_cid,
            ),
        )
        if not ready:
            raise IntentRepositoryIntegrityError("goal authority graph contains a cycle")
        for goal_cid in ready:
            ordered.append(goal_cid)
            remaining.pop(goal_cid)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)

    return {
        "schema": GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA,
        "board_namespace": board_namespace,
        "goal_count": goal_count,
        "task_count": task_count,
        "root_goal_cid": root_goal_cid,
        "root_goal_alias": root_goal_alias,
        "goals": goals,
        "goal_edges": edges,
        "tasks": tasks,
        "task_dependencies": task_dependencies,
        "terminal_report_contract": terminal_contract,
        "completion_policy": {
            **{field: True for field in _ROOT_COMPLETION_POLICY_FIELDS},
            _ROOT_TERMINAL_TASK_POLICY_FIELD: terminal_task_alias,
        },
        "terminal_task_cid": terminal_task["task_cid"],
        "receipt_backfill_goal_cids": backfills,
        "authority_spec_id": authority_spec_id,
        "topological_goal_cids": ordered,
    }


def _database_portal_completion_binding(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the compact Portal-to-canonical completion lineage binding."""

    binding = _mapping(value, noun="database portal completion binding")
    expected_fields = {
        "schema",
        "task_cid",
        "attempt_id",
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "baseline_commit",
        "baseline_tree",
        "implementation_commit",
        "completion_event_id",
        "receipt_id",
    }
    if (
        set(binding) != expected_fields
        or binding.get("schema") != _DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA
    ):
        raise IntentRepositoryIntegrityError(
            "database portal completion binding schema is not closed"
        )
    normalized = {
        "schema": _DATABASE_PORTAL_COMPLETION_BINDING_SCHEMA,
        "task_cid": _identifier(binding.get("task_cid"), noun="portal task_cid"),
        "attempt_id": _identifier(
            binding.get("attempt_id"), noun="portal attempt_id"
        ),
        "binding_id": _identifier(
            binding.get("binding_id"), noun="portal binding_id"
        ),
        "portal_receipt_id": _identifier(
            binding.get("portal_receipt_id"), noun="portal receipt identity"
        ),
        "evidence_digest": _identifier(
            binding.get("evidence_digest"), noun="portal evidence digest"
        ),
        "baseline_commit": str(binding.get("baseline_commit") or ""),
        "baseline_tree": str(binding.get("baseline_tree") or ""),
        "implementation_commit": str(binding.get("implementation_commit") or ""),
        "completion_event_id": _identifier(
            binding.get("completion_event_id"), noun="portal completion event identity"
        ),
    }
    for field in (
        "binding_id",
        "portal_receipt_id",
        "evidence_digest",
        "completion_event_id",
    ):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized[field]) is None:
            raise IntentRepositoryIntegrityError(
                f"database portal completion binding {field} is malformed"
            )
    for field in ("baseline_commit", "baseline_tree", "implementation_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", normalized[field]) is None:
            raise IntentRepositoryIntegrityError(
                f"database portal completion binding {field} is malformed"
            )
    expected_receipt_id = "sha256:" + hashlib.sha256(
        canonical_json_bytes(normalized)
    ).hexdigest()
    if binding.get("receipt_id") != expected_receipt_id:
        raise IntentRepositoryIntegrityError(
            "database portal completion binding identity is invalid"
        )
    normalized["receipt_id"] = expected_receipt_id
    return normalized


def _goal_terminal_producer_artifacts(value: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = _mapping(value, noun="terminal report producer artifacts")
    if (
        set(artifacts) != {"schema", "digest_algorithm", "tasks", "bundle_id"}
        or artifacts.get("schema") != _GOAL_TERMINAL_PRODUCER_ARTIFACTS_SCHEMA
        or artifacts.get("digest_algorithm") != "sha256"
    ):
        raise IntentRepositoryIntegrityError(
            "terminal report producer artifact schema is not closed"
        )
    normalized_tasks: list[dict[str, Any]] = []
    task_aliases: set[str] = set()
    all_paths: set[str] = set()
    for raw_task in _projection_sequence(
        artifacts.get("tasks"),
        noun="terminal report producer artifact tasks",
        maximum=MAX_DEPENDENCIES,
    ):
        task = _mapping(raw_task, noun="terminal report producer artifact task")
        if set(task) != {"task_alias", "artifacts", "bundle_id"}:
            raise IntentRepositoryIntegrityError(
                "terminal report producer artifact task schema is not closed"
            )
        task_alias = _identifier(
            task.get("task_alias"), noun="terminal report producer task alias"
        )
        if task_alias in task_aliases:
            raise IntentRepositoryIntegrityError(
                "terminal report producer artifact task is duplicated"
            )
        task_aliases.add(task_alias)
        normalized_artifacts: list[dict[str, str]] = []
        for raw_artifact in _projection_sequence(
            task.get("artifacts"),
            noun="terminal report producer artifact rows",
            maximum=MAX_OUTPUTS,
        ):
            artifact = _mapping(
                raw_artifact, noun="terminal report producer artifact row"
            )
            if set(artifact) != {"path", "blob_identity"}:
                raise IntentRepositoryIntegrityError(
                    "terminal report producer artifact row schema is not closed"
                )
            path = _identifier(
                artifact.get("path"), noun="terminal report producer artifact path"
            )
            blob_identity = str(artifact.get("blob_identity") or "")
            if (
                path in all_paths
                or re.fullmatch(r"sha256:[0-9a-f]{64}", blob_identity) is None
            ):
                raise IntentRepositoryIntegrityError(
                    "terminal report producer artifact identity is invalid or duplicated"
                )
            all_paths.add(path)
            normalized_artifacts.append(
                {"path": path, "blob_identity": blob_identity}
            )
        normalized_artifacts.sort(key=lambda item: item["path"])
        if not normalized_artifacts:
            raise IntentRepositoryIntegrityError(
                "terminal report producer artifact task is empty"
            )
        task_body: dict[str, Any] = {
            "task_alias": task_alias,
            "artifacts": normalized_artifacts,
        }
        task_body["bundle_id"] = "sha256:" + hashlib.sha256(
            canonical_json_bytes(task_body)
        ).hexdigest()
        if task.get("bundle_id") != task_body["bundle_id"]:
            raise IntentRepositoryIntegrityError(
                "terminal report producer artifact task identity is invalid"
            )
        normalized_tasks.append(task_body)
    normalized_tasks.sort(key=lambda item: item["task_alias"])
    if not normalized_tasks:
        raise IntentRepositoryIntegrityError(
            "terminal report producer artifacts are empty"
        )
    normalized: dict[str, Any] = {
        "schema": _GOAL_TERMINAL_PRODUCER_ARTIFACTS_SCHEMA,
        "digest_algorithm": "sha256",
        "tasks": normalized_tasks,
    }
    normalized["bundle_id"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(normalized)
    ).hexdigest()
    if artifacts.get("bundle_id") != normalized["bundle_id"]:
        raise IntentRepositoryIntegrityError(
            "terminal report producer artifact bundle identity is invalid"
        )
    return normalized


def _goal_terminal_producer_receipt_bindings(
    value: Any,
    *,
    producer_receipts: Mapping[str, str],
    producer_artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifact_bundles = {
        str(item.get("task_alias") or ""): str(item.get("bundle_id") or "")
        for item in producer_artifacts.get("tasks", [])
        if isinstance(item, Mapping)
    }
    normalized: list[dict[str, Any]] = []
    aliases: set[str] = set()
    for raw_item in _projection_sequence(
        value,
        noun="terminal report producer receipt bindings",
        maximum=MAX_DEPENDENCIES,
    ):
        item = _mapping(raw_item, noun="terminal report producer receipt binding")
        if set(item) != {
            "schema",
            "task_alias",
            "task_cid",
            "completion_receipt_cid",
            "portal_completion_binding",
            "artifact_bundle_id",
            "binding_id",
        } or item.get("schema") != _GOAL_TERMINAL_PRODUCER_RECEIPT_BINDING_SCHEMA:
            raise IntentRepositoryIntegrityError(
                "terminal report producer receipt binding schema is not closed"
            )
        task_alias = _identifier(
            item.get("task_alias"), noun="terminal report producer task alias"
        )
        task_cid = _identifier(
            item.get("task_cid"), noun="terminal report producer task_cid"
        )
        completion_receipt_cid = _identifier(
            item.get("completion_receipt_cid"),
            noun="terminal report producer completion receipt",
        )
        artifact_bundle_id = str(item.get("artifact_bundle_id") or "")
        portal_binding = _database_portal_completion_binding(
            _mapping(
                item.get("portal_completion_binding"),
                noun="terminal report producer Portal completion binding",
            )
        )
        body: dict[str, Any] = {
            "schema": _GOAL_TERMINAL_PRODUCER_RECEIPT_BINDING_SCHEMA,
            "task_alias": task_alias,
            "task_cid": task_cid,
            "completion_receipt_cid": completion_receipt_cid,
            "portal_completion_binding": portal_binding,
            "artifact_bundle_id": artifact_bundle_id,
        }
        body["binding_id"] = "sha256:" + hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest()
        if (
            task_alias in aliases
            or portal_binding["task_cid"] != task_cid
            or producer_receipts.get(task_alias) != completion_receipt_cid
            or artifact_bundles.get(task_alias) != artifact_bundle_id
            or re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_bundle_id) is None
            or item.get("binding_id") != body["binding_id"]
        ):
            raise IntentRepositoryIntegrityError(
                "terminal report producer receipt binding is invalid"
            )
        aliases.add(task_alias)
        normalized.append(body)
    normalized.sort(key=lambda item: item["task_alias"])
    if aliases != set(producer_receipts) or aliases != set(artifact_bundles):
        raise IntentRepositoryIntegrityError(
            "terminal report producer receipt binding population is not exact"
        )
    return normalized


def _goal_terminal_report_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _mapping(value, noun="terminal report evidence")
    expected_fields = {
        "schema",
        "terminal_report_contract_id",
        "task_cid",
        "task_alias",
        "task_revision",
        "completion_receipt_cid",
        "completion_evidence_digest",
        "control_receipt_id",
        "portal_receipt_id",
        "portal_completion_binding",
        "producer_receipts",
        "producer_artifacts",
        "producer_receipt_bindings",
        "validation_run_id",
        "validation_result_id",
        "validation_evidence_id",
        "report_artifacts",
        "evidence_id",
    }
    if (
        set(evidence) != expected_fields
        or evidence.get("schema") != GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA
    ):
        raise IntentRepositoryIntegrityError(
            "terminal report evidence schema is not closed"
        )
    normalized: dict[str, Any] = {
        "schema": GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA,
        "terminal_report_contract_id": _identifier(
            evidence.get("terminal_report_contract_id"),
            noun="terminal_report_contract_id",
        ),
        "task_cid": _identifier(evidence.get("task_cid"), noun="terminal task_cid"),
        "task_alias": _identifier(
            evidence.get("task_alias"), noun="terminal task_alias"
        ),
        "task_revision": _positive_int(
            evidence.get("task_revision"), noun="terminal task revision"
        ),
        "completion_receipt_cid": _identifier(
            evidence.get("completion_receipt_cid"),
            noun="terminal completion_receipt_cid",
        ),
        "completion_evidence_digest": _identifier(
            evidence.get("completion_evidence_digest"),
            noun="terminal completion_evidence_digest",
        ),
        "control_receipt_id": _identifier(
            evidence.get("control_receipt_id"), noun="terminal control_receipt_id"
        ),
        "portal_receipt_id": _identifier(
            evidence.get("portal_receipt_id"), noun="terminal portal_receipt_id"
        ),
        "portal_completion_binding": _database_portal_completion_binding(
            _mapping(
                evidence.get("portal_completion_binding"),
                noun="terminal portal completion binding",
            )
        ),
        "validation_run_id": _identifier(
            evidence.get("validation_run_id"), noun="terminal validation_run_id"
        ),
        "validation_result_id": _identifier(
            evidence.get("validation_result_id"),
            noun="terminal validation_result_id",
        ),
        "validation_evidence_id": _identifier(
            evidence.get("validation_evidence_id"),
            noun="terminal validation_evidence_id",
        ),
    }
    if re.fullmatch(
        r"sha256:[0-9a-f]{64}", normalized["portal_receipt_id"]
    ) is None:
        raise IntentRepositoryIntegrityError(
            "terminal report portal receipt identity is malformed"
        )
    if (
        normalized["portal_completion_binding"]["portal_receipt_id"]
        != normalized["portal_receipt_id"]
    ):
        raise IntentRepositoryIntegrityError(
            "terminal report Portal receipt differs from its completion binding"
        )
    raw_producer_receipts = _mapping(
        evidence.get("producer_receipts"), noun="terminal report producer receipts"
    )
    if not 1 <= len(raw_producer_receipts) <= MAX_DEPENDENCIES:
        raise IntentRepositoryIntegrityError(
            "terminal report producer receipt population is not bounded and nonempty"
        )
    producer_receipts: dict[str, str] = {}
    for raw_alias, raw_receipt_id in raw_producer_receipts.items():
        alias = _identifier(raw_alias, noun="terminal report producer task alias")
        receipt_id = _identifier(
            raw_receipt_id, noun="terminal report producer receipt identity"
        )
        if alias in producer_receipts:
            raise IntentRepositoryIntegrityError(
                "terminal report producer receipt aliases are not unique"
            )
        producer_receipts[alias] = receipt_id
    normalized["producer_receipts"] = dict(sorted(producer_receipts.items()))
    normalized["producer_artifacts"] = _goal_terminal_producer_artifacts(
        _mapping(
            evidence.get("producer_artifacts"),
            noun="terminal report producer artifacts",
        )
    )
    normalized["producer_receipt_bindings"] = (
        _goal_terminal_producer_receipt_bindings(
            evidence.get("producer_receipt_bindings"),
            producer_receipts=normalized["producer_receipts"],
            producer_artifacts=normalized["producer_artifacts"],
        )
    )
    artifacts: list[dict[str, str]] = []
    artifact_paths: set[str] = set()
    for item in _projection_sequence(
        evidence.get("report_artifacts"),
        noun="terminal report artifacts",
        maximum=MAX_OUTPUTS,
    ):
        artifact = _mapping(item, noun="terminal report artifact")
        if set(artifact) != {
            "path",
            "blob_identity",
            "portal_baseline_blob_identity",
        }:
            raise IntentRepositoryIntegrityError(
                "terminal report artifact schema is not closed"
            )
        path = _identifier(artifact.get("path"), noun="terminal report artifact path")
        blob_identity = str(artifact.get("blob_identity") or "")
        portal_baseline_blob_identity = str(
            artifact.get("portal_baseline_blob_identity") or ""
        )
        if (
            path in artifact_paths
            or re.fullmatch(r"sha256:[0-9a-f]{64}", blob_identity) is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", portal_baseline_blob_identity
            )
            is None
            or blob_identity == portal_baseline_blob_identity
        ):
            raise IntentRepositoryIntegrityError(
                "terminal report artifact identity is invalid or unchanged from its "
                "Portal baseline"
            )
        artifact_paths.add(path)
        artifacts.append(
            {
                "path": path,
                "blob_identity": blob_identity,
                "portal_baseline_blob_identity": portal_baseline_blob_identity,
            }
        )
    if len(artifacts) != 2:
        raise IntentRepositoryIntegrityError(
            "terminal report evidence must bind exactly the JSON and Markdown reports"
        )
    normalized["report_artifacts"] = artifacts
    normalized["evidence_id"] = content_identity(normalized)
    if evidence.get("evidence_id") != normalized["evidence_id"]:
        raise IntentRepositoryIntegrityError("terminal report evidence identity is invalid")
    return normalized


def _goal_runtime_settlement_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact, content-addressed runtime settlement proof."""

    binding = _mapping(value, noun="VRIF runtime settlement binding")
    expected_fields = {
        "schema",
        "settled",
        "receipt_cid",
        "snapshot_cid",
        "owner_generation",
        "target",
        "config_cid",
        "profile_cid",
        "lane_snapshot_cids",
        "merge_queue_receipt_cid",
        "merge_queue_snapshot_cid",
        "active_counts",
        "retired_ready_task_cids",
        "binding_id",
    }
    if (
        set(binding) != expected_fields
        or binding.get("schema") != GOAL_RUNTIME_SETTLEMENT_BINDING_SCHEMA
        or binding.get("settled") is not True
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement binding schema or state is not exact"
        )
    cid_fields = (
        "receipt_cid",
        "snapshot_cid",
        "config_cid",
        "profile_cid",
        "merge_queue_receipt_cid",
        "merge_queue_snapshot_cid",
    )
    if any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", str(binding.get(field) or ""))
        is None
        for field in cid_fields
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement binding contains an invalid content identity"
        )
    owner_generation = _positive_int(
        binding.get("owner_generation"),
        noun="VRIF runtime settlement owner generation",
    )
    target = _mapping(binding.get("target"), noun="VRIF runtime settlement target")
    if set(target) != {"binding_schema", "repository_id", "branch"} or target.get(
        "binding_schema"
    ) != _MERGE_TARGET_BINDING_SCHEMA:
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement target schema is not exact"
        )
    repository_id = _identifier(
        target.get("repository_id"), noun="runtime target repository_id"
    )
    branch = _identifier(target.get("branch"), noun="runtime target branch")
    if re.fullmatch(r"repository:baguqeera[a-z2-7]{52}", repository_id) is None:
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement target repository identity is invalid"
        )

    lane_values = binding.get("lane_snapshot_cids")
    if (
        not isinstance(lane_values, list)
        or len(lane_values) != 4
        or len(set(lane_values)) != 4
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(item or "")) is None
            for item in lane_values
        )
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement must bind four ordered lane snapshots"
        )
    active_counts = _mapping(
        binding.get("active_counts"), noun="VRIF runtime active counts"
    )
    if set(active_counts) != {"coordination", "execution", "merge_queue", "total"} or any(
        type(active_counts.get(field)) is not int or active_counts.get(field) != 0
        for field in ("coordination", "execution", "merge_queue", "total")
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement binding is not exactly inactive"
        )
    retired_values = binding.get("retired_ready_task_cids")
    if isinstance(retired_values, (str, bytes, bytearray)) or not isinstance(
        retired_values, Sequence
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF retired ready task identities must be a sequence"
        )
    retired_ready_task_cids = [
        _identifier(item, noun="retired ready task_cid") for item in retired_values
    ]
    if (
        len(retired_ready_task_cids) > MAX_PAGE_LIMIT
        or retired_ready_task_cids != sorted(set(retired_ready_task_cids))
        or any(
            re.fullmatch(r"baguqeera[a-z2-7]{52}", task_cid) is None
            for task_cid in retired_ready_task_cids
        )
    ):
        raise IntentRepositoryIntegrityError(
            "VRIF retired ready task identities are not exact sorted CIDs"
        )
    normalized = {
        **dict(binding),
        "owner_generation": owner_generation,
        "target": {
            "binding_schema": _MERGE_TARGET_BINDING_SCHEMA,
            "repository_id": repository_id,
            "branch": branch,
        },
        "lane_snapshot_cids": list(lane_values),
        "active_counts": dict(active_counts),
        "retired_ready_task_cids": retired_ready_task_cids,
    }
    supplied_binding_id = str(normalized.pop("binding_id") or "")
    normalized["binding_id"] = "sha256:" + hashlib.sha256(
        _canonical(normalized, noun="VRIF runtime settlement binding").encode(
            "utf-8"
        )
    ).hexdigest()
    if supplied_binding_id != normalized["binding_id"]:
        raise IntentRepositoryIntegrityError(
            "VRIF runtime settlement binding identity is invalid"
        )
    return normalized


def _goal_root_completion_gate(
    value: Mapping[str, Any],
    *,
    authority_spec_id: str,
) -> dict[str, Any]:
    gate = _mapping(value, noun="root goal completion gate")
    expected_fields = {
        "schema",
        "authority_spec_id",
        "source_head",
        "repository_tree_id",
        "predecessor_gate_id",
        "owner_generation",
        "owner_restart_admission_id",
        "owner_restart_receipt_id",
        "completion_policy",
        "runtime_settlement_binding",
        "terminal_report_evidence",
        "gate_id",
    }
    if set(gate) != expected_fields or gate.get("schema") != GOAL_ROOT_COMPLETION_GATE_SCHEMA:
        raise IntentRepositoryIntegrityError("root goal completion gate schema is not closed")
    if str(gate.get("authority_spec_id") or "") != authority_spec_id:
        raise IntentRepositoryIntegrityError("root goal completion gate has foreign authority")
    for field in (
        "source_head",
        "repository_tree_id",
        "owner_restart_admission_id",
        "owner_restart_receipt_id",
    ):
        if not str(gate.get(field) or "").strip():
            raise IntentRepositoryIntegrityError(
                f"root goal completion gate is missing {field}"
            )
    predecessor_gate_id = _optional_identifier(
        gate.get("predecessor_gate_id"), noun="predecessor_gate_id"
    )
    owner_generation = _positive_int(
        gate.get("owner_generation"), noun="root gate owner_generation"
    )
    runtime_settlement_binding = _goal_runtime_settlement_binding(
        _mapping(
            gate.get("runtime_settlement_binding"),
            noun="root runtime settlement binding",
        )
    )
    if runtime_settlement_binding["owner_generation"] != owner_generation:
        raise IntentRepositoryIntegrityError(
            "root gate and runtime settlement owner generations differ"
        )
    policy = _mapping(gate.get("completion_policy"), noun="root completion policy")
    if set(policy) != {
        *_ROOT_COMPLETION_POLICY_FIELDS,
        _ROOT_TERMINAL_TASK_POLICY_FIELD,
    } or any(
        policy.get(field) is not True for field in _ROOT_COMPLETION_POLICY_FIELDS
    ) or not str(policy.get(_ROOT_TERMINAL_TASK_POLICY_FIELD) or "").strip():
        raise IntentRepositoryIntegrityError("root goal completion policy is not exact")
    terminal_report_evidence = _goal_terminal_report_evidence(
        _mapping(
            gate.get("terminal_report_evidence"),
            noun="root terminal report evidence",
        )
    )
    gate_id = str(gate.get("gate_id") or "")
    body = dict(gate)
    body.pop("gate_id", None)
    if gate_id != content_identity(body):
        raise IntentRepositoryIntegrityError("root goal completion gate identity is invalid")
    return {
        **dict(gate),
        "predecessor_gate_id": predecessor_gate_id,
        "owner_generation": owner_generation,
        "completion_policy": policy,
        "runtime_settlement_binding": runtime_settlement_binding,
        "terminal_report_evidence": terminal_report_evidence,
    }


def _goal_completion_receipt(
    *,
    authority_spec_id: str,
    goal_cid: str,
    goal_alias: str,
    goal_revision: int,
    task_receipts: Sequence[Mapping[str, Any]],
    child_goal_receipts: Sequence[Mapping[str, Any]],
    dependency_goal_receipts: Sequence[Mapping[str, Any]],
    receipt_backfill: bool,
    root_completion_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": GOAL_COMPLETION_RECEIPT_SCHEMA,
        "authority_spec_id": authority_spec_id,
        "goal_cid": goal_cid,
        "goal_alias": goal_alias,
        "goal_revision": int(goal_revision),
        "completion_kind": (
            "preseeded_completion_receipt_backfill"
            if receipt_backfill
            else "current_authority_completion"
        ),
        "task_receipts": [dict(item) for item in task_receipts],
        "child_goal_receipts": [dict(item) for item in child_goal_receipts],
        "dependency_goal_receipts": [dict(item) for item in dependency_goal_receipts],
        "root_completion_gate": (
            dict(root_completion_gate) if root_completion_gate is not None else None
        ),
    }
    body["receipt_id"] = content_identity(body)
    return body


def _goal_receipt_has_valid_identity(
    receipt: Mapping[str, Any],
    *,
    authority_spec_id: str,
    goal_cid: str,
    goal_alias: str,
    goal_revision: int,
    expected_root_completion_gate: Mapping[str, Any] | None,
) -> bool:
    """Validate an emitted goal receipt independently of current descendants."""

    fields = {
        "schema",
        "authority_spec_id",
        "goal_cid",
        "goal_alias",
        "goal_revision",
        "completion_kind",
        "task_receipts",
        "child_goal_receipts",
        "dependency_goal_receipts",
        "root_completion_gate",
        "receipt_id",
    }
    if (
        set(receipt) != fields
        or receipt.get("schema") != GOAL_COMPLETION_RECEIPT_SCHEMA
        or receipt.get("authority_spec_id") != authority_spec_id
        or receipt.get("goal_cid") != goal_cid
        or receipt.get("goal_alias") != goal_alias
        or receipt.get("goal_revision") != goal_revision
        or receipt.get("completion_kind")
        not in {
            "current_authority_completion",
            "preseeded_completion_receipt_backfill",
        }
        or receipt.get("root_completion_gate")
        != (
            dict(expected_root_completion_gate)
            if expected_root_completion_gate is not None
            else None
        )
        or any(
            not isinstance(receipt.get(name), list)
            or not all(isinstance(item, Mapping) for item in receipt.get(name, []))
            for name in (
                "task_receipts",
                "child_goal_receipts",
                "dependency_goal_receipts",
            )
        )
    ):
        return False
    body = dict(receipt)
    receipt_id = str(body.pop("receipt_id", "") or "")
    return receipt_id == content_identity(body)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentReceipt:
    """Durable receipt for one intent mutation."""

    SCHEMA: ClassVar[str] = INTENT_RECEIPT_SCHEMA

    event_id: str
    event_type: str
    global_sequence: int
    recorded_at: str
    subject_id: str
    revision: int
    changed: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "global_sequence": int(self.global_sequence),
            "recorded_at": self.recorded_at,
            "subject_id": self.subject_id,
            "revision": int(self.revision),
            "changed": bool(self.changed),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class IntentSnapshot:
    """Generation-bound snapshot of intent projections."""

    SCHEMA: ClassVar[str] = INTENT_SNAPSHOT_SCHEMA

    objective_count: int
    goal_count: int
    plan_count: int
    task_count: int
    dependency_count: int
    event_watermark: int
    projection_cid: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "objective_count": int(self.objective_count),
            "goal_count": int(self.goal_count),
            "plan_count": int(self.plan_count),
            "task_count": int(self.task_count),
            "dependency_count": int(self.dependency_count),
            "event_watermark": int(self.event_watermark),
            "projection_cid": self.projection_cid,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class QueueEntry:
    """Queue backoff / selection state for one task."""

    SCHEMA: ClassVar[str] = QUEUE_ENTRY_SCHEMA

    task_cid: str
    attempt: int = 0
    retry_not_before_ms: int = 0
    selection_penalty: int = 0
    consecutive_failures: int = 0
    state: str = "ready"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "attempt": int(self.attempt),
            "retry_not_before_ms": int(self.retry_not_before_ms),
            "selection_penalty": int(self.selection_penalty),
            "consecutive_failures": int(self.consecutive_failures),
            "state": self.state,
            "reason": self.reason,
        }

    def is_cooled_down(self, *, now_ms: int | None = None) -> bool:
        clock = _now_ms() if now_ms is None else int(now_ms)
        return int(self.retry_not_before_ms) > clock


@dataclass(frozen=True)
class PlanHead:
    """Active plan head for a goal."""

    SCHEMA: ClassVar[str] = PLAN_HEAD_SCHEMA

    plan_cid: str
    goal_cid: str
    revision: int
    status: str
    superseded_by: str = ""
    continuation_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_cid": self.plan_cid,
            "goal_cid": self.goal_cid,
            "revision": int(self.revision),
            "status": self.status,
            "superseded_by": self.superseded_by,
            "continuation_of": self.continuation_of,
        }


# ---------------------------------------------------------------------------
# IntentRepository
# ---------------------------------------------------------------------------


class IntentRepository:
    """Transactional authority for intent-domain control-plane state.

    Interface: ``IntentRepository@1``.
    """

    INTERFACE: ClassVar[str] = INTENT_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = INTENT_REPOSITORY_SCHEMA

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        bound_connection: Any | None = None,
        owner_id: str = DEFAULT_OWNER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        install_schema: bool = True,
        evidence_freshness_seconds: int = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
        lock_timeout_seconds: float = 30.0,
        clock_ms: Any | None = None,
    ) -> None:
        _require_duckdb()
        if bound_connection is not None:
            if database_path is not None and is_quack_transport_target(database_path):
                raise IntentRepositoryError(
                    "a bound owner connection cannot also target Quack transport"
                )
            if install_schema:
                raise IntentRepositoryError("bound owner connections require install_schema=False")
            if not callable(getattr(bound_connection, "execute", None)):
                raise IntentRepositoryError("bound owner connection must provide execute()")
            self._open_target = Path(database_path or "bound-owner-control-plane.duckdb")
            self._quack_transport = False
            self._bound_connection = bound_connection
            self._bound_connection_lock = threading.RLock()
            self._bound_transaction_depth = 0
            self._quack_read_connection = None
            self.database_path = self._open_target
        elif database_path is None:
            raise IntentRepositoryError("database_path or bound_connection is required")
        elif is_quack_transport_target(database_path):
            self._open_target = quack_transport_uri(database_path)
            self._quack_transport = True
            self._bound_connection = None
            self._bound_connection_lock = threading.RLock()
            self._bound_transaction_depth = 0
            self._quack_read_connection = None
            # Path identity is unused for file locks; keep a stable placeholder.
            self.database_path = Path(self._open_target)
        else:
            self._open_target = Path(database_path).absolute()
            self._quack_transport = False
            self._bound_connection = None
            self._bound_connection_lock = threading.RLock()
            self._bound_transaction_depth = 0
            self._quack_read_connection = None
            self.database_path = self._open_target
        self.owner_id = _identifier(owner_id, noun="owner_id")
        self.session_id = _identifier(session_id, noun="session_id")
        if (
            isinstance(evidence_freshness_seconds, bool)
            or not isinstance(evidence_freshness_seconds, int)
            or evidence_freshness_seconds < 0
        ):
            raise IntentRepositoryBoundsError(
                "evidence_freshness_seconds must be a non-negative integer"
            )
        self.evidence_freshness_seconds = int(evidence_freshness_seconds)
        if lock_timeout_seconds <= 0:
            raise IntentRepositoryBoundsError("lock_timeout_seconds must be positive")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock_ms = clock_ms or _now_ms
        self._lock_path = (
            None
            if self._quack_transport
            else self.database_path.with_name(f".{self.database_path.name}.intent.lock")
        )
        self._open = False
        self._closed = False
        self._quack_connection: Any | None = None
        if self._quack_transport:
            # Schema is owned by the Quack state-owner / trusted materializer.
            install_schema = False
        if install_schema:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.database_path.exists():
                install_control_plane_schema(
                    self.database_path,
                    application_version="0.0.45",
                    tool_version="1.5.2",
                    owner_id=self.owner_id,
                )
            else:
                # Ensure schema is present for pre-created empty files.
                try:
                    connection = open_duckdb_connection(self.database_path)
                    try:
                        tables = {
                            str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()
                        }
                    finally:
                        connection.close()
                    if "tasks" not in tables:
                        install_control_plane_schema(
                            self.database_path,
                            application_version="0.0.45",
                            tool_version="1.5.2",
                            owner_id=self.owner_id,
                        )
                except Exception:
                    install_control_plane_schema(
                        self.database_path,
                        application_version="0.0.45",
                        tool_version="1.5.2",
                        owner_id=self.owner_id,
                    )
        self._open = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open and not self._closed

    @property
    def uses_quack_transport(self) -> bool:
        """Whether reads use Quack and mutations require typed owner commands."""

        return self._quack_transport

    @property
    def uses_bound_connection(self) -> bool:
        """Whether lifecycle belongs to an injected exclusive-owner connection."""

        return self._bound_connection is not None

    def close(self) -> None:
        self._closed = True
        self._open = False
        connection = self._quack_read_connection
        self._quack_read_connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def __enter__(self) -> IntentRepository:
        self._require_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed or not self._open:
            raise IntentRepositoryNotOpenError("intent repository is not open")

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[Any]:
        self._require_open()
        if self._bound_connection is not None:
            # The exclusive state owner retains connection lifecycle authority.
            # Repository calls serialize on that connection and own only their
            # transaction boundary; close() never closes the injected handle.
            with self._bound_connection_lock:
                connection = self._bound_connection
                if self._bound_transaction_depth:
                    yield connection
                    return
                if write:
                    connection.execute("BEGIN TRANSACTION")
                    self._bound_transaction_depth = 1
                    try:
                        yield connection
                        connection.execute("COMMIT")
                    except BaseException:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                    finally:
                        self._bound_transaction_depth = 0
                else:
                    yield connection
            return
        if self._quack_transport:
            with self._bound_connection_lock:
                connection = self._quack_read_connection
                if connection is not None and not quack_session_is_live(connection):
                    try:
                        connection.close()
                    except Exception:
                        pass
                    self._quack_read_connection = None
                    connection = None
                if connection is None:
                    self._quack_read_connection = open_duckdb_connection(
                        self._open_target
                    )
                    connection = self._quack_read_connection
                try:
                    yield connection
                except BaseException as exc:
                    if _is_quack_session_dead(exc):
                        try:
                            connection.close()
                        except Exception:
                            pass
                        self._quack_read_connection = None
                    raise
            return
        # Match DuckDBTaskSource / StateTransaction durability: begin with SQL,
        # commit/rollback with SQL, and always close the adapter explicitly.
        # Avoid relying on DuckDBConnection.__exit__ transaction bookkeeping,
        # which can mark a SQL-started transaction inactive before COMMIT runs.
        if write and not self._quack_transport:
            with exclusive_file_lock(self._lock_path, timeout_seconds=self.lock_timeout_seconds):
                connection = open_duckdb_connection(self._open_target)
                try:
                    connection.execute("BEGIN TRANSACTION")
                    try:
                        yield connection
                        connection.execute("COMMIT")
                    except BaseException:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                finally:
                    connection.close()
            return
        if write and self._quack_transport:
            # Serialize before opening the remote connection.  The owner
            # deliberately restarts its read-only endpoint after each admitted
            # commit; a waiting writer must not retain a connection to the
            # prior replica generation.
            expected_store = str(
                os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
            ).strip()
            write_lock = quack_owner_mutation_write_lock_path(expected_store)
            if write_lock is None:
                raise IntentRepositoryIntegrityError(
                    "quack write transaction has no accepted-root lock path"
                )
            with exclusive_file_lock(
                write_lock, timeout_seconds=self.lock_timeout_seconds
            ):
                connection = open_duckdb_connection(self._open_target)
                try:
                    binding = getattr(connection, "_quack_mutation_binding", None)
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("store_id") != expected_store
                    ):
                        raise IntentRepositoryIntegrityError(
                            "quack write lock is not bound to the live store"
                        )
                    connection.execute("BEGIN TRANSACTION")
                    try:
                        yield connection
                        connection.execute("COMMIT")
                    except DuckDBQuackMutationConflictError as exc:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise IntentRepositoryConflictError(
                            "remote task revision or event-head CAS conflicted"
                        ) from exc
                    except DuckDBQuackMutationTransitionError as exc:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise IntentRepositoryTransitionError(
                            "remote task status transition is not admitted"
                        ) from exc
                    except DuckDBQuackMutationUnknownOutcomeError as exc:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise IntentRepositoryUnknownOutcomeError(
                            "remote mutation outcome requires exact reconciliation"
                        ) from exc
                    except BaseException:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                finally:
                    connection.close()
            return
        if self._quack_transport:
            # Replica publication is fail-closed: the owner withdraws and
            # restarts the endpoint synchronously after each admitted bundle.
            # Hold the same bounded store lock for the complete open/query/
            # close window so a trusted read can never straddle that refresh.
            expected_store = str(
                os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
            ).strip()
            read_lock = quack_owner_mutation_write_lock_path(expected_store)
            if read_lock is None:
                raise IntentRepositoryIntegrityError(
                    "quack read transaction has no accepted-root lock path"
                )
            with exclusive_file_lock(
                read_lock, timeout_seconds=self.lock_timeout_seconds
            ):
                connection = open_duckdb_connection(self._open_target)
                try:
                    binding = getattr(connection, "_quack_mutation_binding", None)
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("store_id") != expected_store
                    ):
                        raise IntentRepositoryIntegrityError(
                            "quack read lock is not bound to the live store"
                        )
                    yield connection
                finally:
                    connection.close()
            return
        connection = open_duckdb_connection(self._open_target)
        try:
            if write:
                connection.execute("BEGIN TRANSACTION")
                try:
                    yield connection
                    connection.execute("COMMIT")
                except BaseException:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
            else:
                yield connection
        finally:
            connection.close()

    def recover_idempotent_owner_command(
        self,
        *,
        request_id: str,
        command: str,
        command_payload: Mapping[str, Any],
        store_id: str,
        store_generation: str,
    ) -> Mapping[str, Any] | None:
        """Return an exact durable owner-command result without executing it."""

        if not self.uses_bound_connection:
            raise IntentRepositoryError(
                "idempotent owner command recovery requires a bound owner connection"
            )
        rid = _identifier(request_id, noun="request_id")
        command_name = _identifier(command, noun="owner command")
        store = str(store_id or "").strip()
        if not store or "\x00" in store or len(store.encode("utf-8")) > MAX_ID_BYTES:
            raise IntentRepositoryError("store_id is empty or exceeds its bound")
        generation = _identifier(store_generation, noun="store_generation")
        payload_map = _mapping(command_payload, noun="owner command payload")
        command_id = content_identity(
            {"command": command_name, "payload": payload_map}
        )
        idempotency_key = f"quack-owner-command:{rid}"
        with self._connection(write=False) as connection:
            prior = connection.execute(
                """
                SELECT command_kind, command_id, store_id, result_digest,
                       body_json
                FROM idempotency_records
                WHERE idempotency_key = ?
                """,
                [idempotency_key],
            ).fetchone()
        if prior is None:
            return None
        body = _decode_json(prior[4], noun="owner command idempotency body")
        if not isinstance(body, Mapping):
            raise IntentRepositoryIntegrityError(
                "owner command idempotency body is malformed"
            )
        result = body.get("result")
        if (
            str(prior[0]) != command_name
            or str(prior[1]) != command_id
            or str(prior[2]) != store
            or body.get("store_generation") != generation
            or not isinstance(result, Mapping)
            or str(prior[3]) != content_identity(dict(result))
        ):
            raise IntentRepositoryConflictError(
                "owner command request identity was reused with stale bindings"
            )
        return MappingProxyType(dict(result))

    def run_idempotent_owner_command(
        self,
        *,
        request_id: str,
        command: str,
        command_payload: Mapping[str, Any],
        store_id: str,
        store_generation: str,
        operation: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Run and durably memoize one typed owner command atomically.

        The command mutation and its response share the same transaction on
        the bound state-owner connection.  A request left in the inbox across
        an owner crash therefore returns its stored response instead of
        replaying a non-idempotent repository operation.
        """

        if not self.uses_bound_connection:
            raise IntentRepositoryError(
                "idempotent owner commands require a bound owner connection"
            )
        rid = _identifier(request_id, noun="request_id")
        command_name = _identifier(command, noun="owner command")
        store = str(store_id or "").strip()
        if not store or "\x00" in store or len(store.encode("utf-8")) > MAX_ID_BYTES:
            raise IntentRepositoryError("store_id is empty or exceeds its bound")
        generation = _identifier(store_generation, noun="store_generation")
        payload_map = _mapping(command_payload, noun="owner command payload")
        command_id = content_identity({"command": command_name, "payload": payload_map})
        idempotency_key = f"quack-owner-command:{rid}"
        with self._connection(write=True) as connection:
            prior = connection.execute(
                """
                SELECT command_kind, command_id, store_id, result_digest,
                       body_json
                FROM idempotency_records
                WHERE idempotency_key = ?
                """,
                [idempotency_key],
            ).fetchone()
            if prior is not None:
                body = _decode_json(prior[4], noun="owner command idempotency body")
                if not isinstance(body, Mapping):
                    raise IntentRepositoryIntegrityError(
                        "owner command idempotency body is malformed"
                    )
                result = body.get("result")
                if (
                    str(prior[0]) != command_name
                    or str(prior[1]) != command_id
                    or str(prior[2]) != store
                    or body.get("store_generation") != generation
                    or not isinstance(result, Mapping)
                    or str(prior[3]) != content_identity(dict(result))
                ):
                    raise IntentRepositoryConflictError(
                        "owner command request identity was reused with stale bindings"
                    )
                return MappingProxyType(dict(result))
            result = operation()
            if not isinstance(result, Mapping):
                raise IntentRepositoryIntegrityError(
                    "owner command operation did not return a mapping"
                )
            result_map = dict(result)
            stored_body = {
                "request_id": rid,
                "store_generation": generation,
                "result": result_map,
            }
            connection.execute(
                """
                INSERT INTO idempotency_records (
                    idempotency_key, command_kind, command_id, store_id,
                    session_id, result_digest, created_at, expires_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    idempotency_key,
                    command_name,
                    command_id,
                    store,
                    self.session_id,
                    content_identity(result_map),
                    _utc_iso(),
                    None,
                    _canonical(stored_body, noun="owner command idempotency body"),
                ],
            )
            return MappingProxyType(result_map)

    # -- event plumbing ------------------------------------------------------

    def _next_global_sequence(self, connection: Any) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        return int(row[0] if row else 0) + 1

    def _next_stream_sequence(self, connection: Any) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM domain_events WHERE stream_id = ?",
            [INTENT_STREAM_ID],
        ).fetchone()
        return int(row[0] if row else 0) + 1

    def _append_event(
        self,
        connection: Any,
        *,
        event_type: IntentEventType | str,
        subject_id: str,
        body: Mapping[str, Any],
        task_cid: str = "",
        attempt_id: str = "",
    ) -> IntentReceipt:
        global_sequence = self._next_global_sequence(connection)
        if global_sequence > MAX_EVENTS:
            raise IntentRepositoryBoundsError("domain event population exceeded")
        stream_sequence = self._next_stream_sequence(connection)
        recorded_at = _utc_iso()
        event_type_value = (
            event_type.value if isinstance(event_type, IntentEventType) else str(event_type)
        )
        body_payload = {
            "schema": INTENT_EVENT_SCHEMA,
            "event_type": event_type_value,
            "subject_id": subject_id,
            "body": _jsonable(dict(body)),
            "recorded_at": recorded_at,
            "owner_id": self.owner_id,
        }
        event_id = content_identity(
            {
                "stream_id": INTENT_STREAM_ID,
                "sequence": stream_sequence,
                "global_sequence": global_sequence,
                "event_type": event_type_value,
                "body": body_payload,
            }
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                INTENT_STREAM_ID,
                stream_sequence,
                global_sequence,
                event_type_value,
                task_cid or "",
                attempt_id or "",
                self.session_id,
                recorded_at,
                _canonical(body_payload, noun="event body"),
            ],
        )
        return IntentReceipt(
            event_id=event_id,
            event_type=event_type_value,
            global_sequence=global_sequence,
            recorded_at=recorded_at,
            subject_id=subject_id,
            revision=int(body.get("revision") or 0),
            changed=True,
            details=MappingProxyType(dict(body)),
        )

    def event_watermark(self) -> int:
        with self._connection(write=False) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()
            return int(row[0] if row else 0)

    def list_events(
        self,
        *,
        after_global_sequence: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> tuple[Mapping[str, Any], ...]:
        selected = _bounded_limit(limit)
        after = _nonneg_int(after_global_sequence, noun="after_global_sequence")
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence, event_type,
                       task_cid, attempt_id, session_id, recorded_at, body_json
                FROM domain_events
                WHERE global_sequence > ?
                ORDER BY global_sequence ASC
                LIMIT ?
                """,
                [after, selected],
            ).fetchall()
        return tuple(
            MappingProxyType(
                {
                    "event_id": str(row[0]),
                    "stream_id": str(row[1]),
                    "sequence": int(row[2]),
                    "global_sequence": int(row[3]),
                    "event_type": str(row[4]),
                    "task_cid": str(row[5] or ""),
                    "attempt_id": str(row[6] or ""),
                    "session_id": str(row[7] or ""),
                    "recorded_at": str(row[8]),
                    "body": _decode_json(row[9], noun="event body"),
                }
            )
            for row in rows
        )

    # -- objectives ----------------------------------------------------------

    def upsert_objective(
        self,
        *,
        objective_id: str,
        objective_alias: str,
        title: str,
        status: str = "open",
        priority: str = "P2",
        parent_objective_id: str = "",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        oid = _identifier(objective_id, noun="objective_id")
        alias = _identifier(objective_alias, noun="objective_alias")
        title_text = str(title or "").strip() or alias
        status_text = str(status or "open").strip().lower()
        priority_text = str(priority or "P2").strip() or "P2"
        parent = _optional_identifier(parent_objective_id, noun="parent_objective_id")
        body_map = _mapping(body, noun="objective body")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            existing = connection.execute(
                "SELECT revision, status, body_json FROM objectives WHERE objective_id = ?",
                [oid],
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "objective CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO objectives (
                        objective_id, objective_alias, parent_objective_id,
                        title, status, priority, created_at, updated_at,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        oid,
                        alias,
                        parent,
                        title_text,
                        status_text,
                        priority_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="objective body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if expected_revision is not None and expected_revision != current_revision:
                    raise IntentRepositoryConflictError("objective revision CAS is stale")
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE objectives SET
                        objective_alias = ?, parent_objective_id = ?,
                        title = ?, status = ?, priority = ?, updated_at = ?,
                        revision = ?, body_json = ?
                    WHERE objective_id = ? AND revision = ?
                    """,
                    [
                        alias,
                        parent,
                        title_text,
                        status_text,
                        priority_text,
                        now,
                        revision,
                        _canonical(body_map, noun="objective body"),
                        oid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO objective_revisions (
                    objective_id, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="objective revision body"),
                    now,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.OBJECTIVE_UPSERTED,
                subject_id=oid,
                body={
                    "objective_id": oid,
                    "objective_alias": alias,
                    "parent_objective_id": parent,
                    "title": title_text,
                    "status": status_text,
                    "priority": priority_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def get_objective(self, objective_id: str) -> Mapping[str, Any] | None:
        oid = _identifier(objective_id, noun="objective_id")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT objective_id, objective_alias, parent_objective_id,
                       title, status, priority, created_at, updated_at,
                       revision, body_json
                FROM objectives WHERE objective_id = ? OR objective_alias = ?
                LIMIT 1
                """,
                [oid, oid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2] or ""),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "created_at": str(row[6]),
                "updated_at": str(row[7]),
                "revision": int(row[8]),
                "body": _decode_json(row[9], noun="objective body"),
            }
        )

    # -- goals ---------------------------------------------------------------

    def upsert_goal(
        self,
        *,
        goal_cid: str,
        goal_alias: str,
        title: str,
        objective_id: str = "",
        parent_goal_cid: str = "",
        ordinal: int = 0,
        status: str = "open",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        gcid = _identifier(goal_cid, noun="goal_cid")
        alias = _identifier(goal_alias, noun="goal_alias")
        title_text = str(title or "").strip() or alias
        oid = _optional_identifier(objective_id, noun="objective_id")
        parent = _optional_identifier(parent_goal_cid, noun="parent_goal_cid")
        ord_value = _nonneg_int(ordinal, noun="ordinal")
        status_text = str(status or "open").strip().lower()
        body_map = _mapping(body, noun="goal body")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            existing = connection.execute(
                "SELECT revision FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "goal CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO goals (
                        goal_cid, goal_alias, objective_id, parent_goal_cid,
                        ordinal, title, status, created_at, updated_at,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        gcid,
                        alias,
                        oid,
                        parent,
                        ord_value,
                        title_text,
                        status_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="goal body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if expected_revision is not None and expected_revision != current_revision:
                    raise IntentRepositoryConflictError("goal revision CAS is stale")
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE goals SET
                        goal_alias = ?, objective_id = ?, parent_goal_cid = ?,
                        ordinal = ?, title = ?, status = ?, updated_at = ?,
                        revision = ?, body_json = ?
                    WHERE goal_cid = ? AND revision = ?
                    """,
                    [
                        alias,
                        oid,
                        parent,
                        ord_value,
                        title_text,
                        status_text,
                        now,
                        revision,
                        _canonical(body_map, noun="goal body"),
                        gcid,
                        current_revision,
                    ],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_UPSERTED,
                subject_id=gcid,
                body={
                    "goal_cid": gcid,
                    "goal_alias": alias,
                    "objective_id": oid,
                    "parent_goal_cid": parent,
                    "ordinal": ord_value,
                    "title": title_text,
                    "status": status_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def get_goal(self, goal_cid: str) -> Mapping[str, Any] | None:
        gcid = _identifier(goal_cid, noun="goal_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                       ordinal, title, status, created_at, updated_at,
                       revision, body_json
                FROM goals WHERE goal_cid = ? OR goal_alias = ?
                LIMIT 1
                """,
                [gcid, gcid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2] or ""),
                "parent_goal_cid": str(row[3] or ""),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "created_at": str(row[7]),
                "updated_at": str(row[8]),
                "revision": int(row[9]),
                "body": _decode_json(row[10], noun="goal body"),
            }
        )

    def cas_goal_status(
        self,
        *,
        goal_cid: str,
        expected_revision: int,
        new_status: str,
        receipt: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        """CAS one goal status after child tasks and child goals are complete."""

        gcid = _identifier(goal_cid, noun="goal_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        status_text = _status(new_status, allowed=_GOAL_STATUSES, noun="goal")
        receipt_map = _mapping(receipt, noun="goal status receipt")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            rows = connection.execute(
                """
                SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                       ordinal, title, status, revision, body_json
                FROM goals WHERE goal_cid = ? OR goal_alias = ?
                ORDER BY goal_cid LIMIT 2
                """,
                [gcid, gcid],
            ).fetchall()
            if not rows:
                raise KeyError(gcid)
            if len(rows) > 1:
                raise IntentRepositoryIntegrityError("goal CID/alias lookup is ambiguous")
            goal_row = rows[0]
            resolved_cid = str(goal_row[0])
            previous_status = str(goal_row[6])
            current_revision = int(goal_row[7])
            if current_revision != expected:
                raise IntentRepositoryConflictError("goal revision CAS is stale")
            if previous_status == status_text:
                return IntentReceipt(
                    event_id="",
                    event_type=IntentEventType.GOAL_UPSERTED.value,
                    global_sequence=self._next_global_sequence(connection) - 1,
                    recorded_at=now,
                    subject_id=resolved_cid,
                    revision=current_revision,
                    changed=False,
                    details=MappingProxyType(
                        {
                            "goal_cid": resolved_cid,
                            "goal_alias": str(goal_row[1]),
                            "status": status_text,
                            "previous_status": previous_status,
                        }
                    ),
                )

            if status_text in _GOAL_COMPLETED_STATUSES:
                incomplete_tasks = [
                    str(item[0] or item[1] or "")
                    for item in connection.execute(
                        """
                        SELECT task_alias, task_cid, status
                        FROM tasks WHERE goal_cid = ?
                        ORDER BY ordinal, task_alias
                        """,
                        [resolved_cid],
                    ).fetchall()
                    if str(item[2] or "").strip().lower() not in _COMPLETED_STATUSES
                ]
                incomplete_children = [
                    str(item[0] or item[1] or "")
                    for item in connection.execute(
                        """
                        SELECT goal_alias, goal_cid, status
                        FROM goals WHERE parent_goal_cid = ?
                        ORDER BY ordinal, goal_alias
                        """,
                        [resolved_cid],
                    ).fetchall()
                    if str(item[2] or "").strip().lower()
                    not in _GOAL_COMPLETED_STATUSES
                ]
                missing = [
                    *(f"task:{alias}" for alias in incomplete_tasks if alias),
                    *(f"goal:{alias}" for alias in incomplete_children if alias),
                ]
                if missing:
                    raise IntentCompletionError(
                        "goal completion refused while children remain open: "
                        + ", ".join(missing)
                    )

            revision = current_revision + 1
            body_map = _decode_json(goal_row[8], noun="goal body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            if receipt_map:
                body_map["completion_receipt"] = receipt_map
            connection.execute(
                """
                UPDATE goals SET status = ?, updated_at = ?, revision = ?,
                    body_json = ?
                WHERE goal_cid = ? AND revision = ?
                """,
                [
                    status_text,
                    now,
                    revision,
                    _canonical(body_map, noun="goal body"),
                    resolved_cid,
                    current_revision,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_UPSERTED,
                subject_id=resolved_cid,
                body={
                    "goal_cid": resolved_cid,
                    "goal_alias": str(goal_row[1]),
                    "objective_id": str(goal_row[2] or ""),
                    "parent_goal_cid": str(goal_row[3] or ""),
                    "ordinal": int(goal_row[4]),
                    "title": str(goal_row[5]),
                    "previous_status": previous_status,
                    "status": status_text,
                    "revision": revision,
                    "receipt": receipt_map,
                    "recorded_at": now,
                    "body": body_map,
                },
            )

    @staticmethod
    def _current_task_completion_binding(
        task: Mapping[str, Any],
        receipt_rows: Sequence[Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Validate the one receipt bound to a task's current successful revision."""

        task_cid = str(task["task_cid"])
        task_alias = str(task["task_alias"])
        task_goal_cid = str(task["goal_cid"])
        task_revision = int(task["revision"])
        reasons: list[str] = []
        status = str(task["status"] or "").strip().lower()
        if status not in _SUCCESSFUL_TASK_STATUSES:
            reasons.append(f"task_status_not_successful:{status or 'empty'}")
            return None, reasons
        task_body = task.get("body")
        task_body = task_body if isinstance(task_body, Mapping) else {}
        current_control_receipt = task_body.get("completion_receipt")
        if not isinstance(current_control_receipt, Mapping) or not current_control_receipt:
            reasons.append("task_current_control_receipt_missing")
            return None, reasons

        matches: list[dict[str, Any]] = []
        for row in receipt_rows:
            if str(row[1]) != task_cid:
                continue
            body = _decode_json(row[9], noun="task completion receipt body")
            if not isinstance(body, Mapping) or body.get("revision") != task_revision:
                continue
            matches.append(
                {
                    "receipt_cid": str(row[0]),
                    "task_cid": str(row[1]),
                    "goal_cid": str(row[2]),
                    "evidence_digest": str(row[8]),
                    "body": dict(body),
                }
            )
        if len(matches) != 1:
            reasons.append("task_current_completion_receipt_population_not_exact")
            return None, reasons
        observed = matches[0]
        body = observed["body"]
        control_receipt = body.get("receipt")
        evidence_digests = body.get("evidence_digests")
        if (
            body.get("schema") != COMPLETION_EVIDENCE_SCHEMA
            or not isinstance(control_receipt, Mapping)
            or not control_receipt
            or not isinstance(evidence_digests, list)
            or dict(control_receipt) != dict(current_control_receipt)
            or observed["goal_cid"] != task_goal_cid
        ):
            reasons.append("task_current_completion_receipt_body_invalid")
            return None, reasons
        expected_evidence_digest = content_identity(
            {
                "task_cid": task_cid,
                "revision": task_revision,
                "receipt": dict(control_receipt),
                "evidence_digests": list(evidence_digests),
            }
        )
        expected_receipt_cid = content_identity(
            {
                "namespace": "completion-receipt",
                "task_cid": task_cid,
                "revision": task_revision,
                "evidence_digest": expected_evidence_digest,
            }
        )
        if (
            observed["evidence_digest"] != expected_evidence_digest
            or observed["receipt_cid"] != expected_receipt_cid
        ):
            reasons.append("task_current_completion_receipt_identity_invalid")
            return None, reasons
        return (
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "task_revision": task_revision,
                "completion_receipt_cid": expected_receipt_cid,
                "completion_evidence_digest": expected_evidence_digest,
                "control_receipt_id": content_identity(dict(control_receipt)),
            },
            reasons,
        )

    @staticmethod
    def _goal_settlement_counts(connection: Any) -> dict[str, int]:
        """Return fail-closed counts for mutable work that must be settled."""

        # Each mutable relation has an intentionally closed vocabulary.  A
        # positive-only filter (for example, ``state = 'accepted'``) is unsafe
        # because an unrecognised state silently disappears from the gate.  We
        # group the finite state/marker surface and make every unknown value or
        # impossible finish/release marker a typed, projected gate failure.
        # Empty vocabularies are deliberate: those normalized tables currently
        # have no compatible canonical writer.  Similarly named lane and queue
        # sidecars have different schemas and are settled by the separate
        # guarded runtime receipt; admitting their vocabularies here would
        # invent an adapter and hide orphaned canonical rows.
        relation_specs = (
            (
                "active_task_blocks",
                "task_blocks",
                "state",
                frozenset({"active"}),
                frozenset({"cleared"}),
                "cleared_at",
                frozenset({"cleared"}),
            ),
            (
                "active_task_assignments",
                "task_assignments",
                "state",
                frozenset(),
                frozenset(),
                "released_at",
                frozenset(),
            ),
            (
                "active_task_claims",
                "task_claims",
                "state",
                frozenset(),
                frozenset(),
                "released_at",
                frozenset(),
            ),
            (
                "active_resource_claims",
                "resource_claims",
                "state",
                frozenset(),
                frozenset(),
                None,
                frozenset(),
            ),
            (
                "active_path_claims",
                "path_claims",
                "state",
                frozenset(),
                frozenset(),
                None,
                frozenset(),
            ),
            (
                "active_leases",
                "leases",
                "state",
                frozenset({"accepted"}),
                frozenset({"released", "expired", "completed"}),
                None,
                frozenset(),
            ),
            (
                "active_maintenance_leases",
                "maintenance_leases",
                "state",
                frozenset({"active"}),
                frozenset({"released"}),
                "released_at",
                frozenset({"released"}),
            ),
            (
                "active_effect_claims",
                "effect_claims",
                "state",
                frozenset(),
                frozenset(),
                None,
                frozenset(),
            ),
            (
                "running_task_attempts",
                "task_attempts",
                "status",
                _ACTIVE_ATTEMPT_STATUSES,
                _TERMINAL_ATTEMPT_STATUSES,
                "finished_at",
                _TERMINAL_ATTEMPT_STATUSES,
            ),
            (
                "running_attempt_phases",
                "attempt_phases",
                "status",
                frozenset(),
                frozenset(),
                "exited_at",
                frozenset(),
            ),
            (
                "running_provider_invocations",
                "provider_invocations",
                "status",
                frozenset(),
                frozenset(),
                "finished_at",
                frozenset(),
            ),
            (
                "running_validation_runs",
                "validation_runs",
                "status",
                frozenset(),
                frozenset({"passed", "failed", "error", "skipped"}),
                "finished_at",
                frozenset({"passed", "failed", "error", "skipped"}),
            ),
            (
                "running_merge_attempts",
                "merge_attempts",
                "status",
                frozenset(),
                frozenset(),
                "finished_at",
                frozenset(),
            ),
            (
                "active_refill_epochs",
                "refill_epochs",
                "status",
                frozenset(),
                frozenset(),
                "finished_at",
                frozenset(),
            ),
            (
                "pending_recovery_actions",
                "recovery_actions",
                "status",
                frozenset(),
                frozenset(),
                None,
                frozenset(),
            ),
            (
                "unsettled_merge_queue_entries",
                "merge_queue_entries",
                "status",
                frozenset(),
                frozenset(),
                None,
                frozenset(),
            ),
        )
        counts: dict[str, int] = {}
        invalid_total = 0
        for (
            count_name,
            table,
            vocabulary_column,
            active_states,
            terminal_states,
            marker_column,
            marker_required_states,
        ) in relation_specs:
            marker_expression = (
                "CASE WHEN NULLIF(TRIM(COALESCE(CAST("
                f"{marker_column} AS VARCHAR), '')), '') IS NULL THEN 0 ELSE 1 END"
                if marker_column is not None
                else "0"
            )
            rows = connection.execute(
                f"SELECT {vocabulary_column}, {marker_expression} AS marker_set, "
                f"COUNT(*) FROM {table} GROUP BY {vocabulary_column}, marker_set"
            ).fetchall()
            active_count = 0
            invalid_count = 0
            for row in rows:
                raw_state = str(row[0] or "")
                state = raw_state.strip().lower()
                marker_set = int(row[1] or 0) == 1
                row_count = int(row[2] or 0)
                if raw_state != state or state not in active_states | terminal_states:
                    invalid_count += row_count
                    continue
                if state in active_states:
                    active_count += row_count
                    if marker_set:
                        invalid_count += row_count
                elif state in marker_required_states and not marker_set:
                    invalid_count += row_count
            counts[count_name] = active_count
            counts[f"invalid_{table}_rows"] = invalid_count
            invalid_total += invalid_count
        counts["invalid_settlement_rows"] = invalid_total
        return counts

    @classmethod
    def _goal_authority_state_on(
        cls,
        connection: Any,
        specification: Mapping[str, Any],
        *,
        candidate_root_completion_gate: Mapping[str, Any] | None = None,
        root_gate_context: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = _goal_completion_authority_spec(specification)
        goal_rows = connection.execute(
            """
            SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                   ordinal, title, status, revision, body_json
            FROM goals ORDER BY ordinal, goal_alias, goal_cid
            """
        ).fetchall()
        task_rows = connection.execute(
            """
            SELECT task_cid, task_alias, goal_cid, status, revision, body_json
            FROM tasks ORDER BY task_alias, task_cid
            """
        ).fetchall()
        edge_rows = connection.execute(
            """
            SELECT parent_goal_cid, child_goal_cid, edge_kind
            FROM goal_edges ORDER BY edge_kind, parent_goal_cid, child_goal_cid
            """
        ).fetchall()
        dependency_rows = connection.execute(
            """
            SELECT task_cid, dependency_task_cid, kind
            FROM task_dependencies ORDER BY task_cid, dependency_task_cid, kind
            """
        ).fetchall()
        receipt_rows = connection.execute(
            """
            SELECT receipt_cid, task_cid, goal_cid, attempt_id,
                   claim_cid, fencing_token, completed_at,
                   validation_run_id, evidence_digest, body_json
            FROM completion_receipts
            ORDER BY task_cid, completed_at, receipt_cid
            """
        ).fetchall()
        goal_receipt_event_rows = connection.execute(
            "SELECT body_json FROM domain_events WHERE event_type = ? "
            "ORDER BY global_sequence",
            [IntentEventType.GOAL_UPSERTED.value],
        ).fetchall()

        expected_goal_rows = [
            (
                item["goal_cid"],
                item["goal_alias"],
                item["parent_goal_cid"],
                int(item["ordinal"]),
            )
            for item in spec["goals"]
        ]
        observed_goal_rows = [
            (str(row[0]), str(row[1]), str(row[3] or ""), int(row[4]))
            for row in goal_rows
        ]
        if observed_goal_rows != expected_goal_rows:
            raise IntentRepositoryIntegrityError(
                "database goal population differs from exact completion authority"
            )
        expected_task_rows = [
            (item["task_cid"], item["task_alias"], item["goal_cid"])
            for item in spec["tasks"]
        ]
        observed_task_rows = [
            (str(row[0]), str(row[1]), str(row[2])) for row in task_rows
        ]
        if observed_task_rows != expected_task_rows:
            raise IntentRepositoryIntegrityError(
                "database task population differs from exact goal authority"
            )
        observed_edges = [
            {
                "parent_goal_cid": str(row[0]),
                "child_goal_cid": str(row[1]),
                "edge_kind": str(row[2]),
            }
            for row in edge_rows
        ]
        if observed_edges != spec["goal_edges"]:
            raise IntentRepositoryIntegrityError(
                "database goal edges differ from exact completion authority"
            )
        observed_task_dependencies = [
            {
                "task_cid": str(row[0]),
                "dependency_task_cid": str(row[1]),
                "kind": str(row[2]),
            }
            for row in dependency_rows
        ]
        if observed_task_dependencies != spec["task_dependencies"]:
            raise IntentRepositoryIntegrityError(
                "database task dependencies differ from exact completion authority"
            )

        terminal_task_cid = spec["terminal_task_cid"]
        terminal_output_rows = connection.execute(
            "SELECT path FROM task_outputs WHERE task_cid = ? ORDER BY ordinal, path",
            [terminal_task_cid],
        ).fetchall()
        terminal_acceptance_rows = connection.execute(
            "SELECT criterion FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
            [terminal_task_cid],
        ).fetchall()
        terminal_validation_rows = connection.execute(
            "SELECT argv_json FROM task_validations WHERE task_cid = ? ORDER BY ordinal",
            [terminal_task_cid],
        ).fetchall()
        observed_terminal_outputs = [str(row[0]) for row in terminal_output_rows]
        observed_terminal_acceptance = [
            str(row[0]) for row in terminal_acceptance_rows
        ]
        observed_terminal_validations: list[list[str]] = []
        for row in terminal_validation_rows:
            argv = _decode_json(row[0], noun="terminal validation argv")
            if (
                isinstance(argv, (str, bytes, bytearray))
                or not isinstance(argv, Sequence)
            ):
                raise IntentRepositoryIntegrityError(
                    "database terminal validation argv is malformed"
                )
            observed_terminal_validations.append([str(part) for part in argv])
        terminal_contract = spec["terminal_report_contract"]
        if (
            observed_terminal_outputs
            != terminal_contract["declared_output_paths"]
            or observed_terminal_acceptance
            != terminal_contract["acceptance_criteria"]
            or observed_terminal_validations
            != terminal_contract["validation_commands"]
        ):
            raise IntentRepositoryIntegrityError(
                "database terminal report contract differs from exact completion authority"
            )
        spec_task_cid_by_alias = {
            item["task_alias"]: item["task_cid"] for item in spec["tasks"]
        }
        for producer_alias, expected_paths in terminal_contract[
            "producer_output_paths"
        ].items():
            producer_task_cid = spec_task_cid_by_alias[producer_alias]
            producer_rows = connection.execute(
                "SELECT path FROM task_outputs WHERE task_cid = ? ORDER BY ordinal",
                [producer_task_cid],
            ).fetchall()
            if [str(row[0]) for row in producer_rows] != expected_paths:
                raise IntentRepositoryIntegrityError(
                    "database terminal report producer outputs differ from exact authority"
                )
            producer_validation_rows = connection.execute(
                "SELECT argv_json FROM task_validations "
                "WHERE task_cid = ? ORDER BY ordinal",
                [producer_task_cid],
            ).fetchall()
            observed_producer_validations: list[list[str]] = []
            for row in producer_validation_rows:
                argv = _decode_json(row[0], noun="producer task validation argv")
                if (
                    not isinstance(argv, Sequence)
                    or isinstance(argv, (str, bytes, bytearray))
                ):
                    raise IntentRepositoryIntegrityError(
                        "database producer validation argv is malformed"
                    )
                observed_producer_validations.append([str(part) for part in argv])
            if observed_producer_validations != terminal_contract[
                "producer_validation_commands"
            ][producer_alias]:
                raise IntentRepositoryIntegrityError(
                    "database terminal report producer validations differ from exact authority"
                )

        tasks: dict[str, dict[str, Any]] = {}
        task_alias_by_cid: dict[str, str] = {}
        for row in task_rows:
            body = _decode_json(row[5], noun="task body")
            task = {
                "task_cid": str(row[0]),
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "status": str(row[3]),
                "revision": int(row[4]),
                "body": body if isinstance(body, Mapping) else {},
            }
            tasks[task["task_cid"]] = task
            task_alias_by_cid[task["task_cid"]] = task["task_alias"]
        task_bindings: dict[str, dict[str, Any] | None] = {}
        task_reasons: dict[str, list[str]] = {}
        for task_cid, task in tasks.items():
            binding, reasons = cls._current_task_completion_binding(task, receipt_rows)
            task_bindings[task_cid] = binding
            task_reasons[task_cid] = reasons

        dependency_failures: list[dict[str, str]] = []
        for row in dependency_rows:
            owner = str(row[0])
            dependency = str(row[1])
            kind = str(row[2])
            if owner not in tasks or dependency not in tasks:
                dependency_failures.append(
                    {
                        "task_cid": owner,
                        "dependency_task_cid": dependency,
                        "kind": kind,
                        "reason": "unknown_task_dependency_endpoint",
                    }
                )
                continue
            if task_bindings.get(dependency) is None:
                dependency_failures.append(
                    {
                        "task_cid": owner,
                        "dependency_task_cid": dependency,
                        "kind": kind,
                        "reason": "dependency_not_successfully_receipted",
                    }
                )

        settlement_counts = cls._goal_settlement_counts(connection)
        terminal_task = tasks[spec["terminal_task_cid"]]
        terminal_task_binding = task_bindings.get(spec["terminal_task_cid"])
        terminal_producer_task_cids = sorted(
            {
                edge["dependency_task_cid"]
                for edge in spec["task_dependencies"]
                if edge["task_cid"] == spec["terminal_task_cid"]
            },
            key=lambda task_cid: task_alias_by_cid[task_cid],
        )
        terminal_producer_receipts = {
            task_alias_by_cid[task_cid]: str(
                task_bindings[task_cid]["completion_receipt_cid"]
            )
            for task_cid in terminal_producer_task_cids
            if task_bindings.get(task_cid) is not None
        }
        terminal_producer_receipts_satisfied = bool(
            len(terminal_producer_task_cids) == 4
            and len(terminal_producer_receipts)
            == len(terminal_producer_task_cids)
        )
        terminal_producer_portal_bindings: dict[str, dict[str, str]] = {}
        portal_validation_fields = {
            "outcome",
            "evidence_digest",
            "argv",
            "validator",
            "task_cid",
            "attempt_id",
            "portal_receipt_id",
            "portal_completion_binding",
        }
        replayed_portal_validation_fields = portal_validation_fields | {"replayed"}
        for producer_task_cid in terminal_producer_task_cids:
            producer_task = tasks[producer_task_cid]
            producer_alias = task_alias_by_cid[producer_task_cid]
            producer_control_receipt = producer_task["body"].get(
                "completion_receipt"
            )
            producer_validation = (
                producer_control_receipt.get("validation")
                if isinstance(producer_control_receipt, Mapping)
                else None
            )
            producer_portal_binding: dict[str, str] | None = None
            if isinstance(producer_validation, Mapping) and isinstance(
                producer_validation.get("portal_completion_binding"), Mapping
            ):
                try:
                    producer_portal_binding = _database_portal_completion_binding(
                        producer_validation["portal_completion_binding"]
                    )
                except IntentRepositoryError:
                    producer_portal_binding = None
            producer_receipt_bodies = [
                _decode_json(row[9], noun="producer completion receipt body")
                for row in receipt_rows
                if task_bindings.get(producer_task_cid) is not None
                and str(row[0])
                == str(task_bindings[producer_task_cid]["completion_receipt_cid"])
            ]
            producer_evidence_digests = (
                producer_receipt_bodies[0].get("evidence_digests")
                if len(producer_receipt_bodies) == 1
                and isinstance(producer_receipt_bodies[0], Mapping)
                else None
            )
            if (
                task_bindings.get(producer_task_cid) is not None
                and isinstance(producer_control_receipt, Mapping)
                and producer_control_receipt.get("operation") == "database_complete"
                and isinstance(producer_validation, Mapping)
                and (
                    set(producer_validation) == portal_validation_fields
                    or (
                        set(producer_validation)
                        == replayed_portal_validation_fields
                        and type(producer_validation.get("replayed")) is bool
                    )
                )
                and producer_validation.get("outcome") == "passed"
                and producer_validation.get("argv")
                == list(_TERMINAL_REPORT_VALIDATION_ARGV)
                and producer_validation.get("validator")
                == _TERMINAL_REPORT_VALIDATOR
                and producer_validation.get("task_cid") == producer_task_cid
                and producer_validation.get("attempt_id")
                == producer_control_receipt.get("attempt_id")
                and producer_validation.get("evidence_digest")
                == producer_control_receipt.get("evidence_digest")
                and producer_portal_binding is not None
                and producer_portal_binding["task_cid"] == producer_task_cid
                and producer_portal_binding["attempt_id"]
                == producer_control_receipt.get("attempt_id")
                and producer_portal_binding["portal_receipt_id"]
                == producer_validation.get("portal_receipt_id")
                and producer_portal_binding["evidence_digest"]
                == producer_control_receipt.get("evidence_digest")
                and isinstance(producer_evidence_digests, list)
                and producer_evidence_digests
                == [producer_control_receipt.get("evidence_digest")]
            ):
                terminal_producer_portal_bindings[producer_alias] = (
                    producer_portal_binding
                )
        terminal_producer_portal_bindings_satisfied = bool(
            len(terminal_producer_portal_bindings)
            == len(terminal_producer_task_cids)
            == 4
        )
        terminal_control_receipt = terminal_task["body"].get("completion_receipt")
        terminal_validation = (
            terminal_control_receipt.get("validation")
            if isinstance(terminal_control_receipt, Mapping)
            else None
        )
        terminal_portal_completion_binding: dict[str, str] | None = None
        if isinstance(terminal_validation, Mapping) and isinstance(
            terminal_validation.get("portal_completion_binding"), Mapping
        ):
            try:
                terminal_portal_completion_binding = (
                    _database_portal_completion_binding(
                        terminal_validation["portal_completion_binding"]
                    )
                )
            except IntentRepositoryError:
                terminal_portal_completion_binding = None
        terminal_receipt_body: Mapping[str, Any] | None = None
        if terminal_task_binding is not None:
            matching_terminal_receipts: list[Mapping[str, Any]] = []
            for row in receipt_rows:
                if str(row[0]) != str(
                    terminal_task_binding["completion_receipt_cid"]
                ):
                    continue
                decoded = _decode_json(row[9], noun="terminal completion receipt body")
                if isinstance(decoded, Mapping):
                    matching_terminal_receipts.append(decoded)
            if len(matching_terminal_receipts) == 1:
                terminal_receipt_body = matching_terminal_receipts[0]
        terminal_evidence_digests = (
            terminal_receipt_body.get("evidence_digests")
            if isinstance(terminal_receipt_body, Mapping)
            else None
        )
        terminal_validation_lineage: dict[str, str] | None = None
        if (
            isinstance(terminal_control_receipt, Mapping)
            and isinstance(terminal_validation, Mapping)
        ):
            validation_run_rows = [
                tuple(row[index] for index in range(5))
                for row in connection.execute(
                    """
                    SELECT run_id, attempt_id, status, command_digest, body_json
                    FROM validation_runs
                    WHERE task_cid = ?
                    ORDER BY run_id
                    """,
                    [terminal_task["task_cid"]],
                ).fetchall()
            ]
            validation_result_rows = [
                tuple(row[index] for index in range(5))
                for row in connection.execute(
                    """
                    SELECT run_id, result_id, outcome, evidence_digest, body_json
                    FROM validation_results
                    WHERE task_cid = ?
                    ORDER BY run_id, result_id
                    """,
                    [terminal_task["task_cid"]],
                ).fetchall()
            ]
            # Quack materializes each remote scan independently and cannot execute
            # the corresponding two-table streaming join.  Preserve every match
            # (including duplicates) so ambiguous lineage still fails closed below.
            validation_lineage_rows = sorted(
                (
                    (*run_row, *result_row[1:])
                    for run_row in validation_run_rows
                    for result_row in validation_result_rows
                    if str(run_row[0]) == str(result_row[0])
                ),
                key=lambda row: (str(row[0]), str(row[5])),
            )
            matching_validation_lineage: list[dict[str, str]] = []
            expected_validation_body = dict(terminal_validation)
            expected_validation_run_body = {
                "argv": list(_TERMINAL_REPORT_VALIDATION_ARGV),
                **expected_validation_body,
            }
            for row in validation_lineage_rows:
                run_body = _decode_json(row[4], noun="terminal validation run body")
                result_body = _decode_json(
                    row[8], noun="terminal validation result body"
                )
                if (
                    str(row[1]) != str(terminal_control_receipt.get("attempt_id") or "")
                    or str(row[2]) != "passed"
                    or str(row[3])
                    != content_identity(
                        {"argv": list(_TERMINAL_REPORT_VALIDATION_ARGV)}
                    )
                    or run_body != expected_validation_run_body
                    or str(row[6]) != "passed"
                    or str(row[7])
                    != str(terminal_control_receipt.get("evidence_digest") or "")
                    or result_body != expected_validation_body
                ):
                    continue
                expected_evidence_id = content_identity(
                    {
                        "task_cid": terminal_task["task_cid"],
                        "evidence_kind": "validation",
                        "digest": str(row[7]),
                        "run_id": str(row[0]),
                    }
                )
                evidence_rows = connection.execute(
                    """
                    SELECT evidence_id, evidence_kind, digest, body_json
                    FROM evidence_nodes
                    WHERE task_cid = ? AND evidence_id = ?
                    """,
                    [terminal_task["task_cid"], expected_evidence_id],
                ).fetchall()
                if len(evidence_rows) != 1:
                    continue
                evidence_row = evidence_rows[0]
                evidence_body = _decode_json(
                    evidence_row[3], noun="terminal validation evidence body"
                )
                if (
                    str(evidence_row[0]) != expected_evidence_id
                    or str(evidence_row[1]) != "validation"
                    or str(evidence_row[2]) != str(row[7])
                    or evidence_body
                    != {
                        "run_id": str(row[0]),
                        "result_id": str(row[5]),
                        "argv": list(_TERMINAL_REPORT_VALIDATION_ARGV),
                        "outcome": "passed",
                    }
                ):
                    continue
                matching_validation_lineage.append(
                    {
                        "validation_run_id": str(row[0]),
                        "validation_result_id": str(row[5]),
                        "validation_evidence_id": expected_evidence_id,
                    }
                )
            if len(matching_validation_lineage) == 1:
                terminal_validation_lineage = matching_validation_lineage[0]
        terminal_production_receipt_satisfied = bool(
            terminal_task_binding is not None
            and isinstance(terminal_control_receipt, Mapping)
            and terminal_control_receipt.get("operation") == "database_complete"
            and isinstance(terminal_validation, Mapping)
            and terminal_validation.get("outcome") == "passed"
            and terminal_validation.get("argv")
            == list(_TERMINAL_REPORT_VALIDATION_ARGV)
            and terminal_validation.get("validator") == _TERMINAL_REPORT_VALIDATOR
            and terminal_validation.get("task_cid") == terminal_task["task_cid"]
            and terminal_validation.get("attempt_id")
            == terminal_control_receipt.get("attempt_id")
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(terminal_validation.get("portal_receipt_id") or ""),
            )
            is not None
            and terminal_validation.get("evidence_digest")
            == terminal_control_receipt.get("evidence_digest")
            and terminal_portal_completion_binding is not None
            and terminal_portal_completion_binding["task_cid"]
            == terminal_task["task_cid"]
            and terminal_portal_completion_binding["attempt_id"]
            == terminal_control_receipt.get("attempt_id")
            and terminal_portal_completion_binding["portal_receipt_id"]
            == terminal_validation.get("portal_receipt_id")
            and terminal_portal_completion_binding["evidence_digest"]
            == terminal_control_receipt.get("evidence_digest")
            and isinstance(terminal_evidence_digests, list)
            and len(terminal_evidence_digests) == 1
            and terminal_evidence_digests[0]
            == terminal_control_receipt.get("evidence_digest")
            and terminal_validation_lineage is not None
        )
        database_gates = {
            "exact_goal_population": True,
            "exact_goal_edges": True,
            "exact_task_population": True,
            "all_tasks_successful": all(
                str(task["status"] or "").strip().lower()
                in _SUCCESSFUL_TASK_STATUSES
                for task in tasks.values()
            ),
            "all_current_task_receipts_valid": all(
                binding is not None for binding in task_bindings.values()
            ),
            "all_task_dependencies_successful": not dependency_failures,
            "settlement_state_integrity": (
                settlement_counts["invalid_settlement_rows"] == 0
            ),
            "blocking_obligations_empty": all(
                settlement_counts[name] == 0
                for name in (
                    "active_task_blocks",
                    "active_refill_epochs",
                    "pending_recovery_actions",
                )
            ),
            "active_mutating_claims_empty": all(
                settlement_counts[name] == 0
                for name in (
                    "active_task_assignments",
                    "active_task_claims",
                    "active_resource_claims",
                    "active_path_claims",
                    "active_leases",
                    "active_maintenance_leases",
                    "active_effect_claims",
                )
            ),
            "attempts_and_validations_settled": all(
                settlement_counts[name] == 0
                for name in (
                    "running_task_attempts",
                    "running_attempt_phases",
                    "running_provider_invocations",
                    "running_validation_runs",
                    "running_merge_attempts",
                )
            ),
            "merge_queue_settled": settlement_counts["unsettled_merge_queue_entries"] == 0,
            "runtime_settlement_gate_satisfied": False,
            "retired_ready_tasks_satisfied": False,
            "terminal_report_contract_satisfied": True,
            "terminal_report_completion_receipt_satisfied": (
                terminal_production_receipt_satisfied
            ),
            "terminal_report_validation_lineage_satisfied": (
                terminal_validation_lineage is not None
            ),
            "terminal_report_producer_receipts_satisfied": (
                terminal_producer_receipts_satisfied
            ),
            "terminal_report_producer_portal_bindings_satisfied": (
                terminal_producer_portal_bindings_satisfied
            ),
            "terminal_report_producer_artifacts_satisfied": False,
            "terminal_report_producer_receipt_bindings_satisfied": False,
            "terminal_report_gate_satisfied": False,
            "ducklake_non_authoritative": True,
        }

        goals_by_cid: dict[str, dict[str, Any]] = {}
        for row in goal_rows:
            body = _decode_json(row[8], noun="goal body")
            goals_by_cid[str(row[0])] = {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2] or ""),
                "parent_goal_cid": str(row[3] or ""),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "body": body if isinstance(body, Mapping) else {},
            }
        emitted_goal_receipts: set[tuple[str, int, str]] = set()
        for event_row in goal_receipt_event_rows:
            envelope = _decode_json(event_row[0], noun="goal receipt event")
            event_body = envelope.get("body") if isinstance(envelope, Mapping) else None
            emitted_receipt = (
                event_body.get("receipt")
                if isinstance(event_body, Mapping)
                else None
            )
            if not isinstance(emitted_receipt, Mapping):
                continue
            try:
                emitted_goal_receipts.add(
                    (
                        str(event_body.get("goal_cid") or ""),
                        int(event_body.get("revision") or 0),
                        _canonical(emitted_receipt, noun="emitted goal receipt"),
                    )
                )
            except (TypeError, ValueError, IntentRepositoryError):
                continue
        direct_tasks: dict[str, list[str]] = {
            item["goal_cid"]: [] for item in spec["goals"]
        }
        for task in spec["tasks"]:
            direct_tasks[task["goal_cid"]].append(task["task_cid"])
        child_goals: dict[str, list[str]] = {
            item["goal_cid"]: [] for item in spec["goals"]
        }
        dependency_goals: dict[str, list[str]] = {
            item["goal_cid"]: [] for item in spec["goals"]
        }
        for edge in spec["goal_edges"]:
            if edge["edge_kind"] == "goal_parent":
                child_goals[edge["parent_goal_cid"]].append(edge["child_goal_cid"])
            else:
                dependency_goals[edge["child_goal_cid"]].append(
                    edge["parent_goal_cid"]
                )

        candidate_gate: dict[str, Any] | None = None
        if candidate_root_completion_gate is not None:
            candidate_gate = _goal_root_completion_gate(
                candidate_root_completion_gate,
                authority_spec_id=spec["authority_spec_id"],
            )
            if candidate_gate.get("completion_policy") != spec["completion_policy"]:
                raise IntentRepositoryIntegrityError(
                    "candidate root completion gate policy differs from its authority spec"
                )
        goal_bindings: dict[str, dict[str, Any] | None] = {}
        goal_reasons: dict[str, list[str]] = {}
        completion_inputs: dict[str, dict[str, Any]] = {}
        goal_projection: list[dict[str, Any]] = []
        ready_goal_cids: list[str] = []
        invalid_goal_cids: list[str] = []
        incomplete_goal_cids: list[str] = []
        backfill_allowlist = set(spec["receipt_backfill_goal_cids"])
        terminal_report_gate_evidence: Mapping[str, Any] | None = None

        for goal_cid in spec["topological_goal_cids"]:
            goal = goals_by_cid[goal_cid]
            reasons: list[str] = []
            task_receipt_values: list[dict[str, Any]] = []
            for task_cid in sorted(
                direct_tasks[goal_cid], key=lambda value: task_alias_by_cid[value]
            ):
                binding = task_bindings.get(task_cid)
                if binding is None:
                    reasons.extend(
                        f"task:{task_alias_by_cid[task_cid]}:{reason}"
                        for reason in task_reasons[task_cid]
                    )
                else:
                    task_receipt_values.append(dict(binding))

            child_receipts: list[dict[str, Any]] = []
            for child_cid in sorted(
                child_goals[goal_cid], key=lambda value: goals_by_cid[value]["goal_alias"]
            ):
                binding = goal_bindings.get(child_cid)
                if binding is None:
                    reasons.append(
                        f"child_goal:{goals_by_cid[child_cid]['goal_alias']}:"
                        "current_receipt_missing_or_invalid"
                    )
                else:
                    child_receipts.append(dict(binding))
            dependency_receipts: list[dict[str, Any]] = []
            for dependency_cid in sorted(
                dependency_goals[goal_cid],
                key=lambda value: goals_by_cid[value]["goal_alias"],
            ):
                binding = goal_bindings.get(dependency_cid)
                if binding is None:
                    reasons.append(
                        f"dependency_goal:{goals_by_cid[dependency_cid]['goal_alias']}:"
                        "current_receipt_missing_or_invalid"
                    )
                else:
                    dependency_receipts.append(dict(binding))

            status = str(goal["status"] or "").strip().lower()
            completed = status in _GOAL_COMPLETED_STATUSES
            stored_receipt = goal["body"].get("completion_receipt")
            is_root = goal_cid == spec["root_goal_cid"]
            stored_root_gate: dict[str, Any] | None = None
            if (
                is_root
                and isinstance(stored_receipt, Mapping)
                and isinstance(stored_receipt.get("root_completion_gate"), Mapping)
            ):
                try:
                    stored_root_gate = _goal_root_completion_gate(
                        stored_receipt["root_completion_gate"],
                        authority_spec_id=spec["authority_spec_id"],
                    )
                    if stored_root_gate.get("completion_policy") != spec["completion_policy"]:
                        raise IntentRepositoryIntegrityError(
                            "stored root completion gate policy differs from its authority spec"
                        )
                except IntentRepositoryError:
                    reasons.append("root_completion_gate_invalid")
            root_gate_for_current_receipt = (
                stored_root_gate if is_root and completed else candidate_gate if is_root else None
            )
            if is_root:
                candidate_gate_admitted = bool(
                    candidate_gate is not None
                    and (
                        (
                            not completed
                            and not candidate_gate.get("predecessor_gate_id")
                        )
                        or (
                            completed
                            and stored_root_gate is not None
                            and (
                                candidate_gate.get("gate_id")
                                == stored_root_gate.get("gate_id")
                                or (
                                    candidate_gate.get("predecessor_gate_id")
                                    == stored_root_gate.get("gate_id")
                                    and int(candidate_gate.get("owner_generation") or 0)
                                    > int(stored_root_gate.get("owner_generation") or 0)
                                )
                            )
                        )
                    )
                )
                if candidate_gate is not None and not candidate_gate_admitted:
                    reasons.append("root_completion_gate_predecessor_or_generation_invalid")
                effective_candidate_gate = (
                    candidate_gate if candidate_gate_admitted else None
                )
                current_gate = effective_candidate_gate or root_gate_for_current_receipt
                context = (
                    root_gate_context
                    if isinstance(root_gate_context, Mapping)
                    else {}
                )
                gate_runtime_binding = (
                    current_gate.get("runtime_settlement_binding")
                    if isinstance(current_gate, Mapping)
                    else None
                )
                context_runtime_binding: Mapping[str, Any] | None = None
                if isinstance(context.get("runtime_settlement_binding"), Mapping):
                    try:
                        context_runtime_binding = _goal_runtime_settlement_binding(
                            context["runtime_settlement_binding"]
                        )
                    except IntentRepositoryError:
                        context_runtime_binding = None
                database_gates["runtime_settlement_gate_satisfied"] = bool(
                    isinstance(gate_runtime_binding, Mapping)
                    and context_runtime_binding is not None
                    and dict(gate_runtime_binding) == dict(context_runtime_binding)
                )
                expected_retired_ready_task_cids = sorted(
                    task["task_cid"]
                    for task in spec["tasks"]
                    if task["task_alias"] in {"VRIF-013", "VRIF-014", "VRIF-015"}
                )
                observed_retired_ready_task_cids = (
                    list(gate_runtime_binding.get("retired_ready_task_cids") or [])
                    if isinstance(gate_runtime_binding, Mapping)
                    else []
                )
                database_gates["retired_ready_tasks_satisfied"] = bool(
                    len(expected_retired_ready_task_cids) == 3
                    and set(expected_retired_ready_task_cids).issubset(
                        observed_retired_ready_task_cids
                    )
                    and all(
                        task_cid in tasks
                        and str(tasks[task_cid]["status"] or "").strip().lower()
                        in _SUCCESSFUL_TASK_STATUSES
                        and task_bindings.get(task_cid) is not None
                        for task_cid in observed_retired_ready_task_cids
                    )
                )
                terminal_gate_evidence = (
                    current_gate.get("terminal_report_evidence")
                    if isinstance(current_gate, Mapping)
                    else None
                )
                terminal_report_gate_evidence = (
                    terminal_gate_evidence
                    if isinstance(terminal_gate_evidence, Mapping)
                    else None
                )
                terminal_artifacts = (
                    terminal_gate_evidence.get("report_artifacts")
                    if isinstance(terminal_gate_evidence, Mapping)
                    else None
                )
                terminal_producer_artifacts = (
                    terminal_gate_evidence.get("producer_artifacts")
                    if isinstance(terminal_gate_evidence, Mapping)
                    else None
                )
                observed_producer_output_paths = {
                    str(item.get("task_alias") or ""): [
                        str(artifact.get("path") or "")
                        for artifact in item.get("artifacts", [])
                        if isinstance(artifact, Mapping)
                    ]
                    for item in terminal_producer_artifacts.get("tasks", [])
                    if isinstance(item, Mapping)
                } if isinstance(terminal_producer_artifacts, Mapping) else {}
                database_gates["terminal_report_producer_artifacts_satisfied"] = (
                    observed_producer_output_paths
                    == {
                        alias: sorted(paths)
                        for alias, paths in terminal_contract[
                            "producer_output_paths"
                        ].items()
                    }
                )
                artifact_bundle_by_alias = {
                    str(item.get("task_alias") or ""): str(
                        item.get("bundle_id") or ""
                    )
                    for item in terminal_producer_artifacts.get("tasks", [])
                    if isinstance(item, Mapping)
                } if isinstance(terminal_producer_artifacts, Mapping) else {}
                producer_receipt_binding_rows = (
                    terminal_gate_evidence.get("producer_receipt_bindings")
                    if isinstance(terminal_gate_evidence, Mapping)
                    else None
                )
                observed_producer_receipt_bindings = {
                    str(item.get("task_alias") or ""): item
                    for item in producer_receipt_binding_rows or []
                    if isinstance(item, Mapping)
                } if isinstance(producer_receipt_binding_rows, list) else {}
                database_gates[
                    "terminal_report_producer_receipt_bindings_satisfied"
                ] = bool(
                    terminal_producer_portal_bindings_satisfied
                    and set(observed_producer_receipt_bindings)
                    == set(terminal_contract["producer_output_paths"])
                    and all(
                        item.get("task_cid")
                        == spec_task_cid_by_alias[producer_alias]
                        and item.get("completion_receipt_cid")
                        == terminal_producer_receipts.get(producer_alias)
                        and item.get("portal_completion_binding")
                        == terminal_producer_portal_bindings.get(producer_alias)
                        and item.get("artifact_bundle_id")
                        == artifact_bundle_by_alias.get(producer_alias)
                        for producer_alias, item in (
                            observed_producer_receipt_bindings.items()
                        )
                    )
                )
                database_gates["terminal_report_gate_satisfied"] = bool(
                    terminal_production_receipt_satisfied
                    and terminal_task_binding is not None
                    and isinstance(terminal_gate_evidence, Mapping)
                    and terminal_gate_evidence.get("terminal_report_contract_id")
                    == terminal_contract["contract_id"]
                    and terminal_gate_evidence.get("task_cid")
                    == terminal_task["task_cid"]
                    and terminal_gate_evidence.get("task_alias")
                    == terminal_task["task_alias"]
                    and terminal_gate_evidence.get("task_revision")
                    == terminal_task["revision"]
                    and terminal_gate_evidence.get("completion_receipt_cid")
                    == terminal_task_binding["completion_receipt_cid"]
                    and terminal_gate_evidence.get("completion_evidence_digest")
                    == terminal_task_binding["completion_evidence_digest"]
                    and terminal_gate_evidence.get("control_receipt_id")
                    == terminal_task_binding["control_receipt_id"]
                    and terminal_gate_evidence.get("portal_receipt_id")
                    == terminal_validation.get("portal_receipt_id")
                    and terminal_portal_completion_binding is not None
                    and terminal_gate_evidence.get("portal_completion_binding")
                    == terminal_portal_completion_binding
                    and terminal_producer_receipts_satisfied
                    and terminal_gate_evidence.get("producer_receipts")
                    == terminal_producer_receipts
                    and database_gates[
                        "terminal_report_producer_artifacts_satisfied"
                    ]
                    and database_gates[
                        "terminal_report_producer_receipt_bindings_satisfied"
                    ]
                    and terminal_validation_lineage is not None
                    and terminal_gate_evidence.get("validation_run_id")
                    == terminal_validation_lineage["validation_run_id"]
                    and terminal_gate_evidence.get("validation_result_id")
                    == terminal_validation_lineage["validation_result_id"]
                    and terminal_gate_evidence.get("validation_evidence_id")
                    == terminal_validation_lineage["validation_evidence_id"]
                    and isinstance(terminal_artifacts, list)
                    and [item.get("path") for item in terminal_artifacts]
                    == terminal_contract["required_report_paths"]
                )
                failed_database_gates = sorted(
                    name for name, passed in database_gates.items() if passed is not True
                )
                reasons.extend(f"completion_gate:{name}" for name in failed_database_gates)
                if root_gate_for_current_receipt is None:
                    reasons.append("root_completion_gate_missing")
                if current_gate is not None:
                    if context:
                        if (
                            context.get("current_tree_clean") is not True
                            or str(context.get("source_head") or "")
                            != str(current_gate.get("source_head") or "")
                            or str(context.get("repository_tree_id") or "")
                            != str(current_gate.get("repository_tree_id") or "")
                        ):
                            reasons.append("root_completion_gate_not_current")
                    elif candidate_gate is None:
                        reasons.append("root_completion_gate_currentness_unverified")

            receipt_backfill = bool(
                isinstance(stored_receipt, Mapping)
                and stored_receipt.get("completion_kind")
                == "preseeded_completion_receipt_backfill"
            )
            expected_current_receipt: dict[str, Any] | None = None
            if completed and isinstance(stored_receipt, Mapping):
                expected_current_receipt = _goal_completion_receipt(
                    authority_spec_id=spec["authority_spec_id"],
                    goal_cid=goal_cid,
                    goal_alias=goal["goal_alias"],
                    goal_revision=int(goal["revision"]),
                    task_receipts=task_receipt_values,
                    child_goal_receipts=child_receipts,
                    dependency_goal_receipts=dependency_receipts,
                    receipt_backfill=receipt_backfill,
                    root_completion_gate=root_gate_for_current_receipt,
                )
                if dict(stored_receipt) != expected_current_receipt:
                    reasons.append("goal_current_completion_receipt_invalid")
            elif completed:
                reasons.append("goal_current_completion_receipt_missing")

            stored_receipt_integrity = bool(
                completed
                and isinstance(stored_receipt, Mapping)
                and _goal_receipt_has_valid_identity(
                    stored_receipt,
                    authority_spec_id=spec["authority_spec_id"],
                    goal_cid=goal_cid,
                    goal_alias=goal["goal_alias"],
                    goal_revision=int(goal["revision"]),
                    expected_root_completion_gate=(
                        stored_root_gate if is_root else None
                    ),
                )
                and (
                    goal_cid,
                    int(goal["revision"]),
                    _canonical(stored_receipt, noun="stored goal receipt"),
                )
                in emitted_goal_receipts
            )

            receipt_valid = bool(completed and not reasons and expected_current_receipt)
            if receipt_valid and expected_current_receipt is not None:
                goal_bindings[goal_cid] = {
                    "goal_cid": goal_cid,
                    "goal_alias": goal["goal_alias"],
                    "goal_revision": int(goal["revision"]),
                    "completion_receipt_id": expected_current_receipt["receipt_id"],
                }
            else:
                goal_bindings[goal_cid] = None

            absent_backfill = bool(
                completed
                and not isinstance(stored_receipt, Mapping)
                and goal_cid in backfill_allowlist
            )
            prerequisite_reasons = [
                reason
                for reason in reasons
                if reason
                not in {
                    "goal_current_completion_receipt_missing",
                    "goal_current_completion_receipt_invalid",
                }
            ]
            nonroot_receipt_refresh = bool(
                completed
                and not is_root
                and stored_receipt_integrity
                and expected_current_receipt is not None
                and dict(stored_receipt) != expected_current_receipt
                and not prerequisite_reasons
            )
            root_gate_refresh = bool(
                is_root
                and receipt_valid
                and effective_candidate_gate is not None
                and stored_root_gate is not None
                and effective_candidate_gate.get("gate_id")
                != stored_root_gate.get("gate_id")
            )
            root_input_refresh = bool(
                is_root
                and completed
                and stored_receipt_integrity
                and effective_candidate_gate is not None
                and stored_root_gate is not None
                and effective_candidate_gate.get("gate_id")
                == stored_root_gate.get("gate_id")
                and expected_current_receipt is not None
                and dict(stored_receipt) != expected_current_receipt
                and not prerequisite_reasons
            )
            ready = bool(
                root_gate_refresh
                or root_input_refresh
                or nonroot_receipt_refresh
                or (
                    not prerequisite_reasons
                    and (
                        status in _GOAL_OPEN_STATUSES
                        or absent_backfill
                    )
                )
            )
            if ready:
                ready_goal_cids.append(goal_cid)
                completion_inputs[goal_cid] = {
                    "task_receipts": task_receipt_values,
                    "child_goal_receipts": child_receipts,
                    "dependency_goal_receipts": dependency_receipts,
                    "receipt_backfill": (
                        False
                        if root_gate_refresh
                        or root_input_refresh
                        or nonroot_receipt_refresh
                        else absent_backfill
                    ),
                    "root_completion_gate": (
                        effective_candidate_gate
                        if root_gate_refresh or root_input_refresh
                        else root_gate_for_current_receipt
                    ),
                }
            if not receipt_valid:
                incomplete_goal_cids.append(goal_cid)
            if completed and not receipt_valid:
                invalid_goal_cids.append(goal_cid)
            goal_reasons[goal_cid] = sorted(set(reasons))
            goal_projection.append(
                {
                    "goal_cid": goal_cid,
                    "goal_alias": goal["goal_alias"],
                    "parent_goal_cid": goal["parent_goal_cid"],
                    "ordinal": int(goal["ordinal"]),
                    "status": goal["status"],
                    "revision": int(goal["revision"]),
                    "completion_receipt_id": (
                        str(expected_current_receipt.get("receipt_id") or "")
                        if receipt_valid and expected_current_receipt is not None
                        else ""
                    ),
                    "receipt_valid": receipt_valid,
                    "ready_for_completion": ready,
                    "incomplete_reasons": sorted(set(reasons)),
                }
            )

        goal_status_counts: dict[str, int] = {}
        for goal in goal_projection:
            status = str(goal["status"])
            goal_status_counts[status] = goal_status_counts.get(status, 0) + 1
        all_goal_receipts_valid = all(
            bool(goal["receipt_valid"]) for goal in goal_projection
        )
        completion_gates = {
            **database_gates,
            "all_exact_goal_receipts_valid": all_goal_receipts_valid,
            "root_completion_gate_current": bool(
                goal_bindings.get(spec["root_goal_cid"])
            ),
        }
        all_goals_satisfied = bool(
            all_goal_receipts_valid
            and all(value is True for value in completion_gates.values())
        )
        watermark_row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        projection = {
            "schema": GOAL_AUTHORITY_PROJECTION_SCHEMA,
            "authority": "duckdb_via_quack_state_owner",
            "authority_spec_id": spec["authority_spec_id"],
            "board_namespace": spec["board_namespace"],
            "event_watermark": int(watermark_row[0] if watermark_row else 0),
            "goal_count": len(goal_projection),
            "goal_status_counts": dict(sorted(goal_status_counts.items())),
            "root_goal": next(
                item for item in goal_projection if item["goal_cid"] == spec["root_goal_cid"]
            ),
            "goals": goal_projection,
            "goal_edges": [dict(item) for item in spec["goal_edges"]],
            "task_dependencies": [
                dict(item) for item in spec["task_dependencies"]
            ],
            "incomplete_goal_ids": [goals_by_cid[item]["goal_alias"] for item in incomplete_goal_cids],
            "invalid_goal_ids": [goals_by_cid[item]["goal_alias"] for item in invalid_goal_cids],
            "ready_goal_ids": [goals_by_cid[item]["goal_alias"] for item in ready_goal_cids],
            "task_receipt_invalid_ids": [
                task_alias_by_cid[task_cid]
                for task_cid, binding in task_bindings.items()
                if binding is None
            ],
            "task_dependency_failures": dependency_failures,
            "terminal_report_authority": {
                "task_cid": terminal_task["task_cid"],
                "task_alias": terminal_task["task_alias"],
                "status": terminal_task["status"],
                "revision": int(terminal_task["revision"]),
                "completion_receipt_cid": (
                    str(terminal_task_binding["completion_receipt_cid"])
                    if terminal_task_binding is not None
                    else ""
                ),
                "terminal_report_contract_id": terminal_contract["contract_id"],
                "declared_output_paths": list(
                    terminal_contract["declared_output_paths"]
                ),
                "required_report_paths": list(
                    terminal_contract["required_report_paths"]
                ),
                "validation_commands": [
                    list(item) for item in terminal_contract["validation_commands"]
                ],
                "production_completion_receipt_satisfied": (
                    terminal_production_receipt_satisfied
                ),
                "control_receipt_id": (
                    str(terminal_task_binding["control_receipt_id"])
                    if terminal_task_binding is not None
                    else ""
                ),
                "portal_receipt_id": (
                    str(terminal_validation.get("portal_receipt_id") or "")
                    if isinstance(terminal_validation, Mapping)
                    else ""
                ),
                "portal_completion_binding": (
                    dict(terminal_portal_completion_binding)
                    if terminal_portal_completion_binding is not None
                    else {}
                ),
                "producer_receipts": dict(terminal_producer_receipts),
                "producer_artifacts": (
                    dict(terminal_report_gate_evidence.get("producer_artifacts") or {})
                    if isinstance(terminal_report_gate_evidence, Mapping)
                    else {}
                ),
                "validation_lineage": (
                    dict(terminal_validation_lineage)
                    if terminal_validation_lineage is not None
                    else {}
                ),
                "report_artifacts": (
                    [
                        dict(item)
                        for item in terminal_report_gate_evidence.get(
                            "report_artifacts", []
                        )
                    ]
                    if isinstance(terminal_report_gate_evidence, Mapping)
                    else []
                ),
                "satisfied": database_gates["terminal_report_gate_satisfied"],
            },
            "settlement_counts": settlement_counts,
            "completion_policy": dict(spec["completion_policy"]),
            "completion_gates": completion_gates,
            "all_goals_satisfied": all_goals_satisfied,
            "ducklake_authoritative": False,
        }
        internal = {
            "spec": spec,
            "goals_by_cid": goals_by_cid,
            "completion_inputs": completion_inputs,
            "ready_goal_cids": ready_goal_cids,
        }
        return projection, internal

    def goal_authority_projection(
        self,
        specification: Mapping[str, Any],
        *,
        root_gate_context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Project exact, read-only goal authority through the current transport."""

        with self._connection(write=False) as connection:
            projection = _stable_goal_authority_projection_on(
                connection,
                specification,
                root_gate_context=root_gate_context,
            )
        return _content_addressed_projection(
            projection,
            maximum_bytes=MAX_GOAL_AUTHORITY_PROJECTION_BYTES,
            noun="goal authority projection",
        )

    def reconcile_goal_completion_authority(
        self,
        specification: Mapping[str, Any],
        *,
        root_completion_gate: Mapping[str, Any] | None = None,
        root_gate_current_validator: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Mapping[str, Any]:
        """Atomically close every newly satisfied goal in topological order.

        This mutation is intentionally owner-only.  Callers attached through
        Quack may read :meth:`goal_authority_projection`, but they cannot
        supply a new goal population or completion receipt for admission.
        """

        if not self.uses_bound_connection:
            raise IntentRepositoryError(
                "goal completion reconciliation requires the exclusive owner's bound connection"
            )
        changed_goal_ids: list[str] = []
        with self._connection(write=True) as connection:
            spec = _goal_completion_authority_spec(specification)
            maximum = int(spec["goal_count"])

            def admitted_root_gate() -> Mapping[str, Any] | None:
                if not isinstance(root_completion_gate, Mapping):
                    return None
                if root_gate_current_validator is None:
                    return root_completion_gate
                try:
                    return (
                        root_completion_gate
                        if root_gate_current_validator(root_completion_gate) is True
                        else None
                    )
                except Exception:
                    return None

            for _index in range(maximum + 1):
                current_root_gate = admitted_root_gate()
                projection, internal = self._goal_authority_state_on(
                    connection,
                    specification,
                    candidate_root_completion_gate=current_root_gate,
                    root_gate_context=(
                        {
                            "current_tree_clean": True,
                            "source_head": current_root_gate.get("source_head"),
                            "repository_tree_id": current_root_gate.get(
                                "repository_tree_id"
                            ),
                            "runtime_settlement_binding": current_root_gate.get(
                                "runtime_settlement_binding"
                            ),
                        }
                        if isinstance(current_root_gate, Mapping)
                        else None
                    ),
                )
                ready = list(internal["ready_goal_cids"])
                if not ready:
                    break
                goal_cid = ready[0]
                goal = internal["goals_by_cid"][goal_cid]
                inputs = internal["completion_inputs"][goal_cid]
                if (
                    goal_cid == spec["root_goal_cid"]
                    and root_gate_current_validator is not None
                    and admitted_root_gate() is None
                ):
                    raise IntentRepositoryConflictError(
                        "root completion gate changed before its owner-side CAS"
                    )
                current_revision = int(goal["revision"])
                target_revision = current_revision + 1
                receipt = _goal_completion_receipt(
                    authority_spec_id=spec["authority_spec_id"],
                    goal_cid=goal_cid,
                    goal_alias=goal["goal_alias"],
                    goal_revision=target_revision,
                    task_receipts=inputs["task_receipts"],
                    child_goal_receipts=inputs["child_goal_receipts"],
                    dependency_goal_receipts=inputs["dependency_goal_receipts"],
                    receipt_backfill=bool(inputs["receipt_backfill"]),
                    root_completion_gate=inputs["root_completion_gate"],
                )
                body = dict(goal["body"])
                body["completion_receipt"] = receipt
                now = _utc_iso()
                connection.execute(
                    """
                    UPDATE goals SET status = 'completed', updated_at = ?,
                        revision = ?, body_json = ?
                    WHERE goal_cid = ? AND revision = ?
                    """,
                    [
                        now,
                        target_revision,
                        _canonical(body, noun="goal body"),
                        goal_cid,
                        current_revision,
                    ],
                )
                observed = connection.execute(
                    "SELECT status, revision, body_json FROM goals WHERE goal_cid = ?",
                    [goal_cid],
                ).fetchone()
                if (
                    observed is None
                    or str(observed[0]) != "completed"
                    or int(observed[1]) != target_revision
                    or _decode_json(observed[2], noun="goal body") != body
                ):
                    raise IntentRepositoryConflictError(
                        "goal completion CAS lost its exact revision"
                    )
                self._append_event(
                    connection,
                    event_type=IntentEventType.GOAL_UPSERTED,
                    subject_id=goal_cid,
                    body={
                        "goal_cid": goal_cid,
                        "goal_alias": goal["goal_alias"],
                        "objective_id": goal["objective_id"],
                        "parent_goal_cid": goal["parent_goal_cid"],
                        "ordinal": int(goal["ordinal"]),
                        "title": goal["title"],
                        "previous_status": goal["status"],
                        "status": "completed",
                        "revision": target_revision,
                        "receipt": receipt,
                        "recorded_at": now,
                        "body": body,
                    },
                )
                changed_goal_ids.append(goal["goal_alias"])
            else:
                raise IntentRepositoryIntegrityError(
                    "goal completion reconciliation did not converge within the exact population"
                )
            final_root_gate = admitted_root_gate()
            final_projection, _internal = self._goal_authority_state_on(
                connection,
                specification,
                candidate_root_completion_gate=final_root_gate,
                root_gate_context=(
                    {
                        "current_tree_clean": True,
                        "source_head": final_root_gate.get("source_head"),
                        "repository_tree_id": final_root_gate.get(
                            "repository_tree_id"
                        ),
                        "runtime_settlement_binding": final_root_gate.get(
                            "runtime_settlement_binding"
                        ),
                    }
                    if isinstance(final_root_gate, Mapping)
                    else None
                ),
            )
            if (
                spec["root_goal_alias"] in changed_goal_ids
                and root_gate_current_validator is not None
                and admitted_root_gate() is None
            ):
                raise IntentRepositoryConflictError(
                    "root completion gate changed before transaction commit"
                )
        projected = _content_addressed_projection(
            final_projection,
            maximum_bytes=MAX_GOAL_AUTHORITY_PROJECTION_BYTES,
            noun="goal authority projection",
        )
        return MappingProxyType(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/goal-completion-reconciliation@1",
                "changed": bool(changed_goal_ids),
                "changed_goal_ids": changed_goal_ids,
                "goal_authority": dict(projected),
            }
        )

    def link_goal_edge(
        self,
        *,
        parent_goal_cid: str,
        child_goal_cid: str,
        edge_kind: str = "depends_on",
    ) -> IntentReceipt:
        parent = _identifier(parent_goal_cid, noun="parent_goal_cid")
        child = _identifier(child_goal_cid, noun="child_goal_cid")
        kind = _identifier(edge_kind, noun="edge_kind")
        if parent == child:
            raise IntentRepositoryError("goal edge cannot be reflexive")
        with self._connection(write=True) as connection:
            for gcid in (parent, child):
                row = connection.execute(
                    "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
                ).fetchone()
                if row is None:
                    raise IntentRepositoryIntegrityError(f"goal {gcid!r} does not exist for edge")
            connection.execute(
                """
                DELETE FROM goal_edges
                WHERE parent_goal_cid = ? AND child_goal_cid = ? AND edge_kind = ?
                """,
                [parent, child, kind],
            )
            connection.execute(
                """
                INSERT INTO goal_edges (
                    parent_goal_cid, child_goal_cid, edge_kind
                ) VALUES (?, ?, ?)
                """,
                [parent, child, kind],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_EDGE_LINKED,
                subject_id=parent,
                body={
                    "parent_goal_cid": parent,
                    "child_goal_cid": child,
                    "edge_kind": kind,
                    "revision": 0,
                },
            )

    def list_goal_edges(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return a bounded, stable projection of the admitted goal graph."""

        selected = _bounded_limit(limit)
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT parent_goal_cid, child_goal_cid, edge_kind
                FROM goal_edges
                ORDER BY parent_goal_cid, child_goal_cid, edge_kind
                LIMIT ?
                """,
                [selected],
            ).fetchall()
        return tuple(
            MappingProxyType(
                {
                    "parent_goal_cid": str(row[0]),
                    "child_goal_cid": str(row[1]),
                    "edge_kind": str(row[2]),
                }
            )
            for row in rows
        )

    def reopen_goal(
        self,
        *,
        goal_cid: str,
        expected_revision: int,
        reason: str = "reopened",
    ) -> IntentReceipt:
        gcid = _identifier(goal_cid, noun="goal_cid")
        expected = _nonneg_int(expected_revision, noun="expected_revision")
        reason_text = str(reason or "reopened").strip() or "reopened"
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, status, body_json FROM goals WHERE goal_cid = ?",
                [gcid],
            ).fetchone()
            if row is None:
                raise KeyError(gcid)
            current_revision = int(row[0])
            if current_revision != expected:
                raise IntentRepositoryConflictError("goal revision CAS is stale")
            previous_status = str(row[1])
            body_map = _decode_json(row[2], noun="goal body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            body_map["reopen_reason"] = reason_text
            body_map["previous_status"] = previous_status
            revision = current_revision + 1
            connection.execute(
                """
                UPDATE goals SET status = ?, updated_at = ?, revision = ?,
                    body_json = ?
                WHERE goal_cid = ? AND revision = ?
                """,
                [
                    "reopened",
                    now,
                    revision,
                    _canonical(body_map, noun="goal body"),
                    gcid,
                    current_revision,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_REOPENED,
                subject_id=gcid,
                body={
                    "goal_cid": gcid,
                    "previous_status": previous_status,
                    "status": "reopened",
                    "reason": reason_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    # -- plans (also exposed via PlanRevisionRepository) ---------------------

    def upsert_plan(
        self,
        *,
        plan_cid: str,
        goal_cid: str,
        plan_alias: str,
        status: str = "active",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        set_head: bool = True,
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        gcid = _identifier(goal_cid, noun="goal_cid")
        alias = _identifier(plan_alias, noun="plan_alias")
        status_text = str(status or "active").strip().lower()
        body_map = _mapping(body, noun="plan body")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            goal_row = connection.execute(
                "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if goal_row is None:
                raise IntentRepositoryIntegrityError(f"goal {gcid!r} does not exist for plan")
            existing = connection.execute(
                "SELECT revision FROM plans WHERE plan_cid = ?", [pcid]
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "plan CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO plans (
                        plan_cid, goal_cid, plan_alias, status, created_at,
                        updated_at, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        pcid,
                        gcid,
                        alias,
                        status_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="plan body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if expected_revision is not None and expected_revision != current_revision:
                    raise IntentRepositoryConflictError("plan revision CAS is stale")
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE plans SET goal_cid = ?, plan_alias = ?, status = ?,
                        updated_at = ?, revision = ?, body_json = ?
                    WHERE plan_cid = ? AND revision = ?
                    """,
                    [
                        gcid,
                        alias,
                        status_text,
                        now,
                        revision,
                        _canonical(body_map, noun="plan body"),
                        pcid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    pcid,
                    revision,
                    _canonical(body_map, noun="plan revision body"),
                    now,
                ],
            )
            if set_head and status_text == "active":
                # Demote other active heads for the same goal.
                connection.execute(
                    """
                    UPDATE plans SET status = 'superseded', updated_at = ?
                    WHERE goal_cid = ? AND plan_cid <> ? AND status = 'active'
                    """,
                    [now, gcid, pcid],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_UPSERTED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "goal_cid": gcid,
                    "plan_alias": alias,
                    "status": status_text,
                    "revision": revision,
                    "body": body_map,
                    "set_head": bool(set_head),
                    "recorded_at": now,
                },
            )

    def get_plan(self, plan_cid: str) -> Mapping[str, Any] | None:
        pcid = _identifier(plan_cid, noun="plan_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT plan_cid, goal_cid, plan_alias, status, created_at,
                       updated_at, revision, body_json
                FROM plans WHERE plan_cid = ? OR plan_alias = ?
                LIMIT 1
                """,
                [pcid, pcid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "created_at": str(row[4]),
                "updated_at": str(row[5]),
                "revision": int(row[6]),
                "body": _decode_json(row[7], noun="plan body"),
            }
        )

    def get_plan_head(self, goal_cid: str) -> PlanHead | None:
        gcid = _identifier(goal_cid, noun="goal_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT plan_cid, goal_cid, revision, status, body_json
                FROM plans
                WHERE goal_cid = ? AND status = 'active'
                ORDER BY revision DESC, plan_cid ASC
                LIMIT 1
                """,
                [gcid],
            ).fetchone()
        if row is None:
            return None
        body = _decode_json(row[4], noun="plan body")
        body_map = body if isinstance(body, dict) else {}
        return PlanHead(
            plan_cid=str(row[0]),
            goal_cid=str(row[1]),
            revision=int(row[2]),
            status=str(row[3]),
            superseded_by=str(body_map.get("superseded_by") or ""),
            continuation_of=str(body_map.get("continuation_of") or ""),
        )

    def append_plan_revision(
        self,
        *,
        plan_cid: str,
        body: Mapping[str, Any] | None = None,
        expected_revision: int,
        delta: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        body_map = _mapping(body, noun="plan body")
        delta_map = _mapping(delta, noun="plan delta") if delta is not None else {}
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, body_json, goal_cid, plan_alias, status "
                "FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            current_revision = int(row[0])
            if current_revision != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            previous_body = _decode_json(row[1], noun="plan body")
            if not isinstance(previous_body, dict):
                previous_body = {}
            merged = dict(previous_body)
            merged.update(body_map)
            if delta_map:
                merged["last_delta"] = delta_map
            revision = current_revision + 1
            connection.execute(
                """
                UPDATE plans SET revision = ?, updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    revision,
                    now,
                    _canonical(merged, noun="plan body"),
                    pcid,
                    current_revision,
                ],
            )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [pcid, revision, _canonical(merged, noun="plan revision"), now],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_REVISION_APPENDED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "goal_cid": str(row[2]),
                    "plan_alias": str(row[3]),
                    "status": str(row[4]),
                    "revision": revision,
                    "body": merged,
                    "delta": delta_map,
                    "recorded_at": now,
                },
            )

    def supersede_plan(
        self,
        *,
        plan_cid: str,
        successor_plan_cid: str,
        expected_revision: int,
        reason: str = "superseded",
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        successor = _identifier(successor_plan_cid, noun="successor_plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        reason_text = str(reason or "superseded").strip() or "superseded"
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, body_json, goal_cid FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            if int(row[0]) != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            succ = connection.execute(
                "SELECT 1 FROM plans WHERE plan_cid = ?", [successor]
            ).fetchone()
            if succ is None:
                raise IntentRepositoryIntegrityError(f"successor plan {successor!r} does not exist")
            body_map = _decode_json(row[1], noun="plan body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            body_map["superseded_by"] = successor
            body_map["supersede_reason"] = reason_text
            revision = expected + 1
            connection.execute(
                """
                UPDATE plans SET status = 'superseded', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    revision,
                    now,
                    _canonical(body_map, noun="plan body"),
                    pcid,
                    expected,
                ],
            )
            connection.execute(
                """
                UPDATE plans SET status = 'active', updated_at = ?
                WHERE plan_cid = ?
                """,
                [now, successor],
            )
            connection.execute(
                """
                INSERT INTO planning_decisions (
                    decision_id, plan_cid, goal_cid, decision_kind,
                    decided_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    content_identity(
                        {
                            "kind": "supersession",
                            "plan_cid": pcid,
                            "successor": successor,
                            "revision": revision,
                        }
                    ),
                    pcid,
                    str(row[2]),
                    "supersession",
                    now,
                    _canonical(
                        {
                            "predecessor": pcid,
                            "successor": successor,
                            "reason": reason_text,
                        },
                        noun="supersession decision",
                    ),
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_SUPERSEDED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "successor_plan_cid": successor,
                    "goal_cid": str(row[2]),
                    "reason": reason_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def continue_plan(
        self,
        *,
        plan_cid: str,
        continuation_plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        """Create/activate a continuation plan bound to the predecessor head."""

        pcid = _identifier(plan_cid, noun="plan_cid")
        cont = _identifier(continuation_plan_cid, noun="continuation_plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        body_map = _mapping(body, noun="continuation body")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, goal_cid, plan_alias, body_json FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            if int(row[0]) != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            gcid = str(row[1])
            cont_body = dict(body_map)
            cont_body["continuation_of"] = pcid
            cont_exists = connection.execute(
                "SELECT revision FROM plans WHERE plan_cid = ?", [cont]
            ).fetchone()
            if cont_exists is None:
                connection.execute(
                    """
                    INSERT INTO plans (
                        plan_cid, goal_cid, plan_alias, status, created_at,
                        updated_at, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cont,
                        gcid,
                        f"{row[2]}-cont",
                        "active",
                        now,
                        now,
                        1,
                        _canonical(cont_body, noun="continuation plan body"),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO plan_revisions (
                        plan_cid, revision, body_json, recorded_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        cont,
                        1,
                        _canonical(cont_body, noun="continuation revision"),
                        now,
                    ],
                )
                cont_revision = 1
            else:
                cont_revision = int(cont_exists[0]) + 1
                connection.execute(
                    """
                    UPDATE plans SET status = 'active', revision = ?,
                        updated_at = ?, body_json = ?
                    WHERE plan_cid = ?
                    """,
                    [
                        cont_revision,
                        now,
                        _canonical(cont_body, noun="continuation plan body"),
                        cont,
                    ],
                )
            pred_body = _decode_json(row[3], noun="plan body")
            if not isinstance(pred_body, dict):
                pred_body = {}
            pred_body = dict(pred_body)
            pred_body["continued_by"] = cont
            connection.execute(
                """
                UPDATE plans SET status = 'continued', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    expected + 1,
                    now,
                    _canonical(pred_body, noun="plan body"),
                    pcid,
                    expected,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_CONTINUED,
                subject_id=cont,
                body={
                    "plan_cid": cont,
                    "continuation_of": pcid,
                    "goal_cid": gcid,
                    "revision": cont_revision,
                    "body": cont_body,
                    "recorded_at": now,
                },
            )

    # -- tasks ---------------------------------------------------------------

    def upsert_task(
        self,
        *,
        task_cid: str,
        task_alias: str,
        goal_cid: str,
        ordinal: int = 0,
        status: str = "ready",
        priority: str = "P2",
        plan_cid: str = "",
        objective_id: str = "",
        body: Mapping[str, Any] | None = None,
        identity: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        dependencies: Sequence[str] | None = None,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        acceptance: Sequence[Mapping[str, Any] | str] | None = None,
        validations: Sequence[Mapping[str, Any] | str | Sequence[str]] | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        alias = _identifier(task_alias, noun="task_alias")
        gcid = _identifier(goal_cid, noun="goal_cid")
        ord_value = _nonneg_int(ordinal, noun="ordinal")
        status_text = _status(status, allowed=_TASK_STATUSES, noun="task")
        priority_text = str(priority or "P2").strip() or "P2"
        pcid = _optional_identifier(plan_cid, noun="plan_cid")
        oid = _optional_identifier(objective_id, noun="objective_id")
        body_map = dict(_mapping(body, noun="task body"))
        identity_map = _mapping(identity, noun="task identity")
        # Identity material is canonical and must not include mutable aliases
        # as keys; always bind the durable task_cid.
        identity_map = {
            **identity_map,
            "task_cid": tcid,
            "task_alias": alias,
        }
        now = _utc_iso()

        with self._connection(write=True) as connection:
            goal_row = connection.execute(
                "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if goal_row is None:
                raise IntentRepositoryIntegrityError(f"goal {gcid!r} does not exist for task")
            existing = connection.execute(
                "SELECT revision, status, task_alias, body_json "
                "FROM tasks WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            supplied_receipt = body_map.get("completion_receipt")
            supplied_transfer = (
                supplied_receipt.get("virgin_task_transfer")
                if isinstance(supplied_receipt, Mapping)
                else None
            )
            if existing is None:
                if supplied_transfer is not None:
                    raise IntentRepositoryTransitionError(
                        "virgin_task_transfer is owner-reserved"
                    )
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "task CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_cid, objective_id,
                        ordinal, status, revision, priority, created_at,
                        updated_at, identity_json, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        alias,
                        gcid,
                        pcid,
                        oid,
                        ord_value,
                        status_text,
                        revision,
                        priority_text,
                        now,
                        now,
                        _canonical(identity_map, noun="task identity"),
                        _canonical(body_map, noun="task body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if expected_revision is not None and expected_revision != current_revision:
                    raise IntentRepositoryConflictError("task revision CAS is stale")
                stored_alias = str(existing[2] or "")
                stored_body = _decode_json(existing[3], noun="stored task body")
                stored_receipt = (
                    stored_body.get("completion_receipt")
                    if isinstance(stored_body, Mapping)
                    else None
                )
                stored_transfer = (
                    stored_receipt.get("virgin_task_transfer")
                    if isinstance(stored_receipt, Mapping)
                    else None
                )
                if stored_transfer is None:
                    if supplied_transfer is not None:
                        raise IntentRepositoryTransitionError(
                            "virgin_task_transfer is owner-reserved"
                        )
                else:
                    shard_count = (
                        stored_transfer.get("task_shard_count")
                        if isinstance(stored_transfer, Mapping)
                        else None
                    )
                    if (
                        isinstance(shard_count, bool)
                        or not isinstance(shard_count, int)
                        or shard_count <= 1
                        or alias != stored_alias
                    ):
                        raise IntentRepositoryTransitionError(
                            "upsert would invalidate a virgin-transfer binding"
                        )
                    _database_virgin_transfer_binding(
                        task_cid=tcid,
                        task_alias=stored_alias,
                        receipt=stored_receipt,
                        shard_count=shard_count,
                    )
                    raise IntentRepositoryTransitionError(
                        "upsert cannot rewrite a virgin-transfer-assigned task"
                    )
                revision = current_revision + 1
                # Canonical task_cid is immutable; alias/goal/plan may update.
                connection.execute(
                    """
                    UPDATE tasks SET
                        task_alias = ?, goal_cid = ?, plan_cid = ?,
                        objective_id = ?, ordinal = ?, status = ?,
                        revision = ?, priority = ?, updated_at = ?,
                        identity_json = ?, body_json = ?
                    WHERE task_cid = ? AND revision = ?
                    """,
                    [
                        alias,
                        gcid,
                        pcid,
                        oid,
                        ord_value,
                        status_text,
                        revision,
                        priority_text,
                        now,
                        _canonical(identity_map, noun="task identity"),
                        _canonical(body_map, noun="task body"),
                        tcid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="task revision body"),
                    now,
                ],
            )
            resolved_dependencies: list[str] = []
            if dependencies is not None:
                self._set_dependencies_on(connection, tcid, dependencies)
                resolved_dependencies = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT dependency_task_cid FROM task_dependencies "
                        "WHERE task_cid = ? ORDER BY dependency_task_cid",
                        [tcid],
                    ).fetchall()
                ]
            if outputs is not None:
                self._set_outputs_on(connection, tcid, outputs)
            if acceptance is not None:
                self._set_acceptance_on(connection, tcid, acceptance)
            if validations is not None:
                self._set_validations_on(connection, tcid, validations)
            event_body: dict[str, Any] = {
                "task_cid": tcid,
                "task_alias": alias,
                "goal_cid": gcid,
                "plan_cid": pcid,
                "objective_id": oid,
                "ordinal": ord_value,
                "status": status_text,
                "priority": priority_text,
                "revision": revision,
                "identity": identity_map,
                "body": body_map,
                "recorded_at": now,
            }
            # Only emit relation fields that were explicitly provided so rebuild
            # does not wipe prior edges when an upsert omits them.
            if dependencies is not None:
                event_body["dependencies"] = resolved_dependencies
            if outputs is not None:
                event_body["outputs"] = [
                    dict(item) if isinstance(item, Mapping) else item for item in outputs
                ]
            if acceptance is not None:
                event_body["acceptance"] = [
                    dict(item) if isinstance(item, Mapping) else item for item in acceptance
                ]
            if validations is not None:
                event_body["validations"] = [
                    list(item)
                    if isinstance(item, Sequence)
                    and not isinstance(item, (str, Mapping, bytes, bytearray))
                    else (dict(item) if isinstance(item, Mapping) else item)
                    for item in validations
                ]
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_UPSERTED,
                subject_id=tcid,
                task_cid=tcid,
                body=event_body,
            )

    def _set_dependencies_on(
        self, connection: Any, task_cid: str, dependencies: Sequence[str]
    ) -> None:
        if len(dependencies) > MAX_DEPENDENCIES:
            raise IntentRepositoryBoundsError("dependency count exceeds bound")
        connection.execute("DELETE FROM task_dependencies WHERE task_cid = ?", [task_cid])
        seen: set[str] = set()
        for raw in dependencies:
            dep = _identifier(raw, noun="dependency_task_cid")
            # Prefer durable CID when the dependency was referenced by alias.
            resolved = connection.execute(
                "SELECT task_cid FROM tasks "
                "WHERE task_cid = ? OR task_alias = ? "
                "ORDER BY task_cid LIMIT 1",
                [dep, dep],
            ).fetchone()
            if resolved is not None:
                dep = str(resolved[0])
            if dep == task_cid:
                raise IntentRepositoryError("task cannot depend on itself")
            if dep in seen:
                continue
            seen.add(dep)
            connection.execute(
                """
                INSERT INTO task_dependencies (
                    task_cid, dependency_task_cid, kind
                ) VALUES (?, ?, ?)
                """,
                [task_cid, dep, "depends_on"],
            )

    def _set_outputs_on(
        self, connection: Any, task_cid: str, outputs: Sequence[Mapping[str, Any]]
    ) -> None:
        if len(outputs) > MAX_OUTPUTS:
            raise IntentRepositoryBoundsError("output count exceeds bound")
        connection.execute("DELETE FROM task_outputs WHERE task_cid = ?", [task_cid])
        for ordinal, item in enumerate(outputs):
            mapping = _mapping(item, noun="task output")
            path = _output_path(
                mapping.get("path") or mapping.get("effect_id") or f"output:{ordinal}",
                noun="output path",
            )
            connection.execute(
                """
                INSERT INTO task_outputs (
                    task_cid, ordinal, path, effect_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    path,
                    _canonical(mapping, noun="output effect"),
                ],
            )

    def _set_acceptance_on(
        self,
        connection: Any,
        task_cid: str,
        acceptance: Sequence[Mapping[str, Any] | str],
    ) -> None:
        if len(acceptance) > MAX_ACCEPTANCE:
            raise IntentRepositoryBoundsError("acceptance count exceeds bound")
        connection.execute("DELETE FROM task_acceptance WHERE task_cid = ?", [task_cid])
        for ordinal, item in enumerate(acceptance):
            if isinstance(item, str):
                criterion = item.strip()
                policy: dict[str, Any] = {"criterion": criterion}
            else:
                mapping = _mapping(item, noun="acceptance")
                criterion = str(
                    mapping.get("criterion")
                    or mapping.get("statement")
                    or mapping.get("criterion_key")
                    or f"criterion:{ordinal}"
                ).strip()
                policy = dict(mapping)
            if not criterion:
                raise IntentRepositoryError("acceptance criterion must not be empty")
            connection.execute(
                """
                INSERT INTO task_acceptance (
                    task_cid, ordinal, criterion, evidence_policy_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    criterion,
                    _canonical(policy, noun="acceptance policy"),
                ],
            )

    def _set_validations_on(
        self,
        connection: Any,
        task_cid: str,
        validations: Sequence[Mapping[str, Any] | str | Sequence[str]],
    ) -> None:
        if len(validations) > MAX_VALIDATIONS:
            raise IntentRepositoryBoundsError("validation count exceeds bound")
        connection.execute("DELETE FROM task_validations WHERE task_cid = ?", [task_cid])
        for ordinal, item in enumerate(validations):
            if isinstance(item, str):
                argv = [item]
                policy: dict[str, Any] = {}
            elif isinstance(item, Mapping):
                mapping = _mapping(item, noun="validation")
                raw_argv = mapping.get("argv") or mapping.get("validation_commands")
                if isinstance(raw_argv, str):
                    argv = [raw_argv]
                elif isinstance(raw_argv, Sequence):
                    argv = [str(part) for part in raw_argv]
                else:
                    argv = [str(mapping.get("command") or f"validation:{ordinal}")]
                policy = {
                    key: value
                    for key, value in mapping.items()
                    if key not in {"argv", "validation_commands", "command"}
                }
            elif isinstance(item, Sequence):
                argv = [str(part) for part in item]
                policy = {}
            else:
                raise IntentRepositoryError("validation entry has unsupported type")
            connection.execute(
                """
                INSERT INTO task_validations (
                    task_cid, ordinal, argv_json, policy_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    _canonical(list(argv), noun="validation argv"),
                    _canonical(policy, noun="validation policy"),
                ],
            )

    def set_task_dependencies(self, task_cid: str, dependencies: Sequence[str]) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if row is None:
                raise KeyError(tcid)
            self._set_dependencies_on(connection, tcid, dependencies)
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_DEPENDENCIES_SET,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "dependencies": [
                        _identifier(item, noun="dependency_task_cid") for item in dependencies
                    ],
                    "revision": int(row[0]),
                },
            )

    def get_task(self, task_cid_or_alias: str) -> Mapping[str, Any] | None:
        key = _identifier(task_cid_or_alias, noun="task_cid")
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                       ordinal, status, revision, priority, created_at,
                       updated_at, identity_json, body_json
                FROM tasks
                WHERE task_cid = ? OR task_alias = ?
                ORDER BY task_cid
                LIMIT 2
                """,
                [key, key],
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise IntentRepositoryIntegrityError("task CID/alias lookup is ambiguous")
            row = rows[0]
            tcid = str(row[0])
            deps = [
                str(item[0])
                for item in connection.execute(
                    "SELECT dependency_task_cid FROM task_dependencies "
                    "WHERE task_cid = ? ORDER BY dependency_task_cid",
                    [tcid],
                ).fetchall()
            ]
            outputs = [
                {
                    "ordinal": int(item[0]),
                    "path": str(item[1]),
                    "effect": _decode_json(item[2], noun="output effect"),
                }
                for item in connection.execute(
                    "SELECT ordinal, path, effect_json FROM task_outputs "
                    "WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
            acceptance = [
                {
                    "ordinal": int(item[0]),
                    "criterion": str(item[1]),
                    "evidence_policy": _decode_json(item[2], noun="acceptance policy"),
                }
                for item in connection.execute(
                    "SELECT ordinal, criterion, evidence_policy_json "
                    "FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
            validations = [
                {
                    "ordinal": int(item[0]),
                    "argv": _decode_json(item[1], noun="validation argv"),
                    "policy": _decode_json(item[2], noun="validation policy"),
                }
                for item in connection.execute(
                    "SELECT ordinal, argv_json, policy_json "
                    "FROM task_validations WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
        return MappingProxyType(
            {
                "task_cid": tcid,
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3] or ""),
                "objective_id": str(row[4] or ""),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8] or ""),
                "created_at": str(row[9] or ""),
                "updated_at": str(row[10] or ""),
                "identity": _decode_json(row[11], noun="task identity"),
                "body": _decode_json(row[12], noun="task body"),
                "dependencies": tuple(deps),
                "outputs": tuple(outputs),
                "acceptance": tuple(acceptance),
                "validations": tuple(validations),
            }
        )

    def list_tasks(
        self,
        *,
        status: str | Iterable[str] | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[Mapping[str, Any], ...]:
        selected = _bounded_limit(limit)
        off = _nonneg_int(offset, noun="offset")
        statuses: tuple[str, ...]
        if status is None:
            statuses = ()
        elif isinstance(status, str):
            statuses = (_status(status, allowed=_TASK_STATUSES, noun="task"),)
        else:
            statuses = tuple(
                sorted({_status(item, allowed=_TASK_STATUSES, noun="task") for item in status})
            )
        with self._connection(write=False) as connection:
            if statuses:
                placeholders = ", ".join("?" for _ in statuses)
                rows = connection.execute(
                    f"""
                    SELECT task_cid FROM tasks
                    WHERE status IN ({placeholders})
                    ORDER BY ordinal, task_cid
                    LIMIT ? OFFSET ?
                    """,
                    [*statuses, selected, off],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT task_cid FROM tasks
                    ORDER BY ordinal, task_cid
                    LIMIT ? OFFSET ?
                    """,
                    [selected, off],
                ).fetchall()
        results: list[Mapping[str, Any]] = []
        for row in rows:
            task = self.get_task(str(row[0]))
            if task is not None:
                results.append(task)
        return tuple(results)

    # -- evidence / completion -----------------------------------------------

    def record_evidence(
        self,
        *,
        task_cid: str,
        evidence_kind: str,
        digest: str,
        body: Mapping[str, Any] | None = None,
        evidence_id: str = "",
        parent_evidence_id: str = "",
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        kind = _identifier(evidence_kind, noun="evidence_kind")
        digest_text = _identifier(digest, noun="digest")
        body_map = _mapping(body, noun="evidence body")
        eid = (
            _identifier(evidence_id, noun="evidence_id")
            if evidence_id
            else content_identity(
                {
                    "task_cid": tcid,
                    "evidence_kind": kind,
                    "digest": digest_text,
                    "body": body_map,
                }
            )
        )
        parent = _optional_identifier(parent_evidence_id, noun="parent_evidence_id")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            count = connection.execute(
                "SELECT COUNT(*) FROM evidence_nodes WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if count and int(count[0]) >= MAX_EVIDENCE:
                raise IntentRepositoryBoundsError("evidence population exceeds bound")
            connection.execute(
                "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                [eid],
            )
            connection.execute(
                """
                INSERT INTO evidence_nodes (
                    evidence_id, parent_evidence_id, task_cid, evidence_kind,
                    digest, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    eid,
                    parent,
                    tcid,
                    kind,
                    digest_text,
                    now,
                    _canonical(body_map, noun="evidence body"),
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.EVIDENCE_RECORDED,
                subject_id=eid,
                task_cid=tcid,
                body={
                    "evidence_id": eid,
                    "parent_evidence_id": parent,
                    "task_cid": tcid,
                    "evidence_kind": kind,
                    "digest": digest_text,
                    "body": body_map,
                    "created_at": now,
                    "revision": 0,
                },
            )

    def record_validation_result(
        self,
        *,
        task_cid: str,
        outcome: str,
        evidence_digest: str,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        outcome_text = str(outcome or "").strip().lower()
        if outcome_text not in {"passed", "failed", "error", "skipped"}:
            raise IntentRepositoryError(f"validation outcome {outcome!r} is not in the closed set")
        digest = _identifier(evidence_digest, noun="evidence_digest")
        body_map = _mapping(body, noun="validation body")
        argv_list = [str(item) for item in (argv or ())]
        now = _utc_iso()
        run_id = content_identity(
            {
                "task_cid": tcid,
                "attempt_id": attempt_id,
                "argv": argv_list,
                "recorded_at": now,
            }
        )
        result_id = content_identity(
            {
                "run_id": run_id,
                "outcome": outcome_text,
                "evidence_digest": digest,
            }
        )
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            connection.execute(
                """
                INSERT INTO validation_runs (
                    run_id, task_cid, attempt_id, started_at, finished_at,
                    status, command_digest, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    tcid,
                    attempt_id or "",
                    now,
                    now,
                    outcome_text,
                    content_identity({"argv": argv_list}),
                    _canonical(
                        {"argv": argv_list, **body_map},
                        noun="validation run body",
                    ),
                ],
            )
            connection.execute(
                """
                INSERT INTO validation_results (
                    result_id, run_id, task_cid, ordinal, outcome,
                    evidence_digest, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result_id,
                    run_id,
                    tcid,
                    0,
                    outcome_text,
                    digest,
                    _canonical(body_map, noun="validation result body"),
                ],
            )
            # Mirror a current evidence node so completion can join on digests.
            if outcome_text == "passed":
                evidence_id = content_identity(
                    {
                        "task_cid": tcid,
                        "evidence_kind": "validation",
                        "digest": digest,
                        "run_id": run_id,
                    }
                )
                connection.execute(
                    "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                    [evidence_id],
                )
                connection.execute(
                    """
                    INSERT INTO evidence_nodes (
                        evidence_id, parent_evidence_id, task_cid, evidence_kind,
                        digest, created_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evidence_id,
                        "",
                        tcid,
                        "validation",
                        digest,
                        now,
                        _canonical(
                            {
                                "run_id": run_id,
                                "result_id": result_id,
                                "argv": argv_list,
                                "outcome": outcome_text,
                            },
                            noun="validation evidence body",
                        ),
                    ],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.VALIDATION_RECORDED,
                subject_id=result_id,
                task_cid=tcid,
                attempt_id=attempt_id or "",
                body={
                    "result_id": result_id,
                    "run_id": run_id,
                    "task_cid": tcid,
                    "outcome": outcome_text,
                    "evidence_digest": digest,
                    "argv": argv_list,
                    "body": body_map,
                    "recorded_at": now,
                    "revision": 0,
                },
            )

    def current_evidence_for_task(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        tcid = _identifier(task_cid, noun="task_cid")
        clock = int(now_ms if now_ms is not None else self._clock_ms())
        freshness_ms = self.evidence_freshness_seconds * 1000
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, parent_evidence_id, task_cid, evidence_kind,
                       digest, created_at, body_json
                FROM evidence_nodes
                WHERE task_cid = ?
                ORDER BY created_at DESC, evidence_id ASC
                """,
                [tcid],
            ).fetchall()
        current: list[Mapping[str, Any]] = []
        for row in rows:
            created_at = str(row[5] or "")
            created_ms = _parse_iso_ms(created_at)
            if freshness_ms > 0 and created_ms > 0:
                if clock - created_ms > freshness_ms:
                    continue
            current.append(
                MappingProxyType(
                    {
                        "evidence_id": str(row[0]),
                        "parent_evidence_id": str(row[1] or ""),
                        "task_cid": str(row[2] or ""),
                        "evidence_kind": str(row[3]),
                        "digest": str(row[4]),
                        "created_at": created_at,
                        "body": _decode_json(row[6], noun="evidence body"),
                    }
                )
            )
        return tuple(current)

    def qualification_authority_for_task(
        self,
        task_cid: str,
    ) -> Mapping[str, Any]:
        """Return bounded canonical rows underlying task qualification.

        Evidence nodes are indexes, not self-authorizing proof.  This view
        lets callers bind them to the task identity, validation run/result,
        and completion-receipt authorities without issuing raw SQL.
        """

        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=False) as connection:
            task_row = connection.execute(
                """
                SELECT identity_json, extension_schema, extension_json
                FROM tasks WHERE task_cid = ?
                """,
                [tcid],
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            run_rows = connection.execute(
                """
                SELECT run_id, task_cid, attempt_id, started_at, finished_at,
                       status, command_digest, body_json
                FROM validation_runs WHERE task_cid = ?
                ORDER BY started_at, run_id
                LIMIT ?
                """,
                [tcid, MAX_VALIDATIONS + 1],
            ).fetchall()
            result_rows = connection.execute(
                """
                SELECT result_id, run_id, task_cid, ordinal, outcome,
                       evidence_digest, body_json
                FROM validation_results WHERE task_cid = ?
                ORDER BY run_id, ordinal, result_id
                LIMIT ?
                """,
                [tcid, MAX_VALIDATIONS + 1],
            ).fetchall()
            completion_rows = connection.execute(
                """
                SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                       fencing_token, completed_at, validation_run_id,
                       evidence_digest, body_json
                FROM completion_receipts WHERE task_cid = ?
                ORDER BY completed_at, receipt_cid
                LIMIT ?
                """,
                [tcid, MAX_EVIDENCE + 1],
            ).fetchall()
        if len(run_rows) > MAX_VALIDATIONS or len(result_rows) > MAX_VALIDATIONS:
            raise IntentRepositoryBoundsError(
                "task qualification validation population exceeds bound"
            )
        if len(completion_rows) > MAX_EVIDENCE:
            raise IntentRepositoryBoundsError(
                "task qualification completion population exceeds bound"
            )
        return MappingProxyType(
            {
                "task_cid": tcid,
                "identity": _decode_json(task_row[0], noun="task identity"),
                "extension_schema": str(task_row[1] or ""),
                "extension": _decode_json(task_row[2], noun="task extension"),
                "validation_runs": tuple(
                    MappingProxyType(
                        {
                            "run_id": str(row[0]),
                            "task_cid": str(row[1]),
                            "attempt_id": str(row[2] or ""),
                            "started_at": str(row[3]),
                            "finished_at": str(row[4] or ""),
                            "status": str(row[5]),
                            "command_digest": str(row[6]),
                            "body": _decode_json(row[7], noun="validation run body"),
                        }
                    )
                    for row in run_rows
                ),
                "validation_results": tuple(
                    MappingProxyType(
                        {
                            "result_id": str(row[0]),
                            "run_id": str(row[1]),
                            "task_cid": str(row[2]),
                            "ordinal": int(row[3]),
                            "outcome": str(row[4]),
                            "evidence_digest": str(row[5]),
                            "body": _decode_json(row[6], noun="validation result body"),
                        }
                    )
                    for row in result_rows
                ),
                "completion_receipts": tuple(
                    MappingProxyType(
                        {
                            "receipt_cid": str(row[0]),
                            "task_cid": str(row[1]),
                            "goal_cid": str(row[2]),
                            "attempt_id": str(row[3] or ""),
                            "claim_cid": str(row[4] or ""),
                            "fencing_token": int(row[5]),
                            "completed_at": str(row[6]),
                            "validation_run_id": str(row[7] or ""),
                            "evidence_digest": str(row[8]),
                            "body": _decode_json(row[9], noun="completion receipt body"),
                        }
                    )
                    for row in completion_rows
                ),
            }
        )

    def required_evidence_satisfied(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return whether every acceptance criterion has current evidence."""

        task = self.get_task(task_cid)
        if task is None:
            raise KeyError(task_cid)
        acceptance = list(task.get("acceptance") or ())
        if not acceptance:
            # No declared acceptance: completion still requires at least one
            # current validation evidence node (fail-closed for empty claims).
            current = self.current_evidence_for_task(task_cid, now_ms=now_ms)
            validation = [
                item
                for item in current
                if str(item.get("evidence_kind") or "") in {"validation", "test", "acceptance"}
            ]
            if validation:
                return True, ()
            return False, ("required:current_validation_evidence",)

        current = self.current_evidence_for_task(task_cid, now_ms=now_ms)
        digests = {str(item.get("digest") or "") for item in current if item.get("digest")}
        kinds = {str(item.get("evidence_kind") or "") for item in current}
        missing: list[str] = []
        for item in acceptance:
            if not isinstance(item, Mapping):
                continue
            policy = item.get("evidence_policy") or {}
            if not isinstance(policy, Mapping):
                policy = {}
            criterion = str(item.get("criterion") or "")
            required_digest = str(
                policy.get("required_digest")
                or policy.get("evidence_digest")
                or policy.get("digest")
                or ""
            ).strip()
            required_kind = str(policy.get("evidence_kind") or policy.get("kind") or "").strip()
            if required_digest:
                if required_digest not in digests:
                    missing.append(f"digest:{required_digest}")
                continue
            if required_kind:
                if required_kind not in kinds:
                    missing.append(f"kind:{required_kind}")
                continue
            # Default: any current evidence satisfies the criterion when no
            # explicit digest/kind is declared, but evidence must exist.
            if not current:
                missing.append(f"criterion:{criterion or item.get('ordinal')}")
        return (not missing), tuple(missing)

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_revision: int,
        new_status: str,
        receipt: Mapping[str, Any] | None = None,
        expected_control_receipt: Mapping[str, Any] | None = None,
        evidence_digests: Sequence[str] | None = None,
        allow_completion_without_evidence: bool = False,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        status_text = _status(new_status, allowed=_TASK_STATUSES, noun="task")
        receipt_map = _mapping(receipt, noun="status receipt")
        expected_receipt_map = (
            None
            if expected_control_receipt is None
            else _mapping(
                expected_control_receipt,
                noun="expected control receipt",
            )
        )
        now = _utc_iso()

        with self._connection(write=True) as connection:
            row = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, status, revision, body_json
                FROM tasks WHERE task_cid = ? OR task_alias = ?
                ORDER BY task_cid LIMIT 2
                """,
                [tcid, tcid],
            ).fetchall()
            if not row:
                raise KeyError(tcid)
            if len(row) > 1:
                raise IntentRepositoryIntegrityError("task CID/alias lookup is ambiguous")
            task_row = row[0]
            resolved_cid = str(task_row[0])
            previous_status = str(task_row[3])
            current_revision = int(task_row[4])
            if current_revision != expected:
                raise IntentRepositoryConflictError("task revision CAS is stale")
            body_map = _decode_json(task_row[5], noun="task body")
            if not isinstance(body_map, dict):
                body_map = {}
            current_control_receipt = body_map.get("completion_receipt")
            if expected_receipt_map is not None and (
                not isinstance(current_control_receipt, Mapping)
                or dict(current_control_receipt)
                != dict(expected_receipt_map)
            ):
                raise IntentRepositoryConflictError(
                    "task control receipt CAS is stale"
                )
            if previous_status == status_text:
                return IntentReceipt(
                    event_id="",
                    event_type=IntentEventType.TASK_STATUS_CHANGED.value,
                    global_sequence=self._next_global_sequence(connection) - 1,
                    recorded_at=now,
                    subject_id=resolved_cid,
                    revision=current_revision,
                    changed=False,
                    details=MappingProxyType(
                        {
                            "task_cid": resolved_cid,
                            "status": status_text,
                            "previous_status": previous_status,
                        }
                    ),
                )

            completing = status_text in _COMPLETED_STATUSES
            if completing and not allow_completion_without_evidence:
                # Gate completion on current required evidence inside the same
                # transaction that mutates status.
                missing = self._missing_evidence_on(
                    connection,
                    resolved_cid,
                    evidence_digests=evidence_digests,
                )
                if missing:
                    raise IntentCompletionError(
                        "completion refused without current required evidence: "
                        + ", ".join(missing)
                    )

            revision = current_revision + 1
            body_map = dict(body_map)
            receipt_map = _prepare_database_virgin_transfer_receipt_on(
                connection,
                task={
                    "task_cid": resolved_cid,
                    "task_alias": str(task_row[1]),
                    "status": previous_status,
                    "revision": current_revision,
                    "body": body_map,
                },
                previous_status=previous_status,
                current_revision=current_revision,
                new_status=status_text,
                receipt=receipt_map,
                now_ms=self._clock_ms(),
            )
            if receipt_map:
                body_map["completion_receipt"] = _receipt_with_preserved_reopen_count(
                    receipt_map,
                    body_map.get("completion_receipt"),
                )
                if receipt_map.get("operation") in {
                    "reopen_unimplemented_unknown_callback_quarantine",
                    "requeue_unimplemented_stale_attempt",
                }:
                    raw_reopen_count = receipt_map.get("unknown_callback_reopen_count")
                    if raw_reopen_count is not None:
                        try:
                            body_map["unknown_callback_reopen_count"] = max(
                                0, int(raw_reopen_count)
                            )
                        except (TypeError, ValueError):
                            pass
            connection.execute(
                """
                UPDATE tasks SET status = ?, revision = ?, updated_at = ?,
                    body_json = ?
                WHERE task_cid = ? AND revision = ?
                """,
                [
                    status_text,
                    revision,
                    now,
                    _canonical(body_map, noun="task body"),
                    resolved_cid,
                    current_revision,
                ],
            )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    resolved_cid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="task revision body"),
                    now,
                ],
            )
            event_body: dict[str, Any] = {
                "task_cid": resolved_cid,
                "task_alias": str(task_row[1]),
                "goal_cid": str(task_row[2]),
                "previous_status": previous_status,
                "status": status_text,
                "revision": revision,
                "receipt": receipt_map,
                "recorded_at": now,
            }
            if completing:
                evidence_digest = content_identity(
                    {
                        "task_cid": resolved_cid,
                        "revision": revision,
                        "receipt": receipt_map,
                        "evidence_digests": list(evidence_digests or ()),
                    }
                )
                receipt_cid = content_identity(
                    {
                        "namespace": "completion-receipt",
                        "task_cid": resolved_cid,
                        "revision": revision,
                        "evidence_digest": evidence_digest,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO completion_receipts (
                        receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                        fencing_token, completed_at, validation_run_id,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt_cid,
                        resolved_cid,
                        str(task_row[2]),
                        "",
                        "",
                        0,
                        now,
                        "",
                        evidence_digest,
                        _canonical(
                            {
                                "schema": COMPLETION_EVIDENCE_SCHEMA,
                                "receipt": receipt_map,
                                "evidence_digests": list(evidence_digests or ()),
                                "revision": revision,
                            },
                            noun="completion receipt",
                        ),
                    ],
                )
                event_body["completion_receipt_cid"] = receipt_cid
                event_body["evidence_digest"] = evidence_digest
                event_body["evidence_digests"] = list(evidence_digests or ())
                return self._append_event(
                    connection,
                    event_type=IntentEventType.COMPLETION_RECORDED,
                    subject_id=resolved_cid,
                    task_cid=resolved_cid,
                    body=event_body,
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_STATUS_CHANGED,
                subject_id=resolved_cid,
                task_cid=resolved_cid,
                body=event_body,
            )

    def _missing_evidence_on(
        self,
        connection: Any,
        task_cid: str,
        *,
        evidence_digests: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> tuple[str, ...]:
        return missing_current_evidence_on(
            connection,
            task_cid,
            evidence_digests=evidence_digests,
            now_ms=int(now_ms if now_ms is not None else self._clock_ms()),
            evidence_freshness_seconds=self.evidence_freshness_seconds,
        )

    # -- queue / attempts / blocks -------------------------------------------

    def _record_queue_backoff_on(
        self,
        connection: Any,
        *,
        task_cid: str,
        delay_ms: int,
        reason: str,
        selection_penalty: int,
        now_ms: int,
        exact_retry_not_before_ms: int | None = None,
    ) -> IntentReceipt:
        """Write one queue cooldown on an already-owned transaction."""

        retry_not_before = (
            now_ms + delay_ms
            if exact_retry_not_before_ms is None
            else exact_retry_not_before_ms
        )
        lease = connection.execute(
            "SELECT attempt, fencing_token FROM leases WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        if lease is None:
            attempt = 1
            connection.execute(
                """
                INSERT INTO leases (
                    task_cid, claim_cid, resolution_cid, claimant_did,
                    logical_epoch, fencing_token, expires_at_ms, attempt,
                    state, started_at_ms, release_reason, retry_not_before_ms,
                    owner_session_id, fence_epoch, revision, extension_schema,
                    extension_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task_cid,
                    f"claim:queue:{task_cid}",
                    f"resolution:queue:{task_cid}",
                    self.owner_id,
                    1,
                    1,
                    0,
                    attempt,
                    "released",
                    now_ms,
                    reason,
                    retry_not_before,
                    self.session_id,
                    1,
                    1,
                    QUEUE_ENTRY_SCHEMA,
                    _canonical(
                        {
                            "selection_penalty": selection_penalty,
                            "consecutive_failures": 1,
                            "reason": reason,
                        },
                        noun="queue extension",
                    ),
                ],
            )
        else:
            attempt = int(lease[0]) + 1
            connection.execute(
                """
                UPDATE leases SET
                    attempt = ?, retry_not_before_ms = ?,
                    release_reason = ?, state = 'released',
                    extension_schema = ?, extension_json = ?,
                    revision = revision + 1
                WHERE task_cid = ?
                """,
                [
                    attempt,
                    retry_not_before,
                    reason,
                    QUEUE_ENTRY_SCHEMA,
                    _canonical(
                        {
                            "selection_penalty": selection_penalty,
                            "consecutive_failures": attempt,
                            "reason": reason,
                        },
                        noun="queue extension",
                    ),
                    task_cid,
                ],
            )
        return self._append_event(
            connection,
            event_type=IntentEventType.QUEUE_BACKOFF,
            subject_id=task_cid,
            task_cid=task_cid,
            body={
                "task_cid": task_cid,
                "attempt": attempt,
                "retry_not_before_ms": retry_not_before,
                "delay_ms": delay_ms,
                "selection_penalty": selection_penalty,
                "reason": reason,
                "revision": attempt,
            },
        )

    def record_queue_backoff(
        self,
        *,
        task_cid: str,
        delay_ms: int,
        reason: str = "backoff",
        selection_penalty: int = 0,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        delay = _nonneg_int(delay_ms, noun="delay_ms")
        reason_text = str(reason or "backoff").strip() or "backoff"
        penalty = _nonneg_int(selection_penalty, noun="selection_penalty")
        now_ms = int(self._clock_ms())
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            return self._record_queue_backoff_on(
                connection,
                task_cid=tcid,
                delay_ms=delay,
                reason=reason_text,
                selection_penalty=penalty,
                now_ms=now_ms,
            )

    def record_queue_backoff_and_cas_task_status(
        self,
        *,
        task_cid: str,
        expected_revision: int,
        expected_control_receipt: Mapping[str, Any],
        new_status: str,
        receipt: Mapping[str, Any],
        delay_ms: int,
        reason: str,
        selection_penalty: int = 0,
        exact_retry_not_before_ms: int | None = None,
    ) -> Mapping[str, Any]:
        """Atomically persist one guarded cooldown and retry status.

        The prior task receipt, task revision, queue mutation, and status
        mutation share one owner transaction.  A foreign lane therefore
        cannot leave a stale cooldown behind by winning between queue-first
        and status-CAS operations.
        """

        tcid = _identifier(task_cid, noun="task_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        expected_receipt = _mapping(
            expected_control_receipt,
            noun="expected control receipt",
        )
        status_text = _status(new_status, allowed=_TASK_STATUSES, noun="task")
        if status_text != "retrying":
            raise ValueError("guarded queue/status transition must target retrying")
        receipt_map = _mapping(receipt, noun="status receipt")
        delay = _nonneg_int(delay_ms, noun="delay_ms")
        reason_text = str(reason or "backoff").strip() or "backoff"
        penalty = _nonneg_int(selection_penalty, noun="selection_penalty")
        exact_deadline = (
            None
            if exact_retry_not_before_ms is None
            else _nonneg_int(
                exact_retry_not_before_ms,
                noun="exact_retry_not_before_ms",
            )
        )
        now_ms = int(self._clock_ms())
        now = _utc_iso()

        with self._connection(write=True) as connection:
            rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, status, revision, body_json
                FROM tasks WHERE task_cid = ? OR task_alias = ?
                ORDER BY task_cid LIMIT 2
                """,
                [tcid, tcid],
            ).fetchall()
            if not rows:
                raise KeyError(tcid)
            if len(rows) > 1:
                raise IntentRepositoryIntegrityError(
                    "task CID/alias lookup is ambiguous"
                )
            task_row = rows[0]
            resolved_cid = str(task_row[0])
            previous_status = str(task_row[3])
            current_revision = int(task_row[4])
            body_map = _decode_json(task_row[5], noun="task body")
            if not isinstance(body_map, dict):
                body_map = {}
            current_receipt = body_map.get("completion_receipt")
            if current_revision != expected:
                raise IntentRepositoryConflictError("task revision CAS is stale")
            if (
                not isinstance(current_receipt, Mapping)
                or dict(current_receipt) != dict(expected_receipt)
            ):
                raise IntentRepositoryConflictError(
                    "task control receipt CAS is stale"
                )

            lease = connection.execute(
                """
                SELECT retry_not_before_ms, release_reason, extension_json
                FROM leases WHERE task_cid = ?
                """,
                [resolved_cid],
            ).fetchone()
            extension = (
                _decode_json(lease[2], noun="queue extension")
                if lease is not None
                else {}
            )
            existing_reason = str(
                (extension.get("reason") if isinstance(extension, Mapping) else "")
                or (lease[1] if lease is not None else "")
                or ""
            )
            if (
                lease is not None
                and receipt_map.get("operation")
                in {
                    "database_portal_protected_path_retry_recovery",
                    "database_portal_landed_completion_revalidation",
                }
                and existing_reason != reason_text
            ):
                raise IntentRepositoryConflictError(
                    "typed recovery found a foreign queue entry"
                )
            if previous_status == status_text:
                expected_queue_reason = expected_receipt.get("queue_reason")
                expected_queue_deadline = expected_receipt.get(
                    "retry_not_before_ms"
                )
                if (
                    not isinstance(expected_queue_reason, str)
                    or expected_queue_reason != reason_text
                ):
                    raise IntentRepositoryConflictError(
                        "retrying control receipt does not authorize this queue"
                    )
                if (
                    type(expected_queue_deadline) is not int
                    or expected_queue_deadline < 0
                ):
                    raise IntentRepositoryConflictError(
                        "retrying control queue does not match its receipt"
                    )
                if lease is not None and (
                    existing_reason != reason_text
                    or int(lease[0] or 0) != expected_queue_deadline
                ):
                    raise IntentRepositoryConflictError(
                        "retrying control queue does not match its receipt"
                    )
            desired_retry_not_before_ms = (
                now_ms + delay if exact_deadline is None else exact_deadline
            )
            queue_reused = bool(
                lease is not None
                and existing_reason == reason_text
                and (
                    previous_status == status_text
                    or int(lease[0] or 0) == desired_retry_not_before_ms
                )
            )
            if queue_reused:
                queue_receipt: IntentReceipt | None = None
                retry_not_before_ms = int(lease[0] or 0)
            else:
                queue_receipt = self._record_queue_backoff_on(
                    connection,
                    task_cid=resolved_cid,
                    delay_ms=delay,
                    reason=reason_text,
                    selection_penalty=penalty,
                    now_ms=now_ms,
                    exact_retry_not_before_ms=exact_deadline,
                )
                retry_not_before_ms = desired_retry_not_before_ms

            queue_receipt_dict = (
                queue_receipt.to_dict() if queue_receipt is not None else {}
            )
            transition_receipt = dict(receipt_map)
            if transition_receipt.get("queue_reason") != reason_text:
                raise IntentRepositoryConflictError(
                    "retry receipt does not bind the guarded queue reason"
                )
            if (
                transition_receipt.get("operation")
                == "database_portal_validation_retry_successor_recovery"
            ):
                # Successor recovery deliberately binds the stable queue
                # reason and deadline while omitting the variable queue-event
                # receipt from the task body.  The event remains available in
                # this method's separate queue_receipt result, but carries no
                # additional transition authority.
                if transition_receipt.get("queue_receipt") != {}:
                    raise IntentRepositoryConflictError(
                        "validation retry successor must omit its durable "
                        "queue-event receipt"
                    )
            else:
                transition_receipt["queue_receipt"] = queue_receipt_dict
            if "queue_reused" in transition_receipt:
                transition_receipt["queue_reused"] = queue_reused
            if "retry_not_before_ms" in transition_receipt:
                transition_receipt["retry_not_before_ms"] = retry_not_before_ms
            if previous_status == status_text and queue_receipt is None:
                transition_receipt = dict(expected_receipt)
                status_receipt = IntentReceipt(
                    event_id="",
                    event_type=IntentEventType.TASK_STATUS_CHANGED.value,
                    global_sequence=self._next_global_sequence(connection) - 1,
                    recorded_at=now,
                    subject_id=resolved_cid,
                    revision=current_revision,
                    changed=False,
                    details=MappingProxyType(
                        {
                            "task_cid": resolved_cid,
                            "status": status_text,
                            "previous_status": previous_status,
                        }
                    ),
                )
            else:
                revision = current_revision + 1
                body_map = dict(body_map)
                body_map["completion_receipt"] = transition_receipt
                encoded_body = _canonical(body_map, noun="task body")
                connection.execute(
                    """
                    UPDATE tasks SET status = ?, revision = ?, updated_at = ?,
                        body_json = ?
                    WHERE task_cid = ? AND revision = ?
                    """,
                    [
                        status_text,
                        revision,
                        now,
                        encoded_body,
                        resolved_cid,
                        current_revision,
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO task_revisions (
                        task_cid, revision, status, body_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [resolved_cid, revision, status_text, encoded_body, now],
                )
                status_receipt = self._append_event(
                    connection,
                    event_type=IntentEventType.TASK_STATUS_CHANGED,
                    subject_id=resolved_cid,
                    task_cid=resolved_cid,
                    body={
                        "task_cid": resolved_cid,
                        "task_alias": str(task_row[1]),
                        "goal_cid": str(task_row[2]),
                        "previous_status": previous_status,
                        "status": status_text,
                        "revision": revision,
                        "receipt": transition_receipt,
                        "recorded_at": now,
                    },
                )
            return MappingProxyType(
                {
                    "previous_status": previous_status,
                    "queue_receipt": queue_receipt_dict,
                    "queue_reused": queue_reused,
                    "retry_not_before_ms": retry_not_before_ms,
                    "status_receipt": status_receipt.to_dict(),
                    "transition_receipt": transition_receipt,
                }
            )

    def record_queue_retry(self, *, task_cid: str) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=True) as connection:
            lease = connection.execute(
                "SELECT attempt FROM leases WHERE task_cid = ?", [tcid]
            ).fetchone()
            if lease is None:
                raise KeyError(tcid)
            connection.execute(
                """
                UPDATE leases SET
                    retry_not_before_ms = 0,
                    release_reason = '',
                    extension_json = ?,
                    revision = revision + 1
                WHERE task_cid = ?
                """,
                [
                    _canonical(
                        {
                            "selection_penalty": 0,
                            "consecutive_failures": 0,
                            "reason": "",
                        },
                        noun="queue extension",
                    ),
                    tcid,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.QUEUE_RETRY,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "attempt": int(lease[0]),
                    "retry_not_before_ms": 0,
                    "revision": int(lease[0]),
                },
            )

    def get_queue_entry(self, task_cid: str) -> QueueEntry | None:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT task_cid, attempt, retry_not_before_ms, state,
                       release_reason, extension_json
                FROM leases WHERE task_cid = ?
                """,
                [tcid],
            ).fetchone()
        if row is None:
            return None
        extension = _decode_json(row[5], noun="queue extension")
        if not isinstance(extension, dict):
            extension = {}
        return QueueEntry(
            task_cid=str(row[0]),
            attempt=int(row[1] or 0),
            retry_not_before_ms=int(row[2] or 0),
            selection_penalty=int(extension.get("selection_penalty") or 0),
            consecutive_failures=int(extension.get("consecutive_failures") or 0),
            state=str(row[3] or "released"),
            reason=str(row[4] or extension.get("reason") or ""),
        )

    def record_attempt(
        self,
        *,
        task_cid: str,
        status: str = "started",
        owner_session_id: str = "",
        fencing_token: int = 1,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        status_text = _status(
            status or "started",
            allowed=_ATTEMPT_STATUSES,
            noun="attempt",
        )
        owner = _optional_identifier(owner_session_id, noun="owner_session_id") or self.session_id
        fence = _positive_int(fencing_token, noun="fencing_token")
        now = _utc_iso()
        finished_at = now if status_text in _TERMINAL_ATTEMPT_STATUSES else ""
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) FROM task_attempts WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            attempt_number = int(row[0] if row else 0) + 1
            attempt_id = content_identity(
                {
                    "task_cid": tcid,
                    "attempt_number": attempt_number,
                    "owner_session_id": owner,
                }
            )
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_cid, attempt_number, owner_session_id,
                    fencing_token, fence_epoch, started_at, finished_at,
                    status, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt_id,
                    tcid,
                    attempt_number,
                    owner,
                    fence,
                    1,
                    now,
                    finished_at,
                    status_text,
                    1,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.ATTEMPT_RECORDED,
                subject_id=attempt_id,
                task_cid=tcid,
                attempt_id=attempt_id,
                body={
                    "attempt_id": attempt_id,
                    "task_cid": tcid,
                    "attempt_number": attempt_number,
                    "owner_session_id": owner,
                    "fencing_token": fence,
                    "status": status_text,
                    "revision": 1,
                    "started_at": now,
                    "finished_at": finished_at,
                },
            )

    def block_task(
        self,
        *,
        task_cid: str,
        blocker_kind: str,
        blocker_id: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        kind = _identifier(blocker_kind, noun="blocker_kind")
        bid = _identifier(blocker_id, noun="blocker_id")
        reason_text = str(reason or "").strip() or "blocked"
        now = _utc_iso()
        block_id = content_identity(
            {
                "task_cid": tcid,
                "blocker_kind": kind,
                "blocker_id": bid,
                "reason": reason_text,
                "created_at": now,
            }
        )
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT revision FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            current_revision = int(task_row[0])
            if (
                expected_revision is not None
                and _positive_int(expected_revision, noun="expected_revision") != current_revision
            ):
                raise IntentRepositoryConflictError("task revision CAS is stale while blocking")
            connection.execute(
                """
                INSERT INTO task_blocks (
                    block_id, task_cid, blocker_kind, blocker_id, reason,
                    created_at, cleared_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [block_id, tcid, kind, bid, reason_text, now, "", "active"],
            )
            updated = connection.execute(
                """
                UPDATE tasks SET status = 'blocked', revision = ?, updated_at = ?
                WHERE task_cid = ? AND revision = ?
                RETURNING revision
                """,
                [current_revision + 1, now, tcid, current_revision],
            ).fetchone()
            if updated is None:
                raise IntentRepositoryConflictError("task revision CAS changed while blocking")
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_BLOCKED,
                subject_id=block_id,
                task_cid=tcid,
                body={
                    "block_id": block_id,
                    "task_cid": tcid,
                    "blocker_kind": kind,
                    "blocker_id": bid,
                    "reason": reason_text,
                    "revision": current_revision + 1,
                    "created_at": now,
                },
            )

    def unblock_task(
        self,
        *,
        task_cid: str,
        block_id: str = "",
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT revision, status FROM tasks WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            current_revision = int(task_row[0])
            if (
                expected_revision is not None
                and _positive_int(expected_revision, noun="expected_revision") != current_revision
            ):
                raise IntentRepositoryConflictError("task revision CAS is stale while unblocking")
            if block_id:
                bid = _identifier(block_id, noun="block_id")
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE block_id = ? AND task_cid = ?
                    """,
                    [now, bid, tcid],
                )
            else:
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE task_cid = ? AND state = 'active'
                    """,
                    [now, tcid],
                )
            revision = current_revision + 1
            updated = connection.execute(
                """
                UPDATE tasks SET status = 'ready', revision = ?, updated_at = ?
                WHERE task_cid = ? AND revision = ?
                RETURNING revision
                """,
                [revision, now, tcid, current_revision],
            ).fetchone()
            if updated is None:
                raise IntentRepositoryConflictError("task revision CAS changed while unblocking")
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_UNBLOCKED,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "block_id": block_id or "",
                    "revision": revision,
                    "cleared_at": now,
                },
            )

    def unstall_stale_in_progress_tasks(
        self,
        *,
        now: datetime | None = None,
        stale_seconds: int = STALE_IN_PROGRESS_UNSTALL_SECONDS,
    ) -> dict[str, Any]:
        """Retry leftover in_progress gates so dependents can become ready."""

        with self._connection(write=True) as connection:
            return apply_stale_in_progress_unstall(
                connection, now=now, stale_seconds=stale_seconds
            )

    # -- readiness / selection -----------------------------------------------

    def select_ready_tasks(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        now_ms: int | None = None,
        include_completion_candidates: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return dependency-ready tasks that are not cooling down or blocked.

        Completed tasks are never selected. Tasks that *would* complete still
        require current evidence when ``include_completion_candidates`` is used
        by higher layers; this selector itself only returns non-terminal ready
        work.
        """

        selected = _bounded_limit(limit)
        clock = int(now_ms if now_ms is not None else self._clock_ms())
        _ = include_completion_candidates  # reserved for future selection modes
        with self._connection(write=False) as connection:
            task_rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, ordinal, status, revision
                FROM tasks
                ORDER BY ordinal, task_cid
                """
            ).fetchall()
            dep_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid FROM task_dependencies"
            ).fetchall()
            lease_rows = connection.execute(
                "SELECT task_cid, retry_not_before_ms FROM leases"
            ).fetchall()
            active_blocks = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT task_cid FROM task_blocks WHERE state = 'active'"
                ).fetchall()
            }
            completed = {
                str(row[0])
                for row in connection.execute(
                    "SELECT task_cid FROM tasks WHERE status IN ("
                    + ", ".join("?" for _ in _COMPLETED_STATUSES)
                    + ")",
                    list(_COMPLETED_STATUSES),
                ).fetchall()
            }
        dependencies: dict[str, set[str]] = {}
        # DuckDBRow is a Mapping: iterate rows and index columns, never unpack.
        for row in dep_rows:
            dependencies.setdefault(str(row[0]), set()).add(str(row[1]))
        cooldown = {str(row[0]): int(row[1] or 0) for row in lease_rows}
        ready: list[Mapping[str, Any]] = []
        for row in task_rows:
            tcid = str(row[0])
            alias = str(row[1])
            goal_cid = str(row[2])
            ordinal = int(row[3])
            status = str(row[4])
            revision = int(row[5])
            if status not in _READY_STATUSES:
                continue
            if tcid in active_blocks:
                continue
            if cooldown.get(tcid, 0) > clock:
                continue
            deps = dependencies.get(tcid, set())
            if not deps.issubset(completed):
                continue
            ready.append(
                MappingProxyType(
                    {
                        "task_cid": tcid,
                        "task_alias": alias,
                        "goal_cid": goal_cid,
                        "ordinal": ordinal,
                        "status": status,
                        "revision": revision,
                        "dependencies": tuple(sorted(deps)),
                    }
                )
            )
            if len(ready) >= selected:
                break
        return tuple(ready)

    # -- recovery / rebuild --------------------------------------------------

    def recover(self) -> IntentReceipt:
        """Recover intent projections from admitted events if they diverge.

        Recovery is a pure database operation: rebuild projections from the
        event stream and emit a recovery receipt. No external files are read.
        """

        before = self.snapshot()
        rebuilt = self.rebuild_projections_from_events()
        with self._connection(write=True) as connection:
            return self._append_event(
                connection,
                event_type=IntentEventType.RECOVERY_APPLIED,
                subject_id="intent:recovery",
                body={
                    "before_projection_cid": before.projection_cid,
                    "after_projection_cid": rebuilt.projection_cid,
                    "event_watermark": rebuilt.event_watermark,
                    "revision": rebuilt.event_watermark,
                    "recorded_at": _utc_iso(),
                },
            )

    def rebuild_projections_from_events(self) -> IntentSnapshot:
        """Clear intent projections and re-apply admitted intent events.

        Returns the rebuilt snapshot. Domain events themselves are retained.
        """

        with self._connection(write=True) as connection:
            events = connection.execute(
                """
                SELECT event_id, event_type, task_cid, attempt_id,
                       body_json, global_sequence
                FROM domain_events
                WHERE stream_id = ?
                ORDER BY global_sequence ASC
                """,
                [INTENT_STREAM_ID],
            ).fetchall()
            replayed_validation_run_ids: set[str] = set()
            replayed_validation_result_ids: set[str] = set()
            # Preserve non-intent domain events; only rebuild intent projections.
            for table in _PROJECTION_TABLES:
                try:
                    connection.execute(f"DELETE FROM {table}")
                except Exception:
                    # Some tables may be empty or not present in partial installs.
                    pass
            # Leases are shared with the lease coordinator; only clear queue
            # entries owned by this repository's extension schema.
            try:
                connection.execute(
                    "DELETE FROM leases WHERE extension_schema = ?",
                    [_SHARED_QUEUE_LEASE_SCHEMA],
                )
            except Exception:
                pass
            for event_row in events:
                # DuckDBRow iterates keys; index into values explicitly.
                event_type = str(event_row[1])
                event_attempt_id = str(event_row[3] or "")
                body_json = event_row[4]
                body_wrapper = _decode_json(body_json, noun="event body")
                if not isinstance(body_wrapper, dict):
                    continue
                payload = body_wrapper.get("body")
                if not isinstance(payload, dict):
                    payload = body_wrapper
                if event_type == IntentEventType.VALIDATION_RECORDED.value:
                    run_id = str(payload.get("run_id") or "")
                    result_id = str(payload.get("result_id") or "")
                    if run_id:
                        replayed_validation_run_ids.add(run_id)
                    if result_id:
                        replayed_validation_result_ids.add(result_id)
                self._apply_event_payload(
                    connection,
                    event_type=event_type,
                    payload=payload,
                    attempt_id=event_attempt_id,
                )
            # DuckDB's immediate unique-index checks can reject a delete and
            # reinsert of the same ``(run_id, ordinal)`` in one transaction.
            # Validation projections are therefore updated in place during
            # replay, then rows absent from the admitted event stream are
            # removed before this transaction commits.
            for row in connection.execute("SELECT result_id FROM validation_results").fetchall():
                result_id = str(row[0])
                if result_id not in replayed_validation_result_ids:
                    connection.execute(
                        "DELETE FROM validation_results WHERE result_id = ?",
                        [result_id],
                    )
            for row in connection.execute("SELECT run_id FROM validation_runs").fetchall():
                run_id = str(row[0])
                if run_id not in replayed_validation_run_ids:
                    connection.execute(
                        "DELETE FROM validation_runs WHERE run_id = ?",
                        [run_id],
                    )
        return self.snapshot()

    def _apply_event_payload(
        self,
        connection: Any,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        attempt_id: str = "",
    ) -> None:
        """Project one admitted event into current-state tables (idempotent)."""

        now = str(payload.get("recorded_at") or _utc_iso())
        if event_type == IntentEventType.OBJECTIVE_UPSERTED.value:
            oid = str(payload["objective_id"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute("DELETE FROM objectives WHERE objective_id = ?", [oid])
            connection.execute(
                """
                INSERT INTO objectives (
                    objective_id, objective_alias, parent_objective_id, title,
                    status, priority, created_at, updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    str(payload.get("objective_alias") or oid),
                    str(payload.get("parent_objective_id") or ""),
                    str(payload.get("title") or oid),
                    str(payload.get("status") or "open"),
                    str(payload.get("priority") or "P2"),
                    now,
                    now,
                    revision,
                    _canonical(body, noun="objective body"),
                ],
            )
            connection.execute(
                "DELETE FROM objective_revisions WHERE objective_id = ? AND revision = ?",
                [oid, revision],
            )
            connection.execute(
                """
                INSERT INTO objective_revisions (
                    objective_id, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    revision,
                    str(payload.get("status") or "open"),
                    _canonical(body, noun="objective revision"),
                    now,
                ],
            )
            return

        if event_type == IntentEventType.GOAL_UPSERTED.value:
            gcid = str(payload["goal_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute("DELETE FROM goals WHERE goal_cid = ?", [gcid])
            connection.execute(
                """
                INSERT INTO goals (
                    goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                    title, status, created_at, updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    gcid,
                    str(payload.get("goal_alias") or gcid),
                    str(payload.get("objective_id") or ""),
                    str(payload.get("parent_goal_cid") or ""),
                    int(payload.get("ordinal") or 0),
                    str(payload.get("title") or gcid),
                    str(payload.get("status") or "open"),
                    now,
                    now,
                    revision,
                    _canonical(body, noun="goal body"),
                ],
            )
            return

        if event_type == IntentEventType.GOAL_EDGE_LINKED.value:
            parent = str(payload["parent_goal_cid"])
            child = str(payload["child_goal_cid"])
            kind = str(payload.get("edge_kind") or "depends_on")
            connection.execute(
                """
                DELETE FROM goal_edges
                WHERE parent_goal_cid = ? AND child_goal_cid = ? AND edge_kind = ?
                """,
                [parent, child, kind],
            )
            connection.execute(
                """
                INSERT INTO goal_edges (
                    parent_goal_cid, child_goal_cid, edge_kind
                ) VALUES (?, ?, ?)
                """,
                [parent, child, kind],
            )
            return

        if event_type == IntentEventType.GOAL_REOPENED.value:
            gcid = str(payload["goal_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute(
                """
                UPDATE goals SET status = 'reopened', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE goal_cid = ?
                """,
                [revision, now, _canonical(body, noun="goal body"), gcid],
            )
            return

        if event_type in {
            IntentEventType.PLAN_UPSERTED.value,
            IntentEventType.PLAN_REVISION_APPENDED.value,
            IntentEventType.PLAN_CONTINUED.value,
        }:
            pcid = str(payload["plan_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            status = str(payload.get("status") or "active")
            goal_cid = str(payload.get("goal_cid") or "")
            if event_type == IntentEventType.PLAN_CONTINUED.value:
                status = "active"
            connection.execute("DELETE FROM plans WHERE plan_cid = ?", [pcid])
            connection.execute(
                """
                INSERT INTO plans (
                    plan_cid, goal_cid, plan_alias, status, created_at,
                    updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    pcid,
                    goal_cid,
                    str(payload.get("plan_alias") or pcid),
                    status,
                    now,
                    now,
                    revision,
                    _canonical(body, noun="plan body"),
                ],
            )
            connection.execute(
                "DELETE FROM plan_revisions WHERE plan_cid = ? AND revision = ?",
                [pcid, revision],
            )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [pcid, revision, _canonical(body, noun="plan revision"), now],
            )
            # Mirror live upsert_plan head demotion so rebuild status matches.
            if (
                event_type == IntentEventType.PLAN_UPSERTED.value
                and bool(payload.get("set_head"))
                and status == "active"
                and goal_cid
            ):
                connection.execute(
                    """
                    UPDATE plans SET status = 'superseded', updated_at = ?
                    WHERE goal_cid = ? AND plan_cid <> ? AND status = 'active'
                    """,
                    [now, goal_cid, pcid],
                )
            if event_type == IntentEventType.PLAN_CONTINUED.value:
                predecessor = str(payload.get("continuation_of") or "")
                if predecessor:
                    pred_row = connection.execute(
                        "SELECT revision, body_json FROM plans WHERE plan_cid = ?",
                        [predecessor],
                    ).fetchone()
                    if pred_row is not None:
                        pred_body = _decode_json(pred_row[1], noun="plan body")
                        if not isinstance(pred_body, dict):
                            pred_body = {}
                        else:
                            pred_body = dict(pred_body)
                        pred_body["continued_by"] = pcid
                        connection.execute(
                            """
                            UPDATE plans SET status = 'continued',
                                revision = ?, updated_at = ?, body_json = ?
                            WHERE plan_cid = ?
                            """,
                            [
                                int(pred_row[0]) + 1,
                                now,
                                _canonical(pred_body, noun="plan body"),
                                predecessor,
                            ],
                        )
            return

        if event_type == IntentEventType.PLAN_SUPERSEDED.value:
            pcid = str(payload["plan_cid"])
            successor = str(payload.get("successor_plan_cid") or "")
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute(
                """
                UPDATE plans SET status = 'superseded', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ?
                """,
                [revision, now, _canonical(body, noun="plan body"), pcid],
            )
            if successor:
                connection.execute(
                    """
                    UPDATE plans SET status = 'active', updated_at = ?
                    WHERE plan_cid = ?
                    """,
                    [now, successor],
                )
            return

        if event_type == IntentEventType.TASK_UPSERTED.value:
            tcid = str(payload["task_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            identity = (
                payload.get("identity")
                if isinstance(payload.get("identity"), dict)
                else {"task_cid": tcid}
            )
            connection.execute("DELETE FROM tasks WHERE task_cid = ?", [tcid])
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    str(payload.get("task_alias") or tcid),
                    str(payload.get("goal_cid") or ""),
                    str(payload.get("plan_cid") or ""),
                    str(payload.get("objective_id") or ""),
                    int(payload.get("ordinal") or 0),
                    str(payload.get("status") or "ready"),
                    revision,
                    str(payload.get("priority") or "P2"),
                    now,
                    now,
                    _canonical(identity, noun="task identity"),
                    _canonical(body, noun="task body"),
                ],
            )
            connection.execute(
                "DELETE FROM task_revisions WHERE task_cid = ? AND revision = ?",
                [tcid, revision],
            )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    revision,
                    str(payload.get("status") or "ready"),
                    _canonical(body, noun="task revision body"),
                    now,
                ],
            )
            if "dependencies" in payload:
                deps = payload.get("dependencies") or []
                if isinstance(deps, Sequence) and not isinstance(deps, (str, bytes)):
                    self._set_dependencies_on(connection, tcid, [str(item) for item in deps])
            if "outputs" in payload:
                outputs = payload.get("outputs") or []
                if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
                    self._set_outputs_on(
                        connection,
                        tcid,
                        [item for item in outputs if isinstance(item, Mapping)],
                    )
            if "acceptance" in payload:
                acceptance = payload.get("acceptance") or []
                if isinstance(acceptance, Sequence) and not isinstance(acceptance, (str, bytes)):
                    self._set_acceptance_on(connection, tcid, list(acceptance))
            if "validations" in payload:
                validations = payload.get("validations") or []
                if isinstance(validations, Sequence) and not isinstance(validations, (str, bytes)):
                    self._set_validations_on(connection, tcid, list(validations))
            return

        if event_type == IntentEventType.TASK_DEPENDENCIES_SET.value:
            tcid = str(payload["task_cid"])
            deps = payload.get("dependencies") or []
            if isinstance(deps, Sequence):
                self._set_dependencies_on(connection, tcid, [str(item) for item in deps])
            return

        if event_type in {
            IntentEventType.TASK_STATUS_CHANGED.value,
            IntentEventType.COMPLETION_RECORDED.value,
        }:
            tcid = str(payload["task_cid"])
            revision = int(payload.get("revision") or 1)
            status = str(payload.get("status") or "ready")
            receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else {}
            existing_body_row = connection.execute(
                "SELECT body_json FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if existing_body_row is not None:
                body = _decode_json(existing_body_row[0], noun="task body")
                if not isinstance(body, dict):
                    body = {}
                else:
                    body = dict(body)
            else:
                body = {}
            if receipt:
                body["completion_receipt"] = _receipt_with_preserved_reopen_count(
                    receipt,
                    body.get("completion_receipt"),
                )
            connection.execute(
                """
                UPDATE tasks SET status = ?, revision = ?, updated_at = ?,
                    body_json = ?
                WHERE task_cid = ?
                """,
                [
                    status,
                    revision,
                    now,
                    _canonical(body, noun="task body"),
                    tcid,
                ],
            )
            # Ensure row exists when replaying status after a partial wipe.
            exists = connection.execute("SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]).fetchone()
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_cid, objective_id,
                        ordinal, status, revision, priority, created_at,
                        updated_at, identity_json, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        str(payload.get("task_alias") or tcid),
                        str(payload.get("goal_cid") or ""),
                        "",
                        "",
                        0,
                        status,
                        revision,
                        "P2",
                        now,
                        now,
                        _canonical({"task_cid": tcid}, noun="identity"),
                        _canonical(body, noun="task body"),
                    ],
                )
            connection.execute(
                "DELETE FROM task_revisions WHERE task_cid = ? AND revision = ?",
                [tcid, revision],
            )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    revision,
                    status,
                    _canonical(body, noun="task revision body"),
                    now,
                ],
            )
            if event_type == IntentEventType.COMPLETION_RECORDED.value:
                receipt_cid = str(
                    payload.get("completion_receipt_cid")
                    or content_identity(
                        {
                            "task_cid": tcid,
                            "revision": revision,
                            "status": status,
                        }
                    )
                )
                evidence_digest = str(
                    payload.get("evidence_digest")
                    or content_identity({"task_cid": tcid, "revision": revision})
                )
                raw_evidence_digests = payload.get("evidence_digests", [])
                if not isinstance(raw_evidence_digests, list) or any(
                    not isinstance(item, str) or not item for item in raw_evidence_digests
                ):
                    raise IntentRepositoryIntegrityError(
                        "completion event evidence_digests are malformed"
                    )
                if "evidence_digests" not in payload:
                    reconstructable_legacy_digest = content_identity(
                        {
                            "task_cid": tcid,
                            "revision": revision,
                            "receipt": receipt,
                            "evidence_digests": [],
                        }
                    )
                    if evidence_digest != reconstructable_legacy_digest:
                        raise IntentRepositoryIntegrityError(
                            "legacy completion event omitted nonempty evidence_digests"
                        )
                reconstructed_evidence_digest = content_identity(
                    {
                        "task_cid": tcid,
                        "revision": revision,
                        "receipt": receipt,
                        "evidence_digests": list(raw_evidence_digests),
                    }
                )
                reconstructed_receipt_cid = content_identity(
                    {
                        "namespace": "completion-receipt",
                        "task_cid": tcid,
                        "revision": revision,
                        "evidence_digest": reconstructed_evidence_digest,
                    }
                )
                if (
                    evidence_digest != reconstructed_evidence_digest
                    or receipt_cid != reconstructed_receipt_cid
                ):
                    raise IntentRepositoryIntegrityError(
                        "completion event evidence identity does not reconstruct"
                    )
                connection.execute(
                    "DELETE FROM completion_receipts WHERE receipt_cid = ?",
                    [receipt_cid],
                )
                connection.execute(
                    """
                    INSERT INTO completion_receipts (
                        receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                        fencing_token, completed_at, validation_run_id,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt_cid,
                        tcid,
                        str(payload.get("goal_cid") or ""),
                        "",
                        "",
                        0,
                        now,
                        "",
                        evidence_digest,
                        _canonical(
                            {
                                "schema": COMPLETION_EVIDENCE_SCHEMA,
                                "receipt": receipt,
                                "evidence_digests": list(raw_evidence_digests),
                                "revision": revision,
                            },
                            noun="completion receipt",
                        ),
                    ],
                )
            return

        if event_type == IntentEventType.EVIDENCE_RECORDED.value:
            evidence_id = str(payload["evidence_id"])
            created_at = _event_timestamp(
                payload.get("created_at"), noun="evidence recorded created_at"
            )
            connection.execute(
                "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                [evidence_id],
            )
            connection.execute(
                """
                INSERT INTO evidence_nodes (
                    evidence_id, parent_evidence_id, task_cid, evidence_kind,
                    digest, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    evidence_id,
                    str(payload.get("parent_evidence_id") or ""),
                    str(payload.get("task_cid") or ""),
                    str(payload.get("evidence_kind") or "evidence"),
                    str(payload.get("digest") or ""),
                    created_at,
                    _canonical(
                        payload.get("body") if isinstance(payload.get("body"), dict) else {},
                        noun="evidence body",
                    ),
                ],
            )
            return

        if event_type == IntentEventType.VALIDATION_RECORDED.value:
            run_id = str(payload.get("run_id") or "")
            result_id = str(payload.get("result_id") or "")
            tcid = str(payload.get("task_cid") or "")
            if run_id:
                connection.execute(
                    """
                    INSERT INTO validation_runs (
                        run_id, task_cid, attempt_id, started_at, finished_at,
                        status, command_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id) DO UPDATE SET
                        task_cid = EXCLUDED.task_cid,
                        attempt_id = EXCLUDED.attempt_id,
                        started_at = EXCLUDED.started_at,
                        finished_at = EXCLUDED.finished_at,
                        status = EXCLUDED.status,
                        command_digest = EXCLUDED.command_digest,
                        body_json = EXCLUDED.body_json
                    """,
                    [
                        run_id,
                        tcid,
                        attempt_id,
                        now,
                        now,
                        str(payload.get("outcome") or "passed"),
                        content_identity({"argv": list(payload.get("argv") or ())}),
                        _canonical(
                            {
                                "argv": list(payload.get("argv") or ()),
                                **(
                                    payload.get("body")
                                    if isinstance(payload.get("body"), dict)
                                    else {}
                                ),
                            },
                            noun="validation run",
                        ),
                    ],
                )
            if result_id:
                connection.execute(
                    """
                    INSERT INTO validation_results (
                        result_id, run_id, task_cid, ordinal, outcome,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, ordinal) DO UPDATE SET
                        result_id = EXCLUDED.result_id,
                        task_cid = EXCLUDED.task_cid,
                        outcome = EXCLUDED.outcome,
                        evidence_digest = EXCLUDED.evidence_digest,
                        body_json = EXCLUDED.body_json
                    """,
                    [
                        result_id,
                        run_id,
                        tcid,
                        0,
                        str(payload.get("outcome") or "passed"),
                        str(payload.get("evidence_digest") or ""),
                        _canonical(
                            payload.get("body") if isinstance(payload.get("body"), dict) else {},
                            noun="validation result",
                        ),
                    ],
                )
            if str(payload.get("outcome") or "") == "passed" and tcid:
                evidence_id = content_identity(
                    {
                        "task_cid": tcid,
                        "evidence_kind": "validation",
                        "digest": str(payload.get("evidence_digest") or ""),
                        "run_id": run_id,
                    }
                )
                connection.execute(
                    "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                    [evidence_id],
                )
                connection.execute(
                    """
                    INSERT INTO evidence_nodes (
                        evidence_id, parent_evidence_id, task_cid, evidence_kind,
                        digest, created_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evidence_id,
                        "",
                        tcid,
                        "validation",
                        str(payload.get("evidence_digest") or ""),
                        now,
                        _canonical(
                            {
                                "run_id": run_id,
                                "result_id": result_id,
                                "argv": list(payload.get("argv") or ()),
                                "outcome": str(payload.get("outcome") or "passed"),
                            },
                            noun="validation evidence",
                        ),
                    ],
                )
            return

        if event_type == IntentEventType.QUEUE_BACKOFF.value:
            tcid = str(payload["task_cid"])
            attempt = int(payload.get("attempt") or 1)
            retry = int(payload.get("retry_not_before_ms") or 0)
            reason = str(payload.get("reason") or "backoff")
            penalty = int(payload.get("selection_penalty") or 0)
            exists = connection.execute(
                "SELECT 1 FROM leases WHERE task_cid = ?", [tcid]
            ).fetchone()
            extension = _canonical(
                {
                    "selection_penalty": penalty,
                    "consecutive_failures": attempt,
                    "reason": reason,
                },
                noun="queue extension",
            )
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO leases (
                        task_cid, claim_cid, resolution_cid, claimant_did,
                        logical_epoch, fencing_token, expires_at_ms, attempt,
                        state, started_at_ms, release_reason, retry_not_before_ms,
                        owner_session_id, fence_epoch, revision, extension_schema,
                        extension_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        f"claim:queue:{tcid}",
                        f"resolution:queue:{tcid}",
                        self.owner_id,
                        1,
                        1,
                        0,
                        attempt,
                        "released",
                        0,
                        reason,
                        retry,
                        self.session_id,
                        1,
                        1,
                        QUEUE_ENTRY_SCHEMA,
                        extension,
                    ],
                )
            else:
                connection.execute(
                    """
                    UPDATE leases SET attempt = ?, retry_not_before_ms = ?,
                        release_reason = ?, extension_schema = ?,
                        extension_json = ?, revision = revision + 1
                    WHERE task_cid = ?
                    """,
                    [attempt, retry, reason, QUEUE_ENTRY_SCHEMA, extension, tcid],
                )
            return

        if event_type == IntentEventType.QUEUE_RETRY.value:
            tcid = str(payload["task_cid"])
            connection.execute(
                """
                UPDATE leases SET retry_not_before_ms = 0, release_reason = '',
                    extension_json = ?, revision = revision + 1
                WHERE task_cid = ?
                """,
                [
                    _canonical(
                        {
                            "selection_penalty": 0,
                            "consecutive_failures": 0,
                            "reason": "",
                        },
                        noun="queue extension",
                    ),
                    tcid,
                ],
            )
            return

        if event_type == IntentEventType.ATTEMPT_RECORDED.value:
            attempt_id = str(payload["attempt_id"])
            status = _status(
                payload.get("status") or "started",
                allowed=_ATTEMPT_STATUSES,
                noun="attempt",
            )
            started_at = str(payload.get("started_at") or now)
            raw_finished_at = str(payload.get("finished_at") or "")
            finished_at = (
                raw_finished_at or started_at
                if status in _TERMINAL_ATTEMPT_STATUSES
                else ""
            )
            connection.execute("DELETE FROM task_attempts WHERE attempt_id = ?", [attempt_id])
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_cid, attempt_number, owner_session_id,
                    fencing_token, fence_epoch, started_at, finished_at,
                    status, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt_id,
                    str(payload.get("task_cid") or ""),
                    int(payload.get("attempt_number") or 1),
                    str(payload.get("owner_session_id") or self.session_id),
                    int(payload.get("fencing_token") or 1),
                    1,
                    started_at,
                    finished_at,
                    status,
                    1,
                ],
            )
            return

        if event_type == IntentEventType.TASK_BLOCKED.value:
            block_id = str(payload["block_id"])
            tcid = str(payload["task_cid"])
            created_at = _event_timestamp(
                payload.get("created_at"), noun="task blocked created_at"
            )
            connection.execute("DELETE FROM task_blocks WHERE block_id = ?", [block_id])
            connection.execute(
                """
                INSERT INTO task_blocks (
                    block_id, task_cid, blocker_kind, blocker_id, reason,
                    created_at, cleared_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    block_id,
                    tcid,
                    str(payload.get("blocker_kind") or "manual"),
                    str(payload.get("blocker_id") or "unknown"),
                    str(payload.get("reason") or "blocked"),
                    created_at,
                    "",
                    "active",
                ],
            )
            connection.execute(
                """
                UPDATE tasks SET status = 'blocked', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [int(payload.get("revision") or 1), created_at, tcid],
            )
            return

        if event_type == IntentEventType.TASK_UNBLOCKED.value:
            tcid = str(payload["task_cid"])
            cleared_at = _event_timestamp(
                payload.get("cleared_at"), noun="task unblocked cleared_at"
            )
            block_id = str(payload.get("block_id") or "")
            if block_id:
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE block_id = ? AND task_cid = ?
                    """,
                    [cleared_at, block_id, tcid],
                )
            else:
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE task_cid = ? AND state = 'active'
                    """,
                    [cleared_at, tcid],
                )
            connection.execute(
                """
                UPDATE tasks SET status = 'ready', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [int(payload.get("revision") or 1), cleared_at, tcid],
            )
            return

        # Recovery and unknown types are intentionally no-ops for projection.

    def snapshot(self) -> IntentSnapshot:
        with self._connection(write=False) as connection:
            objective_count = int(
                connection.execute("SELECT COUNT(*) FROM objectives").fetchone()[0]
            )
            goal_count = int(connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0])
            plan_count = int(connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0])
            task_count = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            dependency_count = int(
                connection.execute("SELECT COUNT(*) FROM task_dependencies").fetchone()[0]
            )
            watermark = int(
                connection.execute(
                    "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
                ).fetchone()[0]
            )
            task_rows = connection.execute(
                """
                SELECT task_cid, status, revision FROM tasks
                ORDER BY task_cid
                """
            ).fetchall()
            plan_rows = connection.execute(
                """
                SELECT plan_cid, status, revision FROM plans
                ORDER BY plan_cid
                """
            ).fetchall()
            goal_rows = connection.execute(
                """
                SELECT goal_cid, status, revision FROM goals
                ORDER BY goal_cid
                """
            ).fetchall()
        material = {
            "objectives": objective_count,
            "goals": [
                {"goal_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in goal_rows
            ],
            "plans": [
                {"plan_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in plan_rows
            ],
            "tasks": [
                {"task_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in task_rows
            ],
            "dependency_count": dependency_count,
            "event_watermark": watermark,
        }
        return IntentSnapshot(
            objective_count=objective_count,
            goal_count=goal_count,
            plan_count=plan_count,
            task_count=task_count,
            dependency_count=dependency_count,
            event_watermark=watermark,
            projection_cid=content_identity(material),
            recorded_at=_utc_iso(),
        )

    def task_revision_history_projection(self, task_cid_or_alias: str) -> Mapping[str, Any]:
        """Return bounded task-body revisions for legacy spec-CID replay.

        Task relations are current plan specification and are deliberately not
        duplicated in this lifecycle history.  Callers combine one historical
        body with a separately read full plan projection, then require the
        receipt-bound legacy spec CID before treating that body as a baseline.
        """

        key = _identifier(task_cid_or_alias, noun="task_cid")
        with self._connection(write=False) as connection:
            rows = connection.execute(
                "SELECT task_cid FROM tasks "
                "WHERE task_cid = ? OR task_alias = ? "
                "ORDER BY task_cid LIMIT 2",
                [key, key],
            ).fetchall()
            if not rows:
                raise KeyError(key)
            if len(rows) > 1:
                raise IntentRepositoryIntegrityError("task CID/alias lookup is ambiguous")
            task_cid = str(rows[0][0])
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM task_revisions WHERE task_cid = ?",
                    [task_cid],
                ).fetchone()[0]
            )
            if count > MAX_PROJECTION_RECORDS:
                raise IntentRepositoryBoundsError("task revision history exceeds projection bound")
            revision_rows = connection.execute(
                "SELECT revision, status, body_json FROM task_revisions "
                "WHERE task_cid = ? ORDER BY revision",
                [task_cid],
            ).fetchall()
        return _content_addressed_projection(
            {
                "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
                "task_cid": task_cid,
                "revisions": [
                    {
                        "revision": int(row[0]),
                        "status": str(row[1]),
                        "body": _decode_json(row[2], noun="task revision body"),
                    }
                    for row in revision_rows
                ],
            },
            maximum_bytes=MAX_PLAN_PROJECTION_BYTES,
            noun="task revision history projection",
        )

    def plan_projection(self, *, task_cids: Sequence[str] = ()) -> Mapping[str, Any]:
        """Return a bounded, full-fidelity projection of current plan intent.

        Unlike :meth:`snapshot`, this projection is suitable for plan CAS and
        steering admission: it binds the complete task specification and the
        dependency kind, output, acceptance, and validation relations.  It is
        a read-only projection and deliberately excludes wall-clock metadata
        that does not affect the current plan's meaning.
        """

        requested = _projection_task_cids(task_cids)
        with self._connection(write=False) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                bounded_tables = (
                    "objectives",
                    "goals",
                    "goal_edges",
                    "plans",
                )
                for table in bounded_tables:
                    count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    if count > MAX_PROJECTION_RECORDS:
                        raise IntentRepositoryBoundsError(f"{table} projection count exceeds bound")

                objective_rows = connection.execute(
                    """
                    SELECT objective_id, objective_alias, parent_objective_id,
                           title, status, priority, revision, body_json,
                           extension_schema, extension_json
                    FROM objectives ORDER BY objective_id
                    """
                ).fetchall()
                goal_rows = connection.execute(
                    """
                    SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                           ordinal, title, status, revision, body_json
                    FROM goals ORDER BY goal_cid
                    """
                ).fetchall()
                edge_rows = connection.execute(
                    """
                    SELECT parent_goal_cid, child_goal_cid, edge_kind
                    FROM goal_edges
                    ORDER BY parent_goal_cid, child_goal_cid, edge_kind
                    """
                ).fetchall()
                plan_rows = connection.execute(
                    """
                    SELECT plan_cid, goal_cid, plan_alias, status, revision,
                           body_json
                    FROM plans ORDER BY plan_cid
                    """
                ).fetchall()

                if requested:
                    placeholders = ", ".join("?" for _ in requested)
                    task_rows = connection.execute(
                        f"""
                        SELECT task_cid, task_alias, goal_cid, plan_cid,
                               objective_id, ordinal, status, revision, priority,
                               identity_json, body_json, extension_schema,
                               extension_json
                        FROM tasks WHERE task_cid IN ({placeholders})
                        ORDER BY task_cid
                        """,
                        list(requested),
                    ).fetchall()
                    found = {str(row[0]) for row in task_rows}
                    missing = sorted(set(requested) - found)
                    if missing:
                        raise KeyError(
                            "unknown task_cids in plan projection: " + ", ".join(missing)
                        )
                else:
                    task_count = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
                    if task_count > MAX_PAGE_LIMIT:
                        raise IntentRepositoryBoundsError("projection task count exceeds bound")
                    task_rows = connection.execute(
                        """
                        SELECT task_cid, task_alias, goal_cid, plan_cid,
                               objective_id, ordinal, status, revision, priority,
                               identity_json, body_json, extension_schema,
                               extension_json
                        FROM tasks ORDER BY task_cid
                        """
                    ).fetchall()

                projected_task_cids = tuple(str(row[0]) for row in task_rows)
                dependencies_by_task: dict[str, list[dict[str, Any]]] = {
                    task_cid: [] for task_cid in projected_task_cids
                }
                outputs_by_task: dict[str, list[dict[str, Any]]] = {
                    task_cid: [] for task_cid in projected_task_cids
                }
                acceptance_by_task: dict[str, list[dict[str, Any]]] = {
                    task_cid: [] for task_cid in projected_task_cids
                }
                validations_by_task: dict[str, list[dict[str, Any]]] = {
                    task_cid: [] for task_cid in projected_task_cids
                }
                if projected_task_cids:
                    placeholders = ", ".join("?" for _ in projected_task_cids)
                    relation_queries = {
                        "task_dependencies": (
                            "SELECT task_cid, dependency_task_cid, kind "
                            f"FROM task_dependencies WHERE task_cid IN ({placeholders}) "
                            "ORDER BY task_cid, dependency_task_cid, kind"
                        ),
                        "task_outputs": (
                            "SELECT task_cid, ordinal, path, effect_json "
                            f"FROM task_outputs WHERE task_cid IN ({placeholders}) "
                            "ORDER BY task_cid, ordinal"
                        ),
                        "task_acceptance": (
                            "SELECT task_cid, ordinal, criterion, evidence_policy_json "
                            f"FROM task_acceptance WHERE task_cid IN ({placeholders}) "
                            "ORDER BY task_cid, ordinal"
                        ),
                        "task_validations": (
                            "SELECT task_cid, ordinal, argv_json, policy_json "
                            f"FROM task_validations WHERE task_cid IN ({placeholders}) "
                            "ORDER BY task_cid, ordinal"
                        ),
                    }
                    relation_rows: dict[str, list[Any]] = {}
                    for table, query in relation_queries.items():
                        count = int(
                            connection.execute(
                                "SELECT COUNT(*) FROM "
                                + table
                                + f" WHERE task_cid IN ({placeholders})",
                                list(projected_task_cids),
                            ).fetchone()[0]
                        )
                        if count > MAX_PROJECTION_RECORDS:
                            raise IntentRepositoryBoundsError(
                                f"{table} projection count exceeds bound"
                            )
                        relation_rows[table] = connection.execute(
                            query, list(projected_task_cids)
                        ).fetchall()

                    for row in relation_rows["task_dependencies"]:
                        dependencies_by_task[str(row[0])].append(
                            {
                                "dependency_task_cid": str(row[1]),
                                "kind": str(row[2]),
                            }
                        )
                    for row in relation_rows["task_outputs"]:
                        outputs_by_task[str(row[0])].append(
                            {
                                "ordinal": int(row[1]),
                                "path": str(row[2]),
                                "effect": _decode_json(row[3], noun="output effect"),
                            }
                        )
                    for row in relation_rows["task_acceptance"]:
                        acceptance_by_task[str(row[0])].append(
                            {
                                "ordinal": int(row[1]),
                                "criterion": str(row[2]),
                                "evidence_policy": _decode_json(row[3], noun="acceptance policy"),
                            }
                        )
                    for row in relation_rows["task_validations"]:
                        validations_by_task[str(row[0])].append(
                            {
                                "ordinal": int(row[1]),
                                "argv": _decode_json(row[2], noun="validation argv"),
                                "policy": _decode_json(row[3], noun="validation policy"),
                            }
                        )

                watermark = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
                    ).fetchone()[0]
                )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        objectives = [
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2] or ""),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "revision": int(row[6]),
                "body": _decode_json(row[7], noun="objective body"),
                "extension_schema": str(row[8] or ""),
                "extension": _decode_json(row[9], noun="objective extension"),
            }
            for row in objective_rows
        ]
        goals = [
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2] or ""),
                "parent_goal_cid": str(row[3] or ""),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "body": _decode_json(row[8], noun="goal body"),
            }
            for row in goal_rows
        ]
        goal_edges = [
            {
                "parent_goal_cid": str(row[0]),
                "child_goal_cid": str(row[1]),
                "edge_kind": str(row[2]),
            }
            for row in edge_rows
        ]
        plans = [
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "revision": int(row[4]),
                "body": _decode_json(row[5], noun="plan body"),
            }
            for row in plan_rows
        ]
        tasks: list[dict[str, Any]] = []
        for row in task_rows:
            task_cid = str(row[0])
            task: dict[str, Any] = {
                "task_cid": task_cid,
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3] or ""),
                "objective_id": str(row[4] or ""),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8] or ""),
                "identity": _decode_json(row[9], noun="task identity"),
                "body": _decode_json(row[10], noun="task body"),
                "extension_schema": str(row[11] or ""),
                "extension": _decode_json(row[12], noun="task extension"),
                "dependencies": dependencies_by_task[task_cid],
                "outputs": outputs_by_task[task_cid],
                "acceptance": acceptance_by_task[task_cid],
                "validations": validations_by_task[task_cid],
            }
            task["spec_cid"] = task_projection_spec_cid(task)
            tasks.append(task)

        return _content_addressed_projection(
            {
                "schema": INTENT_PLAN_PROJECTION_SCHEMA,
                "event_watermark": watermark,
                "objectives": objectives,
                "goals": goals,
                "goal_edges": goal_edges,
                "plans": plans,
                "tasks": tasks,
            },
            maximum_bytes=MAX_PLAN_PROJECTION_BYTES,
            noun="intent plan projection",
        )

    def completion_evidence_projection(self, *, task_cids: Sequence[str] = ()) -> Mapping[str, Any]:
        """Return exact current task states and durable completion receipts."""

        requested = _projection_task_cids(task_cids)
        with self._connection(write=False) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                if requested:
                    placeholders = ", ".join("?" for _ in requested)
                    task_rows = connection.execute(
                        "SELECT task_cid, status, revision FROM tasks "
                        f"WHERE task_cid IN ({placeholders}) ORDER BY task_cid",
                        list(requested),
                    ).fetchall()
                    found = {str(row[0]) for row in task_rows}
                    missing = sorted(set(requested) - found)
                    if missing:
                        raise KeyError(
                            "unknown task_cids in completion projection: " + ", ".join(missing)
                        )
                else:
                    task_count = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
                    if task_count > MAX_PAGE_LIMIT:
                        raise IntentRepositoryBoundsError(
                            "completion projection task count exceeds bound"
                        )
                    task_rows = connection.execute(
                        "SELECT task_cid, status, revision FROM tasks ORDER BY task_cid"
                    ).fetchall()

                projected_task_cids = tuple(str(row[0]) for row in task_rows)
                if projected_task_cids:
                    placeholders = ", ".join("?" for _ in projected_task_cids)
                    receipt_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM completion_receipts "
                            f"WHERE task_cid IN ({placeholders})",
                            list(projected_task_cids),
                        ).fetchone()[0]
                    )
                    if receipt_count > MAX_EVIDENCE:
                        raise IntentRepositoryBoundsError(
                            "completion receipt projection count exceeds bound"
                        )
                    receipt_rows = connection.execute(
                        """
                        SELECT receipt_cid, task_cid, goal_cid, attempt_id,
                               claim_cid, fencing_token, completed_at,
                               validation_run_id, evidence_digest, body_json
                        FROM completion_receipts
                        WHERE task_cid IN ("""
                        + placeholders
                        + ") ORDER BY task_cid, completed_at, receipt_cid",
                        list(projected_task_cids),
                    ).fetchall()
                else:
                    receipt_rows = []
                watermark = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
                    ).fetchone()[0]
                )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        task_states = [
            {
                "task_cid": str(row[0]),
                "status": str(row[1]),
                "revision": int(row[2]),
            }
            for row in task_rows
        ]
        completion_receipts = [
            {
                "receipt_cid": str(row[0]),
                "task_cid": str(row[1]),
                "goal_cid": str(row[2]),
                "attempt_id": str(row[3] or ""),
                "claim_cid": str(row[4] or ""),
                "fencing_token": int(row[5]),
                "completed_at": str(row[6]),
                "validation_run_id": str(row[7] or ""),
                "evidence_digest": str(row[8]),
                "body": _decode_json(row[9], noun="completion receipt body"),
            }
            for row in receipt_rows
        ]
        return _content_addressed_projection(
            {
                "schema": INTENT_COMPLETION_PROJECTION_SCHEMA,
                "event_watermark": watermark,
                "task_states": task_states,
                "completion_receipts": completion_receipts,
            },
            maximum_bytes=MAX_COMPLETION_PROJECTION_BYTES,
            noun="intent completion projection",
        )

    def plan_revisions(self) -> PlanRevisionRepository:
        """Return the plan-revision repository view over this intent store."""

        return PlanRevisionRepository(self)


# ---------------------------------------------------------------------------
# PlanRevisionRepository
# ---------------------------------------------------------------------------


class PlanRevisionRepository:
    """Plan revision heads, deltas, supersession, and continuation.

    Interface: ``PlanRevisionRepository@1``.

    Thin, typed facade over :class:`IntentRepository` so plan-revision callers
    do not need the full intent surface. All mutations remain single-transaction
    database operations with domain events.
    """

    INTERFACE: ClassVar[str] = PLAN_REVISION_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = PLAN_REVISION_REPOSITORY_SCHEMA

    def __init__(self, intent: IntentRepository) -> None:
        if not isinstance(intent, IntentRepository):
            raise TypeError("PlanRevisionRepository requires an IntentRepository")
        self._intent = intent

    @property
    def intent(self) -> IntentRepository:
        return self._intent

    def upsert(
        self,
        *,
        plan_cid: str,
        goal_cid: str,
        plan_alias: str,
        status: str = "active",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        set_head: bool = True,
    ) -> IntentReceipt:
        return self._intent.upsert_plan(
            plan_cid=plan_cid,
            goal_cid=goal_cid,
            plan_alias=plan_alias,
            status=status,
            body=body,
            expected_revision=expected_revision,
            set_head=set_head,
        )

    def append_revision(
        self,
        *,
        plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
        delta: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.append_plan_revision(
            plan_cid=plan_cid,
            expected_revision=expected_revision,
            body=body,
            delta=delta,
        )

    def supersede(
        self,
        *,
        plan_cid: str,
        successor_plan_cid: str,
        expected_revision: int,
        reason: str = "superseded",
    ) -> IntentReceipt:
        return self._intent.supersede_plan(
            plan_cid=plan_cid,
            successor_plan_cid=successor_plan_cid,
            expected_revision=expected_revision,
            reason=reason,
        )

    def continue_from(
        self,
        *,
        plan_cid: str,
        continuation_plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.continue_plan(
            plan_cid=plan_cid,
            continuation_plan_cid=continuation_plan_cid,
            expected_revision=expected_revision,
            body=body,
        )

    def get(self, plan_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_plan(plan_cid)

    def head(self, goal_cid: str) -> PlanHead | None:
        return self._intent.get_plan_head(goal_cid)

    def list_revisions(self, plan_cid: str) -> tuple[Mapping[str, Any], ...]:
        pcid = _identifier(plan_cid, noun="plan_cid")
        with self._intent._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT plan_cid, revision, body_json, recorded_at
                FROM plan_revisions
                WHERE plan_cid = ?
                ORDER BY revision ASC
                """,
                [pcid],
            ).fetchall()
        return tuple(
            MappingProxyType(
                {
                    "plan_cid": str(row[0]),
                    "revision": int(row[1]),
                    "body": _decode_json(row[2], noun="plan revision body"),
                    "recorded_at": str(row[3]),
                }
            )
            for row in rows
        )


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------


def _parse_iso_ms(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return int(moment.timestamp() * 1000)
    except ValueError:
        return 0


def missing_current_evidence_on(
    connection: Any,
    task_cid: str,
    *,
    evidence_digests: Sequence[str] | None,
    now_ms: int,
    evidence_freshness_seconds: int = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> tuple[str, ...]:
    """Evaluate the canonical task-completion evidence gate on one transaction.

    This function is shared with the Quack state owner so an authenticated
    remote bundle cannot rely on a client-side precheck or reinterpret the
    task's current acceptance policy.
    """

    clock = int(now_ms)
    freshness_ms = int(evidence_freshness_seconds) * 1000
    acceptance_rows = connection.execute(
        """
        SELECT ordinal, criterion, evidence_policy_json
        FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal
        """,
        [task_cid],
    ).fetchall()
    evidence_rows = connection.execute(
        """
        SELECT evidence_kind, digest, created_at
        FROM evidence_nodes WHERE task_cid = ?
        """,
        [task_cid],
    ).fetchall()
    current_digests: set[str] = set()
    current_kinds: set[str] = set()
    # DuckDBRow iterates column names (Mapping protocol); always index values.
    for row in evidence_rows:
        kind = str(row[0])
        digest = str(row[1])
        created_at = str(row[2] or "")
        created_ms = _parse_iso_ms(created_at)
        if freshness_ms > 0 and created_ms > 0 and clock - created_ms > freshness_ms:
            continue
        current_digests.add(digest)
        current_kinds.add(kind)
    # Caller-supplied digests are advisory cross-checks only; completion
    # authority comes from current stored evidence nodes, never invented
    # digests that are not already recorded against the task.
    if evidence_digests:
        provided = {
            _identifier(item, noun="evidence_digest") for item in evidence_digests
        }
        if not provided.issubset(current_digests):
            return tuple(
                f"digest:{digest}" for digest in sorted(provided - current_digests)
            )
    missing: list[str] = []
    if not acceptance_rows:
        if not current_digests:
            missing.append("required:current_validation_evidence")
        return tuple(missing)
    for row in acceptance_rows:
        ordinal = row[0]
        criterion = row[1]
        policy = _decode_json(row[2], noun="acceptance policy")
        if not isinstance(policy, dict):
            policy = {}
        required_digest = str(
            policy.get("required_digest")
            or policy.get("evidence_digest")
            or policy.get("digest")
            or ""
        ).strip()
        required_kind = str(
            policy.get("evidence_kind") or policy.get("kind") or ""
        ).strip()
        if required_digest:
            if required_digest not in current_digests:
                missing.append(f"digest:{required_digest}")
            continue
        if required_kind:
            if required_kind not in current_kinds:
                missing.append(f"kind:{required_kind}")
            continue
        if not current_digests:
            missing.append(f"criterion:{criterion or ordinal}")
    return tuple(missing)


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def _stable_goal_authority_projection_on(
    connection: Any,
    specification: Mapping[str, Any],
    *,
    root_gate_context: Mapping[str, Any] | None = None,
    transaction_owned_by_caller: bool = False,
) -> dict[str, Any]:
    """Read the complete authority projection from one MVCC snapshot.

    Settlement and execution tables are not all coupled to ``domain_events``;
    an event-watermark sandwich alone therefore cannot prove that the many
    normalized reads observed one database state.  Quack permits transaction
    control while keeping data SQL read-only, so this helper owns a short read
    transaction unless its caller already owns one.  The watermark remains an
    additional integrity assertion inside that snapshot.
    """

    transaction_state = getattr(connection, "in_transaction", False)
    if callable(transaction_state):
        transaction_state = transaction_state()
    nested = transaction_owned_by_caller or transaction_state is True
    owns_transaction = False
    if not nested:
        try:
            connection.execute("BEGIN TRANSACTION")
            owns_transaction = True
        except Exception as exc:
            # Some DB-API adapters do not expose transaction state.  Preserve
            # a caller-owned transaction only when the backend explicitly says
            # that one is already active; all other begin failures fail closed.
            message = str(exc).strip().lower()
            if not any(
                marker in message
                for marker in (
                    "transaction already active",
                    "already in a transaction",
                    "cannot start a transaction within a transaction",
                )
            ):
                raise IntentRepositoryConflictError(
                    "goal authority projection could not start an MVCC read transaction"
                ) from exc
            nested = True
    try:
        before_row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        before = int(before_row[0] if before_row else 0)
        projection, _internal = IntentRepository._goal_authority_state_on(
            connection,
            specification,
            root_gate_context=root_gate_context,
        )
        after_row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        after = int(after_row[0] if after_row else 0)
        if before != int(projection.get("event_watermark") or 0) or before != after:
            raise IntentRepositoryConflictError(
                "goal authority projection changed inside its MVCC snapshot"
            )
        if owns_transaction:
            connection.execute("COMMIT")
            owns_transaction = False
        return projection
    except BaseException:
        if owns_transaction:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
        raise


def goal_authority_projection_on_connection(
    connection: Any,
    specification: Mapping[str, Any],
    *,
    root_gate_context: Mapping[str, Any] | None = None,
    transaction_owned_by_caller: bool = False,
) -> Mapping[str, Any]:
    """Project goal authority on an already admitted read connection.

    The VRIF status operator uses its authenticated Quack attachment here so
    it does not open the live DuckDB file or create a second transport session.
    This helper contains no mutation path.
    """

    if not callable(getattr(connection, "execute", None)):
        raise IntentRepositoryIntegrityError(
            "goal authority projection requires a readable connection"
        )
    projection = _stable_goal_authority_projection_on(
        connection,
        specification,
        root_gate_context=root_gate_context,
        transaction_owned_by_caller=transaction_owned_by_caller,
    )
    return _content_addressed_projection(
        projection,
        maximum_bytes=MAX_GOAL_AUTHORITY_PROJECTION_BYTES,
        noun="goal authority projection",
    )


def open_intent_repository(
    database_path: str | Path | None = None,
    *,
    bound_connection: Any | None = None,
    owner_id: str = DEFAULT_OWNER_ID,
    session_id: str = DEFAULT_SESSION_ID,
    install_schema: bool = True,
    evidence_freshness_seconds: int = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_ms: Any | None = None,
) -> IntentRepository:
    """Open an intent repository against ``control.duckdb`` (or test path)."""

    return IntentRepository(
        database_path,
        bound_connection=bound_connection,
        owner_id=owner_id,
        session_id=session_id,
        install_schema=install_schema,
        evidence_freshness_seconds=evidence_freshness_seconds,
        clock_ms=clock_ms,
    )


__all__ = (
    "INTENT_REPOSITORY_INTERFACE",
    "PLAN_REVISION_REPOSITORY_INTERFACE",
    "INTENT_REPOSITORY_SCHEMA",
    "PLAN_REVISION_REPOSITORY_SCHEMA",
    "INTENT_PLAN_PROJECTION_SCHEMA",
    "INTENT_COMPLETION_PROJECTION_SCHEMA",
    "GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA",
    "GOAL_COMPLETION_RECEIPT_SCHEMA",
    "GOAL_ROOT_COMPLETION_GATE_SCHEMA",
    "GOAL_RUNTIME_SETTLEMENT_BINDING_SCHEMA",
    "GOAL_AUTHORITY_PROJECTION_SCHEMA",
    "GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA",
    "GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA",
    "TASK_PROJECTION_SPEC_SCHEMA",
    "TASK_AUTHORITY_SPEC_SCHEMA",
    "TASK_REVISION_HISTORY_PROJECTION_SCHEMA",
    "MAX_PROJECTION_RECORDS",
    "MAX_TASK_PROJECTION_BYTES",
    "MAX_PLAN_PROJECTION_BYTES",
    "MAX_COMPLETION_PROJECTION_BYTES",
    "MAX_GOAL_AUTHORITY_PROJECTION_BYTES",
    "IntentEventType",
    "IntentRepository",
    "IntentRepositoryError",
    "IntentRepositoryConflictError",
    "IntentRepositoryTransitionError",
    "IntentRepositoryUnknownOutcomeError",
    "IntentRepositoryIntegrityError",
    "IntentRepositoryBoundsError",
    "IntentRepositoryNotOpenError",
    "IntentCompletionError",
    "IntentEvidenceError",
    "DuckDBUnavailableError",
    "IntentReceipt",
    "IntentSnapshot",
    "QueueEntry",
    "PlanHead",
    "PlanRevisionRepository",
    "task_projection_spec_cid",
    "task_authority_spec_cid",
    "goal_authority_projection_on_connection",
    "open_intent_repository",
    "duckdb_available",
)
