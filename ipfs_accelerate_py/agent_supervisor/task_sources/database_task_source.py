"""Database-backed task source adapter over the intent repository (DQP-012).

Interface: ``DatabaseTaskSource@1``

Presents the public task/plan/objective APIs used by scheduler and daemon
callers while retaining **canonical identities** (``task_cid``, ``plan_cid``,
``goal_cid``, ``objective_id``). Mutable display aliases never redefine those
keys.

This adapter is the control-plane cutover path for
:class:`~.duckdb_task_source.DuckDBTaskSource` consumers: status CAS, readiness,
snapshots, and completion all route through :class:`IntentRepository` so that
projections and domain events advance in one database transaction (no
cross-file saga).

Completion cannot be selected without current required evidence — the
intent-repository completion gate is enforced on every complete transition.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import content_identity
from .control_plane_migrations import duckdb_available
from .duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS,
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
    QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
    QuackOwnerCommandRemoteError,
    is_quack_transport_target,
    quack_transport_uri,
    submit_quack_owner_command,
    validate_quack_owner_command,
)
from .intent_repository import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    IntentCompletionError,
    IntentReceipt,
    IntentRepository,
    IntentRepositoryBoundsError,
    IntentRepositoryConflictError,
    IntentRepositoryError,
    IntentRepositoryIntegrityError,
    PlanRevisionRepository,
    QueueEntry,
    open_intent_repository,
    task_projection_spec_cid,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

DATABASE_TASK_SOURCE_INTERFACE: Final[str] = "DatabaseTaskSource@1"
DATABASE_TASK_SOURCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
)
DATABASE_TASK_SOURCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-task-source-snapshot@1"
)
DATABASE_TASK_PAGE_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/database-task-page@1"
DATABASE_TASK_CAS_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/database-task-cas@1"

DEFAULT_QUERY_LIMIT: Final[int] = DEFAULT_PAGE_LIMIT
MAX_QUERY_LIMIT: Final[int] = MAX_PAGE_LIMIT
_REARM_IMMUTABLE_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "completed",
        "skipped",
        "complete",
        "done",
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
    }
)
_REOPENED_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "proposed",
        "admitted",
        "pending",
        "ready",
        "todo",
        "queued",
        "retrying",
        "claimed",
        "in_progress",
        "running",
    }
)
TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION: Final[str] = (
    "database_portal_typed_deferral_budget_exhausted"
)
TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION: Final[str] = (
    "database_portal_typed_deferral_budget_supersession"
)
TYPED_DEFERRAL_BUDGET_SUPERSESSION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-typed-deferral-budget-supersession-request@2"
)
TYPED_DEFERRAL_BUDGET_SUPERSESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-typed-deferral-budget-supersession@2"
)
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-typed-deferral-provider-evidence-admission@1"
)
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS: Final[int] = 5 * 60 * 1000
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_BYTES: Final[int] = 16 * 1024
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL: Final[object] = object()
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_SEAL: Final[object] = object()
_TYPED_DEFERRAL_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-typed-deferral-budget@1"
)
_TYPED_DEFERRAL_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_TYPED_DEFERRAL_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
_TYPED_DEFERRAL_OWNER_REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")
_TYPED_DEFERRAL_ROUTE_TUPLE_FIELDS = (
    "primary_provider_id",
    "primary_model_id",
    "fallback_provider_id",
    "fallback_model_id",
    "fallback_trigger",
    "fallback_reasoning_effort",
)
_TYPED_DEFERRAL_SUPERSESSION_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "operation",
        "task_cid",
        "attempt_id",
        "exhausted_observation_id",
        "exhausted_receipt_id",
        "source_head",
        "source_tree",
        "repair_head",
        "repair_tree",
        "provider_evidence_admission",
    }
)
_TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_FIELDS = frozenset(
    {
        "schema",
        "task_cid",
        "task_revision",
        "attempt_id",
        "exhausted_observation_id",
        "exhausted_receipt_id",
        "exhausted_finished_at_ms",
        "source_head",
        "source_tree",
        "repair_head",
        "repair_tree",
        "quota_probe_receipt_id",
        "quota_probe_observed_at_ms",
        "route_outcome_id",
        "route_id",
        "quota_evidence_id",
        "admitted_at_ms",
        "max_age_ms",
        "owner_command",
        "owner_command_request_id",
        "owner_store_id",
        "owner_store_generation",
    }
)
_TYPED_DEFERRAL_BLOCK_RECEIPT_FIELDS = frozenset(
    {
        "operation",
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "execution_phase",
        "execution_revision",
        "execution_finished_at_ms",
        "reason",
        "retryable",
        "attempt_consumed",
        "typed_deferral_slot_consumed",
        "retry_budget",
        "prior_queue_entry_preserved_inactive",
        "coordination",
        "control_expected_status",
        "control_expected_revision",
    }
)
_TYPED_DEFERRAL_SUPERSESSION_FIELDS = frozenset(
    {
        "schema",
        "operation",
        "task_cid",
        "task_alias",
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "exhausted_observation_id",
        "exhausted_generation_fingerprint",
        "exhausted_receipt_id",
        "exhausted_receipt",
        "source_head",
        "source_tree",
        "repair_head",
        "repair_tree",
        "quota_probe_receipt_id",
        "quota_probe_receipt",
        "route_outcome_id",
        "route_outcome",
        "route_id",
        "quota_evidence_id",
        "provider_evidence_admission_id",
        "provider_evidence_admitted_at_ms",
        "provider_evidence_max_age_ms",
        "provider_evidence_owner_command",
        "provider_evidence_owner_command_request_id",
        "provider_evidence_owner_store_id",
        "provider_evidence_owner_store_generation",
        "control_expected_status",
        "control_expected_revision",
        "supersession_id",
    }
)


# ---------------------------------------------------------------------------
# Errors (mirror duckdb_task_source public vocabulary)
# ---------------------------------------------------------------------------


class DatabaseTaskSourceError(IntentRepositoryError):
    """Base fail-closed error for the database task source adapter."""


class TaskSourceIntegrityError(DatabaseTaskSourceError, IntentRepositoryIntegrityError):
    """Schema, identity, or projection integrity failure."""


class TaskSourceConflictError(DatabaseTaskSourceError, IntentRepositoryConflictError):
    """CAS head or expected-revision conflict."""


class TaskSourceBoundsError(DatabaseTaskSourceError, IntentRepositoryBoundsError):
    """A query or population bound was exceeded."""


class TaskSourceCompletionError(DatabaseTaskSourceError, IntentCompletionError):
    """Completion refused without current required evidence."""


class TypedDeferralRecoveryError(ValueError):
    """An exhausted-budget recovery receipt is absent, stale, or invalid."""


@dataclass(frozen=True)
class _TypedDeferralProviderEvidenceAdmission:
    """Process-local capability minted only after an owner-run fresh canary."""

    admission_json: str = field(repr=False)
    quota_probe_receipt_json: str = field(repr=False)
    route_outcome_json: str = field(repr=False)
    admission_id: str
    seal: object = field(repr=False, compare=False)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskRecord:
    """Canonical task projection row returned by public APIs."""

    task_cid: str
    task_alias: str
    goal_cid: str
    ordinal: int
    status: str
    revision: int
    body: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    plan_cid: str = ""
    objective_id: str = ""
    priority: str = ""
    outputs: tuple[Mapping[str, Any], ...] = ()
    acceptance: tuple[Mapping[str, Any], ...] = ()
    validations: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "goal_cid": self.goal_cid,
            "plan_cid": self.plan_cid,
            "objective_id": self.objective_id,
            "ordinal": int(self.ordinal),
            "status": self.status,
            "revision": int(self.revision),
            "priority": self.priority,
            "body": dict(self.body),
            "dependencies": list(self.dependencies),
            "outputs": [dict(item) for item in self.outputs],
            "acceptance": [dict(item) for item in self.acceptance],
            "validations": [dict(item) for item in self.validations],
        }


@dataclass(frozen=True)
class TaskPage:
    """Bounded page of tasks with optional continuation cursor."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_PAGE_SCHEMA

    tasks: tuple[TaskRecord, ...]
    revision: int
    next_cursor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tasks": [item.to_dict() for item in self.tasks],
            "revision": int(self.revision),
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class CASResult:
    """Result of a compare-and-set status transition."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_CAS_SCHEMA

    task: TaskRecord
    previous_status: str
    revision: int
    event_cursor: int
    changed: bool
    receipt_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task": self.task.to_dict(),
            "previous_status": self.previous_status,
            "revision": int(self.revision),
            "event_cursor": int(self.event_cursor),
            "changed": bool(self.changed),
            "receipt_cid": self.receipt_cid,
        }


@dataclass(frozen=True)
class TaskSourceSnapshot:
    """Bounded snapshot of the database task-source projection."""

    SCHEMA: ClassVar[str] = DATABASE_TASK_SOURCE_SNAPSHOT_SCHEMA

    source_schema: str
    schema_version: int
    plan_root_cid: str
    repository_tree_id: str
    projection_cid: str
    formal_plan_id: str
    source_identity: str
    revision: int
    event_cursor: int
    goal_count: int
    task_count: int
    dependency_count: int
    terminal: bool
    objective_count: int = 0
    plan_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "source_schema": self.source_schema,
            "schema_version": int(self.schema_version),
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "projection_cid": self.projection_cid,
            "formal_plan_id": self.formal_plan_id,
            "source_identity": self.source_identity,
            "revision": int(self.revision),
            "event_cursor": int(self.event_cursor),
            "goal_count": int(self.goal_count),
            "task_count": int(self.task_count),
            "dependency_count": int(self.dependency_count),
            "terminal": bool(self.terminal),
            "objective_count": int(self.objective_count),
            "plan_count": int(self.plan_count),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cursor_encode(revision: int, offset: int) -> str:
    payload = json.dumps(
        {"v": 1, "revision": int(revision), "offset": int(offset)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _cursor_decode(cursor: str, *, revision: int) -> int:
    text = str(cursor or "").strip()
    if not text:
        return 0
    padded = text + ("=" * (-len(text) % 4))
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskSourceConflictError("task page cursor is malformed") from exc
    if not isinstance(payload, Mapping):
        raise TaskSourceConflictError("task page cursor is malformed")
    if int(payload.get("v") or 0) != 1:
        raise TaskSourceConflictError("task page cursor version is unsupported")
    if int(payload.get("revision") or -1) != int(revision):
        raise TaskSourceConflictError("task page cursor revision is stale")
    offset = int(payload.get("offset") or 0)
    if offset < 0:
        raise TaskSourceConflictError("task page cursor offset is invalid")
    return offset


def _task_key(value: str | TaskRecord | Mapping[str, Any]) -> str:
    if isinstance(value, TaskRecord):
        return value.task_cid
    if isinstance(value, Mapping):
        for key in ("task_cid", "cid", "id", "task_alias", "alias"):
            raw = value.get(key)
            if raw:
                return str(raw).strip()
        raise TaskSourceIntegrityError("task key mapping is missing identity")
    text = str(value or "").strip()
    if not text:
        raise TaskSourceIntegrityError("task key must not be empty")
    return text


def _sha256_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, *, noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypedDeferralRecoveryError(f"{noun} must be a mapping")
    return dict(value)


def _positive_integer(value: Any, *, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypedDeferralRecoveryError(f"{noun} must be a positive integer")
    return int(value)


def _source_generation(task_body: Mapping[str, Any]) -> tuple[str, str]:
    source_head = str(task_body.get("base_revision") or "")
    source_tree = str(task_body.get("base_repository_tree_id") or "")
    if _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(source_head) is None:
        raise TypedDeferralRecoveryError(
            "typed-deferral recovery task has no exact source HEAD"
        )
    if _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(source_tree) is None:
        raise TypedDeferralRecoveryError(
            "typed-deferral recovery task has no exact source tree"
        )
    return source_head, source_tree


def _blocked_context(
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    body = _mapping(task_body, noun="typed-deferral task body")
    receipt = _mapping(
        body.get("completion_receipt"),
        noun="typed-deferral exhausted receipt",
    )
    if (
        set(receipt) != _TYPED_DEFERRAL_BLOCK_RECEIPT_FIELDS
        or receipt.get("operation") != TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION
        or receipt.get("reason") != "typed_portal_deferral_budget_exhausted"
        or receipt.get("retryable") is not False
        or receipt.get("attempt_consumed") is not False
        or receipt.get("typed_deferral_slot_consumed") is not True
        or not isinstance(
            receipt.get("prior_queue_entry_preserved_inactive"), bool
        )
        or receipt.get("execution_phase") != "failed"
        or receipt.get("control_expected_status")
        not in {"in_progress", "retrying"}
    ):
        raise TypedDeferralRecoveryError(
            "task is not bound to an exact exhausted typed-deferral receipt"
        )
    revision = _positive_integer(task_revision, noun="task revision")
    if receipt.get("control_expected_revision") != revision - 1:
        raise TypedDeferralRecoveryError(
            "typed-deferral exhausted receipt is stale for the task revision"
        )
    attempt_number = _positive_integer(
        receipt.get("attempt_number"), noun="exhausted attempt number"
    )
    _positive_integer(
        receipt.get("execution_revision"), noun="exhausted execution revision"
    )
    _positive_integer(
        receipt.get("execution_finished_at_ms"),
        noun="exhausted execution finish time",
    )
    for field in ("attempt_id", "claim_id", "lease_id", "owner_session_id"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise TypedDeferralRecoveryError(
                f"typed-deferral exhausted receipt has no {field}"
            )
    for field in ("fencing_token", "fence_epoch"):
        _positive_integer(receipt.get(field), noun=f"exhausted {field}")

    budget = _mapping(receipt.get("retry_budget"), noun="exhausted retry budget")
    observation_id = str(budget.get("observation_id") or "")
    budget_body = dict(budget)
    budget_body.pop("observation_id", None)
    if (
        budget.get("schema") != _TYPED_DEFERRAL_BUDGET_SCHEMA
        or budget.get("task_cid") != task_cid
        or budget.get("task_generation") != task_cid
        or budget.get("exhausted") is not True
        or budget.get("attempt_consumed") is not False
        or budget.get("typed_deferral_slot_consumed") is not True
        or _TYPED_DEFERRAL_SHA256_RE.fullmatch(observation_id) is None
        or observation_id != _sha256_identity(budget_body)
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral exhausted observation is invalid"
        )
    if not isinstance(receipt.get("coordination"), Mapping):
        raise TypedDeferralRecoveryError(
            "typed-deferral exhausted coordination is malformed"
        )
    source_head, source_tree = _source_generation(body)
    # The local binding includes the complete original receipt because the
    # retrying CAS replaces body.completion_receipt.  Restart reconciliation
    # can therefore reproduce, rather than trust, the overwritten authority.
    return (
        receipt,
        budget,
        observation_id,
        source_head,
        source_tree,
    )


def _canonical_typed_deferral_mapping_json(
    value: Mapping[str, Any], *, noun: str
) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypedDeferralRecoveryError(f"{noun} is not canonical JSON") from exc
    if len(encoded.encode("utf-8")) > _TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_BYTES:
        raise TypedDeferralRecoveryError(f"{noun} exceeds the closed size bound")
    return encoded


def _decode_typed_deferral_mapping_json(value: Any, *, noun: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypedDeferralRecoveryError(f"{noun} is not sealed JSON")
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TypedDeferralRecoveryError(f"{noun} is not sealed JSON") from exc
    if not isinstance(decoded, dict) or (
        _canonical_typed_deferral_mapping_json(decoded, noun=noun) != value
    ):
        raise TypedDeferralRecoveryError(f"{noun} is not canonical sealed JSON")
    return decoded


def _typed_deferral_owner_binding(value: Any, *, noun: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise TypedDeferralRecoveryError(f"{noun} is invalid")
    return value


def _validated_provider_pair(
    failure_receipt: Mapping[str, Any],
    route_outcome: Mapping[str, Any],
    *,
    exhausted_finished_at_ms: int,
    admission_now_ms: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from ...agent_implementation_route import (
        resolve_agent_implementation_route,
        valid_agent_implementation_failure_receipt,
    )
    from ..runtime.provider_failure_policy import (
        valid_grok_hard_quota_receipt,
        valid_grok_route_outcome,
    )

    failure = _mapping(failure_receipt, noun="quota probe receipt")
    outcome = _mapping(route_outcome, noun="provider route outcome")
    nonce = str(failure.get("nonce") or "")
    model = str(failure.get("primary_model") or "")
    probe_returncode = failure.get("probe_returncode")
    observed_at_ms = failure.get("observed_at_ms")
    if (
        isinstance(admission_now_ms, bool)
        or not isinstance(admission_now_ms, int)
        or admission_now_ms <= 0
        or isinstance(probe_returncode, bool)
        or not isinstance(probe_returncode, int)
        or isinstance(observed_at_ms, bool)
        or not isinstance(observed_at_ms, int)
        or observed_at_ms < exhausted_finished_at_ms
        or not valid_agent_implementation_failure_receipt(
            failure,
            nonce=nonce,
            model=model,
            probe_returncode=probe_returncode,
            now_ms=admission_now_ms,
            max_age_ms=_TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS,
        )
        or not valid_grok_hard_quota_receipt(
            failure,
            nonce=nonce,
            model=model,
            returncode=probe_returncode,
        )
    ):
        raise TypedDeferralRecoveryError(
            "quota probe is not fresh, post-exhaustion hard-quota evidence"
        )

    route_plan = _mapping(outcome.get("route_plan"), noun="provider route plan")
    try:
        resolved = resolve_agent_implementation_route(
            **{
                field: route_plan.get(field)
                for field in _TYPED_DEFERRAL_ROUTE_TUPLE_FIELDS
            }
        )
    except (TypeError, ValueError) as exc:
        raise TypedDeferralRecoveryError(
            "provider route is not a reviewed canonical tuple"
        ) from exc
    canonical_route = resolved.as_outcome_dict()
    quota_evidence_id = str(outcome.get("quota_evidence_id") or "")
    if (
        resolved.fallback_trigger != "primary_quota_exhausted"
        or resolved.fallback_reasoning_effort != "high"
        or route_plan != canonical_route
        or outcome.get("decision") != "fallback_succeeded"
        or outcome.get("verifier_status") != "confirmed_quota"
        or outcome.get("fallback_dispatched") is not True
        or outcome.get("fallback_returncode") != 0
        or _TYPED_DEFERRAL_SHA256_RE.fullmatch(quota_evidence_id) is None
        or not valid_grok_route_outcome(
            outcome,
            receipt=failure,
            route_plan=canonical_route,
            runner_returncode=0,
        )
    ):
        raise TypedDeferralRecoveryError(
            "provider route is not a confirmed successful quota/high fallback"
        )
    return failure, outcome, canonical_route


def _validated_provider_evidence_admission(
    admission: object,
    *,
    task_cid: str,
    task_revision: int,
    exhausted_receipt: Mapping[str, Any],
    exhausted_observation_id: str,
    source_head: str,
    source_tree: str,
    repair_head: str,
    repair_tree: str,
) -> tuple[
    _TypedDeferralProviderEvidenceAdmission,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        not isinstance(admission, _TypedDeferralProviderEvidenceAdmission)
        or admission.seal is not _TYPED_DEFERRAL_PROVIDER_EVIDENCE_SEAL
    ):
        raise TypedDeferralRecoveryError(
            "provider evidence was not admitted by the state owner"
        )
    body = _decode_typed_deferral_mapping_json(
        admission.admission_json,
        noun="provider evidence admission",
    )
    failure = _decode_typed_deferral_mapping_json(
        admission.quota_probe_receipt_json,
        noun="quota probe receipt",
    )
    outcome = _decode_typed_deferral_mapping_json(
        admission.route_outcome_json,
        noun="provider route outcome",
    )
    admitted_at_ms = body.get("admitted_at_ms")
    owner_request_id = body.get("owner_command_request_id")
    if (
        set(body) != _TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_FIELDS
        or body.get("schema")
        != _TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_SCHEMA
        or admission.admission_id != content_identity(body)
        or body.get("task_cid") != task_cid
        or body.get("task_revision") != task_revision
        or body.get("attempt_id") != exhausted_receipt.get("attempt_id")
        or body.get("exhausted_observation_id") != exhausted_observation_id
        or body.get("exhausted_receipt_id") != content_identity(exhausted_receipt)
        or body.get("exhausted_finished_at_ms")
        != exhausted_receipt.get("execution_finished_at_ms")
        or body.get("source_head") != source_head
        or body.get("source_tree") != source_tree
        or body.get("repair_head") != repair_head
        or body.get("repair_tree") != repair_tree
        or body.get("max_age_ms")
        != _TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS
        or body.get("owner_command")
        != QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET
        or isinstance(admitted_at_ms, bool)
        or not isinstance(admitted_at_ms, int)
        or admitted_at_ms <= 0
        or not isinstance(owner_request_id, str)
        or _TYPED_DEFERRAL_OWNER_REQUEST_ID_RE.fullmatch(owner_request_id) is None
    ):
        raise TypedDeferralRecoveryError(
            "provider evidence admission is stale or mismatched"
        )
    _typed_deferral_owner_binding(
        body.get("owner_store_id"), noun="owner store identity"
    )
    _typed_deferral_owner_binding(
        body.get("owner_store_generation"), noun="owner store generation"
    )
    failure, outcome, route = _validated_provider_pair(
        failure,
        outcome,
        exhausted_finished_at_ms=int(exhausted_receipt["execution_finished_at_ms"]),
        admission_now_ms=admitted_at_ms,
    )
    if (
        body.get("quota_probe_receipt_id") != failure.get("receipt_id")
        or body.get("quota_probe_observed_at_ms") != failure.get("observed_at_ms")
        or body.get("route_outcome_id") != outcome.get("outcome_id")
        or body.get("route_id") != route.get("route_id")
        or body.get("quota_evidence_id") != outcome.get("quota_evidence_id")
    ):
        raise TypedDeferralRecoveryError(
            "provider evidence admission identities do not reproduce"
        )
    return admission, body, failure, outcome, route


def _admit_owner_typed_deferral_provider_evidence(
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
    quota_probe_receipt: Mapping[str, Any],
    route_outcome: Mapping[str, Any],
    owner_command_request_id: str,
    owner_store_id: str,
    owner_store_generation: str,
    admitted_at_ms: int | None = None,
    _owner_admission_sentinel: object,
) -> _TypedDeferralProviderEvidenceAdmission:
    """Mint a process-local capability after the owner runs one fresh canary."""

    if (
        _owner_admission_sentinel
        is not _TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
    ):
        raise TypedDeferralRecoveryError(
            "provider evidence admission requires state-owner authority"
        )
    (
        exhausted_receipt,
        _budget,
        observation_id,
        source_head,
        source_tree,
    ) = _blocked_context(
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
    )
    repair_head_text = str(repair_head or "")
    repair_tree_text = str(repair_tree or "")
    if (
        _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_head_text) is None
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_tree_text) is None
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral repair requires an exact commit and tree"
        )
    request_id = _typed_deferral_owner_binding(
        owner_command_request_id, noun="owner command request identity"
    )
    if _TYPED_DEFERRAL_OWNER_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise TypedDeferralRecoveryError("owner command request identity is invalid")
    store_id = _typed_deferral_owner_binding(
        owner_store_id, noun="owner store identity"
    )
    store_generation = _typed_deferral_owner_binding(
        owner_store_generation, noun="owner store generation"
    )
    observed_now = (
        int(time.time() * 1000) if admitted_at_ms is None else admitted_at_ms
    )
    failure, outcome, route = _validated_provider_pair(
        quota_probe_receipt,
        route_outcome,
        exhausted_finished_at_ms=int(exhausted_receipt["execution_finished_at_ms"]),
        admission_now_ms=observed_now,
    )
    body = {
        "schema": _TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_SCHEMA,
        "task_cid": task_cid,
        "task_revision": int(task_revision),
        "attempt_id": str(exhausted_receipt["attempt_id"]),
        "exhausted_observation_id": observation_id,
        "exhausted_receipt_id": content_identity(exhausted_receipt),
        "exhausted_finished_at_ms": int(
            exhausted_receipt["execution_finished_at_ms"]
        ),
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": repair_head_text,
        "repair_tree": repair_tree_text,
        "quota_probe_receipt_id": str(failure.get("receipt_id") or ""),
        "quota_probe_observed_at_ms": int(failure["observed_at_ms"]),
        "route_outcome_id": str(outcome.get("outcome_id") or ""),
        "route_id": str(route.get("route_id") or ""),
        "quota_evidence_id": str(outcome.get("quota_evidence_id") or ""),
        "admitted_at_ms": observed_now,
        "max_age_ms": _TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS,
        "owner_command": QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
        "owner_command_request_id": request_id,
        "owner_store_id": store_id,
        "owner_store_generation": store_generation,
    }
    return _TypedDeferralProviderEvidenceAdmission(
        admission_json=_canonical_typed_deferral_mapping_json(
            body, noun="provider evidence admission"
        ),
        quota_probe_receipt_json=_canonical_typed_deferral_mapping_json(
            failure, noun="quota probe receipt"
        ),
        route_outcome_json=_canonical_typed_deferral_mapping_json(
            outcome, noun="provider route outcome"
        ),
        admission_id=content_identity(body),
        seal=_TYPED_DEFERRAL_PROVIDER_EVIDENCE_SEAL,
    )


def build_typed_deferral_budget_supersession_request(
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
    provider_evidence_admission: object,
) -> dict[str, Any]:
    """Build a closed request around one state-owner-minted capability."""

    receipt, _budget, observation_id, source_head, source_tree = _blocked_context(
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
    )
    repair_head_text = str(repair_head or "")
    repair_tree_text = str(repair_tree or "")
    if (
        _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_head_text) is None
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_tree_text) is None
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral repair requires an exact commit and tree"
        )
    _validated_provider_evidence_admission(
        provider_evidence_admission,
        task_cid=task_cid,
        task_revision=task_revision,
        exhausted_receipt=receipt,
        exhausted_observation_id=observation_id,
        source_head=source_head,
        source_tree=source_tree,
        repair_head=repair_head_text,
        repair_tree=repair_tree_text,
    )
    return {
        "schema": TYPED_DEFERRAL_BUDGET_SUPERSESSION_REQUEST_SCHEMA,
        "operation": TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION,
        "task_cid": task_cid,
        "attempt_id": str(receipt["attempt_id"]),
        "exhausted_observation_id": observation_id,
        "exhausted_receipt_id": content_identity(receipt),
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": repair_head_text,
        "repair_tree": repair_tree_text,
        "provider_evidence_admission": provider_evidence_admission,
    }


def _build_owner_typed_deferral_budget_supersession_request(
    *,
    task_cid: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    repair_head: str,
    repair_tree: str,
    quota_probe_receipt: Mapping[str, Any],
    route_outcome: Mapping[str, Any],
    owner_command_request_id: str,
    owner_store_id: str,
    owner_store_generation: str,
    admitted_at_ms: int | None = None,
    _owner_admission_sentinel: object,
) -> dict[str, Any]:
    """Owner callback API: seal one fresh pair and bind its exact repair."""

    admission = _admit_owner_typed_deferral_provider_evidence(
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
        quota_probe_receipt=quota_probe_receipt,
        route_outcome=route_outcome,
        owner_command_request_id=owner_command_request_id,
        owner_store_id=owner_store_id,
        owner_store_generation=owner_store_generation,
        admitted_at_ms=admitted_at_ms,
        _owner_admission_sentinel=_owner_admission_sentinel,
    )
    return build_typed_deferral_budget_supersession_request(
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
        repair_head=repair_head,
        repair_tree=repair_tree,
        provider_evidence_admission=admission,
    )


def admit_typed_deferral_budget_supersession(
    *,
    task_cid: str,
    task_alias: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate a request and return the owner-authored durable CAS receipt."""

    raw = _mapping(request, noun="typed-deferral supersession request")
    (
        exhausted_receipt,
        budget,
        observation_id,
        source_head,
        source_tree,
    ) = _blocked_context(
        task_cid=task_cid,
        task_revision=task_revision,
        task_body=task_body,
    )
    if (
        set(raw) != _TYPED_DEFERRAL_SUPERSESSION_REQUEST_FIELDS
        or raw.get("schema")
        != TYPED_DEFERRAL_BUDGET_SUPERSESSION_REQUEST_SCHEMA
        or raw.get("operation")
        != TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION
        or raw.get("task_cid") != task_cid
        or raw.get("attempt_id") != exhausted_receipt.get("attempt_id")
        or raw.get("exhausted_observation_id") != observation_id
        or raw.get("exhausted_receipt_id")
        != content_identity(exhausted_receipt)
        or raw.get("source_head") != source_head
        or raw.get("source_tree") != source_tree
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(str(raw.get("repair_head") or "")) is None
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(str(raw.get("repair_tree") or "")) is None
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession request is stale or mismatched"
        )
    _admission, admission_body, failure, outcome, route = (
        _validated_provider_evidence_admission(
            raw.get("provider_evidence_admission"),
            task_cid=task_cid,
            task_revision=task_revision,
            exhausted_receipt=exhausted_receipt,
            exhausted_observation_id=observation_id,
            source_head=source_head,
            source_tree=source_tree,
            repair_head=str(raw["repair_head"]),
            repair_tree=str(raw["repair_tree"]),
        )
    )
    durable = {
        "schema": TYPED_DEFERRAL_BUDGET_SUPERSESSION_SCHEMA,
        "operation": TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION,
        "task_cid": task_cid,
        "task_alias": str(task_alias or ""),
        "attempt_id": str(exhausted_receipt["attempt_id"]),
        "attempt_number": int(exhausted_receipt["attempt_number"]),
        "claim_id": str(exhausted_receipt["claim_id"]),
        "lease_id": str(exhausted_receipt["lease_id"]),
        "owner_session_id": str(exhausted_receipt["owner_session_id"]),
        "fencing_token": int(exhausted_receipt["fencing_token"]),
        "fence_epoch": int(exhausted_receipt["fence_epoch"]),
        "exhausted_observation_id": observation_id,
        "exhausted_generation_fingerprint": str(
            budget.get("generation_fingerprint") or ""
        ),
        "exhausted_receipt_id": content_identity(exhausted_receipt),
        "exhausted_receipt": exhausted_receipt,
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": str(raw["repair_head"]),
        "repair_tree": str(raw["repair_tree"]),
        "quota_probe_receipt_id": str(failure.get("receipt_id") or ""),
        "quota_probe_receipt": failure,
        "route_outcome_id": str(outcome.get("outcome_id") or ""),
        "route_outcome": outcome,
        "route_id": str(route.get("route_id") or ""),
        "quota_evidence_id": str(outcome.get("quota_evidence_id") or ""),
        "provider_evidence_admission_id": _admission.admission_id,
        "provider_evidence_admitted_at_ms": int(
            admission_body["admitted_at_ms"]
        ),
        "provider_evidence_max_age_ms": int(admission_body["max_age_ms"]),
        "provider_evidence_owner_command": str(
            admission_body["owner_command"]
        ),
        "provider_evidence_owner_command_request_id": str(
            admission_body["owner_command_request_id"]
        ),
        "provider_evidence_owner_store_id": str(
            admission_body["owner_store_id"]
        ),
        "provider_evidence_owner_store_generation": str(
            admission_body["owner_store_generation"]
        ),
        "control_expected_status": "blocked",
        "control_expected_revision": int(task_revision),
    }
    durable["supersession_id"] = content_identity(durable)
    return durable


def validate_typed_deferral_budget_supersession(
    receipt: Mapping[str, Any],
    *,
    task_cid: str,
    task_alias: str,
    task_revision: int,
    task_body: Mapping[str, Any],
    attempt: Mapping[str, Any],
    exhausted_budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce one durable supersession against restart-time authority."""

    raw = _mapping(receipt, noun="typed-deferral supersession receipt")
    identity_body = dict(raw)
    supersession_id = identity_body.pop("supersession_id", None)
    if (
        set(raw) != _TYPED_DEFERRAL_SUPERSESSION_FIELDS
        or raw.get("schema") != TYPED_DEFERRAL_BUDGET_SUPERSESSION_SCHEMA
        or raw.get("operation")
        != TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION
        or supersession_id != content_identity(identity_body)
        or raw.get("task_cid") != task_cid
        or raw.get("task_alias") != str(task_alias or "")
        or raw.get("control_expected_status") != "blocked"
        or raw.get("control_expected_revision") != int(task_revision) - 1
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(str(raw.get("repair_head") or "")) is None
        or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(str(raw.get("repair_tree") or "")) is None
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession identity is invalid or stale"
        )
    current_body = _mapping(task_body, noun="retrying task body")
    source_head, source_tree = _source_generation(current_body)
    if raw.get("source_head") != source_head or raw.get("source_tree") != source_tree:
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession source generation changed"
        )

    exhausted_receipt = _mapping(
        raw.get("exhausted_receipt"), noun="superseded exhausted receipt"
    )
    original_body = dict(current_body)
    original_body["completion_receipt"] = exhausted_receipt
    (
        reproduced_receipt,
        reproduced_budget,
        observation_id,
        _original_head,
        _original_tree,
    ) = _blocked_context(
        task_cid=task_cid,
        task_revision=int(raw["control_expected_revision"]),
        task_body=original_body,
    )
    if (
        raw.get("exhausted_receipt_id") != content_identity(reproduced_receipt)
        or raw.get("exhausted_observation_id") != observation_id
        or reproduced_budget != dict(exhausted_budget)
        or raw.get("exhausted_generation_fingerprint")
        != exhausted_budget.get("generation_fingerprint")
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession does not match the exhausted observation"
        )

    attempt_raw = _mapping(attempt, noun="latest failed attempt")
    expected_attempt = {
        "task_cid": task_cid,
        "attempt_id": raw.get("attempt_id"),
        "attempt_number": raw.get("attempt_number"),
        "claim_id": raw.get("claim_id"),
        "lease_id": raw.get("lease_id"),
        "owner_session_id": raw.get("owner_session_id"),
        "fencing_token": raw.get("fencing_token"),
        "fence_epoch": raw.get("fence_epoch"),
    }
    if any(attempt_raw.get(key) != value for key, value in expected_attempt.items()):
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession does not match the latest failed attempt"
        )

    admitted_at_ms = raw.get("provider_evidence_admitted_at_ms")
    owner_request_id = raw.get("provider_evidence_owner_command_request_id")
    if (
        raw.get("provider_evidence_max_age_ms")
        != _TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS
        or raw.get("provider_evidence_owner_command")
        != QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET
        or not isinstance(owner_request_id, str)
        or _TYPED_DEFERRAL_OWNER_REQUEST_ID_RE.fullmatch(owner_request_id) is None
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral provider admission policy does not reproduce"
        )
    owner_store_id = _typed_deferral_owner_binding(
        raw.get("provider_evidence_owner_store_id"),
        noun="owner store identity",
    )
    owner_store_generation = _typed_deferral_owner_binding(
        raw.get("provider_evidence_owner_store_generation"),
        noun="owner store generation",
    )
    failure, outcome, route = _validated_provider_pair(
        _mapping(raw.get("quota_probe_receipt"), noun="quota probe receipt"),
        _mapping(raw.get("route_outcome"), noun="provider route outcome"),
        exhausted_finished_at_ms=int(
            exhausted_receipt["execution_finished_at_ms"]
        ),
        admission_now_ms=admitted_at_ms,
    )
    if (
        raw.get("quota_probe_receipt_id") != failure.get("receipt_id")
        or raw.get("route_outcome_id") != outcome.get("outcome_id")
        or raw.get("route_id") != route.get("route_id")
        or raw.get("quota_evidence_id") != outcome.get("quota_evidence_id")
    ):
        raise TypedDeferralRecoveryError(
            "typed-deferral supersession provider identities do not reproduce"
        )
    admission_body = {
        "schema": _TYPED_DEFERRAL_PROVIDER_EVIDENCE_ADMISSION_SCHEMA,
        "task_cid": task_cid,
        "task_revision": int(raw["control_expected_revision"]),
        "attempt_id": str(raw["attempt_id"]),
        "exhausted_observation_id": observation_id,
        "exhausted_receipt_id": content_identity(exhausted_receipt),
        "exhausted_finished_at_ms": int(
            exhausted_receipt["execution_finished_at_ms"]
        ),
        "source_head": source_head,
        "source_tree": source_tree,
        "repair_head": str(raw["repair_head"]),
        "repair_tree": str(raw["repair_tree"]),
        "quota_probe_receipt_id": str(failure.get("receipt_id") or ""),
        "quota_probe_observed_at_ms": int(failure["observed_at_ms"]),
        "route_outcome_id": str(outcome.get("outcome_id") or ""),
        "route_id": str(route.get("route_id") or ""),
        "quota_evidence_id": str(outcome.get("quota_evidence_id") or ""),
        "admitted_at_ms": admitted_at_ms,
        "max_age_ms": _TYPED_DEFERRAL_PROVIDER_EVIDENCE_MAX_AGE_MS,
        "owner_command": QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
        "owner_command_request_id": owner_request_id,
        "owner_store_id": owner_store_id,
        "owner_store_generation": owner_store_generation,
    }
    if raw.get("provider_evidence_admission_id") != content_identity(admission_body):
        raise TypedDeferralRecoveryError(
            "typed-deferral provider admission identity does not reproduce"
        )
    return raw


def typed_deferral_budget_supersession_matches(
    receipt: Mapping[str, Any] | None,
    **current: Any,
) -> bool:
    """Return false for every absent/forged/stale restart-time receipt."""

    try:
        validate_typed_deferral_budget_supersession(receipt or {}, **current)
    except (TypeError, ValueError):
        return False
    return True




def _as_task_record(row: Mapping[str, Any]) -> TaskRecord:
    body = row.get("body") if isinstance(row.get("body"), Mapping) else {}
    deps = row.get("dependencies") or ()
    if isinstance(deps, str):
        dep_tuple = (deps,)
    elif isinstance(deps, Sequence):
        dep_tuple = tuple(str(item) for item in deps)
    else:
        dep_tuple = ()
    outputs = row.get("outputs") or ()
    acceptance = row.get("acceptance") or ()
    validations = row.get("validations") or ()
    return TaskRecord(
        task_cid=str(row["task_cid"]),
        task_alias=str(row.get("task_alias") or row["task_cid"]),
        goal_cid=str(row.get("goal_cid") or ""),
        ordinal=int(row.get("ordinal") or 0),
        status=str(row.get("status") or "ready"),
        revision=int(row.get("revision") or 0),
        body=MappingProxyType(dict(body)),
        dependencies=dep_tuple,
        plan_cid=str(row.get("plan_cid") or ""),
        objective_id=str(row.get("objective_id") or ""),
        priority=str(row.get("priority") or ""),
        outputs=tuple(
            MappingProxyType(dict(item)) for item in outputs if isinstance(item, Mapping)
        ),
        acceptance=tuple(
            MappingProxyType(dict(item)) for item in acceptance if isinstance(item, Mapping)
        ),
        validations=tuple(
            MappingProxyType(dict(item)) for item in validations if isinstance(item, Mapping)
        ),
    )


def _intent_receipt_from_dict(payload: Mapping[str, Any]) -> IntentReceipt:
    expected = {
        "schema",
        "event_id",
        "event_type",
        "global_sequence",
        "recorded_at",
        "subject_id",
        "revision",
        "changed",
        "details",
    }
    if set(payload) != expected or payload.get("schema") != IntentReceipt.SCHEMA:
        raise TaskSourceIntegrityError("typed owner receipt is malformed")
    details = payload.get("details")
    if not isinstance(details, Mapping):
        raise TaskSourceIntegrityError("typed owner receipt details are malformed")
    integer_fields = ("global_sequence", "revision")
    if any(type(payload.get(field)) is not int for field in integer_fields):
        raise TaskSourceIntegrityError("typed owner receipt integers are malformed")
    if type(payload.get("changed")) is not bool:
        raise TaskSourceIntegrityError("typed owner receipt changed flag is malformed")
    return IntentReceipt(
        event_id=str(payload["event_id"]),
        event_type=str(payload["event_type"]),
        global_sequence=int(payload["global_sequence"]),
        recorded_at=str(payload["recorded_at"]),
        subject_id=str(payload["subject_id"]),
        revision=int(payload["revision"]),
        changed=bool(payload["changed"]),
        details=MappingProxyType(dict(details)),
    )


def _cas_result_from_dict(payload: Mapping[str, Any]) -> CASResult:
    expected = {
        "schema",
        "task",
        "previous_status",
        "revision",
        "event_cursor",
        "changed",
        "receipt_cid",
    }
    if set(payload) != expected or payload.get("schema") != CASResult.SCHEMA:
        raise TaskSourceIntegrityError("typed owner CAS result is malformed")
    task_payload = payload.get("task")
    if not isinstance(task_payload, Mapping):
        raise TaskSourceIntegrityError("typed owner CAS task is malformed")
    if (
        any(type(payload.get(field)) is not int for field in ("revision", "event_cursor"))
        or type(payload.get("changed")) is not bool
    ):
        raise TaskSourceIntegrityError("typed owner CAS scalars are malformed")
    return CASResult(
        task=_as_task_record(task_payload),
        previous_status=str(payload["previous_status"]),
        revision=int(payload["revision"]),
        event_cursor=int(payload["event_cursor"]),
        changed=bool(payload["changed"]),
        receipt_cid=str(payload["receipt_cid"]),
    )


def _raise_typed_owner_error(exc: QuackOwnerCommandRemoteError) -> None:
    if exc.code == "conflict":
        raise TaskSourceConflictError(exc.message) from exc
    if exc.code == "completion_refused":
        raise TaskSourceCompletionError(exc.message) from exc
    if exc.code == "bounds":
        raise TaskSourceBoundsError(exc.message) from exc
    if exc.code == "integrity":
        raise TaskSourceIntegrityError(exc.message) from exc
    if exc.code == "not_found":
        raise KeyError(exc.message) from exc
    raise DatabaseTaskSourceError(exc.message) from exc


def quack_owner_command_error_code(exc: BaseException) -> str:
    """Map repository failures to the closed client-side error vocabulary."""

    if isinstance(exc, (TaskSourceCompletionError, IntentCompletionError)):
        return "completion_refused"
    if isinstance(exc, (TaskSourceConflictError, IntentRepositoryConflictError)):
        return "conflict"
    if isinstance(exc, (TaskSourceBoundsError, IntentRepositoryBoundsError)):
        return "bounds"
    if isinstance(exc, (TaskSourceIntegrityError, IntentRepositoryIntegrityError)):
        return "integrity"
    if isinstance(exc, KeyError):
        return "not_found"
    return "owner_error"


def execute_quack_owner_command(
    repository: IntentRepository,
    command: str,
    payload: Mapping[str, Any],
    *,
    request_id: str = "",
    store_id: str = "",
    store_generation: str = "",
    typed_deferral_provider_evidence_factory: Any = None,
) -> Mapping[str, Any]:
    """Execute one admitted command through canonical repository methods.

    The repository should be constructed with ``bound_connection=`` on the
    state owner's already-open DuckDB connection.  No SQL arrives from the
    requester and every mutation retains the repository's transaction gates.
    """

    if not isinstance(repository, IntentRepository):
        raise TaskSourceIntegrityError("typed owner command requires an IntentRepository")
    if not repository.uses_bound_connection:
        raise TaskSourceIntegrityError(
            "typed owner command requires the exclusive owner's bound connection"
        )
    args = validate_quack_owner_command(command, payload)
    source = DatabaseTaskSource(intent=repository)

    def execute_once() -> Mapping[str, Any]:
        if command == QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS:
            result = source.compare_and_set_status(
                args["task_cid_or_alias"],
                args["expected_revision"],
                args["status"],
                args.get("receipt"),
                expected_control_receipt=args.get(
                    "expected_control_receipt"
                ),
                evidence_digests=args.get("evidence_digests"),
            )
            return result.to_dict()
        if command == QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS:
            receipt = source.compare_and_set_goal_status(
                args["goal_cid_or_alias"],
                args["expected_revision"],
                args["status"],
                args.get("receipt"),
            )
            return receipt.to_dict()
        if command == QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK:
            result = source.rearm_blocked_task(
                args["task_cid_or_alias"],
                receipt=args.get("receipt"),
            )
            return result.to_dict()
        if command == QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET:
            task = source.get_task(args["task_cid_or_alias"])
            if task is None:
                raise KeyError(args["task_cid_or_alias"])
            status = str(task.status or "").strip().lower()
            if status != "blocked":
                # A replay under a different request identity cannot rerun a
                # provider effect after the first recovery or a later
                # terminal CAS. Preserve the ordinary idempotent no-op rules.
                result = source.rearm_blocked_task(task.task_cid)
                return result.to_dict()
            if not all((request_id, store_id, store_generation)):
                raise TaskSourceIntegrityError(
                    "typed-deferral recovery requires exact owner-command bindings"
                )
            if not callable(typed_deferral_provider_evidence_factory):
                raise TaskSourceIntegrityError(
                    "typed-deferral recovery requires the owner provider boundary"
                )
            # Reject ordinary/stale blocked tasks before crossing the paid
            # provider boundary. The sealed request repeats this check after
            # the canary so a concurrent generation change also fails closed.
            _blocked_context(
                task_cid=task.task_cid,
                task_revision=int(task.revision),
                task_body=task.body,
            )
            evidence = typed_deferral_provider_evidence_factory(
                task_cid=task.task_cid,
                task_revision=int(task.revision),
                task_body=dict(task.body),
                repair_head=args["repair_head"],
                repair_tree=args["repair_tree"],
            )
            if not isinstance(evidence, Mapping) or set(evidence) != {
                "quota_probe_receipt",
                "route_outcome",
            }:
                raise TaskSourceIntegrityError(
                    "owner provider boundary returned malformed recovery evidence"
                )
            request = _build_owner_typed_deferral_budget_supersession_request(
                task_cid=task.task_cid,
                task_revision=int(task.revision),
                task_body=task.body,
                repair_head=args["repair_head"],
                repair_tree=args["repair_tree"],
                quota_probe_receipt=_mapping(
                    evidence.get("quota_probe_receipt"),
                    noun="owner quota probe receipt",
                ),
                route_outcome=_mapping(
                    evidence.get("route_outcome"),
                    noun="owner provider route outcome",
                ),
                owner_command_request_id=request_id,
                owner_store_id=store_id,
                owner_store_generation=store_generation,
                _owner_admission_sentinel=(
                    _TYPED_DEFERRAL_PROVIDER_EVIDENCE_OWNER_SENTINEL
                ),
            )
            result = source.rearm_blocked_task(task.task_cid, receipt=request)
            return result.to_dict()
        if command == QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS:
            result = source.record_queue_backoff_and_cas_status(
                task_cid=args["task_cid"],
                expected_revision=args["expected_revision"],
                expected_control_receipt=args["expected_control_receipt"],
                status=args["status"],
                receipt=args["receipt"],
                delay_ms=args["delay_ms"],
                reason=args["reason"],
                selection_penalty=args.get("selection_penalty", 0),
                exact_retry_not_before_ms=args.get(
                    "exact_retry_not_before_ms"
                ),
            )
            return {
                **{
                    key: value
                    for key, value in result.items()
                    if key != "cas_result"
                },
                "cas_result": result["cas_result"].to_dict(),
            }
        if command == QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF:
            receipt = source.record_queue_backoff(
                task_cid=args["task_cid"],
                delay_ms=args["delay_ms"],
                reason=args.get("reason", "backoff"),
                selection_penalty=args.get("selection_penalty", 0),
            )
        elif command == QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY:
            receipt = source.record_queue_retry(task_cid=args["task_cid"])
        elif command == QUACK_OWNER_COMMAND_RECORD_EVIDENCE:
            receipt = source.record_evidence(
                task_cid=args["task_cid"],
                evidence_kind=args["evidence_kind"],
                digest=args["digest"],
                body=args.get("body"),
            )
        elif command == QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT:
            receipt = source.record_validation_result(
                task_cid=args["task_cid"],
                outcome=args["outcome"],
                evidence_digest=args["evidence_digest"],
                argv=args.get("argv"),
                attempt_id=args.get("attempt_id", ""),
                body=args.get("body"),
            )
        else:  # validate_quack_owner_command keeps this branch unreachable.
            raise TaskSourceIntegrityError("typed owner command is not implemented")
        return receipt.to_dict()

    bindings = (request_id, store_id, store_generation)
    if any(bindings):
        if not all(bindings):
            raise TaskSourceIntegrityError(
                "durable owner command replay requires request, store, and generation bindings"
            )
        return repository.run_idempotent_owner_command(
            request_id=request_id,
            command=command,
            command_payload=args,
            store_id=store_id,
            store_generation=store_generation,
            operation=execute_once,
        )
    return execute_once()


# ---------------------------------------------------------------------------
# DatabaseTaskSource
# ---------------------------------------------------------------------------


class DatabaseTaskSource:
    """Public task-source adapter backed by :class:`IntentRepository`.

    Interface: ``DatabaseTaskSource@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_TASK_SOURCE_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_TASK_SOURCE_SCHEMA

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        intent: IntentRepository | None = None,
        owner_id: str = "database-task-source:local",
        repository_tree_id: str = "",
        plan_root_cid: str = "",
        install_schema: bool = True,
        evidence_freshness_seconds: int = 3600,
        clock_ms: Any | None = None,
    ) -> None:
        if intent is not None:
            self._intent = intent
            self.database_path = Path(intent.database_path)
        else:
            if database_path is None:
                raise ValueError("DatabaseTaskSource requires database_path or intent")
            if is_quack_transport_target(database_path):
                store = quack_transport_uri(database_path)
                self.database_path = store
            else:
                store = Path(database_path).absolute()
                self.database_path = store
            self._intent = open_intent_repository(
                store,
                owner_id=owner_id,
                install_schema=install_schema,
                evidence_freshness_seconds=evidence_freshness_seconds,
                clock_ms=clock_ms,
            )
        self.path = self.database_path
        self.repository_tree_id = str(repository_tree_id or "")
        self.plan_root_cid = str(plan_root_cid or "")
        self.owner_id = owner_id

    # -- lifecycle -----------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return duckdb_available()

    is_available = available

    @property
    def intent(self) -> IntentRepository:
        return self._intent

    @property
    def plans(self) -> PlanRevisionRepository:
        return self._intent.plan_revisions()

    def close(self) -> None:
        self._intent.close()

    def __enter__(self) -> DatabaseTaskSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- materialize / ingest ------------------------------------------------

    def materialize(
        self,
        population: Mapping[str, Any],
        *,
        repository_tree_id: str = "",
        plan_root_cid: str = "",
    ) -> Mapping[str, Any]:
        """Ingest a bounded population of objectives/goals/plans/tasks.

        Accepts a mapping shaped like formal-plan / prompt-graph fixtures and
        writes them through the intent repository in discrete transactions.
        Canonical CIDs supplied by the caller are retained.
        """

        if not isinstance(population, Mapping):
            raise TaskSourceIntegrityError("population must be a mapping")
        tree_id = str(
            repository_tree_id
            or population.get("repository_tree_id")
            or self.repository_tree_id
            or "tree:unknown"
        )
        self.repository_tree_id = tree_id
        root = str(plan_root_cid or population.get("plan_root_cid") or self.plan_root_cid or "")

        objectives = population.get("objectives") or population.get("goals") or ()
        if isinstance(objectives, Mapping):
            objectives = (objectives,)
        goal_cids: list[str] = []
        goal_cids_by_alias: dict[str, str] = {}
        for index, item in enumerate(objectives):
            if not isinstance(item, Mapping):
                continue
            goal_cid = str(item.get("goal_cid") or item.get("goal_id") or f"goal:cid:{index + 1}")
            goal_alias = str(
                item.get("goal_alias") or item.get("goal_id") or item.get("alias") or goal_cid
            )
            objective_id = str(item.get("objective_id") or item.get("owner_actor_id") or "")
            if objective_id:
                self._intent.upsert_objective(
                    objective_id=objective_id,
                    objective_alias=str(item.get("objective_alias") or objective_id),
                    title=str(item.get("title") or objective_id),
                    status=str(item.get("status") or "open"),
                    priority=str(item.get("priority") or "P2"),
                    body={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "objective_id",
                            "objective_alias",
                            "title",
                            "status",
                            "priority",
                        }
                    },
                )
            self._intent.upsert_goal(
                goal_cid=goal_cid,
                goal_alias=goal_alias,
                title=str(item.get("title") or goal_alias),
                objective_id=objective_id,
                parent_goal_cid=str(item.get("parent_goal_cid") or ""),
                ordinal=int(item.get("ordinal") or index + 1),
                status=str(item.get("status") or "open"),
                body={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                },
            )
            goal_cids.append(goal_cid)
            goal_cids_by_alias[goal_alias] = goal_cid

        default_goal = goal_cids[0] if goal_cids else "goal:default"
        if not goal_cids:
            self._intent.upsert_goal(
                goal_cid=default_goal,
                goal_alias="G-DEFAULT",
                title="Default goal",
                ordinal=1,
            )
            goal_cids.append(default_goal)
            goal_cids_by_alias["G-DEFAULT"] = default_goal

        goal_edges = population.get("goal_edges") or ()
        if isinstance(goal_edges, Mapping):
            goal_edges = (goal_edges,)
        goal_edge_count = 0
        for item in goal_edges:
            if not isinstance(item, Mapping):
                continue
            parent_ref = str(item.get("parent_goal_cid") or item.get("parent") or "")
            child_ref = str(item.get("child_goal_cid") or item.get("child") or "")
            parent_cid = goal_cids_by_alias.get(parent_ref, parent_ref)
            child_cid = goal_cids_by_alias.get(child_ref, child_ref)
            self._intent.link_goal_edge(
                parent_goal_cid=parent_cid,
                child_goal_cid=child_cid,
                edge_kind=str(item.get("edge_kind") or "goal_dependency"),
            )
            goal_edge_count += 1

        plans = population.get("plans") or ()
        if isinstance(plans, Mapping):
            plans = (plans,)
        plan_cids: list[str] = []
        for index, item in enumerate(plans):
            if not isinstance(item, Mapping):
                continue
            plan_cid = str(item.get("plan_cid") or item.get("plan_id") or f"plan:{index + 1}")
            plan_alias = str(item.get("plan_alias") or item.get("alias") or plan_cid)
            goal_cid = str(item.get("goal_cid") or default_goal)
            self._intent.upsert_plan(
                plan_cid=plan_cid,
                goal_cid=goal_cid,
                plan_alias=plan_alias,
                status=str(item.get("status") or "active"),
                body=dict(item),
            )
            plan_cids.append(plan_cid)

        if root:
            self.plan_root_cid = root
        elif plan_cids:
            self.plan_root_cid = plan_cids[0]
        else:
            # Synthetic plan head so snapshot identity is non-empty.
            synthetic = content_identity({"repository_tree_id": tree_id, "goals": goal_cids})
            self._intent.upsert_plan(
                plan_cid=synthetic,
                goal_cid=default_goal,
                plan_alias="plan-root",
                status="active",
                body={"repository_tree_id": tree_id},
            )
            self.plan_root_cid = synthetic
            plan_cids.append(synthetic)

        taskboard = population.get("taskboard") or population.get("tasks") or ()
        if isinstance(taskboard, Mapping):
            taskboard = (taskboard,)
        task_cids: list[str] = []
        for index, item in enumerate(taskboard):
            if not isinstance(item, Mapping):
                continue
            task_cid = str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}")
            task_alias = str(
                item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid
            )
            goal_ref = str(item.get("goal_cid") or item.get("goal_id") or default_goal)
            # Resolve aliases to the durable goal_cid before task insert.
            existing_goal = self._intent.get_goal(goal_ref)
            if existing_goal is not None:
                goal_cid = str(existing_goal["goal_cid"])
            else:
                goal_cid = goal_ref
                self._intent.upsert_goal(
                    goal_cid=goal_cid,
                    goal_alias=str(item.get("goal_id") or goal_cid),
                    title=str(item.get("goal_id") or goal_cid),
                    ordinal=index + 1,
                )
            deps_raw = item.get("depends_on") or item.get("dependencies") or ()
            if isinstance(deps_raw, str):
                dependencies = [deps_raw]
            elif isinstance(deps_raw, Sequence):
                dependencies = [str(dep) for dep in deps_raw]
            else:
                dependencies = []
            # Resolve dependency aliases to durable CIDs before task insert so
            # readiness joins never depend on mutable display aliases.
            resolved_deps: list[str] = []
            for dep in dependencies:
                dep_text = str(dep or "").strip()
                if not dep_text:
                    continue
                prior = self._intent.get_task(dep_text)
                resolved_deps.append(str(prior["task_cid"]) if prior is not None else dep_text)
            outputs_raw = item.get("effects") or item.get("outputs") or ()
            outputs: list[Mapping[str, Any]] = []
            if isinstance(outputs_raw, Sequence):
                for effect in outputs_raw:
                    if isinstance(effect, Mapping):
                        outputs.append(dict(effect))
            acceptance_raw = item.get("acceptance_criteria") or item.get("acceptance") or ()
            acceptance: list[Any] = []
            if isinstance(acceptance_raw, (str, Mapping)):
                acceptance = [acceptance_raw]
            elif isinstance(acceptance_raw, Sequence):
                acceptance = list(acceptance_raw)
            validations_raw = item.get("validation_commands") or item.get("validations") or ()
            validations: list[Any] = []
            if isinstance(validations_raw, str):
                validations = [validations_raw]
            elif isinstance(validations_raw, Sequence):
                validations = list(validations_raw)
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=task_alias,
                goal_cid=goal_cid,
                plan_cid=str(item.get("plan_cid") or self.plan_root_cid or ""),
                objective_id=str(item.get("objective_id") or ""),
                ordinal=int(item.get("ordinal") or index + 1),
                status=str(item.get("status") or "ready"),
                priority=str(item.get("priority") or "P2"),
                body={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "task_cid",
                        "task_id",
                        "task_alias",
                        "cid",
                        "goal_cid",
                        "goal_id",
                        "depends_on",
                        "dependencies",
                        "effects",
                        "outputs",
                        "acceptance_criteria",
                        "acceptance",
                        "validation_commands",
                        "validations",
                        "status",
                        "priority",
                        "ordinal",
                        "plan_cid",
                        "objective_id",
                    }
                },
                identity={
                    "task_cid": task_cid,
                    "task_alias": task_alias,
                    "repository_tree_id": tree_id,
                },
                dependencies=resolved_deps,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
            )
            task_cids.append(task_cid)

        snap = self._intent.snapshot()
        receipt = {
            "schema": DATABASE_TASK_SOURCE_SCHEMA,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": tree_id,
            "projection_cid": snap.projection_cid,
            "task_count": len(task_cids),
            "goal_count": len(goal_cids),
            "goal_edge_count": goal_edge_count,
            "plan_count": len(plan_cids),
            "event_watermark": snap.event_watermark,
            "task_cids": list(task_cids),
        }
        return MappingProxyType(receipt)

    # -- reads ---------------------------------------------------------------

    def snapshot(self) -> TaskSourceSnapshot:
        snap = self._intent.snapshot()
        terminal = True
        plan_root = self.plan_root_cid
        for task in self._intent.list_tasks(limit=MAX_QUERY_LIMIT):
            status = str(task.get("status") or "")
            if status not in {
                "completed",
                "skipped",
                "cancelled",
                "failed",
                "quarantined",
                "complete",
                "done",
            }:
                terminal = False
            if not plan_root:
                head = self.plans.head(str(task.get("goal_cid") or ""))
                if head is not None:
                    plan_root = head.plan_cid
        return TaskSourceSnapshot(
            source_schema=DATABASE_TASK_SOURCE_SCHEMA,
            schema_version=1,
            plan_root_cid=plan_root,
            repository_tree_id=self.repository_tree_id,
            projection_cid=snap.projection_cid,
            formal_plan_id=plan_root,
            source_identity=content_identity(
                {
                    "plan_root_cid": plan_root,
                    "repository_tree_id": self.repository_tree_id,
                    "projection_cid": snap.projection_cid,
                }
            ),
            revision=max(1, snap.event_watermark),
            event_cursor=snap.event_watermark,
            goal_count=snap.goal_count,
            task_count=snap.task_count,
            dependency_count=snap.dependency_count,
            terminal=terminal and snap.task_count > 0,
            objective_count=snap.objective_count,
            plan_count=snap.plan_count,
        )

    def plan_projection(self, *, task_cids: Sequence[str] = ()) -> Mapping[str, Any]:
        """Forward the full-fidelity intent plan projection."""

        return self._intent.plan_projection(task_cids=task_cids)

    def task_revision_history_projection(self, task_cid_or_alias: str) -> Mapping[str, Any]:
        """Forward bounded lifecycle bodies used for legacy spec-CID replay."""

        return self._intent.task_revision_history_projection(task_cid_or_alias)

    def completion_evidence_projection(self, *, task_cids: Sequence[str] = ()) -> Mapping[str, Any]:
        """Forward exact completion receipts without creating new authority."""

        return self._intent.completion_evidence_projection(task_cids=task_cids)

    def plan_revision_projection_cid(self) -> str:
        """Return the full plan projection CID used for revision verification."""

        return str(self.plan_projection().get("projection_cid") or "")

    def plan_revision_projection_paths(self) -> tuple[Path, ...]:
        """Declare local files mutated by a plan-revision apply."""

        if isinstance(self.database_path, Path):
            return (self.database_path.resolve(),)
        raise TaskSourceIntegrityError(
            "remote state-owner targets do not expose rollback file paths"
        )

    @staticmethod
    def _plan_population_mapping(source: Any) -> Mapping[str, Any]:
        if isinstance(source, Mapping):
            return source
        for method_name in ("to_dict", "to_record"):
            method = getattr(source, method_name, None)
            if callable(method):
                projected = method()
                if isinstance(projected, Mapping):
                    return projected
        raise TaskSourceIntegrityError(
            "plan revision input must expose a canonical mapping projection"
        )

    @staticmethod
    def _projection_tasks(
        projection: Mapping[str, Any], *, noun: str
    ) -> dict[str, Mapping[str, Any]]:
        raw_tasks = projection.get("tasks")
        if not isinstance(raw_tasks, list):
            raise TaskSourceIntegrityError(f"{noun} has no typed task population")
        tasks: dict[str, Mapping[str, Any]] = {}
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise TaskSourceIntegrityError(f"{noun} task record is malformed")
            task_cid = str(item.get("task_cid") or "")
            if not task_cid or task_cid in tasks:
                raise TaskSourceIntegrityError(f"{noun} task identities are missing or duplicated")
            tasks[task_cid] = item
        return tasks

    @staticmethod
    def _lifecycle_for_status(status: str) -> Any:
        from ..planning.plan_revision_contracts import LifecycleState

        normalized = str(status or "").strip().lower()
        if normalized in {"proposed"}:
            return LifecycleState.PROPOSED
        if normalized in {"admitted"}:
            return LifecycleState.ADMITTED
        if normalized in {"ready", "todo", "pending", "queued", "retrying"}:
            return LifecycleState.UNSTARTED
        if normalized == "blocked":
            return LifecycleState.BLOCKED
        if normalized == "claimed":
            return LifecycleState.CLAIMED
        if normalized in {"in_progress", "running"}:
            return LifecycleState.RUNNING
        if normalized in {"completed", "complete", "done", "skipped"}:
            return LifecycleState.COMPLETED
        if normalized in {"failed", "cancelled", "quarantined", "rejected"}:
            return LifecycleState.FAILED
        raise TaskSourceIntegrityError(
            f"task status {status!r} has no plan-revision lifecycle mapping"
        )

    @staticmethod
    def _task_upsert_relations(
        task: Mapping[str, Any],
    ) -> tuple[list[str], list[Mapping[str, Any]], list[Any], list[Any]]:
        dependencies = [
            str(item.get("dependency_task_cid") or "")
            for item in task.get("dependencies", [])
            if isinstance(item, Mapping)
        ]
        if any(not dependency for dependency in dependencies):
            raise TaskSourceIntegrityError("candidate task contains an empty dependency identity")
        outputs: list[Mapping[str, Any]] = []
        for item in task.get("outputs", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("effect"), Mapping):
                raise TaskSourceIntegrityError("candidate task output effect must be a mapping")
            outputs.append(dict(item["effect"]))
        acceptance: list[Any] = []
        for item in task.get("acceptance", []):
            if not isinstance(item, Mapping) or not isinstance(
                item.get("evidence_policy"), Mapping
            ):
                raise TaskSourceIntegrityError("candidate task acceptance policy must be a mapping")
            acceptance.append(dict(item["evidence_policy"]))
        validations: list[Any] = []
        for item in task.get("validations", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("policy"), Mapping):
                raise TaskSourceIntegrityError("candidate task validation policy must be a mapping")
            validations.append({"argv": list(item.get("argv") or ()), **dict(item["policy"])})
        return dependencies, outputs, acceptance, validations

    def apply_plan_revision(
        self,
        *,
        revision: Any = None,
        admission: Any = None,
        goal_graph: Any = None,
        aliases: Mapping[str, str] | None = None,
        repository_tree_id: str = "",
        retained_task_cids: Sequence[str] = (),
        claimed_task_cids: Sequence[str] = (),
        deferred_item_keys: Sequence[str] = (),
        origin: str = "create",
        delta: Any = None,
        store_continuation: Any | None = None,
        idempotency_key: str = "",
        fencing_token: int | None = None,
    ) -> Mapping[str, Any]:
        """Apply a checked create/steer revision without rewriting task history.

        Steer applies are intentionally narrow: additive tasks and exact-CAS
        amendments of unstarted/blocked task specifications.  Existing logical
        task CIDs, lifecycle states, completion receipts, attempts, and accepted
        history remain in the live IntentRepository.  PlanRevisionStore owns the
        enclosing byte backup/rollback transaction.
        """

        del aliases
        source = goal_graph if goal_graph is not None else admission
        population = self._plan_population_mapping(source)
        normalized_origin = str(origin or "").strip().lower()
        if normalized_origin.endswith("create"):
            if self.plan_projection().get("tasks"):
                raise TaskSourceConflictError(
                    "create cannot rewrite an existing task population; replay "
                    "must be resolved by PlanRevisionStore"
                )
            result = self.materialize(
                population,
                repository_tree_id=repository_tree_id,
                plan_root_cid=str(getattr(revision, "plan_root_cid", "") or ""),
            )
            return MappingProxyType(
                {
                    **dict(result),
                    "projection_cid": self.plan_revision_projection_cid(),
                    "deferred_item_keys": list(deferred_item_keys),
                }
            )

        if not normalized_origin.endswith("steer"):
            raise TaskSourceIntegrityError(f"unsupported plan origin: {origin!r}")
        if store_continuation is None or not idempotency_key:
            raise TaskSourceConflictError("steer requires PlanRevisionStore rollback authority")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TaskSourceConflictError("steer requires a fencing token")
        if fencing_token < 1:
            raise TaskSourceConflictError("steer fencing token must be positive")
        if delta is None or revision is None:
            raise TaskSourceIntegrityError("steer requires a revision and closed delta")

        current_projection = self.plan_projection()
        current_tasks = self._projection_tasks(current_projection, noun="current plan projection")
        source_root = str(self.plan_root_cid or "")
        if not source_root:
            active_plans = [
                item
                for item in current_projection.get("plans", [])
                if isinstance(item, Mapping) and item.get("status") == "active"
            ]
            if len(active_plans) != 1:
                raise TaskSourceIntegrityError("steer requires one exact active predecessor plan")
            source_root = str(active_plans[0].get("plan_cid") or "")
        predecessor = self._intent.get_plan(source_root)
        if predecessor is None:
            raise TaskSourceIntegrityError(
                "steer predecessor plan is absent from the intent repository"
            )

        with tempfile.TemporaryDirectory(prefix="intent-plan-candidate-") as temp_dir:
            candidate_source = DatabaseTaskSource(
                Path(temp_dir) / "candidate.duckdb",
                repository_tree_id=repository_tree_id,
                plan_root_cid=str(getattr(revision, "plan_root_cid", "") or ""),
            )
            try:
                candidate_source.materialize(
                    population,
                    repository_tree_id=repository_tree_id,
                    plan_root_cid=str(getattr(revision, "plan_root_cid", "") or ""),
                )
                candidate_projection = candidate_source.plan_projection()
            finally:
                candidate_source.close()
        candidate_tasks = self._projection_tasks(
            candidate_projection, noun="candidate plan projection"
        )

        current_cids = set(current_tasks)
        candidate_cids = set(candidate_tasks)
        if current_cids - candidate_cids:
            raise TaskSourceConflictError(
                "steer candidate would drop existing logical tasks: "
                + ", ".join(sorted(current_cids - candidate_cids))
            )
        retained = {str(task_cid) for task_cid in retained_task_cids}
        claimed = {str(task_cid) for task_cid in claimed_task_cids}
        if (retained | claimed) - current_cids:
            raise TaskSourceConflictError(
                "steer lifecycle population references an unknown live task"
            )

        from ..planning.plan_revision_contracts import (
            DeltaEffectClass,
            LifecycleState,
            PlanDeltaOperation,
        )

        amendments: dict[str, Any] = {}
        additions: dict[str, Any] = {}
        for item in tuple(getattr(delta, "items", ())):
            effect_class = getattr(item, "effect_class", None)
            if effect_class is not DeltaEffectClass.MATERIALIZABLE_NOW:
                continue
            operation = getattr(item, "operation", None)
            if operation in {
                PlanDeltaOperation.AMEND_UNSTARTED_TASK,
                PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
            }:
                target = str(getattr(item, "target_cid", "") or "")
                if not target or target in amendments:
                    raise TaskSourceIntegrityError("amend delta target is missing or duplicated")
                amendments[target] = item
            elif operation is PlanDeltaOperation.ADD_TASK:
                task_cid = str(getattr(item, "after_record_cid", "") or "")
                if not task_cid or task_cid in additions:
                    raise TaskSourceIntegrityError(
                        "add-task delta identity is missing or duplicated"
                    )
                additions[task_cid] = item
            elif operation in {
                PlanDeltaOperation.ATTACH_EVIDENCE,
                PlanDeltaOperation.RECORD_UNCERTAINTY,
            }:
                continue
            else:
                raise TaskSourceIntegrityError(
                    "DatabaseTaskSource cannot materialize operation "
                    f"{getattr(operation, 'value', operation)!r}"
                )

        new_cids = candidate_cids - current_cids
        if new_cids != set(additions):
            raise TaskSourceConflictError(
                "candidate task additions do not match the admitted delta"
            )
        changed_existing = {
            task_cid
            for task_cid in current_cids
            if task_projection_spec_cid(current_tasks[task_cid])
            != task_projection_spec_cid(candidate_tasks[task_cid])
        }
        if changed_existing != set(amendments):
            raise TaskSourceConflictError(
                "candidate task amendments do not match the admitted delta"
            )

        for task_cid, item in amendments.items():
            current = current_tasks[task_cid]
            candidate = candidate_tasks[task_cid]
            lifecycle = self._lifecycle_for_status(str(current.get("status") or ""))
            if lifecycle not in {
                LifecycleState.PROPOSED,
                LifecycleState.ADMITTED,
                LifecycleState.UNSTARTED,
                LifecycleState.READY,
                LifecycleState.BLOCKED,
            }:
                raise TaskSourceConflictError(
                    f"task {task_cid} is no longer amendable ({lifecycle.value})"
                )
            expected_lifecycle = getattr(item, "expected_target_lifecycle", None)
            if expected_lifecycle not in {lifecycle, LifecycleState.READY}:
                raise TaskSourceConflictError(
                    f"task {task_cid} lifecycle changed before steer apply"
                )
            live_spec = task_projection_spec_cid(current)
            if str(getattr(item, "expected_target_spec_revision", "")) != live_spec:
                raise TaskSourceConflictError(f"task {task_cid} specification CAS is stale")
            candidate_spec = task_projection_spec_cid(candidate)
            if str(getattr(item, "after_record_cid", "")) != candidate_spec:
                raise TaskSourceConflictError(
                    f"task {task_cid} replacement spec CID is not the candidate"
                )
        if claimed & changed_existing:
            raise TaskSourceConflictError("steer would amend claimed task history")

        for task_cid in sorted(
            amendments, key=lambda cid: (int(candidate_tasks[cid].get("ordinal") or 0), cid)
        ):
            candidate = candidate_tasks[task_cid]
            live = current_tasks[task_cid]
            dependencies, outputs, acceptance, validations = self._task_upsert_relations(candidate)
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=str(candidate["task_alias"]),
                goal_cid=str(candidate["goal_cid"]),
                plan_cid=str(live.get("plan_cid") or ""),
                objective_id=str(candidate.get("objective_id") or ""),
                ordinal=int(candidate.get("ordinal") or 0),
                status=str(live.get("status") or "ready"),
                priority=str(candidate.get("priority") or ""),
                body=dict(candidate.get("body") or {}),
                identity=dict(candidate.get("identity") or {}),
                dependencies=dependencies,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
                expected_revision=int(live.get("revision") or 0),
            )

        candidate_root = str(getattr(revision, "plan_root_cid", "") or "")
        if not candidate_root:
            raise TaskSourceIntegrityError("steer revision has no plan root CID")
        for task_cid in sorted(
            additions, key=lambda cid: (int(candidate_tasks[cid].get("ordinal") or 0), cid)
        ):
            candidate = candidate_tasks[task_cid]
            candidate_lifecycle = self._lifecycle_for_status(str(candidate.get("status") or ""))
            if candidate_lifecycle not in {
                LifecycleState.PROPOSED,
                LifecycleState.ADMITTED,
                LifecycleState.READY,
                LifecycleState.UNSTARTED,
                LifecycleState.BLOCKED,
            }:
                raise TaskSourceConflictError(
                    f"new task {task_cid} has a non-admissible initial lifecycle"
                )
            dependencies, outputs, acceptance, validations = self._task_upsert_relations(candidate)
            self._intent.upsert_task(
                task_cid=task_cid,
                task_alias=str(candidate["task_alias"]),
                goal_cid=str(candidate["goal_cid"]),
                plan_cid=candidate_root,
                objective_id=str(candidate.get("objective_id") or ""),
                ordinal=int(candidate.get("ordinal") or 0),
                status=str(candidate.get("status") or "ready"),
                priority=str(candidate.get("priority") or ""),
                body=dict(candidate.get("body") or {}),
                identity=dict(candidate.get("identity") or {}),
                dependencies=dependencies,
                outputs=outputs,
                acceptance=acceptance,
                validations=validations,
                expected_revision=0,
            )

        continuation = self.plans.continue_from(
            plan_cid=source_root,
            continuation_plan_cid=candidate_root,
            expected_revision=int(predecessor.get("revision") or 0),
            body={
                "revision_cid": str(getattr(revision, "revision_cid", "") or ""),
                "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
                "idempotency_key": idempotency_key,
                "repository_tree_id": repository_tree_id,
            },
        )
        self.plan_root_cid = candidate_root
        projection = self.plan_projection()
        projected_tasks = self._projection_tasks(projection, noun="applied plan projection")
        for task_cid in set(amendments) | set(additions):
            if task_projection_spec_cid(projected_tasks[task_cid]) != (
                task_projection_spec_cid(candidate_tasks[task_cid])
            ):
                raise TaskSourceIntegrityError(
                    f"task {task_cid} failed post-apply spec verification"
                )
        return MappingProxyType(
            {
                "projection_cid": str(projection.get("projection_cid") or ""),
                "receipt_cid": continuation.event_id,
                "plan_root_cid": candidate_root,
                "changed": bool(amendments or additions),
                "replayed": False,
                "amended_task_cids": sorted(amendments),
                "added_task_cids": sorted(additions),
                "deferred_item_keys": list(deferred_item_keys),
                "delta_cid": str(getattr(delta, "delta_cid", "") or ""),
            }
        )

    def get_task(
        self, task_cid_or_alias: str | TaskRecord | Mapping[str, Any]
    ) -> TaskRecord | None:
        key = _task_key(task_cid_or_alias)
        row = self._intent.get_task(key)
        if row is None:
            return None
        return _as_task_record(row)

    get = get_task

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        snap = self._intent.snapshot()
        revision = max(1, snap.event_watermark)
        offset = _cursor_decode(cursor, revision=revision) if cursor else 0
        # Intent repository max page is MAX_QUERY_LIMIT; requesting limit+1 at
        # the ceiling raises. Cap the probe and treat a full max page as more.
        fetch_limit = min(limit + 1, MAX_QUERY_LIMIT)
        rows = self._intent.list_tasks(status=status, limit=fetch_limit, offset=offset)
        if limit >= MAX_QUERY_LIMIT:
            has_more = len(rows) >= MAX_QUERY_LIMIT
            page_rows = rows[:limit]
        else:
            has_more = len(rows) > limit
            page_rows = rows[:limit]
        tasks = tuple(_as_task_record(row) for row in page_rows)
        next_cursor = _cursor_encode(revision, offset + len(tasks)) if has_more else ""
        return TaskPage(tasks=tasks, revision=revision, next_cursor=next_cursor)

    def ready_tasks(
        self,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        # completed_ids / blocked_ids are advisory overlays for callers that
        # track ephemeral state outside the durable projection.
        completed = {str(item).strip() for item in completed_ids if str(item).strip()}
        blocked = {str(item).strip() for item in blocked_ids if str(item).strip()}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        selected = self._intent.select_ready_tasks(limit=MAX_QUERY_LIMIT)
        filtered: list[TaskRecord] = []
        for item in selected:
            tcid = str(item["task_cid"])
            alias = str(item.get("task_alias") or "")
            if tcid in blocked or alias in blocked:
                continue
            if tcid in completed or alias in completed:
                continue
            full = self._intent.get_task(tcid)
            if full is None:
                continue
            filtered.append(_as_task_record(full))
            if len(filtered) >= limit:
                break
        snap = self._intent.snapshot()
        return TaskPage(
            tasks=tuple(filtered),
            revision=max(1, snap.event_watermark),
        )

    readiness = ready_tasks

    def get_objective(self, objective_id: str) -> Mapping[str, Any] | None:
        return self._intent.get_objective(objective_id)

    def get_goal(self, goal_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_goal(goal_cid)

    def get_plan(self, plan_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_plan(plan_cid)

    # -- mutations -----------------------------------------------------------

    def compare_and_set_status(
        self,
        task_cid_or_alias: str | TaskRecord | Mapping[str, Any],
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
        *,
        expected_control_receipt: Mapping[str, Any] | None = None,
        evidence_digests: Sequence[str] | None = None,
    ) -> CASResult:
        key = _task_key(task_cid_or_alias)
        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
                    {
                        "task_cid_or_alias": key,
                        "expected_revision": expected_revision,
                        "status": status,
                        "receipt": dict(receipt) if receipt is not None else None,
                        "expected_control_receipt": (
                            dict(expected_control_receipt)
                            if expected_control_receipt is not None
                            else None
                        ),
                        "evidence_digests": (
                            list(evidence_digests) if evidence_digests is not None else None
                        ),
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _cas_result_from_dict(result)
        prior = self._intent.get_task(key)
        if prior is None:
            raise KeyError(key)
        previous_status = str(prior["status"])
        transition_receipt = receipt
        if (
            previous_status.strip().lower() == "blocked"
            and str(status or "").strip().lower() in _REOPENED_TASK_STATUSES
        ):
            body = prior.get("body")
            completion_receipt = (
                body.get("completion_receipt")
                if isinstance(body, Mapping)
                else None
            )
            if (
                isinstance(completion_receipt, Mapping)
                and completion_receipt.get("operation")
                == "database_portal_typed_deferral_budget_exhausted"
            ):
                try:
                    transition_receipt = admit_typed_deferral_budget_supersession(
                        task_cid=str(prior["task_cid"]),
                        task_alias=str(prior.get("task_alias") or ""),
                        task_revision=int(prior["revision"]),
                        task_body=body,
                        request=receipt,
                    )
                except TypedDeferralRecoveryError as exc:
                    raise TaskSourceConflictError(
                        "exhausted typed-deferral task remains blocked: "
                        + str(exc)
                    ) from exc
        try:
            intent_receipt = self._intent.cas_task_status(
                task_cid=str(prior["task_cid"]),
                expected_revision=int(expected_revision),
                new_status=status,
                receipt=transition_receipt,
                expected_control_receipt=expected_control_receipt,
                evidence_digests=evidence_digests,
            )
        except IntentCompletionError as exc:
            raise TaskSourceCompletionError(str(exc)) from exc
        except IntentRepositoryConflictError as exc:
            raise TaskSourceConflictError(str(exc)) from exc
        updated = self._intent.get_task(str(prior["task_cid"]))
        if updated is None:
            raise TaskSourceIntegrityError("task disappeared after CAS")
        task = _as_task_record(updated)
        receipt_cid = ""
        if intent_receipt.changed:
            details = dict(intent_receipt.details)
            receipt_cid = str(details.get("completion_receipt_cid") or "")
            if not receipt_cid and intent_receipt.event_id:
                receipt_cid = intent_receipt.event_id
        return CASResult(
            task=task,
            previous_status=previous_status,
            revision=int(intent_receipt.revision or task.revision),
            event_cursor=int(intent_receipt.global_sequence),
            changed=bool(intent_receipt.changed),
            receipt_cid=receipt_cid,
        )

    cas_status = compare_and_set_status

    def compare_and_set_goal_status(
        self,
        goal_cid_or_alias: str | Mapping[str, Any],
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        """CAS a goal after child tasks and child goals are complete."""

        if isinstance(goal_cid_or_alias, Mapping):
            key = str(
                goal_cid_or_alias.get("goal_cid")
                or goal_cid_or_alias.get("goal_alias")
                or ""
            ).strip()
        else:
            key = str(goal_cid_or_alias or "").strip()
        if not key:
            raise TaskSourceIntegrityError("goal CAS requires a goal CID or alias")
        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS,
                    {
                        "goal_cid_or_alias": key,
                        "expected_revision": expected_revision,
                        "status": status,
                        "receipt": dict(receipt) if receipt is not None else None,
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _intent_receipt_from_dict(result)
        try:
            return self._intent.cas_goal_status(
                goal_cid=key,
                expected_revision=int(expected_revision),
                new_status=status,
                receipt=receipt,
            )
        except IntentCompletionError as exc:
            raise TaskSourceCompletionError(str(exc)) from exc
        except IntentRepositoryConflictError as exc:
            raise TaskSourceConflictError(str(exc)) from exc

    def rearm_blocked_task(
        self,
        task_cid_or_alias: str | TaskRecord | Mapping[str, Any],
        *,
        receipt: Mapping[str, Any] | None = None,
    ) -> CASResult:
        """CAS a blocked task to retrying using the owner's current revision.

        Quack clients must not ATTACH only to read ``expected_revision``.
        The exclusive owner reads and mutates on its bound connection.
        """

        key = _task_key(task_cid_or_alias)
        compact = dict(receipt or {})
        compact.setdefault("operation", "database_declared_outputs_on_head_rearm")
        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
                    {
                        "task_cid_or_alias": key,
                        "receipt": compact,
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _cas_result_from_dict(result)
        record = self.get_task(key)
        if record is None:
            raise KeyError(key)
        status = str(record.status or "").strip().lower()
        if status == "retrying":
            return CASResult(
                task=record,
                previous_status="retrying",
                revision=int(record.revision),
                event_cursor=0,
                changed=False,
            )
        if status in _REARM_IMMUTABLE_TASK_STATUSES:
            # A completed repair receipt is durable history and can be replayed
            # by every lane after the task has subsequently reached a terminal
            # state.  Treat that replay as an idempotent no-op: terminal task
            # authority is immutable and must never be reopened by recovery.
            return CASResult(
                task=record,
                previous_status=status,
                revision=int(record.revision),
                event_cursor=0,
                changed=False,
            )
        if status != "blocked":
            raise TaskSourceConflictError(
                f"rearm requires blocked status, observed {status!r}"
            )
        return self.compare_and_set_status(
            record.task_cid,
            int(record.revision),
            "retrying",
            compact,
        )

    def recover_typed_deferral_budget(
        self,
        task_cid_or_alias: str | TaskRecord | Mapping[str, Any],
        *,
        repair_head: str,
        repair_tree: str,
        timeout_seconds: float = 660.0,
    ) -> CASResult:
        """Ask the fenced owner to run the real canary and rearm one block.

        Provider receipts are intentionally absent from this client API.  The
        credential-bearing state owner executes the inert quota/high canary,
        admits its fresh output in process, and performs the revision CAS.
        """

        key = _task_key(task_cid_or_alias)
        repair_head_text = str(repair_head or "")
        repair_tree_text = str(repair_tree or "")
        if (
            _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_head_text) is None
            or _TYPED_DEFERRAL_GIT_OBJECT_RE.fullmatch(repair_tree_text) is None
        ):
            raise TaskSourceBoundsError(
                "typed-deferral recovery requires an exact repair commit/tree"
            )
        if not self._intent.uses_quack_transport:
            raise TaskSourceIntegrityError(
                "typed-deferral provider recovery requires the fenced state owner"
            )
        try:
            result = submit_quack_owner_command(
                QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
                {
                    "task_cid_or_alias": key,
                    "repair_head": repair_head_text,
                    "repair_tree": repair_tree_text,
                },
                timeout_seconds=timeout_seconds,
            )
        except QuackOwnerCommandRemoteError as exc:
            _raise_typed_owner_error(exc)
        return _cas_result_from_dict(result)

    def record_queue_backoff(
        self,
        *,
        task_cid: str,
        delay_ms: int,
        reason: str = "backoff",
        selection_penalty: int = 0,
    ) -> IntentReceipt:
        """Persist a task-selection cooldown through the intent authority.

        Database-authoritative executors must not rely on an attempt-local
        JSON queue: those projections are disposable and a replacement
        attempt receives a different state directory.  This checked adapter
        keeps the existing :class:`IntentRepository` as the sole queue
        authority while avoiding private repository access by callers.
        """

        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
                    {
                        "task_cid": task_cid,
                        "delay_ms": delay_ms,
                        "reason": reason,
                        "selection_penalty": selection_penalty,
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _intent_receipt_from_dict(result)
        return self._intent.record_queue_backoff(
            task_cid=task_cid,
            delay_ms=delay_ms,
            reason=reason,
            selection_penalty=selection_penalty,
        )

    @staticmethod
    def _guarded_queue_status_result(
        payload: Mapping[str, Any],
        *,
        cas_result: CASResult,
    ) -> Mapping[str, Any]:
        expected = {
            "previous_status",
            "queue_receipt",
            "queue_reused",
            "retry_not_before_ms",
            "transition_receipt",
        }
        if set(payload) != expected:
            raise TaskSourceIntegrityError(
                "guarded queue/status result fields are malformed"
            )
        queue_receipt = payload.get("queue_receipt")
        transition_receipt = payload.get("transition_receipt")
        if (
            not isinstance(queue_receipt, Mapping)
            or not isinstance(transition_receipt, Mapping)
            or type(payload.get("queue_reused")) is not bool
            or type(payload.get("retry_not_before_ms")) is not int
            or int(payload["retry_not_before_ms"]) < 0
        ):
            raise TaskSourceIntegrityError(
                "guarded queue/status result values are malformed"
            )
        return MappingProxyType(
            {
                "previous_status": str(payload["previous_status"]),
                "queue_receipt": dict(queue_receipt),
                "queue_reused": bool(payload["queue_reused"]),
                "retry_not_before_ms": int(payload["retry_not_before_ms"]),
                "transition_receipt": dict(transition_receipt),
                "cas_result": cas_result,
            }
        )

    def record_queue_backoff_and_cas_status(
        self,
        *,
        task_cid: str,
        expected_revision: int,
        expected_control_receipt: Mapping[str, Any],
        status: str,
        receipt: Mapping[str, Any],
        delay_ms: int,
        reason: str,
        selection_penalty: int = 0,
        exact_retry_not_before_ms: int | None = None,
    ) -> Mapping[str, Any]:
        """Atomically guard, cool, and transition one shared control row."""

        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
                    {
                        "task_cid": task_cid,
                        "expected_revision": expected_revision,
                        "expected_control_receipt": dict(
                            expected_control_receipt
                        ),
                        "status": status,
                        "receipt": dict(receipt),
                        "delay_ms": delay_ms,
                        "reason": reason,
                        "selection_penalty": selection_penalty,
                        **(
                            {
                                "exact_retry_not_before_ms": (
                                    exact_retry_not_before_ms
                                )
                            }
                            if exact_retry_not_before_ms is not None
                            else {}
                        ),
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            if not isinstance(result, Mapping) or "cas_result" not in result:
                raise TaskSourceIntegrityError(
                    "guarded queue/status owner response is malformed"
                )
            result_map = dict(result)
            cas_payload = result_map.pop("cas_result")
            if not isinstance(cas_payload, Mapping):
                raise TaskSourceIntegrityError(
                    "guarded queue/status owner CAS response is malformed"
                )
            return self._guarded_queue_status_result(
                result_map,
                cas_result=_cas_result_from_dict(cas_payload),
            )

        prior = self._intent.get_task(task_cid)
        if prior is None:
            raise KeyError(task_cid)
        try:
            result = self._intent.record_queue_backoff_and_cas_task_status(
                task_cid=task_cid,
                expected_revision=expected_revision,
                expected_control_receipt=expected_control_receipt,
                new_status=status,
                receipt=receipt,
                delay_ms=delay_ms,
                reason=reason,
                selection_penalty=selection_penalty,
                exact_retry_not_before_ms=exact_retry_not_before_ms,
            )
        except IntentRepositoryConflictError as exc:
            raise TaskSourceConflictError(str(exc)) from exc
        result_map = dict(result)
        status_payload = result_map.pop("status_receipt", None)
        if not isinstance(status_payload, Mapping):
            raise TaskSourceIntegrityError(
                "guarded queue/status repository receipt is malformed"
            )
        status_receipt = _intent_receipt_from_dict(status_payload)
        updated = self._intent.get_task(str(prior["task_cid"]))
        if updated is None:
            raise TaskSourceIntegrityError(
                "task disappeared after guarded queue/status transition"
            )
        task = _as_task_record(updated)
        receipt_cid = status_receipt.event_id if status_receipt.changed else ""
        return self._guarded_queue_status_result(
            result_map,
            cas_result=CASResult(
                task=task,
                previous_status=str(result_map.get("previous_status") or ""),
                revision=int(status_receipt.revision or task.revision),
                event_cursor=int(status_receipt.global_sequence),
                changed=bool(status_receipt.changed),
                receipt_cid=receipt_cid,
            ),
        )

    def record_queue_retry(self, *, task_cid: str) -> IntentReceipt:
        """Clear one canonical task cooldown through the intent authority."""

        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
                    {"task_cid": task_cid},
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _intent_receipt_from_dict(result)
        return self._intent.record_queue_retry(task_cid=task_cid)

    def get_queue_entry(self, task_cid: str) -> QueueEntry | None:
        """Return the canonical queue state without exposing raw SQL."""

        return self._intent.get_queue_entry(task_cid)

    def record_evidence(
        self,
        *,
        task_cid: str,
        evidence_kind: str,
        digest: str,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
                    {
                        "task_cid": task_cid,
                        "evidence_kind": evidence_kind,
                        "digest": digest,
                        "body": dict(body) if body is not None else None,
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _intent_receipt_from_dict(result)
        return self._intent.record_evidence(
            task_cid=task_cid,
            evidence_kind=evidence_kind,
            digest=digest,
            body=body,
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
        if self._intent.uses_quack_transport:
            try:
                result = submit_quack_owner_command(
                    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
                    {
                        "task_cid": task_cid,
                        "outcome": outcome,
                        "evidence_digest": evidence_digest,
                        "argv": list(argv) if argv is not None else None,
                        "attempt_id": attempt_id,
                        "body": dict(body) if body is not None else None,
                    },
                )
            except QuackOwnerCommandRemoteError as exc:
                _raise_typed_owner_error(exc)
            return _intent_receipt_from_dict(result)
        return self._intent.record_validation_result(
            task_cid=task_cid,
            outcome=outcome,
            evidence_digest=evidence_digest,
            argv=argv,
            attempt_id=attempt_id,
            body=body,
        )

    def select_ready_tasks(self, *, limit: int = DEFAULT_QUERY_LIMIT) -> TaskPage:
        return self.ready_tasks(limit=limit)

    def rebuild_from_events(self) -> TaskSourceSnapshot:
        """Rebuild projections from admitted events and return a snapshot."""

        self._intent.rebuild_projections_from_events()
        return self.snapshot()

    def projection_matches_events(self) -> bool:
        """Return whether a rebuild yields the same projection CID."""

        before = self._intent.snapshot()
        after = self._intent.rebuild_projections_from_events()
        return before.projection_cid == after.projection_cid


__all__ = (
    "DATABASE_TASK_SOURCE_INTERFACE",
    "DATABASE_TASK_SOURCE_SCHEMA",
    "DatabaseTaskSource",
    "DatabaseTaskSourceError",
    "TaskSourceIntegrityError",
    "TaskSourceConflictError",
    "TaskSourceBoundsError",
    "TaskSourceCompletionError",
    "TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION",
    "TYPED_DEFERRAL_BUDGET_SUPERSESSION_OPERATION",
    "TYPED_DEFERRAL_BUDGET_SUPERSESSION_REQUEST_SCHEMA",
    "TYPED_DEFERRAL_BUDGET_SUPERSESSION_SCHEMA",
    "TypedDeferralRecoveryError",
    "admit_typed_deferral_budget_supersession",
    "build_typed_deferral_budget_supersession_request",
    "typed_deferral_budget_supersession_matches",
    "validate_typed_deferral_budget_supersession",
    "TaskRecord",
    "TaskPage",
    "CASResult",
    "TaskSourceSnapshot",
    "duckdb_available",
)
