"""DuckDB-backed self-improvement epochs, challengers, and rollouts (DQP-031).

Interface: ``ImprovementEpochRepository@1``

:class:`ImprovementEpochRepository` stores improvement epochs, stage
transitions, isolated challengers, rollout decisions, token metrics, and
receipts as transactional rows. Challengers use ordinary worktree, session,
and lease identities — not special identity classes — so they schedule
through the same coordination surface as production work.

Self-improvement work can be planned as ordinary goals and tasks in the
same database, enabling the control plane to treat improvement as first-class
intent rather than an out-of-band side channel.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE: Final[str] = "ImprovementEpochRepository@1"
IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-repository@1"
)
IMPROVEMENT_EPOCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch@1"
)
EPOCH_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-transition@1"
)
CHALLENGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-challenger@1"
)
ROLLOUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-rollout@1"
)
TOKEN_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-token-metrics@1"
)
EPOCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-receipt@1"
)
PLANNED_GOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-planned-goal@1"
)
PLANNED_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-planned-task@1"
)

MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_ID_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_CHALLENGERS_PER_EPOCH: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 50
MAX_PAGE_LIMIT: Final[int] = 500

# Ordinary identity kinds used by challengers (not special classes).
ORDINARY_IDENTITY_KINDS: Final[frozenset[str]] = frozenset(
    {
        "worktree",
        "session",
        "lease",
    }
)

ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS improvement_epoch_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS improvement_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    repository_tree VARCHAR NOT NULL,
    objective_revision VARCHAR NOT NULL DEFAULT '',
    policy_id VARCHAR NOT NULL DEFAULT '',
    capability_snapshot_id VARCHAR NOT NULL DEFAULT '',
    stage VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT NOT NULL DEFAULT 0,
    run_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_epochs_status_idx
    ON improvement_epochs(status, started_at_ms);
CREATE INDEX IF NOT EXISTS improvement_epochs_repo_idx
    ON improvement_epochs(repository_id, started_at_ms);

CREATE TABLE IF NOT EXISTS improvement_epoch_transitions (
    transition_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    from_stage VARCHAR NOT NULL,
    to_stage VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at_ms BIGINT NOT NULL,
    ordinal BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_epoch_transitions_epoch_idx
    ON improvement_epoch_transitions(epoch_id, ordinal);

CREATE TABLE IF NOT EXISTS improvement_challengers (
    challenger_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    isolation_mode VARCHAR NOT NULL DEFAULT 'ordinary_identities',
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS improvement_challengers_epoch_uidx
    ON improvement_challengers(epoch_id);
CREATE INDEX IF NOT EXISTS improvement_challengers_worktree_idx
    ON improvement_challengers(worktree_id, status);

CREATE TABLE IF NOT EXISTS improvement_rollouts (
    rollout_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    decision VARCHAR NOT NULL,
    baseline_receipt_id VARCHAR NOT NULL DEFAULT '',
    challenger_receipt_id VARCHAR NOT NULL DEFAULT '',
    decided_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_rollouts_epoch_idx
    ON improvement_rollouts(epoch_id, decided_at_ms);

CREATE TABLE IF NOT EXISTS improvement_token_metrics (
    metrics_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    provider_calls BIGINT NOT NULL DEFAULT 0,
    context_bytes BIGINT NOT NULL DEFAULT 0,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_token_metrics_epoch_idx
    ON improvement_token_metrics(epoch_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS improvement_epoch_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    receipt_kind VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_epoch_receipts_epoch_idx
    ON improvement_epoch_receipts(epoch_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS improvement_planned_goals (
    goal_cid VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_planned_goals_epoch_idx
    ON improvement_planned_goals(epoch_id, created_at_ms);

CREATE TABLE IF NOT EXISTS improvement_planned_tasks (
    task_cid VARCHAR PRIMARY KEY,
    goal_cid VARCHAR NOT NULL,
    epoch_id VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL DEFAULT 0,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS improvement_planned_tasks_goal_idx
    ON improvement_planned_tasks(goal_cid, ordinal);
CREATE INDEX IF NOT EXISTS improvement_planned_tasks_epoch_idx
    ON improvement_planned_tasks(epoch_id, created_at_ms);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ImprovementEpochError(RuntimeError):
    """Base fail-closed error for improvement-epoch repository operations."""

    code = "DQP_IMPROVEMENT_EPOCH_ERROR"


class ImprovementEpochNotOpenError(ImprovementEpochError):
    """Operation requires an open repository."""

    code = "DQP_IMPROVEMENT_EPOCH_NOT_OPEN"


class ImprovementEpochNotFoundError(ImprovementEpochError):
    """Epoch identity is absent."""

    code = "DQP_IMPROVEMENT_EPOCH_NOT_FOUND"


class ImprovementEpochConflictError(ImprovementEpochError):
    """Stage CAS, duplicate challenger, or identity conflict."""

    code = "DQP_IMPROVEMENT_EPOCH_CONFLICT"


class ImprovementEpochBoundsError(ImprovementEpochError, ValueError):
    """Payload or bound exceeded."""

    code = "DQP_IMPROVEMENT_EPOCH_BOUNDS"


class ImprovementChallengerIdentityError(ImprovementEpochError):
    """Challenger did not supply ordinary worktree/session/lease identities."""

    code = "DQP_CHALLENGER_IDENTITY"


class DuckDBUnavailableError(ImprovementEpochError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EpochStage(str, Enum):
    BASELINE = "baseline"
    PROPOSE = "propose"
    SHADOW = "shadow"
    EVALUATE = "evaluate"
    REJECT = "reject"
    RETAIN = "retain"
    CANARY = "canary"
    RECHECK = "recheck"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REFILL = "refill"
    STOP = "stop"


class EpochStatus(str, Enum):
    OPEN = "open"
    ACTIONABLE = "actionable"
    SUCCESSORS_CREATED = "successors_created"
    HEALTHY_EXHAUSTED = "healthy_exhausted"
    ROLLED_BACK = "rolled_back"
    STOPPED = "stopped"
    INELIGIBLE = "ineligible"


class ChallengerStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    EVALUATED = "evaluated"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class RolloutDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    RETAIN_BASELINE = "retain_baseline"
    ROLLBACK = "rollback"
    ABSTAIN = "abstain"


# Legal stage transitions (finite state machine).
LEGAL_STAGE_TRANSITIONS: Final[Mapping[EpochStage, frozenset[EpochStage]]] = (
    MappingProxyType(
        {
            EpochStage.BASELINE: frozenset(
                {EpochStage.PROPOSE, EpochStage.STOP, EpochStage.REFILL}
            ),
            EpochStage.PROPOSE: frozenset(
                {EpochStage.SHADOW, EpochStage.REJECT, EpochStage.STOP}
            ),
            EpochStage.SHADOW: frozenset(
                {EpochStage.EVALUATE, EpochStage.REJECT, EpochStage.STOP}
            ),
            EpochStage.EVALUATE: frozenset(
                {
                    EpochStage.REJECT,
                    EpochStage.RETAIN,
                    EpochStage.CANARY,
                    EpochStage.PROMOTE,
                    EpochStage.STOP,
                }
            ),
            EpochStage.REJECT: frozenset({EpochStage.STOP, EpochStage.REFILL}),
            EpochStage.RETAIN: frozenset({EpochStage.STOP, EpochStage.REFILL}),
            EpochStage.CANARY: frozenset(
                {EpochStage.RECHECK, EpochStage.ROLLBACK, EpochStage.STOP}
            ),
            EpochStage.RECHECK: frozenset(
                {EpochStage.PROMOTE, EpochStage.ROLLBACK, EpochStage.STOP}
            ),
            EpochStage.PROMOTE: frozenset({EpochStage.STOP, EpochStage.REFILL}),
            EpochStage.ROLLBACK: frozenset({EpochStage.STOP, EpochStage.REFILL}),
            EpochStage.REFILL: frozenset({EpochStage.STOP, EpochStage.BASELINE}),
            EpochStage.STOP: frozenset(),
        }
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_ID_BYTES) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise ImprovementEpochError(f"{name} contains NUL")
    if required and not text:
        raise ImprovementEpochError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise ImprovementEpochBoundsError(f"{name} exceeds {maximum} bytes")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ImprovementEpochBoundsError(f"{name} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ImprovementEpochBoundsError(
            f"{name} must be a non-negative integer"
        ) from exc
    if number < 0:
        raise ImprovementEpochBoundsError(f"{name} must be a non-negative integer")
    return number


def _positive_int(value: Any, name: str) -> int:
    number = _nonneg_int(value, name)
    if number < 1:
        raise ImprovementEpochBoundsError(f"{name} must be a positive integer")
    return number


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _digest_of(value: Any) -> str:
    return _sha256_hex(_canonical_json(value).encode("utf-8"))


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ImprovementEpochBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        keys = []
    if keys:
        return {str(key): row[key] for key in keys}
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _row_get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
        upper = name.upper()
        if upper in mapping and mapping[upper] is not None:
            return mapping[upper]
        lower = name.lower()
        if lower in mapping and mapping[lower] is not None:
            return mapping[lower]
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value is not None:
            return value
    return default


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _parse_stage(value: Any) -> EpochStage:
    text = _text(value, "stage")
    try:
        return EpochStage(text)
    except ValueError as exc:
        raise ImprovementEpochError(f"unknown epoch stage: {text}") from exc


def _parse_status(value: Any) -> EpochStatus:
    text = _text(value, "status")
    try:
        return EpochStatus(text)
    except ValueError as exc:
        raise ImprovementEpochError(f"unknown epoch status: {text}") from exc


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementEpoch:
    """One identity-bound self-improvement evaluation epoch."""

    SCHEMA: ClassVar[str] = IMPROVEMENT_EPOCH_SCHEMA
    INTERFACE: ClassVar[str] = IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE

    epoch_id: str
    repository_id: str
    repository_tree: str
    stage: EpochStage | str
    status: EpochStatus | str
    revision: int
    started_at_ms: int
    objective_revision: str = ""
    policy_id: str = ""
    capability_snapshot_id: str = ""
    finished_at_ms: int = 0
    run_id: str = ""
    worktree_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_id", _text(self.epoch_id, "epoch_id"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "repository_tree", _text(self.repository_tree, "repository_tree")
        )
        stage = self.stage if isinstance(self.stage, EpochStage) else _parse_stage(self.stage)
        status = (
            self.status
            if isinstance(self.status, EpochStatus)
            else _parse_status(self.status)
        )
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", _positive_int(int(self.revision), "revision"))
        object.__setattr__(
            self, "started_at_ms", _nonneg_int(int(self.started_at_ms), "started_at_ms")
        )
        object.__setattr__(
            self,
            "finished_at_ms",
            _nonneg_int(int(self.finished_at_ms), "finished_at_ms"),
        )
        object.__setattr__(
            self,
            "objective_revision",
            _text(self.objective_revision, "objective_revision", required=False),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=False)
        )
        object.__setattr__(
            self,
            "capability_snapshot_id",
            _text(
                self.capability_snapshot_id,
                "capability_snapshot_id",
                required=False,
            ),
        )
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", required=False))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "epoch_id": self.epoch_id,
            "repository_id": self.repository_id,
            "repository_tree": self.repository_tree,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "revision": int(self.revision),
            "started_at_ms": int(self.started_at_ms),
            "finished_at_ms": int(self.finished_at_ms),
            "run_id": self.run_id,
            "worktree_id": self.worktree_id,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class EpochTransition:
    """One recorded stage transition for an epoch."""

    SCHEMA: ClassVar[str] = EPOCH_TRANSITION_SCHEMA

    transition_id: str
    epoch_id: str
    from_stage: EpochStage | str
    to_stage: EpochStage | str
    recorded_at_ms: int
    ordinal: int
    reason: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_stage = (
            self.from_stage.value
            if isinstance(self.from_stage, EpochStage)
            else str(self.from_stage)
        )
        to_stage = (
            self.to_stage.value
            if isinstance(self.to_stage, EpochStage)
            else str(self.to_stage)
        )
        return {
            "schema": self.SCHEMA,
            "transition_id": self.transition_id,
            "epoch_id": self.epoch_id,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "reason": self.reason,
            "recorded_at_ms": int(self.recorded_at_ms),
            "ordinal": int(self.ordinal),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ChallengerRecord:
    """Isolated challenger bound to ordinary worktree/session/lease ids."""

    SCHEMA: ClassVar[str] = CHALLENGER_SCHEMA

    challenger_id: str
    epoch_id: str
    worktree_id: str
    session_id: str
    lease_id: str
    status: ChallengerStatus | str
    created_at_ms: int
    isolation_mode: str = "ordinary_identities"
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "challenger_id", _text(self.challenger_id, "challenger_id")
        )
        object.__setattr__(self, "epoch_id", _text(self.epoch_id, "epoch_id"))
        object.__setattr__(self, "worktree_id", _text(self.worktree_id, "worktree_id"))
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        status = (
            self.status
            if isinstance(self.status, ChallengerStatus)
            else ChallengerStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "created_at_ms", _nonneg_int(int(self.created_at_ms), "created_at_ms")
        )
        object.__setattr__(
            self,
            "isolation_mode",
            _text(self.isolation_mode, "isolation_mode"),
        )
        if self.isolation_mode != "ordinary_identities":
            raise ImprovementChallengerIdentityError(
                "challenger must use ordinary worktree/session/lease identities"
            )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "challenger_id": self.challenger_id,
            "epoch_id": self.epoch_id,
            "worktree_id": self.worktree_id,
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "status": self.status.value,
            "isolation_mode": self.isolation_mode,
            "created_at_ms": int(self.created_at_ms),
            "identity_kinds": sorted(ORDINARY_IDENTITY_KINDS),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class RolloutRecord:
    """Baseline/challenger rollout decision for one epoch."""

    SCHEMA: ClassVar[str] = ROLLOUT_SCHEMA

    rollout_id: str
    epoch_id: str
    decision: RolloutDecision | str
    decided_at_ms: int
    baseline_receipt_id: str = ""
    challenger_receipt_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        decision = (
            self.decision.value
            if isinstance(self.decision, RolloutDecision)
            else str(self.decision)
        )
        return {
            "schema": self.SCHEMA,
            "rollout_id": self.rollout_id,
            "epoch_id": self.epoch_id,
            "decision": decision,
            "baseline_receipt_id": self.baseline_receipt_id,
            "challenger_receipt_id": self.challenger_receipt_id,
            "decided_at_ms": int(self.decided_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class TokenMetrics:
    """Token/economy metrics for one epoch observation."""

    SCHEMA: ClassVar[str] = TOKEN_METRICS_SCHEMA

    metrics_id: str
    epoch_id: str
    input_tokens: int
    output_tokens: int
    provider_calls: int
    context_bytes: int
    recorded_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "metrics_id": self.metrics_id,
            "epoch_id": self.epoch_id,
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "provider_calls": int(self.provider_calls),
            "context_bytes": int(self.context_bytes),
            "recorded_at_ms": int(self.recorded_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class EpochReceipt:
    """Content-addressed epoch receipt."""

    SCHEMA: ClassVar[str] = EPOCH_RECEIPT_SCHEMA

    receipt_id: str
    epoch_id: str
    receipt_kind: str
    digest: str
    recorded_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "receipt_id": self.receipt_id,
            "epoch_id": self.epoch_id,
            "receipt_kind": self.receipt_kind,
            "digest": self.digest,
            "recorded_at_ms": int(self.recorded_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class PlannedGoal:
    """Self-improvement goal planned in the same database."""

    SCHEMA: ClassVar[str] = PLANNED_GOAL_SCHEMA

    goal_cid: str
    epoch_id: str
    title: str
    status: str
    created_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "goal_cid": self.goal_cid,
            "epoch_id": self.epoch_id,
            "title": self.title,
            "status": self.status,
            "created_at_ms": int(self.created_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class PlannedTask:
    """Self-improvement task planned under a goal in the same database."""

    SCHEMA: ClassVar[str] = PLANNED_TASK_SCHEMA

    task_cid: str
    goal_cid: str
    epoch_id: str
    title: str
    status: str
    ordinal: int
    created_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "goal_cid": self.goal_cid,
            "epoch_id": self.epoch_id,
            "title": self.title,
            "status": self.status,
            "ordinal": int(self.ordinal),
            "created_at_ms": int(self.created_at_ms),
            "body": dict(self.body),
        }


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ImprovementEpochRepository:
    """DuckDB authority for improvement epochs, challengers, and planning."""

    INTERFACE: ClassVar[str] = IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for ImprovementEpochRepository; install "
                "the optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "ImprovementEpochRepository":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE),
                    ("schema", IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA),
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO improvement_epoch_metadata(key, value)
                        VALUES (?, ?)
                        """,
                        [key, value],
                    )
                self._connection = connection
                self._closed = False
                self._commit_if_idle(connection)
                return self
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "ImprovementEpochRepository":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def authority_policy(self) -> dict[str, Any]:
        return {
            "semantic_authority": "database",
            "challenger_identity": "ordinary_worktree_session_lease",
            "special_challenger_identity_classes": "none",
            "self_improvement_planning": "goals_and_tasks_in_same_database",
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "max_challengers_per_epoch": MAX_CHALLENGERS_PER_EPOCH,
        }

    # -- connection helpers --------------------------------------------------

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise ImprovementEpochNotOpenError(
                "ImprovementEpochRepository is not open"
            )
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                commit = getattr(connection, "commit", None)
                if callable(commit):
                    commit()
                    return
            raw = getattr(connection, "_connection", None)
            raw_commit = getattr(raw, "commit", None) if raw is not None else None
            if callable(raw_commit):
                raw_commit()
                return
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        except Exception:
            pass

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback) and getattr(connection, "in_transaction", False):
                rollback()
                return
            raw = getattr(connection, "_connection", None)
            raw_rollback = getattr(raw, "rollback", None) if raw is not None else None
            if callable(raw_rollback):
                raw_rollback()
        except Exception:
            pass

    # -- public API ----------------------------------------------------------

    def create_epoch(
        self,
        *,
        repository_id: str,
        repository_tree: str,
        objective_revision: str = "",
        policy_id: str = "",
        capability_snapshot_id: str = "",
        run_id: str = "",
        worktree_id: str = "",
        epoch_id: str | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> ImprovementEpoch:
        """Create an open epoch at the BASELINE stage."""

        now_ms = int(self._clock_ms())
        eid = _text(epoch_id or _new_id("epoch"), "epoch_id")
        repo = _text(repository_id, "repository_id", maximum=MAX_TEXT_BYTES)
        tree = _text(repository_tree, "repository_tree", maximum=MAX_TEXT_BYTES)
        payload = _bounded_mapping(body, name="body")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if existing is not None:
                    raise ImprovementEpochConflictError(
                        f"epoch already exists: {eid}"
                    )
                connection.execute(
                    """
                    INSERT INTO improvement_epochs (
                        epoch_id, repository_id, repository_tree,
                        objective_revision, policy_id, capability_snapshot_id,
                        stage, status, revision, started_at_ms, finished_at_ms,
                        run_id, worktree_id, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        eid,
                        repo,
                        tree,
                        _text(objective_revision, "objective_revision", required=False),
                        _text(policy_id, "policy_id", required=False),
                        _text(
                            capability_snapshot_id,
                            "capability_snapshot_id",
                            required=False,
                        ),
                        EpochStage.BASELINE.value,
                        EpochStatus.OPEN.value,
                        1,
                        now_ms,
                        0,
                        _text(run_id, "run_id", required=False),
                        _text(worktree_id, "worktree_id", required=False),
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO improvement_epoch_transitions (
                        transition_id, epoch_id, from_stage, to_stage, reason,
                        recorded_at_ms, ordinal, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        _new_id("transition"),
                        eid,
                        EpochStage.BASELINE.value,
                        EpochStage.BASELINE.value,
                        "epoch_created",
                        now_ms,
                        0,
                        _canonical_json({"schema": EPOCH_TRANSITION_SCHEMA}),
                    ],
                )
                self._commit_if_idle(connection)
                return ImprovementEpoch(
                    epoch_id=eid,
                    repository_id=repo,
                    repository_tree=tree,
                    objective_revision=str(objective_revision or ""),
                    policy_id=str(policy_id or ""),
                    capability_snapshot_id=str(capability_snapshot_id or ""),
                    stage=EpochStage.BASELINE,
                    status=EpochStatus.OPEN,
                    revision=1,
                    started_at_ms=now_ms,
                    finished_at_ms=0,
                    run_id=str(run_id or ""),
                    worktree_id=str(worktree_id or ""),
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_epoch(self, epoch_id: str) -> ImprovementEpoch:
        eid = _text(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM improvement_epochs WHERE epoch_id = ?",
                [eid],
            ).fetchone()
            if row is None:
                raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
            return self._row_to_epoch(_row_mapping(row))

    def transition(
        self,
        epoch_id: str,
        *,
        to_stage: EpochStage | str,
        expected_revision: int,
        reason: str = "",
        status: EpochStatus | str | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> tuple[ImprovementEpoch, EpochTransition]:
        """Advance epoch stage under revision CAS."""

        eid = _text(epoch_id, "epoch_id")
        target = (
            to_stage if isinstance(to_stage, EpochStage) else _parse_stage(to_stage)
        )
        expected = _positive_int(int(expected_revision), "expected_revision")
        now_ms = int(self._clock_ms())
        payload = _bounded_mapping(body, name="body")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                row = connection.execute(
                    "SELECT * FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
                current = self._row_to_epoch(_row_mapping(row))
                if int(current.revision) != expected:
                    raise ImprovementEpochConflictError(
                        "epoch revision CAS conflict"
                    )
                allowed = LEGAL_STAGE_TRANSITIONS.get(current.stage, frozenset())
                if target not in allowed:
                    raise ImprovementEpochConflictError(
                        f"illegal stage transition {current.stage.value} -> {target.value}"
                    )
                new_status = current.status
                if status is not None:
                    new_status = (
                        status
                        if isinstance(status, EpochStatus)
                        else _parse_status(status)
                    )
                elif target is EpochStage.STOP:
                    new_status = EpochStatus.STOPPED
                elif target is EpochStage.ROLLBACK:
                    new_status = EpochStatus.ROLLED_BACK
                elif target is EpochStage.REFILL:
                    new_status = EpochStatus.SUCCESSORS_CREATED
                elif target is EpochStage.PROMOTE:
                    new_status = EpochStatus.ACTIONABLE

                finished = (
                    now_ms
                    if target in {EpochStage.STOP, EpochStage.ROLLBACK}
                    else int(current.finished_at_ms)
                )
                new_revision = expected + 1
                connection.execute(
                    """
                    UPDATE improvement_epochs
                    SET stage = ?, status = ?, revision = ?, finished_at_ms = ?
                    WHERE epoch_id = ? AND revision = ?
                    """,
                    [
                        target.value,
                        new_status.value,
                        new_revision,
                        finished,
                        eid,
                        expected,
                    ],
                )
                ordinal_row = connection.execute(
                    """
                    SELECT COALESCE(MAX(ordinal), -1) AS max_ord
                    FROM improvement_epoch_transitions
                    WHERE epoch_id = ?
                    """,
                    [eid],
                ).fetchone()
                ordinal_mapping = _row_mapping(ordinal_row)
                ordinal_raw = _row_get(
                    ordinal_mapping, "max_ord", "0", default=None
                )
                if ordinal_raw is None and ordinal_mapping:
                    ordinal_raw = next(iter(ordinal_mapping.values()))
                ordinal = int(ordinal_raw if ordinal_raw is not None else -1) + 1
                transition_id = _new_id("transition")
                connection.execute(
                    """
                    INSERT INTO improvement_epoch_transitions (
                        transition_id, epoch_id, from_stage, to_stage, reason,
                        recorded_at_ms, ordinal, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        transition_id,
                        eid,
                        current.stage.value,
                        target.value,
                        _text(reason, "reason", required=False, maximum=MAX_TEXT_BYTES),
                        now_ms,
                        ordinal,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                epoch = ImprovementEpoch(
                    epoch_id=current.epoch_id,
                    repository_id=current.repository_id,
                    repository_tree=current.repository_tree,
                    objective_revision=current.objective_revision,
                    policy_id=current.policy_id,
                    capability_snapshot_id=current.capability_snapshot_id,
                    stage=target,
                    status=new_status,
                    revision=new_revision,
                    started_at_ms=current.started_at_ms,
                    finished_at_ms=finished,
                    run_id=current.run_id,
                    worktree_id=current.worktree_id,
                    body=dict(current.body),
                )
                transition = EpochTransition(
                    transition_id=transition_id,
                    epoch_id=eid,
                    from_stage=current.stage,
                    to_stage=target,
                    reason=str(reason or ""),
                    recorded_at_ms=now_ms,
                    ordinal=ordinal,
                    body=payload,
                )
                return epoch, transition
            except Exception:
                self._rollback_if_open(connection)
                raise

    def rollback_epoch(
        self,
        epoch_id: str,
        *,
        expected_revision: int,
        reason: str = "operator_rollback",
    ) -> tuple[ImprovementEpoch, EpochTransition]:
        """Transition epoch to ROLLBACK under revision CAS."""

        return self.transition(
            epoch_id,
            to_stage=EpochStage.ROLLBACK,
            expected_revision=expected_revision,
            reason=reason,
            status=EpochStatus.ROLLED_BACK,
        )

    def register_challenger(
        self,
        epoch_id: str,
        *,
        worktree_id: str,
        session_id: str,
        lease_id: str,
        challenger_id: str | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> ChallengerRecord:
        """Register exactly one isolated challenger with ordinary identities.

        Challengers intentionally reuse ordinary worktree/session/lease
        identities rather than introducing a special identity class.
        """

        eid = _text(epoch_id, "epoch_id")
        wt = _text(worktree_id, "worktree_id")
        session = _text(session_id, "session_id")
        lease = _text(lease_id, "lease_id")
        # Reject special identity prefixes that would bypass ordinary coordination.
        for name, value in (
            ("worktree_id", wt),
            ("session_id", session),
            ("lease_id", lease),
        ):
            lowered = value.casefold()
            if lowered.startswith("special:") or lowered.startswith("privileged:"):
                raise ImprovementChallengerIdentityError(
                    f"{name} must be an ordinary identity, not a special class"
                )
        now_ms = int(self._clock_ms())
        cid = _text(challenger_id or _new_id("challenger"), "challenger_id")
        payload = _bounded_mapping(body, name="body")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                epoch_row = connection.execute(
                    "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if epoch_row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
                existing = connection.execute(
                    """
                    SELECT challenger_id FROM improvement_challengers
                    WHERE epoch_id = ?
                    """,
                    [eid],
                ).fetchone()
                if existing is not None:
                    raise ImprovementEpochConflictError(
                        f"epoch already has an isolated challenger: {eid}"
                    )
                connection.execute(
                    """
                    INSERT INTO improvement_challengers (
                        challenger_id, epoch_id, worktree_id, session_id,
                        lease_id, status, isolation_mode, created_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cid,
                        eid,
                        wt,
                        session,
                        lease,
                        ChallengerStatus.REGISTERED.value,
                        "ordinary_identities",
                        now_ms,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                return ChallengerRecord(
                    challenger_id=cid,
                    epoch_id=eid,
                    worktree_id=wt,
                    session_id=session,
                    lease_id=lease,
                    status=ChallengerStatus.REGISTERED,
                    created_at_ms=now_ms,
                    isolation_mode="ordinary_identities",
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_challenger(self, epoch_id: str) -> ChallengerRecord | None:
        eid = _text(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT * FROM improvement_challengers WHERE epoch_id = ?
                """,
                [eid],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_challenger(_row_mapping(row))

    def record_rollout(
        self,
        epoch_id: str,
        *,
        decision: RolloutDecision | str,
        baseline_receipt_id: str = "",
        challenger_receipt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RolloutRecord:
        eid = _text(epoch_id, "epoch_id")
        dec = (
            decision
            if isinstance(decision, RolloutDecision)
            else RolloutDecision(str(decision))
        )
        now_ms = int(self._clock_ms())
        payload = _bounded_mapping(body, name="body")
        rollout_id = _new_id("rollout")

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                epoch_row = connection.execute(
                    "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if epoch_row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
                connection.execute(
                    """
                    INSERT INTO improvement_rollouts (
                        rollout_id, epoch_id, decision, baseline_receipt_id,
                        challenger_receipt_id, decided_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        rollout_id,
                        eid,
                        dec.value,
                        _text(
                            baseline_receipt_id,
                            "baseline_receipt_id",
                            required=False,
                        ),
                        _text(
                            challenger_receipt_id,
                            "challenger_receipt_id",
                            required=False,
                        ),
                        now_ms,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                return RolloutRecord(
                    rollout_id=rollout_id,
                    epoch_id=eid,
                    decision=dec,
                    decided_at_ms=now_ms,
                    baseline_receipt_id=str(baseline_receipt_id or ""),
                    challenger_receipt_id=str(challenger_receipt_id or ""),
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def record_token_metrics(
        self,
        epoch_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_calls: int = 0,
        context_bytes: int = 0,
        body: Mapping[str, Any] | None = None,
    ) -> TokenMetrics:
        eid = _text(epoch_id, "epoch_id")
        now_ms = int(self._clock_ms())
        metrics_id = _new_id("metrics")
        payload = _bounded_mapping(body, name="body")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                epoch_row = connection.execute(
                    "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if epoch_row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
                connection.execute(
                    """
                    INSERT INTO improvement_token_metrics (
                        metrics_id, epoch_id, input_tokens, output_tokens,
                        provider_calls, context_bytes, recorded_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        metrics_id,
                        eid,
                        _nonneg_int(int(input_tokens), "input_tokens"),
                        _nonneg_int(int(output_tokens), "output_tokens"),
                        _nonneg_int(int(provider_calls), "provider_calls"),
                        _nonneg_int(int(context_bytes), "context_bytes"),
                        now_ms,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                return TokenMetrics(
                    metrics_id=metrics_id,
                    epoch_id=eid,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    provider_calls=int(provider_calls),
                    context_bytes=int(context_bytes),
                    recorded_at_ms=now_ms,
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def record_receipt(
        self,
        epoch_id: str,
        *,
        receipt_kind: str,
        body: Mapping[str, Any] | None = None,
        receipt_id: str | None = None,
    ) -> EpochReceipt:
        eid = _text(epoch_id, "epoch_id")
        kind = _text(receipt_kind, "receipt_kind")
        payload = _bounded_mapping(body, name="body")
        digest = _digest_of(
            {"epoch_id": eid, "receipt_kind": kind, "body": payload}
        )
        rid = _text(receipt_id or f"receipt:{digest[7:39]}", "receipt_id")
        now_ms = int(self._clock_ms())
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                epoch_row = connection.execute(
                    "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if epoch_row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")
                connection.execute(
                    """
                    INSERT INTO improvement_epoch_receipts (
                        receipt_id, epoch_id, receipt_kind, digest,
                        recorded_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        rid,
                        eid,
                        kind,
                        digest,
                        now_ms,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                return EpochReceipt(
                    receipt_id=rid,
                    epoch_id=eid,
                    receipt_kind=kind,
                    digest=digest,
                    recorded_at_ms=now_ms,
                    body=payload,
                )
            except Exception:
                self._rollback_if_open(connection)
                raise

    def plan_as_goals_and_tasks(
        self,
        epoch_id: str,
        *,
        goals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Plan self-improvement work as goals/tasks in the same database.

        Each goal mapping may include ``title``, optional ``goal_cid``, and
        ``tasks`` (sequence of mappings with ``title`` / optional ``task_cid``).
        """

        eid = _text(epoch_id, "epoch_id")
        if not goals:
            raise ImprovementEpochBoundsError("goals must not be empty")
        now_ms = int(self._clock_ms())
        created_goals: list[PlannedGoal] = []
        created_tasks: list[PlannedTask] = []

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                epoch_row = connection.execute(
                    "SELECT epoch_id, status FROM improvement_epochs WHERE epoch_id = ?",
                    [eid],
                ).fetchone()
                if epoch_row is None:
                    raise ImprovementEpochNotFoundError(f"epoch not found: {eid}")

                for index, raw_goal in enumerate(goals):
                    if not isinstance(raw_goal, Mapping):
                        raise ImprovementEpochBoundsError(
                            f"goals[{index}] must be a mapping"
                        )
                    title = _text(
                        raw_goal.get("title") or f"improvement-goal-{index}",
                        "goal.title",
                        maximum=MAX_TEXT_BYTES,
                    )
                    goal_cid = _text(
                        raw_goal.get("goal_cid") or _new_id("goal"),
                        "goal_cid",
                    )
                    goal_body = _bounded_mapping(
                        {
                            k: v
                            for k, v in dict(raw_goal).items()
                            if k not in {"title", "goal_cid", "tasks", "status"}
                        },
                        name="goal.body",
                    )
                    goal_status = _text(
                        raw_goal.get("status") or "open",
                        "goal.status",
                        required=False,
                    ) or "open"
                    connection.execute(
                        """
                        INSERT INTO improvement_planned_goals (
                            goal_cid, epoch_id, title, status, created_at_ms, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            goal_cid,
                            eid,
                            title,
                            goal_status,
                            now_ms,
                            _canonical_json(goal_body),
                        ],
                    )
                    created_goals.append(
                        PlannedGoal(
                            goal_cid=goal_cid,
                            epoch_id=eid,
                            title=title,
                            status=goal_status,
                            created_at_ms=now_ms,
                            body=goal_body,
                        )
                    )
                    tasks = raw_goal.get("tasks") or ()
                    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
                        raise ImprovementEpochBoundsError(
                            f"goals[{index}].tasks must be a sequence"
                        )
                    for task_index, raw_task in enumerate(tasks):
                        if not isinstance(raw_task, Mapping):
                            raise ImprovementEpochBoundsError(
                                f"goals[{index}].tasks[{task_index}] must be a mapping"
                            )
                        task_title = _text(
                            raw_task.get("title")
                            or f"improvement-task-{index}-{task_index}",
                            "task.title",
                            maximum=MAX_TEXT_BYTES,
                        )
                        task_cid = _text(
                            raw_task.get("task_cid") or _new_id("task"),
                            "task_cid",
                        )
                        task_status = _text(
                            raw_task.get("status") or "ready",
                            "task.status",
                            required=False,
                        ) or "ready"
                        task_body = _bounded_mapping(
                            {
                                k: v
                                for k, v in dict(raw_task).items()
                                if k not in {"title", "task_cid", "status", "ordinal"}
                            },
                            name="task.body",
                        )
                        ordinal = _nonneg_int(
                            int(raw_task.get("ordinal", task_index)),
                            "task.ordinal",
                        )
                        connection.execute(
                            """
                            INSERT INTO improvement_planned_tasks (
                                task_cid, goal_cid, epoch_id, title, status,
                                ordinal, created_at_ms, body_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                task_cid,
                                goal_cid,
                                eid,
                                task_title,
                                task_status,
                                ordinal,
                                now_ms,
                                _canonical_json(task_body),
                            ],
                        )
                        created_tasks.append(
                            PlannedTask(
                                task_cid=task_cid,
                                goal_cid=goal_cid,
                                epoch_id=eid,
                                title=task_title,
                                status=task_status,
                                ordinal=ordinal,
                                created_at_ms=now_ms,
                                body=task_body,
                            )
                        )

                # Mark epoch as successors_created when planning succeeds.
                connection.execute(
                    """
                    UPDATE improvement_epochs
                    SET status = ?
                    WHERE epoch_id = ?
                    """,
                    [EpochStatus.SUCCESSORS_CREATED.value, eid],
                )
                self._commit_if_idle(connection)
                return {
                    "epoch_id": eid,
                    "goals": [item.to_dict() for item in created_goals],
                    "tasks": [item.to_dict() for item in created_tasks],
                    "goal_count": len(created_goals),
                    "task_count": len(created_tasks),
                    "same_database": True,
                    "database_path": str(self._path),
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def list_planned_goals(self, epoch_id: str) -> tuple[PlannedGoal, ...]:
        eid = _text(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM improvement_planned_goals
                WHERE epoch_id = ?
                ORDER BY created_at_ms ASC, goal_cid ASC
                """,
                [eid],
            ).fetchall()
            return tuple(self._row_to_goal(_row_mapping(row)) for row in rows)

    def list_planned_tasks(self, epoch_id: str) -> tuple[PlannedTask, ...]:
        eid = _text(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM improvement_planned_tasks
                WHERE epoch_id = ?
                ORDER BY ordinal ASC, task_cid ASC
                """,
                [eid],
            ).fetchall()
            return tuple(self._row_to_task(_row_mapping(row)) for row in rows)

    def list_transitions(self, epoch_id: str) -> tuple[EpochTransition, ...]:
        eid = _text(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM improvement_epoch_transitions
                WHERE epoch_id = ?
                ORDER BY ordinal ASC
                """,
                [eid],
            ).fetchall()
            return tuple(self._row_to_transition(_row_mapping(row)) for row in rows)

    def list_epochs(
        self,
        *,
        repository_id: str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        cursor: str = "",
    ) -> dict[str, Any]:
        page_limit = int(limit)
        if page_limit < 1 or page_limit > MAX_PAGE_LIMIT:
            raise ImprovementEpochBoundsError(
                f"limit must be in 1..{MAX_PAGE_LIMIT}"
            )
        offset = 0
        if cursor:
            try:
                offset = max(0, int(cursor))
            except (TypeError, ValueError) as exc:
                raise ImprovementEpochBoundsError(
                    "cursor must be a non-negative integer offset"
                ) from exc
        with self._lock:
            connection = self._require()
            if repository_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM improvement_epochs
                    WHERE repository_id = ?
                    ORDER BY started_at_ms ASC, epoch_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [
                        _text(repository_id, "repository_id", maximum=MAX_TEXT_BYTES),
                        page_limit + 1,
                        offset,
                    ],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM improvement_epochs
                    ORDER BY started_at_ms ASC, epoch_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [page_limit + 1, offset],
                ).fetchall()
            items = [self._row_to_epoch(_row_mapping(row)) for row in rows[:page_limit]]
            has_more = len(rows) > page_limit
            return {
                "items": [item.to_dict() for item in items],
                "next_cursor": str(offset + page_limit) if has_more else "",
                "has_more": has_more,
            }

    # -- row mappers ---------------------------------------------------------

    @staticmethod
    def _row_to_epoch(mapping: Mapping[str, Any]) -> ImprovementEpoch:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return ImprovementEpoch(
            epoch_id=str(_row_get(mapping, "epoch_id", default="") or ""),
            repository_id=str(_row_get(mapping, "repository_id", default="") or ""),
            repository_tree=str(
                _row_get(mapping, "repository_tree", default="") or ""
            ),
            objective_revision=str(
                _row_get(mapping, "objective_revision", default="") or ""
            ),
            policy_id=str(_row_get(mapping, "policy_id", default="") or ""),
            capability_snapshot_id=str(
                _row_get(mapping, "capability_snapshot_id", default="") or ""
            ),
            stage=str(_row_get(mapping, "stage", default="") or ""),
            status=str(_row_get(mapping, "status", default="") or ""),
            revision=int(_row_get(mapping, "revision", default=1)),
            started_at_ms=int(_row_get(mapping, "started_at_ms", default=0)),
            finished_at_ms=int(_row_get(mapping, "finished_at_ms", default=0)),
            run_id=str(_row_get(mapping, "run_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            body=body,
        )

    @staticmethod
    def _row_to_transition(mapping: Mapping[str, Any]) -> EpochTransition:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return EpochTransition(
            transition_id=str(_row_get(mapping, "transition_id", default="") or ""),
            epoch_id=str(_row_get(mapping, "epoch_id", default="") or ""),
            from_stage=str(_row_get(mapping, "from_stage", default="") or ""),
            to_stage=str(_row_get(mapping, "to_stage", default="") or ""),
            reason=str(_row_get(mapping, "reason", default="") or ""),
            recorded_at_ms=int(_row_get(mapping, "recorded_at_ms", default=0)),
            ordinal=int(_row_get(mapping, "ordinal", default=0)),
            body=body,
        )

    @staticmethod
    def _row_to_challenger(mapping: Mapping[str, Any]) -> ChallengerRecord:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return ChallengerRecord(
            challenger_id=str(_row_get(mapping, "challenger_id", default="") or ""),
            epoch_id=str(_row_get(mapping, "epoch_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            session_id=str(_row_get(mapping, "session_id", default="") or ""),
            lease_id=str(_row_get(mapping, "lease_id", default="") or ""),
            status=str(_row_get(mapping, "status", default="") or ""),
            isolation_mode=str(
                _row_get(mapping, "isolation_mode", default="ordinary_identities")
                or "ordinary_identities"
            ),
            created_at_ms=int(_row_get(mapping, "created_at_ms", default=0)),
            body=body,
        )

    @staticmethod
    def _row_to_goal(mapping: Mapping[str, Any]) -> PlannedGoal:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return PlannedGoal(
            goal_cid=str(_row_get(mapping, "goal_cid", default="") or ""),
            epoch_id=str(_row_get(mapping, "epoch_id", default="") or ""),
            title=str(_row_get(mapping, "title", default="") or ""),
            status=str(_row_get(mapping, "status", default="") or ""),
            created_at_ms=int(_row_get(mapping, "created_at_ms", default=0)),
            body=body,
        )

    @staticmethod
    def _row_to_task(mapping: Mapping[str, Any]) -> PlannedTask:
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(str(body_raw or "{}"))
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return PlannedTask(
            task_cid=str(_row_get(mapping, "task_cid", default="") or ""),
            goal_cid=str(_row_get(mapping, "goal_cid", default="") or ""),
            epoch_id=str(_row_get(mapping, "epoch_id", default="") or ""),
            title=str(_row_get(mapping, "title", default="") or ""),
            status=str(_row_get(mapping, "status", default="") or ""),
            ordinal=int(_row_get(mapping, "ordinal", default=0)),
            created_at_ms=int(_row_get(mapping, "created_at_ms", default=0)),
            body=body,
        )


def open_improvement_epoch_repository(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
) -> ImprovementEpochRepository:
    """Open and return an initialized :class:`ImprovementEpochRepository`."""

    return ImprovementEpochRepository(
        database_path,
        clock_ms=clock_ms,
    ).open()


__all__ = (
    "CHALLENGER_SCHEMA",
    "DEFAULT_PAGE_LIMIT",
    "EPOCH_RECEIPT_SCHEMA",
    "EPOCH_TRANSITION_SCHEMA",
    "IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE",
    "IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA",
    "IMPROVEMENT_EPOCH_SCHEMA",
    "LEGAL_STAGE_TRANSITIONS",
    "MAX_CHALLENGERS_PER_EPOCH",
    "MAX_PAGE_LIMIT",
    "ORDINARY_IDENTITY_KINDS",
    "PLANNED_GOAL_SCHEMA",
    "PLANNED_TASK_SCHEMA",
    "ROLLOUT_SCHEMA",
    "TOKEN_METRICS_SCHEMA",
    "ChallengerRecord",
    "ChallengerStatus",
    "DuckDBUnavailableError",
    "EpochReceipt",
    "EpochStage",
    "EpochStatus",
    "EpochTransition",
    "ImprovementChallengerIdentityError",
    "ImprovementEpoch",
    "ImprovementEpochBoundsError",
    "ImprovementEpochConflictError",
    "ImprovementEpochError",
    "ImprovementEpochNotFoundError",
    "ImprovementEpochNotOpenError",
    "ImprovementEpochRepository",
    "PlannedGoal",
    "PlannedTask",
    "RolloutDecision",
    "RolloutRecord",
    "TokenMetrics",
    "duckdb_available",
    "open_improvement_epoch_repository",
)
