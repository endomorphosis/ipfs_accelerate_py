"""Durable, dependency-aware execution of formal proof-plan DAGs.

The scheduler has four deliberately small trust and coordination boundaries:

* :class:`ProofPlan` remains the immutable source of graph structure;
* lock-aware DuckDB transactions provide process leases and fencing tokens;
* provider work is cooperative-cancellable and never trusted merely because
  its callback returned successfully; and
* conclusive receipts are deduplicated by ``(plan_id, obligation_id)`` before
  they become the authoritative portfolio outcome.

Callbacks may run translation, model drafting, solving, reconstruction, kernel
verification, validation, attestation, or artifact persistence.  Ready nodes
from all stages share the bounded pool, so independent work overlaps without
weakening dependency order.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    initialize_duckdb_database,
    open_duckdb_connection,
    resolve_duckdb_path,
)
from .formal_verification_contracts import (
    AttemptStatus,
    ContractValidationError,
    ProofAttempt,
    ProofPlan,
    ProofPlanStep,
    ProofReceipt,
    ProofStage,
    canonical_json,
)
from .formal_verification_provider import (
    CancellationToken,
    ProofProviderError,
    ProviderFailureCode,
    ProviderResponse,
)
from .resource_scheduler import (
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProofResourceClass,
    ProviderCapacity,
    ResourceAdmissionLease,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
    normalize_resource_class,
    resource_pool,
)


PROOF_SCHEDULER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-scheduler-state@1"
)
DEFAULT_PROOF_LEASE_SECONDS: Final = 300
DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.02
# Opt-in validation-pipeline ordering.  A phase is a barrier, while stages in
# the same phase remain independent and may use the scheduler's full bounded
# parallelism.  In particular, model and solver candidates can race after all
# translation work, and reconstruction can overlap independent kernel checks.
STAGED_PROOF_PHASES: Final[tuple[tuple[ProofStage, ...], ...]] = (
    (ProofStage.TRANSLATE,),
    (ProofStage.MODEL_DRAFT, ProofStage.SOLVE),
    (ProofStage.RECONSTRUCT, ProofStage.KERNEL_VERIFY),
    (ProofStage.VALIDATE,),
    (ProofStage.ATTEST,),
    (ProofStage.PERSIST,),
)
_STAGED_PROOF_PHASE_BY_STAGE: Final = {
    stage: phase
    for phase, stages in enumerate(STAGED_PROOF_PHASES)
    for stage in stages
}


def _proof_phase(stage: ProofStage | str) -> int:
    normalized = stage if isinstance(stage, ProofStage) else ProofStage(str(stage))
    return _STAGED_PROOF_PHASE_BY_STAGE[normalized]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProofNodeState(str, Enum):
    """Durable lifecycle state of one proof-plan node."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            ProofNodeState.SUCCEEDED,
            ProofNodeState.FAILED,
            ProofNodeState.UNSUPPORTED,
            ProofNodeState.BLOCKED,
            ProofNodeState.CANCELLED,
        }

    @property
    def dependency_satisfied(self) -> bool:
        return self is ProofNodeState.SUCCEEDED


# Compatibility spellings useful to callers which refer to plan steps rather
# than graph nodes.
ProofStepState = ProofNodeState
StepState = ProofNodeState


_STATUS_TO_STATE = {
    AttemptStatus.PLANNED: ProofNodeState.PENDING,
    AttemptStatus.RUNNING: ProofNodeState.RUNNING,
    AttemptStatus.SUCCEEDED: ProofNodeState.SUCCEEDED,
    AttemptStatus.FAILED: ProofNodeState.FAILED,
    AttemptStatus.UNSUPPORTED: ProofNodeState.UNSUPPORTED,
    AttemptStatus.UNAVAILABLE: ProofNodeState.UNSUPPORTED,
    AttemptStatus.TIMED_OUT: ProofNodeState.FAILED,
    AttemptStatus.CANCELLED: ProofNodeState.CANCELLED,
    AttemptStatus.BLOCKED: ProofNodeState.BLOCKED,
}
_STATE_TO_STATUS = {
    ProofNodeState.PENDING: AttemptStatus.PLANNED,
    ProofNodeState.READY: AttemptStatus.PLANNED,
    ProofNodeState.RUNNING: AttemptStatus.RUNNING,
    ProofNodeState.SUCCEEDED: AttemptStatus.SUCCEEDED,
    ProofNodeState.FAILED: AttemptStatus.FAILED,
    ProofNodeState.UNSUPPORTED: AttemptStatus.UNSUPPORTED,
    ProofNodeState.BLOCKED: AttemptStatus.BLOCKED,
    ProofNodeState.CANCELLED: AttemptStatus.CANCELLED,
}


@dataclass(frozen=True)
class ProofSchedulerConfig:
    """Execution and lease limits.

    A zero ``max_parallel`` inherits the immutable plan limit.  Stage and
    resource-class limits are local policy refinements and can only reduce that
    global limit.
    """

    max_parallel: int = 0
    lease_seconds: int = DEFAULT_PROOF_LEASE_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    stage_limits: Mapping[ProofStage | str, int] = field(default_factory=dict)
    resource_limits: Mapping[str, int] = field(default_factory=dict)
    max_cpu_proof_concurrency: int = 0
    max_model_concurrency: int = 0
    max_artifact_concurrency: int = 0
    stage_barriers: bool = False
    # Compatibility spelling used by validation-pipeline callers.  ``None``
    # means the canonical ``stage_barriers`` value is authoritative.
    staged_execution: bool | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_parallel, bool)
            or not isinstance(self.max_parallel, int)
            or self.max_parallel < 0
        ):
            raise ValueError("max_parallel must be a non-negative integer")
        for name in (
            "max_cpu_proof_concurrency",
            "max_model_concurrency",
            "max_artifact_concurrency",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.lease_seconds, bool) or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        if (
            isinstance(self.poll_interval_seconds, bool)
            or not isinstance(self.poll_interval_seconds, (int, float))
            or self.poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be positive")
        if not isinstance(self.stage_barriers, bool):
            raise ValueError("stage_barriers must be a boolean")
        if self.staged_execution is not None and not isinstance(
            self.staged_execution, bool
        ):
            raise ValueError("staged_execution must be a boolean or None")
        if (
            self.staged_execution is not None
            and self.stage_barriers
            and self.staged_execution is not self.stage_barriers
        ):
            raise ValueError(
                "stage_barriers and staged_execution must not conflict"
            )
        resolved_stage_barriers = (
            self.stage_barriers
            if self.staged_execution is None
            else self.staged_execution
        )
        object.__setattr__(self, "stage_barriers", resolved_stage_barriers)
        object.__setattr__(self, "staged_execution", resolved_stage_barriers)
        normalized_stages: dict[str, int] = {}
        for raw_stage, raw_limit in self.stage_limits.items():
            stage = raw_stage if isinstance(raw_stage, ProofStage) else ProofStage(str(raw_stage))
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit <= 0:
                raise ValueError("stage limits must be positive integers")
            normalized_stages[stage.value] = raw_limit
        normalized_resources: dict[str, int] = {}
        for raw_resource, raw_limit in self.resource_limits.items():
            resource = str(raw_resource).strip()
            if not resource:
                raise ValueError("resource-class limit names must not be empty")
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit <= 0:
                raise ValueError("resource limits must be positive integers")
            normalized_resources[resource] = raw_limit
        object.__setattr__(self, "stage_limits", normalized_stages)
        object.__setattr__(self, "resource_limits", normalized_resources)


@dataclass(frozen=True)
class ProofStepPriority:
    """Observable scheduling priority for one ready node."""

    step_id: str
    critical_path_length: int
    downstream_unlock_count: int
    user_priority: int = 0

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        # Highest values first, deterministic step id last.
        return (
            -self.critical_path_length,
            -self.downstream_unlock_count,
            -self.user_priority,
            self.step_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "critical_path_length": self.critical_path_length,
            "downstream_unlock_count": self.downstream_unlock_count,
            "user_priority": self.user_priority,
        }


@dataclass(frozen=True)
class ScheduledProofStep:
    step: ProofPlanStep
    priority: ProofStepPriority

    @property
    def step_id(self) -> str:
        return self.step.step_id

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step.to_dict(), "priority": self.priority.to_dict()}


@dataclass(frozen=True)
class ProofStepResult:
    """Normalized callback result.

    A successful callback is not itself proof authority.  Only typed
    :class:`ProofReceipt` evidence is considered for portfolio conclusion.
    """

    status: AttemptStatus = AttemptStatus.SUCCEEDED
    attempt: ProofAttempt | None = None
    receipts: tuple[ProofReceipt, ...] = ()
    output_ids: tuple[str, ...] = ()
    error_code: str = ""
    error_message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status
        if not isinstance(status, AttemptStatus):
            status = AttemptStatus(str(status))
        if not status.terminal:
            raise ContractValidationError(
                "proof executor results must have a terminal attempt status"
            )
        object.__setattr__(self, "status", status)
        receipts = tuple(
            item if isinstance(item, ProofReceipt) else ProofReceipt.from_dict(item)
            for item in self.receipts
        )
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(
            self,
            "output_ids",
            tuple(sorted({str(item).strip() for item in self.output_ids if str(item).strip()})),
        )
        object.__setattr__(self, "error_code", str(self.error_code or "").strip())
        object.__setattr__(self, "error_message", str(self.error_message or "").strip())
        # Verify JSON safety without accepting callback-owned mutable state.
        object.__setattr__(self, "metadata", json.loads(canonical_json(dict(self.metadata))))

    @property
    def receipt(self) -> ProofReceipt | None:
        return self.receipts[0] if self.receipts else None


@dataclass(frozen=True)
class ProofExecutionContext:
    """Inputs and controls supplied to a node executor."""

    plan: ProofPlan
    step: ProofPlanStep
    cancellation_token: CancellationToken
    dependency_attempts: tuple[ProofAttempt, ...]
    dependency_receipts: tuple[ProofReceipt, ...]
    fencing_token: int
    owner_id: str
    resource_lease: ResourceAdmissionLease | None
    _heartbeat: Callable[[], bool] = field(repr=False, compare=False)

    @property
    def token(self) -> CancellationToken:
        return self.cancellation_token

    @property
    def step_id(self) -> str:
        return self.step.step_id

    @property
    def stage(self) -> ProofStage:
        return self.step.stage

    @property
    def provider_id(self) -> str:
        return self.step.provider_id

    @property
    def obligation_id(self) -> str:
        return self.step.obligation_id

    @property
    def depends_on(self) -> tuple[str, ...]:
        return self.step.depends_on

    @property
    def resource_class(self) -> str:
        if self.resource_lease is not None:
            return self.resource_lease.resource_class
        return normalize_resource_class(self.step.resource_class, stage=self.step.stage)

    @property
    def resource_budget(self) -> ResourceLeaseBudget | None:
        """Supervisor lease budget inherited by nested executors."""

        return self.resource_lease.budget if self.resource_lease is not None else None

    @property
    def child_resource_limits(self) -> Mapping[str, int]:
        """Serializable portfolio/kernel limits derived from the parent lease."""

        if self.resource_lease is None:
            return {}
        return self.resource_lease.child_limits.to_dict()

    # Short spelling used by portfolio and kernel adapters.
    @property
    def child_limits(self) -> Mapping[str, int]:
        return self.child_resource_limits

    @property
    def inputs(self) -> tuple[str, ...]:
        values: set[str] = set()
        for attempt in self.dependency_attempts:
            values.update(attempt.output_ids)
        values.update(receipt.receipt_id for receipt in self.dependency_receipts)
        return tuple(sorted(values))

    def heartbeat(self) -> bool:
        """Extend the durable lease if this execution still owns it."""

        return self._heartbeat()


@dataclass(frozen=True)
class ProofNodeSnapshot:
    step_id: str
    state: ProofNodeState
    reason_code: str = ""
    attempt_id: str = ""
    cancellation_requested: bool = False
    stage: str = ""
    resource_class: str = ""
    dependency_step_ids: tuple[str, ...] = ()
    lease_owner_id: str = ""
    fencing_token: int = 0
    lease_expires_at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "attempt_id": self.attempt_id,
            "cancellation_requested": self.cancellation_requested,
            "stage": self.stage,
            "resource_class": self.resource_class,
            "dependency_step_ids": list(self.dependency_step_ids),
            "lease_owner_id": self.lease_owner_id,
            "fencing_token": self.fencing_token,
            "lease_expires_at_ms": self.lease_expires_at_ms,
        }


@dataclass(frozen=True)
class ProofLeaseSnapshot:
    """Redacted durable ownership record for one proof-plan step.

    Lease bearer tokens are intentionally absent.  Operators can still
    distinguish a live owner, a fenced predecessor, and a clean release using
    the monotonically increasing fencing token and timestamps.
    """

    step_id: str
    owner_id: str
    fencing_token: int
    acquired_at_ms: int
    heartbeat_at_ms: int
    expires_at_ms: int
    active: bool
    released_at_ms: int = 0
    release_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "owner_id": self.owner_id,
            "fencing_token": self.fencing_token,
            "acquired_at_ms": self.acquired_at_ms,
            "heartbeat_at_ms": self.heartbeat_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "active": self.active,
            "released_at_ms": self.released_at_ms,
            "release_reason": self.release_reason,
        }


@dataclass(frozen=True)
class ProofScheduleSnapshot:
    plan_id: str
    nodes: tuple[ProofNodeSnapshot, ...]
    ready: tuple[ScheduledProofStep, ...]
    active_leases: int
    attempts: tuple[ProofAttempt, ...]
    receipts: tuple[ProofReceipt, ...]
    leases: tuple[ProofLeaseSnapshot, ...] = ()

    @property
    def complete(self) -> bool:
        return all(node.state.terminal for node in self.nodes)

    @property
    def state_by_step(self) -> dict[str, ProofNodeState]:
        return {node.step_id: node.state for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCHEDULER_SCHEMA,
            "plan_id": self.plan_id,
            "complete": self.complete,
            "active_leases": self.active_leases,
            "nodes": [node.to_dict() for node in self.nodes],
            "ready": [item.to_dict() for item in self.ready],
            "attempts": [attempt.to_record() for attempt in self.attempts],
            "receipts": [receipt.to_record() for receipt in self.receipts],
            "leases": [lease.to_dict() for lease in self.leases],
            "receipt_lineage": [
                {
                    "receipt_id": receipt.receipt_id,
                    "attempt_id": receipt.attempt_id,
                    "obligation_id": receipt.obligation_id,
                    "authoritative_verdict": receipt.authoritative_verdict.value,
                    "freshness": receipt.freshness.value,
                }
                for receipt in self.receipts
            ],
        }


@dataclass(frozen=True)
class ProofScheduleResult:
    plan: ProofPlan
    snapshot: ProofScheduleSnapshot

    @property
    def complete(self) -> bool:
        return self.snapshot.complete

    @property
    def states(self) -> dict[str, ProofNodeState]:
        return self.snapshot.state_by_step

    @property
    def attempts(self) -> tuple[ProofAttempt, ...]:
        return self.snapshot.attempts

    @property
    def receipts(self) -> tuple[ProofReceipt, ...]:
        return self.snapshot.receipts

    @property
    def authoritative_receipts(self) -> tuple[ProofReceipt, ...]:
        return tuple(
            receipt
            for receipt in self.receipts
            if receipt.authoritative_verdict.conclusive
        )

    @property
    def succeeded(self) -> bool:
        return self.complete and all(
            node.state is ProofNodeState.SUCCEEDED
            or (
                node.state is ProofNodeState.CANCELLED
                and node.reason_code.startswith("portfolio_concluded:")
            )
            for node in self.snapshot.nodes
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.snapshot.to_dict()
        payload["succeeded"] = self.succeeded
        payload["authoritative_receipt_ids"] = [
            receipt.receipt_id for receipt in self.authoritative_receipts
        ]
        return payload


@dataclass(frozen=True)
class _Lease:
    step_id: str
    owner_id: str
    token: str
    fencing_token: int


class _ProofStateStore:
    """Transactional plan, lease, attempt, and receipt state."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None,
        *,
        clock: Callable[[], float],
        duckdb_timeout_seconds: int = 30,
    ) -> None:
        self.path, self._legacy_path = resolve_duckdb_path(
            path,
            default_filename="proof_scheduler.duckdb",
            temporary_prefix="proof-scheduler-",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock
        self._timeout = duckdb_timeout_seconds
        self._initialize()

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def connect(self) -> DuckDBConnection:
        return open_duckdb_connection(
            self.path,
            timeout_seconds=self._timeout,
        )

    def _initialize(self) -> None:
        initialize_duckdb_database(
            self.path,
            legacy_sqlite_path=self._legacy_path,
            timeout_seconds=self._timeout,
            table_names=(
                "proof_plans",
                "proof_nodes",
                "proof_leases",
                "proof_attempts",
                "proof_receipts",
            ),
            schema_sql="""
                CREATE TABLE IF NOT EXISTS proof_plans (
                    plan_id TEXT PRIMARY KEY,
                    plan_json TEXT NOT NULL,
                    created_at_ms BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proof_nodes (
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_id TEXT NOT NULL DEFAULT '',
                    reason_code TEXT NOT NULL DEFAULT '',
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    updated_at_ms BIGINT NOT NULL,
                    PRIMARY KEY(plan_id, step_id),
                    FOREIGN KEY(plan_id) REFERENCES proof_plans(plan_id)
                );
                CREATE TABLE IF NOT EXISTS proof_leases (
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    fencing_token BIGINT NOT NULL,
                    acquired_at_ms BIGINT NOT NULL,
                    heartbeat_at_ms BIGINT NOT NULL,
                    expires_at_ms BIGINT NOT NULL,
                    released_at_ms BIGINT,
                    release_reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(plan_id, step_id)
                );
                CREATE INDEX IF NOT EXISTS proof_leases_active
                    ON proof_leases(plan_id, expires_at_ms, released_at_ms);
                CREATE TABLE IF NOT EXISTS proof_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    attempt_json TEXT NOT NULL,
                    created_at_ms BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS proof_attempts_plan_step
                    ON proof_attempts(plan_id, step_id, created_at_ms);
                CREATE TABLE IF NOT EXISTS proof_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    obligation_id TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    authoritative INTEGER NOT NULL,
                    created_at_ms BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS proof_authoritative_receipts
                    ON proof_receipts(plan_id, obligation_id, authoritative);
                """,
        )

    def register_plan(self, plan: ProofPlan) -> None:
        encoded = canonical_json(plan.to_dict())
        connection = self.connect()
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT plan_json FROM proof_plans WHERE plan_id=?", (plan.plan_id,)
            ).fetchone()
            if row is not None and str(row["plan_json"]) != encoded:
                raise ContractValidationError(
                    "durable proof plan identity has conflicting content"
                )
            connection.execute(
                "INSERT OR IGNORE INTO proof_plans(plan_id, plan_json, created_at_ms) "
                "VALUES (?, ?, ?)",
                (plan.plan_id, encoded, now),
            )
            for step in plan.steps:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO proof_nodes(
                        plan_id, step_id, state, updated_at_ms
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (plan.plan_id, step.step_id, ProofNodeState.PENDING.value, now),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load_plan(self, plan_id: str = "") -> ProofPlan:
        connection = self.connect()
        try:
            if plan_id:
                row = connection.execute(
                    "SELECT plan_json FROM proof_plans WHERE plan_id=?", (plan_id,)
                ).fetchone()
            else:
                rows = connection.execute(
                    "SELECT plan_json FROM proof_plans ORDER BY created_at_ms, plan_id"
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError(
                        "plan_id is required when durable state contains zero or multiple plans"
                    )
                row = rows[0]
        finally:
            connection.close()
        if row is None:
            raise ValueError("durable proof plan was not found")
        return ProofPlan.from_dict(json.loads(str(row["plan_json"])))

    def _expire_locked(self, connection: DuckDBConnection, plan_id: str) -> None:
        now = self._now_ms()
        rows = connection.execute(
            """
            SELECT step_id FROM proof_leases
            WHERE plan_id=? AND released_at_ms IS NULL AND expires_at_ms<=?
            """,
            (plan_id, now),
        ).fetchall()
        for row in rows:
            step_id = str(row["step_id"])
            connection.execute(
                """
                UPDATE proof_leases SET released_at_ms=?, release_reason='lease_expired'
                WHERE plan_id=? AND step_id=? AND released_at_ms IS NULL
                """,
                (now, plan_id, step_id),
            )
            connection.execute(
                """
                UPDATE proof_nodes
                SET state=?, reason_code='lease_expired_requeued', attempt_id='',
                    updated_at_ms=?
                WHERE plan_id=? AND step_id=? AND state=?
                """,
                (
                    ProofNodeState.PENDING.value,
                    now,
                    plan_id,
                    step_id,
                    ProofNodeState.RUNNING.value,
                ),
            )

    def recover(self, plan_id: str) -> None:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, plan_id)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def node_rows(self, plan_id: str) -> list[DuckDBRow]:
        connection = self.connect()
        try:
            return connection.execute(
                "SELECT * FROM proof_nodes WHERE plan_id=? ORDER BY step_id",
                (plan_id,),
            ).fetchall()
        finally:
            connection.close()

    def update_nodes(
        self, plan_id: str, updates: Mapping[str, tuple[ProofNodeState, str]]
    ) -> None:
        if not updates:
            return
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = self._now_ms()
            for step_id, (state, reason) in updates.items():
                connection.execute(
                    """
                    UPDATE proof_nodes SET state=?, reason_code=?, updated_at_ms=?
                    WHERE plan_id=? AND step_id=? AND state NOT IN (?, ?, ?, ?, ?)
                    """,
                    (
                        state.value,
                        reason,
                        now,
                        plan_id,
                        step_id,
                        ProofNodeState.SUCCEEDED.value,
                        ProofNodeState.FAILED.value,
                        ProofNodeState.UNSUPPORTED.value,
                        ProofNodeState.BLOCKED.value,
                        ProofNodeState.CANCELLED.value,
                    ),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim(
        self,
        plan: ProofPlan,
        step: ProofPlanStep,
        *,
        owner_id: str,
        lease_seconds: int,
        max_parallel: int,
        stage_limits: Mapping[str, int],
        resource_limits: Mapping[str, int],
        pool_limits: Mapping[str, int],
        stage_barrier_phase: int | None = None,
    ) -> _Lease | None:
        connection = self.connect()
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_locked(connection, plan.plan_id)
            node = connection.execute(
                "SELECT * FROM proof_nodes WHERE plan_id=? AND step_id=?",
                (plan.plan_id, step.step_id),
            ).fetchone()
            if node is None or str(node["state"]) != ProofNodeState.READY.value:
                connection.rollback()
                return None
            if stage_barrier_phase is not None:
                states = {
                    str(row["step_id"]): ProofNodeState(str(row["state"]))
                    for row in connection.execute(
                        "SELECT step_id, state FROM proof_nodes WHERE plan_id=?",
                        (plan.plan_id,),
                    ).fetchall()
                }
                if _proof_phase(step.stage) != stage_barrier_phase or any(
                    not states[item.step_id].terminal
                    and _proof_phase(item.stage) < stage_barrier_phase
                    for item in plan.steps
                ):
                    connection.rollback()
                    return None
            active = connection.execute(
                """
                SELECT COUNT(*) AS count FROM proof_leases
                WHERE plan_id=? AND released_at_ms IS NULL AND expires_at_ms>?
                """,
                (plan.plan_id, now),
            ).fetchone()
            if int(active["count"]) >= max_parallel:
                connection.rollback()
                return None

            active_steps = connection.execute(
                """
                SELECT n.step_id FROM proof_nodes n
                JOIN proof_leases l
                  ON l.plan_id=n.plan_id AND l.step_id=n.step_id
                WHERE n.plan_id=? AND l.released_at_ms IS NULL
                  AND l.expires_at_ms>? AND n.state=?
                """,
                (plan.plan_id, now, ProofNodeState.RUNNING.value),
            ).fetchall()
            by_id = {item.step_id: item for item in plan.steps}
            active_definitions = [
                by_id[str(row["step_id"])]
                for row in active_steps
                if str(row["step_id"]) in by_id
            ]
            stage_limit = stage_limits.get(step.stage.value)
            if stage_limit is not None and sum(
                item.stage is step.stage for item in active_definitions
            ) >= stage_limit:
                connection.rollback()
                return None
            resource_limit = resource_limits.get(step.resource_class)
            if step.resource_class and resource_limit is not None and sum(
                item.resource_class == step.resource_class for item in active_definitions
            ) >= resource_limit:
                connection.rollback()
                return None
            normalized_class = normalize_resource_class(
                step.resource_class, stage=step.stage
            )
            pool = resource_pool(normalized_class)
            pool_limit = pool_limits.get(pool)
            if pool_limit is not None and sum(
                resource_pool(
                    normalize_resource_class(item.resource_class, stage=item.stage)
                )
                == pool
                for item in active_definitions
            ) >= pool_limit:
                connection.rollback()
                return None

            prior = connection.execute(
                "SELECT fencing_token FROM proof_leases WHERE plan_id=? AND step_id=?",
                (plan.plan_id, step.step_id),
            ).fetchone()
            fencing_token = 1 if prior is None else int(prior["fencing_token"]) + 1
            token = uuid.uuid4().hex
            expires = now + lease_seconds * 1000
            connection.execute(
                """
                INSERT INTO proof_leases(
                    plan_id, step_id, owner_id, token, fencing_token,
                    acquired_at_ms, heartbeat_at_ms, expires_at_ms,
                    released_at_ms, release_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '')
                ON CONFLICT(plan_id, step_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    token=excluded.token,
                    fencing_token=excluded.fencing_token,
                    acquired_at_ms=excluded.acquired_at_ms,
                    heartbeat_at_ms=excluded.heartbeat_at_ms,
                    expires_at_ms=excluded.expires_at_ms,
                    released_at_ms=NULL,
                    release_reason=''
                """,
                (
                    plan.plan_id,
                    step.step_id,
                    owner_id,
                    token,
                    fencing_token,
                    now,
                    now,
                    expires,
                ),
            )
            connection.execute(
                """
                UPDATE proof_nodes SET state=?, reason_code='', updated_at_ms=?
                WHERE plan_id=? AND step_id=?
                """,
                (ProofNodeState.RUNNING.value, now, plan.plan_id, step.step_id),
            )
            connection.commit()
            return _Lease(step.step_id, owner_id, token, fencing_token)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(
        self, plan_id: str, lease: _Lease, lease_seconds: int
    ) -> bool:
        connection = self.connect()
        now = self._now_ms()
        try:
            cursor = connection.execute(
                """
                UPDATE proof_leases SET heartbeat_at_ms=?, expires_at_ms=?
                WHERE plan_id=? AND step_id=? AND owner_id=? AND token=?
                  AND fencing_token=? AND released_at_ms IS NULL AND expires_at_ms>?
                """,
                (
                    now,
                    now + lease_seconds * 1000,
                    plan_id,
                    lease.step_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    now,
                ),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    def lease_is_current(self, plan_id: str, lease: _Lease) -> bool:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT 1 FROM proof_leases
                WHERE plan_id=? AND step_id=? AND owner_id=? AND token=?
                  AND fencing_token=? AND released_at_ms IS NULL AND expires_at_ms>?
                """,
                (
                    plan_id,
                    lease.step_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    self._now_ms(),
                ),
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    def set_attempt(self, plan_id: str, step_id: str, attempt: ProofAttempt) -> None:
        connection = self.connect()
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO proof_attempts(
                    attempt_id, plan_id, step_id, attempt_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    plan_id,
                    step_id,
                    canonical_json(attempt.to_dict()),
                    now,
                ),
            )
            connection.execute(
                "UPDATE proof_nodes SET attempt_id=?, updated_at_ms=? "
                "WHERE plan_id=? AND step_id=?",
                (attempt.attempt_id, now, plan_id, step_id),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def attempts(self, plan_id: str, step_ids: Sequence[str] | None = None) -> tuple[ProofAttempt, ...]:
        connection = self.connect()
        try:
            if step_ids is None:
                rows = connection.execute(
                    "SELECT attempt_json FROM proof_attempts WHERE plan_id=? "
                    "ORDER BY created_at_ms, attempt_id",
                    (plan_id,),
                ).fetchall()
            elif not step_ids:
                return ()
            else:
                marks = ",".join("?" for _ in step_ids)
                rows = connection.execute(
                    f"SELECT attempt_json FROM proof_attempts WHERE plan_id=? "
                    f"AND step_id IN ({marks}) ORDER BY created_at_ms, attempt_id",
                    (plan_id, *step_ids),
                ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProofAttempt.from_dict(json.loads(str(row["attempt_json"]))) for row in rows
        )

    def receipts(self, plan_id: str, obligation_ids: Sequence[str] | None = None) -> tuple[ProofReceipt, ...]:
        connection = self.connect()
        try:
            if obligation_ids is None:
                rows = connection.execute(
                    "SELECT receipt_json FROM proof_receipts WHERE plan_id=? "
                    "ORDER BY created_at_ms, receipt_id",
                    (plan_id,),
                ).fetchall()
            elif not obligation_ids:
                return ()
            else:
                marks = ",".join("?" for _ in obligation_ids)
                rows = connection.execute(
                    f"SELECT receipt_json FROM proof_receipts WHERE plan_id=? "
                    f"AND obligation_id IN ({marks}) ORDER BY created_at_ms, receipt_id",
                    (plan_id, *obligation_ids),
                ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProofReceipt.from_dict(json.loads(str(row["receipt_json"]))) for row in rows
        )

    def store_receipt(self, plan: ProofPlan, step: ProofPlanStep, receipt: ProofReceipt) -> ProofReceipt:
        if receipt.plan_id != plan.plan_id:
            raise ContractValidationError("receipt plan_id does not match scheduled plan")
        if receipt.repository_tree_id != plan.repository_tree_id:
            raise ContractValidationError(
                "receipt repository_tree_id does not match scheduled plan"
            )
        if receipt.obligation_id != step.obligation_id:
            raise ContractValidationError(
                "receipt obligation_id does not match scheduled step"
            )
        authoritative = receipt.authoritative_verdict.conclusive
        connection = self.connect()
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if authoritative:
                existing = connection.execute(
                    """
                    SELECT receipt_json FROM proof_receipts
                    WHERE plan_id=? AND obligation_id=? AND authoritative=1
                    """,
                    (plan.plan_id, receipt.obligation_id),
                ).fetchone()
                if existing is not None:
                    connection.commit()
                    return ProofReceipt.from_dict(json.loads(str(existing["receipt_json"])))
            connection.execute(
                """
                INSERT OR IGNORE INTO proof_receipts(
                    receipt_id, plan_id, obligation_id, receipt_json,
                    authoritative, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    plan.plan_id,
                    receipt.obligation_id,
                    canonical_json(receipt.to_dict()),
                    int(authoritative),
                    now,
                ),
            )
            connection.commit()
            return receipt
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def commit_result(
        self,
        plan: ProofPlan,
        step: ProofPlanStep,
        lease: _Lease,
        *,
        attempt: ProofAttempt,
        receipts: Sequence[ProofReceipt],
        state: ProofNodeState,
        reason_code: str,
    ) -> tuple[bool, ProofNodeState, tuple[ProofReceipt, ...]]:
        """Fence and publish a complete callback result in one transaction.

        Receipt admission must share the same lock and lease check as the node
        transition.  Otherwise a lease could expire between two transactions
        and a stale worker might win the authoritative-receipt uniqueness key.
        """

        for receipt in receipts:
            if receipt.plan_id != plan.plan_id:
                raise ContractValidationError(
                    "receipt plan_id does not match scheduled plan"
                )
            if receipt.repository_tree_id != plan.repository_tree_id:
                raise ContractValidationError(
                    "receipt repository_tree_id does not match scheduled plan"
                )
            if receipt.obligation_id != step.obligation_id:
                raise ContractValidationError(
                    "receipt obligation_id does not match scheduled step"
                )

        connection = self.connect()
        now = self._now_ms()
        stored: list[ProofReceipt] = []
        actual_state = state
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT * FROM proof_leases
                WHERE plan_id=? AND step_id=? AND owner_id=? AND token=?
                  AND fencing_token=? AND released_at_ms IS NULL
                """,
                (
                    plan.plan_id,
                    lease.step_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                ),
            ).fetchone()
            if current is None or int(current["expires_at_ms"]) <= now:
                connection.rollback()
                return False, state, ()

            node = connection.execute(
                "SELECT cancellation_requested, reason_code FROM proof_nodes "
                "WHERE plan_id=? AND step_id=?",
                (plan.plan_id, lease.step_id),
            ).fetchone()
            if node is not None and bool(node["cancellation_requested"]):
                actual_state = ProofNodeState.CANCELLED
                reason_code = str(node["reason_code"]) or "cancellation_requested"

            connection.execute(
                """
                INSERT OR IGNORE INTO proof_attempts(
                    attempt_id, plan_id, step_id, attempt_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    plan.plan_id,
                    step.step_id,
                    canonical_json(attempt.to_dict()),
                    now,
                ),
            )
            # A cancellation request fences publication as well as the node
            # transition.  The attempt remains in the audit log, but its
            # receipts cannot become authoritative after cancellation.
            publishable_receipts = (
                receipts if actual_state is not ProofNodeState.CANCELLED else ()
            )
            for receipt in publishable_receipts:
                authoritative = receipt.authoritative_verdict.conclusive
                existing = None
                if authoritative:
                    existing = connection.execute(
                        """
                        SELECT receipt_json FROM proof_receipts
                        WHERE plan_id=? AND obligation_id=? AND authoritative=1
                        """,
                        (plan.plan_id, receipt.obligation_id),
                    ).fetchone()
                if existing is not None:
                    stored.append(
                        ProofReceipt.from_dict(
                            json.loads(str(existing["receipt_json"]))
                        )
                    )
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO proof_receipts(
                        receipt_id, plan_id, obligation_id, receipt_json,
                        authoritative, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        plan.plan_id,
                        receipt.obligation_id,
                        canonical_json(receipt.to_dict()),
                        int(authoritative),
                        now,
                    ),
                )
                stored.append(receipt)

            connection.execute(
                """
                UPDATE proof_nodes
                SET state=?, attempt_id=?, reason_code=?,
                    cancellation_requested=0, updated_at_ms=?
                WHERE plan_id=? AND step_id=?
                """,
                (
                    actual_state.value,
                    attempt.attempt_id,
                    reason_code,
                    now,
                    plan.plan_id,
                    lease.step_id,
                ),
            )
            connection.execute(
                """
                UPDATE proof_leases
                SET released_at_ms=?, release_reason=?
                WHERE plan_id=? AND step_id=? AND token=? AND fencing_token=?
                """,
                (
                    now,
                    actual_state.value,
                    plan.plan_id,
                    lease.step_id,
                    lease.token,
                    lease.fencing_token,
                ),
            )
            connection.commit()
            return True, actual_state, tuple(stored)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finish(
        self,
        plan_id: str,
        lease: _Lease,
        *,
        state: ProofNodeState,
        attempt_id: str,
        reason_code: str,
    ) -> bool:
        connection = self.connect()
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT * FROM proof_leases
                WHERE plan_id=? AND step_id=? AND owner_id=? AND token=?
                  AND fencing_token=? AND released_at_ms IS NULL
                """,
                (
                    plan_id,
                    lease.step_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                ),
            ).fetchone()
            if current is None or int(current["expires_at_ms"]) <= now:
                connection.rollback()
                return False
            node = connection.execute(
                "SELECT cancellation_requested FROM proof_nodes "
                "WHERE plan_id=? AND step_id=?",
                (plan_id, lease.step_id),
            ).fetchone()
            if node is not None and bool(node["cancellation_requested"]):
                state = ProofNodeState.CANCELLED
                reason_code = reason_code or "cancellation_requested"
            connection.execute(
                """
                UPDATE proof_nodes
                SET state=?, attempt_id=?, reason_code=?,
                    cancellation_requested=0, updated_at_ms=?
                WHERE plan_id=? AND step_id=?
                """,
                (state.value, attempt_id, reason_code, now, plan_id, lease.step_id),
            )
            connection.execute(
                """
                UPDATE proof_leases
                SET released_at_ms=?, release_reason=?
                WHERE plan_id=? AND step_id=? AND token=? AND fencing_token=?
                """,
                (
                    now,
                    state.value,
                    plan_id,
                    lease.step_id,
                    lease.token,
                    lease.fencing_token,
                ),
            )
            connection.commit()
            return True
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancel_steps(
        self, plan_id: str, step_ids: Sequence[str], reason_code: str
    ) -> tuple[str, ...]:
        if not step_ids:
            return ()
        connection = self.connect()
        cancelled: list[str] = []
        now = self._now_ms()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for step_id in step_ids:
                row = connection.execute(
                    "SELECT state FROM proof_nodes WHERE plan_id=? AND step_id=?",
                    (plan_id, step_id),
                ).fetchone()
                if row is None:
                    continue
                state = ProofNodeState(str(row["state"]))
                if state.terminal:
                    continue
                if state is ProofNodeState.RUNNING:
                    connection.execute(
                        """
                        UPDATE proof_nodes SET cancellation_requested=1,
                            reason_code=?, updated_at_ms=?
                        WHERE plan_id=? AND step_id=?
                        """,
                        (reason_code, now, plan_id, step_id),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE proof_nodes SET state=?, reason_code=?,
                            cancellation_requested=0, updated_at_ms=?
                        WHERE plan_id=? AND step_id=?
                        """,
                        (
                            ProofNodeState.CANCELLED.value,
                            reason_code,
                            now,
                            plan_id,
                            step_id,
                        ),
                    )
                cancelled.append(step_id)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return tuple(cancelled)

    def active_lease_count(self, plan_id: str) -> int:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM proof_leases
                WHERE plan_id=? AND released_at_ms IS NULL AND expires_at_ms>?
                """,
                (plan_id, self._now_ms()),
            ).fetchone()
            return int(row["count"])
        finally:
            connection.close()

    def lease_snapshots(self, plan_id: str) -> tuple[ProofLeaseSnapshot, ...]:
        """Return redacted current and historical lease ownership."""

        connection = self.connect()
        now = self._now_ms()
        try:
            rows = connection.execute(
                """
                SELECT step_id, owner_id, fencing_token, acquired_at_ms,
                       heartbeat_at_ms, expires_at_ms, released_at_ms,
                       release_reason
                FROM proof_leases
                WHERE plan_id=?
                ORDER BY step_id
                """,
                (plan_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            ProofLeaseSnapshot(
                step_id=str(row["step_id"]),
                owner_id=str(row["owner_id"]),
                fencing_token=int(row["fencing_token"]),
                acquired_at_ms=int(row["acquired_at_ms"]),
                heartbeat_at_ms=int(row["heartbeat_at_ms"]),
                expires_at_ms=int(row["expires_at_ms"]),
                active=(
                    row["released_at_ms"] is None
                    and int(row["expires_at_ms"]) > now
                ),
                released_at_ms=(
                    int(row["released_at_ms"])
                    if row["released_at_ms"] is not None
                    else 0
                ),
                release_reason=str(row["release_reason"] or ""),
            )
            for row in rows
        )


class ProofScheduler:
    """Execute and resume one durable :class:`ProofPlan`."""

    def __init__(
        self,
        plan: ProofPlan | Mapping[str, Any] | None = None,
        executor: Callable[..., Any] | Mapping[Any, Callable[..., Any]] | None = None,
        *,
        executors: Mapping[Any, Callable[..., Any]] | None = None,
        state_path: str | os.PathLike[str] | None = None,
        store_path: str | os.PathLike[str] | None = None,
        plan_id: str = "",
        max_parallel: int | None = None,
        lease_seconds: int | None = None,
        stage_limits: Mapping[ProofStage | str, int] | None = None,
        resource_limits: Mapping[str, int] | None = None,
        max_cpu_proof_concurrency: int | None = None,
        max_model_concurrency: int | None = None,
        max_artifact_concurrency: int | None = None,
        stage_barriers: bool | None = None,
        staged_execution: bool | None = None,
        staged: bool | None = None,
        config: ProofSchedulerConfig | None = None,
        resource_scheduler: ResourceScheduler | None = None,
        resource_policy: ResourcePolicy | Mapping[str, Any] | None = None,
        resource_lease_budget: ResourceLeaseBudget | None = None,
        host_resource_source: Callable[..., Any] | HostResourceSnapshot | Mapping[str, Any] | None = None,
        provider_capacity_source: Callable[..., Any] | Mapping[str, Any] | Sequence[Any] | None = None,
        owner_id: str = "",
        clock: Callable[[], float] = time.time,
    ) -> None:
        chosen_path = state_path if state_path is not None else store_path
        self._store = _ProofStateStore(chosen_path, clock=clock)
        if plan is None:
            self.plan = self._store.load_plan(plan_id)
        else:
            self.plan = plan if isinstance(plan, ProofPlan) else ProofPlan.from_dict(plan)
            if plan_id and plan_id != self.plan.plan_id:
                raise ValueError("plan_id does not match plan")
            self._store.register_plan(self.plan)

        barrier_aliases = tuple(
            value
            for value in (stage_barriers, staged_execution, staged)
            if value is not None
        )
        if any(not isinstance(value, bool) for value in barrier_aliases):
            raise ValueError(
                "stage_barriers, staged_execution, and staged must be booleans"
            )
        if len(set(barrier_aliases)) > 1:
            raise ValueError(
                "stage_barriers, staged_execution, and staged must not conflict"
            )
        requested_stage_barriers = (
            barrier_aliases[0] if barrier_aliases else None
        )
        if config is not None and any(
            value is not None
            for value in (
                max_parallel,
                lease_seconds,
                stage_limits,
                resource_limits,
                max_cpu_proof_concurrency,
                max_model_concurrency,
                max_artifact_concurrency,
                requested_stage_barriers,
            )
        ):
            raise ValueError("config cannot be combined with individual scheduler limits")
        self.config = config or ProofSchedulerConfig(
            max_parallel=0 if max_parallel is None else max_parallel,
            lease_seconds=(
                DEFAULT_PROOF_LEASE_SECONDS if lease_seconds is None else lease_seconds
            ),
            stage_limits=stage_limits or {},
            resource_limits=resource_limits or {},
            max_cpu_proof_concurrency=(
                0
                if max_cpu_proof_concurrency is None
                else max_cpu_proof_concurrency
            ),
            max_model_concurrency=(
                0 if max_model_concurrency is None else max_model_concurrency
            ),
            max_artifact_concurrency=(
                0
                if max_artifact_concurrency is None
                else max_artifact_concurrency
            ),
            stage_barriers=bool(requested_stage_barriers),
        )
        if self.config.stage_barriers:
            self._validate_stage_barrier_dependencies()
        self.max_parallel = self.config.max_parallel or self.plan.max_parallel
        if self.max_parallel > self.plan.max_parallel:
            # The immutable plan is the authority; runtime configuration can
            # constrain but never expand it.
            self.max_parallel = self.plan.max_parallel
        if resource_scheduler is not None and resource_policy is not None:
            raise ValueError(
                "resource_scheduler cannot be combined with resource_policy"
            )
        self._implicit_resource_admission = (
            resource_scheduler is None
            and resource_policy is None
            and resource_lease_budget is None
            and host_resource_source is None
            and provider_capacity_source is None
        )
        if resource_scheduler is None:
            if isinstance(resource_policy, ResourcePolicy):
                policy = resource_policy
            else:
                policy_values = dict(resource_policy or {})
                policy_values.setdefault("max_lanes", self.max_parallel)
                # Legacy proof scheduling did not require provider telemetry.
                # Explicit policies retain their own fail-closed default.
                if resource_policy is None:
                    policy_values["require_provider_telemetry"] = False
                policy = ResourcePolicy.from_mapping(policy_values)
            resource_scheduler = ResourceScheduler(policy)
        self.resource_scheduler = resource_scheduler
        self._host_resource_source = host_resource_source
        self._provider_capacity_source = provider_capacity_source
        policy = self.resource_scheduler.policy
        cpu_limit = (
            self.config.max_cpu_proof_concurrency
            or policy.max_cpu_proof_concurrency
            or self.max_parallel
        )
        model_limit = (
            self.config.max_model_concurrency
            or policy.max_model_concurrency
            or self.max_parallel
        )
        artifact_limit = (
            self.config.max_artifact_concurrency
            or policy.max_artifact_concurrency
            or self.max_parallel
        )
        self.resource_lease_budget = resource_lease_budget or (
            ResourceLeaseBudget.from_resource_budget(
                self.plan.resource_budget,
                max_parallel=self.max_parallel,
                max_cpu_proof_concurrency=cpu_limit,
                max_model_concurrency=model_limit,
                max_artifact_concurrency=artifact_limit,
                maximum_provider_latency_ms=policy.maximum_provider_latency_ms,
            )
        )
        self.max_parallel = min(
            self.max_parallel,
            policy.max_lanes,
            self.resource_lease_budget.max_parallel,
            self.resource_lease_budget.max_processes,
        )
        self._pool_limits = {
            "cpu-proof": min(
                cpu_limit,
                self.resource_lease_budget.max_cpu_proof_concurrency,
            ),
            "model": min(
                model_limit,
                self.resource_lease_budget.max_model_concurrency,
            ),
            "artifact": min(
                artifact_limit,
                self.resource_lease_budget.max_artifact_concurrency,
            ),
        }
        if self.max_parallel <= 0:
            raise ValueError("effective proof concurrency must be positive")
        self._last_resource_decisions: dict[str, AdmissionDecision] = {}
        self.owner_id = owner_id.strip() or f"proof-scheduler:{os.getpid()}:{uuid.uuid4().hex}"
        self._default_executor: Callable[..., Any] | None = None
        self._executors: dict[str, Callable[..., Any]] = {}
        if callable(executor):
            self._default_executor = executor
        elif isinstance(executor, Mapping):
            self._register_executors(executor)
        elif executor is not None:
            raise TypeError("executor must be callable or a mapping")
        if executors:
            self._register_executors(executors)
        self._tokens: dict[str, CancellationToken] = {}
        self._cancel_lock = threading.Lock()
        self._cancelled = False
        self._store.recover(self.plan.plan_id)
        self.reconcile()

    @property
    def state_path(self) -> Path:
        return self._store.path

    @property
    def db_path(self) -> Path:
        return self._store.path

    def _register_executors(self, values: Mapping[Any, Callable[..., Any]]) -> None:
        for key, callback in values.items():
            if not callable(callback):
                raise TypeError("proof executors must be callable")
            if isinstance(key, ProofStage):
                normalized = key.value
            else:
                normalized = str(key).strip()
            if not normalized:
                raise ValueError("proof executor keys must not be empty")
            self._executors[normalized] = callback

    def _executor_for(self, step: ProofPlanStep) -> Callable[..., Any]:
        callback = (
            self._executors.get(step.step_id)
            or self._executors.get(step.provider_id)
            or self._executors.get(step.stage.value)
            or (self._executors.get(step.resource_class) if step.resource_class else None)
            or self._default_executor
        )
        if callback is None:
            raise LookupError(f"no proof executor is configured for step {step.step_id}")
        return callback

    @staticmethod
    def _positive_metadata_int(
        metadata: Mapping[str, Any],
        *names: str,
        default: int = 0,
    ) -> int:
        resources = metadata.get("resources")
        sources = (metadata, resources if isinstance(resources, Mapping) else {})
        for source in sources:
            for name in names:
                value = source.get(name)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    return value
        return default

    def resource_requirement(self, step: ProofPlanStep) -> LaneResourceRequirements:
        """Translate one proof node into the shared admission vocabulary."""

        resource_class = normalize_resource_class(
            step.resource_class,
            stage=step.stage,
        )
        metadata = step.metadata
        requested_processes = self._positive_metadata_int(
            metadata,
            "process_slots",
            "required_processes",
            "max_processes",
            "portfolio_max_parallel",
            "portfolio_width",
            "kernel_max_parallel",
            default=1,
        )
        requested_processes = max(1, requested_processes)
        pool = resource_pool(resource_class)
        process_slots = min(
            requested_processes,
            self.resource_lease_budget.max_processes,
            self._pool_limits[pool],
        )
        model_work = resource_class == ProofResourceClass.MODEL_DRAFT.value
        token_budget = self._positive_metadata_int(
            metadata,
            "token_budget",
            "model_token_limit",
            "max_new_tokens",
            default=(
                self.plan.resource_budget.model_token_limit if model_work else 0
            ),
        )
        quota_units = self._positive_metadata_int(
            metadata,
            "quota_units",
            "provider_quota",
            "quota_cost",
            default=1,
        )
        capabilities = metadata.get("required_capabilities", ())
        if isinstance(capabilities, str):
            required_capabilities = tuple(
                item.strip().lower()
                for item in capabilities.split(",")
                if item.strip()
            )
        elif isinstance(capabilities, Sequence):
            required_capabilities = tuple(
                sorted(
                    {
                        str(item).strip().lower()
                        for item in capabilities
                        if str(item).strip()
                    }
                )
            )
        else:
            required_capabilities = ()
        if model_work:
            required_capabilities = tuple(
                sorted(
                    set(required_capabilities)
                    | {
                        item
                        if item.startswith(("llm:", "host:"))
                        else f"llm:{item}"
                        for item in required_capabilities
                    }
                )
            )
        return LaneResourceRequirements(
            lane_id=step.step_id,
            resource_class=resource_class,
            required_capabilities=required_capabilities,
            provider_id=step.provider_id if model_work else "",
            requires_provider=model_work,
            context_tokens=self._positive_metadata_int(
                metadata,
                "context_tokens",
                "required_context_tokens",
                "estimated_context_tokens",
            ),
            token_budget=token_budget,
            quota_units=quota_units,
            memory_bytes=self._positive_metadata_int(
                metadata,
                "memory_bytes",
                "required_memory_bytes",
            ),
            disk_bytes=self._positive_metadata_int(
                metadata,
                "disk_bytes",
                "required_disk_bytes",
            ),
            max_provider_latency_ms=self._positive_metadata_int(
                metadata,
                "max_provider_latency_ms",
                "latency_budget_ms",
            ),
            process_slots=process_slots,
        )

    @staticmethod
    def _read_capacity_source(source: Any, step: ProofPlanStep) -> Any:
        if not callable(source):
            return source
        try:
            signature = inspect.signature(source)
        except (TypeError, ValueError):
            return source()
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        ]
        return source(step) if positional else source()

    def _acquire_resource(
        self,
        step: ProofPlanStep,
    ) -> tuple[AdmissionDecision, ResourceAdmissionLease | None]:
        host = self._read_capacity_source(self._host_resource_source, step)
        if host is None and self._implicit_resource_admission:
            # Preserve deterministic behavior for legacy callers that did not
            # opt into live telemetry. Explicit unified-admission callers use
            # the configured source (or ResourceScheduler's live sampler).
            host = HostResourceSnapshot(
                worker_limit=self.max_parallel,
                available_worker_capacity=self.max_parallel,
            )
        providers = self._read_capacity_source(
            self._provider_capacity_source, step
        )
        decision, lease = self.resource_scheduler.acquire(
            self.resource_requirement(step),
            budget=self.resource_lease_budget,
            host=host,
            providers=providers,
            path=self.state_path.parent,
        )
        self._last_resource_decisions[step.step_id] = decision
        return decision, lease

    @property
    def resource_decisions(self) -> Mapping[str, AdmissionDecision]:
        return dict(self._last_resource_decisions)

    @property
    def stage_barriers(self) -> bool:
        """Whether this scheduler enforces the canonical proof phase barriers."""

        return self.config.stage_barriers

    @staticmethod
    def proof_phase(stage: ProofStage | str) -> int:
        """Return the validation-pipeline phase for a proof stage."""

        return _proof_phase(stage)

    @staticmethod
    def _normalize_stages(
        stages: Iterable[ProofStage | str] | ProofStage | str | None,
    ) -> frozenset[ProofStage] | None:
        if stages is None:
            return None
        if isinstance(stages, (ProofStage, str)):
            values: Iterable[ProofStage | str] = (stages,)
        else:
            values = stages
        normalized = frozenset(
            value if isinstance(value, ProofStage) else ProofStage(str(value))
            for value in values
        )
        if not normalized:
            raise ValueError("stages must contain at least one proof stage")
        return normalized

    def _validate_stage_barrier_dependencies(self) -> None:
        """Reject plans whose dependency edges run backwards across phases.

        Such an edge cannot make progress in a staged validation pipeline:
        the earlier phase would wait on work which the barrier deliberately
        withholds until a later phase.  Ordinary DAG execution remains
        backwards compatible because this check is opt-in.
        """

        by_id = {step.step_id: step for step in self.plan.steps}
        for step in self.plan.steps:
            phase = self.proof_phase(step.stage)
            for dependency_id in step.depends_on:
                dependency = by_id[dependency_id]
                dependency_phase = self.proof_phase(dependency.stage)
                if dependency_phase > phase:
                    raise ContractValidationError(
                        "staged proof dependency runs backwards: "
                        f"{step.step_id} ({step.stage.value}) depends on "
                        f"{dependency.step_id} ({dependency.stage.value})"
                    )

    def _eligible_phase(
        self,
        snapshot: ProofScheduleSnapshot,
        stages: frozenset[ProofStage] | None,
    ) -> int | None:
        by_id = {step.step_id: step for step in self.plan.steps}
        phases = [
            self.proof_phase(by_id[node.step_id].stage)
            for node in snapshot.nodes
            if not node.state.terminal
            and (stages is None or by_id[node.step_id].stage in stages)
        ]
        return min(phases) if phases else None

    def _selected_nodes_complete(
        self,
        snapshot: ProofScheduleSnapshot,
        stages: frozenset[ProofStage],
    ) -> bool:
        by_id = {step.step_id: step for step in self.plan.steps}
        return all(
            node.state.terminal
            for node in snapshot.nodes
            if by_id[node.step_id].stage in stages
        )

    def priorities(self) -> dict[str, ProofStepPriority]:
        critical = self.plan.critical_path_lengths
        unlocks = self.plan.downstream_unlock_counts
        result: dict[str, ProofStepPriority] = {}
        for step in self.plan.steps:
            raw_user = step.metadata.get("priority", 0)
            user = raw_user if isinstance(raw_user, int) and not isinstance(raw_user, bool) else 0
            result[step.step_id] = ProofStepPriority(
                step_id=step.step_id,
                critical_path_length=critical[step.step_id],
                downstream_unlock_count=unlocks[step.step_id],
                user_priority=user,
            )
        return result

    priority_by_step = priorities

    def _state_map(self) -> dict[str, ProofNodeState]:
        return {
            str(row["step_id"]): ProofNodeState(str(row["state"]))
            for row in self._store.node_rows(self.plan.plan_id)
        }

    def reconcile(self) -> ProofScheduleSnapshot:
        """Recover expired work and project dependency states to a fixed point."""

        self._store.recover(self.plan.plan_id)
        by_id = {step.step_id: step for step in self.plan.steps}
        while True:
            states = self._state_map()
            updates: dict[str, tuple[ProofNodeState, str]] = {}
            for step_id in self.plan.topological_step_ids:
                state = states[step_id]
                if state.terminal or state is ProofNodeState.RUNNING:
                    continue
                step = by_id[step_id]
                dependency_states = [states[item] for item in step.depends_on]
                if not dependency_states:
                    updates[step_id] = (ProofNodeState.READY, "")
                    continue
                if step.dependency_mode == "any":
                    if any(item.dependency_satisfied for item in dependency_states):
                        updates[step_id] = (ProofNodeState.READY, "")
                    elif all(item.terminal for item in dependency_states):
                        unsupported = all(
                            item in {
                                ProofNodeState.UNSUPPORTED,
                                ProofNodeState.BLOCKED,
                                ProofNodeState.CANCELLED,
                            }
                            for item in dependency_states
                        )
                        updates[step_id] = (
                            ProofNodeState.UNSUPPORTED
                            if unsupported
                            else ProofNodeState.BLOCKED,
                            "no_dependency_alternative_succeeded",
                        )
                elif all(item.dependency_satisfied for item in dependency_states):
                    updates[step_id] = (ProofNodeState.READY, "")
                else:
                    failed = [
                        (dependency, states[dependency])
                        for dependency in step.depends_on
                        if states[dependency].terminal
                        and not states[dependency].dependency_satisfied
                    ]
                    if failed:
                        unsupported = all(
                            item
                            in {
                                ProofNodeState.UNSUPPORTED,
                                ProofNodeState.BLOCKED,
                                ProofNodeState.CANCELLED,
                            }
                            for _, item in failed
                        )
                        reason = "unsupported_dependency:" if unsupported else "blocked_dependency:"
                        updates[step_id] = (
                            ProofNodeState.UNSUPPORTED
                            if unsupported
                            else ProofNodeState.BLOCKED,
                            reason + ",".join(item for item, _ in failed),
                        )
            changed = {
                key: value
                for key, value in updates.items()
                if states.get(key) != value[0]
            }
            if not changed:
                break
            self._store.update_nodes(self.plan.plan_id, changed)
        return self.snapshot()

    def ready_steps(
        self,
        stages: Iterable[ProofStage | str] | ProofStage | str | None = None,
        *,
        phase: int | None = None,
    ) -> tuple[ScheduledProofStep, ...]:
        """Return ready work, optionally limited to proof stages or one phase."""

        selected_stages = self._normalize_stages(stages)
        states = self._state_map()
        priorities = self.priorities()
        by_id = {step.step_id: step for step in self.plan.steps}
        ready = [
            ScheduledProofStep(by_id[step_id], priorities[step_id])
            for step_id, state in states.items()
            if state is ProofNodeState.READY
            and (
                selected_stages is None
                or by_id[step_id].stage in selected_stages
            )
            and (
                phase is None
                or self.proof_phase(by_id[step_id].stage) == phase
            )
        ]
        return tuple(sorted(ready, key=lambda item: item.priority.sort_key))

    def snapshot(self) -> ProofScheduleSnapshot:
        rows = self._store.node_rows(self.plan.plan_id)
        leases = self._store.lease_snapshots(self.plan.plan_id)
        active_leases = {
            lease.step_id: lease for lease in leases if lease.active
        }
        steps = {step.step_id: step for step in self.plan.steps}
        nodes = tuple(
            ProofNodeSnapshot(
                step_id=str(row["step_id"]),
                state=ProofNodeState(str(row["state"])),
                reason_code=str(row["reason_code"]),
                attempt_id=str(row["attempt_id"]),
                cancellation_requested=bool(row["cancellation_requested"]),
                stage=steps[str(row["step_id"])].stage.value,
                resource_class=normalize_resource_class(
                    steps[str(row["step_id"])].resource_class,
                    stage=steps[str(row["step_id"])].stage,
                ),
                dependency_step_ids=steps[str(row["step_id"])].depends_on,
                lease_owner_id=(
                    active_leases[str(row["step_id"])].owner_id
                    if str(row["step_id"]) in active_leases
                    else ""
                ),
                fencing_token=(
                    active_leases[str(row["step_id"])].fencing_token
                    if str(row["step_id"]) in active_leases
                    else 0
                ),
                lease_expires_at_ms=(
                    active_leases[str(row["step_id"])].expires_at_ms
                    if str(row["step_id"]) in active_leases
                    else 0
                ),
            )
            for row in rows
        )
        return ProofScheduleSnapshot(
            plan_id=self.plan.plan_id,
            nodes=nodes,
            ready=self.ready_steps(),
            active_leases=self._store.active_lease_count(self.plan.plan_id),
            attempts=self._store.attempts(self.plan.plan_id),
            receipts=self._store.receipts(self.plan.plan_id),
            leases=leases,
        )

    def cancel(
        self,
        reason: str = "scheduler_cancelled",
        *,
        stages: Iterable[ProofStage | str] | ProofStage | str | None = None,
    ) -> tuple[str, ...]:
        """Request cooperative cancellation of unfinished selected nodes."""

        reason_code = str(reason or "scheduler_cancelled").strip()
        selected_stages = self._normalize_stages(stages)
        by_id = {step.step_id: step for step in self.plan.steps}
        with self._cancel_lock:
            if selected_stages is None:
                self._cancelled = True
            for step_id, token in self._tokens.items():
                if (
                    selected_stages is None
                    or by_id[step_id].stage in selected_stages
                ):
                    token.cancel()
        states = self._state_map()
        return self._store.cancel_steps(
            self.plan.plan_id,
            [
                step_id
                for step_id, state in states.items()
                if not state.terminal
                and (
                    selected_stages is None
                    or by_id[step_id].stage in selected_stages
                )
            ],
            reason_code,
        )

    request_cancel = cancel

    def _dependency_context(self, step: ProofPlanStep) -> tuple[tuple[ProofAttempt, ...], tuple[ProofReceipt, ...]]:
        return (
            self._store.attempts(self.plan.plan_id, step.depends_on),
            self._store.receipts(
                self.plan.plan_id,
                tuple(
                    item.obligation_id
                    for item in self.plan.steps
                    if item.step_id in step.depends_on
                ),
            ),
        )

    @staticmethod
    def _call_executor(
        callback: Callable[..., Any],
        step: ProofPlanStep,
        context: ProofExecutionContext,
    ) -> Any:
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            result = callback(step, context)
        else:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            has_variadic = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if has_variadic or len(positional) >= 2:
                result = callback(step, context)
            elif len(positional) == 1:
                parameter_name = positional[0].name.lower()
                result = (
                    callback(step)
                    if parameter_name in {"step", "node", "proof_step", "plan_step"}
                    else callback(context)
                )
            else:
                result = callback()
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    @staticmethod
    def _coerce_result(value: Any) -> ProofStepResult:
        if isinstance(value, ProofStepResult):
            return value
        if isinstance(value, ProofReceipt):
            return ProofStepResult(receipts=(value,))
        if isinstance(value, ProofAttempt):
            return ProofStepResult(status=value.status, attempt=value)
        if isinstance(value, ProviderResponse):
            if not value.ok:
                assert value.error is not None
                raise ProofProviderError(
                    value.error.code,
                    value.error.message,
                    retryable=value.error.retryable,
                    details=value.error.details,
                )
            return ProofScheduler._coerce_result(value.require_result())
        if value is None:
            return ProofStepResult()
        if isinstance(value, (tuple, list)):
            attempts = [item for item in value if isinstance(item, ProofAttempt)]
            receipts = tuple(item for item in value if isinstance(item, ProofReceipt))
            if len(attempts) > 1 or len(attempts) + len(receipts) != len(value):
                raise TypeError("proof result sequences may contain one attempt and receipts")
            return ProofStepResult(
                status=attempts[0].status if attempts else AttemptStatus.SUCCEEDED,
                attempt=attempts[0] if attempts else None,
                receipts=receipts,
            )
        if isinstance(value, Mapping):
            if value.get("schema") == ProofReceipt.SCHEMA:
                return ProofStepResult(receipts=(ProofReceipt.from_dict(value),))
            if value.get("schema") == ProofAttempt.SCHEMA:
                attempt = ProofAttempt.from_dict(value)
                return ProofStepResult(status=attempt.status, attempt=attempt)
            raw_receipts = value.get("receipts")
            if raw_receipts is None and value.get("receipt") is not None:
                raw_receipts = (value["receipt"],)
            receipts = tuple(
                item if isinstance(item, ProofReceipt) else ProofReceipt.from_dict(item)
                for item in (raw_receipts or ())
            )
            raw_attempt = value.get("attempt")
            attempt = (
                raw_attempt
                if isinstance(raw_attempt, ProofAttempt)
                else ProofAttempt.from_dict(raw_attempt)
                if isinstance(raw_attempt, Mapping)
                else None
            )
            raw_status = value.get(
                "status",
                attempt.status if attempt is not None else AttemptStatus.SUCCEEDED,
            )
            if not isinstance(raw_status, AttemptStatus):
                status_text = str(
                    getattr(raw_status, "value", raw_status)
                ).strip().lower()
                aliases = {
                    "accepted": AttemptStatus.SUCCEEDED,
                    "candidate": AttemptStatus.SUCCEEDED,
                    "complete": AttemptStatus.SUCCEEDED,
                    "completed": AttemptStatus.SUCCEEDED,
                    "disproved": AttemptStatus.SUCCEEDED,
                    "proved": AttemptStatus.SUCCEEDED,
                    "rejected": AttemptStatus.SUCCEEDED,
                    "success": AttemptStatus.SUCCEEDED,
                    "error": AttemptStatus.FAILED,
                    "unavailable": AttemptStatus.UNAVAILABLE,
                    "timeout": AttemptStatus.TIMED_OUT,
                    "canceled": AttemptStatus.CANCELLED,
                }
                raw_status = aliases.get(status_text, status_text)
            if "returncode" in value and int(value.get("returncode") or 0) != 0:
                raw_status = (
                    AttemptStatus.TIMED_OUT
                    if value.get("timed_out")
                    else AttemptStatus.FAILED
                )
            raw_output_ids = value.get("output_ids") or ()
            output_ids = (
                list(raw_output_ids)
                if isinstance(raw_output_ids, Sequence)
                and not isinstance(raw_output_ids, (str, bytes, bytearray))
                else [raw_output_ids]
            )
            for key in (
                "artifact_id",
                "candidate_id",
                "reconstruction_id",
                "translation_id",
            ):
                identifier = value.get(key)
                if isinstance(identifier, str) and identifier.strip():
                    output_ids.append(identifier.strip())
            return ProofStepResult(
                status=raw_status,
                attempt=attempt,
                receipts=receipts,
                output_ids=tuple(output_ids),
                error_code=str(value.get("error_code") or ""),
                error_message=str(value.get("error_message") or ""),
                metadata=value.get("metadata") or {},
            )
        if isinstance(value, AttemptStatus):
            return ProofStepResult(status=value)
        if isinstance(value, str):
            return ProofStepResult(status=AttemptStatus(value))
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return ProofScheduler._coerce_result(converted)
        raise TypeError(f"unsupported proof executor result: {type(value).__name__}")

    def _running_attempt(
        self, step: ProofPlanStep, lease: _Lease, context: ProofExecutionContext
    ) -> ProofAttempt:
        return ProofAttempt(
            plan_id=self.plan.plan_id,
            step_id=step.step_id,
            obligation_id=step.obligation_id,
            repository_tree_id=self.plan.repository_tree_id,
            provider_id=step.provider_id,
            stage=step.stage,
            status=AttemptStatus.RUNNING,
            input_ids=context.inputs,
            started_at=_utc_now(),
            metadata={
                "scheduler_owner_id": self.owner_id,
                "fencing_token": lease.fencing_token,
                **(
                    {
                        "resource_lease_id": context.resource_lease.lease_id,
                        "resource_class": context.resource_lease.resource_class,
                        "resource_pool": context.resource_lease.resource_pool,
                        "child_resource_limits": context.child_resource_limits,
                    }
                    if context.resource_lease is not None
                    else {}
                ),
            },
        )

    def _final_attempt(
        self,
        step: ProofPlanStep,
        running: ProofAttempt,
        result: ProofStepResult,
    ) -> ProofAttempt:
        if result.attempt is not None:
            attempt = result.attempt
            if not attempt.status.terminal:
                raise ContractValidationError(
                    "executor attempt must have a terminal status"
                )
            if attempt.status is not result.status:
                raise ContractValidationError(
                    "executor attempt status does not match step result status"
                )
            bindings = (
                (attempt.plan_id, self.plan.plan_id, "plan_id"),
                (attempt.step_id, step.step_id, "step_id"),
                (attempt.obligation_id, step.obligation_id, "obligation_id"),
                (
                    attempt.repository_tree_id,
                    self.plan.repository_tree_id,
                    "repository_tree_id",
                ),
                (attempt.provider_id, step.provider_id, "provider_id"),
                (attempt.stage, step.stage, "stage"),
            )
            for actual, expected, name in bindings:
                if actual != expected:
                    raise ContractValidationError(
                        f"executor attempt {name} does not match scheduled step"
                    )
            return attempt
        outputs = set(result.output_ids)
        outputs.update(receipt.receipt_id for receipt in result.receipts)
        return ProofAttempt(
            plan_id=self.plan.plan_id,
            step_id=step.step_id,
            obligation_id=step.obligation_id,
            repository_tree_id=self.plan.repository_tree_id,
            provider_id=step.provider_id,
            stage=step.stage,
            status=result.status,
            input_ids=running.input_ids,
            output_ids=tuple(outputs),
            started_at=running.started_at,
            finished_at=_utc_now(),
            error_code=result.error_code,
            error_message=result.error_message,
            metadata={**running.metadata, **dict(result.metadata)},
        )

    def _prepare_invocation(
        self,
        step: ProofPlanStep,
        lease: _Lease,
        token: CancellationToken,
        resource_lease: ResourceAdmissionLease | None,
    ) -> tuple[ProofExecutionContext, ProofAttempt]:
        dependency_attempts, dependency_receipts = self._dependency_context(step)
        context = ProofExecutionContext(
            plan=self.plan,
            step=step,
            cancellation_token=token,
            dependency_attempts=dependency_attempts,
            dependency_receipts=dependency_receipts,
            fencing_token=lease.fencing_token,
            owner_id=self.owner_id,
            resource_lease=resource_lease,
            _heartbeat=lambda: self._store.heartbeat(
                self.plan.plan_id, lease, self.config.lease_seconds
            ),
        )
        running = self._running_attempt(step, lease, context)
        self._store.set_attempt(self.plan.plan_id, step.step_id, running)
        return context, running

    def _invoke(
        self,
        step: ProofPlanStep,
        lease: _Lease,
        token: CancellationToken,
        resource_lease: ResourceAdmissionLease | None,
        prepared: tuple[ProofExecutionContext, ProofAttempt] | None = None,
    ) -> tuple[ProofStepResult, ProofAttempt]:
        context, running = (
            prepared
            if prepared is not None
            else self._prepare_invocation(step, lease, token, resource_lease)
        )
        if token.cancelled:
            result = ProofStepResult(
                status=AttemptStatus.CANCELLED,
                error_code="cancelled_before_start",
            )
            return result, self._final_attempt(step, running, result)
        try:
            callback = self._executor_for(step)
            result = self._coerce_result(self._call_executor(callback, step, context))
            final_attempt = self._final_attempt(step, running, result)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure_code = exc.code if isinstance(exc, ProofProviderError) else None
            cancelled = (
                token.cancelled or failure_code is ProviderFailureCode.CANCELLED
            )
            if cancelled:
                status = AttemptStatus.CANCELLED
            elif failure_code in {
                ProviderFailureCode.UNAVAILABLE,
                ProviderFailureCode.UNSUPPORTED,
            }:
                status = AttemptStatus.UNSUPPORTED
            elif failure_code is ProviderFailureCode.TIMED_OUT:
                status = AttemptStatus.TIMED_OUT
            else:
                status = AttemptStatus.FAILED
            result = ProofStepResult(
                status=status,
                error_code=(
                    failure_code.value
                    if failure_code is not None
                    else "cancelled"
                    if cancelled
                    else "executor_exception"
                ),
                error_message=f"{type(exc).__name__}: {exc}",
            )
            final_attempt = self._final_attempt(step, running, result)
        return result, final_attempt

    def _portfolio_cancel(
        self,
        winner: ProofPlanStep,
        receipt: ProofReceipt,
    ) -> tuple[str, ...]:
        if not winner.portfolio_id or not receipt.authoritative_verdict.conclusive:
            return ()
        sibling_ids = [
            step.step_id
            for step in self.plan.steps
            if step.step_id != winner.step_id
            and step.portfolio_id == winner.portfolio_id
            and step.obligation_id == winner.obligation_id
        ]
        cancelled = self._store.cancel_steps(
            self.plan.plan_id,
            sibling_ids,
            f"portfolio_concluded:{winner.step_id}:{receipt.authoritative_verdict.value}",
        )
        with self._cancel_lock:
            for step_id in cancelled:
                token = self._tokens.get(step_id)
                if token is not None:
                    token.cancel()
        return cancelled

    def _finalize(
        self,
        step: ProofPlanStep,
        lease: _Lease,
        result: ProofStepResult,
        attempt: ProofAttempt,
    ) -> bool:
        state = _STATUS_TO_STATE[attempt.status]
        committed, actual_state, stored_receipts = self._store.commit_result(
            self.plan,
            step,
            lease,
            attempt=attempt,
            receipts=result.receipts,
            state=state,
            reason_code=result.error_code,
        )
        if committed and actual_state is ProofNodeState.SUCCEEDED:
            for receipt in stored_receipts:
                if receipt.authoritative_verdict.conclusive:
                    self._portfolio_cancel(step, receipt)
                    break
        return committed

    def _sync_cancellation_tokens(self) -> None:
        snapshots = {item.step_id: item for item in self.snapshot().nodes}
        with self._cancel_lock:
            for step_id, token in self._tokens.items():
                node = snapshots.get(step_id)
                if node is not None and node.cancellation_requested:
                    token.cancel()

    def run(
        self,
        *,
        timeout_seconds: float | None = None,
        stages: Iterable[ProofStage | str] | ProofStage | str | None = None,
    ) -> ProofScheduleResult:
        """Run all work, or only the requested proof stages.

        Leases are extended while local callbacks run.  Other scheduler
        processes can execute remaining ready nodes against the same database;
        transactional claims preserve the plan-wide ``max_parallel`` bound.

        Supplying ``stages`` is the durable partial-execution API used by the
        validation scheduler.  It enforces canonical phase prerequisites and
        returns as soon as the selected nodes are terminal, leaving later
        nodes pending or ready for a subsequent call.  With no ``stages`` and
        no configured barriers, execution retains the original dependency-only
        behavior.
        """

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        selected_stages = self._normalize_stages(stages)
        enforce_barriers = self.config.stage_barriers or selected_stages is not None
        if enforce_barriers:
            self._validate_stage_barrier_dependencies()
        started = time.monotonic()
        futures: dict[
            Future[tuple[ProofStepResult, ProofAttempt]],
            tuple[
                ProofPlanStep,
                _Lease,
                CancellationToken,
                ResourceAdmissionLease,
            ],
        ] = {}
        with ThreadPoolExecutor(
            max_workers=self.max_parallel,
            thread_name_prefix="proof-plan",
        ) as pool:
            while True:
                self.reconcile()
                self._sync_cancellation_tokens()
                snapshot = self.snapshot()
                if selected_stages is not None:
                    selected_complete = self._selected_nodes_complete(
                        snapshot, selected_stages
                    )
                    if selected_complete and not futures:
                        break
                elif snapshot.complete and not futures:
                    break
                if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                    self.cancel(
                        "scheduler_timeout",
                        stages=selected_stages,
                    )
                    # Reconcile the cancellation before considering any READY
                    # nodes from the snapshot captured before the timeout.
                    continue

                eligible_phase = (
                    self._eligible_phase(snapshot, selected_stages)
                    if enforce_barriers
                    else None
                )
                if selected_stages is not None and eligible_phase is not None:
                    by_id = {step.step_id: step for step in self.plan.steps}
                    unfinished_prerequisites = [
                        node.step_id
                        for node in snapshot.nodes
                        if not node.state.terminal
                        and by_id[node.step_id].stage not in selected_stages
                        and self.proof_phase(by_id[node.step_id].stage)
                        < eligible_phase
                    ]
                    if unfinished_prerequisites and not futures:
                        raise ContractValidationError(
                            "selected proof stages cannot run before unfinished "
                            "earlier phases: "
                            + ", ".join(sorted(unfinished_prerequisites))
                        )

                ready = self.ready_steps(
                    selected_stages,
                    phase=eligible_phase if enforce_barriers else None,
                )
                if selected_stages is not None and not ready and not futures:
                    # Another process may have claimed a node between the
                    # snapshot above and this ready query.  Refresh before
                    # diagnosing a dependency deadlock, and wait when durable
                    # lease ownership demonstrates legitimate peer progress.
                    refreshed = self.snapshot()
                    refreshed_phase = (
                        self._eligible_phase(refreshed, selected_stages)
                        if enforce_barriers
                        else None
                    )
                    refreshed_ready = self.ready_steps(
                        selected_stages,
                        phase=(
                            refreshed_phase if enforce_barriers else None
                        ),
                    )
                    if refreshed_ready:
                        continue
                    selected_unfinished = [
                        node.step_id
                        for node in refreshed.nodes
                        if not node.state.terminal
                        and next(
                            step.stage
                            for step in self.plan.steps
                            if step.step_id == node.step_id
                        )
                        in selected_stages
                    ]
                    if selected_unfinished and (
                        refreshed.active_leases > 0
                        or self._store.active_lease_count(
                            self.plan.plan_id
                        ) > 0
                        or any(
                            node.state is ProofNodeState.RUNNING
                            and node.step_id in selected_unfinished
                            for node in refreshed.nodes
                        )
                    ):
                        time.sleep(self.config.poll_interval_seconds)
                        continue
                    if selected_unfinished:
                        raise ContractValidationError(
                            "selected proof stages made no progress; unfinished "
                            "nodes are not ready: "
                            + ", ".join(sorted(selected_unfinished))
                        )

                admitted: list[
                    tuple[
                        ProofPlanStep,
                        _Lease,
                        CancellationToken,
                        ResourceAdmissionLease,
                    ]
                ] = []
                for scheduled in ready:
                    if len(futures) + len(admitted) >= self.max_parallel:
                        break
                    with self._cancel_lock:
                        if self._cancelled:
                            break
                    running_steps = [
                        running_step
                        for running_step, _lease, _token, _resource_lease
                        in futures.values()
                    ] + [item[0] for item in admitted]
                    stage_limit = self.config.stage_limits.get(
                        scheduled.step.stage.value
                    )
                    if stage_limit is not None and sum(
                        item.stage is scheduled.step.stage
                        for item in running_steps
                    ) >= stage_limit:
                        continue
                    class_limit = self.config.resource_limits.get(
                        scheduled.step.resource_class
                    )
                    if class_limit is not None and sum(
                        item.resource_class == scheduled.step.resource_class
                        for item in running_steps
                    ) >= class_limit:
                        continue
                    scheduled_pool = resource_pool(
                        normalize_resource_class(
                            scheduled.step.resource_class,
                            stage=scheduled.step.stage,
                        )
                    )
                    if sum(
                        resource_pool(
                            normalize_resource_class(
                                item.resource_class,
                                stage=item.stage,
                            )
                        )
                        == scheduled_pool
                        for item in running_steps
                    ) >= self._pool_limits[scheduled_pool]:
                        continue
                    _decision, resource_lease = self._acquire_resource(
                        scheduled.step
                    )
                    if resource_lease is None:
                        continue
                    lease = self._store.claim(
                        self.plan,
                        scheduled.step,
                        owner_id=self.owner_id,
                        lease_seconds=self.config.lease_seconds,
                        max_parallel=self.max_parallel,
                        stage_limits=self.config.stage_limits,
                        resource_limits=self.config.resource_limits,
                        pool_limits=self._pool_limits,
                        stage_barrier_phase=(
                            eligible_phase if enforce_barriers else None
                        ),
                    )
                    if lease is None:
                        self.resource_scheduler.release(resource_lease)
                        continue
                    token = CancellationToken()
                    with self._cancel_lock:
                        self._tokens[scheduled.step_id] = token
                    admitted.append(
                        (scheduled.step, lease, token, resource_lease)
                    )

                # Claim the whole ready batch before callbacks can finish and
                # release their leases. This keeps short independent proof
                # steps parallel even when opening durable state is slower
                # than the callbacks themselves.
                prepared_admissions = []
                try:
                    for step, lease, token, resource_lease in admitted:
                        prepared_admissions.append(
                            (
                                step,
                                lease,
                                token,
                                resource_lease,
                                self._prepare_invocation(
                                    step,
                                    lease,
                                    token,
                                    resource_lease,
                                ),
                            )
                        )
                except BaseException:
                    # No callbacks have been submitted yet, so return the
                    # complete admission batch to READY and release every
                    # local resource grant before propagating the error.
                    for step, lease, token, resource_lease in admitted:
                        token.cancel()
                        with self._cancel_lock:
                            self._tokens.pop(step.step_id, None)
                        try:
                            self._store.finish(
                                self.plan.plan_id,
                                lease,
                                state=ProofNodeState.READY,
                                attempt_id="",
                                reason_code="preparation_aborted",
                            )
                        except Exception:
                            # Preserve the preparation error. Expiry recovery
                            # remains the fallback if durable cleanup fails.
                            pass
                        finally:
                            self.resource_scheduler.release(resource_lease)
                    raise
                for (
                    step,
                    lease,
                    token,
                    resource_lease,
                    prepared,
                ) in prepared_admissions:
                    future = pool.submit(
                        self._invoke,
                        step,
                        lease,
                        token,
                        resource_lease,
                        prepared,
                    )
                    futures[future] = (
                        step,
                        lease,
                        token,
                        resource_lease,
                    )

                for future, (_, lease, _, _) in tuple(futures.items()):
                    self._store.heartbeat(
                        self.plan.plan_id, lease, self.config.lease_seconds
                    )

                if futures:
                    completed, _ = wait(
                        tuple(futures),
                        timeout=self.config.poll_interval_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        step, lease, _, resource_lease = futures.pop(future)
                        with self._cancel_lock:
                            self._tokens.pop(step.step_id, None)
                        try:
                            result, attempt = future.result()
                        except BaseException:
                            # KeyboardInterrupt/SystemExit from a callback is
                            # intentionally not converted inside the worker.
                            self.cancel("scheduler_interrupted")
                            raise
                        finally:
                            self.resource_scheduler.release(resource_lease)
                        self._finalize(step, lease, result, attempt)
                else:
                    time.sleep(self.config.poll_interval_seconds)

        final = self.reconcile()
        return ProofScheduleResult(self.plan, final)

    execute = run
    resume = run

    def run_stages(
        self,
        stages: Iterable[ProofStage | str] | ProofStage | str,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofScheduleResult:
        """Compatibility-friendly spelling for durable partial execution."""

        return self.run(timeout_seconds=timeout_seconds, stages=stages)


def execute_proof_plan(
    plan: ProofPlan | Mapping[str, Any],
    executor: Callable[..., Any] | Mapping[Any, Callable[..., Any]],
    **scheduler_options: Any,
) -> ProofScheduleResult:
    """Convenience entry point for one complete proof-plan execution."""

    run_options = {}
    if "timeout_seconds" in scheduler_options:
        run_options["timeout_seconds"] = scheduler_options.pop("timeout_seconds")
    if "stages" in scheduler_options:
        run_options["stages"] = scheduler_options.pop("stages")
    return ProofScheduler(plan, executor, **scheduler_options).run(**run_options)


run_proof_plan = execute_proof_plan


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_PROOF_LEASE_SECONDS",
    "PROOF_SCHEDULER_SCHEMA",
    "STAGED_PROOF_PHASES",
    "ProofExecutionContext",
    "ProofLeaseSnapshot",
    "ProofNodeSnapshot",
    "ProofNodeState",
    "ProofScheduleResult",
    "ProofScheduleSnapshot",
    "ProofScheduler",
    "ProofSchedulerConfig",
    "ProofStepPriority",
    "ProofStepResult",
    "ProofStepState",
    "ScheduledProofStep",
    "StepState",
    "execute_proof_plan",
    "run_proof_plan",
]
