"""Prompt-v3 run monitor contracts and joined RUNNING evidence (ASE3-008).

Detection never grants restart authority. A monitor cannot attest to its own
guardian. Client sessions never own monitor liveness.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

# Policy constants from the ASE3-008 acceptance criteria.
HEARTBEAT_INTERVAL_MS: Final = 5_000
STALE_HEARTBEAT_MS: Final = 30_000
SEMANTIC_PROGRESS_BUDGET_MS: Final = 300_000
MAX_CANARY_RECOVERIES: Final = 3
CANARY_WINDOW_MS: Final = 30 * 60 * 1000

RUN_MONITOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/run-monitor@1"
)


class StallClass(str, Enum):
    HEALTHY = "healthy"
    STALE_HEARTBEAT = "stale_heartbeat"
    DEAD_PROCESS = "dead_process"
    PID_REUSE = "pid_reuse"
    FROZEN_PROGRESS = "frozen_progress"
    FALSE_IDLE = "false_idle"
    SOFT_COMPLETE = "soft_complete"
    MISSING_JOIN = "missing_join"
    UNKNOWN_OUTCOME = "unknown_outcome"


class RecoveryAction(str, Enum):
    NONE = "none"
    ADOPT = "adopt"
    RESTART = "restart"
    RESCUE = "rescue"
    OPERATOR = "operator"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class SemanticProgressClock:
    """Immutable phase/cursor movement is progress; log noise is not."""

    phase: str
    cursor_cid: str
    observed_at_ms: int
    source: str = "configured_board_scheduler"

    def __post_init__(self) -> None:
        if not self.phase or not self.cursor_cid or self.observed_at_ms <= 0:
            raise ValueError("semantic progress clock is incomplete")
        if self.source in {"log", "stdout", "noise"}:
            raise ValueError("log noise is not semantic progress")

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {"schema": RUN_MONITOR_SCHEMA + "/semantic-progress@1", **asdict(self)}
        )


@dataclass(frozen=True)
class SemanticProgressCursorVector:
    run_id: str
    sequence: int
    predecessor_cid: str
    cursor_cid: str
    phase: str

    @property
    def content_id(self) -> str:
        return self.cursor_cid


@dataclass(frozen=True)
class ProcessEvidence:
    role: str  # lifecycle | monitor
    process_cid: str
    process_birth_identity: str
    lease_id: str
    fencing_generation: int
    heartbeat_at_ms: int
    event_cursor: str
    generation: int
    healthy: bool

    def __post_init__(self) -> None:
        if self.role not in {"lifecycle", "monitor"}:
            raise ValueError("process evidence role must be lifecycle or monitor")
        if (
            not self.process_cid
            or not self.process_birth_identity
            or not self.lease_id
            or self.fencing_generation < 1
            or self.generation < 1
        ):
            raise ValueError(f"{self.role} process evidence is incomplete")


@dataclass(frozen=True)
class RunHealthSnapshot:
    run_id: str
    run_revision: int
    lifecycle: ProcessEvidence
    monitor: ProcessEvidence
    semantic_progress: SemanticProgressClock | None
    tree_reachable: bool
    observed_at_ms: int
    provider_healthy: bool = True
    resource_healthy: bool = True

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": RUN_MONITOR_SCHEMA + "/health-snapshot@1",
                "run_id": self.run_id,
                "run_revision": self.run_revision,
                "lifecycle": asdict(self.lifecycle),
                "monitor": asdict(self.monitor),
                "semantic_progress": (
                    None
                    if self.semantic_progress is None
                    else asdict(self.semantic_progress)
                ),
                "tree_reachable": self.tree_reachable,
                "observed_at_ms": self.observed_at_ms,
                "provider_healthy": self.provider_healthy,
                "resource_healthy": self.resource_healthy,
            }
        )


@dataclass(frozen=True)
class JoinedRunningEvidence:
    """Same-revision join required for RUNNING."""

    snapshot: RunHealthSnapshot
    joined: bool
    reason_codes: tuple[str, ...]

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": RUN_MONITOR_SCHEMA + "/joined-running@1",
                "snapshot_cid": self.snapshot.content_id,
                "joined": self.joined,
                "reason_codes": list(self.reason_codes),
            }
        )


@dataclass(frozen=True)
class StallClassifier:
    """Classify stalls; never grants restart authority by itself."""

    heartbeat_stale_ms: int = STALE_HEARTBEAT_MS
    progress_budget_ms: int = SEMANTIC_PROGRESS_BUDGET_MS

    def classify(
        self,
        snapshot: RunHealthSnapshot,
        *,
        now_ms: int | None = None,
        pid_alive: bool = True,
        birth_matches: bool = True,
    ) -> StallClass:
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        if not birth_matches:
            return StallClass.PID_REUSE
        if not pid_alive or not snapshot.monitor.healthy:
            return StallClass.DEAD_PROCESS
        if clock - snapshot.monitor.heartbeat_at_ms >= self.heartbeat_stale_ms:
            return StallClass.STALE_HEARTBEAT
        if snapshot.semantic_progress is None:
            return StallClass.FALSE_IDLE
        if clock - snapshot.semantic_progress.observed_at_ms >= self.progress_budget_ms:
            return StallClass.FROZEN_PROGRESS
        if snapshot.lifecycle.state if hasattr(snapshot.lifecycle, "state") else False:
            pass
        return StallClass.HEALTHY


def join_running_evidence(
    snapshot: RunHealthSnapshot,
    *,
    now_ms: int | None = None,
    require_same_revision: bool = True,
) -> JoinedRunningEvidence:
    """RUNNING requires one same-revision join of lifecycle + monitor evidence."""

    clock = int(now_ms if now_ms is not None else time.time() * 1000)
    reasons: list[str] = []
    life = snapshot.lifecycle
    mon = snapshot.monitor

    if life.role != "lifecycle" or mon.role != "monitor":
        reasons.append("role_mismatch")
    if not life.process_cid or not mon.process_cid:
        reasons.append("missing_process_cid")
    if not life.process_birth_identity or not mon.process_birth_identity:
        reasons.append("missing_birth_identity")
    if not life.lease_id or not mon.lease_id:
        reasons.append("missing_lease")
    if life.fencing_generation < 1 or mon.fencing_generation < 1:
        reasons.append("missing_fence")
    if clock - life.heartbeat_at_ms >= STALE_HEARTBEAT_MS:
        reasons.append("lifecycle_heartbeat_stale")
    if clock - mon.heartbeat_at_ms >= STALE_HEARTBEAT_MS:
        reasons.append("monitor_heartbeat_stale")
    if not life.event_cursor or not mon.event_cursor:
        reasons.append("missing_event_cursor")
    if not snapshot.tree_reachable:
        reasons.append("tree_unreachable")
    if not life.healthy or not mon.healthy:
        reasons.append("unhealthy_component")
    # Self-attestation forbidden: monitor birth must differ from lifecycle birth.
    if mon.process_birth_identity == life.process_birth_identity:
        reasons.append("monitor_self_attestation")
    if mon.process_cid == life.process_cid:
        reasons.append("monitor_self_attestation_process")

    joined = not reasons
    if joined:
        reasons = ("joined_running",)
    return JoinedRunningEvidence(
        snapshot=snapshot, joined=joined, reason_codes=tuple(reasons)
    )


@dataclass(frozen=True)
class RecoveryPolicy:
    max_canary_recoveries: int = MAX_CANARY_RECOVERIES
    canary_window_ms: int = CANARY_WINDOW_MS
    allow_restart: bool = True

    def authorize(
        self,
        stall: StallClass,
        *,
        recoveries_in_window: int,
        authorized_callback: bool,
    ) -> RecoveryAction:
        if not authorized_callback:
            # Detection never implies restart authority.
            return RecoveryAction.OPERATOR if stall is not StallClass.HEALTHY else RecoveryAction.NONE
        if stall is StallClass.HEALTHY:
            return RecoveryAction.NONE
        if stall is StallClass.UNKNOWN_OUTCOME:
            return RecoveryAction.OPERATOR
        if recoveries_in_window >= self.max_canary_recoveries:
            return RecoveryAction.OPERATOR
        if stall in {StallClass.DEAD_PROCESS, StallClass.STALE_HEARTBEAT}:
            return RecoveryAction.RESTART if self.allow_restart else RecoveryAction.ADOPT
        if stall is StallClass.PID_REUSE:
            return RecoveryAction.RESCUE
        if stall in {StallClass.FROZEN_PROGRESS, StallClass.FALSE_IDLE}:
            return RecoveryAction.ADOPT
        if stall is StallClass.SOFT_COMPLETE:
            return RecoveryAction.SHUTDOWN
        return RecoveryAction.OPERATOR


@dataclass(frozen=True)
class RecoveryReceipt:
    run_id: str
    stall: str
    action: str
    authorized: bool
    generation: int
    reason_codes: tuple[str, ...]
    recovered_at_ms: int

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": RUN_MONITOR_SCHEMA + "/recovery@1",
                "run_id": self.run_id,
                "stall": self.stall,
                "action": self.action,
                "authorized": self.authorized,
                "generation": self.generation,
                "reason_codes": list(self.reason_codes),
                "recovered_at_ms": self.recovered_at_ms,
            }
        )


@dataclass(frozen=True)
class SupervisorDoctorService:
    """Bounded doctor integration surface for monitor recovery."""

    service_id: str = "supervisor-doctor"

    def diagnose(self, stall: StallClass, snapshot: RunHealthSnapshot) -> tuple[str, ...]:
        codes = [f"stall:{stall.value}", f"run:{snapshot.run_id}"]
        if not snapshot.tree_reachable:
            codes.append("tree_unreachable")
        if stall is StallClass.UNKNOWN_OUTCOME:
            codes.append("unknown_outcome_no_replay")
        return tuple(codes)


@dataclass(frozen=True)
class ReviewedHostNamespaceReconciler:
    """Reviewed guardian that alone may start/adopt the durable monitor."""

    guardian_identity: str
    host_namespace: str
    review_cid: str

    def __post_init__(self) -> None:
        if not self.guardian_identity or not self.host_namespace or not self.review_cid:
            raise ValueError("guardian review binding is incomplete")
        if self.guardian_identity in {"cli", "mcp", "python-client", "monitor"}:
            raise ValueError("client or monitor cannot act as host-namespace guardian")

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": RUN_MONITOR_SCHEMA + "/guardian@1",
                **asdict(self),
            }
        )

    def authorize_monitor_start(self, *, requester: str) -> bool:
        return requester == self.guardian_identity


@dataclass(frozen=True)
class MonitorAdoptionReceipt:
    run_id: str
    process_cid: str
    process_birth_identity: str
    generation: int
    guardian_cid: str
    adopted: bool
    created_at_ms: int

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(
            {
                "schema": RUN_MONITOR_SCHEMA + "/adoption@1",
                **asdict(self),
            }
        )


__all__ = [
    "CANARY_WINDOW_MS",
    "HEARTBEAT_INTERVAL_MS",
    "MAX_CANARY_RECOVERIES",
    "MonitorAdoptionReceipt",
    "ProcessEvidence",
    "RecoveryAction",
    "RecoveryPolicy",
    "RecoveryReceipt",
    "ReviewedHostNamespaceReconciler",
    "RUN_MONITOR_SCHEMA",
    "RunHealthSnapshot",
    "JoinedRunningEvidence",
    "SEMANTIC_PROGRESS_BUDGET_MS",
    "STALE_HEARTBEAT_MS",
    "SemanticProgressClock",
    "SemanticProgressCursorVector",
    "StallClass",
    "StallClassifier",
    "SupervisorDoctorService",
    "join_running_evidence",
]
