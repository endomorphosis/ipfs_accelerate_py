"""Typed expert control backend, CLI, and MCP parity helpers.

Mutations call SupervisorControlService-shaped operations.  MCP must not shell
out.  start_training returns training_unavailable without an admitted corpus.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, TrainingAvailability, required_text
from .drift import ExpertState
from .rights import TrainingCorpusAdmission

READ_OPERATIONS: Final[tuple[str, ...]] = (
    "list_experts",
    "get_expert",
    "list_epochs",
    "list_checkpoints",
    "list_drift_events",
)
MUTATION_OPERATIONS: Final[tuple[str, ...]] = (
    "shadow_expert",
    "demote_expert",
    "revoke_expert",
    "start_training",
    "rollback_expert",
)
ALL_OPERATIONS: Final[tuple[str, ...]] = READ_OPERATIONS + MUTATION_OPERATIONS


@dataclass(frozen=True)
class ExpertControlRequest:
    operation: str
    expert_id: str = ""
    dry_run: bool = False
    idempotency_key: str = ""
    admission: TrainingCorpusAdmission | None = None

    def __post_init__(self) -> None:
        op = required_text(self.operation, "operation")
        if op not in ALL_OPERATIONS:
            raise ResidualIntelligenceError(f"unknown expert operation: {op}")
        object.__setattr__(self, "operation", op)
        if type(self.dry_run) is not bool:
            raise ResidualIntelligenceError("dry_run must be boolean")


@dataclass(frozen=True)
class ExpertControlResult:
    operation: str
    ok: bool
    status: str
    audit_id: str
    payload: Mapping[str, Any]


class ResidualExpertControlBackend:
    def __init__(self, *, service_call: Callable[[ExpertControlRequest], Mapping[str, Any]] | None = None) -> None:
        self._service_call = service_call
        self._experts: dict[str, ExpertState] = {}

    def execute(self, request: ExpertControlRequest) -> ExpertControlResult:
        if request.operation == "start_training":
            if request.admission is None or request.admission.admission_decision is not TrainingAvailability.ADMITTED:
                return ExpertControlResult(
                    operation=request.operation,
                    ok=False,
                    status="training_unavailable",
                    audit_id="audit:training-unavailable",
                    payload={"candidate_only": True},
                )
        if request.dry_run:
            return ExpertControlResult(
                operation=request.operation,
                ok=True,
                status="dry_run",
                audit_id=f"audit:dry-run:{request.operation}",
                payload={"candidate_only": True, "applied": False},
            )
        if self._service_call is not None:
            payload = self._service_call(request)
            return ExpertControlResult(
                operation=request.operation,
                ok=bool(payload.get("ok", True)),
                status=str(payload.get("status") or "applied"),
                audit_id=str(payload.get("audit_id") or "audit:service"),
                payload=payload,
            )
        if request.operation == "revoke_expert":
            self._experts[request.expert_id] = ExpertState.REVOKED
        elif request.operation == "demote_expert":
            self._experts[request.expert_id] = ExpertState.DEGRADED
        elif request.operation == "shadow_expert":
            self._experts[request.expert_id] = ExpertState.SHADOW
        return ExpertControlResult(
            operation=request.operation,
            ok=True,
            status="applied",
            audit_id=f"audit:{request.idempotency_key or request.operation}",
            payload={"state": self._experts.get(request.expert_id, ExpertState.CANDIDATE).value},
        )


def register_expert_operations() -> dict[str, tuple[str, ...]]:
    return {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}


def mcp_dispatch(backend: ResidualExpertControlBackend, request: ExpertControlRequest) -> ExpertControlResult:
    """MCP must call the typed backend directly, never a shell."""

    return backend.execute(request)
