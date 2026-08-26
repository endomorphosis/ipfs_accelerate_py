"""Typed residual-expert controls for Python, CLI, and MCP.

Adapters in this module own decoding only.  The canonical
``SupervisorControlService.execute_expert`` method owns every policy decision,
and MCP never converts a request into a shell command or CLI-text round trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TextIO

from ..control.control_plane import (
    EXPERT_CONTROL_OPERATIONS,
    EXPERT_MUTATION_OPERATIONS,
    EXPERT_READ_OPERATIONS,
    SupervisorControlService,
)
from .contracts import ResidualIntelligenceError
from .rights import TrainingCorpusAdmission


READ_OPERATIONS: Final[tuple[str, ...]] = EXPERT_READ_OPERATIONS
MUTATION_OPERATIONS: Final[tuple[str, ...]] = EXPERT_MUTATION_OPERATIONS
ALL_OPERATIONS: Final[tuple[str, ...]] = EXPERT_CONTROL_OPERATIONS
EXPERT_MUTATION_SCOPE: Final[str] = "experts.mutate"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ResidualIntelligenceError(f"{name} must be text")
    result = value.strip()
    if required and not result:
        raise ResidualIntelligenceError(f"{name} must not be empty")
    if len(result.encode("utf-8")) > 2048:
        raise ResidualIntelligenceError(f"{name} exceeds 2048 bytes")
    return result


@dataclass(frozen=True)
class ExpertControlAuthorization:
    """Explicit transport-neutral authority for an expert mutation."""

    subject: str
    permitted: bool
    scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _text(self.subject, "authorization subject"))
        if type(self.permitted) is not bool:
            raise ResidualIntelligenceError("authorization permitted must be boolean")
        if isinstance(self.scopes, (str, bytes)):
            raise ResidualIntelligenceError("authorization scopes must be an array")
        object.__setattr__(self, "scopes", tuple(sorted({_text(item, "authorization scope") for item in self.scopes})))

    @property
    def allows_mutation(self) -> bool:
        return self.permitted and EXPERT_MUTATION_SCOPE in self.scopes

    def to_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "permitted": self.permitted, "scopes": list(self.scopes)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertControlAuthorization":
        if not isinstance(payload, Mapping) or set(payload).difference({"subject", "permitted", "scopes"}):
            raise ResidualIntelligenceError("authorization must be a closed object")
        return cls(payload.get("subject", ""), payload.get("permitted"), tuple(payload.get("scopes") or ()))


@dataclass(frozen=True)
class ExpertControlBudget:
    """Bounded control-plane work; it never grants training resources."""

    max_units: int
    requested_units: int

    def __post_init__(self) -> None:
        for name in ("max_units", "requested_units"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResidualIntelligenceError(f"budget {name} must be a non-negative integer")
        if self.requested_units > self.max_units:
            raise ResidualIntelligenceError("budget requested_units exceeds max_units")

    def to_dict(self) -> dict[str, int]:
        return {"max_units": self.max_units, "requested_units": self.requested_units}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertControlBudget":
        if not isinstance(payload, Mapping) or set(payload) != {"max_units", "requested_units"}:
            raise ResidualIntelligenceError("budget must contain max_units and requested_units")
        return cls(payload["max_units"], payload["requested_units"])


@dataclass(frozen=True)
class ExpertControlRequest:
    """One complete residual-expert request submitted to the control service."""

    operation: str
    expert_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    idempotency_key: str = ""
    authorization: ExpertControlAuthorization | None = None
    lease_id: str = ""
    fencing_epoch: int | None = None
    budget: ExpertControlBudget | None = None
    admission: TrainingCorpusAdmission | None = None

    def __post_init__(self) -> None:
        operation = _text(self.operation, "operation")
        if operation not in ALL_OPERATIONS:
            raise ResidualIntelligenceError(f"unknown expert operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "expert_id", _text(self.expert_id, "expert_id", required=False))
        if not isinstance(self.parameters, Mapping):
            raise ResidualIntelligenceError("parameters must be an object")
        try:
            encoded = json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ResidualIntelligenceError("parameters must be JSON-compatible") from exc
        if len(encoded.encode("utf-8")) > 32_768:
            raise ResidualIntelligenceError("parameters exceed 32768 bytes")
        object.__setattr__(self, "parameters", dict(self.parameters))
        if type(self.dry_run) is not bool:
            raise ResidualIntelligenceError("dry_run must be boolean")
        if self.authorization is not None and not isinstance(self.authorization, ExpertControlAuthorization):
            if not isinstance(self.authorization, Mapping):
                raise ResidualIntelligenceError("authorization must be typed")
            object.__setattr__(self, "authorization", ExpertControlAuthorization.from_dict(self.authorization))
        if self.budget is not None and not isinstance(self.budget, ExpertControlBudget):
            if not isinstance(self.budget, Mapping):
                raise ResidualIntelligenceError("budget must be typed")
            object.__setattr__(self, "budget", ExpertControlBudget.from_dict(self.budget))
        if self.admission is not None and not isinstance(self.admission, TrainingCorpusAdmission):
            if not isinstance(self.admission, Mapping):
                raise ResidualIntelligenceError("admission must be typed")
            object.__setattr__(self, "admission", TrainingCorpusAdmission.from_dict(self.admission))
        if operation in MUTATION_OPERATIONS:
            object.__setattr__(self, "idempotency_key", _text(self.idempotency_key, "idempotency_key"))
            object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
            if isinstance(self.fencing_epoch, bool) or not isinstance(self.fencing_epoch, int) or self.fencing_epoch < 0:
                raise ResidualIntelligenceError("mutation requires a non-negative fencing_epoch")
            if self.authorization is None or not self.authorization.allows_mutation:
                raise ResidualIntelligenceError("mutation requires expert mutation authorization")
            if self.budget is None:
                raise ResidualIntelligenceError("mutation requires a resource budget")
        elif any((self.idempotency_key, self.lease_id, self.fencing_epoch is not None, self.budget is not None)):
            raise ResidualIntelligenceError("read operations cannot carry mutation bindings")

    @property
    def fingerprint(self) -> str:
        body = {
            "operation": self.operation, "expert_id": self.expert_id,
            "parameters": self.parameters, "dry_run": self.dry_run,
            "authorization": self.authorization.to_dict() if self.authorization else None,
            "lease_id": self.lease_id, "fencing_epoch": self.fencing_epoch,
            "budget": self.budget.to_dict() if self.budget else None,
            "admission_id": self.admission.admission_id if self.admission else "",
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"operation": self.operation, "expert_id": self.expert_id, "parameters": dict(self.parameters), "dry_run": self.dry_run}
        if self.operation in MUTATION_OPERATIONS:
            result.update({"idempotency_key": self.idempotency_key, "authorization": self.authorization.to_dict() if self.authorization else None, "lease_id": self.lease_id, "fencing_epoch": self.fencing_epoch, "budget": self.budget.to_dict() if self.budget else None})
        if self.admission is not None:
            result["admission"] = self.admission.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertControlRequest":
        allowed = {"operation", "expert_id", "parameters", "dry_run", "idempotency_key", "authorization", "lease_id", "fencing_epoch", "budget", "admission"}
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise ResidualIntelligenceError("expert control request contains unknown fields")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ExpertControlResult:
    operation: str
    ok: bool
    status: str
    audit_id: str
    payload: Mapping[str, Any]
    idempotent_replay: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertControlResult":
        required = {"operation", "ok", "status", "audit_id", "payload"}
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ResidualIntelligenceError("expert control service returned an invalid result")
        return cls(_text(payload["operation"], "result operation"), bool(payload["ok"]), _text(payload["status"], "result status"), _text(payload["audit_id"], "audit_id"), dict(payload["payload"]) if isinstance(payload["payload"], Mapping) else {}, bool(payload.get("idempotent_replay", False)))

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "ok": self.ok, "status": self.status, "audit_id": self.audit_id, "payload": dict(self.payload), "idempotent_replay": self.idempotent_replay}


class ResidualExpertControlBackend:
    """A thin direct adapter over the canonical service, with no fallback."""

    def __init__(self, service: SupervisorControlService) -> None:
        if not isinstance(service, SupervisorControlService):
            raise TypeError("service must be a SupervisorControlService")
        self._service = service

    @property
    def service(self) -> SupervisorControlService:
        return self._service

    def execute(self, request: ExpertControlRequest) -> ExpertControlResult:
        if not isinstance(request, ExpertControlRequest):
            raise TypeError("request must be an ExpertControlRequest")
        return ExpertControlResult.from_dict(self._service.execute_expert(request))


def register_expert_operations() -> dict[str, tuple[str, ...]]:
    """Publish the exact canonical expert operation catalog."""

    return {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}


def mcp_dispatch(backend: ResidualExpertControlBackend, request: ExpertControlRequest) -> ExpertControlResult:
    """Direct service dispatch; this function never executes a shell."""

    return backend.execute(request)


_CLI_NAMES: Final[dict[str, str]] = {item.rsplit(".", 1)[1].replace("_", "-"): item for item in ALL_OPERATIONS}


def register_expert_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register the ``agent experts`` group on an existing agent parser."""

    experts = subparsers.add_parser("experts", help="Typed residual-expert controls.")
    commands = experts.add_subparsers(dest="expert_command", required=True)
    for command, operation in _CLI_NAMES.items():
        child = commands.add_parser(command, help=f"Run {operation}.")
        child.set_defaults(expert_operation=operation)
        child.add_argument("--request-json", required=True, help="Complete typed ExpertControlRequest JSON.")
        child.add_argument("--output-json", action="store_true", help="Emit canonical result JSON.")
    return experts


def build_expert_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipfs-accelerate")
    root = parser.add_subparsers(dest="root_command", required=True)
    agent = root.add_parser("agent")
    register_expert_cli(agent.add_subparsers(dest="agent_command", required=True))
    return parser


def run_expert_cli(argv: Sequence[str], backend: ResidualExpertControlBackend, *, stdout: TextIO | None = None) -> int:
    """Run one CLI request through precisely the MCP/Python service adapter."""

    args = build_expert_cli_parser().parse_args(list(argv))
    try:
        request = ExpertControlRequest.from_dict(json.loads(args.request_json))
        if request.operation != args.expert_operation:
            raise ResidualIntelligenceError("CLI command and request operation differ")
        result = backend.execute(request)
    except (json.JSONDecodeError, ResidualIntelligenceError, TypeError, ValueError) as exc:
        if stdout is not None:
            stdout.write(json.dumps({"ok": False, "status": "invalid", "error": str(exc)}, sort_keys=True) + "\n")
        return 2
    if stdout is not None:
        stdout.write(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if result.ok else 1


__all__ = [
    "ALL_OPERATIONS", "EXPERT_MUTATION_SCOPE", "MUTATION_OPERATIONS", "READ_OPERATIONS",
    "ExpertControlAuthorization", "ExpertControlBudget", "ExpertControlRequest", "ExpertControlResult",
    "ResidualExpertControlBackend", "build_expert_cli_parser", "mcp_dispatch", "register_expert_cli",
    "register_expert_operations", "run_expert_cli",
]
