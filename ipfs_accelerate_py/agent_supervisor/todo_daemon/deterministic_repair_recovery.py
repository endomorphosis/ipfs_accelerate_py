"""Deterministic-only replay of interrupted repair transitions.

``RepairRecovery@1`` intentionally makes no attempt to *finish* an unknown
operation.  A restart can only replay a journaled receipt, request rollback
of an uncommitted mutation, or issue a typed deterministic retry decision.
It never invents a success receipt and has no provider or model integration.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json, content_identity


REPAIR_RECOVERY_INTERFACE: Final[str] = "RepairRecovery@1"
REPAIR_RECOVERY_VERSION: Final[int] = 1
RECOVERY_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-recovery-request@1"
)
RECOVERY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-recovery-decision@1"
)
RECOVERY_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-recovery-store@1"
)

_TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {"receipt_committed", "rolled_back"}
)
_JOURNAL_STATES: Final[frozenset[str]] = frozenset(
    {"intent", "mutation_applied", "receipt_committed", "rolled_back", "failed"}
)
_TRANSIENT_FAILURES: Final[frozenset[str]] = frozenset(
    {"transient_io", "transient_timeout", "transient_resource"}
)
_FORBIDDEN_ROUTE_TOKENS: Final[frozenset[str]] = frozenset(
    {"provider", "model", "llm", "openai", "anthropic", "codex", "grok"}
)


class DeterministicRecoveryError(RuntimeError):
    """A recovery input or durable projection violates the closed contract."""


class RecoveryDisposition(str, Enum):
    """Closed restart outcomes; none has implicit write authority."""

    REPLAYED_RECEIPT = "replayed_receipt"
    ROLLBACK_REQUIRED = "rollback_required"
    RETRY_DETERMINISTIC = "retry_deterministic"
    DEFER_LEASE_EXPIRED = "defer_lease_expired"
    ABSTAIN = "abstain"


def _identifier(value: Any, name: str, *, optional: bool = False) -> str:
    if not isinstance(value, str):
        raise DeterministicRecoveryError(f"{name} must be a string")
    result = value.strip()
    if not result and not optional:
        raise DeterministicRecoveryError(f"{name} is required")
    if "\x00" in result or len(result.encode("utf-8")) > 4096:
        raise DeterministicRecoveryError(f"{name} is not a bounded identifier")
    return result


def _route_is_forbidden(route: str) -> bool:
    normalized = route.lower().replace("-", "_")
    return any(token in normalized for token in _FORBIDDEN_ROUTE_TOKENS)


@dataclass(frozen=True)
class RecoveryJournalEntry:
    """One append-only, body-free durable transition observation."""

    transition_id: str
    sequence: int
    state: str
    operation_id: str
    gate_id: str
    mutation_id: str = ""
    receipt_id: str = ""
    failure_kind: str = ""
    route_kind: str = "deterministic_operator"

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition_id", _identifier(self.transition_id, "transition_id"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise DeterministicRecoveryError("sequence must be a positive integer")
        state = _identifier(self.state, "state")
        if state not in _JOURNAL_STATES:
            raise DeterministicRecoveryError("journal state is unsupported")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "operation_id", _identifier(self.operation_id, "operation_id"))
        object.__setattr__(self, "gate_id", _identifier(self.gate_id, "gate_id"))
        object.__setattr__(self, "mutation_id", _identifier(self.mutation_id, "mutation_id", optional=True))
        object.__setattr__(self, "receipt_id", _identifier(self.receipt_id, "receipt_id", optional=True))
        object.__setattr__(self, "failure_kind", _identifier(self.failure_kind, "failure_kind", optional=True))
        route = _identifier(self.route_kind, "route_kind")
        if _route_is_forbidden(route):
            raise DeterministicRecoveryError("provider/model routes are forbidden in recovery journals")
        object.__setattr__(self, "route_kind", route)
        if state == "receipt_committed" and (not self.receipt_id or not self.mutation_id):
            raise DeterministicRecoveryError("committed receipt requires receipt_id and mutation_id")
        if state == "mutation_applied" and not self.mutation_id:
            raise DeterministicRecoveryError("mutation_applied requires mutation_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "sequence": self.sequence,
            "state": self.state,
            "operation_id": self.operation_id,
            "gate_id": self.gate_id,
            "mutation_id": self.mutation_id,
            "receipt_id": self.receipt_id,
            "failure_kind": self.failure_kind,
            "route_kind": self.route_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryJournalEntry":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise DeterministicRecoveryError("unknown recovery journal fields: " + ", ".join(sorted(unknown)))
        return cls(**{name: value.get(name, "" if name not in {"sequence"} else 0) for name in allowed})


@dataclass(frozen=True)
class RecoveryRequest:
    """Complete coordinate set required to recover one repair transition."""

    task_id: str
    run_id: str
    journal: tuple[RecoveryJournalEntry, ...]
    required_gate_id: str
    lease_expired: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "required_gate_id", _identifier(self.required_gate_id, "required_gate_id"))
        if not isinstance(self.lease_expired, bool):
            raise DeterministicRecoveryError("lease_expired must be a boolean")
        entries = tuple(
            entry if isinstance(entry, RecoveryJournalEntry) else RecoveryJournalEntry.from_dict(entry)
            for entry in self.journal
        )
        if not entries:
            raise DeterministicRecoveryError("recovery requires a durable journal")
        sequences = [entry.sequence for entry in entries]
        if len(sequences) != len(set(sequences)):
            raise DeterministicRecoveryError("journal sequence values must be unique")
        object.__setattr__(self, "journal", tuple(sorted(entries, key=lambda entry: entry.sequence)))

    @property
    def request_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_REQUEST_SCHEMA,
            "interface": REPAIR_RECOVERY_INTERFACE,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "journal": [entry.to_dict() for entry in self.journal],
            "required_gate_id": self.required_gate_id,
            "lease_expired": self.lease_expired,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryRequest":
        allowed = {"schema", "interface", "task_id", "run_id", "journal", "required_gate_id", "lease_expired"}
        if set(value) - allowed:
            raise DeterministicRecoveryError("unknown recovery request fields")
        if value.get("schema") not in {None, "", RECOVERY_REQUEST_SCHEMA}:
            raise DeterministicRecoveryError("unsupported recovery request schema")
        return cls(
            task_id=value.get("task_id"), run_id=value.get("run_id"),
            journal=tuple(value.get("journal") or ()),
            required_gate_id=value.get("required_gate_id"),
            lease_expired=value.get("lease_expired", False),
        )


@dataclass(frozen=True)
class RecoveryDecision:
    """A replayable decision that does not stand in for an execution receipt."""

    request_id: str
    transition_id: str
    disposition: RecoveryDisposition
    reason_code: str
    replayed_receipt_id: str = ""
    mutation_id: str = ""
    runtime_model_calls: int = 0
    mutation_authorized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        object.__setattr__(self, "transition_id", _identifier(self.transition_id, "transition_id"))
        object.__setattr__(self, "disposition", RecoveryDisposition(self.disposition))
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "replayed_receipt_id", _identifier(self.replayed_receipt_id, "replayed_receipt_id", optional=True))
        object.__setattr__(self, "mutation_id", _identifier(self.mutation_id, "mutation_id", optional=True))
        if self.runtime_model_calls != 0 or self.mutation_authorized:
            raise DeterministicRecoveryError("recovery decisions never authorize a model or mutation")
        if self.disposition is RecoveryDisposition.REPLAYED_RECEIPT and not self.replayed_receipt_id:
            raise DeterministicRecoveryError("receipt replay requires a durable receipt id")

    @property
    def decision_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_DECISION_SCHEMA,
            "interface": REPAIR_RECOVERY_INTERFACE,
            "request_id": self.request_id,
            "transition_id": self.transition_id,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "replayed_receipt_id": self.replayed_receipt_id,
            "mutation_id": self.mutation_id,
            "runtime_model_calls": 0,
            "mutation_authorized": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryDecision":
        allowed = {"schema", "interface", "request_id", "transition_id", "disposition", "reason_code", "replayed_receipt_id", "mutation_id", "runtime_model_calls", "mutation_authorized"}
        if set(value) - allowed:
            raise DeterministicRecoveryError("unknown recovery decision fields")
        if value.get("schema") not in {None, "", RECOVERY_DECISION_SCHEMA}:
            raise DeterministicRecoveryError("unsupported recovery decision schema")
        return cls(**{name: value.get(name, "" if name not in {"runtime_model_calls", "mutation_authorized"} else (0 if name == "runtime_model_calls" else False)) for name in cls.__dataclass_fields__})


def replay_recovery(request: RecoveryRequest | Mapping[str, Any]) -> RecoveryDecision:
    """Derive the only safe recovery action from journal evidence.

    An unresolved mutation requests rollback; it is never applied a second
    time.  A receipt is replayed only when it already exists in the journal.
    """

    value = request if isinstance(request, RecoveryRequest) else RecoveryRequest.from_dict(request)
    by_transition: dict[str, list[RecoveryJournalEntry]] = {}
    for entry in value.journal:
        by_transition.setdefault(entry.transition_id, []).append(entry)
    transition_id = sorted(by_transition)[-1]
    entries = by_transition[transition_id]
    first = entries[0]
    if first.gate_id != value.required_gate_id or any(entry.gate_id != first.gate_id or entry.operation_id != first.operation_id for entry in entries):
        raise DeterministicRecoveryError("journal transition binding or mandatory gate mismatch")
    if value.lease_expired:
        return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.DEFER_LEASE_EXPIRED, "lease_expired")
    terminal = [entry for entry in entries if entry.state in _TERMINAL_STATES]
    if terminal:
        last = terminal[-1]
        if last.state == "receipt_committed":
            return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.REPLAYED_RECEIPT, "durable_receipt_replayed", replayed_receipt_id=last.receipt_id, mutation_id=last.mutation_id)
        return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.ABSTAIN, "durable_rollback_replayed", mutation_id=last.mutation_id)
    last = entries[-1]
    if last.state == "mutation_applied":
        return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.ROLLBACK_REQUIRED, "uncommitted_mutation_requires_rollback", mutation_id=last.mutation_id)
    if last.state == "failed" and last.failure_kind in _TRANSIENT_FAILURES:
        return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.RETRY_DETERMINISTIC, "typed_transient_retry")
    return RecoveryDecision(value.request_id, transition_id, RecoveryDisposition.ABSTAIN, "incomplete_or_nontransient_transition")


def recover_repair_state(
    request: RecoveryRequest | Mapping[str, Any], *, state_path: str | Path | None = None
) -> RecoveryDecision:
    """Replay recovery and atomically memoize the exact decision across restart."""

    value = request if isinstance(request, RecoveryRequest) else RecoveryRequest.from_dict(request)
    decision = replay_recovery(value)
    if state_path is None:
        return decision
    path = Path(state_path)
    if path.exists() and path.is_symlink():
        raise DeterministicRecoveryError("recovery state cannot be a symlink")
    existing: dict[str, Any] = {"schema": RECOVERY_STORE_SCHEMA, "decisions": {}}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema") != RECOVERY_STORE_SCHEMA or not isinstance(existing.get("decisions"), dict):
            raise DeterministicRecoveryError("recovery state has an invalid schema")
    previous = existing["decisions"].get(value.request_id)
    if previous is not None:
        loaded = RecoveryDecision.from_dict(previous)
        if loaded.to_dict() != decision.to_dict():
            raise DeterministicRecoveryError("stored recovery decision diverges from journal replay")
        return loaded
    existing["decisions"][value.request_id] = decision.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(existing))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return decision


__all__ = [
    "DeterministicRecoveryError", "RECOVERY_DECISION_SCHEMA", "RECOVERY_REQUEST_SCHEMA",
    "REPAIR_RECOVERY_INTERFACE", "REPAIR_RECOVERY_VERSION", "RecoveryDecision",
    "RecoveryDisposition", "RecoveryJournalEntry", "RecoveryRequest", "recover_repair_state",
    "replay_recovery",
]
