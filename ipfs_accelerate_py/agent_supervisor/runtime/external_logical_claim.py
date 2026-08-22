"""Prevent duplicate logical acceptance (EAAEF-083).

A logical claim is identified by:

    task_id + plan_revision + base_tree + semantic_root
        + task_spec_cid + idempotency_key

Many attempts may register.  Exactly one result may be accepted for a
given logical key.  Different idempotency keys are different claims.
This module is an in-process fail-closed ledger; it does not overlay
runtime CAS and does not issue signatures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


LOGICAL_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-logical-claim@1"
)
LOGICAL_CLAIM_INTERFACE: Final[str] = "LogicalClaim@1"
LOGICAL_CLAIM_KEY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "plan_revision",
    "base_tree",
    "semantic_root",
    "task_spec_cid",
    "idempotency_key",
)


class LogicalClaimError(ValueError):
    """Malformed or unsafe logical-claim operation."""


class DuplicateLogicalAcceptanceError(LogicalClaimError):
    """A second accept of the same logical key failed closed."""


class UnregisteredAttemptError(LogicalClaimError):
    """accept() requires a previously registered attempt."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LogicalClaimError(f"{name} is required")
    text = value.strip()
    if "\x00" in text:
        raise LogicalClaimError(f"{name} must not contain NUL")
    return text


class _AcceptanceState:
    """Mutable attempt/accept slot for one logical key."""

    __slots__ = ("lock", "attempts", "accepted_attempt_id")

    def __init__(self) -> None:
        self.lock = Lock()
        self.attempts: dict[str, str] = {}
        self.accepted_attempt_id: str | None = None


@dataclass(frozen=True)
class LogicalClaim:
    """One logical result identity. Many attempts may run; one accept."""

    task_id: str
    plan_revision: str
    base_tree: str
    semantic_root: str
    task_spec_cid: str
    idempotency_key: str

    def __post_init__(self) -> None:
        for name in LOGICAL_CLAIM_KEY_FIELDS:
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "_state", _AcceptanceState())

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.task_id,
            self.plan_revision,
            self.base_tree,
            self.semantic_root,
            self.task_spec_cid,
            self.idempotency_key,
        )

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "schema": LOGICAL_CLAIM_SCHEMA,
                "interface": LOGICAL_CLAIM_INTERFACE,
                "task_id": self.task_id,
                "plan_revision": self.plan_revision,
                "base_tree": self.base_tree,
                "semantic_root": self.semantic_root,
                "task_spec_cid": self.task_spec_cid,
                "idempotency_key": self.idempotency_key,
            }
        )

    @property
    def attempts(self) -> tuple[str, ...]:
        state: _AcceptanceState = object.__getattribute__(self, "_state")
        with state.lock:
            return tuple(state.attempts)

    @property
    def accepted_attempt_id(self) -> str | None:
        state: _AcceptanceState = object.__getattribute__(self, "_state")
        with state.lock:
            return state.accepted_attempt_id

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": LOGICAL_CLAIM_SCHEMA,
                "interface": LOGICAL_CLAIM_INTERFACE,
                "task_id": self.task_id,
                "plan_revision": self.plan_revision,
                "base_tree": self.base_tree,
                "semantic_root": self.semantic_root,
                "task_spec_cid": self.task_spec_cid,
                "idempotency_key": self.idempotency_key,
                "logical_claim_key": self.content_id,
            }
        )

    def _receipt(self, *, attempt_id: str, status: str) -> Mapping[str, Any]:
        payload = dict(self.to_dict())
        payload["attempt_id"] = attempt_id
        payload["status"] = status
        payload["accepted_attempt_id"] = self.accepted_attempt_id
        return MappingProxyType(payload)

    def register(self, attempt_id: str) -> Mapping[str, Any]:
        """Record an execution attempt. Duplicate attempt ids are idempotent."""
        attempt = _text(attempt_id, "attempt_id")
        state: _AcceptanceState = object.__getattribute__(self, "_state")
        with state.lock:
            state.attempts.setdefault(attempt, attempt)
        return self._receipt(attempt_id=attempt, status="registered")

    def accept(self, attempt_id: str) -> Mapping[str, Any]:
        """Accept one registered attempt. A second accept of this key fails closed."""
        attempt = _text(attempt_id, "attempt_id")
        state: _AcceptanceState = object.__getattribute__(self, "_state")
        with state.lock:
            if attempt not in state.attempts:
                raise UnregisteredAttemptError("attempt is not registered")
            if state.accepted_attempt_id is not None:
                raise DuplicateLogicalAcceptanceError("duplicate logical acceptance")
            state.accepted_attempt_id = attempt
        return self._receipt(attempt_id=attempt, status="accepted")

    @classmethod
    def bind(
        cls,
        *,
        task_id: str,
        plan_revision: str,
        base_tree: str,
        semantic_root: str,
        task_spec_cid: str,
        idempotency_key: str,
    ) -> "LogicalClaim":
        return cls(
            task_id=task_id,
            plan_revision=plan_revision,
            base_tree=base_tree,
            semantic_root=semantic_root,
            task_spec_cid=task_spec_cid,
            idempotency_key=idempotency_key,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LogicalClaim":
        if not isinstance(payload, Mapping):
            raise LogicalClaimError("logical claim must be an object")
        return cls.bind(
            task_id=str(payload.get("task_id") or ""),
            plan_revision=str(payload.get("plan_revision") or ""),
            base_tree=str(payload.get("base_tree") or payload.get("repository_base_tree") or ""),
            semantic_root=str(
                payload.get("semantic_root") or payload.get("semantic_state_root") or ""
            ),
            task_spec_cid=str(payload.get("task_spec_cid") or ""),
            idempotency_key=str(payload.get("idempotency_key") or ""),
        )


class LogicalClaimLedger:
    """Intern claims by logical key so one key has one acceptance slot."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._claims: dict[tuple[str, str, str, str, str, str], LogicalClaim] = {}

    def bind(self, claim: LogicalClaim | Mapping[str, Any] | None = None, **fields: str) -> LogicalClaim:
        if claim is None:
            bound = LogicalClaim.bind(**fields)
        elif isinstance(claim, LogicalClaim):
            if fields:
                raise LogicalClaimError("claim and field overrides cannot both be supplied")
            bound = claim
        else:
            if fields:
                raise LogicalClaimError("claim and field overrides cannot both be supplied")
            bound = LogicalClaim.from_mapping(claim)
        with self._lock:
            existing = self._claims.get(bound.key)
            if existing is not None:
                return existing
            self._claims[bound.key] = bound
            return bound

    def register(
        self,
        claim: LogicalClaim | Mapping[str, Any],
        attempt_id: str,
    ) -> Mapping[str, Any]:
        return self.bind(claim).register(attempt_id)

    def accept(
        self,
        claim: LogicalClaim | Mapping[str, Any],
        attempt_id: str,
    ) -> Mapping[str, Any]:
        return self.bind(claim).accept(attempt_id)


__all__ = (
    "LOGICAL_CLAIM_INTERFACE",
    "LOGICAL_CLAIM_KEY_FIELDS",
    "LOGICAL_CLAIM_SCHEMA",
    "DuplicateLogicalAcceptanceError",
    "LogicalClaim",
    "LogicalClaimError",
    "LogicalClaimLedger",
    "UnregisteredAttemptError",
)
