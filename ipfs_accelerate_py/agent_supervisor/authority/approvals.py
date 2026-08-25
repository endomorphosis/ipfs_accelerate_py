"""Exact-action approval and denial boundaries (EAAEF-032)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


APPROVAL_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-approval@1"
EFFECTS_REQUIRING_APPROVAL: Final[frozenset[str]] = frozenset(
    {
        "install",
        "network",
        "secret",
        "disclosure",
        "merge",
        "push",
        "destructive",
        "publication",
    }
)


class ApprovalError(ValueError):
    """Approval request is malformed or unauthorized."""


@dataclass(frozen=True)
class ApprovalRecord:
    principal_id: str
    action: str
    input_binding: str
    decision: str
    reason_code: str
    created_at_ms: int

    def __post_init__(self) -> None:
        if str(self.decision) not in {"approved", "denied"}:
            raise ApprovalError("decision must be approved or denied")
        if str(self.action) not in EFFECTS_REQUIRING_APPROVAL:
            raise ApprovalError(f"unknown or unapproved action: {self.action}")
        if not str(self.principal_id).strip() or not str(self.input_binding).strip():
            raise ApprovalError("principal and input binding are required")
        if self.principal_id.startswith("sha256:") or self.principal_id.startswith("worker:"):
            raise ApprovalError("workers and CIDs cannot approve")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": APPROVAL_SCHEMA,
                "principal_id": self.principal_id,
                "action": self.action,
                "input_binding": self.input_binding,
                "decision": self.decision,
                "reason_code": self.reason_code,
                "created_at_ms": int(self.created_at_ms),
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))


class ApprovalLog:
    """Preserve denials; require authenticated input-bound approval."""

    def __init__(self) -> None:
        self._records: list[ApprovalRecord] = []

    def decide(
        self,
        *,
        principal_id: str,
        action: str,
        input_binding: str,
        decision: str,
        reason_code: str,
        created_at_ms: int,
    ) -> ApprovalRecord:
        if self.is_denied(action=action, input_binding=input_binding):
            raise ApprovalError("preserved denial blocks later approval of the same input")
        record = ApprovalRecord(
            principal_id=principal_id,
            action=action,
            input_binding=input_binding,
            decision=decision,
            reason_code=reason_code,
            created_at_ms=created_at_ms,
        )
        self._records.append(record)
        return record

    def is_denied(self, *, action: str, input_binding: str) -> bool:
        return any(
            record.action == action
            and record.input_binding == input_binding
            and record.decision == "denied"
            for record in self._records
        )

    def require(
        self,
        *,
        action: str,
        input_binding: str,
        principal_id: str,
    ) -> ApprovalRecord:
        if action not in EFFECTS_REQUIRING_APPROVAL:
            raise ApprovalError(f"unknown action: {action}")
        if self.is_denied(action=action, input_binding=input_binding):
            raise ApprovalError("preserved denial blocks later approval of the same input")
        matches = [
            record
            for record in self._records
            if record.action == action
            and record.input_binding == input_binding
            and record.principal_id == principal_id
            and record.decision == "approved"
        ]
        if not matches:
            raise ApprovalError(f"{action} requires authenticated input-bound approval")
        return matches[-1]

    def records(self) -> tuple[ApprovalRecord, ...]:
        return tuple(self._records)
