"""Immutable contracts for fail-closed action admission and receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping


class RiskClass(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    HUMAN = "human"


class SideEffectClass(str, Enum):
    NONE = "none"
    LOCAL_READ = "local_read"
    LOCAL_WRITE = "local_write"
    NETWORK = "network"
    EXTERNAL_MUTATION = "external_mutation"


class ActionDecisionKind(str, Enum):
    DENY = "deny"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    HANDOFF = "handoff"
    PERMIT_READ = "permit_read"
    PERMIT_EXECUTE = "permit_execute"


class ActionStatus(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    DENIED = "denied"
    COMPENSATED = "compensated"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ActionProposal:
    """Authority-free proposal emitted by retrieval or a voice route."""

    proposal_id: str
    descriptor_id: str
    logical_action: str
    arguments: Mapping[str, str] = field(default_factory=dict)
    route: str | None = None
    source: str = "voice_route"
    confidence: float = 0.0
    tenant_id: str | None = None
    session_id: str | None = None
    channel: str | None = None
    evidence: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.descriptor_id or not self.logical_action:
            raise ValueError("proposal_id, descriptor_id, and logical_action are required")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.arguments.items()):
            raise ValueError("arguments must be string-to-string")
        # Proposals never carry executable locators.
        banned = ("command", "argv", "executable", "cwd", "env", "shell", "import_path", "url")
        for key in self.arguments:
            if key.lower() in banned or key.lower().endswith("_path"):
                raise ValueError(f"proposal argument {key!r} is not allowed")

    @property
    def arguments_digest(self) -> str:
        return content_digest(dict(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "descriptor_id": self.descriptor_id,
            "logical_action": self.logical_action,
            "arguments": dict(self.arguments),
            "arguments_digest": self.arguments_digest,
            "route": self.route,
            "source": self.source,
            "confidence": self.confidence,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "channel": self.channel,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionDecision:
    """Policy outcome bound to a specific proposal and descriptor revision."""

    decision_id: str
    kind: ActionDecisionKind
    proposal_id: str
    descriptor_id: str
    descriptor_digest: str
    arguments_digest: str
    reason: str
    policy_revision: str = "v1"
    risk_class: RiskClass = RiskClass.READ
    expires_at_epoch_s: float | None = None

    @property
    def permits_execution(self) -> bool:
        return self.kind in {
            ActionDecisionKind.PERMIT_READ,
            ActionDecisionKind.PERMIT_EXECUTE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "kind": self.kind.value,
            "proposal_id": self.proposal_id,
            "descriptor_id": self.descriptor_id,
            "descriptor_digest": self.descriptor_digest,
            "arguments_digest": self.arguments_digest,
            "reason": self.reason,
            "policy_revision": self.policy_revision,
            "risk_class": self.risk_class.value,
            "expires_at_epoch_s": self.expires_at_epoch_s,
            "permits_execution": self.permits_execution,
        }


@dataclass(frozen=True)
class ActionReceipt:
    """Redacted, content-addressable outcome of an admitted invocation."""

    receipt_id: str
    status: ActionStatus
    proposal_id: str
    decision_id: str
    descriptor_id: str
    adapter: str
    interface_identity: str
    started_epoch_s: float | None = None
    completed_epoch_s: float | None = None
    exit_code: int | None = None
    stdout_digest: str | None = None
    stderr_digest: str | None = None
    public_result: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "descriptor_id": self.descriptor_id,
            "adapter": self.adapter,
            "interface_identity": self.interface_identity,
            "started_epoch_s": self.started_epoch_s,
            "completed_epoch_s": self.completed_epoch_s,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "public_result": dict(self.public_result),
            "error": self.error,
            "metadata": dict(self.metadata),
            "receipt_digest": content_digest(
                {
                    "status": self.status.value,
                    "proposal_id": self.proposal_id,
                    "decision_id": self.decision_id,
                    "descriptor_id": self.descriptor_id,
                    "adapter": self.adapter,
                    "interface_identity": self.interface_identity,
                    "exit_code": self.exit_code,
                    "stdout_digest": self.stdout_digest,
                    "stderr_digest": self.stderr_digest,
                    "public_result": dict(self.public_result),
                    "error": self.error,
                }
            ),
        }
