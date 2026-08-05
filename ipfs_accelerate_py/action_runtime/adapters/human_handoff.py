"""Human handoff action adapter (handoff_live_agent + HandoffRequest receipts).

Safety rules:

* policy admits **request creation** via ``ActionDecisionKind.HANDOFF``
  (``permits_execution`` remains false — that is intentional)
* creating a durable ``HandoffRequest`` is **not** a completed transfer
* receipt statuses distinguish accepted / started / succeeded / unknown / failed
* spoken success is forbidden unless the receipt status is ``succeeded``
* no carrier / telephony network calls — transfer outcomes are injected
* free-text summaries are size-bounded and redacted from public receipts by default
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from ..contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionReceipt,
    ActionStatus,
    content_digest,
)

HANDOFF_LOGICAL_ACTION = "handoff_live_agent"
_ALLOWED_LOGICAL_ACTIONS = frozenset({HANDOFF_LOGICAL_ACTION})
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")
_ALLOWED_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})
_ALLOWED_CHANNELS = frozenset({"voice", "chat", "test", "telephone", "sms"})

DEFAULT_MAX_SUMMARY_CHARS = 500
DEFAULT_MAX_REASON_CHARS = 200
DEFAULT_MAX_QUEUE_CHARS = 64

# Statuses that may appear on handoff request + action receipts for this adapter.
HANDOFF_RECEIPT_STATUSES: frozenset[ActionStatus] = frozenset(
    {
        ActionStatus.ACCEPTED,
        ActionStatus.STARTED,
        ActionStatus.SUCCEEDED,
        ActionStatus.UNKNOWN,
        ActionStatus.FAILED,
        ActionStatus.DENIED,
        ActionStatus.CANCELLED,
    }
)

# Terminal transfer outcomes that a fake/real telephony backend may report.
_PROVIDER_TERMINAL_STATUSES: frozenset[ActionStatus] = frozenset(
    {
        ActionStatus.SUCCEEDED,
        ActionStatus.FAILED,
        ActionStatus.UNKNOWN,
        ActionStatus.CANCELLED,
    }
)

_STATUS_RANK: Mapping[ActionStatus, int] = {
    ActionStatus.ACCEPTED: 10,
    ActionStatus.STARTED: 20,
    ActionStatus.SUCCEEDED: 30,
    ActionStatus.FAILED: 30,
    ActionStatus.UNKNOWN: 30,
    ActionStatus.CANCELLED: 30,
    ActionStatus.DENIED: 0,
    ActionStatus.TIMED_OUT: 30,
    ActionStatus.COMPENSATED: 40,
}


class HandoffRequestPhase(str, Enum):
    """Lifecycle phase of a durable handoff request (mirrors receipt status)."""

    ACCEPTED = "accepted"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _phase_to_status(phase: HandoffRequestPhase | str) -> ActionStatus:
    value = phase.value if isinstance(phase, HandoffRequestPhase) else str(phase)
    return ActionStatus(value)


def _status_to_phase(status: ActionStatus | str) -> HandoffRequestPhase:
    value = status.value if isinstance(status, ActionStatus) else str(status)
    return HandoffRequestPhase(value)


@dataclass(frozen=True)
class HandoffSandboxPolicy:
    """Resource and privacy bounds for handoff adapter invocations."""

    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    max_reason_chars: int = DEFAULT_MAX_REASON_CHARS
    max_queue_chars: int = DEFAULT_MAX_QUEUE_CHARS
    # When True (default), public receipts omit free-text summaries.
    redact_summary_in_receipts: bool = True
    # Default queue when the proposal omits one.
    default_queue: str = "live_agent"
    # Default priority when the proposal omits one.
    default_priority: str = "normal"

    def __post_init__(self) -> None:
        if self.max_summary_chars < 1 or self.max_summary_chars > 4_096:
            raise ValueError("max_summary_chars must be in [1, 4096]")
        if self.max_reason_chars < 1 or self.max_reason_chars > 1_024:
            raise ValueError("max_reason_chars must be in [1, 1024]")
        if self.max_queue_chars < 1 or self.max_queue_chars > 128:
            raise ValueError("max_queue_chars must be in [1, 128]")
        if self.default_priority not in _ALLOWED_PRIORITIES:
            raise ValueError(
                f"default_priority must be one of {sorted(_ALLOWED_PRIORITIES)}"
            )


@dataclass(frozen=True)
class HandoffRequest:
    """Durable human-handoff request created by ``handoff_live_agent``.

    Creating this record is **request admission**, not transfer success.
    """

    request_id: str
    status: ActionStatus
    proposal_id: str
    decision_id: str
    descriptor_id: str
    logical_action: str
    tenant_id: str | None
    session_id: str | None
    channel: str | None
    queue: str
    priority: str
    reason: str
    summary: str
    summary_digest: str
    created_at_epoch_s: float
    updated_at_epoch_s: float
    provider_confirmation: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.status not in HANDOFF_RECEIPT_STATUSES:
            raise ValueError(f"unsupported handoff status: {self.status!r}")
        if not self.summary_digest and self.summary:
            object.__setattr__(self, "summary_digest", content_digest(self.summary))

    @property
    def phase(self) -> HandoffRequestPhase:
        return _status_to_phase(self.status)

    @property
    def is_transfer_complete(self) -> bool:
        """True only when a provider confirmed a successful transfer."""

        return self.status is ActionStatus.SUCCEEDED

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "phase": self.phase.value,
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "descriptor_id": self.descriptor_id,
            "logical_action": self.logical_action,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "channel": self.channel,
            "queue": self.queue,
            "priority": self.priority,
            "reason": self.reason,
            "summary_digest": self.summary_digest,
            "summary_present": bool(self.summary),
            "created_at_epoch_s": self.created_at_epoch_s,
            "updated_at_epoch_s": self.updated_at_epoch_s,
            "provider_confirmation": self.provider_confirmation,
            "metadata": dict(self.metadata),
            "is_transfer_complete": self.is_transfer_complete,
            "request_digest": content_digest(
                {
                    "request_id": self.request_id,
                    "status": self.status.value,
                    "proposal_id": self.proposal_id,
                    "decision_id": self.decision_id,
                    "descriptor_id": self.descriptor_id,
                    "queue": self.queue,
                    "priority": self.priority,
                    "reason": self.reason,
                    "summary_digest": self.summary_digest,
                    "provider_confirmation": self.provider_confirmation,
                }
            ),
        }

    def with_status(
        self,
        status: ActionStatus,
        *,
        updated_at_epoch_s: float,
        provider_confirmation: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> "HandoffRequest":
        merged = dict(self.metadata)
        if metadata:
            merged.update({str(k): str(v) for k, v in metadata.items()})
        return HandoffRequest(
            request_id=self.request_id,
            status=status,
            proposal_id=self.proposal_id,
            decision_id=self.decision_id,
            descriptor_id=self.descriptor_id,
            logical_action=self.logical_action,
            tenant_id=self.tenant_id,
            session_id=self.session_id,
            channel=self.channel,
            queue=self.queue,
            priority=self.priority,
            reason=self.reason,
            summary=self.summary,
            summary_digest=self.summary_digest,
            created_at_epoch_s=self.created_at_epoch_s,
            updated_at_epoch_s=updated_at_epoch_s,
            provider_confirmation=(
                provider_confirmation
                if provider_confirmation is not None
                else self.provider_confirmation
            ),
            metadata=merged,
        )


class HandoffRequestStore(Protocol):
    """Backend protocol for durable handoff request storage."""

    def put(self, request: HandoffRequest) -> HandoffRequest:
        """Persist ``request`` (insert or replace) and return the stored record."""
        ...

    def get(self, request_id: str) -> HandoffRequest | None:
        """Return the request for ``request_id`` or ``None``."""
        ...

    def list_requests(
        self,
        *,
        tenant_id: str | None = None,
        status: ActionStatus | None = None,
        limit: int = 100,
    ) -> Sequence[HandoffRequest]:
        """List stored requests, optionally filtered."""
        ...


@dataclass
class InMemoryHandoffRequestStore:
    """Process-local durable store for unit tests and offline fakes.

    "Durable" here means requests survive across adapter invocations and can be
    reloaded by id; they are not ephemeral invoke-only side effects.
    """

    _by_id: dict[str, HandoffRequest] = field(default_factory=dict)

    def put(self, request: HandoffRequest) -> HandoffRequest:
        if not request.request_id:
            raise ValueError("request_id is required")
        self._by_id[request.request_id] = request
        return request

    def get(self, request_id: str) -> HandoffRequest | None:
        return self._by_id.get(request_id)

    def list_requests(
        self,
        *,
        tenant_id: str | None = None,
        status: ActionStatus | None = None,
        limit: int = 100,
    ) -> Sequence[HandoffRequest]:
        rows = list(self._by_id.values())
        if tenant_id is not None:
            rows = [r for r in rows if r.tenant_id == tenant_id]
        if status is not None:
            rows = [r for r in rows if r.status is status]
        rows.sort(key=lambda r: r.created_at_epoch_s, reverse=True)
        return tuple(rows[: max(0, limit)])


@dataclass
class FileHandoffRequestStore:
    """Directory-backed durable store (one JSON file per request).

    Atomic replace per write. Used when an operator wants requests to survive
    process restarts without a product database.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, request_id: str) -> Path:
        safe = _validate_id(request_id, role="request_id")
        return self.root / f"{safe}.json"

    def put(self, request: HandoffRequest) -> HandoffRequest:
        path = self._path_for(request.request_id)
        payload = {
            "request_id": request.request_id,
            "status": request.status.value,
            "proposal_id": request.proposal_id,
            "decision_id": request.decision_id,
            "descriptor_id": request.descriptor_id,
            "logical_action": request.logical_action,
            "tenant_id": request.tenant_id,
            "session_id": request.session_id,
            "channel": request.channel,
            "queue": request.queue,
            "priority": request.priority,
            "reason": request.reason,
            "summary": request.summary,
            "summary_digest": request.summary_digest,
            "created_at_epoch_s": request.created_at_epoch_s,
            "updated_at_epoch_s": request.updated_at_epoch_s,
            "provider_confirmation": request.provider_confirmation,
            "metadata": dict(request.metadata),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return request

    def get(self, request_id: str) -> HandoffRequest | None:
        path = self._path_for(request_id)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _request_from_json(raw)

    def list_requests(
        self,
        *,
        tenant_id: str | None = None,
        status: ActionStatus | None = None,
        limit: int = 100,
    ) -> Sequence[HandoffRequest]:
        rows: list[HandoffRequest] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                rows.append(_request_from_json(raw))
            except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
                continue
        if tenant_id is not None:
            rows = [r for r in rows if r.tenant_id == tenant_id]
        if status is not None:
            rows = [r for r in rows if r.status is status]
        rows.sort(key=lambda r: r.created_at_epoch_s, reverse=True)
        return tuple(rows[: max(0, limit)])


def _request_from_json(raw: Mapping[str, object]) -> HandoffRequest:
    summary = str(raw.get("summary") or "")
    digest = str(raw.get("summary_digest") or "") or content_digest(summary)
    meta_raw = raw.get("metadata") or {}
    if not isinstance(meta_raw, Mapping):
        meta_raw = {}
    return HandoffRequest(
        request_id=str(raw["request_id"]),
        status=ActionStatus(str(raw["status"])),
        proposal_id=str(raw["proposal_id"]),
        decision_id=str(raw["decision_id"]),
        descriptor_id=str(raw["descriptor_id"]),
        logical_action=str(raw.get("logical_action") or HANDOFF_LOGICAL_ACTION),
        tenant_id=(str(raw["tenant_id"]) if raw.get("tenant_id") is not None else None),
        session_id=(
            str(raw["session_id"]) if raw.get("session_id") is not None else None
        ),
        channel=(str(raw["channel"]) if raw.get("channel") is not None else None),
        queue=str(raw.get("queue") or "live_agent"),
        priority=str(raw.get("priority") or "normal"),
        reason=str(raw.get("reason") or ""),
        summary=summary,
        summary_digest=digest,
        created_at_epoch_s=float(raw.get("created_at_epoch_s") or 0.0),
        updated_at_epoch_s=float(raw.get("updated_at_epoch_s") or 0.0),
        provider_confirmation=(
            str(raw["provider_confirmation"])
            if raw.get("provider_confirmation") is not None
            else None
        ),
        metadata={str(k): str(v) for k, v in meta_raw.items()},
    )


@dataclass(frozen=True)
class HumanHandoffActionRegistration:
    """Reviewed human-handoff binding for a catalog descriptor."""

    descriptor_id: str
    logical_action: str = HANDOFF_LOGICAL_ACTION
    sandbox: HandoffSandboxPolicy = field(default_factory=HandoffSandboxPolicy)
    interface_name: str = "human_handoff"

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id is required")
        if self.logical_action not in _ALLOWED_LOGICAL_ACTIONS:
            raise ValueError(
                f"logical_action must be one of {sorted(_ALLOWED_LOGICAL_ACTIONS)}"
            )

    @property
    def interface_identity(self) -> str:
        return f"human_handoff:{self.logical_action}:{self.descriptor_id}"


@dataclass(frozen=True)
class HandoffInvocationContext:
    """Authority-plane facts available at the adapter boundary."""

    confirmed: bool = False
    authenticated: bool = False
    session_tenant_id: str | None = None
    # Optional operator override for queue / priority when proposal omits them.
    default_queue: str | None = None
    default_priority: str | None = None


def allows_spoken_success(
    status_or_receipt: ActionStatus | ActionReceipt | HandoffRequest | str | None,
) -> bool:
    """Return True only when spoken transfer-success is authorized.

    Spoken Abby ``success`` frames that claim a live-agent connection completed
    are forbidden unless the authority-plane receipt status is ``succeeded``.
    Request creation (``accepted``), in-flight (``started``), indeterminate
    (``unknown``), and failure states all return False.
    """

    status = _coerce_status(status_or_receipt)
    return status is ActionStatus.SUCCEEDED


def spoken_outcome_role(
    status_or_receipt: ActionStatus | ActionReceipt | HandoffRequest | str | None,
) -> str:
    """Map a receipt status to an action-link outcome frame role.

    Roles match ``docs/voice_action_dag/schemas/action-link-v1.md``:
    ``success`` | ``denied`` | ``failed`` | ``cancelled`` | ``unknown``.
    """

    status = _coerce_status(status_or_receipt)
    if status is ActionStatus.SUCCEEDED:
        return "success"
    if status is ActionStatus.DENIED:
        return "denied"
    if status is ActionStatus.FAILED:
        return "failed"
    if status is ActionStatus.CANCELLED:
        return "cancelled"
    # accepted / started / unknown / timed_out / missing → never "success"
    return "unknown"


def _coerce_status(
    status_or_receipt: ActionStatus | ActionReceipt | HandoffRequest | str | None,
) -> ActionStatus | None:
    if status_or_receipt is None:
        return None
    if isinstance(status_or_receipt, ActionStatus):
        return status_or_receipt
    if isinstance(status_or_receipt, ActionReceipt):
        return status_or_receipt.status
    if isinstance(status_or_receipt, HandoffRequest):
        return status_or_receipt.status
    if isinstance(status_or_receipt, str):
        try:
            return ActionStatus(status_or_receipt)
        except ValueError:
            return None
    return None


def _validate_id(value: str, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty string")
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"{role} has disallowed characters: {value!r}")
    if ".." in value:
        raise ValueError(f"{role} rejects path traversal: {value!r}")
    return value


def _validate_priority(priority: str) -> str:
    normalized = priority.strip().lower()
    if normalized not in _ALLOWED_PRIORITIES:
        raise ValueError(f"unsupported_priority:{priority!r}")
    return normalized


def _validate_channel(channel: str | None) -> str | None:
    if channel is None or channel == "":
        return None
    normalized = channel.strip().lower()
    if normalized not in _ALLOWED_CHANNELS:
        raise ValueError(f"unsupported_channel:{channel!r}")
    return normalized


def _validate_text(value: str, *, role: str, max_chars: int, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{role} must be a string")
    if "\x00" in value:
        raise ValueError(f"{role} rejects NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{role} must be non-empty")
    if len(value) > max_chars:
        raise ValueError(f"{role}_exceeds_max_chars:{max_chars}")
    return value


def _resolve_tenant(
    proposal: ActionProposal,
    context: HandoffInvocationContext,
) -> str | None:
    proposal_tenant = (proposal.tenant_id or "").strip() or None
    session_tenant = (context.session_tenant_id or "").strip() or None
    if proposal_tenant and session_tenant and proposal_tenant != session_tenant:
        raise ValueError("tenant_session_mismatch")
    return session_tenant or proposal_tenant


def _decision_admits_handoff_request(decision: ActionDecision) -> bool:
    """Handoff request creation is admitted by HANDOFF (not permit_execute)."""

    return decision.kind is ActionDecisionKind.HANDOFF


class HumanHandoffActionAdapter:
    """Create durable handoff requests and track transfer receipt status.

    Supports:

    * ``handoff_live_agent`` — create a durable ``HandoffRequest`` (status
      ``accepted``) after a policy ``handoff`` decision
    * transfer lifecycle updates: ``started`` / ``succeeded`` / ``unknown`` /
      ``failed`` via :meth:`mark_started` and :meth:`record_provider_outcome`
    """

    def __init__(
        self,
        registrations: Sequence[HumanHandoffActionRegistration],
        *,
        store: HandoffRequestStore | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._by_descriptor: dict[str, HumanHandoffActionRegistration] = {}
        for registration in registrations:
            if registration.descriptor_id in self._by_descriptor:
                raise ValueError(
                    f"duplicate handoff registration for {registration.descriptor_id!r}"
                )
            self._by_descriptor[registration.descriptor_id] = registration
        self._store: HandoffRequestStore = store or InMemoryHandoffRequestStore()
        self._clock: Callable[[], float] = clock or time.time

    @property
    def store(self) -> HandoffRequestStore:
        return self._store

    def get_registration(
        self, descriptor_id: str
    ) -> HumanHandoffActionRegistration | None:
        return self._by_descriptor.get(descriptor_id)

    def get_request(self, request_id: str) -> HandoffRequest | None:
        return self._store.get(request_id)

    def invoke(
        self,
        *,
        proposal: ActionProposal,
        decision: ActionDecision,
        context: HandoffInvocationContext | None = None,
    ) -> ActionReceipt:
        """Create a durable handoff request after a policy ``handoff`` decision.

        Returns an ``ActionReceipt`` with status ``accepted`` on success.
        Never returns ``succeeded`` from request creation alone.
        """

        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = float(self._clock())
        ctx = context or HandoffInvocationContext()

        if decision.kind is ActionDecisionKind.DENY:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="human_handoff",
                interface_identity="human_handoff:none",
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error=f"decision_denied:{decision.reason}",
            )

        if decision.kind is ActionDecisionKind.CONFIRM:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="human_handoff",
                interface_identity="human_handoff:none",
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error=f"decision_does_not_admit_handoff:{decision.kind.value}",
            )

        if not _decision_admits_handoff_request(decision):
            # permit_read / permit_execute / clarify are not the handoff path.
            # Handoff request creation must come from ActionDecisionKind.HANDOFF
            # so transfer success is never smuggled via permit_execute.
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="human_handoff",
                interface_identity="human_handoff:none",
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error=f"decision_does_not_admit_handoff:{decision.kind.value}",
            )

        bind_error = self._binding_error(proposal, decision)
        if bind_error is not None:
            return self._failed(
                receipt_id, proposal, decision, bind_error, started
            )

        registration = self._by_descriptor.get(proposal.descriptor_id)
        if registration is None:
            return self._failed(
                receipt_id, proposal, decision, "no_handoff_registration", started
            )

        if proposal.logical_action != registration.logical_action:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "logical_action_registration_mismatch",
                started,
                interface_identity=registration.interface_identity,
            )

        try:
            return self._create_request(
                receipt_id=receipt_id,
                proposal=proposal,
                decision=decision,
                registration=registration,
                context=ctx,
                started=started,
            )
        except ValueError as exc:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"handoff_rejected:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"handoff_error:{type(exc).__name__}:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )

    def mark_started(
        self,
        request_id: str,
        *,
        provider_confirmation: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ActionReceipt:
        """Advance an accepted request to ``started`` (transfer attempt begun)."""

        return self._transition(
            request_id,
            target=ActionStatus.STARTED,
            provider_confirmation=provider_confirmation,
            metadata=metadata,
            allow_from={ActionStatus.ACCEPTED, ActionStatus.STARTED},
        )

    def record_provider_outcome(
        self,
        request_id: str,
        *,
        status: ActionStatus | str,
        provider_confirmation: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ActionReceipt:
        """Record a terminal (or indeterminate) provider transfer outcome.

        Allowed statuses: ``succeeded``, ``failed``, ``unknown``, ``cancelled``.
        A fake telephony adapter may mark ``unknown`` without claiming success.
        """

        target = (
            status if isinstance(status, ActionStatus) else ActionStatus(str(status))
        )
        if target not in _PROVIDER_TERMINAL_STATUSES:
            raise ValueError(
                f"provider outcome status must be one of "
                f"{sorted(s.value for s in _PROVIDER_TERMINAL_STATUSES)}; got {target!r}"
            )
        if target is ActionStatus.SUCCEEDED and not provider_confirmation:
            # Fail closed: never claim transfer success without a confirmation token.
            raise ValueError("provider_confirmation_required_for_succeeded")

        return self._transition(
            request_id,
            target=target,
            provider_confirmation=provider_confirmation,
            metadata=metadata,
            allow_from={
                ActionStatus.ACCEPTED,
                ActionStatus.STARTED,
                ActionStatus.UNKNOWN,  # may refine unknown → succeeded later
            },
        )

    def _transition(
        self,
        request_id: str,
        *,
        target: ActionStatus,
        provider_confirmation: str | None,
        metadata: Mapping[str, str] | None,
        allow_from: set[ActionStatus],
    ) -> ActionReceipt:
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = float(self._clock())
        existing = self._store.get(request_id)
        if existing is None:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.FAILED,
                proposal_id="",
                decision_id="",
                descriptor_id="",
                adapter="human_handoff",
                interface_identity="human_handoff:none",
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error="handoff_request_not_found",
            )

        if existing.status is target:
            # Idempotent no-op: re-emit current state.
            return self._receipt_for_request(
                receipt_id=receipt_id,
                request=existing,
                registration_identity=self._identity_for(existing.descriptor_id),
                started=started,
            )

        if existing.status not in allow_from:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.FAILED,
                proposal_id=existing.proposal_id,
                decision_id=existing.decision_id,
                descriptor_id=existing.descriptor_id,
                adapter="human_handoff",
                interface_identity=self._identity_for(existing.descriptor_id),
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error=(
                    f"invalid_status_transition:"
                    f"{existing.status.value}->{target.value}"
                ),
                metadata={"request_id": existing.request_id},
            )

        # Prevent rank regressions (e.g. succeeded → started).
        if _STATUS_RANK.get(target, 0) < _STATUS_RANK.get(existing.status, 0):
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.FAILED,
                proposal_id=existing.proposal_id,
                decision_id=existing.decision_id,
                descriptor_id=existing.descriptor_id,
                adapter="human_handoff",
                interface_identity=self._identity_for(existing.descriptor_id),
                started_epoch_s=started,
                completed_epoch_s=float(self._clock()),
                error=(
                    f"status_regression_forbidden:"
                    f"{existing.status.value}->{target.value}"
                ),
                metadata={"request_id": existing.request_id},
            )

        now = float(self._clock())
        updated = existing.with_status(
            target,
            updated_at_epoch_s=now,
            provider_confirmation=provider_confirmation,
            metadata=metadata,
        )
        stored = self._store.put(updated)
        return self._receipt_for_request(
            receipt_id=receipt_id,
            request=stored,
            registration_identity=self._identity_for(stored.descriptor_id),
            started=started,
        )

    def _identity_for(self, descriptor_id: str) -> str:
        reg = self._by_descriptor.get(descriptor_id)
        if reg is not None:
            return reg.interface_identity
        return f"human_handoff:{HANDOFF_LOGICAL_ACTION}:{descriptor_id}"

    def _create_request(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: HumanHandoffActionRegistration,
        context: HandoffInvocationContext,
        started: float,
    ) -> ActionReceipt:
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        allowed = {"reason", "priority", "queue", "summary", "preferred_channel"}
        unexpected = set(args) - allowed
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")

        sandbox = registration.sandbox
        reason = _validate_text(
            args.get("reason") or decision.reason or "live_agent_requested",
            role="reason",
            max_chars=sandbox.max_reason_chars,
            allow_empty=False,
        )
        priority = _validate_priority(
            args.get("priority")
            or context.default_priority
            or sandbox.default_priority
        )
        queue_raw = (
            args.get("queue") or context.default_queue or sandbox.default_queue
        )
        queue = _validate_text(
            queue_raw,
            role="queue",
            max_chars=sandbox.max_queue_chars,
            allow_empty=False,
        )
        # Queues are identifiers, not free text.
        queue = _validate_id(queue, role="queue")
        summary = _validate_text(
            args.get("summary") or "",
            role="summary",
            max_chars=sandbox.max_summary_chars,
            allow_empty=True,
        )
        channel = _validate_channel(
            args.get("preferred_channel") or proposal.channel
        )

        now = float(self._clock())
        request = HandoffRequest(
            request_id=f"hoff-{uuid.uuid4().hex[:16]}",
            status=ActionStatus.ACCEPTED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            logical_action=proposal.logical_action,
            tenant_id=tenant_id,
            session_id=proposal.session_id,
            channel=channel,
            queue=queue,
            priority=priority,
            reason=reason,
            summary=summary,
            summary_digest=content_digest(summary) if summary else content_digest(""),
            created_at_epoch_s=now,
            updated_at_epoch_s=now,
            provider_confirmation=None,
            metadata={
                "decision_reason": decision.reason,
                "decision_kind": decision.kind.value,
            },
        )
        stored = self._store.put(request)
        # Defense: store must return the same id/status.
        if stored.request_id != request.request_id:
            raise ValueError("store_request_id_mismatch")
        if stored.status is not ActionStatus.ACCEPTED:
            raise ValueError("store_must_persist_accepted_on_create")

        return self._receipt_for_request(
            receipt_id=receipt_id,
            request=stored,
            registration_identity=registration.interface_identity,
            started=started,
            redact_summary=sandbox.redact_summary_in_receipts,
        )

    def _receipt_for_request(
        self,
        *,
        receipt_id: str,
        request: HandoffRequest,
        registration_identity: str,
        started: float,
        redact_summary: bool = True,
    ) -> ActionReceipt:
        public = self._public_result(request, redact_summary=redact_summary)
        return ActionReceipt(
            receipt_id=receipt_id,
            status=request.status,
            proposal_id=request.proposal_id,
            decision_id=request.decision_id,
            descriptor_id=request.descriptor_id,
            adapter="human_handoff",
            interface_identity=registration_identity,
            started_epoch_s=started,
            completed_epoch_s=float(self._clock()),
            public_result=public,
            metadata={
                "logical_action": request.logical_action,
                "request_id": request.request_id,
                "handoff_status": request.status.value,
                "spoken_success_allowed": (
                    "true" if allows_spoken_success(request) else "false"
                ),
            },
        )

    def _public_result(
        self,
        request: HandoffRequest,
        *,
        redact_summary: bool,
    ) -> dict[str, str]:
        public: dict[str, str] = {
            "ok": "true" if request.status is not ActionStatus.FAILED else "false",
            "request_id": request.request_id,
            "handoff_status": request.status.value,
            "phase": request.phase.value,
            "queue": request.queue,
            "priority": request.priority,
            "reason": request.reason,
            "summary_digest": request.summary_digest,
            "is_transfer_complete": "true" if request.is_transfer_complete else "false",
            "spoken_success_allowed": (
                "true" if allows_spoken_success(request) else "false"
            ),
            "summary_redacted": "true" if redact_summary else "false",
        }
        if request.tenant_id is not None:
            public["tenant_id"] = request.tenant_id
        if request.session_id is not None:
            public["session_id"] = request.session_id
        if request.channel is not None:
            public["channel"] = request.channel
        if request.provider_confirmation is not None:
            public["provider_confirmation"] = request.provider_confirmation
        if not redact_summary and request.summary:
            public["summary"] = request.summary
        else:
            public["summary_present"] = "true" if request.summary else "false"
        return public

    def _binding_error(
        self,
        proposal: ActionProposal,
        decision: ActionDecision,
    ) -> str | None:
        if decision.proposal_id != proposal.proposal_id:
            return "proposal_decision_mismatch"
        if decision.descriptor_id != proposal.descriptor_id:
            return "descriptor_decision_mismatch"
        if decision.arguments_digest != proposal.arguments_digest:
            return "arguments_digest_mismatch"
        if (
            decision.expires_at_epoch_s is not None
            and float(self._clock()) > decision.expires_at_epoch_s
        ):
            return "decision_expired"
        return None

    def _failed(
        self,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        error: str,
        started: float,
        *,
        interface_identity: str = "human_handoff:none",
    ) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="human_handoff",
            interface_identity=interface_identity,
            started_epoch_s=started,
            completed_epoch_s=float(self._clock()),
            error=error,
            metadata={"spoken_success_allowed": "false"},
        )


def default_handoff_registrations() -> tuple[HumanHandoffActionRegistration, ...]:
    """Pilot descriptor registration for ``handoff_live_agent``."""

    return (
        HumanHandoffActionRegistration(
            descriptor_id="voice.human.handoff_live_agent.v1",
            logical_action=HANDOFF_LOGICAL_ACTION,
        ),
    )
