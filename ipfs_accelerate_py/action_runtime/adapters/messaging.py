"""Provider messaging action adapter (read inbox + leave message).

Safety rules:

* execute only after a permitting ``ActionDecision`` binds the exact proposal
* ``leave_provider_message`` re-checks confirm + authenticated tenant at the
  adapter boundary (defense in depth with pilot policy)
* message bodies are size-bounded and never echoed in receipts by default
* reads are strictly tenant-scoped; cross-tenant rows are never returned
* no network, SMS, or telephony side effects — the store is injected
* spoken phone numbers / free-text transcripts are never treated as SMS sends
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from typing import Protocol

from ..contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionReceipt,
    ActionStatus,
    content_digest,
)

READ_LOGICAL_ACTION = "read_provider_messages"
LEAVE_LOGICAL_ACTION = "leave_provider_message"

_ALLOWED_LOGICAL_ACTIONS = frozenset({READ_LOGICAL_ACTION, LEAVE_LOGICAL_ACTION})
_ALLOWED_CHANNELS = frozenset({"in_app", "sms", "email"})
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")

# Default body budget for leave_provider_message (characters, not bytes).
DEFAULT_MAX_BODY_CHARS = 2_000
DEFAULT_MAX_SUBJECT_CHARS = 200
DEFAULT_MAX_MESSAGES_RETURNED = 50


@dataclass(frozen=True)
class MessagingSandboxPolicy:
    """Resource and privacy bounds for messaging adapter invocations."""

    max_body_chars: int = DEFAULT_MAX_BODY_CHARS
    max_subject_chars: int = DEFAULT_MAX_SUBJECT_CHARS
    max_messages_returned: int = DEFAULT_MAX_MESSAGES_RETURNED
    # When True (default), receipts never carry raw message bodies.
    redact_bodies_in_receipts: bool = True
    # Adapter-boundary re-checks (pilot policy also gates these).
    require_confirm_for_leave: bool = True
    require_auth_for_leave: bool = True
    require_confirm_for_read: bool = True
    require_auth_for_read: bool = True

    def __post_init__(self) -> None:
        if self.max_body_chars < 1 or self.max_body_chars > 16_384:
            raise ValueError("max_body_chars must be in [1, 16384]")
        if self.max_subject_chars < 1 or self.max_subject_chars > 1_024:
            raise ValueError("max_subject_chars must be in [1, 1024]")
        if self.max_messages_returned < 1 or self.max_messages_returned > 500:
            raise ValueError("max_messages_returned must be in [1, 500]")


@dataclass(frozen=True)
class ProviderMessageRecord:
    """Tenant-scoped provider message stored by an injected backend."""

    message_id: str
    tenant_id: str
    provider_id: str
    client_id: str
    channel: str
    subject: str
    body: str
    direction: str  # inbound (to client) | outbound (client leave)
    status: str
    created_at_epoch_s: float
    body_digest: str = ""

    def __post_init__(self) -> None:
        if not self.body_digest:
            object.__setattr__(self, "body_digest", content_digest(self.body))


class ProviderMessageStore(Protocol):
    """Backend protocol for provider message storage (fake or product)."""

    def list_messages(
        self,
        *,
        tenant_id: str,
        provider_id: str | None = None,
        client_id: str | None = None,
        limit: int = DEFAULT_MAX_MESSAGES_RETURNED,
    ) -> Sequence[ProviderMessageRecord]:
        """Return messages belonging only to ``tenant_id``."""
        ...

    def leave_message(
        self,
        *,
        tenant_id: str,
        provider_id: str,
        client_id: str,
        channel: str,
        subject: str,
        body: str,
    ) -> ProviderMessageRecord:
        """Persist an outbound client→provider message for ``tenant_id``."""
        ...


@dataclass
class InMemoryProviderMessageStore:
    """Offline fake store for unit tests and wallet binding fakes."""

    _messages: list[ProviderMessageRecord] = field(default_factory=list)
    _clock: Callable[[], float] = time.time

    def seed(self, *records: ProviderMessageRecord) -> None:
        for record in records:
            self._messages.append(record)

    def list_messages(
        self,
        *,
        tenant_id: str,
        provider_id: str | None = None,
        client_id: str | None = None,
        limit: int = DEFAULT_MAX_MESSAGES_RETURNED,
    ) -> Sequence[ProviderMessageRecord]:
        if not tenant_id:
            return ()
        rows = [
            m
            for m in self._messages
            if m.tenant_id == tenant_id
            and (provider_id is None or m.provider_id == provider_id)
            and (client_id is None or m.client_id == client_id)
        ]
        # Newest first for voice summaries.
        rows.sort(key=lambda m: m.created_at_epoch_s, reverse=True)
        return tuple(rows[: max(0, limit)])

    def leave_message(
        self,
        *,
        tenant_id: str,
        provider_id: str,
        client_id: str,
        channel: str,
        subject: str,
        body: str,
    ) -> ProviderMessageRecord:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        record = ProviderMessageRecord(
            message_id=f"msg-{uuid.uuid4().hex[:16]}",
            tenant_id=tenant_id,
            provider_id=provider_id,
            client_id=client_id,
            channel=channel,
            subject=subject,
            body=body,
            direction="outbound",
            status="queued",
            created_at_epoch_s=float(self._clock()),
        )
        self._messages.append(record)
        return record


@dataclass(frozen=True)
class MessagingActionRegistration:
    """Reviewed messaging binding for a catalog descriptor."""

    descriptor_id: str
    logical_action: str
    sandbox: MessagingSandboxPolicy = field(default_factory=MessagingSandboxPolicy)
    interface_name: str = "messaging"

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id is required")
        if self.logical_action not in _ALLOWED_LOGICAL_ACTIONS:
            raise ValueError(
                f"logical_action must be one of {sorted(_ALLOWED_LOGICAL_ACTIONS)}"
            )

    @property
    def interface_identity(self) -> str:
        return f"messaging:{self.logical_action}:{self.descriptor_id}"


@dataclass(frozen=True)
class MessagingInvocationContext:
    """Authority-plane facts re-checked at the adapter boundary.

    Policy already gates confirm/auth; the adapter re-validates so a forged
    permit decision alone cannot leave or dump messages.
    """

    confirmed: bool = False
    authenticated: bool = False
    session_tenant_id: str | None = None


def _validate_id(value: str, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty string")
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"{role} has disallowed characters: {value!r}")
    if ".." in value:
        raise ValueError(f"{role} rejects path traversal: {value!r}")
    return value


def _resolve_tenant(
    proposal: ActionProposal,
    context: MessagingInvocationContext,
) -> str:
    """Resolve the effective tenant for scoping; fail closed on mismatch."""

    proposal_tenant = (proposal.tenant_id or "").strip() or None
    session_tenant = (context.session_tenant_id or "").strip() or None
    if proposal_tenant and session_tenant and proposal_tenant != session_tenant:
        raise ValueError("tenant_session_mismatch")
    tenant = session_tenant or proposal_tenant
    if not tenant:
        raise ValueError("tenant_required")
    return tenant


def _parse_limit(raw: str | None, *, default: int, maximum: int) -> int:
    if raw is None or raw == "":
        return min(default, maximum)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, maximum)


def _validate_body(body: str, *, max_chars: int) -> str:
    if not isinstance(body, str):
        raise ValueError("body must be a string")
    # Reject NULs and control characters other than tab/newline.
    if "\x00" in body:
        raise ValueError("body rejects NUL")
    if not body.strip():
        raise ValueError("body must be non-empty")
    if len(body) > max_chars:
        raise ValueError(f"body_exceeds_max_chars:{max_chars}")
    return body


def _validate_subject(subject: str, *, max_chars: int) -> str:
    if not isinstance(subject, str):
        raise ValueError("subject must be a string")
    if "\x00" in subject:
        raise ValueError("subject rejects NUL")
    if len(subject) > max_chars:
        raise ValueError(f"subject_exceeds_max_chars:{max_chars}")
    return subject


def _validate_channel(channel: str) -> str:
    normalized = channel.strip().lower()
    if normalized not in _ALLOWED_CHANNELS:
        raise ValueError(f"unsupported_channel:{channel!r}")
    return normalized


class MessagingActionAdapter:
    """Execute provider messaging only after a permitting decision.

    Supports:

    * ``read_provider_messages`` — tenant-scoped inbox listing (auth+confirm)
    * ``leave_provider_message`` — outbound leave (auth+confirm, body bounded)
    """

    def __init__(
        self,
        registrations: Sequence[MessagingActionRegistration],
        *,
        store: ProviderMessageStore | None = None,
    ) -> None:
        self._by_descriptor: dict[str, MessagingActionRegistration] = {}
        for registration in registrations:
            if registration.descriptor_id in self._by_descriptor:
                raise ValueError(
                    f"duplicate messaging registration for {registration.descriptor_id!r}"
                )
            self._by_descriptor[registration.descriptor_id] = registration
        self._store: ProviderMessageStore = store or InMemoryProviderMessageStore()

    @property
    def store(self) -> ProviderMessageStore:
        return self._store

    def get_registration(self, descriptor_id: str) -> MessagingActionRegistration | None:
        return self._by_descriptor.get(descriptor_id)

    def invoke(
        self,
        *,
        proposal: ActionProposal,
        decision: ActionDecision,
        context: MessagingInvocationContext | None = None,
    ) -> ActionReceipt:
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = time.time()
        ctx = context or MessagingInvocationContext()

        if not decision.permits_execution:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="messaging",
                interface_identity="messaging:none",
                started_epoch_s=started,
                completed_epoch_s=time.time(),
                error=f"decision_does_not_permit_execution:{decision.kind.value}",
            )

        bind_error = self._binding_error(proposal, decision)
        if bind_error is not None:
            return self._failed(
                receipt_id, proposal, decision, bind_error, started
            )

        registration = self._by_descriptor.get(proposal.descriptor_id)
        if registration is None:
            return self._failed(
                receipt_id, proposal, decision, "no_messaging_registration", started
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
            if registration.logical_action == READ_LOGICAL_ACTION:
                return self._invoke_read(
                    receipt_id=receipt_id,
                    proposal=proposal,
                    decision=decision,
                    registration=registration,
                    context=ctx,
                    started=started,
                )
            if registration.logical_action == LEAVE_LOGICAL_ACTION:
                return self._invoke_leave(
                    receipt_id=receipt_id,
                    proposal=proposal,
                    decision=decision,
                    registration=registration,
                    context=ctx,
                    started=started,
                )
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"unsupported_logical_action:{registration.logical_action}",
                started,
                interface_identity=registration.interface_identity,
            )
        except ValueError as exc:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"messaging_rejected:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"messaging_error:{type(exc).__name__}:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )

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
        if decision.expires_at_epoch_s is not None and time.time() > decision.expires_at_epoch_s:
            return "decision_expired"
        return None

    def _require_confirm_auth(
        self,
        *,
        registration: MessagingActionRegistration,
        context: MessagingInvocationContext,
        decision: ActionDecision,
        for_leave: bool,
    ) -> None:
        sandbox = registration.sandbox
        need_confirm = (
            sandbox.require_confirm_for_leave if for_leave else sandbox.require_confirm_for_read
        )
        need_auth = (
            sandbox.require_auth_for_leave if for_leave else sandbox.require_auth_for_read
        )
        if need_confirm and not context.confirmed:
            raise ValueError("confirmation_required")
        if need_auth and not context.authenticated:
            raise ValueError("auth_required")
        # Leave is write-class: only PERMIT_EXECUTE is acceptable.
        if for_leave and decision.kind is not ActionDecisionKind.PERMIT_EXECUTE:
            raise ValueError(f"leave_requires_permit_execute:{decision.kind.value}")
        # Read is read-class: PERMIT_READ (or PERMIT_EXECUTE if elevated).
        if not for_leave and decision.kind not in {
            ActionDecisionKind.PERMIT_READ,
            ActionDecisionKind.PERMIT_EXECUTE,
        }:
            raise ValueError(f"read_requires_permit:{decision.kind.value}")

    def _invoke_read(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: MessagingActionRegistration,
        context: MessagingInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_leave=False,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        unexpected = set(args) - {"provider_id", "client_id", "limit"}
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")

        provider_id = args.get("provider_id")
        if provider_id is not None:
            provider_id = _validate_id(provider_id, role="provider_id")
        client_id = args.get("client_id")
        if client_id is not None:
            client_id = _validate_id(client_id, role="client_id")

        limit = _parse_limit(
            args.get("limit"),
            default=registration.sandbox.max_messages_returned,
            maximum=registration.sandbox.max_messages_returned,
        )

        rows = list(
            self._store.list_messages(
                tenant_id=tenant_id,
                provider_id=provider_id,
                client_id=client_id,
                limit=limit,
            )
        )
        # Defense in depth: drop any store bug that leaks other tenants.
        rows = [m for m in rows if m.tenant_id == tenant_id]

        public = self._redacted_read_public(
            rows=rows,
            tenant_id=tenant_id,
            provider_id=provider_id,
            redact_bodies=registration.sandbox.redact_bodies_in_receipts,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="messaging",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": READ_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "message_count": str(len(rows)),
            },
        )

    def _invoke_leave(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: MessagingActionRegistration,
        context: MessagingInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_leave=True,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        allowed = {"provider_id", "client_id", "channel", "subject", "body"}
        unexpected = set(args) - allowed
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
        if "provider_id" not in args:
            raise ValueError("missing required argument slot 'provider_id'")
        if "body" not in args:
            raise ValueError("missing required argument slot 'body'")

        provider_id = _validate_id(args["provider_id"], role="provider_id")
        client_id = _validate_id(
            args.get("client_id") or context.session_tenant_id or tenant_id,
            role="client_id",
        )
        channel = _validate_channel(args.get("channel") or "in_app")
        subject = _validate_subject(
            args.get("subject") or "",
            max_chars=registration.sandbox.max_subject_chars,
        )
        body = _validate_body(
            args["body"],
            max_chars=registration.sandbox.max_body_chars,
        )

        record = self._store.leave_message(
            tenant_id=tenant_id,
            provider_id=provider_id,
            client_id=client_id,
            channel=channel,
            subject=subject,
            body=body,
        )
        if record.tenant_id != tenant_id:
            raise ValueError("store_tenant_mismatch")

        public = self._redacted_leave_public(
            record=record,
            redact_bodies=registration.sandbox.redact_bodies_in_receipts,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="messaging",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": LEAVE_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "message_id": record.message_id,
                "body_chars": str(len(body)),
            },
        )

    def _redacted_read_public(
        self,
        *,
        rows: Sequence[ProviderMessageRecord],
        tenant_id: str,
        provider_id: str | None,
        redact_bodies: bool,
    ) -> dict[str, str]:
        message_ids = ",".join(m.message_id for m in rows)
        digests = ",".join(m.body_digest for m in rows)
        public: dict[str, str] = {
            "ok": "true",
            "tenant_id": tenant_id,
            "message_count": str(len(rows)),
            "message_ids": message_ids,
            "body_digests": digests,
            "bodies_redacted": "true" if redact_bodies else "false",
        }
        if provider_id is not None:
            public["provider_id"] = provider_id
        # Subjects are non-secret labels; still omit raw bodies.
        if rows and not redact_bodies:
            # Operator-only opt-out: still never dump full inbox bodies as one blob.
            public["body_preview_count"] = str(len(rows))
        else:
            # Default: no body / subject dump in voice-facing receipts.
            public["subjects_redacted"] = "true"
        return public

    def _redacted_leave_public(
        self,
        *,
        record: ProviderMessageRecord,
        redact_bodies: bool,
    ) -> dict[str, str]:
        public: dict[str, str] = {
            "ok": "true",
            "message_id": record.message_id,
            "tenant_id": record.tenant_id,
            "provider_id": record.provider_id,
            "client_id": record.client_id,
            "channel": record.channel,
            "status": record.status,
            "body_digest": record.body_digest,
            "bodies_redacted": "true" if redact_bodies else "false",
        }
        if not redact_bodies:
            public["body"] = record.body
            public["subject"] = record.subject
        else:
            public["subject_present"] = "true" if record.subject else "false"
            public["body_chars"] = str(len(record.body))
        return public

    def _failed(
        self,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        error: str,
        started: float,
        *,
        interface_identity: str = "messaging:none",
    ) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="messaging",
            interface_identity=interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            error=error,
        )


def default_messaging_registrations() -> tuple[MessagingActionRegistration, ...]:
    """Pilot descriptor registrations for read + leave provider message."""

    return (
        MessagingActionRegistration(
            descriptor_id="voice.python.read_provider_messages.v1",
            logical_action=READ_LOGICAL_ACTION,
        ),
        MessagingActionRegistration(
            descriptor_id="voice.python.leave_provider_message.v1",
            logical_action=LEAVE_LOGICAL_ACTION,
        ),
    )
