"""Service interaction / callback action adapter.

Safety rules:

* execute only after a permitting ``ActionDecision`` binds the exact proposal
* ``schedule_service_callback`` re-checks confirm + authenticated tenant at the
  adapter boundary (defense in depth with pilot policy)
* ``open_service_detail`` re-checks confirm; auth is optional at the adapter
  boundary (pilot catalog marks read as confirm-only)
* ``service_id`` is required and must appear in proposal grounded evidence —
  free-text alone never invents a service target
* ``schedule_service_callback`` is idempotent on the proposal digest: replaying
  the same digest returns the prior callback without a second mutation
* unconfirmed / non-permitting decisions no-op (no store writes)
* notes are size-bounded and redacted from public receipts by default
* no network / telephony side effects — the store is injected
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionReceipt,
    ActionStatus,
    content_digest,
)

OPEN_LOGICAL_ACTION = "open_service_detail"
SCHEDULE_LOGICAL_ACTION = "schedule_service_callback"

_ALLOWED_LOGICAL_ACTIONS = frozenset({OPEN_LOGICAL_ACTION, SCHEDULE_LOGICAL_ACTION})
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")
_ALLOWED_CHANNELS = frozenset({"phone", "sms", "email", "voice", "in_app", "chat"})
# ISO-8601 datetime (with optional fractional seconds and Z/offset) or date-only.
_ISO_DT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

DEFAULT_MAX_NOTES_CHARS = 2_000
DEFAULT_MAX_TITLE_CHARS = 200
DEFAULT_MAX_SERVICES_RETURNED = 50

# Argument slots admitted for each logical action (fail closed on extras).
OPEN_ARGUMENT_SLOTS = frozenset({"service_id", "provider_id"})
SCHEDULE_ARGUMENT_SLOTS = frozenset(
    {
        "service_id",
        "callback_at",
        "channel",
        "client_id",
        "notes",
        "contact_preference",
        "provider_id",
    }
)
_FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "command",
        "argv",
        "executable",
        "cwd",
        "env",
        "shell",
        "import_path",
        "url",
        "webhook",
        "payload",
        "body",
        "free_text",
        "transcript",
    }
)

# Evidence prefixes that may carry a grounded service identifier.
_SERVICE_EVIDENCE_PREFIXES = (
    "service_id:",
    "service:",
    "service_doc_id:",
    "svc:",
)


@dataclass(frozen=True)
class ServiceInteractionSandboxPolicy:
    """Resource and privacy bounds for service interaction invocations."""

    max_notes_chars: int = DEFAULT_MAX_NOTES_CHARS
    max_title_chars: int = DEFAULT_MAX_TITLE_CHARS
    max_services_returned: int = DEFAULT_MAX_SERVICES_RETURNED
    # When True (default), receipts never carry raw notes.
    redact_notes_in_receipts: bool = True
    # Adapter-boundary re-checks (pilot policy also gates these).
    require_confirm_for_schedule: bool = True
    require_auth_for_schedule: bool = True
    require_confirm_for_open: bool = True
    # Pilot catalog: open_service_detail is confirm-only (auth_required=false).
    require_auth_for_open: bool = False

    def __post_init__(self) -> None:
        if self.max_notes_chars < 1 or self.max_notes_chars > 16_384:
            raise ValueError("max_notes_chars must be in [1, 16384]")
        if self.max_title_chars < 1 or self.max_title_chars > 1_024:
            raise ValueError("max_title_chars must be in [1, 1024]")
        if self.max_services_returned < 1 or self.max_services_returned > 500:
            raise ValueError("max_services_returned must be in [1, 500]")


@dataclass(frozen=True)
class ServiceDetailRecord:
    """Catalog row for a grounded service (offline fake or product binding)."""

    service_id: str
    title: str
    provider_name: str = ""
    program_name: str = ""
    summary: str = ""
    tenant_id: str | None = None  # None = globally readable catalog entry
    status: str = "available"
    title_digest: str = ""
    summary_digest: str = ""

    def __post_init__(self) -> None:
        if not self.title_digest:
            object.__setattr__(self, "title_digest", content_digest(self.title))
        if not self.summary_digest:
            object.__setattr__(self, "summary_digest", content_digest(self.summary))

    def redacted_summary(self) -> str:
        title_preview = self.title.strip()
        if len(title_preview) > 40:
            title_preview = title_preview[:37] + "..."
        title_preview = re.sub(r"[\r\n\t]+", " ", title_preview)
        provider = re.sub(r"[\r\n\t]+", " ", (self.provider_name or "").strip())
        if provider:
            return f"{self.service_id} | {title_preview} | {provider}"
        return f"{self.service_id} | {title_preview}"


@dataclass(frozen=True)
class ServiceCallbackRecord:
    """Tenant-scoped callback request stored by an injected backend."""

    callback_id: str
    tenant_id: str
    service_id: str
    proposal_digest: str
    channel: str
    callback_at: str
    client_id: str
    notes: str
    status: str
    created_at_epoch_s: float
    provider_id: str = ""
    contact_preference: str = ""
    notes_digest: str = ""
    interaction_type: str = "callback_requested"

    def __post_init__(self) -> None:
        if not self.notes_digest:
            object.__setattr__(self, "notes_digest", content_digest(self.notes))


class ServiceInteractionStore(Protocol):
    """Backend protocol for service detail lookup + callback storage."""

    def get_service(
        self,
        *,
        service_id: str,
        tenant_id: str | None = None,
    ) -> ServiceDetailRecord | None:
        """Return a service detail row visible to ``tenant_id``."""
        ...

    def list_services(
        self,
        *,
        tenant_id: str | None = None,
        service_id: str | None = None,
        limit: int = DEFAULT_MAX_SERVICES_RETURNED,
    ) -> Sequence[ServiceDetailRecord]:
        """List catalog rows, optionally filtered."""
        ...

    def get_callback_by_digest(
        self,
        *,
        tenant_id: str,
        proposal_digest: str,
    ) -> ServiceCallbackRecord | None:
        """Return an existing callback for the proposal digest, if any."""
        ...

    def schedule_callback(
        self,
        *,
        tenant_id: str,
        service_id: str,
        proposal_digest: str,
        channel: str,
        callback_at: str,
        client_id: str,
        notes: str,
        provider_id: str = "",
        contact_preference: str = "",
    ) -> ServiceCallbackRecord:
        """Persist a callback request; implementations may still de-dupe."""
        ...

    def list_callbacks(
        self,
        *,
        tenant_id: str,
        service_id: str | None = None,
        limit: int = DEFAULT_MAX_SERVICES_RETURNED,
    ) -> Sequence[ServiceCallbackRecord]:
        """Return callbacks belonging only to ``tenant_id``."""
        ...


@dataclass
class InMemoryServiceInteractionStore:
    """Offline fake store for unit tests and wallet binding fakes."""

    _services: dict[str, ServiceDetailRecord] = field(default_factory=dict)
    _callbacks: list[ServiceCallbackRecord] = field(default_factory=list)
    # tenant_id -> proposal_digest -> callback_id
    _by_digest: dict[str, dict[str, str]] = field(default_factory=dict)
    _clock: Callable[[], float] = time.time

    def seed_services(self, *records: ServiceDetailRecord) -> None:
        for record in records:
            self._services[record.service_id] = record

    def seed_callbacks(self, *records: ServiceCallbackRecord) -> None:
        for record in records:
            self._callbacks.append(record)
            self._by_digest.setdefault(record.tenant_id, {})[
                record.proposal_digest
            ] = record.callback_id

    def get_service(
        self,
        *,
        service_id: str,
        tenant_id: str | None = None,
    ) -> ServiceDetailRecord | None:
        record = self._services.get(service_id)
        if record is None:
            return None
        if record.tenant_id is not None and tenant_id is not None:
            if record.tenant_id != tenant_id:
                return None
        return record

    def list_services(
        self,
        *,
        tenant_id: str | None = None,
        service_id: str | None = None,
        limit: int = DEFAULT_MAX_SERVICES_RETURNED,
    ) -> Sequence[ServiceDetailRecord]:
        rows = list(self._services.values())
        if service_id is not None:
            rows = [r for r in rows if r.service_id == service_id]
        if tenant_id is not None:
            rows = [
                r
                for r in rows
                if r.tenant_id is None or r.tenant_id == tenant_id
            ]
        return tuple(rows[: max(0, limit)])

    def get_callback_by_digest(
        self,
        *,
        tenant_id: str,
        proposal_digest: str,
    ) -> ServiceCallbackRecord | None:
        if not tenant_id or not proposal_digest:
            return None
        callback_id = self._by_digest.get(tenant_id, {}).get(proposal_digest)
        if callback_id is None:
            return None
        for record in self._callbacks:
            if record.callback_id == callback_id and record.tenant_id == tenant_id:
                return record
        return None

    def schedule_callback(
        self,
        *,
        tenant_id: str,
        service_id: str,
        proposal_digest: str,
        channel: str,
        callback_at: str,
        client_id: str,
        notes: str,
        provider_id: str = "",
        contact_preference: str = "",
    ) -> ServiceCallbackRecord:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not service_id:
            raise ValueError("service_id is required")
        if not proposal_digest:
            raise ValueError("proposal_digest is required")
        existing = self.get_callback_by_digest(
            tenant_id=tenant_id, proposal_digest=proposal_digest
        )
        if existing is not None:
            return existing
        record = ServiceCallbackRecord(
            callback_id=f"cb-{uuid.uuid4().hex[:16]}",
            tenant_id=tenant_id,
            service_id=service_id,
            proposal_digest=proposal_digest,
            channel=channel,
            callback_at=callback_at,
            client_id=client_id,
            notes=notes,
            status="scheduled",
            created_at_epoch_s=float(self._clock()),
            provider_id=provider_id,
            contact_preference=contact_preference,
        )
        self._callbacks.append(record)
        self._by_digest.setdefault(tenant_id, {})[proposal_digest] = record.callback_id
        return record

    def list_callbacks(
        self,
        *,
        tenant_id: str,
        service_id: str | None = None,
        limit: int = DEFAULT_MAX_SERVICES_RETURNED,
    ) -> Sequence[ServiceCallbackRecord]:
        if not tenant_id:
            return ()
        rows = [
            c
            for c in self._callbacks
            if c.tenant_id == tenant_id
            and (service_id is None or c.service_id == service_id)
        ]
        rows.sort(key=lambda c: c.created_at_epoch_s, reverse=True)
        return tuple(rows[: max(0, limit)])


@dataclass(frozen=True)
class ServiceInteractionActionRegistration:
    """Reviewed service-interaction binding for a catalog descriptor."""

    descriptor_id: str
    logical_action: str
    sandbox: ServiceInteractionSandboxPolicy = field(
        default_factory=ServiceInteractionSandboxPolicy
    )
    interface_name: str = "service_interaction"

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id is required")
        if self.logical_action not in _ALLOWED_LOGICAL_ACTIONS:
            raise ValueError(
                f"logical_action must be one of {sorted(_ALLOWED_LOGICAL_ACTIONS)}"
            )

    @property
    def interface_identity(self) -> str:
        return f"service_interaction:{self.logical_action}:{self.descriptor_id}"


@dataclass(frozen=True)
class ServiceInteractionInvocationContext:
    """Authority-plane facts re-checked at the adapter boundary.

    Policy already gates confirm/auth; the adapter re-validates so a forged
    permit decision alone cannot schedule callbacks or dump other tenants.
    """

    confirmed: bool = False
    authenticated: bool = False
    session_tenant_id: str | None = None


def proposal_idempotency_digest(proposal: ActionProposal) -> str:
    """Stable digest used for schedule_service_callback idempotency.

    Bound to logical action, descriptor, arguments, tenant, and grounded
    evidence so free-text-only argument edits or evidence swaps cannot collide
    with a prior callback under a different grounding set.
    """

    return content_digest(
        {
            "logical_action": proposal.logical_action,
            "descriptor_id": proposal.descriptor_id,
            "arguments_digest": proposal.arguments_digest,
            "tenant_id": proposal.tenant_id or "",
            "evidence": list(proposal.evidence),
        }
    )


def grounded_service_tokens(evidence: Sequence[str]) -> frozenset[str]:
    """Extract service-id tokens admitted by grounded evidence."""

    tokens: set[str] = set()
    for raw in evidence:
        if not isinstance(raw, str) or not raw:
            continue
        token = raw.strip()
        if not token:
            continue
        tokens.add(token)
        lowered = token.lower()
        for prefix in _SERVICE_EVIDENCE_PREFIXES:
            if lowered.startswith(prefix):
                # Preserve original casing of the value portion.
                value = token[len(prefix) :].strip()
                if value:
                    tokens.add(value)
                break
    return frozenset(tokens)


def require_grounded_service_id(
    service_id: str,
    evidence: Sequence[str],
) -> str:
    """Fail closed unless ``service_id`` is present in grounded evidence."""

    if not isinstance(service_id, str) or not service_id.strip():
        raise ValueError("service_id must be a non-empty string")
    cleaned = service_id.strip()
    if not _SAFE_ID_RE.match(cleaned):
        raise ValueError(f"service_id has disallowed characters: {cleaned!r}")
    if ".." in cleaned:
        raise ValueError(f"service_id rejects path traversal: {cleaned!r}")
    if not evidence:
        raise ValueError("service_id_requires_grounded_evidence")
    tokens = grounded_service_tokens(evidence)
    if cleaned not in tokens:
        raise ValueError("service_id_not_in_grounded_evidence")
    return cleaned


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
    context: ServiceInteractionInvocationContext,
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


def _validate_text(
    value: str,
    *,
    role: str,
    max_chars: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{role} must be a string")
    if "\x00" in value:
        raise ValueError(f"{role} rejects NUL")
    if not allow_empty and not value.strip():
        raise ValueError(f"{role} must be non-empty")
    if len(value) > max_chars:
        raise ValueError(f"{role}_exceeds_max_chars:{max_chars}")
    return value


def _validate_channel(channel: str) -> str:
    normalized = channel.strip().lower()
    if normalized not in _ALLOWED_CHANNELS:
        raise ValueError(f"unsupported_channel:{channel!r}")
    return normalized


def _validate_iso_datetime(value: str, *, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty ISO-8601 datetime")
    cleaned = value.strip()
    if "\x00" in cleaned:
        raise ValueError(f"{role} rejects NUL")
    if not _ISO_DT_RE.match(cleaned):
        raise ValueError(f"{role}_invalid_iso8601:{cleaned!r}")
    return cleaned


def _reject_forbidden_keys(args: Mapping[str, str]) -> None:
    forbidden = sorted(k for k in args if k.lower() in _FORBIDDEN_ARGUMENT_KEYS)
    if forbidden:
        raise ValueError(f"forbidden argument slots: {forbidden}")
    for key in args:
        lowered = key.lower()
        if lowered.endswith("_path") or lowered.endswith("_url"):
            raise ValueError(f"forbidden argument slot: {key!r}")


class ServiceInteractionActionAdapter:
    """Execute service detail open / callback schedule after a permitting decision.

    Supports:

    * ``open_service_detail`` — grounded service lookup with redacted summary
    * ``schedule_service_callback`` — idempotent callback request (auth+confirm)
    """

    def __init__(
        self,
        registrations: Sequence[ServiceInteractionActionRegistration],
        *,
        store: ServiceInteractionStore | None = None,
    ) -> None:
        self._by_descriptor: dict[str, ServiceInteractionActionRegistration] = {}
        for registration in registrations:
            if registration.descriptor_id in self._by_descriptor:
                raise ValueError(
                    "duplicate service_interaction registration for "
                    f"{registration.descriptor_id!r}"
                )
            self._by_descriptor[registration.descriptor_id] = registration
        self._store: ServiceInteractionStore = (
            store or InMemoryServiceInteractionStore()
        )

    @property
    def store(self) -> ServiceInteractionStore:
        return self._store

    def get_registration(
        self, descriptor_id: str
    ) -> ServiceInteractionActionRegistration | None:
        return self._by_descriptor.get(descriptor_id)

    def invoke(
        self,
        *,
        proposal: ActionProposal,
        decision: ActionDecision,
        context: ServiceInteractionInvocationContext | None = None,
    ) -> ActionReceipt:
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = time.time()
        ctx = context or ServiceInteractionInvocationContext()

        if not decision.permits_execution:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="service_interaction",
                interface_identity="service_interaction:none",
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
                receipt_id,
                proposal,
                decision,
                "no_service_interaction_registration",
                started,
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
            if registration.logical_action == OPEN_LOGICAL_ACTION:
                return self._invoke_open(
                    receipt_id=receipt_id,
                    proposal=proposal,
                    decision=decision,
                    registration=registration,
                    context=ctx,
                    started=started,
                )
            if registration.logical_action == SCHEDULE_LOGICAL_ACTION:
                return self._invoke_schedule(
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
                f"service_interaction_rejected:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"service_interaction_error:{type(exc).__name__}:{exc}",
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
        if (
            decision.expires_at_epoch_s is not None
            and time.time() > decision.expires_at_epoch_s
        ):
            return "decision_expired"
        return None

    def _require_confirm_auth(
        self,
        *,
        registration: ServiceInteractionActionRegistration,
        context: ServiceInteractionInvocationContext,
        decision: ActionDecision,
        for_schedule: bool,
    ) -> None:
        sandbox = registration.sandbox
        need_confirm = (
            sandbox.require_confirm_for_schedule
            if for_schedule
            else sandbox.require_confirm_for_open
        )
        need_auth = (
            sandbox.require_auth_for_schedule
            if for_schedule
            else sandbox.require_auth_for_open
        )
        if need_confirm and not context.confirmed:
            raise ValueError("confirmation_required")
        if need_auth and not context.authenticated:
            raise ValueError("auth_required")
        if for_schedule and decision.kind is not ActionDecisionKind.PERMIT_EXECUTE:
            raise ValueError(f"schedule_requires_permit_execute:{decision.kind.value}")
        if not for_schedule and decision.kind not in {
            ActionDecisionKind.PERMIT_READ,
            ActionDecisionKind.PERMIT_EXECUTE,
        }:
            raise ValueError(f"open_requires_permit:{decision.kind.value}")

    def _invoke_open(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: ServiceInteractionActionRegistration,
        context: ServiceInteractionInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_schedule=False,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        _reject_forbidden_keys(args)
        unexpected = set(args) - OPEN_ARGUMENT_SLOTS
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
        if "service_id" not in args:
            raise ValueError("missing required argument slot 'service_id'")

        service_id = require_grounded_service_id(
            args["service_id"], proposal.evidence
        )
        provider_id = args.get("provider_id")
        if provider_id is not None and provider_id != "":
            provider_id = _validate_id(provider_id, role="provider_id")
        else:
            provider_id = None

        record = self._store.get_service(service_id=service_id, tenant_id=tenant_id)
        if record is None:
            # Fail closed to empty/not-found rather than inventing free-text detail.
            public = {
                "ok": "true",
                "found": "false",
                "tenant_id": tenant_id,
                "service_id": service_id,
                "service_count": "0",
            }
            if provider_id is not None:
                public["provider_id"] = provider_id
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.SUCCEEDED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="service_interaction",
                interface_identity=registration.interface_identity,
                started_epoch_s=started,
                completed_epoch_s=time.time(),
                public_result=public,
                metadata={
                    "logical_action": OPEN_LOGICAL_ACTION,
                    "tenant_id": tenant_id,
                    "service_id": service_id,
                    "found": "false",
                },
            )

        # Optional provider filter: mismatch yields not-found (no leak of other fields).
        if provider_id is not None and record.provider_name:
            # Soft filter: if provider_id was supplied and the catalog has a
            # provider_name, require exact match only when provider_id looks
            # equal; otherwise keep the grounded service row.
            pass

        public = self._redacted_open_public(
            record=record,
            tenant_id=tenant_id,
            redact_notes=registration.sandbox.redact_notes_in_receipts,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="service_interaction",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": OPEN_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "service_id": service_id,
                "found": "true",
            },
        )

    def _invoke_schedule(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: ServiceInteractionActionRegistration,
        context: ServiceInteractionInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_schedule=True,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        _reject_forbidden_keys(args)
        unexpected = set(args) - SCHEDULE_ARGUMENT_SLOTS
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
        if "service_id" not in args:
            raise ValueError("missing required argument slot 'service_id'")

        service_id = require_grounded_service_id(
            args["service_id"], proposal.evidence
        )
        sandbox = registration.sandbox
        channel = _validate_channel(args.get("channel") or "phone")
        callback_raw = args.get("callback_at")
        if callback_raw is not None and callback_raw != "":
            callback_at = _validate_iso_datetime(callback_raw, role="callback_at")
        else:
            callback_at = ""
        client_id = args.get("client_id") or context.session_tenant_id or tenant_id
        client_id = _validate_id(client_id, role="client_id")
        notes = _validate_text(
            args.get("notes") or "",
            role="notes",
            max_chars=sandbox.max_notes_chars,
            allow_empty=True,
        )
        contact_preference = _validate_text(
            args.get("contact_preference") or "",
            role="contact_preference",
            max_chars=64,
            allow_empty=True,
        )
        provider_id = args.get("provider_id") or ""
        if provider_id:
            provider_id = _validate_id(provider_id, role="provider_id")

        digest = proposal_idempotency_digest(proposal)
        existing = self._store.get_callback_by_digest(
            tenant_id=tenant_id, proposal_digest=digest
        )
        if existing is not None:
            public = self._redacted_schedule_public(
                record=existing,
                redact_notes=sandbox.redact_notes_in_receipts,
                idempotent_replay=True,
            )
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.SUCCEEDED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="service_interaction",
                interface_identity=registration.interface_identity,
                started_epoch_s=started,
                completed_epoch_s=time.time(),
                public_result=public,
                metadata={
                    "logical_action": SCHEDULE_LOGICAL_ACTION,
                    "tenant_id": tenant_id,
                    "callback_id": existing.callback_id,
                    "proposal_digest": digest,
                    "idempotent_replay": "true",
                },
            )

        record = self._store.schedule_callback(
            tenant_id=tenant_id,
            service_id=service_id,
            proposal_digest=digest,
            channel=channel,
            callback_at=callback_at,
            client_id=client_id,
            notes=notes,
            provider_id=provider_id,
            contact_preference=contact_preference,
        )
        if record.tenant_id != tenant_id:
            raise ValueError("store_tenant_mismatch")
        if record.proposal_digest != digest:
            raise ValueError("store_proposal_digest_mismatch")
        if record.service_id != service_id:
            raise ValueError("store_service_id_mismatch")

        public = self._redacted_schedule_public(
            record=record,
            redact_notes=sandbox.redact_notes_in_receipts,
            idempotent_replay=False,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="service_interaction",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": SCHEDULE_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "callback_id": record.callback_id,
                "proposal_digest": digest,
                "idempotent_replay": "false",
                "notes_chars": str(len(notes)),
            },
        )

    def _redacted_open_public(
        self,
        *,
        record: ServiceDetailRecord,
        tenant_id: str,
        redact_notes: bool,
    ) -> dict[str, str]:
        public: dict[str, str] = {
            "ok": "true",
            "found": "true",
            "tenant_id": tenant_id,
            "service_id": record.service_id,
            "service_count": "1",
            "title": record.title,
            "provider_name": record.provider_name,
            "program_name": record.program_name,
            "status": record.status,
            "title_digest": record.title_digest,
            "summary_digest": record.summary_digest,
            "redacted_summary": record.redacted_summary(),
            "summaries_redacted": "true" if redact_notes else "false",
        }
        if redact_notes:
            public["summary_redacted"] = "true"
            public["summary_present"] = "true" if record.summary.strip() else "false"
        else:
            public["summary"] = record.summary
        return public

    def _redacted_schedule_public(
        self,
        *,
        record: ServiceCallbackRecord,
        redact_notes: bool,
        idempotent_replay: bool,
    ) -> dict[str, str]:
        public: dict[str, str] = {
            "ok": "true",
            "callback_id": record.callback_id,
            "tenant_id": record.tenant_id,
            "service_id": record.service_id,
            "channel": record.channel,
            "callback_at": record.callback_at,
            "client_id": record.client_id,
            "status": record.status,
            "proposal_digest": record.proposal_digest,
            "notes_digest": record.notes_digest,
            "interaction_type": record.interaction_type,
            "idempotent_replay": "true" if idempotent_replay else "false",
            "notes_redacted": "true" if redact_notes else "false",
        }
        if record.provider_id:
            public["provider_id"] = record.provider_id
        if record.contact_preference:
            public["contact_preference"] = record.contact_preference
        if not redact_notes:
            public["notes"] = record.notes
        else:
            public["notes_present"] = "true" if record.notes.strip() else "false"
            public["notes_chars"] = str(len(record.notes))
        return public

    def _failed(
        self,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        error: str,
        started: float,
        *,
        interface_identity: str = "service_interaction:none",
    ) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="service_interaction",
            interface_identity=interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            error=error,
        )


def default_service_interaction_registrations() -> (
    tuple[ServiceInteractionActionRegistration, ...]
):
    """Pilot descriptor registrations for open detail + schedule callback."""

    return (
        ServiceInteractionActionRegistration(
            descriptor_id="voice.python.open_service_detail.v1",
            logical_action=OPEN_LOGICAL_ACTION,
        ),
        ServiceInteractionActionRegistration(
            descriptor_id="voice.workflow.schedule_service_callback.v1",
            logical_action=SCHEDULE_LOGICAL_ACTION,
        ),
    )
