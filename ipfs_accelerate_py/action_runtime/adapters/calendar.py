"""Calendar action adapter (read_calendar + create_calendar_reminder).

Safety rules:

* execute only after a permitting ``ActionDecision`` binds the exact proposal
* ``create_calendar_reminder`` re-checks confirm + authenticated tenant at the
  adapter boundary (defense in depth with pilot policy)
* ``read_calendar`` re-checks confirm; auth is optional at the adapter boundary
  (pilot catalog marks read as confirm-only)
* event notes/descriptions are size-bounded and redacted from public receipts
* reads are strictly tenant-scoped; cross-tenant rows are never returned
* no network or ICS side effects — the store is injected
* no raw ICS injection from free text; structured slots only
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

READ_LOGICAL_ACTION = "read_calendar"
CREATE_LOGICAL_ACTION = "create_calendar_reminder"

_ALLOWED_LOGICAL_ACTIONS = frozenset({READ_LOGICAL_ACTION, CREATE_LOGICAL_ACTION})
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")
# ISO-8601 datetime (with optional fractional seconds and Z/offset) or date-only.
_ISO_DT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
# Reject free-text ICS smuggling via any argument value.
_ICS_MARKERS = (
    "BEGIN:VCALENDAR",
    "BEGIN:VEVENT",
    "BEGIN:VALARM",
    "END:VCALENDAR",
    "END:VEVENT",
)

DEFAULT_MAX_TITLE_CHARS = 200
DEFAULT_MAX_NOTES_CHARS = 2_000
DEFAULT_MAX_LOCATION_CHARS = 200
DEFAULT_MAX_EVENTS_RETURNED = 50
DEFAULT_MAX_REMINDER_MINUTES = 60 * 24 * 30  # 30 days

# Argument slots admitted for each logical action (fail closed on extras).
READ_ARGUMENT_SLOTS = frozenset({"limit", "starts_after", "ends_before", "event_id"})
CREATE_ARGUMENT_SLOTS = frozenset(
    {
        "title",
        "starts_at",
        "ends_at",
        "duration_minutes",
        "notes",
        "location",
        "all_day",
        "reminder_minutes_before",
    }
)
# Keys that must never appear (raw ICS / free-text injection vectors).
_FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "ics",
        "raw_ics",
        "ical",
        "vevent",
        "calendar_blob",
        "payload",
        "body",
        "command",
        "argv",
        "executable",
        "url",
        "import_path",
    }
)


@dataclass(frozen=True)
class CalendarSandboxPolicy:
    """Resource and privacy bounds for calendar adapter invocations."""

    max_title_chars: int = DEFAULT_MAX_TITLE_CHARS
    max_notes_chars: int = DEFAULT_MAX_NOTES_CHARS
    max_location_chars: int = DEFAULT_MAX_LOCATION_CHARS
    max_events_returned: int = DEFAULT_MAX_EVENTS_RETURNED
    max_reminder_minutes_before: int = DEFAULT_MAX_REMINDER_MINUTES
    # When True (default), receipts never carry raw notes/descriptions.
    redact_notes_in_receipts: bool = True
    # Adapter-boundary re-checks (pilot policy also gates these).
    require_confirm_for_create: bool = True
    require_auth_for_create: bool = True
    require_confirm_for_read: bool = True
    # Pilot catalog: read_calendar is confirm-only (auth_required=false).
    require_auth_for_read: bool = False

    def __post_init__(self) -> None:
        if self.max_title_chars < 1 or self.max_title_chars > 1_024:
            raise ValueError("max_title_chars must be in [1, 1024]")
        if self.max_notes_chars < 1 or self.max_notes_chars > 16_384:
            raise ValueError("max_notes_chars must be in [1, 16384]")
        if self.max_location_chars < 1 or self.max_location_chars > 1_024:
            raise ValueError("max_location_chars must be in [1, 1024]")
        if self.max_events_returned < 1 or self.max_events_returned > 500:
            raise ValueError("max_events_returned must be in [1, 500]")
        if (
            self.max_reminder_minutes_before < 0
            or self.max_reminder_minutes_before > 60 * 24 * 365
        ):
            raise ValueError("max_reminder_minutes_before must be in [0, 525600]")


@dataclass(frozen=True)
class CalendarEventRecord:
    """Tenant-scoped calendar event/reminder stored by an injected backend."""

    event_id: str
    tenant_id: str
    title: str
    starts_at: str
    ends_at: str
    notes: str
    location: str
    all_day: bool
    reminder_minutes_before: int
    status: str
    created_at_epoch_s: float
    notes_digest: str = ""
    title_digest: str = ""

    def __post_init__(self) -> None:
        if not self.notes_digest:
            object.__setattr__(self, "notes_digest", content_digest(self.notes))
        if not self.title_digest:
            object.__setattr__(self, "title_digest", content_digest(self.title))

    def redacted_summary(self) -> str:
        """Privacy-safe one-line summary for voice-facing receipts.

        Includes structured time and a truncated title; never includes notes.
        """

        title_preview = self.title.strip()
        if len(title_preview) > 40:
            title_preview = title_preview[:37] + "..."
        # Strip characters that could smuggle multi-line free text into a receipt.
        title_preview = re.sub(r"[\r\n\t]+", " ", title_preview)
        all_day_tag = " all_day" if self.all_day else ""
        return f"{self.starts_at} | {title_preview}{all_day_tag}".strip()


class CalendarEventStore(Protocol):
    """Backend protocol for calendar event storage (fake or product)."""

    def list_events(
        self,
        *,
        tenant_id: str,
        starts_after: str | None = None,
        ends_before: str | None = None,
        event_id: str | None = None,
        limit: int = DEFAULT_MAX_EVENTS_RETURNED,
    ) -> Sequence[CalendarEventRecord]:
        """Return events belonging only to ``tenant_id``."""
        ...

    def create_reminder(
        self,
        *,
        tenant_id: str,
        title: str,
        starts_at: str,
        ends_at: str,
        notes: str,
        location: str,
        all_day: bool,
        reminder_minutes_before: int,
    ) -> CalendarEventRecord:
        """Persist a tenant-scoped reminder/event for ``tenant_id``."""
        ...


@dataclass
class InMemoryCalendarEventStore:
    """Offline fake store for unit tests and wallet binding fakes."""

    _events: list[CalendarEventRecord] = field(default_factory=list)
    _clock: Callable[[], float] = time.time

    def seed(self, *records: CalendarEventRecord) -> None:
        for record in records:
            self._events.append(record)

    def list_events(
        self,
        *,
        tenant_id: str,
        starts_after: str | None = None,
        ends_before: str | None = None,
        event_id: str | None = None,
        limit: int = DEFAULT_MAX_EVENTS_RETURNED,
    ) -> Sequence[CalendarEventRecord]:
        if not tenant_id:
            return ()
        rows = [
            e
            for e in self._events
            if e.tenant_id == tenant_id
            and (event_id is None or e.event_id == event_id)
            and (starts_after is None or e.starts_at >= starts_after)
            and (ends_before is None or (e.ends_at or e.starts_at) <= ends_before)
        ]
        # Soonest first for voice schedule summaries.
        rows.sort(key=lambda e: e.starts_at)
        return tuple(rows[: max(0, limit)])

    def create_reminder(
        self,
        *,
        tenant_id: str,
        title: str,
        starts_at: str,
        ends_at: str,
        notes: str,
        location: str,
        all_day: bool,
        reminder_minutes_before: int,
    ) -> CalendarEventRecord:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        record = CalendarEventRecord(
            event_id=f"evt-{uuid.uuid4().hex[:16]}",
            tenant_id=tenant_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            notes=notes,
            location=location,
            all_day=all_day,
            reminder_minutes_before=reminder_minutes_before,
            status="scheduled",
            created_at_epoch_s=float(self._clock()),
        )
        self._events.append(record)
        return record


@dataclass(frozen=True)
class CalendarActionRegistration:
    """Reviewed calendar binding for a catalog descriptor."""

    descriptor_id: str
    logical_action: str
    sandbox: CalendarSandboxPolicy = field(default_factory=CalendarSandboxPolicy)
    interface_name: str = "calendar"

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id is required")
        if self.logical_action not in _ALLOWED_LOGICAL_ACTIONS:
            raise ValueError(
                f"logical_action must be one of {sorted(_ALLOWED_LOGICAL_ACTIONS)}"
            )

    @property
    def interface_identity(self) -> str:
        return f"calendar:{self.logical_action}:{self.descriptor_id}"


@dataclass(frozen=True)
class CalendarInvocationContext:
    """Authority-plane facts re-checked at the adapter boundary.

    Policy already gates confirm/auth; the adapter re-validates so a forged
    permit decision alone cannot create reminders or dump other tenants.
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
    context: CalendarInvocationContext,
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


def _parse_non_negative_int(raw: str, *, role: str, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{role} must be >= 0")
    if value > maximum:
        raise ValueError(f"{role}_exceeds_max:{maximum}")
    return value


def _parse_bool(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"boolean expected, got {raw!r}")


def _validate_iso_datetime(value: str, *, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty ISO-8601 datetime")
    cleaned = value.strip()
    if "\x00" in cleaned:
        raise ValueError(f"{role} rejects NUL")
    if not _ISO_DT_RE.match(cleaned):
        raise ValueError(f"{role}_invalid_iso8601:{cleaned!r}")
    _reject_ics_text(cleaned, role=role)
    return cleaned


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
    _reject_ics_text(value, role=role)
    return value


def _reject_ics_text(value: str, *, role: str) -> None:
    upper = value.upper()
    for marker in _ICS_MARKERS:
        if marker in upper:
            raise ValueError(f"{role}_rejects_raw_ics")


def _reject_forbidden_keys(args: dict[str, str]) -> None:
    forbidden = sorted(k for k in args if k.lower() in _FORBIDDEN_ARGUMENT_KEYS)
    if forbidden:
        raise ValueError(f"forbidden argument slots: {forbidden}")
    # Also reject any key that ends with _path or looks like an ICS blob carrier.
    for key in args:
        lowered = key.lower()
        if lowered.endswith("_path") or lowered.endswith("_ics") or "vevent" in lowered:
            raise ValueError(f"forbidden argument slot: {key!r}")


def _ends_at_from_duration(starts_at: str, duration_minutes: int) -> str:
    """Derive ends_at only when starts_at is a pure ISO date/time we can extend.

    For offline fakes we accept the structured duration and store a sentinel
    ends_at of the form ``starts_at+Nmin`` when we cannot parse wall-clock math
    without a full datetime library dependency. Product backends may replace
    this store and compute real end times.
    """

    if duration_minutes < 0:
        raise ValueError("duration_minutes must be >= 0")
    return f"{starts_at}+{duration_minutes}min"


class CalendarActionAdapter:
    """Execute calendar read/create only after a permitting decision.

    Supports:

    * ``read_calendar`` — tenant-scoped event listing with redacted summaries
    * ``create_calendar_reminder`` — structured-slot reminder create (auth+confirm)
    """

    def __init__(
        self,
        registrations: Sequence[CalendarActionRegistration],
        *,
        store: CalendarEventStore | None = None,
    ) -> None:
        self._by_descriptor: dict[str, CalendarActionRegistration] = {}
        for registration in registrations:
            if registration.descriptor_id in self._by_descriptor:
                raise ValueError(
                    f"duplicate calendar registration for {registration.descriptor_id!r}"
                )
            self._by_descriptor[registration.descriptor_id] = registration
        self._store: CalendarEventStore = store or InMemoryCalendarEventStore()

    @property
    def store(self) -> CalendarEventStore:
        return self._store

    def get_registration(self, descriptor_id: str) -> CalendarActionRegistration | None:
        return self._by_descriptor.get(descriptor_id)

    def invoke(
        self,
        *,
        proposal: ActionProposal,
        decision: ActionDecision,
        context: CalendarInvocationContext | None = None,
    ) -> ActionReceipt:
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = time.time()
        ctx = context or CalendarInvocationContext()

        if not decision.permits_execution:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="calendar",
                interface_identity="calendar:none",
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
                receipt_id, proposal, decision, "no_calendar_registration", started
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
            if registration.logical_action == CREATE_LOGICAL_ACTION:
                return self._invoke_create(
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
                f"calendar_rejected:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"calendar_error:{type(exc).__name__}:{exc}",
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
        registration: CalendarActionRegistration,
        context: CalendarInvocationContext,
        decision: ActionDecision,
        for_create: bool,
    ) -> None:
        sandbox = registration.sandbox
        need_confirm = (
            sandbox.require_confirm_for_create
            if for_create
            else sandbox.require_confirm_for_read
        )
        need_auth = (
            sandbox.require_auth_for_create
            if for_create
            else sandbox.require_auth_for_read
        )
        if need_confirm and not context.confirmed:
            raise ValueError("confirmation_required")
        if need_auth and not context.authenticated:
            raise ValueError("auth_required")
        # Create is write-class: only PERMIT_EXECUTE is acceptable.
        if for_create and decision.kind is not ActionDecisionKind.PERMIT_EXECUTE:
            raise ValueError(f"create_requires_permit_execute:{decision.kind.value}")
        # Read is read-class: PERMIT_READ (or PERMIT_EXECUTE if elevated).
        if not for_create and decision.kind not in {
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
        registration: CalendarActionRegistration,
        context: CalendarInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_create=False,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        _reject_forbidden_keys(args)
        unexpected = set(args) - READ_ARGUMENT_SLOTS
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")

        event_id = args.get("event_id")
        if event_id is not None:
            event_id = _validate_id(event_id, role="event_id")

        starts_after = args.get("starts_after")
        if starts_after is not None and starts_after != "":
            starts_after = _validate_iso_datetime(starts_after, role="starts_after")
        else:
            starts_after = None

        ends_before = args.get("ends_before")
        if ends_before is not None and ends_before != "":
            ends_before = _validate_iso_datetime(ends_before, role="ends_before")
        else:
            ends_before = None

        limit = _parse_limit(
            args.get("limit"),
            default=registration.sandbox.max_events_returned,
            maximum=registration.sandbox.max_events_returned,
        )

        rows = list(
            self._store.list_events(
                tenant_id=tenant_id,
                starts_after=starts_after,
                ends_before=ends_before,
                event_id=event_id,
                limit=limit,
            )
        )
        # Defense in depth: drop any store bug that leaks other tenants.
        rows = [e for e in rows if e.tenant_id == tenant_id]

        public = self._redacted_read_public(
            rows=rows,
            tenant_id=tenant_id,
            redact_notes=registration.sandbox.redact_notes_in_receipts,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="calendar",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": READ_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "event_count": str(len(rows)),
            },
        )

    def _invoke_create(
        self,
        *,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        registration: CalendarActionRegistration,
        context: CalendarInvocationContext,
        started: float,
    ) -> ActionReceipt:
        self._require_confirm_auth(
            registration=registration,
            context=context,
            decision=decision,
            for_create=True,
        )
        tenant_id = _resolve_tenant(proposal, context)
        args = dict(proposal.arguments)
        _reject_forbidden_keys(args)
        unexpected = set(args) - CREATE_ARGUMENT_SLOTS
        if unexpected:
            raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
        if "title" not in args:
            raise ValueError("missing required argument slot 'title'")
        if "starts_at" not in args:
            raise ValueError("missing required argument slot 'starts_at'")

        sandbox = registration.sandbox
        title = _validate_text(
            args["title"],
            role="title",
            max_chars=sandbox.max_title_chars,
            allow_empty=False,
        )
        starts_at = _validate_iso_datetime(args["starts_at"], role="starts_at")

        notes = _validate_text(
            args.get("notes") or "",
            role="notes",
            max_chars=sandbox.max_notes_chars,
            allow_empty=True,
        )
        location = _validate_text(
            args.get("location") or "",
            role="location",
            max_chars=sandbox.max_location_chars,
            allow_empty=True,
        )
        all_day = _parse_bool(args.get("all_day"), default=False)

        reminder_raw = args.get("reminder_minutes_before")
        if reminder_raw is None or reminder_raw == "":
            reminder_minutes = 0
        else:
            reminder_minutes = _parse_non_negative_int(
                reminder_raw,
                role="reminder_minutes_before",
                maximum=sandbox.max_reminder_minutes_before,
            )

        ends_at_raw = args.get("ends_at")
        duration_raw = args.get("duration_minutes")
        if ends_at_raw is not None and ends_at_raw != "":
            ends_at = _validate_iso_datetime(ends_at_raw, role="ends_at")
        elif duration_raw is not None and duration_raw != "":
            duration = _parse_non_negative_int(
                duration_raw,
                role="duration_minutes",
                maximum=60 * 24 * 14,  # two weeks max for a single event
            )
            ends_at = _ends_at_from_duration(starts_at, duration)
        else:
            # Default: zero-duration reminder (point-in-time).
            ends_at = starts_at

        record = self._store.create_reminder(
            tenant_id=tenant_id,
            title=title,
            starts_at=starts_at,
            ends_at=ends_at,
            notes=notes,
            location=location,
            all_day=all_day,
            reminder_minutes_before=reminder_minutes,
        )
        if record.tenant_id != tenant_id:
            raise ValueError("store_tenant_mismatch")

        public = self._redacted_create_public(
            record=record,
            redact_notes=sandbox.redact_notes_in_receipts,
        )
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.SUCCEEDED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="calendar",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            public_result=public,
            metadata={
                "logical_action": CREATE_LOGICAL_ACTION,
                "tenant_id": tenant_id,
                "event_id": record.event_id,
                "notes_chars": str(len(notes)),
            },
        )

    def _redacted_read_public(
        self,
        *,
        rows: Sequence[CalendarEventRecord],
        tenant_id: str,
        redact_notes: bool,
    ) -> dict[str, str]:
        event_ids = ",".join(e.event_id for e in rows)
        title_digests = ",".join(e.title_digest for e in rows)
        notes_digests = ",".join(e.notes_digest for e in rows)
        starts = ",".join(e.starts_at for e in rows)
        # Redacted summaries: structured time + truncated title, never notes.
        summaries = " || ".join(e.redacted_summary() for e in rows)
        public: dict[str, str] = {
            "ok": "true",
            "tenant_id": tenant_id,
            "event_count": str(len(rows)),
            "event_ids": event_ids,
            "title_digests": title_digests,
            "notes_digests": notes_digests,
            "starts_at": starts,
            "summaries_redacted": "true" if redact_notes else "false",
            "redacted_summaries": summaries,
        }
        if redact_notes:
            public["notes_redacted"] = "true"
            public["locations_redacted"] = "true"
        else:
            # Operator-only opt-out: still never dump raw notes as one blob.
            public["notes_present_count"] = str(
                sum(1 for e in rows if e.notes.strip())
            )
        return public

    def _redacted_create_public(
        self,
        *,
        record: CalendarEventRecord,
        redact_notes: bool,
    ) -> dict[str, str]:
        public: dict[str, str] = {
            "ok": "true",
            "event_id": record.event_id,
            "tenant_id": record.tenant_id,
            "title": record.title,
            "starts_at": record.starts_at,
            "ends_at": record.ends_at,
            "all_day": "true" if record.all_day else "false",
            "reminder_minutes_before": str(record.reminder_minutes_before),
            "status": record.status,
            "title_digest": record.title_digest,
            "notes_digest": record.notes_digest,
            "notes_redacted": "true" if redact_notes else "false",
            "redacted_summary": record.redacted_summary(),
        }
        if not redact_notes:
            public["notes"] = record.notes
            public["location"] = record.location
        else:
            public["notes_present"] = "true" if record.notes.strip() else "false"
            public["location_present"] = "true" if record.location.strip() else "false"
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
        interface_identity: str = "calendar:none",
    ) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="calendar",
            interface_identity=interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            error=error,
        )


def default_calendar_registrations() -> tuple[CalendarActionRegistration, ...]:
    """Pilot descriptor registrations for read + create calendar reminder."""

    return (
        CalendarActionRegistration(
            descriptor_id="voice.python.read_calendar.v1",
            logical_action=READ_LOGICAL_ACTION,
        ),
        CalendarActionRegistration(
            descriptor_id="voice.python.create_calendar_reminder.v1",
            logical_action=CREATE_LOGICAL_ACTION,
        ),
    )
