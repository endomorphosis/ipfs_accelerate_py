"""Tenant isolation, confirm+auth, redacted summaries, structured slots for calendar."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.action_runtime.adapters.calendar import (
    CREATE_ARGUMENT_SLOTS,
    DEFAULT_MAX_NOTES_CHARS,
    DEFAULT_MAX_TITLE_CHARS,
    READ_ARGUMENT_SLOTS,
    CalendarActionAdapter,
    CalendarActionRegistration,
    CalendarEventRecord,
    CalendarInvocationContext,
    CalendarSandboxPolicy,
    InMemoryCalendarEventStore,
    default_calendar_registrations,
)
from ipfs_accelerate_py.action_runtime.catalog_211ai import (
    build_pilot_catalog,
    logical_action_to_descriptor_id,
)
from ipfs_accelerate_py.action_runtime.contracts import (
    ActionDecision,
    ActionDecisionKind,
    ActionProposal,
    ActionStatus,
    RiskClass,
    content_digest,
)
from ipfs_accelerate_py.action_runtime.policy_pilot import (
    PilotAdmissionContext,
    PilotPolicy,
)


READ_ID = "voice.python.read_calendar.v1"
CREATE_ID = "voice.python.create_calendar_reminder.v1"


def _proposal(
    logical_action: str,
    *,
    arguments: dict[str, str] | None = None,
    tenant_id: str | None = "tenant-a",
    proposal_id: str = "prop-cal-1",
    **kwargs: object,
) -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    base = {
        "proposal_id": proposal_id,
        "descriptor_id": mapping[logical_action],
        "logical_action": logical_action,
        "arguments": arguments or {},
        "route": "calendar_event_support",
        "channel": "voice",
        "tenant_id": tenant_id,
        "confidence": 0.99,
        "source": "test",
    }
    base.update(kwargs)
    return ActionProposal(**base)  # type: ignore[arg-type]


def _permit(
    proposal: ActionProposal,
    *,
    kind: ActionDecisionKind = ActionDecisionKind.PERMIT_EXECUTE,
    risk_class: RiskClass = RiskClass.WRITE,
) -> ActionDecision:
    return ActionDecision(
        decision_id="dec-cal-1",
        kind=kind,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="digest-test",
        arguments_digest=proposal.arguments_digest,
        reason="test_permit",
        risk_class=risk_class,
    )


def _adapter(
    store: InMemoryCalendarEventStore | None = None,
    *,
    sandbox: CalendarSandboxPolicy | None = None,
) -> CalendarActionAdapter:
    policy = sandbox or CalendarSandboxPolicy()
    regs = (
        CalendarActionRegistration(
            descriptor_id=READ_ID,
            logical_action="read_calendar",
            sandbox=policy,
        ),
        CalendarActionRegistration(
            descriptor_id=CREATE_ID,
            logical_action="create_calendar_reminder",
            sandbox=policy,
        ),
    )
    return CalendarActionAdapter(regs, store=store or InMemoryCalendarEventStore())


def _auth_ctx(
    *,
    tenant_id: str = "tenant-a",
    confirmed: bool = True,
    authenticated: bool = True,
) -> CalendarInvocationContext:
    return CalendarInvocationContext(
        confirmed=confirmed,
        authenticated=authenticated,
        session_tenant_id=tenant_id,
    )


def _confirm_ctx(
    *,
    tenant_id: str = "tenant-a",
    confirmed: bool = True,
) -> CalendarInvocationContext:
    """Read path: confirm only (auth optional under pilot policy)."""

    return CalendarInvocationContext(
        confirmed=confirmed,
        authenticated=False,
        session_tenant_id=tenant_id,
    )


def _seed_cross_tenant(store: InMemoryCalendarEventStore) -> None:
    store.seed(
        CalendarEventRecord(
            event_id="evt-a-1",
            tenant_id="tenant-a",
            title="Pickup appointment",
            starts_at="2026-08-05T10:00:00Z",
            ends_at="2026-08-05T10:30:00Z",
            notes="SECRET notes for tenant-a only — SSN 111-22-3333",
            location="123 Main St",
            all_day=False,
            reminder_minutes_before=30,
            status="scheduled",
            created_at_epoch_s=1_700_000_100.0,
        ),
        CalendarEventRecord(
            event_id="evt-b-1",
            tenant_id="tenant-b",
            title="Other tenant event",
            starts_at="2026-08-05T11:00:00Z",
            ends_at="2026-08-05T11:30:00Z",
            notes="LEAK-ME notes for tenant-b",
            location="Hidden Place",
            all_day=False,
            reminder_minutes_before=15,
            status="scheduled",
            created_at_epoch_s=1_700_000_200.0,
        ),
    )


def test_default_registrations_match_pilot_catalog() -> None:
    regs = default_calendar_registrations()
    assert {r.descriptor_id for r in regs} == {READ_ID, CREATE_ID}
    mapping = logical_action_to_descriptor_id()
    assert mapping["read_calendar"] == READ_ID
    assert mapping["create_calendar_reminder"] == CREATE_ID
    catalog = build_pilot_catalog()
    read_desc = catalog.require(READ_ID)
    create_desc = catalog.require(CREATE_ID)
    assert read_desc.risk_class is RiskClass.READ
    assert create_desc.risk_class is RiskClass.WRITE
    assert create_desc.metadata.get("auth_required") == "true"
    assert read_desc.metadata.get("auth_required") == "false"


def test_structured_slots_are_closed_sets() -> None:
    assert "title" in CREATE_ARGUMENT_SLOTS
    assert "starts_at" in CREATE_ARGUMENT_SLOTS
    assert "ics" not in CREATE_ARGUMENT_SLOTS
    assert "raw_ics" not in CREATE_ARGUMENT_SLOTS
    assert "body" not in CREATE_ARGUMENT_SLOTS
    assert "limit" in READ_ARGUMENT_SLOTS
    assert "event_id" in READ_ARGUMENT_SLOTS


def test_create_denied_without_permitting_decision() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Call clinic",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    decision = ActionDecision(
        decision_id="dec-deny",
        kind=ActionDecisionKind.CONFIRM,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="d",
        arguments_digest=proposal.arguments_digest,
        reason="confirmation_required",
        risk_class=RiskClass.WRITE,
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.DENIED
    assert "does_not_permit" in (receipt.error or "")
    assert isinstance(adapter.store, InMemoryCalendarEventStore)
    assert adapter.store.list_events(tenant_id="tenant-a") == ()


def test_create_requires_confirm_at_adapter_boundary() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Call clinic",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=False, authenticated=True),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "confirmation_required" in (receipt.error or "")


def test_create_requires_auth_at_adapter_boundary() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Call clinic",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(confirmed=True, authenticated=False),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "auth_required" in (receipt.error or "")


def test_create_succeeds_with_confirm_and_auth_and_redacts_notes() -> None:
    store = InMemoryCalendarEventStore()
    adapter = _adapter(store)
    notes = "Bring insurance card and photo ID for intake."
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Housing intake appointment",
            "starts_at": "2026-08-07T14:00:00Z",
            "duration_minutes": "60",
            "notes": notes,
            "location": "Community Center Room 2",
            "reminder_minutes_before": "30",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.adapter == "calendar"
    assert receipt.public_result["ok"] == "true"
    assert receipt.public_result["notes_redacted"] == "true"
    assert "notes" not in receipt.public_result
    assert notes not in str(receipt.to_dict())
    assert receipt.public_result["notes_digest"] == content_digest(notes)
    assert receipt.public_result["title"] == "Housing intake appointment"
    assert receipt.public_result["starts_at"] == "2026-08-07T14:00:00Z"
    assert receipt.public_result["tenant_id"] == "tenant-a"
    assert receipt.public_result["event_id"].startswith("evt-")
    assert "redacted_summary" in receipt.public_result

    stored = store.list_events(tenant_id="tenant-a")
    assert len(stored) == 1
    assert stored[0].notes == notes
    assert stored[0].tenant_id == "tenant-a"
    assert stored[0].reminder_minutes_before == 30


def test_create_title_and_notes_length_bounded() -> None:
    adapter = _adapter(
        sandbox=CalendarSandboxPolicy(max_title_chars=16, max_notes_chars=32)
    )
    too_long_title = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "x" * 17,
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    receipt = adapter.invoke(
        proposal=too_long_title,
        decision=_permit(too_long_title),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "title_exceeds_max_chars" in (receipt.error or "")

    too_long_notes = _proposal(
        "create_calendar_reminder",
        proposal_id="prop-cal-notes",
        arguments={
            "title": "ok title",
            "starts_at": "2026-08-06T09:00:00Z",
            "notes": "y" * 33,
        },
    )
    receipt2 = adapter.invoke(
        proposal=too_long_notes,
        decision=_permit(too_long_notes),
        context=_auth_ctx(),
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "notes_exceeds_max_chars" in (receipt2.error or "")


def test_create_rejects_empty_title_and_bad_datetime() -> None:
    adapter = _adapter()
    empty = _proposal(
        "create_calendar_reminder",
        arguments={"title": "   ", "starts_at": "2026-08-06T09:00:00Z"},
    )
    receipt = adapter.invoke(
        proposal=empty, decision=_permit(empty), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "non-empty" in (receipt.error or "")

    bad_dt = _proposal(
        "create_calendar_reminder",
        proposal_id="prop-bad-dt",
        arguments={"title": "ok", "starts_at": "next Tuesday afternoon"},
    )
    receipt2 = adapter.invoke(
        proposal=bad_dt, decision=_permit(bad_dt), context=_auth_ctx()
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "invalid_iso8601" in (receipt2.error or "")


def test_create_rejects_raw_ics_injection() -> None:
    adapter = _adapter()
    ics_blob = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n"
        "SUMMARY:Injected\r\nEND:VEVENT\r\nEND:VCALENDAR"
    )
    cases = (
        {"title": ics_blob, "starts_at": "2026-08-06T09:00:00Z"},
        {
            "title": "normal",
            "starts_at": "2026-08-06T09:00:00Z",
            "notes": ics_blob,
        },
        {
            "title": "normal",
            "starts_at": "2026-08-06T09:00:00Z",
            "location": ics_blob,
        },
    )
    for index, arguments in enumerate(cases):
        proposal = _proposal(
            "create_calendar_reminder",
            proposal_id=f"prop-ics-{index}",
            arguments=arguments,
        )
        receipt = adapter.invoke(
            proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
        )
        assert receipt.status is ActionStatus.FAILED, arguments
        assert "rejects_raw_ics" in (receipt.error or ""), receipt.error


def test_create_rejects_forbidden_and_unexpected_slots() -> None:
    adapter = _adapter()
    forbidden = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "ok",
            "starts_at": "2026-08-06T09:00:00Z",
            "ics": "BEGIN:VCALENDAR",
        },
    )
    receipt = adapter.invoke(
        proposal=forbidden, decision=_permit(forbidden), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert "forbidden" in (receipt.error or "")

    unexpected = _proposal(
        "create_calendar_reminder",
        proposal_id="prop-unexpected",
        arguments={
            "title": "ok",
            "starts_at": "2026-08-06T09:00:00Z",
            "free_text_blob": "do not accept",
        },
    )
    receipt2 = adapter.invoke(
        proposal=unexpected, decision=_permit(unexpected), context=_auth_ctx()
    )
    assert receipt2.status is ActionStatus.FAILED
    assert "unexpected arguments" in (receipt2.error or "")


def test_default_bounds_are_sane() -> None:
    assert 1 <= DEFAULT_MAX_TITLE_CHARS <= 1_024
    assert 1 <= DEFAULT_MAX_NOTES_CHARS <= 16_384
    policy = CalendarSandboxPolicy()
    assert policy.max_title_chars == DEFAULT_MAX_TITLE_CHARS
    assert policy.redact_notes_in_receipts is True
    assert policy.require_auth_for_create is True
    assert policy.require_confirm_for_create is True
    assert policy.require_auth_for_read is False
    assert policy.require_confirm_for_read is True


def test_read_returns_redacted_summaries_tenant_scoped() -> None:
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)

    proposal = _proposal("read_calendar", arguments={})
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(tenant_id="tenant-a"),
    )
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    assert receipt.public_result["event_count"] == "1"
    assert "evt-a-1" in receipt.public_result["event_ids"]
    assert "evt-b-1" not in receipt.public_result["event_ids"]
    assert "LEAK-ME" not in str(receipt.to_dict())
    assert "SECRET notes" not in str(receipt.to_dict())
    assert "SSN" not in str(receipt.to_dict())
    assert receipt.public_result["summaries_redacted"] == "true"
    assert receipt.public_result["notes_redacted"] == "true"
    assert "notes" not in receipt.public_result
    # Redacted summary includes time + title preview, never secret notes.
    assert "2026-08-05T10:00:00Z" in receipt.public_result["redacted_summaries"]
    assert "Pickup appointment" in receipt.public_result["redacted_summaries"]


def test_read_cannot_select_other_tenant_via_session_mismatch() -> None:
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)
    proposal = _proposal("read_calendar", tenant_id="tenant-a")
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(tenant_id="tenant-b"),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "tenant_session_mismatch" in (receipt.error or "")


def test_read_requires_confirm_but_not_auth() -> None:
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)
    proposal = _proposal("read_calendar")
    decision = _permit(
        proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
    )

    unconfirmed = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_confirm_ctx(confirmed=False),
    )
    assert unconfirmed.status is ActionStatus.FAILED
    assert "confirmation_required" in (unconfirmed.error or "")

    # Auth not required for read under default sandbox / pilot catalog.
    confirmed_no_auth = adapter.invoke(
        proposal=proposal,
        decision=decision,
        context=_confirm_ctx(confirmed=True),
    )
    assert confirmed_no_auth.status is ActionStatus.SUCCEEDED


def test_read_filters_by_event_id_without_crossing_tenants() -> None:
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    store.seed(
        CalendarEventRecord(
            event_id="evt-a-2",
            tenant_id="tenant-a",
            title="Second event",
            starts_at="2026-08-08T12:00:00Z",
            ends_at="2026-08-08T12:30:00Z",
            notes="another secret",
            location="",
            all_day=False,
            reminder_minutes_before=0,
            status="scheduled",
            created_at_epoch_s=1_700_000_300.0,
        )
    )
    adapter = _adapter(store)
    proposal = _proposal(
        "read_calendar",
        arguments={"event_id": "evt-a-1"},
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["event_count"] == "1"
    assert receipt.public_result["event_ids"] == "evt-a-1"

    # Asking for another tenant's event id returns empty for this tenant.
    cross = _proposal(
        "read_calendar",
        proposal_id="prop-cross-evt",
        arguments={"event_id": "evt-b-1"},
    )
    receipt2 = adapter.invoke(
        proposal=cross,
        decision=_permit(
            cross, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_confirm_ctx(tenant_id="tenant-a"),
    )
    assert receipt2.status is ActionStatus.SUCCEEDED
    assert receipt2.public_result["event_count"] == "0"
    assert "LEAK-ME" not in str(receipt2.to_dict())


def test_create_is_tenant_isolated() -> None:
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    adapter = _adapter(store)
    proposal = _proposal(
        "create_calendar_reminder",
        tenant_id="tenant-a",
        arguments={
            "title": "Tenant A only",
            "starts_at": "2026-08-09T08:00:00Z",
            "notes": "private-a",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(proposal),
        context=_auth_ctx(tenant_id="tenant-a"),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    a_events = store.list_events(tenant_id="tenant-a")
    b_events = store.list_events(tenant_id="tenant-b")
    assert any(e.title == "Tenant A only" for e in a_events)
    assert not any(e.title == "Tenant A only" for e in b_events)
    assert all(e.tenant_id == "tenant-b" for e in b_events)


def test_arguments_digest_mismatch_fails_closed() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "ok",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    decision = _permit(proposal)
    decision = ActionDecision(
        decision_id=decision.decision_id,
        kind=decision.kind,
        proposal_id=decision.proposal_id,
        descriptor_id=decision.descriptor_id,
        descriptor_digest=decision.descriptor_digest,
        arguments_digest="0" * 64,
        reason=decision.reason,
        risk_class=decision.risk_class,
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=decision, context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert receipt.error == "arguments_digest_mismatch"


def test_no_registration_fails_closed() -> None:
    adapter = CalendarActionAdapter([])
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "ok",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.FAILED
    assert receipt.error == "no_calendar_registration"


def test_pilot_policy_create_requires_confirm_and_auth() -> None:
    """End-to-end with pilot policy: create needs confirm+auth before permit."""

    catalog = build_pilot_catalog()
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryCalendarEventStore()
    adapter = CalendarActionAdapter(default_calendar_registrations(), store=store)

    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Callback reminder",
            "starts_at": "2026-08-10T15:00:00Z",
            "notes": "Ask about voucher status",
        },
    )

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM
    assert not unconfirmed.permits_execution
    denied = adapter.invoke(proposal=proposal, decision=unconfirmed, context=_auth_ctx())
    assert denied.status is ActionStatus.DENIED

    confirmed_no_auth = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert confirmed_no_auth.kind is ActionDecisionKind.DENY
    assert confirmed_no_auth.reason == "auth_required"

    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(
            confirmed=True,
            authenticated=True,
            session_tenant_id="tenant-a",
        ),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_EXECUTE
    assert admitted.permits_execution
    receipt = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert "notes" not in receipt.public_result
    assert len(store.list_events(tenant_id="tenant-a")) == 1


def test_pilot_policy_read_requires_confirm_only() -> None:
    catalog = build_pilot_catalog()
    policy = PilotPolicy(catalog=catalog)
    store = InMemoryCalendarEventStore()
    _seed_cross_tenant(store)
    adapter = CalendarActionAdapter(default_calendar_registrations(), store=store)
    proposal = _proposal("read_calendar")

    unconfirmed = policy.decide(proposal, PilotAdmissionContext())
    assert unconfirmed.kind is ActionDecisionKind.CONFIRM

    # Confirmed without auth is still permit_read for calendar (auth_required=false).
    admitted = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True, authenticated=False),
    )
    assert admitted.kind is ActionDecisionKind.PERMIT_READ
    receipt = adapter.invoke(
        proposal=proposal,
        decision=admitted,
        context=_confirm_ctx(),
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.public_result["event_count"] == "1"
    assert "LEAK-ME" not in str(receipt.to_dict())
    assert receipt.public_result["summaries_redacted"] == "true"


def test_sandbox_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="max_title_chars"):
        CalendarSandboxPolicy(max_title_chars=0)
    with pytest.raises(ValueError, match="max_events_returned"):
        CalendarSandboxPolicy(max_events_returned=0)
    with pytest.raises(ValueError, match="max_notes_chars"):
        CalendarSandboxPolicy(max_notes_chars=0)


def test_duplicate_registration_rejected() -> None:
    reg = CalendarActionRegistration(
        descriptor_id=READ_ID,
        logical_action="read_calendar",
    )
    with pytest.raises(ValueError, match="duplicate"):
        CalendarActionAdapter([reg, reg])


def test_receipt_public_result_values_are_strings() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "String check",
            "starts_at": "2026-08-06T09:00:00Z",
            "notes": "private",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal, decision=_permit(proposal), context=_auth_ctx()
    )
    assert receipt.status is ActionStatus.SUCCEEDED
    for key, value in receipt.public_result.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_create_requires_permit_execute_not_permit_read() -> None:
    adapter = _adapter()
    proposal = _proposal(
        "create_calendar_reminder",
        arguments={
            "title": "Should fail",
            "starts_at": "2026-08-06T09:00:00Z",
        },
    )
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_permit(
            proposal, kind=ActionDecisionKind.PERMIT_READ, risk_class=RiskClass.READ
        ),
        context=_auth_ctx(),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "create_requires_permit_execute" in (receipt.error or "")


def test_missing_required_slots_fail_closed() -> None:
    adapter = _adapter()
    no_title = _proposal(
        "create_calendar_reminder",
        arguments={"starts_at": "2026-08-06T09:00:00Z"},
    )
    r1 = adapter.invoke(
        proposal=no_title, decision=_permit(no_title), context=_auth_ctx()
    )
    assert r1.status is ActionStatus.FAILED
    assert "title" in (r1.error or "")

    no_starts = _proposal(
        "create_calendar_reminder",
        proposal_id="prop-no-starts",
        arguments={"title": "Missing start"},
    )
    r2 = adapter.invoke(
        proposal=no_starts, decision=_permit(no_starts), context=_auth_ctx()
    )
    assert r2.status is ActionStatus.FAILED
    assert "starts_at" in (r2.error or "")
