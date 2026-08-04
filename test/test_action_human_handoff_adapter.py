"""Durable HandoffRequest + transfer receipt status + spoken-success gate tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.action_runtime.adapters.human_handoff import (
    FileHandoffRequestStore,
    HandoffInvocationContext,
    HandoffRequest,
    HandoffSandboxPolicy,
    HumanHandoffActionAdapter,
    HumanHandoffActionRegistration,
    InMemoryHandoffRequestStore,
    allows_spoken_success,
    default_handoff_registrations,
    spoken_outcome_role,
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
)
from ipfs_accelerate_py.action_runtime.policy_pilot import (
    PilotAdmissionContext,
    PilotPolicy,
)


HANDOFF_ID = "voice.human.handoff_live_agent.v1"
LOGICAL = "handoff_live_agent"


def _proposal(
    *,
    arguments: dict[str, str] | None = None,
    tenant_id: str | None = "tenant-a",
    proposal_id: str = "prop-hoff-1",
    channel: str | None = "voice",
    session_id: str | None = "sess-1",
    **kwargs: object,
) -> ActionProposal:
    mapping = logical_action_to_descriptor_id()
    base = {
        "proposal_id": proposal_id,
        "descriptor_id": mapping[LOGICAL],
        "logical_action": LOGICAL,
        "arguments": arguments or {},
        "route": "live_agent",
        "channel": channel,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "confidence": 0.99,
        "source": "test",
    }
    base.update(kwargs)
    return ActionProposal(**base)  # type: ignore[arg-type]


def _handoff_decision(
    proposal: ActionProposal,
    *,
    reason: str = "handoff_policy_request",
) -> ActionDecision:
    return ActionDecision(
        decision_id="dec-hoff-1",
        kind=ActionDecisionKind.HANDOFF,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="digest-test",
        arguments_digest=proposal.arguments_digest,
        reason=reason,
        risk_class=RiskClass.HUMAN,
    )


def _adapter(
    store: InMemoryHandoffRequestStore | FileHandoffRequestStore | None = None,
    *,
    sandbox: HandoffSandboxPolicy | None = None,
    clock: float | None = None,
) -> HumanHandoffActionAdapter:
    policy = sandbox or HandoffSandboxPolicy()
    regs = (
        HumanHandoffActionRegistration(
            descriptor_id=HANDOFF_ID,
            logical_action=LOGICAL,
            sandbox=policy,
        ),
    )
    ticks = {"t": clock if clock is not None else 1_700_000_000.0}

    def _now() -> float:
        ticks["t"] += 1.0
        return ticks["t"]

    return HumanHandoffActionAdapter(
        regs,
        store=store or InMemoryHandoffRequestStore(),
        clock=_now,
    )


def test_default_registrations_match_pilot_catalog() -> None:
    regs = default_handoff_registrations()
    assert len(regs) == 1
    assert regs[0].descriptor_id == HANDOFF_ID
    assert regs[0].logical_action == LOGICAL
    mapping = logical_action_to_descriptor_id()
    assert mapping[LOGICAL] == HANDOFF_ID
    catalog = build_pilot_catalog()
    desc = catalog.require(HANDOFF_ID)
    assert desc.adapter == "human"
    assert desc.risk_class is RiskClass.HUMAN


def test_pilot_policy_admits_handoff_request_not_execute() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog(), now=lambda: 1_700_000_000.0)
    proposal = _proposal()
    decision = policy.decide(proposal, PilotAdmissionContext())
    assert decision.kind is ActionDecisionKind.HANDOFF
    assert decision.permits_execution is False
    assert "handoff" in decision.reason


def test_handoff_live_agent_creates_durable_request() -> None:
    store = InMemoryHandoffRequestStore()
    adapter = _adapter(store)
    proposal = _proposal(
        arguments={
            "reason": "caller_requested_specialist",
            "priority": "high",
            "queue": "live_agent",
            "summary": "Caller needs housing intake help.",
        }
    )
    decision = _handoff_decision(proposal)
    receipt = adapter.invoke(proposal=proposal, decision=decision)

    assert receipt.status is ActionStatus.ACCEPTED, receipt.to_dict()
    assert receipt.adapter == "human_handoff"
    assert receipt.public_result["handoff_status"] == "accepted"
    assert receipt.public_result["is_transfer_complete"] == "false"
    assert receipt.public_result["spoken_success_allowed"] == "false"
    assert receipt.metadata["spoken_success_allowed"] == "false"
    assert "summary" not in receipt.public_result  # redacted by default

    request_id = receipt.public_result["request_id"]
    assert request_id.startswith("hoff-")
    stored = store.get(request_id)
    assert stored is not None
    assert isinstance(stored, HandoffRequest)
    assert stored.status is ActionStatus.ACCEPTED
    assert stored.reason == "caller_requested_specialist"
    assert stored.priority == "high"
    assert stored.queue == "live_agent"
    assert stored.summary == "Caller needs housing intake help."
    assert stored.tenant_id == "tenant-a"
    assert stored.proposal_id == proposal.proposal_id
    assert stored.decision_id == decision.decision_id

    # Durable across a second lookup (not invoke-only ephemeral).
    again = adapter.get_request(request_id)
    assert again is not None
    assert again.request_id == request_id
    listed = store.list_requests(tenant_id="tenant-a")
    assert len(listed) == 1
    assert listed[0].request_id == request_id


def test_file_store_is_durable_across_adapter_instances(tmp_path) -> None:
    store_path = tmp_path / "handoffs"
    store = FileHandoffRequestStore(store_path)
    adapter = _adapter(store)
    proposal = _proposal(arguments={"reason": "persist_me"})
    receipt = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    request_id = receipt.public_result["request_id"]

    # New adapter + store instance reading the same directory.
    reloaded = FileHandoffRequestStore(store_path)
    adapter2 = _adapter(reloaded)
    found = adapter2.get_request(request_id)
    assert found is not None
    assert found.status is ActionStatus.ACCEPTED
    assert found.reason == "persist_me"


def test_statuses_distinguish_accepted_started_succeeded_unknown_failed() -> None:
    adapter = _adapter()
    proposal = _proposal(arguments={"reason": "lifecycle"})
    create = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    assert create.status is ActionStatus.ACCEPTED
    request_id = create.public_result["request_id"]

    started = adapter.mark_started(request_id, metadata={"bridge": "fake_pstn"})
    assert started.status is ActionStatus.STARTED
    assert started.public_result["handoff_status"] == "started"
    assert allows_spoken_success(started) is False
    assert spoken_outcome_role(started) == "unknown"

    # Indeterminate provider outcome must not claim success.
    unknown = adapter.record_provider_outcome(
        request_id,
        status=ActionStatus.UNKNOWN,
        provider_confirmation=None,
        metadata={"telephony": "no_ack"},
    )
    assert unknown.status is ActionStatus.UNKNOWN
    assert allows_spoken_success(unknown) is False
    assert spoken_outcome_role(unknown) == "unknown"
    assert unknown.public_result["is_transfer_complete"] == "false"

    # Fresh request path for failed.
    p2 = _proposal(proposal_id="prop-hoff-fail", arguments={"reason": "fail_path"})
    c2 = adapter.invoke(proposal=p2, decision=_handoff_decision(p2))
    rid2 = c2.public_result["request_id"]
    adapter.mark_started(rid2)
    failed = adapter.record_provider_outcome(
        rid2,
        status="failed",
        metadata={"error": "queue_full"},
    )
    assert failed.status is ActionStatus.FAILED
    assert allows_spoken_success(failed) is False
    assert spoken_outcome_role(failed) == "failed"

    # Fresh request path for succeeded (requires provider confirmation).
    p3 = _proposal(proposal_id="prop-hoff-ok", arguments={"reason": "ok_path"})
    c3 = adapter.invoke(proposal=p3, decision=_handoff_decision(p3))
    rid3 = c3.public_result["request_id"]
    adapter.mark_started(rid3)
    succeeded = adapter.record_provider_outcome(
        rid3,
        status=ActionStatus.SUCCEEDED,
        provider_confirmation="pstn-confirm-abc123",
    )
    assert succeeded.status is ActionStatus.SUCCEEDED
    assert allows_spoken_success(succeeded) is True
    assert spoken_outcome_role(succeeded) == "success"
    assert succeeded.public_result["is_transfer_complete"] == "true"
    assert succeeded.public_result["spoken_success_allowed"] == "true"
    assert succeeded.public_result["provider_confirmation"] == "pstn-confirm-abc123"


def test_spoken_success_forbidden_without_succeeded_receipt() -> None:
    # Static gate for every non-succeeded status.
    for status in (
        ActionStatus.ACCEPTED,
        ActionStatus.STARTED,
        ActionStatus.UNKNOWN,
        ActionStatus.FAILED,
        ActionStatus.DENIED,
        ActionStatus.CANCELLED,
        None,
        "accepted",
        "started",
        "unknown",
        "failed",
    ):
        assert allows_spoken_success(status) is False
        assert spoken_outcome_role(status) != "success"

    assert allows_spoken_success(ActionStatus.SUCCEEDED) is True
    assert allows_spoken_success("succeeded") is True
    assert spoken_outcome_role(ActionStatus.SUCCEEDED) == "success"

    adapter = _adapter()
    proposal = _proposal(arguments={"reason": "no_false_warmth"})
    receipt = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    assert receipt.status is ActionStatus.ACCEPTED
    assert allows_spoken_success(receipt) is False
    assert receipt.metadata["spoken_success_allowed"] == "false"
    # Content-plane must not treat request creation as transfer success.
    assert spoken_outcome_role(receipt) == "unknown"


def test_succeeded_requires_provider_confirmation() -> None:
    adapter = _adapter()
    proposal = _proposal(arguments={"reason": "need_confirm"})
    receipt = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    request_id = receipt.public_result["request_id"]
    adapter.mark_started(request_id)

    with pytest.raises(ValueError, match="provider_confirmation_required"):
        adapter.record_provider_outcome(
            request_id,
            status=ActionStatus.SUCCEEDED,
            provider_confirmation=None,
        )

    # Request remains started — no silent upgrade.
    stored = adapter.get_request(request_id)
    assert stored is not None
    assert stored.status is ActionStatus.STARTED
    assert allows_spoken_success(stored) is False


def test_denied_and_confirm_decisions_do_not_create_requests() -> None:
    store = InMemoryHandoffRequestStore()
    adapter = _adapter(store)
    proposal = _proposal()

    deny = ActionDecision(
        decision_id="dec-deny",
        kind=ActionDecisionKind.DENY,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="d",
        arguments_digest=proposal.arguments_digest,
        reason="channel_not_allowed",
        risk_class=RiskClass.HUMAN,
    )
    r1 = adapter.invoke(proposal=proposal, decision=deny)
    assert r1.status is ActionStatus.DENIED
    assert store.list_requests() == ()

    confirm = ActionDecision(
        decision_id="dec-confirm",
        kind=ActionDecisionKind.CONFIRM,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="d",
        arguments_digest=proposal.arguments_digest,
        reason="confirmation_required",
        risk_class=RiskClass.HUMAN,
    )
    r2 = adapter.invoke(proposal=proposal, decision=confirm)
    assert r2.status is ActionStatus.DENIED
    assert "does_not_admit_handoff" in (r2.error or "")
    assert store.list_requests() == ()

    # permit_execute must not smuggle handoff request creation.
    permit = ActionDecision(
        decision_id="dec-permit",
        kind=ActionDecisionKind.PERMIT_EXECUTE,
        proposal_id=proposal.proposal_id,
        descriptor_id=proposal.descriptor_id,
        descriptor_digest="d",
        arguments_digest=proposal.arguments_digest,
        reason="smuggle",
        risk_class=RiskClass.HUMAN,
    )
    r3 = adapter.invoke(proposal=proposal, decision=permit)
    assert r3.status is ActionStatus.DENIED
    assert store.list_requests() == ()


def test_binding_mismatch_and_missing_registration_fail_closed() -> None:
    adapter = _adapter()
    proposal = _proposal()
    decision = _handoff_decision(proposal)
    # Mismatched proposal id.
    bad = ActionDecision(
        decision_id=decision.decision_id,
        kind=decision.kind,
        proposal_id="other-prop",
        descriptor_id=decision.descriptor_id,
        descriptor_digest=decision.descriptor_digest,
        arguments_digest=decision.arguments_digest,
        reason=decision.reason,
        risk_class=decision.risk_class,
    )
    r = adapter.invoke(proposal=proposal, decision=bad)
    assert r.status is ActionStatus.FAILED
    assert "proposal_decision_mismatch" in (r.error or "")

    # Unregistered descriptor.
    empty = HumanHandoffActionAdapter(())
    r2 = empty.invoke(proposal=proposal, decision=decision)
    assert r2.status is ActionStatus.FAILED
    assert "no_handoff_registration" in (r2.error or "")


def test_rejects_bad_priority_queue_and_oversize_summary() -> None:
    adapter = _adapter(sandbox=HandoffSandboxPolicy(max_summary_chars=16))
    bad_pri = _proposal(arguments={"priority": "critical"})
    r1 = adapter.invoke(proposal=bad_pri, decision=_handoff_decision(bad_pri))
    assert r1.status is ActionStatus.FAILED
    assert "unsupported_priority" in (r1.error or "")

    oversize = _proposal(
        proposal_id="prop-over",
        arguments={"summary": "x" * 17, "reason": "ok"},
    )
    r2 = adapter.invoke(proposal=oversize, decision=_handoff_decision(oversize))
    assert r2.status is ActionStatus.FAILED
    assert "summary_exceeds_max_chars" in (r2.error or "")

    bad_queue = _proposal(
        proposal_id="prop-q",
        arguments={"queue": "bad queue;rm"},
    )
    r3 = adapter.invoke(proposal=bad_queue, decision=_handoff_decision(bad_queue))
    assert r3.status is ActionStatus.FAILED


def test_tenant_session_mismatch_fail_closed() -> None:
    adapter = _adapter()
    proposal = _proposal(tenant_id="tenant-a")
    receipt = adapter.invoke(
        proposal=proposal,
        decision=_handoff_decision(proposal),
        context=HandoffInvocationContext(session_tenant_id="tenant-b"),
    )
    assert receipt.status is ActionStatus.FAILED
    assert "tenant_session_mismatch" in (receipt.error or "")


def test_invalid_transition_and_missing_request() -> None:
    adapter = _adapter()
    missing = adapter.mark_started("hoff-does-not-exist")
    assert missing.status is ActionStatus.FAILED
    assert "not_found" in (missing.error or "")

    proposal = _proposal(arguments={"reason": "terminal"})
    create = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    rid = create.public_result["request_id"]
    adapter.mark_started(rid)
    adapter.record_provider_outcome(
        rid,
        status=ActionStatus.SUCCEEDED,
        provider_confirmation="ok-1",
    )
    # Cannot regress succeeded → started.
    regress = adapter.mark_started(rid)
    assert regress.status is ActionStatus.FAILED
    assert "invalid_status_transition" in (regress.error or "") or (
        "status_regression" in (regress.error or "")
    )


def test_summary_redacted_in_receipts_by_default() -> None:
    adapter = _adapter()
    secret = "PRIVATE: caller SSN hint 000-00-0000"
    proposal = _proposal(arguments={"reason": "privacy", "summary": secret})
    receipt = adapter.invoke(proposal=proposal, decision=_handoff_decision(proposal))
    assert receipt.status is ActionStatus.ACCEPTED
    assert secret not in str(receipt.to_dict())
    assert "summary" not in receipt.public_result
    assert receipt.public_result["summary_redacted"] == "true"
    assert receipt.public_result["summary_present"] == "true"
    stored = adapter.get_request(receipt.public_result["request_id"])
    assert stored is not None
    assert stored.summary == secret  # retained in durable store for agents


def test_end_to_end_policy_then_adapter() -> None:
    policy = PilotPolicy(catalog=build_pilot_catalog(), now=lambda: 1_700_000_000.0)
    adapter = _adapter()
    proposal = _proposal(arguments={"reason": "e2e_policy"})
    decision = policy.decide(
        proposal,
        PilotAdmissionContext(confirmed=True),
    )
    assert decision.kind is ActionDecisionKind.HANDOFF
    receipt = adapter.invoke(proposal=proposal, decision=decision)
    assert receipt.status is ActionStatus.ACCEPTED
    assert allows_spoken_success(receipt) is False
    # After fake telephony marks unknown, still no spoken success.
    rid = receipt.public_result["request_id"]
    unknown = adapter.record_provider_outcome(rid, status=ActionStatus.UNKNOWN)
    assert unknown.status is ActionStatus.UNKNOWN
    assert allows_spoken_success(unknown) is False
