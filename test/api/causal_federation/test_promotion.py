"""CASF-042 promotion, rollback, and quarantine gate acceptance tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import FederationContractError
from ipfs_accelerate_py.agent_supervisor.federation.promotion import (
    FEDERATION_PROMOTION_GATE_INTERFACE,
    DecisionKind,
    DecisionStatus,
    EvidenceOrigin,
    FederationPromotionGate,
    GateDecision,
    GateEvidence,
    GateProfile,
    GateStatus,
    MissingQualificationCapabilityError,
    PromotionGate,
    PromotionGateError,
    QualificationIdentity,
    StaleQualificationEvidenceError,
    evaluate_promotion,
    required_gates,
    validate_current_decision,
)

REVISION = "a" * 40
TREE = "b" * 40
PREVIOUS_REVISION = "c" * 40
PREVIOUS_TREE = "d" * 40


def _identity(**overrides: object) -> QualificationIdentity:
    values: dict[str, object] = {
        "tenant_id": "tenant:test", "federation_id": "federation:test",
        "repository_id": "repository:accelerate", "revision": REVISION,
        "tree_id": TREE, "schema_id": "schema:casf-v1", "generation_id": "generation:7",
        "policy_id": "policy:qualification", "policy_revision": "revision:1",
        "capability_ids": ("capability:quack",), "task_id": "CASF-042",
        "attempt_id": "attempt:1", "fence_id": "fence:7",
    }
    values.update(overrides)
    return QualificationIdentity(**values)  # type: ignore[arg-type]


def _evidence(
    identity: QualificationIdentity,
    profile: GateProfile = GateProfile.DUCKDB_QUACK,
    **overrides: object,
) -> tuple[GateEvidence, ...]:
    values = []
    for index, gate in enumerate(sorted(required_gates(profile), key=lambda item: item.value)):
        item: dict[str, object] = {
            "identity_id": identity.identity_id, "gate": gate, "status": GateStatus.PASSED,
            "receipt_id": "sha256:" + f"{index:064x}", "origin": EvidenceOrigin.STATE_OWNER,
            "observed_effects": True,
        }
        item.update(overrides)
        values.append(GateEvidence(**item))
    return tuple(values)


def _stale_evidence(identity: QualificationIdentity) -> tuple[GateEvidence, ...]:
    evidence = list(_evidence(identity))
    evidence[0] = replace(evidence[0], identity_id="qualification:stale")
    return tuple(evidence)


def test_core_profile_requires_all_non_compensable_and_assurance_gates() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _evidence(identity))

    assert FederationPromotionGate.INTERFACE == FEDERATION_PROMOTION_GATE_INTERFACE
    assert decision.kind is DecisionKind.PROMOTION
    assert decision.status is DecisionStatus.PERMITTED
    assert decision.permitted is True
    assert decision.to_dict()["authoritative_state_changed"] is False
    assert decision.to_dict()["upstream_reverification_required"] is True
    assert PromotionGate.DUCKLAKE_RECEIPT not in required_gates(GateProfile.DUCKDB_QUACK)


def test_missing_gate_fails_closed_with_a_precise_blocker() -> None:
    identity = _identity()
    evidence = tuple(item for item in _evidence(identity) if item.gate is not PromotionGate.NO_EVENT_LOSS)
    decision = FederationPromotionGate.promote(identity, GateProfile.DUCKDB_QUACK, evidence)

    assert decision.status is DecisionStatus.BLOCKED
    assert decision.blockers == ("missing:no_event_loss",)


@pytest.mark.parametrize("status", [GateStatus.BLOCKED, GateStatus.FAILED, GateStatus.UNAVAILABLE])
def test_nonpassing_required_evidence_blocks_promotion(status: GateStatus) -> None:
    identity = _identity()
    evidence = list(_evidence(identity))
    event_index = next(index for index, item in enumerate(evidence) if item.gate is PromotionGate.NO_EVENT_LOSS)
    evidence[event_index] = replace(evidence[event_index], status=status)

    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, tuple(evidence))
    assert decision.blockers == (status.value + ":no_event_loss",)


def test_stale_identity_cannot_pass_a_current_tree_promotion() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _stale_evidence(identity))
    assert decision.status is DecisionStatus.BLOCKED
    assert decision.blockers[0].startswith("stale_identity:")


def test_ducklake_profile_requires_extra_capability_and_receipt_gates() -> None:
    identity = _identity()
    core = _evidence(identity)
    core_decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, core)
    ducklake_decision = evaluate_promotion(identity, GateProfile.DUCKLAKE, core)

    assert core_decision.permitted is True
    assert ducklake_decision.permitted is False
    assert "missing:ducklake_receipt" in ducklake_decision.blockers
    assert evaluate_promotion(identity, GateProfile.DUCKLAKE, _evidence(identity, GateProfile.DUCKLAKE)).permitted


def test_duplicate_or_unsorted_evidence_is_rejected_before_evaluation() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    with pytest.raises(PromotionGateError, match="duplicate"):
        evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, evidence + (evidence[0],))
    with pytest.raises(PromotionGateError, match="sorted"):
        evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, tuple(reversed(evidence)))


def test_model_authored_or_self_authorizing_evidence_is_rejected() -> None:
    identity = _identity()
    item = _evidence(identity)[0]
    with pytest.raises(FederationContractError, match="manufacture authority"):
        replace(item, model_authored=True)
    with pytest.raises(FederationContractError, match="manufacture authority"):
        replace(item, authority_created=True)
    with pytest.raises(PromotionGateError, match="requires effect observation"):
        replace(item, observed_effects=False)


def test_closed_wire_records_detect_unknown_fields_tampering_and_unsafe_flags() -> None:
    identity = _identity()
    with pytest.raises(FederationContractError, match="unknown"):
        QualificationIdentity.from_dict({**identity.to_dict(), "extra": "no"})
    evidence = _evidence(identity)
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, evidence)
    wire = decision.to_dict()
    wire["authority_created"] = True
    with pytest.raises(FederationContractError, match="unsafe authority"):
        GateDecision.from_dict(wire)
    wire = decision.to_dict()
    wire["decision_id"] = "promotion-decision:tampered"
    with pytest.raises(PromotionGateError, match="identity mismatches"):
        GateDecision.from_dict(wire)


def test_current_freshness_reverification_never_applies_the_transition() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _evidence(identity))
    validated = validate_current_decision(
        decision, current_revision=REVISION, current_tree_id=TREE,
        current_generation_id="generation:7", current_fence_id="fence:7", require_permitted=True,
    )
    assert validated["current_fence_bound"] is True
    assert validated["authoritative_state_changed"] is False
    with pytest.raises(StaleQualificationEvidenceError, match="stale fence"):
        validate_current_decision(
            decision, current_revision=REVISION, current_tree_id=TREE,
            current_generation_id="generation:7", current_fence_id="fence:8",
        )


def test_blocked_decision_cannot_be_required_as_permitted() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, ())
    with pytest.raises(MissingQualificationCapabilityError, match="remains blocked"):
        validate_current_decision(
            decision, current_revision=REVISION, current_tree_id=TREE,
            current_generation_id="generation:7", current_fence_id="fence:7", require_permitted=True,
        )


def test_rollback_requires_a_distinct_same_tenant_fenced_predecessor() -> None:
    active = _identity()
    previous = _identity(
        revision=PREVIOUS_REVISION, tree_id=PREVIOUS_TREE, generation_id="generation:6",
        attempt_id="attempt:previous", fence_id="fence:6",
    )
    decision = FederationPromotionGate.rollback(
        active, previous, GateProfile.DUCKDB_QUACK, _evidence(active)
    )
    assert decision.kind is DecisionKind.ROLLBACK
    assert decision.permitted is True
    assert decision.rollback_target == previous
    assert decision.to_dict()["rollback_target"]["generation_id"] == "generation:6"
    with pytest.raises(PromotionGateError, match="tenant or repository"):
        FederationPromotionGate.rollback(
            active, replace(previous, tenant_id="tenant:other"), GateProfile.DUCKDB_QUACK, _evidence(active)
        )
    with pytest.raises(PromotionGateError, match="distinct predecessor"):
        FederationPromotionGate.rollback(
            active, replace(previous, revision=REVISION), GateProfile.DUCKDB_QUACK, _evidence(active)
        )


def test_quarantine_requires_an_observed_blocker_and_never_mutates_state() -> None:
    identity = _identity()
    with pytest.raises(PromotionGateError, match="requires an observed"):
        FederationPromotionGate.quarantine(identity, GateProfile.DUCKDB_QUACK, _evidence(identity))
    evidence = list(_evidence(identity))
    evidence[0] = replace(evidence[0], status=GateStatus.FAILED)
    decision = FederationPromotionGate.quarantine(identity, GateProfile.DUCKDB_QUACK, tuple(evidence))
    assert decision.kind is DecisionKind.QUARANTINE
    assert decision.status is DecisionStatus.BLOCKED
    assert decision.to_dict()["authoritative_state_changed"] is False


def test_identity_rejects_secrets_noncanonical_capabilities_and_non_git_references() -> None:
    with pytest.raises(FederationContractError, match="credential"):
        _identity(policy_id="Bearer token-should-not-arrive-here")
    with pytest.raises(PromotionGateError, match="sorted and unique"):
        _identity(capability_ids=("capability:z", "capability:a"))
    with pytest.raises(PromotionGateError, match="Git object"):
        _identity(revision="tree:current")
