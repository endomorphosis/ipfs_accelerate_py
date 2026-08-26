"""CASF-042 real-artifact promotion gate acceptance and adversarial tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.chaos import (
    ChaosCapabilityStatus,
    ChaosEvidenceBinding,
    ChaosProofStatus,
    ChaosValidationBinding,
    FederationChaosIdentity,
    build_federation_chaos_suite,
    run_closed_federation_chaos_suite,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationBoundsError,
    FederationContractError,
)
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    FederationControlAuditReceipt,
)
from ipfs_accelerate_py.agent_supervisor.federation.drift_monitor import (
    FederationDriftRoots,
    produce_drift_report,
)
from ipfs_accelerate_py.agent_supervisor.federation.ducklake_projection import (
    ProjectionReceipt,
    ProjectionRecoveryReceipt,
)
from ipfs_accelerate_py.agent_supervisor.federation.fixed_point import FixedPointReceipt
from ipfs_accelerate_py.agent_supervisor.federation.promotion import (
    DECISION_VALIDATION_SCHEMA,
    FEDERATION_PROMOTION_GATE_INTERFACE,
    MAX_JSON_CONTAINER_ITEMS,
    MAX_JSON_TEXT_BYTES,
    PROMOTION_DECISION_SCHEMA,
    PROMOTION_EVIDENCE_BUNDLE_SCHEMA,
    QUALIFICATION_IDENTITY_SCHEMA,
    QUARANTINE_DECISION_SCHEMA,
    ROLLBACK_DECISION_SCHEMA,
    ArtifactAssessment,
    ArtifactStatus,
    DecisionDisposition,
    DecisionKind,
    DecisionStatus,
    EvidenceSlot,
    FederationPromotionGate,
    GateDecision,
    GateProfile,
    MissingQualificationCapabilityError,
    PromotionGateError,
    QualificationEvidenceBundle,
    QualificationIdentity,
    StaleQualificationEvidenceError,
    evaluate_promotion,
    required_slots,
    validate_current_decision,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = ROOT / "benchmarks/agent_supervisor/causal_event_federation"
REVISION = "a" * 40
TREE = "b" * 40
PREVIOUS_REVISION = "c" * 40
PREVIOUS_TREE = "d" * 40


def _ref(index: int) -> str:
    return "sha256:" + f"{index:064x}"


def _identity(**overrides: object) -> QualificationIdentity:
    values: dict[str, object] = {
        "tenant_id": "tenant:test",
        "federation_id": "federation:test",
        "repository_id": "repository:accelerate",
        "revision": REVISION,
        "tree_id": TREE,
        "schema_id": "schema:casf-v1",
        "generation_id": "generation:7",
        "control_plane_generation": 7,
        "policy_id": "policy:qualification",
        "policy_revision": 1,
        "capability_ids": ("capability:quack",),
        "task_id": "CASF-042",
        "attempt_id": "attempt:1",
        "lease_id": "lease:7",
        "fencing_epoch": 7,
        "assignment_revision": 3,
        "worktree_id": "worktree:casf-042",
        "world_snapshot_ref": _ref(42),
        "event_watermark": 42,
    }
    values.update(overrides)
    return QualificationIdentity(**values)  # type: ignore[arg-type]


def _fixed_point(identity: QualificationIdentity) -> dict[str, object]:
    receipt = FixedPointReceipt(
        world_snapshot_ref=identity.world_snapshot_ref,
        event_watermark=identity.event_watermark,
        outstanding_required_work=0,
        fencing_epoch=identity.fencing_epoch,
        outcome="fixed_point",
        evidence_refs=("evidence:fixed-point",),
    )
    return {
        "schema": receipt.SCHEMA,
        "world_snapshot_ref": receipt.world_snapshot_ref,
        "event_watermark": receipt.event_watermark,
        "outstanding_required_work": receipt.outstanding_required_work,
        "fencing_epoch": receipt.fencing_epoch,
        "outcome": receipt.outcome,
        "evidence_refs": list(receipt.evidence_refs),
        "receipt_id": receipt.cid,
    }


def _projection(identity: QualificationIdentity) -> dict[str, object]:
    receipt = ProjectionReceipt(
        status="current",
        source_root="source:events",
        tree_id=identity.tree_id,
        from_watermark=1,
        to_watermark=2,
        source_checksum="checksum:source",
        cursor_watermark=2,
        partition_ids=("partition:one",),
        authoritative=False,
    )
    return {
        "schema": receipt.SCHEMA,
        "status": receipt.status,
        "source_root": receipt.source_root,
        "tree_id": receipt.tree_id,
        "from_watermark": receipt.from_watermark,
        "to_watermark": receipt.to_watermark,
        "source_checksum": receipt.source_checksum,
        "cursor_watermark": receipt.cursor_watermark,
        "partition_ids": list(receipt.partition_ids),
        "authoritative": False,
        "receipt_id": receipt.cid,
    }


def _recovery(identity: QualificationIdentity) -> dict[str, object]:
    receipt = ProjectionRecoveryReceipt(
        status="current",
        tenant_id=identity.tenant_id,
        schema_revision=2,
        recovered_from_watermark=3,
        recovered_to_watermark=4,
        preserved_partition_ids=("partition:one",),
        recovered_partition_ids=("partition:two",),
        rewritten=False,
        authoritative=False,
    )
    return {
        "schema": receipt.SCHEMA,
        "status": receipt.status,
        "tenant_id": receipt.tenant_id,
        "schema_revision": receipt.schema_revision,
        "recovered_from_watermark": receipt.recovered_from_watermark,
        "recovered_to_watermark": receipt.recovered_to_watermark,
        "preserved_partition_ids": list(receipt.preserved_partition_ids),
        "recovered_partition_ids": list(receipt.recovered_partition_ids),
        "rewritten": False,
        "authoritative": False,
        "receipt_id": receipt.cid,
    }


def _drift(identity: QualificationIdentity) -> dict[str, object]:
    roots = FederationDriftRoots(
        tenant_id=identity.tenant_id,
        federation_id=identity.federation_id,
        repository_id=identity.repository_id,
        repository_tree_id=identity.tree_id,
        control_plane_generation=identity.control_plane_generation,
        schema_root=identity.schema_id,
        operation_catalog_root="catalog:operations",
        event_catalog_root="catalog:events",
        causal_graph_root="graph:current",
        causal_graph_revision=4,
        event_watermark=identity.event_watermark,
    )
    return produce_drift_report(
        roots,
        roots,
        observed_at="2026-08-24T00:00:00Z",
    ).to_dict()


def _control_audit(identity: QualificationIdentity) -> dict[str, object]:
    return FederationControlAuditReceipt(
        audit_id="audit:current",
        command_cid="command:current",
        authorization_id="authorization:current",
        result_ref="result:current",
        outcome="dry_run",
        control_plane_generation=identity.control_plane_generation,
        fencing_epoch=identity.fencing_epoch,
        recorded_at="2026-08-24T00:00:00Z",
    ).to_dict()


def _chaos(identity: QualificationIdentity) -> dict[str, object]:
    chaos_identity = FederationChaosIdentity(
        source_revision=identity.revision,
        source_tree=identity.tree_id,
        state_schema=identity.schema_id,
        generation_id=identity.generation_id,
        federation_id=identity.federation_id,
        policy_id=identity.policy_id,
        policy_revision=identity.policy_revision,
        capability_ids=identity.capability_ids,
        task_id="CASF-037",
        attempt_id="attempt:chaos",
        lease_id="lease:chaos",
        fencing_epoch=identity.fencing_epoch,
        assignment_revision=identity.assignment_revision,
        worktree_id="worktree:casf-037",
    )
    suite = build_federation_chaos_suite(chaos_identity)
    validation = ChaosValidationBinding(
        target_revision=identity.revision,
        validated_revision=identity.revision,
        target_tree=identity.tree_id,
        receipt_id=_ref(101),
        result_ref=_ref(102),
        payload_ref=_ref(103),
        attempted=False,
        passed=False,
        returncode=1,
        stale=False,
    )
    evidence = ChaosEvidenceBinding(
        suite_id=suite.suite_id,
        validation=validation,
        rollback_revision=PREVIOUS_REVISION,
        rollback_tree=PREVIOUS_TREE,
        rollback_generation_id="generation:6",
        capability_ids=identity.capability_ids,
        capability_statuses=(ChaosCapabilityStatus.UNAVAILABLE,),
        capability_receipt_ids=("",),
        proof_property_ids=("property:all",),
        proof_statuses=(ChaosProofStatus.UNAVAILABLE,),
        proof_receipt_ids=("",),
        observation_ids=(),
    )
    return run_closed_federation_chaos_suite(suite, evidence).to_dict()


def _manifest(name: str) -> dict[str, object]:
    return json.loads((BENCHMARK_ROOT / name).read_text(encoding="utf-8"))


def _bundle(
    identity: QualificationIdentity,
    **overrides: object,
) -> QualificationEvidenceBundle:
    values: dict[str, object] = {
        "identity_id": identity.identity_id,
        "fixed_point_receipt": _fixed_point(identity),
        "ducklake_projection_receipt": _projection(identity),
        "ducklake_recovery_receipt": _recovery(identity),
        "drift_report": _drift(identity),
        "control_audit_receipt": _control_audit(identity),
        "control_parity_report": None,
        "formal_report": None,
        "adversarial_report": _chaos(identity),
        "idle_benchmark": _manifest("idle_manifest.json"),
        "parallel_benchmark": _manifest("parallel_manifest.json"),
        "load_benchmark": _manifest("load_manifest.json"),
        "token_benchmark": _manifest("token_manifest.json"),
    }
    values.update(overrides)
    return QualificationEvidenceBundle(**values)  # type: ignore[arg-type]


def _assessment(decision: GateDecision, slot: EvidenceSlot):
    return next(item for item in decision.assessments if item.slot is slot)


def test_current_real_artifacts_truthfully_require_quarantine() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity))

    assert FederationPromotionGate.INTERFACE == FEDERATION_PROMOTION_GATE_INTERFACE
    assert FEDERATION_PROMOTION_GATE_INTERFACE == "FederationPromotionGate@1"
    assert QUALIFICATION_IDENTITY_SCHEMA.endswith("qualification-identity@1")
    assert PROMOTION_EVIDENCE_BUNDLE_SCHEMA == "casf/promotion-evidence-bundle@1"
    assert PROMOTION_DECISION_SCHEMA == "casf/promotion-decision@1"
    assert ROLLBACK_DECISION_SCHEMA == "casf/rollback-decision@1"
    assert QUARANTINE_DECISION_SCHEMA == "casf/quarantine-decision@1"
    assert DECISION_VALIDATION_SCHEMA == "casf/promotion-decision-validation@1"
    assert decision.kind is DecisionKind.PROMOTION
    assert decision.status is DecisionStatus.BLOCKED
    assert decision.disposition is DecisionDisposition.QUARANTINE_REQUIRED
    assert all(item.status is not ArtifactStatus.PASSED for item in decision.assessments)
    assert "missing:casf_035_control_parity_report_decoder" in decision.blockers
    assert "missing:casf_036_formal_report_decoder" in decision.blockers
    assert "blocked:casf_037_local_qualification_unavailable" in decision.blockers
    for task in ("casf_038", "casf_039", "casf_040", "casf_041"):
        assert f"unavailable:{task}_live_not_run" in decision.blockers
    for task, slot in (
        ("casf_030", EvidenceSlot.FIXED_POINT),
        ("casf_033", EvidenceSlot.DRIFT),
    ):
        assessment = _assessment(decision, slot)
        assert assessment.status is ArtifactStatus.NONAUTHORITATIVE
        assert f"missing:{task}_accepted_producer_provenance" in assessment.blockers
        assert f"missing:{task}_full_qualification_identity_binding" in assessment.blockers
        assert f"missing:{task}_state_owner_provenance" in assessment.blockers


def test_decision_explicitly_claims_no_release_authority_or_applied_effect() -> None:
    identity = _identity()
    wire = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity)).to_dict()

    assert wire["promotion_eligible"] is False
    assert wire["release_eligible"] is False
    assert wire["quarantine_required"] is True
    assert wire["upstream_reverification_required"] is True
    for name in (
        "promotion_applied",
        "quarantine_applied",
        "rollback_applied",
        "production_state_changed",
        "authoritative_state_changed",
        "authority_created",
        "completion_created",
    ):
        assert wire[name] is False


def test_generic_caller_constructed_passed_evidence_has_no_admission_surface() -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import promotion

    identity = _identity()
    assert not hasattr(promotion, "GateEvidence")
    assert not hasattr(promotion, "GateStatus")
    assert not hasattr(promotion, "EvidenceOrigin")
    fabricated = tuple(
        {
            "gate": slot.value,
            "status": "passed",
            "origin": "state_owner",
            "receipt_id": _ref(index),
        }
        for index, slot in enumerate(required_slots(GateProfile.DUCKDB_QUACK))
    )
    with pytest.raises(PromotionGateError, match="exact evidence bundle"):
        evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, fabricated)  # type: ignore[arg-type]


def test_empty_bundle_fails_closed_and_ducklake_is_independent() -> None:
    identity = _identity()
    empty = QualificationEvidenceBundle(identity_id=identity.identity_id)
    core = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, empty)
    ducklake = evaluate_promotion(identity, GateProfile.DUCKLAKE, empty)

    assert core.status is DecisionStatus.BLOCKED
    assert not any("casf_032" in blocker for blocker in core.blockers)
    assert "missing:casf_032_ducklake_projection" in ducklake.blockers
    assert "missing:casf_032_ducklake_recovery" in ducklake.blockers


def test_current_ducklake_receipts_cannot_establish_the_core_profile() -> None:
    identity = _identity()
    bundle = QualificationEvidenceBundle(
        identity_id=identity.identity_id,
        ducklake_projection_receipt=_projection(identity),
        ducklake_recovery_receipt=_recovery(identity),
    )
    core = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, bundle)
    ducklake = evaluate_promotion(identity, GateProfile.DUCKLAKE, bundle)

    for slot in (EvidenceSlot.DUCKLAKE_PROJECTION, EvidenceSlot.DUCKLAKE_RECOVERY):
        assessment = _assessment(ducklake, slot)
        assert assessment.status is ArtifactStatus.NONAUTHORITATIVE
        assert "missing:casf_032_full_qualification_identity_binding" in assessment.blockers
        assert "missing:casf_032_accepted_producer_provenance" in assessment.blockers
        assert "missing:casf_032_state_owner_provenance" in assessment.blockers
    assert core.status is DecisionStatus.BLOCKED
    assert ducklake.status is DecisionStatus.BLOCKED
    assert "missing:casf_030_fixed_point" in ducklake.blockers


def test_projection_tree_and_receipt_content_are_revalidated() -> None:
    identity = _identity()
    stale = _projection(_identity(tree_id=PREVIOUS_TREE))
    # A self-consistent receipt for another tree remains stale even with a valid CID.
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKLAKE,
        _bundle(identity, ducklake_projection_receipt=stale),
    )
    assert "stale:casf_032_projection_tree" in decision.blockers

    tampered = _fixed_point(identity)
    tampered["receipt_id"] = _ref(99)
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        _bundle(identity, fixed_point_receipt=tampered),
    )
    assert "invalid:casf_030_fixed_point" in decision.blockers


def test_artifact_schema_substitution_and_forged_reports_fail_closed() -> None:
    identity = _identity()
    parallel = _manifest("token_manifest.json")
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        _bundle(
            identity,
            parallel_benchmark=parallel,
            control_parity_report={"schema": "casf/control-parity@1", "passed": True},
            formal_report={"schema": "casf/formal-model-report@1", "passed": True},
        ),
    )

    assert "invalid:casf_039_benchmark" in decision.blockers
    assert "unsupported:casf_035_control_parity_report_decoder" in decision.blockers
    assert "unsupported:casf_036_formal_report_decoder" in decision.blockers


def test_benchmark_nonpromotion_flags_cannot_be_flipped() -> None:
    identity = _identity()
    parallel = _manifest("parallel_manifest.json")
    parallel["promotion_eligible"] = True
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        _bundle(identity, parallel_benchmark=parallel),
    )
    assert "invalid:casf_039_benchmark" in decision.blockers


def test_adversarial_report_is_decoded_but_remains_nonpromotional() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity))
    chaos = _assessment(decision, EvidenceSlot.ADVERSARIAL)
    assert chaos.schema_id == "casf/adversarial-report@1"
    assert chaos.status is ArtifactStatus.BLOCKED
    assert chaos.authoritative is False

    foreign_identity = _identity(tree_id=PREVIOUS_TREE)
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        _bundle(identity, adversarial_report=_chaos(foreign_identity)),
    )
    assert "stale:casf_037_adversarial_identity" in decision.blockers


def test_bundle_identity_mismatch_is_rejected_before_artifact_evaluation() -> None:
    identity = _identity()
    other = _identity(attempt_id="attempt:other")
    with pytest.raises(StaleQualificationEvidenceError, match="another identity"):
        evaluate_promotion(
            identity,
            GateProfile.DUCKDB_QUACK,
            QualificationEvidenceBundle(identity_id=other.identity_id),
        )


def test_decision_round_trip_is_content_addressed_and_tamper_evident() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity))
    replayed = GateDecision.from_dict(decision.to_dict())
    assert replayed == decision
    assert replayed.decision_id == decision.decision_id

    tampered = decision.to_dict()
    tampered["decision_id"] = "promotion-decision:" + _ref(1).removeprefix("sha256:")
    with pytest.raises(PromotionGateError, match="mismatch"):
        GateDecision.from_dict(tampered)

    unsafe = decision.to_dict()
    unsafe["promotion_applied"] = True
    with pytest.raises(FederationContractError, match="applied or authoritative"):
        GateDecision.from_dict(unsafe)


def test_stale_replay_and_required_permission_fail_closed() -> None:
    identity = _identity()
    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity))
    validated = validate_current_decision(decision, current_identity=identity)
    assert validated["current_identity_bound"] is True
    assert validated["quarantine_required"] is True
    assert validated["production_state_changed"] is False

    with pytest.raises(StaleQualificationEvidenceError, match="stale"):
        validate_current_decision(
            decision,
            current_identity=_identity(fencing_epoch=8, lease_id="lease:8"),
        )
    with pytest.raises(MissingQualificationCapabilityError, match="remains blocked"):
        validate_current_decision(
            decision,
            current_identity=identity,
            require_permitted=True,
        )


def test_rollback_and_quarantine_are_recommendations_not_applied_transitions() -> None:
    active = _identity()
    predecessor = _identity(
        revision=PREVIOUS_REVISION,
        tree_id=PREVIOUS_TREE,
        generation_id="generation:6",
        control_plane_generation=6,
        attempt_id="attempt:previous",
        lease_id="lease:6",
        fencing_epoch=6,
        assignment_revision=2,
        worktree_id="worktree:previous",
    )
    rollback = FederationPromotionGate.rollback(
        active,
        predecessor,
        GateProfile.DUCKDB_QUACK,
        _bundle(active),
    )
    assert rollback.kind is DecisionKind.ROLLBACK
    assert "missing:rollback_state_owner_predecessor_receipt" in rollback.blockers
    assert rollback.to_dict()["rollback_applied"] is False
    assert rollback.disposition is DecisionDisposition.QUARANTINE_REQUIRED

    quarantine = FederationPromotionGate.quarantine(
        active,
        GateProfile.DUCKDB_QUACK,
        _bundle(active),
    )
    assert quarantine.kind is DecisionKind.QUARANTINE
    assert quarantine.to_dict()["quarantine_required"] is True
    assert quarantine.to_dict()["quarantine_applied"] is False


def test_exact_types_secrets_cycles_and_bounds_are_rejected() -> None:
    class Text(str):
        pass

    with pytest.raises(PromotionGateError, match="exact text"):
        _identity(tenant_id=Text("tenant:test"))
    with pytest.raises(PromotionGateError, match="exact integer"):
        _identity(control_plane_generation=True)
    with pytest.raises(PromotionGateError, match="content addressed"):
        _identity(world_snapshot_ref="snapshot:mutable")
    with pytest.raises(FederationContractError, match="credential"):
        _identity(policy_id="github_pat_FAKESECRET1234567890")

    identity = _identity()
    cyclic: dict[str, object] = {"schema": "casf/fake@1"}
    cyclic["self"] = cyclic
    with pytest.raises(PromotionGateError, match="cycle"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            control_parity_report=cyclic,
        )
    with pytest.raises(FederationContractError, match="credential"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            formal_report={"schema": "casf/fake@1", "value": "ghp_FAKESECRET123456"},
        )
    with pytest.raises(FederationContractError, match="credential"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            formal_report={"schema": "casf/fake@1", "value": "AKIAABCDEFGHIJKLMNOP"},
        )
    with pytest.raises(FederationBoundsError, match="oversized text"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            formal_report={"schema": "casf/fake@1", "value": "x" * (MAX_JSON_TEXT_BYTES + 1)},
        )
    with pytest.raises(FederationBoundsError, match="oversized container"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            formal_report={
                "schema": "casf/fake@1",
                "items": [None] * (MAX_JSON_CONTAINER_ITEMS + 1),
            },
        )


def test_wire_arrays_and_adversarial_equality_values_are_exact() -> None:
    class Text(str):
        pass

    class AlwaysEqual:
        def __eq__(self, _other: object) -> bool:
            return True

    identity = _identity()
    identity_wire = identity.to_dict()
    identity_wire["capability_ids"] = "capability:quack"
    identity_wire["identity_id"] = "qualification:" + content_identity(
        {key: value for key, value in identity_wire.items() if key != "identity_id"}
    )
    with pytest.raises(PromotionGateError, match="exact JSON array"):
        QualificationIdentity.from_dict(identity_wire)

    subclass_wire = identity.to_dict()
    subclass_wire["tenant_id"] = Text("tenant:test")
    with pytest.raises(PromotionGateError, match="non-JSON exact type"):
        QualificationIdentity.from_dict(subclass_wire)

    decision = evaluate_promotion(identity, GateProfile.DUCKDB_QUACK, _bundle(identity))
    decision_wire = decision.to_dict()
    decision_wire["assessments"] = "fabricated"
    with pytest.raises(PromotionGateError, match="exact JSON array"):
        GateDecision.from_dict(decision_wire)

    assessment_wire = decision.assessments[0].to_dict()
    assessment_wire["blockers"] = "passed"
    with pytest.raises(PromotionGateError, match="exact JSON array"):
        ArtifactAssessment.from_dict(assessment_wire)

    equality_wire = decision.to_dict()
    equality_wire["kind"] = AlwaysEqual()
    with pytest.raises(PromotionGateError, match="non-JSON exact type"):
        GateDecision.from_dict(equality_wire)

    fixed = _fixed_point(identity)
    fixed["evidence_refs"] = "evidence:fixed-point"
    malformed = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        _bundle(identity, fixed_point_receipt=fixed),
    )
    assert "invalid:casf_030_fixed_point" in malformed.blockers


@pytest.mark.parametrize(
    "identity_override",
    (
        {"tenant_id": "tenant:transplanted"},
        {"federation_id": "federation:transplanted"},
        {"repository_id": "repository:transplanted"},
        {"schema_id": "schema:transplanted"},
        {"policy_id": "policy:transplanted"},
        {"policy_revision": 2},
        {"attempt_id": "attempt:transplanted"},
        {"lease_id": "lease:transplanted"},
        {"assignment_revision": 4},
        {"worktree_id": "worktree:transplanted"},
    ),
)
def test_partial_casf_030_032_033_artifacts_never_pass_for_transplanted_identity(
    identity_override: dict[str, object],
) -> None:
    source = _identity()
    current = _identity(**identity_override)
    decision = evaluate_promotion(
        current,
        GateProfile.DUCKLAKE,
        _bundle(
            current,
            fixed_point_receipt=_fixed_point(source),
            ducklake_projection_receipt=_projection(source),
            ducklake_recovery_receipt=_recovery(source),
            drift_report=_drift(source),
        ),
    )

    for slot in (
        EvidenceSlot.FIXED_POINT,
        EvidenceSlot.DUCKLAKE_PROJECTION,
        EvidenceSlot.DUCKLAKE_RECOVERY,
        EvidenceSlot.DRIFT,
    ):
        assert _assessment(decision, slot).status is not ArtifactStatus.PASSED
    assert "missing:casf_030_full_qualification_identity_binding" in decision.blockers
    assert "missing:casf_033_full_qualification_identity_binding" in decision.blockers


def test_task_transplant_is_rejected_and_partial_receipts_bind_no_task() -> None:
    identity_wire = _identity().to_dict()
    identity_wire["task_id"] = "CASF-030"
    identity_wire["identity_id"] = "qualification:" + content_identity(
        {key: value for key, value in identity_wire.items() if key != "identity_id"}
    )
    with pytest.raises(PromotionGateError, match="exact CASF-042"):
        QualificationIdentity.from_dict(identity_wire)

    identity = _identity()
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            fixed_point_receipt=_fixed_point(identity),
            drift_report=_drift(identity),
        ),
    )
    assert "missing:casf_030_full_qualification_identity_binding" in decision.blockers
    assert "missing:casf_033_full_qualification_identity_binding" in decision.blockers


def test_fixed_point_cid_rejects_transplanted_evidence_refs() -> None:
    identity = _identity()
    original = _fixed_point(identity)
    transplanted = dict(original)
    transplanted["evidence_refs"] = ["evidence:caller-transplanted"]
    assert transplanted["receipt_id"] == original["receipt_id"]

    original_decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            fixed_point_receipt=original,
        ),
    )
    transplanted_decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            fixed_point_receipt=transplanted,
        ),
    )
    original_assessment = _assessment(original_decision, EvidenceSlot.FIXED_POINT)
    transplanted_assessment = _assessment(transplanted_decision, EvidenceSlot.FIXED_POINT)
    assert original_assessment.status is ArtifactStatus.NONAUTHORITATIVE
    assert transplanted_assessment.status is ArtifactStatus.INVALID
    assert original_assessment.artifact_id != transplanted_assessment.artifact_id
    assert transplanted_assessment.blockers == ("invalid:casf_030_fixed_point",)


@pytest.mark.parametrize(
    "secret_key",
    (
        "password",
        "passwd",
        "authorization",
        "proxy_authorization",
        "proxyAuthorization",
        "auth_header",
        "authHeader",
        "authorization_header",
        "aws_secret_access_key",
        "aws_access_key_id",
        "awsSecretAccessKey",
        "AWSAccessKeyId",
        "access_key",
        "accessKey",
        "secret",
        "secretkey",
        "secretKey",
        "secret_access_key",
        "secretAccessKey",
        "private_key",
        "privateKey",
        "client_secret",
        "clientSecret",
        "api_key",
        "apiKey",
        "api_token",
        "apiToken",
        "access_token",
        "accessToken",
        "refresh_token",
        "session_token",
        "signing_secret",
        "signingSecret",
        "signing_key",
        "signingKey",
        "webhook_secret",
        "webhookSecret",
        "connection_string",
        "connectionString",
        "cookie",
        "set_cookie",
        "setCookie",
        "jwt",
        "jwt_token",
        "credential",
    ),
)
def test_secret_shaped_json_keys_are_rejected_before_bundle_storage(
    secret_key: str,
) -> None:
    identity = _identity()
    with pytest.raises(FederationContractError, match="unsafe object key"):
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            formal_report={
                "schema": "casf/fake@1",
                "nested": {secret_key: "placeholder-only"},
            },
        )


def test_secret_reference_names_require_content_addressed_public_handles() -> None:
    identity = _identity()
    for key in ("client_secret_ref", "authorization_ref", "connection_string_ref"):
        with pytest.raises(FederationContractError, match="unsafe object key"):
            QualificationEvidenceBundle(
                identity_id=identity.identity_id,
                formal_report={"schema": "casf/fake@1", key: "inline-value"},
            )

    bundle = QualificationEvidenceBundle(
        identity_id=identity.identity_id,
        formal_report={
            "schema": "casf/fake@1",
            "clientSecretRef": _ref(77),
            "authorization_ref": _ref(78),
            "authorization_id": "authorization:public-handle",
        },
    )
    assert bundle.artifact(EvidenceSlot.FORMAL) == {
        "schema": "casf/fake@1",
        "clientSecretRef": _ref(77),
        "authorization_ref": _ref(78),
        "authorization_id": "authorization:public-handle",
    }


def test_benchmark_manifest_validation_executes_no_runner_or_global_cache() -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import promotion

    before = frozenset(name for name in sys.modules if name.startswith("_casf_promotion_"))
    decision = evaluate_promotion(_identity(), GateProfile.DUCKDB_QUACK, _bundle(_identity()))
    after = frozenset(name for name in sys.modules if name.startswith("_casf_promotion_"))

    assert before == after
    assert not hasattr(promotion, "_load_benchmark_runner")
    assert not hasattr(promotion, "_RUNNER_CACHE")
    for slot in (
        EvidenceSlot.IDLE,
        EvidenceSlot.PARALLEL,
        EvidenceSlot.LOAD,
        EvidenceSlot.TOKEN,
    ):
        assert _assessment(decision, slot).status is ArtifactStatus.UNAVAILABLE


def test_benchmark_result_without_native_pure_decoder_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import promotion

    def forbidden_manifest(_task_id: str):
        raise AssertionError("result fallback attempted to read or execute a validator")

    monkeypatch.setattr(promotion, "_load_pinned_benchmark_manifest", forbidden_manifest)
    identity = _identity()
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(
            identity_id=identity.identity_id,
            parallel_benchmark={"schema": "casf/parallel-benchmark@1"},
        ),
    )
    assert "unsupported:casf_039_pure_result_decoder_unavailable" in decision.blockers


def test_pinned_manifest_reader_rejects_symlinks_and_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import promotion

    identity = _identity()
    idle = _manifest("idle_manifest.json")
    target = tmp_path / "target.json"
    target.write_text(json.dumps(idle), encoding="utf-8")
    (tmp_path / "idle_manifest.json").symlink_to(target)
    monkeypatch.setattr(promotion, "_BENCHMARK_ROOT", tmp_path)
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(identity_id=identity.identity_id, idle_benchmark=idle),
    )
    assert "invalid:casf_038_benchmark" in decision.blockers

    (tmp_path / "idle_manifest.json").unlink()
    (tmp_path / "idle_manifest.json").write_text(json.dumps(idle), encoding="utf-8")
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(identity_id=identity.identity_id, idle_benchmark=idle),
    )
    assert "invalid:casf_038_benchmark" in decision.blockers


def test_bundle_copies_inputs_and_rejects_unknown_wire_fields() -> None:
    identity = _identity()
    fixed = _fixed_point(identity)
    bundle = QualificationEvidenceBundle(
        identity_id=identity.identity_id,
        fixed_point_receipt=fixed,
    )
    before = bundle.bundle_id
    fixed["outcome"] = "fabricated"
    assert bundle.bundle_id == before
    assert bundle.artifact(EvidenceSlot.FIXED_POINT)["outcome"] == "fixed_point"  # type: ignore[index]

    wire = bundle.to_dict()
    wire["unknown"] = True
    with pytest.raises(FederationContractError, match="unknown"):
        QualificationEvidenceBundle.from_dict(wire)


def test_missing_bundle_evaluation_has_no_state_or_provider_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.federation import promotion

    identity = _identity()

    def forbidden_manifest(_task_id: str):
        raise AssertionError("missing evidence attempted to read a benchmark manifest")

    assert not hasattr(promotion, "_load_benchmark_runner")
    assert not hasattr(promotion, "_RUNNER_CACHE")
    monkeypatch.setattr(promotion, "_load_pinned_benchmark_manifest", forbidden_manifest)
    decision = evaluate_promotion(
        identity,
        GateProfile.DUCKDB_QUACK,
        QualificationEvidenceBundle(identity_id=identity.identity_id),
    )
    assert decision.status is DecisionStatus.BLOCKED
    assert list(tmp_path.iterdir()) == []
