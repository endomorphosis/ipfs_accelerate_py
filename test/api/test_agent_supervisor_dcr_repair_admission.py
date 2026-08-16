"""Focused DCR-070 deterministic repair-packet admission tests."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.sca_rpr_admission import (
    AdmittedTargetPacket,
    ProofCarryingRepairPacket,
    RepairPacketAdmissionDisposition,
    RepairPacketEvidence,
    RepairPacketEvidenceKind,
    RepairPacketResolver,
    admit_proof_carrying_repair_packet,
)


def _roots() -> RepairAuthorityRoots:
    return RepairAuthorityRoots(
        repository_id="ipfs_accelerate",
        repository_forest_cid="forest-cid",
        git_tree_id="tree-cid",
        policy_root="policy-cid",
        rpr_plan_cid="plan-cid",
        rpr_packet_cid="packet-root-cid",
    )


def _packet():
    roots = _roots()
    bodies = {
        RepairPacketEvidenceKind.EPOCH: {"epoch_cid": "epoch-cid"},
        RepairPacketEvidenceKind.FOREST: {"forest_cid": "forest-cid"},
        RepairPacketEvidenceKind.GRAPH: {"graph_cid": "graph-cid"},
        RepairPacketEvidenceKind.FINDING: {"finding_cid": "finding-cid"},
        RepairPacketEvidenceKind.DOCTOR: {
            "doctor_receipt_cid": "doctor-receipt-cid",
            "service_identity": "doctor-service-cid",
            "roots_cid": "roots-cid",
        },
        RepairPacketEvidenceKind.PLANNER: {
            "planner_dag_cid": "planner-dag-cid",
            "candidate_cid": "candidate-cid",
            "schedule_cid": "schedule-cid",
            "roots_cid": "roots-cid",
        },
        RepairPacketEvidenceKind.REGISTRY: {"registry_cid": "registry-cid"},
        RepairPacketEvidenceKind.DESCRIPTOR: {
            "descriptor_cid": "descriptor-cid",
            "registry_cid": "registry-cid",
        },
        RepairPacketEvidenceKind.OWNER: {
            "owner_root": "external/ipfs_accelerate",
            "git_head": "head-cid",
            "git_tree": "tree-cid",
            "overlay_cid": "overlay-cid",
        },
        RepairPacketEvidenceKind.SOURCE: {
            "source_path": "external/ipfs_accelerate/module.py",
            "source_span": "line-1-2",
            "old_digest": "sha256-old",
        },
        RepairPacketEvidenceKind.PROOF: {
            "proof_or_counterexample_cid": "proof-cid"
        },
        RepairPacketEvidenceKind.LOGIC: {"stage_gate_cid": "logic-gate-cid"},
        RepairPacketEvidenceKind.IMPACT: {
            "impact_cid": "impact-cid",
            "noninterference_cid": "noninterference-cid",
        },
        RepairPacketEvidenceKind.VALIDATION: {"validation_cid": "validation-cid"},
        RepairPacketEvidenceKind.INVERSE: {"inverse_cid": "inverse-cid"},
        RepairPacketEvidenceKind.LEASE: {
            "lease_cid": "lease-cid",
            "fence_cid": "fence-cid",
        },
    }
    values = tuple(
        RepairPacketEvidence(
            kind=kind,
            authority_roots=roots,
            body=bodies[kind],
            status="reconstructed" if kind is RepairPacketEvidenceKind.PROOF else "passing",
        )
        for kind in RepairPacketEvidenceKind
    )
    by_kind = {value.kind: value.content_id for value in values}
    packet = ProofCarryingRepairPacket(
        repair_id="repair-070",
        authority_roots=roots,
        predecessor_evidence_cid="derived-envelope-cid",
        derivation_cid="derivation-cid",
        evidence_cids=by_kind,
        source_path="external/ipfs_accelerate/module.py",
        source_span="line-1-2",
        old_digest="sha256-old",
        owner_root="external/ipfs_accelerate",
        git_head="head-cid",
        git_tree="tree-cid",
        overlay_cid="overlay-cid",
        write_paths=("external/ipfs_accelerate/module.py",),
        inverse_cid="inverse-cid",
        lease_cid="lease-cid",
        fence_cid="fence-cid",
    )
    return packet, RepairPacketResolver(values), roots


def test_complete_local_packet_remains_pending_and_nonexecuting() -> None:
    packet, resolver, roots = _packet()
    result = admit_proof_carrying_repair_packet(
        packet, resolver=resolver, current_roots=roots
    )

    assert result.disposition is RepairPacketAdmissionDisposition.INTEGRATION_PENDING
    assert result.reason_codes == ("integration_pending_dcr050_dcr060_live_receipts",)
    assert result.packet_cid == packet.content_id
    assert result.worktree_created is False
    assert result.execution_authorized is False
    assert result.completion_authorized is False
    assert result.admission_receipt_cid == result.envelope_cid == ""


def test_legacy_llm_packet_never_authorizes_deterministic_runtime() -> None:
    _packet_value, resolver, roots = _packet()
    legacy = AdmittedTargetPacket(
        schema="ipfs_accelerate_py/agent-supervisor/sca-rpr-admitted-packet@1",
        task_id="legacy",
        snapshot_id="snapshot",
        counterexample_id="counterexample",
        reproof_command="pytest",
    )
    result = admit_proof_carrying_repair_packet(
        legacy, resolver=resolver, current_roots=roots
    )

    assert result.disposition is RepairPacketAdmissionDisposition.REJECTED
    assert (
        "legacy_admitted_target_packet_cannot_authorize_deterministic_runtime"
        in result.reason_codes
    )


def test_unresolvable_cid_and_cross_root_write_set_reject_before_worktree() -> None:
    packet, resolver, roots = _packet()
    values = dict(packet.evidence_cids)
    values[RepairPacketEvidenceKind.DOCTOR] = "missing-doctor-cid"
    rejected = ProofCarryingRepairPacket(
        **{
            **packet.__dict__,
            "evidence_cids": values,
            "write_paths": ("external/ipfs_datasets/other.py",),
        }
    )
    result = admit_proof_carrying_repair_packet(
        rejected, resolver=resolver, current_roots=roots
    )

    assert result.disposition is RepairPacketAdmissionDisposition.REJECTED
    assert "unresolvable_or_untyped_dcr050_doctor" in result.reason_codes
    assert "cross_root_write_set_rejected" in result.reason_codes
    assert result.worktree_created is False


def test_lease_and_fence_must_be_distinct_exact_payload_identities() -> None:
    packet, resolver, roots = _packet()
    same = ProofCarryingRepairPacket(**{**packet.__dict__, "fence_cid": "lease-cid"})
    mismatch = ProofCarryingRepairPacket(**{**packet.__dict__, "lease_cid": "other-lease"})

    for candidate in (same, mismatch):
        result = admit_proof_carrying_repair_packet(
            candidate, resolver=resolver, current_roots=roots
        )
        assert result.disposition is RepairPacketAdmissionDisposition.REJECTED
        assert "lease_or_fence_binding_mismatch" in result.reason_codes


def test_inverse_must_match_its_evidence_payload_identity() -> None:
    packet, resolver, roots = _packet()
    result = admit_proof_carrying_repair_packet(
        ProofCarryingRepairPacket(**{**packet.__dict__, "inverse_cid": "other-inverse"}),
        resolver=resolver,
        current_roots=roots,
    )

    assert result.disposition is RepairPacketAdmissionDisposition.REJECTED
    assert "nonempty_inverse_binding_mismatch" in result.reason_codes
