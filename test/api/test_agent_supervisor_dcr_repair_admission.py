"""DCR-070: admit exact proof-carrying repair packets.

Acceptance:
* Any missing/mismatched/unresolvable binding rejects before worktree creation.
* Only derived plus admitted evidence grants execution.
* Synthetic CIDs, booleans, prose, missing plan admission, and stale roots
  cannot authorize mutation / worktree creation.
* Frozen bindings cover epoch, finding, Doctor, Planner, operator, source
  spans/hashes, proof, impact, validations, inverse, owner, and lease.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.sca_rpr_admission import (
    ADMISSION_REASON_CODES,
    DCR_REPAIR_ADMISSION_EVIDENCE,
    FROZEN_BINDING_FIELDS,
    PROOF_CARRYING_REPAIR_PACKET_INTERFACE,
    REPAIR_PACKET_ADMISSION_INTERFACE,
    RPR_INTERFACE,
    AdmissionDisposition,
    AdmissionReason,
    EvidenceStage,
    ProofCarryingRepairPacket,
    RepairPacketAdmission,
    RepairPacketAdmissionError,
    SourceSpanBinding,
    admit_proof_carrying_repair_packet,
    admit_repair_packet,
    build_receipt_store,
    materialize_admission_vectors,
)


WRITE_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/op.py"
)


def _cid(tag: str) -> str:
    return content_identity({"role": "dcr070-receipt", "tag": tag})


def _hash(tag: str) -> str:
    return content_identity({"role": "dcr070-hash", "tag": tag})


def _receipt(tag: str, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "kind": "receipt",
        "tag": tag,
        "value": f"body:{tag}",
    }
    body.update(extra)
    # Strip identity keys so content_identity is stable; store uses derived CID.
    return body


def _span(path: str = WRITE_PATH, digest: str | None = None) -> dict[str, object]:
    return {
        "path": path,
        "start_line": 1,
        "end_line": 3,
        "content_hash": digest or _hash(f"span:{path}"),
    }


def _packet_kwargs(**overrides: object) -> dict[str, object]:
    digest = _hash(f"span:{WRITE_PATH}")
    base: dict[str, object] = {
        "epoch_cid": _cid("epoch"),
        "finding_cid": _cid("finding"),
        "doctor_receipt_cid": _cid("doctor"),
        "planner_receipt_cid": _cid("planner"),
        "plan_cid": _cid("plan"),
        "operator_cid": _cid("operator"),
        "source_spans": (_span(digest=digest),),
        "source_hashes": {WRITE_PATH: digest},
        "proof_cid": _cid("proof"),
        "impact_cid": _cid("impact"),
        "validation_refs": (_cid("validation-a"), _cid("validation-b")),
        "inverse_cid": _cid("inverse"),
        "owner_root": "ipfs-accelerate",
        "lease_id": "lease:node-a:1",
        "fencing_token": "fence:epoch-3:node-a",
        "schedule_cid": _cid("schedule"),
        "candidate_admission_cid": _cid("candidate-admission"),
        "current_evidence_cid": _cid("current-evidence"),
        "forest_cid": _cid("forest"),
        "git_tree_id": _cid("tree"),
        "policy_root": _cid("policy"),
        "evidence_stage": EvidenceStage.DERIVED.value,
        "task_id": "DCR-070",
        "repair_id": "repair:dcr070-fixture",
        "write_paths": (WRITE_PATH,),
        "plan_admission_cid": _cid("plan-admission"),
    }
    base.update(overrides)
    return base


def _packet(**overrides: object) -> ProofCarryingRepairPacket:
    return ProofCarryingRepairPacket(**_packet_kwargs(**overrides))  # type: ignore[arg-type]


def _aligned_fixture() -> tuple[
    ProofCarryingRepairPacket,
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Return packet + receipts + roots + plan admission + schedule that admit."""

    # Build receipts first so packet CIDs match store keys.
    tags = [
        "epoch",
        "finding",
        "doctor",
        "planner",
        "plan",
        "operator",
        "proof",
        "impact",
        "inverse",
        "schedule",
        "candidate-admission",
        "current-evidence",
        "forest",
        "tree",
        "policy",
        "validation-a",
        "validation-b",
        "plan-admission",
    ]
    bodies = {tag: _receipt(tag) for tag in tags}
    cids = {tag: content_identity(body) for tag, body in bodies.items()}

    digest = _hash(f"span:{WRITE_PATH}")
    packet = ProofCarryingRepairPacket(
        epoch_cid=cids["epoch"],
        finding_cid=cids["finding"],
        doctor_receipt_cid=cids["doctor"],
        planner_receipt_cid=cids["planner"],
        plan_cid=cids["plan"],
        operator_cid=cids["operator"],
        source_spans=(SourceSpanBinding.from_dict(_span(digest=digest)),),
        source_hashes={WRITE_PATH: digest},
        proof_cid=cids["proof"],
        impact_cid=cids["impact"],
        validation_refs=(cids["validation-a"], cids["validation-b"]),
        inverse_cid=cids["inverse"],
        owner_root="ipfs-accelerate",
        lease_id="lease:node-a:1",
        fencing_token="fence:epoch-3:node-a",
        schedule_cid=cids["schedule"],
        candidate_admission_cid=cids["candidate-admission"],
        current_evidence_cid=cids["current-evidence"],
        forest_cid=cids["forest"],
        git_tree_id=cids["tree"],
        policy_root=cids["policy"],
        evidence_stage=EvidenceStage.DERIVED,
        task_id="DCR-070",
        repair_id="repair:dcr070-fixture",
        write_paths=(WRITE_PATH,),
        plan_admission_cid=cids["plan-admission"],
    )

    # Enrich plan-admission and schedule bodies with disposition metadata
    # while preserving their content identities by using separate overlays
    # only for the admission call (not the store keys).
    plan_admission_body = {
        **bodies["plan-admission"],
        "disposition": "selected",
        "selected_candidate_cid": cids["candidate-admission"],
        "ok": True,
    }
    # Re-key store: plan-admission body used for resolution must match CID.
    # Keep the pure body in the store; pass enriched plan_admission separately.
    schedule_body = {
        **bodies["schedule"],
        "disposition": "scheduled",
        "assignments": [
            {
                "node_id": "node-a",
                "lease_id": "lease:node-a:1",
                "fencing_token": "fence:epoch-3:node-a",
                "wave": 0,
                "lane": 0,
            }
        ],
    }

    store = {cids[tag]: bodies[tag] for tag in tags}
    # Schedule and plan-admission need their enriched forms in the store only
    # if we want resolution via store; we pass them as explicit kwargs instead.
    current_roots = {
        "forest_cid": cids["forest"],
        "git_tree_id": cids["tree"],
        "policy_root": cids["policy"],
        "epoch_cid": cids["epoch"],
        "current_evidence_cid": cids["current-evidence"],
    }
    return packet, store, current_roots, plan_admission_body, schedule_body


def test_interfaces_and_evidence_are_stable() -> None:
    assert RPR_INTERFACE == "RPR@1"
    assert PROOF_CARRYING_REPAIR_PACKET_INTERFACE == "ProofCarryingRepairPacket@1"
    assert REPAIR_PACKET_ADMISSION_INTERFACE == "RepairPacketAdmission@1"
    assert DCR_REPAIR_ADMISSION_EVIDENCE == "dcr/repair-admission@1"
    assert "epoch_cid" in FROZEN_BINDING_FIELDS
    assert "lease_id" in FROZEN_BINDING_FIELDS
    assert AdmissionReason.ADMITTED.value in ADMISSION_REASON_CODES


def test_happy_path_admits_and_grants_execution() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert isinstance(admission, RepairPacketAdmission)
    assert admission.disposition is AdmissionDisposition.ADMITTED
    assert admission.ok is True
    assert admission.admitted is True
    assert admission.grants_execution is True
    assert admission.allows_worktree_creation is True
    assert admission.runtime_model_calls == 0
    assert admission.packet_cid == packet.packet_cid
    assert admission.authority_transition == "derived->admitted"
    assert admission.canonical_reconstruction_cid == packet.packet_cid
    assert set(packet.referenced_receipt_cids()).issubset(
        set(admission.resolved_receipt_cids)
    )

    subset = admission.evidence_subset()
    assert subset["evidence_id"] == DCR_REPAIR_ADMISSION_EVIDENCE
    assert subset["packet_cid"] == packet.packet_cid
    assert subset["authority_transition"] == "derived->admitted"
    assert subset["grants_execution"] is True

    # Packet itself never self-grants.
    assert packet.grants_execution is False
    assert packet.allows_worktree_creation is False

    # Frozen bindings cover the required effect surface.
    frozen = packet.frozen_bindings()
    for field in (
        "epoch_cid",
        "finding_cid",
        "doctor_receipt_cid",
        "planner_receipt_cid",
        "operator_cid",
        "source_spans",
        "source_hashes",
        "proof_cid",
        "impact_cid",
        "validation_refs",
        "inverse_cid",
        "owner_root",
        "lease_id",
        "fencing_token",
    ):
        assert field in frozen
        assert frozen[field]


def test_alias_admit_repair_packet_matches() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    a = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    b = admit_repair_packet(
        packet.to_dict(),
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert a.admission_cid == b.admission_cid
    assert a.disposition is b.disposition


def test_missing_receipt_rejects_before_worktree_creation() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    # Drop the proof receipt.
    del store[packet.proof_cid]
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert admission.grants_execution is False
    assert admission.allows_worktree_creation is False
    assert any(
        code.startswith(AdmissionReason.UNRESOLVABLE_RECEIPT.value)
        for code in admission.reason_codes
    )
    assert AdmissionReason.WORKTREE_CREATION_DENIED.value in admission.reason_codes


def test_mismatched_receipt_cid_rejects() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    # Poison a stored body so derived identity no longer matches the key.
    with pytest.raises(RepairPacketAdmissionError):
        build_receipt_store(
            {
                packet.proof_cid: {
                    "kind": "receipt",
                    "tag": "tampered",
                    "value": "not-the-original",
                }
            }
        )

    # Direct admission with a hand-built mismatched map is rejected via the
    # store builder inside admit (or unresolvable if key missing).
    poisoned = dict(store)
    poisoned[packet.proof_cid] = {
        "kind": "receipt",
        "tag": "tampered",
        "value": "not-the-original",
    }
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=poisoned,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert admission.allows_worktree_creation is False
    assert any(
        AdmissionReason.RECEIPT_CID_MISMATCH.value in code
        or AdmissionReason.MALFORMED_INPUT.value in code
        for code in admission.reason_codes
    )


def test_synthetic_cid_cannot_construct_packet() -> None:
    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(epoch_cid="true")
    assert AdmissionReason.SYNTHETIC_CID.value in str(excinfo.value)

    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(proof_cid="please fix the handler now with a model")
    assert AdmissionReason.SYNTHETIC_CID.value in str(excinfo.value)

    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(operator_cid="ok")
    assert AdmissionReason.SYNTHETIC_CID.value in str(excinfo.value)


def test_boolean_plan_admission_alone_does_not_grant_execution() -> None:
    packet, store, roots, _plan, schedule = _aligned_fixture()
    # Boolean-only plan admission without a selected disposition still fails
    # closed when disposition is rejected, and boolean keys are flagged.
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission={"admitted": True, "ok": True, "disposition": "rejected"},
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert admission.grants_execution is False
    assert admission.allows_worktree_creation is False
    assert any(
        code.startswith(AdmissionReason.BOOLEAN_AUTHORITY.value)
        or code == AdmissionReason.PLAN_ADMISSION_REJECTED.value
        or code == AdmissionReason.CANDIDATE_NOT_ADMITTED.value
        for code in admission.reason_codes
    )


def test_missing_plan_admission_rejects() -> None:
    packet, store, roots, _plan, schedule = _aligned_fixture()
    # Remove plan_admission_cid from packet and supply no plan_admission kwarg.
    rebuilt = ProofCarryingRepairPacket.from_dict(
        {**packet.to_dict(), "plan_admission_cid": ""}
    )
    # Drop plan-admission from store references by rebuilding without it.
    admission = admit_proof_carrying_repair_packet(
        rebuilt,
        receipts=store,
        current_roots=roots,
        plan_admission=None,
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert AdmissionReason.MISSING_PLAN_ADMISSION.value in admission.reason_codes
    assert admission.allows_worktree_creation is False


def test_stale_roots_reject() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    stale = dict(roots)
    stale["forest_cid"] = _cid("stale-forest")
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=stale,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert any(
        code.startswith(AdmissionReason.STALE_ROOT.value)
        for code in admission.reason_codes
    )
    assert admission.grants_execution is False


def test_lease_mismatch_rejects() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    bad_schedule = dict(schedule)
    bad_schedule["assignments"] = [
        {
            "node_id": "node-a",
            "lease_id": "lease:other",
            "fencing_token": "fence:other",
        }
    ]
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=bad_schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert any(
        code.startswith(AdmissionReason.LEASE_MISMATCH.value)
        for code in admission.reason_codes
    )


def test_observed_evidence_does_not_grant_execution() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    observed = ProofCarryingRepairPacket.from_dict(
        {**packet.to_dict(), "evidence_stage": EvidenceStage.OBSERVED.value}
    )
    admission = admit_proof_carrying_repair_packet(
        observed,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert admission.disposition is AdmissionDisposition.REJECTED
    assert AdmissionReason.EVIDENCE_NOT_DERIVED.value in admission.reason_codes
    assert admission.grants_execution is False


def test_cross_root_write_path_rejected() -> None:
    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(
            write_paths=("swissknife/src/tools/echo.ts",),
            source_spans=(
                _span(
                    path="swissknife/src/tools/echo.ts",
                    digest=_hash("span:swiss"),
                ),
            ),
            source_hashes={
                "swissknife/src/tools/echo.ts": _hash("span:swiss"),
            },
        )
    assert AdmissionReason.CROSS_ROOT_PATH.value in str(excinfo.value)


def test_orchestration_owner_rejected() -> None:
    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(owner_root="orchestration")
    assert AdmissionReason.ORCHESTRATION_WRITE.value in str(excinfo.value)


def test_source_hash_mismatch_rejected() -> None:
    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(
            source_spans=(_span(digest=_hash("span-a")),),
            source_hashes={WRITE_PATH: _hash("span-b")},
        )
    assert AdmissionReason.SOURCE_HASH_MISMATCH.value in str(excinfo.value)


def test_empty_source_spans_rejected() -> None:
    with pytest.raises(RepairPacketAdmissionError) as excinfo:
        _packet(source_spans=(), source_hashes={WRITE_PATH: _hash("x")})
    assert AdmissionReason.EMPTY_SOURCE_SPANS.value in str(excinfo.value)


def test_canonical_roundtrip_preserves_packet_cid() -> None:
    packet, _, _, _, _ = _aligned_fixture()
    rebuilt = ProofCarryingRepairPacket.from_dict(packet.to_dict())
    assert rebuilt.packet_cid == packet.packet_cid
    assert rebuilt.frozen_bindings() == packet.frozen_bindings()


def test_admission_roundtrip_preserves_identity() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    admission = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    rebuilt = RepairPacketAdmission.from_dict(admission.to_dict())
    assert rebuilt.admission_cid == admission.admission_cid
    assert rebuilt.disposition is AdmissionDisposition.ADMITTED


def test_materialize_admission_vectors_in_memory() -> None:
    packet, store, roots, plan_admission, schedule = _aligned_fixture()
    catalog = materialize_admission_vectors(
        [
            {
                "case_id": "happy",
                "packet": packet.to_dict(),
                "receipts": store,
                "current_roots": roots,
                "plan_admission": plan_admission,
                "schedule": schedule,
            },
            {
                "case_id": "missing-plan",
                "packet": {**packet.to_dict(), "plan_admission_cid": ""},
                "receipts": store,
                "current_roots": roots,
                "plan_admission": None,
                "schedule": schedule,
            },
        ]
    )
    assert catalog["evidence_id"] == DCR_REPAIR_ADMISSION_EVIDENCE
    assert catalog["interface"] == RPR_INTERFACE
    assert catalog["runtime_model_calls"] == 0
    assert len(catalog["vectors"]) == 2
    by_id = {item["case_id"]: item for item in catalog["vectors"]}
    assert by_id["happy"]["disposition"] == "admitted"
    assert by_id["happy"]["grants_execution"] is True
    assert by_id["missing-plan"]["disposition"] == "rejected"
    assert by_id["missing-plan"]["allows_worktree_creation"] is False


def test_only_derived_plus_admitted_evidence_grants_execution() -> None:
    """End-to-end: full derived+admitted stack grants; any gap denies worktree."""

    packet, store, roots, plan_admission, schedule = _aligned_fixture()

    admitted = admit_proof_carrying_repair_packet(
        packet,
        receipts=store,
        current_roots=roots,
        plan_admission=plan_admission,
        schedule=schedule,
    )
    assert admitted.grants_execution is True
    assert admitted.allows_worktree_creation is True
    assert admitted.evidence_stage is EvidenceStage.ADMITTED

    # Gap matrix: each single failure denies worktree creation.
    gaps = [
        admit_proof_carrying_repair_packet(
            packet,
            receipts={k: v for k, v in store.items() if k != packet.finding_cid},
            current_roots=roots,
            plan_admission=plan_admission,
            schedule=schedule,
        ),
        admit_proof_carrying_repair_packet(
            packet,
            receipts=store,
            current_roots={**roots, "policy_root": _cid("other-policy")},
            plan_admission=plan_admission,
            schedule=schedule,
        ),
        admit_proof_carrying_repair_packet(
            ProofCarryingRepairPacket.from_dict(
                {**packet.to_dict(), "plan_admission_cid": ""}
            ),
            receipts=store,
            current_roots=roots,
            plan_admission=None,
            schedule=schedule,
        ),
    ]
    for gap in gaps:
        assert gap.disposition is AdmissionDisposition.REJECTED
        assert gap.grants_execution is False
        assert gap.allows_worktree_creation is False
