"""Precision and correct-abstention fixtures for PDR-042 localization."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots as ImpactRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact import (
    DeterministicDoctorImpactAnalyzer,
    DoctorConsumerDisposition,
    DoctorImpactConsumerObservation,
    DoctorImpactFrontierObservation,
    DoctorImpactRequest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization import (
    DOCTOR_CAUSAL_LOCALIZATION_INTERFACE,
    CausalEvidence,
    CausalEvidenceDisposition,
    CausalEvidenceKind,
    CausalLocalizationDisposition,
    DoctorCausalLocalizationError,
    DoctorCausalLocalizationReceipt,
    DoctorCausalLocalizationRequest,
    DoctorCausalLocalizer,
    localize_doctor_cause,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots,
    DoctorSourceUnit,
    FindingKind,
    diagnose_repository,
)


def _snapshot(*, generated: bool = False):
    roots = DoctorAuthorityRoots(
        repository_id="repository:causal-fixture",
        tree_id="tree:causal-fixture",
        dependency_graph_id="graph:causal-fixture",
        policy_id="policy:causal-fixture",
    )
    sources = (
        DoctorSourceUnit(
            path="src/service.py",
            source_bytes=b"def dispatch(payload, context):\n    return payload\n",
            generated=generated,
        ),
        DoctorSourceUnit(
            path="src/caller.py",
            source_bytes=(
                b"from src.service import dispatch\n"
                b"def consume(value):\n    return dispatch(value)\n"
            ),
        ),
    )
    return diagnose_repository(sources, authority_roots=roots)


def _arity_finding(snapshot):
    matches = [item for item in snapshot.findings if item.kind is FindingKind.CALL_ARITY]
    assert len(matches) == 1
    return matches[0]


def _evidence(
    snapshot,
    finding,
    evidence_id: str,
    kind: CausalEvidenceKind | str,
    *,
    causes: tuple[str, ...] = ("cause:dispatch-signature",),
    **kwargs,
) -> CausalEvidence:
    return CausalEvidence(
        evidence_id=evidence_id,
        kind=kind,
        cause_ids=causes,
        fact_refs=finding.observation_refs,
        snapshot_cid=snapshot.snapshot_cid,
        tree_id=snapshot.authority_roots.tree_id,
        graph_id=snapshot.authority_roots.dependency_graph_id,
        index_id=snapshot.authority_roots.ast_index_id,
        **kwargs,
    )


def _fused_evidence(snapshot, finding) -> tuple[CausalEvidence, ...]:
    return (
        _evidence(snapshot, finding, "evidence:contract", "contract_delta"),
        _evidence(snapshot, finding, "evidence:call-graph", "call_graph"),
        _evidence(snapshot, finding, "evidence:dataflow", "dataflow"),
        _evidence(snapshot, finding, "evidence:runtime", "failing_trace"),
        _evidence(
            snapshot,
            finding,
            "evidence:delta-debug",
            "delta_debug",
            minimized=True,
        ),
        _evidence(
            snapshot,
            finding,
            "evidence:unsat-core",
            "unsat_core",
            minimized=True,
        ),
    )


def _impact_roots(snapshot=None) -> ImpactRoots:
    values = {
        "repository_id": "repository:causal-fixture",
        "forest_id": "forest:fixture",
        "tree_id": (
            snapshot.authority_roots.tree_id if snapshot is not None else "tree:fixture"
        ),
        "overlay_id": "overlay:fixture",
        "file_root_id": "files:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": (
            snapshot.authority_roots.dependency_graph_id
            if snapshot is not None
            else "graph:fixture"
        ),
        "corpus_id": "corpus:fixture",
        "index_id": (
            snapshot.authority_roots.ast_index_id
            if snapshot is not None
            else "index:fixture"
        ),
        "model_id": "model:none",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
    }
    return ImpactRoots(**values)


def _impact_receipt(snapshot, *, frontier: bool = False):
    roots = _impact_roots(snapshot)
    frontiers = ()
    if frontier:
        frontiers = (
            DoctorImpactFrontierObservation(
                kind="concurrency", route="thread:worker", required=True
            ),
        )
    return DeterministicDoctorImpactAnalyzer().analyze(
        DoctorImpactRequest(
            roots=roots,
            subject_symbol_id="symbol:dispatch",
            before_contract_ref="contract:dispatch@1",
            after_contract_ref="contract:dispatch@2",
            consumers=(
                DoctorImpactConsumerObservation(
                    consumer_id="consumer:direct",
                    path="src/caller.py",
                    symbol_id="symbol:consume",
                    mandatory=True,
                    disposition=DoctorConsumerDisposition.MIGRATED,
                ),
                DoctorImpactConsumerObservation(
                    consumer_id="consumer:test",
                    path="test/test_caller.py",
                    symbol_id="symbol:test_consume",
                    mandatory=True,
                    disposition=DoctorConsumerDisposition.PROVED_COMPATIBLE,
                ),
            ),
            frontiers=frontiers,
            current_graph_cid=roots.graph_id,
            current_index_cid=roots.index_id,
            current_ast_cid=roots.ast_root_id,
        )
    )


def test_real_checkout_derives_arity_mismatch_without_expected_outcome() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)

    assert finding.expectation_ref in finding.observation_refs
    assert finding.details["observed_argument_count"] == 1
    assert finding.details["expected_argument_count"] == "2"
    assert finding.disposition.value == "supported"


def test_fuses_exact_facts_into_minimal_slice_and_stable_issue_cid() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)
    evidence = _fused_evidence(snapshot, finding)
    first = localize_doctor_cause(
        DoctorCausalLocalizationRequest(
            snapshot=snapshot,
            finding=finding,
            evidence=evidence,
            impact_closure=_impact_receipt(snapshot),
        )
    )
    second = DoctorCausalLocalizer().localize(
        DoctorCausalLocalizationRequest(
            snapshot=snapshot,
            finding=finding,
            evidence=tuple(reversed(evidence)),
            impact_closure=_impact_receipt(snapshot),
        )
    )

    assert DoctorCausalLocalizer.INTERFACE == DOCTOR_CAUSAL_LOCALIZATION_INTERFACE
    assert first.disposition is CausalLocalizationDisposition.LOCALIZED
    assert first.selected_cause_id == "cause:dispatch-signature"
    assert first.issue_cid == second.issue_cid
    assert first.localization_cid == second.localization_cid
    assert first.mismatch_slice.delta_debug_refs
    assert first.mismatch_slice.unsat_core_refs
    assert first.mismatch_slice.contract_refs
    assert first.mismatch_slice.graph_refs
    assert first.mismatch_slice.dataflow_refs
    assert first.mismatch_slice.runtime_refs
    assert set(first.mandatory_consumer_ids) == {
        "consumer:direct",
        "consumer:test",
    }
    assert first.complete_frontier_accounting is True


def test_vector_poison_and_stale_evidence_cannot_choose_cause() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)
    evidence = (
        _evidence(
            snapshot,
            finding,
            "evidence:vector-nearest",
            "vector_nearest",
            causes=("cause:wrong",),
            metadata={"rank": 1, "distance_microunits": 0},
        ),
        _evidence(
            snapshot,
            finding,
            "evidence:poisoned",
            "delta_debug",
            causes=("cause:wrong",),
            poisoned=True,
            minimized=True,
        ),
        CausalEvidence(
            evidence_id="evidence:stale",
            kind="unsat_core",
            cause_ids=("cause:wrong",),
            fact_refs=finding.observation_refs,
            snapshot_cid="snapshot:stale",
            verified=True,
            minimized=True,
        ),
    )
    result = DoctorCausalLocalizer().localize(
        DoctorCausalLocalizationRequest(
            snapshot=snapshot,
            finding=finding,
            evidence=evidence,
        )
    )

    assert result.disposition is CausalLocalizationDisposition.ABSTAINED
    assert result.selected_cause_id == ""
    assert result.nomination_evidence_ids == ("evidence:vector-nearest",)
    assert set(result.rejected_evidence_ids) == {
        "evidence:poisoned",
        "evidence:stale",
    }
    assert (
        result.evidence_dispositions["evidence:vector-nearest"]
        == CausalEvidenceDisposition.NOMINATION_ONLY.value
    )
    assert result.evidence_dispositions["evidence:poisoned"] == "poisoned"
    assert result.evidence_dispositions["evidence:stale"] == "root_mismatch"


def test_poisoned_nomination_does_not_change_semantic_issue_identity() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)
    baseline = snapshot.localize(finding, evidence=_fused_evidence(snapshot, finding))
    poisoned = _evidence(
        snapshot,
        finding,
        "evidence:poisoned-vector",
        "vector_nearest",
        causes=("cause:attacker",),
        poisoned=True,
    )
    enriched = snapshot.localize(
        finding,
        evidence=(*_fused_evidence(snapshot, finding), poisoned),
    )

    assert baseline.issue_cid == enriched.issue_cid
    assert baseline.diagnostic_finding_cid == enriched.diagnostic_finding_cid
    assert enriched.selected_cause_id == baseline.selected_cause_id


def test_conflicting_minimal_solver_facts_correctly_abstain() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)
    evidence = (
        _evidence(snapshot, finding, "evidence:contract", "contract_delta"),
        _evidence(snapshot, finding, "evidence:graph", "dependency_graph"),
        _evidence(
            snapshot,
            finding,
            "evidence:delta",
            "delta_debug",
            causes=("cause:a",),
            minimized=True,
        ),
        _evidence(
            snapshot,
            finding,
            "evidence:core",
            "unsat_core",
            causes=("cause:b",),
            minimized=True,
        ),
    )
    result = snapshot.localize(finding, evidence=evidence)

    assert result.disposition is CausalLocalizationDisposition.ABSTAINED
    assert "contradictory_decisive_evidence" in result.reason_codes
    assert "correct_abstention" in result.reason_codes


def test_generated_native_dynamic_concurrency_and_type_gaps_remain_explicit() -> None:
    snapshot = _snapshot(generated=True)
    finding = _arity_finding(snapshot)
    impact = _impact_receipt(snapshot, frontier=True)
    result = snapshot.localize(
        finding,
        evidence=_fused_evidence(snapshot, finding),
        impact_closure=impact,
        required_frontiers=(
            "frontier:native_ffi",
            "frontier:reflection",
            "frontier:type_analysis",
        ),
    )

    joined = "\n".join(result.open_frontier_refs)
    for marker in (
        "generated_code",
        "native_ffi",
        "reflection",
        "dynamic_dispatch",
        "concurrency",
        "type_analysis",
    ):
        assert marker in joined
    assert set(result.mismatch_slice.open_frontier_refs) == set(
        result.open_frontier_refs
    )


def test_live_impact_mode_refuses_fixture_expected_ids_without_exact_closure() -> None:
    roots = _impact_roots()
    receipt = DeterministicDoctorImpactAnalyzer().analyze(
        DoctorImpactRequest(
            roots=roots,
            subject_symbol_id="symbol:dispatch",
            consumers=(
                DoctorImpactConsumerObservation(
                    consumer_id="consumer:only-known",
                    path="src/caller.py",
                    disposition=DoctorConsumerDisposition.MIGRATED,
                ),
            ),
            expected_consumer_ids=("consumer:only-known",),
            current_graph_cid=roots.graph_id,
            current_index_cid=roots.index_id,
            current_ast_cid=roots.ast_root_id,
            require_authoritative_closure=True,
        )
    )

    assert receipt.mutation_admissible is False
    assert receipt.open_required_frontiers
    assert "authoritative_impact_closure_required" in receipt.reason_codes


def test_receipt_round_trip_and_request_membership_fail_closed() -> None:
    snapshot = _snapshot()
    finding = _arity_finding(snapshot)
    result = snapshot.localize(finding, evidence=_fused_evidence(snapshot, finding))
    restored = DoctorCausalLocalizationReceipt.from_dict(result.to_dict())
    assert restored.content_id == result.content_id
    assert restored.issue_cid == result.issue_cid

    other = diagnose_repository(
        (
            ("src/service.py", "def dispatch(payload, context):\n    return payload\n"),
            (
                "src/caller.py",
                "from src.service import dispatch\ndef consume():\n    return dispatch()\n",
            ),
        ),
        authority_roots=snapshot.authority_roots,
    )
    foreign = _arity_finding(other)
    with pytest.raises(DoctorCausalLocalizationError):
        DoctorCausalLocalizationRequest(snapshot=snapshot, finding=foreign)
