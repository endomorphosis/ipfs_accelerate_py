"""Hermetic tests for exact vs nomination-only CASF causal evidence admission."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization import (
    FEDERATION_KIND_FOR_DOCTOR_KIND,
    DoctorCausalLocalizationRequest,
    localize_doctor_cause,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization import (
    CausalEvidence as DoctorCausalEvidence,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization import (
    CausalEvidenceKind as DoctorKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots,
    DoctorSourceUnit,
    FindingKind,
    diagnose_repository,
)
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_evidence import (
    CausalEvidenceAdmissionError,
    CausalEvidenceAuthorityError,
    CausalEvidenceGateway,
    RetrievalNominationBinding,
    admit_exact_evidence,
    admit_exact_from_doctor,
    authoritative_evidence_ids,
    dispose_with_localization,
    doctor_kind_projection_inventory,
    federation_kind_from_doctor,
    nominations_cannot_prove_independence,
    project_doctor_evidence,
    project_retrieval_candidate,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from test.api.causal_federation.test_causal_graph import (
    _edge,
    _node,
    _open_store,
)
from test.api.causal_federation.test_causal_graph import (
    _evidence as _graph_evidence,
)
from test.api.causal_federation.test_contracts import sample_binding, sample_contract


def _snapshot():
    roots = DoctorAuthorityRoots(
        repository_id="repository:casf-014",
        tree_id="tree:casf-014",
        dependency_graph_id="graph:casf-014",
        policy_id="policy:casf-014",
    )
    sources = (
        DoctorSourceUnit(
            path="src/service.py",
            source_bytes=b"def dispatch(payload, context):\n    return payload\n",
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


def _finding(snapshot):
    matches = [item for item in snapshot.findings if item.kind is FindingKind.CALL_ARITY]
    assert len(matches) == 1
    return matches[0]


def _doctor_evidence(
    snapshot,
    finding,
    evidence_id: str,
    kind: DoctorKind | str,
    *,
    causes: tuple[str, ...] = ("cause:dispatch-signature",),
    **kwargs,
) -> DoctorCausalEvidence:
    return DoctorCausalEvidence(
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


def _localized(snapshot, finding, extra: tuple[DoctorCausalEvidence, ...] = ()):
    evidence = (
        _doctor_evidence(snapshot, finding, "evidence:contract", "contract_delta"),
        _doctor_evidence(snapshot, finding, "evidence:call-graph", "call_graph"),
        _doctor_evidence(snapshot, finding, "evidence:dataflow", "dataflow"),
        _doctor_evidence(snapshot, finding, "evidence:runtime", "failing_trace"),
        _doctor_evidence(
            snapshot,
            finding,
            "evidence:delta-debug",
            "delta_debug",
            minimized=True,
        ),
        _doctor_evidence(
            snapshot,
            finding,
            "evidence:unsat-core",
            "unsat_core",
            minimized=True,
        ),
        *extra,
    )
    return localize_doctor_cause(
        DoctorCausalLocalizationRequest(
            snapshot=snapshot,
            finding=finding,
            evidence=evidence,
        )
    ), evidence


def _federation_evidence(
    binding: contracts.FederationBinding,
    *,
    record_id: str,
    kind: contracts.CausalEvidenceKind,
    authoritative: bool,
    evidence_ref: str = "artifact:casf-014",
) -> contracts.CausalEvidence:
    evidence = sample_contract(contracts.CausalEvidence)
    assert isinstance(evidence, contracts.CausalEvidence)
    return replace(
        evidence,
        record_id=record_id,
        binding=binding,
        evidence_kind=kind,
        evidence_ref=evidence_ref,
        authoritative=authoritative,
    )


def test_every_doctor_kind_has_a_closed_federation_projection() -> None:
    inventory = doctor_kind_projection_inventory()
    assert set(inventory) == {kind.value for kind in DoctorKind}
    assert inventory == {
        kind.value: FEDERATION_KIND_FOR_DOCTOR_KIND[kind] for kind in DoctorKind
    }
    for kind in DoctorKind:
        mapped = federation_kind_from_doctor(kind)
        assert mapped.value == inventory[kind.value]
        if kind in {
            DoctorKind.RETRIEVAL,
            DoctorKind.VECTOR_NEAREST,
            DoctorKind.GRAPHRAG,
            DoctorKind.CACHE,
            DoctorKind.MODEL_NOMINATION,
        }:
            assert mapped is contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION


def test_doctor_localization_never_admits_federation_authority() -> None:
    snapshot = _snapshot()
    finding = _finding(snapshot)
    receipt, _evidence = _localized(snapshot, finding)
    assert receipt.federation_authority_admitted is False
    disposed = dispose_with_localization(receipt)
    assert disposed.federation_authority_admitted is False
    assert "evidence:contract" in disposed.exact_evidence_ids
    assert disposed.localization_cid == receipt.localization_cid


def test_retrieval_doctor_fact_projects_as_nomination_only() -> None:
    snapshot = _snapshot()
    finding = _finding(snapshot)
    nomination = _doctor_evidence(
        snapshot,
        finding,
        "evidence:vector",
        "vector_nearest",
    )
    receipt, _evidence = _localized(snapshot, finding, extra=(nomination,))
    binding = sample_binding()
    projected = project_doctor_evidence(
        nomination,
        binding=binding,
        record_id="evidence:projected-vector",
        localization=receipt,
    )
    assert projected.nomination_only is True
    assert projected.evidence.authoritative is False
    assert (
        projected.evidence.evidence_kind
        is contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION
    )
    with pytest.raises(CausalEvidenceAuthorityError, match="only exact doctor facts"):
        admit_exact_from_doctor(
            nomination,
            binding=binding,
            record_id="evidence:forged-authority",
            localization=receipt,
        )


def test_exact_doctor_fact_is_copied_under_federation_rules() -> None:
    snapshot = _snapshot()
    finding = _finding(snapshot)
    receipt, evidence = _localized(snapshot, finding)
    contract_fact = next(item for item in evidence if item.evidence_id == "evidence:contract")
    binding = sample_binding()
    projected = project_doctor_evidence(
        contract_fact,
        binding=binding,
        record_id="evidence:projected-contract",
        localization=receipt,
    )
    assert projected.evidence.authoritative is False
    admitted = admit_exact_from_doctor(
        contract_fact,
        binding=binding,
        record_id="evidence:admitted-contract",
        localization=receipt,
    )
    assert admitted.source_kind == "federation_native"
    assert admitted.evidence.authoritative is True
    assert (
        admitted.evidence.evidence_kind
        is contracts.CausalEvidenceKind.CONTRACT_DEPENDENCY
    )
    assert admitted.localization_cid == receipt.localization_cid


def test_retrieval_candidate_is_bound_and_nomination_only() -> None:
    binding = sample_binding()
    candidate = SimpleNamespace(
        node_id="symbol:similar",
        source="vector",
        score_millionths=640_000,
        index_root_id="index:rev-1",
        binding=SimpleNamespace(graph_id="tree:casf-014", partition_id="partition:a"),
    )
    projected = project_retrieval_candidate(
        candidate,
        binding=binding,
        record_id="evidence:retrieval-hit",
    )
    assert projected.source_kind == "retrieval_candidate"
    assert projected.nomination_only is True
    assert projected.evidence.authoritative is False
    assert projected.retrieval_binding is not None
    assert projected.retrieval_binding.method == "vector"
    assert projected.retrieval_binding.index_revision == "index:rev-1"
    assert projected.retrieval_binding.tree_id == "tree:casf-014"
    with pytest.raises(contracts.FederationAuthorityError):
        replace(projected.evidence, authoritative=True)


def test_unknown_retrieval_method_fails_closed() -> None:
    with pytest.raises(CausalEvidenceAdmissionError, match="method is not closed"):
        RetrievalNominationBinding(
            index_revision="index:rev-1",
            source_cid="symbol:x",
            tree_id="tree:casf-014",
            method="vibes",
            score_millionths=1,
        )


def test_nominations_cannot_prove_independence_or_authorize_edges() -> None:
    binding = sample_binding()
    nomination = _federation_evidence(
        binding,
        record_id="evidence:nom",
        kind=contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION,
        authoritative=False,
    )
    exact = admit_exact_evidence(
        _federation_evidence(
            binding,
            record_id="evidence:exact",
            kind=contracts.CausalEvidenceKind.EXACT_STATIC_DEPENDENCY,
            authoritative=True,
        )
    )
    nominated = project_retrieval_candidate(
        SimpleNamespace(
            node_id="symbol:similar",
            source="bm25",
            score_millionths=10,
            index_root_id="index:rev-1",
            binding=SimpleNamespace(graph_id="tree:test", partition_id=""),
        ),
        binding=binding,
        record_id="evidence:bm25",
    )
    assert authoritative_evidence_ids((exact, nominated)) == ("evidence:exact",)
    with pytest.raises(CausalEvidenceAuthorityError, match="independence"):
        nominations_cannot_prove_independence(
            (nomination.record_id,),
            claimed_independent_ids=("node:unrelated",),
        )
    with pytest.raises(CausalEvidenceAuthorityError, match="cannot be admitted as exact"):
        admit_exact_evidence(nomination)


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for graph admission")
def test_gateway_persists_exact_evidence_and_rejects_nominated_authority(tmp_path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    gateway = CausalEvidenceGateway(store)
    source = _node(
        binding,
        record_id="node:source",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:source",
    )
    target = _node(
        binding,
        record_id="node:target",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:target",
    )
    revision = store.record_node(
        source,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-014",
        source_root="source:casf-014",
        idempotency_key="idempotency:source",
    ).graph_revision
    revision = store.record_node(
        target,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-014",
        source_root="source:casf-014",
        idempotency_key="idempotency:target",
    ).graph_revision
    exact = admit_exact_evidence(
        _graph_evidence(
            binding,
            record_id="evidence:exact",
            evidence_ref="artifact:exact",
        )
    )
    revision = gateway.record(
        exact,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-014",
        source_root="source:casf-014",
        idempotency_key="idempotency:exact",
    ).graph_revision
    nominated = project_retrieval_candidate(
        SimpleNamespace(
            node_id="symbol:similar",
            source="vector",
            score_millionths=12,
            index_root_id="index:rev-1",
            binding=SimpleNamespace(graph_id=binding.repository_tree_ids[0], partition_id="p1"),
        ),
        binding=binding,
        record_id="evidence:vector",
    )
    revision = gateway.record(
        nominated,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-014",
        source_root="source:casf-014",
        idempotency_key="idempotency:vector",
    ).graph_revision
    with pytest.raises(CausalEvidenceAuthorityError, match="cannot authorize"):
        gateway.record_edge(
            _edge(
                binding,
                record_id="edge:forged",
                source_node_id=source.record_id,
                target_node_id=target.record_id,
                evidence_refs=(nominated.evidence.record_id,),
            ),
            (nominated,),
            federation_id=federation_id,
            expected_graph_revision=revision,
            idempotency_key="idempotency:forged-edge",
        )
    committed = gateway.record_edge(
        _edge(
            binding,
            record_id="edge:exact",
            source_node_id=source.record_id,
            target_node_id=target.record_id,
            evidence_refs=(exact.evidence.record_id,),
        ),
        (exact, nominated),
        federation_id=federation_id,
        expected_graph_revision=revision,
        idempotency_key="idempotency:exact-edge",
    )
    snapshot = store.snapshot(tenant_id=binding.tenant_id, federation_id=federation_id)
    assert committed.graph_revision == snapshot.graph_revision
    assert snapshot.edges[0].evidence_refs == (exact.evidence.record_id,)
    assert any(item.authoritative for item in snapshot.evidence)
    assert any(not item.authoritative for item in snapshot.evidence)
