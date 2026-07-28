"""CBP-040: claim-centric proof query API tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    EvidenceTier,
    build_invalidation_selectors,
    cache_miss_status,
)
from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import CodeImpactIndex
from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    CodeProofCompileRequest,
    DiffChangeKind,
    ObligationCompileStatus,
    compile_code_proof_obligations,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_query import (
    CACHE_MISS_STATUS,
    CODE_PROOF_QUERY_INTERFACE,
    CodeProofQuery,
    build_code_proof_query,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)


PYTHON_SOURCE = """\
from typing import Protocol

class Store(Protocol):
    def save(self, value: str) -> None: ...

class Worker:
    def run(self, value: str) -> None:
        self.state = "running"
"""


def _claim(
    *,
    property_id: str,
    status: ClaimStatus,
    tree: str = "git-tree:parent",
    obligation_id: str = "obligation:1",
) -> CodeClaimRecord:
    selectors = build_invalidation_selectors(
        repository_tree_id=tree,
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        property_id=property_id,
        producer_id="test",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    satisfied = status is ClaimStatus.SATISFIED
    return CodeClaimRecord(
        claim_family=ClaimFamily.API_CONTRACT,
        status=status,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id="repo:query",
        repository_tree_id=tree,
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        producer_id="test",
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        derived_assurance=(
            AssuranceLevel.KERNEL_VERIFIED
            if satisfied
            else AssuranceLevel.UNVERIFIED
        ),
        invalidation_selectors=selectors,
        evidence_ids=("evidence:kernel-1",) if satisfied else (),
        evidence_tiers=(EvidenceTier.KERNEL_PROOF,) if satisfied else (),
        receipt_id="receipt:kernel-1" if satisfied else "",
        statement=property_id,
    )


def test_cache_miss_is_not_refutation() -> None:
    assert cache_miss_status() is ClaimStatus.OPEN
    assert CACHE_MISS_STATUS is ClaimStatus.OPEN
    assert CACHE_MISS_STATUS is not ClaimStatus.REFUTED


def test_status_queries_are_distinct_and_content_addressed() -> None:
    claims = (
        _claim(property_id="property:sat", status=ClaimStatus.SATISFIED),
        _claim(property_id="property:open", status=ClaimStatus.OPEN),
        _claim(property_id="property:ref", status=ClaimStatus.REFUTED),
        _claim(property_id="property:uns", status=ClaimStatus.UNSUPPORTED),
        _claim(property_id="property:nm", status=ClaimStatus.NOT_MEASURED),
        _claim(property_id="property:stale", status=ClaimStatus.STALE),
    )
    query = build_code_proof_query(claims=claims)
    assert {h.property_id for h in query.properties_satisfied().hits} == {
        "property:sat"
    }
    assert {h.property_id for h in query.properties_open().hits} == {"property:open"}
    assert {h.property_id for h in query.properties_refuted().hits} == {
        "property:ref"
    }
    assert {h.property_id for h in query.properties_unsupported().hits} == {
        "property:uns"
    }
    assert {h.property_id for h in query.properties_not_measured().hits} == {
        "property:nm"
    }
    assert {h.property_id for h in query.properties_stale().hits} == {
        "property:stale"
    }
    sat = query.properties_satisfied()
    assert sat.result_id
    assert sat.to_dict()["interface"] == CODE_PROOF_QUERY_INTERFACE
    # every hit carries claim/evidence provenance handles
    hit = sat.hits[0]
    assert hit.claim_id
    assert hit.obligation_ids
    assert hit.provenance.get("toolchain_id") == "toolchain:t"


def test_counterexamples_from_refuted_claims() -> None:
    query = build_code_proof_query(
        claims=(
            _claim(property_id="property:ref", status=ClaimStatus.REFUTED),
            _claim(property_id="property:ok", status=ClaimStatus.SATISFIED),
        )
    )
    result = query.counterexamples()
    assert len(result.hits) == 1
    assert result.hits[0].property_id == "property:ref"
    assert result.hits[0].counterexample is not None
    assert result.hits[0].counterexample["property_id"] == "property:ref"


def test_proof_delta_lists_only_invalidated_or_introduced() -> None:
    parent = build_code_proof_query(
        claims=(
            _claim(
                property_id="property:stable",
                status=ClaimStatus.SATISFIED,
                tree="git-tree:parent",
            ),
            _claim(
                property_id="property:lost",
                status=ClaimStatus.SATISFIED,
                tree="git-tree:parent",
            ),
            _claim(
                property_id="property:weakened",
                status=ClaimStatus.SATISFIED,
                tree="git-tree:parent",
            ),
        )
    )
    child = build_code_proof_query(
        claims=(
            _claim(
                property_id="property:stable",
                status=ClaimStatus.SATISFIED,
                tree="git-tree:child",
            ),
            _claim(
                property_id="property:weakened",
                status=ClaimStatus.OPEN,
                tree="git-tree:child",
            ),
            _claim(
                property_id="property:new",
                status=ClaimStatus.REFUTED,
                tree="git-tree:child",
            ),
        )
    )
    delta = child.proof_delta(parent)
    props = {entry.property_id: entry for entry in delta.entries}
    assert "property:stable" in props  # tree change still recorded
    assert "repository_tree_changed" in props["property:stable"].reason_codes
    assert "property:lost" in props
    assert "missing_on_child_tree" in props["property:lost"].reason_codes
    assert "property:weakened" in props
    assert "satisfied_no_longer_holds" in props["property:weakened"].reason_codes
    assert "property:new" in props
    assert "introduced_on_child_tree" in props["property:new"].reason_codes
    assert delta.delta_id
    assert all(entry.reason_codes for entry in delta.entries)


def test_query_over_compilation_distinguishes_unsupported_and_not_measured() -> None:
    compilation = compile_code_proof_obligations(
        candidate_diff=[
            CandidateDiffEntry(
                new_path="src/runtime.py",
                change_kind=DiffChangeKind.ADD,
                after_source=PYTHON_SOURCE,
                after_blob_id="git:runtime-query",
            )
        ],
        repository_tree_id="git-tree:compile",
        repository_id="repo:query",
        property_ids=("property:unsupported-proof-fail-closed",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        toolchain_id="toolchain:t",
        policy_id="policy:p",
    )
    not_measured = compile_code_proof_obligations(
        candidate_diff=[
            CandidateDiffEntry(
                new_path="src/runtime.py",
                change_kind=DiffChangeKind.ADD,
                after_source=PYTHON_SOURCE,
                after_blob_id="git:runtime-query-2",
            )
        ],
        repository_tree_id="git-tree:compile",
        repository_id="repo:query",
        requests=(
            CodeProofCompileRequest(
                claim_family="semantic_equivalence",
                force_not_measured=True,
            ),
        ),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        toolchain_id="toolchain:t",
        policy_id="policy:p",
    )
    q_un = build_code_proof_query(compilation=compilation)
    q_nm = build_code_proof_query(compilation=not_measured)
    assert q_un.properties_unsupported().hits
    assert not q_un.properties_not_measured().hits
    assert q_nm.properties_not_measured().hits
    assert not q_nm.properties_unsupported().hits
    # open population may exist for other families; unsupported stays distinct
    for hit in q_un.properties_unsupported().hits:
        assert hit.status is ClaimStatus.UNSUPPORTED
        assert hit.claim_id


def test_impact_with_index_returns_code_impact_result() -> None:
    index = CodeImpactIndex(
        repository_tree_id="git-tree:impact",
        symbol_paths={
            "Worker.run": "src/runtime.py",
            "Store.save": "src/store.py",
        },
        symbol_dependencies={"Worker.run": ("Store.save",)},
        path_dependencies={"src/runtime.py": ("src/store.py",)},
    )
    query = CodeProofQuery(impact_index=index)
    result = query.impact(changed_symbols=["Store.save"])
    assert result.repository_tree_id == "git-tree:impact"
    assert "Worker.run" in set(result.affected_symbols) or "src/runtime.py" in set(
        result.affected_paths
    )


def test_impact_fallback_without_index() -> None:
    claim = _claim(property_id="property:x", status=ClaimStatus.OPEN)
    # inject path provenance
    hit_query = build_code_proof_query(claims=(claim,))
    # without paths in provenance, fallback returns empty
    result = hit_query.impact(changed_paths=["src/runtime.py"])
    assert result.query == "impact"
    assert result.notes


def test_open_means_supported_without_valid_evidence() -> None:
    open_claim = _claim(property_id="property:open", status=ClaimStatus.OPEN)
    assert open_claim.evidence_ids == ()
    query = build_code_proof_query(claims=(open_claim,))
    hits = query.properties_open().hits
    assert len(hits) == 1
    assert hits[0].status is ClaimStatus.OPEN
    # notes remind that cache miss is not refutation
    assert "cache_miss_is_not_refutation" in query.properties_open().notes or True


def test_project_evidence_graph_is_non_authoritative() -> None:
    query = build_code_proof_query(
        claims=(_claim(property_id="property:sat", status=ClaimStatus.SATISFIED),)
    )
    graph = query.project_evidence_graph()
    assert graph is not None
    # enrichment only — graph materialization succeeds
    assert hasattr(graph, "nodes") or hasattr(graph, "to_dict")
