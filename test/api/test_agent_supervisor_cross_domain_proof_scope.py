from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof_scope_index import (
    ArtifactActivityState,
    ProofInputKind,
    ProofScopeIndex,
    ProofScopeIndexError,
    build_cross_domain_proof_scope_index,
    build_proof_scope_index,
)
from ipfs_accelerate_py.agent_supervisor.semantic_dependency_graph import (
    SemanticAuthority,
    SemanticDependencyGraph,
    SemanticEdge,
    SemanticEdgeKind,
    SemanticNode,
    SemanticNodeKind,
    SemanticProvenance,
    SemanticTrust,
)


ROOT = "decision-root:cross-domain-fixture"


def _artifact_snapshot() -> dict[str, object]:
    context_keys = [
        {"kind": kind.value, "value": f"{kind.value}:fixture"}
        for kind in (
            ProofInputKind.IR_FAMILY,
            ProofInputKind.IR_ROOT,
            ProofInputKind.IR_DECLARATION,
            ProofInputKind.IR_CLAIM,
            ProofInputKind.INTENT_ACTION,
            ProofInputKind.INTENT_STATEMENT,
            ProofInputKind.LEGAL_NORM,
            ProofInputKind.LEGAL_APPLICABILITY_FACT,
            ProofInputKind.SECURITY_PRINCIPAL,
            ProofInputKind.SECURITY_RESOURCE,
            ProofInputKind.SECURITY_POLICY,
            ProofInputKind.SECURITY_STATE,
            ProofInputKind.PROGRAM_SNAPSHOT,
            ProofInputKind.AST_EDGE,
            ProofInputKind.EFFECT,
            ProofInputKind.TOOL_OPERATION,
            ProofInputKind.DECISION_CONTEXT,
            ProofInputKind.AUTHORIZATION_DECISION,
            ProofInputKind.EXECUTION_PERMIT,
        )
    ]
    return {
        "contexts": (
            {
                "context_id": "context:affected",
                "root_id": ROOT,
                "scope_keys": context_keys,
                "payload": {"revision": "context-v1"},
            },
            {
                "context_id": "context:independent",
                "root_id": ROOT,
                "scope_keys": [
                    {
                        "kind": "decision_context",
                        "value": "decision_context:independent",
                    }
                ],
                "payload": {"revision": "independent-v1"},
            },
        ),
        "plans": (
            {
                "plan_id": "plan:affected",
                "root_id": ROOT,
                "depends_on": ["context:affected"],
                "payload": {"revision": "plan-v1"},
            },
            {
                "plan_id": "plan:independent",
                "root_id": ROOT,
                "depends_on": ["context:independent"],
                "payload": {"revision": "independent-plan-v1"},
            },
        ),
        "obligations": (
            {
                "artifact_id": "obligation:affected",
                "root_id": ROOT,
                "depends_on": ["plan:affected"],
                "payload": {"revision": "obligation-v1"},
            },
        ),
        "proofs": (
            {
                "proof_id": "proof:affected",
                "root_id": ROOT,
                "depends_on": ["obligation:affected"],
                "payload": {"revision": "proof-v1"},
            },
        ),
        "permits": (
            {
                "permit_id": "permit:affected",
                "root_id": ROOT,
                "depends_on": ["proof:affected"],
                "payload": {"revision": "permit-v1"},
            },
        ),
        "validations": (
            {
                "validation_id": "validation:affected",
                "root_id": ROOT,
                "depends_on": ["proof:affected", "permit:affected"],
                "payload": {"revision": "validation-v1"},
            },
        ),
        "caches": (
            {
                "cache_id": "cache:affected",
                "root_id": ROOT,
                "depends_on": ["validation:affected"],
                "payload": {"revision": "cache-v1"},
            },
        ),
        "merges": (
            {
                "merge_id": "merge:affected",
                "root_id": ROOT,
                "depends_on": ["validation:affected"],
                "payload": {"revision": "merge-v1"},
            },
        ),
    }


def _build(*, previous: ProofScopeIndex | dict[str, object] | None = None) -> ProofScopeIndex:
    return build_cross_domain_proof_scope_index(
        root_id=ROOT,
        previous=previous,
        **_artifact_snapshot(),
    )


def test_all_cross_domain_scope_keys_project_forward_and_reverse_dependents() -> None:
    index = _build()
    expected = {
        "context:affected",
        "plan:affected",
        "obligation:affected",
        "proof:affected",
        "permit:affected",
        "validation:affected",
        "cache:affected",
        "merge:affected",
    }

    for kind in ProofInputKind:
        if kind.value not in {
            item.value
            for item in (
                ProofInputKind.IR_FAMILY,
                ProofInputKind.IR_ROOT,
                ProofInputKind.IR_DECLARATION,
                ProofInputKind.IR_CLAIM,
                ProofInputKind.INTENT_ACTION,
                ProofInputKind.INTENT_STATEMENT,
                ProofInputKind.LEGAL_NORM,
                ProofInputKind.LEGAL_APPLICABILITY_FACT,
                ProofInputKind.SECURITY_PRINCIPAL,
                ProofInputKind.SECURITY_RESOURCE,
                ProofInputKind.SECURITY_POLICY,
                ProofInputKind.SECURITY_STATE,
                ProofInputKind.PROGRAM_SNAPSHOT,
                ProofInputKind.AST_EDGE,
                ProofInputKind.EFFECT,
                ProofInputKind.TOOL_OPERATION,
                ProofInputKind.DECISION_CONTEXT,
                ProofInputKind.AUTHORIZATION_DECISION,
                ProofInputKind.EXECUTION_PERMIT,
            )
        }:
            continue
        dependents = index.dependents(kind, f"{kind.value}:fixture")
        assert set(dependents.artifact_ids) == expected
        assert dependents.obligation_ids == ("obligation:affected",)
        assert dependents.context_ids == ("context:affected",)
        assert dependents.plan_ids == ("plan:affected",)
        assert dependents.proof_ids == ("proof:affected",)
        assert dependents.permit_ids == ("permit:affected",)
        assert dependents.validation_ids == ("validation:affected",)
        assert dependents.cache_ids == ("cache:affected",)
        assert dependents.merge_ids == ("merge:affected",)


def test_semantic_change_invalidates_exact_transitive_closure_and_is_idempotent() -> None:
    index = _build()
    changed = index.invalidate(
        [("intent_action", "intent_action:fixture")],
        max_reason_chain=5,
    )

    assert set(changed.stale_artifact_ids) == {
        "context:affected",
        "plan:affected",
        "obligation:affected",
        "proof:affected",
        "permit:affected",
        "validation:affected",
        "cache:affected",
        "merge:affected",
    }
    assert set(changed.active_artifact_ids) == {
        "context:independent",
        "plan:independent",
    }
    assert changed.artifact_states["merge:affected"] is ArtifactActivityState.STALE
    assert (
        changed.artifact_states["plan:independent"]
        is ArtifactActivityState.ACTIVE
    )
    assert changed.invalidate(
        ["intent_action:intent_action:fixture"],
        max_reason_chain=5,
    ) == changed
    assert (
        changed.dependents(
            "decision_context",
            "decision_context:independent",
            active_only=True,
        ).artifact_ids
        == ("context:independent", "plan:independent")
    )


def test_exact_warm_reuse_and_restart_require_current_canonical_artifacts() -> None:
    cold = _build()
    warm = _build(previous=cold)

    assert warm.stats.reused_artifact_count == len(cold.artifacts)
    assert warm.artifacts == cold.artifacts
    assert all(
        warm_artifact is cold_artifact
        for warm_artifact, cold_artifact in zip(warm.artifacts, cold.artifacts)
    )
    payload = warm.to_json()
    with pytest.raises(ProofScopeIndexError, match="current canonical"):
        ProofScopeIndex.from_json(payload)
    restored = ProofScopeIndex.from_json(
        payload,
        canonical_artifacts=warm.artifacts,
    )
    assert restored == warm

    current = list(warm.artifacts)
    current.pop()
    with pytest.raises(ProofScopeIndexError, match="does not match"):
        ProofScopeIndex.from_json(payload, canonical_artifacts=current)

    forged = json.loads(payload)
    forged["artifact_states"]["merge:affected"] = "active" if (
        forged["artifact_states"]["merge:affected"] == "stale"
    ) else "stale"
    forged.pop("index_id")
    with pytest.raises(ProofScopeIndexError, match="forged"):
        ProofScopeIndex.from_dict(forged, canonical_artifacts=warm.artifacts)


def test_changed_canonical_input_stales_only_unchanged_downstream_products() -> None:
    cold = _build()
    changed_snapshot = _artifact_snapshot()
    contexts = [dict(item) for item in changed_snapshot["contexts"]]
    contexts[0] = {
        **contexts[0],
        "payload": {"revision": "context-v2"},
    }
    changed_snapshot["contexts"] = tuple(contexts)

    rebuilt = build_cross_domain_proof_scope_index(
        root_id=ROOT,
        previous=cold,
        **changed_snapshot,
    )

    assert set(rebuilt.stale_artifact_ids) == {
        "plan:affected",
        "obligation:affected",
        "proof:affected",
        "permit:affected",
        "validation:affected",
        "cache:affected",
        "merge:affected",
    }
    assert set(rebuilt.active_artifact_ids) == {
        "context:affected",
        "context:independent",
        "plan:independent",
    }
    assert rebuilt.is_obligation_active("obligation:affected") is False


def test_cross_domain_graph_rejects_cycles_detachment_roots_aliases_and_activity() -> None:
    base = {
        "root_id": ROOT,
        "artifact_kind": "context",
        "scope_keys": [{"kind": "decision_context", "value": "decision:one"}],
    }
    invalid = (
        (
            [
                {**base, "artifact_id": "a", "depends_on": ["b"]},
                {**base, "artifact_id": "b", "depends_on": ["a"]},
            ],
            "cycle",
        ),
        ([{**base, "artifact_id": "a", "depends_on": ["missing"]}], "detached"),
        ([{**base, "artifact_id": "a", "root_id": "foreign"}], "foreign root"),
        (
            [
                {
                    **base,
                    "artifact_id": "a",
                    "node_id": "different",
                }
            ],
            "ambiguous",
        ),
        ([{**base, "artifact_id": "a", "active": True}], "computed"),
    )
    for artifacts, message in invalid:
        with pytest.raises(ProofScopeIndexError, match=message):
            build_cross_domain_proof_scope_index(
                root_id=ROOT,
                artifacts=artifacts,
            )


def test_detached_legacy_receipts_are_rejected_not_merely_marked_stale() -> None:
    with pytest.raises(ProofScopeIndexError, match="detached receipt"):
        build_proof_scope_index(
            receipts=(
                {
                    "receipt_id": "receipt:detached",
                    "obligation_id": "obligation:missing",
                },
            )
        )


def test_semantic_graph_projection_reverses_proof_evidence_dependencies() -> None:
    common = {
        "root_id": ROOT,
        "trust": SemanticTrust.VERIFIED,
        "authority": SemanticAuthority.VERIFIED_INPUT,
        "version": "fixture@1",
    }
    obligation = SemanticNode(
        node_id="legal:obligation",
        kind=SemanticNodeKind.LEGAL_OBLIGATION,
        source_root_id="legal:root",
        provenance=SemanticProvenance.LEGAL_IR,
        provenance_id="legal:artifact",
        record={
            "family": "legal_ir",
            "node_kind": "declaration",
            "declaration_kind": "obligation",
        },
        **common,
    )
    proof = SemanticNode(
        node_id="proof:legal",
        kind=SemanticNodeKind.PROOF,
        source_root_id="legal:root",
        provenance=SemanticProvenance.PROOF,
        provenance_id="proof:receipt",
        record={"verdict": "proved"},
        **common,
    )
    edge = SemanticEdge(
        source=obligation.node_id,
        target=proof.node_id,
        kind=SemanticEdgeKind.PROVEN_BY,
        root_id=ROOT,
        source_root_id="legal:root",
        provenance=SemanticProvenance.PROOF,
        provenance_id="edge:proof",
        trust=SemanticTrust.VERIFIED,
        authority=SemanticAuthority.VERIFIED_INPUT,
        version="fixture@1",
    )
    graph = SemanticDependencyGraph(ROOT, (obligation, proof), (edge,))

    index = build_cross_domain_proof_scope_index(
        root_id=ROOT,
        semantic_graph=graph,
    )
    dependents = index.dependents("legal_norm", "legal:obligation")
    assert dependents.obligation_ids == ("legal:obligation",)
    assert dependents.proof_ids == ("proof:legal",)
    assert set(
        index.invalidate(
            ["legal_norm:legal:obligation"]
        ).stale_artifact_ids
    ) == {"legal:obligation", "proof:legal"}
