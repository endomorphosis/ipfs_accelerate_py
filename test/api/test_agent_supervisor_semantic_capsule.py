"""FACP-047: content-addressed semantic capsules and invalidation soundness."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.semantic_capsule import (
    ANALYZER_VERSION,
    BUNDLE,
    EVIDENCE_FIELDS,
    EVIDENCE_SCHEMA,
    GOAL_ID,
    HERMETIC_EVALUATOR_ID,
    INDEX_SCHEMA,
    INVALIDATION_SCHEMA,
    SCHEMA,
    TASK_ID,
    CapsuleAction,
    CapsuleKind,
    CapsuleRuleId,
    DatalogAtom,
    HermeticReferenceEvaluator,
    HistoricalReceipt,
    ReceiptKind,
    ReceiptStatus,
    SemanticCapsuleError,
    SemanticCapsuleRecord,
    StaleReceiptError,
    UnknownDependencyError,
    assert_reuse_allowed,
    compile_semantic_capsule,
    compile_semantic_capsules,
    default_capsule_datalog_rules,
    demote_stale_receipts,
    explain_path,
    invalidate_capsules,
    project_capsule_kind,
    require_live_receipt,
    update_semantic_capsules,
    verify_capsule_compile_result,
)
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph import (
    SemanticAuthority,
    SemanticDependencyGraph,
    SemanticEdge,
    SemanticEdgeKind,
    SemanticNode,
    SemanticNodeKind,
    SemanticProvenance,
    SemanticTrust,
    build_semantic_dependency_graph,
)


ROOT = "decision-root:sha256:facp047-fixture"
VERSION = "facp047-producer@1"


def _node(
    node_id: str,
    kind: SemanticNodeKind,
    *,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
    trust: SemanticTrust = SemanticTrust.VERIFIED,
    authority: SemanticAuthority = SemanticAuthority.AUTHORITATIVE,
    root_id: str = ROOT,
    record: dict | None = None,
) -> SemanticNode:
    return SemanticNode(
        node_id=node_id,
        kind=kind,
        root_id=root_id,
        source_root_id=f"source:{node_id}",
        provenance=provenance,
        provenance_id=f"record:{node_id}",
        trust=trust,
        authority=authority,
        version=VERSION,
        record=record or {"id": node_id},
    )


def _edge(
    source: str,
    target: str,
    kind: SemanticEdgeKind,
    *,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
    authority: SemanticAuthority = SemanticAuthority.AUTHORITATIVE,
    trust: SemanticTrust = SemanticTrust.VERIFIED,
    mandatory: bool = True,
    root_id: str = ROOT,
) -> SemanticEdge:
    return SemanticEdge(
        source=source,
        target=target,
        kind=kind,
        root_id=root_id,
        provenance=provenance,
        provenance_id=f"edge-record:{source}:{kind.value}:{target}",
        trust=trust,
        authority=authority,
        version=VERSION,
        mandatory=mandatory,
    )


def _fixture_graph(*, symbol_payload: str = "alpha") -> SemanticDependencyGraph:
    """Connected graph: symbol -> obligation -> proof/test/release + unrelated."""

    nodes = (
        _node(
            "symbol.main",
            SemanticNodeKind.SYMBOL,
            record={"id": "symbol.main", "payload": symbol_payload},
        ),
        _node("contract.main", SemanticNodeKind.OBLIGATION),
        _node(
            "proof.main",
            SemanticNodeKind.PROOF,
            provenance=SemanticProvenance.PROOF,
        ),
        _node(
            "test.main",
            SemanticNodeKind.MONITOR,
            provenance=SemanticProvenance.MONITOR,
        ),
        _node(
            "release.main",
            SemanticNodeKind.MERGE_EVIDENCE,
            provenance=SemanticProvenance.MERGE,
        ),
        _node("effect.main", SemanticNodeKind.EFFECT),
        _node("policy.main", SemanticNodeKind.SECURITY_POLICY),
        _node("env.main", SemanticNodeKind.ENVIRONMENT),
        _node(
            "symbol.unrelated",
            SemanticNodeKind.SYMBOL,
            record={"id": "symbol.unrelated", "payload": "other"},
        ),
        _node(
            "contract.unrelated",
            SemanticNodeKind.OBLIGATION,
        ),
        _node(
            "proof.unrelated",
            SemanticNodeKind.PROOF,
            provenance=SemanticProvenance.PROOF,
        ),
        _node(
            "annotation.noise",
            SemanticNodeKind.ANNOTATION,
            provenance=SemanticProvenance.GRAPHRAG,
            trust=SemanticTrust.UNTRUSTED,
            authority=SemanticAuthority.PROPOSAL_ONLY,
        ),
    )
    edges = (
        _edge("contract.main", "symbol.main", SemanticEdgeKind.DEPENDS_ON),
        _edge(
            "contract.main",
            "proof.main",
            SemanticEdgeKind.PROVEN_BY,
            provenance=SemanticProvenance.PROOF,
        ),
        _edge(
            "contract.main",
            "test.main",
            SemanticEdgeKind.MONITORED_BY,
            provenance=SemanticProvenance.MONITOR,
        ),
        _edge("release.main", "contract.main", SemanticEdgeKind.REQUIRES),
        _edge("release.main", "proof.main", SemanticEdgeKind.REQUIRES),
        _edge("release.main", "test.main", SemanticEdgeKind.REQUIRES),
        _edge("effect.main", "symbol.main", SemanticEdgeKind.DEPENDS_ON),
        _edge("policy.main", "symbol.main", SemanticEdgeKind.CONSTRAINED_BY),
        _edge("contract.main", "env.main", SemanticEdgeKind.REQUIRES),
        _edge(
            "contract.unrelated",
            "symbol.unrelated",
            SemanticEdgeKind.DEPENDS_ON,
        ),
        _edge(
            "contract.unrelated",
            "proof.unrelated",
            SemanticEdgeKind.PROVEN_BY,
            provenance=SemanticProvenance.PROOF,
        ),
        # Non-authoritative annotation edge must not enter capsules.
        _edge(
            "contract.main",
            "annotation.noise",
            SemanticEdgeKind.DEPENDS_ON,
            provenance=SemanticProvenance.GRAPHRAG,
            trust=SemanticTrust.UNTRUSTED,
            authority=SemanticAuthority.PROPOSAL_ONLY,
            mandatory=False,
        ),
    )
    return build_semantic_dependency_graph(root_id=ROOT, nodes=nodes, edges=edges)


def test_schema_constants_and_evidence_envelope() -> None:
    result = compile_semantic_capsules(_fixture_graph())
    payload = result.to_dict()
    assert SCHEMA == "facp/semantic-capsule@1"
    assert INVALIDATION_SCHEMA == "facp/invalidation-soundness@1"
    assert INDEX_SCHEMA == "facp/semantic-capsule-index@1"
    assert EVIDENCE_SCHEMA == SCHEMA
    assert TASK_ID == "FACP-047"
    assert GOAL_ID == "FACP-G610"
    assert BUNDLE == "facp/incremental/capsules"
    assert ANALYZER_VERSION == "semantic-capsule/v1"
    assert HERMETIC_EVALUATOR_ID.endswith("hermetic_reference_evaluator/v1")
    assert payload["schema"] == SCHEMA
    assert payload["evidence_schema"] == EVIDENCE_SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["goal_id"] == GOAL_ID
    assert payload["bundle"] == BUNDLE
    assert payload["analyzer_version"] == ANALYZER_VERSION
    assert set(EVIDENCE_FIELDS) == {
        "exports",
        "requires",
        "effects",
        "authority",
        "abstract_state",
        "assumptions",
        "guarantees",
        "proofs",
        "tests",
        "public_data",
        "environment",
        "source_cids",
    }


def test_capsule_kinds_cover_g610_surface() -> None:
    graph = _fixture_graph()
    result = compile_semantic_capsules(graph)
    kinds = {item.kind for item in result.capsules}
    assert kinds >= {
        CapsuleKind.SYMBOL,
        CapsuleKind.CONTRACT,
        CapsuleKind.EFFECT,
        CapsuleKind.POLICY,
        CapsuleKind.PROOF,
        CapsuleKind.TEST,
        CapsuleKind.ENVIRONMENT,
        CapsuleKind.RELEASE,
    }
    # Projection map covers each G610 surface from graph kinds.
    samples = {
        SemanticNodeKind.SYMBOL: CapsuleKind.SYMBOL,
        SemanticNodeKind.OBLIGATION: CapsuleKind.CONTRACT,
        SemanticNodeKind.EFFECT: CapsuleKind.EFFECT,
        SemanticNodeKind.SECURITY_POLICY: CapsuleKind.POLICY,
        SemanticNodeKind.PROOF: CapsuleKind.PROOF,
        SemanticNodeKind.MONITOR: CapsuleKind.TEST,
        SemanticNodeKind.ENVIRONMENT: CapsuleKind.ENVIRONMENT,
        SemanticNodeKind.MERGE_EVIDENCE: CapsuleKind.RELEASE,
    }
    for node_kind, expected in samples.items():
        node = _node(f"sample.{node_kind.value}", node_kind)
        if node_kind is SemanticNodeKind.PROOF:
            node = _node(
                f"sample.{node_kind.value}",
                node_kind,
                provenance=SemanticProvenance.PROOF,
            )
        assert project_capsule_kind(node) is expected


def test_identity_is_content_addressed_and_order_independent() -> None:
    graph = _fixture_graph()
    a = compile_semantic_capsules(graph)
    # Rebuild with shuffled iteration order via reverse tuples.
    nodes = tuple(reversed(graph.nodes))
    edges = tuple(reversed(graph.edges))
    shuffled = SemanticDependencyGraph(
        root_id=graph.root_id, nodes=nodes, edges=edges
    )
    b = compile_semantic_capsules(shuffled)
    assert a.index.index_cid == b.index.index_cid
    assert [c.capsule_cid for c in a.capsules] == [
        c.capsule_cid for c in b.capsules
    ]
    assert shuffled.graph_id == graph.graph_id


def test_clean_rebuild_equals_incremental_update() -> None:
    graph = _fixture_graph()
    cold = compile_semantic_capsules(graph)
    incremental = update_semantic_capsules(cold, graph)
    assert cold.index.index_cid == incremental.index.index_cid
    assert [c.to_dict() for c in cold.capsules] == [
        c.to_dict() for c in incremental.capsules
    ]
    # Every current CID is reused when the graph is unchanged.
    assert set(incremental.reused_cids) == {
        c.capsule_cid for c in incremental.capsules
    }
    assert incremental.invalidated_cids == ()


def test_seeded_symbol_change_invalidates_required_proof_test_release_only() -> None:
    baseline_graph = _fixture_graph(symbol_payload="alpha")
    baseline = compile_semantic_capsules(baseline_graph)
    changed_graph = _fixture_graph(symbol_payload="beta")
    updated = update_semantic_capsules(baseline, changed_graph)

    by_node_before = baseline.index.by_node_id()
    by_node_after = updated.index.by_node_id()

    assert (
        by_node_before["symbol.main"].capsule_cid
        != by_node_after["symbol.main"].capsule_cid
    )
    # Dependents that bind the changed symbol (or its dependents) change CID.
    assert (
        by_node_before["contract.main"].capsule_cid
        != by_node_after["contract.main"].capsule_cid
    )
    assert (
        by_node_before["release.main"].capsule_cid
        != by_node_after["release.main"].capsule_cid
    )

    # Unrelated subgraph must keep identical capsule CIDs and stay reusable.
    assert (
        by_node_before["symbol.unrelated"].capsule_cid
        == by_node_after["symbol.unrelated"].capsule_cid
    )
    assert (
        by_node_before["proof.unrelated"].capsule_cid
        == by_node_after["proof.unrelated"].capsule_cid
    )
    assert by_node_after["symbol.unrelated"].capsule_cid in updated.reused_cids
    assert by_node_after["proof.unrelated"].capsule_cid in updated.reused_cids

    # Every required validation on the affected path is invalidated for reuse,
    # even when the proof/test capsule bytes themselves are unchanged.
    invalidated_subjects = {
        item.subject_cid
        for item in updated.explanations
        if item.action is CapsuleAction.INVALIDATE
    }
    for node_id in (
        "symbol.main",
        "contract.main",
        "proof.main",
        "test.main",
        "release.main",
    ):
        cid = by_node_after[node_id].capsule_cid
        assert cid in invalidated_subjects
        assert cid not in updated.reused_cids
        assert (
            cid in updated.invalidated_cids
            or by_node_before[node_id].capsule_cid in updated.invalidated_cids
        )


def test_effect_and_policy_changes_invalidate_downstream() -> None:
    base = _fixture_graph()
    baseline = compile_semantic_capsules(base)

    nodes = []
    for node in base.nodes:
        if node.node_id == "effect.main":
            nodes.append(
                _node(
                    "effect.main",
                    SemanticNodeKind.EFFECT,
                    record={"id": "effect.main", "payload": "mutated-effect"},
                )
            )
        elif node.node_id == "policy.main":
            nodes.append(
                _node(
                    "policy.main",
                    SemanticNodeKind.SECURITY_POLICY,
                    record={"id": "policy.main", "payload": "mutated-policy"},
                )
            )
        else:
            nodes.append(node)
    mutated = build_semantic_dependency_graph(
        root_id=ROOT, nodes=nodes, edges=base.edges
    )
    updated = update_semantic_capsules(baseline, mutated)
    before = baseline.index.by_node_id()
    after = updated.index.by_node_id()
    assert before["effect.main"].capsule_cid != after["effect.main"].capsule_cid
    assert before["policy.main"].capsule_cid != after["policy.main"].capsule_cid
    # Unrelated remains stable.
    assert (
        before["symbol.unrelated"].capsule_cid
        == after["symbol.unrelated"].capsule_cid
    )


def test_environment_and_source_cid_participate() -> None:
    base = _fixture_graph()
    baseline = compile_semantic_capsules(base)
    nodes = []
    for node in base.nodes:
        if node.node_id == "env.main":
            nodes.append(
                _node(
                    "env.main",
                    SemanticNodeKind.ENVIRONMENT,
                    record={
                        "id": "env.main",
                        "source_cid": "source-cid:mutated-env",
                    },
                )
            )
        else:
            nodes.append(node)
    mutated = build_semantic_dependency_graph(
        root_id=ROOT, nodes=nodes, edges=base.edges
    )
    updated = update_semantic_capsules(baseline, mutated)
    before = baseline.index.by_node_id()
    after = updated.index.by_node_id()
    assert before["env.main"].capsule_cid != after["env.main"].capsule_cid
    assert "source-cid:mutated-env" in after["env.main"].evidence.source_cids
    # Contract depends on env, so its capsule CID must also change.
    assert before["contract.main"].capsule_cid != after["contract.main"].capsule_cid


def test_unrelated_capsule_not_invalidated() -> None:
    baseline = compile_semantic_capsules(_fixture_graph(symbol_payload="a"))
    updated = update_semantic_capsules(
        baseline, _fixture_graph(symbol_payload="b")
    )
    before = baseline.index.by_node_id()
    after = updated.index.by_node_id()
    assert before["proof.unrelated"].capsule_cid == after["proof.unrelated"].capsule_cid
    reuse = explain_path(updated, after["proof.unrelated"].capsule_cid)
    assert reuse.action is CapsuleAction.REUSE
    assert reuse.rule_id == CapsuleRuleId.INPUTS_UNCHANGED.value


def test_every_reuse_and_invalidation_has_minimal_path() -> None:
    baseline = compile_semantic_capsules(_fixture_graph(symbol_payload="a"))
    updated = update_semantic_capsules(
        baseline, _fixture_graph(symbol_payload="b")
    )
    subjects = {c.capsule_cid for c in updated.capsules}
    explained = {
        item.subject_cid
        for item in updated.explanations
        if item.action in {CapsuleAction.REUSE, CapsuleAction.INVALIDATE}
    }
    # Every capsule that was reused or invalidated has an explanation.
    for cid in subjects:
        if cid in updated.reused_cids or cid in {
            item.subject_cid
            for item in updated.explanations
            if item.action is CapsuleAction.INVALIDATE
        }:
            explanation = explain_path(updated, cid)
            assert explanation.path.capsule_cids[0] == explanation.seed_cid
            assert explanation.path.capsule_cids[-1] == explanation.subject_cid
            assert len(explanation.path.capsule_cids) == len(
                set(explanation.path.capsule_cids)
            )
    assert explained


def test_unknown_dependency_refuses_reuse() -> None:
    # Mandatory edge from a capsule subject to an unmapped RESOURCE node is an
    # unknown dependency for reuse (RESOURCE does not project to a capsule).
    nodes = (
        _node("symbol.x", SemanticNodeKind.SYMBOL),
        _node("contract.x", SemanticNodeKind.OBLIGATION),
        _node(
            "proof.x",
            SemanticNodeKind.PROOF,
            provenance=SemanticProvenance.PROOF,
        ),
        _node("resource.hidden", SemanticNodeKind.RESOURCE),
    )
    edges = (
        _edge("contract.x", "symbol.x", SemanticEdgeKind.DEPENDS_ON),
        _edge(
            "contract.x",
            "proof.x",
            SemanticEdgeKind.PROVEN_BY,
            provenance=SemanticProvenance.PROOF,
        ),
        _edge("contract.x", "resource.hidden", SemanticEdgeKind.REQUIRES),
    )
    graph = build_semantic_dependency_graph(root_id=ROOT, nodes=nodes, edges=edges)
    result = compile_semantic_capsules(graph)
    contract = result.index.by_node_id()["contract.x"]
    assert contract.has_unknown_dependency
    assert "resource.hidden" in contract.unknown_dependency_refs
    with pytest.raises(UnknownDependencyError):
        assert_reuse_allowed(contract)


def test_stale_historical_receipts_demote() -> None:
    baseline_graph = _fixture_graph(symbol_payload="alpha")
    baseline = compile_semantic_capsules(baseline_graph)
    proof_cid = baseline.index.by_node_id()["proof.main"].capsule_cid
    test_cid = baseline.index.by_node_id()["test.main"].capsule_cid
    release_cid = baseline.index.by_node_id()["release.main"].capsule_cid
    receipts = (
        HistoricalReceipt(
            receipt_id="receipt.proof.1",
            kind=ReceiptKind.PROOF,
            bound_capsule_cid=proof_cid,
            produced_graph_id=baseline.index.graph_id,
        ),
        HistoricalReceipt(
            receipt_id="receipt.test.1",
            kind=ReceiptKind.TEST,
            bound_capsule_cid=test_cid,
            produced_graph_id=baseline.index.graph_id,
        ),
        HistoricalReceipt(
            receipt_id="receipt.release.1",
            kind=ReceiptKind.RELEASE,
            bound_capsule_cid=release_cid,
            produced_graph_id=baseline.index.graph_id,
        ),
    )
    updated = update_semantic_capsules(
        baseline,
        _fixture_graph(symbol_payload="beta"),
        receipts=receipts,
    )
    demoted_ids = {item.receipt_id for item in updated.demoted_receipts}
    assert demoted_ids == {
        "receipt.proof.1",
        "receipt.test.1",
        "receipt.release.1",
    }
    for item in updated.demoted_receipts:
        assert item.status is ReceiptStatus.DEMOTED
        with pytest.raises(StaleReceiptError):
            require_live_receipt(item)

    # Matching current receipts stay live.
    fresh_proof = updated.index.by_node_id()["proof.main"].capsule_cid
    live = demote_stale_receipts(
        updated.index,
        [
            HistoricalReceipt(
                receipt_id="receipt.proof.fresh",
                kind=ReceiptKind.PROOF,
                bound_capsule_cid=fresh_proof,
                produced_graph_id=updated.index.graph_id,
            )
        ],
    )
    assert live[0].status is ReceiptStatus.LIVE


def test_hermetic_datalog_derives_invalidation() -> None:
    facts = (
        DatalogAtom("Changed", ("seed",)),
        DatalogAtom("DependsOn", ("proof", "seed")),
        DatalogAtom("DependsOn", ("release", "proof")),
        DatalogAtom("Capsule", ("seed", "symbol")),
        DatalogAtom("Capsule", ("proof", "proof")),
        DatalogAtom("Capsule", ("release", "release")),
        DatalogAtom("RequiredKind", ("proof",)),
        DatalogAtom("RequiredKind", ("release",)),
        DatalogAtom("BindsReceipt", ("r1", "proof")),
    )
    evaluation = HermeticReferenceEvaluator().evaluate(
        facts, default_capsule_datalog_rules()
    )
    invalidated = {row[1] for row in evaluation.facts("Invalidates")}
    assert "seed" in invalidated
    assert "proof" in invalidated
    assert "release" in invalidated
    revalidation = evaluation.facts("RequiresRevalidation")
    assert ("proof", "proof") in revalidation
    assert ("release", "release") in revalidation
    demote = evaluation.facts("Demote")
    assert any(row[0] == "r1" for row in demote)
    assert CapsuleRuleId.TRANSITIVE_INVALIDATION.value in evaluation.derived_rule_ids


def test_proposal_annotations_never_become_capsules() -> None:
    result = compile_semantic_capsules(_fixture_graph())
    node_ids = {item.node_id for item in result.capsules}
    assert "annotation.noise" not in node_ids
    annotation = _node(
        "annotation.noise",
        SemanticNodeKind.ANNOTATION,
        provenance=SemanticProvenance.MODEL,
        trust=SemanticTrust.UNTRUSTED,
        authority=SemanticAuthority.PROPOSAL_ONLY,
    )
    assert project_capsule_kind(annotation) is None


def test_forged_capsule_cid_and_schema_fail_closed() -> None:
    capsule = compile_semantic_capsule(_fixture_graph(), "symbol.main")
    payload = capsule.to_dict()
    payload["capsule_cid"] = "semantic-capsule:sha256:" + ("0" * 64)
    with pytest.raises(SemanticCapsuleError, match="identity mismatch"):
        SemanticCapsuleRecord.from_dict(payload)
    payload = capsule.to_dict()
    payload["schema"] = "facp/semantic-capsule@999"
    with pytest.raises(SemanticCapsuleError, match="unsupported capsule schema"):
        SemanticCapsuleRecord.from_dict(payload)


def test_raw_source_not_embedded() -> None:
    result = compile_semantic_capsules(_fixture_graph())
    for capsule in result.capsules:
        payload = capsule.identity_payload()
        blob = str(payload)
        assert "def " not in blob
        assert "class " not in blob
        assert "/home/" not in blob
        assert "raw_source" not in payload
        assert "source_text" not in payload
        # Source refs only.
        assert capsule.evidence.source_cids
        assert all(
            item.startswith("source:") or item.startswith("semantic-node:")
            or item.startswith("source-cid:")
            for item in capsule.evidence.source_cids
        )


def test_does_not_rebuild_graph_authority() -> None:
    graph = _fixture_graph()
    graph_id = graph.graph_id
    snapshot_nodes = copy.deepcopy([n.to_dict() for n in graph.nodes])
    result = compile_semantic_capsules(graph)
    again = compile_semantic_capsules(graph)
    assert result.index.graph_id == graph_id
    assert again.index.graph_id == graph_id
    assert [n.to_dict() for n in graph.nodes] == snapshot_nodes
    for capsule in result.capsules:
        assert capsule.graph_id == graph_id


def test_verify_round_trip() -> None:
    result = compile_semantic_capsules(_fixture_graph())
    verified = verify_capsule_compile_result(result)
    assert verified.index.index_cid == result.index.index_cid
    as_dict = verify_capsule_compile_result(result.to_dict())
    assert as_dict.index.index_cid == result.index.index_cid


def test_invalidate_capsules_with_explicit_seeds() -> None:
    baseline = compile_semantic_capsules(_fixture_graph(symbol_payload="a"))
    changed = _fixture_graph(symbol_payload="b")
    result = invalidate_capsules(
        baseline, changed, seeds=("symbol.main",)
    )
    after = result.index.by_node_id()
    subjects = {
        item.subject_cid
        for item in result.explanations
        if item.action is CapsuleAction.INVALIDATE
    }
    assert after["symbol.main"].capsule_cid in subjects
    assert after["release.main"].capsule_cid in subjects
    assert after["proof.unrelated"].capsule_cid not in subjects


def test_compile_unknown_node_fails_closed() -> None:
    with pytest.raises(UnknownDependencyError):
        compile_semantic_capsule(_fixture_graph(), "missing.node")
