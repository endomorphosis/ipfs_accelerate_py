from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    build_code_evidence_graph,
)
from ipfs_accelerate_py.agent_supervisor.semantic_dependency_graph import (
    ClosureBounds,
    CrossRootEdgeError,
    SemanticAuthority,
    SemanticDependencyGraph,
    SemanticEdge,
    SemanticEdgeKind,
    SemanticGraphBoundsError,
    SemanticGraphError,
    MandatoryClosure,
    SemanticNode,
    SemanticNodeKind,
    SemanticProvenance,
    SemanticTrust,
    UnsafeDependencyCycleError,
    build_semantic_dependency_graph,
)


ROOT = "decision-root:sha256:fixture"
VERSION = "fixture-producer@1"


def _node(
    node_id: str,
    kind: SemanticNodeKind,
    *,
    provenance: SemanticProvenance = SemanticProvenance.SOURCE,
    trust: SemanticTrust = SemanticTrust.VERIFIED,
    authority: SemanticAuthority = SemanticAuthority.AUTHORITATIVE,
    root_id: str = ROOT,
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
        record={"id": node_id},
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


def _authority_graph(
    *, irrelevant: tuple[SemanticNode, ...] = ()
) -> SemanticDependencyGraph:
    nodes = (
        _node("decision", SemanticNodeKind.DECISION),
        _node("plan", SemanticNodeKind.PLAN),
        _node("action", SemanticNodeKind.ACTION),
        _node("constraint", SemanticNodeKind.LEGAL_OBLIGATION),
        _node("authorization", SemanticNodeKind.AUTHORIZATION),
        _node("obligation", SemanticNodeKind.OBLIGATION),
        _node(
            "proof",
            SemanticNodeKind.PROOF,
            provenance=SemanticProvenance.PROOF,
        ),
        _node(
            "monitor",
            SemanticNodeKind.MONITOR,
            provenance=SemanticProvenance.MONITOR,
        ),
        _node(
            "annotation",
            SemanticNodeKind.ANNOTATION,
            provenance=SemanticProvenance.GRAPHRAG,
            trust=SemanticTrust.UNTRUSTED,
            authority=SemanticAuthority.PROPOSAL_ONLY,
        ),
        *irrelevant,
    )
    edges = (
        _edge("decision", "plan", SemanticEdgeKind.REQUIRES),
        _edge("plan", "action", SemanticEdgeKind.IMPLEMENTS),
        _edge("action", "constraint", SemanticEdgeKind.CONSTRAINED_BY),
        _edge("action", "authorization", SemanticEdgeKind.REQUIRES),
        _edge("action", "obligation", SemanticEdgeKind.REQUIRES),
        _edge(
            "obligation",
            "proof",
            SemanticEdgeKind.PROVEN_BY,
            provenance=SemanticProvenance.PROOF,
        ),
        _edge(
            "action",
            "monitor",
            SemanticEdgeKind.MONITORED_BY,
            provenance=SemanticProvenance.MONITOR,
        ),
        _edge(
            "decision",
            "annotation",
            SemanticEdgeKind.AFFECTS,
            provenance=SemanticProvenance.GRAPHRAG,
            trust=SemanticTrust.UNTRUSTED,
            authority=SemanticAuthority.PROPOSAL_ONLY,
        ),
    )
    return SemanticDependencyGraph(ROOT, nodes, edges)


def test_typed_graph_and_forward_closure_are_canonical_and_deterministic() -> None:
    graph = _authority_graph()
    reversed_graph = SemanticDependencyGraph(
        ROOT, tuple(reversed(graph.nodes)), tuple(reversed(graph.edges))
    )

    assert graph.graph_id == reversed_graph.graph_id
    assert graph.to_json() == reversed_graph.to_json()
    assert SemanticDependencyGraph.from_json(graph.to_json()) == graph

    closure = graph.mandatory_closure("decision")
    assert set(closure.node_ids) == {
        "decision",
        "plan",
        "action",
        "constraint",
        "authorization",
        "obligation",
        "proof",
        "monitor",
    }
    assert closure.annotation_node_ids == ("annotation",)
    assert "annotation" not in closure.node_ids
    assert closure.complete and closure.to_dict()["truncated"] is False
    assert MandatoryClosure.from_dict(closure.to_dict()) == closure


def test_all_required_edge_families_are_typed() -> None:
    assert {item.value for item in SemanticEdgeKind} == {
        "requires",
        "constrained_by",
        "applies_to",
        "exception_to",
        "conflicts_with",
        "authorizes",
        "denies",
        "implements",
        "affects",
        "depends_on",
        "proven_by",
        "monitored_by",
        "invalidates",
        "sourced_from",
    }


def test_normalized_ir_projects_every_constraint_family() -> None:
    common = {
        "root_cid_v1": "bafyroot",
        "root_supervisor_digest": "sha256:root",
        "source_artifact_id": "artifact",
        "artifact_schema_version": "1",
        "trust_state": "trusted",
        "declared_authority": "authoritative",
    }

    def artifact(family: str, declarations: tuple[str, ...]) -> dict[str, object]:
        return {
            **common,
            "family": family,
            "root_artifact_id": f"{family}:root",
            "declarations": [
                {
                    "node_id": f"{family}:{kind}",
                    "family": family,
                    "node_kind": "declaration",
                    "declaration_kind": kind,
                    "trust_state": "trusted",
                    "result_authority": (
                        "descriptive_input"
                        if family == "intent_ir"
                        else "constraint_input"
                        if family == "legal_ir"
                        else "policy_input"
                    ),
                }
                for kind in declarations
            ],
            "formal_views": [],
            "claims": [],
            "assumptions": [],
            "obligations": [],
            "result_authority": [],
        }

    intent = artifact(
        "intent_ir",
        (
            "statement",
            "goal",
            "action",
            "control_flow",
            "precondition",
            "guard",
            "invariant",
            "effect",
            "postcondition",
            "assumption",
            "failure",
            "retry",
            "verification",
        ),
    )
    legal = artifact(
        "legal_ir",
        (
            "statement",
            "obligation",
            "prohibition",
            "permission",
            "power",
            "exception",
        ),
    )
    security = artifact(
        "security_ir",
        (
            "statement",
            "principal",
            "asset",
            "resource",
            "zone",
            "channel",
            "policy",
            "state_machine",
        ),
    )

    graph = build_semantic_dependency_graph(
        root_id=ROOT,
        normalized_ir_artifacts=(intent, legal, security),
    )

    assert {
        node.kind
        for node in graph.nodes
        if node.kind.value.startswith(("intent_", "legal_", "security_"))
    }.issuperset(
        {
            SemanticNodeKind.INTENT_GOAL,
            SemanticNodeKind.INTENT_DECLARATION,
            SemanticNodeKind.INTENT_VERIFICATION,
            SemanticNodeKind.LEGAL_OBLIGATION,
            SemanticNodeKind.LEGAL_DECLARATION,
            SemanticNodeKind.LEGAL_PROHIBITION,
            SemanticNodeKind.LEGAL_PERMISSION,
            SemanticNodeKind.LEGAL_POWER,
            SemanticNodeKind.LEGAL_EXCEPTION,
            SemanticNodeKind.SECURITY_PRINCIPAL,
            SemanticNodeKind.SECURITY_DECLARATION,
            SemanticNodeKind.SECURITY_RESOURCE,
            SemanticNodeKind.SECURITY_POLICY,
            SemanticNodeKind.SECURITY_STATE_MACHINE,
        }
    )
    assert all(node.root_id == ROOT for node in graph.nodes)
    assert {node.source_root_id for node in graph.nodes} == {
        "intent_ir:root",
        "legal_ir:root",
        "security_ir:root",
    }


def test_forged_authority_cross_root_edges_and_unsafe_cycles_fail_closed() -> None:
    with pytest.raises(SemanticGraphError, match="cannot create authoritative"):
        _node(
            "forged",
            SemanticNodeKind.AUTHORIZATION,
            provenance=SemanticProvenance.MODEL,
        )
    with pytest.raises(SemanticGraphError, match="require SecurityIR"):
        _edge("authorization", "action", SemanticEdgeKind.AUTHORIZES)

    nodes = (
        _node("decision", SemanticNodeKind.DECISION),
        _node("foreign", SemanticNodeKind.OBLIGATION, root_id="other-root"),
    )
    with pytest.raises(CrossRootEdgeError):
        SemanticDependencyGraph(ROOT, nodes, ())

    cyclic_nodes = (
        _node("decision", SemanticNodeKind.DECISION),
        _node("a", SemanticNodeKind.OBLIGATION),
        _node("b", SemanticNodeKind.PREMISE),
    )
    with pytest.raises(UnsafeDependencyCycleError):
        SemanticDependencyGraph(
            ROOT,
            cyclic_nodes,
            (
                _edge("decision", "a", SemanticEdgeKind.REQUIRES),
                _edge("a", "b", SemanticEdgeKind.DEPENDS_ON),
                _edge("b", "a", SemanticEdgeKind.DEPENDS_ON),
            ),
        )


def test_deserialization_rejects_forged_bindings_and_closure_is_bounded() -> None:
    graph = _authority_graph()
    payload = json.loads(graph.to_json())
    payload["edges"][0]["root_id"] = "foreign"
    payload["edges"][0].pop("edge_id")
    payload.pop("graph_id")
    with pytest.raises(CrossRootEdgeError):
        SemanticDependencyGraph.from_dict(payload)

    with pytest.raises(SemanticGraphBoundsError, match="max_nodes"):
        graph.mandatory_closure(
            "decision",
            bounds=ClosureBounds(
                max_nodes=2,
                max_edges=100,
                max_depth=100,
                max_annotations=100,
            ),
        )


def test_irrelevant_graph_growth_does_not_change_decision_closure() -> None:
    base = _authority_graph()
    grown = _authority_graph(
        irrelevant=tuple(
            _node(f"irrelevant-{index}", SemanticNodeKind.RESOURCE)
            for index in range(100)
        )
    )

    assert base.graph_id != grown.graph_id
    assert (
        base.mandatory_closure("decision").closure_id
        == grown.mandatory_closure("decision").closure_id
    )


def test_legacy_code_evidence_authority_is_preserved_without_enrichment() -> None:
    evidence = build_code_evidence_graph(
        task_records=(
            {
                "task_id": "ASI-130",
                "repository_tree_id": "tree-1",
            },
        ),
        obligations=(
            {
                "obligation_id": "obligation-1",
                "task_id": "ASI-130",
                "repository_tree_id": "tree-1",
            },
        ),
        proof_records=(
            {
                "receipt_id": "proof-1",
                "obligation_id": "obligation-1",
                "repository_tree_id": "tree-1",
                "verdict": "proved",
                "authoritative_assurance": "kernel_verified",
                "freshness": "current",
            },
        ),
        enrichments=(
            {
                "id": "graphrag-1",
                "source": "GraphRAG",
                "edge_kind": "mentions",
                "targets": ["ASI-130"],
            },
        ),
    )

    semantic = evidence.to_semantic_dependency_graph(root_id=ROOT)

    assert semantic.edges_by_kind(SemanticEdgeKind.PROVEN_BY)[0].authoritative
    assert semantic.nodes_by_kind(SemanticNodeKind.ANNOTATION)[0].authoritative is False


def test_program_behavior_projection_binds_worktree_ast_tools_and_effects() -> None:
    behavior = {
        "schema": "ipfs_accelerate_py/agent-supervisor/program-behavior@1",
        "schema_version": 1,
        "behavior_root": "behavior:1",
        "repository_snapshot_id": "snapshot:1",
        "execution_tree_root": "execution:1",
        "program_root": "program:1",
        "ast_root": "ast:1",
        "tool_catalog_root": "tools:1",
        "environment_root": "environment:1",
        "effect_manifest_root": "effects:1",
        "repository": {
            "snapshot_id": "snapshot:1",
            "execution_tree_root": "execution:1",
            "head_tree_id": "tree:head",
            "head_commit_id": "commit:head",
            "entries": [
                {
                    "entry_id": "entry:service",
                    "path": "service.py",
                    "status": "modified",
                }
            ],
        },
        "analysis": {
            "program_root": "program:1",
            "ast_root": "ast:1",
            "observations": [
                {
                    "kind": "symbol",
                    "path": "service.py",
                    "ast_record_id": "ast-record:1",
                    "subject": "service",
                    "relationship": "defines",
                    "target": "service.run",
                }
            ],
        },
        "tools": {
            "catalog_root": "tools:1",
            "tools": [
                {
                    "tool_id": "python",
                    "version": "3.12",
                    "version_digest": "sha256:tool",
                }
            ],
        },
        "environment": {
            "environment_root": "environment:1",
            "python_version": "3.12",
        },
        "effects": {
            "manifest_root": "effects:1",
            "effects": [
                {
                    "effect_id": "effect:write",
                    "kind": "file",
                    "operation": "write",
                    "target": "service.py",
                }
            ],
        },
    }

    graph = build_semantic_dependency_graph(
        root_id=ROOT,
        program_behavior=behavior,
    )

    assert {
        SemanticNodeKind.PROGRAM,
        SemanticNodeKind.WORKTREE,
        SemanticNodeKind.REPOSITORY_TREE,
        SemanticNodeKind.FILE,
        SemanticNodeKind.AST,
        SemanticNodeKind.SYMBOL,
        SemanticNodeKind.TOOL,
        SemanticNodeKind.ENVIRONMENT,
        SemanticNodeKind.EFFECT,
        SemanticNodeKind.RESOURCE,
    }.issubset({node.kind for node in graph.nodes})
    assert all(node.authoritative for node in graph.nodes)
    assert all(edge.authoritative and edge.mandatory for edge in graph.edges)
