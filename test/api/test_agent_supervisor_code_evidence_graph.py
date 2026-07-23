from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    CODE_EVIDENCE_GRAPH_KIND,
    canonical_code_evidence_graph_records,
    query_artifact,
    read_code_evidence_graph,
    read_code_evidence_graph_projection,
    write_code_evidence_graph_artifact,
)
from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    CodeEvidenceGraph,
    EvidenceEdgeKind,
    EvidenceGraphValidationError,
    EvidenceNodeKind,
    build_code_evidence_graph,
)


def _records() -> dict[str, list[dict[str, object]]]:
    return {
        "task_records": [
            {
                "task_id": "REF-250",
                "canonical_task_cid": "task-cid-250",
                "depends_on": ["REF-248", "REF-245"],
                "repository_tree_id": "tree-candidate",
                "status": "in_progress",
            },
            {"task_id": "REF-245", "canonical_task_cid": "task-cid-245"},
            {"task_id": "REF-248", "canonical_task_cid": "task-cid-248"},
        ],
        "ast_records": [
            {
                "scope_id": "scope-symbol",
                "kind": "qualified_symbol",
                "qualified_symbol": "agent_supervisor.code_evidence_graph.CodeEvidenceGraph",
                "repository_tree_id": "tree-candidate",
                "path": "agent_supervisor/code_evidence_graph.py",
                "source_hash": "sha256:source",
            }
        ],
        "obligations": [
            {
                "obligation_id": "obligation-projection",
                "task_id": "REF-250",
                "repository_tree_id": "tree-candidate",
                "ast_scope_ids": ["scope-symbol"],
                "premise_ids": [],
                "required_assurance": "kernel_verified",
                "statement": "JSON and DuckDB projections are equivalent.",
            }
        ],
        "attempts": [
            {
                "attempt_id": "attempt-1",
                "obligation_id": "obligation-projection",
                "repository_tree_id": "tree-candidate",
                "status": "succeeded",
            }
        ],
        "proof_records": [
            {
                "receipt_id": "proof-receipt-1",
                "attempt_id": "attempt-1",
                "obligation_id": "obligation-projection",
                "repository_tree_id": "tree-candidate",
                "verdict": "proved",
                "authoritative_assurance": "kernel_verified",
                "freshness": "current",
            }
        ],
        "validation_records": [
            {
                "validation_receipt_id": "validation-1",
                "task_id": "REF-250",
                "repository_tree_id": "tree-candidate",
                "status": "passed",
                "freshness": "current",
                "obligation_ids": ["obligation-projection"],
            }
        ],
        "merge_records": [
            {
                "merge_receipt_id": "merge-1",
                "task_id": "REF-250",
                "repository_tree_id": "tree-candidate",
                "status": "merged",
                "completion_status": "completed",
            }
        ],
    }


def test_graph_is_deterministic_and_derived_from_typed_record_channels() -> None:
    records = _records()
    first = build_code_evidence_graph(**records)
    reversed_records = {name: list(reversed(values)) for name, values in records.items()}
    second = build_code_evidence_graph(**reversed_records)

    assert first.graph_id == second.graph_id
    assert first.to_json() == second.to_json()
    assert CodeEvidenceGraph.from_json(first.to_json()) == first
    assert first.find_nodes(
        kind=EvidenceNodeKind.SYMBOL,
        symbol="agent_supervisor.code_evidence_graph.CodeEvidenceGraph",
    )
    assert {edge.kind for edge in first.edges}.issuperset(
        {
            EvidenceEdgeKind.DEPENDS_ON,
            EvidenceEdgeKind.DEFINES_SYMBOL,
            EvidenceEdgeKind.HAS_OBLIGATION,
            EvidenceEdgeKind.COVERS,
            EvidenceEdgeKind.ATTEMPT_FOR,
            EvidenceEdgeKind.PROVES,
            EvidenceEdgeKind.VALIDATES,
            EvidenceEdgeKind.MERGED,
            EvidenceEdgeKind.COMPLETES,
        }
    )
    assert all(
        edge.authoritative
        for edge in first.edges
        if edge.kind
        in {
            EvidenceEdgeKind.PROVES,
            EvidenceEdgeKind.VALIDATES,
            EvidenceEdgeKind.MERGED,
            EvidenceEdgeKind.COMPLETES,
            EvidenceEdgeKind.COVERS,
        }
    )


def test_json_and_duckdb_round_trip_canonical_records_and_indexes(
    tmp_path: Path,
) -> None:
    graph = build_code_evidence_graph(**_records())
    json_path = tmp_path / "code-evidence.json"

    rendered = write_code_evidence_graph_artifact(json_path, graph)
    duckdb_path = json_path.with_suffix(".duckdb")

    assert rendered["query_store"]["artifact_kind"] == CODE_EVIDENCE_GRAPH_KIND
    assert json_path.exists() and duckdb_path.exists()
    assert read_code_evidence_graph(json_path) == graph
    assert read_code_evidence_graph(duckdb_path) == graph
    assert (
        canonical_code_evidence_graph_records(json_path)
        == canonical_code_evidence_graph_records(duckdb_path)
        == graph.canonical_records()
    )
    assert read_code_evidence_graph_projection(json_path)["graph_id"] == graph.graph_id

    task_rows = query_artifact(
        json_path,
        table="task_index",
        columns=("task_id",),
        where="task_id = 'REF-250'",
        kind=CODE_EVIDENCE_GRAPH_KIND,
    )["rows"]
    assert task_rows == [{"task_id": "REF-250"}]
    assert query_artifact(
        duckdb_path,
        table="tree_index",
        columns=("tree_id",),
        where="tree_id = 'tree-candidate'",
    )["row_count"] == 1
    assert query_artifact(
        json_path,
        table="symbol_index",
        columns=("symbol",),
    )["rows"] == [
        {
            "symbol": "agent_supervisor.code_evidence_graph.CodeEvidenceGraph"
        }
    ]
    assert query_artifact(
        json_path,
        table="obligation_index",
        columns=("obligation_id",),
    )["rows"] == [{"obligation_id": "obligation-projection"}]
    assert {
        row["assurance"]
        for row in query_artifact(
            json_path, table="assurance_index", columns=("assurance",)
        )["rows"]
    } == {"kernel_verified"}
    assert query_artifact(
        json_path,
        table="freshness_index",
        columns=("freshness",),
        where="freshness = 'current'",
    )["row_count"] == 2
    assert query_artifact(
        json_path,
        table="dependency_index",
        columns=("edge_kind",),
    )["row_count"] == 2


@pytest.mark.parametrize(
    "forged_kind", ["proves", "merged", "covers", "completes", "validates"]
)
def test_llm_and_graphrag_enrichment_cannot_create_authoritative_edges(
    forged_kind: str,
) -> None:
    with pytest.raises(EvidenceGraphValidationError, match="enrichment cannot create"):
        build_code_evidence_graph(
            **_records(),
            enrichments=[
                {
                    "id": f"llm-{forged_kind}",
                    "source": "GraphRAG",
                    "edge_kind": forged_kind,
                    "targets": ["REF-250"],
                    "claim": "A model says this task is complete.",
                }
            ],
        )


def test_deserialization_rejects_forged_edge_authority() -> None:
    graph = build_code_evidence_graph(**_records())
    payload = json.loads(graph.to_json())
    proof = next(edge for edge in payload["edges"] if edge["kind"] == "proves")
    proof["provenance"] = "enrichment"
    proof.pop("edge_id")
    proof["authoritative"] = True

    with pytest.raises(EvidenceGraphValidationError, match="enrichment cannot create"):
        CodeEvidenceGraph.from_dict(payload)


def test_non_authoritative_enrichment_is_preserved_as_descriptive_context() -> None:
    graph = build_code_evidence_graph(
        **_records(),
        enrichments=[
            {
                "id": "graphrag-context-1",
                "source": "GraphRAG",
                "edge_kind": "mentions",
                "targets": ["REF-250"],
                "summary": "This task concerns a graph projection.",
            }
        ],
    )

    edges = graph.edges_by_kind(EvidenceEdgeKind.MENTIONS)
    assert len(edges) == 1
    assert edges[0].authoritative is False
    assert graph.nodes_by_kind(EvidenceNodeKind.ENRICHMENT)[0].authoritative is False
