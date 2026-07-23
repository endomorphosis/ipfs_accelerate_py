from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_retrieval import (
    DEFAULT_SIGNAL_WEIGHTS,
    SIGNAL_ORDER,
    BackendState,
    BoundedGraphRAGRetriever,
    RetrievalBudgetError,
    RetrievalLimits,
    RetrievalQuery,
    RetrievalValidationError,
    retrieve_analysis_evidence,
)
from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    materialize_code_evidence_graph,
)


def _record(task_id: str, title: str, **values: object) -> dict[str, object]:
    return {
        "task_id": task_id,
        "title": title,
        "status": "todo",
        "track": "analysis",
        **values,
    }


def _fused_fixture() -> dict[str, object]:
    graph = {
        "graph_id": "graph-fixture",
        "nodes": [
            {
                "node_id": "node-task",
                "kind": "task",
                "record_key": "task-record",
                "task_id": "TASK-1",
                "provenance": "task",
                "record": {
                    "task_id": "TASK-1",
                    "title": "Secure parser proof repair",
                    "goal_id": "GOAL-1",
                },
            },
            {
                "node_id": "node-obligation",
                "kind": "obligation",
                "record_key": "OBL-1",
                "obligation_id": "OBL-1",
                "provenance": "proof",
                "record": {"obligation_id": "OBL-1"},
            },
            {
                "node_id": "node-symbol",
                "kind": "symbol",
                "record_key": "src/parser.py:Parser.validate",
                "symbol": "Parser.validate",
                "provenance": "ast",
                "record": {
                    "qualified_name": "Parser.validate",
                    "path": "src/parser.py",
                },
            },
        ],
        "edges": [
            {
                "edge_id": "edge-task-obligation",
                "source": "node-task",
                "target": "node-obligation",
                "kind": "has_obligation",
            },
            {
                "edge_id": "edge-obligation-symbol",
                "source": "node-obligation",
                "target": "node-symbol",
                "kind": "covers",
            },
        ],
    }
    records = [
        _record(
            "TASK-1",
            "Secure parser proof repair",
            goal_id="GOAL-1",
            ast_symbols=["Parser.validate"],
            predicted_files=["src/parser.py"],
            embedding=[1.0, 0.0],
        ),
        _record(
            "TASK-0",
            "Parser prerequisite",
            goal_id="GOAL-0",
            embedding=[0.0, 1.0],
        ),
    ]
    dependency_graph = {
        "nodes": {
            "cid-0": {"task_cid": "cid-0", "task_id": "TASK-0"},
            "cid-1": {"task_cid": "cid-1", "task_id": "TASK-1"},
        },
        "edges": [
            {
                "source_task_cid": "cid-0",
                "target_task_cid": "cid-1",
                "kind": "output_input",
            }
        ],
    }
    coverage = {
        "graph_id": "coverage-fixture",
        "criteria": [
            {
                "criterion_id": "criterion-1",
                "goal_id": "GOAL-1",
                "criterion": "Parser validation is proved",
                "status": "uncovered",
                "task_ids": ["TASK-1"],
            }
        ],
        "edges": [
            {
                "edge_id": "coverage-edge-1",
                "criterion_id": "criterion-1",
                "goal_id": "GOAL-1",
                "task_id": "TASK-1",
                "status": "uncovered",
                "value": "Parser.validate",
            }
        ],
    }
    proof_scope_index = {
        "index_id": "proof-index-fixture",
        "obligations": [
            {
                "obligation_id": "OBL-1",
                "scope_ids": ["scope-1"],
                "scope_keys": [
                    {"kind": "qualified_symbol", "value": "Parser.validate"}
                ],
                "dependency_ids": [],
            }
        ],
        "receipts": [],
        "invalidations": [],
        "blobs": [
            {
                "path": "src/parser.py",
                "blob_id": "blob-1",
                "scopes": [
                    {
                        "scope_id": "scope-1",
                        "path": "src/parser.py",
                        "blob_id": "blob-1",
                        "keys": [
                            {
                                "kind": "qualified_symbol",
                                "value": "Parser.validate",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return {
        "evidence_graph": graph,
        "records": records,
        "dependency_graph": dependency_graph,
        "goal_coverage": coverage,
        "proof_scope_index": proof_scope_index,
    }


def test_ranking_and_serialization_are_stable_across_input_order() -> None:
    records = [
        _record("TASK-3", "alpha parser helper"),
        _record("TASK-1", "alpha parser implementation"),
        _record("TASK-2", "unrelated renderer"),
    ]

    first = retrieve_analysis_evidence("alpha parser", records=records)
    second = retrieve_analysis_evidence("alpha parser", records=reversed(records))

    assert first.to_json() == second.to_json()
    assert [item.task_id for item in first.results] == ["TASK-3", "TASK-1"]
    assert first.response_id == second.response_id


def test_all_six_signals_are_fused_without_weight_renormalization() -> None:
    retriever = BoundedGraphRAGRetriever(**_fused_fixture())
    response = retriever.retrieve(
        RetrievalQuery(
            text="secure parser proof gap",
            task_ids=("TASK-0",),
            goal_ids=("GOAL-1",),
            symbols=("Parser.validate",),
            embedding=(1.0, 0.0),
        )
    )

    task = next(item for item in response.results if item.task_id == "TASK-1")
    assert tuple(task.signal_scores) == SIGNAL_ORDER
    assert all(item.available for item in task.signal_scores.values())
    assert all(item.score > 0 for item in task.signal_scores.values())
    assert task.score == pytest.approx(
        sum(
            item.score * DEFAULT_SIGNAL_WEIGHTS[name]
            for name, item in task.signal_scores.items()
        ),
        abs=2e-6,
    )
    assert response.backend_health["vector"].state is BackendState.HEALTHY
    assert "Fixed-weight fusion" in task.ranking_explanation


class _BrokenVectorBackend:
    def health(self) -> dict[str, object]:
        return {"healthy": True, "status": "ready"}

    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        raise RuntimeError("vector service offline")


def test_optional_backend_failure_is_explicit_and_keeps_fixed_semantics() -> None:
    records = [_record("TASK-1", "alpha parser")]
    baseline = retrieve_analysis_evidence("alpha", records=records)
    degraded = retrieve_analysis_evidence(
        "alpha", records=records, vector_backend=_BrokenVectorBackend()
    )

    assert baseline.results[0].score == degraded.results[0].score
    assert degraded.backend_health["vector"].state is BackendState.UNHEALTHY
    assert degraded.results[0].signal_scores["vector"].available is False
    assert degraded.results[0].signal_scores["vector"].contribution == 0.0
    assert degraded.results[0].signal_scores["lexical"].weight == DEFAULT_SIGNAL_WEIGHTS["lexical"]
    assert "RuntimeError" in degraded.backend_health["vector"].detail


class _VectorBackend:
    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        return [{"task_id": "TASK-VECTOR", "score": 1.0}]


def test_vector_backend_scores_only_existing_evidence_references() -> None:
    response = retrieve_analysis_evidence(
        "needle",
        records=[
            _record("TASK-LEXICAL", "needle"),
            _record("TASK-VECTOR", "semantic candidate"),
        ],
        vector_backend=_VectorBackend(),
    )

    vector = next(item for item in response.results if item.task_id == "TASK-VECTOR")
    assert vector.signal_scores["vector"].score == 1.0
    assert vector.evidence_references
    assert response.backend_health["vector"].candidate_count == 1


def test_provenance_is_stable_and_large_or_unsafe_payloads_never_enter_results() -> None:
    marker = "NEVER-COPY-THIS-SOURCE-BODY"
    response = retrieve_analysis_evidence(
        "bounded evidence",
        records=[
            {
                "record_id": "record-safe-1",
                "task_id": "TASK-1",
                "title": "bounded evidence",
                "source_body": marker * 10_000,
                "decoded_model_text": "MODEL-" + marker,
                "payload": {
                    "nested_graph": [{"body": marker}],
                    "model_response": marker,
                },
                "embedding": [0.1] * 10_000,
            }
        ],
        artifact_id="artifact:sha256:abc",
    )

    result = response.results[0]
    serialized = response.to_json()
    assert marker not in serialized
    assert "decoded_model_text" not in serialized
    assert "nested_graph" not in serialized
    assert len(result.evidence_references) == 1
    reference = result.evidence_references[0]
    assert reference.source_id == "record-safe-1"
    assert reference.artifact_id == "artifact:sha256:abc"
    assert reference.reference_id == result.evidence_references[0].reference_id


def test_graph_and_todo_views_deduplicate_by_semantic_task_identity() -> None:
    graph = {
        "graph_id": "graph-dedupe",
        "nodes": [
            {
                "node_id": "node-task-1",
                "kind": "task",
                "record_key": "graph-task-record",
                "task_id": "TASK-1",
                "provenance": "task",
                "record": {"task_id": "TASK-1", "title": "shared parser task"},
            }
        ],
        "edges": [],
    }
    response = retrieve_analysis_evidence(
        "shared parser",
        evidence_graph=graph,
        todo_records=[
            _record("TASK-1", "shared parser task", record_id="todo-record-1"),
            _record("TASK-1", "shared parser task", record_id="todo-record-1"),
        ],
    )

    matching = [item for item in response.results if item.task_id == "TASK-1"]
    assert len(matching) == 1
    assert {
        reference.source_kind for reference in matching[0].evidence_references
    } == {"code_evidence_node", "todo_index_record"}


def test_typed_graph_obligation_with_task_id_propagates_failed_proof_gap() -> None:
    graph = materialize_code_evidence_graph(
        tasks=[{"task_id": "TASK-1", "title": "repair parser proof"}],
        obligations=[{"obligation_id": "OBL-1", "task_id": "TASK-1"}],
        proof_receipts=[
            {
                "receipt_id": "RECEIPT-1",
                "obligation_id": "OBL-1",
                "verdict": "failed",
            }
        ],
    )
    response = retrieve_analysis_evidence(
        RetrievalQuery(text="repair parser proof", task_ids=("TASK-1",)),
        evidence_graph=graph,
    )

    task = next(item for item in response.results if item.task_id == "TASK-1")
    obligation = next(
        item for item in response.results if item.obligation_id == "OBL-1"
    )
    assert task.signal_scores["proof_gap"].score == pytest.approx(0.85)
    assert obligation.signal_scores["proof_gap"].score == 1.0
    assert any(
        reference.record_id == "OBL-1"
        for reference in task.evidence_references
    )


def test_count_candidate_and_total_serialized_byte_limits_are_strict() -> None:
    records = [
        _record(f"TASK-{index:02d}", f"common bounded evidence candidate {index}")
        for index in range(20)
    ]
    count_response = retrieve_analysis_evidence(
        "common bounded evidence",
        records=records,
        limits=RetrievalLimits(
            max_results=3,
            max_candidates=5,
            max_bytes=16_384,
        ),
    )

    assert len(count_response.results) == 3
    assert count_response.truncation.dropped_by_candidate_limit == 15
    assert count_response.truncation.dropped_by_count_limit == 2
    assert count_response.truncation.truncated is True

    byte_response = retrieve_analysis_evidence(
        "common bounded evidence",
        records=records,
        limits=RetrievalLimits(
            max_results=20,
            max_candidates=20,
            max_bytes=4_096,
        ),
    )
    encoded = byte_response.to_json().encode("utf-8")
    assert len(encoded) <= 4_096
    assert byte_response.truncation.output_bytes == len(encoded)
    assert byte_response.truncation.dropped_by_byte_limit > 0
    assert byte_response.truncation.truncated is True


def test_no_match_still_reports_every_backend_and_truncation_metadata() -> None:
    response = retrieve_analysis_evidence(
        "absent",
        records=[_record("TASK-1", "different")],
    )

    assert response.results == ()
    assert tuple(response.backend_health) == SIGNAL_ORDER
    assert response.backend_health["lexical"].state is BackendState.HEALTHY
    assert response.backend_health["vector"].state is BackendState.UNAVAILABLE
    assert response.truncation.considered_count == 1
    assert response.truncation.returned_count == 0
    assert response.truncation.output_bytes == len(response.to_json().encode("utf-8"))


def test_query_and_limit_validation_fail_closed() -> None:
    with pytest.raises(RetrievalValidationError):
        RetrievalQuery(text="")
    with pytest.raises(RetrievalValidationError):
        RetrievalLimits(max_results=0)
    with pytest.raises(RetrievalBudgetError):
        retrieve_analysis_evidence(
            "alpha",
            records=[_record("TASK-1", "alpha")],
            limits=RetrievalLimits(max_bytes=100),
        )
    with pytest.raises(RetrievalValidationError):
        BoundedGraphRAGRetriever(signal_weights={"lexical": 1.0})


def test_output_is_canonical_json_and_contains_no_non_finite_scores() -> None:
    response = retrieve_analysis_evidence(
        {"text": "alpha", "embedding": [1.0, 0.0]},
        records=[_record("TASK-1", "alpha", embedding=[1.0, 0.0])],
    )
    payload = json.loads(response.to_json())

    assert payload == response.to_dict()
    assert payload["query_id"].startswith("analysis-query:sha256:")
    assert payload["response_id"].startswith("analysis-retrieval:sha256:")
    assert payload["ranking"]["unavailable_signal_semantics"].startswith(
        "zero_contribution"
    )
