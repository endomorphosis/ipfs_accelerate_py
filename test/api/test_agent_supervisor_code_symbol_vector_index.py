from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
    CodeSymbolIndexRow,
    CodeSymbolLineage,
    CodeSymbolVectorIndexError,
    CodeSymbolVectorIndexIntegrityError,
    CodeSymbolVectorIndexStaleError,
    CodeVectorIndexSnapshot,
    CodeVectorQuery,
    CodeVectorSearchResult,
    build_code_symbol_vector_index,
    search_code_symbol_vector_index,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)


def _ast(path: str = "src/service.py", source: str = ""):
    source = source or (
        "class Service:\n"
        "    def dispatch(self, request):\n"
        "        self.status = 'running'\n"
        "        return request\n"
    )
    return build_analysis_ast_index(
        [(path, build_python_ast_blob_record(source, blob_identity="blob:service"))]
    )


def _index(ast=None, **kwargs):
    options = {
        "dimensions": 2,
        "producer_id": "fixture-producer@1",
        "chunker_id": "fixture-chunker@1",
        "model_id": "fixture-model",
        "model_revision": "fixture-revision",
        "configuration_id": "fixture-config@1",
        **kwargs,
    }
    vectors = options.pop("vectors", {
        "src.service.Service": (1.0, 0.0),
        "src.service.Service.dispatch": (0.0, 1.0),
        "src.runtime.service.Service": (1.0, 0.0),
        "src.runtime.service.Service.dispatch": (0.0, 1.0),
    })
    return build_code_symbol_vector_index(
        ast or _ast(),
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        coverage_id="coverage:fixture",
        vectors=vectors,
        **options,
    )


def test_snapshot_is_deterministic_body_free_and_binds_all_roots() -> None:
    forward = _index()
    reverse = _index(build_analysis_ast_index(reversed(_ast().path_records)))

    assert forward.index_id == reverse.index_id
    payload = forward.to_dict()
    assert payload["forest_id"] == "forest:fixture"
    assert payload["tree_id"] == "tree:fixture"
    assert payload["coverage_complete"] is True
    assert payload["config"]["producer_id"] == "fixture-producer@1"
    assert payload["config"]["dimensions"] == 2
    assert payload["config"]["metric"] == "cosine"
    assert payload["included_paths"] == ["src/service.py"]
    assert all("source" not in row for row in payload["rows"])
    assert all(row.sidecar.ast_record_id for row in forward.rows)
    assert any(row.sidecar.signature_refs for row in forward.rows)
    assert CodeVectorIndexSnapshot.from_dict(payload).index_id == forward.index_id


def test_query_and_all_hits_are_exact_snapshot_bound_and_advisory_only() -> None:
    index = _index()
    result = index.search((0.0, 1.0))

    assert result.complete is True
    assert result.searched_row_count == len(index.rows)
    assert [item.rank for item in result.hits] == list(range(1, len(result.hits) + 1))
    assert all(item.semantic_authority is False for item in result.hits)
    assert result.semantic_authority is False
    assert result.hits[0].row.symbol == "Service.dispatch"

    stale = CodeVectorQuery(
        "forest:fixture", "tree:other", index.index_id, index.config.config_id,
        2, "cosine", (0.0, 1.0),
    )
    with pytest.raises(CodeSymbolVectorIndexStaleError, match="roots"):
        search_code_symbol_vector_index(index, stale)

    poisoned = result.to_dict()
    poisoned["hits"][0]["semantic_authority"] = True
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="semantic authority"):
        CodeVectorSearchResult.from_dict(poisoned)


def test_dimension_normalization_and_incomplete_results_fail_closed() -> None:
    with pytest.raises(CodeSymbolVectorIndexError, match="dimension mismatch"):
        _index(dimensions=3)
    with pytest.raises(CodeSymbolVectorIndexError, match="l2 normalization"):
        _index(vectors={"src.service.Service": (2.0, 0.0), "src.service.Service.dispatch": (0.0, 2.0)})

    index = _index()
    query = CodeVectorQuery.for_snapshot(index, query_vector=(0.0, 1.0))
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="incomplete"):
        CodeVectorSearchResult(query, index.index_id, (), complete=False)


def test_forged_rows_bodies_and_snapshot_identity_are_rejected() -> None:
    index = _index()
    payload = index.to_dict()

    forged = copy.deepcopy(payload)
    forged["rows"][0]["sidecar"]["source"] = "def forged(): pass"
    with pytest.raises(CodeSymbolVectorIndexError, match="bodies"):
        CodeVectorIndexSnapshot.from_dict(forged)

    forged = copy.deepcopy(payload)
    forged["rows"][0]["embedding"] = [0.0, 1.0]
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="identity mismatch"):
        CodeVectorIndexSnapshot.from_dict(forged)

    row = index.rows[0].to_dict()
    row["row_id"] = "code-symbol-vector-row:sha256:forged"
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="identity mismatch"):
        CodeSymbolIndexRow.from_dict(row)


def test_incremental_rebuild_equals_clean_rebuild_and_lineage_needs_review() -> None:
    old_ast = _ast("src/service.py")
    old = _index(old_ast)
    new_ast = _ast("src/runtime/service.py")
    lineage = CodeSymbolLineage(
        old_path="src/service.py",
        new_path="src/runtime/service.py",
        blob_identity="blob:service",
        review_ref="review:move-service@1",
    )
    incremental = _index(new_ast, previous=old, reviewed_lineage=(lineage,))
    clean = _index(
        new_ast,
        previous=old,
        reviewed_lineage=(lineage,),
        tombstones=incremental.tombstones,
    )

    assert incremental.index_id == clean.index_id
    assert incremental.tombstones
    assert all(item.reason == "path_deleted" for item in incremental.tombstones)
    assert any(row.lineage_ids for row in incremental.rows)
    # Relocation provenance is not a synthetic semantic rename assertion.
    assert all(not hasattr(item, "semantic_rename") for item in incremental.lineage)

    with pytest.raises(CodeSymbolVectorIndexError, match="requires the previous"):
        _index(new_ast, reviewed_lineage=(lineage,))
