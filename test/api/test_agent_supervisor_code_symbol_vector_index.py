from __future__ import annotations

import copy
from dataclasses import replace

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
    resolve_code_symbol_ast_facts,
    validate_code_vector_search_result,
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


@pytest.mark.parametrize("source", [
    "def large():\n    state = [" + ",".join(str(i) for i in range(200)) + "]\n    return state\n",
    "def large():\n" + "".join(f"    state_{i} = {i}\n" for i in range(50)),
    "def large():\n" + "".join(f"    called_{i}()\n" for i in range(50)),
    "def large():\n" + "".join(f"    state_{i} = '{'x' * 240}'\n" for i in range(31)),
], ids=["oversized-effect", "many-effects", "many-calls", "total-fact-byte-budget"])
def test_large_native_fact_sets_remain_complete_and_hydrate_from_exact_ast(source):
    ast = _ast(source=source)
    result = _index(ast, vectors=lambda row: (1.0, 0.0))
    row = next(row for row in result.rows if row.symbol == "large")
    indexed = next(iter(ast.path_records))
    # Bind the entire native fact set; no oversized literal and no truncation.
    assert any(ref.startswith("code-ast-facts:sha256:")
               for ref in (*row.sidecar.signature_refs, *row.sidecar.call_refs, *row.sidecar.effect_refs))
    hydrated = resolve_code_symbol_ast_facts(indexed, row)
    assert hydrated["effect_refs"] == tuple(f for f in indexed.ast_record.state_transitions if f.startswith("large:"))
    assert hydrated["call_refs"] == tuple(f for f in indexed.ast_record.calls if f.startswith("large->"))
    restored = CodeVectorIndexSnapshot.from_dict(result.to_dict())
    assert restored.index_id == result.index_id
    assert resolve_code_symbol_ast_facts(indexed, restored.rows[0]) == hydrated
    changed = _ast(source=source + "\ndef added(): pass\n")
    with pytest.raises(CodeSymbolVectorIndexStaleError, match="binding"):
        resolve_code_symbol_ast_facts(next(iter(changed.path_records)), row)


def test_fact_compaction_does_not_relax_untrusted_feature_reference_bounds():
    with pytest.raises(CodeSymbolVectorIndexError, match="bound"):
        _index(feature_references={"src.service.Service.dispatch": {"effect_refs": ["x" * 321]}})


def test_persisted_inline_native_facts_remain_hydratable_without_allowing_truncation():
    source = "def large():\n" + "".join(f"    state_{i} = '{'x' * 160}'\n" for i in range(10))
    ast = _ast(source=source)
    snapshot = _index(ast, vectors=lambda row: (1.0, 0.0))
    indexed = ast.path_records[0]
    row = snapshot.rows[0]
    effects = tuple(sorted(indexed.ast_record.state_transitions))
    old_row = replace(row, sidecar=replace(row.sidecar, effect_refs=effects))
    old_snapshot = replace(snapshot, rows=(old_row,))
    reopened = CodeVectorIndexSnapshot.from_dict(old_snapshot.to_dict())
    assert resolve_code_symbol_ast_facts(indexed, reopened.rows[0])["effect_refs"] == effects
    truncated = replace(old_row, sidecar=replace(old_row.sidecar, effect_refs=effects[:-1]))
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="do not replay"):
        resolve_code_symbol_ast_facts(indexed, truncated)


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


@pytest.mark.parametrize("zero", [(0.0, 0.0), (-0.0, 0.0)])
def test_zero_cosine_query_has_complete_replayable_empty_nominations(zero) -> None:
    index = _index()
    result = index.search(zero, max_results=1)
    assert len(index.rows) == 2
    assert result.hits == () and result.complete is True
    assert result.searched_row_count == len(index.rows)
    assert result.query.query_vector == (0.0, 0.0)
    assert result.query.max_results == 1
    assert result.query.semantic_authority is result.semantic_authority is False
    restored = CodeVectorSearchResult.from_dict(result.to_dict())
    assert restored == result
    assert search_code_symbol_vector_index(index, restored.query) == restored
    assert validate_code_vector_search_result(index, restored) == restored


def test_zero_query_cannot_forge_hits_row_coverage_or_authority() -> None:
    index = _index()
    result = index.search((0.0, 0.0))
    hit = replace(index.search((1.0, 0.0)).hits[0], query_id=result.query.query_id, score=0.0)
    forged = replace(result, hits=(hit,))
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="zero cosine query"):
        validate_code_vector_search_result(index, CodeVectorSearchResult.from_dict(forged.to_dict()))
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="incomplete"):
        validate_code_vector_search_result(index, replace(result, searched_row_count=0))
    poisoned = result.to_dict()
    poisoned["semantic_authority"] = True
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="semantic authority"):
        CodeVectorSearchResult.from_dict(poisoned)
    poisoned = result.to_dict()
    poisoned["query"]["semantic_authority"] = True
    with pytest.raises(CodeSymbolVectorIndexIntegrityError, match="semantic authority"):
        CodeVectorSearchResult.from_dict(poisoned)


@pytest.mark.parametrize("field,value", [("tree_id", "tree:stale"),
    ("index_id", "index:stale"), ("config_id", "config:stale"), ("forest_id", "forest:stale")])
def test_zero_query_still_rejects_stale_roots(field, value) -> None:
    index = _index()
    query = CodeVectorQuery.for_snapshot(index, query_vector=(0.0, 0.0))
    with pytest.raises(CodeSymbolVectorIndexStaleError, match="roots"):
        search_code_symbol_vector_index(index, replace(query, **{field: value}))


@pytest.mark.parametrize("vector", [(2.0, 0.0), (1e-300, 0.0), (0.5, 0.5)])
def test_zero_query_exception_does_not_relax_nonzero_normalization(vector) -> None:
    with pytest.raises(CodeSymbolVectorIndexError, match="l2 normalization"):
        _index().search(vector)


@pytest.mark.parametrize("vector", [(0.0,), (float("nan"), 0.0), (float("inf"), 0.0)])
def test_zero_query_exception_does_not_relax_query_shape_or_finiteness(vector) -> None:
    with pytest.raises(CodeSymbolVectorIndexError):
        _index().search(vector)


def test_zero_query_exception_does_not_admit_zero_row_embeddings() -> None:
    with pytest.raises(CodeSymbolVectorIndexError, match="l2 normalization"):
        _index(vectors=lambda row: (0.0, 0.0))


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
