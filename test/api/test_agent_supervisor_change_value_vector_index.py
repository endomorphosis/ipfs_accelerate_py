from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_value_vector_index import (
    ChangeValueHit,
    ChangeValueIndexRow,
    ChangeValueIndexSnapshot,
    ChangeValueKind,
    ChangeValueLineage,
    ChangeValueQuery,
    ChangeValueSearchResult,
    ChangeValueSignal,
    ChangeValueVectorIndexError,
    ChangeValueVectorIndexIntegrityError,
    ChangeValueVectorIndexStaleError,
    build_change_value_vector_index,
    search_change_value_vector_index,
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
        "\n"
        "def create_context(config):\n"
        "    return config\n"
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
    vectors = options.pop(
        "vectors",
        {
            "src.service.Service": (1.0, 0.0),
            "src.service.Service.dispatch": (0.0, 1.0),
            "src.service.create_context": (0.7071067811865475, 0.7071067811865475),
            "src.runtime.service.Service": (1.0, 0.0),
            "src.runtime.service.Service.dispatch": (0.0, 1.0),
            "src.runtime.service.create_context": (
                0.7071067811865475,
                0.7071067811865475,
            ),
        },
    )
    return build_change_value_vector_index(
        ast or _ast(),
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        coverage_id="coverage:fixture",
        vectors=vectors,
        **options,
    )


def _query(index: ChangeValueIndexSnapshot, vector=(0.0, 1.0), **kwargs):
    options = {
        "missing_requirement_id": "req:missing-context@1",
        "missing_contract_refs": ("contract:process.context@1",),
        "consumer_path": "src/caller.py:handle",
        "obligation_id": "obligation:caller-handle@1",
        "consumer_context_refs": ("consumer:src/caller.py:handle",),
        **kwargs,
    }
    return ChangeValueQuery.for_snapshot(
        index,
        query_vector=vector,
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
    assert payload["coverage_id"] == "coverage:fixture"
    assert payload["config"]["producer_id"] == "fixture-producer@1"
    assert payload["config"]["chunker_id"] == "fixture-chunker@1"
    assert payload["config"]["normalization"] == "l2"
    assert payload["config"]["model_id"] == "fixture-model"
    assert payload["config"]["model_revision"] == "fixture-revision"
    assert payload["config"]["dimensions"] == 2
    assert payload["config"]["metric"] == "cosine"
    assert payload["config"]["configuration_id"] == "fixture-config@1"
    assert payload["included_paths"] == ["src/service.py"]
    assert all("source" not in row for row in payload["rows"])
    assert all("body" not in row for row in payload["rows"])
    assert all(row.sidecar.ast_record_id for row in forward.rows)
    assert all(row.semantic_authority is False for row in forward.rows)
    assert all(row.compatibility_claim is False for row in forward.rows)
    # Factory / method kinds are nominated with graph+AST+vector provenance.
    kinds = {row.kind for row in forward.rows}
    assert ChangeValueKind.CLASS in kinds or ChangeValueKind.SYMBOL in kinds
    assert any(
        ChangeValueSignal.VECTOR.value in row.signal_provenance
        for row in forward.rows
    )
    assert (
        ChangeValueIndexSnapshot.from_dict(payload).index_id == forward.index_id
    )


def test_query_requires_missing_contract_and_consumer_context() -> None:
    index = _index()
    with pytest.raises(ChangeValueVectorIndexError, match="missing_requirement"):
        ChangeValueQuery(
            index.forest_id,
            index.tree_id,
            index.index_id,
            index.config.config_id,
            2,
            "cosine",
            (0.0, 1.0),
            "",
            ("contract:x",),
            consumer_path="src/caller.py",
        )
    with pytest.raises(ChangeValueVectorIndexError, match="consumer context"):
        ChangeValueQuery(
            index.forest_id,
            index.tree_id,
            index.index_id,
            index.config.config_id,
            2,
            "cosine",
            (0.0, 1.0),
            "req:x",
            ("contract:x",),
        )


def test_query_and_all_hits_are_exact_snapshot_bound_and_advisory_only() -> None:
    index = _index()
    result = index.search(
        (0.0, 1.0),
        missing_requirement_id="req:missing-context@1",
        missing_contract_refs=("contract:process.context@1",),
        consumer_path="src/caller.py:handle",
        obligation_id="obligation:caller-handle@1",
        consumer_context_refs=("consumer:src/caller.py:handle",),
    )

    assert result.complete is True
    assert result.searched_row_count == len(index.rows)
    assert [item.rank for item in result.hits] == list(
        range(1, len(result.hits) + 1)
    )
    assert all(item.semantic_authority is False for item in result.hits)
    assert all(item.compatibility_claim is False for item in result.hits)
    assert all(item.signal_provenance for item in result.hits)
    assert all(
        ChangeValueSignal.VECTOR.value in item.signal_provenance
        for item in result.hits
    )
    assert result.semantic_authority is False
    assert result.compatibility_claim is False
    assert result.query.missing_requirement_id == "req:missing-context@1"
    assert "contract:process.context@1" in result.query.missing_contract_refs
    assert result.query.consumer_path == "src/caller.py:handle"
    assert result.hits[0].row.name == "Service.dispatch"

    stale = ChangeValueQuery(
        "forest:fixture",
        "tree:other",
        index.index_id,
        index.config.config_id,
        2,
        "cosine",
        (0.0, 1.0),
        "req:missing-context@1",
        ("contract:process.context@1",),
        consumer_path="src/caller.py:handle",
    )
    with pytest.raises(ChangeValueVectorIndexStaleError, match="roots"):
        search_change_value_vector_index(index, stale)

    poisoned = result.to_dict()
    poisoned["hits"][0]["semantic_authority"] = True
    with pytest.raises(
        ChangeValueVectorIndexIntegrityError, match="semantic authority"
    ):
        ChangeValueSearchResult.from_dict(poisoned)


def test_dimension_normalization_and_incomplete_results_fail_closed() -> None:
    with pytest.raises(ChangeValueVectorIndexError, match="dimension mismatch"):
        _index(dimensions=3)
    with pytest.raises(ChangeValueVectorIndexError, match="l2 normalization"):
        _index(
            vectors={
                "src.service.Service": (2.0, 0.0),
                "src.service.Service.dispatch": (0.0, 2.0),
                "src.service.create_context": (2.0, 0.0),
            }
        )

    index = _index()
    query = _query(index)
    with pytest.raises(ChangeValueVectorIndexIntegrityError, match="incomplete"):
        ChangeValueSearchResult(query, index.index_id, (), complete=False)


def test_forged_rows_bodies_and_snapshot_identity_are_rejected() -> None:
    index = _index()
    payload = index.to_dict()

    forged = copy.deepcopy(payload)
    forged["rows"][0]["sidecar"]["source"] = "def forged(): pass"
    with pytest.raises(ChangeValueVectorIndexError, match="bodies"):
        ChangeValueIndexSnapshot.from_dict(forged)

    forged = copy.deepcopy(payload)
    forged["rows"][0]["embedding"] = [0.0, 1.0]
    with pytest.raises(
        ChangeValueVectorIndexIntegrityError, match="identity mismatch"
    ):
        ChangeValueIndexSnapshot.from_dict(forged)

    row = index.rows[0].to_dict()
    row["row_id"] = "change-value-vector-row:sha256:forged"
    with pytest.raises(
        ChangeValueVectorIndexIntegrityError, match="identity mismatch"
    ):
        ChangeValueIndexRow.from_dict(row)


def test_incremental_rebuild_equals_clean_rebuild_and_lineage_needs_review() -> None:
    old_ast = _ast("src/service.py")
    old = _index(old_ast)
    new_ast = _ast("src/runtime/service.py")
    lineage = ChangeValueLineage(
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

    with pytest.raises(ChangeValueVectorIndexError, match="requires the previous"):
        _index(new_ast, reviewed_lineage=(lineage,))


def test_same_typed_or_similar_values_receive_no_compatibility_claim() -> None:
    index = _index()
    result = search_change_value_vector_index(index, _query(index, (1.0, 0.0)))

    # Multiple same-typed class/method candidates may appear; none may claim
    # compatibility merely from type or vector proximity.
    assert len(result.hits) >= 1
    for hit in result.hits:
        assert hit.compatibility_claim is False
        assert hit.row.compatibility_claim is False
        payload = hit.to_dict()
        assert payload["compatibility_claim"] is False
        assert "compatible" not in payload or payload.get("compatible") in (
            None,
            False,
        )

    poisoned = result.to_dict()
    poisoned["hits"][0]["compatibility_claim"] = True
    with pytest.raises(
        ChangeValueVectorIndexIntegrityError, match="compatibility claim"
    ):
        ChangeValueSearchResult.from_dict(poisoned)

    poisoned_row = index.rows[0].to_dict()
    poisoned_row["compatibility_claim"] = True
    with pytest.raises(
        ChangeValueVectorIndexIntegrityError, match="compatibility claim"
    ):
        ChangeValueIndexRow.from_dict(poisoned_row)


def test_cross_tree_and_dimension_mismatch_fail() -> None:
    index = _index()
    cross_tree = _query(index)
    stale = ChangeValueQuery(
        "forest:other",
        cross_tree.tree_id,
        cross_tree.index_id,
        cross_tree.config_id,
        cross_tree.dimensions,
        cross_tree.metric,
        cross_tree.query_vector,
        cross_tree.missing_requirement_id,
        cross_tree.missing_contract_refs,
        cross_tree.consumer_context_refs,
        cross_tree.consumer_path,
        cross_tree.obligation_id,
    )
    with pytest.raises(ChangeValueVectorIndexStaleError, match="roots"):
        search_change_value_vector_index(index, stale)

    with pytest.raises(ChangeValueVectorIndexError, match="dimension mismatch"):
        ChangeValueQuery.for_snapshot(
            index,
            query_vector=(0.0, 1.0, 0.0),
            missing_requirement_id="req:x",
            missing_contract_refs=("contract:x",),
            consumer_path="src/caller.py",
        )


def test_rows_carry_graph_and_ast_refs_not_bodies() -> None:
    index = _index(
        feature_references={
            "src.service.create_context": {
                "graph_node_refs": ("graph:node:create_context",),
                "factory_refs": ("factory:create_context",),
                "type_refs": ("type:Config",),
            }
        }
    )
    factory_rows = [
        row for row in index.rows if row.name == "create_context"
    ]
    assert factory_rows
    row = factory_rows[0]
    assert row.kind is ChangeValueKind.FACTORY
    assert "graph:node:create_context" in row.sidecar.graph_node_refs
    assert "factory:create_context" in row.sidecar.factory_refs
    assert row.sidecar.ast_record_id
    assert row.sidecar.blob_identity == "blob:service"
    payload = row.to_dict()
    assert "source" not in payload
    assert "source" not in payload["sidecar"]


def test_hit_signal_provenance_is_retained_through_round_trip() -> None:
    index = _index()
    result = search_change_value_vector_index(index, _query(index))
    restored = ChangeValueSearchResult.from_dict(result.to_dict())
    assert restored.result_id == result.result_id
    for hit in restored.hits:
        assert ChangeValueSignal.VECTOR.value in hit.signal_provenance
        assert hit.semantic_authority is False
        # Round-trip hit identity is stable.
        ChangeValueHit.from_dict(hit.to_dict())
