"""Hermetic tests for CASF BM25, vector, and knowledge-graph projections."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.causal_evidence import (
    RetrievalNominationBinding,
)
from ipfs_accelerate_py.agent_supervisor.federation.retrieval_projection import (
    RetrievalProjectionAuthorityError,
    RetrievalProjectionError,
    RetrievalProjectionStore,
    bind_index,
    bind_kg_relation,
    bind_nomination,
    bind_nomination_from_hit,
    indexes_invalidated_by_tree_change,
    retrieval_establishes_authority,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def test_retrieval_never_mints_authority() -> None:
    assert retrieval_establishes_authority() is False
    binding = sample_binding()
    hit = bind_nomination(
        binding=binding,
        index_id="index:bm25",
        subject_kind="symbol",
        subject_ref="symbol:dispatch",
        score_millionths=640_000,
        method="bm25",
        source_cid="source:dispatch",
        index_revision="1",
        record_id="hit:bm25",
    )
    assert hit.authority_disposition == "nomination_only"


def test_tree_mismatch_fails_closed() -> None:
    binding = sample_binding()
    with pytest.raises(RetrievalProjectionAuthorityError, match="tree identity mismatches"):
        bind_index(
            binding=binding,
            retrieval_method="vector",
            index_revision=1,
            index_root="index:root",
            content_ref="index:content",
            record_id="index:vector",
            tree_id="tree:other",
        )
    with pytest.raises(RetrievalProjectionAuthorityError, match="tree identity mismatches"):
        bind_kg_relation(
            binding=binding,
            index_id="index:kg",
            index_revision=1,
            source_node_id="node:a",
            target_node_id="node:b",
            relationship_kind="calls",
            content_ref="kg:edge",
            record_id="kg:edge",
            tree_id="tree:other",
        )


def test_unknown_retrieval_method_fails_closed() -> None:
    binding = sample_binding()
    with pytest.raises(Exception, match="method is not closed"):
        bind_index(
            binding=binding,
            retrieval_method="vibes",
            index_revision=1,
            index_root="index:root",
            content_ref="index:content",
            record_id="index:bad",
        )


def test_nomination_binding_must_match_federation_tree() -> None:
    binding = sample_binding()
    hit = RetrievalNominationBinding(
        index_revision="1",
        source_cid="source:dispatch",
        tree_id="tree:other",
        method="vector",
        score_millionths=10,
        partition_id="partition:a",
    )
    with pytest.raises(RetrievalProjectionAuthorityError, match="tree identity mismatches"):
        bind_nomination_from_hit(
            hit,
            binding=binding,
            index_id="index:vector",
            subject_kind="symbol",
            subject_ref="symbol:dispatch",
            record_id="hit:vector",
        )
    matching = RetrievalNominationBinding(
        index_revision="1",
        source_cid="source:dispatch",
        tree_id=binding.repository_tree_ids[0],
        method="vector",
        score_millionths=12,
        partition_id="partition:a",
    )
    projected = bind_nomination_from_hit(
        matching,
        binding=binding,
        index_id="index:vector",
        subject_kind="symbol",
        subject_ref="symbol:dispatch",
        record_id="hit:vector",
    )
    assert projected.method == "vector"
    assert projected.nomination_binding.tree_id == binding.repository_tree_ids[0]


def test_tree_change_invalidates_indexes_on_that_tree_only() -> None:
    binding = sample_binding()
    bm25 = bind_index(
        binding=binding,
        retrieval_method="bm25",
        index_revision=1,
        index_root="index:bm25-root",
        content_ref="index:bm25",
        record_id="index:bm25",
    )
    vector = bind_index(
        binding=binding,
        retrieval_method="vector",
        index_revision=2,
        index_root="index:vector-root",
        content_ref="index:vector",
        record_id="index:vector",
    )
    affected = indexes_invalidated_by_tree_change(
        (bm25, vector), tree_id=binding.repository_tree_ids[0]
    )
    assert set(affected) == {"index:bm25", "index:vector"}
    assert indexes_invalidated_by_tree_change((bm25, vector), tree_id="tree:other") == ()


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(RetrievalProjectionError, match="database path"):
        RetrievalProjectionStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for retrieval projection")
def test_store_records_bm25_vector_kg_and_invalidates_tree_indexes(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:retrieval-projection")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:retrieval-projection",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = RetrievalProjectionStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _receipt = _create(
        store,
        request=sample_request(
            binding=binding, maximum_supervisors=2, maximum_subagents=2
        ),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    bm25 = bind_index(
        binding=binding,
        retrieval_method="bm25",
        index_revision=1,
        index_root="index:bm25-root",
        content_ref="index:bm25",
        record_id="index:bm25",
    )
    vector = bind_index(
        binding=binding,
        retrieval_method="vector",
        index_revision=1,
        index_root="index:vector-root",
        content_ref="index:vector",
        record_id="index:vector",
    )
    kg = bind_index(
        binding=binding,
        retrieval_method="kg",
        index_revision=1,
        index_root="index:kg-root",
        content_ref="index:kg",
        record_id="index:kg",
    )
    hit = bind_nomination(
        binding=binding,
        index_id=bm25.record_id,
        subject_kind="symbol",
        subject_ref="symbol:dispatch",
        score_millionths=500_000,
        method="bm25",
        source_cid="source:dispatch",
        index_revision="1",
        record_id="hit:bm25-dispatch",
        partition_id="partition:a",
    )
    relation = bind_kg_relation(
        binding=binding,
        index_id=kg.record_id,
        index_revision=1,
        source_node_id="symbol:dispatch",
        target_node_id="symbol:caller",
        relationship_kind="calls",
        content_ref="kg:calls",
        record_id="kg:dispatch-caller",
    )
    revision = store.graph_revision(
        tenant_id=binding.tenant_id, federation_id=identity.record_id
    )
    for item, method, key in (
        (bm25, store.record_index, "bm25"),
        (vector, store.record_index, "vector"),
        (kg, store.record_index, "kg"),
    ):
        revision = method(
            item,
            federation_id=identity.record_id,
            binding=binding,
            expected_graph_revision=revision,
            idempotency_key=f"idempotency:{key}",
            event_id=f"event:{key}",
        ).graph_revision
    revision = store.record_nomination(
        hit,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:hit",
        event_id="event:hit",
        receipt_id="receipt:bm25-1",
    ).graph_revision
    revision = store.record_kg_relation(
        relation,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:kg-edge",
        event_id="event:kg-edge",
    ).graph_revision
    store.invalidate_indexes(
        (bm25, vector, kg),
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:invalidate",
        event_id="event:tree-change",
    )
    loaded_hit = store.load_nomination(
        record_id="hit:bm25-dispatch",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_kg = store.load_kg_relation(record_id="kg:dispatch-caller")
    loaded_bm25 = store.load_index(
        record_id="index:bm25",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_hit["authority_disposition"] == "nomination_only"
    assert int(loaded_hit["score_micros"]) == 500_000
    assert loaded_kg["authority_disposition"] == "nomination_only"
    assert loaded_kg["relationship_kind"] == "calls"
    assert loaded_bm25["freshness_state"] == "invalidated"
    assert loaded_bm25["retrieval_method"] == "bm25"
    assert retrieval_establishes_authority() is False
