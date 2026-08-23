"""Hermetic tests for the multilevel CASF causal graph store.

Direct DuckDB inspection is used only after the typed embedded client closes.
The store never accepts a database path or arbitrary SQL.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_graph import (
    CausalGraphError,
    CausalGraphStore,
    _causal_templates,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import EventClass
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientSQLError,
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import sample_binding, sample_contract
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for causal-federation repository tests",
)


def _open_store(
    tmp_path: Path,
) -> tuple[CausalGraphStore, contracts.FederationBinding, str]:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:causal-graph-migration")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:causal-graph",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = CausalGraphStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    request = sample_request(
        binding=binding,
        maximum_supervisors=2,
        maximum_subagents=2,
    )
    policy = sample_policy(
        binding,
        maximum_supervisors=2,
        maximum_subagents=2,
        maximum_concurrent_subagents=2,
    )
    identity, _receipt = _create(store, request=request, policy=policy)
    return store, binding, identity.record_id


def _node(
    binding: contracts.FederationBinding,
    *,
    record_id: str,
    level: contracts.CausalLevel,
    subject_ref: str,
    node_type: str = "symbol",
) -> contracts.CausalNode:
    node = sample_contract(contracts.CausalNode)
    assert isinstance(node, contracts.CausalNode)
    return replace(
        node,
        record_id=record_id,
        binding=binding,
        level=level,
        node_type=node_type,
        subject_ref=subject_ref,
    )


def _evidence(
    binding: contracts.FederationBinding,
    *,
    record_id: str,
    evidence_ref: str,
    kind: contracts.CausalEvidenceKind = contracts.CausalEvidenceKind.EXACT_STATIC_DEPENDENCY,
    authoritative: bool = True,
) -> contracts.CausalEvidence:
    evidence = sample_contract(contracts.CausalEvidence)
    assert isinstance(evidence, contracts.CausalEvidence)
    return replace(
        evidence,
        record_id=record_id,
        binding=binding,
        evidence_kind=kind,
        evidence_ref=evidence_ref,
        authoritative=authoritative,
    )


def _edge(
    binding: contracts.FederationBinding,
    *,
    record_id: str,
    source_node_id: str,
    target_node_id: str,
    evidence_refs: tuple[str, ...],
    kind: contracts.CausalEdgeKind = contracts.CausalEdgeKind.DEPENDS_ON,
    nomination_only: bool = False,
) -> contracts.CausalEdge:
    edge = sample_contract(contracts.CausalEdge)
    assert isinstance(edge, contracts.CausalEdge)
    return replace(
        edge,
        record_id=record_id,
        binding=binding,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_kind=kind,
        evidence_refs=evidence_refs,
        nomination_only=nomination_only,
    )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(CausalGraphError, match="database path"):
        CausalGraphStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


def test_sealed_catalog_contains_causal_templates_and_rejects_sql(
    tmp_path: Path,
) -> None:
    store, _binding, _federation_id = _open_store(tmp_path)
    names = set(store.statement_catalog)
    assert {item.name for item in _causal_templates()}.issubset(names)
    with pytest.raises(QuackClientSQLError):
        store._client.execute("select 1", {})  # type: ignore[misc]


def test_records_multilevel_nodes_edges_and_exact_evidence(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    source = _node(
        binding,
        record_id="node:source",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:source",
    )
    target = _node(
        binding,
        record_id="node:target",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:target",
    )
    first = store.record_node(
        source,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:node-source",
    )
    second = store.record_node(
        target,
        federation_id=federation_id,
        expected_graph_revision=first.graph_revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:node-target",
    )
    evidence = _evidence(
        binding,
        record_id="evidence:exact",
        evidence_ref="artifact:exact-dep",
    )
    third = store.record_evidence(
        evidence,
        federation_id=federation_id,
        expected_graph_revision=second.graph_revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:evidence-exact",
    )
    edge = _edge(
        binding,
        record_id="edge:depends",
        source_node_id=source.record_id,
        target_node_id=target.record_id,
        evidence_refs=(evidence.record_id,),
    )
    committed = store.record_edge(
        edge,
        federation_id=federation_id,
        expected_graph_revision=third.graph_revision,
        idempotency_key="idempotency:edge-depends",
    )
    snapshot = store.snapshot(tenant_id=binding.tenant_id, federation_id=federation_id)
    assert committed.graph_revision == 5
    assert snapshot.graph_revision == 5
    assert {item.record_id for item in snapshot.nodes} == {
        source.record_id,
        target.record_id,
    }
    assert snapshot.edges[0].source_node_id == source.record_id
    assert snapshot.edges[0].target_node_id == target.record_id
    assert snapshot.edges[0].evidence_refs == (evidence.record_id,)
    assert snapshot.evidence[0].authoritative is True
    assert store.graph_revision(
        tenant_id=binding.tenant_id, federation_id=federation_id
    ) == 5


def test_stale_graph_revision_is_rejected(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    node = _node(
        binding,
        record_id="node:one",
        level=contracts.CausalLevel.L2_WORK,
        subject_ref="task:one",
        node_type="task",
    )
    store.record_node(
        node,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:node-one",
    )
    with pytest.raises(TransactionError, match="epoch does not match"):
        store.record_node(
            _node(
                binding,
                record_id="node:two",
                level=contracts.CausalLevel.L2_WORK,
                subject_ref="task:two",
                node_type="task",
            ),
            federation_id=federation_id,
            expected_graph_revision=1,
            owner_id="owner:casf-013",
            source_root="source:casf-013",
            idempotency_key="idempotency:node-two",
        )


def test_duplicate_subject_at_the_same_level_conflicts(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    first = store.record_node(
        _node(
            binding,
            record_id="node:alpha",
            level=contracts.CausalLevel.L0_RUNTIME,
            subject_ref="process:one",
            node_type="process",
        ),
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:node-alpha",
    )
    with pytest.raises(TransactionError, match="level and subject"):
        store.record_node(
            _node(
                binding,
                record_id="node:beta",
                level=contracts.CausalLevel.L0_RUNTIME,
                subject_ref="process:one",
                node_type="process",
            ),
            federation_id=federation_id,
            expected_graph_revision=first.graph_revision,
            owner_id="owner:casf-013",
            source_root="source:casf-013",
            idempotency_key="idempotency:node-beta",
        )


def test_idempotent_replay_does_not_advance_revision(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    node = _node(
        binding,
        record_id="node:replay",
        level=contracts.CausalLevel.L3_INTENT,
        subject_ref="goal:replay",
        node_type="goal",
    )
    first = store.record_node(
        node,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:replay",
    )
    replayed = store.record_node(
        node,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:replay",
    )
    assert replayed == first
    assert store.graph_revision(
        tenant_id=binding.tenant_id, federation_id=federation_id
    ) == first.graph_revision


def test_retrieval_nomination_cannot_authorize_an_edge(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    source = _node(
        binding,
        record_id="node:nom-source",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:nom-source",
    )
    target = _node(
        binding,
        record_id="node:nom-target",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:nom-target",
    )
    revision = store.record_node(
        source,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:nom-source",
    ).graph_revision
    revision = store.record_node(
        target,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:nom-target",
    ).graph_revision
    nomination = _evidence(
        binding,
        record_id="evidence:nomination",
        evidence_ref="retrieval:similar-symbol",
        kind=contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION,
        authoritative=False,
    )
    revision = store.record_evidence(
        nomination,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:nomination",
    ).graph_revision
    with pytest.raises(TransactionError, match="exact evidence"):
        store.record_edge(
            _edge(
                binding,
                record_id="edge:nominated-authority",
                source_node_id=source.record_id,
                target_node_id=target.record_id,
                evidence_refs=(nomination.record_id,),
                nomination_only=False,
            ),
            federation_id=federation_id,
            expected_graph_revision=revision,
            idempotency_key="idempotency:nominated-authority",
        )


def test_cross_level_causes_is_rejected_but_abstracts_may_be_adjacent(
    tmp_path: Path,
) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    low = _node(
        binding,
        record_id="node:low",
        level=contracts.CausalLevel.L1_CODE_ARTIFACT,
        subject_ref="symbol:low",
    )
    high = _node(
        binding,
        record_id="node:high",
        level=contracts.CausalLevel.L2_WORK,
        subject_ref="task:high",
        node_type="task",
    )
    revision = store.record_node(
        low,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:low",
    ).graph_revision
    revision = store.record_node(
        high,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:high",
    ).graph_revision
    exact = _evidence(
        binding,
        record_id="evidence:cross",
        evidence_ref="artifact:cross",
    )
    revision = store.record_evidence(
        exact,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:cross-evidence",
    ).graph_revision
    with pytest.raises(TransactionError, match="nominated levels"):
        store.record_edge(
            _edge(
                binding,
                record_id="edge:causes-cross",
                source_node_id=low.record_id,
                target_node_id=high.record_id,
                evidence_refs=(exact.record_id,),
                kind=contracts.CausalEdgeKind.CAUSES,
            ),
            federation_id=federation_id,
            expected_graph_revision=revision,
            idempotency_key="idempotency:causes-cross",
        )
    committed = store.record_edge(
        _edge(
            binding,
            record_id="edge:abstracts",
            source_node_id=low.record_id,
            target_node_id=high.record_id,
            evidence_refs=(exact.record_id,),
            kind=contracts.CausalEdgeKind.ABSTRACTS,
        ),
        federation_id=federation_id,
        expected_graph_revision=revision,
        idempotency_key="idempotency:abstracts",
    )
    snapshot = store.snapshot(tenant_id=binding.tenant_id, federation_id=federation_id)
    assert committed.graph_revision == snapshot.graph_revision
    assert snapshot.edges[0].edge_kind is contracts.CausalEdgeKind.ABSTRACTS


def test_cycles_require_an_explicit_fixed_point_group(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    left = _node(
        binding,
        record_id="node:left",
        level=contracts.CausalLevel.L2_WORK,
        subject_ref="task:left",
        node_type="task",
    )
    right = _node(
        binding,
        record_id="node:right",
        level=contracts.CausalLevel.L2_WORK,
        subject_ref="task:right",
        node_type="task",
    )
    revision = store.record_node(
        left,
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:left",
    ).graph_revision
    revision = store.record_node(
        right,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:right",
    ).graph_revision
    forward_evidence = _evidence(
        binding,
        record_id="evidence:forward",
        evidence_ref="artifact:forward",
    )
    reverse_evidence = _evidence(
        binding,
        record_id="evidence:reverse",
        evidence_ref="artifact:reverse",
    )
    revision = store.record_evidence(
        forward_evidence,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:forward-evidence",
    ).graph_revision
    revision = store.record_evidence(
        reverse_evidence,
        federation_id=federation_id,
        expected_graph_revision=revision,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:reverse-evidence",
    ).graph_revision
    revision = store.record_edge(
        _edge(
            binding,
            record_id="edge:forward",
            source_node_id=left.record_id,
            target_node_id=right.record_id,
            evidence_refs=(forward_evidence.record_id,),
        ),
        federation_id=federation_id,
        expected_graph_revision=revision,
        idempotency_key="idempotency:forward",
    ).graph_revision
    reverse = _edge(
        binding,
        record_id="edge:reverse",
        source_node_id=right.record_id,
        target_node_id=left.record_id,
        evidence_refs=(reverse_evidence.record_id,),
    )
    with pytest.raises(TransactionError, match="fixed-point group"):
        store.record_edge(
            reverse,
            federation_id=federation_id,
            expected_graph_revision=revision,
            idempotency_key="idempotency:reverse-denied",
        )
    committed = store.record_edge(
        reverse,
        federation_id=federation_id,
        expected_graph_revision=revision,
        idempotency_key="idempotency:reverse-admitted",
        fixed_point_group_id="fixed-point:left-right",
    )
    snapshot = store.snapshot(tenant_id=binding.tenant_id, federation_id=federation_id)
    assert committed.graph_revision == snapshot.graph_revision
    assert snapshot.cycle_group_ids == ("fixed-point:left-right",)
    assert {item.record_id for item in snapshot.edges} == {
        "edge:forward",
        "edge:reverse",
    }


def test_causal_graph_changed_event_is_appended_with_outbox(tmp_path: Path) -> None:
    store, binding, federation_id = _open_store(tmp_path)
    commit = store.record_node(
        _node(
            binding,
            record_id="node:evented",
            level=contracts.CausalLevel.L4_FEDERATION,
            subject_ref="federation:evented",
            node_type="federation",
        ),
        federation_id=federation_id,
        expected_graph_revision=1,
        owner_id="owner:casf-013",
        source_root="source:casf-013",
        idempotency_key="idempotency:evented",
    )
    assert commit.event_id.startswith("event:")
    assert commit.outbox_id.startswith("outbox:")
    assert commit.event_global_sequence >= 2
    store._client.close()
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    database = tmp_path / "control.duckdb"
    with open_duckdb_connection(database) as connection:
        row = connection.execute(
            """
            SELECT event_type, causal_graph_revision
            FROM domain_events
            WHERE event_id = ?
            """,
            [commit.event_id],
        ).fetchone()
    assert row is not None
    assert row[0] == EventClass.CAUSAL_GRAPH_CHANGED.value
    assert int(row[1]) == commit.graph_revision


def test_missing_federation_cannot_mint_causal_authority(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    install_control_plane_schema(database, owner_id="owner:causal-graph-migration")
    client = open_embedded_client(
        database,
        owner_id="owner:causal-graph",
        seed_generation=True,
    )
    store = CausalGraphStore(client)
    binding = sample_binding()
    with pytest.raises(TransactionError, match="federation is required"):
        store.record_node(
            _node(
                binding,
                record_id="node:orphan",
                level=contracts.CausalLevel.L1_CODE_ARTIFACT,
                subject_ref="symbol:orphan",
            ),
            federation_id="federation:missing",
            expected_graph_revision=1,
            owner_id="owner:casf-013",
            source_root="source:casf-013",
            idempotency_key="idempotency:orphan",
        )
