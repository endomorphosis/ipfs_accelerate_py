"""Hermetic tests for CASF causal frontier compilation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_frontier import (
    CausalFrontierAuthorityError,
    CausalFrontierError,
    CausalFrontierStore,
    FrontierSubject,
    IndependenceAdmission,
    compile_frontier,
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
from test.api.causal_federation.test_contracts import sample_binding, sample_contract
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def _node(record_id: str, subject_ref: str) -> contracts.CausalNode:
    node = sample_contract(contracts.CausalNode)
    assert isinstance(node, contracts.CausalNode)
    return replace(node, record_id=record_id, subject_ref=subject_ref)


def _edge(
    source: str,
    target: str,
    *,
    nomination_only: bool = False,
    kind: contracts.CausalEdgeKind = contracts.CausalEdgeKind.CAUSES,
    record_id: str | None = None,
) -> contracts.CausalEdge:
    edge = sample_contract(contracts.CausalEdge)
    assert isinstance(edge, contracts.CausalEdge)
    return replace(
        edge,
        record_id=record_id or f"edge:{source}:{target}",
        source_node_id=source,
        target_node_id=target,
        edge_kind=kind,
        nomination_only=nomination_only,
        evidence_refs=("evidence:exact",) if not nomination_only else ("evidence:nomination",),
    )


def _subject(supervisor_id: str, node_id: str) -> FrontierSubject:
    return FrontierSubject(supervisor_id=supervisor_id, node_id=node_id)


def _compile(
    nodes: tuple[contracts.CausalNode, ...],
    edges: tuple[contracts.CausalEdge, ...],
    subjects: tuple[FrontierSubject, ...],
    *,
    changed: tuple[str, ...] = ("node:changed",),
    independence: tuple[IndependenceAdmission, ...] = (),
    admitted_projection_edge_ids: tuple[str, ...] = (),
):
    return compile_frontier(
        event_id="event:change",
        binding=sample_binding(),
        graph_revision=1,
        nodes=nodes,
        edges=edges,
        changed_fact_refs=changed,
        subjects=subjects,
        independence=independence,
        admitted_projection_edge_ids=admitted_projection_edge_ids,
    )


def test_exact_descendants_must_wake() -> None:
    changed = _node("node:changed", "symbol:changed")
    child = _node("node:child", "symbol:child")
    other = _node("node:other", "symbol:other")
    compiled = _compile(
        (changed, child, other),
        (_edge("node:changed", "node:child"),),
        (
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:child", "node:child"),
            _subject("supervisor:other", "node:other"),
        ),
    )
    assert compiled.must_wake == ("supervisor:changed", "supervisor:child")
    assert compiled.may_wake == ("supervisor:other",)
    assert compiled.do_not_wake == ()


def test_unknown_nodes_widen_instead_of_sleeping() -> None:
    changed = _node("node:changed", "symbol:changed")
    other = _node("node:other", "symbol:other")
    compiled = _compile(
        (changed, other),
        (),
        (
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:other", "node:other"),
        ),
    )
    assert compiled.must_wake == ("supervisor:changed",)
    assert compiled.may_wake == ("supervisor:other",)
    assert compiled.do_not_wake == ()


def test_nomination_only_edges_widen_to_may_wake() -> None:
    changed = _node("node:changed", "symbol:changed")
    similar = _node("node:similar", "symbol:similar")
    compiled = _compile(
        (changed, similar),
        (_edge("node:changed", "node:similar", nomination_only=True),),
        (
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:similar", "node:similar"),
        ),
    )
    assert compiled.must_wake == ("supervisor:changed",)
    assert compiled.may_wake == ("supervisor:similar",)


def test_admitted_independence_is_required_for_do_not_wake() -> None:
    changed = _node("node:changed", "symbol:changed")
    independent = _node("node:independent", "symbol:independent")
    compiled = _compile(
        (changed, independent),
        (),
        (
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:idle", "node:independent"),
        ),
        independence=(
            IndependenceAdmission(
                subject=_subject("supervisor:idle", "node:independent"),
                evidence_refs=("evidence:independence",),
                authoritative=True,
            ),
        ),
    )
    assert compiled.do_not_wake == ("supervisor:idle",)
    assert compiled.must_wake == ("supervisor:changed",)


def test_retrieval_cannot_prove_independence() -> None:
    with pytest.raises(CausalFrontierAuthorityError, match="cannot prove independence"):
        IndependenceAdmission(
            subject=_subject("supervisor:idle", "node:independent"),
            evidence_refs=("evidence:vector",),
            authoritative=False,
        )


def test_independence_cannot_suppress_exact_descendants() -> None:
    changed = _node("node:changed", "symbol:changed")
    child = _node("node:child", "symbol:child")
    with pytest.raises(CausalFrontierAuthorityError, match="exact causal descendant"):
        _compile(
            (changed, child),
            (_edge("node:changed", "node:child"),),
            (
                _subject("supervisor:changed", "node:changed"),
                _subject("supervisor:child", "node:child"),
            ),
            independence=(
                IndependenceAdmission(
                    subject=_subject("supervisor:child", "node:child"),
                    evidence_refs=("evidence:independence",),
                    authoritative=True,
                ),
            ),
        )


def test_unadmitted_abstraction_projection_widens() -> None:
    low = _node("node:changed", "symbol:low")
    high = _node("node:high", "symbol:high")
    projection = _edge(
        "node:changed",
        "node:high",
        kind=contracts.CausalEdgeKind.ABSTRACTS,
        record_id="edge:abstracts",
    )
    widened = _compile(
        (low, high),
        (projection,),
        (
            _subject("supervisor:low", "node:changed"),
            _subject("supervisor:high", "node:high"),
        ),
    )
    assert widened.must_wake == ("supervisor:low",)
    assert widened.may_wake == ("supervisor:high",)
    projected = _compile(
        (low, high),
        (projection,),
        (
            _subject("supervisor:low", "node:changed"),
            _subject("supervisor:high", "node:high"),
        ),
        admitted_projection_edge_ids=("edge:abstracts",),
    )
    assert projected.must_wake == ("supervisor:high", "supervisor:low")


def test_frontier_is_complete_for_every_subject() -> None:
    changed = _node("node:changed", "symbol:changed")
    compiled = _compile(
        (changed,),
        (),
        (_subject("supervisor:changed", "node:changed"),),
    )
    assert {item.supervisor_id for item in compiled.entries} == {"supervisor:changed"}
    assert compiled.must_wake + compiled.may_wake + compiled.do_not_wake == (
        "supervisor:changed",
    )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(CausalFrontierError, match="database path"):
        CausalFrontierStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for frontier store")
def test_store_persists_compiled_frontier(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:frontier-migration")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:frontier",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = CausalFrontierStore(client)
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
    changed = _node("node:changed", "symbol:changed")
    idle = _node("node:idle", "symbol:idle")
    compiled = compile_frontier(
        event_id="event:change",
        binding=binding,
        graph_revision=1,
        nodes=(changed, idle),
        edges=(),
        changed_fact_refs=("node:changed",),
        subjects=(
            _subject("supervisor:changed", "node:changed"),
            _subject("supervisor:idle", "node:idle"),
        ),
        independence=(
            IndependenceAdmission(
                subject=_subject("supervisor:idle", "node:idle"),
                evidence_refs=("evidence:independence",),
                authoritative=True,
            ),
        ),
    )
    commit = store.record_frontier(
        compiled,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=store.graph_revision(
            tenant_id=binding.tenant_id, federation_id=identity.record_id
        ),
        idempotency_key="idempotency:frontier",
    )
    loaded = store.load_frontier(
        frontier_id="frontier:" + compiled.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert commit.graph_revision >= 2
    assert int(loaded["frontier"]["must_wake_count"]) == 1
    assert int(loaded["frontier"]["do_not_wake_count"]) == 1
    dispositions = {row["subject_ref"]: row["disposition"] for row in loaded["members"]}
    assert dispositions["supervisor:changed"] == "must_wake"
    assert dispositions["supervisor:idle"] == "do_not_wake"
