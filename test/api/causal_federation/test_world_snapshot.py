"""Hermetic tests for CASF federation world snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_frontier import (
    FrontierSubject,
    IndependenceAdmission,
    compile_frontier,
)
from ipfs_accelerate_py.agent_supervisor.federation.world_snapshot import (
    WorldSnapshotAuthorityError,
    WorldSnapshotError,
    WorldSnapshotStore,
    assemble_federation_world_snapshot,
    snapshot_from_frontier,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_builder import (
    WorldSnapshotAdmissionError,
    build_world_snapshot,
    current_authority_fixture,
    project_casf_world_inputs,
    refuse_ducklake_world_authority,
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
from test.api.causal_federation.test_causal_frontier import _node, _subject
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def _assemble(**overrides: object) -> contracts.FederationWorldSnapshot:
    binding = sample_binding()
    values: dict[str, object] = {
        "binding": binding,
        "event_watermark": 7,
        "task_population_ref": "tasks:test",
        "claim_population_ref": "claims:test",
        "merge_state_ref": "merge:test",
        "proof_state_ref": "proof:test",
        "causal_frontier_ref": "frontier:test",
        "graph_revision": binding.causal_graph_revision,
    }
    values.update(overrides)
    return assemble_federation_world_snapshot(**values)  # type: ignore[arg-type]


def test_snapshot_is_tree_bound_to_federation_semantic_roots() -> None:
    snapshot = _assemble()
    assert snapshot.semantic_roots == sample_binding().semantic_state_roots
    assert snapshot.causal_frontier_ref == "frontier:test"
    decoded = contracts.FederationWorldSnapshot.from_dict(snapshot.to_dict())
    assert decoded == snapshot


def test_divergent_semantic_roots_cannot_admit_a_snapshot() -> None:
    with pytest.raises(WorldSnapshotAuthorityError, match="tree-bound"):
        _assemble(semantic_roots=("semantic:other-tree",))


def test_stale_graph_revision_cannot_admit_a_snapshot() -> None:
    with pytest.raises(WorldSnapshotAuthorityError, match="stale"):
        _assemble(graph_revision=99)


def test_ducklake_cannot_admit_world_snapshot_authority() -> None:
    with pytest.raises(WorldSnapshotAdmissionError, match="DuckLake cannot admit"):
        refuse_ducklake_world_authority({"authoritative": True, "status": "current"})
    with pytest.raises(WorldSnapshotAuthorityError, match="DuckLake cannot admit"):
        _assemble(ducklake_receipt={"authoritative": True, "valid": True})


def test_builder_projection_is_observational_and_non_authoritative() -> None:
    result = build_world_snapshot(current_authority_fixture())
    projected = project_casf_world_inputs(result)
    assert projected["schedulable"] is True
    assert projected["ducklake_authoritative"] is False
    assert projected["datasets_semantic_state_root"].startswith("sha256:")
    binding = sample_binding(
        semantic_state_roots=(projected["datasets_semantic_state_root"],)
    )
    snapshot = assemble_federation_world_snapshot(
        binding=binding,
        event_watermark=3,
        task_population_ref=projected["task_population"],
        claim_population_ref="claims:casf-017",
        merge_state_ref="merge:casf-017",
        proof_state_ref="proof:casf-017",
        causal_frontier_ref="frontier:casf-017",
        graph_revision=binding.causal_graph_revision,
        builder_result=result,
    )
    assert snapshot.semantic_roots == (projected["datasets_semantic_state_root"],)
    stale = build_world_snapshot(
        {
            **current_authority_fixture(),
            "accepted_plan_root": {
                **current_authority_fixture()["accepted_plan_root"],
                "status": "stale",
            },
        }
    )
    with pytest.raises(WorldSnapshotAuthorityError, match="unschedulable"):
        assemble_federation_world_snapshot(
            binding=binding,
            event_watermark=3,
            task_population_ref=projected["task_population"],
            claim_population_ref="claims:casf-017",
            merge_state_ref="merge:casf-017",
            proof_state_ref="proof:casf-017",
            causal_frontier_ref="frontier:casf-017",
            builder_result=stale,
        )


def test_snapshot_binds_compiled_frontier_identity() -> None:
    changed = _node("node:changed", "symbol:changed")
    idle = _node("node:idle", "symbol:idle")
    compiled = compile_frontier(
        event_id="event:change",
        binding=sample_binding(),
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
                subject=FrontierSubject(
                    supervisor_id="supervisor:idle", node_id="node:idle"
                ),
                evidence_refs=("evidence:independence",),
                authoritative=True,
            ),
        ),
    )
    snapshot = snapshot_from_frontier(
        compiled,
        binding=sample_binding(),
        event_watermark=4,
        task_population_ref="tasks:casf-017",
        claim_population_ref="claims:casf-017",
        merge_state_ref="merge:casf-017",
        proof_state_ref="proof:casf-017",
    )
    assert snapshot.causal_frontier_ref == "frontier:" + compiled.cid
    assert snapshot.binding.causal_graph_revision == compiled.graph_revision


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(WorldSnapshotError, match="database path"):
        WorldSnapshotStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for snapshot store")
def test_store_persists_tree_bound_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:world-snapshot")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:world-snapshot",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = WorldSnapshotStore(client)
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
    snapshot = assemble_federation_world_snapshot(
        binding=binding,
        event_watermark=11,
        task_population_ref="tasks:casf-017",
        claim_population_ref="claims:casf-017",
        merge_state_ref="merge:casf-017",
        proof_state_ref="proof:casf-017",
        causal_frontier_ref="frontier:casf-017",
        graph_revision=1,
        record_id="world-snapshot:casf-017",
    )
    commit = store.record_snapshot(
        snapshot,
        federation_id=identity.record_id,
        expected_graph_revision=store.graph_revision(
            tenant_id=binding.tenant_id, federation_id=identity.record_id
        ),
        idempotency_key="idempotency:world-snapshot",
        owner_id="owner:casf-017",
        source_root="source:casf-017",
    )
    loaded = store.load_snapshot(
        snapshot_id=snapshot.record_id,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert commit.graph_revision >= 2
    assert loaded["content_ref"] == snapshot.cid
    assert loaded["semantic_state_root"] == snapshot.semantic_roots[0]
    assert int(loaded["event_watermark"]) == 11
    assert loaded["freshness_state"] == "current"
