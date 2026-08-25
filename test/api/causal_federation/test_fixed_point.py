"""Hermetic tests for CASF conjunctive fixed-point detection."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.fixed_point import (
    FederationFixedPointDetector,
    FixedPointAuthorityError,
    FixedPointError,
    FixedPointObservation,
    FixedPointStore,
)
from ipfs_accelerate_py.agent_supervisor.federation.world_snapshot import (
    assemble_federation_world_snapshot,
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


def _snapshot(*, event_watermark: int = 7):
    binding = sample_binding()
    return assemble_federation_world_snapshot(
        binding=binding,
        event_watermark=event_watermark,
        task_population_ref="tasks:test",
        claim_population_ref="claims:test",
        merge_state_ref="merge:test",
        proof_state_ref="proof:test",
        causal_frontier_ref="frontier:test",
        graph_revision=binding.causal_graph_revision,
    )


def _observation(**overrides: object) -> FixedPointObservation:
    values: dict[str, object] = {
        "snapshot": _snapshot(),
        "event_watermark": 7,
        "fencing_epoch": 2,
    }
    values.update(overrides)
    return FixedPointObservation(**values)  # type: ignore[arg-type]


def _diagnose(observation: FixedPointObservation, **kwargs: object):
    detector = FederationFixedPointDetector()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "expected_fence": observation.fencing_epoch,
    }
    values.update(kwargs)
    return detector.diagnose(observation, **values)  # type: ignore[arg-type]


def _detect(observation: FixedPointObservation, **kwargs: object):
    detector = FederationFixedPointDetector()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "expected_fence": observation.fencing_epoch,
    }
    values.update(kwargs)
    return detector.detect(observation, **values)  # type: ignore[arg-type]


def test_conjunctive_predicate_admits_a_true_fixed_point() -> None:
    receipt = _detect(_observation())
    assert receipt.outcome == "fixed_point"
    assert receipt.event_watermark == 7
    assert receipt.outstanding_required_work == 0
    diagnostics = _diagnose(_observation())
    assert diagnostics.at_fixed_point is True
    assert diagnostics.false_quiet is False
    assert diagnostics.failed_predicates == ()


def test_outstanding_work_fails_the_conjunctive_predicate() -> None:
    observation = _observation(outstanding_required_work=3)
    diagnostics = _diagnose(observation)
    assert diagnostics.at_fixed_point is False
    assert "outstanding_required_work" in diagnostics.failed_predicates
    with pytest.raises(FixedPointAuthorityError, match="conjunctive fixed-point predicate"):
        _detect(observation)


def test_event_watermark_must_match_the_world_snapshot() -> None:
    observation = _observation(event_watermark=9)
    diagnostics = _diagnose(observation)
    assert "event_watermark" in diagnostics.failed_predicates
    with pytest.raises(FixedPointAuthorityError, match="event_watermark"):
        _detect(observation)


def test_quiet_queue_with_remaining_work_is_false_quiet() -> None:
    observation = _observation(board_quiet=True, outstanding_effects=1)
    diagnostics = _diagnose(observation)
    assert diagnostics.false_quiet is True
    assert "quiet_queue" in diagnostics.failed_predicates
    with pytest.raises(FixedPointAuthorityError, match="cannot complete federation work"):
        _detect(observation)


def test_board_status_cannot_complete_federation_work() -> None:
    observation = _observation(board_completed=True)
    diagnostics = _diagnose(observation)
    assert diagnostics.false_quiet is True
    with pytest.raises(FixedPointAuthorityError, match="cannot complete federation work"):
        _detect(observation)


def test_process_exit_cannot_complete_federation_work() -> None:
    observation = _observation(process_exited=True, claimed_complete=True)
    diagnostics = _diagnose(observation)
    assert "process_exit" in diagnostics.failed_predicates
    with pytest.raises(FixedPointAuthorityError, match="cannot complete federation work"):
        _detect(observation)


def test_open_proofs_pending_merges_and_recovery_block_fixed_point() -> None:
    observation = _observation(open_proofs=1, pending_merges=1, recovering_owners=1)
    diagnostics = _diagnose(observation)
    assert diagnostics.failed_predicates == (
        "open_proofs",
        "pending_merges",
        "recovering_owners",
    )
    with pytest.raises(FixedPointAuthorityError, match="open_proofs"):
        _detect(observation)


def test_stale_fence_fails_closed() -> None:
    with pytest.raises(FixedPointAuthorityError, match="fencing epoch is stale"):
        _diagnose(_observation(), expected_fence=1)


def test_ducklake_cannot_admit_a_fixed_point() -> None:
    with pytest.raises(FixedPointAuthorityError, match="DuckLake cannot admit"):
        _detect(_observation(), ducklake_receipt={"fixed_points": True})
    with pytest.raises(FixedPointAuthorityError, match="DuckLake cannot admit"):
        _detect(_observation(), ducklake_receipt={"authoritative": True})


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(FixedPointError, match="database path"):
        FixedPointStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for fixed-point persistence")
def test_store_records_fixed_point_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:fixed-point")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:fixed-point",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = FixedPointStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _created = _create(
        store,
        request=sample_request(binding=binding, maximum_supervisors=2, maximum_subagents=2),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    snapshot = assemble_federation_world_snapshot(
        binding=binding,
        event_watermark=4,
        task_population_ref="tasks:test",
        claim_population_ref="claims:test",
        merge_state_ref="merge:test",
        proof_state_ref="proof:test",
        causal_frontier_ref="frontier:test",
        graph_revision=binding.causal_graph_revision,
    )
    observation = FixedPointObservation(
        snapshot=snapshot,
        event_watermark=4,
        fencing_epoch=1,
    )
    detector = FederationFixedPointDetector()
    receipt = detector.detect(
        observation,
        binding=binding,
        expected_fence=1,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_fixed_point(
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:fixed-point",
    )
    loaded = store.load_fixed_point(
        receipt_id="federation-receipt:" + receipt.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["receipt_kind"] == "fixed_point"
    assert loaded["event_watermark"] == 4
    assert loaded["content_ref"] == receipt.cid
