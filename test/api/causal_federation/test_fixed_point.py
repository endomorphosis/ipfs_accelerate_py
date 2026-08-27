"""Hermetic tests for CASF conjunctive fixed-point detection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationContractError,
    UnknownNormativeFieldError,
)
from ipfs_accelerate_py.agent_supervisor.federation.fixed_point import (
    FederationFixedPointDetector,
    FixedPointAuthorityError,
    FixedPointError,
    FixedPointObservation,
    FixedPointReceipt,
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


def test_fixed_point_receipt_has_an_exact_canonical_wire_round_trip() -> None:
    receipt = _detect(_observation())

    wire = receipt.to_dict()

    assert wire == {
        "schema": FixedPointReceipt.SCHEMA,
        "world_snapshot_ref": receipt.world_snapshot_ref,
        "event_watermark": 7,
        "outstanding_required_work": 0,
        "fencing_epoch": 2,
        "outcome": "fixed_point",
        "evidence_refs": list(receipt.evidence_refs),
        "receipt_id": receipt.cid,
    }
    assert FixedPointReceipt.from_dict(wire) == receipt
    assert FixedPointReceipt.SCHEMA.endswith("fixed-point-receipt@2")
    assert receipt.to_dict(include_identity=False) == {
        key: value for key, value in wire.items() if key != "receipt_id"
    }


def test_fixed_point_receipt_identity_binds_evidence_refs() -> None:
    receipt = _detect(_observation())
    changed = replace(
        receipt,
        evidence_refs=(receipt.evidence_refs[0], "watermark:8"),
    )

    assert changed.cid != receipt.cid
    assert changed.to_dict()["receipt_id"] == changed.cid

    transplanted = changed.to_dict()
    transplanted["receipt_id"] = receipt.cid
    with pytest.raises(FederationContractError, match="identity mismatches"):
        FixedPointReceipt.from_dict(transplanted)


def test_fixed_point_receipt_decoder_rejects_noncanonical_wire_shapes() -> None:
    receipt = _detect(_observation())

    with pytest.raises(FederationContractError, match="exact JSON object"):
        FixedPointReceipt.from_dict(MappingProxyType(receipt.to_dict()))

    unknown = receipt.to_dict()
    unknown["caller_claimed_complete"] = True
    with pytest.raises(UnknownNormativeFieldError, match="unknown fields"):
        FixedPointReceipt.from_dict(unknown)

    missing = receipt.to_dict()
    missing.pop("event_watermark")
    with pytest.raises(FederationContractError, match="missing fields"):
        FixedPointReceipt.from_dict(missing)

    wrong_schema = receipt.to_dict()
    wrong_schema["schema"] = "foreign/fixed-point-receipt@1"
    with pytest.raises(FederationContractError, match="schema must equal"):
        FixedPointReceipt.from_dict(wrong_schema)

    tuple_encoded = receipt.to_dict()
    tuple_encoded["evidence_refs"] = tuple(receipt.evidence_refs)
    with pytest.raises(FederationContractError, match="canonical array"):
        FixedPointReceipt.from_dict(tuple_encoded)

    class _ListSubclass(list):
        pass

    subclass_encoded = receipt.to_dict()
    subclass_encoded["evidence_refs"] = _ListSubclass(receipt.evidence_refs)
    with pytest.raises(FederationContractError, match="canonical array"):
        FixedPointReceipt.from_dict(subclass_encoded)

    class _AlwaysEqual(str):
        def __eq__(self, _other: object) -> bool:
            return True

        __hash__ = str.__hash__

    adversarial_schema = receipt.to_dict()
    adversarial_schema["schema"] = _AlwaysEqual("legacy-or-foreign")
    with pytest.raises(FederationContractError, match="exact string"):
        FixedPointReceipt.from_dict(adversarial_schema)

    adversarial_identity = receipt.to_dict()
    adversarial_identity["receipt_id"] = _AlwaysEqual("forged")
    with pytest.raises(FederationContractError, match="identity mismatches"):
        FixedPointReceipt.from_dict(adversarial_identity)


def test_legacy_v1_fixed_point_receipt_is_audit_only() -> None:
    receipt = _detect(_observation())
    legacy = receipt.to_dict()
    legacy["schema"] = FixedPointReceipt.LEGACY_SCHEMA

    with pytest.raises(FixedPointAuthorityError, match="audit-only"):
        FixedPointReceipt.from_dict(legacy)


def test_fixed_point_receipt_decoder_rejects_invalid_semantics() -> None:
    receipt = _detect(_observation())

    duplicate_evidence = receipt.to_dict()
    duplicate_evidence["evidence_refs"] = [
        receipt.evidence_refs[0],
        receipt.evidence_refs[0],
    ]
    with pytest.raises(FederationContractError, match="contains duplicates"):
        FixedPointReceipt.from_dict(duplicate_evidence)

    nonzero_work = receipt.to_dict()
    nonzero_work["outstanding_required_work"] = 1
    with pytest.raises(FixedPointAuthorityError, match="outstanding required work"):
        FixedPointReceipt.from_dict(nonzero_work)

    boolean_watermark = receipt.to_dict()
    boolean_watermark["event_watermark"] = True
    with pytest.raises(FederationContractError, match="must be an integer"):
        FixedPointReceipt.from_dict(boolean_watermark)


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
