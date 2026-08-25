"""Hermetic tests for CASF DuckLake projection recovery and security receipts."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.ducklake_projection import (
    DuckLakeCapability,
    DuckLakeProjectionAuthorityError,
    DuckLakeProjectionError,
    DuckLakeProjectionRecovery,
    DuckLakeProjectionStore,
    ProjectionPartition,
    ProjectionSecurityContext,
    SourceRange,
    project_event_range,
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


def _range(**overrides: object) -> SourceRange:
    values: dict[str, object] = {
        "from_watermark": 1,
        "to_watermark": 2,
        "source_root": "source:events",
        "tree_id": sample_binding().repository_tree_ids[0],
        "event_count": 2,
    }
    values.update(overrides)
    return SourceRange(**values)  # type: ignore[arg-type]


def _partition(
    *,
    partition_id: str,
    from_watermark: int,
    to_watermark: int,
    file_ref: str,
) -> ProjectionPartition:
    return ProjectionPartition(
        partition_id=partition_id,
        from_watermark=from_watermark,
        to_watermark=to_watermark,
        file_ref=file_ref,
        event_count=to_watermark - from_watermark + 1,
        byte_size=64,
    )


def _security(**overrides: object) -> ProjectionSecurityContext:
    values: dict[str, object] = {
        "tenant_id": sample_binding().tenant_id,
        "schema_revision": 2,
        "expected_schema_revision": 2,
        "payload": {"event_class": "SUPERVISOR_HEALTH_CHANGED"},
    }
    values.update(overrides)
    return ProjectionSecurityContext(**values)  # type: ignore[arg-type]


def _sealed():
    return project_event_range(
        _range(),
        (_partition(partition_id="partition:one", from_watermark=1, to_watermark=2, file_ref="partition:a"),),
        binding=sample_binding(),
        capability=DuckLakeCapability(available=True),
        expected_fence=1,
        fencing_epoch=1,
    )


def _recover(**kwargs: object):
    recovery = DuckLakeProjectionRecovery()
    values: dict[str, object] = {
        "remaining_range": _range(from_watermark=3, to_watermark=4, event_count=2),
        "remaining_partitions": (
            _partition(
                partition_id="partition:two",
                from_watermark=3,
                to_watermark=4,
                file_ref="partition:b",
            ),
        ),
        "binding": sample_binding(),
        "capability": DuckLakeCapability(available=True),
        "security": _security(),
        "expected_fence": 1,
        "fencing_epoch": 1,
        "sealed_receipt": _sealed(),
    }
    values.update(kwargs)
    return recovery.recover(**values)  # type: ignore[arg-type]


def test_interruption_resumes_from_sealed_cursor_without_rewrite() -> None:
    sealed = _sealed()
    receipt = _recover(sealed_receipt=sealed)
    assert receipt.rewritten is False
    assert receipt.authoritative is False
    assert receipt.status == "current"
    assert receipt.preserved_partition_ids == sealed.partition_ids
    assert receipt.recovered_partition_ids == ("partition:two",)
    assert receipt.recovered_from_watermark == 3
    assert receipt.recovered_to_watermark == 4


def test_sealed_range_cannot_be_rewritten() -> None:
    with pytest.raises(DuckLakeProjectionError, match="cannot be rewritten"):
        _recover(remaining_range=_range(from_watermark=2, to_watermark=3, event_count=2))


def test_redaction_rejects_secret_payloads() -> None:
    with pytest.raises(DuckLakeProjectionAuthorityError, match="redacted"):
        _security(payload={"note": "Bearer abcdefghijklmnop"})
    with pytest.raises(DuckLakeProjectionAuthorityError, match="secret keys"):
        _security(payload={"access_token": "handle:ok"})


def test_tenant_isolation_fails_closed() -> None:
    with pytest.raises(DuckLakeProjectionAuthorityError, match="tenant is not isolated"):
        _recover(security=_security(tenant_id="tenant:other"))


def test_schema_evolution_is_typed_lag_not_control_plane_failure() -> None:
    receipt = _recover(security=_security(schema_revision=3, expected_schema_revision=2))
    assert receipt.status == "lagging"
    assert receipt.authoritative is False
    assert receipt.recovered_partition_ids == ()
    assert receipt.preserved_partition_ids == _sealed().partition_ids


def test_unavailable_ducklake_does_not_block_the_control_plane() -> None:
    receipt = _recover(capability=DuckLakeCapability(available=False, lagging=True))
    assert receipt.status == "lagging"
    assert receipt.authoritative is False
    absent = _recover(capability=DuckLakeCapability(available=False))
    assert absent.status == "unavailable"


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for recovery persistence")
def test_store_records_projection_recovery_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:ducklake-recovery")
    assert report.to_version >= 2
    client = open_embedded_client(
        database,
        owner_id="owner:ducklake-recovery",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = DuckLakeProjectionStore(client)
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
    sealed = project_event_range(
        SourceRange(
            from_watermark=1,
            to_watermark=2,
            source_root="source:events",
            tree_id=binding.repository_tree_ids[0],
            event_count=2,
        ),
        (
            _partition(
                partition_id="partition:one",
                from_watermark=1,
                to_watermark=2,
                file_ref="partition:a",
            ),
        ),
        binding=binding,
        capability=DuckLakeCapability(available=True),
        expected_fence=1,
        fencing_epoch=1,
    )
    receipt = DuckLakeProjectionRecovery().recover(
        remaining_range=SourceRange(
            from_watermark=3,
            to_watermark=4,
            source_root="source:events",
            tree_id=binding.repository_tree_ids[0],
            event_count=2,
        ),
        remaining_partitions=(
            _partition(
                partition_id="partition:two",
                from_watermark=3,
                to_watermark=4,
                file_ref="partition:b",
            ),
        ),
        binding=binding,
        capability=DuckLakeCapability(available=True),
        security=ProjectionSecurityContext(
            tenant_id=binding.tenant_id,
            schema_revision=2,
            expected_schema_revision=2,
            payload={"event_class": "SUPERVISOR_HEALTH_CHANGED"},
        ),
        expected_fence=1,
        fencing_epoch=1,
        sealed_receipt=sealed,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_recovery(
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:ducklake-recovery",
    )
    loaded = store.load_recovery(
        receipt_id="federation-receipt:" + receipt.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["receipt_kind"] == "ducklake_projection_recovery"
    assert loaded["content_ref"] == receipt.cid
