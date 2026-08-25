"""Hermetic tests for CASF non-authoritative DuckLake history projection."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import FederationBoundsError
from ipfs_accelerate_py.agent_supervisor.federation.ducklake_projection import (
    MAX_PARTITION_EVENTS,
    DuckLakeCapability,
    DuckLakeProjectionAuthorityError,
    DuckLakeProjectionError,
    DuckLakeProjectionStore,
    DuckLakeProjectionWorker,
    ProjectionCursor,
    ProjectionPartition,
    ProjectionReceipt,
    SourceRange,
    project_event_range,
    projection_establishes_authority,
    projection_establishes_completion,
    source_range_checksum,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection import (
    project_history,
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
        "to_watermark": 4,
        "source_root": "source:events",
        "tree_id": sample_binding().repository_tree_ids[0],
        "event_count": 4,
    }
    values.update(overrides)
    return SourceRange(**values)  # type: ignore[arg-type]


def _partition(
    *,
    partition_id: str = "partition:one",
    from_watermark: int = 1,
    to_watermark: int = 4,
    file_ref: str = "partition:events-1-4",
    byte_size: int = 128,
) -> ProjectionPartition:
    return ProjectionPartition(
        partition_id=partition_id,
        from_watermark=from_watermark,
        to_watermark=to_watermark,
        file_ref=file_ref,
        event_count=to_watermark - from_watermark + 1,
        byte_size=byte_size,
    )


def _project(source_range: SourceRange, partitions, **kwargs: object) -> ProjectionReceipt:
    worker = DuckLakeProjectionWorker()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "capability": DuckLakeCapability(available=True),
        "expected_fence": 1,
        "fencing_epoch": 1,
    }
    values.update(kwargs)
    return worker.project(source_range, partitions, **values)  # type: ignore[arg-type]


def test_projection_never_establishes_authority_or_completion() -> None:
    assert projection_establishes_authority() is False
    assert projection_establishes_completion() is False
    assert project_history({"receipt": True})["authoritative"] is False
    receipt = _project(_range(), (_partition(),))
    assert receipt.authoritative is False
    assert receipt.status == "current"


def test_idempotent_projection_replays_the_same_source_range_checksum() -> None:
    source = _range()
    partitions = (_partition(),)
    first = _project(source, partitions)
    second = _project(source, partitions, previous_receipt=first)
    assert second is first
    third = _project(source, partitions)
    assert third.cid == first.cid
    assert third.source_checksum == source_range_checksum(source, partitions)


def test_source_range_checksum_mismatch_fails_closed() -> None:
    source = _range(source_checksum="checksum:other")
    with pytest.raises(DuckLakeProjectionError, match="source-range checksum"):
        _project(source, (_partition(),))


def test_partitioned_files_are_bounded() -> None:
    with pytest.raises(FederationBoundsError):
        ProjectionPartition(
            partition_id="partition:huge",
            from_watermark=1,
            to_watermark=MAX_PARTITION_EVENTS + 1,
            file_ref="partition:too-big",
            event_count=MAX_PARTITION_EVENTS + 1,
            byte_size=128,
        )
    with pytest.raises(FederationBoundsError):
        ProjectionPartition(
            partition_id="partition:bytes",
            from_watermark=1,
            to_watermark=1,
            file_ref="partition:bytes",
            event_count=1,
            byte_size=9 * 1024 * 1024,
        )


def test_filesystem_partition_path_fails_closed() -> None:
    with pytest.raises(DuckLakeProjectionAuthorityError, match="filesystem paths"):
        ProjectionPartition(
            partition_id="partition:path",
            from_watermark=1,
            to_watermark=1,
            file_ref="/tmp/history.parquet",
            event_count=1,
            byte_size=16,
        )


def test_partition_gap_fails_closed() -> None:
    with pytest.raises(DuckLakeProjectionError, match="gap or overlap"):
        _project(
            _range(),
            (
                _partition(to_watermark=2, file_ref="partition:a"),
                _partition(
                    partition_id="partition:two",
                    from_watermark=4,
                    to_watermark=4,
                    file_ref="partition:b",
                ),
            ),
        )


def test_unavailable_ducklake_does_not_block_the_control_plane() -> None:
    receipt = _project(
        _range(),
        (),
        capability=DuckLakeCapability(available=False),
    )
    assert receipt.status == "unavailable"
    assert receipt.authoritative is False
    assert receipt.partition_ids == ()
    lagging = _project(
        _range(),
        (),
        capability=DuckLakeCapability(available=False, lagging=True),
    )
    assert lagging.status == "lagging"


def test_authoritative_receipt_cannot_be_constructed() -> None:
    with pytest.raises(DuckLakeProjectionAuthorityError, match="cannot admit"):
        ProjectionReceipt(
            status="current",
            source_root="source:events",
            tree_id=sample_binding().repository_tree_ids[0],
            from_watermark=1,
            to_watermark=1,
            source_checksum="checksum:one",
            cursor_watermark=1,
            partition_ids=("partition:one",),
            authoritative=True,
        )
    with pytest.raises(DuckLakeProjectionAuthorityError, match="cannot admit"):
        project_history({"receipt": True, "authoritative": True})


def test_contiguous_cursor_is_required_for_the_next_range() -> None:
    first = _project(_range(), (_partition(),))
    nxt = SourceRange(
        from_watermark=5,
        to_watermark=6,
        source_root="source:events",
        tree_id=sample_binding().repository_tree_ids[0],
        event_count=2,
    )
    receipt = _project(
        nxt,
        (
            _partition(
                partition_id="partition:two",
                from_watermark=5,
                to_watermark=6,
                file_ref="partition:events-5-6",
            ),
        ),
        previous_cursor=ProjectionCursor(
            source_root="source:events",
            watermark=first.cursor_watermark,
            partition_ordinal=1,
        ),
    )
    assert receipt.cursor_watermark == 6
    with pytest.raises(DuckLakeProjectionError, match="not contiguous"):
        _project(
            nxt,
            (
                _partition(
                    partition_id="partition:two",
                    from_watermark=5,
                    to_watermark=6,
                    file_ref="partition:events-5-6",
                ),
            ),
            previous_cursor=ProjectionCursor(
                source_root="source:events",
                watermark=2,
                partition_ordinal=1,
            ),
        )


def test_stale_fence_fails_closed() -> None:
    with pytest.raises(DuckLakeProjectionAuthorityError, match="fencing epoch is stale"):
        _project(_range(), (_partition(),), expected_fence=9)


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(DuckLakeProjectionError, match="database path"):
        DuckLakeProjectionStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for projection persistence")
def test_store_records_ducklake_projection_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:ducklake")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:ducklake",
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
    source = SourceRange(
        from_watermark=1,
        to_watermark=2,
        source_root="source:events",
        tree_id=binding.repository_tree_ids[0],
        event_count=2,
    )
    partition = ProjectionPartition(
        partition_id="partition:one",
        from_watermark=1,
        to_watermark=2,
        file_ref="partition:events-1-2",
        event_count=2,
        byte_size=64,
    )
    receipt = project_event_range(
        source,
        (partition,),
        binding=binding,
        capability=DuckLakeCapability(available=True),
        expected_fence=1,
        fencing_epoch=1,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_projection(
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:ducklake",
    )
    loaded = store.load_projection(
        receipt_id="federation-receipt:" + receipt.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["receipt_kind"] == "ducklake_projection"
    assert loaded["event_watermark"] == 2
    assert loaded["content_ref"] == receipt.cid
