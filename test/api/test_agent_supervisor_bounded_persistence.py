from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor import artifact_store
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (
    ArtifactBlobIntegrityError,
    ArtifactOutcome,
    ArtifactPayloadTooLarge,
    ArtifactQuotaExceeded,
    ArtifactQuotaPolicy,
    BoundedArtifactStore,
    RetentionClass,
    enforce_projection_bound,
    enforce_receipt_bound,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    EventPayloadTooLarge,
    append_jsonl_event,
    event_log_manifest,
    read_jsonl_event_sources,
    rotate_event_log_if_needed,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts import (
    MAX_PROJECTION_BYTES,
    MAX_RECEIPT_BYTES,
)


def _quota(**overrides: int) -> ArtifactQuotaPolicy:
    values = {
        "max_bytes": 128 * 1024,
        "max_blobs": 16,
        "max_projections": 16,
        "max_blob_bytes": 64 * 1024,
        "compaction_batch_size": 32,
        "negative_ttl_seconds": 2,
        "inconclusive_ttl_seconds": 3,
    }
    values.update(overrides)
    return ArtifactQuotaPolicy(**values)


def test_receipts_and_routine_projections_have_hard_byte_bounds() -> None:
    assert len(enforce_receipt_bound({"value": "small"})) < MAX_RECEIPT_BYTES
    assert (
        len(enforce_projection_bound({"value": "small"}))
        < MAX_PROJECTION_BYTES
    )
    with pytest.raises(ArtifactPayloadTooLarge, match="receipt exceeds"):
        enforce_receipt_bound({"value": "x" * MAX_RECEIPT_BYTES})
    with pytest.raises(
        ArtifactPayloadTooLarge, match="routine projection exceeds"
    ):
        enforce_projection_bound({"value": "x" * MAX_PROJECTION_BYTES})


def test_large_bodies_and_nested_graphs_are_stored_once_as_shallow_refs(
    tmp_path: Path,
) -> None:
    store = BoundedArtifactStore(tmp_path / "artifacts", quotas=_quota())
    repeated = "decoded model result" * 100
    graph = {"nodes": [{"id": index} for index in range(20)]}

    reference = store.store_projection(
        {
            "summary": "bounded",
            "decoded_model_text": repeated,
            "source_body": repeated,
            "proof_trace": ["step one", "step two"],
            "checkpoint": {"cursor": 4},
            "nested_artifact_graph": graph,
        },
        projection_kind="stage_receipt",
    )
    projected = store.read_projection(reference)

    assert projected["summary"] == "bounded"
    assert repeated not in json.dumps(projected)
    assert projected["decoded_model_text"] == projected["source_body"]
    assert len(reference.artifact_references) == 4
    assert (
        store.read_blob(
            projected["decoded_model_text"]["artifact_ref"], decode=True
        )
        == repeated
    )
    assert store.metrics().deduplicated_blob_writes == 1

    recursive: dict[str, object] = {}
    recursive["source_body"] = recursive
    with pytest.raises(
        ArtifactBlobIntegrityError, match="canonical JSON|recursive"
    ):
        store.store_projection(recursive, projection_kind="receipt")
    with pytest.raises(
        ArtifactBlobIntegrityError, match="cannot recursively embed"
    ):
        store.store_projection(
            {
                "artifact_ref": {
                    "artifact_id": reference.artifact_id,
                    "digest": reference.digest,
                    "payload": {"duplicated": True},
                }
            }
        )


@pytest.mark.parametrize("outcome", ["negative", "inconclusive"])
def test_non_completion_records_receive_finite_ttls_and_expire(
    tmp_path: Path,
    outcome: str,
) -> None:
    now = [1_000.0]
    store = BoundedArtifactStore(
        tmp_path / outcome,
        quotas=_quota(),
        clock=lambda: now[0],
    )
    reference = store.store_projection(
        {"summary": outcome, "proof_trace": ["not completion evidence"]},
        projection_kind="proof_receipt",
        outcome=outcome,
        retention_class=RetentionClass.AUTHORITATIVE,
    )

    assert reference.retention_class is RetentionClass.NEGATIVE
    assert reference.outcome is ArtifactOutcome(outcome)
    assert not reference.can_complete
    assert reference.expires_at_ms is not None

    now[0] += 4
    result = store.compact(max_items=16)
    assert result.expired >= 1
    assert reference.artifact_id in result.evicted_artifact_ids
    assert any(
        event["artifact_id"] == reference.artifact_id
        and event["reason"] == "expired"
        for event in store.eviction_events()
    )


def test_incremental_quota_compaction_preserves_pinned_stable_references(
    tmp_path: Path,
) -> None:
    now = [2_000.0]
    store = BoundedArtifactStore(
        tmp_path / "quota",
        quotas=_quota(
            max_bytes=12 * 1024,
            max_blobs=3,
            max_projections=2,
            max_blob_bytes=8 * 1024,
            compaction_batch_size=8,
        ),
        clock=lambda: now[0],
    )
    pinned = store.store_projection(
        {"source_body": "p" * 2_000},
        retention_class=RetentionClass.PINNED,
    )
    stable_blob = pinned.artifact_references[0]
    ephemeral = store.store_projection(
        {"source_body": "e" * 2_000},
        retention_class=RetentionClass.EPHEMERAL,
    )
    replacement = store.store_projection(
        {"source_body": "r" * 2_000},
        retention_class=RetentionClass.ROUTINE,
    )

    manifest = store.manifest()
    assert pinned.artifact_id in manifest["projections"]
    assert replacement.artifact_id in manifest["projections"]
    assert ephemeral.artifact_id not in manifest["projections"]
    assert store.read_blob(stable_blob) == b"p" * 2_000
    assert store.read_projection(pinned)["source_body"]["artifact_ref"] == (
        stable_blob.to_dict()
    )
    assert store.metrics().quota_evictions >= 1
    assert store.compact(max_items=1).scanned <= 1


def test_manifest_restart_recovery_and_blob_integrity_are_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recovery"
    store = BoundedArtifactStore(root, quotas=_quota())
    reference = store.store_projection(
        {"source_body": "authoritative source", "summary": "survives"},
        retention_class=RetentionClass.AUTHORITATIVE,
    )
    blob = reference.artifact_references[0]
    assert store.close(timeout_seconds=0.5)

    (root / "manifest.json").write_text("{torn", encoding="utf-8")
    restarted = BoundedArtifactStore(root, quotas=_quota())
    assert restarted.read_projection(reference)["summary"] == "survives"
    assert restarted.read_blob(blob, decode=True) == "authoritative source"
    assert restarted.metrics().manifest_recoveries == 1

    blob_path = restarted._blob_path(blob)
    blob_path.write_bytes(b"tampered")
    assert not restarted.verify_blob(blob)
    with pytest.raises(ArtifactBlobIntegrityError, match="integrity"):
        restarted.read_blob(blob)
    with pytest.raises(ArtifactBlobIntegrityError, match="corrupt blob"):
        restarted.read_projection(reference)


def test_disk_pressure_rejects_new_work_without_damaging_existing_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "pressure"
    store = BoundedArtifactStore(
        root,
        quotas=_quota(min_free_bytes=1_024),
    )
    existing = store.store_projection({"summary": "already durable"})
    monkeypatch.setattr(
        artifact_store.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=10_000, free=0),
    )

    with pytest.raises(ArtifactQuotaExceeded, match="disk-free reserve"):
        store.put_blob("new bytes")

    assert store.read_projection(existing)["summary"] == "already durable"
    assert store.metrics().disk_pressure_rejections == 1


def test_shutdown_is_bounded_and_closed_store_rejects_writes(
    tmp_path: Path,
) -> None:
    store = BoundedArtifactStore(tmp_path / "shutdown", quotas=_quota())
    store.store_projection({"checkpoint": {"cursor": 1}})
    started = time.monotonic()
    assert store.shutdown(timeout_seconds=0.5)
    assert time.monotonic() - started < 0.5
    assert store.shutdown(timeout_seconds=0.5)
    with pytest.raises(RuntimeError, match="closed"):
        store.put_blob("late write")


def test_event_log_bounds_streaming_rotation_and_recovery_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    for ordinal in range(8):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
    with pytest.raises(EventPayloadTooLarge):
        append_jsonl_event(
            path,
            "validation_receipt",
            {"decoded_model_text": "x" * MAX_RECEIPT_BYTES},
        )

    result = rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=2,
    )
    assert result["rotated"] is True
    assert result["archived_count"] == 5
    assert result["retained_count"] == 3
    assert [event["ordinal"] for event in read_jsonl_event_sources([path])] == (
        list(range(8))
    )

    manifest = event_log_manifest(path)
    assert manifest["generation"] == 1
    assert sum(item["event_count"] for item in manifest["files"]) == 8
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest_path.write_text("{torn", encoding="utf-8")
    recovered = event_log_manifest(path)
    assert recovered["generation"] == 0
    assert sum(item["event_count"] for item in recovered["files"]) == 8
