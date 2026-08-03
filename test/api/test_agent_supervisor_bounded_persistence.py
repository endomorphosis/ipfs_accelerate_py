from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor import artifact_store
from ipfs_accelerate_py.agent_supervisor.runtime import event_log as event_log_runtime
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
    EventLogIntegrityFailure,
    EventLogTailRecoveryRequired,
    EventPayloadTooLarge,
    append_jsonl_event,
    event_log_integrity_failure,
    event_log_manifest,
    initial_event_cursor,
    recover_jsonl_event_log_tail,
    read_jsonl_event_page,
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


def test_reconciled_event_log_tail_manifest_is_durable_for_strict_restart(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_review import (
        _strict_event_ledger,
    )

    path = tmp_path / "events.jsonl"
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    stale_manifest = manifest_path.read_bytes()
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    manifest_path.write_bytes(stale_manifest)

    reconciled = event_log_manifest(path)
    persisted = event_log_runtime._load_event_manifest(path)

    assert reconciled["latest_sequence"] == 2
    assert persisted == reconciled
    assert [event["sequence"] for event in _strict_event_ledger(path)] == [
        1,
        2,
    ]


def test_reconciled_manifest_write_failure_does_not_fail_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    stale_manifest = manifest_path.read_bytes()
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    manifest_path.write_bytes(stale_manifest)
    original_write = event_log_runtime._write_manifest_value
    calls = 0

    def fail_reconciliation_write_once(
        event_path: Path,
        value: dict[str, object],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected reconciled-manifest write failure")
        original_write(event_path, value)

    monkeypatch.setattr(
        event_log_runtime,
        "_write_manifest_value",
        fail_reconciliation_write_once,
    )

    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 3},
    )

    assert appended["sequence"] == 3
    assert calls == 2
    assert event_log_runtime._load_event_manifest(path)["latest_sequence"] == 3


def test_first_rotation_manifest_write_failure_uses_fresh_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_review import (
        _strict_event_ledger,
    )

    path = tmp_path / "events.jsonl"
    for ordinal in range(8):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
    original_write = event_log_runtime._write_manifest_value
    calls = 0

    def fail_first_rotation_manifest_write(
        event_path: Path,
        value: dict[str, object],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected first rotation manifest write failure")
        original_write(event_path, value)

    monkeypatch.setattr(
        event_log_runtime,
        "_write_manifest_value",
        fail_first_rotation_manifest_write,
    )
    failed = rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=8,
    )
    monkeypatch.setattr(
        event_log_runtime,
        "_write_manifest_value",
        original_write,
    )

    assert failed["rotated"] is False
    assert failed["reason"] == "write_failed"
    first_archive = next(
        path.parent.glob(f"{path.name}.rotated-g*-*")
    )
    assert event_log_runtime._installed_rotation_generations(path) == (1,)

    recovered = event_log_manifest(path)
    assert recovered["latest_sequence"] == 8
    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 8},
    )
    assert appended["sequence"] == 9

    retried = rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=8,
    )

    assert retried["rotated"] is True
    assert Path(retried["archive_path"]) != first_archive
    assert retried["manifest_generation"] == 2
    assert event_log_runtime._installed_rotation_generations(path) == (1, 2)
    assert [event["sequence"] for event in _strict_event_ledger(path)] == list(
        range(1, 10)
    )


@pytest.mark.parametrize("reader", ("manifest", "page"))
def test_event_log_read_cannot_launder_active_history_regression(
    tmp_path: Path,
    reader: str,
) -> None:
    path = tmp_path / "events.jsonl"
    initial = initial_event_cursor(path)
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest_before = manifest_path.read_bytes()
    first_line = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first_line)

    with pytest.raises(EventLogIntegrityFailure):
        if reader == "manifest":
            event_log_manifest(path)
        else:
            read_jsonl_event_page(path, initial, limit=10)

    latch = event_log_integrity_failure(path)
    assert latch is not None
    assert latch["affected_path"] == path.name
    assert latch["actual_size_bytes"] < latch["expected_size_bytes"]
    assert manifest_path.read_bytes() == manifest_before
    with pytest.raises(EventLogIntegrityFailure):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": 3})


def test_unexpected_archive_cannot_mask_active_history_regression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    for ordinal in range(3):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
    manifest_before = event_log_manifest(path)
    first_line = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first_line)
    path.with_name(f"{path.name}.rotated-mask").write_bytes(b"")

    with pytest.raises(EventLogIntegrityFailure):
        event_log_manifest(path)

    latch = event_log_integrity_failure(path)
    assert latch is not None
    assert latch["affected_path"] == path.name
    assert latch["expected_latest_sequence"] == (
        manifest_before["latest_sequence"]
    )


def test_tail_recovery_cannot_launder_clean_history_regression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    first_line = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first_line)

    result = recover_jsonl_event_log_tail(path)

    assert result["failed_closed"] is True
    assert result["reason"] == "event_log_history_regressed"
    assert event_log_integrity_failure(path) is not None
    with pytest.raises(EventLogIntegrityFailure):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": 3})


def test_tail_recovery_quarantines_only_unindexed_partial_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    with path.open("ab") as stream:
        stream.write(b'{"partial":')

    result = recover_jsonl_event_log_tail(path)

    assert result["repaired"] is True
    assert result["failed_closed"] is False
    assert result["reason"] == "partial_tail_quarantined"
    assert Path(result["quarantine_path"]).read_bytes() == b'{"partial":'
    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 3},
    )
    assert appended["sequence"] == 3
    assert event_log_integrity_failure(path) is None


def test_append_requires_recovery_before_unindexed_partial_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    initial = initial_event_cursor(path)
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    damaged = path.read_bytes()

    with pytest.raises(EventLogTailRecoveryRequired):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": 3})

    assert path.read_bytes() == damaged
    assert event_log_runtime._load_event_manifest(path)["latest_sequence"] == 2
    assert event_log_integrity_failure(path) is None

    recovery = recover_jsonl_event_log_tail(path)
    assert recovery["repaired"] is True
    assert Path(recovery["quarantine_path"]).read_bytes() == b'{"partial":'
    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 3},
    )
    page = read_jsonl_event_page(path, initial, limit=10)
    assert appended["sequence"] == 3
    assert [event["sequence"] for event in page.events] == [1, 2, 3]


def test_tail_recovery_preserves_complete_unindexed_event_before_partial_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest_before_third = manifest_path.read_bytes()
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 3})
    manifest_path.write_bytes(manifest_before_third)
    with path.open("ab") as stream:
        stream.write(b'{"partial":')

    result = recover_jsonl_event_log_tail(path)

    assert result["repaired"] is True
    assert result["failed_closed"] is False
    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 4},
    )
    assert appended["sequence"] == 4
    assert event_log_integrity_failure(path) is None


@pytest.mark.parametrize(
    "damage_mode",
    ("delete", "replace", "shrink"),
)
def test_event_log_append_latches_any_sealed_archive_regression(
    tmp_path: Path,
    damage_mode: str,
) -> None:
    path = tmp_path / "events.jsonl"
    for ordinal in range(8):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
    rotation = rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=2,
    )
    archive_path = Path(rotation["archive_path"])
    archive_payload = archive_path.read_bytes()
    manifest_before = event_log_manifest(path)

    if damage_mode == "delete":
        archive_path.unlink()
    elif damage_mode == "replace":
        replacement = archive_path.with_name(
            f".{archive_path.name}.replacement"
        )
        replacement.write_bytes(archive_payload)
        replacement.replace(archive_path)
    else:
        archive_path.write_bytes(archive_payload[:-1])

    with pytest.raises(EventLogIntegrityFailure):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": 9})

    latch = event_log_integrity_failure(path)
    assert latch is not None
    assert latch["affected_path"] == archive_path.name
    assert latch["expected_latest_sequence"] == (
        manifest_before["latest_sequence"]
    )


@pytest.mark.parametrize("failed_manifest_write", (2, 3))
def test_archive_retirement_manifest_failure_never_latches_healthy_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_manifest_write: int,
) -> None:
    path = tmp_path / "events.jsonl"
    for ordinal in range(8):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
    assert rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=1,
    )["rotated"]
    for ordinal in range(8, 16):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})

    original_write = event_log_runtime._write_manifest_value
    calls = 0

    def fail_selected_write(
        event_path: Path,
        value: dict[str, object],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_manifest_write:
            raise OSError("injected manifest durability failure")
        original_write(event_path, value)

    monkeypatch.setattr(
        event_log_runtime,
        "_write_manifest_value",
        fail_selected_write,
    )
    rotation = rotate_event_log_if_needed(
        path,
        max_bytes=1,
        retain_recent=3,
        max_archives=1,
    )
    monkeypatch.setattr(
        event_log_runtime,
        "_write_manifest_value",
        original_write,
    )

    if failed_manifest_write == 2:
        assert rotation["reason"] == "write_failed"
    else:
        assert rotation["rotated"] is True
    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 16},
    )
    assert appended["sequence"] == 17
    assert event_log_integrity_failure(path) is None


def test_same_size_active_rewrite_latches_canonical_head_regression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 1})
    append_jsonl_event(path, "scheduler_tick", {"ordinal": 2})
    lines = path.read_bytes().splitlines(keepends=True)
    rewritten = json.loads(lines[1])
    rewritten["ordinal"] = 9
    rewritten["event_id"] = event_log_runtime._event_identity(rewritten)
    replacement = (
        json.dumps(
            rewritten,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert len(replacement) == len(lines[1])
    path.write_bytes(lines[0] + replacement)

    with pytest.raises(EventLogIntegrityFailure):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": 3})

    latch = event_log_integrity_failure(path)
    assert latch is not None
    assert latch["affected_path"] == path.name


def test_repeated_archive_retention_keeps_a_strict_verifiable_chain(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_review import (
        _strict_event_ledger,
    )

    path = tmp_path / "events.jsonl"
    archive_names: list[str] = []
    for ordinal in range(20):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})
        if ordinal in {3, 7, 11, 15, 19}:
            rotation = rotate_event_log_if_needed(
                path,
                max_bytes=1,
                retain_recent=2,
                max_archives=1,
            )
            assert rotation["rotated"] is True
            archive_names.append(Path(rotation["archive_path"]).name)

    manifest = event_log_manifest(path)
    retained = _strict_event_ledger(path)

    assert len(set(archive_names)) == len(archive_names)
    assert manifest["earliest_sequence"] > 1
    assert manifest["latest_sequence"] == 20
    assert retained[-1]["ordinal"] == 19
    assert retained[-1]["sequence"] == 20
    assert event_log_integrity_failure(path) is None


def test_strict_ledger_coalesces_exact_rotation_crash_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_review import (
        _strict_event_ledger,
    )

    path = tmp_path / "events.jsonl"
    for ordinal in range(1, 7):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})

    original_atomic_write = event_log_runtime._atomic_write_bytes

    def crash_before_active_replacement(
        target: Path,
        payload: bytes,
    ) -> None:
        if target == path:
            raise RuntimeError("injected crash before active-tail replacement")
        original_atomic_write(target, payload)

    monkeypatch.setattr(
        event_log_runtime,
        "_atomic_write_bytes",
        crash_before_active_replacement,
    )
    with pytest.raises(
        RuntimeError,
        match="injected crash before active-tail replacement",
    ):
        rotate_event_log_if_needed(
            path,
            max_bytes=1,
            retain_recent=2,
            max_archives=8,
        )
    monkeypatch.setattr(
        event_log_runtime,
        "_atomic_write_bytes",
        original_atomic_write,
    )

    manifest = event_log_manifest(path)
    ranges = [
        (
            int(record["first_sequence"]),
            int(record["last_sequence"]),
        )
        for record in manifest["files"]
    ]
    assert ranges == [(1, 4), (1, 6)]
    assert [
        event["sequence"] for event in _strict_event_ledger(path)
    ] == [1, 2, 3, 4, 5, 6]

    appended = append_jsonl_event(
        path,
        "scheduler_tick",
        {"ordinal": 7},
    )
    assert appended["sequence"] == 7
    assert [
        event["sequence"] for event in _strict_event_ledger(path)
    ] == [1, 2, 3, 4, 5, 6, 7]


def test_strict_ledger_rejects_conflicting_rotation_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_review import (
        PostMergeReviewError,
        _strict_event_ledger,
    )

    path = tmp_path / "events.jsonl"
    for ordinal in range(1, 7):
        append_jsonl_event(path, "scheduler_tick", {"ordinal": ordinal})

    original_atomic_write = event_log_runtime._atomic_write_bytes

    def crash_before_active_replacement(
        target: Path,
        payload: bytes,
    ) -> None:
        if target == path:
            raise RuntimeError("injected crash before active-tail replacement")
        original_atomic_write(target, payload)

    monkeypatch.setattr(
        event_log_runtime,
        "_atomic_write_bytes",
        crash_before_active_replacement,
    )
    with pytest.raises(RuntimeError):
        rotate_event_log_if_needed(
            path,
            max_bytes=1,
            retain_recent=2,
            max_archives=8,
        )
    monkeypatch.setattr(
        event_log_runtime,
        "_atomic_write_bytes",
        original_atomic_write,
    )
    manifest = event_log_manifest(path)

    rewritten_events: list[dict[str, object]] = []
    previous_event_id = ""
    for index, raw_line in enumerate(path.read_bytes().splitlines()):
        event = json.loads(raw_line)
        if index == 0:
            event["ordinal"] = 9
        event["previous_event_id"] = previous_event_id
        event["event_id"] = event_log_runtime._event_identity(event)
        previous_event_id = str(event["event_id"])
        rewritten_events.append(event)
    rewritten_payload = b"".join(
        json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for event in rewritten_events
    )
    path.write_bytes(rewritten_payload)

    conflicting_manifest = dict(manifest)
    records = [
        dict(record)
        for record in manifest["files"]
    ]
    active_record = next(
        record for record in records if record["path"] == path.name
    )
    active_record.update(
        {
            "size_bytes": len(rewritten_payload),
            "sha256": hashlib.sha256(rewritten_payload).hexdigest(),
            **event_log_runtime._stat_fields(path),
        }
    )
    conflicting_manifest.update(
        {
            "active_indexed_bytes": len(rewritten_payload),
            "last_event_id": previous_event_id,
            "files": records,
        }
    )
    conflicting_manifest["manifest_digest"] = (
        event_log_runtime._event_manifest_digest(conflicting_manifest)
    )
    event_log_runtime._write_manifest_value(path, conflicting_manifest)

    with pytest.raises(
        PostMergeReviewError,
        match="duplicate sequence has conflicting identity",
    ):
        _strict_event_ledger(path)
