"""Tests for StateExporter@1 (DQP-011).

Evidence subset: byte determinism, pagination, redaction, atomic replacement,
snapshot consistency, lossless round trip, lossy declaration.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    StateAuthorityClass,
    StateExportReceipt,
    StateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (
    EXPORTER_VERSION,
    MARKDOWN_OMITTED_FIELDS,
    NON_AUTHORITY_BANNER,
    PORTABLE_EXPORT_SCHEMA,
    STATE_EXPORTER_INTERFACE,
    ExportMediaType,
    ExportView,
    StateExportFormatError,
    StateExportPayload,
    StateExportRequest,
    StateExportRequestError,
    StateExporter,
    duckdb_available,
    export_state,
    intentional_loss_for,
    media_type_from_path,
    query_revision_for,
    render_state,
    renderer_revision_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts/ops/agent_supervisor/export_control_plane_state.py"

_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _snapshot(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "snapshot:1:7:42",
        "store_id": "control.duckdb",
        "database_uuid": _UUID,
        "generation": 1,
        "schema_revision": 1,
        "revision": 7,
        "fence_epoch": 3,
        "event_watermark": 42,
        "snapshot_digest": _DIGEST,
        "authority_class": StateAuthorityClass.AUTHORITATIVE,
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def _payload(**changes: object) -> StateExportPayload:
    tasks = (
        {
            "task_cid": "task:cid:002",
            "task_alias": "DQP-011",
            "title": "Render exports",
            "status": "todo",
            "priority": "P1",
            "goal_cid": "goal:root",
            "api_key": "should-be-redacted",
        },
        {
            "task_cid": "task:cid:001",
            "task_alias": "DQP-010",
            "title": "Import legacy state",
            "status": "completed",
            "priority": "P0",
            "goal_cid": "goal:root",
        },
    )
    events = (
        {
            "event_id": "evt-2",
            "global_sequence": 2,
            "event_type": "task.updated",
            "task_cid": "task:cid:001",
        },
        {
            "event_id": "evt-1",
            "global_sequence": 1,
            "event_type": "task.created",
            "task_cid": "task:cid:001",
        },
    )
    leases = (
        {
            "task_cid": "task:cid:001",
            "claimant_did": "did:worker:a",
            "state": "active",
        },
    )
    commands = (
        {
            "idempotency_key": "cmd-1",
            "command_kind": "claim",
            "command_id": "command:1",
        },
    )
    values: dict[str, object] = {
        "snapshot": _snapshot(),
        "store_identity": {
            "repository_id": "repository:test",
            "database_uuid": _UUID,
            "store_id": "control.duckdb",
            "schema_fingerprint": _DIGEST,
        },
        "generation": {
            "store_id": "control.duckdb",
            "generation": 1,
            "schema_revision": 1,
            "revision": 7,
            "database_uuid": _UUID,
        },
        "tasks": tasks,
        "leases": leases,
        "events": events,
        "commands": commands,
        "schema_fingerprint": _DIGEST,
    }
    values.update(changes)
    return StateExportPayload(**values)  # type: ignore[arg-type]


def _request(destination: Path | str, **changes: object) -> StateExportRequest:
    values: dict[str, object] = {
        "destination": str(destination),
        "media_type": ExportMediaType.JSON,
        "view": ExportView.PORTABLE,
        "offset": 0,
        "limit": 1000,
        "domains": ("tasks", "leases", "events", "commands"),
        "parameters": {"profile": "test"},
    }
    values.update(changes)
    return StateExportRequest(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cold import / contract surface
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.task_sources.state_export"
    )
    assert module.STATE_EXPORTER_INTERFACE == "StateExporter@1"
    assert module.STATE_EXPORT_RECEIPT_INTERFACE == "StateExportReceipt@1"
    assert module.EXPORTER_VERSION == "state-export/1"
    assert CLI.is_file()


def test_cli_help_exits_zero_without_io() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        },
    )
    assert result.returncode == 0
    assert "StateExporter@1" in result.stdout or "snapshot-bound" in result.stdout


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_request_rejects_unsupported_view_media_pairs() -> None:
    with pytest.raises(StateExportFormatError):
        StateExportRequest(
            destination="out.md",
            media_type=ExportMediaType.MARKDOWN,
            view=ExportView.PORTABLE,
        )
    with pytest.raises(StateExportFormatError):
        StateExportRequest(
            destination="out.json",
            media_type=ExportMediaType.JSON,
            view=ExportView.TASKBOARD,
        )


def test_media_type_from_path_and_revisions() -> None:
    assert media_type_from_path("x.md") is ExportMediaType.MARKDOWN
    assert media_type_from_path("x.jsonl") is ExportMediaType.JSONL
    assert media_type_from_path("x.parquet") is ExportMediaType.PARQUET
    assert renderer_revision_for(ExportMediaType.JSON) == "renderer:json@1"
    assert query_revision_for(ExportView.PORTABLE) == "view:portable@1"
    assert intentional_loss_for(ExportView.TASKBOARD, ExportMediaType.MARKDOWN) is True
    assert intentional_loss_for(ExportView.PORTABLE, ExportMediaType.JSON) is False


def test_request_redacts_secret_parameters() -> None:
    request = StateExportRequest(
        destination="out.json",
        media_type=ExportMediaType.JSON,
        view=ExportView.PORTABLE,
        parameters={"profile": "ok", "password": "x"},
    )
    assert request.parameters["password"] == REDACTION_MARKER
    assert request.parameters["profile"] == "ok"


# ---------------------------------------------------------------------------
# Payload redaction + snapshot binding
# ---------------------------------------------------------------------------


def test_payload_redacts_secret_keys_in_rows() -> None:
    payload = _payload()
    # Secret-bearing task field is redacted at construction.
    redacted_tasks = [task for task in payload.tasks if "api_key" in task]
    assert redacted_tasks
    assert redacted_tasks[0]["api_key"] == REDACTION_MARKER


def test_tasks_sorted_stably_by_task_cid() -> None:
    payload = _payload()
    rows = payload.domain_rows("tasks")
    assert [row["task_cid"] for row in rows] == [
        "task:cid:001",
        "task:cid:002",
    ]
    events = payload.domain_rows("events")
    assert [row["event_id"] for row in events] == ["evt-1", "evt-2"]


# ---------------------------------------------------------------------------
# Byte determinism
# ---------------------------------------------------------------------------


def test_reexport_identical_snapshot_is_byte_identical(tmp_path: Path) -> None:
    payload = _payload()
    dest_a = tmp_path / "a" / "export.json"
    dest_b = tmp_path / "b" / "export.json"
    # Destinations differ but we compare pure render bytes and digests of content.
    request_a = _request(dest_a)
    request_b = _request(dest_b)
    bytes_a = render_state(payload, request_a)
    bytes_b = render_state(payload, request_b)
    assert bytes_a == bytes_b

    receipt_a = export_state(payload, request_a)
    receipt_b = export_state(payload, request_b)
    assert dest_a.read_bytes() == dest_b.read_bytes()
    assert receipt_a.artifact_digest == receipt_b.artifact_digest
    assert receipt_a.binds_snapshot(payload.snapshot)
    assert receipt_b.binds_snapshot(payload.snapshot)
    assert receipt_a.authority_class is StateAuthorityClass.EXPORT
    assert receipt_a.intentional_loss is False
    assert receipt_a.renderer_revision == "renderer:json@1"
    assert receipt_a.query_revision == "view:portable@1"
    assert receipt_a.parameters["exporter_version"] == EXPORTER_VERSION
    # Receipt parameters must stay JSON-canonical after freeze (no tuples).
    record = receipt_a.to_record()
    assert record["content_id"].startswith("b")
    assert isinstance(record["parameters"]["domains"], str)
    assert "tasks" in record["parameters"]["domains"]
    # Round-trip through json.dumps must succeed for CLI receipt printing.
    assert json.loads(json.dumps(record, sort_keys=True))["export_id"] == (
        receipt_a.export_id
    )


def test_second_export_overwrites_atomically_and_stably(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "export.json"
    request = _request(destination)
    first = export_state(payload, request)
    # Corrupt destination then re-export; atomic replace restores exact bytes.
    destination.write_bytes(b"tampered-not-export")
    second = export_state(payload, request)
    assert destination.read_bytes() != b"tampered-not-export"
    assert second.artifact_digest == first.artifact_digest
    assert destination.read_bytes() == render_state(payload, request)


# ---------------------------------------------------------------------------
# Lossless portable round trip
# ---------------------------------------------------------------------------


def test_portable_json_round_trip(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "portable.json"
    request = _request(destination, view=ExportView.PORTABLE, media_type=ExportMediaType.JSON)
    receipt = export_state(payload, request)
    assert receipt.intentional_loss is False

    exporter = StateExporter()
    restored = exporter.load_portable(destination)
    assert restored.snapshot.to_dict() == payload.snapshot.to_dict()
    assert restored.domain_rows("tasks") == payload.domain_rows("tasks")
    assert restored.domain_rows("events") == payload.domain_rows("events")
    assert restored.content_id == payload.content_id

    # Re-render from restored payload is byte-identical.
    again = tmp_path / "portable-again.json"
    again_bytes = render_state(restored, _request(again))
    assert again_bytes == destination.read_bytes()


def test_portable_export_schema_and_non_authority(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "portable.json"
    export_state(payload, _request(destination))
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema"] == PORTABLE_EXPORT_SCHEMA
    assert document["authority_class"] == "export"
    assert document["intentional_loss"] is False
    assert "password" not in json.dumps(document)
    # Redacted marker may appear for secret keys; no live secret value.
    assert "should-be-redacted" not in destination.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lossy markdown declaration
# ---------------------------------------------------------------------------


def test_markdown_declares_non_authority_and_intentional_loss(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "taskboard.md"
    request = _request(
        destination,
        media_type=ExportMediaType.MARKDOWN,
        view=ExportView.TASKBOARD,
    )
    receipt = export_state(payload, request)
    text = destination.read_text(encoding="utf-8")
    assert receipt.intentional_loss is True
    assert receipt.authority_class is StateAuthorityClass.EXPORT
    assert NON_AUTHORITY_BANNER in text
    assert "Intentional loss: true" in text
    assert "Authority class: export" in text
    for omitted in MARKDOWN_OMITTED_FIELDS:
        assert omitted in text
    assert "DQP-010" in text
    assert "DQP-011" in text
    # Markdown must not embed lease/command authority tables.
    assert "did:worker:a" not in text
    assert "cmd-1" not in text
    # Re-export byte-identical.
    assert render_state(payload, request) == destination.read_bytes()


# ---------------------------------------------------------------------------
# Pagination, JSONL, CSV
# ---------------------------------------------------------------------------


def test_pagination_limits_task_page(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "page.md"
    request = _request(
        destination,
        media_type=ExportMediaType.MARKDOWN,
        view=ExportView.TASKBOARD,
        offset=0,
        limit=1,
    )
    text = render_state(payload, request).decode("utf-8")
    # Stable sort puts DQP-010 (task:cid:001) first.
    assert "DQP-010" in text
    assert "DQP-011" not in text
    assert "Task count (page): 1" in text


def test_jsonl_events_export(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "events.jsonl"
    request = _request(
        destination,
        media_type=ExportMediaType.JSONL,
        view=ExportView.EVENTS,
    )
    receipt = export_state(payload, request)
    lines = [
        line
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["event_id"] == "evt-1"
    assert parsed[1]["event_id"] == "evt-2"
    assert receipt.intentional_loss is False
    assert render_state(payload, request) == destination.read_bytes()


def test_csv_analysis_export_is_deterministic(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "analysis.csv"
    request = _request(
        destination,
        media_type=ExportMediaType.CSV,
        view=ExportView.ANALYSIS,
        domains=("tasks", "events"),
    )
    first = render_state(payload, request)
    second = render_state(payload, request)
    assert first == second
    assert first.startswith(b"domain,")
    text = first.decode("utf-8")
    assert "tasks" in text
    assert "events" in text
    receipt = export_state(payload, request)
    assert receipt.intentional_loss is True


def test_status_json_projection_is_lossy(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "status.json"
    request = _request(
        destination,
        media_type=ExportMediaType.JSON,
        view=ExportView.STATUS,
    )
    receipt = export_state(payload, request)
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["intentional_loss"] is True
    assert document["task_count"] == 2
    assert document["status_counts"]["todo"] == 1
    assert document["status_counts"]["completed"] == 1
    assert receipt.intentional_loss is True
    # Status view omits full task bodies.
    assert "tasks" not in document


# ---------------------------------------------------------------------------
# Parquet (optional DuckDB)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for Parquet export")
def test_parquet_export_is_byte_identical(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "analysis.parquet"
    request = _request(
        destination,
        media_type=ExportMediaType.PARQUET,
        view=ExportView.ANALYSIS,
        domains=("tasks",),
    )
    first = render_state(payload, request)
    second = render_state(payload, request)
    # Same snapshot + parameters must be byte-stable across renders.
    assert first == second
    assert first[:4] == b"PAR1"
    assert first.endswith(b"PAR1")
    receipt = export_state(payload, request)
    assert receipt.artifact_digest.startswith("sha256:")
    assert destination.read_bytes() == first
    # Re-export replaces atomically with the same digest.
    again = export_state(payload, request)
    assert again.artifact_digest == receipt.artifact_digest


# ---------------------------------------------------------------------------
# Export does not grant authority / snapshot consistency
# ---------------------------------------------------------------------------


def test_export_receipt_cannot_be_authoritative(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "portable.json"
    receipt = export_state(payload, _request(destination))
    assert receipt.authority_class is not StateAuthorityClass.AUTHORITATIVE
    with pytest.raises(Exception):
        StateExportReceipt(
            export_id=receipt.export_id,
            snapshot_id=receipt.snapshot_id,
            store_id=receipt.store_id,
            database_uuid=receipt.database_uuid,
            schema_revision=receipt.schema_revision,
            generation=receipt.generation,
            revision=receipt.revision,
            event_watermark=receipt.event_watermark,
            renderer_revision=receipt.renderer_revision,
            query_revision=receipt.query_revision,
            artifact_digest=receipt.artifact_digest,
            destination=receipt.destination,
            parameters=dict(receipt.parameters),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            intentional_loss=False,
        )


def test_deleting_export_does_not_affect_payload(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "portable.json"
    export_state(payload, _request(destination))
    destination.unlink()
    # Payload remains fully usable for another export.
    again = tmp_path / "again.json"
    receipt = export_state(payload, _request(again))
    assert again.is_file()
    assert receipt.binds_snapshot(payload.snapshot)


def test_snapshot_mismatch_breaks_bind(tmp_path: Path) -> None:
    payload = _payload()
    destination = tmp_path / "portable.json"
    receipt = export_state(payload, _request(destination))
    other = _snapshot(revision=8, snapshot_id="snapshot:1:8:42")
    assert receipt.binds_snapshot(payload.snapshot) is True
    assert receipt.binds_snapshot(other) is False


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_exports_from_payload(tmp_path: Path) -> None:
    payload = _payload()
    payload_path = tmp_path / "input.json"
    payload_path.write_text(
        json.dumps(payload.to_portable_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "out" / "taskboard.md"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--payload",
            str(payload_path),
            "--destination",
            str(destination),
            "--view",
            "taskboard",
            "--media-type",
            "markdown",
            "--print-receipt",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        },
    )
    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    receipt = json.loads(result.stdout)
    assert receipt["authority_class"] == "export"
    assert receipt["intentional_loss"] is True
    assert NON_AUTHORITY_BANNER in destination.read_text(encoding="utf-8")


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    payload = _payload()
    payload_path = tmp_path / "input.json"
    payload_path.write_text(
        json.dumps(payload.to_portable_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "should-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--payload",
            str(payload_path),
            "--destination",
            str(destination),
            "--view",
            "portable",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        },
    )
    assert result.returncode == 0, result.stderr
    assert not destination.exists()
    receipt = json.loads(result.stdout)
    assert receipt["artifact_digest"].startswith("sha256:")


def test_interface_constant_matches_task_contract() -> None:
    assert STATE_EXPORTER_INTERFACE == "StateExporter@1"
    assert StateExportReceipt.INTERFACE == "StateExportReceipt@1"


def test_request_rejects_empty_destination() -> None:
    with pytest.raises(StateExportRequestError):
        StateExportRequest(
            destination="",
            media_type=ExportMediaType.JSON,
            view=ExportView.PORTABLE,
        )
