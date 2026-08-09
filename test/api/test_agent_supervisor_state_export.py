"""Tests for StateExporter@1 (DQP-011).

Evidence subset: byte determinism, pagination, redaction, atomic replacement,
snapshot consistency, lossless round trip, lossy declaration.

Acceptance:

* Re-export of identical snapshot/parameters is byte-identical
* Tampering with or deleting an export cannot affect runtime decisions
* Lossless portable export round-trips
* Human Markdown declares non-authoritative and intentionally omitted fields
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    ControlPlaneAuthorityError,
    StateAuthorityClass,
    StateExportReceipt,
    StateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (
    DEFAULT_PAGE_LIMIT,
    EXPORT_IS_AUTHORITY,
    MARKDOWN_NON_AUTHORITY_BANNER,
    MARKDOWN_OMITTED_FIELDS,
    STATE_EXPORTER_INTERFACE,
    ExportFormat,
    ExportProfile,
    ExportRequest,
    ExportSourceData,
    StateExportDependencyError,
    StateExportParameterError,
    StateExporter,
    closed_export_formats,
    closed_export_profiles,
    load_portable_bundle,
    portable_round_trip_identical,
    pyarrow_available,
    render_markdown,
    runtime_decisions_ignore_exports,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "export_control_plane_state.py"
)

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


def _source(**changes: object) -> ExportSourceData:
    tasks = (
        {
            "task_cid": "task:cid:002",
            "task_alias": "DQP-011",
            "goal_cid": "goal:root",
            "objective_id": "objective:test",
            "status": "todo",
            "priority": "P1",
            "ordinal": 2,
            "revision": 0,
            "created_at": "1970-01-01T00:00:00Z",
            "updated_at": "1970-01-01T00:00:00Z",
            "title": "Render exports",
            "body_json": {"title": "Render exports", "notes": "secret-ish"},
            "password": "should-be-redacted",
            "api_key": "sk-test-should-redact",
        },
        {
            "task_cid": "task:cid:001",
            "task_alias": "DQP-010",
            "goal_cid": "goal:root",
            "objective_id": "objective:test",
            "status": "todo",
            "priority": "P0",
            "ordinal": 1,
            "revision": 1,
            "created_at": "1970-01-01T00:00:00Z",
            "updated_at": "1970-01-01T00:00:00Z",
            "title": "Import legacy state",
            "token": "raw-token-material",
        },
    )
    events = (
        {
            "event_id": "evt:2",
            "stream_id": "tasks",
            "sequence": 2,
            "global_sequence": 2,
            "event_kind": "queued",
            "task_cid": "task:cid:001",
            "created_at": "1970-01-01T00:00:01Z",
            "password": "event-secret",
        },
        {
            "event_id": "evt:1",
            "stream_id": "tasks",
            "sequence": 1,
            "global_sequence": 1,
            "event_kind": "created",
            "task_cid": "task:cid:001",
            "created_at": "1970-01-01T00:00:00Z",
        },
    )
    goals = (
        {
            "goal_cid": "goal:root",
            "goal_alias": "G-ROOT",
            "objective_id": "objective:test",
            "title": "Root goal",
            "status": "open",
            "ordinal": 1,
        },
    )
    objectives = (
        {
            "objective_id": "objective:test",
            "title": "Control plane",
            "status": "active",
            "ordinal": 1,
        },
    )
    leases = (
        {
            "task_cid": "task:cid:001",
            "claimant_did": "did:worker:1",
            "state": "claimed",
            "token": "lease-token-secret",
        },
    )
    values: dict[str, object] = {
        "snapshot": _snapshot(),
        "tasks": tasks,
        "events": events,
        "goals": goals,
        "objectives": objectives,
        "leases": leases,
        "commands": (),
        "metadata": {"fixture": "dqp-011"},
    }
    values.update(changes)
    return ExportSourceData(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cold import / contract surface
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.task_sources.state_export"
    )
    assert module.STATE_EXPORTER_INTERFACE == "StateExporter@1"
    assert module.EXPORT_IS_AUTHORITY is False
    assert module.EXPORT_AUTHORIZES_MUTATION is False
    assert module.StateExporter.INTERFACE == STATE_EXPORTER_INTERFACE
    assert "human-taskboard" in module.closed_export_profiles()
    assert "parquet" in module.closed_export_formats()


def test_closed_profiles_and_formats_are_stable() -> None:
    assert closed_export_profiles() == (
        "human-taskboard",
        "status-json",
        "events-json",
        "audit-jsonl",
        "analysis-csv",
        "analysis-parquet",
        "portable-bundle",
    )
    assert closed_export_formats() == (
        "markdown",
        "json",
        "jsonl",
        "csv",
        "parquet",
        "bundle",
    )


# ---------------------------------------------------------------------------
# Byte determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [
        ExportProfile.HUMAN_TASKBOARD,
        ExportProfile.STATUS_JSON,
        ExportProfile.EVENTS_JSON,
        ExportProfile.AUDIT_JSONL,
        ExportProfile.ANALYSIS_CSV,
        ExportProfile.PORTABLE_BUNDLE,
    ],
)
def test_reexport_identical_snapshot_parameters_is_byte_identical(
    profile: ExportProfile,
) -> None:
    source = _source()
    exporter = StateExporter()
    request = ExportRequest(
        profile=profile,
        destination=f"out/{profile.value}",
        export_id=f"export:{profile.value}:stable",
        cursor=0,
        limit=100,
    )
    first = exporter.render(source, request)
    second = exporter.render(source, request)
    assert first.body == second.body
    assert first.digest == second.digest
    assert dict(first.files) == dict(second.files)

    # Durable write also reproduces bytes.
    # (Performed in a tmp path via export().)
    # Use a fresh request with absolute destinations below in write tests.


def test_analysis_parquet_byte_identical_when_pyarrow_available(
    tmp_path: Path,
) -> None:
    if not pyarrow_available():
        pytest.skip("pyarrow required for parquet export")
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "tasks.parquet"
    request = ExportRequest(
        profile=ExportProfile.ANALYSIS_PARQUET,
        destination=str(dest),
        export_id="export:parquet:1",
    )
    first = exporter.export(source, request, write=True)
    second = exporter.export(source, request, write=True)
    assert first.artifact.body == second.artifact.body
    assert dest.read_bytes() == first.artifact.body


def test_parquet_unavailable_fails_closed() -> None:
    if pyarrow_available():
        pytest.skip("pyarrow is available; dependency gap path not exercised")
    source = _source()
    exporter = StateExporter()
    request = ExportRequest(
        profile=ExportProfile.ANALYSIS_PARQUET,
        destination="tasks.parquet",
    )
    with pytest.raises(StateExportDependencyError, match="pyarrow"):
        exporter.render(source, request)


# ---------------------------------------------------------------------------
# Markdown lossy declaration
# ---------------------------------------------------------------------------


def test_markdown_declares_non_authoritative_and_omitted_fields() -> None:
    source = _source()
    request = ExportRequest(
        profile=ExportProfile.HUMAN_TASKBOARD,
        destination="board.md",
        export_id="export:md:1",
    )
    artifact = render_markdown(source, request)
    text = artifact.body.decode("utf-8")
    assert MARKDOWN_NON_AUTHORITY_BANNER in text
    assert "NON-AUTHORITATIVE" in text
    assert "Intentional loss" in text or "intentional loss" in text.lower()
    assert "Intentionally omitted fields" in text
    for field_name in ("body_json", "password", "api_key", "token"):
        assert f"`{field_name}`" in text
    assert artifact.intentional_loss is True
    assert artifact.omitted_fields == MARKDOWN_OMITTED_FIELDS
    # Secrets must not appear in cleartext.
    assert "should-be-redacted" not in text
    assert "sk-test-should-redact" not in text
    assert "raw-token-material" not in text
    # Task aliases still appear for human readability.
    assert "DQP-010" in text
    assert "DQP-011" in text


def test_export_receipt_never_authoritative(tmp_path: Path) -> None:
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "status.json"
    result = exporter.export(
        source,
        ExportRequest(
            profile=ExportProfile.STATUS_JSON,
            destination=str(dest),
            export_id="export:status:1",
        ),
        write=True,
    )
    assert result.receipt.authority_class is StateAuthorityClass.EXPORT
    assert result.receipt.intentional_loss is True
    assert result.receipt.binds_snapshot(source.snapshot)
    assert EXPORT_IS_AUTHORITY is False
    assert result.to_dict()["is_authority"] is False
    restored = StateExportReceipt.from_dict(result.receipt.to_record())
    assert restored.content_id == result.receipt.content_id
    with pytest.raises(ControlPlaneAuthorityError):
        StateExportReceipt(
            export_id="export:bad",
            snapshot_id=source.snapshot.snapshot_id,
            store_id=source.snapshot.store_id,
            database_uuid=source.snapshot.database_uuid,
            schema_revision=source.snapshot.schema_revision,
            generation=source.snapshot.generation,
            revision=source.snapshot.revision,
            event_watermark=source.snapshot.event_watermark,
            renderer_revision="renderer:json@1",
            query_revision="view:control-plane-export@1",
            artifact_digest=_DIGEST,
            destination=str(dest),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
        )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_exports_redact_secret_bearing_keys() -> None:
    source = _source()
    exporter = StateExporter()
    for profile in (
        ExportProfile.STATUS_JSON,
        ExportProfile.EVENTS_JSON,
        ExportProfile.AUDIT_JSONL,
        ExportProfile.PORTABLE_BUNDLE,
    ):
        artifact = exporter.render(
            source,
            ExportRequest(
                profile=profile,
                destination=f"x/{profile.value}",
                export_id=f"export:{profile.value}:redact",
            ),
        )
        if artifact.files:
            blob = b"".join(artifact.files.values())
        else:
            blob = artifact.body
        assert b"should-be-redacted" not in blob
        assert b"sk-test-should-redact" not in blob
        assert b"raw-token-material" not in blob
        assert b"event-secret" not in blob
        assert b"lease-token-secret" not in blob
        assert REDACTION_MARKER.encode("utf-8") in blob or profile in {
            ExportProfile.STATUS_JSON,  # status projection drops secret keys
        }


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_is_stable_and_bounded() -> None:
    source = _source()
    exporter = StateExporter()
    page0 = exporter.render(
        source,
        ExportRequest(
            profile=ExportProfile.ANALYSIS_CSV,
            destination="page0.csv",
            export_id="export:csv:p0",
            cursor=0,
            limit=1,
            domain="tasks",
        ),
    )
    page1 = exporter.render(
        source,
        ExportRequest(
            profile=ExportProfile.ANALYSIS_CSV,
            destination="page1.csv",
            export_id="export:csv:p1",
            cursor=1,
            limit=1,
            domain="tasks",
        ),
    )
    text0 = page0.body.decode("utf-8")
    text1 = page1.body.decode("utf-8")
    # Sorted by ordinal: task 001 then 002.
    assert "task:cid:001" in text0
    assert "task:cid:002" not in text0
    assert "task:cid:002" in text1
    assert "task:cid:001" not in text1
    # Re-page is byte stable.
    page0_again = exporter.render(
        source,
        ExportRequest(
            profile=ExportProfile.ANALYSIS_CSV,
            destination="page0.csv",
            export_id="export:csv:p0",
            cursor=0,
            limit=1,
            domain="tasks",
        ),
    )
    assert page0_again.body == page0.body


def test_invalid_pagination_parameters_fail_closed() -> None:
    with pytest.raises(StateExportParameterError):
        ExportRequest(
            profile=ExportProfile.ANALYSIS_CSV,
            destination="x.csv",
            limit=0,
        )
    with pytest.raises(StateExportParameterError):
        ExportRequest(
            profile=ExportProfile.ANALYSIS_CSV,
            destination="x.csv",
            cursor=-1,
        )


# ---------------------------------------------------------------------------
# Atomic replacement + snapshot consistency
# ---------------------------------------------------------------------------


def test_atomic_replacement_and_digest_binding(tmp_path: Path) -> None:
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "board.md"
    request = ExportRequest(
        profile=ExportProfile.HUMAN_TASKBOARD,
        destination=str(dest),
        export_id="export:md:atomic",
    )
    first = exporter.export(source, request, write=True)
    assert dest.is_file()
    assert dest.read_bytes() == first.artifact.body
    assert first.receipt.artifact_digest == first.artifact.digest
    assert first.receipt.snapshot_id == source.snapshot.snapshot_id
    assert first.receipt.database_uuid == source.snapshot.database_uuid
    assert first.receipt.generation == source.snapshot.generation
    assert first.receipt.event_watermark == source.snapshot.event_watermark
    assert first.receipt.renderer_revision
    assert first.receipt.query_revision
    # Receipt sidecar exists for operator inspection only.
    assert Path(str(dest) + ".receipt.json").is_file()

    # Second write replaces atomically with identical bytes.
    second = exporter.export(source, request, write=True)
    assert second.artifact.body == first.artifact.body
    assert dest.read_bytes() == first.artifact.body


def test_snapshot_binding_in_receipt_parameters(tmp_path: Path) -> None:
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "events.json"
    result = exporter.export(
        source,
        ExportRequest(
            profile=ExportProfile.EVENTS_JSON,
            destination=str(dest),
            export_id="export:events:bind",
            cursor=0,
            limit=10,
        ),
        write=True,
    )
    params = dict(result.receipt.parameters)
    assert params["profile"] == "events-json"
    assert params["format"] == "json"
    assert params["intentional_loss"] is False
    assert params["cursor"] == 0
    assert params["limit"] == 10
    assert result.receipt.binds_snapshot(source.snapshot)


# ---------------------------------------------------------------------------
# Lossless portable round-trip
# ---------------------------------------------------------------------------


def test_portable_bundle_round_trip_byte_identical(tmp_path: Path) -> None:
    source = _source()
    assert portable_round_trip_identical(source) is True
    dest = tmp_path / "portable-bundle"
    assert portable_round_trip_identical(source, destination=str(dest)) is True
    restored = load_portable_bundle(dest)
    assert restored.snapshot.content_id == source.snapshot.content_id
    # Secret fields remain redacted after round-trip.
    for task in restored.tasks:
        if "password" in task:
            assert task["password"] == REDACTION_MARKER
        if "api_key" in task:
            assert task["api_key"] == REDACTION_MARKER
        if "token" in task:
            assert task["token"] == REDACTION_MARKER

    exporter = StateExporter()
    request = ExportRequest(
        profile=ExportProfile.PORTABLE_BUNDLE,
        destination=str(tmp_path / "again"),
        export_id="export:portable-roundtrip",
    )
    original = exporter.render(source.redacted(), request)
    # Load path already redacted; re-export from restored must match.
    re_rendered = exporter.render(restored, request)
    assert dict(original.files) == dict(re_rendered.files)


def test_portable_bundle_rejects_authority_claim(tmp_path: Path) -> None:
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "bundle"
    exporter.export(
        source,
        ExportRequest(
            profile=ExportProfile.PORTABLE_BUNDLE,
            destination=str(dest),
            export_id="export:portable:auth",
        ),
        write=True,
    )
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["is_authority"] = True
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    # Digest will also fail; either authority or digest error is fail-closed.
    with pytest.raises(Exception):
        load_portable_bundle(dest)


# ---------------------------------------------------------------------------
# Export non-authority for runtime decisions
# ---------------------------------------------------------------------------


def test_tampering_or_deleting_export_cannot_affect_runtime_decisions(
    tmp_path: Path,
) -> None:
    source = _source()
    exporter = StateExporter()
    dest = tmp_path / "status.json"
    exporter.export(
        source,
        ExportRequest(
            profile=ExportProfile.STATUS_JSON,
            destination=str(dest),
            export_id="export:status:runtime",
        ),
        write=True,
    )
    before = runtime_decisions_ignore_exports(source)
    # Tamper
    dest.write_text('{"tampered": true, "status": "completed"}\n', encoding="utf-8")
    after_tamper = runtime_decisions_ignore_exports(source)
    assert after_tamper == before
    # Delete
    dest.unlink()
    receipt = Path(str(dest) + ".receipt.json")
    if receipt.exists():
        receipt.unlink()
    after_delete = runtime_decisions_ignore_exports(source)
    assert after_delete == before
    # Custom decision still ignores filesystem exports.
    decision = runtime_decisions_ignore_exports(
        source,
        decision_fn=lambda data: (
            data.snapshot.generation,
            tuple(
                sorted(
                    str(row.get("task_cid") or "") + ":" + str(row.get("status") or "")
                    for row in data.tasks
                )
            ),
        ),
    )
    assert decision[0] == 1
    assert "task:cid:001:todo" in decision[1]


def test_export_all_profiles_writes_expected_artifacts(tmp_path: Path) -> None:
    source = _source()
    exporter = StateExporter()
    results = exporter.export_all_profiles(
        source,
        tmp_path / "exports",
        write=True,
        include_parquet=pyarrow_available(),
    )
    assert results
    names = {Path(item.destination).name for item in results}
    assert "human-taskboard.md" in names
    assert "status-json.json" in names
    assert "events-json.json" in names
    assert "audit-jsonl.jsonl" in names
    assert "analysis-csv.csv" in names
    assert "portable-bundle" in names
    for item in results:
        assert item.receipt.authority_class is StateAuthorityClass.EXPORT
        assert item.written is True


# ---------------------------------------------------------------------------
# CLI facade
# ---------------------------------------------------------------------------


def test_cli_profiles_and_export(tmp_path: Path) -> None:
    source = _source()
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(source.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    profiles = subprocess.run(
        [sys.executable, str(CLI_PATH), "profiles", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert profiles.returncode == 0, profiles.stderr
    payload = json.loads(profiles.stdout)
    assert payload["interface"] == "StateExporter@1"
    assert "human-taskboard" in payload["profiles"]
    assert payload["is_authority"] is False

    dest = tmp_path / "board.md"
    export = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "export",
            "--source-json",
            str(source_path),
            "--profile",
            "human-taskboard",
            "--destination",
            str(dest),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert export.returncode == 0, export.stderr
    result = json.loads(export.stdout)
    assert result["is_authority"] is False
    assert result["receipt"]["authority_class"] == "export"
    assert dest.is_file()
    text = dest.read_text(encoding="utf-8")
    assert "NON-AUTHORITATIVE" in text

    # Credential argv is rejected.
    denied = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "export",
            "--token",
            "secret",
            "--source-json",
            str(source_path),
            "--profile",
            "status-json",
            "--destination",
            str(tmp_path / "x.json"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode != 0
    assert "credential" in (denied.stderr + denied.stdout).lower() or denied.returncode == 2


def test_cli_round_trip(tmp_path: Path) -> None:
    source = _source()
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(source.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "bundle"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "round-trip",
            "--source-json",
            str(source_path),
            "--destination",
            str(dest),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["round_trip_identical"] is True


def test_cli_help_is_cold() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "export" in completed.stdout


def test_default_page_limit_constant() -> None:
    assert DEFAULT_PAGE_LIMIT == 1000
    assert ExportFormat.MARKDOWN.value == "markdown"
    assert ExportProfile.PORTABLE_BUNDLE.value == "portable-bundle"
