"""Tests for legacy Markdown/JSON/JSONL/SQLite/DuckDB state import (DQP-010)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
    ConflictPolicy,
    DEFAULT_PARSER_VERSION,
    ImportAtomicityError,
    ImportDomain,
    ImportManifest,
    ImportManifestError,
    ImportMode,
    ImportParseError,
    ImportSchemaError,
    ImportSourceDescriptor,
    ImportSourceKind,
    ImportSourceMutationError,
    LegacyStateImport,
    OUTCOME_APPLIED,
    OUTCOME_PREVIEW,
    OUTCOME_REPLAYED,
    ParsedRecord,
    build_manifest,
    duckdb_available,
    reconcile_records,
    sha256_digest,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_json_bytes,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for legacy state import hermetic tests",
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _source(
    source_id: str,
    path: str,
    *,
    kind: str,
    domain: str = "taskboards",
    parser_version: str = DEFAULT_PARSER_VERSION,
    schema_identity: str = "",
    select_priority: int = 0,
) -> ImportSourceDescriptor:
    return ImportSourceDescriptor(
        source_id=source_id,
        path=path,
        kind=kind,
        domain=domain,
        parser_version=parser_version,
        schema_identity=schema_identity,
        select_priority=select_priority,
    )


def _importer(tmp_path: Path) -> LegacyStateImport:
    return LegacyStateImport(tmp_path / "import_target.duckdb", root=tmp_path)


def test_manifest_refuses_duplicate_source_ids() -> None:
    with pytest.raises(ImportManifestError, match="duplicate source_id"):
        ImportManifest(
            manifest_id="m1",
            sources=(
                _source("a", "a.json", kind="json"),
                _source("a", "b.json", kind="json"),
            ),
        )


def test_manifest_refuses_last_write_wins_policy() -> None:
    with pytest.raises(ImportManifestError, match="conflict_policy"):
        ImportManifest(
            manifest_id="m1",
            sources=(_source("a", "a.json", kind="json"),),
            conflict_policy="last-write-wins",  # type: ignore[arg-type]
        )


def test_preview_markdown_json_jsonl_sqlite_and_duckdb(tmp_path: Path) -> None:
    _write(
        tmp_path / "board.md",
        "## TASK-1 First task\n- Status: todo\n- Priority: P0\n\n"
        "## TASK-2 Second task\n- Status: completed\n",
    )
    _write(
        tmp_path / "goals.json",
        json.dumps(
            {
                "schema": "test/goals@1",
                "records": [
                    {"goal_id": "G1", "title": "Import state", "status": "active"},
                    {"goal_id": "G2", "title": "Export state", "status": "blocked"},
                ],
            }
        ),
    )
    _write(
        tmp_path / "events.jsonl",
        "\n".join(
            [
                json.dumps({"event_id": "E1", "type": "claimed", "task_id": "TASK-1"}),
                json.dumps({"event_id": "E2", "type": "finished", "task_id": "TASK-1"}),
            ]
        )
        + "\n",
    )

    sqlite_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        "CREATE TABLE leases (lease_id TEXT PRIMARY KEY, owner TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO leases VALUES ('L1', 'daemon-a')")
    conn.commit()
    conn.close()

    import duckdb

    duckdb_path = tmp_path / "legacy.duckdb"
    dconn = duckdb.connect(str(duckdb_path))
    dconn.execute(
        "CREATE TABLE artifacts (artifact_id VARCHAR PRIMARY KEY, digest VARCHAR)"
    )
    dconn.execute("INSERT INTO artifacts VALUES ('A1', 'sha256:abc')")
    dconn.close()

    manifest = build_manifest(
        manifest_id="multi-format",
        sources=[
            _source("md", "board.md", kind="markdown", domain="taskboards"),
            _source(
                "json",
                "goals.json",
                kind="json",
                domain="objectives",
                schema_identity="test/goals@1",
            ),
            _source("jsonl", "events.jsonl", kind="jsonl", domain="events"),
            _source("sqlite", "legacy.sqlite3", kind="sqlite", domain="leases"),
            _source("duckdb", "legacy.duckdb", kind="duckdb", domain="artifacts"),
        ],
        conflict_policy=ConflictPolicy.REJECT,
        mode=ImportMode.PREVIEW,
    )
    importer = _importer(tmp_path)
    source_snapshots = {
        path: path.read_bytes()
        for path in (
            tmp_path / "board.md",
            tmp_path / "goals.json",
            tmp_path / "events.jsonl",
            sqlite_path,
            duckdb_path,
        )
    }

    receipt = importer.preview(manifest)

    assert receipt.outcome == OUTCOME_PREVIEW
    assert receipt.mode == "preview"
    assert len(receipt.accepted_rows) == 8  # 2 md + 2 goals + 2 events + 1 lease + 1 artifact
    assert receipt.conflict_count == 0
    domains = {row.domain for row in receipt.accepted_rows}
    assert domains == {
        "taskboards",
        "objectives",
        "events",
        "leases",
        "artifacts",
    }
    for row in receipt.accepted_rows:
        assert row.source_digest.startswith("sha256:")
        assert row.parser_version == DEFAULT_PARSER_VERSION

    # Source immutability: import never mutates declared sources.
    for path, original in source_snapshots.items():
        assert path.read_bytes() == original


def test_exact_replay_is_noop_with_same_receipt(tmp_path: Path) -> None:
    _write(
        tmp_path / "tasks.json",
        json.dumps([{"task_id": "T1", "status": "todo"}]),
    )
    manifest = build_manifest(
        manifest_id="replay",
        sources=[_source("tasks", "tasks.json", kind="json", domain="taskboards")],
        mode=ImportMode.APPLY,
        conflict_policy=ConflictPolicy.REJECT,
    )
    importer = _importer(tmp_path)

    first = importer.apply(manifest)
    second = importer.apply(manifest)

    assert first.outcome == OUTCOME_APPLIED
    assert second.outcome == OUTCOME_REPLAYED
    assert second.replayed is True
    assert second.receipt_cid == first.receipt_cid
    assert second.import_key == first.import_key
    assert [row.to_dict() for row in second.accepted_rows] == [
        row.to_dict() for row in first.accepted_rows
    ]

    # Authoritative rows are not double-inserted.
    rows = importer.list_accepted_rows(first.import_key)
    assert len(rows) == 1
    assert rows[0].record_id == "T1"
    assert rows[0].source_digest == first.accepted_rows[0].source_digest
    assert rows[0].parser_version == DEFAULT_PARSER_VERSION


def test_preview_replay_returns_same_receipt_cid(tmp_path: Path) -> None:
    _write(tmp_path / "one.json", json.dumps({"id": "x", "value": 1}))
    manifest = build_manifest(
        manifest_id="preview-replay",
        sources=[_source("one", "one.json", kind="json")],
        mode=ImportMode.PREVIEW,
    )
    importer = _importer(tmp_path)
    first = importer.preview(manifest)
    second = importer.preview(manifest)
    assert first.receipt_cid == second.receipt_cid
    assert second.replayed is True
    assert first.outcome == OUTCOME_PREVIEW
    assert second.outcome == OUTCOME_REPLAYED


def test_conflict_reject_does_not_last_write_wins(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    _write(
        tmp_path / "b.json",
        json.dumps([{"task_id": "T1", "status": "completed"}]),
    )
    manifest = build_manifest(
        manifest_id="conflict-reject",
        sources=[
            _source("a", "a.json", kind="json"),
            _source("b", "b.json", kind="json"),
        ],
        conflict_policy=ConflictPolicy.REJECT,
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert receipt.conflict_count == 1
    assert receipt.accepted_rows == ()
    assert len(receipt.rejected_rows) == 2
    assert all(row.reason == "conflict_reject" for row in receipt.rejected_rows)


def test_conflict_quarantine_holds_both_authorities(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    _write(
        tmp_path / "b.json",
        json.dumps([{"task_id": "T1", "status": "completed"}]),
    )
    manifest = build_manifest(
        manifest_id="conflict-quarantine",
        sources=[
            _source("a", "a.json", kind="json"),
            _source("b", "b.json", kind="json"),
        ],
        conflict_policy=ConflictPolicy.QUARANTINE,
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert receipt.accepted_rows == ()
    assert len(receipt.quarantined_rows) == 1
    quarantined = receipt.quarantined_rows[0]
    assert quarantined.record_id == "T1"
    assert set(quarantined.source_ids) == {"a", "b"}
    assert quarantined.reason == "conflict_quarantine"


def test_conflict_select_uses_explicit_selected_source(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    _write(
        tmp_path / "b.json",
        json.dumps([{"task_id": "T1", "status": "completed"}]),
    )
    manifest = build_manifest(
        manifest_id="conflict-select",
        sources=[
            _source("a", "a.json", kind="json", select_priority=1),
            _source("b", "b.json", kind="json", select_priority=10),
        ],
        conflict_policy=ConflictPolicy.SELECT,
        selected_sources=("a",),
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0].source_id == "a"
    assert receipt.accepted_rows[0].payload["status"] == "todo"
    assert receipt.accepted_rows[0].disposition == "select"
    assert len(receipt.rejected_rows) == 1
    assert receipt.rejected_rows[0].source_id == "b"


def test_conflict_merge_quarantines_field_conflicts() -> None:
    result = reconcile_records(
        [
            ParsedRecord(
                source_id="a",
                source_digest="sha256:" + ("11" * 32),
                parser_version=DEFAULT_PARSER_VERSION,
                domain="taskboards",
                record_id="T1",
                payload={"task_id": "T1", "status": "todo", "owner": "alpha"},
            ),
            ParsedRecord(
                source_id="b",
                source_digest="sha256:" + ("22" * 32),
                parser_version=DEFAULT_PARSER_VERSION,
                domain="taskboards",
                record_id="T1",
                payload={"task_id": "T1", "status": "todo", "priority": "P0"},
            ),
            ParsedRecord(
                source_id="c",
                source_digest="sha256:" + ("33" * 32),
                parser_version=DEFAULT_PARSER_VERSION,
                domain="taskboards",
                record_id="T2",
                payload={"task_id": "T2", "status": "todo"},
            ),
            ParsedRecord(
                source_id="d",
                source_digest="sha256:" + ("44" * 32),
                parser_version=DEFAULT_PARSER_VERSION,
                domain="taskboards",
                record_id="T2",
                payload={"task_id": "T2", "status": "done"},
            ),
        ],
        conflict_policy=ConflictPolicy.MERGE,
    )
    accepted_ids = {row.record_id for row in result.accepted}
    assert accepted_ids == {"T1"}
    merged = next(row for row in result.accepted if row.record_id == "T1")
    assert merged.payload["owner"] == "alpha"
    assert merged.payload["priority"] == "P0"
    assert merged.disposition == "merge"
    assert merged.source_digest.startswith("sha256:")
    assert merged.parser_version == DEFAULT_PARSER_VERSION
    assert len(result.quarantined) == 1
    assert result.quarantined[0].record_id == "T2"
    assert "merge_field_conflicts" in result.quarantined[0].reason

def test_duplicate_identical_sources_accept_once(tmp_path: Path) -> None:
    payload = json.dumps([{"task_id": "T1", "status": "todo"}])
    _write(tmp_path / "a.json", payload)
    _write(tmp_path / "b.json", payload)
    manifest = build_manifest(
        manifest_id="dupes",
        sources=[
            _source("a", "a.json", kind="json"),
            _source("b", "b.json", kind="json"),
        ],
        conflict_policy=ConflictPolicy.REJECT,
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert receipt.conflict_count == 0
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0].record_id == "T1"


def test_corrupt_json_fails_strict_apply(tmp_path: Path) -> None:
    _write(tmp_path / "bad.json", "{not-json")
    manifest = build_manifest(
        manifest_id="corrupt",
        sources=[_source("bad", "bad.json", kind="json")],
        mode=ImportMode.APPLY,
        strict=True,
    )
    with pytest.raises(ImportParseError, match="corrupt|truncated"):
        _importer(tmp_path).apply(manifest)


def test_truncated_jsonl_fails_closed(tmp_path: Path) -> None:
    # Final line is incomplete JSON and file does not end with newline.
    _write(tmp_path / "events.jsonl", '{"event_id":"E1"}\n{"event_id":')
    manifest = build_manifest(
        manifest_id="truncated",
        sources=[_source("events", "events.jsonl", kind="jsonl", domain="events")],
        mode=ImportMode.APPLY,
    )
    with pytest.raises(ImportParseError, match="truncated|corrupt"):
        _importer(tmp_path).apply(manifest)


def test_unsupported_schema_rejects_rows(tmp_path: Path) -> None:
    _write(
        tmp_path / "events.jsonl",
        json.dumps({"event_id": "E1", "schema": "other/schema@9", "type": "x"})
        + "\n",
    )
    manifest = build_manifest(
        manifest_id="schema",
        sources=[
            _source(
                "events",
                "events.jsonl",
                kind="jsonl",
                domain="events",
                schema_identity="expected/schema@1",
            )
        ],
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert receipt.accepted_rows == ()
    assert len(receipt.rejected_rows) == 1
    assert receipt.rejected_rows[0].reason == "unsupported_schema"


def test_json_schema_mismatch_raises(tmp_path: Path) -> None:
    _write(
        tmp_path / "goals.json",
        json.dumps({"schema": "wrong@1", "records": [{"goal_id": "G1"}]}),
    )
    manifest = build_manifest(
        manifest_id="schema-mismatch",
        sources=[
            _source(
                "goals",
                "goals.json",
                kind="json",
                domain="objectives",
                schema_identity="right@1",
            )
        ],
        mode=ImportMode.APPLY,
    )
    with pytest.raises(ImportSchemaError, match="schema"):
        _importer(tmp_path).apply(manifest)


def test_strict_apply_is_atomic_on_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "ok.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    manifest = build_manifest(
        manifest_id="atomic",
        sources=[_source("ok", "ok.json", kind="json")],
        mode=ImportMode.APPLY,
    )
    importer = _importer(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(importer, "_commit_receipt", boom)
    with pytest.raises(ImportAtomicityError, match="atomically"):
        importer.apply(manifest)

    # Nothing durable should remain for the failed apply.
    assert importer.get_receipt(
        # reconstruct is unnecessary; just ensure no receipts table rows via list
        "missing"
    ) is None
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    with open_duckdb_connection(importer.target_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM legacy_import_receipts"
        ).fetchone()
        rows = connection.execute(
            "SELECT COUNT(*) FROM legacy_import_rows"
        ).fetchone()
    assert count is not None and count[0] == 0
    assert rows is not None and rows[0] == 0


def test_every_accepted_row_has_source_digest_and_parser_version(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "board.md",
        "## DQP-010 Import legacy state\n- Status: todo\n- Track: state-import\n",
    )
    manifest = build_manifest(
        manifest_id="provenance",
        sources=[
            _source(
                "board",
                "board.md",
                kind="markdown",
                domain="taskboards",
                parser_version="legacy-state-import/1",
            )
        ],
        mode=ImportMode.APPLY,
    )
    importer = _importer(tmp_path)
    receipt = importer.apply(manifest)
    assert len(receipt.accepted_rows) == 1
    row = receipt.accepted_rows[0]
    assert row.source_digest.startswith("sha256:")
    assert len(row.source_digest) == len("sha256:") + 64
    assert row.parser_version == "legacy-state-import/1"
    stored = importer.list_accepted_rows(receipt.import_key)
    assert len(stored) == 1
    assert stored[0].source_digest == row.source_digest
    assert stored[0].parser_version == row.parser_version


def test_source_mutation_between_observation_and_apply_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "live.json", json.dumps([{"id": "1", "v": "a"}]))
    manifest = build_manifest(
        manifest_id="mutation",
        sources=[_source("live", "live.json", kind="json")],
        mode=ImportMode.APPLY,
    )
    importer = _importer(tmp_path)

    original_verify = importer._verify_source_immutability

    def mutate_then_verify(observed):
        path.write_text(json.dumps([{"id": "1", "v": "b"}]), encoding="utf-8")
        return original_verify(observed)

    monkeypatch.setattr(importer, "_verify_source_immutability", mutate_then_verify)
    with pytest.raises(ImportSourceMutationError, match="changed after observation"):
        importer.apply(manifest)


def test_rejected_non_object_json_rows(tmp_path: Path) -> None:
    _write(tmp_path / "mixed.json", json.dumps([{"id": "ok"}, "nope", 3]))
    manifest = build_manifest(
        manifest_id="mixed",
        sources=[_source("mixed", "mixed.json", kind="json")],
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0].record_id == "ok"
    assert len(receipt.rejected_rows) == 2
    assert {row.reason for row in receipt.rejected_rows} == {"json_record_not_object"}


def test_manifest_round_trip_dict(tmp_path: Path) -> None:
    manifest = build_manifest(
        manifest_id="round-trip",
        sources=[
            {
                "source_id": "s1",
                "path": "x.jsonl",
                "kind": "jsonl",
                "domain": "events",
                "parser_version": DEFAULT_PARSER_VERSION,
            }
        ],
        conflict_policy="quarantine",
        mode="preview",
        selected_sources=(),
        metadata={"note": "fixture"},
    )
    restored = ImportManifest.from_dict(manifest.to_dict())
    assert restored.fingerprint() == manifest.fingerprint()
    assert restored.conflict_policy is ConflictPolicy.QUARANTINE
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    loaded = ImportManifest.from_path(path)
    assert loaded.manifest_id == "round-trip"
    assert loaded.sources[0].kind is ImportSourceKind.JSONL


def test_digest_helpers_are_stable() -> None:
    assert sha256_digest(b"abc") == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_select_without_choice_quarantines_equal_priority(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    _write(
        tmp_path / "b.json",
        json.dumps([{"task_id": "T1", "status": "completed"}]),
    )
    manifest = build_manifest(
        manifest_id="select-ambiguous",
        sources=[
            _source("a", "a.json", kind="json"),
            _source("b", "b.json", kind="json"),
        ],
        conflict_policy=ConflictPolicy.SELECT,
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert receipt.accepted_rows == ()
    assert len(receipt.quarantined_rows) == 1
    assert receipt.quarantined_rows[0].reason == "select_requires_explicit_choice"


def test_import_domains_cover_declared_sink_classes() -> None:
    values = {item.value for item in ImportDomain}
    assert values == {
        "objectives",
        "taskboards",
        "plan_revisions",
        "queues",
        "events",
        "statuses",
        "worktrees",
        "caches",
        "artifacts",
        "leases",
        "databases",
    }
    kinds = {item.value for item in ImportSourceKind}
    assert kinds == {"markdown", "json", "jsonl", "sqlite", "duckdb"}
    policies = {item.value for item in ConflictPolicy}
    assert policies == {"select", "merge", "quarantine", "reject"}
    assert "last-write-wins" not in policies


def test_priority_select_without_selected_sources(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", json.dumps([{"task_id": "T1", "status": "todo"}]))
    _write(
        tmp_path / "b.json",
        json.dumps([{"task_id": "T1", "status": "completed"}]),
    )
    manifest = build_manifest(
        manifest_id="priority-select",
        sources=[
            _source("a", "a.json", kind="json", select_priority=1),
            _source("b", "b.json", kind="json", select_priority=5),
        ],
        conflict_policy=ConflictPolicy.SELECT,
        mode=ImportMode.APPLY,
    )
    receipt = _importer(tmp_path).apply(manifest)
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0].source_id == "b"
    assert receipt.accepted_rows[0].payload["status"] == "completed"
