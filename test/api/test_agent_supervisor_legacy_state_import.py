"""Tests for LegacyStateImport@1 (DQP-010).

Covers Markdown/JSON/JSONL/SQLite/DuckDB parsing, provenance, conflict
policies (select/merge/quarantine/reject), strict atomic apply, exact replay,
corrupt input rejection, and source immutability.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
    PARSER_VERSION,
    ConflictPolicy,
    ImportConflictResolution,
    ImportDomain,
    ImportManifest,
    ImportMediaType,
    ImportMode,
    ImportSourceMutationError,
    ImportSourceSpec,
    ImportStrictError,
    LegacyStateImport,
    OUTCOME_APPLIED,
    OUTCOME_PREVIEWED,
    OUTCOME_REPLAYED,
    SourceObservation,
    build_import_manifest,
    duckdb_available,
    parse_json_records,
    parse_jsonl_records,
    parse_markdown_records,
    reconcile_records,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _markdown_board(path: Path) -> Path:
    return _write(
        path,
        """# Board

## DQP-010 Import legacy state

- Status: todo
- Priority: P0
- Track: state-import

Import Markdown, JSON, JSONL, SQLite, and DuckDB.

## DQP-011 Render exports

- Status: todo
- Priority: P1
""",
    )


def _json_tasks(path: Path) -> Path:
    return _write(
        path,
        json.dumps(
            {
                "records": [
                    {
                        "id": "DQP-010",
                        "title": "Import legacy state",
                        "status": "todo",
                        "priority": "P0",
                    },
                    {
                        "id": "DQP-011",
                        "title": "Render exports",
                        "status": "todo",
                        "priority": "P1",
                    },
                ]
            },
            indent=2,
        )
        + "\n",
    )


def _jsonl_events(path: Path) -> Path:
    lines = [
        json.dumps({"event_id": "evt-1", "task_id": "DQP-010", "kind": "created"}),
        json.dumps({"event_id": "evt-2", "task_id": "DQP-010", "kind": "queued"}),
    ]
    return _write(path, "\n".join(lines) + "\n")


def _sqlite_queue(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE queue_items (id TEXT PRIMARY KEY, task_id TEXT, status TEXT)"
        )
        connection.execute(
            "INSERT INTO queue_items VALUES ('q1', 'DQP-010', 'ready')"
        )
        connection.execute(
            "INSERT INTO queue_items VALUES ('q2', 'DQP-011', 'blocked')"
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _duckdb_leases(path: Path) -> Path:
    if not duckdb_available():
        pytest.skip("DuckDB is required for DuckDB source fixtures")
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE leases (id VARCHAR PRIMARY KEY, owner VARCHAR, task_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO leases VALUES ('lease-1', 'worker-a', 'DQP-010')"
        )
    finally:
        connection.close()
    return path


def _manifest(
    tmp_path: Path,
    sources: list[ImportSourceSpec],
    **kwargs: object,
) -> ImportManifest:
    values: dict[str, object] = {
        "import_id": "import-test-1",
        "sources": tuple(sources),
        "mode": ImportMode.PREVIEW,
        "strict": True,
        "default_conflict_policy": ConflictPolicy.REJECT,
        "target_database": str(tmp_path / "control_import.duckdb"),
    }
    values.update(kwargs)
    return ImportManifest(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cold import / contract surface
# ---------------------------------------------------------------------------


def test_cold_import_is_side_effect_free() -> None:
    import importlib

    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import"
    )
    assert module.LEGACY_STATE_IMPORT_INTERFACE == "LegacyStateImport@1"
    assert module.IMPORT_MANIFEST_INTERFACE == "ImportManifest@1"
    assert module.IMPORT_RECEIPT_INTERFACE == "ImportReceipt@1"
    assert module.PARSER_VERSION == "legacy-state-import/1"


def test_manifest_refuses_duplicate_source_ids(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    with pytest.raises(Exception, match="duplicate source_id"):
        ImportManifest(
            import_id="dup",
            sources=(
                ImportSourceSpec(
                    source_id="same",
                    path=str(path),
                    media_type=ImportMediaType.JSON,
                ),
                ImportSourceSpec(
                    source_id="same",
                    path=str(path),
                    media_type=ImportMediaType.JSON,
                ),
            ),
        )


def test_manifest_cid_is_stable(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    sources = (
        ImportSourceSpec(
            source_id="tasks",
            path=str(path),
            media_type=ImportMediaType.JSON,
            domain=ImportDomain.TASKBOARDS,
        ),
    )
    left = ImportManifest(import_id="m1", sources=sources)
    right = ImportManifest(import_id="m1", sources=sources)
    assert left.manifest_cid == right.manifest_cid
    assert left.manifest_cid.startswith("b")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_markdown_task_headings() -> None:
    text = "## TASK-1 Hello\n\n- Status: todo\n\nBody line\n"
    records = parse_markdown_records(
        text,
        source_id="md",
        source_digest="sha256:" + ("ab" * 32),
        parser_version=PARSER_VERSION,
        domain="taskboards",
    )
    assert len(records) == 1
    assert records[0].record_id == "TASK-1"
    assert records[0].payload["status"] == "todo"
    assert records[0].payload["title"] == "Hello"
    assert "Body line" in str(records[0].payload.get("body") or "")


def test_parse_json_and_jsonl() -> None:
    digest = "sha256:" + ("cd" * 32)
    json_records = parse_json_records(
        json.dumps([{"id": "a", "value": 1}, {"id": "b", "value": 2}]),
        source_id="j",
        source_digest=digest,
        parser_version=PARSER_VERSION,
        domain="generic",
    )
    assert [item.record_id for item in json_records] == ["a", "b"]

    jsonl_records = parse_jsonl_records(
        '{"id":"x"}\nnot-json\n{"id":"y"}\n',
        source_id="jl",
        source_digest=digest,
        parser_version=PARSER_VERSION,
        domain="events",
    )
    assert jsonl_records[0].record_id == "x"
    assert jsonl_records[1].rejected is True
    assert "corrupt" in jsonl_records[1].reject_reason
    assert jsonl_records[2].record_id == "y"


def test_parse_corrupt_json_is_rejected() -> None:
    records = parse_json_records(
        "{truncated",
        source_id="bad",
        source_digest="sha256:" + ("11" * 32),
        parser_version=PARSER_VERSION,
        domain="generic",
    )
    assert len(records) == 1
    assert records[0].rejected is True
    assert "corrupt" in records[0].reject_reason


# ---------------------------------------------------------------------------
# Multi-format preview with provenance
# ---------------------------------------------------------------------------


def test_preview_all_media_types_with_provenance(tmp_path: Path) -> None:
    md = _markdown_board(tmp_path / "board.md")
    js = _json_tasks(tmp_path / "tasks.json")
    jl = _jsonl_events(tmp_path / "events.jsonl")
    sq = _sqlite_queue(tmp_path / "queue.sqlite3")

    sources = [
        ImportSourceSpec(
            source_id="board-md",
            path=str(md),
            media_type=ImportMediaType.MARKDOWN,
            domain=ImportDomain.TASKBOARDS,
        ),
        ImportSourceSpec(
            source_id="tasks-json",
            path=str(js),
            media_type=ImportMediaType.JSON,
            domain=ImportDomain.TASKBOARDS,
        ),
        ImportSourceSpec(
            source_id="events-jsonl",
            path=str(jl),
            media_type=ImportMediaType.JSONL,
            domain=ImportDomain.EVENTS,
        ),
        ImportSourceSpec(
            source_id="queue-sqlite",
            path=str(sq),
            media_type=ImportMediaType.SQLITE,
            domain=ImportDomain.QUEUES,
        ),
    ]
    if duckdb_available():
        dd = _duckdb_leases(tmp_path / "leases.duckdb")
        sources.append(
            ImportSourceSpec(
                source_id="leases-duckdb",
                path=str(dd),
                media_type=ImportMediaType.DUCKDB,
                domain=ImportDomain.LEASES,
            )
        )

    # Conflicts between markdown and json on DQP-010/DQP-011: quarantine them.
    manifest = _manifest(
        tmp_path,
        sources,
        default_conflict_policy=ConflictPolicy.QUARANTINE,
        strict=False,
    )
    importer = LegacyStateImport(target_database=tmp_path / "import.duckdb")
    receipt = importer.preview(manifest)

    assert receipt.outcome == OUTCOME_PREVIEWED
    assert receipt.applied is False
    assert receipt.parser_version == PARSER_VERSION
    assert len(receipt.source_observations) == len(sources)
    for observation in receipt.source_observations:
        assert isinstance(observation, SourceObservation)
        assert observation.source_digest.startswith("sha256:")
        assert observation.parser_version == PARSER_VERSION
        assert observation.byte_size > 0

    # Every accepted row is traceable to source digest and parser version.
    for row in receipt.accepted_rows:
        assert row["source_digest"].startswith("sha256:")
        assert row["parser_version"] == PARSER_VERSION
        assert row["source_id"]
        assert row["content_cid"].startswith("b")

    # Events and queue items should be accepted (no identity overlap).
    accepted_ids = {row["record_id"] for row in receipt.accepted_rows}
    assert "evt-1" in accepted_ids
    assert "evt-2" in accepted_ids
    assert "q1" in accepted_ids
    assert "q2" in accepted_ids
    if duckdb_available():
        assert "lease-1" in accepted_ids

    # Conflicting task identities were quarantined, not last-write-wins.
    conflict_ids = {item.record_id for item in receipt.conflicts}
    assert "DQP-010" in conflict_ids
    assert "DQP-011" in conflict_ids
    for item in receipt.conflicts:
        if item.record_id in {"DQP-010", "DQP-011"}:
            assert item.policy == ConflictPolicy.QUARANTINE.value
            assert item.decision == "quarantined"


def test_preview_does_not_mutate_sources(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="tasks",
                path=str(path),
                media_type=ImportMediaType.JSON,
            )
        ],
        strict=False,
    )
    LegacyStateImport().preview(manifest)
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------------------
# Conflict policies
# ---------------------------------------------------------------------------


def test_conflict_reject_is_not_last_write_wins(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "left.json",
        json.dumps([{"id": "T1", "status": "todo", "owner": "a"}]) + "\n",
    )
    right = _write(
        tmp_path / "right.json",
        json.dumps([{"id": "T1", "status": "done", "owner": "b"}]) + "\n",
    )
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="left",
                path=str(left),
                media_type=ImportMediaType.JSON,
                domain=ImportDomain.TASKBOARDS,
            ),
            ImportSourceSpec(
                source_id="right",
                path=str(right),
                media_type=ImportMediaType.JSON,
                domain=ImportDomain.TASKBOARDS,
            ),
        ],
        default_conflict_policy=ConflictPolicy.REJECT,
        strict=False,
    )
    receipt = LegacyStateImport().preview(manifest)
    assert receipt.accepted_rows == ()
    assert len(receipt.rejected_rows) == 2
    assert receipt.conflicts[0].decision == "rejected"
    # Explicitly not last-write-wins: right did not silently win.
    assert all(row["record_id"] == "T1" for row in receipt.rejected_rows)


def test_conflict_select_uses_declared_source(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "left.json",
        json.dumps([{"id": "T1", "status": "todo"}]) + "\n",
    )
    right = _write(
        tmp_path / "right.json",
        json.dumps([{"id": "T1", "status": "done"}]) + "\n",
    )
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="left",
                path=str(left),
                media_type=ImportMediaType.JSON,
            ),
            ImportSourceSpec(
                source_id="right",
                path=str(right),
                media_type=ImportMediaType.JSON,
            ),
        ],
        default_conflict_policy=ConflictPolicy.SELECT,
        conflict_resolutions=(
            ImportConflictResolution(
                domain=ImportDomain.GENERIC.value,
                record_id="T1",
                policy=ConflictPolicy.SELECT,
                selected_source_id="left",
            ),
        ),
        strict=False,
    )
    receipt = LegacyStateImport().preview(manifest)
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0]["source_id"] == "left"
    assert receipt.accepted_rows[0]["payload"]["status"] == "todo"
    assert receipt.conflicts[0].decision == "selected"


def test_conflict_merge_complementary_fields(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "left.json",
        json.dumps([{"id": "T1", "title": "Import", "status": "todo"}]) + "\n",
    )
    right = _write(
        tmp_path / "right.json",
        json.dumps([{"id": "T1", "title": "Import", "owner": "lane-1"}]) + "\n",
    )
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="left",
                path=str(left),
                media_type=ImportMediaType.JSON,
            ),
            ImportSourceSpec(
                source_id="right",
                path=str(right),
                media_type=ImportMediaType.JSON,
            ),
        ],
        default_conflict_policy=ConflictPolicy.MERGE,
        strict=False,
    )
    receipt = LegacyStateImport().preview(manifest)
    assert len(receipt.accepted_rows) == 1
    payload = receipt.accepted_rows[0]["payload"]
    assert payload["title"] == "Import"
    assert payload["status"] == "todo"
    assert payload["owner"] == "lane-1"
    assert receipt.conflicts[0].decision == "merged"
    # Merge provenance retains contributing digests/parser versions.
    merge_meta = payload["_import_merge"]
    assert "left" in merge_meta["merged_from_source_ids"]
    assert "right" in merge_meta["merged_from_source_ids"]


def test_conflict_merge_quarantines_contradictions(tmp_path: Path) -> None:
    left = _write(
        tmp_path / "left.json",
        json.dumps([{"id": "T1", "status": "todo"}]) + "\n",
    )
    right = _write(
        tmp_path / "right.json",
        json.dumps([{"id": "T1", "status": "done"}]) + "\n",
    )
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="left",
                path=str(left),
                media_type=ImportMediaType.JSON,
            ),
            ImportSourceSpec(
                source_id="right",
                path=str(right),
                media_type=ImportMediaType.JSON,
            ),
        ],
        default_conflict_policy=ConflictPolicy.MERGE,
        strict=False,
    )
    receipt = LegacyStateImport().preview(manifest)
    assert receipt.accepted_rows == ()
    assert len(receipt.quarantined_rows) == 2
    assert receipt.conflicts[0].decision == "quarantined"


def test_exact_duplicate_sources_deduplicate(tmp_path: Path) -> None:
    payload = json.dumps([{"id": "T1", "status": "todo"}]) + "\n"
    left = _write(tmp_path / "left.json", payload)
    right = _write(tmp_path / "right.json", payload)
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="left",
                path=str(left),
                media_type=ImportMediaType.JSON,
            ),
            ImportSourceSpec(
                source_id="right",
                path=str(right),
                media_type=ImportMediaType.JSON,
            ),
        ],
        strict=False,
    )
    receipt = LegacyStateImport().preview(manifest)
    assert len(receipt.accepted_rows) == 1
    assert receipt.conflicts[0].decision == "deduplicated"


# ---------------------------------------------------------------------------
# Strict atomic apply + exact replay
# ---------------------------------------------------------------------------


def test_strict_apply_refuses_rejected_rows(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.jsonl", '{"id":"ok"}\n{truncated\n')
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="events",
                path=str(path),
                media_type=ImportMediaType.JSONL,
                domain=ImportDomain.EVENTS,
            )
        ],
        mode=ImportMode.APPLY,
        strict=True,
    )
    importer = LegacyStateImport()
    with pytest.raises(ImportStrictError, match="strict import refused"):
        importer.apply(manifest)
    assert importer.list_accepted_rows() == []


def test_strict_apply_commits_atomically_and_replays(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    target = tmp_path / "import.duckdb"
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="tasks",
                path=str(path),
                media_type=ImportMediaType.JSON,
                domain=ImportDomain.TASKBOARDS,
            )
        ],
        mode=ImportMode.APPLY,
        strict=True,
        target_database=str(target) if duckdb_available() else "",
    )
    importer = LegacyStateImport(
        target_database=target if duckdb_available() else None
    )
    first = importer.apply(manifest)
    assert first.outcome == OUTCOME_APPLIED
    assert first.applied is True
    assert first.replayed is False
    assert first.receipt_cid.startswith("b")
    assert len(first.accepted_rows) == 2
    for row in first.accepted_rows:
        assert row["source_digest"].startswith("sha256:")
        assert row["parser_version"] == PARSER_VERSION

    # Source immutability: original file untouched.
    assert "DQP-010" in path.read_text(encoding="utf-8")

    second = importer.apply(manifest)
    assert second.outcome == OUTCOME_REPLAYED
    assert second.replayed is True
    assert second.applied is True
    assert second.receipt_cid == first.receipt_cid
    assert second.manifest_cid == first.manifest_cid
    assert [dict(row) for row in second.accepted_rows] == [
        dict(row) for row in first.accepted_rows
    ]

    # Durable store (when DuckDB present) still has exactly the accepted rows.
    rows = importer.list_accepted_rows(manifest.import_id)
    assert len(rows) == 2
    assert {row["record_id"] for row in rows} == {"DQP-010", "DQP-011"}


def test_apply_detects_source_mutation_before_replay(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="tasks",
                path=str(path),
                media_type=ImportMediaType.JSON,
            )
        ],
        mode=ImportMode.APPLY,
        strict=True,
        target_database="",
    )
    importer = LegacyStateImport()
    first = importer.apply(manifest)
    assert first.outcome == OUTCOME_APPLIED

    # Mutate source after successful apply; exact replay must fail closed.
    path.write_text(
        json.dumps([{"id": "DQP-010", "title": "mutated"}]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ImportSourceMutationError, match="mutated"):
        importer.apply(manifest)


def test_non_strict_apply_accepts_partial_valid_rows(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mixed.jsonl",
        '{"id":"good","value":1}\n{bad\n{"id":"also-good","value":2}\n',
    )
    manifest = _manifest(
        tmp_path,
        [
            ImportSourceSpec(
                source_id="mixed",
                path=str(path),
                media_type=ImportMediaType.JSONL,
            )
        ],
        mode=ImportMode.APPLY,
        strict=False,
        target_database="",
    )
    receipt = LegacyStateImport().apply(manifest)
    assert receipt.outcome == OUTCOME_APPLIED
    assert {row["record_id"] for row in receipt.accepted_rows} == {
        "good",
        "also-good",
    }
    assert len(receipt.rejected_rows) == 1


# ---------------------------------------------------------------------------
# Unsupported schema / reconcile unit
# ---------------------------------------------------------------------------


def test_unsupported_markdown_record_id_rejected() -> None:
    records = parse_markdown_records(
        "## !!!bad Title\n\nbody\n",
        source_id="md",
        source_digest="sha256:" + ("ef" * 32),
        parser_version=PARSER_VERSION,
        domain="taskboards",
    )
    assert records[0].rejected is True
    assert "unsupported" in records[0].reject_reason


def test_reconcile_records_unit() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
        ParsedRecord,
    )
    from types import MappingProxyType

    digest = "sha256:" + ("aa" * 32)
    left = ParsedRecord(
        domain="generic",
        record_id="T1",
        payload=MappingProxyType({"id": "T1", "a": 1}),
        source_id="s1",
        source_digest=digest,
        parser_version=PARSER_VERSION,
        media_type="json",
    )
    right = ParsedRecord(
        domain="generic",
        record_id="T1",
        payload=MappingProxyType({"id": "T1", "b": 2}),
        source_id="s2",
        source_digest=digest,
        parser_version=PARSER_VERSION,
        media_type="json",
    )
    accepted, rejected, quarantined, conflicts = reconcile_records(
        [left, right],
        default_policy=ConflictPolicy.MERGE,
        resolutions=(),
    )
    assert len(accepted) == 1
    assert rejected == []
    assert quarantined == []
    assert conflicts[0].decision == "merged"


def test_build_import_manifest_helper(tmp_path: Path) -> None:
    path = _json_tasks(tmp_path / "tasks.json")
    manifest = build_import_manifest(
        "helper-1",
        [
            {
                "source_id": "tasks",
                "path": str(path),
                "media_type": "json",
                "domain": "taskboards",
            }
        ],
        mode="preview",
        strict=True,
    )
    assert manifest.import_id == "helper-1"
    assert manifest.sources[0].media_type is ImportMediaType.JSON
    receipt = LegacyStateImport().run(manifest)
    assert receipt.outcome == OUTCOME_PREVIEWED
    assert len(receipt.accepted_rows) == 2


def test_from_paths_infers_media_types(tmp_path: Path) -> None:
    md = _markdown_board(tmp_path / "board.md")
    js = _json_tasks(tmp_path / "tasks.json")
    jl = _jsonl_events(tmp_path / "events.jsonl")
    sq = _sqlite_queue(tmp_path / "queue.sqlite3")
    paths = [md, js, jl, sq]
    if duckdb_available():
        paths.append(_duckdb_leases(tmp_path / "leases.duckdb"))
    manifest = ImportManifest.from_paths(
        "from-paths",
        paths,
        mode=ImportMode.PREVIEW,
        strict=False,
        default_conflict_policy=ConflictPolicy.QUARANTINE,
    )
    media = {source.media_type for source in manifest.sources}
    assert ImportMediaType.MARKDOWN in media
    assert ImportMediaType.JSON in media
    assert ImportMediaType.JSONL in media
    assert ImportMediaType.SQLITE in media
    receipt = LegacyStateImport().preview(manifest)
    assert receipt.source_observations
    for observation in receipt.source_observations:
        assert observation.source_digest.startswith("sha256:")


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_duckdb_source_and_durable_store_round_trip(tmp_path: Path) -> None:
    dd = _duckdb_leases(tmp_path / "leases.duckdb")
    target = tmp_path / "control.duckdb"
    manifest = ImportManifest(
        import_id="duckdb-import",
        sources=(
            ImportSourceSpec(
                source_id="leases",
                path=str(dd),
                media_type=ImportMediaType.DUCKDB,
                domain=ImportDomain.LEASES,
            ),
        ),
        mode=ImportMode.APPLY,
        strict=True,
        target_database=str(target),
    )
    importer = LegacyStateImport(target_database=target)
    receipt = importer.apply(manifest)
    assert receipt.outcome == OUTCOME_APPLIED
    assert len(receipt.accepted_rows) == 1
    assert receipt.accepted_rows[0]["record_id"] == "lease-1"
    assert receipt.accepted_rows[0]["source_digest"].startswith("sha256:")

    loaded = importer.get_receipt("duckdb-import")
    assert loaded is not None
    assert loaded.receipt_cid == receipt.receipt_cid

    replay = importer.apply(manifest)
    assert replay.outcome == OUTCOME_REPLAYED
    assert replay.receipt_cid == receipt.receipt_cid
