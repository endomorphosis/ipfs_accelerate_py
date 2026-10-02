"""Historical proof discovery is complete by pages, never proof admission."""
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
    BoardControlPlane, BoardControlPlaneError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBConnection


@pytest.fixture
def plane(tmp_path):
    database = tmp_path / "board.duckdb"
    result = BoardControlPlane(root=tmp_path, database_path=database)
    result._connection = DuckDBConnection(database)
    result._install_schema()
    try:
        yield result
    finally:
        result.close()


def _cache(path, count, *, malformed=False):
    with DuckDBConnection(path) as cx:
        cx.execute("CREATE TABLE proof_cache_entries (key_id VARCHAR PRIMARY KEY, key_json VARCHAR, entry_json VARCHAR)")
        cx.executemany("INSERT INTO proof_cache_entries VALUES (?, ?, ?)", [
            [f"key-{i:06d}", json.dumps({"obligation": {"id": i}}),
             "{" if malformed and i == 1 else json.dumps({"status": "proved", "authoritative": True})]
            for i in range(count)])
    return path


def test_compatibility_scan_collects_more_than_512_without_promoting_status(plane, tmp_path):
    path = _cache(tmp_path / "proof.duckdb", 1027)
    assert plane.ingest_proof_cache_files("board", [path]) == 1027
    report = plane.last_proof_cache_ingestion
    assert [page["scanned"] for page in report["pages"]] == [512, 512, 3]
    assert report["traversal_complete"] and not report["snapshot_consistent"]
    rows = plane._conn().execute("SELECT status, payload_json FROM artefact_proof_cache").fetchall()
    assert len(rows) == 1027
    for row in rows:
        assert row[0] == "historical_candidate"
        assert json.loads(row[1])["requires_fresh_verification"] is True
        assert json.loads(row[1])["proof_authority"] is False


def test_bounded_scan_retains_cursor_and_resume_is_idempotent(plane, tmp_path):
    path = _cache(tmp_path / "proof.duckdb", 515)
    assert plane.ingest_proof_cache_files("board", [path], max_pages=1) == 512
    report = plane.last_proof_cache_ingestion
    assert not report["traversal_complete"]
    page = report["pages"][-1]
    assert page["omissions"] == [{"reason": "page_budget_exhausted"}]
    remaining = plane.ingest_proof_cache_page("board", path, after_key_id=page["next_after_key_id"])
    assert remaining["ingested"] == 3 and remaining["traversal_complete"]
    assert plane.ingest_proof_cache_files("board", [path]) == 515
    assert plane._conn().execute("SELECT count(*) FROM artefact_proof_cache").fetchone()[0] == 515


def test_equal_filenames_and_key_prefixes_do_not_overwrite_other_cache(plane, tmp_path):
    first = _cache(tmp_path / "one" / "proof.duckdb", 2)
    second = _cache(tmp_path / "two" / "proof.duckdb", 2)
    assert plane.ingest_proof_cache_files("board", [first, second]) == 4
    assert plane._conn().execute("SELECT count(*) FROM artefact_proof_cache").fetchone()[0] == 4
    with DuckDBConnection(first) as cx:
        cx.execute("UPDATE proof_cache_entries SET key_id=? || key_id", ["same-prefix:" + "x" * 100])
    assert plane.ingest_proof_cache_files("another-board", [first]) == 2
    assert plane._conn().execute("SELECT count(*) FROM artefact_proof_cache WHERE board_namespace='another-board'").fetchone()[0] == 2


def test_unavailable_and_corrupt_rows_are_explicit_and_report_is_durable(plane, tmp_path):
    path = _cache(tmp_path / "proof.duckdb", 3, malformed=True)
    assert plane.ingest_proof_cache_files("board", [path, tmp_path / "missing.duckdb"]) == 2
    report = plane.last_proof_cache_ingestion
    assert not report["traversal_complete"]
    assert report["pages"][0]["omissions"] == [{"reason": "malformed_cache_row", "key_id": "key-000001"}]
    assert report["pages"][1]["omissions"] == [{"reason": "missing_or_unsupported_cache_file"}]
    stored = plane._conn().execute("SELECT payload_json FROM artefact_generic WHERE artefact_id LIKE 'proof-cache-ingestion:%'").fetchall()
    assert [json.loads(row[0]) for row in stored] == [report]


def test_missing_table_is_unavailable_not_an_empty_complete_cache(plane, tmp_path):
    path = tmp_path / "empty.duckdb"
    with DuckDBConnection(path):
        pass
    report = plane.ingest_proof_cache_page("board", path)
    assert report["status"] == "unavailable" and not report["traversal_complete"]
    assert report["omissions"][0]["reason"] == "cache_read_unavailable"


@pytest.mark.parametrize("size", [True, 0, 513, 1.5])
def test_invalid_page_bounds_are_refused_before_read(plane, tmp_path, size):
    with pytest.raises(BoardControlPlaneError):
        plane.ingest_proof_cache_page("board", tmp_path / "missing.duckdb", page_size=size)
