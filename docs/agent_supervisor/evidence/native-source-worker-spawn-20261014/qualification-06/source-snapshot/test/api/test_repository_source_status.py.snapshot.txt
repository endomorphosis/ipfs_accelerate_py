"""Bounded status delivery and exact, freshness-bound retained cache pages."""
import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_session as sessions
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import _native_scope
from test.api.test_repository_source_session import RecordingIndex


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


class CacheRecordingIndex(RecordingIndex):
    """Exercise real retained classification without repeating native SQL tests."""

    def refresh(self, root, manifest):
        receipt = super().refresh(root, manifest)
        receipt["retained_opaque_cache"] = _native_scope(manifest)[1]
        return receipt


def make(tmp_path, *, count=5):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def value():\n    return 1\n")
    if count:
        cache = source / "__pycache__"
        cache.mkdir(mode=0o750)
        for number in range(count):
            (cache / f"private-cache-{number:04d}.pyc").write_bytes(b"\x00\xffcache")
        (cache / "alias").symlink_to("private-cache-0000.pyc")
        (cache / "empty").mkdir()
    return sessions.RepositorySourceSession(source, tmp_path / "state", index_factory=CacheRecordingIndex)


def error(code, action):
    with pytest.raises(sessions.RepositorySourceSessionError) as caught:
        action()
    assert caught.value.code == code


def test_summary_preserves_internal_evidence_and_exact_native_coverage(tmp_path):
    with make(tmp_path) as session:
        full = session.status()
        retained = copy.deepcopy(full["native_index"]["retained_opaque_cache"])
        summary = session.status_summary()
        cache = summary["native_index"]["retained_opaque_cache"]
        assert "entries" not in cache and "roots" not in cache
        assert b"private-cache" not in encoded(summary)
        assert cache == {
            "schema": sessions.CACHE_SUMMARY_SCHEMA,
            "inventory_schema": retained["schema"],
            "inventory_sha256": retained["inventory_sha256"],
            "roots_count": len(retained["roots"]),
            "details_tool": "repository_cache_inventory",
            **{key: retained[key] for key in (*sessions._CACHE_COUNTS, "classification", *sessions._CACHE_AUTHORITY)},
        }
        assert summary["source_current"] is True and summary["refresh_required"] is False
        assert summary["index_coverage"] == full["index_coverage"]
        assert summary["native_index"]["coverage"] == full["native_index"]["coverage"]
        assert summary["datasets_head"] == full["datasets_head"]
        assert session.status() == full
        cache["inventory_sha256"] = "0" * 64
        summary["native_index"]["coverage"]["fixture"] = False
        assert session.status()["native_index"]["retained_opaque_cache"] == retained
        assert session.status()["native_index"]["coverage"]["fixture"] is True


def test_summary_size_stays_constant_with_cache_population(tmp_path):
    sizes = []
    for count in (1, 220):
        directory = tmp_path / str(count)
        directory.mkdir()
        with make(directory, count=count) as session:
            summary = session.status_summary()
            sizes.append(len(encoded(summary)))
            if count == 220:
                assert len(encoded(session.status())) > 10 * sizes[-1]
                assert summary["native_index"]["retained_opaque_cache"]["entry_count"] == 223
    # Only numeric counts and byte totals grow; no path population is delivered.
    assert abs(sizes[1] - sizes[0]) < 100


def test_pages_reconstruct_exact_original_inventory_digest_and_roots(tmp_path):
    with make(tmp_path, count=105) as session:
        retained = session.status()["native_index"]["retained_opaque_cache"]
        digest = session.status_summary()["native_index"]["retained_opaque_cache"]["inventory_sha256"]
        rows, offset, page_count = [], 0, 0
        while offset is not None:
            page = session.cache_inventory(generation=1, expected_inventory_sha256=digest,
                                           offset=offset, limit=40)
            assert page["generation"] == 1 and page["inventory_sha256"] == digest
            assert page["total"] == retained["entry_count"]
            assert page["offset"] == offset and len(page["entries"]) <= 40
            assert page["roots_count"] == len(retained["roots"])
            assert page["inventory_schema"] == retained["schema"]
            assert page["proof_authority"] is page["execution_authority"] is page["completion_authority"] is False
            rows.extend(page["entries"])
            offset = page["next_offset"]
            page_count += 1
        assert page_count == 3
        assert [row["path"] for row in rows] == sorted(retained["entries"])
        roots = [row["path"] for row in rows if row["cache_root"]]
        entries = {row["path"]: {key: value for key, value in row.items() if key not in {"path", "cache_root"}}
                   for row in rows}
        inventory = dict(schema=page["inventory_schema"], roots=roots, entries=entries)
        assert entries == retained["entries"] and roots == retained["roots"]
        assert hashlib.sha256(encoded(inventory)).hexdigest() == digest
        entries["__pycache__"]["mode"] = 0o777
        assert session.status()["native_index"]["retained_opaque_cache"] == retained


def test_empty_inventory_has_digest_and_terminal_empty_page(tmp_path):
    with make(tmp_path, count=0) as session:
        summary = session.status_summary()["native_index"]["retained_opaque_cache"]
        assert summary["entry_count"] == summary["roots_count"] == 0
        page = session.cache_inventory(generation=1, expected_inventory_sha256=summary["inventory_sha256"])
        assert page["entries"] == [] and page["total"] == 0 and page["next_offset"] is None


@pytest.mark.parametrize("kwargs", [dict(offset=-1), dict(offset=True), dict(offset=513), dict(offset=1.0),
                                    dict(limit=0), dict(limit=True), dict(limit=101), dict(limit="1")])
def test_cache_page_pagination_bounds(tmp_path, kwargs):
    with make(tmp_path) as session:
        digest = session.status_summary()["native_index"]["retained_opaque_cache"]["inventory_sha256"]
        error("invalid_pagination", lambda: session.cache_inventory(
            generation=1, expected_inventory_sha256=digest, **kwargs))
        assert session.status_summary()["source_current"] is True


def test_page_offset_must_not_skip_beyond_retained_population(tmp_path):
    with make(tmp_path) as session:
        cache = session.status_summary()["native_index"]["retained_opaque_cache"]
        page = session.cache_inventory(generation=1, expected_inventory_sha256=cache["inventory_sha256"],
                                       offset=cache["entry_count"])
        assert page["entries"] == [] and page["next_offset"] is None
        error("invalid_pagination", lambda: session.cache_inventory(generation=1,
            expected_inventory_sha256=cache["inventory_sha256"], offset=cache["entry_count"] + 1))


@pytest.mark.parametrize("digest", [None, True, "", "g" * 64, "A" * 64, "0" * 63])
def test_cache_pages_require_valid_inventory_digest(tmp_path, digest):
    with make(tmp_path) as session:
        error("invalid_inventory_hash", lambda: session.cache_inventory(
            generation=1, expected_inventory_sha256=digest))


def test_mismatched_digest_cannot_mix_pages_or_invalidate_good_session(tmp_path):
    with make(tmp_path) as session:
        error("stale_inventory_hash", lambda: session.cache_inventory(
            generation=1, expected_inventory_sha256="0" * 64))
        assert session.status_summary()["source_current"] is True


def test_cache_pages_and_summary_fence_external_changes_then_refresh(tmp_path):
    with make(tmp_path) as session:
        first = session.status_summary()
        digest = first["native_index"]["retained_opaque_cache"]["inventory_sha256"]
        (session.workspace / "__pycache__/private-cache-0000.pyc").write_bytes(b"changed")
        error("stale_source", lambda: session.cache_inventory(generation=1, expected_inventory_sha256=digest))
        stale = session.status_summary()
        assert stale["source_current"] is False and stale["refresh_required"] is True
        assert stale["native_index"]["retained_opaque_cache"]["inventory_sha256"] == digest
        session.refresh(generation=1)
        current = session.status_summary()
        successor = current["native_index"]["retained_opaque_cache"]["inventory_sha256"]
        assert successor != digest and current["generation"] == 2
        error("stale_generation", lambda: session.cache_inventory(generation=1, expected_inventory_sha256=digest))
        error("stale_inventory_hash", lambda: session.cache_inventory(generation=2, expected_inventory_sha256=digest))
        assert session.cache_inventory(generation=2, expected_inventory_sha256=successor)["generation"] == 2


def test_source_change_during_page_assembly_is_refused(tmp_path, monkeypatch):
    with make(tmp_path) as session:
        digest = session.status_summary()["native_index"]["retained_opaque_cache"]["inventory_sha256"]
        current, calls = session._current, []
        def mutate_after_entry_fence():
            current()
            calls.append(True)
            if len(calls) == 1:
                (session.workspace / "__pycache__/new.pyc").write_bytes(b"new cache")
        monkeypatch.setattr(session, "_current", mutate_after_entry_fence)
        error("stale_source", lambda: session.cache_inventory(generation=1, expected_inventory_sha256=digest))
        assert session.status_summary()["refresh_required"] is True


def test_closed_session_rejects_summary_and_detail_queries(tmp_path):
    session = make(tmp_path)
    digest = session.status_summary()["native_index"]["retained_opaque_cache"]["inventory_sha256"]
    session.close()
    error("session_closed", session.status_summary)
    error("session_closed", lambda: session.cache_inventory(generation=1, expected_inventory_sha256=digest))


def test_missing_inventory_is_explicit_not_invented_empty_coverage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with sessions.RepositorySourceSession(source, tmp_path / "state", index_factory=RecordingIndex) as session:
        assert session.status()["source_current"] is True
        error("cache_inventory_unavailable", session.status_summary)
        error("cache_inventory_unavailable", lambda: session.cache_inventory(
            generation=1, expected_inventory_sha256="0" * 64))
