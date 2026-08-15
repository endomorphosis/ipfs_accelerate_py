"""Tests for DuckDBASTIndex@1 (DQP-020).

Acceptance:

* Identical blobs/parser versions deduplicate across worktrees
* Failed/unsupported parses invalidate stale facts and remain explicit unknown
* Private/ignored files and secrets are excluded
* AST rows are derived evidence, not source or semantic authority
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.duckdb_ast_index import (
    AUTHORITY_CLASS,
    DEFAULT_PARSER_ID,
    DUCKDB_AST_INDEX_INTERFACE,
    PARSE_RUN_INTERFACE,
    SOURCE_SNAPSHOT_INTERFACE,
    DuckDBASTIndex,
    ParseStatus,
    SourceFileSpec,
    content_looks_like_secret,
    duckdb_available,
    is_excluded_path,
    language_for_path,
    open_duckdb_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    ASTBlobRecord,
    build_python_ast_blob_record,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DuckDBASTIndex hermetic tests",
)


PYTHON_SERVICE = """\
from typing import Protocol

class ServiceContract(Protocol):
    def dispatch(self, request): ...

class Service:
    def dispatch(self, request):
        self.status = "running"
        return request
"""

PYTHON_CONSUMER = """\
from src.service import Service

def consume(request):
    service = Service()
    return service.dispatch(request)
"""

PYTHON_BROKEN = """\
def broken(
    return None
"""


def _open(tmp_path: Path) -> DuckDBASTIndex:
    return open_duckdb_ast_index(tmp_path / "ast_index.duckdb")


def test_interface_identities() -> None:
    assert DUCKDB_AST_INDEX_INTERFACE == "DuckDBASTIndex@1"
    assert SOURCE_SNAPSHOT_INTERFACE == "SourceSnapshot@1"
    assert PARSE_RUN_INTERFACE == "ParseRun@1"
    assert DuckDBASTIndex.INTERFACE == DUCKDB_AST_INDEX_INTERFACE
    assert AUTHORITY_CLASS == "derived_evidence"
    assert DEFAULT_PARSER_ID.startswith("python-ast@")


def test_language_and_exclusion_helpers() -> None:
    assert language_for_path("src/app.py") == "python"
    assert language_for_path("web/ui.tsx") == "tsx"
    excluded, reason = is_excluded_path(".env")
    assert excluded is True
    assert "env" in reason or "basename" in reason
    excluded, reason = is_excluded_path("secrets/api_token.txt")
    assert excluded is True
    excluded, _ = is_excluded_path("src/service.py")
    assert excluded is False
    assert content_looks_like_secret(
        "-----" + "BEGIN " + "PRIVATE " + "KEY" + "-----\nabc\n"
    )
    assert not content_looks_like_secret("def hello():\n    return 1\n")


def test_ingest_python_snapshot_persists_symbols_imports_calls(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as index:
        result = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:abc",
            worktree_id="worktree:wt-1",
            files=[
                SourceFileSpec(path="src/service.py", content=PYTHON_SERVICE),
                SourceFileSpec(path="src/consumer.py", content=PYTHON_CONSUMER),
            ],
        )
        snapshot_id = result.snapshot.snapshot_id
        assert result.indexed_file_count == 2
        assert result.parse_run.status is ParseStatus.SUCCEEDED
        assert result.snapshot.interface == SOURCE_SNAPSHOT_INTERFACE
        assert result.parse_run.interface == PARSE_RUN_INTERFACE
        assert result.to_dict()["authority"] == AUTHORITY_CLASS

        loaded = index.get_snapshot(snapshot_id)
        assert loaded is not None
        assert loaded.repository_id == "repo:demo"
        assert loaded.tree_id == "tree:abc"
        assert loaded.worktree_id == "worktree:wt-1"
        assert loaded.file_count == 2

        files = index.list_files(snapshot_id)
        assert {item["path"] for item in files} == {
            "src/service.py",
            "src/consumer.py",
        }
        # Source bodies are never retained on file rows.
        for item in files:
            assert "source" not in item
            assert str(item["content_digest"]).startswith("sha256:")

        symbols = index.list_symbols(snapshot_id)
        names = {item.qualified_name for item in symbols}
        assert "Service" in names
        assert "Service.dispatch" in names
        assert "consume" in names
        assert all(item.to_dict()["authority"] == AUTHORITY_CLASS for item in symbols)

        imports = index.list_imports(snapshot_id, path="src/consumer.py")
        assert any("Service" in str(item["module_name"]) for item in imports)

        calls = index.list_calls(snapshot_id, path="src/consumer.py")
        assert calls
        # Call rows are identity-addressed; presence of a call edge for the
        # consumer module is the durable evidence, not the raw source text.
        assert all(item["file_id"] for item in calls)
        assert all(item["caller_symbol_id"] for item in calls)
        assert all(item["callee_symbol_id"] for item in calls)

        nodes = index.list_ast_nodes(snapshot_id, path="src/service.py")
        assert any(item["node_kind"] == "module" for item in nodes)
        assert any(item["node_kind"] in {"class", "method", "function"} for item in nodes)

        meta = index.metadata()
        assert meta["interface"] == DUCKDB_AST_INDEX_INTERFACE
        assert meta["authority"] == AUTHORITY_CLASS


def test_identical_blobs_and_parser_dedupe_across_worktrees(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as index:
        first = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:base",
            worktree_id="worktree:a",
            files=[
                SourceFileSpec(path="src/service.py", content=PYTHON_SERVICE),
            ],
        )
        assert first.new_unit_count == 1
        assert first.reused_unit_count == 0
        assert index.parse_cache_size() == 1

        second = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:feature",
            worktree_id="worktree:b",
            # Same bytes at a different path/worktree must reuse the unit.
            files=[
                SourceFileSpec(
                    path="pkg/service.py",
                    content=PYTHON_SERVICE,
                ),
            ],
        )
        assert second.reused_unit_count == 1
        assert second.new_unit_count == 0
        assert index.parse_cache_size() == 1

        # Both snapshots still materialize their own path-bound symbol rows.
        first_symbols = index.list_symbols(first.snapshot.snapshot_id)
        second_symbols = index.list_symbols(second.snapshot.snapshot_id)
        assert {item.qualified_name for item in first_symbols} == {
            item.qualified_name for item in second_symbols
        }
        assert {item.path for item in first_symbols} == {"src/service.py"}
        assert {item.path for item in second_symbols} == {"pkg/service.py"}

        # Cache entry is content-addressed and body-free.
        digest = first.parse_run.file_results[0].content_digest
        cached = index.get_parse_cache_entry(digest)
        assert cached is not None
        assert cached["parser_id"] == index.parser_id
        assert cached["status"] == ParseStatus.SUCCEEDED.value
        assert "source" not in (cached.get("facts") or {})
        assert cached.get("authority") == AUTHORITY_CLASS


def test_failed_parse_invalidates_stale_facts_and_stays_unknown(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as index:
        good = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:good",
            files=[
                SourceFileSpec(path="src/mod.py", content=PYTHON_SERVICE),
            ],
        )
        assert index.list_symbols(good.snapshot.snapshot_id)
        assert index.list_frontiers(good.snapshot.snapshot_id) == ()

        # New snapshot for the same logical path with broken syntax.
        broken = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:broken",
            files=[
                SourceFileSpec(path="src/mod.py", content=PYTHON_BROKEN),
            ],
        )
        assert broken.parse_run.failed_count == 1
        assert broken.parse_run.status is ParseStatus.FAILED
        assert index.list_symbols(broken.snapshot.snapshot_id) == ()
        assert index.list_imports(broken.snapshot.snapshot_id) == ()
        assert index.list_calls(broken.snapshot.snapshot_id) == ()
        frontiers = index.list_frontiers(broken.snapshot.snapshot_id)
        assert len(frontiers) == 1
        assert frontiers[0].status is ParseStatus.FAILED
        assert frontiers[0].reason
        assert frontiers[0].path == "src/mod.py"
        # Prior good snapshot remains intact (facts are snapshot-scoped).
        assert index.list_symbols(good.snapshot.snapshot_id)


def test_unsupported_language_is_explicit_frontier(tmp_path: Path) -> None:
    with _open(tmp_path) as index:
        result = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:js",
            files=[
                SourceFileSpec(
                    path="web/app.ts",
                    content="export const x: number = 1;\n",
                ),
            ],
        )
        assert result.parse_run.unsupported_count == 1
        assert result.indexed_file_count == 0
        frontiers = index.list_frontiers(result.snapshot.snapshot_id)
        assert len(frontiers) == 1
        assert frontiers[0].status is ParseStatus.UNSUPPORTED
        assert "unsupported_language" in frontiers[0].reason
        assert index.list_symbols(result.snapshot.snapshot_id) == ()


def test_private_ignored_and_secret_files_are_excluded(
    tmp_path: Path,
) -> None:
    secret_body = (
        "-----" + "BEGIN " + "RSA " + "PRIVATE " + "KEY" + "-----\n"
        "abc\n"
        "-----" + "END " + "RSA " + "PRIVATE " + "KEY" + "-----\n"
    )
    with _open(tmp_path) as index:
        result = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:secrets",
            files=[
                SourceFileSpec(path="src/ok.py", content="def ok():\n    return 1\n"),
                SourceFileSpec(path=".env", content="TOKEN=abc\n"),
                SourceFileSpec(
                    path="secrets/token.txt",
                    content="not-a-key-but-in-secrets-dir\n",
                ),
                SourceFileSpec(
                    path="src/leaked.py",
                    content=secret_body,
                ),
                SourceFileSpec(
                    path="src/skip_me.py",
                    content="def skip():\n    return 0\n",
                    ignored=True,
                ),
            ],
        )
        assert result.excluded_file_count == 4
        assert result.indexed_file_count == 1
        paths = {item["path"] for item in index.list_files(result.snapshot.snapshot_id)}
        assert "src/ok.py" in paths
        # Excluded files still appear in the snapshot ledger (with digest only)
        # but never contribute symbols.
        assert ".env" in paths
        symbols = index.list_symbols(result.snapshot.snapshot_id)
        assert {item.qualified_name for item in symbols} == {"ok"}
        frontiers = index.list_frontiers(result.snapshot.snapshot_id)
        statuses = {item.path: item.status for item in frontiers}
        assert statuses[".env"] is ParseStatus.EXCLUDED
        assert statuses["secrets/token.txt"] is ParseStatus.EXCLUDED
        assert statuses["src/leaked.py"] is ParseStatus.EXCLUDED
        assert statuses["src/skip_me.py"] is ParseStatus.EXCLUDED


def test_prebuilt_ast_record_and_nested_symbols(tmp_path: Path) -> None:
    source = """\
class Outer:
    class Inner:
        def method(self):
            return 1

def top():
    return Outer
"""
    record = build_python_ast_blob_record(source, blob_identity="blob:nested")
    assert isinstance(record, ASTBlobRecord)
    assert "Outer.Inner.method" in record.qualified_symbols

    with _open(tmp_path) as index:
        result = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:nested",
            files=[
                SourceFileSpec(
                    path="pkg/nested.py",
                    content=source,
                    ast_record=record,
                ),
            ],
        )
        names = {
            item.qualified_name
            for item in index.list_symbols(result.snapshot.snapshot_id)
        }
        assert "Outer" in names
        assert "Outer.Inner" in names
        assert "Outer.Inner.method" in names
        assert "top" in names


def test_parser_drift_does_not_reuse_cache(tmp_path: Path) -> None:
    with open_duckdb_ast_index(
        tmp_path / "ast.duckdb",
        parser_id="python-ast@schema-1",
    ) as index:
        first = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:1",
            files=[
                SourceFileSpec(path="a.py", content="def a():\n    return 1\n"),
            ],
            parser_id="python-ast@schema-1",
        )
        assert first.new_unit_count == 1

        second = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:2",
            files=[
                SourceFileSpec(path="a.py", content="def a():\n    return 1\n"),
            ],
            parser_id="python-ast@schema-2-drift",
        )
        # Different parser identity is a distinct cache unit.
        assert second.new_unit_count == 1
        assert second.reused_unit_count == 0
        assert index.parse_cache_size() == 2


def test_snapshot_identity_is_stable_for_same_tree(tmp_path: Path) -> None:
    with _open(tmp_path) as index:
        first = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:stable",
            overlay_digest="sha256:" + "a" * 64,
            files=[
                SourceFileSpec(path="a.py", content="def a():\n    return 1\n"),
            ],
        )
        # Re-binding the same snapshot identity through get is stable.
        loaded = index.get_snapshot(first.snapshot.snapshot_id)
        assert loaded is not None
        assert loaded.snapshot_id == first.snapshot.snapshot_id
        assert loaded.to_dict()["authority"] == AUTHORITY_CLASS


def test_cold_import_has_no_side_effects() -> None:
    # Importing the module must not open databases or touch the filesystem
    # beyond normal package import.  Construction alone must not open.
    store = DuckDBASTIndex("/tmp/should-not-exist-until-open.duckdb")
    assert store.is_open is False
