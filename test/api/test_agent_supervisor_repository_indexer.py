from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (
    PolyglotASTProvider,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    DEFAULT_MAX_COMPACT_ROW_BYTES,
    ParserStatus,
    RepositoryIndexBoundsExceeded,
    RepositoryIndexIntegrityError,
    RepositoryIndexer,
    canonical_repository_index_bytes,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
    RepositorySnapshotStats,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _CountingProvider(PolyglotASTProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def extract(self, source, language, **kwargs):
        self.calls.append((language, str(source)))
        return super().extract(source, language, **kwargs)


def _disposition(
    path: str,
    kind: CoverageKind,
    payload: bytes = b"",
    *,
    status: GitStatus = GitStatus.CLEAN,
    rename_from: str = "",
    tracked: bool = True,
    entry_kind: EntryKind = EntryKind.REGULAR,
) -> CoverageDisposition:
    return CoverageDisposition(
        path=path,
        kind=kind,
        git_status=status,
        entry_kind=entry_kind,
        reason_code=f"fixture_{kind.value}",
        policy_rule=f"fixture:{kind.value}",
        content_digest=_digest(payload) if payload else "",
        git_mode=(
            "120000" if entry_kind is EntryKind.SYMLINK else "100644"
        ),
        git_object_id=hashlib.sha1(path.encode()).hexdigest(),
        rename_from=rename_from,
        tracked=tracked,
        overlay=status is not GitStatus.CLEAN,
    )


def _snapshot(
    root: Path,
    dispositions: list[CoverageDisposition],
    *,
    revision: str = "1",
) -> RepositorySnapshot:
    tracked = [item for item in dispositions if item.tracked]
    stats = RepositorySnapshotStats(
        tracked_path_count=len(tracked),
        disposition_count=len(dispositions),
        overlay_path_count=sum(item.overlay for item in dispositions),
        excluded_path_count=sum(
            item.kind is CoverageKind.EXCLUDED for item in dispositions
        ),
        dependency_identity_count=0,
        gitlink_count=0,
        dirty_path_count=sum(
            item.git_status is not GitStatus.CLEAN for item in dispositions
        ),
        deleted_path_count=sum(
            item.git_status
            in {GitStatus.DELETED, GitStatus.STAGED_DELETION}
            for item in dispositions
        ),
        untracked_path_count=sum(
            item.git_status is GitStatus.UNTRACKED for item in dispositions
        ),
        semantic_path_count=sum(
            item.kind is CoverageKind.SEMANTIC_AST for item in dispositions
        ),
        unsupported_path_count=sum(
            item.kind is CoverageKind.UNSUPPORTED for item in dispositions
        ),
        hashed_bytes=0,
    )
    return RepositorySnapshot(
        primary_root=".",
        head_commit_id=revision.rjust(40, "a")[-40:],
        head_tree_id=revision.rjust(40, "b")[-40:],
        index_tree_id=revision.rjust(40, "c")[-40:],
        scope_policy_id="fixture-policy@1",
        scope_id="fixture-scope@1",
        dispositions=tuple(dispositions),
        dependency_identities=(),
        gitlinks=(),
        stats=stats,
        repository_root=str(root),
        git_directory=str(root / ".git"),
    )


def _loader(files: dict[str, bytes]):
    def load(disposition: CoverageDisposition) -> bytes:
        return files[disposition.path]

    return load


def _blob_path(indexer: RepositoryIndexer, reference: dict) -> Path:
    return indexer.cas.store._blob_path(
        indexer.cas.store._coerce_blob_reference(reference)
    )


def test_default_loader_indexes_symlink_identity_without_following_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "target.js").write_text(
        "export const followed = false;\n",
        encoding="utf-8",
    )
    target_text = b"target.js"
    (source / "link.js").symlink_to(target_text.decode())
    snapshot = _snapshot(
        source,
        [
            _disposition(
                "link.js",
                CoverageKind.SEMANTIC_AST,
                target_text,
                entry_kind=EntryKind.SYMLINK,
            )
        ],
    )
    indexer = RepositoryIndexer(tmp_path / "index")

    result = indexer.build(snapshot)

    row = result.row_for_path("link.js")
    assert row is not None and row.source_ref
    assert indexer.cas.read(row.source_ref) == target_text


def test_complete_body_free_index_is_bounded_reused_and_deterministic(
    tmp_path: Path,
) -> None:
    secret = "literal-never-embed-73941"
    files = {
        "src/service.py": (
            f"def dispatch(request):\n    marker = {secret!r}\n"
            "    return request\n"
        ).encode(),
        "schemas/tool.json": json.dumps(
            {
                "$id": "tool",
                "properties": {"request": {"type": "string"}},
            }
        ).encode(),
    }
    dispositions = [
        _disposition(
            "README.md", CoverageKind.TEXT_REFERENCE, b"# fixture\n"
        ),
        _disposition(
            "assets/data.bin", CoverageKind.UNSUPPORTED, b"\x00\x01"
        ),
        _disposition(
            "schemas/tool.json",
            CoverageKind.STRUCTURED_DATA,
            files["schemas/tool.json"],
        ),
        _disposition(
            "src/service.py",
            CoverageKind.SEMANTIC_AST,
            files["src/service.py"],
        ),
    ]
    snapshot = _snapshot(tmp_path / "source", dispositions)
    provider = _CountingProvider()
    indexer = RepositoryIndexer(
        tmp_path / "index",
        provider=provider,
        health_thresholds=AnalyzerHealthThresholds(
            max_excluded_file_ratio=1.0
        ),
    )

    cold = indexer.build(snapshot, source_loader=_loader(files))
    cold_bytes = cold.canonical_bytes
    warm = indexer.build(snapshot, source_loader=_loader(files))

    assert cold.path_count == len(dispositions)
    assert warm.build_stats.reused_path_count == 2
    assert warm.build_stats.parsed_path_count == 0
    assert len(provider.calls) == 2
    assert cold.index_id == warm.index_id
    assert cold_bytes == warm.canonical_bytes
    assert canonical_repository_index_bytes(
        json.loads(cold_bytes)
    ) == cold_bytes
    assert cold.health.status is AnalyzerHealthStatus.HEALTHY
    assert cold.safe_for_completion_reasoning

    rendered = cold_bytes.decode()
    assert secret not in rendered
    assert "return request" not in rendered
    assert all(
        row.serialized_size <= DEFAULT_MAX_COMPACT_ROW_BYTES
        for row in cold.rows
    )
    source_row = cold.row_for_path("src/service.py")
    assert source_row is not None
    assert source_row.source_ref and source_row.ast_ref
    assert (
        indexer.cas.read(source_row.source_ref)
        == files["src/service.py"]
    )
    assert "source" not in indexer.cas.read_json(source_row.ast_ref)

    loaded = indexer.load_current()
    assert loaded.index_id == cold.index_id
    assert loaded.canonical_bytes == cold_bytes

    cache_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "index" / "analysis-cache").glob(
            "entries/*/*.json"
        )
    )
    assert secret not in cache_text
    assert "return request" not in cache_text


def test_same_content_reuse_stays_bound_to_parser_language(
    tmp_path: Path,
) -> None:
    payload = b"value = 1\n"
    files = {
        "src/example.js": payload,
        "src/example.py": payload,
    }
    snapshot = _snapshot(
        tmp_path / "source",
        [
            _disposition(
                path,
                CoverageKind.SEMANTIC_AST,
                body,
            )
            for path, body in files.items()
        ],
    )
    indexer = RepositoryIndexer(tmp_path / "index")

    cold = indexer.build(snapshot, source_loader=_loader(files))
    warm = indexer.build(snapshot, source_loader=_loader(files))

    assert cold.index_id == warm.index_id
    assert cold.invalidations == warm.invalidations == ()
    assert cold.row_for_path("src/example.js").language == "javascript"
    assert cold.row_for_path("src/example.py").language == "python"
    assert warm.row_for_path("src/example.js").language == "javascript"
    assert warm.row_for_path("src/example.py").language == "python"


def test_rename_reuse_and_exact_change_delete_invalidations(
    tmp_path: Path,
) -> None:
    first_files = {
        "a.py": b"def alpha():\n    return 1\n",
        "b.py": b"def beta():\n    return 1\n",
        "c.py": b"def gamma():\n    return 1\n",
    }
    first = _snapshot(
        tmp_path,
        [
            _disposition(path, CoverageKind.SEMANTIC_AST, payload)
            for path, payload in first_files.items()
        ],
        revision="1",
    )
    provider = _CountingProvider()
    indexer = RepositoryIndexer(tmp_path / "index", provider=provider)
    indexer.build(first, source_loader=_loader(first_files))

    second_files = {
        "renamed.py": first_files["a.py"],
        "b.py": b"def beta():\n    return 2\n",
    }
    second = _snapshot(
        tmp_path,
        [
            _disposition(
                "a.py",
                CoverageKind.SEMANTIC_AST,
                first_files["a.py"],
                status=GitStatus.STAGED_DELETION,
            ),
            _disposition(
                "b.py",
                CoverageKind.SEMANTIC_AST,
                second_files["b.py"],
                status=GitStatus.MODIFIED,
            ),
            _disposition(
                "c.py",
                CoverageKind.SEMANTIC_AST,
                first_files["c.py"],
                status=GitStatus.STAGED_DELETION,
            ),
            _disposition(
                "renamed.py",
                CoverageKind.SEMANTIC_AST,
                second_files["renamed.py"],
                status=GitStatus.RENAMED,
                rename_from="a.py",
            ),
        ],
        revision="2",
    )
    result = indexer.build(second, source_loader=_loader(second_files))

    assert result.build_stats.reused_path_count == 1
    assert result.build_stats.renamed_reuse_count == 1
    assert result.build_stats.parsed_path_count == 1
    assert result.ast_index.stats.renamed_path_count == 1
    assert result.ast_index.stats.changed_path_count == 1
    assert result.ast_index.stats.deleted_path_count == 1
    assert result.ast_index.paths == ("b.py", "renamed.py")
    assert {item.reason for item in result.invalidations} == {
        "blob_changed",
        "path_deleted",
    }
    assert {item.path for item in result.invalidations} == {"b.py", "c.py"}
    assert result.row_for_path("a.py").parser_status is ParserStatus.DELETED
    assert len(provider.calls) == 4


def test_corrupt_source_and_ast_blobs_are_reparsed_and_repaired(
    tmp_path: Path,
) -> None:
    files = {"service.py": b"def service():\n    return 1\n"}
    snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "service.py",
                CoverageKind.SEMANTIC_AST,
                files["service.py"],
            )
        ],
    )
    provider = _CountingProvider()
    indexer = RepositoryIndexer(tmp_path / "index", provider=provider)
    first = indexer.build(snapshot, source_loader=_loader(files))
    row = first.rows[0]
    assert row.source_ref and row.ast_ref

    _blob_path(indexer, dict(row.source_ref)).write_bytes(b"corrupt source")
    _blob_path(indexer, dict(row.ast_ref)).write_bytes(b"{corrupt ast")

    recovered = indexer.build(snapshot, source_loader=_loader(files))

    assert recovered.build_stats.reused_path_count == 0
    assert recovered.build_stats.parsed_path_count == 1
    assert recovered.build_stats.corruption_recovery_count == 2
    assert len(provider.calls) == 2
    recovered_row = recovered.rows[0]
    assert indexer.cas.read(recovered_row.source_ref) == files["service.py"]
    assert (
        indexer.cas.read_json(recovered_row.ast_ref)["record_id"]
        == recovered_row.ast_record_id
    )
    assert indexer.load_current().index_id == recovered.index_id


def test_concurrent_readers_only_observe_complete_current_indexes(
    tmp_path: Path,
) -> None:
    first_files = {"value.py": b"VALUE = 1\n"}
    second_files = {"value.py": b"VALUE = 2\n"}
    first_snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "value.py",
                CoverageKind.SEMANTIC_AST,
                first_files["value.py"],
            )
        ],
        revision="1",
    )
    second_snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "value.py",
                CoverageKind.SEMANTIC_AST,
                second_files["value.py"],
                status=GitStatus.MODIFIED,
            )
        ],
        revision="2",
    )
    indexer = RepositoryIndexer(tmp_path / "index")
    first = indexer.build(
        first_snapshot, source_loader=_loader(first_files)
    )

    start = threading.Event()
    stop = threading.Event()
    observed: list[str] = []
    errors: list[BaseException] = []

    def reader() -> None:
        start.wait()
        while not stop.is_set():
            try:
                current = indexer.load_current()
                assert current.path_count == 1
                observed.append(current.index_id)
            except BaseException as exc:  # captured for the parent assertion
                errors.append(exc)
                stop.set()

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for thread in threads:
        thread.start()
    start.set()
    second = indexer.build(
        second_snapshot, source_loader=_loader(second_files)
    )
    stop.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert observed
    assert set(observed).issubset({first.index_id, second.index_id})
    assert indexer.load_current().index_id == second.index_id


def test_parser_health_thresholds_fail_closed_for_partial_analysis(
    tmp_path: Path,
) -> None:
    files = {"broken.py": b"def broken(:\n    pass\n"}
    snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "broken.py",
                CoverageKind.SEMANTIC_AST,
                files["broken.py"],
            )
        ],
    )
    partial = RepositoryIndexer(
        tmp_path / "partial",
        health_thresholds=AnalyzerHealthThresholds(
            max_parser_failures=1,
            max_parser_failure_ratio=1.0,
            max_excluded_file_ratio=1.0,
        ),
    ).build(snapshot, source_loader=_loader(files))
    unhealthy = RepositoryIndexer(
        tmp_path / "unhealthy",
        health_thresholds=AnalyzerHealthThresholds(
            max_parser_failures=0,
            max_parser_failure_ratio=1.0,
            max_excluded_file_ratio=1.0,
        ),
    ).build(snapshot, source_loader=_loader(files))

    assert partial.rows[0].disposition_kind is CoverageKind.PARSE_FAILURE
    assert partial.rows[0].parser_status is ParserStatus.PARSE_FAILURE
    assert partial.health.status is AnalyzerHealthStatus.PARTIAL
    assert "parser_failures_within_budget" in partial.health.reasons
    assert not partial.safe_for_completion_reasoning
    assert unhealthy.health.status is AnalyzerHealthStatus.UNHEALTHY
    assert "parser_failure_budget_exceeded" in unhealthy.health.reasons
    assert not unhealthy.safe_for_completion_reasoning


def test_seed_inventory_accounts_for_all_5771_paths_without_loading_bodies(
    tmp_path: Path,
) -> None:
    dispositions = [
        _disposition(
            f"fixtures/path-{index:04d}.opaque",
            CoverageKind.UNSUPPORTED,
            f"seed-{index}".encode(),
        )
        for index in range(5_771)
    ]
    snapshot = _snapshot(tmp_path, dispositions)

    def forbidden_loader(_disposition: CoverageDisposition) -> bytes:
        raise AssertionError("non-parser disposition body was loaded")

    indexer = RepositoryIndexer(
        tmp_path / "index",
        max_paths=5_771,
        health_thresholds=AnalyzerHealthThresholds(
            max_excluded_file_ratio=1.0
        ),
    )
    result = indexer.build(snapshot, source_loader=forbidden_loader)

    assert result.snapshot.stats.tracked_path_count == 5_771
    assert result.path_count == 5_771
    assert result.build_stats.snapshot_path_count == 5_771
    assert result.build_stats.row_count == 5_771
    assert len({row.path for row in result.rows}) == 5_771
    assert all(
        row.parser_status is ParserStatus.NOT_APPLICABLE
        for row in result.rows
    )
    assert result.health.status is AnalyzerHealthStatus.HEALTHY
    assert indexer.load_current().path_count == 5_771


def test_row_bounds_and_manifest_integrity_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RepositoryIndexBoundsExceeded):
        RepositoryIndexer(tmp_path / "too-small", max_compact_row_bytes=255)

    files = {"ok.py": b"def ok():\n    return True\n"}
    snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "ok.py", CoverageKind.SEMANTIC_AST, files["ok.py"]
            )
        ],
    )
    indexer = RepositoryIndexer(tmp_path / "index")
    result = indexer.build(snapshot, source_loader=_loader(files))
    payload = json.loads(indexer.current_path.read_text(encoding="utf-8"))
    payload["rows"][0]["content_digest"] = "sha256:" + "0" * 64
    indexer.current_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepositoryIndexIntegrityError):
        indexer.load_current()
    assert result.rows[0].serialized_size <= DEFAULT_MAX_COMPACT_ROW_BYTES


def _scope_policy() -> dict:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "swissknife-symbolic-contract-scope@1"
        ),
        "schemaVersion": 1,
        "scopeId": "cli-fixture@1",
        "primaryRepository": "fixture",
        "primaryRoot": ".",
        "providerScopes": [],
        "skipPrefixes": [],
        "skipDirectoryNames": [".git"],
        "dependencyDirectoryNames": ["node_modules"],
        "dependencyLockFiles": [],
        "dependencyManifestFiles": [],
        "dispositionRules": {
            "semanticExtensions": [".py"],
            "structuredExtensions": [".json"],
            "textExtensions": [".md"],
            "binaryExtensions": [],
            "generatedSuffixes": [],
            "generatedPathParts": [],
        },
        "workingTreeOverlay": {
            "mode": "tracked_plus_allowlisted_untracked_source",
            "allowDirtyAnalysis": True,
            "allowlistedUntrackedSuffixes": [".py", ".json", ".md"],
            "allowlistedUntrackedExactNames": [],
        },
        "silentExclusionsAllowed": False,
        "trackedCoverageRequired": 1.0,
    }


def test_cli_indexes_real_git_snapshot_and_writes_all_evidence(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "fixture"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    (repository / "service.py").write_text(
        "def service():\n    return True\n", encoding="utf-8"
    )
    (repository / "README.md").write_text("# fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "service.py", "README.md"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"],
        check=True,
    )
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(_scope_policy()), encoding="utf-8")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts" / (
        "index_repository_contracts.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repository),
            "--scope-config",
            str(scope),
            "--output-root",
            str(output),
            "--shadow",
            "--require-healthy",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["stats"]["row_count"] == 2
    assert summary["health_status"] == "healthy"
    assert summary["shadow"] is True
    for name in (
        "current.json",
        "coverage.json",
        "repository-index.json",
        "analyzer-health.json",
        "summary.json",
    ):
        assert (output / name).is_file()
