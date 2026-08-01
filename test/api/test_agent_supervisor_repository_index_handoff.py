"""SCA-215: publish the current healthy authoritative repository index.

Validates the handoff triple (repository-index / current / analyzer_health)
binds the same roots, refuses to reuse compiler-unavailable rows after a
toolchain identity change, classifies every eligible path as success or a
typed bounded failure, and never issues provider/model/LLM calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

import scripts.index_repository_contracts as index_script
from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_health import (
    POLYGLOT_AST_HEALTH_EVIDENCE,
    POLYGLOT_AST_HEALTH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (
    PolyglotASTLimits,
    PolyglotASTProvider,
    PolyglotASTProviderError,
    PolyglotASTReason,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    ParserStatus,
    RepositoryIndexer,
    RepositoryIndexerError,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
    RepositorySnapshotStats,
)
from scripts.index_repository_contracts import (
    DEFAULT_TYPESCRIPT_VERSION,
    HANDOFF_EVIDENCE,
    HANDOFF_SCHEMA,
    invalidate_stale_compiler_unavailable_state,
    previous_index_has_compiler_unavailable,
    publish_authoritative_handoff,
    resolve_handoff_root,
)


_ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
_SUPERPROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _ACCELERATE_ROOT / "scripts" / "index_repository_contracts.py"
_SCA_DATA = (
    _SUPERPROJECT_ROOT
    / "data"
    / "agent_supervisor"
    / "swissknife_contract_assurance"
)


def _digest(payload: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _disposition(
    path: str,
    kind: CoverageKind,
    payload: bytes = b"",
    *,
    status: GitStatus = GitStatus.CLEAN,
) -> CoverageDisposition:
    return CoverageDisposition(
        path=path,
        kind=kind,
        git_status=status,
        entry_kind=EntryKind.REGULAR,
        reason_code=f"fixture_{kind.value}",
        policy_rule=f"fixture:{kind.value}",
        content_digest=_digest(payload) if payload else "",
        git_mode="100644",
        git_object_id="a" * 40,
        tracked=True,
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
        deleted_path_count=0,
        untracked_path_count=0,
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


class _UnavailableTypescriptProvider(PolyglotASTProvider):
    """Provider that always reports compiler_unavailable for JS/TS."""

    def extract(self, source, language, **kwargs):
        normalized = str(language or "").split("@", 1)[0].casefold()
        if normalized in {"typescript", "javascript", "tsx", "jsx"}:
            raise PolyglotASTProviderError(
                PolyglotASTReason.COMPILER_UNAVAILABLE,
                "the local TypeScript compiler API is unavailable",
            )
        return super().extract(source, language, **kwargs)


def _scope_policy() -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "swissknife-symbolic-contract-scope@1"
        ),
        "schemaVersion": 1,
        "scopeId": "handoff-fixture@1",
        "primaryRepository": "fixture",
        "primaryRoot": ".",
        "providerScopes": [],
        "skipPrefixes": [],
        "skipDirectoryNames": [".git"],
        "dependencyDirectoryNames": ["node_modules"],
        "dependencyLockFiles": [],
        "dependencyManifestFiles": [],
        "dispositionRules": {
            "semanticExtensions": [".py", ".ts", ".js"],
            "structuredExtensions": [".json"],
            "textExtensions": [".md"],
            "binaryExtensions": [],
            "generatedSuffixes": [],
            "generatedPathParts": [],
        },
        "workingTreeOverlay": {
            "mode": "tracked_plus_allowlisted_untracked_source",
            "allowDirtyAnalysis": True,
            "allowlistedUntrackedSuffixes": [".py", ".ts", ".js", ".json", ".md"],
            "allowlistedUntrackedExactNames": [],
        },
        "silentExclusionsAllowed": False,
        "trackedCoverageRequired": 1.0,
    }


def _eligible_rows(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = index.get("rows") or []
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("parser_status")
        in {"indexed", "cache_hit", "parse_failure"}
    ]


def _assert_eligible_paths_typed(index: Mapping[str, Any]) -> None:
    for row in _eligible_rows(index):
        status = row.get("parser_status")
        assert status in {"indexed", "cache_hit", "parse_failure"}
        if status == "parse_failure":
            reason = str(row.get("parser_reason") or row.get("reason_code") or "")
            assert reason.strip(), f"untyped failure at {row.get('path')}"


def _publication_evidence(result) -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "sca-authoritative-publication-evidence@1"
        ),
        "snapshot_id": result.snapshot.snapshot_id,
        "fresh_snapshot_id": result.snapshot.snapshot_id,
        "baseline_result_id": "fixture-baseline-result",
        "coverage_root": "fixture-coverage-root",
        "stages": [
            {
                "name": name,
                "completeness": "complete",
                "reason_codes": [],
                "root_id": f"fixture-{name}",
            }
            for name in ("repository_index", "extraction", "catalog", "publish")
        ],
        "execution": {
            "mode": "deterministic-symbolic",
            "llm_call_count": 0,
            "provider_call_count": 0,
            "model_call_count": 0,
            "model_invocation_enabled": False,
        },
    }


def _authoritative_provider() -> PolyglotASTProvider:
    typescript_path = (
        _SUPERPROJECT_ROOT
        / "swissknife"
        / "node_modules"
        / "typescript"
        / "lib"
        / "typescript.js"
    )
    if not typescript_path.is_file():
        pytest.skip("reviewed TypeScript 5.9.3 toolchain is unavailable")
    return PolyglotASTProvider(
        PolyglotASTLimits(process_timeout_seconds=5.0),
        typescript_path=str(typescript_path),
        expected_typescript_version=DEFAULT_TYPESCRIPT_VERSION,
    )


def test_previous_compiler_unavailable_detection(tmp_path: Path) -> None:
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/sca-repository-index@1",
        "index_id": "fixture",
        "rows": [
            {
                "path": "src/a.ts",
                "parser_status": "parse_failure",
                "parser_reason": (
                    "compiler_unavailable: the local TypeScript compiler "
                    "API is unavailable"
                ),
                "parser_identity": "old-parser",
            }
        ],
    }
    (tmp_path / "current.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    cache_marker = tmp_path / "analysis-cache" / "keep.json"
    cache_marker.parent.mkdir()
    cache_marker.write_text("{}", encoding="utf-8")
    assert previous_index_has_compiler_unavailable(tmp_path) is True
    receipt = invalidate_stale_compiler_unavailable_state(
        tmp_path, current_parser_identity="new-parser"
    )
    assert receipt["had_compiler_unavailable"] is True
    assert receipt["identity_changed"] is True
    assert not (tmp_path / "current.json").exists()
    assert cache_marker.is_file()


def test_compiler_unavailable_rows_are_not_reused_after_toolchain_change(
    tmp_path: Path,
) -> None:
    source = b"export const value: number = 1;\n"
    files = {"src/mod.ts": source}
    dispositions = [
        _disposition("src/mod.ts", CoverageKind.SEMANTIC_AST, source),
        _disposition("README.md", CoverageKind.TEXT_REFERENCE, b"# ok\n"),
    ]
    snapshot = _snapshot(tmp_path, dispositions)
    index_root = tmp_path / "index"

    # Phase 1: toolchain unavailable → typed compiler_unavailable failure.
    unavailable = RepositoryIndexer(
        index_root,
        provider=_UnavailableTypescriptProvider(
            PolyglotASTLimits(process_timeout_seconds=5.0)
        ),
        health_thresholds=AnalyzerHealthThresholds(
            max_parser_failures=10,
            max_parser_failure_ratio=1.0,
            max_excluded_file_ratio=1.0,
            require_canaries=True,
        ),
    )
    first = unavailable.build(snapshot, source_loader=_loader(files))
    assert first.rows[0].path == "README.md" or first.row_for_path("src/mod.ts")
    failed = first.row_for_path("src/mod.ts")
    assert failed is not None
    assert failed.parser_status is ParserStatus.PARSE_FAILURE
    assert "compiler_unavailable" in (failed.parser_reason or "")
    first_identity = failed.parser_identity

    # Phase 2: real TypeScript toolchain → identity changes and path re-parses.
    ts_candidates = [
        _SUPERPROJECT_ROOT
        / "swissknife"
        / "node_modules"
        / "typescript"
        / "lib"
        / "typescript.js",
        Path("swissknife/node_modules/typescript/lib/typescript.js"),
    ]
    typescript_path = next(
        (str(path.resolve()) for path in ts_candidates if path.is_file()),
        None,
    )
    if typescript_path is None:
        pytest.skip("TypeScript 5.9.3 compiler is not installed in this checkout")

    invalidate_stale_compiler_unavailable_state(
        index_root, current_parser_identity="force-reparse-identity"
    )
    healthy_provider = PolyglotASTProvider(
        PolyglotASTLimits(process_timeout_seconds=15.0),
        typescript_path=typescript_path,
        expected_typescript_version=DEFAULT_TYPESCRIPT_VERSION,
    )
    second_indexer = RepositoryIndexer(
        index_root,
        provider=healthy_provider,
        health_thresholds=AnalyzerHealthThresholds(
            max_parser_failures=10,
            max_parser_failure_ratio=1.0,
            max_excluded_file_ratio=1.0,
        ),
    )
    assert second_indexer.parser_identity != first_identity
    second = second_indexer.build(snapshot, source_loader=_loader(files))
    repaired = second.row_for_path("src/mod.ts")
    assert repaired is not None
    assert repaired.parser_identity == second_indexer.parser_identity
    assert repaired.parser_identity != first_identity
    assert "compiler_unavailable" not in (repaired.parser_reason or "")
    assert repaired.parser_status in {
        ParserStatus.INDEXED,
        ParserStatus.PARSE_FAILURE,
    }
    if repaired.parser_status is ParserStatus.PARSE_FAILURE:
        # Still a typed bounded failure, never silent reuse of the old reason.
        assert repaired.parser_reason
        assert repaired.parser_reason != failed.parser_reason


def test_publish_authoritative_handoff_binds_roots_and_zero_llm(
    tmp_path: Path,
) -> None:
    files = {
        "service.py": b"def service():\n    return True\n",
        "README.md": b"# fixture\n",
    }
    snapshot = _snapshot(
        tmp_path,
        [
            _disposition(
                "service.py", CoverageKind.SEMANTIC_AST, files["service.py"]
            ),
            _disposition(
                "README.md", CoverageKind.TEXT_REFERENCE, files["README.md"]
            ),
        ],
    )
    indexer = RepositoryIndexer(
        tmp_path / "work",
        health_thresholds=AnalyzerHealthThresholds(
            max_excluded_file_ratio=1.0,
        ),
    )
    result = indexer.build(snapshot, source_loader=_loader(files))
    handoff_root = tmp_path / "sca"
    provider = _authoritative_provider()

    handoff = publish_authoritative_handoff(
        result,
        handoff_root=handoff_root,
        provider=provider,
        typescript_path=None,
        typescript_version="",
        llm_call_count=0,
        provider_call_count=0,
        model_call_count=0,
        publication_evidence=_publication_evidence(result),
    )

    assert handoff["schema"] == HANDOFF_SCHEMA
    assert handoff["evidence_id"] == HANDOFF_EVIDENCE
    assert handoff["llm_call_count"] == 0
    assert handoff["provider_call_count"] == 0
    assert handoff["model_call_count"] == 0
    assert handoff["roots_agree"] is True
    assert handoff["index_id"] == result.index_id
    assert handoff["snapshot_id"] == result.snapshot.snapshot_id
    assert handoff["coverage_root"] == "fixture-coverage-root"
    assert handoff["health_root"] not in {"healthy", "partial", "unhealthy"}
    assert handoff["generation"].startswith("sha256-")

    repository_index = json.loads(
        (handoff_root / "baseline" / "repository-index.json").read_text(
            encoding="utf-8"
        )
    )
    current = json.loads(
        (handoff_root / "baseline" / "current.json").read_text(encoding="utf-8")
    )
    health = json.loads(
        (handoff_root / "analyzer_health" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert repository_index["index_id"] == current["index_id"] == result.index_id
    assert (
        repository_index["snapshot"]["snapshot_id"]
        == current["snapshot"]["snapshot_id"]
        == result.snapshot.snapshot_id
    )
    assert repository_index["ast_index_id"] == current["ast_index_id"]
    assert health["schema"] == POLYGLOT_AST_HEALTH_SCHEMA
    assert health["evidence_id"] == POLYGLOT_AST_HEALTH_EVIDENCE
    assert health["status"] == "healthy"
    assert health["safe_for_completion_reasoning"] is True
    assert (handoff_root / "authoritative").is_symlink()
    assert (
        handoff_root / "baseline" / "repository-index.json"
    ).is_symlink()
    _assert_eligible_paths_typed(repository_index)
    assert not handoff["untyped_failure_paths"]
    assert result.health.status is AnalyzerHealthStatus.HEALTHY


def test_failed_health_publication_preserves_current_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"ok.py": b"X = 1\n"}
    snapshot = _snapshot(
        tmp_path,
        [_disposition("ok.py", CoverageKind.SEMANTIC_AST, files["ok.py"])],
    )
    result = RepositoryIndexer(tmp_path / "idx").build(
        snapshot,
        source_loader=_loader(files),
    )
    handoff_root = tmp_path / "sca"
    current_path = handoff_root / "baseline" / "current.json"
    current_path.parent.mkdir(parents=True)
    current_path.write_bytes(b'{"index_id":"prior"}\n')

    def fail_health(*_args, **_kwargs):
        raise OSError("injected health publication failure")

    monkeypatch.setattr(
        index_script,
        "write_polyglot_ast_health_report",
        fail_health,
    )

    with pytest.raises(OSError, match="injected health"):
        publish_authoritative_handoff(
            result,
            handoff_root=handoff_root,
            provider=_authoritative_provider(),
            typescript_path=None,
            typescript_version="",
            publication_evidence=_publication_evidence(result),
        )

    assert current_path.read_bytes() == b'{"index_id":"prior"}\n'


def test_publish_handoff_rejects_nonzero_llm_counts(tmp_path: Path) -> None:
    files = {"ok.py": b"X = 1\n"}
    snapshot = _snapshot(
        tmp_path,
        [_disposition("ok.py", CoverageKind.SEMANTIC_AST, files["ok.py"])],
    )
    result = RepositoryIndexer(tmp_path / "idx").build(
        snapshot, source_loader=_loader(files)
    )
    with pytest.raises(ValueError, match="forbids non-zero"):
        publish_authoritative_handoff(
            result,
            handoff_root=tmp_path / "sca",
            provider=PolyglotASTProvider(),
            typescript_path=None,
            typescript_version="",
            llm_call_count=1,
        )


def test_unhealthy_handoff_preserves_prior_authority(tmp_path: Path) -> None:
    files = {"broken.py": b"def broken(\n"}
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
    result = RepositoryIndexer(
        tmp_path / "idx",
        health_thresholds=AnalyzerHealthThresholds(
            max_parser_failures=0,
            max_parser_failure_ratio=0.0,
            max_excluded_file_ratio=1.0,
        ),
    ).build(snapshot, source_loader=_loader(files))
    assert result.health.status is AnalyzerHealthStatus.UNHEALTHY

    handoff_root = tmp_path / "sca"
    current = handoff_root / "baseline" / "current.json"
    current.parent.mkdir(parents=True)
    current.write_bytes(b'{"index_id":"prior"}\n')
    with pytest.raises(
        RepositoryIndexerError,
        match="requires healthy analyzer status",
    ):
        publish_authoritative_handoff(
            result,
            handoff_root=handoff_root,
            provider=_authoritative_provider(),
            typescript_path=None,
            typescript_version="",
            publication_evidence=_publication_evidence(result),
        )
    assert current.read_bytes() == b'{"index_id":"prior"}\n'
    assert not (handoff_root / "authoritative").exists()


def test_stale_snapshot_evidence_publishes_nothing(tmp_path: Path) -> None:
    files = {"ok.py": b"X = 1\n"}
    snapshot = _snapshot(
        tmp_path,
        [_disposition("ok.py", CoverageKind.SEMANTIC_AST, files["ok.py"])],
    )
    result = RepositoryIndexer(tmp_path / "idx").build(
        snapshot,
        source_loader=_loader(files),
    )
    evidence = _publication_evidence(result)
    evidence["fresh_snapshot_id"] = "changed-after-scan"
    with pytest.raises(RepositoryIndexerError, match="changed after"):
        publish_authoritative_handoff(
            result,
            handoff_root=tmp_path / "sca",
            provider=_authoritative_provider(),
            typescript_path=None,
            typescript_version="",
            publication_evidence=evidence,
        )
    assert not (tmp_path / "sca").exists()


def test_resolve_handoff_root_from_baseline_output(tmp_path: Path) -> None:
    baseline = tmp_path / "data" / "baseline"
    baseline.mkdir(parents=True)
    resolved = resolve_handoff_root(
        output_root=baseline,
        handoff_root=None,
        publish_handoff=True,
    )
    assert resolved == baseline.parent.resolve()
    explicit = resolve_handoff_root(
        output_root=baseline,
        handoff_root=tmp_path / "explicit",
        publish_handoff=False,
    )
    assert explicit is None


def test_cli_rejects_analysis_only_handoff_publication(tmp_path: Path) -> None:
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
    handoff_root = tmp_path / "sca"
    output = handoff_root / "baseline"

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--repo-root",
            str(repository),
            "--scope-config",
            str(scope),
            "--output-root",
            str(output),
            "--handoff-root",
            str(handoff_root),
            "--publish-handoff",
            "--skip-extraction",
            "--shadow",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "authoritative handoff mode rejected" in completed.stderr
    assert "--require-healthy is mandatory" in completed.stderr
    assert "--shadow is analysis-only" in completed.stderr
    assert "--skip-extraction cannot publish" in completed.stderr
    assert not (handoff_root / "baseline" / "current.json").exists()
    assert not (handoff_root / "authoritative").exists()


def test_published_sca_artifacts_agree_when_present() -> None:
    """Validate durable SCA-215 outputs when the authoritative index is present.

    The full-tree publication is produced outside this unit suite.  When the
    files exist they must satisfy the handoff acceptance contract.
    """

    repository_index_path = _SCA_DATA / "baseline" / "repository-index.json"
    current_path = _SCA_DATA / "baseline" / "current.json"
    health_path = _SCA_DATA / "analyzer_health" / "report.json"
    if not (
        repository_index_path.is_file()
        and current_path.is_file()
        and health_path.is_file()
    ):
        pytest.skip("authoritative SCA handoff artifacts are not published yet")

    repository_index = json.loads(
        repository_index_path.read_text(encoding="utf-8")
    )
    current = json.loads(current_path.read_text(encoding="utf-8"))
    health = json.loads(health_path.read_text(encoding="utf-8"))

    assert repository_index["index_id"] == current["index_id"]
    assert (
        repository_index["snapshot"]["snapshot_id"]
        == current["snapshot"]["snapshot_id"]
    )
    assert repository_index.get("ast_index_id") == current.get("ast_index_id")
    _assert_eligible_paths_typed(repository_index)

    eligible = _eligible_rows(repository_index)
    unavailable = [
        row
        for row in eligible
        if "compiler_unavailable"
        in f"{row.get('parser_reason', '')} {row.get('reason_code', '')}".casefold()
    ]
    assert unavailable == []
    assert health.get("status") == "healthy"
    assert health.get("safe_for_completion_reasoning") is True
    assert health.get("completion_blocker") is False
    assert health.get("schema") == POLYGLOT_AST_HEALTH_SCHEMA
    assert health.get("evidence_id") == POLYGLOT_AST_HEALTH_EVIDENCE
    assert (_SCA_DATA / "authoritative").is_symlink()
    # Zero LLM/provider/model is a handoff invariant for published artifacts.
    for key in ("llm_call_count", "provider_call_count", "model_call_count"):
        if key in health:
            assert health[key] == 0
        if key in repository_index:
            assert repository_index[key] == 0


def test_handoff_module_exports_default_typescript_version() -> None:
    assert DEFAULT_TYPESCRIPT_VERSION == "5.9.3"
    assert HANDOFF_EVIDENCE == "SCAEV022INDEX"
