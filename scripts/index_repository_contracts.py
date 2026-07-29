#!/usr/bin/env python3
"""Build the complete SwissKnife symbolic contract assurance baseline.

Indexes every tracked path under the reviewed scope policy, then materializes
the SCA-200 graph / proof / cache / mismatch / vulnerability baseline with
zero LLM calls. Unhealthy or incomplete stages withhold no-drift claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (  # noqa: E402
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (  # noqa: E402
    DEFAULT_MAX_ARTIFACT_BYTES,
    materialize_baseline_from_repository_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (  # noqa: E402
    PolyglotASTLimits,
    PolyglotASTProvider,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (  # noqa: E402
    DEFAULT_MAX_COMPACT_ROW_BYTES,
    DEFAULT_MAX_INDEX_PATHS,
    DEFAULT_MAX_PARSER_SOURCE_BYTES,
    DEFAULT_MAX_SOURCE_BYTES,
    RepositoryIndexer,
    RepositoryIndexerError,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (  # noqa: E402
    RepositorySnapshotError,
    default_scope_policy_path,
    load_scope_policy,
)


def _atomic_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot every tracked SwissKnife path, build a complete "
            "incremental CAS-backed AST/coverage index, and materialize the "
            "symbolic contract assurance baseline (graph/proof/cache/mismatch)."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="superproject root or exact primary Git worktree",
    )
    parser.add_argument(
        "--scope-config",
        default=None,
        help="reviewed symbolic-contract scope policy JSON",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="durable baseline root (coverage/findings/summary)",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help=(
            "analysis-only compatibility flag; indexing never mutates source "
            "or backlog state"
        ),
    )
    parser.add_argument(
        "--swissknife-root",
        default=None,
        help="optional SwissKnife checkout for expected-contract extraction",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="index only; still emit withheld baseline stages",
    )
    parser.add_argument(
        "--max-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_ARTIFACT_BYTES,
        help="hard per-file envelope for published baseline artifacts",
    )
    dirty = parser.add_mutually_exclusive_group()
    dirty.add_argument(
        "--allow-dirty",
        dest="allow_dirty",
        action="store_true",
        default=None,
        help="include policy-allowlisted working-tree overlays",
    )
    dirty.add_argument(
        "--clean-only",
        dest="allow_dirty",
        action="store_false",
        help="reject a dirty working tree",
    )
    parser.add_argument(
        "--require-healthy",
        action="store_true",
        help="return status 3 unless parser health is fully healthy",
    )
    parser.add_argument(
        "--max-paths", type=int, default=DEFAULT_MAX_INDEX_PATHS
    )
    parser.add_argument(
        "--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES
    )
    parser.add_argument(
        "--max-parser-source-bytes",
        type=int,
        default=DEFAULT_MAX_PARSER_SOURCE_BYTES,
    )
    parser.add_argument(
        "--max-total-snapshot-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-row-bytes",
        type=int,
        default=DEFAULT_MAX_COMPACT_ROW_BYTES,
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--typescript-path", default=None)
    parser.add_argument("--typescript-version", default="")
    parser.add_argument("--parser-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--parser-output-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-parser-failures", type=int, default=10)
    parser.add_argument(
        "--max-parser-failure-ratio", type=float, default=0.01
    )
    return parser


def _resolve_typescript_path(
    *,
    repo_root: str | os.PathLike[str],
    scope_config: str | os.PathLike[str] | None,
    explicit: str | os.PathLike[str] | None,
) -> str | None:
    """Resolve one bounded compiler path for the reviewed inventory root.

    An explicit path remains authoritative.  Otherwise discovery checks only
    the exact TypeScript compiler installed beneath the scope policy's
    ``primaryRoot``; it never recursively searches the checkout or silently
    borrows a compiler from an unrelated repository.
    """

    if explicit is not None:
        return str(Path(explicit).expanduser().resolve())

    repository = Path(repo_root).expanduser().resolve()
    config_path = (
        Path(scope_config)
        if scope_config is not None
        else default_scope_policy_path(repository)
    )
    policy = load_scope_policy(config_path)
    if policy.primary_root in {"", "."}:
        primary = repository
    else:
        primary = repository.joinpath(*Path(policy.primary_root).parts)
        if not primary.is_dir() and (
            repository.name == policy.primary_repository
            or (repository / ".git").exists()
        ):
            # Match repository snapshot semantics when repo_root already names
            # the reviewed primary worktree.
            primary = repository
    candidate = (
        primary
        / "node_modules"
        / "typescript"
        / "lib"
        / "typescript.js"
    ).resolve()
    return str(candidate) if candidate.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    indexer: RepositoryIndexer | None = None
    try:
        limits = PolyglotASTLimits(
            max_files=min(args.max_paths, 10_000),
            max_file_bytes=min(
                args.max_source_bytes,
                args.max_parser_source_bytes,
            ),
            max_total_bytes=max(args.max_source_bytes, 256 * 1024 * 1024),
            max_output_bytes=args.parser_output_bytes,
            process_timeout_seconds=args.parser_timeout_seconds,
        )
        typescript_path = _resolve_typescript_path(
            repo_root=args.repo_root,
            scope_config=args.scope_config,
            explicit=args.typescript_path,
        )
        provider = PolyglotASTProvider(
            limits,
            node_executable=args.node,
            typescript_path=typescript_path,
            expected_typescript_version=args.typescript_version,
        )
        thresholds = AnalyzerHealthThresholds(
            require_canaries=True,
            max_parser_failures=args.max_parser_failures,
            max_parser_failure_ratio=args.max_parser_failure_ratio,
            max_excluded_file_ratio=1.0,
            min_git_root_discovery_ratio=1.0,
            require_git_root=True,
            min_git_roots=1,
            require_complete_funnel=True,
        )
        indexer = RepositoryIndexer(
            output_root,
            provider=provider,
            health_thresholds=thresholds,
            max_compact_row_bytes=args.max_row_bytes,
            max_source_bytes=args.max_source_bytes,
            max_paths=args.max_paths,
        )
        result = indexer.index_repository(
            args.repo_root,
            scope_config_path=args.scope_config,
            allow_dirty_analysis=args.allow_dirty,
            snapshot_max_paths=args.max_paths,
            snapshot_max_file_bytes=args.max_source_bytes,
            snapshot_max_total_bytes=args.max_total_snapshot_bytes,
        )

        swissknife_root = args.swissknife_root
        if swissknife_root is None and not args.skip_extraction:
            candidate = Path(args.repo_root) / "swissknife"
            if candidate.is_dir():
                swissknife_root = str(candidate)

        if args.skip_extraction:
            from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
                materialize_contract_assurance_baseline,
            )

            baseline = materialize_contract_assurance_baseline(
                repository_index=result,
                extract_expected=False,
                output_root=output_root,
                max_file_bytes=args.max_artifact_bytes,
            )
        else:
            baseline = materialize_baseline_from_repository_index(
                result,
                output_root=output_root,
                repo_root=args.repo_root,
                swissknife_root=swissknife_root,
                max_file_bytes=args.max_artifact_bytes,
            )

        summary = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "sca-repository-index-run@1"
            ),
            "index_id": result.index_id,
            "snapshot_id": result.snapshot.snapshot_id,
            "health_status": result.health.status.value,
            "safe_for_completion_reasoning": (
                result.safe_for_completion_reasoning
            ),
            "shadow": bool(args.shadow),
            "stats": result.build_stats.to_dict(),
            "invalidations": [
                item.to_dict() for item in result.invalidations
            ],
            "baseline_result_id": baseline.result_id,
            "baseline_claims": dict(baseline.claims),
            "llm_call_count": baseline.llm_call_count,
            "contract_count": baseline.findings.get("contract_population", {}).get(
                "emitted_contract_count", 0
            ),
            "findings_root": baseline.findings.get("findings_root", ""),
            "graph_root": baseline.findings.get("graph_root", ""),
        }
        _atomic_json(output_root / "repository-index.json", result.to_dict())
        _atomic_json(
            output_root / "analyzer-health.json", result.health.to_dict()
        )
        _atomic_json(output_root / "summary.json", summary)
        sys.stdout.write(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if (
            args.require_healthy
            and result.health.status is not AnalyzerHealthStatus.HEALTHY
        ):
            return 3
        return 0
    except (
        RepositoryIndexerError,
        RepositorySnapshotError,
        OSError,
        ValueError,
    ) as exc:
        error = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "sca-repository-index-error@1"
            ),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        sys.stderr.write(
            json.dumps(error, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return 2
    finally:
        if indexer is not None:
            indexer.close()


if __name__ == "__main__":
    raise SystemExit(main())
