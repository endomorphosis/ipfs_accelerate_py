#!/usr/bin/env python3
"""Build the incremental SwissKnife repository contract index."""

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
from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (  # noqa: E402
    MismatchAnalysis,
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
    RepositoryIndex,
    RepositoryIndexer,
    RepositoryIndexerError,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (  # noqa: E402
    RepositorySnapshotError,
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


def _atomic_text(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
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


def _coverage_report(result: RepositoryIndex) -> dict[str, Any]:
    """Return a bounded ledger whose row IDs bind the complete index records."""

    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "sca-repository-coverage@1"
        ),
        "snapshot_id": result.snapshot_id,
        "index_id": result.index_id,
        "ast_index_id": result.ast_index_id,
        "scope_id": result.snapshot.scope_id,
        "scope_policy_id": result.snapshot.scope_policy_id,
        "head_commit_id": result.snapshot.head_commit_id,
        "head_tree_id": result.snapshot.head_tree_id,
        "index_tree_id": result.snapshot.index_tree_id,
        "is_clean": result.snapshot.is_clean,
        "health": result.health.to_dict(),
        "stats": result.build_stats.to_dict(),
        "rows": [
            {
                "path": row.path,
                "row_id": row.row_id,
                "disposition_kind": row.disposition_kind.value,
                "declared_kind": row.declared_kind.value,
                "reason_code": row.reason_code,
                "parser_status": row.parser_status.value,
                "parser_reason": row.parser_reason,
                "language": row.language,
                "tracked": row.tracked,
                "overlay": row.overlay,
            }
            for row in result.rows
        ],
    }


def _contract_analysis_receipt(result: RepositoryIndex) -> MismatchAnalysis:
    reason_codes = ["repository_index_only"]
    if result.health.status is not AnalyzerHealthStatus.HEALTHY:
        reason_codes.append("contract_analysis_withheld_until_analyzer_healthy")
        reason_codes.extend(
            f"analyzer_health:{reason}" for reason in result.health.reasons
        )
    else:
        reason_codes.append("contract_claim_pipeline_not_run")
    return MismatchAnalysis(
        snapshot_id=result.snapshot_id,
        findings=(),
        reason_codes=tuple(reason_codes),
    )


def _summary_markdown(
    result: RepositoryIndex,
    analysis: MismatchAnalysis,
) -> str:
    stats = result.build_stats
    analysis_status = (
        "withheld: analyzer health is not healthy"
        if result.health.status is not AnalyzerHealthStatus.HEALTHY
        else "not run: no contract-claim pipeline was provided"
    )
    health_reasons = ", ".join(result.health.reasons) or "none"
    return "\n".join(
        (
            "# SwissKnife Symbolic Contract Baseline",
            "",
            f"- Snapshot ID: `{result.snapshot_id}`",
            f"- Repository index ID: `{result.index_id}`",
            f"- AST index ID: `{result.ast_index_id}`",
            f"- Analyzer health: `{result.health.status.value}`",
            f"- Analyzer health reasons: `{health_reasons}`",
            (
                "- Safe for completion reasoning: "
                f"`{str(result.safe_for_completion_reasoning).lower()}`"
            ),
            f"- Tracked paths: `{stats.tracked_path_count}`",
            f"- Indexed rows: `{stats.row_count}`",
            f"- Parser-eligible paths: `{stats.eligible_parser_path_count}`",
            f"- Parser failures: `{stats.parse_failure_count}`",
            f"- Unsupported parsers: `{stats.unsupported_parser_count}`",
            f"- Contract analysis: `{analysis_status}`",
            f"- Contract findings: `{len(analysis.findings)}`",
            f"- Contract analysis ID: `{analysis.analysis_id}`",
            "- Model calls: `0`",
            "",
            (
                "An empty findings list is not evidence of contract parity while "
                "contract analysis is withheld or not run."
            ),
            "",
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot every tracked SwissKnife path and build a complete "
            "incremental, CAS-backed AST/coverage index."
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
        help="durable index root (contains the sole current.json pointer)",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help=(
            "analysis-only compatibility flag; indexing never mutates source "
            "or backlog state"
        ),
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
        provider = PolyglotASTProvider(
            limits,
            node_executable=args.node,
            typescript_path=args.typescript_path,
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
        }
        analysis = _contract_analysis_receipt(result)
        _atomic_json(output_root / "coverage.json", _coverage_report(result))
        _atomic_json(
            output_root / "contract_findings.json",
            analysis.to_dict(),
        )
        _atomic_text(
            output_root / "summary.md",
            _summary_markdown(result, analysis),
        )
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
