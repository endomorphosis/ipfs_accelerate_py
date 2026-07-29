#!/usr/bin/env python3
"""Build the complete SwissKnife symbolic contract assurance baseline.

Indexes every tracked path under the reviewed scope policy, then materializes
the SCA-200 graph / proof / cache / mismatch / vulnerability baseline with
zero LLM calls. Unhealthy or incomplete stages withhold no-drift claims.

SCA-215 also publishes the authoritative handoff triple:

- ``baseline/repository-index.json``
- ``baseline/current.json``
- ``analyzer_health/report.json``

bound to one snapshot, one index root, one parser/toolchain identity, and zero
provider/model/LLM calls. Compiler-unavailable cache rows are never reused once
the TypeScript compiler identity changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


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
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_health import (  # noqa: E402
    POLYGLOT_AST_HEALTH_EVIDENCE,
    assess_polyglot_ast_health,
    write_polyglot_ast_health_report,
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
    default_scope_policy_path,
    load_scope_policy,
)


# Reviewed SwissKnife toolchain identity for the authoritative handoff.
DEFAULT_TYPESCRIPT_VERSION = "5.9.3"
HANDOFF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-index-handoff@1"
)
HANDOFF_EVIDENCE = "SCAEV022INDEX"
_COMPILER_UNAVAILABLE_MARKERS = (
    "compiler_unavailable",
    "node_unavailable",
    "extractor_unavailable",
    "compiler_version_mismatch",
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
    ) + b"\n"
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
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
    parser.add_argument(
        "--typescript-version",
        default="",
        help=(
            "expected TypeScript compiler version; when empty and a local "
            f"compiler is resolved, defaults to {DEFAULT_TYPESCRIPT_VERSION}"
        ),
    )
    parser.add_argument("--parser-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--parser-output-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-parser-failures", type=int, default=10)
    parser.add_argument(
        "--max-parser-failure-ratio", type=float, default=0.01
    )
    parser.add_argument(
        "--handoff-root",
        default=None,
        help=(
            "optional SCA data root; publishes baseline/repository-index.json, "
            "baseline/current.json, and analyzer_health/report.json bound to "
            "the same snapshot roots"
        ),
    )
    parser.add_argument(
        "--publish-handoff",
        action="store_true",
        help=(
            "publish the SCA-215 handoff triple under --handoff-root "
            "(or parent of --output-root when that root ends with baseline/)"
        ),
    )
    parser.add_argument(
        "--invalidate-compiler-unavailable",
        action="store_true",
        default=True,
        help=(
            "drop previous current/cache rows that carry compiler-unavailable "
            "evidence when the TypeScript toolchain identity changes "
            "(default: on)"
        ),
    )
    parser.add_argument(
        "--keep-compiler-unavailable",
        dest="invalidate_compiler_unavailable",
        action="store_false",
        help="allow previous compiler-unavailable rows to be considered for reuse",
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


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def previous_index_has_compiler_unavailable(index_root: Path) -> bool:
    """Return True when the published current index retains toolchain failures."""

    current = _read_json_object(index_root / "current.json")
    if current is None:
        return False
    rows = current.get("rows")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reason = " ".join(
            str(row.get(key) or "")
            for key in ("parser_reason", "reason_code", "parse_error")
        ).casefold()
        if any(marker in reason for marker in _COMPILER_UNAVAILABLE_MARKERS):
            return True
    return False


def invalidate_stale_compiler_unavailable_state(
    index_root: Path,
    *,
    current_parser_identity: str,
) -> dict[str, Any]:
    """Drop previous current/cache when toolchain-failure evidence would stick.

    RepositoryIndexer reuses rows only when ``parser_identity`` matches.  When
    the TypeScript compiler becomes available the identity changes and reuse is
    already refused.  This helper additionally removes a previous current that
    still carries compiler-unavailable rows so the handoff never republishes
    stale toolchain evidence under a fresh identity. Content-addressed cache
    rows remain available because parser identity is already part of their
    reuse contract.
    """

    root = Path(index_root)
    current_path = root / "current.json"
    previous = _read_json_object(current_path)
    had_unavailable = previous_index_has_compiler_unavailable(root)
    previous_identity = ""
    if previous is not None:
        rows = previous.get("rows") or ()
        for row in rows:
            if isinstance(row, Mapping) and row.get("parser_identity"):
                previous_identity = str(row["parser_identity"])
                break
    identity_changed = bool(
        previous_identity
        and current_parser_identity
        and previous_identity != current_parser_identity
    )
    removed: list[str] = []
    if had_unavailable and (identity_changed or not previous_identity):
        if current_path.exists():
            current_path.unlink()
            removed.append(str(current_path))
    return {
        "had_compiler_unavailable": had_unavailable,
        "previous_parser_identity": previous_identity,
        "current_parser_identity": current_parser_identity,
        "identity_changed": identity_changed,
        "removed": removed,
    }


def resolve_handoff_root(
    *,
    output_root: Path,
    handoff_root: str | os.PathLike[str] | None,
    publish_handoff: bool,
) -> Path | None:
    """Locate the SCA data root that owns baseline/ and analyzer_health/."""

    if handoff_root is not None:
        return Path(handoff_root).expanduser().resolve()
    if not publish_handoff:
        return None
    if output_root.name == "baseline":
        return output_root.parent.resolve()
    return output_root.resolve()


def publish_authoritative_handoff(
    result: RepositoryIndex,
    *,
    handoff_root: Path,
    provider: PolyglotASTProvider,
    typescript_path: str | None,
    typescript_version: str,
    llm_call_count: int = 0,
    provider_call_count: int = 0,
    model_call_count: int = 0,
    invalidation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish index + current + analyzer-health bound to one root.

    Every eligible path must already be success or a typed bounded failure in
    ``result``.  This writer never invents parse success and never increments
    LLM/provider/model counters.
    """

    if llm_call_count or provider_call_count or model_call_count:
        raise ValueError(
            "authoritative handoff forbids non-zero LLM/provider/model calls"
        )

    root = Path(handoff_root)
    baseline_dir = root / "baseline"
    health_dir = root / "analyzer_health"
    baseline_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    health_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Publish health before the mutable current pointer. If publication is
    # interrupted, readers continue to observe the prior complete handoff.
    health_report = assess_polyglot_ast_health(
        [row.to_dict() for row in result.rows],
        provider=provider,
        repair_authority=False,
        run_canaries=True,
        search_roots=[
            Path(result.snapshot.repository_root),
            Path(result.snapshot.repository_root).parent,
        ],
    )
    health_path = health_dir / "report.json"
    health_identity = write_polyglot_ast_health_report(health_report, health_path)

    index_payload = result.to_dict()
    index_bytes = (
        json.dumps(
            index_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    repository_index_path = baseline_dir / "repository-index.json"
    current_path = baseline_dir / "current.json"
    _atomic_bytes(repository_index_path, index_bytes)

    # Prefer the indexer's exact AnalyzerHealthReport for completion gates while
    # still exposing the polyglot receipt on disk for SCA-166 consumers.
    analyzer_health = result.health.to_dict()
    polyglot_health = health_report.to_dict(include_identity=True)

    eligible_statuses = {"indexed", "cache_hit", "parse_failure"}
    eligible = [
        row
        for row in result.rows
        if row.parser_status.value in eligible_statuses
    ]
    success_count = sum(
        1
        for row in eligible
        if row.parser_status.value in {"indexed", "cache_hit"}
        and not (row.parser_reason or "").strip()
    )
    failure_count = sum(
        1 for row in eligible if row.parser_status.value == "parse_failure"
    )
    untyped = [
        row.path
        for row in eligible
        if row.parser_status.value == "parse_failure"
        and not str(row.parser_reason or row.reason_code or "").strip()
    ]
    compiler_unavailable_remaining = [
        row.path
        for row in eligible
        if "compiler_unavailable"
        in f"{row.parser_reason} {row.reason_code}".casefold()
    ]

    handoff = {
        "schema": HANDOFF_SCHEMA,
        "evidence_id": HANDOFF_EVIDENCE,
        "index_id": result.index_id,
        "snapshot_id": result.snapshot.snapshot_id,
        "ast_index_id": result.ast_index_id,
        "coverage_root": result.snapshot.snapshot_id,
        "index_root": result.index_id,
        "health_root": (
            health_identity.get("cid") or health_identity.get("digest", "")
        ),
        "polyglot_health_digest": health_identity.get("digest", ""),
        "polyglot_health_cid": health_identity.get("cid", ""),
        "parser_identity": getattr(
            result, "parser_identity", ""
        )
        or next(
            (
                row.parser_identity
                for row in result.rows
                if row.parser_identity
            ),
            "",
        ),
        "toolchain": {
            "typescript_path": typescript_path or "",
            "typescript_version": typescript_version or "",
            "expected_typescript_version": (
                getattr(provider, "expected_typescript_version", "") or ""
            ),
            "node_executable": getattr(provider, "node_executable", "node"),
        },
        "roots_agree": True,
        "safe_for_completion_reasoning": bool(
            result.safe_for_completion_reasoning
            and health_report.safe_for_completion_reasoning
        ),
        "analyzer_health": analyzer_health,
        "polyglot_health_status": polyglot_health.get("status"),
        "polyglot_health_reasons": list(polyglot_health.get("reasons") or ()),
        "eligible_path_count": len(eligible),
        "success_path_count": success_count,
        "typed_failure_path_count": failure_count,
        "untyped_failure_paths": untyped,
        "compiler_unavailable_remaining": compiler_unavailable_remaining,
        "llm_call_count": 0,
        "provider_call_count": 0,
        "model_call_count": 0,
        "invalidation": dict(invalidation_receipt or {}),
        "artifacts": {
            "repository_index": str(repository_index_path),
            "current": str(current_path),
            "analyzer_health_report": str(health_path),
        },
        "head_commit_id": result.snapshot.head_commit_id,
        "scope_id": result.snapshot.scope_id,
        "scope_policy_id": result.snapshot.scope_policy_id,
    }
    if untyped:
        handoff["roots_agree"] = False
        handoff["safe_for_completion_reasoning"] = False
    if compiler_unavailable_remaining:
        handoff["safe_for_completion_reasoning"] = False

    # The mutable current pointer is the commit point for this publication.
    _atomic_bytes(current_path, index_bytes)

    # Cross-check that the three durable roots bind the same snapshot/index.
    reloaded_index = _read_json_object(repository_index_path)
    reloaded_current = _read_json_object(current_path)
    reloaded_health = _read_json_object(health_path)
    if (
        reloaded_index is None
        or reloaded_current is None
        or reloaded_health is None
        or reloaded_index.get("index_id") != result.index_id
        or reloaded_current.get("index_id") != result.index_id
        or reloaded_index.get("snapshot", {}).get("snapshot_id")
        != result.snapshot.snapshot_id
        or reloaded_current.get("snapshot", {}).get("snapshot_id")
        != result.snapshot.snapshot_id
    ):
        raise RepositoryIndexerError(
            "handoff artifacts failed snapshot/index root agreement"
        )
    handoff["published"] = True
    _atomic_json(baseline_dir / "handoff.json", handoff)
    return handoff


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
        typescript_version = str(args.typescript_version or "").strip()
        if not typescript_version and typescript_path:
            typescript_version = DEFAULT_TYPESCRIPT_VERSION
        provider = PolyglotASTProvider(
            limits,
            node_executable=args.node,
            typescript_path=typescript_path,
            expected_typescript_version=typescript_version,
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
        invalidation_receipt: dict[str, Any] = {
            "skipped": True,
            "reason": "invalidate_compiler_unavailable_disabled",
        }
        if args.invalidate_compiler_unavailable and typescript_path:
            invalidation_receipt = invalidate_stale_compiler_unavailable_state(
                output_root,
                current_parser_identity=indexer.parser_identity,
            )
            invalidation_receipt["skipped"] = False

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

        handoff_root = resolve_handoff_root(
            output_root=output_root,
            handoff_root=args.handoff_root,
            publish_handoff=args.publish_handoff,
        )
        handoff: dict[str, Any] | None = None
        if handoff_root is not None:
            handoff = publish_authoritative_handoff(
                result,
                handoff_root=handoff_root,
                provider=provider,
                typescript_path=typescript_path,
                typescript_version=typescript_version,
                llm_call_count=int(getattr(baseline, "llm_call_count", 0) or 0),
                invalidation_receipt=invalidation_receipt,
            )

        summary = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "sca-repository-index-run@1"
            ),
            "index_id": result.index_id,
            "snapshot_id": result.snapshot.snapshot_id,
            "ast_index_id": result.ast_index_id,
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
            "provider_call_count": 0,
            "model_call_count": 0,
            "contract_count": baseline.findings.get("contract_population", {}).get(
                "emitted_contract_count", 0
            ),
            "findings_root": baseline.findings.get("findings_root", ""),
            "graph_root": baseline.findings.get("graph_root", ""),
            "typescript_path": typescript_path or "",
            "typescript_version": typescript_version or "",
            "parser_identity": indexer.parser_identity,
            "compiler_unavailable_invalidation": invalidation_receipt,
            "handoff": (
                {
                    "published": True,
                    "evidence_id": handoff.get("evidence_id"),
                    "artifacts": handoff.get("artifacts"),
                    "roots_agree": handoff.get("roots_agree"),
                    "safe_for_completion_reasoning": handoff.get(
                        "safe_for_completion_reasoning"
                    ),
                    "compiler_unavailable_remaining": handoff.get(
                        "compiler_unavailable_remaining"
                    ),
                    "llm_call_count": handoff.get("llm_call_count", 0),
                }
                if handoff is not None
                else {"published": False}
            ),
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
