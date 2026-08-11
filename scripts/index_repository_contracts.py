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
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    build_multi_root_repository_index,
    write_provider_index_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (  # noqa: E402
    DEFAULT_PROVIDER_PACKAGE_SPECS,
    RepositorySnapshotError,
    build_repository_snapshot,
    default_scope_policy_path,
    load_scope_policy,
)
from ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_dependencies import (  # noqa: E402
    PINNED_TYPESCRIPT_VERSION,
)

# Reviewed SwissKnife toolchain identity for the authoritative handoff.
DEFAULT_TYPESCRIPT_VERSION = PINNED_TYPESCRIPT_VERSION
HANDOFF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/sca-repository-index-handoff@1"
)
HANDOFF_EVIDENCE = "SCAEV022INDEX"
# SCA-G071 / SCAEV071PROOFCACHE: sole authoritative proof-receipt cache root
# published beside the baseline artifacts.
PROOF_PIPELINE_EVIDENCE = "SCAEV071PROOFCACHE"
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
    provider_indexes = parser.add_mutually_exclusive_group()
    provider_indexes.add_argument(
        "--include-provider-indexes",
        dest="include_provider_indexes",
        action="store_true",
        default=True,
        help=(
            "also scan configured provider package roots "
            "(ipfs_accelerate_py, ipfs_kit_py, ipfs_datasets_py) and publish "
            "provider-index.json beside the primary baseline (default: on)"
        ),
    )
    provider_indexes.add_argument(
        "--skip-provider-indexes",
        dest="include_provider_indexes",
        action="store_false",
        help="skip multi-root provider package indexing",
    )
    parser.add_argument(
        "--require-provider-authority",
        action="store_true",
        help=(
            "return status 4 when multi-root provider indexes are missing, "
            "dirty, partial, opaque, version-divergent, or otherwise fail "
            "exhaustive parity (zero model calls)"
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
    parser.add_argument(
        "--proof-cache-dir",
        default=None,
        help=(
            "durable TrustAwareProofCache directory for SCAEV071PROOFCACHE "
            "end-to-end proof/cache orchestration; defaults to "
            "<output-root>/proof-cache"
        ),
    )
    parser.add_argument(
        "--skip-proof-pipeline",
        action="store_true",
        help=(
            "skip McpContractProver / TrustAwareProofCache orchestration and "
            "retain parity-only baseline terminals"
        ),
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

    if not publish_handoff:
        return None
    if handoff_root is not None:
        return Path(handoff_root).expanduser().resolve()
    if output_root.name == "baseline":
        return output_root.parent.resolve()
    return output_root.resolve()


def validate_authoritative_publication_options(
    args: argparse.Namespace,
) -> None:
    """Reject analysis-only or weakened modes before authoritative publication."""

    if not bool(args.publish_handoff):
        return

    problems: list[str] = []
    if not bool(args.require_healthy):
        problems.append("--require-healthy is mandatory")
    if bool(args.shadow):
        problems.append("--shadow is analysis-only")
    if bool(args.skip_extraction):
        problems.append("--skip-extraction cannot publish a complete handoff")
    if int(args.max_parser_failures) > 10:
        problems.append("--max-parser-failures cannot exceed 10")
    if float(args.max_parser_failure_ratio) > 0.01:
        problems.append("--max-parser-failure-ratio cannot exceed 0.01")
    if not bool(args.invalidate_compiler_unavailable):
        problems.append("--keep-compiler-unavailable is forbidden")
    if not bool(getattr(args, "include_provider_indexes", True)):
        problems.append(
            "--skip-provider-indexes cannot publish a complete multi-root handoff"
        )
    if not bool(getattr(args, "require_provider_authority", False)):
        problems.append("--require-provider-authority is mandatory")
    if problems:
        raise ValueError(
            "authoritative handoff mode rejected: " + "; ".join(problems)
        )


def _provider_authority_summary(multi_root) -> dict[str, Any]:
    """Compact zero-model multi-root authority ledger for the run summary."""

    providers: list[dict[str, Any]] = []
    for item in multi_root.providers:
        observation = item.observation
        providers.append(
            {
                "package": item.package,
                "scope_path": observation.scope_path,
                "indexed": bool(item.indexed),
                "healthy": bool(item.healthy),
                "opaque_gitlink": bool(item.opaque_gitlink),
                "symbol_count": len(item.symbols),
                "symbol_extraction_complete": bool(
                    item.symbol_extraction_complete
                ),
                "status": getattr(
                    observation.status, "value", str(observation.status)
                ),
                "dirty": bool(observation.dirty),
                "version_divergent": bool(observation.version_divergent),
                "head_commit_id": observation.head_commit_id,
                "head_tree_id": observation.head_tree_id,
                "origin_url": observation.origin_url,
            }
        )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "sca-multi-root-provider-authority@1"
        ),
        "evidence_id": "SCAEV043MULTIROOT",
        "multi_root_id": multi_root.multi_root_id,
        "provider_count": len(multi_root.providers),
        "expected_packages": [
            spec.package for spec in DEFAULT_PROVIDER_PACKAGE_SPECS
        ],
        "all_providers_indexed": multi_root.all_providers_indexed,
        "all_providers_healthy": multi_root.all_providers_healthy,
        "all_symbol_extractions_complete": (
            multi_root.all_symbol_extractions_complete
        ),
        "any_opaque_gitlink": multi_root.any_opaque_gitlink,
        "exhaustive_parity_allowed": multi_root.exhaustive_parity_allowed,
        "has_blocking_contradictions": (
            multi_root.multi_root_snapshot.has_blocking_contradictions
            or bool(multi_root.contradictions)
        ),
        "contradiction_count": len(multi_root.contradictions),
        "providers": providers,
        "llm_call_count": 0,
        "provider_call_count": 0,
        "model_call_count": 0,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_generation_link(path: Path, target: str) -> None:
    """Install a stable canonical symlink without replacing regular files."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.path.lexists(path):
        if not path.is_symlink() or os.readlink(path) != target:
            raise RepositoryIndexerError(
                f"authoritative path is not the expected generation link: {path}"
            )
        return
    temporary = path.parent / (
        f".{path.name}.link-{os.getpid()}-{hashlib.sha256(target.encode()).hexdigest()[:8]}"
    )
    try:
        os.symlink(target, temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_immutable_generation(
    *,
    root: Path,
    generation_name: str,
    repository_index_bytes: bytes,
    health_bytes: bytes,
    handoff_bytes: bytes,
) -> None:
    """Publish one immutable generation, then atomically swap one pointer."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{15,127}", generation_name):
        raise RepositoryIndexerError("invalid authoritative generation name")

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = root / ".authoritative-handoff.lock"
    expected = {
        Path("baseline/repository-index.json"): repository_index_bytes,
        Path("baseline/current.json"): repository_index_bytes,
        Path("baseline/handoff.json"): handoff_bytes,
        Path("analyzer_health/report.json"): health_bytes,
    }

    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        staging = Path(
            tempfile.mkdtemp(prefix=".generation-", dir=generations)
        )
        try:
            for relative, payload in expected.items():
                _atomic_bytes(staging / relative, payload)

            final_generation = generations / generation_name
            if final_generation.exists():
                for relative, payload in expected.items():
                    try:
                        existing = (final_generation / relative).read_bytes()
                    except OSError as exc:
                        raise RepositoryIndexerError(
                            "existing authoritative generation is incomplete"
                        ) from exc
                    if existing != payload:
                        raise RepositoryIndexerError(
                            "authoritative generation identity collision"
                        )
                shutil.rmtree(staging)
            else:
                os.replace(staging, final_generation)
                _fsync_directory(generations)

            # These links never change; only `authoritative` is swapped.
            _ensure_generation_link(
                root / "baseline" / "repository-index.json",
                "../authoritative/baseline/repository-index.json",
            )
            _ensure_generation_link(
                root / "baseline" / "current.json",
                "../authoritative/baseline/current.json",
            )
            _ensure_generation_link(
                root / "baseline" / "handoff.json",
                "../authoritative/baseline/handoff.json",
            )
            _ensure_generation_link(
                root / "analyzer_health" / "report.json",
                "../authoritative/analyzer_health/report.json",
            )

            pointer = root / "authoritative"
            if os.path.lexists(pointer) and not pointer.is_symlink():
                raise RepositoryIndexerError(
                    "authoritative generation pointer is not a symlink"
                )
            pointer_target = f"generations/{generation_name}"
            temporary_pointer = root / (
                f".authoritative-{os.getpid()}-{generation_name[:12]}"
            )
            try:
                os.symlink(pointer_target, temporary_pointer)
                os.replace(temporary_pointer, pointer)
                _fsync_directory(root)
            finally:
                try:
                    temporary_pointer.unlink()
                except FileNotFoundError:
                    pass
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _validated_publication_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    result: RepositoryIndex,
) -> dict[str, Any]:
    """Validate complete deterministic extraction and snapshot freshness."""

    if not isinstance(evidence, Mapping):
        raise RepositoryIndexerError(
            "authoritative handoff requires a publication evidence receipt"
        )
    receipt = dict(evidence)
    snapshot_id = str(receipt.get("snapshot_id") or "")
    fresh_snapshot_id = str(receipt.get("fresh_snapshot_id") or "")
    if snapshot_id != result.snapshot.snapshot_id:
        raise RepositoryIndexerError(
            "publication evidence snapshot does not match repository index"
        )
    if fresh_snapshot_id != result.snapshot.snapshot_id:
        raise RepositoryIndexerError(
            "repository changed after the indexed snapshot was built"
        )
    if not str(receipt.get("baseline_result_id") or ""):
        raise RepositoryIndexerError(
            "publication evidence is missing the baseline result identity"
        )
    if not str(receipt.get("coverage_root") or ""):
        raise RepositoryIndexerError(
            "publication evidence is missing the extracted coverage root"
        )

    stages = receipt.get("stages")
    if not isinstance(stages, list):
        raise RepositoryIndexerError(
            "publication evidence is missing baseline stage receipts"
        )
    stage_by_name = {
        str(item.get("name") or ""): item
        for item in stages
        if isinstance(item, Mapping)
    }
    for required in ("repository_index", "extraction", "catalog", "publish"):
        stage = stage_by_name.get(required)
        if (
            stage is None
            or str(stage.get("completeness") or "") != "complete"
            or any(
                str(code).startswith("withheld")
                or str(code).endswith("_unhealthy")
                for code in stage.get("reason_codes") or ()
            )
        ):
            raise RepositoryIndexerError(
                f"authoritative handoff requires complete {required} stage"
            )

    execution = receipt.get("execution")
    if not isinstance(execution, Mapping):
        raise RepositoryIndexerError(
            "publication evidence is missing deterministic execution receipt"
        )
    if any(
        int(execution.get(key, -1)) != 0
        for key in ("llm_call_count", "provider_call_count", "model_call_count")
    ):
        raise RepositoryIndexerError(
            "authoritative handoff requires zero model/provider/LLM calls"
        )
    if str(execution.get("mode") or "") != "deterministic-symbolic":
        raise RepositoryIndexerError(
            "authoritative handoff requires deterministic-symbolic execution"
        )
    return receipt


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
    publication_evidence: Mapping[str, Any] | None = None,
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
    evidence = _validated_publication_evidence(
        publication_evidence,
        result=result,
    )

    if result.health.status is not AnalyzerHealthStatus.HEALTHY:
        raise RepositoryIndexerError(
            "authoritative handoff requires healthy analyzer status"
        )
    if not result.safe_for_completion_reasoning:
        raise RepositoryIndexerError(
            "authoritative handoff requires safe_for_completion_reasoning"
        )

    rows = [row.to_dict() for row in result.rows]
    health_report = assess_polyglot_ast_health(
        rows,
        provider=provider,
        repair_authority=False,
        run_canaries=True,
        search_roots=[
            Path(result.snapshot.repository_root),
            Path(result.snapshot.repository_root).parent,
        ],
    )
    if not health_report.safe_for_completion_reasoning:
        raise RepositoryIndexerError(
            "authoritative handoff requires healthy polyglot AST canaries"
        )

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

    root = Path(handoff_root)
    baseline_dir = root / "baseline"
    health_dir = root / "analyzer_health"
    repository_index_path = baseline_dir / "repository-index.json"
    current_path = baseline_dir / "current.json"
    health_path = health_dir / "report.json"

    handoff = {
        "schema": HANDOFF_SCHEMA,
        "evidence_id": HANDOFF_EVIDENCE,
        "index_id": result.index_id,
        "snapshot_id": result.snapshot.snapshot_id,
        "ast_index_id": result.ast_index_id,
        "coverage_root": str(evidence["coverage_root"]),
        "index_root": result.index_id,
        "health_root": "",
        "polyglot_health_digest": "",
        "polyglot_health_cid": "",
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
        "publication_evidence": evidence,
        "artifacts": {
            "repository_index": "baseline/repository-index.json",
            "current": "baseline/current.json",
            "analyzer_health_report": "analyzer_health/report.json",
        },
        "head_commit_id": result.snapshot.head_commit_id,
        "scope_id": result.snapshot.scope_id,
        "scope_policy_id": result.snapshot.scope_policy_id,
    }
    if untyped:
        raise RepositoryIndexerError(
            "authoritative handoff contains untyped parser failures"
        )
    if compiler_unavailable_remaining:
        raise RepositoryIndexerError(
            "authoritative handoff retains compiler-unavailable rows"
        )

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

    # Build and validate the complete handoff away from authoritative paths.
    with tempfile.TemporaryDirectory(prefix="sca-handoff-") as staging_name:
        staged_health_path = Path(staging_name) / "report.json"
        health_identity = write_polyglot_ast_health_report(
            health_report,
            staged_health_path,
        )
        health_bytes = staged_health_path.read_bytes()
        staged_health = json.loads(health_bytes)
        if not isinstance(staged_health, Mapping):
            raise RepositoryIndexerError(
                "staged analyzer health report is not a JSON object"
            )

        handoff["health_root"] = (
            health_identity.get("cid") or health_identity.get("digest", "")
        )
        handoff["polyglot_health_digest"] = health_identity.get("digest", "")
        handoff["polyglot_health_cid"] = health_identity.get("cid", "")
        generation_name = "sha256-" + hashlib.sha256(
            index_bytes + b"\0" + health_bytes
        ).hexdigest()
        handoff["generation"] = generation_name
        handoff["published"] = True
        handoff_bytes = (
            json.dumps(
                handoff,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    # All files live beneath an immutable generation. One symlink swap is the
    # only publication commit point, so crashes expose either old or new state.
    _publish_immutable_generation(
        root=root,
        generation_name=generation_name,
        repository_index_bytes=index_bytes,
        health_bytes=health_bytes,
        handoff_bytes=handoff_bytes,
    )

    # Cross-check that the durable roots still bind the exact staged payloads.
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
    return handoff


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    requested_output_root = Path(args.output_root)
    output_root = requested_output_root
    indexer: RepositoryIndexer | None = None
    publication_staging: tempfile.TemporaryDirectory[str] | None = None
    try:
        validate_authoritative_publication_options(args)
        if args.publish_handoff:
            publication_staging = tempfile.TemporaryDirectory(
                prefix="sca-authoritative-run-"
            )
            output_root = Path(publication_staging.name) / "baseline"
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

        proof_cache_dir = (
            Path(args.proof_cache_dir).expanduser().resolve()
            if args.proof_cache_dir
            else (output_root / "proof-cache")
        )
        run_proof_pipeline = not bool(args.skip_proof_pipeline)

        if args.skip_extraction:
            from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
                materialize_contract_assurance_baseline,
            )

            baseline = materialize_contract_assurance_baseline(
                repository_index=result,
                extract_expected=False,
                output_root=output_root,
                max_file_bytes=args.max_artifact_bytes,
                proof_cache_dir=proof_cache_dir,
                run_proof_pipeline=run_proof_pipeline,
            )
        else:
            baseline = materialize_baseline_from_repository_index(
                result,
                output_root=output_root,
                repo_root=args.repo_root,
                swissknife_root=swissknife_root,
                max_file_bytes=args.max_artifact_bytes,
                proof_cache_dir=proof_cache_dir,
                run_proof_pipeline=run_proof_pipeline,
            )

        multi_root_summary: dict[str, Any] | None = None
        multi_root_authority_failed = False
        if bool(args.include_provider_indexes):
            multi_root_index_root = output_root / "provider-indexes"
            multi_root = build_multi_root_repository_index(
                args.repo_root,
                index_root=multi_root_index_root,
                scope_config_path=args.scope_config,
                provider_packages=DEFAULT_PROVIDER_PACKAGE_SPECS,
                provider=provider,
                health_thresholds=thresholds,
                include_primary_snapshot=False,
                allow_dirty_analysis=args.allow_dirty,
                max_paths=args.max_paths,
                extract_symbols=True,
            )
            provider_index_path = write_provider_index_baseline(
                multi_root,
                output_root / "provider-index.json",
            )
            multi_root_summary = _provider_authority_summary(multi_root)
            multi_root_summary["provider_index_path"] = str(provider_index_path)
            multi_root_summary["provider_index_root"] = str(multi_root_index_root)
            multi_root_authority_failed = not bool(
                multi_root.exhaustive_parity_allowed
            )

        handoff_root = resolve_handoff_root(
            output_root=requested_output_root,
            handoff_root=args.handoff_root,
            publish_handoff=args.publish_handoff,
        )
        handoff: dict[str, Any] | None = None
        if handoff_root is not None:
            fresh_snapshot = build_repository_snapshot(
                args.repo_root,
                scope_config_path=args.scope_config,
                allow_dirty_analysis=args.allow_dirty,
                max_paths=args.max_paths,
                max_file_bytes=args.max_source_bytes,
                max_total_bytes=args.max_total_snapshot_bytes,
            )
            publication_evidence = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "sca-authoritative-publication-evidence@1"
                ),
                "snapshot_id": baseline.snapshot_id,
                "fresh_snapshot_id": fresh_snapshot.snapshot_id,
                "baseline_result_id": baseline.result_id,
                "coverage_root": str(
                    baseline.findings.get("coverage_id") or ""
                ),
                "stages": [stage.to_dict() for stage in baseline.stages],
                "execution": {
                    "mode": "deterministic-symbolic",
                    "llm_call_count": baseline.llm_call_count,
                    "provider_call_count": 0,
                    "model_call_count": 0,
                    "model_invocation_enabled": False,
                },
            }
            handoff = publish_authoritative_handoff(
                result,
                handoff_root=handoff_root,
                provider=provider,
                typescript_path=typescript_path,
                typescript_version=typescript_version,
                llm_call_count=int(getattr(baseline, "llm_call_count", 0) or 0),
                invalidation_receipt=invalidation_receipt,
                publication_evidence=publication_evidence,
            )
            if multi_root_summary is not None:
                handoff_provider_index = (
                    Path(handoff_root) / "baseline" / "provider-index.json"
                )
                if (output_root / "provider-index.json").is_file():
                    handoff_provider_index.parent.mkdir(
                        parents=True, exist_ok=True, mode=0o700
                    )
                    shutil.copy2(
                        output_root / "provider-index.json",
                        handoff_provider_index,
                    )
                    multi_root_summary["handoff_provider_index_path"] = str(
                        handoff_provider_index
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
            "proof_pipeline": {
                "evidence_id": PROOF_PIPELINE_EVIDENCE,
                "enabled": run_proof_pipeline,
                "proof_cache_dir": str(proof_cache_dir),
                "attempted": baseline.findings.get("proof_outcomes", {}).get(
                    "attempted", 0
                ),
                "proved": baseline.findings.get("proof_outcomes", {}).get(
                    "proved", 0
                ),
                "refuted": baseline.findings.get("proof_outcomes", {}).get(
                    "refuted", 0
                ),
                "cache_hits": baseline.findings.get("proof_outcomes", {}).get(
                    "cache_hits", 0
                ),
                "outcome_count": len(baseline.proof_pipeline_outcomes),
            },
            "typescript_path": typescript_path or "",
            "typescript_version": typescript_version or "",
            "parser_identity": indexer.parser_identity,
            "compiler_unavailable_invalidation": invalidation_receipt,
            "multi_root_providers": multi_root_summary
            if multi_root_summary is not None
            else {"included": False},
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
        if multi_root_summary is not None:
            summary["multi_root_providers"]["included"] = True
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
        if bool(args.require_provider_authority) and multi_root_authority_failed:
            return 4
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
        if publication_staging is not None:
            publication_staging.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
