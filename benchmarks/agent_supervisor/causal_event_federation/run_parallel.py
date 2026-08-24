#!/usr/bin/env python3
"""Fail-closed CASF twelve-supervisor parallel benchmark admission.

CASF-039 freezes the comparison contract for a real twelve-supervisor run.  It
does not manufacture a substitute result: the canonical matrix requires
independent qualified processes, and the required live-capacity attestation is
not available in this tree.  Consequently this runner validates the immutable
recipe and then stops *before* launching a child, opening a store, or accepting
caller-authored telemetry.  A future implementation may only make this runner
measurable by changing the frozen capability contract together with its
independent admission and evidence producers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/parallel-benchmark-manifest@1"
)
RESULT_SCHEMA = "casf/parallel-benchmark@1"
BENCHMARK_ID = "casf-parallel-twelve-supervisor-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
MEASUREMENT_SCOPE = "twelve_independent_supervisor_processes_qualified_live_profile"

REQUIRED_IDENTITIES = (
    "repository_commit",
    "repository_tree",
    "control_plane_generation",
    "schema_fingerprint",
    "policy_ref",
    "policy_revision",
    "capability_ref",
    "federation_id",
    "supervisor_id",
    "task_id",
    "attempt_id",
    "worktree_id",
    "assignment_revision",
    "fencing_epoch",
)
SOURCE_RELATIVE_PATHS = (
    "benchmarks/agent_supervisor/causal_event_federation/parallel_manifest.json",
    "benchmarks/agent_supervisor/causal_event_federation/run_parallel.py",
)
ZERO_TOLERANCE_GATES = (
    "lost_deliveries",
    "duplicate_committed_effects",
    "stale_fence_completion",
    "unauthorized_creation",
    "tenant_leakage",
    "agent_sql",
    "secret_leaks",
    "reduced_assurance",
)
LIVE_CAPABILITY_UNAVAILABLE = {
    "availability": "unavailable",
    "execution_status": "not_run",
    "reason_code": "qualified_twelve_supervisor_live_capacity_not_admitted",
    "required_attestation": "casf/parallel-live-capacity-attestation@1",
    "required_evidence": "current_generation_accepted_gate_and_live_telemetry",
    "metrics_omitted": True,
}


class ParallelBenchmarkError(ValueError):
    """Raised when the frozen parallel benchmark contract is invalid."""


class ParallelCapabilityUnavailable(ParallelBenchmarkError):
    """Raised instead of producing a simulated twelve-supervisor result."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParallelBenchmarkError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ParallelBenchmarkError(f"invalid JSON in {name}: {exc.msg}") from exc


def _read_object(path: Path | str) -> dict[str, Any]:
    resolved = Path(path)
    try:
        decoded = _loads_json(resolved.read_text(encoding="utf-8"), name=str(resolved))
    except OSError as exc:
        raise ParallelBenchmarkError(f"cannot read {resolved}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ParallelBenchmarkError(f"{resolved} must contain a JSON object")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ParallelBenchmarkError(f"cannot hash {path}: {exc}") from exc


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParallelBenchmarkError(f"{name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ParallelBenchmarkError(
            f"{name} has a closed schema "
            f"(unknown={sorted(actual - expected)}, missing={sorted(expected - actual)})"
        )


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 16_384:
        raise ParallelBenchmarkError(f"{name} must be a bounded non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ParallelBenchmarkError(f"{name} must be a lowercase SHA-256 digest")
    return text


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return _object_sha256(manifest)


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload.pop("content_sha256", None)
    return _object_sha256(payload)


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path) if path is not None else Path(__file__).with_name("parallel_manifest.json")
    manifest = _read_object(resolved)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate every field of the immutable, intentionally blocked recipe."""

    manifest = _require_mapping(manifest, "manifest")
    _require_exact_keys(
        manifest,
        {
            "schema", "benchmark_id", "program_id", "objective_id", "frozen", "state",
            "authoritative", "promotion_eligible", "matrix_binding", "execution",
            "parallel_comparison", "zero_tolerance_gates", "identity_requirements",
            "source_modules", "live_capability", "result_storage", "nonclaims",
        },
        "manifest",
    )
    fixed = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": "CASF-039",
        "frozen": True,
        "state": "capability_blocked_specification_only",
        "authoritative": False,
        "promotion_eligible": False,
    }
    if any(manifest[key] != value for key, value in fixed.items()):
        raise ParallelBenchmarkError("manifest identity or non-authoritative state has changed")

    matrix = _require_mapping(manifest["matrix_binding"], "manifest matrix binding")
    expected_matrix = {
        "relative_path": MATRIX_RELATIVE_PATH, "schema": MATRIX_SCHEMA, "sha256": MATRIX_SHA256
    }
    _require_exact_keys(matrix, set(expected_matrix), "manifest matrix binding")
    if dict(matrix) != expected_matrix:
        raise ParallelBenchmarkError("manifest does not bind the exact frozen benchmark matrix")

    execution = _require_mapping(manifest["execution"], "manifest execution")
    expected_execution = {
        "measurement_scope": MEASUREMENT_SCOPE,
        "required_supervisor_processes": 12,
        "subprocess_budget": 12,
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "launch_permitted": False,
    }
    _require_exact_keys(execution, set(expected_execution), "manifest execution")
    if dict(execution) != expected_execution:
        raise ParallelBenchmarkError("manifest execution may not launch or simulate this profile")

    comparison = _require_mapping(manifest["parallel_comparison"], "parallel comparison")
    expected_comparison = {
        "baseline_qualified_supervisors": 1,
        "comparison_identity": "same-host-tasks-providers-tests-proofs-budgets",
        "minimum_accepted_task_throughput_multiplier": 3.0,
        "lower_assurance_permitted": False,
    }
    _require_exact_keys(comparison, set(expected_comparison), "parallel comparison")
    if dict(comparison) != expected_comparison:
        raise ParallelBenchmarkError("parallel comparison assurance or throughput contract changed")
    if tuple(manifest["zero_tolerance_gates"]) != ZERO_TOLERANCE_GATES:
        raise ParallelBenchmarkError("zero-tolerance safety gates have changed")
    if tuple(manifest["identity_requirements"]) != REQUIRED_IDENTITIES:
        raise ParallelBenchmarkError("identity requirements have changed")
    if tuple(manifest["source_modules"]) != SOURCE_RELATIVE_PATHS:
        raise ParallelBenchmarkError("measured source-module set has changed")

    capability = _require_mapping(manifest["live_capability"], "live capability")
    _require_exact_keys(capability, set(LIVE_CAPABILITY_UNAVAILABLE), "live capability")
    if dict(capability) != LIVE_CAPABILITY_UNAVAILABLE:
        raise ParallelBenchmarkError("live parallel capability must remain unavailable/not-run")
    _require_text(manifest["result_storage"], "result storage")
    if not isinstance(manifest["nonclaims"], list) or len(manifest["nonclaims"]) != 8:
        raise ParallelBenchmarkError("manifest must preserve eight explicit nonclaims")
    for index, nonclaim in enumerate(manifest["nonclaims"]):
        _require_text(nonclaim, f"manifest nonclaims[{index}]")


def validate_matrix_binding(
    manifest: Mapping[str, Any], matrix_path: Path | str | None = None
) -> dict[str, str]:
    validate_manifest(manifest)
    resolved = Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml")
    if _file_sha256(resolved) != MATRIX_SHA256:
        raise ParallelBenchmarkError("frozen benchmark matrix content is stale or changed")
    try:
        first_line = resolved.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise ParallelBenchmarkError("frozen benchmark matrix is missing its schema") from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise ParallelBenchmarkError("frozen benchmark matrix schema has changed")
    return dict(manifest["matrix_binding"])


def _bound_repository_and_recipe(
    repository: Path | str,
    manifest_path: Path | str | None,
    matrix_path: Path | str | None,
) -> tuple[Path, Path, Path]:
    root = Path(repository).resolve()
    runner_root = Path(__file__).resolve().parents[3]
    if root != runner_root:
        raise ParallelBenchmarkError(
            "repository must be the exact repository containing this benchmark runner"
        )
    expected_manifest = root / SOURCE_RELATIVE_PATHS[0]
    expected_matrix = root / MATRIX_RELATIVE_PATH
    resolved_manifest = expected_manifest if manifest_path is None else Path(manifest_path).resolve()
    resolved_matrix = expected_matrix if matrix_path is None else Path(matrix_path).resolve()
    if resolved_manifest != expected_manifest.resolve() or resolved_matrix != expected_matrix.resolve():
        raise ParallelBenchmarkError("manifest and matrix must come from the measured repository tree")
    return root, resolved_manifest, resolved_matrix


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParallelBenchmarkError(f"cannot establish repository identity: {exc}") from exc
    return result.stdout.strip()


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = Path(repository).resolve()
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise ParallelBenchmarkError("repository is dirty; exact current-tree evidence is unavailable")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    for value, name in ((commit, "repository commit"), (tree, "repository tree")):
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ParallelBenchmarkError(f"{name} must be a full lowercase Git object id")
    return {"repository_commit": commit, "repository_tree": tree}


def _validate_identities(value: Any) -> dict[str, Any]:
    identities = _require_mapping(value, "benchmark identities")
    _require_exact_keys(identities, set(REQUIRED_IDENTITIES), "benchmark identities")
    checked = dict(identities)
    for key in ("repository_commit", "repository_tree"):
        text = _require_text(checked[key], f"identities.{key}")
        if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
            raise ParallelBenchmarkError(f"identities.{key} must be a full lowercase Git object id")
    for key in ("control_plane_generation", "assignment_revision", "fencing_epoch"):
        if isinstance(checked[key], bool) or not isinstance(checked[key], int) or checked[key] < 1:
            raise ParallelBenchmarkError(f"identities.{key} must be a positive integer")
    for key in REQUIRED_IDENTITIES:
        if key not in {"repository_commit", "repository_tree", "control_plane_generation", "assignment_revision", "fencing_epoch"}:
            _require_text(checked[key], f"identities.{key}")
    if checked["task_id"] != "CASF-039":
        raise ParallelBenchmarkError("identities.task_id must bind CASF-039")
    return checked


def live_capability(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the frozen capability descriptor; it is not a result or receipt."""

    checked = load_manifest() if manifest is None else dict(manifest)
    validate_manifest(checked)
    return dict(checked["live_capability"])


def require_live_capability(manifest: Mapping[str, Any]) -> NoReturn:
    """Stop before any benchmark side effect while capacity is not admitted."""

    validate_manifest(manifest)
    capability = dict(manifest["live_capability"])
    raise ParallelCapabilityUnavailable(
        "qualified twelve-supervisor live capability is unavailable: "
        f"{capability['reason_code']}; required={capability['required_attestation']}; "
        f"evidence={capability['required_evidence']}"
    )


def run_benchmark(
    *,
    repository: Path | str,
    identities: Mapping[str, Any],
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> NoReturn:
    """Admit the frozen recipe, then fail closed without launching a process.

    The arguments deliberately remain part of the public contract so a future
    enabled implementation cannot accept an unbound tree or identity.  They
    are not inspected while the manifest's capability gate is unavailable:
    no caller-provided identity can bypass a missing live admission.
    """

    del identities
    _repository_root, resolved_manifest, resolved_matrix = _bound_repository_and_recipe(
        repository, manifest_path, matrix_path
    )
    manifest = load_manifest(resolved_manifest)
    validate_matrix_binding(manifest, resolved_matrix)
    require_live_capability(manifest)


def validate_result(
    result: Mapping[str, Any],
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
    current_identity: Mapping[str, Any] | None = None,
) -> NoReturn:
    """Reject every alleged result until live capacity is independently admitted."""

    del result, current_identity
    manifest = load_manifest(manifest_path)
    validate_matrix_binding(manifest, matrix_path)
    require_live_capability(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, help="clean repository identity to bind")
    parser.add_argument("--identities", type=Path, help="closed benchmark context identity JSON")
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("parallel_manifest.json")
    )
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matrix.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.repository is None or args.identities is None:
            raise ParallelBenchmarkError("--repository and --identities are required")
        _bound_repository_and_recipe(args.repository, args.manifest, args.matrix)
        _validate_identities(_read_object(args.identities))
        run_benchmark(
            repository=args.repository,
            identities=_read_object(args.identities),
            manifest_path=args.manifest,
            matrix_path=args.matrix,
        )
    except ParallelBenchmarkError as exc:
        print(
            json.dumps(
                {
                    "schema": RESULT_SCHEMA,
                    "execution_status": "not_run",
                    "availability": "unavailable",
                    "authoritative": False,
                    "promotion_eligible": False,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    raise AssertionError("an unavailable capability must not produce a benchmark result")


if __name__ == "__main__":
    raise SystemExit(main())
