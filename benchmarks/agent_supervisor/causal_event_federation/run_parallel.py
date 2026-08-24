#!/usr/bin/env python3
"""Emit the canonical unavailable result for the CASF parallel benchmark.

The frozen matrix requires one qualified single-supervisor baseline arm and a
like-for-like twelve-supervisor candidate arm.  This tree has no admitted live
capacity for either arm.  The runner therefore performs a side-effect-free
capacity preflight first, observes a clean repository commit and tree twice,
and emits a content-addressed unavailable/not-run artifact.  It never turns
missing capacity into measurements, zeroes, or qualification.

An identity-file option remains parser-compatible with the future live runner,
but its contents are deliberately not opened while capacity is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, Protocol

MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/parallel-benchmark-manifest@1"
)
RESULT_SCHEMA = "casf/parallel-benchmark@1"
ERROR_SCHEMA = "casf/parallel-benchmark-error@1"
BENCHMARK_ID = "casf-parallel-twelve-supervisor-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
OBJECTIVE_ID = "CASF-039"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
MANIFEST_RELATIVE_PATH = (
    "benchmarks/agent_supervisor/causal_event_federation/parallel_manifest.json"
)
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
MEASUREMENT_SCOPE = "twelve_independent_supervisor_processes_qualified_live_profile"
ADMITTED_EXECUTION_INTERFACE = "CASFParallelAdmittedExecution@1"
UNAVAILABLE_REASON_CODE = "qualified_twelve_supervisor_live_capacity_not_admitted"
MAX_DOCUMENT_BYTES = 65_536
MAX_IDENTIFIER_BYTES = 256

SOURCE_RELATIVE_PATHS = (
    MANIFEST_RELATIVE_PATH,
    "benchmarks/agent_supervisor/causal_event_federation/run_parallel.py",
)
FUTURE_IDENTITY_REQUIREMENTS = (
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
FUTURE_REQUIRED_ARMS = (
    {
        "arm_id": "baseline",
        "required_supervisor_processes": 1,
        "availability": "unavailable",
        "execution_status": "not_run",
        "qualified": False,
    },
    {
        "arm_id": "candidate",
        "required_supervisor_processes": 12,
        "availability": "unavailable",
        "execution_status": "not_run",
        "qualified": False,
    },
)
LIVE_CAPABILITY_UNAVAILABLE = {
    "availability": "unavailable",
    "execution_status": "not_run",
    "reason_code": UNAVAILABLE_REASON_CODE,
    "required_attestation": "casf/parallel-live-capacity-attestation@1",
    "required_evidence": "current_generation_accepted_gate_and_live_telemetry",
    "metrics_omitted": True,
}
RESULT_STORAGE = (
    "Emit one canonical content-addressed unavailable/not-run artifact bound to the "
    "twice-observed clean repository commit and tree; it grants no scheduling, "
    "completion, acceptance, release, or promotion authority."
)
NONCLAIMS = (
    "This frozen recipe and its unavailable artifact are not benchmark measurements.",
    "Neither the one-supervisor baseline arm nor the twelve-supervisor candidate arm "
    "ran or qualified.",
    "No twelve-supervisor process launch occurred while the capability was unavailable.",
    "An in-process simulation, object graph, synthetic process count, or caller-supplied "
    "observation cannot qualify this benchmark.",
    "The minimum 3.0 throughput multiplier is a future acceptance threshold, not an "
    "observed value.",
    "No direct database access, file fallback, network access, credential use, provider "
    "call, or DuckLake scheduling authority occurred.",
    "Unavailable and not-run evidence contains no metrics and cannot establish completion, "
    "release, acceptance, promotion, or safe parallel behavior.",
    "This benchmark does not qualify 256-agent load, 64-slot concurrency, token efficiency, "
    "multihost behavior, or production behavior.",
)
FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {"metric", "metrics", "value", "values", "result_ref", "result_refs", "results"}
)
SECRET_SHAPES = (
    "-----begin private key-----",
    "authorization: bearer",
    "api_key=",
    "password=",
    "token=",
    "github_pat_",
    "ghp_",
    "sk-",
)


class ParallelBenchmarkError(ValueError):
    """Raised when an invocation, recipe, or unavailable artifact is invalid."""

    def __init__(self, message: str, *, code: str = "invalid_contract") -> None:
        super().__init__(message)
        self.code = code


class AdmittedParallelExecution(Protocol):
    """Future boundary; no implementation is admitted by this frozen manifest."""

    interface: str

    def execute(
        self,
        *,
        repository: Path,
        manifest: Mapping[str, Any],
        matrix_binding: Mapping[str, str],
    ) -> Mapping[str, Any]: ...


def _admitted_execution_boundary(
    _executor: AdmittedParallelExecution,
    *,
    repository: Path,
    manifest: Mapping[str, Any],
    matrix_binding: Mapping[str, str],
) -> NoReturn:
    """Structurally fence the dormant live path until a new admitted contract lands."""

    del repository, manifest, matrix_binding
    raise ParallelBenchmarkError(
        "admitted parallel execution is unavailable in this frozen contract",
        code="admitted_execution_unavailable",
    )


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


def _read_regular_bytes(path: Path | str, *, maximum_bytes: int = MAX_DOCUMENT_BYTES) -> bytes:
    """Read one bounded regular file without following its final symlink."""

    resolved = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ParallelBenchmarkError("cannot open a required bounded regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ParallelBenchmarkError("required input must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ParallelBenchmarkError("required input exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 16_384))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum_bytes:
            raise ParallelBenchmarkError("required input exceeds its byte bound")
        return raw
    except OSError as exc:
        raise ParallelBenchmarkError("cannot read a required bounded regular file") from exc
    finally:
        os.close(descriptor)


def _read_object(path: Path | str) -> dict[str, Any]:
    try:
        raw = _read_regular_bytes(path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParallelBenchmarkError("required JSON input is not UTF-8") from exc
    decoded = _loads_json(raw, name="bounded JSON input")
    if not isinstance(decoded, dict):
        raise ParallelBenchmarkError("required JSON input must contain an object")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_file_sha256(path: Path | str) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


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


def _require_text(value: Any, name: str, *, maximum_bytes: int = 16_384) -> str:
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum_bytes:
        raise ParallelBenchmarkError(f"{name} must be a bounded non-empty string")
    return value


def _require_identifier(value: Any, name: str) -> str:
    text = _require_text(value, name, maximum_bytes=MAX_IDENTIFIER_BYTES)
    lowered = text.casefold()
    if any(shape in lowered for shape in SECRET_SHAPES):
        raise ParallelBenchmarkError(f"{name} must not contain secret-shaped material")
    if any(
        character.isspace() or ord(character) < 33 or ord(character) > 126 for character in text
    ):
        raise ParallelBenchmarkError(f"{name} must be a bounded printable identifier")
    return text


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ParallelBenchmarkError(f"{name} must be a JSON boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ParallelBenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_float(value: Any, expected: float, name: str) -> float:
    if type(value) is not float or value != expected:
        raise ParallelBenchmarkError(f"{name} must be the JSON number {expected:.1f}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_identifier(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ParallelBenchmarkError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require_git_oid(value: Any, name: str) -> str:
    text = _require_identifier(value, name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ParallelBenchmarkError(f"{name} must be a full lowercase Git object id")
    return text


def _copy_arms(value: Any, name: str) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        raise ParallelBenchmarkError(f"{name} must contain exactly the baseline and candidate arms")
    checked: list[dict[str, Any]] = []
    for index, candidate in enumerate(value):
        arm = _require_mapping(candidate, f"{name}[{index}]")
        _require_exact_keys(
            arm,
            {
                "arm_id",
                "required_supervisor_processes",
                "availability",
                "execution_status",
                "qualified",
            },
            f"{name}[{index}]",
        )
        checked.append(
            {
                "arm_id": _require_identifier(arm["arm_id"], f"{name}[{index}].arm_id"),
                "required_supervisor_processes": _require_int(
                    arm["required_supervisor_processes"],
                    f"{name}[{index}].required_supervisor_processes",
                    minimum=1,
                ),
                "availability": _require_identifier(
                    arm["availability"], f"{name}[{index}].availability"
                ),
                "execution_status": _require_identifier(
                    arm["execution_status"], f"{name}[{index}].execution_status"
                ),
                "qualified": _require_bool(arm["qualified"], f"{name}[{index}].qualified"),
            }
        )
    if tuple(checked) != FUTURE_REQUIRED_ARMS:
        raise ParallelBenchmarkError(
            "parallel arms must remain unavailable/not-run with one baseline and twelve candidates"
        )
    return checked


def _validate_live_capability(value: Any, name: str = "live capability") -> dict[str, Any]:
    capability = _require_mapping(value, name)
    _require_exact_keys(capability, set(LIVE_CAPABILITY_UNAVAILABLE), name)
    _require_identifier(capability["availability"], f"{name}.availability")
    _require_identifier(capability["execution_status"], f"{name}.execution_status")
    _require_identifier(capability["reason_code"], f"{name}.reason_code")
    _require_identifier(capability["required_attestation"], f"{name}.required_attestation")
    _require_identifier(capability["required_evidence"], f"{name}.required_evidence")
    _require_bool(capability["metrics_omitted"], f"{name}.metrics_omitted")
    if dict(capability) != LIVE_CAPABILITY_UNAVAILABLE:
        raise ParallelBenchmarkError("live parallel capability must remain unavailable/not-run")
    return dict(capability)


def _capacity_preflight() -> dict[str, Any]:
    """Return the immutable no-I/O capacity fact before any caller input is inspected."""

    return dict(LIVE_CAPABILITY_UNAVAILABLE)


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload.pop("content_sha256", None)
    return _object_sha256(payload)


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    resolved = (
        Path(path) if path is not None else Path(__file__).with_name("parallel_manifest.json")
    )
    manifest = _read_object(resolved)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the complete frozen recipe, including JSON scalar types and prose."""

    manifest = _require_mapping(manifest, "manifest")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "benchmark_id",
            "program_id",
            "objective_id",
            "frozen",
            "state",
            "authoritative",
            "promotion_eligible",
            "matrix_binding",
            "execution",
            "parallel_comparison",
            "zero_tolerance_gates",
            "source_modules",
            "future_identity_requirements",
            "live_capability",
            "result_storage",
            "nonclaims",
        },
        "manifest",
    )
    expected_text = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "state": "capability_unavailable",
        "result_storage": RESULT_STORAGE,
    }
    for key, expected in expected_text.items():
        if _require_text(manifest[key], f"manifest.{key}") != expected:
            raise ParallelBenchmarkError(f"manifest.{key} has changed")
    if _require_bool(manifest["frozen"], "manifest.frozen") is not True:
        raise ParallelBenchmarkError("manifest must remain frozen")
    for key in ("authoritative", "promotion_eligible"):
        if _require_bool(manifest[key], f"manifest.{key}") is not False:
            raise ParallelBenchmarkError(f"manifest.{key} must remain false")

    matrix = _require_mapping(manifest["matrix_binding"], "manifest matrix binding")
    expected_matrix = {
        "relative_path": MATRIX_RELATIVE_PATH,
        "schema": MATRIX_SCHEMA,
        "sha256": MATRIX_SHA256,
    }
    _require_exact_keys(matrix, set(expected_matrix), "manifest matrix binding")
    for key, expected in expected_matrix.items():
        if _require_text(matrix[key], f"manifest matrix binding.{key}") != expected:
            raise ParallelBenchmarkError("manifest does not bind the exact frozen benchmark matrix")

    execution = _require_mapping(manifest["execution"], "manifest execution")
    expected_execution_keys = {
        "measurement_scope",
        "required_supervisor_processes",
        "subprocess_budget",
        "real_independent_processes_required",
        "in_process_simulation_qualifies",
        "admitted_execution_interface",
        "admitted_execution_available",
        "network_permitted",
        "direct_database_access_permitted",
        "ducklake_scheduling_authority_permitted",
        "launch_permitted",
    }
    _require_exact_keys(execution, expected_execution_keys, "manifest execution")
    if (
        _require_text(execution["measurement_scope"], "manifest execution scope")
        != MEASUREMENT_SCOPE
    ):
        raise ParallelBenchmarkError("manifest execution scope has changed")
    if (
        _require_identifier(
            execution["admitted_execution_interface"], "manifest admitted execution interface"
        )
        != ADMITTED_EXECUTION_INTERFACE
    ):
        raise ParallelBenchmarkError("manifest admitted execution interface has changed")
    for key in ("required_supervisor_processes", "subprocess_budget"):
        if _require_int(execution[key], f"manifest execution.{key}", minimum=1) != 12:
            raise ParallelBenchmarkError("manifest must preserve the twelve-process bound")
    expected_execution_bools = {
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "admitted_execution_available": False,
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "launch_permitted": False,
    }
    for key, expected in expected_execution_bools.items():
        if _require_bool(execution[key], f"manifest execution.{key}") is not expected:
            raise ParallelBenchmarkError("manifest execution may not launch or weaken assurance")

    comparison = _require_mapping(manifest["parallel_comparison"], "parallel comparison")
    _require_exact_keys(
        comparison,
        {
            "future_required_arms",
            "comparison_identity",
            "minimum_accepted_task_throughput_multiplier",
            "lower_assurance_permitted",
            "measurement_status",
        },
        "parallel comparison",
    )
    _copy_arms(comparison["future_required_arms"], "parallel comparison arms")
    if (
        _require_identifier(comparison["comparison_identity"], "comparison identity")
        != "same-host-tasks-providers-tests-proofs-budgets"
        or _require_identifier(comparison["measurement_status"], "comparison status") != "not_run"
        or _require_bool(comparison["lower_assurance_permitted"], "lower assurance") is not False
    ):
        raise ParallelBenchmarkError("parallel comparison assurance or status has changed")
    _require_exact_float(
        comparison["minimum_accepted_task_throughput_multiplier"],
        3.0,
        "minimum throughput multiplier",
    )

    if (
        type(manifest["zero_tolerance_gates"]) is not list
        or tuple(manifest["zero_tolerance_gates"]) != ZERO_TOLERANCE_GATES
    ):
        raise ParallelBenchmarkError("zero-tolerance safety gates have changed")
    if type(manifest["source_modules"]) is not list or tuple(manifest["source_modules"]) != (
        SOURCE_RELATIVE_PATHS
    ):
        raise ParallelBenchmarkError("measured source-module set has changed")
    if (
        type(manifest["future_identity_requirements"]) is not list
        or tuple(manifest["future_identity_requirements"]) != FUTURE_IDENTITY_REQUIREMENTS
    ):
        raise ParallelBenchmarkError("future live identity requirements have changed")
    _validate_live_capability(manifest["live_capability"])
    if type(manifest["nonclaims"]) is not list or tuple(manifest["nonclaims"]) != NONCLAIMS:
        raise ParallelBenchmarkError("manifest ordered nonclaims have changed")
    for index, nonclaim in enumerate(manifest["nonclaims"]):
        _require_text(nonclaim, f"manifest nonclaims[{index}]")


def validate_matrix_binding(
    manifest: Mapping[str, Any], matrix_path: Path | str | None = None
) -> dict[str, str]:
    validate_manifest(manifest)
    resolved = (
        Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml")
    )
    raw = _read_regular_bytes(resolved)
    if hashlib.sha256(raw).hexdigest() != MATRIX_SHA256:
        raise ParallelBenchmarkError("frozen benchmark matrix content is stale or changed")
    try:
        first_line = raw.decode("utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise ParallelBenchmarkError("frozen benchmark matrix is missing its schema") from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise ParallelBenchmarkError("frozen benchmark matrix schema has changed")
    return dict(manifest["matrix_binding"])


def _safe_resolve(path: Path | str, *, error_code: str) -> Path:
    """Resolve a caller path without exposing it through an exception or traceback."""

    messages = {
        "invalid_repository": "repository path cannot be resolved safely",
        "invalid_recipe_path": "recipe path cannot be resolved safely",
    }
    if error_code not in messages:
        raise ParallelBenchmarkError("unsupported path classification")
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ParallelBenchmarkError(messages[error_code], code=error_code) from None


def _bound_repository_and_recipe(
    repository: Path | str,
    manifest_path: Path | str | None,
    matrix_path: Path | str | None,
) -> tuple[Path, Path, Path]:
    root = _safe_resolve(repository, error_code="invalid_repository")
    runner_root = _safe_resolve(Path(__file__), error_code="invalid_repository").parents[3]
    if root != runner_root or not root.is_dir():
        raise ParallelBenchmarkError(
            "repository must be the exact repository containing this benchmark runner",
            code="invalid_repository",
        )
    expected_manifest = root / MANIFEST_RELATIVE_PATH
    expected_matrix = root / MATRIX_RELATIVE_PATH
    resolved_manifest = (
        _safe_resolve(expected_manifest, error_code="invalid_recipe_path")
        if manifest_path is None
        else _safe_resolve(manifest_path, error_code="invalid_recipe_path")
    )
    resolved_matrix = (
        _safe_resolve(expected_matrix, error_code="invalid_recipe_path")
        if matrix_path is None
        else _safe_resolve(matrix_path, error_code="invalid_recipe_path")
    )
    canonical_manifest = _safe_resolve(expected_manifest, error_code="invalid_recipe_path")
    canonical_matrix = _safe_resolve(expected_matrix, error_code="invalid_recipe_path")
    if resolved_manifest != canonical_manifest or resolved_matrix != canonical_matrix:
        raise ParallelBenchmarkError(
            "manifest and matrix must be repository-contained canonical paths",
            code="invalid_recipe_path",
        )
    return root, resolved_manifest, resolved_matrix


def _git(repository: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ParallelBenchmarkError(
            "cannot establish bounded repository identity", code="repository_identity_unavailable"
        ) from exc
    return completed.stdout.strip()


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = _safe_resolve(repository, error_code="invalid_repository")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ParallelBenchmarkError(
            "repository is dirty; exact current-tree evidence is unavailable",
            code="repository_not_clean",
        )
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    _require_git_oid(commit, "repository commit")
    _require_git_oid(tree, "repository tree")
    return {"repository_commit": commit, "repository_tree": tree}


def _manifest_binding(manifest_path: Path) -> dict[str, str]:
    return {
        "relative_path": MANIFEST_RELATIVE_PATH,
        "schema": MANIFEST_SCHEMA,
        "raw_sha256": _raw_file_sha256(manifest_path),
    }


def _reject_observation_payload(value: Any, name: str = "benchmark result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_OBSERVATION_KEYS:
                raise ParallelBenchmarkError(f"{name} contains forbidden observation field {key}")
            _reject_observation_payload(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_observation_payload(child, f"{name}[{index}]")


def run_benchmark(
    *,
    repository: Path | str,
    identities: Mapping[str, Any] | None = None,
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return canonical unavailable evidence without inspecting caller identities.

    Capacity preflight is intentionally the first operation.  ``identities`` is
    accepted only for API compatibility and is never iterated, indexed, logged,
    serialized, or read from a path in the unavailable state.
    """

    capacity = _capacity_preflight()
    checked_capacity = _validate_live_capability(capacity, "capacity preflight")
    del identities

    repository_root, resolved_manifest, resolved_matrix = _bound_repository_and_recipe(
        repository, manifest_path, matrix_path
    )
    observed_before = repository_identity(repository_root)
    manifest = load_manifest(resolved_manifest)
    matrix_binding = validate_matrix_binding(manifest, resolved_matrix)
    manifest_binding = _manifest_binding(resolved_manifest)

    comparison = manifest["parallel_comparison"]
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "manifest_binding": manifest_binding,
        "matrix_binding": matrix_binding,
        "content_sha256": "",
        "availability": "unavailable",
        "execution_status": "not_run",
        "ran": False,
        "qualified": False,
        "authoritative": False,
        "promotion_eligible": False,
        "metrics_omitted": True,
        "reason_code": UNAVAILABLE_REASON_CODE,
        "measurement_scope": MEASUREMENT_SCOPE,
        "future_required_arms": [dict(arm) for arm in comparison["future_required_arms"]],
        "comparison_requirement": {
            "comparison_identity": comparison["comparison_identity"],
            "minimum_accepted_task_throughput_multiplier": comparison[
                "minimum_accepted_task_throughput_multiplier"
            ],
            "lower_assurance_permitted": comparison["lower_assurance_permitted"],
            "measurement_status": comparison["measurement_status"],
        },
        "repository_binding": {
            **observed_before,
            "clean": True,
            "observed_before_and_after": True,
        },
        "capability_preflight": checked_capacity,
        "nonclaims": list(manifest["nonclaims"]),
    }

    result["content_sha256"] = result_content_sha256(result)
    return validate_result(
        result,
        manifest_path=resolved_manifest,
        matrix_path=resolved_matrix,
        current_identity=observed_before,
    )


def validate_result(
    result: Mapping[str, Any],
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
    current_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and return the closed unavailable artifact for CASF-042/043."""

    repository_root, manifest_file, matrix_file = _bound_repository_and_recipe(
        _safe_resolve(Path(__file__), error_code="invalid_repository").parents[3],
        manifest_path,
        matrix_path,
    )
    manifest = load_manifest(manifest_file)
    matrix_binding = validate_matrix_binding(manifest, matrix_file)
    result = _require_mapping(result, "benchmark result")
    _reject_observation_payload(result)
    _require_exact_keys(
        result,
        {
            "schema",
            "benchmark_id",
            "program_id",
            "objective_id",
            "manifest_binding",
            "matrix_binding",
            "content_sha256",
            "availability",
            "execution_status",
            "ran",
            "qualified",
            "authoritative",
            "promotion_eligible",
            "metrics_omitted",
            "reason_code",
            "measurement_scope",
            "future_required_arms",
            "comparison_requirement",
            "repository_binding",
            "capability_preflight",
            "nonclaims",
        },
        "benchmark result",
    )
    expected_text = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "availability": "unavailable",
        "execution_status": "not_run",
        "reason_code": UNAVAILABLE_REASON_CODE,
        "measurement_scope": MEASUREMENT_SCOPE,
    }
    for key, expected in expected_text.items():
        if _require_text(result[key], f"result.{key}") != expected:
            raise ParallelBenchmarkError(f"result.{key} has changed")
    for key, expected in {
        "ran": False,
        "qualified": False,
        "authoritative": False,
        "promotion_eligible": False,
        "metrics_omitted": True,
    }.items():
        if _require_bool(result[key], f"result.{key}") is not expected:
            raise ParallelBenchmarkError(f"result.{key} has changed")

    manifest_binding = _require_mapping(result["manifest_binding"], "manifest binding")
    expected_manifest_binding = _manifest_binding(manifest_file)
    _require_exact_keys(manifest_binding, set(expected_manifest_binding), "manifest binding")
    _require_sha256(manifest_binding["raw_sha256"], "manifest raw SHA-256")
    if dict(manifest_binding) != expected_manifest_binding:
        raise ParallelBenchmarkError("result raw manifest binding is stale or changed")
    bound_matrix = _require_mapping(result["matrix_binding"], "result matrix binding")
    _require_exact_keys(bound_matrix, set(matrix_binding), "result matrix binding")
    if dict(bound_matrix) != matrix_binding:
        raise ParallelBenchmarkError("result matrix binding is stale or changed")

    _copy_arms(result["future_required_arms"], "result future required arms")
    requirement = _require_mapping(result["comparison_requirement"], "comparison requirement")
    _require_exact_keys(
        requirement,
        {
            "comparison_identity",
            "minimum_accepted_task_throughput_multiplier",
            "lower_assurance_permitted",
            "measurement_status",
        },
        "comparison requirement",
    )
    expected_requirement = {
        key: manifest["parallel_comparison"][key]
        for key in (
            "comparison_identity",
            "minimum_accepted_task_throughput_multiplier",
            "lower_assurance_permitted",
            "measurement_status",
        )
    }
    _require_exact_float(
        requirement["minimum_accepted_task_throughput_multiplier"],
        3.0,
        "result minimum throughput multiplier",
    )
    _require_bool(requirement["lower_assurance_permitted"], "result lower assurance")
    if dict(requirement) != expected_requirement:
        raise ParallelBenchmarkError("result comparison requirement has changed")

    repository = _require_mapping(result["repository_binding"], "repository binding")
    _require_exact_keys(
        repository,
        {"repository_commit", "repository_tree", "clean", "observed_before_and_after"},
        "repository binding",
    )
    _require_git_oid(repository["repository_commit"], "bound repository commit")
    _require_git_oid(repository["repository_tree"], "bound repository tree")
    if (
        _require_bool(repository["clean"], "repository clean") is not True
        or _require_bool(repository["observed_before_and_after"], "repository twice observed")
        is not True
    ):
        raise ParallelBenchmarkError("repository binding must be clean and observed twice")
    observed_current = repository_identity(repository_root)
    if current_identity is not None:
        asserted_current = _require_mapping(current_identity, "current repository identity")
        _require_exact_keys(
            asserted_current,
            {"repository_commit", "repository_tree"},
            "current identity",
        )
        _require_git_oid(asserted_current["repository_commit"], "current repository commit")
        _require_git_oid(asserted_current["repository_tree"], "current repository tree")
        if dict(asserted_current) != observed_current:
            raise ParallelBenchmarkError(
                "repository commit or tree changed during unavailable-result construction",
                code="repository_identity_changed",
            )
    if any(
        repository[key] != observed_current[key] for key in ("repository_commit", "repository_tree")
    ):
        raise ParallelBenchmarkError("parallel benchmark artifact is stale for the current tree")

    _validate_live_capability(result["capability_preflight"], "result capacity preflight")
    if type(result["nonclaims"]) is not list or tuple(result["nonclaims"]) != NONCLAIMS:
        raise ParallelBenchmarkError("result ordered nonclaims have changed")
    _require_sha256(result["content_sha256"], "result content SHA-256")
    if result["content_sha256"] != result_content_sha256(result):
        raise ParallelBenchmarkError("result content address is stale or changed")
    return dict(result)


def _invalid_artifact(error_code: str) -> dict[str, str]:
    messages = {
        "missing_required_argument": "repository is required",
        "invalid_repository": "repository does not contain this benchmark runner",
        "invalid_recipe_path": "manifest or matrix path is outside the canonical recipe",
        "repository_not_clean": "repository is not clean",
        "repository_identity_unavailable": "repository identity is unavailable",
        "invalid_contract": "parallel benchmark invocation or contract is invalid",
    }
    stable_code = error_code if error_code in messages else "invalid_contract"
    return {
        "schema": ERROR_SCHEMA,
        "execution_status": "invalid",
        "error_code": stable_code,
        "message": messages[stable_code],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, help="clean repository containing this runner")
    parser.add_argument(
        "--identities",
        type=Path,
        help="reserved future input; never opened while live capacity is unavailable",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("parallel_manifest.json")
    )
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matrix.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repository is None:
        print(json.dumps(_invalid_artifact("missing_required_argument"), sort_keys=True))
        return 2
    try:
        # Do not open, resolve, stat, stringify, or otherwise inspect args.identities.
        result = run_benchmark(
            repository=args.repository,
            manifest_path=args.manifest,
            matrix_path=args.matrix,
        )
    except ParallelBenchmarkError as exc:
        print(json.dumps(_invalid_artifact(exc.code), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
