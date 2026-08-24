#!/usr/bin/env python3
"""Fail-closed cross-supervisor token-efficiency benchmark for CASF-041.

The frozen matrix requires an equivalent one-supervisor baseline and a real
twelve-supervisor candidate.  The current tree has no admitted live profile,
so this runner deliberately emits only a deterministic unavailable receipt.
It never substitutes caller telemetry, an in-process simulation, a database,
or zeros for a measurement.  The dormant admission validator preserves the
future measurement contract without granting it any authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, NoReturn, Protocol

MANIFEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation/token-benchmark-manifest@1"
RESULT_SCHEMA = "casf/token-benchmark@1"
ERROR_SCHEMA = "casf/token-benchmark-error@1"
CAPACITY_PREFLIGHT_SCHEMA = "casf/token-capacity-preflight@1"
LIVE_ATTESTATION_SCHEMA = "casf/token-live-capacity-attestation@1"
LIVE_OBSERVATION_SCHEMA = "casf/token-live-observation@1"
ADMITTED_EXECUTION_INTERFACE = "CASFTokenAdmittedExecution@1"
BENCHMARK_ID = "casf-token-cross-supervisor-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
OBJECTIVE_ID = "CASF-041"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
MANIFEST_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/token_manifest.json"
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_token.py"
MEASUREMENT_SCOPE = "one_baseline_and_twelve_independent_supervisor_processes_qualified_live_token_comparison"
REASON_CODE = "qualified_cross_supervisor_token_live_capacity_not_admitted"
BASELINE_SUPERVISOR_PROCESSES = 1
CANDIDATE_SUPERVISOR_PROCESSES = 12

SOURCE_RELATIVE_PATHS = (MANIFEST_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
REQUIRED_IDENTITIES = (
    "repository_commit", "repository_tree", "control_plane_generation", "schema_fingerprint",
    "policy_ref", "policy_revision", "capability_ref", "federation_id", "supervisor_id",
    "task_id", "attempt_id", "worktree_id", "assignment_revision", "fencing_epoch",
)
TOKEN_GATES = {
    "minimum_repeated_context_token_reduction_percent": 50,
    "minimum_input_tokens_per_accepted_criterion_reduction_percent": 40,
    "minimum_duplicate_model_call_reduction_percent": 60,
    "minimum_eligible_semantic_capsule_reuse_percent": 70,
    "minimum_complete_board_scan_reduction_percent": 80,
}
ZERO_TOLERANCE_GATES = (
    "lost_deliveries", "duplicate_committed_effects", "stale_fence_completion",
    "unauthorized_creation", "tenant_leakage", "agent_sql", "secret_leaks",
    "causal_notification_loss", "stale_abstraction_suppression",
    "model_created_authority_or_completion", "ducklake_authority_promotion", "reduced_assurance",
)
NONCLAIMS = (
    "This frozen recipe and its unavailable artifact are not benchmark measurements.",
    "Neither the one-supervisor baseline arm nor the twelve-supervisor candidate arm ran or qualified.",
    "The token-reduction and semantic-capsule-reuse thresholds are future acceptance thresholds, not observed metric values.",
    "An in-process simulation, object graph, synthetic process count, or caller-supplied telemetry cannot qualify this benchmark.",
    "No direct database access, file fallback, network access, credential use, provider call, or DuckLake scheduling authority occurred.",
    "Any future live authority requires an authenticated typed Quack admission and can never come from direct DuckDB access or DuckLake.",
    "Unavailable and not-run evidence contains no metrics and cannot establish completion, release, acceptance, promotion, or token efficiency.",
    "This benchmark does not qualify load, multihost behavior, production behavior, or any capability outside its separately admitted live profile.",
)
RESULT_STORAGE = (
    "Emit one canonical content-addressed unavailable/not-run artifact bound to the "
    "twice-observed clean repository commit and tree, exact manifest, matrix, and "
    "source bytes; it grants no scheduling, completion, acceptance, release, or promotion authority."
)
_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_MATRIX_BYTES = 128 * 1024
_MAX_RUNNER_BYTES = 2 * 1024 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}\Z")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})\Z")
_SECRET_VALUE = re.compile(r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:api[_-]?key|access[_-]?token|token|password|passwd|secret)\s*[:=]\s*\S+|(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{8,})")

_EXECUTION = {
    "measurement_scope": MEASUREMENT_SCOPE,
    "baseline_required_supervisor_processes": BASELINE_SUPERVISOR_PROCESSES,
    "candidate_required_supervisor_processes": CANDIDATE_SUPERVISOR_PROCESSES,
    "subprocess_budget": CANDIDATE_SUPERVISOR_PROCESSES,
    "real_independent_processes_required": True,
    "in_process_simulation_qualifies": False,
    "admitted_execution_interface": ADMITTED_EXECUTION_INTERFACE,
    "admitted_execution_available": False,
    "network_permitted": False,
    "authenticated_typed_quack_required": True,
    "direct_database_access_permitted": False,
    "ducklake_scheduling_authority_permitted": False,
    "launch_permitted": False,
}
_PREFLIGHT = {
    "schema": CAPACITY_PREFLIGHT_SCHEMA, "current_supervisor_processes": 1,
    "required_baseline_supervisor_processes": BASELINE_SUPERVISOR_PROCESSES,
    "required_candidate_supervisor_processes": CANDIDATE_SUPERVISOR_PROCESSES,
    "authenticated_typed_quack_live_capacity": False, "availability": "unavailable",
    "reason_code": REASON_CODE,
}
_LIVE_CAPABILITY = {
    "availability": "unavailable", "execution_status": "not_run", "ran": False,
    "qualified": False, "reason_code": REASON_CODE,
    "required_attestation": LIVE_ATTESTATION_SCHEMA,
    "required_evidence": "current_generation_accepted_gate_current_fences_and_live_host_provider_token_telemetry",
    "metrics_omitted": True,
}
_ARM_BASELINE = {"arm_id": "baseline", "required_supervisor_processes": 1, "availability": "unavailable", "execution_status": "not_run", "qualified": False}
_ARM_CANDIDATE = {"arm_id": "candidate", "required_supervisor_processes": 12, "availability": "unavailable", "execution_status": "not_run", "qualified": False}
_COMPARISON = {"comparison_identity": "same-host-tasks-providers-tests-proofs-budgets", "baseline_arm": _ARM_BASELINE, "candidate_arm": _ARM_CANDIDATE, "measurement_status": "not_run", "lower_assurance_permitted": False}
_EXPECTED_MANIFEST = {
    "schema": MANIFEST_SCHEMA, "benchmark_id": BENCHMARK_ID, "program_id": PROGRAM_ID,
    "objective_id": OBJECTIVE_ID, "frozen": True, "state": "capability_unavailable",
    "authoritative": False, "promotion_eligible": False,
    "matrix_binding": {"relative_path": MATRIX_RELATIVE_PATH, "schema": MATRIX_SCHEMA, "sha256": MATRIX_SHA256},
    "execution": _EXECUTION, "capacity_preflight": _PREFLIGHT, "comparison": _COMPARISON,
    "token_gates": TOKEN_GATES, "zero_tolerance_gates": list(ZERO_TOLERANCE_GATES),
    "future_identity_requirements": list(REQUIRED_IDENTITIES), "source_modules": list(SOURCE_RELATIVE_PATHS),
    "live_capability": _LIVE_CAPABILITY, "result_storage": RESULT_STORAGE, "nonclaims": list(NONCLAIMS),
}


class TokenBenchmarkError(ValueError):
    def __init__(self, message: str, *, reason_code: str = "invalid_contract") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TokenCapabilityUnavailable(TokenBenchmarkError):
    """The caller requested a live profile that the frozen preflight does not admit."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TokenBenchmarkError("duplicate JSON key", reason_code="duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise TokenBenchmarkError(f"non-finite JSON number {value!r} is prohibited", reason_code="invalid_json")


def _loads_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_json)
    except json.JSONDecodeError as exc:
        raise TokenBenchmarkError(f"{name} is not valid JSON", reason_code="invalid_json") from exc


def _read_bounded_regular_bytes(path: Path | str, *, name: str, maximum_bytes: int) -> bytes:
    candidate = Path(path)
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise TokenBenchmarkError(f"{name} is unavailable", reason_code="source_unavailable") from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise TokenBenchmarkError(f"{name} must be a regular non-symlink file", reason_code="unsafe_source_path")
    if initial.st_size > maximum_bytes:
        raise TokenBenchmarkError(f"{name} exceeds its byte limit", reason_code="source_too_large")
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise TokenBenchmarkError(f"{name} is not a bounded regular file", reason_code="unsafe_source_path")
        if any(getattr(initial, field) != getattr(before, field) for field in ("st_dev", "st_ino", "st_mode")):
            raise TokenBenchmarkError(f"{name} changed before its bounded read", reason_code="source_changed")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except TokenBenchmarkError:
        raise
    except OSError as exc:
        raise TokenBenchmarkError(f"{name} could not be read safely", reason_code="source_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise TokenBenchmarkError(f"{name} exceeds its byte limit", reason_code="source_too_large")
    if any(getattr(before, field) != getattr(after, field) for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")):
        raise TokenBenchmarkError(f"{name} changed during its bounded read", reason_code="source_changed")
    return payload


def _read_object(path: Path | str) -> dict[str, Any]:
    raw = _read_bounded_regular_bytes(path, name="JSON input", maximum_bytes=_MAX_MANIFEST_BYTES)
    try:
        decoded = _loads_json(raw.decode("utf-8", errors="strict"), name="JSON input")
    except UnicodeError as exc:
        raise TokenBenchmarkError("JSON input is not UTF-8", reason_code="invalid_json") from exc
    if not isinstance(decoded, dict):
        raise TokenBenchmarkError("JSON input must contain a JSON object", reason_code="invalid_json")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenBenchmarkError(f"{name} must be an object")
    return value


def _exact_structure(value: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        actual = _mapping(value, name)
        if set(actual) != set(expected):
            raise TokenBenchmarkError(f"{name} has a closed schema", reason_code="closed_schema_violation")
        for key, child in expected.items():
            _exact_structure(actual[key], child, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise TokenBenchmarkError(f"{name} must be the exact frozen sequence")
        for index, (actual, child) in enumerate(zip(value, expected, strict=True)):
            _exact_structure(actual, child, f"{name}[{index}]")
    elif type(value) is not type(expected) or value != expected:
        raise TokenBenchmarkError(f"{name} differs from the exact frozen value", reason_code="frozen_contract_changed")


def _text(value: Any, name: str, *, maximum_bytes: int = 512) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value.encode("utf-8")) > maximum_bytes or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise TokenBenchmarkError(f"{name} must be bounded exact text")
    return value


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _SECRET_VALUE.search(text):
        raise TokenBenchmarkError(f"{name} contains credential-shaped material", reason_code="secret_shaped_input")
    if _TOKEN.fullmatch(text) is None:
        raise TokenBenchmarkError(f"{name} must be a compact identity")
    return text


def _content_ref(value: Any, name: str) -> str:
    text = _token(value, name)
    if _CONTENT_REF.fullmatch(text) is None:
        raise TokenBenchmarkError(f"{name} must be a content-addressed reference")
    return text


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TokenBenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _exact_structure(manifest, _EXPECTED_MANIFEST, "manifest")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return _sha256(_canonical_bytes(manifest))


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest = _read_object(Path(path) if path is not None else Path(__file__).with_name("token_manifest.json"))
    validate_manifest(manifest)
    return manifest


def _validate_matrix(raw: bytes) -> None:
    if _sha256(raw) != MATRIX_SHA256:
        raise TokenBenchmarkError("frozen benchmark matrix content is stale or changed", reason_code="matrix_binding_changed")
    try:
        first_line = raw.decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError) as exc:
        raise TokenBenchmarkError("frozen benchmark matrix is missing its schema", reason_code="matrix_binding_changed") from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise TokenBenchmarkError("frozen benchmark matrix schema has changed", reason_code="matrix_binding_changed")


def validate_matrix_binding(manifest: Mapping[str, Any], matrix_path: Path | str | None = None) -> dict[str, str]:
    validate_manifest(manifest)
    raw = _read_bounded_regular_bytes(Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml"), name="benchmark matrix", maximum_bytes=_MAX_MATRIX_BYTES)
    _validate_matrix(raw)
    return dict(manifest["matrix_binding"])


def _existing_path(path: Path | str, *, name: str) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(path)))
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise TokenBenchmarkError(f"{name} is unavailable", reason_code="unsafe_source_path") from None
    if lexical != resolved:
        raise TokenBenchmarkError(f"{name} may not traverse a symlink", reason_code="unsafe_source_path")
    return resolved


def _bound_repository_and_recipe(repository: Path | str, manifest_path: Path | str | None, matrix_path: Path | str | None) -> tuple[Path, Path, Path]:
    root = _existing_path(repository, name="repository")
    if root != _existing_path(__file__, name="benchmark runner").parents[3]:
        raise TokenBenchmarkError("repository must contain this exact benchmark runner", reason_code="repository_binding_changed")
    manifest = _existing_path(root / MANIFEST_RELATIVE_PATH if manifest_path is None else manifest_path, name="benchmark manifest")
    matrix = _existing_path(root / MATRIX_RELATIVE_PATH if matrix_path is None else matrix_path, name="benchmark matrix")
    if manifest != root / MANIFEST_RELATIVE_PATH or matrix != root / MATRIX_RELATIVE_PATH:
        raise TokenBenchmarkError("manifest and matrix must be exact measured-tree files", reason_code="repository_binding_changed")
    return root, manifest, matrix


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TokenBenchmarkError("cannot establish repository identity", reason_code="repository_identity_unavailable") from exc
    return result.stdout.strip()


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = _existing_path(repository, name="repository")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise TokenBenchmarkError("repository is dirty; exact current-tree evidence is unavailable", reason_code="repository_dirty")
    commit, tree = _git(root, "rev-parse", "--verify", "HEAD"), _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise TokenBenchmarkError("repository identity is not a full lowercase Git object id")
    return {"repository_commit": commit, "repository_tree": tree}


class _Snapshot(NamedTuple):
    identity: dict[str, str]
    manifest: dict[str, Any]
    manifest_sha256: str
    matrix_sha256: str
    source_sha256: dict[str, str]


def _snapshot(repository: Path, manifest_path: Path, matrix_path: Path) -> _Snapshot:
    before = repository_identity(repository)
    paths = {MANIFEST_RELATIVE_PATH: manifest_path, RUNNER_RELATIVE_PATH: repository / RUNNER_RELATIVE_PATH}
    def read_sources() -> dict[str, bytes]:
        return {path: _read_bounded_regular_bytes(value, name=f"measured source {index}", maximum_bytes=_MAX_MANIFEST_BYTES if path.endswith(".json") else _MAX_RUNNER_BYTES) for index, (path, value) in enumerate(paths.items())}
    first, matrix_first = read_sources(), _read_bounded_regular_bytes(matrix_path, name="benchmark matrix", maximum_bytes=_MAX_MATRIX_BYTES)
    try:
        manifest = _loads_json(first[MANIFEST_RELATIVE_PATH].decode("utf-8", errors="strict"), name="manifest")
    except UnicodeError as exc:
        raise TokenBenchmarkError("manifest is not UTF-8", reason_code="invalid_json") from exc
    if not isinstance(manifest, dict):
        raise TokenBenchmarkError("manifest must contain a JSON object", reason_code="invalid_json")
    validate_manifest(manifest)
    _validate_matrix(matrix_first)
    second, matrix_second, after = read_sources(), _read_bounded_regular_bytes(matrix_path, name="benchmark matrix", maximum_bytes=_MAX_MATRIX_BYTES), repository_identity(repository)
    if before != after or first != second or matrix_first != matrix_second:
        raise TokenBenchmarkError("repository identity or measured source changed during evidence collection", reason_code="source_changed")
    return _Snapshot(before, manifest, _sha256(first[MANIFEST_RELATIVE_PATH]), _sha256(matrix_first), {key: _sha256(value) for key, value in first.items()})


def capacity_preflight() -> dict[str, Any]:
    """Return the frozen unavailable capacity without inspecting caller input."""
    return dict(_PREFLIGHT)


def _validate_identities(value: Any) -> dict[str, Any]:
    """Validate the identity packet a future admitted run must bind exactly."""
    identities = _mapping(value, "benchmark identities")
    if set(identities) != set(REQUIRED_IDENTITIES):
        raise TokenBenchmarkError("benchmark identities has a closed schema", reason_code="closed_schema_violation")
    checked = dict(identities)
    for key in ("repository_commit", "repository_tree"):
        candidate = _text(checked[key], f"identities.{key}", maximum_bytes=40)
        if re.fullmatch(r"[0-9a-f]{40}", candidate) is None:
            raise TokenBenchmarkError(f"identities.{key} must be a full lowercase Git object id")
        checked[key] = candidate
    for key in ("control_plane_generation", "assignment_revision", "fencing_epoch"):
        checked[key] = _integer(checked[key], f"identities.{key}", minimum=1)
    for key in set(REQUIRED_IDENTITIES) - {"repository_commit", "repository_tree", "control_plane_generation", "assignment_revision", "fencing_epoch"}:
        checked[key] = _token(checked[key], f"identities.{key}")
    if checked["task_id"] != OBJECTIVE_ID:
        raise TokenBenchmarkError("identities.task_id must bind CASF-041")
    return checked


def live_capability(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    checked = load_manifest() if manifest is None else dict(manifest)
    validate_manifest(checked)
    return dict(checked["live_capability"])


def require_live_capability(manifest: Mapping[str, Any]) -> NoReturn:
    validate_manifest(manifest)
    raise TokenCapabilityUnavailable("qualified cross-supervisor token live capacity is unavailable", reason_code=REASON_CODE)


class TokenAdmission(NamedTuple):
    schema: str
    baseline_supervisor_processes: int
    candidate_supervisor_processes: int
    state_authority: str
    quack_receipt_ref: str
    direct_database_access_permitted: bool
    ducklake_scheduling_authority_permitted: bool


class AdmittedTokenExecutor(Protocol):
    interface: str
    def execute_token_comparison(self, *, admission: TokenAdmission) -> Mapping[str, Any]: ...


def _validate_token_admission(admission: TokenAdmission) -> None:
    if type(admission) is not TokenAdmission:
        raise TokenBenchmarkError("live token admission must use the closed typed contract")
    _exact_structure({
        "schema": admission.schema,
        "baseline_supervisor_processes": admission.baseline_supervisor_processes,
        "candidate_supervisor_processes": admission.candidate_supervisor_processes,
        "state_authority": admission.state_authority,
        "direct_database_access_permitted": admission.direct_database_access_permitted,
        "ducklake_scheduling_authority_permitted": admission.ducklake_scheduling_authority_permitted,
    }, {
        "schema": LIVE_ATTESTATION_SCHEMA,
        "baseline_supervisor_processes": 1,
        "candidate_supervisor_processes": 12,
        "state_authority": "authenticated_typed_quack",
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
    }, "live token admission")
    _content_ref(admission.quack_receipt_ref, "admission.quack_receipt_ref")


# Kept as a private compatibility alias while the named validator is the contract surface.
_validate_admission = _validate_token_admission


def validate_admitted_token_observation(admission: TokenAdmission, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a future raw observation; this function never qualifies or promotes it."""
    _validate_token_admission(admission)
    value = _mapping(observation, "token observation")
    expected = {"schema", "comparison_identity", "baseline_supervisor_processes", "candidate_supervisor_processes", "state_transport", "quack_receipt_ref", "direct_database_access_used", "ducklake_scheduling_authority_used", "zero_tolerance_gate_failures", "repeated_context_token_reduction_percent", "input_tokens_per_accepted_criterion_reduction_percent", "duplicate_model_call_reduction_percent", "eligible_semantic_capsule_reuse_percent", "complete_board_scan_reduction_percent"}
    if set(value) != expected:
        raise TokenBenchmarkError("token observation has a closed schema", reason_code="closed_schema_violation")
    if type(value["schema"]) is not str or type(value["comparison_identity"]) is not str or value["schema"] != LIVE_OBSERVATION_SCHEMA or value["comparison_identity"] != _COMPARISON["comparison_identity"]:
        raise TokenBenchmarkError("token observation is not like-for-like")
    if type(value["baseline_supervisor_processes"]) is not int or type(value["candidate_supervisor_processes"]) is not int or value["baseline_supervisor_processes"] != 1 or value["candidate_supervisor_processes"] != 12:
        raise TokenBenchmarkError("token observation violates the 1/12-supervisor bound")
    if _token(value["state_transport"], "token observation.state_transport") != "authenticated_typed_quack" or _content_ref(value["quack_receipt_ref"], "token observation.quack_receipt_ref") != admission.quack_receipt_ref:
        raise TokenBenchmarkError("token observation lacks the admitted typed Quack authority")
    if type(value["direct_database_access_used"]) is not bool or type(value["ducklake_scheduling_authority_used"]) is not bool or value["direct_database_access_used"] is not False or value["ducklake_scheduling_authority_used"] is not False:
        raise TokenBenchmarkError("token observation used a prohibited authority")
    gates = _mapping(value["zero_tolerance_gate_failures"], "zero_tolerance_gate_failures")
    if set(gates) != set(ZERO_TOLERANCE_GATES) or any(type(gates[key]) is not int or gates[key] != 0 for key in gates):
        raise TokenBenchmarkError("token observation fails a zero-tolerance gate")
    mapping = {"repeated_context_token_reduction_percent": "minimum_repeated_context_token_reduction_percent", "input_tokens_per_accepted_criterion_reduction_percent": "minimum_input_tokens_per_accepted_criterion_reduction_percent", "duplicate_model_call_reduction_percent": "minimum_duplicate_model_call_reduction_percent", "eligible_semantic_capsule_reuse_percent": "minimum_eligible_semantic_capsule_reuse_percent", "complete_board_scan_reduction_percent": "minimum_complete_board_scan_reduction_percent"}
    for key, threshold in mapping.items():
        metric = _integer(value[key], f"token observation.{key}")
        if metric < TOKEN_GATES[threshold]:
            raise TokenBenchmarkError(f"token observation fails {key}")
    return dict(value)


def execute_admitted_token(admission: TokenAdmission, executor: AdmittedTokenExecutor) -> NoReturn:
    """Fence the future effect boundary without invoking its executor."""
    if _token(getattr(executor, "interface", None), "admitted token executor interface") != ADMITTED_EXECUTION_INTERFACE:
        raise TokenBenchmarkError("admitted token executor interface is invalid")
    _validate_token_admission(admission)
    raise TokenCapabilityUnavailable("admitted token execution is unavailable at current capacity", reason_code="admitted_execution_unavailable")


def _admitted_execution_boundary(executor: AdmittedTokenExecutor, *, admission: TokenAdmission) -> NoReturn:
    return execute_admitted_token(admission, executor)


def _result(snapshot: _Snapshot) -> dict[str, Any]:
    result = {
        "schema": RESULT_SCHEMA, "benchmark_id": BENCHMARK_ID, "objective_id": OBJECTIVE_ID,
        "availability": "unavailable", "execution_status": "not_run", "ran": False, "qualified": False,
        "authoritative": False, "promotion_eligible": False, "metrics_omitted": True, "reason_code": REASON_CODE,
        "repository_binding": {**snapshot.identity, "clean": True, "observed_before_and_after": True},
        "manifest_binding": {"relative_path": MANIFEST_RELATIVE_PATH, "schema": MANIFEST_SCHEMA, "raw_sha256": snapshot.manifest_sha256},
        "matrix_binding": {"relative_path": MATRIX_RELATIVE_PATH, "schema": MATRIX_SCHEMA, "sha256": snapshot.matrix_sha256},
        "source_binding": {"source_sha256": snapshot.source_sha256, "observed_before_and_after": True},
        "capacity_preflight": capacity_preflight(), "future_required_comparison": _COMPARISON,
        "token_gates": TOKEN_GATES, "zero_tolerance_gates": list(ZERO_TOLERANCE_GATES), "nonclaims": list(NONCLAIMS),
    }
    result["content_sha256"] = result_content_sha256(result)
    return result


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(_mapping(result, "result"))
    payload.pop("content_sha256", None)
    return _sha256(_canonical_bytes(payload))


def _validate_result(result: Mapping[str, Any], snapshot: _Snapshot) -> None:
    expected = _result(snapshot)
    _exact_structure(result, expected, "result")
    if result.get("content_sha256") != result_content_sha256(result):
        raise TokenBenchmarkError("result content digest does not bind the result")


def run_benchmark(repository: Path | str, identities: Any = None, *, manifest_path: Path | str | None = None, matrix_path: Path | str | None = None) -> dict[str, Any]:
    # This must stay first: unavailable operation neither reads nor renders caller identities.
    preflight = capacity_preflight()
    if preflight["availability"] != "unavailable":
        raise TokenBenchmarkError("unexpected token preflight state")
    root, manifest, matrix = _bound_repository_and_recipe(repository, manifest_path, matrix_path)
    snapshot = _snapshot(root, manifest, matrix)
    return _result(snapshot)


def validate_result(result: Mapping[str, Any], *, repository: Path | str, manifest_path: Path | str | None = None, matrix_path: Path | str | None = None, current_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    if capacity_preflight()["availability"] != "unavailable":
        raise TokenBenchmarkError("unexpected token preflight state")
    root, manifest, matrix = _bound_repository_and_recipe(repository, manifest_path, matrix_path)
    snapshot = _snapshot(root, manifest, matrix)
    if current_identity is not None and dict(current_identity) != snapshot.identity:
        raise TokenBenchmarkError("supplied current identity does not match the exact current tree", reason_code="repository_binding_changed")
    _validate_result(result, snapshot)
    return dict(result)


def _invalid_diagnostic(reason_code: str) -> dict[str, str]:
    messages = {
        "missing_required_argument": "repository is required",
        "unsafe_source_path": "a required source path is unavailable or unsafe",
    }
    return {"schema": ERROR_SCHEMA, "execution_status": "invalid", "error_code": reason_code, "message": messages.get(reason_code, "benchmark input or exact-tree binding is invalid")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repository")
    parser.add_argument("--manifest")
    parser.add_argument("--matrix")
    parser.add_argument("--identities")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repository is None:
        print(json.dumps(_invalid_diagnostic("missing_required_argument"), sort_keys=True))
        return 2
    try:
        result = run_benchmark(args.repository, manifest_path=args.manifest, matrix_path=args.matrix)
    except TokenBenchmarkError as exc:
        print(json.dumps(_invalid_diagnostic(exc.reason_code), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
