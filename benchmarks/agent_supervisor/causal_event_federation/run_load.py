#!/usr/bin/env python3
"""Fail-closed CASF 256-agent bounded-load benchmark contract.

The current tree has one supervisor, one registered logical agent, and one
concurrent subagent slot.  The required 12/256/64 live profile is therefore
unavailable.  Capacity is checked before caller-controlled paths, Git,
identity material, workload allocation, process launch, state access, or any
external effect.  The current path emits only a content-addressed not-run
artifact bound to a stable clean source tree; it contains no benchmark metrics.

A dormant admitted execution boundary defines the future live contract.  It
requires authenticated typed Quack authority and rejects direct database or
DuckLake scheduling authority.  Nothing in the current-capacity path can reach
that boundary or manufacture a live observation.
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

MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/load-benchmark-manifest@1"
)
RESULT_SCHEMA = "casf/load-benchmark@1"
ERROR_SCHEMA = "casf/load-benchmark-error@1"
CAPACITY_PREFLIGHT_SCHEMA = "casf/load-capacity-preflight@1"
LIVE_ATTESTATION_SCHEMA = "casf/load-live-capacity-attestation@1"
LIVE_OBSERVATION_SCHEMA = "casf/load-live-observation@1"
ADMITTED_EXECUTION_INTERFACE = "CASFLoadAdmittedExecution@1"
BENCHMARK_ID = "casf-load-256-agent-bounded-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
OBJECTIVE_ID = "CASF-040"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
MEASUREMENT_SCOPE = (
    "twelve_independent_supervisor_processes_256_registered_agents_required_live_profile"
)
REASON_CODE = "qualified_256_agent_bounded_load_live_capacity_not_admitted"

REQUIRED_SUPERVISOR_PROCESSES = 12
REQUIRED_REGISTERED_LOGICAL_AGENTS = 256
MAXIMUM_CONCURRENT_SUBAGENTS = 64
MINIMUM_BOUNDED_TASKS = 1_000
MINIMUM_EVENT_DELIVERIES_WITH_REPLAY = 100_000

CURRENT_SUPERVISOR_PROCESSES = 1
CURRENT_REGISTERED_LOGICAL_AGENTS = 1
CURRENT_MAXIMUM_CONCURRENT_SUBAGENTS = 1

SOURCE_RELATIVE_PATHS = (
    "benchmarks/agent_supervisor/causal_event_federation/load_manifest.json",
    "benchmarks/agent_supervisor/causal_event_federation/run_load.py",
)
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
ZERO_TOLERANCE_GATES = (
    "lost_deliveries",
    "duplicate_committed_effects",
    "stale_fence_completion",
    "unauthorized_creation",
    "tenant_leakage",
    "agent_sql",
    "secret_leaks",
    "causal_notification_loss",
    "stale_abstraction_suppression",
    "model_created_authority_or_completion",
    "ducklake_authority_promotion",
    "reduced_assurance",
)
RESULT_STORAGE = (
    "Emit one canonical content-addressed unavailable/not-run artifact bound to the "
    "twice-observed clean repository commit and tree, exact manifest, matrix, and "
    "source bytes; it grants no scheduling, completion, acceptance, release, or "
    "promotion authority."
)
NONCLAIMS = (
    "This frozen recipe and its unavailable artifact are not benchmark measurements.",
    "No twelve-supervisor, 256-agent, 64-slot, task, delivery, or replay workload "
    "ran or qualified.",
    "The 1000-task and 100000-delivery requirements are future acceptance thresholds, "
    "not observed metric values.",
    "An in-process simulation, object graph, synthetic agent count, or caller-supplied "
    "telemetry cannot qualify this benchmark.",
    "No direct database access, file fallback, network access, credential use, provider "
    "call, or DuckLake scheduling authority occurred.",
    "Any future live authority requires an authenticated typed Quack admission and can "
    "never come from direct DuckDB access or DuckLake.",
    "Unavailable and not-run evidence contains no metrics and cannot establish "
    "completion, release, acceptance, promotion, or safe load behavior.",
    "This benchmark does not qualify token efficiency, multihost behavior, production "
    "behavior, or any capability outside its separately admitted live profile.",
)

_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_MATRIX_BYTES = 128 * 1024
_MAX_RUNNER_BYTES = 2 * 1024 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}\Z")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})\Z")
_SECRET_VALUE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:api[_-]?key|access[_-]?token|token|password|passwd|secret)\s*[:=]\s*\S+|"
    r"(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{8,})"
)

_EXECUTION_CONTRACT = {
    "measurement_scope": MEASUREMENT_SCOPE,
    "required_supervisor_processes": REQUIRED_SUPERVISOR_PROCESSES,
    "registered_logical_agents": REQUIRED_REGISTERED_LOGICAL_AGENTS,
    "maximum_concurrent_subagents": MAXIMUM_CONCURRENT_SUBAGENTS,
    "minimum_bounded_tasks": MINIMUM_BOUNDED_TASKS,
    "minimum_event_deliveries_with_replay": MINIMUM_EVENT_DELIVERIES_WITH_REPLAY,
    "subprocess_budget": REQUIRED_SUPERVISOR_PROCESSES,
    "real_independent_processes_required": True,
    "in_process_simulation_qualifies": False,
    "admitted_execution_interface": ADMITTED_EXECUTION_INTERFACE,
    "admitted_execution_available": False,
    "network_permitted": False,
    "authenticated_typed_quack_required": True,
    "direct_database_access_permitted": False,
    "ducklake_scheduling_authority_permitted": False,
    "ducklake_projection_authoritative": False,
    "launch_permitted": False,
}
_FROZEN_CAPACITY_PREFLIGHT = {
    "schema": CAPACITY_PREFLIGHT_SCHEMA,
    "current_supervisor_processes": CURRENT_SUPERVISOR_PROCESSES,
    "current_registered_logical_agents": CURRENT_REGISTERED_LOGICAL_AGENTS,
    "current_maximum_concurrent_subagents": CURRENT_MAXIMUM_CONCURRENT_SUBAGENTS,
    "required_supervisor_processes": REQUIRED_SUPERVISOR_PROCESSES,
    "required_registered_logical_agents": REQUIRED_REGISTERED_LOGICAL_AGENTS,
    "required_maximum_concurrent_subagents": MAXIMUM_CONCURRENT_SUBAGENTS,
    "authenticated_typed_quack_live_capacity": False,
    "availability": "unavailable",
    "reason_code": REASON_CODE,
}
_LIVE_CAPABILITY_UNAVAILABLE = {
    "availability": "unavailable",
    "execution_status": "not_run",
    "ran": False,
    "qualified": False,
    "reason_code": REASON_CODE,
    "required_attestation": LIVE_ATTESTATION_SCHEMA,
    "required_evidence": (
        "current_generation_accepted_gate_current_fences_and_live_"
        "host_provider_proof_merge_storage_telemetry"
    ),
    "metrics_omitted": True,
}
_FUTURE_REQUIRED_PROFILE = {
    "supervisor_processes": REQUIRED_SUPERVISOR_PROCESSES,
    "registered_logical_agents": REQUIRED_REGISTERED_LOGICAL_AGENTS,
    "maximum_concurrent_subagents": MAXIMUM_CONCURRENT_SUBAGENTS,
    "minimum_bounded_tasks": MINIMUM_BOUNDED_TASKS,
    "minimum_event_deliveries_with_replay": MINIMUM_EVENT_DELIVERIES_WITH_REPLAY,
    "replay_required": True,
    "zero_lost_deliveries_required": True,
    "zero_duplicate_committed_effects_required": True,
    "state_authority": "authenticated_typed_quack",
    "direct_database_access_permitted": False,
    "ducklake_scheduling_authority_permitted": False,
}
_EXPECTED_MANIFEST = {
    "schema": MANIFEST_SCHEMA,
    "benchmark_id": BENCHMARK_ID,
    "program_id": PROGRAM_ID,
    "objective_id": OBJECTIVE_ID,
    "frozen": True,
    "state": "capability_unavailable",
    "authoritative": False,
    "promotion_eligible": False,
    "matrix_binding": {
        "relative_path": MATRIX_RELATIVE_PATH,
        "schema": MATRIX_SCHEMA,
        "sha256": MATRIX_SHA256,
    },
    "execution": _EXECUTION_CONTRACT,
    "capacity_preflight": _FROZEN_CAPACITY_PREFLIGHT,
    "zero_tolerance_gates": list(ZERO_TOLERANCE_GATES),
    "future_identity_requirements": list(REQUIRED_IDENTITIES),
    "source_modules": list(SOURCE_RELATIVE_PATHS),
    "live_capability": _LIVE_CAPABILITY_UNAVAILABLE,
    "result_storage": RESULT_STORAGE,
    "nonclaims": list(NONCLAIMS),
}


class LoadBenchmarkError(ValueError):
    """A closed invalid-input or source-binding diagnostic."""

    def __init__(self, message: str, *, reason_code: str = "invalid_contract") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class LoadCapabilityUnavailable(LoadBenchmarkError):
    """Raised only by callers that explicitly demand unavailable live capacity."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LoadBenchmarkError("duplicate JSON key", reason_code="duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise LoadBenchmarkError(
        f"non-finite JSON number {value!r} is prohibited",
        reason_code="invalid_json",
    )


def _loads_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise LoadBenchmarkError(
            f"{name} is not valid JSON",
            reason_code="invalid_json",
        ) from exc


def _read_bounded_regular_bytes(
    path: Path | str,
    *,
    name: str,
    maximum_bytes: int,
) -> bytes:
    candidate = Path(path)
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise LoadBenchmarkError(
            f"{name} is unavailable",
            reason_code="source_unavailable",
        ) from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise LoadBenchmarkError(
            f"{name} must be a regular non-symlink file",
            reason_code="unsafe_source_path",
        )
    if initial.st_size > maximum_bytes:
        raise LoadBenchmarkError(
            f"{name} exceeds its byte limit",
            reason_code="source_too_large",
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise LoadBenchmarkError(
                f"{name} is not a bounded regular file",
                reason_code="unsafe_source_path",
            )
        identity_fields = ("st_dev", "st_ino", "st_mode")
        if any(getattr(initial, field) != getattr(before, field) for field in identity_fields):
            raise LoadBenchmarkError(
                f"{name} changed before its bounded read",
                reason_code="source_changed",
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except LoadBenchmarkError:
        raise
    except OSError as exc:
        raise LoadBenchmarkError(
            f"{name} could not be read safely",
            reason_code="source_unavailable",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if len(payload) > maximum_bytes:
        raise LoadBenchmarkError(
            f"{name} exceeds its byte limit",
            reason_code="source_too_large",
        )
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise LoadBenchmarkError(
            f"{name} changed during its bounded read",
            reason_code="source_changed",
        )
    return payload


def _decode_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise LoadBenchmarkError(
            f"{name} is not UTF-8",
            reason_code="invalid_json",
        ) from exc
    decoded = _loads_json(text, name=name)
    if not isinstance(decoded, dict):
        raise LoadBenchmarkError(
            f"{name} must contain a JSON object",
            reason_code="invalid_json",
        )
    return decoded


def _read_object(path: Path | str) -> dict[str, Any]:
    raw = _read_bounded_regular_bytes(
        path,
        name="JSON input",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    return _decode_json_object(raw, name="JSON input")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LoadBenchmarkError(f"{name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LoadBenchmarkError(
            f"{name} has a closed schema",
            reason_code="closed_schema_violation",
        )


def _require_exact_structure(value: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        candidate = _require_mapping(value, name)
        _require_exact_keys(candidate, set(expected), name)
        for key, expected_value in expected.items():
            _require_exact_structure(candidate[key], expected_value, f"{name}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise LoadBenchmarkError(f"{name} must be the exact frozen sequence")
        for index, (candidate, expected_value) in enumerate(zip(value, expected, strict=True)):
            _require_exact_structure(candidate, expected_value, f"{name}[{index}]")
        return
    if type(value) is not type(expected) or value != expected:
        raise LoadBenchmarkError(
            f"{name} differs from the exact frozen value",
            reason_code="frozen_contract_changed",
        )


def _require_text(value: Any, name: str, *, maximum_bytes: int = 16_384) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="strict")) > maximum_bytes
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise LoadBenchmarkError(f"{name} must be bounded exact text")
    return value


def _require_token(value: Any, name: str) -> str:
    text = _require_text(value, name, maximum_bytes=512)
    if _SECRET_VALUE.search(text):
        raise LoadBenchmarkError(
            f"{name} contains credential-shaped material",
            reason_code="secret_shaped_input",
        )
    if _TOKEN.fullmatch(text) is None:
        raise LoadBenchmarkError(f"{name} must be a compact identity")
    return text


def _require_content_ref(value: Any, name: str) -> str:
    text = _require_token(value, name)
    if _CONTENT_REF.fullmatch(text) is None:
        raise LoadBenchmarkError(f"{name} must be a content-addressed reference")
    return text


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name, maximum_bytes=64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise LoadBenchmarkError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require_git_oid(value: Any, name: str) -> str:
    text = _require_text(value, name, maximum_bytes=40)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise LoadBenchmarkError(f"{name} must be a full lowercase Git object id")
    return text


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LoadBenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return _object_sha256(manifest)


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(_require_mapping(result, "result"))
    payload.pop("content_sha256", None)
    return _object_sha256(payload)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the complete manifest with exact values and exact runtime types."""

    _require_exact_structure(manifest, _EXPECTED_MANIFEST, "manifest")


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path) if path is not None else Path(__file__).with_name("load_manifest.json")
    manifest = _read_object(resolved)
    validate_manifest(manifest)
    return manifest


def _validate_matrix_bytes(raw: bytes) -> None:
    if _raw_sha256(raw) != MATRIX_SHA256:
        raise LoadBenchmarkError(
            "frozen benchmark matrix content is stale or changed",
            reason_code="matrix_binding_changed",
        )
    try:
        first_line = raw.decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError) as exc:
        raise LoadBenchmarkError(
            "frozen benchmark matrix is missing its schema",
            reason_code="matrix_binding_changed",
        ) from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise LoadBenchmarkError(
            "frozen benchmark matrix schema has changed",
            reason_code="matrix_binding_changed",
        )


def validate_matrix_binding(
    manifest: Mapping[str, Any], matrix_path: Path | str | None = None
) -> dict[str, str]:
    validate_manifest(manifest)
    resolved = (
        Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml")
    )
    raw = _read_bounded_regular_bytes(
        resolved,
        name="benchmark matrix",
        maximum_bytes=_MAX_MATRIX_BYTES,
    )
    _validate_matrix_bytes(raw)
    return dict(manifest["matrix_binding"])


def _canonical_existing_path(path: Path | str, *, name: str) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(path)))
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise LoadBenchmarkError(
            f"{name} is unavailable",
            reason_code="unsafe_source_path",
        ) from None
    if lexical != resolved:
        raise LoadBenchmarkError(
            f"{name} may not traverse a symlink",
            reason_code="unsafe_source_path",
        )
    return resolved


def _bound_repository_and_recipe(
    repository: Path | str,
    manifest_path: Path | str | None,
    matrix_path: Path | str | None,
) -> tuple[Path, Path, Path]:
    root = _canonical_existing_path(repository, name="repository")
    runner_path = _canonical_existing_path(__file__, name="benchmark runner")
    runner_root = runner_path.parents[3]
    if root != runner_root:
        raise LoadBenchmarkError(
            "repository must contain this exact benchmark runner",
            reason_code="repository_binding_changed",
        )

    expected_manifest = root / SOURCE_RELATIVE_PATHS[0]
    expected_matrix = root / MATRIX_RELATIVE_PATH
    supplied_manifest = expected_manifest if manifest_path is None else manifest_path
    supplied_matrix = expected_matrix if matrix_path is None else matrix_path
    resolved_manifest = _canonical_existing_path(supplied_manifest, name="benchmark manifest")
    resolved_matrix = _canonical_existing_path(supplied_matrix, name="benchmark matrix")
    if resolved_manifest != expected_manifest or resolved_matrix != expected_matrix:
        raise LoadBenchmarkError(
            "manifest and matrix must be exact measured-tree files",
            reason_code="repository_binding_changed",
        )
    return root, resolved_manifest, resolved_matrix


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise LoadBenchmarkError(
            "cannot establish repository identity",
            reason_code="repository_identity_unavailable",
        ) from exc
    return result.stdout.strip()


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = _canonical_existing_path(repository, name="repository")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise LoadBenchmarkError(
            "repository is dirty; exact current-tree evidence is unavailable",
            reason_code="repository_dirty",
        )
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    return {
        "repository_commit": _require_git_oid(commit, "repository commit"),
        "repository_tree": _require_git_oid(tree, "repository tree"),
    }


class _SourceSnapshot(NamedTuple):
    repository_commit: str
    repository_tree: str
    manifest: dict[str, Any]
    manifest_raw_sha256: str
    matrix_raw_sha256: str
    source_sha256: dict[str, str]


def _stable_source_snapshot(
    repository: Path,
    manifest_path: Path,
    matrix_path: Path,
) -> _SourceSnapshot:
    before = repository_identity(repository)
    source_paths = {
        SOURCE_RELATIVE_PATHS[0]: manifest_path,
        SOURCE_RELATIVE_PATHS[1]: repository / SOURCE_RELATIVE_PATHS[1],
    }

    first_source = {
        relative_path: _read_bounded_regular_bytes(
            path,
            name=f"measured source {index}",
            maximum_bytes=(
                _MAX_MANIFEST_BYTES if relative_path.endswith(".json") else _MAX_RUNNER_BYTES
            ),
        )
        for index, (relative_path, path) in enumerate(source_paths.items())
    }
    first_matrix = _read_bounded_regular_bytes(
        matrix_path,
        name="benchmark matrix",
        maximum_bytes=_MAX_MATRIX_BYTES,
    )
    manifest = _decode_json_object(first_source[SOURCE_RELATIVE_PATHS[0]], name="manifest")
    validate_manifest(manifest)
    _validate_matrix_bytes(first_matrix)

    second_source = {
        relative_path: _read_bounded_regular_bytes(
            path,
            name=f"measured source {index}",
            maximum_bytes=(
                _MAX_MANIFEST_BYTES if relative_path.endswith(".json") else _MAX_RUNNER_BYTES
            ),
        )
        for index, (relative_path, path) in enumerate(source_paths.items())
    }
    second_matrix = _read_bounded_regular_bytes(
        matrix_path,
        name="benchmark matrix",
        maximum_bytes=_MAX_MATRIX_BYTES,
    )
    after = repository_identity(repository)
    if before != after or first_source != second_source or first_matrix != second_matrix:
        raise LoadBenchmarkError(
            "repository identity or measured source changed during evidence collection",
            reason_code="source_changed",
        )

    return _SourceSnapshot(
        repository_commit=before["repository_commit"],
        repository_tree=before["repository_tree"],
        manifest=manifest,
        manifest_raw_sha256=_raw_sha256(first_source[SOURCE_RELATIVE_PATHS[0]]),
        matrix_raw_sha256=_raw_sha256(first_matrix),
        source_sha256={
            relative_path: _raw_sha256(raw) for relative_path, raw in first_source.items()
        },
    )


def _validate_identities(value: Any) -> dict[str, Any]:
    identities = _require_mapping(value, "benchmark identities")
    _require_exact_keys(identities, set(REQUIRED_IDENTITIES), "benchmark identities")
    checked = dict(identities)
    checked["repository_commit"] = _require_git_oid(
        checked["repository_commit"], "identities.repository_commit"
    )
    checked["repository_tree"] = _require_git_oid(
        checked["repository_tree"], "identities.repository_tree"
    )
    for key in ("control_plane_generation", "assignment_revision", "fencing_epoch"):
        checked[key] = _require_int(checked[key], f"identities.{key}", minimum=1)
    for key in REQUIRED_IDENTITIES:
        if key not in {
            "repository_commit",
            "repository_tree",
            "control_plane_generation",
            "assignment_revision",
            "fencing_epoch",
        }:
            checked[key] = _require_token(checked[key], f"identities.{key}")
    if checked["task_id"] != OBJECTIVE_ID:
        raise LoadBenchmarkError("identities.task_id must bind CASF-040")
    return checked


def capacity_preflight() -> dict[str, Any]:
    """Return the frozen local capacity without inspecting caller-controlled input."""

    return dict(_FROZEN_CAPACITY_PREFLIGHT)


def live_capability(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    checked = load_manifest() if manifest is None else dict(manifest)
    validate_manifest(checked)
    return dict(checked["live_capability"])


def require_live_capability(manifest: Mapping[str, Any]) -> NoReturn:
    validate_manifest(manifest)
    raise LoadCapabilityUnavailable(
        "qualified 256-agent bounded-load live capacity is unavailable",
        reason_code=REASON_CODE,
    )


class LoadAdmission(NamedTuple):
    """Future typed-Quack admission; the frozen current preflight cannot create one."""

    schema: str
    supervisor_processes: int
    registered_logical_agents: int
    maximum_concurrent_subagents: int
    minimum_bounded_tasks: int
    minimum_event_deliveries_with_replay: int
    replay_required: bool
    state_authority: str
    quack_receipt_ref: str
    direct_database_access_permitted: bool
    ducklake_scheduling_authority_permitted: bool


class AdmittedLoadExecutor(Protocol):
    """Effect boundary implemented only by a future authenticated live harness."""

    interface: str

    def execute_load(
        self,
        *,
        admission: LoadAdmission,
        required_profile: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _validate_load_admission(admission: LoadAdmission) -> None:
    if type(admission) is not LoadAdmission:
        raise LoadBenchmarkError("load admission must use the closed typed contract")
    expected = {
        "schema": LIVE_ATTESTATION_SCHEMA,
        "supervisor_processes": REQUIRED_SUPERVISOR_PROCESSES,
        "registered_logical_agents": REQUIRED_REGISTERED_LOGICAL_AGENTS,
        "maximum_concurrent_subagents": MAXIMUM_CONCURRENT_SUBAGENTS,
        "minimum_bounded_tasks": MINIMUM_BOUNDED_TASKS,
        "minimum_event_deliveries_with_replay": MINIMUM_EVENT_DELIVERIES_WITH_REPLAY,
        "replay_required": True,
        "state_authority": "authenticated_typed_quack",
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
    }
    actual = {
        "schema": admission.schema,
        "supervisor_processes": admission.supervisor_processes,
        "registered_logical_agents": admission.registered_logical_agents,
        "maximum_concurrent_subagents": admission.maximum_concurrent_subagents,
        "minimum_bounded_tasks": admission.minimum_bounded_tasks,
        "minimum_event_deliveries_with_replay": (admission.minimum_event_deliveries_with_replay),
        "replay_required": admission.replay_required,
        "state_authority": admission.state_authority,
        "direct_database_access_permitted": admission.direct_database_access_permitted,
        "ducklake_scheduling_authority_permitted": (
            admission.ducklake_scheduling_authority_permitted
        ),
    }
    _require_exact_structure(actual, expected, "live load admission")
    _require_content_ref(admission.quack_receipt_ref, "live load admission Quack receipt")


def _validate_live_observation(value: Any, admission: LoadAdmission) -> dict[str, Any]:
    observation = _require_mapping(value, "live load observation")
    expected_keys = {
        "schema",
        "supervisor_process_ids",
        "logical_agent_ids",
        "maximum_active_subagents",
        "bounded_tasks_completed",
        "event_deliveries_observed",
        "replay_deliveries_observed",
        "event_delivery_count_includes_replay",
        "lost_deliveries",
        "duplicate_committed_effects",
        "zero_tolerance_gate_failures",
        "state_transport",
        "quack_receipt_ref",
        "direct_database_access_used",
        "ducklake_scheduling_authority_used",
    }
    _require_exact_keys(observation, expected_keys, "live load observation")
    if observation["schema"] != LIVE_OBSERVATION_SCHEMA or type(observation["schema"]) is not str:
        raise LoadBenchmarkError("live load observation schema changed")

    process_ids = observation["supervisor_process_ids"]
    if (
        not isinstance(process_ids, list)
        or len(process_ids) != REQUIRED_SUPERVISOR_PROCESSES
        or any(type(pid) is not int or pid < 1 for pid in process_ids)
        or len(set(process_ids)) != REQUIRED_SUPERVISOR_PROCESSES
    ):
        raise LoadBenchmarkError("live load requires exactly twelve independent process IDs")

    agent_ids = observation["logical_agent_ids"]
    if not isinstance(agent_ids, list) or len(agent_ids) != REQUIRED_REGISTERED_LOGICAL_AGENTS:
        raise LoadBenchmarkError("live load requires exactly 256 logical agent IDs")
    checked_agent_ids = [
        _require_token(agent_id, f"live logical agent {index}")
        for index, agent_id in enumerate(agent_ids)
    ]
    if len(set(checked_agent_ids)) != REQUIRED_REGISTERED_LOGICAL_AGENTS:
        raise LoadBenchmarkError("live logical agent IDs must be unique")

    maximum_active = _require_int(
        observation["maximum_active_subagents"],
        "maximum active subagents",
        minimum=1,
    )
    if maximum_active > MAXIMUM_CONCURRENT_SUBAGENTS:
        raise LoadBenchmarkError("live load exceeded the 64-subagent concurrency bound")
    if (
        _require_int(observation["bounded_tasks_completed"], "bounded tasks", minimum=0)
        < MINIMUM_BOUNDED_TASKS
    ):
        raise LoadBenchmarkError("live load completed fewer than 1000 bounded tasks")
    deliveries = _require_int(
        observation["event_deliveries_observed"],
        "event deliveries",
        minimum=0,
    )
    if deliveries < MINIMUM_EVENT_DELIVERIES_WITH_REPLAY:
        raise LoadBenchmarkError("live load observed fewer than 100000 event deliveries")
    replay_deliveries = _require_int(
        observation["replay_deliveries_observed"],
        "replay deliveries",
        minimum=1,
    )
    if replay_deliveries > deliveries:
        raise LoadBenchmarkError("replay deliveries cannot exceed total deliveries")
    if observation["event_delivery_count_includes_replay"] is not True:
        raise LoadBenchmarkError("event delivery count must explicitly include replay")
    if type(observation["event_delivery_count_includes_replay"]) is not bool:
        raise LoadBenchmarkError("replay inclusion must be an exact boolean")

    lost = _require_int(observation["lost_deliveries"], "lost deliveries", minimum=0)
    duplicate_effects = _require_int(
        observation["duplicate_committed_effects"],
        "duplicate committed effects",
        minimum=0,
    )
    if lost != 0 or duplicate_effects != 0:
        raise LoadBenchmarkError("lost deliveries and duplicate committed effects must be zero")
    failures = _require_mapping(
        observation["zero_tolerance_gate_failures"],
        "zero-tolerance gate failures",
    )
    _require_exact_keys(failures, set(ZERO_TOLERANCE_GATES), "zero-tolerance gate failures")
    for gate in ZERO_TOLERANCE_GATES:
        if _require_int(failures[gate], f"gate failure {gate}", minimum=0) != 0:
            raise LoadBenchmarkError(f"zero-tolerance gate failed: {gate}")
    if failures["lost_deliveries"] != lost:
        raise LoadBenchmarkError("lost-delivery gate and observation disagree")
    if failures["duplicate_committed_effects"] != duplicate_effects:
        raise LoadBenchmarkError("duplicate-effect gate and observation disagree")

    state_transport = _require_token(
        observation["state_transport"],
        "live load state transport",
    )
    if state_transport != "authenticated_typed_quack":
        raise LoadBenchmarkError("live load state transport must be authenticated typed Quack")
    receipt_ref = _require_content_ref(
        observation["quack_receipt_ref"],
        "live load observation Quack receipt",
    )
    if receipt_ref != admission.quack_receipt_ref:
        raise LoadBenchmarkError("live observation does not bind the admitted Quack receipt")
    if (
        observation["direct_database_access_used"] is not False
        or type(observation["direct_database_access_used"]) is not bool
    ):
        raise LoadBenchmarkError("direct database access can never qualify a live load run")
    if (
        observation["ducklake_scheduling_authority_used"] is not False
        or type(observation["ducklake_scheduling_authority_used"]) is not bool
    ):
        raise LoadBenchmarkError("DuckLake can never be live load scheduling authority")
    return dict(observation)


def execute_admitted_load(
    admission: LoadAdmission,
    executor: AdmittedLoadExecutor,
) -> NoReturn:
    """Fence the future effect boundary without ever invoking its executor."""

    if _require_token(executor.interface, "admitted load executor interface") != (
        ADMITTED_EXECUTION_INTERFACE
    ):
        raise LoadBenchmarkError("admitted load executor interface changed")
    _validate_load_admission(admission)
    raise LoadCapabilityUnavailable(
        "admitted load execution is unavailable in this frozen contract",
        reason_code="admitted_execution_unavailable",
    )


def validate_admitted_load_observation(
    admission: LoadAdmission,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the future live observation shape without performing an effect."""

    _validate_load_admission(admission)
    return _validate_live_observation(observation, admission)


def _admitted_execution_boundary(
    executor: AdmittedLoadExecutor,
    *,
    admission: LoadAdmission,
) -> NoReturn:
    """Future live boundary, intentionally unreachable from ``run_benchmark``."""

    return execute_admitted_load(admission, executor)


def _manifest_binding(snapshot: _SourceSnapshot) -> dict[str, str]:
    return {
        "relative_path": SOURCE_RELATIVE_PATHS[0],
        "schema": MANIFEST_SCHEMA,
        "raw_sha256": snapshot.manifest_raw_sha256,
    }


def _matrix_binding(snapshot: _SourceSnapshot) -> dict[str, str]:
    return {
        "relative_path": MATRIX_RELATIVE_PATH,
        "schema": MATRIX_SCHEMA,
        "sha256": snapshot.matrix_raw_sha256,
    }


def _repository_binding(snapshot: _SourceSnapshot) -> dict[str, Any]:
    return {
        "repository_commit": snapshot.repository_commit,
        "repository_tree": snapshot.repository_tree,
        "clean": True,
        "observed_before_and_after": True,
    }


def _source_binding(snapshot: _SourceSnapshot) -> dict[str, Any]:
    return {
        "source_sha256": dict(snapshot.source_sha256),
        "observed_before_and_after": True,
    }


def _build_unavailable_result(
    snapshot: _SourceSnapshot,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "manifest_binding": _manifest_binding(snapshot),
        "matrix_binding": _matrix_binding(snapshot),
        "source_binding": _source_binding(snapshot),
        "repository_binding": _repository_binding(snapshot),
        "content_sha256": "",
        "availability": "unavailable",
        "execution_status": "not_run",
        "ran": False,
        "qualified": False,
        "metrics_omitted": True,
        "authoritative": False,
        "promotion_eligible": False,
        "reason_code": REASON_CODE,
        "measurement_scope": MEASUREMENT_SCOPE,
        "future_required_profile": dict(_FUTURE_REQUIRED_PROFILE),
        "capability_preflight": dict(preflight),
        "nonclaims": list(NONCLAIMS),
    }
    result["content_sha256"] = result_content_sha256(result)
    return result


def _validate_result_against_snapshot(
    result: Mapping[str, Any],
    snapshot: _SourceSnapshot,
) -> dict[str, Any]:
    result = _require_mapping(result, "result")
    expected_keys = {
        "schema",
        "benchmark_id",
        "program_id",
        "objective_id",
        "manifest_binding",
        "matrix_binding",
        "source_binding",
        "repository_binding",
        "content_sha256",
        "availability",
        "execution_status",
        "ran",
        "qualified",
        "metrics_omitted",
        "authoritative",
        "promotion_eligible",
        "reason_code",
        "measurement_scope",
        "future_required_profile",
        "capability_preflight",
        "nonclaims",
    }
    _require_exact_keys(result, expected_keys, "result")
    fixed = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "availability": "unavailable",
        "execution_status": "not_run",
        "ran": False,
        "qualified": False,
        "metrics_omitted": True,
        "authoritative": False,
        "promotion_eligible": False,
        "reason_code": REASON_CODE,
        "measurement_scope": MEASUREMENT_SCOPE,
        "manifest_binding": _manifest_binding(snapshot),
        "matrix_binding": _matrix_binding(snapshot),
        "source_binding": _source_binding(snapshot),
        "repository_binding": _repository_binding(snapshot),
        "future_required_profile": _FUTURE_REQUIRED_PROFILE,
        "capability_preflight": _FROZEN_CAPACITY_PREFLIGHT,
        "nonclaims": list(NONCLAIMS),
    }
    for key, expected in fixed.items():
        _require_exact_structure(result[key], expected, f"result.{key}")
    content_sha256 = _require_sha256(result["content_sha256"], "result content SHA-256")
    if content_sha256 != result_content_sha256(result):
        raise LoadBenchmarkError(
            "result content address is invalid",
            reason_code="result_content_changed",
        )
    prohibited = {"metrics", "metric_values", "observations", "result_values", "error"}
    if prohibited.intersection(result):
        raise LoadBenchmarkError("not-run result contains prohibited measurement fields")
    return dict(result)


def run_benchmark(
    *,
    repository: Path | str,
    identities: Any = None,
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> dict[str, Any]:
    """Emit stable not-run evidence; never inspect identities at current capacity."""

    preflight = capacity_preflight()
    _require_exact_structure(preflight, _FROZEN_CAPACITY_PREFLIGHT, "capacity preflight")
    if preflight["availability"] != "unavailable":
        raise LoadBenchmarkError(
            "frozen current capacity unexpectedly changed",
            reason_code="capacity_contract_changed",
        )

    # Identity material is intentionally unread while capacity is unavailable.
    del identities
    root, resolved_manifest, resolved_matrix = _bound_repository_and_recipe(
        repository,
        manifest_path,
        matrix_path,
    )
    snapshot = _stable_source_snapshot(root, resolved_manifest, resolved_matrix)
    result = _build_unavailable_result(snapshot, preflight)
    return _validate_result_against_snapshot(result, snapshot)


def validate_result(
    result: Mapping[str, Any],
    *,
    repository: Path | str,
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> dict[str, Any]:
    """Replay an unavailable artifact against the exact current clean source tree."""

    preflight = capacity_preflight()
    _require_exact_structure(preflight, _FROZEN_CAPACITY_PREFLIGHT, "capacity preflight")
    root, resolved_manifest, resolved_matrix = _bound_repository_and_recipe(
        repository,
        manifest_path,
        matrix_path,
    )
    snapshot = _stable_source_snapshot(root, resolved_manifest, resolved_matrix)
    return _validate_result_against_snapshot(result, snapshot)


def _invalid_diagnostic(reason_code: str) -> dict[str, str]:
    messages = {
        "missing_required_argument": "repository is required",
        "unsafe_source_path": "a required source path is unavailable or unsafe",
        "repository_binding_changed": "repository does not contain the canonical load recipe",
        "repository_identity_unavailable": "repository identity is unavailable",
        "repository_dirty": "repository is not clean",
        "source_unavailable": "a required source is unavailable",
        "source_too_large": "a required source exceeds its byte bound",
        "source_changed": "repository source changed during observation",
        "matrix_binding_changed": "benchmark matrix binding changed",
        "frozen_contract_changed": "frozen load contract changed",
        "invalid_contract": "load benchmark invocation or contract is invalid",
    }
    stable_code = reason_code if reason_code in messages else "invalid_contract"
    return {
        "schema": ERROR_SCHEMA,
        "execution_status": "invalid",
        "error_code": stable_code,
        "message": messages[stable_code],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, help="clean measured repository")
    parser.add_argument(
        "--identities",
        type=Path,
        help="reserved for future typed live admission; never read while unavailable",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("load_manifest.json"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("matrix.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repository is None:
        print(json.dumps(_invalid_diagnostic("missing_required_argument"), sort_keys=True))
        return 2
    try:
        result = run_benchmark(
            repository=args.repository,
            identities=args.identities,
            manifest_path=args.manifest,
            matrix_path=args.matrix,
        )
    except LoadBenchmarkError as exc:
        print(json.dumps(_invalid_diagnostic(exc.reason_code), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
