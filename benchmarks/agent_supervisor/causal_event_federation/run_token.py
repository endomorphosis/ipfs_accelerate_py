#!/usr/bin/env python3
"""Fail-closed, current-tree token-efficiency recipe for CASF-041.

The checked-in scheduler admits only the literal 1/1/1 bootstrap profile, not
the real twelve-supervisor candidate required by the frozen matrix. The public
runner therefore emits only a deterministic unavailable/not-run artifact. A
dormant validator defines the future evidence boundary in raw receipt-backed
populations; it cannot execute work, create authority, or qualify a result.
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
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/token-benchmark-manifest@1"
)
RESULT_SCHEMA = "casf/token-benchmark@1"
ERROR_SCHEMA = "casf/token-benchmark-error@1"
CAPACITY_PREFLIGHT_SCHEMA = "casf/token-capacity-preflight@1"
LIVE_ATTESTATION_SCHEMA = "casf/token-live-capacity-attestation@2"
LIVE_OBSERVATION_SCHEMA = "casf/token-live-observation@2"
LIVE_ARM_SCHEMA = "casf/token-live-arm-observation@1"
ADMITTED_EXECUTION_INTERFACE = "CASFTokenAdmittedExecution@1"
BENCHMARK_ID = "casf-token-cross-supervisor-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
OBJECTIVE_ID = "CASF-041"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.causal-event-supervisor-federation.scheduler_config@1"
)
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MANIFEST_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/token_manifest.json"
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_token.py"
SCHEDULER_RELATIVE_PATH = "config/agent_supervisor_causal_event_federation_scheduler.json"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
SCHEDULER_SHA256 = "708f8b00fbd5343fc7e1ca9eaf668e553992ebe89f9d6e8cbe2f1bdb58cf426f"
MEASUREMENT_SCOPE = (
    "one_baseline_and_twelve_independent_supervisor_processes_qualified_live_token_comparison"
)
REASON_CODE = "qualified_cross_supervisor_token_live_capacity_not_admitted"
LIVE_VERIFIER_REASON_CODE = "registered_state_owner_receipt_verifier_unavailable"
BASELINE_SUPERVISOR_PROCESSES = 1
CANDIDATE_SUPERVISOR_PROCESSES = 12
TARGET_REGISTERED_LOGICAL_AGENTS = 256
TARGET_MAXIMUM_ACTIVE_SUBAGENTS = 64
SOURCE_RELATIVE_PATHS = (
    MANIFEST_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    SCHEDULER_RELATIVE_PATH,
)
MEASURED_RELATIVE_PATHS = (
    MANIFEST_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    MATRIX_RELATIVE_PATH,
    SCHEDULER_RELATIVE_PATH,
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
    "task_id",
    "attempt_id",
    "worktree_id",
    "assignment_revision",
    "fencing_epoch",
    "baseline_supervisor_process_birth_refs",
    "candidate_supervisor_process_birth_refs",
)
SAME_POPULATION_BINDINGS = (
    "host_ref",
    "workload_ref",
    "task_population_ref",
    "criteria_ref",
    "provider_ref",
    "model_ref",
    "tokenizer_ref",
    "budget_ref",
    "tests_ref",
    "proofs_ref",
    "assurance_ref",
)
TOKEN_GATES = {
    "minimum_repeated_context_token_reduction_percent": 50,
    "minimum_input_tokens_per_accepted_criterion_reduction_percent": 40,
    "minimum_duplicate_model_call_reduction_percent": 60,
    "minimum_eligible_semantic_capsule_reuse_percent": 70,
    "minimum_complete_board_scan_reduction_percent": 80,
}
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
NONCLAIMS = (
    "This frozen recipe and its unavailable artifact are not benchmark measurements.",
    "Neither the one-supervisor baseline arm nor the twelve-supervisor candidate arm ran or qualified.",
    "The token-reduction and semantic-capsule-reuse thresholds are future acceptance thresholds, not observed metric values.",
    "No caller-authored percentage, synthetic population, in-process simulation, object graph, or provider/model substitution can qualify this benchmark.",
    "No direct database access, file fallback, network access, credential use, provider call, or DuckLake scheduling authority occurred.",
    "Any future live authority requires an authenticated typed Quack admission and can never come from direct DuckDB access or DuckLake.",
    "Unavailable and not-run evidence contains no metrics and cannot establish completion, release, acceptance, promotion, or token efficiency.",
    "Git porcelain is not treated as proof of a clean tree; every measured path must independently equal its tracked HEAD blob.",
    "This benchmark does not qualify load, multihost behavior, production behavior, or any capability outside its separately admitted live profile.",
)
RESULT_STORAGE = (
    "Emit one canonical content-addressed unavailable/not-run artifact bound to the "
    "twice-observed repository commit and tree and exact tracked HEAD blobs for the "
    "manifest, matrix, runner, and authoritative scheduler; it grants no scheduling, completion, "
    "acceptance, release, or promotion authority."
)
_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_MATRIX_BYTES = 128 * 1024
_MAX_RUNNER_BYTES = 2 * 1024 * 1024
_MAX_SCHEDULER_BYTES = 256 * 1024
_MAX_COUNTER = 2**63 - 1
_TRUSTED_GIT_EXECUTABLE = Path("/usr/bin/git")
_TRUSTED_GIT_EXEC_PATH = Path("/usr/lib/git-core")
_TRUSTED_PROCESS_PATH = "/usr/bin:/bin"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}\Z")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})\Z")
_SECRET_VALUE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:api[_-]?key|access[_-]?token|token|password|passwd|secret)\s*[:=]\s*\S+|"
    r"(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{8,})"
)

_EXECUTION = {
    "measurement_scope": MEASUREMENT_SCOPE,
    "baseline_required_supervisor_processes": 1,
    "candidate_required_supervisor_processes": 12,
    "target_registered_logical_agents": 256,
    "target_maximum_active_subagents": 64,
    "subprocess_budget": 12,
    "real_independent_processes_required": True,
    "arms_execute_sequentially": True,
    "cross_arm_concurrency_permitted": False,
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
    "schema": CAPACITY_PREFLIGHT_SCHEMA,
    "current_supervisor_processes": 1,
    "current_registered_logical_agents": 1,
    "current_maximum_active_subagents": 1,
    "current_lanes": 1,
    "current_provider_concurrency": 1,
    "required_baseline_supervisor_processes": 1,
    "required_candidate_supervisor_processes": 12,
    "target_registered_logical_agents": 256,
    "target_maximum_active_subagents": 64,
    "authenticated_typed_quack_live_capacity": False,
    "high_concurrency_enabled": False,
    "availability": "unavailable",
    "reason_code": REASON_CODE,
}
_LIVE_CAPABILITY = {
    "availability": "unavailable",
    "execution_status": "not_run",
    "ran": False,
    "qualified": False,
    "reason_code": REASON_CODE,
    "required_attestation": LIVE_ATTESTATION_SCHEMA,
    "required_observation": LIVE_OBSERVATION_SCHEMA,
    "required_receipt_verifier": "registered_state_owner_receipt_verifier",
    "registered_receipt_verifier_available": False,
    "caller_supplied_references_qualify": False,
    "required_evidence": "current_generation_receipt_backed_raw_populations_current_fences_and_live_host_provider_token_telemetry",
    "metrics_omitted": True,
}
_ARM_BASELINE = {
    "arm_id": "baseline",
    "required_supervisor_processes": 1,
    "availability": "unavailable",
    "execution_status": "not_run",
    "qualified": False,
}
_ARM_CANDIDATE = {
    "arm_id": "candidate",
    "required_supervisor_processes": 12,
    "availability": "unavailable",
    "execution_status": "not_run",
    "qualified": False,
}
_COMPARISON = {
    "comparison_identity": "same-host-tasks-providers-tests-proofs-budgets",
    "arms_execute_sequentially": True,
    "cross_arm_concurrency_permitted": False,
    "same_population_bindings": list(SAME_POPULATION_BINDINGS),
    "provider_or_model_variants_are_not_arms": True,
    "baseline_arm": _ARM_BASELINE,
    "candidate_arm": _ARM_CANDIDATE,
    "measurement_status": "not_run",
    "lower_assurance_permitted": False,
}
_SCHEDULER_BINDING = {
    "relative_path": SCHEDULER_RELATIVE_PATH,
    "schema": SCHEDULER_SCHEMA,
    "sha256": SCHEDULER_SHA256,
}
_EXPECTED_MANIFEST = {
    "schema": MANIFEST_SCHEMA,
    "benchmark_id": BENCHMARK_ID,
    "program_id": PROGRAM_ID,
    "objective_id": OBJECTIVE_ID,
    "measurement_scope": MEASUREMENT_SCOPE,
    "frozen": True,
    "state": "capability_unavailable",
    "authoritative": False,
    "promotion_eligible": False,
    "matrix_binding": {
        "relative_path": MATRIX_RELATIVE_PATH,
        "schema": MATRIX_SCHEMA,
        "sha256": MATRIX_SHA256,
    },
    "scheduler_binding": _SCHEDULER_BINDING,
    "execution": _EXECUTION,
    "capacity_preflight": _PREFLIGHT,
    "comparison": _COMPARISON,
    "token_gates": TOKEN_GATES,
    "zero_tolerance_gates": list(ZERO_TOLERANCE_GATES),
    "future_identity_requirements": list(REQUIRED_IDENTITIES),
    "source_modules": list(SOURCE_RELATIVE_PATHS),
    "live_capability": _LIVE_CAPABILITY,
    "result_storage": RESULT_STORAGE,
    "nonclaims": list(NONCLAIMS),
}


class TokenBenchmarkError(ValueError):
    def __init__(self, message: str, *, reason_code: str = "invalid_contract") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TokenCapabilityUnavailable(TokenBenchmarkError):
    """A caller requested a live capability that the current tree cannot admit."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TokenBenchmarkError("duplicate JSON key", reason_code="duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise TokenBenchmarkError(
        f"non-finite JSON number {value!r} is prohibited", reason_code="invalid_json"
    )


def _loads_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise TokenBenchmarkError(f"{name} is not valid JSON", reason_code="invalid_json") from exc


def _read_bounded_regular_bytes(path: Path | str, *, name: str, maximum_bytes: int) -> bytes:
    candidate = Path(path)
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise TokenBenchmarkError(
            f"{name} is unavailable", reason_code="source_unavailable"
        ) from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise TokenBenchmarkError(
            f"{name} must be a regular non-symlink file",
            reason_code="unsafe_source_path",
        )
    if initial.st_size > maximum_bytes:
        raise TokenBenchmarkError(f"{name} exceeds its byte limit", reason_code="source_too_large")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise TokenBenchmarkError(
                f"{name} is not a bounded regular file",
                reason_code="unsafe_source_path",
            )
        if any(
            getattr(initial, field) != getattr(before, field)
            for field in ("st_dev", "st_ino", "st_mode")
        ):
            raise TokenBenchmarkError(
                f"{name} changed before its bounded read", reason_code="source_changed"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            payload = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except TokenBenchmarkError:
        raise
    except OSError as exc:
        raise TokenBenchmarkError(
            f"{name} could not be read safely", reason_code="source_unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise TokenBenchmarkError(f"{name} exceeds its byte limit", reason_code="source_too_large")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    ):
        raise TokenBenchmarkError(
            f"{name} changed during its bounded read", reason_code="source_changed"
        )
    return payload


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dict(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TokenBenchmarkError(f"{name} must be an exact JSON object")
    return value


def _exact_structure(value: Any, expected: Any, name: str) -> None:
    if type(expected) is dict:
        actual = _dict(value, name)
        if set(actual) != set(expected):
            raise TokenBenchmarkError(
                f"{name} has a closed schema", reason_code="closed_schema_violation"
            )
        for key, child in expected.items():
            _exact_structure(actual[key], child, f"{name}.{key}")
    elif type(expected) is list:
        if type(value) is not list or len(value) != len(expected):
            raise TokenBenchmarkError(f"{name} must be the exact frozen sequence")
        for index, (actual, child) in enumerate(zip(value, expected, strict=True)):
            _exact_structure(actual, child, f"{name}[{index}]")
    elif type(value) is not type(expected) or value != expected:
        raise TokenBenchmarkError(
            f"{name} differs from the exact frozen value",
            reason_code="frozen_contract_changed",
        )


def _text(value: Any, name: str, *, maximum_bytes: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise TokenBenchmarkError(f"{name} must be bounded exact text")
    if _SECRET_VALUE.search(value):
        raise TokenBenchmarkError(
            f"{name} contains credential-shaped material",
            reason_code="secret_shaped_input",
        )
    return value


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN.fullmatch(text) is None:
        raise TokenBenchmarkError(f"{name} must be a compact identity")
    return text


def _content_ref(value: Any, name: str) -> str:
    text = _token(value, name)
    if _CONTENT_REF.fullmatch(text) is None:
        raise TokenBenchmarkError(f"{name} must be a content-addressed reference")
    return text


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = _MAX_COUNTER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TokenBenchmarkError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _boolean(value: Any, name: str, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise TokenBenchmarkError(f"{name} must be exactly {expected!r}")
    return value


def _read_object(path: Path | str) -> dict[str, Any]:
    raw = _read_bounded_regular_bytes(path, name="JSON input", maximum_bytes=_MAX_MANIFEST_BYTES)
    try:
        decoded = _loads_json(raw.decode("utf-8", errors="strict"), name="JSON input")
    except UnicodeError as exc:
        raise TokenBenchmarkError("JSON input is not UTF-8", reason_code="invalid_json") from exc
    return _dict(decoded, "JSON input")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _exact_structure(manifest, _EXPECTED_MANIFEST, "manifest")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return _sha256(_canonical_bytes(manifest))


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest = _read_object(
        Path(path) if path is not None else Path(__file__).with_name("token_manifest.json")
    )
    validate_manifest(manifest)
    return manifest


def _validate_matrix(raw: bytes) -> None:
    if _sha256(raw) != MATRIX_SHA256:
        raise TokenBenchmarkError(
            "frozen benchmark matrix content is stale or changed",
            reason_code="matrix_binding_changed",
        )
    try:
        first_line = raw.decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError) as exc:
        raise TokenBenchmarkError(
            "frozen benchmark matrix is missing its schema",
            reason_code="matrix_binding_changed",
        ) from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise TokenBenchmarkError(
            "frozen benchmark matrix schema has changed",
            reason_code="matrix_binding_changed",
        )


def _validate_scheduler(raw: bytes) -> None:
    if _sha256(raw) != SCHEDULER_SHA256:
        raise TokenBenchmarkError(
            "authoritative scheduler source is stale or changed",
            reason_code="scheduler_binding_changed",
        )
    try:
        decoded = _loads_json(raw.decode("utf-8", errors="strict"), name="authoritative scheduler")
    except UnicodeError as exc:
        raise TokenBenchmarkError(
            "authoritative scheduler is not UTF-8",
            reason_code="scheduler_binding_changed",
        ) from exc
    scheduler = _dict(decoded, "authoritative scheduler")
    _exact_structure(scheduler.get("schema"), SCHEDULER_SCHEMA, "scheduler.schema")
    _exact_structure(
        scheduler.get("program_identifier"), PROGRAM_ID, "scheduler.program_identifier"
    )
    _exact_structure(
        scheduler.get("bootstrap_capacity"),
        {
            "supervisors": 1,
            "registered_logical_subagents": 1,
            "maximum_active_subagents": 1,
            "lanes": 1,
            "provider_concurrency": 1,
        },
        "scheduler.bootstrap_capacity",
    )
    gate = _dict(scheduler.get("high_concurrency_gate"), "scheduler.high_concurrency_gate")
    _exact_structure(
        gate.get("enabled_at_bootstrap"),
        False,
        "scheduler.high_concurrency_gate.enabled_at_bootstrap",
    )
    _exact_structure(
        gate.get("target_profile"),
        {
            "supervisors": 12,
            "registered_logical_subagents": 256,
            "maximum_active_subagents": 64,
        },
        "scheduler.high_concurrency_gate.target_profile",
    )
    _exact_structure(
        gate.get("missing_or_stale_telemetry_adds_capacity"),
        False,
        "scheduler.high_concurrency_gate.missing_or_stale_telemetry_adds_capacity",
    )
    for field in ("required_accepted_task_ids", "additional_live_requirements"):
        sequence = gate.get(field)
        if (
            type(sequence) is not list
            or not sequence
            or any(type(item) is not str or not item for item in sequence)
        ):
            raise TokenBenchmarkError(
                "scheduler high-concurrency gate is incomplete",
                reason_code="scheduler_binding_changed",
            )


def validate_matrix_binding(
    manifest: Mapping[str, Any], matrix_path: Path | str | None = None
) -> dict[str, str]:
    validate_manifest(manifest)
    raw = _read_bounded_regular_bytes(
        Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml"),
        name="benchmark matrix",
        maximum_bytes=_MAX_MATRIX_BYTES,
    )
    _validate_matrix(raw)
    return dict(manifest["matrix_binding"])


def _existing_path(path: Path | str, *, name: str) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(path)))
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise TokenBenchmarkError(
            f"{name} is unavailable", reason_code="unsafe_source_path"
        ) from None
    if lexical != resolved:
        raise TokenBenchmarkError(
            f"{name} may not traverse a symlink", reason_code="unsafe_source_path"
        )
    return resolved


def _bound_repository_and_recipe(
    repository: Path | str,
    manifest_path: Path | str | None,
    matrix_path: Path | str | None,
) -> tuple[Path, Path, Path, Path]:
    root = _existing_path(repository, name="repository")
    if root != _existing_path(__file__, name="benchmark runner").parents[3]:
        raise TokenBenchmarkError(
            "repository must contain this exact benchmark runner",
            reason_code="repository_binding_changed",
        )
    manifest = _existing_path(
        root / MANIFEST_RELATIVE_PATH if manifest_path is None else manifest_path,
        name="benchmark manifest",
    )
    matrix = _existing_path(
        root / MATRIX_RELATIVE_PATH if matrix_path is None else matrix_path,
        name="benchmark matrix",
    )
    scheduler = _existing_path(root / SCHEDULER_RELATIVE_PATH, name="authoritative scheduler")
    if (manifest, matrix, scheduler) != (
        root / MANIFEST_RELATIVE_PATH,
        root / MATRIX_RELATIVE_PATH,
        root / SCHEDULER_RELATIVE_PATH,
    ):
        raise TokenBenchmarkError(
            "recipe inputs must be exact measured-tree files",
            reason_code="repository_binding_changed",
        )
    return root, manifest, matrix, scheduler


def _trusted_git_assets() -> tuple[str, str]:
    try:
        executable = _TRUSTED_GIT_EXECUTABLE.lstat()
        exec_path = _TRUSTED_GIT_EXEC_PATH.lstat()
    except OSError as exc:
        raise TokenBenchmarkError(
            "trusted Git installation is unavailable",
            reason_code="trusted_git_unavailable",
        ) from exc
    if (
        not stat.S_ISREG(executable.st_mode)
        or executable.st_uid != 0
        or executable.st_mode & 0o022
        or not executable.st_mode & 0o111
    ):
        raise TokenBenchmarkError(
            "trusted Git executable is not a protected executable regular file",
            reason_code="trusted_git_unavailable",
        )
    if not stat.S_ISDIR(exec_path.st_mode) or exec_path.st_uid != 0 or exec_path.st_mode & 0o022:
        raise TokenBenchmarkError(
            "trusted Git exec path is not a protected directory",
            reason_code="trusted_git_unavailable",
        )
    return str(_TRUSTED_GIT_EXECUTABLE), str(_TRUSTED_GIT_EXEC_PATH)


def _git_environment(trusted_exec_path: str | None = None) -> dict[str, str]:
    """Construct a fixed environment with no ambient process injection state."""
    if trusted_exec_path is None:
        _trusted_executable, trusted_exec_path = _trusted_git_assets()
    return {
        "HOME": "/",
        "PATH": _TRUSTED_PROCESS_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_EXEC_PATH": trusted_exec_path,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _git_bytes(repository: Path, *args: str) -> bytes:
    trusted_git, trusted_exec_path = _trusted_git_assets()
    try:
        result = subprocess.run(
            [
                trusted_git,
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(repository),
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=15,
            env=_git_environment(trusted_exec_path),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TokenBenchmarkError(
            "cannot establish repository identity",
            reason_code="repository_identity_unavailable",
        ) from exc
    return result.stdout


def _git(repository: Path, *args: str) -> str:
    try:
        return _git_bytes(repository, *args).decode("utf-8", errors="strict").rstrip("\n")
    except UnicodeError as exc:
        raise TokenBenchmarkError(
            "Git returned non-UTF-8 identity output",
            reason_code="repository_identity_unavailable",
        ) from exc


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = _existing_path(repository, name="repository")
    top_level = _existing_path(_git(root, "rev-parse", "--show-toplevel"), name="Git top level")
    if top_level != root:
        raise TokenBenchmarkError(
            "Git top level differs from the measured repository",
            reason_code="repository_binding_changed",
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise TokenBenchmarkError(
            "repository is dirty; exact current-tree evidence is unavailable",
            reason_code="repository_dirty",
        )
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git(root, "rev-parse", "--verify", "HEAD^{tree}")
    if _GIT_OBJECT_ID.fullmatch(commit) is None or _GIT_OBJECT_ID.fullmatch(tree) is None:
        raise TokenBenchmarkError("repository identity is not a full lowercase Git object id")
    return {"repository_commit": commit, "repository_tree": tree}


class _Snapshot(NamedTuple):
    identity: dict[str, str]
    manifest: dict[str, Any]
    manifest_sha256: str
    matrix_sha256: str
    scheduler_sha256: str
    source_sha256: dict[str, str]
    tracked_head_blob_oid: dict[str, str]


def _tracked_head_blob(repository: Path, tree_oid: str, relative_path: str) -> tuple[str, bytes]:
    entry = _git_bytes(
        repository,
        "ls-tree",
        "-z",
        "--full-tree",
        tree_oid,
        "--",
        relative_path,
    )
    records = [record for record in entry.split(b"\0") if record]
    if len(records) != 1:
        raise TokenBenchmarkError(
            "a measured path is not exactly tracked by HEAD",
            reason_code="head_blob_mismatch",
        )
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, encoded_oid = metadata.split(b" ", 2)
        oid = encoded_oid.decode("ascii", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise TokenBenchmarkError(
            "a measured HEAD tree entry is malformed",
            reason_code="head_blob_mismatch",
        ) from exc
    if (
        encoded_path != relative_path.encode("utf-8")
        or mode not in {b"100644", b"100755"}
        or object_type != b"blob"
        or _GIT_OBJECT_ID.fullmatch(oid) is None
    ):
        raise TokenBenchmarkError(
            "a measured HEAD tree entry is not an exact regular blob",
            reason_code="head_blob_mismatch",
        )
    raw = _git_bytes(repository, "cat-file", "blob", oid)
    framed = f"blob {len(raw)}\0".encode("ascii") + raw
    if len(oid) == 40:
        calculated = hashlib.sha1(framed, usedforsecurity=False).hexdigest()
    else:
        calculated = hashlib.sha256(framed).hexdigest()
    if calculated != oid:
        raise TokenBenchmarkError(
            "a measured HEAD blob failed object-id verification",
            reason_code="head_blob_mismatch",
        )
    return oid, raw


def _tracked_head_blobs(repository: Path, tree_oid: str) -> tuple[dict[str, str], dict[str, bytes]]:
    object_ids: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for relative_path in MEASURED_RELATIVE_PATHS:
        object_id, payload = _tracked_head_blob(repository, tree_oid, relative_path)
        object_ids[relative_path] = object_id
        payloads[relative_path] = payload
    return object_ids, payloads


def _snapshot(
    repository: Path, manifest_path: Path, matrix_path: Path, scheduler_path: Path
) -> _Snapshot:
    before = repository_identity(repository)
    paths = {
        MANIFEST_RELATIVE_PATH: (manifest_path, _MAX_MANIFEST_BYTES),
        RUNNER_RELATIVE_PATH: (repository / RUNNER_RELATIVE_PATH, _MAX_RUNNER_BYTES),
        MATRIX_RELATIVE_PATH: (matrix_path, _MAX_MATRIX_BYTES),
        SCHEDULER_RELATIVE_PATH: (scheduler_path, _MAX_SCHEDULER_BYTES),
    }

    def read_sources() -> dict[str, bytes]:
        return {
            relative: _read_bounded_regular_bytes(
                path, name=f"measured source {relative}", maximum_bytes=limit
            )
            for relative, (path, limit) in paths.items()
        }

    first = read_sources()
    head_oids_first, head_first = _tracked_head_blobs(repository, before["repository_tree"])
    if first != head_first:
        raise TokenBenchmarkError(
            "measured bytes differ from their exact tracked HEAD blobs",
            reason_code="head_blob_mismatch",
        )
    try:
        manifest_value = _loads_json(
            first[MANIFEST_RELATIVE_PATH].decode("utf-8", errors="strict"),
            name="manifest",
        )
    except UnicodeError as exc:
        raise TokenBenchmarkError("manifest is not UTF-8", reason_code="invalid_json") from exc
    manifest = _dict(manifest_value, "manifest")
    validate_manifest(manifest)
    _validate_matrix(first[MATRIX_RELATIVE_PATH])
    _validate_scheduler(first[SCHEDULER_RELATIVE_PATH])
    second = read_sources()
    head_oids_second, head_second = _tracked_head_blobs(repository, before["repository_tree"])
    after = repository_identity(repository)
    if (
        before != after
        or first != second
        or head_oids_first != head_oids_second
        or head_first != head_second
        or second != head_second
    ):
        raise TokenBenchmarkError(
            "repository identity or measured source changed during evidence collection",
            reason_code="source_changed",
        )
    return _Snapshot(
        before,
        manifest,
        _sha256(first[MANIFEST_RELATIVE_PATH]),
        _sha256(first[MATRIX_RELATIVE_PATH]),
        _sha256(first[SCHEDULER_RELATIVE_PATH]),
        {key: _sha256(value) for key, value in first.items()},
        head_oids_first,
    )


def capacity_preflight() -> dict[str, Any]:
    """Return the literal no-I/O 1/1/1 preflight before inspecting caller data."""
    return dict(_PREFLIGHT)


def _validate_preflight(value: Any) -> dict[str, Any]:
    _exact_structure(value, _PREFLIGHT, "capacity preflight")
    return dict(value)


def _deny_live_verification() -> NoReturn:
    raise TokenCapabilityUnavailable(
        "live token evidence cannot be admitted without a registered state-owner receipt verifier",
        reason_code=LIVE_VERIFIER_REASON_CODE,
    )


class TokenIdentity(NamedTuple):
    repository_commit: str
    repository_tree: str
    control_plane_generation: int
    schema_fingerprint: str
    policy_ref: str
    policy_revision: int
    capability_ref: str
    federation_id: str
    task_id: str
    attempt_id: str
    worktree_id: str
    assignment_revision: int
    fencing_epoch: int

    def as_dict(self) -> dict[str, Any]:
        return self._asdict()

    def content_ref(self) -> str:
        return f"sha256:{_sha256(_canonical_bytes(self.as_dict()))}"


def _validate_token_identity(identity: TokenIdentity) -> dict[str, Any]:
    _deny_live_verification()
    if type(identity) is not TokenIdentity:
        raise TokenBenchmarkError("token identity must use the closed typed contract")
    checked = identity.as_dict()
    for field in ("repository_commit", "repository_tree"):
        if type(checked[field]) is not str or _GIT_OBJECT_ID.fullmatch(checked[field]) is None:
            raise TokenBenchmarkError(f"identity.{field} must be a full Git object id")
    for field in (
        "control_plane_generation",
        "policy_revision",
        "assignment_revision",
        "fencing_epoch",
    ):
        checked[field] = _integer(checked[field], f"identity.{field}", minimum=1)
    for field in ("schema_fingerprint", "policy_ref", "capability_ref"):
        checked[field] = _content_ref(checked[field], f"identity.{field}")
    for field in ("federation_id", "task_id", "attempt_id", "worktree_id"):
        checked[field] = _token(checked[field], f"identity.{field}")
    if checked["task_id"] != OBJECTIVE_ID:
        raise TokenBenchmarkError("identity.task_id must bind CASF-041")
    return checked


def _validate_identities(value: Any) -> dict[str, Any]:
    return _validate_token_identity(value)


class TokenAdmission(NamedTuple):
    schema: str
    identity: TokenIdentity
    comparison_plan_ref: str
    quack_admission_receipt_ref: str
    exclusive_state_owner_birth_ref: str
    provider_usage_attestation_ref: str
    baseline_supervisor_process_birth_refs: tuple[str, ...]
    candidate_supervisor_process_birth_refs: tuple[str, ...]
    expected_observation_sequence: int
    replay_guard_ref: str
    provider_ref: str
    model_ref: str
    tokenizer_ref: str
    host_ref: str
    workload_ref: str
    task_population_ref: str
    criteria_ref: str
    budget_ref: str
    tests_ref: str
    proofs_ref: str
    assurance_ref: str
    retry_policy_ref: str
    replay_policy_ref: str
    fallback_policy_ref: str
    cancellation_policy_ref: str
    capsule_policy_ref: str
    board_scan_policy_ref: str
    arms_execute_sequentially: bool
    cross_arm_concurrency_permitted: bool
    state_authority: str
    direct_database_access_permitted: bool
    ducklake_scheduling_authority_permitted: bool
    ducklake_projection_authoritative: bool


class AdmittedTokenExecutor(Protocol):
    interface: str

    def execute_token_comparison(self, *, admission: TokenAdmission) -> Mapping[str, Any]: ...


def _birth_refs(
    value: Any, name: str, *, required_count: int, collection_type: type
) -> tuple[str, ...]:
    if type(value) is not collection_type or len(value) != required_count:
        raise TokenBenchmarkError(
            f"{name} must contain exactly {required_count} typed process births"
        )
    checked = tuple(_content_ref(item, f"{name}[{index}]") for index, item in enumerate(value))
    if tuple(sorted(checked)) != checked or len(set(checked)) != len(checked):
        raise TokenBenchmarkError(f"{name} must be sorted and unique")
    return checked


def _validate_token_admission(admission: TokenAdmission) -> dict[str, Any]:
    _deny_live_verification()
    if type(admission) is not TokenAdmission:
        raise TokenBenchmarkError("live token admission must use the closed typed contract")
    identity = _validate_token_identity(admission.identity)
    exact = {
        "schema": LIVE_ATTESTATION_SCHEMA,
        "arms_execute_sequentially": True,
        "cross_arm_concurrency_permitted": False,
        "state_authority": "authenticated_typed_quack",
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "ducklake_projection_authoritative": False,
    }
    actual = {key: getattr(admission, key) for key in exact}
    _exact_structure(actual, exact, "live token admission")
    content_fields = (
        "comparison_plan_ref",
        "quack_admission_receipt_ref",
        "exclusive_state_owner_birth_ref",
        "provider_usage_attestation_ref",
        "replay_guard_ref",
        "host_ref",
        "workload_ref",
        "task_population_ref",
        "criteria_ref",
        "budget_ref",
        "tests_ref",
        "proofs_ref",
        "assurance_ref",
        "retry_policy_ref",
        "replay_policy_ref",
        "fallback_policy_ref",
        "cancellation_policy_ref",
        "capsule_policy_ref",
        "board_scan_policy_ref",
    )
    checked: dict[str, Any] = {"identity": identity, **actual}
    for field in content_fields:
        checked[field] = _content_ref(getattr(admission, field), f"admission.{field}")
    for field in ("provider_ref", "model_ref", "tokenizer_ref"):
        checked[field] = _token(getattr(admission, field), f"admission.{field}")
    checked["expected_observation_sequence"] = _integer(
        admission.expected_observation_sequence,
        "admission.expected_observation_sequence",
        minimum=1,
    )
    baseline = _birth_refs(
        admission.baseline_supervisor_process_birth_refs,
        "admission.baseline_supervisor_process_birth_refs",
        required_count=1,
        collection_type=tuple,
    )
    candidate = _birth_refs(
        admission.candidate_supervisor_process_birth_refs,
        "admission.candidate_supervisor_process_birth_refs",
        required_count=12,
        collection_type=tuple,
    )
    if set(baseline) & set(candidate):
        raise TokenBenchmarkError("baseline and candidate process births must be disjoint")
    if admission.exclusive_state_owner_birth_ref in set(baseline) | set(candidate):
        raise TokenBenchmarkError(
            "exclusive state-owner birth must be distinct from measured supervisors"
        )
    checked["baseline_supervisor_process_birth_refs"] = baseline
    checked["candidate_supervisor_process_birth_refs"] = candidate
    checked["identity_ref"] = admission.identity.content_ref()
    return checked


_validate_admission = _validate_token_admission
_ARM_RECEIPT_KEYS = (
    "provider_usage_population_ref",
    "task_criterion_population_ref",
    "model_call_population_ref",
    "attempt_population_ref",
    "capsule_population_ref",
    "capsule_freshness_ref",
    "board_scan_population_ref",
)
_ARM_POPULATION_KEYS = (
    "task_count",
    "criterion_opportunities",
    "accepted_criteria",
    "repeated_context_input_tokens",
    "model_input_tokens",
    "primary_model_calls",
    "duplicate_model_calls",
    "retry_model_calls",
    "replay_model_calls",
    "fallback_model_calls",
    "cancelled_model_calls",
    "total_model_calls",
    "eligible_semantic_capsules",
    "reused_semantic_capsules",
    "recomputed_semantic_capsules",
    "rejected_stale_semantic_capsules",
    "stale_semantic_capsules_reused",
    "board_scan_opportunities",
    "complete_board_scans",
    "incremental_board_scans",
)


def _validate_arm(
    value: Any, *, arm_id: str, process_birth_refs: tuple[str, ...]
) -> dict[str, Any]:
    arm = _dict(value, f"{arm_id} arm")
    if set(arm) != {
        "schema",
        "arm_id",
        "arm_execution_ref",
        "supervisor_process_birth_refs",
        "population_receipts",
        "populations",
    }:
        raise TokenBenchmarkError(
            f"{arm_id} arm has a closed schema", reason_code="closed_schema_violation"
        )
    _exact_structure(arm["schema"], LIVE_ARM_SCHEMA, f"{arm_id} arm.schema")
    _exact_structure(arm["arm_id"], arm_id, f"{arm_id} arm.arm_id")
    execution_ref = _content_ref(arm["arm_execution_ref"], f"{arm_id} arm.arm_execution_ref")
    observed_births = _birth_refs(
        arm["supervisor_process_birth_refs"],
        f"{arm_id} arm.supervisor_process_birth_refs",
        required_count=len(process_birth_refs),
        collection_type=list,
    )
    if observed_births != process_birth_refs:
        raise TokenBenchmarkError(f"{arm_id} arm process births are not admitted")
    receipts = _dict(arm["population_receipts"], f"{arm_id} arm.population_receipts")
    if set(receipts) != set(_ARM_RECEIPT_KEYS):
        raise TokenBenchmarkError(
            f"{arm_id} arm receipt schema is closed",
            reason_code="closed_schema_violation",
        )
    checked_receipts = {
        key: _content_ref(receipts[key], f"{arm_id} arm.population_receipts.{key}")
        for key in _ARM_RECEIPT_KEYS
    }
    if len(set(checked_receipts.values())) != len(checked_receipts):
        raise TokenBenchmarkError(f"{arm_id} arm receipt references must be unique")
    populations = _dict(arm["populations"], f"{arm_id} arm.populations")
    if set(populations) != set(_ARM_POPULATION_KEYS):
        raise TokenBenchmarkError(
            f"{arm_id} arm population schema is closed",
            reason_code="closed_schema_violation",
        )
    checked = {
        key: _integer(populations[key], f"{arm_id} arm.populations.{key}")
        for key in _ARM_POPULATION_KEYS
    }
    if checked["task_count"] == 0:
        raise TokenBenchmarkError(f"{arm_id} arm task population must be nonempty")
    if (
        checked["criterion_opportunities"] == 0
        or checked["accepted_criteria"] == 0
        or checked["accepted_criteria"] > checked["criterion_opportunities"]
    ):
        raise TokenBenchmarkError(f"{arm_id} arm criterion population is impossible")
    if checked["repeated_context_input_tokens"] > checked["model_input_tokens"]:
        raise TokenBenchmarkError(f"{arm_id} arm repeated context exceeds model input")
    call_parts = sum(
        checked[key]
        for key in (
            "primary_model_calls",
            "duplicate_model_calls",
            "retry_model_calls",
            "replay_model_calls",
            "fallback_model_calls",
            "cancelled_model_calls",
        )
    )
    if call_parts != checked["total_model_calls"]:
        raise TokenBenchmarkError(f"{arm_id} arm model-call population is not an exact partition")
    if checked["model_input_tokens"] < checked["total_model_calls"]:
        raise TokenBenchmarkError(
            f"{arm_id} arm model input cannot cover its model-call population"
        )
    if (
        checked["eligible_semantic_capsules"] == 0
        or checked["reused_semantic_capsules"] + checked["recomputed_semantic_capsules"]
        != checked["eligible_semantic_capsules"]
        or checked["stale_semantic_capsules_reused"] != 0
    ):
        raise TokenBenchmarkError(f"{arm_id} arm capsule population is impossible")
    if (
        checked["board_scan_opportunities"] == 0
        or checked["complete_board_scans"] + checked["incremental_board_scans"]
        != checked["board_scan_opportunities"]
    ):
        raise TokenBenchmarkError(f"{arm_id} arm board-scan population is impossible")
    return {
        "schema": LIVE_ARM_SCHEMA,
        "arm_id": arm_id,
        "arm_execution_ref": execution_ref,
        "supervisor_process_birth_refs": list(observed_births),
        "population_receipts": checked_receipts,
        "populations": checked,
    }


_OBSERVATION_CONTENT_BINDINGS = (
    "comparison_plan_ref",
    "quack_admission_receipt_ref",
    "exclusive_state_owner_birth_ref",
    "provider_usage_attestation_ref",
    "replay_guard_ref",
    "workload_ref",
    "task_population_ref",
    "criteria_ref",
    "budget_ref",
    "retry_policy_ref",
    "replay_policy_ref",
    "fallback_policy_ref",
    "cancellation_policy_ref",
    "capsule_policy_ref",
    "board_scan_policy_ref",
    "host_ref",
    "tests_ref",
    "proofs_ref",
    "assurance_ref",
    "evidence_coverage_receipt_ref",
    "zero_tolerance_receipt_ref",
)
_OBSERVATION_TOKEN_BINDINGS = ("provider_ref", "model_ref", "tokenizer_ref")


def validate_admitted_token_observation(
    admission: TokenAdmission, observation: Mapping[str, Any]
) -> NoReturn:
    """Deny before inspecting caller evidence until a receipt verifier is registered."""
    _deny_live_verification()
    admitted = _validate_token_admission(admission)
    value = _dict(observation, "token observation")
    expected_keys = {
        "schema",
        "identity_ref",
        "observation_sequence",
        "comparison_plan_ref",
        "quack_admission_receipt_ref",
        "exclusive_state_owner_birth_ref",
        "provider_usage_attestation_ref",
        "replay_guard_ref",
        "host_ref",
        "workload_ref",
        "task_population_ref",
        "criteria_ref",
        "provider_ref",
        "model_ref",
        "tokenizer_ref",
        "budget_ref",
        "tests_ref",
        "proofs_ref",
        "assurance_ref",
        "retry_policy_ref",
        "replay_policy_ref",
        "fallback_policy_ref",
        "cancellation_policy_ref",
        "capsule_policy_ref",
        "board_scan_policy_ref",
        "state_transport",
        "arms_executed_sequentially",
        "cross_arm_concurrency_observed",
        "direct_database_access_used",
        "ducklake_scheduling_authority_used",
        "ducklake_projection_authoritative",
        "attempt_population_complete",
        "retries_included",
        "replays_included",
        "fallbacks_included",
        "cancellations_included",
        "evidence_coverage_preserved",
        "evidence_coverage_receipt_ref",
        "zero_tolerance_gate_failures",
        "zero_tolerance_receipt_ref",
        "baseline_arm",
        "candidate_arm",
    }
    if set(value) != expected_keys:
        raise TokenBenchmarkError(
            "token observation has a closed schema",
            reason_code="closed_schema_violation",
        )
    _exact_structure(value["schema"], LIVE_OBSERVATION_SCHEMA, "observation.schema")
    _exact_structure(value["identity_ref"], admitted["identity_ref"], "observation.identity_ref")
    sequence = _integer(
        value["observation_sequence"], "observation.observation_sequence", minimum=1
    )
    if sequence != admission.expected_observation_sequence:
        raise TokenBenchmarkError("observation sequence is stale, skipped, or replayed")
    for field in _OBSERVATION_CONTENT_BINDINGS:
        _content_ref(value[field], f"observation.{field}")
    for field in _OBSERVATION_TOKEN_BINDINGS:
        _token(value[field], f"observation.{field}")
    for field in (
        "comparison_plan_ref",
        "quack_admission_receipt_ref",
        "exclusive_state_owner_birth_ref",
        "provider_usage_attestation_ref",
        "replay_guard_ref",
        "host_ref",
        "workload_ref",
        "task_population_ref",
        "criteria_ref",
        "provider_ref",
        "model_ref",
        "tokenizer_ref",
        "budget_ref",
        "tests_ref",
        "proofs_ref",
        "assurance_ref",
        "retry_policy_ref",
        "replay_policy_ref",
        "fallback_policy_ref",
        "cancellation_policy_ref",
        "capsule_policy_ref",
        "board_scan_policy_ref",
    ):
        if value[field] != getattr(admission, field):
            raise TokenBenchmarkError(f"observation.{field} differs from its admission")
    _exact_structure(
        value["state_transport"],
        "authenticated_typed_quack",
        "observation.state_transport",
    )
    for field, expected in {
        "arms_executed_sequentially": True,
        "cross_arm_concurrency_observed": False,
        "direct_database_access_used": False,
        "ducklake_scheduling_authority_used": False,
        "ducklake_projection_authoritative": False,
        "attempt_population_complete": True,
        "retries_included": True,
        "replays_included": True,
        "fallbacks_included": True,
        "cancellations_included": True,
        "evidence_coverage_preserved": True,
    }.items():
        _boolean(value[field], f"observation.{field}", expected)
    failures = _dict(value["zero_tolerance_gate_failures"], "zero_tolerance_gate_failures")
    if set(failures) != set(ZERO_TOLERANCE_GATES) or any(
        _integer(failures[gate], f"zero_tolerance_gate_failures.{gate}") != 0
        for gate in ZERO_TOLERANCE_GATES
    ):
        raise TokenBenchmarkError("token observation fails a zero-tolerance gate")
    baseline = _validate_arm(
        value["baseline_arm"],
        arm_id="baseline",
        process_birth_refs=admission.baseline_supervisor_process_birth_refs,
    )
    candidate = _validate_arm(
        value["candidate_arm"],
        arm_id="candidate",
        process_birth_refs=admission.candidate_supervisor_process_birth_refs,
    )
    if baseline["arm_execution_ref"] == candidate["arm_execution_ref"]:
        raise TokenBenchmarkError("baseline and candidate executions must be distinct")
    if set(baseline["population_receipts"].values()) & set(
        candidate["population_receipts"].values()
    ):
        raise TokenBenchmarkError("cross-arm population receipts must be disjoint")
    base = baseline["populations"]
    cand = candidate["populations"]
    for field in (
        "task_count",
        "criterion_opportunities",
        "accepted_criteria",
        "eligible_semantic_capsules",
        "board_scan_opportunities",
    ):
        if base[field] != cand[field]:
            raise TokenBenchmarkError(f"arms do not use the same raw {field} population")
    if (
        base["repeated_context_input_tokens"] == 0
        or base["model_input_tokens"] == 0
        or base["duplicate_model_calls"] == 0
        or base["complete_board_scans"] == 0
    ):
        raise TokenBenchmarkError("baseline denominators must be nonzero for every reduction gate")
    gates = {
        "repeated_context_token_reduction": cand["repeated_context_input_tokens"] * 100
        <= base["repeated_context_input_tokens"]
        * (100 - TOKEN_GATES["minimum_repeated_context_token_reduction_percent"]),
        "input_tokens_per_accepted_criterion_reduction": cand["model_input_tokens"]
        * base["accepted_criteria"]
        * 100
        <= base["model_input_tokens"]
        * cand["accepted_criteria"]
        * (100 - TOKEN_GATES["minimum_input_tokens_per_accepted_criterion_reduction_percent"]),
        "duplicate_model_call_reduction": cand["duplicate_model_calls"] * 100
        <= base["duplicate_model_calls"]
        * (100 - TOKEN_GATES["minimum_duplicate_model_call_reduction_percent"]),
        "eligible_semantic_capsule_reuse": cand["reused_semantic_capsules"] * 100
        >= cand["eligible_semantic_capsules"]
        * TOKEN_GATES["minimum_eligible_semantic_capsule_reuse_percent"],
        "complete_board_scan_reduction": cand["complete_board_scans"] * 100
        <= base["complete_board_scans"]
        * (100 - TOKEN_GATES["minimum_complete_board_scan_reduction_percent"]),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise TokenBenchmarkError(f"raw observation fails exact gate {failed[0]}")
    _deny_live_verification()


def live_capability(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    checked = load_manifest() if manifest is None else dict(manifest)
    validate_manifest(checked)
    return dict(checked["live_capability"])


def require_live_capability(manifest: Mapping[str, Any]) -> NoReturn:
    _deny_live_verification()
    validate_manifest(manifest)
    raise TokenCapabilityUnavailable(
        "qualified cross-supervisor token live capacity is unavailable",
        reason_code=REASON_CODE,
    )


def execute_admitted_token(admission: TokenAdmission, executor: AdmittedTokenExecutor) -> NoReturn:
    _deny_live_verification()
    if (
        _token(getattr(executor, "interface", None), "admitted token executor interface")
        != ADMITTED_EXECUTION_INTERFACE
    ):
        raise TokenBenchmarkError("admitted token executor interface is invalid")
    _validate_token_admission(admission)
    raise TokenCapabilityUnavailable(
        "admitted token execution is unavailable at current capacity",
        reason_code="admitted_execution_unavailable",
    )


def _admitted_execution_boundary(
    executor: AdmittedTokenExecutor, *, admission: TokenAdmission
) -> NoReturn:
    return execute_admitted_token(admission, executor)


def _result(snapshot: _Snapshot, preflight: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": OBJECTIVE_ID,
        "measurement_scope": MEASUREMENT_SCOPE,
        "availability": "unavailable",
        "execution_status": "not_run",
        "ran": False,
        "qualified": False,
        "authoritative": False,
        "promotion_eligible": False,
        "metrics_omitted": True,
        "reason_code": REASON_CODE,
        "repository_binding": {
            **snapshot.identity,
            "git_status_porcelain_empty": True,
            "measured_paths_match_tracked_head_blobs": True,
            "observed_before_and_after": True,
        },
        "manifest_binding": {
            "relative_path": MANIFEST_RELATIVE_PATH,
            "schema": MANIFEST_SCHEMA,
            "raw_sha256": snapshot.manifest_sha256,
        },
        "matrix_binding": {
            "relative_path": MATRIX_RELATIVE_PATH,
            "schema": MATRIX_SCHEMA,
            "sha256": snapshot.matrix_sha256,
        },
        "scheduler_binding": {
            "relative_path": SCHEDULER_RELATIVE_PATH,
            "schema": SCHEDULER_SCHEMA,
            "raw_sha256": snapshot.scheduler_sha256,
        },
        "source_binding": {
            "source_sha256": snapshot.source_sha256,
            "tracked_head_blob_oid": snapshot.tracked_head_blob_oid,
            "exact_tracked_head_bytes": True,
            "observed_before_and_after": True,
        },
        "capacity_preflight": preflight,
        "future_required_comparison": _COMPARISON,
        "token_gates": TOKEN_GATES,
        "zero_tolerance_gates": list(ZERO_TOLERANCE_GATES),
        "future_identity_requirements": list(REQUIRED_IDENTITIES),
        "live_capability": _LIVE_CAPABILITY,
        "nonclaims": list(NONCLAIMS),
    }
    result["content_sha256"] = result_content_sha256(result)
    return result


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(_dict(result, "result"))
    payload.pop("content_sha256", None)
    return _sha256(_canonical_bytes(payload))


def _validate_result(
    result: Mapping[str, Any], snapshot: _Snapshot, preflight: dict[str, Any]
) -> None:
    expected = _result(snapshot, preflight)
    _exact_structure(result, expected, "result")
    if result.get("content_sha256") != result_content_sha256(result):
        raise TokenBenchmarkError("result content digest does not bind the result")


def run_benchmark(
    repository: Path | str,
    identities: Any = None,
    *,
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> dict[str, Any]:
    # This literal, no-I/O check must precede all caller-controlled values.
    preflight = _validate_preflight(capacity_preflight())
    root, manifest, matrix, scheduler = _bound_repository_and_recipe(
        repository, manifest_path, matrix_path
    )
    snapshot = _snapshot(root, manifest, matrix, scheduler)
    del identities
    return _result(snapshot, preflight)


def validate_result(
    result: Mapping[str, Any],
    *,
    repository: Path | str,
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
    current_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    preflight = _validate_preflight(capacity_preflight())
    root, manifest, matrix, scheduler = _bound_repository_and_recipe(
        repository, manifest_path, matrix_path
    )
    snapshot = _snapshot(root, manifest, matrix, scheduler)
    if current_identity is not None:
        supplied = _dict(current_identity, "current identity")
        _exact_structure(supplied, snapshot.identity, "current identity")
    _validate_result(result, snapshot, preflight)
    return dict(result)


def _invalid_diagnostic(reason_code: str) -> dict[str, str]:
    allowed = {
        "missing_required_argument",
        "unsafe_source_path",
        "repository_binding_changed",
        "repository_identity_unavailable",
        "trusted_git_unavailable",
        "repository_dirty",
        "source_unavailable",
        "source_too_large",
        "source_changed",
        "matrix_binding_changed",
        "scheduler_binding_changed",
        "invalid_json",
        "duplicate_json_key",
        "closed_schema_violation",
        "frozen_contract_changed",
        "secret_shaped_input",
        "invalid_contract",
    }
    safe_code = reason_code if reason_code in allowed else "invalid_contract"
    messages = {
        "missing_required_argument": "repository is required",
        "unsafe_source_path": "a required source path is unavailable or unsafe",
    }
    return {
        "schema": ERROR_SCHEMA,
        "execution_status": "invalid",
        "error_code": safe_code,
        "message": messages.get(safe_code, "benchmark input or exact-tree binding is invalid"),
    }


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
        result = run_benchmark(
            args.repository, manifest_path=args.manifest, matrix_path=args.matrix
        )
    except TokenBenchmarkError as exc:
        print(json.dumps(_invalid_diagnostic(exc.reason_code), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
