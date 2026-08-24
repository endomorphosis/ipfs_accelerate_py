#!/usr/bin/env python3
"""Measure the scoped CASF idle baseline in one real child process.

The parent launches one hermetic probe which executes the repository's actual
``StateOwnerEventWait``.  The child completes one bounded idle wait, then the
parent injects one event and observes its lossless delivery.  Every child span
is bound to a fresh challenge, the OS process birth, and a parent-built hash
chain.  The resulting artifact is replayable but deliberately non-authoritative.

This runner never opens DuckDB, invents a Quack transport, or accepts a
caller-authored metrics report.  Typed Quack live execution is explicitly
recorded as unavailable/not-run; the measured result is only an owner-local,
single-process hermetic baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import selectors
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-benchmark-manifest@2"
)
RESULT_SCHEMA = "casf/idle-benchmark@1"
SENSOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-parent-sensor-log@1"
)
PARENT_SPAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-parent-observed-span@1"
)
CHILD_SPAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-child-span@1"
ACTIVITY_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-activity-counter-snapshot@1"
)
INJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-parent-injection@1"
)
ERROR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation/idle-child-error@1"
MATRIX_SCHEMA = "ipfs_accelerate_py/agent-supervisor/causal-event-federation-benchmark-matrix@1"
BENCHMARK_ID = "casf-idle-event-driven-v2"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
MATRIX_SHA256 = "b23681a8c811f2020ef97e1b1b0172c15c87577d7882a4323f67e072dd7dfd9f"
SOURCE_RELATIVE_PATHS = (
    "benchmarks/agent_supervisor/causal_event_federation/run_idle.py",
    "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py",
    "ipfs_accelerate_py/agent_supervisor/federation/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/federation/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/federation/event_wait.py",
    "ipfs_accelerate_py/agent_supervisor/federation/events.py",
    "ipfs_accelerate_py/agent_supervisor/federation/outbox.py",
)
EXPECTED_MODULE_ORIGINS = {
    "ipfs_accelerate_py": "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py.agent_supervisor": "ipfs_accelerate_py/agent_supervisor/__init__.py",
    "ipfs_accelerate_py.agent_supervisor.task_sources": (
        "ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts": (
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.federation": (
        "ipfs_accelerate_py/agent_supervisor/federation/__init__.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.federation.contracts": (
        "ipfs_accelerate_py/agent_supervisor/federation/contracts.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.federation.event_wait": (
        "ipfs_accelerate_py/agent_supervisor/federation/event_wait.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.federation.events": (
        "ipfs_accelerate_py/agent_supervisor/federation/events.py"
    ),
    "ipfs_accelerate_py.agent_supervisor.federation.outbox": (
        "ipfs_accelerate_py/agent_supervisor/federation/outbox.py"
    ),
}
CHILD_ROLE = "state_owner_event_wait_probe"
MEASUREMENT_SCOPE = "single_supervisor_population_hermetic_real_process_probe"
FIRST_SPAN_PARENT = "0" * 64
MAX_JSON_LINE_BYTES = 65_536
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
REQUIRED_IDLE_ZERO_METRICS = (
    "task_board_scans",
    "model_calls",
    "context_recompilations",
    "unchanged_state_writes",
    "wakeup_count",
    "event_deliveries",
    "duplicate_committed_effects",
    "lost_deliveries",
)
REQUIRED_ZERO_WORK_METRICS = REQUIRED_IDLE_ZERO_METRICS[:4]
ACTIVITY_SENSORS = (
    {
        "metric": "task_board_scans",
        "sensor_id": "casf.idle.child.task-board-scans@1",
        "source": "closed_child_activity_counter:task_board_scan",
    },
    {
        "metric": "model_calls",
        "sensor_id": "casf.idle.child.model-calls@1",
        "source": "closed_child_activity_counter:model_call",
    },
    {
        "metric": "context_recompilations",
        "sensor_id": "casf.idle.child.context-recompilations@1",
        "source": "closed_child_activity_counter:context_recompile",
    },
    {
        "metric": "unchanged_state_writes",
        "sensor_id": "casf.idle.child.unchanged-state-writes@1",
        "source": "closed_child_activity_counter:unchanged_state_write",
    },
    {
        "metric": "wakeup_count",
        "sensor_id": "casf.idle.child.wakeup-count@1",
        "source": "closed_child_activity_counter:event_wait_wakeup",
    },
    {
        "metric": "event_deliveries",
        "sensor_id": "casf.idle.child.event-deliveries@1",
        "source": "closed_child_activity_counter:event_batch_delivery",
    },
    {
        "metric": "duplicate_committed_effects",
        "sensor_id": "casf.idle.child.duplicate-committed-effects@1",
        "source": "closed_child_activity_counter:duplicate_committed_effect",
    },
    {
        "metric": "lost_deliveries",
        "sensor_id": "casf.idle.child.lost-deliveries@1",
        "source": "closed_child_activity_counter:lost_delivery",
    },
)
PERMITTED_IDLE_ACTIVITY = (
    "one_bounded_owner_local_event_wait",
    "one_initial_in_memory_event_source_query",
    "parent_process_sensor_observation",
)
SPAN_KINDS = (
    "child_birth",
    "idle_wait_entered",
    "idle_wait_completed",
    "wake_wait_blocked",
    "injected_event_delivered",
)
TYPED_QUACK_UNAVAILABLE = {
    "availability": "unavailable",
    "execution_status": "not_run",
    "reason_code": "typed_quack_live_endpoint_not_supplied",
    "required_interface": "TypedStateOwnerEventWait@1",
    "required_client_interface": "QuackStateClientEventWait@1",
    "metrics_omitted": True,
}
OWNER_LOCAL_CAPABILITY = {
    "available": True,
    "interface": "StateOwnerEventWait@1",
    "transport": "owner_local_condition",
    "server_owned": True,
    "blocking_condition": True,
    "lost_wakeup_check_register_guard": True,
    "idle_repeated_database_scans": False,
    "remote_quack_transport_qualified": False,
    "qualification": "owner_local_hermetic_only",
}


class IdleBenchmarkError(ValueError):
    """Raised when a recipe, process observation, or replay is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdleBenchmarkError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise IdleBenchmarkError(f"invalid JSON in {name}: {exc.msg}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IdleBenchmarkError(f"cannot read {path}: {exc}") from exc
    decoded = _loads_json(raw, name=str(path))
    if not isinstance(decoded, dict):
        raise IdleBenchmarkError(f"{path} must contain a JSON object")
    return decoded


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise IdleBenchmarkError(f"cannot hash {path}: {exc}") from exc


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise IdleBenchmarkError(
            f"{name} has a closed schema "
            f"(unknown={sorted(actual - expected)}, missing={sorted(expected - actual)})"
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IdleBenchmarkError(f"{name} must be an object")
    return value


def _require_text(value: Any, name: str, *, maximum_bytes: int = 16_384) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise IdleBenchmarkError(f"{name} must be a bounded non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise IdleBenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise IdleBenchmarkError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _parse_timestamp(value: Any, name: str) -> datetime:
    text = _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IdleBenchmarkError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise IdleBenchmarkError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return _object_sha256(manifest)


def result_content_sha256(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload.pop("content_sha256", None)
    return _object_sha256(payload)


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path) if path is not None else Path(__file__).with_name("idle_manifest.json")
    manifest = _read_object(resolved)
    validate_manifest(manifest)
    return manifest


def _validate_activity_sensor_catalog(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(ACTIVITY_SENSORS):
        raise IdleBenchmarkError("activity sensor catalog is missing a required sensor")
    checked: list[dict[str, str]] = []
    for index, candidate in enumerate(value):
        sensor = _require_mapping(candidate, f"activity sensor {index}")
        _require_exact_keys(sensor, {"metric", "sensor_id", "source"}, f"activity sensor {index}")
        checked.append(
            {
                "metric": _require_text(sensor["metric"], f"activity sensor {index} metric"),
                "sensor_id": _require_text(sensor["sensor_id"], f"activity sensor {index} id"),
                "source": _require_text(sensor["source"], f"activity sensor {index} source"),
            }
        )
    if tuple(checked) != ACTIVITY_SENSORS:
        raise IdleBenchmarkError("activity sensor IDs or source descriptions have changed")
    return checked


def _validate_source_sha256(value: Any, name: str) -> dict[str, str]:
    source = _require_mapping(value, name)
    _require_exact_keys(source, set(SOURCE_RELATIVE_PATHS), name)
    return {
        relative_path: _require_sha256(source[relative_path], f"{name}.{relative_path}")
        for relative_path in SOURCE_RELATIVE_PATHS
    }


def _validate_imported_module_origins(value: Any, repository: Path, name: str) -> dict[str, str]:
    origins = _require_mapping(value, name)
    _require_exact_keys(origins, set(EXPECTED_MODULE_ORIGINS), name)
    checked = {
        module_name: _require_text(origins[module_name], f"{name}.{module_name}")
        for module_name in EXPECTED_MODULE_ORIGINS
    }
    expected = {
        module_name: str((repository / relative_path).resolve())
        for module_name, relative_path in EXPECTED_MODULE_ORIGINS.items()
    }
    if checked != expected:
        raise IdleBenchmarkError("imported measured-module origins differ from the repository")
    return checked


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise IdleBenchmarkError("manifest must be an object")
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
            "identity_requirements",
            "source_modules",
            "required_idle_zero_metrics",
            "activity_sensors",
            "permitted_idle_activity",
            "typed_quack_live",
            "result_storage",
            "nonclaims",
        },
        "manifest",
    )
    fixed = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "program_id": PROGRAM_ID,
        "objective_id": "CASF-038",
        "frozen": True,
        "state": "specification_only",
        "authoritative": False,
        "promotion_eligible": False,
    }
    if any(manifest[key] != value for key, value in fixed.items()):
        raise IdleBenchmarkError("manifest identity or non-authoritative state has changed")

    matrix = _require_mapping(manifest["matrix_binding"], "manifest matrix binding")
    expected_matrix = {
        "relative_path": MATRIX_RELATIVE_PATH,
        "schema": MATRIX_SCHEMA,
        "sha256": MATRIX_SHA256,
    }
    _require_exact_keys(matrix, set(expected_matrix), "manifest matrix binding")
    if dict(matrix) != expected_matrix:
        raise IdleBenchmarkError("manifest does not bind the exact frozen benchmark matrix")

    execution = _require_mapping(manifest["execution"], "manifest execution")
    expected_execution = {
        "measurement_scope": MEASUREMENT_SCOPE,
        "child_role": CHILD_ROLE,
        "configured_supervisor_population": 1,
        "idle_duration_seconds": 1,
        "wake_timeout_seconds": 5,
        "wait_interface": "StateOwnerEventWait@1",
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
    }
    _require_exact_keys(execution, set(expected_execution), "manifest execution")
    if dict(execution) != expected_execution:
        raise IdleBenchmarkError("manifest real-process execution recipe has changed")
    if tuple(manifest["identity_requirements"]) != REQUIRED_IDENTITIES:
        raise IdleBenchmarkError("manifest identity requirements have changed")
    if tuple(manifest["source_modules"]) != SOURCE_RELATIVE_PATHS:
        raise IdleBenchmarkError("manifest measured source-module set has changed")
    if tuple(manifest["required_idle_zero_metrics"]) != REQUIRED_IDLE_ZERO_METRICS:
        raise IdleBenchmarkError("manifest idle zero-metric contract has changed")
    _validate_activity_sensor_catalog(manifest["activity_sensors"])
    if tuple(manifest["permitted_idle_activity"]) != PERMITTED_IDLE_ACTIVITY:
        raise IdleBenchmarkError("manifest permitted idle activity has changed")
    live = _require_mapping(manifest["typed_quack_live"], "typed Quack live capability")
    _require_exact_keys(live, set(TYPED_QUACK_UNAVAILABLE), "typed Quack live capability")
    if dict(live) != TYPED_QUACK_UNAVAILABLE:
        raise IdleBenchmarkError("typed Quack live capability must remain unavailable/not-run")
    _require_text(manifest["result_storage"], "manifest result storage")
    if not isinstance(manifest["nonclaims"], list) or len(manifest["nonclaims"]) != 7:
        raise IdleBenchmarkError("manifest must preserve seven explicit nonclaims")
    for index, nonclaim in enumerate(manifest["nonclaims"]):
        _require_text(nonclaim, f"manifest nonclaims[{index}]")


def validate_matrix_binding(
    manifest: Mapping[str, Any], matrix_path: Path | str | None = None
) -> dict[str, str]:
    validate_manifest(manifest)
    resolved = (
        Path(matrix_path) if matrix_path is not None else Path(__file__).with_name("matrix.yaml")
    )
    digest = _file_sha256(resolved)
    if digest != MATRIX_SHA256:
        raise IdleBenchmarkError("frozen benchmark matrix content is stale or changed")
    try:
        first_line = resolved.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise IdleBenchmarkError("frozen benchmark matrix is missing its schema") from exc
    if first_line != f"schema: {MATRIX_SCHEMA}":
        raise IdleBenchmarkError("frozen benchmark matrix schema has changed")
    return dict(manifest["matrix_binding"])


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IdleBenchmarkError(f"cannot establish repository identity: {exc}") from exc
    return result.stdout.strip()


def repository_identity(repository: Path | str) -> dict[str, str]:
    root = Path(repository).resolve()
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise IdleBenchmarkError("repository is dirty; exact current-tree evidence is unavailable")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    _require_git_oid(commit, "repository commit")
    _require_git_oid(tree, "repository tree")
    return {"repository_commit": commit, "repository_tree": tree}


def _bound_repository_and_recipe(
    repository: Path | str,
    manifest_path: Path | str | None,
    matrix_path: Path | str | None,
) -> tuple[Path, Path, Path]:
    root = Path(repository).resolve()
    runner_root = Path(__file__).resolve().parents[3]
    if root != runner_root:
        raise IdleBenchmarkError(
            "repository must be the exact repository containing this benchmark runner"
        )
    expected_manifest = (
        root / "benchmarks/agent_supervisor/causal_event_federation/idle_manifest.json"
    )
    expected_matrix = root / MATRIX_RELATIVE_PATH
    resolved_manifest = (
        expected_manifest if manifest_path is None else Path(manifest_path).resolve()
    )
    resolved_matrix = expected_matrix if matrix_path is None else Path(matrix_path).resolve()
    if (
        resolved_manifest != expected_manifest.resolve()
        or resolved_matrix != expected_matrix.resolve()
    ):
        raise IdleBenchmarkError("manifest and matrix must come from the measured repository tree")
    return root, resolved_manifest, resolved_matrix


def _source_sha256(repository: Path) -> dict[str, str]:
    return {
        relative_path: _file_sha256(repository / relative_path)
        for relative_path in SOURCE_RELATIVE_PATHS
    }


def _require_git_oid(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise IdleBenchmarkError(f"{name} must be a full lowercase Git object id")
    return text


def _validate_identities(value: Any) -> dict[str, Any]:
    identities = _require_mapping(value, "benchmark identities")
    _require_exact_keys(identities, set(REQUIRED_IDENTITIES), "benchmark identities")
    checked = dict(identities)
    _require_git_oid(checked["repository_commit"], "identities.repository_commit")
    _require_git_oid(checked["repository_tree"], "identities.repository_tree")
    for key in ("control_plane_generation", "assignment_revision", "fencing_epoch"):
        _require_int(checked[key], f"identities.{key}", minimum=1)
    for key in REQUIRED_IDENTITIES:
        if key not in {
            "repository_commit",
            "repository_tree",
            "control_plane_generation",
            "assignment_revision",
            "fencing_epoch",
        }:
            _require_text(checked[key], f"identities.{key}")
    return checked


def _proc_start_ticks(pid: int) -> int:
    path = Path("/proc") / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
        tail = raw[raw.rindex(")") + 2 :].split()
        # Fields after the command start at proc field 3; starttime is field 22.
        value = int(tail[19])
    except (OSError, ValueError, IndexError) as exc:
        raise IdleBenchmarkError(f"cannot verify process birth for pid {pid}") from exc
    return _require_int(value, "process start ticks", minimum=1)


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="utf-8").strip().lower()
    except OSError as exc:
        raise IdleBenchmarkError("Linux boot-id sensor is unavailable") from exc
    if (
        len(value) != 36
        or any(value[index] != "-" for index in (8, 13, 18, 23))
        or any(
            character not in "0123456789abcdef"
            for index, character in enumerate(value)
            if index not in (8, 13, 18, 23)
        )
    ):
        raise IdleBenchmarkError("Linux boot-id sensor returned an invalid identifier")
    return value


def _runner_sha256() -> str:
    return _file_sha256(Path(__file__).resolve())


def _activity_sensor_catalog_sha256() -> str:
    return _object_sha256(list(ACTIVITY_SENSORS))


class _ActivityCounters:
    """Closed child-owned activity sensors; snapshots are evidence, not claims."""

    def __init__(self) -> None:
        self._values = dict.fromkeys(REQUIRED_IDLE_ZERO_METRICS, 0)

    def record(self, metric: str) -> None:
        if metric not in self._values:
            raise IdleBenchmarkError(f"activity counter names unknown metric {metric}")
        self._values[metric] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": ACTIVITY_SNAPSHOT_SCHEMA,
            "sensor_catalog_sha256": _activity_sensor_catalog_sha256(),
            "captured_monotonic_ns": time.monotonic_ns(),
            "values": dict(self._values),
        }


class _ProbeEventSource:
    """Instrumented in-memory source used only inside the hermetic child."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: list[Any] = []
        self.queries = 0

    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[Any, ...]:
        if (
            consumer_id != "consumer:idle-probe"
            or subscription_id != "subscription:idle-probe"
            or subscription_revision != 1
        ):
            raise IdleBenchmarkError("child wait identity changed")
        with self._condition:
            self.queries += 1
            available = tuple(
                event for event in self._events if event.global_sequence > after_cursor
            )[:maximum_events]
            self._condition.notify_all()
            return available

    def store_generation(self) -> int:
        return 1

    def append(self, event: Any) -> None:
        with self._condition:
            self._events.append(event)

    def wait_for_queries(self, minimum: int, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self.queries < minimum:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


def _child_span(
    *, sequence: int, kind: str, challenge_sha256: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": CHILD_SPAN_SCHEMA,
        "sequence": sequence,
        "kind": kind,
        "challenge_sha256": challenge_sha256,
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "boot_id": _boot_id(),
        "child_monotonic_ns": time.monotonic_ns(),
        "payload": dict(payload),
    }


def _emit_child_span(
    *, sequence: int, kind: str, challenge_sha256: str, payload: Mapping[str, Any]
) -> None:
    encoded = (
        _canonical_bytes(
            _child_span(
                sequence=sequence,
                kind=kind,
                challenge_sha256=challenge_sha256,
                payload=payload,
            )
        )
        + b"\n"
    )
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _deadline_after(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _read_child_injection(timeout_seconds: int) -> dict[str, Any]:
    ready, _, _ = __import__("select").select([sys.stdin.buffer], [], [], timeout_seconds)
    if not ready:
        raise IdleBenchmarkError("parent injection did not arrive before the bounded deadline")
    raw = sys.stdin.buffer.readline(MAX_JSON_LINE_BYTES + 1)
    if not raw or len(raw) > MAX_JSON_LINE_BYTES or not raw.endswith(b"\n"):
        raise IdleBenchmarkError("parent injection is missing or exceeds its bound")
    try:
        decoded = _loads_json(raw.decode("utf-8"), name="parent injection")
    except UnicodeDecodeError as exc:
        raise IdleBenchmarkError("parent injection is not UTF-8") from exc
    injection = _require_mapping(decoded, "parent injection")
    _require_exact_keys(injection, {"schema", "event_token"}, "parent injection")
    if injection["schema"] != INJECTION_SCHEMA:
        raise IdleBenchmarkError("parent injection schema changed")
    _require_text(injection["event_token"], "parent injection event token")
    return dict(injection)


def _child_probe(challenge: str, event_token: str, idle_seconds: int, wake_seconds: int) -> int:
    # Direct script execution places the benchmark directory on sys.path.
    repository_root = Path(__file__).resolve().parents[3]
    # Isolated mode plus an unconditional first entry prevents import shadowing.
    sys.path.insert(0, str(repository_root))

    from ipfs_accelerate_py.agent_supervisor.federation import (  # noqa: PLC0415
        event_wait as event_wait_module,
    )
    from ipfs_accelerate_py.agent_supervisor.federation import (  # noqa: PLC0415
        events as events_module,
    )
    from ipfs_accelerate_py.agent_supervisor.federation import (  # noqa: PLC0415
        outbox as outbox_module,
    )

    imported_origins: dict[str, str] = {}
    for module_name in EXPECTED_MODULE_ORIGINS:
        module = sys.modules.get(module_name)
        imported_origins[module_name] = str(Path(getattr(module, "__file__", "") or "").resolve())
    expected_origins = {
        module_name: str((repository_root / relative_path).resolve())
        for module_name, relative_path in EXPECTED_MODULE_ORIGINS.items()
    }
    if imported_origins != expected_origins:
        raise IdleBenchmarkError("child imported a measured module outside the repository tree")
    StateOwnerEventWait = event_wait_module.StateOwnerEventWait
    EventClass = events_module.EventClass
    EventEffectClass = events_module.EventEffectClass
    EventWaitRequest = events_module.EventWaitRequest
    EventDraft = outbox_module.EventDraft
    materialize_event = outbox_module.materialize_event

    challenge_digest = hashlib.sha256(challenge.encode("utf-8")).hexdigest()
    _emit_child_span(
        sequence=1,
        kind="child_birth",
        challenge_sha256=challenge_digest,
        payload={
            "role": CHILD_ROLE,
            "executable": str(Path(sys.executable).resolve()),
            "process_start_ticks": _proc_start_ticks(os.getpid()),
            "boot_id": _boot_id(),
            "runner_sha256": _runner_sha256(),
            "source_sha256": _source_sha256(repository_root),
            "imported_module_origins": imported_origins,
        },
    )

    source = _ProbeEventSource()
    event_wait = StateOwnerEventWait(source)
    capability = dict(event_wait.capability())
    capability.update({"available": True, "transport": "owner_local_condition"})
    if capability != OWNER_LOCAL_CAPABILITY:
        raise IdleBenchmarkError("owner-local event wait capability changed")
    activity = _ActivityCounters()

    _emit_child_span(
        sequence=2,
        kind="idle_wait_entered",
        challenge_sha256=challenge_digest,
        payload={
            "requested_duration_ns": idle_seconds * 1_000_000_000,
            "activity_snapshot_before": activity.snapshot(),
        },
    )
    idle_started = time.monotonic_ns()
    idle_batch = event_wait.wait_for_events(
        EventWaitRequest(
            consumer_id="consumer:idle-probe",
            after_cursor=0,
            subscription_id="subscription:idle-probe",
            subscription_revision=1,
            deadline=_deadline_after(idle_seconds),
            maximum_events=1,
        )
    )
    idle_elapsed = time.monotonic_ns() - idle_started
    if not idle_batch.timed_out or idle_batch.events or source.queries != 1:
        raise IdleBenchmarkError("bounded idle wait did not time out without delivery")
    _emit_child_span(
        sequence=3,
        kind="idle_wait_completed",
        challenge_sha256=challenge_digest,
        payload={
            "elapsed_ns": idle_elapsed,
            "timed_out": True,
            "event_source_queries": source.queries,
            "activity_snapshot_after": activity.snapshot(),
            "wait_capability": capability,
        },
    )

    wake_result: dict[str, Any] = {}
    wake_failure: list[BaseException] = []

    def wait_for_injected_event() -> None:
        try:
            wake_result["batch"] = event_wait.wait_for_events(
                EventWaitRequest(
                    consumer_id="consumer:idle-probe",
                    after_cursor=0,
                    subscription_id="subscription:idle-probe",
                    subscription_revision=1,
                    deadline=_deadline_after(wake_seconds),
                    maximum_events=1,
                )
            )
        except BaseException as exc:  # pragma: no cover - relayed by the child
            wake_failure.append(exc)

    waiter = threading.Thread(target=wait_for_injected_event, name="casf-idle-waiter")
    waiter.start()
    if not source.wait_for_queries(2, wake_seconds):
        raise IdleBenchmarkError("wake wait did not reach its blocking query")
    _emit_child_span(
        sequence=4,
        kind="wake_wait_blocked",
        challenge_sha256=challenge_digest,
        payload={
            "wake_timeout_ns": wake_seconds * 1_000_000_000,
            "event_source_queries_before_delivery": source.queries,
        },
    )
    injection = _read_child_injection(wake_seconds)
    if injection["event_token"] != event_token:
        raise IdleBenchmarkError("parent injection token differs from the child birth command")
    event, _outbox = materialize_event(
        EventDraft(
            event_type=EventClass.TASK_READY,
            stream_id="stream:idle-probe",
            causal_parent_ids=(),
            correlation_id="correlation:idle-probe",
            causation_id="causation:idle-probe",
            tenant_id="tenant:idle-probe",
            federation_id="federation:idle-probe",
            supervisor_id="supervisor:idle-probe",
            task_id="task:idle-probe",
            repository_id="repository:idle-probe",
            tree_id="tree:idle-probe",
            payload_ref=f"artifact:{event_token}",
            changed_fact_refs=("fact:idle-probe",),
            effect_class=EventEffectClass.READ_ONLY,
            deduplication_key=f"dedupe:{event_token}",
        ),
        stream_sequence=1,
        global_sequence=1,
        recorded_at=_utc_now(),
    )
    source.append(event)
    event_wait.notify_committed(1)
    waiter.join(wake_seconds)
    if waiter.is_alive():
        raise IdleBenchmarkError("injected wake did not finish before its deadline")
    if wake_failure:
        raise IdleBenchmarkError(f"injected wake failed: {type(wake_failure[0]).__name__}")
    batch = wake_result.get("batch")
    if (
        batch is None
        or len(batch.events) != 1
        or batch.events[0].event_id != event.event_id
        or batch.next_cursor != 1
        or source.queries != 3
        or event_wait.wakeup_count != 1
    ):
        raise IdleBenchmarkError("injected event was lost, duplicated, or misidentified")
    activity.record("wakeup_count")
    activity.record("event_deliveries")
    _emit_child_span(
        sequence=5,
        kind="injected_event_delivered",
        challenge_sha256=challenge_digest,
        payload={
            "injection_token_sha256": hashlib.sha256(event_token.encode("utf-8")).hexdigest(),
            "event_id": event.event_id,
            "batch_next_cursor": batch.next_cursor,
            "injected_events": 1,
            "delivered_events": len(batch.events),
            "event_source_queries": source.queries - 1,
            "wait_wakeups": event_wait.wakeup_count,
            "duplicate_deliveries": 0,
            "lost_deliveries": 0,
            "activity_snapshot_after": activity.snapshot(),
        },
    )
    return 0


def _read_child_line(stream: Any, timeout_seconds: float, name: str) -> bytes:
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        if not selector.select(timeout_seconds):
            raise IdleBenchmarkError(f"timed out waiting for child span {name}")
    raw = stream.readline(MAX_JSON_LINE_BYTES + 1)
    if not raw:
        raise IdleBenchmarkError(f"child exited before span {name}")
    if len(raw) > MAX_JSON_LINE_BYTES or not raw.endswith(b"\n"):
        raise IdleBenchmarkError(f"child span {name} exceeds its JSON-line bound")
    return raw


def _observe_child_span(
    raw: bytes,
    *,
    expected_sequence: int,
    expected_kind: str,
    challenge_sha256: str,
    child_pid: int,
    parent_pid: int,
    boot_id: str,
    previous_sha256: str,
) -> dict[str, Any]:
    try:
        decoded = _loads_json(raw.decode("utf-8"), name=f"child span {expected_kind}")
    except UnicodeDecodeError as exc:
        raise IdleBenchmarkError(f"child span {expected_kind} is not UTF-8") from exc
    child = _require_mapping(decoded, f"child span {expected_kind}")
    _require_exact_keys(
        child,
        {
            "schema",
            "sequence",
            "kind",
            "challenge_sha256",
            "child_pid",
            "parent_pid",
            "boot_id",
            "child_monotonic_ns",
            "payload",
        },
        f"child span {expected_kind}",
    )
    if (
        child["schema"] != CHILD_SPAN_SCHEMA
        or child["sequence"] != expected_sequence
        or child["kind"] != expected_kind
        or child["challenge_sha256"] != challenge_sha256
        or child["child_pid"] != child_pid
        or child["parent_pid"] != parent_pid
        or child["boot_id"] != boot_id
    ):
        raise IdleBenchmarkError(f"child span {expected_kind} has a false process identity")
    _require_int(child["child_monotonic_ns"], "child monotonic time", minimum=1)
    _require_mapping(child["payload"], f"child span {expected_kind} payload")
    wrapper = {
        "schema": PARENT_SPAN_SCHEMA,
        "sequence": expected_sequence,
        "parent_observed_monotonic_ns": time.monotonic_ns(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "previous_sha256": previous_sha256,
        "child_span": dict(child),
    }
    wrapper["span_sha256"] = _object_sha256(wrapper)
    return wrapper


def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def run_benchmark(
    *,
    repository: Path | str,
    identities: Mapping[str, Any],
    manifest_path: Path | str | None = None,
    matrix_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run and replay the frozen real-process hermetic baseline."""

    repository_root, resolved_manifest, resolved_matrix = _bound_repository_and_recipe(
        repository, manifest_path, matrix_path
    )
    manifest = load_manifest(resolved_manifest)
    matrix_binding = validate_matrix_binding(manifest, resolved_matrix)
    checked_identities = _validate_identities(identities)
    observed_repository = repository_identity(repository_root)
    observed_source = _source_sha256(repository_root)
    observed_origins = {
        module_name: str((repository_root / relative_path).resolve())
        for module_name, relative_path in EXPECTED_MODULE_ORIGINS.items()
    }
    for key, value in observed_repository.items():
        if checked_identities[key] != value:
            raise IdleBenchmarkError(f"benchmark identity is stale for {key}")

    execution = manifest["execution"]
    idle_seconds = int(execution["idle_duration_seconds"])
    wake_seconds = int(execution["wake_timeout_seconds"])
    challenge = secrets.token_hex(32)
    event_token = f"event:{secrets.token_hex(16)}"
    challenge_digest = hashlib.sha256(challenge.encode("utf-8")).hexdigest()
    runner_digest = _runner_sha256()
    boot_id = _boot_id()
    started_at = _utc_now()
    parent_pid = os.getpid()
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--_child-probe",
        "--_challenge",
        challenge,
        "--_event-token",
        event_token,
        "--_idle-seconds",
        str(idle_seconds),
        "--_wake-seconds",
        str(wake_seconds),
    ]
    child_environment = dict(os.environ)
    child_environment.pop("PYTHONPATH", None)
    child_environment.pop("PYTHONHOME", None)
    child_environment["PYTHONNOUSERSITE"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=str(Path(__file__).resolve().parents[3]),
            env=child_environment,
        )
    except OSError as exc:
        raise IdleBenchmarkError(f"cannot launch real-process idle probe: {exc}") from exc
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_child(process)
        raise IdleBenchmarkError("real-process idle probe pipes are unavailable")

    child_pid = process.pid
    process_start_ticks = _proc_start_ticks(child_pid)
    spans: list[dict[str, Any]] = []
    previous = FIRST_SPAN_PARENT
    try:
        timeouts = (5.0, 5.0, idle_seconds + 3.0, 3.0)
        for sequence, (kind, timeout_seconds) in enumerate(
            zip(SPAN_KINDS[:4], timeouts, strict=True), start=1
        ):
            raw = _read_child_line(process.stdout, timeout_seconds, kind)
            observed = _observe_child_span(
                raw,
                expected_sequence=sequence,
                expected_kind=kind,
                challenge_sha256=challenge_digest,
                child_pid=child_pid,
                parent_pid=parent_pid,
                boot_id=boot_id,
                previous_sha256=previous,
            )
            spans.append(observed)
            previous = observed["span_sha256"]
        birth = spans[0]["child_span"]["payload"]
        if (
            birth.get("process_start_ticks") != process_start_ticks
            or birth.get("boot_id") != boot_id
            or birth.get("runner_sha256") != runner_digest
            or birth.get("source_sha256") != observed_source
            or birth.get("imported_module_origins") != observed_origins
            or birth.get("executable") != str(Path(sys.executable).resolve())
        ):
            raise IdleBenchmarkError("child birth differs from parent-observed process identity")

        injection = (
            _canonical_bytes({"schema": INJECTION_SCHEMA, "event_token": event_token}) + b"\n"
        )
        process.stdin.write(injection)
        process.stdin.flush()
        process.stdin.close()
        raw = _read_child_line(process.stdout, wake_seconds + 3.0, "injected_event_delivered")
        observed = _observe_child_span(
            raw,
            expected_sequence=5,
            expected_kind="injected_event_delivered",
            challenge_sha256=challenge_digest,
            child_pid=child_pid,
            parent_pid=parent_pid,
            boot_id=boot_id,
            previous_sha256=previous,
        )
        spans.append(observed)
        previous = observed["span_sha256"]
        try:
            returncode = process.wait(timeout=3)
        except subprocess.TimeoutExpired as exc:
            raise IdleBenchmarkError("child did not exit after injected delivery") from exc
        stderr = process.stderr.read(MAX_JSON_LINE_BYTES + 1)
        trailing_stdout = process.stdout.read(MAX_JSON_LINE_BYTES + 1)
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:1_000]
            raise IdleBenchmarkError(f"child process failed with code {returncode}: {detail}")
        if stderr or trailing_stdout:
            raise IdleBenchmarkError("child emitted undeclared output outside the sensor log")
    except BaseException:
        _terminate_child(process)
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()

    post_repository = repository_identity(repository_root)
    post_source = _source_sha256(repository_root)
    if post_repository != observed_repository or post_source != observed_source:
        raise IdleBenchmarkError("repository identity or measured source changed during the run")
    finished_at = _utc_now()
    idle_payload = spans[2]["child_span"]["payload"]
    wake_payload = spans[4]["child_span"]["payload"]
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": manifest_sha256(manifest),
        "matrix_binding": matrix_binding,
        "content_sha256": "",
        "run_id": f"idle:{challenge_digest[:32]}",
        "started_at": started_at,
        "finished_at": finished_at,
        "execution_status": "measured",
        "availability": "available",
        "measurement_scope": MEASUREMENT_SCOPE,
        "configured_supervisor_population": 1,
        "evidence_replay_verified": True,
        "authoritative": False,
        "promotion_eligible": False,
        "identities": checked_identities,
        "identity_assurance": {
            "repository_commit_and_tree": "parent_git_observed_clean_tree",
            "remaining_context": "caller_supplied_not_state_owner_attested",
        },
        "activity_sensors": [dict(sensor) for sensor in ACTIVITY_SENSORS],
        "source_evidence": {
            "repository_root": str(repository_root),
            "source_sha256": observed_source,
            "imported_module_origins": observed_origins,
            "observed_before_and_after": True,
        },
        "process_evidence": {
            "child_role": CHILD_ROLE,
            "probe_processes_observed": 1,
            "child_pid": child_pid,
            "parent_pid": parent_pid,
            "process_start_ticks": process_start_ticks,
            "boot_id": boot_id,
            "birth_verified_by_parent": True,
            "challenge_sha256": challenge_digest,
            "executable": str(Path(sys.executable).resolve()),
            "runner_sha256": runner_digest,
            "source_sha256": observed_source,
            "imported_module_origins": observed_origins,
            "returncode": 0,
        },
        "wait_capability": dict(OWNER_LOCAL_CAPABILITY),
        "idle_measurement": {
            "bounded_waits": 1,
            "requested_duration_ns": idle_seconds * 1_000_000_000,
            "measured_duration_ns": idle_payload["elapsed_ns"],
            "event_source_queries": idle_payload["event_source_queries"],
            "activity_snapshot": dict(idle_payload["activity_snapshot_after"]),
            "metrics": dict(idle_payload["activity_snapshot_after"]["values"]),
        },
        "wake_verification": {
            "injection_token_sha256": hashlib.sha256(event_token.encode("utf-8")).hexdigest(),
            "event_id": wake_payload["event_id"],
            "injected_events": wake_payload["injected_events"],
            "delivered_events": wake_payload["delivered_events"],
            "event_source_queries": wake_payload["event_source_queries"],
            "wait_wakeups": wake_payload["wait_wakeups"],
            "duplicate_deliveries": wake_payload["duplicate_deliveries"],
            "lost_deliveries": wake_payload["lost_deliveries"],
            "activity_snapshot": dict(wake_payload["activity_snapshot_after"]),
            "zero_work_metrics_after": {
                key: wake_payload["activity_snapshot_after"]["values"][key]
                for key in REQUIRED_ZERO_WORK_METRICS
            },
        },
        "typed_quack_live": dict(TYPED_QUACK_UNAVAILABLE),
        "sensor_evidence": {
            "schema": SENSOR_SCHEMA,
            "hash_algorithm": "sha256",
            "spans": spans,
            "terminal_sha256": previous,
        },
        "nonclaims": list(manifest["nonclaims"]),
    }
    result["content_sha256"] = result_content_sha256(result)
    validate_result(
        result,
        manifest,
        matrix_path=resolved_matrix,
        current_identity=post_repository,
    )
    return result


def _validate_zero_metrics(value: Any, expected: tuple[str, ...], name: str) -> dict[str, int]:
    metrics = _require_mapping(value, name)
    _require_exact_keys(metrics, set(expected), name)
    checked: dict[str, int] = {}
    for key in expected:
        checked[key] = _require_int(metrics[key], f"{name}.{key}")
        if checked[key] != 0:
            raise IdleBenchmarkError(f"{name}.{key} must be a measured zero")
    return checked


def _validate_activity_snapshot(value: Any, name: str) -> dict[str, Any]:
    snapshot = _require_mapping(value, name)
    _require_exact_keys(
        snapshot,
        {"schema", "sensor_catalog_sha256", "captured_monotonic_ns", "values"},
        name,
    )
    if snapshot["schema"] != ACTIVITY_SNAPSHOT_SCHEMA:
        raise IdleBenchmarkError(f"{name} schema changed")
    if snapshot["sensor_catalog_sha256"] != _activity_sensor_catalog_sha256():
        raise IdleBenchmarkError(f"{name} has missing or changed sensor provenance")
    _require_int(snapshot["captured_monotonic_ns"], f"{name} capture time", minimum=1)
    values = _require_mapping(snapshot["values"], f"{name} values")
    _require_exact_keys(values, set(REQUIRED_IDLE_ZERO_METRICS), f"{name} values")
    for metric in REQUIRED_IDLE_ZERO_METRICS:
        _require_int(values[metric], f"{name} values.{metric}")
    return dict(snapshot)


def _validate_child_payload(kind: str, value: Any) -> dict[str, Any]:
    payload = _require_mapping(value, f"{kind} payload")
    expected_keys = {
        "child_birth": {
            "role",
            "executable",
            "process_start_ticks",
            "boot_id",
            "runner_sha256",
            "source_sha256",
            "imported_module_origins",
        },
        "idle_wait_entered": {"requested_duration_ns", "activity_snapshot_before"},
        "idle_wait_completed": {
            "elapsed_ns",
            "timed_out",
            "event_source_queries",
            "activity_snapshot_after",
            "wait_capability",
        },
        "wake_wait_blocked": {"wake_timeout_ns", "event_source_queries_before_delivery"},
        "injected_event_delivered": {
            "injection_token_sha256",
            "event_id",
            "batch_next_cursor",
            "injected_events",
            "delivered_events",
            "event_source_queries",
            "wait_wakeups",
            "duplicate_deliveries",
            "lost_deliveries",
            "activity_snapshot_after",
        },
    }[kind]
    _require_exact_keys(payload, expected_keys, f"{kind} payload")
    checked = dict(payload)
    if kind == "child_birth":
        if checked["role"] != CHILD_ROLE:
            raise IdleBenchmarkError("child birth role changed")
        _require_text(checked["executable"], "child executable")
        _require_int(checked["process_start_ticks"], "child process start ticks", minimum=1)
        if checked["boot_id"] != _boot_id():
            raise IdleBenchmarkError("child birth boot ID differs from the current Linux boot")
        _require_sha256(checked["runner_sha256"], "child runner SHA-256")
        _validate_source_sha256(checked["source_sha256"], "child measured source SHA-256")
        origins = _require_mapping(
            checked["imported_module_origins"], "child imported module origins"
        )
        _require_exact_keys(origins, set(EXPECTED_MODULE_ORIGINS), "child imported module origins")
        for module_name in EXPECTED_MODULE_ORIGINS:
            _require_text(origins[module_name], f"child imported module origin {module_name}")
    elif kind == "idle_wait_entered":
        _require_int(checked["requested_duration_ns"], "requested idle duration", minimum=1)
        snapshot = _validate_activity_snapshot(
            checked["activity_snapshot_before"], "idle activity snapshot before"
        )
        _validate_zero_metrics(
            snapshot["values"], REQUIRED_IDLE_ZERO_METRICS, "idle metrics before"
        )
    elif kind == "idle_wait_completed":
        _require_int(checked["elapsed_ns"], "measured idle duration", minimum=1)
        if checked["timed_out"] is not True or checked["event_source_queries"] != 1:
            raise IdleBenchmarkError("idle wait was not one bounded no-event source query")
        snapshot = _validate_activity_snapshot(
            checked["activity_snapshot_after"], "idle activity snapshot after"
        )
        _validate_zero_metrics(snapshot["values"], REQUIRED_IDLE_ZERO_METRICS, "idle metrics after")
        capability = _require_mapping(checked["wait_capability"], "owner-local capability")
        _require_exact_keys(capability, set(OWNER_LOCAL_CAPABILITY), "owner-local capability")
        if dict(capability) != OWNER_LOCAL_CAPABILITY:
            raise IdleBenchmarkError("owner-local capability is mislabeled or changed")
    elif kind == "wake_wait_blocked":
        _require_int(checked["wake_timeout_ns"], "wake timeout", minimum=1)
        if checked["event_source_queries_before_delivery"] != 2:
            raise IdleBenchmarkError("wake path was not observed blocked before injection")
    else:
        _require_sha256(checked["injection_token_sha256"], "injection token SHA-256")
        _require_text(checked["event_id"], "injected event id")
        expected_numbers = {
            "batch_next_cursor": 1,
            "injected_events": 1,
            "delivered_events": 1,
            "event_source_queries": 2,
            "wait_wakeups": 1,
            "duplicate_deliveries": 0,
            "lost_deliveries": 0,
        }
        if any(checked[key] != item for key, item in expected_numbers.items()):
            raise IdleBenchmarkError("injected event was lost, duplicated, or not wake-delivered")
        snapshot = _validate_activity_snapshot(
            checked["activity_snapshot_after"], "wake activity snapshot after"
        )
        expected_activity = dict.fromkeys(REQUIRED_IDLE_ZERO_METRICS, 0)
        expected_activity["wakeup_count"] = 1
        expected_activity["event_deliveries"] = 1
        if snapshot["values"] != expected_activity:
            raise IdleBenchmarkError("wake activity sensors differ from the observed delivery")
    return checked


def _validate_sensor_evidence(
    value: Any, process_evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    evidence = _require_mapping(value, "sensor evidence")
    _require_exact_keys(
        evidence, {"schema", "hash_algorithm", "spans", "terminal_sha256"}, "sensor evidence"
    )
    if evidence["schema"] != SENSOR_SCHEMA or evidence["hash_algorithm"] != "sha256":
        raise IdleBenchmarkError("sensor evidence identity changed")
    spans = evidence["spans"]
    if not isinstance(spans, list) or len(spans) != len(SPAN_KINDS):
        raise IdleBenchmarkError("sensor evidence is missing a required process span")
    previous = FIRST_SPAN_PARENT
    checked: list[dict[str, Any]] = []
    prior_child_time = 0
    prior_parent_time = 0
    for sequence, (kind, candidate) in enumerate(zip(SPAN_KINDS, spans, strict=True), start=1):
        span = _require_mapping(candidate, f"parent span {kind}")
        _require_exact_keys(
            span,
            {
                "schema",
                "sequence",
                "parent_observed_monotonic_ns",
                "raw_sha256",
                "previous_sha256",
                "child_span",
                "span_sha256",
            },
            f"parent span {kind}",
        )
        if (
            span["schema"] != PARENT_SPAN_SCHEMA
            or span["sequence"] != sequence
            or span["previous_sha256"] != previous
        ):
            raise IdleBenchmarkError(f"parent span {kind} ordering or chain parent changed")
        parent_time = _require_int(
            span["parent_observed_monotonic_ns"], f"parent span {kind} time", minimum=1
        )
        if parent_time < prior_parent_time:
            raise IdleBenchmarkError("parent sensor time moved backwards")
        child = _require_mapping(span["child_span"], f"child span {kind}")
        _require_exact_keys(
            child,
            {
                "schema",
                "sequence",
                "kind",
                "challenge_sha256",
                "child_pid",
                "parent_pid",
                "boot_id",
                "child_monotonic_ns",
                "payload",
            },
            f"child span {kind}",
        )
        if (
            child["schema"] != CHILD_SPAN_SCHEMA
            or child["sequence"] != sequence
            or child["kind"] != kind
            or child["challenge_sha256"] != process_evidence["challenge_sha256"]
            or child["child_pid"] != process_evidence["child_pid"]
            or child["parent_pid"] != process_evidence["parent_pid"]
            or child["boot_id"] != process_evidence["boot_id"]
        ):
            raise IdleBenchmarkError(f"child span {kind} has false birth or process identity")
        child_time = _require_int(child["child_monotonic_ns"], f"child span {kind} time", minimum=1)
        if child_time < prior_child_time:
            raise IdleBenchmarkError("child sensor time moved backwards")
        _validate_child_payload(kind, child["payload"])
        canonical_raw = _canonical_bytes(child) + b"\n"
        if span["raw_sha256"] != hashlib.sha256(canonical_raw).hexdigest():
            raise IdleBenchmarkError(f"parent raw sensor hash changed for {kind}")
        _require_sha256(span["raw_sha256"], f"parent span {kind} raw SHA-256")
        _require_sha256(span["span_sha256"], f"parent span {kind} SHA-256")
        hash_payload = dict(span)
        claimed_hash = hash_payload.pop("span_sha256")
        if claimed_hash != _object_sha256(hash_payload):
            raise IdleBenchmarkError(f"parent span {kind} hash is not replayable")
        previous = claimed_hash
        prior_child_time = child_time
        prior_parent_time = parent_time
        checked.append(dict(span))
    if evidence["terminal_sha256"] != previous:
        raise IdleBenchmarkError("sensor terminal hash does not close the span chain")

    requested_ns = int(manifest["execution"]["idle_duration_seconds"]) * 1_000_000_000
    entered = checked[1]
    completed = checked[2]
    child_elapsed = completed["child_span"]["payload"]["elapsed_ns"]
    parent_elapsed = (
        completed["parent_observed_monotonic_ns"] - entered["parent_observed_monotonic_ns"]
    )
    # Permit only scheduler/timer rounding below the exact one-second deadline.
    lower_bound = requested_ns - 25_000_000
    if child_elapsed < lower_bound or parent_elapsed < lower_bound:
        raise IdleBenchmarkError("idle sensor span is shorter than the bounded wait recipe")
    return checked


def validate_result(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    matrix_path: Path | str | None = None,
    current_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_manifest = dict(manifest) if manifest is not None else load_manifest()
    validate_manifest(checked_manifest)
    validate_matrix_binding(checked_manifest, matrix_path)
    if not isinstance(result, Mapping):
        raise IdleBenchmarkError("benchmark result must be an object")
    _require_exact_keys(
        result,
        {
            "schema",
            "benchmark_id",
            "manifest_sha256",
            "matrix_binding",
            "content_sha256",
            "run_id",
            "started_at",
            "finished_at",
            "execution_status",
            "availability",
            "measurement_scope",
            "configured_supervisor_population",
            "evidence_replay_verified",
            "authoritative",
            "promotion_eligible",
            "identities",
            "identity_assurance",
            "activity_sensors",
            "source_evidence",
            "process_evidence",
            "wait_capability",
            "idle_measurement",
            "wake_verification",
            "typed_quack_live",
            "sensor_evidence",
            "nonclaims",
        },
        "benchmark result",
    )
    fixed = {
        "schema": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": manifest_sha256(checked_manifest),
        "matrix_binding": dict(checked_manifest["matrix_binding"]),
        "execution_status": "measured",
        "availability": "available",
        "measurement_scope": MEASUREMENT_SCOPE,
        "configured_supervisor_population": 1,
        "evidence_replay_verified": True,
        "authoritative": False,
        "promotion_eligible": False,
        "typed_quack_live": TYPED_QUACK_UNAVAILABLE,
        "nonclaims": list(checked_manifest["nonclaims"]),
    }
    if any(result[key] != value for key, value in fixed.items()):
        raise IdleBenchmarkError("result scope, availability, or non-authoritative state changed")
    _require_text(result["run_id"], "run id")
    started = _parse_timestamp(result["started_at"], "result started_at")
    finished = _parse_timestamp(result["finished_at"], "result finished_at")
    if finished <= started:
        raise IdleBenchmarkError("result wall-clock window is empty")
    _require_sha256(result["content_sha256"], "result content SHA-256")
    if result["content_sha256"] != result_content_sha256(result):
        raise IdleBenchmarkError("result content address is stale or changed")

    identities = _validate_identities(result["identities"])
    if current_identity is None:
        raise IdleBenchmarkError("current identity is required for result replay")
    current = _require_mapping(current_identity, "current identity")
    _require_exact_keys(current, {"repository_commit", "repository_tree"}, "current identity")
    for key, value in current.items():
        if key not in identities:
            raise IdleBenchmarkError(f"current identity names unknown field {key}")
        if identities[key] != value:
            raise IdleBenchmarkError(f"benchmark result is stale for {key}")
    assurance = _require_mapping(result["identity_assurance"], "identity assurance")
    expected_assurance = {
        "repository_commit_and_tree": "parent_git_observed_clean_tree",
        "remaining_context": "caller_supplied_not_state_owner_attested",
    }
    _require_exact_keys(assurance, set(expected_assurance), "identity assurance")
    if dict(assurance) != expected_assurance:
        raise IdleBenchmarkError("identity assurance overstates the observed context")
    _validate_activity_sensor_catalog(result["activity_sensors"])

    source_evidence = _require_mapping(result["source_evidence"], "source evidence")
    _require_exact_keys(
        source_evidence,
        {
            "repository_root",
            "source_sha256",
            "imported_module_origins",
            "observed_before_and_after",
        },
        "source evidence",
    )
    source_root = Path(_require_text(source_evidence["repository_root"], "source repository root"))
    if not source_root.is_absolute() or source_evidence["observed_before_and_after"] is not True:
        raise IdleBenchmarkError("source evidence lacks an exact twice-observed repository root")
    source_hashes = _validate_source_sha256(
        source_evidence["source_sha256"], "source evidence SHA-256"
    )
    imported_origins = _validate_imported_module_origins(
        source_evidence["imported_module_origins"], source_root, "source imported module origins"
    )
    if repository_identity(source_root) != dict(current):
        raise IdleBenchmarkError("source repository identity is stale at replay")
    if _source_sha256(source_root) != source_hashes:
        raise IdleBenchmarkError("measured source hashes are stale at replay")

    process = _require_mapping(result["process_evidence"], "process evidence")
    _require_exact_keys(
        process,
        {
            "child_role",
            "probe_processes_observed",
            "child_pid",
            "parent_pid",
            "process_start_ticks",
            "boot_id",
            "birth_verified_by_parent",
            "challenge_sha256",
            "executable",
            "runner_sha256",
            "source_sha256",
            "imported_module_origins",
            "returncode",
        },
        "process evidence",
    )
    if (
        process["child_role"] != CHILD_ROLE
        or process["probe_processes_observed"] != 1
        or process["birth_verified_by_parent"] is not True
        or process["returncode"] != 0
    ):
        raise IdleBenchmarkError("result does not prove one successfully observed child birth")
    for key in ("child_pid", "parent_pid", "process_start_ticks"):
        _require_int(process[key], f"process evidence {key}", minimum=1)
    _require_sha256(process["challenge_sha256"], "process challenge SHA-256")
    _require_sha256(process["runner_sha256"], "process runner SHA-256")
    if process["runner_sha256"] != _runner_sha256():
        raise IdleBenchmarkError("result was measured by a different runner source")
    if process["boot_id"] != _boot_id():
        raise IdleBenchmarkError("result process birth belongs to a different Linux boot")
    if _validate_source_sha256(process["source_sha256"], "process source SHA-256") != source_hashes:
        raise IdleBenchmarkError("process source hashes differ from parent source evidence")
    if (
        _validate_imported_module_origins(
            process["imported_module_origins"], source_root, "process imported module origins"
        )
        != imported_origins
    ):
        raise IdleBenchmarkError("process imported-module origins differ from parent evidence")
    _require_text(process["executable"], "process executable")

    capability = _require_mapping(result["wait_capability"], "result wait capability")
    _require_exact_keys(capability, set(OWNER_LOCAL_CAPABILITY), "result wait capability")
    if dict(capability) != OWNER_LOCAL_CAPABILITY:
        raise IdleBenchmarkError("result falsely labels the owner-local wait capability")
    spans = _validate_sensor_evidence(result["sensor_evidence"], process, checked_manifest)
    birth_payload = spans[0]["child_span"]["payload"]
    if (
        birth_payload["process_start_ticks"] != process["process_start_ticks"]
        or birth_payload["boot_id"] != process["boot_id"]
        or birth_payload["runner_sha256"] != process["runner_sha256"]
        or birth_payload["source_sha256"] != process["source_sha256"]
        or birth_payload["imported_module_origins"] != process["imported_module_origins"]
        or birth_payload["executable"] != process["executable"]
    ):
        raise IdleBenchmarkError("process evidence differs from the observed birth span")

    idle = _require_mapping(result["idle_measurement"], "idle measurement")
    _require_exact_keys(
        idle,
        {
            "bounded_waits",
            "requested_duration_ns",
            "measured_duration_ns",
            "event_source_queries",
            "activity_snapshot",
            "metrics",
        },
        "idle measurement",
    )
    idle_span = spans[2]["child_span"]["payload"]
    expected_idle = {
        "bounded_waits": 1,
        "requested_duration_ns": int(checked_manifest["execution"]["idle_duration_seconds"])
        * 1_000_000_000,
        "measured_duration_ns": idle_span["elapsed_ns"],
        "event_source_queries": 1,
        "activity_snapshot": dict(idle_span["activity_snapshot_after"]),
        "metrics": dict(idle_span["activity_snapshot_after"]["values"]),
    }
    if dict(idle) != expected_idle:
        raise IdleBenchmarkError("idle measurement differs from its replayable sensor span")
    _validate_zero_metrics(idle["metrics"], REQUIRED_IDLE_ZERO_METRICS, "idle result metrics")

    wake = _require_mapping(result["wake_verification"], "wake verification")
    _require_exact_keys(
        wake,
        {
            "injection_token_sha256",
            "event_id",
            "injected_events",
            "delivered_events",
            "event_source_queries",
            "wait_wakeups",
            "duplicate_deliveries",
            "lost_deliveries",
            "activity_snapshot",
            "zero_work_metrics_after",
        },
        "wake verification",
    )
    wake_span = spans[4]["child_span"]["payload"]
    expected_wake = {
        key: wake_span[key]
        for key in wake
        if key not in {"activity_snapshot", "zero_work_metrics_after"}
    }
    expected_wake["activity_snapshot"] = dict(wake_span["activity_snapshot_after"])
    expected_wake["zero_work_metrics_after"] = {
        key: wake_span["activity_snapshot_after"]["values"][key]
        for key in REQUIRED_ZERO_WORK_METRICS
    }
    if dict(wake) != expected_wake:
        raise IdleBenchmarkError("wake verification differs from its replayable sensor span")
    _validate_zero_metrics(
        wake["zero_work_metrics_after"], REQUIRED_ZERO_WORK_METRICS, "wake result metrics"
    )
    return dict(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, help="clean repository identity to bind")
    parser.add_argument("--identities", type=Path, help="closed benchmark context identity JSON")
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("idle_manifest.json")
    )
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matrix.yaml"))
    parser.add_argument("--_child-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_challenge", help=argparse.SUPPRESS)
    parser.add_argument("--_event-token", help=argparse.SUPPRESS)
    parser.add_argument("--_idle-seconds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_wake-seconds", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._child_probe:
        try:
            if (
                not args._challenge
                or not args._event_token
                or args._idle_seconds != 1
                or args._wake_seconds != 5
            ):
                raise IdleBenchmarkError("child command differs from the frozen recipe")
            return _child_probe(
                args._challenge,
                args._event_token,
                args._idle_seconds,
                args._wake_seconds,
            )
        except BaseException as exc:  # pragma: no cover - parent surfaces the failure
            error = {
                "schema": ERROR_SCHEMA,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1_000],
            }
            sys.stderr.buffer.write(_canonical_bytes(error) + b"\n")
            sys.stderr.buffer.flush()
            return 2
    try:
        if args.repository is None or args.identities is None:
            raise IdleBenchmarkError("--repository and --identities are required")
        result = run_benchmark(
            repository=args.repository,
            identities=_read_object(args.identities),
            manifest_path=args.manifest,
            matrix_path=args.matrix,
        )
    except IdleBenchmarkError as exc:
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "execution_status": "invalid", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
