#!/usr/bin/env python3
"""Fail-closed verifier for the CASF event-driven idle benchmark.

The runner deliberately verifies a compact observation emitted by a live typed
state-owner benchmark.  It does not open ``control.duckdb``, create a fallback
transport, start supervisors, or infer authority from DuckLake.  Those actions
belong to the existing typed state-owner and supervisor runtime boundaries.

An accepted report is still qualification *observation* only.  In particular,
it cannot promote a federation or complete a task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/"
    "idle-benchmark-manifest@1"
)
OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/"
    "idle-benchmark-observation@1"
)
REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation/"
    "idle-benchmark-report@1"
)
BENCHMARK_ID = "casf-idle-event-driven-v1"
PROGRAM_ID = "agent-supervisor-causal-event-federation-v1"
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
REQUIRED_ZERO_METRICS = (
    "task_board_scans",
    "model_calls",
    "context_recompilations",
    "unchanged_state_writes",
    "wakeup_count",
    "event_deliveries",
    "duplicate_committed_effects",
    "lost_deliveries",
)
PERMITTED_ACTIVITY = frozenset(
    {
        "blocked_server_owned_event_wait",
        "required_lease_heartbeat",
        "bounded_health_deadline",
        "explicit_recovery_timer",
    }
)


class IdleBenchmarkError(ValueError):
    """Raised when benchmark inputs cannot prove the closed idle contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdleBenchmarkError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_object(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except OSError as exc:
        raise IdleBenchmarkError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IdleBenchmarkError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise IdleBenchmarkError(f"{path} must contain a JSON object")
    return decoded


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise IdleBenchmarkError(
            f"{name} has a closed schema (unknown={unknown}, missing={missing})"
        )


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 16_384:
        raise IdleBenchmarkError(f"{name} must be a bounded non-empty string")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise IdleBenchmarkError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise IdleBenchmarkError(f"{name} must be a non-negative integer")
    return value


def _parse_timestamp(value: Any, name: str) -> datetime:
    text = _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IdleBenchmarkError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise IdleBenchmarkError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical identity for a validated manifest object."""

    validate_manifest(manifest)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    """Load the frozen manifest while rejecting duplicate or unknown fields."""

    resolved = Path(path) if path is not None else Path(__file__).with_name("idle_manifest.json")
    manifest = _read_object(resolved)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the immutable benchmark recipe before considering evidence."""

    if not isinstance(manifest, Mapping):
        raise IdleBenchmarkError("manifest must be an object")
    _require_exact_keys(
        manifest,
        {
            "schema", "benchmark_id", "program_id", "objective_id", "frozen", "state",
            "authoritative", "promotion_eligible", "execution", "identity_requirements",
            "idle_window", "required_zero_metrics", "permitted_activity", "result_storage",
            "nonclaims",
        },
        "manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["benchmark_id"] != BENCHMARK_ID:
        raise IdleBenchmarkError("manifest identity does not match the idle benchmark")
    if manifest["program_id"] != PROGRAM_ID or manifest["objective_id"] != "CASF-038":
        raise IdleBenchmarkError("manifest program or objective identity does not match")
    if manifest["frozen"] is not True or manifest["state"] != "specification_only":
        raise IdleBenchmarkError("manifest must remain a frozen specification")
    if manifest["authoritative"] is not False or manifest["promotion_eligible"] is not False:
        raise IdleBenchmarkError("benchmark manifest may not claim authority or promotion")

    execution = manifest["execution"]
    if not isinstance(execution, Mapping):
        raise IdleBenchmarkError("manifest execution must be an object")
    _require_exact_keys(
        execution,
        {
            "required_measurement_mode", "required_wait_interface", "required_client_interface",
            "network_permitted", "direct_database_access_permitted",
            "ducklake_scheduling_authority_permitted",
        },
        "manifest execution",
    )
    required_execution = {
        "required_measurement_mode": "live",
        "required_wait_interface": "TypedStateOwnerEventWait@1",
        "required_client_interface": "QuackStateClientEventWait@1",
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
    }
    if dict(execution) != required_execution:
        raise IdleBenchmarkError("manifest execution controls have changed")
    if tuple(manifest["identity_requirements"]) != REQUIRED_IDENTITIES:
        raise IdleBenchmarkError("manifest identity requirements have changed")
    if tuple(manifest["required_zero_metrics"]) != REQUIRED_ZERO_METRICS:
        raise IdleBenchmarkError("manifest zero-metric contract has changed")
    if set(manifest["permitted_activity"]) != PERMITTED_ACTIVITY or len(manifest["permitted_activity"]) != 4:
        raise IdleBenchmarkError("manifest permitted activity has changed")
    window = manifest["idle_window"]
    if not isinstance(window, Mapping):
        raise IdleBenchmarkError("manifest idle_window must be an object")
    _require_exact_keys(
        window,
        {"minimum_duration_seconds", "minimum_completed_waits", "maximum_event_deliveries"},
        "manifest idle_window",
    )
    if (
        _require_positive_int(window["minimum_duration_seconds"], "minimum_duration_seconds") != 60
        or _require_positive_int(window["minimum_completed_waits"], "minimum_completed_waits") != 1
        or _require_nonnegative_int(window["maximum_event_deliveries"], "maximum_event_deliveries") != 0
    ):
        raise IdleBenchmarkError("manifest idle window has changed")
    if not isinstance(manifest["nonclaims"], list) or len(manifest["nonclaims"]) != 3:
        raise IdleBenchmarkError("manifest must preserve its three nonclaims")
    for name in ("result_storage", *manifest["nonclaims"]):
        _require_text(name, "manifest text")


def _validate_capability(capability: Any) -> dict[str, Any]:
    if not isinstance(capability, Mapping):
        raise IdleBenchmarkError("event wait capability is missing")
    required = {
        "available", "interface", "client_interface", "transport", "server_owned",
        "blocking_condition", "adaptive_polling", "event_driven_qualified",
        "idle_repeated_database_scans",
    }
    _require_exact_keys(capability, required, "event wait capability")
    expected = {
        "available": True,
        "interface": "TypedStateOwnerEventWait@1",
        "client_interface": "QuackStateClientEventWait@1",
        "transport": "typed_state_owner_bounded_long_wait",
        "server_owned": True,
        "blocking_condition": True,
        "adaptive_polling": False,
        "event_driven_qualified": True,
        "idle_repeated_database_scans": False,
    }
    if dict(capability) != expected:
        raise IdleBenchmarkError("event wait capability is not the qualified typed wait")
    return dict(capability)


def _validate_observation_shape(observation: Mapping[str, Any]) -> None:
    _require_exact_keys(
        observation,
        {
            "schema", "benchmark_id", "manifest_sha256", "run_id", "started_at", "finished_at",
            "measurement_mode", "identities", "wait_capability", "activity", "metrics",
            "measurement_source", "simulation", "direct_database_access", "network_used",
            "ducklake_scheduling_authority", "promotion_eligible",
        },
        "observation",
    )


def _validate_identities(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise IdleBenchmarkError("observation identities must be an object")
    _require_exact_keys(value, set(REQUIRED_IDENTITIES), "observation identities")
    identities = dict(value)
    for key in REQUIRED_IDENTITIES:
        if key in {"control_plane_generation", "assignment_revision", "fencing_epoch"}:
            _require_positive_int(identities[key], f"identities.{key}")
        else:
            _require_text(identities[key], f"identities.{key}")
    return identities


def _validate_activity(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise IdleBenchmarkError("observation activity must be an object")
    _require_exact_keys(value, PERMITTED_ACTIVITY, "observation activity")
    activity = {str(key): _require_nonnegative_int(item, f"activity.{key}") for key, item in value.items()}
    if activity["blocked_server_owned_event_wait"] < 1:
        raise IdleBenchmarkError("idle observation must include a completed server-owned wait")
    return activity


def _validate_metrics(value: Any, *, wait_count: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise IdleBenchmarkError("observation metrics must be an object")
    expected = {*REQUIRED_ZERO_METRICS, "completed_waits", "server_wait_queries"}
    _require_exact_keys(value, expected, "observation metrics")
    metrics = {str(key): _require_nonnegative_int(item, f"metrics.{key}") for key, item in value.items()}
    for key in REQUIRED_ZERO_METRICS:
        if metrics[key] != 0:
            raise IdleBenchmarkError(f"idle benchmark requires metrics.{key} == 0")
    if metrics["completed_waits"] < 1 or metrics["completed_waits"] != wait_count:
        raise IdleBenchmarkError("completed wait count is missing or inconsistent")
    if metrics["server_wait_queries"] < 1 or metrics["server_wait_queries"] > metrics["completed_waits"]:
        raise IdleBenchmarkError("server wait query count is not bounded by completed waits")
    return metrics


def validate_observation(
    observation: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    current_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact non-authoritative report or fail closed.

    ``current_identity`` is supplied by the live launcher when available.  Its
    explicit comparison prevents a stale observation from being relabelled as
    current-tree evidence without making the benchmark itself state-owner.
    """

    checked_manifest = dict(manifest) if manifest is not None else load_manifest()
    validate_manifest(checked_manifest)
    if not isinstance(observation, Mapping):
        raise IdleBenchmarkError("observation must be an object")
    _validate_observation_shape(observation)
    if observation["schema"] != OBSERVATION_SCHEMA or observation["benchmark_id"] != BENCHMARK_ID:
        raise IdleBenchmarkError("observation identity does not match the idle benchmark")
    if observation["manifest_sha256"] != manifest_sha256(checked_manifest):
        raise IdleBenchmarkError("observation is not bound to this frozen manifest")
    _require_text(observation["run_id"], "run_id")
    started = _parse_timestamp(observation["started_at"], "started_at")
    finished = _parse_timestamp(observation["finished_at"], "finished_at")
    if finished <= started or (finished - started).total_seconds() < 60:
        raise IdleBenchmarkError("idle window is shorter than the required 60 seconds")
    if observation["measurement_mode"] != "live" or observation["measurement_source"] != "typed-state-owner-runtime@1":
        raise IdleBenchmarkError("only a live typed-state-owner observation is admissible")
    if any(
        observation[key] is not False
        for key in (
            "simulation", "direct_database_access", "network_used",
            "ducklake_scheduling_authority", "promotion_eligible",
        )
    ):
        raise IdleBenchmarkError("idle observation claims a prohibited capability or authority")
    identities = _validate_identities(observation["identities"])
    if current_identity is not None:
        if not isinstance(current_identity, Mapping):
            raise IdleBenchmarkError("current identity must be an object")
        for key, expected in current_identity.items():
            if key not in identities:
                raise IdleBenchmarkError(f"current identity names unknown field {key}")
            if identities[key] != expected:
                raise IdleBenchmarkError(f"observation is stale for identity field {key}")
    capability = _validate_capability(observation["wait_capability"])
    activity = _validate_activity(observation["activity"])
    metrics = _validate_metrics(
        observation["metrics"], wait_count=activity["blocked_server_owned_event_wait"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": manifest_sha256(checked_manifest),
        "run_id": observation["run_id"],
        "verified": True,
        "authoritative": False,
        "promotion_eligible": False,
        "identities": identities,
        "wait_capability": capability,
        "activity": activity,
        "metrics": metrics,
        "window_seconds": int((finished - started).total_seconds()),
        "nonclaim": "Verification is qualification observation only; it does not promote or complete work.",
    }


def repository_identity(repository: Path | str) -> dict[str, str]:
    """Read the clean repository identity used to reject stale live evidence."""

    root = Path(repository).resolve()

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise IdleBenchmarkError(f"cannot establish repository identity: {exc}") from exc
        return result.stdout.strip()

    if git("status", "--porcelain=v1"):
        raise IdleBenchmarkError("repository is dirty; live observation cannot claim an exact tree")
    return {"repository_commit": git("rev-parse", "HEAD"), "repository_tree": git("rev-parse", "HEAD^{tree}")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path, help="live typed-state-owner observation JSON")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("idle_manifest.json"))
    parser.add_argument("--repository", type=Path, help="require evidence to match this clean Git tree")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        current = repository_identity(args.repository) if args.repository else None
        report = validate_observation(_read_object(args.observation), manifest, current_identity=current)
    except IdleBenchmarkError as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "verified": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
