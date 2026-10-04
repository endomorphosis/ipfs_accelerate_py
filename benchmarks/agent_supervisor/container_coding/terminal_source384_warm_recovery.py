"""Explicit qualification-only recovery for read-only Source384 observation.

Each native request still passes ordinary scheduler pressure/headroom gates.
This helper never prepares context, invokes inference, or grants proof authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import time

from .benchmark_resource_profile import EXTENDED_SOURCE384_PROFILE, execution_budget
from .terminal_resource_diagnostics import collect_failure_admission

POLICY = "source384-warm-admission-recovery@1"
SCHEMA = "source384-qualification-warm-observation@1"
ENVIRONMENT_KEY = "IPFS_SUPERVISOR_SOURCE384_WARM_RECOVERY"


def validate_selection(profile, policy):
    execution_budget(profile)
    if policy is not None and (policy != POLICY or profile != EXTENDED_SOURCE384_PROFILE):
        raise ValueError("warm recovery requires its explicit policy and extended qualification profile")
    return policy


def _memory_timeout(error, admission):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError
    if type(error) is not LeaseTimeoutError or admission.get("status") != "observed" or admission.get("terminal") != "timeout":
        return False
    gate, sample = admission.get("primary_gate"), admission.get("last_sample")
    if type(sample) is not dict:
        return False
    host, thresholds = sample.get("host"), sample.get("thresholds")
    return (type(gate) is dict and gate.get("status") in {"refused", "backoff"}
        and gate.get("reason") == "proof_memory_stall" and sample.get("reason") == "proof_memory_stall"
        and type(host) is dict and type(thresholds) is dict
        and thresholds.get("memory_stall_percent") == 10.
        and type(host.get("memory_stall_percent")) in (int, float)
        and host["memory_stall_percent"] > thresholds["memory_stall_percent"])


def observe_warm_context(*, repository, expected_receipt, profile=None, policy=None,
                         deadline_monotonic):
    """Replay the same receipt under one enclosing deadline; retain failed attempts.

    The enclosing deadline is captured by the original qualification probe, never
    renewed here. Complete-call elapsed time includes validation and admission;
    this helper cannot measure their split and leaves that distinction unknown.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import (
        validate_source384_context, _raw, MAX_RECEIPT_BYTES,
    )
    validate_selection(profile, policy)
    if policy is not None:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
            LOCAL_BENCHMARK_PROOF_PROFILE, selected_proof_resource_profile,
        )
        if selected_proof_resource_profile() != LOCAL_BENCHMARK_PROOF_PROFILE:
            raise ValueError("warm recovery requires the unchanged local benchmark admission profile")
    started = time.monotonic()
    if (type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic)
            or not started < deadline_monotonic <= started + execution_budget(profile)["qualification_seconds"]):
        raise ValueError("original finite qualification deadline required")
    if type(expected_receipt) is not dict:
        raise ValueError("exact selected Source384 receipt required")
    selected_receipt = _raw(expected_receipt)
    if len(selected_receipt) > MAX_RECEIPT_BYTES:
        raise ValueError("bounded selected Source384 receipt required")
    maximum, attempts = (180., 2) if policy == POLICY else (90., 1)
    deadline = min(started + maximum, deadline_monotonic)
    result = dict(schema=SCHEMA, policy=policy, resource_profile=profile,
        policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_seconds=maximum, effective_seconds=deadline-started, max_attempts=attempts,
        per_attempt_max_seconds=90., backoff_max_seconds=5., attempts=[],
        status="running", elapsed_seconds=0., backoff_seconds=0., backoff_requested_seconds=0.,
        timing_scope="complete_validation_calls_and_explicit_backoff",
        admission_execution_split_measured=False, inference_replayed=False)
    try:
        for attempt in range(1, attempts + 1):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("warm observation recovery deadline expired")
            limit = min(90., left)
            began = time.monotonic()
            try:
                validate_source384_context(repository=repository, expected_receipt=json.loads(selected_receipt),
                    timeout_seconds=limit)
            except BaseException as error:
                admission = collect_failure_admission(error)
                retryable = _memory_timeout(error, admission)
                result["attempts"].append(dict(attempt=attempt, timeout_seconds=limit,
                    elapsed_seconds=max(0., time.monotonic()-began),
                    status="memory_admission_timeout" if retryable else "failed",
                    error_type=(type(error).__name__ if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}",
                        type(error).__name__) else "OtherError"), admission=admission))
                if not retryable or attempt == attempts or time.monotonic() >= deadline:
                    raise
                before = time.monotonic()
                requested = min(5., max(0., deadline-before))
                result["backoff_requested_seconds"] = requested
                try:
                    time.sleep(requested)
                finally:
                    result["backoff_seconds"] += max(0., time.monotonic()-before)
                if time.monotonic() >= deadline:
                    # Preserve the actual native refusal when no second call fits.
                    raise
            else:
                result["attempts"].append(dict(attempt=attempt, timeout_seconds=limit,
                    elapsed_seconds=max(0., time.monotonic()-began), status="validated"))
                if time.monotonic() >= deadline:
                    raise TimeoutError("warm observation exceeded its original deadline")
                result["status"] = "validated"
                return result
    except BaseException as error:
        result["status"] = "failed"
        error.source384_warm_observation = result
        raise
    finally:
        result["elapsed_seconds"] = max(0., time.monotonic()-started)


def validate_observation(value, *, profile, policy, producer_sha256):
    """Validate only closed qualification metadata, never confer source validity."""
    import json
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseTimeoutError
    validate_selection(profile, policy)
    fields = {"schema", "policy", "resource_profile", "policy_source_sha256", "selected_seconds",
        "effective_seconds", "max_attempts", "per_attempt_max_seconds", "backoff_max_seconds",
        "attempts", "status", "elapsed_seconds", "backoff_seconds", "backoff_requested_seconds", "timing_scope",
        "admission_execution_split_measured", "inference_replayed"}
    maximum, count = (180., 2) if policy == POLICY else (90., 1)
    def number(item, low=0., high=900.):
        return type(item) in (int, float) and math.isfinite(item) and low <= item <= high
    if (type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA
            or value["policy"] != policy or value["resource_profile"] != profile
            or type(producer_sha256) is not str or not re.fullmatch("[0-9a-f]{64}", producer_sha256)
            or value["policy_source_sha256"] != producer_sha256
            or type(value["max_attempts"]) is not int or value["max_attempts"] != count
            or type(value["selected_seconds"]) not in (int, float) or value["selected_seconds"] != maximum
            or type(value["per_attempt_max_seconds"]) not in (int, float) or value["per_attempt_max_seconds"] != 90
            or type(value["backoff_max_seconds"]) not in (int, float) or value["backoff_max_seconds"] != 5
            or not number(value["effective_seconds"], high=maximum) or value["effective_seconds"] <= 0
            or not number(value["elapsed_seconds"]) or not number(value["backoff_seconds"])
            or not number(value["backoff_requested_seconds"], high=5.)
            or value["backoff_seconds"] > value["elapsed_seconds"]
            or value["status"] not in {"validated", "failed"}
            or value["timing_scope"] != "complete_validation_calls_and_explicit_backoff"
            or value["admission_execution_split_measured"] is not False
            or value["inference_replayed"] is not False
            or type(value["attempts"]) is not list or not 1 <= len(value["attempts"]) <= count):
        raise ValueError("invalid closed warm observation")
    accounted = value["backoff_seconds"]
    prior_calls = 0.
    tolerance = 1e-6
    if len(value["attempts"]) == 2 and (value["backoff_requested_seconds"] != 5.
            or value["backoff_seconds"] + tolerance < value["backoff_requested_seconds"]):
        raise ValueError("second warm attempt requires its recorded bounded backoff")
    for index, row in enumerate(value["attempts"], 1):
        base = {"attempt", "timeout_seconds", "elapsed_seconds", "status"}
        if (type(row) is not dict or type(row.get("attempt")) is not int or row["attempt"] != index
                or not number(row.get("timeout_seconds"), high=min(90., value["effective_seconds"]))
                or row["timeout_seconds"] <= 0 or not number(row.get("elapsed_seconds"))
                or row["status"] not in {"validated", "memory_admission_timeout", "failed"}
                or set(row) != (base if row["status"] == "validated" else base | {"error_type", "admission"})):
            raise ValueError("invalid closed warm attempt")
        available = value["effective_seconds"] - prior_calls - (value["backoff_seconds"] if index > 1 else 0.)
        if row["timeout_seconds"] > min(90., available) + tolerance:
            raise ValueError("warm attempt renews the shared deadline")
        prior_calls += row["elapsed_seconds"]
        accounted += row["elapsed_seconds"]
        if row["status"] != "validated":
            if type(row["error_type"]) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", row["error_type"]):
                raise ValueError("bounded warm error class required")
            admission = row["admission"]
            if type(admission) is not dict:
                raise ValueError("closed native admission observation required")
            error = LeaseTimeoutError("metadata validation")
            if admission.get("status") == "observed":
                error.admission_observation = dict(schema="resource-admission-observation@1",
                    scope="proof_primary_gate", complete_admission_decision=False,
                    **{key: admission.get(key) for key in ("primary_gate", "last_sample", "terminal")})
                checked = collect_failure_admission(error)
            else:
                reason = admission.get("reason")
                if reason not in {"no_native_admission_error", "no_attached_observation", "invalid_attached_observation"}:
                    raise ValueError("unknown unavailable native observation")
                checked = {**collect_failure_admission(ValueError()), "reason": reason}
            if checked != admission or (row["status"] == "memory_admission_timeout" and (
                    row["error_type"] != "LeaseTimeoutError" or not _memory_timeout(error, checked))):
                raise ValueError("warm admission observation differs from native projection")
        if index < len(value["attempts"]) and row["status"] != "memory_admission_timeout":
            raise ValueError("only native memory admission refusal can be retried")
    if accounted > value["elapsed_seconds"] + tolerance:
        raise ValueError("warm elapsed time omits recorded work or backoff")
    if value["status"] == "validated" and (value["attempts"][-1]["status"] != "validated"
            or value["elapsed_seconds"] >= value["effective_seconds"]):
        raise ValueError("warm success requires validation before the original deadline")
    if len(json.dumps(value, allow_nan=False).encode()) > 16384:
        raise ValueError("warm observation exceeds byte bound")
    return value
