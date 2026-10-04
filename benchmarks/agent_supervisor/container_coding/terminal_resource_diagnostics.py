"""Bounded post-unwind observations; no factory or direct ledger reads.

Existing-facade selection depends on the native scheduler's private process-local
registry/lock layout. Missing, busy, ambiguous or incompatible layouts return an
unavailable observation instead of creating/configuring an owner. Collection
uses the supported snapshot API, which may recover stale leases normally; it is
not an immutable ledger read or the failed request's admission decision.
"""
import json
import math
import sys

SCHEMA = "terminal-scheduler-failure-observation@1"
MODULE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler"
MAX_NUMBER = 2**53 - 1
MAX_BYTES = 4096
REASONS = frozenset({"proof_memory_reservation_required", "proof_memory_headroom",
    "proof_memory_stall", "proof_cpu_stall", "proof_io_stall", "proof_resource_telemetry_unknown",
    "pressure_telemetry_unknown", "memory_pressure", "swap_pressure", "gpu_telemetry_unknown",
    "gpu_memory_telemetry_unknown", "gpu_memory_pressure"})
CAPACITY = ("cpu_slots", "memory_mb", "usable_memory_mb", "reserved_memory_mb",
    "gpu_memory_mb", "usable_gpu_memory_mb", "reserved_gpu_memory_mb",
    "unified_memory_mb", "child_process_slots")
COUNTS = ("active_lease_count", "active_root_lease_count", "active_child_lease_count",
    "waiting_request_count")


def _base():
    return dict(schema=SCHEMA, status="unavailable", observation_boundary="after_unwind",
        exact_admission_decision=False, causal_proof=False,
        native_snapshot_may_recover_stale_owners=True)


def _number(value, *, integer=False, nullable=False):
    if nullable and value is None:
        return None
    if (type(value) not in ((int,) if integer else (int, float))
            or not 0 <= value <= MAX_NUMBER or not math.isfinite(value)):
        raise ValueError("invalid bounded scheduler field")
    return value


def project_failure_resources(sample):
    """Keep the post-unwind scalar schema independent of sampler metadata."""
    integers = {"cpu_slots", "total_memory_mb", "available_memory_mb"}
    return {key: _number(getattr(sample, key), integer=key in integers) for key in (
        "cpu_slots", "total_memory_mb", "available_memory_mb",
        "memory_stall_percent", "cpu_stall_percent", "io_stall_percent")}


def _pressure_sources(value, host):
    """Project path-free PSI attribution from the same admission sample.

    Missing telemetry is distinct from an observed zero. Omitted ancestor
    maxima preserve the aggregate even when the attribution inventory is full.
    This is observational metadata; it does not confer admission authority.
    """
    if (type(value) is not dict or set(value) != {
            "schema", "samples", "omitted_cgroup_scopes", "omitted_maxima"}
            or type(value["schema"]) is not str
            or value["schema"] != "proof-pressure-sources@1" or host is None):
        raise ValueError("invalid pressure attribution")
    samples = value["samples"]
    if type(samples) is not list or not 1 <= len(samples) <= 9:
        raise ValueError("bounded pressure source inventory required")
    metrics = ("memory", "cpu", "io")
    maxima = dict.fromkeys(metrics, 0.)
    rows = []
    for position, row in enumerate(samples):
        if type(row) is not dict or set(row) != {"scope", "depth", *metrics}:
            raise ValueError("invalid pressure source")
        scope, depth = row["scope"], row["depth"]
        if type(scope) is not str:
            raise ValueError("exact pressure scope required")
        if position == 0:
            if scope != "host" or depth is not None:
                raise ValueError("host pressure sample must be first")
        elif scope != "cgroup" or type(depth) is not int or depth != position - 1:
            raise ValueError("ordered cgroup pressure depth required")
        projected = dict(scope=scope, depth=depth)
        for metric in metrics:
            item = row[metric]
            if type(item) is not dict or set(item) != {"avg10", "status"}:
                raise ValueError("invalid pressure metric")
            status, observed = item["status"], item["avg10"]
            if type(status) is not str or status not in {"observed", "unavailable", "malformed"}:
                raise ValueError("invalid pressure status")
            if status == "observed":
                observed = _number(observed)
                if observed > 100:
                    raise ValueError("invalid PSI percentage")
                maxima[metric] = max(maxima[metric], observed)
            elif observed is not None:
                raise ValueError("missing pressure observation must be null")
            projected[metric] = dict(avg10=observed, status=status)
        rows.append(projected)
    omitted = _number(value["omitted_cgroup_scopes"], integer=True)
    tail = value["omitted_maxima"]
    if (type(tail) is not dict or set(tail) != set(metrics)
            or omitted and len(rows) != 9):
        raise ValueError("invalid omitted pressure inventory")
    tail = {key: _number(tail[key], nullable=True) for key in metrics}
    for metric, observed in tail.items():
        if observed is not None:
            if not omitted or observed > 100:
                raise ValueError("invalid omitted PSI percentage")
            maxima[metric] = max(maxima[metric], observed)
        if maxima[metric] != host[metric + "_stall_percent"]:
            raise ValueError("pressure attribution differs from admitted sample")
    return dict(schema=value["schema"], samples=rows,
        omitted_cgroup_scopes=omitted, omitted_maxima=tail)


def collect_failure_admission(error):
    """Read the failed request's attached primary-gate observation, without I/O.

    This captures neither all admission gates nor the cause of host pressure.
    The last sample may precede a cooldown decision; unrelated fairness probes
    can refresh the shared cooldown without sampling this request again.
    Older schedulers and other exceptions explicitly report unavailable.
    """
    base = dict(schema="terminal-admission-failure-observation@1", status="unavailable",
        observation_boundary="request_primary_gate", complete_admission_decision=False,
        causal_proof=False)
    try:
        module = sys.modules.get(MODULE)
        if module is None or not isinstance(error, (module.LeaseTimeoutError, module.LeaseCancelledError)):
            return dict(base, reason="no_native_admission_error")
        value = vars(error).get("admission_observation")
        if value is None:
            return dict(base, reason="no_attached_observation")
        if (type(value) is not dict or set(value) != {
                "schema", "scope", "complete_admission_decision", "primary_gate", "last_sample", "terminal"}
                or value["schema"] != "resource-admission-observation@1"
                or value["scope"] != "proof_primary_gate"
                or value["complete_admission_decision"] is not False
                or value["terminal"] not in {"timeout", "cancelled"}):
            raise ValueError("invalid admission observation")

        def reason(item):
            if item is not None and (type(item) is not str or item not in REASONS):
                raise ValueError("unknown admission reason")
            return item

        gate, sample = value["primary_gate"], value["last_sample"]
        if gate is not None:
            if type(gate) is not dict or set(gate) != {"status", "observed_at", "reason", "backoff_until"}:
                raise ValueError("invalid primary gate")
            if gate["status"] not in {"passed", "refused", "backoff"}:
                raise ValueError("invalid primary status")
            gate = dict(status=gate["status"], observed_at=_number(gate["observed_at"]),
                reason=reason(gate["reason"]), backoff_until=_number(gate["backoff_until"], nullable=True))
        if sample is not None:
            required = {"observed_at", "host", "reserved_root_memory_mb",
                "additional_request_memory_mb", "thresholds", "reason"}
            if type(sample) is not dict or set(sample) not in (
                    required, required | {"pressure_sources"}):
                raise ValueError("invalid primary sample")
            host, thresholds = sample["host"], sample["thresholds"]
            stalls = {"memory_stall_percent", "cpu_stall_percent", "io_stall_percent"}
            if host is not None:
                if type(host) is not dict or set(host) != stalls | {"available_memory_mb"}:
                    raise ValueError("invalid host sample")
                host = {k: _number(v, integer=k == "available_memory_mb") for k, v in host.items()}
            if type(thresholds) is not dict or set(thresholds) != stalls | {"memory_headroom_mb"}:
                raise ValueError("invalid primary thresholds")
            pressure = (_pressure_sources(sample["pressure_sources"], host)
                if "pressure_sources" in sample else None)
            sample = dict(observed_at=_number(sample["observed_at"]), host=host,
                reserved_root_memory_mb=_number(sample["reserved_root_memory_mb"], integer=True, nullable=True),
                additional_request_memory_mb=_number(sample["additional_request_memory_mb"], integer=True, nullable=True),
                thresholds={k: _number(v, integer=k == "memory_headroom_mb") for k, v in thresholds.items()},
                reason=reason(sample["reason"]))
            if pressure is not None:
                sample["pressure_sources"] = pressure
        result = dict(base, status="observed", terminal=value["terminal"], primary_gate=gate, last_sample=sample)
        if len(json.dumps(result, allow_nan=False).encode()) > MAX_BYTES:
            raise ValueError("oversized admission observation")
        return result
    except Exception:
        return dict(base, reason="invalid_attached_observation")


def _shape(snapshot):
    if type(snapshot) is not dict:
        raise ValueError("native scheduler snapshot required")
    capacity, allocated = snapshot["capacity"], snapshot["allocated"]
    backoff, recovery = snapshot["proof_backoff"], snapshot["proof_recovery"]
    if any(type(value) is not dict for value in (capacity, allocated, backoff, recovery)):
        raise ValueError("native scheduler fields required")
    result = dict(capacity={key: _number(capacity[key], integer=True,
        nullable=key in {"gpu_memory_mb", "usable_gpu_memory_mb", "unified_memory_mb"}) for key in CAPACITY},
        allocated={key: _number(allocated[key], integer=True) for key in ("cpu_slots", "memory_mb")},
        **{key: _number(snapshot[key], integer=True) for key in COUNTS}, proof_backoff={}, proof_recovery={})
    if backoff:
        reason = backoff.get("reason")
        result["proof_backoff"] = dict(until=_number(backoff["until"]),
            reason=reason if type(reason) is str and reason in REASONS else "unrecognized")
    if recovery:
        phase = recovery.get("phase")
        result["proof_recovery"] = dict(
            phase=phase if type(phase) is str and phase in {"settling", "paced"} else "unrecognized",
            **{key: _number(recovery[key], integer=True) for key in ("healthy_samples", "grants_remaining")},
            **{key: _number(recovery[key]) for key in ("next_sample_at", "next_grant_at")})
    return result


def collect_failure_scheduler():
    """Observe one existing native owner, without selecting a new configuration.

    The output is bounded; the native snapshot retains its own locking semantics.
    No source, paths, request labels, lease capabilities or exception text escape.
    Collection failures are diagnostic absence, never a new task failure.
    """
    result = _base()
    try:
        module = sys.modules.get(MODULE)
        registry = getattr(module, "_GLOBAL_SCHEDULERS", None)
        lock = getattr(module, "_GLOBAL_SCHEDULERS_LOCK", None)
        native_type = getattr(module, "GlobalResourceScheduler", None)
        if type(registry) is not dict or lock is None or native_type is None:
            return dict(result, reason="existing_owner_unavailable")
        if not lock.acquire(blocking=False):
            return dict(result, reason="existing_owner_busy")
        try:
            if len(registry) != 1:
                return dict(result, reason="existing_owner_missing" if not registry else "existing_owner_ambiguous")
            owner = next(iter(registry.values()))
            if type(owner) is not native_type:
                return dict(result, reason="existing_owner_incompatible")
        finally:
            lock.release()
        result.update(_shape(owner.snapshot()))
        result["status"] = "observed"
        if len(json.dumps(result, allow_nan=False, sort_keys=True).encode()) > MAX_BYTES:
            return dict(_base(), reason="snapshot_exceeds_bound")
        return result
    except Exception:
        return dict(_base(), reason="snapshot_unavailable")
