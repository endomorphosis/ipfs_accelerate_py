"""Bounded post-unwind scheduler observations; no factory or ledger reads.

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
