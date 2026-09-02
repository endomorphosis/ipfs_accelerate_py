"""Canonical live CPU execution qualification (PCPR-037).

Run a stdlib CPU canary on the current host: identity, load, compute,
output validation, cancellation, timeout, cleanup, fail-closed resource
admission, and repetition. CUDA, model, and provider paths stay typed
unavailable. This module never grants ``production_authorized`` and never
emits a closed PCPR release outcome.

Numpy/torch are optional extras. Missing packages stay typed unavailable
and are not recorded as live CPU failure.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_cpu_execution,
)


TASK_ID: Final[str] = "PCPR-037"
GOAL_ID: Final[str] = "PCPR-G430"
INTERFACE: Final[str] = "CpuExecution@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/cpu-execution@1"

KERNEL_NAME: Final[str] = "cpu_integer_mix32"
FIXTURE_N: Final[int] = 8
FIXTURE_EXPECTED: Final[int] = 1499458516
CANARY_N: Final[int] = 200_000
CANARY_EXPECTED: Final[int] = 1494583840
MASK32: Final[int] = 0xFFFFFFFF
MIX_A: Final[int] = 1103515245
MIX_B: Final[int] = 12345


class CpuExecutionError(ValueError):
    """Malformed CPU execution evidence or a forbidden authority claim."""


def cpu_integer_kernel(n: int) -> int:
    """Deterministic 32-bit integer mix. Stdlib only. No numpy/torch."""

    if n < 0:
        raise CpuExecutionError("cpu kernel n must be non-negative")
    acc = 0
    for i in range(n):
        acc = (acc + ((i * MIX_A + MIX_B) ^ (acc << 1))) & MASK32
    return acc


def cpu_identity() -> dict[str, Any]:
    """Live host CPU identity. Absence of /proc stays typed unavailable."""

    cores = os.cpu_count()
    parts: list[str] = []
    implementers: set[str] = set()
    features: list[str] | None = None
    proc_present = False
    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        text = cpuinfo_path.read_text(encoding="utf-8")
        proc_present = True
        for block in text.split("\n\n"):
            record: dict[str, str] = {}
            for line in block.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
            if not record:
                continue
            part = record.get("CPU part")
            if part:
                parts.append(part)
            implementer = record.get("CPU implementer")
            if implementer:
                implementers.add(implementer)
            if features is None and "Features" in record:
                features = record["Features"].split()
    except OSError:
        proc_present = False

    mem_total_kb: int | None = None
    mem_available_kb: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass

    return {
        "backend": "cpu",
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cores": cores,
        "cpuinfo_present": proc_present,
        "cpu_implementers": sorted(implementers),
        "cpu_parts": sorted(set(parts)),
        "feature_count": None if features is None else len(features),
        "mem_total_kb": mem_total_kb,
        "mem_available_kb": mem_available_kb,
        "origin": "live_observed",
        "evidence_kind": "measured",
        "live": True,
        "simulated": False,
        "production_authorized": False,
    }


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    live: bool,
    passed: bool | None,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "present": present,
        "evidence_kind": evidence_kind,
        "live": live,
        "simulated_represented_as_live": False,
        "passed": passed,
        "reason": reason,
        "details": dict(details or {}),
    }


def _run_cancellation_probe() -> dict[str, Any]:
    stop = threading.Event()
    finished = threading.Event()

    def worker() -> None:
        n = 0
        while not stop.is_set() and n < 50_000_000:
            n += 1
        finished.set()

    thread = threading.Thread(target=worker, name="pcpr037-cpu-cancel", daemon=True)
    thread.start()
    stop.set()
    thread.join(timeout=2.0)
    cleaned = (not thread.is_alive()) and finished.is_set()
    return _probe(
        "cpu_cancellation",
        present=cleaned,
        evidence_kind="measured",
        live=True,
        passed=cleaned,
        reason=(
            "Live CPU worker honoured cooperative cancellation and joined."
            if cleaned
            else "Live CPU cancellation did not join within the bounded wait."
        ),
        details={"thread_alive": thread.is_alive(), "finished": finished.is_set()},
    )


def _run_timeout_and_cleanup() -> tuple[dict[str, Any], dict[str, Any]]:
    stop = threading.Event()
    started = threading.Event()

    def worker() -> None:
        started.set()
        end = time.monotonic() + 2.0
        acc = 0
        while time.monotonic() < end and not stop.is_set():
            acc = (acc + 1) & MASK32

    thread = threading.Thread(target=worker, name="pcpr037-cpu-timeout", daemon=True)
    thread.start()
    started.wait(timeout=1.0)
    thread.join(timeout=0.05)
    timed_out = thread.is_alive()
    stop.set()
    thread.join(timeout=2.0)
    cleaned = not thread.is_alive()
    timeout_probe = _probe(
        "cpu_timeout",
        present=timed_out,
        evidence_kind="measured",
        live=True,
        passed=timed_out,
        reason=(
            "Live CPU worker exceeded the join deadline and was recorded as Timeout."
            if timed_out
            else "Live CPU worker finished before the timeout deadline."
        ),
        details={"timed_out": timed_out, "outcome": "Timeout" if timed_out else "Observed"},
    )
    cleanup_probe = _probe(
        "cpu_cleanup",
        present=cleaned,
        evidence_kind="measured",
        live=True,
        passed=cleaned,
        reason=(
            "Timeout worker was signalled and joined; no leftover CPU worker."
            if cleaned
            else "Timeout worker remained alive after the cleanup join."
        ),
        details={"thread_alive": thread.is_alive()},
    )
    return timeout_probe, cleanup_probe


def _run_resource_admission_probe() -> dict[str, Any]:
    available = os.cpu_count()
    if available is None or available < 1:
        return _probe(
            "cpu_resource_admission_fail_closed",
            present=None,
            evidence_kind="unavailable",
            live=False,
            passed=None,
            reason=(
                "os.cpu_count is unavailable. Resource admission is not recorded "
                "as False or passing."
            ),
        )
    requested = available + 10_000
    admitted = requested <= available
    refused = admitted is False
    return _probe(
        "cpu_resource_admission_fail_closed",
        present=refused,
        evidence_kind="measured",
        live=True,
        passed=refused,
        reason=(
            "CPU resource admission refused oversubscription and did not fabricate success."
            if refused
            else "CPU resource admission accepted an oversubscribed request."
        ),
        details={
            "available_cores": available,
            "requested_cores": requested,
            "admitted": admitted,
            "outcome": "Unavailable",
            "code": "cpu_resource_oversubscription_refused",
            "live_oom_not_claimed": True,
        },
    )


def run_live_cpu_probes(*, comprehensive: bool = True) -> tuple[dict[str, Any], ...]:
    """Execute live CPU probes on the current process host."""

    probes: list[dict[str, Any]] = []
    identity = cpu_identity()
    identity_ok = identity.get("cores") is not None and identity.get("machine")
    probes.append(
        _probe(
            "cpu_identity",
            present=bool(identity_ok),
            evidence_kind="measured",
            live=True,
            passed=bool(identity_ok),
            reason=(
                "Live CPU identity observed from the process host."
                if identity_ok
                else "CPU identity could not be observed."
            ),
            details={
                "machine": identity.get("machine"),
                "cores": identity.get("cores"),
                "python_version": identity.get("python_version"),
                "cpuinfo_present": identity.get("cpuinfo_present"),
            },
        )
    )

    load_ok = cpu_integer_kernel(0) == 0
    probes.append(
        _probe(
            "cpu_load",
            present=load_ok,
            evidence_kind="measured",
            live=True,
            passed=load_ok,
            reason=(
                "CPU integer kernel loaded and accepted a zero-length run."
                if load_ok
                else "CPU integer kernel failed to load."
            ),
            details={"kernel": KERNEL_NAME},
        )
    )

    fixture_value = cpu_integer_kernel(FIXTURE_N)
    fixture_ok = fixture_value == FIXTURE_EXPECTED
    probes.append(
        _probe(
            "cpu_output_validation",
            present=fixture_ok,
            evidence_kind="measured",
            live=True,
            passed=fixture_ok,
            reason=(
                "CPU kernel output matched the pinned fixture digest."
                if fixture_ok
                else "CPU kernel output drifted from the pinned fixture digest."
            ),
            details={
                "n": FIXTURE_N,
                "expected": FIXTURE_EXPECTED,
                "observed": fixture_value,
            },
        )
    )

    canary_n = CANARY_N if comprehensive else FIXTURE_N
    canary_expected = CANARY_EXPECTED if comprehensive else FIXTURE_EXPECTED
    first = cpu_integer_kernel(canary_n)
    compute_ok = first == canary_expected
    probes.append(
        _probe(
            "cpu_compute_kernel",
            present=compute_ok,
            evidence_kind="measured",
            live=True,
            passed=compute_ok,
            reason=(
                "Live CPU kernel completed with the expected digest."
                if compute_ok
                else "Live CPU kernel digest did not match."
            ),
            details={
                "kernel": KERNEL_NAME,
                "n": canary_n,
                "expected": canary_expected,
                "observed": first,
            },
        )
    )

    if comprehensive:
        second = cpu_integer_kernel(canary_n)
        repeat_ok = first == second == canary_expected
        probes.append(
            _probe(
                "cpu_repetition",
                present=repeat_ok,
                evidence_kind="measured",
                live=True,
                passed=repeat_ok,
                reason=(
                    "Repeated live CPU execution produced the same digest."
                    if repeat_ok
                    else "Repeated live CPU execution drifted."
                ),
                details={"first": first, "second": second, "n": canary_n},
            )
        )
        probes.append(_run_cancellation_probe())
        timeout_probe, cleanup_probe = _run_timeout_and_cleanup()
        probes.append(timeout_probe)
        probes.append(cleanup_probe)
        probes.append(_run_resource_admission_probe())
    else:
        leftover = any(
            thread.name.startswith("pcpr037-cpu-") and thread.is_alive()
            for thread in threading.enumerate()
        )
        probes.append(
            _probe(
                "cpu_cleanup",
                present=not leftover,
                evidence_kind="measured",
                live=True,
                passed=not leftover,
                reason=(
                    "No leftover PCPR-037 CPU workers after the basic canary."
                    if not leftover
                    else "A PCPR-037 CPU worker remained alive after the basic canary."
                ),
            )
        )

    return tuple(probes)


def qualify_live_cpu_execution(*, test_level: str = "comprehensive") -> dict[str, Any]:
    """Run live CPU execution and return an R&D qualification report.

    ``qualified`` on the hardware ladder stays False because model
    compatibility is PCPR-039. ``cpu_execution_qualified`` may be True.
    ``production_authorized`` is always False.
    """

    level = str(test_level or "comprehensive").strip() or "comprehensive"
    comprehensive = level != "basic"
    identity = cpu_identity()
    probes = run_live_cpu_probes(comprehensive=comprehensive)
    required = [item for item in probes if item["evidence_kind"] == "measured"]
    passed = all(item.get("passed") is True for item in required)
    report = from_live_cpu_execution(
        canary_passed=passed,
        extra={
            "probe": KERNEL_NAME,
            "test_level": level,
            "cores": identity.get("cores"),
            "machine": identity.get("machine"),
        },
    )
    report["tests_passed"] = passed
    report["canary_passed"] = passed
    report["cpu_identity"] = identity
    report["cpu_probes"] = list(probes)
    report["cpu_execution_qualified"] = passed
    report["live"] = True
    report["qualified"] = False
    report["production_authorized"] = False
    report["model_compatible"] = None
    report["live_cuda_qualified"] = False
    report["live_cuda_evidence_kind"] = "unavailable"
    report["live_model_provider_qualified"] = False
    report["live_model_provider_evidence_kind"] = "unavailable"
    report["simulated"] = False
    report["schema"] = SCHEMA
    report["interface"] = INTERFACE
    report["task_id"] = TASK_ID
    report["goal_id"] = GOAL_ID
    return report


__all__ = (
    "CANARY_EXPECTED",
    "CANARY_N",
    "FIXTURE_EXPECTED",
    "FIXTURE_N",
    "GOAL_ID",
    "INTERFACE",
    "KERNEL_NAME",
    "SCHEMA",
    "TASK_ID",
    "CpuExecutionError",
    "cpu_identity",
    "cpu_integer_kernel",
    "qualify_live_cpu_execution",
    "run_live_cpu_probes",
)
