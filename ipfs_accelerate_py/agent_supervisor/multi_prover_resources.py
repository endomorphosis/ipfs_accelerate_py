"""Shared resource admission and execution for every formal prover family.

The formal-planning stack uses tools with very different execution models:
native SMT and ATP processes, interactive theorem-prover kernels, JVM model
checkers, protocol and hyperproperty tools, in-process runtime monitors, model
providers, and artifact stores.  Giving each adapter its own executor or
semaphore makes all of those local limits simultaneously true while the host
is oversubscribed.

This module provides the opposite ownership model.  A
:class:`MultiProverResourceLease` is the one top-level budget.  Serial and
bundle supervisors receive the same lease and may only create accounted child
grants from it.  Child limits are also exported to common native, JVM, and
numeric runtimes so an admitted tool cannot silently recreate a full-size
nested pool.

The module intentionally has no dependency on optional prover packages.
Commands and callables are supplied by adapters, which keeps admission usable
for every current and future toolchain.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .resource_scheduler import HostResourceSnapshot, ProviderCapacity


DEFAULT_MAX_DIAGNOSTIC_BYTES = 64 * 1024
DEFAULT_TERMINATION_GRACE_SECONDS = 1.0
MULTI_PROVER_RESOURCE_VERSION = 1


class MultiProverResourceClass(str, Enum):
    """Execution families sharing the supervisor's CPU/process budget."""

    TRANSLATION = "translation"
    SMT = "smt"
    ATP = "atp"
    ITP_KERNEL = "itp_kernel"
    JVM_MODEL_CHECKING = "jvm_model_checking"
    PROTOCOL_VERIFICATION = "protocol_verification"
    HYPERPROPERTY_CHECKING = "hyperproperty_checking"
    RUNTIME_MONITOR = "runtime_monitor"
    LLM_INFERENCE = "llm_inference"
    ARTIFACT_IO = "artifact_io"

    @property
    def uses_model(self) -> bool:
        return self is MultiProverResourceClass.LLM_INFERENCE

    @property
    def uses_artifact_io(self) -> bool:
        return self is MultiProverResourceClass.ARTIFACT_IO

    @property
    def pool(self) -> str:
        if self.uses_model:
            return "model"
        if self.uses_artifact_io:
            return "artifact"
        return "cpu-proof"


PROVER_RESOURCE_CLASSES = tuple(item.value for item in MultiProverResourceClass)
ProverResourceClass = MultiProverResourceClass
ProverFamily = MultiProverResourceClass

_FAMILY_ALIASES: Mapping[str, MultiProverResourceClass] = {
    "translate": MultiProverResourceClass.TRANSLATION,
    "translator": MultiProverResourceClass.TRANSLATION,
    "cpu-proof-translate": MultiProverResourceClass.TRANSLATION,
    "solver": MultiProverResourceClass.SMT,
    "z3": MultiProverResourceClass.SMT,
    "cvc5": MultiProverResourceClass.SMT,
    "cpu-proof-solver": MultiProverResourceClass.SMT,
    "automated_theorem_prover": MultiProverResourceClass.ATP,
    "hammer": MultiProverResourceClass.ATP,
    "kernel": MultiProverResourceClass.ITP_KERNEL,
    "itp": MultiProverResourceClass.ITP_KERNEL,
    "lean": MultiProverResourceClass.ITP_KERNEL,
    "coq": MultiProverResourceClass.ITP_KERNEL,
    "rocq": MultiProverResourceClass.ITP_KERNEL,
    "isabelle": MultiProverResourceClass.ITP_KERNEL,
    "cpu-proof-kernel": MultiProverResourceClass.ITP_KERNEL,
    "jvm": MultiProverResourceClass.JVM_MODEL_CHECKING,
    "model_checker": MultiProverResourceClass.JVM_MODEL_CHECKING,
    "model-checker": MultiProverResourceClass.JVM_MODEL_CHECKING,
    "tlc": MultiProverResourceClass.JVM_MODEL_CHECKING,
    "apalache": MultiProverResourceClass.JVM_MODEL_CHECKING,
    "protocol": MultiProverResourceClass.PROTOCOL_VERIFICATION,
    "tamarin": MultiProverResourceClass.PROTOCOL_VERIFICATION,
    "proverif": MultiProverResourceClass.PROTOCOL_VERIFICATION,
    "hyperproperty": MultiProverResourceClass.HYPERPROPERTY_CHECKING,
    "hyperltl": MultiProverResourceClass.HYPERPROPERTY_CHECKING,
    "autohyper": MultiProverResourceClass.HYPERPROPERTY_CHECKING,
    "runtime": MultiProverResourceClass.RUNTIME_MONITOR,
    "monitor": MultiProverResourceClass.RUNTIME_MONITOR,
    "mtl": MultiProverResourceClass.RUNTIME_MONITOR,
    "model": MultiProverResourceClass.LLM_INFERENCE,
    "llm": MultiProverResourceClass.LLM_INFERENCE,
    "llm-proof-draft": MultiProverResourceClass.LLM_INFERENCE,
    "artifact": MultiProverResourceClass.ARTIFACT_IO,
    "io": MultiProverResourceClass.ARTIFACT_IO,
    "io-artifact": MultiProverResourceClass.ARTIFACT_IO,
}


def normalize_prover_resource_class(value: Any) -> MultiProverResourceClass:
    """Normalize tool and legacy scheduler names into one complete taxonomy."""

    if isinstance(value, MultiProverResourceClass):
        return value
    normalized = str(getattr(value, "value", value) or "").strip().lower().replace("-", "_")
    for member in MultiProverResourceClass:
        if normalized == member.value:
            return member
    alias = _FAMILY_ALIASES.get(str(getattr(value, "value", value) or "").strip().lower())
    if alias is None:
        alias = _FAMILY_ALIASES.get(normalized)
    if alias is None:
        raise ValueError(f"unsupported prover resource class: {value!r}")
    return alias


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strings(values: Iterable[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _bounded_text(value: Any, maximum_bytes: int) -> str:
    """Return valid UTF-8 no larger than ``maximum_bytes``."""

    if maximum_bytes <= 0:
        return ""
    encoded = str(value or "").encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return encoded.decode("utf-8")
    marker = b"\n...[diagnostic truncated]"
    if maximum_bytes <= len(marker):
        return encoded[:maximum_bytes].decode("utf-8", errors="ignore")
    return (
        encoded[: maximum_bytes - len(marker)].decode("utf-8", errors="ignore")
        + marker.decode("ascii")
    )


def _json_safe(value: Any, maximum_bytes: int) -> Any:
    """Bound an arbitrary adapter result without retaining private raw objects."""

    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        return {"summary": _bounded_text(repr(value), maximum_bytes)}
    if len(encoded) <= maximum_bytes:
        return value
    return {
        "summary": _bounded_text(encoded.decode("utf-8", errors="replace"), maximum_bytes),
        "truncated": True,
        "original_bytes": len(encoded),
    }


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MultiProverResourceBudget:
    """Hard capacity owned by exactly one top-level supervisor lease."""

    cpu_slots: int = 1
    process_slots: int = 1
    thread_slots: int = 1
    memory_bytes: int = 0
    disk_bytes: int = 0
    provider_quota: int = 0
    model_concurrency: int = 1
    artifact_concurrency: int = 1
    max_portfolio_width: int = 1
    wall_time_ms: int = 0
    max_diagnostic_bytes: int = DEFAULT_MAX_DIAGNOSTIC_BYTES
    host_pressure_limit_percent: int = 95

    def __post_init__(self) -> None:
        for name in (
            "cpu_slots",
            "process_slots",
            "thread_slots",
            "model_concurrency",
            "artifact_concurrency",
            "max_portfolio_width",
        ):
            _positive(getattr(self, name), name)
        for name in (
            "memory_bytes",
            "disk_bytes",
            "provider_quota",
            "wall_time_ms",
            "max_diagnostic_bytes",
        ):
            _nonnegative(getattr(self, name), name)
        if not 1 <= self.host_pressure_limit_percent <= 100:
            raise ValueError("host_pressure_limit_percent must be in [1, 100]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultiProverResourceBudget":
        defaults = cls()
        aliases = {
            "cpu_slots": ("cpu_slots", "max_cpu_concurrency", "max_cpu_proof_concurrency"),
            "process_slots": ("process_slots", "max_processes", "max_parallel"),
            "thread_slots": ("thread_slots", "max_threads"),
            "memory_bytes": ("memory_bytes",),
            "disk_bytes": ("disk_bytes",),
            "provider_quota": ("provider_quota", "quota_units"),
            "model_concurrency": ("model_concurrency", "max_model_concurrency"),
            "artifact_concurrency": ("artifact_concurrency", "max_artifact_concurrency"),
            "max_portfolio_width": (
                "max_portfolio_width",
                "portfolio_max_parallel",
                "max_parallel",
            ),
            "wall_time_ms": ("wall_time_ms",),
            "max_diagnostic_bytes": ("max_diagnostic_bytes", "maximum_output_bytes"),
            "host_pressure_limit_percent": ("host_pressure_limit_percent",),
        }
        result: dict[str, int] = {}
        for target, names in aliases.items():
            raw = next((value[name] for name in names if name in value), getattr(defaults, target))
            try:
                result[target] = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{target} must be an integer") from exc
        if "thread_slots" not in value and "max_threads" not in value:
            result["thread_slots"] = max(result["thread_slots"], result["cpu_slots"])
        return cls(**result)

    @classmethod
    def from_supervisor_budget(cls, value: Any) -> "MultiProverResourceBudget":
        """Adapt REF-258 ``ResourceLeaseBudget`` and compatible contracts."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                return cls.from_mapping(payload)
        try:
            payload = asdict(value)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            return cls.from_mapping(payload)
        raise TypeError(
            "budget must be MultiProverResourceBudget, ResourceLeaseBudget, "
            "or a compatible mapping"
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


SharedProverBudget = MultiProverResourceBudget
UnifiedProverBudget = MultiProverResourceBudget


@dataclass(frozen=True)
class ProverResourceRequest:
    """The complete cost of one child, including any pool it may create."""

    task_id: str
    resource_class: MultiProverResourceClass | str
    cpu_slots: int = 1
    process_slots: int = 1
    thread_slots: int = 1
    memory_bytes: int = 0
    disk_bytes: int = 0
    provider_quota: int = 0
    model_slots: int = 0
    artifact_slots: int = 0
    provider_id: str = ""
    critical_path_value: int = 0

    def __post_init__(self) -> None:
        task_id = str(self.task_id or "").strip()
        if not task_id:
            raise ValueError("task_id must not be empty")
        object.__setattr__(self, "task_id", task_id)
        family = normalize_prover_resource_class(self.resource_class)
        object.__setattr__(self, "resource_class", family)
        for name in ("cpu_slots", "process_slots", "thread_slots"):
            _nonnegative(getattr(self, name), name)
        for name in (
            "memory_bytes",
            "disk_bytes",
            "provider_quota",
            "model_slots",
            "artifact_slots",
            "critical_path_value",
        ):
            _nonnegative(getattr(self, name), name)
        if not any((self.cpu_slots, self.process_slots, self.thread_slots)):
            raise ValueError("a resource request must consume CPU, a process, or a thread")
        object.__setattr__(self, "provider_id", str(self.provider_id or "").strip().lower())
        if family.uses_model and self.model_slots == 0:
            object.__setattr__(self, "model_slots", 1)
        if family.uses_artifact_io and self.artifact_slots == 0:
            object.__setattr__(self, "artifact_slots", 1)

    @classmethod
    def for_family(
        cls,
        task_id: str,
        resource_class: MultiProverResourceClass | str,
        **overrides: Any,
    ) -> "ProverResourceRequest":
        """Build conservative defaults for a tool family."""

        family = normalize_prover_resource_class(resource_class)
        defaults: dict[str, Any] = {
            "task_id": task_id,
            "resource_class": family,
            "cpu_slots": 1,
            "process_slots": 1,
            "thread_slots": 1,
        }
        if family is MultiProverResourceClass.RUNTIME_MONITOR:
            defaults["process_slots"] = 0
        elif family is MultiProverResourceClass.LLM_INFERENCE:
            defaults.update(process_slots=0, model_slots=1, provider_quota=1)
        elif family is MultiProverResourceClass.ARTIFACT_IO:
            defaults.update(process_slots=0, artifact_slots=1)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProverResourceRequest":
        """Normalize planner/tool metadata without importing its schema."""

        def integer(*names: str, default: int = 0) -> int:
            raw = next((value[name] for name in names if name in value), default)
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{names[0]} must be an integer") from exc

        family = value.get(
            "resource_class",
            value.get("prover_family", value.get("family", "")),
        )
        return cls(
            task_id=str(value.get("task_id", value.get("lane_id", ""))),
            resource_class=family,
            cpu_slots=integer("cpu_slots", "required_cpu_slots", default=1),
            process_slots=integer(
                "process_slots", "max_processes", "required_processes", default=1
            ),
            thread_slots=integer(
                "thread_slots", "max_threads", "required_threads", default=1
            ),
            memory_bytes=integer("memory_bytes", "required_memory_bytes"),
            disk_bytes=integer("disk_bytes", "required_disk_bytes"),
            provider_quota=integer("provider_quota", "quota_units"),
            model_slots=integer("model_slots", "model_concurrency"),
            artifact_slots=integer("artifact_slots", "artifact_concurrency"),
            provider_id=str(value.get("provider_id", value.get("provider", ""))),
            critical_path_value=integer(
                "critical_path_value", "downstream_unlock_value"
            ),
        )

    @property
    def family(self) -> MultiProverResourceClass:
        return normalize_prover_resource_class(self.resource_class)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resource_class"] = self.family.value
        return payload


ResourceRequest = ProverResourceRequest


@dataclass(frozen=True)
class ResourceUsage:
    cpu_slots: int = 0
    process_slots: int = 0
    thread_slots: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    provider_quota: int = 0
    model_slots: int = 0
    artifact_slots: int = 0

    @classmethod
    def from_request(cls, request: ProverResourceRequest) -> "ResourceUsage":
        return cls(
            cpu_slots=request.cpu_slots,
            process_slots=request.process_slots,
            thread_slots=request.thread_slots,
            memory_bytes=request.memory_bytes,
            disk_bytes=request.disk_bytes,
            provider_quota=request.provider_quota,
            model_slots=request.model_slots,
            artifact_slots=request.artifact_slots,
        )

    def plus(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(
            **{
                field_name: getattr(self, field_name) + getattr(other, field_name)
                for field_name in asdict(self)
            }
        )

    def minus(self, other: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(
            **{
                field_name: max(0, getattr(self, field_name) - getattr(other, field_name))
                for field_name in asdict(self)
            }
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceAdmission:
    admitted: bool
    task_id: str
    reasons: tuple[str, ...] = ()
    top_level_lease_id: str = ""
    child_lease_id: str = ""

    @property
    def allowed(self) -> bool:
        return self.admitted

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["allowed"] = self.allowed
        payload["reason"] = self.reason
        return payload


class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    ADMISSION_REJECTED = "admission_rejected"
    CACHE_HIT = "cache_hit"

    @property
    def successful(self) -> bool:
        return self in (ExecutionStatus.SUCCEEDED, ExecutionStatus.CACHE_HIT)


@dataclass(frozen=True)
class ProverExecutionReceipt:
    """Bounded terminal evidence for one task, including partial outcomes."""

    task_id: str
    resource_class: MultiProverResourceClass | str
    status: ExecutionStatus | str
    started_at_ms: int
    finished_at_ms: int
    result: Any = None
    diagnostics: str = ""
    reasons: tuple[str, ...] = ()
    top_level_lease_id: str = ""
    child_lease_id: str = ""
    process_group_id: int | None = None
    exit_code: int | None = None
    cache_key: str = ""
    deterministic_identity: str = ""
    cache_bypassed_execution: bool = False
    partial: bool = False
    usage: ResourceUsage = field(default_factory=ResourceUsage)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_class", normalize_prover_resource_class(self.resource_class)
        )
        if not isinstance(self.status, ExecutionStatus):
            object.__setattr__(self, "status", ExecutionStatus(str(self.status)))
        object.__setattr__(self, "reasons", _strings(self.reasons))
        if self.finished_at_ms < self.started_at_ms:
            raise ValueError("finished_at_ms must not precede started_at_ms")

    @property
    def successful(self) -> bool:
        return self.status.successful

    @property
    def duration_ms(self) -> int:
        return self.finished_at_ms - self.started_at_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/prover-execution-receipt@1",
            "resource_version": MULTI_PROVER_RESOURCE_VERSION,
            "task_id": self.task_id,
            "resource_class": self.resource_class.value,
            "status": self.status.value,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": self.duration_ms,
            "result": self.result,
            "diagnostics": self.diagnostics,
            "reasons": list(self.reasons),
            "top_level_lease_id": self.top_level_lease_id,
            "child_lease_id": self.child_lease_id,
            "process_group_id": self.process_group_id,
            "exit_code": self.exit_code,
            "cache_key": self.cache_key,
            "deterministic_identity": self.deterministic_identity,
            "cache_bypassed_execution": self.cache_bypassed_execution,
            "partial": self.partial,
            "usage": self.usage.to_dict(),
        }


class ProverCallable(Protocol):
    def __call__(self, context: "ProverExecutionContext") -> Any:
        ...


@dataclass(frozen=True)
class ProverTask:
    """Executable task plus dependency and deterministic-cache metadata."""

    task_id: str
    resources: ProverResourceRequest
    command: tuple[str, ...] = ()
    runner: Callable[..., Any] | None = None
    dependencies: tuple[str, ...] = ()
    timeout_ms: int = 0
    cwd: Path | str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    stdin_text: str | None = None
    deterministic: bool = False
    cache_key: str = ""
    deterministic_identity: str = ""

    def __post_init__(self) -> None:
        task_id = str(self.task_id or "").strip()
        if not task_id:
            raise ValueError("task_id must not be empty")
        object.__setattr__(self, "task_id", task_id)
        if not isinstance(self.resources, ProverResourceRequest):
            raise ValueError("resources must be ProverResourceRequest")
        if self.resources.task_id != task_id:
            raise ValueError("task and resource request identities must match")
        command = tuple(str(item) for item in self.command)
        object.__setattr__(self, "command", command)
        if bool(command) == bool(self.runner):
            raise ValueError("exactly one of command or runner must be supplied")
        dependencies = _strings(self.dependencies)
        if task_id in dependencies:
            raise ValueError("a prover task cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        _nonnegative(self.timeout_ms, "timeout_ms")
        if self.deterministic and not (
            str(self.cache_key).strip() and str(self.deterministic_identity).strip()
        ):
            raise ValueError(
                "deterministic tasks require both cache_key and deterministic_identity"
            )
        if not self.deterministic and (self.cache_key or self.deterministic_identity):
            raise ValueError("cache metadata is only valid for deterministic tasks")
        object.__setattr__(self, "cache_key", str(self.cache_key or "").strip())
        object.__setattr__(
            self, "deterministic_identity", str(self.deterministic_identity or "").strip()
        )
        object.__setattr__(
            self,
            "env",
            {str(key): str(value) for key, value in dict(self.env).items()},
        )

    @classmethod
    def command_task(
        cls,
        task_id: str,
        resource_class: MultiProverResourceClass | str,
        command: Sequence[str],
        *,
        resources: ProverResourceRequest | None = None,
        **kwargs: Any,
    ) -> "ProverTask":
        return cls(
            task_id=task_id,
            resources=resources
            or ProverResourceRequest.for_family(task_id, resource_class),
            command=tuple(command),
            **kwargs,
        )


@dataclass(frozen=True)
class _CacheEntry:
    cache_key: str
    deterministic_identity: str
    resource_class: MultiProverResourceClass
    result: Any
    diagnostics: str
    source_receipt_id: str


class DeterministicResultCache:
    """Thread-safe exact-identity cache; fuzzy or stale matches are impossible."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, task: ProverTask) -> _CacheEntry | None:
        if not task.deterministic:
            return None
        with self._lock:
            entry = self._entries.get(task.cache_key)
            if (
                entry is None
                or entry.deterministic_identity != task.deterministic_identity
                or entry.resource_class is not task.resources.family
            ):
                return None
            return entry

    def put(self, task: ProverTask, receipt: ProverExecutionReceipt) -> bool:
        if not task.deterministic or receipt.status is not ExecutionStatus.SUCCEEDED:
            return False
        entry = _CacheEntry(
            task.cache_key,
            task.deterministic_identity,
            task.resources.family,
            receipt.result,
            receipt.diagnostics,
            _identity(receipt.to_dict()),
        )
        with self._lock:
            self._entries[task.cache_key] = entry
        return True

    def invalidate(self, cache_key: str) -> bool:
        with self._lock:
            return self._entries.pop(str(cache_key), None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class ChildResourceLease:
    """One idempotently releasable reservation beneath a root lease."""

    def __init__(
        self,
        owner: "MultiProverResourceLease",
        lease_id: str,
        request: ProverResourceRequest,
    ) -> None:
        self.owner = owner
        self.lease_id = lease_id
        self.request = request
        self.acquired_at_ms = int(time.time() * 1000)
        self._released = False
        self._lock = threading.Lock()
        self._terminator: Callable[[], None] | None = None

    @property
    def top_level_lease_id(self) -> str:
        return self.owner.lease_id

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def set_terminator(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._released:
                callback()
            else:
                self._terminator = callback

    def cancel(self) -> None:
        with self._lock:
            callback = self._terminator
        if callback is not None:
            callback()

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            self._released = True
            self._terminator = None
        return self.owner._release(self)

    def child_environment(self) -> dict[str, str]:
        """Limits inherited by native libraries, JVMs, and common provers."""

        threads = max(1, self.request.thread_slots)
        processes = max(1, self.request.process_slots)
        result = {
            "PROVER_TOP_LEVEL_LEASE_ID": self.top_level_lease_id,
            "PROVER_CHILD_LEASE_ID": self.lease_id,
            "PROVER_MAX_PROCESSES": str(processes),
            "PROVER_MAX_THREADS": str(threads),
            "PROVER_PORTFOLIO_WIDTH": str(min(processes, threads)),
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "RAYON_NUM_THREADS": str(threads),
            "TBB_NUM_THREADS": str(threads),
        }
        if self.request.family is MultiProverResourceClass.JVM_MODEL_CHECKING:
            result["JAVA_TOOL_OPTIONS"] = f"-XX:ActiveProcessorCount={threads}"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "top_level_lease_id": self.top_level_lease_id,
            "acquired_at_ms": self.acquired_at_ms,
            "released": self.released,
            "request": self.request.to_dict(),
            "child_environment": self.child_environment(),
        }

    def __enter__(self) -> "ChildResourceLease":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


@dataclass(frozen=True)
class ProverExecutionContext:
    task: ProverTask
    lease: ChildResourceLease
    cancellation: threading.Event
    root_cancellation: threading.Event
    deadline_monotonic: float | None

    @property
    def cancelled(self) -> bool:
        return self.cancellation.is_set() or self.root_cancellation.is_set()

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())

    @property
    def child_limits(self) -> Mapping[str, int]:
        request = self.lease.request
        return {
            "max_processes": request.process_slots,
            "max_threads": request.thread_slots,
            "cpu_slots": request.cpu_slots,
            "memory_bytes": request.memory_bytes,
            "disk_bytes": request.disk_bytes,
            "provider_quota": request.provider_quota,
            "model_concurrency": request.model_slots,
        }


def _host_pressure(host: HostResourceSnapshot | Mapping[str, Any] | None) -> int:
    if host is None:
        return 0
    if isinstance(host, HostResourceSnapshot):
        return max(host.cpu_percent, host.memory_percent, host.disk_percent)
    values: list[int] = []
    for name in ("cpu_percent", "memory_percent", "disk_percent"):
        try:
            values.append(max(0, min(100, int(host.get(name, 0)))))
        except (TypeError, ValueError):
            values.append(0)
    return max(values, default=0)


def adaptive_portfolio_width(
    budget: MultiProverResourceBudget,
    host: HostResourceSnapshot | Mapping[str, Any] | None,
    candidates: Sequence[ProverTask | ProverResourceRequest] = (),
    *,
    available_process_slots: int | None = None,
    available_thread_slots: int | None = None,
) -> int:
    """Choose deterministic width from pressure, capacity, and path value.

    Critical work does not bypass hard resource limits.  It can retain one
    additional lane at moderate pressure, while low-value speculative
    portfolios contract earlier.
    """

    process_cap = (
        budget.process_slots
        if available_process_slots is None
        else max(0, available_process_slots)
    )
    thread_cap = (
        budget.thread_slots
        if available_thread_slots is None
        else max(0, available_thread_slots)
    )
    hard_cap = min(
        budget.max_portfolio_width,
        budget.cpu_slots,
        process_cap,
        thread_cap,
    )
    # In-process model/monitor/artifact work can use a thread without a process.
    if candidates and all(
        (item.resources if isinstance(item, ProverTask) else item).process_slots == 0
        for item in candidates
    ):
        hard_cap = min(budget.max_portfolio_width, budget.cpu_slots, thread_cap)
    if hard_cap <= 0:
        return 0
    pressure = _host_pressure(host)
    if pressure >= budget.host_pressure_limit_percent:
        return 0
    headroom = max(1, budget.host_pressure_limit_percent - pressure)
    pressure_width = max(
        1,
        min(hard_cap, math.ceil(hard_cap * headroom / budget.host_pressure_limit_percent)),
    )
    requests = [
        item.resources if isinstance(item, ProverTask) else item for item in candidates
    ]
    highest_value = max((item.critical_path_value for item in requests), default=0)
    if highest_value > 0 and pressure < budget.host_pressure_limit_percent - 5:
        pressure_width = min(hard_cap, pressure_width + 1)
    return pressure_width


class MultiProverResourceLease:
    """Atomic root accounting shared by all serial and bundle supervisors."""

    def __init__(
        self,
        budget: MultiProverResourceBudget,
        *,
        host: HostResourceSnapshot | Mapping[str, Any] | None = None,
        providers: Iterable[ProviderCapacity] | Mapping[str, ProviderCapacity] = (),
        lease_id: str | None = None,
    ) -> None:
        self.budget = budget
        self.host = host
        provider_values = providers.values() if isinstance(providers, Mapping) else providers
        self.providers = {item.provider_id: item for item in provider_values}
        self.lease_id = lease_id or f"multi-prover:{uuid.uuid4().hex}"
        self.acquired_at_ms = int(time.time() * 1000)
        self._deadline_monotonic = (
            time.monotonic() + budget.wall_time_ms / 1000
            if budget.wall_time_ms
            else None
        )
        self.cancellation = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._usage = ResourceUsage()
        self._children: dict[str, ChildResourceLease] = {}
        self._closed = False
        self._provider_active: dict[str, int] = {}
        self._provider_quota: dict[str, int] = {}

    @property
    def usage(self) -> ResourceUsage:
        with self._condition:
            return replace(self._usage)

    @property
    def active_children(self) -> tuple[ChildResourceLease, ...]:
        with self._condition:
            return tuple(self._children[key] for key in sorted(self._children))

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def available(self) -> ResourceUsage:
        with self._condition:
            return ResourceUsage(
                cpu_slots=max(0, self.budget.cpu_slots - self._usage.cpu_slots),
                process_slots=max(0, self.budget.process_slots - self._usage.process_slots),
                thread_slots=max(0, self.budget.thread_slots - self._usage.thread_slots),
                memory_bytes=(
                    max(0, self.budget.memory_bytes - self._usage.memory_bytes)
                    if self.budget.memory_bytes
                    else 0
                ),
                disk_bytes=(
                    max(0, self.budget.disk_bytes - self._usage.disk_bytes)
                    if self.budget.disk_bytes
                    else 0
                ),
                provider_quota=(
                    max(0, self.budget.provider_quota - self._usage.provider_quota)
                    if self.budget.provider_quota
                    else 0
                ),
                model_slots=max(
                    0, self.budget.model_concurrency - self._usage.model_slots
                ),
                artifact_slots=max(
                    0, self.budget.artifact_concurrency - self._usage.artifact_slots
                ),
            )

    def _reasons(self, request: ProverResourceRequest) -> tuple[str, ...]:
        if self._closed:
            return ("top_level_lease_closed",)
        if self.cancellation.is_set():
            return ("top_level_lease_cancelled",)
        if (
            self._deadline_monotonic is not None
            and time.monotonic() >= self._deadline_monotonic
        ):
            return ("lease_wall_time",)
        if _host_pressure(self.host) >= self.budget.host_pressure_limit_percent:
            return ("host_pressure",)
        usage = self._usage
        reasons: list[str] = []
        checks = (
            ("cpu_slots", self.budget.cpu_slots, request.cpu_slots, usage.cpu_slots),
            ("process_slots", self.budget.process_slots, request.process_slots, usage.process_slots),
            ("thread_slots", self.budget.thread_slots, request.thread_slots, usage.thread_slots),
            ("model_concurrency", self.budget.model_concurrency, request.model_slots, usage.model_slots),
            ("artifact_concurrency", self.budget.artifact_concurrency, request.artifact_slots, usage.artifact_slots),
        )
        for reason, limit, requested, used in checks:
            if used + requested > limit:
                reasons.append(reason)
        optional = (
            ("memory_bytes", self.budget.memory_bytes, request.memory_bytes, usage.memory_bytes),
            ("disk_bytes", self.budget.disk_bytes, request.disk_bytes, usage.disk_bytes),
            ("provider_quota", self.budget.provider_quota, request.provider_quota, usage.provider_quota),
        )
        for reason, limit, requested, used in optional:
            if limit and used + requested > limit:
                reasons.append(reason)
        if request.provider_id and request.provider_id in self.providers:
            provider = self.providers[request.provider_id]
            active = self._provider_active.get(request.provider_id, 0)
            reserved_quota = self._provider_quota.get(request.provider_id, 0)
            if not provider.healthy:
                reasons.append("provider_unhealthy")
            if active + request.model_slots > provider.available_concurrency:
                reasons.append("provider_concurrency")
            if (
                provider.quota_remaining >= 0
                and reserved_quota + request.provider_quota
                > provider.quota_remaining
            ):
                reasons.append("provider_quota_remaining")
        elif request.provider_id and self.providers:
            reasons.append("provider_unavailable")
        return tuple(dict.fromkeys(reasons))

    def try_acquire(
        self, request: ProverResourceRequest
    ) -> tuple[ResourceAdmission, ChildResourceLease | None]:
        """Atomically reserve a child without blocking a ready bundle slice."""

        if not isinstance(request, ProverResourceRequest):
            raise TypeError("request must be ProverResourceRequest")
        with self._condition:
            reasons = self._reasons(request)
            if reasons:
                return ResourceAdmission(
                    False, request.task_id, reasons, self.lease_id
                ), None
            child_id = f"{self.lease_id}:child:{uuid.uuid4().hex}"
            child = ChildResourceLease(self, child_id, request)
            self._children[child_id] = child
            self._usage = self._usage.plus(ResourceUsage.from_request(request))
            if request.provider_id:
                self._provider_active[request.provider_id] = (
                    self._provider_active.get(request.provider_id, 0)
                    + request.model_slots
                )
                self._provider_quota[request.provider_id] = (
                    self._provider_quota.get(request.provider_id, 0)
                    + request.provider_quota
                )
            return ResourceAdmission(
                True,
                request.task_id,
                (),
                self.lease_id,
                child_id,
            ), child

    acquire = try_acquire

    def _release(self, child: ChildResourceLease) -> bool:
        with self._condition:
            existing = self._children.pop(child.lease_id, None)
            if existing is None:
                return False
            self._usage = self._usage.minus(ResourceUsage.from_request(child.request))
            if child.request.provider_id:
                self._provider_active[child.request.provider_id] = max(
                    0,
                    self._provider_active.get(child.request.provider_id, 0)
                    - child.request.model_slots,
                )
                self._provider_quota[child.request.provider_id] = max(
                    0,
                    self._provider_quota.get(child.request.provider_id, 0)
                    - child.request.provider_quota,
                )
            self._condition.notify_all()
            return True

    def cancel(self) -> None:
        self.cancellation.set()
        for child in self.active_children:
            child.cancel()

    def close(self) -> None:
        self.cancel()
        for child in self.active_children:
            child.release()
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def portfolio_width(
        self, candidates: Sequence[ProverTask | ProverResourceRequest] = ()
    ) -> int:
        available = self.available()
        return adaptive_portfolio_width(
            self.budget,
            self.host,
            candidates,
            available_process_slots=available.process_slots,
            available_thread_slots=available.thread_slots,
        )

    @property
    def remaining_wall_time_seconds(self) -> float | None:
        if self._deadline_monotonic is None:
            return None
        return max(0.0, self._deadline_monotonic - time.monotonic())

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "schema": "ipfs_accelerate_py/agent-supervisor/multi-prover-lease@1",
                "lease_id": self.lease_id,
                "acquired_at_ms": self.acquired_at_ms,
                "closed": self._closed,
                "cancelled": self.cancellation.is_set(),
                "budget": self.budget.to_dict(),
                "usage": self._usage.to_dict(),
                "available": self.available().to_dict(),
                "active_children": [
                    child.to_dict() for child in self.active_children
                ],
            }

    def __enter__(self) -> "MultiProverResourceLease":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


TopLevelResourceLease = MultiProverResourceLease
SharedResourceLease = MultiProverResourceLease


class MultiProverResourceManager:
    """Factory retaining root leases for process-wide supervisor sharing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[str, MultiProverResourceLease] = {}

    def open_lease(
        self,
        budget: Any,
        *,
        host: HostResourceSnapshot | Mapping[str, Any] | None = None,
        providers: Iterable[ProviderCapacity] | Mapping[str, ProviderCapacity] = (),
    ) -> MultiProverResourceLease:
        normalized = MultiProverResourceBudget.from_supervisor_budget(budget)
        lease = MultiProverResourceLease(
            normalized, host=host, providers=providers
        )
        with self._lock:
            self._leases[lease.lease_id] = lease
        return lease

    acquire = open_lease

    def get(self, lease_id: str) -> MultiProverResourceLease | None:
        with self._lock:
            return self._leases.get(str(lease_id))

    def close(self, lease: MultiProverResourceLease | str) -> bool:
        lease_id = lease.lease_id if isinstance(lease, MultiProverResourceLease) else str(lease)
        with self._lock:
            existing = self._leases.pop(lease_id, None)
        if existing is None:
            return False
        existing.close()
        return True

    @property
    def active_leases(self) -> tuple[MultiProverResourceLease, ...]:
        with self._lock:
            return tuple(
                self._leases[key]
                for key in sorted(self._leases)
                if not self._leases[key].closed
            )


MultiProverResourceController = MultiProverResourceManager


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows
            process.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=max(0.0, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:  # pragma: no cover - unkillable OS process
        pass


def _invoke_runner(
    runner: Callable[..., Any],
    context: ProverExecutionContext,
) -> Any:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(context)
    positional = [
        item
        for item in signature.parameters.values()
        if item.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional and not any(
        item.kind is inspect.Parameter.VAR_POSITIONAL
        for item in signature.parameters.values()
    ):
        return runner()
    return runner(context)


class ProverTaskExecutor:
    """Execute commands/callables under child leases and bounded receipts."""

    def __init__(
        self,
        lease: MultiProverResourceLease,
        *,
        cache: DeterministicResultCache | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.lease = lease
        # An empty cache implements ``__len__`` and is therefore falsey.  It
        # must still be retained so serial/bundle supervisors share entries.
        self.cache = cache if cache is not None else DeterministicResultCache()
        self._monotonic = monotonic

    def cache_receipt(self, task: ProverTask) -> ProverExecutionReceipt | None:
        entry = self.cache.get(task)
        if entry is None:
            return None
        now = int(time.time() * 1000)
        return ProverExecutionReceipt(
            task_id=task.task_id,
            resource_class=task.resources.family,
            status=ExecutionStatus.CACHE_HIT,
            started_at_ms=now,
            finished_at_ms=now,
            result=entry.result,
            diagnostics=entry.diagnostics,
            cache_key=task.cache_key,
            deterministic_identity=task.deterministic_identity,
            cache_bypassed_execution=True,
        )

    def execute(
        self,
        task: ProverTask,
        *,
        cancellation: threading.Event | None = None,
    ) -> ProverExecutionReceipt:
        cancel = cancellation or threading.Event()
        cached = self.cache_receipt(task)
        if cached is not None:
            return cached
        started_ms = int(time.time() * 1000)
        admission, child = self.lease.try_acquire(task.resources)
        if child is None:
            status = (
                ExecutionStatus.CANCELLED
                if any("cancel" in reason for reason in admission.reasons)
                else ExecutionStatus.ADMISSION_REJECTED
            )
            return ProverExecutionReceipt(
                task.task_id,
                task.resources.family,
                status,
                started_ms,
                int(time.time() * 1000),
                reasons=admission.reasons,
                top_level_lease_id=self.lease.lease_id,
                partial=True,
            )
        deadline_candidates: list[float] = []
        if task.timeout_ms:
            deadline_candidates.append(self._monotonic() + task.timeout_ms / 1000)
        remaining_root = self.lease.remaining_wall_time_seconds
        if remaining_root is not None:
            deadline_candidates.append(self._monotonic() + remaining_root)
        deadline = min(deadline_candidates) if deadline_candidates else None
        context = ProverExecutionContext(
            task, child, cancel, self.lease.cancellation, deadline
        )
        try:
            if context.cancelled:
                receipt = self._base_receipt(
                    task,
                    child,
                    started_ms,
                    ExecutionStatus.CANCELLED,
                    diagnostics="cancelled before execution",
                    partial=True,
                )
            elif task.command:
                receipt = self._execute_command(task, context, started_ms)
            else:
                receipt = self._execute_callable(task, context, started_ms)
        finally:
            child.release()
        self.cache.put(task, receipt)
        return receipt

    def _base_receipt(
        self,
        task: ProverTask,
        child: ChildResourceLease,
        started_ms: int,
        status: ExecutionStatus,
        *,
        result: Any = None,
        diagnostics: str = "",
        reasons: Iterable[str] = (),
        process_group_id: int | None = None,
        exit_code: int | None = None,
        partial: bool = False,
    ) -> ProverExecutionReceipt:
        limit = self.lease.budget.max_diagnostic_bytes
        return ProverExecutionReceipt(
            task_id=task.task_id,
            resource_class=task.resources.family,
            status=status,
            started_at_ms=started_ms,
            finished_at_ms=int(time.time() * 1000),
            result=_json_safe(result, limit),
            diagnostics=_bounded_text(diagnostics, limit),
            reasons=tuple(reasons),
            top_level_lease_id=self.lease.lease_id,
            child_lease_id=child.lease_id,
            process_group_id=process_group_id,
            exit_code=exit_code,
            cache_key=task.cache_key,
            deterministic_identity=task.deterministic_identity,
            partial=partial,
            usage=ResourceUsage.from_request(task.resources),
        )

    def _execute_command(
        self,
        task: ProverTask,
        context: ProverExecutionContext,
        started_ms: int,
    ) -> ProverExecutionReceipt:
        env = dict(os.environ)
        env.update(context.lease.child_environment())
        # Explicit task values may tighten or annotate limits; the authoritative
        # lease variables themselves cannot be replaced.
        protected = set(context.lease.child_environment())
        env.update({key: value for key, value in task.env.items() if key not in protected})
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                list(task.command),
                cwd=str(task.cwd) if task.cwd is not None else None,
                env=env,
                stdin=subprocess.PIPE if task.stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
            context.lease.set_terminator(lambda: _terminate_process_group(process))
            process_group_id = process.pid
            capture_limit = self.lease.budget.max_diagnostic_bytes
            stdout_buffer = bytearray()
            stderr_buffer = bytearray()
            truncated = [False]
            capture_lock = threading.Lock()

            def drain(stream: Any, target: bytearray) -> None:
                try:
                    while True:
                        chunk = stream.read(8192)
                        if not chunk:
                            return
                        with capture_lock:
                            remaining = max(0, capture_limit - len(target))
                            if remaining:
                                target.extend(chunk[:remaining])
                            if len(chunk) > remaining:
                                truncated[0] = True
                except (OSError, ValueError):
                    return

            readers = [
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout_buffer),
                    name=f"prover-stdout-{task.task_id}",
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr_buffer),
                    name=f"prover-stderr-{task.task_id}",
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()

            if task.stdin_text is not None:
                def write_stdin() -> None:
                    try:
                        assert process is not None and process.stdin is not None
                        process.stdin.write(task.stdin_text.encode("utf-8"))
                        process.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        return

                threading.Thread(
                    target=write_stdin,
                    name=f"prover-stdin-{task.task_id}",
                    daemon=True,
                ).start()

            status: ExecutionStatus | None = None
            while process.poll() is None:
                if context.cancelled:
                    status = ExecutionStatus.CANCELLED
                    _terminate_process_group(process)
                    break
                remaining = context.remaining_seconds
                if remaining is not None and remaining <= 0:
                    status = ExecutionStatus.TIMED_OUT
                    _terminate_process_group(process)
                    break
                time.sleep(0.02 if remaining is None else min(0.02, remaining))
            if process.poll() is None:
                _terminate_process_group(process)
            for reader in readers:
                reader.join(0.5)
            stdout = bytes(stdout_buffer)
            stderr = bytes(stderr_buffer)
            if status is None:
                status = (
                    ExecutionStatus.SUCCEEDED
                    if process.returncode == 0
                    else ExecutionStatus.FAILED
                )
            diagnostic = (
                stdout.decode("utf-8", errors="replace")
                + ("\n" if stdout and stderr else "")
                + stderr.decode("utf-8", errors="replace")
            )
            if truncated[0]:
                diagnostic += "\n...[diagnostic truncated]"
            result = {
                "command": list(task.command),
                "exit_code": process.returncode,
                "stdout": _bounded_text(
                    stdout.decode("utf-8", errors="replace"),
                    self.lease.budget.max_diagnostic_bytes // 2,
                ),
                "stderr": _bounded_text(
                    stderr.decode("utf-8", errors="replace"),
                    self.lease.budget.max_diagnostic_bytes // 2,
                ),
            }
            return self._base_receipt(
                task,
                context.lease,
                started_ms,
                status,
                result=result,
                diagnostics=diagnostic,
                reasons=(
                    ("timeout",)
                    if status is ExecutionStatus.TIMED_OUT
                    else ("cancelled",)
                    if status is ExecutionStatus.CANCELLED
                    else ()
                ),
                process_group_id=process_group_id,
                exit_code=process.returncode,
                partial=status in (ExecutionStatus.TIMED_OUT, ExecutionStatus.CANCELLED),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if process is not None:
                _terminate_process_group(process)
            return self._base_receipt(
                task,
                context.lease,
                started_ms,
                ExecutionStatus.FAILED,
                diagnostics=f"{type(exc).__name__}: {exc}",
                reasons=("process_error",),
                process_group_id=process.pid if process is not None else None,
                exit_code=process.returncode if process is not None else None,
                partial=True,
            )

    def _execute_callable(
        self,
        task: ProverTask,
        context: ProverExecutionContext,
        started_ms: int,
    ) -> ProverExecutionReceipt:
        outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcomes.put((True, _invoke_runner(task.runner, context)))  # type: ignore[arg-type]
            except BaseException as exc:
                outcomes.put((False, exc))

        thread = threading.Thread(
            target=invoke, name=f"prover-{task.task_id}", daemon=True
        )
        thread.start()
        while thread.is_alive():
            if context.cancelled:
                return self._base_receipt(
                    task,
                    context.lease,
                    started_ms,
                    ExecutionStatus.CANCELLED,
                    diagnostics="callable cancellation requested",
                    reasons=("cancelled",),
                    partial=True,
                )
            remaining = context.remaining_seconds
            if remaining is not None and remaining <= 0:
                context.cancellation.set()
                return self._base_receipt(
                    task,
                    context.lease,
                    started_ms,
                    ExecutionStatus.TIMED_OUT,
                    diagnostics="callable exceeded its wall-time limit",
                    reasons=("timeout",),
                    partial=True,
                )
            thread.join(0.02 if remaining is None else min(0.02, remaining))
        try:
            ok, value = outcomes.get_nowait()
        except queue.Empty:  # pragma: no cover - defensive thread failure
            ok, value = False, RuntimeError("callable exited without a result")
        if not ok:
            return self._base_receipt(
                task,
                context.lease,
                started_ms,
                ExecutionStatus.FAILED,
                diagnostics=f"{type(value).__name__}: {value}",
                reasons=("runner_error",),
                partial=True,
            )
        return self._base_receipt(
            task,
            context.lease,
            started_ms,
            ExecutionStatus.SUCCEEDED,
            result=value,
        )


def dependency_closed_ready_slice(
    tasks: Iterable[ProverTask],
    completed_task_ids: Iterable[str] = (),
    *,
    width: int | None = None,
) -> tuple[ProverTask, ...]:
    """Return only members whose complete dependency set is already satisfied."""

    completed = {str(item) for item in completed_task_ids}
    pending = tuple(tasks)
    ready = [
        task for task in pending if set(task.dependencies).issubset(completed)
    ]
    ready.sort(
        key=lambda task: (
            -task.resources.critical_path_value,
            task.task_id,
        )
    )
    if width is not None:
        ready = ready[: max(0, width)]
    return tuple(ready)


@dataclass(frozen=True)
class BundleExecutionReceipt:
    receipts: tuple[ProverExecutionReceipt, ...]
    started_at_ms: int
    finished_at_ms: int
    top_level_lease_id: str
    cancelled: bool = False

    @property
    def successful_task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.receipts if item.successful)

    @property
    def partial_receipts(self) -> tuple[ProverExecutionReceipt, ...]:
        return tuple(item for item in self.receipts if item.partial)

    @property
    def status_by_task_id(self) -> Mapping[str, ExecutionStatus]:
        return {item.task_id: item.status for item in self.receipts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/prover-bundle-receipt@1",
            "resource_version": MULTI_PROVER_RESOURCE_VERSION,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": self.finished_at_ms - self.started_at_ms,
            "top_level_lease_id": self.top_level_lease_id,
            "cancelled": self.cancelled,
            "successful_task_ids": list(self.successful_task_ids),
            "receipts": [item.to_dict() for item in self.receipts],
        }


class SerialProverSupervisor:
    """Dependency-enforcing serial consumer of an existing root lease."""

    def __init__(
        self,
        lease: MultiProverResourceLease,
        *,
        cache: DeterministicResultCache | None = None,
    ) -> None:
        self.lease = lease
        self.executor = ProverTaskExecutor(lease, cache=cache)

    def execute(
        self,
        tasks: Iterable[ProverTask],
        *,
        completed_task_ids: Iterable[str] = (),
        cancellation: threading.Event | None = None,
    ) -> BundleExecutionReceipt:
        started = int(time.time() * 1000)
        cancel = cancellation or threading.Event()
        completed = {str(item) for item in completed_task_ids}
        receipts: list[ProverExecutionReceipt] = []
        for task in tasks:
            now = int(time.time() * 1000)
            missing = sorted(set(task.dependencies) - completed)
            if missing:
                receipts.append(
                    ProverExecutionReceipt(
                        task.task_id,
                        task.resources.family,
                        ExecutionStatus.BLOCKED,
                        now,
                        now,
                        diagnostics="task dependencies are not complete",
                        reasons=tuple(f"dependency:{item}" for item in missing),
                        top_level_lease_id=self.lease.lease_id,
                        partial=True,
                    )
                )
                continue
            if cancel.is_set() or self.lease.cancellation.is_set():
                receipts.append(
                    ProverExecutionReceipt(
                        task.task_id,
                        task.resources.family,
                        ExecutionStatus.CANCELLED,
                        now,
                        now,
                        reasons=("cancelled",),
                        top_level_lease_id=self.lease.lease_id,
                        partial=True,
                    )
                )
                continue
            receipt = self.executor.execute(task, cancellation=cancel)
            receipts.append(receipt)
            if receipt.successful:
                completed.add(task.task_id)
        return BundleExecutionReceipt(
            tuple(receipts),
            started,
            int(time.time() * 1000),
            self.lease.lease_id,
            cancel.is_set() or self.lease.cancellation.is_set(),
        )

    run = execute


class BundleProverSupervisor:
    """Run dependency-closed ready slices without nested executor pools."""

    def __init__(
        self,
        lease: MultiProverResourceLease,
        *,
        cache: DeterministicResultCache | None = None,
    ) -> None:
        self.lease = lease
        self.executor = ProverTaskExecutor(lease, cache=cache)

    def execute(
        self,
        tasks: Iterable[ProverTask],
        *,
        completed_task_ids: Iterable[str] = (),
        cancellation: threading.Event | None = None,
    ) -> BundleExecutionReceipt:
        started = int(time.time() * 1000)
        cancel = cancellation or threading.Event()
        task_list = tuple(tasks)
        by_id = {task.task_id: task for task in task_list}
        if len(by_id) != len(task_list):
            raise ValueError("bundle task ids must be unique")
        completed = {str(item) for item in completed_task_ids}
        pending = dict(by_id)
        receipts: dict[str, ProverExecutionReceipt] = {}

        while pending:
            if cancel.is_set() or self.lease.cancellation.is_set():
                for task in pending.values():
                    now = int(time.time() * 1000)
                    receipts[task.task_id] = ProverExecutionReceipt(
                        task.task_id,
                        task.resources.family,
                        ExecutionStatus.CANCELLED,
                        now,
                        now,
                        reasons=("cancelled",),
                        top_level_lease_id=self.lease.lease_id,
                        partial=True,
                    )
                break

            ready_all = dependency_closed_ready_slice(pending.values(), completed)
            if not ready_all:
                for task in pending.values():
                    now = int(time.time() * 1000)
                    missing = sorted(set(task.dependencies) - completed)
                    receipts[task.task_id] = ProverExecutionReceipt(
                        task.task_id,
                        task.resources.family,
                        ExecutionStatus.BLOCKED,
                        now,
                        now,
                        diagnostics="no dependency-closed ready path remains",
                        reasons=tuple(f"dependency:{item}" for item in missing),
                        top_level_lease_id=self.lease.lease_id,
                        partial=True,
                    )
                break

            # Exact cache hits are completed before width calculation and never
            # consume host/provider capacity.
            uncached: list[ProverTask] = []
            for task in ready_all:
                cached = self.executor.cache_receipt(task)
                if cached is None:
                    uncached.append(task)
                else:
                    receipts[task.task_id] = cached
                    completed.add(task.task_id)
                    pending.pop(task.task_id, None)
            if not uncached:
                continue

            width = self.lease.portfolio_width(uncached)
            if width <= 0:
                # A zero width can mean measured host pressure or capacity
                # currently held by another supervisor sharing this lease.
                # Ask normal admission for one member so the durable receipt
                # retains the exact hard constraint rather than mislabelling
                # all zero-width cases as host pressure.
                task = uncached[0]
                receipts[task.task_id] = self.executor.execute(
                    task, cancellation=cancel
                )
                pending.pop(task.task_id, None)
                if receipts[task.task_id].successful:
                    completed.add(task.task_id)
                continue

            ready = uncached[:width]
            with ThreadPoolExecutor(
                max_workers=width, thread_name_prefix="shared-prover"
            ) as pool:
                futures: dict[Future[ProverExecutionReceipt], ProverTask] = {
                    pool.submit(self.executor.execute, task, cancellation=cancel): task
                    for task in ready
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        receipt = future.result()
                    except BaseException as exc:  # pragma: no cover - executor guard
                        now = int(time.time() * 1000)
                        receipt = ProverExecutionReceipt(
                            task.task_id,
                            task.resources.family,
                            ExecutionStatus.FAILED,
                            now,
                            now,
                            diagnostics=_bounded_text(
                                f"{type(exc).__name__}: {exc}",
                                self.lease.budget.max_diagnostic_bytes,
                            ),
                            reasons=("supervisor_error",),
                            top_level_lease_id=self.lease.lease_id,
                            partial=True,
                        )
                    receipts[task.task_id] = receipt
                    pending.pop(task.task_id, None)
                    if receipt.successful:
                        completed.add(task.task_id)

        ordered = tuple(receipts[task.task_id] for task in task_list)
        return BundleExecutionReceipt(
            ordered,
            started,
            int(time.time() * 1000),
            self.lease.lease_id,
            cancel.is_set() or self.lease.cancellation.is_set(),
        )

    run = execute


MultiProverBundleSupervisor = BundleProverSupervisor
MultiProverSerialSupervisor = SerialProverSupervisor


__all__ = [
    "BundleExecutionReceipt",
    "BundleProverSupervisor",
    "ChildResourceLease",
    "DEFAULT_MAX_DIAGNOSTIC_BYTES",
    "DEFAULT_TERMINATION_GRACE_SECONDS",
    "DeterministicResultCache",
    "ExecutionStatus",
    "MULTI_PROVER_RESOURCE_VERSION",
    "MultiProverBundleSupervisor",
    "MultiProverResourceBudget",
    "MultiProverResourceClass",
    "MultiProverResourceController",
    "MultiProverResourceLease",
    "MultiProverResourceManager",
    "MultiProverSerialSupervisor",
    "PROVER_RESOURCE_CLASSES",
    "ProverExecutionContext",
    "ProverExecutionReceipt",
    "ProverFamily",
    "ProverResourceClass",
    "ProverResourceRequest",
    "ProverTask",
    "ProverTaskExecutor",
    "ResourceAdmission",
    "ResourceRequest",
    "ResourceUsage",
    "SerialProverSupervisor",
    "SharedProverBudget",
    "SharedResourceLease",
    "TopLevelResourceLease",
    "UnifiedProverBudget",
    "adaptive_portfolio_width",
    "dependency_closed_ready_slice",
    "normalize_prover_resource_class",
]
