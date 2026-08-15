"""Parallel mutation workers with resource admission, timeout, and cancellation (AAE-042).

Interface surface:

* ``MutationWorkerPool@1`` — bounded parallel workers for mutation campaign
  execution.  Reuses the canonical :class:`ResourceScheduler` for concurrency
  and budget admission, and the verification process-tree fence for
  timeout/cancellation reaping.

Normative properties:

* Concurrency and resource budgets are enforced through
  :class:`ResourceScheduler` leases; capacity exhaustion is typed rejection,
  never silent oversubscription.
* Network policy is fail-closed ``deny_all``; credentials and ambient secrets
  are never inherited into worker subprocesses.
* Timeouts and cancellations fence the full process tree (including escaped
  sessions) and never publish late success.
* Infrastructure events (admission, spawn, fence, restart, lease lifecycle)
  are recorded on a separate surface from semantic worker payloads so
  economics and kill rates cannot be confused with host failures.
* Durable attempt journals under an owned checkpoint root make the pool
  restartable; shutdown and crash recovery release leases and reap processes
  so the pool remains leak free.

Cold import is side-effect free: no threads, processes, network, or
filesystem operations run at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Final

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ResourceAdmissionLease,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.verification.process_runner import (
    NETWORK_POLICY_DENY_ALL,
    PROCESS_TREE_CANCELLATION_EVIDENCE,
    VerificationCancellation,
    build_hermetic_environment,
    fence_process_tree,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

MUTATION_WORKER_POOL_INTERFACE: Final[str] = "MutationWorkerPool@1"
MUTATION_WORKER_POOL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worker-pool@1"
)
MUTATION_WORKER_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worker-task@1"
)
MUTATION_WORKER_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worker-result@1"
)
MUTATION_WORKER_INFRA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worker-infrastructure@1"
)
MUTATION_WORKER_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worker-checkpoint@1"
)

ADAPTER_ID: Final[str] = "aae-mutation-worker-pool"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
AAE_MUTATION_WORKERS_EVIDENCE: Final[str] = "aae/mutation-workers@1"

DEFAULT_RESOURCE_CLASS: Final[str] = "cpu-large"
DEFAULT_STAGE: Final[str] = "execution"
DEFAULT_MAX_CONCURRENCY: Final[int] = 2
DEFAULT_TASK_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_POOL_WALL_SECONDS: Final[float] = 0.0  # 0 = unbounded pool wall
DEFAULT_TERM_GRACE_SECONDS: Final[float] = 0.25
DEFAULT_KILL_WAIT_SECONDS: Final[float] = 1.0
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 0.02
DEFAULT_MAX_DIAGNOSTIC: Final[int] = 1_024
DEFAULT_MAX_STDOUT_BYTES: Final[int] = 256 * 1024
DEFAULT_MAX_STDERR_BYTES: Final[int] = 256 * 1024
MAX_CONCURRENCY: Final[int] = 256
MAX_TIMEOUT_SECONDS: Final[float] = 86_400.0
MAX_TASK_ID_BYTES: Final[int] = 256
MAX_METADATA_ENTRIES: Final[int] = 64
MAX_METADATA_VALUE_CHARS: Final[int] = 4_096

Clock = Callable[[], float]
Sleep = Callable[[float], None]
WorkerCallable = Callable[["MutationWorkerContext"], Any]
PopenFactory = Callable[..., Any]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MutationWorkerError(ValueError):
    """Fail-closed error for mutation worker pool contracts."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mutation_worker_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class MutationWorkerPolicyError(MutationWorkerError):
    """Caller requested a disallowed policy (network widen, shell, secrets)."""


class MutationWorkerBoundsError(MutationWorkerError):
    """A bound was exceeded (concurrency, timeout, payload size)."""


# ---------------------------------------------------------------------------
# Dispositions / helpers
# ---------------------------------------------------------------------------


class MutationWorkerDisposition(str, Enum):
    """Closed terminal disposition for one worker attempt.

    Infrastructure-class outcomes (``resource_denied``, ``timeout``,
    ``cancelled``, ``infrastructure_failure``, ``network_policy_denied``)
    are never promoted as semantic mutant kill/survive evidence.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RESOURCE_DENIED = "resource_denied"
    NETWORK_POLICY_DENIED = "network_policy_denied"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"

    @property
    def is_infrastructure(self) -> bool:
        return self in {
            MutationWorkerDisposition.TIMEOUT,
            MutationWorkerDisposition.CANCELLED,
            MutationWorkerDisposition.RESOURCE_DENIED,
            MutationWorkerDisposition.NETWORK_POLICY_DENIED,
            MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
        }

    @property
    def is_semantic(self) -> bool:
        return self in {
            MutationWorkerDisposition.COMPLETED,
            MutationWorkerDisposition.FAILED,
        }


def _clip(text: str, *, limit: int = DEFAULT_MAX_DIAGNOSTIC) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _duration_ms(started: float, clock: Clock) -> int:
    return max(0, int(round((clock() - started) * 1000.0)))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(dict(payload))
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(encoded)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _normalize_timeout(value: Any, *, name: str = "timeout_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MutationWorkerBoundsError(
            f"{name} must be a positive number",
            reason_code="invalid_timeout",
        )
    timeout = float(value)
    if not (0.0 < timeout <= MAX_TIMEOUT_SECONDS):
        raise MutationWorkerBoundsError(
            f"{name} must be in (0, {MAX_TIMEOUT_SECONDS}]",
            reason_code="invalid_timeout",
            details={name: timeout},
        )
    return timeout


def _normalize_concurrency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MutationWorkerBoundsError(
            "max_concurrency must be a positive integer",
            reason_code="invalid_concurrency",
        )
    if not (1 <= value <= MAX_CONCURRENCY):
        raise MutationWorkerBoundsError(
            f"max_concurrency must be in [1, {MAX_CONCURRENCY}]",
            reason_code="invalid_concurrency",
            details={"max_concurrency": value},
        )
    return value


def _normalize_network_policy(value: Any) -> str:
    policy = str(value or "").strip() or NETWORK_POLICY_DENY_ALL
    if policy != NETWORK_POLICY_DENY_ALL:
        raise MutationWorkerPolicyError(
            "network policy must be deny_all",
            reason_code="network_policy_denied",
            details={"network_policy": policy},
        )
    return policy


def _normalize_metadata(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    raw = dict(value or {})
    if len(raw) > MAX_METADATA_ENTRIES:
        raise MutationWorkerBoundsError(
            "metadata exceeds entry bound",
            reason_code="metadata_bounds",
        )
    out: dict[str, str] = {}
    for key, item in raw.items():
        k = str(key)
        v = str(item)
        if len(v) > MAX_METADATA_VALUE_CHARS:
            raise MutationWorkerBoundsError(
                "metadata value exceeds character bound",
                reason_code="metadata_bounds",
                details={"key": k},
            )
        out[k] = v
    return MappingProxyType(out)


def _normalize_task_id(value: Any) -> str:
    task_id = str(value or "").strip()
    if not task_id:
        raise MutationWorkerError(
            "task_id is required",
            reason_code="invalid_task_id",
        )
    if len(task_id.encode("utf-8")) > MAX_TASK_ID_BYTES:
        raise MutationWorkerBoundsError(
            "task_id exceeds byte bound",
            reason_code="invalid_task_id",
        )
    if any(ch in task_id for ch in ("/", "\\", "\0", "\n", "\r")):
        raise MutationWorkerError(
            "task_id must not contain path separators or control characters",
            reason_code="invalid_task_id",
        )
    return task_id


def _safe_checkpoint_name(task_id: str) -> str:
    digest = _sha256_hex(task_id.encode("utf-8"))[:24]
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id
    )[:48]
    return f"{cleaned}.{digest}.json"


def _temporary_binary_stream() -> BinaryIO:
    """Return one temporary binary stream for bounded subprocess capture."""

    return tempfile.TemporaryFile(mode="w+b")


def _read_stream_bytes(stream: BinaryIO | None, *, maximum: int) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    try:
        stream.seek(0)
        data = stream.read() or b""
    except OSError:
        return b"", False
    if len(data) > maximum:
        return data[:maximum], True
    return data, False


def _default_host_snapshot(*, worker_limit: int = 8) -> HostResourceSnapshot:
    limit = max(1, int(worker_limit))
    return HostResourceSnapshot(
        worker_limit=limit,
        available_worker_capacity=limit,
        active_workers=0,
        memory_available_bytes=8 * 1024 * 1024 * 1024,
        disk_available_bytes=8 * 1024 * 1024 * 1024,
        memory_total_bytes=16 * 1024 * 1024 * 1024,
        disk_total_bytes=64 * 1024 * 1024 * 1024,
        capabilities=("cpu",),
        resource_classes=(
            DEFAULT_RESOURCE_CLASS,
            "cpu-small",
            "cpu-medium",
            "cpu-validation",
            "cpu-proof-solver",
            "cpu-proof-type-check",
        ),
    )


def _json_safe(value: Any, *, limit: int = DEFAULT_MAX_DIAGNOSTIC) -> Any:
    """Return a JSON-safe, bounded projection of *value*."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        # Profile-G artifacts reject floats; store as integer millis when whole.
        if value.is_integer():
            return int(value)
        return _clip(repr(value), limit=limit)
    if isinstance(value, str):
        return _clip(value, limit=limit)
    if isinstance(value, bytes):
        return {
            "encoding": "hex",
            "byte_length": len(value),
            "sha256": _sha256_hex(value),
            "preview_hex": value[:64].hex(),
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_METADATA_ENTRIES:
                out["__truncated__"] = True
                break
            out[str(key)] = _json_safe(item, limit=limit)
        return out
    if isinstance(value, (list, tuple)):
        items = [_json_safe(item, limit=limit) for item in list(value)[:64]]
        if len(value) > 64:
            items.append({"__truncated__": True, "length": len(value)})
        return items
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict(), limit=limit)
        except Exception:  # noqa: BLE001 - infrastructure projection only
            return _clip(repr(value), limit=limit)
    return _clip(repr(value), limit=limit)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class MutationWorkerCancellation:
    """Identity-fenced cooperative cancellation for mutation workers.

    Mirrors the verification cancellation fence so late publication after
    cancel is impossible even if a worker subprocess exits zero.
    """

    def __init__(self, cancellation_id: str | None = None) -> None:
        identity = (
            str(cancellation_id or "").strip()
            or f"aae-worker-cancel:{uuid.uuid4().hex}"
        )
        self.cancellation_id = identity
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def cancel(
        self,
        *,
        cancellation_id: str | None = None,
        reason: str = "cancelled",
    ) -> bool:
        presented = (
            self.cancellation_id
            if cancellation_id is None
            else str(cancellation_id or "").strip()
        )
        if presented != self.cancellation_id:
            return False
        with self._lock:
            if self._event.is_set():
                return True
            self._reason = str(reason or "cancelled").strip() or "cancelled"
            self._event.set()
            return True

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancelled(self) -> bool:
        return self.is_cancelled()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def __bool__(self) -> bool:
        return self.is_cancelled()

    def as_verification_cancellation(self) -> VerificationCancellation:
        """Bridge to the verification process-tree cancellation surface."""

        token = VerificationCancellation(cancellation_id=self.cancellation_id)
        if self.is_cancelled():
            token.cancel(
                cancellation_id=self.cancellation_id,
                reason=self.reason or "cancelled",
            )
        return token


# ---------------------------------------------------------------------------
# Policy / budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationWorkerBudget:
    """Hard resource envelope for one worker pool.

    Intersects with :class:`ResourceScheduler` admission.  Zero-valued host
    budgets mean "do not impose an extra scheduler budget beyond concurrency".
    """

    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_processes: int = 0
    wall_time_ms: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    default_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    pool_wall_seconds: float = DEFAULT_POOL_WALL_SECONDS
    network_policy: str = NETWORK_POLICY_DENY_ALL
    resource_class: str = DEFAULT_RESOURCE_CLASS
    stage: str = DEFAULT_STAGE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_concurrency", _normalize_concurrency(self.max_concurrency)
        )
        for name in ("max_processes", "wall_time_ms", "memory_bytes", "disk_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MutationWorkerBoundsError(
                    f"{name} must be a non-negative integer",
                    reason_code="invalid_budget",
                )
        processes = (
            self.max_processes
            if self.max_processes > 0
            else self.max_concurrency
        )
        object.__setattr__(self, "max_processes", processes)
        object.__setattr__(
            self,
            "default_timeout_seconds",
            _normalize_timeout(
                self.default_timeout_seconds, name="default_timeout_seconds"
            ),
        )
        pool_wall = float(self.pool_wall_seconds or 0.0)
        if pool_wall < 0.0 or pool_wall > MAX_TIMEOUT_SECONDS:
            raise MutationWorkerBoundsError(
                "pool_wall_seconds must be in [0, MAX_TIMEOUT_SECONDS]",
                reason_code="invalid_budget",
            )
        object.__setattr__(self, "pool_wall_seconds", pool_wall)
        object.__setattr__(
            self, "network_policy", _normalize_network_policy(self.network_policy)
        )
        resource_class = str(self.resource_class or DEFAULT_RESOURCE_CLASS).strip()
        if not resource_class:
            raise MutationWorkerError(
                "resource_class is required",
                reason_code="invalid_budget",
            )
        object.__setattr__(self, "resource_class", resource_class)
        stage = str(self.stage or DEFAULT_STAGE).strip() or DEFAULT_STAGE
        object.__setattr__(self, "stage", stage)

    def as_resource_lease_budget(self) -> ResourceLeaseBudget:
        return ResourceLeaseBudget(
            max_parallel=self.max_concurrency,
            max_cpu_proof_concurrency=self.max_concurrency,
            max_model_concurrency=1,
            max_artifact_concurrency=self.max_concurrency,
            max_processes=self.max_processes,
            wall_time_ms=self.wall_time_ms,
            memory_bytes=self.memory_bytes,
            disk_bytes=self.disk_bytes,
        )

    def as_resource_policy(self) -> ResourcePolicy:
        return ResourcePolicy(
            max_lanes=self.max_concurrency,
            max_cpu_proof_concurrency=self.max_concurrency,
            max_model_concurrency=1,
            max_artifact_concurrency=self.max_concurrency,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.max_concurrency,
            "max_processes": self.max_processes,
            "wall_time_ms": self.wall_time_ms,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "default_timeout_seconds": self.default_timeout_seconds,
            "pool_wall_seconds": self.pool_wall_seconds,
            "network_policy": self.network_policy,
            "resource_class": self.resource_class,
            "stage": self.stage,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MutationWorkerBudget":
        data = dict(value or {})
        return cls(
            max_concurrency=int(
                data.get("max_concurrency", DEFAULT_MAX_CONCURRENCY)
            ),
            max_processes=int(data.get("max_processes", 0) or 0),
            wall_time_ms=int(data.get("wall_time_ms", 0) or 0),
            memory_bytes=int(data.get("memory_bytes", 0) or 0),
            disk_bytes=int(data.get("disk_bytes", 0) or 0),
            default_timeout_seconds=float(
                data.get("default_timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS)
            ),
            pool_wall_seconds=float(data.get("pool_wall_seconds", 0.0) or 0.0),
            network_policy=str(
                data.get("network_policy", NETWORK_POLICY_DENY_ALL)
                or NETWORK_POLICY_DENY_ALL
            ),
            resource_class=str(
                data.get("resource_class", DEFAULT_RESOURCE_CLASS)
                or DEFAULT_RESOURCE_CLASS
            ),
            stage=str(data.get("stage", DEFAULT_STAGE) or DEFAULT_STAGE),
        )


# ---------------------------------------------------------------------------
# Task / context / infrastructure / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationWorkerTask:
    """One admitted unit of mutation-worker work.

    Exactly one of *runner* or *command* must be provided.  *runner* is the
    in-process path used by higher layers (worktree apply/admit, incremental
    verification hooks).  *command* is an explicit argv subprocess path that
    is process-tree fenced on timeout/cancel.
    """

    task_id: str
    runner: WorkerCallable | None = None
    command: Sequence[str] = ()
    cwd: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    lane_id: str = ""
    resource_class: str = ""
    stage: str = ""
    memory_bytes: int = 0
    disk_bytes: int = 0
    process_slots: int = 1
    candidate_id: str = ""
    candidate_cid: str = ""
    network_policy: str = NETWORK_POLICY_DENY_ALL
    metadata: Mapping[str, str] = field(default_factory=dict)
    stdin: bytes | str | None = None
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES

    def __post_init__(self) -> None:
        task_id = _normalize_task_id(self.task_id)
        object.__setattr__(self, "task_id", task_id)
        has_runner = self.runner is not None
        command = tuple(str(item) for item in (self.command or ()))
        if has_runner == bool(command):
            raise MutationWorkerError(
                "exactly one of runner or command is required",
                reason_code="invalid_task",
            )
        if command:
            if not all(item for item in command):
                raise MutationWorkerError(
                    "command argv items must be non-empty strings",
                    reason_code="invalid_command",
                )
            if any(os.sep in item or (os.altsep and os.altsep in item) for item in command[1:3]):
                # Paths are allowed; shell metacharacters are not interpolated
                # because shell is always false.  No further restriction.
                pass
        object.__setattr__(self, "command", command)
        if self.timeout_seconds is not None:
            object.__setattr__(
                self,
                "timeout_seconds",
                _normalize_timeout(self.timeout_seconds),
            )
        object.__setattr__(
            self, "network_policy", _normalize_network_policy(self.network_policy)
        )
        if isinstance(self.process_slots, bool) or int(self.process_slots) <= 0:
            raise MutationWorkerBoundsError(
                "process_slots must be a positive integer",
                reason_code="invalid_task",
            )
        object.__setattr__(self, "process_slots", int(self.process_slots))
        for name in ("memory_bytes", "disk_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MutationWorkerBoundsError(
                    f"{name} must be a non-negative integer",
                    reason_code="invalid_task",
                )
        for name in ("max_stdout_bytes", "max_stderr_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MutationWorkerBoundsError(
                    f"{name} must be a positive integer",
                    reason_code="invalid_task",
                )
        env = {
            str(key): str(value)
            for key, value in dict(self.environment or {}).items()
        }
        object.__setattr__(self, "environment", MappingProxyType(env))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        object.__setattr__(self, "lane_id", str(self.lane_id or "").strip())
        object.__setattr__(
            self, "resource_class", str(self.resource_class or "").strip()
        )
        object.__setattr__(self, "stage", str(self.stage or "").strip())
        object.__setattr__(self, "candidate_id", str(self.candidate_id or "").strip())
        object.__setattr__(self, "candidate_cid", str(self.candidate_cid or "").strip())
        object.__setattr__(
            self,
            "cwd",
            str(self.cwd).strip() if self.cwd is not None else None,
        )
        if isinstance(self.stdin, str):
            object.__setattr__(self, "stdin", self.stdin.encode("utf-8"))

    @property
    def is_command(self) -> bool:
        return bool(self.command)

    def effective_timeout(self, budget: MutationWorkerBudget) -> float:
        if self.timeout_seconds is not None:
            return float(self.timeout_seconds)
        return float(budget.default_timeout_seconds)

    def effective_lane_id(self, pool_id: str) -> str:
        if self.lane_id:
            return self.lane_id
        return f"aae-worker:{pool_id}:{self.task_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_WORKER_TASK_SCHEMA,
            "task_id": self.task_id,
            "has_runner": self.runner is not None,
            "command": list(self.command),
            "cwd": self.cwd,
            "timeout_seconds": self.timeout_seconds,
            "lane_id": self.lane_id,
            "resource_class": self.resource_class,
            "stage": self.stage,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "process_slots": self.process_slots,
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "network_policy": self.network_policy,
            "metadata": dict(self.metadata),
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "environment_keys": sorted(self.environment),
        }


@dataclass
class MutationWorkerContext:
    """Runtime context passed to in-process worker callables."""

    task: MutationWorkerTask
    cancellation: MutationWorkerCancellation
    lease_id: str
    pool_id: str
    network_policy: str
    deadline_monotonic: float | None
    clock: Clock = time.monotonic

    @property
    def cancelled(self) -> bool:
        return self.cancellation.is_cancelled()

    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - self.clock())

    def check_cancelled(self) -> None:
        if self.cancellation.is_cancelled():
            raise MutationWorkerError(
                self.cancellation.reason or "cancelled",
                reason_code="cancelled",
            )
        if (
            self.deadline_monotonic is not None
            and self.clock() >= self.deadline_monotonic
        ):
            raise MutationWorkerError(
                "worker task exceeded its wall-time limit",
                reason_code="timeout",
            )


@dataclass(frozen=True)
class MutationWorkerInfrastructureRecord:
    """Host/process infrastructure evidence, separate from semantic payload.

    Economics, kill rates, and assurance outcomes must not absorb these
    events as if they were mutant detections.
    """

    disposition: MutationWorkerDisposition
    reason_codes: tuple[str, ...] = ()
    diagnostic: str = ""
    lease_id: str = ""
    admission_admitted: bool | None = None
    admission_reasons: tuple[str, ...] = ()
    process_started: bool = False
    pid: int | None = None
    process_group_id: int | None = None
    process_tree_fenced: bool = False
    timed_out: bool = False
    cancelled: bool = False
    publication_allowed: bool = False
    restart_recovered: bool = False
    network_policy: str = NETWORK_POLICY_DENY_ALL
    resource_class: str = DEFAULT_RESOURCE_CLASS
    stage: str = DEFAULT_STAGE
    duration_ms: int = 0
    exit_code: int | None = None
    stdout_digest: str = ""
    stderr_digest: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    checkpoint_path: str = ""
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, MutationWorkerDisposition):
            object.__setattr__(
                self,
                "disposition",
                MutationWorkerDisposition(str(self.disposition)),
            )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item)),
        )
        object.__setattr__(self, "diagnostic", _clip(self.diagnostic))
        object.__setattr__(
            self,
            "admission_reasons",
            tuple(str(item) for item in self.admission_reasons if str(item)),
        )
        object.__setattr__(
            self,
            "events",
            tuple(MappingProxyType(dict(item)) for item in self.events),
        )
        object.__setattr__(
            self, "network_policy", _normalize_network_policy(self.network_policy)
        )

    @property
    def is_infrastructure(self) -> bool:
        return self.disposition.is_infrastructure

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_WORKER_INFRA_SCHEMA,
            "disposition": self.disposition.value,
            "is_infrastructure": self.is_infrastructure,
            "reason_codes": list(self.reason_codes),
            "diagnostic": self.diagnostic,
            "lease_id": self.lease_id,
            "admission_admitted": self.admission_admitted,
            "admission_reasons": list(self.admission_reasons),
            "process_started": self.process_started,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "process_tree_fenced": self.process_tree_fenced,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "publication_allowed": self.publication_allowed,
            "restart_recovered": self.restart_recovered,
            "network_policy": self.network_policy,
            "resource_class": self.resource_class,
            "stage": self.stage,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "stderr_digest": self.stderr_digest,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "checkpoint_path": self.checkpoint_path,
            "events": [dict(item) for item in self.events],
            "process_tree_cancellation_evidence": PROCESS_TREE_CANCELLATION_EVIDENCE,
        }


@dataclass(frozen=True)
class MutationWorkerResult:
    """Sealed outcome of one worker attempt.

    *payload* holds the semantic worker return value (or command observation)
    only when the disposition is semantic.  *infrastructure* is always present
    and is the sole authority for host/process failure classification.
    """

    task_id: str
    disposition: MutationWorkerDisposition
    infrastructure: MutationWorkerInfrastructureRecord
    payload: Any = None
    candidate_id: str = ""
    candidate_cid: str = ""
    pool_id: str = ""
    cancellation_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, MutationWorkerDisposition):
            object.__setattr__(
                self,
                "disposition",
                MutationWorkerDisposition(str(self.disposition)),
            )
        if not isinstance(
            self.infrastructure, MutationWorkerInfrastructureRecord
        ):
            raise MutationWorkerError(
                "infrastructure must be MutationWorkerInfrastructureRecord",
                reason_code="invalid_result",
            )
        if self.disposition != self.infrastructure.disposition:
            raise MutationWorkerError(
                "result disposition must match infrastructure disposition",
                reason_code="invalid_result",
            )
        object.__setattr__(
            self, "metadata", _normalize_metadata(dict(self.metadata or {}))
        )
        # Never attach semantic payloads to infrastructure-class dispositions.
        if self.disposition.is_infrastructure and self.payload is not None:
            object.__setattr__(self, "payload", None)

    @property
    def is_infrastructure(self) -> bool:
        return self.disposition.is_infrastructure

    @property
    def publication_allowed(self) -> bool:
        return bool(self.infrastructure.publication_allowed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_WORKER_RESULT_SCHEMA,
            "interface": MUTATION_WORKER_POOL_INTERFACE,
            "task_id": self.task_id,
            "disposition": self.disposition.value,
            "is_infrastructure": self.is_infrastructure,
            "publication_allowed": self.publication_allowed,
            "payload": _json_safe(self.payload) if self.payload is not None else None,
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "pool_id": self.pool_id,
            "cancellation_id": self.cancellation_id,
            "metadata": dict(self.metadata),
            "infrastructure": self.infrastructure.to_dict(),
            "evidence": AAE_MUTATION_WORKERS_EVIDENCE,
        }


# ---------------------------------------------------------------------------
# Checkpoint journal (restartability)
# ---------------------------------------------------------------------------


class MutationWorkerCheckpointStore:
    """Atomic per-task journals under an owned checkpoint directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def path_for(self, task_id: str) -> Path:
        return self.root / _safe_checkpoint_name(task_id)

    def write(
        self,
        task_id: str,
        *,
        phase: str,
        payload: Mapping[str, Any],
    ) -> Path:
        body = {
            "schema": MUTATION_WORKER_CHECKPOINT_SCHEMA,
            "task_id": task_id,
            "phase": str(phase),
            "updated_at_ms": int(time.time() * 1000),
            "payload": dict(payload),
        }
        path = self.path_for(task_id)
        with self._lock:
            _atomic_write_json(path, body)
        return path

    def read(self, task_id: str) -> dict[str, Any] | None:
        return _load_json_dict(self.path_for(task_id))

    def mark_complete(self, task_id: str, result: MutationWorkerResult) -> Path:
        return self.write(
            task_id,
            phase="complete",
            payload={
                "result": result.to_dict(),
                "disposition": result.disposition.value,
            },
        )

    def mark_running(
        self,
        task_id: str,
        *,
        lease_id: str,
        pool_id: str,
        attempt: int,
    ) -> Path:
        return self.write(
            task_id,
            phase="running",
            payload={
                "lease_id": lease_id,
                "pool_id": pool_id,
                "attempt": int(attempt),
            },
        )

    def list_incomplete(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        with self._lock:
            for path in sorted(self.root.glob("*.json")):
                data = _load_json_dict(path)
                if not data:
                    continue
                if str(data.get("phase") or "") in {"complete", "released"}:
                    continue
                records.append(data)
        return tuple(records)

    def recover_incomplete_as_infrastructure(
        self,
        *,
        pool_id: str,
        reason: str = "restart_recovered_incomplete",
    ) -> tuple[MutationWorkerResult, ...]:
        """Seal incomplete journals as infrastructure failures after restart."""

        results: list[MutationWorkerResult] = []
        for record in self.list_incomplete():
            task_id = str(record.get("task_id") or "")
            if not task_id:
                continue
            payload = dict(record.get("payload") or {})
            infra = MutationWorkerInfrastructureRecord(
                disposition=MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                reason_codes=(reason, "incomplete_checkpoint"),
                diagnostic=_clip(
                    f"recovered incomplete worker checkpoint phase="
                    f"{record.get('phase')}"
                ),
                lease_id=str(payload.get("lease_id") or ""),
                admission_admitted=True,
                process_started=False,
                publication_allowed=False,
                restart_recovered=True,
                checkpoint_path=str(self.path_for(task_id)),
                events=(
                    {
                        "event": "restart_recovery",
                        "phase": str(record.get("phase") or ""),
                        "pool_id": pool_id,
                    },
                ),
            )
            result = MutationWorkerResult(
                task_id=task_id,
                disposition=MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                infrastructure=infra,
                pool_id=pool_id,
            )
            self.mark_complete(task_id, result)
            results.append(result)
        return tuple(results)

    def clear(self) -> None:
        with self._lock:
            if self.root.exists():
                for path in self.root.glob("*.json"):
                    try:
                        path.unlink()
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class MutationWorkerPool:
    """Bounded parallel mutation workers (``MutationWorkerPool@1``).

    Parameters
    ----------
    budget:
        Concurrency, timeout, and resource envelope.
    resource_scheduler:
        Shared or private :class:`ResourceScheduler`.  When omitted a private
        scheduler is created from *budget*.
    host_snapshot:
        Optional host capacity snapshot for every lease acquisition.  Tests
        inject a generous snapshot; production may leave this ``None`` so the
        scheduler samples live host telemetry.
    checkpoint_dir:
        Owned durable journal root.  When omitted the pool is still correct
        for a single process lifetime but is not restart-durable.
    """

    def __init__(
        self,
        budget: MutationWorkerBudget | Mapping[str, Any] | None = None,
        *,
        resource_scheduler: ResourceScheduler | None = None,
        host_snapshot: HostResourceSnapshot | Mapping[str, Any] | None = None,
        checkpoint_dir: Path | str | None = None,
        pool_id: str | None = None,
        popen_factory: PopenFactory | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
        term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
        kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        require_resource_lease: bool = True,
        thread_name_prefix: str = "aae-mutation-worker",
    ) -> None:
        self.budget = (
            budget
            if isinstance(budget, MutationWorkerBudget)
            else MutationWorkerBudget.from_mapping(budget)
        )
        self.pool_id = (
            str(pool_id or "").strip() or f"aae-pool:{uuid.uuid4().hex[:16]}"
        )
        self._scheduler = resource_scheduler or ResourceScheduler(
            self.budget.as_resource_policy()
        )
        self._host_snapshot = host_snapshot
        self._popen: PopenFactory = popen_factory or subprocess.Popen
        self._clock: Clock = clock or time.monotonic
        self._sleep: Sleep = sleep or time.sleep
        self._term_grace_seconds = float(term_grace_seconds)
        self._kill_wait_seconds = float(kill_wait_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._require_resource_lease = bool(require_resource_lease)
        self._checkpoint = (
            MutationWorkerCheckpointStore(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        self._lock = threading.RLock()
        self._closed = False
        self._pool_cancel = MutationWorkerCancellation(
            cancellation_id=f"aae-pool-cancel:{self.pool_id}"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.budget.max_concurrency,
            thread_name_prefix=thread_name_prefix,
        )
        self._inflight: dict[str, Future[MutationWorkerResult]] = {}
        self._active_leases: dict[str, ResourceAdmissionLease] = {}
        self._active_processes: dict[str, Any] = {}
        self._results: dict[str, MutationWorkerResult] = {}
        self._infrastructure_log: list[dict[str, Any]] = []
        self._started_at = self._clock()
        self._attempt_counts: dict[str, int] = {}

    # -- construction helpers ----------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        default_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
        network_policy: str = NETWORK_POLICY_DENY_ALL,
        resource_class: str = DEFAULT_RESOURCE_CLASS,
        checkpoint_dir: Path | str | None = None,
        resource_scheduler: ResourceScheduler | None = None,
        host_snapshot: HostResourceSnapshot | Mapping[str, Any] | None = None,
        pool_id: str | None = None,
        **budget_fields: Any,
    ) -> "MutationWorkerPool":
        budget = MutationWorkerBudget(
            max_concurrency=max_concurrency,
            default_timeout_seconds=default_timeout_seconds,
            network_policy=network_policy,
            resource_class=resource_class,
            **budget_fields,
        )
        return cls(
            budget,
            resource_scheduler=resource_scheduler,
            host_snapshot=host_snapshot,
            checkpoint_dir=checkpoint_dir,
            pool_id=pool_id,
        )

    # -- public API --------------------------------------------------------

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for future in self._inflight.values() if not future.done())

    @property
    def infrastructure_records(self) -> tuple[dict[str, Any], ...]:
        """Append-only infrastructure event log (never mixed into payloads)."""

        with self._lock:
            return tuple(dict(item) for item in self._infrastructure_log)

    def submit(
        self,
        task: MutationWorkerTask,
        *,
        cancellation: MutationWorkerCancellation | None = None,
    ) -> Future[MutationWorkerResult]:
        """Admit and schedule *task*.  Returns a future for the sealed result."""

        if not isinstance(task, MutationWorkerTask):
            raise MutationWorkerError(
                "task must be a MutationWorkerTask",
                reason_code="invalid_task",
            )
        with self._lock:
            if self._closed:
                raise MutationWorkerError(
                    "worker pool is closed",
                    reason_code="pool_closed",
                )
            if self._pool_cancel.is_cancelled():
                raise MutationWorkerError(
                    "worker pool is cancelled",
                    reason_code="pool_cancelled",
                )
            if task.task_id in self._inflight and not self._inflight[task.task_id].done():
                raise MutationWorkerError(
                    "task_id already in flight",
                    reason_code="duplicate_task",
                    details={"task_id": task.task_id},
                )
            if self.budget.pool_wall_seconds > 0.0:
                elapsed = self._clock() - self._started_at
                if elapsed >= self.budget.pool_wall_seconds:
                    raise MutationWorkerError(
                        "pool wall-time budget exhausted",
                        reason_code="pool_wall_exhausted",
                    )
            future = self._executor.submit(
                self._execute_task, task, cancellation
            )
            self._inflight[task.task_id] = future
        future.add_done_callback(
            lambda fut, task_id=task.task_id: self._on_future_done(task_id, fut)
        )
        return future

    def run(
        self,
        task: MutationWorkerTask,
        *,
        cancellation: MutationWorkerCancellation | None = None,
    ) -> MutationWorkerResult:
        """Submit *task* and wait for its sealed result."""

        return self.submit(task, cancellation=cancellation).result()

    def map(
        self,
        tasks: Sequence[MutationWorkerTask],
        *,
        cancellation: MutationWorkerCancellation | None = None,
        return_when_complete: bool = True,
    ) -> list[MutationWorkerResult]:
        """Run *tasks* under pool concurrency and return sealed results.

        Result order matches *tasks* order.  Individual infrastructure
        failures do not abort siblings.
        """

        if not tasks:
            return []
        seen: set[str] = set()
        for task in tasks:
            if not isinstance(task, MutationWorkerTask):
                raise MutationWorkerError(
                    "tasks must be MutationWorkerTask instances",
                    reason_code="invalid_task",
                )
            if task.task_id in seen:
                raise MutationWorkerError(
                    "duplicate task_id in map()",
                    reason_code="duplicate_task",
                    details={"task_id": task.task_id},
                )
            seen.add(task.task_id)
        futures = [
            (task.task_id, self.submit(task, cancellation=cancellation))
            for task in tasks
        ]
        if return_when_complete:
            for _task_id, future in futures:
                # Drain exceptions into sealed results inside _execute_task.
                future.result()
        by_id = {
            task_id: future.result()
            for task_id, future in futures
        }
        return [by_id[task.task_id] for task in tasks]

    def cancel(
        self,
        *,
        reason: str = "pool_cancelled",
        cancellation_id: str | None = None,
    ) -> bool:
        """Cancel the pool and fence every in-flight process tree."""

        ok = self._pool_cancel.cancel(
            cancellation_id=cancellation_id, reason=reason
        )
        with self._lock:
            processes = list(self._active_processes.items())
            leases = list(self._active_leases.items())
        for _task_id, process in processes:
            try:
                fence_process_tree(
                    process,
                    grace_seconds=self._term_grace_seconds,
                    kill_wait_seconds=self._kill_wait_seconds,
                    require_gone=True,
                )
            except Exception:  # noqa: BLE001 - best-effort fence
                pass
        for _task_id, lease in leases:
            try:
                self._scheduler.release(lease, reason=reason)
            except Exception:  # noqa: BLE001 - best-effort release
                pass
        self._record_infra_event(
            {
                "event": "pool_cancel",
                "reason": reason,
                "process_count": len(processes),
                "lease_count": len(leases),
            }
        )
        return ok

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel: bool = False,
        reason: str = "pool_shutdown",
    ) -> None:
        """Stop accepting work, optionally cancel, and release all resources."""

        with self._lock:
            if self._closed and not self._inflight:
                return
            self._closed = True
        if cancel:
            self.cancel(reason=reason)
        self._executor.shutdown(wait=wait, cancel_futures=cancel)
        with self._lock:
            leases = list(self._active_leases.items())
            processes = list(self._active_processes.items())
            self._active_leases.clear()
            self._active_processes.clear()
            self._inflight.clear()
        for _task_id, process in processes:
            try:
                fence_process_tree(
                    process,
                    grace_seconds=self._term_grace_seconds,
                    kill_wait_seconds=self._kill_wait_seconds,
                    require_gone=True,
                )
            except Exception:  # noqa: BLE001
                pass
        for _task_id, lease in leases:
            try:
                self._scheduler.release(lease, reason=reason)
            except Exception:  # noqa: BLE001
                pass
        self._record_infra_event(
            {
                "event": "pool_shutdown",
                "reason": reason,
                "wait": bool(wait),
                "cancel": bool(cancel),
            }
        )

    def recover(self) -> tuple[MutationWorkerResult, ...]:
        """Seal incomplete checkpoints as infrastructure failures after restart."""

        if self._checkpoint is None:
            return ()
        recovered = self._checkpoint.recover_incomplete_as_infrastructure(
            pool_id=self.pool_id
        )
        with self._lock:
            for result in recovered:
                self._results[result.task_id] = result
                self._infrastructure_log.append(result.infrastructure.to_dict())
        return recovered

    def get_result(self, task_id: str) -> MutationWorkerResult | None:
        with self._lock:
            return self._results.get(task_id)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": MUTATION_WORKER_POOL_SCHEMA,
                "interface": MUTATION_WORKER_POOL_INTERFACE,
                "adapter_id": ADAPTER_ID,
                "board_namespace": BOARD_NAMESPACE,
                "pool_id": self.pool_id,
                "budget": self.budget.to_dict(),
                "closed": self._closed,
                "cancelled": self._pool_cancel.is_cancelled(),
                "active_count": sum(
                    1 for future in self._inflight.values() if not future.done()
                ),
                "completed_count": len(self._results),
                "infrastructure_event_count": len(self._infrastructure_log),
                "checkpoint_dir": (
                    str(self._checkpoint.root) if self._checkpoint else ""
                ),
                "evidence": AAE_MUTATION_WORKERS_EVIDENCE,
                "process_tree_cancellation_evidence": PROCESS_TREE_CANCELLATION_EVIDENCE,
            }

    def __enter__(self) -> "MutationWorkerPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True, cancel=exc_type is not None)

    # -- internals ---------------------------------------------------------

    def _on_future_done(
        self, task_id: str, future: Future[MutationWorkerResult]
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - seal unexpected executor errors
            infra = MutationWorkerInfrastructureRecord(
                disposition=MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                reason_codes=("executor_error", type(exc).__name__),
                diagnostic=_clip(str(exc)),
                publication_allowed=False,
            )
            result = MutationWorkerResult(
                task_id=task_id,
                disposition=MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                infrastructure=infra,
                pool_id=self.pool_id,
            )
        with self._lock:
            self._results[task_id] = result
            self._infrastructure_log.append(result.infrastructure.to_dict())
            self._inflight.pop(task_id, None)
            self._active_leases.pop(task_id, None)
            self._active_processes.pop(task_id, None)

    def _record_infra_event(self, event: Mapping[str, Any]) -> None:
        body = {
            "schema": MUTATION_WORKER_INFRA_SCHEMA,
            "pool_id": self.pool_id,
            "observed_at_ms": int(time.time() * 1000),
            **dict(event),
        }
        with self._lock:
            self._infrastructure_log.append(body)

    def _host(self) -> HostResourceSnapshot:
        if self._host_snapshot is None:
            return _default_host_snapshot(worker_limit=self.budget.max_concurrency)
        if isinstance(self._host_snapshot, HostResourceSnapshot):
            return self._host_snapshot
        return HostResourceSnapshot.from_mapping(self._host_snapshot)

    def _acquire_lease(
        self, task: MutationWorkerTask
    ) -> tuple[Any, ResourceAdmissionLease | None]:
        requirement = LaneResourceRequirements(
            lane_id=task.effective_lane_id(self.pool_id),
            stage=task.stage or self.budget.stage,
            resource_class=task.resource_class or self.budget.resource_class,
            process_slots=task.process_slots,
            requires_provider=False,
            memory_bytes=task.memory_bytes,
            disk_bytes=task.disk_bytes,
        )
        decision, lease = self._scheduler.acquire(
            requirement,
            budget=self.budget.as_resource_lease_budget(),
            host=self._host(),
        )
        return decision, lease

    def _lease_still_held(self, lease: ResourceAdmissionLease) -> bool:
        active_leases = self._scheduler.active_leases
        if callable(active_leases):
            active_leases = active_leases()
        return lease.lease_id in {item.lease_id for item in active_leases}

    def _execute_task(
        self,
        task: MutationWorkerTask,
        cancellation: MutationWorkerCancellation | None,
    ) -> MutationWorkerResult:
        started = self._clock()
        # Do not use ``cancellation or ...``: MutationWorkerCancellation is
        # truthy only when already cancelled (see ``__bool__``).
        cancel = (
            cancellation
            if cancellation is not None
            else MutationWorkerCancellation(
                cancellation_id=f"aae-task-cancel:{self.pool_id}:{task.task_id}"
            )
        )
        events: list[dict[str, Any]] = []
        lease: ResourceAdmissionLease | None = None
        lease_id = ""
        decision = None
        checkpoint_path = ""
        process: Any = None
        process_started = False
        pid: int | None = None
        process_group_id: int | None = None
        process_tree_fenced = False
        timed_out = False
        cancelled = False
        exit_code: int | None = None
        stdout_digest = ""
        stderr_digest = ""
        stdout_truncated = False
        stderr_truncated = False
        payload: Any = None
        reason_codes: list[str] = []
        diagnostic = ""
        disposition = MutationWorkerDisposition.INFRASTRUCTURE_FAILURE
        publication_allowed = False
        resource_class = task.resource_class or self.budget.resource_class
        stage = task.stage or self.budget.stage
        network_policy = task.network_policy

        def seal(
            final_disposition: MutationWorkerDisposition,
            *,
            final_payload: Any = None,
            allow_publication: bool | None = None,
        ) -> MutationWorkerResult:
            nonlocal publication_allowed
            if allow_publication is None:
                allow_publication = final_disposition.is_semantic
            if final_disposition.is_infrastructure:
                allow_publication = False
                final_payload = None
            if cancelled or self._pool_cancel.is_cancelled():
                # Late-success fence: cancel always wins.
                if final_disposition.is_semantic:
                    final_disposition = MutationWorkerDisposition.CANCELLED
                    final_payload = None
                    allow_publication = False
                    if "cancelled" not in reason_codes:
                        reason_codes.append("cancelled")
            infra = MutationWorkerInfrastructureRecord(
                disposition=final_disposition,
                reason_codes=tuple(dict.fromkeys(reason_codes)),
                diagnostic=_clip(diagnostic),
                lease_id=lease_id,
                admission_admitted=(
                    None if decision is None else bool(decision.admitted)
                ),
                admission_reasons=tuple(
                    getattr(decision, "reasons", ()) or ()
                ),
                process_started=process_started,
                pid=pid,
                process_group_id=process_group_id,
                process_tree_fenced=process_tree_fenced,
                timed_out=timed_out
                or final_disposition is MutationWorkerDisposition.TIMEOUT,
                cancelled=cancelled
                or final_disposition is MutationWorkerDisposition.CANCELLED
                or self._pool_cancel.is_cancelled(),
                publication_allowed=bool(allow_publication),
                restart_recovered=False,
                network_policy=network_policy,
                resource_class=resource_class,
                stage=stage,
                duration_ms=_duration_ms(started, self._clock),
                exit_code=exit_code,
                stdout_digest=stdout_digest,
                stderr_digest=stderr_digest,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                checkpoint_path=checkpoint_path,
                events=tuple(events),
            )
            result = MutationWorkerResult(
                task_id=task.task_id,
                disposition=final_disposition,
                infrastructure=infra,
                payload=final_payload,
                candidate_id=task.candidate_id,
                candidate_cid=task.candidate_cid,
                pool_id=self.pool_id,
                cancellation_id=cancel.cancellation_id,
                metadata=task.metadata,
            )
            if self._checkpoint is not None:
                try:
                    self._checkpoint.mark_complete(task.task_id, result)
                except OSError as exc:
                    events.append(
                        {
                            "event": "checkpoint_complete_failed",
                            "error": type(exc).__name__,
                        }
                    )
            return result

        try:
            # Policy fences before admission.
            try:
                network_policy = _normalize_network_policy(task.network_policy)
            except MutationWorkerPolicyError as exc:
                reason_codes.append(exc.reason_code)
                diagnostic = str(exc)
                return seal(MutationWorkerDisposition.NETWORK_POLICY_DENIED)

            if cancel.is_cancelled() or self._pool_cancel.is_cancelled():
                cancelled = True
                reason_codes.append("cancelled_before_admit")
                diagnostic = (
                    cancel.reason
                    or self._pool_cancel.reason
                    or "cancelled before admission"
                )
                return seal(MutationWorkerDisposition.CANCELLED)

            if self._require_resource_lease:
                decision, lease = self._acquire_lease(task)
                events.append(
                    {
                        "event": "resource_admission",
                        "admitted": bool(decision.admitted),
                        "reasons": list(getattr(decision, "reasons", ()) or ()),
                        "lane_id": task.effective_lane_id(self.pool_id),
                    }
                )
                if lease is None or not decision.admitted:
                    reason_codes.append("resource_lease_denied")
                    reason_codes.extend(
                        str(item) for item in (decision.reasons or ())
                    )
                    diagnostic = "resource admission denied the mutation worker lease"
                    return seal(MutationWorkerDisposition.RESOURCE_DENIED)
                lease_id = lease.lease_id
                with self._lock:
                    self._active_leases[task.task_id] = lease
                    attempt = self._attempt_counts.get(task.task_id, 0) + 1
                    self._attempt_counts[task.task_id] = attempt
            else:
                attempt = 1

            if self._checkpoint is not None:
                try:
                    path = self._checkpoint.mark_running(
                        task.task_id,
                        lease_id=lease_id,
                        pool_id=self.pool_id,
                        attempt=attempt,
                    )
                    checkpoint_path = str(path)
                    events.append(
                        {
                            "event": "checkpoint_running",
                            "path": checkpoint_path,
                            "attempt": attempt,
                        }
                    )
                except OSError as exc:
                    reason_codes.append("checkpoint_write_failed")
                    diagnostic = _clip(f"checkpoint write failed: {exc}")
                    return seal(MutationWorkerDisposition.INFRASTRUCTURE_FAILURE)

            # Re-check cancel after admission.
            if cancel.is_cancelled() or self._pool_cancel.is_cancelled():
                cancelled = True
                reason_codes.append("cancelled_before_spawn")
                diagnostic = (
                    cancel.reason
                    or self._pool_cancel.reason
                    or "cancelled before spawn"
                )
                return seal(MutationWorkerDisposition.CANCELLED)

            timeout = task.effective_timeout(self.budget)
            if self.budget.pool_wall_seconds > 0.0:
                remaining_pool = self.budget.pool_wall_seconds - (
                    self._clock() - self._started_at
                )
                if remaining_pool <= 0.0:
                    reason_codes.append("pool_wall_exhausted")
                    diagnostic = "pool wall-time budget exhausted"
                    return seal(MutationWorkerDisposition.TIMEOUT)
                timeout = min(timeout, remaining_pool)
            deadline = self._clock() + timeout

            if task.is_command:
                (
                    disposition,
                    payload,
                    process_started,
                    pid,
                    process_group_id,
                    process_tree_fenced,
                    timed_out,
                    cancelled,
                    exit_code,
                    stdout_digest,
                    stderr_digest,
                    stdout_truncated,
                    stderr_truncated,
                    reason_codes,
                    diagnostic,
                    events,
                ) = self._run_command(
                    task,
                    cancel=cancel,
                    deadline=deadline,
                    events=events,
                )
            else:
                (
                    disposition,
                    payload,
                    timed_out,
                    cancelled,
                    reason_codes,
                    diagnostic,
                    events,
                ) = self._run_callable(
                    task,
                    cancel=cancel,
                    deadline=deadline,
                    lease_id=lease_id,
                    events=events,
                )

            if lease is not None and not self._lease_still_held(lease):
                cancelled = True
                reason_codes.append("lease_revoked")
                diagnostic = diagnostic or "resource lease revoked before publication"
                disposition = MutationWorkerDisposition.CANCELLED
                payload = None

            return seal(disposition, final_payload=payload)

        except MutationWorkerError as exc:
            code = exc.reason_code
            reason_codes.append(code)
            diagnostic = _clip(str(exc))
            if code == "timeout":
                return seal(MutationWorkerDisposition.TIMEOUT)
            if code == "cancelled":
                return seal(MutationWorkerDisposition.CANCELLED)
            if code == "network_policy_denied":
                return seal(MutationWorkerDisposition.NETWORK_POLICY_DENIED)
            return seal(MutationWorkerDisposition.INFRASTRUCTURE_FAILURE)
        except Exception as exc:  # noqa: BLE001 - never leak unsealed errors
            reason_codes.append("unhandled_worker_exception")
            reason_codes.append(type(exc).__name__)
            diagnostic = _clip(str(exc))
            return seal(MutationWorkerDisposition.INFRASTRUCTURE_FAILURE)
        finally:
            if process is not None:
                try:
                    fence_process_tree(
                        process,
                        grace_seconds=self._term_grace_seconds,
                        kill_wait_seconds=self._kill_wait_seconds,
                        require_gone=False,
                    )
                except Exception:  # noqa: BLE001
                    pass
            if lease is not None:
                try:
                    self._scheduler.release(
                        lease,
                        reason=(
                            "cancelled"
                            if cancelled or self._pool_cancel.is_cancelled()
                            else "released"
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass
                with self._lock:
                    self._active_leases.pop(task.task_id, None)
            with self._lock:
                self._active_processes.pop(task.task_id, None)

    def _run_callable(
        self,
        task: MutationWorkerTask,
        *,
        cancel: MutationWorkerCancellation,
        deadline: float,
        lease_id: str,
        events: list[dict[str, Any]],
    ) -> tuple[
        MutationWorkerDisposition,
        Any,
        bool,
        bool,
        list[str],
        str,
        list[dict[str, Any]],
    ]:
        assert task.runner is not None
        reason_codes: list[str] = []
        diagnostic = ""
        timed_out = False
        cancelled = False
        payload: Any = None
        context = MutationWorkerContext(
            task=task,
            cancellation=cancel,
            lease_id=lease_id,
            pool_id=self.pool_id,
            network_policy=task.network_policy,
            deadline_monotonic=deadline,
            clock=self._clock,
        )
        events.append({"event": "callable_start", "task_id": task.task_id})
        box: dict[str, Any] = {"error": None, "payload": None, "done": False}

        def invoke() -> None:
            try:
                context.check_cancelled()
                box["payload"] = task.runner(context)
            except Exception as exc:  # noqa: BLE001 - captured for sealing
                box["error"] = exc
            finally:
                box["done"] = True

        worker_thread = threading.Thread(
            target=invoke,
            name=f"aae-worker-callable-{task.task_id}",
            daemon=True,
        )
        worker_thread.start()
        while not box["done"]:
            if cancel.is_cancelled() or self._pool_cancel.is_cancelled():
                cancelled = True
                reason_codes.append("cancelled")
                diagnostic = (
                    cancel.reason
                    or self._pool_cancel.reason
                    or "cancelled"
                )
                break
            if self._clock() >= deadline:
                timed_out = True
                reason_codes.append("timeout")
                diagnostic = "mutation worker exceeded its wall-time limit"
                break
            remaining = deadline - self._clock()
            self._sleep(min(self._poll_interval_seconds, max(0.0, remaining)))
        if not box["done"]:
            # Cooperative only: cannot forcibly kill pure-Python threads.
            # Wait briefly for cooperative exit; otherwise seal as infra.
            worker_thread.join(timeout=self._term_grace_seconds)
            if not box["done"]:
                reason_codes.append("callable_still_running")
                events.append(
                    {
                        "event": "callable_orphaned_cooperative",
                        "task_id": task.task_id,
                    }
                )
            if cancelled:
                return (
                    MutationWorkerDisposition.CANCELLED,
                    None,
                    timed_out,
                    True,
                    reason_codes,
                    diagnostic,
                    events,
                )
            if timed_out:
                return (
                    MutationWorkerDisposition.TIMEOUT,
                    None,
                    True,
                    cancelled,
                    reason_codes,
                    diagnostic,
                    events,
                )
        worker_thread.join(timeout=self._kill_wait_seconds)
        error = box["error"]
        if error is not None:
            if isinstance(error, MutationWorkerError):
                code = error.reason_code
                reason_codes.append(code)
                diagnostic = _clip(str(error))
                if code == "timeout":
                    return (
                        MutationWorkerDisposition.TIMEOUT,
                        None,
                        True,
                        cancelled,
                        reason_codes,
                        diagnostic,
                        events,
                    )
                if code == "cancelled":
                    return (
                        MutationWorkerDisposition.CANCELLED,
                        None,
                        timed_out,
                        True,
                        reason_codes,
                        diagnostic,
                        events,
                    )
                return (
                    MutationWorkerDisposition.FAILED,
                    None,
                    timed_out,
                    cancelled,
                    reason_codes,
                    diagnostic,
                    events,
                )
            reason_codes.append("runner_exception")
            reason_codes.append(type(error).__name__)
            diagnostic = _clip(str(error))
            return (
                MutationWorkerDisposition.FAILED,
                None,
                timed_out,
                cancelled,
                reason_codes,
                diagnostic,
                events,
            )
        payload = box["payload"]
        # Convention: runner may return Mapping with explicit ok=False.
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            reason_codes.append("runner_reported_failure")
            diagnostic = _clip(str(payload.get("reason") or "runner failed"))
            events.append({"event": "callable_failed", "task_id": task.task_id})
            return (
                MutationWorkerDisposition.FAILED,
                _json_safe(payload),
                timed_out,
                cancelled,
                reason_codes,
                diagnostic,
                events,
            )
        events.append({"event": "callable_completed", "task_id": task.task_id})
        return (
            MutationWorkerDisposition.COMPLETED,
            _json_safe(payload),
            timed_out,
            cancelled,
            reason_codes,
            diagnostic,
            events,
        )

    def _run_command(
        self,
        task: MutationWorkerTask,
        *,
        cancel: MutationWorkerCancellation,
        deadline: float,
        events: list[dict[str, Any]],
    ) -> tuple[
        MutationWorkerDisposition,
        Any,
        bool,
        int | None,
        int | None,
        bool,
        bool,
        bool,
        int | None,
        str,
        str,
        bool,
        bool,
        list[str],
        str,
        list[dict[str, Any]],
    ]:
        reason_codes: list[str] = []
        diagnostic = ""
        timed_out = False
        cancelled = False
        process_started = False
        process_tree_fenced = False
        pid: int | None = None
        process_group_id: int | None = None
        exit_code: int | None = None
        stdout_digest = ""
        stderr_digest = ""
        stdout_truncated = False
        stderr_truncated = False
        payload: Any = None

        cwd = task.cwd or os.getcwd()
        try:
            cwd_path = Path(cwd).expanduser().resolve(strict=True)
        except OSError as exc:
            reason_codes.append("cwd_unavailable")
            diagnostic = _clip(f"cwd unavailable: {exc}")
            return (
                MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                None,
                False,
                None,
                None,
                False,
                False,
                False,
                None,
                "",
                "",
                False,
                False,
                reason_codes,
                diagnostic,
                events,
            )
        if not cwd_path.is_dir():
            reason_codes.append("cwd_unavailable")
            diagnostic = "cwd is not a directory"
            return (
                MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                None,
                False,
                None,
                None,
                False,
                False,
                False,
                None,
                "",
                "",
                False,
                False,
                reason_codes,
                diagnostic,
                events,
            )

        # Hermetic env: never inherit ambient secrets/proxies.
        try:
            env = build_hermetic_environment(
                dict(task.environment),
                path=str(task.environment.get("PATH") or os.environ.get("PATH", "/usr/bin:/bin")),
            )
        except Exception as exc:  # noqa: BLE001
            reason_codes.append("environment_rejected")
            diagnostic = _clip(str(exc))
            return (
                MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                None,
                False,
                None,
                None,
                False,
                False,
                False,
                None,
                "",
                "",
                False,
                False,
                reason_codes,
                diagnostic,
                events,
            )

        events.append(
            {
                "event": "command_spawn",
                "argv0": task.command[0] if task.command else "",
                "cwd": str(cwd_path),
            }
        )
        stream_files = ExitStack()
        process: Any = None
        try:
            stdout_file = stream_files.enter_context(_temporary_binary_stream())
            stderr_file = stream_files.enter_context(_temporary_binary_stream())
            spawn_kwargs: dict[str, Any] = {
                "shell": False,
                "cwd": str(cwd_path),
                "env": env,
                "stdin": (
                    subprocess.PIPE if task.stdin is not None else subprocess.DEVNULL
                ),
                "stdout": stdout_file,
                "stderr": stderr_file,
            }
            if os.name == "posix":
                spawn_kwargs["start_new_session"] = True

            try:
                process = self._popen(list(task.command), **spawn_kwargs)
            except FileNotFoundError:
                reason_codes.append("executable_missing")
                diagnostic = "worker executable was not found at spawn"
                return (
                    MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                    None,
                    False,
                    None,
                    None,
                    False,
                    False,
                    False,
                    None,
                    "",
                    "",
                    False,
                    False,
                    reason_codes,
                    diagnostic,
                    events,
                )
            except OSError as exc:
                reason_codes.append("spawn_failed")
                diagnostic = _clip(
                    f"process spawn failed: {type(exc).__name__}: {exc}"
                )
                return (
                    MutationWorkerDisposition.INFRASTRUCTURE_FAILURE,
                    None,
                    False,
                    None,
                    None,
                    False,
                    False,
                    False,
                    None,
                    "",
                    "",
                    False,
                    False,
                    reason_codes,
                    diagnostic,
                    events,
                )

            process_started = True
            pid = int(getattr(process, "pid", 0) or 0) or None
            if pid is not None and os.name == "posix":
                process_group_id = pid
            with self._lock:
                self._active_processes[task.task_id] = process

            if task.stdin is not None and process.stdin is not None:
                try:
                    process.stdin.write(task.stdin)
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

            while True:
                returncode = process.poll()
                if returncode is not None:
                    exit_code = int(returncode)
                    break
                if cancel.is_cancelled() or self._pool_cancel.is_cancelled():
                    cancelled = True
                    reason_codes.append("cancelled")
                    diagnostic = (
                        cancel.reason
                        or self._pool_cancel.reason
                        or "cancelled"
                    )
                    process_tree_fenced = bool(
                        fence_process_tree(
                            process,
                            grace_seconds=self._term_grace_seconds,
                            kill_wait_seconds=self._kill_wait_seconds,
                            require_gone=True,
                        )
                    )
                    events.append(
                        {
                            "event": "process_tree_fenced",
                            "reason": "cancelled",
                            "pid": pid,
                        }
                    )
                    try:
                        polled = process.poll()
                        exit_code = None if polled is None else int(polled)
                    except Exception:  # noqa: BLE001
                        exit_code = None
                    break
                now = self._clock()
                if now >= deadline:
                    timed_out = True
                    reason_codes.append("timeout")
                    diagnostic = (
                        "mutation worker process exceeded its wall-time limit"
                    )
                    process_tree_fenced = bool(
                        fence_process_tree(
                            process,
                            grace_seconds=self._term_grace_seconds,
                            kill_wait_seconds=self._kill_wait_seconds,
                            require_gone=True,
                        )
                    )
                    events.append(
                        {
                            "event": "process_tree_fenced",
                            "reason": "timeout",
                            "pid": pid,
                        }
                    )
                    try:
                        polled = process.poll()
                        exit_code = None if polled is None else int(polled)
                    except Exception:  # noqa: BLE001
                        exit_code = None
                    break
                remaining = deadline - now
                self._sleep(
                    min(self._poll_interval_seconds, max(0.0, remaining))
                )

            stdout_bytes, stdout_truncated = _read_stream_bytes(
                stdout_file, maximum=task.max_stdout_bytes
            )
            stderr_bytes, stderr_truncated = _read_stream_bytes(
                stderr_file, maximum=task.max_stderr_bytes
            )
            if stdout_truncated:
                reason_codes.append("stdout_truncated")
            if stderr_truncated:
                reason_codes.append("stderr_truncated")
            stdout_digest = _sha256_hex(stdout_bytes) if stdout_bytes else ""
            stderr_digest = _sha256_hex(stderr_bytes) if stderr_bytes else ""

            # Late-success fence.
            if cancel.is_cancelled() or self._pool_cancel.is_cancelled():
                cancelled = True
                if "cancelled" not in reason_codes:
                    reason_codes.append("cancelled")
                diagnostic = (
                    diagnostic
                    or cancel.reason
                    or self._pool_cancel.reason
                    or "cancelled"
                )

            if cancelled:
                disposition = MutationWorkerDisposition.CANCELLED
            elif timed_out:
                disposition = MutationWorkerDisposition.TIMEOUT
            elif exit_code == 0:
                disposition = MutationWorkerDisposition.COMPLETED
                payload = {
                    "exit_code": exit_code,
                    "stdout_digest": stdout_digest,
                    "stderr_digest": stderr_digest,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "stdout_preview": stdout_bytes[:256].decode(
                        "utf-8", "replace"
                    ),
                    "stderr_preview": stderr_bytes[:256].decode(
                        "utf-8", "replace"
                    ),
                }
            else:
                disposition = MutationWorkerDisposition.FAILED
                if "nonzero_exit" not in reason_codes:
                    reason_codes.append("nonzero_exit")
                diagnostic = diagnostic or f"process exited with code {exit_code}"
                payload = {
                    "exit_code": exit_code,
                    "stdout_digest": stdout_digest,
                    "stderr_digest": stderr_digest,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "stdout_preview": stdout_bytes[:256].decode(
                        "utf-8", "replace"
                    ),
                    "stderr_preview": stderr_bytes[:256].decode(
                        "utf-8", "replace"
                    ),
                }

            events.append(
                {
                    "event": "command_finished",
                    "disposition": disposition.value,
                    "exit_code": exit_code,
                    "process_tree_fenced": process_tree_fenced,
                }
            )
            return (
                disposition,
                payload,
                process_started,
                pid,
                process_group_id,
                process_tree_fenced,
                timed_out,
                cancelled,
                exit_code,
                stdout_digest,
                stderr_digest,
                stdout_truncated,
                stderr_truncated,
                reason_codes,
                diagnostic,
                events,
            )
        finally:
            if process is not None:
                with self._lock:
                    self._active_processes.pop(task.task_id, None)
            try:
                stream_files.close()
            except OSError:
                pass


def mutation_worker_pool_descriptor() -> dict[str, Any]:
    """Return the sealed public-symbol descriptor for this module."""

    return {
        "schema": MUTATION_WORKER_POOL_SCHEMA,
        "interface": MUTATION_WORKER_POOL_INTERFACE,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "evidence": AAE_MUTATION_WORKERS_EVIDENCE,
        "process_tree_cancellation_evidence": PROCESS_TREE_CANCELLATION_EVIDENCE,
        "network_policy": NETWORK_POLICY_DENY_ALL,
        "symbols": [
            "MutationWorkerPool",
            "MutationWorkerBudget",
            "MutationWorkerTask",
            "MutationWorkerResult",
            "MutationWorkerInfrastructureRecord",
            "MutationWorkerDisposition",
            "MutationWorkerCancellation",
            "MutationWorkerCheckpointStore",
            "MutationWorkerContext",
            "mutation_worker_pool_descriptor",
        ],
        "invariants": [
            "reuses_resource_scheduler_for_admission",
            "reuses_process_tree_cancellation",
            "enforces_concurrency_budgets_network_policy",
            "records_infrastructure_separately",
            "restartable_via_checkpoint_journal",
            "leak_free_shutdown_releases_leases_and_fences_trees",
            "deny_all_network_policy",
            "no_late_success_after_cancel_or_timeout",
        ],
        "reuses": [
            "ResourceScheduler@1",
            "ivp/process-tree-cancellation@1",
            "hermetic-sandbox network deny_all",
        ],
    }


__all__ = [
    "AAE_MUTATION_WORKERS_EVIDENCE",
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_RESOURCE_CLASS",
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    "MUTATION_WORKER_CHECKPOINT_SCHEMA",
    "MUTATION_WORKER_INFRA_SCHEMA",
    "MUTATION_WORKER_POOL_INTERFACE",
    "MUTATION_WORKER_POOL_SCHEMA",
    "MUTATION_WORKER_RESULT_SCHEMA",
    "MUTATION_WORKER_TASK_SCHEMA",
    "MutationWorkerBoundsError",
    "MutationWorkerBudget",
    "MutationWorkerCancellation",
    "MutationWorkerCheckpointStore",
    "MutationWorkerContext",
    "MutationWorkerDisposition",
    "MutationWorkerError",
    "MutationWorkerInfrastructureRecord",
    "MutationWorkerPolicyError",
    "MutationWorkerPool",
    "MutationWorkerResult",
    "MutationWorkerTask",
    "mutation_worker_pool_descriptor",
]
