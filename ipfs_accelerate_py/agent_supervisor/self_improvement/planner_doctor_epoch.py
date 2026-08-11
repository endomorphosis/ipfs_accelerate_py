"""PDR-080: bounded live Planner/Doctor self-improvement epochs.

``PlannerDoctorEpoch@1`` is the production lifecycle surface that the
supervisor daemon invokes under an *explicit* mode and policy.  It freezes
anchors and budgets once, runs a finite BASELINE→…→STOP state machine with
exactly one isolated challenger, persists every transition, and resumes
idempotently from a durable journal after crash.

This module deliberately does **not** call the test-oriented
``run_self_improvement_epoch`` helper.  Daemon integration must use
:func:`run_planner_doctor_epoch` / :class:`PlannerDoctorEpochController`
only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .self_improvement_v2 import (
    MAX_V2_SUCCESSOR_GOALS,
    MAX_V2_SUCCESSOR_TASKS,
    V2ResidualKind,
    V2ResidualSignal,
)
from .supervisor_v2_benchmark import V2CausalReceipt


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

PLANNER_DOCTOR_EPOCH_INTERFACE: Final[str] = "PlannerDoctorEpoch@1"
PLANNER_DOCTOR_EPOCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch@1"
)
PLANNER_DOCTOR_EPOCH_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch-binding@1"
)
PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch-journal@1"
)
PLANNER_DOCTOR_EPOCH_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch-manifest@1"
)
PLANNER_DOCTOR_EPOCH_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch-transition@1"
)
PLANNER_DOCTOR_EPOCH_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-epoch-result@1"
)
PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION: Final[int] = 1
PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID: Final[str] = "PDR-080"
PLANNER_DOCTOR_EPOCH_POLICY_ID: Final[str] = (
    "policy:planner-doctor-epoch@1"
)

# Hard epoch maxima (cannot be enlarged by caller policy).
MAX_EPOCHS_PER_RUN: Final[int] = 8
MAX_GOALS_PER_EPOCH: Final[int] = MAX_V2_SUCCESSOR_GOALS  # 8
MAX_TASKS_PER_EPOCH: Final[int] = MAX_V2_SUCCESSOR_TASKS  # 24
MAX_CHALLENGERS_PER_EPOCH: Final[int] = 1
MAX_MODEL_CALLS_PER_EPOCH: Final[int] = 4
MAX_REPAIRS_PER_EPOCH: Final[int] = 2
MAX_PROCESSES_PER_EPOCH: Final[int] = 32
MAX_WALL_SECONDS_PER_EPOCH: Final[int] = 3_600
MAX_CPU_SECONDS_PER_EPOCH: Final[int] = 3_600
MAX_MEMORY_BYTES_PER_EPOCH: Final[int] = 8 * 1024 * 1024 * 1024
MAX_GPU_SECONDS_PER_EPOCH: Final[int] = 600
MAX_DISK_BYTES_PER_EPOCH: Final[int] = 4 * 1024 * 1024 * 1024
MAX_STORAGE_BYTES_PER_EPOCH: Final[int] = 8 * 1024 * 1024 * 1024
MAX_TOKENS_PER_EPOCH: Final[int] = 500_000
MAX_COST_MICROS_PER_EPOCH: Final[int] = 5_000_000  # $5.00
MAX_JOURNAL_TRANSITIONS: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 512
MAX_ID_BYTES: Final[int] = 192
MAX_PROTECTED_ANCHORS: Final[int] = 64
MAX_STAGE_SPANS: Final[int] = 64

# Protected seed anchors that the epoch freezes and never mutates.
DEFAULT_PROTECTED_ANCHOR_PATHS: Final[tuple[str, ...]] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md",
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
    "config/agent_supervisor_planner_doctor_authority_policy.json",
    "config/agent_supervisor_planner_doctor_benchmark.json",
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json",
)

_CONTENT_ID = __import__("re").compile(r"^sha256:[0-9a-f]{64}$")
_CODE = __import__("re").compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/@-]{0,191}$")


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class PlannerDoctorEpochError(ValueError):
    """Malformed input, authority violation, or non-replayable epoch state."""


class PlannerDoctorEpochMode(str, Enum):
    """Explicit lifecycle modes.  ``OFF`` refuses every daemon invocation."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    CANARY = "canary"
    # Automatic remains named so policy can declare it, but the controller
    # never elevates into it without an independent operator grant.
    AUTOMATIC = "automatic"


class PlannerDoctorEpochStage(str, Enum):
    """Finite unattended-improvement state machine stages."""

    BASELINE = "baseline"
    PROPOSE = "propose"
    SHADOW = "shadow"
    EVALUATE = "evaluate"
    REJECT = "reject"
    RETAIN = "retain"
    CANARY = "canary"
    RECHECK = "recheck"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REFILL = "refill"
    STOP = "stop"


class PlannerDoctorEpochStopReason(str, Enum):
    """Closed stop vocabulary from the PDR-080 acceptance contract."""

    SAFETY_REGRESSION = "safety_regression"
    QUALITY_REGRESSION = "quality_regression"
    UNCHANGED_RESIDUAL = "unchanged_residual"
    NO_ADMITTED_IMPROVEMENT = "no_admitted_improvement"
    ORACLE_LOSS = "oracle_loss"
    TELEMETRY_LOSS = "telemetry_loss"
    ROLLBACK_FAILURE = "rollback_failure"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    MODE_DISABLED = "mode_disabled"
    POLICY_REQUIRED = "policy_required"
    COMPLETED = "completed"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    MUTATION_FORBIDDEN = "mutation_forbidden"


TERMINAL_STAGES: Final[frozenset[PlannerDoctorEpochStage]] = frozenset(
    {
        PlannerDoctorEpochStage.STOP,
        PlannerDoctorEpochStage.REJECT,
        PlannerDoctorEpochStage.PROMOTE,
    }
)

# Stages that may be resumed mid-flight after crash.
RESUMABLE_STAGES: Final[frozenset[PlannerDoctorEpochStage]] = frozenset(
    {
        PlannerDoctorEpochStage.BASELINE,
        PlannerDoctorEpochStage.PROPOSE,
        PlannerDoctorEpochStage.SHADOW,
        PlannerDoctorEpochStage.EVALUATE,
        PlannerDoctorEpochStage.RETAIN,
        PlannerDoctorEpochStage.CANARY,
        PlannerDoctorEpochStage.RECHECK,
        PlannerDoctorEpochStage.ROLLBACK,
        PlannerDoctorEpochStage.REFILL,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _digest(value: Any) -> str:
    payload = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "sha256:" + ("0" * 64)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise PlannerDoctorEpochError(f"{name} must be text")
    text = value.strip()
    if not text:
        raise PlannerDoctorEpochError(f"{name} must be non-empty")
    encoded = text.encode("utf-8")
    if len(encoded) > maximum:
        raise PlannerDoctorEpochError(f"{name} exceeds {maximum} bytes")
    return text


def _optional_text(
    value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES
) -> str | None:
    if value is None or value == "":
        return None
    return _text(value, name, maximum=maximum)


def _code(value: Any, name: str) -> str:
    text = _text(value, name, maximum=MAX_ID_BYTES)
    if not _CODE.match(text):
        raise PlannerDoctorEpochError(f"{name} is not a closed code token")
    return text


def _content_id(value: Any, name: str) -> str:
    text = _text(value, name, maximum=MAX_ID_BYTES)
    if not _CONTENT_ID.match(text):
        raise PlannerDoctorEpochError(f"{name} must be a sha256 content id")
    return text


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10**15,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerDoctorEpochError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise PlannerDoctorEpochError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerDoctorEpochError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise PlannerDoctorEpochError(
                f"{name} must be one of "
                f"{sorted(item.value for item in enum_type)}"
            ) from exc
    raise PlannerDoctorEpochError(f"{name} must be a {enum_type.__name__}")


def _timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise PlannerDoctorEpochError(
                "timestamp must be ISO-8601 text"
            ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PlannerDoctorEpochError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_plain(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        flags = getattr(os, "O_DIRECTORY", 0)
        try:
            parent_fd = os.open(str(path.parent), os.O_RDONLY | flags)
        except OSError:
            return
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PlannerDoctorEpochError(
            f"{name} contains unknown keys: {', '.join(unknown)}"
        )


# ---------------------------------------------------------------------------
# Budgets / anchors / policy / binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDoctorEpochBudgets:
    """Finite resource ceilings frozen at epoch start.

    Callers may *tighten* ceilings but never enlarge the hard maxima.
    """

    max_epochs: int = MAX_EPOCHS_PER_RUN
    max_wall_seconds: int = MAX_WALL_SECONDS_PER_EPOCH
    max_cpu_seconds: int = MAX_CPU_SECONDS_PER_EPOCH
    max_memory_bytes: int = MAX_MEMORY_BYTES_PER_EPOCH
    max_gpu_seconds: int = MAX_GPU_SECONDS_PER_EPOCH
    max_disk_bytes: int = MAX_DISK_BYTES_PER_EPOCH
    max_storage_bytes: int = MAX_STORAGE_BYTES_PER_EPOCH
    max_tokens: int = MAX_TOKENS_PER_EPOCH
    max_cost_micros: int = MAX_COST_MICROS_PER_EPOCH
    max_processes: int = MAX_PROCESSES_PER_EPOCH
    max_model_calls: int = MAX_MODEL_CALLS_PER_EPOCH
    max_repairs: int = MAX_REPAIRS_PER_EPOCH
    max_goals: int = MAX_GOALS_PER_EPOCH
    max_tasks: int = MAX_TASKS_PER_EPOCH
    max_challengers: int = MAX_CHALLENGERS_PER_EPOCH

    def __post_init__(self) -> None:
        caps = {
            "max_epochs": MAX_EPOCHS_PER_RUN,
            "max_wall_seconds": MAX_WALL_SECONDS_PER_EPOCH,
            "max_cpu_seconds": MAX_CPU_SECONDS_PER_EPOCH,
            "max_memory_bytes": MAX_MEMORY_BYTES_PER_EPOCH,
            "max_gpu_seconds": MAX_GPU_SECONDS_PER_EPOCH,
            "max_disk_bytes": MAX_DISK_BYTES_PER_EPOCH,
            "max_storage_bytes": MAX_STORAGE_BYTES_PER_EPOCH,
            "max_tokens": MAX_TOKENS_PER_EPOCH,
            "max_cost_micros": MAX_COST_MICROS_PER_EPOCH,
            "max_processes": MAX_PROCESSES_PER_EPOCH,
            "max_model_calls": MAX_MODEL_CALLS_PER_EPOCH,
            "max_repairs": MAX_REPAIRS_PER_EPOCH,
            "max_goals": MAX_GOALS_PER_EPOCH,
            "max_tasks": MAX_TASKS_PER_EPOCH,
            "max_challengers": MAX_CHALLENGERS_PER_EPOCH,
        }
        for name, hard_max in caps.items():
            value = _integer(getattr(self, name), name, minimum=1, maximum=hard_max)
            object.__setattr__(self, name, value)
        if self.max_challengers != 1:
            raise PlannerDoctorEpochError(
                "exactly one isolated challenger is required per epoch"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_epochs": self.max_epochs,
            "max_wall_seconds": self.max_wall_seconds,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_gpu_seconds": self.max_gpu_seconds,
            "max_disk_bytes": self.max_disk_bytes,
            "max_storage_bytes": self.max_storage_bytes,
            "max_tokens": self.max_tokens,
            "max_cost_micros": self.max_cost_micros,
            "max_processes": self.max_processes,
            "max_model_calls": self.max_model_calls,
            "max_repairs": self.max_repairs,
            "max_goals": self.max_goals,
            "max_tasks": self.max_tasks,
            "max_challengers": self.max_challengers,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PlannerDoctorEpochBudgets":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("budgets must be a mapping")
        allowed = set(cls().to_dict())
        _strict_keys(payload, allowed, name="budgets")
        return cls(**{key: payload[key] for key in payload})

    @property
    def budgets_id(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class PlannerDoctorEpochUsage:
    """Consumed resources observed during one epoch attempt."""

    wall_seconds: int = 0
    cpu_seconds: int = 0
    memory_bytes: int = 0
    gpu_seconds: int = 0
    disk_bytes: int = 0
    storage_bytes: int = 0
    tokens: int = 0
    cost_micros: int = 0
    processes: int = 0
    model_calls: int = 0
    repairs: int = 0
    epochs: int = 0
    goals: int = 0
    tasks: int = 0
    challengers: int = 0

    def __post_init__(self) -> None:
        for name in self.to_dict():
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=0),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "memory_bytes": self.memory_bytes,
            "gpu_seconds": self.gpu_seconds,
            "disk_bytes": self.disk_bytes,
            "storage_bytes": self.storage_bytes,
            "tokens": self.tokens,
            "cost_micros": self.cost_micros,
            "processes": self.processes,
            "model_calls": self.model_calls,
            "repairs": self.repairs,
            "epochs": self.epochs,
            "goals": self.goals,
            "tasks": self.tasks,
            "challengers": self.challengers,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PlannerDoctorEpochUsage":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("usage must be a mapping")
        allowed = set(cls().to_dict())
        _strict_keys(payload, allowed, name="usage")
        return cls(**{key: int(payload[key]) for key in payload})

    def merge(self, other: "PlannerDoctorEpochUsage") -> "PlannerDoctorEpochUsage":
        return PlannerDoctorEpochUsage(
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            memory_bytes=max(self.memory_bytes, other.memory_bytes),
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            disk_bytes=self.disk_bytes + other.disk_bytes,
            storage_bytes=max(self.storage_bytes, other.storage_bytes),
            tokens=self.tokens + other.tokens,
            cost_micros=self.cost_micros + other.cost_micros,
            processes=max(self.processes, other.processes),
            model_calls=self.model_calls + other.model_calls,
            repairs=self.repairs + other.repairs,
            epochs=self.epochs + other.epochs,
            goals=self.goals + other.goals,
            tasks=self.tasks + other.tasks,
            challengers=max(self.challengers, other.challengers),
        )

    def exhausted_against(
        self, budgets: PlannerDoctorEpochBudgets
    ) -> tuple[str, ...]:
        checks = (
            ("wall_seconds", self.wall_seconds, budgets.max_wall_seconds),
            ("cpu_seconds", self.cpu_seconds, budgets.max_cpu_seconds),
            ("memory_bytes", self.memory_bytes, budgets.max_memory_bytes),
            ("gpu_seconds", self.gpu_seconds, budgets.max_gpu_seconds),
            ("disk_bytes", self.disk_bytes, budgets.max_disk_bytes),
            ("storage_bytes", self.storage_bytes, budgets.max_storage_bytes),
            ("tokens", self.tokens, budgets.max_tokens),
            ("cost_micros", self.cost_micros, budgets.max_cost_micros),
            ("processes", self.processes, budgets.max_processes),
            ("model_calls", self.model_calls, budgets.max_model_calls),
            ("repairs", self.repairs, budgets.max_repairs),
            ("epochs", self.epochs, budgets.max_epochs),
            ("goals", self.goals, budgets.max_goals),
            ("tasks", self.tasks, budgets.max_tasks),
            ("challengers", self.challengers, budgets.max_challengers),
        )
        return tuple(
            name for name, used, limit in checks if used > limit
        )


@dataclass(frozen=True)
class PlannerDoctorEpochAnchors:
    """Frozen protected-path digests bound before any challenger work."""

    repository_id: str
    tree_id: str
    path_digests: Mapping[str, str]
    authority_policy_revision: str
    benchmark_policy_revision: str
    frozen_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_id", _code(self.repository_id, "repository_id"))
        object.__setattr__(self, "tree_id", _code(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "authority_policy_revision",
            _code(self.authority_policy_revision, "authority_policy_revision"),
        )
        object.__setattr__(
            self,
            "benchmark_policy_revision",
            _code(self.benchmark_policy_revision, "benchmark_policy_revision"),
        )
        object.__setattr__(self, "frozen_at", _timestamp(self.frozen_at))
        if not isinstance(self.path_digests, Mapping):
            raise PlannerDoctorEpochError("path_digests must be a mapping")
        if len(self.path_digests) > MAX_PROTECTED_ANCHORS:
            raise PlannerDoctorEpochError("too many protected anchors")
        normalized: dict[str, str] = {}
        for path, digest in sorted(self.path_digests.items(), key=lambda item: item[0]):
            key = _text(path, "anchor_path", maximum=512)
            if key.startswith("/") or ".." in Path(key).parts:
                raise PlannerDoctorEpochError(
                    f"anchor path must be repository-relative: {key}"
                )
            normalized[key] = _content_id(digest, f"digest[{key}]")
        object.__setattr__(self, "path_digests", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "path_digests": dict(self.path_digests),
            "authority_policy_revision": self.authority_policy_revision,
            "benchmark_policy_revision": self.benchmark_policy_revision,
            "frozen_at": self.frozen_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorEpochAnchors":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("anchors must be a mapping")
        _strict_keys(
            payload,
            {
                "repository_id",
                "tree_id",
                "path_digests",
                "authority_policy_revision",
                "benchmark_policy_revision",
                "frozen_at",
            },
            name="anchors",
        )
        return cls(
            repository_id=payload["repository_id"],
            tree_id=payload["tree_id"],
            path_digests=payload["path_digests"],
            authority_policy_revision=payload["authority_policy_revision"],
            benchmark_policy_revision=payload["benchmark_policy_revision"],
            frozen_at=payload["frozen_at"],
        )

    @property
    def anchors_id(self) -> str:
        # frozen_at is observational metadata and must not change the epoch
        # identity; otherwise crash-resume and lifecycle re-entry would mint
        # a new epoch for the same protected digests.
        return _digest(
            {
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "path_digests": dict(self.path_digests),
                "authority_policy_revision": self.authority_policy_revision,
                "benchmark_policy_revision": self.benchmark_policy_revision,
            }
        )

    def verify_unmutated(self, repo_root: Path) -> tuple[str, ...]:
        """Return paths whose on-disk digest no longer matches the freeze."""

        drifted: list[str] = []
        for relative, expected in self.path_digests.items():
            actual = _file_digest(repo_root / relative)
            if actual != expected and expected != "sha256:" + ("0" * 64):
                # Missing at freeze time (zero digest) is allowed to remain
                # missing; present digests must be bit-stable.
                if expected.endswith("0" * 64) and not (repo_root / relative).is_file():
                    continue
                drifted.append(relative)
        return tuple(drifted)


def freeze_planner_doctor_anchors(
    *,
    repo_root: Path,
    repository_id: str,
    tree_id: str,
    authority_policy_revision: str,
    benchmark_policy_revision: str,
    protected_paths: Sequence[str] = DEFAULT_PROTECTED_ANCHOR_PATHS,
    frozen_at: datetime | str | None = None,
) -> PlannerDoctorEpochAnchors:
    """Hash protected seed artifacts and return an immutable anchor freeze."""

    root = Path(repo_root)
    digests: dict[str, str] = {}
    for relative in protected_paths:
        key = _text(relative, "protected_path", maximum=512)
        digests[key] = _file_digest(root / key)
    return PlannerDoctorEpochAnchors(
        repository_id=repository_id,
        tree_id=tree_id,
        path_digests=digests,
        authority_policy_revision=authority_policy_revision,
        benchmark_policy_revision=benchmark_policy_revision,
        frozen_at=_timestamp(frozen_at),
    )


@dataclass(frozen=True)
class PlannerDoctorEpochPolicy:
    """Explicit operator policy required before the lifecycle may run."""

    policy_id: str = PLANNER_DOCTOR_EPOCH_POLICY_ID
    mode: PlannerDoctorEpochMode = PlannerDoctorEpochMode.OFF
    budgets: PlannerDoctorEpochBudgets = field(
        default_factory=PlannerDoctorEpochBudgets
    )
    allow_mutation: bool = False
    require_live_oracle: bool = True
    require_live_telemetry: bool = True
    require_isolated_challenger: bool = True
    stop_on_unchanged_residual: bool = True
    stop_on_no_admitted_improvement: bool = True
    stop_on_safety_regression: bool = True
    stop_on_quality_regression: bool = True
    automatic_promotion_enabled: bool = False
    policy_revision: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _code(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "mode", _enum(self.mode, PlannerDoctorEpochMode, "mode")
        )
        budgets = self.budgets
        if isinstance(budgets, Mapping):
            budgets = PlannerDoctorEpochBudgets.from_dict(budgets)
        if not isinstance(budgets, PlannerDoctorEpochBudgets):
            raise PlannerDoctorEpochError("budgets must be PlannerDoctorEpochBudgets")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(
            self, "allow_mutation", _boolean(self.allow_mutation, "allow_mutation")
        )
        object.__setattr__(
            self,
            "require_live_oracle",
            _boolean(self.require_live_oracle, "require_live_oracle"),
        )
        object.__setattr__(
            self,
            "require_live_telemetry",
            _boolean(self.require_live_telemetry, "require_live_telemetry"),
        )
        object.__setattr__(
            self,
            "require_isolated_challenger",
            _boolean(
                self.require_isolated_challenger, "require_isolated_challenger"
            ),
        )
        for name in (
            "stop_on_unchanged_residual",
            "stop_on_no_admitted_improvement",
            "stop_on_safety_regression",
            "stop_on_quality_regression",
            "automatic_promotion_enabled",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        object.__setattr__(
            self, "policy_revision", _code(self.policy_revision, "policy_revision")
        )
        if self.mode is PlannerDoctorEpochMode.AUTOMATIC and not self.automatic_promotion_enabled:
            raise PlannerDoctorEpochError(
                "automatic mode requires automatic_promotion_enabled"
            )
        if self.mode is PlannerDoctorEpochMode.AUTOMATIC:
            # Seed configuration keeps automatic off; policy may name it only
            # when an independent grant sets the flag, which still refuses
            # silent elevation inside the controller.
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "mode": self.mode.value,
            "budgets": self.budgets.to_dict(),
            "allow_mutation": self.allow_mutation,
            "require_live_oracle": self.require_live_oracle,
            "require_live_telemetry": self.require_live_telemetry,
            "require_isolated_challenger": self.require_isolated_challenger,
            "stop_on_unchanged_residual": self.stop_on_unchanged_residual,
            "stop_on_no_admitted_improvement": self.stop_on_no_admitted_improvement,
            "stop_on_safety_regression": self.stop_on_safety_regression,
            "stop_on_quality_regression": self.stop_on_quality_regression,
            "automatic_promotion_enabled": self.automatic_promotion_enabled,
            "policy_revision": self.policy_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorEpochPolicy":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("policy must be a mapping")
        allowed = {
            "policy_id",
            "mode",
            "budgets",
            "allow_mutation",
            "require_live_oracle",
            "require_live_telemetry",
            "require_isolated_challenger",
            "stop_on_unchanged_residual",
            "stop_on_no_admitted_improvement",
            "stop_on_safety_regression",
            "stop_on_quality_regression",
            "automatic_promotion_enabled",
            "policy_revision",
        }
        _strict_keys(payload, allowed, name="policy")
        kwargs = dict(payload)
        if "budgets" in kwargs:
            kwargs["budgets"] = PlannerDoctorEpochBudgets.from_dict(kwargs["budgets"])
        return cls(**kwargs)

    @property
    def policy_digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def is_enabled(self) -> bool:
        return self.mode is not PlannerDoctorEpochMode.OFF


@dataclass(frozen=True)
class PlannerDoctorEpochBinding:
    """Complete immutable input identity for one live epoch."""

    repository_id: str
    tree_id: str
    policy: PlannerDoctorEpochPolicy
    anchors: PlannerDoctorEpochAnchors
    objective_revision: str
    board_revision: str
    capability_revision: str
    operator_revision: str
    epoch_index: int = 0
    observed_at: str = field(default_factory=lambda: _timestamp())

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_id", _code(self.repository_id, "repository_id"))
        object.__setattr__(self, "tree_id", _code(self.tree_id, "tree_id"))
        policy = self.policy
        if isinstance(policy, Mapping):
            policy = PlannerDoctorEpochPolicy.from_dict(policy)
        if not isinstance(policy, PlannerDoctorEpochPolicy):
            raise PlannerDoctorEpochError("policy must be PlannerDoctorEpochPolicy")
        object.__setattr__(self, "policy", policy)
        anchors = self.anchors
        if isinstance(anchors, Mapping):
            anchors = PlannerDoctorEpochAnchors.from_dict(anchors)
        if not isinstance(anchors, PlannerDoctorEpochAnchors):
            raise PlannerDoctorEpochError("anchors must be PlannerDoctorEpochAnchors")
        object.__setattr__(self, "anchors", anchors)
        if anchors.repository_id != self.repository_id or anchors.tree_id != self.tree_id:
            raise PlannerDoctorEpochError(
                "anchors must match binding repository/tree identity"
            )
        for name in (
            "objective_revision",
            "board_revision",
            "capability_revision",
            "operator_revision",
        ):
            object.__setattr__(self, name, _code(getattr(self, name), name))
        object.__setattr__(
            self,
            "epoch_index",
            _integer(self.epoch_index, "epoch_index", minimum=0, maximum=MAX_EPOCHS_PER_RUN),
        )
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_EPOCH_BINDING_SCHEMA,
            "contract_version": PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION,
            "interface": PLANNER_DOCTOR_EPOCH_INTERFACE,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy": self.policy.to_dict(),
            "anchors": self.anchors.to_dict(),
            "objective_revision": self.objective_revision,
            "board_revision": self.board_revision,
            "capability_revision": self.capability_revision,
            "operator_revision": self.operator_revision,
            "epoch_index": self.epoch_index,
            "observed_at": self.observed_at,
            "budgets_id": self.policy.budgets.budgets_id,
            "anchors_id": self.anchors.anchors_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorEpochBinding":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("binding must be a mapping")
        _strict_keys(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "repository_id",
                "tree_id",
                "policy",
                "anchors",
                "objective_revision",
                "board_revision",
                "capability_revision",
                "operator_revision",
                "epoch_index",
                "observed_at",
                "budgets_id",
                "anchors_id",
            },
            name="binding",
        )
        schema = payload.get("schema")
        if schema not in (None, PLANNER_DOCTOR_EPOCH_BINDING_SCHEMA):
            raise PlannerDoctorEpochError("unsupported epoch binding schema")
        if (
            payload.get("interface") not in (None, PLANNER_DOCTOR_EPOCH_INTERFACE)
        ):
            raise PlannerDoctorEpochError("unsupported epoch binding interface")
        return cls(
            repository_id=payload["repository_id"],
            tree_id=payload["tree_id"],
            policy=PlannerDoctorEpochPolicy.from_dict(payload["policy"]),
            anchors=PlannerDoctorEpochAnchors.from_dict(payload["anchors"]),
            objective_revision=payload["objective_revision"],
            board_revision=payload["board_revision"],
            capability_revision=payload["capability_revision"],
            operator_revision=payload["operator_revision"],
            epoch_index=int(payload.get("epoch_index", 0)),
            observed_at=payload.get("observed_at") or _timestamp(),
        )

    @property
    def epoch_id(self) -> str:
        return _digest(
            {
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "policy_digest": self.policy.policy_digest,
                "anchors_id": self.anchors.anchors_id,
                "objective_revision": self.objective_revision,
                "board_revision": self.board_revision,
                "capability_revision": self.capability_revision,
                "operator_revision": self.operator_revision,
                "epoch_index": self.epoch_index,
            }
        )


# ---------------------------------------------------------------------------
# Transitions / journal / manifest / evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDoctorEpochTransition:
    """One durable stage transition with optional stop reason."""

    stage: PlannerDoctorEpochStage
    previous_stage: PlannerDoctorEpochStage | None
    recorded_at: str
    usage: PlannerDoctorEpochUsage = field(default_factory=PlannerDoctorEpochUsage)
    stop_reason: PlannerDoctorEpochStopReason | None = None
    detail: str = ""
    baseline_root: str | None = None
    challenger_root: str | None = None
    residual_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", _enum(self.stage, PlannerDoctorEpochStage, "stage")
        )
        if self.previous_stage is not None:
            object.__setattr__(
                self,
                "previous_stage",
                _enum(
                    self.previous_stage,
                    PlannerDoctorEpochStage,
                    "previous_stage",
                ),
            )
        object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at))
        usage = self.usage
        if isinstance(usage, Mapping):
            usage = PlannerDoctorEpochUsage.from_dict(usage)
        if not isinstance(usage, PlannerDoctorEpochUsage):
            raise PlannerDoctorEpochError("usage must be PlannerDoctorEpochUsage")
        object.__setattr__(self, "usage", usage)
        if self.stop_reason is not None:
            object.__setattr__(
                self,
                "stop_reason",
                _enum(
                    self.stop_reason,
                    PlannerDoctorEpochStopReason,
                    "stop_reason",
                ),
            )
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=MAX_TEXT_BYTES)
            )
        if self.baseline_root is not None:
            object.__setattr__(
                self, "baseline_root", _code(self.baseline_root, "baseline_root")
            )
        if self.challenger_root is not None:
            object.__setattr__(
                self, "challenger_root", _code(self.challenger_root, "challenger_root")
            )
        object.__setattr__(
            self,
            "residual_ids",
            tuple(_code(item, "residual_id") for item in self.residual_ids),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(_code(item, "evidence_id") for item in self.evidence_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_EPOCH_TRANSITION_SCHEMA,
            "stage": self.stage.value,
            "previous_stage": (
                self.previous_stage.value if self.previous_stage is not None else None
            ),
            "recorded_at": self.recorded_at,
            "usage": self.usage.to_dict(),
            "stop_reason": (
                self.stop_reason.value if self.stop_reason is not None else None
            ),
            "detail": self.detail,
            "baseline_root": self.baseline_root,
            "challenger_root": self.challenger_root,
            "residual_ids": list(self.residual_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorEpochTransition":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("transition must be a mapping")
        return cls(
            stage=payload["stage"],
            previous_stage=payload.get("previous_stage"),
            recorded_at=payload["recorded_at"],
            usage=PlannerDoctorEpochUsage.from_dict(payload.get("usage")),
            stop_reason=payload.get("stop_reason"),
            detail=payload.get("detail") or "",
            baseline_root=payload.get("baseline_root"),
            challenger_root=payload.get("challenger_root"),
            residual_ids=tuple(payload.get("residual_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
        )

    @property
    def transition_id(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class PlannerDoctorEpochEvaluation:
    """Paired baseline/challenger evaluation inputs for stop decisions."""

    safety_regression: bool = False
    quality_regression: bool = False
    unchanged_residual: bool = False
    admitted_improvement: bool = False
    oracle_available: bool = True
    telemetry_available: bool = True
    rollback_succeeded: bool | None = None
    residual_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    baseline_receipt_id: str | None = None
    challenger_receipt_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        for name in (
            "safety_regression",
            "quality_regression",
            "unchanged_residual",
            "admitted_improvement",
            "oracle_available",
            "telemetry_available",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.rollback_succeeded is not None:
            object.__setattr__(
                self,
                "rollback_succeeded",
                _boolean(self.rollback_succeeded, "rollback_succeeded"),
            )
        object.__setattr__(
            self,
            "residual_ids",
            tuple(_code(item, "residual_id") for item in self.residual_ids),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(_code(item, "evidence_id") for item in self.evidence_ids),
        )
        if self.baseline_receipt_id is not None:
            object.__setattr__(
                self,
                "baseline_receipt_id",
                _code(self.baseline_receipt_id, "baseline_receipt_id"),
            )
        if self.challenger_receipt_id is not None:
            object.__setattr__(
                self,
                "challenger_receipt_id",
                _code(self.challenger_receipt_id, "challenger_receipt_id"),
            )
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=MAX_TEXT_BYTES)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety_regression": self.safety_regression,
            "quality_regression": self.quality_regression,
            "unchanged_residual": self.unchanged_residual,
            "admitted_improvement": self.admitted_improvement,
            "oracle_available": self.oracle_available,
            "telemetry_available": self.telemetry_available,
            "rollback_succeeded": self.rollback_succeeded,
            "residual_ids": list(self.residual_ids),
            "evidence_ids": list(self.evidence_ids),
            "baseline_receipt_id": self.baseline_receipt_id,
            "challenger_receipt_id": self.challenger_receipt_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerDoctorEpochEvaluation":
        if not isinstance(payload, Mapping):
            raise PlannerDoctorEpochError("evaluation must be a mapping")
        return cls(
            safety_regression=bool(payload.get("safety_regression", False)),
            quality_regression=bool(payload.get("quality_regression", False)),
            unchanged_residual=bool(payload.get("unchanged_residual", False)),
            admitted_improvement=bool(payload.get("admitted_improvement", False)),
            oracle_available=bool(payload.get("oracle_available", True)),
            telemetry_available=bool(payload.get("telemetry_available", True)),
            rollback_succeeded=payload.get("rollback_succeeded"),
            residual_ids=tuple(payload.get("residual_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            baseline_receipt_id=payload.get("baseline_receipt_id"),
            challenger_receipt_id=payload.get("challenger_receipt_id"),
            detail=payload.get("detail") or "",
        )


@dataclass(frozen=True)
class PlannerDoctorEpochManifest:
    """Public epoch manifest: binding, budgets, roots, stage spans."""

    binding: PlannerDoctorEpochBinding
    baseline_root: str
    challenger_root: str | None
    challenger_worktree: str | None
    transitions: tuple[PlannerDoctorEpochTransition, ...]
    usage: PlannerDoctorEpochUsage
    stop_reason: PlannerDoctorEpochStopReason | None
    evaluation: PlannerDoctorEpochEvaluation | None = None
    stage_spans: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.binding, PlannerDoctorEpochBinding):
            raise PlannerDoctorEpochError("binding must be PlannerDoctorEpochBinding")
        object.__setattr__(self, "baseline_root", _code(self.baseline_root, "baseline_root"))
        if self.challenger_root is not None:
            object.__setattr__(
                self, "challenger_root", _code(self.challenger_root, "challenger_root")
            )
        if self.challenger_worktree is not None:
            object.__setattr__(
                self,
                "challenger_worktree",
                _text(self.challenger_worktree, "challenger_worktree", maximum=1024),
            )
        transitions = tuple(
            item
            if isinstance(item, PlannerDoctorEpochTransition)
            else PlannerDoctorEpochTransition.from_dict(item)
            for item in self.transitions
        )
        if len(transitions) > MAX_JOURNAL_TRANSITIONS:
            raise PlannerDoctorEpochError("too many epoch transitions")
        object.__setattr__(self, "transitions", transitions)
        usage = self.usage
        if isinstance(usage, Mapping):
            usage = PlannerDoctorEpochUsage.from_dict(usage)
        if not isinstance(usage, PlannerDoctorEpochUsage):
            raise PlannerDoctorEpochError("usage must be PlannerDoctorEpochUsage")
        object.__setattr__(self, "usage", usage)
        if self.stop_reason is not None:
            object.__setattr__(
                self,
                "stop_reason",
                _enum(self.stop_reason, PlannerDoctorEpochStopReason, "stop_reason"),
            )
        if self.evaluation is not None and isinstance(self.evaluation, Mapping):
            object.__setattr__(
                self,
                "evaluation",
                PlannerDoctorEpochEvaluation.from_dict(self.evaluation),
            )
        spans = tuple(MappingProxyType(dict(item)) for item in self.stage_spans)
        if len(spans) > MAX_STAGE_SPANS:
            raise PlannerDoctorEpochError("too many stage spans")
        object.__setattr__(self, "stage_spans", spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_EPOCH_MANIFEST_SCHEMA,
            "contract_version": PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION,
            "interface": PLANNER_DOCTOR_EPOCH_INTERFACE,
            "producer_task_id": PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID,
            "epoch_id": self.binding.epoch_id,
            "binding": self.binding.to_dict(),
            "baseline_root": self.baseline_root,
            "challenger_root": self.challenger_root,
            "challenger_worktree": self.challenger_worktree,
            "transitions": [item.to_dict() for item in self.transitions],
            "usage": self.usage.to_dict(),
            "stop_reason": (
                self.stop_reason.value if self.stop_reason is not None else None
            ),
            "evaluation": (
                self.evaluation.to_dict() if self.evaluation is not None else None
            ),
            "stage_spans": [dict(item) for item in self.stage_spans],
            "budgets": self.binding.policy.budgets.to_dict(),
        }

    @property
    def manifest_id(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class PlannerDoctorEpochResult:
    """Terminal result of one bounded live epoch attempt."""

    binding: PlannerDoctorEpochBinding
    manifest: PlannerDoctorEpochManifest
    current_stage: PlannerDoctorEpochStage
    stop_reason: PlannerDoctorEpochStopReason
    usage: PlannerDoctorEpochUsage
    journal_path: str
    resumed: bool = False
    idempotent_replay: bool = False
    residuals: tuple[V2ResidualSignal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_stage",
            _enum(self.current_stage, PlannerDoctorEpochStage, "current_stage"),
        )
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, PlannerDoctorEpochStopReason, "stop_reason"),
        )
        object.__setattr__(
            self, "journal_path", _text(self.journal_path, "journal_path", maximum=1024)
        )
        object.__setattr__(self, "resumed", _boolean(self.resumed, "resumed"))
        object.__setattr__(
            self,
            "idempotent_replay",
            _boolean(self.idempotent_replay, "idempotent_replay"),
        )
        residuals = tuple(
            item
            if isinstance(item, V2ResidualSignal)
            else V2ResidualSignal.from_dict(item)
            for item in self.residuals
        )
        object.__setattr__(self, "residuals", residuals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLANNER_DOCTOR_EPOCH_RESULT_SCHEMA,
            "contract_version": PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION,
            "interface": PLANNER_DOCTOR_EPOCH_INTERFACE,
            "producer_task_id": PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID,
            "epoch_id": self.binding.epoch_id,
            "binding": self.binding.to_dict(),
            "manifest": self.manifest.to_dict(),
            "current_stage": self.current_stage.value,
            "stop_reason": self.stop_reason.value,
            "usage": self.usage.to_dict(),
            "journal_path": self.journal_path,
            "resumed": self.resumed,
            "idempotent_replay": self.idempotent_replay,
            "residuals": [item.to_dict() for item in self.residuals],
            "result_id": self.result_id,
        }

    @property
    def result_id(self) -> str:
        return _digest(
            {
                "epoch_id": self.binding.epoch_id,
                "stop_reason": self.stop_reason.value,
                "current_stage": self.current_stage.value,
                "manifest_id": self.manifest.manifest_id,
                "usage": self.usage.to_dict(),
            }
        )

    @property
    def terminal(self) -> bool:
        return self.current_stage in TERMINAL_STAGES or self.stop_reason is not None


# ---------------------------------------------------------------------------
# Journal persistence
# ---------------------------------------------------------------------------


def _empty_journal() -> dict[str, Any]:
    return {
        "schema": PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA,
        "contract_version": PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION,
        "epochs": {},
    }


def load_planner_doctor_epoch_journal(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_journal()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlannerDoctorEpochError(
            "planner-doctor epoch journal is malformed"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA
        or not isinstance(payload.get("epochs"), Mapping)
    ):
        raise PlannerDoctorEpochError("unsupported planner-doctor epoch journal")
    return {
        "schema": PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA,
        "contract_version": PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION,
        "epochs": {
            str(key): dict(value)
            for key, value in payload["epochs"].items()
            if isinstance(value, Mapping)
        },
    }


def _persist_journal(path: Path, journal: Mapping[str, Any]) -> None:
    _atomic_write_json(path, journal)


def _journal_entry_from_result(result: PlannerDoctorEpochResult) -> dict[str, Any]:
    return {
        "epoch_id": result.binding.epoch_id,
        "binding": result.binding.to_dict(),
        "manifest": result.manifest.to_dict(),
        "current_stage": result.current_stage.value,
        "stop_reason": result.stop_reason.value,
        "usage": result.usage.to_dict(),
        "result_id": result.result_id,
        "resumed": result.resumed,
        "idempotent_replay": result.idempotent_replay,
        "residuals": [item.to_dict() for item in result.residuals],
        "terminal": True,
    }


def _result_from_journal_entry(
    entry: Mapping[str, Any],
    *,
    journal_path: Path,
    resumed: bool = False,
    idempotent_replay: bool = False,
) -> PlannerDoctorEpochResult:
    binding = PlannerDoctorEpochBinding.from_dict(entry["binding"])
    transitions = tuple(
        PlannerDoctorEpochTransition.from_dict(item)
        for item in entry["manifest"]["transitions"]
    )
    evaluation = entry["manifest"].get("evaluation")
    manifest = PlannerDoctorEpochManifest(
        binding=binding,
        baseline_root=entry["manifest"]["baseline_root"],
        challenger_root=entry["manifest"].get("challenger_root"),
        challenger_worktree=entry["manifest"].get("challenger_worktree"),
        transitions=transitions,
        usage=PlannerDoctorEpochUsage.from_dict(entry["manifest"].get("usage")),
        stop_reason=entry["manifest"].get("stop_reason") or entry.get("stop_reason"),
        evaluation=(
            PlannerDoctorEpochEvaluation.from_dict(evaluation)
            if evaluation is not None
            else None
        ),
        stage_spans=tuple(entry["manifest"].get("stage_spans") or ()),
    )
    residuals = tuple(
        V2ResidualSignal.from_dict(item) for item in entry.get("residuals") or ()
    )
    return PlannerDoctorEpochResult(
        binding=binding,
        manifest=manifest,
        current_stage=PlannerDoctorEpochStage(entry["current_stage"]),
        stop_reason=PlannerDoctorEpochStopReason(entry["stop_reason"]),
        usage=PlannerDoctorEpochUsage.from_dict(entry.get("usage")),
        journal_path=str(journal_path),
        resumed=resumed,
        idempotent_replay=idempotent_replay,
        residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Stop decision
# ---------------------------------------------------------------------------


def decide_epoch_stop(
    *,
    policy: PlannerDoctorEpochPolicy,
    evaluation: PlannerDoctorEpochEvaluation,
    usage: PlannerDoctorEpochUsage,
) -> PlannerDoctorEpochStopReason | None:
    """Return the first hard stop reason, or None if the epoch may continue."""

    exhausted = usage.exhausted_against(policy.budgets)
    if exhausted:
        return PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION
    if policy.require_live_oracle and not evaluation.oracle_available:
        return PlannerDoctorEpochStopReason.ORACLE_LOSS
    if policy.require_live_telemetry and not evaluation.telemetry_available:
        return PlannerDoctorEpochStopReason.TELEMETRY_LOSS
    if policy.stop_on_safety_regression and evaluation.safety_regression:
        return PlannerDoctorEpochStopReason.SAFETY_REGRESSION
    if policy.stop_on_quality_regression and evaluation.quality_regression:
        return PlannerDoctorEpochStopReason.QUALITY_REGRESSION
    if policy.stop_on_unchanged_residual and evaluation.unchanged_residual:
        return PlannerDoctorEpochStopReason.UNCHANGED_RESIDUAL
    if (
        policy.stop_on_no_admitted_improvement
        and not evaluation.admitted_improvement
    ):
        return PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT
    if evaluation.rollback_succeeded is False:
        return PlannerDoctorEpochStopReason.ROLLBACK_FAILURE
    return None


def residuals_from_evaluation(
    evaluation: PlannerDoctorEpochEvaluation,
    *,
    stop_reason: PlannerDoctorEpochStopReason,
) -> tuple[V2ResidualSignal, ...]:
    """Project evaluation residuals into the generation-2 residual vocabulary."""

    if not evaluation.residual_ids:
        if stop_reason in {
            PlannerDoctorEpochStopReason.COMPLETED,
            PlannerDoctorEpochStopReason.IDEMPOTENT_REPLAY,
            PlannerDoctorEpochStopReason.MODE_DISABLED,
            PlannerDoctorEpochStopReason.POLICY_REQUIRED,
            PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT,
        }:
            return ()
        kind = V2ResidualKind.REGRESSION
        if stop_reason is PlannerDoctorEpochStopReason.UNCHANGED_RESIDUAL:
            kind = V2ResidualKind.UNCHANGED_RESIDUAL
        residual_id = f"residual:epoch-{stop_reason.value}"
        return (
            V2ResidualSignal(
                residual_id=residual_id,
                kind=kind,
                title=f"Epoch stopped: {stop_reason.value}",
                detail=(
                    evaluation.detail
                    or f"Planner/Doctor epoch halted for {stop_reason.value}."
                ),
                acceptance_criteria=(
                    f"Resolve the {stop_reason.value} condition before retry",
                ),
                evidence_ids=evaluation.evidence_ids
                or (f"evidence:{stop_reason.value}",),
                predicted_files=(
                    "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
                ),
                predicted_symbols=("run_planner_doctor_epoch",),
                validation_commands=(
                    "python -m pytest test/api/test_agent_supervisor_planner_doctor_epoch.py -q",
                ),
            ),
        )
    residuals: list[V2ResidualSignal] = []
    for residual_id in evaluation.residual_ids:
        kind = (
            V2ResidualKind.UNCHANGED_RESIDUAL
            if evaluation.unchanged_residual
            else V2ResidualKind.BENCHMARK_RESIDUAL
        )
        residuals.append(
            V2ResidualSignal(
                residual_id=residual_id,
                kind=kind,
                title=f"Live epoch residual {residual_id}",
                detail=evaluation.detail
                or f"Paired live epoch reported residual {residual_id}.",
                acceptance_criteria=(
                    f"Close residual {residual_id} under the live oracle",
                ),
                evidence_ids=evaluation.evidence_ids or (f"evidence:{residual_id}",),
                predicted_files=(
                    "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
                ),
                predicted_symbols=("run_planner_doctor_epoch",),
                validation_commands=(
                    "python -m pytest test/api/test_agent_supervisor_planner_doctor_epoch.py -q",
                ),
            )
        )
    return tuple(residuals)


# ---------------------------------------------------------------------------
# Challenger isolation
# ---------------------------------------------------------------------------


def create_isolated_challenger(
    *,
    work_root: Path,
    epoch_id: str,
    baseline_root: str,
) -> tuple[Path, str]:
    """Create exactly one disposable challenger workspace under ``work_root``.

    Returns ``(worktree_path, challenger_root_id)``.  The workspace is a
    directory with a marker file so unit tests do not require a git checkout.
    """

    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    # Enforce single-challenger: refuse if another challenger already exists
    # for this epoch id.
    worktree = root / f"challenger-{epoch_id[-16:]}"
    if worktree.exists():
        marker = worktree / ".planner_doctor_challenger"
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return worktree, str(payload["challenger_root"])
        raise PlannerDoctorEpochError(
            "challenger worktree path exists without isolation marker"
        )
    worktree.mkdir(parents=True, exist_ok=False)
    challenger_root = _digest(
        {
            "epoch_id": epoch_id,
            "baseline_root": baseline_root,
            "isolation": "single-challenger",
        }
    )
    marker_payload = {
        "epoch_id": epoch_id,
        "baseline_root": baseline_root,
        "challenger_root": challenger_root,
        "isolated": True,
        "max_challengers": MAX_CHALLENGERS_PER_EPOCH,
    }
    (worktree / ".planner_doctor_challenger").write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return worktree, challenger_root


def destroy_isolated_challenger(worktree: Path | None) -> bool:
    """Best-effort removal of a challenger workspace.  Returns success."""

    if worktree is None:
        return True
    path = Path(worktree)
    if not path.exists():
        return True
    marker = path / ".planner_doctor_challenger"
    if not marker.is_file():
        return False
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


EpochEvaluationProvider = Callable[
    [PlannerDoctorEpochBinding, str, str | None],
    PlannerDoctorEpochEvaluation,
]
EpochUsageProvider = Callable[
    [PlannerDoctorEpochBinding, PlannerDoctorEpochStage],
    PlannerDoctorEpochUsage,
]


def _default_evaluation(
    binding: PlannerDoctorEpochBinding,
    baseline_root: str,
    challenger_root: str | None,
) -> PlannerDoctorEpochEvaluation:
    """Conservative default: no admitted improvement, sensors present."""

    del binding, baseline_root, challenger_root
    return PlannerDoctorEpochEvaluation(
        safety_regression=False,
        quality_regression=False,
        unchanged_residual=False,
        admitted_improvement=False,
        oracle_available=True,
        telemetry_available=True,
        detail="default evaluation admits no improvement",
    )


def _default_usage(
    binding: PlannerDoctorEpochBinding,
    stage: PlannerDoctorEpochStage,
) -> PlannerDoctorEpochUsage:
    del binding
    # Minimal non-zero wall accounting so budgets are exercised without
    # fabricating resource claims.
    return PlannerDoctorEpochUsage(
        wall_seconds=1 if stage is not PlannerDoctorEpochStage.STOP else 0,
        epochs=1 if stage is PlannerDoctorEpochStage.BASELINE else 0,
        processes=1,
        challengers=1 if stage is PlannerDoctorEpochStage.SHADOW else 0,
    )


@dataclass
class PlannerDoctorEpochController:
    """Finite, durable, single-challenger epoch controller.

    The controller freezes anchors/budgets on first entry, appends every stage
    transition to a journal, and treats a completed journal entry for the same
    ``epoch_id`` as an idempotent no-op replay.
    """

    repo_root: Path
    journal_path: Path
    work_root: Path
    evaluation_provider: EpochEvaluationProvider = field(
        default=_default_evaluation
    )
    usage_provider: EpochUsageProvider = field(default=_default_usage)
    clock: Callable[[], float] = field(default=time.monotonic)

    def run(
        self,
        binding: PlannerDoctorEpochBinding,
        *,
        force_resume: bool = False,
    ) -> PlannerDoctorEpochResult:
        if not isinstance(binding, PlannerDoctorEpochBinding):
            raise PlannerDoctorEpochError("binding must be PlannerDoctorEpochBinding")
        if not binding.policy.is_enabled:
            return self._disabled_result(
                binding,
                PlannerDoctorEpochStopReason.MODE_DISABLED,
            )

        journal = load_planner_doctor_epoch_journal(self.journal_path)
        existing = journal["epochs"].get(binding.epoch_id)
        if existing is not None and existing.get("terminal"):
            return _result_from_journal_entry(
                existing,
                journal_path=self.journal_path,
                resumed=force_resume or bool(existing.get("resumed")),
                idempotent_replay=True,
            )

        started = self.clock()
        transitions: list[PlannerDoctorEpochTransition] = []
        usage = PlannerDoctorEpochUsage()
        previous: PlannerDoctorEpochStage | None = None
        stage = PlannerDoctorEpochStage.BASELINE
        baseline_root = binding.tree_id
        challenger_root: str | None = None
        challenger_worktree: Path | None = None
        evaluation: PlannerDoctorEpochEvaluation | None = None
        stage_spans: list[dict[str, Any]] = []
        resumed = existing is not None

        # Resume from last non-terminal stage if a partial journal exists.
        if existing is not None and not existing.get("terminal"):
            partial = existing
            for raw in partial.get("transitions") or ():
                transitions.append(PlannerDoctorEpochTransition.from_dict(raw))
            if transitions:
                previous = transitions[-1].stage
                stage = self._next_stage_after(previous)
            usage = PlannerDoctorEpochUsage.from_dict(partial.get("usage"))
            baseline_root = str(partial.get("baseline_root") or baseline_root)
            challenger_root = partial.get("challenger_root")
            if partial.get("challenger_worktree"):
                challenger_worktree = Path(str(partial["challenger_worktree"]))
            resumed = True

        stop_reason: PlannerDoctorEpochStopReason | None = None
        # Guard against pathological stage cycles; hard budget is the real bound.
        max_steps = MAX_JOURNAL_TRANSITIONS
        steps = 0

        while stage is not PlannerDoctorEpochStage.STOP and steps < max_steps:
            steps += 1
            stage_started = self.clock()
            entered_stage = stage
            try:
                (
                    stage,
                    stage_stop,
                    usage,
                    baseline_root,
                    challenger_root,
                    challenger_worktree,
                    evaluation,
                ) = self._advance(
                    binding=binding,
                    stage=stage,
                    previous=previous,
                    usage=usage,
                    baseline_root=baseline_root,
                    challenger_root=challenger_root,
                    challenger_worktree=challenger_worktree,
                    evaluation=evaluation,
                    transitions=transitions,
                    started=started,
                    pending_stop=stop_reason,
                )
                if stage_stop is not None:
                    stop_reason = stage_stop
            except PlannerDoctorEpochError as exc:
                stop_reason = PlannerDoctorEpochStopReason.ROLLBACK_FAILURE
                transition = PlannerDoctorEpochTransition(
                    stage=PlannerDoctorEpochStage.STOP,
                    previous_stage=previous,
                    recorded_at=_timestamp(),
                    usage=usage,
                    stop_reason=stop_reason,
                    detail=str(exc)[:MAX_TEXT_BYTES],
                    baseline_root=baseline_root,
                    challenger_root=challenger_root,
                )
                transitions.append(transition)
                stage = PlannerDoctorEpochStage.STOP
            stage_spans.append(
                {
                    "stage": entered_stage.value,
                    "next_stage": stage.value,
                    "elapsed_seconds": max(0, int(self.clock() - stage_started)),
                }
            )
            previous = entered_stage
            # Persist after every successful transition for crash recovery.
            self._checkpoint(
                binding=binding,
                transitions=transitions,
                usage=usage,
                baseline_root=baseline_root,
                challenger_root=challenger_root,
                challenger_worktree=challenger_worktree,
                stop_reason=stop_reason,
                evaluation=evaluation,
                stage_spans=stage_spans,
                terminal=False,
            )

        if stop_reason is None:
            stop_reason = PlannerDoctorEpochStopReason.COMPLETED

        # Ensure challenger cleanup when not completing a promotion path.
        if (
            stop_reason is not PlannerDoctorEpochStopReason.COMPLETED
            or binding.policy.mode
            in {
                PlannerDoctorEpochMode.OBSERVE,
                PlannerDoctorEpochMode.SHADOW,
            }
        ):
            if challenger_worktree is not None:
                if not destroy_isolated_challenger(challenger_worktree):
                    stop_reason = PlannerDoctorEpochStopReason.ROLLBACK_FAILURE
                    transitions.append(
                        PlannerDoctorEpochTransition(
                            stage=PlannerDoctorEpochStage.ROLLBACK,
                            previous_stage=PlannerDoctorEpochStage.STOP,
                            recorded_at=_timestamp(),
                            usage=usage,
                            stop_reason=stop_reason,
                            detail="challenger rollback failed",
                            baseline_root=baseline_root,
                            challenger_root=challenger_root,
                        )
                    )
                else:
                    challenger_worktree = None

        if evaluation is None:
            evaluation = PlannerDoctorEpochEvaluation()

        # Ensure a terminal STOP transition is always recorded.
        if not transitions or transitions[-1].stage is not PlannerDoctorEpochStage.STOP:
            transitions.append(
                PlannerDoctorEpochTransition(
                    stage=PlannerDoctorEpochStage.STOP,
                    previous_stage=previous,
                    recorded_at=_timestamp(),
                    usage=usage,
                    stop_reason=stop_reason,
                    detail=f"terminal stop: {stop_reason.value}",
                    baseline_root=baseline_root,
                    challenger_root=challenger_root,
                    residual_ids=(
                        evaluation.residual_ids if evaluation is not None else ()
                    ),
                    evidence_ids=(
                        evaluation.evidence_ids if evaluation is not None else ()
                    ),
                )
            )

        residuals = residuals_from_evaluation(evaluation, stop_reason=stop_reason)
        manifest = PlannerDoctorEpochManifest(
            binding=binding,
            baseline_root=baseline_root,
            challenger_root=challenger_root,
            challenger_worktree=(
                str(challenger_worktree) if challenger_worktree is not None else None
            ),
            transitions=tuple(transitions),
            usage=usage,
            stop_reason=stop_reason,
            evaluation=evaluation,
            stage_spans=tuple(stage_spans),
        )
        result = PlannerDoctorEpochResult(
            binding=binding,
            manifest=manifest,
            current_stage=PlannerDoctorEpochStage.STOP,
            stop_reason=stop_reason,
            usage=usage,
            journal_path=str(self.journal_path),
            resumed=resumed,
            idempotent_replay=False,
            residuals=residuals,
        )
        journal = load_planner_doctor_epoch_journal(self.journal_path)
        journal["epochs"][binding.epoch_id] = _journal_entry_from_result(result)
        _persist_journal(self.journal_path, journal)
        return result

    def _disabled_result(
        self,
        binding: PlannerDoctorEpochBinding,
        reason: PlannerDoctorEpochStopReason,
    ) -> PlannerDoctorEpochResult:
        transition = PlannerDoctorEpochTransition(
            stage=PlannerDoctorEpochStage.STOP,
            previous_stage=None,
            recorded_at=_timestamp(),
            usage=PlannerDoctorEpochUsage(),
            stop_reason=reason,
            detail="epoch controller refused: mode/policy not enabled",
            baseline_root=binding.tree_id,
        )
        manifest = PlannerDoctorEpochManifest(
            binding=binding,
            baseline_root=binding.tree_id,
            challenger_root=None,
            challenger_worktree=None,
            transitions=(transition,),
            usage=PlannerDoctorEpochUsage(),
            stop_reason=reason,
        )
        return PlannerDoctorEpochResult(
            binding=binding,
            manifest=manifest,
            current_stage=PlannerDoctorEpochStage.STOP,
            stop_reason=reason,
            usage=PlannerDoctorEpochUsage(),
            journal_path=str(self.journal_path),
        )

    def _next_stage_after(
        self, stage: PlannerDoctorEpochStage
    ) -> PlannerDoctorEpochStage:
        order = (
            PlannerDoctorEpochStage.BASELINE,
            PlannerDoctorEpochStage.PROPOSE,
            PlannerDoctorEpochStage.SHADOW,
            PlannerDoctorEpochStage.EVALUATE,
            PlannerDoctorEpochStage.RETAIN,
            PlannerDoctorEpochStage.RECHECK,
            PlannerDoctorEpochStage.REFILL,
            PlannerDoctorEpochStage.STOP,
        )
        try:
            index = order.index(stage)
        except ValueError:
            return PlannerDoctorEpochStage.STOP
        if index + 1 >= len(order):
            return PlannerDoctorEpochStage.STOP
        return order[index + 1]

    def _advance(
        self,
        *,
        binding: PlannerDoctorEpochBinding,
        stage: PlannerDoctorEpochStage,
        previous: PlannerDoctorEpochStage | None,
        usage: PlannerDoctorEpochUsage,
        baseline_root: str,
        challenger_root: str | None,
        challenger_worktree: Path | None,
        evaluation: PlannerDoctorEpochEvaluation | None,
        transitions: list[PlannerDoctorEpochTransition],
        started: float,
        pending_stop: PlannerDoctorEpochStopReason | None = None,
    ) -> tuple[
        PlannerDoctorEpochStage,
        PlannerDoctorEpochStopReason | None,
        PlannerDoctorEpochUsage,
        str,
        str | None,
        Path | None,
        PlannerDoctorEpochEvaluation | None,
    ]:
        # Wall-clock budget from controller clock.
        elapsed = max(0, int(self.clock() - started))
        stage_usage = self.usage_provider(binding, stage)
        stage_usage = replace(
            stage_usage,
            wall_seconds=max(stage_usage.wall_seconds, 0),
        )
        usage = usage.merge(stage_usage)
        usage = replace(usage, wall_seconds=max(usage.wall_seconds, elapsed))

        exhausted = usage.exhausted_against(binding.policy.budgets)
        if exhausted:
            transition = PlannerDoctorEpochTransition(
                stage=stage,
                previous_stage=previous,
                recorded_at=_timestamp(),
                usage=usage,
                stop_reason=PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION,
                detail=f"budget exhausted: {', '.join(exhausted)}",
                baseline_root=baseline_root,
                challenger_root=challenger_root,
            )
            transitions.append(transition)
            return (
                PlannerDoctorEpochStage.STOP,
                PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION,
                usage,
                baseline_root,
                challenger_root,
                challenger_worktree,
                evaluation,
            )

        # Anchor drift check every stage.
        drifted = binding.anchors.verify_unmutated(self.repo_root)
        if drifted:
            transition = PlannerDoctorEpochTransition(
                stage=PlannerDoctorEpochStage.STOP,
                previous_stage=stage,
                recorded_at=_timestamp(),
                usage=usage,
                stop_reason=PlannerDoctorEpochStopReason.MUTATION_FORBIDDEN,
                detail=f"protected anchor drift: {', '.join(drifted)[:400]}",
                baseline_root=baseline_root,
                challenger_root=challenger_root,
            )
            transitions.append(transition)
            return (
                PlannerDoctorEpochStage.STOP,
                PlannerDoctorEpochStopReason.MUTATION_FORBIDDEN,
                usage,
                baseline_root,
                challenger_root,
                challenger_worktree,
                evaluation,
            )

        detail = ""
        residual_ids: tuple[str, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        stop_reason: PlannerDoctorEpochStopReason | None = pending_stop
        next_stage = self._next_stage_after(stage)

        if stage is PlannerDoctorEpochStage.BASELINE:
            baseline_root = binding.tree_id
            usage = replace(usage, epochs=max(usage.epochs, 1))
            detail = "baseline anchors and budgets frozen"

        elif stage is PlannerDoctorEpochStage.PROPOSE:
            # Bounded propose stage: no mutation of seed board; goals/tasks
            # counts are tracked against budgets only.
            usage = replace(
                usage,
                goals=min(usage.goals + 0, binding.policy.budgets.max_goals),
                tasks=min(usage.tasks + 0, binding.policy.budgets.max_tasks),
            )
            detail = "bounded propose recorded without seed-board mutation"
            if binding.policy.mode is PlannerDoctorEpochMode.OBSERVE:
                # Observe mode never opens a challenger.
                next_stage = PlannerDoctorEpochStage.EVALUATE

        elif stage is PlannerDoctorEpochStage.SHADOW:
            if binding.policy.allow_mutation is False and binding.policy.mode in {
                PlannerDoctorEpochMode.ASSIST,
                PlannerDoctorEpochMode.CANARY,
                PlannerDoctorEpochMode.AUTOMATIC,
            }:
                stop_reason = PlannerDoctorEpochStopReason.MUTATION_FORBIDDEN
                detail = "mutation modes require allow_mutation"
                next_stage = PlannerDoctorEpochStage.STOP
            else:
                if binding.policy.require_isolated_challenger:
                    if challenger_worktree is None:
                        worktree, challenger_root = create_isolated_challenger(
                            work_root=self.work_root,
                            epoch_id=binding.epoch_id,
                            baseline_root=baseline_root,
                        )
                        challenger_worktree = worktree
                    usage = replace(usage, challengers=1)
                    detail = "single isolated challenger prepared"
                else:
                    detail = "challenger isolation not required by policy"

        elif stage is PlannerDoctorEpochStage.EVALUATE:
            evaluation = self.evaluation_provider(
                binding, baseline_root, challenger_root
            )
            residual_ids = evaluation.residual_ids
            evidence_ids = evaluation.evidence_ids
            stop_reason = decide_epoch_stop(
                policy=binding.policy,
                evaluation=evaluation,
                usage=usage,
            )
            if stop_reason is not None:
                detail = evaluation.detail or stop_reason.value
                if stop_reason in {
                    PlannerDoctorEpochStopReason.SAFETY_REGRESSION,
                    PlannerDoctorEpochStopReason.QUALITY_REGRESSION,
                    PlannerDoctorEpochStopReason.ORACLE_LOSS,
                    PlannerDoctorEpochStopReason.TELEMETRY_LOSS,
                    PlannerDoctorEpochStopReason.ROLLBACK_FAILURE,
                    PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION,
                }:
                    next_stage = PlannerDoctorEpochStage.ROLLBACK
                else:
                    next_stage = PlannerDoctorEpochStage.STOP
            else:
                # Admitted improvement under shadow stays retain/recheck only;
                # promotion requires canary mode + operator grant.
                if binding.policy.mode in {
                    PlannerDoctorEpochMode.CANARY,
                    PlannerDoctorEpochMode.AUTOMATIC,
                } and binding.policy.automatic_promotion_enabled:
                    next_stage = PlannerDoctorEpochStage.RETAIN
                else:
                    detail = "improvement observed; retained in shadow"
                    next_stage = PlannerDoctorEpochStage.RETAIN

        elif stage is PlannerDoctorEpochStage.RETAIN:
            detail = "challenger retained pending recheck"
            next_stage = PlannerDoctorEpochStage.RECHECK

        elif stage is PlannerDoctorEpochStage.RECHECK:
            # Re-evaluate after retain.
            evaluation = self.evaluation_provider(
                binding, baseline_root, challenger_root
            )
            residual_ids = evaluation.residual_ids
            evidence_ids = evaluation.evidence_ids
            stop_reason = decide_epoch_stop(
                policy=binding.policy,
                evaluation=evaluation,
                usage=usage,
            )
            if stop_reason is not None:
                detail = evaluation.detail or stop_reason.value
                next_stage = PlannerDoctorEpochStage.ROLLBACK
            else:
                detail = "current-tree recheck passed"
                next_stage = PlannerDoctorEpochStage.REFILL

        elif stage is PlannerDoctorEpochStage.REFILL:
            # Refill emission is owned by PDR-081; this stage only records
            # that residuals are available for the derived source.
            detail = "residuals ready for derived refill"
            next_stage = PlannerDoctorEpochStage.STOP
            if stop_reason is None:
                stop_reason = PlannerDoctorEpochStopReason.COMPLETED

        elif stage is PlannerDoctorEpochStage.ROLLBACK:
            ok = destroy_isolated_challenger(challenger_worktree)
            challenger_worktree = None if ok else challenger_worktree
            if not ok:
                stop_reason = PlannerDoctorEpochStopReason.ROLLBACK_FAILURE
                detail = "rollback failed to destroy challenger"
            else:
                detail = "challenger rolled back to baseline"
                if stop_reason is None:
                    stop_reason = PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT
            next_stage = PlannerDoctorEpochStage.STOP

        elif stage is PlannerDoctorEpochStage.REJECT:
            stop_reason = (
                stop_reason or PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT
            )
            detail = "challenger rejected"
            next_stage = PlannerDoctorEpochStage.STOP

        elif stage is PlannerDoctorEpochStage.PROMOTE:
            # Promotion is never silent: require explicit automatic grant.
            if not binding.policy.automatic_promotion_enabled:
                stop_reason = PlannerDoctorEpochStopReason.MUTATION_FORBIDDEN
                detail = "promotion blocked without automatic grant"
                next_stage = PlannerDoctorEpochStage.ROLLBACK
            else:
                detail = "promotion deferred to operator-controlled rollout"
                stop_reason = PlannerDoctorEpochStopReason.COMPLETED
                next_stage = PlannerDoctorEpochStage.STOP

        elif stage is PlannerDoctorEpochStage.CANARY:
            detail = "canary stage recorded"
            next_stage = PlannerDoctorEpochStage.RECHECK

        else:
            next_stage = PlannerDoctorEpochStage.STOP
            detail = f"unhandled stage {stage.value}"

        transition = PlannerDoctorEpochTransition(
            stage=stage,
            previous_stage=previous,
            recorded_at=_timestamp(),
            usage=usage,
            stop_reason=stop_reason,
            detail=detail,
            baseline_root=baseline_root,
            challenger_root=challenger_root,
            residual_ids=residual_ids,
            evidence_ids=evidence_ids,
        )
        transitions.append(transition)
        return (
            next_stage,
            stop_reason,
            usage,
            baseline_root,
            challenger_root,
            challenger_worktree,
            evaluation,
        )

    def _checkpoint(
        self,
        *,
        binding: PlannerDoctorEpochBinding,
        transitions: Sequence[PlannerDoctorEpochTransition],
        usage: PlannerDoctorEpochUsage,
        baseline_root: str,
        challenger_root: str | None,
        challenger_worktree: Path | None,
        stop_reason: PlannerDoctorEpochStopReason | None,
        evaluation: PlannerDoctorEpochEvaluation | None,
        stage_spans: Sequence[Mapping[str, Any]],
        terminal: bool,
    ) -> None:
        journal = load_planner_doctor_epoch_journal(self.journal_path)
        journal["epochs"][binding.epoch_id] = {
            "epoch_id": binding.epoch_id,
            "binding": binding.to_dict(),
            "baseline_root": baseline_root,
            "challenger_root": challenger_root,
            "challenger_worktree": (
                str(challenger_worktree) if challenger_worktree is not None else None
            ),
            "transitions": [item.to_dict() for item in transitions],
            "usage": usage.to_dict(),
            "stop_reason": stop_reason.value if stop_reason is not None else None,
            "evaluation": (
                evaluation.to_dict() if evaluation is not None else None
            ),
            "stage_spans": [dict(item) for item in stage_spans],
            "terminal": terminal,
            "manifest": {
                "schema": PLANNER_DOCTOR_EPOCH_MANIFEST_SCHEMA,
                "baseline_root": baseline_root,
                "challenger_root": challenger_root,
                "challenger_worktree": (
                    str(challenger_worktree)
                    if challenger_worktree is not None
                    else None
                ),
                "transitions": [item.to_dict() for item in transitions],
                "usage": usage.to_dict(),
                "stop_reason": (
                    stop_reason.value if stop_reason is not None else None
                ),
                "evaluation": (
                    evaluation.to_dict() if evaluation is not None else None
                ),
                "stage_spans": [dict(item) for item in stage_spans],
            },
            "current_stage": (
                transitions[-1].stage.value if transitions else "baseline"
            ),
            "resumed": False,
            "idempotent_replay": False,
            "residuals": [],
        }
        _persist_journal(self.journal_path, journal)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_planner_doctor_epoch_binding(
    *,
    repo_root: Path,
    repository_id: str,
    tree_id: str,
    policy: PlannerDoctorEpochPolicy | Mapping[str, Any],
    objective_revision: str,
    board_revision: str,
    capability_revision: str,
    operator_revision: str = "operator:planner-doctor-epoch@1",
    authority_policy_revision: str = "1",
    benchmark_policy_revision: str = "1",
    protected_paths: Sequence[str] = DEFAULT_PROTECTED_ANCHOR_PATHS,
    epoch_index: int = 0,
    observed_at: datetime | str | None = None,
    anchors: PlannerDoctorEpochAnchors | None = None,
) -> PlannerDoctorEpochBinding:
    """Freeze anchors/budgets and build the immutable epoch binding."""

    resolved_policy = (
        policy
        if isinstance(policy, PlannerDoctorEpochPolicy)
        else PlannerDoctorEpochPolicy.from_dict(policy)
    )
    resolved_anchors = anchors or freeze_planner_doctor_anchors(
        repo_root=Path(repo_root),
        repository_id=repository_id,
        tree_id=tree_id,
        authority_policy_revision=authority_policy_revision,
        benchmark_policy_revision=benchmark_policy_revision,
        protected_paths=protected_paths,
        frozen_at=observed_at,
    )
    return PlannerDoctorEpochBinding(
        repository_id=repository_id,
        tree_id=tree_id,
        policy=resolved_policy,
        anchors=resolved_anchors,
        objective_revision=objective_revision,
        board_revision=board_revision,
        capability_revision=capability_revision,
        operator_revision=operator_revision,
        epoch_index=epoch_index,
        observed_at=_timestamp(observed_at),
    )


def run_planner_doctor_epoch(
    *,
    binding: PlannerDoctorEpochBinding,
    repo_root: Path,
    journal_path: Path,
    work_root: Path | None = None,
    evaluation_provider: EpochEvaluationProvider | None = None,
    usage_provider: EpochUsageProvider | None = None,
) -> PlannerDoctorEpochResult:
    """Run one bounded live epoch under the explicit binding policy/mode."""

    controller = PlannerDoctorEpochController(
        repo_root=Path(repo_root),
        journal_path=Path(journal_path),
        work_root=Path(work_root or (Path(journal_path).parent / "challengers")),
        evaluation_provider=evaluation_provider or _default_evaluation,
        usage_provider=usage_provider or _default_usage,
    )
    return controller.run(binding)


def resume_planner_doctor_epoch(
    *,
    binding: PlannerDoctorEpochBinding,
    repo_root: Path,
    journal_path: Path,
    work_root: Path | None = None,
    evaluation_provider: EpochEvaluationProvider | None = None,
    usage_provider: EpochUsageProvider | None = None,
) -> PlannerDoctorEpochResult:
    """Resume or idempotently replay an epoch from the durable journal."""

    controller = PlannerDoctorEpochController(
        repo_root=Path(repo_root),
        journal_path=Path(journal_path),
        work_root=Path(work_root or (Path(journal_path).parent / "challengers")),
        evaluation_provider=evaluation_provider or _default_evaluation,
        usage_provider=usage_provider or _default_usage,
    )
    return controller.run(binding, force_resume=True)


def evaluation_from_causal_receipts(
    baseline: V2CausalReceipt | Mapping[str, Any],
    challenger: V2CausalReceipt | Mapping[str, Any] | None,
    *,
    oracle_available: bool = True,
    telemetry_available: bool = True,
) -> PlannerDoctorEpochEvaluation:
    """Bridge paired ``V2CausalReceipt`` arms into an epoch evaluation."""

    if isinstance(baseline, Mapping):
        baseline = V2CausalReceipt.from_dict(baseline)
    if not isinstance(baseline, V2CausalReceipt):
        raise PlannerDoctorEpochError("baseline must be V2CausalReceipt")
    challenger_receipt: V2CausalReceipt | None
    if challenger is None:
        challenger_receipt = None
    elif isinstance(challenger, Mapping):
        challenger_receipt = V2CausalReceipt.from_dict(challenger)
    elif isinstance(challenger, V2CausalReceipt):
        challenger_receipt = challenger
    else:
        raise PlannerDoctorEpochError("challenger must be V2CausalReceipt or None")

    safety_regression = not baseline.safety_passed
    quality_regression = False
    admitted_improvement = False
    challenger_id = None
    if challenger_receipt is not None:
        challenger_id = challenger_receipt.receipt_id
        if not challenger_receipt.safety_passed:
            safety_regression = True
        # Quality gate: challenger must not worsen expected terminal outcome
        # identity; detailed Pareto is owned by rollout (PDR-082).
        if (
            challenger_receipt.metrics.expected_terminal_outcome
            != baseline.metrics.expected_terminal_outcome
        ):
            quality_regression = True
        admitted_improvement = (
            challenger_receipt.safety_passed
            and not quality_regression
            and challenger_receipt.receipt_id != baseline.receipt_id
        )
    return PlannerDoctorEpochEvaluation(
        safety_regression=safety_regression,
        quality_regression=quality_regression,
        unchanged_residual=False,
        admitted_improvement=admitted_improvement,
        oracle_available=oracle_available,
        telemetry_available=telemetry_available,
        baseline_receipt_id=baseline.receipt_id,
        challenger_receipt_id=challenger_id,
        evidence_ids=tuple(baseline.source_receipt_ids),
        detail="evaluation derived from V2CausalReceipt pair",
    )


def assert_not_self_improvement_epoch_masquerade(module_globals: Mapping[str, Any]) -> None:
    """Fail closed if a daemon path re-exports the test-only epoch helper."""

    forbidden = (
        "run_self_improvement_epoch",
        "evaluate_self_improvement_epoch",
        "build_self_improvement_epoch_binding",
    )
    for name in forbidden:
        if name in module_globals:
            raise PlannerDoctorEpochError(
                f"daemon integration must not expose {name}; "
                "use run_planner_doctor_epoch"
            )


__all__ = [
    "DEFAULT_PROTECTED_ANCHOR_PATHS",
    "MAX_CHALLENGERS_PER_EPOCH",
    "MAX_EPOCHS_PER_RUN",
    "MAX_GOALS_PER_EPOCH",
    "MAX_TASKS_PER_EPOCH",
    "PLANNER_DOCTOR_EPOCH_BINDING_SCHEMA",
    "PLANNER_DOCTOR_EPOCH_CONTRACT_VERSION",
    "PLANNER_DOCTOR_EPOCH_INTERFACE",
    "PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA",
    "PLANNER_DOCTOR_EPOCH_MANIFEST_SCHEMA",
    "PLANNER_DOCTOR_EPOCH_POLICY_ID",
    "PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID",
    "PLANNER_DOCTOR_EPOCH_RESULT_SCHEMA",
    "PLANNER_DOCTOR_EPOCH_SCHEMA",
    "PLANNER_DOCTOR_EPOCH_TRANSITION_SCHEMA",
    "RESUMABLE_STAGES",
    "TERMINAL_STAGES",
    "PlannerDoctorEpochAnchors",
    "PlannerDoctorEpochBinding",
    "PlannerDoctorEpochBudgets",
    "PlannerDoctorEpochController",
    "PlannerDoctorEpochError",
    "PlannerDoctorEpochEvaluation",
    "PlannerDoctorEpochManifest",
    "PlannerDoctorEpochMode",
    "PlannerDoctorEpochPolicy",
    "PlannerDoctorEpochResult",
    "PlannerDoctorEpochStage",
    "PlannerDoctorEpochStopReason",
    "PlannerDoctorEpochTransition",
    "PlannerDoctorEpochUsage",
    "assert_not_self_improvement_epoch_masquerade",
    "build_planner_doctor_epoch_binding",
    "create_isolated_challenger",
    "decide_epoch_stop",
    "destroy_isolated_challenger",
    "evaluation_from_causal_receipts",
    "freeze_planner_doctor_anchors",
    "load_planner_doctor_epoch_journal",
    "residuals_from_evaluation",
    "resume_planner_doctor_epoch",
    "run_planner_doctor_epoch",
]
