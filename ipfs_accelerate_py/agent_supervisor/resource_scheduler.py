"""Live host and LLM-provider admission policy for supervisor lanes.

The objects in this module deliberately serialize without floating point
values.  Profile-G artifacts reject floats, and scheduler decisions often end
up embedded in those artifacts.  Percentages are therefore whole percentages
and all resource sizes, token counts, durations, and capacities are integers.

Provider telemetry is not owned by :mod:`llm_router`; different providers
expose it through different monitors.  :func:`normalize_provider_capacity`
accepts the common telemetry spellings and turns them into one conservative
model.  Unknown limits use ``-1`` (unbounded/not reported), while an explicit
zero is always treated as exhausted.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


UNKNOWN_LIMIT = -1


class ProofResourceClass(str, Enum):
    """Canonical supervisor resource classes used by proof-plan work."""

    TRANSLATION = "cpu-proof-translate"
    SOLVER = "cpu-proof-solver"
    SOLVER_PORTFOLIO = "cpu-proof-solver"
    KERNEL = "cpu-proof-kernel"
    KERNEL_RECONSTRUCTION = "cpu-proof-kernel"
    VALIDATION = "cpu-validation"
    # Deterministic schema/type acceptance is separately backpressured from
    # general tests and static validation.
    TYPE_CHECK = "cpu-proof-type-check"
    DETERMINISTIC_TYPE_CHECK = "cpu-proof-type-check"
    MODEL_DRAFT = "llm-proof-draft"
    MODEL_DRAFTING = "llm-proof-draft"
    ARTIFACT = "io-artifact"

    @property
    def pool(self) -> str:
        if self is ProofResourceClass.MODEL_DRAFT:
            return "model"
        if self is ProofResourceClass.ARTIFACT:
            return "artifact"
        return "cpu-proof"


PROOF_RESOURCE_CLASSES = tuple(item.value for item in ProofResourceClass)
LEGACY_RESOURCE_CLASSES = (
    "cpu-small",
    "cpu-medium",
    "cpu-large",
)
# Default hosts advertise the architecture's distinct work classes.
# Generic bundle classes remain interoperable through the compatibility check
# in ``_host_reasons`` and can still be advertised explicitly by old workers.
DEFAULT_RESOURCE_CLASSES = PROOF_RESOURCE_CLASSES

_RESOURCE_CLASS_ALIASES = {
    "translate": ProofResourceClass.TRANSLATION.value,
    "translation": ProofResourceClass.TRANSLATION.value,
    "proof-translate": ProofResourceClass.TRANSLATION.value,
    "solve": ProofResourceClass.SOLVER.value,
    "solver": ProofResourceClass.SOLVER.value,
    "proof-solver": ProofResourceClass.SOLVER.value,
    "reconstruct": ProofResourceClass.KERNEL.value,
    "kernel": ProofResourceClass.KERNEL.value,
    "kernel_verify": ProofResourceClass.KERNEL.value,
    "kernel-verify": ProofResourceClass.KERNEL.value,
    "validate": ProofResourceClass.VALIDATION.value,
    "validation": ProofResourceClass.VALIDATION.value,
    "typecheck": ProofResourceClass.TYPE_CHECK.value,
    "type-check": ProofResourceClass.TYPE_CHECK.value,
    "type_check": ProofResourceClass.TYPE_CHECK.value,
    "deterministic-type-check": ProofResourceClass.TYPE_CHECK.value,
    "deterministic_type_check": ProofResourceClass.TYPE_CHECK.value,
    "model_draft": ProofResourceClass.MODEL_DRAFT.value,
    "model-draft": ProofResourceClass.MODEL_DRAFT.value,
    "persist": ProofResourceClass.ARTIFACT.value,
    "artifact": ProofResourceClass.ARTIFACT.value,
    "attest": ProofResourceClass.ARTIFACT.value,
}


class ProofWorkKind(str, Enum):
    """Closed work vocabulary for the goal-development proof runtime."""

    MODEL_DRAFT = "model_draft"
    MODEL_DRAFTING = "model_draft"
    TYPE_CHECK = "type_check"
    DETERMINISTIC_TYPE_CHECK = "type_check"
    SOLVER_PORTFOLIO = "solver_portfolio"
    KERNEL_RECONSTRUCTION = "kernel_reconstruction"

    @property
    def resource_class(self) -> ProofResourceClass:
        return {
            ProofWorkKind.MODEL_DRAFT: ProofResourceClass.MODEL_DRAFT,
            ProofWorkKind.TYPE_CHECK: ProofResourceClass.TYPE_CHECK,
            ProofWorkKind.SOLVER_PORTFOLIO: ProofResourceClass.SOLVER,
            ProofWorkKind.KERNEL_RECONSTRUCTION: ProofResourceClass.KERNEL,
        }[self]


_PROOF_WORK_KIND_ALIASES = {
    "draft": ProofWorkKind.MODEL_DRAFT,
    "leanstral": ProofWorkKind.MODEL_DRAFT,
    "model": ProofWorkKind.MODEL_DRAFT,
    "model-draft": ProofWorkKind.MODEL_DRAFT,
    "typecheck": ProofWorkKind.TYPE_CHECK,
    "type-check": ProofWorkKind.TYPE_CHECK,
    "deterministic-type-check": ProofWorkKind.TYPE_CHECK,
    "deterministic_type_check": ProofWorkKind.TYPE_CHECK,
    "solve": ProofWorkKind.SOLVER_PORTFOLIO,
    "solver": ProofWorkKind.SOLVER_PORTFOLIO,
    "solver-portfolio": ProofWorkKind.SOLVER_PORTFOLIO,
    "kernel": ProofWorkKind.KERNEL_RECONSTRUCTION,
    "kernel-reconstruct": ProofWorkKind.KERNEL_RECONSTRUCTION,
    "kernel-reconstruction": ProofWorkKind.KERNEL_RECONSTRUCTION,
}


def normalize_proof_work_kind(value: Any) -> ProofWorkKind:
    """Normalize a public work-kind spelling into the closed runtime enum."""

    if isinstance(value, ProofWorkKind):
        return value
    raw = str(getattr(value, "value", value) or "").strip().lower()
    raw = raw.replace(" ", "_")
    try:
        return ProofWorkKind(raw)
    except ValueError:
        alias = _PROOF_WORK_KIND_ALIASES.get(raw)
        if alias is None:
            alias = _PROOF_WORK_KIND_ALIASES.get(raw.replace("_", "-"))
        if alias is None:
            raise ValueError(f"unsupported proof work kind: {value!r}") from None
        return alias


def resource_class_for_work_kind(value: Any) -> str:
    """Return the canonical, independently limitable class for ``value``."""

    return normalize_proof_work_kind(value).resource_class.value


def normalize_resource_class(value: Any, *, stage: Any = "") -> str:
    """Return a canonical proof resource class while preserving extensions."""

    raw = str(value or "").strip().lower()
    stage_name = getattr(stage, "value", stage)
    stage_raw = str(stage_name or "").strip().lower()
    if raw in PROOF_RESOURCE_CLASSES:
        return raw
    if raw in _RESOURCE_CLASS_ALIASES:
        return _RESOURCE_CLASS_ALIASES[raw]
    if not raw and stage_raw:
        return _RESOURCE_CLASS_ALIASES.get(stage_raw, stage_raw)
    return raw


def resource_pool(resource_class: Any) -> str:
    """Classify a resource class into independently accounted capacity."""

    normalized = normalize_resource_class(resource_class)
    if normalized == ProofResourceClass.MODEL_DRAFT.value:
        return "model"
    if normalized == ProofResourceClass.ARTIFACT.value:
        return "artifact"
    return "cpu-proof"


def _integer(value: Any, default: int = 0, *, minimum: int | None = None) -> int:
    """Coerce telemetry to an integer without leaking floats into artifacts."""

    if value is None or isinstance(value, bool):
        result = default
    else:
        try:
            # Numeric strings such as ``"20.5"`` occur in psutil adapters.
            result = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            result = default
    return max(minimum, result) if minimum is not None else result


def _boolean(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off", "down", "unhealthy", "failed"}:
            return False
        if normalized in {"1", "true", "yes", "on", "up", "healthy", "ready"}:
            return True
    return bool(value)


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        items = value
    else:
        items = (value,)
    return tuple(sorted({str(item).strip().lower() for item in items if str(item).strip()}))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class HostResourceSnapshot:
    """One measured view of the machine hosting a worker pool."""

    observed_at_ms: int = 0
    cpu_percent: int = 0
    memory_percent: int = 0
    disk_percent: int = 0
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    disk_total_bytes: int = 0
    disk_available_bytes: int = 0
    active_phase: str = "scheduler"
    active_workers: int = 0
    worker_limit: int = 1
    available_worker_capacity: int = 1
    capabilities: tuple[str, ...] = ("cpu",)
    resource_classes: tuple[str, ...] = DEFAULT_RESOURCE_CLASSES

    def __post_init__(self) -> None:
        for name in ("cpu_percent", "memory_percent", "disk_percent"):
            value = int(getattr(self, name))
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        for name in (
            "observed_at_ms", "memory_total_bytes", "memory_available_bytes",
            "disk_total_bytes", "disk_available_bytes",
            "active_workers", "worker_limit", "available_worker_capacity",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def occupied_worker_capacity(self) -> int:
        return self.active_workers

    @property
    def cpu_millionths(self) -> int:
        return self.cpu_percent * 10_000

    @property
    def memory_used_bytes(self) -> int:
        return max(0, self.memory_total_bytes - self.memory_available_bytes)

    @property
    def disk_used_bytes(self) -> int:
        return max(0, self.disk_total_bytes - self.disk_available_bytes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = list(self.capabilities)
        payload["resource_classes"] = list(self.resource_classes)
        payload["occupied_worker_capacity"] = self.occupied_worker_capacity
        payload["cpu_millionths"] = self.cpu_millionths
        payload["memory_used_bytes"] = self.memory_used_bytes
        payload["disk_used_bytes"] = self.disk_used_bytes
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HostResourceSnapshot":
        worker_limit = _integer(
            _first(value, ("worker_limit", "max_workers", "max_lanes", "capacity"), 1),
            1,
            minimum=0,
        )
        active = _integer(
            _first(value, ("active_workers", "occupied_worker_capacity", "running_workers"), 0),
            0,
            minimum=0,
        )
        available_value = _first(
            value,
            ("available_worker_capacity", "available_workers", "free_slots"),
            None,
        )
        available = (
            max(0, worker_limit - active)
            if available_value is None
            else _integer(available_value, 0, minimum=0)
        )
        return cls(
            observed_at_ms=_integer(
                _first(value, ("observed_at_ms", "measured_at_ms", "timestamp_ms"), 0),
                0,
                minimum=0,
            ),
            cpu_percent=_integer(_first(value, ("cpu_percent", "cpu_usage_percent", "cpu"), 0), 0, minimum=0),
            memory_percent=_integer(_first(value, ("memory_percent", "memory_usage_percent", "memory"), 0), 0, minimum=0),
            disk_percent=_integer(_first(value, ("disk_percent", "disk_usage_percent", "disk"), 0), 0, minimum=0),
            memory_available_bytes=_integer(
                _first(value, ("memory_available_bytes", "available_memory_bytes", "memory_free_bytes"), 0),
                0,
                minimum=0,
            ),
            memory_total_bytes=_integer(
                _first(value, ("memory_total_bytes", "total_memory_bytes"), 0),
                0,
                minimum=0,
            ),
            disk_available_bytes=_integer(
                _first(value, ("disk_available_bytes", "available_disk_bytes", "disk_free_bytes"), 0),
                0,
                minimum=0,
            ),
            disk_total_bytes=_integer(
                _first(value, ("disk_total_bytes", "total_disk_bytes"), 0),
                0,
                minimum=0,
            ),
            active_phase=str(_first(value, ("active_phase", "phase"), "scheduler") or "scheduler"),
            active_workers=active,
            worker_limit=worker_limit,
            available_worker_capacity=available,
            capabilities=_strings(value.get("capabilities")) or ("cpu",),
            resource_classes=_strings(value.get("resource_classes")) or DEFAULT_RESOURCE_CLASSES,
        )


def sample_host_resources(
    path: Path | str = ".",
    *,
    active_workers: int = 0,
    worker_limit: int = 1,
    active_phase: str = "scheduler",
) -> HostResourceSnapshot:
    """Measure CPU, memory and disk without requiring psutil at import time."""

    active = max(0, int(active_workers))
    limit = max(0, int(worker_limit))
    target = Path(path)
    # disk_usage requires an existing path.  Walk up for new state roots.
    while not target.exists() and target != target.parent:
        target = target.parent

    cpu_percent = 0
    memory_percent = 0
    memory_available = 0
    memory_total = 0
    disk_percent = 0
    disk_available = 0
    disk_total = 0
    try:
        import psutil  # type: ignore[import-not-found]

        cpu_percent = _integer(psutil.cpu_percent(interval=None), 0, minimum=0)
        memory = psutil.virtual_memory()
        memory_percent = _integer(memory.percent, 0, minimum=0)
        memory_available = _integer(memory.available, 0, minimum=0)
        memory_total = _integer(memory.total, 0, minimum=0)
        disk = psutil.disk_usage(str(target))
        disk_percent = _integer(disk.percent, 0, minimum=0)
        disk_available = _integer(disk.free, 0, minimum=0)
        disk_total = _integer(disk.total, 0, minimum=0)
    except (ImportError, AttributeError, OSError):
        # Portable fallbacks retain useful byte headroom even on minimal hosts.
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            memory_available = page_size * available_pages
            memory_total = page_size * physical_pages
            if memory_total > 0:
                memory_percent = max(0, min(100, 100 - (memory_available * 100 // memory_total)))
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        try:
            stat = os.statvfs(target)
            disk_available = int(stat.f_bavail) * int(stat.f_frsize)
            disk_total = int(stat.f_blocks) * int(stat.f_frsize)
            if disk_total > 0:
                disk_percent = max(0, min(100, 100 - (disk_available * 100 // disk_total)))
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    return HostResourceSnapshot(
        observed_at_ms=int(time.time() * 1000),
        cpu_percent=min(100, cpu_percent),
        memory_percent=min(100, memory_percent),
        disk_percent=min(100, disk_percent),
        memory_total_bytes=memory_total,
        memory_available_bytes=memory_available,
        disk_total_bytes=disk_total,
        disk_available_bytes=disk_available,
        active_phase=str(active_phase or "scheduler"),
        active_workers=active,
        worker_limit=limit,
        available_worker_capacity=max(0, limit - active),
    )


@dataclass(frozen=True)
class ProviderCapacity:
    """Normalized live capacity for one llm_router provider."""

    provider_id: str
    healthy: bool = True
    quota_remaining: int = UNKNOWN_LIMIT
    latency_ms: int = 0
    context_window_tokens: int = UNKNOWN_LIMIT
    token_budget_remaining: int = UNKNOWN_LIMIT
    max_concurrency: int = 1
    active_requests: int = 0
    capabilities: tuple[str, ...] = ()
    observed_at_ms: int = 0
    retry_after_ms: int = 0

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        object.__setattr__(self, "provider_id", self.provider_id.strip().lower())
        object.__setattr__(self, "capabilities", _strings(self.capabilities))
        for name in ("latency_ms", "max_concurrency", "active_requests", "observed_at_ms", "retry_after_ms"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("quota_remaining", "context_window_tokens", "token_budget_remaining"):
            if int(getattr(self, name)) < UNKNOWN_LIMIT:
                raise ValueError(f"{name} must be -1 or non-negative")

    @property
    def available_concurrency(self) -> int:
        return max(0, self.max_concurrency - self.active_requests)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = list(self.capabilities)
        payload["available_concurrency"] = self.available_concurrency
        return payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        provider_id: str = "",
    ) -> "ProviderCapacity":
        health_data = _mapping(value.get("health"))
        quota_data = _mapping(value.get("quota"))
        latency_data = _mapping(value.get("latency"))
        context_data = _mapping(value.get("context"))
        token_data = _mapping(value.get("token_budget") or value.get("tokens"))
        concurrency_data = _mapping(value.get("concurrency"))

        def first_from(
            sources: Sequence[Mapping[str, Any]],
            names: Sequence[str],
            default: Any,
        ) -> Any:
            for source in sources:
                found = _first(source, names, None)
                if found is not None:
                    return found
            return default

        identity = str(
            provider_id
            or _first(value, ("provider_id", "provider", "name", "effective_provider_name"), "")
        ).strip().lower()
        status = first_from(
            (value, health_data),
            ("healthy", "available", "ready", "status", "state"),
            True,
        )
        if isinstance(status, str) and status.strip().lower() in {
            "down", "failed", "error", "disabled", "offline", "unhealthy",
            "quota_exhausted", "rate_limited",
        }:
            status = False
        max_concurrency = _integer(
            first_from(
                (value, concurrency_data),
                ("max_concurrency", "concurrency_limit", "max_workers", "capacity", "limit"),
                1,
            ),
            1,
            minimum=0,
        )
        active_requests = _integer(
            first_from(
                (value, concurrency_data),
                ("active_requests", "in_flight", "inflight", "occupied_capacity", "active", "used"),
                0,
            ),
            0,
            minimum=0,
        )
        # Some monitors expose only the number of free request slots.
        available = _first(value, ("available_concurrency", "available_capacity", "free_slots"), None)
        if available is not None and not any(
            key in value for key in ("max_concurrency", "concurrency_limit", "max_workers", "capacity")
        ):
            max_concurrency = active_requests + _integer(available, 0, minimum=0)
        return cls(
            provider_id=identity,
            healthy=_boolean(status, True),
            quota_remaining=_integer(
                first_from(
                    (value, quota_data),
                    ("quota_remaining", "remaining_quota", "requests_remaining", "rate_limit_remaining", "remaining"),
                    UNKNOWN_LIMIT,
                ),
                UNKNOWN_LIMIT,
            ),
            latency_ms=_integer(
                first_from(
                    (value, latency_data),
                    ("latency_ms", "p95_latency_ms", "average_latency_ms", "avg_latency_ms", "p95_ms", "average_ms"),
                    0,
                ),
                0,
                minimum=0,
            ),
            context_window_tokens=_integer(
                first_from(
                    (value, context_data),
                    ("context_window_tokens", "context_tokens", "max_context_tokens", "context_length", "max_tokens", "limit"),
                    UNKNOWN_LIMIT,
                ),
                UNKNOWN_LIMIT,
            ),
            token_budget_remaining=_integer(
                first_from(
                    (value, token_data),
                    ("token_budget_remaining", "remaining_tokens", "tokens_remaining", "token_quota_remaining", "remaining", "available"),
                    UNKNOWN_LIMIT,
                ),
                UNKNOWN_LIMIT,
            ),
            max_concurrency=max_concurrency,
            active_requests=active_requests,
            capabilities=_strings(_first(value, ("capabilities", "supported_capabilities", "features"), ())),
            observed_at_ms=_integer(_first(value, ("observed_at_ms", "measured_at_ms", "timestamp_ms"), 0), 0, minimum=0),
            retry_after_ms=_integer(_first(value, ("retry_after_ms", "backoff_ms", "cooldown_ms"), 0), 0, minimum=0),
        )


def normalize_provider_capacity(
    value: ProviderCapacity | Mapping[str, Any],
    *,
    provider_id: str = "",
) -> ProviderCapacity:
    if isinstance(value, ProviderCapacity):
        if provider_id and provider_id.strip().lower() != value.provider_id:
            return replace(value, provider_id=provider_id.strip().lower())
        return value
    return ProviderCapacity.from_mapping(value, provider_id=provider_id)


def normalize_provider_capacities(
    values: Mapping[str, Any] | Iterable[ProviderCapacity | Mapping[str, Any]] | None,
) -> tuple[ProviderCapacity, ...]:
    """Normalize provider-keyed or sequence telemetry in stable name order."""

    normalized: list[ProviderCapacity] = []
    if values is None:
        return ()
    if isinstance(values, Mapping):
        nested_providers = values.get("providers")
        if isinstance(nested_providers, Mapping):
            return normalize_provider_capacities(nested_providers)
        if isinstance(nested_providers, Iterable) and not isinstance(
            nested_providers, (str, bytes)
        ):
            return normalize_provider_capacities(nested_providers)
        # A mapping containing provider fields is one snapshot; otherwise it is
        # a provider-name -> snapshot collection.
        identity_keys = {"provider_id", "provider", "name", "effective_provider_name"}
        telemetry_keys = identity_keys | {"healthy", "status", "max_concurrency", "latency_ms"}
        if telemetry_keys.intersection(values):
            normalized.append(normalize_provider_capacity(values))
        else:
            for name, raw in values.items():
                if isinstance(raw, ProviderCapacity):
                    normalized.append(normalize_provider_capacity(raw, provider_id=str(name)))
                elif isinstance(raw, Mapping):
                    normalized.append(normalize_provider_capacity(raw, provider_id=str(name)))
    else:
        for raw in values:
            normalized.append(normalize_provider_capacity(raw))
    by_id = {item.provider_id: item for item in normalized}
    return tuple(by_id[name] for name in sorted(by_id))


@dataclass(frozen=True)
class LaneResourceRequirements:
    """Resources and provider features needed by one candidate lane."""

    lane_id: str = ""
    resource_class: str = "cpu-small"
    required_capabilities: tuple[str, ...] = ()
    provider_id: str = ""
    requires_provider: bool = False
    context_tokens: int = 0
    token_budget: int = 0
    quota_units: int = 1
    memory_bytes: int = 0
    disk_bytes: int = 0
    max_provider_latency_ms: int = 0
    process_slots: int = 1

    def __post_init__(self) -> None:
        for name in (
            "context_tokens",
            "token_budget",
            "quota_units",
            "memory_bytes",
            "disk_bytes",
            "max_provider_latency_ms",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.process_slots, bool) or int(self.process_slots) <= 0:
            raise ValueError("process_slots must be a positive integer")
        object.__setattr__(
            self,
            "resource_class",
            normalize_resource_class(self.resource_class),
        )
        object.__setattr__(
            self,
            "provider_id",
            str(self.provider_id or "").strip().lower(),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _strings(self.required_capabilities),
        )

    @property
    def provider_required(self) -> bool:
        return bool(
            self.requires_provider
            or self.provider_id
            or self.context_tokens
            or self.token_budget
            or any(item.startswith("llm:") for item in self.required_capabilities)
        )

    @property
    def resource_pool(self) -> str:
        return resource_pool(self.resource_class)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_capabilities"] = list(self.required_capabilities)
        payload["provider_required"] = self.provider_required
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LaneResourceRequirements":
        # Planning payloads may retain resource hints in the selected task or
        # queue payload.  Top-level explicit fields win.
        queue = _mapping(value.get("queue_payload"))
        task = _mapping(_mapping(value.get("profile_g") or queue.get("profile_g")).get("task"))

        def first(*names: str, default: Any = None) -> Any:
            for source in (value, queue, task):
                found = _first(source, names, None)
                if found is not None:
                    return found
            return default

        provider = str(first("provider_id", "llm_provider", "provider", default="") or "").strip().lower()
        context = _integer(
            first("context_tokens", "required_context_tokens", "estimated_context_tokens", "context_length", default=0),
            0,
            minimum=0,
        )
        tokens = _integer(
            first("token_budget", "required_tokens", "estimated_tokens", "max_new_tokens", default=0),
            0,
            minimum=0,
        )
        return cls(
            lane_id=str(first("lane_id", "bundle_key", "parallel_lane", "task_cid", default="") or ""),
            resource_class=str(first("resource_class", default="cpu-small") or "cpu-small").strip().lower(),
            required_capabilities=_strings(first("required_capabilities", "capabilities", default=())),
            provider_id=provider,
            requires_provider=_boolean(first("requires_provider", "requires_llm", default=False), False),
            context_tokens=context,
            token_budget=tokens,
            quota_units=_integer(first("quota_units", "quota_cost", "request_cost", default=1), 1, minimum=0),
            memory_bytes=_integer(first("memory_bytes", "required_memory_bytes", default=0), 0, minimum=0),
            disk_bytes=_integer(first("disk_bytes", "required_disk_bytes", default=0), 0, minimum=0),
            max_provider_latency_ms=_integer(
                first("max_provider_latency_ms", "max_latency_ms", "latency_budget_ms", default=0),
                0,
                minimum=0,
            ),
            process_slots=_integer(
                first(
                    "process_slots",
                    "required_processes",
                    "processes",
                    "portfolio_width",
                    default=1,
                ),
                1,
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class ChildResourceLimits:
    """Hard limits passed from one supervisor lease to nested child pools."""

    max_processes: int
    portfolio_max_parallel: int
    kernel_max_parallel: int
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    model_token_limit: int = 0
    provider_quota: int = 0
    context_tokens: int = 0
    maximum_provider_latency_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_processes",
            "portfolio_max_parallel",
            "kernel_max_parallel",
            "wall_time_ms",
            "cpu_time_ms",
            "memory_bytes",
            "disk_bytes",
            "model_token_limit",
            "provider_quota",
            "context_tokens",
            "maximum_provider_latency_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceLeaseBudget:
    """One top-level budget shared by proof, model, validation and I/O work.

    Zero byte/time/token values mean that the corresponding plan did not
    declare a finite limit.  Concurrency values are always positive because a
    lease that cannot execute is represented by an admission rejection.
    """

    max_parallel: int = 1
    max_cpu_proof_concurrency: int = 1
    max_model_concurrency: int = 1
    max_artifact_concurrency: int = 1
    max_processes: int = 1
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    model_token_limit: int = 0
    provider_quota: int = 0
    context_tokens: int = 0
    maximum_provider_latency_ms: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_parallel",
            "max_cpu_proof_concurrency",
            "max_model_concurrency",
            "max_artifact_concurrency",
            "max_processes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "wall_time_ms",
            "cpu_time_ms",
            "memory_bytes",
            "disk_bytes",
            "model_token_limit",
            "provider_quota",
            "context_tokens",
            "maximum_provider_latency_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def from_resource_budget(
        cls,
        value: Any,
        *,
        max_parallel: int,
        max_cpu_proof_concurrency: int = 0,
        max_model_concurrency: int = 0,
        max_artifact_concurrency: int = 0,
        maximum_provider_latency_ms: int = 0,
        context_tokens: int = 0,
    ) -> "ResourceLeaseBudget":
        """Adapt a formal ``ResourceBudget`` without importing its module."""

        parallel = max(1, int(max_parallel))

        def budget_value(name: str) -> int:
            raw = (
                value.get(name, 0)
                if isinstance(value, Mapping)
                else getattr(value, name, 0)
            )
            return _integer(raw, 0, minimum=0)

        declared_processes = budget_value("max_processes")
        process_limit = min(parallel, declared_processes) if declared_processes else parallel
        return cls(
            max_parallel=process_limit,
            max_cpu_proof_concurrency=min(
                process_limit,
                max_cpu_proof_concurrency or process_limit,
            ),
            max_model_concurrency=min(
                process_limit,
                max_model_concurrency or process_limit,
            ),
            max_artifact_concurrency=min(
                process_limit,
                max_artifact_concurrency or process_limit,
            ),
            max_processes=process_limit,
            wall_time_ms=budget_value("wall_time_ms"),
            cpu_time_ms=budget_value("cpu_time_ms"),
            memory_bytes=budget_value("memory_bytes"),
            disk_bytes=budget_value("disk_bytes"),
            model_token_limit=budget_value("model_token_limit"),
            provider_quota=budget_value("provider_quota"),
            context_tokens=max(0, int(context_tokens)),
            maximum_provider_latency_ms=max(0, int(maximum_provider_latency_ms)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceLeaseBudget":
        defaults = cls()
        names = tuple(asdict(defaults))
        return cls(
            **{
                name: _integer(
                    value.get(name),
                    getattr(defaults, name),
                    minimum=1
                    if name
                    in {
                        "max_parallel",
                        "max_cpu_proof_concurrency",
                        "max_model_concurrency",
                        "max_artifact_concurrency",
                        "max_processes",
                    }
                    else 0,
                )
                for name in names
            }
        )

    def child_limits(
        self,
        requirement: LaneResourceRequirements,
        *,
        granted_processes: int | None = None,
    ) -> ChildResourceLimits:
        """Derive nested portfolio/kernel limits from this exact lease."""

        processes = max(
            1,
            min(
                self.max_processes,
                requirement.process_slots,
                granted_processes if granted_processes is not None else requirement.process_slots,
            ),
        )
        is_solver = requirement.resource_class == ProofResourceClass.SOLVER.value
        is_kernel = requirement.resource_class == ProofResourceClass.KERNEL.value
        latency = self.maximum_provider_latency_ms
        if requirement.max_provider_latency_ms:
            latency = (
                min(latency, requirement.max_provider_latency_ms)
                if latency
                else requirement.max_provider_latency_ms
            )
        return ChildResourceLimits(
            max_processes=processes,
            portfolio_max_parallel=processes if is_solver else 1,
            kernel_max_parallel=processes if is_kernel else 1,
            wall_time_ms=self.wall_time_ms,
            cpu_time_ms=self.cpu_time_ms,
            memory_bytes=self.memory_bytes,
            disk_bytes=self.disk_bytes,
            model_token_limit=min(
                item
                for item in (self.model_token_limit, requirement.token_budget)
                if item > 0
            )
            if self.model_token_limit and requirement.token_budget
            else (self.model_token_limit or requirement.token_budget),
            provider_quota=min(
                item
                for item in (self.provider_quota, requirement.quota_units)
                if item > 0
            )
            if self.provider_quota and requirement.quota_units
            else (self.provider_quota or requirement.quota_units),
            context_tokens=requirement.context_tokens or self.context_tokens,
            maximum_provider_latency_ms=latency,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


# Descriptive alias used by integrations which name the owning layer.
SupervisorResourceLeaseBudget = ResourceLeaseBudget


@dataclass(frozen=True)
class ResourcePolicy:
    """Configured hard bounds and pre-exhaustion high-watermarks."""

    max_lanes: int = 1
    cpu_high_watermark_percent: int = 90
    memory_high_watermark_percent: int = 90
    disk_high_watermark_percent: int = 95
    minimum_memory_available_bytes: int = 0
    minimum_disk_available_bytes: int = 0
    maximum_provider_latency_ms: int = 120_000
    provider_quota_reserve: int = 0
    provider_token_reserve: int = 0
    require_provider_telemetry: bool = True
    max_cpu_proof_concurrency: int = 0
    max_model_concurrency: int = 0
    max_artifact_concurrency: int = 0
    resource_class_limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_lanes < 0:
            raise ValueError("max_lanes must be non-negative")
        for name in (
            "cpu_high_watermark_percent", "memory_high_watermark_percent",
            "disk_high_watermark_percent",
        ):
            if not 0 <= int(getattr(self, name)) <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        for name in (
            "minimum_memory_available_bytes", "minimum_disk_available_bytes",
            "maximum_provider_latency_ms", "provider_quota_reserve", "provider_token_reserve",
            "max_cpu_proof_concurrency", "max_model_concurrency",
            "max_artifact_concurrency",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        normalized_limits: dict[str, int] = {}
        for raw_name, raw_limit in (self.resource_class_limits or {}).items():
            name = normalize_resource_class(raw_name)
            if not name:
                raise ValueError("resource class limit names must be non-empty")
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit <= 0:
                raise ValueError("resource class limits must be positive integers")
            normalized_limits[name] = raw_limit
        object.__setattr__(self, "resource_class_limits", normalized_limits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourcePolicy":
        defaults = cls()
        return cls(
            max_lanes=_integer(_first(value, ("max_lanes", "worker_limit", "max_workers"), defaults.max_lanes), defaults.max_lanes, minimum=0),
            cpu_high_watermark_percent=_integer(_first(value, ("cpu_high_watermark_percent", "max_cpu_percent"), defaults.cpu_high_watermark_percent), defaults.cpu_high_watermark_percent, minimum=0),
            memory_high_watermark_percent=_integer(_first(value, ("memory_high_watermark_percent", "max_memory_percent"), defaults.memory_high_watermark_percent), defaults.memory_high_watermark_percent, minimum=0),
            disk_high_watermark_percent=_integer(_first(value, ("disk_high_watermark_percent", "max_disk_percent"), defaults.disk_high_watermark_percent), defaults.disk_high_watermark_percent, minimum=0),
            minimum_memory_available_bytes=_integer(_first(value, ("minimum_memory_available_bytes", "min_memory_available_bytes"), 0), 0, minimum=0),
            minimum_disk_available_bytes=_integer(_first(value, ("minimum_disk_available_bytes", "min_disk_available_bytes"), 0), 0, minimum=0),
            maximum_provider_latency_ms=_integer(_first(value, ("maximum_provider_latency_ms", "max_provider_latency_ms", "latency_limit_ms"), defaults.maximum_provider_latency_ms), defaults.maximum_provider_latency_ms, minimum=0),
            provider_quota_reserve=_integer(_first(value, ("provider_quota_reserve", "quota_reserve"), 0), 0, minimum=0),
            provider_token_reserve=_integer(_first(value, ("provider_token_reserve", "token_reserve"), 0), 0, minimum=0),
            require_provider_telemetry=_boolean(value.get("require_provider_telemetry"), True),
            max_cpu_proof_concurrency=_integer(
                _first(
                    value,
                    ("max_cpu_proof_concurrency", "cpu_proof_concurrency", "max_cpu_parallel"),
                    0,
                ),
                0,
                minimum=0,
            ),
            max_model_concurrency=_integer(
                _first(value, ("max_model_concurrency", "model_concurrency"), 0),
                0,
                minimum=0,
            ),
            max_artifact_concurrency=_integer(
                _first(value, ("max_artifact_concurrency", "artifact_concurrency"), 0),
                0,
                minimum=0,
            ),
            resource_class_limits=_mapping(
                _first(value, ("resource_class_limits", "resource_limits"), {})
            ),
        )


@dataclass(frozen=True)
class AdmissionDecision:
    lane_id: str
    admitted: bool
    provider_id: str = ""
    reasons: tuple[str, ...] = ()
    configured_max_lanes: int = 0
    host_available_slots: int = 0
    provider_available_slots: int = 0
    effective_slots: int = 0
    capability_fit_millionths: int = 0
    reserved_quota_units: int = 0
    reserved_tokens: int = 0
    resource_class: str = ""
    resource_pool: str = ""
    reserved_process_slots: int = 0
    reserved_memory_bytes: int = 0
    reserved_disk_bytes: int = 0

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


@dataclass(frozen=True)
class ResourceScheduleSnapshot:
    observed_at_ms: int
    host: HostResourceSnapshot
    providers: tuple[ProviderCapacity, ...]
    policy: ResourcePolicy
    decisions: tuple[AdmissionDecision, ...]
    configured_max_lanes: int
    effective_slots: int
    available_slots: int
    admitted_count: int
    backpressure_reasons: tuple[str, ...] = ()

    @property
    def admitted_lane_ids(self) -> tuple[str, ...]:
        return tuple(item.lane_id for item in self.decisions if item.admitted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at_ms": self.observed_at_ms,
            "host": self.host.to_dict(),
            "providers": [item.to_dict() for item in self.providers],
            "policy": self.policy.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "configured_max_lanes": self.configured_max_lanes,
            "effective_slots": self.effective_slots,
            "available_slots": self.available_slots,
            "admitted_count": self.admitted_count,
            "admitted_lane_ids": list(self.admitted_lane_ids),
            "backpressure_reasons": list(self.backpressure_reasons),
        }


@dataclass
class _ProviderReservation:
    requests: int = 0
    quota: int = 0
    tokens: int = 0


@dataclass(frozen=True)
class ResourceAdmissionLease:
    """A reclaimable grant from the supervisor-level resource budget."""

    lease_id: str
    requirement: LaneResourceRequirements
    decision: AdmissionDecision
    budget: ResourceLeaseBudget
    acquired_at_ms: int

    @property
    def lane_id(self) -> str:
        return self.requirement.lane_id

    @property
    def resource_class(self) -> str:
        return self.requirement.resource_class

    @property
    def resource_pool(self) -> str:
        return self.requirement.resource_pool

    @property
    def provider_id(self) -> str:
        return self.decision.provider_id

    @property
    def child_limits(self) -> ChildResourceLimits:
        return self.budget.child_limits(
            self.requirement,
            granted_processes=self.decision.reserved_process_slots,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "lane_id": self.lane_id,
            "resource_class": self.resource_class,
            "resource_pool": self.resource_pool,
            "provider_id": self.provider_id,
            "acquired_at_ms": self.acquired_at_ms,
            "requirement": self.requirement.to_dict(),
            "decision": self.decision.to_dict(),
            "budget": self.budget.to_dict(),
            "child_limits": self.child_limits.to_dict(),
        }


class ResourceScheduler:
    """Evaluate and reserve host/provider capacity for a reconciliation cycle."""

    def __init__(
        self,
        policy: ResourcePolicy | Mapping[str, Any] | None = None,
        *,
        host_sampler: Callable[..., HostResourceSnapshot] = sample_host_resources,
    ) -> None:
        self.policy = (
            policy
            if isinstance(policy, ResourcePolicy)
            else ResourcePolicy.from_mapping(policy or {})
        )
        self.host_sampler = host_sampler
        self._lease_lock = threading.RLock()
        self._leases: dict[str, ResourceAdmissionLease] = {}

    def _pool_limit(self, pool: str) -> int:
        if pool == "model":
            return self.policy.max_model_concurrency or self.policy.max_lanes
        if pool == "artifact":
            return self.policy.max_artifact_concurrency or self.policy.max_lanes
        return self.policy.max_cpu_proof_concurrency or self.policy.max_lanes

    def _host_reasons(self, host: HostResourceSnapshot, requirement: LaneResourceRequirements) -> list[str]:
        reasons: list[str] = []
        policy = self.policy
        if host.cpu_percent >= policy.cpu_high_watermark_percent:
            reasons.append("host_cpu_high_watermark")
        if host.memory_percent >= policy.memory_high_watermark_percent:
            reasons.append("host_memory_high_watermark")
        if host.disk_percent >= policy.disk_high_watermark_percent:
            reasons.append("host_disk_high_watermark")
        required_memory = max(policy.minimum_memory_available_bytes, requirement.memory_bytes)
        required_disk = max(policy.minimum_disk_available_bytes, requirement.disk_bytes)
        if required_memory and host.memory_available_bytes < required_memory:
            reasons.append("host_memory_headroom")
        if required_disk and host.disk_available_bytes < required_disk:
            reasons.append("host_disk_headroom")
        if (
            requirement.resource_class
            and host.resource_classes
            and requirement.resource_class not in host.resource_classes
        ):
            legacy_compatible = (
                requirement.resource_class in LEGACY_RESOURCE_CLASSES
                and bool(set(host.resource_classes).intersection(PROOF_RESOURCE_CLASSES))
            ) or (
                requirement.resource_class in PROOF_RESOURCE_CLASSES
                and bool(set(host.resource_classes).intersection(LEGACY_RESOURCE_CLASSES))
            )
            if not legacy_compatible:
                reasons.append("resource_class_mismatch")
        if requirement.provider_required:
            host_required = {
                item.removeprefix("host:")
                for item in requirement.required_capabilities
                if item.startswith("host:")
            }
        else:
            host_required = {
                item.removeprefix("host:")
                for item in requirement.required_capabilities
                if not item.startswith("llm:")
            }
        if host_required.difference(host.capabilities):
            reasons.append("host_capability_mismatch")
        return reasons

    def _provider_reasons(
        self,
        provider: ProviderCapacity,
        requirement: LaneResourceRequirements,
        reservation: _ProviderReservation,
    ) -> list[str]:
        policy = self.policy
        reasons: list[str] = []
        if not provider.healthy:
            reasons.append("provider_unhealthy")
        if provider.retry_after_ms > 0:
            reasons.append("provider_backoff")
        latency_limit = policy.maximum_provider_latency_ms
        if requirement.max_provider_latency_ms:
            latency_limit = min(latency_limit, requirement.max_provider_latency_ms)
        if provider.latency_ms > latency_limit:
            reasons.append("provider_latency")
        if provider.available_concurrency - reservation.requests <= 0:
            reasons.append("provider_concurrency")
        if provider.quota_remaining >= 0 and (
            provider.quota_remaining - reservation.quota - requirement.quota_units
            < policy.provider_quota_reserve
        ):
            reasons.append("provider_quota")
        if provider.context_window_tokens >= 0 and requirement.context_tokens > provider.context_window_tokens:
            reasons.append("provider_context")
        if provider.token_budget_remaining >= 0 and (
            provider.token_budget_remaining - reservation.tokens - requirement.token_budget
            < policy.provider_token_reserve
        ):
            reasons.append("provider_token_budget")
        required = {
            item.removeprefix("llm:")
            for item in requirement.required_capabilities
            if not item.startswith("host:")
        }
        if required.difference(provider.capabilities):
            reasons.append("provider_capability_mismatch")
        return reasons

    @staticmethod
    def _provider_sort_key(provider: ProviderCapacity) -> tuple[int, str]:
        # A measured zero is valid and sorts ahead of higher latency. Provider
        # identity makes otherwise equal selection deterministic.
        return (provider.latency_ms, provider.provider_id)

    def evaluate(
        self,
        requirement: LaneResourceRequirements | Mapping[str, Any],
        *,
        host: HostResourceSnapshot | Mapping[str, Any],
        providers: Mapping[str, Any] | Iterable[ProviderCapacity | Mapping[str, Any]] | None = None,
        admitted_workers: int = 0,
        reservations: Mapping[str, _ProviderReservation] | None = None,
        active_requirements: Iterable[LaneResourceRequirements] = (),
    ) -> AdmissionDecision:
        """Evaluate one lane without mutating caller-owned reservation state."""

        req = requirement if isinstance(requirement, LaneResourceRequirements) else LaneResourceRequirements.from_mapping(requirement)
        host_snapshot = host if isinstance(host, HostResourceSnapshot) else HostResourceSnapshot.from_mapping(host)
        normalized = normalize_provider_capacities(providers)
        configured = self.policy.max_lanes
        active_items = tuple(active_requirements)
        occupied_processes = sum(item.process_slots for item in active_items)
        host_slots = max(
            0,
            min(configured, host_snapshot.worker_limit, host_snapshot.active_workers + host_snapshot.available_worker_capacity)
            - host_snapshot.active_workers
            - max(0, int(admitted_workers))
            - occupied_processes,
        )
        host_reasons = self._host_reasons(host_snapshot, req)
        reserved_memory = sum(item.memory_bytes for item in active_items)
        reserved_disk = sum(item.disk_bytes for item in active_items)
        required_memory = max(
            self.policy.minimum_memory_available_bytes,
            req.memory_bytes,
        )
        required_disk = max(
            self.policy.minimum_disk_available_bytes,
            req.disk_bytes,
        )
        if (
            required_memory
            and host_snapshot.memory_available_bytes - reserved_memory < required_memory
            and "host_memory_headroom" not in host_reasons
        ):
            host_reasons.append("host_memory_headroom")
        if (
            required_disk
            and host_snapshot.disk_available_bytes - reserved_disk < required_disk
            and "host_disk_headroom" not in host_reasons
        ):
            host_reasons.append("host_disk_headroom")
        if host_slots < req.process_slots:
            host_reasons.append("host_worker_capacity")
        pool_occupied = sum(
            item.process_slots
            for item in active_items
            if item.resource_pool == req.resource_pool
        )
        if pool_occupied + req.process_slots > self._pool_limit(req.resource_pool):
            host_reasons.append(f"{req.resource_pool.replace('-', '_')}_concurrency")
        class_limit = self.policy.resource_class_limits.get(req.resource_class)
        class_occupied = sum(
            item.process_slots
            for item in active_items
            if item.resource_class == req.resource_class
        )
        if class_limit is not None and class_occupied + req.process_slots > class_limit:
            host_reasons.append("resource_class_concurrency")
        if host_reasons:
            return AdmissionDecision(
                lane_id=req.lane_id,
                admitted=False,
                reasons=tuple(dict.fromkeys(host_reasons)),
                configured_max_lanes=configured,
                host_available_slots=host_slots,
                effective_slots=0,
                resource_class=req.resource_class,
                resource_pool=req.resource_pool,
            )

        # Backwards compatibility: non-LLM lanes do not require provider
        # telemetry, even when a provider monitor is temporarily unavailable.
        if not req.provider_required:
            return AdmissionDecision(
                lane_id=req.lane_id,
                admitted=True,
                configured_max_lanes=configured,
                host_available_slots=host_slots,
                provider_available_slots=host_slots,
                effective_slots=host_slots,
                capability_fit_millionths=1_000_000,
                resource_class=req.resource_class,
                resource_pool=req.resource_pool,
                reserved_process_slots=req.process_slots,
                reserved_memory_bytes=req.memory_bytes,
                reserved_disk_bytes=req.disk_bytes,
            )

        candidates = [item for item in normalized if not req.provider_id or item.provider_id == req.provider_id]
        if not candidates:
            reason = "provider_telemetry_unavailable" if self.policy.require_provider_telemetry else "provider_unavailable"
            return AdmissionDecision(
                lane_id=req.lane_id,
                admitted=not self.policy.require_provider_telemetry,
                provider_id=req.provider_id,
                reasons=(reason,),
                configured_max_lanes=configured,
                host_available_slots=host_slots,
                provider_available_slots=0,
                effective_slots=host_slots if not self.policy.require_provider_telemetry else 0,
                capability_fit_millionths=1_000_000 if not self.policy.require_provider_telemetry else 0,
                resource_class=req.resource_class,
                resource_pool=req.resource_pool,
                reserved_process_slots=req.process_slots if not self.policy.require_provider_telemetry else 0,
                reserved_memory_bytes=req.memory_bytes if not self.policy.require_provider_telemetry else 0,
                reserved_disk_bytes=req.disk_bytes if not self.policy.require_provider_telemetry else 0,
            )

        reserved = reservations or {}
        rejected: list[tuple[ProviderCapacity, list[str]]] = []
        for provider in sorted(candidates, key=self._provider_sort_key):
            reservation = reserved.get(provider.provider_id, _ProviderReservation())
            reasons = self._provider_reasons(provider, req, reservation)
            if reasons:
                rejected.append((provider, reasons))
                continue
            provider_slots = max(0, provider.available_concurrency - reservation.requests)
            return AdmissionDecision(
                lane_id=req.lane_id,
                admitted=True,
                provider_id=provider.provider_id,
                configured_max_lanes=configured,
                host_available_slots=host_slots,
                provider_available_slots=provider_slots,
                effective_slots=min(host_slots, provider_slots),
                capability_fit_millionths=1_000_000,
                reserved_quota_units=req.quota_units,
                reserved_tokens=req.token_budget,
                resource_class=req.resource_class,
                resource_pool=req.resource_pool,
                reserved_process_slots=req.process_slots,
                reserved_memory_bytes=req.memory_bytes,
                reserved_disk_bytes=req.disk_bytes,
            )

        # Preserve all distinct constraint failures. This makes backpressure
        # explainable when multiple providers are unsuitable for different reasons.
        reasons = tuple(dict.fromkeys(reason for _provider, items in rejected for reason in items))
        selected = min((provider for provider, _items in rejected), key=self._provider_sort_key)
        return AdmissionDecision(
            lane_id=req.lane_id,
            admitted=False,
            provider_id=selected.provider_id,
            reasons=reasons or ("provider_unavailable",),
            configured_max_lanes=configured,
            host_available_slots=host_slots,
            provider_available_slots=0,
            effective_slots=0,
            resource_class=req.resource_class,
            resource_pool=req.resource_pool,
        )

    def acquire(
        self,
        requirement: LaneResourceRequirements | Mapping[str, Any],
        *,
        budget: ResourceLeaseBudget | None = None,
        host: HostResourceSnapshot | Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | Iterable[ProviderCapacity | Mapping[str, Any]] | None = None,
        path: Path | str = ".",
    ) -> tuple[AdmissionDecision, ResourceAdmissionLease | None]:
        """Atomically reserve a reclaimable local supervisor lease.

        Live host/provider telemetry remains authoritative on every attempt.
        Released leases disappear from all pool and provider reservations, so
        capacity can immediately be reused without waiting for a new sample.
        """

        req = (
            requirement
            if isinstance(requirement, LaneResourceRequirements)
            else LaneResourceRequirements.from_mapping(requirement)
        )
        lease_budget = budget or ResourceLeaseBudget.from_resource_budget(
            {},
            max_parallel=max(1, self.policy.max_lanes),
            max_cpu_proof_concurrency=self._pool_limit("cpu-proof"),
            max_model_concurrency=self._pool_limit("model"),
            max_artifact_concurrency=self._pool_limit("artifact"),
            maximum_provider_latency_ms=self.policy.maximum_provider_latency_ms,
        )
        if lease_budget.maximum_provider_latency_ms:
            effective_latency = (
                min(
                    lease_budget.maximum_provider_latency_ms,
                    req.max_provider_latency_ms,
                )
                if req.max_provider_latency_ms
                else lease_budget.maximum_provider_latency_ms
            )
            req = replace(req, max_provider_latency_ms=effective_latency)
        with self._lease_lock:
            active = tuple(item.requirement for item in self._leases.values())
            if host is None:
                host_snapshot = self.host_sampler(
                    path,
                    active_workers=0,
                    worker_limit=min(self.policy.max_lanes, lease_budget.max_processes),
                    active_phase="proof_scheduler",
                )
            elif isinstance(host, HostResourceSnapshot):
                host_snapshot = host
            else:
                host_snapshot = HostResourceSnapshot.from_mapping(host)
            provider_reservations: dict[str, _ProviderReservation] = {}
            for lease in self._leases.values():
                if not lease.provider_id:
                    continue
                reservation = provider_reservations.setdefault(
                    lease.provider_id, _ProviderReservation()
                )
                reservation.requests += 1
                reservation.quota += lease.requirement.quota_units
                reservation.tokens += lease.requirement.token_budget
            decision = self.evaluate(
                req,
                host=host_snapshot,
                providers=providers,
                reservations=provider_reservations,
                active_requirements=active,
            )
            if decision.admitted:
                pool_limit = {
                    "cpu-proof": lease_budget.max_cpu_proof_concurrency,
                    "model": lease_budget.max_model_concurrency,
                    "artifact": lease_budget.max_artifact_concurrency,
                }[req.resource_pool]
                pool_used = sum(
                    item.process_slots
                    for item in active
                    if item.resource_pool == req.resource_pool
                )
                total_used = sum(item.process_slots for item in active)
                memory_used = sum(item.memory_bytes for item in active)
                disk_used = sum(item.disk_bytes for item in active)
                tokens_used = sum(
                    item.token_budget for item in active if item.provider_required
                )
                quota_used = sum(
                    item.quota_units for item in active if item.provider_required
                )
                extra_reasons: list[str] = []
                if total_used + req.process_slots > lease_budget.max_processes:
                    extra_reasons.append("lease_process_capacity")
                if pool_used + req.process_slots > pool_limit:
                    extra_reasons.append(
                        f"lease_{req.resource_pool.replace('-', '_')}_concurrency"
                    )
                if (
                    lease_budget.memory_bytes
                    and memory_used + req.memory_bytes > lease_budget.memory_bytes
                ):
                    extra_reasons.append("lease_memory_budget")
                if (
                    lease_budget.disk_bytes
                    and disk_used + req.disk_bytes > lease_budget.disk_bytes
                ):
                    extra_reasons.append("lease_disk_budget")
                if (
                    lease_budget.model_token_limit
                    and tokens_used + req.token_budget
                    > lease_budget.model_token_limit
                ):
                    extra_reasons.append("lease_token_budget")
                if (
                    lease_budget.provider_quota
                    and req.provider_required
                    and quota_used + req.quota_units > lease_budget.provider_quota
                ):
                    extra_reasons.append("lease_provider_quota")
                if extra_reasons:
                    decision = replace(
                        decision,
                        admitted=False,
                        reasons=tuple(extra_reasons),
                        effective_slots=0,
                        reserved_process_slots=0,
                        reserved_quota_units=0,
                        reserved_tokens=0,
                        reserved_memory_bytes=0,
                        reserved_disk_bytes=0,
                    )
            if not decision.admitted:
                return decision, None
            lease = ResourceAdmissionLease(
                lease_id=f"resource-lease:{uuid.uuid4().hex}",
                requirement=req,
                decision=decision,
                budget=lease_budget,
                acquired_at_ms=host_snapshot.observed_at_ms or int(time.time() * 1000),
            )
            self._leases[lease.lease_id] = lease
            return decision, lease

    def release(self, lease: ResourceAdmissionLease | str) -> bool:
        """Release a lease and make all of its capacity immediately reusable."""

        lease_id = lease.lease_id if isinstance(lease, ResourceAdmissionLease) else str(lease)
        with self._lease_lock:
            return self._leases.pop(lease_id, None) is not None

    @property
    def active_leases(self) -> tuple[ResourceAdmissionLease, ...]:
        with self._lease_lock:
            return tuple(
                self._leases[key]
                for key in sorted(self._leases)
            )

    def release_lane(self, lane_id: str) -> int:
        """Release all local resource grants owned by ``lane_id``."""

        with self._lease_lock:
            matching = [
                lease_id
                for lease_id, lease in self._leases.items()
                if lease.lane_id == lane_id
            ]
            for lease_id in matching:
                self._leases.pop(lease_id, None)
            return len(matching)

    def schedule(
        self,
        lanes: Iterable[LaneResourceRequirements | Mapping[str, Any]],
        *,
        host: HostResourceSnapshot | Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | Iterable[ProviderCapacity | Mapping[str, Any]] | None = None,
        path: Path | str = ".",
        active_workers: int = 0,
    ) -> ResourceScheduleSnapshot:
        """Admit lanes in input priority order while reserving shared capacity."""

        requirements = tuple(
            item if isinstance(item, LaneResourceRequirements) else LaneResourceRequirements.from_mapping(item)
            for item in lanes
        )
        if host is None:
            host_snapshot = self.host_sampler(
                path,
                active_workers=active_workers,
                worker_limit=self.policy.max_lanes,
                active_phase="scheduler",
            )
        elif isinstance(host, HostResourceSnapshot):
            host_snapshot = host
        else:
            host_snapshot = HostResourceSnapshot.from_mapping(host)
        normalized = normalize_provider_capacities(providers)
        reservations: dict[str, _ProviderReservation] = {
            item.provider_id: _ProviderReservation() for item in normalized
        }
        decisions: list[AdmissionDecision] = []
        admitted = 0
        admitted_requirements: list[LaneResourceRequirements] = []
        for requirement in requirements:
            decision = self.evaluate(
                requirement,
                host=host_snapshot,
                providers=normalized,
                reservations=reservations,
                active_requirements=admitted_requirements,
            )
            decisions.append(decision)
            if not decision.admitted:
                continue
            admitted += 1
            admitted_requirements.append(requirement)
            if decision.provider_id:
                reservation = reservations.setdefault(decision.provider_id, _ProviderReservation())
                reservation.requests += 1
                reservation.quota += requirement.quota_units
                reservation.tokens += requirement.token_budget

        configured = self.policy.max_lanes
        total_host_capacity = min(
            configured,
            host_snapshot.worker_limit,
            host_snapshot.active_workers + host_snapshot.available_worker_capacity,
        )
        host_blocked = bool(self._host_reasons(host_snapshot, LaneResourceRequirements()))
        host_free = max(0, total_host_capacity - host_snapshot.active_workers)
        effective = 0 if host_blocked else host_free
        # If every candidate needs an LLM, provider capacity is a pool-wide
        # upper bound.  Mixed or legacy work can still fill remaining host
        # slots without consuming a provider request.
        if requirements and all(item.provider_required for item in requirements):
            provider_free = 0
            for provider in normalized:
                if (
                    provider.healthy
                    and provider.retry_after_ms == 0
                    and provider.latency_ms <= self.policy.maximum_provider_latency_ms
                    and provider.quota_remaining != 0
                    and provider.token_budget_remaining != 0
                ):
                    provider_free += provider.available_concurrency
            effective = min(effective, provider_free)
        available = max(0, effective - admitted)
        backpressure = tuple(
            dict.fromkeys(reason for decision in decisions if not decision.admitted for reason in decision.reasons)
        )
        return ResourceScheduleSnapshot(
            observed_at_ms=host_snapshot.observed_at_ms or int(time.time() * 1000),
            host=host_snapshot,
            providers=normalized,
            policy=self.policy,
            decisions=tuple(decisions),
            configured_max_lanes=configured,
            effective_slots=effective,
            available_slots=available,
            admitted_count=admitted,
            backpressure_reasons=backpressure,
        )

    # Descriptive aliases used by scheduler integrations and callers.
    evaluate_lane = evaluate
    schedule_lanes = schedule


class ProofWorkStatus(str, Enum):
    """Observable terminal states of one goal-runtime work request."""

    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"
    CANCELLED = "cancelled"
    BACKPRESSURED = "backpressured"
    FAILED = "failed"


class ProofWorkCancellationToken:
    """Thread-safe cooperative cancellation with a stable first reason."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._cancelled_at_ms = 0

    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation; the first request wins and is retained."""

        normalized = str(reason or "cancelled").strip() or "cancelled"
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = normalized
            self._cancelled_at_ms = int(time.time() * 1000)
            self._event.set()
            return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def cancelled_at_ms(self) -> int:
        with self._lock:
            return self._cancelled_at_ms

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancellation, matching :class:`threading.Event`."""

        return self._event.wait(timeout)

    def is_cancelled(self) -> bool:
        """Common token spelling used by verifier and subprocess adapters."""

        return self.cancelled


@dataclass(frozen=True)
class ProofWorkRequest:
    """A bounded unit of model, type-check, solver, or kernel work."""

    work_id: str
    work_kind: ProofWorkKind
    provider_id: str = ""
    required_capabilities: tuple[str, ...] = ()
    context_tokens: int = 0
    token_budget: int = 0
    quota_units: int = 1
    memory_bytes: int = 0
    disk_bytes: int = 0
    max_provider_latency_ms: int = 0
    process_slots: int = 1
    max_queue_wait_ms: int = 0

    def __post_init__(self) -> None:
        work_id = str(self.work_id or "").strip()
        if not work_id:
            raise ValueError("work_id must be non-empty")
        if "\x00" in work_id:
            raise ValueError("work_id must not contain NUL bytes")
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(
            self, "work_kind", normalize_proof_work_kind(self.work_kind)
        )
        object.__setattr__(
            self, "provider_id", str(self.provider_id or "").strip().lower()
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _strings(self.required_capabilities),
        )
        for name in (
            "context_tokens",
            "token_budget",
            "quota_units",
            "memory_bytes",
            "disk_bytes",
            "max_provider_latency_ms",
            "max_queue_wait_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.process_slots, bool)
            or not isinstance(self.process_slots, int)
            or self.process_slots <= 0
        ):
            raise ValueError("process_slots must be a positive integer")

    @property
    def resource_class(self) -> str:
        return resource_class_for_work_kind(self.work_kind)

    @property
    def requires_provider(self) -> bool:
        return self.work_kind is ProofWorkKind.MODEL_DRAFT

    def to_requirement(self) -> LaneResourceRequirements:
        capabilities = self.required_capabilities
        if self.requires_provider:
            capabilities = tuple(
                f"llm:{item.removeprefix('llm:')}"
                if not item.startswith("host:")
                else item
                for item in capabilities
            )
        return LaneResourceRequirements(
            lane_id=self.work_id,
            resource_class=self.resource_class,
            required_capabilities=capabilities,
            provider_id=self.provider_id,
            requires_provider=self.requires_provider,
            context_tokens=self.context_tokens,
            token_budget=self.token_budget,
            quota_units=self.quota_units,
            memory_bytes=self.memory_bytes,
            disk_bytes=self.disk_bytes,
            max_provider_latency_ms=self.max_provider_latency_ms,
            process_slots=self.process_slots,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "work_kind": self.work_kind.value,
            "resource_class": self.resource_class,
            "provider_id": self.provider_id,
            "required_capabilities": list(self.required_capabilities),
            "context_tokens": self.context_tokens,
            "token_budget": self.token_budget,
            "quota_units": self.quota_units,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "max_provider_latency_ms": self.max_provider_latency_ms,
            "process_slots": self.process_slots,
            "max_queue_wait_ms": self.max_queue_wait_ms,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProofWorkRequest":
        return cls(
            work_id=str(
                _first(value, ("work_id", "task_id", "lane_id", "request_id"), "")
                or ""
            ),
            work_kind=normalize_proof_work_kind(
                _first(value, ("work_kind", "kind", "stage"), "")
            ),
            provider_id=str(
                _first(value, ("provider_id", "route_id", "provider"), "") or ""
            ),
            required_capabilities=_strings(
                _first(value, ("required_capabilities", "capabilities"), ())
            ),
            context_tokens=_integer(
                _first(value, ("context_tokens", "required_context_tokens"), 0),
                0,
                minimum=0,
            ),
            token_budget=_integer(
                _first(value, ("token_budget", "max_new_tokens"), 0),
                0,
                minimum=0,
            ),
            quota_units=_integer(value.get("quota_units"), 1, minimum=0),
            memory_bytes=_integer(value.get("memory_bytes"), 0, minimum=0),
            disk_bytes=_integer(value.get("disk_bytes"), 0, minimum=0),
            max_provider_latency_ms=_integer(
                _first(
                    value,
                    ("max_provider_latency_ms", "maximum_provider_latency_ms"),
                    0,
                ),
                0,
                minimum=0,
            ),
            process_slots=_integer(
                _first(value, ("process_slots", "portfolio_width"), 1),
                1,
                minimum=1,
            ),
            max_queue_wait_ms=_integer(
                _first(value, ("max_queue_wait_ms", "queue_timeout_ms"), 0),
                0,
                minimum=0,
            ),
        )


@dataclass(frozen=True)
class ProofWorkContext:
    """Execution context shared with primary and deterministic fallback work."""

    request: ProofWorkRequest
    cancellation_token: ProofWorkCancellationToken
    admission: AdmissionDecision | None = None
    lease: ResourceAdmissionLease | None = None
    fallback_reason: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def child_limits(self) -> ChildResourceLimits | None:
        return self.lease.child_limits if self.lease is not None else None


@dataclass(frozen=True)
class ProofWorkResult:
    """Structured outcome that never confuses fallback with primary success."""

    request: ProofWorkRequest
    status: ProofWorkStatus
    queued_at_ms: int
    started_at_ms: int
    completed_at_ms: int
    value: Any = None
    admission: AdmissionDecision | None = None
    used_fallback: bool = False
    fallback_reason: str = ""
    reason_codes: tuple[str, ...] = ()
    error: str = ""

    @property
    def successful(self) -> bool:
        """Whether the request returned a usable primary or fallback value.

        This property retains the scheduler's compatibility semantics.  It is
        not a proof-completion signal; trust-sensitive callers must use
        :attr:`primary_succeeded`.
        """

        return self.status in (ProofWorkStatus.SUCCEEDED, ProofWorkStatus.FALLBACK)

    @property
    def primary_succeeded(self) -> bool:
        """Whether the resource-leased primary operation itself succeeded."""

        return self.status is ProofWorkStatus.SUCCEEDED

    @property
    def fallback_succeeded(self) -> bool:
        """Whether a bounded deterministic fallback produced the value."""

        return self.status is ProofWorkStatus.FALLBACK and self.used_fallback

    @property
    def queue_latency_ms(self) -> int:
        boundary = self.started_at_ms or self.completed_at_ms
        return max(0, boundary - self.queued_at_ms)

    @property
    def execution_latency_ms(self) -> int:
        if not self.started_at_ms:
            return 0
        return max(0, self.completed_at_ms - self.started_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status.value,
            "successful": self.successful,
            "primary_succeeded": self.primary_succeeded,
            "fallback_succeeded": self.fallback_succeeded,
            "queued_at_ms": self.queued_at_ms,
            "started_at_ms": self.started_at_ms,
            "completed_at_ms": self.completed_at_ms,
            "queue_latency_ms": self.queue_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "value": self.value,
            "admission": self.admission.to_dict() if self.admission else None,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "reason_codes": list(self.reason_codes),
            "error": self.error,
        }


ProofWorkCallable = Callable[[ProofWorkContext], Any]


class GoalRuntimeResourceScheduler:
    """Bounded execution facade for route-aware goal-development proof work.

    Execution is synchronous in the caller's thread, making ownership and
    shutdown explicit.  Multiple callers may enter concurrently.  Waiting
    requests are FIFO within each resource class, while different classes can
    progress independently.  The underlying :class:`ResourceScheduler`
    atomically accounts all running work.
    """

    def __init__(
        self,
        resource_scheduler: ResourceScheduler | None = None,
        *,
        policy: ResourcePolicy | Mapping[str, Any] | None = None,
        budget: ResourceLeaseBudget | None = None,
        max_queued_tasks: int = 64,
        max_fallback_concurrency: int = 1,
        queue_retry_ms: int = 25,
        host_resource_source: (
            HostResourceSnapshot
            | Mapping[str, Any]
            | Callable[[], HostResourceSnapshot | Mapping[str, Any]]
            | None
        ) = None,
        provider_capacity_source: (
            Mapping[str, Any]
            | Iterable[ProviderCapacity | Mapping[str, Any]]
            | Callable[
                [],
                Mapping[str, Any]
                | Iterable[ProviderCapacity | Mapping[str, Any]]
                | None,
            ]
            | None
        ) = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if resource_scheduler is not None and policy is not None:
            raise ValueError("pass resource_scheduler or policy, not both")
        if (
            isinstance(max_queued_tasks, bool)
            or not isinstance(max_queued_tasks, int)
            or max_queued_tasks <= 0
        ):
            raise ValueError("max_queued_tasks must be a positive integer")
        if (
            isinstance(queue_retry_ms, bool)
            or not isinstance(queue_retry_ms, int)
            or queue_retry_ms <= 0
        ):
            raise ValueError("queue_retry_ms must be a positive integer")
        if (
            isinstance(max_fallback_concurrency, bool)
            or not isinstance(max_fallback_concurrency, int)
            or max_fallback_concurrency <= 0
        ):
            raise ValueError(
                "max_fallback_concurrency must be a positive integer"
            )
        self.resource_scheduler = resource_scheduler or ResourceScheduler(policy)
        self.budget = budget
        self.max_queued_tasks = max_queued_tasks
        self.max_fallback_concurrency = max_fallback_concurrency
        self.queue_retry_ms = queue_retry_ms
        self.host_resource_source = host_resource_source
        self.provider_capacity_source = provider_capacity_source
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._state_lock = threading.RLock()
        self._queues: dict[str, deque[str]] = {}
        self._tokens: dict[str, ProofWorkCancellationToken] = {}
        self._running: set[str] = set()
        self._fallback_running: set[str] = set()

    @staticmethod
    def _sample(source: Any) -> Any:
        return source() if callable(source) else source

    @property
    def queued_count(self) -> int:
        with self._state_lock:
            return sum(len(queue) for queue in self._queues.values())

    @property
    def running_count(self) -> int:
        with self._state_lock:
            return len(self._running)

    @property
    def fallback_running_count(self) -> int:
        with self._state_lock:
            return len(self._fallback_running)

    @property
    def queued_work_ids(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(
                work_id
                for resource_class in sorted(self._queues)
                for work_id in self._queues[resource_class]
            )

    @property
    def running_work_ids(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(sorted(self._running))

    def cancel(self, work_id: str, reason: str = "operator_cancelled") -> bool:
        """Cooperatively cancel queued or running work by stable identity."""

        with self._state_lock:
            token = self._tokens.get(str(work_id))
        return token.cancel(reason) if token is not None else False

    def _enqueue(
        self,
        request: ProofWorkRequest,
        token: ProofWorkCancellationToken,
    ) -> bool:
        with self._state_lock:
            if request.work_id in self._tokens:
                raise ValueError(f"work_id is already scheduled: {request.work_id}")
            queued = sum(len(queue) for queue in self._queues.values())
            if queued >= self.max_queued_tasks:
                return False
            self._tokens[request.work_id] = token
            self._queues.setdefault(request.resource_class, deque()).append(
                request.work_id
            )
            return True

    def _is_head(self, request: ProofWorkRequest) -> bool:
        with self._state_lock:
            queue = self._queues.get(request.resource_class)
            return bool(queue and queue[0] == request.work_id)

    def _mark_running(self, request: ProofWorkRequest) -> None:
        with self._state_lock:
            queue = self._queues.get(request.resource_class)
            if not queue or queue[0] != request.work_id:
                raise RuntimeError("resource queue ownership changed unexpectedly")
            queue.popleft()
            if not queue:
                self._queues.pop(request.resource_class, None)
            self._running.add(request.work_id)

    def _forget(self, request: ProofWorkRequest) -> None:
        with self._state_lock:
            queue = self._queues.get(request.resource_class)
            if queue:
                try:
                    queue.remove(request.work_id)
                except ValueError:
                    pass
                if not queue:
                    self._queues.pop(request.resource_class, None)
            self._running.discard(request.work_id)
            self._fallback_running.discard(request.work_id)
            self._tokens.pop(request.work_id, None)

    def _begin_fallback(
        self,
        request: ProofWorkRequest,
        token: ProofWorkCancellationToken,
    ) -> bool:
        """Atomically move a queued/primary request into bounded fallback."""

        with self._state_lock:
            existing = self._tokens.get(request.work_id)
            if existing is not None and existing is not token:
                raise ValueError(
                    f"work_id is already scheduled: {request.work_id}"
                )
            queue = self._queues.get(request.resource_class)
            if queue:
                try:
                    queue.remove(request.work_id)
                except ValueError:
                    pass
                if not queue:
                    self._queues.pop(request.resource_class, None)
            if len(self._fallback_running) >= self.max_fallback_concurrency:
                return False
            self._tokens[request.work_id] = token
            self._running.add(request.work_id)
            self._fallback_running.add(request.work_id)
            return True

    def _result(
        self,
        request: ProofWorkRequest,
        status: ProofWorkStatus,
        *,
        queued_at_ms: int,
        started_at_ms: int = 0,
        value: Any = None,
        admission: AdmissionDecision | None = None,
        used_fallback: bool = False,
        fallback_reason: str = "",
        reason_codes: Iterable[str] = (),
        error: str = "",
    ) -> ProofWorkResult:
        return ProofWorkResult(
            request=request,
            status=status,
            queued_at_ms=queued_at_ms,
            started_at_ms=started_at_ms,
            completed_at_ms=max(queued_at_ms, self._clock_ms()),
            value=value,
            admission=admission,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            reason_codes=tuple(dict.fromkeys(str(item) for item in reason_codes)),
            error=str(error)[:4_096],
        )

    def _run_fallback(
        self,
        request: ProofWorkRequest,
        token: ProofWorkCancellationToken,
        fallback: ProofWorkCallable | None,
        *,
        queued_at_ms: int,
        reason_codes: tuple[str, ...],
        admission: AdmissionDecision | None = None,
        primary_error: str = "",
    ) -> ProofWorkResult:
        if token.cancelled:
            self._forget(request)
            return self._result(
                request,
                ProofWorkStatus.CANCELLED,
                queued_at_ms=queued_at_ms,
                admission=admission,
                fallback_reason=token.reason,
                reason_codes=("cancelled",),
            )
        fallback_reason = reason_codes[0] if reason_codes else "primary_failed"
        if fallback is None:
            self._forget(request)
            status = (
                ProofWorkStatus.FAILED
                if primary_error
                else ProofWorkStatus.BACKPRESSURED
            )
            return self._result(
                request,
                status,
                queued_at_ms=queued_at_ms,
                admission=admission,
                fallback_reason=fallback_reason,
                reason_codes=reason_codes,
                error=primary_error,
            )
        if not self._begin_fallback(request, token):
            self._forget(request)
            return self._result(
                request,
                ProofWorkStatus.BACKPRESSURED,
                queued_at_ms=queued_at_ms,
                admission=admission,
                fallback_reason=fallback_reason,
                reason_codes=reason_codes + ("fallback_capacity",),
                error=primary_error,
            )
        started_at_ms = max(queued_at_ms, self._clock_ms())
        context = ProofWorkContext(
            request=request,
            cancellation_token=token,
            admission=admission,
            fallback_reason=fallback_reason,
            reason_codes=reason_codes,
        )
        try:
            try:
                value = fallback(context)
            except Exception as exc:
                result = self._result(
                    request,
                    ProofWorkStatus.FAILED,
                    queued_at_ms=queued_at_ms,
                    started_at_ms=started_at_ms,
                    admission=admission,
                    fallback_reason=fallback_reason,
                    reason_codes=reason_codes + ("fallback_failed",),
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                if token.cancelled:
                    result = self._result(
                        request,
                        ProofWorkStatus.CANCELLED,
                        queued_at_ms=queued_at_ms,
                        started_at_ms=started_at_ms,
                        admission=admission,
                        fallback_reason=token.reason,
                        reason_codes=("cancelled",),
                    )
                else:
                    result = self._result(
                        request,
                        ProofWorkStatus.FALLBACK,
                        queued_at_ms=queued_at_ms,
                        started_at_ms=started_at_ms,
                        value=value,
                        admission=admission,
                        used_fallback=True,
                        fallback_reason=fallback_reason,
                        reason_codes=reason_codes,
                        error=primary_error,
                    )
        finally:
            self._forget(request)
        return result

    def execute(
        self,
        request: ProofWorkRequest | Mapping[str, Any],
        operation: ProofWorkCallable,
        *,
        fallback: ProofWorkCallable | None = None,
        cancellation_token: ProofWorkCancellationToken | None = None,
    ) -> ProofWorkResult:
        """Execute one request under bounded admission and deterministic fallback.

        ``max_queue_wait_ms=0`` performs exactly one admission attempt.  A
        positive limit retries transient live-capacity failures until the
        bound expires.  Cancellation never invokes fallback.
        """

        if not callable(operation):
            raise TypeError("operation must be callable")
        if fallback is not None and not callable(fallback):
            raise TypeError("fallback must be callable")
        work = (
            request
            if isinstance(request, ProofWorkRequest)
            else ProofWorkRequest.from_mapping(request)
        )
        token = cancellation_token or ProofWorkCancellationToken()
        queued_at_ms = self._clock_ms()
        if token.cancelled:
            return self._result(
                work,
                ProofWorkStatus.CANCELLED,
                queued_at_ms=queued_at_ms,
                fallback_reason=token.reason,
                reason_codes=("cancelled",),
            )
        if not self._enqueue(work, token):
            return self._run_fallback(
                work,
                token,
                fallback,
                queued_at_ms=queued_at_ms,
                reason_codes=("queue_capacity",),
            )

        deadline_ms = queued_at_ms + work.max_queue_wait_ms
        last_decision: AdmissionDecision | None = None
        while True:
            if token.cancelled:
                self._forget(work)
                return self._result(
                    work,
                    ProofWorkStatus.CANCELLED,
                    queued_at_ms=queued_at_ms,
                    admission=last_decision,
                    fallback_reason=token.reason,
                    reason_codes=("cancelled",),
                )
            now_ms = self._clock_ms()
            if self._is_head(work):
                try:
                    host = self._sample(self.host_resource_source)
                    providers = self._sample(self.provider_capacity_source)
                    last_decision, lease = self.resource_scheduler.acquire(
                        work.to_requirement(),
                        budget=self.budget,
                        host=host,
                        providers=providers,
                    )
                except Exception as exc:
                    return self._run_fallback(
                        work,
                        token,
                        fallback,
                        queued_at_ms=queued_at_ms,
                        reason_codes=("resource_telemetry_error",),
                        primary_error=f"{type(exc).__name__}: {exc}",
                    )
                if lease is not None:
                    self._mark_running(work)
                    started_at_ms = max(queued_at_ms, self._clock_ms())
                    context = ProofWorkContext(
                        request=work,
                        cancellation_token=token,
                        admission=last_decision,
                        lease=lease,
                    )
                    try:
                        value = operation(context)
                    except Exception as exc:
                        primary_error = f"{type(exc).__name__}: {exc}"
                        self.resource_scheduler.release(lease)
                        return self._run_fallback(
                            work,
                            token,
                            fallback,
                            queued_at_ms=queued_at_ms,
                            reason_codes=("primary_execution_failed",),
                            admission=last_decision,
                            primary_error=primary_error,
                        )
                    finally:
                        # ``release`` is idempotent for the exception path.
                        self.resource_scheduler.release(lease)
                    self._forget(work)
                    if token.cancelled:
                        return self._result(
                            work,
                            ProofWorkStatus.CANCELLED,
                            queued_at_ms=queued_at_ms,
                            started_at_ms=started_at_ms,
                            admission=last_decision,
                            fallback_reason=token.reason,
                            reason_codes=("cancelled",),
                        )
                    return self._result(
                        work,
                        ProofWorkStatus.SUCCEEDED,
                        queued_at_ms=queued_at_ms,
                        started_at_ms=started_at_ms,
                        value=value,
                        admission=last_decision,
                    )

            if work.max_queue_wait_ms == 0 or now_ms >= deadline_ms:
                reasons = (
                    last_decision.reasons
                    if last_decision is not None and last_decision.reasons
                    else ("queue_wait_timeout",)
                )
                return self._run_fallback(
                    work,
                    token,
                    fallback,
                    queued_at_ms=queued_at_ms,
                    reason_codes=tuple(reasons),
                    admission=last_decision,
                )
            remaining_ms = max(0, deadline_ms - now_ms)
            token.wait(min(self.queue_retry_ms, remaining_ms) / 1000)

    # Compatibility-friendly verb used by orchestration callers.
    run = execute
    execute_work = execute


# Descriptive aliases used by goal-development integrations.
RouteAwareResourceScheduler = GoalRuntimeResourceScheduler
FormalVerificationResourceScheduler = GoalRuntimeResourceScheduler
ResourceCancellationToken = ProofWorkCancellationToken
ScheduledProofWorkRequest = ProofWorkRequest
ScheduledProofWorkResult = ProofWorkResult


__all__ = [
    "AdmissionDecision",
    "ChildResourceLimits",
    "DEFAULT_RESOURCE_CLASSES",
    "FormalVerificationResourceScheduler",
    "GoalRuntimeResourceScheduler",
    "HostResourceSnapshot",
    "LaneResourceRequirements",
    "LEGACY_RESOURCE_CLASSES",
    "PROOF_RESOURCE_CLASSES",
    "ProofWorkCancellationToken",
    "ProofWorkContext",
    "ProofWorkKind",
    "ProofWorkRequest",
    "ProofWorkResult",
    "ProofWorkStatus",
    "ProofResourceClass",
    "ProviderCapacity",
    "ResourceCancellationToken",
    "ResourceAdmissionLease",
    "ResourceLeaseBudget",
    "ResourcePolicy",
    "ResourceScheduleSnapshot",
    "ResourceScheduler",
    "RouteAwareResourceScheduler",
    "ScheduledProofWorkRequest",
    "ScheduledProofWorkResult",
    "SupervisorResourceLeaseBudget",
    "normalize_provider_capacities",
    "normalize_provider_capacity",
    "normalize_proof_work_kind",
    "normalize_resource_class",
    "resource_class_for_work_kind",
    "resource_pool",
    "sample_host_resources",
]
