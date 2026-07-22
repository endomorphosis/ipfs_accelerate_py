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
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


UNKNOWN_LIMIT = -1
DEFAULT_RESOURCE_CLASSES = ("cpu-small", "cpu-medium", "cpu-large")


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

    def __post_init__(self) -> None:
        for name in ("context_tokens", "token_budget", "quota_units", "memory_bytes", "disk_bytes", "max_provider_latency_ms"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def provider_required(self) -> bool:
        return bool(
            self.requires_provider
            or self.provider_id
            or self.context_tokens
            or self.token_budget
            or any(item.startswith("llm:") for item in self.required_capabilities)
        )

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
        )


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
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

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
        if requirement.resource_class and host.resource_classes and requirement.resource_class not in host.resource_classes:
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
    ) -> AdmissionDecision:
        """Evaluate one lane without mutating caller-owned reservation state."""

        req = requirement if isinstance(requirement, LaneResourceRequirements) else LaneResourceRequirements.from_mapping(requirement)
        host_snapshot = host if isinstance(host, HostResourceSnapshot) else HostResourceSnapshot.from_mapping(host)
        normalized = normalize_provider_capacities(providers)
        configured = self.policy.max_lanes
        host_slots = max(
            0,
            min(configured, host_snapshot.worker_limit, host_snapshot.active_workers + host_snapshot.available_worker_capacity)
            - host_snapshot.active_workers
            - max(0, int(admitted_workers)),
        )
        host_reasons = self._host_reasons(host_snapshot, req)
        if host_slots <= 0:
            host_reasons.append("host_worker_capacity")
        if host_reasons:
            return AdmissionDecision(
                lane_id=req.lane_id,
                admitted=False,
                reasons=tuple(dict.fromkeys(host_reasons)),
                configured_max_lanes=configured,
                host_available_slots=host_slots,
                effective_slots=0,
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
        )

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
        for requirement in requirements:
            decision = self.evaluate(
                requirement,
                host=host_snapshot,
                providers=normalized,
                admitted_workers=admitted,
                reservations=reservations,
            )
            decisions.append(decision)
            if not decision.admitted:
                continue
            admitted += 1
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


__all__ = [
    "AdmissionDecision",
    "HostResourceSnapshot",
    "LaneResourceRequirements",
    "ProviderCapacity",
    "ResourcePolicy",
    "ResourceScheduleSnapshot",
    "ResourceScheduler",
    "normalize_provider_capacities",
    "normalize_provider_capacity",
    "sample_host_resources",
]
