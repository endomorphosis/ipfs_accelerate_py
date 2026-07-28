"""Endpoint-usage projection into fair provider batch admission (ASI-167).

Physical batches reserve endpoint capacity once, settle shared overhead once,
and attribute exact units to members.  Member cancellation or deadline cannot
kill or charge siblings.  Weighted fair queues and per-tenant/goal/task/lane
reserves prevent one scope from consuming an entire shared account window.

This declared module extends the ASREF-landed runtime implementation under
``runtime.provider_batch_scheduler``.  Off mode preserves existing ordering,
batch behavior, and capacity semantics.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import importlib.util
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import provider_batch_scheduler as _runtime
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import *  # noqa: F403
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchAdmissionGrant,
    ProviderBatchCapacity,
    ProviderBatchKey,
    ProviderBatchRequest,
    ProviderBatchResult,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    ProviderCapacity,
)


def _load_declared_resource_scheduler() -> Any:
    """Load the declared ASI-167 resource_scheduler file (ASREF aliases the package name)."""

    module_name = "ipfs_accelerate_py.agent_supervisor._declared_resource_scheduler"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name("resource_scheduler.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load declared resource_scheduler from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_usage_rs = _load_declared_resource_scheduler()
ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID = _usage_rs.ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID
USAGE_CAPACITY_UNAVAILABLE = _usage_rs.USAGE_CAPACITY_UNAVAILABLE
EndpointCapacityProjection = _usage_rs.EndpointCapacityProjection
FairQueueScope = _usage_rs.FairQueueScope
HierarchicalBudgetView = _usage_rs.HierarchicalBudgetView
ResetEventCursor = _usage_rs.ResetEventCursor
SingleFlightRefresh = _usage_rs.SingleFlightRefresh
UsageAdmissionMode = _usage_rs.UsageAdmissionMode
UnknownStalePolicy = _usage_rs.UnknownStalePolicy
WeightedFairQueue = _usage_rs.WeightedFairQueue
evaluate_usage_aware_admission = _usage_rs.evaluate_usage_aware_admission
intersect_with_ancestor_budgets = _usage_rs.intersect_with_ancestor_budgets
project_provider_capacity_from_usage_snapshot = (
    _usage_rs.project_provider_capacity_from_usage_snapshot
)

# ---------------------------------------------------------------------------
# Requirement identities
# ---------------------------------------------------------------------------

ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID: str = (
    "requirement:endpoint-usage-fair-batch-admission.v1"
)
ENDPOINT_USAGE_BATCH_ADMISSION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/endpoint-usage-batch-admission@1"
)
PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID: str = (
    "requirement:physical-batch-reserve-once.v1"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _as_mode(value: Any) -> UsageAdmissionMode:
    if isinstance(value, UsageAdmissionMode):
        return value
    text = str(value or UsageAdmissionMode.OFF.value).strip().lower()
    return UsageAdmissionMode(text)


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


# ---------------------------------------------------------------------------
# Batch capacity projection
# ---------------------------------------------------------------------------


def project_batch_capacity_from_usage_snapshot(
    snapshot: Any,
    *,
    provider_id: str,
    base: ProviderBatchCapacity | Mapping[str, Any] | None = None,
    mode: UsageAdmissionMode | str = UsageAdmissionMode.ENFORCE,
    unknown_policy: UnknownStalePolicy | str = UnknownStalePolicy.FAIL_CLOSED,
    now_ms: int | None = None,
) -> tuple[ProviderBatchCapacity, EndpointCapacityProjection]:
    """Project endpoint snapshot into ProviderBatchCapacity via ProviderCapacity."""

    identity = str(provider_id or "").strip()
    if not identity:
        raise ValueError("provider_id must be non-empty")

    base_batch = (
        base
        if isinstance(base, ProviderBatchCapacity)
        else ProviderBatchCapacity.from_value(identity, base)
    )
    # Map batch capacity into a request-shaped ProviderCapacity for projection.
    base_provider = {
        "provider_id": identity,
        "healthy": base_batch.healthy,
        "quota_remaining": -1,
        "token_budget_remaining": base_batch.token_budget_remaining,
        "max_concurrency": base_batch.max_concurrent_batches,
        "active_requests": max(
            0,
            base_batch.max_concurrent_batches - base_batch.available_concurrent_batches,
        ),
        "retry_after_ms": base_batch.retry_after_ms,
    }
    projection = project_provider_capacity_from_usage_snapshot(
        snapshot,
        provider_id=identity,
        base=base_provider,
        mode=mode,
        unknown_policy=unknown_policy,
        now_ms=now_ms,
    )
    cap = projection.capacity
    available = max(0, cap.max_concurrency - cap.active_requests)
    batch_cap = ProviderBatchCapacity(
        provider_id=identity,
        healthy=cap.healthy,
        max_batch_size=base_batch.max_batch_size,
        max_concurrent_batches=max(0, cap.max_concurrency),
        available_concurrent_batches=available if cap.healthy else 0,
        token_budget_remaining=cap.token_budget_remaining,
        retry_after_ms=cap.retry_after_ms,
    )
    return batch_cap, projection


# ---------------------------------------------------------------------------
# Physical batch reservation — reserve once, attribute members exactly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchMemberAttribution:
    """Exact usage attribution for one batch member."""

    request_id: str
    token_budget: int = 0
    quota_units: int = 1
    overhead_share_micros: int = 0  # millionths of shared overhead assigned
    cancelled: bool = False
    charged: bool = True
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_budget": self.token_budget,
            "quota_units": self.quota_units,
            "overhead_share_micros": self.overhead_share_micros,
            "cancelled": self.cancelled,
            "charged": self.charged,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class PhysicalBatchReservation:
    """One atomic reservation for a physical provider batch.

    * Capacity is reserved once for the whole batch.
    * Shared overhead settles once (not per member).
    * Members receive exact attribution.
    * Member cancel/deadline does not release or re-charge siblings.
    """

    reservation_id: str
    provider_id: str
    batch_id: str
    member_request_ids: tuple[str, ...]
    total_token_budget: int
    shared_overhead_tokens: int
    shared_overhead_settled: bool = False
    member_attributions: dict[str, BatchMemberAttribution] = field(default_factory=dict)
    released: bool = False
    mode: str = UsageAdmissionMode.OFF.value
    usage_revision: str = ""
    scope_id: str = ""
    projection_id: str = ""
    reason_codes: tuple[str, ...] = ()
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def attribute_member(
        self,
        request_id: str,
        *,
        token_budget: int,
        cancelled: bool = False,
        charged: bool | None = None,
        reason_codes: Sequence[str] = (),
    ) -> BatchMemberAttribution:
        rid = str(request_id or "").strip()
        if rid not in self.member_request_ids:
            raise KeyError(f"request_id not in batch: {rid}")
        # Cancelled before dispatch: do not charge that member; siblings unchanged.
        charge = (not cancelled) if charged is None else bool(charged)
        members = [mid for mid in self.member_request_ids]
        n = max(1, len(members))
        # Equal split of overhead micros across members that remain chargeable.
        share = 1_000_000 // n
        attr = BatchMemberAttribution(
            request_id=rid,
            token_budget=max(0, int(token_budget)),
            quota_units=0 if not charge else 1,
            overhead_share_micros=share if charge else 0,
            cancelled=bool(cancelled),
            charged=charge,
            reason_codes=tuple(str(c) for c in reason_codes if str(c)),
        )
        with self._lock:
            self.member_attributions[rid] = attr
        return attr

    def settle_shared_overhead_once(self) -> int:
        """Settle shared overhead tokens exactly once for the physical batch."""

        with self._lock:
            if self.shared_overhead_settled or self.released:
                return 0
            self.shared_overhead_settled = True
            return max(0, int(self.shared_overhead_tokens))

    def cancel_member(self, request_id: str) -> BatchMemberAttribution:
        """Cancel one member without affecting sibling charges or the batch lease."""

        rid = str(request_id or "").strip()
        with self._lock:
            existing = self.member_attributions.get(rid)
            token_budget = existing.token_budget if existing is not None else 0
        return self.attribute_member(
            rid,
            token_budget=token_budget,
            cancelled=True,
            charged=False,
            reason_codes=("member_cancelled", "sibling_isolated"),
        )

    def release(self) -> None:
        with self._lock:
            self.released = True

    def total_charged_tokens(self) -> int:
        with self._lock:
            member_total = sum(
                attr.token_budget for attr in self.member_attributions.values() if attr.charged
            )
            overhead = self.shared_overhead_tokens if self.shared_overhead_settled else 0
            return member_total + overhead

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            attributions = {
                key: value.to_dict()
                for key, value in sorted(self.member_attributions.items())
            }
            return {
                "schema": ENDPOINT_USAGE_BATCH_ADMISSION_SCHEMA,
                "requirement_id": ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID,
                "physical_batch_requirement_id": PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID,
                "reservation_id": self.reservation_id,
                "provider_id": self.provider_id,
                "batch_id": self.batch_id,
                "member_request_ids": list(self.member_request_ids),
                "total_token_budget": self.total_token_budget,
                "shared_overhead_tokens": self.shared_overhead_tokens,
                "shared_overhead_settled": self.shared_overhead_settled,
                "member_attributions": attributions,
                "released": self.released,
                "mode": self.mode,
                "usage_revision": self.usage_revision,
                "scope_id": self.scope_id,
                "projection_id": self.projection_id,
                "reason_codes": list(self.reason_codes),
                "total_charged_tokens": self.total_charged_tokens(),
            }


def reserve_physical_batch(
    requests: Sequence[ProviderBatchRequest | Mapping[str, Any]],
    *,
    provider_id: str,
    snapshot: Any = None,
    mode: UsageAdmissionMode | str = UsageAdmissionMode.ENFORCE,
    unknown_policy: UnknownStalePolicy | str = UnknownStalePolicy.FAIL_CLOSED,
    shared_overhead_tokens: int = 0,
    ancestor_budget: HierarchicalBudgetView | Mapping[str, Any] | None = None,
    base_capacity: ProviderBatchCapacity | Mapping[str, Any] | None = None,
    now_ms: int | None = None,
    batch_id: str = "",
) -> tuple[PhysicalBatchReservation | None, ProviderBatchAdmissionGrant]:
    """Atomically reserve once for a physical batch under usage capacity.

    Returns ``(reservation, grant)``.  On denial, reservation is None and the
    grant carries ``usage_capacity_unavailable`` when enforce mode blocks.
    """

    mode_e = _as_mode(mode)
    identity = str(provider_id or "").strip()
    if not identity:
        raise ValueError("provider_id must be non-empty")

    normalized: list[ProviderBatchRequest] = []
    for item in requests:
        if isinstance(item, ProviderBatchRequest):
            normalized.append(item)
        else:
            normalized.append(ProviderBatchRequest.from_value(item))
    if not normalized:
        return None, ProviderBatchAdmissionGrant(
            admitted=False, reason="empty_batch"
        )

    request_ids = tuple(item.request_id for item in normalized)
    total_tokens = sum(max(0, int(item.token_budget)) for item in normalized)
    overhead = max(0, int(shared_overhead_tokens))
    bid = str(batch_id or f"batch:{uuid.uuid4().hex}")

    batch_cap, projection = project_batch_capacity_from_usage_snapshot(
        snapshot,
        provider_id=identity,
        base=base_capacity,
        mode=mode_e,
        unknown_policy=unknown_policy,
        now_ms=now_ms,
    )
    as_provider = ProviderCapacity(
        provider_id=identity.lower(),
        healthy=batch_cap.healthy,
        token_budget_remaining=batch_cap.token_budget_remaining,
        max_concurrency=batch_cap.max_concurrent_batches,
        active_requests=max(
            0,
            batch_cap.max_concurrent_batches - batch_cap.available_concurrent_batches,
        ),
        retry_after_ms=batch_cap.retry_after_ms,
    )
    budgeted = intersect_with_ancestor_budgets(as_provider, ancestor_budget)
    if (
        budgeted.token_budget_remaining != batch_cap.token_budget_remaining
        or budgeted.max_concurrency != batch_cap.max_concurrent_batches
        or budgeted.healthy != batch_cap.healthy
    ):
        batch_cap = ProviderBatchCapacity(
            provider_id=batch_cap.provider_id,
            healthy=bool(budgeted.healthy and batch_cap.healthy),
            max_batch_size=batch_cap.max_batch_size,
            max_concurrent_batches=budgeted.max_concurrency,
            available_concurrent_batches=max(
                0, budgeted.max_concurrency - budgeted.active_requests
            ),
            token_budget_remaining=budgeted.token_budget_remaining,
            retry_after_ms=budgeted.retry_after_ms,
        )

    reasons: list[str] = list(projection.reason_codes)
    admitted = True
    if mode_e is UsageAdmissionMode.OFF:
        admitted = True
    else:
        if not batch_cap.healthy:
            admitted = False
            reasons.append("provider_unhealthy")
        if batch_cap.retry_after_ms > 0:
            admitted = False
            reasons.append("provider_backoff")
        if batch_cap.available_concurrent_batches <= 0:
            admitted = False
            reasons.append("provider_concurrency")
        if (
            batch_cap.token_budget_remaining >= 0
            and total_tokens + overhead > batch_cap.token_budget_remaining
        ):
            admitted = False
            reasons.append("provider_token_budget")
        if batch_cap.max_batch_size and len(normalized) > batch_cap.max_batch_size:
            admitted = False
            reasons.append("batch_size_exceeded")
        # observe/shadow: do not block
        if mode_e in {UsageAdmissionMode.OBSERVE, UsageAdmissionMode.SHADOW}:
            admitted = True

    if not admitted and mode_e is UsageAdmissionMode.ENFORCE:
        reasons.append(USAGE_CAPACITY_UNAVAILABLE)
        return None, ProviderBatchAdmissionGrant(
            admitted=False,
            reason=reasons[0] if reasons else USAGE_CAPACITY_UNAVAILABLE,
        )

    reservation = PhysicalBatchReservation(
        reservation_id=f"bres:{uuid.uuid4().hex}",
        provider_id=identity,
        batch_id=bid,
        member_request_ids=request_ids,
        total_token_budget=total_tokens,
        shared_overhead_tokens=overhead,
        mode=mode_e.value,
        usage_revision=projection.usage_revision,
        scope_id=projection.scope_id,
        projection_id=projection.projection_id,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
    for item in normalized:
        reservation.attribute_member(
            item.request_id,
            token_budget=item.token_budget,
            cancelled=False,
            charged=True,
        )

    released = {"done": False}
    lock = threading.Lock()

    def _release() -> None:
        with lock:
            if released["done"]:
                return
            released["done"] = True
            reservation.release()

    grant = ProviderBatchAdmissionGrant(
        admitted=True if mode_e is not UsageAdmissionMode.ENFORCE or admitted else admitted,
        reason="" if admitted else (reasons[0] if reasons else ""),
        lease=reservation,
        release=_release,
    )
    # Enforce denial already returned. Assist may admit with reasons.
    if mode_e is UsageAdmissionMode.ASSIST and not admitted:
        # Assist surfaces capacity pressure but may still admit under policy;
        # keep grant admitted with reason for backpressure metrics.
        grant = ProviderBatchAdmissionGrant(
            admitted=True,
            reason=reasons[0] if reasons else "assist_soft_admit",
            lease=reservation,
            release=_release,
        )
    return reservation, grant


# ---------------------------------------------------------------------------
# Usage-aware batch scheduler wrapper
# ---------------------------------------------------------------------------


class UsageAwareProviderBatchScheduler(ProviderBatchScheduler):
    """ProviderBatchScheduler with endpoint usage projection and fair reserves."""

    def __init__(
        self,
        dispatch: Any | None = None,
        *,
        providers: Mapping[str, Any] | None = None,
        config: ProviderBatchSchedulerConfig | None = None,
        capacity_supplier: Any | None = None,
        admission: Any | None = None,
        fallback: Any | None = None,
        clock_ms: Callable[[], int] | None = None,
        usage_mode: UsageAdmissionMode | str = UsageAdmissionMode.OFF,
        unknown_policy: UnknownStalePolicy | str = UnknownStalePolicy.FAIL_CLOSED,
        usage_snapshot_supplier: Callable[[str], Any] | None = None,
        fair_queue: WeightedFairQueue | None = None,
        shared_overhead_tokens: int = 0,
        ancestor_budget: HierarchicalBudgetView | Mapping[str, Any] | None = None,
    ) -> None:
        self.usage_mode = _as_mode(usage_mode)
        self.unknown_policy = (
            unknown_policy
            if isinstance(unknown_policy, UnknownStalePolicy)
            else UnknownStalePolicy(str(unknown_policy))
        )
        self.usage_snapshot_supplier = usage_snapshot_supplier
        self.fair_queue = fair_queue or WeightedFairQueue()
        self.shared_overhead_tokens = max(0, int(shared_overhead_tokens))
        self.ancestor_budget = ancestor_budget
        self.reset_cursor = ResetEventCursor()
        self._single_flight = SingleFlightRefresh()
        self._batch_reservations: dict[str, PhysicalBatchReservation] = {}
        self._reservation_lock = threading.RLock()
        self._usage_clock = clock_ms or (lambda: int(time.time() * 1000))

        usage_capacity_supplier = capacity_supplier
        if self.usage_mode is not UsageAdmissionMode.OFF and usage_snapshot_supplier is not None:
            base_supplier = capacity_supplier

            def _usage_capacity(provider_id: str) -> ProviderBatchCapacity:
                base = None
                if base_supplier is not None:
                    base = base_supplier(provider_id)
                snap = self._single_flight.do(
                    f"batch-snap:{provider_id}",
                    lambda: usage_snapshot_supplier(provider_id),
                )
                cap, projection = project_batch_capacity_from_usage_snapshot(
                    snap,
                    provider_id=provider_id,
                    base=base,
                    mode=self.usage_mode,
                    unknown_policy=self.unknown_policy,
                    now_ms=self._usage_clock(),
                )
                if projection.next_eligible_at_ms:
                    self.reset_cursor.note_next_eligible(
                        projection.next_eligible_at_ms, provider_id
                    )
                return cap

            usage_capacity_supplier = _usage_capacity

        usage_admission = admission
        if self.usage_mode is not UsageAdmissionMode.OFF and admission is None:
            def _usage_admission(
                key: ProviderBatchKey,
                requests: Sequence[ProviderBatchRequest],
                capacity: ProviderBatchCapacity,
            ) -> ProviderBatchAdmissionGrant:
                snap = None
                if usage_snapshot_supplier is not None:
                    snap = self._single_flight.do(
                        f"batch-snap:{key.provider_id}",
                        lambda: usage_snapshot_supplier(key.provider_id),
                    )
                reservation, grant = reserve_physical_batch(
                    requests,
                    provider_id=key.provider_id,
                    snapshot=snap,
                    mode=self.usage_mode,
                    unknown_policy=self.unknown_policy,
                    shared_overhead_tokens=self.shared_overhead_tokens,
                    ancestor_budget=self.ancestor_budget,
                    base_capacity=capacity,
                    now_ms=self._usage_clock(),
                )
                if reservation is not None:
                    with self._reservation_lock:
                        self._batch_reservations[reservation.batch_id] = reservation
                return grant

            usage_admission = _usage_admission

        super().__init__(
            dispatch,
            providers=providers,
            config=config,
            capacity_supplier=usage_capacity_supplier,
            admission=usage_admission,
            fallback=fallback,
            clock_ms=clock_ms,
        )

    def settle_batch_overhead(self, batch_id: str) -> int:
        with self._reservation_lock:
            reservation = self._batch_reservations.get(str(batch_id))
        if reservation is None:
            return 0
        return reservation.settle_shared_overhead_once()

    def cancel_member_attribution(self, batch_id: str, request_id: str) -> BatchMemberAttribution | None:
        with self._reservation_lock:
            reservation = self._batch_reservations.get(str(batch_id))
        if reservation is None:
            return None
        return reservation.cancel_member(request_id)

    def wake_due_resets(self) -> tuple[str, ...]:
        return self.reset_cursor.due(self._usage_clock())


# ---------------------------------------------------------------------------
# Install onto runtime module
# ---------------------------------------------------------------------------

_INSTALL_LOCK = threading.RLock()

_EXPORTS: dict[str, Any] = {
    "ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID": ENDPOINT_USAGE_BATCH_ADMISSION_REQUIREMENT_ID,
    "ENDPOINT_USAGE_BATCH_ADMISSION_SCHEMA": ENDPOINT_USAGE_BATCH_ADMISSION_SCHEMA,
    "PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID": PHYSICAL_BATCH_RESERVE_ONCE_REQUIREMENT_ID,
    "BatchMemberAttribution": BatchMemberAttribution,
    "PhysicalBatchReservation": PhysicalBatchReservation,
    "project_batch_capacity_from_usage_snapshot": project_batch_capacity_from_usage_snapshot,
    "reserve_physical_batch": reserve_physical_batch,
    "UsageAwareProviderBatchScheduler": UsageAwareProviderBatchScheduler,
    # Re-surface resource usage symbols commonly needed with batch admission.
    "ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID": ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID,
    "USAGE_CAPACITY_UNAVAILABLE": USAGE_CAPACITY_UNAVAILABLE,
    "UsageAdmissionMode": UsageAdmissionMode,
    "UnknownStalePolicy": UnknownStalePolicy,
    "WeightedFairQueue": WeightedFairQueue,
    "FairQueueScope": FairQueueScope,
}


def install_endpoint_usage_batch_admission(module: Any | None = None) -> None:
    """Install ASI-167 batch symbols onto the runtime provider_batch_scheduler."""

    target = module if module is not None else _runtime
    with _INSTALL_LOCK:
        for name, value in _EXPORTS.items():
            setattr(target, name, value)
        existing = list(getattr(target, "__all__", ()) or ())
        for name in _EXPORTS:
            if name not in existing:
                existing.append(name)
        try:
            target.__all__ = existing  # type: ignore[attr-defined]
        except Exception:
            pass


install_endpoint_usage_batch_admission()

sys.modules.setdefault(
    "ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler", _runtime
)

__all__ = sorted(
    set(getattr(_runtime, "__all__", ()))
    | set(_EXPORTS)
    | {"install_endpoint_usage_batch_admission"}
)
