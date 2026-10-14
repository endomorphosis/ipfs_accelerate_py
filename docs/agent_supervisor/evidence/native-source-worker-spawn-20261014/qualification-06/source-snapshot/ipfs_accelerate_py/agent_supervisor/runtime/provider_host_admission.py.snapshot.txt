"""Default shared host admission for supervisor provider dispatch.

The datasets GlobalResourceScheduler remains the host CPU/RAM/process owner.
The supervisor scheduler checks local claims and sampled disk headroom only.
This bounded CPU envelope covers one batch callback; it neither grants provider
quota nor assumes that a router batch maps to one remote request. Native GPU
admission and kernel memory enforcement are outside this profile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .provider_batch_scheduler import ProviderBatchAdmissionGrant, ResourceSchedulerBatchAdmission
from .resource_scheduler import ResourceLeaseBudget, ResourcePolicy, ResourceScheduler

MIB = 1024 * 1024


@dataclass(frozen=True)
class ProviderHostBudget:
    cpu_slots: int = 1
    memory_mb: int = 256
    process_slots: int = 1
    max_parallel: int = 4

    def __post_init__(self):
        for name, maximum in (("cpu_slots", 1024), ("memory_mb", 65536),
                ("process_slots", 1024), ("max_parallel", 256)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(name + " must be a bounded positive integer")
        if self.process_slots > self.cpu_slots:
            raise ValueError("CPU slots must cover process slots")


class ProviderHostAdmission:
    """Nonblocking local policy plus one native shared root/child reservation.

    Native imports and authority resolution happen only at admission. An
    explicitly supplied parent delegates its capacity instead of reserving a
    second root. Missing or disabled native safety refuses before dispatch.
    """

    def __init__(self, *, budget: ProviderHostBudget | None = None,
                 local_scheduler: ResourceScheduler | None = None,
                 native_scheduler: Any = None, parent_lease: Any = None, path: Any = "."):
        self.budget = budget or ProviderHostBudget()
        if type(self.budget) is not ProviderHostBudget:
            raise TypeError("exact ProviderHostBudget required")
        process_capacity = self.budget.max_parallel * self.budget.process_slots
        self.local_scheduler = local_scheduler or ResourceScheduler(ResourcePolicy(
            max_lanes=process_capacity, require_provider_telemetry=False))
        if type(self.local_scheduler) is not ResourceScheduler:
            raise TypeError("actual supervisor resource owner required")
        self.native_scheduler = native_scheduler
        self.parent_lease = parent_lease
        self.local_admission = ResourceSchedulerBatchAdmission(self.local_scheduler,
            path=path, memory_bytes=self.budget.memory_mb * MIB,
            budget=ResourceLeaseBudget(max_parallel=process_capacity,
                max_model_concurrency=process_capacity,
                max_processes=process_capacity,
                memory_bytes=self.budget.max_parallel * self.budget.memory_mb * MIB))

    @staticmethod
    def _refused(reason, *, retryable=False):
        return ProviderBatchAdmissionGrant(admitted=False, reason=reason, retryable=retryable)

    def __call__(self, key, requests, capacity):
        if any(request.batch_key != key for request in requests):
            return self._refused("incompatible_resource_claims")
        try:
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
                GlobalResourceScheduler, ResourceLane, ResourceLease,
                LeaseCancelledError, ResourceConfigurationError, ResourceSchedulerError,
                get_global_resource_scheduler,
            )
        except ImportError:
            return self._refused("native_host_admission_unavailable")
        try:
            native = self.native_scheduler
            if native is None:
                native = get_global_resource_scheduler()
            if type(native) is not GlobalResourceScheduler or not native.config.proof_safety_enabled:
                return self._refused("native_host_safety_required")
            if self.parent_lease is not None and (
                type(self.parent_lease) is not ResourceLease
                or self.parent_lease.token.state_path != str(native.state_path)
            ):
                return self._refused("native_host_parent_mismatch")
        except (ResourceSchedulerError, OSError):
            return self._refused("native_host_authority_unavailable")

        # Strengthen the per-call process declaration to the trusted envelope.
        # Each local process slot consumes one lane; injected owner policies may
        # conservatively allow fewer callbacks than max_parallel.
        local_requests = tuple(replace(request, provenance={**request.provenance,
            "process_slots": max(self.budget.process_slots, request.provenance.get("process_slots", 1))})
            if type(request.provenance.get("process_slots", 1)) is int else request
            for request in requests)
        local_key = local_requests[0].batch_key if local_requests else key
        local = self.local_admission(local_key, local_requests, capacity)
        if not local.admitted:
            return local
        requirement = local.lease.requirement
        if "gpu" in requirement.resource_kinds:
            local.release_once()
            return self._refused("native_host_cpu_profile_requires_no_gpu")
        if (requirement.memory_bytes > self.budget.memory_mb * MIB
                or requirement.process_slots > self.budget.process_slots):
            local.release_once()
            return self._refused("native_host_demand_exceeds_budget")
        shared = None
        try:
            # Admission runs under the batch scheduler condition. A blocking
            # native wait here would prevent timely queue cancellation.
            shared = native.try_acquire(ResourceLane.ORCHESTRATION,
                cpu_slots=self.budget.cpu_slots, memory_mb=self.budget.memory_mb,
                child_process_slots=self.budget.process_slots,
                parent_lease=self.parent_lease, request_id=requirement.lane_id)
            if shared is None:
                return self._refused("native_host_pressure_or_capacity", retryable=True)
        except LeaseCancelledError:
            return self._refused("native_host_parent_cancelled")
        except ResourceConfigurationError:
            return self._refused("native_host_configuration_refused")
        except (ResourceSchedulerError, OSError):
            return self._refused("native_host_admission_unavailable", retryable=True)
        finally:
            if shared is None:
                local.release_once()

        def release():
            try:
                shared.release()
            finally:
                local.release_once()

        def before_dispatch(_requests):
            if shared.released or shared.cancelled:
                raise RuntimeError("native_host_lease_no_longer_active")

        return ProviderBatchAdmissionGrant(admitted=True, release=release, lease=shared,
            reason="native_shared_host_admitted", before_dispatch=before_dispatch)

    def compose(self, additional=None):
        """Preserve custom admission hooks while retaining default host safety."""
        if additional is None:
            return self
        if not callable(additional):
            raise TypeError("additional provider admission must be callable")

        def admit(key, requests, capacity):
            host = self(key, requests, capacity)
            if not host.admitted:
                return host
            try:
                custom = ProviderBatchAdmissionGrant.from_value(additional(key, requests, capacity))
            except BaseException:
                host.release_once()
                raise
            if not custom.admitted:
                try:
                    custom.release_once()
                finally:
                    host.release_once()
                return replace(custom, release=None)

            def release():
                try:
                    custom.release_once()
                finally:
                    host.release_once()

            def before_dispatch(requests):
                if host.before_dispatch is not None:
                    host.before_dispatch(requests)
                if custom.before_dispatch is not None:
                    custom.before_dispatch(requests)

            return replace(custom, release=release, before_dispatch=before_dispatch)

        return admit


__all__ = ["ProviderHostAdmission", "ProviderHostBudget"]
