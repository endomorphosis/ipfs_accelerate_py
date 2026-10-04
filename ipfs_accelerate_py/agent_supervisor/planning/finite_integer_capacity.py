"""Live reserved capacity for the closed finite-integer planning preview.

The native compiler's ``admitted`` result describes a resource forecast. No
worker is launched, no reviewed operation is executed, and no signed evidence
or execution authority is created. A retained snapshot is descriptive after
the private scheduler lease is released.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
import time
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import (
    ProofHostResources, collect_proof_host_resources,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLease, ResourceLeaseToken, ResourceLane,
    _owner_is_alive,
)

from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateStage, PlanCreateStageResult,
    freeze_plan_create_input_snapshot,
)
from .adaptive_planner import FrozenPlanningGoal
from .finite_integer_plan_service import (
    FiniteIntegerPlanCreateService, FiniteIntegerPlanServiceError,
    _FiniteNoWorkCandidate, _json,
)
from .parallel_plan_compiler import (
    ParallelPlanCompilationRequest, ParallelPlanCompiler, ParallelExecutionPlan,
)
from .plan_revision_contracts import PlanCreateRequest
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

FINITE_CAPACITY_PROFILE = "finite-integer-live-reserved-capacity@2"
_EXTRA_KEYS = frozenset({"finite_capacity_profile", "finite_capacity_binding"})
_FRESHNESS_MS = 30_000
_FRESHNESS_POLICY = {
    "schema": "finite-integer-capacity-freshness-policy@1", "max_age_ms": _FRESHNESS_MS,
    "clamps": ["native_reservation_expiry", "native_ancestor_expiry", "request_deadline"],
    "expiry_rule": "current_time_ms_must_be_less_than_fresh_until_ms",
    "independent_host_recheck_at_every_fence": True,
    "original_forecast_timestamp_preserved": True,
    "execution_authority": False,
}
_MIB = 1024 * 1024
_SEAL = object()
_RESOURCE_SHADOWS = frozenset({
    "cpu_slots", "process_slots", "child_process_slots", "memory_bytes", "memory_mb",
    "gpu_memory_bytes", "disk_bytes", "resource_contract", "lease_contract",
    "provider_contract", "worktree_contract", "conflict_contract", "merge_strategy",
})


class FiniteIntegerCapacityError(FiniteIntegerPlanServiceError):
    """The actual reservation cannot support the exact reviewed forecast."""


def _wire(value):
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _now_ms():
    return math.ceil(time.time() * 1000)


def _inert_native(value):
    """Preserve exact native configuration floats as inert hexadecimal text."""
    if type(value) is float:
        if not math.isfinite(value):
            raise FiniteIntegerCapacityError("finite native scheduler configuration required")
        return {"native_float_hex": value.hex()}
    if type(value) is dict:
        return {key: _inert_native(member) for key, member in value.items()}
    if type(value) in {list, tuple}:
        return [_inert_native(member) for member in value]
    return value


def _config(scheduler):
    configuration = scheduler.config
    configuration.validate()
    if configuration.proof_safety_enabled is not True:
        raise FiniteIntegerCapacityError("native proof-safety scheduler configuration required")
    return _json(_inert_native({
        "persisted": configuration.persisted_dict(),
        "lease_ttl_seconds": float(configuration.lease_ttl_seconds),
        "auto_renew_leases": configuration.auto_renew_leases,
        "state_path": str(scheduler.state_path),
    }))


def _basis(request, materials):
    if type(request) is not PlanCreateRequest or type(materials) is not PlanCreateMaterials:
        raise FiniteIntegerCapacityError("exact native request and materials required")
    if type(materials.extra) is not dict:
        raise FiniteIntegerCapacityError("exact capacity material context required")
    plain = replace(materials, extra={key: value for key, value in materials.extra.items()
                                     if key not in _EXTRA_KEYS})
    snapshot = freeze_plan_create_input_snapshot(request, materials=plain)
    if snapshot.material_binding.get("reuse_supported") is not True:
        raise FiniteIntegerCapacityError("fully bound model-disabled semantic material context required")
    return _json(snapshot.to_dict())


def _native_scheduler(owner):
    scheduler = owner.scheduler
    if scheduler is None and type(owner.parent_lease) is ResourceLease:
        scheduler = owner.parent_lease._scheduler
    if type(scheduler) is not GlobalResourceScheduler:
        raise FiniteIntegerCapacityError("exact native scheduler from owner or live parent required")
    return scheduler


def _owner_request(owner, request, materials):
    if (type(owner) is not RepositoryPlanPreviewOwner
            or type(request) is not PlanCreateRequest
            or type(materials) is not PlanCreateMaterials
            or not 1024 <= owner.memory_mb <= 4096):
        raise FiniteIntegerCapacityError("native owner, scheduler and bounded proof memory required")
    _native_scheduler(owner)
    if (request.repository_id != owner.expected_head.repository_id
            or request.repository_root != str(owner.repository)
            or request.roots.repository_root_cid != owner.expected_head.snapshot_cid
            or request.roots.dirty_worktree_root != owner.expected_head.snapshot_cid
            or owner.index.catalog.current(request.repository_id) != owner.expected_head):
        raise FiniteIntegerCapacityError("capacity request differs from the current selected native head")
    if (request.budget.max_model_calls != 0 or materials.model_provider is not None
            or materials.parallel_request is not None or materials.parallel_tasks is not None
            or materials.current_roots is not None or materials.admission_materials is not None
            or type(materials.frozen_goal) is not FrozenPlanningGoal or materials.frozen_goal.policy.require_proof):
        raise FiniteIntegerCapacityError("model-off preview without injected parallel or admission materials required")
    if materials.extra.get("root_observation_profile") != "repository-live-root-observation@1":
        raise FiniteIntegerCapacityError("native live source observation profile must be frozen before capacity acquisition")
    context = materials.extra.get("structural_codebase")
    head = {key: value for key, value in owner.expected_head.to_dict().items() if key != "schema"}
    if (type(context) is not dict or context.get("head") != head
            or context.get("semantic_state_cid") != request.roots.program_root
            or cid_for_structured(context) != materials.extra.get("structural_codebase_context_cid")):
        raise FiniteIntegerCapacityError("exact selected structural material context required")


def _authenticate(scheduler, control, state, now):
    if type(control) is ResourceLease:
        if control._scheduler is not scheduler or control.released:
            raise FiniteIntegerCapacityError("live native reservation from the exact scheduler required")
        identifier, key = control.lease_id, control.lease_key
    elif type(control) is ResourceLeaseToken:
        if control.state_path != str(scheduler.state_path):
            raise FiniteIntegerCapacityError("parent token belongs to another scheduler")
        identifier, key = control.lease_id, control.lease_key
    else:
        raise FiniteIntegerCapacityError("native parent lease or token required; lease identifiers are not authority")
    record = state["leases"].get(identifier)
    if type(record) is not dict or record.get("lease_key") != key:
        raise FiniteIntegerCapacityError("native reservation identity does not match scheduler state")
    seen, ancestor = set(), record
    while ancestor is not None:
        lease_id = ancestor.get("lease_id")
        if (lease_id in seen or float(ancestor.get("expires_at", 0)) <= now
                or ancestor.get("cancelled") or ancestor.get("release_requested")
                or not _owner_is_alive(int(ancestor.get("owner_pid", 0)),
                                       str(ancestor.get("owner_birth_marker", "")))):
            raise FiniteIntegerCapacityError("reservation or native parent is expired, cancelled, released or owner-stale")
        seen.add(lease_id)
        parent = ancestor.get("parent_lease_id")
        if not parent:
            break
        ancestor = state["leases"].get(parent)
        if ancestor is None:
            raise FiniteIntegerCapacityError("native reservation parent is missing")
    return record


def _unspent(state, record):
    children = [row for row in state["leases"].values()
                if row.get("parent_lease_id") == record["lease_id"]]
    return {name: int(record.get(name, 0)) - sum(int(row.get(name, 0)) for row in children)
            for name in ("cpu_slots", "memory_mb", "child_process_slots")}


def _parent_budget(owner, state, now):
    if owner.parent_lease is None:
        return None
    parent = _authenticate(_native_scheduler(owner), owner.parent_lease, state, now)
    free = _unspent(state, parent)
    if (free["cpu_slots"] < 2 or free["memory_mb"] < owner.memory_mb * 2
            or free["child_process_slots"] < 2):
        raise FiniteIntegerCapacityError("parent budget must cover capacity child and a sibling source observation")
    return parent


def _host(scheduler, state, now):
    try:
        host = collect_proof_host_resources()
    except Exception as exc:
        raise FiniteIntegerCapacityError("independent current host resource telemetry unavailable") from exc
    if type(host) is not ProofHostResources:
        raise FiniteIntegerCapacityError("exact independent native host telemetry required")
    configuration = scheduler.config
    for name, threshold in (
        ("memory_stall_percent", configuration.proof_memory_stall_percent),
        ("cpu_stall_percent", configuration.proof_cpu_stall_percent),
        ("io_stall_percent", configuration.proof_io_stall_percent),
    ):
        if getattr(host, name) >= threshold:
            raise FiniteIntegerCapacityError("current independent host pressure exceeds native proof threshold: " + name)
    backoff = state.get("proof_backoff", {})
    if float(backoff.get("until", 0)) > now:
        raise FiniteIntegerCapacityError("native proof pressure backoff is still active")
    roots = [row for row in state["leases"].values() if not row.get("parent_lease_id")]
    reserved_cpu = sum(int(row["cpu_slots"]) for row in roots)
    reserved_memory = sum(int(row["memory_mb"]) for row in roots)
    return host, reserved_cpu, reserved_memory


class ReservedFiniteIntegerCapacity:
    """Private live control; serialized bindings cannot reconstruct its lease."""

    def __init__(self, *, seal, owner, request, materials, lease, deadline, config):
        if seal is not _SEAL:
            raise FiniteIntegerCapacityError("capacity must be acquired through the native owner context")
        self.owner, self.lease = owner, lease
        self._request_cid = request.request_cid
        self._request_wire, self._basis_wire = _wire(request.to_dict()), _wire(_basis(request, materials))
        self._config_wire, self._deadline = _wire(config), deadline
        self._active, self._binding_wire = True, None
        self._lease_identity = None
        self.last_observation = None
        self.last_compilation_request = None
        self._forecast_observation = None
        observed = self._observe()
        body = {"schema": "finite-integer-native-capacity-binding@1", "profile": FINITE_CAPACITY_PROFILE,
            "request_cid": request.request_cid, "selected_head": owner.expected_head.to_dict(),
            "material_snapshot": json.loads(self._basis_wire), "scheduler_configuration": config,
            "reservation": observed["reservation"], "initial_observation": observed,
            "freshness_policy": _json(_FRESHNESS_POLICY),
            "freshness_policy_cid": cid_for_structured(_FRESHNESS_POLICY),
            "task_resource_profile": self.task_resource_profile,
            "production_admitted": False, "execution_authority": False,
            "proof_authority": False, "completion_authority": False}
        self._binding_wire = _wire({**body, "binding_cid": cid_for_structured(body)})

    @property
    def material_binding(self):
        return json.loads(self._binding_wire)

    @property
    def task_resource_profile(self):
        return {"cpu_slots": 1, "process_slots": 1, "memory_bytes": self.owner.memory_mb * _MIB,
                "gpu_memory_bytes": 0, "disk_bytes": 0, "resource_class": "cpu"}

    def _remaining(self):
        if not self._active or self.lease.released:
            raise FiniteIntegerCapacityError("capacity reservation context is released")
        if self.owner.cancel_event is not None and self.owner.cancel_event.is_set():
            raise FiniteIntegerCapacityError("capacity owner is cancelled")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise FiniteIntegerCapacityError("capacity preview deadline expired")
        return remaining

    def _observe(self):
        self._remaining()
        scheduler = _native_scheduler(self.owner)
        if _wire(_config(scheduler)) != self._config_wire:
            raise FiniteIntegerCapacityError("native capacity configuration changed")
        now = time.time()
        with scheduler._locked_state(persist=False) as state:
            record = _authenticate(scheduler, self.lease, state, now)
            expected_id = ("finite-capacity:" + self._request_cid)[:256]
            if record.get("owner_pid") != os.getpid() or record.get("request_id") != expected_id:
                raise FiniteIntegerCapacityError("reservation belongs to a different current process")
            parent_id = (self.owner.parent_lease.lease_id if self.owner.parent_lease is not None else None)
            if record.get("parent_lease_id") != parent_id:
                raise FiniteIntegerCapacityError("reservation no longer belongs to the provided parent budget")
            if self.owner.parent_lease is not None:
                _authenticate(scheduler, self.owner.parent_lease, state, now)
            identity = {name: record.get(name) for name in (
                "lease_id", "sequence", "lane", "parent_lease_id", "owner_pid", "owner_birth_marker",
                "request_id", "cpu_slots", "memory_mb", "child_process_slots", "gpu_memory_mb",
                "unified_memory_mb", "requires_gpu")}
            identity["acquired_at_ms"] = math.ceil(float(record["acquired_at"]) * 1000)
            if self._lease_identity is not None and identity != self._lease_identity:
                raise FiniteIntegerCapacityError("native reservation declaration changed")
            if (identity["cpu_slots"] != 1 or identity["memory_mb"] != self.owner.memory_mb
                    or identity["child_process_slots"] != 1 or identity["gpu_memory_mb"] != 0
                    or identity["unified_memory_mb"] != 0 or identity["requires_gpu"] is not False):
                raise FiniteIntegerCapacityError("exact finite capacity reservation required")
            free = _unspent(state, record)
            host, root_cpu, root_memory = _host(scheduler, state, now)
            cpu = min(free["cpu_slots"], host.cpu_slots - max(0, root_cpu - identity["cpu_slots"]))
            memory = min(free["memory_mb"], host.available_memory_mb
                         - scheduler.config.proof_memory_headroom_mb - root_memory)
            if cpu < 1 or memory < self.owner.memory_mb or free["child_process_slots"] < 1:
                raise FiniteIntegerCapacityError("live reservation or independent host headroom no longer covers task profile")
            expires_ms = math.floor(float(record["expires_at"]) * 1000)
            ancestor = record
            while ancestor.get("parent_lease_id"):
                ancestor = state["leases"][ancestor["parent_lease_id"]]
                expires_ms = min(expires_ms, math.floor(float(ancestor["expires_at"]) * 1000))
            self._lease_identity = _json(identity)
        # Use sampling completion time, rather than an earlier wall-clock read.
        measured_ms = _now_ms()
        fresh_until = min(measured_ms + _FRESHNESS_MS, expires_ms,
                          measured_ms + math.floor(self._remaining() * 1000))
        if fresh_until <= measured_ms:
            raise FiniteIntegerCapacityError("reservation observation expired during sampling")
        body = {"schema": "finite-integer-native-capacity-observation@1", "profile": FINITE_CAPACITY_PROFILE,
            "observed_at_ms": measured_ms, "fresh_until_ms": fresh_until, "max_age_ms": _FRESHNESS_MS,
            "freshness_policy_cid": cid_for_structured(_FRESHNESS_POLICY),
            "cpu_slots": cpu, "process_slots": free["child_process_slots"], "memory_bytes": memory * _MIB,
            "gpu_memory_bytes": 0, "disk_bytes": 0, "resource_classes": ["cpu"],
            "reservation": identity, "host": {"cpu_slots": host.cpu_slots,
                "total_memory_mb": host.total_memory_mb, "available_memory_mb": host.available_memory_mb,
                "memory_stall_millipercent": math.ceil(host.memory_stall_percent * 1000),
                "cpu_stall_millipercent": math.ceil(host.cpu_stall_percent * 1000),
                "io_stall_millipercent": math.ceil(host.io_stall_percent * 1000)},
            "root_reserved_cpu_slots": root_cpu, "root_reserved_memory_mb": root_memory,
            "proof_memory_headroom_mb": scheduler.config.proof_memory_headroom_mb,
            "request_cid": self._request_cid,
            "selected_head": self.owner.expected_head.to_dict(),
            "material_snapshot_cid": json.loads(self._basis_wire)["snapshot_cid"],
            "worker_launched": False, "execution_authority": False, "production_admitted": False}
        observation = _json({**body, "snapshot_id": cid_for_structured(body)})
        self.last_observation = _json(observation)
        return observation

    def require_current(self, request, materials, *, bound=True):
        _owner_request(self.owner, request, materials)
        if _wire(request.to_dict()) != self._request_wire or _wire(_basis(request, materials)) != self._basis_wire:
            raise FiniteIntegerCapacityError("exact request and complete pre-capacity materials changed")
        if bound and (materials.extra.get("finite_capacity_profile") != FINITE_CAPACITY_PROFILE
                or _wire(materials.extra.get("finite_capacity_binding")) != self._binding_wire):
            raise FiniteIntegerCapacityError("exact frozen native capacity binding required")
        observation = self._observe()
        if self._forecast_observation is not None:
            original = self._forecast_observation
            now_ms = _now_ms()
            if (now_ms >= original["fresh_until_ms"] or now_ms < original["observed_at_ms"]
                    or any(observation[name] < original[name] for name in (
                        "cpu_slots", "process_slots", "memory_bytes", "gpu_memory_bytes", "disk_bytes"))):
                raise FiniteIntegerCapacityError("retained forecast capacity is stale or no longer available")
        return observation

    def compile(self, *, request, materials, candidate_plan):
        self.require_current(request, materials)
        candidate = _json(candidate_plan)
        if (type(candidate) is not dict or candidate.get("request_cid") != request.request_cid
                or candidate.get("plan_id") != cid_for_structured({key: value for key, value in candidate.items()
                                                                  if key != "plan_id"})
                or type(candidate.get("tasks")) is not list or not candidate["tasks"]):
            raise FiniteIntegerCapacityError("full nonempty selected finite candidate plan required")
        if any(type(task) is not dict or set(task) & _RESOURCE_SHADOWS for task in candidate["tasks"]):
            raise FiniteIntegerCapacityError("reviewed tasks cannot shadow the fixed reserved resource profile")
        with self._source_context() as context:
            if (context.to_dict() != materials.extra["structural_codebase"]
                    or context.cid != materials.extra["structural_codebase_context_cid"]):
                raise FiniteIntegerCapacityError("native source context differs from the frozen capacity materials")
            observed = self.require_current(request, materials)
            snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
            capacity_body = {**observed, "capacity_binding_cid": self.material_binding["binding_cid"],
                "candidate_plan_cid": cid_for_structured(candidate), "candidate_plan_id": candidate["plan_id"],
                "bound_material_snapshot_cid": snapshot.snapshot_cid,
                "task_resource_profile": self.task_resource_profile}
            capacity_body.pop("snapshot_id")
            capacity_snapshot = {**capacity_body, "snapshot_id": cid_for_structured(capacity_body)}
            tasks = tuple({**task, "resource_contract": self.task_resource_profile} for task in candidate["tasks"])
            compilation = ParallelPlanCompilationRequest(tasks=tasks,
                requested_width=max(1, request.budget.max_ready_width),
                repository_snapshot={"tree_id": request.roots.dirty_worktree_root,
                    "repository_tree_id": request.roots.dirty_worktree_root,
                    "selected_head": self.owner.expected_head.to_dict(), "program_root": request.roots.program_root,
                    "candidate_plan_id": candidate["plan_id"], "candidate_plan_cid": cid_for_structured(candidate),
                    "input_snapshot_cid": snapshot.snapshot_cid},
                capacity_snapshot=capacity_snapshot, current_time_ms=_now_ms(),
                deadline_ms=math.floor(time.time() * 1000 + self._remaining() * 1000),
                budget={"max_ready_width": request.budget.max_ready_width, "max_tasks": request.budget.max_tasks,
                        "max_latency_ms": math.floor(self._remaining() * 1000)},
                required_leaf_ids=tuple(candidate["expected_effect_ids"]))
            compiler = ParallelPlanCompiler()
            plan = compiler.compile(compilation)
            if type(plan) is not ParallelExecutionPlan:
                raise FiniteIntegerCapacityError("native parallel compiler output required")
            compiler.replay(plan, compilation)
            self._forecast_observation = _json(capacity_snapshot)
            self.last_compilation_request = _json(compilation.to_replay_dict())
            self.require_current(request, materials)
        # Native source exit is another mutation boundary, including callbacks.
        self.require_current(request, materials)
        return plan

    @contextmanager
    def _source_context(self):
        controls = {"repository_id": self.owner.expected_head.repository_id,
            "expected_head": self.owner.expected_head, "cancel_event": self.owner.cancel_event,
            "timeout_seconds": self._remaining(), "memory_mb": self.owner.memory_mb}
        parent = self.owner.parent_lease
        scheduler = _native_scheduler(self.owner)
        if type(parent) is ResourceLeaseToken:
            # The source adapter accepts ResourceLease only. Acquire an actual
            # sibling under the authenticated token, then use its native child
            # API; never construct a lease from serialized state.
            with scheduler.acquire(ResourceLane.SNAPSHOT_EVALUATION, cpu_slots=1,
                    memory_mb=self.owner.memory_mb, child_process_slots=1,
                    parent_lease=parent, timeout=self._remaining(), cancel_event=self.owner.cancel_event,
                    request_id="finite-capacity-source:" + self._request_cid) as source_parent:
                with structural_codebase_context(self.owner.index, self.owner.repository,
                        parent_lease=source_parent, **controls) as context:
                    yield context
        else:
            with structural_codebase_context(self.owner.index, self.owner.repository,
                    scheduler=scheduler if parent is None else None, parent_lease=parent, **controls) as context:
                yield context


@contextmanager
def reserve_finite_integer_capacity(*, owner, request, materials):
    """Reserve one real forecast envelope within the original owner budget.

    Proof/source work keeps the original parent. This separate child never
    grants execution permission and is released on every exit path.
    """
    _owner_request(owner, request, materials)
    if set(materials.extra) & _EXTRA_KEYS:
        raise FiniteIntegerCapacityError("pre-capacity materials cannot inject reservation bindings")
    _basis(request, materials)
    scheduler = _native_scheduler(owner)
    config = _config(scheduler)
    deadline = time.monotonic() + min(owner.timeout_seconds, request.budget.max_latency_ms / 1000)
    with scheduler._locked_state(persist=False) as state:
        now = time.time()
        _parent_budget(owner, state, now)
        host, root_cpu, root_memory = _host(scheduler, state, now)
        if (host.cpu_slots < root_cpu + (1 if owner.parent_lease is None else 0)
                or host.available_memory_mb < scheduler.config.proof_memory_headroom_mb
                    + root_memory + owner.memory_mb):
            raise FiniteIntegerCapacityError("independent host headroom cannot support the requested reservation")
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=owner.memory_mb,
        child_process_slots=1, parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
        timeout=max(0, deadline - time.monotonic()), request_id="finite-capacity:" + request.request_cid)
    capacity = None
    try:
        capacity = ReservedFiniteIntegerCapacity(seal=_SEAL, owner=owner, request=request,
            materials=materials, lease=lease, deadline=deadline, config=config)
        capacity.require_current(request, materials, bound=False)
        yield capacity
        capacity.require_current(request, materials, bound=False)
    finally:
        if capacity is not None:
            capacity._active = False
        lease.release()


class CapacityBoundFiniteIntegerPlanCreateService(FiniteIntegerPlanCreateService):
    """The closed reviewed projection plus live native resource feasibility."""

    def __init__(self, *, root_observer, operation_bindings, capacity):
        if type(capacity) is not ReservedFiniteIntegerCapacity or not capacity._active:
            raise FiniteIntegerCapacityError("live privately acquired finite capacity required")
        self.capacity = capacity
        self._owned_capacity = capacity
        self.capacity_observation = None
        self.capacity_compilation_request = None
        super().__init__(root_observer=root_observer, operation_bindings=operation_bindings)

    @property
    def diagnostics(self):
        return {**super().diagnostics, "capacity_observation": self.capacity_observation,
                "capacity_compilation_request": self.capacity_compilation_request}

    def _require_materials(self, request, materials):
        if (self.capacity is not self._owned_capacity
                or type(self.capacity) is not ReservedFiniteIntegerCapacity):
            raise FiniteIntegerCapacityError("exact owned capacity control cannot be replaced by stage wiring")
        super()._require_materials(request, materials)
        self.capacity.require_current(request, materials)

    def _require_current_roots(self, request, materials):
        roots = super()._require_current_roots(request, materials)
        self.capacity.require_current(request, materials)
        return roots

    def _stage_parallel_plan(self, request, materials, candidate_plan):
        self._require_materials(request, materials)
        if candidate_plan != self._canonical_plan(request, self.portfolio, self.obligation_graph):
            raise FiniteIntegerCapacityError("capacity forecast differs from the complete selected reviewed projection")
        if isinstance(self.portfolio, _FiniteNoWorkCandidate):
            return super()._stage_parallel_plan(request, materials, candidate_plan)
        plan = self.capacity.compile(request=request, materials=materials, candidate_plan=candidate_plan)
        self.execution_plan = plan
        self.capacity_observation = _json(dict(plan.replay_request)["capacity_snapshot"])
        self.capacity_compilation_request = _json(self.capacity.last_compilation_request)
        blockers = (() if plan.admitted else
                    ("parallel_plan_rejected", *("parallel:" + issue.code.value for issue in plan.issues[:8])))
        result = PlanCreateStageResult(PlanCreateStage.PARALLEL_PLAN, plan.plan_id, plan.admitted,
            blockers=blockers, message="native reserved-resource forecast; feasibility grants no execution authority")
        return result, plan


__all__ = ["FINITE_CAPACITY_PROFILE", "FiniteIntegerCapacityError", "ReservedFiniteIntegerCapacity",
           "reserve_finite_integer_capacity", "CapacityBoundFiniteIntegerPlanCreateService"]
