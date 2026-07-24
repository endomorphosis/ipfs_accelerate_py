from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    ProofPlan,
    ProofPlanStep,
    ProofStage,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof_scheduler import (
    ProofScheduler,
    ProofSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    DEFAULT_RESOURCE_CLASSES,
    HostResourceSnapshot,
    LaneResourceRequirements,
    PROOF_RESOURCE_CLASSES,
    ProofResourceClass,
    ProviderCapacity,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
    normalize_resource_class,
)


TREE = "git-tree:proof-resource"
OBLIGATION = "obligation:proof-resource"


def _host(**overrides: object) -> HostResourceSnapshot:
    values: dict[str, object] = {
        "observed_at_ms": 1_000,
        "cpu_percent": 20,
        "memory_percent": 25,
        "disk_percent": 30,
        "memory_total_bytes": 16_000,
        "memory_available_bytes": 12_000,
        "disk_total_bytes": 64_000,
        "disk_available_bytes": 48_000,
        "active_workers": 0,
        "worker_limit": 4,
        "available_worker_capacity": 4,
        "resource_classes": DEFAULT_RESOURCE_CLASSES,
    }
    values.update(overrides)
    return HostResourceSnapshot(**values)  # type: ignore[arg-type]


def _provider(**overrides: object) -> ProviderCapacity:
    values: dict[str, object] = {
        "provider_id": "provider:model",
        "healthy": True,
        "quota_remaining": 20,
        "latency_ms": 25,
        "context_window_tokens": 16_000,
        "token_budget_remaining": 20_000,
        "max_concurrency": 2,
        "active_requests": 0,
    }
    values.update(overrides)
    return ProviderCapacity(**values)  # type: ignore[arg-type]


def _step(
    step_id: str,
    stage: ProofStage,
    *,
    metadata: dict[str, object] | None = None,
) -> ProofPlanStep:
    return ProofPlanStep(
        step_id=step_id,
        obligation_id=OBLIGATION,
        stage=stage,
        provider_id="provider:model" if stage is ProofStage.MODEL_DRAFT else f"provider:{step_id}",
        resource_class=stage.value,
        metadata=metadata or {},
    )


def _plan(*steps: ProofPlanStep, max_parallel: int = 4) -> ProofPlan:
    return ProofPlan(
        repository_tree_id=TREE,
        obligation_ids=(OBLIGATION,),
        steps=steps,
        policy_id="policy:proof-resource",
        resource_budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=8_000,
            memory_bytes=8_000,
            disk_bytes=16_000,
            max_processes=max_parallel,
            model_token_limit=2_000,
            provider_quota=4,
        ),
        max_parallel=max_parallel,
    )


def test_canonical_resource_classes_distinguish_every_proof_work_type() -> None:
    assert set(PROOF_RESOURCE_CLASSES) == {
        "cpu-proof-translate",
        "cpu-proof-solver",
        "cpu-proof-kernel",
        "cpu-proof-type-check",
        "cpu-validation",
        "llm-proof-draft",
        "io-artifact",
    }
    assert set(PROOF_RESOURCE_CLASSES) <= set(DEFAULT_RESOURCE_CLASSES)
    assert normalize_resource_class("", stage=ProofStage.TRANSLATE) == (
        ProofResourceClass.TRANSLATION.value
    )
    assert normalize_resource_class("solve") == ProofResourceClass.SOLVER.value
    assert normalize_resource_class("kernel_verify") == ProofResourceClass.KERNEL.value
    assert normalize_resource_class("type_check") == ProofResourceClass.TYPE_CHECK.value
    assert normalize_resource_class("validate") == ProofResourceClass.VALIDATION.value
    assert normalize_resource_class("model_draft") == ProofResourceClass.MODEL_DRAFT.value
    assert normalize_resource_class("persist") == ProofResourceClass.ARTIFACT.value


def test_cpu_model_and_artifact_pools_are_independent_and_leases_are_reclaimable() -> None:
    scheduler = ResourceScheduler(
        ResourcePolicy(
            max_lanes=3,
            max_cpu_proof_concurrency=1,
            max_model_concurrency=1,
            max_artifact_concurrency=1,
        )
    )
    budget = ResourceLeaseBudget(
        max_parallel=3,
        max_cpu_proof_concurrency=1,
        max_model_concurrency=1,
        max_artifact_concurrency=1,
        max_processes=3,
    )
    cpu = LaneResourceRequirements(
        lane_id="cpu-1",
        resource_class=ProofResourceClass.SOLVER.value,
    )
    second_cpu = LaneResourceRequirements(
        lane_id="cpu-2",
        resource_class=ProofResourceClass.KERNEL.value,
    )
    model = LaneResourceRequirements(
        lane_id="model",
        resource_class=ProofResourceClass.MODEL_DRAFT.value,
        provider_id="provider:model",
        requires_provider=True,
        token_budget=100,
    )

    first_decision, first = scheduler.acquire(
        cpu, budget=budget, host=_host(), providers=[_provider()]
    )
    model_decision, model_lease = scheduler.acquire(
        model, budget=budget, host=_host(), providers=[_provider()]
    )
    blocked, blocked_lease = scheduler.acquire(
        second_cpu, budget=budget, host=_host(), providers=[_provider()]
    )

    assert first_decision.admitted and first is not None
    assert model_decision.admitted and model_lease is not None
    assert blocked_lease is None
    assert set(blocked.reasons) >= {
        "cpu_proof_concurrency",
    }
    assert len(scheduler.active_leases) == 2

    assert scheduler.release(first)
    reclaimed, reclaimed_lease = scheduler.acquire(
        second_cpu, budget=budget, host=_host(), providers=[_provider()]
    )
    assert reclaimed.admitted and reclaimed_lease is not None
    assert {item.lane_id for item in scheduler.active_leases} == {"model", "cpu-2"}


def test_supervisor_lease_budget_accounts_aggregate_process_memory_disk_and_provider_use() -> None:
    scheduler = ResourceScheduler(
        ResourcePolicy(
            max_lanes=3,
            max_model_concurrency=2,
            max_cpu_proof_concurrency=1,
        )
    )
    budget = ResourceLeaseBudget(
        max_parallel=3,
        max_cpu_proof_concurrency=1,
        max_model_concurrency=2,
        max_artifact_concurrency=1,
        max_processes=1,
        memory_bytes=1_500,
        disk_bytes=1_500,
        model_token_limit=150,
        provider_quota=2,
    )
    first_requirement = LaneResourceRequirements(
        lane_id="first",
        resource_class=ProofResourceClass.MODEL_DRAFT.value,
        provider_id="provider:model",
        requires_provider=True,
        memory_bytes=1_000,
        disk_bytes=1_000,
        token_budget=100,
        quota_units=1,
    )
    second_requirement = LaneResourceRequirements(
        lane_id="second",
        resource_class=ProofResourceClass.MODEL_DRAFT.value,
        provider_id="provider:model",
        requires_provider=True,
        memory_bytes=1_000,
        disk_bytes=1_000,
        token_budget=100,
        quota_units=2,
    )
    _, first = scheduler.acquire(
        first_requirement,
        budget=budget,
        host=_host(),
        providers=[_provider()],
    )
    blocked, second = scheduler.acquire(
        second_requirement,
        budget=budget,
        host=_host(),
        providers=[_provider()],
    )

    assert first is not None
    assert second is None
    assert set(blocked.reasons) >= {
        "lease_process_capacity",
        "lease_memory_budget",
        "lease_disk_budget",
        "lease_token_budget",
        "lease_provider_quota",
    }

    scheduler.release(first)
    reclaimed, second = scheduler.acquire(
        second_requirement,
        budget=budget,
        host=_host(),
        providers=[_provider()],
    )
    assert reclaimed.admitted and second is not None


@pytest.mark.parametrize(
    ("host", "provider", "requirement", "reason"),
    [
        (_host(cpu_percent=90), _provider(), {}, "host_cpu_high_watermark"),
        (
            _host(memory_available_bytes=999),
            _provider(),
            {"memory_bytes": 1_000},
            "host_memory_headroom",
        ),
        (
            _host(disk_available_bytes=999),
            _provider(),
            {"disk_bytes": 1_000},
            "host_disk_headroom",
        ),
        (_host(), _provider(quota_remaining=0), {}, "provider_quota"),
        (_host(), _provider(context_window_tokens=99), {"context_tokens": 100}, "provider_context"),
        (
            _host(),
            _provider(token_budget_remaining=99),
            {"token_budget": 100},
            "provider_token_budget",
        ),
        (_host(), _provider(latency_ms=501), {"max_provider_latency_ms": 500}, "provider_latency"),
    ],
)
def test_live_host_and_provider_backpressure_remains_authoritative(
    host: HostResourceSnapshot,
    provider: ProviderCapacity,
    requirement: dict[str, int],
    reason: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=2))
    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id="model",
            resource_class=ProofResourceClass.MODEL_DRAFT.value,
            provider_id="provider:model",
            requires_provider=True,
            **requirement,
        ),
        host=host,
        providers=[provider],
    )
    assert not decision.admitted
    assert reason in decision.reasons


def test_proof_scheduler_propagates_one_budget_to_portfolio_and_kernel_children(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _step(
            "solver",
            ProofStage.SOLVE,
            metadata={"portfolio_width": 9},
        ),
        _step(
            "kernel",
            ProofStage.KERNEL_VERIFY,
            metadata={"kernel_max_parallel": 9},
        ),
        max_parallel=4,
    )
    budget = ResourceLeaseBudget.from_resource_budget(
        plan.resource_budget,
        max_parallel=2,
        max_cpu_proof_concurrency=2,
        max_model_concurrency=1,
        max_artifact_concurrency=1,
    )
    observed: dict[str, dict[str, int]] = {}

    def execute(context) -> None:
        assert context.resource_budget is budget
        observed[context.step_id] = dict(context.child_limits)

    result = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        resource_lease_budget=budget,
        resource_policy=ResourcePolicy(
            max_lanes=2,
            max_cpu_proof_concurrency=2,
            require_provider_telemetry=True,
        ),
        host_resource_source=_host(),
    ).run()

    assert result.succeeded
    assert observed["solver"]["max_processes"] == 2
    assert observed["solver"]["portfolio_max_parallel"] == 2
    assert observed["solver"]["kernel_max_parallel"] == 1
    assert observed["kernel"]["max_processes"] == 2
    assert observed["kernel"]["kernel_max_parallel"] == 2
    assert observed["kernel"]["portfolio_max_parallel"] == 1
    assert observed["solver"]["memory_bytes"] == plan.resource_budget.memory_bytes
    assert observed["kernel"]["disk_bytes"] == plan.resource_budget.disk_bytes


def test_model_concurrency_is_separate_from_cpu_proof_concurrency(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _step("model", ProofStage.MODEL_DRAFT, metadata={"token_budget": 100}),
        _step("translate-a", ProofStage.TRANSLATE),
        _step("translate-b", ProofStage.TRANSLATE),
        max_parallel=3,
    )
    lock = threading.Lock()
    active_cpu = 0
    peak_cpu = 0
    model_and_cpu_overlap = threading.Event()

    def execute(context) -> None:
        nonlocal active_cpu, peak_cpu
        is_model = context.resource_class == ProofResourceClass.MODEL_DRAFT.value
        with lock:
            if not is_model:
                active_cpu += 1
                peak_cpu = max(peak_cpu, active_cpu)
            if is_model and active_cpu:
                model_and_cpu_overlap.set()
        # Give the independently admitted class time to enter.
        for _ in range(20):
            if model_and_cpu_overlap.is_set():
                break
            time.sleep(0.005)
            if not is_model:
                with lock:
                    if any(
                        lease.resource_pool == "model"
                        for lease in scheduler.resource_scheduler.active_leases
                    ):
                        model_and_cpu_overlap.set()
        time.sleep(0.02)
        if not is_model:
            with lock:
                active_cpu -= 1

    scheduler = ProofScheduler(
        plan,
        execute,
        state_path=tmp_path,
        config=ProofSchedulerConfig(
            max_parallel=2,
            lease_seconds=2,
            max_cpu_proof_concurrency=1,
            max_model_concurrency=1,
        ),
        resource_policy=ResourcePolicy(
            max_lanes=2,
            max_cpu_proof_concurrency=1,
            max_model_concurrency=1,
        ),
        host_resource_source=_host(worker_limit=2, available_worker_capacity=2),
        provider_capacity_source=[_provider(max_concurrency=1)],
    )
    result = scheduler.run()

    assert result.succeeded
    assert peak_cpu == 1
    assert model_and_cpu_overlap.is_set()
    assert scheduler.resource_scheduler.active_leases == ()


def test_proof_scheduler_waits_for_provider_capacity_before_claiming(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _step("model", ProofStage.MODEL_DRAFT, metadata={"token_budget": 100}),
        max_parallel=1,
    )
    samples = 0
    executed = threading.Event()

    def providers() -> list[ProviderCapacity]:
        nonlocal samples
        samples += 1
        return [_provider(quota_remaining=0 if samples < 3 else 2)]

    scheduler = ProofScheduler(
        plan,
        lambda _context: executed.set(),
        state_path=tmp_path,
        resource_policy=ResourcePolicy(max_lanes=1),
        host_resource_source=_host(worker_limit=1, available_worker_capacity=1),
        provider_capacity_source=providers,
    )
    result = scheduler.run(timeout_seconds=2)

    assert result.succeeded
    assert executed.is_set()
    assert samples >= 3
    assert scheduler.resource_decisions["model"].admitted
