from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import DynamicBundleScheduler
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import (
    LeaseCoordinator,
    profile_g_cid,
)
from ipfs_accelerate_py.agent_supervisor.merge.leased_lane import (
    LeasedLaneResult,
    run_leased_lane_result,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    GoalRuntimeResourceScheduler,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProofResourceClass,
    ProofWorkKind,
    ProofWorkRequest,
    ProofWorkStatus,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
    normalize_provider_capacity,
    resource_class_for_work_kind,
    sample_host_resources,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive


def _host(**overrides: object) -> HostResourceSnapshot:
    values: dict[str, object] = {
        "observed_at_ms": 1_000,
        "cpu_percent": 20,
        "memory_percent": 25,
        "disk_percent": 30,
        "memory_available_bytes": 8_000,
        "disk_available_bytes": 16_000,
        "active_phase": "scheduler",
        "active_workers": 0,
        "worker_limit": 4,
        "available_worker_capacity": 4,
        "capabilities": ("cpu", "git"),
        "resource_classes": ("cpu-small", "cpu-medium"),
    }
    values.update(overrides)
    return HostResourceSnapshot(**values)  # type: ignore[arg-type]


def _provider(provider_id: str = "provider-a", **overrides: object) -> ProviderCapacity:
    values: dict[str, object] = {
        "provider_id": provider_id,
        "healthy": True,
        "quota_remaining": 100,
        "latency_ms": 50,
        "context_window_tokens": 32_000,
        "token_budget_remaining": 50_000,
        "max_concurrency": 4,
        "active_requests": 0,
        "capabilities": ("json", "tools"),
        "observed_at_ms": 1_000,
    }
    values.update(overrides)
    return ProviderCapacity(**values)  # type: ignore[arg-type]


def _llm_lane(lane_id: str = "lane-a", **overrides: object) -> LaneResourceRequirements:
    values: dict[str, object] = {
        "lane_id": lane_id,
        "resource_class": "cpu-small",
        "required_capabilities": ("llm:json",),
        "requires_provider": True,
        "context_tokens": 8_000,
        "token_budget": 2_000,
        "quota_units": 1,
    }
    values.update(overrides)
    return LaneResourceRequirements(**values)  # type: ignore[arg-type]


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return None if self.alive else self.returncode


def _write_bundle_index(path: Path, count: int, *, llm: bool = False) -> None:
    bundles: dict[str, object] = {}
    for index in range(1, count + 1):
        task: dict[str, object] = {"task_id": f"RES-{index}"}
        if llm:
            task.update(
                {
                    "required_capabilities": ["json"],
                    "required_context_tokens": 4_000,
                    "token_budget": 1_000,
                }
            )
        bundles[f"objective/resources/{index}"] = {
            "shard_path": f"resources-{index}.todo.md",
            "parallel_lane": f"resources-{index}",
            "tasks": [task],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source_todo": "tasks.todo.md", "bundles": bundles}), encoding="utf-8")


def test_sample_host_resources_reports_measured_cpu_memory_disk_and_worker_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda *, interval=None: 37.4,
        virtual_memory=lambda: SimpleNamespace(percent=62.6, available=4_096, total=16_384),
        disk_usage=lambda _path: SimpleNamespace(percent=71.2, free=8_192, total=32_768),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    snapshot = sample_host_resources(
        tmp_path,
        active_workers=2,
        worker_limit=5,
        active_phase="validation",
    )

    assert snapshot.cpu_percent == 37
    assert snapshot.memory_percent == 63
    assert snapshot.disk_percent == 71
    assert snapshot.memory_available_bytes == 4_096
    assert snapshot.disk_available_bytes == 8_192
    assert snapshot.memory_total_bytes == 16_384
    assert snapshot.disk_total_bytes == 32_768
    assert snapshot.active_phase == "validation"
    assert snapshot.occupied_worker_capacity == 2
    assert snapshot.available_worker_capacity == 3
    assert snapshot.to_dict()["occupied_worker_capacity"] == 2


@pytest.mark.parametrize(
    ("host_overrides", "requirement_overrides", "reason"),
    [
        ({"cpu_percent": 90}, {}, "host_cpu_high_watermark"),
        ({"memory_percent": 90}, {}, "host_memory_high_watermark"),
        ({"disk_percent": 95}, {}, "host_disk_high_watermark"),
        ({"memory_available_bytes": 999}, {"memory_bytes": 1_000}, "host_memory_headroom"),
        ({"disk_available_bytes": 999}, {"disk_bytes": 1_000}, "host_disk_headroom"),
        ({"resource_classes": ("cpu-small",)}, {"resource_class": "gpu"}, "resource_class_mismatch"),
        ({"active_workers": 4, "available_worker_capacity": 0}, {}, "host_worker_capacity"),
    ],
)
def test_host_pressure_applies_backpressure_before_exhaustion(
    host_overrides: dict[str, object],
    requirement_overrides: dict[str, object],
    reason: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))
    decision = scheduler.evaluate(
        LaneResourceRequirements(lane_id="host-only", **requirement_overrides),
        host=_host(**host_overrides),
    )

    assert decision.admitted is False
    assert reason in decision.reasons
    assert decision.effective_slots == 0


@pytest.mark.parametrize(
    ("advertised_resource_classes", "required_resource_class"),
    [
        (("cpu-small",), "cpu-proof-sanitize"),
        ((ProofResourceClass.TYPE_CHECK.value,), "cpu-install-test"),
    ],
)
def test_cpu_extension_resource_classes_use_advertised_local_cpu_capacity(
    advertised_resource_classes: tuple[str, ...],
    required_resource_class: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))

    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id=required_resource_class,
            resource_class=required_resource_class,
        ),
        host=_host(resource_classes=advertised_resource_classes),
    )

    assert decision.admitted is True
    assert "resource_class_mismatch" not in decision.reasons


def test_cpu_extension_resource_classes_do_not_bypass_host_capabilities() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))

    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id="toolchain",
            resource_class="cpu-install-test",
            required_capabilities=("host:container-runtime",),
        ),
        host=_host(resource_classes=(ProofResourceClass.VALIDATION.value,)),
    )

    assert decision.admitted is False
    assert "resource_class_mismatch" not in decision.reasons
    assert "host_capability_mismatch" in decision.reasons


@pytest.mark.parametrize(
    ("advertised_resource_class", "required_resource_class"),
    [
        (ProofResourceClass.SOLVER.value, "exclusive-jvm-toolchain"),
        (ProofResourceClass.SOLVER.value, "jvm-proof-solver"),
        (ProofResourceClass.SOLVER.value, "exclusive-opam-toolchain"),
        (ProofResourceClass.KERNEL.value, "large-kernel-toolchain"),
    ],
)
def test_planner_local_cpu_toolchain_classes_use_cpu_proof_capacity(
    advertised_resource_class: str,
    required_resource_class: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))

    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id=required_resource_class,
            resource_class=required_resource_class,
        ),
        host=_host(resource_classes=(advertised_resource_class,)),
    )

    assert decision.admitted is True
    assert decision.resource_pool == "cpu-proof"
    assert "resource_class_mismatch" not in decision.reasons


@pytest.mark.parametrize(
    "required_resource_class",
    ["gpu", "provider-toolchain", "unknown-toolchain"],
)
def test_unrecognized_non_cpu_resource_classes_remain_fail_closed(
    required_resource_class: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))

    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id=required_resource_class,
            resource_class=required_resource_class,
        ),
        host=_host(resource_classes=(ProofResourceClass.SOLVER.value,)),
    )

    assert decision.admitted is False
    assert "resource_class_mismatch" in decision.reasons


@pytest.mark.parametrize(
    "resource_class",
    ["exclusive-jvm-toolchain", "exclusive-opam-toolchain"],
)
def test_exclusive_planner_resource_classes_serialize_by_default(
    resource_class: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))

    snapshot = scheduler.schedule(
        (
            LaneResourceRequirements(
                lane_id="toolchain-one",
                resource_class=resource_class,
            ),
            LaneResourceRequirements(
                lane_id="toolchain-two",
                resource_class=resource_class,
            ),
        ),
        host=_host(
            resource_classes=(ProofResourceClass.SOLVER.value,),
        ),
    )

    first = snapshot.decision_for("toolchain-one")
    second = snapshot.decision_for("toolchain-two")
    assert first is not None and first.admitted is True
    assert second is not None and second.admitted is False
    assert "resource_class_concurrency" in second.reasons


def test_explicit_exclusive_resource_class_limit_overrides_default() -> None:
    scheduler = ResourceScheduler(
        ResourcePolicy(
            max_lanes=4,
            resource_class_limits={"exclusive-opam-toolchain": 2},
        )
    )

    snapshot = scheduler.schedule(
        (
            LaneResourceRequirements(
                lane_id="opam-one",
                resource_class="exclusive-opam-toolchain",
            ),
            LaneResourceRequirements(
                lane_id="opam-two",
                resource_class="exclusive-opam-toolchain",
            ),
        ),
        host=_host(
            resource_classes=(ProofResourceClass.SOLVER.value,),
        ),
    )

    assert snapshot.admitted_lane_ids == ("opam-one", "opam-two")


@pytest.mark.parametrize(
    ("provider_overrides", "requirement_overrides", "policy_overrides", "reason"),
    [
        ({"healthy": False}, {}, {}, "provider_unhealthy"),
        ({"quota_remaining": 1}, {"quota_units": 1}, {"provider_quota_reserve": 1}, "provider_quota"),
        ({"latency_ms": 501}, {}, {"maximum_provider_latency_ms": 500}, "provider_latency"),
        ({"context_window_tokens": 7_999}, {"context_tokens": 8_000}, {}, "provider_context"),
        (
            {"token_budget_remaining": 2_000},
            {"token_budget": 2_000},
            {"provider_token_reserve": 1},
            "provider_token_budget",
        ),
        ({"max_concurrency": 2, "active_requests": 2}, {}, {}, "provider_concurrency"),
        ({"capabilities": ("text",)}, {}, {}, "provider_capability_mismatch"),
        ({"retry_after_ms": 1}, {}, {}, "provider_backoff"),
    ],
)
def test_provider_constraints_are_all_hard_admission_gates(
    provider_overrides: dict[str, object],
    requirement_overrides: dict[str, object],
    policy_overrides: dict[str, object],
    reason: str,
) -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4, **policy_overrides))
    decision = scheduler.evaluate(
        _llm_lane(**requirement_overrides),
        host=_host(),
        providers=[_provider(**provider_overrides)],
    )

    assert decision.admitted is False
    assert reason in decision.reasons
    assert decision.provider_id == "provider-a"


def test_provider_selection_is_lowest_latency_then_stable_identity() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))
    providers = [
        _provider("provider-z", latency_ms=10),
        _provider("provider-b", latency_ms=5),
        _provider("provider-a", latency_ms=5),
    ]

    decision = scheduler.evaluate(_llm_lane(), host=_host(), providers=providers)

    assert decision.admitted is True
    assert decision.provider_id == "provider-a"
    assert decision.capability_fit_millionths == 1_000_000
    assert decision.provider_available_slots == 4


def test_explicit_provider_requirement_never_falls_through_to_another_provider() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=2))
    providers = [
        _provider("required", healthy=False),
        _provider("healthy-fallback", healthy=True),
    ]

    decision = scheduler.evaluate(
        _llm_lane(provider_id="required"),
        host=_host(worker_limit=2, available_worker_capacity=2),
        providers=providers,
    )

    assert decision.admitted is False
    assert decision.provider_id == "required"
    assert "provider_unhealthy" in decision.reasons


def test_provider_telemetry_aliases_preserve_zero_as_exhausted() -> None:
    provider = normalize_provider_capacity(
        {
            "provider": "router-provider",
            "status": "ready",
            "remaining_quota": 0,
            "avg_latency_ms": 123,
            "max_context_tokens": 16_384,
            "remaining_tokens": 0,
            "active_requests": 2,
            "available_concurrency": 3,
            "features": ["JSON", "tools"],
        }
    )

    assert provider.provider_id == "router-provider"
    assert provider.healthy is True
    assert provider.quota_remaining == 0
    assert provider.token_budget_remaining == 0
    assert provider.latency_ms == 123
    assert provider.context_window_tokens == 16_384
    assert provider.max_concurrency == 5
    assert provider.available_concurrency == 3
    assert provider.capabilities == ("json", "tools")


def test_schedule_reserves_provider_and_host_capacity_in_input_priority_order() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))
    lanes = [_llm_lane(f"lane-{index}", token_budget=10) for index in range(1, 5)]
    provider = _provider(
        max_concurrency=2,
        active_requests=1,
        quota_remaining=10,
        token_budget_remaining=100,
    )

    schedule = scheduler.schedule(
        lanes,
        host=_host(worker_limit=3, active_workers=1, available_worker_capacity=2),
        providers=[provider],
    )

    assert schedule.configured_max_lanes == 4
    # Effective capacity is bounded by the single free provider request slot,
    # not merely by the host's three-worker pool.
    assert schedule.effective_slots == 1
    assert schedule.admitted_lane_ids == ("lane-1",)
    assert schedule.admitted_count == 1
    assert schedule.available_slots == 0
    assert [decision.reason for decision in schedule.decisions] == [
        "",
        "provider_concurrency",
        "provider_concurrency",
        "provider_concurrency",
    ]
    assert "provider_concurrency" in schedule.backpressure_reasons


def test_provider_capacity_does_not_count_provider_free_inference_work() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=4))
    cpu_implementation = LaneResourceRequirements(
        lane_id="cpu-implementation",
        stage="inference",
        resource_class="cpu-medium",
    )

    schedule = scheduler.schedule(
        [_llm_lane("model-adapter", stage="inference")],
        host=_host(worker_limit=4, available_worker_capacity=4),
        providers=[_provider(max_concurrency=1)],
        active_requirements=[cpu_implementation],
    )

    assert schedule.admitted_lane_ids == ("model-adapter",)
    assert schedule.decisions[0].provider_id == "provider-a"
    assert schedule.decisions[0].reason == ""
    assert schedule.stage_capacities[0].stage == "inference"
    assert schedule.stage_capacities[0].provider_available_slots == -1


def test_schedule_accumulates_quota_and_token_reservations() -> None:
    scheduler = ResourceScheduler(
        ResourcePolicy(
            max_lanes=3,
            provider_quota_reserve=1,
            provider_token_reserve=5,
        )
    )
    provider = _provider(
        max_concurrency=3,
        quota_remaining=5,
        token_budget_remaining=25,
    )
    lanes = [
        _llm_lane("first", quota_units=2, token_budget=10),
        _llm_lane("second", quota_units=2, token_budget=10),
        _llm_lane("third", quota_units=2, token_budget=10),
    ]

    schedule = scheduler.schedule(lanes, host=_host(worker_limit=3, available_worker_capacity=3), providers=[provider])

    assert schedule.admitted_lane_ids == ("first", "second")
    assert schedule.decisions[2].admitted is False
    assert set(schedule.decisions[2].reasons) >= {"provider_quota", "provider_token_budget"}


def test_non_llm_lane_remains_schedulable_without_provider_telemetry() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=1, require_provider_telemetry=True))

    decision = scheduler.evaluate(
        LaneResourceRequirements(
            lane_id="deterministic",
            resource_class="cpu-small",
            required_capabilities=("git",),
        ),
        host=_host(worker_limit=1, available_worker_capacity=1),
        providers=None,
    )

    assert decision.admitted is True
    assert decision.provider_id == ""


def test_provider_lane_is_backpressured_when_telemetry_is_missing() -> None:
    scheduler = ResourceScheduler(ResourcePolicy(max_lanes=1, require_provider_telemetry=True))

    decision = scheduler.evaluate(
        _llm_lane(),
        host=_host(worker_limit=1, available_worker_capacity=1),
        providers=None,
    )

    assert decision.admitted is False
    assert decision.reason == "provider_telemetry_unavailable"


def test_dynamic_scheduler_applies_host_backpressure_before_claiming_a_lease(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_bundle_index(index, 2)
    starts: list[object] = []

    def launch(lane: object, _grant: object) -> _Process:
        starts.append(lane)
        return _Process(9_000 + len(starts))

    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        coordination_path=repo / "coordination.sqlite3",
        max_lanes=2,
        launcher=launch,
        process_alive=lambda process: process.alive,
        host_resource_source=lambda *_args, **_kwargs: _host(
            cpu_percent=95,
            worker_limit=2,
            available_worker_capacity=2,
        ),
        poll_interval=0,
    )

    manifest = scheduler.reconcile_once()

    assert starts == []
    assert manifest["capacity"] == 2
    assert manifest["effective_capacity"] == 0
    assert manifest["available_worker_capacity"] == 0
    assert manifest["backpressure_reasons"] == ["host_cpu_high_watermark"]
    assert manifest["resource_schedule"]["host"]["cpu_percent"] == 95
    deferred = [item for item in manifest["scheduler_decisions"] if item["decision"] == "deferred"]
    assert len(deferred) == 2
    assert all(item["reason"] == "host_cpu_high_watermark" for item in deferred)
    assert all("resource_admission" in item for item in deferred)
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        assert all(item["state"] != "accepted" for item in coordinator.list_tasks())


def test_dynamic_scheduler_caps_starts_at_provider_capacity_and_records_selection(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_bundle_index(index, 3, llm=True)
    starts: list[tuple[object, object, _Process]] = []

    def launch(lane: object, grant: object) -> _Process:
        process = _Process(10_000 + len(starts))
        starts.append((lane, grant, process))
        return process

    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        coordination_path=repo / "coordination.sqlite3",
        max_lanes=3,
        launcher=launch,
        process_alive=lambda process: process.alive,
        host_resource_source=lambda *_args, **_kwargs: _host(
            worker_limit=3,
            available_worker_capacity=3,
        ),
        provider_capacity_source=lambda: [
            _provider(
                "provider-fast",
                latency_ms=25,
                max_concurrency=2,
                active_requests=1,
            )
        ],
        poll_interval=0,
    )

    manifest = scheduler.reconcile_once()

    assert len(starts) == 1
    selected_lane = starts[0][0]
    assert selected_lane.llm_provider == "provider-fast"
    assert manifest["effective_capacity"] == 1
    assert manifest["running_count"] == 1
    assert manifest["resource_schedule"]["admitted_count"] == 1
    assert manifest["resource_schedule"]["providers"][0]["provider_id"] == "provider-fast"
    launched = [item for item in manifest["scheduler_decisions"] if item["decision"] == "launched"]
    assert len(launched) == 1
    assert launched[0]["resource_admission"]["provider_id"] == "provider-fast"
    deferred = [item for item in manifest["scheduler_decisions"] if item["decision"] == "deferred"]
    assert len(deferred) == 2
    assert all(item["reason"] == "provider_concurrency" for item in deferred)
    with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
        accepted = [item for item in coordinator.list_tasks() if item["state"] == "accepted"]
    assert len(accepted) == 1


def test_enhanced_heartbeat_round_trips_latest_live_resource_and_provider_capacity(
    tmp_path: Path,
) -> None:
    now = 10_000
    bundle = {
        "bundle_key": "objective/resources/heartbeat",
        "tasks": [{"task_id": "RES-1"}],
    }
    path = tmp_path / "coordination.sqlite3"
    with LeaseCoordinator(path, clock_ms=lambda: now) as coordinator:
        task = coordinator.register_bundle(bundle, created_at_ms=now)
        grant = coordinator.claim(
            task["task_cid"],
            "did:web:resource-worker.example",
            requested_lease_ms=20_000,
            now_ms=now,
        )
        first = coordinator.heartbeat(
            grant,
            capacity_millionths=1_000_000,
            ttl_ms=5_000,
            now_ms=now + 1,
            active_phase="implementation",
            cpu_percent=42,
            memory_percent=51,
            disk_percent=63,
            memory_available_bytes=4_096,
            disk_available_bytes=8_192,
            occupied_workers=1,
            available_workers=0,
            resource_class="cpu-medium",
            provider_id="codex",
            provider_capacity={"healthy": True, "available_concurrency": 2},
            detail={"source": "measured"},
        )
        idle = coordinator.heartbeat(
            grant,
            capacity_millionths=0,
            ttl_ms=5_000,
            now_ms=now + 2,
            active_phase="idle",
            cpu_percent=5,
            memory_percent=20,
            disk_percent=63,
            occupied_workers=0,
            available_workers=1,
            resource_class="cpu-medium",
            provider_id="codex",
            provider_capacity={"healthy": True, "available_concurrency": 3},
        )

        assert first["heartbeat_cid"] != idle["heartbeat_cid"]
        latest = coordinator.latest_heartbeat(task["task_cid"], now_ms=now + 3)
        assert latest is not None
        assert latest["heartbeat_cid"] == idle["heartbeat_cid"]
        assert latest["active_phase"] == "idle"
        assert latest["capacity_millionths"] == 0
        assert latest["occupied_workers"] == 0
        assert latest["available_workers"] == 1
        assert latest["provider_capacity"]["available_concurrency"] == 3
        assert coordinator.latest_heartbeats(provider_id="codex", now_ms=now + 3) == [latest]
        assert coordinator.latest_heartbeat(task["task_cid"], now_ms=now + 5_003) is None
        historical = coordinator.latest_heartbeat(
            task["task_cid"],
            include_expired=True,
            now_ms=now + 5_003,
        )
        assert historical == latest


def test_enhanced_heartbeat_rejects_noncanonical_resource_telemetry(tmp_path: Path) -> None:
    now = 20_000
    with LeaseCoordinator(tmp_path / "coordination.sqlite3", clock_ms=lambda: now) as coordinator:
        task = coordinator.register_bundle(
            {"bundle_key": "objective/resources/canonical", "tasks": [{"task_id": "RES-2"}]},
            created_at_ms=now,
        )
        grant = coordinator.claim(task["task_cid"], "did:web:worker.example", now_ms=now)

        with pytest.raises(ValueError, match="cannot contain floats"):
            coordinator.heartbeat(
                grant,
                capacity_millionths=0,
                provider_capacity={"latency_ms": 1.5},
                now_ms=now + 1,
            )
        assert coordinator.latest_heartbeat(task["task_cid"], now_ms=now + 2) is None


def test_leased_lane_measures_active_resources_then_advertises_idle_capacity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coordination.sqlite3"
    with LeaseCoordinator(path) as coordinator:
        task = coordinator.register_bundle(
            {"bundle_key": "objective/resources/lane", "tasks": [{"task_id": "RES-3"}]}
        )
        grant = coordinator.claim(task["task_cid"], "did:web:measured-worker.example")

    result = run_leased_lane_result(
        coordination_path=path,
        grant=grant,
        command=[sys.executable, "-c", "pass"],
        lease_ms=60_000,
        heartbeat_interval=0.05,
        resource_class="cpu-small",
        provider_id="provider-a",
    )

    assert result.successful
    with LeaseCoordinator(path) as coordinator:
        latest = coordinator.latest_heartbeat(task["task_cid"])
    assert latest is not None
    assert latest["active_phase"] == "idle"
    assert latest["occupied_workers"] == 0
    assert latest["available_workers"] == 1
    assert latest["capacity_millionths"] == 0
    assert latest["resource_class"] == "cpu-small"
    assert latest["provider_id"] == "provider-a"
    assert {"cpu_percent", "memory_percent", "disk_percent"} <= latest.keys()


def test_leased_lane_drains_freshly_completed_slice_and_releases_provider_claim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coordination.sqlite3"
    phase_state = tmp_path / "phase-state.json"
    events_path = tmp_path / "events.jsonl"
    child_pid_path = tmp_path / "child.pid"
    task_cids_by_id = {
        task_id: profile_g_cid({"member": task_id})
        for task_id in ("RES-LONG-1", "RES-LONG-1B")
    }
    first_bundle = {
        "bundle_key": "objective/resources/long-lived-first",
        "tasks": [
            {
                "task_id": task_id,
                "canonical_task_cid": task_cids_by_id[task_id],
            }
            for task_id in task_cids_by_id
        ],
        "execution_slice_task_ids": ["RES-LONG-1", "RES-LONG-1B"],
        "execution_slice_task_cids": list(task_cids_by_id.values()),
    }
    with LeaseCoordinator(path) as coordinator:
        first = coordinator.register_bundle(first_bundle)
        first_grant = coordinator.claim(
            first["task_cid"],
            "did:web:provider-worker.example",
            requested_lease_ms=60_000,
        )

    def write_phase_state(payload: dict[str, object]) -> None:
        temporary = phase_state.with_name(f".{phase_state.name}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(phase_state)

    # A prior lease's terminal projection must not terminate this child.
    write_phase_state(
        {
            "heartbeat_at": "2000-01-01T00:00:00+00:00",
            "active_task_id": "",
            "implementation_in_progress": False,
            "completed_task_ids": ["RES-LONG-1", "RES-LONG-1B"],
            "task_identities": {
                task_id: {"canonical_task_cid": task_cid}
                for task_id, task_cid in task_cids_by_id.items()
            },
        }
    )
    outcome: list[LeasedLaneResult] = []
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            outcome.append(
                run_leased_lane_result(
                    coordination_path=path,
                    grant=first_grant,
                    command=[
                        sys.executable,
                        "-c",
                        (
                            "import os,pathlib,sys,time;"
                            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
                            "time.sleep(60)"
                        ),
                        str(child_pid_path),
                    ],
                    lease_ms=60_000,
                    heartbeat_interval=0.02,
                    resource_class="model",
                    provider_id="provider-a",
                    phase_state_path=phase_state,
                    expected_task_ids=("RES-LONG-1", "RES-LONG-1B"),
                    expected_task_cids_by_id=task_cids_by_id,
                    completion_events_path=events_path,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    lane_thread = threading.Thread(target=execute, daemon=True)
    lane_thread.start()

    deadline = time.monotonic() + 5
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.exists()
    time.sleep(0.15)
    assert lane_thread.is_alive()

    try:
        active_at = datetime.now(timezone.utc).isoformat()
        write_phase_state(
            {
                "heartbeat_at": active_at,
                "active_task_id": "RES-LONG-1",
                "implementation_in_progress": True,
                "completed_task_ids": [],
                "task_identities": {
                    task_id: {"canonical_task_cid": task_cid}
                    for task_id, task_cid in task_cids_by_id.items()
                },
            }
        )
        deadline = time.monotonic() + 5
        observed_active = False
        while time.monotonic() < deadline:
            with LeaseCoordinator(path) as coordinator:
                heartbeat = coordinator.latest_heartbeat(first["task_cid"])
            if (
                heartbeat is not None
                and heartbeat.get("active_phase") == "implementation"
            ):
                observed_active = True
                break
            time.sleep(0.01)
        assert observed_active

        incomplete_at = datetime.now(timezone.utc).isoformat()
        write_phase_state(
            {
                "heartbeat_at": incomplete_at,
                "active_task_id": "",
                "implementation_in_progress": False,
                "completed_task_ids": ["RES-LONG-1"],
                "task_identities": {
                    task_id: {"canonical_task_cid": task_cid}
                    for task_id, task_cid in task_cids_by_id.items()
                },
            }
        )
        time.sleep(0.15)
        assert lane_thread.is_alive()

        completed_at = datetime.now(timezone.utc).isoformat()
        write_phase_state(
            {
                "heartbeat_at": completed_at,
                "active_task_id": "",
                "implementation_in_progress": False,
                "completed_task_ids": ["RES-LONG-1", "RES-LONG-1B"],
                "task_identities": {
                    "RES-LONG-1": {
                        "canonical_task_cid": task_cids_by_id["RES-LONG-1"],
                    },
                    "RES-LONG-1B": {
                        "canonical_task_cid": profile_g_cid(
                            {"wrong-member": "RES-LONG-1B"}
                        ),
                    },
                },
            }
        )
        time.sleep(0.15)
        assert lane_thread.is_alive()

        write_phase_state(
            {
                "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "active_task_id": "",
                "implementation_in_progress": False,
                "completed_task_ids": ["RES-LONG-1", "RES-LONG-1B"],
                "task_identities": {
                    task_id: {"canonical_task_cid": task_cid}
                    for task_id, task_cid in task_cids_by_id.items()
                },
            }
        )
        # Exact phase state alone is not durable completion authority for a
        # bundle lane whose member event-log path is supplied.
        time.sleep(0.15)
        assert lane_thread.is_alive()
        append_jsonl_event(
            events_path,
            "todo_status_updated",
            {
                "updated": True,
                "updated_task_ids": list(task_cids_by_id),
                "completion_receipts": [
                    {
                        "schema": (
                            "ipfs_accelerate_py.agent_supervisor."
                            "member_completion_receipt@1"
                        ),
                        "task_id": task_id,
                        "canonical_task_cid": (
                            profile_g_cid({"wrong-receipt-member": task_id})
                            if task_id == "RES-LONG-1B"
                            else task_cid
                        ),
                        "status": "succeeded",
                    }
                    for task_id, task_cid in task_cids_by_id.items()
                ],
            },
        )
        time.sleep(0.15)
        assert lane_thread.is_alive()
        append_jsonl_event(
            events_path,
            "todo_status_updated",
            {
                "updated": True,
                "updated_task_ids": list(task_cids_by_id),
                "completion_receipts": [
                    {
                        "schema": (
                            "ipfs_accelerate_py.agent_supervisor."
                            "member_completion_receipt@1"
                        ),
                        "task_id": task_id,
                        "canonical_task_cid": task_cid,
                        "status": "succeeded",
                    }
                    for task_id, task_cid in task_cids_by_id.items()
                ],
            },
        )
        lane_thread.join(10)
    finally:
        if lane_thread.is_alive() and child_pid_path.exists():
            try:
                os.kill(int(child_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except (OSError, ValueError):
                pass
            lane_thread.join(5)

    assert errors == []
    assert not lane_thread.is_alive()
    assert len(outcome) == 1
    result = outcome[0]
    assert result.successful
    assert result.disposition == "completed"
    assert result.exit_code == 0
    assert result.child_exit_code not in {None, 0}
    assert result.receipt_cid
    assert result.lease_released is True

    with LeaseCoordinator(path) as coordinator:
        latest = coordinator.latest_heartbeat(first["task_cid"])
        receipts = coordinator.list_receipts(first["task_cid"])
        assert coordinator.active_lease(first["task_cid"]) is None
    assert latest is not None
    assert latest["capacity_millionths"] == 0
    assert latest["active_phase"] == "idle"
    assert latest["occupied_workers"] == 0
    assert latest["available_workers"] == 1
    assert latest["provider_id"] == "provider-a"
    assert len(receipts) == 1
    assert receipts[0]["receipt"]["status"] == "succeeded"
    assert receipts[0]["receipt"]["failure_class"] == "none"
    assert receipts[0]["receipt"]["output_cid"]


def test_leased_lane_rejects_display_only_execution_slice_identity(
    tmp_path: Path,
) -> None:
    coordination_path = tmp_path / "coordination.sqlite3"
    with LeaseCoordinator(coordination_path) as coordinator:
        task = coordinator.register_bundle(
            {
                "bundle_key": "objective/resources/unbound-slice",
                "tasks": [{"task_id": "RES-UNBOUND"}],
            }
        )
        grant = coordinator.claim(
            task["task_cid"],
            "did:web:unbound-worker.example",
        )

    with pytest.raises(ValueError, match="exact expected_task_cids_by_id"):
        run_leased_lane_result(
            coordination_path=coordination_path,
            grant=grant,
            command=(sys.executable, "-c", "pass"),
            lease_ms=60_000,
            heartbeat_interval=0.02,
            expected_task_ids=("RES-UNBOUND",),
        )


def test_dynamic_scheduler_reuses_provider_after_completed_slice_wrapper_exits(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_bundle_index(index, 2, llm=True)
    child_pid_paths = {
        "RES-1": tmp_path / "res-1.pid",
        "RES-2": tmp_path / "res-2.pid",
    }
    starts: list[tuple[object, object, object]] = []

    class ThreadedLaneProcess:
        def __init__(self, lane: object, grant: object) -> None:
            self.lane = lane
            self.grant = grant
            self.result: LeasedLaneResult | None = None
            self.error: BaseException | None = None
            self.pid = 20_000 + len(starts)
            self.thread = threading.Thread(target=self._execute, daemon=True)
            self.thread.start()

        @property
        def task_id(self) -> str:
            return self.lane.task_ids[0]

        @property
        def phase_state_path(self) -> Path:
            return (
                self.lane.state_dir
                / f"{self.lane.state_prefix}_task_state.json"
            )

        def _execute(self) -> None:
            self.lane.state_dir.mkdir(parents=True, exist_ok=True)
            task_id = self.task_id
            command = (
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,pathlib,sys,time;"
                        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
                        "time.sleep(60)"
                    ),
                    str(child_pid_paths[task_id]),
                    ]
                    if task_id == "RES-1"
                    else [
                        sys.executable,
                        "-c",
                        (
                            "import json,pathlib,sys;"
                            "from datetime import datetime,timezone;"
                            "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                            "'heartbeat_at':datetime.now(timezone.utc).isoformat(),"
                            "'active_task_id':'',"
                            "'implementation_in_progress':False,"
                            "'completed_task_ids':[sys.argv[2]],"
                            "'task_identities':{sys.argv[2]:{"
                            "'canonical_task_cid':sys.argv[3]}}"
                            "}),encoding='utf-8')"
                        ),
                        str(self.phase_state_path),
                        task_id,
                        self.lane.expected_task_cids_by_id[task_id],
                    ]
                )
            try:
                self.result = run_leased_lane_result(
                    coordination_path=repo / "coordination.sqlite3",
                    grant=self.grant,
                    command=command,
                    lease_ms=60_000,
                    heartbeat_interval=0.02,
                    capacity_millionths=1_000_000,
                    resource_class=self.lane.resource_class,
                    provider_id=self.lane.llm_provider,
                    phase_state_path=self.phase_state_path,
                    expected_task_ids=tuple(self.lane.task_ids),
                    expected_task_cids_by_id=self.lane.expected_task_cids_by_id,
                )
            except BaseException as exc:  # pragma: no cover - surfaced by wait
                self.error = exc

        def poll(self) -> int | None:
            if self.thread.is_alive():
                return None
            if self.error is not None:
                return 1
            return int(self.result.exit_code if self.result is not None else 1)

        def wait(self, timeout: float | None = None) -> int:
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            return int(self.poll() or 0)

        def terminate(self) -> None:
            self._signal_child(signal.SIGTERM)

        def kill(self) -> None:
            self._signal_child(signal.SIGKILL)

        def _signal_child(self, signum: int) -> None:
            pid_path = child_pid_paths[self.task_id]
            if not pid_path.exists():
                return
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8")), signum)
            except (OSError, ValueError):
                pass

    def launch(lane: object, grant: object) -> ThreadedLaneProcess:
        process = ThreadedLaneProcess(lane, grant)
        starts.append((lane, grant, process))
        return process

    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        coordination_path=repo / "coordination.sqlite3",
        max_lanes=2,
        launcher=launch,
        process_alive=lambda process: process.poll() is None,
        host_resource_source=lambda *_args, **_kwargs: _host(
            worker_limit=2,
            available_worker_capacity=2,
        ),
        provider_capacity_source=lambda: [
            _provider("provider-a", max_concurrency=1, active_requests=0)
        ],
        poll_interval=0,
    )

    first_manifest = scheduler.reconcile_once()
    assert [lane.task_ids for lane, _grant, _process in starts] == [["RES-1"]]
    assert first_manifest["resource_schedule"]["admitted_count"] == 1
    assert any(
        decision["bundle_key"] == "objective/resources/2"
        and decision["reason"] == "provider_concurrency"
        for decision in first_manifest["scheduler_decisions"]
    )
    first_lane, first_grant, first_process = starts[0]
    first_resource_lease = scheduler.resource_scheduler.active_leases[0]
    deadline = time.monotonic() + 5
    while (
        not child_pid_paths["RES-1"].exists()
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert child_pid_paths["RES-1"].exists()

    try:
        first_process.phase_state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = first_process.phase_state_path.with_name(
            f".{first_process.phase_state_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                    "active_task_id": "",
                    "implementation_in_progress": False,
                    "completed_task_ids": ["RES-1"],
                    "task_identities": {
                        "RES-1": {
                            "canonical_task_cid": (
                                first_process.lane.expected_task_cids_by_id[
                                    "RES-1"
                                ]
                            ),
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(first_process.phase_state_path)
        first_process.thread.join(10)
        assert not first_process.thread.is_alive()
        assert first_process.error is None
        assert first_process.result is not None
        assert first_process.result.successful

        second_manifest = scheduler.reconcile_once()
        assert [lane.task_ids for lane, _grant, _process in starts] == [
            ["RES-1"],
            ["RES-2"],
        ]
        second_lane, _second_grant, second_process = starts[1]
        assert second_lane.llm_provider == first_lane.llm_provider == "provider-a"
        assert all(
            lease.lease_id != first_resource_lease.lease_id
            for lease in scheduler.resource_scheduler.active_leases
        )
        assert len(scheduler.resource_scheduler.active_leases) == 1
        assert scheduler.resource_scheduler.active_leases[0].provider_id == (
            "provider-a"
        )
        assert any(
            decision["bundle_key"] == "objective/resources/2"
            and decision["decision"] == "launched"
            and decision["resource_admission"]["provider_id"] == "provider-a"
            for decision in second_manifest["scheduler_decisions"]
        )
        second_process.thread.join(5)
        assert not second_process.thread.is_alive()
        assert second_process.result is not None
        assert second_process.result.successful

        scheduler.reconcile_once()
        assert scheduler.resource_scheduler.active_leases == ()
        with LeaseCoordinator(repo / "coordination.sqlite3") as coordinator:
            first_heartbeat = coordinator.latest_heartbeat(first_grant.task_cid)
        assert first_heartbeat is not None
        assert first_heartbeat["capacity_millionths"] == 0
        assert first_heartbeat["provider_id"] == "provider-a"
    finally:
        for _lane, _grant, process in starts:
            if process.poll() is None:
                process.kill()
                process.thread.join(5)
        scheduler.stop(grace_seconds=0)


def test_leased_lane_rechecks_exact_completion_after_natural_nonzero_exit(
    tmp_path: Path,
) -> None:
    coordination_path = tmp_path / "coordination.sqlite3"
    phase_state_path = tmp_path / "phase-state.json"
    task_id = "RES-FINAL-EXIT"
    task_cid = profile_g_cid({"member": task_id})
    bundle = {
        "bundle_key": "objective/resources/final-exit",
        "tasks": [
            {
                "task_id": task_id,
                "canonical_task_cid": task_cid,
            }
        ],
        "execution_slice_task_ids": [task_id],
        "execution_slice_task_cids": [task_cid],
    }
    with LeaseCoordinator(coordination_path) as coordinator:
        task = coordinator.register_bundle(bundle)
        grant = coordinator.claim(
            task["task_cid"],
            "did:web:final-state-worker.example",
            requested_lease_ms=60_000,
        )

    child_script = """
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
task_id = sys.argv[2]
task_cid = sys.argv[3]
path.write_text(json.dumps({
    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    "active_task_id": "",
    "implementation_in_progress": False,
    "completed_task_ids": [task_id],
    "task_identities": {
        task_id: {"canonical_task_cid": task_cid},
    },
}), encoding="utf-8")
raise SystemExit(42)
"""
    result = run_leased_lane_result(
        coordination_path=coordination_path,
        grant=grant,
        command=(
            sys.executable,
            "-c",
            child_script,
            str(phase_state_path),
            task_id,
            task_cid,
        ),
        lease_ms=60_000,
        heartbeat_interval=0.2,
        phase_state_path=phase_state_path,
        expected_task_ids=(task_id,),
        expected_task_cids_by_id={task_id: task_cid},
    )

    assert result.successful
    assert result.disposition == "completed"
    assert result.exit_code == 0
    assert result.child_exit_code == 42


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").is_dir(),
    reason="late-fork process fencing regression requires Linux /proc",
)
def test_completion_fence_prevents_term_handler_respawn_before_capacity_release(
    tmp_path: Path,
) -> None:
    coordination_path = tmp_path / "coordination.sqlite3"
    phase_state_path = tmp_path / "phase-state.json"
    root_pid_path = tmp_path / "root.pid"
    worker_pid_path = tmp_path / "worker.pid"
    replacement_pid_path = tmp_path / "replacement.pid"
    task_id = "RES-RESPAWN"
    task_cid = profile_g_cid({"member": task_id})
    with LeaseCoordinator(coordination_path) as coordinator:
        task = coordinator.register_bundle(
            {
                "bundle_key": "objective/resources/respawn",
                "tasks": [
                    {
                        "task_id": task_id,
                        "canonical_task_cid": task_cid,
                    }
                ],
                "execution_slice_task_ids": [task_id],
                "execution_slice_task_cids": [task_cid],
            }
        )
        grant = coordinator.claim(
            task["task_cid"],
            "did:web:respawn-worker.example",
            requested_lease_ms=60_000,
        )

    worker_script = """
import os
import pathlib
import signal
import subprocess
import sys
import time

ready_path = pathlib.Path(sys.argv[1])
replacement_path = pathlib.Path(sys.argv[2])
def respawn(_signum, _frame):
    replacement = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    replacement_path.write_text(str(replacement.pid), encoding="utf-8")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, respawn)
ready_path.write_text(str(os.getpid()), encoding="utf-8")
time.sleep(60)
"""
    root_script = """
import os
import pathlib
import subprocess
import sys
import time

root_path = pathlib.Path(sys.argv[1])
root_path.write_text(str(os.getpid()), encoding="utf-8")
subprocess.Popen(
    [sys.executable, "-c", sys.argv[4], sys.argv[2], sys.argv[3]],
    start_new_session=True,
)
time.sleep(60)
"""
    outcomes: list[LeasedLaneResult] = []
    errors: list[BaseException] = []
    idle_liveness: list[tuple[bool, bool]] = []

    def sample_resources(**kwargs: object) -> dict[str, object]:
        if kwargs.get("active_workers") == 0:
            idle_liveness.append(
                (
                    pid_alive(int(root_pid_path.read_text(encoding="utf-8"))),
                    pid_alive(int(worker_pid_path.read_text(encoding="utf-8"))),
                )
            )
        return {}

    def execute() -> None:
        try:
            outcomes.append(
                run_leased_lane_result(
                    coordination_path=coordination_path,
                    grant=grant,
                    command=(
                        sys.executable,
                        "-c",
                        root_script,
                        str(root_pid_path),
                        str(worker_pid_path),
                        str(replacement_pid_path),
                        worker_script,
                    ),
                    lease_ms=60_000,
                    heartbeat_interval=0.02,
                    resource_sampler=sample_resources,
                    phase_state_path=phase_state_path,
                    expected_task_ids=(task_id,),
                    expected_task_cids_by_id={task_id: task_cid},
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    lane_thread = threading.Thread(target=execute, daemon=True)
    lane_thread.start()
    deadline = time.monotonic() + 5
    while (
        (not root_pid_path.exists() or not worker_pid_path.exists())
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert root_pid_path.exists()
    assert worker_pid_path.exists()

    try:
        phase_state_path.write_text(
            json.dumps(
                {
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                    "active_task_id": "",
                    "implementation_in_progress": False,
                    "completed_task_ids": [task_id],
                    "task_identities": {
                        task_id: {"canonical_task_cid": task_cid},
                    },
                }
            ),
            encoding="utf-8",
        )
        lane_thread.join(10)
    finally:
        for pid_path in (replacement_pid_path, worker_pid_path, root_pid_path):
            if not pid_path.exists():
                continue
            try:
                process_id = int(pid_path.read_text(encoding="utf-8"))
                if pid_alive(process_id):
                    os.kill(process_id, signal.SIGKILL)
            except (OSError, ValueError):
                pass
        lane_thread.join(5)

    assert errors == []
    assert len(outcomes) == 1 and outcomes[0].successful
    assert replacement_pid_path.exists() is False
    assert idle_liveness == [(False, False)]


def test_goal_runtime_work_kinds_have_four_distinct_resource_classes() -> None:
    classes = {
        kind: resource_class_for_work_kind(kind)
        for kind in ProofWorkKind
    }

    assert classes == {
        ProofWorkKind.MODEL_DRAFT: ProofResourceClass.MODEL_DRAFT.value,
        ProofWorkKind.TYPE_CHECK: ProofResourceClass.TYPE_CHECK.value,
        ProofWorkKind.SOLVER_PORTFOLIO: ProofResourceClass.SOLVER.value,
        ProofWorkKind.KERNEL_RECONSTRUCTION: ProofResourceClass.KERNEL.value,
    }
    assert len(set(classes.values())) == 4
    assert ProofResourceClass.TYPE_CHECK.value != ProofResourceClass.VALIDATION.value


def test_route_unavailability_returns_observable_deterministic_fallback() -> None:
    primary_called = False
    fallback_reasons: list[tuple[str, ...]] = []
    scheduler = GoalRuntimeResourceScheduler(
        policy=ResourcePolicy(max_lanes=1),
        host_resource_source=_host(
            worker_limit=1,
            available_worker_capacity=1,
        ),
        provider_capacity_source=[_provider(healthy=False)],
    )

    def primary(_context: object) -> None:
        nonlocal primary_called
        primary_called = True

    def fallback(context: object) -> dict[str, str]:
        fallback_reasons.append(context.reason_codes)
        return {"source": "deterministic", "reason": context.fallback_reason}

    result = scheduler.execute(
        ProofWorkRequest(
            work_id="draft:unavailable",
            work_kind=ProofWorkKind.MODEL_DRAFT,
            provider_id="provider-a",
            token_budget=100,
        ),
        primary,
        fallback=fallback,
    )

    assert not primary_called
    assert result.status is ProofWorkStatus.FALLBACK
    assert result.successful and result.used_fallback
    assert not result.primary_succeeded
    assert result.fallback_succeeded
    assert result.to_dict()["primary_succeeded"] is False
    assert result.fallback_reason == "provider_unhealthy"
    assert fallback_reasons == [("provider_unhealthy",)]
    assert result.admission is not None
    assert not result.admission.admitted
    assert result.value["source"] == "deterministic"
    assert scheduler.queued_count == scheduler.running_count == 0


def test_model_and_type_check_classes_can_execute_concurrently() -> None:
    entered: set[ProofWorkKind] = set()
    lock = threading.Lock()
    both_entered = threading.Event()
    release = threading.Event()
    results: dict[str, object] = {}
    scheduler = GoalRuntimeResourceScheduler(
        policy=ResourcePolicy(
            max_lanes=2,
            max_cpu_proof_concurrency=1,
            max_model_concurrency=1,
            resource_class_limits={
                ProofResourceClass.MODEL_DRAFT.value: 1,
                ProofResourceClass.TYPE_CHECK.value: 1,
            },
        ),
        max_queued_tasks=2,
        host_resource_source=_host(
            worker_limit=2,
            available_worker_capacity=2,
        ),
        provider_capacity_source=[_provider(max_concurrency=1)],
    )

    def execute(context: object) -> str:
        with lock:
            entered.add(context.request.work_kind)
            if len(entered) == 2:
                both_entered.set()
        assert release.wait(2)
        return context.request.work_kind.value

    requests = (
        ProofWorkRequest(
            work_id="draft:one",
            work_kind=ProofWorkKind.MODEL_DRAFT,
            provider_id="provider-a",
            token_budget=10,
            max_queue_wait_ms=1_000,
        ),
        ProofWorkRequest(
            work_id="type:one",
            work_kind=ProofWorkKind.TYPE_CHECK,
            max_queue_wait_ms=1_000,
        ),
    )
    threads = [
        threading.Thread(
            target=lambda item=item: results.setdefault(
                item.work_id, scheduler.execute(item, execute)
            )
        )
        for item in requests
    ]
    for thread in threads:
        thread.start()
    assert both_entered.wait(2)
    assert scheduler.running_count == 2
    release.set()
    for thread in threads:
        thread.join(2)

    assert set(entered) == {
        ProofWorkKind.MODEL_DRAFT,
        ProofWorkKind.TYPE_CHECK,
    }
    assert all(
        result.status is ProofWorkStatus.SUCCEEDED
        for result in results.values()
    )
    assert scheduler.resource_scheduler.active_leases == ()


def test_bounded_queue_backpressure_and_cancellation_are_observable() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    results: dict[str, object] = {}
    fallback_calls: list[str] = []
    scheduler = GoalRuntimeResourceScheduler(
        policy=ResourcePolicy(
            max_lanes=1,
            max_cpu_proof_concurrency=1,
            resource_class_limits={ProofResourceClass.SOLVER.value: 1},
        ),
        max_queued_tasks=1,
        queue_retry_ms=5,
        host_resource_source=_host(
            worker_limit=1,
            available_worker_capacity=1,
        ),
    )

    def first(_context: object) -> str:
        first_entered.set()
        assert release_first.wait(2)
        return "first"

    first_thread = threading.Thread(
        target=lambda: results.setdefault(
            "first",
            scheduler.execute(
                ProofWorkRequest(
                    work_id="solver:first",
                    work_kind=ProofWorkKind.SOLVER_PORTFOLIO,
                ),
                first,
            ),
        )
    )
    first_thread.start()
    assert first_entered.wait(2)

    second_thread = threading.Thread(
        target=lambda: results.setdefault(
            "second",
            scheduler.execute(
                ProofWorkRequest(
                    work_id="solver:second",
                    work_kind=ProofWorkKind.SOLVER_PORTFOLIO,
                    max_queue_wait_ms=1_000,
                ),
                lambda _context: "must-not-run",
                fallback=lambda _context: fallback_calls.append("second"),
            ),
        )
    )
    second_thread.start()
    deadline = time.monotonic() + 2
    while scheduler.queued_count != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert scheduler.queued_work_ids == ("solver:second",)

    third = scheduler.execute(
        ProofWorkRequest(
            work_id="solver:third",
            work_kind=ProofWorkKind.SOLVER_PORTFOLIO,
        ),
        lambda _context: "must-not-run",
        fallback=lambda context: fallback_calls.append(context.fallback_reason)
        or "safe",
    )
    assert third.status is ProofWorkStatus.FALLBACK
    assert third.fallback_reason == "queue_capacity"
    assert third.reason_codes == ("queue_capacity",)

    assert scheduler.cancel("solver:second", "goal_closed")
    second_thread.join(2)
    release_first.set()
    first_thread.join(2)

    assert results["second"].status is ProofWorkStatus.CANCELLED
    assert results["second"].fallback_reason == "goal_closed"
    assert fallback_calls == ["queue_capacity"]
    assert results["first"].status is ProofWorkStatus.SUCCEEDED
    assert scheduler.queued_count == scheduler.running_count == 0
    assert scheduler.resource_scheduler.active_leases == ()


def test_fallback_remains_registered_running_and_cooperatively_cancellable() -> None:
    fallback_entered = threading.Event()
    results: dict[str, object] = {}
    scheduler = GoalRuntimeResourceScheduler(
        policy=ResourcePolicy(max_lanes=1),
        max_fallback_concurrency=1,
        host_resource_source=_host(
            worker_limit=1,
            available_worker_capacity=1,
        ),
        provider_capacity_source=[_provider(healthy=False)],
    )

    def fallback(context: object) -> str:
        fallback_entered.set()
        assert context.cancellation_token.wait(2)
        return "discarded-after-cancellation"

    thread = threading.Thread(
        target=lambda: results.setdefault(
            "result",
            scheduler.execute(
                ProofWorkRequest(
                    work_id="draft:cancel-fallback",
                    work_kind=ProofWorkKind.MODEL_DRAFT,
                    provider_id="provider-a",
                ),
                lambda _context: "must-not-run",
                fallback=fallback,
            ),
        )
    )
    thread.start()
    assert fallback_entered.wait(2)

    assert scheduler.running_work_ids == ("draft:cancel-fallback",)
    assert scheduler.running_count == 1
    assert scheduler.fallback_running_count == 1
    assert scheduler.cancel("draft:cancel-fallback", "root_changed")
    thread.join(2)

    result = results["result"]
    assert result.status is ProofWorkStatus.CANCELLED
    assert not result.successful
    assert not result.primary_succeeded
    assert result.fallback_reason == "root_changed"
    assert scheduler.running_count == scheduler.fallback_running_count == 0
    assert not scheduler.cancel("draft:cancel-fallback")


def test_concurrent_fallbacks_obey_explicit_fallback_capacity() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    results: dict[str, object] = {}
    second_fallback_called = False
    scheduler = GoalRuntimeResourceScheduler(
        policy=ResourcePolicy(max_lanes=2),
        max_fallback_concurrency=1,
        host_resource_source=_host(
            worker_limit=2,
            available_worker_capacity=2,
        ),
        provider_capacity_source=[_provider(healthy=False)],
    )

    def first_fallback(_context: object) -> str:
        first_entered.set()
        assert release_first.wait(2)
        return "first-safe-result"

    first_thread = threading.Thread(
        target=lambda: results.setdefault(
            "first",
            scheduler.execute(
                ProofWorkRequest(
                    work_id="draft:fallback-one",
                    work_kind=ProofWorkKind.MODEL_DRAFT,
                    provider_id="provider-a",
                ),
                lambda _context: "must-not-run",
                fallback=first_fallback,
            ),
        )
    )
    first_thread.start()
    assert first_entered.wait(2)
    assert scheduler.fallback_running_count == 1

    def second_fallback(_context: object) -> str:
        nonlocal second_fallback_called
        second_fallback_called = True
        return "must-not-run"

    second = scheduler.execute(
        ProofWorkRequest(
            work_id="draft:fallback-two",
            work_kind=ProofWorkKind.MODEL_DRAFT,
            provider_id="provider-a",
        ),
        lambda _context: "must-not-run",
        fallback=second_fallback,
    )

    assert second.status is ProofWorkStatus.BACKPRESSURED
    assert not second.successful
    assert not second.used_fallback
    assert "fallback_capacity" in second.reason_codes
    assert not second_fallback_called
    assert scheduler.running_work_ids == ("draft:fallback-one",)

    release_first.set()
    first_thread.join(2)
    assert results["first"].status is ProofWorkStatus.FALLBACK
    assert scheduler.running_count == scheduler.fallback_running_count == 0
