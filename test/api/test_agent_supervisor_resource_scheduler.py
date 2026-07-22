from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import DynamicBundleScheduler
from ipfs_accelerate_py.agent_supervisor.lease_coordination import LeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.leased_lane import run_leased_lane_result
from ipfs_accelerate_py.agent_supervisor.resource_scheduler import (
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
    normalize_provider_capacity,
    sample_host_resources,
)


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
