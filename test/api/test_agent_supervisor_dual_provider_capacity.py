from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import (
    LeaseCoordinator,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    LEGACY_ADOPTION_BARRIER_REASON,
    DynamicBundleScheduler,
    build_arg_parser,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_capacity_snapshot import (
    DUAL_REVIEW_PROVIDER_ID,
    DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES,
    PROVIDER_CAPACITY_BUDGET_SEMANTICS,
    load_provider_capacity_snapshot,
    synthesize_dual_review_provider_capacity,
    write_provider_capacity_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ProviderCapacity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_review as legacy_review,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    PRODUCTION_CLI_POLICY_NAME,
)


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True


def _host(*, workers: int) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        observed_at_ms=int(time.time() * 1000),
        cpu_percent=10,
        memory_percent=10,
        disk_percent=10,
        memory_available_bytes=1_000_000_000,
        disk_available_bytes=1_000_000_000,
        active_workers=0,
        worker_limit=workers,
        available_worker_capacity=workers,
        capabilities=("cpu", "git"),
        resource_classes=("cpu-small", "cpu-medium"),
    )


def _capacity(
    provider_id: str,
    *,
    observed_at_ms: int,
    max_concurrency: int = 4,
    active_requests: int = 0,
    healthy: bool = True,
    capabilities: tuple[str, ...] | None = None,
) -> ProviderCapacity:
    return ProviderCapacity(
        provider_id=provider_id,
        healthy=healthy,
        quota_remaining=100,
        latency_ms=25,
        context_window_tokens=100_000,
        token_budget_remaining=100_000,
        max_concurrency=max_concurrency,
        active_requests=active_requests,
        capabilities=(
            tuple(DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES[provider_id])
            if capabilities is None
            else capabilities
        ),
        observed_at_ms=observed_at_ms,
    )


def _write_bundle_index(
    path: Path,
    roles: list[str],
) -> None:
    bundles = {
        f"objective/capacity/{index}": {
            "shard_path": f"capacity-{index}.todo.md",
            "parallel_lane": f"capacity-{index}",
            "tasks": [
                {
                    "task_id": f"CAP-{index}",
                    "metadata": {"Provider role": role},
                }
            ],
        }
        for index, role in enumerate(roles, start=1)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"source_todo": "tasks.todo.md", "bundles": bundles}),
        encoding="utf-8",
    )


def _write_legacy_policy(path: Path, *, enabled: bool) -> None:
    template = copy.deepcopy(
        legacy_review.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE
    )
    body = {
        "schema": legacy_review.LEGACY_LANDED_REVIEW_POLICY_SCHEMA,
        "interface": legacy_review.LEGACY_LANDED_REVIEW_POLICY_INTERFACE,
        "enabled": enabled,
        "issuer_key_id": "ed25519:sha256:" + "c" * 64,
        "current_head": "a" * 40,
        "current_tree_id": "b" * 40,
        "max_leaf_tokens": template["max_leaf_tokens"],
        "providers": template["providers"],
        "tasks": template["tasks"],
        "historical_provider": "unverified",
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    payload = {**body, "policy_id": content_identity(body)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def _scheduler(
    repo: Path,
    *,
    provider_capacity_source: object,
    launcher: object,
    max_lanes: int,
    max_age_ms: int = 1_000,
) -> DynamicBundleScheduler:
    return DynamicBundleScheduler(
        bundle_index_path=repo / "index.json",
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        coordination_path=repo / "coordination.duckdb",
        max_lanes=max_lanes,
        launcher=launcher,  # type: ignore[arg-type]
        process_alive=lambda process: process.alive,
        host_resource_source=lambda *_args, **_kwargs: _host(
            workers=max_lanes
        ),
        provider_capacity_source=provider_capacity_source,  # type: ignore[arg-type]
        provider_capacity_max_age_ms=max_age_ms,
        resource_policy={"require_provider_telemetry": False},
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_context_budget_tokens=32_768,
        poll_interval=0,
    )


def test_private_snapshot_round_trip_is_monotonic_and_owner_only(
    tmp_path: Path,
) -> None:
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "capacity.json"
    observed = 100_000
    providers = (
        _capacity("grok_cli", observed_at_ms=observed),
        _capacity("codex_cli", observed_at_ms=observed),
    )

    payload = write_provider_capacity_snapshot(
        path,
        providers,
        max_age_ms=1_000,
        now_ms=observed,
    )

    assert payload["budget_semantics"] == PROVIDER_CAPACITY_BUDGET_SEMANTICS
    assert path.stat().st_mode & 0o777 == 0o600
    assert [item.provider_id for item in load_provider_capacity_snapshot(
        path, max_age_ms=1_000, now_ms=observed
    )] == ["codex_cli", "grok_cli"]

    older = tuple(
        _capacity(item.provider_id, observed_at_ms=observed - 1)
        for item in providers
    )
    with pytest.raises(ValueError, match="move observation time backward"):
        write_provider_capacity_snapshot(
            path,
            older,
            max_age_ms=1_000,
            now_ms=observed + 1,
        )
    conflicting = (
        _capacity(
            "grok_cli",
            observed_at_ms=observed,
            active_requests=1,
        ),
        providers[1],
    )
    with pytest.raises(ValueError, match="conflicts at the same observation"):
        write_provider_capacity_snapshot(
            path,
            conflicting,
            max_age_ms=1_000,
            now_ms=observed + 1,
        )
    with pytest.raises(ValueError, match="stale"):
        load_provider_capacity_snapshot(
            path,
            max_age_ms=1_000,
            now_ms=observed + 1_001,
        )
    os.chmod(path, 0o640)
    with pytest.raises(ValueError, match="permissions"):
        load_provider_capacity_snapshot(
            path,
            max_age_ms=1_000,
            now_ms=observed,
        )


def test_snapshot_cas_is_monotonic_for_each_provider_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capacity.json"
    os.chmod(tmp_path, 0o700)
    initial = (
        _capacity("grok_cli", observed_at_ms=100),
        _capacity("codex_cli", observed_at_ms=1_000),
    )
    write_provider_capacity_snapshot(
        path,
        initial,
        max_age_ms=2_000,
        now_ms=1_000,
    )

    # The envelope minimum advances from 100 to 500, but Codex rolls back.
    with pytest.raises(ValueError, match="time backward for codex_cli"):
        write_provider_capacity_snapshot(
            path,
            (
                _capacity("grok_cli", observed_at_ms=900),
                _capacity("codex_cli", observed_at_ms=500),
            ),
            max_age_ms=2_000,
            now_ms=1_001,
        )

    # Advancing only Codex is valid even though the envelope minimum is equal.
    advanced = (
        initial[0],
        _capacity("codex_cli", observed_at_ms=1_100),
    )
    write_provider_capacity_snapshot(
        path,
        advanced,
        max_age_ms=2_000,
        now_ms=1_100,
    )
    loaded = {
        item.provider_id: item
        for item in load_provider_capacity_snapshot(
            path,
            max_age_ms=2_000,
            now_ms=1_100,
        )
    }
    assert loaded["grok_cli"].observed_at_ms == 100
    assert loaded["codex_cli"].observed_at_ms == 1_100

    with pytest.raises(ValueError, match="observation time for grok_cli"):
        write_provider_capacity_snapshot(
            path,
            (
                _capacity(
                    "grok_cli",
                    observed_at_ms=100,
                    active_requests=1,
                ),
                _capacity("codex_cli", observed_at_ms=1_200),
            ),
            max_age_ms=2_000,
            now_ms=1_200,
        )


def test_private_snapshot_rejects_unreviewed_provider_inventory(
    tmp_path: Path,
) -> None:
    os.chmod(tmp_path, 0o700)
    grok = _capacity("grok_cli", observed_at_ms=100)
    extra = ProviderCapacity.from_mapping(
        {**grok.to_dict(), "provider_id": "other_provider"}
    )

    with pytest.raises(ValueError, match="inventory must be exactly"):
        write_provider_capacity_snapshot(
            tmp_path / "capacity.json",
            (
                grok,
                _capacity("codex_cli", observed_at_ms=100),
                extra,
            ),
            max_age_ms=1_000,
            now_ms=100,
        )


def test_pair_free_concurrency_is_minimum_and_requires_role_capabilities() -> None:
    now = 200_000
    capacities = (
        _capacity(
            "grok_cli",
            observed_at_ms=now,
            max_concurrency=4,
            active_requests=1,
        ),
        _capacity(
            "codex_cli",
            observed_at_ms=now,
            max_concurrency=3,
            active_requests=1,
        ),
    )

    pair = {
        item.provider_id: item
        for item in synthesize_dual_review_provider_capacity(
            capacities,
            max_age_ms=1_000,
            now_ms=now,
        )
    }[DUAL_REVIEW_PROVIDER_ID]
    assert pair.healthy is True
    assert pair.max_concurrency == 3
    assert pair.available_concurrency == 2

    missing_role = (
        capacities[0],
        _capacity(
            "codex_cli",
            observed_at_ms=now,
            capabilities=("codex-cli",),
        ),
    )
    closed = {
        item.provider_id: item
        for item in synthesize_dual_review_provider_capacity(
            missing_role,
            max_age_ms=1_000,
            now_ms=now,
        )
    }[DUAL_REVIEW_PROVIDER_ID]
    assert closed.healthy is False
    assert closed.available_concurrency == 0


def test_production_planning_keeps_typed_local_lane_provider_free(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index = repo / "index.json"
    _write_bundle_index(
        index,
        ["grok-implement, codex-review", "deterministic-only"],
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_context_budget_tokens=32_768,
        optimize_bundles=False,
    )
    by_task = {lane.task_ids[0]: lane for lane in lanes}

    assert by_task["CAP-1"].llm_provider == DUAL_REVIEW_PROVIDER_ID
    assert by_task["CAP-1"].required_context_tokens == 32_768
    assert by_task["CAP-2"].llm_provider == ""
    assert by_task["CAP-2"].required_context_tokens == 0
    assert not any(
        item.startswith("llm:")
        for item in by_task["CAP-2"].required_capabilities
    )


@pytest.mark.parametrize(
    "metadata_case",
    [
        "deterministic-alias",
        "deterministic-space-alias",
        "implementation-provider-fallback",
        "execution-mode-fallback",
        "top-level-provider-role",
        "combined-local-roles",
    ],
)
def test_production_planner_does_not_broaden_child_typed_local_contract(
    tmp_path: Path,
    metadata_case: str,
) -> None:
    repo = tmp_path / metadata_case
    index_path = repo / "index.json"
    _write_bundle_index(index_path, ["deterministic-only"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    task = index["bundles"]["objective/capacity/1"]["tasks"][0]
    metadata = task["metadata"]
    if metadata_case == "deterministic-alias":
        metadata["Provider role"] = "deterministic"
    elif metadata_case == "deterministic-space-alias":
        metadata["Provider role"] = "deterministic only"
    elif metadata_case == "implementation-provider-fallback":
        metadata.clear()
        metadata["Implementation provider"] = "deterministic-only"
    elif metadata_case == "execution-mode-fallback":
        metadata.clear()
        metadata["Execution mode"] = "deterministic-only"
    elif metadata_case == "top-level-provider-role":
        metadata.clear()
        task["Provider role"] = "deterministic-only"
    else:
        metadata["Provider role"] = "deterministic-only, operator-only"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    lane = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_context_budget_tokens=32_768,
        optimize_bundles=False,
    )[0]

    assert lane.llm_provider == DUAL_REVIEW_PROVIDER_ID
    assert lane.required_context_tokens == 32_768


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_capacity_overlay_is_exact_and_default_off(
    tmp_path: Path,
    enabled: bool,
) -> None:
    repo = tmp_path / ("enabled" if enabled else "disabled")
    index_path = repo / "index.json"
    _write_bundle_index(index_path, ["deterministic-only"] * 9)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tasks = [
        bundle["tasks"][0]
        for bundle in index["bundles"].values()
    ]
    template_tasks = copy.deepcopy(
        legacy_review.EXACT_EIGHT_LEGACY_LANDED_POLICY_TEMPLATE["tasks"]
    )
    for task, template_task in zip(tasks[:8], template_tasks, strict=True):
        task["task_id"] = template_task["task_id"]
        task["canonical_task_key"] = template_task["canonical_task_key"]
        task["canonical_task_cid"] = template_task["canonical_task_cid"]
    tasks[8]["task_id"] = "OTHER"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    policy_path = repo / "legacy-policy.json"
    key_path = repo / "legacy-policy.key"
    _write_legacy_policy(policy_path, enabled=enabled)
    key_path.write_bytes(b"development-test-key")
    key_path.chmod(0o600)

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        implement=True,
        implementation_timeout=900.0,
        legacy_landed_review_policy_path=policy_path,
        legacy_landed_review_key_path=key_path,
        optimize_bundles=False,
    )
    by_task = {
        str(lane.queue_payload["tasks"][0]["task_id"]): lane
        for lane in lanes
    }

    for template_task in template_tasks:
        assert by_task[template_task["task_id"]].llm_provider == (
            DUAL_REVIEW_PROVIDER_ID if enabled else ""
        )
    assert by_task["OTHER"].llm_provider == ""
    assert by_task["OTHER"].task_ids == ([] if enabled else ["OTHER"])
    assert by_task["OTHER"].claimable is (not enabled)
    if enabled:
        assert by_task["OTHER"].queue_payload["blocked_reason"] == (
            LEGACY_ADOPTION_BARRIER_REASON
        )
    for lane in lanes:
        stall_index = lane.command.index("--implementation-log-stall-seconds")
        assert float(lane.command[stall_index + 1]) == 900.0
        timeout_index = lane.command.index("--implementation-timeout")
        assert float(lane.command[timeout_index + 1]) == 900.0


def test_four_model_lanes_are_capped_by_two_free_pair_slots(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_bundle_index(
        repo / "index.json",
        ["grok-implement, codex-review"] * 4,
    )
    now = int(time.time() * 1000)
    providers = (
        _capacity(
            "grok_cli",
            observed_at_ms=now,
            max_concurrency=4,
            active_requests=1,
        ),
        _capacity(
            "codex_cli",
            observed_at_ms=now,
            max_concurrency=3,
            active_requests=1,
        ),
    )
    starts: list[object] = []

    def launch(lane: object, _grant: object) -> _Process:
        starts.append(lane)
        return _Process(20_000 + len(starts))

    scheduler = _scheduler(
        repo,
        provider_capacity_source=lambda: providers,
        launcher=launch,
        max_lanes=4,
        max_age_ms=30_000,
    )
    manifest = scheduler.reconcile_once()

    assert len(starts) == 2
    assert all(
        lane.llm_provider == DUAL_REVIEW_PROVIDER_ID  # type: ignore[attr-defined]
        for lane in starts
    )
    assert manifest["resource_schedule"]["admitted_count"] == 2
    deferred = [
        item
        for item in manifest["scheduler_decisions"]
        if item["decision"] == "deferred"
    ]
    assert len(deferred) == 2
    assert all(item["reason"] == "provider_concurrency" for item in deferred)
    with LeaseCoordinator(repo / "coordination.duckdb") as coordinator:
        assert len(
            [
                item
                for item in coordinator.list_tasks()
                if item["state"] == "accepted"
            ]
        ) == 2


@pytest.mark.parametrize(
    "telemetry_kind",
    ["missing", "one-provider", "stale", "unhealthy", "missing-role"],
)
def test_bad_pair_telemetry_blocks_before_coordination_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry_kind: str,
) -> None:
    repo = tmp_path / telemetry_kind
    _write_bundle_index(
        repo / "index.json",
        ["grok-implement, codex-review"],
    )
    now = int(time.time() * 1000)
    grok = _capacity("grok_cli", observed_at_ms=now)
    telemetry: tuple[ProviderCapacity, ...]
    if telemetry_kind == "missing":
        telemetry = ()
    elif telemetry_kind == "one-provider":
        telemetry = (grok,)
    elif telemetry_kind == "stale":
        telemetry = (
            _capacity("grok_cli", observed_at_ms=now - 1_001),
            _capacity("codex_cli", observed_at_ms=now - 1_001),
        )
    elif telemetry_kind == "unhealthy":
        telemetry = (
            grok,
            _capacity("codex_cli", observed_at_ms=now, healthy=False),
        )
    else:
        telemetry = (
            grok,
            _capacity(
                "codex_cli",
                observed_at_ms=now,
                capabilities=("codex-cli",),
            ),
        )
    starts: list[object] = []
    claims: list[str] = []
    original_claim_ready = LeaseCoordinator.claim_ready

    def claim_ready(self: LeaseCoordinator, *args: object, **kwargs: object):
        claims.append("called")
        return original_claim_ready(self, *args, **kwargs)

    monkeypatch.setattr(LeaseCoordinator, "claim_ready", claim_ready)
    scheduler = _scheduler(
        repo,
        provider_capacity_source=lambda: telemetry,
        launcher=lambda lane, _grant: starts.append(lane),
        max_lanes=1,
    )
    manifest = scheduler.reconcile_once()

    assert starts == []
    assert claims == []
    deferred = [
        item
        for item in manifest["scheduler_decisions"]
        if item["decision"] == "deferred"
    ]
    assert len(deferred) == 1
    assert deferred[0]["reason"] == "provider_unhealthy"
    with LeaseCoordinator(repo / "coordination.duckdb") as coordinator:
        assert all(
            item["state"] != "accepted" for item in coordinator.list_tasks()
        )


def test_production_capacity_cannot_be_rescued_by_ambient_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _write_bundle_index(
        repo / "index.json",
        ["grok-implement, codex-review"],
    )
    now = int(time.time() * 1000)
    ambient = {
        item.provider_id: item.to_dict()
        for item in (
            _capacity("grok_cli", observed_at_ms=now),
            _capacity("codex_cli", observed_at_ms=now),
        )
    }
    monkeypatch.setenv(
        "IPFS_ACCELERATE_LLM_ROUTER_CAPACITY_JSON",
        json.dumps({"providers": ambient}),
    )
    claims: list[str] = []
    original_claim_ready = LeaseCoordinator.claim_ready

    def claim_ready(self: LeaseCoordinator, *args: object, **kwargs: object):
        claims.append("called")
        return original_claim_ready(self, *args, **kwargs)

    monkeypatch.setattr(LeaseCoordinator, "claim_ready", claim_ready)
    scheduler = DynamicBundleScheduler(
        bundle_index_path=repo / "index.json",
        repo_root=repo,
        state_root=repo / "state",
        coordination_path=repo / "coordination.duckdb",
        max_lanes=1,
        launcher=lambda _lane, _grant: pytest.fail("ambient JSON launched work"),
        host_resource_source=lambda *_args, **_kwargs: _host(workers=1),
        resource_policy={"require_provider_telemetry": False},
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
    )

    manifest = scheduler.reconcile_once()
    assert claims == []
    assert manifest["running_count"] == 0


def test_provider_capacity_max_age_cli_is_explicit() -> None:
    args = build_arg_parser().parse_args(
        [
            "--bundle-index-path",
            "index.json",
            "--provider-capacity-max-age-ms",
            "12345",
        ]
    )
    assert args.provider_capacity_max_age_ms == 12_345


def test_generic_scheduler_retains_ordinary_provider_capacity_json(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index_path = repo / "index.json"
    _write_bundle_index(index_path, [""])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    task = index["bundles"]["objective/capacity/1"]["tasks"][0]
    task.update(
        {
            "required_context_tokens": 1_000,
            "token_budget": 100,
        }
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    capacity_path = repo / "ordinary-provider-capacity.json"
    capacity_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider_id": "provider-fast",
                        "healthy": True,
                        "quota_remaining": 10,
                        "context_window_tokens": 8_000,
                        "token_budget_remaining": 8_000,
                        "max_concurrency": 1,
                        "active_requests": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    starts: list[object] = []

    scheduler = DynamicBundleScheduler(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        coordination_path=repo / "coordination.duckdb",
        provider_capacity_path=capacity_path,
        max_lanes=1,
        launcher=lambda lane, _grant: (
            starts.append(lane) or _Process(30_000)
        ),
        process_alive=lambda process: process.alive,
        host_resource_source=lambda *_args, **_kwargs: _host(workers=1),
    )

    manifest = scheduler.reconcile_once()

    assert len(starts) == 1
    assert starts[0].llm_provider == "provider-fast"  # type: ignore[attr-defined]
    assert manifest["running_count"] == 1
