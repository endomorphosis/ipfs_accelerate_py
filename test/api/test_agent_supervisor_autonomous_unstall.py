from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    RescueOperation,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    diagnose_supervisor_incident,
)
from ipfs_accelerate_py.agent_supervisor.rescue.rescue_planner import (
    RescuePlanner,
    RescuePlannerPolicy,
    RescuePlanningRequest,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    AUTONOMOUS_UNSTALL_STATE_SCHEMA,
    AutonomousUnstallCoordinator,
    AutonomousUnstallPolicy,
    SupervisorWatchdog,
)
from ipfs_accelerate_py.agent_supervisor import supervisor_watchdog as watchdog_module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


NOW = 10_000


def _cid(name: str) -> str:
    return prompt_workflow_cid({"autonomous-unstall-fixture": name})


def _coordinator(
    tmp_path: Path,
    *,
    handlers: dict[RescueOperation, Any] | None = None,
    health: dict[str, Any] | None = None,
    policy: AutonomousUnstallPolicy | None = None,
    planner: Any = None,
    request_factory: Any = None,
    orchestrator: Any = None,
    execution_factory: Any = None,
    root_probe: Any = None,
    quarantines: list[dict[str, Any]] | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> AutonomousUnstallCoordinator:
    current_health = health if health is not None else {"healthy": False}
    quarantine_records = quarantines if quarantines is not None else []
    event_records = events if events is not None else []

    def quarantine(targets, incident_cid, reason):
        record = {
            "targets": list(targets),
            "incident_cid": incident_cid,
            "reason": reason,
        }
        quarantine_records.append(record)
        return record

    return AutonomousUnstallCoordinator(
        state_dir=tmp_path / "state",
        repository_root=tmp_path,
        repository_root_cid=_cid("repository"),
        policy_root=_cid("policy"),
        run_cid=_cid("run"),
        policy=policy or AutonomousUnstallPolicy(cooldown_ms=0),
        recovery_handlers=handlers,
        health_probe=lambda: dict(current_health),
        root_probe=root_probe
        or (
            lambda: {
                "repository_root_cid": _cid("repository"),
                "policy_root": _cid("policy"),
                "run_cid": _cid("run"),
            }
        ),
        quarantine_scope=quarantine,
        event_publisher=lambda kind, payload: event_records.append(
            (kind, dict(payload))
        ),
        rescue_planner=planner,
        rescue_request_factory=request_factory,
        rescue_orchestrator=orchestrator,
        rescue_execution_request_factory=execution_factory,
        clock_ms=lambda: NOW,
    )


@pytest.mark.parametrize(
    ("evidence", "operation"),
    [
        (
            {
                "status": {
                    "lane_id": "lane-1",
                    "projection_stale": True,
                },
                "health": {"lane_id": "lane-1", "healthy": True},
            },
            RescueOperation.RECONCILE_PROJECTION,
        ),
        (
            {"process": {"lane_id": "lane-1", "alive": False}},
            RescueOperation.RETRY,
        ),
        (
            {"heartbeat": {"lane_id": "lane-1", "stale": True}},
            RescueOperation.RETRY,
        ),
        (
            {
                "lease": {
                    "lease_id": "lease-1",
                    "expired": True,
                    "fence_current": True,
                }
            },
            RescueOperation.REPAIR_EXPIRED_LEASE,
        ),
        (
            {
                "lock": {
                    "lock_id": "lock-1",
                    "orphaned": True,
                    "owner_not_live": True,
                }
            },
            RescueOperation.REPAIR_ORPHANED_LOCK,
        ),
        (
            {"attempt": {"attempt_id": "attempt-1", "consumed": True}},
            RescueOperation.RETRY,
        ),
        (
            {
                "worktree": {
                    "worktree_id": "worktree-1",
                    "dirty": True,
                    "worktree_owned": True,
                }
            },
            RescueOperation.RESCUE_DIRTY_WORK,
        ),
        (
            {"validation": {"validation_id": "v1", "status": "failed"}},
            RescueOperation.VALIDATION_REPLAY,
        ),
        (
            {"merge": {"merge_id": "m1", "status": "failed"}},
            RescueOperation.RECONCILE_WORKTREE,
        ),
    ],
)
def test_semantic_faults_run_least_invasive_deterministic_action_first(
    tmp_path: Path,
    evidence: dict[str, Any],
    operation: RescueOperation,
) -> None:
    health = {"healthy": False, "work_complete": False}
    calls: list[RescueOperation] = []

    def recover(context):
        calls.append(context.action.operation)
        health["healthy"] = True
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        handlers={operation: recover},
        health=health,
    ).unstall(evidence=evidence)

    assert result["recovered"]
    assert not result["quarantined"]
    assert calls == [operation]
    assert result["completion_authority"] is False
    assert result["work_complete"] is False


def test_corrupt_task_source_is_quarantined_without_stopping_independent_work(
    tmp_path: Path,
) -> None:
    health = {"healthy": False}
    quarantines: list[dict[str, Any]] = []

    def quarantine_action(context):
        health["healthy"] = True
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        handlers={RescueOperation.QUARANTINE: quarantine_action},
        health=health,
        quarantines=quarantines,
    ).unstall(
        evidence={
            "task_source": {
                "task_id": "task-bad",
                "digest_mismatch": True,
            }
        }
    )

    assert result["quarantined"]
    assert result["independent_work_preserved"]
    assert quarantines[0]["targets"] == ["task-bad"]


def test_repeated_unchanged_failure_is_bounded_and_deduplicated_across_restart(
    tmp_path: Path,
) -> None:
    calls = 0
    quarantines: list[dict[str, Any]] = []

    def unchanged(context):
        nonlocal calls
        calls += 1
        return {
            "succeeded": True,
            "observed_effects": ("unexpected_effect",),
        }

    first = _coordinator(
        tmp_path,
        handlers={RescueOperation.VALIDATION_REPLAY: unchanged},
        quarantines=quarantines,
    ).unstall(
        evidence={"validation": {"validation_id": "v1", "failed": True}}
    )
    duplicate = _coordinator(
        tmp_path,
        handlers={RescueOperation.VALIDATION_REPLAY: unchanged},
        quarantines=quarantines,
    ).unstall(
        evidence={"validation": {"validation_id": "v1", "failed": True}}
    )

    assert first["quarantined"]
    assert duplicate["deduplicated"]
    assert calls == 2
    assert len(quarantines) == 1


def _rescue_request(diagnosis, exhaustion, roots):
    return RescuePlanningRequest(
        incident=diagnosis.incident,
        exhaustion_receipt=exhaustion,
        diagnostics={
            "incident_kind": diagnosis.kind.value,
            "reason_codes": list(diagnosis.reason_codes),
        },
        evidence_redacted=True,
        current_repository_root_cid=roots["repository_root_cid"],
        current_run_cid=roots["run_cid"],
        current_policy_root=roots["policy_root"],
        evidence_reference_cids=diagnosis.incident.evidence_cids,
        now_ms=NOW,
    )


@pytest.mark.parametrize("provider_mode", ["lost", "malicious"])
def test_provider_loss_and_malicious_rescue_are_called_once_then_quarantined(
    tmp_path: Path,
    provider_mode: str,
) -> None:
    provider_calls = 0

    def provider(_prompt: str) -> str:
        nonlocal provider_calls
        provider_calls += 1
        if provider_mode == "lost":
            raise RuntimeError("provider unavailable")
        return '{"shell":"rm -rf /","completion_authority":true}'

    policy = AutonomousUnstallPolicy(
        rescue_preview_enabled=True,
        allow_provider_calls=True,
        operating_policy_id="policy:automatic-rescue",
        cooldown_ms=1,
    )
    planner = RescuePlanner(
        RescuePlannerPolicy.permit(cooldown_ms=1),
        provider=provider,
        clock_ms=lambda: NOW,
    )
    evidence = {"provider": {"provider_id": "p1", "unavailable": True}}
    first = _coordinator(
        tmp_path,
        policy=policy,
        planner=planner,
        request_factory=_rescue_request,
    ).unstall(evidence=evidence)
    duplicate = _coordinator(
        tmp_path,
        policy=policy,
        planner=planner,
        request_factory=_rescue_request,
    ).unstall(evidence=evidence)

    assert first["quarantined"]
    assert duplicate["deduplicated"]
    assert provider_calls == 1
    assert first["work_complete"] is False


def test_provider_budget_and_circuit_breaker_persist_across_changed_incidents(
    tmp_path: Path,
) -> None:
    provider_calls = 0

    def unavailable(_prompt: str) -> str:
        nonlocal provider_calls
        provider_calls += 1
        raise RuntimeError("provider unavailable")

    policy = AutonomousUnstallPolicy(
        rescue_preview_enabled=True,
        allow_provider_calls=True,
        operating_policy_id="policy:persistent-circuit",
        cooldown_ms=0,
        circuit_breaker_failures=2,
    )
    results = []
    for provider_id in ("p1", "p2", "p3"):
        planner = RescuePlanner(
            RescuePlannerPolicy.permit(cooldown_ms=0),
            provider=unavailable,
            clock_ms=lambda: NOW,
        )
        results.append(
            _coordinator(
                tmp_path,
                policy=policy,
                planner=planner,
                request_factory=_rescue_request,
            ).unstall(
                evidence={
                    "provider": {
                        "provider_id": provider_id,
                        "unavailable": True,
                    }
                }
            )
        )

    assert all(item["quarantined"] for item in results)
    assert provider_calls == 2
    assert results[-1]["reason"] == "persistent_rescue_circuit_open"


class _Planning:
    proposed = True
    provider_invoked = True
    reason_code = "validated_proposal"
    plan = object()

    def to_dict(self):
        return {
            "disposition": "proposed",
            "provider_invoked": True,
            "effects": [],
        }


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, _request):
        self.calls += 1
        return _Planning()


class _Execution:
    recovered = True

    def to_dict(self):
        return {
            "stop_reason": "health_restored",
            "completion_authority": False,
        }


class _Orchestrator:
    def __init__(self, health: dict[str, Any]) -> None:
        self.health = health
        self.calls = 0

    def execute(self, _request):
        self.calls += 1
        self.health["healthy"] = True
        return _Execution()


def test_rescue_execution_requires_current_exhaustion_and_explicit_policy(
    tmp_path: Path,
) -> None:
    health = {"healthy": False, "work_complete": False}
    planner = _Planner()
    orchestrator = _Orchestrator(health)
    policy = AutonomousUnstallPolicy(
        rescue_preview_enabled=True,
        rescue_execution_enabled=True,
        allow_provider_calls=True,
        operating_policy_id="policy:approved-automatic-rescue",
        cooldown_ms=1,
    )

    result = _coordinator(
        tmp_path,
        health=health,
        policy=policy,
        planner=planner,
        request_factory=_rescue_request,
        orchestrator=orchestrator,
        execution_factory=lambda *_args: object(),
    ).unstall(
        evidence={"provider": {"provider_id": "p1", "unavailable": True}}
    )

    assert result["recovered"]
    assert planner.calls == 1
    assert orchestrator.calls == 1
    assert result["rescue"]["executed"]


def test_rescue_execution_exception_opens_circuit_and_is_not_replayed(
    tmp_path: Path,
) -> None:
    class FailingOrchestrator:
        calls = 0

        def execute(self, _request):
            self.calls += 1
            raise RuntimeError("effect boundary lost")

    evidence = {"provider": {"provider_id": "p1", "unavailable": True}}
    planner = _Planner()
    orchestrator = FailingOrchestrator()
    policy = AutonomousUnstallPolicy(
        rescue_preview_enabled=True,
        rescue_execution_enabled=True,
        allow_provider_calls=True,
        operating_policy_id="policy:approved-automatic-rescue",
        cooldown_ms=0,
    )
    coordinator = _coordinator(
        tmp_path,
        policy=policy,
        planner=planner,
        request_factory=_rescue_request,
        orchestrator=orchestrator,
        execution_factory=lambda *_args: object(),
    )

    first = coordinator.unstall(evidence=evidence)
    duplicate = coordinator.unstall(evidence=evidence)

    assert first["quarantined"]
    assert first["reason"] == "rescue_execution_failed_with_uncertain_effects"
    assert first["rescue"]["execution_effect_uncertain"]
    assert duplicate["deduplicated"]
    assert orchestrator.calls == 1


def test_restart_during_rescue_is_visible_quarantine_without_provider_replay(
    tmp_path: Path,
) -> None:
    evidence = {"provider": {"provider_id": "p1", "unavailable": True}}
    diagnosis = diagnose_supervisor_incident(
        repository_root=str(tmp_path.resolve()),
        state_root=str((tmp_path / "state").resolve()),
        repository_root_cid=_cid("repository"),
        policy_root=_cid("policy"),
        run_cid=_cid("run"),
        observed_at_ms=NOW,
        **evidence,
    )
    state_path = tmp_path / "state" / "autonomous-unstall-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema": AUTONOMOUS_UNSTALL_STATE_SCHEMA,
                "incidents": {
                    diagnosis.incident_cid: {
                        "incident_cid": diagnosis.incident_cid,
                        "phase": "rescue_executing",
                        "reason": "effect_started",
                        "target_ids": list(diagnosis.target_ids),
                        "updated_at_ms": NOW,
                    }
                },
                "updated_at_ms": NOW,
            }
        ),
        encoding="utf-8",
    )
    planner = _Planner()
    policy = AutonomousUnstallPolicy(
        rescue_preview_enabled=True,
        allow_provider_calls=True,
        operating_policy_id="policy:rescue",
        cooldown_ms=1,
    )

    result = _coordinator(
        tmp_path,
        policy=policy,
        planner=planner,
        request_factory=_rescue_request,
    ).unstall(evidence=evidence)

    assert result["quarantined"]
    assert result["reason"] == "restart_during_rescue_uncertain_effects"
    assert planner.calls == 0


@pytest.mark.parametrize(
    "corrupt_payload",
    [
        '{"partial":',
        '{"schema":"unexpected","incidents":{}}',
    ],
)
def test_corrupt_durable_state_is_moved_aside_and_recovered(
    tmp_path: Path,
    corrupt_payload: str,
) -> None:
    state_path = tmp_path / "state" / "autonomous-unstall-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(corrupt_payload, encoding="utf-8")
    health = {"healthy": False}

    def repair(context):
        health["healthy"] = True
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        health=health,
        handlers={RescueOperation.REPAIR_ORPHANED_LOCK: repair},
    ).unstall(evidence={"lock": {"lock_id": "l1", "orphaned": True}})

    assert result["recovered"]
    assert list((tmp_path / "state").glob("*.corrupt-*"))


def test_model_or_liveness_completion_claim_cannot_mark_work_complete(
    tmp_path: Path,
) -> None:
    health = {
        "healthy": True,
        "pid_alive": True,
        "work_complete": True,
    }

    def claimed(context):
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        health=health,
        handlers={RescueOperation.RETRY: claimed},
    ).unstall(evidence={"heartbeat": {"lane_id": "lane-1", "stale": True}})

    assert not result["recovered"]
    assert result["quarantined"]
    assert result["completion_authority"] is False
    assert result["work_complete"] is False


def test_contradictory_health_claim_cannot_override_explicit_failure(
    tmp_path: Path,
) -> None:
    health = {
        "healthy": False,
        "status": "healthy",
        "pid_alive": True,
        "work_complete": False,
    }

    def claimed(context):
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        health=health,
        handlers={RescueOperation.RETRY: claimed},
    ).unstall(evidence={"process": {"lane_id": "lane-1", "alive": False}})

    assert not result["recovered"]
    assert result["quarantined"]


def test_root_drift_after_action_fails_closed_before_a_second_effect(
    tmp_path: Path,
) -> None:
    health = {"healthy": False}
    roots = {
        "repository_root_cid": _cid("repository"),
        "policy_root": _cid("policy"),
        "run_cid": _cid("run"),
    }
    calls = 0

    def recover(context):
        nonlocal calls
        calls += 1
        roots["run_cid"] = _cid("changed-run")
        health["healthy"] = True
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        health=health,
        handlers={RescueOperation.RETRY: recover},
        root_probe=lambda: dict(roots),
    ).unstall(evidence={"heartbeat": {"lane_id": "lane-1", "stale": True}})

    assert result["quarantined"]
    assert result["reason"] == "semantic_root_drift_after_deterministic_recovery"
    assert calls == 1


def test_corrupt_runtime_state_is_quarantined_and_repair_is_visible(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state" / "autonomous-unstall-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema": AUTONOMOUS_UNSTALL_STATE_SCHEMA,
                "incidents": {},
                "rescue_runtime": {"provider_calls": "many"},
                "updated_at_ms": NOW,
            }
        ),
        encoding="utf-8",
    )
    health = {"healthy": False}

    def repair(context):
        health["healthy"] = True
        return {
            "succeeded": True,
            "observed_effects": context.action.expected_effects,
        }

    result = _coordinator(
        tmp_path,
        health=health,
        handlers={RescueOperation.REPAIR_ORPHANED_LOCK: repair},
    ).unstall(evidence={"lock": {"lock_id": "l1", "orphaned": True}})

    assert result["recovered"]
    assert (
        result["state_repair"]["reason"]
        == "corrupt_coordination_state_quarantined"
    )
    assert list((tmp_path / "state").glob("*.corrupt-*"))


def test_implementation_quarantine_is_scope_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    config = PortalSupervisorConfig(
        todo_path=tmp_path / "tasks.todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        repo_root=tmp_path,
    )
    supervisor = PortalImplementationSupervisor(config)
    PortalTaskState(
        active_task_id="TASK-1",
        active_attempt=1,
        implementation_in_progress=True,
        ready_count=2,
    ).save(config.state_path)

    unrelated = supervisor._quarantine_autonomous_unstall_scope(
        ("TASK-2",),
        _cid("unrelated-incident"),
        "unrelated_failure",
    )
    assert unrelated["task_id"] == ""
    assert PortalTaskState.load(config.state_path).active_task_id == "TASK-1"

    first = supervisor._quarantine_autonomous_unstall_scope(
        ("lane:implementation",),
        _cid("active-incident"),
        "active_lane_failure",
    )
    duplicate = supervisor._quarantine_autonomous_unstall_scope(
        ("lane:implementation",),
        _cid("active-incident"),
        "active_lane_failure",
    )
    strategy = json.loads(config.strategy_path.read_text(encoding="utf-8"))

    assert first["task_id"] == "TASK-1"
    assert duplicate["deduplicated"]
    assert strategy["blocked_tasks"] == ["TASK-1"]
    assert sum(
        item["incident_cid"] == _cid("active-incident")
        for item in strategy["autonomous_unstall_quarantines"]
    ) == 1
    assert PortalTaskState.load(config.state_path).active_task_id == ""


def test_rescue_policy_cannot_be_implicit() -> None:
    with pytest.raises(ValueError, match="operating_policy_id"):
        AutonomousUnstallPolicy(
            rescue_preview_enabled=True,
            allow_provider_calls=True,
        )


def test_watchdog_runs_unified_ladder_and_rechecks_lane_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "lane-state"
    state_dir.mkdir()
    manifest_path = tmp_path / "lanes.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tree_id": "tree-1",
                "autonomous_unstall_policy": {
                    "enabled": True,
                    "cooldown_ms": 0,
                },
                "lanes": [
                    {
                        "bundle_key": "lane-1",
                        "state_dir": str(state_dir),
                        "state_prefix": "lane_1",
                    }
                ],
                "started": [],
            }
        ),
        encoding="utf-8",
    )
    events: list[tuple[str, dict[str, Any]]] = []
    restart_calls = 0

    monkeypatch.setattr(
        watchdog_module,
        "pid_alive",
        lambda pid: pid == 123,
    )

    def restart(_lane: dict[str, Any]) -> dict[str, Any]:
        nonlocal restart_calls
        restart_calls += 1
        (state_dir / "lane_1_bundle_supervisor.pid").write_text(
            "123\n",
            encoding="utf-8",
        )
        (state_dir / "lane_1_status.json").write_text(
            json.dumps({"state": "running"}),
            encoding="utf-8",
        )
        return {
            "restarted": True,
            "new_pid": 123,
            "receipt_id": "restart-receipt-1",
        }

    report = SupervisorWatchdog(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        lifecycle_restart=restart,
        control_event_publisher=lambda kind, payload: events.append(
            (kind, dict(payload))
        ),
    )._check_cycle()

    lane_report = report["reports"][0]
    result = lane_report["autonomous_unstall"]
    assert lane_report["action"] == "autonomous_unstall_recovered"
    assert result["recovered"]
    assert result["work_complete"] is False
    assert restart_calls == 1
    assert any(
        attempt["operation"] == "restart_lane"
        and attempt["outcome"] == "succeeded"
        for attempt in result["deterministic"]["attempts"]
    )
    assert lane_report["pid_check"]["alive"]
    assert not lane_report["heartbeat_check"]["stale"]
    assert events[-1][0] == "autonomous_unstall_recovered"
