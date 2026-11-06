"""Explicit infrastructure policy retains durable delay and spent attempts.

Only private JSON fixtures are changed. Native observations, coding processes
and service commands are doubles; their existing authority tests stay separate.
"""
import json
import importlib.util
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


def _case(tmp_path, infrastructure, monkeypatch, *, finite=True):
    board = dict(id="science", cwd=str(tmp_path), hold_files=[], probe=dict(argv=["fixture-probe"]))
    if finite or infrastructure:
        # Infrastructure policy always requires a finite cumulative cap. Its
        # extended-history case uses cap10/history7; only legacy is uncapped.
        board["max_repair_attempts"] = 3 if finite else 10
    if infrastructure:
        board["recovery_policy"] = dict(profile="infrastructure-only/v1",
            repairable_stalls=["probe_failed"], max_actions_per_window=3,
            window_seconds=86400, circuit_breaker_seconds=3600)
    cfg = dict(state_dir=str(tmp_path), boards=[board], repair_worker=dict(cwd=str(tmp_path),
        argv=["fixture-codex", "exec"], retry_seconds=1800, max_backoff_seconds=21600))
    path = tmp_path / "repairs" / "science" / "job.json"
    monkeypatch.setattr(repair, "board_is_complete", lambda *_: False)
    monkeypatch.setattr(repair, "board_wait_stall", lambda *_: None)
    return cfg, board, path


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy", "infrastructure"])
@pytest.mark.parametrize("change", ["quick_failure", "route_change", "workspace_change"])
def test_future_due_cannot_be_bypassed_by_infrastructure_churn(tmp_path, monkeypatch, infrastructure, change):
    cfg, board, path = _case(tmp_path, infrastructure, monkeypatch)
    job = dict(status="queued", attempts=1, queued_at=1, last_started_at=1000,
        finished_at=1500, returncode=2, next_attempt_at=9000,
        repair_route=repair.LLM_ROUTER_ROUTE, repair_workspace=str(tmp_path))
    if change == "quick_failure":
        job["finished_at"] = 1010
    elif change == "route_change":
        job["repair_route"] = "codex_exec"
    else:
        job["repair_workspace"] = str(tmp_path / "old-engine")
    write_json(path, job)
    before = path.read_bytes()
    selected = repair.next_job(cfg, 2000)
    assert (selected is None) is infrastructure
    assert path.read_bytes() == before
    selected = repair.next_job(cfg, 9000)
    assert selected and selected[0]["id"] == board["id"]
    assert read_json(path)["attempts"] == 1


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy", "infrastructure"])
def test_repeated_enqueue_does_not_shorten_infrastructure_route_migration_delay(tmp_path, monkeypatch, infrastructure):
    cfg, _, path = _case(tmp_path, infrastructure, monkeypatch)
    monkeypatch.setattr(repair.time, "time", lambda: 2000)
    monkeypatch.setattr(repair, "infrastructure_repair_allowed", lambda *_: True)
    write_json(path, dict(status="queued", attempts=1, next_attempt_at=9000,
                         repair_route="codex_exec", last_started_at=1000))
    incident = tmp_path / "incident.json"
    for signature in ("old-source", "new-source", "same-stall-again"):
        write_json(incident, dict(board_id="science", signature=signature, observation=dict(health="unknown")))
        assert repair.enqueue(cfg, incident)["status"] == "queued"
        current = read_json(path)
        assert current["attempts"] == 1
        assert current["next_attempt_at"] == (9000 if infrastructure else 2000)
        assert current["latest_incident"]["signature"] == signature


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy", "infrastructure"])
@pytest.mark.parametrize("trigger", ["fresh_evidence", "productive_continuation"])
def test_reconciliation_observes_evidence_without_advancing_infrastructure_delay(tmp_path, monkeypatch, infrastructure, trigger):
    cfg, _, path = _case(tmp_path, infrastructure, monkeypatch)
    before, after = dict(source="before"), dict(source="after")
    baseline = dict(board_id="science", health="degraded", details=dict(fixture_evidence=before))
    sample = dict(board_id="science", health="degraded", details=dict(fixture_evidence=after))
    write_json(path, dict(status="queued", attempts=1, last_started_at=1000, finished_at=1500,
        next_attempt_at=9000, latest_probe=baseline, latest_incident=dict(observation=baseline)))
    write_json(tmp_path / "science" / "state.json", dict(board_id="science", health="degraded",
        observed_at=1999, observation=sample))
    monkeypatch.setattr(repair, "repair_evidence", lambda observation: observation["details"]["fixture_evidence"])
    monkeypatch.setattr(repair, "_new_repair_evidence", lambda a, b: trigger == "fresh_evidence" and a != b)
    progress = dict(attempt=dict(attempts=1), before=before, after=after)
    monkeypatch.setattr(repair, "_completed_attempt_progress", lambda *_: progress if trigger == "productive_continuation" else None)
    monkeypatch.setattr(repair, "verify_job_recovery", lambda *_: dict(verified=False))
    results = repair.reconcile_queued_jobs(cfg, 2000, runner=lambda *_args, **_kwargs:
        dict(returncode=0, stdout=json.dumps(sample)))
    current = read_json(path)
    assert current["attempts"] == 1 and current["status"] == "queued"
    assert current["next_attempt_at"] == (9000 if infrastructure else 2000)
    assert any(row["status"] == "continuation_advanced" for row in results) is (not infrastructure)
    assert ("productive_continuation" in current) is (not infrastructure and trigger == "productive_continuation")


def test_direct_infrastructure_worker_cannot_bypass_future_due_with_untrusted_board_copy(tmp_path, monkeypatch):
    cfg, _, path = _case(tmp_path, True, monkeypatch)
    monkeypatch.setattr(repair.time, "time", lambda: 2000)
    write_json(path, dict(status="queued", attempts=1, next_attempt_at=9000))
    before = path.read_bytes()
    monkeypatch.setattr(repair, "command", lambda *_a, **_k: pytest.fail("future infrastructure job must not call services"))
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *_a, **_k: pytest.fail("future infrastructure job must not launch"))
    result = repair.run_job(cfg, dict(id="science", cwd=str(tmp_path)), path)
    assert result == dict(status="repair_backoff", board_id="science", next_attempt_at=9000)
    assert path.read_bytes() == before


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy", "infrastructure"])
def test_preflight_refusal_after_clock_reversal_preserves_infrastructure_deadline(tmp_path, monkeypatch, infrastructure):
    cfg, board, path = _case(tmp_path, infrastructure, monkeypatch)
    write_json(path, dict(status="queued", attempts=1, next_attempt_at=9000))
    clock = [10000]
    monkeypatch.setattr(repair.time, "time", lambda: clock[0])
    calls = []
    def service(spec, **_kwargs):
        assert spec["argv"][:3] == ["systemctl", "--user", "is-active"]
        calls.append(spec)
        return dict(returncode=3, stdout="inactive")
    def refusal(_policy):
        clock[0] = 2000
        return "fixture_refusal_after_clock_reversal"
    monkeypatch.setattr(repair, "command", service)
    monkeypatch.setattr(repair, "_launch_preflight", refusal)
    monkeypatch.setattr(repair.subprocess, "Popen", lambda *_a, **_k: pytest.fail("preflight refusal must not spawn"))
    result = repair.run_job(cfg, board, path)
    current = read_json(path)
    assert len(calls) == 1 and result["status"] == "launch_deferred"
    assert result["reason"] == "fixture_refusal_after_clock_reversal"
    assert current["next_attempt_at"] == (9000 if infrastructure else 2060)
    assert current["attempts"] == 1 and current["launch_failures"] == 1


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy", "infrastructure"])
@pytest.mark.parametrize("finite", [False, True], ids=["extended_history", "finite"])
@pytest.mark.parametrize("verified", [False, True], ids=["productive_failure", "verified_recovery"])
def test_worker_completion_preserves_infrastructure_backoff_and_cumulative_attempts(
    tmp_path, monkeypatch, infrastructure, finite, verified,
):
    from ipfs_accelerate_py.agent_supervisor.rescue import diagnostic_handoff

    cfg, board, path = _case(tmp_path, infrastructure, monkeypatch, finite=finite)
    prior_attempts = 1 if finite else 7
    write_json(path, dict(status="queued", attempts=prior_attempts, last_started_at=1000,
        finished_at=1010, returncode=2, next_attempt_at=0,
        latest_incident=dict(board_id="science", observation=dict(health="stalled"))))
    clock = [2000]
    monkeypatch.setattr(repair.time, "time", lambda: clock[0])
    monkeypatch.setattr(repair, "_launch_preflight", lambda *_: None)
    monkeypatch.setattr(repair, "_storage_preflight", lambda *_: None)
    monkeypatch.setattr(repair, "_repair_temporary_directory", lambda *_: tmp_path)
    monkeypatch.setattr(repair, "_continuation_context", lambda *_: {})
    monkeypatch.setattr(repair, "repair_prompt", lambda *_: "only a fictional repair")
    monkeypatch.setattr(diagnostic_handoff, "load_diagnostic_handoff", lambda *_a, **_k: None)
    sample = dict(board_id="science", health="degraded", details={})
    def command(spec, **_kwargs):
        if spec["argv"][0] == "systemctl":
            assert spec["argv"][2] == "is-active"
            return dict(returncode=3, stdout="inactive")
        assert spec["argv"] == ["fixture-probe"]
        return dict(returncode=0, stdout=json.dumps(sample))
    monkeypatch.setattr(repair, "command", command)
    verification_calls = []
    def verification(*_args):
        verification_calls.append(1)
        # The initial infrastructure check must remain unhealthy so a due
        # fictional coding job actually reaches terminal accounting.
        final = len(verification_calls) > int(infrastructure)
        return dict(verified=verified and final, reason="fixture_result", wait=False)
    monkeypatch.setattr(repair, "verify_job_recovery", verification)
    monkeypatch.setattr(repair, "_completed_attempt_progress", lambda *_: dict(
        attempt=dict(attempts=prior_attempts), before=dict(source="old"), after=dict(source="new")))
    launches = []
    class Process:
        def __init__(self, argv, **kwargs):
            assert argv[0] == "systemd-run"
            assert kwargs["stdin"].read() == b"only a fictional repair"
            launches.append(read_json(path)["attempts"])
        def wait(self, timeout):
            clock[0] = 2100
            return 2
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    result = repair.run_job(cfg, board, path)
    current = read_json(path)
    charged = prior_attempts + 1 if infrastructure or finite else 1
    assert launches == [charged]
    assert current["attempts"] == (charged if infrastructure or finite or not verified else 0)
    if verified:
        assert result["status"] == "verified_healthy" and current["next_attempt_at"] == 0
    else:
        assert result["status"] == "queued"
        expected = 2000 + min(21600, 1800 * 2 ** min(charged - 1, 4)) if infrastructure else 2400
        assert current["next_attempt_at"] == expected
        assert ("productive_continuation" in current) is (not infrastructure)


def _actual_pause_after_runnable_selection(tmp_path, monkeypatch, infrastructure):
    """Reuse the owning monitor/probe reconstruction instead of a paused bool."""
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog as fleet

    source = Path(repair.__file__).resolve().parents[3] / "test/api/test_agent_supervisor_service_fleet_probe.py"
    spec = importlib.util.spec_from_file_location("_repair_pause_original_probe_fixtures", source)
    assert spec and spec.loader
    fixtures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixtures
    spec.loader.exec_module(fixtures)
    monkeypatch.setattr(repair.time, "time", lambda: fixtures.NOW)
    ready_report = fixtures.report()
    for lane in ready_report["lanes"]:
        lane["health"]["evidence"].update(global_ready_count=2, task_count=137)
    ready_report["lanes"][0]["health"]["evidence"].update(ready_count=2, selectable_ready_count=2)
    ready = fixtures.observe(tmp_path, ready_report)
    paused = fixtures.observe(tmp_path, fixtures._backpressure_report())
    assert ready["details"]["service_fleet"]["work_availability"] == "runnable_work"
    assert paused["details"]["service_fleet"]["work_availability"] == "backpressure_wait"
    board = dict(id="legal-ir", cwd=str(tmp_path), hold_files=[], stall_seconds=60,
        max_repair_attempts=3, probe=dict(argv=["fixture-probe"]), service_fleet_binding=dict(
        board_namespace=fixtures.NAMESPACE, lanes=fixtures.UNITS, max_age_seconds=180))
    if infrastructure:
        board["recovery_policy"] = dict(profile="infrastructure-only/v1",
            repairable_stalls=["service_fleet_unhealthy", "stalled_no_progress"],
            max_actions_per_window=3, window_seconds=86400, circuit_breaker_seconds=3600)
    previous = dict(last_progress_at=100, progress_token=ready["progress_token"], work_available_since=900)
    cached = fleet.assess(ready, previous, board, fixtures.NOW)
    assert cached["health"] == "stalled"
    root = tmp_path / "watch"
    cfg = dict(state_dir=str(root), boards=[board], repair_worker=dict(cwd=str(tmp_path),
        argv=["fixture-codex", "exec"], retry_seconds=1800, max_backoff_seconds=21600))
    write_json(root / board["id"] / "state.json", dict(board_id=board["id"],
        observed_at=fixtures.NOW, health=cached["health"], observation=cached["observation"]))
    budget = root / board["id"] / "infrastructure-budget.json"
    write_json(budget, dict(schema="agent-supervisor/infrastructure-budget@1", board_id=board["id"],
        window_started_at=900, actions=3, total_actions=3, circuit_open_until=4500))
    path = root / "repairs" / board["id"] / "job.json"
    write_json(path, dict(status="queued", attempts=1, last_started_at=100, finished_at=200,
        next_attempt_at=900, latest_probe=cached["observation"], latest_incident=dict(
        board_id=board["id"], observation=cached["observation"])))
    return cfg, board, path, budget, cached["observation"], paused


@pytest.mark.parametrize("infrastructure", [False, True], ids=["legacy_diagnostic", "infrastructure"])
@pytest.mark.parametrize("stage", ["reconciliation", "prelaunch", "completion"])
def test_real_monitor_probe_pause_class_survives_all_repair_retirements(
    tmp_path, monkeypatch, infrastructure, stage,
):
    from ipfs_accelerate_py.agent_supervisor.rescue import diagnostic_handoff

    cfg, board, path, budget, runnable, paused = _actual_pause_after_runnable_selection(
        tmp_path, monkeypatch, infrastructure)
    original_budget = budget.read_bytes()
    original = read_json(path)
    launches, commands = [], []
    if stage == "reconciliation":
        # Force only the existing evidence-recheck opportunity. The observed
        # pause and recovery verdict still come from the actual owners.
        monkeypatch.setattr(repair, "_completed_attempt_progress", lambda *_: dict(
            attempt=dict(attempts=1), before={}, after=repair.repair_evidence(runnable)))
        monkeypatch.setattr(repair, "command", lambda *_a, **_k: pytest.fail("reconciliation must use its explicit probe"))
        monkeypatch.setattr(repair.subprocess, "Popen", lambda *_a, **_k: pytest.fail("wait cannot spawn"))
        rows = repair.reconcile_queued_jobs(cfg, 1000, runner=lambda *_a, **_k:
            dict(returncode=0, stdout=json.dumps(paused)))
        assert rows == [dict(board_id="legal-ir", status="retired_wait", wait_stall="service_fleet_backpressure_wait")]
        result = rows[0]
    else:
        monkeypatch.setattr(repair, "_launch_preflight", lambda *_: None)
        monkeypatch.setattr(repair, "_storage_preflight", lambda *_: None)
        monkeypatch.setattr(repair, "_repair_temporary_directory", lambda *_: tmp_path)
        monkeypatch.setattr(repair, "_continuation_context", lambda *_: {})
        monkeypatch.setattr(repair, "repair_prompt", lambda *_: "fictional repair only")
        monkeypatch.setattr(diagnostic_handoff, "load_diagnostic_handoff", lambda *_a, **_k: None)
        probes = []
        def command(spec, **_kwargs):
            commands.append(spec["argv"])
            if spec["argv"][0] == "systemctl":
                assert spec["argv"][2] == "is-active"  # No start/restart/stop.
                return dict(returncode=3, stdout="inactive")
            assert spec["argv"] == ["fixture-probe"]
            probes.append(1)
            sample = paused if stage == "prelaunch" or len(probes) > 1 else runnable
            return dict(returncode=0, stdout=json.dumps(sample))
        class Process:
            def __init__(self, argv, **_kwargs):
                assert stage == "completion" and argv[0] == "systemd-run"
                launches.append(argv)
            def wait(self, timeout):
                return 2
        monkeypatch.setattr(repair, "command", command)
        monkeypatch.setattr(repair.subprocess, "Popen", Process)
        result = repair.run_job(cfg, board, path)
        assert len(launches) == int(stage == "completion")
        assert commands[0][:3] == ["systemctl", "--user", "is-active"]
        assert commands.count(["fixture-probe"]) == (1 if stage == "prelaunch" else 2)
    current = read_json(path)
    assert result["status"] == current["status"] == "retired_wait"
    assert result["wait_stall"] == current["wait_stall"] == "service_fleet_backpressure_wait"
    assert current["verification"]["wait"] is True
    for name in ("native_admission", "task_completion_authority", "goal_completion_authority", "completion_authority"):
        assert current["verification"][name] is False
    if stage == "completion":
        assert current["attempts"] == 2 and current["next_attempt_at"] == 4600
    else:
        assert current["attempts"] == original["attempts"] == 1
        assert current["next_attempt_at"] == original["next_attempt_at"] == 900
    assert budget.read_bytes() == original_budget
