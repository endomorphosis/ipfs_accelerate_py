"""Queue continuation must follow fresh evidence, not heartbeat or agent text."""
import copy
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import read_json, write_json


def observation(*, completed=19, health="blocked", admitted=True, head="old", **extra):
    return {"board_id": "pctdd", "health": health, "busy": True,
            "completion_candidate": False, "progress_token": "stable",
            "details": {"authenticated_task_observation": admitted,
                        "task_counts": {"completed": completed}, "blocked_task_ids": [],
                        "source_heads": {".": head}}, **extra}


@pytest.fixture
def queue(tmp_path):
    board = {"id": "pctdd", "cwd": str(tmp_path), "probe": {"argv": ["native-status"]}, "hold_files": []}
    config = {"state_dir": str(tmp_path), "boards": [board], "repair_worker": {"cwd": str(tmp_path)}}
    path = tmp_path / "repairs/pctdd/job.json"
    job = {"board_id": "pctdd", "status": "queued", "attempts": 7, "last_started_at": 1000,
           "finished_at": 1500, "next_attempt_at": 22600, "report_path": str(path.parent / "report.json"),
           "latest_probe": observation(), "latest_incident": {"observation": observation()}}
    write_json(path, job)
    return config, board, path, job


def cache(config, sample, *, at=1900):
    write_json(Path(config["state_dir"]) / "pctdd/state.json",
               {"board_id": "pctdd", "observed_at": at, "health": sample["health"], "observation": sample})


def probe(sample):
    return {"returncode": 0, "stdout": json.dumps(sample)}


def test_recovered_job_retires_only_after_new_admitted_probe(queue):
    config, _, path, job = queue
    sample = observation(health="healthy", completed=20)
    cache(config, sample)
    calls = []
    def runner(spec, **kwargs):
        calls.append(spec)
        return probe(sample)
    result = repair.reconcile_queued_jobs(config, 2000, runner=runner)
    current = read_json(path)
    assert result == [{"board_id": "pctdd", "status": "verified_healthy"}]
    assert len(calls) == 1
    assert current["status"] == "verified_healthy"
    assert current["prior_attempts"] == 7 and current["attempts"] == 0
    assert current["report_path"] == job["report_path"]
    assert repair.next_job(config, 23000) is None


@pytest.mark.parametrize("fresh", [observation(), observation(health="healthy", admitted=False),
                                  {**observation(health="healthy"), "details": None}])
def test_cached_health_and_unadmitted_fresh_health_do_not_retire_job(queue, fresh):
    config, _, path, job = queue
    cache(config, observation(health="healthy"))
    repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: probe(fresh))
    current = read_json(path)
    assert current["status"] == "queued" and current["attempts"] == 7
    assert current["next_attempt_at"] == job["next_attempt_at"]


@pytest.mark.parametrize("sample", [observation(completed=20), observation(head="reviewed-successor", admitted=False)])
def test_native_progress_or_source_successor_advances_continuation_once(queue, sample):
    config, _, path, job = queue
    cache(config, sample, at=1500)
    result = repair.reconcile_queued_jobs(config, 1600, runner=lambda *a, **kw: probe(sample))
    current = read_json(path)
    assert result[0]["status"] == "continuation_advanced"
    assert current["next_attempt_at"] == 1800  # five minutes after prior finish
    assert current["prior_next_attempt_at"] == 22600 and current["attempts"] == 7
    assert current["report_path"] == job["report_path"]
    assert current["latest_incident"]["observation"] == sample
    # Even if another component defers the job, the same evidence cannot replay
    # its cooldown override. No native task budget appears in the queue writes.
    current["next_attempt_at"] = 22600
    write_json(path, current)
    cache(config, sample, at=1900)
    assert repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("duplicate recheck")) == []
    assert read_json(path)["next_attempt_at"] == 22600


def test_noisy_progress_tokens_and_owner_heartbeats_do_not_advance_queue(queue):
    config, _, path, _ = queue
    sample = observation(progress_token="different", reason_codes=["lane_restarting"])
    sample["details"].update(event_cursor=5000, observed_at="new", owner={"pid": 555, "cpu_ticks": 9000})
    cache(config, sample)
    assert repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("heartbeat-only probe")) == []
    assert read_json(path)["next_attempt_at"] == 22600


def test_unauthenticated_task_count_cannot_advance_queue(queue):
    config, _, path, _ = queue
    cache(config, observation(completed=5000, admitted=False))
    repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("unadmitted counts"))
    assert read_json(path)["next_attempt_at"] == 22600


@pytest.mark.parametrize("at", [1000, 2100])
def test_stale_or_future_watchdog_sample_does_not_select_probe(queue, at):
    config, _, _, _ = queue
    cache(config, observation(health="healthy"), at=at)
    assert repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("stale selector")) == []


@pytest.mark.parametrize("fence", ["hold", "running"])
def test_initial_hold_or_running_job_is_untouched(queue, fence):
    config, board, path, job = queue
    if fence == "hold":
        hold = path.parent / "HOLD"
        hold.touch()
        board["hold_files"] = [str(hold)]
    else:
        job["status"] = "running"
        write_json(path, job)
    cache(config, observation(health="healthy"))
    before = path.read_bytes()
    assert repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("fenced probe")) == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("fence", ["hold", "new_attempt"])
def test_fence_arriving_during_probe_prevents_queue_mutation(queue, fence):
    config, board, path, _ = queue
    hold = path.parent / "HOLD"
    board["hold_files"] = [str(hold)]
    sample = observation(health="healthy")
    cache(config, sample)
    expected = []
    def runner(*args, **kwargs):
        if fence == "hold":
            hold.touch()
        else:
            job = read_json(path)
            job.update(status="running", attempts=8, last_started_at=1999)
            write_json(path, job)
        expected.append(path.read_bytes())
        return probe(sample)
    assert repair.reconcile_queued_jobs(config, 2000, runner=runner) == []
    assert path.read_bytes() == expected[0]


def test_reconciliation_never_publishes_from_completion_counts(queue, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion
    config, board, path, _ = queue
    board["publication"] = {"board_id": "pctdd"}
    sample = observation(health="complete", completed=54, completion_candidate=True)
    cache(config, sample)
    monkeypatch.setattr(fleet_completion, "publish_completed_board", lambda *a: pytest.fail("recheck cannot publish"))
    repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: probe(sample))
    assert read_json(path)["status"] == "queued"
    assert read_json(path)["next_attempt_at"] == 2000


def test_stale_healthy_idle_sample_cannot_hide_original_task_stall(queue):
    config, _, path, job = queue
    job["latest_incident"]["observation"].update(health="stalled", reason_codes=["no_task_progress"])
    write_json(path, job)
    sample = observation(health="healthy", busy=False)
    cache(config, sample)
    repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: probe(sample))
    assert read_json(path)["status"] == "queued"


def test_rejected_probe_does_not_reuse_cached_progress_and_is_throttled(queue):
    config, _, path, _ = queue
    cache(config, observation(completed=20))
    repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: {"returncode": 1, "stdout": ""})
    assert read_json(path)["next_attempt_at"] == 22600
    assert repair.reconcile_queued_jobs(config, 2030, runner=lambda *a, **kw: pytest.fail("repeated failed recheck")) == []


def test_increasing_goals_and_completion_regression_do_not_trigger_continuation():
    prior = {"completed": 20, "receipts": 20, "unsettled_goals": 5}
    assert not repair._new_repair_evidence(prior, {"completed": 19, "receipts": 19, "unsettled_goals": 6})
    assert repair._new_repair_evidence(prior, {**prior, "unsettled_goals": 4})
    snapshot = observation()
    snapshot["details"]["blocked_task_ids"] = ["PCTDD-007", "PCTDD-005", "PCTDD-007"]
    reverse = copy.deepcopy(snapshot)
    reverse["details"]["blocked_task_ids"].reverse()
    assert repair.repair_evidence(snapshot) == repair.repair_evidence(reverse)


def test_restored_admission_revisits_queued_failure_without_heartbeat_replay(queue):
    config, _, path, job = queue
    job["latest_probe"] = observation(health="stopped", admitted=False)
    # A later admitted incident does not erase the failed post-job observation.
    job["latest_incident"]["observation"] = observation(completed=19)
    write_json(path, job)
    sample = observation(completed=20)
    cache(config, sample)
    result = repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: probe(sample))
    assert result[0]["status"] == "continuation_advanced"
    current = read_json(path)
    assert current["attempts"] == 7 and current["next_attempt_at"] == 2000
    current["next_attempt_at"] = 22600
    write_json(path, current)
    # Count regression and mere repeated admission cannot replay the trigger.
    cache(config, observation(completed=19), at=2200)
    assert repair.reconcile_queued_jobs(config, 2300, runner=lambda *a, **kw: pytest.fail("old admission")) == []
    # New independently admitted progress is still actionable.
    newer = observation(completed=21)
    cache(config, newer, at=2300)
    assert repair.reconcile_queued_jobs(config, 2400, runner=lambda *a, **kw: probe(newer))[0]["status"] == "continuation_advanced"


def test_dangling_hold_fences_watchdog_and_queue(queue):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import tick_board
    config, board, path, _ = queue
    hold = path.parent / "HOLD"
    hold.symlink_to(path.parent / "missing-operator-evidence")
    board["hold_files"] = [str(hold)]
    sample = observation(health="healthy")
    cache(config, sample)
    assert repair.next_job(config, 30000) is None
    assert repair.queue_status(config, 30000)["held"][0]["board_id"] == "pctdd"
    assert repair.reconcile_queued_jobs(config, 2000, runner=lambda *a, **kw: pytest.fail("held recheck")) == []
    assert repair.run_job(config, board, path)["status"] == "operator_hold"
    assert repair.verify_job_recovery(board, {}, sample, path.parent)["reason"] == "operator_hold"
    state = tick_board(board, Path(config["state_dir"]), apply=True,
                       runner=lambda *a, **kw: probe(sample), now=2000)
    assert state["health"] == "operator_hold" and state["holds"] == [str(hold)]
