"""A verified partial repair earns a bounded continuation without claiming success."""

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair
from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import (
    read_json,
    write_json,
)


def native(*, head="before", receipts=50, goals=32, admitted=True):
    return {
        "board_id": "spar",
        "health": "blocked",
        "reason_codes": ["goals_unsettled"],
        "completion_candidate": False,
        "progress_token": "same",
        "busy": False,
        "details": {
            "authenticated_task_observation": admitted,
            "source_heads": {".": head},
            "task_counts": {"completed": 51},
            "completion_receipt_count": receipts,
            "unsettled_goal_count": goals,
            "blocked_task_ids": [],
        },
    }


def response(observation):
    return {"returncode": 0, "stdout": json.dumps(observation)}


@pytest.fixture
def queued(tmp_path):
    board = {
        "id": "spar",
        "cwd": str(tmp_path),
        "hold_files": [],
        "probe": {"argv": ["configured-native-status"]},
    }
    config = {
        "state_dir": str(tmp_path),
        "boards": [board],
        "repair_worker": {"cwd": str(tmp_path), "argv": ["coding-worker"]},
    }
    path = tmp_path / "repairs/spar/job.json"
    job = {
        "board_id": "spar",
        "status": "queued",
        "attempts": 6,
        "next_attempt_at": 0,
        "latest_incident": {"observation": native()},
    }
    write_json(path, job)
    return config, board, path, job


def run(queued, monkeypatch, before, after, *, code=0, during=None, seconds=2000):
    config, board, path, _ = queued
    clock = [1000]
    samples = iter([before, after])
    calls = []

    def command(spec, **kwargs):
        if spec["argv"][0] == "systemctl":
            return {"returncode": 3, "stdout": "inactive"}
        assert spec == board["probe"]
        calls.append(spec)
        sample = next(samples)
        if isinstance(sample, Exception):
            raise sample
        return response(sample)

    class Process:
        def __init__(self, *args, **kwargs):
            started = read_json(path)
            assert started["status"] == "running"
            assert started["started_evidence_attempt"]["attempts"] == 7
            # Deliberately false worker prose is never evidence of recovery.
            Path(started["report_path"]).write_text('{"status":"complete"}')

        def wait(self, timeout):
            clock[0] += seconds
            if during:
                during(board, path)
            return code

    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(repair.shutil, "which", lambda executable: "/test/" + executable)
    monkeypatch.setattr(repair.time, "time", lambda: clock[0])
    monkeypatch.setattr(repair.subprocess, "Popen", Process)
    result = repair.run_job(config, board, path)
    assert len(calls) == 2
    return result, read_json(path)


@pytest.mark.parametrize(
    "after", [native(head="reviewed-successor"), native(receipts=51), native(goals=31)]
)
@pytest.mark.parametrize("code", [0, 1, 124])
def test_fresh_partial_fix_continues_even_after_failed_worker_exit(
    queued, monkeypatch, after, code
):
    result, job = run(queued, monkeypatch, native(), after, code=code)
    assert result["status"] == "queued" and job["attempts"] == 7
    assert job["verification"] == {"verified": False, "reason": "board_not_healthy"}
    assert job["next_attempt_at"] == job["finished_at"] + 300 == 3300
    assert job["productive_continuation"]["before"] == repair.repair_evidence(native())
    assert job["productive_continuation"]["after"] == repair.repair_evidence(after)
    assert (
        job["productive_continuation"]["attempt"]["report_path"] == job["report_path"]
    )
    assert repair.next_job(queued[0], 3299) is None
    assert repair.next_job(queued[0], 3300)[0]["id"] == "spar"


@pytest.mark.parametrize(
    "before,after",
    [
        (native(), native()),
        (native(admitted=False), native(receipts=51)),
        (native(), native(receipts=51, admitted=False)),
        (native(), native(receipts=49, goals=33)),
        (RuntimeError("owner unavailable"), native(receipts=51)),
    ],
)
def test_no_authenticated_progress_keeps_backoff_despite_complete_report(
    queued, monkeypatch, before, after
):
    _, job = run(queued, monkeypatch, before, after)
    assert job["next_attempt_at"] == 22600 and job["attempts"] == 7
    assert "productive_continuation" not in job


def test_job_longer_than_backoff_still_waits_five_minutes(queued, monkeypatch):
    _, job = run(queued, monkeypatch, native(), native(), seconds=22000)
    assert job["next_attempt_at"] == 23300


def test_hold_arriving_during_productive_job_prevents_advance(queued, monkeypatch):
    def hold(board, path):
        hold = path.parent / "HOLD"
        hold.touch()
        board["hold_files"] = [str(hold)]

    _, job = run(queued, monkeypatch, native(), native(receipts=51), during=hold)
    assert job["next_attempt_at"] == 22600 and "productive_continuation" not in job


def test_new_attempt_cannot_be_overwritten_by_old_finish(queued, monkeypatch):
    def replace(board, path):
        job = read_json(path)
        job.update(attempts=8, last_started_at=2999, report_path="new-attempt.json")
        write_json(path, job)

    result, job = run(
        queued, monkeypatch, native(), native(receipts=51), during=replace
    )
    assert result["status"] == "completion_superseded"
    assert job["status"] == "running" and job["attempts"] == 8
    assert "finished_at" not in job


def legacy(queued):
    config, _, path, job = queued
    after = native(head="reviewed-successor", receipts=51)
    job.update(
        attempts=7,
        last_started_at=1000,
        finished_at=1500,
        next_attempt_at=22600,
        reconciled_at=999.3,
        reconciled_evidence=repair.repair_evidence(native()),
        report_path=str(path.parent / "report-1000-7.json"),
        latest_probe=after,
    )
    # The latest watchdog incident has already consumed the changed source.
    job["latest_incident"] = {"observation": after, "observed_at": 1550}
    write_json(path, job)
    cache(config, after, 1550)
    return after


def cache(config, sample, now):
    write_json(
        Path(config["state_dir"]) / "spar/state.json",
        {
            "board_id": "spar",
            "health": sample["health"],
            "observation": sample,
            "observed_at": now,
        },
    )


def test_finished_legacy_attempt_earns_one_continuation_with_fresh_matching_query(
    queued,
):
    config, _, path, _ = queued
    after = legacy(queued)
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        return response(after)

    result = repair.reconcile_queued_jobs(config, 1600, runner=runner)
    job = read_json(path)
    assert len(calls) == 1 and result[0]["status"] == "continuation_advanced"
    assert job["attempts"] == 7 and job["next_attempt_at"] == 1800
    assert job["prior_next_attempt_at"] == 22600
    assert job["continuation_reason"] == "authenticated_productive_repair"
    marker = copy.deepcopy(job["productive_continuation"])
    job["next_attempt_at"] = 22600
    write_json(path, job)
    cache(config, after, 1900)
    assert (
        repair.reconcile_queued_jobs(
            config, 2000, runner=lambda *a, **kw: pytest.fail("replayed finish")
        )
        == []
    )
    assert read_json(path)["productive_continuation"] == marker
    assert read_json(path)["next_attempt_at"] == 22600


@pytest.mark.parametrize(
    "mutation",
    [
        {"reconciled_at": 800},
        {"reconciled_at": 1001},
        {"reconciled_evidence": {}},
        {"reconciled_evidence": repair.repair_evidence(native(admitted=False))},
        {"finished_at": 999},
        {"finished_at": 1601},
        {"report_path": ""},
        {
            "started_evidence": repair.repair_evidence(native()),
            "started_evidence_at": 999,
            "started_evidence_attempt": {"attempts": 6},
        },
    ],
)
def test_legacy_baseline_is_bounded_and_bound_to_exact_attempt(queued, mutation):
    config, _, path, _ = queued
    legacy(queued)
    job = read_json(path)
    job.update(mutation)
    write_json(path, job)
    assert (
        repair.reconcile_queued_jobs(
            config, 1600, runner=lambda *a, **kw: pytest.fail("invalid baseline")
        )
        == []
    )
    assert read_json(path)["next_attempt_at"] == 22600


@pytest.mark.parametrize(
    "fresh",
    [
        native(),
        native(head="reviewed-successor", receipts=51, admitted=False),
        native(head="reviewed-successor", receipts=49),
    ],
)
def test_legacy_finish_requires_current_native_evidence_to_match(queued, fresh):
    config, _, path, _ = queued
    legacy(queued)
    repair.reconcile_queued_jobs(config, 1600, runner=lambda *a, **kw: response(fresh))
    assert read_json(path)["next_attempt_at"] == 22600
    assert "productive_continuation" not in read_json(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("finished_at", 1501),
        ("report_path", "different.json"),
        ("reconciled_at", 999.4),
    ],
)
def test_legacy_finish_rechecks_queue_identity_after_probe(queued, field, value):
    config, _, path, _ = queued
    after = legacy(queued)
    expected = []

    def runner(*args, **kwargs):
        job = read_json(path)
        job[field] = value
        write_json(path, job)
        expected.append(path.read_bytes())
        return response(after)

    assert repair.reconcile_queued_jobs(config, 1600, runner=runner) == []
    assert path.read_bytes() == expected[0]


def test_old_finished_evidence_does_not_revive_a_job(queued):
    config, _, path, _ = queued
    after = legacy(queued)
    cache(config, after, 24000)
    assert (
        repair.reconcile_queued_jobs(
            config, 24000, runner=lambda *a, **kw: pytest.fail("old finish")
        )
        == []
    )
    assert read_json(path)["next_attempt_at"] == 22600


@pytest.mark.parametrize("health,candidate", [("healthy", False), ("complete", True)])
@pytest.mark.parametrize("valid", [False, None, "true"])
def test_failed_configured_source_integrity_prevents_recovery_and_publication(
    tmp_path, monkeypatch, health, candidate, valid
):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_completion

    monkeypatch.setattr(
        fleet_completion,
        "publish_completed_board",
        lambda *a: pytest.fail("unverified source publication"),
    )
    observation = native()
    observation.update(health=health, completion_candidate=candidate)
    observation["details"]["source_integrity"] = {"configured": True, "valid": valid}
    result = repair.verify_job_recovery(
        {"publication": {"board_id": "spar"}}, {}, observation, tmp_path
    )
    assert result == {"verified": False, "reason": "source_integrity_not_verified"}


def test_valid_configured_source_integrity_preserves_normal_health_verification(
    tmp_path,
):
    observation = native()
    observation.update(health="healthy")
    observation["details"]["source_integrity"] = {"configured": True, "valid": True}
    assert repair.verify_job_recovery({}, {}, observation, tmp_path)["verified"] is True
