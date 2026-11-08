"""WWL-100: meaningful progress and durable incident identity.

Canonical task, validation and merge advances and accepted artifact changes
move the cursor. Log and heartbeat spam, PID changes and repeated same-stage
retries do not refresh that cursor or refund a repair budget.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_progress import (
    BUDGETS, CANONICAL_STAGES, DEFAULT_LIMITS, MEANINGFUL_REASONS, ProgressIdentityError,
    ProgressLedger, canonical_stage, incident_identity, normalize_defect,
)

RUN = "wwl-run-100"
TASK = "WWL-100"
OTHER = "WWL-101"


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact(path: str, text: str, *, accepted: bool = True) -> dict:
    return {"path": path, "digest": digest(text), "accepted": accepted}


def event(stage: str, at: float, **extra):
    body = {
        "run_id": RUN, "task_id": TASK, "stage": stage, "observed_at": at,
        "defect": {"failure_key": "native_validation_failed", "message": "pid=4 attempt=1"},
    }
    body.update(extra)
    return body


def test_canonical_stages_are_task_validation_and_merge():
    assert CANONICAL_STAGES == ("task", "validation", "merge")
    assert canonical_stage("implementation") == "task"
    assert canonical_stage("testing") == "validation"
    assert canonical_stage("promotion") == "merge"
    assert MEANINGFUL_REASONS == (
        "verified_merge", "validation_stage_advance", "canonical_task_transition", "accepted_artifact_change",
    )
    assert DEFAULT_LIMITS == {"task": 3, "validation": 3, "merge": 3, "provider": 4, "repair": 2, "recover": 2}
    with pytest.raises(ProgressIdentityError, match="canonical"):
        canonical_stage("heartbeat")


def test_stage_advances_and_accepted_artifacts_move_the_cursor_only():
    ledger = ProgressLedger(RUN)
    task = ledger.observe(event("implementation", 100, pid=10, attempt=1))
    assert task["meaningful"] is True
    assert task["reason"] == "canonical_task_transition"
    assert task["cursor_stage"] == "task"
    assert task["progress_advanced"] is True
    assert task["progress_reset"] is False
    assert task["budgets_reset"] is False
    assert task["completion_authority"] is False
    assert task["last_progress_at"] == 100
    incident = task["incident_id"]
    charged = ledger.charge(incident, "repair")
    assert charged["remaining"] == 1 and charged["spent"] == 1 and charged["budgets_reset"] is False

    token = task["progress_token"]
    progress_at = task["last_progress_at"]
    for index in range(20):
        noise = ledger.observe({
            "run_id": RUN, "kind": "log", "task_id": OTHER, "stage": "merge", "attempt": 50 + index,
            "pid": 4000 + index, "heartbeat_at": 1000 + index, "observed_at": 1000 + index,
            "log": "heartbeat still alive " + ("x" * 1000), "log_bytes": 10_000_000 + index,
            "defect": {"failure_key": "brand_new_defect", "pid": index},
            "accepted_artifacts": [artifact("artifacts/worldwide-legal-search/smuggled.json", f"noise-{index}")],
            "merge_accepted": True, "merge_receipt": digest(f"smuggle-{index}"),
            "reset_budget": False, "completion_authority": False,
        })
        beat = ledger.observe({"run_id": RUN, "kind": "heartbeat", "heartbeat_at_ms": 2_000_000 + index, "pid": 70 + index})
        process = ledger.observe({"run_id": RUN, "kind": "pid", "pid": 80 + index, "observed_at": 3000 + index})
        retry = ledger.observe(event(
            "task", 4000 + index, kind="retry", attempt=2 + index, pid=90 + index,
            heartbeat_at=4000 + index, log_bytes=index, log="retry line",
        ))
        for decision in (noise, beat, process, retry):
            assert decision["meaningful"] is False
            assert decision["progress_advanced"] is False
            assert decision["progress_reset"] is False
            assert decision["budgets_reset"] is False
            assert decision["progress_token"] == token
            assert decision["last_progress_at"] == progress_at
            assert decision["incident_id"] == incident
            assert decision["repair_budget"]["repair"] == {"spent": 1, "limit": 2, "remaining": 1}
            assert decision["completion_authority"] is False
        assert retry["same_stage_retry"] is True
        assert retry["pid_changed"] is True
        assert retry["reason"] == "same_stage_retry"

    validation = ledger.observe(event("validation", 5000, attempt=1, pid=91))
    assert validation["reason"] == "validation_stage_advance"
    assert validation["progress_advanced"] is True
    assert validation["last_progress_at"] == 5000
    assert validation["budgets_reset"] is False
    assert validation["incident_id"] != incident
    assert ledger.incidents[incident]["spent"]["repair"] == 1
    assert validation["repair_budget"]["repair"]["remaining"] == DEFAULT_LIMITS["repair"]

    changed = ledger.observe(event(
        "validation", 5100, accepted_artifacts=[artifact("artifacts/worldwide-legal-search/receipt.json", "v1")],
    ))
    assert changed["reason"] == "accepted_artifact_change"
    assert changed["progress_token"] != validation["progress_token"]
    assert changed["last_progress_at"] == 5100
    assert ledger.incidents[incident]["spent"]["repair"] == 1
    same = ledger.observe(event(
        "validation", 5200, accepted_artifacts=[artifact("artifacts/worldwide-legal-search/receipt.json", "v1")],
        pid=200, attempt=9,
    ))
    assert same["meaningful"] is False
    assert same["progress_token"] == changed["progress_token"]
    assert same["last_progress_at"] == 5100
    rejected = ledger.observe(event(
        "validation", 5300, accepted_artifacts=[artifact("artifacts/worldwide-legal-search/receipt.json", "v2", accepted=False)],
    ))
    assert rejected["meaningful"] is False
    assert rejected["progress_token"] == changed["progress_token"]

    pending = ledger.observe(event("merge", 5400, attempt=1, merge_accepted=False, merge_receipt=digest("not-yet")))
    assert pending["meaningful"] is False
    assert pending["reason"] == "unverified_merge"
    assert pending["cursor_stage"] == "validation"
    assert pending["observed_stage"] == "merge"
    assert pending["last_progress_at"] == 5100
    merge_incident = pending["incident_id"]
    ledger.charge(merge_incident, "repair")
    again = ledger.observe(event("promotion", 5500, kind="retry", attempt=2, pid=300, log="merge retry"))
    assert again["reason"] == "same_stage_retry"
    assert again["incident_id"] == merge_incident
    assert again["repair_budget"]["repair"]["remaining"] == 1
    assert again["progress_token"] == changed["progress_token"]
    verified = ledger.observe(event("merge", 5600, merge_accepted=True, merge_receipt=digest("merged-tree")))
    assert verified["reason"] == "verified_merge"
    assert verified["cursor_stage"] == "merge"
    assert verified["progress_advanced"] is True
    assert verified["last_progress_at"] == 5600
    assert verified["incident_id"] == merge_incident
    assert verified["repair_budget"]["repair"]["remaining"] == 1
    assert verified["budgets_reset"] is False
    assert ledger.incidents[incident]["spent"]["repair"] == 1


def test_incident_identity_ignores_pid_attempt_heartbeat_and_log_churn():
    left = incident_identity(
        run_id=RUN, task_id=TASK, stage="validation",
        defect="AssertionError pid=12 attempt=3 heartbeat_at=10 log_bytes=99 at 2026-10-30T05:14:23Z",
    )
    right = incident_identity(
        run_id=RUN, task_id=TASK, stage="validate",
        defect="AssertionError pid=99 attempt=8 heartbeat_at=77 log_bytes=5 at 2026-10-30T06:00:00+00:00",
    )
    assert left == right
    keyed = incident_identity(run_id=RUN, task_id=TASK, stage="validation", defect={
        "failure_key": "native_validation_failed", "message": "pid=1 attempt=2 " + ("log line\n" * 20),
        "pid": 44, "heartbeat_at": 9,
    })
    reordered = incident_identity(run_id=RUN, task_id=TASK, stage="testing", defect={
        "heartbeat_at": 90, "message": "completely different log text pid=8",
        "failure_key": "native_validation_failed",
    })
    assert keyed == reordered
    assert keyed != incident_identity(run_id=RUN, task_id=TASK, stage="validation", defect={"failure_key": "other_defect"})
    assert left != incident_identity(run_id=RUN, task_id=OTHER, stage="validation", defect="AssertionError")
    assert normalize_defect("disk full pid=3") == normalize_defect("disk full pid=9")
    assert normalize_defect({"reason": "disk full pid=3"}) == normalize_defect({"message": "disk full pid=9"})


def test_same_stage_regression_resumes_the_spent_budget():
    ledger = ProgressLedger(RUN)
    opened = ledger.observe(event("task", 10, attempt=1))
    incident = opened["incident_id"]
    ledger.charge(incident, "repair")
    ledger.charge(incident, "repair")
    blocked = ledger.charge(incident, "repair")
    assert blocked["status"] == "blocked"
    assert blocked["charged"] is False
    assert blocked["exhausted"] is True
    assert blocked["remaining"] == 0
    assert blocked["completion_authority"] is False
    advanced = ledger.observe(event("validation", 20))
    assert advanced["progress_advanced"] is True
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    returned = ledger.observe(event("task", 30, attempt=4, pid=999, log="back to implementation"))
    assert returned["meaningful"] is False
    assert returned["incident_id"] == incident
    assert returned["repair_budget"]["repair"]["remaining"] == 0
    assert returned["last_progress_at"] == 20
    assert returned["budgets_reset"] is False
    other = ledger.observe(event("task", 40, task_id=OTHER, defect={"failure_key": "native_validation_failed"}))
    assert other["reason"] == "canonical_task_transition"
    assert other["progress_advanced"] is True
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    home = ledger.observe(event("task", 50))
    assert home["incident_id"] == incident
    assert home["repair_budget"]["repair"]["remaining"] == 0
    assert home["reason"] == "canonical_task_transition"


def test_budgets_are_separate_and_retries_do_not_refund_them():
    ledger = ProgressLedger(RUN)
    opened = ledger.observe(event("validation", 5))
    incident = opened["incident_id"]
    assert ledger.charge(incident, "provider")["remaining"] == 3
    assert ledger.charge(incident, "task")["remaining"] == 2
    assert ledger.charge(incident, "recover")["remaining"] == 1
    repair = ledger.charge(incident, "repair")
    assert repair["remaining"] == 1
    token = opened["progress_token"]
    progress_at = opened["last_progress_at"]
    for attempt in range(3, 12):
        decision = ledger.observe(event("validation", 100 + attempt, kind="same_stage_retry", attempt=attempt, pid=attempt))
        assert decision["meaningful"] is False
        assert decision["progress_token"] == token
        assert decision["last_progress_at"] == progress_at
        assert decision["repair_budget"]["repair"]["remaining"] == 1
        assert decision["repair_budget"]["provider"]["remaining"] == 3
        assert decision["repair_budget"]["task"]["remaining"] == 2
        assert decision["repair_budget"]["recover"]["remaining"] == 1
        assert decision["repair_budget"]["validation"]["remaining"] == 3
        assert decision["repair_budget"]["merge"]["remaining"] == 3
    exhausted = ledger.charge(incident, "repair")
    assert exhausted["remaining"] == 0 and exhausted["charged"] is True
    assert ledger.charge(incident, "repair")["status"] == "blocked"
    assert ledger.incidents[incident]["spent"]["task"] == 1
    assert set(decision["repair_budget"]) == set(BUDGETS)


def test_checkpoint_preserves_cursor_and_budgets_across_restart():
    ledger = ProgressLedger(RUN)
    first = ledger.observe(event("task", 10, pid=15, attempt=1))
    ledger.charge(first["incident_id"], "merge")
    ledger.observe(event("validation", 20, accepted_artifacts=[artifact("artifacts/worldwide-legal-search/receipt.json", "kept")]))
    ledger.observe({"run_id": RUN, "kind": "heartbeat", "heartbeat_at": 25, "pid": 16})
    ledger.observe(event("validation", 30, kind="retry", attempt=2, pid=17))
    saved = ledger.to_dict()
    raw = json.dumps(saved, sort_keys=True)
    restored = ProgressLedger.from_dict(json.loads(raw))
    assert restored.to_dict() == saved
    assert restored.cursor["stage"] == "validation"
    assert restored.progress_token != first["progress_token"]
    assert restored.last_progress_at == 20
    follow = restored.observe(event("validation", 500, kind="retry", attempt=3, pid=90, log_bytes=5, heartbeat_at=500))
    assert follow["meaningful"] is False
    assert follow["progress_token"] == restored.progress_token
    assert follow["last_progress_at"] == 20
    assert follow["budgets_reset"] is False
    assert restored.incidents[first["incident_id"]]["spent"]["merge"] == 1
    tampered = json.loads(raw)
    tampered["progress_token"] = "sha256:" + ("ab" * 32)
    with pytest.raises(ProgressIdentityError, match="progress token"):
        ProgressLedger.from_dict(tampered)
    widened = json.loads(raw)
    widened["incidents"][0]["limits"]["repair"] = widened["incidents"][0]["limits"]["repair"] + 1
    with pytest.raises(ProgressIdentityError, match="budget"):
        ProgressLedger.from_dict(widened)


def test_checkpoint_file_round_trip_is_atomic_and_rejects_completion(tmp_path):
    ledger = ProgressLedger(RUN)
    ledger.observe(event("merge", 7, merge_accepted=True, accepted_tree=digest("tree")))
    path = tmp_path / "state" / "progress.json"
    ledger.save(path)
    loaded = ProgressLedger.load(path)
    assert loaded.to_dict() == ledger.to_dict()
    assert loaded.cursor["stage"] == "merge"
    assert loaded.completion_authority is False
    forged = ledger.to_dict()
    forged["completion_authority"] = True
    with pytest.raises(ProgressIdentityError, match="completion"):
        ProgressLedger.from_dict(forged)
    with pytest.raises(ProgressIdentityError, match="reset"):
        ledger.observe(event("merge", 8, reset_budget=True))
    with pytest.raises(ProgressIdentityError, match="completion"):
        ledger.observe(event("merge", 8, completion_authority=True))
    assert ledger.last_progress_at == 7
    with pytest.raises(ProgressIdentityError, match="logs and heartbeats"):
        ledger.observe(event("merge", 9, accepted_artifacts=[artifact("supervisor.log", "nope")]))
    with pytest.raises(ProgressIdentityError, match="run"):
        ledger.observe(event("merge", 9, run_id="other-run"))


def test_new_verified_merge_receipt_is_progress_and_replay_is_not():
    ledger = ProgressLedger(RUN)
    first = ledger.observe(event("merge", 1, merge_accepted=True, merge_receipt=digest("one")))
    second = ledger.observe(event("merge", 2, merge_accepted=True, merge_receipt=digest("one"), pid=2, attempt=2))
    assert second["meaningful"] is False
    assert second["progress_token"] == first["progress_token"]
    assert second["last_progress_at"] == 1
    third = ledger.observe(event("merged", 3, merge_accepted=True, merge_receipt=digest("two")))
    assert third["reason"] == "verified_merge"
    assert third["progress_token"] != first["progress_token"]
    assert third["incident_id"] == first["incident_id"]
    assert third["repair_budget"]["repair"]["spent"] == 0
