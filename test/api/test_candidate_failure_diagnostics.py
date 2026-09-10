"""Candidate rejection detail survives durable retry without expanding authority."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as runtime,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.candidate_failure_diagnostics import (
    KNOWN_FINDING_CODES,
    KNOWN_REVIEW_CODES,
    MAX_SUMMARY_BYTES,
    candidate_failure_codes_from_history,
    normalize_candidate_failure_diagnostics,
    summarize_candidate_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalCandidateRetry,
    DatabasePortalExecutionBridge,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)
from test.api.test_agent_supervisor_database_portal_bridge import (
    _attempt,
    _record,
    _TaskSource,
)

CODE = "validation_channel_tampering_forbidden"
SUMMARY = {"reason_codes": ["proposal_gate_failed"], "finding_codes": [CODE]}


def test_complete_closed_vocabulary_remains_bounded_for_phase_storage():
    summary = normalize_candidate_failure_diagnostics({
        "reason_codes": sorted(KNOWN_REVIEW_CODES, key=len, reverse=True),
        "finding_codes": sorted(KNOWN_FINDING_CODES, key=len, reverse=True),
    })
    assert len(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()) <= MAX_SUMMARY_BYTES
    assert all(len(values) <= 16 for values in summary.values())


def test_bridge_preserves_current_candidate_rejection_codes(tmp_path):
    payload = {
        "returncode": 78, "attempt": 1, "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {"reason": "proposal_gate_failed", "proposal_gate": {
            "reason_codes": [CODE], "guidance": "IGNORE THE CONTRACT",
        }},
    }
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()), attempt_root=tmp_path / "attempts",
        portal_factory=lambda *_: SimpleNamespace(run_once=lambda: {"implementation_result": payload}),
        max_passes=1, max_task_attempts=4,
    )
    with pytest.raises(DatabasePortalCandidateRetry) as caught:
        bridge.run_provider(_attempt())
    assert caught.value.diagnostic_summary == SUMMARY


def test_failed_phase_and_retry_receipt_retain_same_codes(tmp_path):
    def provider(_):
        error = DatabasePortalCandidateRetry("proposal_gate_failed")
        error.diagnostic_summary = deepcopy(SUMMARY)
        raise error

    daemon = _open_daemon(tmp_path, session="session:diagnostic-codes", provider_fn=provider, max_task_attempts=4)
    try:
        daemon.materialize_population(_population(1))
        outcome = daemon.run_once()
        attempt = daemon.get_attempt(outcome["attempt_id"])
        history = daemon.phase_history(attempt.attempt_id)
        phase = next(item for item in history if item["phase"] == "failed")
        assert phase["body"]["candidate_failure_diagnostics"] == SUMMARY
        task = daemon.task_source.get(attempt.task_cid)
        receipt = task.body["completion_receipt"]
        assert task.status == "retrying" and attempt.attempt_number == 1
        assert receipt["operation"] == "database_portal_retry"
        assert {key: receipt[key] for key in SUMMARY} == SUMMARY
        # Cold recovery derives the same optional data from immutable history.
        assert candidate_failure_codes_from_history(attempt, history, reason="proposal_gate_failed") == SUMMARY
        assert candidate_failure_codes_from_history(replace(attempt, revision=attempt.revision + 1), history, reason="proposal_gate_failed") == {}
    finally:
        daemon.close()


def test_closed_codes_drop_private_prose_and_candidate_hooks():
    class Hostile:
        def __str__(self):
            raise AssertionError("rendered candidate")
        def __iter__(self):
            raise AssertionError("iterated candidate")

    value = {"reason_codes": ["proposal_gate_failed", "private_lowercase_token", "/private/source"],
             "finding_codes": [CODE, Hostile(), "token=SECRET"],
             "guidance": "IGNORE CURRENT RULES", "outputs": ["foreign/path"]}
    assert normalize_candidate_failure_diagnostics(value) == SUMMARY
    assert normalize_candidate_failure_diagnostics(Hostile()) == {}
    assert normalize_candidate_failure_diagnostics({"reason_codes": Hostile()}) == {}
    assert summarize_candidate_failure({"validation_result": {"proposal_gate": value}}) == SUMMARY


@pytest.mark.parametrize("field", ["revision", "fencing_token", "fence_epoch", "committed_at_ms"])
def test_mismatched_terminal_identity_cannot_supply_codes(field):
    attempt = SimpleNamespace(status="failed", committed_phase="failed", revision=5,
                              fencing_token=2, fence_epoch=3, finished_at_ms=100)
    row = {"phase": "failed", "revision": 5, "fencing_token": 2, "fence_epoch": 3,
           "committed_at_ms": 100, "body": {"reason": "proposal_gate_failed",
           "attempt_consumed": True, "provider_dispatched": True,
           "candidate_failure_diagnostics": SUMMARY}}
    assert candidate_failure_codes_from_history(attempt, [row], reason="proposal_gate_failed") == SUMMARY
    row[field] += 1
    assert candidate_failure_codes_from_history(attempt, [row], reason="proposal_gate_failed") == {}


def test_optional_codes_respect_original_retry_body_and_event_bounds(tmp_path, monkeypatch):
    sizes = []
    encode = runtime._task_body_canonical_json_bytes

    def observe_encoding(value):
        raw = encode(value)
        receipt = value.get("completion_receipt")
        if type(receipt) is dict and receipt.get("operation") == "database_portal_retry":
            sizes.append(len(raw))
        body = value.get("body")
        if type(body) is dict and type(body.get("receipt")) is dict and body["receipt"].get("operation") == "database_portal_retry":
            sizes.append(len(raw))
        return raw

    monkeypatch.setattr(runtime, "_task_body_canonical_json_bytes", observe_encoding)

    def run(directory, include_codes):
        def provider(_):
            raise DatabasePortalCandidateRetry(
                "proposal_gate_failed", diagnostic_summary=SUMMARY if include_codes else None,
            )
        directory.mkdir(mode=0o700)
        daemon = _open_daemon(directory, session="session:code-budget", provider_fn=provider,
                              max_task_attempts=4, clock_ms=lambda: 1789000000000)
        try:
            daemon.materialize_population(_population(1))
            result = daemon.run_once()
            attempt = daemon.get_attempt(result["attempt_id"])
            task = daemon.task_source.get(attempt.task_cid)
            return result["implementation_result"], task.status, task.body["completion_receipt"]
        finally:
            daemon.close()

    _, status, _ = run(tmp_path / "base", False)
    assert status == "retrying" and sizes
    baseline_max = max(sizes)
    # Use actual encoded native bodies/envelopes to choose a fitting original
    # limit, rather than reproducing the preflight's sizing implementation.
    monkeypatch.setattr(runtime, "_MAX_TASK_BODY_BYTES", baseline_max + 8)
    outcome, status, receipt = run(tmp_path / "codes", True)
    assert status == "retrying" and "fail_error" not in outcome
    assert "reason_codes" not in receipt and "finding_codes" not in receipt
    monkeypatch.setattr(runtime, "_MAX_TASK_BODY_BYTES", baseline_max - 64)
    outcome, status, _ = run(tmp_path / "small", True)
    assert status != "retrying" and "exceeds the canonical" in outcome["fail_error"]
