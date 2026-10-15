"""Replay finalization keeps task deadlines and the older profile budgets.

Only the collectors are authored boundary doubles. Profile selection and caller
deadline handling are real; these observations are not evidence of native replay.
"""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as terminal
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence


def _report(profile):
    report = {"arm": "full", "semantic_decision_dialogue_expected": True,
              "semantic_decision_dialogues": [], "provider_invocations": []}
    if profile != "base":
        report.update(semantic_decision_typed_nomination_expected=True,
            semantic_decision_owner_binding_expected=True,
            semantic_decision_nomination_sha256="a" * 64,
            semantic_decision_expected_task_id="authored-task",
            semantic_decision_owner_bindings=[{"typed_nomination": {}}])
    if profile == "typed-retrieval":
        report["semantic_decision_dialogues"] = [{"worker_launch": {"retrieval_query_binding": {
            "artifact": ".runtime/retrieval.json", "expected_sha256": "b" * 64,
            "task_id": "authored-task"}}}]
    return report


@pytest.mark.parametrize("profile,cap", [("base", 15), ("typed", 15), ("typed-retrieval", 30)])
@pytest.mark.parametrize("remaining", [65, 12.5, .5])
def test_export_allowance_is_profile_bound_and_never_extends_remaining_task_deadline(tmp_path, monkeypatch, profile, cap, remaining):
    calls = []
    report = _report(profile)
    original = deepcopy(report)
    monkeypatch.setattr(terminal.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", lambda **kwargs:
        calls.append(("export", kwargs)) or {"status": "authored-export"})
    monkeypatch.setattr(audit, "collect_terminal_context_audit", lambda **kwargs:
        calls.append(("audit", kwargs)) or {"status": "authored-audit"})
    terminal._final_context_audit(report, state=tmp_path, deadline=100.0 + remaining)
    assert [kind for kind, _ in calls] == ["export", "audit"]
    assert calls[0][1]["timeout_seconds"] == max(0, min(cap, remaining - 1))
    assert calls[1][1]["timeout_seconds"] == max(0, min(audit.MAX_AUDIT_SECONDS, remaining - 1))
    assert {key: report[key] for key in original} == original
    assert report["context_replay_evidence"] == {"status": "authored-export"}
    assert report["context_input_audit"] == {"status": "authored-audit"}


def test_malformed_retrieval_profile_refuses_export_and_retains_independent_audit(tmp_path, monkeypatch):
    report = _report("typed-retrieval")
    report["semantic_decision_dialogues"][0]["worker_launch"]["retrieval_query_binding"]["task_id"] = "foreign-task"
    monkeypatch.setattr(terminal.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", lambda **kwargs:
        pytest.fail("malformed profile reached export"))
    monkeypatch.setattr(audit, "collect_terminal_context_audit", lambda **kwargs: {"status": "authored-audit"})
    terminal._final_context_audit(report, state=tmp_path, deadline=165.0)
    assert report["context_replay_evidence"]["status"] == "unavailable"
    assert report["context_replay_evidence"]["error_type"] == "ValueError"
    assert report["context_input_audit"] == {"status": "authored-audit"}
