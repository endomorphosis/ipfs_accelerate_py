"""Authored finalization boundaries; no provider, container or worker execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from test.api.test_semantic_typed_nomination_usage import _case
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence
from benchmarks.agent_supervisor.container_coding import terminal_source_view_observation as source


FIELDS = ("semantic_decision_dialogues", "semantic_decision_dialogue_expected",
    "semantic_decision_owner_bindings", "semantic_decision_owner_binding_expected",
    "semantic_decision_expected_task_id", "semantic_decision_nomination_sha256",
    "semantic_decision_typed_nomination_expected")


def decision_report():
    return {"arm": "full", "task_completed": True,
        "provider_invocations": [{"phase": "coding", "invocation_id": "authored-native"}],
        "provider_usage_accounting": {"retained_native_total": 120},
        "semantic_decision_dialogues": [{"session_id": "authored-dialogue"}],
        "semantic_decision_dialogue_expected": True,
        "semantic_decision_owner_bindings": [{"binding_sha256": "a" * 64}],
        "semantic_decision_owner_binding_expected": True,
        "semantic_decision_expected_task_id": "authored-task",
        "semantic_decision_nomination_sha256": "b" * 64,
        "semantic_decision_typed_nomination_expected": False,
        "unrelated_private_field": "PRIVATE-UNRELATED-FIELD"}


@pytest.fixture
def boundaries(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    events = []
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: clock.now))
    def capture(**kwargs):
        events.append(("capture", kwargs))
        return {"schema": "terminal-context-evidence-export-receipt@1", "status": "unavailable",
            "reason": "authored_no_export", "provider_calls": 0,
            "execution_authority": False, "completion_authority": False}
    def observe(**kwargs):
        events.append(("audit", kwargs))
        return audit._base()
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", capture)
    monkeypatch.setattr(audit, "collect_terminal_context_audit", observe)
    return clock, events


@pytest.mark.parametrize("deadline,spent,capture_budget,audit_budget", [
    (100.0, 4.0, 15.0, 30.0), (20.0, 8.0, 15.0, 11.0),
    (5.0, 5.0, 4.0, 0.0), (0.0, 0.0, 0.0, 0.0),
])
def test_closed_fields_capture_first_and_audit_recomputes_remaining_budget(
    boundaries, monkeypatch, deadline, spent, capture_budget, audit_budget,
):
    clock, events = boundaries
    original_capture = evidence.collect_terminal_context_evidence
    def capture(**kwargs):
        result = original_capture(**kwargs)
        clock.now += spent
        return result
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", capture)
    report = decision_report()
    before = deepcopy(report)
    state = Path("/authored/state")
    driver._final_context_audit(report, state=state, deadline=deadline)
    assert [name for name, _ in events] == ["capture", "audit"]
    for _, kwargs in events:
        assert kwargs["decision_report"] == {name: before[name] for name in FIELDS}
        assert set(kwargs["decision_report"]) == set(FIELDS)
        assert kwargs["state"] == state and kwargs["workspace_root"] == driver.WORKTREES
        assert kwargs["receipts"] == before["provider_invocations"]
    assert events[0][1]["timeout_seconds"] == capture_budget
    assert events[1][1]["timeout_seconds"] == audit_budget
    assert {k: v for k, v in report.items() if k not in {"context_replay_evidence", "context_input_audit"}} == before


@pytest.mark.parametrize("arm", ["no-index", None])
def test_non_full_arm_does_not_call_any_final_context_boundary(boundaries, arm):
    _, events = boundaries
    report = decision_report()
    report["arm"] = arm
    before = deepcopy(report)
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=50)
    assert events == [] and report == before


def test_provider_free_doctor_bypass_precedes_expected_decision_capture(boundaries, monkeypatch):
    _, events = boundaries
    report = decision_report()
    monkeypatch.setattr(source, "provider_free_coding_application_observed", lambda value: value is report)
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=50)
    assert events == [] and "context_replay_evidence" not in report
    assert report["context_input_audit"]["status"] == "not_applicable"
    assert report["context_input_audit"]["reason"] == "provider_free_doctor_materialization_observed"
    assert report["task_completed"] is True


def test_plain_world_receipt_keeps_existing_receipt_only_audit(boundaries):
    _, events = boundaries
    report = {"arm": "full", "provider_invocations": [{"phase": "coding", "world_task_packet": {}}]}
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=9)
    assert [name for name, _ in events] == ["audit"]
    assert "decision_report" not in events[0][1] and events[0][1]["timeout_seconds"] == 8
    assert "context_replay_evidence" not in report


def test_observed_owner_binding_alone_requests_bounded_capture(boundaries):
    _, events = boundaries
    report = {"arm": "full", "provider_invocations": [],
        "semantic_decision_owner_bindings": [{"binding_sha256": "a" * 64}]}
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=9)
    assert [name for name, _ in events] == ["capture", "audit"]


@pytest.mark.parametrize("error,kind", [
    (ValueError("PRIVATE-ERROR-MESSAGE"), "ValueError"),
    (type("PRIVATE-ERROR-TYPE", (Exception,), {})("PRIVATE-ERROR-MESSAGE"), "other"),
])
def test_capture_refusal_metadata_stays_closed_and_still_audits(boundaries, monkeypatch, error, kind):
    _, events = boundaries
    def refused(**kwargs):
        raise error
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", refused)
    report = decision_report()
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=30)
    assert [name for name, _ in events] == ["audit"]
    receipt = report["context_replay_evidence"]
    assert receipt["status"] == "unavailable" and receipt["error_type"] == kind
    assert "PRIVATE-" not in json.dumps(receipt)
    assert report["task_completed"] is True and report["provider_usage_accounting"] == {"retained_native_total": 120}


def test_audit_failure_keeps_capture_receipt_and_native_usage(boundaries, monkeypatch):
    def refused(**kwargs):
        raise RuntimeError("PRIVATE-AUDIT-ERROR")
    monkeypatch.setattr(audit, "collect_terminal_context_audit", refused)
    report = decision_report()
    driver._final_context_audit(report, state=Path("/authored/state"), deadline=30)
    assert report["context_replay_evidence"]["reason"] == "authored_no_export"
    assert report["context_input_audit"]["reason"] == "audit_finalization_unavailable"
    assert "PRIVATE-" not in json.dumps(report["context_input_audit"])
    assert report["task_completed"] is True and report["provider_usage_accounting"] == {"retained_native_total": 120}


@pytest.mark.parametrize("stage", ["capture", "audit"])
@pytest.mark.parametrize("error_class", [KeyboardInterrupt, SystemExit, driver._BenchmarkPhaseExpired])
def test_interruption_records_no_private_error_and_preserves_original_identity(
    boundaries, monkeypatch, stage, error_class,
):
    _, events = boundaries
    interruption = error_class("PRIVATE-CANCELLATION-CONTENT")
    def interrupted(**kwargs):
        raise interruption
    target = evidence if stage == "capture" else audit
    name = "collect_terminal_context_evidence" if stage == "capture" else "collect_terminal_context_audit"
    monkeypatch.setattr(target, name, interrupted)
    report = decision_report()
    with pytest.raises(error_class) as caught:
        driver._final_context_audit(report, state=Path("/authored/state"), deadline=30)
    assert caught.value is interruption
    assert [name for name, _ in events] == ([] if stage == "capture" else ["capture"])
    metadata = {key: report[key] for key in ("context_replay_evidence", "context_input_audit") if key in report}
    assert "PRIVATE-" not in json.dumps(metadata)
    assert report["task_completed"] is True and report["provider_usage_accounting"] == {"retained_native_total": 120}


@pytest.mark.parametrize("dialogue_first", [False, True])
def test_log_ingress_joins_nested_and_printed_native_rows_in_both_orders(dialogue_first):
    from test.api.test_terminal_decision_usage import _native, _attempt, _dialogue
    first, second = _native(), _native(2, inputs=50, outputs=10)
    dialogue = _dialogue(_attempt(first), _attempt(second, number=2))
    observations = [dialogue, first, second] if dialogue_first else [first, second, dialogue]
    report = {"provider_invocations": []}
    driver._collect_provider_usage_rows(report, observations + [deepcopy(dialogue), deepcopy(first)])
    assert len(report["semantic_decision_dialogues"]) == 1 and len(report["provider_invocations"]) == 2
    driver._final_usage_accounting(report)
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] == 180


@pytest.mark.parametrize("dialogue_first", [False, True])
def test_log_ingress_keeps_printed_native_conflict_before_or_after_dialogue(dialogue_first):
    from test.api.test_terminal_decision_usage import _native, _attempt, _dialogue
    first, changed = _native(), _native(inputs=101)
    dialogue = _dialogue(_attempt(first))
    observations = [dialogue, first, changed] if dialogue_first else [first, changed, dialogue]
    report = {"provider_invocations": []}
    driver._collect_provider_usage_rows(report, observations)
    assert len(report["provider_invocations"]) == 2
    driver._final_usage_accounting(report)
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] is None


def test_owner_binding_log_row_selects_nested_native_conflict_preservation():
    from test.api.test_terminal_decision_usage import _native
    owner = {"schema": "supervisor-semantic-decision-owner-binding@1", "binding_sha256": "a" * 64}
    report = {"provider_invocations": []}
    driver._collect_provider_usage_rows(report, [_native(), _native(inputs=101), owner, deepcopy(owner)])
    assert report["semantic_decision_owner_bindings"] == [owner]
    assert len(report["provider_invocations"]) == 2


def test_unselected_native_grok_ingress_keeps_previous_receipt_shape():
    original = {"schema": "router-implementation-invocation@1", "provider": "grok_cli",
        "invocation_id": "authored-grok", "native_rollout_usage": {"usage": {"cache_write_input_tokens": 7}}}
    report = {"provider_invocations": []}
    driver._collect_provider_usage_rows(report, [original, deepcopy(original), {"schema": "unrelated@1"}])
    assert report == {"provider_invocations": [{"phase": "coding", **original}]}


def _pending_typed_report():
    report = _case()
    trusted = report.pop("context_input_audit")
    report.update(arm="full", task_completed=True)
    return report, trusted


def _controlled_observers(monkeypatch, observe):
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setattr(source, "provider_free_coding_application_observed", lambda report: False)
    monkeypatch.setattr(audit, "collect_terminal_context_audit", observe)

    def unavailable_capture(**kwargs):
        # Typed-lineage export is an independently unsupported transport
        # family; the trusted in-container audit may still verify its inputs.
        return {"schema": "terminal-context-evidence-export-receipt@1",
            "status": "unavailable", "reason": "typed_lineage_export_not_supported",
            "provider_calls": 0, "execution_authority": False, "completion_authority": False}

    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", unavailable_capture)


def test_real_typed_accounting_refreshes_after_trusted_audit_and_matches_harbor(monkeypatch):
    report, trusted = _pending_typed_report()
    retained = deepcopy(report)
    before_audit = []

    def observe(**kwargs):
        pending = deepcopy(report["provider_usage_accounting"])
        before_audit.append(pending)
        assert pending["reconciliation"]["known_provider_token_subtotal"] == 120
        assert pending["report_provider_total_tokens"] is None
        assert pending["semantic_decision_coverage"]["authoritative_usage_coverage_complete"] is False
        assert "typed_nomination_verification_population_unavailable" in {
            issue["reason"] for issue in pending["semantic_decision_coverage"]["issues"]}
        return deepcopy(trusted)

    _controlled_observers(monkeypatch, observe)
    driver._final_context_and_usage(report, state=Path("/authored/state"), deadline=60.0)
    assert len(before_audit) == 1
    stored = report["provider_usage_accounting"]
    projected = harbor.measured_usage(report)
    assert report["context_input_audit"] == trusted
    assert stored["semantic_decision_coverage"]["authoritative_usage_coverage_complete"] is True
    assert stored["semantic_decision_coverage"]["typed_nomination_verified_owner_bindings_observed"] == 1
    assert stored["report_provider_total_tokens"] == projected["total_tokens"] == 120
    assert stored["reconciliation"]["known_counter_subtotals"]["cached_input_tokens"] == 50
    assert projected["cached_input_tokens"] == 50 and projected["provider_calls"] == 1
    assert {key: value for key, value in report.items() if key not in {
        "provider_usage_accounting", "context_replay_evidence", "context_input_audit"}} == retained


@pytest.mark.parametrize("error_class", [KeyboardInterrupt, SystemExit, driver._BenchmarkPhaseExpired])
def test_audit_interruption_preserves_original_identity_and_pre_audit_charge(monkeypatch, error_class):
    report, _trusted = _pending_typed_report()
    retained = deepcopy(report)
    interruption = error_class("PRIVATE-INTERRUPTION-CONTENT")
    before_audit = []

    def observe(**kwargs):
        before_audit.append(deepcopy(report["provider_usage_accounting"]))
        raise interruption

    _controlled_observers(monkeypatch, observe)
    with pytest.raises(error_class) as captured:
        driver._final_context_and_usage(report, state=Path("/authored/state"), deadline=60.0)
    assert captured.value is interruption
    assert len(before_audit) == 1 and report["provider_usage_accounting"] == before_audit[0]
    assert report["provider_usage_accounting"]["reconciliation"]["known_provider_token_subtotal"] == 120
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] is None
    assert report["context_input_audit"]["reason"] == "audit_finalization_unavailable"
    assert {key: value for key, value in report.items() if key not in {
        "provider_usage_accounting", "context_replay_evidence", "context_input_audit"}} == retained


def test_plain_unselected_finalization_keeps_receipt_only_audit_without_usage_projection(monkeypatch):
    report = {"arm": "full", "task_completed": True, "provider_invocations": [],
        "provider_usage_accounting": {"legacy_retained_observation": 17}}
    retained = deepcopy(report)
    observations = []
    trusted = audit._base()

    def observe(**kwargs):
        observations.append(kwargs)
        return deepcopy(trusted)

    _controlled_observers(monkeypatch, observe)
    monkeypatch.setattr(driver, "_final_usage_accounting", lambda report:
                        pytest.fail("unselected finalization called decision accounting"))
    monkeypatch.setattr(evidence, "collect_terminal_context_evidence", lambda **kwargs:
                        pytest.fail("unselected finalization requested decision evidence"))
    driver._final_context_and_usage(report, state=Path("/authored/state"), deadline=60.0)
    assert len(observations) == 1 and "decision_report" not in observations[0]
    assert report["context_input_audit"] == trusted
    assert {key: value for key, value in report.items() if key != "context_input_audit"} == retained
