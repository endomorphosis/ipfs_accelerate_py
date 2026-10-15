"""Same-attempt rescue cannot acquire a new provider lease or native authority."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask
from test.api.test_agent_supervisor_implementation_auto_rescue import _inline_rescue_test_daemon
from test.api.test_provider_failure_checkpoint_retention import failed_child_case, run_failed_child


def _clock(monkeypatch, initial=100.0):
    value = [initial]
    proxy = SimpleNamespace(**{name: getattr(module.time, name) for name in dir(module.time)})
    proxy.monotonic = lambda: value[0]
    monkeypatch.setattr(module, "time", proxy)
    monkeypatch.setattr(supervisor_runtime, "time", proxy)
    return value


def _case(tmp_path, monkeypatch):
    daemon, events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    daemon._database_attempt_authority = None
    clock = _clock(monkeypatch)
    return daemon, events, clock


def _failure():
    return {"passed": False, "reason": "declared_validation_failed",
        "error": "validation_command_failed", "failed_commands": ["python check.py"],
        "failure_review": {"decision": "guide_rescue", "reason_codes": ["validation_command_failed"]}}


def _rescue(daemon, tmp_path, *, deadline, failure=None):
    return daemon._automatic_implementation_rescue(
        task=PortalTask(task_id="BUD-001", title="Repair result", status="in_progress",
            completion="validation passes", priority="high", track="test", outputs=["result.txt"]),
        attempt=1, workspace_path=tmp_path, branch_name="agent/bud-001", baseline_ref="a" * 40,
        validation_result=failure or _failure(), log_path=tmp_path / "rescue.log", state=None,
        command=[sys.executable, "-c", "from pathlib import Path; Path('result.txt').write_text('fixed')"],
        base_prompt="Repair the declared result and pass validation.", provider_deadline=deadline)


@pytest.mark.parametrize("deadline,reason", [
    (None, "missing"), (True, "invalid"), (False, "invalid"), ("120", "invalid"),
    (float("nan"), "invalid"), (float("inf"), "invalid"), (float("-inf"), "invalid"),
    (10 ** 500, "invalid"),
    (0, "exhausted"), (99.0, "exhausted"), (100.0, "exhausted"),
])
def test_absent_invalid_or_exhausted_budget_never_dispatches(tmp_path, monkeypatch, deadline, reason):
    daemon, events, _clock_value = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "run_process_group_stream", lambda *a, **k: pytest.fail("no provider budget"))
    result = _rescue(daemon, tmp_path, deadline=deadline)
    code = "provider_rescue_deadline_" + reason
    assert result["passed"] is False
    assert result["auto_rescue"]["provider_passes"] == 0
    assert result["auto_rescue"]["provider_rescue_blocked_reason"] == code
    assert any(name == "implementation_auto_rescue_provider_blocked" and row["reason"] == code
               for name, row in events)
    assert not (tmp_path / "result.txt").exists()


def test_real_rescue_child_uses_remaining_budget_after_environment_preparation(tmp_path, monkeypatch):
    daemon, events, clock = _case(tmp_path, monkeypatch)
    original = module.run_process_group_stream
    calls = []
    def environment(*args, **kwargs):
        clock[0] += 7.0
        return {}
    def launch(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(daemon, "_implementation_process_environment", environment)
    monkeypatch.setattr(module, "run_process_group_stream", launch)
    result = _rescue(daemon, tmp_path, deadline=130.0)
    assert result["passed"] is True
    assert result["auto_rescue"]["provider_passes"] == 1
    assert (tmp_path / "result.txt").read_text() == "fixed"
    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == 23.0
    assert calls[0]["max_timeout_seconds"] == 23.0
    assert calls[0]["absolute_deadline"] == 130.0
    starts = [row for name, row in events if name == "implementation_auto_rescue_provider_started"]
    assert len(starts) == 1 and starts[0]["provider_pid"] > 0


@pytest.mark.parametrize("expiry_stage", ["environment", "prepared_event"])
def test_expiry_during_rescue_preparation_cannot_launch_child(tmp_path, monkeypatch, expiry_stage):
    daemon, events, clock = _case(tmp_path, monkeypatch)
    if expiry_stage == "environment":
        def environment(*args, **kwargs):
            clock[0] = 131.0
            return {}
        monkeypatch.setattr(daemon, "_implementation_process_environment", environment)
    else:
        def record(name, payload):
            events.append((name, dict(payload)))
            if name == "implementation_auto_rescue_provider_prepared":
                clock[0] = 131.0
        monkeypatch.setattr(daemon, "_record_event", record)
    monkeypatch.setattr(module, "run_process_group_stream", lambda *a, **k: pytest.fail("deadline expired"))
    result = _rescue(daemon, tmp_path, deadline=130.0)
    assert result["passed"] is False
    assert result["auto_rescue"]["provider_passes"] == 0
    assert result["auto_rescue"]["provider_rescue_blocked_reason"] == "provider_rescue_deadline_exhausted"
    assert not any(name == "implementation_auto_rescue_provider_started" for name, _ in events)


def test_expiry_during_child_birth_persistence_cannot_become_success(tmp_path, monkeypatch):
    daemon, events, clock = _case(tmp_path, monkeypatch)
    def record(name, payload):
        events.append((name, dict(payload)))
        if name == "implementation_auto_rescue_provider_started":
            clock[0] = 131.0
    monkeypatch.setattr(daemon, "_record_event", record)
    monkeypatch.setattr(daemon, "_run_validation_with_candidate_binding",
        lambda *a, **k: pytest.fail("expired child cannot proceed to successful rescue validation"))
    result = _rescue(daemon, tmp_path, deadline=130.0)
    assert result["passed"] is False
    assert result["auto_rescue"]["provider_passes"] == 1
    assert result["auto_rescue"]["provider_rescue_blocked_reason"] == "provider_rescue_deadline_exhausted"
    assert len([name for name, _ in events if name == "implementation_auto_rescue_provider_started"]) == 1
    assert any(name == "implementation_auto_rescue_provider_failed" for name, _ in events)


@pytest.mark.parametrize("authority", [{}, {"database_attempt_id": "fixture"}])
def test_database_bound_rescue_requires_fresh_owner_dispatch(tmp_path, monkeypatch, authority):
    daemon, _events, _clock_value = _case(tmp_path, monkeypatch)
    daemon._database_attempt_authority = authority
    monkeypatch.setattr(module, "run_process_group_stream", lambda *a, **k: pytest.fail("no fresh owner dispatch"))
    result = _rescue(daemon, tmp_path, deadline=130.0)
    assert result["passed"] is False
    assert result["auto_rescue"]["provider_passes"] == 0
    assert result["auto_rescue"]["provider_rescue_blocked_reason"] == "database_provider_rescue_requires_owner_dispatch"


@pytest.mark.parametrize("deadline", [None, 99.0])
def test_deterministic_staging_remains_available_without_provider_budget(tmp_path, monkeypatch, deadline):
    daemon, events, _clock_value = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(daemon, "_dirty_in_scope_declared_output_paths", lambda *a, **k: ("result.txt",))
    monkeypatch.setattr(module, "run_process_group_stream", lambda *a, **k: pytest.fail("deterministic stage only"))
    failure = {"passed": False, "reason": "proposal_gate_failed", "error": "proposal_validation_failed",
        "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"],
        "failure_review": {"decision": "guide_rescue", "reason_codes": ["proposal_gate_failed", "empty_or_no_change"],
            "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"]}}
    result = _rescue(daemon, tmp_path, deadline=deadline, failure=failure)
    assert result["passed"] is True
    assert result["auto_rescue"]["stage_used"] is True
    assert result["auto_rescue"]["provider_passes"] == 0
    assert result["auto_rescue"]["provider_rescue_blocked_reason"] == (
        "provider_rescue_deadline_missing" if deadline is None else "provider_rescue_deadline_exhausted")
    assert any(name == "implementation_auto_rescue_stage_and_revalidate" for name, _ in events)


@pytest.mark.parametrize("validation_elapsed", [3.0, 21.0])
def test_primary_dispatch_starts_single_deadline_before_child_and_validation(tmp_path, monkeypatch, validation_elapsed):
    daemon, task, state, _baseline = failed_child_case(tmp_path)
    (tmp_path / "failed_child.py").write_text("from pathlib import Path\nPath('src/value.py').write_text('VALUE = 2\\n')\n")
    task.metadata["implementation max timeout seconds"] = "20"
    task.validation[:] = ["python -c 'raise SystemExit(1)'"]
    clock = _clock(monkeypatch)
    original_mutation = daemon._decision_runtime_mutation
    original_stream = module.run_process_group_stream
    original_review = daemon._apply_implementation_failure_review
    original_rescue = daemon._automatic_implementation_rescue
    calls, rescue = [], []
    def mutation(kind, payload, action):
        if payload.get("operation") == "implementation_provider":
            clock[0] = 200.0  # setup/indexing is outside the provider lease
        return original_mutation(kind, payload, action)
    def launch(*args, **kwargs):
        calls.append({"started": clock[0], **kwargs})
        result = original_stream(*args, **kwargs)
        clock[0] += 2.0
        return result
    def review(**kwargs):
        clock[0] += validation_elapsed
        return original_review(**kwargs)
    def capture(**kwargs):
        rescue.append({"deadline": kwargs["provider_deadline"], "observed": clock[0]})
        if validation_elapsed > 20:
            return original_rescue(**kwargs)
        return dict(kwargs["validation_result"])
    monkeypatch.setattr(daemon, "_decision_runtime_mutation", mutation)
    monkeypatch.setattr(module, "run_process_group_stream", launch)
    monkeypatch.setattr(daemon, "_apply_implementation_failure_review", review)
    monkeypatch.setattr(daemon, "_automatic_implementation_rescue", capture)
    result = run_failed_child(daemon, task, state)
    assert len(calls) == 1
    assert calls[0]["started"] == 200.0
    assert calls[0]["timeout_seconds"] == 20.0
    assert calls[0]["absolute_deadline"] == 220.0
    assert rescue and rescue[0]["deadline"] == 220.0
    assert rescue[0]["observed"] == 202.0 + validation_elapsed
    assert result["validation_result"]["passed"] is False
    assert result["board_completion"]["complete"] is False
    if validation_elapsed > 20:
        assert result["validation_result"]["auto_rescue"]["provider_rescue_blocked_reason"] == "provider_rescue_deadline_exhausted"
