"""Staging cannot fix an already admitted, unchanged runtime assertion."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    plan_automatic_implementation_rescue,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def failed_candidate(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Rescue fixture")
    _git(repo, "config", "user.email", "rescue@example.invalid")
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("VALUE = 2\n")
    (repo / "test_runtime.py").write_text(
        "from app import VALUE\n\ndef test_runtime_value():\n    assert VALUE == 3\n"
    )
    state = tmp_path / "state"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md", state_path=state / "state.json",
        strategy_path=state / "strategy.json", events_path=state / "events.jsonl",
        repo_root=repo, worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="RESCUE-ASSERT-001", title="Fix the runtime value", status="todo",
        completion="The runtime assertion passes", priority="P0", track="test",
        outputs=["app.py", "test_runtime.py"],
        validation=["python -m pytest -q test_runtime.py"],
    )
    log = tmp_path / "validation.log"
    proposal = daemon._validate_implementation_patch(repo, task, baseline_ref=baseline)
    assert proposal.accepted is True
    result = daemon._run_validation_with_candidate_binding(
        repo, task, log, baseline_ref=baseline, proposal_validation=proposal,
    )
    result = daemon._apply_implementation_failure_review(
        task=task, attempt=1, workspace_path=repo, validation_result=result,
        log_path=log, proposal_validation=proposal, baseline_ref=baseline,
    )
    assert result["passed"] is False, result
    assert result["proposal_validation"] is proposal
    assert result["proposal_gate"]["accepted"] is True
    assert result["failed_tests"] == ["test_runtime.py::test_runtime_value"], result
    assert result["failure_review"]["finding_codes"] == []
    assert result["failure_review"]["missing_expected_outputs"] == []
    (tmp_path / "failure-diagnostic.json").write_text(json.dumps({
        "gate": result["proposal_gate"], "rows": result.get("results"),
        "exceptions": result.get("exception_types"),
        "live_receipt_id": proposal.receipt.receipt_id,
        "live_task": proposal.proposal.task_id,
        "live_tree": proposal.proposal.repository_tree_id,
    }, indent=2))
    return daemon, task, repo, baseline, log, result


@pytest.mark.parametrize("record_shape", ["native", "host_assertion_adapter"])
def test_real_native_failed_assertion_does_not_revalidate_unchanged_staged_bytes(
    failed_candidate, monkeypatch: pytest.MonkeyPatch, record_shape: str,
) -> None:
    daemon, task, repo, baseline, log, result = failed_candidate
    before = {path: (repo / path).read_bytes() for path in task.outputs}
    before_index = _git(repo, "diff", "--cached", "--name-only")
    if record_shape == "host_assertion_adapter":
        # Project the actual failed pytest exit as the Host callback does.
        # The live admission, executed assertion and current bytes are real.
        row = result["results"][0]
        assert row["returncode"] == 1
        row.update(error="failed_assertion", infrastructure_failure=False,
            native_executor_failure={"returncode": row["returncode"]})
    assert before_index == ""
    assert "test_runtime.py" in daemon._dirty_in_scope_declared_output_paths(repo, task)
    calls = []
    validate = daemon._run_validation_with_candidate_binding

    def revalidate(*args, **kwargs):
        calls.append(True)
        return validate(*args, **kwargs)

    monkeypatch.setattr(daemon, "_run_validation_with_candidate_binding", revalidate)
    deadline = time.monotonic() + 30
    rescued = daemon._automatic_implementation_rescue(
        task=task, attempt=1, workspace_path=repo, branch_name="fixture",
        baseline_ref=baseline, validation_result=result, log_path=log, state=None,
        allow_provider_rescue=False, provider_deadline=deadline,
    )
    assert calls == [], "Git staging must not repeat an unchanged failed runtime test"
    assert rescued["passed"] is False
    assert rescued["auto_rescue"]["stage_used"] is False
    assert rescued["auto_rescue"]["provider_passes"] == 0
    assert rescued["failed_tests"] == result["failed_tests"]
    assert _git(repo, "diff", "--cached", "--name-only") == before_index
    assert {path: (repo / path).read_bytes() for path in task.outputs} == before
    assert deadline > time.monotonic()


@pytest.mark.parametrize("invalid", [
    "not_native", "gate_id", "gate_paths", "wrong_task", "wrong_baseline",
    "stale_source", "missing_output", "unknown_tests", "unknown_row", "cached",
    "preflight", "source_changed", "malformed_entry", "deleted_entry", "unsealed",
])
def test_unknown_or_stale_failure_cannot_skip_staging(failed_candidate, invalid: str) -> None:
    daemon, task, repo, baseline, _log, original = failed_candidate
    result = copy.deepcopy(original)
    if invalid == "not_native":
        result["proposal_validation"] = SimpleNamespace(accepted=True)
    elif invalid == "gate_id":
        result["proposal_gate"]["proposal_id"] = "unrelated"
    elif invalid == "gate_paths":
        result["proposal_gate"]["changed_paths"] = [None]
    elif invalid == "wrong_task":
        task = dataclasses.replace(task, task_id="OTHER-001")
    elif invalid == "wrong_baseline":
        baseline = "b" * 40
    elif invalid == "stale_source":
        (repo / "app.py").write_text("VALUE = 4\n")
    elif invalid == "missing_output":
        (repo / "test_runtime.py").unlink()
    elif invalid == "unknown_tests":
        result["failed_tests"] = ["collection error"]
    elif invalid == "unknown_row":
        result["results"][0].pop("timed_out", None)
    elif invalid == "cached":
        result["results"][0]["cache_hit"] = True
    elif invalid == "preflight":
        result["exception_types"] = []
        result["results"][0]["error"] = "python_global_preflight_failed"
    elif invalid == "source_changed":
        result["source_unchanged"] = False
    elif invalid == "unsealed":
        result["results"][0]["validation_python_launcher"]["sealed"] = False
    elif invalid in ("malformed_entry", "deleted_entry"):
        entry = result["proposal_validation"].proposal.candidate_diff[0]
        object.__setattr__(entry, "new_path", None)
        if invalid == "deleted_entry":
            object.__setattr__(entry, "after_source", None)
    assert daemon._current_admitted_test_failure(
        result, task=task, workspace_path=repo, baseline_ref=baseline,
    ) is False


def test_explicit_unstaged_preflight_still_stages_and_revalidates(
    failed_candidate, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, repo, baseline, log, _result = failed_candidate
    # This command requires a genuinely tracked new output. Staging is a real
    # repair here; an admitted runtime failure with no such finding is different.
    (repo / "app.py").write_text("VALUE = 3\n")
    (repo / "test_runtime.py").write_text(
        "import subprocess\nfrom app import VALUE\n\ndef test_runtime_value():\n"
        "    assert VALUE == 3\n"
        "    assert subprocess.check_output(['git', 'ls-files', '--', 'test_runtime.py'], text=True).strip() == 'test_runtime.py'\n"
    )
    assert _git(repo, "ls-files", "--", "test_runtime.py") == ""
    validation = daemon._run_validation_with_candidate_binding
    calls = []

    def revalidate(*args, **kwargs):
        calls.append(True)
        return validation(*args, **kwargs)

    monkeypatch.setattr(daemon, "_run_validation_with_candidate_binding", revalidate)
    rescued = daemon._automatic_implementation_rescue(
        task=task, attempt=1, workspace_path=repo, branch_name="fixture",
        baseline_ref=baseline, log_path=log, state=None, allow_provider_rescue=False,
        validation_result={
            "passed": False, "attempted": False, "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "finding_codes": ["expected_output_ignored_or_unstaged"],
            "failure_review": {"decision": "guide_rescue",
                "reason_codes": ["proposal_gate_failed"],
                "finding_codes": ["expected_output_ignored_or_unstaged"]},
        },
    )
    assert calls == [True]
    assert rescued["passed"] is True, rescued
    assert rescued["auto_rescue"]["stage_used"] is True
    assert _git(repo, "ls-files", "--", "test_runtime.py") == "test_runtime.py"


@pytest.mark.parametrize("provider_allowed", [False, True])
def test_only_owner_verified_failure_skips_dirty_only_stage(provider_allowed: bool) -> None:
    failure = {
        "passed": False, "attempted": True, "error": "validation_command_failed",
        "failed_tests": ["test_runtime.py::test_runtime_value"],
        "proposal_gate": {"accepted": True},
        "failure_review": {"decision": "guide_rescue",
            "reason_codes": ["validation_command_failed"]},
    }
    common = dict(validation_result=failure, expected_outputs=["app.py"],
        dirty_in_scope_paths=["app.py"], expected_outputs_present_on_disk=True,
        allow_provider_rescue=provider_allowed)
    unknown = plan_automatic_implementation_rescue(**common)
    assert unknown.action is AutoRescueAction.STAGE_AND_REVALIDATE
    verified = plan_automatic_implementation_rescue(
        **common, current_admitted_test_failure=True,
    )
    assert verified.action is (
        AutoRescueAction.INLINE_PROVIDER_RESCUE if provider_allowed else AutoRescueAction.NONE
    )
