"""Opt-in contract refusals happen before provider effects or spent attempts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module


def _task(**changes):
    fields = dict(
        task_id="CONTRACT-001", title="Produce reviewed output", status="ready",
        completion="manual", priority="P1", track="runtime", outputs=[],
        validation=[], metadata={}, canonical_task_cid="task:reviewed-contract",
    )
    fields.update(changes)
    return module.PortalTask(**fields)


def _outer(tmp_path):
    daemon = object.__new__(module.PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon._canonical_ref = lambda task: task.canonical_task_cid
    return daemon


@pytest.mark.parametrize("enabled, writing", [(False, True), (True, False)])
def test_legacy_and_explicit_local_execution_remain_unchanged(tmp_path, monkeypatch, enabled, writing):
    daemon = _outer(tmp_path)
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, str(int(enabled)))
    daemon._proposal_scope_paths = lambda _task: pytest.fail("exempt tasks must not acquire new requirements")
    daemon._require_provider_task_contract(_task(), provider_writing=writing)


@pytest.mark.parametrize("outputs, validation, reason", [
    ([], [], "provider_task_contract_missing_scope_and_validation"),
    ([], ["python -m pytest tests/test_contract.py"], "provider_task_contract_missing_editable_scope"),
    (["result.json"], [], "provider_task_contract_missing_validation_commands"),
])
def test_missing_fields_have_stable_typed_nonconsuming_receipts(tmp_path, monkeypatch, outputs, validation, reason):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    daemon = _outer(tmp_path)
    task = _task(outputs=outputs, validation=validation)
    receipts = []
    for _ in range(2):
        with pytest.raises(module.ProviderTaskContractDeferred) as caught:
            daemon._require_provider_task_contract(task, provider_writing=True)
        error = caught.value
        assert isinstance(error, module.ImplementationRetryDeferred)
        assert error.reason == reason
        assert error.backoff_seconds == 300
        assert error.receipt["provider_call_allowed"] is False
        assert error.receipt["attempt_consumed"] is False
        assert error.receipt["canonical_task_cid"] == task.canonical_task_cid
        receipts.append(error.receipt)
    assert receipts[0] == receipts[1]
    assert receipts[0]["contract_fingerprint"]


@pytest.mark.parametrize("placeholder", ["", " ", "none", "N/A", "true", ":", "# validation later", "`none`"])
def test_placeholder_validation_is_not_a_command(tmp_path, monkeypatch, placeholder):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        _outer(tmp_path)._require_provider_task_contract(
            _task(outputs=["result.json"], validation=[placeholder]), provider_writing=True,
        )
    assert caught.value.reason == "provider_task_contract_missing_validation_commands"


@pytest.mark.parametrize("declaration", ["allowed paths", "predicted files"])
def test_reviewed_scope_does_not_require_output_deliverables(tmp_path, monkeypatch, declaration):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    _outer(tmp_path)._require_provider_task_contract(
        _task(metadata={declaration: "src/helper.py, tests/test_contract.py"},
              validation=["python -m pytest tests/test_contract.py"]),
        provider_writing=True,
    )


def test_invalid_shell_command_is_deferred_before_provider(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        _outer(tmp_path)._require_provider_task_contract(
            _task(outputs=["result.json"], validation=["python $(choose_validation.py)"]),
            provider_writing=True,
        )
    assert caught.value.reason == "provider_task_contract_invalid_validation_command"


@pytest.mark.parametrize("command", [
    "`python -m pytest tests/test_contract.py`",
    'test -z "$(rg -n \'ignore=tests/foo\' Makefile || true)"',
])
def test_execution_supported_command_normalization_is_preserved(tmp_path, monkeypatch, command):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    _outer(tmp_path)._require_provider_task_contract(
        _task(outputs=["result.json"], validation=[command]), provider_writing=True,
    )


def test_contract_revision_changes_fingerprint_and_can_be_repaired(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    daemon = _outer(tmp_path)
    task = _task(validation=["python -m pytest tests/test_a.py"])
    with pytest.raises(module.ProviderTaskContractDeferred) as first:
        daemon._require_provider_task_contract(task, provider_writing=True)
    task = replace(task, validation=["python -m pytest tests/test_b.py"],
                   canonical_task_cid="task:revised-contract")
    with pytest.raises(module.ProviderTaskContractDeferred) as second:
        daemon._require_provider_task_contract(task, provider_writing=True)
    assert first.value.receipt["contract_fingerprint"] != second.value.receipt["contract_fingerprint"]
    task = replace(task, metadata={"allowed paths": "result.json"})
    daemon._require_provider_task_contract(task, provider_writing=True)


def test_invalid_task_defers_before_worktree_and_other_ready_task_advances(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    board = repo / "tasks.md"
    board.write_text(
        "# Tasks\n\n## CONTRACT-001 Missing contract\n\n"
        "- Status: todo\n- Completion: manual\n- Priority: P0\n- Track: runtime\n"
        "\n## CONTRACT-002 Valid contract\n\n"
        "- Status: todo\n- Completion: manual\n- Priority: P1\n- Track: runtime\n"
        "- Outputs: result.json\n- Validation: python -m pytest tests/test_contract.py\n"
    )
    for command in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "tasks.md"],
        ["git", "-c", "user.name=Contract Test", "-c", "user.email=contract@example.invalid",
         "commit", "-qm", "reviewed test tasks"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    state_dir = repo / "state"
    daemon = module.PortalImplementationDaemon(
        todo_path=board, state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        repo_root=repo, task_header_prefix="## CONTRACT-", implement=True,
        use_ephemeral_worktree=True, worktree_root=tmp_path / "worktrees",
    )
    reached = []

    def provider_boundary(task):
        reached.append(task.task_id)
        raise module.ImplementationRetryDeferred("authored provider stop", backoff_seconds=300)

    monkeypatch.setattr(daemon, "_require_primary_provider_readiness", provider_boundary)
    monkeypatch.setattr(daemon, "_build_implementation_prompt", lambda *_args: pytest.fail("no prompt authorized"))
    first = daemon.run_once()["implementation_result"]
    assert first["reason"] == "provider_task_contract_missing_scope_and_validation"
    assert first["provider_dispatched"] is first["attempt_consumed"] is False
    assert first["deferred"] is True
    assert first["retryable"] is True
    assert first["contract_fingerprint"] == first["provider_task_contract"]["contract_fingerprint"]
    assert reached == []
    second = daemon.run_once()["implementation_result"]
    assert second["task_id"] == "CONTRACT-002"
    assert second["reason"] == "authored_provider_stop"
    assert reached == ["CONTRACT-002"]
    state = module.PortalTaskState.load(daemon.state_path)
    assert not state.implementation_attempts
    assert not state.implementation_attempts_by_cid
    assert state.implementation_in_progress is False
    assert not list((tmp_path / "worktrees").glob("workspace_*"))
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert any(event.get("type") == "implementation_retry_deferred"
               and event.get("contract_fingerprint") == first["contract_fingerprint"] for event in events)
    assert not any(event.get("type") in {"provider_invocation_committed", "provider_callback_started"}
                   for event in events)
