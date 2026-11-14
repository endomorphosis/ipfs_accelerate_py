"""Per-task tool availability refuses provider work without spending attempts."""
import json
import subprocess

import pytest

import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon as module


def task(**changes):
    values = dict(task_id="TOOLS-001", title="Validate retained evidence", status="ready",
                  completion="manual", priority="P0", track="runtime", outputs=["receipt.json"],
                  validation=["python3 -m pytest tests/test_receipt.py"], metadata={},
                  canonical_task_cid="task:tools-contract")
    values.update(changes)
    return module.PortalTask(**values)


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    outer = object.__new__(module.PortalImplementationDaemon)
    outer.repo_root = tmp_path
    outer._canonical_ref = lambda item: item.canonical_task_cid
    outer._proposal_scope_paths = lambda _item: ["receipt.json"]
    return outer


def test_absent_metadata_never_adds_an_executable_requirement(daemon, monkeypatch):
    monkeypatch.setattr(module, "validation_executable_path", lambda _: pytest.fail("tool policy was not requested"))
    daemon._require_provider_task_contract(task(), provider_writing=True)


@pytest.mark.parametrize("enabled,writing", [(False, True), (True, False)])
def test_disabled_gate_and_local_execution_keep_legacy_behavior(daemon, monkeypatch, enabled, writing):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, str(int(enabled)))
    monkeypatch.setattr(module, "validation_executable_path", lambda _: pytest.fail("exempt task checked tools"))
    daemon._require_provider_task_contract(task(metadata={"required validation executables": "../bad tool"}),
                                         provider_writing=writing)


def test_missing_tool_is_not_rescued_by_operator_path_or_working_directory(daemon, tmp_path, monkeypatch):
    name = "contract_private_tool_9b148"
    binary = tmp_path / name
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("availability must not execute a tool"))
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task(metadata={"required validation executables": name}),
                                             provider_writing=True)
    error = caught.value
    assert error.reason == "provider_task_contract_missing_validation_executables"
    assert error.receipt["missing_validation_executables"] == [name]
    assert str(tmp_path) not in error.receipt["approved_validation_executable_path"].split(":")
    assert not error.receipt["attempt_consumed"] and not error.receipt["provider_call_allowed"]


def test_available_approved_tool_is_checked_without_running_it(daemon, monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("availability must not run a tool"))
    daemon._require_provider_task_contract(task(metadata={"required validation executables": "python3"}),
                                         provider_writing=True)


@pytest.mark.parametrize("declaration", ["", "lake,,lean", "/bin/lake", "../lake", "lake --help",
                                        "lake;lean", "$(lake)", "--lake", ".", None, ["lake"]])
def test_invalid_csv_or_non_bare_names_defer_without_looking_up_tools(daemon, monkeypatch, declaration):
    monkeypatch.setattr(module, "validation_executable_path", lambda _: pytest.fail("invalid tool declaration looked up"))
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task(metadata={"required validation executables": declaration}),
                                             provider_writing=True)
    assert caught.value.reason == "provider_task_contract_invalid_required_validation_executables"
    assert caught.value.receipt["invalid_required_validation_executables"]
    assert caught.value.receipt["attempt_consumed"] is False


def test_invalid_approved_path_policy_is_a_nonconsuming_typed_deferral(daemon, monkeypatch):
    def invalid(_environment):
        raise module.ValidationRuntimeError("approved validation PATH is writable")

    monkeypatch.setattr(module, "validation_executable_path", invalid)
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task(metadata={"required validation executables": "lake, lean"}),
                                             provider_writing=True)
    assert caught.value.reason == "provider_task_contract_validation_executable_path_invalid"
    assert caught.value.receipt["missing_validation_executables"] == ["lake", "lean"]
    assert "writable" in caught.value.receipt["validation_executable_path_error"]
    assert caught.value.receipt["attempt_consumed"] is False


def test_missing_names_are_bound_into_stable_contract_fingerprint(daemon, monkeypatch):
    monkeypatch.setattr(module, "validation_executable_path", lambda _: "/approved/bin")
    available = set()
    monkeypatch.setattr(module.shutil, "which", lambda name, *, path: f"{path}/{name}" if name in available else None)
    receipts = []
    for declaration in ("lake, lean", "lean,lake,lake"):
        with pytest.raises(module.ProviderTaskContractDeferred) as caught:
            daemon._require_provider_task_contract(task(metadata={"required validation executables": declaration}),
                                                 provider_writing=True)
        receipts.append(caught.value.receipt)
    assert receipts[0] == receipts[1]
    available.add("lake")
    with pytest.raises(module.ProviderTaskContractDeferred) as caught:
        daemon._require_provider_task_contract(task(metadata={"required validation executables": "lake,lean"}),
                                             provider_writing=True)
    assert caught.value.receipt["missing_validation_executables"] == ["lean"]
    assert caught.value.receipt["contract_fingerprint"] != receipts[0]["contract_fingerprint"]
    available.add("lean")
    daemon._require_provider_task_contract(task(metadata={"required validation executables": "lake,lean"}),
                                         provider_writing=True)


def test_missing_executable_defers_only_its_task_and_other_ready_task_advances(tmp_path, monkeypatch):
    monkeypatch.setenv(module.REQUIRE_PROVIDER_TASK_CONTRACT_ENV, "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    board = repo / "tasks.md"
    board.write_text(
        "# Tasks\n\n## TOOLS-001 Requires unavailable tool\n\n"
        "- status: todo\n- completion: manual\n- priority: P0\n- track: runtime\n"
        "- outputs: receipt.json\n- validation: python3 -m pytest tests/test_receipt.py\n"
        "- required validation executables: contract_unavailable_tool_9b148\n"
        "\n## TOOLS-002 Independent valid task\n\n"
        "- status: todo\n- completion: manual\n- priority: P1\n- track: runtime\n"
        "- outputs: result.json\n- validation: python3 -m pytest tests/test_receipt.py\n"
    )
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "tasks.md"],
                    ["git", "-c", "user.name=Tool Contract Test", "-c", "user.email=tools@example.invalid",
                     "commit", "-qm", "declared task tools"]):
        subprocess.run(command, cwd=repo, check=True)
    state_dir = repo / "state"
    outer = module.PortalImplementationDaemon(todo_path=board, state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        repo_root=repo, task_header_prefix="## TOOLS-", implement=True,
        use_ephemeral_worktree=True, worktree_root=tmp_path / "worktrees")
    reached = []

    def provider_boundary(item):
        reached.append(item.task_id)
        raise module.ImplementationRetryDeferred("authored provider stop", backoff_seconds=300)

    monkeypatch.setattr(outer, "_require_primary_provider_readiness", provider_boundary)
    monkeypatch.setattr(outer, "_build_implementation_prompt", lambda *_a: pytest.fail("no prompt was authorized"))
    first = outer.run_once()["implementation_result"]
    assert first["reason"] == "provider_task_contract_missing_validation_executables"
    assert first["provider_dispatched"] is first["attempt_consumed"] is False
    assert first["provider_task_contract"]["missing_validation_executables"] == ["contract_unavailable_tool_9b148"]
    assert first["deferred"] is first["retryable"] is True
    assert reached == []
    second = outer.run_once()["implementation_result"]
    assert second["task_id"] == "TOOLS-002" and reached == ["TOOLS-002"]
    state = module.PortalTaskState.load(outer.state_path)
    assert not state.implementation_attempts and not state.implementation_attempts_by_cid
    assert not list((tmp_path / "worktrees").glob("workspace_*"))
    events = [json.loads(line) for line in outer.events_path.read_text().splitlines()]
    assert not any(event.get("type") in {"provider_invocation_committed", "provider_callback_started"} for event in events)
