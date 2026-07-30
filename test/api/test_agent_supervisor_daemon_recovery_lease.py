from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon as daemon_module
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_mutation_lock_path,
    update_checkout_mutation_lease,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    TaskSourceIntegrityError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class _CompletionRuntime:
    def __init__(self) -> None:
        self.completion_routes: list[dict[str, object]] = []

    def route(self, boundary: str, payload: dict[str, object]):
        if boundary != "completion":
            return None
        self.completion_routes.append(dict(payload))
        return SimpleNamespace(
            receipt=SimpleNamespace(
                receipt_id=f"completion-{len(self.completion_routes)}"
            )
        )


def _protected_daemon(
    tmp_path: Path,
    *,
    task_source_kind: str = "",
) -> tuple[Path, Path, PortalImplementationDaemon]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Todos

## ACCEL-001 Protected completion

- Status: todo
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed todo")
    state_dir = repo / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind=task_source_kind,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md",),
    )
    return repo, todo_path, daemon


def test_protected_callback_exception_is_recovered_before_run_once_task_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, todo_path, daemon = _protected_daemon(tmp_path)
    original_commit = daemon._commit_generated_file_update_locked

    def fail_after_board_write(*_args, **_kwargs):
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        fail_after_board_write,
    )
    with pytest.raises(RuntimeError, match="injected commit failure"):
        daemon._mark_task_completed_in_todo("ACCEL-001")

    lease = daemon._current_checkout_mutation_lease()
    assert lease is not None
    metadata = json.loads(lease.lock_path.read_text(encoding="utf-8"))
    assert metadata["protected_recovery_required"] is True
    assert metadata["protected_recovery_intent"]["task_id"] == "ACCEL-001"
    assert "todo.md" in _git(repo, "status", "--porcelain", "--", "todo.md")

    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        original_commit,
    )
    task_loads: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_load_tasks",
        lambda: task_loads.append("loaded") or [],
    )

    daemon.run_once()

    assert task_loads == ["loaded"]
    assert daemon._current_checkout_mutation_lease() is None
    assert not checkout_mutation_lock_path(repo).exists()
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")


def test_dead_daemon_recovery_lease_is_cas_adopted_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, todo_path, daemon = _protected_daemon(tmp_path)
    original_commit = daemon._commit_generated_file_update_locked
    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("crash after board write")
        ),
    )
    with pytest.raises(RuntimeError, match="crash after board write"):
        daemon._mark_task_completed_in_todo("ACCEL-001")

    old_lease = daemon._current_checkout_mutation_lease()
    assert old_lease is not None
    dead_metadata = {
        **dict(old_lease.metadata),
        "pid": 999_999_999,
        "owner_script": "dead-implementation-daemon",
    }
    dead_lease = update_checkout_mutation_lease(
        old_lease,
        dead_metadata,
    )
    assert dead_lease is not None
    daemon._clear_checkout_mutation_context()

    restarted = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md",),
    )
    task_loads: list[str] = []
    monkeypatch.setattr(
        restarted,
        "_load_tasks",
        lambda: task_loads.append("loaded") or [],
    )
    restarted.run_once()

    assert task_loads == ["loaded"]
    assert not checkout_mutation_lock_path(repo).exists()
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    adopted = [
        event
        for event in events
        if event["type"] == "checkout_mutation_recovery_adopted"
    ]
    assert adopted
    assert adopted[-1]["prior_lease_id"] == dead_lease.lease_id

    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        original_commit,
    )


def test_exception_after_trusted_commit_retains_journal_until_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _todo_path, daemon = _protected_daemon(tmp_path)
    original_commit = daemon._commit_generated_file_update_locked

    def commit_then_raise(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise RuntimeError("crash before terminal publication")

    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        commit_then_raise,
    )
    with pytest.raises(RuntimeError, match="terminal publication"):
        daemon._mark_task_completed_in_todo("ACCEL-001")

    assert daemon._current_checkout_mutation_lease() is not None
    assert checkout_mutation_lock_path(repo).exists()
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""

    monkeypatch.setattr(
        daemon,
        "_commit_generated_file_update_locked",
        original_commit,
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [])
    daemon.run_once()

    assert daemon._current_checkout_mutation_lease() is None
    assert not checkout_mutation_lock_path(repo).exists()


def test_protected_markdown_task_source_cas_is_fenced_and_committed(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
        MarkdownTaskSource,
    )
    from test.api.test_agent_supervisor_task_source_e2e import (
        _canonical_fixture,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _graph, admission, aliases, _tree_id = _canonical_fixture()
    todo_path = repo / "todo.md"
    MarkdownTaskSource(
        todo_path,
        root=repo,
        task_prefix="FIX",
        board_namespace="fixture",
    ).materialize(admission, aliases=aliases)
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed canonical todo")
    state_dir = repo / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind="markdown",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## FIX-",
        implementation_protected_paths=("todo.md",),
    )

    result = daemon._mark_task_completed_in_todo("FIX-001")

    assert result["updated"] is True
    assert result["durable"] is True
    assert result["protected_board_postcondition"]["trusted"] is True
    assert result["commit_result"]["committed"] is True
    assert not checkout_mutation_lock_path(repo).exists()
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8")


def test_nested_non_recovery_mutation_is_denied_by_protected_capability(
    tmp_path: Path,
) -> None:
    _repo, _todo_path, daemon = _protected_daemon(tmp_path)
    nested_calls: list[str] = []

    def protected_callback() -> dict[str, object]:
        nested = daemon._run_checkout_mutation_transaction(
            task_id="OTHER-1",
            operation="merge_branch_to_main",
            callback=lambda: nested_calls.append("called") or {"merged": True},
            failure_fields={"merged": False},
        )
        return {"nested": nested}

    result = daemon._run_checkout_mutation_transaction(
        task_id="ACCEL-001",
        operation="mark_tasks_completed",
        callback=protected_callback,
        failure_fields={"updated": False},
    )

    assert nested_calls == []
    assert result["nested"]["merged"] is False
    assert result["nested"]["reason"] == (
        "checkout_mutation_nested_operation_not_allowed"
    )


def test_completion_publication_reuses_receipt_after_journal_cas_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _todo_path, daemon = _protected_daemon(tmp_path)
    runtime = _CompletionRuntime()
    daemon.decision_runtime = runtime
    task = daemon._load_tasks()[0]
    intent = daemon._completion_publication_intent(
        task,
        merged_tree_id=_git(repo, "rev-parse", "HEAD"),
        evidence={"passed": True, "completion_authoritative": True},
    )
    success_calls: list[str] = []
    original_record_success = daemon._record_exact_task_queue_success

    def record_success(queue_path: Path, task_cid: str) -> None:
        success_calls.append(task_cid)
        original_record_success(queue_path, task_cid)

    monkeypatch.setattr(
        daemon,
        "_record_exact_task_queue_success",
        record_success,
    )
    original_update = daemon_module.update_checkout_mutation_lease
    failed_publication_cas = False

    def fail_first_publication_cas(lease, metadata, **kwargs):
        nonlocal failed_publication_cas
        if (
            not failed_publication_cas
            and "completion_publication" in metadata
        ):
            failed_publication_cas = True
            return None
        return original_update(lease, metadata, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "update_checkout_mutation_lease",
        fail_first_publication_cas,
    )

    first = daemon._mark_task_completed_in_todo(
        "ACCEL-001",
        completion_intent=intent,
    )

    assert first["checkout_mutation_lease_retained"] is True
    assert len(runtime.completion_routes) == 1
    assert runtime.completion_routes[0]["completion_intent_id"] == (
        intent["intent_id"]
    )
    assert success_calls == [intent["queue_task_cid"]]

    recovered = daemon._recover_protected_checkout_mutation()

    assert recovered["checkout_mutation_lease_recovered"] is True
    assert len(runtime.completion_routes) == 1
    assert success_calls == [intent["queue_task_cid"]]
    assert not checkout_mutation_lock_path(repo).exists()


def test_partial_markdown_bundle_never_publishes_whole_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
        MarkdownTaskSource,
    )
    from test.api.test_agent_supervisor_task_source_e2e import (
        _canonical_fixture,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _graph, admission, aliases, _tree_id = _canonical_fixture()
    todo_path = repo / "todo.md"
    MarkdownTaskSource(
        todo_path,
        root=repo,
        task_prefix="FIX",
        board_namespace="fixture",
    ).materialize(admission, aliases=aliases)
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed canonical todo")
    state_dir = repo / "state"
    runtime = _CompletionRuntime()
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind="markdown",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## FIX-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=runtime,
    )
    task = next(
        item for item in daemon._load_tasks() if item.task_id == "FIX-001"
    )
    intent = daemon._completion_publication_intent(
        task,
        merged_tree_id=_git(repo, "rev-parse", "HEAD"),
        evidence={"passed": True, "completion_authoritative": True},
    )
    original_cas = daemon.task_source.compare_and_swap_status
    cas_calls = 0

    def fail_second_cas(*args, **kwargs):
        nonlocal cas_calls
        cas_calls += 1
        if cas_calls == 2:
            raise TaskSourceIntegrityError("injected second-member failure")
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(
        daemon.task_source,
        "compare_and_swap_status",
        fail_second_cas,
    )

    failed = daemon._mark_tasks_completed_in_todo(
        ("FIX-001", "FIX-002"),
        primary_task_id="FIX-001",
        completion_reason="bundle_work_order",
        completion_intent=intent,
    )

    assert failed["reason"] == "task_source_update_failed"
    assert failed["checkout_mutation_lease_retained"] is True
    assert runtime.completion_routes == []
    assert (
        daemon.task_source.get("FIX-001").status == "completed"
    )
    assert (
        daemon.task_source.get("FIX-002").status != "completed"
    )

    monkeypatch.setattr(
        daemon.task_source,
        "compare_and_swap_status",
        original_cas,
    )
    recovery = daemon._recover_protected_checkout_mutation()

    assert recovery["checkout_mutation_lease_recovered"] is True
    assert runtime.completion_routes != []
    assert (
        daemon.task_source.get("FIX-002").status == "completed"
    )
    assert not checkout_mutation_lock_path(repo).exists()


def test_same_process_consumer_recovers_foreign_state_and_todo_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, primary_todo, primary = _protected_daemon(tmp_path)
    secondary_todo = repo / "secondary.md"
    secondary_todo.write_text(
        """# Todos

## OTHER-001 Cross-state completion

- Status: todo
- Priority: P1
- Track: ops
""",
        encoding="utf-8",
    )
    _git(repo, "add", "secondary.md")
    _git(repo, "commit", "-m", "seed secondary todo")
    secondary_state = repo / "secondary-state"
    producer = PortalImplementationDaemon(
        todo_path=secondary_todo,
        state_path=secondary_state / "task_state.json",
        strategy_path=secondary_state / "strategy.json",
        events_path=secondary_state / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## OTHER-",
        implementation_protected_paths=("todo.md", "secondary.md"),
    )
    monkeypatch.setattr(
        producer,
        "_commit_generated_file_update_locked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("leave cross-state recovery lease")
        ),
    )
    with pytest.raises(RuntimeError, match="cross-state"):
        producer._mark_task_completed_in_todo("OTHER-001")

    consumer = PortalImplementationDaemon(
        todo_path=primary_todo,
        state_path=primary.state_path,
        strategy_path=primary.strategy_path,
        events_path=primary.events_path,
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md", "secondary.md"),
    )

    recovered = consumer._recover_protected_checkout_mutation()

    assert recovered["checkout_mutation_lease_recovered"] is True
    assert not checkout_mutation_lock_path(repo).exists()
    assert "- Status: completed" in secondary_todo.read_text(
        encoding="utf-8"
    )
    events = [
        json.loads(line)
        for line in primary.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        event["type"] == "checkout_mutation_recovery_attached"
        for event in events
    )
