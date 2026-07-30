from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_mutation_lock_path,
    update_checkout_mutation_lease,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.persistent_task_queue import (
    PersistentTaskQueue,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    DualTaskSource,
    TaskSourceIntegrityError,
    open_task_source,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
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


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    return repo


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


def _seed_plain_protected_daemon(
    tmp_path: Path,
    *,
    runtime: _CompletionRuntime | None = None,
) -> tuple[Path, Path, PortalImplementationDaemon]:
    repo = _init_repo(tmp_path)
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
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=runtime,
    )
    return repo, todo_path, daemon


def _materialize_canonical_sources(
    repo: Path,
    *,
    markdown_name: str = "todo.md",
    database_name: str = "tasks.duckdb",
):
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
        DuckDBTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
        MarkdownTaskSource,
    )
    from test.api.test_agent_supervisor_task_source_e2e import (
        _canonical_fixture,
    )

    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown_path = repo / markdown_name
    markdown_backend = MarkdownTaskSource(
        markdown_path,
        root=repo,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    markdown_backend.materialize(admission, aliases=aliases)
    database_path = repo / database_name
    database_backend = DuckDBTaskSource(database_path)
    database_backend.materialize(
        graph,
        repository_tree_id=tree_id,
    )
    return (
        markdown_path,
        database_path,
        open_task_source(markdown_backend),
        open_task_source(database_backend),
    )


def _completion_intent(
    daemon: PortalImplementationDaemon,
    repo: Path,
    task_id: str,
) -> dict[str, object]:
    task = next(
        candidate
        for candidate in daemon._load_tasks()
        if candidate.task_id == task_id
    )
    return daemon._completion_publication_intent(
        task,
        merged_tree_id=_git(repo, "rev-parse", "HEAD"),
        evidence={
            "passed": True,
            "completion_authoritative": True,
        },
    )


def _make_retained_lease_owner_dead(
    daemon: PortalImplementationDaemon,
):
    lease = daemon._current_checkout_mutation_lease()
    assert lease is not None
    dead = update_checkout_mutation_lease(
        lease,
        {
            **dict(lease.metadata),
            "pid": 999_999_999,
            "owner_script": "dead-implementation-daemon",
        },
    )
    assert dead is not None
    daemon._clear_checkout_mutation_context()
    return dead


def _assert_queue_completed(
    queue_path: Path,
    task_cid: str,
) -> None:
    queue = PersistentTaskQueue.load(queue_path)
    entry = queue.entries.get(queue.resolve_key(task_cid))
    assert entry is not None
    assert entry.last_completed_at > 0


def test_restart_reuses_durable_publication_after_lease_journal_cas_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_runtime = _CompletionRuntime()
    repo, todo_path, daemon = _seed_plain_protected_daemon(
        tmp_path,
        runtime=first_runtime,
    )
    intent = _completion_intent(daemon, repo, "ACCEL-001")
    sink = intent["publication_sink"]
    assert isinstance(sink, dict)
    queue_path = Path(str(sink["task_queue_path"]))

    queue_successes: list[tuple[Path, str]] = []
    original_record_success = PersistentTaskQueue.record_success

    def capture_record_success(
        queue: PersistentTaskQueue,
        task_cid: str,
    ) -> None:
        assert queue._path is not None
        queue_successes.append(
            (queue._path.resolve(strict=False), task_cid)
        )
        original_record_success(queue, task_cid)

    monkeypatch.setattr(
        PersistentTaskQueue,
        "record_success",
        capture_record_success,
    )
    original_update = daemon_module.update_checkout_mutation_lease
    failed_publication_journal = False

    def fail_first_publication_journal(lease, metadata, **kwargs):
        nonlocal failed_publication_journal
        if (
            not failed_publication_journal
            and "completion_publication" in metadata
        ):
            failed_publication_journal = True
            return None
        return original_update(lease, metadata, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "update_checkout_mutation_lease",
        fail_first_publication_journal,
    )

    first = daemon._mark_task_completed_in_todo(
        "ACCEL-001",
        completion_intent=intent,
    )

    assert first["checkout_mutation_lease_retained"] is True
    assert len(first_runtime.completion_routes) == 1
    assert queue_successes == [
        (
            queue_path.resolve(strict=False),
            str(intent["queue_task_cid"]),
        )
    ]
    publication_path = daemon._completion_publication_record_path(
        sink,
        str(intent["intent_id"]),
    )
    publication_record = json.loads(
        publication_path.read_text(encoding="utf-8")
    )
    assert publication_record["publication"]["published"] is True
    assert publication_record["publication"]["decision_receipt_id"] == (
        "completion-1"
    )
    assert publication_record["queue_recorded"] is True

    _make_retained_lease_owner_dead(daemon)
    restarted_runtime = _CompletionRuntime()
    restarted = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=restarted_runtime,
    )

    recovered = restarted._recover_protected_checkout_mutation()

    assert recovered["checkout_mutation_lease_recovered"] is True
    assert restarted_runtime.completion_routes == []
    assert len(first_runtime.completion_routes) == 1
    assert queue_successes == [
        (
            queue_path.resolve(strict=False),
            str(intent["queue_task_cid"]),
        )
    ]
    _assert_queue_completed(
        queue_path,
        str(intent["queue_task_cid"]),
    )
    assert not checkout_mutation_lock_path(repo).exists()


def test_partial_protected_bundle_survives_commit_before_marker_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    (
        todo_path,
        _database_path,
        _markdown,
        _database,
    ) = _materialize_canonical_sources(repo)
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed canonical todo")
    state_dir = repo / "state"
    initial_runtime = _CompletionRuntime()
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind="markdown",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## FIX-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=initial_runtime,
    )
    intent = _completion_intent(daemon, repo, "FIX-001")
    original_cas = daemon.task_source.compare_and_swap_status
    cas_calls = 0

    def fail_second_cas(*args, **kwargs):
        nonlocal cas_calls
        cas_calls += 1
        if cas_calls == 2:
            raise TaskSourceIntegrityError(
                "injected second-member failure"
            )
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(
        daemon.task_source,
        "compare_and_swap_status",
        fail_second_cas,
    )

    partial = daemon._mark_tasks_completed_in_todo(
        ("FIX-001", "FIX-002"),
        primary_task_id="FIX-001",
        completion_reason="bundle_work_order",
        completion_intent=intent,
    )

    assert partial["reason"] == "task_source_update_failed"
    assert partial["checkout_mutation_lease_retained"] is True
    assert initial_runtime.completion_routes == []
    assert daemon.task_source.get("FIX-001").status == "completed"
    assert daemon.task_source.get("FIX-002").status != "completed"

    _make_retained_lease_owner_dead(daemon)
    replay_runtime = _CompletionRuntime()
    replaying = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind="markdown",
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=repo,
        task_header_prefix="## FIX-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=replay_runtime,
    )
    original_update = daemon_module.update_checkout_mutation_lease
    failed_callback_marker = False

    def fail_first_callback_marker(lease, metadata, **kwargs):
        nonlocal failed_callback_marker
        if (
            not failed_callback_marker
            and "protected_callback_success" in metadata
        ):
            failed_callback_marker = True
            return None
        return original_update(lease, metadata, **kwargs)

    monkeypatch.setattr(
        daemon_module,
        "update_checkout_mutation_lease",
        fail_first_callback_marker,
    )

    replayed = replaying._recover_protected_checkout_mutation()

    assert replayed["updated_task_ids"] == ["FIX-002"]
    assert replayed["already_completed_task_ids"] == ["FIX-001"]
    assert replayed["reason"] == (
        "protected_recovery_callback_marker_pending"
    )
    assert replayed["checkout_mutation_lease_retained"] is True
    assert replay_runtime.completion_routes == []
    assert _git(repo, "status", "--porcelain", "--", "todo.md") == ""

    monkeypatch.setattr(
        daemon_module,
        "update_checkout_mutation_lease",
        original_update,
    )
    _make_retained_lease_owner_dead(replaying)
    final_runtime = _CompletionRuntime()
    final_daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        task_source_kind="markdown",
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=repo,
        task_header_prefix="## FIX-",
        implementation_protected_paths=("todo.md",),
        decision_runtime=final_runtime,
    )
    released_metadata: list[dict[str, object]] = []
    original_release = final_daemon._release_checkout_mutation_lease

    def capture_release(lease) -> bool:
        released_metadata.append(dict(lease.metadata))
        return original_release(lease)

    monkeypatch.setattr(
        final_daemon,
        "_release_checkout_mutation_lease",
        capture_release,
    )

    recovered = final_daemon._recover_protected_checkout_mutation()

    assert recovered["checkout_mutation_lease_recovered"] is True
    assert final_daemon.task_source.get("FIX-001").status == "completed"
    assert final_daemon.task_source.get("FIX-002").status == "completed"
    assert len(final_runtime.completion_routes) == 1
    assert replay_runtime.completion_routes == []
    marker = released_metadata[-1]["protected_callback_success"]
    assert isinstance(marker, dict)
    assert [member["task_id"] for member in marker["members"]] == [
        "FIX-001",
        "FIX-002",
    ]
    assert released_metadata[-1]["completion_publication"]["published"] is (
        True
    )
    assert not checkout_mutation_lock_path(repo).exists()


def test_external_partial_bundle_is_non_durable_until_recovery_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    (
        _markdown_path,
        database_path,
        _markdown,
        _database,
    ) = _materialize_canonical_sources(repo)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed repository")
    state_dir = repo / "state"
    first_runtime = _CompletionRuntime()
    daemon = PortalImplementationDaemon(
        todo_path=database_path,
        task_source_kind="duckdb",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        decision_runtime=first_runtime,
    )
    intent = _completion_intent(daemon, repo, "FIX-001")
    original_cas = daemon.task_source.compare_and_swap_status
    cas_calls = 0

    def fail_second_cas(*args, **kwargs):
        nonlocal cas_calls
        cas_calls += 1
        if cas_calls == 2:
            raise TaskSourceIntegrityError(
                "injected external second-member failure"
            )
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(
        daemon.task_source,
        "compare_and_swap_status",
        fail_second_cas,
    )

    partial = daemon._mark_tasks_completed_in_todo(
        ("FIX-001", "FIX-002"),
        primary_task_id="FIX-001",
        completion_reason="bundle_work_order",
        completion_intent=intent,
    )

    assert partial["reason"] == "task_source_update_failed"
    assert partial["durable"] is False
    assert daemon._todo_completion_is_durable(partial) is False
    assert partial["completion_callback_recovery_required"] is True
    assert "completion_publication" not in partial
    assert first_runtime.completion_routes == []
    sink = intent["publication_sink"]
    assert isinstance(sink, dict)
    publication_path = daemon._completion_publication_record_path(
        sink,
        str(intent["intent_id"]),
    )
    assert not publication_path.exists()
    callback_path = daemon._completion_callback_record_path(
        sink,
        str(intent["intent_id"]),
    )
    assert json.loads(
        callback_path.read_text(encoding="utf-8")
    )["phase"] == "pending"

    restarted_runtime = _CompletionRuntime()
    restarted = PortalImplementationDaemon(
        todo_path=database_path,
        task_source_kind="duckdb",
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=repo,
        decision_runtime=restarted_runtime,
    )

    recovered = restarted._recover_pending_external_completion_callbacks()

    assert recovered == {
        "required": True,
        "blocked": False,
        "recovered": 1,
    }
    assert restarted.task_source.get("FIX-001").status == "completed"
    assert restarted.task_source.get("FIX-002").status == "completed"
    assert len(restarted_runtime.completion_routes) == 1
    assert publication_path.exists()
    completed_callback = json.loads(
        callback_path.read_text(encoding="utf-8")
    )
    assert completed_callback["phase"] == "completed"
    assert completed_callback["completion_publication"]["published"] is True


def test_cross_state_recovery_uses_only_journal_bound_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_runtime = _CompletionRuntime()
    repo, primary_todo, primary = _seed_plain_protected_daemon(
        tmp_path,
    )
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
        decision_runtime=producer_runtime,
    )
    intent = _completion_intent(producer, repo, "OTHER-001")

    def fail_publication(_intent):
        raise RuntimeError("leave completion intent for cross-state recovery")

    monkeypatch.setattr(
        producer,
        "_publish_completion_intent",
        fail_publication,
    )

    retained = producer._mark_task_completed_in_todo(
        "OTHER-001",
        completion_intent=intent,
    )

    assert retained["checkout_mutation_lease_retained"] is True
    assert retained["completion_publication"]["published"] is False
    assert producer_runtime.completion_routes == []
    _make_retained_lease_owner_dead(producer)

    consumer_runtime = _CompletionRuntime()
    consumer = PortalImplementationDaemon(
        todo_path=primary_todo,
        state_path=primary.state_path,
        strategy_path=primary.strategy_path,
        events_path=primary.events_path,
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implementation_protected_paths=("todo.md", "secondary.md"),
        decision_runtime=consumer_runtime,
    )

    recovered = consumer._recover_protected_checkout_mutation()

    assert recovered["checkout_mutation_lease_recovered"] is True
    assert len(consumer_runtime.completion_routes) == 1
    sink = intent["publication_sink"]
    assert isinstance(sink, dict)
    target_queue_path = Path(str(sink["task_queue_path"]))
    adopter_queue_path = primary.state_path.parent / "task_queue.json"
    assert target_queue_path != adopter_queue_path
    _assert_queue_completed(
        target_queue_path,
        str(intent["queue_task_cid"]),
    )
    adopter_queue = PersistentTaskQueue.load(adopter_queue_path)
    assert (
        adopter_queue.resolve_key(str(intent["queue_task_cid"]))
        not in adopter_queue.entries
    )
    assert not checkout_mutation_lock_path(repo).exists()


def test_dual_task_source_is_classified_as_protected_markdown_writer(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (
        markdown_path,
        _database_path,
        markdown,
        database,
    ) = _materialize_canonical_sources(repo)
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed dual projection")
    state_dir = repo / "state"
    state_dir.mkdir()
    dual = DualTaskSource(
        database,
        markdown,
        journal_path=state_dir / "dual-task-source.json",
    )
    daemon = PortalImplementationDaemon(
        task_source=dual,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_protected_paths=("todo.md",),
    )

    assert daemon._task_source_writes_markdown_checkout() is True
    assert daemon._task_source_markdown_checkout_paths() == (
        markdown_path,
    )
    assert daemon._todo_board_is_implementation_protected() is True
    assert daemon._todo_mutation_requires_checkout_lease() is True
    assert daemon._protected_paths_for_checkout_mutation(
        "mark_tasks_completed",
        None,
    ) == (markdown_path.resolve(),)
    expectation = daemon._completion_callback_expectation(("FIX-001",))
    binding = expectation["task_source"]
    assert binding["source_kind"] == "dual"
    assert binding["writes_markdown_checkout"] is True
    assert binding["protected_checkout"] is True
