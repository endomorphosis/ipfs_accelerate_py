"""Real Git publication contention; synthetic provider boundary, no services."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import completion_git_lock as continuation
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from test.api.test_grok_turn_limit_candidate import native_case
from test.api.test_provider_failure_checkpoint_retention import git, run_failed_child


@pytest.fixture
def lock_case(native_case, monkeypatch):
    daemon, task, state, calls = native_case
    lock = Path(git(daemon.repo_root, "rev-parse", "--absolute-git-dir")) / "index.lock"
    original = daemon._mark_task_completed_in_todo
    created = threading.Event()
    def publish(*args, **kwargs):
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, b"synthetic foreign Git owner\n")
        os.close(fd)
        created.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(daemon, "_mark_task_completed_in_todo", publish)
    yield daemon, task, state, calls, lock, created
    if lock.exists():
        assert lock.read_bytes() == b"synthetic foreign Git owner\n"
        lock.unlink()


def _records(daemon):
    return list((daemon.state_path.parent / "completion-git-locks").glob("*.json"))


def _request(daemon, result):
    return daemon.merge_queue.get(result["merge_result"]["request_id"])


def _fresh(daemon):
    return native.PortalImplementationDaemon(todo_path=daemon.todo_path, state_path=daemon.state_path,
        strategy_path=daemon.strategy_path, events_path=daemon.events_path,
        repo_root=daemon.repo_root, task_header_prefix=daemon.task_header_prefix,
        worktree_root=daemon.worktree_root, merge_target_branch=daemon.resolved_merge_target_branch,
        merge_queue_dir=daemon.merge_queue_dir, board_namespace=daemon.board_namespace)


def test_exact_transient_deferral_retries_without_removing_foreign_git_lock(lock_case, monkeypatch):
    daemon, task, state, calls, lock, created = lock_case
    seen = threading.Event()
    original_event = daemon._record_event
    def event(kind, payload, **kwargs):
        if kind == "checkout_git_mutation_deferred":
            seen.set()
        return original_event(kind, payload, **kwargs)
    monkeypatch.setattr(daemon, "_record_event", event)
    def owner_releases():
        assert created.wait(10) and seen.wait(10)
        assert lock.read_bytes() == b"synthetic foreign Git owner\n"
        lock.unlink()  # Only the synthetic foreign owner releases its inode.
    owner = threading.Thread(target=owner_releases)
    owner.start()
    result = run_failed_child(daemon, task, state)
    owner.join(timeout=10)
    assert not owner.is_alive()
    assert result["merge_result"]["merged"], result
    assert result["board_completion"]["complete"] and daemon._load_tasks()[0].status == "completed"
    assert len(calls) == 1 and state.implementation_attempts[task.task_id] == 1
    assert seen.is_set() and len(_records(daemon)) == 1
    assert _request(daemon, result).status == "completed"


def test_persistent_contention_survives_restart_and_publishes_completion_only(lock_case, monkeypatch):
    daemon, task, state, calls, lock, _ = lock_case
    result = run_failed_child(daemon, task, state)
    assert result["board_completion"]["pending_merge"] and not result["board_completion"]["complete"], result
    assert daemon._load_tasks()[0].status == "todo"
    request = _request(daemon, result)
    assert request.status == "pending" and request.failure_count == 0
    assert request.metadata["deferrals"][-1]["metadata"]["completion_only_pending"] is True
    assert not daemon.merge_queue.pending_requests()
    assert [item.request_id for item in daemon.merge_queue.pending_requests(include_deferred=True)] == [request.request_id]
    assert lock.read_bytes() == b"synthetic foreign Git owner\n"
    record = _records(daemon)[0]
    before = record.read_bytes()
    integration = git(daemon.repo_root, "rev-parse", "HEAD")
    waiting = _fresh(daemon)
    monkeypatch.setattr(waiting, "_select_next_task", lambda *a, **k: pytest.fail("pending completion cannot select another provider"))
    tick = waiting.run_once()
    assert tick["blocked"] and tick["selection_idle_reason"] == "integrated_completion_pending", tick
    assert tick["implementation_result"] is None and tick["provider_dispatched"] is False
    lock.unlink()
    fresh = _fresh(daemon)
    monkeypatch.setattr(fresh, "_merge_branch_to_main", lambda *a, **k: pytest.fail("completion cannot remerge"))
    monkeypatch.setattr(fresh, "_merge_train_callback", lambda *a, **k: pytest.fail("filtered continuation cannot invoke merger"))
    # Advance the queue's synthetic clock through its real durable cooldown.
    fresh.merge_queue._clock = lambda: request.retry_not_before + 1
    progress = fresh._consume_one_merge_candidate()
    assert progress["accepted"] is True and progress["merge_result"]["reason"] == "merge_completion_git_lock_published", progress
    assert fresh._load_tasks()[0].status == "completed"
    assert fresh.merge_queue.get(request.request_id).status == "completed"
    assert record.read_bytes() == before
    assert git(daemon.repo_root, "rev-list", "--min-parents=2", "HEAD").splitlines() == [integration]
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["manual_hold", "durable_hold", "strategy", "tamper", "symlink", "stale_target", "validation", "quarantine", "protected_fence", "inner_hold", "inner_tamper", "database_authority"])
def test_changed_authority_remains_pending_without_completion_or_remerge(lock_case, monkeypatch, damage):
    daemon, task, state, calls, lock, _ = lock_case
    result = run_failed_child(daemon, task, state)
    request = _request(daemon, result)
    assert request.status == "pending", result
    record = _records(daemon)[0]
    before = record.read_bytes()
    lock.unlink()
    if damage == "manual_hold":
        daemon.todo_path.write_text(daemon.todo_path.read_text() + "- Manual hold: true\n")
    elif damage == "durable_hold":
        held = native.PortalTaskState.load(daemon.state_path)
        held.blocked_task_ids = [task.task_id]
        held.save(daemon.state_path)
    elif damage == "strategy":
        daemon.strategy_path.write_text(json.dumps({"blocked_tasks": [task.task_id]}))
    elif damage == "tamper":
        payload = json.loads(before)
        payload["integration_commit"] = "a" * 40
        record.write_text(json.dumps(payload))
    elif damage == "symlink":
        record.rename(record.with_suffix(".preserved"))
        record.symlink_to(record.with_suffix(".preserved"))
    elif damage == "stale_target":
        (daemon.repo_root / "other.py").write_text("CHANGED = True\n")
        git(daemon.repo_root, "add", "other.py")
        git(daemon.repo_root, "commit", "-m", "Unvalidated newer target")
    elif damage == "validation":
        # Mutating the live row's sealed candidate metadata must not be
        # substituted for the originally retained proof.
        with daemon.merge_queue._connect() as db:
            metadata = dict(request.metadata)
            metadata["validation_proof"] = {**metadata["validation_proof"], "passed": False}
            db.execute("UPDATE merge_requests SET metadata_json=? WHERE request_id=?", (json.dumps(metadata), request.request_id))
    elif damage == "quarantine":
        # The filtered consumer never revives this exact row, even integrated.
        with daemon.merge_queue._connect() as db:
            db.execute("UPDATE merge_requests SET status='quarantined' WHERE request_id=?", (request.request_id,))
    elif damage == "protected_fence":
        daemon._implementation_protected_active_snapshot_path().write_text('{"synthetic":"unknown active fence"}')
    fresh = _fresh(daemon)
    fresh.merge_queue._clock = lambda: request.retry_not_before + 1
    if damage == "database_authority":
        fresh._database_attempt_authority = object()
    if damage in {"inner_hold", "inner_tamper"}:
        from contextlib import contextmanager
        original_guard = continuation.checkout_git_mutation
        @contextmanager
        def after_lock(*args, **kwargs):
            with original_guard(*args, **kwargs) as value:
                if damage == "inner_hold":
                    held = native.PortalTaskState.load(fresh.state_path)
                    held.blocked_task_ids = [task.task_id]
                    held.save(fresh.state_path)
                else:
                    record.write_text('{"synthetic":"changed after checkout acquisition"}')
                yield value
        monkeypatch.setattr(continuation, "checkout_git_mutation", after_lock)
    monkeypatch.setattr(fresh, "_merge_branch_to_main", lambda *a, **k: pytest.fail("held continuation cannot merge"))
    # Call only the exact filtered consumer, excluding ordinary train's legacy
    # quarantine revival policy from this continuation authority.
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
    train = MergeTrain(fresh.repo_root, fresh.merge_queue, target_branch=fresh.resolved_merge_target_branch,
        merge_callback=lambda claimed: continuation.resume(fresh, claimed))
    progress = train.run_once(request_filter=lambda item: continuation.present(fresh, item))
    stored = fresh.merge_queue.get(request.request_id)
    assert fresh._load_tasks()[0].status != "completed" and len(calls) == 1
    if damage == "quarantine":
        assert progress is None and stored.status == "quarantined"
    else:
        assert progress["accepted"] is False and progress["acceptance_pending"] is True, progress
        assert stored.status == "pending" and stored.failure_count == 0
    if damage not in {"tamper", "symlink", "inner_tamper"}:
        assert record.read_bytes() == before


@pytest.mark.parametrize("field,value", [("attempted", True), ("updated", True), ("durable", True),
    ("committed", True), ("merged", True), ("provider_dispatched", True), ("reason", "completion_receipt_binding_mismatch"),
    ("operation", "other_operation")])
def test_unknown_or_effectful_failure_never_gets_continuation_authority(field, value):
    result = {"reason": "checkout_git_foreign_lock_present", "operation": "_mark_tasks_completed_in_todo",
        "blocked": True, "infrastructure_failure": True,
        **{key: False for key in ("attempted", "updated", "durable", "committed", "merged", "provider_dispatched")}}
    assert continuation.exact_deferral(result)
    result[field] = value
    assert not continuation.exact_deferral(result)


@pytest.mark.parametrize("failure", ["filtered_consumer", "pending_inventory"])
def test_unknown_continuation_custody_blocks_tick_before_new_selection(lock_case, monkeypatch, failure):
    daemon, task, state, calls, lock, _ = lock_case
    result = run_failed_child(daemon, task, state)
    assert _request(daemon, result).status == "pending"
    fresh = _fresh(daemon)
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
    def observe(*args, **kwargs):
        assert kwargs.get("request_filter") is not None
        if failure == "filtered_consumer":
            raise OSError("synthetic unavailable consumer")
        return None
    monkeypatch.setattr(MergeTrain, "run_once", observe)
    if failure == "pending_inventory":
        def unavailable(**kwargs):
            raise OSError("synthetic unavailable inventory")
        monkeypatch.setattr(fresh.merge_queue, "pending_requests", unavailable)
    monkeypatch.setattr(fresh, "_select_next_task", lambda *a, **k: pytest.fail("unknown custody cannot select a task"))
    head = git(daemon.repo_root, "rev-parse", "HEAD")
    tick = fresh.run_once()
    assert tick["blocked"] and tick["reason"] == "completion_continuation_observation_unavailable", tick
    assert tick["implementation_result"] is None and tick["provider_dispatched"] is False
    assert daemon._load_tasks()[0].status == "todo" and len(calls) == 1
    assert git(daemon.repo_root, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("hold", ["quarantined", "cancelled"])
def test_terminal_continuation_hold_preserves_row_history_and_stages(lock_case, monkeypatch, hold):
    from dataclasses import asdict
    daemon, task, state, calls, lock, _ = lock_case
    result = run_failed_child(daemon, task, state)
    request = _request(daemon, result)
    daemon.merge_queue._clock = lambda: request.retry_not_before + 1
    if hold == "quarantined":
        claimed = daemon.merge_queue.claim_pending_request(request.request_id, consumer_id="synthetic-operator")
        daemon.merge_queue.quarantine(claimed, reason="manual_hold")
    else:
        daemon.merge_queue.cancel(request.request_id, reason="manual_hold")
    before = asdict(daemon.merge_queue.get(request.request_id))
    stage_files = {str(path.relative_to(daemon.merge_queue.queue_dir)): path.read_bytes()
        for path in daemon.merge_queue.queue_dir.rglob("*.json")}
    fresh = _fresh(daemon)
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
    monkeypatch.setattr(MergeTrain, "run_once", lambda *a, **k: pytest.fail("terminal hold cannot enter train/revival"))
    monkeypatch.setattr(fresh, "_select_next_task", lambda *a, **k: pytest.fail("terminal hold cannot select provider"))
    tick = fresh.run_once()
    assert tick["blocked"] and tick["reason"] == "completion_continuation_terminal_hold", tick
    assert asdict(fresh.merge_queue.get(request.request_id)) == before
    assert stage_files == {str(path.relative_to(daemon.merge_queue.queue_dir)): path.read_bytes()
        for path in daemon.merge_queue.queue_dir.rglob("*.json")}
    assert daemon._load_tasks()[0].status == "todo" and len(calls) == 1


def test_first_marker_created_then_observation_failure_stays_pending(lock_case, monkeypatch):
    daemon, task, state, calls, lock, _ = lock_case
    original_event = daemon._record_event
    def fail_after_write(kind, *args, **kwargs):
        if kind == "merge_completion_git_lock_deferred":
            raise OSError("synthetic event failure after first immutable record")
        return original_event(kind, *args, **kwargs)
    monkeypatch.setattr(daemon, "_record_event", fail_after_write)
    result = run_failed_child(daemon, task, state)
    assert _records(daemon)
    assert result["merge_result"]["train_result"]["completion_only_pending"] is True, result
    assert not result["board_completion"]["complete"] and len(calls) == 1
    fresh = _fresh(daemon)
    monkeypatch.setattr(fresh, "_select_next_task", lambda *a, **k: pytest.fail("new uncertain continuation cannot select work"))
    tick = fresh.run_once()
    assert tick["blocked"] and tick["implementation_result"] is None
    assert daemon._load_tasks()[0].status == "todo"
