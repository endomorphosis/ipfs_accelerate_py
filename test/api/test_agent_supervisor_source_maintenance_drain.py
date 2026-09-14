"""Native source drain closes new intents without changing prior work custody."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import checkout_lock as locks
from ipfs_accelerate_py.agent_supervisor.merge.source_maintenance import dispatch_drain_lease
from ipfs_accelerate_py.agent_supervisor.merge import source_maintenance as drain
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    PORTAL_RETRY_DEFERRAL_SCHEMA,
)


@pytest.fixture
def native(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = repo
    daemon.board_namespace = "source-drain-test"
    return repo, daemon, daemon._protected_path_maintenance_lock_path()


def intent(daemon, path):
    return daemon._try_acquire_implementation_dispatch_intent(path, {"task_id": "T-1"})


@pytest.mark.parametrize("existing", [False, True])
def test_drain_does_not_create_or_rewrite_attempt(native, tmp_path, monkeypatch, existing):
    repo, daemon, lock = native
    path = tmp_path / "intent.json"
    if existing:
        path.write_bytes(b"retained exact previous attempt\n")
    before = path.read_bytes() if existing else None
    monkeypatch.setattr(daemon, "_try_acquire_implementation_task_claim",
                        lambda *_: pytest.fail("drain reached claim publication/reclamation"))
    with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None) as gate:
        gate()
        result = intent(daemon, path)
        assert result[:4] == (False, "source_maintenance_drain_active", None, None)
        assert result[4]["dispatch_admission"] == "drain"
        assert (path.read_bytes() if path.exists() else None) == before
    assert not lock.exists()
    with pytest.raises(RuntimeError, match="no longer held"):
        gate()


@pytest.mark.parametrize("ordinary", [False, True])
def test_ordinary_intent_admission_and_maintenance_wait_unchanged(native, tmp_path, monkeypatch, ordinary):
    repo, daemon, lock = native
    if ordinary:
        lock.write_text(json.dumps({"kind": "implementation-protected-maintenance", "pid": os.getpid()}))
    calls = []
    monkeypatch.setattr(daemon, "_try_acquire_implementation_task_claim",
                        lambda *_: (calls.append("publish") or True, "acquired", None))
    result = intent(daemon, tmp_path / "intent.json")
    assert result[:4] == (True, "acquired", None, {"task_id": "T-1"})
    assert bool(result[4]) == ordinary
    assert calls == ["publish"]


def test_claim_publication_resumes_after_drain(native, tmp_path, monkeypatch):
    repo, daemon, _ = native
    published = []
    monkeypatch.setattr(daemon, "_try_acquire_implementation_task_claim",
                        lambda *_: (published.append(True) or True, "acquired", None))
    with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None):
        assert intent(daemon, tmp_path / "intent")[0] is False
    assert intent(daemon, tmp_path / "intent")[0] is True
    assert published == [True]


@pytest.mark.parametrize("record", [b"malformed", b'{"pid":1,"dispatch_admission":"unknown"}'])
def test_unreadable_or_unknown_drain_cannot_publish_intent(native, tmp_path, monkeypatch, record):
    _, daemon, lock = native
    lock.write_bytes(record)
    monkeypatch.setattr(daemon, "_try_acquire_implementation_task_claim",
                        lambda *_: pytest.fail("unknown drain admitted a claim"))
    result = intent(daemon, tmp_path / "intent")
    assert result[:4] == (False, "source_maintenance_drain_unverified", None, None)
    assert lock.read_bytes() == record


@pytest.mark.parametrize("record", [b"malformed", b"{}", b'{"pid":99999999,"lease_id":"old"}'])
def test_drain_never_reclaims_incumbent(native, record):
    repo, daemon, lock = native
    lock.write_bytes(record)
    before = lock.stat()
    with pytest.raises(RuntimeError, match="contended"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None):
            pytest.fail("incumbent was admitted")
    assert lock.read_bytes() == record
    assert lock.stat().st_ino == before.st_ino


@pytest.mark.parametrize("change", ["metadata", "replacement", "missing"])
def test_changed_drain_is_not_authority_and_is_not_removed(native, change):
    repo, daemon, lock = native
    with pytest.raises(RuntimeError, match="release binding changed"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None) as gate:
            if change == "missing":
                lock.unlink()  # Test-owned lease only.
            else:
                value = json.loads(lock.read_text())
                value["operation"] = "foreign"
                if change == "replacement":
                    replacement = lock.with_suffix(".replacement")
                    replacement.write_text(json.dumps(value))
                    replacement.replace(lock)
                else:
                    lock.write_text(json.dumps(value))
            with pytest.raises(RuntimeError, match="lease changed"):
                gate()
    if change != "missing":
        assert json.loads(lock.read_text())["operation"] == "foreign"


def test_admission_failure_before_publish_and_during_lease(native):
    repo, daemon, lock = native
    allowed = False

    def admit():
        if not allowed:
            raise RuntimeError("native custody lost")

    with pytest.raises(RuntimeError, match="custody lost"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=admit):
            pytest.fail("unadmitted")
    assert not lock.exists()
    allowed = True
    with pytest.raises(RuntimeError, match="custody lost"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=admit) as gate:
            allowed = False
            gate()
    assert not lock.exists()


def test_namespace_isolation(native, tmp_path, monkeypatch):
    repo, daemon, _ = native
    monkeypatch.setattr(daemon, "_try_acquire_implementation_task_claim",
                        lambda *_: (True, "acquired", None))
    with dispatch_drain_lease(repo, "other-board", admission_gate=lambda: None):
        assert intent(daemon, tmp_path / "intent")[0] is True


def test_owner_birth_change_cannot_use_or_release_lease(native, monkeypatch):
    repo, daemon, lock = native
    with pytest.raises(RuntimeError, match="owner birth changed"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None) as gate:
            birth = drain.read_process_birth(os.getpid())
            monkeypatch.setattr(drain, "read_process_birth",
                                lambda _: replace(birth, start_time_ticks=birth.start_time_ticks + 1))
            with pytest.raises(RuntimeError, match="owner birth changed"):
                gate()
    assert lock.exists()


def test_exact_metadata_release_rechecks_under_native_update_gate(native, monkeypatch):
    repo, daemon, lock = native
    original = drain.release_checkout_mutation_lease

    def changed_before_release(lease, **kwargs):
        value = json.loads(lock.read_text())
        value["operation"] = "changed-at-release-boundary"
        lock.write_text(json.dumps(value))
        return original(lease, **kwargs)

    monkeypatch.setattr(drain, "release_checkout_mutation_lease", changed_before_release)
    with pytest.raises(RuntimeError, match="release unverified"):
        with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None):
            pass
    assert json.loads(lock.read_text())["operation"] == "changed-at-release-boundary"


def test_drain_publication_serializes_with_real_dispatch_gate(native):
    repo, daemon, lock = native
    code = (
        "import sys; from pathlib import Path; "
        "from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import serialized_lock_update\n"
        "with serialized_lock_update(Path(sys.argv[1])):\n"
        " print('held', flush=True)\n"
        " sys.stdin.readline()\n"
    )
    child = subprocess.Popen([sys.executable, "-P", "-c", code, str(lock)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(Path(drain.__file__).resolve().parents[3])})
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError):
            with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None,
                                      timeout_seconds=0.05):
                pytest.fail("drain bypassed active dispatch gate")
        assert not lock.exists()
    finally:
        child.stdin.close()
        child.wait(timeout=10)
    with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None) as gate:
        gate()


@pytest.mark.parametrize("inspection", ["drain", "unknown", "error"])
def test_real_run_returns_nonconsuming_deferral_before_provider(native, tmp_path, monkeypatch, inspection):
    repo, _, _ = native
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "todo.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=repo, implement=True, implementation_command="must-not-run",
        use_ephemeral_worktree=True, worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
    )
    task = PortalTask(task_id="T-1", title="test", status="todo", completion="auto", priority="P1", track="implementation")
    monkeypatch.setattr(daemon, "_board_task_is_completed", lambda *_: False)
    monkeypatch.setattr(daemon, "_find_live_inflight_implementation", lambda: None)
    monkeypatch.setattr(daemon, "_require_plan_runtime_before_claim", lambda *_a, **_k: None)
    monkeypatch.setattr(daemon, "_retry_no_change_pre_dispatch_scope", lambda *_: None)
    monkeypatch.setattr(daemon, "_active_provider_capacity_backoff_for_task", lambda *_: {})
    for method in ("_try_acquire_implementation_task_claim", "_acquire_implementation_resource_claims",
                   "_require_primary_provider_readiness", "_run_implementation_in_ephemeral_worktree"):
        monkeypatch.setattr(daemon, method, lambda *_a, **_k: pytest.fail("drain reached provider/claim"))
    with dispatch_drain_lease(repo, daemon.board_namespace, admission_gate=lambda: None):
        if inspection == "unknown":
            monkeypatch.setattr(daemon, "_active_protected_path_maintenance_claim_serialized",
                                lambda *_: {"coordination_error": "unreadable"})
        elif inspection == "error":
            def unreadable(*_):
                raise PermissionError("fixture permission refusal")
            monkeypatch.setattr(daemon, "_active_protected_path_maintenance_claim_serialized", unreadable)
        result = daemon._run_implementation(task, PortalTaskState())
    assert result["reason"] == {
        "drain": "implementation_source_maintenance_drain_active",
        "unknown": "implementation_source_maintenance_drain_unverified",
        "error": "implementation_maintenance_coordination_failed",
    }[inspection]
    assert result["deferral_schema"] == PORTAL_RETRY_DEFERRAL_SCHEMA
    assert result["attempt_consumed"] is False
    assert result["provider_dispatched"] is False
    assert result["dispatch_intent_created"] is False
