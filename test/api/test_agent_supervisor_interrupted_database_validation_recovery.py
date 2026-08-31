from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    acquire_checkout_mutation_lease,
    checkout_lock_metadata,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor as portal_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


def _task() -> PortalTask:
    return PortalTask(
        task_id="PCTDD-031",
        title="resume exact controller validation",
        status="todo",
        completion="auto",
        priority="P1",
        track="implementation",
        validation=["python -m pytest -q focused.py"],
    )


def _evidence() -> dict[str, Any]:
    return {
        "evidence_id": "sha256:" + "a" * 64,
        "reconciliation_receipt": {
            "nested_state": {
                "active_task_id": "PCTDD-031",
                "active_attempt": 1,
                "active_branch": "implementation/pctdd-031-attempt-1",
            }
        },
    }


def _daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_mode: str,
) -> tuple[PortalImplementationDaemon, list[str]]:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "portal-state.json"
    PortalTaskState().save(daemon.state_path)
    task = _task()
    workspace = tmp_path / "isolated-worktree"
    workspace.mkdir()
    calls: list[str] = []
    lease = object()

    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, _payload: calls.append(f"event:{event_type}"),
    )
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events", lambda: [])

    def acquire(**_kwargs: Any) -> tuple[object, str, None, float]:
        calls.append("lease:acquire")
        return lease, "acquired", None, 0.0

    def release(observed: object) -> bool:
        assert observed is lease
        calls.append("lease:release")
        return True

    def authority(
        _evidence_value: dict[str, Any],
        *,
        state: PortalTaskState,
    ) -> dict[str, Any]:
        assert not state.implementation_in_progress
        assert calls[-1] == "lease:acquire"
        calls.append("authority")
        return {
            "ok": True,
            "task": task,
            "workspace_path": workspace,
        }

    candidate = {
        "prepared": True,
        "task": task,
        "task_id": task.task_id,
        "candidate_branch": "rescue/worktree/pctdd-031",
        "baseline_ref": "1" * 40,
        "candidate_commit": "2" * 40,
        "candidate_authority_id": "baguqeera-candidate",
        "changed_submodule_paths": ["external/ipfs_accelerate"],
        "acceptance_inferred": False,
        "provider_dispatched": False,
        "attempt_consumed": False,
    }

    def prepare(
        _authority: dict[str, Any],
        *,
        state: PortalTaskState,
        checkout_lease: object,
    ) -> dict[str, Any]:
        assert not state.implementation_in_progress
        assert checkout_lease is lease
        calls.append("candidate:prepare")
        return dict(candidate)

    def release_claim(
        _state: PortalTaskState,
        *,
        unfinished_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert unfinished_candidate == candidate
        assert calls[-1] == "candidate:prepare"
        calls.append("claim:release")
        return {"reconciled": True, "blocked": False}

    replacement_claim = {"lease_id": "replacement-lease"}

    def build_claim(
        _task_value: PortalTask,
        _attempt: int,
        _started_at: str,
    ) -> dict[str, Any]:
        calls.append("claim:build-replacement")
        return dict(replacement_claim)

    def acquire_claim(
        _path: Path,
        metadata: dict[str, Any],
    ) -> tuple[bool, str, None]:
        assert metadata == replacement_claim
        calls.append("claim:acquire-replacement")
        return True, "acquired", None

    def validate(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["worktree_path"] == workspace
        assert kwargs["task"] is task
        assert kwargs["changed_submodule_paths"] == [
            "external/ipfs_accelerate"
        ]
        assert kwargs["preacquired_task_claim"] == replacement_claim
        calls.append("controller:validate")
        merged = validation_mode == "merged"
        queued = validation_mode == "queued"
        return {
            "returncode": 0 if merged else 1,
            "provider_dispatched": False,
            "attempt_consumed": False,
            "merge_result": {
                "merged": merged,
                "queued": queued,
                "reason": (
                    "merged" if merged else "queued" if queued else "not_attempted"
                ),
            },
            "validation_result": {
                "attempted": True,
                "passed": merged or queued,
            },
        }

    monkeypatch.setattr(daemon, "_acquire_checkout_mutation_lease", acquire)
    monkeypatch.setattr(daemon, "_release_checkout_mutation_lease", release)
    monkeypatch.setattr(
        daemon, "_interrupted_database_validation_authority", authority
    )
    monkeypatch.setattr(
        daemon, "_prepare_interrupted_database_validation_candidate", prepare
    )
    monkeypatch.setattr(
        daemon, "_reconcile_quiesced_implementation_task_claim", release_claim
    )
    monkeypatch.setattr(
        daemon, "_build_implementation_task_claim_metadata", build_claim
    )
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task_value: SimpleNamespace(
            canonical_task_cid="baguqeera-task"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_task_claim_path",
        lambda _task_id, canonical_task_cid: (
            tmp_path / f"{canonical_task_cid}.claim"
        ),
    )
    monkeypatch.setattr(
        daemon, "_try_acquire_implementation_task_claim", acquire_claim
    )
    monkeypatch.setattr(
        daemon, "reconcile_validated_worktree_candidate", validate
    )
    return daemon, calls


def test_interrupted_database_validation_resumes_existing_controller_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="merged")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["provider_dispatched"] is False
    assert result["attempt_consumed"] is False
    assert calls == [
        "lease:acquire",
        "authority",
        "candidate:prepare",
        "claim:release",
        "claim:build-replacement",
        "claim:acquire-replacement",
        "lease:release",
        "controller:validate",
        "event:interrupted_database_validation_reconciled",
    ]


def test_interrupted_database_validation_failure_is_typed_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="failed")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "interrupted_database_validation_rejected"
    assert result["controller_validation_terminal"] is True
    assert result["durable_merge_handoff"] is False
    assert result["provider_dispatched"] is False
    assert result["attempt_consumed"] is False
    assert "controller:validate" in calls


def test_interrupted_database_validation_accepts_durable_queue_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="queued")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "interrupted_database_validation_queued"
    assert result["controller_validation_terminal"] is False
    assert result["durable_merge_handoff"] is True
    assert calls.index("claim:acquire-replacement") < calls.index(
        "lease:release"
    )


def test_interrupted_database_validation_replays_exact_queued_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="queued")
    queued_validation = {
        "returncode": 1,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "merge_result": {
            "merged": False,
            "queued": True,
            "reason": "queued",
        },
        "validation_result": {"attempted": True, "passed": True},
    }
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [
            {
                "type": "interrupted_database_validation_reconciled",
                "event_id": "sha256:" + "b" * 64,
                "reason": "interrupted_database_validation_queued",
                "task_id": "PCTDD-031",
                "canonical_task_cid": "baguqeera-task",
                "attempt": 1,
                "database_evidence_id": "sha256:" + "a" * 64,
                "candidate_authority_id": "baguqeera-candidate",
                "candidate_commit": "2" * 40,
                "provider_dispatched": False,
                "attempt_consumed": False,
                "durable_merge_handoff": True,
                "controller_validation": queued_validation,
            }
        ],
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [_task()])

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["replayed"] is True
    assert result["durable_merge_handoff"] is True
    assert "lease:acquire" not in calls
    assert "controller:validate" not in calls


def test_database_recovery_identity_rejects_non_json_values() -> None:
    assert PortalImplementationDaemon._database_recovery_sha256(
        {"path": Path("not-json")}
    ) == ""
    assert PortalImplementationDaemon._database_recovery_sha256(
        {"number": float("nan")}
    ) == ""


def test_historical_projection_normalizes_group_writable_regular_mode() -> None:
    identity = {
        "state": "present",
        "kind": "regular_file",
        "device": 1,
        "inode": 2,
        "mode": 0o100664,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
        "sha256": "a" * 64,
    }

    assert (
        PortalImplementationDaemon._interrupted_validation_protected_identity_projection(
            identity
        )
        == {
            "state": "present",
            "kind": "regular_file",
            "mode": "100644",
            "size": 3,
            "sha256": "a" * 64,
        }
    )
    assert (
        PortalImplementationDaemon._interrupted_validation_protected_identity_projection(
            {**identity, "mode": 0o100775}
        )["mode"]
        == "100755"
    )


def test_clearance_authority_loader_rejects_final_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    target = owned / "target.json"
    target.write_text('{"exact":true}', encoding="utf-8")
    final_link = owned / "final.json"
    final_link.symlink_to(target)

    assert (
        PortalImplementationDaemon._load_interrupted_validation_clearance_authority_file(
            final_link
        )
        == (None, "invalid")
    )

    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(owned, target_is_directory=True)
    assert (
        PortalImplementationDaemon._load_interrupted_validation_clearance_authority_file(
            parent_link / "target.json"
        )
        == (None, "invalid")
    )


def test_clearance_removal_rejects_hardlink_and_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"schema": "exact-test@1", "value": 1}
    hardlinked = tmp_path / "hardlinked.json"
    hardlinked.write_text(json.dumps(record), encoding="utf-8")
    sibling = tmp_path / "sibling.json"
    sibling.hardlink_to(hardlinked)
    assert not (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            hardlinked,
            record,
        )
    )
    assert hardlinked.is_file() and sibling.is_file()

    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(record), encoding="utf-8")
    displaced = tmp_path / "displaced.json"
    original_stat = portal_supervisor.os.stat
    named_reads = 0

    def swapping_stat(
        path: str | bytes | int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal named_reads
        if (
            path == selected.name
            and kwargs.get("dir_fd") is not None
            and kwargs.get("follow_symlinks") is False
        ):
            named_reads += 1
            if named_reads == 1:
                selected.replace(displaced)
                replacement.replace(selected)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(portal_supervisor.os, "stat", swapping_stat)
    assert not (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            selected,
            record,
        )
    )
    assert selected.is_file() and displaced.is_file()


def test_clearance_removal_restores_pinned_barrier_after_rename_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"schema": "exact-test@1", "value": 1}
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps({"schema": "hostile-test@1", "value": 2}),
        encoding="utf-8",
    )
    displaced = tmp_path / "displaced.json"
    original_rename = portal_supervisor.os.rename
    swapped = False

    def swapping_rename(
        source: str | bytes,
        destination: str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        if source == selected.name and not swapped:
            swapped = True
            selected.replace(displaced)
            replacement.replace(selected)
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(portal_supervisor.os, "rename", swapping_rename)
    assert not (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            selected,
            record,
        )
    )
    assert swapped
    assert json.loads(selected.read_text(encoding="utf-8")) == record
    assert json.loads(displaced.read_text(encoding="utf-8")) == record
    quarantine = next(
        tmp_path.glob(".selected.json.clearance-quarantine-*")
    )
    assert (quarantine / "pinned-authority-record").samefile(selected)
    assert json.loads(
        (quarantine / "retired-authority-record").read_text(encoding="utf-8")
    ) == {"schema": "hostile-test@1", "value": 2}


def test_clearance_removal_retires_exact_inode_without_pathname_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"schema": "exact-test@1", "value": 1}
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    original_unlink = portal_supervisor.os.unlink

    def rejecting_selected_unlink(
        path: str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == selected.name:
            raise AssertionError("authority pathname must not be unlinked")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(portal_supervisor.os, "unlink", rejecting_selected_unlink)
    assert (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            selected,
            record,
        )
    )
    assert not selected.exists()
    quarantine = next(
        tmp_path.glob(".selected.json.clearance-quarantine-*")
    )
    pinned = quarantine / "pinned-authority-record"
    retired = quarantine / "retired-authority-record"
    assert pinned.samefile(retired)
    assert json.loads(pinned.read_text(encoding="utf-8")) == record
    # Direct replay proves the deterministic quarantine is a durable terminal
    # phase even when the process died before returning to the caller.
    assert (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            selected,
            record,
        )
    )


def test_clearance_removal_resumes_after_pinned_barrier_crash(
    tmp_path: Path,
) -> None:
    record = {"schema": "exact-test@1", "value": 1}
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    canonical = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    quarantine = tmp_path / (
        ".selected.json.clearance-quarantine-"
        + hashlib.sha256(canonical).hexdigest()
    )
    quarantine.mkdir(mode=0o700)
    (quarantine / "pinned-authority-record").hardlink_to(selected)

    assert (
        PortalImplementationDaemon._remove_interrupted_validation_clearance_authority_file(
            selected,
            record,
        )
    )
    assert not selected.exists()
    assert (quarantine / "pinned-authority-record").samefile(
        quarantine / "retired-authority-record"
    )


def test_shared_transition_history_is_bounded_at_128_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.implementation_protected_paths = ("protected.txt",)
    identity = {
        "state": "present",
        "kind": "regular_file",
        "device": 1,
        "inode": 2,
        "mode": 0o100664,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
        "sha256": "a" * 64,
    }
    commits = "\n".join(f"{index:040x}" for index in range(1, 130))
    monkeypatch.setattr(
        daemon,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=commits,
        ),
    )
    snapshot = {
        "workspace": {"root": str(tmp_path)},
        "shared_checkout": {
            "root": str(tmp_path),
            "git_head": "f" * 40,
            "paths": {"protected.txt": identity},
        },
    }

    assert not daemon._interrupted_validation_shared_protected_transition_authority(
        event_paths={"protected.txt": identity},
        current_snapshot=snapshot,
    )


def test_shared_transition_maps_real_git_state_through_trusted_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-b", "main")
    protected = repo / "protected.txt"
    protected.write_text("before\n", encoding="utf-8")
    protected.chmod(0o664)
    git("add", "protected.txt")
    git(
        "-c",
        "user.name=Initial",
        "-c",
        "user.email=initial@example.invalid",
        "commit",
        "-m",
        "initial protected state",
    )
    before_head = git("rev-parse", "HEAD")

    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = repo
    daemon.implementation_protected_paths = ("protected.txt",)
    before_identity = daemon._implementation_protected_path_identity(
        repo,
        "protected.txt",
    )
    assert before_identity["mode"] == 0o100664

    protected.write_text("after\n", encoding="utf-8")
    protected.chmod(0o664)
    git("add", "protected.txt")
    git(
        "-c",
        "user.name=Implementation Daemon",
        "-c",
        "user.email=implementation-daemon@example.invalid",
        "commit",
        "-m",
        "PCTDD-000: mark todo completed",
    )
    after_head = git("rev-parse", "HEAD")
    current_identity = daemon._implementation_protected_path_identity(
        repo,
        "protected.txt",
    )
    workspace = tmp_path / "isolated"
    workspace.mkdir()
    current_snapshot = {
        "workspace": {"root": str(workspace)},
        "shared_checkout": {
            "root": str(repo.resolve()),
            "git_head": after_head,
            "paths": {"protected.txt": current_identity},
        },
    }

    authority = (
        daemon._interrupted_validation_shared_protected_transition_authority(
            event_paths={"protected.txt": before_identity},
            current_snapshot=current_snapshot,
        )
    )

    assert authority["event_state_commit"] == before_head
    assert authority["current_commit"] == after_head
    assert authority["event_protected_paths"]["protected.txt"]["mode"] == (
        "100644"
    )
    assert authority["trusted_transition"]["commits"] == [
        {
            "commit": after_head,
            "author_email": "implementation-daemon@example.invalid",
            "subject": "PCTDD-000: mark todo completed",
        }
    ]


def test_recovery_checkout_lease_requires_exact_current_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path / "repo"
    daemon.repo_root.mkdir()
    daemon.state_path = tmp_path / "state" / "portal-state.json"
    daemon.state_path.parent.mkdir()
    lock_path = tmp_path / "checkout.lock"
    monkeypatch.setattr(daemon, "_repo_merge_lock_path", lambda: lock_path)
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=daemon.repo_root,
        task_id="PCTDD-031",
        attempt=1,
        branch="implementation/pctdd-031-attempt-1",
        extra={
            "operation": "interrupted_database_validation_recovery",
            "state_dir": str(daemon.state_path.parent.resolve()),
            "state_path": str(daemon.state_path.resolve()),
            "database_evidence_id": "sha256:" + "a" * 64,
        },
    )
    lease, reason, _existing, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _metadata: True,
    )
    assert reason == "acquired" and lease is not None
    authority = {
        "task": _task(),
        "workspace_path": tmp_path / "worktree",
        "attempt": 1,
        "original_branch": "implementation/pctdd-031-attempt-1",
        "database_evidence_id": "sha256:" + "a" * 64,
    }

    exact = daemon._interrupted_database_validation_checkout_lease_authority(
        lease,
        authority,
        operation="interrupted_database_validation_recovery",
    )
    assert exact["current"] is True

    lock_path.write_text(
        json.dumps({**metadata, "operation": "other-operation"}),
        encoding="utf-8",
    )
    replaced = daemon._interrupted_database_validation_checkout_lease_authority(
        lease,
        authority,
        operation="interrupted_database_validation_recovery",
    )
    assert replaced["current"] is False
    assert replaced["reason"] == "lease_identity_or_metadata_changed"


def test_candidate_preparation_reuses_held_lease_and_ignores_head_only_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path / "repo"
    daemon.repo_root.mkdir()
    daemon.worktree_root = tmp_path / "worktrees"
    workspace = daemon.worktree_root / "isolated"
    workspace.mkdir(parents=True)
    daemon.state_path = tmp_path / "state" / "portal-state.json"
    daemon.state_path.parent.mkdir()
    PortalTaskState().save(daemon.state_path)
    daemon.implementation_protected_paths = ("protected.txt",)
    before_head = "1" * 40
    after_head = "2" * 40
    path_identity = {"state": "file", "sha256": "3" * 64}
    before = {
        "workspace": {
            "root": str(workspace.resolve()),
            "git_head": before_head,
            "paths": {"protected.txt": path_identity},
        },
        "shared_checkout": {
            "root": str(daemon.repo_root.resolve()),
            "git_head": before_head,
            "paths": {"protected.txt": path_identity},
        },
    }
    after = {
        "workspace": {**before["workspace"], "git_head": after_head},
        "shared_checkout": dict(before["shared_checkout"]),
    }
    snapshots = iter((before, after, after))
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_snapshot",
        lambda _workspace: next(snapshots),
    )
    monkeypatch.setattr(
        daemon,
        "_interrupted_database_validation_checkout_lease_authority",
        lambda *_args, **_kwargs: {"current": True, "reason": "exact"},
    )
    monkeypatch.setattr(
        daemon,
        "_commit_worktree_changes",
        lambda *_args, **_kwargs: {
            "commit": after_head,
            "submodule_results": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_acquire_implementation_protected_verification_lock",
        lambda **_kwargs: pytest.fail("held checkout lease was reacquired"),
    )
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task_value: SimpleNamespace(
            canonical_task_key="task/v1/exact",
            canonical_task_cid="baguqeera-task",
            board_namespace="exact-board",
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _workspace, ref: before_head if ref == before_head else after_head,
    )
    monkeypatch.setattr(daemon, "_git_current_branch", lambda _workspace: "rescue")
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        daemon,
        "_committed_submodule_paths",
        lambda _results: [],
    )
    monkeypatch.setattr(daemon, "_record_event", lambda *_args: None)
    monkeypatch.setattr(
        daemon,
        "_mark_implementation_finished",
        lambda state, *, finished_at: setattr(
            state,
            "implementation_in_progress",
            False,
        ),
    )
    authority = {
        "task": _task(),
        "task_id": "PCTDD-031",
        "workspace_path": workspace,
        "attempt": 1,
        "current_branch": "rescue",
        "original_branch": "implementation/pctdd-031-attempt-1",
        "baseline_ref": before_head,
        "preparation_head": before_head,
        "database_evidence_id": "sha256:" + "a" * 64,
    }

    result = daemon._prepare_interrupted_database_validation_candidate(
        authority,
        state=PortalTaskState.load(daemon.state_path),
        checkout_lease=object(),  # helper above supplies exact lease authority
    )

    assert result["prepared"] is True
    assert result["candidate_commit"] == after_head


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        ("none", "interrupted_validation_self_deadlock_recovered"),
        ("missing_active", "interrupted_validation_self_deadlock_recovered"),
        ("event", "self_deadlock_incident_event_pair_invalid"),
        ("incident_ambiguity", "self_deadlock_incident_event_pair_invalid"),
        ("incident_symlink", "self_deadlock_incident_record_invalid"),
        ("active_extra", "self_deadlock_active_snapshot_invalid"),
        ("timeout", "self_deadlock_event_pair_missing"),
        ("receipt", "self_deadlock_clearance_receipt_changed"),
        ("remove", "self_deadlock_clearance_removal_failed"),
        (
            "path",
            "self_deadlock_protected_snapshot_changed_or_unstable",
        ),
        (
            "source",
            "self_deadlock_protected_snapshot_changed_or_unstable",
        ),
        (
            "branch",
            "self_deadlock_protected_snapshot_changed_or_unstable",
        ),
        ("lease", "self_deadlock_clearance_inputs_changed"),
    ),
)
def test_self_deadlock_incident_recovery_is_exact_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_reason: str,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path / "repo"
    daemon.repo_root.mkdir()
    daemon.worktree_root = tmp_path / "worktrees"
    workspace = daemon.worktree_root / "isolated"
    workspace.mkdir(parents=True)
    daemon.state_path = tmp_path / "state" / "portal-state.json"
    daemon.state_path.parent.mkdir()
    daemon.implementation_protected_paths = ("protected.txt",)
    before_head = "1" * 40
    after_head = "2" * 40
    identity = {
        "state": "present",
        "kind": "regular_file",
        "device": 1,
        "inode": 2,
        "mode": 0o100644,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
        "sha256": "3" * 64,
    }
    before_snapshot = {
        "workspace": {
            "root": str(workspace.resolve()),
            "git_head": before_head,
            "paths": {"protected.txt": identity},
        },
        "shared_checkout": {
            "root": str(daemon.repo_root.resolve()),
            "git_head": "4" * 40,
            "paths": {"protected.txt": identity},
        },
    }
    incident = {
        "schema": "implementation-protected-path-incident-v1",
        "reason": "implementation_protected_path_mutated",
        "task_id": "PCTDD-031",
        "attempt": 1,
        "workspace_path": str(workspace),
        "protected_paths": ["protected.txt"],
        "mutations": [
            {
                "scope": "workspace",
                "path": "",
                "change": "scope_snapshot_changed",
                "before": {
                    "root": str(workspace.resolve()),
                    "git_head": before_head,
                },
                "after": {
                    "root": str(workspace.resolve()),
                    "git_head": after_head,
                },
            }
        ],
        "shared_checkout_restored": False,
        "requires_operator_clearance": True,
        "latched_at": "2026-08-31T00:00:00Z",
    }
    active = {
        "schema": "implementation-protected-path-active-v1",
        "recorded_at": "2026-08-31T00:00:00Z",
        "task_id": "PCTDD-031",
        "attempt": 1,
        "workspace_path": str(workspace.resolve()),
        "ephemeral_worktree": True,
        "protected_paths": ["protected.txt"],
        "snapshot": before_snapshot,
    }
    if fault == "active_extra":
        active["private_extension"] = "must-not-enter-clearance"
    incident_path = daemon._implementation_protected_incident_path()
    active_path = daemon._implementation_protected_active_snapshot_path()
    incident_path.write_text(json.dumps(incident), encoding="utf-8")
    if fault == "incident_symlink":
        incident_target = tmp_path / "incident-target.json"
        incident_path.replace(incident_target)
        incident_path.symlink_to(incident_target)
    if fault != "missing_active":
        active_path.write_text(json.dumps(active), encoding="utf-8")
    lock = {
        "acquired": False,
        "reason": "lock_exists",
        "lock_path": str(tmp_path / "merge.lock"),
        "waited_seconds": 0.0 if fault == "timeout" else 30.0,
        "lock_owner_pid": 1234,
        "lock_owner_task_id": "PCTDD-031",
        "lock_owner_branch": "implementation/pctdd-031-attempt-1",
        "lock_owner_operation": "interrupted_database_validation_recovery",
    }
    timeout_payload = {
        "reason": "implementation_protected_path_verification_lock_timeout",
        "task_id": "PCTDD-031",
        "attempt": 1,
        "workspace_path": str(workspace),
        "protected_paths": ["protected.txt"],
        "mutations": [
            {
                "scope": "shared_checkout",
                "path": "protected.txt",
                "change": "verification_inconclusive",
                "before": identity,
                "after": {
                    "state": "error",
                    "error": "implementation_protected_path_verification_lock_timeout",
                },
            }
        ],
        "shared_checkout_restored": False,
        "verification_deferred": True,
        "lock": lock,
    }
    recovery_event = {
        "type": "implementation_shutdown_reconciliation_blocked",
        "reason": "task_claim_reconciliation_blocked",
        "task_id": "PCTDD-031",
        "attempt": 1,
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "sequence": 8,
        "previous_event_id": "sha256:" + "4" * 64,
        "event_id": "sha256:" + "5" * 64,
    }
    incident_payload = {
        key: incident[key]
        for key in (
            "reason",
            "task_id",
            "attempt",
            "workspace_path",
            "protected_paths",
            "mutations",
            "shared_checkout_restored",
        )
    }
    incident_event = {
        "type": "implementation_protected_path_mutated",
        **incident_payload,
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "sequence": 9,
        "previous_event_id": recovery_event["event_id"],
        "event_id": "sha256:" + "6" * 64,
    }
    incident_blocked_event = {
        "type": "interrupted_database_validation_recovery_blocked",
        "reason": "recovery_protected_path_mutated",
        "reconciled": False,
        "blocked": True,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "candidate": {
            "prepared": False,
            "reason": "recovery_protected_path_mutated",
            "protected_path_violation": incident_payload,
        },
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "sequence": 10,
        "previous_event_id": (
            "sha256:" + ("0" if fault == "event" else "6") * 64
        ),
        "event_id": "sha256:" + "7" * 64,
    }
    timeout_event = {
        "type": "implementation_protected_path_verification_lock_timeout",
        **timeout_payload,
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "sequence": 11,
        "previous_event_id": incident_blocked_event["event_id"],
        "event_id": "sha256:" + "8" * 64,
    }
    blocked_event = {
        "type": "interrupted_database_validation_recovery_blocked",
        "reason": "recovery_protected_path_mutated",
        "reconciled": False,
        "blocked": True,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "candidate": {
            "prepared": False,
            "reason": "recovery_protected_path_mutated",
            "protected_path_violation": timeout_payload,
        },
        "stream_id": "stream-1",
        "snapshot_id": "snapshot-1",
        "sequence": 12,
        "previous_event_id": timeout_event["event_id"],
        "event_id": "sha256:" + "9" * 64,
    }
    lifecycle_events = [
        recovery_event,
        incident_event,
        incident_blocked_event,
        timeout_event,
        blocked_event,
    ]
    if fault == "incident_ambiguity":
        duplicate_incident = {
            **incident_event,
            "sequence": 13,
            "previous_event_id": blocked_event["event_id"],
            "event_id": "sha256:" + "b" * 64,
        }
        duplicate_blocked = {
            **incident_blocked_event,
            "sequence": 14,
            "previous_event_id": duplicate_incident["event_id"],
            "event_id": "sha256:" + "c" * 64,
        }
        lifecycle_events.extend((duplicate_incident, duplicate_blocked))
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: lifecycle_events,
    )
    monkeypatch.setattr(daemon, "_repo_merge_lock_path", lambda: tmp_path / "merge.lock")
    monkeypatch.setattr(daemon, "_is_git_worktree", lambda _path: True)
    monkeypatch.setattr(daemon, "_git_current_branch", lambda _path: "rescue")
    monkeypatch.setattr(daemon, "_resolved_commit_ref", lambda *_args: after_head)
    monkeypatch.setattr(
        daemon,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda *_args: True,
    )
    candidate_authority = {
        "schema": "candidate-test@1",
        "branch": "rescue",
        "candidate_commit": after_head,
    }
    candidate_checks = 0

    def current_candidate_authority(**_kwargs: Any) -> dict[str, Any]:
        nonlocal candidate_checks
        candidate_checks += 1
        if fault == "branch" and candidate_checks > 1:
            return {}
        return dict(candidate_authority)

    monkeypatch.setattr(
        daemon,
        "_interrupted_validation_candidate_git_authority",
        current_candidate_authority,
    )
    monkeypatch.setattr(
        daemon,
        "_interrupted_validation_control_source_authority",
        lambda: (
            {}
            if fault == "source"
            else {"schema": "control-source-test@1", "clean": True}
        ),
    )
    historical_git_authority = {
        "schema": "protected-git-test@1",
        "before_commit": before_head,
        "after_commit": after_head,
    }
    monkeypatch.setattr(
        daemon,
        "_interrupted_validation_historical_protected_git_authority",
        lambda **_kwargs: dict(historical_git_authority),
    )
    def shared_transition_authority(**kwargs: Any) -> dict[str, Any]:
        # The transition authority must receive the exact filesystem identity,
        # not its lossy Git-observable projection. It owns that conversion and
        # rejects projections presented as source authority.
        assert kwargs["event_paths"] == {"protected.txt": identity}
        return {
            "schema": "shared-protected-transition-test@1",
            "trusted": True,
        }

    monkeypatch.setattr(
        daemon,
        "_interrupted_validation_shared_protected_transition_authority",
        shared_transition_authority,
    )
    monkeypatch.setattr(
        daemon,
        "_active_protected_path_maintenance_claim",
        lambda: None,
    )
    lease = object()
    monkeypatch.setattr(
        daemon,
        "_acquire_checkout_mutation_lease",
        lambda **_kwargs: (lease, "acquired", None, 0.0),
    )
    lease_checks = 0

    def lease_authority(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal lease_checks
        lease_checks += 1
        return {
            "current": not (fault == "lease" and lease_checks == 1),
            "reason": "test",
        }

    monkeypatch.setattr(
        daemon,
        "_interrupted_database_validation_checkout_lease_authority",
        lease_authority,
    )
    monkeypatch.setattr(daemon, "_release_checkout_mutation_lease", lambda _lease: True)
    current_identity = (
        {**identity, "sha256": "8" * 64} if fault == "path" else identity
    )
    current_snapshot = {
        "workspace": {
            "root": str(workspace.resolve()),
            "git_head": after_head,
            "paths": {"protected.txt": current_identity},
        },
        "shared_checkout": {
            "root": str(daemon.repo_root.resolve()),
            "git_head": "9" * 40,
            "paths": {"protected.txt": identity},
        },
    }
    monkeypatch.setattr(
        daemon,
        "_implementation_protected_path_snapshot",
        lambda _workspace: current_snapshot,
    )
    monkeypatch.setattr(
        daemon,
        "_clear_implementation_protected_snapshot",
        lambda **_kwargs: active_path.unlink(missing_ok=True),
    )
    monkeypatch.setattr(daemon, "_record_event", lambda *_args: None)
    authority = {
        "ok": True,
        "task": _task(),
        "task_id": "PCTDD-031",
        "attempt": 1,
        "workspace_path": workspace,
        "original_branch": "implementation/pctdd-031-attempt-1",
        "current_branch": "rescue",
        "baseline_ref": "0" * 40,
        "preparation_head": after_head,
        "recovery_event_id": recovery_event["event_id"],
        "database_evidence_id": "sha256:" + "a" * 64,
    }
    if fault == "receipt":
        clearance_path = (
            daemon._interrupted_validation_self_deadlock_clearance_path(
                database_evidence_id="sha256:" + "a" * 64,
                task_id="PCTDD-031",
                attempt=1,
            )
        )
        assert clearance_path is not None
        differing_body = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "interrupted-validation-self-deadlock-clearance@1"
            ),
            "phase": "clearance_authorized",
            "task_id": "PCTDD-031",
            "attempt": 1,
            "workspace_path": str(workspace),
            "database_evidence_id": "sha256:" + "a" * 64,
            "incident": incident,
            "active_snapshot_state": "exact",
            "active_snapshot": active,
            "historical_protected_authority": None,
            "incident_event_id": incident_event["event_id"],
            "blocked_event_id": incident_blocked_event["event_id"],
            "baseline_ref": "0" * 40,
            "candidate_commit": "f" * 40,
            "protected_snapshot": current_snapshot,
            "shared_protected_transition_authority": {
                "schema": "shared-protected-transition-test@1",
                "trusted": True,
            },
            "control_source_authority": {
                "schema": "control-source-test@1",
                "clean": True,
            },
            "intended_removals": [str(incident_path), str(active_path)],
        }
        differing = {
            **differing_body,
            "recovery_id": content_identity(differing_body),
        }
        clearance_path.write_text(
            json.dumps(differing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        differing_bytes = clearance_path.read_bytes()
    original_remove = (
        daemon._remove_interrupted_validation_clearance_authority_file
    )
    if fault == "remove":
        def fail_active_removal(
            selected: Path,
            expected: dict[str, Any] | None,
        ) -> bool:
            if selected == active_path:
                return False
            return original_remove(selected, expected)

        monkeypatch.setattr(
            daemon,
            "_remove_interrupted_validation_clearance_authority_file",
            fail_active_removal,
        )

    result = daemon._recover_interrupted_database_validation_self_deadlock_incident(
        evidence={"evidence_id": "sha256:" + "a" * 64},
        authority=authority,
        incident=incident,
    )

    assert result["reason"] == expected_reason
    successful = fault in {"none", "missing_active"}
    assert result["cleared"] is successful
    assert incident_path.exists() is (
        fault not in {"none", "missing_active", "remove"}
    )
    assert active_path.exists() is (fault not in {"none", "missing_active"})
    if successful:
        assert result["provider_dispatched"] is False
        receipt_path = Path(result["receipt_path"])
        assert receipt_path.is_file()
        immutable_bytes = receipt_path.read_bytes()
        receipt = json.loads(immutable_bytes)
        assert receipt["active_snapshot_state"] == (
            "absent" if fault == "missing_active" else "exact"
        )
        assert (receipt["active_snapshot"] is None) is (
            fault == "missing_active"
        )
        assert (
            receipt["historical_protected_authority"] is not None
        ) is (fault == "missing_active")

        replayed = (
            daemon._recover_interrupted_database_validation_self_deadlock_incident(
                evidence={"evidence_id": "sha256:" + "a" * 64},
                authority=authority,
                incident={},
            )
        )
        assert replayed["cleared"] is True
        assert replayed["recovery_id"] == result["recovery_id"]
        assert receipt_path.read_bytes() == immutable_bytes

    elif fault == "receipt":
        assert clearance_path.read_bytes() == differing_bytes
    elif fault == "remove":
        receipt_path = (
            daemon._interrupted_validation_self_deadlock_clearance_path(
                database_evidence_id="sha256:" + "a" * 64,
                task_id="PCTDD-031",
                attempt=1,
            )
        )
        assert receipt_path is not None and receipt_path.is_file()
        immutable_bytes = receipt_path.read_bytes()
        monkeypatch.setattr(
            daemon,
            "_remove_interrupted_validation_clearance_authority_file",
            original_remove,
        )
        resumed = (
            daemon._recover_interrupted_database_validation_self_deadlock_incident(
                evidence={"evidence_id": "sha256:" + "a" * 64},
                authority=authority,
                incident={},
            )
        )
        assert resumed["cleared"] is True, resumed
        assert not active_path.exists()
        assert receipt_path.read_bytes() == immutable_bytes


def test_interrupted_validation_authority_admits_only_exact_fenced_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "attempt" / "portal-state.json"
    daemon.state_path.parent.mkdir()
    daemon.worktree_root = tmp_path / "worktrees"
    workspace = daemon.worktree_root / "isolated"
    workspace.mkdir(parents=True)
    task = _task()
    identity = SimpleNamespace(
        canonical_task_key="task/v1/exact",
        canonical_task_cid="baguqeera-task",
        board_namespace="exact-board",
    )
    original_branch = "implementation/pctdd-031-attempt-1"
    baseline = "1" * 40
    current_head = "2" * 40
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_worktree_path=str(workspace),
    )
    record = SimpleNamespace(
        is_terminal=True,
        terminal_reason="controlled_restart_dead_owner",
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        branch=original_branch,
        workspace_path=str(workspace),
        record_id="lifecycle-record",
        fence=5,
    )
    daemon.worktree_lifecycle = SimpleNamespace(
        load_workspace=lambda _workspace: record
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(daemon, "_identity_for_task", lambda _task: identity)
    monkeypatch.setattr(daemon, "_canonical_ref", lambda _task: identity.canonical_task_cid)
    monkeypatch.setattr(daemon, "_is_git_worktree", lambda _path: True)
    monkeypatch.setattr(
        daemon,
        "_git_current_branch",
        lambda _path: "rescue/worktree/implementation-pctdd-031-attempt-1-a",
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _path, ref: baseline if ref == baseline else current_head,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda _path, ancestor, descendant: (
            ancestor == baseline and descendant == current_head
        ),
    )

    binding = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1",
        "interface": "DatabasePortalExecutionBridge@1",
        "attempt_id": "attempt-1",
        "claim_id": "claim-1",
        "task_cid": "outer-task-cid",
        "task_alias": task.task_id,
        "goal_cid": "goal",
        "plan_cid": "plan",
        "task_revision": 3,
        "fencing_token": 7,
        "fence_epoch": 2,
        "lease_id": "outer-lease",
        "task_body_digest": "sha256:" + "3" * 64,
        "projection_seed_digest": "sha256:" + "4" * 64,
        "projection_immutable_digest": "sha256:" + "5" * 64,
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
    }
    binding["binding_id"] = daemon._database_recovery_sha256(binding)
    portal = {
        "blocked": True,
        "reconciled": False,
        "reason": "task_claim_reconciliation_blocked",
        "attempt_recovery": {
            "canonical_task_cid": identity.canonical_task_cid
        },
        "protected_path_reconciliation": {
            "blocked": False,
            "reason": "crash_reconciliation_unchanged",
            "workspace_path": str(workspace),
        },
        "worktree_lifecycle_reconciliation": {
            "blocked": False,
            "reconciled": True,
            "state": "terminal",
            "workspace_path": str(workspace),
            "attempt": 1,
            "record_id": record.record_id,
            "fence": record.fence,
        },
        "task_claim_reconciliation": {
            "blocked": True,
            "reason": "canonical_task_not_terminal",
        },
    }
    nested = {
        "active": True,
        "active_phase": "validating",
        "active_task_id": task.task_id,
        "active_attempt": 1,
        "active_worktree_path": str(workspace),
        "active_branch": original_branch,
        "active_phase_detail": "; ".join(task.validation),
    }
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-reconciliation@1",
        "stage": "blocked",
        "binding_id": binding["binding_id"],
        "attempt_root": str(daemon.state_path.parent),
        "task_alias": task.task_id,
        "task_cid": binding["task_cid"],
        "attempt_id": binding["attempt_id"],
        "claim_id": binding["claim_id"],
        "fencing_token": binding["fencing_token"],
        "fence_epoch": binding["fence_epoch"],
        "nested_state": nested,
        "portal_reconciliation": portal,
        "provider_runner_fence": {
            "applicable": True,
            "fenced": True,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
        },
    }
    receipt["receipt_id"] = daemon._database_recovery_sha256(receipt)
    evidence = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-interrupted-validation-recovery@1",
        "binding": binding,
        "recovery_identity": "sha256:" + "6" * 64,
        "equivalent_receipt_ids": [receipt["receipt_id"]],
        "reconciliation_receipt": receipt,
    }
    evidence["evidence_id"] = daemon._database_recovery_sha256(evidence)
    start_event = {
        "type": "implementation_started",
        "event_id": "start-event",
        "sequence": 1,
        "task_id": task.task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "attempt": 1,
        "worktree_path": str(workspace),
        "branch": original_branch,
        "baseline_ref": baseline,
        "workspace_setup": {"base_commit": baseline},
    }
    recovery_event = {
        **portal,
        "type": "implementation_shutdown_reconciliation_blocked",
        "event_id": "recovery-event",
        "sequence": 2,
        "task_id": task.task_id,
        "attempt": 1,
    }
    harmless_outer_preflight = {
        "type": "worktree_reconciliation_validation_finished",
        "sequence": 3,
        "task_id": task.task_id,
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "returncode": 1,
        "worktree_path": str(workspace),
        "baseline_ref": baseline,
        "validation_result": {
            "attempted": False,
            "passed": False,
            "reason": "reconciliation_validation_exception",
            "error": "reconciled candidate worktree is not clean",
        },
        "merge_result": {"merged": False, "reason": "not_attempted"},
    }
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [start_event, recovery_event, harmless_outer_preflight],
    )

    authority = daemon._interrupted_database_validation_authority(
        evidence,
        state=state,
    )

    assert authority["ok"] is True
    assert authority["baseline_ref"] == baseline
    assert authority["database_evidence_id"] == evidence["evidence_id"]

    poisoned = dict(evidence)
    poisoned["recovery_identity"] = Path("not-json")
    assert daemon._interrupted_database_validation_authority(
        poisoned,
        state=state,
    )["reason"] == "database_recovery_evidence_identity_invalid"


def test_bridge_coalesces_duplicate_recovery_receipts_without_count_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = DatabasePortalExecutionBridge.__new__(
        DatabasePortalExecutionBridge
    )
    bridge.attempt_root = tmp_path / "attempts"
    attempt = SimpleNamespace(attempt_id="attempt-1")
    paths = bridge._paths(attempt)
    paths.reconciliation.mkdir(parents=True)
    binding = {"task_alias": "PCTDD-031", "binding_id": "binding-1"}
    workspace = str(tmp_path / "worktree")
    matching: dict[str, dict[str, Any]] = {}
    for index in range(132):
        receipt_id = f"sha256:{index:064x}"
        (paths.reconciliation / f"{index:064x}.json").touch()
        if index >= 130:
            matching[receipt_id] = {
                "stage": "blocked",
                "receipt_id": receipt_id,
                "blocked": True,
                "reconciled": False,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "binding_id": binding["binding_id"],
                "task_alias": binding["task_alias"],
                "terminal_provider_evidence": False,
                "provider_runner_reconciliation_authority": (
                    "ordinary_provider_runner_fence"
                ),
                "nested_state": {
                    "active": True,
                    "active_phase": "validating",
                    "active_task_id": binding["task_alias"],
                    "active_attempt": 1,
                    "active_worktree_path": workspace,
                    "active_branch": "implementation/pctdd-031-attempt-1",
                    "state_path": str(paths.state),
                    "state_digest": "sha256:" + "7" * 64,
                },
                "provider_runner_fence": {
                    "applicable": True,
                    "fenced": True,
                    "safe_to_restart": True,
                    "reason": "ordinary_provider_runner_exact_birth_fenced",
                },
                "portal_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "task_claim_reconciliation_blocked",
                    "protected_path_reconciliation": {
                        "blocked": False,
                        "reason": "crash_reconciliation_unchanged",
                        "task_id": binding["task_alias"],
                        "workspace_path": workspace,
                    },
                    "worktree_lifecycle_reconciliation": {
                        "blocked": False,
                        "reconciled": True,
                        "state": "terminal",
                        "task_id": binding["task_alias"],
                        "workspace_path": workspace,
                        "attempt": 1,
                        "record_id": "record-1",
                        "fence": 5,
                    },
                    "task_claim_reconciliation": {
                        "blocked": True,
                        "reconciled": False,
                        "reason": "canonical_task_not_terminal",
                        "task_id": binding["task_alias"],
                        "canonical_task_cid": "baguqeera-task",
                    },
                    "attempt_recovery": {
                        "consumed": False,
                        "attempt": 1,
                        "task_id": binding["task_alias"],
                    },
                },
            }
    prepared_id = "sha256:" + "f" * 64
    (paths.reconciliation / ("f" * 64 + ".json")).touch()

    def load(
        _attempt: Any,
        receipt_id: str,
        *,
        required_stage: str = "",
    ) -> dict[str, Any]:
        assert required_stage == ""
        if receipt_id == prepared_id:
            return {"stage": "prepared", "receipt_id": receipt_id}
        return matching.get(
            receipt_id,
            {
                "stage": "blocked",
                "receipt_id": receipt_id,
                "blocked": True,
                "reconciled": False,
                "reason": "unrelated",
            },
        )

    monkeypatch.setattr(bridge, "load_reconciliation_receipt", load)

    evidence = bridge._interrupted_validation_recovery_evidence(
        attempt,
        binding,
    )

    assert evidence is not None
    assert len(evidence["equivalent_receipt_ids"]) == 2
    assert evidence["reconciliation_receipt"]["receipt_id"] == min(
        matching
    )


def test_bridge_reconciles_more_than_128_exact_immutable_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = SimpleNamespace(
        attempt_id="attempt-1",
        claim_id="claim-1",
        task_cid="task-cid-1",
        task_alias="PCTDD-031",
        attempt_number=1,
        owner_session_id="owner-1",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease-1",
        body={},
    )
    received_evidence: list[dict[str, Any]] = []

    class ExactPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, Any]:
            raise AssertionError("exact interrupted-validation evidence was ignored")

        def reconcile_interrupted_database_validation_attempt(
            self,
            evidence: dict[str, Any],
        ) -> dict[str, Any]:
            received_evidence.append(evidence)
            return {"reconciled": True, "blocked": False}

        def close(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: ExactPortal(),
    )
    bridge.attempt_root.mkdir()
    paths = bridge._paths(attempt)
    paths.reconciliation.mkdir(parents=True)
    paths.binding.write_text("{}\n", encoding="utf-8")
    paths.task_projection.write_text("projection\n", encoding="utf-8")
    paths.state.write_text("{}\n", encoding="utf-8")
    state_digest = "sha256:" + "7" * 64
    workspace = str(tmp_path / "isolated-worktree")
    binding = {
        "attempt_id": attempt.attempt_id,
        "task_alias": attempt.task_alias,
        "binding_id": "sha256:" + "8" * 64,
        "projection_immutable_digest": "sha256:" + "9" * 64,
    }
    nested_state = {
        "present": True,
        "state_path": str(paths.state),
        "state_digest": state_digest,
        "active": True,
        "active_task_id": attempt.task_alias,
        "active_attempt": 1,
        "active_phase": "validating",
        "active_phase_detail": "python -m pytest -q focused.py",
        "active_worktree_path": workspace,
        "active_branch": "implementation/pctdd-031-attempt-1",
    }

    def receipt_payload(index: int, *, digest: str = state_digest) -> dict[str, Any]:
        return {
            "stage": "blocked",
            "trigger": f"duplicate-{index}",
            "reconciled_at": f"2026-08-31T00:00:{index:03d}Z",
            "reconciled": False,
            "blocked": True,
            "reason": "nested_portal_attempt_reconciliation_blocked",
            "binding_id": binding["binding_id"],
            "nested_state": {
                "active": True,
                "active_phase": "validating",
                "active_task_id": attempt.task_alias,
                "active_attempt": 1,
                "active_worktree_path": workspace,
                "active_branch": "implementation/pctdd-031-attempt-1",
                "state_path": str(paths.state),
                "state_digest": digest,
            },
            "provider_runner_fence": {
                "applicable": True,
                "fenced": True,
                "safe_to_restart": True,
                "reason": "ordinary_provider_runner_exact_birth_fenced",
            },
            "provider_runner_reconciliation_authority": (
                "ordinary_provider_runner_fence"
            ),
            "portal_reconciliation": {
                "blocked": True,
                "reconciled": False,
                "reason": "task_claim_reconciliation_blocked",
                "protected_path_reconciliation": {
                    "blocked": False,
                    "reason": "crash_reconciliation_unchanged",
                    "task_id": attempt.task_alias,
                    "workspace_path": workspace,
                },
                "worktree_lifecycle_reconciliation": {
                    "blocked": False,
                    "reconciled": True,
                    "state": "terminal",
                    "task_id": attempt.task_alias,
                    "workspace_path": workspace,
                    "attempt": 1,
                    "record_id": "record-1",
                    "fence": 5,
                },
                "task_claim_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "canonical_task_not_terminal",
                    "task_id": attempt.task_alias,
                    "canonical_task_cid": "baguqeera-task",
                },
                "attempt_recovery": {
                    "consumed": False,
                    "attempt": 1,
                    "task_id": attempt.task_alias,
                },
            },
            "terminal_provider_evidence": False,
        }

    for index in range(132):
        bridge.persist_reconciliation_receipt(
            attempt,
            receipt_payload(index),
        )

    monkeypatch.setattr(
        bridge,
        "_record_for_attempt",
        lambda _source, _attempt: object(),
    )
    monkeypatch.setattr(
        bridge,
        "_render_projection",
        lambda _attempt, _record: "projection\n",
    )
    monkeypatch.setattr(
        bridge,
        "_binding",
        lambda _attempt, _record, _seed: dict(binding),
    )
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _binding: None)
    monkeypatch.setattr(
        bridge,
        "_verify_projection",
        lambda _paths, _binding: "projection\n",
    )
    monkeypatch.setattr(
        bridge,
        "_projection_task_identity",
        lambda _paths, _binding, _projection: {
            "task_id": attempt.task_alias,
            "canonical_task_key": "task/v1/exact",
            "canonical_task_cid": "baguqeera-task",
            "board_namespace": "exact-board",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: ({"active_phase": "validating"}, state_digest),
    )
    monkeypatch.setattr(
        bridge,
        "_verify_nested_state_identity",
        lambda _paths, _binding, _identity, **_kwargs: dict(nested_state),
    )
    bridge._binding_lookup = lambda _attempt: {**binding, "stage": "portal_entered"}
    monkeypatch.setattr(bridge, "recover_provider_result", lambda _attempt: None)
    monkeypatch.setattr(
        portal_supervisor,
        "fence_ordinary_provider_runner",
        lambda _state, *, grace_seconds: {
            "applicable": True,
            "safe_to_restart": True,
            "fenced": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
        },
    )

    result = bridge.reconcile_quiesced_attempt(attempt)

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert len(received_evidence) == 1
    assert len(received_evidence[0]["equivalent_receipt_ids"]) == 132

    malformed = paths.reconciliation / "not-content-addressed.json"
    malformed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        DatabasePortalBridgeError,
        match="reconciliation evidence store is not exact",
    ):
        bridge.reconcile_quiesced_attempt(attempt)
    malformed.unlink()

    bridge.persist_reconciliation_receipt(
        attempt,
        receipt_payload(999, digest="sha256:" + "a" * 64),
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="interrupted validation evidence is ambiguous",
    ):
        bridge.reconcile_quiesced_attempt(attempt)
