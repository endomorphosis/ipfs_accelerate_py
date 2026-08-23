"""Fail-closed tests for database Portal protected-path auto-recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA,
    DatabasePortalAttemptPaths,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "state": "present",
        "kind": "regular_file",
        "mode": metadata.st_mode,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "links": metadata.st_nlink,
        "sha256": _digest(path),
    }


def _attempt() -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:protected-path",
        claim_id="claim:protected-path",
        task_cid="task:protected-path",
        task_alias="PCAR-TEST-001",
        attempt_number=3,
        owner_session_id="session:protected-path",
        fencing_token=7,
        fence_epoch=2,
        lease_id="lease:protected-path",
        committed_phase="claimed",
        status="running",
        started_at_ms=1,
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:protected-path",
        task_alias="PCAR-TEST-001",
        goal_cid="goal:protected-path",
        plan_cid="plan:protected-path",
        revision=4,
        priority="P0",
        dependencies=(),
        outputs=({"path": "inventory/result.json"},),
        validations=({"argv": ["python", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "focused validation passes"},),
        body={
            "objective": "Recover one exact cleanup race",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "focused validation passes",
        },
    )


class _TaskSource:
    def __init__(self, record: SimpleNamespace) -> None:
        self.record = record

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(repository_tree_id="tree:protected-path")

    def get_task(self, task_cid: str) -> SimpleNamespace | None:
        return self.record if task_cid == self.record.task_cid else None


class _ExactWorkspaceDisposalReconciler:
    def __init__(self, paths: object) -> None:
        self.paths = paths
        self.closed = False

    def _reconcile_implementation_protected_path_fence(
        self,
        *,
        protected_path_recovery_guard: object,
        protected_path_recovery_io: object,
    ) -> dict[str, object]:
        intent = json.loads(
            (
                self.paths.root
                / "database-portal-protected-path-recovery-intent.json"
            ).read_text(encoding="utf-8")
        )
        assert isinstance(protected_path_recovery_guard, dict)
        assert isinstance(protected_path_recovery_io, dict)
        assert protected_path_recovery_guard["clearance_id"] == intent["clearance_id"]
        incident_path = self.paths.root / "implementation-protected-path-incident.json"
        active_path = self.paths.root / "implementation-protected-path-active.json"
        incident = json.loads(incident_path.read_text(encoding="utf-8"))
        clearance_id = intent["clearance_id"]
        receipt = {
            "schema": "implementation-protected-path-auto-clearance-v1",
            "clearance_id": clearance_id,
            "cleared_at": "2026-08-21T00:00:01Z",
            "reason": "ephemeral_workspace_protected_deletions_shared_intact",
            "task_id": intent["task_alias"],
            "attempt": intent["portal_attempt"],
            "workspace_path": intent["workspace_path"],
            "mutated_paths": intent["mutated_paths"],
            "scopes": ["workspace"],
            "changes": ["deleted"],
            "class_codes": ["workspace_protected_deletion"],
            "shared_protected_paths_present": intent["mutated_paths"],
            "incident_latched_at": incident["latched_at"],
        }
        receipt_path = self.paths.root / (
            "implementation-protected-path-auto-clearance-"
            f"{clearance_id.removeprefix('sha256:')[:16]}.json"
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        incident_path.unlink()
        active_path.unlink()
        result = {
            "cleared": True,
            "auto": True,
            "blocked": False,
            "reason": receipt["reason"],
            "clearance_id": clearance_id,
            "receipt_path": str(receipt_path),
            "task_id": intent["task_alias"],
            "attempt": intent["portal_attempt"],
            "workspace_path": intent["workspace_path"],
            "mutated_paths": intent["mutated_paths"],
            "class_codes": ["workspace_protected_deletion"],
        }
        append_jsonl_event(
            self.paths.events,
            "implementation_protected_path_incident_auto_cleared",
            result,
        )
        return result

    def close_event_runtime(self) -> None:
        self.closed = True


def _recovery_fixture(
    tmp_path: Path,
) -> tuple[
    DatabasePortalExecutionBridge,
    DatabaseTaskAttempt,
    SimpleNamespace,
    object,
    Path,
    list[_ExactWorkspaceDisposalReconciler],
]:
    repository = tmp_path / "repository"
    protected = repository / "docs" / "protected.md"
    protected.parent.mkdir(parents=True)
    protected.write_text("sealed\n", encoding="utf-8")
    worktree_root = repository / ".managed-worktrees"
    worktree_root.mkdir()
    record = _record()
    task_source = _TaskSource(record)
    created: list[_ExactWorkspaceDisposalReconciler] = []

    def factory(paths: object, _alias: str) -> _ExactWorkspaceDisposalReconciler:
        portal = _ExactWorkspaceDisposalReconciler(paths)
        created.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository,
        worktree_root=worktree_root,
        implementation_protected_paths=("docs/protected.md",),
        max_passes=1,
    )
    attempt = _attempt()
    paths, _binding = bridge._ensure_attempt_projection(attempt, record)
    record.revision += 1
    identity = _identity(protected)
    workspace = worktree_root / "disposed"
    active = {
        "schema": "implementation-protected-path-active-v1",
        "task_id": record.task_alias,
        "attempt": 1,
        "workspace_path": str(workspace.absolute()),
        "ephemeral_worktree": True,
        "protected_paths": ["docs/protected.md"],
        "snapshot": {
            "workspace": {
                "root": str(workspace.absolute()),
                "paths": {"docs/protected.md": identity},
            },
            "shared_checkout": {
                "root": str(repository.resolve()),
                "paths": {"docs/protected.md": identity},
            },
        },
    }
    mutation = {
        "scope": "workspace",
        "path": "docs/protected.md",
        "change": "deleted",
        "before": identity,
        "after": {"state": "missing"},
    }
    incident = {
        "schema": "implementation-protected-path-incident-v1",
        "reason": "implementation_protected_path_mutated",
        "requires_operator_clearance": True,
        "shared_checkout_restored": False,
        "task_id": record.task_alias,
        "attempt": 1,
        "workspace_path": str(workspace.absolute()),
        "protected_paths": ["docs/protected.md"],
        "mutations": [mutation],
        "latched_at": "2026-08-21T00:00:00Z",
    }
    (paths.root / "implementation-protected-path-active.json").write_text(
        json.dumps(active, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (paths.root / "implementation-protected-path-incident.json").write_text(
        json.dumps(incident, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_jsonl_event(
        paths.events,
        "implementation_protected_path_mutated",
        {
            "task_id": record.task_alias,
            "attempt": 1,
            "workspace_path": str(workspace.absolute()),
            "mutations": [mutation],
        },
    )
    return bridge, attempt, record, paths, protected, created


def _bind_real_portal_factory(bridge: DatabasePortalExecutionBridge) -> None:
    def factory(
        paths: DatabasePortalAttemptPaths,
        _alias: str,
    ) -> PortalImplementationDaemon:
        return PortalImplementationDaemon(
            todo_path=paths.task_projection,
            state_path=paths.state,
            strategy_path=paths.strategy,
            events_path=paths.events,
            implementation_log_dir=paths.implementation_logs,
            repo_root=bridge.repository_root,
            worktree_root=bridge.worktree_root,
            implement=True,
            implementation_command="must-not-run",
            implementation_protected_paths=("docs/protected.md",),
        )

    bridge.portal_factory = factory


def test_recovers_only_exact_disposed_workspace_and_replays(
    tmp_path: Path,
) -> None:
    bridge, attempt, _record_value, paths, _protected, created = _recovery_fixture(
        tmp_path
    )

    receipt = bridge.recover_protected_path_retry(attempt)

    assert receipt["schema"] == DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA
    assert receipt["disposition"] == "retry"
    assert receipt["reason"] == "ephemeral_workspace_protected_deletions_recovered"
    assert receipt["attempt_consumed"] is True
    assert receipt["class_codes"] == ["workspace_protected_deletion"]
    assert not (paths.root / "implementation-protected-path-incident.json").exists()
    assert not (paths.root / "implementation-protected-path-active.json").exists()
    assert created[0].closed is True
    assert bridge.recover_protected_path_retry(attempt) == receipt

    # Crash replay after clearance but before the final bridge receipt remains
    # deterministic because the durable intent precedes fence clearance.
    (paths.root / "database-portal-protected-path-recovery.json").unlink()
    assert bridge.recover_protected_path_retry(attempt) == receipt


def test_real_portal_reconciler_produces_bridge_verifiable_clearance(
    tmp_path: Path,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )

    _bind_real_portal_factory(bridge)

    receipt = bridge.recover_protected_path_retry(attempt)

    assert receipt["schema"] == DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA
    assert receipt["workspace_path"].endswith("/.managed-worktrees/disposed")
    assert not (paths.root / "implementation-protected-path-incident.json").exists()
    assert not (paths.root / "implementation-protected-path-active.json").exists()


def test_replays_crash_between_incident_and_active_fence_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    original_unlink = os.unlink
    injected = False

    def fail_once_for_active(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if (
            path == active_path.name
            and kwargs.get("dir_fd") is not None
            and not injected
        ):
            injected = True
            raise OSError("injected crash after incident removal")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_once_for_active)
    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)
    monkeypatch.setattr(os, "unlink", original_unlink)

    assert not incident_path.exists()
    assert active_path.is_file()
    receipt = bridge.recover_protected_path_retry(attempt)
    assert receipt["schema"] == DATABASE_PORTAL_PROTECTED_PATH_RECOVERY_SCHEMA
    assert not active_path.exists()
    clearance_events = [
        json.loads(line)
        for line in paths.events.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("type")
        == "implementation_protected_path_incident_auto_cleared"
    ]
    assert len(clearance_events) == 1


def test_active_only_replay_rejects_tampered_clearance_without_unlinking_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    original_unlink = os.unlink
    injected = False

    def fail_once_for_active(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if (
            path == active_path.name
            and kwargs.get("dir_fd") is not None
            and not injected
        ):
            injected = True
            raise OSError("injected crash after incident removal")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_once_for_active)
    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)
    monkeypatch.setattr(os, "unlink", original_unlink)
    assert not incident_path.exists()
    assert active_path.is_file()

    clearance_path = next(
        paths.root.glob("implementation-protected-path-auto-clearance-*.json")
    )
    clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
    clearance["incident_latched_at"] = "2026-08-21T00:00:09Z"
    clearance_path.write_text(
        json.dumps(clearance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert active_path.is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


def test_shared_path_change_before_leased_clear_keeps_fence_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    original_builder = PortalImplementationDaemon._build_protected_path_auto_clear_guard
    calls = 0

    def race_before_revalidation(
        daemon: PortalImplementationDaemon,
        **kwargs: object,
    ) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            protected.write_text("raced shared mutation\n", encoding="utf-8")
        return original_builder(daemon, **kwargs)

    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_build_protected_path_auto_clear_guard",
        race_before_revalidation,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert calls == 2
    assert (paths.root / "implementation-protected-path-incident.json").is_file()
    assert (paths.root / "implementation-protected-path-active.json").is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


@pytest.mark.parametrize(
    "tamper",
    ["shared_content", "shared_mutation", "missing_snapshot", "output_overlap"],
)
def test_ambiguous_or_genuine_mutation_remains_latched(
    tmp_path: Path,
    tamper: str,
) -> None:
    bridge, attempt, record, paths, protected, _created = _recovery_fixture(tmp_path)
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    incident_before = incident_path.read_bytes()

    if tamper == "shared_content":
        protected.write_text("committed but untrusted\n", encoding="utf-8")
    elif tamper == "shared_mutation":
        incident = json.loads(incident_path.read_text(encoding="utf-8"))
        incident["mutations"][0]["scope"] = "shared_checkout"
        incident_path.write_text(json.dumps(incident), encoding="utf-8")
        incident_before = incident_path.read_bytes()
    elif tamper == "missing_snapshot":
        active_path.unlink()
    else:
        record.outputs = ({"path": "docs"},)

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert incident_path.read_bytes() == incident_before
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_linked_protected_content_remains_latched(
    tmp_path: Path,
    link_kind: str,
) -> None:
    bridge, attempt, _record_value, paths, protected, _created = _recovery_fixture(
        tmp_path
    )
    replacement = protected.with_name("replacement.md")
    replacement.write_bytes(protected.read_bytes())
    protected.unlink()
    if link_kind == "symlink":
        protected.symlink_to(replacement.name)
    else:
        protected.hardlink_to(replacement)

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert (paths.root / "implementation-protected-path-incident.json").is_file()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_linked_attempt_event_stream_cannot_escape_state_boundary(
    tmp_path: Path,
    link_kind: str,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(paths.events.read_bytes())
    outside_before = outside.read_bytes()
    paths.events.unlink()
    if link_kind == "symlink":
        paths.events.symlink_to(outside)
    else:
        paths.events.hardlink_to(outside)

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert outside.read_bytes() == outside_before
    assert (paths.root / "implementation-protected-path-incident.json").is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


def test_event_stream_symlink_swap_after_last_static_scan_keeps_fences_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(paths.events.read_bytes())
    outside_before = outside.read_bytes()
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    original_verify = bridge._verify_protected_path_attempt_boundary
    scans = 0

    def swap_after_last_static_scan(
        observed_paths: DatabasePortalAttemptPaths,
    ) -> None:
        nonlocal scans
        original_verify(observed_paths)
        scans += 1
        if scans == 2:
            observed_paths.events.unlink()
            observed_paths.events.symlink_to(outside)

    monkeypatch.setattr(
        bridge,
        "_verify_protected_path_attempt_boundary",
        swap_after_last_static_scan,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert scans == 2
    assert outside.read_bytes() == outside_before
    assert incident_path.is_file()
    assert active_path.is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


def test_event_stream_symlink_swap_after_capability_binding_keeps_fences_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(paths.events.read_bytes())
    outside_before = outside.read_bytes()
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    original_builder = PortalImplementationDaemon._build_protected_path_auto_clear_guard
    calls = 0

    def swap_before_under_lease_append(
        daemon: PortalImplementationDaemon,
        **kwargs: object,
    ) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            paths.events.unlink()
            paths.events.symlink_to(outside)
        return original_builder(daemon, **kwargs)

    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_build_protected_path_auto_clear_guard",
        swap_before_under_lease_append,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert calls == 2
    assert outside.read_bytes() == outside_before
    assert incident_path.is_file()
    assert active_path.is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


def test_fence_symlink_swap_after_capability_binding_keeps_fences_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    _bind_real_portal_factory(bridge)
    incident_path = paths.root / "implementation-protected-path-incident.json"
    active_path = paths.root / "implementation-protected-path-active.json"
    outside = tmp_path / "outside-incident.json"
    outside.write_bytes(incident_path.read_bytes())
    outside_before = outside.read_bytes()
    original_builder = PortalImplementationDaemon._build_protected_path_auto_clear_guard
    calls = 0

    def swap_incident_before_clear(
        daemon: PortalImplementationDaemon,
        **kwargs: object,
    ) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            incident_path.unlink()
            incident_path.symlink_to(outside)
        return original_builder(daemon, **kwargs)

    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_build_protected_path_auto_clear_guard",
        swap_incident_before_clear,
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert calls == 2
    assert outside.read_bytes() == outside_before
    assert incident_path.exists()
    assert active_path.is_file()
    assert not (paths.root / "database-portal-protected-path-recovery.json").exists()


def test_linked_attempt_directory_cannot_redirect_recovery_writes(
    tmp_path: Path,
) -> None:
    bridge, attempt, _record_value, paths, _protected, _created = _recovery_fixture(
        tmp_path
    )
    moved = paths.root.with_name(paths.root.name + "-moved")
    paths.root.rename(moved)
    paths.root.symlink_to(moved, target_is_directory=True)

    with pytest.raises(DatabasePortalBridgeError):
        bridge.recover_protected_path_retry(attempt)

    assert (moved / "implementation-protected-path-incident.json").is_file()
    assert not (moved / "database-portal-protected-path-recovery.json").exists()
