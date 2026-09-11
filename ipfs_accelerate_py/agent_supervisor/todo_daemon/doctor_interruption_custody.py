"""Fresh native custody for a declared, positively closed Doctor callback.

JSON observations identify what to inspect. They cannot replace the actual
writer lock, native lifecycle checks, complete source/candidate/checkpoint
readback, or the canonical task and claim authority checked by the caller.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from typing import Any

from ..runtime import doctor_worktree_adapter as doctor
from ..merge.worktree_lifecycle import (
    WorktreeLifecycleStore, WorkspaceLifecycleRecord, current_process_birth,
)
from ..core.multiformats_identity import cid_for_dag_json
from .native_doctor_callback import (
    NativeDoctorCallback, DoctorCallbackDenied, PROFILE_KEY, STARTED, CLOSED,
    identity,
)


def observations(daemon: Any, attempt: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = daemon._require_connection().execute(
        "SELECT event_type,body_json FROM daemon_execution_events WHERE attempt_id=? "
        "AND event_type IN (?, ?) ORDER BY event_type", [attempt.attempt_id, STARTED, CLOSED],
    ).fetchall()
    if len(rows) != 2 or sorted(row[0] for row in rows) != sorted([STARTED, CLOSED]):
        raise DoctorCallbackDenied("one exact started and closed native callback is required")
    values = {}
    for row in rows:
        event_type, raw = row[0], row[1]
        value = json.loads(raw)
        if value.get("observation_id") != identity({
            key: item for key, item in value.items() if key != "observation_id"
        }):
            raise DoctorCallbackDenied("native callback observation identity changed")
        values[event_type] = value
    started, closed = values[STARTED], values[CLOSED]
    if (
        started.get("attempt") != attempt.to_dict()
        or started.get("profile") != attempt.body.get(PROFILE_KEY)
        or closed.get("started_observation_id") != started["observation_id"]
        or closed.get("profile_id") != started["profile"]["profile_id"]
        or closed.get("session_id") != started.get("session_id")
        or closed.get("callback_outcome") != "unknown"
        or closed.get("settlement_authority") is not False
        or closed.get("completion_authority") is not False
    ):
        raise DoctorCallbackDenied("native callback observations are crossed")
    return started, closed


class DoctorInterruptionCustody:
    def __init__(self, daemon: Any, attempt: Any, callback: NativeDoctorCallback):
        from .implementation_daemon import DatabaseImplementationDaemon
        if type(daemon) is not DatabaseImplementationDaemon or type(callback) is not NativeDoctorCallback:
            raise DoctorCallbackDenied("native Doctor producer and daemon are required")
        self.daemon = daemon
        self.callback = callback
        self.adapter = callback.adapter
        self.started, self.closed = observations(daemon, attempt)
        if callback.bind(daemon) != self.started["profile"]:
            raise DoctorCallbackDenied("native callback source/profile changed")
        self.attempt = attempt
        self.fd = -1
        lock_name = hashlib.sha256(str(self.adapter.repository_root).encode()).hexdigest()
        path = self.adapter.state_root / "locks" / (lock_name + ".lock")
        self.lock_path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
                raise DoctorCallbackDenied("native Doctor lock identity is invalid")
            self.lock_identity = (info.st_dev, info.st_ino)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.require_current()
        except BaseException:
            self.close()
            raise

    def require_current(self) -> dict[str, Any]:
        if self.fd < 0 or self.callback.profile() != self.started["profile"]:
            raise DoctorCallbackDenied("native Doctor custody is closed or source changed")
        named = self.lock_path.lstat()
        held = os.fstat(self.fd)
        if any(
            not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_nlink != 1 or (info.st_dev, info.st_ino) != self.lock_identity
            for info in (named, held)
        ):
            raise DoctorCallbackDenied("native Doctor lock path or held identity changed")
        sid = self.started["session_id"]
        directory = self.adapter.state_root / "sessions" / sid
        workspace = directory / "worktree"
        if str(directory) != self.started["session_dir"] or str(workspace) != self.started["worktree_path"]:
            raise DoctorCallbackDenied("native Doctor session path changed")
        raw = doctor._secure_file_bytes(directory / "intent.json", maximum=16 * 1024 * 1024)
        journal = json.loads(raw)
        if (
            raw != doctor._canonical_bytes(journal) + b"\n"
            or cid_for_dag_json(journal, for_identity=True)
            != self.closed["apply_receipt"]["durable_effect_ref"]
            or journal.get("baseline") != self.started["baseline"]
            or journal.get("session_id") != sid
            or journal.get("repository_root") != str(self.adapter.repository_root)
            or journal.get("worktree_root") != str(workspace)
            or journal.get("state") != doctor.DoctorWorktreeState.APPLYING.value
            or journal.get("target_ref") != ""
            or journal.get("desired_commit_oid") != ""
        ):
            raise DoctorCallbackDenied("native Doctor effect journal changed")
        baseline = self.started["baseline"]
        entries = doctor._parse_tree(self.adapter._git(
            self.adapter.repository_root, "ls-tree", "-rz", "-r", "--full-tree",
            self.started["profile"]["base_commit"],
        ).stdout)
        snapshot = doctor.DoctorWorktreeSnapshot(
            session_id=sid, worktree_root=str(workspace),
            base_commit_oid=baseline["base_commit_oid"], git_tree_oid=baseline["git_tree_oid"],
            tree_cid=baseline["tree_cid"], forest_cid=baseline["forest_cid"],
            blob_cids=tuple((row["path"], row["cid"]) for row in baseline["blob_cids"]),
            path_hashes=tuple((row["path"], row["sha256"]) for row in baseline["path_hashes"]),
            gitlinks=tuple(doctor.DoctorGitlink(row["path"], row["commit_oid"]) for row in baseline["gitlinks"]),
        )
        # The session is a read-only view under this class's retained real lock;
        # no restore, close, cleanup or apply method is invoked on it.
        session = doctor.DoctorWorktreeSession(
            adapter=self.adapter, session_id=sid, base_ref=journal["base_ref"],
            base_commit_oid=baseline["base_commit_oid"], worktree_root=workspace,
            session_dir=directory, lock_stream=None, baseline=snapshot, entries=entries,
            git_admin_hash=journal["git_admin_hash"], state=doctor.DoctorWorktreeState.APPLYING,
        )
        observed = self.adapter.snapshot(session).to_dict()
        if observed != self.closed["observed_candidate"]:
            raise DoctorCallbackDenied("native Doctor candidate changed")
        checkpoint = directory / "checkpoint"
        manifest_raw = doctor._secure_file_bytes(checkpoint / "manifest.json", maximum=16 * 1024 * 1024)
        manifest = json.loads(manifest_raw)
        hashes = dict(snapshot.path_hashes)
        records = manifest.get("entries")
        if (
            type(records) is not list
            or manifest.get("session_id") != sid
            or manifest.get("base_commit_oid") != snapshot.base_commit_oid
            or manifest.get("tree_cid") != snapshot.tree_cid
            or manifest.get("forest_cid") != snapshot.forest_cid
            or len(records) != len(hashes)
            or {row.get("path") for row in records} != set(hashes)
        ):
            raise DoctorCallbackDenied("native Doctor checkpoint population changed")
        expected_files = {"manifest.json"}
        for index, row in enumerate(sorted(records, key=lambda item: item["path"])):
            expected_name = f"{index:08d}.blob"
            if row.get("storage_name") != expected_name:
                raise DoctorCallbackDenied("native Doctor checkpoint storage binding changed")
            expected_files.add(expected_name)
            body = doctor._secure_file_bytes(checkpoint / expected_name, maximum=doctor._MAX_EDIT_BYTES)
            digest = "sha256:" + hashlib.sha256(body).hexdigest()
            if row.get("sha256") != digest or hashes[row["path"]] != digest:
                raise DoctorCallbackDenied("native Doctor checkpoint bytes changed")
        if {path.name for path in checkpoint.iterdir()} != expected_files:
            raise DoctorCallbackDenied("native Doctor checkpoint inventory changed")
        lifecycle = WorktreeLifecycleStore(repo_root=self.adapter.repository_root)
        captured = WorkspaceLifecycleRecord.from_dict(self.started["lifecycle"])
        if lifecycle.load_workspace(workspace) != captured:
            raise DoctorCallbackDenied("native Doctor workspace lifecycle changed")
        return {
            "started_observation_id": self.started["observation_id"],
            "closed_observation_id": self.closed["observation_id"],
            "candidate_id": identity(observed),
            "checkpoint_id": identity(manifest),
            "lifecycle_record_id": captured.record_id,
            "effect_scope": "native_exact_local_edits_without_ref_mutation",
            "callback_outcome": "unknown", "settlement_authority": False,
            "completion_authority": False,
        }

    def quarantine(self, authority: dict[str, Any]) -> dict[str, Any]:
        self.require_current()
        lifecycle = WorktreeLifecycleStore(repo_root=self.adapter.repository_root)
        captured = WorkspaceLifecycleRecord.from_dict(self.started["lifecycle"])
        if captured.owner == current_process_birth():
            return lifecycle.quarantine_current_owner(
                captured, fence_authority=authority, reason="native_unresolved_interruption",
            )
        return lifecycle.quarantine_exact_dead_owner(
            captured.workspace_path, expected_record_id=captured.record_id,
            expected_fence=captured.fence, expected_lease_id=captured.lease_id,
            expected_task_id=captured.task_id, expected_canonical_task_cid=captured.canonical_task_cid,
            expected_attempt=captured.attempt, expected_branch=captured.branch,
            expected_merge_target=captured.merge_target, expected_repo_root=captured.repo_root,
            expected_state_dir=captured.state_dir, fence_authority=authority,
            reason="native_unresolved_interruption",
        )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
