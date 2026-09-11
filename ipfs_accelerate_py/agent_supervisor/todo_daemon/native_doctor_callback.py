"""A bounded native callback whose effect scope is declared before task claim.

This adapter accepts exact local Doctor edits, never arbitrary executable code,
provider invocations or ref mutations. Its closed observation is evidence for a
separate interruption protocol, not task completion or callback settlement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..runtime import doctor_worktree_adapter as doctor
from ..merge.worktree_lifecycle import WorktreeLifecycleStore

PROFILE_KEY = "native_doctor_callback_profile"
PROFILE_SCHEMA = "ipfs_accelerate_py/native-doctor-callback-profile@1"
STARTED = "native_doctor_callback_started"
CLOSED = "native_doctor_callback_closed"


class DoctorCallbackDenied(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = (
        "todo_daemon/native_doctor_callback.py",
        "todo_daemon/doctor_interruption_custody.py",
        "todo_daemon/unresolved_interruption.py",
        "todo_daemon/implementation_daemon.py",
        "runtime/doctor_worktree_adapter.py",
        "merge/database_coordination.py",
        "merge/unresolved_interruption_barrier.py",
        "merge/worktree_lifecycle.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


class NativeDoctorCallback:
    """A fixed local edit plan; binding it after an attempt starts is refused."""

    def __init__(
        self,
        *,
        repository_root: Path,
        state_root: Path,
        edits: Sequence[doctor.DoctorExactEdit],
        max_attempts: int = 3,
    ) -> None:
        if type(max_attempts) is not int or not 1 <= max_attempts <= 1024:
            raise DoctorCallbackDenied("native Doctor attempt budget must be between 1 and 1024")
        self.max_attempts = max_attempts
        values = tuple(edits)
        if not values or any(type(item) is not doctor.DoctorExactEdit for item in values):
            raise DoctorCallbackDenied("exact native Doctor edits are required")
        self.edits = values
        self.adapter = doctor.DoctorWorktreeAdapter(
            repository_root=repository_root,
            state_root=state_root,
            permitted_paths=tuple(item.path for item in values),
            permitted_refs=(),
        )
        self._sources = _source_hashes()
        self._git_hash = hashlib.sha256(Path(self.adapter.git_executable).read_bytes()).hexdigest()
        self._daemon: Any = None

    def profile(self) -> dict[str, Any]:
        if (
            type(self.adapter) is not doctor.DoctorWorktreeAdapter
            or self.adapter.fault_injector is not None
            or self.adapter.permitted_refs
            or _source_hashes() != self._sources
            or hashlib.sha256(Path(self.adapter.git_executable).read_bytes()).hexdigest()
            != self._git_hash
        ):
            raise DoctorCallbackDenied("native callback executor binding changed")
        value = {
            "schema": PROFILE_SCHEMA,
            "repository_root": str(self.adapter.repository_root),
            "state_root": str(self.adapter.state_root),
            "base_commit": self.adapter._git_text(
                self.adapter.repository_root, "rev-parse", "--verify", "HEAD^{commit}",
            ),
            "allowed_paths": list(self.adapter.permitted_paths),
            "permitted_refs": [],
            "max_attempts": self.max_attempts,
            "source_sha256": dict(self._sources),
            "git_sha256": self._git_hash,
            "edits": [
                {
                    "path": item.path,
                    "before_hash": item.before_hash,
                    "after_sha256": hashlib.sha256(item.after_bytes).hexdigest(),
                    "after_bytes": len(item.after_bytes),
                    "mode": item.mode,
                    "expected_after_hash": item.expected_after_hash,
                    "step_id": item.step_id,
                    "group_id": item.group_id,
                }
                for item in self.edits
            ],
        }
        return {**value, "profile_id": identity(value)}

    def bind(self, daemon: Any) -> dict[str, Any]:
        if self._daemon is not None and self._daemon is not daemon:
            raise DoctorCallbackDenied("callback is already bound to another daemon")
        self._daemon = daemon
        return self.profile()

    def __call__(self, attempt: Any) -> Mapping[str, Any]:
        daemon = self._daemon
        if daemon is None or dict(attempt.body.get(PROFILE_KEY) or {}) != self.profile():
            raise DoctorCallbackDenied("callback was not admitted with this attempt")
        daemon._protect_attempt_write(attempt)
        if attempt.committed_phase != "context":
            raise DoctorCallbackDenied("native Doctor callback requires exact context phase")
        rows = daemon._require_connection().execute(
            "SELECT event_type FROM daemon_execution_events WHERE attempt_id=? "
            "AND event_type IN (?, ?)", [attempt.attempt_id, STARTED, CLOSED],
        ).fetchall()
        if rows:
            raise DoctorCallbackDenied("native callback already started; recovery is required")
        session_id = "attempt-" + hashlib.sha256(attempt.attempt_id.encode()).hexdigest()[:32]
        base_commit = str(attempt.body[PROFILE_KEY]["base_commit"])
        lifecycle = WorktreeLifecycleStore(repo_root=self.adapter.repository_root)
        captured = lifecycle.begin_preparing(
            task_id=attempt.task_alias or attempt.task_cid,
            canonical_task_cid=attempt.task_cid,
            attempt=attempt.attempt_number,
            lane_id=daemon.owner_session_id,
            workspace_path=self.adapter.state_root / "sessions" / session_id / "worktree",
            branch=base_commit,
            merge_target=base_commit,
            state_dir=str(self.adapter.state_root),
        )
        session = self.adapter.prepare(base_ref=base_commit, session_id=session_id)
        try:
            started = {
                "profile": dict(attempt.body[PROFILE_KEY]),
                "attempt": attempt.to_dict(),
                "session_id": session_id,
                "worktree_path": str(session.worktree_root),
                "session_dir": str(session.session_dir),
                "baseline": session.baseline.to_dict(),
                "lifecycle": captured.to_dict(),
            }
            started["observation_id"] = identity(started)
        except BaseException:
            session.close(remove_worktree=False)
            raise
        try:
            daemon._protect_attempt_write(attempt)
            daemon._record_event(
                STARTED, attempt_id=attempt.attempt_id, task_cid=attempt.task_cid,
                body=started,
            )
            session.write_intent()
            receipt = session.apply_group(self.edits, group_id="native-callback")
            observed = self.adapter.snapshot(session)
            # This evidence write records already completed bounded operations.
            # It grants no task mutation or settlement authority after lease loss.
            closed = {
                "started_observation_id": started["observation_id"],
                "profile_id": started["profile"]["profile_id"],
                "session_id": session_id,
                "observed_candidate": observed.to_dict(),
                "apply_receipt": receipt.to_dict(),
                "callback_outcome": "unknown",
                "settlement_authority": False,
                "completion_authority": False,
            }
            closed["observation_id"] = identity(closed)
            daemon._record_event(
                CLOSED, attempt_id=attempt.attempt_id, task_cid=attempt.task_cid,
                body=closed,
            )
            return {
                "status": "native_doctor_candidate_prepared", "accepted": True,
                "candidate_observation_id": closed["observation_id"],
                "completion_authority": False,
            }
        finally:
            session.close(remove_worktree=False)


def claim_profile(daemon: Any) -> dict[str, Any] | None:
    callback = daemon._provider_fn
    return callback.bind(daemon) if type(callback) is NativeDoctorCallback else None


def require_declared_callback(daemon: Any, attempt: Any, callback: Any) -> None:
    profile = attempt.body.get(PROFILE_KEY)
    if profile is None:
        if type(callback) is NativeDoctorCallback:
            raise DoctorCallbackDenied("native recovery profile cannot be attached retrospectively")
        return
    if (
        type(callback) is not NativeDoctorCallback
        or callback is not daemon._provider_fn
        or callback.bind(daemon) != profile
    ):
        raise DoctorCallbackDenied("native recovery attempt cannot dispatch another callback")
    configured = daemon.max_task_attempts
    if type(configured) is not int or configured < 0:
        raise DoctorCallbackDenied("native callback attempt budget is invalid")
    if attempt.attempt_number > min(callback.max_attempts, configured or callback.max_attempts):
        raise DoctorCallbackDenied("native callback spent attempt budget is exhausted")
