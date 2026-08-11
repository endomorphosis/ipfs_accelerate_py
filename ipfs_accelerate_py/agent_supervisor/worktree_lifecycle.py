"""Fenced cross-lane worktree ownership and cleanup eligibility.

Managed implementation worktrees are lifecycle resources, not inferences from
branch ancestry or a momentary process scan.  A lane must acquire a durable
task/workspace claim and monotonic fence *before* publishing a cleanup-visible
``git worktree``, and every cleanup path must compare-and-delete only terminal
or provably stale records.

This closes the six-lane race where a freshly created worktree whose branch tip
still matched the merge target was classified as already-merged before the
child process became discoverable, leaving the worker alive in an unlinked
checkout.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .checkout_lock import git_common_dir, serialized_lock_update
from .proof.formal_verification_contracts import content_identity


WORKTREE_LIFECYCLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/worktree-lifecycle-record@1"
)
WORKTREE_LIFECYCLE_DIRNAME = "agent-worktree-lifecycle"
FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID = (
    "asi-171:fenced-cross-lane-worktree-lifecycle"
)

DEFAULT_LEASE_SECONDS = 21_600.0
DEFAULT_STARTUP_GRACE_SECONDS = 120.0
DEFAULT_CLOCK: Callable[[], float] = time.time

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class WorkspaceLifecycleState(str, Enum):
    """Durable ownership phases for one managed worktree attempt."""

    PREPARING = "preparing"
    ACTIVE = "active"
    SETTLING = "settling"
    TERMINAL = "terminal"

    @property
    def is_terminal(self) -> bool:
        return self is WorkspaceLifecycleState.TERMINAL

    @property
    def is_nonterminal(self) -> bool:
        return not self.is_terminal


class OwnerLiveness(str, Enum):
    """Outcome of process-birth identity inspection."""

    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"  # missing /proc, IO error → fail closed


class CleanupDisposition(str, Enum):
    """Whether a cleanup caller may delete/prune/reuse a worktree."""

    ALLOW = "allow"
    DENY = "deny"
    RECLAIM_THEN_ALLOW = "reclaim_then_allow"


class LifecycleFailureKind(str, Enum):
    """Classify failures so internal races never spend retries/providers."""

    IMPLEMENTATION = "implementation"
    LIFECYCLE_RACE = "lifecycle_race"
    LIFECYCLE_SETUP = "lifecycle_setup"


class WorktreeLifecycleError(RuntimeError):
    """Base error for malformed lifecycle operations."""


class FenceMismatchError(WorktreeLifecycleError):
    """A CAS transition observed an unexpected fence token."""


class OwnershipError(WorktreeLifecycleError):
    """Caller does not own the current lifecycle lease/fence."""


class DuplicateAttemptError(WorktreeLifecycleError):
    """Another nonterminal claim already owns the task or workspace."""


@dataclass(frozen=True)
class ProcessBirthIdentity:
    """PID-reuse-resistant process identity for lifecycle owners."""

    pid: int
    start_time_ticks: int
    boot_id: str = ""
    parent_pid: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": int(self.pid),
            "start_time_ticks": int(self.start_time_ticks),
            "boot_id": str(self.boot_id or ""),
            "parent_pid": int(self.parent_pid or 0),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ProcessBirthIdentity":
        data = payload or {}
        return cls(
            pid=int(data.get("pid") or 0),
            start_time_ticks=int(data.get("start_time_ticks") or 0),
            boot_id=str(data.get("boot_id") or ""),
            parent_pid=int(data.get("parent_pid") or 0),
        )


@dataclass(frozen=True)
class CleanupDecision:
    """Authoritative cleanup eligibility for one workspace/branch."""

    disposition: CleanupDisposition
    reason: str
    record: "WorkspaceLifecycleRecord | None" = None
    failure_kind: LifecycleFailureKind = LifecycleFailureKind.IMPLEMENTATION
    provider_call_allowed: bool = True
    attempt_consumed: bool = True

    @property
    def allowed(self) -> bool:
        return self.disposition in {
            CleanupDisposition.ALLOW,
            CleanupDisposition.RECLAIM_THEN_ALLOW,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "failure_kind": self.failure_kind.value,
            "provider_call_allowed": self.provider_call_allowed,
            "attempt_consumed": self.attempt_consumed,
            "record": None if self.record is None else self.record.to_dict(),
        }


@dataclass(frozen=True)
class WorkspaceLifecycleRecord:
    """Durable fenced ownership record for one managed worktree attempt."""

    task_id: str
    canonical_task_cid: str
    attempt: int
    lane_id: str
    state: WorkspaceLifecycleState
    owner: ProcessBirthIdentity
    lease_id: str
    fence: int
    workspace_path: str
    branch: str
    merge_target: str
    created_at: float
    updated_at: float
    expires_at: float
    repo_root: str = ""
    state_dir: str = ""
    terminal_reason: str = ""
    record_id: str = ""
    schema: str = WORKTREE_LIFECYCLE_SCHEMA

    def __post_init__(self) -> None:
        if self.fence < 1:
            raise WorktreeLifecycleError("fence must be a positive integer")
        if self.attempt < 0:
            raise WorktreeLifecycleError("attempt must be non-negative")
        if not self.lease_id:
            raise WorktreeLifecycleError("lease_id is required")
        if not self.workspace_path:
            raise WorktreeLifecycleError("workspace_path is required")
        if not self.branch:
            raise WorktreeLifecycleError("branch is required")
        if not str(self.task_id or "").strip() and not str(
            self.canonical_task_cid or ""
        ).strip():
            raise WorktreeLifecycleError("task identity is required")
        if not self.record_id:
            object.__setattr__(self, "record_id", self.compute_record_id())

    @staticmethod
    def compute_record_id_for(
        *,
        canonical_task_cid: str,
        task_id: str,
        attempt: int,
        workspace_path: str,
    ) -> str:
        identity = {
            "kind": "worktree-lifecycle-record",
            "canonical_task_cid": str(canonical_task_cid or ""),
            "task_id": str(task_id or ""),
            "attempt": int(attempt),
            "workspace_path": str(workspace_path or ""),
        }
        return content_identity(identity)

    def compute_record_id(self) -> str:
        return self.compute_record_id_for(
            canonical_task_cid=self.canonical_task_cid,
            task_id=self.task_id,
            attempt=self.attempt,
            workspace_path=self.workspace_path,
        )

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def is_nonterminal(self) -> bool:
        return self.state.is_nonterminal

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "task_id": self.task_id,
            "canonical_task_cid": self.canonical_task_cid,
            "attempt": int(self.attempt),
            "lane_id": self.lane_id,
            "state": self.state.value,
            "owner": self.owner.to_dict(),
            "lease_id": self.lease_id,
            "fence": int(self.fence),
            "workspace_path": self.workspace_path,
            "branch": self.branch,
            "merge_target": self.merge_target,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "expires_at": float(self.expires_at),
            "repo_root": self.repo_root,
            "state_dir": self.state_dir,
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkspaceLifecycleRecord":
        schema = str(payload.get("schema") or WORKTREE_LIFECYCLE_SCHEMA)
        if schema not in (WORKTREE_LIFECYCLE_SCHEMA,):
            raise WorktreeLifecycleError(f"unsupported lifecycle schema: {schema}")
        state_raw = str(payload.get("state") or WorkspaceLifecycleState.PREPARING.value)
        try:
            state = WorkspaceLifecycleState(state_raw)
        except ValueError as exc:
            raise WorktreeLifecycleError(f"unknown lifecycle state: {state_raw}") from exc
        owner_payload = payload.get("owner")
        if not isinstance(owner_payload, Mapping):
            # Backward-compatible flat owner fields.
            owner_payload = {
                "pid": payload.get("owner_pid") or payload.get("pid") or 0,
                "start_time_ticks": payload.get("owner_start_time_ticks")
                or payload.get("start_time_ticks")
                or 0,
                "boot_id": payload.get("owner_boot_id") or payload.get("boot_id") or "",
                "parent_pid": payload.get("owner_parent_pid") or 0,
            }
        return cls(
            schema=schema,
            record_id=str(payload.get("record_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            canonical_task_cid=str(payload.get("canonical_task_cid") or ""),
            attempt=int(payload.get("attempt") or 0),
            lane_id=str(payload.get("lane_id") or ""),
            state=state,
            owner=ProcessBirthIdentity.from_dict(owner_payload),
            lease_id=str(payload.get("lease_id") or ""),
            fence=int(payload.get("fence") or 0),
            workspace_path=str(payload.get("workspace_path") or ""),
            branch=str(payload.get("branch") or ""),
            merge_target=str(payload.get("merge_target") or ""),
            created_at=float(payload.get("created_at") or 0.0),
            updated_at=float(payload.get("updated_at") or 0.0),
            expires_at=float(payload.get("expires_at") or 0.0),
            repo_root=str(payload.get("repo_root") or ""),
            state_dir=str(payload.get("state_dir") or ""),
            terminal_reason=str(payload.get("terminal_reason") or ""),
        )


def _read_boot_id(*, proc_root: Path = Path("/proc")) -> str:
    try:
        return (proc_root / "sys" / "kernel" / "random" / "boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError):
        return ""


def read_process_birth(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessBirthIdentity | None:
    """Return process-birth identity for ``pid``, or None if the PID is gone."""

    if pid <= 0:
        return None
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        # Distinguishing "PID gone" from "inspection unavailable" is done by
        # callers that also probe ``proc_root`` itself.
        raise
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) < 20 or fields[0] == "Z":
        return None
    try:
        parent_pid = int(fields[1])
        start_time_ticks = int(fields[19])
    except (TypeError, ValueError, IndexError):
        return None
    return ProcessBirthIdentity(
        pid=int(pid),
        start_time_ticks=start_time_ticks,
        boot_id=_read_boot_id(proc_root=proc_root),
        parent_pid=parent_pid,
    )


def current_process_birth(
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessBirthIdentity:
    """Return birth identity for the calling process."""

    pid = os.getpid()
    try:
        identity = read_process_birth(pid, proc_root=proc_root)
    except OSError:
        identity = None
    if identity is not None:
        return identity
    # Fall back when /proc is unavailable so the record can still be written;
    # cleanup will fail closed on UNKNOWN liveness.
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=0,
        boot_id=_read_boot_id(proc_root=proc_root),
        parent_pid=os.getppid() if hasattr(os, "getppid") else 0,
    )


def proc_available(*, proc_root: Path = Path("/proc")) -> bool:
    """Return whether process inspection is available (fail closed otherwise)."""

    try:
        return proc_root.is_dir()
    except OSError:
        return False


def owner_liveness(
    owner: ProcessBirthIdentity,
    *,
    proc_root: Path = Path("/proc"),
) -> OwnerLiveness:
    """Evaluate whether the recorded owner process still exists."""

    if owner.pid <= 0:
        return OwnerLiveness.DEAD
    if not proc_available(proc_root=proc_root):
        return OwnerLiveness.UNKNOWN
    try:
        current = read_process_birth(owner.pid, proc_root=proc_root)
    except OSError:
        return OwnerLiveness.UNKNOWN
    if current is None:
        return OwnerLiveness.DEAD
    if owner.start_time_ticks and current.start_time_ticks != owner.start_time_ticks:
        return OwnerLiveness.DEAD  # PID reuse
    if owner.boot_id and current.boot_id and owner.boot_id != current.boot_id:
        return OwnerLiveness.DEAD
    return OwnerLiveness.ALIVE


def new_lease_id(*, seed: str = "") -> str:
    material = f"{seed}:{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def normalize_workspace_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve(strict=False))
    except OSError:
        return str(candidate)


def workspace_record_filename(workspace_path: str | Path) -> str:
    normalized = normalize_workspace_path(workspace_path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"ws-{digest}.json"


def task_attempt_index_filename(
    *,
    canonical_task_cid: str,
    task_id: str,
    attempt: int,
) -> str:
    identity = f"{canonical_task_cid or task_id}\0{int(attempt)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    safe = _SAFE_NAME.sub("_", (canonical_task_cid or task_id or "task")[:48]).strip(
        "._-"
    ) or "task"
    return f"task-{safe}-{digest}-a{int(attempt)}.json"


def lifecycle_store_dir(repo_root: Path) -> Path:
    """Return the shared durable lifecycle directory for one Git repository."""

    return git_common_dir(repo_root) / WORKTREE_LIFECYCLE_DIRNAME


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def classify_lifecycle_race(reason: str) -> CleanupDecision:
    """Return a no-provider, no-retry decision for internal lifecycle races."""

    return CleanupDecision(
        disposition=CleanupDisposition.DENY,
        reason=reason,
        failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
        provider_call_allowed=False,
        attempt_consumed=False,
    )


@dataclass
class WorktreeLifecycleStore:
    """Durable CAS store for fenced worktree lifecycle records.

    Records are shared across every lane of one physical Git repository via
    ``git_common_dir``.  All mutating operations hold an advisory update guard
    for the target path and compare the expected fence before publishing.
    """

    repo_root: Path
    lease_seconds: float = DEFAULT_LEASE_SECONDS
    startup_grace_seconds: float = DEFAULT_STARTUP_GRACE_SECONDS
    clock: Callable[[], float] = field(default=DEFAULT_CLOCK)
    proc_root: Path = field(default_factory=lambda: Path("/proc"))
    store_dir: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        if self.store_dir is None:
            self.store_dir = lifecycle_store_dir(self.repo_root)
        else:
            self.store_dir = Path(self.store_dir)
        self.lease_seconds = max(1.0, float(self.lease_seconds))
        self.startup_grace_seconds = max(0.0, float(self.startup_grace_seconds))

    # ------------------------------------------------------------------ paths

    def workspace_path_for(self, workspace: str | Path) -> Path:
        assert self.store_dir is not None
        return self.store_dir / workspace_record_filename(workspace)

    def task_index_path_for(
        self,
        *,
        canonical_task_cid: str,
        task_id: str,
        attempt: int,
    ) -> Path:
        assert self.store_dir is not None
        return self.store_dir / task_attempt_index_filename(
            canonical_task_cid=canonical_task_cid,
            task_id=task_id,
            attempt=attempt,
        )

    # ---------------------------------------------------------------- loading

    def load_workspace(
        self,
        workspace: str | Path,
    ) -> WorkspaceLifecycleRecord | None:
        path = self.workspace_path_for(workspace)
        payload = _load_json_dict(path)
        if payload is None:
            return None
        try:
            return WorkspaceLifecycleRecord.from_dict(payload)
        except (TypeError, ValueError, WorktreeLifecycleError):
            return None

    def load_task_attempt(
        self,
        *,
        canonical_task_cid: str,
        task_id: str,
        attempt: int,
    ) -> WorkspaceLifecycleRecord | None:
        index_path = self.task_index_path_for(
            canonical_task_cid=canonical_task_cid,
            task_id=task_id,
            attempt=attempt,
        )
        payload = _load_json_dict(index_path)
        if payload is None:
            return None
        workspace = str(payload.get("workspace_path") or "")
        if not workspace:
            try:
                return WorkspaceLifecycleRecord.from_dict(payload)
            except (TypeError, ValueError, WorktreeLifecycleError):
                return None
        record = self.load_workspace(workspace)
        if record is not None and (
            str(payload.get("record_id") or "") == record.record_id
            and str(payload.get("lease_id") or "") == record.lease_id
            and normalize_workspace_path(workspace)
            == normalize_workspace_path(record.workspace_path)
        ):
            return record
        if record is None:
            try:
                return WorkspaceLifecycleRecord.from_dict(payload)
            except (TypeError, ValueError, WorktreeLifecycleError):
                return None
        # The secondary index no longer names the record at this workspace.
        # Never resolve a prior task/attempt to a replacement owner's record.
        return None

    def iter_records(self) -> Iterable[WorkspaceLifecycleRecord]:
        assert self.store_dir is not None
        if not self.store_dir.is_dir():
            return
        for path in sorted(self.store_dir.glob("ws-*.json")):
            payload = _load_json_dict(path)
            if payload is None:
                continue
            try:
                yield WorkspaceLifecycleRecord.from_dict(payload)
            except (TypeError, ValueError, WorktreeLifecycleError):
                continue

    def find_by_branch(self, branch: str) -> list[WorkspaceLifecycleRecord]:
        target = str(branch or "").removeprefix("refs/heads/")
        return [
            record
            for record in self.iter_records()
            if record.branch.removeprefix("refs/heads/") == target
        ]

    def find_nonterminal_for_workspace(
        self,
        workspace: str | Path,
    ) -> WorkspaceLifecycleRecord | None:
        record = self.load_workspace(workspace)
        if record is not None and record.is_nonterminal:
            return record
        return None

    @staticmethod
    def _task_index_payload(
        record: WorkspaceLifecycleRecord,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": WORKTREE_LIFECYCLE_SCHEMA,
            "workspace_path": record.workspace_path,
            "record_id": record.record_id,
            "task_id": record.task_id,
            "canonical_task_cid": record.canonical_task_cid,
            "attempt": record.attempt,
            "fence": record.fence,
            "lease_id": record.lease_id,
            "state": record.state.value,
        }
        if record.terminal_reason:
            payload["terminal_reason"] = record.terminal_reason
        return payload

    @staticmethod
    def _task_index_names_record(
        payload: Mapping[str, Any] | None,
        record: WorkspaceLifecycleRecord,
        *,
        allow_state_mismatch: bool = False,
    ) -> bool:
        """Return whether a secondary pointer still names ``record``.

        A lower fence is accepted for interrupted-write recovery: workspace
        records are published before their secondary projection while both
        locks are held.  A different record, workspace, lease, or a future
        fence is never overwritten.
        """

        if payload is None:
            return True
        try:
            indexed_fence = int(payload.get("fence") or 0)
        except (TypeError, ValueError):
            return False
        if (
            str(payload.get("record_id") or "") != record.record_id
            or str(payload.get("lease_id") or "") != record.lease_id
            or normalize_workspace_path(str(payload.get("workspace_path") or ""))
            != normalize_workspace_path(record.workspace_path)
            or indexed_fence > int(record.fence)
        ):
            return False
        if indexed_fence == int(record.fence) and not allow_state_mismatch:
            indexed_state = str(payload.get("state") or "")
            if indexed_state and indexed_state != record.state.value:
                return False
        return True

    def _require_task_index_authority(
        self,
        index_path: Path,
        record: WorkspaceLifecycleRecord,
    ) -> None:
        if not self._task_index_names_record(_load_json_dict(index_path), record):
            raise OwnershipError(
                "task/attempt index no longer names lifecycle record"
            )

    def _reclaim_dead_under_index_lock(
        self,
        expected: WorkspaceLifecycleRecord,
        *,
        index_path: Path,
        reclaimer_lease_id: str,
        reason: str,
        clock_now: float,
        require_expired: bool,
        expected_state_dir: str = "",
        reclaimer: ProcessBirthIdentity | None = None,
    ) -> WorkspaceLifecycleRecord | None:
        """Terminalize one exact dead claim while its task index is locked."""

        record_path = self.workspace_path_for(expected.workspace_path)
        with serialized_lock_update(record_path):
            current = self.load_workspace(expected.workspace_path)
            if current is None:
                return None
            if current.is_terminal:
                if self._task_index_names_record(
                    _load_json_dict(index_path),
                    current,
                    allow_state_mismatch=True,
                ):
                    _atomic_write_json(
                        index_path,
                        self._task_index_payload(current),
                    )
                return current
            if (
                current.record_id != expected.record_id
                or current.lease_id != expected.lease_id
                or current.fence != expected.fence
            ):
                return None
            self._require_task_index_authority(index_path, current)
            if (
                owner_liveness(current.owner, proc_root=self.proc_root)
                is not OwnerLiveness.DEAD
            ):
                return None
            if require_expired and clock_now < float(current.expires_at):
                return None
            if expected_state_dir:
                if (
                    not current.repo_root
                    or normalize_workspace_path(current.repo_root)
                    != normalize_workspace_path(self.repo_root)
                    or not current.state_dir
                    or normalize_workspace_path(current.state_dir)
                    != normalize_workspace_path(expected_state_dir)
                ):
                    return None
            updated = replace(
                current,
                state=WorkspaceLifecycleState.TERMINAL,
                owner=reclaimer or current.owner,
                lease_id=reclaimer_lease_id or current.lease_id,
                fence=int(current.fence) + 1,
                updated_at=clock_now,
                expires_at=(clock_now if expected_state_dir else current.expires_at),
                terminal_reason=str(reason or "stale_reclamation"),
            )
            _atomic_write_json(record_path, updated.to_dict())
            _atomic_write_json(index_path, self._task_index_payload(updated))
            return updated

    # -------------------------------------------------------------- acquisition

    def begin_preparing(
        self,
        *,
        task_id: str,
        canonical_task_cid: str = "",
        attempt: int,
        lane_id: str,
        workspace_path: str | Path,
        branch: str,
        merge_target: str,
        lease_id: str | None = None,
        state_dir: str = "",
        owner: ProcessBirthIdentity | None = None,
        allow_replace_stale: bool = True,
    ) -> WorkspaceLifecycleRecord:
        """Acquire ownership and publish a preparing record *before* worktree add.

        The preparing record is cleanup-visible.  Callers must invoke this
        before ``git worktree add`` so peer lanes never treat the checkout as
        an unclaimed already-merged orphan.
        """

        workspace = normalize_workspace_path(workspace_path)
        branch_name = str(branch or "").removeprefix("refs/heads/")
        if not branch_name:
            raise WorktreeLifecycleError("branch is required")
        now = float(self.clock())
        owner_identity = owner or current_process_birth(proc_root=self.proc_root)
        lease = lease_id or new_lease_id(
            seed=f"{task_id}:{canonical_task_cid}:{attempt}:{lane_id}"
        )
        record_path = self.workspace_path_for(workspace)
        index_path = self.task_index_path_for(
            canonical_task_cid=canonical_task_cid,
            task_id=task_id,
            attempt=attempt,
        )

        def _reject_other_task_attempt_claim() -> None:
            other = self.load_task_attempt(
                canonical_task_cid=canonical_task_cid,
                task_id=task_id,
                attempt=attempt,
            )
            if (
                other is None
                or other.is_terminal
                or normalize_workspace_path(other.workspace_path) == workspace
            ):
                return
            if not allow_replace_stale:
                raise DuplicateAttemptError(
                    "task/attempt replacement disabled by caller"
                )
            other_live = owner_liveness(other.owner, proc_root=self.proc_root)
            if other_live in {OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN}:
                raise DuplicateAttemptError(
                    "task/attempt already has a nonterminal workspace claim"
                )
            # Owner is provably dead. Same-lane restarts must reclaim without
            # waiting the full lease window (default 6h), otherwise daemon
            # restarts thrash on implementation_retry_deferred forever.
            same_lane = bool(
                state_dir
                and other.state_dir
                and normalize_workspace_path(state_dir)
                == normalize_workspace_path(other.state_dir)
            )
            expired = now >= float(other.expires_at)
            if not expired and not same_lane:
                raise DuplicateAttemptError(
                    "task/attempt claim lease has not expired"
                )
            reclaimed = self._reclaim_dead_under_index_lock(
                other,
                index_path=index_path,
                reclaimer_lease_id=lease if expired else "",
                reason=(
                    "dead_owner_same_lane_attempt_reclaim"
                    if same_lane
                    else "dead_owner_expired_attempt_reclaim"
                ),
                clock_now=now,
                require_expired=not same_lane,
                expected_state_dir=(other.state_dir if same_lane else ""),
                reclaimer=(owner_identity if same_lane else None),
            )
            if reclaimed is None:
                raise DuplicateAttemptError(
                    "task/attempt dead-owner reclaim refused"
                )

        # Serialize first on the stable task/attempt identity.  A losing lane
        # is rejected before it materializes a timestamp-specific workspace
        # guard, which also makes distinct-workspace claims atomic.
        with serialized_lock_update(index_path):
            _reject_other_task_attempt_claim()
            with serialized_lock_update(record_path):
                existing = self.load_workspace(workspace)
                if existing is not None and existing.is_nonterminal:
                    if not allow_replace_stale:
                        raise DuplicateAttemptError(
                            "workspace replacement disabled by caller"
                        )
                    if (
                        existing.task_id != str(task_id)
                        or existing.canonical_task_cid
                        != str(canonical_task_cid or "")
                        or existing.attempt != int(attempt)
                    ):
                        raise DuplicateAttemptError(
                            "workspace already belongs to another task/attempt"
                        )
                    liveness = owner_liveness(
                        existing.owner, proc_root=self.proc_root
                    )
                    expired = now >= float(existing.expires_at)
                    same_lane = bool(
                        state_dir
                        and existing.state_dir
                        and normalize_workspace_path(state_dir)
                        == normalize_workspace_path(existing.state_dir)
                    )
                    if liveness is OwnerLiveness.ALIVE:
                        raise DuplicateAttemptError(
                            "workspace already claimed by live owner "
                            f"pid={existing.owner.pid}"
                        )
                    if liveness is OwnerLiveness.UNKNOWN:
                        raise DuplicateAttemptError(
                            "workspace claim exists and process inspection is "
                            "unavailable"
                        )
                    if not expired and not same_lane:
                        # Dead peer-lane owner: keep lease-gated reclaim so
                        # concurrent lanes cannot race while the record is
                        # still within its advertised exclusivity window.
                        raise DuplicateAttemptError(
                            "workspace claim lease has not expired for stale "
                            "owner"
                        )
                    self._require_task_index_authority(index_path, existing)
                    # Only a provably dead owner can be replaced.  Same-lane
                    # controlled restart may precede expiry; peer lanes wait.
                    next_fence = int(existing.fence) + 1
                else:
                    next_fence = (
                        1 if existing is None else int(existing.fence) + 1
                    )

                # Recheck after taking the workspace guard because transitions
                # and legacy writers may update the index without this guard.
                _reject_other_task_attempt_claim()
                record = WorkspaceLifecycleRecord(
                    task_id=str(task_id),
                    canonical_task_cid=str(canonical_task_cid or ""),
                    attempt=int(attempt),
                    lane_id=str(lane_id or ""),
                    state=WorkspaceLifecycleState.PREPARING,
                    owner=owner_identity,
                    lease_id=lease,
                    fence=next_fence,
                    workspace_path=workspace,
                    branch=branch_name,
                    merge_target=str(merge_target or ""),
                    created_at=(
                        now if existing is None else float(existing.created_at)
                    ),
                    updated_at=now,
                    expires_at=now + self.lease_seconds,
                    repo_root=str(self.repo_root.resolve(strict=False)),
                    state_dir=str(state_dir or ""),
                    terminal_reason="",
                )
                _atomic_write_json(record_path, record.to_dict())
                _atomic_write_json(index_path, self._task_index_payload(record))
                return record

    # -------------------------------------------------------------- transitions

    def _require_owner(
        self,
        record: WorkspaceLifecycleRecord,
        *,
        lease_id: str,
        expected_fence: int,
    ) -> None:
        if str(lease_id) != str(record.lease_id):
            raise OwnershipError("lifecycle lease does not match record owner")
        if int(expected_fence) != int(record.fence):
            raise FenceMismatchError(
                f"expected fence {expected_fence}, found {record.fence}"
            )

    def transition(
        self,
        workspace: str | Path,
        new_state: WorkspaceLifecycleState | str,
        *,
        lease_id: str,
        expected_fence: int,
        renew_lease: bool = True,
        terminal_reason: str = "",
    ) -> WorkspaceLifecycleRecord:
        """Owner-only CAS state transition with optional lease renewal."""

        if isinstance(new_state, str):
            new_state = WorkspaceLifecycleState(new_state)
        record_path = self.workspace_path_for(workspace)
        initial = self.load_workspace(workspace)
        if initial is None:
            raise WorktreeLifecycleError("lifecycle record missing")
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self.load_workspace(workspace)
                if current is None:
                    raise WorktreeLifecycleError("lifecycle record missing")
                self._require_owner(
                    current, lease_id=lease_id, expected_fence=expected_fence
                )
                if current.record_id != initial.record_id:
                    raise OwnershipError(
                        "workspace record changed before task-index lock"
                    )
                self._require_task_index_authority(index_path, current)
                if (
                    current.is_terminal
                    and new_state is not WorkspaceLifecycleState.TERMINAL
                ):
                    raise WorktreeLifecycleError(
                        "cannot revive a terminal lifecycle record"
                    )
                now = float(self.clock())
                expires_at = (
                    now + self.lease_seconds
                    if renew_lease
                    else float(current.expires_at)
                )
                # Fence advances on every authoritative mutation so concurrent
                # cleaners cannot delete against a stale view.
                updated = replace(
                    current,
                    state=new_state,
                    fence=int(current.fence) + 1,
                    updated_at=now,
                    expires_at=expires_at,
                    terminal_reason=(
                        str(terminal_reason or current.terminal_reason)
                        if new_state is WorkspaceLifecycleState.TERMINAL
                        else ""
                    ),
                )
                _atomic_write_json(record_path, updated.to_dict())
                _atomic_write_json(
                    index_path,
                    self._task_index_payload(updated),
                )
                return updated

    def rebind_workspace(
        self,
        workspace: str | Path,
        new_workspace: str | Path,
        *,
        lease_id: str,
        expected_fence: int,
    ) -> WorkspaceLifecycleRecord:
        """Owner-only CAS rebind when pool resolution changes the physical path.

        The provisional preparing path may differ from the pooled checkout.  The
        claim must move with the owner without opening a window where peer
        cleanup can treat either path as unclaimed, and without tripping the
        duplicate task/attempt guard against our own nonterminal record.
        """

        old_normalized = normalize_workspace_path(workspace)
        new_normalized = normalize_workspace_path(new_workspace)
        if old_normalized == new_normalized:
            current = self.load_workspace(old_normalized)
            if current is None:
                raise WorktreeLifecycleError("lifecycle record missing")
            self._require_owner(
                current, lease_id=lease_id, expected_fence=expected_fence
            )
            return current

        old_path = self.workspace_path_for(old_normalized)
        new_path = self.workspace_path_for(new_normalized)
        initial = self.load_workspace(old_normalized)
        if initial is None:
            raise WorktreeLifecycleError("lifecycle record missing")
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )

        def _rebind_body() -> WorkspaceLifecycleRecord:
            current = self.load_workspace(old_normalized)
            if current is None:
                raise WorktreeLifecycleError("lifecycle record missing")
            self._require_owner(
                current, lease_id=lease_id, expected_fence=expected_fence
            )
            if current.record_id != initial.record_id:
                raise OwnershipError(
                    "workspace record changed before task-index lock"
                )
            self._require_task_index_authority(index_path, current)
            if current.is_terminal:
                raise WorktreeLifecycleError(
                    "cannot rebind a terminal lifecycle record"
                )
            existing_new = self.load_workspace(new_normalized)
            if existing_new is not None and existing_new.is_nonterminal:
                # Rebinding owns only this task index.  Reclaiming another
                # task's target record here would require its index lock and
                # creates an avoidable multi-index transaction.  Cleanup must
                # terminalize/delete that exact claim first.
                raise DuplicateAttemptError(
                    "target workspace already has a nonterminal claim"
                )
            now = float(self.clock())
            updated = replace(
                current,
                workspace_path=new_normalized,
                fence=int(current.fence) + 1,
                updated_at=now,
                expires_at=now + self.lease_seconds,
                record_id=WorkspaceLifecycleRecord.compute_record_id_for(
                    canonical_task_cid=current.canonical_task_cid,
                    task_id=current.task_id,
                    attempt=current.attempt,
                    workspace_path=new_normalized,
                ),
            )
            _atomic_write_json(new_path, updated.to_dict())
            _atomic_write_json(index_path, self._task_index_payload(updated))
            if old_path != new_path:
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
            return updated

        # Lock both paths in a stable order.  When filenames collide, one
        # advisory guard is enough (non-recursive flock would deadlock).
        with serialized_lock_update(index_path):
            if old_path == new_path:
                with serialized_lock_update(old_path):
                    return _rebind_body()
            first, second = sorted([old_path, new_path], key=lambda item: str(item))
            with serialized_lock_update(first):
                with serialized_lock_update(second):
                    return _rebind_body()

    def mark_active(
        self,
        workspace: str | Path,
        *,
        lease_id: str,
        expected_fence: int,
    ) -> WorkspaceLifecycleRecord:
        return self.transition(
            workspace,
            WorkspaceLifecycleState.ACTIVE,
            lease_id=lease_id,
            expected_fence=expected_fence,
        )

    def mark_settling(
        self,
        workspace: str | Path,
        *,
        lease_id: str,
        expected_fence: int,
    ) -> WorkspaceLifecycleRecord:
        return self.transition(
            workspace,
            WorkspaceLifecycleState.SETTLING,
            lease_id=lease_id,
            expected_fence=expected_fence,
        )

    def mark_terminal(
        self,
        workspace: str | Path,
        *,
        lease_id: str,
        expected_fence: int,
        reason: str = "owner_terminal",
    ) -> WorkspaceLifecycleRecord:
        return self.transition(
            workspace,
            WorkspaceLifecycleState.TERMINAL,
            lease_id=lease_id,
            expected_fence=expected_fence,
            renew_lease=False,
            terminal_reason=reason,
        )

    def renew_lease(
        self,
        workspace: str | Path,
        *,
        lease_id: str,
        expected_fence: int,
    ) -> WorkspaceLifecycleRecord:
        """Heartbeat: advance fence and push expiry without changing state."""

        record_path = self.workspace_path_for(workspace)
        initial = self.load_workspace(workspace)
        if initial is None:
            raise WorktreeLifecycleError("lifecycle record missing")
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self.load_workspace(workspace)
                if current is None:
                    raise WorktreeLifecycleError("lifecycle record missing")
                self._require_owner(
                    current, lease_id=lease_id, expected_fence=expected_fence
                )
                if current.record_id != initial.record_id:
                    raise OwnershipError(
                        "workspace record changed before task-index lock"
                    )
                self._require_task_index_authority(index_path, current)
                if current.is_terminal:
                    raise WorktreeLifecycleError(
                        "cannot renew a terminal lifecycle record"
                    )
                now = float(self.clock())
                updated = replace(
                    current,
                    fence=int(current.fence) + 1,
                    updated_at=now,
                    expires_at=now + self.lease_seconds,
                )
                _atomic_write_json(record_path, updated.to_dict())
                _atomic_write_json(
                    index_path,
                    self._task_index_payload(updated),
                )
                return updated

    # ---------------------------------------------------------------- cleanup

    def evaluate_cleanup(
        self,
        *,
        workspace_path: str | Path | None = None,
        branch: str = "",
        caller_lease_id: str = "",
        now: float | None = None,
    ) -> CleanupDecision:
        """Decide whether cleanup may delete/prune/unregister a worktree.

        Nonterminal claims are never cleaned, including the window between
        ``git worktree add`` and child-process discovery, even when the branch
        tip is an ancestor of the merge target.
        """

        clock_now = float(self.clock() if now is None else now)
        record: WorkspaceLifecycleRecord | None = None
        if workspace_path is not None:
            record = self.load_workspace(workspace_path)
        if record is None and branch:
            matches = [
                item
                for item in self.find_by_branch(branch)
                if item.is_nonterminal
            ]
            if matches:
                # Prefer the newest nonterminal claim for the branch.
                record = max(matches, key=lambda item: (item.updated_at, item.fence))

        if record is None:
            return CleanupDecision(
                disposition=CleanupDisposition.ALLOW,
                reason="no_lifecycle_record",
                record=None,
            )

        if record.is_terminal:
            return CleanupDecision(
                disposition=CleanupDisposition.ALLOW,
                reason="terminal_record",
                record=record,
            )

        # Owner may always settle/dispose its own workspace.
        if caller_lease_id and caller_lease_id == record.lease_id:
            return CleanupDecision(
                disposition=CleanupDisposition.ALLOW,
                reason="caller_is_record_owner",
                record=record,
            )

        liveness = owner_liveness(record.owner, proc_root=self.proc_root)
        if liveness is OwnerLiveness.ALIVE:
            return CleanupDecision(
                disposition=CleanupDisposition.DENY,
                reason=f"nonterminal_{record.state.value}_owner_alive",
                record=record,
                failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                provider_call_allowed=False,
                attempt_consumed=False,
            )

        if liveness is OwnerLiveness.UNKNOWN:
            return CleanupDecision(
                disposition=CleanupDisposition.DENY,
                reason="process_inspection_unavailable",
                record=record,
                failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                provider_call_allowed=False,
                attempt_consumed=False,
            )

        # Owner is dead.  Still require lease expiry (plus optional grace for
        # brand-new preparing records that may be mid-publication).
        age = clock_now - float(record.created_at)
        expired = clock_now >= float(record.expires_at)
        if not expired:
            if (
                record.state is WorkspaceLifecycleState.PREPARING
                and age < self.startup_grace_seconds
            ):
                return CleanupDecision(
                    disposition=CleanupDisposition.DENY,
                    reason="preparing_startup_grace",
                    record=record,
                    failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                    provider_call_allowed=False,
                    attempt_consumed=False,
                )
            return CleanupDecision(
                disposition=CleanupDisposition.DENY,
                reason="owner_dead_lease_unexpired",
                record=record,
                failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                provider_call_allowed=False,
                attempt_consumed=False,
            )

        return CleanupDecision(
            disposition=CleanupDisposition.RECLAIM_THEN_ALLOW,
            reason="stale_owner_lease_expired",
            record=record,
            failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
            provider_call_allowed=False,
            attempt_consumed=False,
        )

    def reclaim_stale(
        self,
        workspace: str | Path,
        *,
        reclaimer_lease_id: str = "",
        reason: str = "stale_reclamation",
        now: float | None = None,
    ) -> WorkspaceLifecycleRecord | None:
        """Advance fence when the owner is provably dead and lease has expired.

        Returns the terminal record on success, or None if reclamation was
        refused (live/unknown owner, unexpired lease, missing record, etc.).
        """

        clock_now = float(self.clock() if now is None else now)
        initial = self.load_workspace(workspace)
        if initial is None:
            return None
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )
        with serialized_lock_update(index_path):
            return self._reclaim_dead_under_index_lock(
                initial,
                index_path=index_path,
                reclaimer_lease_id=reclaimer_lease_id,
                reason=reason,
                clock_now=clock_now,
                require_expired=True,
            )

    def reclaim_expired_nonterminal(
        self,
        *,
        reclaimer_lease_id: str = "",
        reason: str = "expired_lease_auto_reclaim",
        task_id_prefix: str = "",
    ) -> list[WorkspaceLifecycleRecord]:
        """Terminalize expired claims whose process-birth owners are dead."""

        recovered: list[WorkspaceLifecycleRecord] = []
        prefix = str(task_id_prefix or "")
        for record in list(self.iter_records()):
            if record.is_terminal:
                continue
            if prefix and not str(record.task_id).startswith(prefix):
                continue
            expires_at = float(getattr(record, "expires_at", 0.0) or 0.0)
            updated = self.reclaim_stale(
                record.workspace_path,
                reclaimer_lease_id=(
                    reclaimer_lease_id
                    or new_lease_id(
                        seed=(
                            "expired-auto-reclaim-zero-lease"
                            if expires_at <= 0.0
                            else "expired-auto-reclaim"
                        )
                    )
                ),
                reason=(
                    f"{reason}:zero_or_missing_expires_at"
                    if expires_at <= 0.0
                    else reason
                ),
            )
            if updated is not None and updated.is_terminal:
                recovered.append(updated)
        self.reconcile_stale_task_indexes(task_id_prefix=prefix)
        return recovered

    def reconcile_stale_task_indexes(
        self,
        *,
        task_id_prefix: str = "",
    ) -> int:
        """Align task-attempt pointers with terminal workspace records."""

        assert self.store_dir is not None
        if not self.store_dir.is_dir():
            return 0
        repaired = 0
        prefix = str(task_id_prefix or "")
        for path in sorted(self.store_dir.glob("task-*.json")):
            with serialized_lock_update(path):
                payload = _load_json_dict(path)
                if payload is None:
                    continue
                task_id = str(payload.get("task_id") or "")
                if prefix and not task_id.startswith(prefix):
                    continue
                if str(payload.get("state") or "") in {
                    "",
                    WorkspaceLifecycleState.TERMINAL.value,
                }:
                    continue
                workspace = str(payload.get("workspace_path") or "")
                if not workspace:
                    continue
                record_path = self.workspace_path_for(workspace)
                with serialized_lock_update(record_path):
                    # Re-read both authorities under the shared lock order.
                    current_payload = _load_json_dict(path)
                    record = self.load_workspace(workspace)
                    if (
                        current_payload is None
                        or record is None
                        or not record.is_terminal
                        or not self._task_index_names_record(
                            current_payload,
                            record,
                            allow_state_mismatch=True,
                        )
                    ):
                        continue
                    _atomic_write_json(
                        path,
                        self._task_index_payload(record),
                    )
                    repaired += 1
        return repaired

    def reclaim_dead_owner_for_controlled_restart(
        self,
        workspace: str | Path,
        *,
        expected_state_dir: str | Path,
        reclaimer_lease_id: str = "",
        reclaimer: ProcessBirthIdentity | None = None,
        reason: str = "controlled_restart_dead_owner",
        now: float | None = None,
    ) -> WorkspaceLifecycleRecord | None:
        """Fence a dead same-lane owner during an explicit controlled restart.

        Normal cleanup remains lease-expiry gated.  This recovery path is for
        an operator-controlled daemon restart and fails closed unless the
        record belongs to this repository and exact lane state directory and
        the prior process-birth identity is provably dead.
        """

        expected_state = normalize_workspace_path(expected_state_dir)
        if not expected_state:
            return None
        expected_repo = normalize_workspace_path(self.repo_root)
        clock_now = float(self.clock() if now is None else now)
        initial = self.load_workspace(workspace)
        if initial is None or initial.is_terminal:
            return None
        if (
            not initial.repo_root
            or normalize_workspace_path(initial.repo_root) != expected_repo
        ):
            return None
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )
        with serialized_lock_update(index_path):
            return self._reclaim_dead_under_index_lock(
                initial,
                index_path=index_path,
                reclaimer_lease_id=(
                    reclaimer_lease_id
                    or new_lease_id(seed="controlled-restart-reclaim")
                ),
                reason=reason,
                clock_now=clock_now,
                require_expired=False,
                expected_state_dir=expected_state,
                reclaimer=(
                    reclaimer
                    or current_process_birth(proc_root=self.proc_root)
                ),
            )

    def reclaim_dead_owners_for_controlled_restart(
        self,
        *,
        expected_state_dir: str | Path,
        reclaimer_lease_id: str = "",
        reason: str = "controlled_restart_dead_owner",
    ) -> list[WorkspaceLifecycleRecord]:
        """Fence all provably dead records owned by one restarted lane."""

        recovered: list[WorkspaceLifecycleRecord] = []
        for record in list(self.iter_records()):
            if record.is_terminal:
                continue
            updated = self.reclaim_dead_owner_for_controlled_restart(
                record.workspace_path,
                expected_state_dir=expected_state_dir,
                reclaimer_lease_id=reclaimer_lease_id,
                reason=reason,
            )
            if updated is not None:
                recovered.append(updated)
        return recovered

    def compare_and_delete(
        self,
        workspace: str | Path,
        *,
        expected_fence: int,
        lease_id: str = "",
    ) -> bool:
        """Remove the lifecycle record only when fence (and optional lease) match."""

        record_path = self.workspace_path_for(workspace)
        initial = self.load_workspace(workspace)
        if initial is None:
            return True
        index_path = self.task_index_path_for(
            canonical_task_cid=initial.canonical_task_cid,
            task_id=initial.task_id,
            attempt=initial.attempt,
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self.load_workspace(workspace)
                if current is None:
                    return True
                if current.record_id != initial.record_id:
                    return False
                if int(current.fence) != int(expected_fence):
                    return False
                if lease_id and str(current.lease_id) != str(lease_id):
                    return False
                index_payload = _load_json_dict(index_path)
                if not self._task_index_names_record(index_payload, current):
                    return False
                try:
                    record_path.unlink()
                except FileNotFoundError:
                    pass
                # Compare-and-delete the secondary pointer; never unlink a
                # later workspace claim for the same task/attempt.
                if self._task_index_names_record(
                    _load_json_dict(index_path), current
                ):
                    try:
                        index_path.unlink()
                    except FileNotFoundError:
                        pass
                return True

    def finalize_exact_after_effect(
        self,
        expected: WorkspaceLifecycleRecord,
        *,
        reason: str,
        effect: Callable[[], Any],
    ) -> tuple[Any, WorkspaceLifecycleRecord]:
        """Run one destructive cleanup effect under the exact owner locks.

        The task-index lock is acquired before the workspace lock.  A new
        claimant therefore cannot publish a replacement record between the
        ownership CAS and physical worktree/pool disposal.  The lifecycle
        authorities are removed only after the effect completes.
        """

        record_path = self.workspace_path_for(expected.workspace_path)
        index_path = self.task_index_path_for(
            canonical_task_cid=expected.canonical_task_cid,
            task_id=expected.task_id,
            attempt=expected.attempt,
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self.load_workspace(expected.workspace_path)
                if current is None:
                    raise OwnershipError("lifecycle record missing before cleanup")
                if (
                    current.record_id != expected.record_id
                    or normalize_workspace_path(current.workspace_path)
                    != normalize_workspace_path(expected.workspace_path)
                    or current.lease_id != expected.lease_id
                    or current.fence != expected.fence
                ):
                    raise OwnershipError(
                        "lifecycle identity or fence changed before cleanup"
                    )
                self._require_task_index_authority(index_path, current)

                effect_result = effect()
                if current.is_terminal:
                    terminal = current
                else:
                    now = float(self.clock())
                    terminal = replace(
                        current,
                        state=WorkspaceLifecycleState.TERMINAL,
                        fence=int(current.fence) + 1,
                        updated_at=now,
                        terminal_reason=str(reason or "cleanup_finished"),
                    )
                    _atomic_write_json(record_path, terminal.to_dict())
                    _atomic_write_json(
                        index_path,
                        self._task_index_payload(terminal),
                    )
                if not self._task_index_names_record(
                    _load_json_dict(index_path), terminal
                ):
                    raise OwnershipError(
                        "task/attempt index changed during cleanup finalization"
                    )
                try:
                    record_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    index_path.unlink()
                except FileNotFoundError:
                    pass
                return effect_result, terminal

    def run_exact_owner_effect(
        self,
        expected: WorkspaceLifecycleRecord,
        *,
        effect: Callable[[], Any],
    ) -> Any:
        """Run an owner mutation while replacement publication is locked out."""

        record_path = self.workspace_path_for(expected.workspace_path)
        index_path = self.task_index_path_for(
            canonical_task_cid=expected.canonical_task_cid,
            task_id=expected.task_id,
            attempt=expected.attempt,
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self.load_workspace(expected.workspace_path)
                if current is None:
                    raise OwnershipError("lifecycle record missing before effect")
                if (
                    current.is_terminal
                    or current.record_id != expected.record_id
                    or normalize_workspace_path(current.workspace_path)
                    != normalize_workspace_path(expected.workspace_path)
                    or current.lease_id != expected.lease_id
                    or current.fence != expected.fence
                ):
                    raise OwnershipError(
                        "lifecycle identity or fence changed before effect"
                    )
                self._require_task_index_authority(index_path, current)
                return effect()

    def run_effect_if_unclaimed(
        self,
        workspace: str | Path,
        *,
        effect: Callable[[], Any],
    ) -> Any:
        """Run cleanup while holding the workspace lock only if still unclaimed."""

        record_path = self.workspace_path_for(workspace)
        with serialized_lock_update(record_path):
            if self.load_workspace(workspace) is not None:
                raise OwnershipError(
                    "workspace lifecycle claim appeared before cleanup"
                )
            return effect()

    def authorize_cleanup(
        self,
        *,
        workspace_path: str | Path,
        branch: str = "",
        caller_lease_id: str = "",
    ) -> CleanupDecision:
        """Evaluate and, when stale, reclaim under the store lock path.

        This is the single entry point cleanup code should call before
        ``git worktree remove`` / prune / branch delete / pool reuse.
        """

        decision = self.evaluate_cleanup(
            workspace_path=workspace_path,
            branch=branch,
            caller_lease_id=caller_lease_id,
        )
        if decision.disposition is CleanupDisposition.RECLAIM_THEN_ALLOW:
            # Branch fallback can find a preparing claim whose provisional
            # workspace differs from the stable pooled path supplied by the
            # caller. Reclaim the authoritative record, never the lookup hint.
            reclaim_workspace = (
                decision.record.workspace_path
                if decision.record is not None
                else workspace_path
            )
            reclaimed = self.reclaim_stale(
                reclaim_workspace,
                reclaimer_lease_id=caller_lease_id or new_lease_id(seed="reclaim"),
                reason=decision.reason,
            )
            if reclaimed is None:
                # Lost the reclaim race; re-evaluate.
                refreshed = self.evaluate_cleanup(
                    workspace_path=workspace_path,
                    branch=branch,
                    caller_lease_id=caller_lease_id,
                )
                if (
                    refreshed.disposition
                    is CleanupDisposition.RECLAIM_THEN_ALLOW
                ):
                    # Re-evaluation still requires an authoritative reclaim.
                    # Never expose that unresolved intermediate disposition as
                    # mutation authority to the caller.
                    return CleanupDecision(
                        disposition=CleanupDisposition.DENY,
                        reason="stale_reclaim_race_unresolved",
                        record=refreshed.record,
                        failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                        provider_call_allowed=False,
                        attempt_consumed=False,
                    )
                return refreshed
            return CleanupDecision(
                disposition=CleanupDisposition.ALLOW,
                reason="reclaimed_stale_record",
                record=reclaimed,
                failure_kind=LifecycleFailureKind.LIFECYCLE_RACE,
                provider_call_allowed=False,
                attempt_consumed=False,
            )
        return decision


def lifecycle_race_result(
    *,
    reason: str,
    task_id: str = "",
    attempt: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an implementation-daemon result for an internal lifecycle race.

    The result never authorizes a provider call and never consumes a retry.
    """

    payload: dict[str, Any] = {
        "skipped": True,
        "reason": reason,
        "task_id": task_id,
        "attempt": int(attempt or 0),
        "failure_kind": LifecycleFailureKind.LIFECYCLE_RACE.value,
        "provider_call_allowed": False,
        "attempt_consumed": False,
        "lifecycle_race": True,
        "requirement_id": FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID,
    }
    if extra:
        payload.update(dict(extra))
    return payload


__all__ = [
    "CleanupDecision",
    "CleanupDisposition",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_STARTUP_GRACE_SECONDS",
    "DuplicateAttemptError",
    "FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID",
    "FenceMismatchError",
    "LifecycleFailureKind",
    "OwnerLiveness",
    "OwnershipError",
    "ProcessBirthIdentity",
    "WORKTREE_LIFECYCLE_DIRNAME",
    "WORKTREE_LIFECYCLE_SCHEMA",
    "WorkspaceLifecycleRecord",
    "WorkspaceLifecycleState",
    "WorktreeLifecycleError",
    "WorktreeLifecycleStore",
    "classify_lifecycle_race",
    "current_process_birth",
    "lifecycle_race_result",
    "lifecycle_store_dir",
    "new_lease_id",
    "normalize_workspace_path",
    "owner_liveness",
    "proc_available",
    "read_process_birth",
]
