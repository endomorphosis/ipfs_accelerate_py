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
from typing import Any, Callable, Iterable, Mapping

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
    """Return process-birth identity for ``pid``, or ``None`` if it is gone.

    A readable but malformed proc record is not evidence that a process is
    dead.  Raise ``OSError`` for that inconclusive case so liveness callers
    fail closed with ``UNKNOWN``.
    """

    if pid <= 0:
        return None
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        # A missing PID directory proves absence.  A present PID directory
        # whose stat entry cannot be read is an inspection race/anomaly, not
        # proof of death.
        try:
            (proc_root / str(pid)).stat()
        except FileNotFoundError:
            return None
        except OSError:
            raise
        raise OSError(f"process stat unavailable for pid {pid}") from exc
    except OSError:
        # Distinguishing "PID gone" from "inspection unavailable" is done by
        # callers that also probe ``proc_root`` itself.
        raise
    close = raw.rfind(")")
    if close < 0:
        raise OSError(f"malformed process stat for pid {pid}: missing comm")
    fields = raw[close + 2 :].split()
    if fields and fields[0] == "Z":
        return None
    if len(fields) < 20:
        raise OSError(f"malformed process stat for pid {pid}: missing fields")
    try:
        parent_pid = int(fields[1])
        start_time_ticks = int(fields[19])
    except (TypeError, ValueError, IndexError):
        raise OSError(
            f"malformed process stat for pid {pid}: invalid identity fields"
        ) from None
    if start_time_ticks <= 0:
        raise OSError(
            f"malformed process stat for pid {pid}: invalid start time"
        )
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
        if record is None:
            try:
                return WorkspaceLifecycleRecord.from_dict(payload)
            except (TypeError, ValueError, WorktreeLifecycleError):
                return None
        return record

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
            other_live = owner_liveness(other.owner, proc_root=self.proc_root)
            if other_live in {OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN}:
                raise DuplicateAttemptError(
                    "task/attempt already has a nonterminal workspace claim"
                )
            if now < float(other.expires_at):
                raise DuplicateAttemptError(
                    "task/attempt claim lease has not expired"
                )

        # Serialize first on the stable task/attempt identity.  A losing lane
        # is rejected before it materializes a timestamp-specific workspace
        # guard, which also makes distinct-workspace claims atomic.
        with serialized_lock_update(index_path):
            _reject_other_task_attempt_claim()
            with serialized_lock_update(record_path):
                existing = self.load_workspace(workspace)
                if existing is not None and existing.is_nonterminal:
                    liveness = owner_liveness(
                        existing.owner, proc_root=self.proc_root
                    )
                    expired = now >= float(existing.expires_at)
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
                    if not expired and not allow_replace_stale:
                        raise DuplicateAttemptError(
                            "workspace claim exists and lease has not expired"
                        )
                    if not expired:
                        # Owner is dead but lease still valid: only reclaim
                        # after expiry.
                        raise DuplicateAttemptError(
                            "workspace claim lease has not expired for stale "
                            "owner"
                        )
                    # Dead + expired → reclaim with fence advancement below.
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
                _atomic_write_json(
                    index_path,
                    {
                        "schema": WORKTREE_LIFECYCLE_SCHEMA,
                        "workspace_path": workspace,
                        "record_id": record.record_id,
                        "task_id": record.task_id,
                        "canonical_task_cid": record.canonical_task_cid,
                        "attempt": record.attempt,
                        "fence": record.fence,
                        "lease_id": record.lease_id,
                        "state": record.state.value,
                    },
                )
                return record

    @staticmethod
    def _normalized_binding(
        *,
        task_id: str,
        canonical_task_cid: str,
        attempt: int,
        workspace_path: str | Path,
        branch: str,
        merge_target: str,
        repo_root: str | Path,
        state_dir: str | Path,
    ) -> dict[str, Any]:
        return {
            "task_id": str(task_id or ""),
            "canonical_task_cid": str(canonical_task_cid or ""),
            "attempt": int(attempt),
            "workspace_path": normalize_workspace_path(workspace_path),
            "branch": str(branch or "").removeprefix("refs/heads/"),
            "merge_target": str(merge_target or "").removeprefix("refs/heads/"),
            "repo_root": (
                str(Path(repo_root).resolve(strict=False)) if repo_root else ""
            ),
            "state_dir": (
                str(Path(state_dir).resolve(strict=False)) if state_dir else ""
            ),
        }

    def _require_exact_dead_owner(
        self,
        current: WorkspaceLifecycleRecord,
        *,
        expected_record_id: str,
        expected_fence: int,
        expected_lease_id: str,
        expected_binding: Mapping[str, Any],
        index_path: Path,
        allow_terminal: bool = False,
        expected_owner: ProcessBirthIdentity | None = None,
    ) -> None:
        """Validate one dead-owner claim without mutating durable state."""

        if current.is_terminal and not allow_terminal:
            raise WorktreeLifecycleError(
                "terminal lifecycle record cannot be adopted"
            )
        if current.record_id != current.compute_record_id():
            raise WorktreeLifecycleError(
                "orphan lifecycle record identity is invalid"
            )
        if current.record_id != str(expected_record_id or ""):
            raise FenceMismatchError("orphan lifecycle record id changed")
        self._require_owner(
            current,
            lease_id=expected_lease_id,
            expected_fence=expected_fence,
        )
        if expected_owner is not None and current.owner != expected_owner:
            raise OwnershipError("orphan lifecycle owner identity changed")

        actual_binding = self._normalized_binding(
            task_id=current.task_id,
            canonical_task_cid=current.canonical_task_cid,
            attempt=current.attempt,
            workspace_path=current.workspace_path,
            branch=current.branch,
            merge_target=current.merge_target,
            repo_root=current.repo_root,
            state_dir=current.state_dir,
        )
        mismatches = sorted(
            key
            for key, expected in expected_binding.items()
            if actual_binding.get(key) != expected
        )
        if mismatches:
            raise OwnershipError(
                "orphan lifecycle binding mismatch: " + ", ".join(mismatches)
            )

        expected_index = {
            "schema": WORKTREE_LIFECYCLE_SCHEMA,
            "workspace_path": current.workspace_path,
            "record_id": current.record_id,
            "task_id": current.task_id,
            "canonical_task_cid": current.canonical_task_cid,
            "attempt": int(current.attempt),
            "fence": int(current.fence),
            "lease_id": current.lease_id,
            "state": current.state.value,
        }
        index_payload = _load_json_dict(index_path)
        if (
            index_payload is None
            or set(index_payload) != set(expected_index)
            or type(index_payload.get("attempt")) is not int
            or type(index_payload.get("fence")) is not int
            or any(
                type(index_payload.get(key)) is not str
                for key in (
                    "schema",
                    "workspace_path",
                    "record_id",
                    "task_id",
                    "canonical_task_cid",
                    "lease_id",
                    "state",
                )
            )
            or index_payload != expected_index
        ):
            raise WorktreeLifecycleError(
                "orphan lifecycle task index mismatch"
            )

        liveness = owner_liveness(current.owner, proc_root=self.proc_root)
        if liveness is OwnerLiveness.ALIVE:
            raise OwnershipError("orphan lifecycle owner is still alive")
        if liveness is OwnerLiveness.UNKNOWN:
            raise OwnershipError(
                "orphan lifecycle owner liveness is unknown"
            )

    def _load_strict_workspace_record(
        self,
        workspace: str | Path,
    ) -> WorkspaceLifecycleRecord:
        """Load the exact persisted schema used for immediate dead adoption.

        Normal lifecycle reads remain backward compatible.  Immediate
        no-expiry adoption is intentionally stricter: no missing field may be
        synthesized and no legacy flat owner identity may be trusted.
        """

        payload = _load_json_dict(self.workspace_path_for(workspace))
        if payload is None:
            raise WorktreeLifecycleError(
                "orphan lifecycle record missing or malformed"
            )
        required_fields = {
            "schema",
            "record_id",
            "task_id",
            "canonical_task_cid",
            "attempt",
            "lane_id",
            "state",
            "owner",
            "lease_id",
            "fence",
            "workspace_path",
            "branch",
            "merge_target",
            "created_at",
            "updated_at",
            "expires_at",
            "repo_root",
            "state_dir",
            "terminal_reason",
        }
        if set(payload) != required_fields:
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted record shape is invalid"
            )
        if (
            type(payload["schema"]) is not str
            or payload["schema"] != WORKTREE_LIFECYCLE_SCHEMA
        ):
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted schema is invalid"
            )
        if (
            type(payload["record_id"]) is not str
            or not payload["record_id"]
        ):
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted record id is invalid"
            )
        for field_name in (
            "task_id",
            "canonical_task_cid",
            "lane_id",
            "state",
            "lease_id",
            "workspace_path",
            "branch",
            "merge_target",
            "repo_root",
            "state_dir",
            "terminal_reason",
        ):
            if type(payload[field_name]) is not str:
                raise WorktreeLifecycleError(
                    "orphan lifecycle persisted field type is invalid: "
                    f"{field_name}"
                )
        if not payload["lease_id"]:
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted lease is invalid"
            )
        for field_name in ("attempt", "fence"):
            if type(payload[field_name]) is not int:
                raise WorktreeLifecycleError(
                    "orphan lifecycle persisted field type is invalid: "
                    f"{field_name}"
                )
        if payload["attempt"] < 0 or payload["fence"] < 1:
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted attempt/fence is invalid"
            )
        for field_name in ("created_at", "updated_at", "expires_at"):
            if type(payload[field_name]) is not float:
                raise WorktreeLifecycleError(
                    "orphan lifecycle persisted timestamp type is invalid: "
                    f"{field_name}"
                )

        owner_payload = payload["owner"]
        if not isinstance(owner_payload, dict) or set(owner_payload) != {
            "pid",
            "start_time_ticks",
            "boot_id",
            "parent_pid",
        }:
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted owner shape is invalid"
            )
        for field_name in ("pid", "start_time_ticks", "parent_pid"):
            if type(owner_payload[field_name]) is not int:
                raise WorktreeLifecycleError(
                    "orphan lifecycle persisted owner type is invalid: "
                    f"{field_name}"
                )
        if (
            owner_payload["pid"] <= 0
            or owner_payload["start_time_ticks"] <= 0
            or owner_payload["parent_pid"] < 0
            or type(owner_payload["boot_id"]) is not str
            or not owner_payload["boot_id"]
        ):
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted owner identity is invalid"
            )
        try:
            record = WorkspaceLifecycleRecord.from_dict(payload)
        except (TypeError, ValueError, WorktreeLifecycleError) as exc:
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted record is invalid"
            ) from exc
        if payload != record.to_dict():
            raise WorktreeLifecycleError(
                "orphan lifecycle persisted record is noncanonical"
            )
        return record

    def require_exact_dead_owner(
        self,
        workspace: str | Path,
        *,
        expected_record_id: str,
        expected_fence: int,
        expected_lease_id: str,
        expected_task_id: str,
        expected_canonical_task_cid: str,
        expected_attempt: int,
        expected_branch: str,
        expected_merge_target: str,
        expected_repo_root: str,
        expected_state_dir: str,
        allow_terminal: bool = False,
    ) -> WorkspaceLifecycleRecord:
        """Read-only proof that an exact claim's owner is dead.

        This is a precheck only.  A later mutation must repeat it while holding
        the task-index and workspace-record guards.  Terminal claims are
        accepted only for exact cleanup retries, never adoption.
        """

        current = self._load_strict_workspace_record(workspace)
        expected_binding = self._normalized_binding(
            task_id=expected_task_id,
            canonical_task_cid=expected_canonical_task_cid,
            attempt=expected_attempt,
            workspace_path=workspace,
            branch=expected_branch,
            merge_target=expected_merge_target,
            repo_root=expected_repo_root,
            state_dir=expected_state_dir,
        )
        index_path = self.task_index_path_for(
            canonical_task_cid=expected_binding["canonical_task_cid"],
            task_id=expected_binding["task_id"],
            attempt=expected_binding["attempt"],
        )
        self._require_exact_dead_owner(
            current,
            expected_record_id=expected_record_id,
            expected_fence=expected_fence,
            expected_lease_id=expected_lease_id,
            expected_binding=expected_binding,
            index_path=index_path,
            allow_terminal=allow_terminal,
        )
        return current

    def adopt_dead_owner(
        self,
        workspace: str | Path,
        *,
        expected_record_id: str,
        expected_fence: int,
        expected_lease_id: str,
        expected_task_id: str,
        expected_canonical_task_cid: str,
        expected_attempt: int,
        expected_branch: str,
        expected_merge_target: str,
        expected_repo_root: str,
        expected_state_dir: str,
        lane_id: str,
    ) -> WorkspaceLifecycleRecord:
        """Atomically adopt an exactly bound claim from a dead owner.

        Lease expiry is intentionally not required.  The caller supplies the
        immutable record/fence/lease tuple and every execution binding, while
        this store independently proves the old process-birth identity dead
        under the same record and task-index guards used for acquisition.
        Alive or uninspectable owners and any identity/index mismatch leave
        both durable records byte-for-byte untouched.
        """

        expected_binding = self._normalized_binding(
            task_id=expected_task_id,
            canonical_task_cid=expected_canonical_task_cid,
            attempt=expected_attempt,
            workspace_path=workspace,
            branch=expected_branch,
            merge_target=expected_merge_target,
            repo_root=expected_repo_root,
            state_dir=expected_state_dir,
        )
        # Avoid materializing a guard for a caller-supplied mismatched task
        # identity.  The same proof is repeated under both authoritative locks.
        prechecked = self.require_exact_dead_owner(
            workspace,
            expected_record_id=expected_record_id,
            expected_fence=expected_fence,
            expected_lease_id=expected_lease_id,
            expected_task_id=expected_task_id,
            expected_canonical_task_cid=expected_canonical_task_cid,
            expected_attempt=expected_attempt,
            expected_branch=expected_branch,
            expected_merge_target=expected_merge_target,
            expected_repo_root=expected_repo_root,
            expected_state_dir=expected_state_dir,
        )
        normalized_workspace = expected_binding["workspace_path"]
        record_path = self.workspace_path_for(normalized_workspace)
        index_path = self.task_index_path_for(
            canonical_task_cid=expected_binding["canonical_task_cid"],
            task_id=expected_binding["task_id"],
            attempt=expected_binding["attempt"],
        )
        new_owner = current_process_birth(proc_root=self.proc_root)
        if (
            new_owner.pid != os.getpid()
            or new_owner.start_time_ticks <= 0
            or not new_owner.boot_id
            or owner_liveness(new_owner, proc_root=self.proc_root)
            is not OwnerLiveness.ALIVE
        ):
            raise OwnershipError(
                "replacement lifecycle owner identity is not provably alive"
            )
        new_lease = new_lease_id(
            seed=(
                f"adopt:{expected_task_id}:"
                f"{expected_canonical_task_cid}:{expected_attempt}:"
                f"{normalized_workspace}"
            )
        )
        if not new_lease or new_lease == expected_lease_id:
            raise OwnershipError(
                "replacement lifecycle lease was not freshly rotated"
            )

        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self._load_strict_workspace_record(
                    normalized_workspace
                )
                self._require_exact_dead_owner(
                    current,
                    expected_record_id=expected_record_id,
                    expected_fence=expected_fence,
                    expected_lease_id=expected_lease_id,
                    expected_binding=expected_binding,
                    index_path=index_path,
                    expected_owner=prechecked.owner,
                )

                now = float(self.clock())
                adopted = replace(
                    current,
                    lane_id=str(lane_id or ""),
                    state=WorkspaceLifecycleState.ACTIVE,
                    owner=new_owner,
                    lease_id=new_lease,
                    fence=int(current.fence) + 1,
                    updated_at=now,
                    expires_at=now + self.lease_seconds,
                    terminal_reason="",
                )
                _atomic_write_json(record_path, adopted.to_dict())
                _atomic_write_json(
                    index_path,
                    {
                        "schema": WORKTREE_LIFECYCLE_SCHEMA,
                        "workspace_path": adopted.workspace_path,
                        "record_id": adopted.record_id,
                        "task_id": adopted.task_id,
                        "canonical_task_cid": (
                            adopted.canonical_task_cid
                        ),
                        "attempt": adopted.attempt,
                        "fence": adopted.fence,
                        "lease_id": adopted.lease_id,
                        "state": adopted.state.value,
                    },
                )
                return adopted

    def finalize_exact_dead_owner(
        self,
        workspace: str | Path,
        *,
        expected_record_id: str,
        expected_fence: int,
        expected_lease_id: str,
        expected_owner: ProcessBirthIdentity,
        expected_task_id: str,
        expected_canonical_task_cid: str,
        expected_attempt: int,
        expected_branch: str,
        expected_merge_target: str,
        expected_repo_root: str,
        expected_state_dir: str,
        reason: str,
    ) -> WorkspaceLifecycleRecord:
        """Terminalize and delete one phase-pinned dead-owner authority.

        Both durable records are authenticated again while holding the
        task-index and workspace guards.  Unlike generic owner transitions,
        this operation also pins the process-birth identity captured by the
        caller's earlier quiescence proof.
        """

        expected_binding = self._normalized_binding(
            task_id=expected_task_id,
            canonical_task_cid=expected_canonical_task_cid,
            attempt=expected_attempt,
            workspace_path=workspace,
            branch=expected_branch,
            merge_target=expected_merge_target,
            repo_root=expected_repo_root,
            state_dir=expected_state_dir,
        )
        normalized_workspace = expected_binding["workspace_path"]
        record_path = self.workspace_path_for(normalized_workspace)
        index_path = self.task_index_path_for(
            canonical_task_cid=expected_binding["canonical_task_cid"],
            task_id=expected_binding["task_id"],
            attempt=expected_binding["attempt"],
        )
        with serialized_lock_update(index_path):
            with serialized_lock_update(record_path):
                current = self._load_strict_workspace_record(
                    normalized_workspace
                )
                self._require_exact_dead_owner(
                    current,
                    expected_record_id=expected_record_id,
                    expected_fence=expected_fence,
                    expected_lease_id=expected_lease_id,
                    expected_binding=expected_binding,
                    index_path=index_path,
                    allow_terminal=True,
                    expected_owner=expected_owner,
                )
                terminal = current
                if current.is_nonterminal:
                    now = float(self.clock())
                    terminal = replace(
                        current,
                        state=WorkspaceLifecycleState.TERMINAL,
                        fence=int(current.fence) + 1,
                        updated_at=now,
                        expires_at=now,
                        terminal_reason=str(reason or "finalized"),
                    )
                    _atomic_write_json(record_path, terminal.to_dict())
                    _atomic_write_json(
                        index_path,
                        {
                            "schema": WORKTREE_LIFECYCLE_SCHEMA,
                            "workspace_path": terminal.workspace_path,
                            "record_id": terminal.record_id,
                            "task_id": terminal.task_id,
                            "canonical_task_cid": (
                                terminal.canonical_task_cid
                            ),
                            "attempt": terminal.attempt,
                            "fence": terminal.fence,
                            "lease_id": terminal.lease_id,
                            "state": terminal.state.value,
                        },
                    )
                record_path.unlink()
                index_path.unlink()
                return terminal

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
        with serialized_lock_update(record_path):
            current = self.load_workspace(workspace)
            if current is None:
                raise WorktreeLifecycleError("lifecycle record missing")
            self._require_owner(
                current, lease_id=lease_id, expected_fence=expected_fence
            )
            if current.is_terminal and new_state is not WorkspaceLifecycleState.TERMINAL:
                raise WorktreeLifecycleError("cannot revive a terminal lifecycle record")
            now = float(self.clock())
            expires_at = (
                now + self.lease_seconds if renew_lease else float(current.expires_at)
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
            index_path = self.task_index_path_for(
                canonical_task_cid=updated.canonical_task_cid,
                task_id=updated.task_id,
                attempt=updated.attempt,
            )
            _atomic_write_json(
                index_path,
                {
                    "schema": WORKTREE_LIFECYCLE_SCHEMA,
                    "workspace_path": updated.workspace_path,
                    "record_id": updated.record_id,
                    "task_id": updated.task_id,
                    "canonical_task_cid": updated.canonical_task_cid,
                    "attempt": updated.attempt,
                    "fence": updated.fence,
                    "lease_id": updated.lease_id,
                    "state": updated.state.value,
                },
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

        def _rebind_body() -> WorkspaceLifecycleRecord:
            current = self.load_workspace(old_normalized)
            if current is None:
                raise WorktreeLifecycleError("lifecycle record missing")
            self._require_owner(
                current, lease_id=lease_id, expected_fence=expected_fence
            )
            if current.is_terminal:
                raise WorktreeLifecycleError(
                    "cannot rebind a terminal lifecycle record"
                )
            existing_new = self.load_workspace(new_normalized)
            if existing_new is not None and existing_new.is_nonterminal:
                other_live = owner_liveness(
                    existing_new.owner, proc_root=self.proc_root
                )
                if other_live is not OwnerLiveness.DEAD:
                    raise DuplicateAttemptError(
                        "target workspace already has a nonterminal claim"
                    )
                now = float(self.clock())
                if now < float(existing_new.expires_at):
                    raise DuplicateAttemptError(
                        "target workspace claim lease has not expired"
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
            index_path = self.task_index_path_for(
                canonical_task_cid=updated.canonical_task_cid,
                task_id=updated.task_id,
                attempt=updated.attempt,
            )
            _atomic_write_json(
                index_path,
                {
                    "schema": WORKTREE_LIFECYCLE_SCHEMA,
                    "workspace_path": new_normalized,
                    "record_id": updated.record_id,
                    "task_id": updated.task_id,
                    "canonical_task_cid": updated.canonical_task_cid,
                    "attempt": updated.attempt,
                    "fence": updated.fence,
                    "lease_id": updated.lease_id,
                    "state": updated.state.value,
                },
            )
            if old_path != new_path:
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
            return updated

        # Lock both paths in a stable order.  When filenames collide, one
        # advisory guard is enough (non-recursive flock would deadlock).
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
        with serialized_lock_update(record_path):
            current = self.load_workspace(workspace)
            if current is None:
                raise WorktreeLifecycleError("lifecycle record missing")
            self._require_owner(
                current, lease_id=lease_id, expected_fence=expected_fence
            )
            if current.is_terminal:
                raise WorktreeLifecycleError("cannot renew a terminal lifecycle record")
            now = float(self.clock())
            updated = replace(
                current,
                fence=int(current.fence) + 1,
                updated_at=now,
                expires_at=now + self.lease_seconds,
            )
            _atomic_write_json(record_path, updated.to_dict())
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
        """Advance fence and mark terminal when owner is dead and lease expired.

        Returns the terminal record on success, or None if reclamation was
        refused (live owner, unexpired lease, missing record, etc.).
        """

        clock_now = float(self.clock() if now is None else now)
        record_path = self.workspace_path_for(workspace)
        with serialized_lock_update(record_path):
            current = self.load_workspace(workspace)
            if current is None:
                return None
            if current.is_terminal:
                return current
            liveness = owner_liveness(current.owner, proc_root=self.proc_root)
            if liveness is not OwnerLiveness.DEAD:
                return None
            if clock_now < float(current.expires_at):
                return None
            updated = replace(
                current,
                state=WorkspaceLifecycleState.TERMINAL,
                fence=int(current.fence) + 1,
                updated_at=clock_now,
                terminal_reason=str(reason or "stale_reclamation"),
                lease_id=reclaimer_lease_id or current.lease_id,
            )
            _atomic_write_json(record_path, updated.to_dict())
            return updated

    def compare_and_delete(
        self,
        workspace: str | Path,
        *,
        expected_fence: int,
        lease_id: str = "",
    ) -> bool:
        """Remove the lifecycle record only when fence (and optional lease) match."""

        record_path = self.workspace_path_for(workspace)
        with serialized_lock_update(record_path):
            current = self.load_workspace(workspace)
            if current is None:
                return True
            if int(current.fence) != int(expected_fence):
                return False
            if lease_id and str(current.lease_id) != str(lease_id):
                return False
            index_path = self.task_index_path_for(
                canonical_task_cid=current.canonical_task_cid,
                task_id=current.task_id,
                attempt=current.attempt,
            )
            try:
                record_path.unlink()
            except FileNotFoundError:
                pass
            try:
                index_path.unlink()
            except FileNotFoundError:
                pass
            return True

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
