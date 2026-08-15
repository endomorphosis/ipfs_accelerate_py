"""Incremental semantic-state session coordinator (SCH-012).

Interface: ``SemanticStateSession@1`` / sch/session@1

Coordinates repeated local runs over existing ports only:

* Watch notifications debounce solely to schedule a canonical scan.
* Concurrent equal snapshot CIDs coalesce; work never duplicates.
* Accepted attempts serialize by repository fence so stale callbacks
  cannot publish or overwrite a root.
* Restart replays ``runtime.event_log`` pages, verifies immutable
  artifacts, reconciles WAL/root state, and resumes only nonterminal
  attempts whose lease/fence still matches.
* Corrupt or truncated event tails recover when safe, otherwise fail closed.
* Explicit shutdown cancels and joins every owned worker.

Importing this module starts no threads, processes, databases, or network
calls. Event rows, mtimes, and task-queue entries are never semantic truth;
only fenced scans and generation-bearing root CAS advance state.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    HarnessMode,
    RootRef,
    _bool,
    _nonneg_int,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableSemanticStatePort,
    RootConflict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)

# ---------------------------------------------------------------------------
# Interface pins
# ---------------------------------------------------------------------------

SESSION_INTERFACE = "SemanticStateSession@1"
SESSION_SCHEMA = "ipfs-accelerate.semantic-state-session@1"
ADAPTER_ID = "ipfs-accelerate.semantic-state.session"
BUNDLE_ID = "sch/session@1"

_EVENT_WATCH = "session_watch_notification"
_EVENT_SCAN_SCHEDULED = "session_scan_scheduled"
_EVENT_SCAN_COALESCED = "session_scan_coalesced"
_EVENT_SCAN_STARTED = "session_scan_started"
_EVENT_SCAN_COMPLETED = "session_scan_completed"
_EVENT_SCAN_CANCELLED = "session_scan_cancelled"
_EVENT_TRANSITION_ACCEPTED = "session_transition_accepted"
_EVENT_TRANSITION_REJECTED = "session_transition_rejected"
_EVENT_TRANSITION_CANDIDATE = "session_transition_candidate"
_EVENT_ROOT_PUBLISH_DENIED = "session_root_publish_denied"
_EVENT_RESTART = "session_restart"
_EVENT_REPLAY = "session_replayed"
_EVENT_RECOVERY = "session_event_recovery"
_EVENT_SHUTDOWN = "session_shutdown"
_EVENT_FAILED_CLOSED = "session_failed_closed"

_DEFAULT_DEBOUNCE_MS = 50
_DEFAULT_FENCE_TTL_MS = 300_000
_DEFAULT_JOIN_TIMEOUT_S = 5.0
_MAX_DIAGNOSTIC = 512
_MAX_COALESCED = 1_000_000
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SessionError(HarnessError):
    """Closed session contract or coordination failure."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "session_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.retryable = bool(retryable)


class SessionFailedClosed(SessionError):
    """Corrupt state or recovery boundary that must not continue."""

    def __init__(self, message: str, *, reason_code: str = "failed_closed") -> None:
        super().__init__(message, reason_code=reason_code, retryable=False)


class SessionShutdownError(SessionError):
    """Operation rejected because the session is shutting down or stopped."""

    def __init__(self, message: str = "session is shut down") -> None:
        super().__init__(message, reason_code="session_shutdown", retryable=False)


class SessionRootPublishDenied(SessionError):
    """Fence/verification gate refused root publication."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "root_publish_denied",
    ) -> None:
        super().__init__(message, reason_code=reason_code, retryable=False)


class SessionRootConflict(SessionError):
    """Root CAS lost; the prior RootRef was left unchanged."""

    def __init__(self, message: str, *, current_root: RootRef | None = None) -> None:
        super().__init__(message, reason_code="root_conflict", retryable=True)
        self.current_root = current_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clip(text: str, *, limit: int = _MAX_DIAGNOSTIC) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 3)] + "..."


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return validate_opaque_cid(value, name)


# ---------------------------------------------------------------------------
# Status / policy records
# ---------------------------------------------------------------------------


class SessionPhase(str, Enum):
    """Lifecycle phase for an incremental semantic-state session."""

    IDLE = "idle"
    WATCHING = "watching"
    SCANNING = "scanning"
    RESTARTING = "restarting"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED_CLOSED = "failed_closed"


@dataclass(frozen=True)
class SessionPolicy:
    """Closed policy for debounce, journaling, fencing, and recovery."""

    repository_id: str
    event_log_path: str | Path | None = None
    checkpoint_path: str | Path | None = None
    mode: str = HarnessMode.DEVELOPMENT.value
    debounce_ms: int = _DEFAULT_DEBOUNCE_MS
    fence_ttl_ms: int = _DEFAULT_FENCE_TTL_MS
    max_pending_snapshots: int = 64
    fail_closed_on_corrupt_log: bool = True
    join_timeout_s: float = _DEFAULT_JOIN_TIMEOUT_S
    worker_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        mode = _text(self.mode, "mode")
        try:
            mode = HarnessMode(mode).value
        except ValueError as exc:
            raise SessionError(
                f"mode has unsupported value {self.mode!r}",
                reason_code="invalid_policy",
            ) from exc
        object.__setattr__(self, "mode", mode)
        debounce = int(self.debounce_ms)
        if debounce < 0:
            raise SessionError("debounce_ms must be nonnegative", reason_code="invalid_policy")
        object.__setattr__(self, "debounce_ms", debounce)
        fence_ttl = int(self.fence_ttl_ms)
        if fence_ttl < 1:
            raise SessionError("fence_ttl_ms must be positive", reason_code="invalid_policy")
        object.__setattr__(self, "fence_ttl_ms", fence_ttl)
        max_pending = int(self.max_pending_snapshots)
        if max_pending < 1:
            raise SessionError(
                "max_pending_snapshots must be positive", reason_code="invalid_policy"
            )
        object.__setattr__(self, "max_pending_snapshots", max_pending)
        join_timeout = float(self.join_timeout_s)
        if join_timeout <= 0:
            raise SessionError(
                "join_timeout_s must be positive", reason_code="invalid_policy"
            )
        object.__setattr__(self, "join_timeout_s", join_timeout)
        object.__setattr__(
            self,
            "fail_closed_on_corrupt_log",
            _bool(self.fail_closed_on_corrupt_log, "fail_closed_on_corrupt_log"),
        )
        object.__setattr__(
            self, "worker_enabled", _bool(self.worker_enabled, "worker_enabled")
        )
        if self.event_log_path is not None:
            object.__setattr__(self, "event_log_path", Path(self.event_log_path))
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "event_log_path": (
                None if self.event_log_path is None else str(self.event_log_path)
            ),
            "checkpoint_path": (
                None if self.checkpoint_path is None else str(self.checkpoint_path)
            ),
            "mode": self.mode,
            "debounce_ms": self.debounce_ms,
            "fence_ttl_ms": self.fence_ttl_ms,
            "max_pending_snapshots": self.max_pending_snapshots,
            "fail_closed_on_corrupt_log": self.fail_closed_on_corrupt_log,
            "join_timeout_s": self.join_timeout_s,
            "worker_enabled": self.worker_enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionPolicy":
        if not isinstance(data, Mapping):
            raise SessionError("SessionPolicy must be an object", reason_code="invalid_policy")
        return cls(
            repository_id=str(data.get("repository_id") or ""),
            event_log_path=data.get("event_log_path"),
            checkpoint_path=data.get("checkpoint_path"),
            mode=str(data.get("mode") or HarnessMode.DEVELOPMENT.value),
            debounce_ms=int(data.get("debounce_ms", _DEFAULT_DEBOUNCE_MS)),
            fence_ttl_ms=int(data.get("fence_ttl_ms", _DEFAULT_FENCE_TTL_MS)),
            max_pending_snapshots=int(data.get("max_pending_snapshots", 64)),
            fail_closed_on_corrupt_log=bool(
                data.get("fail_closed_on_corrupt_log", True)
            ),
            join_timeout_s=float(data.get("join_timeout_s", _DEFAULT_JOIN_TIMEOUT_S)),
            worker_enabled=bool(data.get("worker_enabled", False)),
        )


@dataclass(frozen=True)
class SessionStatus:
    """Deterministic status snapshot for CLI/status surfaces."""

    phase: str
    repository_id: str
    mode: str
    current_root: RootRef | None
    pending_snapshot_cids: tuple[str, ...]
    active_attempt_id: str | None
    active_snapshot_cid: str | None
    accepted_generation: int
    coalesced_watch_count: int
    scans_completed: int
    scans_coalesced: int
    cursor_position: int
    failed_closed: bool
    shutdown: bool
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "repository_id": self.repository_id,
            "mode": self.mode,
            "current_root": (
                None if self.current_root is None else self.current_root.to_dict()
            ),
            "pending_snapshot_cids": list(self.pending_snapshot_cids),
            "active_attempt_id": self.active_attempt_id,
            "active_snapshot_cid": self.active_snapshot_cid,
            "accepted_generation": self.accepted_generation,
            "coalesced_watch_count": self.coalesced_watch_count,
            "scans_completed": self.scans_completed,
            "scans_coalesced": self.scans_coalesced,
            "cursor_position": self.cursor_position,
            "failed_closed": self.failed_closed,
            "shutdown": self.shutdown,
            "diagnostic": self.diagnostic,
            "interface": SESSION_INTERFACE,
            "schema": SESSION_SCHEMA,
            "board_namespace": BOARD_NAMESPACE,
        }


@dataclass(frozen=True)
class WatchAck:
    """Acknowledgement that a watch notification was admitted or coalesced."""

    snapshot_cid: str
    scheduled: bool
    coalesced: bool
    pending_count: int
    attempt_id: str | None
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_cid": self.snapshot_cid,
            "scheduled": self.scheduled,
            "coalesced": self.coalesced,
            "pending_count": self.pending_count,
            "attempt_id": self.attempt_id,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class ScanResult:
    """Bounded result returned by a session scan executor."""

    snapshot_cid: str
    attempt_id: str
    status: str
    output_artifact_cids: tuple[str, ...] = ()
    new_root_cid: str | None = None
    verified: bool = False
    diagnostic: str = ""
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_cid": self.snapshot_cid,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "output_artifact_cids": list(self.output_artifact_cids),
            "new_root_cid": self.new_root_cid,
            "verified": self.verified,
            "diagnostic": self.diagnostic,
            "reason_codes": list(self.reason_codes),
        }


class ScanExecutor(Protocol):
    """Deterministic local canonical scan scheduled by watch notifications."""

    def __call__(
        self,
        *,
        repository_id: str,
        snapshot_cid: str,
        attempt_id: str,
        fencing_token: int,
        cancellation: CancellationToken,
    ) -> Mapping[str, Any] | ScanResult:
        ...


@dataclass
class _PendingScan:
    snapshot_cid: str
    attempt_id: str
    coalesce_count: int = 1
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    sources: list[str] = field(default_factory=list)


@dataclass
class _SessionFence:
    attempt_id: str
    fencing_token: int
    lease_id: str
    snapshot_cid: str
    expires_at_ms: int
    cancelled: bool = False
    terminal: bool = False
    verified: bool = False
    accepted: bool = False
    new_root_cid: str | None = None

    def is_live(self, now_ms: int) -> bool:
        return (
            not self.cancelled
            and not self.terminal
            and int(now_ms) < int(self.expires_at_ms)
        )


@dataclass
class _AcceptedTransition:
    attempt_id: str
    fencing_token: int
    snapshot_cid: str
    root: RootRef
    event_sequence: int | None = None


# ---------------------------------------------------------------------------
# Session coordinator
# ---------------------------------------------------------------------------


class SemanticStateSession:
    """Restartable watcher/session coordinator for one repository.

    Watch callbacks only schedule a canonical scan. Equal snapshot CIDs
    coalesce under one attempt. Root publication requires a live fence,
    matching token, and an explicit verified acceptance gate.
    """

    interface = SESSION_INTERFACE
    schema = SESSION_SCHEMA

    def __init__(
        self,
        policy: SessionPolicy | Mapping[str, Any],
        *,
        durable_port: DurableSemanticStatePort | None = None,
        scan_executor: ScanExecutor | Callable[..., Mapping[str, Any] | ScanResult] | None = None,
        clock_ms: Callable[[], int] | None = None,
        initial_root: RootRef | None = None,
    ) -> None:
        self._policy = (
            policy if isinstance(policy, SessionPolicy) else SessionPolicy.from_dict(policy)
        )
        self._durable = durable_port
        self._scan_executor = scan_executor
        self._clock_ms = clock_ms or _now_ms
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._phase = SessionPhase.IDLE
        self._shutdown = False
        self._failed_closed = False
        self._diagnostic = ""
        self._pending: dict[str, _PendingScan] = {}
        self._fences: dict[str, _SessionFence] = {}
        self._active_attempt_id: str | None = None
        self._active_snapshot_cid: str | None = None
        self._accepted: list[_AcceptedTransition] = []
        self._current_root: RootRef | None = initial_root
        self._cursor_position = 0
        self._next_fence = 1
        self._coalesced_watch_count = 0
        self._scans_completed = 0
        self._scans_coalesced = 0
        self._cancellations: dict[str, CancellationToken] = {}
        self._owned_threads: list[threading.Thread] = []
        self._worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._last_scan_results: list[ScanResult] = []
        # Cold construction only: no event I/O, recover, or worker start here.
        if self._current_root is None and self._durable is not None:
            # Read is optional and local; never starts a daemon.
            try:
                self._current_root = self._durable.read_root(self._policy.repository_id)
            except Exception:
                self._current_root = None

    # -- properties --------------------------------------------------------

    @property
    def policy(self) -> SessionPolicy:
        return self._policy

    @property
    def repository_id(self) -> str:
        return self._policy.repository_id

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    @property
    def is_failed_closed(self) -> bool:
        with self._lock:
            return self._failed_closed

    # -- status ------------------------------------------------------------

    def status(self) -> SessionStatus:
        with self._lock:
            return SessionStatus(
                phase=self._phase.value,
                repository_id=self._policy.repository_id,
                mode=self._policy.mode,
                current_root=self._current_root,
                pending_snapshot_cids=tuple(sorted(self._pending.keys())),
                active_attempt_id=self._active_attempt_id,
                active_snapshot_cid=self._active_snapshot_cid,
                accepted_generation=(
                    0
                    if self._current_root is None
                    else int(self._current_root.generation)
                ),
                coalesced_watch_count=self._coalesced_watch_count,
                scans_completed=self._scans_completed,
                scans_coalesced=self._scans_coalesced,
                cursor_position=self._cursor_position,
                failed_closed=self._failed_closed,
                shutdown=self._shutdown,
                diagnostic=self._diagnostic,
            )

    # -- watch admission ---------------------------------------------------

    def notify_watch(
        self,
        snapshot_cid: str,
        *,
        source: str = "watcher",
        metadata: Mapping[str, Any] | None = None,
    ) -> WatchAck:
        """Admit a watch notification that only schedules a canonical scan.

        Concurrent equal ``snapshot_cid`` values coalesce into one pending
        attempt. Notifications never become state and never publish a root.
        """

        cid = validate_opaque_cid(snapshot_cid, "snapshot_cid")
        source_text = _clip(_text(source, "source") if source else "watcher")
        with self._condition:
            self._ensure_open_for_work()
            now = self._clock_ms()
            self._journal(
                _EVENT_WATCH,
                {
                    "repository_id": self._policy.repository_id,
                    "snapshot_cid": cid,
                    "source": source_text,
                    "metadata_keys": sorted(
                        str(k) for k in dict(metadata or {}).keys()
                    )[:32],
                },
            )
            existing = self._pending.get(cid)
            active_match = (
                self._active_snapshot_cid == cid and self._active_attempt_id is not None
            )
            if existing is not None or active_match:
                if existing is not None:
                    existing.coalesce_count = min(
                        _MAX_COALESCED, existing.coalesce_count + 1
                    )
                    existing.last_seen_ms = now
                    if source_text not in existing.sources:
                        existing.sources.append(source_text)
                    attempt_id = existing.attempt_id
                    pending_count = existing.coalesce_count
                else:
                    # Coalesce onto the in-flight equal snapshot: do not start
                    # a second scan for the same CID.
                    attempt_id = self._active_attempt_id
                    pending_count = 1
                self._coalesced_watch_count += 1
                self._scans_coalesced += 1
                sequence = self._journal(
                    _EVENT_SCAN_COALESCED,
                    {
                        "repository_id": self._policy.repository_id,
                        "snapshot_cid": cid,
                        "attempt_id": attempt_id,
                        "source": source_text,
                    },
                )
                self._condition.notify_all()
                return WatchAck(
                    snapshot_cid=cid,
                    scheduled=False,
                    coalesced=True,
                    pending_count=pending_count,
                    attempt_id=attempt_id,
                    sequence=sequence,
                )

            if len(self._pending) >= self._policy.max_pending_snapshots:
                raise SessionError(
                    "pending snapshot capacity exhausted",
                    reason_code="pending_capacity_exhausted",
                    retryable=True,
                )

            attempt_id = f"scan:{cid[:16]}:{uuid.uuid4().hex[:12]}"
            pending = _PendingScan(
                snapshot_cid=cid,
                attempt_id=attempt_id,
                coalesce_count=1,
                first_seen_ms=now,
                last_seen_ms=now,
                sources=[source_text],
            )
            self._pending[cid] = pending
            if self._phase in {SessionPhase.IDLE, SessionPhase.WATCHING}:
                self._phase = SessionPhase.WATCHING
            sequence = self._journal(
                _EVENT_SCAN_SCHEDULED,
                {
                    "repository_id": self._policy.repository_id,
                    "snapshot_cid": cid,
                    "attempt_id": attempt_id,
                    "source": source_text,
                },
            )
            self._condition.notify_all()
            return WatchAck(
                snapshot_cid=cid,
                scheduled=True,
                coalesced=False,
                pending_count=1,
                attempt_id=attempt_id,
                sequence=sequence,
            )

    # -- drain / process ---------------------------------------------------

    def process_pending(self, *, limit: int = 1) -> list[ScanResult]:
        """Synchronously run up to ``limit`` pending coalesced scans.

        Preferred by hermetic tests. Background workers call the same path.
        """

        if limit < 1:
            raise SessionError("limit must be positive", reason_code="invalid_request")
        results: list[ScanResult] = []
        for _ in range(limit):
            result = self._process_one_pending()
            if result is None:
                break
            results.append(result)
        return results

    def drain(self, *, max_scans: int = 64) -> list[ScanResult]:
        """Process all currently pending scans (bounded)."""

        return self.process_pending(limit=max(1, int(max_scans)))

    def _process_one_pending(self) -> ScanResult | None:
        with self._condition:
            self._ensure_open_for_work()
            if not self._pending:
                return None
            # Deterministic order by snapshot CID.
            snapshot_cid = sorted(self._pending.keys())[0]
            pending = self._pending.pop(snapshot_cid)
            attempt_id = pending.attempt_id
            now = self._clock_ms()
            token = self._next_fence
            self._next_fence += 1
            fence = _SessionFence(
                attempt_id=attempt_id,
                fencing_token=token,
                lease_id=f"lease:{attempt_id}",
                snapshot_cid=snapshot_cid,
                expires_at_ms=now + self._policy.fence_ttl_ms,
            )
            self._fences[attempt_id] = fence
            cancellation = CancellationToken(attempt_id)
            self._cancellations[attempt_id] = cancellation
            self._active_attempt_id = attempt_id
            self._active_snapshot_cid = snapshot_cid
            self._phase = SessionPhase.SCANNING
            self._journal(
                _EVENT_SCAN_STARTED,
                {
                    "repository_id": self._policy.repository_id,
                    "snapshot_cid": snapshot_cid,
                    "attempt_id": attempt_id,
                    "fencing_token": token,
                    "coalesce_count": pending.coalesce_count,
                },
            )

        # Execute outside the repository admission lock section that selects
        # work, but still under a per-attempt serialization for root publish
        # via publish_root / accept_transition.
        try:
            if cancellation.is_cancelled() or self._shutdown:
                result = ScanResult(
                    snapshot_cid=snapshot_cid,
                    attempt_id=attempt_id,
                    status="cancelled",
                    diagnostic="cancelled before scan",
                    reason_codes=("cancelled",),
                )
            elif self._scan_executor is None:
                result = ScanResult(
                    snapshot_cid=snapshot_cid,
                    attempt_id=attempt_id,
                    status="completed",
                    diagnostic="scan scheduled (no executor)",
                    reason_codes=("scan_noop",),
                )
            else:
                raw = self._scan_executor(
                    repository_id=self._policy.repository_id,
                    snapshot_cid=snapshot_cid,
                    attempt_id=attempt_id,
                    fencing_token=token,
                    cancellation=cancellation,
                )
                result = self._coerce_scan_result(
                    raw,
                    snapshot_cid=snapshot_cid,
                    attempt_id=attempt_id,
                )
                if cancellation.is_cancelled() or self._shutdown:
                    result = ScanResult(
                        snapshot_cid=snapshot_cid,
                        attempt_id=attempt_id,
                        status="cancelled",
                        diagnostic=cancellation.reason or "cancelled",
                        reason_codes=("cancelled",),
                        output_artifact_cids=result.output_artifact_cids,
                        new_root_cid=result.new_root_cid,
                        verified=False,
                    )
        except Exception as exc:
            result = ScanResult(
                snapshot_cid=snapshot_cid,
                attempt_id=attempt_id,
                status="failed",
                diagnostic=_clip(str(exc)),
                reason_codes=("scan_failed",),
            )

        with self._condition:
            fence = self._fences.get(attempt_id)
            if fence is not None:
                fence.terminal = True
                if result.status == "cancelled":
                    fence.cancelled = True
                fence.verified = bool(result.verified)
                fence.new_root_cid = result.new_root_cid
            self._active_attempt_id = None
            self._active_snapshot_cid = None
            self._scans_completed += 1
            self._last_scan_results.append(result)
            if len(self._last_scan_results) > 64:
                self._last_scan_results = self._last_scan_results[-64:]
            event_type = (
                _EVENT_SCAN_CANCELLED
                if result.status == "cancelled"
                else _EVENT_SCAN_COMPLETED
            )
            self._journal(
                event_type,
                {
                    "repository_id": self._policy.repository_id,
                    "snapshot_cid": snapshot_cid,
                    "attempt_id": attempt_id,
                    "status": result.status,
                    "verified": bool(result.verified),
                    "new_root_cid": result.new_root_cid,
                    "output_artifact_cids": list(result.output_artifact_cids),
                    "reason_codes": list(result.reason_codes),
                    "diagnostic": result.diagnostic,
                },
            )
            if self._phase is SessionPhase.SCANNING:
                self._phase = (
                    SessionPhase.WATCHING if self._pending else SessionPhase.IDLE
                )
            self._condition.notify_all()
        return result

    def _coerce_scan_result(
        self,
        raw: Mapping[str, Any] | ScanResult,
        *,
        snapshot_cid: str,
        attempt_id: str,
    ) -> ScanResult:
        if isinstance(raw, ScanResult):
            return raw
        if not isinstance(raw, Mapping):
            raise SessionError("scan executor must return a mapping or ScanResult")
        status = str(raw.get("status") or "completed")
        outputs = raw.get("output_artifact_cids") or ()
        if isinstance(outputs, (str, bytes)):
            raise SessionError("output_artifact_cids must be a sequence")
        output_cids = tuple(
            validate_opaque_cid(item, "output_artifact_cids") for item in outputs
        )
        new_root = _optional_cid(raw.get("new_root_cid"), "new_root_cid")
        verified = bool(raw.get("verified", False))
        reasons = raw.get("reason_codes") or ()
        reason_codes = tuple(str(item) for item in reasons if str(item).strip())
        return ScanResult(
            snapshot_cid=snapshot_cid,
            attempt_id=attempt_id,
            status=status,
            output_artifact_cids=output_cids,
            new_root_cid=new_root,
            verified=verified,
            diagnostic=_clip(str(raw.get("diagnostic") or "")),
            reason_codes=reason_codes,
        )

    # -- fenced root publication ------------------------------------------

    def accept_transition(
        self,
        *,
        attempt_id: str,
        fencing_token: int,
        new_root_cid: str,
        expected: RootRef | None | object = _UNSET,
        verified: bool = True,
        snapshot_cid: str | None = None,
    ) -> RootRef:
        """Accept and publish a root only under a live matching fence.

        Unverified transitions are journaled as candidates and never advance
        the current ``RootRef``. Stale or expired fences are denied.

        ``expected`` defaults to the session's current root. Pass an explicit
        ``None`` for the empty-root bootstrap CAS token.
        """

        attempt_id = _text(attempt_id, "attempt_id")
        token = _nonneg_int(fencing_token, "fencing_token")
        if token < 1:
            raise SessionError(
                "fencing_token must be positive", reason_code="invalid_request"
            )
        root_cid = validate_opaque_cid(new_root_cid, "new_root_cid")
        with self._lock:
            self._ensure_open_for_work()
            fence = self._fences.get(attempt_id)
            now = self._clock_ms()
            if fence is None:
                self._deny_publish(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    reason="unknown_attempt",
                    new_root_cid=root_cid,
                )
                raise SessionRootPublishDenied(
                    "unknown attempt cannot publish",
                    reason_code="unknown_attempt",
                )
            if fence.cancelled:
                self._deny_publish(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    reason="fence_cancelled",
                    new_root_cid=root_cid,
                )
                raise SessionRootPublishDenied(
                    "cancelled fence cannot publish",
                    reason_code="fence_cancelled",
                )
            if int(now) >= int(fence.expires_at_ms):
                self._deny_publish(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    reason="fence_expired",
                    new_root_cid=root_cid,
                )
                raise SessionRootPublishDenied(
                    "expired fence cannot publish",
                    reason_code="fence_expired",
                )
            if int(fence.fencing_token) != int(token):
                self._deny_publish(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    reason="stale_fencing_token",
                    new_root_cid=root_cid,
                )
                raise SessionRootPublishDenied(
                    "stale fencing token cannot publish",
                    reason_code="stale_fencing_token",
                )
            if not verified:
                snap = snapshot_cid or fence.snapshot_cid
                self._journal(
                    _EVENT_TRANSITION_CANDIDATE,
                    {
                        "repository_id": self._policy.repository_id,
                        "attempt_id": attempt_id,
                        "fencing_token": token,
                        "snapshot_cid": snap,
                        "new_root_cid": root_cid,
                        "verified": False,
                    },
                )
                raise SessionRootPublishDenied(
                    "unverified transition cannot publish a root",
                    reason_code="unverified_transition",
                )

            if expected is _UNSET:
                expected_root = self._current_root
            else:
                expected_root = expected  # type: ignore[assignment]
            if self._durable is None:
                # Hermetic path: still serialize and record acceptance without
                # an injected durable port. Generation-bearing expected tokens
                # mirror DurableSemanticStatePort CAS rules.
                if expected_root is None:
                    if self._current_root is not None:
                        raise SessionRootConflict(
                            "root CAS conflict: expected empty root is stale",
                            current_root=self._current_root,
                        )
                    published = RootRef(root_cid=root_cid, generation=1)
                else:
                    if (
                        self._current_root is None
                        or self._current_root.root_cid != expected_root.root_cid
                        or self._current_root.generation != expected_root.generation
                    ):
                        raise SessionRootConflict(
                            "root CAS conflict: expected token is stale",
                            current_root=self._current_root,
                        )
                    published = RootRef(
                        root_cid=root_cid,
                        generation=int(expected_root.generation) + 1,
                    )
            else:
                try:
                    published = self._durable.compare_and_swap_root(
                        self._policy.repository_id,
                        expected_root,
                        root_cid,
                    )
                except RootConflict as exc:
                    current = None
                    try:
                        current = self._durable.read_root(self._policy.repository_id)
                    except Exception:
                        current = self._current_root
                    self._journal(
                        _EVENT_TRANSITION_REJECTED,
                        {
                            "repository_id": self._policy.repository_id,
                            "attempt_id": attempt_id,
                            "fencing_token": token,
                            "reason": "root_conflict",
                            "new_root_cid": root_cid,
                        },
                    )
                    raise SessionRootConflict(
                        str(exc), current_root=current
                    ) from exc

            fence.accepted = True
            fence.verified = True
            fence.terminal = True
            fence.new_root_cid = root_cid
            self._current_root = published
            sequence = self._journal(
                _EVENT_TRANSITION_ACCEPTED,
                {
                    "repository_id": self._policy.repository_id,
                    "attempt_id": attempt_id,
                    "fencing_token": token,
                    "snapshot_cid": snapshot_cid or fence.snapshot_cid,
                    "root_cid": published.root_cid,
                    "generation": published.generation,
                    "verified": True,
                },
            )
            self._accepted.append(
                _AcceptedTransition(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    snapshot_cid=snapshot_cid or fence.snapshot_cid,
                    root=published,
                    event_sequence=sequence,
                )
            )
            self._checkpoint_cursor()
            return published

    def publish_root(
        self,
        *,
        attempt_id: str,
        fencing_token: int,
        new_root_cid: str,
        expected: RootRef | None | object = _UNSET,
        verified: bool = True,
    ) -> RootRef:
        """Alias for :meth:`accept_transition` (CLI/harness wording)."""

        return self.accept_transition(
            attempt_id=attempt_id,
            fencing_token=fencing_token,
            new_root_cid=new_root_cid,
            expected=expected,
            verified=verified,
        )

    def _deny_publish(
        self,
        *,
        attempt_id: str,
        fencing_token: int,
        reason: str,
        new_root_cid: str | None,
    ) -> None:
        self._journal(
            _EVENT_ROOT_PUBLISH_DENIED,
            {
                "repository_id": self._policy.repository_id,
                "attempt_id": attempt_id,
                "fencing_token": fencing_token,
                "reason": reason,
                "new_root_cid": new_root_cid,
            },
        )

    # -- restart / replay --------------------------------------------------

    def restart(self) -> SessionStatus:
        """Recover event log, replay pages, reconcile roots, resume safe work.

        Guarantees:

        * Accepted transitions already journaled are restored into memory and
          the current root is re-read from the durable port when present.
        * Candidate / unverified transitions never publish a root.
        * Nonterminal attempts resume only when their fence is still live.
        * Corrupt or truncated event tails recover or fail closed.
        """

        with self._lock:
            if self._shutdown:
                raise SessionShutdownError()
            self._phase = SessionPhase.RESTARTING
            self._journal(
                _EVENT_RESTART,
                {
                    "repository_id": self._policy.repository_id,
                    "cursor_position": self._cursor_position,
                },
            )

        recovery = self._recover_event_log()
        with self._lock:
            if recovery.get("failed_closed"):
                self._mark_failed_closed(
                    str(recovery.get("reason") or "event_log_recovery_failed")
                )
                raise SessionFailedClosed(
                    f"event log recovery failed closed: {recovery.get('reason')}",
                    reason_code=str(recovery.get("reason") or "failed_closed"),
                )
            self._journal(
                _EVENT_RECOVERY,
                {
                    "repository_id": self._policy.repository_id,
                    "repaired": bool(recovery.get("repaired")),
                    "reason": str(recovery.get("reason") or "valid"),
                    "valid_count": int(recovery.get("valid_count") or 0),
                    "invalid_bytes": int(recovery.get("invalid_bytes") or 0),
                },
            )

        replay_report = self._replay_from_checkpoint()
        wal_report: Mapping[str, Any] = {}
        if self._durable is not None:
            try:
                wal_report = dict(self._durable.recover() or {})
            except Exception as exc:
                with self._lock:
                    self._mark_failed_closed(_clip(str(exc)))
                raise SessionFailedClosed(
                    f"durable root recovery failed closed: {exc}",
                    reason_code="durable_recovery_failed",
                ) from exc
            try:
                root = self._durable.read_root(self._policy.repository_id)
            except Exception as exc:
                with self._lock:
                    self._mark_failed_closed(_clip(str(exc)))
                raise SessionFailedClosed(
                    f"durable root read failed closed: {exc}",
                    reason_code="durable_root_unreadable",
                ) from exc
            with self._lock:
                # Never invent a root from candidate journal rows; durable
                # pointer is authoritative when present.
                if root is not None:
                    self._current_root = root

        with self._lock:
            # Drop nonterminal fences that cannot safely resume.
            now = self._clock_ms()
            resumed: list[str] = []
            for attempt_id, fence in list(self._fences.items()):
                if fence.terminal or fence.accepted:
                    continue
                if fence.cancelled or not fence.is_live(now):
                    fence.cancelled = True
                    fence.terminal = True
                    continue
                # Re-queue only the live nonterminal snapshot scan.
                if fence.snapshot_cid not in self._pending:
                    self._pending[fence.snapshot_cid] = _PendingScan(
                        snapshot_cid=fence.snapshot_cid,
                        attempt_id=attempt_id,
                        coalesce_count=1,
                        first_seen_ms=now,
                        last_seen_ms=now,
                        sources=["restart"],
                    )
                    resumed.append(attempt_id)
            self._journal(
                _EVENT_REPLAY,
                {
                    "repository_id": self._policy.repository_id,
                    "events_applied": int(replay_report.get("events_applied") or 0),
                    "accepted_restored": len(self._accepted),
                    "resumed_attempts": resumed,
                    "wal_recovered": bool(wal_report),
                    "current_generation": (
                        0
                        if self._current_root is None
                        else int(self._current_root.generation)
                    ),
                },
            )
            self._checkpoint_cursor()
            self._phase = (
                SessionPhase.WATCHING if self._pending else SessionPhase.IDLE
            )
            return self.status()

    def replay(self) -> SessionStatus:
        """Replay journal pages without scheduling new work or publishing."""

        with self._lock:
            if self._shutdown:
                raise SessionShutdownError()
            if self._failed_closed:
                raise SessionFailedClosed(
                    self._diagnostic or "session failed closed",
                    reason_code="failed_closed",
                )
        report = self._replay_from_checkpoint()
        with self._lock:
            self._journal(
                _EVENT_REPLAY,
                {
                    "repository_id": self._policy.repository_id,
                    "events_applied": int(report.get("events_applied") or 0),
                    "mode": "replay_only",
                    "accepted_restored": len(self._accepted),
                    "current_generation": (
                        0
                        if self._current_root is None
                        else int(self._current_root.generation)
                    ),
                },
            )
            self._checkpoint_cursor()
            return self.status()

    def _recover_event_log(self) -> dict[str, Any]:
        path = self._policy.event_log_path
        if path is None:
            return {
                "repaired": False,
                "failed_closed": False,
                "reason": "no_event_log",
                "valid_count": 0,
                "invalid_bytes": 0,
            }
        from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
            read_event_cursor_checkpoint,
            recover_jsonl_event_log_tail,
        )

        checkpoint = None
        if self._policy.checkpoint_path is not None:
            try:
                checkpoint = read_event_cursor_checkpoint(self._policy.checkpoint_path)
            except Exception as exc:
                if self._policy.fail_closed_on_corrupt_log:
                    return {
                        "repaired": False,
                        "failed_closed": True,
                        "reason": "checkpoint_malformed",
                        "error": type(exc).__name__,
                        "valid_count": 0,
                        "invalid_bytes": 0,
                    }
                checkpoint = None
        try:
            result = recover_jsonl_event_log_tail(path, checkpoint=checkpoint)
        except Exception as exc:
            if self._policy.fail_closed_on_corrupt_log:
                return {
                    "repaired": False,
                    "failed_closed": True,
                    "reason": "event_log_recovery_exception",
                    "error": type(exc).__name__,
                    "diagnostic": _clip(str(exc)),
                    "valid_count": 0,
                    "invalid_bytes": 0,
                }
            return {
                "repaired": False,
                "failed_closed": False,
                "reason": "recovery_exception_soft",
                "error": type(exc).__name__,
                "valid_count": 0,
                "invalid_bytes": 0,
            }
        if result.get("failed_closed") and self._policy.fail_closed_on_corrupt_log:
            return dict(result)
        if result.get("failed_closed") and not self._policy.fail_closed_on_corrupt_log:
            soft = dict(result)
            soft["failed_closed"] = False
            soft["reason"] = f"soft_{result.get('reason')}"
            return soft
        return dict(result)

    def _replay_from_checkpoint(self) -> dict[str, Any]:
        """Rebuild session memory by replaying the full retained event log.

        The durable cursor checkpoint is consulted only during tail recovery
        (to prove anchors remain in the retained chain). State rebuild always
        starts from the log's initial cursor so accepted transitions are not
        lost when the checkpoint already points at the tip.
        """

        path = self._policy.event_log_path
        if path is None or not Path(path).exists():
            return {"events_applied": 0}

        from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
            event_log_initial_cursor,
            event_log_latest_cursor,
            read_jsonl_event_page,
        )

        try:
            cursor = event_log_initial_cursor(path)
        except Exception as exc:
            if self._policy.fail_closed_on_corrupt_log:
                with self._lock:
                    self._mark_failed_closed(_clip(str(exc)))
                raise SessionFailedClosed(
                    f"unable to open event log for replay: {exc}",
                    reason_code="event_log_unreadable",
                ) from exc
            return {"events_applied": 0, "reason": "event_log_unreadable"}

        applied = 0
        # Replay rebuilds derived session memory; do not clear durable root.
        restored_accepted: list[_AcceptedTransition] = []
        restored_fences: dict[str, _SessionFence] = {}
        restored_pending: dict[str, _PendingScan] = {}

        while True:
            try:
                page = read_jsonl_event_page(path, cursor, limit=256)
            except Exception as exc:
                if self._policy.fail_closed_on_corrupt_log:
                    with self._lock:
                        self._mark_failed_closed(_clip(str(exc)))
                    raise SessionFailedClosed(
                        f"event replay failed closed: {exc}",
                        reason_code="event_replay_failed",
                    ) from exc
                break
            if not page.events:
                break
            for event in page.events:
                applied += 1
                self._apply_replay_event(
                    event,
                    accepted=restored_accepted,
                    fences=restored_fences,
                    pending=restored_pending,
                )
            cursor = page.next_cursor
            if not page.has_more:
                break

        with self._lock:
            # Full rebuild of accepted transitions from the journal, then keep
            # any newer in-memory accepts that are not yet durable in the log.
            journal_keys = {
                (item.attempt_id, item.root.root_cid, item.root.generation)
                for item in restored_accepted
            }
            live_extra = [
                item
                for item in self._accepted
                if (
                    item.attempt_id,
                    item.root.root_cid,
                    item.root.generation,
                )
                not in journal_keys
            ]
            self._accepted = list(restored_accepted) + live_extra
            # Prefer durable/current root over candidate journal noise.
            if restored_accepted and self._current_root is None:
                # Only bootstrap from accepted journal rows when durable port
                # has no pointer; never from candidates.
                self._current_root = restored_accepted[-1].root
            for attempt_id, fence in restored_fences.items():
                existing = self._fences.get(attempt_id)
                if existing is None or (not existing.accepted and fence.accepted):
                    self._fences[attempt_id] = fence
            for snap, pending in restored_pending.items():
                if snap not in self._pending and (
                    self._active_snapshot_cid != snap
                ):
                    self._pending[snap] = pending
            try:
                latest = event_log_latest_cursor(path)
                self._cursor_position = int(latest.position)
            except Exception:
                self._cursor_position = int(getattr(cursor, "position", 0) or 0)
        return {"events_applied": applied}

    def _apply_replay_event(
        self,
        event: Mapping[str, Any],
        *,
        accepted: list[_AcceptedTransition],
        fences: dict[str, _SessionFence],
        pending: dict[str, _PendingScan],
    ) -> None:
        event_type = str(event.get("type") or "")
        repo = str(event.get("repository_id") or "")
        if repo and repo != self._policy.repository_id:
            return
        attempt_id = str(event.get("attempt_id") or "")
        snapshot_cid = str(event.get("snapshot_cid") or "")
        sequence = event.get("sequence")
        seq_int = int(sequence) if isinstance(sequence, int) and not isinstance(sequence, bool) else None

        if event_type == _EVENT_SCAN_SCHEDULED and snapshot_cid and attempt_id:
            try:
                validate_opaque_cid(snapshot_cid, "snapshot_cid")
            except HarnessError:
                return
            if snapshot_cid not in pending:
                pending[snapshot_cid] = _PendingScan(
                    snapshot_cid=snapshot_cid,
                    attempt_id=attempt_id,
                    coalesce_count=1,
                    first_seen_ms=self._clock_ms(),
                    last_seen_ms=self._clock_ms(),
                    sources=["replay"],
                )
            fences.setdefault(
                attempt_id,
                _SessionFence(
                    attempt_id=attempt_id,
                    fencing_token=int(event.get("fencing_token") or 0) or 1,
                    lease_id=f"lease:{attempt_id}",
                    snapshot_cid=snapshot_cid,
                    expires_at_ms=self._clock_ms() + self._policy.fence_ttl_ms,
                    terminal=False,
                ),
            )
            return

        if event_type == _EVENT_SCAN_STARTED and attempt_id and snapshot_cid:
            token = int(event.get("fencing_token") or 0) or 1
            fences[attempt_id] = _SessionFence(
                attempt_id=attempt_id,
                fencing_token=token,
                lease_id=f"lease:{attempt_id}",
                snapshot_cid=snapshot_cid,
                expires_at_ms=self._clock_ms() + self._policy.fence_ttl_ms,
                terminal=False,
            )
            pending.pop(snapshot_cid, None)
            return

        if event_type in {_EVENT_SCAN_COMPLETED, _EVENT_SCAN_CANCELLED} and attempt_id:
            fence = fences.get(attempt_id)
            if fence is not None:
                fence.terminal = True
                if event_type == _EVENT_SCAN_CANCELLED:
                    fence.cancelled = True
                fence.verified = bool(event.get("verified", False))
                fence.new_root_cid = event.get("new_root_cid")  # type: ignore[assignment]
            if snapshot_cid:
                pending.pop(snapshot_cid, None)
            return

        if event_type == _EVENT_TRANSITION_ACCEPTED and attempt_id:
            root_cid = event.get("root_cid")
            generation = event.get("generation")
            if not isinstance(root_cid, str) or not isinstance(generation, int):
                return
            try:
                root = RootRef(
                    root_cid=validate_opaque_cid(root_cid, "root_cid"),
                    generation=_nonneg_int(generation, "generation"),
                )
            except HarnessError:
                return
            token = int(event.get("fencing_token") or 0) or 1
            snap = snapshot_cid or str(event.get("snapshot_cid") or "unknown")
            accepted.append(
                _AcceptedTransition(
                    attempt_id=attempt_id,
                    fencing_token=token,
                    snapshot_cid=snap,
                    root=root,
                    event_sequence=seq_int,
                )
            )
            fence = fences.get(attempt_id)
            if fence is not None:
                fence.accepted = True
                fence.verified = True
                fence.terminal = True
                fence.new_root_cid = root.root_cid
            # Accepted work is terminal; drop any residual pending for snap.
            if snap and snap != "unknown":
                pending.pop(snap, None)
            return

        if event_type == _EVENT_TRANSITION_CANDIDATE:
            # Explicitly never promote candidates on replay.
            return

        if event_type == _EVENT_ROOT_PUBLISH_DENIED:
            return

    # -- worker lifecycle --------------------------------------------------

    def start(self) -> None:
        """Optionally start a background drain worker when policy allows."""

        with self._condition:
            self._ensure_open_for_work()
            if not self._policy.worker_enabled:
                self._phase = SessionPhase.WATCHING
                return
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker_stop.clear()
            worker = threading.Thread(
                target=self._worker_main,
                name=f"semantic-session-{self._policy.repository_id}",
                daemon=True,
            )
            self._worker = worker
            self._owned_threads.append(worker)
            self._phase = SessionPhase.WATCHING
            worker.start()

    def _worker_main(self) -> None:
        debounce_s = max(0.0, self._policy.debounce_ms / 1000.0)
        while not self._worker_stop.is_set() and not self._shutdown:
            with self._condition:
                if self._failed_closed or self._shutdown:
                    break
                if not self._pending:
                    self._condition.wait(timeout=max(0.05, debounce_s))
                    continue
                # Debounce: wait until the newest pending notification is old enough.
                now = self._clock_ms()
                newest = max(item.last_seen_ms for item in self._pending.values())
                wait_ms = self._policy.debounce_ms - (now - newest)
                if wait_ms > 0:
                    self._condition.wait(timeout=wait_ms / 1000.0)
                    continue
            try:
                self._process_one_pending()
            except SessionShutdownError:
                break
            except SessionFailedClosed:
                break
            except Exception as exc:
                with self._lock:
                    self._diagnostic = _clip(str(exc))

    def shutdown(self, *, reason: str = "shutdown") -> SessionStatus:
        """Cancel owned work, stop the worker, and join owned threads."""

        reason_text = _clip(reason or "shutdown")
        with self._condition:
            if self._phase is SessionPhase.STOPPED and self._shutdown:
                return self.status()
            self._phase = SessionPhase.SHUTTING_DOWN
            self._shutdown = True
            self._worker_stop.set()
            # Cancel every known cancellation token and live fence.
            for token in list(self._cancellations.values()):
                try:
                    token.cancel(cancellation_id=token.cancellation_id, reason=reason_text)
                except Exception:
                    pass
            for fence in self._fences.values():
                if not fence.terminal:
                    fence.cancelled = True
            # Drop pending scans; they will not run after shutdown.
            self._pending.clear()
            self._active_attempt_id = None
            self._active_snapshot_cid = None
            self._journal(
                _EVENT_SHUTDOWN,
                {
                    "repository_id": self._policy.repository_id,
                    "reason": reason_text,
                },
            )
            self._condition.notify_all()

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self._policy.join_timeout_s)
        # Join any other owned threads.
        for thread in list(self._owned_threads):
            if thread is worker:
                continue
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=self._policy.join_timeout_s)

        with self._lock:
            self._worker = None
            self._phase = SessionPhase.STOPPED
            self._checkpoint_cursor()
            return self.status()

    def close(self) -> None:
        """Context-manager alias for :meth:`shutdown`."""

        self.shutdown(reason="close")

    def __enter__(self) -> "SemanticStateSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _ensure_open_for_work(self) -> None:
        if self._shutdown or self._phase is SessionPhase.STOPPED:
            raise SessionShutdownError()
        if self._failed_closed or self._phase is SessionPhase.FAILED_CLOSED:
            raise SessionFailedClosed(
                self._diagnostic or "session failed closed",
                reason_code="failed_closed",
            )

    def _mark_failed_closed(self, diagnostic: str) -> None:
        self._failed_closed = True
        self._phase = SessionPhase.FAILED_CLOSED
        self._diagnostic = _clip(diagnostic)
        self._journal(
            _EVENT_FAILED_CLOSED,
            {
                "repository_id": self._policy.repository_id,
                "diagnostic": self._diagnostic,
            },
        )

    def _journal(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> int | None:
        path = self._policy.event_log_path
        if path is None:
            return None
        from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
            append_jsonl_event,
        )

        safe = {
            key: value
            for key, value in dict(payload).items()
            if key
            not in {
                "secret",
                "prompt",
                "source_body",
                "model_output",
                "api_key",
                "messages",
            }
        }
        safe.setdefault("schema", SESSION_SCHEMA)
        safe.setdefault("interface", SESSION_INTERFACE)
        safe.setdefault("board_namespace", BOARD_NAMESPACE)
        try:
            event = append_jsonl_event(path, event_type, safe)
        except Exception as exc:
            # Journal failures fail closed only when policy demands durability.
            if self._policy.fail_closed_on_corrupt_log:
                self._failed_closed = True
                self._phase = SessionPhase.FAILED_CLOSED
                self._diagnostic = _clip(f"journal append failed: {exc}")
            return None
        sequence = event.get("sequence")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            self._cursor_position = max(self._cursor_position, int(sequence))
            return int(sequence)
        return None

    def _checkpoint_cursor(self) -> None:
        path = self._policy.event_log_path
        checkpoint_path = self._policy.checkpoint_path
        if path is None or checkpoint_path is None:
            return
        if not Path(path).exists():
            return
        from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
            event_log_latest_cursor,
            write_event_cursor_checkpoint,
        )

        try:
            cursor = event_log_latest_cursor(path)
            write_event_cursor_checkpoint(checkpoint_path, cursor)
            self._cursor_position = int(cursor.position)
        except Exception as exc:
            if self._policy.fail_closed_on_corrupt_log:
                self._diagnostic = _clip(f"checkpoint write failed: {exc}")

    def fence_for(self, attempt_id: str) -> _SessionFence | None:
        """Test/diagnostic accessor for the attempt fence."""

        with self._lock:
            return self._fences.get(attempt_id)

    def accepted_transitions(self) -> tuple[_AcceptedTransition, ...]:
        with self._lock:
            return tuple(self._accepted)

    def last_scan_results(self) -> tuple[ScanResult, ...]:
        with self._lock:
            return tuple(self._last_scan_results)


# ---------------------------------------------------------------------------
# Module entrypoints
# ---------------------------------------------------------------------------


def watch_session(
    policy: SessionPolicy | Mapping[str, Any],
    snapshot_cid: str,
    *,
    session: SemanticStateSession | None = None,
    durable_port: DurableSemanticStatePort | None = None,
    scan_executor: ScanExecutor | Callable[..., Mapping[str, Any] | ScanResult] | None = None,
    source: str = "watch_session",
    process: bool = False,
    **session_kwargs: Any,
) -> tuple[SemanticStateSession, WatchAck, list[ScanResult]]:
    """Admit a watch notification on a session, optionally draining scans."""

    owner = session or SemanticStateSession(
        policy,
        durable_port=durable_port,
        scan_executor=scan_executor,
        **session_kwargs,
    )
    ack = owner.notify_watch(snapshot_cid, source=source)
    results: list[ScanResult] = []
    if process:
        results = owner.drain()
    return owner, ack, results


def replay_session(
    policy: SessionPolicy | Mapping[str, Any],
    *,
    session: SemanticStateSession | None = None,
    durable_port: DurableSemanticStatePort | None = None,
    restart: bool = False,
    **session_kwargs: Any,
) -> tuple[SemanticStateSession, SessionStatus]:
    """Open/reuse a session and replay (or full restart-reconcile) its journal."""

    owner = session or SemanticStateSession(
        policy,
        durable_port=durable_port,
        **session_kwargs,
    )
    status = owner.restart() if restart else owner.replay()
    return owner, status


def semantic_state_session_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticStateSession@1."""

    return {
        "interface": SESSION_INTERFACE,
        "schema": SESSION_SCHEMA,
        "bundle": BUNDLE_ID,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "symbols": [
            "SemanticStateSession",
            "SessionPolicy",
            "SessionStatus",
            "SessionPhase",
            "WatchAck",
            "ScanResult",
            "watch_session",
            "replay_session",
            "semantic_state_session_descriptor",
        ],
        "composes": [
            "runtime.event_log",
            "DurableSemanticStatePort",
            "CancellationToken",
            "RootRef",
        ],
        "invariants": [
            "watch_notifications_only_schedule_canonical_scan",
            "concurrent_equal_snapshot_cids_coalesce",
            "accepted_attempts_serialize_by_repository_fence",
            "stale_or_expired_fence_cannot_publish_root",
            "unverified_transition_cannot_publish_root",
            "restart_restores_accepted_without_publishing_unverified",
            "corrupt_or_truncated_events_recover_or_fail_closed",
            "explicit_shutdown_cancels_and_joins_owned_work",
            "cold_import_starts_no_resources",
            "events_mtime_queue_are_not_semantic_truth",
        ],
        "forbids": [
            "publish_from_watch_event_alone",
            "duplicate_equal_snapshot_work",
            "overwrite_root_without_generation_cas",
            "resume_expired_or_mismatched_fence",
            "background_work_on_import",
            "task_queue_as_authority",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "BUNDLE_ID",
    "SESSION_INTERFACE",
    "SESSION_SCHEMA",
    "ScanExecutor",
    "ScanResult",
    "SemanticStateSession",
    "SessionError",
    "SessionFailedClosed",
    "SessionPhase",
    "SessionPolicy",
    "SessionRootConflict",
    "SessionRootPublishDenied",
    "SessionShutdownError",
    "SessionStatus",
    "WatchAck",
    "replay_session",
    "semantic_state_session_descriptor",
    "watch_session",
]
