"""Fail-closed lifecycle recovery for the managed Quack state owner.

``QuackOwnerWatchdog`` deliberately contains no database client.  The caller
supplies both the owner observation and the authenticated readiness probe, so
this module can coordinate process recovery without opening DuckDB (or
mistaking a transport failure for proof that the owner is dead).

The recovery boundary is intentionally narrow:

* desired ``stopped`` is durable in the watchdog instance and never launches;
* only an exact ``OwnerLiveness.DEAD`` observation can authorize a launch;
* ``ALIVE``-but-unhealthy and ``UNKNOWN`` observations abstain;
* a non-blocking process-shared lock elects one recovery winner;
* a new owner is not recovered until an authenticated, identity-bound
  readiness callback succeeds;
* failed launches/readiness checks consume a bounded exponential-backoff
  budget; and
* a newly launched or explicitly adopted owner is stopped as an exact,
  isolated process tree, never by an unverified raw PID.

Public status and receipts use closed fields.  They contain identifiers only,
never callback exception text, credentials, authentication material, or
arbitrary readiness payloads.

Cold import performs no filesystem, database, network, subprocess, signal, or
environment operation.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import signal
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    read_process_birth,
)

QUACK_OWNER_WATCHDOG_INTERFACE: Final[str] = "QuackOwnerWatchdog@1"
QUACK_OWNER_WATCHDOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-watchdog@1"
)
QUACK_OWNER_WATCHDOG_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-watchdog-receipt@1"
)
QUACK_OWNER_TERMINATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-termination-receipt@1"
)


class QuackOwnerWatchdogError(RuntimeError):
    """Base error for malformed or unsafe lifecycle operations."""


class QuackOwnerWatchdogPolicyError(QuackOwnerWatchdogError, ValueError):
    """The requested lifecycle policy is invalid or unsafe."""


class QuackOwnerStartAbsentError(QuackOwnerWatchdogError):
    """A start attempt proved that no owner process was created.

    Only this typed failure is retryable. An arbitrary start callback error
    has an unknown external outcome and therefore exhausts automatic recovery
    rather than risking a second owner.
    """


class DesiredOwnerState(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Operator-selected state; observations cannot alter it."""

    RUNNING = "running"
    STOPPED = "stopped"


class OwnerHealth(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Authenticated application health, distinct from process liveness."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class WatchdogDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed result vocabulary for one lifecycle decision."""

    HEALTHY = "healthy"
    INTENTIONALLY_STOPPED = "intentionally_stopped"
    RESTARTED = "restarted"
    ADOPTED = "adopted"
    BACKOFF = "backoff"
    RETRY_EXHAUSTED = "retry_exhausted"
    LOCK_CONTENDED = "lock_contended"
    ABSTAIN_ALIVE_UNHEALTHY = "abstain_alive_unhealthy"
    ABSTAIN_UNKNOWN = "abstain_unknown"
    ABSTAIN_NOT_PROVABLY_DEAD = "abstain_not_provably_dead"
    START_FAILED = "start_failed"
    READINESS_FAILED = "readiness_failed"
    STOPPED = "stopped"
    STOP_FAILED = "stop_failed"


class _ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def process_birth_id(birth: ProcessBirthIdentity) -> str:
    """Return the repository's stable multi-factor process-birth identity."""

    if not isinstance(birth, ProcessBirthIdentity):
        raise TypeError("birth must be ProcessBirthIdentity")
    material = (
        f"{int(birth.pid)}:{int(birth.start_time_ticks)}:"
        f"{str(birth.boot_id or '')}:{int(birth.parent_pid or 0)}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"birth:{digest[7:39]}"


def process_births_match(
    expected: ProcessBirthIdentity,
    observed: ProcessBirthIdentity | None,
) -> bool:
    """Compare every process-birth factor; a raw PID is never sufficient."""

    if observed is None:
        return False
    return (
        int(expected.pid) == int(observed.pid)
        and int(expected.start_time_ticks) > 0
        and int(expected.start_time_ticks) == int(observed.start_time_ticks)
        and (
            not expected.boot_id
            or not observed.boot_id
            or str(expected.boot_id) == str(observed.boot_id)
        )
    )


@dataclass(frozen=True)
class QuackOwnerBinding:
    """Non-secret database/server identities required at readiness."""

    store_id: str
    schema_revision: str
    database_uuid: str
    schema_fingerprint: str
    generation: int
    server_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "store_id",
            "schema_revision",
            "database_uuid",
            "schema_fingerprint",
        ):
            if not str(getattr(self, name) or "").strip():
                raise QuackOwnerWatchdogPolicyError(f"{name} is required")
        if int(self.generation) < 1:
            raise QuackOwnerWatchdogPolicyError("generation must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "schema_revision": self.schema_revision,
            "database_uuid": self.database_uuid,
            "schema_fingerprint": self.schema_fingerprint,
            "generation": int(self.generation),
            "server_id": self.server_id,
        }


@dataclass(frozen=True)
class QuackOwnerObservation:
    """Exact liveness and authenticated health observed by the caller."""

    process_birth: ProcessBirthIdentity | None
    liveness: OwnerLiveness
    health: OwnerHealth = OwnerHealth.UNKNOWN
    authenticated_ready: bool = False
    absence_proven: bool = False
    binding: QuackOwnerBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.liveness, OwnerLiveness):
            raise TypeError("liveness must be OwnerLiveness")
        if not isinstance(self.health, OwnerHealth):
            raise TypeError("health must be OwnerHealth")
        if self.process_birth is not None and not isinstance(
            self.process_birth, ProcessBirthIdentity
        ):
            raise TypeError("process_birth must be ProcessBirthIdentity or None")
        if self.liveness is OwnerLiveness.ALIVE and self.process_birth is None:
            raise QuackOwnerWatchdogPolicyError(
                "an alive observation requires exact process birth"
            )
        if self.authenticated_ready and self.health is not OwnerHealth.HEALTHY:
            raise QuackOwnerWatchdogPolicyError(
                "authenticated_ready requires healthy application state"
            )

    @property
    def provably_dead(self) -> bool:
        """Death is explicit; absence/transport failure is not inferred."""

        return self.liveness is OwnerLiveness.DEAD and (
            self.process_birth is not None or bool(self.absence_proven)
        )


@dataclass(frozen=True)
class AuthenticatedReadiness:
    """Closed result of the caller's authenticated live identity query."""

    authenticated: bool
    ready: bool
    process_birth_id: str
    binding: QuackOwnerBinding

    def __post_init__(self) -> None:
        if not isinstance(self.binding, QuackOwnerBinding):
            raise TypeError("binding must be QuackOwnerBinding")
        if not str(self.process_birth_id or "").strip():
            raise QuackOwnerWatchdogPolicyError("process_birth_id is required")

    @property
    def admitted(self) -> bool:
        return bool(self.authenticated and self.ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": bool(self.authenticated),
            "ready": bool(self.ready),
            "process_birth_id": self.process_birth_id,
            "binding": self.binding.to_dict(),
        }


@dataclass(frozen=True)
class SpawnedQuackOwner:
    """Managed local owner and its exact process-tree identity."""

    process: _ProcessLike | None
    process_birth: ProcessBirthIdentity
    isolated_process_group: bool = True
    adopted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.process_birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if self.process_birth.pid <= 0:
            raise QuackOwnerWatchdogPolicyError("owner pid must be positive")
        if self.process_birth.start_time_ticks <= 0:
            raise QuackOwnerWatchdogPolicyError(
                "owner process birth must include start_time_ticks"
            )
        if self.process is not None and int(self.process.pid) != int(
            self.process_birth.pid
        ):
            raise QuackOwnerWatchdogPolicyError(
                "process handle pid does not match process birth"
            )
        if not self.isolated_process_group:
            raise QuackOwnerWatchdogPolicyError(
                "managed owner must use an isolated process group"
            )

    @property
    def birth_id(self) -> str:
        return process_birth_id(self.process_birth)


@dataclass(frozen=True)
class QuackOwnerWatchdogPolicy:
    """Bounded recovery and safe-termination policy."""

    max_restart_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    termination_grace_seconds: float = 5.0
    termination_kill_wait_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_restart_attempts) <= 64:
            raise QuackOwnerWatchdogPolicyError(
                "max_restart_attempts must be between 1 and 64"
            )
        if not 0.0 <= float(self.initial_backoff_seconds) <= 3_600.0:
            raise QuackOwnerWatchdogPolicyError(
                "initial_backoff_seconds is out of bounds"
            )
        if not (
            float(self.initial_backoff_seconds)
            <= float(self.maximum_backoff_seconds)
            <= 86_400.0
        ):
            raise QuackOwnerWatchdogPolicyError(
                "maximum_backoff_seconds is out of bounds"
            )
        if not 1.0 <= float(self.backoff_multiplier) <= 16.0:
            raise QuackOwnerWatchdogPolicyError(
                "backoff_multiplier must be between 1 and 16"
            )
        if not 0.0 <= float(self.termination_grace_seconds) <= 300.0:
            raise QuackOwnerWatchdogPolicyError(
                "termination_grace_seconds is out of bounds"
            )
        if not 0.0 <= float(self.termination_kill_wait_seconds) <= 300.0:
            raise QuackOwnerWatchdogPolicyError(
                "termination_kill_wait_seconds is out of bounds"
            )


@dataclass(frozen=True)
class QuackOwnerRetryState:
    """In-memory bounded retry state suitable for caller projection."""

    consecutive_failures: int = 0
    retry_not_before: float = 0.0
    exhausted: bool = False
    last_failure_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive_failures": int(self.consecutive_failures),
            "retry_not_before": float(self.retry_not_before),
            "exhausted": bool(self.exhausted),
            "last_failure_code": self.last_failure_code,
        }


@dataclass(frozen=True)
class ProcessTreeTerminationReceipt:
    """Token-free evidence for exact process-tree termination."""

    process_birth_id: str
    termination_confirmed: bool
    terminate_sent: bool = False
    kill_sent: bool = False
    already_absent: bool = False
    reason: str = ""
    schema: str = QUACK_OWNER_TERMINATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "process_birth_id": self.process_birth_id,
            "termination_confirmed": bool(self.termination_confirmed),
            "terminate_sent": bool(self.terminate_sent),
            "kill_sent": bool(self.kill_sent),
            "already_absent": bool(self.already_absent),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QuackOwnerWatchdogReceipt:
    """Closed, credential-free result of one watchdog operation."""

    disposition: WatchdogDisposition
    desired_state: DesiredOwnerState
    liveness: OwnerLiveness
    health: OwnerHealth
    reason: str
    operational_ready: bool = False
    recovered: bool = False
    restart_attempted: bool = False
    restart_attempt: int = 0
    one_winner_lock_acquired: bool = False
    authenticated_readiness: bool = False
    process_birth_id: str = ""
    binding: QuackOwnerBinding | None = None
    retry_state: QuackOwnerRetryState = QuackOwnerRetryState()
    termination: ProcessTreeTerminationReceipt | None = None
    schema: str = QUACK_OWNER_WATCHDOG_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": QUACK_OWNER_WATCHDOG_INTERFACE,
            "disposition": self.disposition.value,
            "desired_state": self.desired_state.value,
            "liveness": self.liveness.value,
            "health": self.health.value,
            "reason": self.reason,
            "operational_ready": bool(self.operational_ready),
            "recovered": bool(self.recovered),
            "restart_attempted": bool(self.restart_attempted),
            "restart_attempt": int(self.restart_attempt),
            "one_winner_lock_acquired": bool(self.one_winner_lock_acquired),
            "authenticated_readiness": bool(self.authenticated_readiness),
            "process_birth_id": self.process_birth_id,
            "binding": None if self.binding is None else self.binding.to_dict(),
            "retry_state": self.retry_state.to_dict(),
            "termination": (
                None if self.termination is None else self.termination.to_dict()
            ),
        }


def _read_exact_birth(
    birth_reader: Callable[[int], ProcessBirthIdentity | None],
    expected: ProcessBirthIdentity,
) -> tuple[bool, bool]:
    """Return ``(exact, absent)`` while treating inspection errors as unknown."""

    try:
        observed = birth_reader(int(expected.pid))
    except OSError:
        return False, False
    if observed is None:
        return False, True
    return process_births_match(expected, observed), False


def terminate_spawned_owner(
    owner: SpawnedQuackOwner,
    *,
    grace_seconds: float = 5.0,
    kill_wait_seconds: float = 2.0,
    birth_reader: Callable[[int], ProcessBirthIdentity | None] = read_process_birth,
    group_id_reader: Callable[[int], int] = os.getpgid,
    group_signal: Callable[[int, int], None] = os.killpg,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> ProcessTreeTerminationReceipt:
    """Terminate one exact isolated process tree with PID-reuse fencing.

    The birth identity and process-group boundary are rechecked before every
    signal.  Unknown identity, PID reuse, or a non-isolated group causes an
    abstention rather than signaling an unrelated process.
    """

    if not isinstance(owner, SpawnedQuackOwner):
        raise TypeError("owner must be SpawnedQuackOwner")
    birth_id = owner.birth_id
    process = owner.process
    if process is not None and process.poll() is not None:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=True,
            already_absent=True,
            reason="process_already_exited",
        )

    exact, absent = _read_exact_birth(birth_reader, owner.process_birth)
    if absent:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=True,
            already_absent=True,
            reason="process_already_absent",
        )
    if not exact:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            reason="process_birth_unknown_or_reused",
        )
    if not owner.isolated_process_group:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            reason="process_group_not_isolated",
        )
    try:
        group_id = int(group_id_reader(int(owner.process_birth.pid)))
    except (OSError, ValueError):
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            reason="process_group_unknown",
        )
    if group_id != int(owner.process_birth.pid):
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            reason="process_group_not_isolated",
        )

    terminate_sent = False
    try:
        group_signal(group_id, signal.SIGTERM)
        terminate_sent = True
    except ProcessLookupError:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=True,
            already_absent=True,
            reason="process_disappeared_before_terminate",
        )
    except OSError:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            reason="terminate_signal_failed",
        )

    deadline = monotonic() + max(0.0, float(grace_seconds))
    while monotonic() < deadline:
        exact, absent = _read_exact_birth(birth_reader, owner.process_birth)
        if absent:
            return ProcessTreeTerminationReceipt(
                process_birth_id=birth_id,
                termination_confirmed=True,
                terminate_sent=terminate_sent,
                reason="terminated_after_graceful_signal",
            )
        if not exact:
            return ProcessTreeTerminationReceipt(
                process_birth_id=birth_id,
                termination_confirmed=False,
                terminate_sent=terminate_sent,
                reason="process_birth_changed_during_termination",
            )
        sleeper(min(0.05, max(0.0, deadline - monotonic())))

    exact, absent = _read_exact_birth(birth_reader, owner.process_birth)
    if absent:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=True,
            terminate_sent=terminate_sent,
            reason="terminated_after_graceful_signal",
        )
    if not exact:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            terminate_sent=terminate_sent,
            reason="process_birth_changed_before_kill",
        )
    try:
        if int(group_id_reader(int(owner.process_birth.pid))) != group_id:
            raise OSError(errno.ESRCH, "process group changed")
        group_signal(group_id, signal.SIGKILL)
        kill_sent = True
    except ProcessLookupError:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=True,
            terminate_sent=terminate_sent,
            already_absent=True,
            reason="process_disappeared_before_kill",
        )
    except OSError:
        return ProcessTreeTerminationReceipt(
            process_birth_id=birth_id,
            termination_confirmed=False,
            terminate_sent=terminate_sent,
            reason="kill_signal_failed",
        )

    deadline = monotonic() + max(0.0, float(kill_wait_seconds))
    while monotonic() < deadline:
        _exact, absent = _read_exact_birth(birth_reader, owner.process_birth)
        if absent:
            return ProcessTreeTerminationReceipt(
                process_birth_id=birth_id,
                termination_confirmed=True,
                terminate_sent=terminate_sent,
                kill_sent=kill_sent,
                reason="terminated_after_kill_signal",
            )
        sleeper(min(0.05, max(0.0, deadline - monotonic())))
    _exact, absent = _read_exact_birth(birth_reader, owner.process_birth)
    return ProcessTreeTerminationReceipt(
        process_birth_id=birth_id,
        termination_confirmed=bool(absent),
        terminate_sent=terminate_sent,
        kill_sent=kill_sent,
        reason=("terminated_after_kill_signal" if absent else "termination_timeout"),
    )


class _OneWinnerLock:
    """Short process-shared recovery lock; it is not the database owner lock."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle: Any | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise QuackOwnerWatchdogPolicyError(
                "recovery lock cannot be opened safely"
            ) from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise QuackOwnerWatchdogPolicyError(
                    "recovery lock must be a regular file"
                )
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "a+b", buffering=0)
            descriptor = -1
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                return False
            self.handle = handle
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


StartOwner = Callable[[], SpawnedQuackOwner]
ReadinessProbe = Callable[[SpawnedQuackOwner], AuthenticatedReadiness]
ObserveOwner = Callable[[], QuackOwnerObservation]
TerminateOwner = Callable[[SpawnedQuackOwner], ProcessTreeTerminationReceipt]
Clock = Callable[[], float]


class QuackOwnerWatchdog:
    """Coordinate bounded, dead-only recovery of one managed Quack owner."""

    INTERFACE = QUACK_OWNER_WATCHDOG_INTERFACE
    SCHEMA = QUACK_OWNER_WATCHDOG_SCHEMA

    def __init__(
        self,
        *,
        lock_path: str | Path,
        start_owner: StartOwner,
        readiness_probe: ReadinessProbe,
        expected_binding: QuackOwnerBinding,
        observe_owner: ObserveOwner | None = None,
        terminate_owner: TerminateOwner | None = None,
        policy: QuackOwnerWatchdogPolicy | None = None,
        clock: Clock = time.monotonic,
        desired_state: DesiredOwnerState = DesiredOwnerState.RUNNING,
    ) -> None:
        if not isinstance(expected_binding, QuackOwnerBinding):
            raise TypeError("expected_binding must be QuackOwnerBinding")
        if not isinstance(desired_state, DesiredOwnerState):
            raise TypeError("desired_state must be DesiredOwnerState")
        self.lock_path = Path(lock_path)
        self._start_owner = start_owner
        self._readiness_probe = readiness_probe
        self._expected_binding = expected_binding
        self._observe_owner = observe_owner
        self._policy = policy or QuackOwnerWatchdogPolicy()
        self._clock = clock
        self._terminate_owner = terminate_owner or (
            lambda owner: terminate_spawned_owner(
                owner,
                grace_seconds=self._policy.termination_grace_seconds,
                kill_wait_seconds=self._policy.termination_kill_wait_seconds,
            )
        )
        self._desired_state = desired_state
        self._retry_state = QuackOwnerRetryState()
        self._managed_owner: SpawnedQuackOwner | None = None
        self._guard = threading.RLock()

    @property
    def desired_state(self) -> DesiredOwnerState:
        with self._guard:
            return self._desired_state

    @property
    def retry_state(self) -> QuackOwnerRetryState:
        with self._guard:
            return self._retry_state

    def current_owner(self) -> SpawnedQuackOwner | None:
        """Return the retained local handle; never serialized in status."""

        with self._guard:
            return self._managed_owner

    def set_desired_state(self, state: DesiredOwnerState) -> None:
        if not isinstance(state, DesiredOwnerState):
            raise TypeError("state must be DesiredOwnerState")
        with self._guard:
            self._desired_state = state

    def reset_retry_state(self) -> None:
        """Operator-controlled recovery after the bounded budget is exhausted."""

        with self._guard:
            self._retry_state = QuackOwnerRetryState()

    def status(self) -> dict[str, Any]:
        """Return closed, credential-free local policy/retry status."""

        with self._guard:
            managed = self._managed_owner
            return {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "desired_state": self._desired_state.value,
                "managed_owner_present": managed is not None,
                "managed_process_birth_id": (
                    "" if managed is None else managed.birth_id
                ),
                "retry_state": self._retry_state.to_dict(),
                "expected_binding": self._expected_binding.to_dict(),
            }

    def _receipt(
        self,
        observation: QuackOwnerObservation,
        disposition: WatchdogDisposition,
        reason: str,
        **changes: Any,
    ) -> QuackOwnerWatchdogReceipt:
        return QuackOwnerWatchdogReceipt(
            disposition=disposition,
            desired_state=self._desired_state,
            liveness=observation.liveness,
            health=observation.health,
            reason=reason,
            retry_state=self._retry_state,
            **changes,
        )

    def _non_restart_receipt(
        self, observation: QuackOwnerObservation
    ) -> QuackOwnerWatchdogReceipt | None:
        if self._desired_state is DesiredOwnerState.STOPPED:
            return self._receipt(
                observation,
                WatchdogDisposition.INTENTIONALLY_STOPPED,
                "desired_state_stopped",
            )
        if observation.liveness is OwnerLiveness.UNKNOWN:
            return self._receipt(
                observation,
                WatchdogDisposition.ABSTAIN_UNKNOWN,
                "owner_liveness_unknown",
            )
        if observation.liveness is OwnerLiveness.ALIVE:
            if (
                observation.health is OwnerHealth.HEALTHY
                and observation.authenticated_ready
                and observation.binding is not None
                and self._binding_matches_expected(observation.binding)
            ):
                self._retry_state = QuackOwnerRetryState()
                return self._receipt(
                    observation,
                    WatchdogDisposition.HEALTHY,
                    "owner_authenticated_and_healthy",
                    operational_ready=True,
                    authenticated_readiness=True,
                    process_birth_id=(
                        ""
                        if observation.process_birth is None
                        else process_birth_id(observation.process_birth)
                    ),
                    binding=observation.binding,
                )
            return self._receipt(
                observation,
                WatchdogDisposition.ABSTAIN_ALIVE_UNHEALTHY,
                "alive_owner_not_authenticated_healthy",
                process_birth_id=(
                    ""
                    if observation.process_birth is None
                    else process_birth_id(observation.process_birth)
                ),
                binding=observation.binding,
            )
        if not observation.provably_dead:
            return self._receipt(
                observation,
                WatchdogDisposition.ABSTAIN_NOT_PROVABLY_DEAD,
                "owner_absence_not_proven",
            )
        return None

    def _binding_matches_expected(self, binding: QuackOwnerBinding) -> bool:
        expected = self._expected_binding
        return (
            binding.store_id == expected.store_id
            and binding.schema_revision == expected.schema_revision
            and binding.database_uuid == expected.database_uuid
            and binding.schema_fingerprint == expected.schema_fingerprint
            and int(binding.generation) >= int(expected.generation)
        )

    def _readiness_admitted(
        self,
        owner: SpawnedQuackOwner,
        readiness: AuthenticatedReadiness,
        observation: QuackOwnerObservation,
    ) -> bool:
        if not isinstance(readiness, AuthenticatedReadiness):
            return False
        if not readiness.admitted or readiness.process_birth_id != owner.birth_id:
            return False
        if not self._binding_matches_expected(readiness.binding):
            return False
        if observation.binding is not None:
            prior = observation.binding
            current = readiness.binding
            if (
                current.store_id != prior.store_id
                or current.schema_revision != prior.schema_revision
                or current.database_uuid != prior.database_uuid
                or current.schema_fingerprint != prior.schema_fingerprint
                or int(current.generation) <= int(prior.generation)
            ):
                return False
        return True

    def _record_failure(self, code: str) -> tuple[int, float]:
        failures = int(self._retry_state.consecutive_failures) + 1
        delay = min(
            float(self._policy.maximum_backoff_seconds),
            float(self._policy.initial_backoff_seconds)
            * (float(self._policy.backoff_multiplier) ** max(0, failures - 1)),
        )
        exhausted = failures >= int(self._policy.max_restart_attempts)
        retry_at = self._clock() + delay
        self._retry_state = QuackOwnerRetryState(
            consecutive_failures=failures,
            retry_not_before=retry_at,
            exhausted=exhausted,
            last_failure_code=code,
        )
        return failures, retry_at

    def _record_terminal_failure(self, code: str) -> None:
        """Exhaust recovery after an operation with an unknown live outcome."""

        failures = int(self._retry_state.consecutive_failures) + 1
        self._retry_state = QuackOwnerRetryState(
            consecutive_failures=failures,
            retry_not_before=self._clock(),
            exhausted=True,
            last_failure_code=code,
        )

    def ensure(
        self, observation: QuackOwnerObservation
    ) -> QuackOwnerWatchdogReceipt:
        """Ensure desired-running, restarting only an exactly dead owner."""

        if not isinstance(observation, QuackOwnerObservation):
            raise TypeError("observation must be QuackOwnerObservation")
        with self._guard:
            no_restart = self._non_restart_receipt(observation)
            if no_restart is not None:
                return no_restart
            now = self._clock()
            if self._retry_state.exhausted:
                return self._receipt(
                    observation,
                    WatchdogDisposition.RETRY_EXHAUSTED,
                    "bounded_restart_budget_exhausted",
                )
            if now < self._retry_state.retry_not_before:
                return self._receipt(
                    observation,
                    WatchdogDisposition.BACKOFF,
                    "restart_backoff_active",
                )

            winner = _OneWinnerLock(self.lock_path)
            if not winner.acquire():
                return self._receipt(
                    observation,
                    WatchdogDisposition.LOCK_CONTENDED,
                    "another_recovery_winner_holds_lock",
                )
            try:
                # Re-observe inside the one-winner boundary so a waiter cannot
                # use a stale DEAD observation after another owner recovered.
                current = self._observe_owner() if self._observe_owner else observation
                no_restart = self._non_restart_receipt(current)
                if no_restart is not None:
                    return no_restart
                attempt = int(self._retry_state.consecutive_failures) + 1
                try:
                    owner = self._start_owner()
                except QuackOwnerStartAbsentError:
                    self._record_failure("start_callback_failed")
                    return self._receipt(
                        current,
                        WatchdogDisposition.START_FAILED,
                        "start_callback_failed",
                        restart_attempted=True,
                        restart_attempt=attempt,
                        one_winner_lock_acquired=True,
                    )
                except Exception:  # callback text may contain credentials
                    self._record_terminal_failure("start_outcome_unknown")
                    return self._receipt(
                        current,
                        WatchdogDisposition.START_FAILED,
                        "start_outcome_unknown",
                        restart_attempted=True,
                        restart_attempt=attempt,
                        one_winner_lock_acquired=True,
                    )
                if not isinstance(owner, SpawnedQuackOwner):
                    self._record_terminal_failure("invalid_start_result")
                    return self._receipt(
                        current,
                        WatchdogDisposition.START_FAILED,
                        "invalid_start_result",
                        restart_attempted=True,
                        restart_attempt=attempt,
                        one_winner_lock_acquired=True,
                    )
                try:
                    readiness = self._readiness_probe(owner)
                except Exception:  # callback text may contain credentials
                    readiness = None
                except BaseException:
                    # Operator shutdown/cancellation must not be converted into
                    # a retryable health failure.  Fence the process created by
                    # this attempt before propagating the control-flow signal.
                    # Retain an unconfirmed owner so the runner's final shutdown
                    # boundary can retry exact process-tree termination.
                    try:
                        interrupted_termination = self._terminate_owner(owner)
                    except Exception:  # never expose callback or credential text
                        self._managed_owner = owner
                    else:
                        if not interrupted_termination.termination_confirmed:
                            self._managed_owner = owner
                    raise
                admitted = (
                    isinstance(readiness, AuthenticatedReadiness)
                    and self._readiness_admitted(owner, readiness, current)
                )
                if not admitted:
                    try:
                        termination = self._terminate_owner(owner)
                    except Exception:  # closed reason; never expose callback text
                        termination = ProcessTreeTerminationReceipt(
                            process_birth_id=owner.birth_id,
                            termination_confirmed=False,
                            reason="termination_callback_failed",
                        )
                    if termination.termination_confirmed:
                        self._record_failure("authenticated_readiness_failed")
                    else:
                        self._record_terminal_failure(
                            "spawned_owner_not_fenced"
                        )
                    return self._receipt(
                        current,
                        WatchdogDisposition.READINESS_FAILED,
                        "authenticated_readiness_failed",
                        restart_attempted=True,
                        restart_attempt=attempt,
                        one_winner_lock_acquired=True,
                        process_birth_id=owner.birth_id,
                        termination=termination,
                    )

                assert isinstance(readiness, AuthenticatedReadiness)
                self._managed_owner = owner
                self._retry_state = QuackOwnerRetryState()
                return self._receipt(
                    current,
                    WatchdogDisposition.RESTARTED,
                    "owner_restarted_and_authenticated",
                    operational_ready=True,
                    recovered=True,
                    restart_attempted=True,
                    restart_attempt=attempt,
                    one_winner_lock_acquired=True,
                    authenticated_readiness=True,
                    process_birth_id=owner.birth_id,
                    binding=readiness.binding,
                )
            finally:
                winner.release()

    def adopt_owner(
        self,
        owner: SpawnedQuackOwner,
        observation: QuackOwnerObservation,
        readiness: AuthenticatedReadiness,
    ) -> QuackOwnerWatchdogReceipt:
        """Explicitly retain an already-live managed-local owner for shutdown."""

        if not isinstance(owner, SpawnedQuackOwner):
            raise TypeError("owner must be SpawnedQuackOwner")
        if not isinstance(observation, QuackOwnerObservation):
            raise TypeError("observation must be QuackOwnerObservation")
        with self._guard:
            if self._desired_state is DesiredOwnerState.STOPPED:
                return self._receipt(
                    observation,
                    WatchdogDisposition.INTENTIONALLY_STOPPED,
                    "desired_state_stopped",
                )
            if (
                observation.liveness is not OwnerLiveness.ALIVE
                or observation.health is not OwnerHealth.HEALTHY
                or not observation.authenticated_ready
                or observation.process_birth is None
                or not process_births_match(owner.process_birth, observation.process_birth)
                or not readiness.admitted
                or readiness.process_birth_id != owner.birth_id
                or not self._binding_matches_expected(readiness.binding)
                or observation.binding is None
                or readiness.binding != observation.binding
            ):
                return self._receipt(
                    observation,
                    WatchdogDisposition.ABSTAIN_ALIVE_UNHEALTHY,
                    "adoption_identity_or_readiness_not_admitted",
                )
            self._managed_owner = replace(owner, adopted=True)
            self._retry_state = QuackOwnerRetryState()
            return self._receipt(
                observation,
                WatchdogDisposition.ADOPTED,
                "live_managed_owner_adopted",
                operational_ready=True,
                authenticated_readiness=True,
                process_birth_id=owner.birth_id,
                binding=readiness.binding,
            )

    def stop(
        self, owner: SpawnedQuackOwner | None = None
    ) -> QuackOwnerWatchdogReceipt:
        """Intentionally stop the retained exact owner process tree."""

        with self._guard:
            self._desired_state = DesiredOwnerState.STOPPED
            selected = owner or self._managed_owner
            observation = QuackOwnerObservation(
                process_birth=(None if selected is None else selected.process_birth),
                liveness=(
                    OwnerLiveness.DEAD if selected is None else OwnerLiveness.ALIVE
                ),
                health=OwnerHealth.UNKNOWN,
                absence_proven=(selected is None),
            )
            if selected is None:
                return self._receipt(
                    observation,
                    WatchdogDisposition.INTENTIONALLY_STOPPED,
                    "no_managed_owner_to_stop",
                )
            try:
                termination = self._terminate_owner(selected)
            except Exception:  # closed reason; never expose callback text
                termination = ProcessTreeTerminationReceipt(
                    process_birth_id=selected.birth_id,
                    termination_confirmed=False,
                    reason="termination_callback_failed",
                )
            if termination.termination_confirmed:
                if self._managed_owner is not None and (
                    self._managed_owner.birth_id == selected.birth_id
                ):
                    self._managed_owner = None
                return self._receipt(
                    observation,
                    WatchdogDisposition.STOPPED,
                    "intentional_process_tree_stop_confirmed",
                    process_birth_id=selected.birth_id,
                    termination=termination,
                )
            return self._receipt(
                observation,
                WatchdogDisposition.STOP_FAILED,
                "intentional_process_tree_stop_not_confirmed",
                process_birth_id=selected.birth_id,
                termination=termination,
            )


__all__ = [
    "AuthenticatedReadiness",
    "DesiredOwnerState",
    "OwnerHealth",
    "ProcessTreeTerminationReceipt",
    "QUACK_OWNER_TERMINATION_RECEIPT_SCHEMA",
    "QUACK_OWNER_WATCHDOG_INTERFACE",
    "QUACK_OWNER_WATCHDOG_RECEIPT_SCHEMA",
    "QUACK_OWNER_WATCHDOG_SCHEMA",
    "QuackOwnerBinding",
    "QuackOwnerObservation",
    "QuackOwnerRetryState",
    "QuackOwnerStartAbsentError",
    "QuackOwnerWatchdog",
    "QuackOwnerWatchdogError",
    "QuackOwnerWatchdogPolicy",
    "QuackOwnerWatchdogPolicyError",
    "QuackOwnerWatchdogReceipt",
    "SpawnedQuackOwner",
    "WatchdogDisposition",
    "process_birth_id",
    "process_births_match",
    "terminate_spawned_owner",
]
