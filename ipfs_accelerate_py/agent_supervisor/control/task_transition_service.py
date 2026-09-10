"""Candidate task-transition gateway over :class:`IntentRepository`.

This module is deliberately a *candidate* adapter.  It does not install a
second writer, open a database connection, or provide a compatibility path for
legacy callers.  The existing ``IntentRepository`` remains responsible for
the transactional compare-and-set, task-revision materialization, and domain
event append.  Production routing remains deferred to ASEH-060/ASEH-061.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from threading import Condition, RLock
from typing import Any, Callable, Final, Protocol, TypeVar

from ..task_sources.control_plane_contracts import (
    CommandKind,
    StateAuthorityClass,
    StateCommand,
)
from ..task_sources.intent_repository import (
    IntentReceipt,
    IntentRepository,
    IntentRepositoryConflictError,
    IntentRepositoryError,
)


TASK_TRANSITION_SERVICE_INTERFACE: Final[str] = "TaskTransitionService@1"
TASK_TRANSITION_SERVICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-transition-service@1"
)
PRODUCTION_CUTOVER_DEFERRED_TO: Final[tuple[str, str]] = ("ASEH-060", "ASEH-061")

_COMMAND_PARAMETER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "task_cid",
        "task_cid_or_alias",
        "new_status",
        "status",
        "receipt",
        "evidence_digests",
        # These values identify the durable task claim which authorizes an
        # effect or a terminal transition.  They are deliberately command
        # parameters rather than service constructor state: a caller cannot
        # accidentally reuse a claim after a takeover.
        "lease_id",
        "fencing_token",
        "claim_revision",
    }
)
_TRANSITION_COMMAND_KINDS: Final[frozenset[CommandKind]] = frozenset(
    {CommandKind.APPEND, CommandKind.CLAIM, CommandKind.RELEASE, CommandKind.RECOVER}
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "skipped", "complete", "done", "cancelled", "failed", "quarantined", "rejected"}
)

_ResultT = TypeVar("_ResultT")


class TaskTransitionServiceError(RuntimeError):
    """Base fail-closed error for the candidate transition gateway."""


class TransitionAuthorityError(TaskTransitionServiceError):
    """A command attempted to widen or bypass the repository authority."""


class TransitionCompatibilityBypassError(TransitionAuthorityError):
    """A legacy/direct-write compatibility path was deliberately refused."""


class TransitionConflictError(TaskTransitionServiceError):
    """The command's task revision is stale or otherwise conflicts."""


class TransitionLeaseError(TransitionAuthorityError):
    """An effect or terminal transition lacks current fenced ownership."""


class TransitionIdempotencyConflictError(TransitionConflictError):
    """An idempotency key was reused for a different protected operation."""


class TransitionIntegrityError(TaskTransitionServiceError):
    """Repository output failed the event/materialized-revision invariant."""


class TransitionCommandError(TransitionAuthorityError, ValueError):
    """A state command cannot be interpreted as one closed task transition."""


class TransitionAuthorityWarning(RuntimeWarning):
    """Runtime signal emitted immediately before a compatibility bypass fails."""


@dataclass(frozen=True)
class TaskLeaseFence:
    """The exact task-claim authority tuple required for protected work.

    ``TaskTransitionService`` does not issue these records.  The existing
    durable coordinator (or a production adapter over it) must validate this
    exact tuple while it holds its own authoritative serialization boundary.
    Keeping the tuple closed makes a stale owner, lease, token, epoch, or
    claim revision unrepresentable as a valid protected operation.
    """

    task_cid: str
    owner_session_id: str
    lease_id: str
    fencing_token: int
    fence_epoch: int
    claim_revision: int

    def __post_init__(self) -> None:
        for name in ("task_cid", "owner_session_id", "lease_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TransitionLeaseError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        for name in ("fencing_token", "fence_epoch", "claim_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TransitionLeaseError(f"{name} must be a positive integer")


class TaskLeaseFenceAuthority(Protocol):
    """Runs a callback only while an exact task claim remains authoritative.

    Implementations must validate all fields of ``lease`` immediately before
    the callback and keep the validation and callback in one authoritative
    serialization boundary.  This is what closes the race between a stale
    actor's last read and an owner takeover.  ``DatabaseCoordinator`` already
    provides the underlying durable claim/fence primitives; its integration
    adapter belongs at the production cutover, not in this candidate gateway.
    """

    def execute_fenced(self, lease: TaskLeaseFence, callback: Callable[[], _ResultT]) -> _ResultT:
        """Validate ``lease`` as current, then execute ``callback`` once."""


@dataclass(frozen=True)
class _IdempotentResult:
    """Internal exact binding for a completed protected operation."""

    fingerprint: tuple[Any, ...]
    value: Any


@dataclass(frozen=True)
class TaskTransitionResult:
    """Read-only outcome of one repository-backed task status transition."""

    command_id: str
    task_cid: str
    previous_status: str
    status: str
    revision: int
    changed: bool
    receipt: IntentReceipt
    task: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", MappingProxyType(dict(self.task)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_TRANSITION_SERVICE_SCHEMA,
            "command_id": self.command_id,
            "task_cid": self.task_cid,
            "previous_status": self.previous_status,
            "status": self.status,
            "revision": self.revision,
            "changed": self.changed,
            "receipt": self.receipt.to_dict(),
            "task": dict(self.task),
            "production_cutover_deferred_to": list(PRODUCTION_CUTOVER_DEFERRED_TO),
        }


def _command_parameters(command: StateCommand) -> tuple[str, str, Mapping[str, Any], tuple[str, ...]]:
    """Extract one closed status-CAS request from an admitted state command."""

    if not isinstance(command, StateCommand):
        raise TransitionCommandError("transition requires a StateCommand")
    if command.authority_class is not StateAuthorityClass.AUTHORITATIVE:
        raise TransitionAuthorityError(
            "candidate task transitions require authoritative StateCommand authority"
        )
    if command.command_kind not in _TRANSITION_COMMAND_KINDS:
        raise TransitionCommandError(
            "StateCommand kind is not admitted for a task status transition"
        )
    parameters = dict(command.parameters)
    unknown = set(parameters) - _COMMAND_PARAMETER_KEYS
    if unknown:
        raise TransitionCommandError(
            "task transition command has unsupported parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    task_cid = parameters.get("task_cid", parameters.get("task_cid_or_alias"))
    if "task_cid" in parameters and "task_cid_or_alias" in parameters:
        raise TransitionCommandError("task transition command has ambiguous task identity")
    if not isinstance(task_cid, str) or not task_cid.strip():
        raise TransitionCommandError("task transition command requires task_cid")
    new_status = parameters.get("new_status", parameters.get("status"))
    if "new_status" in parameters and "status" in parameters:
        raise TransitionCommandError("task transition command has ambiguous target status")
    if not isinstance(new_status, str) or not new_status.strip():
        raise TransitionCommandError("task transition command requires new_status")
    receipt = parameters.get("receipt", {})
    if receipt is None:
        receipt = {}
    if not isinstance(receipt, Mapping):
        raise TransitionCommandError("task transition receipt must be a mapping")
    raw_evidence = parameters.get("evidence_digests", ())
    if raw_evidence is None:
        raw_evidence = ()
    if isinstance(raw_evidence, (str, bytes, bytearray)) or not isinstance(raw_evidence, Sequence):
        raise TransitionCommandError("evidence_digests must be a sequence of non-empty strings")
    evidence = tuple(raw_evidence)
    if any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise TransitionCommandError("evidence_digests must be a sequence of non-empty strings")
    if command.expected_revision < 1:
        raise TransitionCommandError("task transition expected_revision must be positive")
    return task_cid.strip(), new_status.strip(), dict(receipt), evidence


class TaskTransitionService:
    """Narrow, repository-only candidate gateway for task status CAS.

    It intentionally exposes neither a connection nor an SQL hook.  This
    makes direct table writes impossible through the service and preserves the
    existing ``IntentRepository`` as the one mutation authority.
    """

    INTERFACE: Final[str] = TASK_TRANSITION_SERVICE_INTERFACE
    SCHEMA: Final[str] = TASK_TRANSITION_SERVICE_SCHEMA
    PRODUCTION_CUTOVER_DEFERRED_TO: Final[tuple[str, str]] = PRODUCTION_CUTOVER_DEFERRED_TO

    def __init__(
        self,
        repository: IntentRepository,
        *,
        lease_fence_authority: TaskLeaseFenceAuthority | None = None,
    ) -> None:
        if not isinstance(repository, IntentRepository):
            raise TransitionAuthorityError(
                "TaskTransitionService requires the existing IntentRepository authority"
            )
        if lease_fence_authority is not None and not callable(
            getattr(lease_fence_authority, "execute_fenced", None)
        ):
            raise TransitionAuthorityError(
                "lease_fence_authority must provide execute_fenced(lease, callback)"
            )
        self._repository = repository
        self._lease_fence_authority = lease_fence_authority
        # The condition is intentionally local to this adapter: it coalesces
        # duplicate delivery attempts before a provider/effect callback runs.
        # It is not a second durable authority; the lease authority remains
        # responsible for cross-process ownership and takeover ordering.
        self._idempotency_condition = Condition(RLock())
        self._idempotent_results: dict[tuple[str, str], _IdempotentResult] = {}
        self._idempotent_in_flight: dict[tuple[str, str], tuple[Any, ...]] = {}

    @property
    def repository(self) -> IntentRepository:
        """The existing authority, exposed for read-only contract inspection."""

        return self._repository

    @property
    def lease_fence_authority(self) -> TaskLeaseFenceAuthority | None:
        """The injected authoritative fenced-execution boundary, if any."""

        return self._lease_fence_authority

    def transition(self, command: StateCommand) -> TaskTransitionResult:
        """Apply one status transition through ``IntentRepository.cas_task_status``.

        No conflict is retried: retrying a stale command can apply a transition
        the caller did not authorize.  Callers must reread and issue a new
        ``StateCommand`` with the new task revision.
        """

        task_key, new_status, receipt, evidence_digests = _command_parameters(command)
        before = self._repository.get_task(task_key)
        if before is None:
            raise KeyError(task_key)
        previous_status = str(before.get("status") or "")
        resolved_cid = str(before.get("task_cid") or "")
        if not resolved_cid:
            raise TransitionIntegrityError("repository returned a task without task_cid")
        operation = lambda: self._transition_once(
            command=command,
            resolved_cid=resolved_cid,
            previous_status=previous_status,
            new_status=new_status,
            receipt=receipt,
            evidence_digests=evidence_digests,
        )
        if self._requires_fenced_authority(new_status):
            lease = self._lease_for_protected_operation(command, resolved_cid)
            return self._execute_fenced_idempotently(
                command=command,
                task_cid=resolved_cid,
                lease=lease,
                operation_name=f"transition:{new_status}",
                callback=operation,
            )
        return operation()

    def execute_effect(self, command: StateCommand, callback: Callable[[], _ResultT]) -> _ResultT:
        """Run one externally effectful callback behind current task fencing.

        A duplicate delivery with the same idempotency key and exact command
        binding receives the original result without invoking ``callback``.
        A key reused for a distinct task, owner, lease, fence, revision, or
        operation fails closed.  The injected authority linearizes execution
        against takeover; merely checking a lease before calling this method
        is deliberately insufficient.
        """

        if not callable(callback):
            raise TransitionCommandError("effect callback must be callable")
        task_key, _new_status, _receipt, _evidence = _command_parameters(command)
        task = self._repository.get_task(task_key)
        if task is None:
            raise KeyError(task_key)
        task_cid = str(task.get("task_cid") or "")
        if not task_cid:
            raise TransitionIntegrityError("repository returned a task without task_cid")
        lease = self._lease_for_protected_operation(command, task_cid)
        return self._execute_fenced_idempotently(
            command=command,
            task_cid=task_cid,
            lease=lease,
            operation_name="effect",
            callback=lambda: self._execute_effect_once(
                task_cid=task_cid,
                command=command,
                callback=callback,
            ),
        )

    def _execute_effect_once(
        self,
        *,
        task_cid: str,
        command: StateCommand,
        callback: Callable[[], _ResultT],
    ) -> _ResultT:
        """Assert the task CAS head at the same fenced boundary as an effect.

        ``IntentRepository.cas_task_status`` treats an equal status as a
        revision-CAS no-op: it verifies the exact revision without appending a
        spurious event or advancing the task.  That lets external effects bind
        to the task's current revision even when this call itself has no status
        transition to persist.
        """

        task = self._repository.get_task(task_cid)
        if task is None:
            raise KeyError(task_cid)
        current_status = str(task.get("status") or "")
        if not current_status:
            raise TransitionIntegrityError("repository returned a task without status")
        try:
            receipt = self._repository.cas_task_status(
                task_cid=task_cid,
                expected_revision=command.expected_revision,
                new_status=current_status,
            )
        except IntentRepositoryConflictError as exc:
            raise TransitionConflictError("effectful execution CAS conflict") from exc
        except IntentRepositoryError as exc:
            raise TaskTransitionServiceError("IntentRepository refused effect CAS") from exc
        if receipt.changed or receipt.revision != command.expected_revision:
            raise TransitionIntegrityError("effectful execution did not preserve its CAS head")
        return callback()

    def _transition_once(
        self,
        *,
        command: StateCommand,
        resolved_cid: str,
        previous_status: str,
        new_status: str,
        receipt: Mapping[str, Any],
        evidence_digests: tuple[str, ...],
    ) -> TaskTransitionResult:
        """Perform the repository CAS and verify its durable projection."""

        try:
            mutation_receipt = self._repository.cas_task_status(
                task_cid=resolved_cid,
                expected_revision=command.expected_revision,
                new_status=new_status,
                receipt=receipt,
                evidence_digests=evidence_digests,
            )
        except IntentRepositoryConflictError as exc:
            raise TransitionConflictError("task transition CAS conflict") from exc
        except IntentRepositoryError as exc:
            raise TaskTransitionServiceError("IntentRepository refused task transition") from exc

        after = self._repository.get_task(resolved_cid)
        if after is None:
            raise TransitionIntegrityError("task disappeared after repository CAS")
        revision = int(after.get("revision") or 0)
        status = str(after.get("status") or "")
        self._assert_revision_parity(
            task_cid=resolved_cid,
            expected_status=new_status,
            materialized_revision=revision,
            materialized_status=status,
            receipt=mutation_receipt,
        )
        return TaskTransitionResult(
            command_id=command.command_id,
            task_cid=resolved_cid,
            previous_status=previous_status,
            status=status,
            revision=revision,
            changed=bool(mutation_receipt.changed),
            receipt=mutation_receipt,
            task=after,
        )

    @staticmethod
    def _requires_fenced_authority(new_status: str) -> bool:
        """Terminalization is an effectful operation even without a callback."""

        return new_status.strip().lower().replace("-", "_").replace(" ", "_") in _TERMINAL_STATUSES

    def _lease_for_protected_operation(
        self, command: StateCommand, task_cid: str
    ) -> TaskLeaseFence:
        """Close the command-to-claim binding before it reaches an effect."""

        if self._lease_fence_authority is None:
            raise TransitionLeaseError(
                "effectful execution and terminalization require a lease_fence_authority"
            )
        parameters = dict(command.parameters)
        lease_id = parameters.get("lease_id")
        token = parameters.get("fencing_token")
        claim_revision = parameters.get("claim_revision")
        if not isinstance(lease_id, str) or not lease_id.strip():
            raise TransitionLeaseError("protected operation requires lease_id")
        for name, value in (("fencing_token", token), ("claim_revision", claim_revision)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TransitionLeaseError(f"protected operation requires positive {name}")
        if command.fence_epoch < 1:
            raise TransitionLeaseError("protected operation requires a positive fence_epoch")
        return TaskLeaseFence(
            task_cid=task_cid,
            owner_session_id=command.session_id,
            lease_id=lease_id,
            fencing_token=token,
            fence_epoch=command.fence_epoch,
            claim_revision=claim_revision,
        )

    def _execute_fenced_idempotently(
        self,
        *,
        command: StateCommand,
        task_cid: str,
        lease: TaskLeaseFence,
        operation_name: str,
        callback: Callable[[], _ResultT],
    ) -> _ResultT:
        """Linearize a protected effect and coalesce exact duplicate delivery."""

        fingerprint = (
            operation_name,
            command.command_kind.value,
            command.expected_generation,
            command.expected_revision,
            lease.owner_session_id,
            lease.lease_id,
            lease.fencing_token,
            lease.fence_epoch,
            lease.claim_revision,
        )
        key = (task_cid, command.idempotency_key)
        with self._idempotency_condition:
            while key in self._idempotent_in_flight:
                if self._idempotent_in_flight[key] != fingerprint:
                    raise TransitionIdempotencyConflictError(
                        "idempotency key is already bound to another protected operation"
                    )
                self._idempotency_condition.wait()
            existing = self._idempotent_results.get(key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise TransitionIdempotencyConflictError(
                        "idempotency key is already bound to another protected operation"
                    )
                return existing.value
            self._idempotent_in_flight[key] = fingerprint

        try:
            authority = self._lease_fence_authority
            assert authority is not None  # established by _lease_for_protected_operation
            value = authority.execute_fenced(lease, callback)
        except BaseException:
            with self._idempotency_condition:
                self._idempotent_in_flight.pop(key, None)
                self._idempotency_condition.notify_all()
            raise

        with self._idempotency_condition:
            self._idempotent_in_flight.pop(key, None)
            self._idempotent_results[key] = _IdempotentResult(
                fingerprint=fingerprint,
                value=value,
            )
            self._idempotency_condition.notify_all()
        return value

    apply = transition
    execute = transition

    def _assert_revision_parity(
        self,
        *,
        task_cid: str,
        expected_status: str,
        materialized_revision: int,
        materialized_status: str,
        receipt: IntentReceipt,
    ) -> None:
        """Refuse a response whose event receipt and task projection diverge."""

        if receipt.revision != materialized_revision:
            raise TransitionIntegrityError(
                "repository event revision does not match materialized task revision"
            )
        if materialized_status != expected_status.strip().lower().replace("-", "_").replace(" ", "_"):
            raise TransitionIntegrityError("repository materialized status differs from command target")
        if not receipt.changed:
            if receipt.event_id:
                raise TransitionIntegrityError("unchanged transition unexpectedly has a durable event")
            return
        if not receipt.event_id:
            raise TransitionIntegrityError("changed transition has no durable event receipt")
        details = dict(receipt.details)
        if int(details.get("revision") or 0) != materialized_revision:
            raise TransitionIntegrityError("repository event body revision does not match task revision")
        events = self._repository.list_events(
            after_global_sequence=max(0, receipt.global_sequence - 1), limit=1
        )
        if len(events) != 1 or str(events[0].get("event_id") or "") != receipt.event_id:
            raise TransitionIntegrityError("repository event receipt cannot be read back exactly")
        event_body = events[0].get("body")
        if not isinstance(event_body, Mapping):
            raise TransitionIntegrityError("repository event body is malformed")
        payload = event_body.get("body")
        if not isinstance(payload, Mapping):
            raise TransitionIntegrityError("repository event payload is malformed")
        if (
            str(payload.get("task_cid") or "") != task_cid
            or int(payload.get("revision") or 0) != materialized_revision
        ):
            raise TransitionIntegrityError(
                "durable event payload does not match materialized task revision"
            )

    def reject_compatibility_bypass(self, *, caller: str, operation: str = "task mutation") -> None:
        """Warn and reject a proposed legacy/direct transition path.

        The warning is intentionally emitted before raising so currently
        running candidate integrations are observable without granting a
        fallback mutation path.
        """

        caller_text = str(caller or "").strip() or "unknown"
        operation_text = str(operation or "").strip() or "task mutation"
        warnings.warn(
            f"compatibility bypass rejected for {operation_text} from {caller_text}; "
            "use TaskTransitionService.transition with an authoritative StateCommand",
            TransitionAuthorityWarning,
            stacklevel=2,
        )
        raise TransitionCompatibilityBypassError(
            "compatibility bypass is not an admitted task-state mutation path"
        )

    def transition_legacy(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility trap retained only to warn and fail closed."""

        del args
        self.reject_compatibility_bypass(caller=str(kwargs.get("caller") or "legacy"))

    legacy_transition = transition_legacy


__all__ = (
    "PRODUCTION_CUTOVER_DEFERRED_TO",
    "TASK_TRANSITION_SERVICE_INTERFACE",
    "TASK_TRANSITION_SERVICE_SCHEMA",
    "TaskLeaseFence",
    "TaskLeaseFenceAuthority",
    "TaskTransitionResult",
    "TaskTransitionService",
    "TaskTransitionServiceError",
    "TransitionAuthorityError",
    "TransitionAuthorityWarning",
    "TransitionCommandError",
    "TransitionCompatibilityBypassError",
    "TransitionConflictError",
    "TransitionIdempotencyConflictError",
    "TransitionIntegrityError",
    "TransitionLeaseError",
)
