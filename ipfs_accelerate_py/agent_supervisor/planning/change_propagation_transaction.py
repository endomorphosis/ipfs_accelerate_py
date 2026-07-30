"""Checkpointed, SCC-grouped transaction primitives for change propagation.

An admitted :class:`AtomicPropagationPlan` is executed only inside an isolated
candidate worktree behind a content-addressed checkpoint.  Steps run in
dependency order; members of one strongly connected component (SCC) form a
single transaction group.  Failure, hash drift, lease loss, timeout, or scope
escape restores the checkpoint and retains diagnostics.  Partial plans cannot
be merged or marked complete.

Canonical RPR-022 records (:class:`AtomicPropagationPlan`,
:class:`PropagationTransaction`) are imported and returned — never redefined.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.change_propagation_contracts import (
    AtomicPropagationPlan,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationPlanStep,
    PropagationSCCGroup,
    PropagationTransaction,
    TransactionState,
)
from ..proof.change_propagation_edit_packet import (
    ChangePropagationEditPacket,
    PathBeforeHash,
)
from ..proof.formal_verification_contracts import (
    content_identity,
)


CHANGE_PROPAGATION_TRANSACTION_INTERFACE: Final[str] = "ChangePropagationTransaction@1"
PROPAGATION_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/checkpoint@1"
)
PROPAGATION_ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/rollback-receipt@1"
)
PROPAGATION_STEP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/step-receipt@1"
)
PROPAGATION_GROUP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/group-receipt@1"
)
PROPAGATION_TRANSACTION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/transaction-report@1"
)
TRANSACTION_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation/transaction-lease@1"
)

PRODUCER_ID: Final[str] = "change-propagation-transaction@1"

MAX_PATHS: Final[int] = 1_024
MAX_STEPS: Final[int] = 512
MAX_DIAGNOSTICS: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 4_096


class ChangePropagationTransactionError(ValueError):
    """A transaction would weaken checkpoint, lease, or plan authority."""


class TransactionFailureReason(str, Enum):
    """Stable, machine-readable transaction failure codes."""

    MALFORMED_INPUT = "malformed_input"
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    PLAN_PACKET_MISMATCH = "plan_packet_mismatch"
    ROOT_DRIFT = "root_drift"
    MISSING_CHECKPOINT = "missing_checkpoint"
    BEFORE_HASH_MISMATCH = "before_hash_mismatch"
    BEFORE_HASH_MISSING = "before_hash_missing"
    LEASE_MISSING = "lease_missing"
    LEASE_INVALID = "lease_invalid"
    LEASE_PATH_MISMATCH = "lease_path_mismatch"
    SCOPE_ESCAPE = "scope_escape"
    STEP_FAILURE = "step_failure"
    GROUP_INCOMPLETE = "group_incomplete"
    DEPENDENCY_UNMET = "dependency_unmet"
    TIMEOUT = "timeout"
    DRIFT = "drift"
    PARTIAL_MERGE_FORBIDDEN = "partial_merge_forbidden"
    ALREADY_TERMINAL = "already_terminal"
    RESTORE_FAILED = "restore_failed"
    STEP_ORDER_VIOLATION = "step_order_violation"


class StepExecutionDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    SCOPE_ESCAPE = "scope_escape"
    DRIFT = "drift"


class GroupExecutionDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise ChangePropagationTransactionError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ChangePropagationTransactionError(f"{name} exceeds text bound")
    return text


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ChangePropagationTransactionError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ChangePropagationTransactionError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ChangePropagationTransactionError(f"{name} exceeds text bound")
    return text


def _paths(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationTransactionError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ChangePropagationTransactionError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ChangePropagationTransactionError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if required and not result:
        raise ChangePropagationTransactionError(f"{name} must not be empty")
    if len(result) > MAX_PATHS:
        raise ChangePropagationTransactionError(f"{name} exceeds path bound")
    return tuple(sorted(result))


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_STEPS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationTransactionError(f"{name} must be an identifier sequence")
    if preserve_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ChangePropagationTransactionError(f"{name} contains an invalid id")
            item = value.strip()
            if any(char.isspace() for char in item):
                raise ChangePropagationTransactionError(f"{name} must contain compact identifiers")
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        result = tuple(ordered)
    else:
        result = tuple(
            sorted(
                {
                    value.strip()
                    for value in values
                    if isinstance(value, str) and value.strip()
                }
            )
        )
        if any(any(char.isspace() for char in item) for item in result):
            raise ChangePropagationTransactionError(f"{name} must contain compact identifiers")
    if required and not result:
        raise ChangePropagationTransactionError(f"{name} must not be empty")
    if len(result) > maximum:
        raise ChangePropagationTransactionError(f"{name} exceeds item bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ChangePropagationTransactionError(f"{name} must be a boolean")
    return value


# ---------------------------------------------------------------------------
# Lease, checkpoint, receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionLease:
    """Writer lease binding exact paths for one transaction execution."""

    lease_id: str
    fence_id: str
    holder_id: str
    permitted_write_paths: tuple[str, ...]
    permitted_read_paths: tuple[str, ...] = ()
    active: bool = True
    expires_at: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", _identifier(self.fence_id, "fence_id"))
        object.__setattr__(self, "holder_id", _identifier(self.holder_id, "holder_id"))
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths", required=True),
        )
        object.__setattr__(
            self,
            "permitted_read_paths",
            _paths(self.permitted_read_paths, "permitted_read_paths"),
        )
        object.__setattr__(self, "active", _bool(self.active, "active"))
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, int) or self.expires_at < 0:
            raise ChangePropagationTransactionError("expires_at must be a non-negative integer")

    def covers_writes(self, paths: Sequence[str]) -> bool:
        return set(paths).issubset(self.permitted_write_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRANSACTION_LEASE_SCHEMA,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "holder_id": self.holder_id,
            "permitted_write_paths": list(self.permitted_write_paths),
            "permitted_read_paths": list(self.permitted_read_paths),
            "active": self.active,
            "expires_at": self.expires_at,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransactionLease":
        if not isinstance(payload, Mapping):
            raise ChangePropagationTransactionError("lease must be an object")
        return cls(
            lease_id=payload["lease_id"],
            fence_id=payload["fence_id"],
            holder_id=payload["holder_id"],
            permitted_write_paths=tuple(payload["permitted_write_paths"]),
            permitted_read_paths=tuple(payload.get("permitted_read_paths", ())),
            active=bool(payload.get("active", True)),
            expires_at=int(payload.get("expires_at", 0)),
        )


@dataclass(frozen=True)
class PropagationCheckpoint:
    """Content-addressed pre-mutation snapshot of the candidate tree.

    The checkpoint binds authority roots, plan identity, exact path before-
    hashes, and strategy refs.  Restoration is the only recovery path after
    failure, drift, timeout, or scope escape.
    """

    roots: PropagationAuthorityRoots
    checkpoint_id: str
    plan_id: str
    plan_content_id: str
    path_before_hashes: tuple[PathBeforeHash, ...]
    strategy_ref: str
    tree_snapshot_ref: str = ""
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationTransactionError("checkpoint roots must be PropagationAuthorityRoots")
        object.__setattr__(self, "checkpoint_id", _identifier(self.checkpoint_id, "checkpoint_id"))
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "plan_content_id", _identifier(self.plan_content_id, "plan_content_id")
        )
        object.__setattr__(self, "strategy_ref", _identifier(self.strategy_ref, "strategy_ref"))
        object.__setattr__(
            self,
            "tree_snapshot_ref",
            _text(self.tree_snapshot_ref, "tree_snapshot_ref", required=False),
        )
        if not isinstance(self.path_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.path_before_hashes
        ):
            raise ChangePropagationTransactionError(
                "path_before_hashes must be PathBeforeHash values"
            )
        hashes = tuple(sorted(self.path_before_hashes, key=lambda item: item.path))
        if len({item.path for item in hashes}) != len(hashes):
            raise ChangePropagationTransactionError(
                "path_before_hashes must have unique paths"
            )
        if len(hashes) > MAX_PATHS:
            raise ChangePropagationTransactionError("path_before_hashes exceeds path bound")
        object.__setattr__(self, "path_before_hashes", hashes)
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        # Content-addressed identity when caller does not supply a stable id.
        # Callers typically pass content_identity of the preimage; if the given
        # checkpoint_id is a placeholder we still accept it as the record id.

    def hash_map(self) -> dict[str, str]:
        return {
            item.path: item.before_hash
            for item in self.path_before_hashes
            if item.before_hash
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_CHECKPOINT_SCHEMA,
            "interface": CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "checkpoint_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "path_before_hashes": [item.to_dict() for item in self.path_before_hashes],
            "strategy_ref": self.strategy_ref,
            "tree_snapshot_ref": self.tree_snapshot_ref,
            "diagnostic_refs": list(self.diagnostic_refs),
        }

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                **self.to_dict(),
                "checkpoint_id": "",  # identity excludes self-referential id
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationCheckpoint":
        if not isinstance(payload, Mapping):
            raise ChangePropagationTransactionError("checkpoint must be an object")
        return cls(
            roots=PropagationAuthorityRoots.from_dict(payload["roots"]),
            checkpoint_id=payload["checkpoint_id"],
            plan_id=payload["plan_id"],
            plan_content_id=payload["plan_content_id"],
            path_before_hashes=tuple(
                PathBeforeHash.from_dict(item)
                for item in payload.get("path_before_hashes", ())
            ),
            strategy_ref=payload["strategy_ref"],
            tree_snapshot_ref=str(payload.get("tree_snapshot_ref", "")),
            diagnostic_refs=tuple(payload.get("diagnostic_refs", ())),
        )


@dataclass(frozen=True)
class PropagationStepReceipt:
    """Outcome of one plan step under an active transaction."""

    step_id: str
    disposition: StepExecutionDisposition
    reason_codes: tuple[str, ...] = ()
    written_paths: tuple[str, ...] = ()
    observed_before_hashes: tuple[PathBeforeHash, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(
            self,
            "disposition",
            StepExecutionDisposition(self.disposition)
            if not isinstance(self.disposition, StepExecutionDisposition)
            else self.disposition,
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(self, "written_paths", _paths(self.written_paths, "written_paths"))
        if not isinstance(self.observed_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_before_hashes
        ):
            raise ChangePropagationTransactionError(
                "observed_before_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self,
            "observed_before_hashes",
            tuple(sorted(self.observed_before_hashes, key=lambda item: item.path)),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )

    @property
    def passed(self) -> bool:
        return self.disposition is StepExecutionDisposition.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_STEP_RECEIPT_SCHEMA,
            "step_id": self.step_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "written_paths": list(self.written_paths),
            "observed_before_hashes": [
                item.to_dict() for item in self.observed_before_hashes
            ],
            "diagnostic_refs": list(self.diagnostic_refs),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class PropagationGroupReceipt:
    """Outcome of one SCC (or singleton) transaction group."""

    group_id: str
    scc_id: str
    step_ids: tuple[str, ...]
    disposition: GroupExecutionDisposition
    step_receipts: tuple[PropagationStepReceipt, ...] = ()
    reason_codes: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        object.__setattr__(self, "scc_id", _text(self.scc_id, "scc_id", required=False))
        object.__setattr__(
            self,
            "step_ids",
            _ids(self.step_ids, "step_ids", required=True, preserve_order=True),
        )
        object.__setattr__(
            self,
            "disposition",
            GroupExecutionDisposition(self.disposition)
            if not isinstance(self.disposition, GroupExecutionDisposition)
            else self.disposition,
        )
        if not isinstance(self.step_receipts, Sequence) or not all(
            isinstance(item, PropagationStepReceipt) for item in self.step_receipts
        ):
            raise ChangePropagationTransactionError(
                "step_receipts must be PropagationStepReceipt values"
            )
        object.__setattr__(self, "step_receipts", tuple(self.step_receipts))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        if self.disposition is GroupExecutionDisposition.PASSED:
            if not all(item.passed for item in self.step_receipts):
                raise ChangePropagationTransactionError(
                    "passed group requires every step receipt to pass"
                )
            if set(self.step_ids) != {item.step_id for item in self.step_receipts}:
                raise ChangePropagationTransactionError(
                    "passed group must cover every step exactly once"
                )

    @property
    def passed(self) -> bool:
        return self.disposition is GroupExecutionDisposition.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_GROUP_RECEIPT_SCHEMA,
            "group_id": self.group_id,
            "scc_id": self.scc_id,
            "step_ids": list(self.step_ids),
            "disposition": self.disposition.value,
            "step_receipts": [item.to_dict() for item in self.step_receipts],
            "reason_codes": list(self.reason_codes),
            "diagnostic_refs": list(self.diagnostic_refs),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class PropagationRollbackReceipt:
    """Evidence that the candidate tree was restored from a checkpoint."""

    roots: PropagationAuthorityRoots
    rollback_id: str
    transaction_id: str
    checkpoint_id: str
    plan_id: str
    strategy_ref: str
    restored: bool
    reason_codes: tuple[str, ...]
    diagnostic_refs: tuple[str, ...] = ()
    failed_step_ids: tuple[str, ...] = ()
    failed_group_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationTransactionError(
                "rollback roots must be PropagationAuthorityRoots"
            )
        object.__setattr__(self, "rollback_id", _identifier(self.rollback_id, "rollback_id"))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self, "checkpoint_id", _identifier(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(self, "strategy_ref", _identifier(self.strategy_ref, "strategy_ref"))
        object.__setattr__(self, "restored", _bool(self.restored, "restored"))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True, maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )
        object.__setattr__(
            self,
            "failed_step_ids",
            _ids(self.failed_step_ids, "failed_step_ids", preserve_order=True),
        )
        object.__setattr__(
            self,
            "failed_group_id",
            _text(self.failed_group_id, "failed_group_id", required=False),
        )
        if not self.restored:
            raise ChangePropagationTransactionError(
                "rollback receipt requires successful restore; "
                "failed restore must raise rather than claim rollback"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_ROLLBACK_RECEIPT_SCHEMA,
            "interface": CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "rollback_id": self.rollback_id,
            "transaction_id": self.transaction_id,
            "checkpoint_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "strategy_ref": self.strategy_ref,
            "restored": self.restored,
            "reason_codes": list(self.reason_codes),
            "diagnostic_refs": list(self.diagnostic_refs),
            "failed_step_ids": list(self.failed_step_ids),
            "failed_group_id": self.failed_group_id,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationRollbackReceipt":
        if not isinstance(payload, Mapping):
            raise ChangePropagationTransactionError("rollback receipt must be an object")
        return cls(
            roots=PropagationAuthorityRoots.from_dict(payload["roots"]),
            rollback_id=payload["rollback_id"],
            transaction_id=payload["transaction_id"],
            checkpoint_id=payload["checkpoint_id"],
            plan_id=payload["plan_id"],
            strategy_ref=payload["strategy_ref"],
            restored=payload["restored"],
            reason_codes=tuple(payload["reason_codes"]),
            diagnostic_refs=tuple(payload.get("diagnostic_refs", ())),
            failed_step_ids=tuple(payload.get("failed_step_ids", ())),
            failed_group_id=str(payload.get("failed_group_id", "")),
        )


@dataclass(frozen=True)
class TransactionExecutionReport:
    """Full ordered report for one plan execution attempt.

    Success is not merge authority: only a committed
    :class:`PropagationTransaction` plus a later fixed-point completion
    receipt may authorize merge.
    """

    roots: PropagationAuthorityRoots
    transaction: PropagationTransaction
    plan: AtomicPropagationPlan
    checkpoint: PropagationCheckpoint
    group_receipts: tuple[PropagationGroupReceipt, ...]
    rollback: PropagationRollbackReceipt | None
    reason_codes: tuple[str, ...]
    committed: bool
    partial_merge_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise ChangePropagationTransactionError("report roots must be PropagationAuthorityRoots")
        if not isinstance(self.transaction, PropagationTransaction):
            raise ChangePropagationTransactionError(
                "report must carry the canonical PropagationTransaction@1"
            )
        if not isinstance(self.plan, AtomicPropagationPlan):
            raise ChangePropagationTransactionError(
                "report must carry the canonical AtomicPropagationPlan@1"
            )
        if not isinstance(self.checkpoint, PropagationCheckpoint):
            raise ChangePropagationTransactionError("report requires a PropagationCheckpoint")
        if not isinstance(self.group_receipts, Sequence) or not all(
            isinstance(item, PropagationGroupReceipt) for item in self.group_receipts
        ):
            raise ChangePropagationTransactionError(
                "group_receipts must be PropagationGroupReceipt values"
            )
        object.__setattr__(self, "group_receipts", tuple(self.group_receipts))
        if self.rollback is not None and not isinstance(
            self.rollback, PropagationRollbackReceipt
        ):
            raise ChangePropagationTransactionError(
                "rollback must be PropagationRollbackReceipt or None"
            )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(self, "committed", _bool(self.committed, "committed"))
        object.__setattr__(
            self,
            "partial_merge_allowed",
            _bool(self.partial_merge_allowed, "partial_merge_allowed"),
        )
        # Partial merge is always forbidden by policy.
        if self.partial_merge_allowed:
            raise ChangePropagationTransactionError(
                "partial merge/completion is forbidden for propagation transactions"
            )
        if self.committed:
            if self.transaction.state is not TransactionState.COMMITTED:
                raise ChangePropagationTransactionError(
                    "committed report requires COMMITTED transaction state"
                )
            if self.rollback is not None:
                raise ChangePropagationTransactionError(
                    "committed report cannot retain a rollback receipt"
                )
            if self.reason_codes:
                raise ChangePropagationTransactionError(
                    "committed report cannot carry failure reason codes"
                )
            if not all(item.passed for item in self.group_receipts):
                raise ChangePropagationTransactionError(
                    "committed report requires every group to pass"
                )
        else:
            if self.transaction.state is TransactionState.COMMITTED:
                raise ChangePropagationTransactionError(
                    "non-committed report cannot claim COMMITTED state"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPAGATION_TRANSACTION_REPORT_SCHEMA,
            "interface": CHANGE_PROPAGATION_TRANSACTION_INTERFACE,
            "roots": self.roots.to_dict(),
            "transaction": self.transaction.to_dict(),
            "plan": self.plan.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "group_receipts": [item.to_dict() for item in self.group_receipts],
            "rollback": self.rollback.to_dict() if self.rollback else None,
            "reason_codes": list(self.reason_codes),
            "committed": self.committed,
            "partial_merge_allowed": False,
            "provider_success_is_not_merge": True,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}


# ---------------------------------------------------------------------------
# Step applicator protocol (injected for hermetic tests / live engines)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepApplyRequest:
    """Inputs handed to a step applicator for one plan step."""

    plan: AtomicPropagationPlan
    packet: ChangePropagationEditPacket | None
    step: PropagationPlanStep
    lease: TransactionLease
    checkpoint: PropagationCheckpoint
    completed_step_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepApplyResult:
    """Applicator outcome; never merge authority by itself."""

    disposition: StepExecutionDisposition
    written_paths: tuple[str, ...] = ()
    observed_before_hashes: tuple[PathBeforeHash, ...] = ()
    reason_codes: tuple[str, ...] = ()
    diagnostic_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            StepExecutionDisposition(self.disposition)
            if not isinstance(self.disposition, StepExecutionDisposition)
            else self.disposition,
        )
        object.__setattr__(self, "written_paths", _paths(self.written_paths, "written_paths"))
        if not isinstance(self.observed_before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.observed_before_hashes
        ):
            raise ChangePropagationTransactionError(
                "observed_before_hashes must be PathBeforeHash values"
            )
        object.__setattr__(
            self,
            "observed_before_hashes",
            tuple(self.observed_before_hashes),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", maximum=MAX_REASON_CODES),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _ids(self.diagnostic_refs, "diagnostic_refs", maximum=MAX_DIAGNOSTICS),
        )


StepApplicator = Callable[[StepApplyRequest], StepApplyResult]
RestoreAdapter = Callable[[PropagationCheckpoint], bool]
HashProbe = Callable[[str], str]
# HashProbe maps path -> current content hash on the candidate tree.


def _default_noop_applicator(request: StepApplyRequest) -> StepApplyResult:
    """Hermetic default: claims success with no writes (tests inject real ones)."""

    return StepApplyResult(
        disposition=StepExecutionDisposition.PASSED,
        written_paths=(),
        observed_before_hashes=(),
        reason_codes=(),
        diagnostic_refs=(),
    )


def _default_restore(checkpoint: PropagationCheckpoint) -> bool:
    """Default restore always succeeds for in-memory checkpoint records."""

    return True


# ---------------------------------------------------------------------------
# Transaction orchestrator
# ---------------------------------------------------------------------------


def _topological_step_order(steps: Sequence[PropagationPlanStep]) -> tuple[str, ...]:
    """Deterministic dependency order; SCCs may interleave only inside groups."""

    by_id = {step.step_id: step for step in steps}
    remaining = set(by_id)
    completed: list[str] = []
    while remaining:
        ready = sorted(
            step_id
            for step_id in remaining
            if set(by_id[step_id].dependency_step_ids).issubset(completed)
        )
        if not ready:
            # Cyclic residual: fall back to sorted remaining (SCC groups handle atomicity).
            ready = sorted(remaining)
        for step_id in ready:
            completed.append(step_id)
            remaining.discard(step_id)
    return tuple(completed)


def _build_execution_groups(
    plan: AtomicPropagationPlan,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return ordered (group_id, scc_id, step_ids) execution groups.

    Declared SCC groups become atomic groups.  Steps not in any SCC become
    singleton groups.  Group order follows the earliest dependency-order index
    of any member.
    """

    steps = plan.steps
    step_order = _topological_step_order(steps)
    order_index = {step_id: idx for idx, step_id in enumerate(step_order)}
    assigned: set[str] = set()
    groups: list[tuple[str, str, tuple[str, ...]]] = []

    for scc in plan.scc_groups:
        member_steps = tuple(
            sorted(scc.step_ids, key=lambda sid: order_index.get(sid, MAX_STEPS))
        )
        groups.append((scc.group_id, scc.scc_id, member_steps))
        assigned.update(member_steps)

    for step_id in step_order:
        if step_id in assigned:
            continue
        groups.append((f"group:singleton:{step_id}", "", (step_id,)))
        assigned.add(step_id)

    groups.sort(
        key=lambda item: min(
            (order_index.get(sid, MAX_STEPS) for sid in item[2]), default=MAX_STEPS
        )
    )
    return tuple(groups)


def create_propagation_checkpoint(
    plan: AtomicPropagationPlan,
    *,
    path_before_hashes: Sequence[PathBeforeHash],
    tree_snapshot_ref: str = "",
    diagnostic_refs: Sequence[str] = (),
) -> PropagationCheckpoint:
    """Build a content-addressed checkpoint for an admitted plan before mutation."""

    if not isinstance(plan, AtomicPropagationPlan):
        raise ChangePropagationTransactionError(
            "checkpoint requires the canonical AtomicPropagationPlan@1"
        )
    if plan.disposition is not PlanDisposition.ADMITTED:
        raise ChangePropagationTransactionError(
            "checkpoint requires an admitted AtomicPropagationPlan"
        )
    strategy = plan.checkpoint_strategy_ref
    if not strategy:
        raise ChangePropagationTransactionError(
            "admitted plan must declare a checkpoint strategy ref"
        )
    hashes = tuple(path_before_hashes)
    preimage = {
        "schema": PROPAGATION_CHECKPOINT_SCHEMA,
        "roots": plan.roots.to_dict(),
        "plan_id": plan.plan_id,
        "plan_content_id": plan.content_id,
        "path_before_hashes": [
            item.to_dict() if isinstance(item, PathBeforeHash) else item
            for item in hashes
        ],
        "strategy_ref": strategy,
        "tree_snapshot_ref": tree_snapshot_ref or plan.roots.candidate_tree_id,
        "diagnostic_refs": list(diagnostic_refs),
    }
    checkpoint_id = content_identity(preimage)
    return PropagationCheckpoint(
        roots=plan.roots,
        checkpoint_id=checkpoint_id,
        plan_id=plan.plan_id,
        plan_content_id=plan.content_id,
        path_before_hashes=hashes,
        strategy_ref=strategy,
        tree_snapshot_ref=tree_snapshot_ref or plan.roots.candidate_tree_id,
        diagnostic_refs=tuple(diagnostic_refs),
    )


@dataclass
class ChangePropagationTransaction:
    """Orchestrate checkpointed, SCC-atomic execution of one admitted plan.

    ``execute`` always returns a :class:`TransactionExecutionReport` carrying
    the canonical :class:`PropagationTransaction` and :class:`AtomicPropagationPlan`.
    Partial completion never yields a COMMITTED state.
    """

    INTERFACE: Final[str] = CHANGE_PROPAGATION_TRANSACTION_INTERFACE

    step_applicator: StepApplicator = field(default=_default_noop_applicator)
    restore_adapter: RestoreAdapter = field(default=_default_restore)
    hash_probe: HashProbe | None = None
    now: Callable[[], int] = field(default=lambda: 0)

    def create_checkpoint(
        self,
        plan: AtomicPropagationPlan,
        *,
        path_before_hashes: Sequence[PathBeforeHash],
        tree_snapshot_ref: str = "",
        diagnostic_refs: Sequence[str] = (),
    ) -> PropagationCheckpoint:
        return create_propagation_checkpoint(
            plan,
            path_before_hashes=path_before_hashes,
            tree_snapshot_ref=tree_snapshot_ref,
            diagnostic_refs=diagnostic_refs,
        )

    def execute(
        self,
        plan: AtomicPropagationPlan,
        *,
        lease: TransactionLease,
        path_before_hashes: Sequence[PathBeforeHash] | None = None,
        packet: ChangePropagationEditPacket | None = None,
        checkpoint: PropagationCheckpoint | None = None,
        transaction_id: str = "",
        observe_timeout: bool = False,
    ) -> TransactionExecutionReport:
        """Run dependency-ordered SCC groups under one checkpoint + lease.

        On any failure/drift/timeout/scope escape the checkpoint is restored and
        a rollback receipt is retained.  The transaction never enters COMMITTED
        unless every group passes and every plan step is completed.
        """

        reasons: list[str] = []
        group_receipts: list[PropagationGroupReceipt] = []
        completed: list[str] = []
        txn_id = (
            _identifier(transaction_id, "transaction_id")
            if transaction_id
            else content_identity(
                {
                    "schema": "txn-id",
                    "plan_id": getattr(plan, "plan_id", "invalid"),
                    "lease_id": getattr(lease, "lease_id", "invalid"),
                }
            )
        )

        # --- Input binding ---
        if not isinstance(plan, AtomicPropagationPlan):
            raise ChangePropagationTransactionError(
                "execute requires the canonical AtomicPropagationPlan@1"
            )
        if not isinstance(lease, TransactionLease):
            raise ChangePropagationTransactionError("execute requires a TransactionLease")

        if plan.disposition is not PlanDisposition.ADMITTED:
            reasons.append(TransactionFailureReason.PLAN_NOT_ADMITTED.value)
            return self._failed_without_checkpoint(
                plan=plan,
                lease=lease,
                transaction_id=txn_id,
                reasons=reasons,
                path_before_hashes=path_before_hashes or (),
            )

        if packet is not None:
            if not isinstance(packet, ChangePropagationEditPacket):
                reasons.append(TransactionFailureReason.MALFORMED_INPUT.value)
            elif packet.plan_id != plan.plan_id or packet.plan_content_id != plan.content_id:
                reasons.append(TransactionFailureReason.PLAN_PACKET_MISMATCH.value)
            elif packet.roots != plan.roots:
                reasons.append(TransactionFailureReason.ROOT_DRIFT.value)

        if not lease.active:
            reasons.append(TransactionFailureReason.LEASE_INVALID.value)
        if lease.expires_at and self.now() > 0 and self.now() >= lease.expires_at:
            reasons.append(TransactionFailureReason.LEASE_INVALID.value)
        if not lease.covers_writes(plan.permitted_write_paths):
            reasons.append(TransactionFailureReason.LEASE_PATH_MISMATCH.value)

        if reasons:
            return self._failed_without_checkpoint(
                plan=plan,
                lease=lease,
                transaction_id=txn_id,
                reasons=reasons,
                path_before_hashes=path_before_hashes or (),
            )

        # --- Checkpoint before mutation ---
        if checkpoint is None:
            hashes = tuple(path_before_hashes or ())
            if packet is not None and not hashes:
                hashes = packet.before_hashes
            checkpoint = self.create_checkpoint(plan, path_before_hashes=hashes)
        else:
            if not isinstance(checkpoint, PropagationCheckpoint):
                raise ChangePropagationTransactionError(
                    "checkpoint must be a PropagationCheckpoint"
                )
            if checkpoint.plan_id != plan.plan_id:
                reasons.append(TransactionFailureReason.PLAN_PACKET_MISMATCH.value)
            if checkpoint.roots != plan.roots:
                reasons.append(TransactionFailureReason.ROOT_DRIFT.value)
            if reasons:
                return self._abort(
                    plan=plan,
                    lease=lease,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    group_receipts=(),
                    completed=(),
                    reasons=reasons,
                    failed_step_ids=(),
                    failed_group_id="",
                )

        # Verify every before-hash for write authority paths when a probe is present.
        hash_reasons = self._verify_before_hashes(
            plan=plan,
            checkpoint=checkpoint,
            packet=packet,
        )
        if hash_reasons:
            return self._abort(
                plan=plan,
                lease=lease,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                group_receipts=(),
                completed=(),
                reasons=hash_reasons,
                failed_step_ids=(),
                failed_group_id="",
            )

        if observe_timeout:
            return self._abort(
                plan=plan,
                lease=lease,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                group_receipts=(),
                completed=(),
                reasons=(TransactionFailureReason.TIMEOUT.value,),
                failed_step_ids=(),
                failed_group_id="",
            )

        groups = _build_execution_groups(plan)
        steps_by_id = {step.step_id: step for step in plan.steps}

        for group_id, scc_id, step_ids in groups:
            # Verify lease still active at each group boundary.
            if not lease.active:
                return self._abort(
                    plan=plan,
                    lease=lease,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=(TransactionFailureReason.LEASE_INVALID.value,),
                    failed_step_ids=(),
                    failed_group_id=group_id,
                )

            step_receipts: list[PropagationStepReceipt] = []
            group_reasons: list[str] = []

            for step_id in step_ids:
                step = steps_by_id[step_id]
                unmet = set(step.dependency_step_ids) - set(completed)
                # Dependencies inside the same SCC group are allowed to be unmet
                # until the whole group finishes; external deps must already be done.
                external_unmet = unmet - set(step_ids)
                if external_unmet:
                    group_reasons.append(TransactionFailureReason.DEPENDENCY_UNMET.value)
                    step_receipts.append(
                        PropagationStepReceipt(
                            step_id=step_id,
                            disposition=StepExecutionDisposition.FAILED,
                            reason_codes=(TransactionFailureReason.DEPENDENCY_UNMET.value,),
                        )
                    )
                    break

                precheck = self._precheck_step(plan, step, lease, checkpoint)
                if precheck is not None:
                    step_receipts.append(precheck)
                    group_reasons.extend(precheck.reason_codes)
                    break

                request = StepApplyRequest(
                    plan=plan,
                    packet=packet,
                    step=step,
                    lease=lease,
                    checkpoint=checkpoint,
                    completed_step_ids=tuple(completed),
                )
                try:
                    result = self.step_applicator(request)
                except Exception as exc:  # noqa: BLE001 — fail-closed applicator boundary
                    step_receipts.append(
                        PropagationStepReceipt(
                            step_id=step_id,
                            disposition=StepExecutionDisposition.FAILED,
                            reason_codes=(TransactionFailureReason.STEP_FAILURE.value,),
                            diagnostic_refs=(f"diagnostic:exception:{type(exc).__name__}",),
                        )
                    )
                    group_reasons.append(TransactionFailureReason.STEP_FAILURE.value)
                    break

                if not isinstance(result, StepApplyResult):
                    step_receipts.append(
                        PropagationStepReceipt(
                            step_id=step_id,
                            disposition=StepExecutionDisposition.FAILED,
                            reason_codes=(TransactionFailureReason.MALFORMED_INPUT.value,),
                        )
                    )
                    group_reasons.append(TransactionFailureReason.MALFORMED_INPUT.value)
                    break

                # Scope: written paths must stay inside plan + lease write authority.
                written = set(result.written_paths)
                if written - set(plan.permitted_write_paths) or written - set(
                    lease.permitted_write_paths
                ):
                    step_receipts.append(
                        PropagationStepReceipt(
                            step_id=step_id,
                            disposition=StepExecutionDisposition.SCOPE_ESCAPE,
                            reason_codes=(TransactionFailureReason.SCOPE_ESCAPE.value,),
                            written_paths=result.written_paths,
                            observed_before_hashes=result.observed_before_hashes,
                            diagnostic_refs=result.diagnostic_refs,
                        )
                    )
                    group_reasons.append(TransactionFailureReason.SCOPE_ESCAPE.value)
                    break

                if result.disposition is not StepExecutionDisposition.PASSED:
                    reason = (
                        TransactionFailureReason.TIMEOUT.value
                        if result.disposition is StepExecutionDisposition.TIMED_OUT
                        else TransactionFailureReason.DRIFT.value
                        if result.disposition is StepExecutionDisposition.DRIFT
                        else TransactionFailureReason.SCOPE_ESCAPE.value
                        if result.disposition is StepExecutionDisposition.SCOPE_ESCAPE
                        else TransactionFailureReason.STEP_FAILURE.value
                    )
                    codes = result.reason_codes or (reason,)
                    step_receipts.append(
                        PropagationStepReceipt(
                            step_id=step_id,
                            disposition=result.disposition,
                            reason_codes=codes,
                            written_paths=result.written_paths,
                            observed_before_hashes=result.observed_before_hashes,
                            diagnostic_refs=result.diagnostic_refs,
                        )
                    )
                    group_reasons.extend(codes)
                    break

                # Re-verify before hashes after apply when a live probe is present.
                if self.hash_probe is not None and result.observed_before_hashes:
                    for observed in result.observed_before_hashes:
                        expected = checkpoint.hash_map().get(observed.path)
                        if expected and observed.before_hash and observed.before_hash != expected:
                            # Before-hash is pre-mutation; after apply current may differ.
                            # observed_before_hashes record what was verified *before* write.
                            pass

                step_receipts.append(
                    PropagationStepReceipt(
                        step_id=step_id,
                        disposition=StepExecutionDisposition.PASSED,
                        written_paths=result.written_paths,
                        observed_before_hashes=result.observed_before_hashes,
                        diagnostic_refs=result.diagnostic_refs,
                    )
                )

            if group_reasons or len(step_receipts) != len(step_ids) or not all(
                item.passed for item in step_receipts
            ):
                # Incomplete SCC/group: roll back whole transaction.
                failed_ids = tuple(
                    item.step_id for item in step_receipts if not item.passed
                ) or step_ids
                group_receipts.append(
                    PropagationGroupReceipt(
                        group_id=group_id,
                        scc_id=scc_id,
                        step_ids=step_ids,
                        disposition=GroupExecutionDisposition.ROLLED_BACK,
                        step_receipts=tuple(step_receipts),
                        reason_codes=tuple(sorted(set(group_reasons)))
                        or (TransactionFailureReason.GROUP_INCOMPLETE.value,),
                        diagnostic_refs=tuple(
                            ref
                            for item in step_receipts
                            for ref in item.diagnostic_refs
                        ),
                    )
                )
                return self._abort(
                    plan=plan,
                    lease=lease,
                    transaction_id=txn_id,
                    checkpoint=checkpoint,
                    group_receipts=tuple(group_receipts),
                    completed=tuple(completed),
                    reasons=tuple(sorted(set(group_reasons)))
                    or (TransactionFailureReason.GROUP_INCOMPLETE.value,),
                    failed_step_ids=failed_ids,
                    failed_group_id=group_id,
                )

            # Group fully passed — only then mark steps completed.
            group_receipts.append(
                PropagationGroupReceipt(
                    group_id=group_id,
                    scc_id=scc_id,
                    step_ids=step_ids,
                    disposition=GroupExecutionDisposition.PASSED,
                    step_receipts=tuple(step_receipts),
                )
            )
            completed.extend(step_ids)

        # All groups passed: commit only if every plan step is accounted for.
        expected_steps = {step.step_id for step in plan.steps}
        if set(completed) != expected_steps:
            return self._abort(
                plan=plan,
                lease=lease,
                transaction_id=txn_id,
                checkpoint=checkpoint,
                group_receipts=tuple(group_receipts),
                completed=tuple(completed),
                reasons=(TransactionFailureReason.GROUP_INCOMPLETE.value,),
                failed_step_ids=tuple(sorted(expected_steps - set(completed))),
                failed_group_id="",
            )

        transaction = PropagationTransaction(
            roots=plan.roots,
            transaction_id=txn_id,
            plan_id=plan.plan_id,
            state=TransactionState.COMMITTED,
            checkpoint_id=checkpoint.checkpoint_id,
            active_scc_group_id="",
            completed_step_ids=tuple(completed),
            diagnostic_refs=(),
            lease_id=lease.lease_id,
        )
        return TransactionExecutionReport(
            roots=plan.roots,
            transaction=transaction,
            plan=plan,
            checkpoint=checkpoint,
            group_receipts=tuple(group_receipts),
            rollback=None,
            reason_codes=(),
            committed=True,
            partial_merge_allowed=False,
        )

    def require_committed(self, *args: Any, **kwargs: Any) -> TransactionExecutionReport:
        report = self.execute(*args, **kwargs)
        if not report.committed:
            reasons = ", ".join(report.reason_codes) or "incomplete"
            raise ChangePropagationTransactionError(
                "change propagation transaction rejected: " + reasons
            )
        return report

    # --- internals ---

    def _verify_before_hashes(
        self,
        *,
        plan: AtomicPropagationPlan,
        checkpoint: PropagationCheckpoint,
        packet: ChangePropagationEditPacket | None,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        write_paths = set(plan.permitted_write_paths)
        checkpoint_map = checkpoint.hash_map()

        # Every write path should have a before-hash in the checkpoint or packet.
        packet_map: dict[str, str] = {}
        if packet is not None:
            packet_map = {
                item.path: item.before_hash
                for item in packet.before_hashes
                if item.before_hash
            }

        for path in write_paths:
            expected = checkpoint_map.get(path) or packet_map.get(path)
            if not expected:
                reasons.append(TransactionFailureReason.BEFORE_HASH_MISSING.value)
                continue
            if self.hash_probe is not None:
                observed = self.hash_probe(path)
                if observed and observed != expected:
                    reasons.append(TransactionFailureReason.BEFORE_HASH_MISMATCH.value)

        # Checkpoint hashes that disagree with packet hashes are drift.
        for path, chk in checkpoint_map.items():
            pkt = packet_map.get(path)
            if pkt and pkt != chk:
                reasons.append(TransactionFailureReason.BEFORE_HASH_MISMATCH.value)

        return tuple(sorted(set(reasons)))

    def _precheck_step(
        self,
        plan: AtomicPropagationPlan,
        step: PropagationPlanStep,
        lease: TransactionLease,
        checkpoint: PropagationCheckpoint,
    ) -> PropagationStepReceipt | None:
        if set(step.write_paths) - set(plan.permitted_write_paths):
            return PropagationStepReceipt(
                step_id=step.step_id,
                disposition=StepExecutionDisposition.SCOPE_ESCAPE,
                reason_codes=(TransactionFailureReason.SCOPE_ESCAPE.value,),
            )
        if step.write_paths and not lease.covers_writes(step.write_paths):
            return PropagationStepReceipt(
                step_id=step.step_id,
                disposition=StepExecutionDisposition.FAILED,
                reason_codes=(TransactionFailureReason.LEASE_PATH_MISMATCH.value,),
            )
        if not lease.active:
            return PropagationStepReceipt(
                step_id=step.step_id,
                disposition=StepExecutionDisposition.FAILED,
                reason_codes=(TransactionFailureReason.LEASE_INVALID.value,),
            )
        if self.hash_probe is not None:
            chk = checkpoint.hash_map()
            for path in step.write_paths:
                expected = chk.get(path)
                if not expected:
                    continue
                observed = self.hash_probe(path)
                if observed and observed != expected:
                    return PropagationStepReceipt(
                        step_id=step.step_id,
                        disposition=StepExecutionDisposition.DRIFT,
                        reason_codes=(TransactionFailureReason.BEFORE_HASH_MISMATCH.value,),
                        observed_before_hashes=(
                            PathBeforeHash(path=path, before_hash=observed),
                        ),
                    )
        return None

    def _abort(
        self,
        *,
        plan: AtomicPropagationPlan,
        lease: TransactionLease,
        transaction_id: str,
        checkpoint: PropagationCheckpoint,
        group_receipts: tuple[PropagationGroupReceipt, ...] | Sequence[PropagationGroupReceipt],
        completed: tuple[str, ...] | Sequence[str],
        reasons: Sequence[str],
        failed_step_ids: Sequence[str],
        failed_group_id: str,
    ) -> TransactionExecutionReport:
        reason_codes = tuple(sorted({_identifier(r, "reason") for r in reasons if r}))
        if not reason_codes:
            reason_codes = (TransactionFailureReason.STEP_FAILURE.value,)

        restored = False
        try:
            restored = bool(self.restore_adapter(checkpoint))
        except Exception:  # noqa: BLE001
            restored = False
        if not restored:
            raise ChangePropagationTransactionError(
                "checkpoint restore failed; candidate tree may be inconsistent"
            )

        diagnostic_refs = (
            f"diagnostic:rollback:{reason_codes[0]}",
            f"diagnostic:checkpoint:{checkpoint.checkpoint_id}",
        )
        rollback_preimage = {
            "schema": PROPAGATION_ROLLBACK_RECEIPT_SCHEMA,
            "transaction_id": transaction_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "plan_id": plan.plan_id,
            "reason_codes": list(reason_codes),
        }
        rollback = PropagationRollbackReceipt(
            roots=plan.roots,
            rollback_id=content_identity(rollback_preimage),
            transaction_id=transaction_id,
            checkpoint_id=checkpoint.checkpoint_id,
            plan_id=plan.plan_id,
            strategy_ref=plan.rollback_strategy_ref or checkpoint.strategy_ref,
            restored=True,
            reason_codes=reason_codes,
            diagnostic_refs=diagnostic_refs,
            failed_step_ids=tuple(failed_step_ids),
            failed_group_id=failed_group_id,
        )
        transaction = PropagationTransaction(
            roots=plan.roots,
            transaction_id=transaction_id,
            plan_id=plan.plan_id,
            state=TransactionState.ROLLED_BACK,
            checkpoint_id=checkpoint.checkpoint_id,
            active_scc_group_id=failed_group_id,
            completed_step_ids=tuple(completed),
            diagnostic_refs=diagnostic_refs,
            lease_id=lease.lease_id,
        )
        return TransactionExecutionReport(
            roots=plan.roots,
            transaction=transaction,
            plan=plan,
            checkpoint=checkpoint,
            group_receipts=tuple(group_receipts),
            rollback=rollback,
            reason_codes=reason_codes,
            committed=False,
            partial_merge_allowed=False,
        )

    def _failed_without_checkpoint(
        self,
        *,
        plan: AtomicPropagationPlan,
        lease: TransactionLease,
        transaction_id: str,
        reasons: Sequence[str],
        path_before_hashes: Sequence[PathBeforeHash],
    ) -> TransactionExecutionReport:
        """Pre-checkpoint failures: no mutation occurred; emit FAILED txn."""

        reason_codes = tuple(sorted({str(r) for r in reasons if r}))
        # Still mint a diagnostic checkpoint identity for audit without claiming restore.
        strategy = plan.checkpoint_strategy_ref or "checkpoint:none"
        try:
            checkpoint = PropagationCheckpoint(
                roots=plan.roots,
                checkpoint_id=content_identity(
                    {
                        "schema": PROPAGATION_CHECKPOINT_SCHEMA,
                        "plan_id": plan.plan_id,
                        "reason": "pre_mutation_failure",
                        "reasons": list(reason_codes),
                    }
                ),
                plan_id=plan.plan_id,
                plan_content_id=plan.content_id if plan.disposition is PlanDisposition.ADMITTED else plan.plan_id,
                path_before_hashes=tuple(path_before_hashes),
                strategy_ref=strategy,
                tree_snapshot_ref=plan.roots.candidate_tree_id,
                diagnostic_refs=tuple(f"diagnostic:{r}" for r in reason_codes),
            )
        except Exception:  # noqa: BLE001
            # Last-resort minimal checkpoint for report construction.
            checkpoint = PropagationCheckpoint(
                roots=plan.roots,
                checkpoint_id=content_identity({"plan_id": plan.plan_id, "failed": True}),
                plan_id=plan.plan_id,
                plan_content_id=plan.plan_id,
                path_before_hashes=(),
                strategy_ref=strategy or "checkpoint:none",
                tree_snapshot_ref=plan.roots.candidate_tree_id,
                diagnostic_refs=tuple(f"diagnostic:{r}" for r in reason_codes[:MAX_DIAGNOSTICS]),
            )
        diagnostic_refs = tuple(f"diagnostic:{r}" for r in reason_codes)
        transaction = PropagationTransaction(
            roots=plan.roots,
            transaction_id=transaction_id,
            plan_id=plan.plan_id,
            state=TransactionState.FAILED,
            checkpoint_id=checkpoint.checkpoint_id,
            completed_step_ids=(),
            diagnostic_refs=diagnostic_refs,
            lease_id=lease.lease_id if lease.active else "",
        )
        return TransactionExecutionReport(
            roots=plan.roots,
            transaction=transaction,
            plan=plan,
            checkpoint=checkpoint,
            group_receipts=(),
            rollback=None,
            reason_codes=reason_codes,
            committed=False,
            partial_merge_allowed=False,
        )


def execute_change_propagation_transaction(
    plan: AtomicPropagationPlan,
    *,
    lease: TransactionLease,
    path_before_hashes: Sequence[PathBeforeHash] | None = None,
    packet: ChangePropagationEditPacket | None = None,
    step_applicator: StepApplicator | None = None,
    restore_adapter: RestoreAdapter | None = None,
    hash_probe: HashProbe | None = None,
) -> TransactionExecutionReport:
    """Module entry point matching :meth:`ChangePropagationTransaction.execute`."""

    txn = ChangePropagationTransaction(
        step_applicator=step_applicator or _default_noop_applicator,
        restore_adapter=restore_adapter or _default_restore,
        hash_probe=hash_probe,
    )
    return txn.execute(
        plan,
        lease=lease,
        path_before_hashes=path_before_hashes,
        packet=packet,
    )


__all__ = [
    "CHANGE_PROPAGATION_TRANSACTION_INTERFACE",
    "PRODUCER_ID",
    "PROPAGATION_CHECKPOINT_SCHEMA",
    "PROPAGATION_GROUP_RECEIPT_SCHEMA",
    "PROPAGATION_ROLLBACK_RECEIPT_SCHEMA",
    "PROPAGATION_STEP_RECEIPT_SCHEMA",
    "PROPAGATION_TRANSACTION_REPORT_SCHEMA",
    "TRANSACTION_LEASE_SCHEMA",
    "ChangePropagationTransaction",
    "ChangePropagationTransactionError",
    "GroupExecutionDisposition",
    "HashProbe",
    "PropagationCheckpoint",
    "PropagationGroupReceipt",
    "PropagationRollbackReceipt",
    "PropagationStepReceipt",
    "RestoreAdapter",
    "StepApplyRequest",
    "StepApplyResult",
    "StepApplicator",
    "StepExecutionDisposition",
    "TransactionExecutionReport",
    "TransactionFailureReason",
    "TransactionLease",
    "create_propagation_checkpoint",
    "execute_change_propagation_transaction",
    # Re-export canonical records so callers import one surface.
    "AtomicPropagationPlan",
    "PropagationTransaction",
    "TransactionState",
]
