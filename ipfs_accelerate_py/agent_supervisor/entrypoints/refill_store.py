"""Durable refill saga state and append/adoption CAS (ASE3-021).

Persists a monotonic cursor through::

    EVALUATING -> APPEND_RESERVED -> APPENDED -> PLAN_INVALIDATED
    -> RECOMPILED -> DISPATCHED | ADOPTED

and terminal ``EXHAUSTED``. The store is intentionally separate from provider
attempt CAS and planning-effect CAS. Refill remains dormant until ASE3-026
consumes a validated pre-effect activation authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

REFILL_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/durable-refill-state@1"
)
REFILL_CURSOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/durable-refill-saga-cursor@1"
)
REFILL_APPEND_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/refill-append-receipt@1"
)
REFILL_ADOPTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/refill-adoption-receipt@1"
)
SIGNED_REFILL_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/signed-refill-policy@1"
)
PROMPT_REPLAY_REQUIRED: Final = "PROMPT_REPLAY_REQUIRED"


class RefillSagaPhase(str, Enum):
    EVALUATING = "EVALUATING"
    APPEND_RESERVED = "APPEND_RESERVED"
    APPENDED = "APPENDED"
    PLAN_INVALIDATED = "PLAN_INVALIDATED"
    RECOMPILED = "RECOMPILED"
    DISPATCHED = "DISPATCHED"
    ADOPTED = "ADOPTED"
    EXHAUSTED = "EXHAUSTED"


_PHASE_ORDER: tuple[RefillSagaPhase, ...] = (
    RefillSagaPhase.EVALUATING,
    RefillSagaPhase.APPEND_RESERVED,
    RefillSagaPhase.APPENDED,
    RefillSagaPhase.PLAN_INVALIDATED,
    RefillSagaPhase.RECOMPILED,
    RefillSagaPhase.DISPATCHED,
)

_TERMINAL: frozenset[RefillSagaPhase] = frozenset(
    {
        RefillSagaPhase.DISPATCHED,
        RefillSagaPhase.ADOPTED,
        RefillSagaPhase.EXHAUSTED,
    }
)


class RefillStoreError(ValueError):
    """Raised for durable refill store invariant violations."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class SignedRefillPolicy:
    """Bounded signed refill policy; never enlarged by generated work."""

    schema: str
    policy_cid: str
    max_epochs: int
    max_new_work_per_epoch: int
    max_unchanged_epochs: int
    activation_authorized: bool = False
    signer_identity_did: str = ""

    def __post_init__(self) -> None:
        if self.schema != SIGNED_REFILL_POLICY_SCHEMA:
            raise RefillStoreError("unsupported signed refill policy schema")
        if not self.policy_cid:
            raise RefillStoreError("policy_cid is required")
        for name in ("max_epochs", "max_new_work_per_epoch", "max_unchanged_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RefillStoreError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_cid": self.policy_cid,
            "max_epochs": self.max_epochs,
            "max_new_work_per_epoch": self.max_new_work_per_epoch,
            "max_unchanged_epochs": self.max_unchanged_epochs,
            "activation_authorized": self.activation_authorized,
            "signer_identity_did": self.signer_identity_did,
        }


@dataclass(frozen=True)
class RefillPhaseDeadline:
    """Monitor deadline distinguishing a live phase from a stall."""

    phase: str
    deadline_ms: int
    published_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "deadline_ms": self.deadline_ms,
            "published_at_ms": self.published_at_ms,
        }

    def expired(self, now_ms: int) -> bool:
        return int(now_ms) >= int(self.deadline_ms)


@dataclass
class DurableRefillSagaCursor:
    """Monotonic durable saga cursor for one logical refill attempt."""

    schema: str
    logical_attempt_id: str
    plan_root_cid: str
    tree_id: str
    epoch: int
    phase: str
    predecessor_cid: str
    phase_cid: str
    reservation_id: str = ""
    append_receipt_cid: str = ""
    plan_invalidation_cid: str = ""
    recompile_cid: str = ""
    dispatch_cid: str = ""
    gap_identities: tuple[str, ...] = ()
    deadline: RefillPhaseDeadline | None = None
    fence_token: str = ""
    updated_at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "plan_root_cid": self.plan_root_cid,
            "tree_id": self.tree_id,
            "epoch": self.epoch,
            "phase": self.phase,
            "predecessor_cid": self.predecessor_cid,
            "phase_cid": self.phase_cid,
            "reservation_id": self.reservation_id,
            "append_receipt_cid": self.append_receipt_cid,
            "plan_invalidation_cid": self.plan_invalidation_cid,
            "recompile_cid": self.recompile_cid,
            "dispatch_cid": self.dispatch_cid,
            "gap_identities": list(self.gap_identities),
            "deadline": None if self.deadline is None else self.deadline.to_dict(),
            "fence_token": self.fence_token,
            "updated_at_ms": self.updated_at_ms,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DurableRefillSagaCursor":
        deadline_raw = value.get("deadline")
        deadline = None
        if isinstance(deadline_raw, Mapping):
            deadline = RefillPhaseDeadline(
                phase=str(deadline_raw.get("phase") or ""),
                deadline_ms=int(deadline_raw.get("deadline_ms") or 0),
                published_at_ms=int(deadline_raw.get("published_at_ms") or 0),
            )
        return cls(
            schema=str(value.get("schema") or REFILL_CURSOR_SCHEMA),
            logical_attempt_id=str(value.get("logical_attempt_id") or ""),
            plan_root_cid=str(value.get("plan_root_cid") or ""),
            tree_id=str(value.get("tree_id") or ""),
            epoch=int(value.get("epoch") or 0),
            phase=str(value.get("phase") or ""),
            predecessor_cid=str(value.get("predecessor_cid") or ""),
            phase_cid=str(value.get("phase_cid") or ""),
            reservation_id=str(value.get("reservation_id") or ""),
            append_receipt_cid=str(value.get("append_receipt_cid") or ""),
            plan_invalidation_cid=str(value.get("plan_invalidation_cid") or ""),
            recompile_cid=str(value.get("recompile_cid") or ""),
            dispatch_cid=str(value.get("dispatch_cid") or ""),
            gap_identities=tuple(value.get("gap_identities") or ()),
            deadline=deadline,
            fence_token=str(value.get("fence_token") or ""),
            updated_at_ms=int(value.get("updated_at_ms") or 0),
        )


@dataclass
class DurableRefillState:
    """Durable refill budget/circuit state across processes and restarts."""

    schema: str
    plan_root_cid: str
    tree_id: str
    epoch: int = 0
    unchanged_epochs: int = 0
    last_gap_set: tuple[str, ...] = ()
    seen_gap_ids: tuple[str, ...] = ()
    cooldown_until_epoch: int = 0
    activation_authorized: bool = False
    active_cursor: DurableRefillSagaCursor | None = None
    history: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_root_cid": self.plan_root_cid,
            "tree_id": self.tree_id,
            "epoch": self.epoch,
            "unchanged_epochs": self.unchanged_epochs,
            "last_gap_set": list(self.last_gap_set),
            "seen_gap_ids": list(self.seen_gap_ids),
            "cooldown_until_epoch": self.cooldown_until_epoch,
            "activation_authorized": self.activation_authorized,
            "active_cursor": (
                None if self.active_cursor is None else self.active_cursor.to_dict()
            ),
            "history": [dict(item) for item in self.history],
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DurableRefillState":
        cursor_raw = value.get("active_cursor")
        cursor = (
            DurableRefillSagaCursor.from_dict(cursor_raw)
            if isinstance(cursor_raw, Mapping)
            else None
        )
        return cls(
            schema=str(value.get("schema") or REFILL_STORE_SCHEMA),
            plan_root_cid=str(value.get("plan_root_cid") or ""),
            tree_id=str(value.get("tree_id") or ""),
            epoch=int(value.get("epoch") or 0),
            unchanged_epochs=int(value.get("unchanged_epochs") or 0),
            last_gap_set=tuple(value.get("last_gap_set") or ()),
            seen_gap_ids=tuple(value.get("seen_gap_ids") or ()),
            cooldown_until_epoch=int(value.get("cooldown_until_epoch") or 0),
            activation_authorized=bool(value.get("activation_authorized")),
            active_cursor=cursor,
            history=tuple(dict(item) for item in (value.get("history") or ())),
        )


@dataclass(frozen=True)
class RefillAppendReceipt:
    schema: str
    logical_attempt_id: str
    plan_root_cid: str
    tree_id: str
    epoch: int
    gap_identities: tuple[str, ...]
    expected_revision: int
    append_cid: str
    created_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "plan_root_cid": self.plan_root_cid,
            "tree_id": self.tree_id,
            "epoch": self.epoch,
            "gap_identities": list(self.gap_identities),
            "expected_revision": self.expected_revision,
            "append_cid": self.append_cid,
            "created_at_ms": self.created_at_ms,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True)
class RefillAdoptionReceipt:
    schema: str
    logical_attempt_id: str
    phase: str
    plan_root_cid: str
    tree_id: str
    epoch: int
    winner: bool
    adopted_at_ms: int
    append_receipt_cid: str = ""
    dispatch_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "phase": self.phase,
            "plan_root_cid": self.plan_root_cid,
            "tree_id": self.tree_id,
            "epoch": self.epoch,
            "winner": self.winner,
            "adopted_at_ms": self.adopted_at_ms,
            "append_receipt_cid": self.append_receipt_cid,
            "dispatch_cid": self.dispatch_cid,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True)
class PlanInvalidationReceipt:
    schema: str
    logical_attempt_id: str
    plan_root_cid: str
    previous_revision: int
    invalidated_at_ms: int
    reason_code: str = "refill_append"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "plan_root_cid": self.plan_root_cid,
            "previous_revision": self.previous_revision,
            "invalidated_at_ms": self.invalidated_at_ms,
            "reason_code": self.reason_code,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())


def _phase_index(phase: str) -> int:
    try:
        return _PHASE_ORDER.index(RefillSagaPhase(phase))
    except ValueError as exc:
        if phase in {item.value for item in _TERMINAL}:
            return len(_PHASE_ORDER)
        raise RefillStoreError(f"unknown refill phase {phase!r}") from exc


class RefillStore:
    """File-backed multiproc durable refill state + saga cursor CAS."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise RefillStoreError("refill store root must not be a symlink")
        self._lock = threading.RLock()

    def _state_path(self, plan_root_cid: str) -> Path:
        digest = hashlib.sha256(plan_root_cid.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.state.json"

    def _cursor_path(self, logical_attempt_id: str) -> Path:
        digest = hashlib.sha256(logical_attempt_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.cursor.json"

    def _write_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.is_symlink():
            raise RefillStoreError(f"refill path is a symlink: {path}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        body = _canonical(payload) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if path.is_symlink():
            raise RefillStoreError(f"refill path is a symlink: {path}")
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RefillStoreError("refill record is torn") from exc
        if not isinstance(payload, dict):
            raise RefillStoreError("refill record is invalid")
        return payload

    def load_state(self, plan_root_cid: str) -> DurableRefillState | None:
        with self._lock:
            raw = self._read_json(self._state_path(plan_root_cid))
            return None if raw is None else DurableRefillState.from_dict(raw)

    def save_state(self, state: DurableRefillState) -> DurableRefillState:
        with self._lock:
            self._write_atomic(self._state_path(state.plan_root_cid), state.to_dict())
            return state

    def load_cursor(self, logical_attempt_id: str) -> DurableRefillSagaCursor | None:
        with self._lock:
            raw = self._read_json(self._cursor_path(logical_attempt_id))
            return None if raw is None else DurableRefillSagaCursor.from_dict(raw)

    def begin_or_adopt(
        self,
        *,
        logical_attempt_id: str,
        plan_root_cid: str,
        tree_id: str,
        epoch: int,
        gap_identities: Sequence[str] = (),
        phase_budget_ms: int = 60_000,
        now_ms: int | None = None,
        activation_authorized: bool = False,
    ) -> tuple[DurableRefillSagaCursor, bool, RefillAdoptionReceipt | None]:
        """Reserve EVALUATING or adopt an existing cursor winner."""

        if not activation_authorized:
            raise RefillStoreError(
                "refill saga is dormant until ASE3-026 activation authorization"
            )
        with self._lock:
            existing = self.load_cursor(logical_attempt_id)
            clock = int(now_ms if now_ms is not None else _now_ms())
            if existing is not None:
                adoption = None
                if RefillSagaPhase(existing.phase) in _TERMINAL or existing.phase in {
                    p.value for p in _TERMINAL
                }:
                    adoption = RefillAdoptionReceipt(
                        schema=REFILL_ADOPTION_RECEIPT_SCHEMA,
                        logical_attempt_id=existing.logical_attempt_id,
                        phase=existing.phase,
                        plan_root_cid=existing.plan_root_cid,
                        tree_id=existing.tree_id,
                        epoch=existing.epoch,
                        winner=False,
                        adopted_at_ms=clock,
                        append_receipt_cid=existing.append_receipt_cid,
                        dispatch_cid=existing.dispatch_cid,
                    )
                return existing, False, adoption

            fence = hashlib.sha256(
                f"{logical_attempt_id}:{epoch}:{clock}:{os.getpid()}".encode()
            ).hexdigest()
            cursor = DurableRefillSagaCursor(
                schema=REFILL_CURSOR_SCHEMA,
                logical_attempt_id=str(logical_attempt_id),
                plan_root_cid=str(plan_root_cid),
                tree_id=str(tree_id),
                epoch=int(epoch),
                phase=RefillSagaPhase.EVALUATING.value,
                predecessor_cid="",
                phase_cid="",
                gap_identities=tuple(str(item) for item in gap_identities),
                deadline=RefillPhaseDeadline(
                    phase=RefillSagaPhase.EVALUATING.value,
                    deadline_ms=clock + int(phase_budget_ms),
                    published_at_ms=clock,
                ),
                fence_token=fence,
                updated_at_ms=clock,
            )
            cursor.phase_cid = cursor.content_id
            self._write_atomic(self._cursor_path(logical_attempt_id), cursor.to_dict())
            return cursor, True, None

    def advance(
        self,
        logical_attempt_id: str,
        *,
        fence_token: str,
        next_phase: str,
        tree_id: str,
        phase_budget_ms: int = 60_000,
        now_ms: int | None = None,
        reservation_id: str = "",
        append_receipt_cid: str = "",
        plan_invalidation_cid: str = "",
        recompile_cid: str = "",
        dispatch_cid: str = "",
        gap_identities: Sequence[str] | None = None,
    ) -> DurableRefillSagaCursor:
        """Advance one exact phase with predecessor CAS binding."""

        with self._lock:
            cursor = self.load_cursor(logical_attempt_id)
            if cursor is None:
                raise RefillStoreError("refill cursor not found")
            if fence_token != cursor.fence_token:
                raise RefillStoreError("refill cursor fence token mismatch")
            if cursor.tree_id != tree_id:
                raise RefillStoreError("refill cursor tree_id drifted")
            clock = int(now_ms if now_ms is not None else _now_ms())
            if cursor.deadline is not None and cursor.deadline.expired(clock):
                raise RefillStoreError("refill phase deadline expired")

            current = RefillSagaPhase(cursor.phase)
            target = RefillSagaPhase(next_phase)
            if current in _TERMINAL:
                raise RefillStoreError(f"cursor already terminal at {current.value}")
            if target is RefillSagaPhase.ADOPTED:
                # ADOPTED may be entered from DISPATCHED only via adopt_terminal.
                raise RefillStoreError("use adopt_terminal for ADOPTED")
            if target is RefillSagaPhase.EXHAUSTED:
                expected_from = current
            else:
                expected_index = _phase_index(current.value) + 1
                if target not in _TERMINAL and _phase_index(target.value) != expected_index:
                    raise RefillStoreError(
                        f"cannot advance {current.value} -> {target.value}"
                    )

            predecessor = cursor.phase_cid
            cursor.predecessor_cid = predecessor
            cursor.phase = target.value
            if reservation_id:
                cursor.reservation_id = reservation_id
            if append_receipt_cid:
                cursor.append_receipt_cid = append_receipt_cid
            if plan_invalidation_cid:
                cursor.plan_invalidation_cid = plan_invalidation_cid
            if recompile_cid:
                cursor.recompile_cid = recompile_cid
            if dispatch_cid:
                cursor.dispatch_cid = dispatch_cid
            if gap_identities is not None:
                cursor.gap_identities = tuple(str(item) for item in gap_identities)
            cursor.updated_at_ms = clock
            cursor.deadline = RefillPhaseDeadline(
                phase=target.value,
                deadline_ms=clock + int(phase_budget_ms),
                published_at_ms=clock,
            )
            # phase_cid recomputed after mutation
            cursor.phase_cid = ""
            cursor.phase_cid = cursor.content_id
            self._write_atomic(self._cursor_path(logical_attempt_id), cursor.to_dict())
            return cursor

    def adopt_terminal(
        self,
        logical_attempt_id: str,
        *,
        now_ms: int | None = None,
    ) -> RefillAdoptionReceipt:
        with self._lock:
            cursor = self.load_cursor(logical_attempt_id)
            if cursor is None:
                raise RefillStoreError("refill cursor not found")
            clock = int(now_ms if now_ms is not None else _now_ms())
            phase = RefillSagaPhase(cursor.phase)
            if phase is RefillSagaPhase.ADOPTED:
                return RefillAdoptionReceipt(
                    schema=REFILL_ADOPTION_RECEIPT_SCHEMA,
                    logical_attempt_id=cursor.logical_attempt_id,
                    phase=cursor.phase,
                    plan_root_cid=cursor.plan_root_cid,
                    tree_id=cursor.tree_id,
                    epoch=cursor.epoch,
                    winner=False,
                    adopted_at_ms=clock,
                    append_receipt_cid=cursor.append_receipt_cid,
                    dispatch_cid=cursor.dispatch_cid,
                )
            if phase is not RefillSagaPhase.DISPATCHED:
                raise RefillStoreError(
                    f"cannot adopt terminal from phase {cursor.phase}"
                )
            cursor.predecessor_cid = cursor.phase_cid
            cursor.phase = RefillSagaPhase.ADOPTED.value
            cursor.updated_at_ms = clock
            cursor.phase_cid = ""
            cursor.phase_cid = cursor.content_id
            self._write_atomic(self._cursor_path(logical_attempt_id), cursor.to_dict())
            return RefillAdoptionReceipt(
                schema=REFILL_ADOPTION_RECEIPT_SCHEMA,
                logical_attempt_id=cursor.logical_attempt_id,
                phase=cursor.phase,
                plan_root_cid=cursor.plan_root_cid,
                tree_id=cursor.tree_id,
                epoch=cursor.epoch,
                winner=True,
                adopted_at_ms=clock,
                append_receipt_cid=cursor.append_receipt_cid,
                dispatch_cid=cursor.dispatch_cid,
            )


__all__ = [
    "DurableRefillSagaCursor",
    "DurableRefillState",
    "PlanInvalidationReceipt",
    "PROMPT_REPLAY_REQUIRED",
    "REFILL_ADOPTION_RECEIPT_SCHEMA",
    "REFILL_APPEND_RECEIPT_SCHEMA",
    "REFILL_CURSOR_SCHEMA",
    "REFILL_STORE_SCHEMA",
    "RefillAdoptionReceipt",
    "RefillAppendReceipt",
    "RefillPhaseDeadline",
    "RefillSagaPhase",
    "RefillStore",
    "RefillStoreError",
    "SIGNED_REFILL_POLICY_SCHEMA",
    "SignedRefillPolicy",
]
