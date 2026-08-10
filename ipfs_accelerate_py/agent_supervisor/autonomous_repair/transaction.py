"""DCR-072: isolated multi-root repair transactions with rollback.

Interfaces
----------
* ``MultiRootRepairTransaction@1`` — fenced multi-owner write transaction.
* ``RollbackJournal@1`` — ordered inverse journal for crash/cancel recovery.

Predicted symbols: :class:`MultiRootRepairTransaction`, :class:`RollbackJournal`,
:class:`FencedWrite`.

Normative rules (fail-closed)
-----------------------------
* Never write the user checkout; only isolated owner worktrees.
* Writes require a matching path lease and fencing token.
* Symlink escape, out-of-scope paths, stale fences, dirty-unbound worktrees,
  and lease races reject without promoted mutation.
* Failure, crash, cancellation, or partial write rolls back every journaled
  write and derived artifact; runtime model calls remain 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / schemas
# ---------------------------------------------------------------------------

MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE: Final[str] = "MultiRootRepairTransaction@1"
ROLLBACK_JOURNAL_INTERFACE: Final[str] = "RollbackJournal@1"
FENCED_WRITE_INTERFACE: Final[str] = "FencedWrite@1"
DCR_TRANSACTION_EVIDENCE: Final[str] = "dcr/transaction@1"
DCR_TRANSACTION_VERSION: Final[int] = 1

FENCED_WRITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-fenced-write@1"
)
ROLLBACK_JOURNAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-rollback-journal@1"
)
TRANSACTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-transaction-receipt@1"
)
TRANSACTION_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-transaction-receipts-catalog@1"
)
DEFAULT_TRANSACTION_RECEIPTS_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/transaction-receipts.json"
)

# Closed owner roots admitted by DCR multi-root repair.
ADMITTED_OWNER_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "external/ipfs_accelerate",
        "external/ipfs_datasets",
        "external/ipfs_kit",
        "Mcp-Plus-Plus",
        "swissknife",
    }
)

_GLOBAL_LEASE_REGISTRY_LOCK = threading.RLock()
# lease_id -> fencing_token currently held open across processes-of-record.
_OPEN_LEASE_FENCES: dict[str, str] = {}


class MultiRootTransactionError(ContractValidationError):
    """Malformed transaction input or closed-boundary violation."""


class TransactionDisposition(str, Enum):  # noqa: UP042 - Python 3.8
    """Closed lifecycle outcomes for one multi-root transaction."""

    OPEN = "open"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class TransactionRejectReason(str, Enum):  # noqa: UP042
    STALE_FENCE = "stale_fence"
    DIRTY_UNBOUND = "dirty_unbound"
    OUT_OF_SCOPE = "out_of_scope"
    SYMLINK_ESCAPE = "symlink_escape"
    LEASE_RACE = "lease_race"
    PARTIAL_WRITE = "partial_write"
    CRASH = "crash"
    CANCELLED = "cancelled"
    USER_CHECKOUT_FORBIDDEN = "user_checkout_forbidden"
    MISSING_LEASE = "missing_lease"
    UNKNOWN_OWNER = "unknown_owner"
    NOT_OPEN = "not_open"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _safe_rel(path: str, *, field: str = "path") -> str:
    text = str(path or "").strip().replace("\\", "/")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or ".." in pure.parts
        or (pure.parts and pure.parts[0].endswith(":"))
        or "\x00" in text
    ):
        raise MultiRootTransactionError(f"invalid repository path for {field}")
    return pure.as_posix()


def _owner_for_path(path: str) -> str | None:
    rel = _safe_rel(path)
    for owner in sorted(ADMITTED_OWNER_ROOTS, key=len, reverse=True):
        if rel == owner or rel.startswith(owner + "/"):
            return owner
    return None


def _file_digest(path: Path) -> str:
    if not path.exists():
        return "sha256:" + ("0" * 64)
    if path.is_symlink():
        target = os.readlink(path)
        return _sha256_text(f"symlink:{target}")
    return _sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FencedWrite(CanonicalContract):
    """One journaled write under a path lease and fencing token."""

    SCHEMA: ClassVar[str] = FENCED_WRITE_SCHEMA
    INTERFACE: ClassVar[str] = FENCED_WRITE_INTERFACE

    path: str
    owner_root: str
    lease_id: str
    fencing_token: str
    before_hash: str
    after_hash: str
    bytes_written: int = 0
    worktree_id: str = ""
    node_id: str = ""
    created: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_rel(self.path, field="path"))
        object.__setattr__(
            self, "owner_root", _safe_rel(self.owner_root, field="owner_root")
        )
        if self.owner_root not in ADMITTED_OWNER_ROOTS:
            raise MultiRootTransactionError("unknown_owner")
        object.__setattr__(self, "lease_id", str(self.lease_id or "").strip())
        object.__setattr__(
            self, "fencing_token", str(self.fencing_token or "").strip()
        )
        if not self.lease_id or not self.fencing_token:
            raise MultiRootTransactionError("missing_lease")
        object.__setattr__(self, "before_hash", str(self.before_hash or "").strip())
        object.__setattr__(self, "after_hash", str(self.after_hash or "").strip())
        object.__setattr__(self, "bytes_written", int(self.bytes_written or 0))
        object.__setattr__(self, "worktree_id", str(self.worktree_id or ""))
        object.__setattr__(self, "node_id", str(self.node_id or ""))
        object.__setattr__(self, "created", bool(self.created))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "path": self.path,
            "owner_root": self.owner_root,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "bytes_written": self.bytes_written,
            "worktree_id": self.worktree_id,
            "node_id": self.node_id,
            "created": self.created,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["content_id"] = content_identity(payload)
        return payload

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())


@dataclass
class RollbackJournal:
    """Ordered inverse journal (``RollbackJournal@1``)."""

    INTERFACE: ClassVar[str] = ROLLBACK_JOURNAL_INTERFACE
    SCHEMA: ClassVar[str] = ROLLBACK_JOURNAL_SCHEMA

    transaction_id: str
    entries: list[FencedWrite] = field(default_factory=list)
    process_changes: list[dict[str, Any]] = field(default_factory=list)
    derived_artifacts: list[str] = field(default_factory=list)
    sealed: bool = False

    def append_write(self, write: FencedWrite) -> None:
        if self.sealed:
            raise MultiRootTransactionError("journal_sealed")
        self.entries.append(write)

    def append_process_change(self, change: Mapping[str, Any]) -> None:
        if self.sealed:
            raise MultiRootTransactionError("journal_sealed")
        self.process_changes.append(dict(change))

    def record_derived(self, path: str) -> None:
        if self.sealed:
            raise MultiRootTransactionError("journal_sealed")
        self.derived_artifacts.append(_safe_rel(path, field="derived_artifact"))

    def seal(self) -> None:
        self.sealed = True

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "transaction_id": self.transaction_id,
            "entries": [item.to_dict() for item in self.entries],
            "process_changes": list(self.process_changes),
            "derived_artifacts": list(self.derived_artifacts),
            "sealed": self.sealed,
            "entry_count": len(self.entries),
        }


@dataclass(frozen=True)
class PathLeaseBinding:
    """Lease + fence binding required before any write."""

    lease_id: str
    fencing_token: str
    owner_root: str
    permitted_write_paths: tuple[str, ...]
    fence_epoch: int = 1
    expected_fence_epoch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", str(self.lease_id or "").strip())
        object.__setattr__(
            self, "fencing_token", str(self.fencing_token or "").strip()
        )
        object.__setattr__(
            self, "owner_root", _safe_rel(self.owner_root, field="owner_root")
        )
        paths = tuple(
            _safe_rel(item, field="permitted_write_paths")
            for item in self.permitted_write_paths
        )
        object.__setattr__(self, "permitted_write_paths", paths)
        object.__setattr__(self, "fence_epoch", int(self.fence_epoch))
        if self.expected_fence_epoch is not None:
            object.__setattr__(
                self, "expected_fence_epoch", int(self.expected_fence_epoch)
            )

    def covers(self, path: str) -> bool:
        rel = _safe_rel(path)
        for permitted in self.permitted_write_paths:
            if rel == permitted or rel.startswith(permitted.rstrip("/") + "/"):
                return True
        return False


@dataclass(frozen=True)
class TransactionReceipt(CanonicalContract):
    """Body-free receipt for one multi-root transaction attempt."""

    SCHEMA: ClassVar[str] = TRANSACTION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE

    transaction_id: str
    disposition: TransactionDisposition
    reason_codes: tuple[str, ...] = ()
    journal: Mapping[str, Any] = field(default_factory=dict)
    worktree_ids: tuple[str, ...] = ()
    root_ids: tuple[str, ...] = ()
    commit_order: tuple[str, ...] = ()
    promoted_paths: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, TransactionDisposition):
            raise MultiRootTransactionError("invalid disposition")
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(
            self, "worktree_ids", tuple(str(item) for item in self.worktree_ids)
        )
        object.__setattr__(
            self, "root_ids", tuple(str(item) for item in self.root_ids)
        )
        object.__setattr__(
            self, "commit_order", tuple(str(item) for item in self.commit_order)
        )
        object.__setattr__(
            self,
            "promoted_paths",
            tuple(str(item) for item in self.promoted_paths),
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_write_authority", False)
        if self.disposition is TransactionDisposition.COMMITTED:
            # Commitment is evidence of staged owner worktrees only — never
            # authority to mutate the user checkout.
            pass

    @property
    def ok(self) -> bool:
        return self.disposition is TransactionDisposition.COMMITTED

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "transaction_id": self.transaction_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "journal": dict(self.journal),
            "worktree_ids": list(self.worktree_ids),
            "root_ids": list(self.root_ids),
            "commit_order": list(self.commit_order),
            "promoted_paths": list(self.promoted_paths),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
            "content_id": content_identity(
                {
                    "transaction_id": self.transaction_id,
                    "disposition": self.disposition.value,
                    "reason_codes": list(self.reason_codes),
                    "journal": dict(self.journal),
                    "worktree_ids": list(self.worktree_ids),
                    "root_ids": list(self.root_ids),
                    "commit_order": list(self.commit_order),
                    "promoted_paths": list(self.promoted_paths),
                }
            ),
        }


class MultiRootRepairTransaction:
    """Fenced multi-owner write transaction (``MultiRootRepairTransaction@1``).

    The transaction root is an isolated sandbox.  Owner worktrees live under
    ``<transaction_root>/owners/<owner>`` and never alias the user checkout.
    """

    INTERFACE: ClassVar[str] = MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE

    def __init__(
        self,
        *,
        transaction_root: str | Path,
        user_checkout: str | Path | None = None,
        leases: Sequence[PathLeaseBinding] = (),
        commit_order: Sequence[str] = (),
        require_clean_owners: bool = True,
        transaction_id: str | None = None,
    ) -> None:
        self._root = Path(transaction_root).resolve()
        self._user_checkout = (
            Path(user_checkout).resolve() if user_checkout is not None else None
        )
        if self._user_checkout is not None and self._root == self._user_checkout:
            raise MultiRootTransactionError(
                TransactionRejectReason.USER_CHECKOUT_FORBIDDEN.value
            )
        self._leases = {
            lease.lease_id: lease for lease in leases if lease.lease_id
        }
        order = tuple(
            _safe_rel(item, field="commit_order") for item in commit_order
        ) or tuple(sorted(ADMITTED_OWNER_ROOTS))
        self._commit_order = order
        self._require_clean = bool(require_clean_owners)
        self.transaction_id = transaction_id or content_identity(
            {
                "root": str(self._root),
                "leases": sorted(self._leases),
                "order": list(self._commit_order),
            }
        )
        self.journal = RollbackJournal(transaction_id=self.transaction_id)
        self._disposition = TransactionDisposition.OPEN
        self._reason_codes: list[str] = []
        self._owner_dirs: dict[str, Path] = {}
        self._held_leases: list[str] = []
        self._lock = threading.RLock()
        self._promoted: list[str] = []
        self._open = True

    @property
    def disposition(self) -> TransactionDisposition:
        return self._disposition

    @property
    def runtime_model_calls(self) -> int:
        return 0

    @property
    def worktree_ids(self) -> tuple[str, ...]:
        return tuple(
            f"worktree:{owner}:{path}" for owner, path in sorted(self._owner_dirs.items())
        )

    def bind_owner_worktree(
        self,
        owner_root: str,
        *,
        seed_files: Mapping[str, str | bytes] | None = None,
        dirty: bool = False,
    ) -> Path:
        """Create or return the isolated owner worktree directory."""

        with self._lock:
            self._require_open()
            owner = _safe_rel(owner_root, field="owner_root")
            if owner not in ADMITTED_OWNER_ROOTS:
                self._reject(TransactionRejectReason.UNKNOWN_OWNER, owner)
            if dirty and self._require_clean:
                self._reject(TransactionRejectReason.DIRTY_UNBOUND, owner)
            path = self._root / "owners" / owner
            if self._user_checkout is not None:
                try:
                    path.resolve().relative_to(self._user_checkout)
                except ValueError:
                    pass
                else:
                    # Owner dir nested inside user checkout is forbidden.
                    self._reject(
                        TransactionRejectReason.USER_CHECKOUT_FORBIDDEN, owner
                    )
            path.mkdir(parents=True, exist_ok=True)
            if seed_files:
                for rel, content in seed_files.items():
                    rel_path = _safe_rel(rel, field="seed_path")
                    # Seed paths are relative to owner root.
                    target = path / rel_path
                    if ".." in PurePosixPath(rel_path).parts:
                        self._reject(TransactionRejectReason.OUT_OF_SCOPE, rel_path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        target.write_bytes(content)
                    else:
                        target.write_text(str(content), encoding="utf-8")
            self._owner_dirs[owner] = path
            return path

    def acquire_lease(self, lease: PathLeaseBinding) -> None:
        """Register and globally fence one path lease for this transaction."""

        with self._lock:
            self._require_open()
            if (
                lease.expected_fence_epoch is not None
                and lease.fence_epoch != lease.expected_fence_epoch
            ):
                self._reject(
                    TransactionRejectReason.STALE_FENCE,
                    f"epoch {lease.fence_epoch}!={lease.expected_fence_epoch}",
                )
            with _GLOBAL_LEASE_REGISTRY_LOCK:
                held = _OPEN_LEASE_FENCES.get(lease.lease_id)
                if held is not None and held != lease.fencing_token:
                    self._reject(
                        TransactionRejectReason.LEASE_RACE,
                        f"lease {lease.lease_id} held by other fence",
                    )
                _OPEN_LEASE_FENCES[lease.lease_id] = lease.fencing_token
            self._leases[lease.lease_id] = lease
            self._held_leases.append(lease.lease_id)

    def write_file(
        self,
        *,
        path: str,
        content: str | bytes,
        lease_id: str,
        fencing_token: str,
        node_id: str = "",
        simulate_partial: bool = False,
        simulate_crash: bool = False,
    ) -> FencedWrite:
        """Apply one fenced write under an isolated owner worktree."""

        with self._lock:
            self._require_open()
            rel = _safe_rel(path)
            owner = _owner_for_path(rel)
            if owner is None:
                self._reject(TransactionRejectReason.OUT_OF_SCOPE, rel)
            lease = self._leases.get(lease_id)
            if lease is None:
                self._reject(TransactionRejectReason.MISSING_LEASE, lease_id)
            if lease.fencing_token != fencing_token:
                self._reject(TransactionRejectReason.STALE_FENCE, fencing_token)
            if lease.owner_root != owner:
                self._reject(TransactionRejectReason.OUT_OF_SCOPE, rel)
            if not lease.covers(rel):
                self._reject(TransactionRejectReason.OUT_OF_SCOPE, rel)
            if owner not in self._owner_dirs:
                self._reject(TransactionRejectReason.DIRTY_UNBOUND, owner)

            owner_dir = self._owner_dirs[owner]
            # Path relative to owner root.
            suffix = rel[len(owner) :].lstrip("/")
            target = (owner_dir / suffix).resolve()
            try:
                target.relative_to(owner_dir.resolve())
            except ValueError:
                self._reject(TransactionRejectReason.SYMLINK_ESCAPE, rel)

            # Symlink escape: refuse writing through a symlinked parent or path.
            for parent in [target, *target.parents]:
                if parent == owner_dir.resolve():
                    break
                if parent.is_symlink():
                    self._reject(TransactionRejectReason.SYMLINK_ESCAPE, rel)
                try:
                    parent.relative_to(owner_dir.resolve())
                except ValueError:
                    self._reject(TransactionRejectReason.SYMLINK_ESCAPE, rel)

            created = not target.exists()
            before = _file_digest(target) if target.exists() else _sha256_bytes(b"")
            # Snapshot prior bytes for rollback.
            prior_bytes: bytes | None
            if target.exists() and target.is_file() and not target.is_symlink():
                prior_bytes = target.read_bytes()
            else:
                prior_bytes = None

            if simulate_crash:
                self.journal.append_process_change(
                    {"kind": "crash_marker", "path": rel}
                )
                self._reject(TransactionRejectReason.CRASH, rel)

            target.parent.mkdir(parents=True, exist_ok=True)
            data = content if isinstance(content, bytes) else str(content).encode("utf-8")
            if simulate_partial:
                # Write only a prefix then fail — still journaled for rollback.
                target.write_bytes(data[: max(1, len(data) // 2)])
                self.journal.append_write(
                    FencedWrite(
                        path=rel,
                        owner_root=owner,
                        lease_id=lease_id,
                        fencing_token=fencing_token,
                        before_hash=before,
                        after_hash=_file_digest(target),
                        bytes_written=target.stat().st_size,
                        worktree_id=str(owner_dir),
                        node_id=node_id,
                        created=created,
                    )
                )
                # Store prior for rollback via derived side channel.
                self.journal.append_process_change(
                    {
                        "kind": "prior_bytes",
                        "path": rel,
                        "prior_b64": None
                        if prior_bytes is None
                        else prior_bytes.hex(),
                        "created": created,
                    }
                )
                self._reject(TransactionRejectReason.PARTIAL_WRITE, rel)

            target.write_bytes(data)
            after = _file_digest(target)
            write = FencedWrite(
                path=rel,
                owner_root=owner,
                lease_id=lease_id,
                fencing_token=fencing_token,
                before_hash=before,
                after_hash=after,
                bytes_written=len(data),
                worktree_id=str(owner_dir),
                node_id=node_id,
                created=created,
            )
            self.journal.append_write(write)
            self.journal.append_process_change(
                {
                    "kind": "prior_bytes",
                    "path": rel,
                    "prior_b64": None if prior_bytes is None else prior_bytes.hex(),
                    "created": created,
                }
            )
            return write

    def cancel(self) -> TransactionReceipt:
        """Cancel and roll back every journaled write."""

        with self._lock:
            if not self._open and self._disposition is not TransactionDisposition.OPEN:
                return self.receipt()
            self._reason_codes.append(TransactionRejectReason.CANCELLED.value)
            return self._rollback(TransactionDisposition.CANCELLED)

    def abort(self, reason: str = TransactionRejectReason.PARTIAL_WRITE.value) -> TransactionReceipt:
        with self._lock:
            self._reason_codes.append(str(reason))
            return self._rollback(TransactionDisposition.ROLLED_BACK)

    def commit(self) -> TransactionReceipt:
        """Seal the journal without promoting into the user checkout.

        Owner worktrees remain isolated evidence.  ``promoted_paths`` is empty
        by design — promotion is a later publication stage (DCR-074).
        """

        with self._lock:
            self._require_open()
            # Verify all journaled owners still match fences / digests.
            for write in self.journal.entries:
                owner_dir = self._owner_dirs.get(write.owner_root)
                if owner_dir is None:
                    return self._rollback(TransactionDisposition.ROLLED_BACK)
                suffix = write.path[len(write.owner_root) :].lstrip("/")
                target = owner_dir / suffix
                if _file_digest(target) != write.after_hash:
                    self._reason_codes.append("journal_digest_mismatch")
                    return self._rollback(TransactionDisposition.ROLLED_BACK)
            self.journal.seal()
            self._disposition = TransactionDisposition.COMMITTED
            self._open = False
            self._release_leases()
            # Commit order is recorded; no user-checkout mutation occurs.
            self._promoted = []
            return self.receipt()

    def receipt(self) -> TransactionReceipt:
        return TransactionReceipt(
            transaction_id=self.transaction_id,
            disposition=self._disposition,
            reason_codes=tuple(dict.fromkeys(self._reason_codes)),
            journal=self.journal.to_dict(),
            worktree_ids=self.worktree_ids,
            root_ids=tuple(sorted(self._owner_dirs)),
            commit_order=self._commit_order,
            promoted_paths=tuple(self._promoted),
        )

    def owner_path(self, owner_root: str) -> Path | None:
        return self._owner_dirs.get(_safe_rel(owner_root, field="owner_root"))

    def read_owner_file(self, path: str) -> bytes | None:
        rel = _safe_rel(path)
        owner = _owner_for_path(rel)
        if owner is None or owner not in self._owner_dirs:
            return None
        suffix = rel[len(owner) :].lstrip("/")
        target = self._owner_dirs[owner] / suffix
        if not target.is_file():
            return None
        return target.read_bytes()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._open or self._disposition is not TransactionDisposition.OPEN:
            raise MultiRootTransactionError(TransactionRejectReason.NOT_OPEN.value)

    def _reject(self, reason: TransactionRejectReason, detail: str = "") -> None:
        code = reason.value
        if detail:
            self._reason_codes.append(f"{code}:{detail}")
        else:
            self._reason_codes.append(code)
        self._rollback(TransactionDisposition.REJECTED)
        raise MultiRootTransactionError(code)

    def _rollback(self, disposition: TransactionDisposition) -> TransactionReceipt:
        # Inverse journal in reverse order.
        prior_map: dict[str, tuple[bool, bytes | None]] = {}
        for change in self.journal.process_changes:
            if change.get("kind") == "prior_bytes":
                path = str(change.get("path") or "")
                created = bool(change.get("created"))
                hex_bytes = change.get("prior_b64")
                prior = None if hex_bytes in (None, "") else bytes.fromhex(str(hex_bytes))
                prior_map[path] = (created, prior)

        for write in reversed(self.journal.entries):
            owner_dir = self._owner_dirs.get(write.owner_root)
            if owner_dir is None:
                continue
            suffix = write.path[len(write.owner_root) :].lstrip("/")
            target = owner_dir / suffix
            created, prior = prior_map.get(write.path, (write.created, None))
            try:
                if created:
                    if target.exists() or target.is_symlink():
                        if target.is_dir() and not target.is_symlink():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                else:
                    if prior is None:
                        if target.exists():
                            target.unlink()
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(prior)
            except OSError:
                self._reason_codes.append(f"rollback_error:{write.path}")

        for derived in list(self.journal.derived_artifacts):
            # Derived artifacts are relative to transaction root only.
            try:
                rel = _safe_rel(derived, field="derived")
            except MultiRootTransactionError:
                continue
            path = self._root / rel
            if path.exists():
                try:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except OSError:
                    self._reason_codes.append(f"rollback_derived_error:{rel}")

        self.journal.seal()
        self._disposition = disposition
        self._open = False
        self._promoted = []
        self._release_leases()
        return self.receipt()

    def _release_leases(self) -> None:
        with _GLOBAL_LEASE_REGISTRY_LOCK:
            for lease_id in self._held_leases:
                current = _OPEN_LEASE_FENCES.get(lease_id)
                lease = self._leases.get(lease_id)
                if lease is not None and current == lease.fencing_token:
                    _OPEN_LEASE_FENCES.pop(lease_id, None)
            self._held_leases.clear()


def materialize_transaction_receipts(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
    receipt: TransactionReceipt | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize transaction-receipts.json evidence for DCR-072."""

    if receipt is None:
        # Fixture: open→write→commit under an isolated sandbox.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="dcr072-tx-") as tmp:
            tx = MultiRootRepairTransaction(
                transaction_root=tmp,
                user_checkout=None,
                commit_order=("external/ipfs_accelerate",),
            )
            lease = PathLeaseBinding(
                lease_id="lease:dcr072-fixture",
                fencing_token="fence:dcr072-fixture",
                owner_root="external/ipfs_accelerate",
                permitted_write_paths=(
                    "external/ipfs_accelerate/ipfs_accelerate_py/fixture.py",
                ),
                fence_epoch=1,
                expected_fence_epoch=1,
            )
            tx.acquire_lease(lease)
            tx.bind_owner_worktree(
                "external/ipfs_accelerate",
                seed_files={
                    "ipfs_accelerate_py/fixture.py": "value = 0\n",
                },
            )
            tx.write_file(
                path="external/ipfs_accelerate/ipfs_accelerate_py/fixture.py",
                content="value = 1\n",
                lease_id=lease.lease_id,
                fencing_token=lease.fencing_token,
                node_id="node:fixture",
            )
            receipt_obj = tx.commit()
    elif isinstance(receipt, TransactionReceipt):
        receipt_obj = receipt
    else:
        # Mapping path: wrap as committed evidence-only projection.
        receipt_obj = TransactionReceipt(
            transaction_id=str(receipt.get("transaction_id") or "tx:mapping"),
            disposition=TransactionDisposition(
                str(receipt.get("disposition") or "committed")
            ),
            reason_codes=tuple(receipt.get("reason_codes") or ()),
            journal=dict(receipt.get("journal") or {}),
            worktree_ids=tuple(receipt.get("worktree_ids") or ()),
            root_ids=tuple(receipt.get("root_ids") or ()),
            commit_order=tuple(receipt.get("commit_order") or ()),
            promoted_paths=tuple(receipt.get("promoted_paths") or ()),
        )

    payload = {
        "schema": TRANSACTION_CATALOG_SCHEMA,
        "interface": MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE,
        "evidence_id": DCR_TRANSACTION_EVIDENCE,
        "version": DCR_TRANSACTION_VERSION,
        "receipt": receipt_obj.to_dict(),
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "promotes_user_checkout": False,
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else base.joinpath(*PurePosixPath(DEFAULT_TRANSACTION_RECEIPTS_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ADMITTED_OWNER_ROOTS",
    "DCR_TRANSACTION_EVIDENCE",
    "DCR_TRANSACTION_VERSION",
    "DEFAULT_TRANSACTION_RECEIPTS_PATH",
    "FENCED_WRITE_INTERFACE",
    "MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE",
    "ROLLBACK_JOURNAL_INTERFACE",
    "FencedWrite",
    "MultiRootRepairTransaction",
    "MultiRootTransactionError",
    "PathLeaseBinding",
    "RollbackJournal",
    "TransactionDisposition",
    "TransactionReceipt",
    "TransactionRejectReason",
    "materialize_transaction_receipts",
]
