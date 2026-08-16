"""DCR-072 isolated-worktree transaction foundation.

This module deliberately does not discover Git worktrees or mutate a caller's
checkout.  DCR-070's typed admission receipt is not available yet, so the
public controller validates a supplied isolated worktree and then returns an
explicit integration-pending receipt without writing anything.  The strict
bindings below are the narrow future hand-off point; they are not authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .root_ownership import RootBinding

DCR072_TRANSACTION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/dcr-072-transaction@1"
DCR072_ISOLATED_MARKER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/dcr-072-isolated-worktree-marker@1"
)
ISOLATED_WORKTREE_MARKER: Final = ".dcr072-isolated-worktree.json"
_MAX_WRITE_BYTES: Final = 8 * 1024 * 1024


class TransactionDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed result states; none grants completion or merge authority."""

    INTEGRATION_PENDING = "integration_pending"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    VALIDATION_PENDING = "validation_pending"


class TransactionState(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed transaction journal lifecycle."""

    RECEIVED = "received"
    FENCED = "fenced"
    INTEGRATION_PENDING = "integration_pending"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    VALIDATION_PENDING = "validation_pending"


class TransactionDenied(ValueError):
    """A transaction binding was incomplete, stale, or unsafe."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TransactionDenied(f"{field} must be non-empty canonical text")
    if any(character.isspace() for character in value):
        raise TransactionDenied(f"{field} must not contain whitespace")
    return value


def _digest(value: object, field: str) -> str:
    value = _text(value, field)
    if not value.startswith("sha256:") or len(value) != 71:
        raise TransactionDenied(f"{field} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise TransactionDenied(f"{field} must be a sha256 digest") from exc
    return value


def _relative_path(value: object, field: str) -> str:
    value = _text(value, field).replace("\\", "/")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "\x00" in value:
        raise TransactionDenied(f"{field} escapes the isolated worktree")
    if candidate.as_posix() in {".", ISOLATED_WORKTREE_MARKER}:
        raise TransactionDenied(f"{field} is reserved")
    return candidate.as_posix()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransactionDenied(f"{field} must be an object")
    return value


def isolated_baseline_digest(root: Path | str) -> str:
    """Digest caller-supplied isolated bytes without Git or checkout discovery.

    Symlinks are categorically denied so a preflight digest cannot conceal a
    path escape.  The marker is excluded because it authenticates this local
    isolation boundary rather than a governed source byte.
    """

    real_root = Path(root).resolve(strict=True)
    if not real_root.is_dir():
        raise TransactionDenied("isolated root must be a directory")
    digest = hashlib.sha256()
    for candidate in sorted(real_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(real_root).as_posix()
        if relative == ISOLATED_WORKTREE_MARKER:
            continue
        if candidate.is_symlink():
            raise TransactionDenied("isolated worktree contains a symlink")
        if candidate.is_dir():
            digest.update(b"directory\0" + relative.encode("utf-8") + b"\0")
            continue
        if not candidate.is_file():
            raise TransactionDenied("isolated worktree contains an unsupported entry")
        digest.update(b"file\0" + relative.encode("utf-8") + b"\0")
        digest.update(candidate.read_bytes())
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class Dcr003WorktreeBinding:
    """DCR-003 identity plus the owner and clean DCR-072 baseline."""

    owner: str
    root: RootBinding
    clean_baseline_digest: str

    def __post_init__(self) -> None:
        _text(self.owner, "owner")
        _digest(self.clean_baseline_digest, "clean_baseline_digest")
        if self.root.dirty:
            raise TransactionDenied("DCR-003 root binding must be clean")
        if self.root.overlay_digest != self.clean_baseline_digest:
            raise TransactionDenied("DCR-003 overlay and clean baseline must match")

    def to_dict(self) -> dict[str, object]:
        return {
            "clean_baseline_digest": self.clean_baseline_digest,
            "owner": self.owner,
            "root": self.root.to_dict(),
        }


@dataclass(frozen=True)
class Dcr070AdmissionBinding:
    """Temporary closed DCR-070 input binding; it cannot authorize a write.

    A real DCR-070 typed receipt must replace this object before this module
    receives a mutation-capable route.  Its CID is merely a dependency label.
    """

    admission_cid: str
    integration_pending: bool = True

    def __post_init__(self) -> None:
        _text(self.admission_cid, "admission_cid")
        if self.integration_pending is not True:
            raise TransactionDenied("fallback DCR-070 binding is always integration pending")

    @property
    def mutation_authorized(self) -> bool:
        return False


@dataclass(frozen=True)
class Dcr071FencedWrite:
    """Exact static preview byte pair and its inverse, before any mutation."""

    relative_path: str
    before_digest: str
    after_bytes: bytes
    after_digest: str
    inverse_bytes: bytes
    inverse_digest: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, "relative_path")
        _digest(self.before_digest, "before_digest")
        _digest(self.after_digest, "after_digest")
        _digest(self.inverse_digest, "inverse_digest")
        for value, field in ((self.after_bytes, "after_bytes"), (self.inverse_bytes, "inverse_bytes")):
            if not isinstance(value, bytes) or len(value) > _MAX_WRITE_BYTES:
                raise TransactionDenied(f"{field} must be bounded bytes")
        if _sha256(self.after_bytes) != self.after_digest:
            raise TransactionDenied("after bytes do not match after digest")
        if _sha256(self.inverse_bytes) != self.inverse_digest:
            raise TransactionDenied("inverse bytes do not match inverse digest")
        if self.before_digest != self.inverse_digest:
            raise TransactionDenied("inverse must restore the exact before digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "after_digest": self.after_digest,
            "before_digest": self.before_digest,
            "inverse_digest": self.inverse_digest,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class Dcr071OperatorPreview:
    """DCR-071 preview binding and its exact fenced write set."""

    preview_cid: str
    operator_id: str
    writes: tuple[Dcr071FencedWrite, ...]

    def __post_init__(self) -> None:
        _text(self.preview_cid, "preview_cid")
        _text(self.operator_id, "operator_id")
        if not self.writes:
            raise TransactionDenied("operator preview must contain at least one write")
        paths = tuple(write.relative_path for write in self.writes)
        if len(paths) != len(set(paths)):
            raise TransactionDenied("operator preview has duplicate write paths")

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "preview_cid": self.preview_cid,
            "writes": [write.to_dict() for write in self.writes],
        }


@dataclass(frozen=True)
class TransactionRequest:
    """Closed caller input; only its explicit root may be inspected."""

    transaction_id: str
    lease_id: str
    fence_id: str
    dcr003: Dcr003WorktreeBinding
    dcr070: Dcr070AdmissionBinding
    dcr071: Dcr071OperatorPreview
    cancelled: bool = False

    def __post_init__(self) -> None:
        for value, field in (
            (self.transaction_id, "transaction_id"),
            (self.lease_id, "lease_id"),
            (self.fence_id, "fence_id"),
        ):
            _text(value, field)
        if self.lease_id == self.fence_id:
            raise TransactionDenied("lease and fence identities must be distinct")
        if not isinstance(self.cancelled, bool):
            raise TransactionDenied("cancelled must be boolean")


@dataclass(frozen=True)
class FencedWriteReceipt:
    """Per-write non-promotion receipt.  This foundation never promotes bytes."""

    relative_path: str
    before_digest: str
    after_digest: str
    inverse_digest: str
    fence_id: str
    promoted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "after_digest": self.after_digest,
            "before_digest": self.before_digest,
            "fence_id": self.fence_id,
            "inverse_digest": self.inverse_digest,
            "promoted": self.promoted,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class TransactionJournal:
    """Canonical auditable state for one never-promoted transaction."""

    transaction_id: str
    state: TransactionState
    disposition: TransactionDisposition
    reason: str
    root_realpath: str
    baseline_digest: str | None
    admission_cid: str
    preview_cid: str
    lease_id: str
    fence_id: str
    writes: tuple[FencedWriteReceipt, ...]
    rollback_verified: bool

    @property
    def journal_cid(self) -> str:
        return content_identity(self.to_dict())

    @property
    def execution_authorized(self) -> bool:
        return False

    @property
    def completion_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "admission_cid": self.admission_cid,
            "baseline_digest": self.baseline_digest,
            "completion_authorized": False,
            "disposition": self.disposition.value,
            "execution_authorized": False,
            "fence_id": self.fence_id,
            "interface": "Dcr072IsolatedTransaction@1",
            "lease_id": self.lease_id,
            "preview_cid": self.preview_cid,
            "reason": self.reason,
            "rollback_verified": self.rollback_verified,
            "root_realpath": self.root_realpath,
            "schema": DCR072_TRANSACTION_SCHEMA,
            "state": self.state.value,
            "transaction_id": self.transaction_id,
            "writes": [write.to_dict() for write in self.writes],
        }


def _marker(root: Path) -> Mapping[str, Any]:
    marker = root / ISOLATED_WORKTREE_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise TransactionDenied("isolated-worktree marker is absent or unsafe")
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransactionDenied("isolated-worktree marker is not canonical JSON") from exc
    record = _mapping(value, "isolated-worktree marker")
    expected = {"baseline_digest", "owner", "root_realpath", "schema"}
    if set(record) != expected or record.get("schema") != DCR072_ISOLATED_MARKER_SCHEMA:
        raise TransactionDenied("isolated-worktree marker schema is not exact")
    return record


class IsolatedTransactionController:
    """Validate a caller-created isolated owner worktree without discovering one."""

    def _validate(self, request: TransactionRequest, root: Path | str) -> tuple[Path, str]:
        real_root = Path(root).resolve(strict=True)
        if not real_root.is_dir():
            raise TransactionDenied("isolated root must be a directory")
        bound_root = Path(request.dcr003.root.realpath).resolve(strict=True)
        if real_root != bound_root:
            raise TransactionDenied("caller root does not match DCR-003 realpath")
        marker = _marker(real_root)
        if marker["owner"] != request.dcr003.owner:
            raise TransactionDenied("isolated-worktree marker owner does not match")
        if marker["root_realpath"] != str(real_root):
            raise TransactionDenied("isolated-worktree marker root does not match")
        if marker["baseline_digest"] != request.dcr003.clean_baseline_digest:
            raise TransactionDenied("isolated-worktree marker baseline does not match")
        baseline = isolated_baseline_digest(real_root)
        if baseline != request.dcr003.clean_baseline_digest:
            raise TransactionDenied("isolated worktree baseline is dirty or stale")
        for write in request.dcr071.writes:
            candidate = real_root / write.relative_path
            if candidate.is_symlink() or not candidate.is_file():
                raise TransactionDenied("fenced write target is absent or unsafe")
            if candidate.resolve(strict=True).parent != (real_root / write.relative_path).parent.resolve(
                strict=True
            ):
                raise TransactionDenied("fenced write target escapes the isolated root")
            if _sha256(candidate.read_bytes()) != write.before_digest:
                raise TransactionDenied("fenced write before bytes are stale")
        return real_root, baseline

    def run(self, request: TransactionRequest, *, isolated_root: Path | str) -> TransactionJournal:
        """Return a fail-closed journal and never write until real DCR-070 exists."""

        if not isinstance(request, TransactionRequest):
            raise TransactionDenied("request must be a typed TransactionRequest")
        try:
            root, baseline = self._validate(request, isolated_root)
        except (OSError, TransactionDenied) as exc:
            return self._journal(request, Path(isolated_root), TransactionState.REJECTED,
                TransactionDisposition.REJECTED, str(exc), None, False)
        if request.cancelled:
            return self._journal(request, root, TransactionState.CANCELLED,
                TransactionDisposition.CANCELLED, "cancelled before any fenced write", baseline, True)
        # This is intentionally the sole post-validation route until an exact
        # DCR-070 type is integrated.  No caller-controlled CID or boolean can
        # alter it, so fake admissions cannot turn a preview into a mutation.
        return self._journal(request, root, TransactionState.INTEGRATION_PENDING,
            TransactionDisposition.INTEGRATION_PENDING,
            "DCR-070 typed admission integration is pending; no bytes promoted", baseline, True)

    @staticmethod
    def _journal(
        request: TransactionRequest,
        root: Path,
        state: TransactionState,
        disposition: TransactionDisposition,
        reason: str,
        baseline: str | None,
        rollback_verified: bool,
    ) -> TransactionJournal:
        writes = tuple(
            FencedWriteReceipt(
                relative_path=write.relative_path,
                before_digest=write.before_digest,
                after_digest=write.after_digest,
                inverse_digest=write.inverse_digest,
                fence_id=request.fence_id,
            )
            for write in request.dcr071.writes
        )
        return TransactionJournal(
            transaction_id=request.transaction_id,
            state=state,
            disposition=disposition,
            reason=reason,
            root_realpath=str(root.resolve(strict=False)),
            baseline_digest=baseline,
            admission_cid=request.dcr070.admission_cid,
            preview_cid=request.dcr071.preview_cid,
            lease_id=request.lease_id,
            fence_id=request.fence_id,
            writes=writes,
            rollback_verified=rollback_verified,
        )


def canonical_transaction_journal_bytes(journal: TransactionJournal) -> bytes:
    """Canonical serialization for a typed journal only."""

    if not isinstance(journal, TransactionJournal):
        raise TransactionDenied("journal must be a TransactionJournal")
    return canonical_json_bytes(journal.to_dict())


__all__ = [
    "DCR072_ISOLATED_MARKER_SCHEMA",
    "DCR072_TRANSACTION_SCHEMA",
    "Dcr003WorktreeBinding",
    "Dcr070AdmissionBinding",
    "Dcr071FencedWrite",
    "Dcr071OperatorPreview",
    "FencedWriteReceipt",
    "ISOLATED_WORKTREE_MARKER",
    "IsolatedTransactionController",
    "TransactionDenied",
    "TransactionDisposition",
    "TransactionJournal",
    "TransactionRequest",
    "TransactionState",
    "canonical_transaction_journal_bytes",
    "isolated_baseline_digest",
]
