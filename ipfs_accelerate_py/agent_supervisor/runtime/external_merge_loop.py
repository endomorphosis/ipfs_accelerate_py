"""In-memory admitted-path merge loop (EAAEF-100).

Verify patch and bound receipts, require an independent reviewer principal
(never the worker), enqueue, apply the patch with exact authority, recompute
the canonical repository tree identity, and settle the merge queue before
downstream acceptance.

This loop is process-local. It does not git-push, talk to live Quack, or
treat DuckLake as merge authority. Missing or forged receipts fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_MERGE_LOOP_INTERFACE: Final[str] = "ExternalMergeLoop@1"
EXTERNAL_MERGE_LOOP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-merge-loop@1"
)
MERGE_LOOP_RECEIPT_INTERFACE: Final[str] = "ExternalMergeLoopReceipt@1"
MERGE_LOOP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-merge-loop-receipt@1"
)
MERGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-merge-bound-receipt@1"
)
GIT_TREE_OR_PATCH_INTERFACE: Final[str] = "GitTreeOrPatchIdentity@1"
GIT_TREE_OR_PATCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/git-tree-or-patch-identity@1"
)
MERGE_QUEUE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-merge-queue-entry@1"
)

REQUIRED_RECEIPT_KINDS: Final[tuple[str, ...]] = ("patch", "review", "test")
ALLOWED_RECEIPT_KINDS: Final[frozenset[str]] = frozenset(
    (*REQUIRED_RECEIPT_KINDS, "proof")
)
MERGE_LOOP_STEPS: Final[tuple[str, ...]] = (
    "verify_receipts",
    "require_independent_reviewer",
    "enqueue",
    "apply",
    "recompute_tree_identity",
    "settle_queue",
)

_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {"content_id", "cid", "identity", "canonical_id"}
)


class MergeLoopError(ValueError):
    """Admitted-path merge failed closed."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class WorkerSelfMergeError(MergeLoopError):
    """A worker principal attempted to merge its own work."""


class MissingReceiptError(MergeLoopError):
    """A required patch or bound receipt was absent."""


class ForgedReceiptError(MergeLoopError):
    """A bound receipt identity did not match its canonical payload."""


class MergeAuthorityError(MergeLoopError):
    """Base tree, patch, or reviewer authority did not match exactly."""


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise MergeLoopError(f"{name} is required", reason_code="malformed")
    if "\x00" in text:
        raise MergeLoopError(f"{name} must not contain NUL", reason_code="malformed")
    return text


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise MergeLoopError(f"{name} must be an object", reason_code="malformed")


def _receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): receipt[key] for key in receipt if str(key) not in _IDENTITY_KEYS}


def canonical_tree_identity(files: Mapping[str, str]) -> str:
    """Return the content-addressed identity of an in-memory repository tree."""

    entries = [
        {
            "path": path,
            "blob_id": content_identity({"path": path, "blob": content}),
        }
        for path, content in sorted(
            (str(path), str(content)) for path, content in files.items()
        )
    ]
    return content_identity(
        {
            "schema": GIT_TREE_OR_PATCH_SCHEMA,
            "interface": GIT_TREE_OR_PATCH_INTERFACE,
            "kind": "git_tree",
            "entries": entries,
        }
    )


def patch_identity(files: Mapping[str, str]) -> str:
    """Return the content-addressed identity of a patch overlay."""

    return content_identity(
        {
            "schema": GIT_TREE_OR_PATCH_SCHEMA,
            "interface": GIT_TREE_OR_PATCH_INTERFACE,
            "kind": "patch",
            "files": dict(sorted((str(path), str(content)) for path, content in files.items())),
        }
    )


def issue_receipt(*, kind: str, **fields: Any) -> dict[str, Any]:
    """Issue a bound receipt whose content identity covers the canonical payload."""

    kind_name = _text(kind, "kind")
    if kind_name not in ALLOWED_RECEIPT_KINDS:
        raise MergeLoopError(f"unknown receipt kind: {kind_name}", reason_code="malformed")
    payload = {
        "schema": MERGE_RECEIPT_SCHEMA,
        "kind": kind_name,
        **{str(key): value for key, value in fields.items() if str(key) not in _IDENTITY_KEYS},
    }
    receipt = dict(payload)
    receipt["content_id"] = content_identity(payload)
    return receipt


def verify_receipt(receipt: object, *, kind: str = "") -> dict[str, Any]:
    """Fail closed on a missing or forged bound receipt."""

    if receipt is None:
        raise MissingReceiptError(
            f"{kind or 'required'} receipt is missing",
            reason_code="missing_receipt",
        )
    body = _mapping(receipt, f"{kind or 'required'} receipt")
    claimed = _text(body.get("content_id") or body.get("cid"), "receipt content identity")
    expected = content_identity(_receipt_payload(body))
    if claimed != expected:
        raise ForgedReceiptError("forged receipt rejected", reason_code="forged_receipt")
    observed_kind = _text(body.get("kind"), "receipt kind")
    if kind and observed_kind != kind:
        raise MergeLoopError(
            "receipt kind does not match the admitted slot",
            reason_code="malformed",
        )
    if observed_kind not in ALLOWED_RECEIPT_KINDS:
        raise MergeLoopError(
            f"unknown receipt kind: {observed_kind}",
            reason_code="malformed",
        )
    return dict(body)


def verify_receipts(receipts: object) -> dict[str, dict[str, Any]]:
    """Verify every required receipt; missing members fail closed."""

    body = _mapping(receipts, "receipts")
    verified: dict[str, dict[str, Any]] = {}
    for kind in REQUIRED_RECEIPT_KINDS:
        if kind not in body or body.get(kind) is None:
            raise MissingReceiptError(
                f"{kind} receipt is missing",
                reason_code="missing_receipt",
            )
        verified[kind] = verify_receipt(body[kind], kind=kind)
    for kind, receipt in body.items():
        name = _text(kind, "receipt kind")
        if name in verified:
            continue
        if name not in ALLOWED_RECEIPT_KINDS:
            raise MergeLoopError(
                f"unknown receipt kind: {name}",
                reason_code="malformed",
            )
        verified[name] = verify_receipt(receipt, kind=name)
    return verified


def require_independent_reviewer(
    *,
    worker_principal_id: object,
    reviewer_principal_id: object,
    principal_id: object = "",
) -> str:
    """Reject worker self-merge; require a distinct reviewer principal."""

    worker = _text(worker_principal_id, "worker_principal_id")
    reviewer = _text(
        reviewer_principal_id,
        "reviewer_principal_id",
        required=False,
    )
    caller = _text(principal_id, "principal_id", required=False)
    if not reviewer:
        raise WorkerSelfMergeError(
            "merge requires an independent reviewer principal",
            reason_code="missing_reviewer",
        )
    if reviewer == worker or (caller and caller == worker):
        raise WorkerSelfMergeError(
            "worker self-merge is forbidden",
            reason_code="worker_self_merge",
        )
    return reviewer


def _files_mapping(value: object, name: str) -> dict[str, str]:
    body = _mapping(value, name)
    files: dict[str, str] = {}
    for path, content in body.items():
        name_path = _text(path, "path")
        if name_path in files:
            raise MergeLoopError("duplicate patch path", reason_code="malformed")
        files[name_path] = str(content)
    return files


@dataclass(frozen=True)
class MergeQueueEntry:
    """One in-memory merge-queue member awaiting apply or settlement."""

    entry_id: str
    task_id: str
    worker_principal_id: str
    reviewer_principal_id: str
    principal_id: str
    base_tree_id: str
    patch_id: str
    files: Mapping[str, str]
    status: str

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": MERGE_QUEUE_ENTRY_SCHEMA,
                "entry_id": self.entry_id,
                "task_id": self.task_id,
                "worker_principal_id": self.worker_principal_id,
                "reviewer_principal_id": self.reviewer_principal_id,
                "principal_id": self.principal_id,
                "base_tree_id": self.base_tree_id,
                "patch_id": self.patch_id,
                "files": dict(self.files),
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class MergeLoopReceipt:
    """Canonical result of one admitted merge or an empty-queue settlement."""

    status: str
    task_id: str
    worker_principal_id: str
    reviewer_principal_id: str
    principal_id: str
    base_tree_id: str
    tree_id: str
    patch_id: str
    queue_status: str
    queue_length: int
    steps: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _text(self.status, "status"))
        if self.status not in {"merged", "settled"}:
            raise MergeLoopError("unknown merge-loop status", reason_code="malformed")
        object.__setattr__(self, "queue_status", _text(self.queue_status, "queue_status"))
        if self.queue_status != "settled":
            raise MergeLoopError(
                "downstream acceptance requires a settled merge queue",
                reason_code="unsettled_queue",
            )
        if int(self.queue_length) != 0:
            raise MergeLoopError(
                "downstream acceptance requires an empty merge queue",
                reason_code="unsettled_queue",
            )
        object.__setattr__(self, "steps", tuple(self.steps))

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": MERGE_LOOP_RECEIPT_SCHEMA,
                "interface": MERGE_LOOP_RECEIPT_INTERFACE,
                "contract_version": CONTRACT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": self.status,
                "task_id": self.task_id,
                "worker_principal_id": self.worker_principal_id,
                "reviewer_principal_id": self.reviewer_principal_id,
                "principal_id": self.principal_id,
                "base_tree_id": self.base_tree_id,
                "tree_id": self.tree_id,
                "patch_id": self.patch_id,
                "queue_status": self.queue_status,
                "queue_length": int(self.queue_length),
                "steps": list(self.steps),
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))


class ExternalMergeLoop:
    """Process-local merge queue, tree, and settlement authority."""

    INTERFACE: Final[str] = EXTERNAL_MERGE_LOOP_INTERFACE
    SCHEMA: Final[str] = EXTERNAL_MERGE_LOOP_SCHEMA

    def __init__(self, *, files: Mapping[str, str] | None = None) -> None:
        self._files: dict[str, str] = _files_mapping(files or {}, "files")
        self._tree_id: str = canonical_tree_identity(self._files)
        self._queue: list[MergeQueueEntry] = []
        self._ordinal: int = 0

    @property
    def tree_id(self) -> str:
        return self._tree_id

    @property
    def files(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._files))

    @property
    def queue(self) -> tuple[MergeQueueEntry, ...]:
        return tuple(self._queue)

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    def verify_receipts(self, receipts: object) -> dict[str, dict[str, Any]]:
        return verify_receipts(receipts)

    def require_independent_reviewer(
        self,
        *,
        worker_principal_id: object,
        reviewer_principal_id: object,
        principal_id: object = "",
    ) -> str:
        return require_independent_reviewer(
            worker_principal_id=worker_principal_id,
            reviewer_principal_id=reviewer_principal_id,
            principal_id=principal_id,
        )

    def enqueue(
        self,
        *,
        task_id: str,
        worker_principal_id: str,
        reviewer_principal_id: str,
        principal_id: str,
        base_tree_id: str,
        patch_id: str,
        files: Mapping[str, str],
    ) -> MergeQueueEntry:
        if base_tree_id != self._tree_id:
            raise MergeAuthorityError(
                "base tree does not match the canonical repository state",
                reason_code="stale_base",
            )
        observed_patch = patch_identity(files)
        if observed_patch != patch_id:
            raise MergeAuthorityError(
                "patch identity does not match the admitted overlay",
                reason_code="patch_mismatch",
            )
        self._ordinal += 1
        entry = MergeQueueEntry(
            entry_id=content_identity(
                {
                    "schema": MERGE_QUEUE_ENTRY_SCHEMA,
                    "ordinal": self._ordinal,
                    "task_id": task_id,
                    "patch_id": patch_id,
                    "base_tree_id": base_tree_id,
                }
            ),
            task_id=task_id,
            worker_principal_id=worker_principal_id,
            reviewer_principal_id=reviewer_principal_id,
            principal_id=principal_id,
            base_tree_id=base_tree_id,
            patch_id=patch_id,
            files=MappingProxyType(dict(files)),
            status="pending",
        )
        self._queue.append(entry)
        return entry

    def apply(self, entry: MergeQueueEntry) -> MergeQueueEntry:
        if entry not in self._queue:
            raise MergeLoopError("unknown merge-queue entry", reason_code="unknown_entry")
        if entry.base_tree_id != self._tree_id:
            raise MergeAuthorityError(
                "base tree does not match the canonical repository state",
                reason_code="stale_base",
            )
        applied_files = dict(self._files)
        applied_files.update(dict(entry.files))
        self._files = applied_files
        index = self._queue.index(entry)
        applied = MergeQueueEntry(
            entry_id=entry.entry_id,
            task_id=entry.task_id,
            worker_principal_id=entry.worker_principal_id,
            reviewer_principal_id=entry.reviewer_principal_id,
            principal_id=entry.principal_id,
            base_tree_id=entry.base_tree_id,
            patch_id=entry.patch_id,
            files=entry.files,
            status="applied",
        )
        self._queue[index] = applied
        return applied

    def recompute_tree_identity(self) -> str:
        self._tree_id = canonical_tree_identity(self._files)
        return self._tree_id

    def settle(self, entry: MergeQueueEntry | None = None) -> MergeLoopReceipt:
        if entry is None:
            if self._queue:
                raise MergeLoopError(
                    "merge queue has unsettled entries",
                    reason_code="unsettled_queue",
                )
            return MergeLoopReceipt(
                status="settled",
                task_id="",
                worker_principal_id="",
                reviewer_principal_id="",
                principal_id="",
                base_tree_id=self._tree_id,
                tree_id=self._tree_id,
                patch_id="",
                queue_status="settled",
                queue_length=0,
                steps=("settle_queue",),
            )
        if entry not in self._queue:
            raise MergeLoopError("unknown merge-queue entry", reason_code="unknown_entry")
        if entry.status != "applied":
            raise MergeLoopError(
                "cannot settle an unapplied merge-queue entry",
                reason_code="unsettled_queue",
            )
        self._queue = [item for item in self._queue if item.entry_id != entry.entry_id]
        if self._queue:
            raise MergeLoopError(
                "downstream acceptance requires an empty merge queue",
                reason_code="unsettled_queue",
            )
        return MergeLoopReceipt(
            status="merged",
            task_id=entry.task_id,
            worker_principal_id=entry.worker_principal_id,
            reviewer_principal_id=entry.reviewer_principal_id,
            principal_id=entry.principal_id,
            base_tree_id=entry.base_tree_id,
            tree_id=self._tree_id,
            patch_id=entry.patch_id,
            queue_status="settled",
            queue_length=0,
            steps=MERGE_LOOP_STEPS,
        )

    def merge(self, proposal: Mapping[str, Any] | MergeLoopReceipt) -> MergeLoopReceipt:
        """Run the admitted path and return a settled merge receipt."""

        body = proposal.to_dict() if isinstance(proposal, MergeLoopReceipt) else _mapping(
            proposal, "proposal"
        )
        task_id = _text(body.get("task_id"), "task_id")
        worker = _text(body.get("worker_principal_id"), "worker_principal_id")
        reviewer = _text(
            body.get("reviewer_principal_id"),
            "reviewer_principal_id",
            required=False,
        )
        principal = _text(body.get("principal_id"), "principal_id", required=False) or reviewer
        files = _files_mapping(body.get("files") or {}, "files")
        if not files:
            raise MergeLoopError("patch files are required", reason_code="malformed")
        patch_id = patch_identity(files)
        receipts = self.verify_receipts(body.get("receipts") or {})
        self._bind_exact_authority(
            receipts,
            task_id=task_id,
            patch_id=patch_id,
            reviewer_principal_id=reviewer,
        )
        reviewer = self.require_independent_reviewer(
            worker_principal_id=worker,
            reviewer_principal_id=reviewer,
            principal_id=principal,
        )
        base_tree_id = _text(body.get("base_tree_id"), "base_tree_id")
        entry = self.enqueue(
            task_id=task_id,
            worker_principal_id=worker,
            reviewer_principal_id=reviewer,
            principal_id=principal,
            base_tree_id=base_tree_id,
            patch_id=patch_id,
            files=files,
        )
        applied = self.apply(entry)
        self.recompute_tree_identity()
        return self.settle(applied)

    def _bind_exact_authority(
        self,
        receipts: Mapping[str, Mapping[str, Any]],
        *,
        task_id: str,
        patch_id: str,
        reviewer_principal_id: str,
    ) -> None:
        for kind, receipt in receipts.items():
            bound_patch = _text(receipt.get("patch_id"), f"{kind} patch_id", required=False)
            if bound_patch and bound_patch != patch_id:
                raise MergeAuthorityError(
                    "receipt is not bound to the admitted patch",
                    reason_code="authority_mismatch",
                )
            bound_task = _text(receipt.get("task_id"), f"{kind} task_id", required=False)
            if bound_task and bound_task != task_id:
                raise MergeAuthorityError(
                    "receipt is not bound to the admitted task",
                    reason_code="authority_mismatch",
                )
        review = receipts["review"]
        bound_reviewer = _text(
            review.get("principal_id") or review.get("reviewer_principal_id"),
            "review principal",
            required=False,
        )
        if reviewer_principal_id and bound_reviewer and bound_reviewer != reviewer_principal_id:
            raise MergeAuthorityError(
                "review receipt is not bound to the independent reviewer",
                reason_code="authority_mismatch",
            )
        decision = _text(review.get("decision"), "review decision", required=False)
        if decision and decision != "accept":
            raise MergeAuthorityError(
                "review receipt does not accept the admitted patch",
                reason_code="authority_mismatch",
            )


def merge_accepted_result(
    proposal: Mapping[str, Any],
    *,
    loop: ExternalMergeLoop | None = None,
) -> MergeLoopReceipt:
    """Merge one accepted result through a process-local admitted path."""

    host = loop if loop is not None else ExternalMergeLoop()
    return host.merge(proposal)


__all__ = (
    "ALLOWED_RECEIPT_KINDS",
    "CONTRACT_VERSION",
    "EXTERNAL_MERGE_LOOP_INTERFACE",
    "EXTERNAL_MERGE_LOOP_SCHEMA",
    "ExternalMergeLoop",
    "ForgedReceiptError",
    "GIT_TREE_OR_PATCH_INTERFACE",
    "GIT_TREE_OR_PATCH_SCHEMA",
    "MERGE_LOOP_RECEIPT_INTERFACE",
    "MERGE_LOOP_RECEIPT_SCHEMA",
    "MERGE_LOOP_STEPS",
    "MERGE_RECEIPT_SCHEMA",
    "MergeAuthorityError",
    "MergeLoopError",
    "MergeLoopReceipt",
    "MergeQueueEntry",
    "MissingReceiptError",
    "REQUIRED_RECEIPT_KINDS",
    "SCHEMA_VERSION",
    "WorkerSelfMergeError",
    "canonical_tree_identity",
    "issue_receipt",
    "merge_accepted_result",
    "patch_identity",
    "require_independent_reviewer",
    "verify_receipt",
    "verify_receipts",
)
