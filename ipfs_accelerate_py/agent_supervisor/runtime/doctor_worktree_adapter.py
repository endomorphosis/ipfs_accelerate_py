"""Real isolated worktree mutation for the deterministic Doctor.

Interface: ``DoctorWorktreeAdapter@1``

This module is intentionally narrower than a general Git wrapper.  It creates
one detached, no-checkout worktree, materialises tracked regular files directly
from Git objects (so checkout filters and hooks cannot execute), applies exact
whole-file byte replacements, and integrates the resulting tree with
``git update-ref <ref> <new> <old>`` compare-and-swap.

The adapter never executes target code.  Every child process is a fixed local
Git command under a scrubbed environment, with hooks, protocols, global
configuration, prompts, and credential helpers disabled.  Symlinks, hardlinks,
special files, path escapes, edits outside the allowlist, and unexpected files
fail closed.  Durable intent/effect journals and byte checkpoints are fsynced
before mutation/ref-CAS.  Restore rereads the complete tracked forest and
compares independently recomputed blob/tree/forest CIDs before it can claim
success.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..multiformats_identity import cid_for_bytes, cid_for_dag_json


DOCTOR_WORKTREE_ADAPTER_INTERFACE: Final[str] = "DoctorWorktreeAdapter@1"
DOCTOR_WORKTREE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-worktree-snapshot@1"
)
DOCTOR_WORKTREE_APPLY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-worktree-apply@1"
)
DOCTOR_WORKTREE_INTENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-worktree-intent@1"
)
DOCTOR_WORKTREE_QUARANTINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-worktree-quarantine@1"
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")
_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^refs/(?:heads|doctor)/[A-Za-z0-9][A-Za-z0-9._/-]*$"
)
_SESSION_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_MAX_PATH_BYTES: Final[int] = 1_024
_MAX_EDIT_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_FILES: Final[int] = 100_000
_MAX_CHECKPOINT_BYTES: Final[int] = 512 * 1024 * 1024


class DoctorWorktreeError(RuntimeError):
    """A worktree operation could not prove confinement or exact effects."""


class DoctorWorktreeSecurityError(DoctorWorktreeError):
    """A path, filesystem object, environment, or process boundary was unsafe."""


class DoctorWorktreeTamperError(DoctorWorktreeSecurityError):
    """Observed bytes or repository state disagreed with an authority binding."""


class DoctorWorktreeCasError(DoctorWorktreeError):
    """The target ref changed before the compare-and-swap."""


class DoctorWorktreeQuarantineError(DoctorWorktreeError):
    """Exact restoration failed and the worktree has been quarantined."""


class DoctorWorktreeState(str, Enum):
    PREPARED = "prepared"
    APPLYING = "applying"
    EFFECTS_DURABLE = "effects_durable"
    CAS_APPLIED = "cas_applied"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"

    @property
    def terminal(self) -> bool:
        return self in {
            DoctorWorktreeState.COMMITTED,
            DoctorWorktreeState.ROLLED_BACK,
            DoctorWorktreeState.QUARANTINED,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _safe_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SESSION_RE.fullmatch(value):
        raise DoctorWorktreeError(f"{name} must be a compact safe identifier")
    return value


def _repository_path(value: str, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise DoctorWorktreeSecurityError(f"{name} is not an exact repository path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or value in {".", ""}
        or ".." in pure.parts
        or any(part in {"", ".", ".git"} for part in pure.parts)
        or any(char in value for char in "*?[]{}")
        or len(value.encode("utf-8")) > _MAX_PATH_BYTES
    ):
        raise DoctorWorktreeSecurityError(f"{name} escapes the repository")
    return value


def _normal_paths(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorWorktreeSecurityError(f"{name} must be a path sequence")
    paths = tuple(sorted({_repository_path(item, name) for item in values}))
    if not paths:
        raise DoctorWorktreeSecurityError(f"{name} must not be empty")
    return paths


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(dict(value)) + b"\n")


def _secure_file_bytes(path: Path, *, maximum: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise DoctorWorktreeSecurityError(
            f"durable state file is not one regular file: {path}"
        )
    _assert_no_symlink_ancestors(path.parent)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise DoctorWorktreeSecurityError(
                    f"durable state file exceeds its bound: {path}"
                )
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise DoctorWorktreeTamperError(
                f"durable state changed while being read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _assert_no_symlink_ancestors(path: Path, *, stop: Path | None = None) -> None:
    current = path.absolute()
    boundary = stop.absolute() if stop is not None else None
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(info.st_mode):
                raise DoctorWorktreeSecurityError(
                    f"symlink ancestor is forbidden: {current}"
                )
        if current.parent == current or (boundary is not None and current == boundary):
            return
        current = current.parent


@dataclass(frozen=True)
class DoctorExactEdit:
    """One exact whole-file replacement.

    ``before_hash`` binds the complete file bytes.  ``after_bytes`` are the
    complete desired bytes, not a diff.  Requiring both makes reapplication,
    stale spans, newline conversion, and partial writes unambiguous.
    """

    path: str
    before_hash: str
    after_bytes: bytes
    expected_after_hash: str = ""
    step_id: str = ""
    group_id: str = ""
    mode: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repository_path(self.path))
        if not isinstance(self.before_hash, str) or not self.before_hash.strip():
            raise DoctorWorktreeError("before_hash is required")
        object.__setattr__(self, "before_hash", self.before_hash.strip())
        if not isinstance(self.after_bytes, bytes):
            raise DoctorWorktreeError("after_bytes must be exact bytes")
        if len(self.after_bytes) > _MAX_EDIT_BYTES:
            raise DoctorWorktreeError("after_bytes exceeds the per-edit byte bound")
        after_hash = self.expected_after_hash.strip()
        if after_hash and not _SHA256_RE.fullmatch(after_hash):
            raise DoctorWorktreeError("expected_after_hash must be sha256:<hex>")
        if after_hash and after_hash != _sha256(self.after_bytes):
            raise DoctorWorktreeTamperError(
                "expected_after_hash does not match after_bytes"
            )
        object.__setattr__(self, "expected_after_hash", after_hash or _sha256(self.after_bytes))
        for name in ("step_id", "group_id"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _safe_component(value, name))
        if self.mode is not None and self.mode not in {0o100644, 0o100755}:
            raise DoctorWorktreeSecurityError("edit mode must be 100644 or 100755")


# Descriptive aliases retained for callers that use file/worktree terminology.
DoctorFileEdit = DoctorExactEdit
DoctorWorktreeEdit = DoctorExactEdit


@dataclass(frozen=True)
class DoctorGitlink:
    path: str
    commit_oid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repository_path(self.path))
        if not _OID_RE.fullmatch(self.commit_oid):
            raise DoctorWorktreeError("gitlink commit_oid is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "commit_oid": self.commit_oid}


@dataclass(frozen=True)
class DoctorBlobEffect:
    path: str
    mode: int
    before_hash: str
    after_hash: str
    before_blob_cid: str
    after_blob_cid: str
    before_git_oid: str
    after_git_oid: str
    byte_count: int
    step_id: str = ""
    group_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "before_blob_cid": self.before_blob_cid,
            "after_blob_cid": self.after_blob_cid,
            "before_git_oid": self.before_git_oid,
            "after_git_oid": self.after_git_oid,
            "byte_count": self.byte_count,
            "step_id": self.step_id,
            "group_id": self.group_id,
        }


@dataclass(frozen=True)
class DoctorWorktreeSnapshot:
    session_id: str
    worktree_root: str
    base_commit_oid: str
    git_tree_oid: str
    tree_cid: str
    forest_cid: str
    blob_cids: tuple[tuple[str, str], ...]
    path_hashes: tuple[tuple[str, str], ...]
    gitlinks: tuple[DoctorGitlink, ...] = ()

    def hash_map(self) -> dict[str, str]:
        return dict(self.path_hashes)

    def blob_map(self) -> dict[str, str]:
        return dict(self.blob_cids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_WORKTREE_SNAPSHOT_SCHEMA,
            "interface": DOCTOR_WORKTREE_ADAPTER_INTERFACE,
            "session_id": self.session_id,
            "worktree_root": self.worktree_root,
            "base_commit_oid": self.base_commit_oid,
            "git_tree_oid": self.git_tree_oid,
            "tree_cid": self.tree_cid,
            "forest_cid": self.forest_cid,
            "blob_cids": [
                {"path": path, "cid": cid} for path, cid in self.blob_cids
            ],
            "path_hashes": [
                {"path": path, "sha256": digest}
                for path, digest in self.path_hashes
            ],
            "gitlinks": [item.to_dict() for item in self.gitlinks],
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict(), for_identity=True)


@dataclass(frozen=True)
class DoctorWorktreeApplyReceipt:
    session_id: str
    group_id: str
    before_tree_cid: str
    after_tree_cid: str
    before_forest_cid: str
    after_forest_cid: str
    effects: tuple[DoctorBlobEffect, ...]
    bytes_reread: bool = True
    durable_effect_ref: str = ""

    def __post_init__(self) -> None:
        if not self.effects:
            raise DoctorWorktreeError("an apply receipt requires a nonempty expected change")
        if self.before_tree_cid == self.after_tree_cid:
            raise DoctorWorktreeTamperError("apply receipt cannot certify a no-op tree")
        if any(item.before_hash == item.after_hash for item in self.effects):
            raise DoctorWorktreeTamperError("apply receipt contains a no-op effect")

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.effects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_WORKTREE_APPLY_SCHEMA,
            "interface": DOCTOR_WORKTREE_ADAPTER_INTERFACE,
            "session_id": self.session_id,
            "group_id": self.group_id,
            "before_tree_cid": self.before_tree_cid,
            "after_tree_cid": self.after_tree_cid,
            "before_forest_cid": self.before_forest_cid,
            "after_forest_cid": self.after_forest_cid,
            "effects": [item.to_dict() for item in self.effects],
            "changed_paths": list(self.changed_paths),
            "bytes_reread": self.bytes_reread,
            "durable_effect_ref": self.durable_effect_ref,
            "nonempty_expected_change": True,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict(), for_identity=True)


@dataclass(frozen=True)
class DoctorRefCasReceipt:
    session_id: str
    ref_name: str
    expected_commit_oid: str
    desired_commit_oid: str
    git_tree_oid: str
    tree_cid: str
    forest_cid: str
    durable_effect_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ref_name": self.ref_name,
            "expected_commit_oid": self.expected_commit_oid,
            "desired_commit_oid": self.desired_commit_oid,
            "git_tree_oid": self.git_tree_oid,
            "tree_cid": self.tree_cid,
            "forest_cid": self.forest_cid,
            "durable_effect_ref": self.durable_effect_ref,
            "cas_applied": True,
        }


@dataclass(frozen=True)
class DoctorRestoreProof:
    session_id: str
    restored: bool
    quarantined: bool
    expected_tree_cid: str
    observed_tree_cid: str
    expected_forest_cid: str
    observed_forest_cid: str
    ref_restored: bool
    gitlinks_equal: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "restored": self.restored,
            "quarantined": self.quarantined,
            "expected_tree_cid": self.expected_tree_cid,
            "observed_tree_cid": self.observed_tree_cid,
            "expected_forest_cid": self.expected_forest_cid,
            "observed_forest_cid": self.observed_forest_cid,
            "ref_restored": self.ref_restored,
            "gitlinks_equal": self.gitlinks_equal,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    mode: int
    object_type: str
    oid: str


def _parse_tree(raw: bytes) -> tuple[_TreeEntry, ...]:
    result: list[_TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path_bytes = record.split(b"\t", 1)
            mode_raw, object_type_raw, oid_raw = header.split(b" ", 2)
            path = path_bytes.decode("utf-8", "strict")
            mode = int(mode_raw, 8)
            object_type = object_type_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise DoctorWorktreeTamperError("Git returned a malformed tree entry") from exc
        _repository_path(path)
        if not _OID_RE.fullmatch(oid):
            raise DoctorWorktreeTamperError("Git returned a malformed object id")
        result.append(_TreeEntry(path, mode, object_type, oid))
    if len(result) > _MAX_FILES:
        raise DoctorWorktreeError("repository exceeds the tracked-file bound")
    return tuple(result)


def _parse_index(raw: bytes) -> tuple[_TreeEntry, ...]:
    result: list[_TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path_bytes = record.split(b"\t", 1)
            mode_raw, oid_raw, stage_raw = header.split(b" ", 2)
            mode = int(mode_raw, 8)
            oid = oid_raw.decode("ascii")
            stage = int(stage_raw)
            path = path_bytes.decode("utf-8", "strict")
        except (ValueError, UnicodeError) as exc:
            raise DoctorWorktreeTamperError(
                "Git returned a malformed index entry"
            ) from exc
        if stage != 0:
            raise DoctorWorktreeTamperError(
                "unmerged index stages are forbidden"
            )
        _repository_path(path)
        if not _OID_RE.fullmatch(oid):
            raise DoctorWorktreeTamperError("Git index object id is malformed")
        object_type = "commit" if mode == 0o160000 else "blob"
        result.append(_TreeEntry(path, mode, object_type, oid))
    if len({item.path for item in result}) != len(result):
        raise DoctorWorktreeTamperError("Git index has duplicate paths")
    return tuple(result)


@dataclass
class DoctorWorktreeSession:
    """One locked disposable worktree and its durable recovery state."""

    adapter: "DoctorWorktreeAdapter"
    session_id: str
    base_ref: str
    base_commit_oid: str
    worktree_root: Path
    session_dir: Path
    lock_stream: Any
    baseline: DoctorWorktreeSnapshot
    entries: tuple[_TreeEntry, ...]
    git_admin_hash: str = ""
    state: DoctorWorktreeState = DoctorWorktreeState.PREPARED
    target_ref: str = ""
    desired_commit_oid: str = ""
    apply_receipts: list[DoctorWorktreeApplyReceipt] = field(default_factory=list)
    _closed: bool = False

    @property
    def checkpoint_dir(self) -> Path:
        return self.session_dir / "checkpoint"

    @property
    def journal_path(self) -> Path:
        return self.session_dir / "intent.json"

    def _fault(self, boundary: str) -> None:
        callback = self.adapter.fault_injector
        if callback is not None:
            callback(boundary)

    def _journal(self, state: DoctorWorktreeState, **extra: Any) -> str:
        self.state = state
        record: dict[str, Any] = {
            "schema": DOCTOR_WORKTREE_INTENT_SCHEMA,
            "interface": DOCTOR_WORKTREE_ADAPTER_INTERFACE,
            "session_id": self.session_id,
            "repository_root": str(self.adapter.repository_root),
            "worktree_root": str(self.worktree_root),
            "base_ref": self.base_ref,
            "base_commit_oid": self.base_commit_oid,
            "baseline": self.baseline.to_dict(),
            "state": state.value,
            "target_ref": self.target_ref,
            "desired_commit_oid": self.desired_commit_oid,
            "git_admin_hash": self.git_admin_hash,
            "apply_receipt_ids": [item.content_id for item in self.apply_receipts],
            **extra,
        }
        _atomic_json(self.journal_path, record)
        return cid_for_dag_json(record, for_identity=True)

    def write_intent(self) -> str:
        """Persist exact baseline bytes and a PREPARED intent before mutation."""

        self._require_open()
        manifest_entries: list[dict[str, Any]] = []
        total = 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tracked = {item.path: item for item in self.entries if item.object_type == "blob"}
        for index, path in enumerate(sorted(tracked)):
            body = self.adapter._read_confined(self.worktree_root, path)
            total += len(body)
            if total > self.adapter.max_checkpoint_bytes:
                raise DoctorWorktreeError("checkpoint exceeds durable byte bound")
            storage_name = f"{index:08d}.blob"
            _atomic_bytes(self.checkpoint_dir / storage_name, body)
            manifest_entries.append(
                {
                    "path": path,
                    "storage_name": storage_name,
                    "sha256": _sha256(body),
                    "mode": tracked[path].mode,
                    "git_oid": tracked[path].oid,
                }
            )
        manifest = {
            "session_id": self.session_id,
            "base_commit_oid": self.base_commit_oid,
            "tree_cid": self.baseline.tree_cid,
            "forest_cid": self.baseline.forest_cid,
            "gitlinks": [item.to_dict() for item in self.baseline.gitlinks],
            "entries": manifest_entries,
        }
        _atomic_json(self.checkpoint_dir / "manifest.json", manifest)
        checkpoint_id = cid_for_dag_json(manifest, for_identity=True)
        intent_id = self._journal(
            DoctorWorktreeState.PREPARED,
            checkpoint_id=checkpoint_id,
        )
        self._fault("after_intent_fsync")
        return intent_id

    def apply_group(
        self,
        edits: Sequence[DoctorExactEdit],
        *,
        group_id: str,
    ) -> DoctorWorktreeApplyReceipt:
        """Apply one nonempty SCC/impact group or restore its exact pre-state."""

        self._require_open()
        group = _safe_component(group_id, "group_id")
        if not isinstance(edits, Sequence) or isinstance(
            edits, (str, bytes, bytearray)
        ):
            raise DoctorWorktreeError("edits must be a sequence")
        values = tuple(edits)
        if not values:
            raise DoctorWorktreeError("a mutation group requires a nonempty expected change")
        if not all(isinstance(item, DoctorExactEdit) for item in values):
            raise DoctorWorktreeError("edits must be DoctorExactEdit values")
        if len({item.path for item in values}) != len(values):
            raise DoctorWorktreeError("a group cannot edit one path more than once")
        if not self.journal_path.exists():
            self.write_intent()
        self.adapter._scan_worktree(self, expected_new_paths=())
        before = self.adapter.snapshot(self)
        group_before: dict[str, tuple[bytes, int]] = {}
        effects: list[DoctorBlobEffect] = []
        self._journal(DoctorWorktreeState.APPLYING, active_group_id=group)
        self._fault("before_group_apply")
        try:
            for edit in values:
                if edit.path not in self.adapter.permitted_paths:
                    raise DoctorWorktreeSecurityError(
                        f"edit path is outside exact allowlist: {edit.path}"
                    )
                body = self.adapter._read_confined(self.worktree_root, edit.path)
                info = self.adapter._lstat_confined(self.worktree_root, edit.path)
                mode = 0o100755 if info.st_mode & stat.S_IXUSR else 0o100644
                group_before[edit.path] = (body, mode)
                if not self.adapter.hash_matches(body, edit.before_hash):
                    raise DoctorWorktreeTamperError(
                        f"before_hash mismatch for {edit.path}"
                    )
                if body == edit.after_bytes:
                    raise DoctorWorktreeTamperError(
                        f"nonempty expected change is a no-op for {edit.path}"
                    )
                desired_mode = edit.mode or mode
                self.adapter._write_confined(
                    self.worktree_root,
                    edit.path,
                    edit.after_bytes,
                    desired_mode,
                )
                reread = self.adapter._read_confined(self.worktree_root, edit.path)
                if reread != edit.after_bytes:
                    raise DoctorWorktreeTamperError(
                        f"reread bytes disagree after applying {edit.path}"
                    )
                after_hash = _sha256(reread)
                if after_hash != edit.expected_after_hash:
                    raise DoctorWorktreeTamperError(
                        f"after_hash mismatch for {edit.path}"
                    )
                effects.append(
                    DoctorBlobEffect(
                        path=edit.path,
                        mode=desired_mode,
                        before_hash=_sha256(body),
                        after_hash=after_hash,
                        before_blob_cid=cid_for_bytes(body),
                        after_blob_cid=cid_for_bytes(reread),
                        before_git_oid=self.adapter._hash_object(body, write=False),
                        after_git_oid=self.adapter._hash_object(reread, write=False),
                        byte_count=len(reread),
                        step_id=edit.step_id,
                        group_id=group,
                    )
                )
                self._fault(f"after_edit:{edit.path}")
            self.adapter._scan_worktree(self, expected_new_paths=())
            after = self.adapter.snapshot(self)
            receipt = DoctorWorktreeApplyReceipt(
                session_id=self.session_id,
                group_id=group,
                before_tree_cid=before.tree_cid,
                after_tree_cid=after.tree_cid,
                before_forest_cid=before.forest_cid,
                after_forest_cid=after.forest_cid,
                effects=tuple(effects),
            )
            self.apply_receipts.append(receipt)
            durable_ref = self._journal(
                DoctorWorktreeState.APPLYING,
                active_group_id="",
                last_group_id=group,
                last_group_receipt=receipt.to_dict(),
            )
            receipt = DoctorWorktreeApplyReceipt(
                session_id=receipt.session_id,
                group_id=receipt.group_id,
                before_tree_cid=receipt.before_tree_cid,
                after_tree_cid=receipt.after_tree_cid,
                before_forest_cid=receipt.before_forest_cid,
                after_forest_cid=receipt.after_forest_cid,
                effects=receipt.effects,
                durable_effect_ref=durable_ref,
            )
            self.apply_receipts[-1] = receipt
            self._fault("after_group_effect_fsync")
            return receipt
        except BaseException:
            for path, (body, mode) in group_before.items():
                try:
                    self.adapter._write_confined(self.worktree_root, path, body, mode)
                except BaseException:
                    # The full transaction restore below is the authoritative
                    # recovery path and will quarantine on any disagreement.
                    pass
            try:
                self.restore(reason=f"group_failed:{group}")
            except BaseException:
                pass
            raise

    # Convenience spelling used by API callers.
    apply_edits = apply_group

    def commit_ref(
        self,
        *,
        target_ref: str,
        expected_commit_oid: str | None = None,
        message: str = "deterministic doctor transaction",
    ) -> DoctorRefCasReceipt:
        """Write Git objects, fsync effects, then atomically CAS one allowlisted ref."""

        self._require_open()
        if not self.apply_receipts:
            raise DoctorWorktreeError("ref CAS requires a nonempty applied change")
        ref = self.adapter._validate_ref(target_ref)
        expected = expected_commit_oid or self.base_commit_oid
        if not _OID_RE.fullmatch(expected):
            raise DoctorWorktreeError("expected_commit_oid is invalid")
        live = self.adapter._git_text(
            self.adapter.repository_root, "rev-parse", "--verify", f"{ref}^{{commit}}"
        )
        if live != expected:
            raise DoctorWorktreeCasError(
                f"ref CAS conflict: expected {expected}, observed {live}"
            )
        self.target_ref = ref
        self.adapter._scan_worktree(self, expected_new_paths=())
        final_snapshot = self.adapter.snapshot(self)
        entries = {item.path: item for item in self.entries}
        changed_paths = {
            effect.path for receipt in self.apply_receipts for effect in receipt.effects
        }
        for path in sorted(changed_paths):
            body = self.adapter._read_confined(self.worktree_root, path)
            oid = self.adapter._hash_object(body, write=True)
            mode = entries[path].mode
            info = self.adapter._lstat_confined(self.worktree_root, path)
            observed_mode = 0o100755 if info.st_mode & stat.S_IXUSR else 0o100644
            mode = observed_mode if mode in {0o100644, 0o100755} else mode
            self.adapter._git(
                self.worktree_root,
                "update-index",
                "--add",
                "--cacheinfo",
                f"{mode:o},{oid},{path}",
            )
        tree_oid = self.adapter._git_text(self.worktree_root, "write-tree")
        index_tree_cid = self.adapter.snapshot(self).tree_cid
        if index_tree_cid != final_snapshot.tree_cid:
            raise DoctorWorktreeTamperError(
                "index materialization disagrees with independently reread tree"
            )
        env = {
            "GIT_AUTHOR_NAME": "Deterministic Doctor",
            "GIT_AUTHOR_EMAIL": "doctor@invalid.local",
            "GIT_COMMITTER_NAME": "Deterministic Doctor",
            "GIT_COMMITTER_EMAIL": "doctor@invalid.local",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
        commit_oid = self.adapter._git_text(
            self.worktree_root,
            "commit-tree",
            tree_oid,
            "-p",
            expected,
            "-m",
            message,
            extra_env=env,
        )
        self.desired_commit_oid = commit_oid
        durable_ref = self._journal(
            DoctorWorktreeState.EFFECTS_DURABLE,
            target_ref=ref,
            expected_commit_oid=expected,
            desired_commit_oid=commit_oid,
            git_tree_oid=tree_oid,
            tree_cid=final_snapshot.tree_cid,
            forest_cid=final_snapshot.forest_cid,
            apply_receipts=[item.to_dict() for item in self.apply_receipts],
        )
        self._fault("after_effects_fsync_before_cas")
        result = self.adapter._git(
            self.adapter.repository_root,
            "update-ref",
            "--no-deref",
            ref,
            commit_oid,
            expected,
            check=False,
        )
        if result.returncode:
            raise DoctorWorktreeCasError("git update-ref compare-and-swap failed")
        observed = self.adapter._git_text(
            self.adapter.repository_root, "rev-parse", "--verify", f"{ref}^{{commit}}"
        )
        if observed != commit_oid:
            raise DoctorWorktreeTamperError("ref reread disagrees after CAS")
        self._fsync_ref(ref)
        self._journal(
            DoctorWorktreeState.CAS_APPLIED,
            target_ref=ref,
            expected_commit_oid=expected,
            desired_commit_oid=commit_oid,
            git_tree_oid=tree_oid,
            tree_cid=final_snapshot.tree_cid,
            forest_cid=final_snapshot.forest_cid,
            durable_effect_ref=durable_ref,
        )
        self._fault("after_cas_fsync")
        receipt = DoctorRefCasReceipt(
            session_id=self.session_id,
            ref_name=ref,
            expected_commit_oid=expected,
            desired_commit_oid=commit_oid,
            git_tree_oid=tree_oid,
            tree_cid=final_snapshot.tree_cid,
            forest_cid=final_snapshot.forest_cid,
            durable_effect_ref=durable_ref,
        )
        self._journal(
            DoctorWorktreeState.COMMITTED,
            cas_receipt=receipt.to_dict(),
        )
        self._fault("after_commit_fsync")
        return receipt

    # Compatibility spelling matching the contract prose.
    compare_and_swap_ref = commit_ref

    def restore(self, *, reason: str = "transaction_abort") -> DoctorRestoreProof:
        """Restore checkpoint bytes/ref/index and independently compare all roots."""

        self._require_open()
        expected = self.baseline
        ref_restored = True
        observed_tree = ""
        observed_forest = ""
        gitlinks_equal = False
        try:
            manifest_path = self.checkpoint_dir / "manifest.json"
            manifest = json.loads(
                _secure_file_bytes(
                    manifest_path, maximum=16 * 1024 * 1024
                ).decode("utf-8", "strict")
            )
            if not isinstance(manifest, Mapping):
                raise DoctorWorktreeTamperError("checkpoint manifest is malformed")
            if manifest.get("session_id") != self.session_id:
                raise DoctorWorktreeTamperError("checkpoint session binding changed")
            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise DoctorWorktreeTamperError("checkpoint entries are malformed")
            expected_paths: set[str] = set()
            for record in entries:
                if not isinstance(record, Mapping):
                    raise DoctorWorktreeTamperError("checkpoint entry is malformed")
                path = _repository_path(str(record.get("path") or ""))
                storage = str(record.get("storage_name") or "")
                if "/" in storage or not storage.endswith(".blob"):
                    raise DoctorWorktreeTamperError("checkpoint storage path escaped")
                payload = _secure_file_bytes(
                    self.checkpoint_dir / storage,
                    maximum=_MAX_EDIT_BYTES,
                )
                if _sha256(payload) != record.get("sha256"):
                    raise DoctorWorktreeTamperError("checkpoint bytes were tampered")
                mode = int(record.get("mode") or 0)
                self.adapter._write_confined(self.worktree_root, path, payload, mode)
                expected_paths.add(path)
            self.adapter._remove_unexpected(self.worktree_root, expected_paths)
            self.adapter._git(self.worktree_root, "read-tree", self.base_commit_oid)
            if self.target_ref and self.desired_commit_oid:
                live = self.adapter._git_text(
                    self.adapter.repository_root,
                    "rev-parse",
                    "--verify",
                    f"{self.target_ref}^{{commit}}",
                )
                if live == self.desired_commit_oid:
                    rollback = self.adapter._git(
                        self.adapter.repository_root,
                        "update-ref",
                        "--no-deref",
                        self.target_ref,
                        self.base_commit_oid,
                        self.desired_commit_oid,
                        check=False,
                    )
                    ref_restored = rollback.returncode == 0
                    if ref_restored:
                        self._fsync_ref(self.target_ref)
                elif live != self.base_commit_oid:
                    ref_restored = False
            observed = self.adapter.snapshot(self)
            observed_tree = observed.tree_cid
            observed_forest = observed.forest_cid
            gitlinks_equal = observed.gitlinks == expected.gitlinks
            byte_equal = observed.path_hashes == expected.path_hashes
            restored = (
                byte_equal
                and observed.tree_cid == expected.tree_cid
                and observed.forest_cid == expected.forest_cid
                and gitlinks_equal
                and ref_restored
            )
            if not restored:
                raise DoctorWorktreeTamperError(
                    "independent root comparison rejected restoration"
                )
            proof = DoctorRestoreProof(
                session_id=self.session_id,
                restored=True,
                quarantined=False,
                expected_tree_cid=expected.tree_cid,
                observed_tree_cid=observed_tree,
                expected_forest_cid=expected.forest_cid,
                observed_forest_cid=observed_forest,
                ref_restored=True,
                gitlinks_equal=True,
                reason=reason,
            )
            self._journal(
                DoctorWorktreeState.ROLLED_BACK,
                restore_proof=proof.to_dict(),
            )
            return proof
        except BaseException as exc:
            proof = DoctorRestoreProof(
                session_id=self.session_id,
                restored=False,
                quarantined=True,
                expected_tree_cid=expected.tree_cid,
                observed_tree_cid=observed_tree,
                expected_forest_cid=expected.forest_cid,
                observed_forest_cid=observed_forest,
                ref_restored=ref_restored,
                gitlinks_equal=gitlinks_equal,
                reason=f"{reason}:{type(exc).__name__}",
            )
            self.adapter._quarantine(self, proof)
            return proof

    def default_restore(self, _checkpoint: Any = None) -> bool:
        """Transaction restore adapter that independently compares roots."""

        return self.restore().restored

    def close(self, *, remove_worktree: bool = True) -> None:
        if self._closed:
            return
        try:
            if remove_worktree and self.state in {
                DoctorWorktreeState.COMMITTED,
                DoctorWorktreeState.ROLLED_BACK,
            }:
                self.adapter._git(
                    self.adapter.repository_root,
                    "worktree",
                    "remove",
                    "--force",
                    str(self.worktree_root),
                    check=False,
                )
        finally:
            try:
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.lock_stream.close()
                self._closed = True

    def _fsync_ref(self, ref: str) -> None:
        common = Path(
            self.adapter._git_text(
                self.adapter.repository_root, "rev-parse", "--git-common-dir"
            )
        )
        if not common.is_absolute():
            common = (self.adapter.repository_root / common).resolve()
        ref_path = common / ref
        if ref_path.is_file():
            descriptor = os.open(ref_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(ref_path.parent)
        _fsync_directory(common)

    def _require_open(self) -> None:
        if self._closed:
            raise DoctorWorktreeError("worktree session is closed")
        if self.state is DoctorWorktreeState.QUARANTINED:
            raise DoctorWorktreeQuarantineError("worktree session is quarantined")

    def __enter__(self) -> "DoctorWorktreeSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None and not self.state.terminal:
            self.restore(reason=f"context_exit:{getattr(exc_type, '__name__', 'error')}")
        self.close()


@dataclass
class DoctorWorktreeAdapter:
    """Factory and recovery owner for isolated Doctor worktree sessions."""

    repository_root: Path | str
    state_root: Path | str
    permitted_paths: tuple[str, ...]
    permitted_refs: tuple[str, ...] = ()
    git_executable: str = "git"
    max_checkpoint_bytes: int = _MAX_CHECKPOINT_BYTES
    fault_injector: Callable[[str], None] | None = None
    require_clean_base: bool = True

    INTERFACE: Final[str] = DOCTOR_WORKTREE_ADAPTER_INTERFACE

    def __post_init__(self) -> None:
        repository = Path(self.repository_root).resolve(strict=True)
        if not repository.is_dir():
            raise DoctorWorktreeError("repository_root must be a directory")
        _assert_no_symlink_ancestors(repository)
        state = Path(self.state_root).absolute()
        _assert_no_symlink_ancestors(state.parent)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = state.resolve(strict=True)
        if state == repository or repository in state.parents:
            raise DoctorWorktreeSecurityError(
                "state_root must be outside the protected repository"
            )
        object.__setattr__(self, "repository_root", repository)
        object.__setattr__(self, "state_root", state)
        object.__setattr__(
            self,
            "permitted_paths",
            _normal_paths(self.permitted_paths, "permitted_paths"),
        )
        refs = tuple(sorted({self._validate_ref(item) for item in self.permitted_refs}))
        object.__setattr__(self, "permitted_refs", refs)
        if (
            isinstance(self.max_checkpoint_bytes, bool)
            or not isinstance(self.max_checkpoint_bytes, int)
            or self.max_checkpoint_bytes <= 0
            or self.max_checkpoint_bytes > _MAX_CHECKPOINT_BYTES
        ):
            raise DoctorWorktreeError("max_checkpoint_bytes is out of bounds")
        git_path = shutil.which(self.git_executable)
        if git_path is None:
            raise DoctorWorktreeError("git executable is unavailable")
        object.__setattr__(self, "git_executable", git_path)
        top = Path(
            self._git_text(repository, "rev-parse", "--show-toplevel")
        ).resolve(strict=True)
        if top != repository:
            raise DoctorWorktreeSecurityError(
                "repository_root must be the exact Git top-level"
            )
        alternates = self._git_common_dir() / "objects/info/alternates"
        if alternates.exists() and alternates.read_bytes().strip():
            raise DoctorWorktreeSecurityError(
                "Git object alternates escape the allowlisted repository"
            )
        (state / "sessions").mkdir(exist_ok=True, mode=0o700)
        (state / "locks").mkdir(exist_ok=True, mode=0o700)
        (state / "quarantine").mkdir(exist_ok=True, mode=0o700)

    @property
    def allowlisted_paths(self) -> tuple[str, ...]:
        return self.permitted_paths

    def prepare(
        self,
        *,
        base_ref: str = "HEAD",
        session_id: str = "",
    ) -> DoctorWorktreeSession:
        """Acquire the writer lock and create a detached no-checkout worktree."""

        sid = session_id or "doctor-" + hashlib.sha256(
            f"{self.repository_root}:{base_ref}".encode("utf-8")
        ).hexdigest()[:24]
        _safe_component(sid, "session_id")
        if self.require_clean_base:
            dirty = self._git_text(
                self.repository_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            )
            if dirty:
                raise DoctorWorktreeSecurityError(
                    "base repository has tracked dirty bytes"
                )
        base_commit = self._git_text(
            self.repository_root, "rev-parse", "--verify", f"{base_ref}^{{commit}}"
        )
        if not _OID_RE.fullmatch(base_commit):
            raise DoctorWorktreeTamperError("resolved base commit is malformed")
        lock_name = hashlib.sha256(str(self.repository_root).encode("utf-8")).hexdigest()
        lock_path = self.state_root / "locks" / f"{lock_name}.lock"
        lock_stream = lock_path.open("a+b")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise DoctorWorktreeError("writer lease is already held") from exc
        session_dir = self.state_root / "sessions" / sid
        worktree = session_dir / "worktree"
        if session_dir.exists():
            lock_stream.close()
            raise DoctorWorktreeError("session_id already exists")
        session_dir.mkdir(parents=True, mode=0o700)
        try:
            result = self._git(
                self.repository_root,
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(worktree),
                base_commit,
                check=False,
            )
            if result.returncode:
                raise DoctorWorktreeError("git worktree add failed")
            self._git(worktree, "read-tree", base_commit)
            entries = _parse_tree(
                self._git(
                    self.repository_root,
                    "ls-tree",
                    "-rz",
                    "-r",
                    "--full-tree",
                    base_commit,
                ).stdout
            )
            self._materialize(worktree, entries)
            provisional = DoctorWorktreeSession(
                adapter=self,
                session_id=sid,
                base_ref=base_ref,
                base_commit_oid=base_commit,
                worktree_root=worktree,
                session_dir=session_dir,
                lock_stream=lock_stream,
                baseline=DoctorWorktreeSnapshot(
                    session_id=sid,
                    worktree_root=str(worktree),
                    base_commit_oid=base_commit,
                    git_tree_oid="pending",
                    tree_cid="pending",
                    forest_cid="pending",
                    blob_cids=(),
                    path_hashes=(),
                ),
                entries=entries,
                git_admin_hash=self._validate_git_admin(worktree),
            )
            baseline = self.snapshot(provisional)
            provisional.baseline = baseline
            provisional.write_intent()
            return provisional
        except BaseException:
            self._git(
                self.repository_root,
                "worktree",
                "remove",
                "--force",
                str(worktree),
                check=False,
            )
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                lock_stream.close()
            raise

    # Common factory spellings.
    create = prepare
    create_worktree = prepare

    def snapshot(self, session: DoctorWorktreeSession) -> DoctorWorktreeSnapshot:
        """Reread exact bytes and compute raw blob plus DAG tree/forest CIDs."""

        self._scan_worktree(session, expected_new_paths=())
        baseline_entries = {item.path: item for item in session.entries}
        index_entries = {
            item.path: item
            for item in _parse_index(
                self._git(
                    session.worktree_root,
                    "ls-files",
                    "--stage",
                    "-z",
                ).stdout
            )
        }
        if set(index_entries) != set(baseline_entries):
            raise DoctorWorktreeTamperError(
                "Git index path population changed outside the exact edit contract"
            )
        for path, baseline_entry in baseline_entries.items():
            indexed = index_entries[path]
            if baseline_entry.mode == 0o160000:
                if (
                    indexed.mode != 0o160000
                    or indexed.oid != baseline_entry.oid
                ):
                    raise DoctorWorktreeTamperError(
                        f"gitlink identity changed for {path}"
                    )
            elif indexed.mode not in {0o100644, 0o100755}:
                raise DoctorWorktreeTamperError(
                    f"regular index entry changed type for {path}"
                )
        blobs: list[tuple[str, str]] = []
        hashes: list[tuple[str, str]] = []
        tree_records: list[dict[str, Any]] = []
        gitlinks: list[DoctorGitlink] = []
        for path in sorted(index_entries):
            entry = index_entries[path]
            if entry.object_type == "commit" and entry.mode == 0o160000:
                link = DoctorGitlink(path=path, commit_oid=entry.oid)
                gitlinks.append(link)
                tree_records.append(
                    {"path": path, "mode": entry.mode, "gitlink": entry.oid}
                )
                continue
            if entry.object_type != "blob" or entry.mode not in {0o100644, 0o100755}:
                raise DoctorWorktreeSecurityError(
                    f"unsupported tracked entry {path} mode={entry.mode:o}"
                )
            body = self._read_confined(session.worktree_root, path)
            info = self._lstat_confined(session.worktree_root, path)
            mode = 0o100755 if info.st_mode & stat.S_IXUSR else 0o100644
            blob_cid = cid_for_bytes(body)
            digest = _sha256(body)
            blobs.append((path, blob_cid))
            hashes.append((path, digest))
            tree_records.append(
                {
                    "path": path,
                    "mode": mode,
                    "blob_cid": blob_cid,
                    "sha256": digest,
                    "byte_count": len(body),
                }
            )
        tree_cid = cid_for_dag_json(
            {"kind": "doctor-tree", "entries": tree_records}, for_identity=True
        )
        forest_cid = cid_for_dag_json(
            {
                "kind": "doctor-forest",
                "root_tree_cid": tree_cid,
                "gitlinks": [item.to_dict() for item in gitlinks],
            },
            for_identity=True,
        )
        git_tree_oid = self._git_text(
            self.repository_root,
            "rev-parse",
            "--verify",
            f"{session.base_commit_oid}^{{tree}}",
        )
        return DoctorWorktreeSnapshot(
            session_id=session.session_id,
            worktree_root=str(session.worktree_root),
            base_commit_oid=session.base_commit_oid,
            git_tree_oid=git_tree_oid,
            tree_cid=tree_cid,
            forest_cid=forest_cid,
            blob_cids=tuple(blobs),
            path_hashes=tuple(hashes),
            gitlinks=tuple(gitlinks),
        )

    def recover_incomplete(self) -> tuple[DoctorRestoreProof, ...]:
        """Recover every durable nonterminal session, quarantining uncertainty."""

        proofs: list[DoctorRestoreProof] = []
        for journal in sorted((self.state_root / "sessions").glob("*/intent.json")):
            try:
                record = json.loads(
                    _secure_file_bytes(
                        journal, maximum=16 * 1024 * 1024
                    ).decode("utf-8", "strict")
                )
                state = DoctorWorktreeState(str(record["state"]))
                if state.terminal:
                    continue
                proofs.append(self._recover_record(journal.parent, record))
            except BaseException as exc:
                sid = journal.parent.name
                proof = DoctorRestoreProof(
                    session_id=sid,
                    restored=False,
                    quarantined=True,
                    expected_tree_cid="unknown",
                    observed_tree_cid="",
                    expected_forest_cid="unknown",
                    observed_forest_cid="",
                    ref_restored=False,
                    gitlinks_equal=False,
                    reason=f"recovery_manifest:{type(exc).__name__}",
                )
                _atomic_json(
                    self.state_root / "quarantine" / f"{sid}.json",
                    {
                        "schema": DOCTOR_WORKTREE_QUARANTINE_SCHEMA,
                        "proof": proof.to_dict(),
                    },
                )
                proofs.append(proof)
        return tuple(proofs)

    recover = recover_incomplete

    def hash_matches(self, body: bytes, claimed: str) -> bool:
        if claimed == _sha256(body):
            return True
        if claimed == cid_for_bytes(body):
            return True
        if _OID_RE.fullmatch(claimed):
            return claimed == self._hash_object(body, write=False)
        return False

    def _recover_record(
        self, session_dir: Path, record: Mapping[str, Any]
    ) -> DoctorRestoreProof:
        sid = _safe_component(str(record["session_id"]), "session_id")
        lock_name = hashlib.sha256(str(self.repository_root).encode("utf-8")).hexdigest()
        lock_stream = (self.state_root / "locks" / f"{lock_name}.lock").open("a+b")
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise DoctorWorktreeError("cannot recover while writer lease is held") from exc
        baseline_raw = record.get("baseline")
        if not isinstance(baseline_raw, Mapping):
            lock_stream.close()
            raise DoctorWorktreeTamperError("recovery baseline is missing")
        entries = _parse_tree(
            self._git(
                self.repository_root,
                "ls-tree",
                "-rz",
                "-r",
                "--full-tree",
                str(record["base_commit_oid"]),
            ).stdout
        )
        baseline = DoctorWorktreeSnapshot(
            session_id=sid,
            worktree_root=str(record["worktree_root"]),
            base_commit_oid=str(record["base_commit_oid"]),
            git_tree_oid=str(baseline_raw["git_tree_oid"]),
            tree_cid=str(baseline_raw["tree_cid"]),
            forest_cid=str(baseline_raw["forest_cid"]),
            blob_cids=tuple(
                (str(item["path"]), str(item["cid"]))
                for item in baseline_raw.get("blob_cids", ())
            ),
            path_hashes=tuple(
                (str(item["path"]), str(item["sha256"]))
                for item in baseline_raw.get("path_hashes", ())
            ),
            gitlinks=tuple(
                DoctorGitlink(str(item["path"]), str(item["commit_oid"]))
                for item in baseline_raw.get("gitlinks", ())
            ),
        )
        session = DoctorWorktreeSession(
            adapter=self,
            session_id=sid,
            base_ref=str(record["base_ref"]),
            base_commit_oid=str(record["base_commit_oid"]),
            worktree_root=Path(str(record["worktree_root"])),
            session_dir=session_dir,
            lock_stream=lock_stream,
            baseline=baseline,
            entries=entries,
            git_admin_hash=str(record.get("git_admin_hash") or ""),
            state=DoctorWorktreeState(str(record["state"])),
            target_ref=str(record.get("target_ref") or ""),
            desired_commit_oid=str(record.get("desired_commit_oid") or ""),
        )
        proof = session.restore(reason="startup_recovery")
        session.close()
        return proof

    def _git_common_dir(self) -> Path:
        raw = self._git_text(
            self.repository_root, "rev-parse", "--git-common-dir"
        )
        common = Path(raw)
        if not common.is_absolute():
            common = self.repository_root / common
        return common.resolve(strict=True)

    def _validate_git_admin(
        self, worktree: Path, expected_hash: str = ""
    ) -> str:
        """Bind the trusted worktree administrative pointer on every scan."""

        admin = worktree / ".git"
        info = admin.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DoctorWorktreeSecurityError(
                "worktree .git pointer must be one regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(admin, flags)
        try:
            body = os.read(descriptor, 16_384)
            if os.read(descriptor, 1):
                raise DoctorWorktreeSecurityError(
                    "worktree .git pointer exceeds its bound"
                )
        finally:
            os.close(descriptor)
        try:
            text = body.decode("utf-8", "strict").strip()
        except UnicodeError as exc:
            raise DoctorWorktreeSecurityError(
                "worktree .git pointer is not UTF-8"
            ) from exc
        if not text.startswith("gitdir: "):
            raise DoctorWorktreeSecurityError("worktree .git pointer is malformed")
        target = Path(text.removeprefix("gitdir: ").strip())
        if not target.is_absolute():
            target = admin.parent / target
        target = target.resolve(strict=True)
        common = self._git_common_dir()
        if common not in {target, *target.parents}:
            raise DoctorWorktreeSecurityError(
                "worktree .git pointer escaped the repository common dir"
            )
        digest = _sha256(body)
        if expected_hash and digest != expected_hash:
            raise DoctorWorktreeTamperError("worktree .git pointer changed")
        return digest

    def _validate_ref(self, ref: str) -> str:
        if not isinstance(ref, str) or not _REF_RE.fullmatch(ref):
            raise DoctorWorktreeSecurityError("target ref is outside the ref allowlist")
        if ".." in ref or "//" in ref or ref.endswith(("/", ".", ".lock")):
            raise DoctorWorktreeSecurityError("target ref is malformed")
        if self.permitted_refs and ref not in self.permitted_refs:
            raise DoctorWorktreeSecurityError("target ref was not explicitly permitted")
        return ref

    def _git_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
            "HOME": str(self.state_root),
            "XDG_CONFIG_HOME": str(self.state_root / "xdg-config"),
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "/bin/false",
            "SSH_ASKPASS": "/bin/false",
            "GCM_INTERACTIVE": "Never",
            "GIT_ALLOW_PROTOCOL": "",
        }
        if extra:
            for key, value in extra.items():
                if key.startswith("GIT_") and isinstance(value, str):
                    environment[key] = value
        return environment

    def _git(
        self,
        cwd: Path,
        *arguments: str,
        check: bool = True,
        input_bytes: bytes | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        argv = [
            self.git_executable,
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.allow=never",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "core.fsync=committed",
            "-C",
            str(cwd),
            *arguments,
        ]
        result = subprocess.run(
            argv,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._git_env(extra_env),
            cwd="/",
            timeout=120,
            close_fds=True,
            start_new_session=True,
            check=False,
        )
        if check and result.returncode:
            raise DoctorWorktreeError(
                "allowlisted local Git command failed: "
                + result.stderr.decode("utf-8", "replace")[:500]
            )
        return result

    def _git_text(
        self,
        cwd: Path,
        *arguments: str,
        extra_env: Mapping[str, str] | None = None,
    ) -> str:
        return self._git(cwd, *arguments, extra_env=extra_env).stdout.decode(
            "utf-8", "strict"
        ).strip()

    def _hash_object(self, body: bytes, *, write: bool) -> str:
        args = ["hash-object"]
        if write:
            args.append("-w")
        args.append("--stdin")
        oid = self._git(
            self.repository_root, *args, input_bytes=body
        ).stdout.decode("ascii").strip()
        if not _OID_RE.fullmatch(oid):
            raise DoctorWorktreeTamperError("git hash-object returned a malformed oid")
        return oid

    def _materialize(
        self, worktree: Path, entries: Sequence[_TreeEntry]
    ) -> None:
        total = 0
        for entry in entries:
            if entry.object_type == "commit" and entry.mode == 0o160000:
                continue
            if entry.object_type != "blob":
                raise DoctorWorktreeSecurityError(
                    f"unsupported Git object type at {entry.path}"
                )
            if entry.mode == 0o120000:
                raise DoctorWorktreeSecurityError(
                    f"tracked symlink is forbidden: {entry.path}"
                )
            if entry.mode not in {0o100644, 0o100755}:
                raise DoctorWorktreeSecurityError(
                    f"special tracked mode is forbidden: {entry.path}"
                )
            body = self._git(
                self.repository_root, "cat-file", "blob", entry.oid
            ).stdout
            total += len(body)
            if total > self.max_checkpoint_bytes:
                raise DoctorWorktreeError("materialized worktree exceeds byte bound")
            self._write_confined(worktree, entry.path, body, entry.mode)

    def _lstat_confined(self, root: Path, relative: str) -> os.stat_result:
        path = root / _repository_path(relative)
        resolved_parent = path.parent.resolve(strict=True)
        if root.resolve(strict=True) not in {
            resolved_parent,
            *resolved_parent.parents,
        }:
            raise DoctorWorktreeSecurityError("path parent escaped worktree")
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise DoctorWorktreeSecurityError("symlink path is forbidden")
        if not stat.S_ISREG(info.st_mode):
            raise DoctorWorktreeSecurityError("non-regular path is forbidden")
        if info.st_nlink != 1:
            raise DoctorWorktreeSecurityError("hardlinked path is forbidden")
        return info

    def _read_confined(self, root: Path, relative: str) -> bytes:
        path = root / _repository_path(relative)
        self._lstat_confined(root, relative)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise DoctorWorktreeSecurityError("opened path changed type/link count")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > _MAX_EDIT_BYTES:
                    raise DoctorWorktreeError("tracked file exceeds byte bound")
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise DoctorWorktreeTamperError("path changed while being reread")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _write_confined(
        self, root: Path, relative: str, payload: bytes, mode: int
    ) -> None:
        path = root / _repository_path(relative)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        resolved_root = root.resolve(strict=True)
        resolved_parent = parent.resolve(strict=True)
        if resolved_root not in {resolved_parent, *resolved_parent.parents}:
            raise DoctorWorktreeSecurityError("write parent escaped worktree")
        _assert_no_symlink_ancestors(parent, stop=resolved_root)
        if path.exists():
            self._lstat_confined(root, relative)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".doctor-write-", dir=parent
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o755 if mode == 0o100755 else 0o644)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(parent)
            reread = self._read_confined(root, relative)
            if reread != payload:
                raise DoctorWorktreeTamperError("atomic write reread mismatch")
        except BaseException:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _scan_worktree(
        self,
        session: DoctorWorktreeSession,
        *,
        expected_new_paths: Sequence[str],
    ) -> None:
        self._validate_git_admin(
            session.worktree_root, expected_hash=session.git_admin_hash
        )
        allowed = {
            item.path for item in session.entries if item.object_type == "blob"
        } | set(expected_new_paths)
        root = session.worktree_root
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            clean_dirs: list[str] = []
            for name in dirnames:
                child = directory_path / name
                info = child.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise DoctorWorktreeSecurityError(
                        f"symlink directory is forbidden: {child}"
                    )
                if not stat.S_ISDIR(info.st_mode):
                    raise DoctorWorktreeSecurityError(
                        f"special directory entry is forbidden: {child}"
                    )
                if child == root / ".git":
                    continue
                clean_dirs.append(name)
            dirnames[:] = clean_dirs
            for name in filenames:
                child = directory_path / name
                if child == root / ".git":
                    continue
                relative = child.relative_to(root).as_posix()
                _repository_path(relative)
                info = child.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise DoctorWorktreeSecurityError(
                        f"hostile filesystem object: {relative}"
                    )
                if relative not in allowed:
                    raise DoctorWorktreeSecurityError(
                        f"unexpected/unallowlisted worktree path: {relative}"
                    )
        for path in allowed:
            self._lstat_confined(root, path)

    def _remove_unexpected(self, root: Path, expected_paths: set[str]) -> None:
        for directory, dirnames, filenames in os.walk(root, topdown=False):
            directory_path = Path(directory)
            for name in filenames:
                child = directory_path / name
                if child == root / ".git":
                    continue
                relative = child.relative_to(root).as_posix()
                if relative not in expected_paths:
                    info = child.lstat()
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                        raise DoctorWorktreeSecurityError(
                            "cannot safely remove hostile unexpected object"
                        )
                    child.unlink()
                    _fsync_directory(child.parent)
            for name in dirnames:
                child = directory_path / name
                if child == root / ".git":
                    continue
                try:
                    child.rmdir()
                except OSError:
                    pass

    def _quarantine(
        self, session: DoctorWorktreeSession, proof: DoctorRestoreProof
    ) -> None:
        session.state = DoctorWorktreeState.QUARANTINED
        record = {
            "schema": DOCTOR_WORKTREE_QUARANTINE_SCHEMA,
            "interface": DOCTOR_WORKTREE_ADAPTER_INTERFACE,
            "session_id": session.session_id,
            "repository_root": str(self.repository_root),
            "worktree_root": str(session.worktree_root),
            "proof": proof.to_dict(),
        }
        _atomic_json(
            self.state_root / "quarantine" / f"{session.session_id}.json",
            record,
        )
        try:
            session._journal(
                DoctorWorktreeState.QUARANTINED, quarantine=record
            )
        except BaseException:
            pass


__all__ = [
    "DOCTOR_WORKTREE_ADAPTER_INTERFACE",
    "DoctorBlobEffect",
    "DoctorExactEdit",
    "DoctorFileEdit",
    "DoctorGitlink",
    "DoctorRefCasReceipt",
    "DoctorRestoreProof",
    "DoctorWorktreeAdapter",
    "DoctorWorktreeApplyReceipt",
    "DoctorWorktreeCasError",
    "DoctorWorktreeEdit",
    "DoctorWorktreeError",
    "DoctorWorktreeQuarantineError",
    "DoctorWorktreeSecurityError",
    "DoctorWorktreeSession",
    "DoctorWorktreeSnapshot",
    "DoctorWorktreeState",
    "DoctorWorktreeTamperError",
]
