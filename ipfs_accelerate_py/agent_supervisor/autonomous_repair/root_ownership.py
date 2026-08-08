"""DCR-003 fail-closed ownership and submodule-pin admission.

The repair engine can reason across a repository forest, but it cannot turn a
path, a dirty worktree, or a submodule move into authority by itself.  This
module binds every admission to real paths and to the exact Git head, tree,
and dirty-overlay fingerprint observed at admission time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

REPAIR_ROOT_OWNERSHIP_INTERFACE: Final[str] = "RepairRootOwnership@1"
SUBMODULE_PIN_ADMISSION_INTERFACE: Final[str] = "SubmodulePinAdmission@1"
REPAIR_ROOTS_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1"
ROOT_OWNERSHIP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-root-ownership-receipt@1"
)
SUBMODULE_PIN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/submodule-pin-admission-receipt@1"
)
_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40,64}$")
_ROOT_ROLES: Final[frozenset[str]] = frozenset({"consumer", "provider", "orchestration_only"})


class RootOwnershipDenied(PermissionError):
    """A path, binding, or pin request is outside the deterministic policy."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _receipt_id(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_relative(value: object, *, field: str, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\x00" in normalized
        or (path.parts and path.parts[0].endswith(":"))
        or (path.as_posix() == "." and not allow_dot)
    ):
        raise ValueError(f"{field} is unsafe")
    return path.as_posix()


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RootOwnershipDenied(f"root {root} is not an admissible Git worktree: {detail}")
    return result.stdout


def _overlay_fingerprint(root: Path) -> tuple[str, bool]:
    status = _git(root, "status", "--porcelain=v1", "-z")
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"diff\0")
    digest.update(_git(root, "diff", "--no-ext-diff", "--binary", "HEAD"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    _fingerprint_overlay_entries(digest, root, untracked, label=b"untracked\0")
    # Ignored files can still alter a source/config lookup at runtime.  They
    # therefore participate in the binding even though Git does not call the
    # worktree dirty.  A later policy may exclude reviewed cache roots, but an
    # undeclared ignore rule is never an identity bypass.
    _fingerprint_overlay_entries(digest, root, ignored, label=b"ignored\0")
    return "sha256:" + digest.hexdigest(), bool(status)


def _fingerprint_overlay_entries(
    digest: hashlib._Hash,
    root: Path,
    encoded_paths: bytes,
    *,
    label: bytes,
) -> None:
    for raw_path in sorted(part for part in encoded_paths.split(b"\0") if part):
        relative = raw_path.decode("utf-8", "surrogateescape")
        raw_candidate = root / relative
        candidate = raw_candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RootOwnershipDenied("overlay path escapes root") from exc
        digest.update(label)
        digest.update(raw_path)
        try:
            stat_result = raw_candidate.lstat()
        except OSError as exc:
            raise RootOwnershipDenied("overlay entry cannot be inspected") from exc
        digest.update(b"mode\0")
        digest.update(str(stat_result.st_mode).encode("ascii"))
        if raw_candidate.is_symlink():
            digest.update(b"symlink\0")
            # os.readlink is available on every supported Python version;
            # pathlib.Path.readlink would silently raise the floor to 3.9.
            digest.update(os.readlink(raw_candidate).encode("utf-8", "surrogateescape"))
        elif raw_candidate.is_file():
            digest.update(b"file\0")
            try:
                with raw_candidate.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
            except OSError as exc:
                raise RootOwnershipDenied("overlay file cannot be read") from exc
        else:
            raise RootOwnershipDenied("overlay entry is not a regular file or symlink")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class RootBinding:
    """Current immutable identity of one owned Git worktree."""

    root_id: str
    realpath: str
    head: str
    tree: str
    overlay_digest: str
    dirty: bool

    def __post_init__(self) -> None:
        if not self.root_id or not self.realpath:
            raise ValueError("root binding requires root_id and realpath")
        for field in ("head", "tree"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
                raise ValueError(f"root binding {field} must be a Git object id")
        if not isinstance(self.overlay_digest, str) or not self.overlay_digest:
            raise ValueError("root binding overlay_digest must be non-empty")
        if not isinstance(self.dirty, bool):
            raise ValueError("root binding dirty must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dirty": self.dirty,
            "head": self.head,
            "overlay_digest": self.overlay_digest,
            "realpath": self.realpath,
            "root_id": self.root_id,
            "tree": self.tree,
        }


@dataclass(frozen=True)
class RepairRoot:
    """One named ownership boundary from the reviewed policy document."""

    root_id: str
    relative_path: str
    role: str
    allowed_write_prefixes: tuple[str, ...]
    pin_path: str

    @property
    def writable(self) -> bool:
        return self.role != "orchestration_only" and bool(self.allowed_write_prefixes)


@dataclass(frozen=True)
class RootOwnershipReceipt:
    """Canonical proof that a single bound root owns every admitted write."""

    claimed_root_id: str
    defect_root_id: str
    write_paths: tuple[str, ...]
    bindings: tuple[RootBinding, ...]
    receipt_id: str

    def to_dict(self) -> dict[str, Any]:
        body = {
            "bindings": [binding.to_dict() for binding in self.bindings],
            "claimed_root_id": self.claimed_root_id,
            "defect_root_id": self.defect_root_id,
            "interface": REPAIR_ROOT_OWNERSHIP_INTERFACE,
            "schema": ROOT_OWNERSHIP_RECEIPT_SCHEMA,
            "write_paths": list(self.write_paths),
        }
        return {**body, "receipt_id": self.receipt_id}


@dataclass(frozen=True)
class SubmodulePinReceipt:
    """Canonical proof that a parent Gitlink can advance to one child head."""

    target_root_id: str
    pin_path: str
    predecessor: str
    successor: str
    root_receipt_id: str
    validation_receipt_id: str
    receipt_id: str

    def to_dict(self) -> dict[str, Any]:
        body = {
            "interface": SUBMODULE_PIN_ADMISSION_INTERFACE,
            "pin_path": self.pin_path,
            "predecessor": self.predecessor,
            "root_receipt_id": self.root_receipt_id,
            "schema": SUBMODULE_PIN_RECEIPT_SCHEMA,
            "successor": self.successor,
            "target_root_id": self.target_root_id,
            "validation_receipt_id": self.validation_receipt_id,
        }
        return {**body, "receipt_id": self.receipt_id}


class RepairRootOwnership:
    """Resolve real-path owners and admit only fully bound, single-root writes."""

    INTERFACE: Final[str] = REPAIR_ROOT_OWNERSHIP_INTERFACE

    def __init__(self, *, workspace_root: Path | str, roots: Sequence[RepairRoot]) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        self._roots = tuple(roots)
        self._by_id = {root.root_id: root for root in self._roots}
        if len(self._roots) != len(self._by_id) or not self._roots:
            raise ValueError("repair roots must have unique non-empty ids")
        if set(self._by_id) != {
            "orchestration",
            "swissknife",
            "mcp-plus-plus",
            "ipfs-accelerate",
            "ipfs-datasets",
            "ipfs-kit",
        }:
            raise ValueError("repair roots must declare the exact DCR-003 root set")
        orchestration = self._by_id["orchestration"]
        if orchestration.role != "orchestration_only" or orchestration.writable:
            raise ValueError("orchestration root must be write-disabled")
        self._real_roots: dict[str, Path] = {}
        for root in self._roots:
            candidate = (self.workspace_root / root.relative_path).resolve()
            try:
                candidate.relative_to(self.workspace_root)
            except ValueError as exc:
                raise ValueError(f"root {root.root_id} escapes workspace") from exc
            if not candidate.is_dir():
                raise ValueError(f"root {root.root_id} is not an existing directory")
            self._real_roots[root.root_id] = candidate
        non_orchestration = [
            self._real_roots[root.root_id]
            for root in self._roots
            if root.role != "orchestration_only"
        ]
        for index, first in enumerate(non_orchestration):
            for second in non_orchestration[index + 1 :]:
                if _contains(first, second) or _contains(second, first):
                    raise ValueError("non-orchestration repair roots may not overlap")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        workspace_root: Path | str,
    ) -> RepairRootOwnership:
        if not isinstance(raw, Mapping):
            raise ValueError("root policy must be an object")
        expected_policy_fields = {"schema", "interface", "roots"}
        if set(raw) != expected_policy_fields:
            drift = sorted(set(raw).symmetric_difference(expected_policy_fields))
            raise ValueError(
                "root policy fields must match the reviewed schema exactly: " + ",".join(drift)
            )
        if raw.get("schema") != REPAIR_ROOTS_SCHEMA:
            raise ValueError("unsupported repair root policy schema")
        if raw.get("interface") != REPAIR_ROOT_OWNERSHIP_INTERFACE:
            raise ValueError("unsupported repair root policy interface")
        records = raw.get("roots")
        if not isinstance(records, list):
            raise ValueError("root policy roots must be a list")
        roots: list[RepairRoot] = []
        expected_root_fields = {
            "id",
            "relative_path",
            "role",
            "allowed_write_prefixes",
            "pin_path",
        }
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"roots[{index}] must be an object")
            if set(record) != expected_root_fields:
                drift = sorted(set(record).symmetric_difference(expected_root_fields))
                raise ValueError(
                    f"roots[{index}] fields must match the reviewed schema exactly: "
                    + ",".join(drift)
                )
            root_id = record.get("id")
            role = record.get("role")
            if not isinstance(root_id, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", root_id):
                raise ValueError(f"roots[{index}].id is unsafe")
            if role not in _ROOT_ROLES:
                raise ValueError(f"roots[{index}].role is unsupported")
            prefixes = record.get("allowed_write_prefixes")
            if not isinstance(prefixes, list):
                raise ValueError(f"roots[{index}].allowed_write_prefixes must be a list")
            normalized_prefixes = tuple(
                _safe_relative(prefix, field=f"roots[{index}].allowed_write_prefixes")
                if prefix != "."
                else "."
                for prefix in prefixes
            )
            if len(normalized_prefixes) != len(set(normalized_prefixes)):
                raise ValueError(f"roots[{index}] has duplicate write prefixes")
            if role == "orchestration_only" and normalized_prefixes:
                raise ValueError("orchestration root cannot admit ordinary writes")
            pin_path = record.get("pin_path")
            if not isinstance(pin_path, str):
                raise ValueError(f"roots[{index}].pin_path must be text")
            roots.append(
                RepairRoot(
                    root_id=root_id,
                    relative_path=_safe_relative(
                        record.get("relative_path"),
                        field=f"roots[{index}].relative_path",
                        allow_dot=True,
                    ),
                    role=role,
                    allowed_write_prefixes=normalized_prefixes,
                    pin_path=(
                        _safe_relative(pin_path, field=f"roots[{index}].pin_path")
                        if pin_path
                        else ""
                    ),
                )
            )
        return cls(workspace_root=workspace_root, roots=roots)

    @classmethod
    def from_file(
        cls,
        policy_path: Path | str,
        *,
        workspace_root: Path | str,
    ) -> RepairRootOwnership:
        try:
            raw = json.loads(
                Path(policy_path).read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("repair root policy is unreadable") from exc
        return cls.from_mapping(raw, workspace_root=workspace_root)

    load = from_file

    @property
    def roots(self) -> tuple[RepairRoot, ...]:
        return self._roots

    def root_path(self, root_id: str) -> Path:
        self._root(root_id)
        return self._real_roots[root_id]

    def capture_binding(self, root_id: str) -> RootBinding:
        root = self._root(root_id)
        realpath = self._real_roots[root.root_id]
        head = _git(realpath, "rev-parse", "HEAD").decode().strip().lower()
        tree = _git(realpath, "rev-parse", "HEAD^{tree}").decode().strip().lower()
        overlay_digest, dirty = _overlay_fingerprint(realpath)
        return RootBinding(
            root_id=root.root_id,
            realpath=str(realpath),
            head=head,
            tree=tree,
            overlay_digest=overlay_digest,
            dirty=dirty,
        )

    def capture_bindings(self) -> dict[str, RootBinding]:
        return {root.root_id: self.capture_binding(root.root_id) for root in self._roots}

    def owner_for(self, path: Path | str) -> RepairRoot:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        realpath = candidate.resolve(strict=False)
        try:
            realpath.relative_to(self.workspace_root)
        except ValueError as exc:
            raise RootOwnershipDenied("write path escapes workspace realpath") from exc
        owners = [
            root
            for root in self._roots
            if root.role != "orchestration_only"
            and _contains(self._real_roots[root.root_id], realpath)
        ]
        if len(owners) == 1:
            return owners[0]
        if len(owners) > 1:
            raise RootOwnershipDenied("write path has ambiguous root owner")
        if realpath == self.workspace_root:
            return self._by_id["orchestration"]
        raise RootOwnershipDenied("write path has no declared root owner")

    resolve_owner = owner_for

    def admit_write(
        self,
        write_paths: Sequence[Path | str],
        *,
        claimed_root_id: str,
        bindings: Mapping[str, RootBinding],
        defect_root_id: str | None = None,
    ) -> RootOwnershipReceipt:
        target = self._root(claimed_root_id)
        if not target.writable:
            raise RootOwnershipDenied("orchestration-only root cannot admit writes")
        if not isinstance(write_paths, Sequence) or isinstance(write_paths, (str, bytes)):
            raise RootOwnershipDenied("write_paths must be a non-empty sequence")
        if not write_paths:
            raise RootOwnershipDenied("write_paths must be a non-empty sequence")
        checked_bindings = self._verify_bindings(bindings)
        defect_id = defect_root_id or claimed_root_id
        defect = self._root(defect_id)
        if defect.root_id != target.root_id:
            if defect.role == "provider" and target.role == "consumer":
                raise RootOwnershipDenied(
                    "provider defect cannot be repaired by consumer weakening"
                )
            raise RootOwnershipDenied("cross-root repair writes are not admitted")
        canonical_paths: list[str] = []
        seen: set[str] = set()
        for raw_path in write_paths:
            owner = self.owner_for(raw_path)
            if owner.root_id != target.root_id:
                raise RootOwnershipDenied("cross-root write path is not admitted")
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            realpath = candidate.resolve(strict=False)
            root_path = self._real_roots[target.root_id]
            try:
                relative = realpath.relative_to(root_path).as_posix()
            except ValueError as exc:
                raise RootOwnershipDenied("write path escapes declared root") from exc
            if not self._write_prefix_admits(target, relative):
                raise RootOwnershipDenied("write path is outside declared write scope")
            workspace_relative = realpath.relative_to(self.workspace_root).as_posix()
            if workspace_relative in seen:
                raise RootOwnershipDenied("duplicate write paths are not admitted")
            seen.add(workspace_relative)
            canonical_paths.append(workspace_relative)
        ordered_bindings = tuple(checked_bindings[root_id] for root_id in sorted(checked_bindings))
        body = {
            "bindings": [binding.to_dict() for binding in ordered_bindings],
            "claimed_root_id": target.root_id,
            "defect_root_id": defect.root_id,
            "interface": REPAIR_ROOT_OWNERSHIP_INTERFACE,
            "schema": ROOT_OWNERSHIP_RECEIPT_SCHEMA,
            "write_paths": sorted(canonical_paths),
        }
        return RootOwnershipReceipt(
            claimed_root_id=target.root_id,
            defect_root_id=defect.root_id,
            write_paths=tuple(body["write_paths"]),
            bindings=ordered_bindings,
            receipt_id=_receipt_id(body),
        )

    admit = admit_write

    def _root(self, root_id: str) -> RepairRoot:
        if not isinstance(root_id, str) or root_id not in self._by_id:
            raise RootOwnershipDenied("unknown repair root")
        return self._by_id[root_id]

    def _verify_bindings(self, bindings: Mapping[str, RootBinding]) -> dict[str, RootBinding]:
        if not isinstance(bindings, Mapping):
            raise RootOwnershipDenied("root bindings must be a mapping")
        expected_ids = set(self._by_id)
        actual_ids = set(bindings)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            raise RootOwnershipDenied("unbound dirty or unknown roots: " + ",".join(missing))
        checked: dict[str, RootBinding] = {}
        changed: list[str] = []
        for root_id in sorted(expected_ids):
            binding = bindings[root_id]
            if not isinstance(binding, RootBinding) or binding.root_id != root_id:
                raise RootOwnershipDenied("root binding type or id mismatch")
            current = self.capture_binding(root_id)
            if binding != current:
                changed.append(root_id)
            checked[root_id] = binding
        if changed:
            raise RootOwnershipDenied("changed roots: " + ",".join(changed))
        return checked

    def _write_prefix_admits(self, root: RepairRoot, relative: str) -> bool:
        path = PurePosixPath(relative)
        for prefix in root.allowed_write_prefixes:
            if (
                prefix == "."
                or path == PurePosixPath(prefix)
                or PurePosixPath(prefix) in path.parents
            ):
                return True
        return False


class SubmodulePinAdmission:
    """Admit a Gitlink advance only after the target root is bound and proved."""

    INTERFACE: Final[str] = SUBMODULE_PIN_ADMISSION_INTERFACE

    def __init__(self, ownership: RepairRootOwnership) -> None:
        if not isinstance(ownership, RepairRootOwnership):
            raise TypeError("ownership must be RepairRootOwnership")
        self.ownership = ownership

    def admit_pin_update(
        self,
        *,
        target_root_id: str,
        predecessor: str,
        successor: str,
        bindings: Mapping[str, RootBinding],
        root_receipt: RootOwnershipReceipt,
        validation_receipt: Mapping[str, Any],
        changed_root_ids: Sequence[str],
    ) -> SubmodulePinReceipt:
        target = self.ownership._root(target_root_id)
        if target.role == "orchestration_only" or not target.pin_path:
            raise RootOwnershipDenied("target root has no admissible submodule pin")
        predecessor = self._commit(predecessor, field="predecessor")
        successor = self._commit(successor, field="successor")
        if predecessor == successor:
            raise RootOwnershipDenied("submodule pin successor must advance")
        if tuple(changed_root_ids) != (target.root_id,):
            raise RootOwnershipDenied("submodule pin must name exactly its changed root")
        checked = self.ownership._verify_bindings(bindings)
        target_binding = checked[target.root_id]
        if target_binding.head != successor or target_binding.dirty:
            raise RootOwnershipDenied("premature pin: successor is not a clean bound target head")
        if not isinstance(root_receipt, RootOwnershipReceipt):
            raise RootOwnershipDenied("premature pin: missing canonical root receipt")
        if root_receipt.claimed_root_id != target.root_id:
            raise RootOwnershipDenied("premature pin: root receipt targets another root")
        receipt_binding = next(
            (binding for binding in root_receipt.bindings if binding.root_id == target.root_id),
            None,
        )
        if receipt_binding != target_binding:
            raise RootOwnershipDenied("premature pin: root receipt binding changed")
        try:
            expected_root_receipt = self.ownership.admit_write(
                root_receipt.write_paths,
                claimed_root_id=root_receipt.claimed_root_id,
                defect_root_id=root_receipt.defect_root_id,
                bindings=checked,
            )
        except (RootOwnershipDenied, ValueError) as exc:
            raise RootOwnershipDenied(
                "premature pin: root receipt is not currently admissible"
            ) from exc
        if root_receipt != expected_root_receipt:
            raise RootOwnershipDenied("premature pin: root receipt is incomplete or non-canonical")
        if not isinstance(validation_receipt, Mapping):
            raise RootOwnershipDenied("premature pin: missing validation receipt")
        validation = dict(validation_receipt)
        if (
            validation.get("passed") is not True
            or validation.get("root_id") != target.root_id
            or validation.get("head") != successor
        ):
            raise RootOwnershipDenied("premature pin: target validation is insufficient")
        validation_body = {key: value for key, value in validation.items() if key != "receipt_id"}
        validation_id = validation.get("receipt_id")
        if not isinstance(validation_id, str) or validation_id != _receipt_id(validation_body):
            raise RootOwnershipDenied("premature pin: validation receipt is not canonical")
        current_pin, indexed_pin = self._current_gitlink(target.pin_path)
        if current_pin != predecessor or indexed_pin != predecessor:
            raise RootOwnershipDenied("submodule pin predecessor does not match parent tree")
        body = {
            "interface": SUBMODULE_PIN_ADMISSION_INTERFACE,
            "pin_path": target.pin_path,
            "predecessor": predecessor,
            "root_receipt_id": root_receipt.receipt_id,
            "schema": SUBMODULE_PIN_RECEIPT_SCHEMA,
            "successor": successor,
            "target_root_id": target.root_id,
            "validation_receipt_id": validation_id,
        }
        return SubmodulePinReceipt(
            target_root_id=target.root_id,
            pin_path=target.pin_path,
            predecessor=predecessor,
            successor=successor,
            root_receipt_id=root_receipt.receipt_id,
            validation_receipt_id=validation_id,
            receipt_id=_receipt_id(body),
        )

    admit = admit_pin_update

    def _current_gitlink(self, pin_path: str) -> tuple[str, str]:
        parent = self.ownership.root_path("orchestration")
        line = _git(parent, "ls-tree", "HEAD", "--", pin_path).decode().strip()
        match = re.fullmatch(r"160000 commit ([0-9a-f]{40,64})\t.+", line)
        if match is None:
            raise RootOwnershipDenied("declared pin path is not a parent Gitlink")
        indexed_line = _git(parent, "ls-files", "--stage", "--", pin_path).decode().strip()
        indexed = re.fullmatch(r"160000 ([0-9a-f]{40,64}) 0\t.+", indexed_line)
        if indexed is None:
            raise RootOwnershipDenied("declared pin path is not an unstaged index Gitlink")
        return match.group(1), indexed.group(1)

    @staticmethod
    def _commit(value: object, *, field: str) -> str:
        if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value.lower()):
            raise RootOwnershipDenied(f"{field} must be a full Git object id")
        return value.lower()


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "REPAIR_ROOT_OWNERSHIP_INTERFACE",
    "REPAIR_ROOTS_SCHEMA",
    "ROOT_OWNERSHIP_RECEIPT_SCHEMA",
    "SUBMODULE_PIN_ADMISSION_INTERFACE",
    "SUBMODULE_PIN_RECEIPT_SCHEMA",
    "RepairRoot",
    "RepairRootOwnership",
    "RootBinding",
    "RootOwnershipDenied",
    "RootOwnershipReceipt",
    "SubmodulePinAdmission",
    "SubmodulePinReceipt",
]
