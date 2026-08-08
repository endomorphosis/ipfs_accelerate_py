"""Fail-closed current multi-root identity for deterministic contract repair.

The DCR forest is an immutable observation, not a claim that its containing
commit can name itself.  Source changes and child gitlinks are committed
before capture.  The resulting clean subject may then advance through only
the bounded DCR-011 carrier, integration, and todo-completion transitions.
Any other descendant is a historical, integrity-valid document but is not
current and cannot authorize downstream evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..autonomous_repair.no_llm_policy import (
    DeterministicRepairAuthorityPolicy,
)
from ..autonomous_repair.root_ownership import RepairRootOwnership

REPAIR_FOREST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest@2"
)
PORTABLE_FOREST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest-portable@2"
)
LOCAL_FOREST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest-local@2"
)
DIRTY_OVERLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-dirty-overlay@2"
)
FOREST_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest-validation@2"
)
FOREST_LIFECYCLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest-lifecycle@1"
)
FOREST_EXCLUSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-overlay-exclusions@1"
)

FOREST_FILENAME: Final[str] = "forest.json"
DCR_TASK_ID: Final[str] = "DCR-011"
DCR_ARTIFACT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/forest.json"
)
DCR_TODO_PATH: Final[str] = (
    "implementation_plan/docs/"
    "48-ipfs-accelerate-deterministic-swissknife-mcplusplus-repair.todo.md"
)
DCR_SCHEDULER_POLICY_PATH: Final[str] = (
    "config/deterministic_swissknife_mcplusplus_repair_scheduler.json"
)
DCR_CARRIER_SUBJECT: Final[str] = (
    "DCR-011: Materialize one current multi-root forest and overlay identity"
)
DCR_TODO_SUBJECT: Final[str] = "DCR-011: mark todo completed"
DCR_ROOT_IDS: Final[tuple[str, ...]] = (
    "ipfs-accelerate",
    "ipfs-datasets",
    "ipfs-kit",
    "mcp-plus-plus",
    "orchestration",
    "swissknife",
)
_SCHEDULER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor."
    "deterministic_swissknife_mcplusplus_repair.scheduler_config@1"
)
_RUNTIME_ROOT: Final[str] = "data/agent_supervisor/deterministic_contract_repair"
_REVIEWED_RUNTIME_PREFIXES: Final[tuple[str, ...]] = (
    f"{_RUNTIME_ROOT}/evidence",
    f"{_RUNTIME_ROOT}/logs",
    f"{_RUNTIME_ROOT}/merge-queue",
    f"{_RUNTIME_ROOT}/state",
    f"{_RUNTIME_ROOT}/worktrees",
)
_CACHE_COMPONENTS: Final[frozenset[str]] = frozenset({"__pycache__", ".pytest_cache"})
_OID_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_MAX_GITLINK_DEPTH: Final[int] = 16
_GIT_TIMEOUT_SECONDS: Final[int] = 30
_GIT_CONTEXT_VARIABLES: Final[tuple[str, ...]] = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_DIR",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_WORK_TREE",
)


class DeterministicRepairForestError(ValueError):
    """A required root or identity invariant could not be proven."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "deterministic_repair_forest_error")
        super().__init__(message or self.reason_code)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeterministicRepairForestError(
                "duplicate_json_key", f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeterministicRepairForestError("noncanonical_forest_value") from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _content_id(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeterministicRepairForestError("noncanonical_forest_value") from exc


def _read_json_bytes(value: bytes, *, reason: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except DeterministicRepairForestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeterministicRepairForestError(reason) from exc
    if not isinstance(payload, Mapping):
        raise DeterministicRepairForestError(reason)
    return payload


def _read_json(path: Path, *, reason: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DeterministicRepairForestError(reason, str(path)) from exc
    return _read_json_bytes(raw, reason=reason)


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in _GIT_CONTEXT_VARIABLES:
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            environment.pop(name, None)
    environment["GIT_LITERAL_PATHSPECS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return environment


def _safe_relative(value: object, *, field: str, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeterministicRepairForestError(
            "invalid_root_policy", f"{field} is required"
        )
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or "\x00" in normalized
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise DeterministicRepairForestError("invalid_root_policy", f"unsafe {field}")
    canonical = path.as_posix()
    if canonical == "." and allow_dot:
        return canonical
    if canonical in {"", "."}:
        raise DeterministicRepairForestError("invalid_root_policy", f"unsafe {field}")
    return canonical.rstrip("/")


def _git(
    root: Path,
    *arguments: str,
    binary: bool = True,
    reason: str = "git_observation_failed",
) -> bytes | str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeterministicRepairForestError(reason, str(root)) from exc
    if result.returncode:
        raise DeterministicRepairForestError(reason, str(root))
    if binary:
        return result.stdout
    return os.fsdecode(result.stdout).rstrip("\r\n")


def _git_oid(root: Path, *arguments: str) -> str:
    value = str(_git(root, *arguments, binary=False)).strip().lower()
    if not _OID_PATTERN.fullmatch(value):
        raise DeterministicRepairForestError("invalid_git_identity", str(root))
    return value


def _git_path_text(root: Path, *arguments: str) -> str:
    """Return Git path output without corrupting case-sensitive coordinates."""

    return str(_git(root, *arguments, binary=False))


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, descendant),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _decode_path_list(raw: bytes, *, field: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        decoded = encoded.decode("utf-8", "surrogateescape")
        try:
            relative = _safe_relative(decoded, field=field)
        except DeterministicRepairForestError as exc:
            raise DeterministicRepairForestError("overlay_path_unsafe") from exc
        paths.add(relative)
    return tuple(sorted(paths))


def _closed_exclusions(
    root_id: str,
    runtime_prefixes: Sequence[str] = _REVIEWED_RUNTIME_PREFIXES,
) -> tuple[str, ...]:
    values = ["component:**/__pycache__", "component:**/.pytest_cache"]
    if root_id == "orchestration":
        values.append(f"path:{DCR_ARTIFACT_PATH}")
        values.extend(f"prefix:{prefix}" for prefix in runtime_prefixes)
    return tuple(values)


def _validate_requested_exclusions(
    requested: Mapping[str, Sequence[str]] | None,
) -> None:
    """Reject caller-authored identity holes; only reviewed rules are accepted."""

    if requested is None:
        return
    if not isinstance(requested, Mapping):
        raise DeterministicRepairForestError("unreviewed_overlay_exclusion")
    for root_id, values in requested.items():
        if root_id not in DCR_ROOT_IDS or isinstance(values, (str, bytes)):
            raise DeterministicRepairForestError("unreviewed_overlay_exclusion")
        allowed_paths = {DCR_ARTIFACT_PATH} if root_id == "orchestration" else set()
        for value in values:
            try:
                normalized = _safe_relative(value, field="overlay exclusion")
            except DeterministicRepairForestError as exc:
                raise DeterministicRepairForestError(
                    "unreviewed_overlay_exclusion"
                ) from exc
            if normalized not in allowed_paths:
                raise DeterministicRepairForestError(
                    "unreviewed_overlay_exclusion", normalized
                )


def _path_is_excluded(
    root_id: str,
    relative: str,
    runtime_prefixes: Sequence[str] = _REVIEWED_RUNTIME_PREFIXES,
) -> bool:
    if root_id == "orchestration" and relative == DCR_ARTIFACT_PATH:
        return True
    if root_id == "orchestration" and any(
        relative == prefix or relative.startswith(f"{prefix}/")
        for prefix in runtime_prefixes
    ):
        return True
    return bool(_CACHE_COMPONENTS.intersection(PurePosixPath(relative).parts))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DeterministicRepairForestError(
            "overlay_entry_unreadable", str(path)
        ) from exc
    return digest.hexdigest()


def _path_parent_within(root: Path, candidate: Path) -> None:
    try:
        candidate.parent.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise DeterministicRepairForestError("overlay_path_escape") from exc


def _index_records(root: Path, relative: str) -> tuple[str, ...]:
    raw = _git(root, "ls-files", "--stage", "-z", "--", relative)
    assert isinstance(raw, bytes)
    records: list[str] = []
    for row in raw.split(b"\0"):
        if not row:
            continue
        metadata, separator, _path = row.partition(b"\t")
        if not separator:
            raise DeterministicRepairForestError("invalid_git_index_entry")
        try:
            text = metadata.decode("ascii", "strict")
        except UnicodeDecodeError as exc:
            raise DeterministicRepairForestError("invalid_git_index_entry") from exc
        fields = text.split()
        if len(fields) != 3 or fields[0] not in {
            "100644",
            "100755",
            "120000",
            "160000",
        }:
            raise DeterministicRepairForestError("invalid_git_index_entry")
        if not _OID_PATTERN.fullmatch(fields[1]) or not fields[2].isdigit():
            raise DeterministicRepairForestError("invalid_git_index_entry")
        records.append(" ".join(fields))
    return tuple(sorted(records))


def _worktree_path_identity(
    root: Path, relative: str, *, index_records: tuple[str, ...]
) -> dict[str, str]:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _path_parent_within(root, candidate)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        return {"kind": "missing", "mode": "", "digest": ""}
    except OSError as exc:
        raise DeterministicRepairForestError("overlay_entry_unreadable") from exc

    mode = format(stat.S_IMODE(metadata.st_mode), "04o")
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = os.readlink(candidate).encode("utf-8", "surrogateescape")
        except OSError as exc:
            raise DeterministicRepairForestError("overlay_entry_unreadable") from exc
        return {
            "kind": "symlink",
            "mode": mode,
            "digest": hashlib.sha256(target).hexdigest(),
        }
    if stat.S_ISREG(metadata.st_mode):
        return {"kind": "file", "mode": mode, "digest": _file_digest(candidate)}
    if stat.S_ISDIR(metadata.st_mode) and any(
        record.startswith("160000 ") for record in index_records
    ):
        child_head = _git_oid(candidate, "rev-parse", "HEAD")
        child_status = _git(
            candidate,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        assert isinstance(child_status, bytes)
        return {
            "kind": "gitlink-worktree",
            "mode": mode,
            "digest": _content_id(
                {
                    "head": child_head,
                    "status_sha256": hashlib.sha256(child_status).hexdigest(),
                }
            ),
        }
    raise DeterministicRepairForestError(
        "overlay_entry_unsupported", f"{root}:{relative}"
    )


@dataclass(frozen=True)
class DirtyOverlay:
    """Exact non-excluded index/worktree/untracked/ignored observation."""

    entries: tuple[Mapping[str, Any], ...] = ()
    exclusions: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return _content_id(self.to_portable_dict())

    @property
    def dirty(self) -> bool:
        return bool(self.entries)

    def to_portable_dict(self) -> dict[str, Any]:
        return {
            "schema": DIRTY_OVERLAY_SCHEMA,
            "entries": [dict(item) for item in self.entries],
            "exclusions": list(self.exclusions),
        }


def _overlay_once(
    root: Path,
    *,
    root_id: str,
    runtime_prefixes: Sequence[str] = _REVIEWED_RUNTIME_PREFIXES,
) -> DirtyOverlay:
    commands = {
        "tracked": (
            "diff",
            "--name-only",
            "-z",
            "--ignore-submodules=none",
            "HEAD",
            "--",
        ),
        "index": (
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--ignore-submodules=none",
            "HEAD",
            "--",
        ),
        "untracked": ("ls-files", "--others", "--exclude-standard", "-z"),
        "ignored": (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ),
    }
    sources_by_path: dict[str, set[str]] = {}
    for source, arguments in commands.items():
        raw = _git(root, *arguments)
        assert isinstance(raw, bytes)
        for relative in _decode_path_list(raw, field=f"{source} overlay path"):
            if _path_is_excluded(root_id, relative, runtime_prefixes):
                continue
            sources_by_path.setdefault(relative, set()).add(source)

    flagged = _git(root, "ls-files", "-v", "-z")
    assert isinstance(flagged, bytes)
    for record in flagged.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise DeterministicRepairForestError("invalid_git_index_flag")
        # H is the ordinary cached tag.  Lowercase tags expose
        # assume-unchanged; S exposes skip-worktree/sparse entries.  Bind every
        # nonordinary entry so index hints cannot hide worktree changes.
        tag_bytes = record[:1]
        if tag_bytes == b"H":
            continue
        try:
            tag = tag_bytes.decode("ascii", "strict")
        except UnicodeDecodeError as exc:
            raise DeterministicRepairForestError("invalid_git_index_flag") from exc
        relative = _decode_path_list(
            record[2:] + b"\0",
            field="flagged index path",
        )[0]
        if _path_is_excluded(root_id, relative, runtime_prefixes):
            continue
        sources_by_path.setdefault(relative, set()).add(f"index-flag:{tag}")

    rows: list[dict[str, Any]] = []
    for relative in sorted(sources_by_path):
        index = _index_records(root, relative)
        rows.append(
            {
                "path": relative,
                "sources": sorted(sources_by_path[relative]),
                "index": list(index),
                "worktree": _worktree_path_identity(
                    root, relative, index_records=index
                ),
            }
        )
    return DirtyOverlay(
        entries=tuple(rows),
        exclusions=_closed_exclusions(root_id, runtime_prefixes),
    )


def _gitlinks_at_commit(root: Path, commit: str) -> tuple[tuple[str, str], ...]:
    raw = _git(root, "ls-tree", "-r", "-z", commit)
    assert isinstance(raw, bytes)
    rows: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise DeterministicRepairForestError("invalid_git_tree_entry")
        if fields[0] != b"160000":
            continue
        try:
            oid = fields[2].decode("ascii", "strict").lower()
            relative = _safe_relative(
                encoded_path.decode("utf-8", "surrogateescape"),
                field="gitlink path",
            )
        except (UnicodeDecodeError, DeterministicRepairForestError) as exc:
            raise DeterministicRepairForestError("invalid_gitlink_identity") from exc
        if not _OID_PATTERN.fullmatch(oid):
            raise DeterministicRepairForestError("invalid_gitlink_identity")
        rows.append((relative, oid))
    return tuple(sorted(rows))


def _checkout_probe(
    candidate: Path,
    *,
    recorded: str,
    full_path: str,
) -> str | None:
    """Return exact initialized checkout coordinates, or ``None`` if absent."""

    if not candidate.is_dir():
        return None
    try:
        top_level = Path(
            _git_path_text(
                candidate,
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
            )
        ).resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (DeterministicRepairForestError, OSError, RuntimeError):
        return None
    # An empty uninitialized submodule directory otherwise discovers its
    # parent repository.  That is absence, not evidence for the child.
    if top_level != resolved:
        return None
    child_head = _git_oid(candidate, "rev-parse", "HEAD")
    if child_head != recorded:
        raise DeterministicRepairForestError("gitlink_head_mismatch", full_path)
    return _git_oid(candidate, "rev-parse", "HEAD^{tree}")


def _recursive_gitlinks(
    root: Path,
    head: str,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Bind direct Merkle pins and observe only actual recursive checkouts.

    Clean initialized and uninitialized children produce the same direct-pin
    identity.  Recursion exists solely to bind dirt that can affect an actual
    initialized checkout; an absent checkout is a closed Merkle leaf.
    """

    boundary = root.resolve(strict=True)
    direct_pins: list[dict[str, Any]] = []
    dirty_overlays: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    active: set[tuple[str, str]] = set()
    root_probe = _checkout_probe(boundary, recorded=head, full_path=".")
    if root_probe is None:
        raise DeterministicRepairForestError("root_git_worktree_mismatch")
    root_key = (head, root_probe)
    active.add(root_key)

    def walk(
        checkout: Path,
        commit: str,
        prefix: str,
        depth: int,
    ) -> None:
        if depth > _MAX_GITLINK_DEPTH:
            raise DeterministicRepairForestError("recursive_gitlink_depth_exceeded")
        for relative, recorded in _gitlinks_at_commit(checkout, commit):
            full_path = f"{prefix}/{relative}" if prefix else relative
            candidate = checkout.joinpath(*PurePosixPath(relative).parts)
            try:
                candidate.resolve(strict=False).relative_to(boundary)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DeterministicRepairForestError(
                    "gitlink_path_escape", full_path
                ) from exc
            checkout_tree = (
                _checkout_probe(
                    candidate,
                    recorded=recorded,
                    full_path=full_path,
                )
                or ""
            )
            if depth == 0:
                # The immutable Gitlink commit is the recursive Merkle
                # boundary.  Cache/object-store/checkout availability never
                # changes this portable identity.
                direct_pins.append(
                    {
                        "path": relative,
                        "commit": recorded,
                        "closure_state": "merkle_leaf",
                        "repository_identity": _content_id(
                            {"gitlink_commit": recorded}
                        ),
                    }
                )
            if not checkout_tree:
                local_rows.append(
                    {
                        "path": full_path,
                        "commit": recorded,
                        "checkout_state": "uninitialized",
                        "closure_state": "merkle_leaf",
                        "overlay_digest": "",
                    }
                )
                continue

            repository_key = (recorded, checkout_tree)
            closure_state = "cycle" if repository_key in active else "initialized"
            child_overlay = _overlay_once(candidate, root_id="gitlink")
            dirty_overlay = child_overlay if child_overlay.dirty else None
            if dirty_overlay is not None:
                dirty_overlays.append(
                    {
                        "path": full_path,
                        "commit": recorded,
                        "tree": checkout_tree,
                        "overlay": dirty_overlay.to_portable_dict(),
                        "overlay_digest": dirty_overlay.digest,
                    }
                )
            local_rows.append(
                {
                    "path": full_path,
                    "commit": recorded,
                    "checkout_state": "initialized",
                    "closure_state": closure_state,
                    "overlay_digest": child_overlay.digest,
                }
            )
            if closure_state == "cycle":
                continue
            active.add(repository_key)
            try:
                walk(
                    candidate,
                    recorded,
                    full_path,
                    depth + 1,
                )
            finally:
                active.remove(repository_key)

    walk(boundary, head, "", 0)
    return (
        tuple(sorted(direct_pins, key=lambda item: str(item["path"]))),
        tuple(sorted(dirty_overlays, key=lambda item: str(item["path"]))),
        tuple(sorted(local_rows, key=lambda item: str(item["path"]))),
    )


def _parent_pin(workspace: Path, pin_path: str) -> str:
    if not pin_path:
        return ""
    raw = _git(workspace, "ls-tree", "-z", "HEAD", "--", pin_path)
    assert isinstance(raw, bytes)
    records = [item for item in raw.split(b"\0") if item]
    if len(records) != 1:
        raise DeterministicRepairForestError("parent_gitlink_missing", pin_path)
    metadata, separator, _encoded_path = records[0].partition(b"\t")
    fields = metadata.split()
    if not separator or len(fields) != 3 or fields[0] != b"160000":
        raise DeterministicRepairForestError("parent_gitlink_missing", pin_path)
    try:
        oid = fields[2].decode("ascii", "strict").lower()
    except UnicodeDecodeError as exc:
        raise DeterministicRepairForestError("invalid_gitlink_identity") from exc
    if not _OID_PATTERN.fullmatch(oid):
        raise DeterministicRepairForestError("invalid_gitlink_identity")
    return oid


def _authority_schema_valid(authority: Mapping[str, Any]) -> bool:
    try:
        DeterministicRepairAuthorityPolicy.from_mapping(authority)
    except ValueError:
        return False
    return True


def _scheduler_runtime_prefixes(scheduler: Mapping[str, Any]) -> tuple[str, ...]:
    if scheduler.get("schema") != _SCHEDULER_SCHEMA:
        raise DeterministicRepairForestError("invalid_scheduler_policy")
    runtime_paths = scheduler.get("runtime_paths")
    expected_fields = {
        "root",
        "state",
        "worktrees",
        "merge_queue",
        "logs",
        "evidence",
        "generated_runtime_artifacts_are_completion_authority",
    }
    if not isinstance(runtime_paths, Mapping) or set(runtime_paths) != expected_fields:
        raise DeterministicRepairForestError("invalid_scheduler_runtime_paths")
    expected_paths = {
        "root": _RUNTIME_ROOT,
        "state": f"{_RUNTIME_ROOT}/state",
        "worktrees": f"{_RUNTIME_ROOT}/worktrees",
        "merge_queue": f"{_RUNTIME_ROOT}/merge-queue",
        "logs": f"{_RUNTIME_ROOT}/logs",
        "evidence": f"{_RUNTIME_ROOT}/evidence",
    }
    if (
        runtime_paths.get("generated_runtime_artifacts_are_completion_authority")
        is not False
    ):
        raise DeterministicRepairForestError("invalid_scheduler_runtime_paths")
    try:
        observed = {
            key: _safe_relative(runtime_paths.get(key), field=f"runtime_paths.{key}")
            for key in expected_paths
        }
    except DeterministicRepairForestError as exc:
        raise DeterministicRepairForestError("invalid_scheduler_runtime_paths") from exc
    if observed != expected_paths:
        raise DeterministicRepairForestError("invalid_scheduler_runtime_paths")
    return tuple(sorted(observed[key] for key in observed if key != "root"))


def _policy_inputs(
    workspace: Path,
    root_policy: Mapping[str, Any] | None,
    authority_policy: Mapping[str, Any] | None,
    scheduler_policy: Mapping[str, Any] | None,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[str, ...],
    RepairRootOwnership,
]:
    roots = (
        root_policy
        if root_policy is not None
        else _read_json(
            workspace / "config/deterministic_contract_repair_roots.json",
            reason="root_policy_unreadable",
        )
    )
    authority = (
        authority_policy
        if authority_policy is not None
        else _read_json(
            workspace / "config/deterministic_contract_repair_authority.json",
            reason="authority_policy_unreadable",
        )
    )
    scheduler = (
        scheduler_policy
        if scheduler_policy is not None
        else _read_json(
            workspace / DCR_SCHEDULER_POLICY_PATH,
            reason="scheduler_policy_unreadable",
        )
    )
    if (
        not isinstance(roots, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(scheduler, Mapping)
    ):
        raise DeterministicRepairForestError("invalid_policy_document")
    try:
        ownership = RepairRootOwnership.from_mapping(roots, workspace_root=workspace)
    except (ValueError, PermissionError) as exc:
        raise DeterministicRepairForestError("invalid_root_policy") from exc
    if tuple(sorted(root.root_id for root in ownership.roots)) != DCR_ROOT_IDS:
        raise DeterministicRepairForestError("required_root_set_changed")
    if not _authority_schema_valid(authority):
        raise DeterministicRepairForestError("invalid_authority_policy")
    runtime_prefixes = _scheduler_runtime_prefixes(scheduler)
    return roots, authority, scheduler, runtime_prefixes, ownership


def _capture_roots_once(
    workspace: Path,
    ownership: RepairRootOwnership,
    runtime_prefixes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    portable: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    for declared in sorted(ownership.roots, key=lambda item: item.root_id):
        root = ownership.root_path(declared.root_id)
        head = _git_oid(root, "rev-parse", "HEAD")
        tree = _git_oid(root, "rev-parse", "HEAD^{tree}")
        pin = _parent_pin(workspace, declared.pin_path)
        if declared.root_id != "orchestration" and pin != head:
            raise DeterministicRepairForestError(
                "parent_gitlink_mismatch", declared.root_id
            )
        # The complete forest capture below is repeated and compared, so one
        # overlay pass here is sufficient and avoids four full scans per root.
        overlay = _overlay_once(
            root,
            root_id=declared.root_id,
            runtime_prefixes=runtime_prefixes,
        )
        recursive_gitlinks, recursive_dirty_overlays, recursive_checkouts = (
            _recursive_gitlinks(
                root,
                head,
            )
        )
        unavailable_count = sum(
            item.get("checkout_state") == "uninitialized"
            for item in recursive_checkouts
        )
        portable.append(
            {
                "id": declared.root_id,
                "relative_path": declared.relative_path,
                "role": declared.role,
                "allowed_write_prefixes": list(declared.allowed_write_prefixes),
                "pin_path": declared.pin_path,
                "head": head,
                "tree": tree,
                "parent_gitlink_pin": pin,
                "recursive_gitlinks": [dict(item) for item in recursive_gitlinks],
                "recursive_dirty_overlays": [
                    dict(item) for item in recursive_dirty_overlays
                ],
                "overlay": overlay.to_portable_dict(),
                "overlay_digest": overlay.digest,
            }
        )
        common = _git_path_text(
            root, "rev-parse", "--path-format=absolute", "--git-common-dir"
        )
        local.append(
            {
                "id": declared.root_id,
                "configured_path": declared.relative_path,
                "resolved_path": str(root),
                "git_common_dir": common,
                "recursive_checkouts": [dict(item) for item in recursive_checkouts],
                "recursive_closure_complete": unavailable_count == 0,
                "recursive_unavailable_count": unavailable_count,
            }
        )
    return portable, local


def _capture_roots(
    workspace: Path,
    ownership: RepairRootOwnership,
    runtime_prefixes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_portable, _first_local = _capture_roots_once(
        workspace,
        ownership,
        runtime_prefixes,
    )
    second_portable, second_local = _capture_roots_once(
        workspace,
        ownership,
        runtime_prefixes,
    )
    if first_portable != second_portable:
        raise DeterministicRepairForestError("forest_capture_race")
    return second_portable, second_local


@dataclass(frozen=True)
class RepositoryForestManifest:
    """One immutable portable forest and capture-time host projection."""

    portable: Mapping[str, Any]
    local: Mapping[str, Any]

    @property
    def forest_id(self) -> str:
        return str(self.portable.get("forest_id") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_FOREST_SCHEMA,
            "forest_id": self.forest_id,
            "portable": dict(self.portable),
            "local": dict(self.local),
            "authoritative": False,
            "completion_authorized": False,
        }


def _default_workspace() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config/deterministic_contract_repair_roots.json").is_file():
            return candidate
    raise DeterministicRepairForestError("workspace_missing")


def materialize_repair_forest(
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    root_policy: Mapping[str, Any] | None = None,
    authority_policy: Mapping[str, Any] | None = None,
    scheduler_policy: Mapping[str, Any] | None = None,
    overlay_exclusions: Mapping[str, Sequence[str]] | None = None,
) -> RepositoryForestManifest:
    """Capture the exact clean-pin DCR subject with closed exclusions."""

    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    if not workspace.is_dir():
        raise DeterministicRepairForestError("workspace_missing")
    _validate_requested_exclusions(overlay_exclusions)
    roots, authority, scheduler, runtime_prefixes, ownership = _policy_inputs(
        workspace,
        root_policy,
        authority_policy,
        scheduler_policy,
    )
    portable_roots, local_roots = _capture_roots(
        workspace,
        ownership,
        runtime_prefixes,
    )
    orchestration = next(
        item for item in portable_roots if item["id"] == "orchestration"
    )
    policy_binding = {
        "root_policy_digest": _content_id(roots),
        "authority_policy_digest": _content_id(authority),
        "scheduler_policy_digest": _content_id(scheduler),
        "root_policy_schema": str(roots.get("schema") or ""),
        "authority_policy_schema": str(authority.get("schema") or ""),
        "scheduler_policy_schema": str(scheduler.get("schema") or ""),
        "exclusion_policy": {
            "schema": FOREST_EXCLUSION_SCHEMA,
            "runtime_path_source": DCR_SCHEDULER_POLICY_PATH,
            "roots": {
                root_id: list(_closed_exclusions(root_id, runtime_prefixes))
                for root_id in DCR_ROOT_IDS
            },
        },
    }
    lifecycle = {
        "schema": FOREST_LIFECYCLE_SCHEMA,
        "task_id": DCR_TASK_ID,
        "subject_root_id": "orchestration",
        "subject_head": orchestration["head"],
        "subject_tree": orchestration["tree"],
        "artifact_path": DCR_ARTIFACT_PATH,
        "todo_path": DCR_TODO_PATH,
        "carrier_subject": DCR_CARRIER_SUBJECT,
        "todo_subject": DCR_TODO_SUBJECT,
        "max_transition_commits": 3,
    }
    identity = {
        "schema": PORTABLE_FOREST_SCHEMA,
        "policy": policy_binding,
        "required_root_ids": list(DCR_ROOT_IDS),
        "roots": portable_roots,
        "lifecycle": lifecycle,
    }
    forest_id = _content_id(identity)
    portable = {**identity, "forest_id": forest_id}
    local = {
        "schema": LOCAL_FOREST_SCHEMA,
        "forest_id": forest_id,
        "portable_forest_id": forest_id,
        "roots": local_roots,
    }
    return RepositoryForestManifest(portable=portable, local=local)


@dataclass(frozen=True)
class ForestValidation:
    """Integrity and freshness are intentionally separate authority claims."""

    integrity_valid: bool = False
    current: bool = False
    downstream_authorized: bool = False
    lifecycle_state: str = "invalid"
    forest_id: str = ""
    observed_repository_commit: str = ""
    current_repository_commit: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Fail-closed compatibility alias used by downstream callers."""

        return self.downstream_authorized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FOREST_VALIDATION_SCHEMA,
            "valid": self.valid,
            "integrity_valid": self.integrity_valid,
            "current": self.current,
            "downstream_authorized": self.downstream_authorized,
            "lifecycle_state": self.lifecycle_state,
            "forest_id": self.forest_id,
            "observed_repository_commit": self.observed_repository_commit,
            "current_repository_commit": self.current_repository_commit,
            "reason_codes": list(self.reason_codes),
        }


def _manifest_payload(
    source: Mapping[str, Any] | str | os.PathLike[str],
) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    return _read_json(Path(source), reason="forest_unreadable")


def _portable_has_host_leak(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"resolved_path", "git_common_dir", "local_locator"}:
                return True
            if _portable_has_host_leak(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_portable_has_host_leak(item) for item in value)
    return False


def _local_root_projection_valid(
    item: Mapping[str, Any],
    *,
    configured_path: str,
) -> bool:
    if set(item) != {
        "id",
        "configured_path",
        "resolved_path",
        "git_common_dir",
        "recursive_checkouts",
        "recursive_closure_complete",
        "recursive_unavailable_count",
    }:
        return False
    if item.get("configured_path") != configured_path:
        return False
    for key in ("resolved_path", "git_common_dir"):
        coordinate = item.get(key)
        if not isinstance(coordinate, str) or not Path(coordinate).is_absolute():
            return False
    checkouts = item.get("recursive_checkouts")
    if not isinstance(checkouts, list):
        return False
    paths: list[str] = []
    unavailable = 0
    for row in checkouts:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "commit",
            "checkout_state",
            "closure_state",
            "overlay_digest",
        }:
            return False
        try:
            path = _safe_relative(row.get("path"), field="recursive checkout path")
        except DeterministicRepairForestError:
            return False
        commit = row.get("commit")
        checkout_state = row.get("checkout_state")
        closure_state = row.get("closure_state")
        overlay_digest = row.get("overlay_digest")
        if not isinstance(commit, str) or not _OID_PATTERN.fullmatch(commit):
            return False
        if checkout_state == "uninitialized":
            unavailable += 1
            if closure_state != "merkle_leaf" or overlay_digest != "":
                return False
        elif checkout_state == "initialized":
            if closure_state not in {"initialized", "cycle"} or not (
                isinstance(overlay_digest, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", overlay_digest)
            ):
                return False
        else:
            return False
        paths.append(path)
    if paths != sorted(set(paths)):
        return False
    return (
        item.get("recursive_closure_complete") is (unavailable == 0)
        and type(item.get("recursive_unavailable_count")) is int
        and item.get("recursive_unavailable_count") == unavailable
    )


def _document_integrity(
    payload: Mapping[str, Any],
) -> tuple[RepositoryForestManifest | None, tuple[str, ...]]:
    reasons: list[str] = []
    expected_top = {
        "schema",
        "forest_id",
        "portable",
        "local",
        "authoritative",
        "completion_authorized",
    }
    if set(payload) != expected_top or payload.get("schema") != REPAIR_FOREST_SCHEMA:
        return None, ("invalid_forest_document",)
    if (
        payload.get("authoritative") is not False
        or payload.get("completion_authorized") is not False
    ):
        reasons.append("invalid_forest_authority_claim")
    portable = payload.get("portable")
    local = payload.get("local")
    if not isinstance(portable, Mapping) or not isinstance(local, Mapping):
        return None, ("invalid_forest_document",)
    expected_portable = {
        "schema",
        "forest_id",
        "policy",
        "required_root_ids",
        "roots",
        "lifecycle",
    }
    if (
        set(portable) != expected_portable
        or portable.get("schema") != PORTABLE_FOREST_SCHEMA
    ):
        reasons.append("invalid_portable_projection")
    claimed = portable.get("forest_id")
    identity = {key: value for key, value in portable.items() if key != "forest_id"}
    try:
        recomputed = _content_id(identity)
    except DeterministicRepairForestError:
        recomputed = ""
    if not isinstance(claimed, str) or claimed != recomputed:
        reasons.append("portable_forest_id_mismatch")
    if payload.get("forest_id") != claimed:
        reasons.append("document_forest_id_mismatch")
    if _portable_has_host_leak(portable):
        reasons.append("portable_projection_host_leak")
    required = portable.get("required_root_ids")
    roots = portable.get("roots")
    if required != list(DCR_ROOT_IDS) or not isinstance(roots, list):
        reasons.append("required_root_set_changed")
        roots = []
    root_ids = [item.get("id") for item in roots if isinstance(item, Mapping)]
    if root_ids != list(DCR_ROOT_IDS) or len(root_ids) != len(roots):
        reasons.append("required_root_set_changed")
    policy = portable.get("policy")
    if not isinstance(policy, Mapping):
        reasons.append("invalid_policy_binding")
    else:
        expected_policy_fields = {
            "root_policy_digest",
            "authority_policy_digest",
            "scheduler_policy_digest",
            "root_policy_schema",
            "authority_policy_schema",
            "scheduler_policy_schema",
            "exclusion_policy",
        }
        if set(policy) != expected_policy_fields:
            reasons.append("invalid_policy_binding")
        for key in (
            "root_policy_digest",
            "authority_policy_digest",
            "scheduler_policy_digest",
        ):
            value = policy.get(key)
            if not isinstance(value, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value
            ):
                reasons.append("invalid_policy_binding")
                break
        if (
            policy.get("root_policy_schema")
            != "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1"
            or policy.get("authority_policy_schema")
            != "ipfs_accelerate_py/agent-supervisor/"
            "deterministic-repair-authority-policy@1"
            or policy.get("scheduler_policy_schema") != _SCHEDULER_SCHEMA
        ):
            reasons.append("invalid_policy_binding")
        exclusions = policy.get("exclusion_policy")
        expected_exclusions = {
            "schema": FOREST_EXCLUSION_SCHEMA,
            "runtime_path_source": DCR_SCHEDULER_POLICY_PATH,
            "roots": {
                root_id: list(_closed_exclusions(root_id)) for root_id in DCR_ROOT_IDS
            },
        }
        if exclusions != expected_exclusions:
            reasons.append("unreviewed_overlay_exclusion")
    lifecycle = portable.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        reasons.append("invalid_lifecycle_policy")
        lifecycle = {}
    lifecycle_expected = {
        "schema": FOREST_LIFECYCLE_SCHEMA,
        "task_id": DCR_TASK_ID,
        "subject_root_id": "orchestration",
        "artifact_path": DCR_ARTIFACT_PATH,
        "todo_path": DCR_TODO_PATH,
        "carrier_subject": DCR_CARRIER_SUBJECT,
        "todo_subject": DCR_TODO_SUBJECT,
        "max_transition_commits": 3,
    }
    for key, value in lifecycle_expected.items():
        if lifecycle.get(key) != value:
            reasons.append("invalid_lifecycle_policy")
            break
    by_id = {str(item.get("id")): item for item in roots if isinstance(item, Mapping)}
    orchestration = by_id.get("orchestration")
    if not isinstance(orchestration, Mapping):
        reasons.append("required_root_set_changed")
    elif lifecycle.get("subject_head") != orchestration.get("head") or lifecycle.get(
        "subject_tree"
    ) != orchestration.get("tree"):
        reasons.append("lifecycle_subject_mismatch")

    expected_local = {"schema", "forest_id", "portable_forest_id", "roots"}
    if set(local) != expected_local or local.get("schema") != LOCAL_FOREST_SCHEMA:
        reasons.append("invalid_local_projection")
    if local.get("forest_id") != claimed or local.get("portable_forest_id") != claimed:
        reasons.append("projection_forest_id_mismatch")
    local_roots = local.get("roots")
    if not isinstance(local_roots, list):
        reasons.append("invalid_local_projection")
        local_roots = []
    local_ids = [item.get("id") for item in local_roots if isinstance(item, Mapping)]
    if local_ids != list(DCR_ROOT_IDS) or len(local_ids) != len(local_roots):
        reasons.append("invalid_local_projection")
    portable_paths = {
        str(item.get("id")): str(item.get("relative_path"))
        for item in roots
        if isinstance(item, Mapping)
    }
    for item in local_roots:
        if not isinstance(item, Mapping):
            reasons.append("invalid_local_projection")
            continue
        root_id = str(item.get("id") or "")
        if not _local_root_projection_valid(
            item,
            configured_path=portable_paths.get(root_id, ""),
        ):
            reasons.append("invalid_local_projection")
    if reasons:
        return None, tuple(dict.fromkeys(reasons))
    return RepositoryForestManifest(portable=portable, local=local), ()


def _commit_parents(root: Path, commit: str) -> tuple[str, ...]:
    text = str(_git(root, "rev-list", "--parents", "-n", "1", commit, binary=False))
    fields = text.split()
    if (
        not fields
        or fields[0] != commit
        or not all(_OID_PATTERN.fullmatch(item) for item in fields)
    ):
        raise DeterministicRepairForestError("invalid_commit_graph")
    return tuple(fields[1:])


def _commit_subject(root: Path, commit: str) -> str:
    return str(_git(root, "show", "-s", "--format=%s", commit, binary=False))


def _commit_tree(root: Path, commit: str) -> str:
    return _git_oid(root, "rev-parse", f"{commit}^{{tree}}")


def _changed_paths(root: Path, parent: str, commit: str) -> tuple[str, ...]:
    raw = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        parent,
        commit,
        "--",
    )
    assert isinstance(raw, bytes)
    return _decode_path_list(raw, field="commit changed path")


def _todo_delta_is_exact(root: Path, observed: str, completed: str) -> bool:
    try:
        before_raw = _git(root, "show", f"{observed}:{DCR_TODO_PATH}")
        after_raw = _git(root, "show", f"{completed}:{DCR_TODO_PATH}")
        assert isinstance(before_raw, bytes) and isinstance(after_raw, bytes)
        before = before_raw.decode("utf-8")
        after = after_raw.decode("utf-8")
    except (DeterministicRepairForestError, UnicodeDecodeError):
        return False
    marker = "## DCR-011 "
    before_start = before.find(marker)
    after_start = after.find(marker)
    if before_start < 0 or after_start < 0:
        return False
    before_end = before.find("\n## ", before_start + len(marker))
    after_end = after.find("\n## ", after_start + len(marker))
    before_end = len(before) if before_end < 0 else before_end
    after_end = len(after) if after_end < 0 else after_end
    before_block = before[before_start:before_end]
    after_block = after[after_start:after_end]
    if before_block.count("- Status: todo") != 1:
        return False
    if after_block.count("- Status: completed") != 1:
        return False
    restored_block = after_block.replace("- Status: completed", "- Status: todo", 1)
    restored = after[:after_start] + restored_block + after[after_end:]
    return restored == before


def _artifact_matches_document(
    root: Path, payload: Mapping[str, Any], *, carrier_commit: str | None = None
) -> bool:
    expected = _artifact_bytes(payload)
    try:
        if carrier_commit:
            tree_entry = _git(
                root,
                "ls-tree",
                "-z",
                carrier_commit,
                "--",
                DCR_ARTIFACT_PATH,
            )
            assert isinstance(tree_entry, bytes)
            records = [record for record in tree_entry.split(b"\0") if record]
            if len(records) != 1:
                return False
            metadata, separator, encoded_path = records[0].partition(b"\t")
            fields = metadata.split()
            if (
                not separator
                or fields[:2] != [b"100644", b"blob"]
                or encoded_path != DCR_ARTIFACT_PATH.encode("utf-8", "surrogateescape")
            ):
                return False
            observed = _git(root, "show", f"{carrier_commit}:{DCR_ARTIFACT_PATH}")
            assert isinstance(observed, bytes)
        else:
            path = root / DCR_ARTIFACT_PATH
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111:
                return False
            observed = path.read_bytes()
    except (DeterministicRepairForestError, OSError):
        return False
    return observed == expected


def _lifecycle_state(
    root: Path,
    payload: Mapping[str, Any],
    *,
    subject: str,
    current: str,
) -> tuple[str, tuple[str, ...]]:
    if not _artifact_matches_document(root, payload):
        return "stale", ("capture_artifact_mismatch",)
    if current == subject:
        return "captured", ()
    if not _git_is_ancestor(root, subject, current):
        return "stale", ("observed_repository_commit_not_ancestor",)
    raw = _git(
        root,
        "rev-list",
        "--ancestry-path",
        "--reverse",
        "--topo-order",
        f"{subject}..{current}",
        binary=False,
    )
    commits = tuple(item for item in str(raw).splitlines() if item)
    if (
        not commits
        or len(commits) > 3
        or any(not _OID_PATTERN.fullmatch(item) for item in commits)
    ):
        return "stale", ("unrecognized_lifecycle_transition",)

    carrier = commits[0]
    if (
        _commit_parents(root, carrier) != (subject,)
        or _commit_subject(root, carrier) != DCR_CARRIER_SUBJECT
        or _changed_paths(root, subject, carrier) != (DCR_ARTIFACT_PATH,)
        or not _artifact_matches_document(root, payload, carrier_commit=carrier)
    ):
        return "stale", ("carrier_transition_invalid",)
    if len(commits) == 1:
        return "artifact_carried", ()

    merge = commits[1]
    merge_parents = _commit_parents(root, merge)
    if merge_parents != (subject, carrier):
        return "stale", ("integration_transition_invalid",)
    subject_text = _commit_subject(root, merge).lower()
    if (
        _commit_tree(root, merge) != _commit_tree(root, carrier)
        or not subject_text.startswith("merge branch '")
        or "dcr-011" not in subject_text
    ):
        return "stale", ("integration_transition_invalid",)
    if len(commits) == 2:
        return "integrated", ()

    position = 2
    completed = commits[position]
    if (
        position != len(commits) - 1
        or _commit_parents(root, completed) != (merge,)
        or _commit_subject(root, completed) != DCR_TODO_SUBJECT
        or _changed_paths(root, merge, completed) != (DCR_TODO_PATH,)
        or not _todo_delta_is_exact(root, subject, completed)
    ):
        return "stale", ("todo_transition_invalid",)
    return "todo_completed", ()


def _root_drift_reasons(
    expected_roots: Sequence[Mapping[str, Any]],
    current_roots: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    expected = {str(item["id"]): item for item in expected_roots}
    current = {str(item["id"]): item for item in current_roots}
    reasons: list[str] = []
    if set(expected) != set(current):
        return ("required_root_set_changed",)
    for root_id in DCR_ROOT_IDS:
        old = expected[root_id]
        new = current[root_id]
        ignored_keys = {"head", "tree"} if root_id == "orchestration" else set()
        old_comparable = {
            key: value for key, value in old.items() if key not in ignored_keys
        }
        new_comparable = {
            key: value for key, value in new.items() if key not in ignored_keys
        }
        if old_comparable == new_comparable:
            continue
        if old.get("parent_gitlink_pin") != new.get("parent_gitlink_pin"):
            reasons.append(f"{root_id}:parent_gitlink_changed")
        elif old.get("recursive_gitlinks") != new.get("recursive_gitlinks"):
            reasons.append(f"{root_id}:gitlink_closure_changed")
        elif old.get("recursive_dirty_overlays") != new.get("recursive_dirty_overlays"):
            reasons.append(f"{root_id}:gitlink_overlay_changed")
        elif old.get("overlay_digest") != new.get("overlay_digest"):
            reasons.append(f"{root_id}:overlay_changed")
        else:
            reasons.append(f"{root_id}:root_identity_changed")
        if root_id != "orchestration" and (
            old.get("head") != new.get("head") or old.get("tree") != new.get("tree")
        ):
            reasons.append(f"{root_id}:git_identity_changed")
    return tuple(dict.fromkeys(reasons))


def validate_repair_forest(
    source: Mapping[str, Any] | str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    root_policy: Mapping[str, Any] | None = None,
    authority_policy: Mapping[str, Any] | None = None,
    scheduler_policy: Mapping[str, Any] | None = None,
    overlay_exclusions: Mapping[str, Sequence[str]] | None = None,
) -> ForestValidation:
    """Validate intrinsic identity, then currentness under the bounded lifecycle."""

    try:
        payload = _manifest_payload(source)
    except DeterministicRepairForestError as exc:
        return ForestValidation(reason_codes=(exc.reason_code,))
    manifest, integrity_reasons = _document_integrity(payload)
    forest_id = str(payload.get("forest_id") or "")
    if manifest is None:
        return ForestValidation(forest_id=forest_id, reason_codes=integrity_reasons)
    portable = manifest.portable
    lifecycle = portable["lifecycle"]
    subject = str(lifecycle["subject_head"])
    try:
        workspace = (
            Path(workspace_root).resolve(strict=True)
            if workspace_root is not None
            else _default_workspace().resolve(strict=True)
        )
        _validate_requested_exclusions(overlay_exclusions)
        roots, authority, scheduler, runtime_prefixes, ownership = _policy_inputs(
            workspace,
            root_policy,
            authority_policy,
            scheduler_policy,
        )
        expected_policy = portable["policy"]
        if (
            expected_policy.get("root_policy_digest") != _content_id(roots)
            or expected_policy.get("authority_policy_digest") != _content_id(authority)
            or expected_policy.get("scheduler_policy_digest") != _content_id(scheduler)
        ):
            return ForestValidation(
                integrity_valid=True,
                lifecycle_state="stale",
                forest_id=forest_id,
                observed_repository_commit=subject,
                reason_codes=("policy_changed",),
            )
        current_roots, current_local_roots = _capture_roots(
            workspace,
            ownership,
            runtime_prefixes,
        )
        current_head = next(
            str(item["head"]) for item in current_roots if item["id"] == "orchestration"
        )
    except DeterministicRepairForestError as exc:
        return ForestValidation(
            integrity_valid=True,
            lifecycle_state="stale",
            forest_id=forest_id,
            observed_repository_commit=subject,
            reason_codes=(exc.reason_code,),
        )

    root_reasons = _root_drift_reasons(portable["roots"], current_roots)
    if root_reasons:
        return ForestValidation(
            integrity_valid=True,
            lifecycle_state="stale",
            forest_id=forest_id,
            observed_repository_commit=subject,
            current_repository_commit=current_head,
            reason_codes=root_reasons,
        )
    try:
        state, lifecycle_reasons = _lifecycle_state(
            workspace,
            payload,
            subject=subject,
            current=current_head,
        )
    except DeterministicRepairForestError as exc:
        state, lifecycle_reasons = "stale", (exc.reason_code,)
    if not lifecycle_reasons and state in {"captured", "artifact_carried"}:
        current_local = {
            "schema": LOCAL_FOREST_SCHEMA,
            "forest_id": forest_id,
            "portable_forest_id": forest_id,
            "roots": current_local_roots,
        }
        if manifest.local != current_local:
            state, lifecycle_reasons = "stale", ("local_projection_changed",)
    current = not lifecycle_reasons
    return ForestValidation(
        integrity_valid=True,
        current=current,
        downstream_authorized=current,
        lifecycle_state=state,
        forest_id=forest_id,
        observed_repository_commit=subject,
        current_repository_commit=current_head,
        reason_codes=lifecycle_reasons,
    )


def write_repair_forest(
    output_path: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    root_policy: Mapping[str, Any] | None = None,
    authority_policy: Mapping[str, Any] | None = None,
    scheduler_policy: Mapping[str, Any] | None = None,
) -> RepositoryForestManifest:
    """Atomically write only the reviewed DCR artifact path."""

    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    raw_destination = Path(output_path)
    destination = (
        raw_destination
        if raw_destination.is_absolute()
        else workspace / raw_destination
    )
    expected = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    try:
        destination.parent.resolve(strict=True).relative_to(workspace)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DeterministicRepairForestError("forest_output_path_invalid") from exc
    if destination.resolve(strict=False) != expected.resolve(strict=False):
        raise DeterministicRepairForestError("forest_output_path_invalid")

    manifest = materialize_repair_forest(
        workspace,
        root_policy=root_policy,
        authority_policy=authority_policy,
        scheduler_policy=scheduler_policy,
    )
    encoded = _artifact_bytes(manifest.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{FOREST_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise DeterministicRepairForestError("forest_write_failed") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the live artifact; exit zero only for downstream authority."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)
    workspace = Path(arguments.workspace).resolve(strict=False)
    artifact = Path(arguments.artifact).resolve(strict=False)
    expected = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    if artifact != expected.resolve(strict=False):
        result = ForestValidation(reason_codes=("forest_output_path_invalid",))
    else:
        result = validate_repair_forest(artifact, workspace)
    sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return (
        0
        if result.integrity_valid and result.current and result.downstream_authorized
        else 1
    )


__all__ = [
    "DCR_ARTIFACT_PATH",
    "DCR_CARRIER_SUBJECT",
    "DCR_ROOT_IDS",
    "DCR_SCHEDULER_POLICY_PATH",
    "DCR_TODO_PATH",
    "DCR_TODO_SUBJECT",
    "DIRTY_OVERLAY_SCHEMA",
    "FOREST_FILENAME",
    "FOREST_VALIDATION_SCHEMA",
    "LOCAL_FOREST_SCHEMA",
    "PORTABLE_FOREST_SCHEMA",
    "REPAIR_FOREST_SCHEMA",
    "DeterministicRepairForestError",
    "DirtyOverlay",
    "ForestValidation",
    "RepositoryForestManifest",
    "main",
    "materialize_repair_forest",
    "validate_repair_forest",
    "write_repair_forest",
]


if __name__ == "__main__":
    raise SystemExit(main())
