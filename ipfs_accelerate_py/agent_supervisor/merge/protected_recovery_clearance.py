"""Proof-checked operator clearance for retained supervisor recovery journals.

This module intentionally does not expose a force-unlock operation.  It can
only authorize the exact, otherwise-valid protected-path history rejected as
``protected_generated_history_untrusted``.  The authorization is local: the
OS account owning the checkout and lock must provide an explicit identity,
note, review identity, lock digest, lease id, and complete ordered commit list.
It is not a cryptographic signature and it never represents task completion,
verification, release, or production-promotion authority.
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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from .checkout_lock import (
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    GENERATED_PROTECTED_BOARD_COMMIT_MARKER,
    checkout_mutation_lock_path,
    checkout_repository_id,
    git_common_dir,
    serialized_lock_update,
)

REVIEW_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-clearance-review@1"
)
AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-clearance-authorization@1"
)
ROTATED_LEASE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-clearance-lease@1"
)
RELEASE_INTENT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-clearance-release-intent@1"
)
FINAL_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-clearance-final@1"
)
RESTORATION_AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-fence-restoration-authorization@1"
)
RESTORATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-recovery-fence-restoration-receipt@1"
)
SECURITY_MODEL = "local-os-owner-explicit-review-v1"
DECISION = "approved_protected_history_only"
NON_AUTHORITY = {
    "completion_authority": False,
    "verification_authority": False,
    "release_authority": False,
    "production_promotion_authority": False,
}


class ProtectedRecoveryClearanceError(RuntimeError):
    """Fail-closed operator-clearance rejection with a stable reason."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class _LockSnapshot:
    path: Path
    metadata: dict[str, Any]
    raw: bytes
    sha256: str
    device: int
    inode: int
    uid: int

    @property
    def lease_id(self) -> str:
        return str(self.metadata.get("lease_id") or "")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _process_is_running(pid: int) -> bool:
    """Conservatively classify PID liveness without importing daemon code."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # An inconclusive OS probe must preserve the durable fence.
        return True
    return True


def _identity(body: Mapping[str, Any]) -> str:
    return content_identity(dict(body))


def _with_identity(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = _identity(result)
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _run_git(
    repo: Path,
    args: Sequence[str],
    *,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            text=text,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            "git_query_failed",
            command=list(args),
            error_type=type(exc).__name__,
        ) from exc


def _git_output(repo: Path, *args: str) -> str:
    result = _run_git(repo, args)
    if result.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "git_query_failed",
            command=list(args),
            returncode=result.returncode,
            stderr=str(result.stderr or "")[-2000:],
        )
    return str(result.stdout or "").strip()


def _resolve_commit(repo: Path, value: str) -> str:
    result = _run_git(repo, ["rev-parse", "--verify", f"{value}^{{commit}}"])
    if result.returncode != 0 or not str(result.stdout or "").strip():
        raise ProtectedRecoveryClearanceError(
            "commit_unresolvable",
            commit=value,
        )
    return str(result.stdout).strip()


def _read_lock(path: Path) -> _LockSnapshot:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ProtectedRecoveryClearanceError("lock_not_regular_file")
        if int(before.st_nlink) != 1:
            raise ProtectedRecoveryClearanceError("lock_hardlink_unsafe")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise ProtectedRecoveryClearanceError("lock_permissions_unsafe")
        if int(before.st_size) > 1024 * 1024:
            raise ProtectedRecoveryClearanceError("lock_oversized")
        raw = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ProtectedRecoveryClearanceError("lock_absent") from exc
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            "lock_unreadable",
            error_type=type(exc).__name__,
        ) from exc
    before_identity = (int(before.st_dev), int(before.st_ino))
    if before_identity != (int(after.st_dev), int(after.st_ino)):
        raise ProtectedRecoveryClearanceError("lock_changed_during_read")
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError("lock_malformed") from exc
    if not isinstance(metadata, dict):
        raise ProtectedRecoveryClearanceError("lock_malformed")
    return _LockSnapshot(
        path=path,
        metadata=metadata,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        device=before_identity[0],
        inode=before_identity[1],
        uid=int(before.st_uid),
    )


def _same_lock(left: _LockSnapshot, right: _LockSnapshot) -> bool:
    return bool(
        left.path == right.path
        and left.device == right.device
        and left.inode == right.inode
        and left.sha256 == right.sha256
        and left.raw == right.raw
        and left.metadata == right.metadata
    )


def _repo_binding(repo_root: Path) -> dict[str, str]:
    try:
        repo = repo_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtectedRecoveryClearanceError("repository_invalid") from exc
    if not repo.is_dir():
        raise ProtectedRecoveryClearanceError("repository_invalid")
    top = Path(_git_output(repo, "rev-parse", "--show-toplevel")).resolve()
    if top != repo:
        raise ProtectedRecoveryClearanceError(
            "worktree_root_mismatch",
            requested=str(repo),
            actual=str(top),
        )
    common = git_common_dir(repo).resolve()
    lock_path = checkout_mutation_lock_path(repo).resolve()
    expected_lock = (
        common / "implementation-main-merge.lock"
    ).resolve(strict=False)
    if lock_path != expected_lock:
        raise ProtectedRecoveryClearanceError("common_directory_lock_mismatch")
    return {
        "repo_root": str(repo),
        "worktree_root": str(top),
        "repository_id": checkout_repository_id(repo),
        "git_common_dir": str(common),
        "lock_path": str(lock_path),
    }


def _clearance_executor_identity() -> dict[str, Any]:
    """Bind reviews and receipts to the exact clearance implementation."""

    module_path = Path(__file__).resolve(strict=True)
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    module_repo = Path(
        _git_output(module_path.parent, "rev-parse", "--show-toplevel")
    ).resolve()
    try:
        relative_module = module_path.relative_to(module_repo).as_posix()
    except ValueError as exc:
        raise ProtectedRecoveryClearanceError(
            "clearance_executor_repository_mismatch"
        ) from exc
    head = _resolve_commit(module_repo, "HEAD")
    tree = _git_output(module_repo, "rev-parse", "HEAD^{tree}")
    blob = _git_output(module_repo, "hash-object", "--", relative_module)
    return {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "protected-recovery-clearance-executor@1"
        ),
        "module_path": str(module_path),
        "module_relative_path": relative_module,
        "module_sha256": module_sha256,
        "module_git_blob": blob,
        "implementation_repo_root": str(module_repo),
        "implementation_repository_id": checkout_repository_id(module_repo),
        "implementation_head": head,
        "implementation_tree": tree,
    }


def _validated_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProtectedRecoveryClearanceError("protected_paths_missing")
    paths: list[str] = []
    for item in value:
        path = str(item)
        parsed = PurePosixPath(path)
        if (
            not path
            or path != parsed.as_posix()
            or parsed.is_absolute()
            or ".." in parsed.parts
            or "\x00" in path
            or path in paths
        ):
            raise ProtectedRecoveryClearanceError(
                "protected_paths_invalid",
                path=path,
            )
        paths.append(path)
    return tuple(paths)


def _literal_pathspecs(paths: Sequence[str]) -> tuple[str, ...]:
    """Return top-level literal pathspecs for already-validated repo paths."""

    return tuple(f":(top,literal){path}" for path in paths)


def _journal_basis(
    binding: Mapping[str, str],
    lock: _LockSnapshot,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], dict[str, Any]]:
    metadata = lock.metadata
    if metadata.get("protected_recovery_required") is not True:
        raise ProtectedRecoveryClearanceError("protected_recovery_not_required")
    if str(metadata.get("protected_recovery_owner") or "") != (
        "implementation_supervisor"
    ):
        raise ProtectedRecoveryClearanceError("recovery_owner_mismatch")
    if str(metadata.get("kind") or "") != "merge":
        raise ProtectedRecoveryClearanceError("kind_mismatch")
    if str(metadata.get("owner_script") or "") != "implementation_supervisor.py":
        raise ProtectedRecoveryClearanceError("owner_script_mismatch")
    if not lock.lease_id:
        raise ProtectedRecoveryClearanceError("lease_id_missing")
    journal_root_text = str(
        metadata.get("worktree_root")
        or metadata.get("repo_root")
        or ""
    )
    if not journal_root_text:
        raise ProtectedRecoveryClearanceError("repository_invalid")
    try:
        journal_repo = Path(journal_root_text).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError("repository_invalid") from exc
    if journal_repo != Path(binding["repo_root"]):
        raise ProtectedRecoveryClearanceError(
            "repository_mismatch",
            journal_repo_root=str(journal_repo),
        )
    for field_name, expected in (
        ("worktree_root", binding["worktree_root"]),
        ("repository_id", binding["repository_id"]),
    ):
        actual = str(metadata.get(field_name) or "")
        if actual and actual != expected:
            raise ProtectedRecoveryClearanceError(f"{field_name}_mismatch")
    for field_name in ("state_dir", "state_path"):
        value = str(metadata.get(field_name) or "")
        if not value:
            continue
        try:
            candidate = Path(value).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProtectedRecoveryClearanceError(
                f"{field_name}_invalid"
            ) from exc
        if not _is_relative_to(candidate, Path(binding["repo_root"])):
            raise ProtectedRecoveryClearanceError(f"{field_name}_mismatch")

    paths = _validated_paths(metadata.get("protected_paths"))
    guard = metadata.get("protected_release_guard")
    if not isinstance(guard, Mapping):
        raise ProtectedRecoveryClearanceError("guard_missing")
    guard = dict(guard)
    unsigned_guard = dict(guard)
    guard_id = str(unsigned_guard.pop("guard_id", "") or "")
    if not guard_id or _identity(unsigned_guard) != guard_id:
        raise ProtectedRecoveryClearanceError("guard_identity_mismatch")
    if tuple(_validated_paths(guard.get("protected_paths"))) != paths:
        raise ProtectedRecoveryClearanceError("guard_paths_mismatch")
    if list(guard.get("discovery_errors") or []):
        raise ProtectedRecoveryClearanceError(
            "protected_generated_scope_discovery_failed"
        )

    intent = metadata.get("protected_recovery_intent")
    if not isinstance(intent, Mapping):
        raise ProtectedRecoveryClearanceError("intent_missing")
    intent = dict(intent)
    if str(intent.get("schema") or "") != (
        "ipfs_accelerate_py.agent_supervisor."
        "supervisor-protected-recovery-intent@1"
    ):
        raise ProtectedRecoveryClearanceError("intent_schema_mismatch")
    unsigned_intent = dict(intent)
    intent_id = str(unsigned_intent.pop("intent_id", "") or "")
    if not intent_id or _identity(unsigned_intent) != intent_id:
        raise ProtectedRecoveryClearanceError("intent_identity_mismatch")
    if tuple(_validated_paths(intent.get("protected_paths"))) != paths:
        raise ProtectedRecoveryClearanceError("intent_paths_mismatch")
    if str(intent.get("guard_id") or "") != guard_id:
        raise ProtectedRecoveryClearanceError("intent_guard_mismatch")
    if not str(intent.get("operation") or "") or not str(
        intent.get("producer") or ""
    ):
        raise ProtectedRecoveryClearanceError("intent_operation_missing")

    scopes = guard.get("scopes")
    if not isinstance(scopes, list) or len(scopes) != 1:
        # The initial operator protocol deliberately handles only one exact
        # repository scope.  Nested repositories need their own review.
        raise ProtectedRecoveryClearanceError("guard_scope_unsupported")
    scope = scopes[0]
    if not isinstance(scope, Mapping):
        raise ProtectedRecoveryClearanceError("guard_scope_invalid")
    scope = dict(scope)
    try:
        scope_root = Path(str(scope.get("git_root") or "")).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError("guard_scope_invalid") from exc
    if scope_root != Path(binding["repo_root"]):
        raise ProtectedRecoveryClearanceError("guard_scope_root_mismatch")
    scope_paths = _validated_paths(scope.get("paths"))
    if tuple(sorted(scope_paths)) != tuple(sorted(paths)):
        raise ProtectedRecoveryClearanceError("guard_scope_paths_mismatch")
    before_query = scope.get("before_head_query")
    if not isinstance(before_query, Mapping) or before_query.get("ok") is not True:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_snapshot_failed"
        )
    before_head = str(scope.get("before_head") or "")
    if str(before_query.get("head") or "") != before_head or not before_head:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_snapshot_failed"
        )

    try:
        owner_pid = int(metadata.get("pid") or 0)
    except (TypeError, ValueError):
        owner_pid = 0
    if owner_pid <= 0:
        raise ProtectedRecoveryClearanceError("recovery_owner_pid_invalid")
    if _process_is_running(owner_pid):
        raise ProtectedRecoveryClearanceError(
            "supervisor_protected_recovery_owner_active",
            owner_pid=owner_pid,
        )
    return guard, intent, paths, scope


def _object_format(repo: Path) -> tuple[str, int]:
    value = _git_output(repo, "rev-parse", "--show-object-format")
    if value == "sha1":
        return value, 40
    if value == "sha256":
        return value, 64
    raise ProtectedRecoveryClearanceError(
        "repository_object_format_unsupported",
        object_format=value,
    )


def _protected_delta(
    repo: Path,
    commit: str,
    paths: Sequence[str],
    *,
    parent: str | None,
    parent_in_scope: bool,
) -> dict[str, Any]:
    prefix = [
        "diff-tree",
        "--no-commit-id",
        "--no-renames",
        "-r",
    ]
    if parent is None:
        comparison = ["--root", commit]
    else:
        comparison = [parent, commit]
    changed = _run_git(
        repo,
        [
            *prefix,
            "--name-only",
            "-z",
            *comparison,
            "--",
            *_literal_pathspecs(paths),
        ],
        text=False,
    )
    patch = _run_git(
        repo,
        [
            *prefix,
            "-p",
            "--binary",
            "--no-ext-diff",
            *comparison,
            "--",
            *_literal_pathspecs(paths),
        ],
        text=False,
    )
    if changed.returncode != 0 or patch.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_query_failed"
        )
    changed_paths = [
        item.decode("utf-8", errors="surrogateescape")
        for item in bytes(changed.stdout or b"").split(b"\x00")
        if item
    ]
    unexpected = sorted(set(changed_paths) - set(paths))
    if unexpected:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_path_escape",
            unexpected_paths=unexpected,
        )
    parent_tree = (
        _git_output(repo, "rev-parse", f"{parent}^{{tree}}")
        if parent is not None
        else ""
    )
    commit_tree = _git_output(repo, "rev-parse", f"{commit}^{{tree}}")
    return {
        "parent": parent or "",
        "parent_in_scope": parent_in_scope,
        "parent_tree": parent_tree,
        "commit_tree": commit_tree,
        "changed": bool(changed_paths),
        "changed_protected_paths": [
            path for path in paths if path in set(changed_paths)
        ],
        "protected_patch_sha256": hashlib.sha256(
            bytes(patch.stdout or b"")
        ).hexdigest(),
    }


def _commit_record(
    repo: Path,
    commit: str,
    paths: Sequence[str],
    *,
    before_head: str,
) -> dict[str, Any]:
    fields = _git_output(
        repo,
        "show",
        "-s",
        "--format=%H%x00%P%x00%T%x00%an%x00%ae%x00%s",
        commit,
    ).split("\x00", 5)
    if len(fields) != 6 or fields[0] != commit:
        raise ProtectedRecoveryClearanceError("protected_history_malformed")
    parents = fields[1].split() if fields[1] else []
    parent_deltas: list[dict[str, Any]] = []
    for parent in parents or [None]:
        if parent is None:
            parent_in_scope = False
        elif parent == before_head:
            parent_in_scope = True
        else:
            ancestry = _run_git(
                repo,
                ["merge-base", "--is-ancestor", before_head, parent],
            )
            if ancestry.returncode not in (0, 1):
                raise ProtectedRecoveryClearanceError(
                    "protected_generated_history_query_failed"
                )
            parent_in_scope = ancestry.returncode == 0
        parent_deltas.append(
            _protected_delta(
                repo,
                commit,
                paths,
                parent=parent,
                parent_in_scope=parent_in_scope,
            )
        )
    changed_path_set = {
        path
        for delta in parent_deltas
        for path in delta["changed_protected_paths"]
    }
    changed_paths = [path for path in paths if path in changed_path_set]
    author_email = fields[4]
    subject = fields[5]
    protected_relevant = any(
        delta["changed"] is True and delta["parent_in_scope"] is True
        for delta in parent_deltas
    )
    # Merge commits require explicit operator review whenever any parent edge
    # changes the protected projection.  Generator provenance is intentionally
    # not treated as sufficient merge-resolution authority.
    trusted_generator = bool(
        protected_relevant
        and len(parents) <= 1
        and author_email == BACKLOG_REFINERY_AUTHOR_EMAIL
        and subject.endswith(GENERATED_PROTECTED_BOARD_COMMIT_MARKER)
    )
    return {
        "commit": commit,
        "parents": parents,
        "tree": fields[2],
        "author_name": fields[3],
        "author_email": author_email,
        "subject": subject,
        "protected_relevant": protected_relevant,
        "trusted_generator": trusted_generator,
        "changed_protected_paths": changed_paths,
        "protected_parent_deltas": parent_deltas,
        "protected_patch_sha256": hashlib.sha256(
            json.dumps(
                parent_deltas,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _inspect_valid_journal(
    repo: Path,
    binding: Mapping[str, str],
    lock: _LockSnapshot,
    *,
    confirm_lock: bool,
) -> dict[str, Any]:
    guard, intent, paths, scope = _journal_basis(binding, lock)
    before_head = _resolve_commit(repo, str(scope["before_head"]))
    if before_head != str(scope["before_head"]):
        raise ProtectedRecoveryClearanceError("before_head_not_full_oid")
    before_tree = _git_output(repo, "rev-parse", f"{before_head}^{{tree}}")

    current_head = _resolve_commit(repo, "HEAD")
    current_tree = _git_output(repo, "rev-parse", "HEAD^{tree}")
    status = _run_git(
        repo,
        ["status", "--porcelain", "--untracked-files=all"],
    )
    if status.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "repository_status_query_failed"
        )
    status_text = str(status.stdout or "")
    if status_text.strip():
        raise ProtectedRecoveryClearanceError(
            "repository_dirty",
            status=status_text,
        )
    ancestry = _run_git(
        repo,
        ["merge-base", "--is-ancestor", before_head, current_head],
    )
    if ancestry.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_rewritten"
        )

    history_query = _run_git(
        repo,
        [
            "rev-list",
            "--ancestry-path",
            "--topo-order",
            f"{before_head}..{current_head}",
        ],
    )
    if history_query.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_query_failed"
        )
    history_oids = [
        line.strip()
        for line in str(history_query.stdout or "").splitlines()
        if line.strip()
    ]
    range_records = [
        _commit_record(
            repo,
            oid,
            paths,
            before_head=before_head,
        )
        for oid in history_oids
    ]
    history = [
        record
        for record in range_records
        if record["protected_relevant"] is True
    ]
    aggregate_names = _run_git(
        repo,
        [
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            before_head,
            current_head,
            "--",
            *_literal_pathspecs(paths),
        ],
        text=False,
    )
    aggregate_patch = _run_git(
        repo,
        [
            "diff",
            "--no-renames",
            "--binary",
            "--no-ext-diff",
            before_head,
            current_head,
            "--",
            *_literal_pathspecs(paths),
        ],
        text=False,
    )
    if aggregate_names.returncode != 0 or aggregate_patch.returncode != 0:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_query_failed"
        )
    aggregate_changed_set = {
        item.decode("utf-8", errors="surrogateescape")
        for item in bytes(aggregate_names.stdout or b"").split(b"\x00")
        if item
    }
    unexpected_aggregate = sorted(aggregate_changed_set - set(paths))
    if unexpected_aggregate:
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_path_escape",
            unexpected_paths=unexpected_aggregate,
        )
    history_changed_set = {
        path
        for record in history
        for path in record["changed_protected_paths"]
    }
    if not aggregate_changed_set.issubset(history_changed_set):
        raise ProtectedRecoveryClearanceError(
            "protected_generated_history_missing_commit",
            unexplained_paths=sorted(
                aggregate_changed_set - history_changed_set
            ),
        )
    aggregate_delta = {
        "changed_protected_paths": [
            path for path in paths if path in aggregate_changed_set
        ],
        "protected_patch_sha256": hashlib.sha256(
            bytes(aggregate_patch.stdout or b"")
        ).hexdigest(),
    }
    range_id = _identity(
        {
            "order": "newest_first_topological",
            "oids": history_oids,
            "records": range_records,
        }
    )
    history_id = _identity(
        {
            "order": "newest_first_topological",
            "range_id": range_id,
            "range_oids": history_oids,
            "protected_records": history,
            "aggregate_protected_delta": aggregate_delta,
        }
    )
    untrusted = [
        str(item["commit"])
        for item in history
        if item.get("trusted_generator") is not True
    ]
    reason = (
        "protected_generated_history_untrusted"
        if untrusted
        else (
            "protected_generated_history_trusted"
            if history
            else "protected_outputs_clean_history_unchanged"
        )
    )

    confirmed_head = _resolve_commit(repo, "HEAD")
    confirmed_tree = _git_output(repo, "rev-parse", "HEAD^{tree}")
    confirmed_status = _run_git(
        repo,
        ["status", "--porcelain", "--untracked-files=all"],
    )
    if (
        confirmed_status.returncode != 0
        or str(confirmed_status.stdout or "").strip()
        or confirmed_head != current_head
        or confirmed_tree != current_tree
    ):
        raise ProtectedRecoveryClearanceError(
            "protected_generated_release_state_changed"
        )
    if confirm_lock:
        confirmed_lock = _read_lock(lock.path)
        if not _same_lock(lock, confirmed_lock):
            raise ProtectedRecoveryClearanceError("lock_changed_during_review")

    object_format, oid_length = _object_format(repo)
    body: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "eligible": reason == "protected_generated_history_untrusted",
        "reason": reason,
        "decision_scope": DECISION,
        "authority": dict(NON_AUTHORITY),
        "security_model": SECURITY_MODEL,
        "clearance_executor": _clearance_executor_identity(),
        **dict(binding),
        "lock_sha256": lock.sha256,
        "lock_device": lock.device,
        "lock_inode": lock.inode,
        "lock_uid": lock.uid,
        "lease_id": lock.lease_id,
        "journal_repo_root": str(lock.metadata.get("repo_root") or ""),
        "journal_state_dir": str(lock.metadata.get("state_dir") or ""),
        "journal_state_path": str(lock.metadata.get("state_path") or ""),
        "guard_id": str(guard.get("guard_id") or ""),
        "intent_id": str(intent.get("intent_id") or ""),
        "journal_operation": str(lock.metadata.get("operation") or ""),
        "intent_operation": str(intent.get("operation") or ""),
        "intent_producer": str(intent.get("producer") or ""),
        "protected_paths": list(paths),
        "before_head": before_head,
        "before_tree": before_tree,
        "current_head": current_head,
        "current_tree": current_tree,
        "confirmed_head": confirmed_head,
        "confirmed_tree": confirmed_tree,
        "repository_status_clean": True,
        "ancestry_proven": True,
        "history_order": "newest_first_topological",
        "range_oids": history_oids,
        "range_records": range_records,
        "range_id": range_id,
        "history": history,
        "history_id": history_id,
        "aggregate_protected_delta": aggregate_delta,
        "untrusted_commits": untrusted,
        "object_format": object_format,
        "full_oid_length": oid_length,
    }
    return _with_identity(body, "review_id")


def inspect_protected_recovery(repo_root: Path) -> dict[str, Any]:
    """Return a read-only, content-addressed review of the exact journal."""

    try:
        binding = _repo_binding(repo_root)
        lock = _read_lock(Path(binding["lock_path"]))
        return _inspect_valid_journal(
            Path(binding["repo_root"]),
            binding,
            lock,
            confirm_lock=True,
        )
    except ProtectedRecoveryClearanceError as exc:
        body = {
            "schema": REVIEW_SCHEMA,
            "eligible": False,
            "reason": exc.reason,
            "decision_scope": DECISION,
            "authority": dict(NON_AUTHORITY),
            "security_model": SECURITY_MODEL,
            "details": exc.details,
        }
        return _with_identity(body, "review_id")


def _validate_full_approvals(
    repo: Path,
    approvals: Sequence[str],
    *,
    expected: Sequence[str],
    oid_length: int,
) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in approvals)
    pattern = re.compile(rf"[0-9a-f]{{{oid_length}}}")
    if not normalized or any(
        not pattern.fullmatch(item) for item in normalized
    ):
        raise ProtectedRecoveryClearanceError(
            "approved_commit_not_full_oid"
        )
    if len(set(normalized)) != len(normalized):
        raise ProtectedRecoveryClearanceError("approved_commit_duplicate")
    for item in normalized:
        if _resolve_commit(repo, item) != item:
            raise ProtectedRecoveryClearanceError(
                "approved_commit_not_full_oid",
                commit=item,
            )
    if normalized != tuple(expected):
        raise ProtectedRecoveryClearanceError(
            "operator_commit_approval_mismatch",
            required=list(expected),
            supplied=list(normalized),
        )
    return normalized


def _prepare_receipt_dir(
    receipt_dir: Path,
    *,
    repo: Path,
    common_dir: Path,
) -> Path:
    if not receipt_dir.is_absolute():
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_must_be_absolute"
        )
    resolved = receipt_dir.resolve(strict=False)
    if _is_relative_to(resolved, repo) or _is_relative_to(
        resolved, common_dir
    ):
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_inside_repository"
        )
    resolved.mkdir(parents=True, mode=0o700, exist_ok=True)
    resolved = resolved.resolve(strict=True)
    metadata = resolved.stat(follow_symlinks=False)
    current_uid = getattr(os, "geteuid", lambda: -1)()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_not_directory"
        )
    if int(metadata.st_uid) != current_uid:
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_owner_mismatch"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_permissions_unsafe"
        )
    if _is_relative_to(resolved, repo) or _is_relative_to(
        resolved, common_dir
    ):
        raise ProtectedRecoveryClearanceError(
            "receipt_directory_inside_repository"
        )
    return resolved


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = (
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        _repair_published_receipt_hardlink(path)
        if path.is_file() and path.read_bytes() == rendered:
            return dict(payload)
        raise ProtectedRecoveryClearanceError(
            "receipt_identity_collision",
            receipt_path=str(path),
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".pending",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(rendered):
            written = os.write(descriptor, rendered[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != rendered:
                raise ProtectedRecoveryClearanceError(
                    "receipt_identity_collision",
                    receipt_path=str(path),
                ) from None
        _fsync_directory(path.parent)
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return dict(payload)


def _repair_published_receipt_hardlink(path: Path) -> None:
    """Finish a receipt publication interrupted after its no-clobber link."""

    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            "receipt_publication_recovery_failed"
        ) from exc
    if int(info.st_nlink) == 1:
        return
    if (
        not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 2
        or int(info.st_uid) != getattr(os, "geteuid", lambda: -1)()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise ProtectedRecoveryClearanceError(
            "receipt_publication_recovery_failed"
        )
    candidates: list[Path] = []
    prefix = f".{path.name}."
    for candidate in path.parent.iterdir():
        if not (
            candidate.name.startswith(prefix)
            and candidate.name.endswith(".pending")
        ):
            continue
        try:
            candidate_info = candidate.stat(follow_symlinks=False)
        except OSError:
            continue
        if (
            int(candidate_info.st_dev) == int(info.st_dev)
            and int(candidate_info.st_ino) == int(info.st_ino)
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ProtectedRecoveryClearanceError(
            "receipt_publication_recovery_failed"
        )
    candidates[0].unlink()
    _fsync_directory(path.parent)
    repaired = path.stat(follow_symlinks=False)
    if int(repaired.st_nlink) != 1:
        raise ProtectedRecoveryClearanceError(
            "receipt_publication_recovery_failed"
        )


def _receipt_path(root: Path, kind: str, identity: str) -> Path:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return root / f"protected-recovery-{kind}-{digest}.json"


def _validated_receipt(
    path: Path,
    *,
    root: Path,
    kind: str,
    schema: str,
    identity_field: str,
    timestamp_field: str | None,
) -> dict[str, Any]:
    _repair_published_receipt_hardlink(path)
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            f"{kind}_receipt_invalid"
        ) from exc
    current_uid = getattr(os, "geteuid", lambda: -1)()
    if (
        not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
        or int(info.st_uid) != current_uid
        or stat.S_IMODE(info.st_mode) != 0o600
        or int(info.st_size) > 1024 * 1024
    ):
        raise ProtectedRecoveryClearanceError(f"{kind}_receipt_invalid")
    try:
        before_identity = (int(info.st_dev), int(info.st_ino))
        value = json.loads(path.read_text(encoding="utf-8"))
        after = path.stat(follow_symlinks=False)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError(
            f"{kind}_receipt_invalid"
        ) from exc
    if before_identity != (int(after.st_dev), int(after.st_ino)):
        raise ProtectedRecoveryClearanceError(f"{kind}_receipt_invalid")
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ProtectedRecoveryClearanceError(f"{kind}_receipt_invalid")
    unsigned = dict(value)
    identity = str(unsigned.pop(identity_field, "") or "")
    if timestamp_field is not None:
        timestamp = unsigned.pop(timestamp_field, None)
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ProtectedRecoveryClearanceError(
                f"{kind}_receipt_invalid"
            )
    if not identity or _identity(unsigned) != identity:
        raise ProtectedRecoveryClearanceError(f"{kind}_receipt_invalid")
    if path.resolve() != _receipt_path(root, kind, identity).resolve():
        raise ProtectedRecoveryClearanceError(f"{kind}_receipt_invalid")
    return value


def _matching_receipts(
    root: Path,
    *,
    kind: str,
    schema: str,
    identity_field: str,
    timestamp_field: str | None,
    required: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    matches: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(root.glob(f"protected-recovery-{kind}-*.json")):
        receipt = _validated_receipt(
            path,
            root=root,
            kind=kind,
            schema=schema,
            identity_field=identity_field,
            timestamp_field=timestamp_field,
        )
        if all(receipt.get(key) == value for key, value in required.items()):
            matches.append((receipt, path))
    return matches


def _restoration_snapshot(
    repo: Path,
    binding: Mapping[str, str],
    raw: bytes,
    *,
    expected_lease_id: str,
    expected_lock_sha256: str,
    approved_commits: Sequence[str],
) -> tuple[_LockSnapshot, dict[str, Any], tuple[str, ...]]:
    """Validate exact recovered journal bytes without publishing them."""

    if not raw or len(raw) > 1024 * 1024:
        raise ProtectedRecoveryClearanceError("restoration_snapshot_size_invalid")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_lock_sha256:
        raise ProtectedRecoveryClearanceError(
            "restoration_snapshot_digest_mismatch",
            observed_sha256=observed_sha256,
        )
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError(
            "restoration_snapshot_malformed"
        ) from exc
    if not isinstance(metadata, dict):
        raise ProtectedRecoveryClearanceError("restoration_snapshot_malformed")
    if str(metadata.get("lease_id") or "") != expected_lease_id:
        raise ProtectedRecoveryClearanceError(
            "restoration_snapshot_lease_mismatch"
        )
    snapshot = _LockSnapshot(
        path=Path(binding["lock_path"]),
        metadata=dict(metadata),
        raw=raw,
        sha256=observed_sha256,
        device=0,
        inode=0,
        uid=getattr(os, "geteuid", lambda: -1)(),
    )
    review = _inspect_valid_journal(
        repo,
        binding,
        snapshot,
        confirm_lock=False,
    )
    if review.get("eligible") is not True or review.get("reason") != (
        "protected_generated_history_untrusted"
    ):
        raise ProtectedRecoveryClearanceError(
            "restoration_snapshot_not_authorizable",
            reason=review.get("reason"),
        )
    approvals = _validate_full_approvals(
        repo,
        approved_commits,
        expected=review["untrusted_commits"],
        oid_length=int(review["full_oid_length"]),
    )
    return snapshot, review, approvals


def _validated_historical_source_review(
    value: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    snapshot_review: Mapping[str, Any],
    operator_uid: int,
) -> dict[str, Any]:
    """Validate the captured pre-disappearance review as exact provenance."""

    if not isinstance(value, Mapping):
        raise ProtectedRecoveryClearanceError("source_review_invalid")
    review = dict(value)
    unsigned = dict(review)
    review_id = str(unsigned.pop("review_id", "") or "")
    if not review_id or _identity(unsigned) != review_id:
        raise ProtectedRecoveryClearanceError(
            "source_review_identity_mismatch"
        )
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("eligible") is not True
        or review.get("reason")
        != "protected_generated_history_untrusted"
        or review.get("decision_scope") != DECISION
        or dict(review.get("authority") or {}) != NON_AUTHORITY
        or review.get("security_model") != SECURITY_MODEL
        or review.get("repository_status_clean") is not True
        or review.get("ancestry_proven") is not True
    ):
        raise ProtectedRecoveryClearanceError("source_review_invalid")
    if int(review.get("lock_uid") or -1) != operator_uid:
        raise ProtectedRecoveryClearanceError("source_review_owner_mismatch")
    if int(review.get("lock_device") or 0) <= 0 or int(
        review.get("lock_inode") or 0
    ) <= 0:
        raise ProtectedRecoveryClearanceError("source_review_lock_invalid")
    expected_fields = {
        "repo_root": binding["repo_root"],
        "worktree_root": binding["worktree_root"],
        "repository_id": binding["repository_id"],
        "git_common_dir": binding["git_common_dir"],
        "lock_path": binding["lock_path"],
        "lock_sha256": snapshot_review["lock_sha256"],
        "lease_id": snapshot_review["lease_id"],
        "guard_id": snapshot_review["guard_id"],
        "intent_id": snapshot_review["intent_id"],
        "protected_paths": snapshot_review["protected_paths"],
        "before_head": snapshot_review["before_head"],
        "before_tree": snapshot_review["before_tree"],
        "current_head": snapshot_review["current_head"],
        "current_tree": snapshot_review["current_tree"],
        "confirmed_head": snapshot_review["confirmed_head"],
        "confirmed_tree": snapshot_review["confirmed_tree"],
        "untrusted_commits": snapshot_review["untrusted_commits"],
        "object_format": snapshot_review["object_format"],
        "full_oid_length": snapshot_review["full_oid_length"],
    }
    mismatches = sorted(
        field
        for field, expected in expected_fields.items()
        if review.get(field) != expected
    )
    if mismatches:
        raise ProtectedRecoveryClearanceError(
            "source_review_binding_mismatch",
            mismatched_fields=mismatches,
        )
    if not isinstance(review.get("history"), list) or not str(
        review.get("history_id") or ""
    ):
        raise ProtectedRecoveryClearanceError("source_review_history_invalid")
    history_order = str(review.get("history_order") or "")
    if history_order == "newest_first":
        expected_history_id = _identity(
            {"order": history_order, "items": review["history"]}
        )
    elif history_order == "newest_first_topological":
        range_oids = review.get("range_oids")
        range_id = str(review.get("range_id") or "")
        aggregate_delta = review.get("aggregate_protected_delta")
        if (
            not isinstance(range_oids, list)
            or not range_id
            or not isinstance(aggregate_delta, Mapping)
        ):
            raise ProtectedRecoveryClearanceError(
                "source_review_history_invalid"
            )
        expected_history_id = _identity(
            {
                "order": history_order,
                "range_id": range_id,
                "range_oids": range_oids,
                "protected_records": review["history"],
                "aggregate_protected_delta": dict(aggregate_delta),
            }
        )
    else:
        raise ProtectedRecoveryClearanceError(
            "source_review_history_invalid"
        )
    if str(review["history_id"]) != expected_history_id:
        raise ProtectedRecoveryClearanceError(
            "source_review_history_invalid"
        )
    return review


def _validated_operator_event_id(value: str) -> str:
    event_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", event_id):
        raise ProtectedRecoveryClearanceError("operator_event_id_invalid")
    return event_id


def _restoration_authorization_path(
    root: Path,
    basis: Mapping[str, Any],
) -> tuple[str, Path]:
    authorization_id = _identity(basis)
    return (
        authorization_id,
        _receipt_path(root, "restoration-authorization", authorization_id),
    )


def _load_or_write_restoration_authorization(
    root: Path,
    basis: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    authorization_id, path = _restoration_authorization_path(root, basis)
    if path.exists():
        existing = _validated_receipt(
            path,
            root=root,
            kind="restoration-authorization",
            schema=RESTORATION_AUTHORIZATION_SCHEMA,
            identity_field="restoration_authorization_id",
            timestamp_field="authorized_at",
        )
        unsigned = dict(existing)
        observed_id = str(
            unsigned.pop("restoration_authorization_id", "") or ""
        )
        unsigned.pop("authorized_at", None)
        if observed_id != authorization_id or unsigned != dict(basis):
            raise ProtectedRecoveryClearanceError(
                "restoration_authorization_receipt_invalid"
            )
        return existing, path
    receipt = {
        **dict(basis),
        "authorized_at": _utc_now(),
        "restoration_authorization_id": authorization_id,
    }
    _durable_receipt(path, receipt)
    return receipt, path


def _finalize_restoration_receipt(
    root: Path,
    authorization: Mapping[str, Any],
    restored: _LockSnapshot,
) -> tuple[dict[str, Any], Path]:
    basis = {
        "schema": RESTORATION_RECEIPT_SCHEMA,
        "restoration_authorization_id": str(
            authorization["restoration_authorization_id"]
        ),
        "source_review_id": str(authorization["source_review_id"]),
        "operator_event_id": str(authorization["operator_event_id"]),
        "snapshot_review_id": str(authorization["snapshot_review_id"]),
        "restored_lease_id": restored.lease_id,
        "restored_lock_sha256": restored.sha256,
        "restored_exact_snapshot": True,
        "fence_present": True,
        "decision": "restore_exact_missing_protected_recovery_fence",
        "authority": dict(NON_AUTHORITY),
        "security_model": SECURITY_MODEL,
        "clearance_executor": dict(authorization["clearance_executor"]),
    }
    receipt_id = _identity(basis)
    path = _receipt_path(root, "restoration-final", receipt_id)
    if path.exists():
        existing = _validated_receipt(
            path,
            root=root,
            kind="restoration-final",
            schema=RESTORATION_RECEIPT_SCHEMA,
            identity_field="restoration_receipt_id",
            timestamp_field="restored_at",
        )
        unsigned = dict(existing)
        observed_id = str(unsigned.pop("restoration_receipt_id", "") or "")
        unsigned.pop("restored_at", None)
        if observed_id != receipt_id or unsigned != basis:
            raise ProtectedRecoveryClearanceError(
                "restoration_receipt_invalid"
            )
        return existing, path
    receipt = {
        **basis,
        "restored_at": _utc_now(),
        "restoration_receipt_id": receipt_id,
    }
    _durable_receipt(path, receipt)
    return receipt, path


def _restoration_final_path(
    root: Path,
    authorization: Mapping[str, Any],
    snapshot: _LockSnapshot,
) -> Path:
    basis = {
        "schema": RESTORATION_RECEIPT_SCHEMA,
        "restoration_authorization_id": str(
            authorization["restoration_authorization_id"]
        ),
        "source_review_id": str(authorization["source_review_id"]),
        "operator_event_id": str(authorization["operator_event_id"]),
        "snapshot_review_id": str(authorization["snapshot_review_id"]),
        "restored_lease_id": snapshot.lease_id,
        "restored_lock_sha256": snapshot.sha256,
        "restored_exact_snapshot": True,
        "fence_present": True,
        "decision": "restore_exact_missing_protected_recovery_fence",
        "authority": dict(NON_AUTHORITY),
        "security_model": SECURITY_MODEL,
        "clearance_executor": dict(authorization["clearance_executor"]),
    }
    return _receipt_path(root, "restoration-final", _identity(basis))


def _restoration_test_checkpoint(_phase: str) -> None:
    """Private fault-injection checkpoint used only by focused tests."""


def _read_exact_staging_file(
    path: Path,
    *,
    expected: bytes,
    allowed_links: frozenset[int],
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_uid) != getattr(os, "geteuid", lambda: -1)()
            or stat.S_IMODE(before.st_mode) != 0o600
            or int(before.st_nlink) not in allowed_links
            or int(before.st_size) != len(expected)
        ):
            raise ProtectedRecoveryClearanceError(
                "restoration_staging_file_invalid"
            )
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        (int(before.st_dev), int(before.st_ino), int(before.st_size))
        != (int(after.st_dev), int(after.st_ino), int(after.st_size))
        or raw != expected
    ):
        raise ProtectedRecoveryClearanceError(
            "restoration_staging_file_invalid"
        )
    return raw, after


def _discard_incomplete_restoration_staging(path: Path) -> None:
    """Remove only a safe, owner-private, single-link staging remnant."""

    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            "restoration_staging_file_invalid"
        ) from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
        or int(info.st_uid) != getattr(os, "geteuid", lambda: -1)()
        or stat.S_IMODE(info.st_mode) != 0o600
        or int(info.st_size) > 1024 * 1024
    ):
        raise ProtectedRecoveryClearanceError(
            "restoration_staging_file_invalid"
        )
    path.unlink()
    _fsync_directory(path.parent)


def _publish_restored_fence_no_clobber(
    lock_path: Path,
    raw: bytes,
    *,
    restoration_authorization_id: str,
) -> _LockSnapshot:
    """Publish exact journal bytes without ever exposing a partial fence."""

    suffix = hashlib.sha256(
        restoration_authorization_id.encode("utf-8")
    ).hexdigest()
    staging = lock_path.with_name(
        f".{lock_path.name}.{suffix}.restoring"
    )
    try:
        lock_info = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        lock_info = None
    if lock_info is not None:
        try:
            staging_info = staging.stat(follow_symlinks=False)
        except FileNotFoundError:
            staging_info = None
        if staging_info is not None and (
            int(staging_info.st_dev), int(staging_info.st_ino)
        ) == (int(lock_info.st_dev), int(lock_info.st_ino)):
            _read_exact_staging_file(
                staging,
                expected=raw,
                allowed_links=frozenset({2}),
            )
            staging.unlink()
            _fsync_directory(lock_path.parent)
            return _read_lock(lock_path)
        raise ProtectedRecoveryClearanceError(
            "restoration_lock_already_present"
        )

    created_here = False
    try:
        try:
            _read_exact_staging_file(
                staging,
                expected=raw,
                allowed_links=frozenset({1}),
            )
        except FileNotFoundError:
            pass
        except ProtectedRecoveryClearanceError as exc:
            if exc.reason != "restoration_staging_file_invalid":
                raise
            _discard_incomplete_restoration_staging(staging)
        if not staging.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(staging, flags, 0o600)
            except FileExistsError as exc:
                raise ProtectedRecoveryClearanceError(
                    "restoration_staging_race"
                ) from exc
            created_here = True
            try:
                os.fchmod(descriptor, 0o600)
                _restoration_test_checkpoint("after_temp_open")
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        raise OSError("short fence restoration write")
                    offset += written
                    _restoration_test_checkpoint("during_temp_write")
                os.fsync(descriptor)
                _restoration_test_checkpoint("after_temp_fsync")
            finally:
                os.close(descriptor)
            _read_exact_staging_file(
                staging,
                expected=raw,
                allowed_links=frozenset({1}),
            )
        try:
            os.link(staging, lock_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProtectedRecoveryClearanceError(
                "restoration_lock_race"
            ) from exc
        _restoration_test_checkpoint("after_publish_before_directory_fsync")
        _fsync_directory(lock_path.parent)
        staging.unlink()
        _fsync_directory(lock_path.parent)
        restored = _read_lock(lock_path)
        if restored.raw != raw or restored.sha256 != hashlib.sha256(raw).hexdigest():
            raise ProtectedRecoveryClearanceError(
                "restoration_publication_unverified"
            )
        return restored
    finally:
        try:
            staging_info = staging.stat(follow_symlinks=False)
        except FileNotFoundError:
            staging_info = None
        if staging_info is not None:
            try:
                lock_info = lock_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                lock_info = None
            if lock_info is not None and (
                int(staging_info.st_dev), int(staging_info.st_ino)
            ) == (int(lock_info.st_dev), int(lock_info.st_ino)):
                staging.unlink()
                _fsync_directory(lock_path.parent)
            elif created_here:
                staging.unlink()
                _fsync_directory(lock_path.parent)


def restore_missing_protected_recovery_fence(
    repo_root: Path,
    *,
    receipt_dir: Path,
    snapshot_bytes: bytes,
    expected_lease_id: str,
    expected_lock_sha256: str,
    source_review: Mapping[str, Any],
    operator_event_id: str,
    approved_commits: Sequence[str],
    operator_identity: str,
    operator_note: str,
) -> dict[str, Any]:
    """Restore one exact disappeared fence before normal reviewed clearance."""

    identity = str(operator_identity).strip()
    note = str(operator_note).strip()
    event_id = _validated_operator_event_id(operator_event_id)
    if not identity:
        raise ProtectedRecoveryClearanceError("operator_identity_required")
    if not note:
        raise ProtectedRecoveryClearanceError("operator_note_required")
    if not expected_lease_id:
        raise ProtectedRecoveryClearanceError("expected_lease_id_required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_lock_sha256):
        raise ProtectedRecoveryClearanceError(
            "expected_lock_sha256_invalid"
        )
    operator_uid = getattr(os, "geteuid", lambda: -1)()
    if operator_uid < 0:
        raise ProtectedRecoveryClearanceError("os_owner_check_unavailable")

    binding = _repo_binding(repo_root)
    repo = Path(binding["repo_root"])
    common = Path(binding["git_common_dir"])
    root = _prepare_receipt_dir(
        receipt_dir,
        repo=repo,
        common_dir=common,
    )
    repo_stat = repo.stat()
    common_stat = common.stat()
    if int(repo_stat.st_uid) != operator_uid or int(common_stat.st_uid) != (
        operator_uid
    ):
        raise ProtectedRecoveryClearanceError("os_owner_mismatch")

    snapshot, review, approvals = _restoration_snapshot(
        repo,
        binding,
        bytes(snapshot_bytes),
        expected_lease_id=expected_lease_id,
        expected_lock_sha256=expected_lock_sha256,
        approved_commits=approved_commits,
    )
    historical_review = _validated_historical_source_review(
        source_review,
        binding=binding,
        snapshot_review=review,
        operator_uid=operator_uid,
    )
    basis = {
        "schema": RESTORATION_AUTHORIZATION_SCHEMA,
        "security_model": SECURITY_MODEL,
        "cryptographic_operator_signature": False,
        "operator_identity": identity,
        "operator_uid": operator_uid,
        "operator_note": note,
        "operator_event_id": event_id,
        "decision": "restore_exact_missing_protected_recovery_fence",
        "authority": dict(NON_AUTHORITY),
        "clearance_executor": dict(review["clearance_executor"]),
        "source_review_id": str(historical_review["review_id"]),
        "source_review": historical_review,
        "source_review_content_identity_verified": True,
        "source_review_cryptographically_authenticated": False,
        "snapshot_review_id": str(review["review_id"]),
        "snapshot_lock_sha256": snapshot.sha256,
        "snapshot_lease_id": snapshot.lease_id,
        "repo_root": binding["repo_root"],
        "worktree_root": binding["worktree_root"],
        "repository_id": binding["repository_id"],
        "git_common_dir": binding["git_common_dir"],
        "lock_path": binding["lock_path"],
        "guard_id": str(review["guard_id"]),
        "intent_id": str(review["intent_id"]),
        "protected_paths": list(review["protected_paths"]),
        "before_head": str(review["before_head"]),
        "reviewed_head": str(review["current_head"]),
        "reviewed_tree": str(review["current_tree"]),
        "history_id": str(review["history_id"]),
        "approved_commits": list(approvals),
        "unauthorized_absence_observed": True,
    }
    lock_path = Path(binding["lock_path"])
    with serialized_lock_update(lock_path, timeout_seconds=5.0):
        guarded_review = _inspect_valid_journal(
            repo,
            binding,
            snapshot,
            confirm_lock=False,
        )
        if str(guarded_review["review_id"]) != str(review["review_id"]):
            raise ProtectedRecoveryClearanceError(
                "restoration_repository_changed_before_authorization"
            )
        event_authorizations = _matching_receipts(
            root,
            kind="restoration-authorization",
            schema=RESTORATION_AUTHORIZATION_SCHEMA,
            identity_field="restoration_authorization_id",
            timestamp_field="authorized_at",
            required={"operator_event_id": event_id},
        )
        expected_authorization_id, expected_authorization_path = (
            _restoration_authorization_path(root, basis)
        )
        if event_authorizations and not (
            len(event_authorizations) == 1
            and event_authorizations[0][1] == expected_authorization_path
        ):
            raise ProtectedRecoveryClearanceError(
                "restoration_operator_event_reused"
            )
        authorization: dict[str, Any] | None = None
        authorization_path = expected_authorization_path
        if event_authorizations:
            authorization, authorization_path = event_authorizations[0]
            if str(authorization["restoration_authorization_id"]) != (
                expected_authorization_id
            ):
                raise ProtectedRecoveryClearanceError(
                    "restoration_authorization_receipt_invalid"
                )
        try:
            current = _read_lock(lock_path)
        except ProtectedRecoveryClearanceError as exc:
            if exc.reason == "lock_hardlink_unsafe" and authorization is not None:
                current = _publish_restored_fence_no_clobber(
                    lock_path,
                    snapshot.raw,
                    restoration_authorization_id=str(
                        authorization["restoration_authorization_id"]
                    ),
                )
            elif exc.reason != "lock_absent":
                raise
            else:
                current = None
        if current is not None and current.uid != operator_uid:
            raise ProtectedRecoveryClearanceError(
                "restoration_lock_owner_mismatch"
            )
        if current is not None and (
            current.raw != snapshot.raw or current.sha256 != snapshot.sha256
        ):
            raise ProtectedRecoveryClearanceError(
                "restoration_lock_already_present"
            )
        if authorization is None:
            if current is not None:
                raise ProtectedRecoveryClearanceError(
                    "restoration_lock_already_present"
                )
            authorization, authorization_path = (
                _load_or_write_restoration_authorization(root, basis)
            )
        if str(authorization["restoration_authorization_id"]) != (
            expected_authorization_id
        ):
            raise ProtectedRecoveryClearanceError(
                "restoration_authorization_receipt_invalid"
            )
        final_path = _restoration_final_path(root, authorization, snapshot)
        if final_path.exists():
            _finalize_restoration_receipt(root, authorization, snapshot)
            if current is None:
                raise ProtectedRecoveryClearanceError(
                    "restoration_repeat_absence_requires_new_operator_event"
                )
            restored = current
            idempotent = True
        elif current is not None:
            restored = current
            idempotent = True
        else:
            restored = _publish_restored_fence_no_clobber(
                lock_path,
                snapshot.raw,
                restoration_authorization_id=str(
                    authorization["restoration_authorization_id"]
                ),
            )
            idempotent = False
        fresh_review = _inspect_valid_journal(
            repo,
            binding,
            restored,
            confirm_lock=True,
        )
        receipt, receipt_path = _finalize_restoration_receipt(
            root,
            authorization,
            restored,
        )
    return {
        "restored": True,
        "idempotent_restoration": idempotent,
        "restored_exact_snapshot": True,
        "decision": "restore_exact_missing_protected_recovery_fence",
        "authority": dict(NON_AUTHORITY),
        "restoration_authorization_id": authorization[
            "restoration_authorization_id"
        ],
        "restoration_authorization_path": str(authorization_path),
        "restoration_receipt_id": receipt["restoration_receipt_id"],
        "restoration_receipt_path": str(receipt_path),
        "restored_lock_sha256": restored.sha256,
        "restored_lease_id": restored.lease_id,
        "fresh_review": fresh_review,
        "fresh_review_id": fresh_review["review_id"],
    }


def _authorization_basis(
    *,
    review: Mapping[str, Any],
    approvals: Sequence[str],
    operator_identity: str,
    operator_note: str,
    operator_uid: int,
) -> dict[str, Any]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "security_model": SECURITY_MODEL,
        "cryptographic_operator_signature": False,
        "operator_identity": operator_identity,
        "operator_uid": operator_uid,
        "operator_note": operator_note,
        "decision": DECISION,
        "authority": dict(NON_AUTHORITY),
        "clearance_executor": dict(review["clearance_executor"]),
        "review_id": str(review["review_id"]),
        "lock_sha256": str(review["lock_sha256"]),
        "expected_lease_id": str(review["lease_id"]),
        "repo_root": str(review["repo_root"]),
        "worktree_root": str(review["worktree_root"]),
        "repository_id": str(review["repository_id"]),
        "git_common_dir": str(review["git_common_dir"]),
        "lock_path": str(review["lock_path"]),
        "guard_id": str(review["guard_id"]),
        "intent_id": str(review["intent_id"]),
        "protected_paths": list(review["protected_paths"]),
        "before_head": str(review["before_head"]),
        "before_tree": str(review["before_tree"]),
        "reviewed_head": str(review["current_head"]),
        "reviewed_tree": str(review["current_tree"]),
        "history_id": str(review["history_id"]),
        "approved_commits": list(approvals),
    }


def _load_or_write_authorization(
    root: Path,
    basis: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    authorization_id = _identity(basis)
    path = _receipt_path(root, "authorization", authorization_id)
    if path.exists():
        existing = _validated_receipt(
            path,
            root=root,
            kind="authorization",
            schema=AUTHORIZATION_SCHEMA,
            identity_field="authorization_id",
            timestamp_field="authorized_at",
        )
        unsigned = dict(existing)
        observed_id = str(unsigned.pop("authorization_id", "") or "")
        unsigned.pop("authorized_at", None)
        if observed_id != authorization_id or unsigned != dict(basis):
            raise ProtectedRecoveryClearanceError(
                "authorization_receipt_invalid"
            )
        return existing, path
    receipt = {
        **dict(basis),
        "authorized_at": _utc_now(),
        "authorization_id": authorization_id,
    }
    _durable_receipt(path, receipt)
    return receipt, path


def _rotated_metadata(
    original: _LockSnapshot,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    marker = {
        "schema": ROTATED_LEASE_SCHEMA,
        "authorization_id": str(authorization["authorization_id"]),
        "review_id": str(authorization["review_id"]),
        "original_lease_id": original.lease_id,
        "original_lock_sha256": original.sha256,
        "decision": DECISION,
        "authority": dict(NON_AUTHORITY),
    }
    new_lease_id = _identity(
        {
            "kind": "operator-protected-recovery-clearance-lease",
            **marker,
        }
    )
    return {
        **dict(original.metadata),
        "lease_id": new_lease_id,
        # The authorization lease is deliberately process-independent.  A
        # crash between CAS rotation and release must be resumable by the
        # same exact authorization under a new executor PID.
        "pid": 0,
        "owner_script": Path(__file__).name,
        "protected_recovery_clearance_state": "authorized",
        "protected_recovery_clearance": marker,
        "protected_recovery_clearance_original_metadata": dict(
            original.metadata
        ),
    }


def _write_replacement(
    expected: _LockSnapshot,
    metadata: Mapping[str, Any],
) -> _LockSnapshot:
    with serialized_lock_update(expected.path, timeout_seconds=5.0):
        current = _read_lock(expected.path)
        if not _same_lock(expected, current):
            raise ProtectedRecoveryClearanceError("clearance_lock_cas_failed")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{expected.path.name}.",
            suffix=".clearance",
            dir=str(expected.path.parent),
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            rendered = json.dumps(
                dict(metadata), indent=2, sort_keys=True
            ).encode("utf-8")
            offset = 0
            while offset < len(rendered):
                written = os.write(descriptor, rendered[offset:])
                if written <= 0:
                    raise OSError("short lock write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, expected.path)
            _fsync_directory(expected.path.parent)
            published = _read_lock(expected.path)
            if published.metadata != dict(metadata):
                raise ProtectedRecoveryClearanceError(
                    "clearance_lock_rotation_unverified"
                )
            return published
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


def _original_snapshot_from_rotated(
    current: _LockSnapshot,
    *,
    expected_lease_id: str,
    expected_lock_sha256: str,
    authorization_id: str,
    expected_review_id: str,
) -> _LockSnapshot | None:
    marker = current.metadata.get("protected_recovery_clearance")
    original = current.metadata.get(
        "protected_recovery_clearance_original_metadata"
    )
    if not isinstance(marker, Mapping) or not isinstance(original, Mapping):
        return None
    if (
        str(marker.get("schema") or "") != ROTATED_LEASE_SCHEMA
        or str(marker.get("original_lease_id") or "") != expected_lease_id
        or str(marker.get("original_lock_sha256") or "")
        != expected_lock_sha256
        or str(marker.get("authorization_id") or "") != authorization_id
        or str(marker.get("review_id") or "") != expected_review_id
        or str(marker.get("decision") or "") != DECISION
        or dict(marker.get("authority") or {}) != NON_AUTHORITY
    ):
        return None
    original_dict = dict(original)
    if str(original_dict.get("lease_id") or "") != expected_lease_id:
        return None
    synthetic_raw = json.dumps(
        original_dict, indent=2, sort_keys=True
    ).encode("utf-8")
    return _LockSnapshot(
        path=current.path,
        metadata=original_dict,
        raw=synthetic_raw,
        sha256=expected_lock_sha256,
        device=int(
            current.metadata.get("protected_recovery_clearance_original_device")
            or 0
        ),
        inode=int(
            current.metadata.get("protected_recovery_clearance_original_inode")
            or 0
        ),
        uid=current.uid,
    )


def _release_intent_basis(
    authorization: Mapping[str, Any],
    rotated: _LockSnapshot,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RELEASE_INTENT_SCHEMA,
        "authorization_id": str(authorization["authorization_id"]),
        "review_id": str(review["review_id"]),
        "rotated_lease_id": rotated.lease_id,
        "rotated_lock_sha256": rotated.sha256,
        "revalidated_head": str(review["current_head"]),
        "revalidated_tree": str(review["current_tree"]),
        "revalidated_history_id": str(review["history_id"]),
        "clearance_executor": dict(review["clearance_executor"]),
        "decision": DECISION,
        "authority": dict(NON_AUTHORITY),
    }


def _finalize_receipt(
    root: Path,
    authorization: Mapping[str, Any],
    release_intent: Mapping[str, Any],
    *,
    recovered_after_crash: bool,
) -> tuple[dict[str, Any], Path]:
    basis = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "authorization_id": str(authorization["authorization_id"]),
        "review_id": str(authorization["review_id"]),
        "release_intent_id": str(release_intent["release_intent_id"]),
        "original_lease_id": str(authorization["expected_lease_id"]),
        "rotated_lease_id": str(release_intent["rotated_lease_id"]),
        "released": True,
        "decision": DECISION,
        "authority": dict(NON_AUTHORITY),
        "security_model": SECURITY_MODEL,
        "clearance_executor": dict(authorization["clearance_executor"]),
        "recovered_after_crash": recovered_after_crash,
    }
    final_id = _identity(basis)
    path = _receipt_path(root, "final", final_id)
    if path.exists():
        existing = _validated_receipt(
            path,
            root=root,
            kind="final",
            schema=FINAL_RECEIPT_SCHEMA,
            identity_field="final_receipt_id",
            timestamp_field="released_at",
        )
        unsigned = dict(existing)
        observed = str(unsigned.pop("final_receipt_id", "") or "")
        unsigned.pop("released_at", None)
        if observed != final_id or unsigned != basis:
            raise ProtectedRecoveryClearanceError("final_receipt_invalid")
        return existing, path
    receipt = {
        **basis,
        "released_at": _utc_now(),
        "final_receipt_id": final_id,
    }
    _durable_receipt(path, receipt)
    return receipt, path


def apply_protected_recovery_clearance(
    repo_root: Path,
    *,
    receipt_dir: Path,
    expected_lease_id: str,
    expected_review_id: str,
    expected_lock_sha256: str,
    approved_commits: Sequence[str],
    operator_identity: str,
    operator_note: str,
) -> dict[str, Any]:
    """Authorize and release one exact retained journal, or fail closed."""

    identity = str(operator_identity).strip()
    note = str(operator_note).strip()
    if not identity:
        raise ProtectedRecoveryClearanceError("operator_identity_required")
    if not note:
        raise ProtectedRecoveryClearanceError("operator_note_required")
    if not expected_lease_id:
        raise ProtectedRecoveryClearanceError("expected_lease_id_required")
    if not expected_review_id:
        raise ProtectedRecoveryClearanceError("expected_review_id_required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_lock_sha256):
        raise ProtectedRecoveryClearanceError(
            "expected_lock_sha256_invalid"
        )
    operator_uid = getattr(os, "geteuid", lambda: -1)()
    if operator_uid < 0:
        raise ProtectedRecoveryClearanceError("os_owner_check_unavailable")

    binding = _repo_binding(repo_root)
    repo = Path(binding["repo_root"])
    common = Path(binding["git_common_dir"])
    root = _prepare_receipt_dir(
        receipt_dir,
        repo=repo,
        common_dir=common,
    )
    lock_path = Path(binding["lock_path"])

    try:
        current = _read_lock(lock_path)
    except ProtectedRecoveryClearanceError as exc:
        if exc.reason != "lock_absent":
            raise
        # Crash recovery is permitted only after a durable authorization and
        # exact release intent were both published.
        current_executor = _clearance_executor_identity()
        _, oid_length = _object_format(repo)
        approvals = _validate_full_approvals(
            repo,
            approved_commits,
            expected=approved_commits,
            oid_length=oid_length,
        )
        required_authorization = {
            "security_model": SECURITY_MODEL,
            "cryptographic_operator_signature": False,
            "operator_identity": identity,
            "operator_uid": operator_uid,
            "operator_note": note,
            "decision": DECISION,
            "authority": dict(NON_AUTHORITY),
            "review_id": expected_review_id,
            "lock_sha256": expected_lock_sha256,
            "expected_lease_id": expected_lease_id,
            "repo_root": binding["repo_root"],
            "worktree_root": binding["worktree_root"],
            "repository_id": binding["repository_id"],
            "git_common_dir": binding["git_common_dir"],
            "lock_path": binding["lock_path"],
            "approved_commits": list(approvals),
            "clearance_executor": current_executor,
        }
        authorizations = _matching_receipts(
            root,
            kind="authorization",
            schema=AUTHORIZATION_SCHEMA,
            identity_field="authorization_id",
            timestamp_field="authorized_at",
            required=required_authorization,
        )
        if len(authorizations) != 1:
            raise ProtectedRecoveryClearanceError(
                "lock_absent_without_authorization"
            ) from None
        authorization, _authorization_path = authorizations[0]
        release_intents = _matching_receipts(
            root,
            kind="release-intent",
            schema=RELEASE_INTENT_SCHEMA,
            identity_field="release_intent_id",
            timestamp_field=None,
            required={
                "authorization_id": authorization["authorization_id"],
                "review_id": expected_review_id,
                "revalidated_head": authorization["reviewed_head"],
                "revalidated_tree": authorization["reviewed_tree"],
                "revalidated_history_id": authorization["history_id"],
                "clearance_executor": current_executor,
                "decision": DECISION,
                "authority": dict(NON_AUTHORITY),
            },
        )
        if len(release_intents) != 1:
            raise ProtectedRecoveryClearanceError(
                "lock_absent_without_release_intent"
            ) from None
        release_intent, _release_intent_path = release_intents[0]
        if not str(release_intent.get("rotated_lease_id") or "") or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(release_intent.get("rotated_lock_sha256") or ""),
        ):
            raise ProtectedRecoveryClearanceError(
                "release-intent_receipt_invalid"
            ) from None
        final, final_path = _finalize_receipt(
            root,
            authorization,
            release_intent,
            recovered_after_crash=True,
        )
        return {
            "released": True,
            "idempotent_recovery": True,
            "authorization_id": authorization["authorization_id"],
            "final_receipt_id": final["final_receipt_id"],
            "final_receipt_path": str(final_path),
            "decision": DECISION,
            "authority": dict(NON_AUTHORITY),
        }

    repo_stat = repo.stat()
    common_stat = common.stat()
    if current.uid != operator_uid or int(repo_stat.st_uid) != operator_uid or int(
        common_stat.st_uid
    ) != operator_uid:
        raise ProtectedRecoveryClearanceError("os_owner_mismatch")

    # Normal path: exact original journal inspection and authorization.
    if current.lease_id == expected_lease_id:
        if current.sha256 != expected_lock_sha256:
            raise ProtectedRecoveryClearanceError("lock_digest_mismatch")
        review = _inspect_valid_journal(
            repo, binding, current, confirm_lock=True
        )
        if str(review["review_id"]) != expected_review_id:
            raise ProtectedRecoveryClearanceError("review_identity_mismatch")
        if review.get("eligible") is not True or review.get("reason") != (
            "protected_generated_history_untrusted"
        ):
            raise ProtectedRecoveryClearanceError(
                "clearance_reason_not_authorizable",
                reason=review.get("reason"),
            )
        approvals = _validate_full_approvals(
            repo,
            approved_commits,
            expected=review["untrusted_commits"],
            oid_length=int(review["full_oid_length"]),
        )
        basis = _authorization_basis(
            review=review,
            approvals=approvals,
            operator_identity=identity,
            operator_note=note,
            operator_uid=operator_uid,
        )
        authorization, authorization_path = _load_or_write_authorization(
            root, basis
        )
        rotated_metadata = _rotated_metadata(current, authorization)
        rotated_metadata[
            "protected_recovery_clearance_original_device"
        ] = current.device
        rotated_metadata[
            "protected_recovery_clearance_original_inode"
        ] = current.inode
        rotated = _write_replacement(current, rotated_metadata)
        original = current
    else:
        # Resume only the exact deterministic authorization lease produced by
        # this protocol.  An unrelated replacement always fails closed.
        authorization_matches = _matching_receipts(
            root,
            kind="authorization",
            schema=AUTHORIZATION_SCHEMA,
            identity_field="authorization_id",
            timestamp_field="authorized_at",
            required={
                "review_id": expected_review_id,
                "expected_lease_id": expected_lease_id,
                "lock_sha256": expected_lock_sha256,
                "operator_identity": identity,
                "operator_uid": operator_uid,
                "operator_note": note,
                "approved_commits": list(approved_commits),
                "repo_root": binding["repo_root"],
                "worktree_root": binding["worktree_root"],
                "repository_id": binding["repository_id"],
                "git_common_dir": binding["git_common_dir"],
                "lock_path": binding["lock_path"],
                "clearance_executor": _clearance_executor_identity(),
                "decision": DECISION,
                "authority": dict(NON_AUTHORITY),
            },
        )
        if len(authorization_matches) != 1:
            raise ProtectedRecoveryClearanceError("expected_lease_mismatch")
        authorization, authorization_path = authorization_matches[0]
        original = _original_snapshot_from_rotated(
            current,
            expected_lease_id=expected_lease_id,
            expected_lock_sha256=expected_lock_sha256,
            authorization_id=str(authorization["authorization_id"]),
            expected_review_id=expected_review_id,
        )
        if original is None:
            raise ProtectedRecoveryClearanceError("expected_lease_mismatch")
        expected_rotated = _rotated_metadata(original, authorization)
        expected_rotated[
            "protected_recovery_clearance_original_device"
        ] = int(
            current.metadata.get("protected_recovery_clearance_original_device")
            or 0
        )
        expected_rotated[
            "protected_recovery_clearance_original_inode"
        ] = int(
            current.metadata.get("protected_recovery_clearance_original_inode")
            or 0
        )
        if current.metadata != expected_rotated:
            raise ProtectedRecoveryClearanceError(
                "authorized_lease_metadata_mismatch"
            )
        rotated = current
        review = _inspect_valid_journal(
            repo, binding, original, confirm_lock=False
        )
        if str(review["review_id"]) != expected_review_id:
            raise ProtectedRecoveryClearanceError("review_identity_mismatch")

    # Hold the common-directory update lock across final repository
    # revalidation, durable release intent, exact lease check, and unlink.
    with serialized_lock_update(lock_path, timeout_seconds=5.0):
        confirmed_rotated = _read_lock(lock_path)
        if not _same_lock(rotated, confirmed_rotated):
            raise ProtectedRecoveryClearanceError("authorized_lease_replaced")
        revalidated = _inspect_valid_journal(
            repo,
            binding,
            original,
            confirm_lock=False,
        )
        if str(revalidated["review_id"]) != expected_review_id:
            raise ProtectedRecoveryClearanceError(
                "repository_changed_after_authorization"
            )
        intent_basis = _release_intent_basis(
            authorization,
            confirmed_rotated,
            revalidated,
        )
        release_intent = _with_identity(intent_basis, "release_intent_id")
        release_intent_path = _receipt_path(
            root,
            "release-intent",
            str(release_intent["release_intent_id"]),
        )
        _durable_receipt(release_intent_path, release_intent)
        final_lock = _read_lock(lock_path)
        if not _same_lock(confirmed_rotated, final_lock):
            raise ProtectedRecoveryClearanceError("authorized_lease_replaced")
        lock_path.unlink()
        _fsync_directory(lock_path.parent)

    final, final_path = _finalize_receipt(
        root,
        authorization,
        release_intent,
        recovered_after_crash=False,
    )
    return {
        "released": True,
        "idempotent_recovery": False,
        "decision": DECISION,
        "authority": dict(NON_AUTHORITY),
        "authorization_id": authorization["authorization_id"],
        "authorization_receipt_path": str(authorization_path),
        "release_intent_path": str(release_intent_path),
        "final_receipt_id": final["final_receipt_id"],
        "final_receipt_path": str(final_path),
    }


def _read_operator_input_file(
    path: Path,
    *,
    repo: Path,
    common_dir: Path,
) -> bytes:
    """Read one owner-only regular operator input with stable fd identity."""

    if not path.is_absolute():
        raise ProtectedRecoveryClearanceError(
            "operator_input_path_must_be_absolute"
        )
    try:
        parent = path.parent.resolve(strict=True)
        parent_info = parent.stat(follow_symlinks=False)
    except (OSError, RuntimeError) as exc:
        raise ProtectedRecoveryClearanceError(
            "operator_input_path_invalid"
        ) from exc
    operator_uid = getattr(os, "geteuid", lambda: -1)()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or int(parent_info.st_uid) != operator_uid
        or stat.S_IMODE(parent_info.st_mode) & 0o022
    ):
        raise ProtectedRecoveryClearanceError(
            "operator_input_directory_unsafe"
        )
    resolved = parent / path.name
    if _is_relative_to(resolved, repo) or _is_relative_to(
        resolved, common_dir
    ):
        raise ProtectedRecoveryClearanceError(
            "operator_input_inside_repository"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ProtectedRecoveryClearanceError(
            "operator_input_unreadable",
            error_type=type(exc).__name__,
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or int(before.st_uid) != operator_uid
            or stat.S_IMODE(before.st_mode) != 0o600
            or int(before.st_size) <= 0
            or int(before.st_size) > 1024 * 1024
        ):
            raise ProtectedRecoveryClearanceError(
                "operator_input_file_unsafe"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    identity_after = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    raw = b"".join(chunks)
    if identity_before != identity_after or len(raw) != int(before.st_size):
        raise ProtectedRecoveryClearanceError(
            "operator_input_changed_during_read"
        )
    return raw


def _read_operator_json_mapping(
    path: Path,
    *,
    repo: Path,
    common_dir: Path,
) -> dict[str, Any]:
    raw = _read_operator_input_file(
        path,
        repo=repo,
        common_dir=common_dir,
    )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number: {value}")
            ),
        )
    except (UnicodeError, ValueError) as exc:
        raise ProtectedRecoveryClearanceError(
            "operator_input_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProtectedRecoveryClearanceError(
            "operator_input_json_invalid"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or proof-check an operator clearance of one retained "
            "implementation-supervisor protected-recovery journal."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--repo-root", type=Path, required=True)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--repo-root", type=Path, required=True)
    restore_parser.add_argument("--receipt-dir", type=Path, required=True)
    restore_parser.add_argument("--snapshot-path", type=Path, required=True)
    restore_parser.add_argument(
        "--source-review-path", type=Path, required=True
    )
    restore_parser.add_argument("--expected-lease-id", required=True)
    restore_parser.add_argument("--expected-lock-sha256", required=True)
    restore_parser.add_argument("--operator-event-id", required=True)
    restore_parser.add_argument(
        "--approve-commit",
        action="append",
        default=[],
        help="Exact full commit OID, repeated in review order.",
    )
    restore_parser.add_argument("--operator-identity", required=True)
    restore_parser.add_argument("--operator-note", required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--repo-root", type=Path, required=True)
    apply_parser.add_argument("--receipt-dir", type=Path, required=True)
    apply_parser.add_argument("--expected-lease-id", required=True)
    apply_parser.add_argument("--expected-review-id", required=True)
    apply_parser.add_argument("--expected-lock-sha256", required=True)
    apply_parser.add_argument(
        "--approve-commit",
        action="append",
        default=[],
        help="Exact full commit OID, repeated in review order.",
    )
    apply_parser.add_argument("--operator-identity", required=True)
    apply_parser.add_argument("--operator-note", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_protected_recovery(args.repo_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "restore":
            binding = _repo_binding(args.repo_root)
            repo = Path(binding["repo_root"])
            common = Path(binding["git_common_dir"])
            result = restore_missing_protected_recovery_fence(
                args.repo_root,
                receipt_dir=args.receipt_dir,
                snapshot_bytes=_read_operator_input_file(
                    args.snapshot_path,
                    repo=repo,
                    common_dir=common,
                ),
                expected_lease_id=args.expected_lease_id,
                expected_lock_sha256=args.expected_lock_sha256,
                source_review=_read_operator_json_mapping(
                    args.source_review_path,
                    repo=repo,
                    common_dir=common,
                ),
                operator_event_id=args.operator_event_id,
                approved_commits=args.approve_commit,
                operator_identity=args.operator_identity,
                operator_note=args.operator_note,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = apply_protected_recovery_clearance(
            args.repo_root,
            receipt_dir=args.receipt_dir,
            expected_lease_id=args.expected_lease_id,
            expected_review_id=args.expected_review_id,
            expected_lock_sha256=args.expected_lock_sha256,
            approved_commits=args.approve_commit,
            operator_identity=args.operator_identity,
            operator_note=args.operator_note,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ProtectedRecoveryClearanceError as exc:
        print(
            json.dumps(
                {
                    (
                        "restored"
                        if getattr(args, "command", "") == "restore"
                        else "released"
                    ): False,
                    "reason": exc.reason,
                    "details": exc.details,
                    "decision": DECISION,
                    "authority": dict(NON_AUTHORITY),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
