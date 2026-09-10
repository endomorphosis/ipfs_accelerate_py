"""Reusable Git worktree ownership and cleanup helpers for todo daemons."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
)
from .engine import CommandResult, run_command
from .git_utils import (
    git_worktree_paths_from_porcelain as _shared_git_worktree_paths_from_porcelain,
)
from .git_utils import (
    paths_from_git_status_porcelain as _shared_paths_from_git_status_porcelain,
)
from .git_utils import (
    untracked_paths_from_git_status_porcelain as _shared_untracked_paths_from_git_status_porcelain,
)

CommandRunner = Callable[..., CommandResult]
OwnerAlivePredicate = Callable[[int, Path, Path], bool]
TraceResultFormatter = Callable[[CommandResult, int], Any]
WorktreeOwnerWriter = Callable[[Path], None]
WorktreePrepare = Callable[[Path], Any]
WorktreeReuseAuthorizer = Callable[[Path, str, str], tuple[bool, str]]


WORKTREE_POOL_SCHEMA = "agent-supervisor-worktree-pool-v1"
WORKTREE_POOL_QUARANTINE_SCHEMA = "agent-supervisor-worktree-pool-quarantine-v1"
WORKTREE_POOL_MISSING_RELEASE_SCHEMA = (
    "agent-supervisor-worktree-pool-missing-release-v1"
)
WORKTREE_POOL_MUTATION_GUARD_TIMEOUT_SECONDS = 5.0
_WORKTREE_POOL_QUARANTINE_FIELDS = frozenset(
    {
        "schema",
        "quarantine_id",
        "entry_id",
        "workspace_path",
        "branch",
        "repo_root",
        "repo_common_dir",
        "pool_state_cid",
        "pool_lock_cid",
        "pool_lease_pid",
        "board_namespace",
        "task_id",
        "canonical_task_cid",
        "attempt",
        "merge_target",
        "lifecycle_record_id",
        "lifecycle_fence",
        "lifecycle_lease_id",
        "owner_process_birth",
        "predecessor_state_dir",
        "current_state_dir",
        "git_preimage_cid",
        "git_registered_head",
        "git_worktree_lock_reason",
        "reason",
    }
)
_WORKTREE_POOL_MISSING_RELEASE_FIELDS = frozenset(
    {
        "schema",
        "evidence_id",
        "entry_id",
        "workspace_path",
        "branch",
        "base_commit",
        "repo_root",
        "repo_common_dir",
        "pool_state_cid",
        "pool_state_sha256",
        "pool_lock_cid",
        "pool_lock_sha256",
        "pool_lease_pid",
        "retained_branch_head",
        "branch_disposition",
        "release_phase",
        "implementation_started",
        "provider_dispatched",
        "lifecycle",
    }
)


def _canonical_worktree_pool_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _worktree_pool_payload_cid(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_worktree_pool_json_bytes(payload)
    ).hexdigest()


def worktree_pool_payload_cid(payload: Mapping[str, Any]) -> str:
    """Return the exact canonical SHA-256 identity used by pool custody."""

    return _worktree_pool_payload_cid(payload)


def _git_worktree_registration(
    repo_root: Path,
    workspace_path: Path,
) -> tuple[dict[str, str] | None, str]:
    """Resolve one exact lexical worktree registration and its Git lock."""

    try:
        result = run_command(
            ("git", "worktree", "list", "--porcelain", "-z"),
            cwd=repo_root,
            timeout_seconds=30,
        )
    except OSError:
        return None, "git_worktree_registration_unavailable"
    if not result.ok:
        return None, "git_worktree_registration_unavailable"
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for field in result.stdout.split("\0"):
        if not field:
            if current:
                records.append(current)
                current = {}
            continue
        key, separator, value = field.partition(" ")
        current[key] = value if separator else ""
    if current:
        records.append(current)
    expected = Path(os.path.abspath(workspace_path))
    matches = [
        record
        for record in records
        if record.get("worktree")
        and Path(os.path.abspath(str(record["worktree"]))) == expected
    ]
    if not matches:
        return None, "git_worktree_registration_absent"
    if len(matches) != 1:
        return None, "git_worktree_registration_ambiguous"
    return matches[0], "git_worktree_registration_exact"


def _strict_worktree_pool_json_object(path: Path) -> dict[str, Any] | None:
    """Load one regular JSON object with duplicate-key rejection."""

    if path.is_symlink() or not path.is_file():
        return None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def _worktree_pool_entry_id_for_workspace(path: Path) -> str:
    match = re.fullmatch(
        r"workspace(?P<separator>[_-])([0-9a-f]{12})(?P=separator)([0-9a-f]{12})",
        path.name,
    )
    return f"{match.group(2)}-{match.group(3)}" if match else ""


def worktree_pool_entry_id_for_workspace(path: Path | str) -> str:
    """Return the current or retained-legacy pool entry identity."""

    return _worktree_pool_entry_id_for_workspace(Path(path))


def worktree_pool_entry_guard_binding(
    *,
    worktree_root: Path | str,
    workspace_path: Path | str,
) -> dict[str, Any]:
    """Bind one lexical pooled workspace to its existing update guard.

    A pooled-looking direct child fails closed when its pool-state directory
    is missing or unsafe. Non-pooled worktrees remain outside this authority.
    """

    base: dict[str, Any] = {
        "pooled": False,
        "guard_available": False,
    }
    raw_root = Path(worktree_root)
    raw_workspace = Path(workspace_path)
    try:
        lexical_root = Path(os.path.abspath(raw_root))
        lexical_workspace = Path(os.path.abspath(raw_workspace))
    except (OSError, RuntimeError, ValueError):
        return {**base, "reason": "worktree_pool_root_unavailable"}
    entry_id = _worktree_pool_entry_id_for_workspace(lexical_workspace)
    if not entry_id or lexical_workspace.parent != lexical_root:
        return {**base, "reason": "not_a_pooled_workspace"}

    # Classify the lexical direct child before resolving the root.  A missing,
    # replaced, or otherwise uninspectable pool root must not turn a path that
    # looks exactly like a managed pool entry into an unrestricted worktree.
    lexical_pool_root = lexical_root / ".pool-state"
    lexical_binding = {
        **base,
        "pooled": True,
        "entry_id": entry_id,
        "pool_root": str(lexical_pool_root),
        "lock_path": str(lexical_pool_root / f"{entry_id}.lock"),
    }
    try:
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return {**lexical_binding, "reason": "worktree_pool_root_unavailable"}

    pool_root = root / ".pool-state"
    binding = {
        **base,
        "pooled": True,
        "entry_id": entry_id,
        "pool_root": str(pool_root),
        "lock_path": str(pool_root / f"{entry_id}.lock"),
    }
    try:
        pool_stat = pool_root.lstat()
    except FileNotFoundError:
        return {**binding, "reason": "worktree_pool_state_root_missing"}
    except OSError as exc:
        return {
            **binding,
            "reason": "worktree_pool_state_root_uninspectable",
            "error_type": type(exc).__name__,
        }
    if stat_module.S_ISLNK(pool_stat.st_mode) or not stat_module.S_ISDIR(
        pool_stat.st_mode
    ):
        return {**binding, "reason": "worktree_pool_state_root_unsafe"}
    custody_identities: dict[str, dict[str, int] | None] = {}
    pool_repo_root = ""
    for custody_path in (
        pool_root / f"{entry_id}.json",
        pool_root / f"{entry_id}.lock",
    ):
        try:
            custody_stat = custody_path.lstat()
        except FileNotFoundError:
            custody_identities[custody_path.name] = None
            continue
        except OSError as exc:
            return {
                **binding,
                "reason": "worktree_pool_entry_custody_uninspectable",
                "error_type": type(exc).__name__,
            }
        if not stat_module.S_ISREG(custody_stat.st_mode):
            return {**binding, "reason": "worktree_pool_entry_custody_unsafe"}
        custody_identities[custody_path.name] = {
            "device": int(custody_stat.st_dev),
            "inode": int(custody_stat.st_ino),
            "mode": int(stat_module.S_IFMT(custody_stat.st_mode)),
        }
        if custody_path.name == f"{entry_id}.json":
            state_payload = _strict_worktree_pool_json_object(custody_path)
            if state_payload is None:
                return {
                    **binding,
                    "reason": "worktree_pool_entry_custody_invalid",
                }
            pool_repo_root = str(state_payload.get("repo_root") or "")
    return {
        **binding,
        "guard_available": True,
        "reason": "worktree_pool_entry_guard_bound",
        "pool_root_identity": {
            "device": int(pool_stat.st_dev),
            "inode": int(pool_stat.st_ino),
            "mode": int(stat_module.S_IFMT(pool_stat.st_mode)),
        },
        "custody_identities": custody_identities,
        "pool_repo_root": pool_repo_root,
    }


def _valid_worktree_pool_quarantine_payload(
    payload: Mapping[str, Any],
    *,
    worktree_root: Path,
    workspace_path: Path,
    entry_id: str,
    expected_branch: str = "",
) -> tuple[bool, str]:
    if set(payload) != _WORKTREE_POOL_QUARANTINE_FIELDS:
        return False, "quarantine_fields_invalid"
    unsigned = dict(payload)
    quarantine_id = unsigned.pop("quarantine_id", None)
    git_worktree_lock_reason = unsigned.pop(
        "git_worktree_lock_reason",
        None,
    )
    owner = payload.get("owner_process_birth")
    if (
        payload.get("schema") != WORKTREE_POOL_QUARANTINE_SCHEMA
        or payload.get("entry_id") != entry_id
        or type(quarantine_id) is not str
        or quarantine_id != _worktree_pool_payload_cid(unsigned)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", quarantine_id)
        or git_worktree_lock_reason
        != f"agent-supervisor-quarantine-v1:{quarantine_id}"
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(payload.get("pool_state_cid") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(payload.get("pool_lock_cid") or ""),
        )
        or type(payload.get("pool_lease_pid")) is not int
        or int(payload.get("pool_lease_pid") or 0) <= 1
        or type(payload.get("attempt")) is not int
        or int(payload.get("attempt") or 0) < 1
        or type(payload.get("lifecycle_fence")) is not int
        or int(payload.get("lifecycle_fence") or 0) < 1
        or not isinstance(owner, dict)
        or set(owner)
        != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        or type(owner.get("pid")) is not int
        or owner.get("pid") != payload.get("pool_lease_pid")
        or type(owner.get("start_time_ticks")) is not int
        or int(owner.get("start_time_ticks") or 0) <= 0
        or type(owner.get("parent_pid")) is not int
        or type(owner.get("boot_id")) is not str
        or any(
            type(payload.get(field)) is not str or not payload.get(field)
            for field in (
                "workspace_path",
                "branch",
                "repo_root",
                "repo_common_dir",
                "board_namespace",
                "task_id",
                "canonical_task_cid",
                "merge_target",
                "lifecycle_record_id",
                "lifecycle_lease_id",
                "predecessor_state_dir",
                "current_state_dir",
                "git_preimage_cid",
                "git_registered_head",
                "git_worktree_lock_reason",
                "reason",
            )
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(payload.get("git_preimage_cid") or ""),
        )
        or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(payload.get("git_registered_head") or ""),
        )
    ):
        return False, "quarantine_identity_invalid"
    try:
        marker_workspace = Path(str(payload["workspace_path"]))
        marker_workspace_resolved = marker_workspace.resolve(strict=True)
        marker_workspace_resolved.relative_to(worktree_root)
        marker_repo = Path(str(payload["repo_root"]))
        marker_common = Path(str(payload["repo_common_dir"]))
        predecessor_state = Path(str(payload["predecessor_state_dir"]))
        current_state = Path(str(payload["current_state_dir"]))
        pool_root = worktree_root / ".pool-state"
        pool_state_path = pool_root / f"{entry_id}.json"
        pool_lock_path = pool_root / f"{entry_id}.lock"
        if any(
            candidate.is_symlink()
            for candidate in (
                marker_workspace,
                marker_repo,
                marker_common,
                predecessor_state,
                current_state,
                pool_root,
                pool_state_path,
                pool_lock_path,
            )
        ):
            return False, "quarantine_path_symlink"
        marker_repo_resolved = marker_repo.resolve(strict=True)
        marker_common_resolved = marker_common.resolve(strict=True)
        predecessor_state_resolved = predecessor_state.resolve(strict=True)
        current_state_resolved = current_state.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False, "quarantine_path_invalid"
    if (
        marker_workspace_resolved != workspace_path
        or marker_workspace_resolved.parent != worktree_root
        or not marker_workspace.is_dir()
        or not marker_repo.is_dir()
        or not marker_common.is_dir()
        or not predecessor_state.is_dir()
        or not current_state.is_dir()
        or marker_repo_resolved != Path(str(payload["repo_root"]))
        or marker_common_resolved != Path(str(payload["repo_common_dir"]))
        or predecessor_state_resolved
        != Path(str(payload["predecessor_state_dir"]))
        or current_state_resolved != Path(str(payload["current_state_dir"]))
        or (
            bool(expected_branch)
            and str(payload["branch"]).removeprefix("refs/heads/")
            != str(expected_branch).removeprefix("refs/heads/")
        )
    ):
        return False, "quarantine_workspace_binding_mismatch"

    pool_state = _strict_worktree_pool_json_object(pool_state_path)
    pool_lock = _strict_worktree_pool_json_object(pool_lock_path)
    try:
        pool_state_cid = (
            _worktree_pool_payload_cid(pool_state)
            if pool_state is not None
            else ""
        )
        pool_lock_cid = (
            _worktree_pool_payload_cid(pool_lock)
            if pool_lock is not None
            else ""
        )
    except (TypeError, ValueError):
        return False, "quarantine_pool_binding_mismatch"
    if (
        pool_state is None
        or pool_lock is None
        or pool_state_cid != payload.get("pool_state_cid")
        or pool_lock_cid != payload.get("pool_lock_cid")
        or pool_state.get("schema") != WORKTREE_POOL_SCHEMA
        or pool_state.get("lease_token") != entry_id
        or pool_state.get("state") != "leased"
        or pool_state.get("lease_pid") != payload.get("pool_lease_pid")
        or str(pool_state.get("path") or "") != str(marker_workspace_resolved)
        or str(pool_state.get("repo_root") or "") != str(marker_repo_resolved)
        or str(pool_state.get("repo_common_dir") or "")
        != str(marker_common_resolved)
        or str(pool_state.get("branch") or "").removeprefix("refs/heads/")
        != str(payload["branch"]).removeprefix("refs/heads/")
        or set(pool_lock) != {"pid", "created_at_epoch"}
        or pool_lock.get("pid") != payload.get("pool_lease_pid")
    ):
        return False, "quarantine_pool_binding_mismatch"
    registration, registration_reason = _git_worktree_registration(
        marker_repo_resolved,
        marker_workspace,
    )
    if (
        registration is None
        or registration_reason != "git_worktree_registration_exact"
        or str(registration.get("branch") or "").removeprefix(
            "refs/heads/"
        )
        != str(payload["branch"]).removeprefix("refs/heads/")
        or str(registration.get("HEAD") or "")
        != str(payload["git_registered_head"])
        or str(registration.get("locked") or "")
        != str(payload["git_worktree_lock_reason"])
    ):
        return False, "quarantine_git_lock_binding_mismatch"
    return True, "durable_worktree_pool_quarantine"


def inspect_worktree_pool_quarantine(
    *,
    worktree_root: Path | str,
    workspace_path: Path | str,
    expected_branch: str = "",
) -> dict[str, Any]:
    """Return a typed exact quarantine result for one pooled workspace.

    ``invalid`` is deliberately distinct from ``valid`` but remains a cleanup
    fence.  A malformed or foreign marker can never authorize recovery, while
    cleanup also cannot destroy the evidence an operator needs to repair it.
    """

    raw_root = Path(worktree_root)
    raw_workspace = Path(workspace_path)
    base: dict[str, Any] = {
        "status": "absent",
        "valid": False,
        "cleanup_fenced": False,
    }
    try:
        root = raw_root.resolve(strict=True)
        lexical_root = Path(os.path.abspath(raw_root))
        lexical_workspace = Path(os.path.abspath(raw_workspace))
    except (OSError, RuntimeError, ValueError):
        return {**base, "reason": "not_a_pooled_workspace"}
    entry_id = _worktree_pool_entry_id_for_workspace(lexical_workspace)
    if (
        not entry_id
        or lexical_workspace.parent not in {lexical_root, root}
    ):
        return {**base, "reason": "not_a_pooled_workspace"}
    quarantine_root = root / ".pool-state" / "quarantine"
    marker_path = quarantine_root / f"{entry_id}.json"
    try:
        quarantine_root_stat = quarantine_root.lstat()
    except FileNotFoundError:
        return {
            **base,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_absent",
        }
    except OSError as exc:
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_directory_uninspectable",
            "error_type": type(exc).__name__,
        }
    if stat_module.S_ISLNK(
        quarantine_root_stat.st_mode
    ) or not stat_module.S_ISDIR(quarantine_root_stat.st_mode):
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_directory_unsafe",
        }
    try:
        marker_stat = marker_path.lstat()
    except FileNotFoundError:
        return {
            **base,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_absent",
        }
    except OSError as exc:
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_marker_uninspectable",
            "error_type": type(exc).__name__,
        }
    try:
        workspace = raw_workspace.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_workspace_uninspectable",
        }
    try:
        workspace.relative_to(root)
    except ValueError:
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_workspace_binding_mismatch",
        }
    if (
        stat_module.S_ISLNK(marker_stat.st_mode)
        or not stat_module.S_ISREG(marker_stat.st_mode)
    ):
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_marker_unsafe",
        }
    payload = _strict_worktree_pool_json_object(marker_path)
    if payload is None:
        return {
            **base,
            "status": "invalid",
            "cleanup_fenced": True,
            "entry_id": entry_id,
            "marker_path": str(marker_path),
            "reason": "quarantine_marker_malformed",
        }
    valid, reason = _valid_worktree_pool_quarantine_payload(
        payload,
        worktree_root=root,
        workspace_path=workspace,
        entry_id=entry_id,
        expected_branch=expected_branch,
    )
    return {
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "cleanup_fenced": True,
        "entry_id": entry_id,
        "marker_path": str(marker_path),
        "reason": reason,
        **({"marker": payload} if valid else {}),
    }


def inspect_worktree_pool_missing_release_terminal(
    *,
    repo_root: Path | str,
    worktree_root: Path | str,
    workspace_path: Path | str,
    expected_branch: str = "",
    expected_lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one deterministic pre-dispatch missing-workspace terminal.

    The active ``<entry>.json`` state always dominates: a receipt published
    while that file remains is proposal-only.  Terminal status requires the
    exact original state bytes at the deterministic released-state path,
    absence from both the filesystem and Git worktree registry, and a retained
    branch still pointing at the acquired base commit.
    """

    base: dict[str, Any] = {
        "status": "absent",
        "valid": False,
        "cleanup_fenced": False,
    }
    raw_root = Path(worktree_root)
    raw_workspace = Path(workspace_path)
    try:
        root = raw_root.resolve(strict=True)
        authoritative_repo = Path(repo_root).resolve(strict=True)
        lexical_workspace = Path(os.path.abspath(raw_workspace))
    except (OSError, RuntimeError, ValueError):
        return {**base, "reason": "missing_release_identity_unresolvable"}
    entry_id = _worktree_pool_entry_id_for_workspace(lexical_workspace)
    if not entry_id or lexical_workspace.parent != root:
        return {**base, "reason": "not_a_pooled_workspace"}
    state_root = root / ".pool-state"
    active_state_path = state_root / f"{entry_id}.json"
    lock_path = state_root / f"{entry_id}.lock"
    terminal_state_path = state_root / f".{entry_id}.released-state"
    receipt_path = state_root / f".{entry_id}.released-receipt"
    detail = {
        "entry_id": entry_id,
        "active_state_path": str(active_state_path),
        "lock_path": str(lock_path),
        "terminal_state_path": str(terminal_state_path),
        "receipt_path": str(receipt_path),
    }
    try:
        terminal_stat = terminal_state_path.lstat()
    except FileNotFoundError:
        terminal_stat = None
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_terminal_uninspectable",
            "error_type": type(exc).__name__,
        }
    try:
        receipt_stat = receipt_path.lstat()
    except FileNotFoundError:
        receipt_stat = None
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_receipt_uninspectable",
            "error_type": type(exc).__name__,
        }
    terminal_exists = terminal_stat is not None
    receipt_exists = receipt_stat is not None
    if not terminal_exists and not receipt_exists:
        return {**base, **detail, "reason": "missing_release_absent"}
    try:
        active_stat = active_state_path.lstat()
    except FileNotFoundError:
        active_stat = None
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_active_state_uninspectable",
            "error_type": type(exc).__name__,
        }
    active_exists = active_stat is not None
    if receipt_exists and active_exists and not terminal_exists:
        evidence_state_path = active_state_path
        evidence_state_stat = active_stat
        proposal = True
    elif receipt_exists and terminal_exists and not active_exists:
        evidence_state_path = terminal_state_path
        evidence_state_stat = terminal_stat
        proposal = False
    else:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": (
                "missing_release_active_state_conflicts_with_terminal"
                if active_exists and terminal_exists
                else "missing_release_terminal_pair_incomplete"
            ),
        }
    try:
        evidence_state_bytes = evidence_state_path.read_bytes()
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_terminal_unreadable",
            "error_type": type(exc).__name__,
        }
    if any(
        stat_module.S_ISLNK(item.st_mode)
        or not stat_module.S_ISREG(item.st_mode)
        for item in (evidence_state_stat, receipt_stat)
    ):
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_terminal_unsafe",
        }
    receipt = _strict_worktree_pool_json_object(receipt_path)
    evidence_state = _strict_worktree_pool_json_object(evidence_state_path)
    if receipt is None or evidence_state is None:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_terminal_malformed",
        }
    unsigned = dict(receipt)
    evidence_id = unsigned.pop("evidence_id", None)
    lifecycle = receipt.get("lifecycle")
    owner = lifecycle.get("owner") if isinstance(lifecycle, Mapping) else None
    expected_branch_name = str(expected_branch or "").removeprefix(
        "refs/heads/"
    )
    evidence_state_cid = _worktree_pool_payload_cid(evidence_state)
    evidence_state_sha256 = "sha256:" + hashlib.sha256(
        evidence_state_bytes
    ).hexdigest()
    identity_valid = bool(
        set(receipt) == _WORKTREE_POOL_MISSING_RELEASE_FIELDS
        and receipt.get("schema") == WORKTREE_POOL_MISSING_RELEASE_SCHEMA
        and isinstance(evidence_id, str)
        and evidence_id == _worktree_pool_payload_cid(unsigned)
        and receipt.get("entry_id") == entry_id
        and receipt.get("workspace_path") == str(lexical_workspace)
        and str(receipt.get("repo_root") or "")
        == str(authoritative_repo)
        and str(receipt.get("branch") or "").removeprefix("refs/heads/")
        == expected_branch_name
        and re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            str(receipt.get("base_commit") or ""),
        )
        and receipt.get("retained_branch_head")
        == receipt.get("base_commit")
        and receipt.get("branch_disposition")
        == "retained_exact_expected_head"
        and receipt.get("release_phase")
        == "failed_setup_before_provider"
        and receipt.get("implementation_started") is False
        and receipt.get("provider_dispatched") is False
        and isinstance(lifecycle, Mapping)
        and isinstance(owner, Mapping)
        and receipt.get("pool_state_cid") == evidence_state_cid
        and receipt.get("pool_state_sha256") == evidence_state_sha256
        and evidence_state.get("schema") == WORKTREE_POOL_SCHEMA
        and evidence_state.get("lease_token") == entry_id
        and evidence_state.get("state") == "leased"
        and evidence_state.get("lease_pid")
        == receipt.get("pool_lease_pid")
        and evidence_state.get("base_commit")
        == receipt.get("base_commit")
        and str(evidence_state.get("path") or "")
        == str(lexical_workspace)
        and str(evidence_state.get("repo_root") or "")
        == str(authoritative_repo)
        and evidence_state.get("repo_common_dir")
        == receipt.get("repo_common_dir")
        and str(evidence_state.get("branch") or "").removeprefix(
            "refs/heads/"
        )
        == expected_branch_name
        and lifecycle.get("workspace_path") == str(lexical_workspace)
        and str(lifecycle.get("branch") or "").removeprefix("refs/heads/")
        == expected_branch_name
        and str(lifecycle.get("repo_root") or "")
        == str(authoritative_repo)
        and owner.get("pid") == receipt.get("pool_lease_pid")
        and (
            expected_lifecycle is None
            or dict(lifecycle) == dict(expected_lifecycle)
        )
    )
    if not identity_valid:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_terminal_identity_mismatch",
        }
    try:
        raw_workspace.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_workspace_uninspectable",
            "error_type": type(exc).__name__,
        }
    else:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_workspace_present",
        }
    registration, registration_reason = _git_worktree_registration(
        authoritative_repo,
        raw_workspace,
    )
    if registration is not None or registration_reason != (
        "git_worktree_registration_absent"
    ):
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_git_registration_present",
            "git_registration_reason": registration_reason,
        }
    branch_ref = f"refs/heads/{expected_branch_name}"
    try:
        branch_probe = run_command(
            (
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{branch_ref}^{{commit}}",
            ),
            cwd=authoritative_repo,
            timeout_seconds=30,
        )
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_retained_branch_uninspectable",
            "error_type": type(exc).__name__,
        }
    if (
        branch_probe.returncode != 0
        or branch_probe.stdout.strip() != receipt.get("retained_branch_head")
    ):
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_retained_branch_changed",
        }
    try:
        lock_stat = lock_path.lstat()
    except FileNotFoundError:
        lock_present = False
    except OSError as exc:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_lock_uninspectable",
            "error_type": type(exc).__name__,
        }
    else:
        lock_present = True
        if stat_module.S_ISLNK(lock_stat.st_mode) or not stat_module.S_ISREG(
            lock_stat.st_mode
        ):
            return {
                **base,
                **detail,
                "status": "invalid",
                "cleanup_fenced": True,
                "reason": "missing_release_lock_unsafe",
            }
        lock_payload = _strict_worktree_pool_json_object(lock_path)
        try:
            lock_bytes = lock_path.read_bytes()
        except OSError as exc:
            return {
                **base,
                **detail,
                "status": "invalid",
                "cleanup_fenced": True,
                "reason": "missing_release_lock_unreadable",
                "error_type": type(exc).__name__,
            }
        if (
            lock_payload is None
            or receipt.get("pool_lock_cid")
            != _worktree_pool_payload_cid(lock_payload)
            or receipt.get("pool_lock_sha256")
            != "sha256:" + hashlib.sha256(lock_bytes).hexdigest()
        ):
            return {
                **base,
                **detail,
                "status": "invalid",
                "cleanup_fenced": True,
                "reason": "missing_release_lock_changed",
            }
    if proposal and not lock_present:
        return {
            **base,
            **detail,
            "status": "invalid",
            "cleanup_fenced": True,
            "reason": "missing_release_proposal_lock_absent",
        }
    return {
        **base,
        **detail,
        "status": "pending" if proposal else "valid",
        "valid": not proposal,
        "proposal_valid": proposal,
        "cleanup_fenced": True,
        "reason": (
            "missing_release_proposal_valid"
            if proposal
            else "missing_release_terminal_valid"
        ),
        "evidence": receipt,
        "lock_present": lock_present,
    }


_WORKTREE_POOL_MUTATION_GUARD_STATE = threading.local()


@contextmanager
def guarded_worktree_pool_mutation(
    *,
    repo_root: Path | str,
    worktree_root: Path | str,
    workspace_path: Path | str,
    expected_branch: str = "",
    operation: str,
    allow_quarantine_publication: bool = False,
    allow_absent_registration_metadata_cleanup: bool = False,
    expected_dead_owner_pid: int = 0,
    expected_current_owner_pid: int = 0,
    expected_branch_head: str = "",
) -> Iterator[dict[str, Any]]:
    """Serialize a workspace mutation with exact quarantine publication."""

    binding = worktree_pool_entry_guard_binding(
        worktree_root=worktree_root,
        workspace_path=workspace_path,
    )
    if binding.get("pooled") is not True:
        yield {
            "allowed": True,
            "pooled": False,
            "reason": str(binding.get("reason") or "not_a_pooled_workspace"),
            "operation": operation,
        }
        return
    if binding.get("guard_available") is not True:
        yield {
            "allowed": False,
            "pooled": True,
            "reason": "worktree_pool_mutation_guard_unavailable",
            "operation": operation,
            "binding": binding,
        }
        return

    from ..merge.checkout_lock import serialized_lock_update

    lock_path = Path(str(binding["lock_path"]))
    # The registry is thread-local and PID-bound.  A forked child can inherit
    # Python memory but may never inherit logical custody after CLOEXEC closes
    # the advisory-lock descriptor.  Same-thread nesting is required because a
    # synchronous merge callback may invoke daemon cleanup while its supervisor
    # already holds this exact entry guard.
    process_id = os.getpid()
    held = getattr(_WORKTREE_POOL_MUTATION_GUARD_STATE, "held", None)
    if not isinstance(held, dict) or getattr(
        _WORKTREE_POOL_MUTATION_GUARD_STATE,
        "pid",
        None,
    ) != process_id:
        held = {}
        _WORKTREE_POOL_MUTATION_GUARD_STATE.held = held
        _WORKTREE_POOL_MUTATION_GUARD_STATE.pid = process_id
    guard_key = str(lock_path)
    reentrant = int(held.get(guard_key, 0)) > 0
    manager = (
        None
        if reentrant
        else serialized_lock_update(
            lock_path,
            timeout_seconds=WORKTREE_POOL_MUTATION_GUARD_TIMEOUT_SECONDS,
        )
    )
    try:
        if manager is not None:
            manager.__enter__()
    except (OSError, RuntimeError, TimeoutError) as exc:
        yield {
            "allowed": False,
            "pooled": True,
            "reason": "worktree_pool_mutation_guard_unavailable",
            "operation": operation,
            "binding": binding,
            "error_type": type(exc).__name__,
            "error": str(exc)[-500:],
        }
        return
    held[guard_key] = int(held.get(guard_key, 0)) + 1
    try:
        rebound = worktree_pool_entry_guard_binding(
            worktree_root=worktree_root,
            workspace_path=workspace_path,
        )
        binding_fields = (
            "pooled",
            "guard_available",
            "entry_id",
            "pool_root",
            "lock_path",
            "pool_root_identity",
            "custody_identities",
            "pool_repo_root",
        )
        if any(binding.get(field) != rebound.get(field) for field in binding_fields):
            yield {
                "allowed": False,
                "pooled": True,
                "reason": "worktree_pool_mutation_guard_binding_changed",
                "operation": operation,
                "binding": binding,
                "current_binding": rebound,
                "reentrant": reentrant,
            }
            return
        try:
            quarantine = inspect_worktree_pool_quarantine(
                worktree_root=worktree_root,
                workspace_path=workspace_path,
                expected_branch=expected_branch,
            )
        except Exception as exc:
            quarantine = {
                "status": "invalid",
                "valid": False,
                "cleanup_fenced": True,
                "reason": "quarantine_inspection_failed",
                "error_type": type(exc).__name__,
            }
        if quarantine.get("cleanup_fenced") is True:
            decision = {
                "allowed": False,
                "pooled": True,
                "reason": (
                    "durable_worktree_pool_quarantine"
                    if quarantine.get("valid") is True
                    else "worktree_pool_quarantine_unverifiable"
                ),
                "operation": operation,
                "binding": binding,
                "quarantine": quarantine,
                "reentrant": reentrant,
            }
            if (
                allow_quarantine_publication
                and quarantine.get("valid") is True
            ):
                decision.update(
                    {
                        "allowed": True,
                        "reason": "exact_quarantine_publication_retry",
                    }
                )
            yield decision
        else:
            try:
                authoritative_repo_root = Path(repo_root).resolve(strict=True)
                recorded_repo_root = str(rebound.get("pool_repo_root") or "")
                if recorded_repo_root and Path(recorded_repo_root).resolve(
                    strict=True
                ) != authoritative_repo_root:
                    raise ValueError("pool repository authority changed")
            except (OSError, RuntimeError, ValueError) as exc:
                yield {
                    "allowed": False,
                    "pooled": True,
                    "reason": "worktree_pool_repository_binding_unavailable",
                    "operation": operation,
                    "binding": binding,
                    "current_binding": rebound,
                    "error_type": type(exc).__name__,
                    "reentrant": reentrant,
                }
                return
            locked_registration, registration_reason = (
                _git_worktree_registration(
                    authoritative_repo_root,
                    Path(workspace_path),
                )
            )
            if locked_registration is None:
                metadata_cleanup_admitted = False
                metadata_branch_disposition = ""
                metadata_branch_head = ""
                dead_owner_admitted = bool(
                    type(expected_dead_owner_pid) is int
                    and expected_dead_owner_pid > 1
                    and not pid_is_alive(expected_dead_owner_pid)
                )
                current_owner_admitted = bool(
                    type(expected_current_owner_pid) is int
                    and expected_current_owner_pid == os.getpid()
                    and pid_is_alive(expected_current_owner_pid)
                    and re.fullmatch(
                        r"[0-9a-f]{40}|[0-9a-f]{64}",
                        str(expected_branch_head or ""),
                    )
                )
                if (
                    allow_absent_registration_metadata_cleanup
                    and registration_reason
                    == "git_worktree_registration_absent"
                    and (dead_owner_admitted or current_owner_admitted)
                ):
                    try:
                        Path(workspace_path).lstat()
                    except FileNotFoundError:
                        branch_ref = str(expected_branch or "").removeprefix(
                            "refs/heads/"
                        )
                        branch_probe = run_command(
                            (
                                "git",
                                "show-ref",
                                "--verify",
                                "--quiet",
                                "--",
                                f"refs/heads/{branch_ref}",
                            ),
                            cwd=authoritative_repo_root,
                            timeout_seconds=30,
                        )
                        if bool(branch_ref) and branch_probe.returncode == 1:
                            metadata_cleanup_admitted = True
                            metadata_branch_disposition = "absent"
                        elif (
                            current_owner_admitted or dead_owner_admitted
                        ) and branch_probe.returncode == 0:
                            branch_head_probe = run_command(
                                (
                                    "git",
                                    "rev-parse",
                                    "--verify",
                                    "--end-of-options",
                                    f"refs/heads/{branch_ref}^{{commit}}",
                                ),
                                cwd=authoritative_repo_root,
                                timeout_seconds=30,
                            )
                            metadata_branch_head = (
                                branch_head_probe.stdout.strip()
                            )
                            if (
                                branch_head_probe.returncode == 0
                                and metadata_branch_head
                                == str(expected_branch_head)
                            ):
                                # An exact current lane, or an independently
                                # verified dead owner, may repair missing
                                # checkout sidecars while retaining the exact
                                # untouched task branch.  No source or ref
                                # bytes are mutated by this admission.
                                metadata_cleanup_admitted = True
                                metadata_branch_disposition = (
                                    "retained_exact_expected_head"
                                )
                    except (OSError, ValueError):
                        metadata_cleanup_admitted = False
                if metadata_cleanup_admitted:
                    yield {
                        "allowed": True,
                        "pooled": True,
                        "metadata_only": True,
                        "reason": "absent_worktree_metadata_cleanup_admitted",
                        "operation": operation,
                        "binding": binding,
                        "quarantine": quarantine,
                        "git_registration_reason": registration_reason,
                        "expected_dead_owner_pid": expected_dead_owner_pid,
                        "expected_current_owner_pid": (
                            expected_current_owner_pid
                        ),
                        "branch_disposition": metadata_branch_disposition,
                        "branch_head": metadata_branch_head,
                        "reentrant": reentrant,
                    }
                    return
                yield {
                    "allowed": False,
                    "pooled": True,
                    "reason": "git_worktree_registration_unavailable",
                    "operation": operation,
                    "binding": binding,
                    "quarantine": quarantine,
                    "git_registration_reason": registration_reason,
                    "reentrant": reentrant,
                }
                return
            git_lock_reason = str(
                locked_registration.get("locked") or ""
            )
            registration_locked = "locked" in locked_registration
            exact_pending_lock = bool(
                re.fullmatch(
                    r"agent-supervisor-quarantine-v1:sha256:[0-9a-f]{64}",
                    git_lock_reason,
                )
            )
            if registration_locked and not (
                allow_quarantine_publication and exact_pending_lock
            ):
                yield {
                    "allowed": False,
                    "pooled": True,
                    "reason": (
                        "pending_worktree_pool_quarantine"
                        if exact_pending_lock
                        else "foreign_git_worktree_lock"
                    ),
                    "operation": operation,
                    "binding": binding,
                    "quarantine": quarantine,
                    "git_registration": locked_registration,
                    "git_registration_reason": registration_reason,
                    "reentrant": reentrant,
                }
                return
            yield {
                "allowed": True,
                "pooled": True,
                "reason": "worktree_pool_mutation_guard_acquired",
                "operation": operation,
                "binding": binding,
                "quarantine": quarantine,
                "reentrant": reentrant,
            }
    finally:
        remaining = int(held.get(guard_key, 1)) - 1
        if remaining > 0:
            held[guard_key] = remaining
        else:
            held.pop(guard_key, None)
        if manager is not None:
            manager.__exit__(None, None, None)


def python_identifier_worktree_basename(*segments: object) -> str:
    """Return a deterministic ASCII identifier for a generated worktree.

    Ruff treats an absolute checkout root containing ``__init__.py`` as a
    package segment.  Keep generated checkout basenames valid Python
    identifiers so absolute-path validation does not report N999.  Branch and
    persisted legacy path names are deliberately outside this normalization.
    """

    normalized = [
        component
        for segment in segments
        if (component := re.sub(r"[^A-Za-z0-9_]+", "_", str(segment)).strip("_"))
    ]
    basename = "_".join(normalized) or "worktree"
    if basename[0].isdigit():
        basename = f"worktree_{basename}"
    return basename


def _run_command_with_timeout(
    run_command_fn: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> CommandResult:
    normalized_timeout = max(1, int(timeout_seconds))
    try:
        return run_command_fn(tuple(command), cwd=cwd, timeout_seconds=normalized_timeout)
    except TypeError as exc:
        if "timeout_seconds" not in str(exc):
            raise
        return run_command_fn(tuple(command), cwd=cwd, timeout=normalized_timeout)


def _trace_key(label: Optional[str], name: str) -> str:
    return name if not label else f"{label}_{name}"


def _compact_trace_result(result: CommandResult, limit: int) -> dict[str, Any]:
    return result.compact(limit=limit)


def git_status_paths(stdout: str) -> list[str]:
    """Return paths from ``git status --porcelain`` output."""

    return _shared_paths_from_git_status_porcelain(stdout)


def untracked_paths_from_git_status(stdout: str) -> list[str]:
    """Return untracked paths from ``git status --porcelain`` output."""

    return _shared_untracked_paths_from_git_status_porcelain(stdout)


def git_worktree_paths_from_porcelain(stdout: str) -> list[Path]:
    """Return registered Git worktree paths from porcelain output."""

    return _shared_git_worktree_paths_from_porcelain(stdout)


def normalize_worktree_path(path: str | Path) -> str:
    """Return a slash-normalized worktree-relative path string."""

    return str(path).replace("\\", "/").strip()


def unique_worktree_paths(paths: Sequence[str | Path]) -> list[str]:
    """Return non-empty worktree paths, slash-normalized and deduplicated in order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = normalize_worktree_path(path)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def repo_relative_worktree_path(path: str | Path, *, repo_root: Path) -> str:
    """Return ``path`` relative to ``repo_root`` when possible, normalized for Git pathspecs."""

    candidate = Path(path)
    absolute_candidate = candidate if candidate.is_absolute() else repo_root / candidate
    try:
        return absolute_candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return normalize_worktree_path(candidate.as_posix())


def worktree_path_allowed(path: str | Path, *, allowed_prefixes: Sequence[str]) -> bool:
    """Return whether a normalized worktree path is inside one of the allowed prefixes."""

    normalized = normalize_worktree_path(path)
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def resolve_worktree_file_edit_path(
    root: Path,
    path: str | Path,
    *,
    allowed_prefixes: Sequence[str],
    error_prefix: str = "Worktree edit",
) -> Path:
    """Resolve a complete-file edit path under ``root`` after traversal and allowlist checks."""

    raw_path = str(path)
    normalized = normalize_worktree_path(raw_path)
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"{error_prefix} path is unsafe: {raw_path!r}")
    if not worktree_path_allowed(normalized, allowed_prefixes=allowed_prefixes):
        raise ValueError(f"{error_prefix} path is outside daemon allowlist: {raw_path!r}")
    return root / normalized


def disallowed_worktree_paths(
    paths: Sequence[str | Path],
    *,
    allowed_prefixes: Sequence[str],
    ignored_paths: Sequence[str | Path] = (),
) -> list[str]:
    """Return changed worktree paths outside the daemon's write allowlist."""

    ignored = set(unique_worktree_paths(ignored_paths))
    disallowed: list[str] = []
    for path in unique_worktree_paths(paths):
        if path in ignored:
            continue
        if worktree_path_allowed(path, allowed_prefixes=allowed_prefixes):
            continue
        disallowed.append(path)
    return disallowed


def dirty_worktree_paths(
    *,
    repo_root: Path,
    paths: Sequence[str | Path],
    timeout_seconds: int = 60,
    run_command_fn: CommandRunner = run_command,
) -> list[str]:
    """Return dirty Git status paths for a normalized path subset."""

    normalized_paths = unique_worktree_paths(paths)
    if not normalized_paths:
        return []
    status = _run_command_with_timeout(
        run_command_fn,
        ("git", "status", "--porcelain", "--", *normalized_paths),
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    if not status.ok:
        return []
    return git_status_paths(status.stdout)


def worktree_diff(
    *,
    worktree_path: Path,
    paths: Sequence[str | Path],
    raw_trace: Optional[dict[str, Any]] = None,
    label: str = "worktree",
    timeout_seconds: int = 60,
    run_command_fn: CommandRunner = run_command,
    trace_result_formatter: TraceResultFormatter = _compact_trace_result,
) -> str:
    """Return a binary Git diff for a normalized worktree path subset.

    Untracked files are staged with intent-to-add before diffing so callers can
    harvest new complete-file changes without accepting the whole worktree.
    """

    normalized_paths = unique_worktree_paths(paths)
    if not normalized_paths:
        if raw_trace is not None:
            raw_trace[_trace_key(label, "status")] = {"skipped": True, "reason": "no_paths"}
            raw_trace[_trace_key(label, "untracked_paths")] = []
            raw_trace[_trace_key(label, "git_diff")] = {"skipped": True, "reason": "no_paths"}
        return ""

    status_result = _run_command_with_timeout(
        run_command_fn,
        ("git", "status", "--porcelain", "--", *normalized_paths),
        cwd=worktree_path,
        timeout_seconds=timeout_seconds,
    )
    if raw_trace is not None:
        raw_trace[_trace_key(label, "status")] = trace_result_formatter(status_result, 12000)

    untracked_paths = untracked_paths_from_git_status(status_result.stdout)
    if raw_trace is not None:
        raw_trace[_trace_key(label, "untracked_paths")] = untracked_paths

    if untracked_paths:
        add_intent = _run_command_with_timeout(
            run_command_fn,
            ("git", "add", "-N", "--", *untracked_paths),
            cwd=worktree_path,
            timeout_seconds=timeout_seconds,
        )
        if raw_trace is not None:
            raw_trace[_trace_key(label, "git_add_intent_to_add")] = trace_result_formatter(
                add_intent,
                12000,
            )

    diff_result = _run_command_with_timeout(
        run_command_fn,
        ("git", "diff", "--binary", "--", *normalized_paths),
        cwd=worktree_path,
        timeout_seconds=timeout_seconds,
    )
    if raw_trace is not None:
        raw_trace[_trace_key(label, "git_diff")] = trace_result_formatter(diff_result, 20000)
    return diff_result.stdout if diff_result.ok else ""


def worktree_file_edits(
    worktree_path: Path,
    changed_files: Sequence[str | Path],
    *,
    allowed_prefixes: Sequence[str],
) -> list[dict[str, str]]:
    """Read complete UTF-8 file edits from an isolated worktree for allowed paths."""

    edits: list[dict[str, str]] = []
    for path_text in unique_worktree_paths(changed_files):
        if not worktree_path_allowed(path_text, allowed_prefixes=allowed_prefixes):
            continue
        path = worktree_path / path_text
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        edits.append({"path": path_text, "content": content})
    return edits


def write_worktree_file_edits_to_root(
    root: Path,
    edits: Sequence[Mapping[str, Any]],
    *,
    allowed_prefixes: Sequence[str],
    error_prefix: str = "Worktree edit",
) -> None:
    """Write complete file edits into ``root`` after allowlist and traversal checks."""

    for edit in edits:
        path = resolve_worktree_file_edit_path(
            root,
            str(edit.get("path", "")),
            allowed_prefixes=allowed_prefixes,
            error_prefix=error_prefix,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(edit.get("content", "")), encoding="utf-8")


def pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` appears live and signalable."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_command_line(pid: int) -> str:
    """Return a process command line from procfs when available."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def pid_looks_like_worktree_owner(
    pid: int,
    *,
    repo_root: Path,
    worktree_path: Path,
    daemon_process_fragment: str = "",
    daemon_repo_hint_fragment: str = "--repo-root",
    worker_process_fragment: str = "codex",
) -> bool:
    """Return whether a live process plausibly owns a daemon worktree."""

    if not pid_is_alive(pid):
        return False
    command_line = pid_command_line(pid)
    if not command_line:
        return True
    normalized_repo = str(repo_root.resolve())
    normalized_worktree = str(worktree_path.resolve())
    if daemon_process_fragment and daemon_process_fragment in command_line:
        return normalized_repo in command_line or daemon_repo_hint_fragment in command_line
    if (
        worker_process_fragment
        and worker_process_fragment in command_line
        and normalized_worktree in command_line
    ):
        return True
    return False


def owner_pid_from_worktree(path: Path, owner: Mapping[str, Any]) -> Optional[int]:
    """Return the owner pid from metadata or a trailing ``_<pid>`` worktree name."""

    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid > 0:
        return pid
    match = re.search(r"_(\d+)$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning ``{}`` on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_worktree_owner_file(
    path: Path,
    *,
    schema: str,
    repo_root: Path,
    pid: Optional[int] = None,
    attempt: int = 0,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write a reusable daemon worktree-owner metadata file."""

    payload: dict[str, Any] = {
        "schema": schema,
        "pid": os.getpid() if pid is None else int(pid),
        "attempt": int(attempt),
        "repo_root": str(repo_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_at_epoch": time.time(),
    }
    if extra:
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@dataclass
class GitWorktreeSession:
    """State for a managed detached Git worktree lifecycle."""

    repo_root: Path
    path: Path
    metadata_rel: str
    owner_rel: str
    raw_trace: dict[str, Any] = field(default_factory=dict)
    add_result: Optional[CommandResult] = None

    @property
    def ready(self) -> bool:
        """Return whether the detached worktree was created successfully."""

        return bool(self.add_result and self.add_result.ok)


@dataclass
class WorktreeLease:
    """An exclusive task-local checkout borrowed from :class:`WorktreePool`.

    Pool bookkeeping deliberately lives outside ``path``.  The checkout can
    therefore be staged with ``git add -A`` without committing lease metadata.
    A lease must be released before the checkout can be handed to another
    task.  ``release(reusable=True)`` still discards a dirty checkout rather
    than silently erasing task output.
    """

    pool: "WorktreePool" = field(repr=False)
    path: Path
    cache_key: str
    base_ref: str
    base_commit: str
    branch_name: str
    dependency_paths: tuple[str, ...]
    reused: bool
    setup_seconds: float
    estimated_seconds_saved: float
    entry_id: str
    invalidation_reasons: tuple[str, ...] = ()
    reuse_authorizer: Optional[WorktreeReuseAuthorizer] = field(
        default=None,
        repr=False,
    )
    acquired_at_epoch: float = field(default_factory=time.time)
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def cache_hit(self) -> bool:
        """Return whether setup was served from a previously prepared entry."""

        return self.reused

    @property
    def metadata(self) -> dict[str, Any]:
        """Return stable, event-log-friendly reuse measurements."""

        return {
            "cache_key": self.cache_key,
            "base_ref": self.base_ref,
            "base_commit": self.base_commit,
            "branch": self.branch_name,
            "worktree_path": str(self.path),
            "dependency_paths": list(self.dependency_paths),
            "reused": self.reused,
            "cache_hit": self.cache_hit,
            "setup_seconds": round(self.setup_seconds, 6),
            "estimated_seconds_saved": round(self.estimated_seconds_saved, 6),
            "setup_time_saved_seconds": round(self.estimated_seconds_saved, 6),
            "invalidation_reason": self.invalidation_reasons[-1]
            if self.invalidation_reasons
            else "",
            "invalidation_reasons": list(self.invalidation_reasons),
            "entry_id": self.entry_id,
        }

    def release(
        self,
        *,
        reusable: bool = True,
        missing_release_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return this checkout to the pool, or discard it when unsafe."""

        if self._released:
            return {"released": False, "reason": "already_released", **self.metadata}
        result = self.pool.release(
            self,
            reusable=reusable,
            missing_release_context=missing_release_context,
        )
        if result.get("released") is True:
            self._released = True
        return result

    def __enter__(self) -> "WorktreeLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release(reusable=exc_type is None)


class WorktreePool:
    """Pool prepared Git worktrees without sharing task-local mutations.

    Entries are keyed by an explicit dependency/setup key and the resolved base
    commit.  Warm acquisition is allowed only for a registered, recursively
    clean checkout whose dependency HEADs match the recorded prepared state.
    Atomic sidecar lock files make a checkout exclusive across daemon processes.

    The pool intentionally does not infer a cache key.  Callers should include
    every input that affects preparation (for example submodule gitlinks,
    lockfiles, platform and dependency setup version) in ``cache_key``.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        worktree_root: Path,
        run_command_fn: CommandRunner = run_command,
        max_entries: int = 4,
        command_timeout_seconds: int = 120,
        state_dirname: str = ".pool-state",
        reuse_authorizer: Optional[WorktreeReuseAuthorizer] = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.worktree_root = worktree_root.resolve()
        self.run_command_fn = run_command_fn
        self.max_entries = max(1, int(max_entries))
        self.command_timeout_seconds = max(1, int(command_timeout_seconds))
        self.state_root = self.worktree_root / state_dirname
        self.reuse_authorizer = reuse_authorizer
        try:
            common_dir_result = _run_command_with_timeout(
                run_command_fn,
                ("git", "rev-parse", "--git-common-dir"),
                cwd=self.repo_root,
                timeout_seconds=self.command_timeout_seconds,
            )
        except OSError:
            common_dir_text = ""
        else:
            common_dir_text = common_dir_result.stdout.strip() if common_dir_result.ok else ""
        common_dir = Path(common_dir_text) if common_dir_text else self.repo_root / ".git"
        self.repo_common_dir = (
            common_dir if common_dir.is_absolute() else self.repo_root / common_dir
        ).resolve()
        self._metrics: dict[str, Any] = {
            "acquisitions": 0,
            "cold_acquisitions": 0,
            "warm_acquisitions": 0,
            "rejected_entries": 0,
            "reclaimed_dead_leases": 0,
            "released_entries": 0,
            "discarded_entries": 0,
            "setup_seconds": 0.0,
            "estimated_seconds_saved": 0.0,
            "rejection_reasons": {},
        }

    @property
    def metrics(self) -> dict[str, Any]:
        """Return a copy of measured process-local pool activity."""

        result = dict(self._metrics)
        result["rejection_reasons"] = dict(self._metrics["rejection_reasons"])
        result["setup_seconds"] = round(float(result["setup_seconds"]), 6)
        result["estimated_seconds_saved"] = round(float(result["estimated_seconds_saved"]), 6)
        attempted = int(result["acquisitions"])
        result["warm_hit_rate"] = (
            round(int(result["warm_acquisitions"]) / attempted, 6) if attempted else 0.0
        )
        result["idle_entries"] = sum(
            1
            for state in self._states()
            if state.get("state") == "idle" and not self._lock_path(state).exists()
        )
        return result

    def publish_exact_quarantine(
        self,
        *,
        workspace_path: Path | str,
        expected_pool_state_cid: str,
        expected_pool_lock_cid: str,
        expected_git_preimage_cid: str,
        board_namespace: str,
        task_id: str,
        canonical_task_cid: str,
        attempt: int,
        expected_branch: str,
        merge_target: str,
        lifecycle_record_id: str,
        lifecycle_fence: int,
        lifecycle_lease_id: str,
        owner_process_birth: Mapping[str, Any],
        predecessor_state_dir: Path | str,
        current_state_dir: Path | str,
        reason: str,
    ) -> dict[str, Any]:
        """Durably reserve one exact dead lease outside the reusable pool.

        The pool state and ownership lock deliberately remain unchanged and
        non-idle.  The immutable marker is published with a no-replace link,
        so a crash after publication but before lifecycle finalization is
        safely retried by comparing the complete content-addressed payload.
        """

        base: dict[str, Any] = {
            "published": False,
            "valid": False,
            "cleanup_fenced": False,
        }
        try:
            worktree_root = self.worktree_root.resolve(strict=True)
            workspace_raw = Path(workspace_path)
            workspace = workspace_raw.resolve(strict=True)
            workspace.relative_to(worktree_root)
            entry_id = _worktree_pool_entry_id_for_workspace(workspace)
            canonical_pool_root = worktree_root / ".pool-state"
            predecessor_raw = Path(predecessor_state_dir)
            current_raw = Path(current_state_dir)
            predecessor_state = predecessor_raw.resolve(strict=True)
            current_state = current_raw.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return {**base, "reason": "quarantine_identity_unresolvable"}
        if (
            not entry_id
            or workspace.parent != worktree_root
            or workspace_raw.is_symlink()
            or not workspace_raw.is_dir()
            or self.state_root != canonical_pool_root
            or canonical_pool_root.is_symlink()
            or not canonical_pool_root.is_dir()
            or predecessor_raw.is_symlink()
            or current_raw.is_symlink()
            or not predecessor_raw.is_dir()
            or not current_raw.is_dir()
            or type(attempt) is not int
            or attempt < 1
            or type(lifecycle_fence) is not int
            or lifecycle_fence < 1
            or set(owner_process_birth)
            != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
            or type(owner_process_birth.get("pid")) is not int
            or int(owner_process_birth.get("pid") or 0) <= 1
            or type(owner_process_birth.get("start_time_ticks")) is not int
            or int(owner_process_birth.get("start_time_ticks") or 0) <= 0
            or type(owner_process_birth.get("parent_pid")) is not int
            or type(owner_process_birth.get("boot_id")) is not str
            or any(
                type(value) is not str or not value
                for value in (
                    expected_pool_state_cid,
                    expected_pool_lock_cid,
                    expected_git_preimage_cid,
                    board_namespace,
                    task_id,
                    canonical_task_cid,
                    expected_branch,
                    merge_target,
                    lifecycle_record_id,
                    lifecycle_lease_id,
                    reason,
                )
            )
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", expected_pool_state_cid
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", expected_pool_lock_cid
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", expected_git_preimage_cid
            )
            is None
        ):
            return {**base, "reason": "quarantine_identity_invalid"}

        state_path = canonical_pool_root / f"{entry_id}.json"
        lock_path = canonical_pool_root / f"{entry_id}.lock"
        quarantine_root = canonical_pool_root / "quarantine"
        marker_path = quarantine_root / f"{entry_id}.json"

        try:
            with guarded_worktree_pool_mutation(
                repo_root=self.repo_root,
                worktree_root=worktree_root,
                workspace_path=workspace_raw,
                expected_branch=expected_branch,
                operation="publish_exact_worktree_pool_quarantine",
                allow_quarantine_publication=True,
            ) as publication_admission:
                if publication_admission.get("allowed") is not True:
                    return {
                        **base,
                        "cleanup_fenced": bool(
                            publication_admission.get("pooled") is True
                        ),
                        "reason": str(
                            publication_admission.get("reason")
                            or "quarantine_publication_guard_unavailable"
                        ),
                        "publication_admission": publication_admission,
                    }
                pool_state = _strict_worktree_pool_json_object(state_path)
                pool_lock = _strict_worktree_pool_json_object(lock_path)
                if pool_state is None or pool_lock is None:
                    return {
                        **base,
                        "reason": "quarantine_pool_evidence_unavailable",
                    }
                try:
                    state_cid = _worktree_pool_payload_cid(pool_state)
                    lock_cid = _worktree_pool_payload_cid(pool_lock)
                except (TypeError, ValueError):
                    return {
                        **base,
                        "reason": "quarantine_pool_evidence_invalid",
                    }
                owner_pid = int(owner_process_birth["pid"])
                if (
                    state_cid != expected_pool_state_cid
                    or lock_cid != expected_pool_lock_cid
                    or pool_state.get("schema") != WORKTREE_POOL_SCHEMA
                    or pool_state.get("lease_token") != entry_id
                    or pool_state.get("state") != "leased"
                    or pool_state.get("lease_pid") != owner_pid
                    or str(pool_state.get("path") or "") != str(workspace)
                    or str(pool_state.get("repo_root") or "")
                    != str(self.repo_root)
                    or str(pool_state.get("repo_common_dir") or "")
                    != str(self.repo_common_dir)
                    or str(pool_state.get("branch") or "").removeprefix(
                        "refs/heads/"
                    )
                    != str(expected_branch).removeprefix("refs/heads/")
                    or set(pool_lock) != {"pid", "created_at_epoch"}
                    or pool_lock.get("pid") != owner_pid
                ):
                    return {
                        **base,
                        "reason": "quarantine_pool_evidence_changed",
                    }

                # The branch is an implementation binding, independent of the
                # merge target.  Validate it separately after extracting it
                # from the exact leased state.
                branch = str(pool_state.get("branch") or "").removeprefix(
                    "refs/heads/"
                )
                if not branch:
                    return {
                        **base,
                        "reason": "quarantine_pool_evidence_changed",
                    }
                registration, registration_reason = _git_worktree_registration(
                    self.repo_root,
                    workspace_raw,
                )
                registered_head = str(
                    (registration or {}).get("HEAD") or ""
                )
                if (
                    registration is None
                    or registration_reason != "git_worktree_registration_exact"
                    or str(registration.get("branch") or "").removeprefix(
                        "refs/heads/"
                    )
                    != branch
                    or re.fullmatch(
                        r"[0-9a-f]{40}|[0-9a-f]{64}",
                        registered_head,
                    )
                    is None
                ):
                    return {
                        **base,
                        "reason": "quarantine_git_registration_unavailable",
                    }
                unsigned: dict[str, Any] = {
                    "schema": WORKTREE_POOL_QUARANTINE_SCHEMA,
                    "entry_id": entry_id,
                    "workspace_path": str(workspace),
                    "branch": branch,
                    "repo_root": str(self.repo_root),
                    "repo_common_dir": str(self.repo_common_dir),
                    "pool_state_cid": state_cid,
                    "pool_lock_cid": lock_cid,
                    "pool_lease_pid": owner_pid,
                    "board_namespace": board_namespace,
                    "task_id": task_id,
                    "canonical_task_cid": canonical_task_cid,
                    "attempt": attempt,
                    "merge_target": str(merge_target).removeprefix(
                        "refs/heads/"
                    ),
                    "lifecycle_record_id": lifecycle_record_id,
                    "lifecycle_fence": lifecycle_fence,
                    "lifecycle_lease_id": lifecycle_lease_id,
                    "owner_process_birth": dict(owner_process_birth),
                    "predecessor_state_dir": str(predecessor_state),
                    "current_state_dir": str(current_state),
                    "git_preimage_cid": expected_git_preimage_cid,
                    "git_registered_head": registered_head,
                    "reason": reason,
                }
                quarantine_id = _worktree_pool_payload_cid(unsigned)
                lock_reason = (
                    "agent-supervisor-quarantine-v1:" + quarantine_id
                )
                existing_lock_reason = str(registration.get("locked") or "")
                if (
                    "locked" in registration
                    and existing_lock_reason != lock_reason
                ):
                    return {
                        **base,
                        "cleanup_fenced": True,
                        "reason": "quarantine_git_worktree_foreign_lock",
                    }
                if "locked" not in registration:
                    lock_result = self._run(
                        (
                            "git",
                            "worktree",
                            "lock",
                            "--reason",
                            lock_reason,
                            str(workspace_raw),
                        ),
                        cwd=self.repo_root,
                    )
                    if not lock_result.ok:
                        return {
                            **base,
                            "reason": "quarantine_git_worktree_lock_failed",
                            "git_lock": lock_result.compact(limit=2000),
                        }
                locked_registration, locked_reason = (
                    _git_worktree_registration(self.repo_root, workspace_raw)
                )
                if (
                    locked_registration is None
                    or locked_reason != "git_worktree_registration_exact"
                    or str(locked_registration.get("locked") or "")
                    != lock_reason
                    or str(locked_registration.get("HEAD") or "")
                    != registered_head
                    or str(
                        locked_registration.get("branch") or ""
                    ).removeprefix("refs/heads/")
                    != branch
                ):
                    return {
                        **base,
                        "cleanup_fenced": True,
                        "reason": "quarantine_git_worktree_lock_unproven",
                    }
                marker = {
                    **unsigned,
                    "git_worktree_lock_reason": lock_reason,
                    "quarantine_id": quarantine_id,
                }

                if quarantine_root.exists() or quarantine_root.is_symlink():
                    if quarantine_root.is_symlink() or not quarantine_root.is_dir():
                        return {
                            **base,
                            "cleanup_fenced": True,
                            "reason": "quarantine_directory_unsafe",
                        }
                else:
                    quarantine_root.mkdir(mode=0o700)
                    # The marker directory itself is part of the durable
                    # authority. Seal its new parent-directory entry before a
                    # marker can be accepted; fsyncing only the child directory
                    # would not survive every crash/power-loss ordering.
                    parent_directory = os.open(
                        canonical_pool_root,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        os.fsync(parent_directory)
                    finally:
                        os.close(parent_directory)

                existing = inspect_worktree_pool_quarantine(
                    worktree_root=worktree_root,
                    workspace_path=workspace,
                    expected_branch=branch,
                )
                if existing.get("status") != "absent":
                    if (
                        existing.get("valid") is True
                        and existing.get("marker") == marker
                    ):
                        return {
                            **existing,
                            "published": True,
                            "idempotent": True,
                        }
                    return {
                        **base,
                        "cleanup_fenced": True,
                        "reason": "quarantine_marker_conflict",
                        "existing": existing,
                    }

                encoded = _canonical_worktree_pool_json_bytes(marker) + b"\n"
                temporary = quarantine_root / (
                    f".{entry_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(temporary, flags, 0o600)
                try:
                    view = memoryview(encoded)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError(
                                "quarantine marker write made no progress"
                            )
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                try:
                    os.link(
                        temporary,
                        marker_path,
                        follow_symlinks=False,
                    )
                    directory = os.open(
                        quarantine_root,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                except FileExistsError:
                    pass
                finally:
                    temporary.unlink(missing_ok=True)

                final = inspect_worktree_pool_quarantine(
                    worktree_root=worktree_root,
                    workspace_path=workspace,
                    expected_branch=branch,
                )
                if final.get("valid") is not True or final.get("marker") != marker:
                    return {
                        **base,
                        "cleanup_fenced": bool(final.get("cleanup_fenced")),
                        "reason": "quarantine_marker_publication_unproven",
                        "observed": final,
                    }
                return {**final, "published": True, "idempotent": False}
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                **base,
                "reason": "quarantine_publication_failed",
                "error_type": type(exc).__name__,
            }

    def acquire(
        self,
        *,
        cache_key: str,
        base_ref: str = "HEAD",
        branch_name: str = "",
        dependency_paths: Sequence[str | Path] = (),
        prepare: Optional[WorktreePrepare] = None,
        activate: Optional[WorktreePrepare] = None,
        worktree_path: Optional[Path] = None,
        authorize_reuse: Optional[WorktreeReuseAuthorizer] = None,
    ) -> WorktreeLease:
        """Exclusively acquire a clean prepared worktree.

        ``prepare`` is run only on the cold path, after Git creates the main
        checkout.  ``activate`` runs on both paths after binding the task branch
        and is intended for inexpensive task-specific submodule branch setup.
        Both callbacks must finish with all repositories clean.

        When supplied, ``authorize_reuse`` runs as a read-only ``preflight``
        before the sidecar claim and as ``claimed`` immediately after it.  A
        denial, malformed response, or authorization error fails closed: the
        existing entry is left untouched and acquisition continues on a
        distinct cold checkout.
        """

        normalized_key = str(cache_key).strip()
        if not normalized_key:
            raise ValueError("worktree pool cache_key must not be empty")
        dependencies = tuple(unique_worktree_paths(dependency_paths))
        for dependency in dependencies:
            candidate = Path(dependency)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"worktree dependency path is unsafe: {dependency!r}")
        base_commit = self._rev_parse(self.repo_root, base_ref)
        if not base_commit:
            raise RuntimeError(f"cannot resolve worktree pool base ref {base_ref!r}")
        requested_path = worktree_path.resolve() if worktree_path is not None else None
        if requested_path is not None:
            try:
                requested_path.relative_to(self.worktree_root)
            except ValueError as exc:
                raise ValueError("pooled worktree path must be inside worktree_root") from exc

        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._metrics["acquisitions"] += 1
        acquired_started = time.monotonic()
        reclaimed_dead_leases = self._reclaim_dead_leases()
        invalidation_reasons = ["dead_lease_owner"] * len(reclaimed_dead_leases)
        effective_authorizer = authorize_reuse or self.reuse_authorizer
        for state in self._states():
            if not self._state_matches(
                state, cache_key=normalized_key, base_commit=base_commit, dependencies=dependencies
            ):
                continue
            lock_path, admission_reason = self._try_claim_authorized(
                state,
                authorize_reuse=effective_authorizer,
            )
            if admission_reason:
                invalidation_reasons.append(admission_reason)
            if lock_path is None:
                continue
            if state.get("state") == "initializing":
                invalidation_reasons.append("stale_initializing_entry")
                self._reject_and_discard(
                    state, reason="stale_initializing_entry", lock_path=lock_path
                )
                continue
            valid, reason = self._validate_idle_entry(state)
            if not valid:
                invalidation_reasons.append(reason)
                self._reject_and_discard(state, reason=reason, lock_path=lock_path)
                continue
            path = Path(str(state["path"]))
            if requested_path is not None and path.resolve() != requested_path:
                if requested_path.exists():
                    invalidation_reasons.append("requested_path_exists")
                    self._reject_and_discard(
                        state, reason="requested_path_exists", lock_path=lock_path
                    )
                    continue
                requested_path.parent.mkdir(parents=True, exist_ok=True)
                move = self._run(
                    ("git", "worktree", "move", str(path), str(requested_path)),
                    cwd=self.repo_root,
                )
                if not move.ok:
                    invalidation_reasons.append("worktree_move_failed")
                    self._reject_and_discard(
                        state, reason="worktree_move_failed", lock_path=lock_path
                    )
                    continue
                path = requested_path
                state["path"] = str(path)
                self._write_state(state)
            bind = self._bind_task_branch(path, branch_name=branch_name, base_commit=base_commit)
            if not bind.ok:
                invalidation_reasons.append("branch_bind_failed")
                self._reject_and_discard(state, reason="branch_bind_failed", lock_path=lock_path)
                continue
            try:
                if activate is not None:
                    activate(path)
            except BaseException:
                self._discard_state(state)
                self._remove_lock(lock_path)
                raise
            active_clean, active_reason = self._repositories_clean(path, dependencies)
            expected_dependency_heads = {
                str(key): str(value)
                for key, value in dict(state.get("dependency_heads") or {}).items()
            }
            if (
                not active_clean
                or self._dependency_heads(path, dependencies) != expected_dependency_heads
            ):
                rejection_reason = (
                    active_reason if not active_clean else "dependency_head_mismatch_after_activate"
                )
                invalidation_reasons.append(rejection_reason)
                self._reject_and_discard(state, reason=rejection_reason, lock_path=lock_path)
                continue
            elapsed = time.monotonic() - acquired_started
            estimated_saved = max(0.0, float(state.get("cold_setup_seconds") or 0.0) - elapsed)
            state.update(
                {
                    "state": "leased",
                    "branch": branch_name,
                    "lease_pid": os.getpid(),
                    "leased_at_epoch": time.time(),
                    "last_used_at_epoch": time.time(),
                    "use_count": int(state.get("use_count") or 0) + 1,
                }
            )
            self._write_state(state)
            self._metrics["warm_acquisitions"] += 1
            self._metrics["setup_seconds"] += elapsed
            self._metrics["estimated_seconds_saved"] += estimated_saved
            lease = self._lease_from_state(
                state,
                base_ref=base_ref,
                branch_name=branch_name,
                reused=True,
                setup_seconds=elapsed,
                estimated_seconds_saved=estimated_saved,
                invalidation_reasons=tuple(invalidation_reasons),
            )
            lease.reuse_authorizer = effective_authorizer
            return lease

        lease = self._create_cold_entry(
            cache_key=normalized_key,
            base_ref=base_ref,
            base_commit=base_commit,
            branch_name=branch_name,
            dependencies=dependencies,
            prepare=prepare,
            activate=activate,
            requested_path=requested_path,
            started=acquired_started,
            invalidation_reasons=tuple(invalidation_reasons),
        )
        lease.reuse_authorizer = effective_authorizer
        return lease

    def _try_claim_authorized(
        self,
        state: Mapping[str, Any],
        *,
        authorize_reuse: Optional[WorktreeReuseAuthorizer],
    ) -> tuple[Optional[Path], str]:
        """Claim one entry only while its external lifecycle permits reuse."""

        # A leased or initializing checkout may contain recoverable crash
        # output.  Dead ownership permits a dedicated recovery/discard path;
        # it never makes that checkout a warm cache candidate.
        if state.get("state") != "idle":
            return None, "non_idle_entry_reserved"
        if authorize_reuse is not None:
            admitted, reason = self._authorize_entry_reuse(
                state,
                authorize_reuse=authorize_reuse,
                phase="preflight",
            )
            if not admitted:
                self._record_rejection(reason)
                return None, reason
        lock_path = self._try_claim(state, require_idle=True)
        if lock_path is None:
            return None, ""
        if authorize_reuse is not None:
            admitted, reason = self._authorize_entry_reuse(
                state,
                authorize_reuse=authorize_reuse,
                phase="claimed",
            )
            if not admitted:
                self._record_rejection(reason)
                # Release only the sidecar lock created by _try_claim.
                # Lifecycle denial never authorizes state/worktree cleanup.
                self._remove_lock(lock_path)
                return None, reason
        return lock_path, ""

    @staticmethod
    def _authorize_entry_reuse(
        state: Mapping[str, Any],
        *,
        authorize_reuse: WorktreeReuseAuthorizer,
        phase: str,
    ) -> tuple[bool, str]:
        """Fail closed around the claim of a lifecycle-sensitive pool entry."""

        path = Path(str(state.get("path") or ""))
        branch = str(state.get("branch") or "")
        try:
            allowed, reason = authorize_reuse(path, branch, phase)
        except Exception:
            return False, "worktree_reuse_authorization_unknown"
        normalized_reason = str(reason or "").strip()
        if allowed is not True:
            return (
                False,
                (
                    f"worktree_reuse_denied:{normalized_reason}"
                    if normalized_reason
                    else "worktree_reuse_authorization_denied"
                ),
            )
        return True, normalized_reason or "worktree_reuse_authorized"

    @contextmanager
    def lease(self, **kwargs: Any) -> Iterator[WorktreeLease]:
        """Context-manager form of :meth:`acquire`."""

        borrowed = self.acquire(**kwargs)
        try:
            yield borrowed
        except BaseException:
            borrowed.release(reusable=False)
            raise
        else:
            borrowed.release(reusable=True)

    def release(
        self,
        lease: WorktreeLease,
        *,
        reusable: bool = True,
        missing_release_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Release an exclusive lease, retaining it only after safe scrubbing."""

        state = self._read_state(lease.entry_id)
        lock_path = self.state_root / f"{lease.entry_id}.lock"
        if not state or str(state.get("lease_token")) != lease.entry_id:
            self._remove_lock(lock_path)
            return {"released": False, "reason": "lease_state_missing", **lease.metadata}
        authorize_reuse = lease.reuse_authorizer or self.reuse_authorizer
        if authorize_reuse is not None:
            admitted, admission_reason = self._authorize_entry_reuse(
                state,
                authorize_reuse=authorize_reuse,
                phase="claimed",
            )
            if not admitted:
                self._record_rejection(admission_reason)
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": admission_reason,
                    **lease.metadata,
                }
        if not reusable:
            try:
                lease.path.lstat()
            except FileNotFoundError:
                return self._discard_missing_current_lease_metadata(
                    lease,
                    observed_state=state,
                    lock_path=lock_path,
                    missing_release_context=missing_release_context,
                )
            except (OSError, ValueError):
                # An uninspectable path is not evidence of absence.  Preserve
                # its custody instead of falling through to destructive Git
                # or filesystem cleanup.
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_identity_unverifiable",
                    **lease.metadata,
                }
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            discarded = discard.get("removed") is True
            if discarded:
                self._metrics["discarded_entries"] += 1
            return {
                "released": discarded,
                "deferred": not discarded,
                "retryable": not discarded,
                "pooled": False,
                "reason": "reuse_disabled",
                "discard": discard,
                **lease.metadata,
            }

        clean, reason = self._repositories_clean(lease.path, lease.dependency_paths)
        if not clean:
            # Never reset an uncommitted task workspace merely to obtain a pool
            # hit.  Removing the managed checkout is deterministic and prevents
            # accidental cross-task mutation sharing.
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            self._record_rejection(reason)
            discarded = discard.get("removed") is True
            if discarded:
                self._metrics["discarded_entries"] += 1
            return {
                "released": discarded,
                "deferred": not discarded,
                "retryable": not discarded,
                "pooled": False,
                "reason": reason,
                "discard": discard,
                **lease.metadata,
            }

        restored, reason = self._restore_prepared_state(state)
        if not restored:
            discard = self._discard_state(state)
            self._remove_lock(lock_path)
            self._record_rejection(reason)
            discarded = discard.get("removed") is True
            if discarded:
                self._metrics["discarded_entries"] += 1
            return {
                "released": discarded,
                "deferred": not discarded,
                "retryable": not discarded,
                "pooled": False,
                "reason": reason,
                "discard": discard,
                **lease.metadata,
            }

        state.update(
            {
                "state": "idle",
                "branch": "",
                "lease_pid": 0,
                "released_at_epoch": time.time(),
                "last_used_at_epoch": time.time(),
            }
        )
        self._write_state(state)
        self._remove_lock(lock_path)
        self._metrics["released_entries"] += 1
        self._prune_excess_idle(exclude_entry_id=lease.entry_id)
        return {
            "released": True,
            "pooled": True,
            "reason": "clean_prepared_workspace",
            **lease.metadata,
        }

    def _discard_missing_current_lease_metadata(
        self,
        lease: WorktreeLease,
        *,
        observed_state: Mapping[str, Any],
        lock_path: Path,
        missing_release_context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Release only exact current custody for an already-gone checkout.

        This is deliberately narrower than normal worktree disposal.  It is
        used when setup loses its checkout after acquisition but before a
        provider can run.  The checkout must be absent both lexically and from
        Git's administrative registry, the in-memory lease must still match
        the exact pool state and lock owned by this PID, and the task branch
        must either be absent or remain at the untouched acquisition commit.
        The exact branch is retained; this path mutates pool metadata only.
        """

        context = (
            dict(missing_release_context)
            if isinstance(missing_release_context, Mapping)
            else {}
        )
        observed_lock = _strict_worktree_pool_json_object(lock_path)
        if (
            observed_state.get("lease_pid") != os.getpid()
            or not isinstance(observed_lock, Mapping)
            or observed_lock.get("pid") != os.getpid()
        ):
            return {
                "released": False,
                "deferred": True,
                "retryable": True,
                "reason": "missing_workspace_pool_custody_changed",
                **lease.metadata,
            }
        lifecycle = context.get("lifecycle")
        lifecycle_fields = {
            "schema",
            "record_id",
            "task_id",
            "canonical_task_cid",
            "attempt",
            "lane_id",
            "state",
            "owner",
            "lease_id",
            "fence",
            "workspace_path",
            "branch",
            "merge_target",
            "created_at",
            "updated_at",
            "expires_at",
            "repo_root",
            "state_dir",
            "terminal_reason",
        }
        owner = lifecycle.get("owner") if isinstance(lifecycle, Mapping) else None
        context_valid = bool(
            set(context) == {
                "release_phase",
                "implementation_started",
                "provider_dispatched",
                "lifecycle",
            }
            and context.get("release_phase")
            == "failed_setup_before_provider"
            and context.get("implementation_started") is False
            and context.get("provider_dispatched") is False
            and isinstance(lifecycle, Mapping)
            and set(lifecycle) == lifecycle_fields
            and lifecycle.get("state") in {"preparing", "active"}
            and isinstance(owner, Mapping)
            and set(owner)
            == {"pid", "start_time_ticks", "boot_id", "parent_pid"}
            and owner.get("pid") == os.getpid()
            and type(owner.get("start_time_ticks")) is int
            and int(owner.get("start_time_ticks") or 0) > 0
            and lifecycle.get("workspace_path") == str(lease.path)
            and str(lifecycle.get("branch") or "").removeprefix(
                "refs/heads/"
            )
            == str(lease.branch_name or "").removeprefix("refs/heads/")
            and lifecycle.get("repo_root") == str(self.repo_root)
            and type(lifecycle.get("attempt")) is int
            and int(lifecycle.get("attempt") or 0) >= 1
            and type(lifecycle.get("fence")) is int
            and int(lifecycle.get("fence") or 0) >= 1
            and all(
                isinstance(lifecycle.get(field), str)
                and bool(lifecycle.get(field))
                for field in (
                    "record_id",
                    "task_id",
                    "canonical_task_cid",
                    "lease_id",
                    "merge_target",
                    "state_dir",
                )
            )
        )
        if not context_valid:
            return {
                "released": False,
                "deferred": True,
                "retryable": True,
                "reason": "missing_workspace_lifecycle_context_unavailable",
                **lease.metadata,
            }

        with guarded_worktree_pool_mutation(
            repo_root=self.repo_root,
            worktree_root=self.worktree_root,
            workspace_path=lease.path,
            expected_branch=lease.branch_name,
            operation="worktree_pool_discard_missing_current_lease_metadata",
            allow_absent_registration_metadata_cleanup=True,
            expected_current_owner_pid=os.getpid(),
            expected_branch_head=lease.base_commit,
        ) as mutation_admission:
            if (
                mutation_admission.get("allowed") is not True
                or mutation_admission.get("metadata_only") is not True
            ):
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": str(
                        mutation_admission.get("reason")
                        or "missing_workspace_metadata_cleanup_denied"
                    ),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            if mutation_admission.get("branch_disposition") != (
                "retained_exact_expected_head"
            ):
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_retained_branch_unproven",
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }

            state_path = self._state_path(lease.entry_id)
            current_state = self._read_state(lease.entry_id)
            current_lock = _strict_worktree_pool_json_object(lock_path)
            expected_branch = str(lease.branch_name or "").removeprefix(
                "refs/heads/"
            )
            state_matches = bool(
                current_state == dict(observed_state)
                and current_state.get("schema") == WORKTREE_POOL_SCHEMA
                and current_state.get("lease_token") == lease.entry_id
                and current_state.get("state") == "leased"
                and current_state.get("lease_pid") == os.getpid()
                and str(current_state.get("path") or "") == str(lease.path)
                and str(current_state.get("repo_root") or "")
                == str(self.repo_root)
                and str(current_state.get("repo_common_dir") or "")
                == str(self.repo_common_dir)
                and str(current_state.get("cache_key") or "")
                == lease.cache_key
                and str(current_state.get("base_commit") or "")
                == lease.base_commit
                and str(current_state.get("branch") or "").removeprefix(
                    "refs/heads/"
                )
                == expected_branch
                and tuple(
                    str(item)
                    for item in current_state.get("dependency_paths") or ()
                )
                == lease.dependency_paths
            )
            lock_matches = bool(
                isinstance(current_lock, dict)
                and set(current_lock) == {"pid", "created_at_epoch"}
                and current_lock.get("pid") == os.getpid()
            )
            if not state_matches or not lock_matches:
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_pool_custody_changed",
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            try:
                active_state_bytes = state_path.read_bytes()
                active_lock_bytes = lock_path.read_bytes()
            except OSError as exc:
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_pool_custody_unreadable",
                    "error_type": type(exc).__name__,
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }

            try:
                lease.path.lstat()
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_identity_unverifiable",
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            else:
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_reappeared",
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            registration, registration_reason = _git_worktree_registration(
                self.repo_root,
                lease.path,
            )
            if (
                registration is not None
                or registration_reason != "git_worktree_registration_absent"
            ):
                return {
                    "released": False,
                    "deferred": True,
                    "retryable": True,
                    "reason": "missing_workspace_registration_changed",
                    "git_registration_reason": registration_reason,
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }

            terminal_state_path = (
                self.state_root / f".{lease.entry_id}.released-state"
            )
            receipt_path = (
                self.state_root / f".{lease.entry_id}.released-receipt"
            )

            def fsync_state_root() -> None:
                directory = os.open(
                    self.state_root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)

            receipt_unsigned: dict[str, Any] = {
                "schema": WORKTREE_POOL_MISSING_RELEASE_SCHEMA,
                "entry_id": lease.entry_id,
                "workspace_path": str(lease.path),
                "branch": lease.branch_name,
                "base_commit": lease.base_commit,
                "repo_root": str(self.repo_root),
                "repo_common_dir": str(self.repo_common_dir),
                "pool_state_cid": _worktree_pool_payload_cid(current_state),
                "pool_state_sha256": "sha256:"
                + hashlib.sha256(active_state_bytes).hexdigest(),
                "pool_lock_cid": _worktree_pool_payload_cid(current_lock),
                "pool_lock_sha256": "sha256:"
                + hashlib.sha256(active_lock_bytes).hexdigest(),
                "pool_lease_pid": os.getpid(),
                "retained_branch_head": lease.base_commit,
                "branch_disposition": str(
                    mutation_admission.get("branch_disposition") or ""
                ),
                "release_phase": "failed_setup_before_provider",
                "implementation_started": False,
                "provider_dispatched": False,
                "lifecycle": dict(lifecycle),
            }
            receipt = {
                **receipt_unsigned,
                "evidence_id": _worktree_pool_payload_cid(
                    receipt_unsigned
                ),
            }
            existing_receipt = _strict_worktree_pool_json_object(receipt_path)
            if receipt_path.exists() and existing_receipt != receipt:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_receipt_conflict",
                    "receipt_path": str(receipt_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            if existing_receipt is None:
                encoded_receipt = (
                    _canonical_worktree_pool_json_bytes(receipt) + b"\n"
                )
                temporary_receipt = self.state_root / (
                    f".{lease.entry_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                descriptor: int | None = None
                try:
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(temporary_receipt, flags, 0o600)
                    view = memoryview(encoded_receipt)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError(
                                "missing release receipt write made no progress"
                            )
                        view = view[written:]
                    os.fsync(descriptor)
                    os.close(descriptor)
                    descriptor = None
                    os.link(
                        temporary_receipt,
                        receipt_path,
                        follow_symlinks=False,
                    )
                    fsync_state_root()
                except FileExistsError:
                    pass
                except OSError as exc:
                    return {
                        "released": False,
                        "deferred": True,
                        "retryable": True,
                        "reason": (
                            "missing_workspace_terminal_receipt_publish_failed"
                        ),
                        "error_type": type(exc).__name__,
                        "receipt_path": str(receipt_path),
                        "mutation_admission": mutation_admission,
                        **lease.metadata,
                    }
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
                    temporary_receipt.unlink(missing_ok=True)
                if _strict_worktree_pool_json_object(receipt_path) != receipt:
                    return {
                        "released": False,
                        "deferred": False,
                        "retryable": False,
                        "cleanup_fenced": True,
                        "reason": (
                            "missing_workspace_terminal_receipt_unproven"
                        ),
                        "receipt_path": str(receipt_path),
                        "mutation_admission": mutation_admission,
                        **lease.metadata,
                    }

            try:
                terminal_state_path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_state_uninspectable",
                    "error_type": type(exc).__name__,
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            else:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_state_conflict",
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }

            try:
                state_path.rename(terminal_state_path)
                fsync_state_root()
            except OSError as exc:
                rollback_error = ""
                rollback_durable = False
                try:
                    if (
                        terminal_state_path.is_file()
                        and not state_path.exists()
                    ):
                        terminal_state_path.rename(state_path)
                    fsync_state_root()
                    rollback_durable = True
                except OSError as rollback_exc:
                    rollback_error = (
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )[-500:]
                restored_state = _strict_worktree_pool_json_object(state_path)
                restored_lock = _strict_worktree_pool_json_object(lock_path)
                try:
                    restored_state_bytes = state_path.read_bytes()
                    restored_lock_bytes = lock_path.read_bytes()
                except OSError:
                    restored_state_bytes = b""
                    restored_lock_bytes = b""
                rollback_exact = bool(
                    rollback_durable
                    and restored_state == current_state
                    and restored_lock == current_lock
                    and restored_state_bytes == active_state_bytes
                    and restored_lock_bytes == active_lock_bytes
                    and not terminal_state_path.exists()
                )
                return {
                    "released": False,
                    "deferred": rollback_exact,
                    "retryable": rollback_exact,
                    "cleanup_fenced": not rollback_exact,
                    "reason": (
                        "missing_workspace_terminal_publish_rolled_back"
                        if rollback_exact
                        else "missing_workspace_terminal_publish_indeterminate"
                    ),
                    "error_type": type(exc).__name__,
                    "rollback_error": rollback_error,
                    "rollback_exact": rollback_exact,
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            try:
                terminal_state_bytes = terminal_state_path.read_bytes()
            except OSError as exc:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_evidence_unreadable",
                    "error_type": type(exc).__name__,
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            if terminal_state_bytes != active_state_bytes:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_evidence_changed",
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }
            terminal_inspection = (
                inspect_worktree_pool_missing_release_terminal(
                    repo_root=self.repo_root,
                    worktree_root=self.worktree_root,
                    workspace_path=lease.path,
                    expected_branch=lease.branch_name,
                    expected_lifecycle=lifecycle,
                )
            )
            if terminal_inspection.get("valid") is not True:
                return {
                    "released": False,
                    "deferred": False,
                    "retryable": False,
                    "cleanup_fenced": True,
                    "reason": "missing_workspace_terminal_evidence_unverifiable",
                    "terminal_inspection": terminal_inspection,
                    "terminal_state_path": str(terminal_state_path),
                    "mutation_admission": mutation_admission,
                    **lease.metadata,
                }

            # The authoritative lease state is now durably terminal.  Lock
            # unlink and its directory fsync are cleanup only: failures may
            # retain an orphan lock as evidence, but can never be reported as
            # retryable active custody or trigger source/ref mutation.
            lock_cleanup_error = ""
            lock_cleanup_durable = False
            try:
                lock_path.unlink()
                fsync_state_root()
                lock_cleanup_durable = True
            except OSError as exc:
                lock_cleanup_error = f"{type(exc).__name__}: {exc}"[-500:]
            try:
                orphan_lock_retained = lock_path.exists()
            except OSError:
                orphan_lock_retained = True
            self._metrics["discarded_entries"] += 1
            return {
                "released": True,
                "pooled": False,
                "reason": "reuse_disabled",
                "metadata_only": True,
                "branch_disposition": str(
                    mutation_admission.get("branch_disposition") or ""
                ),
                "terminal_state_path": str(terminal_state_path),
                "terminal_state_cid": _worktree_pool_payload_cid(
                    current_state
                ),
                "terminal_state_sha256": "sha256:"
                + hashlib.sha256(active_state_bytes).hexdigest(),
                "terminal_state_durable": True,
                "lock_cleanup_durable": lock_cleanup_durable,
                "cleanup_degraded": not lock_cleanup_durable,
                "lock_cleanup_error": lock_cleanup_error,
                "orphan_lock_retained": orphan_lock_retained,
                "discard": {
                    "path": str(lease.path),
                    "removed": True,
                    "reason": "missing_workspace_metadata_discarded",
                    "metadata_only": True,
                },
                **lease.metadata,
            }

    def finalize_missing_release_proposal(
        self,
        *,
        workspace_path: Path | str,
        expected_branch: str,
        expected_lifecycle: Mapping[str, Any],
        proc_root: Path = Path("/proc"),
    ) -> dict[str, Any]:
        """Promote one exact dead-owner proposal to durable terminal evidence.

        This is the restart seam for a crash after receipt publication but
        before the active pool state was renamed.  It never authorizes source
        or ref mutation and never treats the proposal itself as terminal.
        """

        initial = inspect_worktree_pool_missing_release_terminal(
            repo_root=self.repo_root,
            worktree_root=self.worktree_root,
            workspace_path=workspace_path,
            expected_branch=expected_branch,
            expected_lifecycle=expected_lifecycle,
        )
        if initial.get("proposal_valid") is not True:
            return {
                "finalized": False,
                "cleanup_fenced": initial.get("cleanup_fenced") is True,
                "reason": str(
                    initial.get("reason")
                    or "missing_release_proposal_unavailable"
                ),
                "inspection": initial,
            }
        evidence = initial.get("evidence")
        lifecycle = (
            evidence.get("lifecycle")
            if isinstance(evidence, Mapping)
            else None
        )
        owner_payload = (
            lifecycle.get("owner")
            if isinstance(lifecycle, Mapping)
            else None
        )
        try:
            owner = ProcessBirthIdentity.from_dict(
                owner_payload if isinstance(owner_payload, Mapping) else None
            )
            liveness = owner_liveness(owner, proc_root=proc_root)
        except (TypeError, ValueError):
            liveness = OwnerLiveness.UNKNOWN
            owner = ProcessBirthIdentity(pid=0, start_time_ticks=0)
        if liveness is not OwnerLiveness.DEAD:
            return {
                "finalized": False,
                "cleanup_fenced": True,
                "reason": "missing_release_proposal_owner_not_dead",
                "owner_liveness": liveness.value,
                "inspection": initial,
            }

        base_commit = str(evidence.get("base_commit") or "")
        with guarded_worktree_pool_mutation(
            repo_root=self.repo_root,
            worktree_root=self.worktree_root,
            workspace_path=workspace_path,
            expected_branch=expected_branch,
            operation="worktree_pool_finalize_missing_release_proposal",
            allow_absent_registration_metadata_cleanup=True,
            expected_dead_owner_pid=owner.pid,
            expected_branch_head=base_commit,
        ) as admission:
            if (
                admission.get("allowed") is not True
                or admission.get("metadata_only") is not True
                or admission.get("branch_disposition")
                != "retained_exact_expected_head"
            ):
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": str(
                        admission.get("reason")
                        or "missing_release_proposal_guard_denied"
                    ),
                    "mutation_admission": admission,
                    "inspection": initial,
                }
            current = inspect_worktree_pool_missing_release_terminal(
                repo_root=self.repo_root,
                worktree_root=self.worktree_root,
                workspace_path=workspace_path,
                expected_branch=expected_branch,
                expected_lifecycle=expected_lifecycle,
            )
            if (
                current.get("proposal_valid") is not True
                or current.get("evidence") != evidence
            ):
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": "missing_release_proposal_changed",
                    "inspection": current,
                    "mutation_admission": admission,
                }
            active_path = Path(str(current["active_state_path"]))
            terminal_path = Path(str(current["terminal_state_path"]))
            lock_path = Path(str(current["lock_path"]))
            try:
                active_bytes = active_path.read_bytes()
            except OSError as exc:
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": "missing_release_proposal_state_unreadable",
                    "error_type": type(exc).__name__,
                    "inspection": current,
                }

            def fsync_state_root() -> None:
                descriptor = os.open(
                    self.state_root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

            try:
                terminal_path.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": "missing_release_proposal_terminal_uninspectable",
                    "error_type": type(exc).__name__,
                }
            else:
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": "missing_release_proposal_terminal_conflict",
                }
            try:
                active_path.rename(terminal_path)
                fsync_state_root()
            except OSError as exc:
                rollback_exact = False
                try:
                    if terminal_path.is_file() and not active_path.exists():
                        terminal_path.rename(active_path)
                    fsync_state_root()
                    rollback_exact = bool(
                        active_path.read_bytes() == active_bytes
                        and not terminal_path.exists()
                    )
                except OSError:
                    rollback_exact = False
                return {
                    "finalized": False,
                    "deferred": rollback_exact,
                    "retryable": rollback_exact,
                    "cleanup_fenced": not rollback_exact,
                    "reason": (
                        "missing_release_proposal_promotion_rolled_back"
                        if rollback_exact
                        else "missing_release_proposal_promotion_indeterminate"
                    ),
                    "error_type": type(exc).__name__,
                }
            terminal = inspect_worktree_pool_missing_release_terminal(
                repo_root=self.repo_root,
                worktree_root=self.worktree_root,
                workspace_path=workspace_path,
                expected_branch=expected_branch,
                expected_lifecycle=expected_lifecycle,
            )
            if terminal.get("valid") is not True:
                return {
                    "finalized": False,
                    "cleanup_fenced": True,
                    "reason": "missing_release_promoted_terminal_unverifiable",
                    "inspection": terminal,
                }
            lock_cleanup_error = ""
            try:
                lock_path.unlink()
                fsync_state_root()
            except FileNotFoundError:
                pass
            except OSError as exc:
                lock_cleanup_error = f"{type(exc).__name__}: {exc}"[-500:]
            final = inspect_worktree_pool_missing_release_terminal(
                repo_root=self.repo_root,
                worktree_root=self.worktree_root,
                workspace_path=workspace_path,
                expected_branch=expected_branch,
                expected_lifecycle=expected_lifecycle,
            )
            return {
                "finalized": final.get("valid") is True,
                "cleanup_fenced": True,
                "reason": (
                    "missing_release_proposal_promoted"
                    if final.get("valid") is True
                    else "missing_release_promoted_terminal_unverifiable"
                ),
                "lock_cleanup_error": lock_cleanup_error,
                "inspection": final,
            }

    def invalidate(self, *, cache_key: Optional[str] = None) -> dict[str, Any]:
        """Discard idle entries, optionally limited to one setup cache key."""

        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for state in self._states():
            if cache_key is not None and str(state.get("cache_key")) != str(cache_key):
                continue
            lock_path, admission_reason = self._try_claim_authorized(
                state,
                authorize_reuse=self.reuse_authorizer,
            )
            if lock_path is None:
                skipped.append(
                    {
                        "path": str(state.get("path") or ""),
                        "reason": admission_reason or "leased",
                    }
                )
                continue
            removed.append(self._discard_state(state))
            self._remove_lock(lock_path)
        return {"removed": removed, "skipped": skipped}

    def reconcile_orphaned_metadata(
        self,
        *,
        max_entries: int = 100,
    ) -> dict[str, Any]:
        """Remove bounded dead-lease sidecars after proving their checkout is gone.

        Worktree reconciliation normally starts from ``git worktree list``.
        Consequently, a daemon that dies before releasing its pool lease can
        leave JSON and lock sidecars behind after another recovery path removes
        both the checkout and its task branch. Such records cannot be reached
        by the normal worktree scan, and records for old base commits are not
        considered by :meth:`acquire`.

        This cleanup deliberately removes metadata only. Any live owner,
        present checkout, surviving branch, malformed identity, or concurrent
        state replacement is preserved for a later operator/recovery pass.
        The existing entry-specific claim guard serializes stale-lock takeover,
        and the state is re-read before unlinking so a replacement lease cannot
        be deleted from an earlier observation.
        """

        limit = max(0, int(max_entries))
        states = [
            state
            for state in self._states()
            if state.get("state") in {"initializing", "leased"}
        ]
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        def skip(
            state: Mapping[str, Any],
            reason: str,
            **detail: Any,
        ) -> None:
            skipped.append(
                {
                    "entry_id": str(state.get("lease_token") or ""),
                    "path": str(state.get("path") or ""),
                    "branch": str(state.get("branch") or ""),
                    "reason": reason,
                    **detail,
                }
            )

        for state in states[:limit]:
            entry_id = str(state.get("lease_token") or "")
            state_path = self._state_path(entry_id)
            lock_path = self._lock_path(state)
            if (
                not re.fullmatch(r"[A-Za-z0-9._-]+", entry_id)
                or state_path.is_symlink()
                or lock_path.is_symlink()
            ):
                skip(state, "unsafe_metadata_identity")
                continue
            if (
                str(state.get("repo_root") or "") != str(self.repo_root)
                or str(state.get("repo_common_dir") or "")
                != str(self.repo_common_dir)
            ):
                skip(state, "repository_identity_mismatch")
                continue
            try:
                lease_pid = int(state.get("lease_pid") or 0)
            except (TypeError, ValueError):
                skip(state, "lease_owner_unverifiable")
                continue
            if lease_pid <= 0:
                skip(state, "lease_owner_unverifiable")
                continue
            if pid_is_alive(lease_pid):
                skip(state, "live_lease_owner", owner_pid=lease_pid)
                continue

            if lock_path.exists():
                lock = read_json_object(lock_path)
                if not lock:
                    skip(state, "lock_owner_unverifiable")
                    continue
                try:
                    lock_pid = int(lock.get("pid") or 0)
                except (TypeError, ValueError):
                    skip(state, "lock_owner_unverifiable")
                    continue
                if lock_pid <= 0:
                    skip(state, "lock_owner_unverifiable")
                    continue
                if pid_is_alive(lock_pid):
                    skip(state, "live_lock_owner", owner_pid=lock_pid)
                    continue

            raw_path = str(state.get("path") or "").strip()
            unresolved_workspace_path = Path(raw_path)
            try:
                if (
                    not unresolved_workspace_path.is_absolute()
                    or unresolved_workspace_path.is_symlink()
                ):
                    raise ValueError("workspace path is not an absolute directory")
                workspace_path = unresolved_workspace_path.resolve(
                    strict=False
                )
                workspace_path.relative_to(self.worktree_root)
            except (OSError, RuntimeError, ValueError):
                skip(state, "workspace_path_invalid")
                continue
            if (
                not raw_path
                or workspace_path == self.worktree_root
                or workspace_path.exists()
            ):
                skip(state, "workspace_present_or_unsafe")
                continue

            branch = str(state.get("branch") or "").strip()
            branch_check = self._run(
                ("git", "check-ref-format", "--branch", branch),
                cwd=self.repo_root,
            )
            if not branch or not branch_check.ok:
                skip(state, "branch_identity_unverifiable")
                continue
            branch_ref = (
                branch
                if branch.startswith("refs/heads/")
                else f"refs/heads/{branch}"
            )
            branch_presence, branch_probe = self._branch_ref_presence(
                branch_ref
            )
            if branch_presence == "unverifiable":
                skip(
                    state,
                    "branch_presence_unverifiable",
                    branch_probe=branch_probe,
                )
                continue
            if branch_presence == "present":
                skip(state, "branch_present")
                continue

            claimed_lock = self._try_claim(
                state,
                allow_absent_registration_metadata_cleanup=True,
            )
            if claimed_lock is None:
                skip(state, "lease_or_claim_owner_active")
                continue
            try:
                current = self._read_state(entry_id)
                if current != dict(state):
                    skip(state, "state_changed_during_cleanup")
                    continue
                try:
                    current_lease_pid = int(current.get("lease_pid") or 0)
                except (TypeError, ValueError):
                    skip(state, "lease_owner_changed_or_unverifiable")
                    continue
                if current_lease_pid <= 0:
                    skip(state, "lease_owner_changed_or_unverifiable")
                    continue
                if pid_is_alive(current_lease_pid):
                    skip(
                        state,
                        "lease_owner_changed_or_live",
                        owner_pid=current_lease_pid,
                    )
                    continue
                unresolved_current_path = Path(
                    str(current.get("path") or "")
                )
                current_path = unresolved_current_path.resolve(
                    strict=False
                )
                if (
                    unresolved_current_path.is_symlink()
                    or current_path.exists()
                ):
                    skip(state, "workspace_reappeared_during_cleanup")
                    continue
                current_branch = str(current.get("branch") or "").strip()
                current_branch_ref = (
                    current_branch
                    if current_branch.startswith("refs/heads/")
                    else f"refs/heads/{current_branch}"
                )
                current_branch_presence, current_branch_probe = (
                    self._branch_ref_presence(current_branch_ref)
                )
                if current_branch_presence == "unverifiable":
                    skip(
                        state,
                        "branch_recheck_unverifiable",
                        branch_probe=current_branch_probe,
                    )
                    continue
                if current_branch_presence == "present":
                    skip(state, "branch_reappeared_during_cleanup")
                    continue
                state_path.unlink()
                removed.append(
                    {
                        "entry_id": entry_id,
                        "path": str(workspace_path),
                        "branch": branch,
                        "lease_pid": lease_pid,
                        "reason": "dead_lease_workspace_and_branch_absent",
                    }
                )
            except (OSError, RuntimeError) as exc:
                skip(
                    state,
                    "metadata_cleanup_failed",
                    error_type=type(exc).__name__,
                )
            finally:
                self._remove_lock(claimed_lock)

        return {
            "attempted": True,
            "max_entries": limit,
            "candidate_count": len(states),
            "inspected_count": min(len(states), limit),
            "removed_count": len(removed),
            "skipped_count": len(skipped),
            "truncated": len(states) > limit,
            "removed": removed,
            "skipped": skipped,
        }

    def _create_cold_entry(
        self,
        *,
        cache_key: str,
        base_ref: str,
        base_commit: str,
        branch_name: str,
        dependencies: tuple[str, ...],
        prepare: Optional[WorktreePrepare],
        activate: Optional[WorktreePrepare],
        requested_path: Optional[Path],
        started: float,
        invalidation_reasons: tuple[str, ...],
    ) -> WorktreeLease:
        digest = hashlib.sha256(f"{cache_key}\0{base_commit}".encode("utf-8")).hexdigest()[:12]
        entry_id = f"{digest}-{uuid.uuid4().hex[:12]}"
        path = (
            requested_path
            or (
                self.worktree_root
                / python_identifier_worktree_basename("workspace", entry_id)
            )
        ).resolve()
        try:
            path.relative_to(self.worktree_root)
        except ValueError as exc:
            raise ValueError("pooled worktree path must be inside worktree_root") from exc
        if path.exists():
            raise FileExistsError(f"pooled worktree path already exists: {path}")
        lock_path = self.state_root / f"{entry_id}.lock"
        self._create_lock(lock_path)
        state: dict[str, Any] = {
            "schema": WORKTREE_POOL_SCHEMA,
            "lease_token": entry_id,
            "path": str(path),
            "repo_root": str(self.repo_root),
            "repo_common_dir": str(self.repo_common_dir),
            "cache_key": cache_key,
            "base_commit": base_commit,
            "dependency_paths": list(dependencies),
            "state": "initializing",
            "lease_pid": os.getpid(),
            "created_at_epoch": time.time(),
            "last_used_at_epoch": time.time(),
            "use_count": 1,
        }
        self._write_state(state)
        add_command = ["git", "worktree", "add"]
        if branch_name:
            add_command.extend(["-b", branch_name])
        else:
            add_command.append("--detach")
        add_command.extend([str(path), base_commit])
        add = self._run(add_command, cwd=self.repo_root)
        if not add.ok:
            self._discard_state(state)
            self._remove_lock(lock_path)
            raise RuntimeError(f"failed to create pooled worktree: {add.stderr or add.stdout}")
        try:
            if prepare is not None:
                prepare(path)
            if activate is not None:
                activate(path)
            clean, reason = self._repositories_clean(path, dependencies)
            if not clean:
                raise RuntimeError(f"prepared worktree is not reusable: {reason}")
            dependency_heads = self._dependency_heads(path, dependencies)
            if len(dependency_heads) != len(dependencies):
                raise RuntimeError(
                    "prepared worktree dependency is missing or has no resolvable HEAD"
                )
        except BaseException:
            self._discard_state(state)
            self._remove_lock(lock_path)
            raise
        elapsed = time.monotonic() - started
        state.update(
            {
                "state": "leased",
                "branch": branch_name,
                "dependency_heads": dependency_heads,
                "cold_setup_seconds": elapsed,
            }
        )
        self._write_state(state)
        self._metrics["cold_acquisitions"] += 1
        self._metrics["setup_seconds"] += elapsed
        return self._lease_from_state(
            state,
            base_ref=base_ref,
            branch_name=branch_name,
            reused=False,
            setup_seconds=elapsed,
            estimated_seconds_saved=0.0,
            invalidation_reasons=invalidation_reasons,
        )

    def _lease_from_state(
        self,
        state: Mapping[str, Any],
        *,
        base_ref: str,
        branch_name: str,
        reused: bool,
        setup_seconds: float,
        estimated_seconds_saved: float,
        invalidation_reasons: tuple[str, ...] = (),
    ) -> WorktreeLease:
        return WorktreeLease(
            pool=self,
            path=Path(str(state["path"])),
            cache_key=str(state["cache_key"]),
            base_ref=base_ref,
            base_commit=str(state["base_commit"]),
            branch_name=branch_name,
            dependency_paths=tuple(str(item) for item in state.get("dependency_paths") or ()),
            reused=reused,
            setup_seconds=setup_seconds,
            estimated_seconds_saved=estimated_seconds_saved,
            entry_id=str(state["lease_token"]),
            invalidation_reasons=invalidation_reasons,
        )

    def _state_matches(
        self,
        state: Mapping[str, Any],
        *,
        cache_key: str,
        base_commit: str,
        dependencies: tuple[str, ...],
    ) -> bool:
        return (
            state.get("schema") == WORKTREE_POOL_SCHEMA
            and state.get("state") in {"idle", "leased", "initializing"}
            and str(state.get("repo_root")) == str(self.repo_root)
            and str(state.get("repo_common_dir")) == str(self.repo_common_dir)
            and str(state.get("cache_key")) == cache_key
            and str(state.get("base_commit")) == base_commit
            and tuple(str(item) for item in state.get("dependency_paths") or ()) == dependencies
        )

    def _validate_idle_entry(self, state: Mapping[str, Any]) -> tuple[bool, str]:
        if state.get("state") != "idle":
            return False, "non_idle_entry_reserved"
        path = Path(str(state.get("path") or ""))
        if not path.is_dir():
            return False, "workspace_missing"
        if not path.name.isidentifier():
            # Old pool records used ``workspace-<digest>-<nonce>``.  Keep
            # those records readable so normal fenced cleanup can remove
            # them, but never return their Ruff-invalid checkout roots to a
            # new validation lease.
            return False, "worktree_basename_not_python_identifier"
        registered = self._run(("git", "worktree", "list", "--porcelain"), cwd=self.repo_root)
        registered_paths = {
            str(candidate.resolve())
            for candidate in git_worktree_paths_from_porcelain(registered.stdout)
        }
        if not registered.ok or str(path.resolve()) not in registered_paths:
            return False, "worktree_not_registered"
        if self._rev_parse(path, "HEAD") != str(state.get("base_commit") or ""):
            return False, "base_commit_mismatch"
        clean, reason = self._repositories_clean(
            path,
            tuple(str(item) for item in state.get("dependency_paths") or ()),
        )
        if not clean:
            return False, reason
        expected_heads = {
            str(key): str(value) for key, value in dict(state.get("dependency_heads") or {}).items()
        }
        if self._dependency_heads(path, tuple(expected_heads)) != expected_heads:
            return False, "dependency_head_mismatch"
        return True, "ready"

    def _repositories_clean(self, path: Path, dependencies: Sequence[str]) -> tuple[bool, str]:
        status = self._run(("git", "status", "--porcelain", "--untracked-files=all"), cwd=path)
        if not status.ok:
            return False, "worktree_status_failed"
        if status.stdout.strip():
            return False, "dirty_worktree"
        for relative in dependencies:
            target = path / relative
            if not target.is_dir() or not self._rev_parse(target, "HEAD"):
                return False, f"dependency_missing:{relative}"
            dependency_status = self._run(
                ("git", "status", "--porcelain", "--untracked-files=all"),
                cwd=target,
            )
            if not dependency_status.ok:
                return False, f"dependency_status_failed:{relative}"
            if dependency_status.stdout.strip():
                return False, f"dirty_dependency:{relative}"
        return True, "clean"

    def _dependency_heads(self, path: Path, dependencies: Sequence[str]) -> dict[str, str]:
        heads: dict[str, str] = {}
        for relative in dependencies:
            head = self._rev_parse(path / relative, "HEAD")
            if head:
                heads[str(relative)] = head
        return heads

    def _restore_prepared_state(self, state: Mapping[str, Any]) -> tuple[bool, str]:
        path = Path(str(state["path"]))
        dependency_heads = {
            str(key): str(value) for key, value in dict(state.get("dependency_heads") or {}).items()
        }
        # Restore children before the parent because the task branch may have
        # changed a gitlink.  -ffd removes task-local untracked context but keeps
        # ignored dependency caches such as node_modules.
        for relative, head in sorted(
            dependency_heads.items(), key=lambda item: item[0].count("/"), reverse=True
        ):
            target = path / relative
            for command in (
                ("git", "switch", "--detach", head),
                ("git", "reset", "--hard", head),
                ("git", "clean", "-ffd"),
            ):
                if not self._run(command, cwd=target).ok:
                    return False, f"dependency_restore_failed:{relative}"
        base_commit = str(state["base_commit"])
        for command in (
            ("git", "switch", "--detach", base_commit),
            ("git", "reset", "--hard", base_commit),
            ("git", "clean", "-ffd"),
        ):
            if not self._run(command, cwd=path).ok:
                return False, "base_restore_failed"
        clean, reason = self._repositories_clean(path, tuple(dependency_heads))
        if not clean:
            return False, reason
        if self._dependency_heads(path, tuple(dependency_heads)) != dependency_heads:
            return False, "dependency_head_mismatch_after_restore"
        return True, "restored"

    def _bind_task_branch(self, path: Path, *, branch_name: str, base_commit: str) -> CommandResult:
        if not branch_name:
            return self._run(("git", "switch", "--detach", base_commit), cwd=path)
        return self._run(("git", "switch", "-C", branch_name, base_commit), cwd=path)

    def _states(self) -> list[dict[str, Any]]:
        if not self.state_root.exists():
            return []
        states: list[dict[str, Any]] = []
        for path in sorted(self.state_root.glob("*.json")):
            state = read_json_object(path)
            # Bind the token to its sidecar filename.  Besides ignoring partial
            # or foreign JSON, this prevents a corrupted token from selecting
            # an arbitrary path when state and lock files are removed.
            if (
                state.get("schema") == WORKTREE_POOL_SCHEMA
                and str(state.get("lease_token") or "") == path.stem
            ):
                states.append(state)
        return states

    def _state_path(self, entry_id: str) -> Path:
        return self.state_root / f"{entry_id}.json"

    def _lock_path(self, state: Mapping[str, Any]) -> Path:
        return self.state_root / f"{state.get('lease_token', '')}.lock"

    def _read_state(self, entry_id: str) -> dict[str, Any]:
        return read_json_object(self._state_path(entry_id))

    def _write_state(self, state: Mapping[str, Any]) -> None:
        entry_id = str(state["lease_token"])
        path = self._state_path(entry_id)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        encoded = (
            json.dumps(dict(state), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("worktree pool state write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(path)

    def _create_lock(self, lock_path: Path) -> None:
        payload = json.dumps({"pid": os.getpid(), "created_at_epoch": time.time()}).encode("utf-8")
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    def _try_claim(
        self,
        state: Mapping[str, Any],
        *,
        require_idle: bool = False,
        allow_absent_registration_metadata_cleanup: bool = False,
    ) -> Optional[Path]:
        lock_path = self._lock_path(state)
        expected_dead_owner_pid = 0
        if allow_absent_registration_metadata_cleanup:
            try:
                expected_dead_owner_pid = int(state.get("lease_pid") or 0)
            except (TypeError, ValueError):
                expected_dead_owner_pid = 0
        # The advisory guard covers the complete inspect/remove/recreate
        # transaction.  Without it, two dead-owner reclaimers can both inspect
        # the stale record before either removes it; the loser can then unlink
        # the winner's newly-created live ownership record.  The same guard is
        # also the durable-quarantine publication guard, so a stale claimant
        # cannot replace exact custody after quarantine has been accepted.
        with guarded_worktree_pool_mutation(
            repo_root=self.repo_root,
            worktree_root=self.worktree_root,
            workspace_path=Path(str(state.get("path") or "")),
            expected_branch=str(state.get("branch") or ""),
            operation="worktree_pool_try_claim",
            allow_absent_registration_metadata_cleanup=(
                allow_absent_registration_metadata_cleanup
            ),
            expected_dead_owner_pid=expected_dead_owner_pid,
        ) as mutation_admission:
            if mutation_admission.get("allowed") is not True:
                self._record_rejection(
                    str(
                        mutation_admission.get("reason")
                        or "worktree_pool_mutation_denied"
                    )
                )
                return None
            if require_idle:
                current = self._read_state(
                    str(state.get("lease_token") or "")
                )
                if current != dict(state) or current.get("state") != "idle":
                    return None
            if state.get("state") != "idle":
                try:
                    state_owner_pid = int(state.get("lease_pid") or 0)
                except (TypeError, ValueError):
                    state_owner_pid = 0
                if state_owner_pid and pid_is_alive(state_owner_pid):
                    return None
            try:
                self._create_lock(lock_path)
                if require_idle:
                    current = self._read_state(
                        str(state.get("lease_token") or "")
                    )
                    if (
                        current != dict(state)
                        or current.get("state") != "idle"
                    ):
                        self._remove_lock(lock_path)
                        return None
                return lock_path
            except FileExistsError:
                lock = read_json_object(lock_path)
                try:
                    owner_pid = int(lock.get("pid") or 0)
                except (TypeError, ValueError):
                    owner_pid = 0
                if owner_pid and pid_is_alive(owner_pid):
                    return None
                # A dead claimant never authorizes immediate reuse.  Reclaiming
                # the lock merely permits the normal clean/stale validation
                # below.
                self._remove_lock(lock_path)
                try:
                    self._create_lock(lock_path)
                    if require_idle:
                        current = self._read_state(
                            str(state.get("lease_token") or "")
                        )
                        if (
                            current != dict(state)
                            or current.get("state") != "idle"
                        ):
                            self._remove_lock(lock_path)
                            return None
                    return lock_path
                except FileExistsError:
                    return None

    def _reclaim_dead_leases(self) -> list[dict[str, Any]]:
        """Discard missing-workspace dead leases after exclusively fencing them.

        Reclamation is intentionally independent of cache key and base commit:
        a crashed task commonly leaves a lease bound to a baseline that no
        future acquisition will request.  Unknown owners (missing/non-positive
        PIDs), live state owners, live lock claimants, and existing workspaces
        that may contain recoverable crash output remain untouched.
        """

        reclaimed: list[dict[str, Any]] = []
        for observed in self._states():
            if observed.get("state") not in {"leased", "initializing"}:
                continue
            try:
                observed_owner_pid = int(observed.get("lease_pid") or 0)
            except (TypeError, ValueError):
                observed_owner_pid = 0
            if (
                observed_owner_pid <= 0
                or pid_is_alive(observed_owner_pid)
                or not self._workspace_path_is_absent(observed)
            ):
                continue

            lock_path = self._try_claim(
                observed,
                allow_absent_registration_metadata_cleanup=True,
            )
            if lock_path is None:
                continue
            try:
                entry_id = str(observed.get("lease_token") or "")
                current = self._read_state(entry_id)
                if (
                    current.get("schema") != WORKTREE_POOL_SCHEMA
                    or str(current.get("lease_token") or "") != entry_id
                    or current.get("state") not in {"leased", "initializing"}
                ):
                    continue
                try:
                    current_owner_pid = int(current.get("lease_pid") or 0)
                except (TypeError, ValueError):
                    current_owner_pid = 0
                # Re-check under the claimed sidecar.  A live owner that won a
                # race before the claim must never lose its checkout.  Existing
                # crash output remains available to the supervisor rescue path.
                if (
                    current_owner_pid <= 0
                    or pid_is_alive(current_owner_pid)
                    or not self._workspace_path_is_absent(current)
                ):
                    continue
                discard = self._discard_state(current)
                if discard.get("removed") is True:
                    self._record_rejection("dead_lease_owner")
                    self._metrics["reclaimed_dead_leases"] += 1
                    self._metrics["discarded_entries"] += 1
                    reclaimed.append(
                        {
                            "entry_id": entry_id,
                            "state": str(current.get("state") or ""),
                            "lease_pid": current_owner_pid,
                            "path": str(current.get("path") or ""),
                            "discard": discard,
                        }
                    )
            finally:
                self._remove_lock(lock_path)
        return reclaimed

    @staticmethod
    def _workspace_path_is_absent(state: Mapping[str, Any]) -> bool:
        raw_path = str(state.get("path") or "").strip()
        if not raw_path:
            return False
        try:
            Path(raw_path).lstat()
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False
        return False

    @staticmethod
    def _remove_lock(lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _reject_and_discard(
        self, state: Mapping[str, Any], *, reason: str, lock_path: Path
    ) -> None:
        self._record_rejection(reason)
        self._discard_state(state)
        self._remove_lock(lock_path)

    def _record_rejection(self, reason: str) -> None:
        self._metrics["rejected_entries"] += 1
        reasons = self._metrics["rejection_reasons"]
        reasons[reason] = int(reasons.get(reason) or 0) + 1

    def _discard_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(state.get("path") or "").strip()
        path = Path(raw_path) if raw_path else self.worktree_root
        try:
            resolved = path.resolve()
            resolved.relative_to(self.worktree_root)
            safe_path = resolved != self.worktree_root
        except (OSError, RuntimeError, ValueError):
            safe_path = False
        if not safe_path:
            entry_id = str(state.get("lease_token") or "")
            if re.fullmatch(r"[A-Za-z0-9._-]+", entry_id):
                self._state_path(entry_id).unlink(missing_ok=True)
            return {
                "path": raw_path,
                "removed": False,
                "reason": "unsafe_or_invalid_worktree_path",
            }
        # This discard already holds the pool entry's authorized sidecar lock.
        # Git requires force twice for a worktree left locked by a crash during
        # initialization.  A single force used to leave the registration
        # behind while the code deleted its only pool-state owner.
        remove = self._run(
            ("git", "worktree", "remove", "--force", "--force", str(path)),
            cwd=self.repo_root,
        )
        registry = self._run(
            ("git", "worktree", "list", "--porcelain"),
            cwd=self.repo_root,
        )
        registered = True
        if registry.ok:
            expected = path.resolve(strict=False)
            registered = any(
                candidate.resolve(strict=False) == expected
                for candidate in git_worktree_paths_from_porcelain(registry.stdout)
            )
        # Delete residual bytes only after Git proves that this exact path is
        # no longer registered.  Git lock/contention or an unverifiable
        # registry preserves both checkout bytes and the authoritative sidecar.
        if registry.ok and not registered and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        removed = not path.exists() and registry.ok and not registered
        if removed:
            try:
                self._state_path(str(state.get("lease_token") or "")).unlink()
            except FileNotFoundError:
                pass
        return {
            "path": str(path),
            "removed": removed,
            "reason": (
                "discarded"
                if removed
                else (
                    "worktree_registry_unverifiable"
                    if not registry.ok
                    else "worktree_registration_persisted"
                )
            ),
            "git_remove": remove.compact(limit=2000),
            "git_registry": registry.compact(limit=2000),
            "registered": registered,
            "state_preserved": not removed,
        }

    def _prune_excess_idle(self, *, exclude_entry_id: str) -> None:
        idle = sorted(
            (state for state in self._states() if state.get("state") == "idle"),
            key=lambda state: float(state.get("last_used_at_epoch") or 0.0),
        )
        while len(idle) > self.max_entries:
            state = idle.pop(0)
            if str(state.get("lease_token")) == exclude_entry_id and idle:
                state = idle.pop(0)
            lock_path, _admission_reason = self._try_claim_authorized(
                state,
                authorize_reuse=self.reuse_authorizer,
            )
            if lock_path is None:
                continue
            self._discard_state(state)
            self._remove_lock(lock_path)

    def _rev_parse(self, cwd: Path, ref: str) -> str:
        result = self._run(("git", "rev-parse", "--verify", f"{ref}^{{commit}}"), cwd=cwd)
        return result.stdout.strip() if result.ok else ""

    def _branch_ref_presence(
        self,
        branch_ref: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``present``, ``absent``, or ``unverifiable`` for a branch ref."""

        command = (
            "git",
            "show-ref",
            "--verify",
            "--quiet",
            "--",
            branch_ref,
        )
        try:
            result = self._run(command, cwd=self.repo_root)
        except OSError as exc:
            return (
                "unverifiable",
                {
                    "error_type": type(exc).__name__,
                },
            )
        if result.returncode == 0:
            return "present", {"returncode": 0}
        if result.returncode == 1:
            return "absent", {"returncode": 1}
        return (
            "unverifiable",
            {
                "returncode": result.returncode,
                "error": result.stderr.strip()[:500],
            },
        )

    def _run(self, command: Sequence[str], *, cwd: Path) -> CommandResult:
        return _run_command_with_timeout(
            self.run_command_fn,
            command,
            cwd=cwd,
            timeout_seconds=self.command_timeout_seconds,
        )


@contextmanager
def managed_git_worktree(
    *,
    repo_root: Path,
    worktree_path: Path,
    metadata_rel: str,
    owner_rel: str,
    trace_context: Optional[Mapping[str, Any]] = None,
    run_command_fn: CommandRunner = run_command,
    owner_writer: Optional[WorktreeOwnerWriter] = None,
    add_timeout_seconds: int = 60,
    remove_timeout_seconds: int = 60,
    prune_on_exit: bool = True,
) -> Iterator[GitWorktreeSession]:
    """Create a detached Git worktree and always remove/prune it on exit."""

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    raw_trace: dict[str, Any] = dict(trace_context or {})
    raw_trace.update(
        {
            "worktree_path": str(worktree_path),
            "metadata_path": metadata_rel,
            "owner_path": owner_rel,
        }
    )
    session = GitWorktreeSession(
        repo_root=repo_root,
        path=worktree_path,
        metadata_rel=metadata_rel,
        owner_rel=owner_rel,
        raw_trace=raw_trace,
    )
    try:
        add_result = run_command_fn(
            ("git", "worktree", "add", "--detach", str(worktree_path), "HEAD"),
            cwd=repo_root,
            timeout_seconds=max(1, int(add_timeout_seconds)),
        )
        session.add_result = add_result
        raw_trace["worktree_add"] = add_result.compact(limit=12000)
        if add_result.ok and owner_writer is not None:
            owner_writer(worktree_path / owner_rel)
        yield session
    finally:
        remove_result = run_command_fn(
            ("git", "worktree", "remove", "--force", str(worktree_path)),
            cwd=repo_root,
            timeout_seconds=max(1, int(remove_timeout_seconds)),
        )
        raw_trace["worktree_remove"] = remove_result.compact(limit=12000)
        if not remove_result.ok and worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        if prune_on_exit:
            prune_result = run_command_fn(
                ("git", "worktree", "prune", "--expire", "now"),
                cwd=repo_root,
                timeout_seconds=max(1, int(remove_timeout_seconds)),
            )
            raw_trace["worktree_prune_after_remove"] = prune_result.compact(limit=12000)


def cleanup_stale_daemon_worktrees(
    *,
    repo_root: Path,
    worktree_root: Path,
    stale_after_seconds: int,
    owner_filename: str,
    patterns: Sequence[str] = ("cycle_*", "repair_*"),
    run_command_fn: CommandRunner = run_command,
    owner_alive: Optional[OwnerAlivePredicate] = None,
    now_epoch: Optional[float] = None,
) -> dict[str, Any]:
    """Remove daemon-created worktrees whose owner is gone and whose age is stale."""

    stale_after = max(1, int(stale_after_seconds))
    result: dict[str, Any] = {
        "valid": True,
        "worktree_root": str(worktree_root),
        "stale_after_seconds": stale_after,
        "patterns": list(patterns),
        "removed": [],
        "skipped": [],
        "errors": [],
    }
    prune_before = run_command_fn(
        ("git", "worktree", "prune", "--expire", "now"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["prune_before"] = prune_before.compact(limit=12000)
    if not worktree_root.exists():
        return result

    root_resolved = worktree_root.resolve()
    list_result = run_command_fn(
        ("git", "worktree", "list", "--porcelain"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["worktree_list"] = list_result.compact(limit=12000)
    registered_paths = {str(path) for path in git_worktree_paths_from_porcelain(list_result.stdout)}
    now = time.time() if now_epoch is None else float(now_epoch)

    candidates: list[Path] = []
    seen_candidates: set[Path] = set()
    for pattern in patterns:
        for candidate in worktree_root.glob(pattern):
            resolved_candidate = candidate.resolve()
            if resolved_candidate in seen_candidates:
                continue
            seen_candidates.add(resolved_candidate)
            candidates.append(candidate)

    for candidate in sorted(candidates):
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root_resolved):
                result["skipped"].append(
                    {"path": str(candidate), "reason": "outside_worktree_root"}
                )
                continue
            if not candidate.is_dir():
                result["skipped"].append({"path": str(candidate), "reason": "not_directory"})
                continue
            owner = read_json_object(candidate / owner_filename)
            owner_pid = owner_pid_from_worktree(candidate, owner)
            owner_is_alive = bool(
                owner_pid
                and owner_alive is not None
                and owner_alive(owner_pid, repo_root, candidate)
            )
            try:
                created_at = float(owner.get("created_at_epoch") or candidate.stat().st_mtime)
            except (OSError, TypeError, ValueError):
                created_at = candidate.stat().st_mtime
            age_seconds = max(0.0, now - created_at)
            if owner_is_alive:
                result["skipped"].append(
                    {
                        "path": str(candidate),
                        "reason": "owner_pid_alive",
                        "owner_pid": owner_pid,
                        "age_seconds": round(age_seconds, 3),
                    }
                )
                continue
            if age_seconds < stale_after:
                result["skipped"].append(
                    {
                        "path": str(candidate),
                        "reason": "not_stale_yet",
                        "owner_pid": owner_pid,
                        "age_seconds": round(age_seconds, 3),
                    }
                )
                continue

            registered = str(resolved) in registered_paths
            if registered:
                remove_result = run_command_fn(
                    ("git", "worktree", "remove", "--force", str(resolved)),
                    cwd=repo_root,
                    timeout_seconds=60,
                )
                if not remove_result.ok and candidate.exists():
                    shutil.rmtree(candidate, ignore_errors=True)
            else:
                shutil.rmtree(candidate, ignore_errors=True)
                remove_result = CommandResult(
                    ("shutil.rmtree", str(resolved)),
                    0 if not candidate.exists() else 1,
                    "",
                    "" if not candidate.exists() else "directory still exists after rmtree",
                )
            record = {
                "path": str(candidate),
                "registered": registered,
                "owner_pid": owner_pid,
                "age_seconds": round(age_seconds, 3),
                "remove": remove_result.compact(limit=12000),
            }
            if remove_result.ok:
                result["removed"].append(record)
            else:
                result["valid"] = False
                result["errors"].append(record)
        except Exception as exc:
            result["valid"] = False
            result["errors"].append(
                {"path": str(candidate), "exception": f"{type(exc).__name__}: {exc}"}
            )

    prune_after = run_command_fn(
        ("git", "worktree", "prune", "--expire", "now"),
        cwd=repo_root,
        timeout_seconds=60,
    )
    result["prune_after"] = prune_after.compact(limit=12000)
    if not prune_after.ok:
        result["valid"] = False
    return result
