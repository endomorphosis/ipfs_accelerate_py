"""Disposable Git worktree mutation executor and admission pipeline (AAE-041).

Interface surface:

* ``IsolatedMutationWorktreeExecutor@1`` — sole mutation-worktree lifecycle
  owner: creates and destroys disposable owned worktrees, applies bounded
  edits inside them, and runs the AAE-024 ``admit_mutation`` pipeline.

Normative properties:

* AAE-024 only validates a *caller-supplied* worktree; this module is the
  only authority that creates and destroys mutation worktrees.
* Mutations never target production trees or branches; checkouts are
  detached under an owned worktree root.
* Writes cannot escape owned roots, touch credentials/network, or alter
  undeclared authority surfaces (verifier/policy/key/oracle).
* Cleanup is fenced through ``WorktreeLifecycleStore`` and recoverable from
  durable attempt journals after prepare/apply/cleanup interruptions.

Cold import is side-effect free: no Git, ledger, process, network, or
filesystem operations run at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.admission import (
    ADMIT_MUTATION_INTERFACE,
    AdmissionDisposition,
    AdmissionError,
    AdmissionReasonCode,
    MutationAdmissionResult,
    admit_mutation,
    blocked_authority_path,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    CleanupDecision,
    CleanupDisposition,
    DuplicateAttemptError,
    FenceMismatchError,
    OwnershipError,
    WorkspaceLifecycleRecord,
    WorkspaceLifecycleState,
    WorktreeLifecycleError,
    WorktreeLifecycleStore,
    current_process_birth,
    normalize_workspace_path,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    DetectionAssuranceManifest,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    MutationCandidate,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE: Final[str] = (
    "IsolatedMutationWorktreeExecutor@1"
)
MUTATION_WORKTREE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worktree@1"
)
MUTATION_WORKTREE_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-worktree-attempt@1"
)
MUTATION_EXECUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-execution@1"
)
MUTATION_APPLY_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-mutation-apply@1"
)

ADAPTER_ID: Final[str] = "aae-isolated-mutation-worktree-executor"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
AAE_ISOLATED_EXECUTOR_EVIDENCE: Final[str] = "aae/isolated-executor@1"

MAX_PATCH_BYTES: Final[int] = 2_000_000
MAX_FILE_BYTES: Final[int] = 1_000_000
MAX_FILES: Final[int] = 256
MAX_PATH_BYTES: Final[int] = 512
MAX_DIAGNOSTIC: Final[int] = 1_024
GIT_TIMEOUT_SECONDS: Final[int] = 60

_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")
_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")

# Paths never writable by a mutant (always forbidden).
_DEFAULT_FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    ".git/",
    ".git",
    ".agent_supervisor/",
    "data/agent_supervisor/",
    "__pycache__/",
    ".env",
    ".ssh/",
    ".aws/",
    ".gnupg/",
    "secrets/",
    "credentials/",
)

# Production-control surfaces that mutants must not edit unless the caller
# explicitly enables an authority-fixture campaign.
_DEFAULT_PROTECTED_PREFIXES: Final[tuple[str, ...]] = (
    "docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md",
    "docs/architecture/adversarial_assurance_engine.objectives.md",
    "docs/architecture/adversarial_assurance_engine.todo.md",
    "config/adversarial_assurance_engine_scheduler.json",
    "config/adversarial_assurance_prerequisites.json",
    "scripts/validate_adversarial_assurance_engine_board.py",
    "ipfs_accelerate_py/agent_supervisor/verification/",
    "ipfs_accelerate_py/agent_supervisor/proof/",
    "ipfs_accelerate_py/agent_supervisor/validation/",
)


# ---------------------------------------------------------------------------
# Errors / phases
# ---------------------------------------------------------------------------


class MutationWorktreeError(ValueError):
    """Fail-closed error for mutation worktree lifecycle or apply."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mutation_worktree_error",
        path: str = "",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "mutation_worktree_error")
        self.path = str(path or "")


class MutationWorktreeFenceError(MutationWorktreeError):
    """Ownership or fence token does not authorize the requested mutation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "fence_mismatch",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class MutationWorktreePhase(str, Enum):
    """Durable phases for one isolated mutation worktree attempt."""

    PREPARING = "preparing"
    READY = "ready"
    APPLYING = "applying"
    APPLIED = "applied"
    ADMITTING = "admitting"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    CLEANING = "cleaning"
    TERMINAL = "terminal"

    @property
    def is_terminal(self) -> bool:
        return self is MutationWorktreePhase.TERMINAL

    @property
    def is_mutable(self) -> bool:
        return self in {
            MutationWorktreePhase.READY,
            MutationWorktreePhase.APPLYING,
            MutationWorktreePhase.APPLIED,
            MutationWorktreePhase.ADMITTING,
            MutationWorktreePhase.ADMITTED,
            MutationWorktreePhase.REJECTED,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _scrubbed_git_env() -> dict[str, str]:
    """Minimal local Git environment: no credentials, no network prompts."""

    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        # Explicitly deny credential helpers / remote interaction.
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "protocol.file.allow",
        "GIT_CONFIG_VALUE_1": "always",
        "GIT_HTTP_LOW_SPEED_LIMIT": "0",
        "GIT_HTTP_LOW_SPEED_TIME": "0",
        "GIT_ASKPASS": "",
        "SSH_ASKPASS": "",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
    }


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    stdin: str | bytes | None = None,
    timeout_seconds: int = GIT_TIMEOUT_SECONDS,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a fixed local Git argv under a scrubbed minimal environment."""

    # Reject network-capable subcommands fail-closed.
    if args and str(args[0]).lower() in {
        "fetch",
        "pull",
        "push",
        "clone",
        "ls-remote",
        "remote",
        "submodule",
    }:
        raise MutationWorktreeError(
            f"network-capable git subcommand rejected: {args[0]}",
            reason_code="network_denied",
        )

    env = _scrubbed_git_env()
    text_mode = not isinstance(stdin, (bytes, bytearray))
    try:
        completed = subprocess.run(
            ["git", *list(args)],
            cwd=str(cwd),
            input=stdin,
            text=text_mode,
            encoding="utf-8" if text_mode else None,
            errors="strict" if text_mode else None,
            capture_output=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise MutationWorktreeError(
            f"git {' '.join(str(a) for a in args[:3])} timed out",
            reason_code="git_timeout",
        ) from exc
    except OSError as exc:
        raise MutationWorktreeError(
            f"git invocation failed: {exc}",
            reason_code="git_unavailable",
        ) from exc

    if text_mode:
        return completed  # type: ignore[return-value]

    # Normalize bytes results to str for uniform callers.
    return subprocess.CompletedProcess(
        args=completed.args,
        returncode=completed.returncode,
        stdout=(completed.stdout or b"").decode("utf-8", errors="replace"),
        stderr=(completed.stderr or b"").decode("utf-8", errors="replace"),
    )


def _rev_parse(repo: Path, rev: str) -> str:
    result = _run_git(["rev-parse", "--verify", rev], cwd=repo)
    if result.returncode != 0:
        raise MutationWorktreeError(
            f"cannot resolve revision {rev!r}",
            reason_code="stale_base",
        )
    value = (result.stdout or "").strip()
    if not _OID_RE.fullmatch(value):
        raise MutationWorktreeError(
            f"malformed object id for {rev!r}",
            reason_code="stale_base",
        )
    return value


def _tree_for_commit(repo: Path, commit: str) -> str:
    return _rev_parse(repo, f"{commit}^{{tree}}")


def _write_tree(repo: Path) -> str:
    result = _run_git(["write-tree"], cwd=repo)
    if result.returncode != 0:
        raise MutationWorktreeError(
            _clip(result.stderr or "write-tree failed"),
            reason_code="tree_capture_failed",
        )
    value = (result.stdout or "").strip()
    if not _OID_RE.fullmatch(value):
        raise MutationWorktreeError(
            "write-tree returned a malformed oid",
            reason_code="tree_capture_failed",
        )
    return value


def _head_commit(repo: Path) -> str:
    return _rev_parse(repo, "HEAD")


def _is_linked_worktree(path: Path) -> bool:
    """Return True when ``path`` is a linked Git worktree (``.git`` file)."""

    git_entry = path / ".git"
    if git_entry.is_file():
        try:
            content = git_entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return content.startswith("gitdir:")
    return False


def _is_production_repo_root(path: Path) -> bool:
    """Return True when path looks like a primary (non-linked) Git checkout."""

    git_entry = path / ".git"
    return git_entry.is_dir()


def _normalize_repo_path(value: str, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MutationWorktreeError(
            f"{name} is not a repository path",
            reason_code="unsafe_path",
            path=str(value),
        )
    raw = value.replace("\\", "/").strip()
    if raw.startswith("./"):
        raw = raw[2:]
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or raw != pure.as_posix()
        or raw in {".", ""}
        or ".." in pure.parts
        or any(part in {"", ".", ".git"} for part in pure.parts)
        or len(raw.encode("utf-8")) > MAX_PATH_BYTES
    ):
        raise MutationWorktreeError(
            f"{name} escapes the repository: {value!r}",
            reason_code="unsafe_path",
            path=raw,
        )
    return pure.as_posix()


def _path_matches_prefix(path: str, pattern: str) -> bool:
    if not pattern:
        return False
    if pattern.endswith("/**"):
        root = pattern[:-3].rstrip("/")
        return path == root or path.startswith(root + "/")
    if pattern.endswith("/"):
        return path == pattern[:-1] or path.startswith(pattern)
    if "*" in pattern or "?" in pattern or "[" in pattern:
        import fnmatch

        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            path, pattern.rstrip("/") + "/**"
        )
    return path == pattern or path.startswith(pattern.rstrip("/") + "/")


def _is_under_any(path: str, patterns: Sequence[str]) -> bool:
    return any(_path_matches_prefix(path, pattern) for pattern in patterns)


def _safe_branch_segment(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", str(value or "").strip()).strip("-")
    return cleaned or "task"


def _content_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationWriteScope:
    """Immutable write-scope for one mutation apply.

    ``allowed_paths`` is the task allowlist. ``effect_paths``, when non-empty,
    further narrows the write set to the declared effect surface.
    ``task_owned_paths`` is the authority ceiling: the scope may narrow it but
    never widen beyond it.
    """

    allowed_paths: tuple[str, ...]
    effect_paths: tuple[str, ...] = ()
    task_owned_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = _DEFAULT_FORBIDDEN_PREFIXES
    protected_paths: tuple[str, ...] = _DEFAULT_PROTECTED_PREFIXES
    max_files: int = MAX_FILES
    max_bytes: int = MAX_FILE_BYTES
    allow_authority_fixture: bool = False

    def __post_init__(self) -> None:
        def _norm_pattern(item: object, *, name: str) -> str:
            text = str(item).replace("\\", "/").strip()
            if not text:
                raise MutationWorktreeError(
                    f"{name} entry must not be empty",
                    reason_code="invalid_scope",
                )
            if text.endswith(("/", "*")) or "*" in text or "?" in text:
                return text
            return _normalize_repo_path(text, name=name)

        allowed = tuple(
            dict.fromkeys(
                _norm_pattern(item, name="allowed_paths") for item in self.allowed_paths
            )
        )
        if not allowed:
            raise MutationWorktreeError(
                "allowed_paths must not be empty",
                reason_code="invalid_scope",
            )
        effect = tuple(
            dict.fromkeys(
                _normalize_repo_path(str(item), name="effect_paths")
                for item in self.effect_paths
            )
        )
        owned = tuple(
            dict.fromkeys(
                _norm_pattern(item, name="task_owned_paths")
                for item in (self.task_owned_paths or allowed)
            )
        )
        forbidden = tuple(
            dict.fromkeys(str(item).replace("\\", "/").strip() for item in self.forbidden_paths)
        )
        protected = tuple(
            dict.fromkeys(
                str(item).replace("\\", "/").strip() for item in self.protected_paths
            )
        )
        if type(self.allow_authority_fixture) is not bool:
            raise MutationWorktreeError(
                "allow_authority_fixture must be boolean",
                reason_code="invalid_scope",
            )
        if type(self.max_files) is not int or self.max_files < 1:
            raise MutationWorktreeError(
                "max_files must be a positive integer",
                reason_code="invalid_scope",
            )
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise MutationWorktreeError(
                "max_bytes must be a positive integer",
                reason_code="invalid_scope",
            )
        object.__setattr__(self, "allowed_paths", allowed)
        object.__setattr__(self, "effect_paths", effect)
        object.__setattr__(self, "task_owned_paths", owned)
        object.__setattr__(self, "forbidden_paths", forbidden)
        object.__setattr__(self, "protected_paths", protected)

    def admits(self, path: str) -> tuple[bool, str]:
        """Return ``(ok, reason_code)`` for one normalized repository path."""

        try:
            normalized = _normalize_repo_path(path)
        except MutationWorktreeError:
            return False, "unsafe_path"
        if _is_under_any(normalized, self.forbidden_paths):
            return False, "forbidden_path"
        if _is_under_any(normalized, self.protected_paths):
            return False, "protected_path"
        if not self.allow_authority_fixture and blocked_authority_path(normalized):
            return False, "authority_path_blocked"
        if not _is_under_any(normalized, self.task_owned_paths):
            return False, "outside_task_owned"
        if not _is_under_any(normalized, self.allowed_paths):
            return False, "outside_allowlist"
        if self.effect_paths and not _is_under_any(normalized, self.effect_paths):
            return False, "outside_effect_scope"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_paths": list(self.allowed_paths),
            "effect_paths": list(self.effect_paths),
            "task_owned_paths": list(self.task_owned_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "protected_paths": list(self.protected_paths),
            "max_files": int(self.max_files),
            "max_bytes": int(self.max_bytes),
            "allow_authority_fixture": bool(self.allow_authority_fixture),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationWriteScope":
        if not isinstance(data, Mapping):
            raise MutationWorktreeError(
                "MutationWriteScope payload must be an object",
                reason_code="invalid_scope",
            )
        return cls(
            allowed_paths=tuple(data.get("allowed_paths") or ()),
            effect_paths=tuple(data.get("effect_paths") or ()),
            task_owned_paths=tuple(data.get("task_owned_paths") or ()),
            forbidden_paths=tuple(
                data["forbidden_paths"]
                if "forbidden_paths" in data
                else _DEFAULT_FORBIDDEN_PREFIXES
            ),
            protected_paths=tuple(
                data["protected_paths"]
                if "protected_paths" in data
                else _DEFAULT_PROTECTED_PREFIXES
            ),
            max_files=int(data.get("max_files") or MAX_FILES),
            max_bytes=int(data.get("max_bytes") or MAX_FILE_BYTES),
            allow_authority_fixture=bool(data.get("allow_authority_fixture", False)),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: MutationCandidate | Mapping[str, Any],
        *,
        allow_authority_fixture: bool = False,
        extra_allowed: Sequence[str] = (),
    ) -> "MutationWriteScope":
        """Derive a write scope from a candidate's declared ``scope_paths``."""

        if isinstance(candidate, Mapping):
            paths = tuple(candidate.get("scope_paths") or ())
        else:
            paths = tuple(candidate.scope_paths or ())
        if not paths and not extra_allowed:
            raise MutationWorktreeError(
                "candidate scope_paths is empty; cannot derive write scope",
                reason_code="invalid_scope",
            )
        allowed = tuple(dict.fromkeys([*paths, *extra_allowed]))
        return cls(
            allowed_paths=allowed,
            effect_paths=tuple(paths),
            task_owned_paths=allowed,
            allow_authority_fixture=allow_authority_fixture,
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationApplyResult:
    """Outcome of applying bounded file replacements inside a disposable worktree."""

    applied: bool
    reason_codes: tuple[str, ...]
    paths: tuple[str, ...]
    pre_tree: str
    post_tree: str
    base_commit: str
    worktree_path: str
    path_digests: Mapping[str, str] = field(default_factory=dict)
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_APPLY_RESULT_SCHEMA,
            "applied": bool(self.applied),
            "reason_codes": list(self.reason_codes),
            "paths": list(self.paths),
            "pre_tree": self.pre_tree,
            "post_tree": self.post_tree,
            "base_commit": self.base_commit,
            "worktree_path": self.worktree_path,
            "path_digests": dict(self.path_digests),
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class MutationExecutionResult:
    """Sealed outcome of create → apply → admit (optional cleanup)."""

    executed: bool
    disposition: str
    reason_codes: tuple[str, ...]
    candidate_id: str
    candidate_cid: str
    worktree_path: str
    lease_id: str
    fence: int
    base_commit: str
    base_tree: str
    pre_tree: str
    post_tree: str
    root_head: str
    apply: Mapping[str, Any] | None
    admission: Mapping[str, Any] | None
    cleaned: bool
    identity_cid: str
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_EXECUTION_RESULT_SCHEMA,
            "interface": ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE,
            "executed": bool(self.executed),
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "worktree_path": self.worktree_path,
            "lease_id": self.lease_id,
            "fence": int(self.fence),
            "base_commit": self.base_commit,
            "base_tree": self.base_tree,
            "pre_tree": self.pre_tree,
            "post_tree": self.post_tree,
            "root_head": self.root_head,
            "apply": None if self.apply is None else dict(self.apply),
            "admission": None if self.admission is None else dict(self.admission),
            "cleaned": bool(self.cleaned),
            "identity_cid": self.identity_cid,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
            "evidence": AAE_ISOLATED_EXECUTOR_EVIDENCE,
        }

    @property
    def admitted(self) -> bool:
        return self.disposition == AdmissionDisposition.ADMITTED.value


def _stable_execution_identity(
    *,
    candidate_cid: str,
    disposition: str,
    reason_codes: Sequence[str],
    base_commit: str,
    post_tree: str,
    apply_paths: Sequence[str],
    admission_cid: str | None,
) -> str:
    return cid_for_structured(
        {
            "schema": MUTATION_EXECUTION_RESULT_SCHEMA,
            "kind": "aae-mutation-execution-identity",
            "candidate_cid": candidate_cid,
            "disposition": disposition,
            "reason_codes": list(reason_codes),
            "base_commit": base_commit,
            "post_tree": post_tree,
            "apply_paths": list(apply_paths),
            "admission_cid": admission_cid,
        }
    )


# ---------------------------------------------------------------------------
# Apply helpers
# ---------------------------------------------------------------------------


def _normalize_replacements(
    file_replacements: Mapping[str, str | bytes],
    *,
    scope: MutationWriteScope,
) -> list[tuple[str, bytes]]:
    if not isinstance(file_replacements, Mapping) or not file_replacements:
        raise MutationWorktreeError(
            "file_replacements must be a non-empty mapping",
            reason_code="no_edits",
        )
    if len(file_replacements) > scope.max_files:
        raise MutationWorktreeError(
            f"too many file replacements ({len(file_replacements)} > {scope.max_files})",
            reason_code="too_many_paths",
        )

    ordered: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw_path, raw_content in file_replacements.items():
        path = _normalize_repo_path(str(raw_path), name="file_replacements path")
        if path in seen:
            raise MutationWorktreeError(
                f"duplicate replacement path: {path}",
                reason_code="duplicate_path",
                path=path,
            )
        seen.add(path)
        ok, reason = scope.admits(path)
        if not ok:
            raise MutationWorktreeError(
                f"path not admitted by write scope: {path} ({reason})",
                reason_code=reason,
                path=path,
            )
        if isinstance(raw_content, bytes):
            content = raw_content
        elif isinstance(raw_content, str):
            content = raw_content.encode("utf-8")
        else:
            raise MutationWorktreeError(
                f"replacement content for {path} must be str or bytes",
                reason_code="invalid_content",
                path=path,
            )
        if len(content) > scope.max_bytes:
            raise MutationWorktreeError(
                f"file too large: {path} ({len(content)} bytes)",
                reason_code="file_too_large",
                path=path,
            )
        total_bytes += len(content)
        if total_bytes > MAX_PATCH_BYTES:
            raise MutationWorktreeError(
                "aggregate replacement payload exceeds patch budget",
                reason_code="patch_too_large",
            )
        ordered.append((path, content))
    ordered.sort(key=lambda item: item[0])
    return ordered


def apply_file_replacements(
    file_replacements: Mapping[str, str | bytes],
    *,
    worktree_root: Path | str,
    scope: MutationWriteScope | Mapping[str, Any],
    expected_base_commit: str | None = None,
    expected_base_tree: str | None = None,
    require_linked_worktree: bool = True,
) -> MutationApplyResult:
    """Apply path→bytes replacements inside a disposable linked worktree only.

    Never mutates a production repository root (``.git`` directory present and
    not a linked worktree). Failures leave the disposable tree hard-reset to
    the bound base when possible.
    """

    root = Path(worktree_root)
    write_scope = (
        scope
        if isinstance(scope, MutationWriteScope)
        else MutationWriteScope.from_dict(scope)
    )
    paths_attempted: list[str] = []

    def _fail(
        *codes: str,
        diagnostic: str = "",
        pre_tree: str = "",
        base: str = "",
    ) -> MutationApplyResult:
        return MutationApplyResult(
            applied=False,
            reason_codes=tuple(codes),
            paths=tuple(paths_attempted),
            pre_tree=pre_tree,
            post_tree=pre_tree,
            base_commit=base,
            worktree_path=normalize_workspace_path(root),
            diagnostic=_clip(diagnostic),
        )

    if not root.is_dir():
        return _fail("worktree_missing", diagnostic="worktree path missing")

    if require_linked_worktree:
        if _is_production_repo_root(root):
            return _fail(
                "production_root",
                diagnostic="refusing to mutate a production repository root",
            )
        if not _is_linked_worktree(root):
            return _fail(
                "not_disposable_worktree",
                diagnostic="path is not a linked disposable git worktree",
            )

    try:
        replacements = _normalize_replacements(file_replacements, scope=write_scope)
    except MutationWorktreeError as exc:
        return _fail(exc.reason_code, diagnostic=str(exc))

    paths_attempted = [path for path, _ in replacements]

    try:
        head = _head_commit(root)
        if expected_base_commit and head != expected_base_commit:
            return _fail(
                "stale_base",
                diagnostic="worktree HEAD does not match expected base commit",
                base=expected_base_commit,
            )
        base = expected_base_commit or head
        pre_tree = _tree_for_commit(root, head)
        if expected_base_tree and pre_tree != expected_base_tree:
            return _fail(
                "stale_base",
                diagnostic="worktree tree does not match expected base tree",
                pre_tree=pre_tree,
                base=base,
            )
        status = _run_git(["status", "--porcelain"], cwd=root)
        if status.returncode == 0 and (status.stdout or "").strip():
            return _fail(
                "dirty_worktree",
                diagnostic="worktree is dirty before apply",
                pre_tree=pre_tree,
                base=base,
            )
    except MutationWorktreeError as exc:
        return _fail(exc.reason_code, diagnostic=str(exc))

    digests: dict[str, str] = {}
    try:
        for rel_path, content in replacements:
            target = root / rel_path
            # Escape guard: resolved path must remain under worktree root.
            try:
                resolved = target.resolve(strict=False)
                root_resolved = root.resolve(strict=True)
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                raise MutationWorktreeError(
                    f"path escapes owned worktree root: {rel_path}",
                    reason_code="path_escape",
                    path=rel_path,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            digests[rel_path] = _content_digest(content)

        # Stage so write-tree reflects applied bytes (no commit on production).
        _run_git(["add", "-A", "--", *[p for p, _ in replacements]], cwd=root)
        post_tree = _write_tree(root)
    except MutationWorktreeError as exc:
        if base:
            _run_git(["reset", "--hard", base], cwd=root)
            _run_git(["clean", "-fdx"], cwd=root)
        return _fail(
            exc.reason_code,
            diagnostic=str(exc),
            pre_tree=pre_tree,
            base=base,
        )
    except OSError as exc:
        if base:
            _run_git(["reset", "--hard", base], cwd=root)
            _run_git(["clean", "-fdx"], cwd=root)
        return _fail(
            "apply_failed",
            diagnostic=str(exc),
            pre_tree=pre_tree,
            base=base,
        )

    return MutationApplyResult(
        applied=True,
        reason_codes=(),
        paths=tuple(paths_attempted),
        pre_tree=pre_tree,
        post_tree=post_tree,
        base_commit=base,
        worktree_path=normalize_workspace_path(root),
        path_digests=MappingProxyType(digests),
        diagnostic="",
    )


# ---------------------------------------------------------------------------
# Isolated worktree handle
# ---------------------------------------------------------------------------


@dataclass
class IsolatedMutationWorktree:
    """Fenced disposable mutation worktree bound to one attempt identity.

    Ownership mutations go through ``WorktreeLifecycleStore``. Publication and
    cleanup require the current lease/fence; peer or stale owners are rejected.
    """

    repo_root: Path
    worktree_path: Path
    worktree_parent: Path
    base_commit: str
    base_tree: str
    task_id: str
    attempt: int
    lane_id: str
    lease_id: str
    fence: int
    lifecycle_store: WorktreeLifecycleStore
    phase: MutationWorktreePhase = MutationWorktreePhase.PREPARING
    branch: str = ""
    merge_target: str = "HEAD"
    canonical_task_cid: str = ""
    root_head_at_create: str = ""
    journal_path: Path | None = None
    retain_on_success: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.worktree_path = Path(self.worktree_path)
        self.worktree_parent = Path(self.worktree_parent).resolve()
        if not self.branch:
            safe_task = _safe_branch_segment(self.task_id)
            self.branch = f"aae-mutant/{safe_task}-a{int(self.attempt)}"
        if self.journal_path is None:
            store_dir = self.lifecycle_store.store_dir
            assert store_dir is not None
            digest = hashlib.sha256(
                normalize_workspace_path(self.worktree_path).encode("utf-8")
            ).hexdigest()[:16]
            self.journal_path = Path(store_dir) / f"aae-attempt-{digest}.json"

    # ---------------------------------------------------------------- journal

    def _journal_payload(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": MUTATION_WORKTREE_ATTEMPT_SCHEMA,
            "interface": ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE,
            "board_namespace": BOARD_NAMESPACE,
            "adapter_id": ADAPTER_ID,
            "task_id": self.task_id,
            "canonical_task_cid": self.canonical_task_cid,
            "attempt": int(self.attempt),
            "lane_id": self.lane_id,
            "lease_id": self.lease_id,
            "fence": int(self.fence),
            "phase": self.phase.value,
            "repo_root": str(self.repo_root),
            "worktree_path": str(self.worktree_path),
            "worktree_parent": str(self.worktree_parent),
            "base_commit": self.base_commit,
            "base_tree": self.base_tree,
            "branch": self.branch,
            "merge_target": self.merge_target,
            "root_head_at_create": self.root_head_at_create,
            "updated_at": time.time(),
        }
        payload.update(extra)
        return payload

    def _write_journal(self, **extra: Any) -> None:
        if self.journal_path is None:
            return
        _atomic_write_json(self.journal_path, self._journal_payload(**extra))

    def _sync_fence_from_record(self, record: WorkspaceLifecycleRecord) -> None:
        self.lease_id = record.lease_id
        self.fence = int(record.fence)

    # --------------------------------------------------------------- ownership

    def _require_owner(self, *, lease_id: str, fence: int) -> None:
        if str(lease_id) != str(self.lease_id) or int(fence) != int(self.fence):
            raise MutationWorktreeFenceError(
                "caller lease/fence does not match live worktree owner",
                reason_code="stale_owner",
            )
        record = self.lifecycle_store.load_workspace(self.worktree_path)
        if record is None:
            raise MutationWorktreeFenceError(
                "lifecycle record missing for worktree",
                reason_code="missing_record",
            )
        if str(record.lease_id) != str(lease_id) or int(record.fence) != int(fence):
            raise MutationWorktreeFenceError(
                "lifecycle fence advanced; caller is stale",
                reason_code="stale_owner",
            )

    def assert_base_current(self) -> None:
        """Fail closed when the disposable worktree left the bound base."""

        if not self.worktree_path.is_dir():
            raise MutationWorktreeError(
                "worktree path missing",
                reason_code="worktree_missing",
            )
        head = _head_commit(self.worktree_path)
        tree = _tree_for_commit(self.worktree_path, head)
        # After apply the index/tree may differ; HEAD commit should still match.
        if head != self.base_commit:
            raise MutationWorktreeError(
                "worktree base commit is stale",
                reason_code="stale_base",
            )
        if self.phase in {
            MutationWorktreePhase.READY,
            MutationWorktreePhase.PREPARING,
        } and tree != self.base_tree:
            raise MutationWorktreeError(
                "worktree base tree is stale before apply",
                reason_code="stale_base",
            )

    def assert_root_unmutated(self) -> None:
        """Prove the caller's repository root HEAD is unchanged."""

        if not self.root_head_at_create:
            return
        head = _head_commit(self.repo_root)
        if head != self.root_head_at_create:
            raise MutationWorktreeError(
                "repository root HEAD mutated unexpectedly",
                reason_code="root_mutated",
            )

    def assert_under_owned_root(self) -> None:
        """Prove the disposable worktree path stays under the owned parent."""

        try:
            self.worktree_path.resolve().relative_to(self.worktree_parent.resolve())
        except (OSError, ValueError) as exc:
            raise MutationWorktreeError(
                "worktree path escapes owned worktree parent root",
                reason_code="path_escape",
            ) from exc
        if normalize_workspace_path(self.worktree_path) == normalize_workspace_path(
            self.repo_root
        ):
            raise MutationWorktreeError(
                "worktree path must not equal production repo root",
                reason_code="production_root",
            )

    # ---------------------------------------------------------------- lifecycle

    def mark_ready(self) -> None:
        with self._lock:
            record = self.lifecycle_store.mark_active(
                self.worktree_path,
                lease_id=self.lease_id,
                expected_fence=self.fence,
            )
            self._sync_fence_from_record(record)
            self.phase = MutationWorktreePhase.READY
            self._write_journal()

    def apply_replacements(
        self,
        file_replacements: Mapping[str, str | bytes],
        scope: MutationWriteScope | Mapping[str, Any],
        *,
        lease_id: str | None = None,
        fence: int | None = None,
    ) -> MutationApplyResult:
        """Apply path replacements under fence ownership."""

        with self._lock:
            if lease_id is not None and fence is not None:
                self._require_owner(lease_id=lease_id, fence=fence)
            self.assert_under_owned_root()
            try:
                self.assert_base_current()
            except MutationWorktreeError as exc:
                result = MutationApplyResult(
                    applied=False,
                    reason_codes=(exc.reason_code,),
                    paths=(),
                    pre_tree=self.base_tree,
                    post_tree=self.base_tree,
                    base_commit=self.base_commit,
                    worktree_path=normalize_workspace_path(self.worktree_path),
                    diagnostic=_clip(str(exc)),
                )
                self.phase = MutationWorktreePhase.REJECTED
                self._write_journal(apply=result.to_dict())
                return result

            self.phase = MutationWorktreePhase.APPLYING
            self._write_journal()
            result = apply_file_replacements(
                file_replacements,
                worktree_root=self.worktree_path,
                scope=scope,
                expected_base_commit=self.base_commit,
                expected_base_tree=self.base_tree,
                require_linked_worktree=True,
            )
            self.phase = (
                MutationWorktreePhase.APPLIED
                if result.applied
                else MutationWorktreePhase.REJECTED
            )
            self._write_journal(apply=result.to_dict())
            self.assert_root_unmutated()
            return result

    def admit(
        self,
        candidate: MutationCandidate | Mapping[str, Any],
        *,
        lease_id: str | None = None,
        fence: int | None = None,
        assurance_manifest: DetectionAssuranceManifest | Mapping[str, Any] | None = None,
        declared_paths: Sequence[str] | None = None,
        allow_authority_fixture: bool = False,
        notes: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MutationAdmissionResult:
        """Run AAE-024 admission against this owned disposable worktree."""

        with self._lock:
            if lease_id is not None and fence is not None:
                self._require_owner(lease_id=lease_id, fence=fence)
            self.assert_under_owned_root()
            self.phase = MutationWorktreePhase.ADMITTING
            self._write_journal()
            try:
                result = admit_mutation(
                    candidate,
                    worktree_path=self.worktree_path,
                    lease_id=self.lease_id,
                    fence=self.fence,
                    lifecycle_store=self.lifecycle_store,
                    repo_root=self.repo_root,
                    base_commit=self.base_commit,
                    declared_paths=declared_paths,
                    assurance_manifest=assurance_manifest,
                    allow_authority_fixture=allow_authority_fixture,
                    require_lifecycle=True,
                    notes=notes,
                    metadata=metadata,
                )
            except AdmissionError as exc:
                self.phase = MutationWorktreePhase.REJECTED
                self._write_journal(admission_error=_clip(str(exc)))
                raise
            self.phase = (
                MutationWorktreePhase.ADMITTED
                if result.admitted
                else MutationWorktreePhase.REJECTED
            )
            self._write_journal(admission=result.to_dict())
            self.assert_root_unmutated()
            return result

    def authorize_cleanup(
        self,
        *,
        caller_lease_id: str = "",
    ) -> CleanupDecision:
        return self.lifecycle_store.authorize_cleanup(
            workspace_path=self.worktree_path,
            branch=self.branch,
            caller_lease_id=caller_lease_id or self.lease_id,
        )

    def cleanup(
        self,
        *,
        lease_id: str | None = None,
        fence: int | None = None,
        reason: str = "owner_cleanup",
        force_peer: bool = False,
    ) -> dict[str, Any]:
        """Fenced cleanup of the disposable worktree.

        The live owner may always clean its own attempt. Peers may clean only
        when ``authorize_cleanup`` allows it (terminal or reclaimed-stale).
        """

        with self._lock:
            if self._closed and self.phase is MutationWorktreePhase.TERMINAL:
                return {
                    "cleaned": True,
                    "already_terminal": True,
                    "worktree_path": str(self.worktree_path),
                }
            owner_cleanup = False
            if lease_id is not None and fence is not None:
                try:
                    self._require_owner(lease_id=lease_id, fence=fence)
                    owner_cleanup = True
                except MutationWorktreeFenceError:
                    if not force_peer:
                        raise
            if not owner_cleanup:
                decision = self.authorize_cleanup(
                    caller_lease_id=str(lease_id or "")
                )
                if not decision.allowed:
                    raise MutationWorktreeFenceError(
                        f"cleanup denied: {decision.reason}",
                        reason_code="cleanup_denied",
                    )
            else:
                self.phase = MutationWorktreePhase.CLEANING
                self._write_journal()
                try:
                    record = self.lifecycle_store.mark_terminal(
                        self.worktree_path,
                        lease_id=self.lease_id,
                        expected_fence=self.fence,
                        reason=reason,
                    )
                    self._sync_fence_from_record(record)
                except (OwnershipError, FenceMismatchError, WorktreeLifecycleError) as exc:
                    raise MutationWorktreeFenceError(
                        str(exc),
                        reason_code="cleanup_denied",
                    ) from exc

            self.phase = MutationWorktreePhase.CLEANING
            self._write_journal()
            removed = self._remove_worktree()
            self.phase = MutationWorktreePhase.TERMINAL
            self._closed = True
            self._write_journal(cleaned=True, removed=removed, reason=reason)
            if self.journal_path is not None and self.journal_path.exists():
                try:
                    self.journal_path.unlink()
                except OSError:
                    pass
            return {
                "cleaned": True,
                "removed": removed,
                "worktree_path": str(self.worktree_path),
                "reason": reason,
                "lease_id": self.lease_id,
                "fence": int(self.fence),
            }

    def _remove_worktree(self) -> bool:
        path = self.worktree_path
        # Never remove production root even if misconfigured.
        if normalize_workspace_path(path) == normalize_workspace_path(self.repo_root):
            raise MutationWorktreeError(
                "refusing to remove production repository root",
                reason_code="production_root",
            )
        try:
            path.resolve().relative_to(self.worktree_parent.resolve())
        except (OSError, ValueError) as exc:
            raise MutationWorktreeError(
                "refusing to remove path outside owned worktree parent",
                reason_code="path_escape",
            ) from exc

        if not path.exists():
            _run_git(
                ["worktree", "prune", "--expire", "now"],
                cwd=self.repo_root,
            )
            return False
        result = _run_git(
            ["worktree", "remove", "--force", str(path)],
            cwd=self.repo_root,
        )
        if result.returncode != 0 and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        _run_git(["worktree", "prune", "--expire", "now"], cwd=self.repo_root)
        return not path.exists()

    def close(self) -> dict[str, Any]:
        """Owner context-manager cleanup."""

        if self._closed:
            return {"cleaned": True, "already_terminal": True}
        return self.cleanup(
            lease_id=self.lease_id,
            fence=self.fence,
            reason="context_close",
        )

    def __enter__(self) -> "IsolatedMutationWorktree":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except MutationWorktreeFenceError:
            recover_mutation_worktree(
                lifecycle_store=self.lifecycle_store,
                worktree_path=self.worktree_path,
                repo_root=self.repo_root,
                worktree_parent=self.worktree_parent,
            )

    def to_dict(self) -> dict[str, Any]:
        return self._journal_payload()


# ---------------------------------------------------------------------------
# Create / recover
# ---------------------------------------------------------------------------


def _create_detached_worktree_at_base(
    *,
    repo_root: Path,
    worktree_path: Path,
    base_commit: str,
) -> None:
    """Create a detached disposable worktree at the exact scanned base commit."""

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if worktree_path.exists():
        raise MutationWorktreeError(
            f"worktree path already exists: {worktree_path}",
            reason_code="worktree_exists",
        )
    resolved = _rev_parse(repo_root, base_commit)
    # Detached: never creates or updates a production branch tip.
    result = _run_git(
        ["worktree", "add", "--detach", str(worktree_path), resolved],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise MutationWorktreeError(
            _clip(result.stderr or result.stdout or "git worktree add failed"),
            reason_code="worktree_create_failed",
        )
    head = _head_commit(worktree_path)
    if head != resolved:
        _run_git(
            ["worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_root,
        )
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        raise MutationWorktreeError(
            "created worktree HEAD does not match requested base",
            reason_code="stale_base",
        )
    if not _is_linked_worktree(worktree_path):
        _run_git(
            ["worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_root,
        )
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        raise MutationWorktreeError(
            "created path is not a linked disposable worktree",
            reason_code="worktree_create_failed",
        )


def create_mutation_worktree(
    *,
    repo_root: Path | str,
    worktree_path: Path | str | None = None,
    worktree_parent: Path | str | None = None,
    base_commit: str | None = None,
    base_tree: str | None = None,
    task_id: str,
    attempt: int = 1,
    lane_id: str = "aae-mutation",
    canonical_task_cid: str = "",
    merge_target: str = "HEAD",
    branch: str = "",
    lifecycle_store: WorktreeLifecycleStore | None = None,
    lease_id: str | None = None,
    retain_on_success: bool = False,
) -> IsolatedMutationWorktree:
    """Acquire a fenced claim then create a disposable worktree at ``base_commit``.

    The lifecycle preparing record is published **before** ``git worktree add``
    so peer cleaners never treat the checkout as an unclaimed orphan. The
    production repository root HEAD and branches are never mutated.
    """

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise MutationWorktreeError(
            "repo_root is not a Git repository",
            reason_code="invalid_repo",
        )
    if not _is_production_repo_root(root) and not (root / ".git").exists():
        raise MutationWorktreeError(
            "repo_root is not a Git repository",
            reason_code="invalid_repo",
        )

    root_head = _head_commit(root)
    resolved_base = _rev_parse(root, base_commit or "HEAD")
    resolved_tree = base_tree or _tree_for_commit(root, resolved_base)
    actual_tree = _tree_for_commit(root, resolved_base)
    if actual_tree != resolved_tree:
        raise MutationWorktreeError(
            "base_tree does not match base_commit",
            reason_code="stale_base",
        )

    store = lifecycle_store or WorktreeLifecycleStore(repo_root=root)

    if worktree_parent is not None:
        parent = Path(worktree_parent).resolve()
    elif worktree_path is not None:
        parent = Path(worktree_path).resolve().parent
    else:
        parent = (root.parent / "aae-mutation-worktrees").resolve()
    parent.mkdir(parents=True, exist_ok=True)

    # Owned-root fence: parent must not be inside the production checkout, and
    # must never equal the production root itself.
    try:
        parent.relative_to(root)
        # Parent is under production root — only allowed if it is a dedicated
        # subdirectory that is not the root itself; still reject equality.
        if parent == root:
            raise MutationWorktreeError(
                "worktree_parent must not equal production repo root",
                reason_code="production_root",
            )
    except ValueError:
        # Parent is outside production root — preferred.
        pass

    if worktree_path is None:
        digest = hashlib.sha256(
            f"{task_id}:{attempt}:{resolved_base}:{uuid.uuid4().hex}".encode("utf-8")
        ).hexdigest()[:12]
        workspace = parent / f"mutant-{digest}"
    else:
        workspace = Path(worktree_path)

    # Reject production-root targets before owned-parent escape checks so the
    # reason code is specific when a caller passes the repo root itself.
    if normalize_workspace_path(workspace) == normalize_workspace_path(root):
        raise MutationWorktreeError(
            "worktree_path must not equal production repo root",
            reason_code="production_root",
        )

    if worktree_path is not None:
        try:
            workspace.resolve().relative_to(parent)
        except (OSError, ValueError) as exc:
            # Allow exact parent/child when path not yet created.
            try:
                workspace.parent.resolve().relative_to(parent)
            except (OSError, ValueError):
                raise MutationWorktreeError(
                    "worktree_path escapes owned worktree parent",
                    reason_code="path_escape",
                ) from exc

    branch_name = branch or (
        f"aae-mutant/{_safe_branch_segment(task_id)}-a{int(attempt)}"
    )

    # Publish preparing claim before materializing the checkout.
    try:
        record = store.begin_preparing(
            task_id=task_id,
            canonical_task_cid=canonical_task_cid,
            attempt=int(attempt),
            lane_id=lane_id,
            workspace_path=workspace,
            branch=branch_name,
            merge_target=merge_target,
            lease_id=lease_id,
            state_dir=str(store.store_dir or ""),
            owner=current_process_birth(),
        )
    except DuplicateAttemptError as exc:
        raise MutationWorktreeFenceError(
            str(exc),
            reason_code="duplicate_attempt",
        ) from exc

    isolated = IsolatedMutationWorktree(
        repo_root=root,
        worktree_path=workspace,
        worktree_parent=parent,
        base_commit=resolved_base,
        base_tree=resolved_tree,
        task_id=task_id,
        attempt=int(attempt),
        lane_id=lane_id,
        lease_id=record.lease_id,
        fence=int(record.fence),
        lifecycle_store=store,
        phase=MutationWorktreePhase.PREPARING,
        branch=record.branch,
        merge_target=merge_target,
        canonical_task_cid=canonical_task_cid,
        root_head_at_create=root_head,
        retain_on_success=retain_on_success,
    )
    isolated._write_journal()

    try:
        _create_detached_worktree_at_base(
            repo_root=root,
            worktree_path=workspace,
            base_commit=resolved_base,
        )
        isolated.assert_base_current()
        isolated.assert_under_owned_root()
        isolated.mark_ready()
    except Exception as exc:
        try:
            isolated.phase = MutationWorktreePhase.CLEANING
            isolated._write_journal(prepare_error=_clip(str(exc)))
            try:
                store.mark_terminal(
                    workspace,
                    lease_id=isolated.lease_id,
                    expected_fence=isolated.fence,
                    reason="prepare_failed",
                )
            except (OwnershipError, FenceMismatchError, WorktreeLifecycleError):
                pass
            try:
                isolated._remove_worktree()
            except MutationWorktreeError:
                pass
            isolated.phase = MutationWorktreePhase.TERMINAL
            isolated._closed = True
            isolated._write_journal(prepare_error=_clip(str(exc)), cleaned=True)
        except Exception:
            pass
        if isinstance(exc, (MutationWorktreeError, MutationWorktreeFenceError)):
            raise
        raise MutationWorktreeError(
            _clip(str(exc)),
            reason_code="worktree_create_failed",
        ) from exc

    # Prove production root unchanged after create.
    isolated.assert_root_unmutated()
    return isolated


def recover_mutation_worktree(
    *,
    lifecycle_store: WorktreeLifecycleStore,
    worktree_path: Path | str,
    repo_root: Path | str | None = None,
    worktree_parent: Path | str | None = None,
    journal_path: Path | str | None = None,
    caller_lease_id: str = "",
) -> dict[str, Any]:
    """Recover interrupted prepare/apply/admit/cleanup states for one workspace.

    * preparing without a checkout → terminal + prune
    * applying/admitting with dirty disposable tree → hard-reset to base when
      journal base is known, then allow owner/peer cleanup per fence policy
    * cleaning / terminal with residual path → force-remove when authorized

    Only paths under the owned worktree parent (when known) are removed.
    """

    workspace = Path(worktree_path)
    record = lifecycle_store.load_workspace(workspace)
    store_dir = lifecycle_store.store_dir
    journal: dict[str, Any] | None = None
    if journal_path is not None:
        journal = _load_json_dict(Path(journal_path))
    elif store_dir is not None:
        digest = hashlib.sha256(
            normalize_workspace_path(workspace).encode("utf-8")
        ).hexdigest()[:16]
        journal = _load_json_dict(Path(store_dir) / f"aae-attempt-{digest}.json")

    root = Path(repo_root) if repo_root is not None else None
    if root is None and journal and journal.get("repo_root"):
        root = Path(str(journal["repo_root"]))
    if root is None and record is not None and record.repo_root:
        root = Path(record.repo_root)
    if root is None:
        root = Path(lifecycle_store.repo_root)

    parent: Path | None = None
    if worktree_parent is not None:
        parent = Path(worktree_parent).resolve()
    elif journal and journal.get("worktree_parent"):
        parent = Path(str(journal["worktree_parent"])).resolve()

    phase = ""
    base_commit = ""
    if journal:
        phase = str(journal.get("phase") or "")
        base_commit = str(journal.get("base_commit") or "")
    if record is not None and not phase:
        phase = record.state.value

    actions: list[str] = []

    # Interrupted apply/admit: restore disposable tree to bound base when possible.
    if (
        phase
        in {
            MutationWorktreePhase.APPLYING.value,
            MutationWorktreePhase.ADMITTING.value,
            MutationWorktreePhase.APPLIED.value,
            "applying",
            "admitting",
            "applied",
        }
        and workspace.is_dir()
        and base_commit
    ):
        reset = _run_git(["reset", "--hard", base_commit], cwd=workspace)
        clean = _run_git(["clean", "-fdx"], cwd=workspace)
        actions.append(
            "reset_base"
            if reset.returncode == 0 and clean.returncode == 0
            else "reset_failed"
        )

    owner_lease = str(
        (journal or {}).get("lease_id")
        or (record.lease_id if record is not None else "")
        or caller_lease_id
        or ""
    )
    raw_fence = (journal or {}).get("fence") if journal else None
    if raw_fence is None and record is not None:
        raw_fence = record.fence
    try:
        owner_fence = int(raw_fence) if raw_fence is not None else 0
    except (TypeError, ValueError):
        owner_fence = int(record.fence) if record is not None else 0

    decision = lifecycle_store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id=caller_lease_id or owner_lease,
    )

    owner_may_force = (
        record is not None
        and record.is_nonterminal
        and owner_lease
        and owner_lease == str(record.lease_id)
        and owner_fence == int(record.fence)
        and (
            not decision.allowed
            or phase
            in {
                MutationWorktreePhase.PREPARING.value,
                MutationWorktreePhase.APPLYING.value,
                MutationWorktreePhase.ADMITTING.value,
                MutationWorktreePhase.CLEANING.value,
                "preparing",
                "applying",
                "admitting",
                "cleaning",
            }
        )
    )
    if owner_may_force and not (
        decision.allowed and record is not None and record.is_terminal
    ):
        try:
            lifecycle_store.mark_terminal(
                workspace,
                lease_id=owner_lease,
                expected_fence=owner_fence,
                reason="recovery",
            )
            actions.append("marked_terminal")
            decision = lifecycle_store.authorize_cleanup(
                workspace_path=workspace,
                caller_lease_id=owner_lease,
            )
        except (OwnershipError, FenceMismatchError, WorktreeLifecycleError):
            actions.append("terminal_denied")

    removed = False
    may_remove = decision.allowed or (
        record is not None and record.state is WorkspaceLifecycleState.TERMINAL
    )
    # Hard fence: never remove production root or paths outside owned parent.
    if may_remove and workspace.exists():
        if normalize_workspace_path(workspace) == normalize_workspace_path(root):
            actions.append("production_root_denied")
            may_remove = False
        elif parent is not None:
            try:
                workspace.resolve().relative_to(parent)
            except (OSError, ValueError):
                actions.append("path_escape_denied")
                may_remove = False

    if may_remove:
        if workspace.exists():
            result = _run_git(
                ["worktree", "remove", "--force", str(workspace)],
                cwd=root,
            )
            if result.returncode != 0 and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)
            removed = not workspace.exists()
            actions.append("removed" if removed else "remove_failed")
        _run_git(["worktree", "prune", "--expire", "now"], cwd=root)
        actions.append("pruned")
        if store_dir is not None:
            digest = hashlib.sha256(
                normalize_workspace_path(workspace).encode("utf-8")
            ).hexdigest()[:16]
            journal_file = Path(store_dir) / f"aae-attempt-{digest}.json"
            if journal_file.exists():
                try:
                    journal_file.unlink()
                    actions.append("journal_cleared")
                except OSError:
                    actions.append("journal_clear_failed")
    else:
        actions.append(f"cleanup_denied:{decision.reason}")

    return {
        "schema": MUTATION_WORKTREE_ATTEMPT_SCHEMA,
        "recovered": True,
        "worktree_path": str(workspace),
        "phase": phase,
        "actions": actions,
        "cleanup_allowed": bool(decision.allowed),
        "cleanup_reason": decision.reason,
        "removed": removed,
    }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class IsolatedMutationWorktreeExecutor:
    """Sole mutation-worktree lifecycle owner and admission pipeline.

    Interface: ``IsolatedMutationWorktreeExecutor@1``

    Creates and destroys disposable owned worktrees; applies bounded edits;
    delegates semantic admission to AAE-024 ``admit_mutation`` (which never
    creates or destroys worktrees).
    """

    repo_root: Path
    worktree_parent: Path
    lifecycle_store: WorktreeLifecycleStore
    lane_id: str = "aae-mutation"
    retain_on_success: bool = False

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.worktree_parent = Path(self.worktree_parent).resolve()
        if normalize_workspace_path(self.worktree_parent) == normalize_workspace_path(
            self.repo_root
        ):
            raise MutationWorktreeError(
                "worktree_parent must not equal production repo root",
                reason_code="production_root",
            )
        self.worktree_parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        *,
        repo_root: Path | str,
        worktree_parent: Path | str | None = None,
        lifecycle_store: WorktreeLifecycleStore | None = None,
        lane_id: str = "aae-mutation",
        retain_on_success: bool = False,
        store_dir: Path | str | None = None,
    ) -> "IsolatedMutationWorktreeExecutor":
        root = Path(repo_root).resolve()
        parent = (
            Path(worktree_parent).resolve()
            if worktree_parent is not None
            else (root.parent / "aae-mutation-worktrees").resolve()
        )
        store = lifecycle_store or WorktreeLifecycleStore(
            repo_root=root,
            store_dir=Path(store_dir) if store_dir is not None else None,
        )
        return cls(
            repo_root=root,
            worktree_parent=parent,
            lifecycle_store=store,
            lane_id=lane_id,
            retain_on_success=retain_on_success,
        )

    def create_worktree(
        self,
        *,
        task_id: str,
        attempt: int = 1,
        base_commit: str | None = None,
        base_tree: str | None = None,
        canonical_task_cid: str = "",
        worktree_path: Path | str | None = None,
        lease_id: str | None = None,
        branch: str = "",
    ) -> IsolatedMutationWorktree:
        return create_mutation_worktree(
            repo_root=self.repo_root,
            worktree_path=worktree_path,
            worktree_parent=self.worktree_parent,
            base_commit=base_commit,
            base_tree=base_tree,
            task_id=task_id,
            attempt=attempt,
            lane_id=self.lane_id,
            canonical_task_cid=canonical_task_cid,
            branch=branch,
            lifecycle_store=self.lifecycle_store,
            lease_id=lease_id,
            retain_on_success=self.retain_on_success,
        )

    def execute_and_admit(
        self,
        candidate: MutationCandidate | Mapping[str, Any],
        *,
        file_replacements: Mapping[str, str | bytes],
        task_id: str,
        attempt: int = 1,
        base_commit: str | None = None,
        base_tree: str | None = None,
        assurance_manifest: DetectionAssuranceManifest | Mapping[str, Any] | None = None,
        scope: MutationWriteScope | Mapping[str, Any] | None = None,
        declared_paths: Sequence[str] | None = None,
        allow_authority_fixture: bool = False,
        cleanup: bool = True,
        canonical_task_cid: str = "",
        notes: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MutationExecutionResult:
        """Create a disposable worktree, apply edits, run admission, and clean up.

        Production repository HEAD and branches are never mutated. Failures are
        sealed into :class:`MutationExecutionResult` when possible; malformed
        API inputs may raise :class:`MutationWorktreeError` /
        :class:`AdmissionError` before sealing.
        """

        if isinstance(candidate, Mapping):
            candidate_id = str(candidate.get("candidate_id") or "")
            candidate_cid = str(candidate.get("candidate_cid") or "")
        else:
            candidate_id = candidate.candidate_id
            candidate_cid = candidate.candidate_cid

        write_scope = scope
        if write_scope is None:
            write_scope = MutationWriteScope.from_candidate(
                candidate,
                allow_authority_fixture=allow_authority_fixture,
            )
        elif isinstance(write_scope, Mapping):
            write_scope = MutationWriteScope.from_dict(write_scope)

        root_head_before = _head_commit(self.repo_root)
        isolated: IsolatedMutationWorktree | None = None
        apply_result: MutationApplyResult | None = None
        admission_result: MutationAdmissionResult | None = None
        cleaned = False
        reason_codes: list[str] = []
        diagnostic = ""
        disposition = "infrastructure_failure"

        try:
            isolated = self.create_worktree(
                task_id=task_id,
                attempt=attempt,
                base_commit=base_commit,
                base_tree=base_tree,
                canonical_task_cid=canonical_task_cid,
            )
            apply_result = isolated.apply_replacements(
                file_replacements,
                write_scope,
                lease_id=isolated.lease_id,
                fence=isolated.fence,
            )
            if not apply_result.applied:
                reason_codes.extend(apply_result.reason_codes or ("apply_failed",))
                diagnostic = apply_result.diagnostic
                disposition = "invalid_mutant"
            else:
                admission_result = isolated.admit(
                    candidate,
                    lease_id=isolated.lease_id,
                    fence=isolated.fence,
                    assurance_manifest=assurance_manifest,
                    declared_paths=declared_paths
                    or tuple(apply_result.paths)
                    or None,
                    allow_authority_fixture=allow_authority_fixture,
                    notes=notes,
                    metadata=metadata,
                )
                disposition = admission_result.disposition
                reason_codes.extend(admission_result.reason_codes)
                diagnostic = admission_result.diagnostic
        except (MutationWorktreeError, MutationWorktreeFenceError) as exc:
            reason_codes.append(exc.reason_code)
            diagnostic = _clip(str(exc))
            disposition = "infrastructure_failure"
        except AdmissionError as exc:
            reason_codes.append(exc.reason_code)
            diagnostic = _clip(str(exc))
            disposition = "infrastructure_failure"
        finally:
            if isolated is not None and cleanup and not isolated._closed:
                try:
                    isolated.cleanup(
                        lease_id=isolated.lease_id,
                        fence=isolated.fence,
                        reason="execute_and_admit_cleanup",
                    )
                    cleaned = True
                except MutationWorktreeFenceError:
                    recovery = recover_mutation_worktree(
                        lifecycle_store=self.lifecycle_store,
                        worktree_path=isolated.worktree_path,
                        repo_root=self.repo_root,
                        worktree_parent=self.worktree_parent,
                        caller_lease_id=isolated.lease_id,
                    )
                    cleaned = bool(recovery.get("removed"))
                    if not cleaned:
                        reason_codes.append("cleanup_incomplete")

        # Production root must still match pre-create HEAD.
        root_head_after = _head_commit(self.repo_root)
        if root_head_after != root_head_before:
            reason_codes.append("root_mutated")
            disposition = "infrastructure_failure"
            diagnostic = _clip(
                diagnostic + "; production root HEAD mutated during execution"
            )

        apply_paths = tuple(apply_result.paths) if apply_result is not None else ()
        admission_cid = (
            admission_result.admission_cid if admission_result is not None else None
        )
        identity = _stable_execution_identity(
            candidate_cid=candidate_cid,
            disposition=disposition,
            reason_codes=reason_codes,
            base_commit=(
                isolated.base_commit
                if isolated is not None
                else (base_commit or root_head_before)
            ),
            post_tree=(
                apply_result.post_tree
                if apply_result is not None and apply_result.applied
                else (
                    isolated.base_tree
                    if isolated is not None
                    else (base_tree or "")
                )
            ),
            apply_paths=apply_paths,
            admission_cid=admission_cid,
        )

        executed = (
            apply_result is not None
            and apply_result.applied
            and admission_result is not None
        )
        return MutationExecutionResult(
            executed=executed,
            disposition=disposition,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            candidate_id=candidate_id,
            candidate_cid=candidate_cid,
            worktree_path=(
                normalize_workspace_path(isolated.worktree_path)
                if isolated is not None
                else ""
            ),
            lease_id=isolated.lease_id if isolated is not None else "",
            fence=int(isolated.fence) if isolated is not None else 0,
            base_commit=(
                isolated.base_commit
                if isolated is not None
                else (base_commit or root_head_before)
            ),
            base_tree=(
                isolated.base_tree if isolated is not None else (base_tree or "")
            ),
            pre_tree=apply_result.pre_tree if apply_result is not None else "",
            post_tree=apply_result.post_tree if apply_result is not None else "",
            root_head=root_head_before,
            apply=apply_result.to_dict() if apply_result is not None else None,
            admission=(
                admission_result.to_dict() if admission_result is not None else None
            ),
            cleaned=cleaned,
            identity_cid=identity,
            diagnostic=_clip(diagnostic),
            metadata=dict(metadata or {}),
        )

    def recover(
        self,
        worktree_path: Path | str,
        *,
        caller_lease_id: str = "",
        journal_path: Path | str | None = None,
    ) -> dict[str, Any]:
        return recover_mutation_worktree(
            lifecycle_store=self.lifecycle_store,
            worktree_path=worktree_path,
            repo_root=self.repo_root,
            worktree_parent=self.worktree_parent,
            journal_path=journal_path,
            caller_lease_id=caller_lease_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_WORKTREE_SCHEMA,
            "interface": ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE,
            "adapter_id": ADAPTER_ID,
            "board_namespace": BOARD_NAMESPACE,
            "repo_root": str(self.repo_root),
            "worktree_parent": str(self.worktree_parent),
            "lane_id": self.lane_id,
            "retain_on_success": bool(self.retain_on_success),
            "evidence": AAE_ISOLATED_EXECUTOR_EVIDENCE,
            "admit_mutation_interface": ADMIT_MUTATION_INTERFACE,
        }


def isolated_mutation_worktree_executor_descriptor() -> dict[str, Any]:
    """Return the sealed public-symbol descriptor for this module."""

    return {
        "schema": MUTATION_WORKTREE_SCHEMA,
        "interface": ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "evidence": AAE_ISOLATED_EXECUTOR_EVIDENCE,
        "admit_mutation_interface": ADMIT_MUTATION_INTERFACE,
        "symbols": [
            "IsolatedMutationWorktreeExecutor",
            "IsolatedMutationWorktree",
            "MutationWriteScope",
            "MutationApplyResult",
            "MutationExecutionResult",
            "MutationWorktreePhase",
            "create_mutation_worktree",
            "apply_file_replacements",
            "recover_mutation_worktree",
            "isolated_mutation_worktree_executor_descriptor",
        ],
        "invariants": [
            "sole_mutation_worktree_lifecycle_owner",
            "never_mutates_production_trees_or_branches",
            "never_escapes_owned_worktree_roots",
            "no_credentials_or_network",
            "no_undeclared_authority_edits",
            "fenced_recoverable_cleanup",
            "aae024_validates_caller_supplied_worktree_only",
        ],
    }


__all__ = [
    "AAE_ISOLATED_EXECUTOR_EVIDENCE",
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "ISOLATED_MUTATION_WORKTREE_EXECUTOR_INTERFACE",
    "MUTATION_APPLY_RESULT_SCHEMA",
    "MUTATION_EXECUTION_RESULT_SCHEMA",
    "MUTATION_WORKTREE_ATTEMPT_SCHEMA",
    "MUTATION_WORKTREE_SCHEMA",
    "IsolatedMutationWorktree",
    "IsolatedMutationWorktreeExecutor",
    "MutationApplyResult",
    "MutationExecutionResult",
    "MutationWorktreeError",
    "MutationWorktreeFenceError",
    "MutationWorktreePhase",
    "MutationWriteScope",
    "apply_file_replacements",
    "create_mutation_worktree",
    "isolated_mutation_worktree_executor_descriptor",
    "recover_mutation_worktree",
]
