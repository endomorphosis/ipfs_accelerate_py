"""Isolated Git worktree application for VerifiedGuiOptimizer (VGO-050).

Interfaces owned by this module:

* ``GuiIsolatedWorktreeExecutor@1`` — create a fenced disposable worktree,
  apply only an admitted patch, and promote nothing
* ``GuiPatchApplicationReceipt@1`` — typed record of the parent revision,
  observed diff, lease/fence, and cleanup state

This module never mutates the canonical branch.  Rejected, review-gated, or
interrupted proposals leave the source checkout byte-identical.  Host paths
and subprocess argv are fixed here; browser strings, broad roots, destructive
reset, and caller command strings are rejected.

Fail-closed invariants:

* patch scope must ALLOW before any worktree is created;
* the exact source revision and worktree parent are recorded;
* post-apply status/diff is rechecked against the admitted scope;
* lease/fence failures create no checkout;
* subprocess operations use ``/usr/bin/git`` and a closed verb set;
* ``promoted`` is always false.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..merge.worktree_lifecycle import (
    DuplicateAttemptError,
    FenceMismatchError,
    OwnershipError,
    WorkspaceLifecycleRecord,
    WorkspaceLifecycleState,
    WorktreeLifecycleError,
    WorktreeLifecycleStore,
    new_lease_id,
)
from .authority import (
    AuthorityReasonCode,
    GuiAuthorityError,
    GuiHostBoundaryPolicy,
)
from .patch_scope import (
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
    GuiImprovementProposalView,
    GuiPatchScopeDecision,
    GuiPatchScopeError,
    GuiPatchScopeGate,
    PatchScopeObservation,
    PatchScopeReasonCode,
    default_patch_scope_gate,
    parse_unified_diff,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE: Final[str] = (
    "GuiIsolatedWorktreeExecutor@1"
)
GUI_PATCH_APPLICATION_RECEIPT_INTERFACE: Final[str] = (
    "GuiPatchApplicationReceipt@1"
)
GUI_ISOLATED_WORKTREE_EXECUTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "isolated-worktree-executor@1"
)
GUI_PATCH_APPLICATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "patch-application-receipt@1"
)

HOST_GIT_EXECUTABLE: Final[str] = "/usr/bin/git"
HOST_GIT_HOOKS_PATH: Final[str] = "/dev/null"
HOST_VALIDATION_PATH: Final[str] = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
)

ALLOWED_GIT_VERBS: Final[frozenset[str]] = frozenset(
    {
        "apply",
        "branch",
        "diff",
        "rev-parse",
        "status",
        "symbolic-ref",
        "worktree",
    }
)
FORBIDDEN_GIT_VERBS: Final[frozenset[str]] = frozenset(
    {
        "am",
        "checkout",
        "cherry-pick",
        "clean",
        "config",
        "filter-branch",
        "gc",
        "merge",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "switch",
    }
)
FORBIDDEN_GIT_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--directory",
        "--exec",
        "--hard",
        "--mixed",
        "--soft",
        "--squash",
        "--unsafe-paths",
    }
)

ISOLATED_BRANCH_PREFIX: Final[str] = "vgo/isolated/"
_ISOLATED_BRANCH_RE = re.compile(r"^vgo/isolated/[A-Za-z0-9._/-]+$")
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CANONICAL_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")
_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_COMMAND_META_RE = re.compile(r"[;&|`$<>\n]|\$\(|\)")

BROAD_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "/",
        "/dev",
        "/etc",
        "/home",
        "/media",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
        "/var/tmp",
    }
)

_APPLY_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "attempt",
        "canonical_task_cid",
        "diff_text",
        "invalidation",
        "lane_id",
        "lease_id",
        "observation",
        "proposal",
        "repository_path",
        "source_revision",
        "task_id",
        "worktree_parent",
    }
)
_FORBIDDEN_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "argv",
        "browser_input",
        "command",
        "commands",
        "cwd",
        "destructive_reset",
        "file_path",
        "git_command",
        "host_path",
        "reset",
        "shell",
        "working_directory",
    }
)
_OBSERVATION_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_argument_digest",
        "action_binding_ids",
        "action_contract_evidence",
        "application_ids",
        "touched_component_ids",
        "touched_screenshot_ids",
        "touched_state_effect_ids",
        "touched_test_ids",
        "unresolved_paths",
        "visual_effect_observed",
    }
)

_GIT_ENV_BLOCKLIST: Final[tuple[str, ...]] = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_AUTHOR_DATE",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_NAME",
    "GIT_COMMITTER_DATE",
    "GIT_COMMITTER_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_DIR",
    "GIT_EDITOR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_SEQUENCE_EDITOR",
    "GIT_WORK_TREE",
)


class GuiWorktreeExecutorError(GuiAuthorityError):
    """Malformed worktree-executor input.  Never grants application."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_worktree_executor_input",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class ApplicationDisposition(str, Enum):
    """Closed outcomes for ``GuiPatchApplicationReceipt@1``."""

    APPLIED = "applied"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    FENCED = "fenced"


class CleanupState(str, Enum):
    """Whether the disposable worktree still exists."""

    RETAINED = "retained"
    REMOVED = "removed"
    NEVER_CREATED = "never_created"
    REMOVE_FAILED = "remove_failed"


class WorktreeExecutorReasonCode(str, Enum):
    """Stable reason codes for isolated worktree application."""

    APPLIED = "applied"
    NOT_PROMOTED = "not_promoted"
    SCOPE_REJECTED = "scope_rejected"
    SCOPE_REQUIRES_REVIEW = "scope_requires_review"
    LEASE_FENCE_FAILURE = "lease_fence_failure"
    UNDECLARED_FILE_POST_APPLY = "undeclared_file_post_apply"
    DIFF_SCOPE_MISMATCH = "diff_scope_mismatch"
    BROWSER_PATH_FORBIDDEN = "browser_path_forbidden"
    BROAD_ROOT_FORBIDDEN = "broad_root_forbidden"
    COMMAND_STRING_FORBIDDEN = "command_string_forbidden"
    DESTRUCTIVE_RESET_FORBIDDEN = "destructive_reset_forbidden"
    SOURCE_REVISION_MISMATCH = "source_revision_mismatch"
    MISSING_SOURCE_REVISION = "missing_source_revision"
    REPOSITORY_INVALID = "repository_invalid"
    WORKTREE_PARENT_INVALID = "worktree_parent_invalid"
    GIT_UNAVAILABLE = "git_unavailable"
    GIT_OPERATION_FAILED = "git_operation_failed"
    INTERRUPTED = "interrupted"
    CLEANUP_REMOVED = "cleanup_removed"
    CLEANUP_RETAINED = "cleanup_retained"
    CLEANUP_NEVER_CREATED = "cleanup_never_created"
    CANONICAL_MUTATION_DETECTED = "canonical_mutation_detected"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    INVALID_WORKTREE_EXECUTOR_INPUT = "invalid_worktree_executor_input"
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiWorktreeExecutorError(
            f"{name} must be a string",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiWorktreeExecutorError(
            f"{name} must not contain NUL",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiWorktreeExecutorError(
            f"{name} must not be empty",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiWorktreeExecutorError(
            f"{name} must not contain NUL",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name},
        )
    if text_value == "" or text_value != text_value.strip():
        raise GuiWorktreeExecutorError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name},
        )
    return text_value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiWorktreeExecutorError(
            f"{name} must be an integer",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 1:
        raise GuiWorktreeExecutorError(
            f"{name} must be a positive integer",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": name, "value": value},
        )
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiWorktreeExecutorError(
            f"{name} must be a JSON object",
            reason_code=WorktreeExecutorReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiWorktreeExecutorError(
                f"{name} keys must be strings",
                reason_code=WorktreeExecutorReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiWorktreeExecutorError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=WorktreeExecutorReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_forbidden_keys(payload: Mapping[str, Any], noun: str) -> None:
    forbidden = sorted(set(payload) & set(_FORBIDDEN_REQUEST_KEYS))
    if forbidden:
        reason = WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
        if any(key in {"destructive_reset", "reset"} for key in forbidden):
            reason = WorktreeExecutorReasonCode.DESTRUCTIVE_RESET_FORBIDDEN.value
        if any(
            key in {"browser_input", "host_path", "file_path"}
            for key in forbidden
        ):
            reason = WorktreeExecutorReasonCode.BROWSER_PATH_FORBIDDEN.value
        raise GuiWorktreeExecutorError(
            f"{noun} contains forbidden host-control fields: {forbidden}",
            reason_code=reason,
            details={"noun": noun, "forbidden_fields": forbidden},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiWorktreeExecutorError(
            f"{key} must not be null when present",
            reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _text(payload[key], key, required=False)


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _identifier(payload[key], key)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = _require_mapping(value, "details")
    return MappingProxyType(dict(mapping))


def _safe_token(value: str, *, fallback: str = "task") -> str:
    cleaned = _SAFE_TOKEN_RE.sub("-", value).strip(".-_")
    return cleaned[:48] or fallback


def _sha256_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_browser_or_command_path(value: str) -> str:
    """Return a reason code when ``value`` is not a host filesystem path."""
    lowered = value.strip().lower()
    if lowered.startswith(("file:", "http:", "https:", "blob:", "data:", "about:")):
        return WorktreeExecutorReasonCode.BROWSER_PATH_FORBIDDEN.value
    if _URI_RE.match(value) and not value.startswith("/"):
        return WorktreeExecutorReasonCode.BROWSER_PATH_FORBIDDEN.value
    if _WINDOWS_DRIVE_RE.match(value) or value.startswith("//"):
        return WorktreeExecutorReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    if _COMMAND_META_RE.search(value):
        return WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
    if any(
        token in lowered
        for token in (
            "git apply",
            "git reset",
            "git checkout",
            "git clean",
            "rm -rf",
        )
    ):
        return WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
    return ""


def _resolve_host_directory(
    value: Any,
    name: str,
    *,
    must_exist: bool,
) -> Path:
    raw = _text(value, name)
    injected = _looks_like_browser_or_command_path(raw)
    if injected:
        raise GuiWorktreeExecutorError(
            f"{name} is not an explicit host directory",
            reason_code=injected,
            details={"field": name, "value": raw},
        )
    if ".." in Path(raw).parts:
        raise GuiWorktreeExecutorError(
            f"{name} must not contain parent-directory segments",
            reason_code=WorktreeExecutorReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            details={"field": name, "value": raw},
        )
    try:
        resolved = Path(raw).expanduser().resolve(strict=False)
    except OSError as exc:
        raise GuiWorktreeExecutorError(
            f"{name} could not be resolved",
            reason_code=WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value
            if name == "worktree_parent"
            else WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
            details={"field": name, "error": str(exc)},
        ) from exc
    rendered = str(resolved)
    if rendered in BROAD_ROOTS:
        raise GuiWorktreeExecutorError(
            f"{name} must not be a broad filesystem root",
            reason_code=WorktreeExecutorReasonCode.BROAD_ROOT_FORBIDDEN.value,
            details={"field": name, "value": rendered},
        )
    if not resolved.is_absolute():
        raise GuiWorktreeExecutorError(
            f"{name} must be an absolute host path",
            reason_code=WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value
            if name == "worktree_parent"
            else WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
            details={"field": name, "value": rendered},
        )
    if must_exist and not resolved.is_dir():
        raise GuiWorktreeExecutorError(
            f"{name} must be an existing directory",
            reason_code=WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value
            if name == "worktree_parent"
            else WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
            details={"field": name, "value": rendered},
        )
    return resolved


def _path_is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Host-fixed Git runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostGitResult:
    """Captured result of one host-fixed git argv."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return (self.stdout or self.stderr or "").strip()


def sealed_git_environment() -> dict[str, str]:
    """Return the host-fixed environment used for every git subprocess."""
    env = {
        "PATH": HOST_VALIDATION_PATH,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    home = os.environ.get("HOME")
    if type(home) is str and home:
        env["HOME"] = home
    tmpdir = os.environ.get("TMPDIR")
    if type(tmpdir) is str and tmpdir:
        env["TMPDIR"] = tmpdir
    return env


@dataclass(frozen=True)
class HostGitRunner:
    """Execute a closed git verb set with a host-fixed executable.

    Callers pass only the verb and its arguments.  The runner prefixes
    ``/usr/bin/git -c core.hooksPath=/dev/null`` and never uses a shell.
    """

    executable: str = HOST_GIT_EXECUTABLE
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        executable = _text(self.executable, "executable")
        if executable != HOST_GIT_EXECUTABLE:
            raise GuiWorktreeExecutorError(
                "git executable is fixed by the host",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"executable": executable},
            )
        object.__setattr__(self, "executable", executable)
        timeout = self.timeout_seconds
        if type(timeout) is not float and type(timeout) is not int:
            raise GuiWorktreeExecutorError(
                "timeout_seconds must be a number",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            )
        if float(timeout) <= 0:
            raise GuiWorktreeExecutorError(
                "timeout_seconds must be positive",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))

    def available(self) -> bool:
        return Path(self.executable).is_file() and os.access(
            self.executable, os.X_OK
        )

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
    ) -> HostGitResult:
        """Run one validated git argv in ``cwd`` without a shell."""
        if type(argv) is not list and type(argv) is not tuple:
            raise GuiWorktreeExecutorError(
                "git argv must be a sequence of strings",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"value_type": type(argv).__name__},
            )
        tokens = tuple(_exact_str(item, f"argv[{index}]") for index, item in enumerate(argv))
        if not tokens:
            raise GuiWorktreeExecutorError(
                "git argv must not be empty",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
            )
        self.validate_argv(tokens, cwd=cwd)
        if not self.available():
            return HostGitResult(
                argv=(self.executable, *tokens),
                returncode=127,
                stdout="",
                stderr=f"{self.executable} is not available",
            )
        full = (
            self.executable,
            "-c",
            f"core.hooksPath={HOST_GIT_HOOKS_PATH}",
            *tokens,
        )
        try:
            completed = subprocess.run(
                full,
                cwd=str(cwd),
                env=sealed_git_environment(),
                input=input_text,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            return HostGitResult(
                argv=full,
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            return HostGitResult(
                argv=full,
                returncode=124,
                stdout=str(exc.stdout or ""),
                stderr="git operation timed out",
            )
        except OSError as exc:
            return HostGitResult(
                argv=full,
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        return HostGitResult(
            argv=full,
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def validate_argv(self, argv: Sequence[str], *, cwd: Path) -> None:
        """Reject verbs, flags, and cwd values the host does not authorize."""
        verb = argv[0]
        if verb in FORBIDDEN_GIT_VERBS or verb == "reset":
            raise GuiWorktreeExecutorError(
                f"git verb {verb!r} is forbidden",
                reason_code=(
                    WorktreeExecutorReasonCode.DESTRUCTIVE_RESET_FORBIDDEN.value
                    if verb in {"reset", "clean", "checkout"}
                    else WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
                ),
                details={"verb": verb},
            )
        if verb not in ALLOWED_GIT_VERBS:
            raise GuiWorktreeExecutorError(
                f"git verb {verb!r} is not in the host-fixed set",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"verb": verb},
            )
        for token in argv:
            if token in FORBIDDEN_GIT_FLAGS or token.startswith("--unsafe"):
                raise GuiWorktreeExecutorError(
                    f"git flag {token!r} is forbidden",
                    reason_code=(
                        WorktreeExecutorReasonCode.DESTRUCTIVE_RESET_FORBIDDEN.value
                        if token in {"--hard", "--soft", "--mixed"}
                        else WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value
                    ),
                    details={"flag": token},
                )
            if token == "-c":
                raise GuiWorktreeExecutorError(
                    "git -c is reserved for the host runner",
                    reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                )
        if verb == "worktree":
            self._validate_worktree_argv(argv)
        elif verb == "branch":
            self._validate_branch_argv(argv)
        elif verb == "apply":
            self._validate_apply_argv(argv)
        cwd_path = Path(cwd)
        if not cwd_path.is_dir():
            raise GuiWorktreeExecutorError(
                "git cwd must be an existing directory",
                reason_code=WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
                details={"cwd": str(cwd_path)},
            )

    def _validate_worktree_argv(self, argv: Sequence[str]) -> None:
        if len(argv) < 2:
            raise GuiWorktreeExecutorError(
                "git worktree requires a subcommand",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
            )
        sub = argv[1]
        if sub == "add":
            if "-b" not in argv or "--force" in argv or "-f" in argv:
                raise GuiWorktreeExecutorError(
                    "worktree add must create a named isolated branch without --force",
                    reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                )
            branch = argv[argv.index("-b") + 1]
            if not _ISOLATED_BRANCH_RE.fullmatch(branch):
                raise GuiWorktreeExecutorError(
                    "worktree branch must use the isolated prefix",
                    reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                    details={"branch": branch},
                )
            return
        if sub == "remove":
            if "--force" not in argv:
                raise GuiWorktreeExecutorError(
                    "worktree remove requires --force on the isolated path only",
                    reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                )
            return
        raise GuiWorktreeExecutorError(
            f"git worktree {sub!r} is not authorized",
            reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
        )

    def _validate_branch_argv(self, argv: Sequence[str]) -> None:
        if len(argv) != 3 or argv[1] != "-D":
            raise GuiWorktreeExecutorError(
                "git branch may only force-delete an isolated branch",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
            )
        if not _ISOLATED_BRANCH_RE.fullmatch(argv[2]):
            raise GuiWorktreeExecutorError(
                "refusing to delete a non-isolated branch",
                reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"branch": argv[2]},
            )

    def _validate_apply_argv(self, argv: Sequence[str]) -> None:
        allowed_flags = {"--check", "--index", "--whitespace=nowarn"}
        for token in argv[1:]:
            if token not in allowed_flags:
                raise GuiWorktreeExecutorError(
                    f"git apply flag {token!r} is not authorized",
                    reason_code=WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                    details={"flag": token},
                )


# ---------------------------------------------------------------------------
# Typed request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalCheckoutSnapshot:
    """Exact identity of the source checkout before any worktree exists."""

    repository_path: str
    branch: str
    revision: str
    porcelain: str

    def matches(self, other: "CanonicalCheckoutSnapshot") -> bool:
        return (
            self.repository_path == other.repository_path
            and self.branch == other.branch
            and self.revision == other.revision
            and self.porcelain == other.porcelain
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "branch": self.branch,
            "porcelain": self.porcelain,
            "repository_path": self.repository_path,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class GuiWorktreeApplyRequest:
    """Closed apply request for ``GuiIsolatedWorktreeExecutor@1``."""

    repository_path: str
    worktree_parent: str
    proposal: Any
    diff_text: str
    invalidation: Any
    observation: Any = None
    source_revision: str = ""
    task_id: str = ""
    canonical_task_cid: str = ""
    attempt: int = 1
    lane_id: str = "vgo-lane-1"
    lease_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_path",
            _text(self.repository_path, "repository_path"),
        )
        object.__setattr__(
            self,
            "worktree_parent",
            _text(self.worktree_parent, "worktree_parent"),
        )
        object.__setattr__(self, "diff_text", _exact_str(self.diff_text, "diff_text"))
        if "\x00" in self.diff_text or not self.diff_text.strip():
            raise GuiWorktreeExecutorError(
                "diff_text must be a nonempty unified diff",
                reason_code=PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                details={"field": "diff_text"},
            )
        revision = self.source_revision
        if revision:
            text = _text(revision, "source_revision")
            if not _FULL_SHA_RE.fullmatch(text):
                raise GuiWorktreeExecutorError(
                    "source_revision must be a 40-character lowercase SHA-1",
                    reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                    details={"field": "source_revision"},
                )
            object.__setattr__(self, "source_revision", text)
        else:
            object.__setattr__(self, "source_revision", "")
        task_id = self.task_id or ""
        object.__setattr__(
            self,
            "task_id",
            _identifier(task_id, "task_id") if task_id else "",
        )
        cid = self.canonical_task_cid or ""
        object.__setattr__(
            self,
            "canonical_task_cid",
            _identifier(cid, "canonical_task_cid") if cid else "",
        )
        object.__setattr__(self, "attempt", _positive_int(self.attempt, "attempt"))
        object.__setattr__(
            self,
            "lane_id",
            _identifier(self.lane_id, "lane_id") if self.lane_id else "vgo-lane-1",
        )
        lease = self.lease_id or ""
        object.__setattr__(
            self,
            "lease_id",
            _identifier(lease, "lease_id") if lease else "",
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GuiWorktreeApplyRequest":
        payload = _require_mapping(raw, "request")
        _reject_forbidden_keys(payload, "request")
        _reject_unknown(payload, _APPLY_REQUEST_KEYS, "request")
        for key in ("repository_path", "worktree_parent", "proposal", "diff_text"):
            if key not in payload:
                raise GuiWorktreeExecutorError(
                    f"request.{key} is required",
                    reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                    details={"field": key},
                )
        attempt = payload["attempt"] if "attempt" in payload else 1
        return cls(
            repository_path=payload["repository_path"],
            worktree_parent=payload["worktree_parent"],
            proposal=payload["proposal"],
            diff_text=payload["diff_text"],
            invalidation=payload.get("invalidation"),
            observation=payload.get("observation"),
            source_revision=_optional_text(payload, "source_revision"),
            task_id=_optional_identifier(payload, "task_id"),
            canonical_task_cid=_optional_identifier(payload, "canonical_task_cid"),
            attempt=attempt,
            lane_id=_optional_identifier(payload, "lane_id") or "vgo-lane-1",
            lease_id=_optional_identifier(payload, "lease_id"),
        )

    @classmethod
    def from_any(cls, value: Any) -> "GuiWorktreeApplyRequest":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        raise GuiWorktreeExecutorError(
            "request must be a GuiWorktreeApplyRequest or JSON object",
            reason_code=WorktreeExecutorReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(value).__name__},
        )


@dataclass(frozen=True)
class GuiPatchApplicationReceipt:
    """Typed, fail-closed receipt for ``GuiPatchApplicationReceipt@1``."""

    disposition: ApplicationDisposition
    reason_codes: tuple[str, ...]
    interface: str = GUI_PATCH_APPLICATION_RECEIPT_INTERFACE
    schema: str = GUI_PATCH_APPLICATION_RECEIPT_SCHEMA
    applied: bool = False
    promoted: bool = False
    repository_path: str = ""
    worktree_path: str = ""
    worktree_parent: str = ""
    isolated_branch: str = ""
    canonical_branch: str = ""
    source_revision: str = ""
    parent_revision: str = ""
    observed_diff: str = ""
    observed_paths: tuple[str, ...] = ()
    admitted_paths: tuple[str, ...] = ()
    cleanup_state: CleanupState = CleanupState.NEVER_CREATED
    lease_id: str = ""
    fence: int = 0
    lifecycle_state: str = ""
    proposal_id: str = ""
    patch_digest: str = ""
    observed_diff_digest: str = ""
    scope_decision: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.disposition) is not ApplicationDisposition:
            object.__setattr__(
                self, "disposition", ApplicationDisposition(str(self.disposition))
            )
        if type(self.cleanup_state) is not CleanupState:
            object.__setattr__(
                self, "cleanup_state", CleanupState(str(self.cleanup_state))
            )
        codes = tuple(
            sorted({_text(code, "reason_code") for code in (self.reason_codes or ())})
        )
        if not codes:
            codes = (WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,)
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "applied", self.applied is True)
        # Promotion is a later acceptance step.  This receipt cannot grant it.
        object.__setattr__(self, "promoted", False)
        if self.applied and self.disposition is not ApplicationDisposition.APPLIED:
            object.__setattr__(self, "applied", False)
        if self.disposition is not ApplicationDisposition.APPLIED:
            object.__setattr__(self, "applied", False)
        object.__setattr__(
            self,
            "scope_decision",
            _freeze_mapping(
                dict(self.scope_decision) if self.scope_decision is not None else {}
            ),
        )
        object.__setattr__(
            self,
            "details",
            _freeze_mapping(dict(self.details) if self.details is not None else {}),
        )
        object.__setattr__(
            self,
            "observed_paths",
            tuple(_text(path, "observed_path") for path in self.observed_paths),
        )
        object.__setattr__(
            self,
            "admitted_paths",
            tuple(_text(path, "admitted_path") for path in self.admitted_paths),
        )
        object.__setattr__(
            self, "message", str(self.message or "") if self.message is not None else ""
        )
        digest = self.patch_digest
        if digest and not _CANONICAL_DIGEST_RE.fullmatch(digest):
            object.__setattr__(self, "patch_digest", _sha256_digest(digest))
        observed_digest = self.observed_diff_digest
        if observed_digest and not _CANONICAL_DIGEST_RE.fullmatch(observed_digest):
            object.__setattr__(
                self, "observed_diff_digest", _sha256_digest(observed_digest)
            )

    @property
    def rejected(self) -> bool:
        return self.disposition is ApplicationDisposition.REJECTED

    @property
    def interrupted(self) -> bool:
        return self.disposition is ApplicationDisposition.INTERRUPTED

    @property
    def fenced(self) -> bool:
        return self.disposition is ApplicationDisposition.FENCED

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted_paths": list(self.admitted_paths),
            "applied": self.applied,
            "canonical_branch": self.canonical_branch,
            "cleanup_state": self.cleanup_state.value,
            "details": dict(self.details),
            "disposition": self.disposition.value,
            "fenced": self.fenced,
            "fence": self.fence,
            "interface": self.interface,
            "interrupted": self.interrupted,
            "isolated_branch": self.isolated_branch,
            "lease_id": self.lease_id,
            "lifecycle_state": self.lifecycle_state,
            "message": self.message,
            "observed_diff": self.observed_diff,
            "observed_diff_digest": self.observed_diff_digest,
            "observed_paths": list(self.observed_paths),
            "parent_revision": self.parent_revision,
            "patch_digest": self.patch_digest,
            "promoted": False,
            "proposal_id": self.proposal_id,
            "reason_codes": list(self.reason_codes),
            "rejected": self.rejected,
            "repository_path": self.repository_path,
            "schema": self.schema,
            "scope_decision": dict(self.scope_decision),
            "source_revision": self.source_revision,
            "worktree_parent": self.worktree_parent,
            "worktree_path": self.worktree_path,
        }


# ---------------------------------------------------------------------------
# GuiIsolatedWorktreeExecutor@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiIsolatedWorktreeExecutor:
    """Apply an admitted patch only inside a fenced isolated worktree.

    Interface: ``GuiIsolatedWorktreeExecutor@1``.
    """

    scope_gate: GuiPatchScopeGate = field(default_factory=default_patch_scope_gate)
    git_runner: HostGitRunner = field(default_factory=HostGitRunner)
    host_boundary: GuiHostBoundaryPolicy = field(
        default_factory=GuiHostBoundaryPolicy
    )
    interface: str = GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE
    schema: str = GUI_ISOLATED_WORKTREE_EXECUTOR_SCHEMA

    def __post_init__(self) -> None:
        if type(self.scope_gate) is not GuiPatchScopeGate:
            raise GuiWorktreeExecutorError(
                "scope_gate must be a GuiPatchScopeGate",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                details={"value_type": type(self.scope_gate).__name__},
            )
        if not isinstance(self.git_runner, HostGitRunner):
            raise GuiWorktreeExecutorError(
                "git_runner must be a HostGitRunner",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                details={"value_type": type(self.git_runner).__name__},
            )
        if type(self.host_boundary) is not GuiHostBoundaryPolicy:
            raise GuiWorktreeExecutorError(
                "host_boundary must be a GuiHostBoundaryPolicy",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                details={"value_type": type(self.host_boundary).__name__},
            )
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    def apply(
        self, request: GuiWorktreeApplyRequest | Mapping[str, Any]
    ) -> GuiPatchApplicationReceipt:
        """Apply an admitted proposal in an isolated worktree only."""
        try:
            typed = GuiWorktreeApplyRequest.from_any(request)
        except GuiWorktreeExecutorError as exc:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                exc.reason_code
                or WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                message=str(exc),
                details=dict(exc.details),
            )
        try:
            return self._apply_typed(typed)
        except GuiWorktreeExecutorError as exc:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                exc.reason_code
                or WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                message=str(exc),
                details=dict(exc.details),
                repository_path=typed.repository_path,
                worktree_parent=typed.worktree_parent,
            )

    def apply_request(
        self, request: Mapping[str, Any]
    ) -> GuiPatchApplicationReceipt:
        """Apply a closed JSON request mapping."""
        return self.apply(request)

    def recheck(
        self, receipt: GuiPatchApplicationReceipt
    ) -> GuiPatchApplicationReceipt:
        """Re-evaluate a retained worktree against the admitted scope.

        Extra files that appear after apply fail closed and the disposable
        checkout is removed.  The canonical branch is not touched.
        """
        if type(receipt) is not GuiPatchApplicationReceipt:
            raise GuiWorktreeExecutorError(
                "receipt must be a GuiPatchApplicationReceipt",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            )
        if not receipt.applied or receipt.cleanup_state is not CleanupState.RETAINED:
            return receipt
        worktree = Path(receipt.worktree_path)
        repo = Path(receipt.repository_path)
        if not worktree.is_dir() or not repo.is_dir():
            return self._replace_receipt(
                receipt,
                disposition=ApplicationDisposition.INTERRUPTED,
                extra_codes=(WorktreeExecutorReasonCode.INTERRUPTED.value,),
                applied=False,
                cleanup_state=CleanupState.REMOVE_FAILED,
                message="retained worktree is no longer present for recheck",
            )
        snapshot = self._snapshot_canonical(repo)
        if snapshot.revision != receipt.parent_revision:
            return self._replace_receipt(
                receipt,
                extra_codes=(
                    WorktreeExecutorReasonCode.CANONICAL_MUTATION_DETECTED.value,
                ),
                message="canonical revision changed after apply",
            )
        observed_diff, observed_paths, extra_untracked = self._collect_worktree_diff(
            worktree
        )
        undeclared = tuple(
            path
            for path in (*observed_paths, *extra_untracked)
            if path not in receipt.admitted_paths
        )
        if extra_untracked or undeclared:
            cleaned = self._cleanup_isolated(
                repo,
                worktree,
                receipt.isolated_branch,
                lease_id=receipt.lease_id,
                expected_fence=receipt.fence,
            )
            post = self._snapshot_canonical(repo)
            codes = [
                WorktreeExecutorReasonCode.UNDECLARED_FILE_POST_APPLY.value,
                PatchScopeReasonCode.UNDECLARED_FILE.value,
            ]
            if not snapshot.matches(post):
                codes.append(
                    WorktreeExecutorReasonCode.CANONICAL_MUTATION_DETECTED.value
                )
            return self._replace_receipt(
                receipt,
                disposition=ApplicationDisposition.REJECTED,
                extra_codes=tuple(codes),
                applied=False,
                cleanup_state=cleaned,
                observed_diff=observed_diff,
                observed_paths=tuple(dict.fromkeys((*observed_paths, *extra_untracked))),
                message="post-apply recheck found undeclared files",
                details={
                    **dict(receipt.details),
                    "extra_untracked": list(extra_untracked),
                    "undeclared_paths": list(undeclared),
                },
            )
        return self._replace_receipt(
            receipt,
            observed_diff=observed_diff,
            observed_paths=observed_paths,
        )

    def discard(
        self, receipt: GuiPatchApplicationReceipt
    ) -> GuiPatchApplicationReceipt:
        """Remove a retained isolated worktree without promoting it."""
        if type(receipt) is not GuiPatchApplicationReceipt:
            raise GuiWorktreeExecutorError(
                "receipt must be a GuiPatchApplicationReceipt",
                reason_code=WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
            )
        if receipt.cleanup_state is CleanupState.NEVER_CREATED:
            return receipt
        if receipt.cleanup_state is CleanupState.REMOVED:
            return receipt
        repo = Path(receipt.repository_path)
        worktree = Path(receipt.worktree_path)
        try:
            cleaned = self._cleanup_isolated(
                repo,
                worktree,
                receipt.isolated_branch,
                lease_id=receipt.lease_id,
                expected_fence=receipt.fence,
            )
        except (OwnershipError, FenceMismatchError, WorktreeLifecycleError) as exc:
            return self._replace_receipt(
                receipt,
                extra_codes=(WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value,),
                message=str(exc),
            )
        return self._replace_receipt(
            receipt,
            applied=False if cleaned is CleanupState.REMOVED else receipt.applied,
            cleanup_state=cleaned,
            extra_codes=(WorktreeExecutorReasonCode.CLEANUP_REMOVED.value,)
            if cleaned is CleanupState.REMOVED
            else (),
            message="isolated worktree discarded without promotion",
        )

    # ------------------------------------------------------------------ apply

    def _apply_typed(
        self, request: GuiWorktreeApplyRequest
    ) -> GuiPatchApplicationReceipt:
        if not self.git_runner.available():
            return self._receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.GIT_UNAVAILABLE.value,
                message=f"{HOST_GIT_EXECUTABLE} is required and was not found",
                repository_path=request.repository_path,
                worktree_parent=request.worktree_parent,
            )

        repo = _resolve_host_directory(
            request.repository_path, "repository_path", must_exist=True
        )
        parent = _resolve_host_directory(
            request.worktree_parent, "worktree_parent", must_exist=True
        )
        if parent == repo or _path_is_inside(parent, repo):
            return self._receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value,
                message="worktree_parent must be outside the canonical checkout",
                repository_path=str(repo),
                worktree_parent=str(parent),
            )

        snapshot = self._snapshot_canonical(repo)
        if not snapshot.revision:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
                message="repository does not have a recorded HEAD revision",
                repository_path=str(repo),
                worktree_parent=str(parent),
                canonical_branch=snapshot.branch,
            )
        if request.source_revision and request.source_revision != snapshot.revision:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.SOURCE_REVISION_MISMATCH.value,
                message="declared source_revision does not match repository HEAD",
                repository_path=str(repo),
                worktree_parent=str(parent),
                canonical_branch=snapshot.branch,
                source_revision=request.source_revision,
                parent_revision=snapshot.revision,
                details={
                    "declared_revision": request.source_revision,
                    "head_revision": snapshot.revision,
                },
            )

        try:
            proposal = GuiImprovementProposalView.from_any(request.proposal)
        except GuiPatchScopeError as exc:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                exc.reason_code
                or WorktreeExecutorReasonCode.INVALID_WORKTREE_EXECUTOR_INPUT.value,
                message=str(exc),
                repository_path=str(repo),
                worktree_parent=str(parent),
                canonical_branch=snapshot.branch,
                source_revision=snapshot.revision,
                parent_revision=snapshot.revision,
                details=dict(exc.details),
            )

        try:
            observation = self._observation_from_diff(
                request.diff_text, request.observation
            )
        except GuiPatchScopeError as exc:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                exc.reason_code
                or PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value,
                message=str(exc),
                repository_path=str(repo),
                worktree_parent=str(parent),
                canonical_branch=snapshot.branch,
                source_revision=snapshot.revision,
                parent_revision=snapshot.revision,
                proposal_id=proposal.proposal_id,
                details=dict(exc.details),
            )

        try:
            scope = self.scope_gate.evaluate(
                proposal, observation, invalidation=request.invalidation
            )
        except GuiPatchScopeError as exc:
            return self._receipt(
                ApplicationDisposition.REJECTED,
                exc.reason_code
                or PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
                message=str(exc),
                repository_path=str(repo),
                worktree_parent=str(parent),
                canonical_branch=snapshot.branch,
                source_revision=snapshot.revision,
                parent_revision=snapshot.revision,
                proposal_id=proposal.proposal_id,
                admitted_paths=proposal.intended_file_paths,
                observed_paths=observation.observed_paths,
                details=dict(exc.details),
            )

        admitted_paths = observation.observed_paths
        if not scope.allowed:
            reason = (
                WorktreeExecutorReasonCode.SCOPE_REQUIRES_REVIEW.value
                if scope.requires_human_review
                else WorktreeExecutorReasonCode.SCOPE_REJECTED.value
            )
            return self._invariant_receipt(
                ApplicationDisposition.REJECTED,
                reason,
                *scope.reason_codes,
                WorktreeExecutorReasonCode.CLEANUP_NEVER_CREATED.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                observed_paths=observation.observed_paths,
                scope=scope,
                cleanup_state=CleanupState.NEVER_CREATED,
                patch_digest=_sha256_digest(request.diff_text),
                message=scope.message or "proposal was not admitted for execution",
            )

        task_id = request.task_id or f"vgo-{_safe_token(proposal.proposal_id)}"
        attempt = request.attempt
        isolated_branch = (
            f"{ISOLATED_BRANCH_PREFIX}{_safe_token(task_id)}/{int(attempt)}"
        )
        if isolated_branch == snapshot.branch or not _ISOLATED_BRANCH_RE.fullmatch(
            isolated_branch
        ):
            return self._invariant_receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.COMMAND_STRING_FORBIDDEN.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=CleanupState.NEVER_CREATED,
                message="refusing to use the canonical branch as an isolated branch",
            )

        lease_id = request.lease_id or new_lease_id(
            seed=f"{task_id}:{attempt}:{snapshot.revision}"
        )
        worktree = parent / (
            f"vgo-{_safe_token(task_id)}-a{int(attempt)}-{lease_id[:12]}"
        )
        if not _path_is_inside(worktree, parent) or worktree.exists():
            return self._invariant_receipt(
                ApplicationDisposition.REJECTED,
                WorktreeExecutorReasonCode.WORKTREE_PARENT_INVALID.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                worktree_path=str(worktree),
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=CleanupState.NEVER_CREATED,
                message="isolated worktree path is not a fresh child of worktree_parent",
            )

        store = self._lifecycle_store(repo)
        try:
            record = store.begin_preparing(
                task_id=task_id,
                canonical_task_cid=request.canonical_task_cid
                or proposal.proposal_id,
                attempt=attempt,
                lane_id=request.lane_id,
                workspace_path=worktree,
                branch=isolated_branch,
                merge_target=snapshot.branch,
                lease_id=lease_id,
            )
        except DuplicateAttemptError as exc:
            return self._invariant_receipt(
                ApplicationDisposition.FENCED,
                WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                isolated_branch=isolated_branch,
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=CleanupState.NEVER_CREATED,
                lease_id=lease_id,
                message=str(exc),
            )
        except WorktreeLifecycleError as exc:
            return self._invariant_receipt(
                ApplicationDisposition.FENCED,
                WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                isolated_branch=isolated_branch,
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=CleanupState.NEVER_CREATED,
                lease_id=lease_id,
                message=str(exc),
            )

        created = False
        try:
            add_result = self.git_runner.run(
                (
                    "worktree",
                    "add",
                    "-b",
                    isolated_branch,
                    str(worktree),
                    snapshot.revision,
                ),
                cwd=repo,
            )
            if not add_result.ok:
                raise _Interrupt(add_result.output or "git worktree add failed")
            created = True
            record = store.mark_active(
                worktree,
                lease_id=record.lease_id,
                expected_fence=record.fence,
            )
            check = self.git_runner.run(
                ("apply", "--check", "--whitespace=nowarn"),
                cwd=worktree,
                input_text=request.diff_text,
            )
            if not check.ok:
                raise _Interrupt(check.output or "git apply --check failed")
            applied = self.git_runner.run(
                ("apply", "--index", "--whitespace=nowarn"),
                cwd=worktree,
                input_text=request.diff_text,
            )
            if not applied.ok:
                raise _Interrupt(applied.output or "git apply --index failed")

            observed_diff, observed_paths, extra_untracked = (
                self._collect_worktree_diff(worktree)
            )
            post_observation = PatchScopeObservation(
                hunks=parse_unified_diff(observed_diff) if observed_diff.strip() else observation.hunks,
                touched_component_ids=observation.touched_component_ids,
                touched_state_effect_ids=observation.touched_state_effect_ids,
                touched_test_ids=observation.touched_test_ids,
                touched_screenshot_ids=observation.touched_screenshot_ids,
                application_ids=observation.application_ids,
                action_binding_ids=observation.action_binding_ids,
                action_argument_digest=observation.action_argument_digest,
                action_contract_evidence=observation.action_contract_evidence,
                visual_effect_observed=observation.visual_effect_observed,
                unresolved_paths=observation.unresolved_paths,
            )
            post_scope = self.scope_gate.evaluate(
                proposal, post_observation, invalidation=request.invalidation
            )
            undeclared = tuple(
                path
                for path in (*observed_paths, *extra_untracked)
                if path not in admitted_paths
            )
            path_mismatch = tuple(observed_paths) != tuple(admitted_paths)
            if (
                extra_untracked
                or undeclared
                or path_mismatch
                or not post_scope.allowed
            ):
                cleaned = self._cleanup_isolated(
                    repo,
                    worktree,
                    isolated_branch,
                    lease_id=record.lease_id,
                    expected_fence=record.fence,
                    store=store,
                )
                codes = [
                    WorktreeExecutorReasonCode.UNDECLARED_FILE_POST_APPLY.value
                    if extra_untracked or undeclared
                    else WorktreeExecutorReasonCode.DIFF_SCOPE_MISMATCH.value,
                    WorktreeExecutorReasonCode.SCOPE_REJECTED.value,
                ]
                codes.extend(post_scope.reason_codes)
                return self._finish(
                    ApplicationDisposition.REJECTED,
                    *codes,
                    snapshot=snapshot,
                    worktree_parent=str(parent),
                    worktree_path=str(worktree),
                    isolated_branch=isolated_branch,
                    proposal_id=proposal.proposal_id,
                    admitted_paths=admitted_paths,
                    observed_paths=tuple(
                        dict.fromkeys((*observed_paths, *extra_untracked))
                    ),
                    observed_diff=observed_diff,
                    scope=post_scope,
                    cleanup_state=cleaned,
                    lease_id=record.lease_id,
                    fence=record.fence,
                    lifecycle_state=WorkspaceLifecycleState.TERMINAL.value,
                    patch_digest=_sha256_digest(request.diff_text),
                    message="post-apply recheck rejected the worktree",
                    details={
                        "extra_untracked": list(extra_untracked),
                        "undeclared_paths": list(undeclared),
                        "path_mismatch": path_mismatch,
                    },
                )

            record = store.mark_settling(
                worktree,
                lease_id=record.lease_id,
                expected_fence=record.fence,
            )
            return self._finish(
                ApplicationDisposition.APPLIED,
                WorktreeExecutorReasonCode.APPLIED.value,
                WorktreeExecutorReasonCode.NOT_PROMOTED.value,
                WorktreeExecutorReasonCode.CLEANUP_RETAINED.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                worktree_path=str(worktree),
                isolated_branch=isolated_branch,
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                observed_paths=observed_paths,
                observed_diff=observed_diff,
                scope=post_scope,
                cleanup_state=CleanupState.RETAINED,
                lease_id=record.lease_id,
                fence=record.fence,
                lifecycle_state=record.state.value,
                patch_digest=_sha256_digest(request.diff_text),
                message="patch applied in an isolated worktree and was not promoted",
                details={
                    "task_id": task_id,
                    "attempt": attempt,
                    "lane_id": request.lane_id,
                    "proposal_interface": GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
                },
            )
        except _Interrupt as exc:
            cleaned = (
                self._cleanup_isolated(
                    repo,
                    worktree,
                    isolated_branch,
                    lease_id=record.lease_id,
                    expected_fence=record.fence,
                    store=store,
                )
                if created or worktree.exists()
                else CleanupState.NEVER_CREATED
            )
            if not created:
                self._terminalize(
                    store, worktree, record, reason="worktree_add_failed"
                )
            return self._finish(
                ApplicationDisposition.INTERRUPTED,
                WorktreeExecutorReasonCode.INTERRUPTED.value,
                WorktreeExecutorReasonCode.GIT_OPERATION_FAILED.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                worktree_path=str(worktree) if created else "",
                isolated_branch=isolated_branch,
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=cleaned,
                lease_id=record.lease_id,
                fence=record.fence,
                lifecycle_state=WorkspaceLifecycleState.TERMINAL.value,
                patch_digest=_sha256_digest(request.diff_text),
                message=str(exc),
            )
        except (OwnershipError, FenceMismatchError, WorktreeLifecycleError) as exc:
            cleaned = CleanupState.NEVER_CREATED
            if created:
                cleaned = self._cleanup_isolated(
                    repo,
                    worktree,
                    isolated_branch,
                    lease_id=record.lease_id,
                    expected_fence=record.fence,
                    store=store,
                    ignore_ownership=True,
                )
            return self._finish(
                ApplicationDisposition.FENCED,
                WorktreeExecutorReasonCode.LEASE_FENCE_FAILURE.value,
                snapshot=snapshot,
                worktree_parent=str(parent),
                worktree_path=str(worktree) if created else "",
                isolated_branch=isolated_branch,
                proposal_id=proposal.proposal_id,
                admitted_paths=admitted_paths,
                scope=scope,
                cleanup_state=cleaned,
                lease_id=record.lease_id,
                fence=record.fence,
                message=str(exc),
            )

    # ---------------------------------------------------------------- helpers

    def _observation_from_diff(
        self, diff_text: str, caller_observation: Any
    ) -> PatchScopeObservation:
        hunks = parse_unified_diff(diff_text)
        if caller_observation is None:
            return PatchScopeObservation(hunks=hunks)
        if type(caller_observation) is PatchScopeObservation:
            return PatchScopeObservation(
                hunks=hunks,
                touched_component_ids=caller_observation.touched_component_ids,
                touched_state_effect_ids=caller_observation.touched_state_effect_ids,
                touched_test_ids=caller_observation.touched_test_ids,
                touched_screenshot_ids=caller_observation.touched_screenshot_ids,
                application_ids=caller_observation.application_ids,
                action_binding_ids=caller_observation.action_binding_ids,
                action_argument_digest=caller_observation.action_argument_digest,
                action_contract_evidence=caller_observation.action_contract_evidence,
                visual_effect_observed=caller_observation.visual_effect_observed,
                unresolved_paths=caller_observation.unresolved_paths,
            )
        payload = _require_mapping(caller_observation, "observation")
        # Caller hunks never override the parsed diff; only metadata is reused.
        metadata_payload = {
            key: payload[key]
            for key in _OBSERVATION_METADATA_KEYS
            if key in payload
        }
        if metadata_payload:
            metadata_payload["hunks"] = [
                {
                    "path": hunk.path,
                    "operation": hunk.operation.value,
                    "added_lines": hunk.added_lines,
                    "deleted_lines": hunk.deleted_lines,
                    "old_path": hunk.old_path,
                    "start_line": hunk.start_line,
                    "end_line": hunk.end_line,
                    "diff_text": hunk.diff_text,
                    "change_kinds": [kind.value for kind in hunk.change_kinds],
                    "content_markers": list(hunk.content_markers),
                }
                for hunk in hunks
            ]
            return PatchScopeObservation.from_mapping(metadata_payload)
        return PatchScopeObservation(hunks=hunks)

    def _snapshot_canonical(self, repo: Path) -> CanonicalCheckoutSnapshot:
        inside = self.git_runner.run(("rev-parse", "--is-inside-work-tree"), cwd=repo)
        if not inside.ok or inside.stdout.strip() != "true":
            raise GuiWorktreeExecutorError(
                "repository_path is not a Git work tree",
                reason_code=WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
                details={"repository_path": str(repo), "output": inside.output},
            )
        toplevel = self.git_runner.run(("rev-parse", "--show-toplevel"), cwd=repo)
        if not toplevel.ok:
            raise GuiWorktreeExecutorError(
                "repository toplevel could not be resolved",
                reason_code=WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
                details={"output": toplevel.output},
            )
        top = Path(toplevel.stdout.strip()).resolve(strict=False)
        if top != repo.resolve(strict=False):
            raise GuiWorktreeExecutorError(
                "repository_path must be the Git toplevel, not a subdirectory",
                reason_code=WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
                details={
                    "repository_path": str(repo),
                    "toplevel": str(top),
                },
            )
        revision = self.git_runner.run(("rev-parse", "--verify", "HEAD"), cwd=repo)
        if not revision.ok or not _FULL_SHA_RE.fullmatch(revision.stdout.strip()):
            raise GuiWorktreeExecutorError(
                "repository HEAD revision is missing",
                reason_code=WorktreeExecutorReasonCode.MISSING_SOURCE_REVISION.value,
                details={"output": revision.output},
            )
        branch_result = self.git_runner.run(
            ("rev-parse", "--abbrev-ref", "HEAD"), cwd=repo
        )
        branch = branch_result.stdout.strip() if branch_result.ok else ""
        if not branch or branch == "HEAD":
            symbolic = self.git_runner.run(("symbolic-ref", "--short", "HEAD"), cwd=repo)
            branch = symbolic.stdout.strip() if symbolic.ok else ""
        if not branch:
            raise GuiWorktreeExecutorError(
                "canonical branch could not be recorded",
                reason_code=WorktreeExecutorReasonCode.REPOSITORY_INVALID.value,
            )
        status = self.git_runner.run(
            ("status", "--porcelain=v1", "-uall"), cwd=repo
        )
        return CanonicalCheckoutSnapshot(
            repository_path=str(repo.resolve(strict=False)),
            branch=branch,
            revision=revision.stdout.strip(),
            porcelain=status.stdout if status.ok else status.output,
        )

    def _collect_worktree_diff(
        self, worktree: Path
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        diff = self.git_runner.run(
            (
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--no-textconv",
                "HEAD",
            ),
            cwd=worktree,
        )
        cached = self.git_runner.run(
            (
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-color",
                "--no-textconv",
            ),
            cwd=worktree,
        )
        observed_text = diff.stdout if diff.ok and diff.stdout.strip() else cached.stdout
        paths: list[str] = []
        if observed_text.strip():
            hunks = parse_unified_diff(observed_text)
            seen: set[str] = set()
            for hunk in hunks:
                for path in hunk.observed_paths:
                    if path not in seen:
                        seen.add(path)
                        paths.append(path)
        status = self.git_runner.run(
            ("status", "--porcelain=v1", "-uall"), cwd=worktree
        )
        extra: list[str] = []
        if status.ok:
            for raw_line in status.stdout.splitlines():
                if not raw_line:
                    continue
                if raw_line.startswith("??"):
                    extra.append(raw_line[3:].strip())
        return observed_text, tuple(paths), tuple(extra)

    def _lifecycle_store(self, repo: Path) -> WorktreeLifecycleStore:
        return WorktreeLifecycleStore(repo_root=repo)

    def _cleanup_isolated(
        self,
        repo: Path,
        worktree: Path,
        isolated_branch: str,
        *,
        lease_id: str,
        expected_fence: int,
        store: WorktreeLifecycleStore | None = None,
        ignore_ownership: bool = False,
    ) -> CleanupState:
        lifecycle = store or self._lifecycle_store(repo)
        record = lifecycle.load_workspace(worktree)
        if record is not None and not ignore_ownership:
            if str(lease_id) != str(record.lease_id):
                raise OwnershipError("lifecycle lease does not match record owner")
            if int(expected_fence) != int(record.fence):
                raise FenceMismatchError(
                    f"expected fence {expected_fence}, found {record.fence}"
                )
            decision = lifecycle.evaluate_cleanup(
                workspace_path=worktree, caller_lease_id=lease_id
            )
            if not decision.allowed:
                raise OwnershipError(decision.reason)
        removed_ok = True
        if worktree.exists():
            remove = self.git_runner.run(
                ("worktree", "remove", "--force", str(worktree)),
                cwd=repo,
            )
            if not remove.ok and worktree.exists():
                try:
                    shutil.rmtree(worktree)
                except OSError:
                    removed_ok = False
        if _ISOLATED_BRANCH_RE.fullmatch(isolated_branch):
            self.git_runner.run(("branch", "-D", isolated_branch), cwd=repo)
        if record is not None:
            self._terminalize(
                lifecycle, worktree, record, reason="isolated_cleanup"
            )
        if worktree.exists():
            return CleanupState.REMOVE_FAILED
        return CleanupState.REMOVED if removed_ok else CleanupState.REMOVE_FAILED

    def _terminalize(
        self,
        store: WorktreeLifecycleStore,
        worktree: Path,
        record: WorkspaceLifecycleRecord,
        *,
        reason: str,
    ) -> None:
        try:
            store.mark_terminal(
                worktree,
                lease_id=record.lease_id,
                expected_fence=record.fence,
                reason=reason,
            )
        except (OwnershipError, FenceMismatchError, WorktreeLifecycleError):
            current = store.load_workspace(worktree)
            if current is None or current.is_terminal:
                return
            try:
                store.mark_terminal(
                    worktree,
                    lease_id=current.lease_id,
                    expected_fence=current.fence,
                    reason=reason,
                )
            except (OwnershipError, FenceMismatchError, WorktreeLifecycleError):
                return

    def _receipt(
        self,
        disposition: ApplicationDisposition,
        *reason_codes: str,
        message: str = "",
        details: Mapping[str, Any] | None = None,
        repository_path: str = "",
        worktree_path: str = "",
        worktree_parent: str = "",
        isolated_branch: str = "",
        canonical_branch: str = "",
        source_revision: str = "",
        parent_revision: str = "",
        observed_diff: str = "",
        observed_paths: Sequence[str] = (),
        admitted_paths: Sequence[str] = (),
        cleanup_state: CleanupState = CleanupState.NEVER_CREATED,
        lease_id: str = "",
        fence: int = 0,
        lifecycle_state: str = "",
        proposal_id: str = "",
        patch_digest: str = "",
        scope: GuiPatchScopeDecision | None = None,
    ) -> GuiPatchApplicationReceipt:
        return GuiPatchApplicationReceipt(
            disposition=disposition,
            reason_codes=reason_codes,
            applied=disposition is ApplicationDisposition.APPLIED,
            promoted=False,
            repository_path=repository_path,
            worktree_path=worktree_path,
            worktree_parent=worktree_parent,
            isolated_branch=isolated_branch,
            canonical_branch=canonical_branch,
            source_revision=source_revision or parent_revision,
            parent_revision=parent_revision,
            observed_diff=observed_diff,
            observed_paths=tuple(observed_paths),
            admitted_paths=tuple(admitted_paths),
            cleanup_state=cleanup_state,
            lease_id=lease_id,
            fence=fence,
            lifecycle_state=lifecycle_state,
            proposal_id=proposal_id,
            patch_digest=patch_digest,
            observed_diff_digest=_sha256_digest(observed_diff) if observed_diff else "",
            scope_decision=scope.to_dict() if scope is not None else {},
            message=message,
            details=details or {},
        )

    def _invariant_receipt(
        self,
        disposition: ApplicationDisposition,
        *reason_codes: str,
        snapshot: CanonicalCheckoutSnapshot,
        **kwargs: Any,
    ) -> GuiPatchApplicationReceipt:
        post = self._snapshot_canonical(Path(snapshot.repository_path))
        codes = list(reason_codes)
        if not snapshot.matches(post):
            codes.append(WorktreeExecutorReasonCode.CANONICAL_MUTATION_DETECTED.value)
        return self._receipt(
            disposition,
            *codes,
            repository_path=snapshot.repository_path,
            canonical_branch=snapshot.branch,
            source_revision=snapshot.revision,
            parent_revision=snapshot.revision,
            **kwargs,
        )

    def _finish(
        self,
        disposition: ApplicationDisposition,
        *reason_codes: str,
        snapshot: CanonicalCheckoutSnapshot,
        **kwargs: Any,
    ) -> GuiPatchApplicationReceipt:
        return self._invariant_receipt(
            disposition, *reason_codes, snapshot=snapshot, **kwargs
        )

    def _replace_receipt(
        self,
        receipt: GuiPatchApplicationReceipt,
        *,
        disposition: ApplicationDisposition | None = None,
        extra_codes: Sequence[str] = (),
        applied: bool | None = None,
        cleanup_state: CleanupState | None = None,
        observed_diff: str | None = None,
        observed_paths: Sequence[str] | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> GuiPatchApplicationReceipt:
        codes = list(receipt.reason_codes)
        codes.extend(extra_codes)
        new_diff = receipt.observed_diff if observed_diff is None else observed_diff
        return GuiPatchApplicationReceipt(
            disposition=disposition or receipt.disposition,
            reason_codes=tuple(codes),
            applied=receipt.applied if applied is None else applied,
            promoted=False,
            repository_path=receipt.repository_path,
            worktree_path=receipt.worktree_path,
            worktree_parent=receipt.worktree_parent,
            isolated_branch=receipt.isolated_branch,
            canonical_branch=receipt.canonical_branch,
            source_revision=receipt.source_revision,
            parent_revision=receipt.parent_revision,
            observed_diff=new_diff,
            observed_paths=(
                receipt.observed_paths
                if observed_paths is None
                else tuple(observed_paths)
            ),
            admitted_paths=receipt.admitted_paths,
            cleanup_state=cleanup_state or receipt.cleanup_state,
            lease_id=receipt.lease_id,
            fence=receipt.fence,
            lifecycle_state=receipt.lifecycle_state,
            proposal_id=receipt.proposal_id,
            patch_digest=receipt.patch_digest,
            observed_diff_digest=_sha256_digest(new_diff) if new_diff else "",
            scope_decision=dict(receipt.scope_decision),
            message=receipt.message if message is None else message,
            details=dict(receipt.details) if details is None else dict(details),
        )


class _Interrupt(RuntimeError):
    """Internal control-flow for a failed apply after a worktree exists."""


def default_isolated_worktree_executor() -> GuiIsolatedWorktreeExecutor:
    """Return a fail-closed executor with default scope and host git."""
    return GuiIsolatedWorktreeExecutor()


__all__ = (
    "ALLOWED_GIT_VERBS",
    "ApplicationDisposition",
    "BROAD_ROOTS",
    "CleanupState",
    "FORBIDDEN_GIT_VERBS",
    "GUI_ISOLATED_WORKTREE_EXECUTOR_INTERFACE",
    "GUI_ISOLATED_WORKTREE_EXECUTOR_SCHEMA",
    "GUI_PATCH_APPLICATION_RECEIPT_INTERFACE",
    "GUI_PATCH_APPLICATION_RECEIPT_SCHEMA",
    "GuiIsolatedWorktreeExecutor",
    "GuiPatchApplicationReceipt",
    "GuiWorktreeApplyRequest",
    "GuiWorktreeExecutorError",
    "HOST_GIT_EXECUTABLE",
    "HostGitResult",
    "HostGitRunner",
    "ISOLATED_BRANCH_PREFIX",
    "WorktreeExecutorReasonCode",
    "default_isolated_worktree_executor",
    "sealed_git_environment",
)
