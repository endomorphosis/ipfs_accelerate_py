"""Safe fenced worktree creation and bounded patch validation.

Interface: ``IsolatedPatchWorktree@1``

This module composes the existing supervisor authorities:

* ``WorktreeLifecycleStore`` for durable prepare/active/settling/terminal
  ownership and peer-cleanup fencing;
* ``todo_daemon.worktrees`` helpers for path normalization and (when useful)
  managed session bookkeeping patterns;
* ``validation.proposal_validation.parse_unified_patch`` for fail-closed
  unified-diff admission (no binary, symlink, gitlink, or traversal);
* optional production-context preimage coverage when a visibility map is
  supplied.

It does **not** invent a second worktree or proposal authority. Mutations are
confined to a disposable detached worktree. The caller's repository root is
never written by validation or apply. Stale bases, invisible preimages,
malformed or out-of-scope patches, and failed ``git apply --check`` leave both
the target root and the worktree tree unchanged. Concurrent or stale owners
cannot publish results or clean a live peer worktree. Interrupted
prepare/apply/cleanup phases recover through durable lifecycle records and an
attempt journal.

Cold import is side-effect free: no Git, threads, processes, databases, or
network calls run at import time.
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
from typing import Any, Final

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
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    LeaseBinding,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ParsedPatchFile,
    ProposalValidationError,
    parse_unified_patch,
)

ISOLATED_PATCH_WORKTREE_INTERFACE: Final[str] = "IsolatedPatchWorktree@1"
WORKTREE_ADAPTER_SCHEMA: Final[str] = "semantic-state-isolated-worktree@1"
ADAPTER_ID: Final[str] = "semantic-isolated-patch-worktree"
ATTEMPT_JOURNAL_SCHEMA: Final[str] = "semantic-state-worktree-attempt@1"

_MAX_PATCH_BYTES: Final[int] = 2_000_000
_MAX_PATCH_FILES: Final[int] = 256
_MAX_PATH_BYTES: Final[int] = 512
_MAX_DIAGNOSTIC: Final[int] = 512
_GIT_TIMEOUT_SECONDS: Final[int] = 60
_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")

# Control, runtime, and supervisor state paths are never writable by a patch.
_DEFAULT_FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    ".git/",
    ".git",
    ".agent_supervisor/",
    "data/agent_supervisor/",
    "__pycache__/",
    ".env",
    ".ssh/",
    ".aws/",
)

_DEFAULT_PROTECTED_PREFIXES: Final[tuple[str, ...]] = (
    "docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md",
    "docs/architecture/semantic_compression_harness.objectives.md",
    "docs/architecture/semantic_compression_harness.todo.md",
    "config/semantic_state_dependencies.seal.json",
    "scripts/validate_semantic_state_dependencies.py",
    "test/api/semantic_state/test_dependency_seal.py",
)


class PatchValidationError(HarnessError):
    """A patch failed closed admission or apply-check."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "patch_rejected",
        path: str = "",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "patch_rejected")
        self.path = str(path or "")


class WorktreeFenceError(HarnessError):
    """Ownership or fence token does not authorize the requested mutation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "fence_mismatch",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "fence_mismatch")


class WorktreePhase(str, Enum):
    """Durable phases for one isolated attempt worktree."""

    PREPARING = "preparing"
    READY = "ready"
    VALIDATING = "validating"
    APPLYING = "applying"
    APPLIED = "applied"
    REJECTED = "rejected"
    CLEANING = "cleaning"
    TERMINAL = "terminal"

    @property
    def is_terminal(self) -> bool:
        return self is WorktreePhase.TERMINAL

    @property
    def is_mutable(self) -> bool:
        return self in {
            WorktreePhase.READY,
            WorktreePhase.VALIDATING,
            WorktreePhase.APPLYING,
            WorktreePhase.APPLIED,
            WorktreePhase.REJECTED,
        }


def _clip(text: str, *, limit: int = _MAX_DIAGNOSTIC) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


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


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    timeout_seconds: int = _GIT_TIMEOUT_SECONDS,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a fixed local Git argv under a scrubbed minimal environment."""

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        completed = subprocess.run(
            ["git", *list(args)],
            cwd=str(cwd),
            input=stdin,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise PatchValidationError(
            f"git {' '.join(args[:3])} timed out",
            reason_code="git_timeout",
        ) from exc
    except OSError as exc:
        raise PatchValidationError(
            f"git invocation failed: {exc}",
            reason_code="git_unavailable",
        ) from exc
    if check and completed.returncode != 0:
        raise PatchValidationError(
            _clip(completed.stderr or completed.stdout or "git command failed"),
            reason_code="git_failed",
        )
    return completed


def _rev_parse(repo: Path, rev: str) -> str:
    result = _run_git(["rev-parse", "--verify", rev], cwd=repo)
    if result.returncode != 0:
        raise PatchValidationError(
            f"cannot resolve revision {rev!r}",
            reason_code="stale_base",
        )
    value = (result.stdout or "").strip()
    if not _OID_RE.fullmatch(value):
        raise PatchValidationError(
            f"malformed object id for {rev!r}",
            reason_code="stale_base",
        )
    return value


def _tree_for_commit(repo: Path, commit: str) -> str:
    return _rev_parse(repo, f"{commit}^{{tree}}")


def _write_tree(repo: Path) -> str:
    """Return the index tree OID (does not create a commit)."""

    result = _run_git(["write-tree"], cwd=repo)
    if result.returncode != 0:
        raise PatchValidationError(
            _clip(result.stderr or "write-tree failed"),
            reason_code="tree_capture_failed",
        )
    value = (result.stdout or "").strip()
    if not _OID_RE.fullmatch(value):
        raise PatchValidationError(
            "write-tree returned a malformed oid",
            reason_code="tree_capture_failed",
        )
    return value


def _head_commit(repo: Path) -> str:
    return _rev_parse(repo, "HEAD")


def _normalize_repo_path(value: str, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PatchValidationError(f"{name} is not a repository path", path=str(value))
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
        or len(raw.encode("utf-8")) > _MAX_PATH_BYTES
    ):
        raise PatchValidationError(
            f"{name} escapes the repository: {value!r}",
            reason_code="unsafe_path",
            path=raw,
        )
    return pure.as_posix()


def _path_matches_prefix(path: str, pattern: str) -> bool:
    """Return True when ``path`` equals or is under a prefix/glob-ish pattern."""

    if not pattern:
        return False
    if pattern.endswith("/**"):
        root = pattern[:-3].rstrip("/")
        return path == root or path.startswith(root + "/")
    if pattern.endswith("/"):
        return path == pattern[:-1] or path.startswith(pattern)
    if "*" in pattern or "?" in pattern or "[" in pattern:
        # Conservative glob: only '*' segment wildcards at ends.
        import fnmatch

        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            path, pattern.rstrip("/") + "/**"
        )
    return path == pattern or path.startswith(pattern.rstrip("/") + "/")


def _is_under_any(path: str, patterns: Sequence[str]) -> bool:
    return any(_path_matches_prefix(path, pattern) for pattern in patterns)


@dataclass(frozen=True)
class PatchScope:
    """Immutable write-scope for one patch admission.

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
    max_files: int = _MAX_PATCH_FILES
    max_bytes: int = _MAX_PATCH_BYTES
    allow_binary: bool = False

    def __post_init__(self) -> None:
        allowed = tuple(
            dict.fromkeys(
                _normalize_repo_path(item, name="allowed_paths")
                if not str(item).endswith(("/", "*"))
                and "*" not in str(item)
                and not str(item).endswith("/")
                else str(item).replace("\\", "/").strip()
                for item in self.allowed_paths
            )
        )
        if not allowed:
            raise PatchValidationError(
                "allowed_paths must not be empty",
                reason_code="invalid_scope",
            )
        effect = tuple(
            dict.fromkeys(
                _normalize_repo_path(item, name="effect_paths")
                for item in self.effect_paths
            )
        )
        owned = tuple(
            dict.fromkeys(
                (
                    _normalize_repo_path(item, name="task_owned_paths")
                    if not str(item).endswith(("/", "*"))
                    and "*" not in str(item)
                    and not str(item).endswith("/")
                    else str(item).replace("\\", "/").strip()
                )
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
        if type(self.allow_binary) is not bool:
            raise PatchValidationError(
                "allow_binary must be boolean", reason_code="invalid_scope"
            )
        if type(self.max_files) is not int or self.max_files < 1:
            raise PatchValidationError(
                "max_files must be a positive integer", reason_code="invalid_scope"
            )
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise PatchValidationError(
                "max_bytes must be a positive integer", reason_code="invalid_scope"
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
        except PatchValidationError:
            return False, "unsafe_path"
        if _is_under_any(normalized, self.forbidden_paths):
            return False, "forbidden_path"
        if _is_under_any(normalized, self.protected_paths):
            return False, "protected_path"
        if not _is_under_any(normalized, self.task_owned_paths):
            return False, "outside_task_owned"
        if not _is_under_any(normalized, self.allowed_paths):
            return False, "outside_allowlist"
        if self.effect_paths and normalized not in self.effect_paths:
            # effect_paths are exact declared write targets
            if not _is_under_any(normalized, self.effect_paths):
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
            "allow_binary": bool(self.allow_binary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatchScope":
        if not isinstance(data, Mapping):
            raise PatchValidationError(
                "PatchScope payload must be an object", reason_code="invalid_scope"
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
            max_files=int(data.get("max_files") or _MAX_PATCH_FILES),
            max_bytes=int(data.get("max_bytes") or _MAX_PATCH_BYTES),
            allow_binary=bool(data.get("allow_binary", False)),
        )


@dataclass(frozen=True)
class PatchValidationResult:
    """Bounded, serializable outcome of patch admission (pre-apply)."""

    accepted: bool
    reason_codes: tuple[str, ...]
    paths: tuple[str, ...]
    operations: tuple[str, ...]
    parsed_files: tuple[ParsedPatchFile, ...] = ()
    pre_tree: str = ""
    diagnostic: str = ""
    patch_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKTREE_ADAPTER_SCHEMA,
            "accepted": bool(self.accepted),
            "reason_codes": list(self.reason_codes),
            "paths": list(self.paths),
            "operations": list(self.operations),
            "parsed_files": [
                {
                    "old_path": item.old_path,
                    "new_path": item.new_path,
                    "operation": item.operation,
                    "additions": int(item.additions),
                    "deletions": int(item.deletions),
                    "binary": bool(item.binary),
                }
                for item in self.parsed_files
            ],
            "pre_tree": self.pre_tree,
            "diagnostic": self.diagnostic,
            "patch_digest": self.patch_digest,
        }


@dataclass(frozen=True)
class PatchApplyResult:
    """Outcome of a checked apply inside an isolated worktree."""

    applied: bool
    validation: PatchValidationResult
    pre_tree: str
    post_tree: str
    base_commit: str
    worktree_path: str
    reason_codes: tuple[str, ...] = ()
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKTREE_ADAPTER_SCHEMA,
            "applied": bool(self.applied),
            "validation": self.validation.to_dict(),
            "pre_tree": self.pre_tree,
            "post_tree": self.post_tree,
            "base_commit": self.base_commit,
            "worktree_path": self.worktree_path,
            "reason_codes": list(self.reason_codes),
            "diagnostic": self.diagnostic,
        }


def _patch_digest(patch_text: str) -> str:
    return "sha256:" + hashlib.sha256(
        patch_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _paths_from_parsed(files: Sequence[ParsedPatchFile]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in files:
        for candidate in (item.old_path, item.new_path):
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(sorted(ordered))


def _extract_preimage_hunks(
    patch_text: str,
) -> dict[str, list[tuple[int, tuple[str, ...]]]]:
    """Map path -> list of (old_start, context/minus body lines).

    Used only for visibility coverage. Malformed patches are rejected earlier
    by ``parse_unified_patch``.
    """

    lines = patch_text.splitlines()
    result: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    index = 0
    current_old = ""
    current_new = ""
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            current_old = ""
            current_new = ""
            index += 1
            continue
        if line.startswith("--- "):
            try:
                current_old = _normalize_repo_path(
                    line[4:].strip().removeprefix("a/").strip('"')
                    if line[4:].strip() not in {"/dev/null", '"/dev/null"'}
                    else ""
                ) if line[4:].strip() not in {"/dev/null", '"/dev/null"'} else ""
            except PatchValidationError:
                current_old = ""
            index += 1
            continue
        if line.startswith("+++ "):
            try:
                current_new = _normalize_repo_path(
                    line[4:].strip().removeprefix("b/").strip('"')
                    if line[4:].strip() not in {"/dev/null", '"/dev/null"'}
                    else ""
                ) if line[4:].strip() not in {"/dev/null", '"/dev/null"'} else ""
            except PatchValidationError:
                current_new = ""
            index += 1
            continue
        match = re.match(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",
            line,
        )
        if match is None:
            index += 1
            continue
        old_start = int(match.group(1))
        body: list[str] = []
        index += 1
        while index < len(lines):
            body_line = lines[index]
            if body_line.startswith(("diff --git ", "@@ ")):
                break
            if body_line == r"\ No newline at end of file":
                index += 1
                continue
            if body_line[:1] in {" ", "-"}:
                body.append(body_line[1:])
            index += 1
        target = current_old or current_new
        if target and body:
            result.setdefault(target, []).append((old_start, tuple(body)))
    return result


def _preimage_visible(
    *,
    path: str,
    hunks: Sequence[tuple[int, tuple[str, ...]]],
    visible_sources: Mapping[str, str | bytes | None],
) -> tuple[bool, str]:
    if path not in visible_sources:
        return False, "invisible_preimage"
    content = visible_sources[path]
    if content is None:
        return False, "invisible_preimage"
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeError:
            return False, "invisible_preimage"
    else:
        text = str(content)
    # Full-file visibility: every preimage line sequence must appear contiguously.
    file_lines = text.splitlines()
    for old_start, body in hunks:
        if not body:
            continue
        # Prefer exact line-number alignment when the provider saw the full file.
        zero_based = max(0, old_start - 1)
        window = file_lines[zero_based : zero_based + len(body)]
        if tuple(window) == body:
            continue
        # Fall back to whole-file search for slices that are fully visible but
        # not line-numbered against the original file.
        matched = False
        if len(body) <= len(file_lines):
            for offset in range(0, len(file_lines) - len(body) + 1):
                if tuple(file_lines[offset : offset + len(body)]) == body:
                    matched = True
                    break
        if not matched:
            return False, "invisible_preimage"
    return True, ""


class PatchValidator:
    """Fail-closed unified-diff admission against an immutable ``PatchScope``."""

    def __init__(self, scope: PatchScope | Mapping[str, Any]) -> None:
        self.scope = scope if isinstance(scope, PatchScope) else PatchScope.from_dict(scope)

    def parse(self, patch_text: str) -> tuple[ParsedPatchFile, ...]:
        if not isinstance(patch_text, str) or not patch_text.strip():
            raise PatchValidationError(
                "patch_text must be a non-empty string",
                reason_code="malformed_patch",
            )
        try:
            return parse_unified_patch(
                patch_text,
                max_files=self.scope.max_files,
                max_bytes=self.scope.max_bytes,
                allow_binary=self.scope.allow_binary,
            )
        except ProposalValidationError as exc:
            raise PatchValidationError(
                str(exc),
                reason_code="malformed_patch",
            ) from exc

    def validate(
        self,
        patch_text: str,
        *,
        worktree_root: Path | None = None,
        expected_base_commit: str | None = None,
        expected_base_tree: str | None = None,
        visible_sources: Mapping[str, str | bytes | None] | None = None,
        run_apply_check: bool = True,
    ) -> PatchValidationResult:
        """Admit a text patch without mutating the worktree.

        When ``worktree_root`` is provided, the base commit/tree are verified and
        ``git apply --check`` runs against that tree. Failures never stage or
        write files.
        """

        digest = _patch_digest(patch_text) if isinstance(patch_text, str) else ""
        try:
            parsed = self.parse(patch_text)
        except PatchValidationError as exc:
            return PatchValidationResult(
                accepted=False,
                reason_codes=(exc.reason_code,),
                paths=(),
                operations=(),
                diagnostic=_clip(str(exc)),
                patch_digest=digest,
            )

        paths = _paths_from_parsed(parsed)
        operations = tuple(item.operation for item in parsed)
        reason_codes: list[str] = []

        for item in parsed:
            if item.binary and not self.scope.allow_binary:
                reason_codes.append("binary_forbidden")
            for candidate in (item.old_path, item.new_path):
                if not candidate:
                    continue
                ok, code = self.scope.admits(candidate)
                if not ok:
                    reason_codes.append(code or "out_of_scope")

        # Deduplicate while preserving order.
        ordered_reasons: list[str] = []
        for code in reason_codes:
            if code not in ordered_reasons:
                ordered_reasons.append(code)
        if ordered_reasons:
            return PatchValidationResult(
                accepted=False,
                reason_codes=tuple(ordered_reasons),
                paths=paths,
                operations=operations,
                parsed_files=parsed,
                diagnostic="patch paths rejected by scope",
                patch_digest=digest,
            )

        if visible_sources is not None:
            hunks = _extract_preimage_hunks(patch_text)
            for path, path_hunks in hunks.items():
                ok, code = _preimage_visible(
                    path=path,
                    hunks=path_hunks,
                    visible_sources=visible_sources,
                )
                if not ok:
                    return PatchValidationResult(
                        accepted=False,
                        reason_codes=(code,),
                        paths=paths,
                        operations=operations,
                        parsed_files=parsed,
                        diagnostic=f"preimage not visible for {path}",
                        patch_digest=digest,
                    )

        pre_tree = ""
        if worktree_root is not None:
            root = Path(worktree_root)
            if not root.is_dir():
                return PatchValidationResult(
                    accepted=False,
                    reason_codes=("worktree_missing",),
                    paths=paths,
                    operations=operations,
                    parsed_files=parsed,
                    diagnostic="worktree root is missing",
                    patch_digest=digest,
                )
            try:
                head = _head_commit(root)
                tree = _tree_for_commit(root, head)
            except PatchValidationError as exc:
                return PatchValidationResult(
                    accepted=False,
                    reason_codes=(exc.reason_code,),
                    paths=paths,
                    operations=operations,
                    parsed_files=parsed,
                    diagnostic=_clip(str(exc)),
                    patch_digest=digest,
                )
            if expected_base_commit and head != expected_base_commit:
                return PatchValidationResult(
                    accepted=False,
                    reason_codes=("stale_base",),
                    paths=paths,
                    operations=operations,
                    parsed_files=parsed,
                    pre_tree=tree,
                    diagnostic=(
                        f"worktree HEAD {head} != expected base {expected_base_commit}"
                    ),
                    patch_digest=digest,
                )
            if expected_base_tree and tree != expected_base_tree:
                return PatchValidationResult(
                    accepted=False,
                    reason_codes=("stale_base",),
                    paths=paths,
                    operations=operations,
                    parsed_files=parsed,
                    pre_tree=tree,
                    diagnostic=(
                        f"worktree tree {tree} != expected base tree {expected_base_tree}"
                    ),
                    patch_digest=digest,
                )
            pre_tree = tree
            if run_apply_check:
                checked = _run_git(
                    ["apply", "--check", "--whitespace=nowarn", "-"],
                    cwd=root,
                    stdin=patch_text,
                )
                if checked.returncode != 0:
                    return PatchValidationResult(
                        accepted=False,
                        reason_codes=("apply_check_failed",),
                        paths=paths,
                        operations=operations,
                        parsed_files=parsed,
                        pre_tree=pre_tree,
                        diagnostic=_clip(
                            checked.stderr or checked.stdout or "git apply --check failed"
                        ),
                        patch_digest=digest,
                    )

        return PatchValidationResult(
            accepted=True,
            reason_codes=(),
            paths=paths,
            operations=operations,
            parsed_files=parsed,
            pre_tree=pre_tree,
            diagnostic="",
            patch_digest=digest,
        )


def validate_patch(
    patch_text: str,
    scope: PatchScope | Mapping[str, Any],
    *,
    worktree_root: Path | None = None,
    expected_base_commit: str | None = None,
    expected_base_tree: str | None = None,
    visible_sources: Mapping[str, str | bytes | None] | None = None,
    run_apply_check: bool = True,
) -> PatchValidationResult:
    """Module-level patch admission entrypoint."""

    return PatchValidator(scope).validate(
        patch_text,
        worktree_root=worktree_root,
        expected_base_commit=expected_base_commit,
        expected_base_tree=expected_base_tree,
        visible_sources=visible_sources,
        run_apply_check=run_apply_check,
    )


def apply_patch(
    patch_text: str,
    *,
    worktree_root: Path,
    scope: PatchScope | Mapping[str, Any],
    expected_base_commit: str | None = None,
    expected_base_tree: str | None = None,
    visible_sources: Mapping[str, str | bytes | None] | None = None,
) -> PatchApplyResult:
    """Validate then apply a text patch inside an isolated worktree only.

    The function refuses to operate on a path that is the caller's bare repo
    root when a ``.git`` **directory** is present and the path is not a linked
    worktree (``.git`` file). Callers should always pass the disposable
    worktree path created by :func:`create_isolated_worktree`.
    """

    root = Path(worktree_root)
    validation = validate_patch(
        patch_text,
        scope,
        worktree_root=root,
        expected_base_commit=expected_base_commit,
        expected_base_tree=expected_base_tree,
        visible_sources=visible_sources,
        run_apply_check=True,
    )
    pre_tree = validation.pre_tree or ""
    base = expected_base_commit or (
        _head_commit(root) if root.is_dir() else ""
    )
    if not validation.accepted:
        return PatchApplyResult(
            applied=False,
            validation=validation,
            pre_tree=pre_tree,
            post_tree=pre_tree,
            base_commit=base,
            worktree_path=str(root),
            reason_codes=validation.reason_codes,
            diagnostic=validation.diagnostic,
        )

    # Capture pre-index tree again immediately before apply.
    try:
        pre_tree = _tree_for_commit(root, _head_commit(root))
        # Include dirty index/worktree state if any prior partial apply existed.
        status = _run_git(["status", "--porcelain"], cwd=root)
        if status.returncode == 0 and (status.stdout or "").strip():
            # Dirty tree before apply is not expected; reject without applying.
            return PatchApplyResult(
                applied=False,
                validation=PatchValidationResult(
                    accepted=False,
                    reason_codes=("dirty_worktree",),
                    paths=validation.paths,
                    operations=validation.operations,
                    parsed_files=validation.parsed_files,
                    pre_tree=pre_tree,
                    diagnostic="worktree is dirty before apply",
                    patch_digest=validation.patch_digest,
                ),
                pre_tree=pre_tree,
                post_tree=pre_tree,
                base_commit=base,
                worktree_path=str(root),
                reason_codes=("dirty_worktree",),
                diagnostic="worktree is dirty before apply",
            )
    except PatchValidationError as exc:
        return PatchApplyResult(
            applied=False,
            validation=PatchValidationResult(
                accepted=False,
                reason_codes=(exc.reason_code,),
                paths=validation.paths,
                operations=validation.operations,
                parsed_files=validation.parsed_files,
                pre_tree=pre_tree,
                diagnostic=_clip(str(exc)),
                patch_digest=validation.patch_digest,
            ),
            pre_tree=pre_tree,
            post_tree=pre_tree,
            base_commit=base,
            worktree_path=str(root),
            reason_codes=(exc.reason_code,),
            diagnostic=_clip(str(exc)),
        )

    applied = _run_git(
        ["apply", "--whitespace=nowarn", "-"],
        cwd=root,
        stdin=patch_text,
    )
    if applied.returncode != 0:
        # Best-effort hard reset to the base commit so a partial apply cannot
        # linger. This only mutates the disposable worktree.
        if base:
            _run_git(["reset", "--hard", base], cwd=root)
            _run_git(["clean", "-fdx"], cwd=root)
        return PatchApplyResult(
            applied=False,
            validation=validation,
            pre_tree=pre_tree,
            post_tree=pre_tree,
            base_commit=base,
            worktree_path=str(root),
            reason_codes=("apply_failed",),
            diagnostic=_clip(applied.stderr or applied.stdout or "git apply failed"),
        )

    # Stage all changes so write-tree reflects the applied patch.
    _run_git(["add", "-A"], cwd=root)
    try:
        post_tree = _write_tree(root)
    except PatchValidationError as exc:
        if base:
            _run_git(["reset", "--hard", base], cwd=root)
            _run_git(["clean", "-fdx"], cwd=root)
        return PatchApplyResult(
            applied=False,
            validation=validation,
            pre_tree=pre_tree,
            post_tree=pre_tree,
            base_commit=base,
            worktree_path=str(root),
            reason_codes=(exc.reason_code,),
            diagnostic=_clip(str(exc)),
        )

    return PatchApplyResult(
        applied=True,
        validation=validation,
        pre_tree=pre_tree,
        post_tree=post_tree,
        base_commit=base,
        worktree_path=str(root),
        reason_codes=(),
        diagnostic="",
    )


@dataclass
class IsolatedWorktree:
    """Fenced disposable worktree bound to one attempt identity.

    Ownership mutations go through ``WorktreeLifecycleStore``. Publication and
    cleanup require the current lease/fence; peer or stale owners are rejected.
    """

    repo_root: Path
    worktree_path: Path
    base_commit: str
    base_tree: str
    task_id: str
    attempt: int
    lane_id: str
    lease_id: str
    fence: int
    lifecycle_store: WorktreeLifecycleStore
    phase: WorktreePhase = WorktreePhase.PREPARING
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
        if not self.branch:
            safe_task = re.sub(r"[^A-Za-z0-9._-]+", "-", self.task_id).strip("-") or "task"
            self.branch = f"semantic/{safe_task}-a{int(self.attempt)}"
        if self.journal_path is None:
            store_dir = self.lifecycle_store.store_dir
            assert store_dir is not None
            digest = hashlib.sha256(
                normalize_workspace_path(self.worktree_path).encode("utf-8")
            ).hexdigest()[:16]
            self.journal_path = Path(store_dir) / f"attempt-{digest}.json"

    # ---------------------------------------------------------------- journal

    def _journal_payload(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": ATTEMPT_JOURNAL_SCHEMA,
            "interface": ISOLATED_PATCH_WORKTREE_INTERFACE,
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
            raise WorktreeFenceError(
                "caller lease/fence does not match live worktree owner",
                reason_code="stale_owner",
            )
        record = self.lifecycle_store.load_workspace(self.worktree_path)
        if record is None:
            raise WorktreeFenceError(
                "lifecycle record missing for worktree",
                reason_code="missing_record",
            )
        if str(record.lease_id) != str(lease_id) or int(record.fence) != int(fence):
            raise WorktreeFenceError(
                "lifecycle fence advanced; caller is stale",
                reason_code="stale_owner",
            )

    def lease_binding(self) -> LeaseBinding:
        return LeaseBinding.from_dict(
            {
                "attempt_id": f"{self.task_id}#{self.attempt}",
                "fencing_token": int(self.fence),
                "lease_id": self.lease_id,
                "logical_epoch": int(self.attempt),
            }
        )

    def assert_base_current(self) -> None:
        """Fail closed when the disposable worktree left the bound base."""

        if not self.worktree_path.is_dir():
            raise PatchValidationError(
                "worktree path missing",
                reason_code="worktree_missing",
            )
        head = _head_commit(self.worktree_path)
        tree = _tree_for_commit(self.worktree_path, head)
        if head != self.base_commit or tree != self.base_tree:
            raise PatchValidationError(
                "worktree base commit/tree is stale",
                reason_code="stale_base",
            )

    def assert_root_unmutated(self) -> None:
        """Prove the caller's repository root HEAD is unchanged."""

        if not self.root_head_at_create:
            return
        head = _head_commit(self.repo_root)
        if head != self.root_head_at_create:
            raise PatchValidationError(
                "repository root HEAD mutated unexpectedly",
                reason_code="root_mutated",
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
            self.phase = WorktreePhase.READY
            self._write_journal()

    def publish(
        self,
        *,
        lease_id: str,
        fence: int,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Publish a bounded result only under the live owner fence."""

        with self._lock:
            try:
                self._require_owner(lease_id=lease_id, fence=fence)
            except WorktreeFenceError:
                raise
            if self.phase not in {WorktreePhase.APPLIED, WorktreePhase.READY, WorktreePhase.REJECTED}:
                raise WorktreeFenceError(
                    f"cannot publish from phase {self.phase.value}",
                    reason_code="invalid_phase",
                )
            record = self.lifecycle_store.mark_settling(
                self.worktree_path,
                lease_id=self.lease_id,
                expected_fence=self.fence,
            )
            self._sync_fence_from_record(record)
            payload = {
                "schema": WORKTREE_ADAPTER_SCHEMA,
                "interface": ISOLATED_PATCH_WORKTREE_INTERFACE,
                "published": True,
                "task_id": self.task_id,
                "attempt": int(self.attempt),
                "lease_id": self.lease_id,
                "fence": int(self.fence),
                "base_commit": self.base_commit,
                "base_tree": self.base_tree,
                "worktree_path": str(self.worktree_path),
                "phase": self.phase.value,
                "result": dict(result),
            }
            self._write_journal(publication=payload)
            return payload

    def validate_patch(
        self,
        patch_text: str,
        scope: PatchScope | Mapping[str, Any],
        *,
        lease_id: str | None = None,
        fence: int | None = None,
        visible_sources: Mapping[str, str | bytes | None] | None = None,
    ) -> PatchValidationResult:
        with self._lock:
            if lease_id is not None and fence is not None:
                self._require_owner(lease_id=lease_id, fence=fence)
            self.phase = WorktreePhase.VALIDATING
            self._write_journal()
            try:
                self.assert_base_current()
            except PatchValidationError as exc:
                result = PatchValidationResult(
                    accepted=False,
                    reason_codes=(exc.reason_code,),
                    paths=(),
                    operations=(),
                    diagnostic=_clip(str(exc)),
                    patch_digest=_patch_digest(patch_text)
                    if isinstance(patch_text, str)
                    else "",
                )
                self.phase = WorktreePhase.REJECTED
                self._write_journal(validation=result.to_dict())
                return result
            result = validate_patch(
                patch_text,
                scope,
                worktree_root=self.worktree_path,
                expected_base_commit=self.base_commit,
                expected_base_tree=self.base_tree,
                visible_sources=visible_sources,
                run_apply_check=True,
            )
            self.phase = (
                WorktreePhase.READY if result.accepted else WorktreePhase.REJECTED
            )
            self._write_journal(validation=result.to_dict())
            self.assert_root_unmutated()
            return result

    def apply_patch(
        self,
        patch_text: str,
        scope: PatchScope | Mapping[str, Any],
        *,
        lease_id: str | None = None,
        fence: int | None = None,
        visible_sources: Mapping[str, str | bytes | None] | None = None,
    ) -> PatchApplyResult:
        with self._lock:
            if lease_id is not None and fence is not None:
                self._require_owner(lease_id=lease_id, fence=fence)
            # Validate first while still at READY/REJECTED-safe phase.
            validation = self.validate_patch(
                patch_text,
                scope,
                lease_id=self.lease_id,
                fence=self.fence,
                visible_sources=visible_sources,
            )
            if not validation.accepted:
                return PatchApplyResult(
                    applied=False,
                    validation=validation,
                    pre_tree=validation.pre_tree,
                    post_tree=validation.pre_tree,
                    base_commit=self.base_commit,
                    worktree_path=str(self.worktree_path),
                    reason_codes=validation.reason_codes,
                    diagnostic=validation.diagnostic,
                )
            self.phase = WorktreePhase.APPLYING
            self._write_journal(validation=validation.to_dict())
            result = apply_patch(
                patch_text,
                worktree_root=self.worktree_path,
                scope=scope,
                expected_base_commit=self.base_commit,
                expected_base_tree=self.base_tree,
                visible_sources=visible_sources,
            )
            self.phase = (
                WorktreePhase.APPLIED if result.applied else WorktreePhase.REJECTED
            )
            self._write_journal(apply=result.to_dict())
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
            if self._closed and self.phase is WorktreePhase.TERMINAL:
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
                except WorktreeFenceError:
                    if not force_peer:
                        raise
            if not owner_cleanup:
                decision = self.authorize_cleanup(
                    caller_lease_id=str(lease_id or "")
                )
                if not decision.allowed:
                    raise WorktreeFenceError(
                        f"cleanup denied: {decision.reason}",
                        reason_code="cleanup_denied",
                    )
            else:
                self.phase = WorktreePhase.CLEANING
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
                    raise WorktreeFenceError(
                        str(exc),
                        reason_code="cleanup_denied",
                    ) from exc

            self.phase = WorktreePhase.CLEANING
            self._write_journal()
            removed = self._remove_worktree()
            self.phase = WorktreePhase.TERMINAL
            self._closed = True
            self._write_journal(cleaned=True, removed=removed, reason=reason)
            # Drop journal only after successful terminal transition so recovery
            # can observe a complete terminal record via lifecycle store.
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

    def __enter__(self) -> "IsolatedWorktree":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except WorktreeFenceError:
            # Best-effort recovery path for interrupted owner cleanup.
            recover_isolated_worktree(
                lifecycle_store=self.lifecycle_store,
                worktree_path=self.worktree_path,
                repo_root=self.repo_root,
            )

    def to_dict(self) -> dict[str, Any]:
        return self._journal_payload()


def _create_detached_worktree_at_base(
    *,
    repo_root: Path,
    worktree_path: Path,
    base_commit: str,
) -> None:
    """Narrow base-ref extension over managed detached worktree creation.

    ``managed_git_worktree`` always binds ``HEAD``. This helper selects the
    exact scanned base commit/tree required by the harness plan.
    """

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if worktree_path.exists():
        raise PatchValidationError(
            f"worktree path already exists: {worktree_path}",
            reason_code="worktree_exists",
        )
    # Verify the base is known to the object database before add.
    resolved = _rev_parse(repo_root, base_commit)
    result = _run_git(
        ["worktree", "add", "--detach", str(worktree_path), resolved],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise PatchValidationError(
            _clip(result.stderr or result.stdout or "git worktree add failed"),
            reason_code="worktree_create_failed",
        )
    head = _head_commit(worktree_path)
    if head != resolved:
        # Failed closed: remove the partial worktree and surface stale base.
        _run_git(
            ["worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_root,
        )
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        raise PatchValidationError(
            "created worktree HEAD does not match requested base",
            reason_code="stale_base",
        )


def create_isolated_worktree(
    *,
    repo_root: Path | str,
    worktree_path: Path | str | None = None,
    base_commit: str | None = None,
    base_tree: str | None = None,
    task_id: str,
    attempt: int = 1,
    lane_id: str = "semantic",
    canonical_task_cid: str = "",
    merge_target: str = "HEAD",
    branch: str = "",
    lifecycle_store: WorktreeLifecycleStore | None = None,
    lease_id: str | None = None,
    retain_on_success: bool = False,
    worktree_parent: Path | str | None = None,
) -> IsolatedWorktree:
    """Acquire a fenced claim then create a disposable worktree at ``base_commit``.

    The lifecycle preparing record is published **before** ``git worktree add``
    so peer cleaners never treat the checkout as an unclaimed orphan.
    """

    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise PatchValidationError(
            "repo_root is not a Git repository",
            reason_code="invalid_repo",
        )
    root_head = _head_commit(root)
    resolved_base = _rev_parse(root, base_commit or "HEAD")
    resolved_tree = base_tree or _tree_for_commit(root, resolved_base)
    actual_tree = _tree_for_commit(root, resolved_base)
    if actual_tree != resolved_tree:
        raise PatchValidationError(
            "base_tree does not match base_commit",
            reason_code="stale_base",
        )

    store = lifecycle_store or WorktreeLifecycleStore(repo_root=root)
    if worktree_path is None:
        parent = Path(worktree_parent) if worktree_parent is not None else root.parent / "semantic-worktrees"
        parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{task_id}:{attempt}:{resolved_base}:{uuid.uuid4().hex}".encode("utf-8")
        ).hexdigest()[:12]
        worktree_path = parent / f"wt-{digest}"
    workspace = Path(worktree_path)

    # Publish preparing claim before materializing the checkout.
    try:
        record = store.begin_preparing(
            task_id=task_id,
            canonical_task_cid=canonical_task_cid,
            attempt=int(attempt),
            lane_id=lane_id,
            workspace_path=workspace,
            branch=branch
            or f"semantic/{re.sub(r'[^A-Za-z0-9._-]+', '-', task_id).strip('-') or 'task'}-a{int(attempt)}",
            merge_target=merge_target,
            lease_id=lease_id,
            state_dir=str(store.store_dir or ""),
            owner=current_process_birth(),
        )
    except DuplicateAttemptError as exc:
        raise WorktreeFenceError(
            str(exc),
            reason_code="duplicate_attempt",
        ) from exc

    isolated = IsolatedWorktree(
        repo_root=root,
        worktree_path=workspace,
        base_commit=resolved_base,
        base_tree=resolved_tree,
        task_id=task_id,
        attempt=int(attempt),
        lane_id=lane_id,
        lease_id=record.lease_id,
        fence=int(record.fence),
        lifecycle_store=store,
        phase=WorktreePhase.PREPARING,
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
        # Confirm tree identity inside the disposable checkout.
        isolated.assert_base_current()
        isolated.mark_ready()
    except Exception as exc:
        # Recoverable prepare failure: mark terminal and remove partial tree.
        try:
            isolated.phase = WorktreePhase.CLEANING
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
            isolated._remove_worktree()
            isolated.phase = WorktreePhase.TERMINAL
            isolated._closed = True
            isolated._write_journal(prepare_error=_clip(str(exc)), cleaned=True)
        except Exception:
            pass
        if isinstance(exc, (PatchValidationError, WorktreeFenceError)):
            raise
        raise PatchValidationError(
            _clip(str(exc)),
            reason_code="worktree_create_failed",
        ) from exc

    return isolated


def recover_isolated_worktree(
    *,
    lifecycle_store: WorktreeLifecycleStore,
    worktree_path: Path | str,
    repo_root: Path | str | None = None,
    journal_path: Path | str | None = None,
    caller_lease_id: str = "",
) -> dict[str, Any]:
    """Recover interrupted prepare/apply/cleanup states for one workspace.

    * preparing without a checkout → terminal + prune
    * applying/validating with dirty disposable tree → hard-reset to base when
      journal base is known, then allow owner/peer cleanup per fence policy
    * cleaning / terminal with residual path → force-remove when authorized
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
        journal = _load_json_dict(Path(store_dir) / f"attempt-{digest}.json")

    root = Path(repo_root) if repo_root is not None else None
    if root is None and journal and journal.get("repo_root"):
        root = Path(str(journal["repo_root"]))
    if root is None and record is not None and record.repo_root:
        root = Path(record.repo_root)
    if root is None:
        root = Path(lifecycle_store.repo_root)

    phase = ""
    base_commit = ""
    if journal:
        phase = str(journal.get("phase") or "")
        base_commit = str(journal.get("base_commit") or "")
    if record is not None and not phase:
        phase = record.state.value

    actions: list[str] = []

    # Interrupted apply: restore disposable tree to bound base when possible.
    if (
        phase in {WorktreePhase.APPLYING.value, WorktreePhase.VALIDATING.value, "applying", "validating"}
        and workspace.is_dir()
        and base_commit
    ):
        reset = _run_git(["reset", "--hard", base_commit], cwd=workspace)
        clean = _run_git(["clean", "-fdx"], cwd=workspace)
        actions.append(
            "reset_base" if reset.returncode == 0 and clean.returncode == 0 else "reset_failed"
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

    # Owner-driven recovery for interrupted nonterminal phases. A live owner
    # presenting the exact lease/fence may always force terminal so prepare
    # or apply crashes do not leave peer-blocking claims forever.
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
                WorktreePhase.PREPARING.value,
                WorktreePhase.APPLYING.value,
                WorktreePhase.VALIDATING.value,
                WorktreePhase.CLEANING.value,
                "preparing",
                "applying",
                "validating",
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
    if decision.allowed or (
        record is not None and record.state is WorkspaceLifecycleState.TERMINAL
    ):
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
            journal_file = Path(store_dir) / f"attempt-{digest}.json"
            if journal_file.exists():
                try:
                    journal_file.unlink()
                    actions.append("journal_cleared")
                except OSError:
                    actions.append("journal_clear_failed")
    else:
        actions.append(f"cleanup_denied:{decision.reason}")

    return {
        "schema": ATTEMPT_JOURNAL_SCHEMA,
        "recovered": True,
        "worktree_path": str(workspace),
        "phase": phase,
        "actions": actions,
        "cleanup_allowed": bool(decision.allowed),
        "cleanup_reason": decision.reason,
        "removed": removed,
    }


def isolated_patch_worktree_descriptor() -> dict[str, Any]:
    """Closed interface metadata for IsolatedPatchWorktree@1."""

    return {
        "interface": ISOLATED_PATCH_WORKTREE_INTERFACE,
        "schema": WORKTREE_ADAPTER_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "composes": [
            "WorktreeLifecycleStore",
            "LeaseCoordinator",
            "todo_daemon.worktrees.managed_git_worktree",
            "todo_daemon.worktrees.GitWorktreeSession",
            "validation.proposal_validation.parse_unified_patch",
            "validation.proposal_validation.validate_untrusted_implementation_proposal",
            "todo_daemon.production_context_slice.assert_proposal_covered_by_context",
        ],
        "symbols": [
            "IsolatedWorktree",
            "PatchValidator",
            "PatchScope",
            "PatchValidationError",
            "PatchValidationResult",
            "PatchApplyResult",
            "WorktreePhase",
            "WorktreeFenceError",
            "create_isolated_worktree",
            "validate_patch",
            "apply_patch",
            "recover_isolated_worktree",
        ],
        "invariants": [
            "stale_base_causes_no_target_or_root_mutation",
            "invisible_preimage_causes_no_mutation",
            "malformed_or_out_of_scope_patch_causes_no_mutation",
            "failed_apply_check_causes_no_mutation",
            "allowed_text_patches_apply_deterministically",
            "concurrent_or_stale_owners_cannot_publish_or_clean_live_peer",
            "interrupted_prepare_apply_cleanup_recover_safely",
            "never_mutate_user_checkout",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "ATTEMPT_JOURNAL_SCHEMA",
    "ISOLATED_PATCH_WORKTREE_INTERFACE",
    "WORKTREE_ADAPTER_SCHEMA",
    "IsolatedWorktree",
    "PatchApplyResult",
    "PatchScope",
    "PatchValidationError",
    "PatchValidationResult",
    "PatchValidator",
    "WorktreeFenceError",
    "WorktreePhase",
    "apply_patch",
    "create_isolated_worktree",
    "isolated_patch_worktree_descriptor",
    "recover_isolated_worktree",
    "validate_patch",
]
