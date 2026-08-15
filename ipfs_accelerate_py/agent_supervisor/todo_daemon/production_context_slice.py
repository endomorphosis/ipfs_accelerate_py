"""Fail-closed, content-addressed source context for production providers.

Model providers must not be given an ambient checkout.  This module builds the
small, explicit source view that may be attached to a production contract
packet.  The view is bound to one canonical task, Git commit/tree, repository,
and read/effect scope.  Visible source and every omitted byte are independently
CID-addressed, so freshness and patch coverage can be checked without treating
a hash of hidden source as source context.

The module deliberately has no provider or daemon dependencies.  Callers can
build a manifest before routing, attach :meth:`ProductionContextSliceManifest.
provider_payload` to the packet, and call :func:`verify_production_context_slice`
again immediately before a provider call or write.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, NoReturn

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ..control.profile_authority import verify_did_key_signature
from .git_environment import sanitized_git_environment

PRODUCTION_CONTEXT_SLICE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-context-slice@1"
)
PRODUCTION_CONTEXT_SLICE_INTERFACE: Final = "ProductionContextSlice@1"
PRODUCTION_EVIDENCE_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-evidence-authority@1"
)
PRODUCTION_EVIDENCE_AUTHORITY_INTERFACE: Final = "ProductionEvidenceAuthority@1"
CANDIDATE_REF_AUTHORITY_APPENDIX_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/candidate-ref-authority-appendix@1"
)
_CONTROL_LAUNCH_RECEIPT_SCHEMA: Final = (
    "proof-carrying-context-engine/control-launch-receipt@1"
)
# Production implement/review packets must be able to surface complete
# declared effect files for full-file replacements.  A 4k-token cap forced
# partial slices on ordinary modules (~16KB+), so admitted Grok proposals
# that replace those files failed closed at write time as context_insufficient.
# Keep a hard upper bound well below the router transport envelope.
MAX_PROVIDER_PROMPT_TOKENS: Final = 32_768
DEFAULT_RESERVED_PROMPT_TOKENS: Final = 1_536
DEFAULT_MAX_SCOPE_PATHS: Final = 8
DEFAULT_MAX_SOURCE_BYTES: Final = 1_048_576
DEFAULT_WHOLE_FILE_BYTES: Final = 131_072
DEFAULT_MAX_EVIDENCE_DECLARATIONS: Final = 16
# Keep the initial prompt small.  Additional explicitly-authorized blobs are
# represented by immutable expansion handles instead of being silently
# discarded or widening the provider's ambient filesystem access.
DEFAULT_MAX_EVIDENCE_SOURCE_PATHS: Final = 8
DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES: Final = 256
DEFAULT_MAX_EVIDENCE_SOURCE_BYTES: Final = 256 * 1_024
DEFAULT_MAX_GOVERNED_REPOSITORY_ROOTS: Final = 8
DEFAULT_MAX_EVIDENCE_SUMMARY_ITEMS: Final = 8
DEFAULT_MAX_EVIDENCE_EXPANSION_ROUND: Final = 63
DEFAULT_MAX_EVIDENCE_REFS: Final = 8
DEFAULT_MAX_EVIDENCE_REF_DIFFS: Final = 4
DEFAULT_MAX_EVIDENCE_REF_DIFF_BYTES: Final = 128 * 1_024
DEFAULT_MAX_EVIDENCE_SUMMARY_BYTES: Final = 64 * 1_024
DEFAULT_MAX_EVIDENCE_TOTAL_SUMMARY_BYTES: Final = 512 * 1_024
DEFAULT_GIT_TIMEOUT_SECONDS: Final = 15.0
DEFAULT_MAX_GIT_CAPTURE_BYTES: Final = 2 * 1_024 * 1_024
DEFAULT_MAX_LAUNCH_AUTHORITY_BYTES: Final = 1 * 1_024 * 1_024
DEFAULT_MAX_EVIDENCE_GIT_CALLS: Final = 1_024
DEFAULT_MAX_EVIDENCE_GIT_OUTPUT_BYTES: Final = 16 * 1_024 * 1_024
DEFAULT_MAX_EVIDENCE_INSPECTED_BLOB_BYTES: Final = 16 * 1_024 * 1_024
DEFAULT_EVIDENCE_SCAN_TIMEOUT_SECONDS: Final = 30.0

_SYSTEM_GIT = next(
    (candidate for candidate in ("/usr/bin/git", "/bin/git") if Path(candidate).is_file()),
    shutil.which("git") or "git",
)
PRODUCTION_GIT_EXECUTABLE: Final = str(Path(_SYSTEM_GIT).resolve())

_GIT_OID_RE = re.compile(r"\A[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_FULL_GIT_REF_RE = re.compile(
    r"\Arefs/pcce-candidates/[A-Za-z0-9][A-Za-z0-9._/-]*\Z"
)
_CANDIDATE_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)
_DRIVE_RE = re.compile(r"\A[A-Za-z]:")
# Credential detectors must ignore ordinary code (type annotations, parameter
# names such as ``token: str``, local variables) while still catching
# assignment of secret-looking literals and transport credentials.
_SECRET_TEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret)"
        r"\s*[:=]\s*['\"][^'\"]{6,}['\"]"
    ),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret|token)"
        r"\s*[:=]\s*[A-Za-z0-9._~+/=-]{8,}"
    ),
    re.compile(
        r"(?i)\btoken\s*[:=]\s*['\"][A-Za-z0-9._\-+/=]{12,}['\"]"
    ),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
)
_SECRET_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)


class ProductionContextSliceError(ValueError):
    """A bounded public failure at the provider-context trust boundary."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(reason_code: str, message: str) -> NoReturn:
    raise ProductionContextSliceError(message, reason_code=reason_code)


def _raw_cid(value: bytes) -> str:
    """Return CIDv1 raw/sha2-256 for exact bytes."""

    digest = hashlib.sha256(value).digest()
    raw = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _token_estimate(value: bytes) -> int:
    """Match the deterministic provider-router estimate."""

    return max(1, (len(value) + 3) // 4)


@dataclass(slots=True)
class _EvidenceScanBudget:
    """One request-wide, fail-closed budget for repository evidence reads."""

    deadline: float
    git_calls: int = 0
    git_output_bytes: int = 0
    inventory_entries: int = 0
    inspected_blob_bytes: int = 0

    def remaining_seconds(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            _fail("evidence_scan_timeout", "evidence scan exceeded its deadline")
        return remaining

    def begin_git_call(self) -> float:
        self.remaining_seconds()
        self.git_calls += 1
        if self.git_calls > DEFAULT_MAX_EVIDENCE_GIT_CALLS:
            _fail("evidence_scan_budget_exceeded", "evidence Git-call budget exceeded")
        return min(DEFAULT_GIT_TIMEOUT_SECONDS, self.remaining_seconds())

    def consume_git_output(self, amount: int) -> None:
        self.remaining_seconds()
        self.git_output_bytes += max(0, int(amount))
        if self.git_output_bytes > DEFAULT_MAX_EVIDENCE_GIT_OUTPUT_BYTES:
            _fail("evidence_scan_budget_exceeded", "evidence Git-output budget exceeded")

    def consume_entries(self, amount: int) -> None:
        self.remaining_seconds()
        self.inventory_entries += max(0, int(amount))
        if self.inventory_entries > DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES:
            _fail("evidence_scan_budget_exceeded", "evidence entry budget exceeded")

    def consume_blob_bytes(self, amount: int) -> None:
        self.remaining_seconds()
        self.inspected_blob_bytes += max(0, int(amount))
        if self.inspected_blob_bytes > DEFAULT_MAX_EVIDENCE_INSPECTED_BLOB_BYTES:
            _fail("evidence_scan_budget_exceeded", "evidence blob-read budget exceeded")


_ACTIVE_EVIDENCE_SCAN_BUDGET: ContextVar[_EvidenceScanBudget | None] = ContextVar(
    "production_evidence_scan_budget",
    default=None,
)


@contextmanager
def production_evidence_scan_budget() -> Any:
    """Reuse one budget through every nested evidence compiler operation."""

    current = _ACTIVE_EVIDENCE_SCAN_BUDGET.get()
    if current is not None:
        current.remaining_seconds()
        yield current
        return
    budget = _EvidenceScanBudget(
        deadline=time.monotonic() + DEFAULT_EVIDENCE_SCAN_TIMEOUT_SECONDS
    )
    token = _ACTIVE_EVIDENCE_SCAN_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _ACTIVE_EVIDENCE_SCAN_BUDGET.reset(token)


def _active_evidence_scan_budget() -> _EvidenceScanBudget | None:
    return _ACTIVE_EVIDENCE_SCAN_BUDGET.get()


def production_evidence_scan_budgeted(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """Give direct API callers the same aggregate budget as daemon callers."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with production_evidence_scan_budget():
            return function(*args, **kwargs)

    return wrapped


def _run_git_bounded(
    repo_root: Path,
    *arguments: str,
    maximum_stdout_bytes: int,
    input_bytes: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    """Execute fixed Git while bounding stdout/stderr during capture."""

    if maximum_stdout_bytes < 1 or maximum_stdout_bytes > DEFAULT_MAX_GIT_CAPTURE_BYTES:
        _fail("budget_invalid", "Git stdout capture bound is invalid")
    if input_bytes is not None and len(input_bytes) > DEFAULT_MAX_GIT_CAPTURE_BYTES:
        _fail("budget_invalid", "Git stdin exceeds its bounded input limit")
    scan_budget = _active_evidence_scan_budget()
    timeout_seconds = (
        scan_budget.begin_git_call()
        if scan_budget is not None
        else DEFAULT_GIT_TIMEOUT_SECONDS
    )
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    stderr_limit = 64 * 1024
    deadline = time.monotonic() + timeout_seconds
    input_view = memoryview(input_bytes or b"")
    input_offset = 0
    try:
        process = subprocess.Popen(
            [PRODUCTION_GIT_EXECUTABLE, "--literal-pathspecs", *arguments],
            cwd=repo_root,
            env=sanitized_git_environment(),
            stdin=(subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        for stream, label in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            if input_view:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    [PRODUCTION_GIT_EXECUTABLE, *arguments],
                    timeout_seconds,
                )
            if scan_budget is not None:
                scan_budget.remaining_seconds()
            for key, _mask in selector.select(min(remaining, 0.1)):
                stream = key.fileobj
                if key.data == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            input_view[input_offset : input_offset + 64 * 1024],
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    input_offset += written
                    if input_offset >= len(input_view):
                        selector.unregister(stream)
                        stream.close()
                    continue
                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    continue
                target = stdout if key.data == "stdout" else stderr
                limit = maximum_stdout_bytes if key.data == "stdout" else stderr_limit
                target.extend(chunk)
                if scan_budget is not None:
                    scan_budget.consume_git_output(len(chunk))
                if len(target) > limit:
                    reason = (
                        "repository_output_too_large"
                        if key.data == "stdout"
                        else "repository_unavailable"
                    )
                    _fail(reason, f"Git {key.data} exceeds its capture bound")
        returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        return returncode, bytes(stdout), bytes(stderr)
    except ProductionContextSliceError:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    except (OSError, BrokenPipeError, subprocess.TimeoutExpired) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise ProductionContextSliceError(
            "repository identity or object is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr, process.stdin):
                if stream is not None and not stream.closed:
                    stream.close()


def _git(repo_root: Path, *arguments: str, input_text: str | None = None) -> str:
    try:
        returncode, stdout, _stderr = _run_git_bounded(
            repo_root,
            *arguments,
            maximum_stdout_bytes=DEFAULT_MAX_GIT_CAPTURE_BYTES,
            input_bytes=(input_text.encode("utf-8") if input_text is not None else None),
        )
        decoded = stdout.decode("utf-8", errors="strict")
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise ProductionContextSliceError(
            "repository identity is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if returncode != 0:
        raise ProductionContextSliceError(
            "repository identity or object is unavailable",
            reason_code="repository_unavailable",
        )
    return decoded


def _git_bytes(
    repo_root: Path,
    *arguments: str,
    maximum_bytes: int = DEFAULT_MAX_GIT_CAPTURE_BYTES,
) -> bytes:
    returncode, output, _stderr = _run_git_bounded(
        repo_root,
        *arguments,
        maximum_stdout_bytes=maximum_bytes,
    )
    if returncode != 0:
        raise ProductionContextSliceError(
            "repository object is unavailable",
            reason_code="repository_unavailable",
        )
    scan_budget = _active_evidence_scan_budget()
    if (
        scan_budget is not None
        and len(arguments) >= 2
        and arguments[0] == "cat-file"
        and arguments[1] == "blob"
    ):
        scan_budget.consume_blob_bytes(len(output))
    return output


def _repository_root(repo_root: str | Path) -> Path:
    root = Path(repo_root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ProductionContextSliceError(
            "repository root is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if stat.S_ISLNK(root_info.st_mode):
        _fail("symlink_escape", "repository root may not be a symbolic link")
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionContextSliceError(
            "repository root is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if not resolved.is_dir():
        _fail("repository_unavailable", "repository root must be a directory")
    reported = _git(resolved, "rev-parse", "--show-toplevel").strip()
    try:
        top = Path(reported).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionContextSliceError(
            "Git top-level is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if top != resolved:
        _fail(
            "repository_root_mismatch",
            "repository root must be the exact Git worktree top-level",
        )
    return resolved


def _canonical_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str):
        _fail("path_invalid", "scope paths must be strings")
    if not raw_path or raw_path != raw_path.strip():
        _fail("path_invalid", "scope paths must be nonempty and canonical")
    if "\x00" in raw_path or "\\" in raw_path:
        _fail("path_escape", "scope path contains a forbidden separator")
    if unicodedata.normalize("NFC", raw_path) != raw_path:
        _fail("path_invalid", "scope path must use NFC Unicode")
    if raw_path.startswith("/") or _DRIVE_RE.match(raw_path):
        _fail("path_escape", "absolute scope paths are forbidden")
    if raw_path.startswith(":"):
        _fail("path_escape", "Git pathspec magic is forbidden")
    parts = raw_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail("path_escape", "scope path contains an empty/dot segment")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or str(path) != raw_path:
        _fail("path_invalid", "scope path is not canonical POSIX relative form")
    if any(ord(character) < 32 for character in raw_path):
        _fail("path_invalid", "scope path contains control characters")
    if any(part.casefold() == ".git" for part in parts):
        _fail("path_escape", "Git administrative paths are forbidden")
    if parts[-1].casefold() in _SECRET_BASENAMES:
        _fail("secret_path_forbidden", "credential-bearing paths are forbidden")
    return raw_path


def _canonical_paths(
    paths: Sequence[str],
    *,
    field_name: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes, bytearray)):
        _fail("scope_invalid", f"{field_name} must be a sequence of paths")
    normalized = tuple(_canonical_path(path) for path in paths)
    if not normalized and not allow_empty:
        _fail("scope_invalid", f"{field_name} must not be empty")
    if len(normalized) > maximum:
        _fail("scope_too_broad", f"{field_name} exceeds its path-count bound")
    if len(set(normalized)) != len(normalized):
        _fail("scope_invalid", f"{field_name} contains duplicate paths")
    return tuple(sorted(normalized))


def _canonical_symbol_hints(
    symbol_hints: Mapping[str, Sequence[str]] | None,
    *,
    read_paths: Sequence[str] | None,
) -> dict[str, list[str]]:
    if symbol_hints is None:
        return {}
    if not isinstance(symbol_hints, Mapping):
        _fail("scope_invalid", "symbol hints must be a path mapping")
    reads = set(read_paths)
    result: dict[str, list[str]] = {}
    for raw_path, raw_hints in symbol_hints.items():
        path = _canonical_path(raw_path)
        if path not in reads:
            _fail("scope_widening", "symbol hints name a path outside read scope")
        if isinstance(raw_hints, (str, bytes, bytearray)):
            _fail("scope_invalid", "symbol hints must be a sequence")
        normalized_hints: list[str] = []
        for item in raw_hints:
            if not isinstance(item, str):
                _fail("scope_invalid", "symbol hints must be strings")
            normalized = item.strip()
            if (
                not normalized
                or normalized != item
                or len(normalized) > 512
                or any(ord(character) < 32 for character in normalized)
            ):
                _fail("scope_invalid", "symbol hints must be canonical")
            normalized_hints.append(normalized)
        if len(set(normalized_hints)) != len(normalized_hints):
            _fail("scope_invalid", "symbol hints must not contain duplicates")
        result[path] = sorted(normalized_hints)
    return {path: result[path] for path in sorted(result)}


def _assert_safe_worktree_path(root: Path, relative: str) -> Path:
    """Reject links and nested repository boundaries before reading a file."""

    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            _fail("source_missing", "declared source path is missing")
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared source path is unavailable",
                reason_code="source_unavailable",
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_escape", "symlinks are forbidden in source scope")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            _fail("path_invalid", "a source path parent is not a directory")
        if index < len(parts) - 1:
            marker = current / ".git"
            try:
                marker.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ProductionContextSliceError(
                    "nested repository boundary is unreadable",
                    reason_code="nested_repository_escape",
                ) from exc
            else:
                _fail(
                    "nested_repository_escape",
                    "nested Git repositories are forbidden in source scope",
                )
    if not current.is_file():
        _fail("path_invalid", "declared source path must be a regular file")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionContextSliceError(
            "declared source path cannot be resolved",
            reason_code="path_escape",
        ) from exc
    if root not in resolved.parents:
        _fail("path_escape", "declared source path escapes the repository")
    return current


def _read_regular_nofollow(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionContextSliceError(
            "declared source changed before its safe read",
            reason_code="source_unavailable",
        ) from exc
    chunks: list[bytes] = []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail("path_invalid", "declared source must remain a regular file")
        if maximum_bytes is not None and info.st_size > maximum_bytes:
            _fail("source_too_large", "declared source exceeds its read bound")
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                _fail("source_too_large", "declared source grew beyond its read bound")
            scan_budget = _active_evidence_scan_budget()
            if scan_budget is not None:
                scan_budget.consume_blob_bytes(len(chunk))
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read_repository_regular_nofollow(
    root: Path,
    relative: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read a repository file through an fd-bound, no-follow path walk.

    A prior ``lstat`` walk is useful diagnostics but is not a security boundary:
    an untrusted checkout can exchange a parent directory for a symlink between
    validation and ``open``.  Keep every parent descriptor open and resolve the
    next component relative to it, anchoring the final file to ``root``.
    """

    canonical = _canonical_path(relative)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        parts = canonical.split("/")
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=current)
            except OSError as exc:
                raise ProductionContextSliceError(
                    "declared source parent changed before its safe read",
                    reason_code="symlink_escape",
                ) from exc
            descriptors.append(child)
            current = child
            try:
                os.stat(".git", dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ProductionContextSliceError(
                    "nested repository boundary became unreadable",
                    reason_code="nested_repository_escape",
                ) from exc
            else:
                _fail(
                    "nested_repository_escape",
                    "evidence crossed an undeclared nested repository boundary",
                )
        try:
            descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared source changed before its safe read",
                reason_code="source_unavailable",
            ) from exc
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail("path_invalid", "declared source must remain a regular file")
        if maximum_bytes is not None and info.st_size > maximum_bytes:
            _fail("source_too_large", "declared source exceeds its read bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                _fail("source_too_large", "declared source grew beyond its read bound")
            chunks.append(chunk)
        return b"".join(chunks)
    except ProductionContextSliceError:
        raise
    except OSError as exc:
        raise ProductionContextSliceError(
            "repository source changed before its safe read",
            reason_code="source_unavailable",
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _assert_safe_effect_path(root: Path, relative: str) -> tuple[Path, bool]:
    """Resolve an existing or prospective effect without crossing boundaries."""

    current = root
    parts = relative.split("/")
    missing = False
    for index, part in enumerate(parts):
        current = current / part
        if missing:
            continue
        try:
            info = current.lstat()
        except FileNotFoundError:
            missing = True
            continue
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared effect path is unavailable",
                reason_code="source_unavailable",
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_escape", "symlinks are forbidden in effect scope")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            _fail("path_invalid", "an effect path parent is not a directory")
        if index < len(parts) - 1:
            marker = current / ".git"
            try:
                marker.lstat()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ProductionContextSliceError(
                    "nested repository boundary is unreadable",
                    reason_code="nested_repository_escape",
                ) from exc
            else:
                _fail(
                    "nested_repository_escape",
                    "nested Git repositories are forbidden in effect scope",
                )
    if not missing:
        try:
            info = current.lstat()
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared effect path is unavailable",
                reason_code="source_unavailable",
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            _fail("path_invalid", "existing effect path must be a regular file")
    return current, not missing


def _tree_entry_optional(
    root: Path,
    commit: str,
    relative: str,
) -> tuple[str, str, bytes] | None:
    raw = _git_bytes(
        root,
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        relative,
    )
    entries = [entry for entry in raw.split(b"\x00") if entry]
    if not entries:
        return None
    if len(entries) != 1:
        _fail("source_untracked", "declared source must be one tracked blob")
    try:
        header, encoded_path = entries[0].split(b"\t", 1)
        mode, object_type, oid = header.decode("ascii").split(" ", 2)
        entry_path = encoded_path.decode("utf-8", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise ProductionContextSliceError(
            "Git tree entry is malformed",
            reason_code="repository_malformed",
        ) from exc
    if entry_path != relative or object_type != "blob":
        _fail("source_untracked", "declared source is not an exact tracked blob")
    if mode not in {"100644", "100755"}:
        reason = "symlink_escape" if mode == "120000" else "nested_repository_escape"
        _fail(reason, "non-regular Git tree entries are forbidden")
    if not _GIT_OID_RE.fullmatch(oid):
        _fail("repository_malformed", "Git blob identity is malformed")
    return mode, oid, _git_bytes(root, "cat-file", "blob", oid)


def _tree_entry(root: Path, commit: str, relative: str) -> tuple[str, str, bytes]:
    entry = _tree_entry_optional(root, commit, relative)
    if entry is None:
        _fail("source_untracked", "declared source must be one tracked blob")
    return entry


def _absence_proof(
    *,
    path: str,
    repository_binding: Mapping[str, str],
) -> dict[str, str]:
    core = {
        "baseline_commit": repository_binding["baseline_commit"],
        "baseline_tree": repository_binding["baseline_tree"],
        "path": path,
        "repository_cid": repository_binding["repository_cid"],
        "state": "absent",
    }
    return {**core, "absence_cid": content_identity(core)}


def _assert_secret_free(text: str) -> None:
    if any(pattern.search(text) for pattern in _SECRET_TEXT_PATTERNS):
        _fail("secret_detected", "declared source may contain credentials")


def _line_offsets(value: bytes) -> tuple[int, ...]:
    offsets = [0]
    cursor = 0
    for line in value.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    if offsets[-1] != len(value):
        offsets.append(len(value))
    return tuple(offsets)


def _line_interval(
    offsets: Sequence[int],
    *,
    start_line: int,
    end_line: int,
) -> tuple[int, int]:
    if start_line < 1 or end_line < start_line:
        _fail("ast_malformed", "AST line interval is malformed")
    start_index = start_line - 1
    if start_index >= len(offsets):
        _fail("ast_malformed", "AST line interval exceeds source")
    start = offsets[start_index]
    end = offsets[min(end_line, len(offsets) - 1)]
    return start, end


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    start_line: int
    end_line: int
    kind: str
    qualified_name: str
    priority: int


def _decorated_start(node: ast.AST) -> int:
    lines = [int(getattr(node, "lineno", 1))]
    for decorator in getattr(node, "decorator_list", ()):
        lines.append(int(getattr(decorator, "lineno", lines[0])))
    return min(lines)


def _python_candidates(
    text: str,
    encoded: bytes,
    *,
    symbols: Sequence[str],
) -> tuple[_Candidate, ...]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        raise ProductionContextSliceError(
            "Python source cannot be parsed for a deterministic slice",
            reason_code="ast_parse_failed",
        ) from exc
    offsets = _line_offsets(encoded)
    wanted = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
    discovered: list[tuple[ast.AST, str, str]] = []

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                kind = (
                    "class"
                    if isinstance(node, ast.ClassDef)
                    else "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                discovered.append((node, qualified, kind))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, qualified)

    visit(tree.body)
    exact = [item for item in discovered if item[1] in wanted]
    if wanted and len({qualified for _, qualified, _ in exact}) != len(wanted):
        _fail("symbol_missing", "one or more requested Python symbols are absent")
    if not wanted:
        _fail(
            "symbol_scope_required",
            "large Python sources require exact qualified symbol hints",
        )

    candidates: list[_Candidate] = []
    # Imports and the module docstring are useful context but remain optional.
    # Collapse them into at most one contiguous module preamble so large
    # import blocks cannot explode provider prompt tokens slice-by-slice.
    preamble_nodes: list[ast.AST] = []
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node is tree.body[0]
        )
        if isinstance(node, (ast.Import, ast.ImportFrom)) or is_docstring:
            preamble_nodes.append(node)
            continue
        # Only leading preamble is treated as optional module context.
        if preamble_nodes:
            break
    if preamble_nodes:
        start_line = int(getattr(preamble_nodes[0], "lineno", 1))
        end_line = int(
            getattr(
                preamble_nodes[-1],
                "end_lineno",
                getattr(preamble_nodes[-1], "lineno", start_line),
            )
        )
        start, end = _line_interval(
            offsets,
            start_line=start_line,
            end_line=end_line,
        )
        candidates.append(
            _Candidate(
                start,
                end,
                start_line,
                end_line,
                "module_context",
                "<module>",
                10,
            )
        )
    for node, qualified, kind in exact:
        start_line = _decorated_start(node)
        end_line = int(getattr(node, "end_lineno", start_line))
        start, end = _line_interval(
            offsets,
            start_line=start_line,
            end_line=end_line,
        )
        candidates.append(
            _Candidate(start, end, start_line, end_line, kind, qualified, 0)
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.priority,
                item.start,
                item.end,
                item.qualified_name,
            ),
        )
    )


def _merge_candidates(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    """Drop nested/overlapping candidates without creating non-AST boundaries."""

    accepted: list[_Candidate] = []
    for candidate in candidates:
        if any(
            existing.start <= candidate.start and existing.end >= candidate.end
            for existing in accepted
        ):
            continue
        if any(
            not (candidate.end <= existing.start or candidate.start >= existing.end)
            for existing in accepted
        ):
            # This can occur for a requested class and one of its methods.  The
            # earlier, task-priority candidate owns the range deterministically.
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item.start, item.end)))


def _signature_candidate(
    source: bytes,
    candidate: _Candidate,
    *,
    max_bytes: int,
) -> _Candidate:
    """Shrink an oversized AST candidate to signature/docstring-only bytes.

    Large classes such as production run-registry implementations can exceed the
    provider prompt budget even when correctly selected by symbol hint.  The
    residual partition still covers the omitted body so coverage proofs remain
    exact while the provider only sees the bounded header.
    """

    if candidate.end - candidate.start <= max_bytes:
        return candidate
    raw = source[candidate.start : candidate.end]
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return candidate
    lines = text.splitlines(keepends=True)
    if not lines:
        return candidate
    cursor = len(lines[0].encode("utf-8"))
    # Capture an immediate indented docstring if present.
    if len(lines) > 1:
        second = lines[1].lstrip()
        if second.startswith(('"""', "'''")):
            quote = '"""' if second.startswith('"""') else "'''"
            acc = lines[1]
            if acc.count(quote) < 2:
                for line in lines[2:]:
                    acc += line
                    if quote in line:
                        break
            cursor = len(lines[0].encode("utf-8")) + len(acc.encode("utf-8"))
    # Hard-cap and align to the last full line within max_bytes.
    window = raw[: min(len(raw), max_bytes)]
    nl = window.rfind(b"\n")
    if nl >= 0:
        line_cap = nl + 1
    else:
        line_cap = len(window)
    cursor = max(1, min(cursor, line_cap, max_bytes, len(raw)))
    end = candidate.start + cursor
    prefix = source[:end]
    end_line = prefix.count(b"\n") + (0 if prefix.endswith(b"\n") else 1)
    return _Candidate(
        candidate.start,
        end,
        candidate.start_line,
        max(candidate.start_line, end_line),
        f"{candidate.kind}_header",
        candidate.qualified_name,
        candidate.priority,
    )


def _segment(
    source: bytes,
    candidate: _Candidate,
) -> dict[str, Any]:
    raw = source[candidate.start : candidate.end]
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProductionContextSliceError(
            "source slice is not UTF-8",
            reason_code="source_encoding_invalid",
        ) from exc
    return {
        "byte_end": candidate.end,
        "byte_length": len(raw),
        "byte_start": candidate.start,
        "content_cid": _raw_cid(raw),
        "end_line": candidate.end_line,
        "kind": candidate.kind,
        "qualified_name": candidate.qualified_name,
        "start_line": candidate.start_line,
        "utf8_text": text,
    }


def _residuals(source: bytes, slices: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cursor = 0
    for item in sorted(slices, key=lambda value: int(value["byte_start"])):
        start = int(item["byte_start"])
        end = int(item["byte_end"])
        if start < cursor or end < start or end > len(source):
            _fail("slice_overlap", "source slices overlap or exceed the blob")
        if cursor < start:
            raw = source[cursor:start]
            result.append(
                {
                    "byte_end": start,
                    "byte_length": len(raw),
                    "byte_start": cursor,
                    "content_cid": _raw_cid(raw),
                }
            )
        cursor = end
    if cursor < len(source):
        raw = source[cursor:]
        result.append(
            {
                "byte_end": len(source),
                "byte_length": len(raw),
                "byte_start": cursor,
                "content_cid": _raw_cid(raw),
            }
        )
    return result


def _source_record(
    *,
    path: str,
    mode: str,
    git_blob_oid: str,
    source: bytes,
    effect: bool,
    symbol_hints: Sequence[str],
    whole_file_bytes: int,
) -> dict[str, Any]:
    if b"\x00" in source:
        _fail("binary_source_forbidden", "binary/NUL-bearing sources are forbidden")
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProductionContextSliceError(
            "declared source is not UTF-8",
            reason_code="source_encoding_invalid",
        ) from exc
    _assert_secret_free(text)
    language = "python" if path.endswith((".py", ".pyi")) else "text"
    if len(source) <= whole_file_bytes:
        selection = {
            "mode": "whole-file@1",
            "qualified_symbols": [],
        }
        end_line = max(1, len(source.splitlines()))
        candidates = (
            _Candidate(0, len(source), 1, end_line, "whole_file", "<module>", 0),
        )
    elif language == "python":
        normalized_symbols = tuple(sorted(set(symbol_hints)))
        selection = {
            "mode": "python-qualified-symbols@1",
            "qualified_symbols": list(normalized_symbols),
        }
        candidates = _python_candidates(
            text,
            source,
            symbols=normalized_symbols,
        )
    else:
        _fail(
            "context_insufficient",
            "large non-Python sources cannot be safely sliced without a parser",
        )
    merged = _merge_candidates(candidates)
    # Always apply protocol-default header compaction to oversized AST
    # candidates.  This is independent of ``whole_file_bytes`` (which only
    # chooses whole-file vs AST mode) so build and verify remain deterministic
    # even when verification rebuilds with a zero rebuild threshold.
    compact = tuple(
        _signature_candidate(
            source, candidate, max_bytes=DEFAULT_WHOLE_FILE_BYTES
        )
        for candidate in merged
    )
    slices = [_segment(source, candidate) for candidate in compact]
    if not slices:
        _fail("context_insufficient", "declared source produced no visible context")
    residuals = _residuals(source, slices)
    partition = [
        {
            "byte_end": int(item["byte_end"]),
            "byte_start": int(item["byte_start"]),
            "content_cid": str(item["content_cid"]),
            "visibility": "visible",
        }
        for item in slices
    ] + [
        {
            "byte_end": int(item["byte_end"]),
            "byte_start": int(item["byte_start"]),
            "content_cid": str(item["content_cid"]),
            "visibility": "residual",
        }
        for item in residuals
    ]
    partition.sort(key=lambda item: (item["byte_start"], item["byte_end"]))
    return {
        "byte_length": len(source),
        "effect": bool(effect),
        "file_cid": _raw_cid(source),
        "full_visible_coverage": not residuals,
        "git_blob_oid": git_blob_oid,
        "git_mode": mode,
        "language": language,
        "partition_root_cid": content_identity(
            {
                "byte_length": len(source),
                "file_cid": _raw_cid(source),
                "segments": partition,
            }
        ),
        "path": path,
        "residuals": residuals,
        "selection": selection,
        "source_slices": slices,
    }


@dataclass(frozen=True, slots=True)
class ProductionContextSliceManifest:
    """Immutable wrapper around one canonical provider context manifest."""

    _payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        detached = json.loads(canonical_json_bytes(dict(self._payload)))
        object.__setattr__(self, "_payload", MappingProxyType(detached))

    @property
    def manifest_cid(self) -> str:
        return str(self._payload["manifest_cid"])

    @property
    def task_id(self) -> str:
        return str(self._payload["task_binding"]["task_id"])

    @property
    def canonical_task_cid(self) -> str:
        return str(self._payload["task_binding"]["canonical_task_cid"])

    @property
    def baseline_commit(self) -> str:
        return str(self._payload["repository_binding"]["baseline_commit"])

    @property
    def provider_input_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(dict(self._payload)))

    def provider_payload(self) -> dict[str, Any]:
        """Return the exact bounded value to attach to a contract packet."""

        return {"context_slice": self.to_dict()}


def _task_binding(task_id: str, task_payload: Mapping[str, Any]) -> dict[str, str]:
    tid = str(task_id or "").strip()
    if not tid or tid != task_id or len(tid) > 256:
        _fail("task_binding_invalid", "task_id is required and canonical")
    if not isinstance(task_payload, Mapping):
        _fail("task_binding_invalid", "task payload must be a mapping")
    payload_task_id = task_payload.get("task_id")
    if payload_task_id is not None and str(payload_task_id) != tid:
        _fail("task_binding_mismatch", "task payload task_id does not match")
    try:
        task_cid = content_identity(dict(task_payload))
    except Exception as exc:
        raise ProductionContextSliceError(
            "task payload is not canonical",
            reason_code="task_binding_invalid",
        ) from exc
    return {"canonical_task_cid": task_cid, "task_id": tid}


def _repository_binding(root: Path, baseline_ref: str) -> dict[str, str]:
    ref = str(baseline_ref or "").strip()
    if not ref or ref.startswith("-") or any(character.isspace() for character in ref):
        _fail("baseline_invalid", "baseline ref is required and canonical")
    commit = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if commit != head:
        _fail("baseline_stale", "baseline must be the current worktree HEAD")
    tree = _git(root, "rev-parse", "--verify", f"{commit}^{{tree}}").strip()
    if not _GIT_OID_RE.fullmatch(commit) or not _GIT_OID_RE.fullmatch(tree):
        _fail("repository_malformed", "Git commit/tree identity is malformed")
    try:
        object_format = _git(root, "rev-parse", "--show-object-format").strip()
    except ProductionContextSliceError:
        object_format = "sha1" if len(commit) == 40 else "sha256"
    if object_format not in {"sha1", "sha256"}:
        _fail("repository_malformed", "unsupported Git object format")
    binding_core = {
        "baseline_commit": commit,
        "baseline_tree": tree,
        "object_format": object_format,
    }
    return {
        **binding_core,
        "repository_cid": content_identity(
            {"schema": "git-repository-snapshot@1", **binding_core}
        ),
        "snapshot_id": f"git-commit:{commit}",
    }


def _assert_repository_tracked_clean(root: Path) -> None:
    """Check tracked/index bytes without consulting recursive submodules/hooks."""

    fixed_config = (
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "diff.external=",
    )
    for cached in (False, True):
        arguments = [
            *fixed_config,
            "diff",
            "--quiet",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
        ]
        if cached:
            arguments.append("--cached")
        arguments.extend(("HEAD", "--"))
        returncode, _stdout, _stderr = _run_git_bounded(
            root,
            *arguments,
            maximum_stdout_bytes=1,
        )
        if returncode == 1:
            _fail("launch_authority_stale", "launch repository has tracked changes")
        if returncode != 0:
            _fail("repository_unavailable", "launch cleanliness check failed")


@dataclass(frozen=True, slots=True)
class _EvidenceRepositoryRoot:
    namespace: str
    path: Path
    binding: Mapping[str, str]
    parent_gitlink_oid: str = ""


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    path: str
    repository_namespace: str
    repository_path: str
    git_mode: str
    git_blob_oid: str
    byte_length: int


def _preflight_configured_governed_root_paths(
    raw_root: str | Path,
    governed_repository_roots: Sequence[str],
) -> None:
    """Reject configured checkout symlinks before invoking Git at any root."""

    root = Path(raw_root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ProductionContextSliceError(
            "repository root is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if stat.S_ISLNK(root_info.st_mode):
        _fail("symlink_escape", "repository root may not be a symbolic link")
    governed = _canonical_paths(
        governed_repository_roots,
        field_name="governed_repository_roots",
        maximum=DEFAULT_MAX_GOVERNED_REPOSITORY_ROOTS,
        allow_empty=True,
    )
    for namespace in governed:
        current = root
        for component in namespace.split("/"):
            current = current / component
            try:
                info = current.lstat()
            except OSError as exc:
                raise ProductionContextSliceError(
                    "governed evidence root is unavailable",
                    reason_code="repository_unavailable",
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _fail(
                    "symlink_escape",
                    "governed evidence roots must be real checkout directories",
                )


@dataclass(frozen=True, slots=True)
class ProductionEvidenceAuthorityManifest:
    """Immutable, bounded read authority attached to a production packet.

    This authority is deliberately separate from ``ProductionContextSlice``:
    evidence inputs may span governed read-only Git roots, while patch effects
    and the writer remain constrained to the task's exact output paths.
    """

    _payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        detached = json.loads(canonical_json_bytes(dict(self._payload)))
        object.__setattr__(self, "_payload", MappingProxyType(detached))

    @property
    def evidence_cid(self) -> str:
        return str(self._payload["evidence_cid"])

    @property
    def source_count(self) -> int:
        return len(self._payload["sources"])

    @property
    def provider_ready(self) -> bool:
        return self._payload["readiness"]["provider_ready"] is True

    @property
    def expansion_handles(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            MappingProxyType(dict(item))
            for item in self._payload["expansion_handles"]
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(dict(self._payload)))

    def provider_payload(self) -> dict[str, Any]:
        return {"evidence_authority": self.to_dict()}


def _tree_object_optional(
    root: Path,
    commit: str,
    relative: str,
) -> tuple[str, str, str] | None:
    raw = _git_bytes(
        root,
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        relative,
    )
    entries = [entry for entry in raw.split(b"\x00") if entry]
    if not entries:
        return None
    if len(entries) != 1:
        _fail("evidence_input_ambiguous", "evidence input is not one Git object")
    try:
        header, encoded_path = entries[0].split(b"\t", 1)
        mode, object_type, oid = header.decode("ascii").split(" ", 2)
        entry_path = encoded_path.decode("utf-8", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise ProductionContextSliceError(
            "evidence Git entry is malformed",
            reason_code="repository_malformed",
        ) from exc
    if entry_path != relative or not _GIT_OID_RE.fullmatch(oid):
        _fail("repository_malformed", "evidence Git identity is malformed")
    return mode, object_type, oid


def _assert_safe_evidence_directory(root: Path, relative: str) -> Path:
    """Resolve a tracked directory without following links/repository escapes."""

    if not relative:
        return root
    current = root
    for part in relative.split("/"):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            _fail("evidence_source_missing", "evidence directory is missing")
        except OSError as exc:
            raise ProductionContextSliceError(
                "evidence directory is unavailable",
                reason_code="evidence_source_unavailable",
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_escape", "symlinks are forbidden in evidence scope")
        if not stat.S_ISDIR(info.st_mode):
            _fail("path_invalid", "evidence directory path is not a directory")
        try:
            (current / ".git").lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ProductionContextSliceError(
                "nested evidence repository boundary is unreadable",
                reason_code="nested_repository_escape",
            ) from exc
        else:
            _fail(
                "nested_repository_escape",
                "evidence crossed an undeclared nested repository boundary",
            )
    return current


def _evidence_repository_roots(
    root: Path,
    *,
    baseline_ref: str,
    governed_repository_roots: Sequence[str],
) -> tuple[_EvidenceRepositoryRoot, ...]:
    outer = _repository_binding(root, baseline_ref)
    governed = _canonical_paths(
        governed_repository_roots,
        field_name="governed_repository_roots",
        maximum=DEFAULT_MAX_GOVERNED_REPOSITORY_ROOTS,
        allow_empty=True,
    )
    for left in governed:
        if any(
            right != left and right.startswith(left + "/")
            for right in governed
        ):
            _fail(
                "repository_root_overlap",
                "governed repository roots must not overlap",
            )
    result = [
        _EvidenceRepositoryRoot(
            namespace=".",
            path=root,
            binding=MappingProxyType(outer),
        )
    ]
    for namespace in governed:
        entry = _tree_object_optional(root, outer["baseline_commit"], namespace)
        if entry is None or entry[0] != "160000" or entry[1] != "commit":
            _fail(
                "gitlink_binding_missing",
                "governed evidence root is not an exact parent gitlink",
            )
        gitlink_oid = entry[2]
        # A configured namespace is not permission to follow a checkout
        # symlink.  Walk every component with lstat before Git observes it.
        child_candidate = root
        for component in namespace.split("/"):
            child_candidate = child_candidate / component
            try:
                info = child_candidate.lstat()
            except OSError as exc:
                raise ProductionContextSliceError(
                    "governed evidence root is unavailable",
                    reason_code="repository_unavailable",
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _fail(
                    "symlink_escape",
                    "governed evidence roots must be real checkout directories",
                )
        try:
            child_resolved = child_candidate.resolve(strict=True)
            child_resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProductionContextSliceError(
                "governed evidence root escaped the outer worktree",
                reason_code="symlink_escape",
            ) from exc
        child = _repository_root(child_candidate)
        child_binding = _repository_binding(child, gitlink_oid)
        if child_binding["baseline_commit"] != gitlink_oid:
            _fail("gitlink_stale", "governed evidence root differs from its gitlink")
        result.append(
            _EvidenceRepositoryRoot(
                namespace=namespace,
                path=child,
                binding=MappingProxyType(child_binding),
                parent_gitlink_oid=gitlink_oid,
            )
        )
    return tuple(result)


def _read_external_regular_nofollow(
    path: str | Path,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read one operator-selected absolute file without following any link."""

    raw = Path(path)
    if not raw.is_absolute() or str(raw) != str(raw).strip():
        _fail("launch_authority_invalid", "launch authority path must be absolute")
    parts = raw.parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    try:
        current = os.open(parts[0], directory_flags)
        descriptors.append(current)
        for component in parts[1:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
        ):
            _fail(
                "launch_authority_invalid",
                "launch authority must be an operator-owned mode-0600 regular file",
            )
        if info.st_size < 2 or info.st_size > maximum_bytes:
            _fail("launch_authority_invalid", "launch authority size is invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                _fail("launch_authority_invalid", "launch authority exceeds its bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ):
            _fail("launch_authority_stale", "launch authority changed during read")
        return b"".join(chunks)
    except ProductionContextSliceError:
        raise
    except OSError as exc:
        raise ProductionContextSliceError(
            "launch authority is unavailable",
            reason_code="launch_authority_unavailable",
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _strict_json_object(value: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductionContextSliceError(
            f"{label} is not strict JSON",
            reason_code="launch_authority_invalid",
        ) from exc
    if not isinstance(parsed, dict):
        _fail("launch_authority_invalid", f"{label} must be a JSON object")
    return parsed


@production_evidence_scan_budgeted
def load_verified_production_provider_launch_authority(
    *,
    receipt_path: str | Path,
    expected_receipt_content_id: str,
    repo_root: str | Path,
    governed_repository_roots: Sequence[str],
    baseline_ref: str = "HEAD",
) -> dict[str, str]:
    """Verify the external launch receipt that selects candidate authority.

    The expected receipt CID is supplied by the exact launch argv.  The
    receipt, in turn, binds the final outer control commit/tree, all four
    governed direct gitlinks and checkout HEADs, and one protected candidate
    appendix.  This prevents a valid appendix from replaying on a descendant
    control baseline while avoiding an appendix/containing-commit CID cycle.
    """

    expected = str(expected_receipt_content_id or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        _fail("launch_authority_missing", "exact launch receipt CID is required")
    receipt_bytes = _read_external_regular_nofollow(
        receipt_path,
        maximum_bytes=DEFAULT_MAX_LAUNCH_AUTHORITY_BYTES,
    )
    receipt = _strict_json_object(receipt_bytes, label="launch authority receipt")
    recorded_cid = str(receipt.get("content_id") or "")
    unsigned = {key: value for key, value in receipt.items() if key != "content_id"}
    computed_cid = "sha256:" + hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if recorded_cid != expected or computed_cid != expected:
        _fail("launch_authority_mismatch", "launch receipt CID differs from argv authority")
    if (
        receipt.get("schema") != _CONTROL_LAUNCH_RECEIPT_SCHEMA
        or receipt.get("status") != "admitted"
    ):
        _fail("launch_authority_invalid", "launch receipt is not admitted")

    governed = _canonical_paths(
        governed_repository_roots,
        field_name="governed_repository_roots",
        maximum=DEFAULT_MAX_GOVERNED_REPOSITORY_ROOTS,
        allow_empty=True,
    )
    if len(governed) != 4:
        _fail(
            "launch_authority_mismatch",
            "candidate launch authority requires exactly four governed roots",
        )
    _preflight_configured_governed_root_paths(repo_root, governed)
    root = _repository_root(repo_root)
    repositories = _evidence_repository_roots(
        root,
        baseline_ref=baseline_ref,
        governed_repository_roots=governed,
    )
    control = receipt.get("control")
    if not isinstance(control, Mapping):
        _fail("launch_authority_invalid", "launch control binding is absent")
    outer = repositories[0]
    if (
        control.get("commit") != outer.binding["baseline_commit"]
        or control.get("tree") != outer.binding["baseline_tree"]
        or control.get("clean") is not True
    ):
        _fail("launch_authority_stale", "launch control commit/tree is stale")
    _assert_repository_tracked_clean(root)

    raw_repository_records = receipt.get("repositories")
    if not isinstance(raw_repository_records, Mapping):
        _fail("launch_authority_invalid", "launch repository bindings are absent")
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_record in raw_repository_records.values():
        if not isinstance(raw_record, Mapping):
            _fail("launch_authority_invalid", "launch repository binding is malformed")
        record_path = str(raw_record.get("path") or "")
        if record_path in records_by_path:
            _fail("launch_authority_invalid", "launch repository path is duplicated")
        records_by_path[record_path] = raw_record
    if set(records_by_path) != set(governed):
        _fail("launch_authority_mismatch", "launch repository forest differs")
    for repository in repositories[1:]:
        record = records_by_path[repository.namespace]
        if (
            record.get("gitlink") != repository.parent_gitlink_oid
            or record.get("head") != repository.binding["baseline_commit"]
            or record.get("clean") is not True
        ):
            _fail("launch_authority_stale", "governed launch binding is stale")
        _assert_repository_tracked_clean(repository.path)

    candidate = receipt.get("candidate_authority")
    sealed = receipt.get("sealed_post_commit_receipt_bindings")
    candidate_keys = frozenset(
        {
            "appendix_path",
            "appendix_cid",
            "signer_identity_did",
            "board_projection_id",
        }
    )
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != candidate_keys
        or not isinstance(sealed, Mapping)
        or sealed.get("candidate_authority_appendix_cid")
        != candidate.get("appendix_cid")
    ):
        _fail("launch_authority_invalid", "candidate launch binding is malformed")
    board_namespace = str(receipt.get("board_namespace") or "").strip()
    result = {
        "appendix_path": str(candidate.get("appendix_path") or ""),
        "appendix_cid": str(candidate.get("appendix_cid") or ""),
        "signer_identity_did": str(candidate.get("signer_identity_did") or ""),
        "board_projection_id": str(candidate.get("board_projection_id") or ""),
        "board_namespace": board_namespace,
        "launch_receipt_content_id": expected,
    }
    if (
        not board_namespace
        or not result["appendix_path"]
        or not result["appendix_cid"].startswith("b")
        or not result["signer_identity_did"].startswith("did:key:")
        or not result["board_projection_id"]
    ):
        _fail("launch_authority_invalid", "candidate launch values are invalid")
    return result


def _evidence_root_for_path(
    path: str,
    roots: Sequence[_EvidenceRepositoryRoot],
) -> tuple[_EvidenceRepositoryRoot, str]:
    matches = [
        root
        for root in roots
        if root.namespace != "."
        and (path == root.namespace or path.startswith(root.namespace + "/"))
    ]
    selected = max(matches, key=lambda item: len(item.namespace), default=roots[0])
    if selected.namespace == ".":
        return selected, path
    relative = path[len(selected.namespace) :].lstrip("/")
    return selected, relative


def _python_evidence_inventory_summary(value: bytes) -> dict[str, Any]:
    """Return a compact deterministic import/export index, never source text."""

    imports: set[str] = set()
    symbols: set[str] = set()
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return {
            "imports": [],
            "language": "python",
            "parse_status": "non_utf8",
            "public_symbols": [],
            "truncated": False,
        }
    try:
        module = ast.parse(text)
    except (SyntaxError, ValueError, OverflowError):
        return {
            "imports": [],
            "language": "python",
            "parse_status": "invalid",
            "public_symbols": [],
            "truncated": False,
        }
    for node in module.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * int(node.level or 0)
            imports.add(prefix + str(node.module or ""))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name and not node.name.startswith("_"):
                symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    symbols.add(target.id)
    ordered_imports = sorted(imports)
    ordered_symbols = sorted(symbols)
    truncated = (
        len(ordered_imports) > DEFAULT_MAX_EVIDENCE_SUMMARY_ITEMS
        or len(ordered_symbols) > DEFAULT_MAX_EVIDENCE_SUMMARY_ITEMS
    )
    return {
        "imports": ordered_imports[:DEFAULT_MAX_EVIDENCE_SUMMARY_ITEMS],
        "language": "python",
        "parse_status": "parsed",
        "public_symbols": ordered_symbols[:DEFAULT_MAX_EVIDENCE_SUMMARY_ITEMS],
        "truncated": truncated,
    }


def _git_blob_size(repository: Path, oid: str) -> int:
    raw = _git(repository, "cat-file", "-s", oid).strip()
    try:
        size = int(raw)
    except ValueError as exc:
        raise ProductionContextSliceError(
            "Git blob size is malformed",
            reason_code="repository_malformed",
        ) from exc
    if size < 0:
        _fail("repository_malformed", "Git blob size is negative")
    return size


def _provider_path_exclusion(
    path: str,
    *,
    protected_paths: Sequence[str] = (),
) -> str:
    """Return a fixed disclosure category for provider-forbidden paths."""

    lowered = path.casefold()
    parts = lowered.split("/")
    if any(
        path == protected or path.startswith(protected + "/")
        for protected in protected_paths
    ):
        return "supervisor_protected"
    if any(part == ".git" for part in parts):
        return "git_administration"
    if "artifacts" in parts:
        # Generated/dependency artifacts are admitted through their own
        # verified receipt channel, never by ambient directory/ref scanning.
        return "generated_artifact"
    if parts[-1] in _SECRET_BASENAMES:
        return "secret_path"
    if (
        lowered.endswith(".todo.md")
        or "task-board" in lowered
        or "task_board" in lowered
        or "control-launch-receipt" in lowered
        or "provider-receipt" in lowered
        or "expansion-receipt" in lowered
    ):
        return "supervisor_control"
    if parts and parts[0] in {
        "state",
        "logs",
        "worktrees",
        "coordination",
        "scheduler",
    }:
        return "generated_control_or_state"
    if (
        any(part in {"hidden", "private", "answers", "expected_answers"} for part in parts)
        and any("benchmark" in part or "fixture" in part for part in parts)
    ):
        return "hidden_evaluation"
    return ""


def _directory_evidence_candidates(
    repository: _EvidenceRepositoryRoot,
    *,
    relative: str,
    declaration_path: str,
    maximum: int,
    protected_paths: Sequence[str] = (),
    summary_budget: list[int] | None = None,
    maximum_blob_bytes: int = DEFAULT_MAX_EVIDENCE_SOURCE_BYTES,
) -> tuple[
    tuple[_EvidenceCandidate, ...],
    tuple[Mapping[str, Any], ...],
    str,
    Mapping[str, int],
    str,
]:
    _assert_safe_evidence_directory(repository.path, relative)
    commit = repository.binding["baseline_commit"]
    tree_oid = (
        repository.binding["baseline_tree"]
        if not relative
        else str((_tree_object_optional(repository.path, commit, relative) or ("", "", ""))[2])
    )
    raw = _git_bytes(
        repository.path,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        *( ("--", relative) if relative else () ),
    )
    entries = [entry for entry in raw.split(b"\x00") if entry]
    scan_budget = _active_evidence_scan_budget()
    if scan_budget is not None:
        scan_budget.consume_entries(len(entries))
    if len(entries) > maximum:
        _fail(
            "evidence_inventory_too_broad",
            "evidence directory exceeds its immutable inventory bound",
        )
    candidates: list[_EvidenceCandidate] = []
    inventory: list[dict[str, str]] = []
    excluded_counts: dict[str, int] = {}
    excluded_identities: list[dict[str, str]] = []
    for raw_entry in entries:
        try:
            header, encoded_path = raw_entry.split(b"\t", 1)
            mode, object_type, oid = header.decode("ascii").split(" ", 2)
            repository_path = encoded_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise ProductionContextSliceError(
                "evidence directory inventory is malformed",
                reason_code="repository_malformed",
            ) from exc
        global_path = (
            repository_path
            if repository.namespace == "."
            else f"{repository.namespace}/{repository_path}"
        )
        exclusion = _provider_path_exclusion(
            global_path,
            protected_paths=protected_paths,
        )
        if exclusion:
            excluded_counts[exclusion] = excluded_counts.get(exclusion, 0) + 1
            excluded_identities.append(
                {
                    "class": exclusion,
                    "git_mode": mode,
                    "git_object_oid": oid,
                    "path_cid": content_identity({"path": global_path}),
                }
            )
            continue
        if (
            object_type != "blob"
            or mode not in {"100644", "100755"}
            or not _GIT_OID_RE.fullmatch(oid)
        ):
            reason = (
                "symlink"
                if mode == "120000"
                else "non_regular_git_object"
            )
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            excluded_identities.append(
                {
                    "class": reason,
                    "git_mode": mode,
                    "git_object_oid": oid,
                    "path_cid": content_identity({"path": global_path}),
                }
            )
            continue
        try:
            canonical_global = _canonical_path(global_path)
            canonical_local = _canonical_path(repository_path)
        except ProductionContextSliceError as exc:
            reason = f"forbidden_path:{exc.reason_code}"
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            excluded_identities.append(
                {
                    "class": reason,
                    "git_mode": mode,
                    "git_object_oid": oid,
                    "path_cid": content_identity({"path": global_path}),
                }
            )
            continue
        blob_size = _git_blob_size(repository.path, oid)
        if blob_size > maximum_blob_bytes:
            reason = "oversize"
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            excluded_identities.append(
                {
                    "class": reason,
                    "git_mode": mode,
                    "git_object_oid": oid,
                    "path_cid": content_identity({"path": canonical_global}),
                }
            )
            continue
        blob_bytes = _git_bytes(repository.path, "cat-file", "blob", oid)
        if len(blob_bytes) != blob_size:
            _fail("repository_malformed", "Git blob length changed during evidence scan")
        _assert_safe_worktree_path(repository.path, canonical_local)
        current_bytes = _read_repository_regular_nofollow(
            repository.path,
            canonical_local,
            maximum_bytes=maximum_blob_bytes,
        )
        if current_bytes != blob_bytes:
            _fail("evidence_blob_stale", "evidence inventory differs from its Git blob")
        try:
            text = blob_bytes.decode("utf-8", errors="strict")
            _assert_secret_free(text)
        except UnicodeDecodeError:
            reason = "non_utf8"
        except ProductionContextSliceError as exc:
            if exc.reason_code != "secret_detected":
                raise
            reason = "secret_content"
        else:
            reason = "binary" if "\x00" in text else ""
        if reason:
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            excluded_identities.append(
                {
                    "class": reason,
                    "git_mode": mode,
                    "git_object_oid": oid,
                    "path_cid": content_identity({"path": canonical_global}),
                }
            )
            continue
        inventory_entry: dict[str, Any] = {
            "git_mode": mode,
            "git_object_oid": oid,
            "object_type": object_type,
            "path": canonical_global,
            "byte_length": blob_size,
        }
        inventory.append(inventory_entry)
        can_summarize = (
            canonical_local.endswith(".py")
            and blob_size <= DEFAULT_MAX_EVIDENCE_SUMMARY_BYTES
            and (
                summary_budget is None
                or summary_budget[0] >= blob_size
            )
        )
        if can_summarize:
            if summary_budget is not None:
                summary_budget[0] -= blob_size
            inventory_entry["summary"] = _python_evidence_inventory_summary(
                blob_bytes
            )
        else:
            inventory_entry["summary"] = {
                "imports": [],
                "language": "python" if canonical_local.endswith(".py") else "other",
                "parse_status": (
                    "oversize" if canonical_local.endswith(".py") else "not_applicable"
                ),
                "public_symbols": [],
                "truncated": False,
            }
        inventory_entry["provider_visibility"] = "source_eligible"
        candidates.append(
            _EvidenceCandidate(
                path=canonical_global,
                repository_namespace=repository.namespace,
                repository_path=canonical_local,
                git_mode=mode,
                git_blob_oid=oid,
                byte_length=blob_size,
            )
        )
    if not candidates:
        _fail(
            "evidence_input_empty",
            "evidence directory contains no eligible tracked source blobs",
        )
    inventory_cid = content_identity(
        {
            "declaration_path": declaration_path,
            "entries": inventory,
            "repository_cid": repository.binding["repository_cid"],
            "tree_oid": tree_oid,
        }
    )
    return (
        tuple(sorted(candidates, key=lambda item: item.path)),
        tuple(MappingProxyType(dict(item)) for item in inventory),
        inventory_cid,
        MappingProxyType(dict(sorted(excluded_counts.items()))),
        content_identity(
            {
                "declaration_path": declaration_path,
                "excluded_identities": excluded_identities,
                "repository_cid": repository.binding["repository_cid"],
            }
        ),
    )


def _canonical_evidence_candidate_ids(
    declarations: Sequence[str],
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(declarations, (str, bytes, bytearray)):
        _fail("evidence_ref_invalid", "evidence candidate IDs must be a sequence")
    if len(declarations) > maximum:
        _fail("evidence_ref_too_broad", "evidence candidate IDs exceed their bound")
    parsed: list[str] = []
    for raw in declarations:
        if (
            not isinstance(raw, str)
            or raw != raw.strip()
            or not _CANDIDATE_ID_RE.fullmatch(raw)
        ):
            _fail(
                "evidence_ref_invalid",
                "task evidence must name one canonical candidate ID, never a Git ref",
            )
        parsed.append(raw)
    if len(set(parsed)) != len(parsed):
        _fail("evidence_ref_invalid", "evidence candidate IDs contain duplicates")
    return tuple(sorted(parsed))


@production_evidence_scan_budgeted
def load_production_candidate_ref_authority_appendix(
    *,
    repo_root: str | Path,
    authority_path: str,
    expected_appendix_cid: str,
    baseline_ref: str = "HEAD",
) -> dict[str, Any]:
    """Load one launch-selected signed appendix from the exact outer baseline.

    The task may name only candidate IDs.  The authority records themselves
    live in a supervisor-protected, tracked canonical-JSON file selected by the
    operator launch configuration.  The file does not contain its own baseline
    commit/tree, so tracking it cannot create a content-address fixed-point;
    the external launch receipt separately binds this CID and the final control
    commit/tree.
    """

    root = _repository_root(repo_root)
    relative = _canonical_path(authority_path)
    expected = str(expected_appendix_cid or "").strip()
    if not expected.startswith("b") or len(expected) > 128:
        _fail(
            "evidence_ref_authority_missing",
            "launch configuration lacks the exact candidate appendix CID",
        )
    binding = _repository_binding(root, baseline_ref)
    _mode, _oid, baseline_bytes = _tree_entry(
        root,
        binding["baseline_commit"],
        relative,
    )
    if len(baseline_bytes) > DEFAULT_MAX_EVIDENCE_SOURCE_BYTES:
        _fail(
            "evidence_ref_authority_invalid",
            "candidate authority appendix exceeds its byte bound",
        )
    _assert_safe_worktree_path(root, relative)
    current_bytes = _read_repository_regular_nofollow(
        root,
        relative,
        maximum_bytes=DEFAULT_MAX_EVIDENCE_SOURCE_BYTES,
    )
    if current_bytes != baseline_bytes:
        _fail(
            "evidence_ref_authority_stale",
            "candidate authority appendix differs from its baseline blob",
        )
    try:
        parsed = json.loads(baseline_bytes.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionContextSliceError(
            "candidate authority appendix is not canonical JSON",
            reason_code="evidence_ref_authority_invalid",
        ) from exc
    if not isinstance(parsed, Mapping) or canonical_json_bytes(dict(parsed)) != baseline_bytes:
        _fail(
            "evidence_ref_authority_invalid",
            "candidate authority appendix bytes are not canonical",
        )
    payload = json.loads(canonical_json_bytes(dict(parsed)))
    appendix_cid = str(payload.get("appendix_cid") or "")
    if appendix_cid != expected:
        _fail(
            "evidence_ref_authority_mismatch",
            "candidate authority appendix differs from launch authority",
        )
    return payload


def _verified_candidate_ref_authorities(
    appendix: Mapping[str, Any] | None,
    *,
    declarations: Sequence[str],
    repositories: Sequence[_EvidenceRepositoryRoot],
    task_binding: Mapping[str, str],
    expected_board_namespace: str,
    expected_board_projection_id: str,
    expected_signer_did: str,
) -> Mapping[str, Mapping[str, Any]]:
    """Verify a signed, launch-bound candidate preservation appendix.

    The appendix is operator authority supplied by supervisor configuration,
    never task metadata.  It binds the already-final control/board identity;
    a later launch receipt may bind ``appendix_cid`` without either object
    pointing back at the other (and therefore without a content-ID cycle).
    """

    if not declarations:
        if appendix:
            _fail("evidence_ref_authority_mismatch", "unused candidate authority supplied")
        return MappingProxyType({})
    if (
        not expected_board_namespace
        or not expected_board_projection_id
        or not expected_signer_did
    ):
        _fail(
            "evidence_ref_authority_missing",
            "candidate refs require exact launch/board preservation authority",
        )
    appendix_keys = frozenset(
        {
            "schema",
            "board_namespace",
            "board_projection_id",
            "records",
            "signer_identity_did",
            "signature",
            "appendix_cid",
        }
    )
    if not isinstance(appendix, Mapping) or set(appendix) != appendix_keys:
        _fail("evidence_ref_authority_invalid", "candidate appendix shape is invalid")
    appendix_payload = dict(appendix)
    appendix_cid = str(appendix_payload.pop("appendix_cid") or "")
    signature = str(appendix_payload.pop("signature") or "")
    signer_did = str(appendix_payload.get("signer_identity_did") or "")
    records = appendix_payload.get("records")
    if (
        appendix_payload.get("schema") != CANDIDATE_REF_AUTHORITY_APPENDIX_SCHEMA
        or appendix_payload.get("board_namespace") != expected_board_namespace
        or appendix_payload.get("board_projection_id")
        != expected_board_projection_id
        or signer_did != expected_signer_did
        or not isinstance(records, list)
        or len(records) > DEFAULT_MAX_EVIDENCE_REFS
        or appendix_cid
        != content_identity({**appendix_payload, "signature": signature})
    ):
        _fail("evidence_ref_authority_mismatch", "candidate appendix binding differs")
    try:
        verify_did_key_signature(
            identity_did=signer_did,
            payload=appendix_payload,
            signature=signature,
        )
    except Exception as exc:
        raise ProductionContextSliceError(
            "candidate authority signature is invalid",
            reason_code="evidence_ref_signature_invalid",
        ) from exc
    repositories_by_namespace = {item.namespace: item for item in repositories}
    expected_keys = frozenset(
        {
            "record_cid",
            "candidate_id",
            "authority_mode",
            "source_board_namespace",
            "source_task_id",
            "source_task_cid",
            "target_task_id",
            "target_task_cid",
            "repository_namespace",
            "preserved_ref",
            "origin_base_commit",
            "origin_base_tree",
            "candidate_commit",
            "candidate_tree",
            "merge_base",
            "ancestry_verified",
            "implementation_started_event_id",
            "worktree_preserved_event_id",
        }
    )
    verified: dict[str, Mapping[str, Any]] = {}
    declaration_set = set(declarations)
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            _fail("evidence_ref_authority_invalid", "candidate authority shape is invalid")
        record = dict(raw)
        record_cid = str(record.pop("record_cid") or "")
        if record_cid != content_identity(record):
            _fail("evidence_ref_authority_invalid", "candidate authority CID is invalid")
        namespace = str(record.get("repository_namespace") or "")
        ref_name = str(record.get("preserved_ref") or "")
        commit = str(record.get("candidate_commit") or "")
        candidate_id = str(record.get("candidate_id") or "")
        repository = repositories_by_namespace.get(namespace)
        if (
            candidate_id not in declaration_set
            or not _CANDIDATE_ID_RE.fullmatch(candidate_id)
            or repository is None
            or candidate_id in verified
            or not _FULL_GIT_REF_RE.fullmatch(ref_name)
            or len(ref_name) > 512
            or ".." in ref_name
            or "//" in ref_name
            or ref_name.endswith(("/", ".", ".lock"))
            or "@{" in ref_name
            or not _GIT_OID_RE.fullmatch(commit)
        ):
            _fail("evidence_ref_authority_mismatch", "candidate authority is out of scope")
        mode = record.get("authority_mode")
        if (
            mode not in {"same_task", "operator_signed_cross_task"}
            or record.get("target_task_id") != task_binding["task_id"]
            or record.get("target_task_cid") != task_binding["canonical_task_cid"]
            or record.get("ancestry_verified") is not True
            or not str(record.get("implementation_started_event_id") or "")
            or not str(record.get("worktree_preserved_event_id") or "")
        ):
            _fail("evidence_ref_authority_mismatch", "candidate authority binding differs")
        if mode == "same_task" and (
            record.get("source_board_namespace") != expected_board_namespace
            or record.get("source_task_id") != task_binding["task_id"]
            or record.get("source_task_cid") != task_binding["canonical_task_cid"]
        ):
            _fail("evidence_ref_authority_mismatch", "same-task candidate lineage differs")
        if mode == "operator_signed_cross_task" and (
            not str(record.get("source_board_namespace") or "")
            or not str(record.get("source_task_id") or "")
            or not str(record.get("source_task_cid") or "")
        ):
            _fail("evidence_ref_authority_mismatch", "cross-task source lineage is absent")
        oid_fields = (
            "origin_base_commit",
            "origin_base_tree",
            "candidate_commit",
            "candidate_tree",
            "merge_base",
        )
        if any(not _GIT_OID_RE.fullmatch(str(record.get(name) or "")) for name in oid_fields):
            _fail("evidence_ref_authority_invalid", "candidate authority OID is invalid")
        candidate_tree = _git(
            repository.path, "rev-parse", "--verify", f"{commit}^{{tree}}"
        ).strip()
        origin = str(record.get("origin_base_commit") or "")
        origin_tree = _git(
            repository.path, "rev-parse", "--verify", f"{origin}^{{tree}}"
        ).strip()
        merge_base = _git(repository.path, "merge-base", origin, commit).strip()
        launch_merge_base = _git(
            repository.path,
            "merge-base",
            origin,
            repository.binding["baseline_commit"],
        ).strip()
        if (
            candidate_tree != record.get("candidate_tree")
            or origin_tree != record.get("origin_base_tree")
            or merge_base != origin
            or launch_merge_base != origin
            or record.get("merge_base") != merge_base
        ):
            _fail("evidence_ref_authority_stale", "candidate ancestry/tree is stale")
        verified[candidate_id] = MappingProxyType(
            {
                **record,
                "record_cid": record_cid,
                "authority_appendix_cid": appendix_cid,
            }
        )
    if set(verified) != declaration_set:
        _fail("evidence_ref_authority_missing", "candidate ref lacks preservation authority")
    return MappingProxyType(verified)


def _ref_path_exclusion(
    path: str,
    *,
    protected_paths: Sequence[str] = (),
) -> str:
    return _provider_path_exclusion(path, protected_paths=protected_paths)


def _path_intersects_any(path: str, candidates: Sequence[str]) -> bool:
    return any(
        path == candidate
        or path.startswith(candidate + "/")
        or candidate.startswith(path + "/")
        for candidate in candidates
    )


def _parse_name_status(value: bytes) -> tuple[tuple[str, str], ...]:
    fields = [field for field in value.split(b"\x00") if field]
    result: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        if b"\t" in field:
            raw_status, raw_path = field.split(b"\t", 1)
            index += 1
        else:
            if index + 1 >= len(fields):
                _fail("repository_malformed", "ref change inventory is malformed")
            raw_status, raw_path = field, fields[index + 1]
            index += 2
        try:
            status = raw_status.decode("ascii", errors="strict")
            if len(raw_path) > 4096:
                _fail(
                    "evidence_ref_path_invalid",
                    "candidate ref contains an overlong path",
                )
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ProductionContextSliceError(
                "ref change inventory is not canonical UTF-8",
                reason_code="repository_malformed",
            ) from exc
        if status not in {"A", "D", "M", "T", "U", "X", "B"}:
            _fail("repository_malformed", "unsupported ref change status")
        result.append((status, path))
    return tuple(result)


def _build_evidence_ref_bindings(
    *,
    repositories: Sequence[_EvidenceRepositoryRoot],
    evidence_refs: Sequence[str],
    priority_paths: Sequence[str],
    evidence_inputs: Sequence[str],
    protected_paths: Sequence[str],
    candidate_ref_authorities: Mapping[str, Mapping[str, Any]],
    context_round: int,
    expansion_selections: Mapping[str, int],
    max_refs: int,
    max_ref_diffs: int,
    max_ref_diff_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    declarations = _canonical_evidence_candidate_ids(
        evidence_refs,
        maximum=max_refs,
    )
    if not declarations:
        return [], [], 0
    priorities = _canonical_paths(
        priority_paths,
        field_name="evidence_priority_paths",
        maximum=DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES,
        allow_empty=True,
    )
    repository_by_namespace = {item.namespace: item for item in repositories}
    bindings: list[dict[str, Any]] = []
    handles: list[dict[str, Any]] = []
    total_diff_bytes = 0
    allocated_diff_bytes = 0
    for candidate_id in declarations:
        authority_key = f"ref:{candidate_id}"
        preservation = candidate_ref_authorities.get(candidate_id)
        if preservation is None:
            _fail("evidence_ref_authority_missing", "candidate preservation is absent")
        namespace = str(preservation["repository_namespace"])
        ref_name = str(preservation["preserved_ref"])
        declared_commit = str(preservation["candidate_commit"])
        repository = repository_by_namespace.get(namespace)
        if repository is None:
            _fail(
                "evidence_ref_root_forbidden",
                "evidence ref does not name a configured governed root",
            )
        resolved_commit = _git(
            repository.path,
            "rev-parse",
            "--verify",
            f"{ref_name}^{{commit}}",
        ).strip()
        if resolved_commit != declared_commit:
            _fail("evidence_ref_stale", "evidence ref tip differs from its exact binding")
        ref_tree = _git(
            repository.path,
            "rev-parse",
            "--verify",
            f"{resolved_commit}^{{tree}}",
        ).strip()
        if not _GIT_OID_RE.fullmatch(ref_tree):
            _fail("repository_malformed", "evidence ref tree is malformed")
        origin_base_commit = str(preservation["origin_base_commit"])
        raw_changes = _git_bytes(
            repository.path,
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            origin_base_commit,
            resolved_commit,
            "--",
        )
        parsed_changes = _parse_name_status(raw_changes)
        scan_budget = _active_evidence_scan_budget()
        if scan_budget is not None:
            scan_budget.consume_entries(len(parsed_changes))
        if len(parsed_changes) > DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES:
            _fail("evidence_ref_too_broad", "evidence ref change set exceeds its bound")
        changed: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        excluded_counts: dict[str, int] = {}
        excluded_identities: list[dict[str, str]] = []
        for status, raw_path in parsed_changes:
            try:
                local_path = _canonical_path(raw_path)
                global_path = (
                    local_path if namespace == "." else f"{namespace}/{local_path}"
                )
                global_path = _canonical_path(global_path)
            except ProductionContextSliceError as exc:
                reason = f"path:{exc.reason_code}"
                excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
                excluded_identities.append(
                    {
                        "class": reason,
                        "path_sha256": hashlib.sha256(
                            raw_path.encode("utf-8", errors="strict")
                        ).hexdigest(),
                        "status": status,
                    }
                )
                continue
            priority_class = (
                "owned_or_predicted"
                if _path_intersects_any(global_path, priorities)
                else (
                    "evidence_input"
                    if _path_intersects_any(global_path, evidence_inputs)
                    else "other"
                )
            )
            exclusion = _ref_path_exclusion(
                global_path,
                protected_paths=protected_paths,
            )
            if priority_class == "other" and not exclusion:
                exclusion = "unrelated"
            if exclusion:
                excluded_counts[exclusion] = excluded_counts.get(exclusion, 0) + 1
                excluded_identities.append(
                    {
                        "class": exclusion,
                        "path_cid": content_identity({"path": global_path}),
                        "status": status,
                    }
                )
                continue
            old_entry = _tree_object_optional(
                repository.path,
                origin_base_commit,
                local_path,
            )
            new_entry = _tree_object_optional(
                repository.path,
                resolved_commit,
                local_path,
            )
            record: dict[str, Any] = {
                "excluded_class": exclusion,
                "new_git_mode": new_entry[0] if new_entry else "",
                "new_git_object_oid": new_entry[2] if new_entry else "",
                "old_git_mode": old_entry[0] if old_entry else "",
                "old_git_object_oid": old_entry[2] if old_entry else "",
                "path": global_path,
                "priority_class": priority_class,
                "status": status,
            }
            new_blob_size = (
                _git_blob_size(repository.path, new_entry[2])
                if new_entry and new_entry[1] == "blob"
                else 0
            )
            record["new_byte_length"] = new_blob_size
            if (
                new_entry
                and new_entry[1] == "blob"
                and local_path.endswith(".py")
                and new_blob_size <= DEFAULT_MAX_EVIDENCE_SUMMARY_BYTES
            ):
                record["summary"] = _python_evidence_inventory_summary(
                    _git_bytes(repository.path, "cat-file", "blob", new_entry[2])
                )
            else:
                record["summary"] = {
                    "imports": [],
                    "language": "other",
                    "parse_status": "not_applicable",
                    "public_symbols": [],
                    "truncated": False,
                }
            changed.append(record)
            eligible.append(record)
        eligible.sort(
            key=lambda item: (
                {"owned_or_predicted": 0, "evidence_input": 1, "other": 2}[
                    item["priority_class"]
                ],
                item["path"],
            )
        )
        # Classify every actually materializable diff before publishing an
        # expansion handle.  Permanently unavailable/secret/oversize diffs are
        # represented only by aggregate counts and can never advance a context
        # generation to the same visible bytes.
        materializable: list[tuple[dict[str, Any], dict[str, Any]]] = []
        permanently_withheld_count = 0
        private_withheld_paths: dict[str, str] = {}
        for record in eligible:
            local_path = (
                record["path"]
                if namespace == "."
                else record["path"][len(namespace) + 1 :]
            )
            old_size = (
                _git_blob_size(repository.path, record["old_git_object_oid"])
                if record["old_git_object_oid"]
                else 0
            )
            new_size = int(record["new_byte_length"])
            if old_size + new_size > max_ref_diff_bytes:
                permanently_withheld_count += 1
                continue
            diff_bytes = _git_bytes(
                repository.path,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=3",
                "--no-renames",
                origin_base_commit,
                resolved_commit,
                "--",
                local_path,
            )
            if (
                not diff_bytes
                or len(diff_bytes) > max_ref_diff_bytes
                or allocated_diff_bytes + len(diff_bytes) > max_ref_diff_bytes
            ):
                permanently_withheld_count += 1
                continue
            if b"\x00" in diff_bytes:
                permanently_withheld_count += 1
                private_withheld_paths[record["path"]] = "binary_diff"
                continue
            try:
                diff_text = diff_bytes.decode("utf-8", errors="strict")
                _assert_secret_free(diff_text)
            except UnicodeDecodeError:
                permanently_withheld_count += 1
                private_withheld_paths[record["path"]] = "non_utf8_diff"
                continue
            except ProductionContextSliceError:
                permanently_withheld_count += 1
                private_withheld_paths[record["path"]] = "secret_diff"
                continue
            diff = {
                "byte_length": len(diff_bytes),
                "diff_cid": _raw_cid(diff_bytes),
                "path": record["path"],
                "priority_class": record["priority_class"],
                "unified_diff_utf8": diff_text,
            }
            allocated_diff_bytes += len(diff_bytes)
            materializable.append((record, diff))

        if private_withheld_paths:
            for private_path, reason in sorted(private_withheld_paths.items()):
                excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
                excluded_identities.append(
                    {
                        "class": reason,
                        "path_cid": content_identity({"path": private_path}),
                        "status": next(
                            str(item["status"])
                            for item in eligible
                            if item["path"] == private_path
                        ),
                    }
                )
            changed = [
                item for item in changed if item["path"] not in private_withheld_paths
            ]

        # Context generations are cumulative.  Infrastructure retries reuse
        # generation zero byte-for-byte; an explicit, supervisor-admitted
        # expansion grows the visible prefix and never drops earlier evidence.
        window_start = 0
        window_end = min(
            len(materializable),
            (1 + expansion_selections.get(authority_key, 0)) * max_ref_diffs,
        )
        window = materializable[:window_end]
        diffs = [dict(diff) for _record, diff in window]
        selected_paths = [str(record["path"]) for record, _diff in window]
        total_diff_bytes += sum(int(item["byte_length"]) for item in diffs)
        core: dict[str, Any] = {
            "baseline_commit": repository.binding["baseline_commit"],
            "baseline_tree": repository.binding["baseline_tree"],
            "candidate_ref_authority_cid": preservation["record_cid"],
            "candidate_ref_authority_appendix_cid": preservation[
                "authority_appendix_cid"
            ],
            "changed_path_count": len(parsed_changes),
            "changed_paths": changed,
            "candidate_id": candidate_id,
            "declared_commit": declared_commit,
            "diffs": diffs,
            "excluded_path_classes": dict(sorted(excluded_counts.items())),
            "excluded_path_identity_cid": content_identity(
                {
                    "excluded_identities": excluded_identities,
                    "candidate_id": candidate_id,
                    "repository_namespace": namespace,
                }
            ),
            "ref_tree": ref_tree,
            "origin_base_commit": origin_base_commit,
            "origin_base_tree": preservation["origin_base_tree"],
            "repository_namespace": namespace,
            "selection": {
                "eligible_path_count": len(eligible),
                "context_round": context_round,
                "selected_paths": selected_paths,
                "window_end": window_start + len(window),
                "window_start": window_start,
                "withheld_diff_count": permanently_withheld_count,
            },
        }
        binding = {**core, "ref_evidence_cid": content_identity(core)}
        bindings.append(binding)
        omitted = max(0, len(materializable) - len(selected_paths))
        handle_core = {
            "authority_key": authority_key,
            "authorized_candidate_id": candidate_id,
            "eligible_path_count": len(eligible),
            "omitted_diff_count": omitted,
            "ref_evidence_cid": binding["ref_evidence_cid"],
            "selection_generation": expansion_selections.get(authority_key, 0),
        }
        if omitted:
            handles.append(
                {
                    **handle_core,
                    "expansion_cid": content_identity(handle_core),
                    "request_contract": "supervisor-rebuild-ref-window@1",
                    "request_parameters": {
                        "maximum_paths": max_ref_diffs,
                        "next_selection_generation": (
                            expansion_selections.get(authority_key, 0) + 1
                        ),
                    },
                }
            )
    return bindings, handles, total_diff_bytes


def _stabilize_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bind observed token metadata without including the root in itself."""

    observed = -1
    for _attempt in range(8):
        payload["budget"]["evidence_manifest_tokens"] = max(0, observed)
        payload["evidence_cid"] = content_identity(
            {key: value for key, value in payload.items() if key != "evidence_cid"}
        )
        current = _token_estimate(canonical_json_bytes(payload))
        if current == observed:
            return payload
        observed = current
    _fail("budget_invalid", "evidence token metadata did not stabilize")


@production_evidence_scan_budgeted
def build_production_evidence_authority(
    *,
    repo_root: str | Path,
    task_id: str,
    task_payload: Mapping[str, Any],
    evidence_inputs: Sequence[str],
    evidence_refs: Sequence[str] = (),
    candidate_ref_authority_appendix: Mapping[str, Any] | None = None,
    board_namespace: str = "",
    board_projection_id: str = "",
    candidate_authority_signer_did: str = "",
    priority_paths: Sequence[str] = (),
    governed_repository_roots: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
    baseline_ref: str = "HEAD",
    max_evidence_tokens: int = 16_384,
    max_declarations: int = DEFAULT_MAX_EVIDENCE_DECLARATIONS,
    max_source_paths: int = DEFAULT_MAX_EVIDENCE_SOURCE_PATHS,
    max_directory_entries: int = DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES,
    max_source_bytes: int = DEFAULT_MAX_EVIDENCE_SOURCE_BYTES,
    max_refs: int = DEFAULT_MAX_EVIDENCE_REFS,
    max_ref_diffs: int = DEFAULT_MAX_EVIDENCE_REF_DIFFS,
    max_ref_diff_bytes: int = DEFAULT_MAX_EVIDENCE_REF_DIFF_BYTES,
    context_round: int = 0,
    parent_evidence_cid: str = "",
    selected_expansion_cids: Sequence[str] = (),
    expansion_selections: Mapping[str, int] | None = None,
) -> ProductionEvidenceAuthorityManifest:
    """Compile explicit task evidence into bounded immutable read authority.

    Directory declarations are resolved from Git trees, never a filesystem
    walk.  At most ``max_source_paths`` blobs are placed in the initial prompt;
    every remaining authorized blob is represented by a CID-bound expansion
    handle.  Those handles grant no ambient or write access.
    """

    limits = (
        ("max_evidence_tokens", max_evidence_tokens, MAX_PROVIDER_PROMPT_TOKENS),
        ("max_declarations", max_declarations, DEFAULT_MAX_EVIDENCE_DECLARATIONS),
        ("max_source_paths", max_source_paths, DEFAULT_MAX_EVIDENCE_SOURCE_PATHS),
        (
            "max_directory_entries",
            max_directory_entries,
            DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES,
        ),
        ("max_source_bytes", max_source_bytes, DEFAULT_MAX_EVIDENCE_SOURCE_BYTES),
        ("max_refs", max_refs, DEFAULT_MAX_EVIDENCE_REFS),
        ("max_ref_diffs", max_ref_diffs, DEFAULT_MAX_EVIDENCE_REF_DIFFS),
        (
            "max_ref_diff_bytes",
            max_ref_diff_bytes,
            DEFAULT_MAX_EVIDENCE_REF_DIFF_BYTES,
        ),
    )
    for name, value, hard_maximum in limits:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > hard_maximum
        ):
            _fail("budget_invalid", f"{name} exceeds its production bound")
    if (
        isinstance(context_round, bool)
        or not isinstance(context_round, int)
        or context_round < 0
        or context_round > DEFAULT_MAX_EVIDENCE_EXPANSION_ROUND
    ):
        _fail("expansion_round_invalid", "evidence expansion round is invalid")
    selected_handle_cids = tuple(sorted(str(item) for item in selected_expansion_cids))
    raw_selections = dict(expansion_selections or {})
    normalized_selections: dict[str, int] = {}
    for raw_key, raw_value in raw_selections.items():
        if (
            not isinstance(raw_key, str)
            or not raw_key
            or raw_key != raw_key.strip()
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 1
            or raw_value > DEFAULT_MAX_EVIDENCE_EXPANSION_ROUND
        ):
            _fail("expansion_chain_invalid", "evidence expansion selection is invalid")
        normalized_selections[raw_key] = raw_value
    if len(selected_handle_cids) != len(set(selected_handle_cids)) or any(
        not item.startswith("b") or len(item) > 128 for item in selected_handle_cids
    ):
        _fail("expansion_handle_invalid", "selected expansion CIDs are invalid")
    if context_round == 0:
        if parent_evidence_cid or selected_handle_cids or normalized_selections:
            _fail("expansion_chain_invalid", "initial evidence cannot name a parent")
    elif (
        not str(parent_evidence_cid).startswith("b")
        or not selected_handle_cids
    ):
        _fail("expansion_chain_invalid", "expanded evidence requires parent and handle CIDs")

    _preflight_configured_governed_root_paths(
        repo_root,
        governed_repository_roots,
    )
    root = _repository_root(repo_root)
    task_binding = _task_binding(task_id, task_payload)
    declarations = _canonical_paths(
        evidence_inputs,
        field_name="evidence_inputs",
        maximum=max_declarations,
    )
    for index, left in enumerate(declarations):
        if any(
            right.startswith(left + "/") or left.startswith(right + "/")
            for right in declarations[index + 1 :]
        ):
            _fail(
                "evidence_input_overlap",
                "overlapping evidence declarations must be deduplicated by the task board",
            )
    protected = _canonical_paths(
        protected_paths,
        field_name="protected_paths",
        maximum=DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES,
        allow_empty=True,
    )
    repositories = _evidence_repository_roots(
        root,
        baseline_ref=baseline_ref,
        governed_repository_roots=governed_repository_roots,
    )
    ref_declarations = _canonical_evidence_candidate_ids(
        evidence_refs,
        maximum=max_refs,
    )
    verified_ref_authorities = _verified_candidate_ref_authorities(
        candidate_ref_authority_appendix,
        declarations=ref_declarations,
        repositories=repositories,
        task_binding=task_binding,
        expected_board_namespace=str(board_namespace),
        expected_board_projection_id=str(board_projection_id),
        expected_signer_did=str(candidate_authority_signer_did),
    )
    allowed_selection_keys = {
        *(f"input:{item}" for item in declarations),
        *(f"ref:{candidate_id}" for candidate_id in ref_declarations),
    }
    if not set(normalized_selections).issubset(allowed_selection_keys):
        _fail("expansion_chain_invalid", "expansion selection is outside evidence authority")

    declaration_records: list[dict[str, Any]] = []
    candidates_by_declaration: list[tuple[_EvidenceCandidate, ...]] = []
    inventory_by_declaration: list[tuple[Mapping[str, Any], ...]] = []
    excluded_by_declaration: list[Mapping[str, int]] = []
    excluded_identity_cids: list[str] = []
    summary_budget = [DEFAULT_MAX_EVIDENCE_TOTAL_SUMMARY_BYTES]
    total_inventory_entries = 0
    for declaration in declarations:
        repository, relative = _evidence_root_for_path(declaration, repositories)
        direct_exclusion = _provider_path_exclusion(
            declaration,
            protected_paths=protected,
        )
        if direct_exclusion and relative:
            _fail(
                "evidence_path_forbidden",
                "explicit evidence input names provider-forbidden supervisor data",
            )
        commit = repository.binding["baseline_commit"]
        if not relative:
            object_kind = "directory"
            object_oid = repository.binding["baseline_tree"]
            (
                candidates,
                inventory_entries,
                inventory_cid,
                excluded_counts,
                excluded_identity_cid,
            ) = _directory_evidence_candidates(
                repository,
                relative="",
                declaration_path=declaration,
                maximum=max_directory_entries,
                protected_paths=protected,
                summary_budget=summary_budget,
                maximum_blob_bytes=max_source_bytes,
            )
        else:
            entry = _tree_object_optional(repository.path, commit, relative)
            if entry is None:
                _fail(
                    "evidence_input_missing",
                    "declared evidence input is absent from the bound Git tree",
                )
            mode, object_type, object_oid = entry
            if object_type == "tree" and mode == "040000":
                object_kind = "directory"
                (
                    candidates,
                    inventory_entries,
                    inventory_cid,
                    excluded_counts,
                    excluded_identity_cid,
                ) = (
                    _directory_evidence_candidates(
                        repository,
                        relative=relative,
                        declaration_path=declaration,
                        maximum=max_directory_entries,
                        protected_paths=protected,
                        summary_budget=summary_budget,
                        maximum_blob_bytes=max_source_bytes,
                    )
                )
            elif object_type == "blob" and mode in {"100644", "100755"}:
                object_kind = "file"
                _assert_safe_worktree_path(repository.path, relative)
                blob_size = _git_blob_size(repository.path, object_oid)
                candidate = _EvidenceCandidate(
                    path=declaration,
                    repository_namespace=repository.namespace,
                    repository_path=relative,
                    git_mode=mode,
                    git_blob_oid=object_oid,
                    byte_length=blob_size,
                )
                candidates = (candidate,)
                blob_bytes = (
                    _git_bytes(repository.path, "cat-file", "blob", object_oid)
                    if (
                        blob_size <= DEFAULT_MAX_EVIDENCE_SUMMARY_BYTES
                        and summary_budget[0] >= blob_size
                    )
                    else b""
                )
                if blob_bytes:
                    summary_budget[0] -= blob_size
                inventory_entries = (
                    MappingProxyType(
                        {
                            "git_mode": mode,
                            "git_object_oid": object_oid,
                            "object_type": object_type,
                            "path": declaration,
                            "byte_length": blob_size,
                            "provider_visibility": "source_eligible",
                            "summary": (
                                _python_evidence_inventory_summary(blob_bytes)
                                if (
                                    relative.endswith(".py")
                                    and blob_size <= DEFAULT_MAX_EVIDENCE_SUMMARY_BYTES
                                )
                                else {
                                    "imports": [],
                                    "language": (
                                        "python" if relative.endswith(".py") else "other"
                                    ),
                                    "parse_status": (
                                        "oversize"
                                        if relative.endswith(".py")
                                        else "not_applicable"
                                    ),
                                    "public_symbols": [],
                                    "truncated": False,
                                }
                            ),
                        }
                    ),
                )
                inventory_cid = content_identity(
                    {
                        "declaration_path": declaration,
                        "entries": [dict(inventory_entries[0])],
                        "repository_cid": repository.binding["repository_cid"],
                    }
                )
                excluded_counts = MappingProxyType({})
                excluded_identity_cid = content_identity(
                    {
                        "declaration_path": declaration,
                        "excluded_identities": [],
                        "repository_cid": repository.binding["repository_cid"],
                    }
                )
            else:
                reason = (
                    "symlink_escape"
                    if mode == "120000"
                    else "nested_repository_escape"
                )
                _fail(reason, "evidence input is not a regular file or directory")
        candidates_by_declaration.append(candidates)
        inventory_by_declaration.append(inventory_entries)
        excluded_by_declaration.append(excluded_counts)
        excluded_identity_cids.append(excluded_identity_cid)
        total_inventory_entries += len(inventory_entries)
        if total_inventory_entries > max_directory_entries:
            _fail(
                "evidence_inventory_too_broad",
                "combined evidence inventory exceeds its global entry bound",
            )
        declaration_records.append(
            {
                "authorized_path": declaration,
                "excluded_entry_counts": dict(excluded_counts),
                "excluded_identity_cid": excluded_identity_cid,
                "inventory_cid": inventory_cid,
                "inventory_entries": [dict(item) for item in inventory_entries],
                "inventory_entry_count": len(inventory_entries),
                "kind": object_kind,
                "repository_namespace": repository.namespace,
                "tree_or_blob_oid": object_oid,
            }
        )

    # Exact files are operator-selected anchors and therefore precede sampled
    # directory members.  Directory candidates are then interleaved so one
    # large tree cannot consume the complete initial evidence window.
    file_candidates: list[_EvidenceCandidate] = [
        candidates[0]
        for record, candidates in zip(
            declaration_records,
            candidates_by_declaration,
            strict=True,
        )
        if record["kind"] == "file"
    ]
    directory_candidates = [
        candidates
        for record, candidates in zip(
            declaration_records,
            candidates_by_declaration,
            strict=True,
        )
        if record["kind"] == "directory"
    ]
    maximum_candidates = max(
        (len(items) for items in directory_candidates),
        default=0,
    )
    all_ordered_directory_candidates: list[_EvidenceCandidate] = []
    for index in range(maximum_candidates):
        for items in directory_candidates:
            if index < len(items):
                all_ordered_directory_candidates.append(items[index])

    # Explicit file declarations remain visible in every context generation.
    # A generation grows the deterministic directory prefix cumulatively;
    # ordinary implementation/infrastructure retries always rebuild round zero
    # and therefore cannot rotate source bytes accidentally.
    unique_file_candidates = list(
        {item.path: item for item in file_candidates}.values()
    )
    explicit_file_bytes = sum(item.byte_length for item in unique_file_candidates)
    if explicit_file_bytes > max_source_bytes:
        _fail(
            "evidence_budget_exceeded",
            "explicit evidence files exceed the aggregate source-byte budget",
        )
    # A requestable handle may name only a source that can actually be added
    # in a later generation.  Entries beyond the immutable aggregate byte cap
    # remain represented by inventory/exclusion identities, never by a handle
    # that would rebuild to the same provider-visible bytes.
    remaining_source_bytes = max_source_bytes - explicit_file_bytes
    ordered_directory_candidates: list[_EvidenceCandidate] = []
    byte_budget_unavailable_by_path: set[str] = set()
    for candidate in all_ordered_directory_candidates:
        if candidate.byte_length <= remaining_source_bytes:
            ordered_directory_candidates.append(candidate)
            remaining_source_bytes -= candidate.byte_length
        else:
            byte_budget_unavailable_by_path.add(candidate.path)
    directory_slot_count = max_source_paths - len(unique_file_candidates)
    if directory_candidates and directory_slot_count < 1:
        _fail(
            "evidence_scope_too_broad",
            "explicit files leave no bounded directory expansion slot",
        )
    directory_window_start = 0
    directory_window_end = min(
        len(ordered_directory_candidates),
        max(0, directory_slot_count),
    )
    directory_window = ordered_directory_candidates[:directory_window_end]
    ordered_candidates = [*unique_file_candidates, *directory_window]
    initially_selected_paths = {item.path for item in ordered_candidates}
    for declaration, record, candidates in zip(
        declarations,
        declaration_records,
        candidates_by_declaration,
        strict=True,
    ):
        if record["kind"] != "directory":
            continue
        expansion_count = normalized_selections.get(f"input:{declaration}", 0)
        if expansion_count:
            remaining = [
                item
                for item in candidates
                if item.path not in initially_selected_paths
                and item.path not in byte_budget_unavailable_by_path
            ]
            ordered_candidates.extend(
                remaining[: expansion_count * max_source_paths]
            )

    candidate_owners: dict[str, list[str]] = {}
    for declaration, candidates in zip(
        declarations,
        candidates_by_declaration,
        strict=True,
    ):
        for candidate in candidates:
            candidate_owners.setdefault(candidate.path, []).append(declaration)

    repository_by_namespace = {item.namespace: item for item in repositories}
    sources: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    total_source_bytes = 0
    for candidate in ordered_candidates:
        generation_source_limit = max_source_paths * (
            1 + sum(normalized_selections.values())
        )
        if candidate.path in selected_paths or len(sources) >= generation_source_limit:
            continue
        repository = repository_by_namespace[candidate.repository_namespace]
        baseline_size = _git_blob_size(repository.path, candidate.git_blob_oid)
        if baseline_size != candidate.byte_length:
            _fail("evidence_blob_stale", "evidence blob size changed during scan")
        if total_source_bytes + baseline_size > max_source_bytes:
            _fail(
                "evidence_budget_exceeded",
                "selected evidence exceeds its aggregate source-byte budget",
            )
        baseline_bytes = _git_bytes(
            repository.path,
            "cat-file",
            "blob",
            candidate.git_blob_oid,
        )
        if len(baseline_bytes) != baseline_size:
            _fail("repository_malformed", "Git blob length changed during read")
        _assert_safe_worktree_path(
            repository.path,
            candidate.repository_path,
        )
        current_bytes = _read_repository_regular_nofollow(
            repository.path,
            candidate.repository_path,
            maximum_bytes=max_source_bytes,
        )
        if current_bytes != baseline_bytes:
            _fail("evidence_blob_stale", "evidence source differs from its Git blob")
        if b"\x00" in baseline_bytes:
            _fail("evidence_source_unavailable", "selected evidence is binary")
        try:
            text = baseline_bytes.decode("utf-8", errors="strict")
            _assert_secret_free(text)
        except UnicodeDecodeError as exc:
            raise ProductionContextSliceError(
                "selected evidence is not canonical UTF-8",
                reason_code="evidence_source_unavailable",
            ) from exc
        except ProductionContextSliceError:
            # Directory candidates were classified before selection; a later
            # mismatch is stale evidence rather than a skippable expansion.
            raise
        total_source_bytes += len(baseline_bytes)
        selected_paths.add(candidate.path)
        sources.append(
            {
                "authorized_by": sorted(candidate_owners[candidate.path]),
                "byte_length": len(baseline_bytes),
                "file_cid": _raw_cid(baseline_bytes),
                "git_blob_oid": candidate.git_blob_oid,
                "git_mode": candidate.git_mode,
                "path": candidate.path,
                "repository_path": candidate.repository_path,
                "repository_namespace": candidate.repository_namespace,
                "utf8_text": text,
            }
        )
    if not sources:
        _fail(
            "evidence_context_empty",
            "required evidence inputs produced no provider-visible source",
        )

    expansion_handles: list[dict[str, Any]] = []
    required_unmaterialized_count = 0
    expandable_by_declaration: dict[str, tuple[_EvidenceCandidate, ...]] = {}
    for declaration, candidates in zip(
        declarations,
        candidates_by_declaration,
        strict=True,
    ):
        expandable_by_declaration[declaration] = tuple(
            candidate
            for candidate in candidates
            if candidate.path not in byte_budget_unavailable_by_path
        )
    byte_budget_unavailable_by_declaration = {
        declaration: sum(
            1
            for candidate in candidates
            if candidate.path in byte_budget_unavailable_by_path
        )
        for declaration, candidates in zip(
            declarations,
            candidates_by_declaration,
            strict=True,
        )
    }
    for declaration, record in zip(
        declarations,
        declaration_records,
        strict=True,
    ):
        unavailable = byte_budget_unavailable_by_declaration[declaration]
        if unavailable:
            excluded = dict(record["excluded_entry_counts"])
            excluded["source_byte_budget"] = unavailable
            record["excluded_entry_counts"] = dict(sorted(excluded.items()))
    for declaration, record, candidates, excluded_counts in zip(
        declarations,
        declaration_records,
        (
            expandable_by_declaration[declaration]
            for declaration in declarations
        ),
        excluded_by_declaration,
        strict=True,
    ):
        selected = sorted(
            candidate.path for candidate in candidates if candidate.path in selected_paths
        )
        omitted = [
            candidate for candidate in candidates if candidate.path not in selected_paths
        ]
        if record["kind"] == "file" and omitted:
            required_unmaterialized_count += len(omitted)
        unavailable_count = (
            sum(excluded_counts.values())
            + byte_budget_unavailable_by_declaration[declaration]
        )
        core = {
            "authority_key": f"input:{declaration}",
            "authorized_path": declaration,
            "inventory_cid": record["inventory_cid"],
            "kind": record["kind"],
            "omitted_source_count": len(omitted),
            "omitted_path_preview": [
                {
                    "git_blob_oid": candidate.git_blob_oid,
                    "git_mode": candidate.git_mode,
                    "path": candidate.path,
                    "repository_namespace": candidate.repository_namespace,
                    "repository_path": candidate.repository_path,
                }
                for candidate in omitted[:32]
            ],
            "repository_namespace": record["repository_namespace"],
            "selection_generation": normalized_selections.get(
                f"input:{declaration}", 0
            ),
            "selected_paths": selected,
            "tree_or_blob_oid": record["tree_or_blob_oid"],
            "unavailable_entry_count": unavailable_count,
        }
        if omitted:
            expansion_handles.append(
                {
                    **core,
                    "expansion_cid": content_identity(core),
                    "request_contract": (
                        "supervisor-rebuild-with-explicit-evidence-paths@1"
                    ),
                    "request_parameters": {
                        "authorized_prefix": declaration,
                        "next_selection_generation": (
                            normalized_selections.get(f"input:{declaration}", 0) + 1
                        ),
                        "maximum_paths": max_source_paths,
                        "path_kind": "tracked-regular-file",
                    },
                }
            )

    repository_records = [
        {
            "baseline_commit": item.binding["baseline_commit"],
            "baseline_tree": item.binding["baseline_tree"],
            "object_format": item.binding["object_format"],
            "parent_gitlink_oid": item.parent_gitlink_oid,
            "repository_cid": item.binding["repository_cid"],
            "root_path": item.namespace,
            "snapshot_id": item.binding["snapshot_id"],
        }
        for item in repositories
    ]
    ref_bindings, ref_handles, ref_diff_bytes = _build_evidence_ref_bindings(
        repositories=repositories,
        evidence_refs=evidence_refs,
        priority_paths=priority_paths,
        evidence_inputs=declarations,
        protected_paths=protected,
        candidate_ref_authorities=verified_ref_authorities,
        context_round=context_round,
        expansion_selections=normalized_selections,
        max_refs=max_refs,
        max_ref_diffs=max_ref_diffs,
        max_ref_diff_bytes=max_ref_diff_bytes,
    )
    expansion_handles.extend(ref_handles)
    omitted_source_count = sum(
        int(item.get("omitted_source_count") or 0)
        for item in expansion_handles
        if isinstance(item, Mapping)
    )
    omitted_diff_count = sum(
        int(item.get("omitted_diff_count") or 0)
        for item in expansion_handles
        if isinstance(item, Mapping)
    )
    unavailable_entry_count = sum(
        sum(int(value) for value in counts.values())
        for counts in excluded_by_declaration
    ) + len(byte_budget_unavailable_by_path) + sum(
        int((binding.get("selection") or {}).get("withheld_diff_count") or 0)
        for binding in ref_bindings
        if isinstance(binding, Mapping)
    )
    dispatch_ready = bool(sources) and required_unmaterialized_count == 0
    complete = bool(
        dispatch_ready
        and omitted_source_count == 0
        and omitted_diff_count == 0
        and unavailable_entry_count == 0
    )
    payload: dict[str, Any] = {
        "authority": {
            "completion_authoritative": False,
            "provider_may_claim_omitted_evidence": False,
            "provider_may_read_ambient_filesystem": False,
            "repository_write_allowed": False,
        },
        "budget": {
            "evidence_manifest_tokens": 0,
            "max_declarations": max_declarations,
            "max_directory_entries": max_directory_entries,
            "max_evidence_tokens": max_evidence_tokens,
            "max_ref_diff_bytes": max_ref_diff_bytes,
            "max_ref_diffs": max_ref_diffs,
            "max_refs": max_refs,
            "max_source_bytes": max_source_bytes,
            "max_source_paths": max_source_paths,
            "source_bytes": total_source_bytes,
            "ref_diff_bytes": ref_diff_bytes,
            "token_estimator": "utf8-bytes-ceil-div-4@1",
        },
        "declarations": declaration_records,
        "evidence_cid": "",
        "expansion_handles": expansion_handles,
        "interface": PRODUCTION_EVIDENCE_AUTHORITY_INTERFACE,
        "readiness": {
            "provider_ready": dispatch_ready,
            "dispatch_ready": dispatch_ready,
            "complete": complete,
            "reason_code": (
                ""
                if dispatch_ready
                else "context_expansion_required"
            ),
            "required_unmaterialized_count": required_unmaterialized_count,
            "omitted_source_count": omitted_source_count,
            "omitted_diff_count": omitted_diff_count,
            "unavailable_entry_count": unavailable_entry_count,
        },
        "repository_binding": repository_records[0],
        "ref_bindings": ref_bindings,
        "root_bindings": repository_records,
        "schema": PRODUCTION_EVIDENCE_AUTHORITY_SCHEMA,
        "selection": {
            "directory_candidate_count": len(ordered_directory_candidates),
            "directory_window_end": len(directory_window),
            "directory_window_start": directory_window_start,
            "context_round": context_round,
            "explicit_file_anchor_count": len(unique_file_candidates),
        },
        "sources": sorted(sources, key=lambda item: item["path"]),
        "task_binding": task_binding,
    }
    chain_core = {
        "context_round": context_round,
        "parent_evidence_cid": str(parent_evidence_cid),
        "selected_expansion_cids": list(selected_handle_cids),
        "expansion_selections": dict(sorted(normalized_selections.items())),
    }
    payload["expansion_chain"] = {
        **chain_core,
        "chain_cid": content_identity(chain_core),
    }
    payload = _stabilize_evidence_payload(payload)
    if payload["budget"]["evidence_manifest_tokens"] > max_evidence_tokens:
        _fail(
            "evidence_budget_exceeded",
            "bounded evidence authority does not fit its provider allocation",
        )
    return ProductionEvidenceAuthorityManifest(payload)


@production_evidence_scan_budgeted
def verify_production_evidence_authority(
    manifest: ProductionEvidenceAuthorityManifest | Mapping[str, Any],
    *,
    repo_root: str | Path,
    current_task_id: str,
    current_task_payload: Mapping[str, Any],
    expected_evidence_inputs: Sequence[str],
    expected_evidence_refs: Sequence[str] = (),
    expected_candidate_ref_authority_appendix: Mapping[str, Any] | None = None,
    expected_board_namespace: str = "",
    expected_board_projection_id: str = "",
    expected_candidate_authority_signer_did: str = "",
    expected_priority_paths: Sequence[str] = (),
    governed_repository_roots: Sequence[str] = (),
    expected_protected_paths: Sequence[str] = (),
    baseline_ref: str = "HEAD",
    expected_context_round: int | None = None,
    expected_parent_evidence_cid: str | None = None,
    expected_selected_expansion_cids: Sequence[str] | None = None,
    expected_expansion_selections: Mapping[str, int] | None = None,
) -> ProductionEvidenceAuthorityManifest:
    """Rebuild and compare an evidence authority against current Git state."""

    payload = (
        manifest.to_dict()
        if isinstance(manifest, ProductionEvidenceAuthorityManifest)
        else _manifest_payload(manifest)
    )
    _exact_keys(
        payload,
        frozenset(
            {
                "authority",
                "budget",
                "declarations",
                "evidence_cid",
                "expansion_chain",
                "expansion_handles",
                "interface",
                "readiness",
                "ref_bindings",
                "repository_binding",
                "root_bindings",
                "schema",
                "selection",
                "sources",
                "task_binding",
            }
        ),
        location="evidence authority",
    )
    if (
        payload.get("schema") != PRODUCTION_EVIDENCE_AUTHORITY_SCHEMA
        or payload.get("interface") != PRODUCTION_EVIDENCE_AUTHORITY_INTERFACE
    ):
        _fail("evidence_manifest_malformed", "evidence schema/interface is invalid")
    expected_authority = {
        "completion_authoritative": False,
        "provider_may_claim_omitted_evidence": False,
        "provider_may_read_ambient_filesystem": False,
        "repository_write_allowed": False,
    }
    if payload.get("authority") != expected_authority:
        _fail("authority_claim", "evidence manifest widens provider authority")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        _fail("evidence_manifest_malformed", "evidence selection is malformed")
    _exact_keys(
        selection,
        frozenset(
            {
                "directory_candidate_count",
                "directory_window_end",
                "directory_window_start",
                "context_round",
                "explicit_file_anchor_count",
            }
        ),
        location="evidence selection",
    )
    context_round = selection.get("context_round")
    if (
        isinstance(context_round, bool)
        or not isinstance(context_round, int)
        or context_round < 0
        or context_round > DEFAULT_MAX_EVIDENCE_EXPANSION_ROUND
    ):
        _fail("expansion_round_invalid", "evidence expansion round is invalid")
    if expected_context_round is not None and context_round != expected_context_round:
        _fail(
            "scope_authority_mismatch",
            "evidence context round differs from supervisor authority",
        )
    chain = payload.get("expansion_chain")
    if not isinstance(chain, Mapping):
        _fail("expansion_chain_invalid", "evidence expansion chain is missing")
    _exact_keys(
        chain,
        frozenset(
            {
                "chain_cid",
                "context_round",
                "parent_evidence_cid",
                "selected_expansion_cids",
                "expansion_selections",
            }
        ),
        location="evidence expansion chain",
    )
    selected_cids = chain.get("selected_expansion_cids")
    expansion_selections = chain.get("expansion_selections")
    if not isinstance(selected_cids, list) or any(
        not isinstance(item, str) for item in selected_cids
    ) or not isinstance(expansion_selections, Mapping):
        _fail("expansion_chain_invalid", "selected expansion CIDs are malformed")
    chain_core = {
        "context_round": context_round,
        "parent_evidence_cid": str(chain.get("parent_evidence_cid") or ""),
        "selected_expansion_cids": list(selected_cids),
        "expansion_selections": dict(expansion_selections),
    }
    if chain.get("context_round") != context_round or chain.get(
        "chain_cid"
    ) != content_identity(chain_core):
        _fail("expansion_chain_invalid", "evidence expansion chain CID is invalid")
    if (
        expected_parent_evidence_cid is not None
        and chain_core["parent_evidence_cid"] != expected_parent_evidence_cid
    ):
        _fail("scope_authority_mismatch", "evidence parent CID differs from authority")
    expected_selected = (
        tuple(sorted(str(item) for item in expected_selected_expansion_cids))
        if expected_selected_expansion_cids is not None
        else None
    )
    if expected_selected is not None and tuple(selected_cids) != expected_selected:
        _fail("scope_authority_mismatch", "selected expansion CIDs differ from authority")
    if (
        expected_expansion_selections is not None
        and dict(expansion_selections) != dict(expected_expansion_selections)
    ):
        _fail("scope_authority_mismatch", "evidence expansion selections differ")
    given_cid = str(payload.get("evidence_cid") or "")
    expected_cid = content_identity(
        {key: value for key, value in payload.items() if key != "evidence_cid"}
    )
    if given_cid != expected_cid:
        _fail("evidence_cid_mismatch", "evidence manifest CID is invalid")
    budget = payload.get("budget")
    if not isinstance(budget, Mapping):
        _fail("evidence_manifest_malformed", "evidence budget is malformed")
    expected_budget_keys = frozenset(
        {
            "evidence_manifest_tokens",
            "max_declarations",
            "max_directory_entries",
            "max_evidence_tokens",
            "max_ref_diff_bytes",
            "max_ref_diffs",
            "max_refs",
            "max_source_bytes",
            "max_source_paths",
            "source_bytes",
            "ref_diff_bytes",
            "token_estimator",
        }
    )
    _exact_keys(budget, expected_budget_keys, location="evidence budget")
    numeric_bounds = {
        "max_declarations": DEFAULT_MAX_EVIDENCE_DECLARATIONS,
        "max_directory_entries": DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES,
        "max_evidence_tokens": MAX_PROVIDER_PROMPT_TOKENS,
        "max_ref_diff_bytes": DEFAULT_MAX_EVIDENCE_REF_DIFF_BYTES,
        "max_ref_diffs": DEFAULT_MAX_EVIDENCE_REF_DIFFS,
        "max_refs": DEFAULT_MAX_EVIDENCE_REFS,
        "max_source_bytes": DEFAULT_MAX_EVIDENCE_SOURCE_BYTES,
        "max_source_paths": DEFAULT_MAX_EVIDENCE_SOURCE_PATHS,
    }
    for name, maximum in numeric_bounds.items():
        value = budget.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > maximum
        ):
            _fail("budget_invalid", f"evidence {name} is invalid")
    observed = budget.get("evidence_manifest_tokens")
    source_bytes = budget.get("source_bytes")
    ref_diff_bytes = budget.get("ref_diff_bytes")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 1
        or observed > budget["max_evidence_tokens"]
        or isinstance(source_bytes, bool)
        or not isinstance(source_bytes, int)
        or source_bytes < 1
        or source_bytes > budget["max_source_bytes"]
        or isinstance(ref_diff_bytes, bool)
        or not isinstance(ref_diff_bytes, int)
        or ref_diff_bytes < 0
        or ref_diff_bytes > budget["max_ref_diff_bytes"]
        or budget.get("token_estimator") != "utf8-bytes-ceil-div-4@1"
        or observed != _token_estimate(canonical_json_bytes(payload))
    ):
        _fail("budget_invalid", "evidence budget observations are invalid")
    rebuilt = build_production_evidence_authority(
        repo_root=repo_root,
        task_id=current_task_id,
        task_payload=current_task_payload,
        evidence_inputs=expected_evidence_inputs,
        evidence_refs=expected_evidence_refs,
        candidate_ref_authority_appendix=(
            expected_candidate_ref_authority_appendix
        ),
        board_namespace=expected_board_namespace,
        board_projection_id=expected_board_projection_id,
        candidate_authority_signer_did=(
            expected_candidate_authority_signer_did
        ),
        priority_paths=expected_priority_paths,
        governed_repository_roots=governed_repository_roots,
        protected_paths=expected_protected_paths,
        baseline_ref=baseline_ref,
        max_evidence_tokens=int(budget["max_evidence_tokens"]),
        max_declarations=int(budget["max_declarations"]),
        max_source_paths=int(budget["max_source_paths"]),
        max_directory_entries=int(budget["max_directory_entries"]),
        max_source_bytes=int(budget["max_source_bytes"]),
        max_refs=int(budget["max_refs"]),
        max_ref_diffs=int(budget["max_ref_diffs"]),
        max_ref_diff_bytes=int(budget["max_ref_diff_bytes"]),
        context_round=context_round,
        parent_evidence_cid=chain_core["parent_evidence_cid"],
        selected_expansion_cids=tuple(selected_cids),
        expansion_selections=dict(expansion_selections),
    )
    if payload != rebuilt.to_dict():
        _fail(
            "evidence_stale",
            "evidence authority differs from the current task/repository forest",
        )
    return rebuilt


def derive_production_context_read_paths(
    *,
    repo_root: str | Path,
    baseline_ref: str,
    effect_paths: Sequence[str],
) -> tuple[str, ...]:
    """Derive the only implicit read scope: existing tracked effect blobs."""

    root = _repository_root(repo_root)
    repository = _repository_binding(root, baseline_ref)
    effects = _canonical_paths(
        effect_paths,
        field_name="effect_paths",
        maximum=DEFAULT_MAX_SCOPE_PATHS,
    )
    reads: list[str] = []
    for path in effects:
        _target, exists = _assert_safe_effect_path(root, path)
        entry = _tree_entry_optional(root, repository["baseline_commit"], path)
        if entry is not None:
            if not exists:
                _fail("blob_stale", "tracked effect path is missing")
            reads.append(path)
        elif exists:
            _fail(
                "effect_path_occupied",
                "new effect path is already occupied in the worktree",
            )
    return tuple(reads)


def build_production_context_slice(
    *,
    repo_root: str | Path,
    task_id: str,
    task_payload: Mapping[str, Any],
    read_paths: Sequence[str],
    effect_paths: Sequence[str],
    baseline_ref: str = "HEAD",
    symbol_hints: Mapping[str, Sequence[str]] | None = None,
    max_provider_prompt_tokens: int = MAX_PROVIDER_PROMPT_TOKENS,
    reserved_prompt_tokens: int = DEFAULT_RESERVED_PROMPT_TOKENS,
    token_counter: Callable[[bytes], int] = _token_estimate,
    max_scope_paths: int = DEFAULT_MAX_SCOPE_PATHS,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    whole_file_bytes: int = DEFAULT_WHOLE_FILE_BYTES,
) -> ProductionContextSliceManifest:
    """Build a deterministic source manifest for one exact current task.

    ``reserved_prompt_tokens`` belongs to the outer provider envelope, task
    contract, and response instructions.  The router must still measure the
    completed envelope; this builder guarantees that its contribution cannot
    consume the reserved capacity.
    """

    for name, value in (
        ("max_provider_prompt_tokens", max_provider_prompt_tokens),
        ("reserved_prompt_tokens", reserved_prompt_tokens),
        ("max_scope_paths", max_scope_paths),
        ("max_source_bytes", max_source_bytes),
        ("whole_file_bytes", whole_file_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _fail("budget_invalid", f"{name} must be a positive integer")
    if max_provider_prompt_tokens > MAX_PROVIDER_PROMPT_TOKENS:
        _fail(
            "budget_invalid",
            f"provider prompt budget may not exceed {MAX_PROVIDER_PROMPT_TOKENS}",
        )
    if reserved_prompt_tokens < DEFAULT_RESERVED_PROMPT_TOKENS:
        _fail(
            "budget_invalid",
            "outer provider envelope must retain the minimum reserved budget",
        )
    if reserved_prompt_tokens >= max_provider_prompt_tokens:
        _fail("budget_invalid", "reserved prompt budget leaves no source context")
    if not callable(token_counter):
        _fail("budget_invalid", "token_counter must be callable")
    if token_counter is not _token_estimate:
        _fail(
            "budget_invalid",
            "production context requires the deterministic router token estimator",
        )
    if max_scope_paths > DEFAULT_MAX_SCOPE_PATHS:
        _fail("budget_invalid", "scope path bound may not exceed the protocol maximum")
    if max_source_bytes > DEFAULT_MAX_SOURCE_BYTES:
        _fail("budget_invalid", "source byte bound may not exceed the protocol maximum")

    root = _repository_root(repo_root)
    task_binding = _task_binding(task_id, task_payload)
    repository_binding = _repository_binding(root, baseline_ref)
    effects = _canonical_paths(
        effect_paths,
        field_name="effect_paths",
        maximum=max_scope_paths,
    )
    if read_paths is None:
        reads = derive_production_context_read_paths(
            repo_root=root,
            baseline_ref=repository_binding["baseline_commit"],
            effect_paths=effects,
        )
    else:
        reads = _canonical_paths(
            read_paths,
            field_name="read_paths",
            maximum=max_scope_paths,
            allow_empty=True,
        )
    hints = _canonical_symbol_hints(symbol_hints, read_paths=reads)

    absence_proofs: list[dict[str, str]] = []
    for path in effects:
        _target, exists = _assert_safe_effect_path(root, path)
        entry = _tree_entry_optional(
            root,
            repository_binding["baseline_commit"],
            path,
        )
        if entry is not None:
            if path not in reads:
                _fail(
                    "scope_invalid",
                    "existing effect paths must be included in read scope",
                )
            if not exists:
                _fail("blob_stale", "tracked effect path is missing from the worktree")
        else:
            if exists:
                _fail(
                    "effect_path_occupied",
                    "new effect path is already occupied in the worktree",
                )
            if path in reads:
                _fail("scope_invalid", "absent effect path cannot be in read scope")
            absence_proofs.append(
                _absence_proof(path=path, repository_binding=repository_binding)
            )

    sources: list[dict[str, Any]] = []
    total_source_bytes = 0
    for path in reads:
        _assert_safe_worktree_path(root, path)
        mode, oid, baseline_bytes = _tree_entry(
            root,
            repository_binding["baseline_commit"],
            path,
        )
        try:
            current_bytes = _read_repository_regular_nofollow(
                root,
                path,
                maximum_bytes=max_source_bytes,
            )
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared source cannot be read",
                reason_code="source_unavailable",
            ) from exc
        if current_bytes != baseline_bytes:
            _fail("blob_stale", "declared source differs from the baseline blob")
        total_source_bytes += len(baseline_bytes)
        if total_source_bytes > max_source_bytes:
            _fail("scope_too_broad", "declared source exceeds its byte bound")
        sources.append(
            _source_record(
                path=path,
                mode=mode,
                git_blob_oid=oid,
                source=baseline_bytes,
                effect=path in effects,
                symbol_hints=tuple(hints.get(path, ())),
                whole_file_bytes=whole_file_bytes,
            )
        )

    scope = {
        "absence_proofs": absence_proofs,
        "effect_paths": list(effects),
        "read_paths": list(reads),
        "symbol_hints": hints,
        "scope_cid": content_identity(
            {
                "absence_proofs": absence_proofs,
                "effect_paths": list(effects),
                "read_paths": list(reads),
                "symbol_hints": hints,
                "task_binding": task_binding,
            }
        ),
    }
    unsigned: dict[str, Any] = {
        "authority": {
            "completion_authoritative": False,
            "provider_may_read_undeclared_paths": False,
            "provider_may_replace_unseen_bytes": False,
            "repository_write_allowed": False,
        },
        "budget": {
            "max_provider_prompt_tokens": max_provider_prompt_tokens,
            "reserved_prompt_tokens": reserved_prompt_tokens,
            "token_estimator": "utf8-bytes-ceil-div-4@1",
        },
        "interface": PRODUCTION_CONTEXT_SLICE_INTERFACE,
        "repository_binding": repository_binding,
        "schema": PRODUCTION_CONTEXT_SLICE_SCHEMA,
        "scope": scope,
        "sources": sources,
        "task_binding": task_binding,
    }
    manifest_cid = content_identity(unsigned)
    payload = {**unsigned, "manifest_cid": manifest_cid}
    encoded = canonical_json_bytes(payload)
    try:
        tokens = token_counter(encoded)
    except Exception as exc:
        raise ProductionContextSliceError(
            "context token counter failed",
            reason_code="budget_invalid",
        ) from exc
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        _fail("budget_invalid", "context token counter returned an invalid value")
    context_limit = max_provider_prompt_tokens - reserved_prompt_tokens
    if tokens > context_limit:
        _fail(
            "context_budget_exceeded",
            "exact bounded source context does not fit the provider prompt",
        )
    payload["budget"] = {
        **payload["budget"],
        "context_manifest_tokens": tokens,
        "context_token_limit": context_limit,
    }
    # Budget observations are derived metadata.  Bind them in the final root.
    payload["manifest_cid"] = content_identity(
        {key: value for key, value in payload.items() if key != "manifest_cid"}
    )
    final_tokens = token_counter(canonical_json_bytes(payload))
    if final_tokens > context_limit:
        _fail(
            "context_budget_exceeded",
            "context budget metadata exceeds the provider prompt allocation",
        )
    payload["budget"]["context_manifest_tokens"] = final_tokens
    # The observed token field changed, so stabilize the self-independent root.
    payload["manifest_cid"] = content_identity(
        {key: value for key, value in payload.items() if key != "manifest_cid"}
    )
    return ProductionContextSliceManifest(payload)


def _manifest_payload(
    manifest: ProductionContextSliceManifest | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(manifest, ProductionContextSliceManifest):
        return manifest.to_dict()
    if not isinstance(manifest, Mapping):
        _fail("manifest_malformed", "context manifest must be a mapping")
    try:
        return json.loads(canonical_json_bytes(dict(manifest)))
    except Exception as exc:
        raise ProductionContextSliceError(
            "context manifest is not canonical",
            reason_code="manifest_malformed",
        ) from exc


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    keys = set(value)
    if keys != set(expected) or not all(isinstance(key, str) for key in keys):
        _fail(
            "corpus_widening",
            f"{location} contains missing or unrecognized fields",
        )


def _verify_partition(source: bytes, record: Mapping[str, Any]) -> None:
    slices = record.get("source_slices")
    residuals = record.get("residuals")
    if not isinstance(slices, list) or not isinstance(residuals, list) or not slices:
        _fail("manifest_malformed", "source partition is malformed")
    segments: list[tuple[int, int, str, str]] = []
    for visibility, items in (("visible", slices), ("residual", residuals)):
        for item in items:
            if not isinstance(item, Mapping):
                _fail("manifest_malformed", "source segment is malformed")
            expected_keys = (
                frozenset(
                    {
                        "byte_end",
                        "byte_length",
                        "byte_start",
                        "content_cid",
                        "end_line",
                        "kind",
                        "qualified_name",
                        "start_line",
                        "utf8_text",
                    }
                )
                if visibility == "visible"
                else frozenset(
                    {"byte_end", "byte_length", "byte_start", "content_cid"}
                )
            )
            _exact_keys(item, expected_keys, location=f"{visibility} segment")
            start = item.get("byte_start")
            end = item.get("byte_end")
            length = item.get("byte_length")
            cid = item.get("content_cid")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (start, end, length)
            ) or not isinstance(cid, str):
                _fail("manifest_malformed", "source segment identity is malformed")
            if start < 0 or end < start or length != end - start or end > len(source):
                _fail("manifest_malformed", "source segment bounds are malformed")
            raw = source[start:end]
            if _raw_cid(raw) != cid:
                _fail("blob_stale", "source segment CID is stale")
            if visibility == "visible":
                if any(
                    isinstance(item.get(key), bool)
                    or not isinstance(item.get(key), int)
                    for key in ("start_line", "end_line")
                ) or not all(
                    isinstance(item.get(key), str)
                    for key in ("kind", "qualified_name", "utf8_text")
                ):
                    _fail("manifest_malformed", "visible segment shape is malformed")
                try:
                    text = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise ProductionContextSliceError(
                        "visible source is not UTF-8",
                        reason_code="manifest_malformed",
                    ) from exc
                if str(item.get("utf8_text")) != text:
                    _fail("blob_stale", "visible source text is stale")
                _assert_secret_free(text)
            elif "utf8_text" in item:
                _fail("corpus_widening", "residual source bytes must remain omitted")
            segments.append((start, end, cid, visibility))
    segments.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    partition_projection: list[dict[str, Any]] = []
    for start, end, cid, visibility in segments:
        if start != cursor:
            _fail("manifest_malformed", "source partition is not byte-complete")
        cursor = end
        partition_projection.append(
            {
                "byte_end": end,
                "byte_start": start,
                "content_cid": cid,
                "visibility": visibility,
            }
        )
    if cursor != len(source):
        _fail("manifest_malformed", "source partition is not byte-complete")
    expected_partition = content_identity(
        {
            "byte_length": len(source),
            "file_cid": _raw_cid(source),
            "segments": partition_projection,
        }
    )
    if str(record.get("partition_root_cid")) != expected_partition:
        _fail("manifest_cid_mismatch", "source partition root is invalid")
    if not isinstance(record.get("full_visible_coverage"), bool) or record.get(
        "full_visible_coverage"
    ) != (not residuals):
        _fail("manifest_malformed", "full-coverage declaration is invalid")


def verify_production_context_slice(
    manifest: ProductionContextSliceManifest | Mapping[str, Any],
    *,
    repo_root: str | Path,
    current_task_id: str,
    current_task_payload: Mapping[str, Any],
    expected_read_paths: Sequence[str],
    expected_effect_paths: Sequence[str],
    expected_symbol_hints: Mapping[str, Sequence[str]] | None = None,
    baseline_ref: str = "HEAD",
) -> ProductionContextSliceManifest:
    """Verify all task/repository/blob/scope/CID bindings against current state."""

    payload = _manifest_payload(manifest)
    _exact_keys(
        payload,
        frozenset(
            {
                "authority",
                "budget",
                "interface",
                "manifest_cid",
                "repository_binding",
                "schema",
                "scope",
                "sources",
                "task_binding",
            }
        ),
        location="manifest",
    )
    if payload.get("schema") != PRODUCTION_CONTEXT_SLICE_SCHEMA or payload.get(
        "interface"
    ) != PRODUCTION_CONTEXT_SLICE_INTERFACE:
        _fail("manifest_malformed", "context manifest schema/interface is invalid")
    given_cid = str(payload.get("manifest_cid") or "")
    expected_cid = content_identity(
        {key: value for key, value in payload.items() if key != "manifest_cid"}
    )
    if not given_cid or given_cid != expected_cid:
        _fail("manifest_cid_mismatch", "context manifest root CID is invalid")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        _fail("manifest_malformed", "context authority declaration is malformed")
    expected_authority = {
        "completion_authoritative": False,
        "provider_may_read_undeclared_paths": False,
        "provider_may_replace_unseen_bytes": False,
        "repository_write_allowed": False,
    }
    if dict(authority) != expected_authority:
        _fail("authority_claim", "context manifest attempts to widen provider authority")
    budget = payload.get("budget")
    if not isinstance(budget, Mapping):
        _fail("manifest_malformed", "context budget declaration is malformed")
    _exact_keys(
        budget,
        frozenset(
            {
                "context_manifest_tokens",
                "context_token_limit",
                "max_provider_prompt_tokens",
                "reserved_prompt_tokens",
                "token_estimator",
            }
        ),
        location="budget",
    )
    maximum = budget.get("max_provider_prompt_tokens")
    reserved = budget.get("reserved_prompt_tokens")
    observed = budget.get("context_manifest_tokens")
    limit = budget.get("context_token_limit")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (maximum, reserved, observed, limit)
    ):
        _fail("budget_invalid", "context budget values must be integers")
    if (
        maximum < 1
        or maximum > MAX_PROVIDER_PROMPT_TOKENS
        or reserved < DEFAULT_RESERVED_PROMPT_TOKENS
        or reserved >= maximum
        or limit != maximum - reserved
        or observed < 1
        or observed > limit
        or budget.get("token_estimator") != "utf8-bytes-ceil-div-4@1"
    ):
        _fail("budget_invalid", "context budget declaration is invalid")
    actual_tokens = _token_estimate(canonical_json_bytes(payload))
    if actual_tokens != observed or actual_tokens > limit:
        _fail("context_budget_exceeded", "context manifest exceeds its bound")
    expected_task = _task_binding(current_task_id, current_task_payload)
    if payload.get("task_binding") != expected_task:
        _fail("task_binding_stale", "context manifest belongs to another task revision")
    root = _repository_root(repo_root)
    expected_repository = _repository_binding(root, baseline_ref)
    if payload.get("repository_binding") != expected_repository:
        _fail("tree_stale", "context manifest belongs to another repository tree")
    scope = payload.get("scope")
    sources = payload.get("sources")
    if not isinstance(scope, Mapping) or not isinstance(sources, list):
        _fail("manifest_malformed", "context scope or sources are malformed")
    _exact_keys(
        scope,
        frozenset(
            {
                "absence_proofs",
                "effect_paths",
                "read_paths",
                "scope_cid",
                "symbol_hints",
            }
        ),
        location="scope",
    )
    reads = _canonical_paths(
        scope.get("read_paths", []),
        field_name="read_paths",
        maximum=DEFAULT_MAX_SCOPE_PATHS,
        allow_empty=True,
    )
    effects = _canonical_paths(
        scope.get("effect_paths", []),
        field_name="effect_paths",
        maximum=DEFAULT_MAX_SCOPE_PATHS,
    )
    operator_reads = _canonical_paths(
        expected_read_paths,
        field_name="expected_read_paths",
        maximum=DEFAULT_MAX_SCOPE_PATHS,
        allow_empty=True,
    )
    operator_effects = _canonical_paths(
        expected_effect_paths,
        field_name="expected_effect_paths",
        maximum=DEFAULT_MAX_SCOPE_PATHS,
    )
    operator_hints = _canonical_symbol_hints(
        expected_symbol_hints,
        read_paths=operator_reads,
    )
    manifest_hints = _canonical_symbol_hints(
        scope.get("symbol_hints"),
        read_paths=reads,
    )
    if (
        reads != operator_reads
        or effects != operator_effects
        or manifest_hints != operator_hints
    ):
        _fail(
            "scope_authority_mismatch",
            "context scope differs from the operator/task-derived scope",
        )
    expected_absences: list[dict[str, str]] = []
    for path in effects:
        _target, exists = _assert_safe_effect_path(root, path)
        entry = _tree_entry_optional(
            root,
            expected_repository["baseline_commit"],
            path,
        )
        if entry is not None:
            if path not in reads:
                _fail(
                    "scope_invalid",
                    "existing effect path is absent from read scope",
                )
            if not exists:
                _fail("blob_stale", "tracked effect path is missing")
        else:
            if exists:
                _fail(
                    "absence_proof_stale",
                    "new effect path is no longer absent",
                )
            if path in reads:
                _fail("scope_invalid", "absent effect path appears in read scope")
            expected_absences.append(
                _absence_proof(path=path, repository_binding=expected_repository)
            )
    raw_absences = scope.get("absence_proofs")
    if not isinstance(raw_absences, list) or raw_absences != expected_absences:
        _fail("absence_proof_stale", "effect-path absence proof is invalid")
    expected_scope_cid = content_identity(
        {
            "absence_proofs": expected_absences,
            "effect_paths": list(effects),
            "read_paths": list(reads),
            "symbol_hints": operator_hints,
            "task_binding": expected_task,
        }
    )
    if str(scope.get("scope_cid")) != expected_scope_cid:
        _fail("scope_stale", "context scope CID is invalid")
    if len(sources) != len(reads):
        _fail("scope_widening", "source records do not exactly cover read scope")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in sources:
        if not isinstance(record, Mapping):
            _fail("manifest_malformed", "source record is malformed")
        _exact_keys(
            record,
            frozenset(
                {
                    "byte_length",
                    "effect",
                    "file_cid",
                    "full_visible_coverage",
                    "git_blob_oid",
                    "git_mode",
                    "language",
                    "partition_root_cid",
                    "path",
                    "residuals",
                    "selection",
                    "source_slices",
                }
            ),
            location="source record",
        )
        path = _canonical_path(record.get("path"))
        if path in by_path or path not in reads:
            _fail("scope_widening", "source record is outside exact read scope")
        by_path[path] = record
    if set(by_path) != set(reads):
        _fail("scope_widening", "source records do not match exact read scope")
    total_source_bytes = 0
    for path in reads:
        record = by_path[path]
        _assert_safe_worktree_path(root, path)
        mode, oid, baseline_bytes = _tree_entry(
            root,
            expected_repository["baseline_commit"],
            path,
        )
        try:
            current_bytes = _read_repository_regular_nofollow(
                root,
                path,
                maximum_bytes=DEFAULT_MAX_SOURCE_BYTES,
            )
        except OSError as exc:
            raise ProductionContextSliceError(
                "declared source cannot be read",
                reason_code="source_unavailable",
            ) from exc
        if current_bytes != baseline_bytes:
            _fail("blob_stale", "current source differs from the bound blob")
        total_source_bytes += len(baseline_bytes)
        if total_source_bytes > DEFAULT_MAX_SOURCE_BYTES:
            _fail("scope_too_broad", "source scope exceeds the protocol byte bound")
        if (
            isinstance(record.get("byte_length"), bool)
            or not isinstance(record.get("byte_length"), int)
            or not isinstance(record.get("effect"), bool)
            or not isinstance(record.get("full_visible_coverage"), bool)
            or not all(
                isinstance(record.get(key), str)
                for key in (
                    "file_cid",
                    "git_blob_oid",
                    "git_mode",
                    "language",
                    "partition_root_cid",
                    "path",
                )
            )
        ):
            _fail("manifest_malformed", "source record field types are invalid")
        if (
            record.get("git_mode") != mode
            or record.get("git_blob_oid") != oid
            or record.get("file_cid") != _raw_cid(baseline_bytes)
            or record.get("byte_length") != len(baseline_bytes)
            or bool(record.get("effect")) != (path in effects)
        ):
            _fail("blob_stale", "source blob identity or role is stale")
        _verify_partition(baseline_bytes, record)
        selection = record.get("selection")
        if not isinstance(selection, Mapping):
            _fail("manifest_malformed", "source selection policy is malformed")
        _exact_keys(
            selection,
            frozenset({"mode", "qualified_symbols"}),
            location="source selection",
        )
        mode_name = selection.get("mode")
        symbols = selection.get("qualified_symbols")
        if not isinstance(symbols, list) or not all(
            isinstance(symbol, str) for symbol in symbols
        ):
            _fail("manifest_malformed", "source symbol selection is malformed")
        if symbols != sorted(set(symbols)):
            _fail("manifest_malformed", "source symbol selection is not canonical")
        if mode_name == "whole-file@1":
            if symbols:
                _fail("manifest_malformed", "whole-file selection cannot name symbols")
            rebuild_threshold = max(1, len(baseline_bytes))
        elif mode_name == "python-qualified-symbols@1":
            if not symbols or not path.endswith((".py", ".pyi")):
                _fail("manifest_malformed", "AST source selection is malformed")
            if symbols != operator_hints.get(path, []):
                _fail(
                    "scope_authority_mismatch",
                    "AST symbols differ from operator/task-derived hints",
                )
            # Zero forces AST mode on rebuild; header compaction itself uses
            # DEFAULT_WHOLE_FILE_BYTES inside _source_record and is therefore
            # independent of this rebuild threshold.
            rebuild_threshold = 0
        else:
            _fail("manifest_malformed", "source selection mode is unsupported")
        expected_record = _source_record(
            path=path,
            mode=mode,
            git_blob_oid=oid,
            source=baseline_bytes,
            effect=path in effects,
            symbol_hints=tuple(symbols),
            whole_file_bytes=rebuild_threshold,
        )
        if dict(record) != expected_record:
            _fail(
                "manifest_malformed",
                "source slice is not the deterministic selection for its blob",
            )
    return ProductionContextSliceManifest(payload)


def _covered(
    intervals: Sequence[tuple[int, int]],
    start: int,
    end: int,
    *,
    full_visible_coverage: bool,
) -> bool:
    if start == end:
        if full_visible_coverage:
            return any(left <= start <= right for left, right in intervals)
        # Residual CIDs are identity, not context.  A zero-preimage insertion
        # exactly on a visible/residual boundary is therefore not authorized.
        return any(left < start < right for left, right in intervals)
    cursor = start
    for left, right in sorted(intervals):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def _patch_preimage_ranges(
    patch: str,
    *,
    sources: Mapping[str, bytes],
    absent_paths: frozenset[str],
) -> dict[str, list[tuple[int, int]]]:
    if not patch.strip() or "GIT binary patch" in patch or "@@@" in patch:
        _fail("proposal_malformed", "proposal patch is empty or unsupported")
    result: dict[str, list[tuple[int, int]]] = {}
    headers: dict[str, tuple[str, str]] = {}
    current_path = ""
    old_header = ""
    new_header = ""
    lines = patch.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            if current_path:
                headers[current_path] = (old_header, new_header)
            match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)\n?", line)
            if not match or match.group(1) != match.group(2):
                _fail("proposal_malformed", "renames and quoted patch paths are forbidden")
            current_path = _canonical_path(match.group(1))
            if current_path in result:
                _fail("proposal_malformed", "patch path appears in multiple diff blocks")
            result[current_path] = []
            old_header = ""
            new_header = ""
            index += 1
            continue
        if line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            _fail("proposal_malformed", "patch rename/copy metadata is forbidden")
        if line.startswith("--- "):
            if not current_path or old_header:
                _fail("proposal_malformed", "patch old-path header is misplaced")
            old_header = line[4:].rstrip("\r\n")
            index += 1
            continue
        if line.startswith("+++ "):
            if not current_path or not old_header or new_header:
                _fail("proposal_malformed", "patch new-path header is misplaced")
            new_header = line[4:].rstrip("\r\n")
            index += 1
            continue
        hunk = _HUNK_RE.match(line.rstrip("\n"))
        if hunk:
            if not current_path or not old_header or not new_header:
                _fail("proposal_malformed", "patch hunk lacks exact path headers")
            if current_path not in sources and current_path not in absent_paths:
                _fail("proposal_scope_violation", "patch hunk path is outside context")
            old_start = int(hunk.group(1))
            old_count = int(hunk.group(2) or "1")
            consumed = 0
            index += 1
            while index < len(lines):
                body = lines[index]
                if body.startswith(("@@ ", "diff --git ")):
                    break
                if body.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not body or body[0] not in {" ", "+", "-"}:
                    _fail("proposal_malformed", "patch hunk line is malformed")
                if body[0] in {" ", "-"}:
                    consumed += 1
                index += 1
            if consumed != old_count:
                _fail("proposal_malformed", "patch hunk preimage count is malformed")
            source = sources.get(current_path, b"")
            offsets = _line_offsets(source)
            if old_count == 0:
                if old_start < 0 or old_start >= len(offsets):
                    _fail("proposal_malformed", "patch insertion exceeds current source")
                point = offsets[old_start]
                result[current_path].append((point, point))
            else:
                start_index = old_start - 1
                end_index = start_index + old_count
                if start_index < 0 or end_index >= len(offsets):
                    _fail("proposal_malformed", "patch hunk exceeds current source")
                result[current_path].append((offsets[start_index], offsets[end_index]))
            continue
        index += 1
    if current_path:
        headers[current_path] = (old_header, new_header)
    if not result or any(not ranges for ranges in result.values()):
        _fail("proposal_malformed", "proposal patch contains no bounded hunks")
    for path in result:
        old_path, new_path = headers.get(path, ("", ""))
        if path in absent_paths:
            valid_headers = (("/dev/null", f"b/{path}"),)
        else:
            valid_headers = (
                (f"a/{path}", f"b/{path}"),
                (f"a/{path}", "/dev/null"),
            )
        if (old_path, new_path) not in valid_headers:
            _fail(
                "proposal_scope_violation",
                "patch ---/+++ paths do not match their diff path and effect state",
            )
    return result


def assert_proposal_covered_by_context(
    manifest: ProductionContextSliceManifest | Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    repo_root: str | Path,
    current_task_id: str,
    current_task_payload: Mapping[str, Any],
    expected_read_paths: Sequence[str],
    expected_effect_paths: Sequence[str],
    expected_symbol_hints: Mapping[str, Sequence[str]] | None = None,
    baseline_ref: str = "HEAD",
) -> None:
    """Reject writes that require source bytes the provider could not see.

    Full-file replacements require full visible blob coverage.  Unified-diff
    preimages must be wholly contained in visible slices and must pass
    ``git apply --check`` against the bound current checkout.  Residual CIDs
    establish identity and preservation only; they never count as model
    context.
    """

    verified = verify_production_context_slice(
        manifest,
        repo_root=repo_root,
        current_task_id=current_task_id,
        current_task_payload=current_task_payload,
        expected_read_paths=expected_read_paths,
        expected_effect_paths=expected_effect_paths,
        expected_symbol_hints=expected_symbol_hints,
        baseline_ref=baseline_ref,
    )
    if not isinstance(proposal, Mapping):
        _fail("proposal_malformed", "provider proposal must be a mapping")
    body: Mapping[str, Any] = proposal
    nested = proposal.get("proposal")
    if isinstance(nested, Mapping):
        body = nested
    payload = verified.to_dict()
    effect_paths = set(payload["scope"]["effect_paths"])
    absent_paths = {
        proof["path"] for proof in payload["scope"]["absence_proofs"]
    }
    records = {item["path"]: item for item in payload["sources"]}
    root = _repository_root(repo_root)

    replacements = body.get("files")
    patch = str(body.get("patch") or "")
    if replacements not in (None, []):
        if patch.strip():
            _fail("proposal_malformed", "proposal may not mix patches and replacements")
        if not isinstance(replacements, list) or not replacements:
            _fail("proposal_malformed", "file replacements must be a nonempty list")
        seen: set[str] = set()
        for replacement in replacements:
            if not isinstance(replacement, Mapping):
                _fail("proposal_malformed", "file replacement is malformed")
            path = _canonical_path(replacement.get("path") or replacement.get("file"))
            if path in seen or path not in effect_paths:
                _fail("proposal_scope_violation", "replacement path is outside effect scope")
            seen.add(path)
            if path not in absent_paths and (
                path not in records
                or not bool(records[path].get("full_visible_coverage"))
            ):
                _fail(
                    "context_insufficient",
                    "full-file replacement requires full visible source coverage",
                )
            if replacement.get("content") is None and replacement.get("new_content") is None:
                _fail("proposal_malformed", "file replacement content is missing")
        return
    if not patch.strip():
        _fail("proposal_malformed", "proposal has no patch or file replacement")
    sources = {
        path: _tree_entry(root, verified.baseline_commit, path)[2]
        for path in records
    }
    ranges = _patch_preimage_ranges(
        patch,
        sources=sources,
        absent_paths=frozenset(absent_paths),
    )
    if not set(ranges).issubset(effect_paths):
        _fail("proposal_scope_violation", "patch path is outside effect scope")
    for path, hunks in ranges.items():
        if path in absent_paths:
            continue
        visible = [
            (int(item["byte_start"]), int(item["byte_end"]))
            for item in records[path]["source_slices"]
        ]
        if any(
            not _covered(
                visible,
                start,
                end,
                full_visible_coverage=bool(
                    records[path].get("full_visible_coverage")
                ),
            )
            for start, end in hunks
        ):
            _fail(
                "context_insufficient",
                "patch preimage is not wholly visible to the provider",
            )
    try:
        checked = subprocess.run(
            [
                PRODUCTION_GIT_EXECUTABLE,
                "--literal-pathspecs",
                "apply",
                "--check",
                "--whitespace=nowarn",
                "-",
            ],
            cwd=root,
            env=sanitized_git_environment(),
            input=patch,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
            timeout=DEFAULT_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise ProductionContextSliceError(
            "patch preimage could not be checked",
            reason_code="proposal_malformed",
        ) from exc
    if checked.returncode != 0:
        _fail("proposal_preimage_stale", "patch does not apply to the bound source")


__all__ = [
    "DEFAULT_MAX_EVIDENCE_DECLARATIONS",
    "DEFAULT_MAX_EVIDENCE_DIRECTORY_ENTRIES",
    "DEFAULT_MAX_EVIDENCE_SOURCE_BYTES",
    "DEFAULT_MAX_EVIDENCE_SOURCE_PATHS",
    "DEFAULT_MAX_GOVERNED_REPOSITORY_ROOTS",
    "DEFAULT_MAX_SCOPE_PATHS",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_RESERVED_PROMPT_TOKENS",
    "DEFAULT_WHOLE_FILE_BYTES",
    "MAX_PROVIDER_PROMPT_TOKENS",
    "PRODUCTION_CONTEXT_SLICE_INTERFACE",
    "PRODUCTION_CONTEXT_SLICE_SCHEMA",
    "PRODUCTION_EVIDENCE_AUTHORITY_INTERFACE",
    "PRODUCTION_EVIDENCE_AUTHORITY_SCHEMA",
    "ProductionContextSliceError",
    "ProductionContextSliceManifest",
    "ProductionEvidenceAuthorityManifest",
    "assert_proposal_covered_by_context",
    "build_production_context_slice",
    "build_production_evidence_authority",
    "derive_production_context_read_paths",
    "load_production_candidate_ref_authority_appendix",
    "load_verified_production_provider_launch_authority",
    "production_evidence_scan_budgeted",
    "verify_production_context_slice",
    "verify_production_evidence_authority",
]
