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
import stat
import subprocess
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, NoReturn

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)

PRODUCTION_CONTEXT_SLICE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-context-slice@1"
)
PRODUCTION_CONTEXT_SLICE_INTERFACE: Final = "ProductionContextSlice@1"
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

_GIT_OID_RE = re.compile(r"\A[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
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
        r"(?i)\btoken\s*[:=]\s*['\"][A-Za-z0-9._\-+/=]{12,}['\"]"
    ),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
        r"(?:[A-Za-z0-9+/=\s]+)"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----"
    ),
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


def _git(repo_root: Path, *arguments: str, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        raise ProductionContextSliceError(
            "repository identity is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if result.returncode != 0:
        raise ProductionContextSliceError(
            "repository identity or object is unavailable",
            reason_code="repository_unavailable",
        )
    return result.stdout


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ProductionContextSliceError(
            "repository object is unavailable",
            reason_code="repository_unavailable",
        ) from exc
    if result.returncode != 0:
        raise ProductionContextSliceError(
            "repository object is unavailable",
            reason_code="repository_unavailable",
        )
    return bytes(result.stdout)


def _repository_root(repo_root: str | Path) -> Path:
    root = Path(repo_root)
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


def _read_regular_nofollow(path: Path) -> bytes:
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
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


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



def _top_level_python_symbols(text: str) -> tuple[str, ...]:
    """Return deterministic top-level class/function names for budgeted slicing."""

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ()
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return tuple(sorted(set(names)))


def _visible_slice_bytes(
    text: str,
    source: bytes,
    *,
    symbols: Sequence[str],
    max_candidate_bytes: int,
) -> int:
    """Return merged visible byte length for the requested symbols."""

    candidates = _python_candidates(text, source, symbols=symbols)
    compact = tuple(
        _signature_candidate(source, candidate, max_bytes=max_candidate_bytes)
        for candidate in _merge_candidates(candidates)
    )
    return sum(candidate.end - candidate.start for candidate in compact)


def _budgeted_top_level_python_symbols(
    text: str,
    source: bytes,
    *,
    max_visible_bytes: int,
) -> tuple[str, ...]:
    """Select a deterministic top-level subset that fits the visible budget.

    Large modules (for example ``entrypoints/contracts.py``) cannot be attached
    whole-file to a production implement packet.  When operators omit symbol
    hints, greedily keep the smallest top-level definitions first so more
    interface-adjacent types fit under ``max_visible_bytes``.  Ties break by
    name so the selection is stable across runs.
    """

    names = _top_level_python_symbols(text)
    if not names:
        return ()
    budget = max(1, int(max_visible_bytes))
    # Match the post-selection compaction used by ``_source_record``.
    header_cap = DEFAULT_WHOLE_FILE_BYTES

    sized: list[tuple[int, str]] = []
    for name in names:
        try:
            body_bytes = _visible_slice_bytes(
                text,
                source,
                symbols=(name,),
                max_candidate_bytes=header_cap,
            )
        except ProductionContextSliceError:
            continue
        sized.append((body_bytes, name))
    if not sized:
        return ()
    # Prefer compact definitions so more names fit; stable name order on ties.
    sized.sort(key=lambda item: (item[0], item[1]))

    selected: list[str] = []
    for _size, name in sized:
        trial = tuple(sorted({*selected, name}))
        try:
            total = _visible_slice_bytes(
                text,
                source,
                symbols=trial,
                max_candidate_bytes=header_cap,
            )
        except ProductionContextSliceError:
            continue
        if total <= budget:
            selected.append(name)
    return tuple(sorted(selected))


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
    # utf8-bytes-ceil-div-4@1: refuse whole-file when the raw blob alone cannot
    # fit a conservative prompt headroom.  This keeps large contract modules
    # (e.g. entrypoints/contracts.py) from blocking new-file tasks that only
    # need a few related symbols.
    estimated_tokens = (len(source) + 3) // 4
    whole_file_ok = len(source) <= whole_file_bytes and estimated_tokens <= max(
        1, whole_file_bytes // 4
    )
    if whole_file_ok:
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
        if not normalized_symbols:
            # Budget-fit a deterministic subset of top-level definitions so
            # oversized modules remain implementable without operator hints.
            normalized_symbols = _budgeted_top_level_python_symbols(
                text,
                source,
                max_visible_bytes=max(4_096, whole_file_bytes // 2),
            )
        if not normalized_symbols:
            _fail(
                "symbol_scope_required",
                "large Python sources require exact qualified symbol hints",
            )
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
        target = _assert_safe_worktree_path(root, path)
        mode, oid, baseline_bytes = _tree_entry(
            root,
            repository_binding["baseline_commit"],
            path,
        )
        try:
            current_bytes = _read_regular_nofollow(target)
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
        # Cap whole-file inclusion by remaining provider prompt headroom so a
        # single oversized read path cannot exhaust the budget for new effects.
        context_limit = max_provider_prompt_tokens - reserved_prompt_tokens
        budgeted_whole = min(
            whole_file_bytes,
            max(4_096, context_limit * 3),
        )
        path_operator_hints = tuple(hints.get(path, ()))
        record = _source_record(
            path=path,
            mode=mode,
            git_blob_oid=oid,
            source=baseline_bytes,
            effect=path in effects,
            symbol_hints=path_operator_hints,
            whole_file_bytes=budgeted_whole,
        )
        # Promote deterministic auto-selected symbols into the bound scope so
        # verify can re-check the same operator-effective hints.  Operator-
        # supplied hints remain authoritative and are never overwritten.
        selection = record.get("selection")
        if (
            isinstance(selection, Mapping)
            and selection.get("mode") == "python-qualified-symbols@1"
            and not path_operator_hints
        ):
            auto_symbols = selection.get("qualified_symbols")
            if isinstance(auto_symbols, list) and auto_symbols:
                hints[path] = list(auto_symbols)
        sources.append(record)

    # Re-order after promotions so the scope map stays path-sorted/canonical.
    if hints:
        hints = {path: hints[path] for path in sorted(hints)}

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
    operator_provided_hints = _canonical_symbol_hints(
        expected_symbol_hints,
        read_paths=operator_reads,
    )
    manifest_hints = _canonical_symbol_hints(
        scope.get("symbol_hints"),
        read_paths=reads,
    )
    if reads != operator_reads or effects != operator_effects:
        _fail(
            "scope_authority_mismatch",
            "context scope differs from the operator/task-derived scope",
        )
    if operator_provided_hints:
        if manifest_hints != operator_provided_hints:
            _fail(
                "scope_authority_mismatch",
                "context scope differs from the operator/task-derived scope",
            )
        operator_hints = operator_provided_hints
    else:
        # Operator omitted path→symbol bindings.  The builder may have filled
        # scope.symbol_hints with a deterministic budgeted auto-subset; treat
        # those as operator-effective hints after per-source validation below.
        operator_hints = manifest_hints
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
        target = _assert_safe_worktree_path(root, path)
        mode, oid, baseline_bytes = _tree_entry(
            root,
            expected_repository["baseline_commit"],
            path,
        )
        try:
            current_bytes = _read_regular_nofollow(target)
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
            if not operator_provided_hints:
                # Re-derive the budgeted auto-subset under the same envelope the
                # builder used so callers cannot widen symbol scope by writing
                # arbitrary qualified names into an empty-hint manifest.
                try:
                    source_text = baseline_bytes.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise ProductionContextSliceError(
                        "declared source is not UTF-8",
                        reason_code="source_encoding_invalid",
                    ) from exc
                context_limit = int(maximum) - int(reserved)
                budgeted_whole = min(
                    DEFAULT_WHOLE_FILE_BYTES,
                    max(4_096, context_limit * 3),
                )
                auto_symbols = list(
                    _budgeted_top_level_python_symbols(
                        source_text,
                        baseline_bytes,
                        max_visible_bytes=max(4_096, budgeted_whole // 2),
                    )
                )
                if symbols != auto_symbols:
                    _fail(
                        "scope_authority_mismatch",
                        "auto AST symbols are not the budgeted deterministic subset",
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
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=root,
            input=patch,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        raise ProductionContextSliceError(
            "patch preimage could not be checked",
            reason_code="proposal_malformed",
        ) from exc
    if checked.returncode != 0:
        _fail("proposal_preimage_stale", "patch does not apply to the bound source")


__all__ = [
    "DEFAULT_MAX_SCOPE_PATHS",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_RESERVED_PROMPT_TOKENS",
    "DEFAULT_WHOLE_FILE_BYTES",
    "MAX_PROVIDER_PROMPT_TOKENS",
    "PRODUCTION_CONTEXT_SLICE_INTERFACE",
    "PRODUCTION_CONTEXT_SLICE_SCHEMA",
    "ProductionContextSliceError",
    "ProductionContextSliceManifest",
    "assert_proposal_covered_by_context",
    "build_production_context_slice",
    "derive_production_context_read_paths",
    "verify_production_context_slice",
]
