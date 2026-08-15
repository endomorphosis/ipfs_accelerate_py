"""Strict, content-addressed validation of implementation proposals.

The proposal gate is deliberately cheaper than test, semantic, or proof
validation.  It normalizes the candidate diff, checks its authority and path
envelope, proves that it has an observable effect, and emits a tamper-evident
receipt.  The receipt is not completion or proof authority; it is an admission
token consumed by :mod:`validation_scheduler`.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import shlex
import stat
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from ..proof.code_proof_obligations import CandidateDiffEntry, DiffChangeKind

NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID = (
    "314133036252270790078901745919131980427"
)
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID = "ASI-G100"
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_REVISION = "ASI-G100@asi-091"
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION = (
    "asi-g100-objective-validation@1"
)
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_CONFIGURATION_REVISION = (
    "strict-proposal-fail-fast-completion@1"
)
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS = 2
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_PRODUCING_TASK_IDS = ("ASI-031",)
NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA = (
    (
        "Proposal admission deterministically checks schema, authority, "
        "baseline and candidate identity, non-empty effective change, "
        "normalized path safety, and task-owned scope before any expensive "
        "validation. Empty or effectless diffs and every out-of-scope path "
        "fail closed with bounded typed diagnostics"
    ),
    "policy cannot widen task scope",
    (
        "rejected output cannot claim proof, completion, merge eligibility, "
        "or authority"
    ),
    (
        "the scheduler cannot be reached through the validated pipeline after "
        "preflight rejection. The exact requirement ID is emitted only by a "
        "tamper-evident receipt that binds the current tree, objective, policy, "
        "proposal, baseline, scope, normalized diff, complete ordered gate "
        "trace, failure result, proof that expensive dispatch remained closed, "
        "and content digest"
    ),
)
PROPOSAL_VALIDATION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-policy@1"
)
PROPOSAL_VALIDATION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-request@1"
)
PROPOSAL_VALIDATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-validation-receipt@1"
)
PROPOSAL_GATE_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-gate-evidence@1"
)
PROPOSAL_REJECTION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposal-rejection-evidence@1"
)
UNTRUSTED_PROPOSAL_ADMISSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/untrusted-proposal-admission@2"
)


class ProposalValidationError(ValueError):
    """Raised when a persisted proposal-validation record is inconsistent."""


class ProposalGate(str, Enum):
    SCHEMA = "schema"
    STRUCTURE = "structure"
    AUTHORITY = "authority"
    PATCH = "patch"
    PATH = "path"
    CONTENT = "content"
    VALIDATION = "validation"
    AST_INTERFACE = "ast_interface"


ORDERED_PROPOSAL_GATES = tuple(ProposalGate)

# These are the completion-gate terms that proposal admission can actually
# establish.  Impact-selected test execution, semantic/proof evaluation,
# merge, and freshness remain downstream responsibilities.
PROPOSAL_OWNED_GATE_GROUPS: tuple[
    tuple[str, tuple[ProposalGate, ...]], ...
] = (
    ("schema", (ProposalGate.SCHEMA, ProposalGate.STRUCTURE)),
    ("authority", (ProposalGate.AUTHORITY,)),
    ("patch", (ProposalGate.PATCH, ProposalGate.CONTENT)),
    ("path", (ProposalGate.PATH,)),
    ("ast_interface", (ProposalGate.AST_INTERFACE,)),
)


class ProposalFindingCode(str, Enum):
    INVALID_SCHEMA = "invalid_schema"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    OUTPUT_TOO_LARGE = "output_too_large"
    OUTPUT_TOO_DEEP = "output_too_deep"
    AUTHORITY_MISMATCH = "authority_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    STALE_BASELINE = "stale_baseline"
    STALE_PROPOSAL_REPLAY = "stale_proposal_replay"
    FORGED_AUTHORITY_CLAIM = "forged_authority_claim"
    EMPTY_PATCH = "empty_patch"
    NO_SEMANTIC_CHANGE = "no_semantic_change"
    PATCH_TOO_LARGE = "patch_too_large"
    PATCH_PARSE_ERROR = "patch_parse_error"
    PATCH_MISMATCH = "patch_mismatch"
    UNSAFE_PATH = "unsafe_path"
    PATH_OUTSIDE_SCOPE = "path_outside_scope"
    DECLARED_PATH_MISMATCH = "declared_path_mismatch"
    OPERATION_MISMATCH = "operation_mismatch"
    SYMLINK_BOUNDARY_FORBIDDEN = "symlink_boundary_forbidden"
    SUBMODULE_BOUNDARY_FORBIDDEN = "submodule_boundary_forbidden"
    SECRET_CHANGE_FORBIDDEN = "secret_change_forbidden"
    BINARY_CHANGE_FORBIDDEN = "binary_change_forbidden"
    LARGE_FILE_FORBIDDEN = "large_file_forbidden"
    GENERATED_CHANGE_FORBIDDEN = "generated_change_forbidden"
    TEST_DELETION_FORBIDDEN = "test_deletion_forbidden"
    TEST_WEAKENING_FORBIDDEN = "test_weakening_forbidden"
    COMMAND_FORBIDDEN = "command_forbidden"
    PYTHON_SYNTAX_ERROR = "python_syntax_error"
    INVALID_ENCODING = "invalid_encoding"
    DUPLICATE_FIELD = "duplicate_field"
    NON_CANONICAL_ID = "non_canonical_id"
    BASELINE_CONTENT_MISMATCH = "baseline_content_mismatch"
    CANDIDATE_IDENTITY_MISMATCH = "candidate_identity_mismatch"
    EXPECTED_EFFECT_MISMATCH = "expected_effect_mismatch"
    EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED = (
        "expected_output_ignored_or_unstaged"
    )
    HARDLINK_BOUNDARY_FORBIDDEN = "hardlink_boundary_forbidden"
    PROTECTED_PATH_FORBIDDEN = "protected_path_forbidden"
    REPOSITORY_PATH_RACE = "repository_path_race"
    REPOSITORY_CONTENT_MISMATCH = "repository_content_mismatch"
    ARCHIVE_CHANGE_FORBIDDEN = "archive_change_forbidden"
    VALIDATION_WEAKENING_FORBIDDEN = "validation_weakening_forbidden"
    # LPR-017 overlay gate findings (only emitted when enable_live_logic_repair).
    OMITTED_CALLERS = "omitted_callers"
    SIGNATURE_ARITY_INCREASE = "signature_arity_increase"
    UNKNOWN_FRONTIER_REQUIRED = "unknown_frontier_required"
    LOGIC_REPAIR_OVERLAY_REJECTED = "logic_repair_overlay_rejected"


QUALIFYING_FAIL_FAST_CODES = frozenset(
    {
        ProposalFindingCode.EMPTY_PATCH,
        ProposalFindingCode.NO_SEMANTIC_CHANGE,
        ProposalFindingCode.PATH_OUTSIDE_SCOPE,
        ProposalFindingCode.UNSAFE_PATH,
    }
)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProposalValidationError("non-finite values are not canonical")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProposalValidationError("record keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_canonical_json)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict())
    raise ProposalValidationError(
        f"unsupported canonical value: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _requirement_claims(
    payload: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Read a canonical optional requirement projection from a record."""

    if "proved_requirement_ids" not in payload:
        return None
    raw = payload["proved_requirement_ids"]
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise ProposalValidationError(
            "proved_requirement_ids must be a canonical string sequence"
        )
    claimed = tuple(raw)
    if claimed != _strings(claimed):
        raise ProposalValidationError(
            "proved_requirement_ids must be sorted and unique"
        )
    return claimed


def _path_matches(path: str, pattern: str) -> bool:
    pattern = str(pattern).strip().replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    if not pattern:
        return False
    if any(character in pattern for character in "*?["):
        return fnmatch.fnmatchcase(path, pattern)
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern or path.startswith(pattern.rstrip("/") + "/")


def _entry_has_effect(entry: CandidateDiffEntry) -> bool:
    if entry.change_kind in {DiffChangeKind.ADD, DiffChangeKind.COPY}:
        return bool(entry.after_source is not None or entry.after_blob_id)
    if entry.change_kind is DiffChangeKind.DELETE:
        return bool(entry.before_source is not None or entry.before_blob_id)
    if entry.old_path != entry.new_path:
        return True
    if entry.before_source is not None and entry.after_source is not None:
        if entry.before_source == entry.after_source:
            return False
        if entry.is_python:
            try:
                before_tree = ast.parse(entry.before_source)
                after_tree = ast.parse(entry.after_source)
            except (SyntaxError, ValueError, TypeError):
                # Syntax findings are emitted by the dedicated Python gate;
                # do not mislabel an unparsable edit as a semantic no-op.
                return True
            return ast.dump(
                before_tree,
                annotate_fields=True,
                include_attributes=False,
            ) != ast.dump(
                after_tree,
                annotate_fields=True,
                include_attributes=False,
            )
        # For opaque text, an all-whitespace rewrite is the only semantic
        # equivalence that can be established cheaply and deterministically.
        return entry.before_source.strip() != entry.after_source.strip()
    if entry.before_blob_id and entry.after_blob_id:
        return entry.before_blob_id != entry.after_blob_id
    # A modification with only one materialized side is observable, although
    # the later AST/scope compiler may conservatively reject it.
    return bool(
        entry.before_source is not None
        or entry.after_source is not None
        or entry.before_blob_id
        or entry.after_blob_id
    )


def _output_depth(value: Any, depth: int = 0) -> int:
    """Return the maximum container depth without recursively following objects."""

    if isinstance(value, Mapping):
        return max(
            (depth + 1, *(_output_depth(item, depth + 1) for item in value.values()))
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return max((depth + 1, *(_output_depth(item, depth + 1) for item in value)))
    return depth


def _safe_patch_path(value: str) -> str:
    value = str(value).strip()
    if value == "/dev/null":
        return ""
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ProposalValidationError("malformed quoted patch path") from exc
    if not parts:
        raise ProposalValidationError("empty patch path")
    raw = parts[0].replace("\\", "/")
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or "\x00" in raw
        or any(ord(character) < 32 for character in raw)
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProposalValidationError(f"unsafe patch path: {raw!r}")
    return path.as_posix()


_CANONICAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
_SHA256_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARCHIVE_SUFFIXES = (
    ".7z",
    ".bz2",
    ".gz",
    ".jar",
    ".rar",
    ".tar",
    ".tbz",
    ".tgz",
    ".war",
    ".xz",
    ".zip",
)
_ARCHIVE_MAGIC = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
)
_GENERATED_MARKERS_RE = re.compile(
    r"(?im)^\s*(?:[#/;*-]+\s*)?(?:"
    r"@generated\b|"
    r"generated\s+(?:file|code)\b|"
    r"automatically\s+generated\b|"
    r"do\s+not\s+edit"
    r"(?:\s+(?:this|the)\s+(?:file|code))?\s*[.!]?\s*$|"
    r"do\s+not\s+edit\s*[:;.!-]\s*[^\r\n]{0,80}"
    r"\b(?:generated|auto-generated|automatically\s+generated)\b|"
    r"do\s+not\s+edit\s+(?:(?:this|the)\s+)?"
    r"(?:generated|auto-generated|automatically\s+generated)\s+"
    r"(?:file|code)\b"
    r")"
)
_VALIDATION_CONFIG_PATHS = (
    ".github/workflows/",
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
)
_VALIDATION_WEAKENING_ADDITION_RE = re.compile(
    r"(?im)(?:"
    r"^\s*(?:addopts|filterwarnings)\s*=|"
    r"^\s*(?:continue-on-error|fail-fast)\s*[:=]\s*(?:true|false)\s*$|"
    r"^\s*xfail_strict\s*=\s*false\s*$|"
    r"\bpytest\b[^\r\n]*(?:\s-k\s|--ignore\b|--deselect\b|--maxfail\b)|"
    r"\|\|\s*true\b"
    r")"
)


def _validation_config_change_is_additive(entry: CandidateDiffEntry) -> bool:
    """Allow only non-weakening insertions into an existing config file."""

    if (
        entry.change_kind is DiffChangeKind.DELETE
        or entry.before_source is None
        or entry.after_source is None
    ):
        return False
    before_lines = entry.before_source.splitlines(keepends=True)
    after_lines = entry.after_source.splitlines(keepends=True)
    added_lines: list[str] = []
    after_index = 0
    for before_line in before_lines:
        while (
            after_index < len(after_lines)
            and after_lines[after_index] != before_line
        ):
            added_lines.append(after_lines[after_index])
            after_index += 1
        if after_index >= len(after_lines):
            return False
        after_index += 1
    added_lines.extend(after_lines[after_index:])
    return not _VALIDATION_WEAKENING_ADDITION_RE.search("".join(added_lines))


def _strict_repo_path(value: Any, *, field_name: str) -> str:
    """Validate a provider path without lossy normalization."""

    if not isinstance(value, str):
        raise ProposalValidationError(f"{field_name} must be a string")
    if value != value.strip() or "\\" in value:
        raise ProposalValidationError(
            f"{field_name} must be a canonical repository-relative path"
        )
    if value.startswith(("/", "./", "../", "//")) or "//" in value:
        raise ProposalValidationError(
            f"{field_name} must be a canonical repository-relative path"
        )
    pure = PurePosixPath(value)
    if (
        not value
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(ord(character) < 32 for character in value)
        or "\x00" in value
    ):
        raise ProposalValidationError(
            f"{field_name} must be a canonical repository-relative path"
        )
    return value


def _strict_text(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
    max_bytes: int = 16_384,
) -> str:
    if not isinstance(value, str):
        raise ProposalValidationError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ProposalValidationError(f"{field_name} must not be empty")
    if "\x00" in value or value.startswith("\ufeff") or any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise ProposalValidationError(f"{field_name} has an invalid encoding")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise ProposalValidationError(f"{field_name} has an invalid encoding") from exc
    if size > max_bytes:
        raise ProposalValidationError(f"{field_name} exceeds its byte bound")
    return value


def _strict_id(value: Any, *, field_name: str) -> str:
    result = _strict_text(value, field_name=field_name, max_bytes=256)
    if result != result.strip() or _CANONICAL_ID_RE.fullmatch(result) is None:
        raise ProposalValidationError(f"{field_name} is not a canonical identifier")
    return result


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _source_digest(source: str | None) -> str:
    if source is None:
        return ""
    return _sha256_bytes(source.encode("utf-8", errors="strict"))


def _looks_binary(value: bytes) -> bool:
    if not value:
        return False
    sample = value[:8_192]
    if b"\x00" in sample:
        return True
    controls = sum(
        byte < 32 and byte not in {9, 10, 12, 13} for byte in sample
    )
    return controls * 20 > len(sample)


def _looks_archive(path: str, value: bytes = b"") -> bool:
    lowered = path.lower()
    return lowered.endswith(_ARCHIVE_SUFFIXES) or any(
        value.startswith(magic) for magic in _ARCHIVE_MAGIC
    )


def _bounded_plain_value(
    value: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string_bytes: int,
) -> None:
    """Validate an already decoded provider value without invoking user code."""

    stack: list[tuple[Any, int]] = [(value, 1)]
    seen: set[int] = set()
    item_count = 0
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise ProposalValidationError("provider output exceeds the depth bound")
        if current is None or type(current) is bool:
            continue
        if type(current) is int:
            if current.bit_length() > 256:
                raise ProposalValidationError(
                    "provider output integer exceeds the numeric bound"
                )
            continue
        if type(current) is float:
            if current != current or current in (float("inf"), float("-inf")):
                raise ProposalValidationError("provider output contains a non-finite number")
            continue
        if type(current) is str:
            _strict_text(
                current,
                field_name="provider string",
                allow_empty=True,
                max_bytes=max_string_bytes,
            )
            continue
        if type(current) not in {dict, list, tuple}:
            raise ProposalValidationError(
                "provider output contains an unsupported value type"
            )
        identity = id(current)
        if identity in seen:
            raise ProposalValidationError("provider output contains a container cycle")
        seen.add(identity)
        children: Iterable[Any]
        if type(current) is dict:
            for key in current:
                if type(key) is not str:
                    raise ProposalValidationError("provider object keys must be strings")
                _strict_text(
                    key,
                    field_name="provider object key",
                    max_bytes=256,
                )
            children = current.values()
            item_count += len(current)
        else:
            children = current
            item_count += len(current)
        if item_count > max_items:
            raise ProposalValidationError("provider output exceeds the item-count bound")
        stack.extend((child, depth + 1) for child in children)


class _DuplicateJSONField(ProposalValidationError):
    pass


def _decode_untrusted_output(
    value: bytes | bytearray | str | Mapping[str, Any],
    *,
    max_bytes: int,
    max_depth: int,
    max_items: int,
) -> tuple[dict[str, Any], str]:
    """Decode one JSON object using a closed, bounded, duplicate-free envelope."""

    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise ProposalValidationError("provider output mapping must be a plain object")
        decoded = value
        _bounded_plain_value(
            decoded,
            max_depth=max_depth,
            max_items=max_items,
            max_string_bytes=max_bytes,
        )
        encoded = _canonical_json(decoded).encode("utf-8", errors="strict")
        if len(encoded) > max_bytes:
            raise ProposalValidationError("provider output exceeds the byte bound")
        return dict(decoded), _sha256_bytes(encoded)

    if isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        try:
            raw = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ProposalValidationError("provider output is not canonical UTF-8") from exc
    else:
        raise ProposalValidationError(
            "provider output must be UTF-8 JSON bytes, text, or a plain object"
        )
    if not raw or len(raw) > max_bytes:
        raise ProposalValidationError("provider output violates the byte bound")
    if raw.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw:
        raise ProposalValidationError("provider output is not canonical UTF-8")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProposalValidationError("provider output is not canonical UTF-8") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise _DuplicateJSONField("provider output contains duplicate object fields")
            result[key] = item
        return result

    def reject_constant(_value: str) -> None:
        raise ProposalValidationError("provider output contains a non-finite number")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except _DuplicateJSONField:
        raise
    except (
        json.JSONDecodeError,
        RecursionError,
        ProposalValidationError,
        ValueError,
    ) as exc:
        raise ProposalValidationError("provider output is not valid bounded JSON") from exc
    if type(decoded) is not dict:
        raise ProposalValidationError("provider output must be one JSON object")
    _bounded_plain_value(
        decoded,
        max_depth=max_depth,
        max_items=max_items,
        max_string_bytes=max_bytes,
    )
    return decoded, _sha256_bytes(raw)


@dataclass(frozen=True)
class ParsedPatchFile:
    """A bounded projection of one file section in a unified Git patch."""

    old_path: str
    new_path: str
    operation: str
    additions: int = 0
    deletions: int = 0
    binary: bool = False


_EMPTY_GIT_BLOB_IDS = (
    "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "473a0f4c3be8a93681a267e3b1e9a7dcda1185436fe141f7749120a303721813",
)


def _is_empty_git_blob_id(value: str) -> bool:
    """Return whether an abbreviated SHA-1/SHA-256 object id is the empty blob."""

    return bool(value) and any(
        blob_id.startswith(value) for blob_id in _EMPTY_GIT_BLOB_IDS
    )


def parse_unified_patch(
    patch_text: str,
    *,
    max_files: int = 256,
    max_bytes: int = 2_000_000,
    allow_binary: bool = False,
) -> tuple[ParsedPatchFile, ...]:
    """Parse and validate a Git-style unified diff without invoking a shell.

    Besides extracting paths and operations, the parser checks hunk line
    counts.  It rejects symlink/gitlink modes and opaque binary payloads before
    any patch tool or test process can be started.
    """

    if not isinstance(patch_text, str) or not patch_text.strip():
        raise ProposalValidationError("patch_text must be a non-empty string")
    if len(patch_text.encode("utf-8", errors="surrogatepass")) > max_bytes:
        raise ProposalValidationError("patch exceeds the byte bound")
    lines = patch_text.splitlines()
    files: list[ParsedPatchFile] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("diff --git "):
            if not line.strip():
                index += 1
                continue
            raise ProposalValidationError("patch content precedes a diff --git header")
        try:
            header = shlex.split(line)
        except ValueError as exc:
            raise ProposalValidationError("malformed diff --git header") from exc
        if len(header) != 4 or header[:2] != ["diff", "--git"]:
            raise ProposalValidationError("malformed diff --git header")
        old_path = _safe_patch_path(header[2])
        new_path = _safe_patch_path(header[3])
        operation = "modify"
        additions = deletions = 0
        binary = False
        saw_content = False
        saw_old_header = saw_new_header = False
        saw_new_file_mode = saw_deleted_file_mode = False
        index_old_hash = index_new_hash = ""
        index += 1
        while index < len(lines) and not lines[index].startswith("diff --git "):
            current = lines[index]
            mode = re.match(r"^(?:(?:new |deleted )?file mode|(?:new|old) mode) (\d+)$", current)
            if mode and mode.group(1) in {"120000", "160000"}:
                boundary = "symlink" if mode.group(1) == "120000" else "gitlink"
                raise ProposalValidationError(f"{boundary} patch modes are forbidden")
            if current.startswith(("GIT binary patch", "Binary files ")):
                binary = True
                if not allow_binary:
                    raise ProposalValidationError("binary patch payloads are forbidden")
            if current.startswith("new file mode "):
                operation = "add"
                saw_new_file_mode = True
            elif current.startswith("deleted file mode "):
                operation = "delete"
                saw_deleted_file_mode = True
            elif current.startswith(("old mode ", "new mode ")):
                operation = "type_change"
            elif current.startswith("rename from "):
                old_path = _safe_patch_path(current[len("rename from ") :])
                operation = "rename"
            elif current.startswith("rename to "):
                new_path = _safe_patch_path(current[len("rename to ") :])
                operation = "rename"
            elif current.startswith("copy from "):
                old_path = _safe_patch_path(current[len("copy from ") :])
                operation = "copy"
            elif current.startswith("copy to "):
                new_path = _safe_patch_path(current[len("copy to ") :])
                operation = "copy"
            elif current.startswith("--- "):
                old_path = _safe_patch_path(current[4:])
                saw_old_header = True
                if not old_path:
                    operation = "add"
            elif current.startswith("+++ "):
                new_path = _safe_patch_path(current[4:])
                saw_new_header = True
                if not new_path:
                    operation = "delete"
            elif current.startswith("@@ "):
                match = re.match(
                    r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: .*)?$",
                    current,
                )
                if match is None:
                    raise ProposalValidationError("malformed unified-diff hunk header")
                remaining_old = (
                    int(match.group(1)) if match.group(1) is not None else 1
                )
                remaining_new = (
                    int(match.group(2)) if match.group(2) is not None else 1
                )
                index += 1
                while index < len(lines):
                    body = lines[index]
                    if body.startswith(("diff --git ", "@@ ")):
                        break
                    if body == r"\ No newline at end of file":
                        index += 1
                        continue
                    if not body or body[0] not in {" ", "+", "-"}:
                        raise ProposalValidationError("malformed unified-diff hunk body")
                    if body[0] in {" ", "-"}:
                        remaining_old -= 1
                    if body[0] in {" ", "+"}:
                        remaining_new -= 1
                    additions += body[0] == "+"
                    deletions += body[0] == "-"
                    if remaining_old < 0 or remaining_new < 0:
                        raise ProposalValidationError("unified-diff hunk exceeds declared size")
                    index += 1
                    if remaining_old == remaining_new == 0:
                        break
                if remaining_old or remaining_new:
                    raise ProposalValidationError("truncated unified-diff hunk")
                saw_content = saw_content or additions > 0 or deletions > 0
                continue
            elif current.startswith("index "):
                match = re.fullmatch(
                    r"index ([0-9a-fA-F]{4,64})\.\.([0-9a-fA-F]{4,64})"
                    r"(?: \d+)?",
                    current,
                )
                if match is None or index_old_hash or index_new_hash:
                    raise ProposalValidationError(
                        "malformed or duplicate Git patch index metadata"
                    )
                index_old_hash = match.group(1).lower()
                index_new_hash = match.group(2).lower()
            elif current and not binary and not current.startswith(
                (
                    "similarity index ",
                    "dissimilarity index ",
                    r"\ No newline at end of file",
                )
            ):
                # Only declarative Git patch metadata is accepted.  This keeps
                # prose, shell fragments, and concatenated provider output out
                # of the patch envelope.
                raise ProposalValidationError("unrecognized Git patch content")
            index += 1
        if operation in {"add", "delete", "modify"} and not binary:
            effectful_hunk = saw_old_header and saw_new_header and saw_content
            empty_file_add = (
                operation == "add"
                and saw_new_file_mode
                and not saw_deleted_file_mode
                and bool(index_old_hash)
                and set(index_old_hash) == {"0"}
                and _is_empty_git_blob_id(index_new_hash)
            )
            empty_file_delete = (
                operation == "delete"
                and saw_deleted_file_mode
                and not saw_new_file_mode
                and _is_empty_git_blob_id(index_old_hash)
                and bool(index_new_hash)
                and set(index_new_hash) == {"0"}
            )
            if not (effectful_hunk or empty_file_add or empty_file_delete):
                raise ProposalValidationError(
                    "text patch section requires headers and an effectful hunk "
                    "or canonical empty-file metadata"
                )
        files.append(
            ParsedPatchFile(
                old_path=old_path,
                new_path=new_path,
                operation=operation,
                additions=additions,
                deletions=deletions,
                binary=binary,
            )
        )
        if len(files) > max_files:
            raise ProposalValidationError("patch touches more files than policy allows")
    if not files:
        raise ProposalValidationError("proposal must contain a Git-style unified diff")
    return tuple(files)


@dataclass(frozen=True)
class ProposalOperation:
    """One exact, rationale-bound operation declared by a structured proposal."""

    operation: str
    path: str
    old_path: str = ""
    rationale_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operation = str(self.operation or "").strip().lower().replace("-", "_")
        if operation not in {item.value for item in DiffChangeKind} - {"unknown"}:
            raise ProposalValidationError(f"unsupported proposal operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "path", _safe_patch_path(self.path))
        object.__setattr__(
            self, "old_path", _safe_patch_path(self.old_path) if self.old_path else ""
        )
        object.__setattr__(self, "rationale_refs", _strings(self.rationale_refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "path": self.path,
            "old_path": self.old_path,
            "rationale_refs": self.rationale_refs,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalOperation":
        return cls(
            operation=str(
                value.get("operation")
                or value.get("op")
                or value.get("change_kind")
                or ""
            ),
            path=str(value.get("path") or value.get("new_path") or ""),
            old_path=str(value.get("old_path") or ""),
            rationale_refs=tuple(
                value.get("rationale_refs") or value.get("rationale_references") or ()
            ),
        )


@dataclass(frozen=True)
class ProposalValidationStep:
    """A non-shell validation argv and the requirements it exercises."""

    command: tuple[str, ...]
    rationale_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        command = self.command
        if isinstance(command, str):
            try:
                command = tuple(shlex.split(command))
            except ValueError as exc:
                raise ProposalValidationError("malformed validation command") from exc
        else:
            command = tuple(str(part) for part in command)
        if not command or any(not part or "\x00" in part for part in command):
            raise ProposalValidationError("validation command argv must not be empty")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "rationale_refs", _strings(self.rationale_refs))

    def to_dict(self) -> dict[str, Any]:
        return {"command": self.command, "rationale_refs": self.rationale_refs}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | Sequence[str]) -> "ProposalValidationStep":
        if isinstance(value, Mapping):
            return cls(
                command=value.get("command") or value.get("argv") or (),
                rationale_refs=tuple(
                    value.get("rationale_refs")
                    or value.get("rationale_references")
                    or ()
                ),
            )
        return cls(command=value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProposalRisk:
    risk: str
    mitigation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk", " ".join(str(self.risk or "").split()))
        object.__setattr__(
            self, "mitigation", " ".join(str(self.mitigation or "").split())
        )
        if not self.risk or not self.mitigation:
            raise ProposalValidationError("proposal risks require risk and mitigation")

    def to_dict(self) -> dict[str, str]:
        return {"risk": self.risk, "mitigation": self.mitigation}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str) -> "ProposalRisk":
        if isinstance(value, Mapping):
            return cls(
                risk=str(value.get("risk") or value.get("description") or ""),
                mitigation=str(value.get("mitigation") or value.get("control") or ""),
            )
        return cls(risk=str(value), mitigation="review and targeted validation")


@dataclass(frozen=True)
class ProposalValidationFinding:
    code: ProposalFindingCode
    gate: ProposalGate
    message: str
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ProposalFindingCode(self.code))
        object.__setattr__(self, "gate", ProposalGate(self.gate))
        message = " ".join(str(self.message).split())
        object.__setattr__(self, "message", message[:240])
        object.__setattr__(self, "path", str(self.path or "").strip())
        if not self.message:
            raise ProposalValidationError("proposal finding message is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "gate": self.gate.value,
            "path": self.path,
            "message": self.message,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProposalValidationFinding":
        return cls(
            code=payload.get("code", ""),
            gate=payload.get("gate", ""),
            path=str(payload.get("path") or ""),
            message=str(payload.get("message") or ""),
        )


@dataclass(frozen=True)
class ProposalValidationPolicy:
    allowed_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...] = (
        ".git/",
        ".env",
        ".ssh/",
    )
    expected_task_id: str = ""
    expected_plan_id: str = ""
    expected_repository_id: str = ""
    expected_repository_tree_id: str = ""
    expected_objective_id: str = ""
    expected_context_id: str = ""
    expected_baseline_id: str = ""
    expected_replay_nonce: str = ""
    consumed_proposal_ids: tuple[str, ...] = ()
    symlink_paths: tuple[str, ...] = ()
    submodule_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    sensitive_path_patterns: tuple[str, ...] = (
        ".env*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*credentials*",
        "*secrets*",
        ".aws/",
        ".ssh/",
    )
    generated_path_patterns: tuple[str, ...] = (
        "build/",
        "dist/",
        "*.generated.*",
        "*.min.js",
        "*_pb2.py",
    )
    allowed_validation_commands: tuple[tuple[str, ...], ...] = (
        ("python", "-m", "pytest"),
        ("python", "-m", "unittest"),
        ("pytest",),
        ("ruff",),
        ("mypy",),
    )
    allow_binary: bool = False
    allow_secrets: bool = False
    allow_large_files: bool = False
    allow_generated: bool = False
    allow_test_deletion: bool = False
    allow_test_weakening: bool = False
    allow_archives: bool = False
    allow_hardlinks: bool = False
    allow_validation_config_changes: bool = False
    require_declared_paths: bool = True
    require_python_syntax: bool = True
    require_structured_details: bool = False
    require_patch_text: bool = False
    max_diff_entries: int = 256
    max_patch_bytes: int = 2_000_000
    max_output_bytes: int = 2_500_000
    max_output_depth: int = 16
    max_file_bytes: int = 1_000_000
    max_operations: int = 256
    max_expected_effects: int = 256
    max_path_depth: int = 32
    max_path_bytes: int = 512
    max_output_items: int = 16_384
    max_findings: int = 32
    policy_version: str = "strict-proposal-v1"
    policy_id: str = ""
    # This is the immutable scope assigned by the task authority.  The policy
    # may narrow it through ``allowed_paths`` but can never widen it.
    task_owned_paths: tuple[str, ...] = ()
    # LPR-017: when true, intercept ordinary proposals as read-only overlays
    # and reject/expand signature changes that omit resolved callers.
    enable_live_logic_repair: bool = False
    # Optional bound callers / frontier for hermetic overlay analysis.
    logic_repair_resolved_callers: tuple[str, ...] = ()
    logic_repair_unknown_frontier: tuple[str, ...] = ()
    logic_repair_compatibility_proofs: tuple[str, ...] = ()
    logic_repair_no_change_proofs: tuple[str, ...] = ()
    # When true, expand write set instead of hard-rejecting omitted callers.
    logic_repair_expand_write_set: bool = True

    def __post_init__(self) -> None:
        allowed = _strings(self.allowed_paths)
        forbidden = _strings(self.forbidden_paths)
        if not allowed:
            raise ProposalValidationError("allowed_paths must not be empty")
        task_owned = _strings(self.task_owned_paths) or allowed
        object.__setattr__(self, "allowed_paths", allowed)
        object.__setattr__(self, "task_owned_paths", task_owned)
        object.__setattr__(self, "forbidden_paths", forbidden)
        for name in (
            "consumed_proposal_ids",
            "symlink_paths",
            "submodule_paths",
            "protected_paths",
            "sensitive_path_patterns",
            "generated_path_patterns",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        commands: list[tuple[str, ...]] = []
        for command in self.allowed_validation_commands:
            normalized = (
                tuple(shlex.split(command))
                if isinstance(command, str)
                else tuple(str(part) for part in command)
            )
            if normalized and normalized not in commands:
                commands.append(normalized)
        if not commands:
            raise ProposalValidationError(
                "allowed_validation_commands must not be empty"
            )
        object.__setattr__(self, "allowed_validation_commands", tuple(commands))
        for name in (
            "allow_binary",
            "allow_secrets",
            "allow_large_files",
            "allow_generated",
            "allow_test_deletion",
            "allow_test_weakening",
            "allow_archives",
            "allow_hardlinks",
            "allow_validation_config_changes",
            "require_declared_paths",
            "require_python_syntax",
            "require_structured_details",
            "require_patch_text",
            "enable_live_logic_repair",
            "logic_repair_expand_write_set",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProposalValidationError(f"{name} must be a boolean")
        for name in (
            "logic_repair_resolved_callers",
            "logic_repair_unknown_frontier",
            "logic_repair_compatibility_proofs",
            "logic_repair_no_change_proofs",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        for name in (
            "expected_task_id",
            "expected_plan_id",
            "expected_repository_id",
            "expected_repository_tree_id",
            "expected_objective_id",
            "expected_context_id",
            "expected_baseline_id",
            "expected_replay_nonce",
            "policy_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "max_diff_entries",
            "max_patch_bytes",
            "max_output_bytes",
            "max_output_depth",
            "max_file_bytes",
            "max_operations",
            "max_expected_effects",
            "max_path_depth",
            "max_path_bytes",
            "max_output_items",
            "max_findings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ProposalValidationError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        claimed = str(self.policy_id or "").strip()
        object.__setattr__(self, "policy_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal policy identity mismatch")
        object.__setattr__(self, "policy_id", actual)

    def path_is_allowed(self, path: str) -> bool:
        """Return whether ``path`` is inside the policy's mutable envelope."""

        return any(_path_matches(path, pattern) for pattern in self.allowed_paths)

    def path_is_task_owned(self, path: str) -> bool:
        """Return whether ``path`` is inside the immutable task scope."""

        return any(
            _path_matches(path, pattern) for pattern in self.task_owned_paths
        )

    def path_is_in_scope(self, path: str) -> bool:
        """Return whether both proposal authority envelopes contain ``path``."""

        return self.path_is_allowed(path) and self.path_is_task_owned(path)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_POLICY_SCHEMA,
            "allowed_paths": self.allowed_paths,
            "task_owned_paths": self.task_owned_paths,
            "forbidden_paths": self.forbidden_paths,
            "expected_task_id": self.expected_task_id,
            "expected_plan_id": self.expected_plan_id,
            "expected_repository_id": self.expected_repository_id,
            "expected_repository_tree_id": self.expected_repository_tree_id,
            "expected_objective_id": self.expected_objective_id,
            "expected_context_id": self.expected_context_id,
            "expected_baseline_id": self.expected_baseline_id,
            "expected_replay_nonce": self.expected_replay_nonce,
            "consumed_proposal_ids": self.consumed_proposal_ids,
            "symlink_paths": self.symlink_paths,
            "submodule_paths": self.submodule_paths,
            "protected_paths": self.protected_paths,
            "sensitive_path_patterns": self.sensitive_path_patterns,
            "generated_path_patterns": self.generated_path_patterns,
            "allowed_validation_commands": self.allowed_validation_commands,
            "allow_binary": self.allow_binary,
            "allow_secrets": self.allow_secrets,
            "allow_large_files": self.allow_large_files,
            "allow_generated": self.allow_generated,
            "allow_test_deletion": self.allow_test_deletion,
            "allow_test_weakening": self.allow_test_weakening,
            "allow_archives": self.allow_archives,
            "allow_hardlinks": self.allow_hardlinks,
            "allow_validation_config_changes": self.allow_validation_config_changes,
            "require_declared_paths": self.require_declared_paths,
            "require_python_syntax": self.require_python_syntax,
            "require_structured_details": self.require_structured_details,
            "require_patch_text": self.require_patch_text,
            "max_diff_entries": self.max_diff_entries,
            "max_patch_bytes": self.max_patch_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_output_depth": self.max_output_depth,
            "max_file_bytes": self.max_file_bytes,
            "max_operations": self.max_operations,
            "max_expected_effects": self.max_expected_effects,
            "max_path_depth": self.max_path_depth,
            "max_path_bytes": self.max_path_bytes,
            "max_output_items": self.max_output_items,
            "max_findings": self.max_findings,
            "policy_version": self.policy_version,
            "enable_live_logic_repair": self.enable_live_logic_repair,
            "logic_repair_resolved_callers": self.logic_repair_resolved_callers,
            "logic_repair_unknown_frontier": self.logic_repair_unknown_frontier,
            "logic_repair_compatibility_proofs": (
                self.logic_repair_compatibility_proofs
            ),
            "logic_repair_no_change_proofs": self.logic_repair_no_change_proofs,
            "logic_repair_expand_write_set": self.logic_repair_expand_write_set,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationPolicy":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_POLICY_SCHEMA)
        if schema != PROPOSAL_VALIDATION_POLICY_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal policy schema: {schema}")
        return cls(
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            task_owned_paths=tuple(
                payload.get("task_owned_paths")
                or payload.get("allowed_paths")
                or ()
            ),
            forbidden_paths=tuple(payload.get("forbidden_paths") or ()),
            expected_task_id=str(payload.get("expected_task_id") or ""),
            expected_plan_id=str(payload.get("expected_plan_id") or ""),
            expected_repository_id=str(payload.get("expected_repository_id") or ""),
            expected_repository_tree_id=str(
                payload.get("expected_repository_tree_id") or ""
            ),
            expected_objective_id=str(payload.get("expected_objective_id") or ""),
            expected_context_id=str(payload.get("expected_context_id") or ""),
            expected_baseline_id=str(payload.get("expected_baseline_id") or ""),
            expected_replay_nonce=str(payload.get("expected_replay_nonce") or ""),
            consumed_proposal_ids=tuple(payload.get("consumed_proposal_ids") or ()),
            symlink_paths=tuple(payload.get("symlink_paths") or ()),
            submodule_paths=tuple(payload.get("submodule_paths") or ()),
            protected_paths=tuple(payload.get("protected_paths") or ()),
            sensitive_path_patterns=tuple(
                payload.get("sensitive_path_patterns")
                or cls.__dataclass_fields__["sensitive_path_patterns"].default
            ),
            generated_path_patterns=tuple(
                payload.get("generated_path_patterns")
                or cls.__dataclass_fields__["generated_path_patterns"].default
            ),
            allowed_validation_commands=tuple(
                tuple(command) if not isinstance(command, str) else command
                for command in (
                    payload.get("allowed_validation_commands")
                    or cls.__dataclass_fields__["allowed_validation_commands"].default
                )
            ),
            allow_binary=payload.get("allow_binary", False),
            allow_secrets=payload.get("allow_secrets", False),
            allow_large_files=payload.get("allow_large_files", False),
            allow_generated=payload.get("allow_generated", False),
            allow_test_deletion=payload.get("allow_test_deletion", False),
            allow_test_weakening=payload.get("allow_test_weakening", False),
            allow_archives=payload.get("allow_archives", False),
            allow_hardlinks=payload.get("allow_hardlinks", False),
            allow_validation_config_changes=payload.get(
                "allow_validation_config_changes", False
            ),
            require_declared_paths=payload.get("require_declared_paths", True),
            require_python_syntax=payload.get("require_python_syntax", True),
            require_structured_details=payload.get(
                "require_structured_details", False
            ),
            require_patch_text=payload.get("require_patch_text", False),
            max_diff_entries=int(payload.get("max_diff_entries", 256)),
            max_patch_bytes=int(payload.get("max_patch_bytes", 2_000_000)),
            max_output_bytes=int(payload.get("max_output_bytes", 2_500_000)),
            max_output_depth=int(payload.get("max_output_depth", 16)),
            max_file_bytes=int(payload.get("max_file_bytes", 1_000_000)),
            max_operations=int(payload.get("max_operations", 256)),
            max_expected_effects=int(payload.get("max_expected_effects", 256)),
            max_path_depth=int(payload.get("max_path_depth", 32)),
            max_path_bytes=int(payload.get("max_path_bytes", 512)),
            max_output_items=int(payload.get("max_output_items", 16_384)),
            max_findings=int(payload.get("max_findings", 32)),
            policy_version=str(payload.get("policy_version") or "strict-proposal-v1"),
            policy_id=str(payload.get("policy_id") or ""),
            enable_live_logic_repair=bool(
                payload.get("enable_live_logic_repair", False)
            ),
            logic_repair_resolved_callers=tuple(
                payload.get("logic_repair_resolved_callers") or ()
            ),
            logic_repair_unknown_frontier=tuple(
                payload.get("logic_repair_unknown_frontier") or ()
            ),
            logic_repair_compatibility_proofs=tuple(
                payload.get("logic_repair_compatibility_proofs") or ()
            ),
            logic_repair_no_change_proofs=tuple(
                payload.get("logic_repair_no_change_proofs") or ()
            ),
            logic_repair_expand_write_set=bool(
                payload.get("logic_repair_expand_write_set", True)
            ),
        )


@dataclass(frozen=True)
class ProposalExpectedEffect:
    """Provider-declared, independently recomputable effect for one path."""

    operation: str
    path: str
    before_sha256: str = ""
    after_sha256: str = ""

    def __post_init__(self) -> None:
        operation = str(self.operation or "").strip()
        if operation not in {item.value for item in DiffChangeKind} - {"unknown"}:
            raise ProposalValidationError("expected effect has an invalid operation")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self, "path", _strict_repo_path(self.path, field_name="expected_effect.path")
        )
        for name in ("before_sha256", "after_sha256"):
            value = str(getattr(self, name) or "")
            if value and _SHA256_ID_RE.fullmatch(value) is None:
                raise ProposalValidationError(
                    f"expected_effect.{name} must be a canonical SHA-256 identity"
                )
            object.__setattr__(self, name, value)
        if operation == "add" and self.before_sha256:
            raise ProposalValidationError("add effect cannot declare before content")
        if operation == "delete" and self.after_sha256:
            raise ProposalValidationError("delete effect cannot declare after content")
        if operation not in {"add", "delete"} and (
            not self.before_sha256 or not self.after_sha256
        ):
            raise ProposalValidationError(
                "modified effects require before and after content identities"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "operation": self.operation,
            "path": self.path,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalExpectedEffect":
        if type(value) is not dict:
            raise ProposalValidationError("expected effect must be a plain object")
        expected = {"operation", "path", "before_sha256", "after_sha256"}
        if set(value) != expected:
            raise ProposalValidationError(
                "expected effect must contain exactly its versioned fields"
            )
        if any(type(value[name]) is not str for name in expected):
            raise ProposalValidationError("expected effect fields must be strings")
        return cls(
            operation=value["operation"],
            path=value["path"],
            before_sha256=value["before_sha256"],
            after_sha256=value["after_sha256"],
        )


@dataclass(frozen=True)
class ImplementationProposal:
    task_id: str
    accepted_plan_id: str
    repository_id: str
    repository_tree_id: str
    objective_id: str
    baseline_id: str
    candidate_diff: tuple[CandidateDiffEntry, ...]
    declared_paths: tuple[str, ...]
    operations: tuple[ProposalOperation, ...] = ()
    rationale_references: tuple[str, ...] = ()
    validation_plan: tuple[ProposalValidationStep, ...] = ()
    risks: tuple[ProposalRisk, ...] = ()
    authority_claims: Mapping[str, Any] = field(default_factory=dict)
    expected_effects: tuple[ProposalExpectedEffect, ...] = ()
    patch_text: str = ""
    replay_nonce: str = ""
    context_id: str = ""
    proposal_version: str = "1"
    proposal_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "accepted_plan_id",
            "repository_id",
            "repository_tree_id",
            "objective_id",
            "baseline_id",
            "context_id",
            "replay_nonce",
            "proposal_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "task_id",
            "accepted_plan_id",
            "repository_tree_id",
            "objective_id",
            "baseline_id",
        ):
            if not getattr(self, name):
                raise ProposalValidationError(f"{name} is required")
        entries = tuple(
            item
            if isinstance(item, CandidateDiffEntry)
            else CandidateDiffEntry.from_mapping(item)
            for item in self.candidate_diff
        )
        object.__setattr__(self, "candidate_diff", entries)
        object.__setattr__(self, "declared_paths", _strings(self.declared_paths))
        operations = tuple(
            item
            if isinstance(item, ProposalOperation)
            else ProposalOperation.from_mapping(item)
            for item in self.operations
        )
        validations = tuple(
            item
            if isinstance(item, ProposalValidationStep)
            else ProposalValidationStep.from_mapping(item)
            for item in self.validation_plan
        )
        risks = tuple(
            item
            if isinstance(item, ProposalRisk)
            else ProposalRisk.from_mapping(item)
            for item in self.risks
        )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self, "rationale_references", _strings(self.rationale_references)
        )
        object.__setattr__(self, "validation_plan", validations)
        object.__setattr__(self, "risks", risks)
        effects = tuple(
            item
            if isinstance(item, ProposalExpectedEffect)
            else ProposalExpectedEffect.from_mapping(item)
            for item in self.expected_effects
        )
        object.__setattr__(self, "expected_effects", effects)
        claims = {
            str(key).strip(): _canonical(value)
            for key, value in sorted(dict(self.authority_claims).items())
            if str(key).strip()
        }
        object.__setattr__(self, "authority_claims", claims)
        object.__setattr__(self, "patch_text", str(self.patch_text or ""))
        claimed = str(self.proposal_id or "").strip()
        object.__setattr__(self, "proposal_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal identity mismatch")
        object.__setattr__(self, "proposal_id", actual)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        paths: set[str] = set()
        for entry in self.candidate_diff:
            paths.update(path for path in (entry.old_path, entry.new_path) if path)
        return tuple(sorted(paths))

    @property
    def effective_entries(self) -> tuple[CandidateDiffEntry, ...]:
        return tuple(entry for entry in self.candidate_diff if _entry_has_effect(entry))

    @property
    def diff_digest(self) -> str:
        return _identity(
            [entry.to_dict(include_sources=True) for entry in self.candidate_diff]
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_REQUEST_SCHEMA,
            "proposal_version": self.proposal_version,
            "task_id": self.task_id,
            "accepted_plan_id": self.accepted_plan_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "baseline_id": self.baseline_id,
            "context_id": self.context_id,
            "replay_nonce": self.replay_nonce,
            "declared_paths": self.declared_paths,
            "operations": [operation.to_dict() for operation in self.operations],
            "rationale_references": self.rationale_references,
            "validation_plan": [step.to_dict() for step in self.validation_plan],
            "risks": [risk.to_dict() for risk in self.risks],
            "authority_claims": dict(self.authority_claims),
            "expected_effects": [effect.to_dict() for effect in self.expected_effects],
            "patch_text": self.patch_text,
            "candidate_diff": [
                entry.to_dict(include_sources=True) for entry in self.candidate_diff
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "changed_paths": self.changed_paths,
            "diff_digest": self.diff_digest,
            "proposal_id": self.proposal_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationProposal":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_REQUEST_SCHEMA)
        if schema != PROPOSAL_VALIDATION_REQUEST_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal schema: {schema}")
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            accepted_plan_id=str(
                payload.get("accepted_plan_id") or payload.get("plan_id") or ""
            ),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(
                payload.get("repository_tree_id") or payload.get("tree_id") or ""
            ),
            objective_id=str(
                payload.get("objective_id") or payload.get("goal_id") or ""
            ),
            baseline_id=str(payload.get("baseline_id") or ""),
            context_id=str(payload.get("context_id") or ""),
            replay_nonce=str(payload.get("replay_nonce") or ""),
            declared_paths=tuple(payload.get("declared_paths") or ()),
            operations=tuple(
                ProposalOperation.from_mapping(item)
                for item in payload.get("operations") or ()
            ),
            rationale_references=tuple(
                payload.get("rationale_references")
                or payload.get("rationale_refs")
                or ()
            ),
            validation_plan=tuple(
                ProposalValidationStep.from_mapping(item)
                for item in payload.get("validation_plan") or ()
            ),
            risks=tuple(
                ProposalRisk.from_mapping(item) for item in payload.get("risks") or ()
            ),
            authority_claims=dict(payload.get("authority_claims") or {}),
            expected_effects=tuple(
                ProposalExpectedEffect.from_mapping(item)
                for item in payload.get("expected_effects") or ()
            ),
            patch_text=str(payload.get("patch_text") or payload.get("patch") or ""),
            candidate_diff=tuple(
                CandidateDiffEntry.from_mapping(item)
                for item in payload.get("candidate_diff") or ()
            ),
            proposal_version=str(payload.get("proposal_version") or "1"),
            proposal_id=str(payload.get("proposal_id") or ""),
        )
        if payload.get("diff_digest") and payload["diff_digest"] != result.diff_digest:
            raise ProposalValidationError("proposal diff digest mismatch")
        if payload.get("changed_paths") and tuple(payload["changed_paths"]) != result.changed_paths:
            raise ProposalValidationError("proposal changed paths mismatch")
        return result


ProposalValidationRequest = ImplementationProposal


@dataclass(frozen=True)
class ProposalRejectionEvidence:
    requirement_id: str
    task_id: str
    repository_id: str
    proposal_id: str
    receipt_id: str
    repository_tree_id: str
    objective_id: str
    baseline_id: str
    policy_id: str
    diff_digest: str
    allowed_paths: tuple[str, ...]
    task_owned_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    gate_trace: tuple[str, ...]
    rejection_codes: tuple[str, ...]
    expensive_node_ids: tuple[str, ...]
    expensive_checks_started: int = 0
    evidence_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "task_id",
            "repository_id",
            "proposal_id",
            "receipt_id",
            "repository_tree_id",
            "objective_id",
            "baseline_id",
            "policy_id",
            "diff_digest",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if self.requirement_id != NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID:
            raise ProposalValidationError("unsupported fail-fast requirement")
        if not all(
            (
                self.proposal_id,
                self.receipt_id,
                self.task_id,
                self.repository_id,
                self.repository_tree_id,
                self.objective_id,
                self.baseline_id,
                self.policy_id,
                self.diff_digest,
            )
        ):
            raise ProposalValidationError("rejection evidence binding is incomplete")
        object.__setattr__(self, "allowed_paths", _strings(self.allowed_paths))
        object.__setattr__(
            self, "task_owned_paths", _strings(self.task_owned_paths)
        )
        object.__setattr__(self, "changed_paths", _strings(self.changed_paths))
        trace = tuple(str(item or "").strip() for item in self.gate_trace)
        if trace != tuple(gate.value for gate in ORDERED_PROPOSAL_GATES):
            raise ProposalValidationError(
                "rejection evidence requires the complete ordered gate trace"
            )
        object.__setattr__(self, "gate_trace", trace)
        if not self.allowed_paths or not self.task_owned_paths:
            raise ProposalValidationError(
                "rejection evidence requires policy and task-owned scope"
            )
        codes = _strings(self.rejection_codes)
        if not set(codes).intersection(code.value for code in QUALIFYING_FAIL_FAST_CODES):
            raise ProposalValidationError(
                "rejection evidence requires a no-op or out-of-scope code"
            )
        object.__setattr__(self, "rejection_codes", codes)
        object.__setattr__(self, "expensive_node_ids", _strings(self.expensive_node_ids))
        if isinstance(self.expensive_checks_started, bool):
            raise ProposalValidationError("expensive_checks_started must be an integer")
        if int(self.expensive_checks_started) != 0:
            raise ProposalValidationError(
                "fail-fast evidence requires closed expensive dispatch"
            )
        object.__setattr__(self, "expensive_checks_started", 0)
        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("rejection evidence identity mismatch")
        object.__setattr__(self, "evidence_id", actual)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_REJECTION_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "task_id": self.task_id,
            "repository_id": self.repository_id,
            "proposal_id": self.proposal_id,
            "receipt_id": self.receipt_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "baseline_id": self.baseline_id,
            "policy_id": self.policy_id,
            "diff_digest": self.diff_digest,
            "allowed_paths": self.allowed_paths,
            "task_owned_paths": self.task_owned_paths,
            "changed_paths": self.changed_paths,
            "gate_trace": self.gate_trace,
            "rejection_codes": self.rejection_codes,
            "expensive_node_ids": self.expensive_node_ids,
            "expensive_checks_started": self.expensive_checks_started,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalRejectionEvidence":
        schema = str(payload.get("schema") or PROPOSAL_REJECTION_EVIDENCE_SCHEMA)
        if schema != PROPOSAL_REJECTION_EVIDENCE_SCHEMA:
            raise ProposalValidationError(
                f"unsupported rejection evidence schema: {schema}"
            )
        return cls(
            requirement_id=str(payload.get("requirement_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            proposal_id=str(payload.get("proposal_id") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            baseline_id=str(payload.get("baseline_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            diff_digest=str(payload.get("diff_digest") or ""),
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            task_owned_paths=tuple(payload.get("task_owned_paths") or ()),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            gate_trace=tuple(payload.get("gate_trace") or ()),
            rejection_codes=tuple(payload.get("rejection_codes") or ()),
            expensive_node_ids=tuple(payload.get("expensive_node_ids") or ()),
            expensive_checks_started=payload.get("expensive_checks_started", -1),
            evidence_id=str(payload.get("evidence_id") or ""),
        )


@dataclass(frozen=True)
class ProposalValidationReceipt:
    proposal_id: str
    policy_id: str
    repository_tree_id: str
    objective_id: str
    diff_digest: str
    allowed_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    accepted: bool
    findings: tuple[ProposalValidationFinding, ...]
    gate_trace: tuple[ProposalGate, ...]
    expensive_node_ids: tuple[str, ...] = ()
    expensive_checks_started: int = 0
    rejection_evidence: ProposalRejectionEvidence | None = None
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "policy_id",
            "repository_tree_id",
            "objective_id",
            "diff_digest",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
            if not getattr(self, name):
                raise ProposalValidationError(f"{name} is required")
        object.__setattr__(self, "allowed_paths", _strings(self.allowed_paths))
        object.__setattr__(self, "changed_paths", _strings(self.changed_paths))
        findings = tuple(
            item
            if isinstance(item, ProposalValidationFinding)
            else ProposalValidationFinding.from_dict(item)
            for item in self.findings
        )
        object.__setattr__(self, "findings", findings)
        trace = tuple(ProposalGate(item) for item in self.gate_trace)
        if trace != ORDERED_PROPOSAL_GATES:
            raise ProposalValidationError(
                "proposal gate trace must cover every ordered proposal gate"
            )
        object.__setattr__(self, "gate_trace", trace)
        if not isinstance(self.accepted, bool):
            raise ProposalValidationError("accepted must be a boolean")
        if bool(self.accepted) == bool(findings):
            raise ProposalValidationError(
                "accepted proposals have no findings; rejected proposals require findings"
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "expensive_node_ids", _strings(self.expensive_node_ids))
        if isinstance(self.expensive_checks_started, bool) or int(
            self.expensive_checks_started
        ) < 0:
            raise ProposalValidationError(
                "expensive_checks_started must be a non-negative integer"
            )
        object.__setattr__(
            self, "expensive_checks_started", int(self.expensive_checks_started)
        )
        evidence = self.rejection_evidence
        if evidence is not None and not isinstance(evidence, ProposalRejectionEvidence):
            evidence = ProposalRejectionEvidence.from_dict(evidence)
        object.__setattr__(self, "rejection_evidence", None)
        claimed = str(self.receipt_id or "").strip()
        object.__setattr__(self, "receipt_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise ProposalValidationError("proposal receipt identity mismatch")
        object.__setattr__(self, "receipt_id", actual)
        if evidence is not None:
            if (
                self.accepted
                or evidence.receipt_id != actual
                or evidence.proposal_id != self.proposal_id
                or evidence.repository_tree_id != self.repository_tree_id
                or evidence.objective_id != self.objective_id
                or evidence.policy_id != self.policy_id
                or evidence.diff_digest != self.diff_digest
                or evidence.allowed_paths != self.allowed_paths
                or evidence.changed_paths != self.changed_paths
                or evidence.gate_trace
                != tuple(gate.value for gate in self.gate_trace)
                or evidence.expensive_node_ids != self.expensive_node_ids
                or evidence.expensive_checks_started != self.expensive_checks_started
                or not set(evidence.rejection_codes).issubset(
                    finding.code.value for finding in findings
                )
            ):
                raise ProposalValidationError(
                    "rejection evidence is detached from proposal receipt"
                )
            object.__setattr__(self, "rejection_evidence", evidence)

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(finding.code.value for finding in self.findings)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (
            self.rejection_evidence.proved_requirement_ids
            if self.rejection_evidence is not None
            else ()
        )

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        """Proposal admission never proves the resulting implementation."""

        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def merge_eligible(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def freshness_authoritative(self) -> bool:
        return False

    @property
    def proposal_gate_evidence(self) -> Mapping[str, Any]:
        """Project explicit proposal-owned gate results for the parent join.

        The projection is derived exclusively from this content-addressed
        receipt.  It makes positive gate coverage visible without upgrading
        proposal admission into semantic-proof or completion authority.
        """

        gates: dict[str, Any] = {}
        for name, members in PROPOSAL_OWNED_GATE_GROUPS:
            codes = _strings(
                finding.code.value
                for finding in self.findings
                if finding.gate in members
            )
            gates[name] = {
                # Findings are intentionally bounded.  A rejected receipt may
                # therefore have additional failures that are not projected;
                # only the globally accepted verdict can support a positive
                # per-gate claim.
                "passed": self.accepted and not codes,
                "finding_codes": codes,
            }
        payload = {
            "schema": PROPOSAL_GATE_EVIDENCE_SCHEMA,
            "proposal_id": self.proposal_id,
            "policy_id": self.policy_id,
            "receipt_id": self.receipt_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "diff_digest": self.diff_digest,
            "gates": gates,
            "all_owned_gates_passed": all(
                item["passed"] for item in gates.values()
            ),
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        return {**payload, "evidence_id": _identity(payload)}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_VALIDATION_RECEIPT_SCHEMA,
            "proposal_id": self.proposal_id,
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "diff_digest": self.diff_digest,
            "allowed_paths": self.allowed_paths,
            "changed_paths": self.changed_paths,
            "accepted": self.accepted,
            "findings": [finding.to_dict() for finding in self.findings],
            "gate_trace": [gate.value for gate in self.gate_trace],
            "expensive_node_ids": self.expensive_node_ids,
            "expensive_checks_started": self.expensive_checks_started,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "rejection_evidence": (
                self.rejection_evidence.to_dict()
                if self.rejection_evidence is not None
                else None
            ),
            "proved_requirement_ids": self.proved_requirement_ids,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
            "merge_eligible": False,
            "authoritative": False,
            "freshness_authoritative": False,
            "proposal_gate_evidence": self.proposal_gate_evidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationReceipt":
        schema = str(payload.get("schema") or PROPOSAL_VALIDATION_RECEIPT_SCHEMA)
        if schema != PROPOSAL_VALIDATION_RECEIPT_SCHEMA:
            raise ProposalValidationError(f"unsupported proposal receipt schema: {schema}")
        for field_name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
            "merge_eligible",
            "authoritative",
            "freshness_authoritative",
        ):
            if payload.get(field_name) not in (None, False):
                raise ProposalValidationError(
                    f"proposal receipt cannot claim {field_name}"
                )
        receipt = cls(
            proposal_id=str(payload.get("proposal_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            diff_digest=str(payload.get("diff_digest") or ""),
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            accepted=payload.get("accepted", False),
            findings=tuple(
                ProposalValidationFinding.from_dict(item)
                for item in payload.get("findings") or ()
            ),
            gate_trace=tuple(payload.get("gate_trace") or ()),
            expensive_node_ids=tuple(payload.get("expensive_node_ids") or ()),
            expensive_checks_started=payload.get("expensive_checks_started", 0),
            receipt_id=str(payload.get("receipt_id") or ""),
        )
        evidence_payload = payload.get("rejection_evidence")
        if evidence_payload:
            evidence = ProposalRejectionEvidence.from_dict(evidence_payload)
            receipt = cls(
                proposal_id=receipt.proposal_id,
                policy_id=receipt.policy_id,
                repository_tree_id=receipt.repository_tree_id,
                objective_id=receipt.objective_id,
                diff_digest=receipt.diff_digest,
                allowed_paths=receipt.allowed_paths,
                changed_paths=receipt.changed_paths,
                accepted=receipt.accepted,
                findings=receipt.findings,
                gate_trace=receipt.gate_trace,
                expensive_node_ids=receipt.expensive_node_ids,
                expensive_checks_started=receipt.expensive_checks_started,
                rejection_evidence=evidence,
                receipt_id=receipt.receipt_id,
            )
        claimed_requirements = _requirement_claims(payload)
        if (
            claimed_requirements is not None
            and claimed_requirements != receipt.proved_requirement_ids
        ):
            raise ProposalValidationError("proposal requirement claims mismatch")
        claimed_gate_evidence = payload.get("proposal_gate_evidence")
        if (
            not isinstance(claimed_gate_evidence, Mapping)
            or _canonical(claimed_gate_evidence)
            != _canonical(receipt.proposal_gate_evidence)
        ):
            raise ProposalValidationError("proposal gate evidence mismatch")
        return receipt

    def with_dispatch_outcome(
        self,
        *,
        expensive_node_ids: Iterable[str],
        expensive_checks_started: int,
        task_id: str = "",
        repository_id: str = "",
        baseline_id: str = "",
        task_owned_paths: Iterable[str] = (),
    ) -> "ProposalValidationReceipt":
        """Bind the scheduler-owned dispatch outcome and derive evidence."""

        # Dispatch closure is the evidence needed for a rejected proposal.
        # Accepted proposal identity stays stable when handed to downstream
        # semantic and completion gates.
        if self.accepted:
            return self
        node_ids = _strings(expensive_node_ids)
        if (
            isinstance(expensive_checks_started, bool)
            or not isinstance(expensive_checks_started, int)
            or expensive_checks_started < 0
        ):
            raise ProposalValidationError(
                "expensive_checks_started must be a non-negative integer"
            )
        started = expensive_checks_started
        base = ProposalValidationReceipt(
            proposal_id=self.proposal_id,
            policy_id=self.policy_id,
            repository_tree_id=self.repository_tree_id,
            objective_id=self.objective_id,
            diff_digest=self.diff_digest,
            allowed_paths=self.allowed_paths,
            changed_paths=self.changed_paths,
            accepted=self.accepted,
            findings=self.findings,
            gate_trace=self.gate_trace,
            expensive_node_ids=node_ids,
            expensive_checks_started=started,
        )
        qualifying = (
            not base.accepted
            and started == 0
            and bool(str(task_id or "").strip())
            and bool(str(repository_id or "").strip())
            and bool(str(baseline_id or "").strip())
            and bool(_strings(task_owned_paths))
            and set(base.rejection_codes).intersection(
                code.value for code in QUALIFYING_FAIL_FAST_CODES
            )
        )
        if not qualifying:
            return base
        evidence = ProposalRejectionEvidence(
            requirement_id=NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
            task_id=task_id,
            repository_id=repository_id,
            proposal_id=base.proposal_id,
            receipt_id=base.receipt_id,
            repository_tree_id=base.repository_tree_id,
            objective_id=base.objective_id,
            baseline_id=baseline_id,
            policy_id=base.policy_id,
            diff_digest=base.diff_digest,
            allowed_paths=base.allowed_paths,
            task_owned_paths=_strings(task_owned_paths),
            changed_paths=base.changed_paths,
            gate_trace=tuple(gate.value for gate in base.gate_trace),
            rejection_codes=base.rejection_codes,
            expensive_node_ids=base.expensive_node_ids,
            expensive_checks_started=0,
        )
        return ProposalValidationReceipt(
            proposal_id=base.proposal_id,
            policy_id=base.policy_id,
            repository_tree_id=base.repository_tree_id,
            objective_id=base.objective_id,
            diff_digest=base.diff_digest,
            allowed_paths=base.allowed_paths,
            changed_paths=base.changed_paths,
            accepted=base.accepted,
            findings=base.findings,
            gate_trace=base.gate_trace,
            expensive_node_ids=base.expensive_node_ids,
            expensive_checks_started=0,
            rejection_evidence=evidence,
            receipt_id=base.receipt_id,
        )


@dataclass(frozen=True)
class ProposalValidationResult:
    proposal: ImplementationProposal
    policy: ProposalValidationPolicy
    receipt: ProposalValidationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, ImplementationProposal):
            object.__setattr__(
                self, "proposal", ImplementationProposal.from_dict(self.proposal)
            )
        if not isinstance(self.policy, ProposalValidationPolicy):
            object.__setattr__(
                self, "policy", ProposalValidationPolicy.from_dict(self.policy)
            )
        if not isinstance(self.receipt, ProposalValidationReceipt):
            object.__setattr__(
                self, "receipt", ProposalValidationReceipt.from_dict(self.receipt)
            )
        if (
            self.receipt.proposal_id != self.proposal.proposal_id
            or self.receipt.policy_id != self.policy.policy_id
            or self.receipt.repository_tree_id != self.proposal.repository_tree_id
            or self.receipt.objective_id != self.proposal.objective_id
            or self.receipt.diff_digest != self.proposal.diff_digest
            or self.receipt.allowed_paths != self.policy.allowed_paths
            or self.receipt.changed_paths != self.proposal.changed_paths
        ):
            raise ProposalValidationError("proposal result binding mismatch")
        rejection = self.receipt.rejection_evidence
        if rejection is not None and (
            rejection.task_id != self.proposal.task_id
            or rejection.repository_id != self.proposal.repository_id
            or rejection.baseline_id != self.proposal.baseline_id
            or rejection.task_owned_paths != self.policy.task_owned_paths
        ):
            raise ProposalValidationError(
                "rejection evidence is detached from proposal authority"
            )

    @property
    def accepted(self) -> bool:
        return self.receipt.accepted

    @property
    def findings(self) -> tuple[ProposalValidationFinding, ...]:
        return self.receipt.findings

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        """Project only requirement evidence produced by the bound receipt.

        Proposal admission cannot manufacture proof claims.  The only
        requirement currently produced at this layer is scheduler-bound
        fail-fast rejection evidence; accepted proposals therefore always
        project an empty population.
        """

        return self.receipt.proved_requirement_ids

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def merge_eligible(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def freshness_authoritative(self) -> bool:
        return False

    @property
    def admission_binding(self) -> Mapping[str, object]:
        """Project the complete accepted authority consumed downstream.

        The projection is diagnostic binding data, never proof authority. It
        gives validation, semantic, and completion receipts one canonical
        vocabulary for the accepted proposal, policy, tree, and change
        identities instead of letting each consumer select a weaker subset.
        """

        self.require_admitted_binding()
        return {
            "task_id": self.proposal.task_id,
            "accepted_plan_id": self.proposal.accepted_plan_id,
            "repository_id": self.proposal.repository_id,
            "repository_tree_id": self.proposal.repository_tree_id,
            "objective_id": self.proposal.objective_id,
            "baseline_id": self.proposal.baseline_id,
            "context_id": self.proposal.context_id,
            "proposal_id": self.proposal.proposal_id,
            "policy_id": self.policy.policy_id,
            "receipt_id": self.receipt.receipt_id,
            "diff_digest": self.proposal.diff_digest,
            "changed_paths": self.proposal.changed_paths,
            "accepted": True,
            "proof_authoritative": False,
            "completion_authoritative": False,
            "merge_eligible": False,
            "authoritative": False,
            "freshness_authoritative": False,
        }

    def require_admitted_binding(
        self,
        *,
        task_id: str = "",
        accepted_plan_id: str = "",
        repository_id: str = "",
        repository_tree_id: str = "",
        objective_id: str = "",
        baseline_id: str = "",
        context_id: str = "",
        proposal_id: str = "",
        policy_id: str = "",
        receipt_id: str = "",
        diff_digest: str = "",
    ) -> "ProposalValidationResult":
        """Return this result only when it is the exact accepted authority.

        Downstream validation, proof, and completion bridges use this helper
        instead of partially repeating proposal binding checks.  Admission
        never grants proof or completion authority.
        """

        if not self.accepted:
            raise ProposalValidationError(
                "rejected proposal cannot create downstream validation authority"
            )
        expected = {
            "task_id": str(task_id or "").strip(),
            "accepted_plan_id": str(accepted_plan_id or "").strip(),
            "repository_id": str(repository_id or "").strip(),
            "repository_tree_id": str(repository_tree_id or "").strip(),
            "objective_id": str(objective_id or "").strip(),
            "baseline_id": str(baseline_id or "").strip(),
            "context_id": str(context_id or "").strip(),
            "proposal_id": str(proposal_id or "").strip(),
            "policy_id": str(policy_id or "").strip(),
            "receipt_id": str(receipt_id or "").strip(),
            "diff_digest": str(diff_digest or "").strip(),
        }
        actual = {
            "task_id": self.proposal.task_id,
            "accepted_plan_id": self.proposal.accepted_plan_id,
            "repository_id": self.proposal.repository_id,
            "repository_tree_id": self.proposal.repository_tree_id,
            "objective_id": self.proposal.objective_id,
            "baseline_id": self.proposal.baseline_id,
            "context_id": self.proposal.context_id,
            "proposal_id": self.proposal.proposal_id,
            "policy_id": self.policy.policy_id,
            "receipt_id": self.receipt.receipt_id,
            "diff_digest": self.proposal.diff_digest,
        }
        mismatched = tuple(
            name
            for name, value in expected.items()
            if value and value != actual[name]
        )
        if mismatched:
            raise ProposalValidationError(
                "proposal admission binding mismatch: " + ", ".join(mismatched)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "policy": self.policy.to_dict(),
            "receipt": self.receipt.to_dict(),
            "accepted": self.accepted,
            "proved_requirement_ids": self.proved_requirement_ids,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
            "merge_eligible": False,
            "authoritative": False,
            "freshness_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalValidationResult":
        for field_name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
            "merge_eligible",
            "authoritative",
            "freshness_authoritative",
        ):
            if payload.get(field_name) not in (None, False):
                raise ProposalValidationError(
                    f"proposal result cannot claim {field_name}"
                )
        result = cls(
            proposal=ImplementationProposal.from_dict(payload.get("proposal") or {}),
            policy=ProposalValidationPolicy.from_dict(payload.get("policy") or {}),
            receipt=ProposalValidationReceipt.from_dict(payload.get("receipt") or {}),
        )
        if "accepted" in payload and bool(payload["accepted"]) != result.accepted:
            raise ProposalValidationError("proposal result verdict mismatch")
        claimed_requirements = _requirement_claims(payload)
        if (
            claimed_requirements is not None
            and claimed_requirements != result.proved_requirement_ids
        ):
            raise ProposalValidationError(
                "proposal result requirement claims mismatch"
            )
        return result

    def with_dispatch_outcome(
        self,
        *,
        expensive_node_ids: Iterable[str],
        expensive_checks_started: int,
    ) -> "ProposalValidationResult":
        return ProposalValidationResult(
            proposal=self.proposal,
            policy=self.policy,
            receipt=self.receipt.with_dispatch_outcome(
                expensive_node_ids=expensive_node_ids,
                expensive_checks_started=expensive_checks_started,
                task_id=self.proposal.task_id,
                repository_id=self.proposal.repository_id,
                baseline_id=self.proposal.baseline_id,
                task_owned_paths=self.policy.task_owned_paths,
            ),
        )

    def evaluate_objective_completion(
        self,
        *,
        producing_tasks: Sequence[Any] = (),
        current_state: Any = "active",
        evidence: Sequence[Any] = (),
        tasks_complete: bool = False,
        coverage: Any = None,
        analyzer_health: Any = None,
        exhaustion_quorum: Any = None,
        required_exhaustive_receipts: int = (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS
        ),
        child_goals: Sequence[Any] = (),
        now: Any = None,
        freshness_seconds: float = 3600.0,
        clock_skew_seconds: float = 300.0,
        analysis_inconclusive: bool = False,
        blocked_reason: str = "",
    ) -> Any:
        """Evaluate ASI-G100 through its closed current-tree completion gate.

        A fail-fast rejection is operational evidence, not a passing
        completion validation.  This second phase fixes the producer and
        criterion populations and requires independently fresh validation,
        analyzer-health, and exhaustion records before the shared lifecycle
        may advance.
        """

        return _evaluate_fail_fast_objective_completion(
            self,
            producing_tasks=producing_tasks,
            current_state=current_state,
            evidence=evidence,
            tasks_complete=tasks_complete,
            coverage=coverage,
            analyzer_health=analyzer_health,
            exhaustion_quorum=exhaustion_quorum,
            required_exhaustive_receipts=required_exhaustive_receipts,
            child_goals=child_goals,
            now=now,
            freshness_seconds=freshness_seconds,
            clock_skew_seconds=clock_skew_seconds,
            analysis_inconclusive=analysis_inconclusive,
            blocked_reason=blocked_reason,
        )


def _evaluate_fail_fast_objective_completion(
    result: ProposalValidationResult,
    *,
    producing_tasks: Sequence[Any],
    current_state: Any,
    evidence: Sequence[Any],
    tasks_complete: bool,
    coverage: Any,
    analyzer_health: Any,
    exhaustion_quorum: Any,
    required_exhaustive_receipts: int,
    child_goals: Sequence[Any],
    now: Any,
    freshness_seconds: float,
    clock_skew_seconds: float,
    analysis_inconclusive: bool,
    blocked_reason: str,
) -> Any:
    """Join the G100 operational rejection with independent completion proof."""

    from ..objectives.goal_completion import evaluate_goal_completion

    if (
        isinstance(required_exhaustive_receipts, bool)
        or not isinstance(required_exhaustive_receipts, int)
        or required_exhaustive_receipts
        != NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS
    ):
        raise ValueError(
            "required_exhaustive_receipts must equal the configured ASI-G100 "
            f"count {NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS}"
        )
    if not isinstance(tasks_complete, bool):
        raise ValueError("tasks_complete must be a boolean")
    if not isinstance(analysis_inconclusive, bool):
        raise ValueError("analysis_inconclusive must be a boolean")
    for name, value in (
        ("freshness_seconds", freshness_seconds),
        ("clock_skew_seconds", clock_skew_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a non-negative number")

    def payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return dict(converted)
        return {}

    def normalized(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def parsed_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    current = parsed_datetime(now) or datetime.now(timezone.utc)
    max_age = timedelta(seconds=float(freshness_seconds))
    clock_skew = timedelta(seconds=float(clock_skew_seconds))

    def fresh(value: Any) -> bool:
        observed = parsed_datetime(value)
        return bool(
            observed is not None
            and observed <= current + clock_skew
            and current - observed <= max_age
        )

    proposal = result.proposal
    policy = result.policy
    receipt = result.receipt
    rejection = receipt.rejection_evidence
    qualifying_codes = {code.value for code in QUALIFYING_FAIL_FAST_CODES}
    operational_complete = bool(
        proposal.objective_id == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
        and policy.expected_objective_id
        == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
        and not result.accepted
        and rejection is not None
        and result.proved_requirement_ids
        == (NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,)
        and receipt.expensive_checks_started == 0
        and rejection.expensive_checks_started == 0
        and set(rejection.rejection_codes).intersection(qualifying_codes)
        and receipt.gate_trace == ORDERED_PROPOSAL_GATES
        and rejection.gate_trace
        == tuple(gate.value for gate in ORDERED_PROPOSAL_GATES)
        and rejection.task_id == proposal.task_id
        and rejection.repository_id == proposal.repository_id
        and rejection.proposal_id == proposal.proposal_id
        and rejection.receipt_id == receipt.receipt_id
        and rejection.repository_tree_id == proposal.repository_tree_id
        and rejection.objective_id == proposal.objective_id
        and rejection.baseline_id == proposal.baseline_id
        and rejection.policy_id == policy.policy_id
        and rejection.diff_digest == proposal.diff_digest
        and rejection.allowed_paths == policy.allowed_paths
        and rejection.task_owned_paths == policy.task_owned_paths
        and rejection.changed_paths == proposal.changed_paths
        and result.proof_authoritative is False
        and result.code_proof_authoritative is False
        and result.completion_authoritative is False
        and result.merge_eligible is False
        and result.authoritative is False
        and result.freshness_authoritative is False
    )

    terminal_states = {
        "complete",
        "completed",
        "passed",
        "success",
        "succeeded",
        "verified",
        "verified_complete",
    }
    producer_values = [payload(item) for item in producing_tasks]
    producer_ids = [
        str(item.get("task_id", item.get("id", "")) or "").strip()
        for item in producer_values
    ]
    producer_population_complete = bool(
        len(producer_ids)
        == len(NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_PRODUCING_TASK_IDS)
        and len(producer_ids) == len(set(producer_ids))
        and set(producer_ids)
        == set(NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_PRODUCING_TASK_IDS)
        and all(
            normalized(item.get("status", item.get("state", "")))
            in terminal_states
            for item in producer_values
        )
    )

    expected_criteria = {
        normalized(item)
        for item in NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA
    }
    evidence_values: list[dict[str, Any]] = []
    receipt_ids_by_criterion: dict[str, set[str]] = {}
    evidence_criteria: list[str] = []
    evidence_bound = True
    for item in evidence:
        record = payload(item)
        source_value = record.get("evidence", record)
        source = (
            dict(source_value)
            if isinstance(source_value, Mapping)
            else record
        )
        evidence_values.append(source)
        criterion = normalized(
            source.get(
                "acceptance_criterion",
                source.get("criterion", source.get("acceptance", "")),
            )
        )
        evidence_criteria.append(criterion)
        receipt_id = str(
            source.get(
                "provenance_cid",
                source.get(
                    "receipt_id",
                    source.get("evidence_id", source.get("receipt_cid", "")),
                ),
            )
            or ""
        ).strip()
        if criterion and receipt_id:
            receipt_ids_by_criterion.setdefault(criterion, set()).add(
                receipt_id
            )
        validation = source.get("validation_receipt")
        validation = validation if isinstance(validation, Mapping) else {}
        evidence_bound = bool(
            evidence_bound
            and source.get("validation_passed") is True
            and source.get("repository_tree")
            == proposal.repository_tree_id
            and normalized(validation.get("status")) in {"passed", "verified"}
            and validation.get("requirement_id")
            == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID
            and validation.get("objective_id")
            == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
            and validation.get("repository_id") == proposal.repository_id
            and validation.get("tree_id") == proposal.repository_tree_id
            and validation.get("validation_policy_id") == policy.policy_id
            and validation.get("operational_receipt_id")
            == (rejection.evidence_id if rejection is not None else "")
        )
    evidence_population_complete = bool(
        operational_complete
        and evidence_bound
        and len(evidence_values) == len(expected_criteria)
        and len(evidence_criteria) == len(set(evidence_criteria))
        and set(evidence_criteria) == expected_criteria
        and all(
            len(receipt_ids_by_criterion.get(criterion, set())) == 1
            for criterion in expected_criteria
        )
    )

    coverage_projection = getattr(coverage, "completion_gate_evidence", None)
    if callable(coverage_projection):
        try:
            projected = coverage_projection(
                NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
            )
        except (TypeError, ValueError):
            projected = {}
        coverage_value = dict(projected) if isinstance(projected, Mapping) else {}
    else:
        coverage_value = payload(coverage)
    rows_value = coverage_value.get("criteria")
    rows = rows_value if isinstance(rows_value, list) else []

    def row_criterion(row: Mapping[str, Any]) -> str:
        return normalized(
            row.get(
                "criterion",
                row.get(
                    "acceptance_criterion",
                    row.get("acceptance", ""),
                ),
            )
        )

    def implementation_bound(row: Mapping[str, Any]) -> bool:
        for name in (
            "implementation",
            "implementation_binding",
            "changed_files",
            "predicted_files",
            "ast_symbols",
            "interfaces",
        ):
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                return True
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and any(str(item or "").strip() for item in value)
            ):
                return True
        return False

    def validation_ids(row: Mapping[str, Any]) -> set[str]:
        raw = row.get(
            "validation_receipt_ids",
            row.get("validation_receipt_id", ()),
        )
        if isinstance(raw, str):
            raw = (raw,)
        if not (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes, bytearray))
        ):
            return set()
        return {
            str(item or "").strip()
            for item in raw
            if str(item or "").strip()
        }

    row_keys = [
        row_criterion(row) for row in rows if isinstance(row, Mapping)
    ]
    coverage_bound = bool(
        evidence_population_complete
        and coverage_value.get("verified") is True
        and coverage_value.get("repository_tree")
        == proposal.repository_tree_id
        and coverage_value.get(
            "repository_id", proposal.repository_id
        )
        == proposal.repository_id
        and len(row_keys) == len(set(row_keys)) == len(expected_criteria)
        and set(row_keys) == expected_criteria
        and all(
            isinstance(row, Mapping)
            and implementation_bound(row)
            and len(validation_ids(row)) == 1
            and validation_ids(row)
            == receipt_ids_by_criterion.get(row_criterion(row), set())
            for row in rows
        )
    )
    if not coverage_bound:
        coverage_value = {
            **coverage_value,
            "verified": False,
            "passed": False,
            "reason_codes": [
                (
                    "validation_evidence_population_incomplete"
                    if not evidence_population_complete
                    else "coverage_validation_receipt_unbound"
                )
            ],
        }

    expected_binding = {
        "repository_id": proposal.repository_id,
        "tree_id": proposal.repository_tree_id,
        "analyzer_version": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_CONFIGURATION_REVISION
        ),
        "objective_revision": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_REVISION
        ),
    }
    health_value = payload(analyzer_health)
    health_binding_value = health_value.get("binding")
    health_binding = (
        {
            key: health_binding_value.get(key)
            for key in expected_binding
        }
        if isinstance(health_binding_value, Mapping)
        else {}
    )
    health_valid = bool(
        health_binding == expected_binding
        and normalized(health_value.get("status")) == "healthy"
        and health_value.get("healthy") is True
        and health_value.get("safe_for_completion_reasoning") is True
    )
    if not health_valid:
        health_value = {
            **health_value,
            "healthy": False,
            "safe_for_completion_reasoning": False,
        }

    quorum_value = payload(exhaustion_quorum)
    quorum_binding_value = quorum_value.get("binding")
    quorum_binding = (
        {
            key: quorum_binding_value.get(key)
            for key in expected_binding
        }
        if isinstance(quorum_binding_value, Mapping)
        else {}
    )
    members_value = quorum_value.get("members")
    members = members_value if isinstance(members_value, list) else []

    def independent_member_field(name: str) -> bool:
        values = [
            str(member.get(name) or "").strip()
            for member in members
            if isinstance(member, Mapping)
        ]
        return bool(
            len(values) == len(members)
            and all(values)
            and len(values) == len(set(values))
        )

    quorum_valid = bool(
        quorum_value.get("required_members")
        == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("member_count", len(members)) == len(members)
        and len(members)
        == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS
        and quorum_value.get("satisfied") is True
        and quorum_value.get("quorum_met", quorum_value.get("satisfied")) is True
        and health_valid
        and quorum_binding == expected_binding == health_binding
        and independent_member_field("member_id")
        and independent_member_field("evidence_channel")
        and independent_member_field("receipt_cid")
        and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and normalized(member.get("scan_mode")) == "exhaustive"
            and fresh(member.get("finished_at"))
            and isinstance(member.get("binding"), Mapping)
            and {
                key: member["binding"].get(key)
                for key in expected_binding
            }
            == expected_binding
            for member in members
        )
    )
    if not quorum_valid:
        quorum_value = {
            **quorum_value,
            "satisfied": False,
            "quorum_met": False,
        }

    return evaluate_goal_completion(
        current_state=current_state,
        acceptance_criteria=(
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA
        ),
        evidence=evidence,
        tasks_complete=bool(
            tasks_complete
            and producer_population_complete
            and operational_complete
            and not child_goals
        ),
        repository_tree=proposal.repository_tree_id,
        repository_id=proposal.repository_id,
        now=current,
        freshness_seconds=float(freshness_seconds),
        clock_skew_seconds=float(clock_skew_seconds),
        coverage=coverage_value,
        analyzer_health=health_value,
        exhaustion_quorum=quorum_value,
        child_goals=(),
        analysis_inconclusive=analysis_inconclusive,
        blocked_reason=blocked_reason,
        require_completion_gate=True,
    )


_SHELL_META_RE = re.compile(r"(?:[;&|<>`]|[$]\(|\r|\n)")
# Bare operator tokens (argv already shlex-split). Distinct from metacharacters
# that appear *inside* a longer argument (e.g. ripgrep alternation ``a|b``).
_SHELL_OPERATOR_TOKENS = frozenset(
    {
        ";",
        "|",
        "&",
        "<",
        ">",
        ">>",
        "<<",
        "|&",
        "&>",
        "2>",
        "2>&1",
        "`",
    }
)
# Subshell / control-character expansion remains forbidden even inside args of
# reviewed compound commands, because validation may re-join argv for a shell.
_SHELL_EXPANSION_RE = re.compile(r"(?:[`\r\n]|\$\()")
_PRIVATE_KEY_CONTENT_RE = re.compile(
    r"(?im)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?im)(?:^|[,{;.\s])["']?"""
    r"""(?:api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|"""
    r"""client[_-]?secret|password|passwd)["']?\s*(?P<delimiter>[:=])\s*"""
    r"""(?P<value>(?:[rubf]{0,2})(?:"(?:\\.|[^"\\\r\n])*"|"""
    r"""'(?:\\.|[^'\\\r\n])*')|[^\s,;}#]+)"""
)
_QUOTED_SECRET_VALUE_RE = re.compile(
    r"""(?is)^(?:[rubf]{0,2})(?P<quote>["'])(?P<value>.*)(?P=quote)$"""
)
_DYNAMIC_SECRET_VALUE_RE = re.compile(
    r"""(?ix)^(?:"""
    r"""\$\{|\$[a-z_]|"""
    r"""(?:[a-z_]\w*\.)*[a-z_]\w*\s*\(|"""
    r"""(?:[a-z_]\w*\.)+[a-z_]\w*$|"""
    r"""(?:[a-z_]\w*\.)*[a-z_]\w*\[|"""
    r"""[A-Z][A-Z0-9_]*$|"""
    r"""none$|null$|true$|false$"""
    r""")"""
)
_SECRET_PLACEHOLDER_RE = re.compile(
    r"""(?ix)(?:"""
    r"""example|placeholder|redacted|change[_-]?me|replace[_-]?me|"""
    r"""your[_-](?:api[_-]?key|token|password|secret)|"""
    r"""dummy|fake[_-]?secret"""
    r""")"""
)
_SYNTHETIC_TEST_SECRET_CANARY_RE = re.compile(
    r"""(?ix)^(?:"""
    r"""(?:literal|synthetic|canary|super|test[-_ ]?only)[-_ ]"""
    r"""(?:secret|api[-_ ]?key|access[-_ ]?token|auth[-_ ]?token|"""
    r"""refresh[-_ ]?token|client[-_ ]?secret|password)"""
    r"""(?:[-_ ]value)?|"""
    r"""(?:integration|unit)[-_ ]test[-_ ](?:api[-_ ]?)?key"""
    r"""[-_ ]not[-_ ]secret|"""
    r"""(?:token|key)[-_ ](?:alpha|beta)(?:[-_ ]different)?|"""
    r"""not[-_ ]a[-_ ]real[-_ ]"""
    r"""(?:password|secret|api[-_ ]?key|access[-_ ]?token)|"""
    r"""should[-_ ]not[-_ ]appear"""
    r""")$"""
)
_SYNTHETIC_TEST_SECRET_REFERENCE_RE = re.compile(
    r"""(?x)^(?:env://[A-Z][A-Z0-9_]{1,127}|"""
    r"""vault://[A-Za-z0-9_.][A-Za-z0-9_./-]{0,255})$"""
)
# Exact documentation/redaction sentinel only.  Do not exempt
# ``should-not-appear`` (and similar synthetic canaries) — those remain
# concrete secret-like values in production sources and are rejected there.
_NEVER_EXPOSE_SENTINEL_RE = re.compile(
    r"""(?ix)^(?:should|must)[_-]?never[_-]?(?:appear|persist|log|store|commit)$"""
)
_TEST_ONLY_NON_SECRET_SENTINEL_RE = re.compile(
    r"(?i)^sk[_-]live[_-]not[_-]a[_-]real[_-]key$"
)
_SECRET_CLASSIFICATION_LABEL_RE = re.compile(
    r"""(?ix)^secret[_-]?material$"""
)


def _introduces_secret_content(
    before_source: str | None,
    after_source: str | None,
) -> bool:
    """Return whether a candidate adds or changes secret-like content.

    Candidate entries contain the complete before and after source. Scanning
    only the latter rejects unrelated edits whenever a file already contains
    a secret-like environment lookup. Compare match populations so unchanged
    pre-existing content does not acquire new secret-mutation authority.
    """

    before_matches = Counter(
        match.group(0) for match in _SECRET_CONTENT_RE.finditer(before_source or "")
    )
    after_matches = Counter(
        match.group(0) for match in _SECRET_CONTENT_RE.finditer(after_source or "")
    )
    return any(
        count > before_matches.get(value, 0)
        for value, count in after_matches.items()
    )


_TEST_SKIP_RE = re.compile(
    r"(?im)(?:pytest[.]mark[.](?:skip|xfail)|unittest[.]skip|"
    r"\bskipTest\s*\(|\bassert\s+True\b)"
)
_ASSERTION_RE = re.compile(
    r"(?m)(?:\bassert\b|self[.]assert[A-Z]\w*\s*\(|pytest[.]raises\s*\()"
)


def _path_at_boundary(path: str, boundaries: Sequence[str]) -> bool:
    return any(
        path == boundary.rstrip("/") or path.startswith(boundary.rstrip("/") + "/")
        for boundary in boundaries
    )


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith(("test/", "tests/")) or name.startswith("test_")


def _is_scoped_python_test_source(
    path: str,
    policy: ProposalValidationPolicy,
) -> bool:
    """Return whether ``path`` is a task-owned Python test source.

    Test modules commonly describe the security property they exercise in
    their filename (for example, ``test_wallet_processor_secrets.py``).
    Such a name is not itself evidence that the candidate persists a secret.
    Keep the exception narrow: non-source fixtures and paths outside either
    authority envelope remain subject to the sensitive-path gate.
    """

    return (
        path.endswith((".py", ".pyi"))
        and _is_test_path(path)
        and policy.path_is_in_scope(path)
    )


def _is_scoped_test_fixture(
    path: str,
    policy: ProposalValidationPolicy,
) -> bool:
    """Return whether ``path`` is inside a task-owned test-fixture tree.

    Fixture files may need inert credential-shaped canaries to prove that an
    importer rejects prohibited material.  The content exception remains
    narrower than ordinary test-source authority: it applies only below the
    conventional fixture roots, only inside both policy scope envelopes, and
    only to exact synthetic values recognized below.  Private keys and
    concrete credential values remain forbidden.
    """

    return path.startswith(("test/fixtures/", "tests/fixtures/")) and (
        policy.path_is_in_scope(path)
    )


def _is_inert_test_package_marker_companion(
    entry: CandidateDiffEntry,
    policy: ProposalValidationPolicy,
) -> bool:
    """Allow only an empty test-package marker enclosing declared test work."""

    path = entry.path
    if (
        entry.change_kind is not DiffChangeKind.ADD
        or entry.before_source is not None
        or entry.after_source is None
        or not path.endswith("/__init__.py")
        or not _is_test_path(path)
    ):
        return False
    try:
        if ast.parse(entry.after_source, filename=path).body:
            return False
    except (SyntaxError, TypeError, ValueError):
        return False
    package_prefix = path.rsplit("/", 1)[0] + "/"

    def has_declared_descendant(patterns: Sequence[str]) -> bool:
        return any(
            normalized.startswith(package_prefix)
            and normalized != path
            and not any(character in normalized for character in "*?[")
            for raw_pattern in patterns
            if (
                normalized := str(raw_pattern)
                .strip()
                .replace("\\", "/")
                .removeprefix("./")
            )
        )

    return has_declared_descendant(
        policy.allowed_paths
    ) and has_declared_descendant(policy.task_owned_paths)


def _introduced_candidate_text(entry: CandidateDiffEntry) -> str:
    """Return only candidate lines not already present in the baseline.

    A linear multiset subtraction is enough for secret admission: moved lines
    are not new authority, while duplicated and modified lines remain visible.
    It also avoids an adversarial quadratic sequence diff in this fail-fast
    gate.
    """

    after = entry.after_source
    if after is None or entry.change_kind is DiffChangeKind.DELETE:
        return ""
    before = entry.before_source
    if before is None or entry.change_kind in {
        DiffChangeKind.ADD,
        DiffChangeKind.COPY,
        DiffChangeKind.RENAME,
    }:
        return after
    if before == after:
        return ""

    baseline_lines = Counter(before.splitlines(keepends=True))
    introduced: list[str] = []
    for line in after.splitlines(keepends=True):
        if baseline_lines[line]:
            baseline_lines[line] -= 1
        else:
            introduced.append(line)
    return "".join(introduced)


def _is_concrete_secret_value(
    raw_value: str,
    *,
    allow_test_sentinel: bool = False,
    allow_never_expose_sentinel: bool = False,
) -> bool:
    value = raw_value.strip()
    quoted = _QUOTED_SECRET_VALUE_RE.fullmatch(value)
    if quoted:
        value = quoted.group("value").strip()
    elif _DYNAMIC_SECRET_VALUE_RE.match(value):
        return False

    if len(value) < 12:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]{11,}", value):
        return False
    if _SECRET_PLACEHOLDER_RE.search(value):
        return False
    # Public-boundary schemas may map sensitive field names to this exact
    # classification label.  It describes how a value must be handled; it is
    # not credential material.  Keep this exception exact so a longer value
    # containing the same words still fails closed.
    if _SECRET_CLASSIFICATION_LABEL_RE.fullmatch(value):
        return False
    # Security tests commonly need a deterministic value that proves secret
    # material is rejected or redacted. Only accept an exact "never expose"
    # sentinel so a concrete credential containing those words still fails
    # closed.
    if (
        allow_never_expose_sentinel
        and _NEVER_EXPOSE_SENTINEL_RE.fullmatch(value)
    ):
        return False
    # A focused security fixture uses this exact value to exercise rejection
    # of secret-bearing fields.  Admit it only in test files and only as the
    # complete literal; prefixes/suffixes remain concrete secret material.
    if allow_test_sentinel and _TEST_ONLY_NON_SECRET_SENTINEL_RE.fullmatch(value):
        return False
    return True


def _is_synthetic_test_secret_canary(raw_value: str) -> bool:
    """Return whether a value is an explicit non-credential test canary."""

    value = raw_value.strip()
    quoted = _QUOTED_SECRET_VALUE_RE.fullmatch(value)
    if quoted:
        value = quoted.group("value").strip()
    return bool(
        _SYNTHETIC_TEST_SECRET_CANARY_RE.fullmatch(value)
        or _SYNTHETIC_TEST_SECRET_REFERENCE_RE.fullmatch(value)
    )


def _entry_introduces_secret(
    entry: CandidateDiffEntry,
    *,
    allow_synthetic_test_canaries: bool = False,
) -> bool:
    introduced = _introduced_candidate_text(entry)
    if not introduced:
        return False
    if _PRIVATE_KEY_CONTENT_RE.search(introduced):
        return True
    path = entry.new_path or entry.old_path
    allow_test_sentinel = _is_test_path(path)
    # The proposal gate must be able to define and maintain its own closed
    # never-expose sentinel vocabulary.  Outside that authority module the
    # same credential-shaped literals remain test-only canaries.
    allow_never_expose_sentinel = (
        allow_test_sentinel
        or path.endswith("/proposal_validation.py")
    )
    for match in _SECRET_ASSIGNMENT_RE.finditer(introduced):
        value = match.group("value")
        if not _is_concrete_secret_value(
            value,
            allow_test_sentinel=allow_test_sentinel,
            allow_never_expose_sentinel=allow_never_expose_sentinel,
        ):
            continue
        if (
            allow_synthetic_test_canaries
            and _is_synthetic_test_secret_canary(value)
        ):
            continue
        return True
    return False


def _python_test_names(source: str) -> frozenset[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return frozenset()
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )


def _metadata_size(metadata: Mapping[str, Any]) -> int:
    sizes = [0]
    for name in ("size_bytes", "before_size_bytes", "after_size_bytes"):
        try:
            value = int(metadata.get(name) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            sizes.append(value)
    return max(sizes)


def _command_is_allowed(
    command: Sequence[str], prefixes: Sequence[Sequence[str]]
) -> bool:
    """Return whether argv is allowed under the reviewed command prefixes.

    Task boards store validation as shell text and may include reviewed ``&&``
    / ``||`` compound commands. Exact full-command allowlist hits may use those
    separators, but every clause must still satisfy the normal executable and
    eval guards.

    For exact allowlisted compounds, ``|`` / ``;`` / ``&`` characters that appear
    *inside* a longer argv token (for example a ripgrep alternation pattern) are
    not treated as shell operators — the argv was already ``shlex``-split. Bare
    operator tokens, shell interpreters, eval flags, and subshell/backtick
    expansion syntax remain forbidden. Prefix matches still reject any token
    that embeds shell metacharacters.
    """

    command_t = tuple(str(part) for part in command)
    prefixes_t = tuple(tuple(str(part) for part in prefix) for prefix in prefixes)

    def clause_executable_is_safe(clause: tuple[str, ...]) -> bool:
        if not clause:
            return False
        executable = (
            clause[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
        )
        executable = executable.removesuffix(".exe")
        if executable in {
            "bash",
            "cmd",
            "dash",
            "fish",
            "ksh",
            "powershell",
            "pwsh",
            "sh",
            "zsh",
        }:
            return False
        if (
            len(clause) >= 2
            and executable in {"node", "perl", "python", "python3", "ruby"}
            and clause[1] in {"-c", "-e", "--eval"}
        ):
            return False
        return True

    def clause_is_safe(clause: tuple[str, ...]) -> bool:
        if not clause or any(_SHELL_META_RE.search(part) for part in clause):
            return False
        return clause_executable_is_safe(clause)

    def compound_clause_is_safe(clause: tuple[str, ...]) -> bool:
        """Safety for a clause of an exact-allowlisted ``&&`` / ``||`` chain.

        Allows regex/path characters such as ``|`` inside longer tokens while
        still denying bare operators, empty clauses, shells, and eval forms.
        """

        if not clause:
            return False
        for part in clause:
            if part in _SHELL_OPERATOR_TOKENS or part in {"&&", "||"}:
                return False
            if _SHELL_EXPANSION_RE.search(part):
                return False
        return clause_executable_is_safe(clause)

    if command_t in prefixes_t and ("&&" in command_t or "||" in command_t):
        clauses: list[tuple[str, ...]] = []
        start = 0
        for index, part in enumerate(command_t):
            if part in {"&&", "||"}:
                clauses.append(command_t[start:index])
                start = index + 1
            elif part in _SHELL_OPERATOR_TOKENS or _SHELL_EXPANSION_RE.search(part):
                return False
        clauses.append(command_t[start:])
        return all(compound_clause_is_safe(clause) for clause in clauses)

    # Exact task-board allowlist hit: trust the reviewed command when the
    # executable itself is not a shell interpreter. This admits env-assignment
    # prefixes (PYTHONPATH=...) that are already on the task validation plan.
    # Bare operator tokens and subshell expansion remain forbidden even when
    # the exact argv is allowlisted (an allowlist cannot waive shell chaining).
    if command_t in prefixes_t and clause_executable_is_safe(command_t):
        if any(
            part in _SHELL_OPERATOR_TOKENS or part in {"&&", "||"}
            for part in command_t
        ):
            return False
        if any(_SHELL_EXPANSION_RE.search(part) for part in command_t):
            return False
        return True

    if not clause_is_safe(command_t):
        return False
    return any(
        len(command_t) >= len(prefix) and command_t[: len(prefix)] == prefix
        for prefix in prefixes_t
    )


def _entry_operation(entry: CandidateDiffEntry) -> tuple[str, str, str]:
    return (entry.change_kind.value, entry.path, entry.old_path)


def _operation_matches_entry(
    operation: ProposalOperation, entry: CandidateDiffEntry
) -> bool:
    if operation.operation != entry.change_kind.value or operation.path != entry.path:
        return False
    if operation.operation in {"rename", "copy"}:
        return operation.old_path == entry.old_path
    return not operation.old_path or operation.old_path == entry.old_path


def _patch_content_matches(
    patch_text: str,
    parsed_files: Sequence[ParsedPatchFile],
    entries: Sequence[CandidateDiffEntry],
) -> bool:
    """Apply hunks in memory and compare them with materialized candidate text."""

    sections = re.split(r"(?=^diff --git )", patch_text, flags=re.MULTILINE)
    sections = [section for section in sections if section.startswith("diff --git ")]
    if len(sections) != len(parsed_files):
        return False
    remaining = list(entries)
    for section, parsed in zip(sections, parsed_files):
        match_index = next(
            (
                index
                for index, entry in enumerate(remaining)
                if entry.path == (parsed.new_path or parsed.old_path)
                or (
                    entry.old_path == parsed.old_path
                    and entry.new_path == parsed.new_path
                )
            ),
            None,
        )
        if match_index is None:
            return False
        entry = remaining.pop(match_index)
        if entry.binary or (
            entry.before_source is None and entry.after_source is None
        ):
            continue
        before_source = entry.before_source or ""
        after_source = entry.after_source or ""
        before_lines = before_source.splitlines()
        result: list[str] = []
        cursor = 0
        section_lines = section.splitlines()
        line_index = 0
        saw_hunk = False
        while line_index < len(section_lines):
            header = re.match(
                r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$",
                section_lines[line_index],
            )
            if header is None:
                line_index += 1
                continue
            saw_hunk = True
            old_start = int(header.group(1))
            old_count = int(header.group(2)) if header.group(2) is not None else 1
            new_count = int(header.group(4)) if header.group(4) is not None else 1
            old_index = max(0, old_start - 1)
            if old_index < cursor or old_index > len(before_lines):
                return False
            result.extend(before_lines[cursor:old_index])
            cursor = old_index
            consumed_old = produced_new = 0
            line_index += 1
            while line_index < len(section_lines):
                line = section_lines[line_index]
                if line.startswith(("@@ ", "diff --git ")):
                    break
                if line == r"\ No newline at end of file":
                    line_index += 1
                    continue
                if not line or line[0] not in {" ", "+", "-"}:
                    break
                text = line[1:]
                if line[0] in {" ", "-"}:
                    if cursor >= len(before_lines) or before_lines[cursor] != text:
                        return False
                    cursor += 1
                    consumed_old += 1
                if line[0] in {" ", "+"}:
                    result.append(text)
                    produced_new += 1
                line_index += 1
                if consumed_old == old_count and produced_new == new_count:
                    break
            if consumed_old != old_count or produced_new != new_count:
                return False
        if saw_hunk:
            result.extend(before_lines[cursor:])
            if result != after_source.splitlines():
                return False
        elif before_source != after_source:
            return False
    return not remaining


class ProposalValidator:
    """Deterministic evaluator for the strict proposal envelope."""

    def __init__(self, policy: ProposalValidationPolicy) -> None:
        self.policy = policy

    def validate(
        self, proposal: ImplementationProposal | Mapping[str, Any]
    ) -> ProposalValidationResult:
        input_fields: frozenset[str] = frozenset()
        if not isinstance(proposal, ImplementationProposal):
            input_fields = frozenset(str(key) for key in proposal)
            proposal = ImplementationProposal.from_dict(proposal)
        policy = self.policy
        findings: list[ProposalValidationFinding] = []

        def add(
            code: ProposalFindingCode,
            gate: ProposalGate,
            message: str,
            path: str = "",
        ) -> None:
            if len(findings) < policy.max_findings:
                findings.append(ProposalValidationFinding(code, gate, message, path))

        strict = (
            proposal.proposal_version == "2"
            or policy.require_structured_details
            or policy.require_patch_text
        )
        if strict and input_fields:
            allowed_fields = frozenset(proposal.to_dict())
            unexpected_fields = input_fields - allowed_fields
            if unexpected_fields:
                add(
                    ProposalFindingCode.INVALID_SCHEMA,
                    ProposalGate.SCHEMA,
                    "structured proposal contains undeclared top-level fields",
                )
            missing_fields = frozenset(proposal._identity_payload()) - input_fields
            if missing_fields:
                add(
                    ProposalFindingCode.MISSING_REQUIRED_FIELD,
                    ProposalGate.SCHEMA,
                    "structured proposal omits versioned top-level fields",
                )
        if proposal.proposal_version not in {"1", "2"}:
            add(
                ProposalFindingCode.INVALID_SCHEMA,
                ProposalGate.SCHEMA,
                "unsupported implementation proposal version",
            )
        proposal_payload = proposal._identity_payload()
        if (
            len(_canonical_json(proposal_payload).encode("utf-8"))
            > policy.max_output_bytes
        ):
            add(
                ProposalFindingCode.OUTPUT_TOO_LARGE,
                ProposalGate.SCHEMA,
                "structured proposal exceeds the output byte bound",
            )
        if _output_depth(proposal_payload) > policy.max_output_depth:
            add(
                ProposalFindingCode.OUTPUT_TOO_DEEP,
                ProposalGate.SCHEMA,
                "structured proposal exceeds the nesting depth bound",
            )

        if strict:
            required_components = (
                ("operations", proposal.operations),
                ("rationale_references", proposal.rationale_references),
                ("validation_plan", proposal.validation_plan),
                ("risks", proposal.risks),
                ("authority_claims", proposal.authority_claims),
                ("patch_text", proposal.patch_text),
                ("replay_nonce", proposal.replay_nonce),
            )
            for name, value in required_components:
                if not value:
                    add(
                        ProposalFindingCode.MISSING_REQUIRED_FIELD,
                        ProposalGate.STRUCTURE,
                        f"structured proposal requires {name}",
                    )
            if len(proposal.operations) > policy.max_operations:
                add(
                    ProposalFindingCode.OUTPUT_TOO_LARGE,
                    ProposalGate.STRUCTURE,
                    "structured proposal exceeds the operation count bound",
                )
            rationale_refs = set(proposal.rationale_references)
            for operation in proposal.operations:
                if not operation.rationale_refs or not set(
                    operation.rationale_refs
                ).issubset(rationale_refs):
                    add(
                        ProposalFindingCode.MISSING_REQUIRED_FIELD,
                        ProposalGate.STRUCTURE,
                        "operation requires declared rationale references",
                        operation.path,
                    )
            for step in proposal.validation_plan:
                if not step.rationale_refs or not set(step.rationale_refs).issubset(
                    rationale_refs
                ):
                    add(
                        ProposalFindingCode.MISSING_REQUIRED_FIELD,
                        ProposalGate.STRUCTURE,
                        "validation step requires declared rationale references",
                    )

        # Authority is exact and non-compensable.
        expected = (
            ("task_id", policy.expected_task_id),
            ("accepted_plan_id", policy.expected_plan_id),
            ("repository_id", policy.expected_repository_id),
            ("repository_tree_id", policy.expected_repository_tree_id),
            ("objective_id", policy.expected_objective_id),
        )
        for name, required in expected:
            if required and getattr(proposal, name) != required:
                add(
                    ProposalFindingCode.STALE_BASELINE
                    if name == "repository_tree_id"
                    else ProposalFindingCode.AUTHORITY_MISMATCH,
                    ProposalGate.AUTHORITY,
                    f"{name} does not match the frozen proposal authority",
                )

        for name, required, code in (
            (
                "context_id",
                policy.expected_context_id,
                ProposalFindingCode.CONTEXT_MISMATCH,
            ),
            (
                "baseline_id",
                policy.expected_baseline_id,
                ProposalFindingCode.STALE_BASELINE,
            ),
            (
                "replay_nonce",
                policy.expected_replay_nonce,
                ProposalFindingCode.STALE_PROPOSAL_REPLAY,
            ),
        ):
            if required and getattr(proposal, name) != required:
                add(
                    code,
                    ProposalGate.AUTHORITY,
                    f"{name} does not match the frozen proposal authority",
                )
        if proposal.proposal_id in policy.consumed_proposal_ids:
            add(
                ProposalFindingCode.STALE_PROPOSAL_REPLAY,
                ProposalGate.AUTHORITY,
                "proposal identity has already been consumed",
            )

        authority_values = {
            "task_id": proposal.task_id,
            "accepted_plan_id": proposal.accepted_plan_id,
            "repository_id": proposal.repository_id,
            "repository_tree_id": proposal.repository_tree_id,
            "objective_id": proposal.objective_id,
            "baseline_id": proposal.baseline_id,
            "context_id": proposal.context_id,
        }
        if strict:
            for name, actual in authority_values.items():
                if not actual:
                    add(
                        ProposalFindingCode.MISSING_REQUIRED_FIELD,
                        ProposalGate.AUTHORITY,
                        f"structured proposal requires canonical {name}",
                    )
                if proposal.authority_claims.get(name) != actual:
                    add(
                        ProposalFindingCode.FORGED_AUTHORITY_CLAIM,
                        ProposalGate.AUTHORITY,
                        f"authority_claims.{name} is missing or detached",
                    )
        for name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
            "merge_eligible",
            "merge_authoritative",
            "freshness_authoritative",
            "authoritative",
            "authority",
            "completed",
            "proof_complete",
        ):
            if proposal.authority_claims.get(name) not in (None, False):
                add(
                    ProposalFindingCode.FORGED_AUTHORITY_CLAIM,
                    ProposalGate.AUTHORITY,
                    f"implementation proposal cannot claim {name}",
                )

        entries = proposal.candidate_diff
        if len(entries) > policy.max_diff_entries:
            add(
                ProposalFindingCode.PATCH_TOO_LARGE,
                ProposalGate.PATCH,
                "candidate diff exceeds the entry bound",
            )
        patch_bytes = sum(
            len((entry.before_source or "").encode("utf-8", errors="surrogatepass"))
            + len((entry.after_source or "").encode("utf-8", errors="surrogatepass"))
            for entry in entries
        )
        if patch_bytes > policy.max_patch_bytes:
            add(
                ProposalFindingCode.PATCH_TOO_LARGE,
                ProposalGate.PATCH,
                "candidate diff exceeds the byte bound",
            )
        if not entries:
            add(
                ProposalFindingCode.EMPTY_PATCH,
                ProposalGate.PATCH,
                "candidate diff contains no file changes",
            )
        elif not proposal.effective_entries:
            add(
                ProposalFindingCode.NO_SEMANTIC_CHANGE,
                ProposalGate.PATCH,
                "candidate diff has no observable content or path change",
            )
        if len(proposal.expected_effects) > policy.max_expected_effects:
            add(
                ProposalFindingCode.OUTPUT_TOO_LARGE,
                ProposalGate.STRUCTURE,
                "expected effects exceed the count bound",
            )
        if proposal.expected_effects:
            actual_effects = tuple(
                sorted(
                    (
                        entry.change_kind.value,
                        entry.path,
                        _source_digest(entry.before_source),
                        _source_digest(entry.after_source),
                    )
                    for entry in entries
                )
            )
            declared_effects = tuple(
                sorted(
                    (
                        effect.operation,
                        effect.path,
                        effect.before_sha256,
                        effect.after_sha256,
                    )
                    for effect in proposal.expected_effects
                )
            )
            if declared_effects != actual_effects:
                add(
                    ProposalFindingCode.EXPECTED_EFFECT_MISMATCH,
                    ProposalGate.PATCH,
                    "declared expected effects do not exactly match candidate content",
                )
        for entry in entries:
            old_required = entry.change_kind in {
                DiffChangeKind.MODIFY,
                DiffChangeKind.DELETE,
                DiffChangeKind.RENAME,
                DiffChangeKind.COPY,
                DiffChangeKind.TYPE_CHANGE,
            }
            new_required = entry.change_kind in {
                DiffChangeKind.ADD,
                DiffChangeKind.MODIFY,
                DiffChangeKind.RENAME,
                DiffChangeKind.COPY,
                DiffChangeKind.TYPE_CHANGE,
            }
            old_forbidden = entry.change_kind is DiffChangeKind.ADD
            new_forbidden = entry.change_kind is DiffChangeKind.DELETE
            malformed_shape = bool(
                entry.change_kind is DiffChangeKind.UNKNOWN
                or (old_required and not entry.old_path)
                or (new_required and not entry.new_path)
                or (old_forbidden and entry.old_path)
                or (new_forbidden and entry.new_path)
                or (
                    entry.change_kind is DiffChangeKind.RENAME
                    and entry.old_path == entry.new_path
                )
            )
            if malformed_shape:
                add(
                    ProposalFindingCode.UNSAFE_PATH,
                    ProposalGate.PATH,
                    "candidate operation has an unsafe or incomplete path shape",
                    entry.new_path or entry.old_path,
                )

        if proposal.operations:
            unmatched = list(entries)
            for operation in proposal.operations:
                matched_index = next(
                    (
                        index
                        for index, entry in enumerate(unmatched)
                        if _operation_matches_entry(operation, entry)
                    ),
                    None,
                )
                if matched_index is None:
                    add(
                        ProposalFindingCode.OPERATION_MISMATCH,
                        ProposalGate.PATCH,
                        "declared operation does not match candidate diff",
                        operation.path,
                    )
                else:
                    unmatched.pop(matched_index)
            if unmatched:
                add(
                    ProposalFindingCode.OPERATION_MISMATCH,
                    ProposalGate.PATCH,
                    "candidate diff contains undeclared operations",
                    unmatched[0].path,
                )

        parsed_patch: tuple[ParsedPatchFile, ...] = ()
        if proposal.patch_text:
            try:
                parsed_patch = parse_unified_patch(
                    proposal.patch_text,
                    max_files=policy.max_diff_entries,
                    max_bytes=policy.max_patch_bytes,
                    allow_binary=policy.allow_binary,
                )
            except ProposalValidationError as exc:
                add(
                    ProposalFindingCode.PATCH_PARSE_ERROR,
                    ProposalGate.PATCH,
                    str(exc),
                )
            else:
                parsed_projection = tuple(
                    (item.operation, item.new_path or item.old_path, item.old_path)
                    for item in parsed_patch
                )
                candidate_projection = tuple(_entry_operation(entry) for entry in entries)
                if len(parsed_projection) != len(candidate_projection) or any(
                    not any(
                        parsed_kind == kind
                        and parsed_path == path
                        and (
                            kind not in {"rename", "copy"}
                            or parsed_old == old_path
                        )
                        for kind, path, old_path in candidate_projection
                    )
                    for parsed_kind, parsed_path, parsed_old in parsed_projection
                ):
                    add(
                        ProposalFindingCode.PATCH_MISMATCH,
                        ProposalGate.PATCH,
                        "patch paths or operations do not match candidate diff",
                    )
                elif not _patch_content_matches(
                    proposal.patch_text, parsed_patch, entries
                ):
                    add(
                        ProposalFindingCode.PATCH_MISMATCH,
                        ProposalGate.PATCH,
                        "patch content does not match materialized candidate diff",
                    )
        elif policy.require_patch_text and not strict:
            # Kept separate for policies that opt a v1 caller into raw patch
            # verification without all v2 structured fields.
            add(
                ProposalFindingCode.MISSING_REQUIRED_FIELD,
                ProposalGate.PATCH,
                "proposal requires patch_text",
            )

        actual_paths = proposal.changed_paths
        if policy.require_declared_paths and proposal.declared_paths != actual_paths:
            add(
                ProposalFindingCode.DECLARED_PATH_MISMATCH,
                ProposalGate.PATH,
                "declared paths do not exactly match the normalized candidate diff",
            )
        for entry in entries:
            inert_test_package_marker = (
                _is_inert_test_package_marker_companion(entry, policy)
            )
            for path in (entry.old_path, entry.new_path):
                if not path:
                    continue
                if path.startswith("/") or path == ".git" or path.startswith(".git/"):
                    add(
                        ProposalFindingCode.UNSAFE_PATH,
                        ProposalGate.PATH,
                        "candidate path crosses a protected repository boundary",
                        path,
                    )
                    continue
                if any(_path_matches(path, denied) for denied in policy.forbidden_paths):
                    add(
                        ProposalFindingCode.UNSAFE_PATH,
                        ProposalGate.PATH,
                        "candidate path is forbidden by repository policy",
                        path,
                    )
                if any(
                    _path_matches(path, protected)
                    for protected in policy.protected_paths
                ):
                    add(
                        ProposalFindingCode.PROTECTED_PATH_FORBIDDEN,
                        ProposalGate.PATH,
                        "candidate path is operator-protected",
                        path,
                    )
                if _path_at_boundary(path, policy.symlink_paths):
                    add(
                        ProposalFindingCode.SYMLINK_BOUNDARY_FORBIDDEN,
                        ProposalGate.PATH,
                        "candidate path crosses a symlink boundary",
                        path,
                    )
                if _path_at_boundary(path, policy.submodule_paths):
                    add(
                        ProposalFindingCode.SUBMODULE_BOUNDARY_FORBIDDEN,
                        ProposalGate.PATH,
                        "candidate path crosses a submodule boundary",
                        path,
                    )
                if (
                    not policy.path_is_allowed(path)
                    and not inert_test_package_marker
                ):
                    add(
                        ProposalFindingCode.PATH_OUTSIDE_SCOPE,
                        ProposalGate.PATH,
                        "candidate path is outside the task-owned scope",
                        path,
                    )
                if (
                    not policy.path_is_task_owned(path)
                    and not inert_test_package_marker
                ):
                    add(
                        ProposalFindingCode.PATH_OUTSIDE_SCOPE,
                        ProposalGate.PATH,
                        "candidate path is outside the immutable task-owned scope",
                        path,
                    )
            if entry.binary and not policy.allow_binary:
                add(
                    ProposalFindingCode.BINARY_CHANGE_FORBIDDEN,
                    ProposalGate.PATH,
                    "binary changes require explicit policy authority",
                    entry.path,
                )
            source_bytes = (entry.after_source or "").encode(
                "utf-8", errors="surrogatepass"
            )
            if not policy.allow_binary and _looks_binary(source_bytes):
                add(
                    ProposalFindingCode.BINARY_CHANGE_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "binary candidate content requires explicit policy authority",
                    entry.path,
                )
            if not policy.allow_archives and _looks_archive(entry.path, source_bytes):
                add(
                    ProposalFindingCode.ARCHIVE_CHANGE_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "archive changes require explicit policy authority",
                    entry.path,
                )
            if (
                not policy.allow_large_files
                and max(
                    len((entry.before_source or "").encode("utf-8", errors="surrogatepass")),
                    len((entry.after_source or "").encode("utf-8", errors="surrogatepass")),
                    _metadata_size(entry.metadata),
                )
                > policy.max_file_bytes
            ):
                add(
                    ProposalFindingCode.LARGE_FILE_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "large-file changes require explicit policy authority",
                    entry.path,
                )
            sensitive_path = any(
                _path_matches(entry.path, pattern)
                or fnmatch.fnmatchcase(entry.path, pattern)
                or fnmatch.fnmatchcase(entry.path.rsplit("/", 1)[-1], pattern)
                for pattern in policy.sensitive_path_patterns
            )
            scoped_python_test_source = _is_scoped_python_test_source(
                entry.path,
                policy,
            )
            scoped_test_fixture = _is_scoped_test_fixture(
                entry.path,
                policy,
            )
            sensitive_content = _entry_introduces_secret(
                entry,
                allow_synthetic_test_canaries=(
                    scoped_python_test_source or scoped_test_fixture
                ),
            )
            path_requires_secret_authority = (
                sensitive_path
                and not scoped_python_test_source
            )
            if not policy.allow_secrets and (
                path_requires_secret_authority or sensitive_content
            ):
                add(
                    ProposalFindingCode.SECRET_CHANGE_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "secret-bearing paths or content require explicit authority",
                    entry.path,
                )
            if entry.generated is True and not policy.allow_generated:
                add(
                    ProposalFindingCode.GENERATED_CHANGE_FORBIDDEN,
                    ProposalGate.PATH,
                    "generated-file changes require explicit policy authority",
                    entry.path,
                )
            if (
                not policy.allow_generated
                and entry.generated is not True
                and (
                    any(
                        _path_matches(entry.path, pattern)
                        or fnmatch.fnmatchcase(entry.path, pattern)
                        or fnmatch.fnmatchcase(
                            entry.path.rsplit("/", 1)[-1], pattern
                        )
                        for pattern in policy.generated_path_patterns
                    )
                    or _GENERATED_MARKERS_RE.search(entry.after_source or "")
                )
            ):
                add(
                    ProposalFindingCode.GENERATED_CHANGE_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "generated content markers require explicit policy authority",
                    entry.path,
                )
            if any(
                _path_matches(entry.path, config_path)
                for config_path in _VALIDATION_CONFIG_PATHS
            ):
                if not policy.allow_validation_config_changes:
                    add(
                        ProposalFindingCode.VALIDATION_WEAKENING_FORBIDDEN,
                        ProposalGate.CONTENT,
                        "validation configuration changes require explicit task authority",
                        entry.path,
                    )
                elif not _validation_config_change_is_additive(entry):
                    add(
                        ProposalFindingCode.VALIDATION_WEAKENING_FORBIDDEN,
                        ProposalGate.CONTENT,
                        "authorized validation configuration changes must be additive and non-weakening",
                        entry.path,
                    )
            if (
                _is_test_path(entry.path)
                and entry.change_kind is not DiffChangeKind.DELETE
                and not policy.allow_test_weakening
                and entry.before_source is not None
                and entry.after_source is not None
                and (
                    _TEST_SKIP_RE.search(entry.after_source)
                    and not _TEST_SKIP_RE.search(entry.before_source)
                    or len(_ASSERTION_RE.findall(entry.after_source))
                    < len(_ASSERTION_RE.findall(entry.before_source))
                    or not _python_test_names(entry.before_source).issubset(
                        _python_test_names(entry.after_source)
                    )
                )
            ):
                add(
                    ProposalFindingCode.TEST_WEAKENING_FORBIDDEN,
                    ProposalGate.CONTENT,
                    "test assertions or execution were weakened",
                    entry.path,
                )
            if not policy.allow_test_deletion and (
                (
                    entry.change_kind is DiffChangeKind.DELETE
                    and _is_test_path(entry.path)
                )
                or (
                    entry.change_kind is DiffChangeKind.RENAME
                    and _is_test_path(entry.old_path)
                    and not _is_test_path(entry.new_path)
                )
            ):
                add(
                    ProposalFindingCode.TEST_DELETION_FORBIDDEN,
                    ProposalGate.PATH,
                    "test deletion requires explicit task authority",
                    entry.path,
                )
            if (
                policy.require_python_syntax
                and entry.is_python
                and entry.change_kind is not DiffChangeKind.DELETE
                and entry.after_source is not None
            ):
                try:
                    ast.parse(entry.after_source, filename=entry.path)
                except (SyntaxError, ValueError) as exc:
                    add(
                        ProposalFindingCode.PYTHON_SYNTAX_ERROR,
                        ProposalGate.AST_INTERFACE,
                        f"candidate Python does not parse: {exc.msg if isinstance(exc, SyntaxError) else exc}",
                        entry.path,
                    )

        for step in proposal.validation_plan:
            if not _command_is_allowed(
                step.command, policy.allowed_validation_commands
            ):
                add(
                    ProposalFindingCode.COMMAND_FORBIDDEN,
                    ProposalGate.VALIDATION,
                    "validation command is not an allowed argv prefix",
                )

        # LPR-017: intercept ordinary proposals as read-only candidate overlays
        # before mutation.  Default-off preserves legacy proposal flows.
        if policy.enable_live_logic_repair and not findings:
            try:
                from ..todo_daemon.live_logic_repair_controller import (
                    CandidateOverlayContractDeltaGate,
                    LiveLogicRepairPolicy,
                    OverlayGateDisposition,
                )

                base_sources: dict[str, str] = {}
                candidate_sources: dict[str, str] = {}
                write_set: list[str] = []
                for entry in proposal.effective_entries:
                    path = entry.path
                    write_set.append(path)
                    if entry.before_source is not None:
                        base_sources[path] = entry.before_source
                    if entry.after_source is not None:
                        candidate_sources[path] = entry.after_source
                overlay_policy = LiveLogicRepairPolicy(
                    enable_live_logic_repair=True,
                    expand_write_set_on_omission=(
                        policy.logic_repair_expand_write_set
                    ),
                    reject_omitted_callers=True,
                )
                gate = CandidateOverlayContractDeltaGate(overlay_policy)
                overlay_result = gate.evaluate(
                    proposal_id=proposal.proposal_id,
                    repository_id=proposal.repository_id,
                    base_tree_id=proposal.repository_tree_id
                    or proposal.baseline_id
                    or "tree:base",
                    candidate_tree_id=proposal.repository_tree_id
                    or "tree:candidate",
                    write_set=write_set,
                    base_sources=base_sources,
                    candidate_sources=candidate_sources,
                    resolved_callers=policy.logic_repair_resolved_callers,
                    unknown_frontier=policy.logic_repair_unknown_frontier,
                    compatibility_proofs=(
                        policy.logic_repair_compatibility_proofs
                    ),
                    no_change_proofs=policy.logic_repair_no_change_proofs,
                )
                if overlay_result.disposition in {
                    OverlayGateDisposition.REJECTED,
                    OverlayGateDisposition.ABSTAINED,
                    OverlayGateDisposition.DEFERRED,
                }:
                    code = ProposalFindingCode.LOGIC_REPAIR_OVERLAY_REJECTED
                    if "omitted_callers" in overlay_result.reason_codes:
                        code = ProposalFindingCode.OMITTED_CALLERS
                    elif (
                        "unknown_frontier_required"
                        in overlay_result.reason_codes
                    ):
                        code = ProposalFindingCode.UNKNOWN_FRONTIER_REQUIRED
                    elif (
                        "signature_arity_increase"
                        in overlay_result.reason_codes
                    ):
                        code = ProposalFindingCode.SIGNATURE_ARITY_INCREASE
                    add(
                        code,
                        ProposalGate.AST_INTERFACE,
                        overlay_result.detail
                        or "live logic-repair overlay rejected proposal",
                    )
                # EXPANDED is allowed only when the expanded write set remains
                # inside the existing proposal scope; otherwise reject.
                elif (
                    overlay_result.disposition
                    is OverlayGateDisposition.EXPANDED
                ):
                    expanded = set(overlay_result.expanded_write_set)
                    scope = set(proposal.changed_paths) | set(write_set)
                    if not expanded.issubset(scope):
                        add(
                            ProposalFindingCode.OMITTED_CALLERS,
                            ProposalGate.AST_INTERFACE,
                            (
                                "signature change requires caller paths "
                                "outside the proposal write set; reject or "
                                "re-admit an expanded atomic plan"
                            ),
                        )
            except Exception as exc:  # fail-closed
                add(
                    ProposalFindingCode.LOGIC_REPAIR_OVERLAY_REJECTED,
                    ProposalGate.AST_INTERFACE,
                    f"live logic-repair overlay gate failed: {exc}",
                )

        # Gate trace is complete even after a failure because all proposal
        # checks are bounded and cheap.  This yields better repair diagnostics
        # without admitting any expensive descendant.
        findings.sort(
            key=lambda item: (
                ORDERED_PROPOSAL_GATES.index(item.gate),
                item.path,
                item.code.value,
                item.message,
            )
        )
        receipt = ProposalValidationReceipt(
            proposal_id=proposal.proposal_id,
            policy_id=policy.policy_id,
            repository_tree_id=proposal.repository_tree_id,
            objective_id=proposal.objective_id,
            diff_digest=proposal.diff_digest,
            allowed_paths=policy.allowed_paths,
            changed_paths=proposal.changed_paths,
            accepted=not findings,
            findings=tuple(findings),
            gate_trace=ORDERED_PROPOSAL_GATES,
        )
        return ProposalValidationResult(proposal, policy, receipt)


class _RepositoryEnvelopeIssue(ProposalValidationError):
    def __init__(
        self,
        code: ProposalFindingCode,
        message: str,
        path: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


def _strict_sequence(
    value: Any,
    *,
    field_name: str,
    maximum: int,
) -> Sequence[Any]:
    if type(value) not in {list, tuple}:
        raise ProposalValidationError(f"{field_name} must be a sequence")
    if len(value) > maximum:
        raise ProposalValidationError(f"{field_name} exceeds its item-count bound")
    return value


def _strict_string_sequence(
    value: Any,
    *,
    field_name: str,
    maximum: int,
    canonical: bool = True,
) -> tuple[str, ...]:
    sequence = _strict_sequence(value, field_name=field_name, maximum=maximum)
    result = tuple(
        _strict_text(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(sequence)
    )
    if canonical and result != tuple(sorted(set(result))):
        raise ProposalValidationError(f"{field_name} must be sorted and unique")
    return result


def _validate_strict_provider_mapping(
    payload: dict[str, Any],
    policy: ProposalValidationPolicy,
) -> ImplementationProposal:
    """Validate exact v2 field types before any normalizing constructor runs."""

    identity_fields = {
        "schema",
        "proposal_version",
        "task_id",
        "accepted_plan_id",
        "repository_id",
        "repository_tree_id",
        "objective_id",
        "baseline_id",
        "context_id",
        "replay_nonce",
        "declared_paths",
        "operations",
        "rationale_references",
        "validation_plan",
        "risks",
        "authority_claims",
        "expected_effects",
        "patch_text",
        "candidate_diff",
    }
    derived_fields = {"changed_paths", "diff_digest", "proposal_id"}
    if set(payload) != identity_fields | derived_fields:
        raise ProposalValidationError(
            "provider proposal must contain exactly the versioned top-level fields"
        )
    if payload["schema"] != PROPOSAL_VALIDATION_REQUEST_SCHEMA:
        raise ProposalValidationError("provider proposal schema is unsupported")
    if payload["proposal_version"] != "2":
        raise ProposalValidationError("provider proposal must use version 2")
    for name in (
        "task_id",
        "accepted_plan_id",
        "repository_id",
        "repository_tree_id",
        "objective_id",
        "baseline_id",
        "context_id",
        "replay_nonce",
    ):
        _strict_id(payload[name], field_name=name)

    declared_paths = _strict_string_sequence(
        payload["declared_paths"],
        field_name="declared_paths",
        maximum=policy.max_diff_entries * 2,
    )
    for index, path in enumerate(declared_paths):
        _strict_repo_path(path, field_name=f"declared_paths[{index}]")
        if len(path.encode("utf-8")) > policy.max_path_bytes:
            raise ProposalValidationError("declared path exceeds the byte bound")
        if len(PurePosixPath(path).parts) > policy.max_path_depth:
            raise ProposalValidationError("declared path exceeds the depth bound")

    rationale_references = _strict_string_sequence(
        payload["rationale_references"],
        field_name="rationale_references",
        maximum=policy.max_operations * 4,
    )
    operations = _strict_sequence(
        payload["operations"],
        field_name="operations",
        maximum=policy.max_operations,
    )
    for index, operation in enumerate(operations):
        if type(operation) is not dict or set(operation) != {
            "operation",
            "path",
            "old_path",
            "rationale_refs",
        }:
            raise ProposalValidationError(
                f"operations[{index}] must contain exactly its versioned fields"
            )
        _strict_text(operation["operation"], field_name=f"operations[{index}].operation")
        _strict_repo_path(operation["path"], field_name=f"operations[{index}].path")
        if operation["old_path"]:
            _strict_repo_path(
                operation["old_path"], field_name=f"operations[{index}].old_path"
            )
        _strict_string_sequence(
            operation["rationale_refs"],
            field_name=f"operations[{index}].rationale_refs",
            maximum=policy.max_operations * 4,
        )

    validation_plan = _strict_sequence(
        payload["validation_plan"],
        field_name="validation_plan",
        maximum=policy.max_operations,
    )
    for index, step in enumerate(validation_plan):
        if type(step) is not dict or set(step) != {"command", "rationale_refs"}:
            raise ProposalValidationError(
                f"validation_plan[{index}] must contain exactly its versioned fields"
            )
        command = _strict_sequence(
            step["command"],
            field_name=f"validation_plan[{index}].command",
            maximum=64,
        )
        for part_index, part in enumerate(command):
            _strict_text(
                part,
                field_name=f"validation_plan[{index}].command[{part_index}]",
                max_bytes=4_096,
            )
        _strict_string_sequence(
            step["rationale_refs"],
            field_name=f"validation_plan[{index}].rationale_refs",
            maximum=policy.max_operations * 4,
        )

    risks = _strict_sequence(
        payload["risks"], field_name="risks", maximum=policy.max_operations
    )
    for index, risk in enumerate(risks):
        if type(risk) is not dict or set(risk) != {"risk", "mitigation"}:
            raise ProposalValidationError(
                f"risks[{index}] must contain exactly its versioned fields"
            )
        _strict_text(risk["risk"], field_name=f"risks[{index}].risk")
        _strict_text(risk["mitigation"], field_name=f"risks[{index}].mitigation")

    claims = payload["authority_claims"]
    if type(claims) is not dict:
        raise ProposalValidationError("authority_claims must be a plain object")
    identity_claims = {
        "task_id",
        "accepted_plan_id",
        "repository_id",
        "repository_tree_id",
        "objective_id",
        "baseline_id",
        "context_id",
    }
    non_authority_claims = {
        "proof_authoritative",
        "code_proof_authoritative",
        "completion_authoritative",
        "merge_eligible",
        "merge_authoritative",
        "freshness_authoritative",
        "authoritative",
        "completed",
        "proof_complete",
    }
    if set(claims) - identity_claims - non_authority_claims:
        raise ProposalValidationError("authority_claims contains unsupported fields")
    if not identity_claims.issubset(claims):
        raise ProposalValidationError("authority_claims omits canonical identity fields")
    for name in identity_claims:
        _strict_id(claims[name], field_name=f"authority_claims.{name}")
    for name in set(claims) & non_authority_claims:
        if type(claims[name]) is not bool:
            raise ProposalValidationError(
                f"authority_claims.{name} must be a boolean"
            )

    effects = _strict_sequence(
        payload["expected_effects"],
        field_name="expected_effects",
        maximum=policy.max_expected_effects,
    )
    if not effects:
        raise ProposalValidationError("expected_effects must not be empty")
    for effect in effects:
        ProposalExpectedEffect.from_mapping(effect)

    entries = _strict_sequence(
        payload["candidate_diff"],
        field_name="candidate_diff",
        maximum=policy.max_diff_entries,
    )
    if not entries:
        raise ProposalValidationError("candidate_diff must not be empty")
    entry_fields = {
        "old_path",
        "new_path",
        "change_kind",
        "before_blob_id",
        "after_blob_id",
        "binary",
        "generated",
        "metadata",
        "before_source",
        "after_source",
    }
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != entry_fields:
            raise ProposalValidationError(
                f"candidate_diff[{index}] must contain exactly its versioned fields"
            )
        for name in ("old_path", "new_path"):
            path = entry[name]
            if type(path) is not str:
                raise ProposalValidationError(
                    f"candidate_diff[{index}].{name} must be a string"
                )
            if path:
                _strict_repo_path(
                    path, field_name=f"candidate_diff[{index}].{name}"
                )
                if len(path.encode("utf-8")) > policy.max_path_bytes:
                    raise ProposalValidationError("candidate path exceeds the byte bound")
                if len(PurePosixPath(path).parts) > policy.max_path_depth:
                    raise ProposalValidationError("candidate path exceeds the depth bound")
        if entry["change_kind"] not in {
            item.value for item in DiffChangeKind
        } - {"unknown"}:
            raise ProposalValidationError("candidate change kind is not canonical")
        if type(entry["binary"]) is not bool:
            raise ProposalValidationError("candidate binary flag must be a boolean")
        if entry["generated"] is not None and type(entry["generated"]) is not bool:
            raise ProposalValidationError("candidate generated flag must be boolean or null")
        if type(entry["metadata"]) is not dict:
            raise ProposalValidationError("candidate metadata must be a plain object")
        metadata_fields = {
            "after_mode",
            "after_size_bytes",
            "before_mode",
            "before_size_bytes",
            "media_type",
            "size_bytes",
        }
        if set(entry["metadata"]) - metadata_fields:
            raise ProposalValidationError(
                "candidate metadata contains unsupported fields"
            )
        for name, value in entry["metadata"].items():
            if name.endswith("_bytes") or name == "size_bytes":
                if type(value) is not int or value < 0:
                    raise ProposalValidationError(
                        f"candidate metadata {name} must be a non-negative integer"
                    )
            elif type(value) is not str:
                raise ProposalValidationError(
                    f"candidate metadata {name} must be a string"
                )
        for source_name in ("before_source", "after_source"):
            source = entry[source_name]
            if source is not None:
                _strict_text(
                    source,
                    field_name=f"candidate_diff[{index}].{source_name}",
                    allow_empty=True,
                    max_bytes=policy.max_file_bytes,
                )
        for source_name, blob_name in (
            ("before_source", "before_blob_id"),
            ("after_source", "after_blob_id"),
        ):
            blob_id = entry[blob_name]
            if type(blob_id) is not str:
                raise ProposalValidationError(
                    f"candidate_diff[{index}].{blob_name} must be a string"
                )
            expected = _source_digest(entry[source_name])
            if blob_id != expected:
                raise ProposalValidationError(
                    f"candidate_diff[{index}].{blob_name} is detached from materialized content"
                )

    _strict_text(
        payload["patch_text"],
        field_name="patch_text",
        max_bytes=policy.max_patch_bytes,
    )
    proposal = ImplementationProposal.from_dict(payload)
    # These derived identities are mandatory in the hostile envelope.  The
    # constructor recomputes them, so aliases or stale candidate material fail.
    if payload["proposal_id"] != proposal.proposal_id:
        raise ProposalValidationError("proposal_id is detached from canonical content")
    if payload["diff_digest"] != proposal.diff_digest:
        raise ProposalValidationError("diff_digest is detached from candidate content")
    if tuple(payload["changed_paths"]) != proposal.changed_paths:
        raise ProposalValidationError("changed_paths is detached from candidate paths")
    return proposal


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _lstat_chain(root: Path, path: str) -> tuple[tuple[str, os.stat_result], ...]:
    result: list[tuple[str, os.stat_result]] = []
    current = root
    for index, part in enumerate(PurePosixPath(path).parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if index != len(PurePosixPath(path).parts) - 1:
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.REPOSITORY_CONTENT_MISMATCH,
                    "candidate parent path does not exist",
                    path,
                )
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise _RepositoryEnvelopeIssue(
                ProposalFindingCode.SYMLINK_BOUNDARY_FORBIDDEN,
                "repository path crosses a live symlink boundary",
                path,
            )
        if index < len(PurePosixPath(path).parts) - 1 and not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise _RepositoryEnvelopeIssue(
                ProposalFindingCode.REPOSITORY_CONTENT_MISMATCH,
                "candidate parent path is not a directory",
                path,
            )
        result.append(("/".join(PurePosixPath(path).parts[: index + 1]), metadata))
    return tuple(result)


def _read_repository_file(
    root: Path,
    path: str,
    *,
    maximum: int,
    allow_hardlinks: bool,
) -> tuple[bytes | None, tuple[tuple[str, os.stat_result], ...]]:
    before_chain = _lstat_chain(root, path)
    if not before_chain or before_chain[-1][0] != path:
        return None, before_chain
    target = before_chain[-1][1]
    if not stat.S_ISREG(target.st_mode):
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.REPOSITORY_CONTENT_MISMATCH,
            "candidate baseline path is not a regular file",
            path,
        )
    if target.st_nlink > 1 and not allow_hardlinks:
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.HARDLINK_BOUNDARY_FORBIDDEN,
            "candidate baseline path has multiple hard links",
            path,
        )
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | close_on_exec | no_follow | getattr(
        os, "O_DIRECTORY", 0
    )
    file_flags = os.O_RDONLY | close_on_exec | no_follow
    directory_descriptor = -1
    descriptor = -1
    try:
        root_before = os.lstat(root)
        directory_descriptor = os.open(root, directory_flags)
        if _stat_fingerprint(root_before) != _stat_fingerprint(
            os.fstat(directory_descriptor)
        ):
            raise OSError("repository root identity changed")
        parts = PurePosixPath(path).parts
        for index, part in enumerate(parts[:-1]):
            child_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            child_stat = os.fstat(child_descriptor)
            expected_name, expected_stat = before_chain[index]
            if (
                expected_name != "/".join(parts[: index + 1])
                or _stat_fingerprint(child_stat)
                != _stat_fingerprint(expected_stat)
            ):
                os.close(child_descriptor)
                raise OSError("repository ancestor identity changed")
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.REPOSITORY_PATH_RACE,
            "candidate baseline changed during no-follow open",
            path,
        ) from exc
    os.close(directory_descriptor)
    try:
        opened_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    value = b"".join(chunks)
    if len(value) > maximum:
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.LARGE_FILE_FORBIDDEN,
            "candidate baseline exceeds the file byte bound",
            path,
        )
    after_chain = _lstat_chain(root, path)
    if (
        _stat_fingerprint(root_before)
        != _stat_fingerprint(os.lstat(root))
        or
        _stat_fingerprint(target) != _stat_fingerprint(opened_before)
        or _stat_fingerprint(opened_before) != _stat_fingerprint(opened_after)
        or len(before_chain) != len(after_chain)
        or any(
            before_name != after_name
            or _stat_fingerprint(before_stat) != _stat_fingerprint(after_stat)
            for (before_name, before_stat), (after_name, after_stat) in zip(
                before_chain, after_chain
            )
        )
    ):
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.REPOSITORY_PATH_RACE,
            "candidate baseline identity changed while it was inspected",
            path,
        )
    return value, after_chain


def _validate_repository_envelope(
    proposal: ImplementationProposal,
    policy: ProposalValidationPolicy,
    repository_root: str | os.PathLike[str],
) -> str:
    root = Path(repository_root)
    if not root.is_absolute():
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.UNSAFE_PATH,
            "repository root must be absolute",
        )
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.REPOSITORY_CONTENT_MISMATCH,
            "repository root is unavailable",
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise _RepositoryEnvelopeIssue(
            ProposalFindingCode.SYMLINK_BOUNDARY_FORBIDDEN,
            "repository root must be a real directory",
        )

    snapshots: list[dict[str, Any]] = []
    for entry in proposal.candidate_diff:
        path = entry.old_path or entry.new_path
        for candidate_path in (entry.old_path, entry.new_path):
            if not candidate_path:
                continue
            if any(
                _path_matches(candidate_path, protected)
                for protected in policy.protected_paths
            ):
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.PROTECTED_PATH_FORBIDDEN,
                    "candidate path is operator-protected",
                    candidate_path,
                )
            # A nested .git marker identifies a submodule or an embedded
            # repository.  Its contents are untrusted and are never parsed.
            parent = PurePosixPath(candidate_path).parent
            accumulated = Path()
            for part in parent.parts:
                accumulated /= part
                marker = root / accumulated / ".git"
                try:
                    marker_stat = os.lstat(marker)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(marker_stat.st_mode) or stat.S_ISREG(
                    marker_stat.st_mode
                ) or stat.S_ISDIR(marker_stat.st_mode):
                    raise _RepositoryEnvelopeIssue(
                        ProposalFindingCode.SUBMODULE_BOUNDARY_FORBIDDEN,
                        "candidate path crosses a nested repository boundary",
                        candidate_path,
                    )

        baseline, chain = _read_repository_file(
            root,
            path,
            maximum=policy.max_file_bytes,
            allow_hardlinks=policy.allow_hardlinks,
        )
        expects_existing = entry.change_kind is not DiffChangeKind.ADD
        if expects_existing != (baseline is not None):
            raise _RepositoryEnvelopeIssue(
                ProposalFindingCode.REPOSITORY_CONTENT_MISMATCH,
                "candidate operation does not match baseline path existence",
                path,
            )
        if baseline is not None:
            if _looks_binary(baseline) and not policy.allow_binary:
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.BINARY_CHANGE_FORBIDDEN,
                    "binary baseline content requires explicit policy authority",
                    path,
                )
            if _looks_archive(path, baseline) and not policy.allow_archives:
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.ARCHIVE_CHANGE_FORBIDDEN,
                    "archive baseline content requires explicit policy authority",
                    path,
                )
            try:
                baseline_text = baseline.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.INVALID_ENCODING,
                    "candidate baseline is not canonical UTF-8 text",
                    path,
                ) from exc
            if baseline_text.startswith("\ufeff") or any(
                0xD800 <= ord(character) <= 0xDFFF for character in baseline_text
            ):
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.INVALID_ENCODING,
                    "candidate baseline is not canonical UTF-8 text",
                    path,
                )
            if entry.before_source != baseline_text:
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.BASELINE_CONTENT_MISMATCH,
                    "materialized before content does not match the repository baseline",
                    path,
                )
            if entry.before_blob_id != _sha256_bytes(baseline):
                raise _RepositoryEnvelopeIssue(
                    ProposalFindingCode.BASELINE_CONTENT_MISMATCH,
                    "before content identity does not match the repository baseline",
                    path,
                )
        snapshots.append(
            {
                "path": path,
                "exists": baseline is not None,
                "sha256": _sha256_bytes(baseline) if baseline is not None else "",
                "chain": [
                    (name, _stat_fingerprint(metadata)) for name, metadata in chain
                ],
            }
        )
    return _identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/repository-snapshot@2",
            "root": _stat_fingerprint(root_stat),
            "paths": snapshots,
        }
    )


@dataclass(frozen=True)
class UntrustedProposalAdmissionResult:
    """Bounded pre-dispatch outcome for hostile provider and repository data."""

    accepted: bool
    policy_id: str
    input_digest: str
    repository_snapshot_id: str = ""
    findings: tuple[ProposalValidationFinding, ...] = ()
    proposal_validation: ProposalValidationResult | None = None
    expensive_checks_started: int = 0

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise ProposalValidationError("untrusted admission accepted must be boolean")
        if self.expensive_checks_started != 0:
            raise ProposalValidationError(
                "untrusted admission cannot start expensive checks"
            )
        if self.accepted != (
            self.proposal_validation is not None
            and self.proposal_validation.accepted
            and not self.findings
            and bool(self.repository_snapshot_id)
        ):
            raise ProposalValidationError("untrusted admission verdict is inconsistent")

    @property
    def dispatch_allowed(self) -> bool:
        return self.accepted

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(sorted({finding.code.value for finding in self.findings}))

    @property
    def proposal(self) -> ImplementationProposal | None:
        return (
            self.proposal_validation.proposal
            if self.proposal_validation is not None
            else None
        )

    @property
    def admission_id(self) -> str:
        return _identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": UNTRUSTED_PROPOSAL_ADMISSION_SCHEMA,
            "accepted": self.accepted,
            "policy_id": self.policy_id,
            "input_digest": self.input_digest,
            "repository_snapshot_id": self.repository_snapshot_id,
            "findings": [finding.to_dict() for finding in self.findings],
            "proposal_validation_receipt_id": (
                self.proposal_validation.receipt.receipt_id
                if self.proposal_validation is not None
                else ""
            ),
            "expensive_checks_started": 0,
            "proof_authoritative": False,
            "completion_authoritative": False,
            "merge_eligible": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "admission_id": self.admission_id}


def validate_untrusted_implementation_proposal(
    provider_output: bytes | bytearray | str | Mapping[str, Any],
    *,
    policy: ProposalValidationPolicy | Mapping[str, Any],
    repository_root: str | os.PathLike[str],
) -> UntrustedProposalAdmissionResult:
    """Fail-fast admission for a hostile provider output and checkout.

    This function performs no writes, subprocess calls, imports, archive
    extraction, or validation dispatch.  A caller may dispatch expensive work
    only when ``dispatch_allowed`` is true.
    """

    if not isinstance(policy, ProposalValidationPolicy):
        policy = ProposalValidationPolicy.from_dict(policy)
    input_digest = ""

    def rejected(
        code: ProposalFindingCode,
        message: str,
        *,
        gate: ProposalGate = ProposalGate.SCHEMA,
        path: str = "",
        proposal_validation: ProposalValidationResult | None = None,
    ) -> UntrustedProposalAdmissionResult:
        finding = ProposalValidationFinding(code, gate, message, path)
        combined = (
            tuple(proposal_validation.findings) + (finding,)
            if proposal_validation is not None
            else (finding,)
        )
        combined = tuple(
            sorted(
                combined[: policy.max_findings],
                key=lambda item: (
                    ORDERED_PROPOSAL_GATES.index(item.gate),
                    item.path,
                    item.code.value,
                    item.message,
                ),
            )
        )
        return UntrustedProposalAdmissionResult(
            accepted=False,
            policy_id=policy.policy_id,
            input_digest=input_digest,
            findings=combined,
            proposal_validation=proposal_validation,
        )

    try:
        payload, input_digest = _decode_untrusted_output(
            provider_output,
            max_bytes=policy.max_output_bytes,
            max_depth=policy.max_output_depth,
            max_items=policy.max_output_items,
        )
    except _DuplicateJSONField as exc:
        return rejected(ProposalFindingCode.DUPLICATE_FIELD, str(exc))
    except ProposalValidationError as exc:
        message = str(exc)
        if "not canonical UTF-8" in message or "invalid encoding" in message:
            code = ProposalFindingCode.INVALID_ENCODING
        elif "depth bound" in message:
            code = ProposalFindingCode.OUTPUT_TOO_DEEP
        elif "byte bound" in message or "item-count bound" in message:
            code = ProposalFindingCode.OUTPUT_TOO_LARGE
        else:
            code = ProposalFindingCode.INVALID_SCHEMA
        return rejected(code, str(exc))

    try:
        proposal = _validate_strict_provider_mapping(payload, policy)
    except ProposalValidationError as exc:
        message = str(exc)
        if "identifier" in message:
            code = ProposalFindingCode.NON_CANONICAL_ID
        elif "identity" in message or "detached" in message:
            code = ProposalFindingCode.CANDIDATE_IDENTITY_MISMATCH
        elif "encoding" in message:
            code = ProposalFindingCode.INVALID_ENCODING
        elif "path" in message:
            code = ProposalFindingCode.UNSAFE_PATH
        elif "byte bound" in message or "item-count" in message:
            code = ProposalFindingCode.OUTPUT_TOO_LARGE
        else:
            code = ProposalFindingCode.INVALID_SCHEMA
        return rejected(code, message)

    validation = validate_implementation_proposal(proposal, policy=policy)
    if not validation.accepted:
        return UntrustedProposalAdmissionResult(
            accepted=False,
            policy_id=policy.policy_id,
            input_digest=input_digest,
            findings=validation.findings,
            proposal_validation=validation,
        )
    try:
        snapshot_id = _validate_repository_envelope(
            proposal, policy, repository_root
        )
    except _RepositoryEnvelopeIssue as exc:
        return rejected(
            exc.code,
            str(exc),
            gate=(
                ProposalGate.CONTENT
                if exc.code
                in {
                    ProposalFindingCode.BASELINE_CONTENT_MISMATCH,
                    ProposalFindingCode.BINARY_CHANGE_FORBIDDEN,
                    ProposalFindingCode.ARCHIVE_CHANGE_FORBIDDEN,
                    ProposalFindingCode.INVALID_ENCODING,
                }
                else ProposalGate.PATH
            ),
            path=exc.path,
            proposal_validation=validation,
        )
    return UntrustedProposalAdmissionResult(
        accepted=True,
        policy_id=policy.policy_id,
        input_digest=input_digest,
        repository_snapshot_id=snapshot_id,
        proposal_validation=validation,
    )


def validate_implementation_proposal(
    proposal: ImplementationProposal | Mapping[str, Any],
    *,
    policy: ProposalValidationPolicy | Mapping[str, Any],
) -> ProposalValidationResult:
    """Validate one proposal against a frozen strict policy."""

    if not isinstance(policy, ProposalValidationPolicy):
        policy = ProposalValidationPolicy.from_dict(policy)
    return ProposalValidator(policy).validate(proposal)


validate_proposal = validate_implementation_proposal
StrictProposalValidator = ProposalValidator


__all__ = [
    "ImplementationProposal",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_CONFIGURATION_REVISION",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_REVISION",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_PRODUCING_TASK_IDS",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID",
    "NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIRED_EXHAUSTIVE_RECEIPTS",
    "ORDERED_PROPOSAL_GATES",
    "ParsedPatchFile",
    "PROPOSAL_GATE_EVIDENCE_SCHEMA",
    "PROPOSAL_OWNED_GATE_GROUPS",
    "PROPOSAL_REJECTION_EVIDENCE_SCHEMA",
    "PROPOSAL_VALIDATION_POLICY_SCHEMA",
    "PROPOSAL_VALIDATION_RECEIPT_SCHEMA",
    "PROPOSAL_VALIDATION_REQUEST_SCHEMA",
    "ProposalFindingCode",
    "ProposalExpectedEffect",
    "ProposalGate",
    "ProposalOperation",
    "ProposalRejectionEvidence",
    "ProposalRisk",
    "ProposalValidationError",
    "ProposalValidationFinding",
    "ProposalValidationPolicy",
    "ProposalValidationReceipt",
    "ProposalValidationRequest",
    "ProposalValidationResult",
    "ProposalValidationStep",
    "ProposalValidator",
    "StrictProposalValidator",
    "UNTRUSTED_PROPOSAL_ADMISSION_SCHEMA",
    "UntrustedProposalAdmissionResult",
    "parse_unified_patch",
    "validate_implementation_proposal",
    "validate_untrusted_implementation_proposal",
    "validate_proposal",
]
