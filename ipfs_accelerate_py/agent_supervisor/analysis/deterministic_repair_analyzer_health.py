"""Fail-closed whole-scope analyzer health for deterministic contract repair.

DCR-012 binds every path in the six RepairRootOwnership HEAD trees to exactly
one disposition, compresses the ledger under the supervisor admission limit,
and refuses completion-safe claims when compiler, lifecycle, or active-source
parse evidence is incomplete.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..integrations.contract_repair_dependencies import PINNED_TYPESCRIPT_VERSION
from ..validation.typescript_validation_image import (
    TYPESCRIPT_COMPILER_JS,
    TYPESCRIPT_COMPILER_SHA256,
    TYPESCRIPT_NODE_LAUNCHER,
    TYPESCRIPT_NODE_VERSION,
    TYPESCRIPT_PACKAGE_JSON,
    TYPESCRIPT_PACKAGE_SHA256,
    TYPESCRIPT_VALIDATION_IMAGE,
    TYPESCRIPT_VERSION,
    typescript_validation_toolchain_contract,
)
from .analyzer_health import AnalyzerHealthStatus
from .deterministic_repair_forest import (
    DCR_ROOT_IDS,
    DCR_TODO_PATH,
    REPAIR_FOREST_SCHEMA,
    _document_integrity,
    _git_is_ancestor,
    _git_oid,
    _OID_PATTERN,
)
from .polyglot_ast_provider import (
    PolyglotASTLimits,
    PolyglotASTProvider,
    PolyglotASTProviderError,
)

ANALYZER_HEALTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-analyzer-health@1"
)
ANALYZER_HEALTH_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-analyzer-health-validation@1"
)
ANALYZER_HEALTH_LIFECYCLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-analyzer-health-lifecycle@1"
)
DISPOSITION_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-disposition-ledger@1"
)
DISPOSITION_CODEC: Final[str] = "dcr-disposition-dictionary-prefix@1"
REPOSITORY_INDEX_INTERFACE: Final[str] = "RepositoryIndex@1"
ANALYZER_HEALTH_INTERFACE: Final[str] = "AnalyzerHealth@1"

DCR_TASK_ID: Final[str] = "DCR-012"
DCR_ARTIFACT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/analyzer-health.json"
)
DCR_FOREST_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/forest.json"
)
DCR_CARRIER_SUBJECT: Final[str] = (
    "DCR-012: Restore analyzer health and exact parser accounting"
)
DCR_TODO_SUBJECT: Final[str] = "DCR-012: mark todo completed"
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_SOURCE_BYTES: Final[int] = 32 * 1024 * 1024
AUTHORITY_TIMEOUT_SECONDS: Final[int] = 900
_GIT_TIMEOUT_SECONDS: Final[int] = 60
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

_SEMANTIC_SUFFIXES: Final[Mapping[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}
_STRUCTURED_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".jsonc"})
_BINARY_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".wasm",
        ".so",
        ".dylib",
        ".dll",
        ".o",
        ".a",
        ".class",
        ".jar",
        ".pyc",
        ".pyo",
        ".bin",
        ".dat",
        ".sqlite",
        ".db",
        ".parquet",
        ".npy",
        ".npz",
        ".onnx",
        ".pt",
        ".pth",
        ".pkl",
        ".pickle",
        ".lock",
    }
)
_REVIEWED_UNSUPPORTED_REASONS: Final[frozenset[str]] = frozenset(
    {
        "binary_or_generated",
        "dependency_tool_identity",
        "text_reference",
        "symlink_entry",
        "gitlink_entry",
        "oversized_source",
        "empty_blob",
        "non_utf8_source",
        "jsonc_structured",
        "language_unsupported",
        "mode_unsupported",
    }
)
_ACTIVE_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    set(_SEMANTIC_SUFFIXES) | set(_STRUCTURED_SUFFIXES)
)


class DeterministicRepairAnalyzerHealthError(ValueError):
    """A required analyzer-health invariant could not be proven."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(
            reason_code or "deterministic_repair_analyzer_health_error"
        )
        super().__init__(message or self.reason_code)


class DispositionKind(str, Enum):
    SEMANTIC_AST = "semantic_ast"
    STRUCTURED_DATA = "structured_data"
    TEXT_REFERENCE = "text_reference"
    BINARY_OR_GENERATED = "binary_or_generated"
    UNSUPPORTED = "unsupported"
    PARSE_FAILURE = "parse_failure"
    EXCLUDED = "excluded"


class ParserStatus(str, Enum):
    INDEXED = "indexed"
    PARSE_FAILURE = "parse_failure"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeterministicRepairAnalyzerHealthError(
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
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeterministicRepairAnalyzerHealthError(
            "noncanonical_analyzer_health_value"
        ) from exc


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
        raise DeterministicRepairAnalyzerHealthError(
            "noncanonical_analyzer_health_value"
        ) from exc


def _read_json_bytes(value: bytes, *, reason: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except DeterministicRepairAnalyzerHealthError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeterministicRepairAnalyzerHealthError(reason) from exc
    if not isinstance(payload, Mapping):
        raise DeterministicRepairAnalyzerHealthError(reason)
    return payload


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


def _run_git(
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
        raise DeterministicRepairAnalyzerHealthError(reason, str(root)) from exc
    if result.returncode:
        raise DeterministicRepairAnalyzerHealthError(reason, str(root))
    if binary:
        return result.stdout
    return os.fsdecode(result.stdout).rstrip("\r\n")


def _default_workspace() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config/deterministic_contract_repair_roots.json").is_file():
            return candidate
    raise DeterministicRepairAnalyzerHealthError("workspace_missing")


@dataclass(frozen=True)
class PathDisposition:
    root_id: str
    path: str
    kind: DispositionKind
    reason_code: str
    content_digest: str
    language: str
    parser_status: ParserStatus
    mode: str
    blob_oid: str

    def to_row(self) -> tuple[str, ...]:
        return (
            self.root_id,
            self.path,
            self.kind.value,
            self.reason_code,
            self.content_digest,
            self.language,
            self.parser_status.value,
            self.mode,
            self.blob_oid,
        )


def _classify_path(path: str, mode: str) -> tuple[DispositionKind, str, str, ParserStatus]:
    if mode == "120000":
        return (
            DispositionKind.UNSUPPORTED,
            "symlink_entry",
            "",
            ParserStatus.NOT_APPLICABLE,
        )
    if mode == "160000":
        return (
            DispositionKind.UNSUPPORTED,
            "gitlink_entry",
            "",
            ParserStatus.NOT_APPLICABLE,
        )
    if mode not in {"100644", "100755"}:
        return (
            DispositionKind.UNSUPPORTED,
            "mode_unsupported",
            "",
            ParserStatus.UNSUPPORTED,
        )
    suffix = Path(path).suffix.casefold()
    name = Path(path).name.casefold()
    if suffix in _BINARY_SUFFIXES or name in {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.lock",
        "poetry.lock",
        "composer.lock",
        "go.sum",
    }:
        return (
            DispositionKind.BINARY_OR_GENERATED,
            "binary_or_generated"
            if suffix in _BINARY_SUFFIXES
            else "dependency_tool_identity",
            "",
            ParserStatus.NOT_APPLICABLE,
        )
    if suffix in _SEMANTIC_SUFFIXES:
        return (
            DispositionKind.SEMANTIC_AST,
            "semantic_ast",
            _SEMANTIC_SUFFIXES[suffix],
            ParserStatus.INDEXED,
        )
    if suffix == ".jsonc":
        return (
            DispositionKind.STRUCTURED_DATA,
            "jsonc_structured",
            "jsonc",
            ParserStatus.INDEXED,
        )
    if suffix == ".json":
        return (
            DispositionKind.STRUCTURED_DATA,
            "structured_data",
            "json",
            ParserStatus.INDEXED,
        )
    if suffix in {
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".csv",
        ".tsv",
        ".html",
        ".css",
        ".scss",
        ".less",
        ".svg",
        ".xml",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
        ".sql",
        ".graphql",
        ".gql",
        ".proto",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".swift",
        ".rb",
        ".php",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".r",
        ".jl",
        ".lua",
        ".vim",
        ".el",
        ".lisp",
        ".clj",
        ".ex",
        ".exs",
        ".erl",
        ".hs",
        ".ml",
        ".mli",
        ".fs",
        ".fsx",
        ".dart",
        ".scala",
        ".sbt",
        ".gradle",
        ".makefile",
        ".cmake",
        ".dockerfile",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
        ".env",
        ".example",
        ".sample",
        ".template",
        ".in",
        ".am",
        ".ac",
        ".m4",
        ".patch",
        ".diff",
        ".map",
        ".lockb",
    } or name in {
        "makefile",
        "dockerfile",
        "license",
        "copying",
        "authors",
        "changelog",
        "readme",
        "gemfile",
        "rakefile",
        "procfile",
        "vagrantfile",
    }:
        return (
            DispositionKind.TEXT_REFERENCE,
            "text_reference",
            "",
            ParserStatus.NOT_APPLICABLE,
        )
    return (
        DispositionKind.UNSUPPORTED,
        "language_unsupported",
        "",
        ParserStatus.UNSUPPORTED,
    )


def _strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments outside string literals."""

    result: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    string_quote = ""
    escaped = False
    while index < length:
        char = text[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            string_quote = char
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length:
            nxt = text[index + 1]
            if nxt == "/":
                index += 2
                while index < length and text[index] not in "\r\n":
                    index += 1
                continue
            if nxt == "*":
                index += 2
                while index + 1 < length and not (
                    text[index] == "*" and text[index + 1] == "/"
                ):
                    index += 1
                index = min(length, index + 2)
                continue
        result.append(char)
        index += 1
    return "".join(result)


def _enumerate_root_paths(
    root: Path, root_id: str, tree_oid: str
) -> list[dict[str, str]]:
    raw = _run_git(root, "ls-tree", "-r", "-z", tree_oid)
    assert isinstance(raw, bytes)
    rows: list[dict[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, separator, path_bytes = entry.partition(b"\t")
        if not separator:
            raise DeterministicRepairAnalyzerHealthError("invalid_tree_entry", root_id)
        fields = metadata.split()
        if len(fields) != 3:
            raise DeterministicRepairAnalyzerHealthError("invalid_tree_entry", root_id)
        mode = fields[0].decode("ascii", "strict")
        object_type = fields[1].decode("ascii", "strict")
        oid = fields[2].decode("ascii", "strict").lower()
        if object_type not in {"blob", "commit"} or not _OID_PATTERN.fullmatch(oid):
            raise DeterministicRepairAnalyzerHealthError("invalid_tree_entry", root_id)
        path = path_bytes.decode("utf-8", "surrogateescape")
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise DeterministicRepairAnalyzerHealthError("unsafe_tree_path", path)
        rows.append(
            {
                "root_id": root_id,
                "path": path,
                "mode": mode,
                "blob_oid": oid,
            }
        )
    rows.sort(key=lambda item: (item["root_id"], item["path"]))
    return rows


def _blob_bytes(root: Path, oid: str, *, max_bytes: int) -> bytes:
    try:
        result = subprocess.run(
            ("git", "cat-file", "blob", oid),
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeterministicRepairAnalyzerHealthError("blob_unreadable", oid) from exc
    if result.returncode:
        raise DeterministicRepairAnalyzerHealthError("blob_unreadable", oid)
    payload = result.stdout
    if len(payload) > max_bytes:
        raise DeterministicRepairAnalyzerHealthError("oversized_source", oid)
    return payload


def _content_digest_for_blob(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _encode_disposition_ledger(
    rows: Sequence[PathDisposition],
) -> tuple[dict[str, Any], str]:
    ordered = sorted(rows, key=lambda item: (item.root_id, item.path))
    uncompressed_rows = [list(item.to_row()) for item in ordered]
    uncompressed_digest = _content_id(uncompressed_rows)
    dictionaries = {
        "root_id": sorted({item.root_id for item in ordered}),
        "kind": sorted({item.kind.value for item in ordered}),
        "reason_code": sorted({item.reason_code for item in ordered}),
        "language": sorted({item.language for item in ordered}),
        "parser_status": sorted({item.parser_status.value for item in ordered}),
        "mode": sorted({item.mode for item in ordered}),
    }
    root_index = {value: index for index, value in enumerate(dictionaries["root_id"])}
    kind_index = {value: index for index, value in enumerate(dictionaries["kind"])}
    reason_index = {
        value: index for index, value in enumerate(dictionaries["reason_code"])
    }
    language_index = {
        value: index for index, value in enumerate(dictionaries["language"])
    }
    status_index = {
        value: index for index, value in enumerate(dictionaries["parser_status"])
    }
    mode_index = {value: index for index, value in enumerate(dictionaries["mode"])}

    packed = bytearray()
    previous_path = ""
    for item in ordered:
        path = item.path
        shared = 0
        limit = min(len(previous_path), len(path))
        while shared < limit and previous_path[shared] == path[shared]:
            shared += 1
        suffix = path[shared:]
        packed.extend(struct.pack(">H", shared))
        suffix_bytes = suffix.encode("utf-8", "surrogateescape")
        packed.extend(struct.pack(">H", len(suffix_bytes)))
        packed.extend(suffix_bytes)
        packed.extend(
            struct.pack(
                ">BBBBBB",
                root_index[item.root_id],
                kind_index[item.kind.value],
                reason_index[item.reason_code],
                language_index[item.language],
                status_index[item.parser_status.value],
                mode_index[item.mode],
            )
        )
        digest = item.content_digest
        if digest.startswith("sha256:") and len(digest) == 71:
            packed.append(1)
            packed.extend(bytes.fromhex(digest[7:]))
        else:
            packed.append(0)
        oid = item.blob_oid
        if _OID_PATTERN.fullmatch(oid) and len(oid) == 40:
            packed.append(1)
            packed.extend(bytes.fromhex(oid))
        else:
            packed.append(0)
        previous_path = path

    compressed = zlib.compress(bytes(packed), level=9)
    ledger = {
        "schema": DISPOSITION_LEDGER_SCHEMA,
        "codec": DISPOSITION_CODEC,
        "row_count": len(ordered),
        "uncompressed_digest": uncompressed_digest,
        "dictionary": dictionaries,
        "payload_encoding": "zlib+base64",
        "payload": base64.b64encode(compressed).decode("ascii"),
        "payload_sha256": _sha256(compressed),
    }
    return ledger, uncompressed_digest


def decode_disposition_ledger(
    ledger: Mapping[str, Any],
) -> tuple[tuple[PathDisposition, ...], str]:
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("schema") != DISPOSITION_LEDGER_SCHEMA
        or ledger.get("codec") != DISPOSITION_CODEC
        or ledger.get("payload_encoding") != "zlib+base64"
    ):
        raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
    try:
        row_count = int(ledger["row_count"])
        dictionary = ledger["dictionary"]
        payload_b64 = str(ledger["payload"])
        claimed_digest = str(ledger["uncompressed_digest"])
        claimed_payload_digest = str(ledger["payload_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeterministicRepairAnalyzerHealthError(
            "invalid_disposition_ledger"
        ) from exc
    if not isinstance(dictionary, Mapping):
        raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
    try:
        compressed = base64.b64decode(payload_b64.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise DeterministicRepairAnalyzerHealthError(
            "invalid_disposition_ledger"
        ) from exc
    if _sha256(compressed) != claimed_payload_digest:
        raise DeterministicRepairAnalyzerHealthError("disposition_payload_digest_mismatch")
    try:
        packed = zlib.decompress(compressed)
    except zlib.error as exc:
        raise DeterministicRepairAnalyzerHealthError(
            "invalid_disposition_ledger"
        ) from exc

    def table(name: str) -> list[str]:
        values = dictionary.get(name)
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
        return list(values)

    roots = table("root_id")
    kinds = table("kind")
    reasons = table("reason_code")
    languages = table("language")
    statuses = table("parser_status")
    modes = table("mode")
    offset = 0
    previous_path = ""
    rows: list[PathDisposition] = []

    def take(count: int) -> bytes:
        nonlocal offset
        if offset + count > len(packed):
            raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
        chunk = packed[offset : offset + count]
        offset += count
        return chunk

    for _ in range(row_count):
        shared = struct.unpack(">H", take(2))[0]
        suffix_len = struct.unpack(">H", take(2))[0]
        suffix = take(suffix_len).decode("utf-8", "surrogateescape")
        if shared > len(previous_path):
            raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
        path = previous_path[:shared] + suffix
        indexes = take(6)
        root_i, kind_i, reason_i, language_i, status_i, mode_i = indexes
        try:
            root_id = roots[root_i]
            kind = DispositionKind(kinds[kind_i])
            reason = reasons[reason_i]
            language = languages[language_i]
            status = ParserStatus(statuses[status_i])
            mode = modes[mode_i]
        except (IndexError, ValueError) as exc:
            raise DeterministicRepairAnalyzerHealthError(
                "invalid_disposition_ledger"
            ) from exc
        has_digest = take(1)[0]
        if has_digest == 1:
            content_digest = "sha256:" + take(32).hex()
        elif has_digest == 0:
            content_digest = ""
        else:
            raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
        has_oid = take(1)[0]
        if has_oid == 1:
            blob_oid = take(20).hex()
        elif has_oid == 0:
            blob_oid = ""
        else:
            raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
        rows.append(
            PathDisposition(
                root_id=root_id,
                path=path,
                kind=kind,
                reason_code=reason,
                content_digest=content_digest,
                language=language,
                parser_status=status,
                mode=mode,
                blob_oid=blob_oid,
            )
        )
        previous_path = path
    if offset != len(packed):
        raise DeterministicRepairAnalyzerHealthError("invalid_disposition_ledger")
    recomputed = [list(item.to_row()) for item in rows]
    recomputed_digest = _content_id(recomputed)
    if recomputed_digest != claimed_digest:
        raise DeterministicRepairAnalyzerHealthError(
            "disposition_uncompressed_digest_mismatch"
        )
    return tuple(rows), recomputed_digest


def _root_merkle(rows: Sequence[PathDisposition]) -> dict[str, str]:
    by_root: dict[str, list[tuple[str, ...]]] = {root_id: [] for root_id in DCR_ROOT_IDS}
    for item in rows:
        by_root.setdefault(item.root_id, []).append(item.to_row())
    return {
        root_id: _content_id(sorted(values))
        for root_id, values in sorted(by_root.items())
    }


def _prove_historical_forest(
    workspace: Path, forest_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove DCR-011 forest integrity without requiring live currentness."""

    manifest, reasons = _document_integrity(forest_payload)
    if manifest is None:
        raise DeterministicRepairAnalyzerHealthError(
            "forest_integrity_invalid", ",".join(reasons)
        )
    if forest_payload.get("schema") != REPAIR_FOREST_SCHEMA:
        raise DeterministicRepairAnalyzerHealthError("forest_schema_invalid")
    portable = manifest.portable
    lifecycle = portable.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != "DCR-011":
        raise DeterministicRepairAnalyzerHealthError("forest_lifecycle_invalid")
    subject = str(lifecycle.get("subject_head") or "")
    if not _OID_PATTERN.fullmatch(subject):
        raise DeterministicRepairAnalyzerHealthError("forest_lifecycle_invalid")
    current = _git_oid(workspace, "rev-parse", "HEAD")
    if current != subject and not _git_is_ancestor(workspace, subject, current):
        raise DeterministicRepairAnalyzerHealthError(
            "forest_subject_not_ancestor", subject
        )
    # Historical completion is the subject plus the exact three-commit
    # DCR-011 carrier/integration/todo chain when present; otherwise the
    # capture subject itself remains a valid historical binding point.
    completion_commit = subject
    raw = _run_git(
        workspace,
        "rev-list",
        "--ancestry-path",
        "--reverse",
        "--topo-order",
        f"{subject}..{current}",
        binary=False,
    )
    commits = tuple(item for item in str(raw).splitlines() if item)
    if commits:
        # Prefer the earliest todo-completed DCR-011 tip when the chain is exact.
        try:
            from .deterministic_repair_forest import _lifecycle_state

            for index in range(1, min(len(commits), 3) + 1):
                tip = commits[index - 1]
                state, state_reasons = _lifecycle_state(
                    workspace,
                    forest_payload,
                    subject=subject,
                    current=tip,
                )
                if not state_reasons and state == "todo_completed":
                    completion_commit = tip
                    break
        except Exception:
            completion_commit = subject
    return {
        "forest_id": manifest.forest_id,
        "subject_head": subject,
        "subject_tree": str(lifecycle.get("subject_tree") or ""),
        "completion_commit": completion_commit,
        "integrity_valid": True,
        "required_root_ids": list(DCR_ROOT_IDS),
        "root_heads": {
            str(item.get("id")): str(item.get("head"))
            for item in portable.get("roots", ())
            if isinstance(item, Mapping)
        },
        "root_trees": {
            str(item.get("id")): str(item.get("tree"))
            for item in portable.get("roots", ())
            if isinstance(item, Mapping)
        },
    }


def _discover_host_typescript() -> tuple[str, str]:
    """Best-effort host TypeScript paths when the sealed image tree is absent."""

    candidates: list[Path] = []
    env_js = os.environ.get("IPFS_ACCELERATE_TYPESCRIPT_JS", "").strip()
    if env_js:
        candidates.append(Path(env_js))
    candidates.append(Path(TYPESCRIPT_COMPILER_JS))
    node_path_env = os.environ.get("NODE_PATH", "")
    for entry in node_path_env.split(os.pathsep):
        if not entry:
            continue
        candidates.append(Path(entry) / "typescript" / "lib" / "typescript.js")
        candidates.append(Path(entry) / "typescript.js")
    # Common local installs relative to the accelerate package.
    package_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        (
            package_root / "node_modules" / "typescript" / "lib" / "typescript.js",
            package_root
            / "ipfs_accelerate_js"
            / "node_modules"
            / "typescript"
            / "lib"
            / "typescript.js",
        )
    )
    for compiler in candidates:
        try:
            if compiler.is_file():
                package = compiler.parent.parent / "package.json"
                return str(compiler), str(package if package.is_file() else "")
        except OSError:
            continue
    return TYPESCRIPT_COMPILER_JS, TYPESCRIPT_PACKAGE_JSON


def _resolve_typescript_identity() -> dict[str, Any]:
    env_js = os.environ.get("IPFS_ACCELERATE_TYPESCRIPT_JS", "").strip()
    env_pkg = os.environ.get("IPFS_ACCELERATE_TYPESCRIPT_PACKAGE_JSON", "").strip()
    env_version = os.environ.get("IPFS_ACCELERATE_TYPESCRIPT_VERSION", "").strip()
    toolchain = typescript_validation_toolchain_contract()
    discovered_js, discovered_pkg = _discover_host_typescript()
    compiler_path = env_js or discovered_js
    package_path = env_pkg or discovered_pkg or TYPESCRIPT_PACKAGE_JSON
    version = env_version or TYPESCRIPT_VERSION
    if not env_version and package_path:
        try:
            package_payload = json.loads(Path(package_path).read_text(encoding="utf-8"))
            if isinstance(package_payload, Mapping) and package_payload.get("version"):
                version = str(package_payload["version"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    node_path = os.environ.get("IPFS_ACCELERATE_NODE_BINARY", "").strip() or "node"
    identity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/dcr-typescript-toolchain@1",
        "expected_version": PINNED_TYPESCRIPT_VERSION,
        "configured_version": version,
        "compiler_path": compiler_path,
        "package_path": package_path,
        "node_executable": node_path,
        "image_id": TYPESCRIPT_VALIDATION_IMAGE,
        "sealed_compiler_sha256": TYPESCRIPT_COMPILER_SHA256,
        "sealed_package_sha256": TYPESCRIPT_PACKAGE_SHA256,
        "sealed_node_version": TYPESCRIPT_NODE_VERSION,
        "sealed_node_launcher": TYPESCRIPT_NODE_LAUNCHER,
        "toolchain_contract": toolchain,
        "available": False,
        "digest_matched": False,
        "canary_passed": False,
        "reason_codes": [],
    }
    reasons: list[str] = []
    compiler = Path(compiler_path)
    package = Path(package_path)
    if version != PINNED_TYPESCRIPT_VERSION:
        reasons.append("compiler_version_mismatch")
    if not compiler.is_file():
        reasons.append("compiler_unavailable")
    if not package.is_file():
        reasons.append("compiler_package_unavailable")
    compiler_digest = ""
    package_digest = ""
    if compiler.is_file():
        compiler_digest = _sha256(compiler.read_bytes())
        if compiler_digest != f"sha256:{TYPESCRIPT_COMPILER_SHA256}" and compiler_path in {
            TYPESCRIPT_COMPILER_JS
        }:
            # Host/dev trees may not match the sealed image digest; sealed
            # validation still requires exact image bindings.
            if compiler_path.startswith("/opt/ipfs-validation-toolchains/"):
                reasons.append("compiler_digest_mismatch")
        identity["compiler_sha256"] = compiler_digest
    if package.is_file():
        package_digest = _sha256(package.read_bytes())
        identity["package_sha256"] = package_digest
        if package_path.startswith("/opt/ipfs-validation-toolchains/") and (
            package_digest != f"sha256:{TYPESCRIPT_PACKAGE_SHA256}"
        ):
            reasons.append("package_digest_mismatch")
    canary_passed = False
    if not reasons:
        try:
            provider = PolyglotASTProvider(
                PolyglotASTLimits(
                    max_files=1,
                    max_file_bytes=4096,
                    max_total_bytes=4096,
                    max_output_bytes=64 * 1024,
                    process_timeout_seconds=15.0,
                    node_memory_mib=256,
                ),
                node_executable=node_path,
                typescript_path=str(compiler),
                expected_typescript_version=PINNED_TYPESCRIPT_VERSION,
                persistent_typescript_worker=True,
            )
            try:
                extraction = provider.extract_with_metadata(
                    "export const dcrCanary = 1;\n",
                    "typescript",
                )
                canary_passed = (
                    extraction.compiler_version == PINNED_TYPESCRIPT_VERSION
                    and not extraction.record.parse_error
                )
                identity["observed_compiler_version"] = extraction.compiler_version
                if not canary_passed:
                    reasons.append("compiler_canary_failed")
            finally:
                provider.close_typescript_worker()
        except PolyglotASTProviderError as exc:
            reasons.append(exc.reason_code or "compiler_canary_failed")
        except OSError:
            reasons.append("compiler_canary_failed")
    identity["available"] = "compiler_unavailable" not in reasons
    identity["digest_matched"] = not any(
        item.endswith("digest_mismatch") for item in reasons
    )
    identity["canary_passed"] = canary_passed
    identity["reason_codes"] = reasons
    return identity


def _parse_source(
    provider: PolyglotASTProvider | None,
    *,
    language: str,
    payload: bytes,
    content_digest: str,
) -> tuple[ParserStatus, str, DispositionKind]:
    if language == "jsonc":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return (
                ParserStatus.UNSUPPORTED,
                "non_utf8_source",
                DispositionKind.UNSUPPORTED,
            )
        try:
            json.loads(_strip_jsonc(text))
        except json.JSONDecodeError as exc:
            return (
                ParserStatus.PARSE_FAILURE,
                f"jsonc_decode_error:{exc.msg}",
                DispositionKind.PARSE_FAILURE,
            )
        return ParserStatus.INDEXED, "jsonc_structured", DispositionKind.STRUCTURED_DATA
    if language == "json":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return (
                ParserStatus.UNSUPPORTED,
                "non_utf8_source",
                DispositionKind.UNSUPPORTED,
            )
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return (
                ParserStatus.PARSE_FAILURE,
                f"json_decode_error:{exc.msg}",
                DispositionKind.PARSE_FAILURE,
            )
        return ParserStatus.INDEXED, "structured_data", DispositionKind.STRUCTURED_DATA
    if language == "python":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return (
                ParserStatus.UNSUPPORTED,
                "non_utf8_source",
                DispositionKind.UNSUPPORTED,
            )
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return (
                ParserStatus.PARSE_FAILURE,
                f"python_syntax_error:{exc.msg}",
                DispositionKind.PARSE_FAILURE,
            )
        return ParserStatus.INDEXED, "semantic_ast", DispositionKind.SEMANTIC_AST
    if provider is None:
        return (
            ParserStatus.PARSE_FAILURE,
            "compiler_unavailable",
            DispositionKind.PARSE_FAILURE,
        )
    try:
        record = provider.extract(
            payload,
            language,
            source_sha256=content_digest,
        )
    except PolyglotASTProviderError as exc:
        return (
            ParserStatus.PARSE_FAILURE,
            exc.reason_code,
            DispositionKind.PARSE_FAILURE,
        )
    if record.parse_error:
        return (
            ParserStatus.PARSE_FAILURE,
            "parser_reported_failure",
            DispositionKind.PARSE_FAILURE,
        )
    return ParserStatus.INDEXED, "semantic_ast", DispositionKind.SEMANTIC_AST


def materialize_analyzer_health(
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    parse_sources: bool = True,
) -> dict[str, Any]:
    """Enumerate all six HEAD trees and emit a compact analyzer-health receipt."""

    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    forest_file = (
        Path(forest_path)
        if forest_path is not None
        else workspace.joinpath(*PurePosixPath(DCR_FOREST_PATH).parts)
    )
    forest_payload = _read_json_bytes(
        forest_file.read_bytes(), reason="forest_unreadable"
    )
    forest_proof = _prove_historical_forest(workspace, forest_payload)
    portable_roots = {
        str(item["id"]): item
        for item in forest_payload["portable"]["roots"]
        if isinstance(item, Mapping)
    }
    if set(portable_roots) != set(DCR_ROOT_IDS):
        raise DeterministicRepairAnalyzerHealthError("required_root_set_changed")

    toolchain = _resolve_typescript_identity()
    provider: PolyglotASTProvider | None = None
    if parse_sources and toolchain.get("canary_passed"):
        provider = PolyglotASTProvider(
            PolyglotASTLimits(
                max_files=10_000,
                max_file_bytes=max_source_bytes,
                max_total_bytes=max(max_source_bytes, 64 * 1024 * 1024),
                max_output_bytes=min(max_source_bytes, 8 * 1024 * 1024),
                process_timeout_seconds=30.0,
                node_memory_mib=512,
            ),
            node_executable=str(toolchain.get("node_executable") or "node"),
            typescript_path=str(toolchain.get("compiler_path") or ""),
            expected_typescript_version=PINNED_TYPESCRIPT_VERSION,
            persistent_typescript_worker=True,
        )

    dispositions: list[PathDisposition] = []
    funnel = Counter()
    try:
        for root_id in DCR_ROOT_IDS:
            root_meta = portable_roots[root_id]
            relative = str(root_meta.get("relative_path") or ".")
            root_path = workspace if relative in {"", "."} else workspace / relative
            if not root_path.is_dir():
                raise DeterministicRepairAnalyzerHealthError(
                    "root_missing", root_id
                )
            tree_oid = str(root_meta.get("tree") or "")
            if not _OID_PATTERN.fullmatch(tree_oid):
                raise DeterministicRepairAnalyzerHealthError(
                    "invalid_root_tree", root_id
                )
            # Prefer the forest-bound tree; fall back only if the object is
            # absent (should not happen for a historically valid forest).
            try:
                entries = _enumerate_root_paths(root_path, root_id, tree_oid)
            except DeterministicRepairAnalyzerHealthError:
                head = str(root_meta.get("head") or "HEAD")
                entries = _enumerate_root_paths(root_path, root_id, head)
            for entry in entries:
                kind, reason, language, parser_status = _classify_path(
                    entry["path"], entry["mode"]
                )
                content_digest = ""
                blob_oid = entry["blob_oid"]
                if kind in {
                    DispositionKind.SEMANTIC_AST,
                    DispositionKind.STRUCTURED_DATA,
                }:
                    try:
                        size_raw = _run_git(
                            root_path, "cat-file", "-s", blob_oid, binary=False
                        )
                        size = int(str(size_raw))
                    except (
                        DeterministicRepairAnalyzerHealthError,
                        TypeError,
                        ValueError,
                    ):
                        size = -1
                    if size < 0:
                        kind = DispositionKind.UNSUPPORTED
                        reason = "blob_unreadable"
                        parser_status = ParserStatus.UNSUPPORTED
                        language = ""
                    elif size > max_source_bytes:
                        kind = DispositionKind.UNSUPPORTED
                        reason = "oversized_source"
                        parser_status = ParserStatus.UNSUPPORTED
                        language = ""
                    elif size == 0:
                        content_digest = _content_digest_for_blob(b"")
                        if parse_sources and language in {"json", "jsonc"}:
                            parser_status, reason, kind = _parse_source(
                                provider,
                                language=language,
                                payload=b"",
                                content_digest=content_digest,
                            )
                        elif parse_sources and language == "python":
                            parser_status = ParserStatus.INDEXED
                            reason = "semantic_ast"
                            kind = DispositionKind.SEMANTIC_AST
                    elif parse_sources:
                        try:
                            payload = _blob_bytes(
                                root_path, blob_oid, max_bytes=max_source_bytes
                            )
                        except DeterministicRepairAnalyzerHealthError as exc:
                            kind = DispositionKind.UNSUPPORTED
                            reason = exc.reason_code
                            parser_status = ParserStatus.UNSUPPORTED
                            language = ""
                        else:
                            content_digest = _content_digest_for_blob(payload)
                            parser_status, reason, kind = _parse_source(
                                provider,
                                language=language,
                                payload=payload,
                                content_digest=content_digest,
                            )
                dispositions.append(
                    PathDisposition(
                        root_id=root_id,
                        path=entry["path"],
                        kind=kind,
                        reason_code=reason,
                        content_digest=content_digest,
                        language=language,
                        parser_status=parser_status,
                        mode=entry["mode"],
                        blob_oid=blob_oid,
                    )
                )
                funnel[f"kind:{kind.value}"] += 1
                funnel[f"status:{parser_status.value}"] += 1
                funnel[f"reason:{reason}"] += 1
    finally:
        if provider is not None:
            provider.close_typescript_worker()

    if not dispositions:
        raise DeterministicRepairAnalyzerHealthError("empty_disposition_ledger")
    paths = [(item.root_id, item.path) for item in dispositions]
    if len(paths) != len(set(paths)):
        raise DeterministicRepairAnalyzerHealthError("duplicate_disposition_path")

    ledger, uncompressed_digest = _encode_disposition_ledger(dispositions)
    root_merkle = _root_merkle(dispositions)
    active_failures = sum(
        1
        for item in dispositions
        if item.parser_status is ParserStatus.PARSE_FAILURE
        and PurePosixPath(item.path).suffix.casefold() in _ACTIVE_SOURCE_SUFFIXES
    )
    unreviewed_unsupported = sorted(
        {
            item.reason_code
            for item in dispositions
            if item.kind is DispositionKind.UNSUPPORTED
            and item.reason_code not in _REVIEWED_UNSUPPORTED_REASONS
        }
    )
    parse_failures = sum(
        1
        for item in dispositions
        if item.parser_status is ParserStatus.PARSE_FAILURE
    )
    toolchain_blocking = list(toolchain.get("reason_codes") or [])
    safe = (
        not toolchain_blocking
        and bool(toolchain.get("canary_passed"))
        and active_failures == 0
        and not unreviewed_unsupported
        and parse_failures == 0
    )
    health_status = (
        AnalyzerHealthStatus.HEALTHY.value
        if safe
        else (
            AnalyzerHealthStatus.PARTIAL.value
            if parse_failures or active_failures or toolchain_blocking
            else AnalyzerHealthStatus.UNHEALTHY.value
        )
    )
    orchestration_head = _git_oid(workspace, "rev-parse", "HEAD")
    orchestration_tree = _git_oid(workspace, "rev-parse", "HEAD^{tree}")
    lifecycle = {
        "schema": ANALYZER_HEALTH_LIFECYCLE_SCHEMA,
        "task_id": DCR_TASK_ID,
        "subject_root_id": "orchestration",
        "subject_head": orchestration_head,
        "subject_tree": orchestration_tree,
        "artifact_path": DCR_ARTIFACT_PATH,
        "todo_path": DCR_TODO_PATH,
        "carrier_subject": DCR_CARRIER_SUBJECT,
        "todo_subject": DCR_TODO_SUBJECT,
        "max_transition_commits": 3,
        "predecessor_forest_id": forest_proof["forest_id"],
        "predecessor_completion_commit": forest_proof["completion_commit"],
    }
    identity = {
        "schema": ANALYZER_HEALTH_SCHEMA,
        "interface": ANALYZER_HEALTH_INTERFACE,
        "repository_index_interface": REPOSITORY_INDEX_INTERFACE,
        "forest_id": forest_proof["forest_id"],
        "forest_historical_proof": forest_proof,
        "lifecycle": lifecycle,
        "toolchain": toolchain,
        "funnel": {
            "path_count": len(dispositions),
            "parse_failures": parse_failures,
            "active_source_parse_failures": active_failures,
            "unreviewed_unsupported": unreviewed_unsupported,
            "counts": dict(sorted(funnel.items())),
        },
        "root_merkle": root_merkle,
        "disposition_ledger": ledger,
        "disposition_uncompressed_digest": uncompressed_digest,
        "health": {
            "status": health_status,
            "safe_for_completion_reasoning": safe,
            "thresholds": {
                "max_active_source_parse_failures": 0,
                "require_compiler_canary": True,
                "require_reviewed_unsupported_only": True,
                "max_source_bytes": max_source_bytes,
            },
            "reason_codes": (
                toolchain_blocking
                + (
                    ["active_source_parse_failures"]
                    if active_failures
                    else []
                )
                + (
                    ["unreviewed_unsupported_classifications"]
                    if unreviewed_unsupported
                    else []
                )
                + (["parse_failures_present"] if parse_failures else [])
            ),
        },
        "max_source_bytes": max_source_bytes,
        "parser_versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "typescript": PINNED_TYPESCRIPT_VERSION,
            "json": "stdlib-json@1",
            "jsonc": "dcr-jsonc@1",
            "typescript_extractor": "typescript-ast-extractor@2",
            "persistent_typescript_worker": True,
        },
        "authoritative": False,
        "completion_authorized": False,
    }
    analyzer_health_id = _content_id(identity)
    return {
        **identity,
        "analyzer_health_id": analyzer_health_id,
    }


def write_analyzer_health(
    destination: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    parse_sources: bool = True,
) -> dict[str, Any]:
    payload = materialize_analyzer_health(
        workspace_root,
        forest_path=forest_path,
        max_source_bytes=max_source_bytes,
        parse_sources=parse_sources,
    )
    encoded = _artifact_bytes(payload)
    if len(encoded) > max_bytes:
        raise DeterministicRepairAnalyzerHealthError(
            "artifact_exceeds_admission_limit",
            f"{len(encoded)} > {max_bytes}",
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return payload


@dataclass(frozen=True)
class AnalyzerHealthValidation:
    integrity_valid: bool = False
    current: bool = False
    downstream_authorized: bool = False
    lifecycle_state: str = "invalid"
    analyzer_health_id: str = ""
    forest_id: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.downstream_authorized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYZER_HEALTH_VALIDATION_SCHEMA,
            "valid": self.valid,
            "integrity_valid": self.integrity_valid,
            "current": self.current,
            "downstream_authorized": self.downstream_authorized,
            "lifecycle_state": self.lifecycle_state,
            "analyzer_health_id": self.analyzer_health_id,
            "forest_id": self.forest_id,
            "reason_codes": list(self.reason_codes),
        }


def _artifact_matches(
    root: Path, payload: Mapping[str, Any], *, carrier_commit: str | None = None
) -> bool:
    expected = _artifact_bytes(payload)
    try:
        if carrier_commit:
            observed = _run_git(root, "show", f"{carrier_commit}:{DCR_ARTIFACT_PATH}")
            assert isinstance(observed, bytes)
        else:
            path = root / DCR_ARTIFACT_PATH
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111:
                return False
            observed = path.read_bytes()
    except (DeterministicRepairAnalyzerHealthError, OSError):
        return False
    return observed == expected


def _todo_delta_is_exact(root: Path, observed: str, completed: str) -> bool:
    try:
        before_raw = _run_git(root, "show", f"{observed}:{DCR_TODO_PATH}")
        after_raw = _run_git(root, "show", f"{completed}:{DCR_TODO_PATH}")
        assert isinstance(before_raw, bytes) and isinstance(after_raw, bytes)
        before = before_raw.decode("utf-8")
        after = after_raw.decode("utf-8")
    except (DeterministicRepairAnalyzerHealthError, UnicodeDecodeError):
        return False
    marker = "## DCR-012 "
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


def _commit_parents(root: Path, commit: str) -> tuple[str, ...]:
    text = str(_run_git(root, "rev-list", "--parents", "-n", "1", commit, binary=False))
    fields = text.split()
    if (
        not fields
        or fields[0] != commit
        or not all(_OID_PATTERN.fullmatch(item) for item in fields)
    ):
        raise DeterministicRepairAnalyzerHealthError("invalid_commit_graph")
    return tuple(fields[1:])


def _commit_subject(root: Path, commit: str) -> str:
    return str(_run_git(root, "show", "-s", "--format=%s", commit, binary=False))


def _commit_tree(root: Path, commit: str) -> str:
    return str(_run_git(root, "rev-parse", f"{commit}^{{tree}}", binary=False)).lower()


def _changed_paths(root: Path, parent: str, commit: str) -> tuple[str, ...]:
    raw = _run_git(
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
    paths: list[str] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        paths.append(encoded.decode("utf-8", "surrogateescape"))
    return tuple(sorted(paths))


def _lifecycle_state(
    root: Path,
    payload: Mapping[str, Any],
    *,
    subject: str,
    current: str,
) -> tuple[str, tuple[str, ...]]:
    if not _artifact_matches(root, payload):
        return "stale", ("capture_artifact_mismatch",)
    if current == subject:
        return "captured", ()
    if not _git_is_ancestor(root, subject, current):
        return "stale", ("observed_repository_commit_not_ancestor",)
    raw = _run_git(
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
        or not _artifact_matches(root, payload, carrier_commit=carrier)
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
        or "dcr-012" not in subject_text
    ):
        return "stale", ("integration_transition_invalid",)
    if len(commits) == 2:
        return "integrated", ()
    completed = commits[2]
    if (
        len(commits) != 3
        or _commit_parents(root, completed) != (merge,)
        or _commit_subject(root, completed) != DCR_TODO_SUBJECT
        or _changed_paths(root, merge, completed) != (DCR_TODO_PATH,)
        or not _todo_delta_is_exact(root, subject, completed)
    ):
        return "stale", ("todo_transition_invalid",)
    return "todo_completed", ()


def _document_integrity_analyzer(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    reasons: list[str] = []
    if payload.get("schema") != ANALYZER_HEALTH_SCHEMA:
        reasons.append("invalid_schema")
    claimed = payload.get("analyzer_health_id")
    try:
        recomputed = _content_id(
            {key: value for key, value in payload.items() if key != "analyzer_health_id"}
        )
    except DeterministicRepairAnalyzerHealthError:
        recomputed = ""
    if not isinstance(claimed, str) or claimed != recomputed:
        reasons.append("analyzer_health_id_mismatch")
    ledger = payload.get("disposition_ledger")
    if not isinstance(ledger, Mapping):
        reasons.append("invalid_disposition_ledger")
        rows: tuple[PathDisposition, ...] = ()
    else:
        try:
            rows, digest = decode_disposition_ledger(ledger)
            if digest != payload.get("disposition_uncompressed_digest"):
                reasons.append("disposition_uncompressed_digest_mismatch")
            if int(ledger.get("row_count", -1)) != len(rows):
                reasons.append("disposition_row_count_mismatch")
            if len(rows) != len({(item.root_id, item.path) for item in rows}):
                reasons.append("duplicate_disposition_path")
            observed_roots = {item.root_id for item in rows}
            if observed_roots - set(DCR_ROOT_IDS):
                reasons.append("unknown_root_in_dispositions")
            if set(DCR_ROOT_IDS) - observed_roots:
                reasons.append("missing_root_in_dispositions")
            funnel = payload.get("funnel")
            if not isinstance(funnel, Mapping) or int(
                funnel.get("path_count", -1)
            ) != len(rows):
                reasons.append("funnel_path_count_mismatch")
        except DeterministicRepairAnalyzerHealthError as exc:
            reasons.append(exc.reason_code)
            rows = ()
    forest_proof = payload.get("forest_historical_proof")
    if not isinstance(forest_proof, Mapping) or not forest_proof.get("integrity_valid"):
        reasons.append("forest_historical_proof_invalid")
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != DCR_TASK_ID:
        reasons.append("invalid_lifecycle_policy")
    health = payload.get("health")
    if not isinstance(health, Mapping):
        reasons.append("invalid_health_block")
    if reasons:
        return None, tuple(dict.fromkeys(reasons))
    return dict(payload), ()


def validate_analyzer_health(
    source: Mapping[str, Any] | str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> AnalyzerHealthValidation:
    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    if isinstance(source, Mapping):
        payload = dict(source)
        encoded = _artifact_bytes(payload)
    else:
        path = Path(source)
        try:
            encoded = path.read_bytes()
        except OSError as exc:
            return AnalyzerHealthValidation(reason_codes=("artifact_unreadable",))
        try:
            payload = dict(_read_json_bytes(encoded, reason="artifact_unreadable"))
        except DeterministicRepairAnalyzerHealthError as exc:
            return AnalyzerHealthValidation(reason_codes=(exc.reason_code,))
    if len(encoded) > max_bytes:
        return AnalyzerHealthValidation(
            reason_codes=("artifact_exceeds_admission_limit",)
        )
    document, integrity_reasons = _document_integrity_analyzer(payload)
    analyzer_health_id = str(payload.get("analyzer_health_id") or "")
    forest_id = str(payload.get("forest_id") or "")
    if document is None:
        return AnalyzerHealthValidation(
            analyzer_health_id=analyzer_health_id,
            forest_id=forest_id,
            reason_codes=integrity_reasons,
        )
    # Re-verify the bound forest historically without requiring live currentness.
    forest_file = (
        Path(forest_path)
        if forest_path is not None
        else workspace.joinpath(*PurePosixPath(DCR_FOREST_PATH).parts)
    )
    try:
        forest_payload = _read_json_bytes(
            forest_file.read_bytes(), reason="forest_unreadable"
        )
        forest_proof = _prove_historical_forest(workspace, forest_payload)
        if forest_proof.get("forest_id") != payload.get("forest_id"):
            return AnalyzerHealthValidation(
                integrity_valid=True,
                analyzer_health_id=analyzer_health_id,
                forest_id=forest_id,
                reason_codes=("forest_id_mismatch",),
            )
    except DeterministicRepairAnalyzerHealthError as exc:
        return AnalyzerHealthValidation(
            integrity_valid=True,
            analyzer_health_id=analyzer_health_id,
            forest_id=forest_id,
            reason_codes=(exc.reason_code,),
        )
    lifecycle = payload["lifecycle"]
    subject = str(lifecycle.get("subject_head") or "")
    try:
        current = _git_oid(workspace, "rev-parse", "HEAD")
        state, lifecycle_reasons = _lifecycle_state(
            workspace,
            payload,
            subject=subject,
            current=current,
        )
    except DeterministicRepairAnalyzerHealthError as exc:
        state, lifecycle_reasons = "stale", (exc.reason_code,)
    current_flag = not lifecycle_reasons
    health = payload.get("health") if isinstance(payload.get("health"), Mapping) else {}
    safe = bool(health.get("safe_for_completion_reasoning"))
    completion_reasons: tuple[str, ...] = ()
    # Final todo-completed authority still requires healthy parse evidence.
    if current_flag and state == "todo_completed" and not safe:
        completion_reasons = ("analyzer_not_completion_safe",)
    downstream = (
        current_flag
        and not completion_reasons
        and state
        in {
            "captured",
            "artifact_carried",
            "integrated",
            "todo_completed",
        }
    )
    return AnalyzerHealthValidation(
        integrity_valid=True,
        current=current_flag,
        downstream_authorized=downstream,
        lifecycle_state=state,
        analyzer_health_id=analyzer_health_id,
        forest_id=forest_id,
        reason_codes=lifecycle_reasons + completion_reasons,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--forest", default=DCR_FOREST_PATH)
    parser.add_argument("--artifact", default=DCR_ARTIFACT_PATH)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Enumerate and classify without invoking language parsers.",
    )
    arguments = parser.parse_args(argv)
    workspace = Path(arguments.workspace).resolve(strict=False)
    forest = Path(arguments.forest)
    if not forest.is_absolute():
        forest = workspace / forest
    artifact = Path(arguments.artifact)
    if not artifact.is_absolute():
        artifact = workspace / artifact
    if arguments.command == "materialize":
        try:
            payload = write_analyzer_health(
                artifact,
                workspace,
                forest_path=forest,
                max_bytes=arguments.max_bytes,
                max_source_bytes=arguments.max_source_bytes,
                parse_sources=not arguments.skip_parse,
            )
        except DeterministicRepairAnalyzerHealthError as exc:
            sys.stdout.write(
                json.dumps(
                    {"ok": False, "reason_code": exc.reason_code, "message": str(exc)},
                    sort_keys=True,
                )
                + "\n"
            )
            return 1
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "analyzer_health_id": payload.get("analyzer_health_id"),
                    "path_count": payload.get("funnel", {}).get("path_count"),
                    "safe_for_completion_reasoning": payload.get("health", {}).get(
                        "safe_for_completion_reasoning"
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    expected = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    if artifact.resolve(strict=False) != expected.resolve(strict=False):
        result = AnalyzerHealthValidation(reason_codes=("analyzer_output_path_invalid",))
    else:
        result = validate_analyzer_health(
            artifact,
            workspace,
            forest_path=forest,
            max_bytes=arguments.max_bytes,
        )
    sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return (
        0
        if result.integrity_valid and result.current and result.downstream_authorized
        else 1
    )


__all__ = [
    "ANALYZER_HEALTH_INTERFACE",
    "ANALYZER_HEALTH_SCHEMA",
    "DCR_ARTIFACT_PATH",
    "DCR_CARRIER_SUBJECT",
    "DCR_FOREST_PATH",
    "DCR_TASK_ID",
    "DCR_TODO_SUBJECT",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DISPOSITION_CODEC",
    "AnalyzerHealthValidation",
    "DeterministicRepairAnalyzerHealthError",
    "DispositionKind",
    "ParserStatus",
    "PathDisposition",
    "decode_disposition_ledger",
    "main",
    "materialize_analyzer_health",
    "validate_analyzer_health",
    "write_analyzer_health",
]


if __name__ == "__main__":
    raise SystemExit(main())
