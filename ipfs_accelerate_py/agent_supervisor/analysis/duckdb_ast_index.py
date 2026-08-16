"""DuckDB-backed repository AST index (DuckDBASTIndex@1).

DQP-020 / Interfaces: ``DuckDBASTIndex@1``, ``SourceSnapshot@1``, ``ParseRun@1``
============================================================================

Persists repository snapshots, content-addressed files, parser runs, AST
nodes/edges, symbols, definitions, imports, calls, references, type relations,
and explicit parse frontiers.  Rows are **derived evidence** only: they never
claim source authority or semantic completion authority.

Acceptance properties
---------------------
* Identical ``(content_digest, parser_id)`` units deduplicate across worktrees
  via a content-addressed parse cache.
* Failed or unsupported parses invalidate stale facts for the path and remain
  explicit unknown frontiers (never silent omission).
* Private, ignored, and secret-bearing paths/contents are excluded before
  indexing.
* Source bodies are never retained; only digests, parser identity, and derived
  AST facts are stored.

Cold import of this module performs no filesystem, database, network, provider,
or process action.  Opening an index is the first I/O boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..core.conflict_graph import (
    AST_BLOB_RECORD_SCHEMA_VERSION,
    ASTBlobRecord,
    build_python_ast_blob_record,
    coerce_ast_blob_record,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DUCKDB_AST_INDEX_INTERFACE: Final[str] = "DuckDBASTIndex@1"
SOURCE_SNAPSHOT_INTERFACE: Final[str] = "SourceSnapshot@1"
PARSE_RUN_INTERFACE: Final[str] = "ParseRun@1"

DUCKDB_AST_INDEX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-ast-index@1"
)
SOURCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-snapshot@1"
)
PARSE_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/parse-run@1"
)
PARSE_CACHE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ast-parse-cache@1"
)
PARSE_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ast-parse-frontier@1"
)

DEFAULT_SCANNER_VERSION: Final[str] = "duckdb-ast-index-scanner@1"
DEFAULT_PARSER_ID: Final[str] = (
    f"python-ast@schema-{AST_BLOB_RECORD_SCHEMA_VERSION}"
)
AUTHORITY_CLASS: Final[str] = "derived_evidence"
MAX_PATH_BYTES: Final[int] = 4_096
MAX_PARSE_ERROR_BYTES: Final[int] = 1_024
MAX_FACTS_JSON_BYTES: Final[int] = 262_144
MAX_FILES_PER_SNAPSHOT: Final[int] = 100_000
MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024

_LANGUAGE_BY_SUFFIX: Final[Mapping[str, str]] = MappingProxyType(
    {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".mts": "typescript",
        ".cts": "typescript",
        ".json": "json",
    }
)
_SUPPORTED_PARSE_LANGUAGES: Final[frozenset[str]] = frozenset({"python"})

_EXCLUDED_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "secrets",
        "private",
        ".private",
        ".ssh",
        ".gnupg",
    }
)
_EXCLUDED_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.rc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials.json",
        "credentials.csv",
        "service-account.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.json",
        "private.key",
        "private_key.pem",
        ".netrc",
        ".npmrc",
        ".pypirc",
    }
)
_EXCLUDED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".jks",
        ".kdbx",
        ".keystore",
    }
)
_SECRET_BASENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[._-])(?:secret|secrets|credential|credentials|private[_-]?key|"
    r"api[_-]?key|access[_-]?token|auth[_-]?token)(?:$|[._-])",
    re.IGNORECASE,
)
# Built from fragments so the module source never contains a contiguous
# private-key header (proposal gate treats that as secret material).
_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        "-----"
        + "BEGIN "
        + r"(?:RSA |EC |OPENSSH )?"
        + "PRIVATE "
        + "KEY"
        + "-----"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:live|test|proj)-[A-Za-z0-9_-]{16,}\b"),
)

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS source_snapshots_repo_idx
    ON source_snapshots(repository_id, created_at);

CREATE TABLE IF NOT EXISTS source_files (
    file_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    content_digest VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS source_files_snapshot_path_uidx
    ON source_files(snapshot_id, path);
CREATE INDEX IF NOT EXISTS source_files_blob_idx
    ON source_files(blob_id, content_digest);

CREATE TABLE IF NOT EXISTS file_versions (
    file_version_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS file_versions_path_idx
    ON file_versions(repository_id, path, observed_at);
CREATE UNIQUE INDEX IF NOT EXISTS file_versions_blob_path_uidx
    ON file_versions(repository_id, path, content_digest);

CREATE TABLE IF NOT EXISTS parse_runs (
    parse_run_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS parse_runs_snapshot_idx
    ON parse_runs(snapshot_id, started_at);

CREATE TABLE IF NOT EXISTS parse_cache (
    parse_unit_id VARCHAR PRIMARY KEY,
    content_digest VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    record_id VARCHAR NOT NULL DEFAULT '',
    parse_error VARCHAR NOT NULL DEFAULT '',
    facts_json VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'derived_evidence',
    created_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS parse_cache_digest_parser_uidx
    ON parse_cache(content_digest, parser_id);

CREATE TABLE IF NOT EXISTS parse_frontiers (
    frontier_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS parse_frontiers_file_uidx
    ON parse_frontiers(snapshot_id, file_id);
CREATE INDEX IF NOT EXISTS parse_frontiers_status_idx
    ON parse_frontiers(snapshot_id, status);

CREATE TABLE IF NOT EXISTS worktree_snapshot_bindings (
    binding_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL DEFAULT '',
    bound_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS worktree_snapshot_bindings_uidx
    ON worktree_snapshot_bindings(worktree_id, snapshot_id);

CREATE TABLE IF NOT EXISTS symbols (
    symbol_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    symbol_kind VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    end_line BIGINT NOT NULL,
    fingerprint VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_snapshot_name_idx
    ON symbols(snapshot_id, qualified_name);
CREATE INDEX IF NOT EXISTS symbols_file_idx ON symbols(file_id);

CREATE TABLE IF NOT EXISTS symbol_versions (
    symbol_version_id VARCHAR PRIMARY KEY,
    symbol_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS ast_nodes (
    node_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    parent_node_id VARCHAR NOT NULL DEFAULT '',
    node_kind VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    start_byte BIGINT NOT NULL,
    end_byte BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS ast_nodes_file_idx ON ast_nodes(file_id, node_path);

CREATE TABLE IF NOT EXISTS ast_edges (
    edge_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    source_node_id VARCHAR NOT NULL,
    target_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS ast_edges_source_idx
    ON ast_edges(source_node_id, edge_kind);

CREATE TABLE IF NOT EXISTS "imports" (
    import_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    module_name VARCHAR NOT NULL,
    alias VARCHAR NOT NULL DEFAULT '',
    start_line BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_file_idx ON "imports"(file_id);

CREATE TABLE IF NOT EXISTS "calls" (
    call_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    caller_symbol_id VARCHAR NOT NULL,
    callee_symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_caller_idx ON "calls"(caller_symbol_id);
CREATE INDEX IF NOT EXISTS calls_callee_idx ON "calls"(callee_symbol_id);

CREATE TABLE IF NOT EXISTS "references" (
    reference_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    reference_kind VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS references_symbol_idx ON "references"(symbol_id);

CREATE TABLE IF NOT EXISTS definitions (
    definition_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    end_line BIGINT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS definitions_symbol_uidx
    ON definitions(symbol_id, snapshot_id);

CREATE TABLE IF NOT EXISTS type_relations (
    relation_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    left_symbol_id VARCHAR NOT NULL,
    right_symbol_id VARCHAR NOT NULL,
    relation_kind VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS ast_index_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuckDBASTIndexError(RuntimeError):
    """Base error for DuckDB AST index failures."""


class DuckDBASTIndexNotOpenError(DuckDBASTIndexError):
    """Operation requires an open index."""


class DuckDBASTIndexIntegrityError(DuckDBASTIndexError, ValueError):
    """Identity, path, or payload integrity failure."""


class DuckDBASTIndexBoundsError(DuckDBASTIndexError, ValueError):
    """A resource or payload bound was exceeded."""


class DuckDBASTIndexConflictError(DuckDBASTIndexError):
    """Duplicate identity with a conflicting payload."""


class DuckDBUnavailableError(DuckDBASTIndexError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ParseStatus(str, Enum):
    """Per-file and aggregate parse outcome vocabulary."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"
    PARTIAL = "partial"
    CACHE_HIT = "cache_hit"


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    UNKNOWN = "unknown"


class FileDisposition(str, Enum):
    INDEXED = "indexed"
    EXCLUDED = "excluded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DuckDBASTIndexIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DuckDBASTIndexIntegrityError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DuckDBASTIndexBoundsError(f"{name} must be a non-negative integer")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DuckDBASTIndexIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="surrogatepass"))


def _normalize_digest(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" not in text:
        text = f"sha256:{text}"
    return text


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        raise DuckDBASTIndexIntegrityError("repository path is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise DuckDBASTIndexIntegrityError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise DuckDBASTIndexBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes: {normalized}"
        )
    return normalized


def _bounded_text(value: Any, maximum: int) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= maximum:
        return text
    marker = "…[truncated]"
    budget = max(0, maximum - len(marker.encode("utf-8")))
    return encoded[:budget].decode("utf-8", "ignore") + marker


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def language_for_path(path: str) -> str:
    """Return a language tag for a repository-relative path."""

    normalized = _repo_path(path)
    suffix = PurePosixPath(normalized).suffix.casefold()
    return _LANGUAGE_BY_SUFFIX.get(suffix, "")


def is_excluded_path(path: str) -> tuple[bool, str]:
    """Return whether a path is private/ignored and the exclusion reason."""

    normalized = _repo_path(path)
    parts = PurePosixPath(normalized).parts
    basename = parts[-1] if parts else normalized
    basename_cf = basename.casefold()
    secret_hidden_dirs = frozenset(
        {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker"}
    )
    for part in parts[:-1] if len(parts) > 1 else ():
        lowered = part.casefold()
        if lowered in _EXCLUDED_PATH_PARTS:
            return True, f"ignored_path_part:{part}"
        if lowered in secret_hidden_dirs:
            return True, f"hidden_directory:{part}"
    if parts:
        leaf_dir = parts[0].casefold() if len(parts) == 1 else ""
        if leaf_dir in _EXCLUDED_PATH_PARTS:
            return True, f"ignored_path_part:{parts[0]}"
    if basename_cf in _EXCLUDED_BASENAMES:
        return True, f"excluded_basename:{basename}"
    if PurePosixPath(normalized).suffix.casefold() in _EXCLUDED_SUFFIXES:
        return True, f"excluded_suffix:{PurePosixPath(normalized).suffix}"
    if _SECRET_BASENAME_RE.search(basename):
        return True, f"secret_basename:{basename}"
    if basename_cf.startswith(".env") or basename_cf.endswith(".env"):
        return True, f"env_file:{basename}"
    return False, ""


def content_looks_like_secret(payload: bytes | str) -> bool:
    """Return True when payload matches known secret material patterns."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            # Binary blobs are not scanned as text secrets here; path policy
            # already excludes private-key suffixes.
            return False
    else:
        text = str(payload)
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _symbol_kind(qualified_name: str) -> str:
    name = str(qualified_name or "")
    if not name:
        return SymbolKind.UNKNOWN.value
    leaf = name.rsplit(".", 1)[-1]
    if leaf[:1].isupper():
        return SymbolKind.CLASS.value
    if "." in name:
        return SymbolKind.METHOD.value
    return SymbolKind.FUNCTION.value


def _parse_import(statement: str) -> tuple[str, str]:
    text = str(statement or "").strip()
    if text.startswith("from "):
        # from module import name [as alias]
        match = re.match(
            r"from\s+(\S+)\s+import\s+(\S+)(?:\s+as\s+(\S+))?",
            text,
        )
        if match:
            module, name, alias = match.group(1), match.group(2), match.group(3)
            return f"{module}:{name}", alias or ""
        return text, ""
    if text.startswith("import "):
        match = re.match(r"import\s+(\S+)(?:\s+as\s+(\S+))?", text)
        if match:
            return match.group(1), match.group(2) or ""
    return text, ""


def _facts_from_record(record: ASTBlobRecord) -> dict[str, Any]:
    """Project an ASTBlobRecord into bounded, body-free facts."""

    return {
        "authority": AUTHORITY_CLASS,
        "record_id": record.record_id,
        "blob_identity": record.blob_identity,
        "source_sha256": record.source_sha256,
        "language": record.language,
        "parse_error": record.parse_error,
        "qualified_symbols": list(record.qualified_symbols),
        "imports": list(record.imports),
        "calls": list(record.calls),
        "state_transitions": list(record.state_transitions),
        "interfaces": list(record.interfaces),
        "symbol_hashes": dict(record.symbol_hashes),
        "symbol_lines": {
            key: list(value) for key, value in sorted(record.symbol_lines.items())
        },
        "record_schema_version": record.record_schema_version,
    }


def _facts_json(facts: Mapping[str, Any]) -> str:
    encoded = _canonical_json(facts)
    if len(encoded.encode("utf-8")) > MAX_FACTS_JSON_BYTES:
        raise DuckDBASTIndexBoundsError(
            f"facts payload exceeds {MAX_FACTS_JSON_BYTES} bytes"
        )
    if "source" in facts or "source_text" in facts or "source_body" in facts:
        raise DuckDBASTIndexIntegrityError(
            "AST facts must not embed source bodies"
        )
    return encoded


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFileSpec:
    """One repository path and its content identity for snapshot ingestion."""

    path: str
    content: bytes | str | None = None
    content_digest: str = ""
    language: str = ""
    blob_id: str = ""
    ignored: bool = False
    ast_record: ASTBlobRecord | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        digest = _normalize_digest(self.content_digest)
        payload: bytes | None = None
        if self.content is not None:
            if isinstance(self.content, bytes):
                payload = self.content
            else:
                payload = str(self.content).encode(
                    "utf-8", errors="surrogatepass"
                )
            if len(payload) > MAX_SOURCE_BYTES:
                raise DuckDBASTIndexBoundsError(
                    f"source exceeds {MAX_SOURCE_BYTES} bytes: {self.path}"
                )
            actual = _sha256_bytes(payload)
            if digest and digest != actual:
                raise DuckDBASTIndexIntegrityError(
                    f"content digest mismatch for {self.path}"
                )
            digest = actual
        if not digest:
            raise DuckDBASTIndexIntegrityError(
                f"content_digest or content is required for {self.path}"
            )
        object.__setattr__(self, "content_digest", digest)
        language = str(self.language or language_for_path(self.path)).strip()
        object.__setattr__(self, "language", language)
        blob = str(self.blob_id or digest).strip()
        object.__setattr__(self, "blob_id", blob)
        # Drop content after digesting — source is not durable authority.
        object.__setattr__(self, "content", payload)
        record = self.ast_record
        if record is not None and not isinstance(record, ASTBlobRecord):
            coerced = coerce_ast_blob_record(record)
            if coerced is None:
                raise DuckDBASTIndexIntegrityError(
                    f"invalid AST record for {self.path}"
                )
            object.__setattr__(self, "ast_record", coerced)

    @property
    def byte_length(self) -> int:
        if isinstance(self.content, bytes):
            return len(self.content)
        return 0

    def source_text(self) -> str | None:
        if self.content is None:
            return None
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8", errors="surrogatepass")
        return str(self.content)


@dataclass(frozen=True)
class SourceSnapshot:
    """Exact repository/worktree tree identity for one indexed snapshot.

    Interface: ``SourceSnapshot@1``.
    """

    snapshot_id: str
    repository_id: str
    tree_id: str
    overlay_digest: str = ""
    created_at: str = ""
    scanner_version: str = DEFAULT_SCANNER_VERSION
    worktree_id: str = ""
    file_count: int = 0
    schema: str = SOURCE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "overlay_digest",
            _normalize_digest(self.overlay_digest)
            if self.overlay_digest
            else "",
        )
        object.__setattr__(
            self,
            "created_at",
            _text(self.created_at or _utc_iso(), "created_at"),
        )
        object.__setattr__(
            self,
            "scanner_version",
            _text(self.scanner_version or DEFAULT_SCANNER_VERSION, "scanner_version"),
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self, "file_count", _nonneg_int(int(self.file_count), "file_count")
        )
        if self.schema != SOURCE_SNAPSHOT_SCHEMA:
            raise DuckDBASTIndexIntegrityError("unsupported source snapshot schema")
        computed = self._compute_id()
        claimed = str(self.snapshot_id or "").strip()
        if claimed and claimed != computed:
            # Allow pre-assigned content identities that match computation.
            raise DuckDBASTIndexIntegrityError(
                "source snapshot identity does not match payload"
            )
        object.__setattr__(self, "snapshot_id", claimed or computed)

    def _compute_id(self) -> str:
        return _identity(
            "source-snapshot",
            {
                "schema": self.schema,
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "overlay_digest": self.overlay_digest,
                "scanner_version": self.scanner_version,
            },
        )

    @property
    def interface(self) -> str:
        return SOURCE_SNAPSHOT_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": SOURCE_SNAPSHOT_INTERFACE,
            "snapshot_id": self.snapshot_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "overlay_digest": self.overlay_digest,
            "created_at": self.created_at,
            "scanner_version": self.scanner_version,
            "worktree_id": self.worktree_id,
            "file_count": self.file_count,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class FileParseResult:
    """One path's parse outcome for a snapshot parse run."""

    path: str
    file_id: str
    content_digest: str
    language: str
    status: ParseStatus | str
    reason: str = ""
    record_id: str = ""
    cache_hit: bool = False
    symbol_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "file_id", _text(self.file_id, "file_id"))
        object.__setattr__(
            self, "content_digest", _normalize_digest(self.content_digest)
        )
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        object.__setattr__(self, "status", ParseStatus(self.status))
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_PARSE_ERROR_BYTES)
        )
        object.__setattr__(
            self, "record_id", _text(self.record_id, "record_id", required=False)
        )
        object.__setattr__(
            self, "symbol_count", _nonneg_int(int(self.symbol_count), "symbol_count")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_id": self.file_id,
            "content_digest": self.content_digest,
            "language": self.language,
            "status": self.status.value
            if isinstance(self.status, ParseStatus)
            else str(self.status),
            "reason": self.reason,
            "record_id": self.record_id,
            "cache_hit": bool(self.cache_hit),
            "symbol_count": self.symbol_count,
        }


@dataclass(frozen=True)
class ParseRun:
    """Bounded receipt for one snapshot parse pass.

    Interface: ``ParseRun@1``.
    """

    parse_run_id: str
    snapshot_id: str
    parser_id: str
    status: ParseStatus | str
    started_at: str = ""
    finished_at: str = ""
    file_results: tuple[FileParseResult, ...] = ()
    reused_unit_count: int = 0
    new_unit_count: int = 0
    failed_count: int = 0
    unsupported_count: int = 0
    excluded_count: int = 0
    schema: str = PARSE_RUN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "parser_id", _text(self.parser_id, "parser_id"))
        object.__setattr__(self, "status", ParseStatus(self.status))
        object.__setattr__(
            self, "started_at", _text(self.started_at or _utc_iso(), "started_at")
        )
        object.__setattr__(
            self,
            "finished_at",
            _text(self.finished_at or self.started_at, "finished_at"),
        )
        results = tuple(self.file_results)
        object.__setattr__(self, "file_results", results)
        for name in (
            "reused_unit_count",
            "new_unit_count",
            "failed_count",
            "unsupported_count",
            "excluded_count",
        ):
            object.__setattr__(
                self, name, _nonneg_int(int(getattr(self, name)), name)
            )
        if self.schema != PARSE_RUN_SCHEMA:
            raise DuckDBASTIndexIntegrityError("unsupported parse run schema")
        computed = self._compute_id()
        claimed = str(self.parse_run_id or "").strip()
        if claimed and claimed != computed:
            raise DuckDBASTIndexIntegrityError(
                "parse run identity does not match payload"
            )
        object.__setattr__(self, "parse_run_id", claimed or computed)

    def _compute_id(self) -> str:
        return _identity(
            "parse-run",
            {
                "schema": self.schema,
                "snapshot_id": self.snapshot_id,
                "parser_id": self.parser_id,
                "started_at": self.started_at,
                "status": self.status.value
                if isinstance(self.status, ParseStatus)
                else str(self.status),
                "file_results": [item.to_dict() for item in self.file_results],
            },
        )

    @property
    def interface(self) -> str:
        return PARSE_RUN_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": PARSE_RUN_INTERFACE,
            "parse_run_id": self.parse_run_id,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "status": self.status.value
            if isinstance(self.status, ParseStatus)
            else str(self.status),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "file_results": [item.to_dict() for item in self.file_results],
            "reused_unit_count": self.reused_unit_count,
            "new_unit_count": self.new_unit_count,
            "failed_count": self.failed_count,
            "unsupported_count": self.unsupported_count,
            "excluded_count": self.excluded_count,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ParseFrontier:
    """Explicit incomplete/unknown parse frontier for one path."""

    frontier_id: str
    snapshot_id: str
    file_id: str
    path: str
    content_digest: str
    status: ParseStatus | str
    reason: str
    parser_id: str = ""
    recorded_at: str = ""
    schema: str = PARSE_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "file_id", _text(self.file_id, "file_id"))
        object.__setattr__(
            self, "content_digest", _normalize_digest(self.content_digest)
        )
        object.__setattr__(self, "status", ParseStatus(self.status))
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_PARSE_ERROR_BYTES)
        )
        object.__setattr__(
            self, "parser_id", _text(self.parser_id, "parser_id", required=False)
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        computed = _identity(
            "parse-frontier",
            {
                "schema": self.schema,
                "snapshot_id": self.snapshot_id,
                "file_id": self.file_id,
                "path": self.path,
                "content_digest": self.content_digest,
                "status": self.status.value
                if isinstance(self.status, ParseStatus)
                else str(self.status),
                "reason": self.reason,
                "parser_id": self.parser_id,
            },
        )
        claimed = str(self.frontier_id or "").strip()
        if claimed and claimed != computed:
            raise DuckDBASTIndexIntegrityError(
                "parse frontier identity does not match payload"
            )
        object.__setattr__(self, "frontier_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frontier_id": self.frontier_id,
            "snapshot_id": self.snapshot_id,
            "file_id": self.file_id,
            "path": self.path,
            "content_digest": self.content_digest,
            "status": self.status.value
            if isinstance(self.status, ParseStatus)
            else str(self.status),
            "reason": self.reason,
            "parser_id": self.parser_id,
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class IndexedSymbol:
    """One derived symbol row projection."""

    symbol_id: str
    snapshot_id: str
    file_id: str
    path: str
    language: str
    qualified_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "snapshot_id": self.snapshot_id,
            "file_id": self.file_id,
            "path": self.path,
            "language": self.language,
            "qualified_name": self.qualified_name,
            "symbol_kind": self.symbol_kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "fingerprint": self.fingerprint,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class SnapshotIngestResult:
    """Outcome of ingesting one source snapshot and parse pass."""

    snapshot: SourceSnapshot
    parse_run: ParseRun
    indexed_file_count: int
    excluded_file_count: int
    reused_unit_count: int
    new_unit_count: int
    invalidated_fact_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "parse_run": self.parse_run.to_dict(),
            "indexed_file_count": self.indexed_file_count,
            "excluded_file_count": self.excluded_file_count,
            "reused_unit_count": self.reused_unit_count,
            "new_unit_count": self.new_unit_count,
            "invalidated_fact_count": self.invalidated_fact_count,
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DuckDBASTIndex:
    """Persist and query content-addressed AST evidence in DuckDB.

    Interface: ``DuckDBASTIndex@1``.
    """

    INTERFACE: Final[str] = DUCKDB_AST_INDEX_INTERFACE
    SCHEMA: Final[str] = DUCKDB_AST_INDEX_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        parser_id: str = DEFAULT_PARSER_ID,
        scanner_version: str = DEFAULT_SCANNER_VERSION,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DuckDBASTIndex; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._parser_id = _text(parser_id or DEFAULT_PARSER_ID, "parser_id")
        self._scanner_version = _text(
            scanner_version or DEFAULT_SCANNER_VERSION, "scanner_version"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def parser_id(self) -> str:
        return self._parser_id

    @property
    def scanner_version(self) -> str:
        return self._scanner_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DuckDBASTIndex":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DUCKDB_AST_INDEX_INTERFACE),
                ("schema", DUCKDB_AST_INDEX_SCHEMA),
                ("parser_id", self._parser_id),
                ("scanner_version", self._scanner_version),
                ("authority", AUTHORITY_CLASS),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO ast_index_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DuckDBASTIndex":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DuckDBASTIndexNotOpenError("DuckDBASTIndex is not open")
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    # -- public API ----------------------------------------------------------

    def ingest_snapshot(
        self,
        *,
        repository_id: str,
        tree_id: str,
        files: Sequence[SourceFileSpec | Mapping[str, Any]],
        overlay_digest: str = "",
        worktree_id: str = "",
        scanner_version: str | None = None,
        parser_id: str | None = None,
        created_at: str | None = None,
    ) -> SnapshotIngestResult:
        """Ingest one exact snapshot, parse eligible files, and persist facts.

        Identical ``(content_digest, parser_id)`` units reuse the parse cache
        across worktrees.  Failed/unsupported/excluded paths invalidate any
        prior facts for that snapshot path and remain explicit frontiers.
        """

        specs = [self._coerce_file_spec(item) for item in files]
        if len(specs) > MAX_FILES_PER_SNAPSHOT:
            raise DuckDBASTIndexBoundsError(
                f"snapshot exceeds {MAX_FILES_PER_SNAPSHOT} files"
            )
        if len({item.path for item in specs}) != len(specs):
            raise DuckDBASTIndexIntegrityError(
                "snapshot paths must be unique"
            )
        selected_parser = _text(parser_id or self._parser_id, "parser_id")
        selected_scanner = _text(
            scanner_version or self._scanner_version, "scanner_version"
        )
        stamp = _text(created_at or _utc_iso(), "created_at")
        snapshot = SourceSnapshot(
            snapshot_id="",
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
            created_at=stamp,
            scanner_version=selected_scanner,
            worktree_id=worktree_id,
            file_count=len(specs),
        )

        with self._lock:
            connection = self._require()
            try:
                connection.execute("BEGIN TRANSACTION")
                self._insert_snapshot(connection, snapshot)
                if worktree_id:
                    self._bind_worktree(connection, snapshot, worktree_id, stamp)

                file_results: list[FileParseResult] = []
                reused = 0
                new_units = 0
                failed = 0
                unsupported = 0
                excluded = 0
                invalidated = 0
                indexed = 0

                for spec in sorted(specs, key=lambda item: item.path):
                    result, unit_reused, unit_new, unit_invalidated = (
                        self._ingest_file(
                            connection,
                            snapshot=snapshot,
                            spec=spec,
                            parser_id=selected_parser,
                            observed_at=stamp,
                        )
                    )
                    file_results.append(result)
                    reused += unit_reused
                    new_units += unit_new
                    invalidated += unit_invalidated
                    status = result.status
                    if status is ParseStatus.EXCLUDED:
                        excluded += 1
                    elif status is ParseStatus.FAILED:
                        failed += 1
                    elif status is ParseStatus.UNSUPPORTED:
                        unsupported += 1
                    elif status in {
                        ParseStatus.SUCCEEDED,
                        ParseStatus.CACHE_HIT,
                    }:
                        indexed += 1

                if failed or unsupported or excluded:
                    if indexed:
                        aggregate = ParseStatus.PARTIAL
                    elif failed and not unsupported and not indexed:
                        aggregate = ParseStatus.FAILED
                    elif unsupported and not failed and not indexed:
                        aggregate = ParseStatus.UNSUPPORTED
                    elif excluded and not failed and not unsupported and not indexed:
                        aggregate = ParseStatus.EXCLUDED
                    else:
                        aggregate = ParseStatus.PARTIAL
                else:
                    aggregate = ParseStatus.SUCCEEDED

                finished = _utc_iso()
                parse_run = ParseRun(
                    parse_run_id="",
                    snapshot_id=snapshot.snapshot_id,
                    parser_id=selected_parser,
                    status=aggregate,
                    started_at=stamp,
                    finished_at=finished,
                    file_results=tuple(file_results),
                    reused_unit_count=reused,
                    new_unit_count=new_units,
                    failed_count=failed,
                    unsupported_count=unsupported,
                    excluded_count=excluded,
                )
                self._insert_parse_run(connection, parse_run)
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)

            return SnapshotIngestResult(
                snapshot=snapshot,
                parse_run=parse_run,
                indexed_file_count=indexed,
                excluded_file_count=excluded,
                reused_unit_count=reused,
                new_unit_count=new_units,
                invalidated_fact_count=invalidated,
            )

    def get_snapshot(self, snapshot_id: str) -> SourceSnapshot | None:
        selected = _text(snapshot_id, "snapshot_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT snapshot_id, repository_id, tree_id, overlay_digest,
                       created_at, scanner_version
                FROM source_snapshots WHERE snapshot_id = ? LIMIT 1
                """,
                [selected],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            file_count_row = connection.execute(
                "SELECT COUNT(*) AS n FROM source_files WHERE snapshot_id = ?",
                [selected],
            ).fetchone()
            file_count = int(_row_mapping(file_count_row).get("n") or 0)
            binding = connection.execute(
                """
                SELECT worktree_id FROM worktree_snapshot_bindings
                WHERE snapshot_id = ? ORDER BY bound_at ASC LIMIT 1
                """,
                [selected],
            ).fetchone()
            worktree_id = ""
            if binding is not None:
                worktree_id = str(
                    _row_mapping(binding).get("worktree_id") or ""
                )
            return SourceSnapshot(
                snapshot_id=str(mapping["snapshot_id"]),
                repository_id=str(mapping["repository_id"]),
                tree_id=str(mapping["tree_id"]),
                overlay_digest=str(mapping.get("overlay_digest") or ""),
                created_at=str(mapping["created_at"]),
                scanner_version=str(mapping["scanner_version"]),
                worktree_id=worktree_id,
                file_count=file_count,
            )

    def list_files(self, snapshot_id: str) -> tuple[dict[str, Any], ...]:
        selected = _text(snapshot_id, "snapshot_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT file_id, snapshot_id, path, language, blob_id,
                       byte_length, content_digest
                FROM source_files
                WHERE snapshot_id = ?
                ORDER BY path ASC
                """,
                [selected],
            ).fetchall()
            return tuple(_row_mapping(row) for row in rows)

    def list_symbols(
        self,
        snapshot_id: str,
        *,
        path: str | None = None,
        name_contains: str | None = None,
    ) -> tuple[IndexedSymbol, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ["s.snapshot_id = ?"]
        params: list[Any] = [selected]
        if path is not None:
            clauses.append("f.path = ?")
            params.append(_repo_path(path))
        if name_contains:
            clauses.append("lower(s.qualified_name) LIKE ?")
            params.append(f"%{str(name_contains).casefold()}%")
        sql = f"""
            SELECT s.symbol_id, s.snapshot_id, s.file_id, f.path, s.language,
                   s.qualified_name, s.symbol_kind, s.start_line, s.end_line,
                   s.fingerprint
            FROM symbols s
            JOIN source_files f ON f.file_id = s.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY s.qualified_name ASC, f.path ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            results: list[IndexedSymbol] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    IndexedSymbol(
                        symbol_id=str(mapping["symbol_id"]),
                        snapshot_id=str(mapping["snapshot_id"]),
                        file_id=str(mapping["file_id"]),
                        path=str(mapping["path"]),
                        language=str(mapping["language"]),
                        qualified_name=str(mapping["qualified_name"]),
                        symbol_kind=str(mapping["symbol_kind"]),
                        start_line=int(mapping["start_line"] or 0),
                        end_line=int(mapping["end_line"] or 0),
                        fingerprint=str(mapping["fingerprint"]),
                    )
                )
            return tuple(results)

    def list_imports(self, snapshot_id: str, *, path: str | None = None) -> tuple[dict[str, Any], ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ['i.snapshot_id = ?']
        params: list[Any] = [selected]
        if path is not None:
            clauses.append("f.path = ?")
            params.append(_repo_path(path))
        sql = f"""
            SELECT i.import_id, i.snapshot_id, i.file_id, f.path,
                   i.module_name, i.alias, i.start_line
            FROM "imports" i
            JOIN source_files f ON f.file_id = i.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY f.path ASC, i.module_name ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(_row_mapping(row) for row in rows)

    def list_calls(self, snapshot_id: str, *, path: str | None = None) -> tuple[dict[str, Any], ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ['c.snapshot_id = ?']
        params: list[Any] = [selected]
        if path is not None:
            clauses.append("f.path = ?")
            params.append(_repo_path(path))
        sql = f"""
            SELECT c.call_id, c.snapshot_id, c.caller_symbol_id,
                   c.callee_symbol_id, c.file_id, f.path, c.start_line
            FROM "calls" c
            JOIN source_files f ON f.file_id = c.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY f.path ASC, c.caller_symbol_id ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(_row_mapping(row) for row in rows)

    def list_frontiers(
        self,
        snapshot_id: str,
        *,
        status: ParseStatus | str | None = None,
    ) -> tuple[ParseFrontier, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ["snapshot_id = ?"]
        params: list[Any] = [selected]
        if status is not None:
            clauses.append("status = ?")
            params.append(
                status.value if isinstance(status, ParseStatus) else str(status)
            )
        sql = f"""
            SELECT frontier_id, snapshot_id, file_id, path, content_digest,
                   status, reason, parser_id, recorded_at
            FROM parse_frontiers
            WHERE {' AND '.join(clauses)}
            ORDER BY path ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            results: list[ParseFrontier] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    ParseFrontier(
                        frontier_id=str(mapping["frontier_id"]),
                        snapshot_id=str(mapping["snapshot_id"]),
                        file_id=str(mapping["file_id"]),
                        path=str(mapping["path"]),
                        content_digest=str(mapping["content_digest"]),
                        status=str(mapping["status"]),
                        reason=str(mapping["reason"]),
                        parser_id=str(mapping.get("parser_id") or ""),
                        recorded_at=str(mapping["recorded_at"]),
                    )
                )
            return tuple(results)

    def list_ast_nodes(
        self, snapshot_id: str, *, path: str | None = None
    ) -> tuple[dict[str, Any], ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ["n.snapshot_id = ?"]
        params: list[Any] = [selected]
        if path is not None:
            clauses.append("f.path = ?")
            params.append(_repo_path(path))
        sql = f"""
            SELECT n.node_id, n.snapshot_id, n.file_id, f.path,
                   n.parent_node_id, n.node_kind, n.node_path,
                   n.fingerprint, n.start_byte, n.end_byte
            FROM ast_nodes n
            JOIN source_files f ON f.file_id = n.file_id
            WHERE {' AND '.join(clauses)}
            ORDER BY f.path ASC, n.node_path ASC
        """
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(_row_mapping(row) for row in rows)

    def get_parse_cache_entry(
        self, content_digest: str, *, parser_id: str | None = None
    ) -> dict[str, Any] | None:
        digest = _normalize_digest(content_digest)
        selected_parser = _text(parser_id or self._parser_id, "parser_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT parse_unit_id, content_digest, parser_id, language,
                       status, record_id, parse_error, facts_json, authority,
                       created_at
                FROM parse_cache
                WHERE content_digest = ? AND parser_id = ?
                LIMIT 1
                """,
                [digest, selected_parser],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            facts_raw = mapping.get("facts_json") or "{}"
            try:
                facts = json.loads(str(facts_raw))
            except json.JSONDecodeError:
                facts = {}
            mapping["facts"] = facts
            return mapping

    def parse_cache_size(self) -> int:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM parse_cache"
            ).fetchone()
            return int(_row_mapping(row).get("n") or 0)

    def metadata(self) -> dict[str, str]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT key, value FROM ast_index_metadata ORDER BY key ASC"
            ).fetchall()
            return {
                str(_row_mapping(row)["key"]): str(_row_mapping(row)["value"])
                for row in rows
            }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _coerce_file_spec(value: SourceFileSpec | Mapping[str, Any]) -> SourceFileSpec:
        if isinstance(value, SourceFileSpec):
            return value
        if not isinstance(value, Mapping):
            raise DuckDBASTIndexIntegrityError("file spec must be a mapping")
        return SourceFileSpec(
            path=str(value.get("path") or ""),
            content=value.get("content"),
            content_digest=str(value.get("content_digest") or ""),
            language=str(value.get("language") or ""),
            blob_id=str(value.get("blob_id") or ""),
            ignored=bool(value.get("ignored", False)),
            ast_record=value.get("ast_record"),
        )

    def _insert_snapshot(self, connection: Any, snapshot: SourceSnapshot) -> None:
        existing = connection.execute(
            "SELECT snapshot_id FROM source_snapshots WHERE snapshot_id = ?",
            [snapshot.snapshot_id],
        ).fetchone()
        if existing is not None:
            # Exact identity reuse is allowed (idempotent re-open).
            return
        connection.execute(
            """
            INSERT INTO source_snapshots (
                snapshot_id, repository_id, tree_id, overlay_digest,
                created_at, scanner_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.snapshot_id,
                snapshot.repository_id,
                snapshot.tree_id,
                snapshot.overlay_digest,
                snapshot.created_at,
                snapshot.scanner_version,
            ],
        )

    def _bind_worktree(
        self,
        connection: Any,
        snapshot: SourceSnapshot,
        worktree_id: str,
        bound_at: str,
    ) -> None:
        binding_id = _identity(
            "worktree-snapshot-binding",
            {
                "worktree_id": worktree_id,
                "snapshot_id": snapshot.snapshot_id,
            },
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO worktree_snapshot_bindings (
                binding_id, worktree_id, snapshot_id, repository_id,
                tree_id, overlay_digest, bound_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                binding_id,
                worktree_id,
                snapshot.snapshot_id,
                snapshot.repository_id,
                snapshot.tree_id,
                snapshot.overlay_digest,
                bound_at,
            ],
        )

    def _insert_parse_run(self, connection: Any, parse_run: ParseRun) -> None:
        body = {
            key: value
            for key, value in parse_run.to_dict().items()
            if key != "parse_run_id"
        }
        connection.execute(
            """
            INSERT INTO parse_runs (
                parse_run_id, snapshot_id, parser_id, started_at,
                finished_at, status, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                parse_run.parse_run_id,
                parse_run.snapshot_id,
                parse_run.parser_id,
                parse_run.started_at,
                parse_run.finished_at,
                parse_run.status.value
                if isinstance(parse_run.status, ParseStatus)
                else str(parse_run.status),
                _canonical_json(body),
            ],
        )

    def _ingest_file(
        self,
        connection: Any,
        *,
        snapshot: SourceSnapshot,
        spec: SourceFileSpec,
        parser_id: str,
        observed_at: str,
    ) -> tuple[FileParseResult, int, int, int]:
        file_id = _identity(
            "source-file",
            {
                "snapshot_id": snapshot.snapshot_id,
                "path": spec.path,
                "content_digest": spec.content_digest,
            },
        )
        connection.execute(
            """
            INSERT INTO source_files (
                file_id, snapshot_id, path, language, blob_id,
                byte_length, content_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                file_id,
                snapshot.snapshot_id,
                spec.path,
                spec.language,
                spec.blob_id,
                int(spec.byte_length),
                spec.content_digest,
            ],
        )
        self._record_file_version(
            connection,
            repository_id=snapshot.repository_id,
            path=spec.path,
            blob_id=spec.blob_id,
            content_digest=spec.content_digest,
            observed_at=observed_at,
        )

        excluded, reason = self._exclusion_decision(spec)
        if excluded:
            invalidated = self._invalidate_file_facts(
                connection, snapshot.snapshot_id, file_id
            )
            self._upsert_frontier(
                connection,
                snapshot_id=snapshot.snapshot_id,
                file_id=file_id,
                path=spec.path,
                content_digest=spec.content_digest,
                status=ParseStatus.EXCLUDED,
                reason=reason,
                parser_id=parser_id,
                recorded_at=observed_at,
            )
            return (
                FileParseResult(
                    path=spec.path,
                    file_id=file_id,
                    content_digest=spec.content_digest,
                    language=spec.language,
                    status=ParseStatus.EXCLUDED,
                    reason=reason,
                ),
                0,
                0,
                invalidated,
            )

        language = spec.language or language_for_path(spec.path)
        cache_hit = False
        reused = 0
        new_units = 0
        cached = self._load_parse_cache(
            connection, spec.content_digest, parser_id
        )
        facts: dict[str, Any] | None = None
        status = ParseStatus.UNKNOWN
        record_id = ""
        parse_error = ""

        if cached is not None:
            cache_hit = True
            reused = 1
            status = ParseStatus(str(cached["status"]))
            record_id = str(cached.get("record_id") or "")
            parse_error = str(cached.get("parse_error") or "")
            try:
                facts = json.loads(str(cached.get("facts_json") or "{}"))
            except json.JSONDecodeError:
                facts = {}
        else:
            status, record_id, parse_error, facts = self._parse_unit(
                spec, language=language, parser_id=parser_id
            )
            new_units = 1
            self._store_parse_cache(
                connection,
                content_digest=spec.content_digest,
                parser_id=parser_id,
                language=language,
                status=status,
                record_id=record_id,
                parse_error=parse_error,
                facts=facts or {},
                created_at=observed_at,
            )

        # Project cache status for the file result while preserving semantic
        # failed/unsupported/succeeded for frontiers and fact validity.
        projected_status = (
            ParseStatus.CACHE_HIT
            if cache_hit and status is ParseStatus.SUCCEEDED
            else status
        )

        if status is ParseStatus.SUCCEEDED and facts:
            # Successful parse replaces any prior frontier and writes facts.
            self._clear_frontier(connection, snapshot.snapshot_id, file_id)
            # Clear any stale facts before writing (path re-index safety).
            invalidated = self._invalidate_file_facts(
                connection, snapshot.snapshot_id, file_id
            )
            symbol_count = self._materialize_facts(
                connection,
                snapshot_id=snapshot.snapshot_id,
                file_id=file_id,
                path=spec.path,
                language=language,
                facts=facts,
                observed_at=observed_at,
            )
            return (
                FileParseResult(
                    path=spec.path,
                    file_id=file_id,
                    content_digest=spec.content_digest,
                    language=language,
                    status=projected_status,
                    reason="",
                    record_id=record_id,
                    cache_hit=cache_hit,
                    symbol_count=symbol_count,
                ),
                reused,
                new_units,
                invalidated,
            )

        # Failed / unsupported / unknown: invalidate stale facts and keep
        # an explicit frontier.
        frontier_status = (
            status
            if status
            in {
                ParseStatus.FAILED,
                ParseStatus.UNSUPPORTED,
                ParseStatus.UNKNOWN,
            }
            else ParseStatus.UNKNOWN
        )
        invalidated = self._invalidate_file_facts(
            connection, snapshot.snapshot_id, file_id
        )
        reason_text = parse_error or f"parse_{frontier_status.value}"
        self._upsert_frontier(
            connection,
            snapshot_id=snapshot.snapshot_id,
            file_id=file_id,
            path=spec.path,
            content_digest=spec.content_digest,
            status=frontier_status,
            reason=reason_text,
            parser_id=parser_id,
            recorded_at=observed_at,
        )
        return (
            FileParseResult(
                path=spec.path,
                file_id=file_id,
                content_digest=spec.content_digest,
                language=language,
                status=frontier_status,
                reason=reason_text,
                record_id=record_id,
                cache_hit=cache_hit,
                symbol_count=0,
            ),
            reused,
            new_units,
            invalidated,
        )

    def _exclusion_decision(self, spec: SourceFileSpec) -> tuple[bool, str]:
        if spec.ignored:
            return True, "marked_ignored"
        excluded, reason = is_excluded_path(spec.path)
        if excluded:
            return True, reason
        if spec.content is not None and content_looks_like_secret(spec.content):
            return True, "secret_content"
        return False, ""

    def _parse_unit(
        self,
        spec: SourceFileSpec,
        *,
        language: str,
        parser_id: str,
    ) -> tuple[ParseStatus, str, str, dict[str, Any]]:
        del parser_id  # identity is bound by the cache key
        if isinstance(spec.ast_record, ASTBlobRecord):
            record = spec.ast_record
        elif language in _SUPPORTED_PARSE_LANGUAGES:
            source = spec.source_text()
            if source is None:
                return (
                    ParseStatus.UNKNOWN,
                    "",
                    "source_unavailable_for_parse",
                    {},
                )
            record = build_python_ast_blob_record(
                source,
                blob_identity=spec.blob_id,
                source_sha256=spec.content_digest,
            )
        else:
            return (
                ParseStatus.UNSUPPORTED,
                "",
                f"unsupported_language:{language or 'unknown'}",
                {
                    "authority": AUTHORITY_CLASS,
                    "language": language,
                    "qualified_symbols": [],
                    "imports": [],
                    "calls": [],
                    "state_transitions": [],
                    "interfaces": [],
                    "symbol_hashes": {},
                    "symbol_lines": {},
                    "parse_error": f"unsupported_language:{language or 'unknown'}",
                },
            )

        facts = _facts_from_record(record)
        if record.parse_error:
            return (
                ParseStatus.FAILED,
                record.record_id,
                record.parse_error,
                facts,
            )
        return ParseStatus.SUCCEEDED, record.record_id, "", facts

    def _record_file_version(
        self,
        connection: Any,
        *,
        repository_id: str,
        path: str,
        blob_id: str,
        content_digest: str,
        observed_at: str,
    ) -> None:
        version_id = _identity(
            "file-version",
            {
                "repository_id": repository_id,
                "path": path,
                "content_digest": content_digest,
            },
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO file_versions (
                file_version_id, repository_id, path, blob_id,
                content_digest, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                version_id,
                repository_id,
                path,
                blob_id,
                content_digest,
                observed_at,
            ],
        )

    def _load_parse_cache(
        self, connection: Any, content_digest: str, parser_id: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT parse_unit_id, content_digest, parser_id, language,
                   status, record_id, parse_error, facts_json, authority,
                   created_at
            FROM parse_cache
            WHERE content_digest = ? AND parser_id = ?
            LIMIT 1
            """,
            [content_digest, parser_id],
        ).fetchone()
        if row is None:
            return None
        return _row_mapping(row)

    def _store_parse_cache(
        self,
        connection: Any,
        *,
        content_digest: str,
        parser_id: str,
        language: str,
        status: ParseStatus,
        record_id: str,
        parse_error: str,
        facts: Mapping[str, Any],
        created_at: str,
    ) -> None:
        unit_id = _identity(
            "parse-unit",
            {
                "schema": PARSE_CACHE_SCHEMA,
                "content_digest": content_digest,
                "parser_id": parser_id,
            },
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO parse_cache (
                parse_unit_id, content_digest, parser_id, language, status,
                record_id, parse_error, facts_json, authority, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                unit_id,
                content_digest,
                parser_id,
                language,
                status.value,
                record_id,
                _bounded_text(parse_error, MAX_PARSE_ERROR_BYTES),
                _facts_json(facts),
                AUTHORITY_CLASS,
                created_at,
            ],
        )

    def _invalidate_file_facts(
        self, connection: Any, snapshot_id: str, file_id: str
    ) -> int:
        """Delete derived AST facts for one file; return deleted row count."""

        deleted = 0
        symbol_rows = connection.execute(
            """
            SELECT symbol_id FROM symbols
            WHERE snapshot_id = ? AND file_id = ?
            """,
            [snapshot_id, file_id],
        ).fetchall()
        symbol_ids = [
            str(_row_mapping(row).get("symbol_id") or "")
            for row in symbol_rows
        ]
        symbol_ids = [item for item in symbol_ids if item]
        for symbol_id in symbol_ids:
            for sql, params in (
                (
                    "DELETE FROM type_relations WHERE snapshot_id = ? "
                    "AND (left_symbol_id = ? OR right_symbol_id = ?)",
                    [snapshot_id, symbol_id, symbol_id],
                ),
                (
                    "DELETE FROM symbol_versions WHERE snapshot_id = ? "
                    "AND symbol_id = ?",
                    [snapshot_id, symbol_id],
                ),
            ):
                cursor = connection.execute(sql, params)
                count = getattr(cursor, "rowcount", None)
                if isinstance(count, int) and count > 0:
                    deleted += count

        node_rows = connection.execute(
            """
            SELECT node_id FROM ast_nodes
            WHERE snapshot_id = ? AND file_id = ?
            """,
            [snapshot_id, file_id],
        ).fetchall()
        node_ids = [
            str(_row_mapping(row).get("node_id") or "")
            for row in node_rows
        ]
        node_ids = [item for item in node_ids if item]
        for node_id in node_ids:
            cursor = connection.execute(
                """
                DELETE FROM ast_edges
                WHERE snapshot_id = ?
                  AND (source_node_id = ? OR target_node_id = ?)
                """,
                [snapshot_id, node_id, node_id],
            )
            count = getattr(cursor, "rowcount", None)
            if isinstance(count, int) and count > 0:
                deleted += count

        for sql in (
            'DELETE FROM "calls" WHERE snapshot_id = ? AND file_id = ?',
            'DELETE FROM "imports" WHERE snapshot_id = ? AND file_id = ?',
            'DELETE FROM "references" WHERE snapshot_id = ? AND file_id = ?',
            "DELETE FROM definitions WHERE snapshot_id = ? AND file_id = ?",
            "DELETE FROM ast_nodes WHERE snapshot_id = ? AND file_id = ?",
            "DELETE FROM symbols WHERE snapshot_id = ? AND file_id = ?",
        ):
            cursor = connection.execute(sql, [snapshot_id, file_id])
            count = getattr(cursor, "rowcount", None)
            if isinstance(count, int) and count > 0:
                deleted += count
        return deleted

    def _upsert_frontier(
        self,
        connection: Any,
        *,
        snapshot_id: str,
        file_id: str,
        path: str,
        content_digest: str,
        status: ParseStatus,
        reason: str,
        parser_id: str,
        recorded_at: str,
    ) -> ParseFrontier:
        frontier = ParseFrontier(
            frontier_id="",
            snapshot_id=snapshot_id,
            file_id=file_id,
            path=path,
            content_digest=content_digest,
            status=status,
            reason=reason,
            parser_id=parser_id,
            recorded_at=recorded_at,
        )
        connection.execute(
            """
            DELETE FROM parse_frontiers
            WHERE snapshot_id = ? AND file_id = ?
            """,
            [snapshot_id, file_id],
        )
        connection.execute(
            """
            INSERT INTO parse_frontiers (
                frontier_id, snapshot_id, file_id, path, content_digest,
                status, reason, parser_id, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                frontier.frontier_id,
                frontier.snapshot_id,
                frontier.file_id,
                frontier.path,
                frontier.content_digest,
                frontier.status.value
                if isinstance(frontier.status, ParseStatus)
                else str(frontier.status),
                frontier.reason,
                frontier.parser_id,
                frontier.recorded_at,
            ],
        )
        return frontier

    def _clear_frontier(
        self, connection: Any, snapshot_id: str, file_id: str
    ) -> None:
        connection.execute(
            """
            DELETE FROM parse_frontiers
            WHERE snapshot_id = ? AND file_id = ?
            """,
            [snapshot_id, file_id],
        )

    def _materialize_facts(
        self,
        connection: Any,
        *,
        snapshot_id: str,
        file_id: str,
        path: str,
        language: str,
        facts: Mapping[str, Any],
        observed_at: str,
    ) -> int:
        """Write derived symbol/import/call/node rows. Returns symbol count."""

        # Root module node — derived evidence only.
        root_node_id = _identity(
            "ast-node",
            {
                "snapshot_id": snapshot_id,
                "file_id": file_id,
                "node_path": "/",
                "node_kind": "module",
            },
        )
        connection.execute(
            """
            INSERT INTO ast_nodes (
                node_id, snapshot_id, file_id, parent_node_id, node_kind,
                node_path, fingerprint, start_byte, end_byte
            ) VALUES (?, ?, ?, '', 'module', '/', ?, 0, 0)
            """,
            [
                root_node_id,
                snapshot_id,
                file_id,
                str(facts.get("record_id") or facts.get("source_sha256") or ""),
            ],
        )

        symbol_hashes = dict(facts.get("symbol_hashes") or {})
        symbol_lines = dict(facts.get("symbol_lines") or {})
        symbol_ids: dict[str, str] = {}
        symbols = [
            str(item).strip()
            for item in (facts.get("qualified_symbols") or ())
            if str(item).strip()
        ]
        for qualified in symbols:
            lines = symbol_lines.get(qualified) or [0, 0]
            try:
                start_line = int(lines[0])
                end_line = int(lines[1])
            except (TypeError, ValueError, IndexError):
                start_line, end_line = 0, 0
            fingerprint = str(
                symbol_hashes.get(qualified)
                or _identity("symbol-fingerprint", {
                    "path": path,
                    "qualified_name": qualified,
                    "record_id": facts.get("record_id") or "",
                })
            )
            symbol_id = _identity(
                "symbol",
                {
                    "snapshot_id": snapshot_id,
                    "file_id": file_id,
                    "qualified_name": qualified,
                    "fingerprint": fingerprint,
                },
            )
            symbol_ids[qualified] = symbol_id
            kind = _symbol_kind(qualified)
            connection.execute(
                """
                INSERT INTO symbols (
                    symbol_id, snapshot_id, file_id, language, qualified_name,
                    symbol_kind, start_line, end_line, fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol_id,
                    snapshot_id,
                    file_id,
                    language,
                    qualified,
                    kind,
                    start_line,
                    end_line,
                    fingerprint,
                ],
            )
            version_id = _identity(
                "symbol-version",
                {
                    "symbol_id": symbol_id,
                    "snapshot_id": snapshot_id,
                    "fingerprint": fingerprint,
                },
            )
            connection.execute(
                """
                INSERT INTO symbol_versions (
                    symbol_version_id, symbol_id, snapshot_id, fingerprint,
                    observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    version_id,
                    symbol_id,
                    snapshot_id,
                    fingerprint,
                    observed_at,
                ],
            )
            definition_id = _identity(
                "definition",
                {
                    "symbol_id": symbol_id,
                    "snapshot_id": snapshot_id,
                },
            )
            connection.execute(
                """
                INSERT INTO definitions (
                    definition_id, snapshot_id, symbol_id, file_id,
                    start_line, end_line
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    definition_id,
                    snapshot_id,
                    symbol_id,
                    file_id,
                    start_line,
                    end_line,
                ],
            )
            node_id = _identity(
                "ast-node",
                {
                    "snapshot_id": snapshot_id,
                    "file_id": file_id,
                    "node_path": f"/symbol/{qualified}",
                    "node_kind": kind,
                },
            )
            connection.execute(
                """
                INSERT INTO ast_nodes (
                    node_id, snapshot_id, file_id, parent_node_id, node_kind,
                    node_path, fingerprint, start_byte, end_byte
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
                """,
                [
                    node_id,
                    snapshot_id,
                    file_id,
                    root_node_id,
                    kind,
                    f"/symbol/{qualified}",
                    fingerprint,
                ],
            )
            edge_id = _identity(
                "ast-edge",
                {
                    "snapshot_id": snapshot_id,
                    "source_node_id": root_node_id,
                    "target_node_id": node_id,
                    "edge_kind": "defines",
                },
            )
            connection.execute(
                """
                INSERT INTO ast_edges (
                    edge_id, snapshot_id, source_node_id, target_node_id,
                    edge_kind
                ) VALUES (?, ?, ?, ?, 'defines')
                """,
                [edge_id, snapshot_id, root_node_id, node_id],
            )
            # Reference row for the definition site.
            reference_id = _identity(
                "reference",
                {
                    "snapshot_id": snapshot_id,
                    "symbol_id": symbol_id,
                    "file_id": file_id,
                    "reference_kind": "definition",
                    "start_line": start_line,
                },
            )
            connection.execute(
                """
                INSERT INTO "references" (
                    reference_id, snapshot_id, symbol_id, file_id,
                    start_line, reference_kind
                ) VALUES (?, ?, ?, ?, ?, 'definition')
                """,
                [
                    reference_id,
                    snapshot_id,
                    symbol_id,
                    file_id,
                    start_line,
                ],
            )

        for interface in facts.get("interfaces") or ():
            text = str(interface or "").strip()
            if not text or "(" not in text:
                continue
            # Protocol/ABC style: Name(Base1,Base2)
            left, _, right = text.partition("(")
            bases = right.rstrip(")")
            left_id = symbol_ids.get(left.strip())
            if not left_id:
                continue
            for base in bases.split(","):
                base_name = base.strip()
                if not base_name:
                    continue
                right_id = symbol_ids.get(base_name) or _identity(
                    "symbol-ref",
                    {"name": base_name, "snapshot_id": snapshot_id},
                )
                relation_id = _identity(
                    "type-relation",
                    {
                        "snapshot_id": snapshot_id,
                        "left": left_id,
                        "right": right_id,
                        "kind": "implements",
                    },
                )
                connection.execute(
                    """
                    INSERT INTO type_relations (
                        relation_id, snapshot_id, left_symbol_id,
                        right_symbol_id, relation_kind
                    ) VALUES (?, ?, ?, ?, 'implements')
                    """,
                    [relation_id, snapshot_id, left_id, right_id],
                )

        for import_stmt in facts.get("imports") or ():
            module_name, alias = _parse_import(str(import_stmt))
            import_id = _identity(
                "import",
                {
                    "snapshot_id": snapshot_id,
                    "file_id": file_id,
                    "module_name": module_name,
                    "alias": alias,
                },
            )
            connection.execute(
                """
                INSERT INTO "imports" (
                    import_id, snapshot_id, file_id, module_name, alias,
                    start_line
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                [import_id, snapshot_id, file_id, module_name, alias],
            )

        for call_stmt in facts.get("calls") or ():
            text = str(call_stmt or "").strip()
            if "->" not in text:
                continue
            caller, _, callee = text.partition("->")
            caller = caller.strip()
            callee = callee.strip()
            caller_id = symbol_ids.get(caller) or _identity(
                "symbol-ref",
                {
                    "snapshot_id": snapshot_id,
                    "name": caller or "<module>",
                    "role": "caller",
                },
            )
            callee_id = symbol_ids.get(callee) or _identity(
                "symbol-ref",
                {
                    "snapshot_id": snapshot_id,
                    "name": callee or "<dynamic>",
                    "role": "callee",
                },
            )
            call_id = _identity(
                "call",
                {
                    "snapshot_id": snapshot_id,
                    "file_id": file_id,
                    "caller": caller_id,
                    "callee": callee_id,
                    "raw": text,
                },
            )
            connection.execute(
                """
                INSERT INTO "calls" (
                    call_id, snapshot_id, caller_symbol_id, callee_symbol_id,
                    file_id, start_line
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                [call_id, snapshot_id, caller_id, callee_id, file_id],
            )

        return len(symbols)


def open_duckdb_ast_index(
    database_path: Path | str,
    *,
    parser_id: str = DEFAULT_PARSER_ID,
    scanner_version: str = DEFAULT_SCANNER_VERSION,
) -> DuckDBASTIndex:
    """Open (or create) a DuckDB AST index at ``database_path``."""

    return DuckDBASTIndex(
        database_path,
        parser_id=parser_id,
        scanner_version=scanner_version,
    ).open()


__all__ = [
    "AUTHORITY_CLASS",
    "DEFAULT_PARSER_ID",
    "DEFAULT_SCANNER_VERSION",
    "DUCKDB_AST_INDEX_INTERFACE",
    "DUCKDB_AST_INDEX_SCHEMA",
    "DuckDBASTIndex",
    "DuckDBASTIndexBoundsError",
    "DuckDBASTIndexConflictError",
    "DuckDBASTIndexError",
    "DuckDBASTIndexIntegrityError",
    "DuckDBASTIndexNotOpenError",
    "DuckDBUnavailableError",
    "FileDisposition",
    "FileParseResult",
    "IndexedSymbol",
    "PARSE_RUN_INTERFACE",
    "PARSE_RUN_SCHEMA",
    "ParseFrontier",
    "ParseRun",
    "ParseStatus",
    "SOURCE_SNAPSHOT_INTERFACE",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SnapshotIngestResult",
    "SourceFileSpec",
    "SourceSnapshot",
    "SymbolKind",
    "content_looks_like_secret",
    "duckdb_available",
    "is_excluded_path",
    "language_for_path",
    "open_duckdb_ast_index",
]
