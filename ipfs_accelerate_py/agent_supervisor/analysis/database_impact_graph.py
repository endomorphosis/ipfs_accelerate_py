"""DuckDB-backed symbol dependency, impact, and neighborhood queries.

DQP-023 / Interfaces: ``DatabaseImpactGraph@1``, ``ImpactClosure@1``,
``ChangedSymbolNeighborhood@1``
============================================================================

Materializes versioned, bounded SQL views over snapshot-bound AST symbol
graphs so symbolic planning can query callers, callees, imports, types,
tests, contracts, proofs, config, docs, and unresolved dynamic frontiers
for a mutation or task without treating graph proximity as semantic
authority.

Acceptance properties
---------------------
* Every resolved consumer receives exactly one disposition.
* An open or unsupported frontier blocks automatic repair
  (``blocks_automatic_repair``).
* Query results bind snapshot, parser, policy, and schema identities.
* Similarity and graph proximity remain nomination rather than semantic
  authority (``authority`` is always ``derived_evidence`` / nomination).

Evidence subset covered by queries and tests: recursion, SCC, aliases,
reexports, dynamic calls, generated code, cross-language, deletion,
parser uncertainty, and pagination.

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a graph is the first I/O boundary.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_IMPACT_GRAPH_INTERFACE: Final[str] = "DatabaseImpactGraph@1"
IMPACT_CLOSURE_INTERFACE: Final[str] = "ImpactClosure@1"
CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE: Final[str] = (
    "ChangedSymbolNeighborhood@1"
)

DATABASE_IMPACT_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-impact-graph@1"
)
IMPACT_CLOSURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-closure@1"
)
CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/changed-symbol-neighborhood@1"
)
IMPACT_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-edge@1"
)
IMPACT_CONSUMER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-consumer-disposition@1"
)
IMPACT_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-query-frontier@1"
)
IMPACT_SCC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-query-scc@1"
)

DEFAULT_POLICY_ID: Final[str] = "database-impact-policy@1"
DEFAULT_GRAPH_VERSION: Final[str] = "database-impact-graph@1"
AUTHORITY_CLASS: Final[str] = "derived_evidence"
NOMINATION_AUTHORITY: Final[str] = "nomination_only"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_SEEDS: Final[int] = 4_096
MAX_CONSUMERS: Final[int] = 16_384
MAX_EDGES: Final[int] = 65_536
MAX_DEPTH: Final[int] = 256
MAX_SCCS: Final[int] = 1_024
MAX_PAGE_SIZE: Final[int] = 1_024
DEFAULT_PAGE_SIZE: Final[int] = 100
HARD_MAX_DEPTH: Final[int] = 4_096

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS impact_graph_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS impact_graph_revisions (
    revision_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    schema_id VARCHAR NOT NULL,
    materialization_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    edge_count BIGINT NOT NULL DEFAULT 0,
    symbol_count BIGINT NOT NULL DEFAULT 0,
    frontier_count BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_graph_revisions_snapshot_idx
    ON impact_graph_revisions(snapshot_id, created_at);

CREATE TABLE IF NOT EXISTS impact_symbols (
    symbol_key VARCHAR PRIMARY KEY,
    revision_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL DEFAULT '',
    qualified_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    language VARCHAR NOT NULL DEFAULT '',
    symbol_kind VARCHAR NOT NULL DEFAULT '',
    is_generated INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    is_test INTEGER NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS impact_symbols_revision_name_idx
    ON impact_symbols(revision_id, qualified_name);
CREATE INDEX IF NOT EXISTS impact_symbols_revision_path_idx
    ON impact_symbols(revision_id, path);

CREATE TABLE IF NOT EXISTS impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    revision_id VARCHAR NOT NULL,
    source_symbol VARCHAR NOT NULL,
    target_symbol VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'derived_evidence',
    path VARCHAR NOT NULL DEFAULT '',
    is_dynamic INTEGER NOT NULL DEFAULT 0,
    is_generated INTEGER NOT NULL DEFAULT 0,
    is_cross_language INTEGER NOT NULL DEFAULT 0,
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS impact_edges_revision_source_idx
    ON impact_edges(revision_id, source_symbol, edge_kind);
CREATE INDEX IF NOT EXISTS impact_edges_revision_target_idx
    ON impact_edges(revision_id, target_symbol, edge_kind);
CREATE INDEX IF NOT EXISTS impact_edges_revision_kind_idx
    ON impact_edges(revision_id, edge_kind);

CREATE TABLE IF NOT EXISTS impact_frontiers (
    frontier_id VARCHAR PRIMARY KEY,
    revision_id VARCHAR NOT NULL,
    symbol_key VARCHAR NOT NULL DEFAULT '',
    path VARCHAR NOT NULL DEFAULT '',
    kind VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    blocks_repair INTEGER NOT NULL DEFAULT 1,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS impact_frontiers_revision_idx
    ON impact_frontiers(revision_id, kind);

CREATE TABLE IF NOT EXISTS impact_query_receipts (
    query_id VARCHAR PRIMARY KEY,
    revision_id VARCHAR NOT NULL,
    query_kind VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    schema_id VARCHAR NOT NULL,
    seed_json VARCHAR NOT NULL,
    completeness VARCHAR NOT NULL,
    blocks_automatic_repair INTEGER NOT NULL DEFAULT 0,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_query_receipts_revision_idx
    ON impact_query_receipts(revision_id, query_kind);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseImpactGraphError(RuntimeError):
    """Base error for database impact graph failures."""


class DatabaseImpactGraphNotOpenError(DatabaseImpactGraphError):
    """Operation requires an open impact graph."""


class DatabaseImpactGraphIntegrityError(DatabaseImpactGraphError, ValueError):
    """Identity, path, or payload integrity failure."""


class DatabaseImpactGraphBoundsError(DatabaseImpactGraphError, ValueError):
    """A resource or payload bound was exceeded."""


class DatabaseImpactGraphConflictError(DatabaseImpactGraphError):
    """Duplicate identity with a conflicting payload."""


class DuckDBUnavailableError(DatabaseImpactGraphError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EdgeKind(str, Enum):
    """Closed vocabulary of impact graph edge kinds.

    Orientation is *dependent → provider* (source depends on target) for
    CALLS, IMPORTS, TYPES, ALIASES, REEXPORTS, TESTS, CONTRACTS, PROOFS,
    CONFIG, DOCS. Reverse impact walks incoming edges of those kinds.
    """

    CALLS = "calls"
    IMPORTS = "imports"
    TYPES = "types"
    ALIASES = "aliases"
    REEXPORTS = "reexports"
    TESTS = "tests"
    CONTRACTS = "contracts"
    PROOFS = "proofs"
    CONFIG = "config"
    DOCS = "docs"
    DYNAMIC = "dynamic"
    GENERATED_FROM = "generated_from"
    REFERENCES = "references"
    NOMINATED = "nominated"  # similarity / proximity; never authority

    @classmethod
    def coerce(cls, value: Any) -> "EdgeKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, EdgeKind] = {
            "call": cls.CALLS,
            "calls": cls.CALLS,
            "caller": cls.CALLS,
            "callee": cls.CALLS,
            "import": cls.IMPORTS,
            "imports": cls.IMPORTS,
            "type": cls.TYPES,
            "types": cls.TYPES,
            "implements": cls.TYPES,
            "alias": cls.ALIASES,
            "aliases": cls.ALIASES,
            "reexport": cls.REEXPORTS,
            "reexports": cls.REEXPORTS,
            "re_export": cls.REEXPORTS,
            "re_exports": cls.REEXPORTS,
            "test": cls.TESTS,
            "tests": cls.TESTS,
            "contract": cls.CONTRACTS,
            "contracts": cls.CONTRACTS,
            "proof": cls.PROOFS,
            "proofs": cls.PROOFS,
            "config": cls.CONFIG,
            "docs": cls.DOCS,
            "doc": cls.DOCS,
            "documents": cls.DOCS,
            "dynamic": cls.DYNAMIC,
            "dynamic_call": cls.DYNAMIC,
            "generated": cls.GENERATED_FROM,
            "generated_from": cls.GENERATED_FROM,
            "reference": cls.REFERENCES,
            "references": cls.REFERENCES,
            "nominated": cls.NOMINATED,
            "similarity": cls.NOMINATED,
            "proximity": cls.NOMINATED,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseImpactGraphIntegrityError(
                f"unsupported edge kind: {value!r}"
            ) from exc


# Edge kinds that expand mandatory reverse impact (consumer → provider).
_MANDATORY_EDGE_KINDS: Final[frozenset[EdgeKind]] = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.IMPORTS,
        EdgeKind.TYPES,
        EdgeKind.ALIASES,
        EdgeKind.REEXPORTS,
        EdgeKind.TESTS,
        EdgeKind.CONTRACTS,
        EdgeKind.PROOFS,
        EdgeKind.CONFIG,
        EdgeKind.DOCS,
        EdgeKind.GENERATED_FROM,
        EdgeKind.REFERENCES,
    }
)

# Edge kinds that are nomination-only and never expand mandatory closure.
_NOMINATION_EDGE_KINDS: Final[frozenset[EdgeKind]] = frozenset(
    {
        EdgeKind.NOMINATED,
    }
)


class ConsumerDisposition(str, Enum):
    """Exactly one disposition per resolved consumer."""

    MUST_REVALIDATE = "must_revalidate"
    MUST_REPAIR = "must_repair"
    REVIEW = "review"
    NOMINATED = "nominated"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
    GENERATED = "generated"
    CROSS_LANGUAGE = "cross_language"
    PARSER_UNCERTAIN = "parser_uncertain"
    OPEN_FRONTIER = "open_frontier"
    UNSUPPORTED = "unsupported"


class FrontierKind(str, Enum):
    DYNAMIC_CALL = "dynamic_call"
    GENERATED_CODE = "generated_code"
    CROSS_LANGUAGE = "cross_language"
    DELETION = "deletion"
    PARSER_UNCERTAINTY = "parser_uncertainty"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNRESOLVED_SYMBOL = "unresolved_symbol"
    UNBOUNDED = "unbounded"


class FrontierDisposition(str, Enum):
    OPEN = "open"
    UNSUPPORTED = "unsupported"
    CLOSED_OBSERVED = "closed_observed"
    NOMINATED_ONLY = "nominated_only"
    UNKNOWN = "unknown"


class ImpactCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL_WITH_FRONTIER = "partial_with_frontier"
    ABSTAINED = "abstained"
    TRUNCATED = "truncated"


class QueryKind(str, Enum):
    IMPACT_CLOSURE = "impact_closure"
    CHANGED_NEIGHBORHOOD = "changed_neighborhood"
    CALLERS = "callers"
    CALLEES = "callees"
    IMPORTS = "imports"
    TYPES = "types"
    TESTS = "tests"
    CONTRACTS = "contracts"
    PROOFS = "proofs"
    CONFIG = "config"
    DOCS = "docs"


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
        raise DatabaseImpactGraphIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseImpactGraphIntegrityError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseImpactGraphBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatabaseImpactGraphBoundsError(
            f"{name} must be a positive integer"
        )
    if maximum is not None and value > maximum:
        raise DatabaseImpactGraphBoundsError(
            f"{name} exceeds maximum {maximum}"
        )
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
        raise DatabaseImpactGraphIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _bounded_text(value: Any, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= maximum:
        return text
    marker = "…[truncated]"
    budget = max(0, maximum - len(marker.encode("utf-8")))
    return encoded[:budget].decode("utf-8", "ignore") + marker


def _repo_path(value: Any, *, required: bool = False) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        if required:
            raise DatabaseImpactGraphIntegrityError(
                "repository path is required"
            )
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise DatabaseImpactGraphIntegrityError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise DatabaseImpactGraphBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes: {normalized}"
        )
    return normalized


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


def _is_test_path(path: str) -> bool:
    lowered = path.casefold()
    parts = PurePosixPath(lowered).parts
    if any(
        part in {"test", "tests", "testing", "fixtures", "mocks"}
        for part in parts
    ):
        return True
    name = PurePosixPath(lowered).name
    return name.startswith("test_") or name.endswith("_test.py")


def _is_generated_path(path: str) -> bool:
    lowered = path.casefold()
    parts = PurePosixPath(lowered).parts
    return any(
        part in {"generated", "gen", "_generated", "build", "dist"}
        for part in parts
    )


def _symbol_key(
    *,
    qualified_name: str,
    path: str = "",
    symbol_id: str = "",
) -> str:
    name = _text(qualified_name, "qualified_name")
    if symbol_id:
        return _identity(
            "impact-symbol",
            {"symbol_id": symbol_id, "qualified_name": name, "path": path},
        )
    return _identity(
        "impact-symbol",
        {"qualified_name": name, "path": path},
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactEdgeSpec:
    """One directed dependent→provider edge to materialize."""

    source_symbol: str
    target_symbol: str
    edge_kind: EdgeKind | str
    path: str = ""
    source_path: str = ""
    target_path: str = ""
    source_language: str = ""
    target_language: str = ""
    is_dynamic: bool = False
    is_generated: bool = False
    is_cross_language: bool = False
    authority: str = AUTHORITY_CLASS
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_symbol", _text(self.source_symbol, "source_symbol")
        )
        object.__setattr__(
            self, "target_symbol", _text(self.target_symbol, "target_symbol")
        )
        object.__setattr__(self, "edge_kind", EdgeKind.coerce(self.edge_kind))
        object.__setattr__(
            self, "path", _repo_path(self.path, required=False)
        )
        object.__setattr__(
            self, "source_path", _repo_path(self.source_path, required=False)
        )
        object.__setattr__(
            self, "target_path", _repo_path(self.target_path, required=False)
        )
        object.__setattr__(
            self,
            "source_language",
            _text(self.source_language, "source_language", required=False),
        )
        object.__setattr__(
            self,
            "target_language",
            _text(self.target_language, "target_language", required=False),
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_TEXT_BYTES)
        )
        authority = str(self.authority or AUTHORITY_CLASS).strip()
        if self.edge_kind is EdgeKind.NOMINATED:
            authority = NOMINATION_AUTHORITY
        object.__setattr__(self, "authority", authority)
        meta = dict(self.metadata or {})
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        src_lang = self.source_language.casefold()
        tgt_lang = self.target_language.casefold()
        if (
            src_lang
            and tgt_lang
            and src_lang != tgt_lang
            and not self.is_cross_language
        ):
            object.__setattr__(self, "is_cross_language", True)


@dataclass(frozen=True)
class ImpactSymbolSpec:
    """Optional symbol metadata attached during materialization."""

    qualified_name: str
    path: str = ""
    language: str = ""
    symbol_kind: str = ""
    symbol_id: str = ""
    is_generated: bool = False
    is_deleted: bool = False
    is_test: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "qualified_name", _text(self.qualified_name, "qualified_name")
        )
        path = _repo_path(self.path, required=False)
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        object.__setattr__(
            self,
            "symbol_kind",
            _text(self.symbol_kind, "symbol_kind", required=False),
        )
        object.__setattr__(
            self, "symbol_id", _text(self.symbol_id, "symbol_id", required=False)
        )
        if path and not self.is_test and _is_test_path(path):
            object.__setattr__(self, "is_test", True)
        if path and not self.is_generated and _is_generated_path(path):
            object.__setattr__(self, "is_generated", True)


@dataclass(frozen=True)
class ImpactFrontierSpec:
    """Explicit open/unsupported frontier endpoint."""

    kind: FrontierKind | str
    disposition: FrontierDisposition | str = FrontierDisposition.OPEN
    symbol_key: str = ""
    path: str = ""
    reason: str = ""
    blocks_repair: bool = True

    def __post_init__(self) -> None:
        kind = self.kind
        if not isinstance(kind, FrontierKind):
            kind = FrontierKind(str(kind).strip().casefold())
        object.__setattr__(self, "kind", kind)
        disposition = self.disposition
        if not isinstance(disposition, FrontierDisposition):
            disposition = FrontierDisposition(
                str(disposition).strip().casefold()
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "symbol_key", _text(self.symbol_key, "symbol_key", required=False)
        )
        object.__setattr__(
            self, "path", _repo_path(self.path, required=False)
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_TEXT_BYTES)
        )
        if disposition in {
            FrontierDisposition.OPEN,
            FrontierDisposition.UNSUPPORTED,
            FrontierDisposition.UNKNOWN,
        }:
            object.__setattr__(self, "blocks_repair", True)


@dataclass(frozen=True)
class ImpactGraphRevision:
    """One materialization bound to snapshot/parser/policy/schema."""

    revision_id: str
    snapshot_id: str
    parser_id: str
    policy_id: str
    schema_id: str
    materialization_id: str
    repository_id: str = ""
    tree_id: str = ""
    created_at: str = ""
    edge_count: int = 0
    symbol_count: int = 0
    frontier_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "parser_id", _text(self.parser_id, "parser_id")
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self, "schema_id", _text(self.schema_id, "schema_id")
        )
        object.__setattr__(
            self,
            "materialization_id",
            _text(self.materialization_id, "materialization_id"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _text(self.repository_id, "repository_id", required=False),
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        for name in ("edge_count", "symbol_count", "frontier_count"):
            object.__setattr__(
                self, name, _nonneg_int(int(getattr(self, name)), name)
            )
        claimed = str(self.revision_id or "").strip()
        computed = _identity(
            "impact-revision",
            {
                "schema": DATABASE_IMPACT_GRAPH_SCHEMA,
                "snapshot_id": self.snapshot_id,
                "parser_id": self.parser_id,
                "policy_id": self.policy_id,
                "schema_id": self.schema_id,
                "materialization_id": self.materialization_id,
            },
        )
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact revision identity does not match payload"
            )
        object.__setattr__(self, "revision_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "materialization_id": self.materialization_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "created_at": self.created_at,
            "edge_count": self.edge_count,
            "symbol_count": self.symbol_count,
            "frontier_count": self.frontier_count,
            "authority": AUTHORITY_CLASS,
            "interface": DATABASE_IMPACT_GRAPH_INTERFACE,
            "schema": DATABASE_IMPACT_GRAPH_SCHEMA,
        }


@dataclass(frozen=True)
class ImpactConsumerRecord:
    """One resolved consumer with exactly one disposition."""

    consumer_id: str
    symbol: str
    disposition: ConsumerDisposition | str
    depth: int
    path: str = ""
    language: str = ""
    edge_kinds: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    mandatory: bool = True
    via_symbol: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "symbol", _text(self.symbol, "symbol")
        )
        disposition = self.disposition
        if not isinstance(disposition, ConsumerDisposition):
            disposition = ConsumerDisposition(str(disposition).strip().casefold())
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "depth", _nonneg_int(int(self.depth), "depth")
        )
        object.__setattr__(
            self, "path", _repo_path(self.path, required=False)
        )
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        kinds = tuple(str(item) for item in self.edge_kinds if str(item))
        object.__setattr__(self, "edge_kinds", kinds)
        edges = tuple(str(item) for item in self.edge_ids if str(item))
        object.__setattr__(self, "edge_ids", edges)
        object.__setattr__(
            self, "via_symbol", _text(self.via_symbol, "via_symbol", required=False)
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_TEXT_BYTES)
        )
        claimed = str(self.consumer_id or "").strip()
        computed = _identity(
            "impact-consumer",
            {
                "schema": IMPACT_CONSUMER_SCHEMA,
                "symbol": self.symbol,
                "disposition": disposition.value,
                "depth": self.depth,
                "path": self.path,
            },
        )
        if claimed and claimed != computed:
            # Allow precomputed ids that may include edge path digests.
            pass
        object.__setattr__(self, "consumer_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPACT_CONSUMER_SCHEMA,
            "consumer_id": self.consumer_id,
            "symbol": self.symbol,
            "disposition": self.disposition.value
            if isinstance(self.disposition, ConsumerDisposition)
            else str(self.disposition),
            "depth": self.depth,
            "path": self.path,
            "language": self.language,
            "edge_kinds": list(self.edge_kinds),
            "edge_ids": list(self.edge_ids),
            "mandatory": bool(self.mandatory),
            "via_symbol": self.via_symbol,
            "reason": self.reason,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactSCCRecord:
    """Strongly connected consumer group (Tarjan condensation unit)."""

    scc_id: str
    member_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        members = tuple(
            sorted({_text(item, "member") for item in self.member_symbols if str(item).strip()})
        )
        if not members:
            raise DatabaseImpactGraphIntegrityError(
                "impact scc requires at least one member"
            )
        if len(members) > MAX_CONSUMERS:
            raise DatabaseImpactGraphBoundsError(
                f"scc exceeds {MAX_CONSUMERS} members"
            )
        object.__setattr__(self, "member_symbols", members)
        claimed = str(self.scc_id or "").strip()
        computed = _identity(
            "impact-scc",
            {"schema": IMPACT_SCC_SCHEMA, "members": list(members)},
        )
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact scc identity does not match payload"
            )
        object.__setattr__(self, "scc_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPACT_SCC_SCHEMA,
            "scc_id": self.scc_id,
            "member_symbols": list(self.member_symbols),
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactFrontierRecord:
    """Persisted frontier endpoint attached to a query or revision."""

    frontier_id: str
    kind: FrontierKind | str
    disposition: FrontierDisposition | str
    symbol_key: str = ""
    path: str = ""
    reason: str = ""
    blocks_repair: bool = True

    def __post_init__(self) -> None:
        kind = self.kind
        if not isinstance(kind, FrontierKind):
            kind = FrontierKind(str(kind).strip().casefold())
        object.__setattr__(self, "kind", kind)
        disposition = self.disposition
        if not isinstance(disposition, FrontierDisposition):
            disposition = FrontierDisposition(
                str(disposition).strip().casefold()
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "symbol_key", _text(self.symbol_key, "symbol_key", required=False)
        )
        object.__setattr__(
            self, "path", _repo_path(self.path, required=False)
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_TEXT_BYTES)
        )
        claimed = str(self.frontier_id or "").strip()
        computed = _identity(
            "impact-frontier",
            {
                "schema": IMPACT_FRONTIER_SCHEMA,
                "kind": kind.value,
                "disposition": disposition.value,
                "symbol_key": self.symbol_key,
                "path": self.path,
                "reason": self.reason,
            },
        )
        object.__setattr__(self, "frontier_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPACT_FRONTIER_SCHEMA,
            "frontier_id": self.frontier_id,
            "kind": self.kind.value
            if isinstance(self.kind, FrontierKind)
            else str(self.kind),
            "disposition": self.disposition.value
            if isinstance(self.disposition, FrontierDisposition)
            else str(self.disposition),
            "symbol_key": self.symbol_key,
            "path": self.path,
            "reason": self.reason,
            "blocks_repair": bool(self.blocks_repair),
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactClosure:
    """Reverse transitive impact closure for seed symbols.

    Interface: ``ImpactClosure@1``.
    """

    query_id: str
    revision_id: str
    snapshot_id: str
    parser_id: str
    policy_id: str
    schema_id: str
    seed_symbols: tuple[str, ...]
    completeness: ImpactCompleteness | str
    consumers: tuple[ImpactConsumerRecord, ...]
    sccs: tuple[ImpactSCCRecord, ...] = ()
    frontiers: tuple[ImpactFrontierRecord, ...] = ()
    nominated: tuple[ImpactConsumerRecord, ...] = ()
    blocks_automatic_repair: bool = False
    truncated: bool = False
    max_depth: int = 0
    edge_count: int = 0
    created_at: str = ""
    schema: str = IMPACT_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision_id", _text(self.revision_id, "revision_id")
        )
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "parser_id", _text(self.parser_id, "parser_id")
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self, "schema_id", _text(self.schema_id, "schema_id")
        )
        seeds = tuple(
            dict.fromkeys(
                _text(item, "seed") for item in self.seed_symbols if str(item).strip()
            )
        )
        if not seeds:
            raise DatabaseImpactGraphIntegrityError(
                "impact closure requires at least one seed symbol"
            )
        if len(seeds) > MAX_SEEDS:
            raise DatabaseImpactGraphBoundsError(
                f"seed count exceeds {MAX_SEEDS}"
            )
        object.__setattr__(self, "seed_symbols", seeds)
        completeness = self.completeness
        if not isinstance(completeness, ImpactCompleteness):
            completeness = ImpactCompleteness(
                str(completeness).strip().casefold()
            )
        object.__setattr__(self, "completeness", completeness)
        consumers = tuple(self.consumers)
        if len(consumers) > MAX_CONSUMERS:
            raise DatabaseImpactGraphBoundsError(
                f"consumer count exceeds {MAX_CONSUMERS}"
            )
        # Exactly one disposition per resolved consumer symbol.
        seen: dict[str, ConsumerDisposition] = {}
        for consumer in consumers:
            if not isinstance(consumer, ImpactConsumerRecord):
                raise DatabaseImpactGraphIntegrityError(
                    "consumers must be ImpactConsumerRecord"
                )
            prior = seen.get(consumer.symbol)
            if prior is not None and prior is not consumer.disposition:
                raise DatabaseImpactGraphIntegrityError(
                    f"consumer {consumer.symbol!r} has multiple dispositions"
                )
            seen[consumer.symbol] = (
                consumer.disposition
                if isinstance(consumer.disposition, ConsumerDisposition)
                else ConsumerDisposition(str(consumer.disposition))
            )
        object.__setattr__(self, "consumers", consumers)
        sccs = tuple(self.sccs)
        if len(sccs) > MAX_SCCS:
            raise DatabaseImpactGraphBoundsError(
                f"scc count exceeds {MAX_SCCS}"
            )
        object.__setattr__(self, "sccs", sccs)
        frontiers = tuple(self.frontiers)
        object.__setattr__(self, "frontiers", frontiers)
        nominated = tuple(self.nominated)
        object.__setattr__(self, "nominated", nominated)
        blocking = bool(self.blocks_automatic_repair) or any(
            item.blocks_repair for item in frontiers
        )
        if completeness is ImpactCompleteness.COMPLETE and blocking:
            raise DatabaseImpactGraphIntegrityError(
                "complete impact closure cannot carry a blocking frontier"
            )
        if completeness is ImpactCompleteness.COMPLETE and self.truncated:
            raise DatabaseImpactGraphIntegrityError(
                "complete impact closure cannot be truncated"
            )
        object.__setattr__(self, "blocks_automatic_repair", blocking)
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        if self.schema != IMPACT_CLOSURE_SCHEMA:
            raise DatabaseImpactGraphIntegrityError(
                "unsupported impact closure schema"
            )
        claimed = str(self.query_id or "").strip()
        computed = _identity(
            "impact-closure",
            {
                "schema": self.schema,
                "revision_id": self.revision_id,
                "snapshot_id": self.snapshot_id,
                "parser_id": self.parser_id,
                "policy_id": self.policy_id,
                "schema_id": self.schema_id,
                "seed_symbols": list(seeds),
                "completeness": completeness.value,
                "consumers": [item.to_dict() for item in consumers],
                "frontiers": [item.to_dict() for item in frontiers],
            },
        )
        object.__setattr__(self, "query_id", claimed or computed)

    @property
    def interface(self) -> str:
        return IMPACT_CLOSURE_INTERFACE

    @property
    def freshness(self) -> dict[str, str]:
        return {
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "revision_id": self.revision_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": IMPACT_CLOSURE_INTERFACE,
            "query_id": self.query_id,
            "revision_id": self.revision_id,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "seed_symbols": list(self.seed_symbols),
            "completeness": self.completeness.value
            if isinstance(self.completeness, ImpactCompleteness)
            else str(self.completeness),
            "consumers": [item.to_dict() for item in self.consumers],
            "sccs": [item.to_dict() for item in self.sccs],
            "frontiers": [item.to_dict() for item in self.frontiers],
            "nominated": [item.to_dict() for item in self.nominated],
            "blocks_automatic_repair": bool(self.blocks_automatic_repair),
            "truncated": bool(self.truncated),
            "max_depth": int(self.max_depth),
            "edge_count": int(self.edge_count),
            "created_at": self.created_at,
            "freshness": self.freshness,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ChangedSymbolNeighborhood:
    """Bounded neighborhood around changed symbols.

    Interface: ``ChangedSymbolNeighborhood@1``.
    """

    query_id: str
    revision_id: str
    snapshot_id: str
    parser_id: str
    policy_id: str
    schema_id: str
    changed_symbols: tuple[str, ...]
    radius: int
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    callers: tuple[str, ...] = ()
    callees: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    contracts: tuple[str, ...] = ()
    proofs: tuple[str, ...] = ()
    config: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    frontiers: tuple[ImpactFrontierRecord, ...] = ()
    page_offset: int = 0
    page_limit: int = DEFAULT_PAGE_SIZE
    total_edge_count: int = 0
    has_more: bool = False
    created_at: str = ""
    schema: str = CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision_id", _text(self.revision_id, "revision_id")
        )
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "parser_id", _text(self.parser_id, "parser_id")
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self, "schema_id", _text(self.schema_id, "schema_id")
        )
        changed = tuple(
            dict.fromkeys(
                _text(item, "changed_symbol")
                for item in self.changed_symbols
                if str(item).strip()
            )
        )
        if not changed:
            raise DatabaseImpactGraphIntegrityError(
                "changed neighborhood requires at least one changed symbol"
            )
        object.__setattr__(self, "changed_symbols", changed)
        object.__setattr__(
            self, "radius", _nonneg_int(int(self.radius), "radius")
        )
        object.__setattr__(self, "nodes", tuple(dict(item) for item in self.nodes))
        object.__setattr__(self, "edges", tuple(dict(item) for item in self.edges))
        for name in (
            "callers",
            "callees",
            "imports",
            "types",
            "tests",
            "contracts",
            "proofs",
            "config",
            "docs",
        ):
            values = tuple(
                dict.fromkeys(
                    str(item) for item in getattr(self, name) if str(item).strip()
                )
            )
            object.__setattr__(self, name, values)
        object.__setattr__(self, "frontiers", tuple(self.frontiers))
        object.__setattr__(
            self, "page_offset", _nonneg_int(int(self.page_offset), "page_offset")
        )
        object.__setattr__(
            self,
            "page_limit",
            _positive_int(int(self.page_limit), "page_limit", maximum=MAX_PAGE_SIZE),
        )
        object.__setattr__(
            self,
            "total_edge_count",
            _nonneg_int(int(self.total_edge_count), "total_edge_count"),
        )
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        if self.schema != CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA:
            raise DatabaseImpactGraphIntegrityError(
                "unsupported changed neighborhood schema"
            )
        claimed = str(self.query_id or "").strip()
        computed = _identity(
            "changed-neighborhood",
            {
                "schema": self.schema,
                "revision_id": self.revision_id,
                "snapshot_id": self.snapshot_id,
                "changed_symbols": list(changed),
                "radius": self.radius,
                "page_offset": self.page_offset,
                "page_limit": self.page_limit,
            },
        )
        object.__setattr__(self, "query_id", claimed or computed)

    @property
    def interface(self) -> str:
        return CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE

    @property
    def freshness(self) -> dict[str, str]:
        return {
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "revision_id": self.revision_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE,
            "query_id": self.query_id,
            "revision_id": self.revision_id,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "changed_symbols": list(self.changed_symbols),
            "radius": self.radius,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "callers": list(self.callers),
            "callees": list(self.callees),
            "imports": list(self.imports),
            "types": list(self.types),
            "tests": list(self.tests),
            "contracts": list(self.contracts),
            "proofs": list(self.proofs),
            "config": list(self.config),
            "docs": list(self.docs),
            "frontiers": [item.to_dict() for item in self.frontiers],
            "page_offset": self.page_offset,
            "page_limit": self.page_limit,
            "total_edge_count": self.total_edge_count,
            "has_more": bool(self.has_more),
            "created_at": self.created_at,
            "freshness": self.freshness,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MaterializationResult:
    """Outcome of materializing one impact graph revision."""

    revision: ImpactGraphRevision
    edge_count: int
    symbol_count: int
    frontier_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision.to_dict(),
            "edge_count": self.edge_count,
            "symbol_count": self.symbol_count,
            "frontier_count": self.frontier_count,
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _tarjan_sccs(
    nodes: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> list[tuple[str, ...]]:
    """Tarjan SCC over *nodes* with sorted adjacency for determinism."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[tuple[str, ...]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adjacency.get(v, ())):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1 or (
                component
                and component[0] in adjacency.get(component[0], ())
            ):
                result.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    result.sort(key=lambda members: (members[0], len(members), members))
    return result


def _disposition_for_symbol(
    *,
    symbol: str,
    path: str,
    language: str,
    is_deleted: bool,
    is_generated: bool,
    is_cross_language: bool,
    edge_kinds: Sequence[str],
    parser_uncertain: bool,
    is_test: bool,
) -> ConsumerDisposition:
    if is_deleted:
        return ConsumerDisposition.DELETED
    if parser_uncertain:
        return ConsumerDisposition.PARSER_UNCERTAIN
    if is_generated or _is_generated_path(path):
        return ConsumerDisposition.GENERATED
    if is_cross_language:
        return ConsumerDisposition.CROSS_LANGUAGE
    kinds = {str(item) for item in edge_kinds}
    if EdgeKind.NOMINATED.value in kinds and kinds <= {
        EdgeKind.NOMINATED.value
    }:
        return ConsumerDisposition.NOMINATED
    if is_test or EdgeKind.TESTS.value in kinds or _is_test_path(path):
        return ConsumerDisposition.MUST_REVALIDATE
    if kinds & {
        EdgeKind.CALLS.value,
        EdgeKind.IMPORTS.value,
        EdgeKind.ALIASES.value,
        EdgeKind.REEXPORTS.value,
        EdgeKind.TYPES.value,
        EdgeKind.CONTRACTS.value,
        EdgeKind.PROOFS.value,
        EdgeKind.CONFIG.value,
        EdgeKind.REFERENCES.value,
    }:
        return ConsumerDisposition.MUST_REPAIR
    if EdgeKind.DOCS.value in kinds:
        return ConsumerDisposition.REVIEW
    del symbol, language
    return ConsumerDisposition.MUST_REVALIDATE


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseImpactGraph:
    """Persist and query symbol dependency impact evidence in DuckDB.

    Interface: ``DatabaseImpactGraph@1``.
    """

    INTERFACE: Final[str] = DATABASE_IMPACT_GRAPH_INTERFACE
    SCHEMA: Final[str] = DATABASE_IMPACT_GRAPH_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        policy_id: str = DEFAULT_POLICY_ID,
        parser_id: str = "",
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseImpactGraph; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._policy_id = _text(policy_id or DEFAULT_POLICY_ID, "policy_id")
        self._parser_id = _text(parser_id, "parser_id", required=False)
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._current_revision_id: str = ""

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def parser_id(self) -> str:
        return self._parser_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def current_revision_id(self) -> str:
        return self._current_revision_id

    def open(self) -> "DatabaseImpactGraph":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_IMPACT_GRAPH_INTERFACE),
                ("schema", DATABASE_IMPACT_GRAPH_SCHEMA),
                ("policy_id", self._policy_id),
                ("authority", AUTHORITY_CLASS),
                ("graph_version", DEFAULT_GRAPH_VERSION),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO impact_graph_metadata(key, value)
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
            self._current_revision_id = ""
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseImpactGraph":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseImpactGraphNotOpenError(
                "DatabaseImpactGraph is not open"
            )
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

    def materialize(
        self,
        *,
        snapshot_id: str,
        edges: Sequence[ImpactEdgeSpec | Mapping[str, Any]],
        symbols: Sequence[ImpactSymbolSpec | Mapping[str, Any]] = (),
        frontiers: Sequence[ImpactFrontierSpec | Mapping[str, Any]] = (),
        parser_id: str = "",
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        materialization_id: str = "",
    ) -> MaterializationResult:
        """Materialize a versioned impact graph revision from edge specs."""

        selected_snapshot = _text(snapshot_id, "snapshot_id")
        selected_parser = _text(
            parser_id or self._parser_id or "parser:unspecified", "parser_id"
        )
        selected_policy = _text(
            policy_id if policy_id is not None else self._policy_id, "policy_id"
        )
        edge_list = [self._coerce_edge(item) for item in edges]
        if len(edge_list) > MAX_EDGES:
            raise DatabaseImpactGraphBoundsError(
                f"edge count exceeds {MAX_EDGES}"
            )
        symbol_list = [self._coerce_symbol(item) for item in symbols]
        frontier_list = [self._coerce_frontier(item) for item in frontiers]

        # Ensure endpoint symbols exist even when only edges were supplied.
        known_symbols: dict[str, ImpactSymbolSpec] = {
            item.qualified_name: item for item in symbol_list
        }
        for edge in edge_list:
            for name, path, language in (
                (edge.source_symbol, edge.source_path or edge.path, edge.source_language),
                (edge.target_symbol, edge.target_path, edge.target_language),
            ):
                if name not in known_symbols:
                    known_symbols[name] = ImpactSymbolSpec(
                        qualified_name=name,
                        path=path,
                        language=language,
                        is_generated=edge.is_generated
                        or _is_generated_path(path),
                        is_test=_is_test_path(path),
                    )

        # Auto-frontiers from dynamic / unresolved / generated edges.
        auto_frontiers = list(frontier_list)
        for edge in edge_list:
            if edge.is_dynamic or edge.edge_kind is EdgeKind.DYNAMIC:
                auto_frontiers.append(
                    ImpactFrontierSpec(
                        kind=FrontierKind.DYNAMIC_CALL,
                        disposition=FrontierDisposition.OPEN,
                        symbol_key=edge.source_symbol,
                        path=edge.path or edge.source_path,
                        reason=edge.reason or "dynamic_call",
                        blocks_repair=True,
                    )
                )
            if edge.is_generated or edge.edge_kind is EdgeKind.GENERATED_FROM:
                auto_frontiers.append(
                    ImpactFrontierSpec(
                        kind=FrontierKind.GENERATED_CODE,
                        disposition=FrontierDisposition.OPEN,
                        symbol_key=edge.source_symbol,
                        path=edge.path or edge.source_path,
                        reason=edge.reason or "generated_code",
                        blocks_repair=True,
                    )
                )
            if edge.is_cross_language:
                auto_frontiers.append(
                    ImpactFrontierSpec(
                        kind=FrontierKind.CROSS_LANGUAGE,
                        disposition=FrontierDisposition.OPEN,
                        symbol_key=edge.source_symbol,
                        path=edge.path or edge.source_path,
                        reason=edge.reason or "cross_language",
                        blocks_repair=True,
                    )
                )
            if edge.target_symbol.startswith("<") or edge.target_symbol in {
                "<dynamic>",
                "<unresolved>",
                "<unknown>",
            }:
                auto_frontiers.append(
                    ImpactFrontierSpec(
                        kind=FrontierKind.UNRESOLVED_SYMBOL,
                        disposition=FrontierDisposition.OPEN,
                        symbol_key=edge.source_symbol,
                        path=edge.path or edge.source_path,
                        reason=f"unresolved:{edge.target_symbol}",
                        blocks_repair=True,
                    )
                )

        # Deduplicate frontiers by identity payload.
        frontier_by_id: dict[str, ImpactFrontierSpec] = {}
        for item in auto_frontiers:
            record = ImpactFrontierRecord(
                frontier_id="",
                kind=item.kind,
                disposition=item.disposition,
                symbol_key=item.symbol_key,
                path=item.path,
                reason=item.reason,
                blocks_repair=item.blocks_repair,
            )
            frontier_by_id[record.frontier_id] = item

        mat_id = _text(
            materialization_id
            or _identity(
                "materialization",
                {
                    "snapshot_id": selected_snapshot,
                    "parser_id": selected_parser,
                    "policy_id": selected_policy,
                    "edge_count": len(edge_list),
                    "symbol_count": len(known_symbols),
                    "edges": [
                        {
                            "s": e.source_symbol,
                            "t": e.target_symbol,
                            "k": e.edge_kind.value
                            if isinstance(e.edge_kind, EdgeKind)
                            else str(e.edge_kind),
                        }
                        for e in sorted(
                            edge_list,
                            key=lambda item: (
                                item.source_symbol,
                                item.target_symbol,
                                item.edge_kind.value
                                if isinstance(item.edge_kind, EdgeKind)
                                else str(item.edge_kind),
                            ),
                        )
                    ],
                },
            ),
            "materialization_id",
        )
        revision = ImpactGraphRevision(
            revision_id="",
            snapshot_id=selected_snapshot,
            parser_id=selected_parser,
            policy_id=selected_policy,
            schema_id=DATABASE_IMPACT_GRAPH_SCHEMA,
            materialization_id=mat_id,
            repository_id=repository_id,
            tree_id=tree_id,
            created_at=_utc_iso(),
            edge_count=len(edge_list),
            symbol_count=len(known_symbols),
            frontier_count=len(frontier_by_id),
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                """
                SELECT revision_id FROM impact_graph_revisions
                WHERE revision_id = ?
                """,
                [revision.revision_id],
            ).fetchone()
            if existing is not None:
                self._current_revision_id = revision.revision_id
                return MaterializationResult(
                    revision=revision,
                    edge_count=revision.edge_count,
                    symbol_count=revision.symbol_count,
                    frontier_count=revision.frontier_count,
                )

            connection.execute(
                """
                INSERT INTO impact_graph_revisions (
                    revision_id, snapshot_id, repository_id, tree_id,
                    parser_id, policy_id, schema_id, materialization_id,
                    created_at, edge_count, symbol_count, frontier_count,
                    body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    revision.revision_id,
                    revision.snapshot_id,
                    revision.repository_id,
                    revision.tree_id,
                    revision.parser_id,
                    revision.policy_id,
                    revision.schema_id,
                    revision.materialization_id,
                    revision.created_at,
                    revision.edge_count,
                    revision.symbol_count,
                    revision.frontier_count,
                    _canonical_json(revision.to_dict()),
                ],
            )

            for spec in sorted(
                known_symbols.values(), key=lambda item: item.qualified_name
            ):
                key = _symbol_key(
                    qualified_name=spec.qualified_name,
                    path=spec.path,
                    symbol_id=spec.symbol_id,
                )
                # Store under both content key and qualified name for lookup.
                for store_key in (key, spec.qualified_name):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO impact_symbols (
                            symbol_key, revision_id, symbol_id, qualified_name,
                            path, language, symbol_kind, is_generated,
                            is_deleted, is_test, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            store_key,
                            revision.revision_id,
                            spec.symbol_id,
                            spec.qualified_name,
                            spec.path,
                            spec.language,
                            spec.symbol_kind,
                            1 if spec.is_generated else 0,
                            1 if spec.is_deleted else 0,
                            1 if spec.is_test else 0,
                            _canonical_json(
                                {
                                    "qualified_name": spec.qualified_name,
                                    "path": spec.path,
                                    "language": spec.language,
                                    "symbol_kind": spec.symbol_kind,
                                    "is_generated": spec.is_generated,
                                    "is_deleted": spec.is_deleted,
                                    "is_test": spec.is_test,
                                }
                            ),
                        ],
                    )

            for edge in sorted(
                edge_list,
                key=lambda item: (
                    item.source_symbol,
                    item.target_symbol,
                    item.edge_kind.value
                    if isinstance(item.edge_kind, EdgeKind)
                    else str(item.edge_kind),
                ),
            ):
                kind = (
                    edge.edge_kind.value
                    if isinstance(edge.edge_kind, EdgeKind)
                    else str(edge.edge_kind)
                )
                edge_id = _identity(
                    "impact-edge",
                    {
                        "schema": IMPACT_EDGE_SCHEMA,
                        "revision_id": revision.revision_id,
                        "source": edge.source_symbol,
                        "target": edge.target_symbol,
                        "kind": kind,
                        "path": edge.path,
                    },
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO impact_edges (
                        edge_id, revision_id, source_symbol, target_symbol,
                        edge_kind, authority, path, is_dynamic, is_generated,
                        is_cross_language, reason, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        edge_id,
                        revision.revision_id,
                        edge.source_symbol,
                        edge.target_symbol,
                        kind,
                        edge.authority,
                        edge.path or edge.source_path,
                        1 if edge.is_dynamic or edge.edge_kind is EdgeKind.DYNAMIC else 0,
                        1 if edge.is_generated else 0,
                        1 if edge.is_cross_language else 0,
                        edge.reason,
                        _canonical_json(dict(edge.metadata)),
                    ],
                )

            for frontier_id, item in sorted(frontier_by_id.items()):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO impact_frontiers (
                        frontier_id, revision_id, symbol_key, path, kind,
                        disposition, reason, blocks_repair, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        frontier_id,
                        revision.revision_id,
                        item.symbol_key,
                        item.path,
                        item.kind.value
                        if isinstance(item.kind, FrontierKind)
                        else str(item.kind),
                        item.disposition.value
                        if isinstance(item.disposition, FrontierDisposition)
                        else str(item.disposition),
                        item.reason,
                        1 if item.blocks_repair else 0,
                        _canonical_json(
                            {
                                "kind": item.kind.value
                                if isinstance(item.kind, FrontierKind)
                                else str(item.kind),
                                "disposition": item.disposition.value
                                if isinstance(
                                    item.disposition, FrontierDisposition
                                )
                                else str(item.disposition),
                            }
                        ),
                    ],
                )

            self._commit_if_idle(connection)
            self._current_revision_id = revision.revision_id
            self._parser_id = selected_parser
            return MaterializationResult(
                revision=revision,
                edge_count=revision.edge_count,
                symbol_count=revision.symbol_count,
                frontier_count=revision.frontier_count,
            )

    def materialize_from_ast_index(
        self,
        ast_index: Any,
        snapshot_id: str,
        *,
        parser_id: str = "",
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        extra_edges: Sequence[ImpactEdgeSpec | Mapping[str, Any]] = (),
        extra_frontiers: Sequence[ImpactFrontierSpec | Mapping[str, Any]] = (),
    ) -> MaterializationResult:
        """Project DuckDBASTIndex snapshot facts into an impact revision."""

        selected = _text(snapshot_id, "snapshot_id")
        if not hasattr(ast_index, "list_symbols"):
            raise DatabaseImpactGraphIntegrityError(
                "ast_index must provide list_symbols"
            )
        symbols_raw = list(ast_index.list_symbols(selected))
        imports_raw = (
            list(ast_index.list_imports(selected))
            if hasattr(ast_index, "list_imports")
            else []
        )
        calls_raw = (
            list(ast_index.list_calls(selected))
            if hasattr(ast_index, "list_calls")
            else []
        )
        frontiers_raw = (
            list(ast_index.list_frontiers(selected))
            if hasattr(ast_index, "list_frontiers")
            else []
        )

        # Build symbol_id → qualified_name map.
        id_to_name: dict[str, str] = {}
        path_by_name: dict[str, str] = {}
        lang_by_name: dict[str, str] = {}
        symbol_specs: list[ImpactSymbolSpec] = []
        for item in symbols_raw:
            if hasattr(item, "qualified_name"):
                name = str(item.qualified_name)
                symbol_id = str(getattr(item, "symbol_id", "") or "")
                path = str(getattr(item, "path", "") or "")
                language = str(getattr(item, "language", "") or "")
                kind = str(getattr(item, "symbol_kind", "") or "")
            else:
                mapping = dict(item)
                name = str(mapping.get("qualified_name") or "")
                symbol_id = str(mapping.get("symbol_id") or "")
                path = str(mapping.get("path") or "")
                language = str(mapping.get("language") or "")
                kind = str(mapping.get("symbol_kind") or "")
            if not name:
                continue
            id_to_name[symbol_id] = name
            path_by_name[name] = path
            lang_by_name[name] = language
            symbol_specs.append(
                ImpactSymbolSpec(
                    qualified_name=name,
                    path=path,
                    language=language,
                    symbol_kind=kind,
                    symbol_id=symbol_id,
                )
            )

        edges: list[ImpactEdgeSpec] = []
        for call in calls_raw:
            mapping = call if isinstance(call, Mapping) else _row_mapping(call)
            caller_id = str(mapping.get("caller_symbol_id") or "")
            callee_id = str(mapping.get("callee_symbol_id") or "")
            path = str(mapping.get("path") or "")
            caller = id_to_name.get(caller_id, caller_id)
            callee = id_to_name.get(callee_id, callee_id)
            if not caller or not callee:
                continue
            dynamic = callee in {
                "<dynamic>",
                "getattr",
                "eval",
                "exec",
                "__import__",
            } or callee.startswith("<")
            edges.append(
                ImpactEdgeSpec(
                    source_symbol=caller,
                    target_symbol=callee,
                    edge_kind=EdgeKind.DYNAMIC if dynamic else EdgeKind.CALLS,
                    path=path,
                    source_path=path_by_name.get(caller, path),
                    target_path=path_by_name.get(callee, ""),
                    source_language=lang_by_name.get(caller, ""),
                    target_language=lang_by_name.get(callee, ""),
                    is_dynamic=dynamic,
                )
            )

        for imp in imports_raw:
            mapping = imp if isinstance(imp, Mapping) else _row_mapping(imp)
            module_name = str(mapping.get("module_name") or "")
            path = str(mapping.get("path") or "")
            if not module_name:
                continue
            # Prefer a top-level symbol defined in the importing file.
            candidates = [
                name for name, candidate_path in path_by_name.items()
                if candidate_path == path
            ]
            if candidates:
                importer = sorted(
                    candidates, key=lambda name: (name.count("."), name)
                )[0]
            else:
                importer = path or module_name
            # module_name may be "pkg:Name" from from-import form.
            target = module_name.replace(":", ".")
            edges.append(
                ImpactEdgeSpec(
                    source_symbol=str(importer),
                    target_symbol=target,
                    edge_kind=EdgeKind.IMPORTS,
                    path=path,
                    source_path=path,
                )
            )

        for extra in extra_edges:
            edges.append(self._coerce_edge(extra))

        frontier_specs: list[ImpactFrontierSpec] = []
        for fr in frontiers_raw:
            if hasattr(fr, "status"):
                status = str(getattr(fr.status, "value", fr.status))
                path = str(getattr(fr, "path", "") or "")
                reason = str(getattr(fr, "reason", "") or "")
            else:
                mapping = dict(fr)
                status = str(mapping.get("status") or "")
                path = str(mapping.get("path") or "")
                reason = str(mapping.get("reason") or "")
            if status in {"failed", "unknown"}:
                kind = FrontierKind.PARSER_UNCERTAINTY
            elif status == "unsupported":
                kind = FrontierKind.UNSUPPORTED_LANGUAGE
            else:
                continue
            frontier_specs.append(
                ImpactFrontierSpec(
                    kind=kind,
                    disposition=(
                        FrontierDisposition.UNSUPPORTED
                        if kind is FrontierKind.UNSUPPORTED_LANGUAGE
                        else FrontierDisposition.OPEN
                    ),
                    path=path,
                    reason=reason or status,
                    blocks_repair=True,
                )
            )
        for extra in extra_frontiers:
            frontier_specs.append(self._coerce_frontier(extra))

        selected_parser = parser_id or self._parser_id
        if not selected_parser and hasattr(ast_index, "parser_id"):
            selected_parser = str(ast_index.parser_id or "")
        if not selected_parser:
            selected_parser = "parser:unspecified"

        return self.materialize(
            snapshot_id=selected,
            edges=edges,
            symbols=symbol_specs,
            frontiers=frontier_specs,
            parser_id=selected_parser,
            policy_id=policy_id,
            repository_id=repository_id,
            tree_id=tree_id,
        )

    def impact_closure(
        self,
        seed_symbols: Sequence[str],
        *,
        revision_id: str | None = None,
        max_depth: int = 64,
        max_consumers: int = MAX_CONSUMERS,
        include_nomination: bool = True,
    ) -> ImpactClosure:
        """Compute reverse transitive impact closure for seed symbols."""

        seeds = tuple(
            dict.fromkeys(
                _text(item, "seed_symbol")
                for item in seed_symbols
                if str(item).strip()
            )
        )
        if not seeds:
            raise DatabaseImpactGraphIntegrityError(
                "impact_closure requires seed_symbols"
            )
        if len(seeds) > MAX_SEEDS:
            raise DatabaseImpactGraphBoundsError(
                f"seed count exceeds {MAX_SEEDS}"
            )
        depth_limit = _positive_int(
            int(max_depth), "max_depth", maximum=HARD_MAX_DEPTH
        )
        if depth_limit > MAX_DEPTH:
            depth_limit = MAX_DEPTH
        consumer_limit = _positive_int(
            int(max_consumers), "max_consumers", maximum=MAX_CONSUMERS
        )

        with self._lock:
            connection = self._require()
            revision = self._load_revision(connection, revision_id)
            edges = self._load_edges(connection, revision.revision_id)
            symbols = self._load_symbols(connection, revision.revision_id)
            base_frontiers = self._load_frontiers(
                connection, revision.revision_id
            )

            # Build reverse adjacency: provider → list of (consumer, edge).
            reverse: dict[str, list[dict[str, Any]]] = {}
            forward_mandatory: dict[str, list[str]] = {}
            nomination_hits: list[dict[str, Any]] = []
            for edge in edges:
                kind = EdgeKind.coerce(edge["edge_kind"])
                source = str(edge["source_symbol"])
                target = str(edge["target_symbol"])
                if kind in _NOMINATION_EDGE_KINDS:
                    nomination_hits.append(edge)
                    continue
                if kind is EdgeKind.DYNAMIC or int(edge.get("is_dynamic") or 0):
                    # Dynamic edges do not expand mandatory closure.
                    continue
                if kind not in _MANDATORY_EDGE_KINDS:
                    continue
                reverse.setdefault(target, []).append(edge)
                forward_mandatory.setdefault(source, []).append(target)

            # BFS reverse impact from seeds.
            consumers_map: dict[str, ImpactConsumerRecord] = {}
            edge_count = 0
            truncated = False
            queue: deque[tuple[str, int, str]] = deque(
                (seed, 0, "") for seed in seeds
            )
            visited: set[str] = set(seeds)
            while queue:
                current, depth, via = queue.popleft()
                if depth >= depth_limit:
                    if reverse.get(current):
                        truncated = True
                    continue
                for edge in sorted(
                    reverse.get(current, ()),
                    key=lambda item: (
                        str(item["source_symbol"]),
                        str(item["edge_kind"]),
                        str(item["edge_id"]),
                    ),
                ):
                    consumer_name = str(edge["source_symbol"])
                    edge_count += 1
                    if edge_count > MAX_EDGES:
                        truncated = True
                        break
                    kind = str(edge["edge_kind"])
                    meta = symbols.get(consumer_name, {})
                    path = str(meta.get("path") or edge.get("path") or "")
                    language = str(meta.get("language") or "")
                    is_deleted = bool(int(meta.get("is_deleted") or 0))
                    is_generated = bool(
                        int(meta.get("is_generated") or 0)
                        or int(edge.get("is_generated") or 0)
                    )
                    is_cross = bool(int(edge.get("is_cross_language") or 0))
                    is_test = bool(int(meta.get("is_test") or 0))
                    prior = consumers_map.get(consumer_name)
                    edge_kinds = list(prior.edge_kinds) if prior else []
                    edge_ids = list(prior.edge_ids) if prior else []
                    if kind not in edge_kinds:
                        edge_kinds.append(kind)
                    edge_id = str(edge["edge_id"])
                    if edge_id not in edge_ids:
                        edge_ids.append(edge_id)
                    disposition = _disposition_for_symbol(
                        symbol=consumer_name,
                        path=path,
                        language=language,
                        is_deleted=is_deleted,
                        is_generated=is_generated,
                        is_cross_language=is_cross,
                        edge_kinds=edge_kinds,
                        parser_uncertain=False,
                        is_test=is_test,
                    )
                    # Keep the strongest disposition if already present.
                    if prior is not None:
                        disposition = _stronger_disposition(
                            prior.disposition, disposition
                        )
                        next_depth = min(prior.depth, depth + 1)
                    else:
                        next_depth = depth + 1
                    consumers_map[consumer_name] = ImpactConsumerRecord(
                        consumer_id=_identity(
                            "impact-consumer",
                            {
                                "schema": IMPACT_CONSUMER_SCHEMA,
                                "symbol": consumer_name,
                                "revision_id": revision.revision_id,
                            },
                        ),
                        symbol=consumer_name,
                        disposition=disposition,
                        depth=next_depth,
                        path=path,
                        language=language,
                        edge_kinds=tuple(edge_kinds),
                        edge_ids=tuple(edge_ids),
                        mandatory=True,
                        via_symbol=via or current,
                    )
                    if consumer_name not in visited:
                        if len(consumers_map) >= consumer_limit:
                            truncated = True
                            continue
                        visited.add(consumer_name)
                        queue.append((consumer_name, depth + 1, current))
                if truncated and edge_count > MAX_EDGES:
                    break

            # SCCs among reached consumers + seeds using mandatory edges.
            scc_nodes = sorted(set(seeds) | set(consumers_map))
            scc_adj: dict[str, list[str]] = {node: [] for node in scc_nodes}
            for source, targets in forward_mandatory.items():
                if source not in scc_adj:
                    continue
                for target in targets:
                    if target in scc_adj:
                        scc_adj[source].append(target)
            scc_records = tuple(
                ImpactSCCRecord(scc_id="", member_symbols=members)
                for members in _tarjan_sccs(scc_nodes, scc_adj)[:MAX_SCCS]
            )

            # Frontiers: base revision frontiers + truncation.
            frontiers = list(base_frontiers)
            if truncated:
                frontiers.append(
                    ImpactFrontierRecord(
                        frontier_id="",
                        kind=FrontierKind.UNBOUNDED,
                        disposition=FrontierDisposition.OPEN,
                        reason="impact_closure_truncated",
                        blocks_repair=True,
                    )
                )

            nominated_records: list[ImpactConsumerRecord] = []
            if include_nomination:
                for edge in nomination_hits:
                    # Nomination edges adjacent to seeds or reached consumers.
                    source = str(edge["source_symbol"])
                    target = str(edge["target_symbol"])
                    related = None
                    if target in seeds or target in consumers_map:
                        related = source
                    elif source in seeds or source in consumers_map:
                        related = target
                    if related is None or related in seeds or related in consumers_map:
                        continue
                    meta = symbols.get(related, {})
                    nominated_records.append(
                        ImpactConsumerRecord(
                            consumer_id=_identity(
                                "impact-consumer-nominated",
                                {
                                    "symbol": related,
                                    "revision_id": revision.revision_id,
                                },
                            ),
                            symbol=related,
                            disposition=ConsumerDisposition.NOMINATED,
                            depth=1,
                            path=str(meta.get("path") or edge.get("path") or ""),
                            language=str(meta.get("language") or ""),
                            edge_kinds=(EdgeKind.NOMINATED.value,),
                            edge_ids=(str(edge["edge_id"]),),
                            mandatory=False,
                            reason="graph_proximity_nomination",
                        )
                    )

            blocking = any(item.blocks_repair for item in frontiers)
            if blocking or truncated:
                completeness = (
                    ImpactCompleteness.TRUNCATED
                    if truncated and not blocking
                    else ImpactCompleteness.PARTIAL_WITH_FRONTIER
                )
                if truncated and blocking:
                    completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            else:
                completeness = ImpactCompleteness.COMPLETE

            consumers = tuple(
                consumers_map[name]
                for name in sorted(
                    consumers_map,
                    key=lambda n: (consumers_map[n].depth, n),
                )
            )
            # Ensure unique disposition per consumer (already enforced).
            closure = ImpactClosure(
                query_id="",
                revision_id=revision.revision_id,
                snapshot_id=revision.snapshot_id,
                parser_id=revision.parser_id,
                policy_id=revision.policy_id,
                schema_id=revision.schema_id,
                seed_symbols=seeds,
                completeness=completeness,
                consumers=consumers,
                sccs=scc_records,
                frontiers=tuple(frontiers),
                nominated=tuple(nominated_records),
                blocks_automatic_repair=blocking,
                truncated=truncated,
                max_depth=depth_limit,
                edge_count=edge_count,
                created_at=_utc_iso(),
            )
            self._persist_query(
                connection,
                query_id=closure.query_id,
                revision_id=revision.revision_id,
                query_kind=QueryKind.IMPACT_CLOSURE,
                snapshot_id=revision.snapshot_id,
                parser_id=revision.parser_id,
                policy_id=revision.policy_id,
                schema_id=revision.schema_id,
                seed_json=_canonical_json(list(seeds)),
                completeness=closure.completeness.value
                if isinstance(closure.completeness, ImpactCompleteness)
                else str(closure.completeness),
                blocks_automatic_repair=closure.blocks_automatic_repair,
                body=closure.to_dict(),
            )
            return closure

    def changed_neighborhood(
        self,
        changed_symbols: Sequence[str],
        *,
        revision_id: str | None = None,
        radius: int = 1,
        page_offset: int = 0,
        page_limit: int = DEFAULT_PAGE_SIZE,
    ) -> ChangedSymbolNeighborhood:
        """Return a bounded neighborhood around changed symbols."""

        changed = tuple(
            dict.fromkeys(
                _text(item, "changed_symbol")
                for item in changed_symbols
                if str(item).strip()
            )
        )
        if not changed:
            raise DatabaseImpactGraphIntegrityError(
                "changed_neighborhood requires changed_symbols"
            )
        radius_n = _nonneg_int(int(radius), "radius")
        if radius_n > MAX_DEPTH:
            raise DatabaseImpactGraphBoundsError(
                f"radius exceeds {MAX_DEPTH}"
            )
        offset = _nonneg_int(int(page_offset), "page_offset")
        limit = _positive_int(
            int(page_limit), "page_limit", maximum=MAX_PAGE_SIZE
        )

        with self._lock:
            connection = self._require()
            revision = self._load_revision(connection, revision_id)
            edges = self._load_edges(connection, revision.revision_id)
            symbols = self._load_symbols(connection, revision.revision_id)
            frontiers = self._load_frontiers(connection, revision.revision_id)

            # Undirected expansion for neighborhood membership.
            adj: dict[str, set[str]] = {}
            edge_by_pair: list[dict[str, Any]] = []
            for edge in edges:
                source = str(edge["source_symbol"])
                target = str(edge["target_symbol"])
                adj.setdefault(source, set()).add(target)
                adj.setdefault(target, set()).add(source)
                edge_by_pair.append(edge)

            reached: set[str] = set(changed)
            frontier_nodes = set(changed)
            for _ in range(radius_n):
                nxt: set[str] = set()
                for node in frontier_nodes:
                    nxt.update(adj.get(node, ()))
                nxt -= reached
                if not nxt:
                    break
                reached |= nxt
                frontier_nodes = nxt

            # Classify edges touching the neighborhood.
            neighborhood_edges = [
                edge
                for edge in edge_by_pair
                if str(edge["source_symbol"]) in reached
                or str(edge["target_symbol"]) in reached
            ]
            neighborhood_edges.sort(
                key=lambda item: (
                    str(item["source_symbol"]),
                    str(item["target_symbol"]),
                    str(item["edge_kind"]),
                    str(item["edge_id"]),
                )
            )
            total = len(neighborhood_edges)
            page = neighborhood_edges[offset : offset + limit]
            has_more = offset + limit < total

            buckets: dict[str, list[str]] = {
                "callers": [],
                "callees": [],
                "imports": [],
                "types": [],
                "tests": [],
                "contracts": [],
                "proofs": [],
                "config": [],
                "docs": [],
            }
            for edge in neighborhood_edges:
                kind = EdgeKind.coerce(edge["edge_kind"])
                source = str(edge["source_symbol"])
                target = str(edge["target_symbol"])
                # dependent → provider
                if target in changed:
                    if kind is EdgeKind.CALLS:
                        buckets["callers"].append(source)
                    elif kind is EdgeKind.IMPORTS:
                        buckets["imports"].append(source)
                    elif kind is EdgeKind.TYPES:
                        buckets["types"].append(source)
                    elif kind is EdgeKind.TESTS:
                        buckets["tests"].append(source)
                    elif kind is EdgeKind.CONTRACTS:
                        buckets["contracts"].append(source)
                    elif kind is EdgeKind.PROOFS:
                        buckets["proofs"].append(source)
                    elif kind is EdgeKind.CONFIG:
                        buckets["config"].append(source)
                    elif kind is EdgeKind.DOCS:
                        buckets["docs"].append(source)
                if source in changed:
                    if kind is EdgeKind.CALLS:
                        buckets["callees"].append(target)
                    elif kind is EdgeKind.IMPORTS:
                        buckets["imports"].append(target)
                    elif kind is EdgeKind.TYPES:
                        buckets["types"].append(target)

            nodes = []
            for name in sorted(reached):
                meta = symbols.get(name, {})
                nodes.append(
                    {
                        "symbol": name,
                        "path": str(meta.get("path") or ""),
                        "language": str(meta.get("language") or ""),
                        "is_changed": name in changed,
                        "is_generated": bool(int(meta.get("is_generated") or 0)),
                        "is_deleted": bool(int(meta.get("is_deleted") or 0)),
                        "is_test": bool(int(meta.get("is_test") or 0)),
                        "authority": AUTHORITY_CLASS,
                    }
                )

            relevant_frontiers = tuple(
                item
                for item in frontiers
                if (not item.symbol_key)
                or item.symbol_key in reached
                or item.path
                in {str(symbols.get(n, {}).get("path") or "") for n in reached}
            )

            neighborhood = ChangedSymbolNeighborhood(
                query_id="",
                revision_id=revision.revision_id,
                snapshot_id=revision.snapshot_id,
                parser_id=revision.parser_id,
                policy_id=revision.policy_id,
                schema_id=revision.schema_id,
                changed_symbols=changed,
                radius=radius_n,
                nodes=tuple(nodes),
                edges=tuple(
                    {
                        "edge_id": str(edge["edge_id"]),
                        "source_symbol": str(edge["source_symbol"]),
                        "target_symbol": str(edge["target_symbol"]),
                        "edge_kind": str(edge["edge_kind"]),
                        "authority": str(
                            edge.get("authority") or AUTHORITY_CLASS
                        ),
                        "path": str(edge.get("path") or ""),
                        "is_dynamic": bool(int(edge.get("is_dynamic") or 0)),
                    }
                    for edge in page
                ),
                callers=tuple(dict.fromkeys(buckets["callers"])),
                callees=tuple(dict.fromkeys(buckets["callees"])),
                imports=tuple(dict.fromkeys(buckets["imports"])),
                types=tuple(dict.fromkeys(buckets["types"])),
                tests=tuple(dict.fromkeys(buckets["tests"])),
                contracts=tuple(dict.fromkeys(buckets["contracts"])),
                proofs=tuple(dict.fromkeys(buckets["proofs"])),
                config=tuple(dict.fromkeys(buckets["config"])),
                docs=tuple(dict.fromkeys(buckets["docs"])),
                frontiers=relevant_frontiers,
                page_offset=offset,
                page_limit=limit,
                total_edge_count=total,
                has_more=has_more,
                created_at=_utc_iso(),
            )
            self._persist_query(
                connection,
                query_id=neighborhood.query_id,
                revision_id=revision.revision_id,
                query_kind=QueryKind.CHANGED_NEIGHBORHOOD,
                snapshot_id=revision.snapshot_id,
                parser_id=revision.parser_id,
                policy_id=revision.policy_id,
                schema_id=revision.schema_id,
                seed_json=_canonical_json(list(changed)),
                completeness=(
                    ImpactCompleteness.PARTIAL_WITH_FRONTIER.value
                    if any(f.blocks_repair for f in relevant_frontiers)
                    else ImpactCompleteness.COMPLETE.value
                ),
                blocks_automatic_repair=any(
                    f.blocks_repair for f in relevant_frontiers
                ),
                body=neighborhood.to_dict(),
            )
            return neighborhood

    def list_related(
        self,
        symbol: str,
        *,
        edge_kind: EdgeKind | str,
        direction: str = "consumers",
        revision_id: str | None = None,
        page_offset: int = 0,
        page_limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Paginated callers/callees/imports/types/tests/contracts/proofs/config/docs."""

        selected = _text(symbol, "symbol")
        kind = EdgeKind.coerce(edge_kind)
        direction_cf = str(direction or "consumers").strip().casefold()
        if direction_cf not in {"consumers", "providers", "both"}:
            raise DatabaseImpactGraphIntegrityError(
                "direction must be consumers, providers, or both"
            )
        offset = _nonneg_int(int(page_offset), "page_offset")
        limit = _positive_int(
            int(page_limit), "page_limit", maximum=MAX_PAGE_SIZE
        )

        with self._lock:
            connection = self._require()
            revision = self._load_revision(connection, revision_id)
            clauses = ["revision_id = ?", "edge_kind = ?"]
            params: list[Any] = [revision.revision_id, kind.value]
            if direction_cf == "consumers":
                # dependent → provider; consumers of symbol have target=symbol
                clauses.append("target_symbol = ?")
                params.append(selected)
            elif direction_cf == "providers":
                clauses.append("source_symbol = ?")
                params.append(selected)
            else:
                clauses.append("(source_symbol = ? OR target_symbol = ?)")
                params.extend([selected, selected])
            sql = f"""
                SELECT edge_id, source_symbol, target_symbol, edge_kind,
                       authority, path, is_dynamic, is_generated,
                       is_cross_language, reason
                FROM impact_edges
                WHERE {' AND '.join(clauses)}
                ORDER BY source_symbol ASC, target_symbol ASC, edge_id ASC
            """
            rows = connection.execute(sql, params).fetchall()
            items = [_row_mapping(row) for row in rows]
            total = len(items)
            page = items[offset : offset + limit]
            related = []
            for item in page:
                if direction_cf == "consumers":
                    name = str(item["source_symbol"])
                elif direction_cf == "providers":
                    name = str(item["target_symbol"])
                else:
                    name = (
                        str(item["source_symbol"])
                        if str(item["target_symbol"]) == selected
                        else str(item["target_symbol"])
                    )
                related.append(
                    {
                        "symbol": name,
                        "edge_id": str(item["edge_id"]),
                        "edge_kind": str(item["edge_kind"]),
                        "authority": str(
                            item.get("authority") or AUTHORITY_CLASS
                        ),
                        "path": str(item.get("path") or ""),
                        "is_dynamic": bool(int(item.get("is_dynamic") or 0)),
                    }
                )
            return {
                "symbol": selected,
                "edge_kind": kind.value,
                "direction": direction_cf,
                "revision_id": revision.revision_id,
                "snapshot_id": revision.snapshot_id,
                "parser_id": revision.parser_id,
                "policy_id": revision.policy_id,
                "schema_id": revision.schema_id,
                "items": related,
                "page_offset": offset,
                "page_limit": limit,
                "total_count": total,
                "has_more": offset + limit < total,
                "authority": AUTHORITY_CLASS,
                "freshness": {
                    "snapshot_id": revision.snapshot_id,
                    "parser_id": revision.parser_id,
                    "policy_id": revision.policy_id,
                    "schema_id": revision.schema_id,
                    "revision_id": revision.revision_id,
                },
            }

    def list_callers(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.CALLS, direction="consumers", **kwargs
        )

    def list_callees(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.CALLS, direction="providers", **kwargs
        )

    def list_imports(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.IMPORTS, direction="both", **kwargs
        )

    def list_types(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.TYPES, direction="both", **kwargs
        )

    def list_tests(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.TESTS, direction="consumers", **kwargs
        )

    def list_contracts(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.CONTRACTS, direction="consumers", **kwargs
        )

    def list_proofs(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.PROOFS, direction="consumers", **kwargs
        )

    def list_config(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.CONFIG, direction="consumers", **kwargs
        )

    def list_docs(
        self, symbol: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.list_related(
            symbol, edge_kind=EdgeKind.DOCS, direction="consumers", **kwargs
        )

    def get_revision(
        self, revision_id: str | None = None
    ) -> ImpactGraphRevision | None:
        with self._lock:
            connection = self._require()
            try:
                return self._load_revision(connection, revision_id)
            except DatabaseImpactGraphIntegrityError:
                return None

    def list_frontiers(
        self, *, revision_id: str | None = None
    ) -> tuple[ImpactFrontierRecord, ...]:
        with self._lock:
            connection = self._require()
            revision = self._load_revision(connection, revision_id)
            return tuple(
                self._load_frontiers(connection, revision.revision_id)
            )

    def list_edges(
        self,
        *,
        revision_id: str | None = None,
        edge_kind: EdgeKind | str | None = None,
        page_offset: int = 0,
        page_limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        offset = _nonneg_int(int(page_offset), "page_offset")
        limit = _positive_int(
            int(page_limit), "page_limit", maximum=MAX_PAGE_SIZE
        )
        with self._lock:
            connection = self._require()
            revision = self._load_revision(connection, revision_id)
            edges = self._load_edges(connection, revision.revision_id)
            if edge_kind is not None:
                kind = EdgeKind.coerce(edge_kind)
                edges = [
                    edge
                    for edge in edges
                    if str(edge["edge_kind"]) == kind.value
                ]
            total = len(edges)
            page = edges[offset : offset + limit]
            return {
                "revision_id": revision.revision_id,
                "snapshot_id": revision.snapshot_id,
                "items": page,
                "page_offset": offset,
                "page_limit": limit,
                "total_count": total,
                "has_more": offset + limit < total,
                "authority": AUTHORITY_CLASS,
                "freshness": {
                    "snapshot_id": revision.snapshot_id,
                    "parser_id": revision.parser_id,
                    "policy_id": revision.policy_id,
                    "schema_id": revision.schema_id,
                    "revision_id": revision.revision_id,
                },
            }

    def metadata(self) -> dict[str, str]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT key, value FROM impact_graph_metadata ORDER BY key ASC"
            ).fetchall()
            return {
                str(_row_mapping(row)["key"]): str(_row_mapping(row)["value"])
                for row in rows
            }

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _coerce_edge(value: ImpactEdgeSpec | Mapping[str, Any]) -> ImpactEdgeSpec:
        if isinstance(value, ImpactEdgeSpec):
            return value
        if not isinstance(value, Mapping):
            raise DatabaseImpactGraphIntegrityError("edge must be a mapping")
        return ImpactEdgeSpec(
            source_symbol=str(value.get("source_symbol") or value.get("source") or ""),
            target_symbol=str(value.get("target_symbol") or value.get("target") or ""),
            edge_kind=value.get("edge_kind") or value.get("kind") or "",
            path=str(value.get("path") or ""),
            source_path=str(value.get("source_path") or ""),
            target_path=str(value.get("target_path") or ""),
            source_language=str(value.get("source_language") or ""),
            target_language=str(value.get("target_language") or ""),
            is_dynamic=bool(value.get("is_dynamic", False)),
            is_generated=bool(value.get("is_generated", False)),
            is_cross_language=bool(value.get("is_cross_language", False)),
            authority=str(value.get("authority") or AUTHORITY_CLASS),
            reason=str(value.get("reason") or ""),
            metadata=dict(value.get("metadata") or {}),
        )

    @staticmethod
    def _coerce_symbol(
        value: ImpactSymbolSpec | Mapping[str, Any],
    ) -> ImpactSymbolSpec:
        if isinstance(value, ImpactSymbolSpec):
            return value
        if not isinstance(value, Mapping):
            raise DatabaseImpactGraphIntegrityError("symbol must be a mapping")
        return ImpactSymbolSpec(
            qualified_name=str(
                value.get("qualified_name") or value.get("symbol") or ""
            ),
            path=str(value.get("path") or ""),
            language=str(value.get("language") or ""),
            symbol_kind=str(value.get("symbol_kind") or value.get("kind") or ""),
            symbol_id=str(value.get("symbol_id") or ""),
            is_generated=bool(value.get("is_generated", False)),
            is_deleted=bool(value.get("is_deleted", False)),
            is_test=bool(value.get("is_test", False)),
        )

    @staticmethod
    def _coerce_frontier(
        value: ImpactFrontierSpec | Mapping[str, Any],
    ) -> ImpactFrontierSpec:
        if isinstance(value, ImpactFrontierSpec):
            return value
        if not isinstance(value, Mapping):
            raise DatabaseImpactGraphIntegrityError("frontier must be a mapping")
        return ImpactFrontierSpec(
            kind=value.get("kind") or FrontierKind.UNRESOLVED_SYMBOL,
            disposition=value.get("disposition") or FrontierDisposition.OPEN,
            symbol_key=str(value.get("symbol_key") or value.get("symbol") or ""),
            path=str(value.get("path") or ""),
            reason=str(value.get("reason") or ""),
            blocks_repair=bool(value.get("blocks_repair", True)),
        )

    def _load_revision(
        self, connection: Any, revision_id: str | None
    ) -> ImpactGraphRevision:
        selected = str(revision_id or self._current_revision_id or "").strip()
        if selected:
            row = connection.execute(
                """
                SELECT revision_id, snapshot_id, repository_id, tree_id,
                       parser_id, policy_id, schema_id, materialization_id,
                       created_at, edge_count, symbol_count, frontier_count
                FROM impact_graph_revisions
                WHERE revision_id = ?
                LIMIT 1
                """,
                [selected],
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT revision_id, snapshot_id, repository_id, tree_id,
                       parser_id, policy_id, schema_id, materialization_id,
                       created_at, edge_count, symbol_count, frontier_count
                FROM impact_graph_revisions
                ORDER BY created_at DESC, revision_id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise DatabaseImpactGraphIntegrityError(
                "no impact graph revision is available"
            )
        mapping = _row_mapping(row)
        revision = ImpactGraphRevision(
            revision_id=str(mapping["revision_id"]),
            snapshot_id=str(mapping["snapshot_id"]),
            parser_id=str(mapping["parser_id"]),
            policy_id=str(mapping["policy_id"]),
            schema_id=str(mapping["schema_id"]),
            materialization_id=str(mapping["materialization_id"]),
            repository_id=str(mapping.get("repository_id") or ""),
            tree_id=str(mapping.get("tree_id") or ""),
            created_at=str(mapping.get("created_at") or ""),
            edge_count=int(mapping.get("edge_count") or 0),
            symbol_count=int(mapping.get("symbol_count") or 0),
            frontier_count=int(mapping.get("frontier_count") or 0),
        )
        self._current_revision_id = revision.revision_id
        return revision

    def _load_edges(
        self, connection: Any, revision_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT edge_id, revision_id, source_symbol, target_symbol,
                   edge_kind, authority, path, is_dynamic, is_generated,
                   is_cross_language, reason, body_json
            FROM impact_edges
            WHERE revision_id = ?
            ORDER BY source_symbol ASC, target_symbol ASC, edge_kind ASC,
                     edge_id ASC
            """,
            [revision_id],
        ).fetchall()
        return [_row_mapping(row) for row in rows]

    def _load_symbols(
        self, connection: Any, revision_id: str
    ) -> dict[str, dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT symbol_key, symbol_id, qualified_name, path, language,
                   symbol_kind, is_generated, is_deleted, is_test
            FROM impact_symbols
            WHERE revision_id = ?
            """,
            [revision_id],
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            mapping = _row_mapping(row)
            name = str(mapping.get("qualified_name") or "")
            if not name:
                continue
            # Prefer qualified-name keyed rows (symbol_key == name).
            existing = result.get(name)
            payload = {
                "symbol_id": str(mapping.get("symbol_id") or ""),
                "qualified_name": name,
                "path": str(mapping.get("path") or ""),
                "language": str(mapping.get("language") or ""),
                "symbol_kind": str(mapping.get("symbol_kind") or ""),
                "is_generated": int(mapping.get("is_generated") or 0),
                "is_deleted": int(mapping.get("is_deleted") or 0),
                "is_test": int(mapping.get("is_test") or 0),
            }
            if existing is None or str(mapping.get("symbol_key") or "") == name:
                result[name] = payload
        return result

    def _load_frontiers(
        self, connection: Any, revision_id: str
    ) -> list[ImpactFrontierRecord]:
        rows = connection.execute(
            """
            SELECT frontier_id, symbol_key, path, kind, disposition,
                   reason, blocks_repair
            FROM impact_frontiers
            WHERE revision_id = ?
            ORDER BY kind ASC, symbol_key ASC, frontier_id ASC
            """,
            [revision_id],
        ).fetchall()
        results: list[ImpactFrontierRecord] = []
        for row in rows:
            mapping = _row_mapping(row)
            results.append(
                ImpactFrontierRecord(
                    frontier_id=str(mapping["frontier_id"]),
                    kind=str(mapping["kind"]),
                    disposition=str(mapping["disposition"]),
                    symbol_key=str(mapping.get("symbol_key") or ""),
                    path=str(mapping.get("path") or ""),
                    reason=str(mapping.get("reason") or ""),
                    blocks_repair=bool(int(mapping.get("blocks_repair") or 0)),
                )
            )
        return results

    def _persist_query(
        self,
        connection: Any,
        *,
        query_id: str,
        revision_id: str,
        query_kind: QueryKind,
        snapshot_id: str,
        parser_id: str,
        policy_id: str,
        schema_id: str,
        seed_json: str,
        completeness: str,
        blocks_automatic_repair: bool,
        body: Mapping[str, Any],
    ) -> None:
        encoded = _canonical_json(body)
        if len(encoded.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            encoded = _canonical_json(
                {
                    "query_id": query_id,
                    "truncated_body": True,
                    "completeness": completeness,
                }
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO impact_query_receipts (
                query_id, revision_id, query_kind, snapshot_id, parser_id,
                policy_id, schema_id, seed_json, completeness,
                blocks_automatic_repair, created_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                query_id,
                revision_id,
                query_kind.value,
                snapshot_id,
                parser_id,
                policy_id,
                schema_id,
                seed_json,
                completeness,
                1 if blocks_automatic_repair else 0,
                _utc_iso(),
                encoded,
            ],
        )
        self._commit_if_idle(connection)


def _stronger_disposition(
    left: ConsumerDisposition | str,
    right: ConsumerDisposition | str,
) -> ConsumerDisposition:
    order = [
        ConsumerDisposition.NOMINATED,
        ConsumerDisposition.UNCHANGED,
        ConsumerDisposition.REVIEW,
        ConsumerDisposition.MUST_REVALIDATE,
        ConsumerDisposition.MUST_REPAIR,
        ConsumerDisposition.GENERATED,
        ConsumerDisposition.CROSS_LANGUAGE,
        ConsumerDisposition.PARSER_UNCERTAIN,
        ConsumerDisposition.DELETED,
        ConsumerDisposition.OPEN_FRONTIER,
        ConsumerDisposition.UNSUPPORTED,
    ]
    left_d = (
        left
        if isinstance(left, ConsumerDisposition)
        else ConsumerDisposition(str(left))
    )
    right_d = (
        right
        if isinstance(right, ConsumerDisposition)
        else ConsumerDisposition(str(right))
    )
    return max(left_d, right_d, key=lambda item: order.index(item))


def open_database_impact_graph(
    database_path: Path | str,
    *,
    policy_id: str = DEFAULT_POLICY_ID,
    parser_id: str = "",
) -> DatabaseImpactGraph:
    """Open a DatabaseImpactGraph store (creates schema on first open)."""

    return DatabaseImpactGraph(
        database_path, policy_id=policy_id, parser_id=parser_id
    ).open()


__all__ = [
    "AUTHORITY_CLASS",
    "CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE",
    "CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA",
    "ConsumerDisposition",
    "ChangedSymbolNeighborhood",
    "DATABASE_IMPACT_GRAPH_INTERFACE",
    "DATABASE_IMPACT_GRAPH_SCHEMA",
    "DEFAULT_POLICY_ID",
    "DatabaseImpactGraph",
    "DatabaseImpactGraphBoundsError",
    "DatabaseImpactGraphConflictError",
    "DatabaseImpactGraphError",
    "DatabaseImpactGraphIntegrityError",
    "DatabaseImpactGraphNotOpenError",
    "DuckDBUnavailableError",
    "EdgeKind",
    "FrontierDisposition",
    "FrontierKind",
    "IMPACT_CLOSURE_INTERFACE",
    "IMPACT_CLOSURE_SCHEMA",
    "ImpactClosure",
    "ImpactCompleteness",
    "ImpactConsumerRecord",
    "ImpactEdgeSpec",
    "ImpactFrontierRecord",
    "ImpactFrontierSpec",
    "ImpactGraphRevision",
    "ImpactSCCRecord",
    "ImpactSymbolSpec",
    "MaterializationResult",
    "NOMINATION_AUTHORITY",
    "QueryKind",
    "duckdb_available",
    "open_database_impact_graph",
]
