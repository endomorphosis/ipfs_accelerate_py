"""Materialized symbol dependency, impact, and changed-neighborhood queries.

DQP-023 / Interfaces: ``DatabaseImpactGraph@1``, ``ImpactClosure@1``,
``ChangedSymbolNeighborhood@1``
============================================================================

Bounded, versioned SQL queries over snapshot-bound AST/symbol evidence produce
callers, callees, imports, types, tests, contracts, proofs, config, docs, and
unresolved dynamic frontiers for a mutation or task.  Every query result binds
snapshot, parser, policy, and schema identities and records freshness.

Acceptance properties
---------------------
* All resolved consumers receive exactly one disposition.
* An open or unsupported frontier blocks automatic repair.
* Query results bind snapshot / parser / policy / schema identities.
* Similarity and graph-proximity edges remain nomination only; they never
  grant semantic authority or close impact coverage.

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a graph store is the first I/O boundary.
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

# Optional co-location with the AST index.  The graph can materialize from an
# open DuckDBASTIndex without requiring it at import time for pure-edge tests.
try:
    from .duckdb_ast_index import (  # type: ignore
        DEFAULT_PARSER_ID as _AST_DEFAULT_PARSER_ID,
        DuckDBASTIndex,
        ParseStatus,
        duckdb_available as _ast_duckdb_available,
    )
except Exception:  # pragma: no cover - optional co-location
    DuckDBASTIndex = None  # type: ignore[misc, assignment]
    ParseStatus = None  # type: ignore[misc, assignment]
    _AST_DEFAULT_PARSER_ID = "python-ast@unknown"
    _ast_duckdb_available = None  # type: ignore[assignment]


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
    "ipfs_accelerate_py/agent-supervisor/impact-consumer-record@1"
)
IMPACT_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-frontier-record@1"
)
QUERY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-query-binding@1"
)

DEFAULT_POLICY_ID: Final[str] = "database-impact-policy@1"
DEFAULT_GRAPH_VERSION: Final[str] = "database-impact-graph@1"
DEFAULT_PARSER_ID: Final[str] = str(_AST_DEFAULT_PARSER_ID)
AUTHORITY_CLASS: Final[str] = "derived_evidence"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_REASON_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_EDGES: Final[int] = 250_000
MAX_CONSUMERS: Final[int] = 16_384
MAX_FRONTIER: Final[int] = 4_096
MAX_SEEDS: Final[int] = 1_024
MAX_DEPTH: Final[int] = 256
MAX_PAGE_SIZE: Final[int] = 4_096
DEFAULT_PAGE_SIZE: Final[int] = 256
DEFAULT_MAX_DEPTH: Final[int] = 32

# Edge kinds whose authority is nomination only (never closes coverage).
_NOMINATION_KINDS: Final[frozenset[str]] = frozenset(
    {
        "similarity",
        "graph_proximity",
        "vector_nomination",
        "history_nomination",
    }
)

# Consumer categories emitted in neighborhood buckets.
_NEIGHBORHOOD_BUCKETS: Final[tuple[str, ...]] = (
    "callers",
    "callees",
    "imports",
    "types",
    "tests",
    "contracts",
    "proofs",
    "config",
    "docs",
    "aliases",
    "reexports",
    "generated",
    "nominations",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS impact_graph_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS impact_symbols (
    symbol_key VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    language VARCHAR NOT NULL DEFAULT '',
    symbol_kind VARCHAR NOT NULL DEFAULT '',
    fingerprint VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_symbols_snapshot_name_idx
    ON impact_symbols(snapshot_id, qualified_name);
CREATE INDEX IF NOT EXISTS impact_symbols_snapshot_id_idx
    ON impact_symbols(snapshot_id, symbol_id);

CREATE TABLE IF NOT EXISTS impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    source_symbol_key VARCHAR NOT NULL,
    target_symbol_key VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    authority VARCHAR NOT NULL,
    semantic_authority INTEGER NOT NULL DEFAULT 0,
    path VARCHAR NOT NULL DEFAULT '',
    evidence_ref VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_edges_snapshot_kind_idx
    ON impact_edges(snapshot_id, edge_kind);
CREATE INDEX IF NOT EXISTS impact_edges_source_idx
    ON impact_edges(source_symbol_key, edge_kind);
CREATE INDEX IF NOT EXISTS impact_edges_target_idx
    ON impact_edges(target_symbol_key, edge_kind);

CREATE TABLE IF NOT EXISTS impact_frontiers (
    frontier_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    subject_key VARCHAR NOT NULL DEFAULT '',
    path VARCHAR NOT NULL DEFAULT '',
    status VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    blocks_automatic_repair INTEGER NOT NULL DEFAULT 1,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_frontiers_snapshot_idx
    ON impact_frontiers(snapshot_id, status);

CREATE TABLE IF NOT EXISTS impact_closures (
    closure_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    schema_id VARCHAR NOT NULL,
    seed_json VARCHAR NOT NULL,
    completeness VARCHAR NOT NULL,
    automatic_repair_allowed INTEGER NOT NULL DEFAULT 0,
    consumer_count BIGINT NOT NULL DEFAULT 0,
    frontier_count BIGINT NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS impact_closures_snapshot_idx
    ON impact_closures(snapshot_id, recorded_at);

CREATE TABLE IF NOT EXISTS impact_materializations (
    materialization_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    schema_id VARCHAR NOT NULL,
    edge_count BIGINT NOT NULL DEFAULT 0,
    symbol_count BIGINT NOT NULL DEFAULT 0,
    frontier_count BIGINT NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS impact_materializations_snapshot_uidx
    ON impact_materializations(snapshot_id, parser_id, policy_id);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseImpactGraphError(RuntimeError):
    """Base error for database impact graph failures."""


class DatabaseImpactGraphNotOpenError(DatabaseImpactGraphError):
    """Operation requires an open impact graph store."""


class DatabaseImpactGraphIntegrityError(DatabaseImpactGraphError, ValueError):
    """Identity, binding, or payload integrity failure."""


class DatabaseImpactGraphBoundsError(DatabaseImpactGraphError, ValueError):
    """A resource or payload bound was exceeded."""


class DatabaseImpactGraphConflictError(DatabaseImpactGraphError):
    """Duplicate identity with a conflicting payload."""


class DuckDBUnavailableError(DatabaseImpactGraphError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ImpactEdgeKind(str, Enum):
    """Closed vocabulary of materializable impact edge kinds."""

    CALLS = "calls"
    IMPORTS = "imports"
    TYPE = "type"
    IMPLEMENTS = "implements"
    ALIAS = "alias"
    RE_EXPORT = "re_export"
    TESTS = "tests"
    CONTRACT = "contract"
    PROOF = "proof"
    CONFIG = "config"
    DOCS = "docs"
    GENERATED = "generated"
    DYNAMIC = "dynamic"
    CROSS_LANGUAGE = "cross_language"
    DELETION = "deletion"
    SIMILARITY = "similarity"
    GRAPH_PROXIMITY = "graph_proximity"
    VECTOR_NOMINATION = "vector_nomination"
    HISTORY_NOMINATION = "history_nomination"
    DEPENDS_ON = "depends_on"

    @classmethod
    def coerce(cls, value: Any) -> "ImpactEdgeKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, ImpactEdgeKind] = {
            "call": cls.CALLS,
            "calls": cls.CALLS,
            "caller": cls.CALLS,
            "import": cls.IMPORTS,
            "imports": cls.IMPORTS,
            "type": cls.TYPE,
            "types": cls.TYPE,
            "typed_as": cls.TYPE,
            "implements": cls.IMPLEMENTS,
            "alias": cls.ALIAS,
            "aliases": cls.ALIAS,
            "reexport": cls.RE_EXPORT,
            "re_export": cls.RE_EXPORT,
            "reexports": cls.RE_EXPORT,
            "test": cls.TESTS,
            "tests": cls.TESTS,
            "contract": cls.CONTRACT,
            "contracts": cls.CONTRACT,
            "proof": cls.PROOF,
            "proofs": cls.PROOF,
            "config": cls.CONFIG,
            "docs": cls.DOCS,
            "documents": cls.DOCS,
            "generated": cls.GENERATED,
            "generated_code": cls.GENERATED,
            "dynamic": cls.DYNAMIC,
            "dynamic_call": cls.DYNAMIC,
            "cross_language": cls.CROSS_LANGUAGE,
            "cross-language": cls.CROSS_LANGUAGE,
            "deletion": cls.DELETION,
            "deleted": cls.DELETION,
            "similarity": cls.SIMILARITY,
            "graph_proximity": cls.GRAPH_PROXIMITY,
            "proximity": cls.GRAPH_PROXIMITY,
            "vector": cls.VECTOR_NOMINATION,
            "vector_nomination": cls.VECTOR_NOMINATION,
            "history": cls.HISTORY_NOMINATION,
            "history_nomination": cls.HISTORY_NOMINATION,
            "depends_on": cls.DEPENDS_ON,
            "dependency": cls.DEPENDS_ON,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseImpactGraphIntegrityError(
                f"unsupported impact edge kind: {value!r}"
            ) from exc


class EdgeAuthority(str, Enum):
    """Whether an edge may expand mandatory impact coverage."""

    AUTHORITATIVE = "authoritative"
    NOMINATION = "nomination"


class ConsumerDisposition(str, Enum):
    """Exactly one closed disposition per resolved consumer."""

    MIGRATE = "migrate"
    ADAPTER = "adapter"
    COMPATIBLE = "compatible"
    UPSTREAM = "upstream"
    ABSTAIN = "abstain"
    REVIEW_ONLY = "review_only"
    EXCLUDED = "excluded"
    FRONTIER = "frontier"


class ImpactCompleteness(str, Enum):
    """Whether the reverse impact closure claims full coverage."""

    COMPLETE = "complete"
    PARTIAL_WITH_FRONTIER = "partial_with_frontier"
    ABSTAINED = "abstained"


class FrontierKind(str, Enum):
    """Closed vocabulary of impact frontier categories."""

    DYNAMIC_CALL = "dynamic_call"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    PARSER_UNCERTAINTY = "parser_uncertainty"
    PARSE_FAILED = "parse_failed"
    GENERATED_CODE = "generated_code"
    DELETION = "deletion"
    CROSS_LANGUAGE = "cross_language"
    ALIAS_AMBIGUITY = "alias_ambiguity"
    REEXPORT_AMBIGUITY = "reexport_ambiguity"
    RESOURCE_BOUND = "resource_bound"
    UNRESOLVED_SYMBOL = "unresolved_symbol"
    OPEN = "open"

    @classmethod
    def coerce(cls, value: Any) -> "FrontierKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, FrontierKind] = {
            "dynamic": cls.DYNAMIC_CALL,
            "dynamic_call": cls.DYNAMIC_CALL,
            "unsupported": cls.UNSUPPORTED_LANGUAGE,
            "unsupported_language": cls.UNSUPPORTED_LANGUAGE,
            "parser": cls.PARSER_UNCERTAINTY,
            "parser_uncertainty": cls.PARSER_UNCERTAINTY,
            "parse_failed": cls.PARSE_FAILED,
            "failed": cls.PARSE_FAILED,
            "generated": cls.GENERATED_CODE,
            "generated_code": cls.GENERATED_CODE,
            "deletion": cls.DELETION,
            "deleted": cls.DELETION,
            "cross_language": cls.CROSS_LANGUAGE,
            "alias": cls.ALIAS_AMBIGUITY,
            "alias_ambiguity": cls.ALIAS_AMBIGUITY,
            "reexport": cls.REEXPORT_AMBIGUITY,
            "reexport_ambiguity": cls.REEXPORT_AMBIGUITY,
            "resource_bound": cls.RESOURCE_BOUND,
            "truncated": cls.RESOURCE_BOUND,
            "unresolved": cls.UNRESOLVED_SYMBOL,
            "unresolved_symbol": cls.UNRESOLVED_SYMBOL,
            "open": cls.OPEN,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseImpactGraphIntegrityError(
                f"unsupported frontier kind: {value!r}"
            ) from exc


class FrontierStatus(str, Enum):
    OPEN = "open"
    UNSUPPORTED = "unsupported"
    REVIEWED = "reviewed"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    if callable(_ast_duckdb_available):
        try:
            return bool(_ast_duckdb_available())
        except Exception:
            pass
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


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    raw = str(getattr(value, "value", value) or "").strip()
    try:
        return enum_cls(raw)
    except ValueError as exc:
        coerce = getattr(enum_cls, "coerce", None)
        if callable(coerce):
            return coerce(value)  # type: ignore[no-any-return]
        raise DatabaseImpactGraphIntegrityError(
            f"unsupported {name}: {value!r}"
        ) from exc


def _is_nomination_kind(kind: ImpactEdgeKind | str) -> bool:
    raw = kind.value if isinstance(kind, ImpactEdgeKind) else str(kind)
    return raw.casefold() in _NOMINATION_KINDS


def _symbol_key(
    *,
    snapshot_id: str,
    symbol_id: str = "",
    qualified_name: str = "",
    path: str = "",
) -> str:
    name = str(qualified_name or "").strip()
    sid = str(symbol_id or "").strip()
    if not name and not sid:
        raise DatabaseImpactGraphIntegrityError(
            "symbol identity requires symbol_id or qualified_name"
        )
    return _identity(
        "impact-symbol",
        {
            "snapshot_id": snapshot_id,
            "symbol_id": sid,
            "qualified_name": name,
            "path": path,
        },
    )


def _disposition_for_edge(
    kind: ImpactEdgeKind,
    *,
    is_seed: bool = False,
    is_frontier: bool = False,
    excluded: bool = False,
) -> ConsumerDisposition:
    if excluded:
        return ConsumerDisposition.EXCLUDED
    if is_frontier:
        return ConsumerDisposition.FRONTIER
    if is_seed:
        return ConsumerDisposition.UPSTREAM
    if _is_nomination_kind(kind):
        # Nominations never create mandatory migrate work.
        return ConsumerDisposition.REVIEW_ONLY
    if kind in {
        ImpactEdgeKind.TESTS,
        ImpactEdgeKind.CONTRACT,
        ImpactEdgeKind.PROOF,
        ImpactEdgeKind.CALLS,
        ImpactEdgeKind.IMPORTS,
        ImpactEdgeKind.DEPENDS_ON,
        ImpactEdgeKind.ALIAS,
        ImpactEdgeKind.RE_EXPORT,
        ImpactEdgeKind.IMPLEMENTS,
        ImpactEdgeKind.TYPE,
        ImpactEdgeKind.GENERATED,
    }:
        return ConsumerDisposition.MIGRATE
    if kind in {ImpactEdgeKind.CONFIG, ImpactEdgeKind.DOCS}:
        return ConsumerDisposition.REVIEW_ONLY
    if kind in {ImpactEdgeKind.DYNAMIC, ImpactEdgeKind.CROSS_LANGUAGE}:
        return ConsumerDisposition.FRONTIER
    if kind is ImpactEdgeKind.DELETION:
        return ConsumerDisposition.MIGRATE
    return ConsumerDisposition.ABSTAIN


def _bucket_for_edge(kind: ImpactEdgeKind) -> str:
    mapping: Mapping[ImpactEdgeKind, str] = {
        ImpactEdgeKind.CALLS: "callers",
        ImpactEdgeKind.IMPORTS: "imports",
        ImpactEdgeKind.TYPE: "types",
        ImpactEdgeKind.IMPLEMENTS: "types",
        ImpactEdgeKind.ALIAS: "aliases",
        ImpactEdgeKind.RE_EXPORT: "reexports",
        ImpactEdgeKind.TESTS: "tests",
        ImpactEdgeKind.CONTRACT: "contracts",
        ImpactEdgeKind.PROOF: "proofs",
        ImpactEdgeKind.CONFIG: "config",
        ImpactEdgeKind.DOCS: "docs",
        ImpactEdgeKind.GENERATED: "generated",
        ImpactEdgeKind.SIMILARITY: "nominations",
        ImpactEdgeKind.GRAPH_PROXIMITY: "nominations",
        ImpactEdgeKind.VECTOR_NOMINATION: "nominations",
        ImpactEdgeKind.HISTORY_NOMINATION: "nominations",
        ImpactEdgeKind.DEPENDS_ON: "callers",
        ImpactEdgeKind.DYNAMIC: "callers",
        ImpactEdgeKind.CROSS_LANGUAGE: "callers",
        ImpactEdgeKind.DELETION: "callers",
    }
    return mapping.get(kind, "callers")


def _tarjan_sccs(
    nodes: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Deterministic Tarjan SCCs over the consumer graph."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for successor in sorted(adjacency.get(node, ())):
            if successor not in indices:
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    # Multi-member SCCs first (transaction units), then stable order.
    multi = [item for item in components if len(item) > 1]
    multi.sort(key=lambda item: (len(item), item))
    return tuple(multi)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryBinding:
    """Exact snapshot/parser/policy/schema binding for one query result."""

    snapshot_id: str
    parser_id: str
    policy_id: str
    schema_id: str
    graph_version: str = DEFAULT_GRAPH_VERSION
    repository_id: str = ""
    tree_id: str = ""
    overlay_digest: str = ""
    recorded_at: str = ""

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
            "graph_version",
            _text(self.graph_version or DEFAULT_GRAPH_VERSION, "graph_version"),
        )
        object.__setattr__(
            self, "repository_id", str(self.repository_id or "").strip()
        )
        object.__setattr__(self, "tree_id", str(self.tree_id or "").strip())
        object.__setattr__(
            self, "overlay_digest", str(self.overlay_digest or "").strip()
        )
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": QUERY_BINDING_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "graph_version": self.graph_version,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "overlay_digest": self.overlay_digest,
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactSymbol:
    """One snapshot-bound symbol projection used as an impact node."""

    symbol_key: str
    snapshot_id: str
    symbol_id: str
    qualified_name: str
    path: str = ""
    language: str = ""
    symbol_kind: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "qualified_name", _text(self.qualified_name, "qualified_name")
        )
        object.__setattr__(
            self, "symbol_id", str(self.symbol_id or "").strip()
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "language", str(self.language or "").strip())
        object.__setattr__(
            self, "symbol_kind", str(self.symbol_kind or "").strip()
        )
        object.__setattr__(
            self, "fingerprint", str(self.fingerprint or "").strip()
        )
        claimed = str(self.symbol_key or "").strip()
        computed = _symbol_key(
            snapshot_id=self.snapshot_id,
            symbol_id=self.symbol_id,
            qualified_name=self.qualified_name,
            path=self.path,
        )
        if claimed and claimed != computed:
            # Accept pre-materialized keys that already embed the identity.
            object.__setattr__(self, "symbol_key", claimed)
        else:
            object.__setattr__(self, "symbol_key", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_key": self.symbol_key,
            "snapshot_id": self.snapshot_id,
            "symbol_id": self.symbol_id,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "language": self.language,
            "symbol_kind": self.symbol_kind,
            "fingerprint": self.fingerprint,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactEdge:
    """One directed dependency edge: source depends on / consumes target."""

    edge_id: str
    snapshot_id: str
    source_symbol_key: str
    target_symbol_key: str
    edge_kind: ImpactEdgeKind | str
    authority: EdgeAuthority | str = EdgeAuthority.AUTHORITATIVE
    semantic_authority: bool = False
    path: str = ""
    evidence_ref: str = ""
    reason: str = ""
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "source_symbol_key",
            _text(self.source_symbol_key, "source_symbol_key"),
        )
        object.__setattr__(
            self,
            "target_symbol_key",
            _text(self.target_symbol_key, "target_symbol_key"),
        )
        kind = ImpactEdgeKind.coerce(self.edge_kind)
        object.__setattr__(self, "edge_kind", kind)
        authority = _enum(self.authority, EdgeAuthority, "authority")
        if _is_nomination_kind(kind):
            authority = EdgeAuthority.NOMINATION
        object.__setattr__(self, "authority", authority)
        # Similarity / proximity never carry semantic authority.
        semantic = bool(self.semantic_authority)
        if authority is EdgeAuthority.NOMINATION or _is_nomination_kind(kind):
            semantic = False
        object.__setattr__(self, "semantic_authority", semantic)
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "evidence_ref", str(self.evidence_ref or "").strip()
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)
        body = dict(self.body or {})
        object.__setattr__(self, "body", MappingProxyType(body))
        computed = _identity(
            "impact-edge",
            {
                "schema": IMPACT_EDGE_SCHEMA,
                "snapshot_id": self.snapshot_id,
                "source_symbol_key": self.source_symbol_key,
                "target_symbol_key": self.target_symbol_key,
                "edge_kind": kind.value,
                "authority": authority.value,
                "path": self.path,
                "evidence_ref": self.evidence_ref,
            },
        )
        claimed = str(self.edge_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact edge identity does not match payload"
            )
        object.__setattr__(self, "edge_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        kind = (
            self.edge_kind.value
            if isinstance(self.edge_kind, ImpactEdgeKind)
            else str(self.edge_kind)
        )
        authority = (
            self.authority.value
            if isinstance(self.authority, EdgeAuthority)
            else str(self.authority)
        )
        return {
            "schema": IMPACT_EDGE_SCHEMA,
            "edge_id": self.edge_id,
            "snapshot_id": self.snapshot_id,
            "source_symbol_key": self.source_symbol_key,
            "target_symbol_key": self.target_symbol_key,
            "edge_kind": kind,
            "authority": authority,
            "semantic_authority": self.semantic_authority,
            "path": self.path,
            "evidence_ref": self.evidence_ref,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class ImpactFrontierRecord:
    """One open/unsupported/unresolved impact frontier endpoint."""

    frontier_id: str
    snapshot_id: str
    kind: FrontierKind | str
    status: FrontierStatus | str = FrontierStatus.OPEN
    subject_key: str = ""
    path: str = ""
    reason: str = ""
    blocks_automatic_repair: bool = True
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        kind = FrontierKind.coerce(self.kind)
        object.__setattr__(self, "kind", kind)
        status = _enum(self.status, FrontierStatus, "status")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "subject_key", str(self.subject_key or "").strip()
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        blocks = bool(self.blocks_automatic_repair)
        if status in {FrontierStatus.OPEN, FrontierStatus.UNSUPPORTED}:
            blocks = True
        if status is FrontierStatus.CLOSED:
            blocks = False
        object.__setattr__(self, "blocks_automatic_repair", blocks)
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)
        body = dict(self.body or {})
        object.__setattr__(self, "body", MappingProxyType(body))
        computed = _identity(
            "impact-frontier",
            {
                "schema": IMPACT_FRONTIER_SCHEMA,
                "snapshot_id": self.snapshot_id,
                "kind": kind.value,
                "status": status.value
                if isinstance(status, FrontierStatus)
                else str(status),
                "subject_key": self.subject_key,
                "path": self.path,
                "reason": self.reason,
            },
        )
        claimed = str(self.frontier_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact frontier identity does not match payload"
            )
        object.__setattr__(self, "frontier_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        kind = (
            self.kind.value
            if isinstance(self.kind, FrontierKind)
            else str(self.kind)
        )
        status = (
            self.status.value
            if isinstance(self.status, FrontierStatus)
            else str(self.status)
        )
        return {
            "schema": IMPACT_FRONTIER_SCHEMA,
            "frontier_id": self.frontier_id,
            "snapshot_id": self.snapshot_id,
            "kind": kind,
            "status": status,
            "subject_key": self.subject_key,
            "path": self.path,
            "reason": self.reason,
            "blocks_automatic_repair": self.blocks_automatic_repair,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactConsumerRecord:
    """One resolved consumer with exactly one disposition."""

    consumer_id: str
    symbol_key: str
    qualified_name: str
    disposition: ConsumerDisposition | str
    depth: int = 0
    path: str = ""
    edge_ids: tuple[str, ...] = ()
    edge_kinds: tuple[str, ...] = ()
    mandatory: bool = True
    is_seed: bool = False
    semantic_authority: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "symbol_key", _text(self.symbol_key, "symbol_key")
        )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, "qualified_name"),
        )
        disposition = _enum(
            self.disposition, ConsumerDisposition, "disposition"
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "depth", _nonneg_int(int(self.depth), "depth")
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        edges = tuple(
            str(item).strip()
            for item in self.edge_ids
            if str(item).strip()
        )
        if len(edges) > MAX_EDGES:
            raise DatabaseImpactGraphBoundsError(
                f"consumer edge_ids exceed {MAX_EDGES}"
            )
        object.__setattr__(self, "edge_ids", edges)
        kinds = tuple(
            str(item).strip()
            for item in self.edge_kinds
            if str(item).strip()
        )
        object.__setattr__(self, "edge_kinds", kinds)
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(self, "is_seed", bool(self.is_seed))
        object.__setattr__(
            self, "semantic_authority", bool(self.semantic_authority)
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        computed = _identity(
            "impact-consumer",
            {
                "schema": IMPACT_CONSUMER_SCHEMA,
                "symbol_key": self.symbol_key,
                "qualified_name": self.qualified_name,
                "disposition": disposition.value
                if isinstance(disposition, ConsumerDisposition)
                else str(disposition),
                "depth": self.depth,
                "path": self.path,
                "edge_ids": list(edges),
            },
        )
        claimed = str(self.consumer_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact consumer identity does not match payload"
            )
        object.__setattr__(self, "consumer_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        disposition = (
            self.disposition.value
            if isinstance(self.disposition, ConsumerDisposition)
            else str(self.disposition)
        )
        return {
            "schema": IMPACT_CONSUMER_SCHEMA,
            "consumer_id": self.consumer_id,
            "symbol_key": self.symbol_key,
            "qualified_name": self.qualified_name,
            "disposition": disposition,
            "depth": self.depth,
            "path": self.path,
            "edge_ids": list(self.edge_ids),
            "edge_kinds": list(self.edge_kinds),
            "mandatory": self.mandatory,
            "is_seed": self.is_seed,
            "semantic_authority": self.semantic_authority,
            "reason": self.reason,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ImpactSCC:
    """One strongly connected consumer group treated as a transaction unit."""

    scc_id: str
    member_consumer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        members = tuple(
            str(item).strip()
            for item in self.member_consumer_ids
            if str(item).strip()
        )
        if not members:
            raise DatabaseImpactGraphIntegrityError(
                "scc requires at least one member"
            )
        object.__setattr__(self, "member_consumer_ids", members)
        computed = _identity(
            "impact-scc",
            {"member_consumer_ids": list(members)},
        )
        claimed = str(self.scc_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "scc identity does not match payload"
            )
        object.__setattr__(self, "scc_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scc_id": self.scc_id,
            "member_consumer_ids": list(self.member_consumer_ids),
        }


@dataclass(frozen=True)
class ImpactClosure:
    """Reverse transitive impact closure with dispositions and frontiers.

    Interface: ``ImpactClosure@1``.
    """

    interface: str = IMPACT_CLOSURE_INTERFACE
    schema: str = IMPACT_CLOSURE_SCHEMA
    closure_id: str = ""
    binding: QueryBinding | None = None
    seeds: tuple[str, ...] = ()
    completeness: ImpactCompleteness | str = ImpactCompleteness.ABSTAINED
    consumers: tuple[ImpactConsumerRecord, ...] = ()
    frontiers: tuple[ImpactFrontierRecord, ...] = ()
    sccs: tuple[ImpactSCC, ...] = ()
    automatic_repair_allowed: bool = False
    truncated: bool = False
    page_offset: int = 0
    page_limit: int = 0
    total_consumer_count: int = 0
    evidence_refs: tuple[str, ...] = ()
    task_id: str = ""
    mutation_id: str = ""
    freshness: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interface",
            _text(self.interface or IMPACT_CLOSURE_INTERFACE, "interface"),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or IMPACT_CLOSURE_SCHEMA, "schema"),
        )
        if self.binding is None or not isinstance(self.binding, QueryBinding):
            raise DatabaseImpactGraphIntegrityError(
                "impact closure requires a QueryBinding"
            )
        seeds = tuple(
            str(item).strip() for item in self.seeds if str(item).strip()
        )
        if len(seeds) > MAX_SEEDS:
            raise DatabaseImpactGraphBoundsError(
                f"seed count exceeds {MAX_SEEDS}"
            )
        object.__setattr__(self, "seeds", seeds)
        completeness = _enum(
            self.completeness, ImpactCompleteness, "completeness"
        )
        object.__setattr__(self, "completeness", completeness)
        consumers = tuple(self.consumers or ())
        if len(consumers) > MAX_CONSUMERS:
            raise DatabaseImpactGraphBoundsError(
                f"consumer count exceeds {MAX_CONSUMERS}"
            )
        # Exactly one disposition per resolved consumer_id / symbol_key.
        seen_ids: set[str] = set()
        seen_keys: set[str] = set()
        for item in consumers:
            if not isinstance(item, ImpactConsumerRecord):
                raise DatabaseImpactGraphIntegrityError(
                    "consumers must be ImpactConsumerRecord"
                )
            if item.consumer_id in seen_ids:
                raise DatabaseImpactGraphIntegrityError(
                    "duplicate consumer_id in impact closure"
                )
            if item.symbol_key in seen_keys:
                raise DatabaseImpactGraphIntegrityError(
                    "resolved consumer symbol_key must appear once "
                    "(exactly one disposition)"
                )
            seen_ids.add(item.consumer_id)
            seen_keys.add(item.symbol_key)
        object.__setattr__(self, "consumers", consumers)
        frontiers = tuple(self.frontiers or ())
        if len(frontiers) > MAX_FRONTIER:
            raise DatabaseImpactGraphBoundsError(
                f"frontier count exceeds {MAX_FRONTIER}"
            )
        object.__setattr__(self, "frontiers", frontiers)
        sccs = tuple(self.sccs or ())
        object.__setattr__(self, "sccs", sccs)
        blocking = any(
            item.blocks_automatic_repair
            and (
                item.status is FrontierStatus.OPEN
                or item.status is FrontierStatus.UNSUPPORTED
                or str(getattr(item.status, "value", item.status))
                in {"open", "unsupported"}
            )
            for item in frontiers
        )
        # FRONTIER consumer dispositions are equivalent to an open frontier.
        consumer_blocks = any(
            item.disposition is ConsumerDisposition.FRONTIER
            or str(getattr(item.disposition, "value", item.disposition))
            == ConsumerDisposition.FRONTIER.value
            for item in consumers
        )
        truncated = bool(self.truncated)
        object.__setattr__(self, "truncated", truncated)
        allowed = bool(self.automatic_repair_allowed)
        if (
            blocking
            or consumer_blocks
            or truncated
            or completeness is not ImpactCompleteness.COMPLETE
        ):
            allowed = False
        object.__setattr__(self, "automatic_repair_allowed", allowed)
        object.__setattr__(
            self,
            "page_offset",
            _nonneg_int(int(self.page_offset), "page_offset"),
        )
        object.__setattr__(
            self, "page_limit", _nonneg_int(int(self.page_limit), "page_limit")
        )
        object.__setattr__(
            self,
            "total_consumer_count",
            _nonneg_int(
                int(self.total_consumer_count or len(consumers)),
                "total_consumer_count",
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(
                str(item).strip()
                for item in self.evidence_refs
                if str(item).strip()
            ),
        )
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(
            self, "mutation_id", str(self.mutation_id or "").strip()
        )
        object.__setattr__(
            self, "freshness", str(self.freshness or "").strip() or "fresh"
        )
        if completeness is ImpactCompleteness.COMPLETE:
            if blocking or truncated:
                raise DatabaseImpactGraphIntegrityError(
                    "complete impact closure cannot retain open frontiers "
                    "or truncation"
                )
            open_like = [
                item
                for item in frontiers
                if str(getattr(item.status, "value", item.status))
                in {"open", "unsupported"}
            ]
            if open_like:
                raise DatabaseImpactGraphIntegrityError(
                    "complete impact closure forbids open/unsupported "
                    "frontiers"
                )
        if completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            if not frontiers and not truncated:
                raise DatabaseImpactGraphIntegrityError(
                    "partial impact closure requires an explicit frontier "
                    "or truncation"
                )
        computed = _identity(
            "impact-closure",
            {
                "schema": self.schema,
                "binding": self.binding.to_dict(),
                "seeds": list(seeds),
                "completeness": completeness.value
                if isinstance(completeness, ImpactCompleteness)
                else str(completeness),
                "consumer_ids": [item.consumer_id for item in consumers],
                "frontier_ids": [item.frontier_id for item in frontiers],
                "task_id": self.task_id,
                "mutation_id": self.mutation_id,
            },
        )
        claimed = str(self.closure_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "impact closure identity does not match payload"
            )
        object.__setattr__(self, "closure_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        completeness = (
            self.completeness.value
            if isinstance(self.completeness, ImpactCompleteness)
            else str(self.completeness)
        )
        return {
            "interface": self.interface,
            "schema": self.schema,
            "closure_id": self.closure_id,
            "binding": self.binding.to_dict() if self.binding else None,
            "seeds": list(self.seeds),
            "completeness": completeness,
            "consumers": [item.to_dict() for item in self.consumers],
            "frontiers": [item.to_dict() for item in self.frontiers],
            "sccs": [item.to_dict() for item in self.sccs],
            "automatic_repair_allowed": self.automatic_repair_allowed,
            "truncated": self.truncated,
            "page_offset": self.page_offset,
            "page_limit": self.page_limit,
            "total_consumer_count": self.total_consumer_count,
            "evidence_refs": list(self.evidence_refs),
            "task_id": self.task_id,
            "mutation_id": self.mutation_id,
            "freshness": self.freshness,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ChangedSymbolNeighborhood:
    """Bounded neighborhood around changed symbols for a mutation/task.

    Interface: ``ChangedSymbolNeighborhood@1``.
    """

    interface: str = CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE
    schema: str = CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA
    neighborhood_id: str = ""
    binding: QueryBinding | None = None
    seeds: tuple[str, ...] = ()
    callers: tuple[ImpactConsumerRecord, ...] = ()
    callees: tuple[ImpactConsumerRecord, ...] = ()
    imports: tuple[ImpactConsumerRecord, ...] = ()
    types: tuple[ImpactConsumerRecord, ...] = ()
    tests: tuple[ImpactConsumerRecord, ...] = ()
    contracts: tuple[ImpactConsumerRecord, ...] = ()
    proofs: tuple[ImpactConsumerRecord, ...] = ()
    config: tuple[ImpactConsumerRecord, ...] = ()
    docs: tuple[ImpactConsumerRecord, ...] = ()
    aliases: tuple[ImpactConsumerRecord, ...] = ()
    reexports: tuple[ImpactConsumerRecord, ...] = ()
    generated: tuple[ImpactConsumerRecord, ...] = ()
    nominations: tuple[ImpactConsumerRecord, ...] = ()
    frontiers: tuple[ImpactFrontierRecord, ...] = ()
    automatic_repair_allowed: bool = False
    truncated: bool = False
    page_offset: int = 0
    page_limit: int = 0
    task_id: str = ""
    mutation_id: str = ""
    freshness: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interface",
            _text(
                self.interface or CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE,
                "interface",
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _text(
                self.schema or CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA, "schema"
            ),
        )
        if self.binding is None or not isinstance(self.binding, QueryBinding):
            raise DatabaseImpactGraphIntegrityError(
                "changed neighborhood requires a QueryBinding"
            )
        seeds = tuple(
            str(item).strip() for item in self.seeds if str(item).strip()
        )
        object.__setattr__(self, "seeds", seeds)
        # Validate nomination authority: nominations never claim semantics.
        for item in self.nominations or ():
            if item.semantic_authority:
                raise DatabaseImpactGraphIntegrityError(
                    "similarity/graph proximity nominations cannot claim "
                    "semantic authority"
                )
        blocking = any(
            item.blocks_automatic_repair
            and str(getattr(item.status, "value", item.status))
            in {"open", "unsupported"}
            for item in (self.frontiers or ())
        )
        truncated = bool(self.truncated)
        object.__setattr__(self, "truncated", truncated)
        allowed = bool(self.automatic_repair_allowed)
        if blocking or truncated:
            allowed = False
        object.__setattr__(self, "automatic_repair_allowed", allowed)
        object.__setattr__(
            self,
            "page_offset",
            _nonneg_int(int(self.page_offset), "page_offset"),
        )
        object.__setattr__(
            self, "page_limit", _nonneg_int(int(self.page_limit), "page_limit")
        )
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        object.__setattr__(
            self, "mutation_id", str(self.mutation_id or "").strip()
        )
        object.__setattr__(
            self, "freshness", str(self.freshness or "").strip() or "fresh"
        )
        for name in _NEIGHBORHOOD_BUCKETS + ("frontiers",):
            value = getattr(self, name)
            object.__setattr__(self, name, tuple(value or ()))
        computed = _identity(
            "changed-neighborhood",
            {
                "schema": self.schema,
                "binding": self.binding.to_dict(),
                "seeds": list(seeds),
                "buckets": {
                    name: [
                        item.consumer_id
                        for item in getattr(self, name)
                        if hasattr(item, "consumer_id")
                    ]
                    for name in _NEIGHBORHOOD_BUCKETS
                },
                "frontier_ids": [
                    item.frontier_id for item in self.frontiers
                ],
                "task_id": self.task_id,
                "mutation_id": self.mutation_id,
            },
        )
        claimed = str(self.neighborhood_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "changed neighborhood identity does not match payload"
            )
        object.__setattr__(self, "neighborhood_id", claimed or computed)

    def all_resolved_consumers(self) -> tuple[ImpactConsumerRecord, ...]:
        """Return every resolved consumer across neighborhood buckets."""

        items: list[ImpactConsumerRecord] = []
        for name in _NEIGHBORHOOD_BUCKETS:
            items.extend(getattr(self, name) or ())
        return tuple(items)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "interface": self.interface,
            "schema": self.schema,
            "neighborhood_id": self.neighborhood_id,
            "binding": self.binding.to_dict() if self.binding else None,
            "seeds": list(self.seeds),
            "automatic_repair_allowed": self.automatic_repair_allowed,
            "truncated": self.truncated,
            "page_offset": self.page_offset,
            "page_limit": self.page_limit,
            "task_id": self.task_id,
            "mutation_id": self.mutation_id,
            "freshness": self.freshness,
            "authority": AUTHORITY_CLASS,
        }
        for name in _NEIGHBORHOOD_BUCKETS:
            payload[name] = [item.to_dict() for item in getattr(self, name)]
        payload["frontiers"] = [item.to_dict() for item in self.frontiers]
        return payload


@dataclass(frozen=True)
class MaterializeResult:
    """Outcome of materializing impact edges for one snapshot."""

    materialization_id: str
    snapshot_id: str
    parser_id: str
    policy_id: str
    schema_id: str
    edge_count: int
    symbol_count: int
    frontier_count: int
    complete: bool
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("edge_count", "symbol_count", "frontier_count"):
            object.__setattr__(
                self, name, _nonneg_int(int(getattr(self, name)), name)
            )
        object.__setattr__(self, "complete", bool(self.complete))
        stamp = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", stamp)
        body = dict(self.body or {})
        object.__setattr__(self, "body", MappingProxyType(body))
        computed = _identity(
            "impact-materialization",
            {
                "snapshot_id": self.snapshot_id,
                "parser_id": self.parser_id,
                "policy_id": self.policy_id,
                "schema_id": self.schema_id,
                "edge_count": self.edge_count,
                "symbol_count": self.symbol_count,
                "frontier_count": self.frontier_count,
                "complete": self.complete,
                "recorded_at": self.recorded_at,
            },
        )
        claimed = str(self.materialization_id or "").strip()
        if claimed and claimed != computed:
            raise DatabaseImpactGraphIntegrityError(
                "materialization identity does not match payload"
            )
        object.__setattr__(self, "materialization_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "materialization_id": self.materialization_id,
            "snapshot_id": self.snapshot_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "edge_count": self.edge_count,
            "symbol_count": self.symbol_count,
            "frontier_count": self.frontier_count,
            "complete": self.complete,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseImpactGraph:
    """Persist and query snapshot-bound symbol impact evidence in DuckDB.

    Interface: ``DatabaseImpactGraph@1``.
    """

    INTERFACE: Final[str] = DATABASE_IMPACT_GRAPH_INTERFACE
    SCHEMA: Final[str] = DATABASE_IMPACT_GRAPH_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        parser_id: str = DEFAULT_PARSER_ID,
        policy_id: str = DEFAULT_POLICY_ID,
        graph_version: str = DEFAULT_GRAPH_VERSION,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseImpactGraph; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._parser_id = _text(parser_id or DEFAULT_PARSER_ID, "parser_id")
        self._policy_id = _text(policy_id or DEFAULT_POLICY_ID, "policy_id")
        self._graph_version = _text(
            graph_version or DEFAULT_GRAPH_VERSION, "graph_version"
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
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def graph_version(self) -> str:
        return self._graph_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

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
                ("parser_id", self._parser_id),
                ("policy_id", self._policy_id),
                ("graph_version", self._graph_version),
                ("authority", AUTHORITY_CLASS),
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

    def binding_for(
        self,
        snapshot_id: str,
        *,
        parser_id: str | None = None,
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        overlay_digest: str = "",
    ) -> QueryBinding:
        return QueryBinding(
            snapshot_id=_text(snapshot_id, "snapshot_id"),
            parser_id=_text(parser_id or self._parser_id, "parser_id"),
            policy_id=_text(policy_id or self._policy_id, "policy_id"),
            schema_id=DATABASE_IMPACT_GRAPH_SCHEMA,
            graph_version=self._graph_version,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
        )

    # -- write API -----------------------------------------------------------

    def upsert_symbol(
        self,
        *,
        snapshot_id: str,
        qualified_name: str,
        symbol_id: str = "",
        path: str = "",
        language: str = "",
        symbol_kind: str = "",
        fingerprint: str = "",
    ) -> ImpactSymbol:
        symbol = ImpactSymbol(
            symbol_key="",
            snapshot_id=snapshot_id,
            symbol_id=symbol_id or qualified_name,
            qualified_name=qualified_name,
            path=path,
            language=language,
            symbol_kind=symbol_kind,
            fingerprint=fingerprint,
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT OR REPLACE INTO impact_symbols (
                    symbol_key, snapshot_id, symbol_id, qualified_name, path,
                    language, symbol_kind, fingerprint, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol.symbol_key,
                    symbol.snapshot_id,
                    symbol.symbol_id,
                    symbol.qualified_name,
                    symbol.path,
                    symbol.language,
                    symbol.symbol_kind,
                    symbol.fingerprint,
                    _canonical_json(symbol.to_dict()),
                ],
            )
            self._commit_if_idle(connection)
        return symbol

    def upsert_edge(
        self,
        *,
        snapshot_id: str,
        source: ImpactSymbol | Mapping[str, Any] | str,
        target: ImpactSymbol | Mapping[str, Any] | str,
        edge_kind: ImpactEdgeKind | str,
        path: str = "",
        evidence_ref: str = "",
        reason: str = "",
        authority: EdgeAuthority | str | None = None,
        semantic_authority: bool | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> ImpactEdge:
        source_symbol = self._coerce_symbol(snapshot_id, source)
        target_symbol = self._coerce_symbol(snapshot_id, target)
        kind = ImpactEdgeKind.coerce(edge_kind)
        if authority is None:
            authority = (
                EdgeAuthority.NOMINATION
                if _is_nomination_kind(kind)
                else EdgeAuthority.AUTHORITATIVE
            )
        if semantic_authority is None:
            semantic_authority = (
                False
                if _is_nomination_kind(kind)
                else authority == EdgeAuthority.AUTHORITATIVE
                or authority == EdgeAuthority.AUTHORITATIVE.value
            )
        edge = ImpactEdge(
            edge_id="",
            snapshot_id=snapshot_id,
            source_symbol_key=source_symbol.symbol_key,
            target_symbol_key=target_symbol.symbol_key,
            edge_kind=kind,
            authority=authority,
            semantic_authority=bool(semantic_authority),
            path=path or source_symbol.path,
            evidence_ref=evidence_ref,
            reason=reason,
            body=dict(body or {}),
        )
        with self._lock:
            connection = self._require()
            self._persist_symbol(connection, source_symbol)
            self._persist_symbol(connection, target_symbol)
            connection.execute(
                """
                INSERT OR REPLACE INTO impact_edges (
                    edge_id, snapshot_id, source_symbol_key, target_symbol_key,
                    edge_kind, authority, semantic_authority, path,
                    evidence_ref, reason, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    edge.edge_id,
                    edge.snapshot_id,
                    edge.source_symbol_key,
                    edge.target_symbol_key,
                    edge.edge_kind.value
                    if isinstance(edge.edge_kind, ImpactEdgeKind)
                    else str(edge.edge_kind),
                    edge.authority.value
                    if isinstance(edge.authority, EdgeAuthority)
                    else str(edge.authority),
                    1 if edge.semantic_authority else 0,
                    edge.path,
                    edge.evidence_ref,
                    edge.reason,
                    edge.recorded_at,
                    _canonical_json(dict(edge.body)),
                ],
            )
            self._commit_if_idle(connection)
        return edge

    def upsert_frontier(
        self,
        *,
        snapshot_id: str,
        kind: FrontierKind | str,
        reason: str,
        status: FrontierStatus | str = FrontierStatus.OPEN,
        subject_key: str = "",
        path: str = "",
        blocks_automatic_repair: bool = True,
        body: Mapping[str, Any] | None = None,
    ) -> ImpactFrontierRecord:
        frontier = ImpactFrontierRecord(
            frontier_id="",
            snapshot_id=snapshot_id,
            kind=kind,
            status=status,
            subject_key=subject_key,
            path=path,
            reason=reason,
            blocks_automatic_repair=blocks_automatic_repair,
            body=dict(body or {}),
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT OR REPLACE INTO impact_frontiers (
                    frontier_id, snapshot_id, kind, subject_key, path, status,
                    reason, blocks_automatic_repair, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    frontier.frontier_id,
                    frontier.snapshot_id,
                    frontier.kind.value
                    if isinstance(frontier.kind, FrontierKind)
                    else str(frontier.kind),
                    frontier.subject_key,
                    frontier.path,
                    frontier.status.value
                    if isinstance(frontier.status, FrontierStatus)
                    else str(frontier.status),
                    frontier.reason,
                    1 if frontier.blocks_automatic_repair else 0,
                    frontier.recorded_at,
                    _canonical_json(dict(frontier.body)),
                ],
            )
            self._commit_if_idle(connection)
        return frontier

    def materialize_from_ast_index(
        self,
        ast_index: Any,
        snapshot_id: str,
        *,
        parser_id: str | None = None,
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        clear_existing: bool = True,
    ) -> MaterializeResult:
        """Project DuckDBASTIndex symbols/calls/imports/frontiers into edges."""

        selected = _text(snapshot_id, "snapshot_id")
        selected_parser = _text(parser_id or self._parser_id, "parser_id")
        selected_policy = _text(policy_id or self._policy_id, "policy_id")
        if not hasattr(ast_index, "list_symbols"):
            raise DatabaseImpactGraphIntegrityError(
                "ast_index must provide list_symbols"
            )

        symbols = list(ast_index.list_symbols(selected))
        imports = (
            list(ast_index.list_imports(selected))
            if hasattr(ast_index, "list_imports")
            else []
        )
        calls = (
            list(ast_index.list_calls(selected))
            if hasattr(ast_index, "list_calls")
            else []
        )
        frontiers_raw = (
            list(ast_index.list_frontiers(selected))
            if hasattr(ast_index, "list_frontiers")
            else []
        )
        files = (
            list(ast_index.list_files(selected))
            if hasattr(ast_index, "list_files")
            else []
        )

        # Prefer live index parser when available.
        if hasattr(ast_index, "parser_id"):
            selected_parser = _text(
                parser_id or getattr(ast_index, "parser_id"), "parser_id"
            )

        with self._lock:
            connection = self._require()
            if clear_existing:
                self._clear_snapshot(connection, selected)

            symbol_by_id: dict[str, ImpactSymbol] = {}
            symbol_by_name: dict[str, list[ImpactSymbol]] = {}
            for item in symbols:
                if hasattr(item, "to_dict"):
                    mapping = item.to_dict()
                elif isinstance(item, Mapping):
                    mapping = dict(item)
                else:
                    continue
                symbol = ImpactSymbol(
                    symbol_key="",
                    snapshot_id=selected,
                    symbol_id=str(mapping.get("symbol_id") or ""),
                    qualified_name=str(mapping.get("qualified_name") or ""),
                    path=str(mapping.get("path") or ""),
                    language=str(mapping.get("language") or ""),
                    symbol_kind=str(mapping.get("symbol_kind") or ""),
                    fingerprint=str(mapping.get("fingerprint") or ""),
                )
                self._persist_symbol(connection, symbol)
                symbol_by_id[symbol.symbol_id] = symbol
                symbol_by_name.setdefault(symbol.qualified_name, []).append(
                    symbol
                )
                # Also index bare leaf names for partial call resolution.
                leaf = symbol.qualified_name.rsplit(".", 1)[-1]
                if leaf != symbol.qualified_name:
                    symbol_by_name.setdefault(leaf, []).append(symbol)

            edge_count = 0
            frontier_records: list[ImpactFrontierRecord] = []

            # Import edges: importer file/module depends on imported module.
            file_module_symbols: dict[str, ImpactSymbol] = {}
            for file_row in files:
                path = str(file_row.get("path") or "")
                if not path:
                    continue
                module_name = path.replace("/", ".").removesuffix(".py")
                module_symbol = ImpactSymbol(
                    symbol_key="",
                    snapshot_id=selected,
                    symbol_id=f"module:{path}",
                    qualified_name=module_name,
                    path=path,
                    language=str(file_row.get("language") or ""),
                    symbol_kind="module",
                )
                self._persist_symbol(connection, module_symbol)
                file_module_symbols[path] = module_symbol
                symbol_by_name.setdefault(module_name, []).append(
                    module_symbol
                )

            for item in imports:
                path = str(item.get("path") or "")
                module_name = str(item.get("module_name") or "").strip()
                if not module_name:
                    continue
                source = file_module_symbols.get(path)
                if source is None:
                    source = ImpactSymbol(
                        symbol_key="",
                        snapshot_id=selected,
                        symbol_id=f"module:{path or module_name}",
                        qualified_name=path or module_name,
                        path=path,
                        symbol_kind="module",
                    )
                    self._persist_symbol(connection, source)
                # Resolve imported module when possible.
                target = self._resolve_name(
                    symbol_by_name, module_name, snapshot_id=selected
                )
                if target is None:
                    target = ImpactSymbol(
                        symbol_key="",
                        snapshot_id=selected,
                        symbol_id=f"import-ref:{module_name}",
                        qualified_name=module_name,
                        path="",
                        symbol_kind="import_ref",
                    )
                    self._persist_symbol(connection, target)
                    frontier_records.append(
                        ImpactFrontierRecord(
                            frontier_id="",
                            snapshot_id=selected,
                            kind=FrontierKind.UNRESOLVED_SYMBOL,
                            status=FrontierStatus.OPEN,
                            subject_key=target.symbol_key,
                            path=path,
                            reason=f"unresolved_import:{module_name}",
                        )
                    )
                edge = ImpactEdge(
                    edge_id="",
                    snapshot_id=selected,
                    source_symbol_key=source.symbol_key,
                    target_symbol_key=target.symbol_key,
                    edge_kind=ImpactEdgeKind.IMPORTS,
                    authority=EdgeAuthority.AUTHORITATIVE,
                    semantic_authority=True,
                    path=path,
                    evidence_ref=str(item.get("import_id") or ""),
                    reason="ast_import",
                )
                self._persist_edge(connection, edge)
                edge_count += 1

            # Prefer raw call strings from parse-cache facts so cross-file
            # callees can be resolved by qualified / leaf name.
            fact_calls: list[tuple[str, str, str, str]] = []
            # (path, caller_name, callee_name, evidence_ref)
            if hasattr(ast_index, "get_parse_cache_entry"):
                for file_row in files:
                    path = str(file_row.get("path") or "")
                    digest = str(file_row.get("content_digest") or "")
                    if not digest:
                        continue
                    try:
                        cached = ast_index.get_parse_cache_entry(
                            digest, parser_id=selected_parser
                        )
                    except Exception:
                        cached = None
                    if not cached:
                        continue
                    facts = cached.get("facts") if isinstance(cached, Mapping) else None
                    if not isinstance(facts, Mapping):
                        continue
                    for call_stmt in facts.get("calls") or ():
                        text = str(call_stmt or "").strip()
                        if "->" not in text:
                            continue
                        caller_name, _, callee_name = text.partition("->")
                        fact_calls.append(
                            (
                                path,
                                caller_name.strip(),
                                callee_name.strip(),
                                text,
                            )
                        )

            if fact_calls:
                for path, caller_name, callee_name, evidence in fact_calls:
                    caller = self._resolve_name(
                        symbol_by_name, caller_name, snapshot_id=selected
                    )
                    if caller is None and caller_name:
                        # Module-level owner may be empty or file module.
                        caller = file_module_symbols.get(path)
                    if caller is None:
                        caller = ImpactSymbol(
                            symbol_key="",
                            snapshot_id=selected,
                            symbol_id=f"caller-ref:{path}:{caller_name}",
                            qualified_name=caller_name or f"<caller:{path}>",
                            path=path,
                            symbol_kind="call_ref",
                        )
                        self._persist_symbol(connection, caller)
                        symbol_by_name.setdefault(
                            caller.qualified_name, []
                        ).append(caller)

                    dynamic = False
                    callee = self._resolve_name(
                        symbol_by_name, callee_name, snapshot_id=selected
                    )
                    if callee is None:
                        # Attribute style: service.dispatch -> Service.dispatch
                        leaf = callee_name.rsplit(".", 1)[-1]
                        attr_matches = list(symbol_by_name.get(leaf) or ())
                        # Prefer method-qualified names ending with .leaf
                        qualified_matches = [
                            item
                            for items in symbol_by_name.values()
                            for item in items
                            if item.qualified_name.endswith(f".{leaf}")
                        ]
                        candidates = attr_matches + [
                            item
                            for item in qualified_matches
                            if item not in attr_matches
                        ]
                        # Unique by symbol_key
                        uniq: dict[str, ImpactSymbol] = {
                            item.symbol_key: item for item in candidates
                        }
                        if len(uniq) == 1:
                            callee = next(iter(uniq.values()))
                        else:
                            dynamic = True
                            callee = ImpactSymbol(
                                symbol_key="",
                                snapshot_id=selected,
                                symbol_id=f"callee-ref:{path}:{callee_name}",
                                qualified_name=callee_name or "<dynamic>",
                                path=path,
                                symbol_kind="call_ref",
                            )
                            self._persist_symbol(connection, callee)
                            frontier_records.append(
                                ImpactFrontierRecord(
                                    frontier_id="",
                                    snapshot_id=selected,
                                    kind=FrontierKind.DYNAMIC_CALL,
                                    status=FrontierStatus.OPEN,
                                    subject_key=callee.symbol_key,
                                    path=path,
                                    reason=(
                                        f"unresolved_or_dynamic_callee:"
                                        f"{callee_name or '<empty>'}"
                                    ),
                                )
                            )
                    kind = (
                        ImpactEdgeKind.DYNAMIC
                        if dynamic
                        else ImpactEdgeKind.CALLS
                    )
                    edge = ImpactEdge(
                        edge_id="",
                        snapshot_id=selected,
                        source_symbol_key=caller.symbol_key,
                        target_symbol_key=callee.symbol_key,
                        edge_kind=kind,
                        authority=EdgeAuthority.AUTHORITATIVE,
                        semantic_authority=not dynamic,
                        path=path,
                        evidence_ref=evidence,
                        reason="ast_call" if not dynamic else "dynamic_call",
                    )
                    self._persist_edge(connection, edge)
                    edge_count += 1
            else:
                # Fall back to already-projected call rows (ids only).
                for item in calls:
                    path = str(item.get("path") or "")
                    caller_id = str(item.get("caller_symbol_id") or "")
                    callee_id = str(item.get("callee_symbol_id") or "")
                    caller = symbol_by_id.get(caller_id)
                    callee = symbol_by_id.get(callee_id)
                    dynamic = False
                    if caller is None:
                        caller = ImpactSymbol(
                            symbol_key="",
                            snapshot_id=selected,
                            symbol_id=caller_id or f"caller-ref:{path}",
                            qualified_name=caller_id or f"<caller:{path}>",
                            path=path,
                            symbol_kind="call_ref",
                        )
                        self._persist_symbol(connection, caller)
                    if callee is None:
                        dynamic = True
                        callee = ImpactSymbol(
                            symbol_key="",
                            snapshot_id=selected,
                            symbol_id=callee_id or f"callee-ref:{path}",
                            qualified_name=callee_id or "<dynamic>",
                            path=path,
                            symbol_kind="call_ref",
                        )
                        self._persist_symbol(connection, callee)
                        frontier_records.append(
                            ImpactFrontierRecord(
                                frontier_id="",
                                snapshot_id=selected,
                                kind=FrontierKind.DYNAMIC_CALL,
                                status=FrontierStatus.OPEN,
                                subject_key=callee.symbol_key,
                                path=path,
                                reason="unresolved_or_dynamic_callee",
                            )
                        )
                    kind = (
                        ImpactEdgeKind.DYNAMIC
                        if dynamic
                        else ImpactEdgeKind.CALLS
                    )
                    edge = ImpactEdge(
                        edge_id="",
                        snapshot_id=selected,
                        source_symbol_key=caller.symbol_key,
                        target_symbol_key=callee.symbol_key,
                        edge_kind=kind,
                        authority=EdgeAuthority.AUTHORITATIVE,
                        semantic_authority=not dynamic,
                        path=path,
                        evidence_ref=str(item.get("call_id") or ""),
                        reason="ast_call" if not dynamic else "dynamic_call",
                    )
                    self._persist_edge(connection, edge)
                    edge_count += 1

            # Parse frontiers from AST index.
            for item in frontiers_raw:
                if hasattr(item, "to_dict"):
                    mapping = item.to_dict()
                elif isinstance(item, Mapping):
                    mapping = dict(item)
                else:
                    continue
                status_raw = str(mapping.get("status") or "").casefold()
                path = str(mapping.get("path") or "")
                reason = str(mapping.get("reason") or status_raw)
                if status_raw in {"failed"}:
                    fkind = FrontierKind.PARSE_FAILED
                    fstatus = FrontierStatus.OPEN
                elif status_raw in {"unsupported"}:
                    fkind = FrontierKind.UNSUPPORTED_LANGUAGE
                    fstatus = FrontierStatus.UNSUPPORTED
                elif status_raw in {"unknown", "partial"}:
                    fkind = FrontierKind.PARSER_UNCERTAINTY
                    fstatus = FrontierStatus.OPEN
                elif status_raw in {"excluded"}:
                    # Excluded is not a repair-blocking frontier by itself.
                    continue
                else:
                    fkind = FrontierKind.OPEN
                    fstatus = FrontierStatus.OPEN
                frontier_records.append(
                    ImpactFrontierRecord(
                        frontier_id="",
                        snapshot_id=selected,
                        kind=fkind,
                        status=fstatus,
                        subject_key=str(mapping.get("file_id") or ""),
                        path=path,
                        reason=reason,
                    )
                )

            for frontier in frontier_records:
                self._persist_frontier(connection, frontier)

            complete = not any(
                item.blocks_automatic_repair
                and str(
                    getattr(item.status, "value", item.status)
                )
                in {"open", "unsupported"}
                for item in frontier_records
            )
            result = MaterializeResult(
                materialization_id="",
                snapshot_id=selected,
                parser_id=selected_parser,
                policy_id=selected_policy,
                schema_id=DATABASE_IMPACT_GRAPH_SCHEMA,
                edge_count=edge_count,
                symbol_count=len(symbol_by_id) + len(file_module_symbols),
                frontier_count=len(frontier_records),
                complete=complete,
                body={
                    "repository_id": repository_id,
                    "tree_id": tree_id,
                },
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO impact_materializations (
                    materialization_id, snapshot_id, parser_id, policy_id,
                    schema_id, edge_count, symbol_count, frontier_count,
                    complete, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.materialization_id,
                    result.snapshot_id,
                    result.parser_id,
                    result.policy_id,
                    result.schema_id,
                    result.edge_count,
                    result.symbol_count,
                    result.frontier_count,
                    1 if result.complete else 0,
                    result.recorded_at,
                    _canonical_json(dict(result.body)),
                ],
            )
            self._commit_if_idle(connection)
            return result

    def materialize_edges(
        self,
        snapshot_id: str,
        edges: Sequence[Mapping[str, Any] | ImpactEdge],
        *,
        symbols: Sequence[Mapping[str, Any] | ImpactSymbol] = (),
        frontiers: Sequence[Mapping[str, Any] | ImpactFrontierRecord] = (),
        parser_id: str | None = None,
        policy_id: str | None = None,
        clear_existing: bool = False,
    ) -> MaterializeResult:
        """Materialize an explicit edge set (hermetic fixtures / mutations)."""

        selected = _text(snapshot_id, "snapshot_id")
        selected_parser = _text(parser_id or self._parser_id, "parser_id")
        selected_policy = _text(policy_id or self._policy_id, "policy_id")
        if len(edges) > MAX_EDGES:
            raise DatabaseImpactGraphBoundsError(
                f"edge count exceeds {MAX_EDGES}"
            )
        with self._lock:
            connection = self._require()
            if clear_existing:
                self._clear_snapshot(connection, selected)
            symbol_count = 0
            for raw in symbols:
                symbol = self._coerce_symbol(selected, raw)
                self._persist_symbol(connection, symbol)
                symbol_count += 1
            edge_count = 0
            for raw in edges:
                if isinstance(raw, ImpactEdge):
                    edge = raw
                else:
                    edge = self.upsert_edge(
                        snapshot_id=selected,
                        source=str(
                            raw.get("source_symbol_key")
                            or raw.get("source")
                            or ""
                        ),
                        target=str(
                            raw.get("target_symbol_key")
                            or raw.get("target")
                            or ""
                        ),
                        edge_kind=raw.get("edge_kind") or raw.get("kind") or "",
                        path=str(raw.get("path") or ""),
                        evidence_ref=str(raw.get("evidence_ref") or ""),
                        reason=str(raw.get("reason") or ""),
                        authority=raw.get("authority"),
                        semantic_authority=raw.get("semantic_authority"),
                        body=dict(raw.get("body") or {}),
                    )
                    edge_count += 1
                    continue
                self._persist_edge(connection, edge)
                edge_count += 1
            frontier_count = 0
            for raw in frontiers:
                if isinstance(raw, ImpactFrontierRecord):
                    frontier = raw
                else:
                    frontier = ImpactFrontierRecord(
                        frontier_id=str(raw.get("frontier_id") or ""),
                        snapshot_id=selected,
                        kind=raw.get("kind") or FrontierKind.OPEN,
                        status=raw.get("status") or FrontierStatus.OPEN,
                        subject_key=str(raw.get("subject_key") or ""),
                        path=str(raw.get("path") or ""),
                        reason=str(raw.get("reason") or ""),
                        blocks_automatic_repair=bool(
                            raw.get("blocks_automatic_repair", True)
                        ),
                        body=dict(raw.get("body") or {}),
                    )
                self._persist_frontier(connection, frontier)
                frontier_count += 1
            complete = frontier_count == 0
            result = MaterializeResult(
                materialization_id="",
                snapshot_id=selected,
                parser_id=selected_parser,
                policy_id=selected_policy,
                schema_id=DATABASE_IMPACT_GRAPH_SCHEMA,
                edge_count=edge_count,
                symbol_count=symbol_count,
                frontier_count=frontier_count,
                complete=complete,
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO impact_materializations (
                    materialization_id, snapshot_id, parser_id, policy_id,
                    schema_id, edge_count, symbol_count, frontier_count,
                    complete, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.materialization_id,
                    result.snapshot_id,
                    result.parser_id,
                    result.policy_id,
                    result.schema_id,
                    result.edge_count,
                    result.symbol_count,
                    result.frontier_count,
                    1 if result.complete else 0,
                    result.recorded_at,
                    _canonical_json(dict(result.body)),
                ],
            )
            self._commit_if_idle(connection)
            return result

    # -- query API -----------------------------------------------------------

    def list_symbols(self, snapshot_id: str) -> tuple[ImpactSymbol, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT symbol_key, snapshot_id, symbol_id, qualified_name,
                       path, language, symbol_kind, fingerprint
                FROM impact_symbols
                WHERE snapshot_id = ?
                ORDER BY qualified_name ASC, path ASC
                """,
                [selected],
            ).fetchall()
            return tuple(self._symbol_from_row(row) for row in rows)

    def list_edges(
        self,
        snapshot_id: str,
        *,
        edge_kind: ImpactEdgeKind | str | None = None,
        authority: EdgeAuthority | str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[ImpactEdge, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        clauses = ["snapshot_id = ?"]
        params: list[Any] = [selected]
        if edge_kind is not None:
            kind = ImpactEdgeKind.coerce(edge_kind)
            clauses.append("edge_kind = ?")
            params.append(kind.value)
        if authority is not None:
            auth = _enum(authority, EdgeAuthority, "authority")
            clauses.append("authority = ?")
            params.append(auth.value)
        page_limit = DEFAULT_PAGE_SIZE if limit is None else int(limit)
        page_limit = max(0, min(page_limit, MAX_PAGE_SIZE))
        page_offset = _nonneg_int(int(offset), "offset")
        sql = f"""
            SELECT edge_id, snapshot_id, source_symbol_key, target_symbol_key,
                   edge_kind, authority, semantic_authority, path,
                   evidence_ref, reason, recorded_at, body_json
            FROM impact_edges
            WHERE {' AND '.join(clauses)}
            ORDER BY edge_kind ASC, source_symbol_key ASC, target_symbol_key ASC
            LIMIT ? OFFSET ?
        """
        params.extend([page_limit, page_offset])
        with self._lock:
            connection = self._require()
            rows = connection.execute(sql, params).fetchall()
            return tuple(self._edge_from_row(row) for row in rows)

    def list_frontiers(
        self, snapshot_id: str
    ) -> tuple[ImpactFrontierRecord, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT frontier_id, snapshot_id, kind, subject_key, path,
                       status, reason, blocks_automatic_repair, recorded_at,
                       body_json
                FROM impact_frontiers
                WHERE snapshot_id = ?
                ORDER BY kind ASC, path ASC, subject_key ASC
                """,
                [selected],
            ).fetchall()
            return tuple(self._frontier_from_row(row) for row in rows)

    def query_callers(
        self,
        snapshot_id: str,
        symbol: str,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        transitive: bool = True,
    ) -> tuple[ImpactConsumerRecord, ...]:
        """Return symbols that call/depend on ``symbol`` (reverse edges)."""

        return self._traverse(
            snapshot_id,
            symbol,
            direction="callers",
            max_depth=max_depth,
            limit=limit,
            offset=offset,
            transitive=transitive,
        )

    def query_callees(
        self,
        snapshot_id: str,
        symbol: str,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        transitive: bool = True,
    ) -> tuple[ImpactConsumerRecord, ...]:
        """Return symbols that ``symbol`` calls/depends on (forward edges)."""

        return self._traverse(
            snapshot_id,
            symbol,
            direction="callees",
            max_depth=max_depth,
            limit=limit,
            offset=offset,
            transitive=transitive,
        )

    def query_impact_closure(
        self,
        snapshot_id: str,
        seeds: Sequence[str],
        *,
        parser_id: str | None = None,
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        overlay_digest: str = "",
        max_depth: int = DEFAULT_MAX_DEPTH,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        task_id: str = "",
        mutation_id: str = "",
        include_nominations: bool = False,
        persist: bool = True,
    ) -> ImpactClosure:
        """Compute reverse impact closure for changed symbols / mutation seeds."""

        selected = _text(snapshot_id, "snapshot_id")
        seed_list = [
            str(item).strip() for item in seeds if str(item).strip()
        ]
        if not seed_list:
            raise DatabaseImpactGraphIntegrityError(
                "impact closure requires at least one seed"
            )
        if len(seed_list) > MAX_SEEDS:
            raise DatabaseImpactGraphBoundsError(
                f"seed count exceeds {MAX_SEEDS}"
            )
        depth_limit = max(0, min(int(max_depth), MAX_DEPTH))
        page_limit = max(0, min(int(limit), MAX_PAGE_SIZE))
        page_offset = _nonneg_int(int(offset), "offset")

        binding = self.binding_for(
            selected,
            parser_id=parser_id,
            policy_id=policy_id,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
        )

        with self._lock:
            connection = self._require()
            symbols = self._load_symbols(connection, selected)
            edges = self._load_edges(connection, selected)
            frontiers = self._load_frontiers(connection, selected)

        # Index reverse adjacency over authoritative edges (and optional
        # nominations that never expand mandatory completeness).
        reverse: dict[str, list[tuple[ImpactEdge, str]]] = {}
        forward: dict[str, list[tuple[ImpactEdge, str]]] = {}
        for edge in edges:
            if (
                not include_nominations
                and edge.authority is EdgeAuthority.NOMINATION
            ):
                continue
            reverse.setdefault(edge.target_symbol_key, []).append(
                (edge, edge.source_symbol_key)
            )
            forward.setdefault(edge.source_symbol_key, []).append(
                (edge, edge.target_symbol_key)
            )

        seed_keys: list[str] = []
        seed_names: list[str] = []
        for seed in seed_list:
            resolved = self._resolve_seed(symbols, seed)
            if resolved is None:
                # Unknown seed remains an explicit open frontier.
                frontiers = frontiers + (
                    ImpactFrontierRecord(
                        frontier_id="",
                        snapshot_id=selected,
                        kind=FrontierKind.UNRESOLVED_SYMBOL,
                        status=FrontierStatus.OPEN,
                        subject_key=seed,
                        reason=f"unknown_seed:{seed}",
                    ),
                )
                continue
            seed_keys.append(resolved.symbol_key)
            seed_names.append(resolved.qualified_name)

        consumers_by_key: dict[str, ImpactConsumerRecord] = {}
        adjacency_for_scc: dict[str, set[str]] = {}
        truncated = False

        # Seed consumers (upstream of their dependents).
        for key, name in zip(seed_keys, seed_names):
            symbol = symbols.get(key)
            consumers_by_key[key] = ImpactConsumerRecord(
                consumer_id="",
                symbol_key=key,
                qualified_name=name,
                disposition=ConsumerDisposition.UPSTREAM,
                depth=0,
                path=symbol.path if symbol else "",
                edge_ids=(),
                edge_kinds=(),
                mandatory=False,
                is_seed=True,
                semantic_authority=True,
                reason="seed",
            )

        queue: deque[tuple[str, int]] = deque(
            (key, 0) for key in seed_keys
        )
        visited_depth: dict[str, int] = {key: 0 for key in seed_keys}
        while queue:
            current, depth = queue.popleft()
            if depth >= depth_limit:
                if reverse.get(current):
                    truncated = True
                continue
            for edge, dependent_key in sorted(
                reverse.get(current, ()),
                key=lambda item: (
                    item[0].edge_id,
                    item[1],
                ),
            ):
                if (
                    edge.authority is EdgeAuthority.NOMINATION
                    and not include_nominations
                ):
                    continue
                next_depth = depth + 1
                prior = visited_depth.get(dependent_key)
                if prior is not None and prior <= next_depth:
                    # Still record supporting edge on existing consumer.
                    existing = consumers_by_key.get(dependent_key)
                    if existing is not None and edge.edge_id not in existing.edge_ids:
                        consumers_by_key[dependent_key] = ImpactConsumerRecord(
                            consumer_id="",
                            symbol_key=existing.symbol_key,
                            qualified_name=existing.qualified_name,
                            disposition=existing.disposition,
                            depth=existing.depth,
                            path=existing.path,
                            edge_ids=existing.edge_ids + (edge.edge_id,),
                            edge_kinds=tuple(
                                sorted(
                                    set(existing.edge_kinds)
                                    | {
                                        edge.edge_kind.value
                                        if isinstance(
                                            edge.edge_kind, ImpactEdgeKind
                                        )
                                        else str(edge.edge_kind)
                                    }
                                )
                            ),
                            mandatory=existing.mandatory,
                            is_seed=existing.is_seed,
                            semantic_authority=existing.semantic_authority,
                            reason=existing.reason,
                        )
                    adjacency_for_scc.setdefault(current, set()).add(
                        dependent_key
                    )
                    continue
                symbol = symbols.get(dependent_key)
                name = (
                    symbol.qualified_name
                    if symbol is not None
                    else dependent_key
                )
                kind = (
                    edge.edge_kind
                    if isinstance(edge.edge_kind, ImpactEdgeKind)
                    else ImpactEdgeKind.coerce(edge.edge_kind)
                )
                is_frontier_edge = kind in {
                    ImpactEdgeKind.DYNAMIC,
                    ImpactEdgeKind.CROSS_LANGUAGE,
                } or edge.authority is EdgeAuthority.NOMINATION
                disposition = _disposition_for_edge(
                    kind,
                    is_seed=False,
                    is_frontier=is_frontier_edge
                    and kind
                    in {
                        ImpactEdgeKind.DYNAMIC,
                        ImpactEdgeKind.CROSS_LANGUAGE,
                    },
                )
                mandatory = (
                    disposition
                    in {
                        ConsumerDisposition.MIGRATE,
                        ConsumerDisposition.ADAPTER,
                    }
                    and edge.authority is EdgeAuthority.AUTHORITATIVE
                )
                consumers_by_key[dependent_key] = ImpactConsumerRecord(
                    consumer_id="",
                    symbol_key=dependent_key,
                    qualified_name=name,
                    disposition=disposition,
                    depth=next_depth,
                    path=symbol.path if symbol else edge.path,
                    edge_ids=(edge.edge_id,),
                    edge_kinds=(
                        kind.value
                        if isinstance(kind, ImpactEdgeKind)
                        else str(kind),
                    ),
                    mandatory=mandatory,
                    is_seed=False,
                    semantic_authority=bool(edge.semantic_authority),
                    reason=edge.reason or kind.value,
                )
                visited_depth[dependent_key] = next_depth
                adjacency_for_scc.setdefault(current, set()).add(dependent_key)
                if len(consumers_by_key) > MAX_CONSUMERS:
                    truncated = True
                    break
                queue.append((dependent_key, next_depth))
            if truncated and len(consumers_by_key) > MAX_CONSUMERS:
                break

        # Resource-bound truncation frontier.
        if truncated:
            frontiers = frontiers + (
                ImpactFrontierRecord(
                    frontier_id="",
                    snapshot_id=selected,
                    kind=FrontierKind.RESOURCE_BOUND,
                    status=FrontierStatus.OPEN,
                    reason="impact_closure_truncated",
                    blocks_automatic_repair=True,
                ),
            )

        all_consumers = tuple(
            sorted(
                consumers_by_key.values(),
                key=lambda item: (item.depth, item.qualified_name, item.symbol_key),
            )
        )
        total = len(all_consumers)
        page = all_consumers[page_offset : page_offset + page_limit]
        if page_offset + page_limit < total:
            truncated = True
            # Pagination itself is not a semantic frontier, but completeness
            # cannot claim full coverage for a partial page response.
            if not any(
                item.kind is FrontierKind.RESOURCE_BOUND
                or str(getattr(item.kind, "value", item.kind))
                == FrontierKind.RESOURCE_BOUND.value
                for item in frontiers
            ):
                frontiers = frontiers + (
                    ImpactFrontierRecord(
                        frontier_id="",
                        snapshot_id=selected,
                        kind=FrontierKind.RESOURCE_BOUND,
                        status=FrontierStatus.OPEN,
                        reason="pagination_partial_page",
                        blocks_automatic_repair=True,
                    ),
                )

        # SCCs over the full consumer set (not just the page).
        consumer_keys = [item.symbol_key for item in all_consumers]
        scc_adj = {
            key: tuple(sorted(adjacency_for_scc.get(key, ())))
            for key in consumer_keys
        }
        # Map symbol keys to consumer ids for SCC membership.
        key_to_consumer = {
            item.symbol_key: item.consumer_id for item in all_consumers
        }
        scc_members = _tarjan_sccs(consumer_keys, scc_adj)
        scc_list: list[ImpactSCC] = []
        for members in scc_members:
            if len(members) <= 1:
                continue
            member_ids = tuple(
                key_to_consumer[member]
                for member in members
                if member in key_to_consumer
            )
            if len(member_ids) > 1:
                scc_list.append(
                    ImpactSCC(scc_id="", member_consumer_ids=member_ids)
                )
        sccs = tuple(scc_list)

        blocking = any(
            item.blocks_automatic_repair
            and str(getattr(item.status, "value", item.status))
            in {"open", "unsupported"}
            for item in frontiers
        )
        # Promote unresolved cross-language / dynamic consumers into frontier
        # records so completeness cannot claim full coverage.
        for item in all_consumers:
            disposition = item.disposition
            if disposition is ConsumerDisposition.FRONTIER or str(
                getattr(disposition, "value", disposition)
            ) == ConsumerDisposition.FRONTIER.value:
                frontiers = frontiers + (
                    ImpactFrontierRecord(
                        frontier_id="",
                        snapshot_id=selected,
                        kind=(
                            FrontierKind.CROSS_LANGUAGE
                            if ImpactEdgeKind.CROSS_LANGUAGE.value
                            in item.edge_kinds
                            else FrontierKind.DYNAMIC_CALL
                        ),
                        status=FrontierStatus.OPEN,
                        subject_key=item.symbol_key,
                        path=item.path,
                        reason=item.reason or "consumer_frontier_disposition",
                    ),
                )
                blocking = True
        if blocking or truncated:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            if not all_consumers and not seed_keys:
                completeness = ImpactCompleteness.ABSTAINED
        else:
            completeness = ImpactCompleteness.COMPLETE

        closure = ImpactClosure(
            binding=binding,
            seeds=tuple(seed_list),
            completeness=completeness,
            consumers=page,
            frontiers=frontiers,
            sccs=sccs,
            automatic_repair_allowed=(
                completeness is ImpactCompleteness.COMPLETE and not blocking
            ),
            truncated=truncated,
            page_offset=page_offset,
            page_limit=page_limit,
            total_consumer_count=total,
            task_id=task_id,
            mutation_id=mutation_id,
            freshness="fresh",
        )

        if persist:
            with self._lock:
                connection = self._require()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO impact_closures (
                        closure_id, snapshot_id, parser_id, policy_id,
                        schema_id, seed_json, completeness,
                        automatic_repair_allowed, consumer_count,
                        frontier_count, recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        closure.closure_id,
                        binding.snapshot_id,
                        binding.parser_id,
                        binding.policy_id,
                        binding.schema_id,
                        _canonical_json(list(closure.seeds)),
                        closure.completeness.value
                        if isinstance(
                            closure.completeness, ImpactCompleteness
                        )
                        else str(closure.completeness),
                        1 if closure.automatic_repair_allowed else 0,
                        len(closure.consumers),
                        len(closure.frontiers),
                        binding.recorded_at,
                        _canonical_json(closure.to_dict()),
                    ],
                )
                self._commit_if_idle(connection)
        return closure

    def query_changed_neighborhood(
        self,
        snapshot_id: str,
        seeds: Sequence[str],
        *,
        parser_id: str | None = None,
        policy_id: str | None = None,
        repository_id: str = "",
        tree_id: str = "",
        overlay_digest: str = "",
        max_depth: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        task_id: str = "",
        mutation_id: str = "",
        include_nominations: bool = True,
    ) -> ChangedSymbolNeighborhood:
        """Return callers/callees/imports/types/tests/... around changed seeds."""

        selected = _text(snapshot_id, "snapshot_id")
        seed_list = [
            str(item).strip() for item in seeds if str(item).strip()
        ]
        if not seed_list:
            raise DatabaseImpactGraphIntegrityError(
                "changed neighborhood requires at least one seed"
            )
        depth_limit = max(0, min(int(max_depth), MAX_DEPTH))
        page_limit = max(0, min(int(limit), MAX_PAGE_SIZE))
        page_offset = _nonneg_int(int(offset), "offset")
        binding = self.binding_for(
            selected,
            parser_id=parser_id,
            policy_id=policy_id,
            repository_id=repository_id,
            tree_id=tree_id,
            overlay_digest=overlay_digest,
        )

        with self._lock:
            connection = self._require()
            symbols = self._load_symbols(connection, selected)
            edges = self._load_edges(connection, selected)
            frontiers = list(self._load_frontiers(connection, selected))

        seed_keys: list[str] = []
        for seed in seed_list:
            resolved = self._resolve_seed(symbols, seed)
            if resolved is None:
                frontiers.append(
                    ImpactFrontierRecord(
                        frontier_id="",
                        snapshot_id=selected,
                        kind=FrontierKind.UNRESOLVED_SYMBOL,
                        status=FrontierStatus.OPEN,
                        subject_key=seed,
                        reason=f"unknown_seed:{seed}",
                    )
                )
                continue
            seed_keys.append(resolved.symbol_key)
        seed_key_set = set(seed_keys)

        buckets: dict[str, list[ImpactConsumerRecord]] = {
            name: [] for name in _NEIGHBORHOOD_BUCKETS
        }
        seen: dict[str, set[str]] = {name: set() for name in _NEIGHBORHOOD_BUCKETS}
        truncated = False

        def _add(
            bucket: str,
            symbol_key: str,
            edge: ImpactEdge,
            *,
            depth: int,
        ) -> None:
            nonlocal truncated
            if symbol_key in seed_key_set:
                return
            if symbol_key in seen[bucket]:
                return
            if len(seen[bucket]) >= page_offset + page_limit:
                truncated = True
                return
            symbol = symbols.get(symbol_key)
            name = symbol.qualified_name if symbol else symbol_key
            kind = (
                edge.edge_kind
                if isinstance(edge.edge_kind, ImpactEdgeKind)
                else ImpactEdgeKind.coerce(edge.edge_kind)
            )
            is_nom = edge.authority is EdgeAuthority.NOMINATION or _is_nomination_kind(
                kind
            )
            disposition = _disposition_for_edge(
                kind,
                is_frontier=kind
                in {ImpactEdgeKind.DYNAMIC, ImpactEdgeKind.CROSS_LANGUAGE},
            )
            if is_nom:
                disposition = ConsumerDisposition.REVIEW_ONLY
            consumer = ImpactConsumerRecord(
                consumer_id="",
                symbol_key=symbol_key,
                qualified_name=name,
                disposition=disposition,
                depth=depth,
                path=symbol.path if symbol else edge.path,
                edge_ids=(edge.edge_id,),
                edge_kinds=(
                    kind.value
                    if isinstance(kind, ImpactEdgeKind)
                    else str(kind),
                ),
                mandatory=(
                    not is_nom
                    and disposition is ConsumerDisposition.MIGRATE
                ),
                is_seed=False,
                semantic_authority=bool(edge.semantic_authority) and not is_nom,
                reason=edge.reason or kind.value,
            )
            seen[bucket].add(symbol_key)
            # Collect all then page later for stable offsets.
            buckets[bucket].append(consumer)

        # Direct (and optional shallow transitive) neighborhood.
        # Callers: reverse of CALLS/DEPENDS_ON/etc from seed.
        # Callees: forward CALLS from seed.
        for edge in sorted(edges, key=lambda item: item.edge_id):
            if (
                not include_nominations
                and edge.authority is EdgeAuthority.NOMINATION
            ):
                continue
            kind = (
                edge.edge_kind
                if isinstance(edge.edge_kind, ImpactEdgeKind)
                else ImpactEdgeKind.coerce(edge.edge_kind)
            )
            if edge.target_symbol_key in seed_key_set:
                # source consumes seed -> source is a caller/import/...
                bucket = _bucket_for_edge(kind)
                if kind is ImpactEdgeKind.CALLS:
                    bucket = "callers"
                _add(bucket, edge.source_symbol_key, edge, depth=1)
            if edge.source_symbol_key in seed_key_set:
                # seed consumes target -> target is a callee / import target
                if kind is ImpactEdgeKind.CALLS:
                    _add("callees", edge.target_symbol_key, edge, depth=1)
                elif kind is ImpactEdgeKind.IMPORTS:
                    _add("imports", edge.target_symbol_key, edge, depth=1)
                elif kind in {
                    ImpactEdgeKind.TYPE,
                    ImpactEdgeKind.IMPLEMENTS,
                }:
                    _add("types", edge.target_symbol_key, edge, depth=1)
                elif _is_nomination_kind(kind):
                    _add("nominations", edge.target_symbol_key, edge, depth=1)
                else:
                    _add(_bucket_for_edge(kind), edge.target_symbol_key, edge, depth=1)

        # Optional shallow expansion beyond depth 1 for recursive callers.
        if depth_limit > 1:
            frontier_keys = {
                item.symbol_key
                for bucket_items in buckets.values()
                for item in bucket_items
            }
            for _ in range(2, depth_limit + 1):
                next_keys: set[str] = set()
                for edge in edges:
                    if edge.authority is EdgeAuthority.NOMINATION:
                        continue
                    if edge.target_symbol_key in frontier_keys:
                        kind = (
                            edge.edge_kind
                            if isinstance(edge.edge_kind, ImpactEdgeKind)
                            else ImpactEdgeKind.coerce(edge.edge_kind)
                        )
                        if kind is ImpactEdgeKind.CALLS:
                            before = len(seen["callers"])
                            _add(
                                "callers",
                                edge.source_symbol_key,
                                edge,
                                depth=_,
                            )
                            if edge.source_symbol_key not in frontier_keys:
                                next_keys.add(edge.source_symbol_key)
                            del before
                if not next_keys:
                    break
                frontier_keys |= next_keys

        paged: dict[str, tuple[ImpactConsumerRecord, ...]] = {}
        for name in _NEIGHBORHOOD_BUCKETS:
            ordered = tuple(
                sorted(
                    buckets[name],
                    key=lambda item: (
                        item.depth,
                        item.qualified_name,
                        item.symbol_key,
                    ),
                )
            )
            slice_items = ordered[page_offset : page_offset + page_limit]
            if page_offset + page_limit < len(ordered):
                truncated = True
            paged[name] = slice_items

        blocking = any(
            item.blocks_automatic_repair
            and str(getattr(item.status, "value", item.status))
            in {"open", "unsupported"}
            for item in frontiers
        )
        if truncated and not any(
            str(getattr(item.kind, "value", item.kind))
            == FrontierKind.RESOURCE_BOUND.value
            for item in frontiers
        ):
            frontiers.append(
                ImpactFrontierRecord(
                    frontier_id="",
                    snapshot_id=selected,
                    kind=FrontierKind.RESOURCE_BOUND,
                    status=FrontierStatus.OPEN,
                    reason="neighborhood_truncated_or_paginated",
                    blocks_automatic_repair=True,
                )
            )
            blocking = True

        return ChangedSymbolNeighborhood(
            binding=binding,
            seeds=tuple(seed_list),
            callers=paged["callers"],
            callees=paged["callees"],
            imports=paged["imports"],
            types=paged["types"],
            tests=paged["tests"],
            contracts=paged["contracts"],
            proofs=paged["proofs"],
            config=paged["config"],
            docs=paged["docs"],
            aliases=paged["aliases"],
            reexports=paged["reexports"],
            generated=paged["generated"],
            nominations=paged["nominations"],
            frontiers=tuple(frontiers),
            automatic_repair_allowed=not blocking and not truncated,
            truncated=truncated,
            page_offset=page_offset,
            page_limit=page_limit,
            task_id=task_id,
            mutation_id=mutation_id,
            freshness="fresh",
        )

    def query_for_mutation(
        self,
        snapshot_id: str,
        *,
        changed_symbols: Sequence[str] = (),
        changed_paths: Sequence[str] = (),
        mutation_id: str = "",
        task_id: str = "",
        parser_id: str | None = None,
        policy_id: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[ImpactClosure, ChangedSymbolNeighborhood]:
        """Convenience: impact closure + neighborhood for a mutation/task."""

        seeds = [str(item).strip() for item in changed_symbols if str(item).strip()]
        if not seeds and changed_paths:
            # Seed every symbol on the changed paths.
            with self._lock:
                connection = self._require()
                symbols = self._load_symbols(connection, snapshot_id)
            path_set = {_repo_path(path) for path in changed_paths}
            seeds = [
                symbol.qualified_name
                for symbol in symbols.values()
                if symbol.path in path_set
            ]
        if not seeds:
            raise DatabaseImpactGraphIntegrityError(
                "mutation query requires changed_symbols or changed_paths "
                "with bound symbols"
            )
        # Also record deletion frontiers for paths with no remaining symbols.
        for path in changed_paths:
            normalized = _repo_path(path)
            with self._lock:
                connection = self._require()
                symbols = self._load_symbols(connection, snapshot_id)
            if not any(item.path == normalized for item in symbols.values()):
                self.upsert_frontier(
                    snapshot_id=snapshot_id,
                    kind=FrontierKind.DELETION,
                    status=FrontierStatus.OPEN,
                    path=normalized,
                    reason=f"deleted_or_empty_path:{normalized}",
                )

        closure = self.query_impact_closure(
            snapshot_id,
            seeds,
            parser_id=parser_id,
            policy_id=policy_id,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
            task_id=task_id,
            mutation_id=mutation_id,
        )
        neighborhood = self.query_changed_neighborhood(
            snapshot_id,
            seeds,
            parser_id=parser_id,
            policy_id=policy_id,
            max_depth=min(max_depth, 2),
            limit=limit,
            offset=offset,
            task_id=task_id,
            mutation_id=mutation_id,
        )
        return closure, neighborhood

    # -- internal helpers ----------------------------------------------------

    def _coerce_symbol(
        self,
        snapshot_id: str,
        value: ImpactSymbol | Mapping[str, Any] | str,
    ) -> ImpactSymbol:
        if isinstance(value, ImpactSymbol):
            return value
        if isinstance(value, Mapping):
            name = str(
                value.get("qualified_name")
                or value.get("name")
                or value.get("symbol_key")
                or ""
            )
            return ImpactSymbol(
                symbol_key=str(value.get("symbol_key") or ""),
                snapshot_id=str(value.get("snapshot_id") or snapshot_id),
                symbol_id=str(
                    value.get("symbol_id") or value.get("id") or name
                ),
                qualified_name=name,
                path=str(value.get("path") or ""),
                language=str(value.get("language") or ""),
                symbol_kind=str(value.get("symbol_kind") or ""),
                fingerprint=str(value.get("fingerprint") or ""),
            )
        text = str(value or "").strip()
        if not text:
            raise DatabaseImpactGraphIntegrityError(
                "symbol reference is required"
            )
        # Allow passing an existing symbol_key directly.
        if text.startswith("impact-symbol:"):
            return ImpactSymbol(
                symbol_key=text,
                snapshot_id=snapshot_id,
                symbol_id=text,
                qualified_name=text,
            )
        return ImpactSymbol(
            symbol_key="",
            snapshot_id=snapshot_id,
            symbol_id=text,
            qualified_name=text,
        )

    def _persist_symbol(self, connection: Any, symbol: ImpactSymbol) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO impact_symbols (
                symbol_key, snapshot_id, symbol_id, qualified_name, path,
                language, symbol_kind, fingerprint, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                symbol.symbol_key,
                symbol.snapshot_id,
                symbol.symbol_id,
                symbol.qualified_name,
                symbol.path,
                symbol.language,
                symbol.symbol_kind,
                symbol.fingerprint,
                _canonical_json(symbol.to_dict()),
            ],
        )

    def _persist_edge(self, connection: Any, edge: ImpactEdge) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO impact_edges (
                edge_id, snapshot_id, source_symbol_key, target_symbol_key,
                edge_kind, authority, semantic_authority, path,
                evidence_ref, reason, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                edge.edge_id,
                edge.snapshot_id,
                edge.source_symbol_key,
                edge.target_symbol_key,
                edge.edge_kind.value
                if isinstance(edge.edge_kind, ImpactEdgeKind)
                else str(edge.edge_kind),
                edge.authority.value
                if isinstance(edge.authority, EdgeAuthority)
                else str(edge.authority),
                1 if edge.semantic_authority else 0,
                edge.path,
                edge.evidence_ref,
                edge.reason,
                edge.recorded_at,
                _canonical_json(dict(edge.body)),
            ],
        )

    def _persist_frontier(
        self, connection: Any, frontier: ImpactFrontierRecord
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO impact_frontiers (
                frontier_id, snapshot_id, kind, subject_key, path, status,
                reason, blocks_automatic_repair, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                frontier.frontier_id,
                frontier.snapshot_id,
                frontier.kind.value
                if isinstance(frontier.kind, FrontierKind)
                else str(frontier.kind),
                frontier.subject_key,
                frontier.path,
                frontier.status.value
                if isinstance(frontier.status, FrontierStatus)
                else str(frontier.status),
                frontier.reason,
                1 if frontier.blocks_automatic_repair else 0,
                frontier.recorded_at,
                _canonical_json(dict(frontier.body)),
            ],
        )

    def _clear_snapshot(self, connection: Any, snapshot_id: str) -> None:
        for table in (
            "impact_edges",
            "impact_frontiers",
            "impact_symbols",
            "impact_closures",
            "impact_materializations",
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE snapshot_id = ?",
                [snapshot_id],
            )

    def _symbol_from_row(self, row: Any) -> ImpactSymbol:
        mapping = _row_mapping(row)
        return ImpactSymbol(
            symbol_key=str(mapping.get("symbol_key") or ""),
            snapshot_id=str(mapping.get("snapshot_id") or ""),
            symbol_id=str(mapping.get("symbol_id") or ""),
            qualified_name=str(mapping.get("qualified_name") or ""),
            path=str(mapping.get("path") or ""),
            language=str(mapping.get("language") or ""),
            symbol_kind=str(mapping.get("symbol_kind") or ""),
            fingerprint=str(mapping.get("fingerprint") or ""),
        )

    def _edge_from_row(self, row: Any) -> ImpactEdge:
        mapping = _row_mapping(row)
        body_raw = mapping.get("body_json") or "{}"
        try:
            body = json.loads(str(body_raw))
        except json.JSONDecodeError:
            body = {}
        return ImpactEdge(
            edge_id=str(mapping.get("edge_id") or ""),
            snapshot_id=str(mapping.get("snapshot_id") or ""),
            source_symbol_key=str(mapping.get("source_symbol_key") or ""),
            target_symbol_key=str(mapping.get("target_symbol_key") or ""),
            edge_kind=str(mapping.get("edge_kind") or ""),
            authority=str(mapping.get("authority") or "authoritative"),
            semantic_authority=bool(int(mapping.get("semantic_authority") or 0)),
            path=str(mapping.get("path") or ""),
            evidence_ref=str(mapping.get("evidence_ref") or ""),
            reason=str(mapping.get("reason") or ""),
            recorded_at=str(mapping.get("recorded_at") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def _frontier_from_row(self, row: Any) -> ImpactFrontierRecord:
        mapping = _row_mapping(row)
        body_raw = mapping.get("body_json") or "{}"
        try:
            body = json.loads(str(body_raw))
        except json.JSONDecodeError:
            body = {}
        return ImpactFrontierRecord(
            frontier_id=str(mapping.get("frontier_id") or ""),
            snapshot_id=str(mapping.get("snapshot_id") or ""),
            kind=str(mapping.get("kind") or FrontierKind.OPEN.value),
            status=str(mapping.get("status") or FrontierStatus.OPEN.value),
            subject_key=str(mapping.get("subject_key") or ""),
            path=str(mapping.get("path") or ""),
            reason=str(mapping.get("reason") or ""),
            blocks_automatic_repair=bool(
                int(mapping.get("blocks_automatic_repair") or 0)
            ),
            recorded_at=str(mapping.get("recorded_at") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def _load_symbols(
        self, connection: Any, snapshot_id: str
    ) -> dict[str, ImpactSymbol]:
        rows = connection.execute(
            """
            SELECT symbol_key, snapshot_id, symbol_id, qualified_name,
                   path, language, symbol_kind, fingerprint
            FROM impact_symbols
            WHERE snapshot_id = ?
            """,
            [snapshot_id],
        ).fetchall()
        result: dict[str, ImpactSymbol] = {}
        for row in rows:
            symbol = self._symbol_from_row(row)
            result[symbol.symbol_key] = symbol
        return result

    def _load_edges(
        self, connection: Any, snapshot_id: str
    ) -> tuple[ImpactEdge, ...]:
        rows = connection.execute(
            """
            SELECT edge_id, snapshot_id, source_symbol_key, target_symbol_key,
                   edge_kind, authority, semantic_authority, path,
                   evidence_ref, reason, recorded_at, body_json
            FROM impact_edges
            WHERE snapshot_id = ?
            ORDER BY edge_id ASC
            """,
            [snapshot_id],
        ).fetchall()
        return tuple(self._edge_from_row(row) for row in rows)

    def _load_frontiers(
        self, connection: Any, snapshot_id: str
    ) -> tuple[ImpactFrontierRecord, ...]:
        rows = connection.execute(
            """
            SELECT frontier_id, snapshot_id, kind, subject_key, path,
                   status, reason, blocks_automatic_repair, recorded_at,
                   body_json
            FROM impact_frontiers
            WHERE snapshot_id = ?
            ORDER BY frontier_id ASC
            """,
            [snapshot_id],
        ).fetchall()
        return tuple(self._frontier_from_row(row) for row in rows)

    def _resolve_seed(
        self,
        symbols: Mapping[str, ImpactSymbol],
        seed: str,
    ) -> ImpactSymbol | None:
        text = str(seed or "").strip()
        if not text:
            return None
        if text in symbols:
            return symbols[text]
        for symbol in symbols.values():
            if symbol.qualified_name == text or symbol.symbol_id == text:
                return symbol
        # Leaf-name fallback when unique.
        matches = [
            symbol
            for symbol in symbols.values()
            if symbol.qualified_name.rsplit(".", 1)[-1] == text
            or symbol.qualified_name.endswith(f".{text}")
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _resolve_name(
        symbol_by_name: Mapping[str, Sequence[ImpactSymbol]],
        name: str,
        *,
        snapshot_id: str,
    ) -> ImpactSymbol | None:
        del snapshot_id
        text = str(name or "").strip()
        if not text:
            return None
        matches = list(symbol_by_name.get(text) or ())
        if len(matches) == 1:
            return matches[0]
        # Try dotted suffix / module path variants.
        alt = text.replace("/", ".")
        matches = list(symbol_by_name.get(alt) or ())
        if len(matches) == 1:
            return matches[0]
        return None

    def _traverse(
        self,
        snapshot_id: str,
        symbol: str,
        *,
        direction: str,
        max_depth: int,
        limit: int,
        offset: int,
        transitive: bool,
    ) -> tuple[ImpactConsumerRecord, ...]:
        selected = _text(snapshot_id, "snapshot_id")
        depth_limit = max(0, min(int(max_depth), MAX_DEPTH))
        if not transitive:
            depth_limit = min(depth_limit, 1)
        page_limit = max(0, min(int(limit), MAX_PAGE_SIZE))
        page_offset = _nonneg_int(int(offset), "offset")

        with self._lock:
            connection = self._require()
            symbols = self._load_symbols(connection, selected)
            edges = self._load_edges(connection, selected)

        seed = self._resolve_seed(symbols, symbol)
        if seed is None:
            return ()

        reverse: dict[str, list[tuple[ImpactEdge, str]]] = {}
        forward: dict[str, list[tuple[ImpactEdge, str]]] = {}
        for edge in edges:
            if edge.authority is EdgeAuthority.NOMINATION:
                continue
            reverse.setdefault(edge.target_symbol_key, []).append(
                (edge, edge.source_symbol_key)
            )
            forward.setdefault(edge.source_symbol_key, []).append(
                (edge, edge.target_symbol_key)
            )
        adjacency = reverse if direction == "callers" else forward

        results: list[ImpactConsumerRecord] = []
        seen: set[str] = {seed.symbol_key}
        queue: deque[tuple[str, int]] = deque([(seed.symbol_key, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= depth_limit:
                continue
            for edge, neighbor in sorted(
                adjacency.get(current, ()),
                key=lambda item: (item[0].edge_id, item[1]),
            ):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                symbol_obj = symbols.get(neighbor)
                kind = (
                    edge.edge_kind
                    if isinstance(edge.edge_kind, ImpactEdgeKind)
                    else ImpactEdgeKind.coerce(edge.edge_kind)
                )
                results.append(
                    ImpactConsumerRecord(
                        consumer_id="",
                        symbol_key=neighbor,
                        qualified_name=(
                            symbol_obj.qualified_name
                            if symbol_obj
                            else neighbor
                        ),
                        disposition=_disposition_for_edge(kind),
                        depth=depth + 1,
                        path=symbol_obj.path if symbol_obj else edge.path,
                        edge_ids=(edge.edge_id,),
                        edge_kinds=(
                            kind.value
                            if isinstance(kind, ImpactEdgeKind)
                            else str(kind),
                        ),
                        mandatory=edge.authority is EdgeAuthority.AUTHORITATIVE,
                        semantic_authority=bool(edge.semantic_authority),
                        reason=edge.reason or direction,
                    )
                )
                queue.append((neighbor, depth + 1))
        ordered = sorted(
            results,
            key=lambda item: (item.depth, item.qualified_name, item.symbol_key),
        )
        return tuple(ordered[page_offset : page_offset + page_limit])


def open_database_impact_graph(
    database_path: Path | str,
    *,
    parser_id: str = DEFAULT_PARSER_ID,
    policy_id: str = DEFAULT_POLICY_ID,
    graph_version: str = DEFAULT_GRAPH_VERSION,
) -> DatabaseImpactGraph:
    """Open (or create) a DatabaseImpactGraph store at ``database_path``."""

    return DatabaseImpactGraph(
        database_path,
        parser_id=parser_id,
        policy_id=policy_id,
        graph_version=graph_version,
    ).open()


__all__ = [
    "AUTHORITY_CLASS",
    "CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE",
    "CHANGED_SYMBOL_NEIGHBORHOOD_SCHEMA",
    "ChangedSymbolNeighborhood",
    "ConsumerDisposition",
    "DATABASE_IMPACT_GRAPH_INTERFACE",
    "DATABASE_IMPACT_GRAPH_SCHEMA",
    "DEFAULT_PARSER_ID",
    "DEFAULT_POLICY_ID",
    "DatabaseImpactGraph",
    "DatabaseImpactGraphBoundsError",
    "DatabaseImpactGraphConflictError",
    "DatabaseImpactGraphError",
    "DatabaseImpactGraphIntegrityError",
    "DatabaseImpactGraphNotOpenError",
    "DuckDBUnavailableError",
    "EdgeAuthority",
    "FrontierKind",
    "FrontierStatus",
    "IMPACT_CLOSURE_INTERFACE",
    "IMPACT_CLOSURE_SCHEMA",
    "ImpactClosure",
    "ImpactCompleteness",
    "ImpactConsumerRecord",
    "ImpactEdge",
    "ImpactEdgeKind",
    "ImpactFrontierRecord",
    "ImpactSCC",
    "ImpactSymbol",
    "MaterializeResult",
    "QueryBinding",
    "duckdb_available",
    "open_database_impact_graph",
]
