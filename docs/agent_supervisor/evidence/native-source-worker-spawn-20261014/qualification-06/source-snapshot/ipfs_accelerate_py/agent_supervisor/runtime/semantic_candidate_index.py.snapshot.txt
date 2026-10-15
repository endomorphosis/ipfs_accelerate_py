"""Bounded, source-derived DuckDB candidate ordering for worker capsules.

Rows and rankings are advisory projections. Native capsule verification,
freshness admission and raw-source policy remain owned by their existing APIs.
Replay uses the same call-local projection, independent of historical databases.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType


MAX_ROWS = 1024
MAX_QUERY_BYTES = 8192
MAX_ROWS_BYTES = 8 * 1024 * 1024
ROW_FIELDS = (
    "repository_id", "scope_cid", "semantic_root_cid", "stable_symbol_id",
    "path", "qualified_name", "source_cid", "version_cid", "capsule_cid",
)
_IDENTITY = ROW_FIELDS[:4]
_DDL = """
CREATE TABLE IF NOT EXISTS semantic_capsule_candidates (
    repository_id VARCHAR NOT NULL,
    scope_cid VARCHAR NOT NULL,
    semantic_root_cid VARCHAR NOT NULL,
    stable_symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    source_cid VARCHAR NOT NULL,
    version_cid VARCHAR NOT NULL,
    capsule_cid VARCHAR NOT NULL,
    lexical_terms VARCHAR[] NOT NULL,
    PRIMARY KEY (repository_id, scope_cid, semantic_root_cid, stable_symbol_id)
)
"""


def _text(value, name):
    if type(value) is not str or not value or len(value.encode()) > MAX_QUERY_BYTES:
        raise ValueError(f"candidate {name} must be bounded nonempty text")
    return value


def _terms(text):
    # Match the original Python selector exactly, including Unicode lowercasing
    # and its minimum two-character ASCII token pattern.
    return sorted(set(re.findall(r"[a-z][a-z0-9_]+", text.lower())))


def _validate_rows(rows):
    if type(rows) not in (list, tuple) or len(rows) > MAX_ROWS:
        raise ValueError("candidate rows must be a sequence of at most 1024 rows")
    checked, identities = [], set()
    size = 0
    for row in rows:
        if type(row) is not dict or set(row) != set(ROW_FIELDS):
            raise ValueError("candidate row requires its closed source-binding fields")
        copied = {name: _text(row[name], name) for name in ROW_FIELDS}
        path = PurePosixPath(copied["path"])
        if (path.is_absolute() or ".." in path.parts or copied["path"] == "."
                or str(path) != copied["path"]):
            raise ValueError("candidate path must be canonical and repository-relative")
        identity = tuple(copied[name] for name in _IDENTITY)
        if identity in identities:
            raise ValueError("candidate snapshot has duplicate symbol identities")
        identities.add(identity)
        size += len(json.dumps(copied, ensure_ascii=False, separators=(",", ":")).encode())
        if size > MAX_ROWS_BYTES:
            raise ValueError("candidate rows exceed the metadata byte bound")
        checked.append(copied)
    return checked


def build_candidate_rows(*, symbols, bundle, scope_cid):
    """Bind the complete current symbol population to its immutable native root.

    Every row verifies capsule membership and the symbol's repository, path,
    source and version bindings. The supplied scope CID is caller-owned source
    scope identity; this projection does not attest to source freshness.
    """
    from ipfs_datasets_py.logic.software_contracts.semantic_index.models import SymbolRecord
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticStateBundle, SortedPairIndex, verify_block_bytes,
        canonical_dag_json_bytes,
    )
    from .semantic_capsule_selection import immutable_bundle_capsule_reader

    _text(scope_cid, "scope_cid")
    if type(symbols) not in (list, tuple) or len(symbols) > MAX_ROWS:
        raise ValueError("candidate symbols must be a sequence of at most 1024 symbols")
    if type(bundle) is not SemanticStateBundle or type(bundle.blocks) is not MappingProxyType:
        raise ValueError("candidate projection requires the exact immutable native bundle")
    root = bundle.root
    raw_root = bundle.blocks[root.root_cid]
    verify_block_bytes(root.root_cid, raw_root)
    if raw_root != canonical_dag_json_bytes(root.identity_payload()):
        raise ValueError("candidate native root differs from its immutable block")
    reader = immutable_bundle_capsule_reader(bundle, root_cid=root.root_cid)
    # The reader verifies this index. A second parse establishes complete input
    # membership instead of silently treating a caller-supplied subset as full.
    index = SortedPairIndex.from_dict({**json.loads(bundle.blocks[root.capsule_index_cid]),
                                     "index_cid": root.capsule_index_cid})
    rows = []
    for symbol in symbols:
        if type(symbol) is not SymbolRecord:
            raise ValueError("candidate projection requires exact native symbol records")
        capsule = reader(symbol.stable_id)
        if (symbol.repository_id != root.repository_id
                or symbol.module_path != capsule.source_slice_path
                or symbol.source_cid != capsule.source_cid
                or symbol.version_cid != capsule.version_cid):
            raise ValueError("candidate symbol differs from its native capsule binding")
        rows.append(dict(repository_id=root.repository_id, scope_cid=scope_cid,
            semantic_root_cid=root.root_cid, stable_symbol_id=symbol.stable_id,
            path=symbol.module_path, qualified_name=symbol.qualified_name,
            source_cid=capsule.source_cid, version_cid=capsule.version_cid,
            capsule_cid=capsule.capsule_cid))
    rows = _validate_rows(rows)
    if {row["stable_symbol_id"] for row in rows} != {key for key, _ in index.pairs}:
        raise ValueError("candidate population differs from the full native capsule index")
    return sorted(rows, key=lambda row: (row["path"], row["qualified_name"], row["stable_symbol_id"]))


def rank_candidate_ids(*, rows, repository_id, scope_cid, semantic_root_cid,
                       query, database_path=None):
    """Rank one exact bounded snapshot with the legacy lexical score in SQL.

    All query text and identities are parameters. Historical and foreign rows
    cannot extend the supplied current population. A persistent database, when
    requested, retains an advisory copy of this snapshot for inspection/reuse;
    cold replay needs only the supplied rows and uses an in-memory database.
    """
    import duckdb

    identity = tuple(_text(value, name) for name, value in (
        ("repository_id", repository_id), ("scope_cid", scope_cid),
        ("semantic_root_cid", semantic_root_cid)))
    if type(query) is not str or len(query.encode()) > MAX_QUERY_BYTES:
        raise ValueError("candidate query must be text of at most 8192 bytes")
    checked = _validate_rows(rows)
    current = [row for row in checked
               if tuple(row[name] for name in _IDENTITY[:3]) == identity]
    target = ":memory:"
    if database_path is not None:
        path = Path(database_path)
        if path.name == "control.duckdb" or path.resolve().name == "control.duckdb":
            raise ValueError("candidate projection refuses exclusive control.duckdb")
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path)
    connection = duckdb.connect(target, config={"threads": 1, "memory_limit": "64MB",
        "temp_directory": "", "max_temp_directory_size": "0B"})
    try:
        connection.execute(_DDL)
        connection.execute("BEGIN TRANSACTION")
        connection.execute("""DELETE FROM semantic_capsule_candidates
            WHERE repository_id = ? AND scope_cid = ? AND semantic_root_cid = ?""", identity)
        if current:
            connection.executemany("""INSERT INTO semantic_capsule_candidates
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                [row[name] for name in ROW_FIELDS]
                + [_terms(row["qualified_name"] + " " + row["path"])] for row in current])
        ranked = connection.execute("""
            SELECT stable_symbol_id FROM semantic_capsule_candidates
            WHERE repository_id = ? AND scope_cid = ? AND semantic_root_cid = ?
            ORDER BY list_count(list_intersect(lexical_terms, ?::VARCHAR[])) DESC,
                     path ASC, qualified_name ASC, stable_symbol_id ASC
            LIMIT ?
            """, [*identity, _terms(query), MAX_ROWS]).fetchall()
        connection.execute("COMMIT")
        return [row[0] for row in ranked]
    except BaseException:
        # DDL/BEGIN may fail before a transaction exists; preserve that original
        # error rather than replacing it with a rollback diagnostic.
        try:
            connection.execute("ROLLBACK")
        except duckdb.Error:
            pass
        raise
    finally:
        connection.close()
