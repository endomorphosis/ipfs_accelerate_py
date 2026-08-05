"""Adapter for persisted ModelManager metadata.

This module reuses the static projection rules but gives persisted local
metadata a distinct source identity and merge precedence.  It never constructs
a ModelManager (which could open databases, inspect credentials, or initialize
optional backends); callers inject already loaded records or an explicit local
JSON/JSONL path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .static import CatalogSourceResult, StaticCatalogSource

DEFAULT_PERSISTENT_PRECEDENCE = 20
_DATABASE_SUFFIXES = frozenset((".duckdb", ".ddb"))


def _read_duckdb(path: Path, max_records: int) -> Any:
    """Read only the ModelManager table from an explicitly supplied database."""

    if not path.is_file():
        raise ValueError("catalog source path is not a local file")
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ValueError("DuckDB is required to read this persistent source") from exc
    connection = None
    try:
        connection = duckdb.connect(str(path), read_only=True)
        cursor = connection.execute("SELECT * FROM model_metadata LIMIT ?", [max_records + 1])
        columns = tuple(item[0] for item in cursor.description)
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    except Exception as exc:
        raise ValueError("persistent catalog database could not be read") from exc
    finally:
        if connection is not None:
            connection.close()
    if len(rows) > max_records:
        raise ValueError("source exceeds maximum record count")
    return rows


class PersistentCatalogSource(StaticCatalogSource):
    """Project an injected ModelManager JSON mapping into catalog records."""

    def __init__(
        self,
        records: Any = None,
        *,
        path: Optional[Any] = None,
        source: str = "model-manager.persistent",
        precedence: Optional[int] = None,
        revision: Optional[str] = None,
        observed_at: Optional[Any] = None,
        default_provider: str = "local",
        max_records: int = 10_000,
    ) -> None:
        super().__init__(
            records,
            path=path,
            source=source,
            precedence=precedence,
            revision=revision,
            observed_at=observed_at,
            default_provider=default_provider,
            max_records=max_records,
            default_precedence=DEFAULT_PERSISTENT_PRECEDENCE,
        )

    def _supplied_value(self) -> Any:
        if self._path is not None and self._path.suffix.casefold() in _DATABASE_SUFFIXES:
            return _read_duckdb(self._path, self.max_records)
        return super()._supplied_value()


PersistentSourceAdapter = PersistentCatalogSource


def adapt_persistent_source(
    records: Any = None, *, path: Optional[Any] = None, **kwargs: Any
) -> CatalogSourceResult:
    """Adapt injected persisted records or one explicit local path."""

    return PersistentCatalogSource(records, path=path, **kwargs).load()


load_persistent_catalog = adapt_persistent_source


__all__ = [
    "DEFAULT_PERSISTENT_PRECEDENCE",
    "PersistentCatalogSource",
    "PersistentSourceAdapter",
    "adapt_persistent_source",
    "load_persistent_catalog",
]
