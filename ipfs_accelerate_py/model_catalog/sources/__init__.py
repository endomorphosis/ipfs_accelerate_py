"""Pure source adapters for the AI service catalog."""

from .persistent import (
    DEFAULT_PERSISTENT_PRECEDENCE,
    PersistentCatalogSource,
    PersistentSourceAdapter,
    adapt_persistent_source,
    load_persistent_catalog,
)
from .static import (
    DEFAULT_STATIC_PRECEDENCE,
    MAX_DIAGNOSTICS,
    MAX_ROW_FIELDS,
    MAX_SOURCE_BYTES,
    MAX_SOURCE_REVISION_BYTES,
    CatalogSourceResult,
    SourceDiagnostic,
    SourceMetadata,
    StaticCatalogSource,
    StaticSourceAdapter,
    adapt_static_source,
    load_static_catalog,
)

__all__ = [
    "CatalogSourceResult",
    "DEFAULT_PERSISTENT_PRECEDENCE",
    "DEFAULT_STATIC_PRECEDENCE",
    "MAX_DIAGNOSTICS",
    "MAX_ROW_FIELDS",
    "MAX_SOURCE_BYTES",
    "MAX_SOURCE_REVISION_BYTES",
    "PersistentCatalogSource",
    "PersistentSourceAdapter",
    "SourceDiagnostic",
    "SourceMetadata",
    "StaticCatalogSource",
    "StaticSourceAdapter",
    "adapt_persistent_source",
    "adapt_static_source",
    "load_persistent_catalog",
    "load_static_catalog",
]
