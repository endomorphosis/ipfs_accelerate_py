"""MCP++ persistence engines.

Primary store is **DuckDB**, with best-effort **Quack** and **DuckLake**
extension load (no network INSTALL). SQLite remains an explicit fallback.
"""

from ipfs_accelerate_py.mcp_server.mcplusplus.storage.engine import (
    DEFAULT_ENGINE,
    DUCKLAKE_CATALOG_ENV,
    ENGINE_ENV,
    EngineConnection,
    EngineError,
    connect_sql_engine,
    loaded_extensions,
    resolve_sql_engine,
)

__all__ = [
    "DEFAULT_ENGINE",
    "DUCKLAKE_CATALOG_ENV",
    "ENGINE_ENV",
    "EngineConnection",
    "EngineError",
    "connect_sql_engine",
    "loaded_extensions",
    "resolve_sql_engine",
]
