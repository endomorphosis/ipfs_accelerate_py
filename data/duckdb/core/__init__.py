"""
Core DuckDB API functionality for benchmark database operations.
"""

# Expose key objects from the modules
try:
    from data.duckdb.core.benchmark_db_api import *
    from data.duckdb.core.benchmark_db_query import *
    from data.duckdb.core.benchmark_db_maintenance import *
except ImportError:
    # Modules may not yet expose these
    pass
