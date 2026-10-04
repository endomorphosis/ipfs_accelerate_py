import gc

import duckdb

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state


def test_reattach_transfers_native_handle_registry_custody(monkeypatch):
    original = duckdb.connect(":memory:")
    wrapper = duckdb_state.DuckDBConnection.wrap(original)
    wrapper._quack_uri = "quack://127.0.0.1:9999"
    candidate = duckdb.connect(":memory:")
    monkeypatch.setattr(duckdb_state, "open_quack_transport_connection",
                        lambda *args, **kwargs: duckdb_state.DuckDBConnection.wrap(candidate))
    try:
        wrapper._reattach_quack_transport()
        gc.collect()
        assert id(original) not in duckdb_state._RAW_WRAPPERS
        assert duckdb_state._RAW_WRAPPERS[id(candidate)][1]() is wrapper
        assert wrapper.execute("SELECT 42").fetchone()[0] == 42
    finally:
        wrapper.close()
    assert id(candidate) not in duckdb_state._RAW_WRAPPERS
