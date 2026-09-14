"""Quack clients must not each inherit a machine-sized DuckDB resource budget."""

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state


def test_quack_client_limits_apply_before_extension_load(monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    native_connect = duckdb.connect
    observed = {}

    class BeforeExtensionLoad:
        def __init__(self, raw):
            self.raw = raw

        def execute(self, sql):
            assert sql == "LOAD quack"
            observed["threads"] = self.raw.execute(
                "SELECT current_setting('threads')"
            ).fetchone()[0]
            observed["memory_limit"] = self.raw.execute(
                "SELECT current_setting('memory_limit')"
            ).fetchone()[0]
            # No extension or network access is needed to exercise the real
            # connection's settings and the failed-attach cleanup path.
            raise RuntimeError("synthetic extension load failure")

        def close(self):
            self.raw.close()
            observed["closed"] = True

    def connect(*args, **kwargs):
        return BeforeExtensionLoad(native_connect(*args, **kwargs))

    monkeypatch.setattr(duckdb, "connect", connect)
    with pytest.raises(RuntimeError, match="synthetic extension load failure"):
        duckdb_state.open_quack_transport_connection("quack:127.0.0.1:45123", token="")

    assert observed["threads"] == 1
    # DuckDB reports decimal 256MB as 244.1 MiB. Compare with a separate
    # explicitly bounded native connection instead of parsing display units.
    with native_connect(":memory:", config={"memory_limit": "256MB", "threads": 1}) as expected:
        assert observed["memory_limit"] == expected.execute(
            "SELECT current_setting('memory_limit')"
        ).fetchone()[0]
    assert observed["closed"] is True
