"""Defaults: DuckDB + Quack is the control plane; DuckLake is not."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    is_quack_transport_target,
    open_quack_transport_connection,
    quack_transport_uri,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)


def test_loopback_quack_uri_is_accepted() -> None:
    assert is_quack_transport_target("quack:127.0.0.1:45123") is True
    assert is_quack_transport_target("quack://127.0.0.1:9") is True
    assert is_quack_transport_target("control.duckdb") is False
    assert is_quack_transport_target("quack:8.8.8.8:45123") is False


def test_absolutized_quack_path_is_still_transport() -> None:
    uri = "quack:127.0.0.1:45123"
    prefixed = Path.cwd() / uri
    assert is_quack_transport_target(prefixed) is True
    assert quack_transport_uri(prefixed) == uri
    collapsed = Path.cwd() / "quack:/127.0.0.1:45123"
    assert quack_transport_uri(collapsed) == "quack://127.0.0.1:45123"


def test_quack_transport_rejects_non_loopback() -> None:
    with pytest.raises(DuckDBConnectionPolicyError, match="non-loopback"):
        open_quack_transport_connection("quack:10.0.0.1:45123")


def test_file_adapter_refuses_to_create_quack_named_database(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DuckDBConnectionPolicyError, match="cannot be opened as DuckDB files"):
        DuckDBConnection("quack:127.0.0.1:45123")
    assert not (tmp_path / "quack:127.0.0.1:45123").exists()


def test_database_task_source_keeps_quack_uri(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeIntent:
        database_path = "quack:127.0.0.1:45123"

        def close(self) -> None:
            return None

    def _fake_open(path, **_kwargs):
        seen["path"] = path
        return _FakeIntent()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.open_intent_repository",
        _fake_open,
    )
    source = DatabaseTaskSource("quack:127.0.0.1:45123", install_schema=False)
    assert seen["path"] == "quack:127.0.0.1:45123"
    assert source.database_path == "quack:127.0.0.1:45123"
    assert IntentRepository("quack:127.0.0.1:45123", install_schema=False)._quack_transport is True
    prefixed = Path.cwd() / "quack:127.0.0.1:45123"
    repo = IntentRepository(prefixed, install_schema=False)
    assert repo._quack_transport is True
    assert repo._open_target == "quack:127.0.0.1:45123"


def test_quack_mutation_dir_follows_store_id(tmp_path, monkeypatch) -> None:
    store = tmp_path / "control.duckdb"
    store.write_bytes(b"")
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_mutation_dir,
    )

    assert quack_owner_mutation_dir() == store.resolve().parent / "quack-owner" / "mutations"


def test_quack_mutation_dir_rejects_missing_or_mismatched_registry_binding(
    tmp_path,
    monkeypatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_mutation_dir,
    )

    legacy_fallback = tmp_path / "quack-owner" / "mutations"
    registry = tmp_path / "registry"
    wrong = tmp_path / "wrong" / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE", "quack")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        str(tmp_path / "control.duckdb"),
    )
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH", raising=False)
    with pytest.raises(DuckDBConnectionPolicyError, match="bound runtime registry"):
        quack_owner_mutation_dir()
    assert not legacy_fallback.exists()

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(wrong),
    )
    with pytest.raises(DuckDBConnectionPolicyError, match="bound runtime registry"):
        quack_owner_mutation_dir()

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
        str(registry),
    )
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    with pytest.raises(DuckDBConnectionPolicyError, match="explicit mutation binding"):
        quack_owner_mutation_dir()
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(wrong),
    )
    with pytest.raises(DuckDBConnectionPolicyError, match="does not match"):
        quack_owner_mutation_dir()
    assert not registry.exists()
    assert not wrong.exists()


def test_quack_mutation_timeout_is_unknown_outcome_without_internal_replay(
    tmp_path,
    monkeypatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        duckdb_state as duckdb_state_module,
    )

    registry = tmp_path / "runtime-registry"
    inbox = registry / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE", "quack")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
        str(registry),
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(inbox),
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "store:timeout-unknown",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "7")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "timeout-unknown-token",
    )
    monotonic_values = iter((0.0, 16.0))
    monkeypatch.setattr(
        duckdb_state_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    client = DuckDBConnection.wrap(object())
    client._default_catalog = "control_plane"  # noqa: SLF001
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="outcome is unknown and must not be replayed blindly",
    ):
        client.execute(
            "UPDATE tasks SET status = ? WHERE task_cid = ?",
            ["in_progress", "task:timeout"],
        )

    # One call publishes one stable request and leaves it for the owner to
    # resolve. The worker does not create a second request on timeout.
    assert len(list(inbox.glob("*.request.json"))) == 1
    assert not list(inbox.glob("*.done.json"))


def test_database_daemon_defaults_to_quack_and_refuses_file_open(tmp_path) -> None:
    with pytest.raises(DatabaseImplementationAuthorityError, match="loopback quack"):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            task_source_kind="duckdb",
        )
