"""Defaults: DuckDB + Quack is the control plane; DuckLake is not."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    QuackTransportContentionError,
    _qualify_quack_statement,
    is_quack_transport_target,
    open_quack_transport_connection,
    persist_quack_attach_token_vault,
    quack_attach_lock_path,
    quack_token_vault_path,
    quack_transport_uri,
    reset_quack_transport_cache,
    resolve_quack_attach_token,
    unstall_stale_in_progress_tasks,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)


def test_qualify_quack_statement_prefixes_unqualified_tables() -> None:
    assert _qualify_quack_statement(
        "SELECT task_cid FROM tasks WHERE task_cid = ?",
        "control_plane",
    ) == "SELECT task_cid FROM control_plane.tasks WHERE task_cid = ?"
    assert _qualify_quack_statement(
        "SELECT count(*) FROM control_plane.tasks",
        "control_plane",
    ) == "SELECT count(*) FROM control_plane.tasks"
    assert _qualify_quack_statement(
        "SELECT d.task_cid FROM task_dependencies d JOIN tasks t ON t.task_cid = d.task_cid",
        "control_plane",
    ) == (
        "SELECT d.task_cid FROM control_plane.task_dependencies d "
        "JOIN control_plane.tasks t ON t.task_cid = d.task_cid"
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


def test_quack_attach_lock_path_follows_store_id(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "control.duckdb"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", raising=False)
    assert quack_attach_lock_path("quack:127.0.0.1:41327") == (
        store.resolve().parent / "quack-owner" / "attach.lock"
    )


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
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_mutation_dir,
    )

    assert quack_owner_mutation_dir() == store.resolve().parent / "quack-owner" / "mutations"


def test_relative_quack_store_never_uses_child_worktree_cwd(tmp_path, monkeypatch) -> None:
    accepted = tmp_path / "accepted"
    child = tmp_path / "child-worktree"
    accepted.mkdir()
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.setenv("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", str(accepted))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "state/control.duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        quack_owner_mutation_dir,
    )

    assert quack_owner_mutation_dir() == accepted / "state" / "quack-owner" / "mutations"


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


class _FakeQuackRaw:
    def __init__(self) -> None:
        self.description = (("value",),)
        self._rows = [(1,)]

    def execute(self, sql, params=None):
        del sql, params
        self.description = (("value",),)
        self._rows = [(1,)]
        return self

    def fetchall(self):
        rows = list(self._rows)
        self._rows = []
        return rows

    def close(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def test_resolve_quack_attach_token_prefers_vault_over_stale_env(
    tmp_path, monkeypatch
) -> None:
    vault = tmp_path / "env___IPFS_ACCELERATE_AGENT_QUACK_TOKEN.quack-token"
    vault.write_text("vaultTok_value1234567890\n", encoding="utf-8")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", str(vault))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "staleEnv_token_value")
    assert resolve_quack_attach_token() == "vaultTok_value1234567890"
    assert resolve_quack_attach_token("explicit_token_ok") == "explicit_token_ok"


def test_resolve_quack_attach_token_persists_missing_vault(
    tmp_path, monkeypatch
) -> None:
    store = tmp_path / "control.duckdb"
    store.write_bytes(b"")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(store))
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "liveTok_value1234567890")
    assert resolve_quack_attach_token() == "liveTok_value1234567890"
    vault = quack_token_vault_path()
    assert vault is not None
    assert vault.is_file()
    assert vault.stat().st_mode & 0o777 == 0o600
    assert vault.read_text(encoding="utf-8").strip() == "liveTok_value1234567890"


def test_intent_repository_reuses_quack_read_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens: list[object] = []

    class _FakeConnection:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            return type("R", (), {"fetchall": lambda self: []})()

        def close(self) -> None:
            return None

        description = None

        def fetchall(self) -> list[object]:
            return []

    def _fake_open(path: object, **_kwargs: object) -> _FakeConnection:
        opens.append(path)
        return _FakeConnection()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository.open_duckdb_connection",
        _fake_open,
    )
    repo = IntentRepository("quack:127.0.0.1:45123", install_schema=False)
    with repo._connection():
        pass
    with repo._connection():
        pass
    assert opens == ["quack:127.0.0.1:45123"]
    repo.close()


def test_intent_repository_keeps_quack_session_after_authorization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens: list[object] = []
    closed: list[int] = []

    class _FakeConnection:
        def execute(self, sql: str, *_args: object, **_kwargs: object) -> object:
            if "boom" in str(sql).lower():
                raise RuntimeError("Invalid Input Error: Authorization failed")
            return type("R", (), {"fetchall": lambda self: []})()

        def close(self) -> None:
            closed.append(1)

        description = None

        def fetchall(self) -> list[object]:
            return []

    def _fake_open(path: object, **_kwargs: object) -> _FakeConnection:
        opens.append(path)
        return _FakeConnection()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository.open_duckdb_connection",
        _fake_open,
    )
    repo = IntentRepository("quack:127.0.0.1:45123", install_schema=False)
    with repo._connection() as connection:
        connection.execute("SELECT 1")
    try:
        with repo._connection() as connection:
            connection.execute("SELECT boom FROM tasks")
    except RuntimeError:
        pass
    with repo._connection() as connection:
        connection.execute("SELECT 1")
    assert opens == ["quack:127.0.0.1:45123"]
    assert closed == []
    repo.close()


def test_intent_repository_reconnects_quack_after_authentication_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opens: list[object] = []

    class _FakeConnection:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, sql: str, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            if self.calls > 1 and "count(*)" not in str(sql).lower():
                raise RuntimeError("Invalid Input Error: Authentication failed")
            return type("R", (), {"fetchall": lambda self: []})()

        def close(self) -> None:
            return None

        description = None

        def fetchall(self) -> list[object]:
            return []

    def _fake_open(path: object, **_kwargs: object) -> _FakeConnection:
        opens.append(path)
        return _FakeConnection()

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository.open_duckdb_connection",
        _fake_open,
    )
    repo = IntentRepository("quack:127.0.0.1:45123", install_schema=False)
    with repo._connection() as connection:
        connection.execute("SELECT 1")
    try:
        with repo._connection() as connection:
            connection.execute("SELECT task_cid FROM tasks")
    except RuntimeError:
        pass
    with repo._connection() as connection:
        connection.execute("SELECT 1")
    assert opens == ["quack:127.0.0.1:45123", "quack:127.0.0.1:45123"]
    repo.close()


def test_resolve_quack_attach_token_follows_env_secret_handle() -> None:
    assert (
        resolve_quack_attach_token(
            "",
            environment={
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": (
                    "env://QUACK_TOKEN"
                ),
                "QUACK_TOKEN": "handle-target-token",
            },
        )
        == "handle-target-token"
    )
    assert (
        resolve_quack_attach_token(
            "",
            environment={
                "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "direct-token",
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": (
                    "env://QUACK_TOKEN"
                ),
                "QUACK_TOKEN": "handle-target-token",
            },
        )
        == "direct-token"
    )
    assert resolve_quack_attach_token("explicit-token", environment={}) == (
        "explicit-token"
    )


def test_persist_quack_attach_token_vault_does_not_overwrite(
    tmp_path, monkeypatch
) -> None:
    vault = tmp_path / "env___IPFS_ACCELERATE_AGENT_QUACK_TOKEN.quack-token"
    vault.write_text("vaultTok_value1234567890\n", encoding="utf-8")
    vault.chmod(0o600)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", str(vault))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "staleEnv_token_value")
    assert persist_quack_attach_token_vault() == vault
    assert vault.read_text(encoding="utf-8").strip() == "vaultTok_value1234567890"


def test_quack_attach_retries_authentication_failed_contention(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()
    attempts = {"n": 0}

    def fake_attach(uri: str, secret: str):
        del uri, secret
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Invalid Input Error: Authentication failed")
        return _FakeQuackRaw()

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    try:
        connection = open_quack_transport_connection("quack:127.0.0.1:41347")
        assert attempts["n"] == 3
        assert connection._pooled is True
    finally:
        reset_quack_transport_cache()


def test_quack_attach_reuses_cached_connection(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()
    attaches = {"n": 0}

    def fake_attach(uri: str, secret: str):
        del uri, secret
        attaches["n"] += 1
        return _FakeQuackRaw()

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    try:
        first = open_quack_transport_connection("quack:127.0.0.1:41347")
        first.close()
        second = open_quack_transport_connection("quack:127.0.0.1:41347")
        assert first is second
        assert attaches["n"] == 1
    finally:
        reset_quack_transport_cache()


def test_cached_quack_probe_waits_for_wrapper_transaction() -> None:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41347"
    reset_quack_transport_cache()
    cached = DuckDBConnection.wrap(duckdb.connect(":memory:"))
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    returned: list[DuckDBConnection] = []
    failures: list[BaseException] = []
    probe_returned = threading.Event()
    try:
        cached.execute("BEGIN TRANSACTION")

        def open_cached() -> None:
            try:
                returned.append(open_quack_transport_connection(uri))
                probe_returned.set()
            except BaseException as exc:
                failures.append(exc)

        peer = threading.Thread(target=open_cached)
        peer.start()
        assert not probe_returned.wait(timeout=0.1)
        cached.rollback()
        peer.join(timeout=3.0)

        assert not peer.is_alive()
        assert failures == []
        assert returned == [cached]
    finally:
        reset_quack_transport_cache()


def test_cached_probe_wait_does_not_hold_global_attach_lock(monkeypatch) -> None:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41349"
    reset_quack_transport_cache()
    cached = DuckDBConnection.wrap(duckdb.connect(":memory:"))
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    owner_ready = threading.Event()
    peer_probe_started = threading.Event()
    owner_may_reopen = threading.Event()
    owner_reopened = threading.Event()
    release_transaction = threading.Event()
    owner_ids: list[int] = []
    failures: list[BaseException] = []
    original_probe = ds._probe_quack_connection

    def observed_probe(connection) -> None:
        if owner_ids and threading.get_ident() != owner_ids[0]:
            peer_probe_started.set()
        original_probe(connection)

    monkeypatch.setattr(ds, "_probe_quack_connection", observed_probe)

    def transaction_owner() -> None:
        try:
            owner_ids.append(threading.get_ident())
            cached.execute("BEGIN TRANSACTION")
            owner_ready.set()
            if not owner_may_reopen.wait(timeout=3.0):
                raise AssertionError("owner reopen was not signalled")
            assert open_quack_transport_connection(uri) is cached
            owner_reopened.set()
            if not release_transaction.wait(timeout=3.0):
                raise AssertionError("transaction release was not signalled")
            cached.rollback()
        except BaseException as exc:
            failures.append(exc)

    def peer_open() -> None:
        try:
            assert open_quack_transport_connection(uri) is cached
        except BaseException as exc:
            failures.append(exc)

    owner = threading.Thread(target=transaction_owner, daemon=True)
    peer = threading.Thread(target=peer_open, daemon=True)
    owner.start()
    assert owner_ready.wait(timeout=3.0)
    peer.start()
    assert peer_probe_started.wait(timeout=3.0)
    owner_may_reopen.set()
    assert owner_reopened.wait(timeout=1.0)
    release_transaction.set()
    owner.join(timeout=3.0)
    peer.join(timeout=3.0)

    assert not owner.is_alive()
    assert not peer.is_alive()
    assert failures == []
    reset_quack_transport_cache()


def test_cache_reset_discards_cross_thread_transaction_owner() -> None:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41350"
    reset_quack_transport_cache()
    cached = DuckDBConnection.wrap(duckdb.connect(":memory:"))
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    transaction_started = threading.Event()
    owner_may_continue = threading.Event()
    failures: list[BaseException] = []

    def transaction_owner() -> None:
        cached.execute("BEGIN TRANSACTION")
        transaction_started.set()
        if not owner_may_continue.wait(timeout=3.0):
            failures.append(AssertionError("owner continuation was not signalled"))
            return
        try:
            cached.execute("COMMIT")
        except BaseException as exc:
            failures.append(exc)

    owner = threading.Thread(target=transaction_owner, daemon=True)
    owner.start()
    assert transaction_started.wait(timeout=3.0)
    reset_quack_transport_cache()
    owner_may_continue.set()
    owner.join(timeout=3.0)

    assert not owner.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], DuckDBConnectionPolicyError)
    assert cached.in_transaction is False
    with ds._QUACK_ATTACH_LOCK:
        assert uri not in ds._QUACK_TRANSPORT_CACHE


def test_cache_reset_preserves_rightful_context_owner_exception() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41351"
    reset_quack_transport_cache()
    cached = DuckDBConnection.wrap(
        _FakeQuackRaw(),
        transaction_on_context=True,
    )
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    entered = threading.Event()
    owner_may_raise = threading.Event()
    sentinel = RuntimeError("rightful context owner sentinel")
    failures: list[BaseException] = []

    def context_owner() -> None:
        try:
            with cached:
                entered.set()
                if not owner_may_raise.wait(timeout=3.0):
                    raise AssertionError("owner continuation was not signalled")
                raise sentinel
        except BaseException as exc:
            failures.append(exc)

    owner = threading.Thread(target=context_owner, daemon=True)
    owner.start()
    assert entered.wait(timeout=3.0)
    reset_quack_transport_cache()
    assert cached._context_depth == 1
    assert cached._context_owner != 0
    owner_may_raise.set()
    owner.join(timeout=3.0)

    assert not owner.is_alive()
    assert len(failures) == 1
    assert failures[0] is sentinel
    assert cached._context_depth == 0
    assert cached._context_owner == 0


def test_failed_cached_probe_preserves_context_owner_exception(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41352"
    reset_quack_transport_cache()
    cached = DuckDBConnection.wrap(
        _FakeQuackRaw(),
        transaction_on_context=True,
    )
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    entered = threading.Event()
    owner_may_raise = threading.Event()
    sentinel = RuntimeError("failed probe context owner sentinel")
    failures: list[BaseException] = []

    def context_owner() -> None:
        try:
            with cached:
                entered.set()
                if not owner_may_raise.wait(timeout=3.0):
                    raise AssertionError("owner continuation was not signalled")
                raise sentinel
        except BaseException as exc:
            failures.append(exc)

    def reject_cached_probe(connection) -> None:
        assert connection is cached
        raise RuntimeError("cached probe failed")

    monkeypatch.setattr(ds, "_probe_quack_connection", reject_cached_probe)
    monkeypatch.setattr(ds, "_attach_quack_once", lambda _uri, _secret: _FakeQuackRaw())
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    owner = threading.Thread(target=context_owner, daemon=True)
    owner.start()
    assert entered.wait(timeout=3.0)
    try:
        replacement = open_quack_transport_connection(uri)
        assert replacement is not cached
        assert cached._context_depth == 1
        assert cached._context_owner != 0
        owner_may_raise.set()
        owner.join(timeout=3.0)

        assert not owner.is_alive()
        assert len(failures) == 1
        assert failures[0] is sentinel
    finally:
        owner_may_raise.set()
        owner.join(timeout=3.0)
        reset_quack_transport_cache()


class _FailingTerminalRaw:
    def __init__(self) -> None:
        self.description = ()
        self.closed = False

    def execute(self, sql, params=None):
        del params
        if " ".join(str(sql).upper().split()) == "COMMIT":
            raise RuntimeError("commit outcome is unknown")
        self.description = ()
        return self

    def fetchall(self):
        return []

    def rollback(self) -> None:
        raise RuntimeError("rollback failed")

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("terminal", ["commit", "rollback"])
def test_uncertain_terminal_evicts_pool_and_notifies_waiting_peer(
    terminal: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    uri = "quack:127.0.0.1:41348"
    reset_quack_transport_cache()
    raw = _FailingTerminalRaw()
    cached = DuckDBConnection.wrap(raw)
    cached._pooled = True
    cached._quack_uri = uri
    with ds._QUACK_ATTACH_LOCK:
        ds._QUACK_TRANSPORT_CACHE[uri] = cached
    peer_failures: list[BaseException] = []
    try:
        cached.execute("BEGIN TRANSACTION")

        def peer_query() -> None:
            try:
                cached.execute("SELECT 1")
            except BaseException as exc:
                peer_failures.append(exc)

        peer = threading.Thread(target=peer_query)
        peer.start()
        if terminal == "commit":
            with pytest.raises(RuntimeError, match="unknown"):
                cached.execute("COMMIT")
        else:
            with pytest.raises(RuntimeError, match="rollback failed"):
                cached.rollback()
        peer.join(timeout=3.0)

        assert not peer.is_alive()
        assert len(peer_failures) == 1
        assert isinstance(peer_failures[0], DuckDBConnectionPolicyError)
        assert raw.closed is True
        assert cached.in_transaction is False
        with ds._QUACK_ATTACH_LOCK:
            assert uri not in ds._QUACK_TRANSPORT_CACHE
    finally:
        reset_quack_transport_cache()


def test_unstall_stale_in_progress_tasks_retries_dead_gate(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    now = datetime(2026, 8, 22, 18, 0, tzinfo=timezone.utc)
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "in_progress",
            9,
            (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-022",
            "PCCE-022",
            "in_progress",
            1,
            (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    result = unstall_stale_in_progress_tasks(connection, now=now, stale_seconds=16_200)
    aliases = [item["task_alias"] for item in result["unstalled"]]
    assert aliases == ["PCCE-021"]
    rows = {
        str(row[0]): (row[1], row[2])
        for row in connection.execute(
            "SELECT task_alias, status, revision FROM tasks"
        ).fetchall()
    }
    assert rows["PCCE-021"] == ("retrying", 10)
    assert rows["PCCE-022"] == ("in_progress", 1)


def test_unstall_drops_status_indexes_that_fatal_status_updates(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    connection = duckdb.connect(str(tmp_path / "control.duckdb"))
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL UNIQUE,
            goal_cid VARCHAR NOT NULL,
            ordinal BIGINT NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX tasks_goal_idx ON tasks(goal_cid, status)")
    connection.execute("CREATE INDEX tasks_status_idx ON tasks(status, ordinal)")
    now = datetime.now(timezone.utc)
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "goal:root",
            21,
            "in_progress",
            9,
            (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
    )
    result = unstall_stale_in_progress_tasks(connection, now=now, stale_seconds=16_200)
    assert [item["task_alias"] for item in result["unstalled"]] == ["PCCE-021"]
    assert any("tasks_status_idx" in sql for sql in result["status_indexes_rebuilt"])
    row = connection.execute(
        "SELECT status, revision FROM tasks WHERE task_alias = 'PCCE-021'"
    ).fetchone()
    assert tuple(row) == ("retrying", 10)
    names = {
        str(_row[0])
        for _row in connection.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
        ).fetchall()
    }
    assert "tasks_status_idx" in names
    assert "tasks_goal_idx" in names


def test_apply_owner_command_payload_unstalls_without_client_sql(tmp_path) -> None:
    import duckdb
    from datetime import datetime, timedelta, timezone

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        apply_owner_command_payload,
    )

    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_cid VARCHAR PRIMARY KEY,
            task_alias VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            revision BIGINT NOT NULL,
            updated_at VARCHAR NOT NULL
        )
        """
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
        [
            "cid-021",
            "PCCE-021",
            "in_progress",
            9,
            stale,
        ],
    )
    reply = apply_owner_command_payload(
        connection,
        {"op": "board_unstall", "stale_seconds": 16_200},
        environment={},
    )
    assert reply["ok"] is True
    assert reply["rowcount"] == 1
    assert reply["board_unstall"]["unstalled"][0]["task_alias"] == "PCCE-021"
    status = connection.execute(
        "SELECT status, revision FROM tasks WHERE task_alias = 'PCCE-021'"
    ).fetchone()
    assert tuple(status) == ("retrying", 10)


def test_typed_policy_rejects_legacy_board_unstall_before_any_mutation(
    tmp_path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        LEGACY_BOARD_UNSTALL_DISABLED,
        LEGACY_BOARD_UNSTALL_POLICY_ENV,
        apply_owner_command_payload,
        owner_should_recycle_for_board_unstall,
        request_owner_board_unstall,
    )

    class _NoMutationConnection:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("typed policy reached raw owner SQL")

    environment = {
        LEGACY_BOARD_UNSTALL_POLICY_ENV: LEGACY_BOARD_UNSTALL_DISABLED,
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR": str(tmp_path / "mutations"),
    }
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="disabled for typed task authority",
    ):
        apply_owner_command_payload(
            _NoMutationConnection(),
            {"op": "board_unstall", "stale_seconds": 16_200},
            environment=environment,
        )

    result = request_owner_board_unstall(
        wait=False,
        environment=environment,
    )
    assert result == {
        "ok": False,
        "requested": False,
        "error": "legacy_board_unstall_disabled",
    }
    assert not (tmp_path / "mutations").exists()

    bounce_dir = tmp_path / "old-owner-inbox"
    bounce_dir.mkdir()
    (bounce_dir / "board-unstall.bounce").write_text("{}\n", encoding="utf-8")
    assert owner_should_recycle_for_board_unstall(
        bounce_dir,
        min_age_seconds=0,
        environment=environment,
    ) is False


def test_request_owner_board_unstall_writes_inbox_without_waiting(
    tmp_path, monkeypatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        request_owner_board_unstall,
    )

    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    result = request_owner_board_unstall(wait=False)
    assert result["ok"] is True
    assert result["requested"] is True
    requests = list(inbox.glob("*.request.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload == {"op": "board_unstall", "stale_seconds": 16_200}
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        clear_owner_board_unstall_bounce,
        owner_should_recycle_for_board_unstall,
    )

    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is True
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=10_000) is False
    clear_owner_board_unstall_bounce(inbox)
    assert owner_should_recycle_for_board_unstall(inbox, min_age_seconds=0) is False


def test_quack_attach_exhausted_contention_raises_typed_error(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds

    reset_quack_transport_cache()

    def fake_attach(uri: str, secret: str):
        del uri, secret
        raise RuntimeError("Invalid Input Error: Authentication failed")

    monkeypatch.setattr(ds, "_attach_quack_once", fake_attach)
    monkeypatch.setattr(ds.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(ds, "resolve_quack_attach_token", lambda token="": "tok")
    monkeypatch.setattr(ds, "QUACK_ATTACH_ATTEMPTS", 3)
    try:
        with pytest.raises(QuackTransportContentionError, match="contended"):
            open_quack_transport_connection("quack:127.0.0.1:41347")
    finally:
        reset_quack_transport_cache()
