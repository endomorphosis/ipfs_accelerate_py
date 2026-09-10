"""Real owner/client boundary for separate derived coordination state."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import (
    DerivedCoordinationClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection,
    TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientIdentityError,
    QuackStateClient,
)


@pytest.fixture
def owner(tmp_path):
    db = tmp_path / "control.duckdb"
    _install(db)
    gateway, connection = _gateway(db, tmp_path / "owner.sock")
    token = gateway.bind_derived_coordination_service()

    def attach(repository="repo:one"):
        return TypedStateOwnerConnection(
            socket_path=gateway.socket_path,
            token=token,
            client_id="ast-client",
            process_birth_id="birth:ast-client",
            store_id="control.duckdb",
            derived_repository_id=repository,
        )

    yield gateway, connection, attach
    gateway.stop()
    connection.close()


def test_separate_native_owner_deduplicates_ast_without_client_file_access(
    owner, monkeypatch
):
    gateway, connection, attach = owner
    from ipfs_accelerate_py.agent_supervisor.analysis import duckdb_ast_index

    monkeypatch.setattr(
        duckdb_ast_index,
        "open_duckdb_connection",
        lambda *a, **kw: pytest.fail("second writer"),
    )
    client = attach()
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        files = [{"path": "main.py", "content": "def hello():\n    return 1\n"}]
        first = api.call("ingest_snapshot", tree_id="tree:one", files=files)
        repeated = api.call("ingest_snapshot", tree_id="tree:one", files=files)
        second = api.call("ingest_snapshot", tree_id="tree:two", files=files)
        assert first["result"] == repeated["result"]
        assert first["result"]["new_unit_count"] == 1
        assert second["result"]["reused_unit_count"] == 1
        snapshot_id = first["result"]["snapshot"]["snapshot_id"]
        observation = api.call("snapshot", snapshot_id=snapshot_id)
        assert observation["result"]["symbols"][0]["qualified_name"] == "hello"
        assert observation["completion_authority"] is False
        assert observation["owner_identity"] == dict(client.identity)
        assert "return 1" not in str(
            connection.execute("SELECT facts_json FROM parse_cache").fetchall()
        )
        gateway._derived_coordination_service._ast.close()
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    "violation", ["scope", "body", "bytes", "files", "sql", "task_write"]
)
def test_native_boundary_rejects_scope_conflicts_unbounded_work_and_board_writes(
    owner, violation
):
    _, _, attach = owner
    client = attach()
    api = DerivedCoordinationClient(client, repository_id="repo:one")
    files = [{"path": "main.py", "content": "x = 1"}]
    api.call("ingest_snapshot", tree_id="tree:one", files=files)
    try:
        with pytest.raises(TypedStateOwnerError):
            if violation == "scope":
                DerivedCoordinationClient(client, repository_id="repo:other").call(
                    "snapshot", snapshot_id="anything"
                )
            elif violation == "body":
                api.call(
                    "ingest_snapshot",
                    tree_id="tree:one",
                    files=[{"path": "main.py", "content": "x = 2"}],
                )
            elif violation == "bytes":
                api.call(
                    "ingest_snapshot",
                    tree_id="tree:two",
                    files=[{"path": "big.py", "content": "a" * 32769}],
                )
            elif violation == "files":
                api.call("ingest_snapshot", tree_id="tree:two", files=files * 9)
            elif violation == "sql":
                api.call("execute_sql", sql="SELECT 1")
            else:
                client.execute_operation(
                    "txn_cas_task_status", ["done", "task:typed-owner", "ready", 0]
                )
    finally:
        client.close()


def test_datasets_guard_allows_references_without_local_ast_writer(owner, monkeypatch):
    _, connection, attach = owner
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py"
    )
    client = attach()
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        record = dict(
            tree_id="tree:source",
            ast_cid="cid:datasets-ast",
            content_hash="sha256:" + "a" * 64,
            state_root="cid:kit-state",
        )
        result = api.call("record_reference", **record)
        assert result["result"]["source_reference_verified"] is False
        assert api.call("list_references", tree_id="tree:source")["result"][
            "references"
        ] == [result["result"]["reference"]]
        with pytest.raises(TypedStateOwnerError):
            api.call(
                "ingest_snapshot",
                tree_id="tree:new",
                files=[{"path": "a.py", "content": "pass"}],
            )
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'ast_index_metadata'"
        ).fetchall()
        assert not tables
    finally:
        client.close()


def test_concurrent_native_clients_share_cache_and_preserve_repository_read_scope(
    owner,
):
    _, _, attach = owner

    def ingest(index):
        client = attach()
        try:
            api = DerivedCoordinationClient(client, repository_id="repo:one")
            return api.call(
                "ingest_snapshot",
                tree_id=f"tree:{index}",
                files=[{"path": "a.py", "content": "x = 1"}],
            )["result"]
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ingest, range(4)))
    assert sum(row["new_unit_count"] for row in results) == 1
    assert sum(row["reused_unit_count"] for row in results) == 3
    client = attach("repo:two")
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:two")
        digest = "sha256:" + hashlib.sha256(b"x = 1").hexdigest()
        assert (
            api.call("parse_cache", content_hash=digest)["result"]["cache_entry"]
            is None
        )
        with pytest.raises(TypedStateOwnerError):
            api.call("snapshot", snapshot_id=results[0]["snapshot"]["snapshot_id"])
    finally:
        client.close()


def _public_client(attach):
    client = QuackStateClient(owner_id="ast-client", store_id="control.duckdb",
                              process_birth_id="birth:ast-client", connection_factory=lambda _endpoint: attach())
    try:
        client.attach("quack:127.0.0.1:12345")
        return client
    except BaseException:
        client.close()
        raise


def test_public_quack_facade_supports_derived_owner_calls(owner):
    _, _, attach = owner
    client = _public_client(attach)
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        result = api.call("list_references", tree_id="tree:public")
        assert result["result"]["references"] == []
        assert result["owner_identity"]["server_id"] == client.session.server_id
        assert result["completion_authority"] is False
    finally:
        client.close()


def test_scoped_sessions_survive_idle_grant_expiry_without_replaying(owner, monkeypatch):
    _, _, attach = owner
    fixed = _public_client(attach)
    opened = []
    sessions = []

    def factory():
        client = _public_client(attach)
        opened.append(client)
        sessions.append(client.session.session_id)
        return client

    api = DerivedCoordinationClient(repository_id="repo:one", connection_factory=factory)
    try:
        first = api.call("list_references", tree_id="tree:idle")
        later = time.time() + 121
        monkeypatch.setattr(time, "time", lambda: later)
        with pytest.raises(TypedStateOwnerError):
            fixed.derived_coordination({"operation": "list_references", "repository_id": "repo:one", "tree_id": "tree:idle"})
        second = api.call("list_references", tree_id="tree:idle")
        assert first["result"] == second["result"]
        assert len(opened) == 2
        assert sessions[0] != sessions[1]
        assert all(not client.attached for client in opened)
    finally:
        fixed.close()


@pytest.mark.parametrize("field,value", [("database_uuid", "different"), ("generation", -1), ("process_birth_id", "different")])
def test_public_derived_facade_rejects_changed_owner(owner, monkeypatch, field, value):
    _, _, attach = owner
    raw = attach()
    client = QuackStateClient(owner_id="ast-client", store_id="control.duckdb",
                              process_birth_id="birth:ast-client", connection_factory=lambda _endpoint: raw)
    try:
        client.attach("quack:127.0.0.1:12345")
        original = raw.derived_coordination

        def changed(payload):
            response = original(payload)
            return {**response, "owner_identity": {**response["owner_identity"], field: value}}

        monkeypatch.setattr(raw, "derived_coordination", changed)
        with pytest.raises(QuackClientIdentityError):
            client.derived_coordination({"operation": "list_references", "repository_id": "repo:one", "tree_id": "tree:identity"})
    finally:
        client.close()


def test_scoped_derived_write_closes_on_unknown_outcome_without_replay():
    calls = []

    class BrokenConnection:
        def derived_coordination(self, payload):
            calls.append(payload)
            raise TimeoutError("outcome unknown")

        def close(self):
            calls.append("closed")

    api = DerivedCoordinationClient(repository_id="repo:one", connection_factory=BrokenConnection)
    with pytest.raises(TimeoutError):
        api.call("record_reference", tree_id="tree:one", ast_cid="cid:ast", content_hash="sha256:source", state_root="cid:state")
    assert len(calls) == 2 and calls[-1] == "closed"
    with pytest.raises(ValueError, match="scope"):
        api.call("list_references", tree_id="tree:one", repository_id="repo:other")
    assert len(calls) == 2
