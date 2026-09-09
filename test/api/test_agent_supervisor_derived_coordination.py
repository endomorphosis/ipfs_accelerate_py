"""Real owner/client boundary for separate derived coordination state."""

from __future__ import annotations

import hashlib
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
