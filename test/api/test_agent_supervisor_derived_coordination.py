"""Real owner/client boundary for separate derived coordination state."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from test.api.causal_federation.test_typed_state_owner import _gateway, _install
from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import (
    DerivedCoordinationClient,
)
from ipfs_accelerate_py.agent_supervisor.analysis.derived_artifacts import (
    ARTIFACT_KINDS, ARTIFACT_SCHEMA,
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


def _artifact(kind="vector_embeddings", **changes):
    return dict(tree_id="git:source-tree", artifact_kind=kind,
                input_digest="sha256:" + "a" * 64,
                producer_id="datasets:test-producer", producer_revision="git:pinned-producer",
                parameters_digest="sha256:" + "b" * 64, **changes)


@pytest.mark.parametrize("kind", ARTIFACT_KINDS)
def test_artifacts_share_immutable_references_without_claiming_verification(owner, monkeypatch, kind):
    gateway, _, attach = owner
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    first, second = _public_client(attach), _public_client(attach)
    try:
        writer = DerivedCoordinationClient(first, repository_id="repo:one")
        reader = DerivedCoordinationClient(second, repository_id="repo:one")
        metadata = _artifact(kind)
        recorded = writer.call("record_artifact", **metadata, artifact_cid="cid:test-artifact")
        assert reader.call("lookup_artifact", **metadata)["result"] == recorded["result"]
        assert writer.call("record_artifact", **metadata, artifact_cid="cid:test-artifact")["result"] == recorded["result"]
        with pytest.raises(TypedStateOwnerError, match="operation_failed"):
            writer.call("record_artifact", **metadata, artifact_cid="cid:different-content")
        assert reader.call("lookup_artifact", **metadata)["result"] == recorded["result"]
        assert recorded["result"]["artifact"]["schema"] == ARTIFACT_SCHEMA
        assert recorded["result"]["artifact_verified"] is False
        assert recorded["completion_authority"] is False
        assert recorded["authority"] == "derived_evidence"
        assert gateway._derived_coordination_service._ast is None
        capabilities = reader.call("capabilities")["result"]
        assert capabilities["artifact_kinds"] == list(ARTIFACT_KINDS)
        assert capabilities["artifact_schema"] == ARTIFACT_SCHEMA
    finally:
        first.close()
        second.close()


def test_artifact_reuse_requires_exact_source_producer_inputs_and_configuration(owner):
    _, _, attach = owner
    client, other = attach(), attach("repo:two")
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        metadata = _artifact()
        api.call("record_artifact", **metadata, artifact_cid="cid:test-artifact")
        for field, value in dict(tree_id="git:changed", input_digest="sha256:" + "c" * 64,
                                 producer_id="different-producer", producer_revision="git:new-producer",
                                 parameters_digest="sha256:" + "d" * 64,
                                 artifact_kind="proof_cache").items():
            assert api.call("lookup_artifact", **{**metadata, field: value})["result"]["artifact"] is None
        scoped = DerivedCoordinationClient(other, repository_id="repo:two")
        assert scoped.call("lookup_artifact", **metadata)["result"]["artifact"] is None
        assert scoped.call("list_artifacts", tree_id=metadata["tree_id"], artifact_kind=metadata["artifact_kind"])["result"]["artifacts"] == []
        with pytest.raises(TypedStateOwnerError):
            DerivedCoordinationClient(client, repository_id="repo:two").call("lookup_artifact", **metadata)
    finally:
        client.close()
        other.close()


@pytest.mark.parametrize("changes", [
    {"verified": True}, {"sql": "DELETE FROM tasks"}, {"artifact_kind": "task_completion"},
    {"input_digest": "unhashed"}, {"parameters_digest": "sha256:short"},
    {"producer_revision": ""}, {"artifact_cid": " "}, {"artifact_cid": "cid:\ninvalid"},
    {"artifact_cid": "x" * 257}, {"producer_id": {"nested": "object"}},
])
def test_artifact_native_boundary_rejects_unbound_or_unbounded_records(owner, changes):
    _, _, attach = owner
    client = attach()
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        with pytest.raises(TypedStateOwnerError):
            api.call("record_artifact", **{**_artifact(), "artifact_cid": "cid:test", **changes})
        assert api.call("lookup_artifact", **_artifact())["result"]["artifact"] is None
    finally:
        client.close()


def test_artifact_pages_are_bounded_and_repository_scoped(owner):
    _, _, attach = owner
    client = attach()
    try:
        api = DerivedCoordinationClient(client, repository_id="repo:one")
        expected = []
        for index in range(67):
            result = api.call("record_artifact", **{**_artifact(), "producer_revision": f"git:{index}"}, artifact_cid=f"cid:{index}")
            expected.append(result["result"]["artifact"]["artifact_key"])
        page = api.call("list_artifacts", tree_id="git:source-tree", artifact_kind="vector_embeddings")["result"]
        assert len(page["artifacts"]) == 64
        assert page["has_more"] is True
        assert len(json.dumps(page).encode()) < 262144
        rest = api.call("list_artifacts", tree_id="git:source-tree", artifact_kind="vector_embeddings", after=page["next_cursor"])["result"]
        assert rest["has_more"] is False
        assert rest["next_cursor"] == ""
        assert [row["artifact_key"] for row in page["artifacts"] + rest["artifacts"]] == sorted(expected)
        for changes in ({"limit": True}, {"limit": 0}, {"limit": 65}, {"after": "raw offset"}):
            with pytest.raises(TypedStateOwnerError):
                api.call("list_artifacts", tree_id="git:source-tree", artifact_kind="vector_embeddings", **changes)
    finally:
        client.close()


def test_concurrent_artifact_publishers_cannot_replace_winning_publication(owner):
    _, _, attach = owner

    def publish(cid):
        client = attach()
        try:
            api = DerivedCoordinationClient(client, repository_id="repo:one")
            return api.call("record_artifact", **_artifact(), artifact_cid=cid)["result"]["artifact"]
        except TypedStateOwnerError as error:
            assert "operation_failed" in str(error)
            return None
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(publish, ["cid:first", "cid:second"] * 2))
    accepted = [record for record in result if record]
    assert len(accepted) == 2
    assert accepted[0] == accepted[1]


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
