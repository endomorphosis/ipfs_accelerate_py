"""Real datasets bundle and a separate private typed owner, never native state."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import (
    DerivedCoordinationClient,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    SemanticStateAdapterError,
    load_semantic_state_provider,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.derived_coordination import (
    PROFILE,
    CoordinatedSemanticStateProvider,
    DerivedSemanticCoordinationError,
)


def _owner(pipe, database, state):
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_FILENAME,
        compact_default_owner_socket_path,
    )
    from test.api.causal_federation.test_typed_state_owner import _gateway, _install

    database, state = Path(database), Path(state)
    _install(database)
    gateway, connection = _gateway(
        database,
        compact_default_owner_socket_path(
            state / TYPED_STATE_OWNER_SOCKET_FILENAME, identity=database
        ),
    )
    try:
        token = state / "derived-coordination.token"
        token.write_text(gateway.bind_derived_coordination_service())
        token.chmod(0o600)
        before = connection.execute(
            "SELECT task_cid,status,revision FROM tasks ORDER BY task_cid"
        ).fetchall()
        pipe.send({"pid": os.getpid(), "ready": True})
        while pipe.poll(60):
            request = pipe.recv()
            if request == "stop":
                break
            assert request == "inspect"
            after = connection.execute(
                "SELECT task_cid,status,revision FROM tasks ORDER BY task_cid"
            ).fetchall()
            tables = connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name='ast_index_metadata'"
            ).fetchall()
            count = connection.execute(
                "SELECT COUNT(*) FROM derived_coordination_references"
            ).fetchone()[0]
            pipe.send(
                {
                    "tasks_unchanged": before == after,
                    "ast_writer_constructed": bool(tables),
                    "references": count,
                }
            )
    finally:
        gateway.stop()
        connection.close()
        pipe.close()


@pytest.fixture
def owner(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_topology import (
        DEFAULTS,
        SCHEMA,
    )

    state = tmp_path / "owner"
    state.mkdir()
    db = tmp_path / "control.duckdb"
    path = tmp_path / "deployment.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "instances": {
                    "derived_coordination": {
                        "managed_by_fleet": True,
                        "database_path": str(db),
                        "state_dir": str(state),
                        "database_program": {
                            **DEFAULTS,
                            "store_id": "control.duckdb",
                            "quack_endpoint": "quack:127.0.0.1:12345",
                            "endpoint_secret_handle": "handle:test",
                            "store_generation": "1",
                            "schema_revision": "1",
                        },
                    }
                },
            }
        )
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_owner, args=(child, str(db), str(state)))
    process.start()
    child.close()
    try:
        assert parent.poll(20), "private owner startup deadline"
        assert parent.recv() == {"pid": process.pid, "ready": True}
        yield path, parent, process.pid
    finally:
        if process.is_alive():
            try:
                parent.send("stop")
            except (BrokenPipeError, EOFError):
                pass
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        assert not process.is_alive()


@pytest.fixture
def source(tmp_path):
    pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state.api")
    provider = load_semantic_state_provider()
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "module.py").write_text(
        "def answer(value: int) -> int:\n    return value + 1\n"
    )

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "source",
    )
    state = provider.scan_repository(repo)
    return provider, state


def scoped(path, state, client_id):
    return DerivedCoordinationClient.from_fleet_deployment(
        path, repository_id=state.repository_id, client_id=client_id
    )


def test_actual_producer_and_second_supervisor_reopen_same_datasets_bytes_from_separate_owner(
    owner, source, monkeypatch
):
    path, pipe, pid = owner
    provider, state = source
    assert pid != os.getpid()
    # Both supervisors have only typed sessions. Only the child owns the DB.
    import duckdb

    monkeypatch.setattr(
        duckdb,
        "connect",
        lambda *a, **k: pytest.fail("supervisor opened derived database"),
    )
    writer = load_semantic_state_provider(
        derived_coordination_client=scoped(path, state, "supervisor:producer")
    )
    bundle = writer.build_semantic_state(state)
    assert writer.last_coordination["status"] == "recorded"
    assert writer.last_coordination["schema"] == PROFILE
    assert bundle.root.producer.repository_state_cid == state.state_cid
    # Existing provider determinism and exact object content are unchanged.
    cold = provider.build_semantic_state(state)
    assert cold.root.root_cid == bundle.root.root_cid and dict(cold.blocks) == dict(
        bundle.blocks
    )
    consumer = CoordinatedSemanticStateProvider(
        provider, scoped(path, state, "supervisor:consumer")
    )
    found = list(
        consumer.iter_discovered_views(
            tree_id=bundle.root.producer.repository_snapshot_cid,
            get_block=dict(bundle.blocks).__getitem__,
        )
    )
    assert len(found) == 1
    observed = found[0]
    assert observed.reference["state_root"] == bundle.root.root_cid
    assert observed.reference["ast_cid"] == state.state_cid
    assert (
        observed.reference["content_hash"]
        == "sha256:"
        + hashlib.sha256(bundle.get_block(bundle.root.root_cid)).hexdigest()
    )
    assert (
        observed.completion_authority is False
        and observed.current_root_authority is False
    )
    for symbol in state.symbols:
        assert (
            observed.view.symbol_node(symbol.stable_id).node_cid
            == provider.view_semantic_state_bundle(bundle)
            .symbol_node(symbol.stable_id)
            .node_cid
        )
    pipe.send("inspect")
    assert pipe.poll(5)
    assert pipe.recv() == {
        "tasks_unchanged": True,
        "ast_writer_constructed": False,
        "references": 1,
    }


def test_coordinator_unknown_response_preserves_completed_producer_without_replay(
    source, monkeypatch
):
    provider, state = source
    calls = []
    closed = []
    built = []
    original = provider._api.build_semantic_state

    def build(*a, **k):
        result = original(*a, **k)
        built.append(result)
        return result

    monkeypatch.setattr(provider._api._resolve(), "build_semantic_state", build)

    def request(payload):
        calls.append(payload)
        raise TimeoutError("private credential must not appear")

    client = DerivedCoordinationClient(
        repository_id=state.repository_id,
        connection_factory=lambda: SimpleNamespace(
            derived_coordination=request, close=lambda: closed.append(True)
        ),
    )
    wrapper = CoordinatedSemanticStateProvider(provider, client)
    bundle = wrapper.build_semantic_state(state)
    assert bundle is built[0] and len(built) == len(calls) == len(closed) == 1
    assert wrapper.last_coordination["status"] == "unknown"
    assert wrapper.last_coordination["source_result_preserved"] is True
    assert "credential" not in str(wrapper.last_coordination)


@pytest.mark.parametrize("corruption", ["scope", "state", "hash", "missing", "corrupt"])
def test_discovery_cannot_replace_datasets_verification(owner, source, corruption):
    path, _, _ = owner
    provider, state = source
    bundle = provider.build_semantic_state(state)
    root = bundle.root
    blocks = dict(bundle.blocks)
    metadata = {
        "tree_id": root.producer.repository_snapshot_cid,
        "ast_cid": state.state_cid,
        "state_root": root.root_cid,
        "content_hash": "sha256:" + hashlib.sha256(blocks[root.root_cid]).hexdigest(),
    }
    if corruption == "scope":
        metadata["tree_id"] = root.root_cid
    elif corruption == "state":
        metadata["ast_cid"] = root.root_cid
    elif corruption == "hash":
        metadata["content_hash"] = "sha256:" + "0" * 64
    elif corruption == "missing":
        blocks.pop(root.root_cid)
    else:
        blocks[root.root_cid] = b"corrupt bytes"
    client = scoped(path, state, "writer")
    client.call("record_reference", **metadata)
    consumer = CoordinatedSemanticStateProvider(
        provider, scoped(path, state, "consumer")
    )
    with pytest.raises((DerivedSemanticCoordinationError, SemanticStateAdapterError)):
        next(
            consumer.iter_discovered_views(
                tree_id=metadata["tree_id"], get_block=blocks.__getitem__
            )
        )


def test_wrong_repository_refuses_before_build_or_owner_request(source, monkeypatch):
    provider, state = source
    calls = []
    monkeypatch.setattr(
        provider._api._resolve(),
        "build_semantic_state",
        lambda *a, **k: pytest.fail("wrong repository reached producer"),
    )
    client = DerivedCoordinationClient(
        repository_id="foreign",
        connection=SimpleNamespace(derived_coordination=lambda p: calls.append(p)),
    )
    wrapper = CoordinatedSemanticStateProvider(provider, client)
    with pytest.raises(DerivedSemanticCoordinationError, match="repository"):
        wrapper.build_semantic_state(state)
    assert calls == []


def test_semantic_verification_error_is_not_downgraded_to_unknown_cache(
    source, monkeypatch
):
    provider, state = source
    calls = []
    monkeypatch.setattr(
        provider._api._resolve(),
        "verify_semantic_state_bundle",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid producer bytes")),
    )
    wrapper = CoordinatedSemanticStateProvider(
        provider,
        DerivedCoordinationClient(
            repository_id=state.repository_id,
            connection=SimpleNamespace(derived_coordination=lambda p: calls.append(p)),
        ),
    )
    with pytest.raises(ValueError, match="invalid producer bytes"):
        wrapper.build_semantic_state(state)
    assert calls == [] and wrapper.last_coordination["status"] == "not_attempted"


def test_coordination_only_size_refusal_preserves_valid_producer_result(
    source, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.semantic_state import (
        derived_coordination as coordination,
    )

    provider, state = source
    bundle = provider.build_semantic_state(state)
    assert (
        provider.verify_semantic_state_bundle(bundle).root_cid == bundle.root.root_cid
    )
    # A valid producer result need not fit a narrower discovery-reference profile.
    monkeypatch.setattr(
        coordination,
        "MAX_ROOT_BLOCK_BYTES",
        len(bundle.get_block(bundle.root.root_cid)) - 1,
    )
    monkeypatch.setattr(
        provider._api._resolve(), "build_semantic_state", lambda *a, **k: bundle
    )
    client = DerivedCoordinationClient(
        repository_id=state.repository_id,
        connection=SimpleNamespace(
            derived_coordination=lambda _: pytest.fail(
                "size refusal made a registry request"
            )
        ),
    )
    wrapper = CoordinatedSemanticStateProvider(provider, client)
    assert wrapper.build_semantic_state(state) is bundle
    assert wrapper.last_coordination["status"] == "not_published"
    assert wrapper.last_coordination["source_result_preserved"] is True


@pytest.mark.parametrize(
    "reply", [{"completion_authority": True}, {"result": {"reference": "secret"}}, None]
)
def test_malformed_registry_response_is_unknown_and_never_replays(source, reply):
    provider, state = source
    calls = []

    def request(payload):
        calls.append(payload)
        return reply

    wrapper = CoordinatedSemanticStateProvider(
        provider,
        DerivedCoordinationClient(
            repository_id=state.repository_id,
            connection=SimpleNamespace(derived_coordination=request),
        ),
    )
    bundle = wrapper.build_semantic_state(state)
    assert (
        provider.verify_semantic_state_bundle(bundle).root_cid == bundle.root.root_cid
    )
    assert len(calls) == 1 and wrapper.last_coordination["status"] == "unknown"
    assert "secret" not in str(wrapper.last_coordination)
