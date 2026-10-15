"""Actual native-owner persistence with authored semantic/provider fixtures.

Fresh persistent DuckDB catalogs, CID checks, root CAS and reopen are real.
Owner admission rows, context/token counts, patches and semantic facts are
fixtures; these tests establish no model, benchmark or Lean qualification.
"""

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessDisposition,
    HarnessError,
    HarnessMode,
    ModelRoute,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableStateError,
    DurableStateIntegrityError,
    DurableStateUnavailable,
    IpfsKitNativeDurableStateAdapter,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessPolicy,
    SemanticCompressionHarness,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.kit_source_forest import _load_kit
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import (
    TerminalAttemptIndex,
    TerminalAttemptIndexError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_kit_py.mcp_server.mcplusplus import duckdb_coordination_storage as native_module
from test.api.semantic_state.test_harness import (
    MemoryContextMaterializer,
    _accept_request,
    _bootstrap,
    _cid,
    _provider,
    _routing,
    _simple_patch,
)
from test.api.semantic_state.test_terminal_attempt_index import OWNER, install_owner

REPOSITORY = "repo-test"
INDEX_NAMESPACE = "terminal-attempt-index/native-harness-fixture/shard-0"


def bind(connection, *, owner=OWNER, module=native_module, lock=None, index_type=TerminalAttemptIndex):
    lock = lock or threading.RLock()
    semantic_store = module.DuckDBCoordinationStore(
        connection, transaction_lock=lock, owner_identity=owner, namespace=REPOSITORY,
    )
    index_store = module.DuckDBCoordinationStore(
        connection, transaction_lock=lock, owner_identity=owner, namespace=INDEX_NAMESPACE,
    )
    durable = IpfsKitNativeDurableStateAdapter(
        semantic_store, module, repository_id=REPOSITORY,
    )
    return durable, index_type(index_store, namespace=INDEX_NAMESPACE)


def harness_on(durable, index, provider):
    return SemanticCompressionHarness(
        durable=durable, terminal_attempt_index=index, providers=(provider,),
        context_materializer=MemoryContextMaterializer(),
        policy=HarnessPolicy(mode=HarnessMode.PRODUCTION.value, use_kit_root_cid=True),
    )


def bootstrap(harness):
    return _bootstrap(harness, repository_id=REPOSITORY)


def request_on(root, **kwargs):
    return _accept_request(root, repository_id=REPOSITORY, **kwargs)


@pytest.fixture
def native(tmp_path):
    path = tmp_path / "owner.duckdb"
    connection = duckdb.connect(str(path))
    install_owner(connection)
    durable, index = bind(connection)
    yield path, connection, durable, index
    connection.close()


def restart_owner(connection):
    """Authored next READY owner; never a claim of actual launcher admission."""
    connection.execute("UPDATE state_servers SET status='stopped', stopped_at='fixture-stop'")
    connection.execute("INSERT INTO store_generations VALUES (2,3,3,2,'database1','birth2')")
    connection.execute(
        "INSERT INTO state_servers VALUES ('server2','store1','database1','birth2',2,3,'ready',NULL)"
    )
    return dict(OWNER, server_id="server2", process_birth_id="birth2", generation=2, fence_epoch=3)


@pytest.mark.parametrize("transport", ["raw", "dag-json"])
@pytest.mark.parametrize("value", ["exact bytes", "caf\u00e9 \u2014 \u5b89\u5168"])
def test_native_json_transport_preserves_exact_cid_and_rejects_other_bytes(native, transport, value):
    _, _, durable, _ = native
    body = {"schema": "native-test@1", "value": value}
    cid = native_module.cid_for_bytes(native_module._canonical_json(body), codec=transport)
    receipt = durable.put(body, expected_cid=cid)
    assert receipt["cid"] == cid and receipt["codec"] == transport
    assert durable.get(cid) == body
    with pytest.raises(DurableStateIntegrityError, match="CID"):
        durable.put(dict(body, value="different bytes"), expected_cid=cid)


def test_native_adapter_preserves_legacy_unicode_identity_and_refuses_nonfinite_values(native):
    _, _, durable, _ = native
    body = {"schema": "legacy-wire-test@1", "value": "caf\u00e9 \u2014 \u5b89\u5168"}
    legacy_cid = cid_for_payload(body)
    native_cid = native_module.cid_for_bytes(native_module._canonical_json(body), codec="raw")
    assert legacy_cid != native_cid
    assert durable.put(body, expected_cid=legacy_cid)["cid"] == legacy_cid
    assert durable.get(legacy_cid) == body
    assert durable.has(legacy_cid)
    assert not durable.has(native_cid)
    with pytest.raises(DurableStateError, match="canonical JSON"):
        durable.put({"schema": "invalid-json@1", "value": float("nan")}, expected_cid=legacy_cid)


def test_native_adapter_with_older_owner_api_refuses_incompatible_bytes_without_alias(native, monkeypatch):
    _, _, durable, _ = native
    body = {"schema": "legacy-wire-test@1", "value": "caf\u00e9"}
    legacy_cid = cid_for_payload(body)
    monkeypatch.setattr(durable.store, "put_wire_json", None)
    with pytest.raises(DurableStateUnavailable, match="canonical wire JSON"):
        durable.put(body, expected_cid=legacy_cid)
    assert not durable.has(legacy_cid)


def test_native_adapter_retains_owner_connection_and_recovery_is_read_only(native, monkeypatch):
    _, connection, durable, index = native
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index

    mirror_calls = []

    def forbidden_mirror(**kwargs):
        mirror_calls.append(kwargs)
        raise AssertionError("native CAS must not open a second metadata writer")

    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record", forbidden_mirror)
    provider = _provider(patch_text=_simple_patch())
    with durable.metadata_scope(), durable:
        root = bootstrap(harness_on(durable, index, provider))
        before = durable.store.current_state_root(REPOSITORY)
        report = durable.recover()
        assert report["current_root"] == root.to_dict()
        assert report["completion_authority"] is False
        assert report["semantic_acceptance_authority"] is False
        assert durable.store.current_state_root(REPOSITORY) == before
    assert connection.execute("SELECT 1").fetchone() == (1,)
    assert mirror_calls == []
    with pytest.raises(DurableStateError, match="different repository namespace"):
        IpfsKitNativeDurableStateAdapter(durable.store, native_module, repository_id="repo:foreign")


def test_native_metadata_scope_keeps_validation_advisory_reads_and_thread_exception_isolation(native, monkeypatch):
    _, _, durable, _ = native
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index

    mirrors = []
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record", lambda **fields: mirrors.append(fields))
    advisory_owner = object()
    monkeypatch.setattr(supervisor_meta_index.SupervisorMetaIndex, "from_env", lambda: advisory_owner)
    valid = _cid("scope-test")
    with pytest.raises(RuntimeError, match="scope error"):
        with durable.metadata_scope():
            assert validate_opaque_cid(valid, "test") == valid
            with pytest.raises(HarnessError):
                validate_opaque_cid("invalid", "test")
            assert supervisor_meta_index._active() is advisory_owner
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(copied.run, validate_opaque_cid, valid, "thread").result() == valid
            assert len(mirrors) == 1  # A copied context cannot suppress another thread.
            with durable.metadata_scope():
                assert validate_opaque_cid(valid, "nested") == valid
            assert len(mirrors) == 1
            raise RuntimeError("scope error")
    assert validate_opaque_cid(valid, "after") == valid
    assert len(mirrors) == 2


def test_accepted_fixture_replays_after_new_owner_without_provider_or_root_cas(native, monkeypatch):
    path, connection, durable, index = native
    provider = _provider(patch_text=_simple_patch())
    first_harness = harness_on(durable, index, provider)
    root = bootstrap(first_harness)
    request = request_on(root, routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value))
    first = first_harness.run(request)
    assert first.result.disposition == HarnessDisposition.ACCEPTED.value
    assert first.result.current_root.generation == 2
    assert first.terminal_persistence == "durably_indexed"
    assert len(provider.calls) == 1
    assert index.lookup(first.attempt_identity)["terminal"] is True
    root_before = durable.store.current_state_root(REPOSITORY)
    index_before = index.store.current_state_root(INDEX_NAMESPACE)
    connection.close()

    reopened = duckdb.connect(str(path))
    try:
        owner = restart_owner(reopened)
        new_durable, new_index = bind(reopened, owner=owner)
        snapshots = []
        snapshot = new_index._snapshot

        def observed_snapshot(*args, **kwargs):
            snapshots.append(True)
            return snapshot(*args, **kwargs)

        monkeypatch.setattr(new_index, "_snapshot", observed_snapshot)
        replay = harness_on(new_durable, new_index, provider).run(request)
        assert replay.historical_replay is True
        assert replay.terminal_persistence == "durably_indexed"
        assert replay.terminal_envelope_cid == first.terminal_envelope_cid
        assert replay.result.current_root == first.result.current_root
        assert "context_freshness_not_revalidated" in replay.result.reasons
        assert len(snapshots) == 1  # One full authenticated index observation.
        assert len(provider.calls) == 1
        assert new_durable.store.current_state_root(REPOSITORY) == root_before
        assert new_index.store.current_state_root(INDEX_NAMESPACE) == index_before
        changed = harness_on(new_durable, new_index, provider).run(
            replace(request, objective="different request under same attempt key")
        )
        assert "attempt_context_changed" in changed.result.reasons
        assert "new_attempt_key_required" in changed.result.reasons
        assert len(provider.calls) == 1
        assert new_index.store.current_state_root(INDEX_NAMESPACE) == index_before
    finally:
        reopened.close()


def test_pending_attempt_reopen_refuses_provider_and_preserves_root(native):
    path, connection, durable, index = native
    provider = _provider(patch_text=_simple_patch())
    harness = harness_on(durable, index, provider)
    root = bootstrap(harness)
    request = request_on(root, routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value))
    identity = request.attempt_identity(mode=harness.policy.mode)
    context = harness._replay_context_cid(request)
    ticket = index.reserve(identity, replay_context_cid=context)
    index_before = index.store.current_state_root(INDEX_NAMESPACE)
    connection.close()
    reopened = duckdb.connect(str(path))
    try:
        new_durable, new_index = bind(reopened, owner=restart_owner(reopened))
        result = harness_on(new_durable, new_index, provider).run(request)
        assert result.result.disposition == HarnessDisposition.REJECTED.value
        assert "attempt_recovery_required" in result.result.reasons
        assert result.terminal_reservation_cid == ticket["pending_cid"]
        assert result.terminal_persistence_error == "attempt_recovery_required"
        assert provider.calls == []
        assert new_durable.read_root(REPOSITORY) == root
        assert new_index.store.current_state_root(INDEX_NAMESPACE) == index_before
    finally:
        reopened.close()


def test_publication_failure_preserves_completed_fixture_root_and_prevents_redispatch(native):
    _, connection, durable, _ = native

    class FailingPublicationIndex(TerminalAttemptIndex):
        def publish(self, identity, record, *, reservation):
            # Persist the actual envelope, then simulate failed discovery CAS.
            cid = self.record_cid(record)
            self.store.put(record, expected_cid=cid, codec="raw", replicate=False)
            raise TerminalAttemptIndexError("injected discovery failure", envelope_cid=cid)

    _, failing = bind(connection, index_type=FailingPublicationIndex)
    provider = _provider(patch_text=_simple_patch())
    harness = harness_on(durable, failing, provider)
    root = bootstrap(harness)
    request = request_on(root, routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value))
    first = harness.run(request)
    assert first.result.disposition == HarnessDisposition.ACCEPTED.value
    assert first.result.current_root.generation == 2
    assert first.accepted_manifest_cid == first.result.current_root.root_cid
    assert first.terminal_persistence == "index_publication_failed"
    assert failing.store.get(first.terminal_envelope_cid)["outcome"]["result"]["current_root"] == first.result.current_root.to_dict()
    current = durable.read_root(REPOSITORY)
    _, fresh_index = bind(connection)
    second = harness_on(durable, fresh_index, provider).run(request)
    assert "attempt_recovery_required" in second.result.reasons
    assert len(provider.calls) == 1
    assert durable.read_root(REPOSITORY) == current
    assert fresh_index.store.current_state_root(INDEX_NAMESPACE)["revision"] == 1


def test_fenced_native_owner_cannot_read_or_publish(native):
    _, connection, durable, index = native
    provider = _provider(patch_text=_simple_patch())
    harness = harness_on(durable, index, provider)
    root = bootstrap(harness)
    restart_owner(connection)
    with pytest.raises(DurableStateError, match="owner|generation|fence"):
        durable.read_root(REPOSITORY)
    # A request with an explicit historical root reaches index preflight, whose
    # owner failure is unavailable rather than a miss permitting dispatch.
    result = harness.run(request_on(root, routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value)))
    assert result.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "attempt_index_unavailable" in result.result.reasons
    assert provider.calls == []


def test_commit_loaded_native_class_and_matching_exceptions(tmp_path):
    """Exact committed local leaf fixture, not the current dirty Kit launch."""
    kit_root = Path(native_module.__file__).resolve().parents[3]
    checkout = tmp_path / "kit"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(kit_root), str(checkout)],
        check=True, capture_output=True, timeout=30,
    )
    leaves = [f"ipfs_kit_py/mcp_server/mcplusplus/{name}.py" for name in (
        "coordination_storage", "duckdb_coordination_storage",
    )]
    subprocess.run(["git", "restore", "--source=HEAD", "--worktree", "--", *leaves],
                   cwd=checkout, check=True, capture_output=True, timeout=30)
    # Put only the new additive SDK leaf into a local fixture commit. The base
    # authority remains the exact committed leaf; no canonical HEAD is changed.
    subprocess.run(["git", "read-tree", "HEAD"], cwd=checkout, check=True,
                   capture_output=True, timeout=5)
    (checkout / leaves[1]).write_bytes((kit_root / leaves[1]).read_bytes())
    subprocess.run(["git", "add", "--", leaves[1]], cwd=checkout, check=True,
                   capture_output=True, timeout=5)
    subprocess.run([
        "git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false",
        "-c", "user.name=Native owner fixture", "-c", "user.email=fixture@invalid",
        "commit", "--allow-empty", "-m", "Local native wire SDK test fixture",
    ], cwd=checkout, check=True, capture_output=True, timeout=10)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
                          capture_output=True, text=True, timeout=5).stdout.strip()
    module = _load_kit(str(tmp_path), {"nested_repositories": [
        {"repository": "ipfs_kit", "path": "kit", "head": head},
    ]})
    assert module.DuckDBCoordinationStore is not native_module.DuckDBCoordinationStore
    connection = duckdb.connect(str(tmp_path / "committed-owner.duckdb"))
    try:
        install_owner(connection)
        durable, index = bind(connection, module=module)
        provider = _provider(patch_text=_simple_patch())
        root = bootstrap(harness_on(durable, index, provider))
        assert durable.read_root(REPOSITORY) == root
        restart_owner(connection)
        with pytest.raises(DurableStateError):
            durable.read_root(REPOSITORY)
    finally:
        connection.close()
