"""Public preparation/dispatch with actual producer, tokenizer and native storage.

Persistent DuckDB bytes, Kit owner fences, source/capsule validation and local
tokenization execute normally. READY rows, launch admission, model compatibility
and provider responses are authored fixtures. A simulated dispatch is rejected;
these tests establish no production launch, model, Lean or semantic acceptance.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    _scan_scoped_sources,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessDisposition,
    HarnessMode,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
    DurableStateError,
    IpfsKitNativeDurableStateAdapter,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessLoopError,
    HarnessPolicy,
    HarnessRootConflict,
    SemanticCompressionHarness,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.owner_context_materializer import (
    MaterializerModelBinding,
    OwnerContextMaterializer,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import (
    TerminalAttemptIndex,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_kit_py.mcp_server.mcplusplus import duckdb_coordination_storage as native_module
from test.api.semantic_state.test_harness import (
    _accept_request,
    _bootstrap,
    _cid,
    _pack,
    _provider,
    _routing,
    _simple_patch,
)
from test.api.semantic_state.test_harness_native_resume import restart_owner
from test.api.semantic_state.test_owner_context_materializer import (
    IMPLEMENTATION_REVISION,
    MODEL_REVISION,
    PRODUCER_REVISION,
    SOURCE_NAMES,
    CurrentProducer,
    FixtureAuthority,
    current,  # noqa: F401 -- ordinary imported pytest fixture; no package repinning.
    decoded_capsule,
)
from test.api.semantic_state.test_terminal_attempt_index import OWNER, install_owner

REPOSITORY = "repo-test"
INDEX_NAMESPACE = "terminal-attempt-index/native-materializer-fixture/shard-0"
TABLES = (
    "store_generations", "state_servers", "kit_coordination_namespaces",
    "kit_coordination_blocks", "kit_coordination_roots", "kit_coordination_transitions",
)


def bind_native(connection, *, owner=OWNER):
    """Two existing namespaces on the same owner connection and transaction lock."""
    lock = threading.RLock()
    semantic_store = native_module.DuckDBCoordinationStore(
        connection, transaction_lock=lock, owner_identity=owner, namespace=REPOSITORY,
    )
    attempt_store = native_module.DuckDBCoordinationStore(
        connection, transaction_lock=lock, owner_identity=owner, namespace=INDEX_NAMESPACE,
    )
    durable = IpfsKitNativeDurableStateAdapter(
        semantic_store, native_module, repository_id=REPOSITORY,
    )
    return durable, TerminalAttemptIndex(attempt_store, namespace=INDEX_NAMESPACE)


def producer_for_native_repository(producer: CurrentProducer) -> CurrentProducer:
    """Scan the authored bytes under the native namespace's repository identity.

    Native storage deliberately rejects the colon in the shared fixture's
    ``repo:test`` identifier. The source producer therefore performs a fresh
    canonical scan for ``repo-test``; no state/root/symbol CID is renamed or
    aliased, and the shared fixture is not modified.
    """
    state = _scan_scoped_sources(
        producer.sources, repository_id=REPOSITORY, max_symbols=32,
    )
    bundle = producer.provider.build_semantic_state(state)
    source_symbols, source_cids = {}, {}
    for kind, path in SOURCE_NAMES.items():
        symbol = next(
            item for item in state.symbols
            if item.module_path == path and str(item.kind) == "module"
        )
        source_symbols[kind] = symbol.stable_id
        source_cids[kind] = symbol.source_cid
    helper = next(
        item for item in state.symbols
        if item.module_path == "pkg/support.py" and item.qualified_name.endswith("helper")
    )
    blocks = dict(bundle.blocks)
    view = producer.provider.open_verified_view(bundle.root.root_cid, blocks.__getitem__)
    dependency = view.capsule(helper.stable_id)
    assert view.root.repository_id == REPOSITORY
    assert view.root.producer.repository_state_cid == state.state_cid
    return replace(
        producer, state=state, bundle=bundle, blocks=blocks,
        source_symbols=source_symbols, source_cids=source_cids,
        raw={source_cids[kind]: producer.sources[path] for kind, path in SOURCE_NAMES.items()},
        dependency_symbol=helper.stable_id, dependency_cid=dependency.capsule_cid,
    )


def catalog_snapshot(connection):
    return {
        table: connection.execute(f'SELECT * FROM "{table}" ORDER BY ALL').fetchall()
        for table in TABLES
    }


def fail_effect(*_args, **_kwargs):
    pytest.fail("preparation or historical replay attempted an execution/storage effect")


@dataclass
class NativeCase:
    database: Path
    connection: Any
    current: CurrentProducer
    durable: Any
    index: Any
    provider: Any
    harness: Any
    authority: Any
    model: Any
    port: Any
    root: Any
    request: Any


@pytest.fixture
def native_case(current: CurrentProducer, tmp_path: Path) -> NativeCase:  # noqa: F811
    current = producer_for_native_repository(current)
    database = tmp_path / "native-materializer-owner.duckdb"
    connection = duckdb.connect(str(database))
    try:
        install_owner(connection)
        durable, index = bind_native(connection)
        provider = _provider(patch_text=_simple_patch(), simulated=True)
        policy = HarnessPolicy(mode=HarnessMode.PRODUCTION.value, use_kit_root_cid=True)
        harness = SemanticCompressionHarness(
            durable=durable, terminal_attempt_index=index, providers=(provider,),
            context_materializer=None, policy=policy,
        )
        root = _bootstrap(harness, repository_id=REPOSITORY)
        authority = FixtureAuthority(current)
        model = MaterializerModelBinding(
            provider_id=provider.provider_id, route=ModelRoute.SMALL_LOCAL_MODEL.value,
            mode=policy.mode, model="authored-fixture-model",
            model_revision_cid=MODEL_REVISION, tokenizer=current.tokenizer,
        )
        port = OwnerContextMaterializer(
            provider=current.provider, producer_revision_cid=PRODUCER_REVISION,
            implementation_revision_cid=IMPLEMENTATION_REVISION,
            authority=authority, policy=policy, model_bindings=(model,),
        )
        harness.context_materializer = port
        pack = _pack(
            objective="apply safe patch",
            target_source_cid=current.source_cids["target_source"],
            surrounding_source_cid=current.source_cids["surrounding_source"],
            test_source_cid=current.source_cids["test_source"],
            dependency_capsule_cids=[current.dependency_cid],
            assumptions=["Authored launch/model owner; no production qualification."],
            exclusions=["Only explicitly nominated source and dependency material."],
        )
        request = _accept_request(
            root, repository_id=REPOSITORY,
            routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value), context_pack=pack,
            datasets_state_cid=current.state.state_cid,
            datasets_semantic_state_root_cid=current.bundle.root.root_cid,
        )
        authority.register(request, port, policy, model)
        yield NativeCase(
            database, connection, current, durable, index, provider, harness,
            authority, model, port, root, request,
        )
    finally:
        connection.close()


def refuse_preparation_effects(case, monkeypatch):
    # Native reads/fence checks remain real. None of these effects are needed
    # to validate a complete prompt, and a reservation cannot be a prerequisite.
    for name in ("put", "compare_and_swap_root", "compare_and_swap_root_receipt"):
        monkeypatch.setattr(case.durable, name, fail_effect)
    for name in ("resolve", "preflight", "lookup", "reserve", "publish"):
        monkeypatch.setattr(case.index, name, fail_effect)
    monkeypatch.setattr(type(case.provider), "generate", fail_effect)
    monkeypatch.setattr(SupervisorMetaIndex, "from_env", classmethod(fail_effect))
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.harness.create_isolated_worktree",
        fail_effect,
    )


def assert_no_execution(case, before):
    assert case.provider.calls == []
    assert case.durable.read_root(REPOSITORY) == case.root
    assert catalog_snapshot(case.connection) == before


def test_public_preparation_authenticates_exact_native_material_without_effects(native_case, monkeypatch):
    case = native_case
    view = case.current.provider.open_verified_view(
        case.current.bundle.root.root_cid, case.current.get_block,
    )
    expected_capsule = view.capsule(case.current.dependency_symbol).to_dict()
    before = catalog_snapshot(case.connection)
    refuse_preparation_effects(case, monkeypatch)
    prepared = case.harness.prepare_model_context(case.request)
    assert prepared.provider_id == case.provider.provider_id
    assert prepared.route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert prepared.mode == HarnessMode.PRODUCTION.value
    assert prepared.base_tree == case.request.base_tree
    assert prepared.observed_root == case.root
    assert prepared.context_pack_cid == cid_for_payload(case.request.context_pack.to_dict())
    assert prepared.materializer_revision_cid == case.port.revision_cid
    assert prepared.input_tokens == case.current.tokenizer.count(prepared.prompt) > 0
    assert prepared.input_tokens != case.request.context_pack.token_totals["total"]
    packet = json.loads(prepared.prompt)
    material = packet["materialized_context"]
    for kind, path in SOURCE_NAMES.items():
        assert material["required_sources"][kind] == {
            "source_cid": case.current.source_cids[kind],
            "text": case.current.sources[path].decode("utf-8"),
        }
    transported = material["dependency_capsules"][0]
    assert decoded_capsule(transported) == expected_capsule
    summary = prepared.to_dict()
    assert "prompt" not in summary
    assert summary["prompt_sha256"] == hashlib.sha256(prepared.prompt.encode()).hexdigest()
    assert summary["execution_authority"] is False
    assert summary["semantic_acceptance_authority"] is False
    assert summary["tokenizer_compatibility_authority"] is False
    assert summary["completion_authority"] is False
    assert summary["provider_invoked"] is False
    assert summary["attempt_reserved"] is False
    assert prepared.to_dict(include_prompt=True)["prompt"] == prepared.prompt
    assert case.authority.admit_calls >= 2
    assert_no_execution(case, before)


def test_public_simulated_dispatch_and_reopen_retain_historical_evidence(native_case, monkeypatch):
    case = native_case
    prepared = case.harness.prepare_model_context(case.request)
    first = case.harness.run(case.request)
    assert first.result.disposition == HarnessDisposition.REJECTED.value
    assert {"production_rejected", "simulated_observation"} <= set(first.result.reasons)
    assert first.simulated is True
    assert first.terminal_persistence == "durably_indexed"
    assert case.provider.calls[0]["prompt"] == prepared.prompt
    assert len(case.provider.calls) == 1
    assert case.durable.read_root(REPOSITORY) == case.root
    record = case.index.lookup(first.attempt_identity)
    assert record["terminal"] is True
    assert record["outcome"]["result"]["disposition"] == HarnessDisposition.REJECTED.value
    assert record["outcome"]["simulated"] is True
    authority_counts = (case.authority.admit_calls, case.authority.verify_calls)
    semantic_before = case.durable.store.current_state_root(REPOSITORY)
    attempt_before = case.index.store.current_state_root(INDEX_NAMESPACE)
    case.connection.close()
    reopened = duckdb.connect(str(case.database))
    try:
        owner = restart_owner(reopened)  # Explicit authored successor, not launcher qualification.
        durable, index = bind_native(reopened, owner=owner)
        harness = SemanticCompressionHarness(
            durable=durable, terminal_attempt_index=index, providers=(case.provider,),
            context_materializer=case.port, policy=case.harness.policy,
        )
        case.authority.active = False
        monkeypatch.setattr(case.authority, "admit", fail_effect)
        monkeypatch.setattr(case.authority, "verify_snapshot", fail_effect)
        monkeypatch.setattr(case.current, "get_block", fail_effect)
        monkeypatch.setattr(case.current, "read_source_blob", fail_effect)
        monkeypatch.setattr(type(case.current.tokenizer), "count", fail_effect)
        monkeypatch.setattr(type(case.provider), "generate", fail_effect)
        asset = case.current.artifact.parent / case.current.descriptor["asset_path"]
        asset.unlink()
        case.current.artifact.unlink()
        before = catalog_snapshot(reopened)
        replay = harness.run(case.request)
        assert replay.historical_replay is True
        assert replay.terminal_persistence == "durably_indexed"
        assert replay.terminal_envelope_cid == first.terminal_envelope_cid
        assert replay.result.disposition == first.result.disposition
        assert replay.result.current_root == case.root
        assert "context_freshness_not_revalidated" in replay.result.reasons
        assert len(case.provider.calls) == 1
        assert authority_counts == (case.authority.admit_calls, case.authority.verify_calls)
        assert durable.store.current_state_root(REPOSITORY) == semantic_before
        assert index.store.current_state_root(INDEX_NAMESPACE) == attempt_before
        assert catalog_snapshot(reopened) == before
    finally:
        reopened.close()


@pytest.mark.parametrize("revision", ["implementation", "model", "authority"])
def test_changed_materializer_revision_requires_new_attempt_key(native_case, revision):
    case = native_case
    first = case.harness.run(case.request)
    assert first.terminal_persistence == "durably_indexed"
    implementation = IMPLEMENTATION_REVISION
    model = case.model
    if revision == "implementation":
        implementation = _cid("different-adapter-source-revision")
    elif revision == "model":
        model = replace(model, model_revision_cid=_cid("different-model-source-revision"))
    else:
        case.authority.revision_cid = _cid("different-launch-authority-source-revision")
    port = OwnerContextMaterializer(
        provider=case.current.provider, producer_revision_cid=PRODUCER_REVISION,
        implementation_revision_cid=implementation, authority=case.authority,
        policy=case.harness.policy, model_bindings=(model,),
    )
    assert port.revision_cid != case.port.revision_cid
    case.harness.context_materializer = port
    before = catalog_snapshot(case.connection)
    authority_counts = (case.authority.admit_calls, case.authority.verify_calls)
    second = case.harness.run(case.request)
    assert second.historical_replay is False
    assert {"attempt_context_changed", "new_attempt_key_required"} <= set(second.result.reasons)
    assert len(case.provider.calls) == 1
    assert authority_counts == (case.authority.admit_calls, case.authority.verify_calls)
    assert catalog_snapshot(case.connection) == before


def test_pending_native_reservation_cannot_dispatch_or_become_a_cache_miss(native_case):
    case = native_case
    identity = case.request.attempt_identity(mode=case.harness.policy.mode)
    context = case.harness._replay_context_cid(case.request)
    ticket = case.index.reserve(identity, replay_context_cid=context)
    before = catalog_snapshot(case.connection)
    outcome = case.harness.run(case.request)
    assert outcome.result.disposition == HarnessDisposition.REJECTED.value
    assert "attempt_recovery_required" in outcome.result.reasons
    assert outcome.terminal_reservation_cid == ticket["pending_cid"]
    assert case.authority.admit_calls == 0
    assert_no_execution(case, before)


def test_fenced_native_owner_refuses_preparation_and_public_dispatch(native_case):
    case = native_case
    restart_owner(case.connection)
    before = catalog_snapshot(case.connection)
    with pytest.raises(DurableStateError, match="owner|generation|fence"):
        case.harness.prepare_model_context(case.request)
    outcome = case.harness.run(case.request)
    assert outcome.result.disposition == HarnessDisposition.UNAVAILABLE.value
    assert "attempt_index_unavailable" in outcome.result.reasons
    assert case.provider.calls == []
    assert case.authority.admit_calls == 0
    assert catalog_snapshot(case.connection) == before


@pytest.mark.parametrize("fault", ["missing", "corrupt", "race"])
def test_native_source_failure_during_preparation_is_read_only(native_case, monkeypatch, fault):
    case = native_case
    cid = case.current.source_cids["target_source"]
    raw = case.current.raw[cid]
    if fault == "missing":
        del case.current.raw[cid]
    elif fault == "corrupt":
        case.current.raw[cid] = raw + b"# corrupt producer blob\n"
    else:
        calls = 0

        def race(requested):
            nonlocal calls
            if requested == cid:
                calls += 1
                return raw if calls == 1 else raw + b"# concurrent mutation\n"
            return case.current.raw[requested]

        case.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=race)
    before = catalog_snapshot(case.connection)
    refuse_preparation_effects(case, monkeypatch)
    with pytest.raises(HarnessLoopError) as refused:
        case.harness.prepare_model_context(case.request)
    assert refused.value.reason_code == "context_materialization_unavailable"
    assert_no_execution(case, before)


@pytest.mark.parametrize("fault", ["missing_tree", "deterministic", "human"])
def test_preparation_requires_explicit_tree_and_provider_route(native_case, monkeypatch, fault):
    case = native_case
    request = case.request
    if fault == "missing_tree":
        request = replace(request, base_tree=None)
    else:
        route = (
            ModelRoute.DETERMINISTIC_ONLY.value if fault == "deterministic"
            else ModelRoute.HUMAN_REVIEW_REQUIRED.value
        )
        request = replace(request, routing_decision=_routing(route))
    before = catalog_snapshot(case.connection)
    refuse_preparation_effects(case, monkeypatch)
    with pytest.raises(HarnessLoopError):
        case.harness.prepare_model_context(request)
    assert case.authority.admit_calls == 0
    assert_no_execution(case, before)


def test_stale_expected_root_refuses_preparation_before_producer_reads(native_case, monkeypatch):
    case = native_case
    request = replace(case.request, expected_root=replace(case.root, generation=case.root.generation + 1))
    before = catalog_snapshot(case.connection)
    refuse_preparation_effects(case, monkeypatch)
    with pytest.raises(HarnessRootConflict):
        case.harness.prepare_model_context(request)
    assert case.authority.admit_calls == 0
    assert_no_execution(case, before)


@pytest.mark.parametrize("fault", ["descriptor", "asset"])
def test_local_tokenizer_drift_during_preparation_cannot_write_or_dispatch(native_case, monkeypatch, fault):
    case = native_case
    path = case.current.artifact
    if fault == "asset":
        path = path.parent / case.current.descriptor["asset_path"]
    path.write_bytes(path.read_bytes() + b" ")
    before = catalog_snapshot(case.connection)
    refuse_preparation_effects(case, monkeypatch)
    with pytest.raises(HarnessLoopError) as refused:
        case.harness.prepare_model_context(case.request)
    assert refused.value.reason_code == "model_context_not_admitted"
    assert_no_execution(case, before)
