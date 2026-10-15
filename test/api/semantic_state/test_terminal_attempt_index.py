"""Real persistent Kit/DuckDB attempt discovery; no provider or prover calls.

Owner rows are an authored fixture, not production launch qualification.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import (
    ATTEMPT_SCHEMA,
    PREFLIGHT_SCHEMA,
    RESOLVE_SCHEMA,
    TerminalAttemptIndex,
    TerminalAttemptIndexCapacityError,
    TerminalAttemptIndexConflictError,
    TerminalAttemptIndexError,
    TerminalAttemptIndexIntegrityError,
    TerminalAttemptIndexPendingError,
    _bounded_object,
)
from ipfs_kit_py.mcp_server.mcplusplus import duckdb_coordination_storage as native_module
from ipfs_kit_py.mcp_server.mcplusplus.duckdb_coordination_storage import DuckDBCoordinationStore

NS = "terminal-attempt-index/test-campaign/shard-01"
OWNER = dict(
    store_id="store1",
    server_id="server1",
    database_uuid="database1",
    process_birth_id="birth1",
    generation=1,
    fence_epoch=2,
    schema_revision=3,
)


def install_owner(connection):
    connection.execute("""CREATE TABLE store_generations (
        generation BIGINT PRIMARY KEY, schema_revision BIGINT NOT NULL,
        fence_epoch BIGINT NOT NULL, revision BIGINT NOT NULL,
        database_uuid VARCHAR NOT NULL, birth_id VARCHAR NOT NULL)""")
    connection.execute("""CREATE TABLE state_servers (
        server_id VARCHAR PRIMARY KEY, store_id VARCHAR NOT NULL,
        database_uuid VARCHAR NOT NULL, process_birth_id VARCHAR NOT NULL,
        generation BIGINT NOT NULL, schema_revision BIGINT NOT NULL,
        status VARCHAR NOT NULL, stopped_at VARCHAR)""")
    connection.execute("INSERT INTO store_generations VALUES (1,3,2,1,'database1','birth1')")
    connection.execute(
        "INSERT INTO state_servers VALUES ('server1','store1','database1','birth1',1,3,'ready',NULL)"
    )


def index_on(connection, *, namespace=NS, lock=None, **limits):
    store = DuckDBCoordinationStore(
        connection,
        transaction_lock=lock or threading.RLock(),
        owner_identity=OWNER,
        namespace=namespace,
    )
    return TerminalAttemptIndex(store, namespace=namespace, **limits)


@pytest.fixture
def native(tmp_path):
    database = tmp_path / "control.duckdb"
    connection = duckdb.connect(str(database))
    install_owner(connection)
    index = index_on(connection)
    yield database, connection, index
    connection.close()


def record(identity="attempt:1", context="context:1", value="exact result"):
    return {
        "schema": ATTEMPT_SCHEMA,
        "attempt_identity": identity,
        "terminal": True,
        "replay_context_cid": context,
        "outcome": {
            "attempt_identity": identity,
            "result": {"value": value},
            "simulated": False,
            "root_conflict": False,
            "human_review_required": False,
            "historical_replay": False,
        },
    }


def reserve_publish(index, identity="attempt:1", context="context:1"):
    ticket = index.reserve(identity, replay_context_cid=context)
    receipt = index.publish(identity, record(identity, context), reservation=ticket)
    return ticket, receipt


def test_actual_persistent_close_reopen_discovers_envelope_and_preserves_pending(native):
    database, connection, index = native
    ticket, receipt = reserve_publish(index)
    index.reserve("attempt:pending", replay_context_cid="context:pending")
    before = index.store.current_state_root(NS)
    connection.close()
    reopened = duckdb.connect(str(database))
    try:
        fresh = index_on(reopened)
        assert fresh.lookup("attempt:1") == record()
        assert (
            fresh.preflight("attempt:1", replay_context_cid="context:1")["envelope_cid"]
            == receipt["envelope_cid"]
        )
        with pytest.raises(TerminalAttemptIndexPendingError, match="recovery"):
            fresh.lookup("attempt:pending")
        assert fresh.publish("attempt:1", record(), reservation=ticket)["status"] == "unchanged"
        assert fresh.store.current_state_root(NS) == before
    finally:
        reopened.close()


def test_interleaved_reservations_publish_without_rebinding_whole_shard_ticket(native):
    _, _, index = native
    first = index.reserve("attempt:1", replay_context_cid="context:1")
    second = index.reserve("attempt:2", replay_context_cid="context:2")
    assert first["index_cid"] != second["index_cid"]
    index.publish("attempt:2", record("attempt:2", "context:2"), reservation=second)
    index.publish("attempt:1", record(), reservation=first)
    assert index.lookup("attempt:1") == record()
    assert index.lookup("attempt:2") == record("attempt:2", "context:2")
    assert index.store.current_state_root(NS)["revision"] == 4


def test_reserve_publish_never_activates_secondary_metadata_owner(native, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
        SupervisorMetaIndex,
    )

    _, _, index = native
    attempts = []

    def observe(cls):
        attempts.append(cls)
        return None

    monkeypatch.setattr(SupervisorMetaIndex, "from_env", classmethod(observe))
    _, receipt = reserve_publish(index)
    assert receipt["durable"] is True
    assert index.lookup("attempt:1") == record()
    assert index.store.current_state_root(NS)["revision"] == 2
    assert attempts == []


def test_resolve_empty_and_terminal_closed_receipts_preserve_preflight(native):
    _, _, index = native
    available = index.resolve("attempt:1", replay_context_cid="context:1")
    assert set(available) == {"schema", "receipt", "envelope"}
    assert available["schema"] == RESOLVE_SCHEMA and available["envelope"] is None
    assert available["receipt"] == index.preflight("attempt:1", replay_context_cid="context:1")
    assert available["receipt"]["schema"] == PREFLIGHT_SCHEMA
    assert available["receipt"]["replay_context_cid"] is None
    _, publication = reserve_publish(index)
    terminal = index.resolve("attempt:1", replay_context_cid="context:1")
    assert terminal["envelope"] == record()
    assert terminal["receipt"] == index.preflight("attempt:1", replay_context_cid="context:1")
    assert terminal["receipt"]["envelope_cid"] == publication["envelope_cid"]
    assert terminal["receipt"]["completion_authority"] is False


def test_resolve_one_snapshot_authenticates_every_child_without_target_reread(native, monkeypatch):
    _, _, index = native
    first_ticket, first = reserve_publish(index)
    second_ticket, second = reserve_publish(index, "attempt:2", "context:2")
    pending = index.reserve("attempt:pending", replay_context_cid="context:pending")
    root = index.store.current_state_root(NS)
    calls, reads, envelopes = [], [], []
    original_call, original_read, original_envelope = index._call, index._read, index._envelope
    original_native_read = index.store._read_bytes
    native_locked_reads = []

    def call(method, *args, **kwargs):
        calls.append(method)
        return original_call(method, *args, **kwargs)

    def read(cid, limit):
        reads.append(cid)
        return original_read(cid, limit)

    def native_read(cid):
        assert index.store._lock._is_owned()
        native_locked_reads.append(cid)
        return original_native_read(cid)

    def envelope(identity, body, context):
        envelopes.append(identity)
        return original_envelope(identity, body, context)

    monkeypatch.setattr(index, "_call", call)
    monkeypatch.setattr(index, "_read", read)
    monkeypatch.setattr(index, "_envelope", envelope)
    monkeypatch.setattr(index.store, "_read_bytes", native_read)
    result = index.resolve("attempt:1", replay_context_cid="context:1")
    assert result["envelope"] == record()
    assert calls.count("current_state_root") == 1
    expected = [
        root["root_cid"],
        first_ticket["pending_cid"],
        first["envelope_cid"],
        second_ticket["pending_cid"],
        second["envelope_cid"],
        pending["pending_cid"],
    ]
    assert sorted(reads) == sorted(expected)
    assert envelopes == ["attempt:1", "attempt:2"]
    assert result["receipt"]["index_cid"] == root["root_cid"]
    assert result["receipt"]["revision"] == root["revision"]
    assert native_locked_reads


@pytest.mark.parametrize("state", ["pending", "terminal"])
def test_resolve_changed_context_and_pending_keep_current_errors(native, state):
    _, _, index = native
    if state == "terminal":
        reserve_publish(index)
    else:
        index.reserve("attempt:1", replay_context_cid="context:1")
        with pytest.raises(TerminalAttemptIndexPendingError):
            index.resolve("attempt:1", replay_context_cid="context:1")
    with pytest.raises(TerminalAttemptIndexConflictError, match="new attempt"):
        index.resolve("attempt:1", replay_context_cid="changed")


def test_available_resolution_is_not_a_reservation_and_latest_capacity_wins(native):
    _, connection, _ = native
    index = index_on(connection, namespace=NS + "/resolve-small", max_entries=1)
    assert (
        index.resolve("attempt:1", replay_context_cid="context:1")["receipt"]["status"]
        == "available"
    )
    index.reserve("attempt:2", replay_context_cid="context:2")
    for operation in (
        lambda: index.resolve("attempt:1", replay_context_cid="context:1"),
        lambda: index.reserve("attempt:1", replay_context_cid="context:1"),
    ):
        with pytest.raises(TerminalAttemptIndexCapacityError):
            operation()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_resolve_does_not_skip_an_unselected_terminal_child(native, damage):
    _, connection, index = native
    reserve_publish(index)
    _, other = reserve_publish(index, "attempt:2", "context:2")
    if damage == "missing":
        connection.execute(
            "DELETE FROM kit_coordination_blocks WHERE namespace=? AND cid=?",
            [NS, other["envelope_cid"]],
        )
    else:
        connection.execute(
            "UPDATE kit_coordination_blocks SET block_bytes=? WHERE namespace=? AND cid=?",
            [b"{}", NS, other["envelope_cid"]],
        )
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.resolve("attempt:1", replay_context_cid="context:1")


def test_duplicate_target_capture_remains_one_body_and_manifest_still_fails(native):
    _, _, index = native
    reserve_publish(index)
    _, manifest = index._snapshot()
    manifest["entries"] *= 2
    manifest["revision"] = 4
    captured = []
    with pytest.raises(TerminalAttemptIndexIntegrityError, match="order"):
        index._validate_manifest(manifest, 4, capture=("attempt:1", captured))
    assert captured == [record()]


def test_resolve_source_and_owner_fences_remain_active(native):
    _, connection, index = native
    reserve_publish(index)
    original = index._source_sha256
    index._source_sha256 = "0" * 64
    with pytest.raises(TerminalAttemptIndexError, match="source changed"):
        index.resolve("attempt:1", replay_context_cid="context:1")
    index._source_sha256 = original
    connection.execute("UPDATE store_generations SET fence_epoch=99")
    with pytest.raises(TerminalAttemptIndexError):
        index.resolve("attempt:1", replay_context_cid="context:1")


def test_duplicate_publication_no_cas_and_different_terminal_cannot_overwrite(native):
    _, _, index = native
    ticket, receipt = reserve_publish(index)
    before = index.store.current_state_root(NS)
    duplicate = index.publish("attempt:1", record(), ticket=ticket)
    assert duplicate["status"] == "unchanged" and duplicate["idempotent_replay"] is True
    assert duplicate["envelope_cid"] == receipt["envelope_cid"]
    with pytest.raises(TerminalAttemptIndexConflictError, match="rebound"):
        index.publish("attempt:1", record(value="different"), reservation=ticket)
    assert index.store.current_state_root(NS) == before


def test_pending_same_context_never_returns_another_dispatchable_ticket(native):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    for operation in (
        lambda: index.reserve("attempt:1", replay_context_cid="context:1"),
        lambda: index.preflight("attempt:1", replay_context_cid="context:1"),
        lambda: index.lookup("attempt:1"),
    ):
        with pytest.raises(TerminalAttemptIndexPendingError) as exc:
            operation()
        assert exc.value.pending_cid == ticket["pending_cid"]
        assert exc.value.replay_context_cid == "context:1"
    with pytest.raises(TerminalAttemptIndexConflictError):
        index.preflight("attempt:1", replay_context_cid="changed")


def test_forged_nonce_and_origin_manifest_refuse_publication(native):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    wrong = dict(ticket, nonce="0" * 32)
    with pytest.raises(TerminalAttemptIndexConflictError, match="another caller"):
        index.publish("attempt:1", record(), reservation=wrong)
    wrong = dict(ticket, revision=ticket["revision"] + 1)
    with pytest.raises(TerminalAttemptIndexIntegrityError, match="manifest"):
        index.publish("attempt:1", record(), reservation=wrong)
    with pytest.raises(TerminalAttemptIndexIntegrityError, match="ticket"):
        index.publish("attempt:1", record())
    assert index.store.current_state_root(NS)["revision"] == 1


@pytest.mark.parametrize("change", ["policy", "extra", "context", "duplicate", "transition"])
def test_origin_manifest_and_transition_are_fully_authenticated(native, change):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    origin = index.store.get(ticket["index_cid"])
    if change == "policy":
        origin["max_entries"] += 1
    elif change == "extra":
        origin["unrecognized"] = True
    elif change == "context":
        origin["entries"][0]["replay_context_cid"] = "wrong context"
    elif change == "duplicate":
        origin["entries"] *= 2
    else:
        transition = index.store.get(ticket["transition_cid"])
        transition["new_root_cid"] = ticket["pending_cid"]
        ticket["transition_cid"] = index.store.put(transition)["cid"]
    if change != "transition":
        ticket["index_cid"] = index.store.put(origin)["cid"]
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.publish("attempt:1", record(), reservation=ticket)
    assert index.store.current_state_root(NS)["revision"] == 1


def test_unicode_terminal_uses_actual_native_raw_bytes_and_roundtrips(native):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    body = record(value="café, λ, 中文, 👩")
    actual = index.record_cid(body)
    receipt = index.publish("attempt:1", body, reservation=ticket)
    assert receipt["envelope_cid"] == actual
    assert native_module.validate_transport_cid(actual) == "raw"
    assert index.store.get_bytes(actual) == native_module._canonical_json(body)
    assert index.lookup("attempt:1") == body
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload

    assert actual != cid_for_payload(body)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_cannot_be_a_terminal_record(native, value):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.publish("attempt:1", record(value=value), reservation=ticket)
    assert index.store.current_state_root(NS)["revision"] == 1


@pytest.mark.parametrize("kind", ["strings", "keys", "aliases", "controls", "unicode", "integer"])
def test_aggregate_oversize_is_refused_before_canonicalizer(kind):
    shared = {"payload": "x" * 200}
    values = {
        "strings": {"items": ["x" * 127] * 20},
        "keys": {f"key{number}": "s" * 40 for number in range(50)},
        "aliases": {"items": [shared] * 20},
        "controls": {"items": ["\x00" * 100]},
        "unicode": {"items": ["👩" * 100] * 2},
        "integer": {"value": 1 << 10_000},
    }
    called = False

    def forbidden(_value):
        nonlocal called
        called = True
        raise AssertionError("oversize input must not reach canonicalization")

    with pytest.raises(TerminalAttemptIndexCapacityError, match="aggregate"):
        _bounded_object(values[kind], 512, canonicalize=forbidden)
    assert called is False


@pytest.mark.parametrize("kind", ["strings", "aliases", "controls", "unicode", "numbers"])
def test_native_json_at_exact_byte_limit_is_not_rejected(kind):
    shared = {"message": "identical string"}
    values = {
        "strings": {"items": ["repeated"] * 5},
        "aliases": {"items": [shared] * 5},
        "controls": {"key\n": '"\\' + "".join(chr(code) for code in range(32))},
        "unicode": {"中文": ["café λ 👩", "\u2028\u2029"]},
        "numbers": {"items": [0, -1, 10**40, -(10**40), 1.25, -0.0, True, False, None]},
    }
    value = values[kind]
    actual = native_module._canonical_json(value)
    calls = []

    def canonicalizer(body):
        calls.append(body)
        return native_module._canonical_json(body)

    copied, raw = _bounded_object(value, len(actual), canonicalize=canonicalizer)
    assert copied == value and raw == actual
    assert len(calls) == 1


def test_large_collection_fanout_never_reaches_serialization():
    def forbidden(_value):
        raise AssertionError("unbounded collection must not reach canonicalization")

    with pytest.raises(TerminalAttemptIndexCapacityError, match="structure"):
        _bounded_object({"items": [None] * 50_001}, 262_144, canonicalize=forbidden)


def test_full_campaign_refuses_before_effect_and_does_not_churn(native):
    _, connection, _ = native
    index = index_on(connection, namespace=NS + "/small", max_entries=1)
    reserve_publish(index)
    before = index.store.current_state_root(index.namespace)
    for operation in (
        lambda: index.preflight("attempt:2"),
        lambda: index.reserve("attempt:2", replay_context_cid="context:2"),
    ):
        with pytest.raises(TerminalAttemptIndexCapacityError):
            operation()
    assert index.store.current_state_root(index.namespace) == before
    assert index.preflight("attempt:1")["status"] == "terminal"


def test_small_manifest_refuses_reservation_before_any_index_publication(native):
    _, connection, _ = native
    index = index_on(connection, namespace=NS + "/bytes", max_manifest_bytes=64)
    with pytest.raises(TerminalAttemptIndexCapacityError):
        index.reserve("attempt:1", replay_context_cid="context:1")
    assert index.store.current_state_root(index.namespace)["revision"] == 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("terminal", 1),
        ("simulated", 0),
        ("root_conflict", "false"),
        ("human_review_required", None),
        ("historical_replay", True),
    ],
)
def test_envelope_exact_boolean_validation(native, flag, value):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    body = record()
    (body if flag == "terminal" else body["outcome"])[flag] = value
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.publish("attempt:1", body, reservation=ticket)
    assert index.store.current_state_root(NS)["revision"] == 1


@pytest.mark.parametrize(
    "mutation", ["outer_identity", "outcome_identity", "context", "outer_extra", "result_scalar"]
)
def test_full_envelope_binding_and_shape_are_authenticated(native, mutation):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    body = record()
    if mutation == "outer_identity":
        body["attempt_identity"] = "other"
    elif mutation == "outcome_identity":
        body["outcome"]["attempt_identity"] = "other"
    elif mutation == "context":
        body["replay_context_cid"] = "other"
    elif mutation == "outer_extra":
        body["authority"] = True
    else:
        body["outcome"]["result"] = "success"
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.publish("attempt:1", body, reservation=ticket)


@pytest.mark.parametrize(
    "target,mutation",
    [
        ("index", "missing"),
        ("envelope", "missing"),
        ("pending", "missing"),
        ("envelope", "corrupt"),
    ],
)
def test_missing_or_corrupt_indexed_blocks_are_not_cache_misses(native, target, mutation):
    _, connection, index = native
    ticket, receipt = reserve_publish(index)
    cid = {
        "index": receipt["index_cid"],
        "envelope": receipt["envelope_cid"],
        "pending": ticket["pending_cid"],
    }[target]
    if mutation == "missing":
        connection.execute(
            "DELETE FROM kit_coordination_blocks WHERE namespace=? AND cid=?", [NS, cid]
        )
    else:
        connection.execute(
            "UPDATE kit_coordination_blocks SET block_bytes=? WHERE namespace=? AND cid=?",
            [b"{}", NS, cid],
        )
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.lookup("attempt:1")
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        index.preflight("new")


@pytest.mark.parametrize(
    "field,value",
    [("fence_epoch", 99), ("generation", 99), ("schema_revision", 99), ("birth_id", "other")],
)
def test_changed_native_owner_fences_index_before_dispatch(native, field, value):
    _, connection, index = native
    connection.execute(f"UPDATE store_generations SET {field}=?", [value])
    with pytest.raises(TerminalAttemptIndexError, match="unavailable"):
        index.preflight("attempt:1")


def test_two_callers_same_identity_have_one_reservation_winner(native):
    _, connection, _ = native
    lock = threading.RLock()
    one = index_on(connection, namespace=NS + "/race", lock=lock)
    two = index_on(connection, namespace=NS + "/race", lock=lock)

    def reserve(index):
        try:
            return index.reserve("attempt:1", replay_context_cid="context:1")
        except (TerminalAttemptIndexPendingError, TerminalAttemptIndexConflictError) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(reserve, [one, two]))
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert one.store.current_state_root(one.namespace)["revision"] == 1


def test_publication_failure_exposes_stored_envelope_and_retains_pending(native, monkeypatch):
    _, _, index = native
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")

    def fail(*args, **kwargs):
        raise RuntimeError("authored publication interruption")

    monkeypatch.setattr(index.store, "compare_and_swap_state_root", fail)
    with pytest.raises(TerminalAttemptIndexError) as exc:
        index.publish("attempt:1", record(), reservation=ticket)
    assert exc.value.envelope_cid is not None
    assert index.store.get(exc.value.envelope_cid) == record()
    with pytest.raises(TerminalAttemptIndexPendingError):
        index.lookup("attempt:1")


def test_record_bound_refuses_oversize_before_terminal_write(native):
    _, connection, _ = native
    index = index_on(connection, namespace=NS + "/record", max_record_bytes=1024)
    ticket = index.reserve("attempt:1", replay_context_cid="context:1")
    with pytest.raises(TerminalAttemptIndexCapacityError):
        index.publish("attempt:1", record(value="x" * 2048), reservation=ticket)
    assert index.store.current_state_root(index.namespace)["revision"] == 1


def test_dynamic_native_kit_module_alias_is_accepted(native):
    _, connection, _ = native
    name = native_module.__package__ + ".authored_alias_duckdb_coordination_storage"
    specification = importlib.util.spec_from_file_location(name, native_module.__file__)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
        store = module.DuckDBCoordinationStore(
            connection,
            transaction_lock=threading.RLock(),
            owner_identity=OWNER,
            namespace=NS + "/alias",
        )
        index = TerminalAttemptIndex(store, namespace=store.namespace)
        reserve_publish(index)
        assert index.lookup("attempt:1") == record()
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "namespace", ["source-forest/example", "proof/example", "terminal-attempt-index/"]
)
def test_no_retargeting_into_other_authority_namespaces(native, namespace):
    _, _, index = native
    with pytest.raises(TerminalAttemptIndexIntegrityError):
        TerminalAttemptIndex(index.store, namespace=namespace)


@pytest.mark.parametrize("lock_kind", ["rlock", "lock"])
def test_process_reopen_reads_real_terminal_without_writing_or_dispatch(native, lock_kind):
    database, connection, index = native
    _, receipt = reserve_publish(index)
    connection.close()
    script = """
import importlib.util,json,sys,threading
from pathlib import Path
A=Path(sys.argv[1]); P=Path(sys.argv[2]); K=Path(sys.argv[3])
sys.path[:0]=[str(P),str(K),str(A)]
for name,path in [('ipfs_accelerate_py',A/'ipfs_accelerate_py'),('ipfs_kit_py',K/'ipfs_kit_py')]:
 spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
import duckdb
from ipfs_kit_py.mcp_server.mcplusplus.duckdb_coordination_storage import DuckDBCoordinationStore
from ipfs_accelerate_py.agent_supervisor.semantic_state.terminal_attempt_index import TerminalAttemptIndex
c=duckdb.connect(sys.argv[4]);owner=json.loads(sys.argv[5]);ns=sys.argv[6]
lock=threading.Lock() if sys.argv[7]=='lock' else threading.RLock()
s=DuckDBCoordinationStore(c,transaction_lock=lock,owner_identity=owner,namespace=ns)
i=TerminalAttemptIndex(s,namespace=ns)
resolved=i.resolve('attempt:1',replay_context_cid='context:1')
result=resolved['receipt']
assert resolved['envelope']['outcome']['result']['value']=='exact result'
assert result==i.preflight('attempt:1',replay_context_cid='context:1')
assert i.lookup('attempt:1')['outcome']['result']['value']=='exact result'
print(json.dumps(result,sort_keys=True));c.close()
"""
    accelerate = Path(__file__).resolve().parents[3]
    external = accelerate.parent
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(accelerate),
            str(external / "ipfs_datasets"),
            str(external / "ipfs_kit"),
            str(database),
            json.dumps(OWNER),
            NS,
            lock_kind,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    observation = json.loads(completed.stdout)
    assert observation["envelope_cid"] == receipt["envelope_cid"]
    assert observation["revision"] == 2
