"""Actual Quack custody supplies bounded, read-only task context.

The listener, client grant, admitted claim and signed command outcomes are
native. Resource fault injection changes admission only, never owner identity.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_intent_world as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import DuckDBConnectionPolicyError
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation_native import (
    _admit, _claim, _native_snapshot, _owner, _run,
)
from test.api.test_repository_source_index import scheduler


@pytest.fixture
def native_case(scenario, tmp_path, scheduler):
    capability = probe_quack_capabilities()
    assert capability.passes_health_check, (
        "Native bridge qualification requires installed Quack: " + capability.reason_code
    )
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        with RepositorySourceSession(case["repository"], tmp_path / "native-world-source",
                index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
            yield dict(case=case, owner=owner, session=session, scheduler=scheduler,
                       task_id=case["task_cid"], admission=case["admission"])
    assert scheduler.snapshot()["active_lease_count"] == 0


def capture(case, **changes):
    arguments = dict(session=case["session"], server=case["owner"].server,
        admission=case["admission"], task_id=case["task_id"], generation=1)
    arguments.update(changes)
    return bridge.capture_repository_owner_intent_world(**arguments)


def validate(observation, case, **changes):
    arguments = dict(session=case["session"], generation=1, task_id=case["task_id"])
    arguments.update(changes)
    return observation.validate_current(**arguments)


def test_actual_owner_capture_preserves_signed_owner_and_private_custody(native_case, monkeypatch):
    case = native_case
    owner = case["owner"]
    before = _native_snapshot(owner)
    original_admission = deepcopy(case["admission"])
    original_connection = IntentRepository._connection
    reads = []
    lock_depth = []
    original_lock = bridge._native_owner_lock
    original_observe = bridge.observe_owner_intent_requirements

    @contextmanager
    def bounded_lock(*args, **kwargs):
        with original_lock(*args, **kwargs):
            lock_depth.append(True)
            try:
                yield
            finally:
                lock_depth.pop()

    def observe_under_bounded_lock(**kwargs):
        assert lock_depth, "strict reader must enter through bounded owner acquisition"
        assert case["scheduler"].snapshot()["active_lease_count"] == 0
        return original_observe(**kwargs)

    @contextmanager
    def bound_only(intent, *args, **kwargs):
        assert intent.uses_bound_connection, "native capture must not reopen the owner file"
        reads.append(intent)
        with original_connection(intent, *args, **kwargs) as connection:
            yield connection

    monkeypatch.setattr(IntentRepository, "_connection", bound_only)
    monkeypatch.setattr(bridge, "_native_owner_lock", bounded_lock)
    monkeypatch.setattr(bridge, "observe_owner_intent_requirements", observe_under_bounded_lock)
    observation = capture(case)
    receipt = validate(observation, case)
    material = json.loads(observation.reference.summary)
    assert reads and observation.reference.required
    assert observation.reference.kind == "repository-intent-world"
    assert receipt["reference_bytes"] == len(observation.reference.canonical_bytes()) <= 32768
    assert receipt["intent_owner_id"] == case["case"]["intent"].owner_id
    assert receipt["intent_owner_id"] != owner.identity.server_id
    binding = dict(receipt["native_owner_binding"])
    assert binding == {
        key: getattr(owner.identity, key) for key in (
            "server_id", "store_id", "database_uuid", "generation", "process_birth_id",
            "repository_id", "schema_revision", "schema_fingerprint",
        )
    } | {"ready_checked": True}
    assert receipt["native_owner_access"] == "in_process_exclusive_connection"
    assert material["native_owner_binding"] == binding
    assert material["observed_task"]["validations"][0]["status"] == "unobserved"
    assert material["intent_world"]["component_status"]["repository_tree"] == "unavailable"
    assert material["intent_world"]["schedulable"] is False
    assert material["intent_binding"]["repository_id"] != material["source_binding"]["datasets_head"]["repository_id"]
    for key in bridge._AUTHORITY:
        assert material[key] is False and receipt[key] is False
    for encoded in (observation.reference.summary, json.dumps(receipt)):
        for private in (str(case["case"]["profile"]), str(case["case"]["lifecycle"]),
                str(owner.database), str(owner.server.config.state_dir),
                owner.identity.listen_uri, owner.credentials.token,
                owner.credentials.socket_path):
            assert private not in encoded
        assert "local_task_contract" not in encoded
        assert "local_observed_validation" not in encoded
    assert _native_snapshot(owner) == before
    assert case["admission"] == original_admission
    receipt["native_owner_binding"].clear()
    assert observation.receipt()["native_owner_binding"] == binding


def test_latest_actual_signed_check_invalidates_old_packet_with_identical_source(native_case, tmp_path):
    case = native_case
    with _claim(case["owner"], tmp_path) as (_daemon, attempt):
        first_check = _run(case["owner"], attempt, timeout=10)
        assert first_check["passed"] is False
        first = capture(case)
        first_selected = json.loads(first.reference.summary)["observed_task"]["validations"][0]
        second_check = _run(case["owner"], attempt, timeout=10)
        assert second_check["passed"] is False
        with pytest.raises(bridge.RepositoryIntentWorldError, match="observation_changed"):
            validate(first, case)
        second = capture(case)
        material = json.loads(second.reference.summary)
        selected = material["observed_task"]["validations"][0]
        assert selected["status"] == "failed"
        assert selected["evidence_digest"] == second_check["results"][0]["evidence_digest"]
        # Identical signed result content can retain the same digest. Native
        # run/result identities and event order still distinguish observations.
        assert first_selected["evidence_digest"] == first_check["results"][0]["evidence_digest"]
        assert selected["run_id"] != first_selected["run_id"]
        assert selected["result_id"] != first_selected["result_id"]
        assert selected["event_sequence"] > first_selected["event_sequence"]
        assert first.receipt()["source_tree_sha256"] == second.receipt()["source_tree_sha256"]
        assert first.receipt()["native_event_watermark"] < second.receipt()["native_event_watermark"]
        assert selected["run_id"] and selected["result_id"] and selected["event_sequence"]


def test_generic_remote_success_cannot_become_signed_public_evidence(native_case, tmp_path):
    case = native_case
    with _claim(case["owner"], tmp_path) as (_daemon, attempt):
        case["owner"].client.record_task_validation(task_cid=case["task_id"], outcome="passed",
            evidence_digest="sha256:" + hashlib.sha256(b"generic-native-success").hexdigest(),
            argv=["unrelated-public-command"], attempt_id=attempt.attempt_id,
            body={"private_dump": "not-a-signed-public-check"},
            idempotency_key="world-native-generic", command_id="world-native-generic")
        observation = capture(case)
        material = json.loads(observation.reference.summary)
        assert material["observed_task"]["validations"][0]["status"] == "unobserved"
        assert material["requirements"][0]["public_checks_passed"] is False
        assert "not-a-signed-public-check" not in observation.reference.summary


def test_stopped_and_restarted_actual_owner_cannot_revalidate_old_observation(native_case):
    case = native_case
    owner = case["owner"]
    observation = capture(case)
    original = observation.receipt()["native_owner_binding"]
    owner.source.close()
    assert owner.server.stop()["stopped"] is True
    with pytest.raises(bridge.RepositoryIntentWorldError):
        validate(observation, case)
    restarted = owner.server.start()
    assert restarted.server_id != original["server_id"]
    assert restarted.generation > original["generation"]
    with pytest.raises(bridge.RepositoryIntentWorldError):
        validate(observation, case)
    successor = capture(case)
    assert successor.receipt()["native_owner_binding"]["server_id"] == restarted.server_id
    assert successor.receipt()["intent_owner_id"] == observation.receipt()["intent_owner_id"]


@pytest.mark.parametrize("mutation", ["admission", "replacement", "consumer"])
def test_native_issued_observation_cannot_change_binding(native_case, mutation):
    case = native_case
    observation = capture(case)
    if mutation == "admission":
        case["admission"]["graph"]["tasks"][0]["objective"] = "unreviewed objective"
    elif mutation == "replacement":
        observation = replace(observation)
    else:
        with pytest.raises(bridge.RepositoryIntentWorldError, match="consumer_binding"):
            validate(observation, case, task_id="LOCAL-TASK")
        return
    with pytest.raises(bridge.RepositoryIntentWorldError):
        validate(observation, case)


def test_alias_foreign_repository_and_client_handle_are_refused(native_case):
    case = native_case
    with pytest.raises(bridge.RepositoryIntentWorldError, match="exact_native_task"):
        capture(case, task_id="LOCAL-TASK")
    changed = deepcopy(case["admission"])
    changed["manifest"]["payload"]["repository_cid"] = "foreign-repository"
    with pytest.raises(ValueError):
        capture(case, admission=changed)
    for handle in (case["owner"].client, case["case"]["intent"], object()):
        with pytest.raises(bridge.RepositoryIntentWorldError, match="native_intent_owner_required"):
            capture(case, server=handle)


def test_new_signed_result_between_native_reads_refuses_mixed_snapshot(native_case, tmp_path, monkeypatch):
    case = native_case
    original = bridge._resources
    calls = []
    with _claim(case["owner"], tmp_path) as (_daemon, attempt):
        @contextmanager
        def changed(*args, **kwargs):
            calls.append(True)
            # The first strict transaction has released the owner lock. Add a
            # real newer result before reserving the world snapshot's lease.
            if len(calls) == 2:
                assert _run(case["owner"], attempt, timeout=10)["passed"] is False
            with original(*args, **kwargs) as lease:
                yield lease
        monkeypatch.setattr(bridge, "_resources", changed)
        with pytest.raises(bridge.RepositoryIntentWorldError, match="capture_changed"):
            capture(case)
    assert len(calls) == 2
    assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_actual_foreign_owner_lock_contention_has_bounded_wait(native_case):
    server = native_case["owner"].server
    acquired = threading.Event()
    release = threading.Event()
    def hold_owner():
        with server._lock:
            acquired.set()
            release.wait(timeout=5)
    thread = threading.Thread(target=hold_owner)
    thread.start()
    try:
        assert acquired.wait(timeout=2)
        started = time.monotonic()
        with pytest.raises(bridge.RepositoryIntentWorldError, match="capture_timeout"):
            with bridge._native_owner_lock(server, started + 0.05):
                pytest.fail("foreign owner lock was acquired before release")
        assert time.monotonic() - started < 2
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    with bridge._native_owner_lock(server, time.monotonic() + 1):
        assert server.ready()["ready"] is True
    assert native_case["scheduler"].snapshot()["active_lease_count"] == 0


def test_nested_transaction_refusal_preserves_callers_transaction(native_case):
    case = native_case
    server = case["owner"].server
    deadline = time.monotonic() + 10
    custody, binding = bridge._native_owner_identity(server, deadline)
    import duckdb
    with server._lock:
        connection = server._connection
        connection.execute("CREATE TEMP TABLE caller_transaction_probe(value INTEGER)")
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("INSERT INTO caller_transaction_probe VALUES (1)")
            with pytest.raises(DuckDBConnectionPolicyError, match="transaction is already active"):
                bridge._native_world_capture(server=server, owner=custody,
                    repository_id=binding["repository_id"], task_id=case["task_id"], deadline=deadline)
            # A nested BEGIN failure must not roll back the borrowed caller's
            # earlier write. Its owner can still commit it after refusal.
            assert connection.execute("SELECT COUNT(*) FROM caller_transaction_probe").fetchone()[0] == 1
            connection.execute("COMMIT")
            assert connection.execute("SELECT COUNT(*) FROM caller_transaction_probe").fetchone()[0] == 1
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise
        finally:
            connection.execute("DROP TABLE caller_transaction_probe")
    assert validate(capture(case), case)["native_intent_freshness_checked"] is True


def test_native_reader_uses_one_private_admission_copy(native_case, monkeypatch):
    original = bridge.observe_owner_intent_requirements
    caller = native_case["admission"]
    captured = []
    def changed(*, admission, server):
        assert admission is not caller
        captured.append(deepcopy(admission))
        value = original(admission=admission, server=server)
        caller["graph"]["tasks"][0]["objective"] = "unreviewed concurrent objective"
        return value
    monkeypatch.setattr(bridge, "observe_owner_intent_requirements", changed)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="owner_or_admission_changed"):
        capture(native_case)
    assert len(captured) == 2 and captured[0] == captured[1]
    assert captured[0]["graph"]["tasks"][0]["objective"] != "unreviewed concurrent objective"
    assert native_case["scheduler"].snapshot()["active_lease_count"] == 0


@pytest.mark.parametrize("mode", ["denied", "cancelled"])
def test_resource_gate_precedes_native_readiness_and_observation(native_case, monkeypatch, mode):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("native work preceded resource admission")
    monkeypatch.setattr(native_case["owner"].server, "ready", forbidden)
    monkeypatch.setattr(bridge, "observe_owner_intent_requirements", forbidden)
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", forbidden)
    @contextmanager
    def resources(*args, **kwargs):
        if mode == "denied":
            raise RuntimeError("resource reservation denied")
        yield SimpleNamespace(cancelled=True)
    monkeypatch.setattr(bridge, "_resources", resources)
    with pytest.raises((bridge.RepositoryIntentWorldError, RuntimeError)):
        capture(native_case)
    assert not calls
    assert native_case["scheduler"].snapshot()["active_lease_count"] == 0


def test_world_failure_rolls_back_native_transaction_and_releases_owner_and_resource(native_case, monkeypatch):
    case = native_case
    owner = case["owner"]
    before = _native_snapshot(owner)
    with owner.server._lock:
        owner.server._connection.execute("CREATE TEMP TABLE world_capture_rollback_probe(value INTEGER)")
    original = bridge.capture_intent_world_snapshot
    def fail_after_native_capture(intent, **kwargs):
        assert intent.uses_bound_connection and kwargs["transaction_owned_by_caller"] is True
        original(intent, **kwargs)
        owner.server._connection.execute("INSERT INTO world_capture_rollback_probe VALUES (1)")
        raise RuntimeError("injected world capture failure")
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", fail_after_native_capture)
    with pytest.raises(RuntimeError, match="injected world capture failure"):
        capture(case)
    with owner.server._lock:
        assert owner.server._connection.execute("SELECT COUNT(*) FROM world_capture_rollback_probe").fetchone()[0] == 0
        owner.server._connection.execute("BEGIN TRANSACTION")
        owner.server._connection.execute("ROLLBACK")
        owner.server._connection.execute("DROP TABLE world_capture_rollback_probe")
    acquired = []
    def check_lock():
        locked = owner.server._lock.acquire(timeout=2)
        acquired.append(locked)
        if locked:
            owner.server._lock.release()
    thread = threading.Thread(target=check_lock)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive() and acquired == [True]
    assert _native_snapshot(owner) == before
    assert case["scheduler"].snapshot()["active_lease_count"] == 0
    monkeypatch.undo()
    assert validate(capture(case), case)["native_intent_freshness_checked"] is True


def test_cancelled_world_transaction_releases_all_reservations(native_case, monkeypatch):
    active = []
    @contextmanager
    def resources(*args, **kwargs):
        lease = SimpleNamespace(cancelled=False)
        active.append(lease)
        try:
            yield lease
        finally:
            active.remove(lease)
    original = bridge.capture_intent_world_snapshot
    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        active[-1].cancelled = True
        return result
    monkeypatch.setattr(bridge, "_resources", resources)
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", cancelled)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="capture_cancelled"):
        capture(native_case)
    assert active == []
    assert native_case["scheduler"].snapshot()["active_lease_count"] == 0
    monkeypatch.undo()
    assert validate(capture(native_case), native_case)["native_intent_freshness_checked"] is True
