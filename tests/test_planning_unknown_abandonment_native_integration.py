"""Actual copied three-store ADMIN settlement with a nonadmitting custody seam.

All canonical/coordination/execution handles, claims, control CASes, ordinals,
writer locks, audit transactions, WAL replay, AF_UNIX grants and closeout stages
are genuine native implementations. Only verify_custody is replaced, explicitly
because this disposable fixture never ran the production orchestration owner.
No launch/terminal/admission receipt is fabricated. These tests do not qualify
production launch custody; a parent-owned authentic finite-launch gate remains.

Fixture UNKNOWN rows model retained native data; no provider/model is invoked.
The exact installed-DuckDB crash-copy helper is a finite admitted child of the
outer test owner, stays in its process group, and exits after real WAL COMMIT.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from types import ModuleType, SimpleNamespace
import uuid

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator, TASK_CLAIM_FAILURE_REARM_OPERATION,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state, typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository as intent_module
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    native_execution_status_owner as status,
    native_execution_wal_maintenance as maintenance,
    planning_unknown_abandonment as policy,
    planning_unknown_abandonment_settlement as settlement,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


HELPER_PATH = Path(__file__).with_name("native_three_store_crash_copy_fixture.py").absolute()
crash_copy = ModuleType("_explicit_native_crash_copy_fixture")
crash_copy.__file__ = str(HELPER_PATH)
exec(compile(HELPER_PATH.read_bytes(), str(HELPER_PATH), "exec", dont_inherit=True), crash_copy.__dict__)
ADMIN_HELPER_PATH = Path(__file__).with_name("native_unknown_admin_owner_fixture.py").absolute()
admin_owner_fixture = ModuleType("_explicit_native_admin_owner_fixture")
admin_owner_fixture.__file__ = str(ADMIN_HELPER_PATH)
exec(compile(ADMIN_HELPER_PATH.read_bytes(), str(ADMIN_HELPER_PATH), "exec", dont_inherit=True),
     admin_owner_fixture.__dict__)
TASK = "fixture-native-unknown-task"
NAMESPACE = "TEST-ONLY-nonadmitting-native-three-store"
LANE_BASENAME = "crash-copy.execution.shard-0001-of-0003.duckdb"
PROVIDER_COLUMNS = ("invocation_id", "attempt_id", "task_cid", "idempotency_key",
                    "owner_session_id", "recorded_at_ms", "result_json")


def pin(path):
    path = Path(path).absolute()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    raw = (json.dumps(policy.plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return pin(path)


def identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def raw_unknown(attempt):
    # Test data, never a claim that a provider actually dispatched in this test.
    result = {"schema": "database-portal-callback-intent@1",
        "callback_state": "started_outcome_unknown", "provider_effect_state": "unknown_may_have_started",
        **{key: getattr(attempt, key) for key in
           ("attempt_id", "task_cid", "claim_id", "lease_id", "fencing_token", "fence_epoch")}}
    return {"invocation_id": "provider:fixture-" + uuid.uuid4().hex,
        "attempt_id": attempt.attempt_id, "task_cid": attempt.task_cid,
        "idempotency_key": "provider:" + attempt.attempt_id,
        "owner_session_id": attempt.owner_session_id, "recorded_at_ms": int(time.time() * 1000),
        "result_json": json.dumps(result, sort_keys=True, separators=(",", ":"))}


def insert_fixture_unknown(daemon, row):
    daemon._require_connection().execute(
        "INSERT INTO provider_invocations VALUES (?,?,?,?,?,?,?)", [row[key] for key in PROVIDER_COLUMNS])


def real_daemon(stores, *, source=None, coordinator=None):
    source = source or DatabaseTaskSource(intent=IntentRepository(stores["canonical"]))
    coordinator = coordinator or DatabaseCoordinator(stores["coordination"])
    # Supplied native adapters keep their own public open lifecycle. The daemon
    # does not implicitly open a caller-owned coordinator before claim_next.
    coordinator.open()
    return DatabaseImplementationDaemon(database_path=stores["canonical"],
        coordination_path=stores["coordination"], execution_path=stores["execution"],
        state_dir=Path(stores["execution"]).parent, state_prefix="crash-copy",
        task_shard_count=3, task_shard_index=1, strict_task_sharding=True,
        authority_mode="embedded_exclusive", task_source_kind="duckdb", task_source=source,
        coordinator=coordinator, require_real_execution=True, lease_ms=3_600_000)


def close_native(daemon):
    # Public adapter properties open the daemon; retain its genuine handles
    # before close so cleanup never attempts to reopen a closed generation.
    source, coordinator = daemon._task_source, daemon._coordinator
    daemon.close()
    if coordinator is not None:
        coordinator.close()
    if source is not None:
        source.close()


def test_supplied_native_coordinator_is_open_before_actual_daemon_claim(tmp_path):
    lane = tmp_path / "supplied-coordinator-lane"
    lane.mkdir(mode=0o700)
    stores = {"canonical": tmp_path / "constructor-canonical.duckdb",
              "coordination": tmp_path / "constructor-coordination.duckdb",
              "execution": lane / LANE_BASENAME}
    repository = IntentRepository(stores["canonical"])
    repository.upsert_goal(goal_cid="constructor-goal", goal_alias="CONSTRUCTOR-G", title="TEST ONLY")
    repository.upsert_task(task_cid=TASK, task_alias="FIXTURE-T", goal_cid="constructor-goal",
        status="ready", body={"board_namespace": NAMESPACE, "objective": "TEST ONLY native lifecycle"})
    source = DatabaseTaskSource(intent=repository)
    coordinator = DatabaseCoordinator(stores["coordination"])
    assert coordinator._connection is None
    daemon = real_daemon(stores, source=source, coordinator=coordinator)
    try:
        assert coordinator._connection is not None
        daemon.open()
        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 1
        assert coordinator.get_task_claim(attempt.claim_id).attempt_id == attempt.attempt_id
        assert source.get_task(TASK).status == "in_progress"
    finally:
        close_native(daemon)


def seed_real_ordinal_history(root):
    root.mkdir(mode=0o700)
    stores = {role: root / (role + ".duckdb") for role in ("canonical", "coordination", "execution")}
    historical_lane, second_lane = root / "historical-lane", root / "second-lane"
    historical_lane.mkdir(mode=0o700)
    second_lane.mkdir(mode=0o700)
    old_execution = historical_lane / LANE_BASENAME
    stores["execution"] = old_execution
    repository = IntentRepository(stores["canonical"])
    repository.upsert_objective(objective_id="fixture-objective", objective_alias="FIXTURE-O", title="TEST ONLY")
    repository.upsert_goal(goal_cid="fixture-goal", goal_alias="FIXTURE-G", title="TEST ONLY")
    repository.upsert_plan(plan_cid="fixture-plan", goal_cid="fixture-goal", plan_alias="FIXTURE-P")
    for number in range(64):
        selected = number == 0
        repository.upsert_task(task_cid=TASK if selected else f"fixture-unrelated-{number:02d}",
            task_alias="FIXTURE-T" if selected else f"FIXTURE-U{number:02d}", goal_cid="fixture-goal",
            objective_id="fixture-objective", plan_cid="fixture-plan", ordinal=number,
            status="ready" if selected else "blocked", body={"board_namespace": NAMESPACE,
                "objective": "TEST ONLY disposable native fixture", "fixture_number": number},
            outputs=[{"path": f"data/fixture-{number}.txt", "effect": "write"}],
            acceptance=[f"TEST ONLY criterion {number}"], validations=[["/usr/bin/true"]])
    source = DatabaseTaskSource(intent=repository)
    coordinator = DatabaseCoordinator(stores["coordination"])
    daemon = real_daemon(stores, source=source, coordinator=coordinator)
    daemon.open()
    first = daemon.claim_next()
    assert first is not None and first.task_cid == TASK and first.attempt_number == 1
    first = daemon.commit_phase(first, "context", body={"fixture_only": True})
    historical_row = raw_unknown(first)
    insert_fixture_unknown(daemon, historical_row)
    failure = daemon._build_portal_failure_settlement_receipt(first,
        failure=RuntimeError("TEST ONLY historical fixture terminal failure"))
    first = daemon._commit_portal_failure_phase(first, failure)
    daemon._fail_coordination_claim(coordinator.get_task_claim(first.claim_id), failure)
    daemon._settle_portal_failure_control_task(first, failure)
    historical_attempt = policy.plain(first.to_dict())
    daemon.close()
    # Explicit disposable fixture preparation, not behavior of the new service.
    # Uses native public rearm authority; no namespace/nonreachability proof.
    rearmed = source.rearm_blocked_task(TASK,
        receipt={"operation": TASK_CLAIM_FAILURE_REARM_OPERATION, "settlement_id": failure["settlement_id"]})
    coordinator.rearm_failed_task(failure_receipt=failure, control_task_observation=rearmed.task.to_dict())
    stores["execution"] = second_lane / LANE_BASENAME
    second_daemon = real_daemon(stores, source=source, coordinator=coordinator)
    second_daemon.open()
    second = second_daemon.claim_next()
    assert second is not None and second.task_cid == TASK and second.attempt_number == 2
    second = second_daemon.commit_phase(second, "context", body={"fixture_only": True, "feature_score": 0.75})
    second_row = raw_unknown(second)
    registered = {"attempt": policy.plain(second.to_dict()),
                  "phases": policy.plain(second_daemon.phase_history(second.attempt_id)),
                  "coordination": {"claim": policy.plain(coordinator.get_task_claim(second.claim_id).to_dict()),
                      "lease": policy.plain(coordinator.get_lease(second.lease_id).to_dict()),
                      "attempt": policy.plain(coordinator.get_task_attempt(second.attempt_id).to_dict()),
                      "previous_attempts": [policy.plain(coordinator.get_task_attempt(first.attempt_id).to_dict())]}}
    close_native(second_daemon)
    return stores, registered, historical_attempt, historical_row, second_row, old_execution


def genuine_maintenance_and_status(created, *, task, logical_store_id, producer):
    path = created["path"]
    writer = path.with_name("." + path.name + ".writer.lock")
    policy_lock = path.with_name("." + path.name + ".lock")
    metadata = created["result"]["metadata"]
    known = {key: value for key, value in metadata.items() if key not in {
        "process_instance_id", "state_schema_revision", "control_schema_profile_id", "control_schema_fingerprint"}}
    selected_attempt = created["result"]["attempt_identity"]
    bound = status.NativeExecutionStatusBinding(backend="duckdb", execution_path=str(path),
        execution_file_identity=identity(path), writer_lock_identity=identity(writer), expected_metadata=known,
        source_bindings={str(item): pin(item)["sha256"] for item in
            (producer, Path(status.__file__).absolute(), Path(duckdb_state.__file__).absolute(), HELPER_PATH)},
        control_store_id=logical_store_id, control_schema_version=5, task_cid=TASK,
        control_revision=task["revision"], expected_control_status="in_progress",
        **{key: selected_attempt[key] for key in policy.IDENTITY_FIELDS if key != "task_cid"})
    backups = path.parent.parent / "retained-wal-backups"
    backups.mkdir(mode=0o700)
    binding = maintenance.NativeExecutionWalMaintenanceBinding.from_dict({
        "schema": maintenance.BINDING_SCHEMA, "status_binding": bound.to_dict(),
        "wal_file_identity": identity(created["wal"]), "policy_lock_identity": identity(policy_lock),
        "backup_directory": str(backups),
        "maintenance_source_sha256": pin(Path(maintenance.__file__).absolute())["sha256"]})
    calls = []

    def disposable_custody(*, force_ack=False, check_only=False):
        # Same explicit nonadmitting guard as the published physical fixture.
        # It never emits native ACK/grant/admission artifacts.
        assert created["child_reaped"] is True and path.parent.is_dir()
        calls.append((force_ack, check_only))

    owner = maintenance.NativeExecutionWalMaintenanceOwner(binding,
        deadline_monotonic=time.monotonic() + 60, custody_guard=disposable_custody)
    try:
        report = owner.run()
    finally:
        owner.close()
    assert report["logical_rows_preserved"] is report["all_metadata_preserved"] is True
    assert report["unknown_not_settled"] is True and not created["wal"].exists()
    assert owner._connection is None and owner._fds == {}
    refreshed = replace(bound, execution_file_identity=report["execution_file_identity"])
    with status.NativeExecutionStatusOwner(refreshed, deadline_monotonic=time.monotonic() + 30) as reader:
        observation = reader.observe_attempt()
    assert observation["original_process_impersonated"] is False
    assert observation["provider_records"][0]["result"]["callback_state"] == "started_outcome_unknown"
    return binding, refreshed, report, observation


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    stores, preimage, old_attempt, old_row, new_row, old_execution = seed_real_ordinal_history(tmp_path / "seed")
    copied = tmp_path / "copies"
    copied.mkdir(mode=0o700)
    seed_pins = {role: pin(path) for role, path in stores.items()}
    for role in ("canonical", "coordination"):
        target = copied / (role + ".duckdb")
        shutil.copyfile(stores[role], target)
        target.chmod(0o600)
        stores[role] = target
    historical = copied / "retained-historical-unknown.duckdb"
    shutil.copyfile(old_execution, historical)
    historical.chmod(0o600)
    producer = Path(importlib.util.find_spec(DatabaseImplementationDaemon.__module__).origin).absolute()
    created = crash_copy.create_crash_copy(copied / "current-lane", Path(seed_pins["execution"]["path"]),
                                            new_row, producer)
    stores["execution"] = created["path"]
    for original in seed_pins.values():
        assert pin(original["path"]) == original
    source = DatabaseTaskSource(intent=IntentRepository(stores["canonical"], install_schema=False), install_schema=False)
    snapshot = {"schema": policy.SNAPSHOT_SCHEMA, **preimage,
        "task": policy.plain(source.get_task(TASK).to_dict()), "intent_task": policy.plain(source.intent.get_task(TASK)),
        "canonical_population": policy.canonical_population(source), "providers": [new_row], "effects": [],
        "historical_unknowns": [{"row": old_row, "attempt": old_attempt, "store_pins": [pin(historical)]}]}
    assert len(snapshot["canonical_population"]["tasks"]) == 64
    source.close()
    maintenance_binding, status_binding, maintenance_report, observation = genuine_maintenance_and_status(
        created, task=snapshot["task"], logical_store_id="fixture-copied-native-store", producer=producer)
    assert {key: observation["attempt"][key] for key in policy.IDENTITY_FIELDS} == {
        key: snapshot["attempt"][key] for key in policy.IDENTITY_FIELDS}
    connection = duckdb.connect(str(stores["canonical"]))
    live_identity = {"server_id": "fixture-admin-" + uuid.uuid4().hex,
                     "generation": 1, "store_id": "fixture-copied-native-store"}
    gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=copied / "admin.sock",
        store_id=live_identity["store_id"], identity=live_identity)
    gateway.start()
    fixture_document = {"schema": "nonadmitting-native-three-store-fixture@1", "fixture_only": True,
        "production_custody_qualified": False, "native_orchestration_receipts_created": False,
        "crash_child_pid": created["child_pid"], "maintenance_receipt": maintenance_report,
        "status_observation": observation, "maintenance_binding": maintenance_binding.to_dict(),
        "status_binding": status_binding.to_dict()}
    fixture_pin = write_new(copied / "nonadmitting-custody-boundary.json", fixture_document)
    snapshot_pin = write_new(copied / "registered-native-preimage.json", snapshot)
    prefix = Path(policy.__file__).absolute().parents[1]
    relatives = ("todo_daemon/planning_unknown_abandonment.py", "todo_daemon/planning_unknown_abandonment_settlement.py",
        "todo_daemon/implementation_daemon.py", "task_sources/typed_state_owner.py", "task_sources/database_task_source.py",
        "task_sources/intent_repository.py", "task_sources/control_plane_contracts.py", "task_sources/duckdb_state.py",
        "merge/database_coordination.py", "runtime/quack_state_server.py", "todo_daemon/native_execution_status_owner.py",
        "todo_daemon/native_execution_wal_maintenance.py")
    context = {"schema": policy.AUTHORITY_SCHEMA, "task_cid": TASK, "namespace": NAMESPACE,
        "expected_task_revision": snapshot["task"]["revision"], "logical_store_id": live_identity["store_id"],
        "stores": {role: str(path) for role, path in stores.items()}, "snapshot_pin": snapshot_pin,
        "source_pins": [pin(prefix / relative) for relative in relatives], "immutable_pins": [fixture_pin],
        "closed_store_pins": {role: pin(stores[role]) for role in ("execution", "coordination")},
        "stopped_processes": [{"pid": created["child_pid"], "start_time_ticks": created["child_start_time_ticks"],
                               "boot_id": created["boot_id"]}],
        "operator_decision": {**policy.DECISION, "unknown_effect_risk_accepted": True,
                              "reason": "TEST ONLY administrative closeout of disposable retained UNKNOWN data"},
        "registered_owner_identity": live_identity,
        "registered_creator": {"pid": os.getpid(), "process_birth_id": typed.kernel_process_birth_id(os.getpid())},
        "custody_artifacts": {"nonadmitting_fixture_manifest": fixture_pin}}
    context["context_cid"] = content_identity(context)
    custody_calls = []

    def explicit_nonadmitting_custody(value, registered, *, check_current_file):
        # Narrow, visible unit boundary: no production custody artifact exists.
        # The full policy, native-store checks and durable stage code stay real.
        assert value["custody_artifacts"] == {"nonadmitting_fixture_manifest": fixture_pin}
        assert policy.decode(policy.read_pin(fixture_pin)) == fixture_document
        assert registered == snapshot
        if check_current_file:
            assert identity(stores["execution"]) == maintenance_report["execution_file_identity"]
        custody_calls.append(check_current_file)

    monkeypatch.setattr(policy, "verify_custody", explicit_nonadmitting_custody)
    gateway.bind_planning_unknown_abandonment_service(expected_identity=live_identity, authority_context=context)
    clients = []
    owner_closed = False
    daemons = []

    def client():
        token, grant = gateway.issue_planning_unknown_abandonment_grant()
        value = typed.TypedStateOwnerConnection(socket_path=gateway.socket_path, token=token,
            client_id=grant.client_id, process_birth_id=grant.process_birth_id, store_id=live_identity["store_id"])
        clients.append(value)
        return value

    def decide(value=None):
        return (value or client()).record_planning_unknown_abandonment_decision(
            context["context_cid"], context["expected_task_revision"])

    def close_audit_owner():
        nonlocal owner_closed
        for value in clients:
            value.close()
        clients.clear()
        if not owner_closed:
            gateway.stop()
            connection.close()
            owner_closed = True

    def reopen():
        close_audit_owner()
        daemon = real_daemon(stores)
        daemon.open()
        daemons.append(daemon)
        return daemon

    yield SimpleNamespace(stores=stores, snapshot=snapshot, context=context, clients=clients, gateway=gateway,
        connection=connection, client=client, decide=decide, close_audit_owner=close_audit_owner, reopen=reopen,
        historical_pin=pin(historical), historical_path=historical, custody_calls=custody_calls,
        preimage=copy.deepcopy(snapshot), created=created)
    for daemon in reversed(daemons):
        close_native(daemon)
    close_audit_owner()


def exact_audit(daemon, case):
    attempt = daemon.get_attempt(case.snapshot["attempt"]["attempt_id"])
    return policy.exact_audit(daemon.task_source, attempt=attempt, binding=daemon._control_binding_for_attempt(attempt))


def assert_exact_preservation(case, daemon):
    attempt = daemon.get_attempt(case.snapshot["attempt"]["attempt_id"])
    assert settlement.provider_rows(daemon, attempt.attempt_id) == case.snapshot["providers"]
    assert pin(case.historical_path) == case.historical_pin
    historical = duckdb.connect(str(case.historical_path), read_only=True)
    try:
        row = historical.execute("SELECT " + ",".join(PROVIDER_COLUMNS) + " FROM provider_invocations WHERE attempt_id=?",
            [case.snapshot["historical_unknowns"][0]["attempt"]["attempt_id"]]).fetchone()
        assert dict(zip(PROVIDER_COLUMNS, row)) == case.snapshot["historical_unknowns"][0]["row"]
    finally:
        historical.close()
    assert pin(case.historical_path) == case.historical_pin
    current = policy.canonical_population(daemon.task_source)
    before = case.snapshot["canonical_population"]
    assert {key: value for key, value in current.items() if key != "tasks"} == {
        key: value for key, value in before.items() if key != "tasks"}
    assert [row for row in current["tasks"] if row["task_cid"] != TASK] == [
        row for row in before["tasks"] if row["task_cid"] != TASK]
    earlier = case.snapshot["coordination"]["previous_attempts"][0]
    assert policy.plain(daemon.coordinator.get_task_attempt(earlier["attempt_id"]).to_dict()) == earlier
    rows = daemon.coordinator._require().execute(
        "SELECT attempt_id,attempt_number,owner_session_id,fencing_token,fence_epoch FROM task_attempts WHERE task_cid=? ORDER BY attempt_number",
        [TASK]).fetchall()
    assert [(row[0], row[1]) for row in rows] == [(earlier["attempt_id"], 1), (attempt.attempt_id, 2)]
    assert tuple(rows[1][key] for key in ("owner_session_id", "fencing_token", "fence_epoch")) == tuple(
        case.snapshot["attempt"][key] for key in ("owner_session_id", "fencing_token", "fence_epoch"))


def assert_closed(case, daemon, result):
    assert result["provider_outcome"] == "unknown" and result["possible_external_effects"] == "unverified"
    for key in ("callback_closed", "remote_cancellation_proved", "nonreachability_asserted",
                "automatic_retry_admitted", "operator_rearm_performed", "successor_attempt_admitted", "completion_authority"):
        assert result[key] is False
    assert result["control_status"] == "blocked" and result["execution_status"] == "failed"
    assert result["control_revision"] == case.context["expected_task_revision"] + 1
    audit = exact_audit(daemon, case)
    assert audit["context"]["registered_owner_identity"] == case.context["registered_owner_identity"]
    assert audit["body"]["recovered_execution_verified_by_this_audit"] is False
    assert_closed_count = daemon._require_connection().execute(
        "SELECT COUNT(*) FROM daemon_execution_events WHERE attempt_id=? AND event_type='terminal_portal_failure_settled'",
        [case.snapshot["attempt"]["attempt_id"]]).fetchone()[0]
    assert assert_closed_count == 1
    assert_exact_preservation(case, daemon)
    assert TASK in daemon._automatic_claim_exclusions()


def test_genuine_aged_audit_lost_response_then_gateway_shutdown_and_fresh_native_closeout(native_case, monkeypatch):
    # The actual native append/event path records an aged immutable audit. No
    # evidence/event row is rewritten to simulate freshness or authenticity.
    genuine_utc = intent_module._utc_iso
    monkeypatch.setattr(intent_module, "_utc_iso", lambda: "2000-01-01T00:00:00+00:00")
    client = native_case.client()
    genuine = client._request

    def lose_after_native_response(*args, **kwargs):
        genuine(*args, **kwargs)
        raise OSError("TEST ONLY native audit committed; client response discarded")

    monkeypatch.setattr(client, "_request", lose_after_native_response)
    with pytest.raises(typed.TypedStateOwnerPlanningUnknownAbandonmentOutcomeUnknownError):
        native_case.decide(client)
    replay = native_case.decide()
    assert replay["replayed"] is True
    assert replay["audit_receipt"]["details"]["created_at"].startswith("2000-01-01")
    assert native_case.connection.execute("SELECT COUNT(*) FROM evidence_nodes WHERE evidence_kind=?",
        [policy.EVIDENCE_KIND]).fetchone()[0] == 1
    monkeypatch.setattr(intent_module, "_utc_iso", genuine_utc)
    old_identity = copy.deepcopy(native_case.context["registered_owner_identity"])
    daemon = native_case.reopen()  # Actual gateway stops and its native connection closes.
    assert daemon.process_instance_id
    monkeypatch.setattr(daemon.task_source.intent, "_clock_ms", lambda: int(time.time() * 1000) + 7_200_000)
    assert daemon.task_source.intent.evidence_freshness_seconds > 0
    assert all(item["evidence_id"] != replay["audit_receipt"]["subject_id"]
               for item in daemon.task_source.intent.current_evidence_for_task(TASK))
    result = daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert_closed(native_case, daemon, result)
    assert exact_audit(daemon, native_case)["context"]["registered_owner_identity"] == old_identity
    assert daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context) == result
    assert_closed(native_case, daemon, result)


@pytest.mark.parametrize("termination", ["graceful_exit", "crash_exit"])
def test_actual_admin_creator_process_exits_before_fresh_native_audit_recovery(native_case, termination):
    # No audit was appended by the parent gateway; close its generation first.
    native_case.close_audit_owner()
    owner = admin_owner_fixture.start_admin_owner(
        Path(native_case.stores["canonical"]).parent / "admin-owner",
        Path(native_case.stores["canonical"]), copy.deepcopy(native_case.context),
        native_case.context["logical_store_id"])
    witness = None
    try:
        context = copy.deepcopy(owner.context)
        assert context["registered_creator"]["pid"] == owner.child_pid != os.getpid()
        assert context["registered_owner_identity"] != native_case.context["registered_owner_identity"]
        client = owner.client()
        decision = client.record_planning_unknown_abandonment_decision(
            context["context_cid"], context["expected_task_revision"])
        assert decision["replayed"] is False
        client.close()
        witness = owner.crash_and_reap() if termination == "crash_exit" else owner.stop_and_reap()
        assert witness["reaped"] is witness["child_reaped"] is True
        assert witness["termination_kind"] == termination
        assert witness["child_pid"] == context["registered_creator"]["pid"]
        assert witness["child_process_birth_id"] == context["registered_creator"]["process_birth_id"]
        assert witness["production_custody_qualified"] is False
        if termination == "crash_exit":
            assert witness["gateway_stopped_by_child"] is witness["connection_closed_by_child"] is False
            assert witness["canonical_wal_present_after_exit"] is True
        else:
            assert witness["gateway_stopped_by_child"] is witness["connection_closed_by_child"] is True
        native_case.context = context
        daemon = native_case.reopen()
        audit = exact_audit(daemon, native_case)
        assert audit["event"]["event_id"] == decision["audit_receipt"]["event_id"]
        assert audit["context"]["registered_creator"] == context["registered_creator"]
        # A fresh genuine daemon resolves the old durable audit. There is no
        # bind call, replacement owner identity, or second decision append.
        result = daemon.settle_planning_unknown_administrative_abandonment(authority_context=context)
        assert_closed(native_case, daemon, result)
        assert daemon.settle_planning_unknown_administrative_abandonment(authority_context=context) == result
        with daemon.task_source.intent._connection(write=False) as connection:
            assert connection.execute("SELECT COUNT(*) FROM evidence_nodes WHERE evidence_kind=?",
                [policy.EVIDENCE_KIND]).fetchone()[0] == 1
    finally:
        if witness is None:
            owner.stop_and_reap()


@pytest.mark.parametrize("stage", ["failed", "coordination", "blocked", "terminal_event"])
def test_native_closeout_resumes_after_each_committed_stage_response_loss(native_case, monkeypatch, stage):
    audit = native_case.decide()
    assert audit["replayed"] is False
    daemon = native_case.reopen()
    names = {"failed": "_commit_portal_failure_phase", "coordination": "_fail_coordination_claim",
             "blocked": "_settle_portal_failure_control_task", "terminal_event": "_record_event"}
    name = names[stage]
    genuine = getattr(daemon, name)
    lost = []

    def lose_after_commit(*args, **kwargs):
        result = genuine(*args, **kwargs)
        if stage != "terminal_event" or args[0] == "terminal_portal_failure_settled":
            lost.append(True)
            raise RuntimeError("TEST ONLY response lost after committed " + stage)
        return result

    monkeypatch.setattr(daemon, name, lose_after_commit)
    with pytest.raises(RuntimeError, match="TEST ONLY response lost"):
        daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert lost == [True]
    assert_exact_preservation(native_case, daemon)
    # Ordinary daemon reconciliation must remain inert at every ADMIN stage.
    assert daemon.reconcile_terminal_portal_failures() == []
    close_native(daemon)
    recovered = native_case.reopen()
    result = recovered.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert_closed(native_case, recovered, result)
    assert recovered.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context) == result
    assert_closed(native_case, recovered, result)


@pytest.mark.parametrize("change", ["accepted_provider_result", "new_effect"])
def test_native_recovery_refuses_changed_result_or_effect_before_failed_stage(native_case, change):
    native_case.decide()
    daemon = native_case.reopen()
    attempt = daemon.get_attempt(native_case.snapshot["attempt"]["attempt_id"])
    connection = daemon._require_connection()
    if change == "accepted_provider_result":
        connection.execute("UPDATE provider_invocations SET result_json=? WHERE attempt_id=?",
            [json.dumps({"accepted": True, "status": "TEST ONLY changed-result"}), attempt.attempt_id])
    else:
        connection.execute("INSERT INTO effect_claims VALUES (?,?,?,?,?,?,?,?)",
            ["effect:TEST-ONLY-new", attempt.attempt_id, TASK, "TEST-ONLY-effect-key", "effect:" + attempt.attempt_id,
             attempt.owner_session_id, int(time.time() * 1000), "{}"])
    original = daemon.get_attempt(attempt.attempt_id).to_dict()
    control = policy.plain(daemon.task_source.intent.get_task(TASK))
    claim = policy.plain(daemon.coordinator.get_task_claim(attempt.claim_id).to_dict())
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert daemon.get_attempt(attempt.attempt_id).to_dict() == original
    assert policy.plain(daemon.task_source.intent.get_task(TASK)) == control
    assert policy.plain(daemon.coordinator.get_task_claim(attempt.claim_id).to_dict()) == claim
    assert daemon._require_connection().execute(
        "SELECT COUNT(*) FROM daemon_execution_events WHERE event_type='terminal_portal_failure_settled'").fetchone()[0] == 0
    assert pin(native_case.historical_path) == native_case.historical_pin


@pytest.mark.parametrize("change", ["evidence_parent", "evidence_digest", "event_owner"])
def test_fresh_native_recovery_authenticates_exact_durable_audit_before_failed(native_case, change):
    decision = native_case.decide()
    daemon = native_case.reopen()
    receipt = decision["audit_receipt"]
    with daemon.task_source.intent._connection(write=True) as connection:
        if change == "evidence_parent":
            connection.execute("UPDATE evidence_nodes SET parent_evidence_id=? WHERE evidence_id=?",
                               ["TEST-ONLY-foreign-parent", receipt["subject_id"]])
        elif change == "evidence_digest":
            connection.execute("UPDATE evidence_nodes SET digest=? WHERE evidence_id=?",
                               ["TEST-ONLY-foreign-digest", receipt["subject_id"]])
        else:
            row = connection.execute(
                "SELECT stream_id,sequence,global_sequence,event_type,body_json FROM domain_events WHERE event_id=?",
                [receipt["event_id"]]).fetchone()
            envelope = policy.decode(row[4])
            envelope["owner_id"] = "TEST-ONLY-another-native-owner"
            identity = content_identity({"stream_id": row[0], "sequence": row[1],
                "global_sequence": row[2], "event_type": row[3], "body": envelope})
            connection.execute("UPDATE domain_events SET event_id=?,body_json=? WHERE event_id=?",
                [identity, canonical_json_bytes(envelope).decode("utf-8"), receipt["event_id"]])
    attempt = daemon.get_attempt(native_case.snapshot["attempt"]["attempt_id"])
    control = policy.plain(daemon.task_source.intent.get_task(TASK))
    claim = policy.plain(daemon.coordinator.get_task_claim(attempt.claim_id).to_dict())
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    assert policy.plain(daemon.task_source.intent.get_task(TASK)) == control
    assert policy.plain(daemon.coordinator.get_task_claim(attempt.claim_id).to_dict()) == claim
    assert_exact_preservation(native_case, daemon)


def test_exact_admin_replay_refuses_later_actual_control_rearm(native_case):
    native_case.decide()
    daemon = native_case.reopen()
    settled = daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert_closed(native_case, daemon, settled)
    rearmed = daemon.task_source.rearm_blocked_task(TASK,
        receipt={"operation": TASK_CLAIM_FAILURE_REARM_OPERATION,
                 "settlement_id": settled["settlement_id"]})
    assert rearmed.task.status == "retrying"
    assert rearmed.task.revision == native_case.context["expected_task_revision"] + 2
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        daemon.settle_planning_unknown_administrative_abandonment(authority_context=native_case.context)
    assert daemon.task_source.get_task(TASK).to_dict() == rearmed.task.to_dict()
    assert_exact_preservation(native_case, daemon)
    assert TASK in daemon._automatic_claim_exclusions()
