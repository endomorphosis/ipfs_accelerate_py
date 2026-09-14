"""Canonical observation must never be confused with replica or drain authority."""

from __future__ import annotations

import os

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
    QuackOwnerMutationError,
    build_mutation_result,
    execute_owner_mutation,
    mutation_operation,
    open_mutation_inbox_directory,
    read_envelope_at,
    service_mutation_inbox,
    write_envelope_atomic_at,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    quack_owner_snapshot as snapshot,
)

TOKEN = "snapshot-test-credential"


@pytest.fixture
def owner(tmp_path):
    path = tmp_path / "control.duckdb"
    repository = open_intent_repository(path, owner_id="test")
    repository.upsert_objective(
        objective_id="objective:test", objective_alias="O", title="Objective"
    )
    repository.upsert_goal(
        goal_cid="goal:test",
        goal_alias="G",
        title="Goal",
        objective_id="objective:test",
    )
    repository.upsert_plan(plan_cid="plan:test", goal_cid="goal:test", plan_alias="P")
    repository.upsert_task(
        task_cid="task:test",
        task_alias="T",
        goal_cid="goal:test",
        plan_cid="plan:test",
        objective_id="objective:test",
        ordinal=1,
        status="ready",
    )
    repository.close()
    connection = open_duckdb_connection(path, prefer_quack=False)
    fingerprint = connection.execute(
        "SELECT value FROM control_plane_metadata WHERE key = 'schema_fingerprint'"
    ).fetchone()[0]
    binding = {
        "server_id": "server:test",
        "store_id": "store:test",
        "database_uuid": "1d14e45e-c386-48f0-8263-46c3b6770a32",
        "schema_revision": 1,
        "schema_fingerprint": fingerprint,
        "generation": 7,
        "process_birth_id": "birth:test",
        "listen_uri": "quack:127.0.0.1:24070",
        "extension_fingerprint": "none",
    }
    connection.execute("DELETE FROM store_generations")
    connection.execute(
        "INSERT INTO store_generations (generation,schema_revision,fence_epoch,revision,database_uuid,birth_id,created_at) VALUES (7,1,3,2,?,?,?)",
        [binding["database_uuid"], binding["process_birth_id"], "2026-09-09T00:00:00Z"],
    )
    connection.execute(
        "INSERT INTO state_servers (server_id,store_id,database_uuid,process_birth_id,listen_uri,extension_fingerprint,schema_revision,generation,started_at,status,revision) VALUES (?,?,?,?,?,?,1,7,?,?,1)",
        [
            binding[k]
            for k in (
                "server_id",
                "store_id",
                "database_uuid",
                "process_birth_id",
                "listen_uri",
                "extension_fingerprint",
            )
        ]
        + ["2026-09-09T00:00:00Z", "ready"],
    )
    try:
        yield connection, binding, tmp_path / "inbox"
    finally:
        connection.close()


def result_for(connection, request):
    counts, observed = execute_owner_mutation(
        connection,
        request,
        refresh_replica=lambda: pytest.fail("read must not refresh a replica"),
    )
    return build_mutation_result(
        request=request, token=TOKEN, ok=True, rowcounts=counts, observed=observed
    )


def test_snapshot_authenticates_complete_inventory_without_effects(owner):
    connection, binding, _ = owner
    before = snapshot.table_commitments(connection)
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    result = result_for(connection, request)
    observed = snapshot.validate_owner_snapshot(result, request=request, token=TOKEN)
    assert observed.generation == 7 and observed.fence_epoch == 3
    assert result["rowcounts"] == [0]
    assert (
        result["observed"]["tables"] == before == snapshot.table_commitments(connection)
    )
    assert {
        "task_claims",
        "effect_claims",
        "leases",
        "completion_receipts",
        "merge_queue_entries",
        "proof_obligations",
    } <= result["observed"]["tables"].keys()
    assert result["observed"]["launch_authority"] is False
    assert result["observed"]["drain_authority"] is False
    assert result["observed"]["completion_authority"] is False


def test_challenge_prevents_replaying_previous_success(owner):
    connection, binding, _ = owner
    first = snapshot.snapshot_request(binding=binding, token=TOKEN)
    second = snapshot.snapshot_request(binding=binding, token=TOKEN)
    result = result_for(connection, first)
    assert first["request_id"] != second["request_id"]
    with pytest.raises(QuackOwnerMutationError, match="authentication"):
        snapshot.validate_owner_snapshot(result, request=second, token=TOKEN)
    with pytest.raises(QuackOwnerMutationError, match="stale"):
        snapshot.validate_owner_snapshot(
            result, request=first, token=TOKEN, now_ms=first["expires_at_ms"] + 1
        )


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE state_servers SET status='stopped'",
        "UPDATE state_servers SET process_birth_id='birth:other'",
        "UPDATE state_servers SET stopped_at='2026-09-09T00:00:01Z'",
        "UPDATE store_generations SET generation=8",
        "UPDATE store_generations SET birth_id='birth:other'",
        "UPDATE control_plane_metadata SET value='changed' WHERE key='schema_fingerprint'",
    ],
)
def test_owner_drift_rejected_and_transaction_rolled_back(owner, change):
    connection, binding, _ = owner
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    connection.execute(change)
    with pytest.raises(QuackOwnerMutationError, match="identity_drifted"):
        result_for(connection, request)
    connection.execute("BEGIN TRANSACTION")
    connection.execute("ROLLBACK")


def test_replica_refused(owner):
    connection, binding, _ = owner
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)

    class Replica:
        _quack_uri = "quack:127.0.0.1:24070"

        def execute(self, *args):
            pytest.fail("replica must not be read")

    with pytest.raises(QuackOwnerMutationError, match="exclusive_writer"):
        result_for(Replica(), request)


def test_tampered_response_and_wrong_token_refused(owner):
    connection, binding, _ = owner
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    result = result_for(connection, request)
    with pytest.raises(QuackOwnerMutationError, match="authentication"):
        snapshot.validate_owner_snapshot(
            result, request=request, token="different-test-token"
        )
    result["observed"]["tables"]["task_claims"]["row_count"] += 1
    with pytest.raises(QuackOwnerMutationError, match="authentication"):
        snapshot.validate_owner_snapshot(result, request=request, token=TOKEN)


def test_bounded_snapshot_fails_without_partial_receipt(owner, monkeypatch):
    connection, binding, _ = owner
    monkeypatch.setattr(snapshot, "MAX_SNAPSHOT_ROWS", 1)
    with pytest.raises(QuackOwnerMutationError, match="population_exceeded"):
        result_for(connection, snapshot.snapshot_request(binding=binding, token=TOKEN))
    connection.execute("BEGIN TRANSACTION")
    connection.execute("ROLLBACK")


def test_table_digest_binds_task_payload_even_without_revision_change(owner):
    connection, _, _ = owner
    before = snapshot.table_commitments(connection)
    connection.execute(
        "UPDATE tasks SET body_json = '{\"changed\":true}' WHERE task_cid='task:test'"
    )
    after = snapshot.table_commitments(connection)
    assert before["tasks"]["row_count"] == after["tasks"]["row_count"]
    assert before["tasks"]["sha256"] != after["tasks"]["sha256"]
    assert before["task_claims"] == after["task_claims"]


@pytest.mark.parametrize(
    "parameters", [[], ["a"], [True], ["a" * 32, "b" * 32], ["A" * 32]]
)
def test_closed_read_shape(parameters):
    with pytest.raises(QuackOwnerMutationError, match="operation_shape"):
        mutation_operation(
            [
                {
                    "template_id": snapshot.OWNER_SNAPSHOT_OPERATION,
                    "parameters": parameters,
                }
            ]
        )


def test_snapshot_cannot_be_composed_into_mutation():
    with pytest.raises(QuackOwnerMutationError):
        mutation_operation(
            [
                {
                    "template_id": snapshot.OWNER_SNAPSHOT_OPERATION,
                    "parameters": ["a" * 32],
                }
            ]
            * 2
        )


def test_owner_inbox_services_read_and_rejects_interrupted_read(owner):
    connection, binding, inbox = owner
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    descriptor = open_mutation_inbox_directory(inbox)
    try:
        write_envelope_atomic_at(
            descriptor, request["request_id"] + ".request.json", request
        )
        assert (
            service_mutation_inbox(
                connection,
                inbox=inbox,
                binding=binding,
                token=TOKEN,
                refresh_replica=lambda: pytest.fail("unexpected refresh"),
            )
            == 1
        )
        result = read_envelope_at(descriptor, request["request_id"] + ".done.json")
        snapshot.validate_owner_snapshot(result, request=request, token=TOKEN)
        interrupted = snapshot.snapshot_request(binding=binding, token=TOKEN)
        write_envelope_atomic_at(
            descriptor, interrupted["request_id"] + ".processing.json", interrupted
        )
        service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=binding,
            token=TOKEN,
            refresh_replica=lambda: pytest.fail("unexpected refresh"),
        )
        retained = read_envelope_at(
            descriptor, interrupted["request_id"] + ".done.json"
        )
        with pytest.raises(
            QuackOwnerMutationError, match="owner_interrupted_no_effect"
        ):
            snapshot.validate_owner_snapshot(retained, request=interrupted, token=TOKEN)
    finally:
        os.close(descriptor)


def test_native_cid_fingerprint_matches_admitted_sha256(owner):
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        _schema_fingerprint_digest,
    )

    connection, binding, _ = owner
    binding = {
        **binding,
        "schema_fingerprint": _schema_fingerprint_digest(binding["schema_fingerprint"]),
    }
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    result = result_for(connection, request)
    snapshot.validate_owner_snapshot(result, request=request, token=TOKEN)


def test_client_observation_runs_existing_inbox_end_to_end(owner, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_owner_mutation as mutation,
    )

    connection, binding, inbox = owner

    def service(_seconds):
        service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=binding,
            token=TOKEN,
            refresh_replica=lambda: pytest.fail("unexpected refresh"),
        )

    monkeypatch.setattr(mutation.time, "sleep", service)
    evidence = snapshot.observe_owner(binding=binding, token=TOKEN, inbox=inbox)
    observed = snapshot.validate_owner_snapshot(
        evidence["result"], request=evidence["request"], token=TOKEN
    )
    assert observed.generation == 7


def test_foreign_generation_read_gets_no_signed_oracle(owner):
    connection, binding, inbox = owner
    request = snapshot.snapshot_request(
        binding={**binding, "generation": 8}, token=TOKEN
    )
    descriptor = open_mutation_inbox_directory(inbox)
    try:
        write_envelope_atomic_at(
            descriptor, request["request_id"] + ".request.json", request
        )
        service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=binding,
            token=TOKEN,
            refresh_replica=lambda: pytest.fail("unexpected refresh"),
        )
        assert not (inbox / (request["request_id"] + ".done.json")).exists()
    finally:
        os.close(descriptor)


def test_signed_missing_inventory_cannot_be_used_as_empty_state(owner):
    connection, binding, _ = owner
    request = snapshot.snapshot_request(binding=binding, token=TOKEN)
    result = result_for(connection, request)
    observed = result["observed"]
    del observed["tables"]["leases"]
    malformed = build_mutation_result(
        request=request, token=TOKEN, ok=True, rowcounts=[0], observed=observed
    )
    with pytest.raises(QuackOwnerMutationError, match="inventory_invalid"):
        snapshot.validate_owner_snapshot(malformed, request=request, token=TOKEN)


def test_physical_schema_drift_cannot_hide_behind_metadata(owner):
    connection, binding, _ = owner
    connection.execute("ALTER TABLE tasks ADD COLUMN unadmitted VARCHAR")
    with pytest.raises(QuackOwnerMutationError, match="physical_schema_drifted"):
        result_for(connection, snapshot.snapshot_request(binding=binding, token=TOKEN))


def test_unknown_native_table_cannot_be_silently_omitted(owner):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        compute_schema_fingerprint,
    )

    connection, binding, _ = owner
    connection.execute("CREATE TABLE unknown_claims (claim_id VARCHAR)")
    fingerprint = compute_schema_fingerprint(connection)
    connection.execute(
        "UPDATE control_plane_metadata SET value=? WHERE key='schema_fingerprint'",
        [fingerprint],
    )
    binding = {**binding, "schema_fingerprint": fingerprint}
    with pytest.raises(QuackOwnerMutationError, match="schema_inventory_drifted"):
        result_for(connection, snapshot.snapshot_request(binding=binding, token=TOKEN))
