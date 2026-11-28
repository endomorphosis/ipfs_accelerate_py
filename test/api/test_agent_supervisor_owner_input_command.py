"""Real bound-owner dispatch controls; no sockets or simulated repositories."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import duckdb_available
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource, execute_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnectionPolicyError, open_duckdb_connection,
    QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository, IntentRepositoryConflictError, IntentRepositoryError,
    OWNER_INPUT_WAIT_KIND, OWNER_INPUT_OBSERVATION_KIND, task_authority_spec_cid,
)


pytestmark = pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
TASK = "task:typed-owner-test"
OWNER = "owner:legal-input-fixture"
KEY = "legal_cohort"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
STORE = {"store_id": "data/control.duckdb", "store_generation": "generation-1"}


@pytest.fixture
def command_path(tmp_path: Path):
    """Small normal materialization matching the existing Quack owner fixture."""
    path = tmp_path / "control.duckdb"
    with DatabaseTaskSource(path) as source:
        source.materialize({
            "repository_tree_id": "tree:typed-owner-test",
            "objectives": [{
                "goal_id": "G1", "goal_cid": "goal:typed-owner-test",
                "objective_id": "objective:typed-owner-test", "title": "Typed owner commands",
            }],
            "taskboard": [{
                "task_id": "T1", "task_cid": TASK, "goal_cid": "goal:typed-owner-test",
                "status": "ready", "admitted": False, "formalized": False,
            }],
        })
    return path


@contextmanager
def _owner(path: Path, *, session="session:first"):
    connection = open_duckdb_connection(path)
    repository = IntentRepository(
        path, bound_connection=connection, install_schema=False,
        owner_id="owner:typed-input-test", session_id=session,
    )
    try:
        yield repository, connection
    finally:
        repository.close()
        connection.close()


def _snapshot(connection):
    result = {}
    for table in ("tasks", "task_blocks", "evidence_nodes", "domain_events", "idempotency_records"):
        rows = connection.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()
        result[table] = tuple(tuple(row[index] for index in range(len(row))) for row in rows)
    return result


def _wait_payload(repository):
    return {
        "task_cid": TASK, "evidence_kind": OWNER_INPUT_WAIT_KIND,
        "body": {
            "owner_id": OWNER, "input_key": KEY,
            "missing_fields": ["original_owner_selection_receipt"],
            "observed_input_digest": DIGEST_A,
            "expected_revision": repository.get_task(TASK)["revision"],
        },
    }


def _wait(repository):
    payload = _wait_payload(repository)
    result = execute_quack_owner_command(
        repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, payload,
        request_id="1" * 32, **STORE,
    )
    return payload, result


def _source_evidence(repository):
    return execute_quack_owner_command(
        repository, QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        {
            "task_cid": TASK, "evidence_kind": "owner_input_source_fixture", "digest": DIGEST_B,
            "body": {
                "task_authority_spec_cid": task_authority_spec_cid(repository.get_task(TASK)),
                "owner_id": OWNER, "input_key": KEY,
            },
        },
        request_id="2" * 32, **STORE,
    )


def test_bound_owner_command_retains_wait_and_observation_and_replays_after_restart(command_path):
    with _owner(command_path) as (repository, connection):
        original = deepcopy(dict(repository.get_task(TASK)))
        wait_payload, waiting = _wait(repository)
        assert waiting["changed"] is True
        blocked = repository.get_task(TASK)
        assert blocked["status"] == "blocked"
        for key in ("body", "identity", "acceptance", "dependencies", "outputs", "validations"):
            assert blocked[key] == original[key]
        before_repeat = _snapshot(connection)
        assert execute_quack_owner_command(
            repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, wait_payload,
            request_id="1" * 32, **STORE,
        ) == waiting
        assert _snapshot(connection) == before_repeat
        evidence = _source_evidence(repository)
        observation_payload = {
            "task_cid": TASK, "evidence_kind": OWNER_INPUT_OBSERVATION_KIND,
            "body": {
                "request_id": waiting["subject_id"], "owner_id": OWNER,
                "observed_input_digest": DIGEST_B,
                "observation_evidence_id": evidence["subject_id"],
            },
        }
        observed = execute_quack_owner_command(
            repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, observation_payload,
            request_id="3" * 32, **STORE,
        )
        assert observed["changed"] is True
        assert observed["details"]["reconsideration_requested"] is True
        assert observed["details"]["request_id"] == waiting["subject_id"]
        assert repository.get_task(TASK)["status"] == "blocked"
        handoffs = tuple(dict(row) for row in repository.owner_input_handoffs(owner_id=OWNER))
        assert len(handoffs) == 1
        assert handoffs[0]["request_id"] == waiting["subject_id"]
        assert handoffs[0]["observed_input_digest"] == DIGEST_B
        assert handoffs[0]["event_cursor"] == observed["global_sequence"]
        retained = _snapshot(connection)

    with _owner(command_path, session="session:after-restart") as (repository, connection):
        for payload, request_id, receipt in (
            (wait_payload, "1" * 32, waiting), (observation_payload, "3" * 32, observed),
        ):
            assert execute_quack_owner_command(
                repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, payload,
                request_id=request_id, **STORE,
            ) == receipt
            assert _snapshot(connection) == retained
        assert tuple(dict(row) for row in repository.owner_input_handoffs(owner_id=OWNER)) == handoffs


@pytest.mark.parametrize("kind", [OWNER_INPUT_WAIT_KIND, OWNER_INPUT_OBSERVATION_KIND])
def test_generic_evidence_command_cannot_exercise_coordination_authority(command_path, kind):
    with _owner(command_path) as (repository, connection):
        body = _wait_payload(repository)["body"]
        if kind == OWNER_INPUT_OBSERVATION_KIND:
            _, waiting = _wait(repository)
            evidence = _source_evidence(repository)
            body = {
                "request_id": waiting["subject_id"], "owner_id": OWNER,
                "observed_input_digest": DIGEST_B,
                "observation_evidence_id": evidence["subject_id"],
            }
        before = _snapshot(connection)
        with pytest.raises(IntentRepositoryError):
            execute_quack_owner_command(
                repository, QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
                {"task_cid": TASK, "evidence_kind": kind, "digest": content_identity(body), "body": body},
                request_id="4" * 32, **STORE,
            )
        assert _snapshot(connection) == before


@pytest.mark.parametrize("invalid", ["outer_field", "body_field", "kind", "none_body"])
def test_owner_input_command_closed_schema_refuses_without_writes(command_path, invalid):
    with _owner(command_path) as (repository, connection):
        payload = _wait_payload(repository)
        if invalid == "outer_field":
            payload["sql"] = "not executable input"
        elif invalid == "body_field":
            payload["body"]["grant_authority"] = True
        elif invalid == "kind":
            payload["evidence_kind"] = "validation"
        else:
            payload["body"] = None
        before = _snapshot(connection)
        with pytest.raises((DuckDBConnectionPolicyError, IntentRepositoryError)):
            execute_quack_owner_command(
                repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, payload,
                request_id="5" * 32, **STORE,
            )
        assert _snapshot(connection) == before


@pytest.mark.parametrize("changed", ["store_id", "store_generation", "payload"])
def test_owner_input_command_replay_refuses_changed_exact_bindings(command_path, changed):
    with _owner(command_path) as (repository, connection):
        payload, _ = _wait(repository)
        bindings = dict(STORE)
        if changed == "payload":
            payload["body"]["observed_input_digest"] = DIGEST_B
        else:
            bindings[changed] += "-different"
        before = _snapshot(connection)
        with pytest.raises(IntentRepositoryConflictError, match="replay binding differs"):
            execute_quack_owner_command(
                repository, QUACK_OWNER_COMMAND_COORDINATE_OWNER_INPUT, payload,
                request_id="1" * 32, **bindings,
            )
        assert _snapshot(connection) == before
