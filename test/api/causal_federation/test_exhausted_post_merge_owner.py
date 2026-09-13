"""Real owner transactions for retained callback recovery after exhaustion.

All task, history and lease rows below belong to a disposable test database.
Canonical history proves admission eligibility; these tests do not claim that
history alone supplies the independent Portal or execution completion proof.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandOutcome,
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    DETERMINISTIC_ONLY_EXECUTION_MODE,
    TaskExecutionRoutePolicy,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.causal_federation.test_admitted_executor import (
    _typed_bootstrap_credentials,
)
from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate


COMMAND = "task.post_merge.exhausted_retry.recover"
CLIENT_ID = "database-implementation-daemon:exhausted-post-merge-test"
def _plain(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _seed_case(tmp_path: Path, *, mutation: str | None = None) -> SimpleNamespace:
    # The pure contract fixture is shared with the independent denial tests.
    from test.api.causal_federation.test_exhausted_post_merge_contract import (
        exhausted_fixture,
    )

    task, history, prior_queue, transition = exhausted_fixture()
    task, history, prior_queue, transition = copy.deepcopy(
        (task, history, prior_queue, transition)
    )
    database = tmp_path / "exhausted-post-merge.duckdb"
    source = DatabaseTaskSource(database)
    semantic = {
        key: value
        for key, value in task["body"].items()
        if key != "completion_receipt"
    }
    source.materialize(
        {
            "repository_tree_id": "tree:exhausted-post-merge-test",
            "plan_root_cid": "plan:exhausted-post-merge-test",
            "goals": [
                {
                    "goal_cid": "goal:exhausted-post-merge-test",
                    "goal_alias": "EXHAUSTED-POST-MERGE-GOAL",
                    "title": "Recover the exact retained callback",
                }
            ],
            "tasks": [
                {
                    **semantic,
                    "task_cid": task["task_cid"],
                    "task_id": task["task_alias"],
                    "goal_cid": "goal:exhausted-post-merge-test",
                    "status": "ready",
                }
            ],
        }
    )
    ready = source.get(task["task_cid"])
    assert ready is not None
    policy = TaskExecutionRoutePolicy.seal(
        snapshot=source.snapshot(),
        tasks=(ready,),
        execution_modes={task["task_alias"]: DETERMINISTIC_ONLY_EXECUTION_MODE},
    )
    route = policy.binding_for_task(ready).to_dict()
    route_lineage = {
        "execution_route_binding": route,
        "execution_route_policy_id": route["policy_id"],
        "execution_route_origin_revision": route["task_revision"],
    }
    # Keep this fixture's semantics and real launch policy aligned.  Operational
    # receipts remain the exact closed historical shapes supplied by the fixture.
    semantic = dict(ready.body)
    semantic.pop("completion_receipt", None)
    for row in history["revisions"]:
        receipt = row["body"].get("completion_receipt")
        if isinstance(receipt, Mapping):
            receipt = {**receipt, **route_lineage}
            row["body"] = {**semantic, "completion_receipt": receipt}
        else:
            row["body"] = dict(semantic)
    transition.update(route_lineage)
    history.pop("projection_cid", None)
    history["projection_cid"] = content_identity(history)
    task.update(ready.to_dict())
    task.update(
        status="blocked", revision=11, body=history["revisions"][-1]["body"]
    )
    source.close()

    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "UPDATE tasks SET status=?, revision=?, body_json=? WHERE task_cid=?",
            ["blocked", 11, canonical_json_bytes(task["body"]).decode(), task["task_cid"]],
        )
        connection.execute(
            "DELETE FROM task_revisions WHERE task_cid=?", [task["task_cid"]]
        )
        for row in history["revisions"]:
            connection.execute(
                "INSERT INTO task_revisions "
                "(task_cid,revision,status,body_json,recorded_at) VALUES (?,?,?,?,?)",
                [
                    task["task_cid"], row["revision"], row["status"],
                    canonical_json_bytes(row["body"]).decode(), "test",
                ],
            )
        fields = list(prior_queue)
        connection.execute(
            "INSERT INTO leases (" + ",".join(fields) + ") VALUES ("
            + ",".join("?" for _ in fields) + ")",
            [prior_queue[field] for field in fields],
        )
        if mutation == "current_body":
            altered = {**task["body"], "unreviewed_change": True}
            connection.execute(
                "UPDATE tasks SET body_json=? WHERE task_cid=?",
                [canonical_json_bytes(altered).decode(), task["task_cid"]],
            )
        elif mutation == "current_revision":
            connection.execute("UPDATE tasks SET revision=12 WHERE task_cid=?", [task["task_cid"]])
        elif mutation == "history_body":
            altered = {**history["revisions"][7]["body"], "unreviewed_change": True}
            connection.execute(
                "UPDATE task_revisions SET body_json=? WHERE task_cid=? AND revision=8",
                [canonical_json_bytes(altered).decode(), task["task_cid"]],
            )
        elif mutation == "history_gap":
            connection.execute(
                "DELETE FROM task_revisions WHERE task_cid=? AND revision=6",
                [task["task_cid"]],
            )
        elif mutation == "cooldown_reason":
            connection.execute("UPDATE leases SET release_reason='foreign' WHERE task_cid=?", [task["task_cid"]])
        elif mutation == "cooldown_active":
            connection.execute("UPDATE leases SET state='claimed' WHERE task_cid=?", [task["task_cid"]])
        elif mutation is not None:
            raise AssertionError(mutation)
    finally:
        connection.close()
    return SimpleNamespace(
        database=database, task=task, history=history, prior_queue=prior_queue,
        transition=transition, policy=policy, route=route,
        now_ms=task["body"]["completion_receipt"]["execution_finished_at_ms"] + 1_000,
    )


@contextmanager
def _owner(
    case: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    *, state_name: str = "owner", attach_adapter: bool = True,
) -> Iterator[SimpleNamespace]:
    server = build_server(
        database_path=case.database,
        state_dir=tmp_path / state_name,
        repository_id="repository:ipfs_accelerate_py",
        store_id="exhausted-post-merge-test",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
    )
    identity = server.start()
    token, grant = server.issue_typed_client_grant_record(
        client_id=CLIENT_ID,
        process_birth_id=identity.process_birth_id,
        allowed_operations=daemon_required_owner_operations(),
        allowed_command_operations=daemon_required_owner_command_operations(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path()))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=CLIENT_ID, store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
    )
    adapter = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        clock = {"now_ms": case.now_ms}
        if attach_adapter:
            adapter = TypedDatabaseTaskSource(
                client, clock_ms=lambda: clock["now_ms"],
                execution_route_policy=case.policy,
            )
        yield SimpleNamespace(
            server=server, identity=identity, client=client, adapter=adapter,
            token=token, grant=grant, clock=clock,
        )
    finally:
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()


def _client_arguments(case: SimpleNamespace) -> dict[str, Any]:
    return {
        "task": case.task,
        "history": case.history,
        "expected_control_receipt": case.task["body"]["completion_receipt"],
        "transition_receipt": case.transition,
        "now_ms": case.now_ms,
    }


def _adapter_arguments(case: SimpleNamespace) -> dict[str, Any]:
    return {
        "task_cid": case.task["task_cid"],
        "expected_revision": 11,
        "expected_control_receipt": case.task["body"]["completion_receipt"],
        "status": "retrying",
        "receipt": case.transition,
        "delay_ms": 0,
        "reason": case.transition["queue_reason"],
        "selection_penalty": 0,
        "exact_retry_not_before_ms": None,
    }


def _snapshot(owner: SimpleNamespace, task_cid: str) -> dict[str, Any]:
    # This is the already-running disposable owner's own connection.  Capturing
    # raw fixture rows also lets the malformed-history tests compare rollback
    # state without relying on the very validator expected to reject them.
    connection = owner.server._command_gateway._connection
    return {
        "generation": owner.client.load_generation(),
        "task": connection.execute("SELECT * FROM tasks WHERE task_cid=?", [task_cid]).fetchall(),
        "history": connection.execute("SELECT * FROM task_revisions WHERE task_cid=? ORDER BY revision", [task_cid]).fetchall(),
        "cooldown": connection.execute("SELECT * FROM leases WHERE task_cid=?", [task_cid]).fetchall(),
        "idempotency": connection.execute("SELECT * FROM idempotency_records ORDER BY idempotency_key").fetchall(),
    }


@pytest.mark.parametrize(
    "mutation",
    ["current_body", "current_revision", "history_body", "history_gap", "cooldown_reason", "cooldown_active"],
)
def test_exhausted_post_merge_owner_rejects_changed_canonical_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.exhausted_post_merge_recovery import (
        build_parameters,
        validate_parameters,
    )

    case = _seed_case(tmp_path, mutation=mutation)
    parameters = build_parameters(
        **_client_arguments(case), prior_queue=case.prior_queue,
    )
    validated = validate_parameters(parameters)
    final_body = {
        **case.task["body"],
        "completion_receipt": validated["final_transition_receipt"],
    }
    # Submit the valid original closed command after changing canonical rows.
    # This exercises the owner's transaction-time denial independently of the
    # adapter's startup checks and the client's earlier cooldown validation.
    with _owner(case, tmp_path, monkeypatch, attach_adapter=False) as owner:
        before = _snapshot(owner, case.task["task_cid"])
        with pytest.raises(TransactionError, match="authorization_denied"):
            owner.client._submit_post_merge_retry_command(
                parameters, canonical_json_bytes(final_body).decode(),
            )
        assert _snapshot(owner, case.task["task_cid"]) == before


def test_exhausted_post_merge_owner_atomic_replay_and_fresh_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _seed_case(tmp_path)
    task_cid = case.task["task_cid"]
    arguments = _client_arguments(case)
    adapter_arguments = _adapter_arguments(case)
    assert COMMAND in daemon_required_owner_command_operations()
    with _owner(case, tmp_path, monkeypatch) as owner:
        adapter, client = owner.adapter, owner.client
        blocked = adapter.get(task_cid)
        assert blocked is not None and blocked.status == "blocked" and blocked.revision == 11
        assert adapter.task_revision_history_projection(task_cid) == case.history
        assert adapter.get_queue_entry(task_cid).attempt == 2
        before = _snapshot(owner, task_cid)
        # Protected exhaustion must remain refused by the ordinary command.
        with pytest.raises(TaskSourceConflictError, match="dedicated authority"):
            adapter.recover_post_merge_retry(**adapter_arguments)
        assert _snapshot(owner, task_cid) == before

        legacy_id = CLIENT_ID + ":previous-command-profile"
        legacy_token, legacy_grant = owner.server.issue_typed_client_grant_record(
            client_id=legacy_id,
            process_birth_id=owner.identity.process_birth_id,
            allowed_operations=daemon_required_owner_operations(),
            allowed_command_operations=tuple(
                operation
                for operation in daemon_required_owner_command_operations()
                if operation != COMMAND
            ),
            peer_pid=os.getpid(),
        )
        with monkeypatch.context() as previous_profile:
            previous_profile.setenv(TYPED_STATE_OWNER_TOKEN_ENV, legacy_token)
            legacy_client = QuackStateClient(
                owner_id=legacy_id, store_id=owner.identity.store_id,
                process_birth_id=owner.identity.process_birth_id,
            )
            try:
                legacy_client.attach(owner.identity.listen_uri, server_id=owner.identity.server_id)
                with pytest.raises((TypedStateOwnerError, QuackClientError, TransactionError)):
                    legacy_client.recover_exhausted_post_merge_retry(**arguments)
            finally:
                legacy_client.close()
                owner.server.revoke_typed_client_grant(legacy_grant.grant_id)
        assert _snapshot(owner, task_cid) == before

        gateway = owner.server._command_gateway
        execute = gateway._execute
        executed: list[str] = []

        def fail_history(operation: Any, parameters: list[Any]) -> Any:
            executed.append(operation.name)
            if operation.name == "executor_insert_task_revision_history":
                raise RuntimeError("injected exhausted recovery history failure")
            return execute(operation, parameters)

        with monkeypatch.context() as fault:
            fault.setattr(gateway, "_execute", fail_history)
            with pytest.raises(TransactionError, match="operation_failed"):
                client.recover_exhausted_post_merge_retry(**arguments)
        assert "executor_update_retry_cooldown" in executed
        assert "executor_cas_task_status_receipt" in executed
        assert "executor_insert_task_revision_history" in executed
        assert _snapshot(owner, task_cid) == before

        recover = client.recover_exhausted_post_merge_retry
        accepted: list[Any] = []

        def lose_committed_response(**kwargs: Any) -> Any:
            result = recover(**kwargs)
            assert result.accepted and result.changed
            accepted.append(result)
            raise QuackClientError("injected exhausted recovery committed response loss")

        with monkeypatch.context() as fault:
            fault.setattr(client, "recover_exhausted_post_merge_retry", lose_committed_response)
            recovered_response = adapter.recover_exhausted_post_merge_retry(**adapter_arguments)
        assert recovered_response["cas_result"].changed is False
        assert len(accepted) == 1
        recovered = adapter.get(task_cid)
        assert recovered is not None and recovered.status == "retrying" and recovered.revision == 12
        assert adapter.get_queue_entry(task_cid).attempt == 3
        receipt = recovered.body["completion_receipt"]
        seed = case.transition["post_merge_completion_recovery_seed"]
        assert receipt["post_merge_completion_recovery_seed"] == seed
        assert recovered.body == {**blocked.body, "completion_receipt": receipt}
        history = adapter.task_revision_history_projection(task_cid)
        assert history["revisions"][:-1] == case.history["revisions"]
        assert len(history["revisions"]) == 12
        committed = _snapshot(owner, task_cid)
        assert committed["generation"].revision == before["generation"].revision + 1

        def forbid_second_command(**_kwargs: Any) -> Any:
            raise AssertionError("durable adapter replay must not issue another command")

        with monkeypatch.context() as fault:
            fault.setattr(client, "recover_exhausted_post_merge_retry", forbid_second_command)
            replay = adapter.recover_exhausted_post_merge_retry(**adapter_arguments)
        assert replay["cas_result"].changed is False
        assert replay["transition_receipt"] == receipt
        assert _snapshot(owner, task_cid) == committed

        # Admission is real and fenced; no provider/effect callback is invoked
        # while obtaining the successor claim and carrying the retained seed.
        calls: list[str] = []
        daemon = DatabaseImplementationDaemon(
            database_path=case.database,
            coordination_path=tmp_path / "claim-coordination.duckdb",
            execution_path=tmp_path / "claim-execution.duckdb",
            owner_session_id="session:exhausted-post-merge-test",
            process_instance_id=owner.identity.process_birth_id,
            authority_mode="quack", task_source_kind="duckdb",
            quack_uri=owner.identity.listen_uri, task_source=adapter,
            close_task_source=False,
            state_owner_bootstrap_credentials=_typed_bootstrap_credentials(
                server=owner.server, identity=owner.identity, client_id=CLIENT_ID,
                token=owner.token, route_policy=case.policy,
            ),
            lease_ms=5_000, clock_ms=lambda: owner.clock["now_ms"],
            max_task_attempts=2,
            provider_fn=lambda _attempt: calls.append("provider"),
            effect_fn=lambda _attempt, _provider: calls.append("effect"),
            validation_fn=lambda _attempt, _effect: calls.append("validation"),
            require_real_execution=True,
        ).open()
        try:
            fresh = daemon.claim_next()
            assert fresh is not None and fresh.task_cid == task_cid
            assert fresh.attempt_number == 4
            assert fresh.attempt_id != case.transition["attempt_id"]
            claimed = adapter.get(task_cid)
            assert claimed is not None and claimed.status == "in_progress"
            assert claimed.revision == 14
            admission = claimed.body["completion_receipt"]
            assert admission["operation"] == "database_attempt_admitted"
            assert admission["post_merge_completion_recovery_seed"] == seed
            assert admission["post_merge_completion_recovery_source_attempt_id"] == seed["attempt_id"]
            assert admission["execution_route_binding"] == case.route
            assert calls == []
        finally:
            daemon.close()
        previous_identity = owner.identity
        previous_token = owner.token
        claimed_body = _plain(claimed.to_dict())

    with _owner(case, tmp_path, monkeypatch, state_name="owner-restarted") as owner:
        assert owner.identity.generation > previous_identity.generation
        # Even possession of the prior process's token must not authenticate a
        # connection to the new owner generation.
        with monkeypatch.context() as stale:
            stale.setenv(TYPED_STATE_OWNER_TOKEN_ENV, previous_token)
            stale_client = QuackStateClient(
                owner_id=CLIENT_ID, store_id=owner.identity.store_id,
                process_birth_id=previous_identity.process_birth_id,
            )
            try:
                with pytest.raises((QuackClientError, TypedStateOwnerError)):
                    stale_client.attach(owner.identity.listen_uri, server_id=owner.identity.server_id)
            finally:
                stale_client.close()
        before = _snapshot(owner, task_cid)
        replay = owner.client.recover_exhausted_post_merge_retry(**arguments)
        assert replay.outcome is CommandOutcome.IDEMPOTENT_REPLAY
        assert replay.changed is False
        assert _snapshot(owner, task_cid) == before
        assert _plain(owner.adapter.get(task_cid).to_dict()) == claimed_body
