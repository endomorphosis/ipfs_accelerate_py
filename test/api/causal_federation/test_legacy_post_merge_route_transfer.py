"""Regression for the one legacy post-merge route/transfer omission."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_JSON_ENV,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    TaskSourceTransitionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    DATABASE_CLAIM_POLICY_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
    StateOwnerBootstrapCredentials,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    DETERMINISTIC_ONLY_EXECUTION_MODE,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    validated_post_merge_retry_predecessor_lineage,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability,
    _migrate,
)


_LINEAGE_FIELDS = (
    "execution_route_binding",
    "execution_route_policy_id",
    "execution_route_origin_revision",
    "virgin_task_transfer",
    "virgin_task_transfer_claim_cursor",
)


def _credentials(
    *,
    server: Any,
    identity: Any,
    client_id: str,
    token: str,
    route_policy: Any,
) -> StateOwnerBootstrapCredentials:
    return StateOwnerBootstrapCredentials(
        endpoint=identity.listen_uri,
        socket_path=str(server.typed_command_socket_path()),
        store_id=identity.store_id,
        server_id=identity.server_id,
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        token=token,
        execution_route_policy=route_policy,
    )


def test_legacy_post_merge_omission_retains_exact_route_and_transfer_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exact recipient can claim a proven legacy recovery row."""

    database = tmp_path / "legacy-post-merge-route-transfer.duckdb"
    task_cid = "task:spar-004"
    task_alias = "SPAR-004"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:legacy-post-merge-route-transfer",
            "plan_root_cid": "plan:legacy-post-merge-route-transfer",
            "goals": [
                {
                    "goal_cid": "goal:legacy-post-merge-route-transfer",
                    "goal_alias": "SPAR-G004",
                    "title": "Legacy post-merge route transfer",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:spar-003",
                    "task_id": "SPAR-003",
                    "goal_cid": "goal:legacy-post-merge-route-transfer",
                    "status": "ready",
                    "priority": "P0",
                    "ordinal": 0,
                },
                {
                    "task_cid": task_cid,
                    "task_id": task_alias,
                    "goal_cid": "goal:legacy-post-merge-route-transfer",
                    "status": "ready",
                    "priority": "P1",
                    "ordinal": 1,
                },
            ],
        }
    )
    source.close()

    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(
            {
                "claim_policy": {
                    "schema": DATABASE_CLAIM_POLICY_SCHEMA,
                    "task_prefix": "SPAR-",
                    "task_shard_count": 3,
                    "strict_task_sharding": True,
                    "idle_lane_work_stealing": "virgin-transfer",
                }
            }
        ),
    )
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "legacy-post-merge-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="spar-legacy-post-merge-route-transfer-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
    )
    client_id = "database-implementation-daemon:legacy-post-merge-route"
    provider_calls: list[str] = []
    clock = {"now_ms": 1_000}
    route_policy: Any = None
    initial_cursor: dict[str, Any] = {}

    identity = server.start()
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=daemon_required_owner_operations(),
        allowed_command_operations=daemon_required_owner_command_operations(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV,
        str(server.typed_command_socket_path()),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
    )
    adapter: TypedDatabaseTaskSource | None = None
    daemon: DatabaseImplementationDaemon | None = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        route_policy = unsealed.seal_execution_route_policy(
            {
                "SPAR-003": DETERMINISTIC_ONLY_EXECUTION_MODE,
                task_alias: DETERMINISTIC_ONLY_EXECUTION_MODE,
            }
        )
        unsealed.close()
        adapter = TypedDatabaseTaskSource(
            client,
            execution_route_policy=route_policy,
            clock_ms=lambda: clock["now_ms"],
        )
        daemon = DatabaseImplementationDaemon(
            database_path=database,
            coordination_path=tmp_path / "lane-0-coordination.duckdb",
            execution_path=tmp_path / "lane-0-execution.duckdb",
            owner_session_id="session:spar-lane:0",
            process_instance_id=identity.process_birth_id,
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri=identity.listen_uri,
            task_source=adapter,
            close_task_source=False,
            state_owner_bootstrap_credentials=_credentials(
                server=server,
                identity=identity,
                client_id=client_id,
                token=token,
                route_policy=route_policy,
            ),
            lease_ms=5_000,
            clock_ms=lambda: clock["now_ms"],
            task_shard_count=3,
            task_shard_index=0,
            strict_task_sharding=True,
            idle_lane_work_stealing="virgin-transfer",
            task_prefix="SPAR-",
            provider_fn=lambda attempt: provider_calls.append(
                attempt.attempt_id
            )
            or {"status": "ok", "accepted": True},
            effect_fn=lambda _attempt, _provider: {"status": "applied"},
            validation_fn=lambda _attempt, _effect: {
                "outcome": "passed",
                "evidence_digest": "sha256:" + "a" * 64,
            },
            require_real_execution=True,
        ).open()
        initial_attempt = daemon.claim_next()
        assert initial_attempt is not None
        assert initial_attempt.task_alias == task_alias
        claimed = adapter.get(task_cid)
        assert claimed is not None and claimed.status == "in_progress"
        claimed_receipt = claimed.body["completion_receipt"]
        assert set(_LINEAGE_FIELDS).issubset(claimed_receipt)
        binding = claimed_receipt["virgin_task_transfer"]
        assert binding["home_shard_index"] == 1
        assert binding["recipient_shard_index"] == 0
        initial_cursor = dict(
            claimed_receipt["virgin_task_transfer_claim_cursor"]
        )
        terminal_reason = "Portal completion lacks one exact implementation commit"
        failed_attempt = daemon.commit_phase(
            initial_attempt,
            "failed",
            body={"reason": terminal_reason},
        )
        assert failed_attempt.status == "failed"
        assert failed_attempt.finished_at_ms is not None
        terminal = {
            "operation": "database_portal_terminal_failure",
            "attempt_id": failed_attempt.attempt_id,
            "attempt_number": failed_attempt.attempt_number,
            "claim_id": failed_attempt.claim_id,
            "lease_id": failed_attempt.lease_id,
            "owner_session_id": failed_attempt.owner_session_id,
            "fencing_token": failed_attempt.fencing_token,
            "fence_epoch": failed_attempt.fence_epoch,
            "execution_phase": "failed",
            "execution_revision": failed_attempt.revision,
            "execution_finished_at_ms": failed_attempt.finished_at_ms,
            "reason": terminal_reason,
            "retryable": False,
            "coordination": {
                "attempt_id": failed_attempt.attempt_id,
                "claim_id": failed_attempt.claim_id,
                "attempt_number": failed_attempt.attempt_number,
            },
            "control_expected_status": "in_progress",
            "control_expected_revision": claimed.revision,
        }
        daemon._cas_task_status_database(
            task_cid,
            expected_revision=claimed.revision,
            new_status="blocked",
            receipt=terminal,
        )
        blocked = adapter.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"
        blocked_receipt = blocked.body["completion_receipt"]
        assert set(_LINEAGE_FIELDS).issubset(blocked_receipt)

        request_id = "request:legacy-spar-004"
        repair_receipt_id = "receipt:legacy-spar-004"
        repair_evidence_id = "sha256:" + "e" * 64
        repair_commit = "b" * 40
        candidate_commit = "c" * 40
        queue_reason = (
            "database_post_merge_declared_outputs_repair:"
            + request_id
            + ":"
            + repair_receipt_id
        )
        seed = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-post-merge-completion-recovery-seed@1"
            ),
            "task_cid": task_cid,
            "task_alias": task_alias,
            "attempt_id": failed_attempt.attempt_id,
            "attempt_number": failed_attempt.attempt_number,
            "claim_id": failed_attempt.claim_id,
            "lease_id": failed_attempt.lease_id,
            "owner_session_id": failed_attempt.owner_session_id,
            "fencing_token": failed_attempt.fencing_token,
            "fence_epoch": failed_attempt.fence_epoch,
            "source_task_revision": blocked.revision,
            "request_id": request_id,
            "candidate_commit": candidate_commit,
            "qualified_target_commit": repair_commit,
            "qualification_kind": "repair",
            "qualification_receipt_id": repair_receipt_id,
            "queue_source_attempt_id": failed_attempt.attempt_id,
            "queue_source_claim_id": failed_attempt.claim_id,
            "queue_source_lease_id": failed_attempt.lease_id,
            "queue_source_fencing_token": failed_attempt.fencing_token,
            "queue_source_fence_epoch": failed_attempt.fence_epoch,
            "queue_source_binding_id": "sha256:" + "d" * 64,
            "queue_source_projection_immutable_digest": "sha256:" + "f" * 64,
            "recovery_evidence_id": repair_evidence_id,
            "terminal_reason": terminal_reason,
        }
        seed["seed_id"] = "sha256:" + hashlib.sha256(
            canonical_json_bytes(seed)
        ).hexdigest()
        transition = {
            "operation": "database_post_merge_declared_outputs_repair_recovery",
            "attempt_id": failed_attempt.attempt_id,
            "attempt_number": failed_attempt.attempt_number,
            "claim_id": failed_attempt.claim_id,
            "lease_id": failed_attempt.lease_id,
            "owner_session_id": failed_attempt.owner_session_id,
            "fencing_token": failed_attempt.fencing_token,
            "fence_epoch": failed_attempt.fence_epoch,
            "execution_phase": "failed",
            "execution_revision": failed_attempt.revision,
            "execution_finished_at_ms": failed_attempt.finished_at_ms,
            "request_id": request_id,
            "candidate_commit": candidate_commit,
            "source_binding_id": seed["queue_source_binding_id"],
            "source_projection_immutable_digest": seed[
                "queue_source_projection_immutable_digest"
            ],
            "queue_reason": queue_reason,
            "queue_receipt": {},
            "coordination": terminal["coordination"],
            "control_expected_status": "blocked",
            "control_expected_revision": blocked.revision,
            "repair_commit": repair_commit,
            "repair_receipt_id": repair_receipt_id,
            "repair_evidence_id": repair_evidence_id,
            "post_merge_completion_recovery_seed": seed,
            **{
                name: blocked_receipt[name]
                for name in _LINEAGE_FIELDS
            },
        }
        clock["now_ms"] = 2_000
        recovered = adapter.recover_post_merge_retry(
            task_cid=task_cid,
            expected_revision=blocked.revision,
            expected_control_receipt=blocked_receipt,
            status="retrying",
            receipt=transition,
            delay_ms=0,
            reason=queue_reason,
        )
        assert recovered["cas_result"].changed is True
        successor = adapter.get(task_cid)
        assert successor is not None and successor.status == "retrying"
        assert set(_LINEAGE_FIELDS).issubset(
            successor.body["completion_receipt"]
        )
    finally:
        if daemon is not None:
            daemon.close()
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()

    # Reproduce the historical persisted bytes while the exclusive owner is
    # stopped.  All other task, cooldown, and revision-history bytes remain
    # owner-produced and canonical.
    connection = open_duckdb_connection(database)
    try:
        row = connection.execute(
            "SELECT revision, body_json FROM tasks WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        assert row is not None
        successor_revision = int(row[0])
        successor_body = json.loads(str(row[1]))
        successor_receipt = dict(successor_body["completion_receipt"])
        for name in _LINEAGE_FIELDS:
            successor_receipt.pop(name)
        successor_body["completion_receipt"] = successor_receipt
        encoded = canonical_json_bytes(successor_body).decode("utf-8")
        connection.execute(
            "UPDATE tasks SET body_json = ? WHERE task_cid = ? AND revision = ?",
            [encoded, task_cid, successor_revision],
        )
        connection.execute(
            "UPDATE task_revisions SET body_json = ? "
            "WHERE task_cid = ? AND revision = ?",
            [encoded, task_cid, successor_revision],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    clock["now_ms"] = 7_000
    identity = server.start()
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=daemon_required_owner_operations(),
        allowed_command_operations=daemon_required_owner_command_operations(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV,
        str(server.typed_command_socket_path()),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
    )
    adapter = None
    daemons: list[DatabaseImplementationDaemon] = []
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        adapter = TypedDatabaseTaskSource(
            client,
            execution_route_policy=route_policy,
            clock_ms=lambda: clock["now_ms"],
        )
        legacy = adapter.get(task_cid)
        assert legacy is not None and legacy.status == "retrying"
        assert not set(_LINEAGE_FIELDS).intersection(
            legacy.body["completion_receipt"]
        )
        lineage = adapter.post_merge_retry_predecessor_lineage(legacy)
        assert set(lineage) == set(_LINEAGE_FIELDS)

        credentials = _credentials(
            server=server,
            identity=identity,
            client_id=client_id,
            token=token,
            route_policy=route_policy,
        )
        for lane_index in range(3):
            daemon = DatabaseImplementationDaemon(
                database_path=database,
                coordination_path=(
                    tmp_path / f"lane-{lane_index}-coordination.duckdb"
                ),
                execution_path=(
                    tmp_path / f"lane-{lane_index}-execution.duckdb"
                ),
                owner_session_id=f"session:spar-lane:{lane_index}",
                process_instance_id=identity.process_birth_id,
                authority_mode="quack",
                task_source_kind="duckdb",
                quack_uri=identity.listen_uri,
                task_source=adapter,
                close_task_source=False,
                state_owner_bootstrap_credentials=credentials,
                lease_ms=5_000,
                clock_ms=lambda: clock["now_ms"],
                task_shard_count=3,
                task_shard_index=lane_index,
                strict_task_sharding=True,
                idle_lane_work_stealing="virgin-transfer",
                task_prefix="SPAR-",
                provider_fn=lambda attempt: provider_calls.append(
                    attempt.attempt_id
                )
                or {"status": "ok", "accepted": True},
                effect_fn=lambda _attempt, _provider: {"status": "applied"},
                validation_fn=lambda _attempt, _effect: {
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                },
                require_real_execution=True,
            ).open()
            daemons.append(daemon)

        routed = [
            task_cid in daemon.sync_ready_tasks_into_coordination()
            for daemon in daemons
        ]
        assert routed == [True, False, False]

        original_cas = adapter.compare_and_set_status
        tested_rejections = {"done": False}

        def reject_legacy_claim_variants_then_apply_exact(
            task_cid_or_alias: Any,
            expected_revision: int,
            status: str,
            receipt: Mapping[str, Any] | None = None,
            *,
            evidence_digests: Any = None,
        ) -> Any:
            assert receipt is not None
            canonical = dict(receipt)
            if tested_rejections["done"]:
                return original_cas(
                    task_cid_or_alias,
                    expected_revision,
                    status,
                    canonical,
                    evidence_digests=evidence_digests,
                )
            assert set(_LINEAGE_FIELDS).issubset(canonical)
            assert canonical["task_shard_index"] == 0
            before = adapter.get(task_cid)
            assert before is not None
            generation = client.load_generation()
            variants: list[dict[str, Any]] = []
            variants.append(
                {
                    key: value
                    for key, value in canonical.items()
                    if key not in _LINEAGE_FIELDS
                }
            )
            partial = dict(canonical)
            partial.pop("virgin_task_transfer_claim_cursor")
            variants.append(partial)
            forged = dict(canonical)
            forged["execution_route_policy_id"] = "policy:forged"
            variants.append(forged)
            home_lane = dict(canonical)
            home_lane["task_shard_index"] = 1
            home_lane["owner_session_id"] = "session:spar-lane:1"
            home_lane["virgin_task_transfer_request"] = {
                **home_lane["virgin_task_transfer_request"],
                "recipient_shard_index": 1,
            }
            variants.append(home_lane)
            for candidate in variants:
                with pytest.raises(
                    (
                        QuackClientError,
                        TaskSourceConflictError,
                        TaskSourceIntegrityError,
                        TaskSourceTransitionError,
                        TransactionError,
                    )
                ):
                    original_cas(
                        task_cid_or_alias,
                        expected_revision,
                        status,
                        candidate,
                        evidence_digests=evidence_digests,
                    )
                assert adapter.get(task_cid) == before
                assert client.load_generation().content_id == generation.content_id
                assert provider_calls == []
            tested_rejections["done"] = True
            return original_cas(
                task_cid_or_alias,
                expected_revision,
                status,
                canonical,
                evidence_digests=evidence_digests,
            )

        monkeypatch.setattr(
            adapter,
            "compare_and_set_status",
            reject_legacy_claim_variants_then_apply_exact,
        )
        attempt = daemons[0].claim_next()
        assert tested_rejections["done"] is True
        assert attempt is not None and attempt.task_cid == task_cid
        assert provider_calls == []
        claimed = adapter.get(task_cid)
        assert claimed is not None and claimed.status == "in_progress"
        claimed_receipt = claimed.body["completion_receipt"]
        assert claimed_receipt["task_shard_index"] == 0
        assert claimed_receipt["virgin_task_transfer"] == (
            lineage["virgin_task_transfer"]
        )
        rotated_cursor = claimed_receipt[
            "virgin_task_transfer_claim_cursor"
        ]
        assert rotated_cursor["binding_id"] == initial_cursor["binding_id"]
        assert rotated_cursor["cursor_id"] != initial_cursor["cursor_id"]
        assert rotated_cursor["claimed_from_revision"] == successor_revision
        assert rotated_cursor["fencing_token"] > initial_cursor["fencing_token"]

        # Reproduce the cross-lane TOCTOU window: ``legacy`` came from the
        # stable ready projection, while the exact history now includes the
        # successful claim as a later canonical revision.  The stale snapshot
        # may recover its read-only route/transfer prefix, but malformed
        # successor numbering and a forged snapshot prefix still fail closed.
        advanced_history = adapter.task_revision_history_projection(task_cid)
        advanced_revisions = advanced_history["revisions"]
        assert len(advanced_revisions) > legacy.revision
        assert dict(
            adapter.post_merge_retry_predecessor_lineage(legacy)
        ) == dict(lineage)
        stale_lineage = validated_post_merge_retry_predecessor_lineage(
            legacy,
            advanced_revisions,
        )
        assert dict(stale_lineage) == dict(lineage)

        malformed_successor = json.loads(json.dumps(advanced_revisions))
        malformed_successor[-1]["revision"] += 1
        with pytest.raises(
            TaskSourceIntegrityError,
            match="history is incomplete or noncanonical",
        ):
            validated_post_merge_retry_predecessor_lineage(
                legacy,
                malformed_successor,
            )

        forged_prefix = json.loads(json.dumps(advanced_revisions))
        forged_prefix[legacy.revision - 1]["body"]["fixture_tamper"] = True
        with pytest.raises(
            TaskSourceIntegrityError,
            match="lacks its exact predecessor/head",
        ):
            validated_post_merge_retry_predecessor_lineage(
                legacy,
                forged_prefix,
            )
    finally:
        for daemon in reversed(daemons):
            daemon.close()
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()
