"""Tests for IntentRepository@1 / DatabaseTaskSource@1 / PlanRevisionRepository@1.

DQP-012 acceptance:

* No cross-file saga is needed (single DB transactions + domain events)
* Completion cannot be selected without current required evidence
* Existing public task/plan/objective APIs retain canonical identities
* Database rebuild from admitted events matches current projections

Evidence subset: CAS heads, supersession, continuation, recovery, dependency
readiness, queue retry, goal reopen, current evidence.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DATABASE_TASK_SOURCE_INTERFACE,
    DATABASE_TASK_SOURCE_SCHEMA,
    MAX_QUERY_LIMIT,
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceCompletionError,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_quack_transport_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    FENCED_PROVIDER_OUTER_AUTHORITY_NONCLAIMS,
    FENCED_PROVIDER_OUTER_HISTORICAL_AUTHORITY_POPULATION_RECEIPT_SCHEMA,
    INTENT_REPOSITORY_INTERFACE,
    MAX_FENCED_PROVIDER_OUTER_DERIVED_IDS,
    MAX_FENCED_PROVIDER_OUTER_POPULATION_ROWS,
    PLAN_REVISION_REPOSITORY_INTERFACE,
    IntentCompletionError,
    IntentEventType,
    IntentRepository,
    IntentRepositoryBoundsError,
    IntentRepositoryConflictError,
    IntentRepositoryIntegrityError,
    PlanRevisionRepository,
    _fenced_provider_outer_bound_parent_projection_counts,
    _fenced_provider_outer_group,
    _fenced_provider_outer_groups_semantically_valid,
    _fenced_provider_outer_parent_identity_projection,
    _fenced_provider_outer_population_filter,
    _fenced_provider_outer_query_profile,
    _fenced_provider_outer_sha256,
    fenced_provider_outer_authority_population_receipt_valid,
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.retained_recovery_contracts import (
    database_fenced_provider_historical_occurrence_authority,
    database_portal_controller_quiescence_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    _PinnedFencedIntentRepository,
    _PinnedReadIntentRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _admitted_observation,
    _isolation_receipt,
    _isolation_server_kwargs,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for intent-repository hermetic tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path) -> IntentRepository:
    return open_intent_repository(tmp_path / "control.duckdb", owner_id="owner:test")


def _seed_graph(repo: IntentRepository) -> dict[str, str]:
    repo.upsert_objective(
        objective_id="objective:dqp-012",
        objective_alias="DQP-O012",
        title="Migrate intent state",
        priority="P0",
        body={"track": "intent-repository"},
    )
    repo.upsert_goal(
        goal_cid="goal:cid:root",
        goal_alias="DQP-G020",
        title="Intent authority",
        objective_id="objective:dqp-012",
        ordinal=1,
    )
    repo.upsert_goal(
        goal_cid="goal:cid:child",
        goal_alias="DQP-G020-A",
        title="Child goal",
        objective_id="objective:dqp-012",
        parent_goal_cid="goal:cid:root",
        ordinal=2,
    )
    repo.link_goal_edge(
        parent_goal_cid="goal:cid:root",
        child_goal_cid="goal:cid:child",
        edge_kind="depends_on",
    )
    repo.upsert_plan(
        plan_cid="plan:cid:v1",
        goal_cid="goal:cid:root",
        plan_alias="plan-v1",
        status="active",
        body={"steps": ["seed", "migrate"]},
    )
    repo.upsert_task(
        task_cid="task:cid:001",
        task_alias="DQP-012-A",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=1,
        status="ready",
        priority="P0",
        acceptance=[
            {
                "criterion": "tests pass",
                "required_digest": "sha256:" + ("ab" * 32),
                "evidence_kind": "validation",
            }
        ],
        validations=[["python", "-m", "pytest", "-q"]],
        outputs=[{"path": "intent_repository.py", "effect": "create"}],
    )
    repo.upsert_task(
        task_cid="task:cid:002",
        task_alias="DQP-012-B",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=2,
        status="ready",
        dependencies=["task:cid:001"],
        acceptance=[{"criterion": "rebuild matches", "evidence_kind": "validation"}],
        validations=["pytest test_rebuild.py"],
    )
    return {
        "objective_id": "objective:dqp-012",
        "goal_cid": "goal:cid:root",
        "plan_cid": "plan:cid:v1",
        "task_a": "task:cid:001",
        "task_b": "task:cid:002",
        "evidence_digest": "sha256:" + ("ab" * 32),
    }


def _outer_receipt_binding(connection) -> dict[str, object]:
    fingerprint = connection.execute(
        "SELECT value FROM control_plane_metadata "
        "WHERE key = 'schema_fingerprint'"
    ).fetchone()
    assert fingerprint is not None
    return {
        "server_id": "server:outer-receipt-test",
        "store_id": "state/control.duckdb",
        "database_uuid": "database:outer-receipt-test",
        "schema_revision": 1,
        "schema_fingerprint": str(fingerprint[0]),
        "generation": 7,
        "process_birth_id": "birth:outer-receipt-test",
        "listen_uri": "quack:127.0.0.1:4242",
        "extension_fingerprint": "sha256:" + ("12" * 32),
    }


def _seed_outer_receipt_subject(repo: IntentRepository) -> dict[str, object]:
    ids = _seed_graph(repo)
    task = repo.get_task(ids["task_a"])
    assert task is not None
    repo.cas_task_status(
        task_cid=ids["task_a"],
        expected_revision=int(task["revision"]),
        new_status="blocked",
        receipt={"reason": "outer-receipt-fixture"},
    )
    task = repo.get_task(ids["task_a"])
    assert task is not None
    subject: dict[str, object] = {
        "task_cid": ids["task_a"],
        "task_alias": "DQP-012-A",
        "task_revision": int(task["revision"]),
        "expected_task_status": "blocked",
        "attempt_id": "attempt:outer-receipt-test",
        "claim_id": "claim:outer-receipt-test",
        "lease_id": "lease:outer-receipt-test",
        "owner_session_id": "owner:outer-receipt-test",
        "fencing_token": 11,
        "fence_epoch": 3,
        "expected_store_id": "state/control.duckdb",
        "expected_store_generation": 7,
        "receipt_nonce": "nonce:outer-receipt-test",
        "receipt_epoch": 1,
    }
    with repo._connection(write=True) as connection:
        connection.execute(
            """
            INSERT INTO task_attempts (
                attempt_id, task_cid, attempt_number, owner_session_id,
                fencing_token, fence_epoch, started_at, finished_at,
                status, revision
            ) VALUES (?, ?, 1, ?, ?, ?, ?, NULL, 'blocked', 1)
            """,
            [
                subject["attempt_id"], subject["task_cid"],
                subject["owner_session_id"], subject["fencing_token"],
                subject["fence_epoch"], "2026-09-02T00:01:00Z",
            ],
        )
        connection.execute(
            """
            INSERT INTO task_claims (
                claim_id, task_cid, owner_session_id, fencing_token,
                fence_epoch, claimed_at, expires_at, released_at,
                state, revision, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'released', 1, ?)
            """,
            [
                subject["claim_id"], subject["task_cid"],
                subject["owner_session_id"], subject["fencing_token"],
                subject["fence_epoch"], "2026-09-02T00:01:00Z",
                "2026-09-02T00:02:00Z", "idempotency:outer-receipt-test",
            ],
        )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt, state,
                started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision,
                extension_schema, extension_json
            ) VALUES (?, ?, '', ?, 1, ?, 0, 1, 'released', 1, 'blocked', 0,
                      ?, ?, 1, '', '{}')
            """,
            [
                subject["task_cid"], subject["claim_id"],
                subject["owner_session_id"], subject["fencing_token"],
                subject["owner_session_id"], subject["fence_epoch"],
            ],
        )
    return subject


def _seed_outer_receipt_authority(
    repo: IntentRepository,
) -> tuple[dict[str, object], dict[str, object]]:
    subject = _seed_outer_receipt_subject(repo)
    with repo._connection(write=True) as connection:
        binding = _outer_receipt_binding(connection)
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at,
                extension_schema, extension_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                7, 1, 3, 1, binding["database_uuid"],
                binding["process_birth_id"], "2026-09-02T00:00:00Z", "", "{}",
            ],
        )
        connection.execute(
            """
            INSERT INTO state_servers (
                server_id, store_id, database_uuid, process_birth_id,
                listen_uri, extension_fingerprint, schema_revision,
                generation, started_at, stopped_at, status, revision,
                extension_schema, extension_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'ready', 1, '', '{}')
            """,
            [
                binding["server_id"], binding["store_id"],
                binding["database_uuid"], binding["process_birth_id"],
                binding["listen_uri"], binding["extension_fingerprint"],
                binding["schema_revision"], binding["generation"],
                "2026-09-02T00:00:00Z",
            ],
        )
    return binding, subject


def _read_outer_receipt(
    repo: IntentRepository,
    binding: dict[str, object],
    subject: dict[str, object],
) -> dict[str, object]:
    replica_observation = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1"
        ),
        "authority": "non_authoritative_read_replica",
        "path": "/sealed/test/read-replica.duckdb",
        "source_database_path": "/sealed/test/control.duckdb",
        "server_id": binding["server_id"],
        "database_uuid": binding["database_uuid"],
        "generation": binding["generation"],
        "schema_revision": binding["schema_revision"],
        "schema_fingerprint": "schema-profile:test",
        "storage_schema_fingerprint": binding["schema_fingerprint"],
        "sha256": "sha256:" + ("34" * 32),
        "size_bytes": 4096,
        "refresh_sequence": 9,
        "refreshed_at_ms": 1_700_000_000_000,
        "live": True,
    }
    mutation_barrier = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "fenced-provider-outer-mutation-barrier@1"
        ),
        "store_id": binding["store_id"],
        "active_request_count": 0,
        "active_processing_count": 0,
        "active_population_digest": (
            "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba8"
            "73c2f11161202b945"
        ),
    }
    with repo._connection(write=False) as connection:
        connection._quack_mutation_binding = dict(binding)
        connection.execute("BEGIN TRANSACTION")
        try:
            source = DatabaseTaskSource(
                intent=_PinnedReadIntentRepository(connection),
                owner_id="outer-receipt-test",
            )
            receipt = dict(
                source.fenced_provider_outer_authority_population_receipt(
                    controller_replica_observation=replica_observation,
                    controller_mutation_barrier=mutation_barrier,
                    **subject
                )
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return receipt


def _historical_authority_for_subject(
    subject: dict[str, object],
) -> dict[str, object]:
    controller = database_portal_controller_quiescence_receipt(
        cleanup={
            "pid": 9001,
            "managed_daemon_identity_record_id": "identity:outer:9001",
            "managed_daemon_process_birth": {
                "pid": 9001,
                "start_time_ticks": 91,
                "boot_id": "boot:outer",
                "parent_pid": 9000,
            },
            "quiesced": True,
            "remaining_pid": None,
            "markers_removed": True,
            "daemon_fence": {
                "fenced": True,
                "safe_to_restart": True,
                "reason": "managed_daemon_owned_process_fenced",
            },
            "provider_runner_fence": {
                "applicable": False,
                "fenced": False,
                "safe_to_restart": True,
                "reason": "ordinary_provider_runner_receipt_absent",
            },
        },
        board_namespace="parallel-content-sealing-proof-carrying-tdd-v1",
        state_prefix="outer",
        owner_store_id=str(subject["expected_store_id"]),
        control_store_generation="pctdd-v1-g9",
        trigger="supervisor_startup_prelaunch",
        controller_process_birth={
            "pid": 9000,
            "start_time_ticks": 81,
            "boot_id": "boot:outer",
            "parent_pid": 1,
        },
        owner_mutation_fence_held=True,
        managed_daemon_launch_lock_held=True,
    )
    inner_subject = {
        "task_cid": subject["task_cid"],
        "task_alias": subject["task_alias"],
        "task_revision": subject["task_revision"],
        "attempt_id": subject["attempt_id"],
        "claim_id": subject["claim_id"],
        "lease_id": subject["lease_id"],
        "attempt_number": 1,
        "owner_session_id": subject["owner_session_id"],
        "fencing_token": subject["fencing_token"],
        "fence_epoch": subject["fence_epoch"],
        "recovery_manifest_id": "sha256:" + "1" * 64,
        "recovery_credit_id": "sha256:" + "2" * 64,
    }
    terminal = {
        "attempt_id": subject["attempt_id"],
        "task_cid": subject["task_cid"],
        "claim_id": subject["claim_id"],
        "attempt_number": 1,
        "owner_session_id": subject["owner_session_id"],
        "lease_id": subject["lease_id"],
        "fencing_token": subject["fencing_token"],
        "fence_epoch": subject["fence_epoch"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "evidence_id": "baguqeera" + "a" * 52,
        "prepared_reconciliation_receipt_id": "sha256:" + "3" * 64,
        "commit_barrier_receipt_id": "sha256:" + "4" * 64,
        "stage": "terminal",
        "receipt_id": "sha256:" + "5" * 64,
        "record_json": {
            "canonical_sha256": "sha256:" + "6" * 64,
            "canonical_byte_length": 512,
        },
    }
    return dict(
        database_fenced_provider_historical_occurrence_authority(
            inner_receipt={
                "receipt_nonce": subject["receipt_nonce"],
                "receipt_epoch": subject["receipt_epoch"],
                "receipt_cid": "sha256:" + "7" * 64,
                "subject": inner_subject,
                "groups": {
                    "database_portal_terminal_reconciliations": {
                        "count": 1,
                        "rows": [terminal],
                    }
                },
            },
            controller_quiescence_receipt=controller,
            board_namespace=(
                "parallel-content-sealing-proof-carrying-tdd-v1"
            ),
            owner_store_id=str(subject["expected_store_id"]),
            control_store_generation="pctdd-v1-g9",
        )
    )


def _seed_outer_receipt_relations(
    repo: IntentRepository,
    subject: dict[str, object],
) -> None:
    with repo._connection(write=True) as connection:
        connection.execute(
            """
            INSERT INTO context_manifests (
                manifest_cid, task_cid, schema_revision,
                repository_tree_id, policy_digest, created_at, body_json
            ) VALUES ('manifest:outer', ?, 1, 'tree:outer',
                      'sha256:policy', ?, '{}')
            """,
            [subject["task_cid"], "2026-09-02T00:03:00Z"],
        )
        connection.execute(
            """
            INSERT INTO prompt_instances (
                instance_id, template_id, task_cid, manifest_cid,
                created_at, input_digest, body_json
            ) VALUES ('prompt:outer', 'template:outer', ?,
                      'manifest:outer', ?, 'sha256:input', '{}')
            """,
            [subject["task_cid"], "2026-09-02T00:04:00Z"],
        )
        connection.execute(
            """
            INSERT INTO provider_calls (
                call_id, task_cid, attempt_id, provider_id,
                prompt_instance_id, started_at, finished_at, status,
                input_digest, output_digest, tokens_in, tokens_out, body_json
            ) VALUES ('call:outer', ?, ?, 'provider:outer', 'prompt:outer',
                      ?, NULL, 'started', 'sha256:input', '', 0, 0, '{}')
            """,
            [
                subject["task_cid"],
                subject["attempt_id"],
                "2026-09-02T00:05:00Z",
            ],
        )
        connection.execute(
            """
            INSERT INTO worktrees (
                worktree_id, repository_id, path, head_commit_id,
                branch_name, owner_session_id, status, created_at,
                updated_at, revision, fence_epoch,
                extension_schema, extension_json
            ) VALUES ('worktree:outer', 'repository:outer', '/sealed/outer',
                      'commit:outer', 'implementation/outer', ?, 'retained',
                      ?, ?, 1, ?, '', '{}')
            """,
            [
                subject["owner_session_id"],
                "2026-09-02T00:06:00Z",
                "2026-09-02T00:06:00Z",
                subject["fence_epoch"],
            ],
        )
        connection.execute(
            """
            INSERT INTO merge_queue_entries (
                entry_id, repository_id, worktree_id, task_cid,
                source_branch, target_branch, status, ordinal,
                enqueued_at, updated_at, revision, fence_epoch
            ) VALUES ('entry:outer-relations', 'repository:outer',
                      'worktree:outer', ?, 'implementation/outer', 'main',
                      'pending', 1, ?, ?, 1, ?)
            """,
            [
                subject["task_cid"],
                "2026-09-02T00:07:00Z",
                "2026-09-02T00:07:00Z",
                subject["fence_epoch"],
            ],
        )
        connection.execute(
            """
            INSERT INTO merge_attempts (
                merge_attempt_id, entry_id, task_cid, worktree_id,
                started_at, finished_at, status, result_commit_id, body_json
            ) VALUES ('merge-attempt:outer-relations',
                      'entry:outer-relations', ?, 'worktree:outer', ?, NULL,
                      'running', '', '{}')
            """,
            [subject["task_cid"], "2026-09-02T00:08:00Z"],
        )
        connection.execute(
            """
            INSERT INTO path_claims (
                claim_id, repository_id, worktree_id, path,
                owner_session_id, task_cid, fencing_token, fence_epoch,
                acquired_at, expires_at, state, revision
            ) VALUES ('path-claim:outer', 'repository:outer',
                      'worktree:outer', 'src/outer.py', ?, ?, ?, ?, ?, ?,
                      'released', 1)
            """,
            [
                subject["owner_session_id"],
                subject["task_cid"],
                subject["fencing_token"],
                subject["fence_epoch"],
                "2026-09-02T00:09:00Z",
                "2026-09-02T00:10:00Z",
            ],
        )
        connection.execute(
            """
            INSERT INTO lease_events (
                event_id, task_cid, claim_cid, event_type,
                fencing_token, observed_at_ms, body_json
            ) VALUES ('lease-event:outer', ?, ?, 'released', ?, 10, '{}')
            """,
            [
                subject["task_cid"],
                subject["claim_id"],
                subject["fencing_token"],
            ],
        )


def _rehash_outer_receipt_group(
    receipt: dict[str, object],
    group_name: str,
) -> None:
    group = receipt["groups"][group_name]
    group["count"] = len(group["rows"])
    group["rows_digest"] = _fenced_provider_outer_sha256(group["rows"])
    receipt["task_population_root"] = _fenced_provider_outer_sha256(
        receipt["groups"]
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_cid")
    receipt["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert INTENT_REPOSITORY_INTERFACE == "IntentRepository@1"
    assert PLAN_REVISION_REPOSITORY_INTERFACE == "PlanRevisionRepository@1"
    assert DATABASE_TASK_SOURCE_INTERFACE == "DatabaseTaskSource@1"
    assert IntentRepository.INTERFACE == INTENT_REPOSITORY_INTERFACE
    assert PlanRevisionRepository.INTERFACE == PLAN_REVISION_REPOSITORY_INTERFACE
    assert DatabaseTaskSource.INTERFACE == DATABASE_TASK_SOURCE_INTERFACE


def test_outer_authority_receipt_is_closed_exact_and_commits_json_payloads(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        first = _read_outer_receipt(repo, binding, subject)
        second = _read_outer_receipt(repo, binding, subject)

        assert first == second
        assert fenced_provider_outer_authority_population_receipt_valid(first)
        assert first["nonclaims"] == list(FENCED_PROVIDER_OUTER_AUTHORITY_NONCLAIMS)
        assert first["persistence_policy"] == (
            "ephemeral_access_controlled_full_receipt_compact_cid_only"
        )
        assert first["publication_assessment"] == {
            "accepted_publication_count": 0,
            "terminal_task_status_count": 0,
            "completion_receipt_count": 0,
            "successful_merge_queue_count": 0,
            "successful_merge_attempt_count": 0,
        }
        assert first["authority"]["owner_binding"] == binding
        assert first["subject"] == {
            key: value
            for key, value in subject.items()
            if key not in {"lease_id", "receipt_nonce", "receipt_epoch"}
        }
        assert first["cross_store_context"] == {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "fenced-provider-outer-cross-store-context@1"
            ),
            "coordination_lease_id": subject["lease_id"],
            "admission_requirement": (
                "must_equal_corrected_current_inner_receipt_before_admission"
            ),
        }
        task_row = first["groups"]["tasks"]["rows"][0]
        assert set(task_row["body_json"]) == {
            "canonical_sha256",
            "canonical_byte_length",
        }
        assert "track" not in str(first)

        with pytest.raises(Exception, match="retained controller"):
            repo.fenced_provider_outer_authority_population_receipt(**subject)


def test_historical_outer_authority_requires_exact_central_absence_and_lane_evidence(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        normalized = _read_outer_receipt(repo, binding, subject)
        with repo._connection(write=True) as connection:
            connection.execute(
                "DELETE FROM leases WHERE task_cid = ?", [subject["task_cid"]]
            )
            connection.execute(
                "DELETE FROM task_claims WHERE task_cid = ?",
                [subject["task_cid"]],
            )
            connection.execute(
                "DELETE FROM task_attempts WHERE task_cid = ?",
                [subject["task_cid"]],
            )
        historical = _historical_authority_for_subject(subject)
        receipt = _read_outer_receipt(
            repo,
            binding,
            {**subject, "historical_occurrence_authority": historical},
        )

    assert receipt["schema"] == (
        FENCED_PROVIDER_OUTER_HISTORICAL_AUTHORITY_POPULATION_RECEIPT_SCHEMA
    )
    assert receipt["groups"]["task_attempts"]["count"] == 0
    assert receipt["groups"]["task_claims"]["count"] == 0
    assert receipt["groups"]["leases"]["count"] == 0
    assert receipt["cross_store_context"][
        "historical_occurrence_authority"
    ] == historical
    assert fenced_provider_outer_authority_population_receipt_valid(receipt)

    ambiguous = copy.deepcopy(receipt)
    ambiguous["groups"]["task_attempts"]["rows"] = copy.deepcopy(
        normalized["groups"]["task_attempts"]["rows"]
    )
    _rehash_outer_receipt_group(ambiguous, "task_attempts")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        ambiguous
    )

    wrong_lane = copy.deepcopy(receipt)
    wrong_lane["cross_store_context"]["historical_occurrence_authority"][
        "subject"
    ]["claim_id"] = "claim:foreign"
    historical_unsigned = dict(
        wrong_lane["cross_store_context"]["historical_occurrence_authority"]
    )
    historical_unsigned.pop("authority_id")
    wrong_lane["cross_store_context"]["historical_occurrence_authority"][
        "authority_id"
    ] = _fenced_provider_outer_sha256(historical_unsigned)
    unsigned = dict(wrong_lane)
    unsigned.pop("receipt_cid")
    wrong_lane["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        wrong_lane
    )


def test_historical_outer_authority_accepts_task_level_validation_and_rejects_orphan_attempt(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        repo.record_evidence(
            task_cid=str(subject["task_cid"]),
            evidence_kind="operator_diagnostic",
            digest="sha256:" + ("45" * 32),
            body={"scope": "task-level"},
        )
        repo.record_validation_result(
            task_cid=str(subject["task_cid"]),
            outcome="failed",
            evidence_digest="sha256:" + ("56" * 32),
            argv=("python", "-m", "pytest", "-q"),
            attempt_id="",
            body={"scope": "task-level"},
        )
        with repo._connection(write=True) as connection:
            connection.execute(
                "DELETE FROM leases WHERE task_cid = ?", [subject["task_cid"]]
            )
            connection.execute(
                "DELETE FROM task_claims WHERE task_cid = ?",
                [subject["task_cid"]],
            )
            connection.execute(
                "DELETE FROM task_attempts WHERE task_cid = ?",
                [subject["task_cid"]],
            )
        historical = _historical_authority_for_subject(subject)
        receipt = _read_outer_receipt(
            repo,
            binding,
            {**subject, "historical_occurrence_authority": historical},
        )

    assert receipt["groups"]["task_attempts"]["count"] == 0
    assert receipt["groups"]["validation_runs"]["count"] == 1
    assert receipt["groups"]["validation_runs"]["rows"][0]["attempt_id"] == ""
    assert receipt["groups"]["validation_runs"]["rows"][0]["status"] == "failed"
    assert receipt["groups"]["validation_results"]["count"] == 1
    assert receipt["groups"]["validation_results"]["rows"][0]["outcome"] == (
        "failed"
    )
    assert receipt["groups"]["evidence_nodes"]["count"] == 1
    validation_events = [
        row
        for row in receipt["groups"]["domain_events"]["rows"]
        if row["event_type"] == "intent.validation_recorded"
    ]
    assert len(validation_events) == 1
    assert validation_events[0]["attempt_id"] == ""
    assert fenced_provider_outer_authority_population_receipt_valid(receipt)

    orphaned = copy.deepcopy(receipt)
    orphaned["groups"]["validation_runs"]["rows"][0]["attempt_id"] = (
        "attempt:orphan"
    )
    _rehash_outer_receipt_group(orphaned, "validation_runs")
    assert not _fenced_provider_outer_groups_semantically_valid(
        orphaned["groups"],
        orphaned["subject"],
    )
    assert not fenced_provider_outer_authority_population_receipt_valid(orphaned)


def test_standard_outer_authority_preserves_attempt_bound_validation_run(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        repo.record_validation_result(
            task_cid=str(subject["task_cid"]),
            outcome="failed",
            evidence_digest="sha256:" + ("67" * 32),
            argv=("python", "-m", "pytest", "-q"),
            attempt_id="",
            body={"scope": "task-level"},
        )
        repo.record_validation_result(
            task_cid=str(subject["task_cid"]),
            outcome="failed",
            evidence_digest="sha256:" + ("78" * 32),
            argv=("python", "-m", "pytest", "-q"),
            attempt_id=str(subject["attempt_id"]),
            body={"scope": "attempt"},
        )
        receipt = _read_outer_receipt(repo, binding, subject)

    assert receipt["groups"]["task_attempts"]["count"] == 1
    assert receipt["groups"]["validation_runs"]["count"] == 2
    assert {
        row["attempt_id"]
        for row in receipt["groups"]["validation_runs"]["rows"]
    } == {"", subject["attempt_id"]}
    assert fenced_provider_outer_authority_population_receipt_valid(receipt)


def test_outer_authority_receipt_uses_real_quack_attached_schema_catalog(
    tmp_path: Path,
) -> None:
    """Issue through the real attached replica, not its client-local catalog."""

    (tmp_path / "control").mkdir()
    receipt_path, isolation_receipt = _isolation_receipt(tmp_path)
    database = Path(str(isolation_receipt["database_path"]))
    with open_intent_repository(database, owner_id="owner:quack-receipt-seed") as repo:
        subject = _seed_outer_receipt_subject(repo)
        with repo._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO attempt_phases (
                    attempt_id, phase_name, entered_at, exited_at, status
                ) VALUES (?, 'provider', ?, NULL, 'started')
                """,
                [subject["attempt_id"], "2026-09-02T00:02:00Z"],
            )

    store_id = "state/control.duckdb"
    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(isolation_receipt),
        store_id=store_id,
        repository_id="repository:audit",
        isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(
            allow_network_install=False,
            allow_local_load=True,
            use_cache=False,
        ),
    )
    connection = None
    try:
        identity = server.start()
        token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
        connection = open_quack_transport_connection(
            identity.listen_uri,
            token=token,
        )
        subject["expected_store_generation"] = identity.generation
        replica_observation = dict(server.status()["read_replica"])
        mutation_barrier = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "fenced-provider-outer-mutation-barrier@1"
            ),
            "store_id": store_id,
            "active_request_count": 0,
            "active_processing_count": 0,
            "active_population_digest": (
                "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba8"
                "73c2f11161202b945"
            ),
        }
        connection.execute("BEGIN TRANSACTION")
        try:
            source = DatabaseTaskSource(
                intent=_PinnedReadIntentRepository(connection),
                owner_id="owner:quack-receipt-reader",
            )
            receipt = dict(
                source.fenced_provider_outer_authority_population_receipt(
                    controller_replica_observation=replica_observation,
                    controller_mutation_barrier=mutation_barrier,
                    **subject,
                )
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        assert fenced_provider_outer_authority_population_receipt_valid(receipt)
        assert receipt["authority"]["owner_binding"]["server_id"] == (
            identity.server_id
        )
        assert receipt["authority"]["owner_binding"]["generation"] == (
            identity.generation
        )
        assert receipt["authority"]["owner_binding"]["store_id"] == store_id
        assert receipt["groups"]["attempt_phases"]["count"] == 1
    finally:
        if connection is not None:
            connection.close()
        server.stop()


def test_outer_authority_receipt_rejects_rehashed_shape_and_subject_splices(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        receipt = _read_outer_receipt(repo, binding, subject)

    extra_row_key = copy.deepcopy(receipt)
    extra_row_key["groups"]["tasks"]["rows"][0]["unreviewed"] = True
    extra_row_key["groups"]["tasks"]["rows_digest"] = (
        _fenced_provider_outer_sha256(
            extra_row_key["groups"]["tasks"]["rows"]
        )
    )
    extra_row_key["task_population_root"] = _fenced_provider_outer_sha256(
        extra_row_key["groups"]
    )
    unsigned = dict(extra_row_key)
    unsigned.pop("receipt_cid")
    extra_row_key["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        extra_row_key
    )

    subject_splice = copy.deepcopy(receipt)
    subject_splice["subject"]["task_cid"] = "task:cid:foreign"
    unsigned = dict(subject_splice)
    unsigned.pop("receipt_cid")
    subject_splice["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        subject_splice
    )

    nested_unknown = copy.deepcopy(receipt)
    nested_unknown["authority"]["owner_binding"]["unreviewed"] = True
    unsigned = dict(nested_unknown)
    unsigned.pop("receipt_cid")
    nested_unknown["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        nested_unknown
    )

    malformed_commitment = copy.deepcopy(receipt)
    malformed_commitment["groups"]["tasks"]["rows"][0]["body_json"] = {
        "canonical_sha256": "sha256:" + ("1" * 64),
        "canonical_byte_length": 2,
        "private": "leak",
    }
    malformed_commitment["groups"]["tasks"]["rows_digest"] = (
        _fenced_provider_outer_sha256(
            malformed_commitment["groups"]["tasks"]["rows"]
        )
    )
    malformed_commitment["task_population_root"] = (
        _fenced_provider_outer_sha256(malformed_commitment["groups"])
    )
    unsigned = dict(malformed_commitment)
    unsigned.pop("receipt_cid")
    malformed_commitment["receipt_cid"] = _fenced_provider_outer_sha256(
        unsigned
    )
    assert not fenced_provider_outer_authority_population_receipt_valid(
        malformed_commitment
    )

    required_null = copy.deepcopy(receipt)
    required_null["groups"]["tasks"]["rows"][0]["task_alias"] = None
    required_null["groups"]["tasks"]["rows_digest"] = (
        _fenced_provider_outer_sha256(
            required_null["groups"]["tasks"]["rows"]
        )
    )
    required_null["task_population_root"] = _fenced_provider_outer_sha256(
        required_null["groups"]
    )
    unsigned = dict(required_null)
    unsigned.pop("receipt_cid")
    required_null["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        required_null
    )

    excessive_nonce = copy.deepcopy(receipt)
    excessive_nonce["receipt_nonce"] = "n" * 513
    unsigned = dict(excessive_nonce)
    unsigned.pop("receipt_cid")
    excessive_nonce["receipt_cid"] = _fenced_provider_outer_sha256(unsigned)
    assert not fenced_provider_outer_authority_population_receipt_valid(
        excessive_nonce
    )

    revision_body_drift = copy.deepcopy(receipt)
    current_revision = revision_body_drift["subject"]["task_revision"]
    current_revision_row = next(
        row
        for row in revision_body_drift["groups"]["task_revisions"]["rows"]
        if row["revision"] == current_revision
    )
    current_revision_row["body_json"] = {
        "canonical_sha256": "sha256:" + ("6" * 64),
        "canonical_byte_length": 2,
    }
    revision_body_drift["groups"]["task_revisions"]["rows_digest"] = (
        _fenced_provider_outer_sha256(
            revision_body_drift["groups"]["task_revisions"]["rows"]
        )
    )
    revision_body_drift["task_population_root"] = (
        _fenced_provider_outer_sha256(revision_body_drift["groups"])
    )
    unsigned = dict(revision_body_drift)
    unsigned.pop("receipt_cid")
    revision_body_drift["receipt_cid"] = _fenced_provider_outer_sha256(
        unsigned
    )
    assert not fenced_provider_outer_authority_population_receipt_valid(
        revision_body_drift
    )


def test_outer_authority_receipt_rejects_rehashed_population_splices(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        receipt = _read_outer_receipt(repo, binding, subject)

    duplicate = copy.deepcopy(receipt)
    duplicate["groups"]["tasks"]["rows"].append(
        copy.deepcopy(duplicate["groups"]["tasks"]["rows"][0])
    )
    _rehash_outer_receipt_group(duplicate, "tasks")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        duplicate
    )

    cross_task = copy.deepcopy(receipt)
    assert cross_task["groups"]["task_outputs"]["rows"]
    cross_task["groups"]["task_outputs"]["rows"][0]["task_cid"] = (
        "task:cid:foreign"
    )
    _rehash_outer_receipt_group(cross_task, "task_outputs")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        cross_task
    )

    nested_join = copy.deepcopy(receipt)
    nested_join["groups"]["attempt_phases"]["rows"].append(
        {
            "attempt_id": "attempt:foreign",
            "phase_name": "provider",
            "entered_at": "2026-09-02T00:01:30Z",
            "exited_at": None,
            "status": "started",
        }
    )
    _rehash_outer_receipt_group(nested_join, "attempt_phases")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        nested_join
    )

    reordered = copy.deepcopy(receipt)
    revision_rows = reordered["groups"]["task_revisions"]["rows"]
    assert len(revision_rows) > 1
    revision_rows.reverse()
    _rehash_outer_receipt_group(reordered, "task_revisions")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        reordered
    )


def test_outer_authority_query_profile_binds_every_secondary_unique_key() -> None:
    profile = _fenced_provider_outer_query_profile()
    assert profile["transport_plan"] == {
        "attached_catalog_scan": "one_remote_table_per_sql_statement",
        "population_bounds": {
            "combined_task_row_bound": 50_000,
            "combined_variable_width_source_byte_bound": 16 * 1024 * 1024,
            "combined_parent_projection_byte_bound": 2 * 1024 * 1024,
            "combined_parent_projection_byte_policy": (
                "sum_of_all_parent_identity_projection_source_bytes_lte_bound"
            ),
        },
    }
    groups = {group["name"]: group for group in profile["groups"]}
    owner_groups = {
        group["name"]: group for group in profile["owner_groups"]
    }
    assert groups["tasks"]["unique_keys"] == [
        ["task_cid"],
        ["task_alias"],
    ]
    assert groups["task_outputs"]["unique_keys"] == [
        ["task_cid", "ordinal"],
        ["task_cid", "path"],
    ]
    assert groups["task_attempts"]["unique_keys"] == [
        ["attempt_id"],
        ["task_cid", "attempt_number"],
    ]
    assert groups["validation_results"]["unique_keys"] == [
        ["result_id"],
        ["run_id", "ordinal"],
    ]
    assert groups["domain_events"]["unique_keys"] == [
        ["event_id"],
        ["stream_id", "sequence"],
        ["global_sequence"],
    ]
    assert groups["worktrees"]["unique_keys"] == [
        ["worktree_id"],
        ["path"],
    ]
    assert owner_groups["state_server_record"]["unique_keys"] == [
        ["server_id"],
        ["process_birth_id"],
    ]
    assert all(group["unique_keys"] for group in groups.values())
    assert all(group["unique_keys"] for group in owner_groups.values())
    assert groups["attempt_phases"]["filter"] == {
        "kind": "closed_parent_id_single_table_scan",
        "target_column": "attempt_id",
        "parent_fields": [
            {"group": "task_attempts", "column": "attempt_id"}
        ],
        "deduplication": "set",
        "ordering": "utf8_lexicographic",
        "preprojection_count_policy": (
            "conservative_sum_of_parent_population_counts_lte_id_count_bound"
        ),
        "empty_predicate": "1=0",
        "nonempty_predicate": "target_column_in_positional_parameters",
        "id_count_bound": MAX_FENCED_PROVIDER_OUTER_DERIVED_IDS,
        "id_byte_bound": 2 * 1024 * 1024,
        "dynamic_sql_byte_bound": 65_536,
    }
    assert groups["provider_responses"]["filter"]["parent_fields"] == [
        {"group": "provider_calls", "column": "call_id"}
    ]
    assert groups["worktrees"]["filter"]["parent_fields"] == [
        {"group": "merge_queue_entries", "column": "worktree_id"},
        {"group": "path_claims", "column": "worktree_id"},
    ]
    assert groups["provider_calls"]["filter"] == {
        "kind": "direct_task_cid",
        "predicate": "task_cid = ?",
        "parameter_source": "subject.task_cid",
    }


def test_outer_authority_single_scan_filter_is_empty_deduplicated_and_bounded() -> None:
    assert _fenced_provider_outer_population_filter(
        group_name="attempt_phases",
        direct_predicate="unused nested predicate",
        task_cid="task:closed",
        groups={"task_attempts": {"rows": []}},
    ) == ("1=0", [])
    assert _fenced_provider_outer_population_filter(
        group_name="worktrees",
        direct_predicate="unused nested predicate",
        task_cid="task:closed",
        groups={
            "merge_queue_entries": {
                "rows": [
                    {"worktree_id": "worktree:b"},
                    {"worktree_id": "worktree:a"},
                ]
            },
            "path_claims": {
                "rows": [
                    {"worktree_id": "worktree:a"},
                    {"worktree_id": "worktree:c"},
                ]
            },
        },
    ) == (
        "worktree_id IN (?, ?, ?)",
        ["worktree:a", "worktree:b", "worktree:c"],
    )
    with pytest.raises(IntentRepositoryBoundsError, match="transport bound"):
        _fenced_provider_outer_population_filter(
            group_name="provider_responses",
            direct_predicate="unused nested predicate",
            task_cid="task:closed",
            groups={
                "provider_calls": {
                    "rows": [
                        {"call_id": f"call:{index:04d}"}
                        for index in range(
                            MAX_FENCED_PROVIDER_OUTER_DERIVED_IDS + 1
                        )
                    ]
                }
            },
        )


def test_outer_authority_rejects_two_route_parent_overflow_before_projection() -> None:
    bounded = {
        "task_attempts": ("task_cid = ?", ["task:closed"], 0),
        "provider_calls": ("task_cid = ?", ["task:closed"], 0),
        "merge_queue_entries": (
            "task_cid = ?",
            ["task:closed"],
            MAX_FENCED_PROVIDER_OUTER_DERIVED_IDS,
        ),
        "path_claims": ("task_cid = ?", ["task:closed"], 1),
    }
    with pytest.raises(
        IntentRepositoryBoundsError,
        match="pre-projection transport bound",
    ):
        _fenced_provider_outer_bound_parent_projection_counts(bounded)


def test_outer_authority_two_route_overflow_never_fetches_parent_ids(
) -> None:
    class NoProjectionConnection:
        calls = 0

        def execute(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("parent identity SQL ran before combined bound")

    connection = NoProjectionConnection()
    bounded = {
        "task_attempts": ("task_cid = ?", ["task:closed"], 0),
        "provider_calls": ("task_cid = ?", ["task:closed"], 0),
        "merge_queue_entries": (
            "task_cid = ?",
            ["task:closed"],
            MAX_FENCED_PROVIDER_OUTER_DERIVED_IDS,
        ),
        "path_claims": ("task_cid = ?", ["task:closed"], 1),
    }
    with pytest.raises(
        IntentRepositoryBoundsError,
        match="pre-projection transport bound",
    ):
        _fenced_provider_outer_parent_identity_projection(
            connection,
            bounded,
        )
    assert connection.calls == 0


def test_outer_authority_receipt_rejects_secondary_unique_collisions(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        receipt = _read_outer_receipt(repo, binding, subject)

    attempt_collision = copy.deepcopy(receipt)
    prior_attempt = attempt_collision["groups"]["task_attempts"]["rows"][0]
    second_attempt = copy.deepcopy(prior_attempt)
    second_attempt["attempt_id"] = prior_attempt["attempt_id"] + ":collision"
    attempt_collision["groups"]["task_attempts"]["rows"].append(
        second_attempt
    )
    _rehash_outer_receipt_group(attempt_collision, "task_attempts")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        attempt_collision
    )

    output_collision = copy.deepcopy(receipt)
    prior_output = output_collision["groups"]["task_outputs"]["rows"][0]
    second_output = copy.deepcopy(prior_output)
    second_output["ordinal"] = prior_output["ordinal"] + 1
    output_collision["groups"]["task_outputs"]["rows"].append(second_output)
    _rehash_outer_receipt_group(output_collision, "task_outputs")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        output_collision
    )

    stream_sequence_collision = copy.deepcopy(receipt)
    event_rows = stream_sequence_collision["groups"]["domain_events"]["rows"]
    assert event_rows
    prior_event = event_rows[-1]
    second_event = copy.deepcopy(prior_event)
    second_event["event_id"] = prior_event["event_id"] + ":collision"
    second_event["global_sequence"] = max(
        row["global_sequence"] for row in event_rows
    ) + 1
    event_rows.append(second_event)
    _rehash_outer_receipt_group(stream_sequence_collision, "domain_events")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        stream_sequence_collision
    )


def test_outer_authority_receipt_rejects_rehashed_relational_splices(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        _seed_outer_receipt_relations(repo, subject)
        receipt = _read_outer_receipt(repo, binding, subject)
    assert fenced_provider_outer_authority_population_receipt_valid(receipt)

    missing_prompt = copy.deepcopy(receipt)
    missing_prompt["groups"]["provider_calls"]["rows"][0][
        "prompt_instance_id"
    ] = "prompt:foreign"
    _rehash_outer_receipt_group(missing_prompt, "provider_calls")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        missing_prompt
    )

    missing_manifest = copy.deepcopy(receipt)
    missing_manifest["groups"]["prompt_instances"]["rows"][0][
        "manifest_cid"
    ] = "manifest:foreign"
    _rehash_outer_receipt_group(missing_manifest, "prompt_instances")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        missing_manifest
    )

    worktree_repository_drift = copy.deepcopy(receipt)
    worktree_repository_drift["groups"]["worktrees"]["rows"][0][
        "repository_id"
    ] = "repository:foreign"
    _rehash_outer_receipt_group(worktree_repository_drift, "worktrees")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        worktree_repository_drift
    )

    worktree_branch_drift = copy.deepcopy(receipt)
    worktree_branch_drift["groups"]["worktrees"]["rows"][0][
        "branch_name"
    ] = "implementation/foreign"
    _rehash_outer_receipt_group(worktree_branch_drift, "worktrees")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        worktree_branch_drift
    )

    path_repository_drift = copy.deepcopy(receipt)
    path_repository_drift["groups"]["path_claims"]["rows"][0][
        "repository_id"
    ] = "repository:foreign"
    _rehash_outer_receipt_group(path_repository_drift, "path_claims")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        path_repository_drift
    )

    missing_claim = copy.deepcopy(receipt)
    missing_claim["groups"]["lease_events"]["rows"][0]["claim_cid"] = (
        "claim:foreign"
    )
    _rehash_outer_receipt_group(missing_claim, "lease_events")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        missing_claim
    )

    merge_worktree_drift = copy.deepcopy(receipt)
    merge_worktree_drift["groups"]["merge_attempts"]["rows"][0][
        "worktree_id"
    ] = "worktree:foreign"
    _rehash_outer_receipt_group(merge_worktree_drift, "merge_attempts")
    assert not fenced_provider_outer_authority_population_receipt_valid(
        merge_worktree_drift
    )

    completion_goal_drift = copy.deepcopy(receipt)
    completion_goal_drift["groups"]["completion_receipts"]["rows"].append(
        {
            "receipt_cid": "receipt:foreign-goal",
            "task_cid": receipt["subject"]["task_cid"],
            "goal_cid": "goal:foreign",
            "attempt_id": receipt["subject"]["attempt_id"],
            "claim_cid": receipt["subject"]["claim_id"],
            "fencing_token": receipt["subject"]["fencing_token"],
            "completed_at": "2026-09-02T00:11:00Z",
            "validation_run_id": "",
            "evidence_digest": "sha256:foreign",
            "body_json": {
                "canonical_sha256": _fenced_provider_outer_sha256({}),
                "canonical_byte_length": 2,
            },
        }
    )
    _rehash_outer_receipt_group(
        completion_goal_drift,
        "completion_receipts",
    )
    assert not _fenced_provider_outer_groups_semantically_valid(
        completion_goal_drift["groups"],
        completion_goal_drift["subject"],
    )
    assert not fenced_provider_outer_authority_population_receipt_valid(
        completion_goal_drift
    )


def test_outer_authority_receipt_closes_provider_and_publication_populations(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        before = _read_outer_receipt(repo, binding, subject)
        with repo._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO provider_calls (
                    call_id, task_cid, attempt_id, provider_id,
                    prompt_instance_id, started_at, finished_at, status,
                    input_digest, output_digest, tokens_in, tokens_out, body_json
                ) VALUES (?, ?, ?, 'provider:test', '', ?, NULL, 'started',
                          'sha256:input', '', 0, 0, '{}')
                """,
                [
                    "call:alternate-key", subject["task_cid"],
                    subject["attempt_id"], "2026-09-02T00:03:00Z",
                ],
            )
            connection.execute(
                """
                INSERT INTO provider_responses (
                    response_id, call_id, received_at, status,
                    output_digest, body_json
                ) VALUES ('response:alternate-key', 'call:alternate-key', ?,
                          'unknown', '', '{}')
                """,
                ["2026-09-02T00:04:00Z"],
            )
        after = _read_outer_receipt(repo, binding, subject)
        assert after["receipt_cid"] != before["receipt_cid"]
        assert after["groups"]["provider_calls"]["count"] == 1
        assert after["groups"]["provider_responses"]["count"] == 1
        assert fenced_provider_outer_authority_population_receipt_valid(after)

        with repo._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO completion_receipts (
                    receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                    fencing_token, completed_at, validation_run_id,
                    evidence_digest, body_json
                ) VALUES ('receipt:accepted', ?, 'goal:cid:root', ?, ?, ?, ?, '',
                          'sha256:evidence', '{}')
                """,
                [
                    subject["task_cid"], subject["attempt_id"],
                    subject["claim_id"], subject["fencing_token"],
                    "2026-09-02T00:05:00Z",
                ],
            )
        with pytest.raises(Exception, match="already has admitted"):
            _read_outer_receipt(repo, binding, subject)


def test_outer_authority_single_scan_includes_all_closed_child_routes(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        _seed_outer_receipt_relations(repo, subject)
        with repo._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO attempt_phases (
                    attempt_id, phase_name, entered_at, exited_at, status
                ) VALUES (?, 'provider', ?, NULL, 'started')
                """,
                [subject["attempt_id"], "2026-09-02T00:02:00Z"],
            )
            connection.execute(
                """
                INSERT INTO provider_responses (
                    response_id, call_id, received_at, status,
                    output_digest, body_json
                ) VALUES ('response:outer', 'call:outer', ?, 'unknown', '', '{}')
                """,
                ["2026-09-02T00:05:30Z"],
            )
            for worktree_id, path, branch_name in (
                (
                    "worktree:path-only",
                    "/sealed/path-only",
                    "implementation/path-only",
                ),
                (
                    "worktree:merge-only",
                    "/sealed/merge-only",
                    "implementation/merge-only",
                ),
            ):
                connection.execute(
                    """
                    INSERT INTO worktrees (
                        worktree_id, repository_id, path, head_commit_id,
                        branch_name, owner_session_id, status, created_at,
                        updated_at, revision, fence_epoch,
                        extension_schema, extension_json
                    ) VALUES (?, 'repository:outer', ?, 'commit:outer', ?, ?,
                              'retained', ?, ?, 1, ?, '', '{}')
                    """,
                    [
                        worktree_id,
                        path,
                        branch_name,
                        subject["owner_session_id"],
                        "2026-09-02T00:11:00Z",
                        "2026-09-02T00:11:00Z",
                        subject["fence_epoch"],
                    ],
                )
            connection.execute(
                """
                INSERT INTO path_claims (
                    claim_id, repository_id, worktree_id, path,
                    owner_session_id, task_cid, fencing_token, fence_epoch,
                    acquired_at, expires_at, state, revision
                ) VALUES ('path-claim:path-only', 'repository:outer',
                          'worktree:path-only', 'src/path-only.py', ?, ?, ?, ?,
                          ?, ?, 'released', 1)
                """,
                [
                    subject["owner_session_id"],
                    subject["task_cid"],
                    subject["fencing_token"],
                    subject["fence_epoch"],
                    "2026-09-02T00:12:00Z",
                    "2026-09-02T00:13:00Z",
                ],
            )
            connection.execute(
                """
                INSERT INTO merge_queue_entries (
                    entry_id, repository_id, worktree_id, task_cid,
                    source_branch, target_branch, status, ordinal,
                    enqueued_at, updated_at, revision, fence_epoch
                ) VALUES ('entry:merge-only', 'repository:outer',
                          'worktree:merge-only', ?, 'implementation/merge-only',
                          'main', 'pending', 2, ?, ?, 1, ?)
                """,
                [
                    subject["task_cid"],
                    "2026-09-02T00:14:00Z",
                    "2026-09-02T00:14:00Z",
                    subject["fence_epoch"],
                ],
            )
        receipt = _read_outer_receipt(repo, binding, subject)

    assert fenced_provider_outer_authority_population_receipt_valid(receipt)
    assert receipt["groups"]["attempt_phases"]["count"] == 1
    assert receipt["groups"]["provider_responses"]["count"] == 1
    assert {
        row["worktree_id"]
        for row in receipt["groups"]["worktrees"]["rows"]
    } == {
        "worktree:outer",
        "worktree:path-only",
        "worktree:merge-only",
    }


def test_outer_authority_receipt_rejects_binding_and_schema_drift(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        wrong_generation = dict(binding)
        wrong_generation["generation"] = 8
        with pytest.raises(Exception, match="generation"):
            _read_outer_receipt(repo, wrong_generation, subject)

        with repo._connection(write=True) as connection:
            connection.execute("ALTER TABLE provider_calls ADD COLUMN surprise VARCHAR")
        with pytest.raises(Exception, match="closed profile"):
            _read_outer_receipt(repo, binding, subject)


@pytest.mark.parametrize(
    "statements",
    (
        ("DROP INDEX task_attempts_task_number_uidx",),
        (
            "CREATE UNIQUE INDEX task_attempts_owner_uidx "
            "ON task_attempts(owner_session_id)",
        ),
        (
            "DROP INDEX task_attempts_task_number_uidx",
            "CREATE UNIQUE INDEX task_attempts_changed_uidx "
            "ON task_attempts(task_cid, owner_session_id)",
        ),
        (
            "DROP INDEX task_attempts_task_number_uidx",
            "CREATE SCHEMA alternate",
            "CREATE TABLE alternate.task_attempts ("
            "task_cid VARCHAR, attempt_number BIGINT)",
            "CREATE UNIQUE INDEX task_attempts_compensation_uidx "
            "ON alternate.task_attempts(task_cid, attempt_number)",
        ),
    ),
)
def test_outer_authority_receipt_rejects_unique_constraint_profile_drift(
    tmp_path: Path,
    statements: tuple[str, ...],
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        with repo._connection(write=True) as connection:
            for statement in statements:
                connection.execute(statement)
        with pytest.raises(IntentRepositoryIntegrityError, match="uniqueness"):
            _read_outer_receipt(repo, binding, subject)


def test_outer_authority_receipt_rejects_expression_unique_index(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        with repo._connection(write=True) as connection:
            connection.execute(
                "CREATE UNIQUE INDEX task_attempts_expression_uidx "
                "ON task_attempts(lower(owner_session_id))"
            )
        with pytest.raises(IntentRepositoryIntegrityError, match="expression"):
            _read_outer_receipt(repo, binding, subject)


def test_outer_authority_receipt_bounds_rows_before_projection() -> None:
    with pytest.raises(IntentRepositoryBoundsError, match="exceeds"):
        _fenced_provider_outer_group(
            group_name="bounded-test",
            table="tasks",
            columns=("task_cid",),
            json_columns=frozenset(),
            bigint_columns=frozenset(),
            rows=[("task:bounded",)]
            * (MAX_FENCED_PROVIDER_OUTER_POPULATION_ROWS + 1),
        )


def test_outer_authority_receipt_rejects_oversized_json_before_decode(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        oversized_json = '"' + ("x" * 262_144) + '"'
        with repo._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [oversized_json, subject["task_cid"]],
            )
        with pytest.raises(IntentRepositoryBoundsError, match="byte bound"):
            _read_outer_receipt(repo, binding, subject)


@pytest.mark.parametrize(
    "field",
    (
        "task_revision",
        "fencing_token",
        "fence_epoch",
        "expected_store_generation",
        "receipt_epoch",
    ),
)
def test_outer_authority_receipt_rejects_bigint_overflow_before_scan(
    field: str,
) -> None:
    class NoScanConnection:
        calls = 0

        def execute(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("database scan occurred before input bounds")

    connection = NoScanConnection()
    repository = _PinnedReadIntentRepository(connection)
    subject = {
        "task_cid": "task:outer-bounds",
        "task_alias": "PCTDD-OUTER-BOUNDS",
        "task_revision": 1,
        "expected_task_status": "blocked",
        "attempt_id": "attempt:outer-bounds",
        "claim_id": "claim:outer-bounds",
        "lease_id": "lease:outer-bounds",
        "owner_session_id": "owner:outer-bounds",
        "fencing_token": 1,
        "fence_epoch": 1,
        "expected_store_id": "store:outer-bounds",
        "expected_store_generation": 1,
        "receipt_nonce": "nonce:outer-bounds",
        "receipt_epoch": 1,
        "controller_replica_observation": {},
        "controller_mutation_barrier": {},
    }
    subject[field] = 2**63
    with pytest.raises(IntentRepositoryBoundsError, match="signed BIGINT"):
        repository.fenced_provider_outer_authority_population_receipt(
            **subject
        )
    assert connection.calls == 0


@pytest.mark.parametrize(
    ("entry_status", "attempt_status"),
    (("accepted", "running"), ("settled", "running"), ("pending", "accepted")),
)
def test_outer_authority_receipt_rejects_authoritative_merge_publication_states(
    tmp_path: Path,
    entry_status: str,
    attempt_status: str,
) -> None:
    with _repo(tmp_path) as repo:
        binding, subject = _seed_outer_receipt_authority(repo)
        with repo._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO merge_queue_entries (
                    entry_id, repository_id, worktree_id, task_cid,
                    source_branch, target_branch, status, ordinal,
                    enqueued_at, updated_at, revision, fence_epoch
                ) VALUES ('entry:outer', 'repo:outer', 'worktree:outer', ?,
                          'implementation/outer', 'main', ?, 1, ?, ?, 1, 1)
                """,
                [
                    subject["task_cid"], entry_status,
                    "2026-09-02T00:03:00Z", "2026-09-02T00:03:00Z",
                ],
            )
            connection.execute(
                """
                INSERT INTO merge_attempts (
                    merge_attempt_id, entry_id, task_cid, worktree_id,
                    started_at, finished_at, status, result_commit_id, body_json
                ) VALUES ('merge-attempt:outer', 'entry:outer', ?,
                          'worktree:outer', ?, NULL, ?, '', '{}')
                """,
                [
                    subject["task_cid"], "2026-09-02T00:03:00Z",
                    attempt_status,
                ],
            )
        with pytest.raises(Exception, match="already has admitted"):
            _read_outer_receipt(repo, binding, subject)


# ---------------------------------------------------------------------------
# Core intent mutations + canonical identities
# ---------------------------------------------------------------------------


def test_objectives_goals_plans_tasks_retain_canonical_ids(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)

        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        assert objective["objective_id"] == "objective:dqp-012"
        # Alias lookup still returns the canonical objective_id.
        by_alias = repo.get_objective("DQP-O012")
        assert by_alias is not None
        assert by_alias["objective_id"] == objective["objective_id"]

        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        assert goal["goal_cid"] == "goal:cid:root"
        assert repo.get_goal("DQP-G020")["goal_cid"] == "goal:cid:root"  # type: ignore[index]

        plan = repo.get_plan(ids["plan_cid"])
        assert plan is not None
        assert plan["plan_cid"] == "plan:cid:v1"

        task = repo.get_task(ids["task_a"])
        assert task is not None
        assert task["task_cid"] == "task:cid:001"
        assert task["task_alias"] == "DQP-012-A"
        # Alias lookup preserves the durable CID.
        by_task_alias = repo.get_task("DQP-012-A")
        assert by_task_alias is not None
        assert by_task_alias["task_cid"] == "task:cid:001"
        assert by_task_alias["dependencies"] == ()
        assert len(by_task_alias["acceptance"]) == 1
        assert len(by_task_alias["validations"]) == 1
        assert len(by_task_alias["outputs"]) == 1
        assert by_task_alias["validations"][0]["argv"] == [
            "python",
            "-m",
            "pytest",
            "-q",
        ]
        assert by_task_alias["validations"][0]["policy"][
            "representation"
        ] == "argv"

        dependent = repo.get_task(ids["task_b"])
        assert dependent is not None
        assert dependent["dependencies"] == ("task:cid:001",)
        assert dependent["validations"][0]["argv"] == [
            "pytest test_rebuild.py"
        ]
        assert dependent["validations"][0]["policy"][
            "representation"
        ] == "shell_text"


def test_cas_heads_reject_stale_revisions(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        revision = int(task["revision"])

        # Provide evidence so a valid completion would succeed.
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["pytest"],
        )
        ok = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=revision,
            new_status="in_progress",
        )
        assert ok.changed is True

        with pytest.raises(IntentRepositoryConflictError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=revision,
                new_status="blocked",
            )

        # Objective CAS
        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        with pytest.raises(IntentRepositoryConflictError):
            repo.upsert_objective(
                objective_id=ids["objective_id"],
                objective_alias="DQP-O012",
                title="stale",
                expected_revision=0,
            )


def test_task_status_cas_can_share_a_caller_owned_transaction(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        revision = int(task["revision"])

        # The factored seam neither commits nor rolls back its caller's
        # transaction.  An outer failure therefore leaves no partial status,
        # revision, or event publication behind.
        with pytest.raises(RuntimeError, match="abort caller transaction"):
            with repo._connection(write=True) as connection:
                receipt = repo._cas_task_status_on_connection(
                    connection,
                    task_cid=ids["task_a"],
                    expected_revision=revision,
                    new_status="blocked",
                )
                assert receipt.changed is True
                assert connection.in_transaction is True
                raise RuntimeError("abort caller transaction")

        unchanged = repo.get_task(ids["task_a"])
        assert unchanged is not None
        assert unchanged["status"] == "ready"
        assert int(unchanged["revision"]) == revision

        with repo._connection(write=True) as connection:
            pinned = _PinnedFencedIntentRepository(
                connection,
                owner_id="database-implementation-daemon:owner:test",
            )
            receipt = pinned.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=revision,
                new_status="blocked",
            )
            assert receipt.changed is True
            assert pinned.task_status_cas_consumed is True
            assert connection.in_transaction is True

        changed = repo.get_task(ids["task_a"])
        assert changed is not None
        assert changed["status"] == "blocked"
        assert int(changed["revision"]) == revision + 1


# ---------------------------------------------------------------------------
# Completion evidence gate
# ---------------------------------------------------------------------------


def test_completion_requires_current_required_evidence(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None

        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        # Wrong digest still fails.
        repo.record_evidence(
            task_cid=ids["task_a"],
            evidence_kind="validation",
            digest="sha256:" + ("cd" * 32),
        )
        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["python", "-m", "pytest", "-q"],
        )
        satisfied, missing = repo.required_evidence_satisfied(ids["task_a"])
        assert satisfied is True
        assert missing == ()

        receipt = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed"},
            evidence_digests=[ids["evidence_digest"]],
        )
        assert receipt.changed is True
        assert receipt.event_type == IntentEventType.COMPLETION_RECORDED.value
        completed = repo.get_task(ids["task_a"])
        assert completed is not None
        assert completed["status"] == "completed"
        # Canonical identity unchanged across completion.
        assert completed["task_cid"] == "task:cid:001"


def test_ready_selection_respects_dependencies_and_excludes_completed(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        ready = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready] == ["task:cid:001"]

        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        ready_after = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready_after] == ["task:cid:002"]


# ---------------------------------------------------------------------------
# Queue backoff / retry
# ---------------------------------------------------------------------------


def test_queue_backoff_and_retry(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        now = 1_700_000_000_000
        repo._clock_ms = lambda: now  # type: ignore[method-assign]

        repo.record_queue_backoff(
            task_cid=ids["task_a"],
            delay_ms=60_000,
            reason="provider capacity",
            selection_penalty=100,
        )
        entry = repo.get_queue_entry(ids["task_a"])
        assert entry is not None
        assert entry.is_cooled_down(now_ms=now) is True
        assert entry.selection_penalty == 100

        ready = repo.select_ready_tasks(now_ms=now)
        assert all(item["task_cid"] != ids["task_a"] for item in ready)

        repo.record_queue_retry(task_cid=ids["task_a"])
        entry_after = repo.get_queue_entry(ids["task_a"])
        assert entry_after is not None
        assert entry_after.retry_not_before_ms == 0
        ready_after = repo.select_ready_tasks(now_ms=now)
        assert any(item["task_cid"] == ids["task_a"] for item in ready_after)


# ---------------------------------------------------------------------------
# Plan supersession / continuation / CAS heads
# ---------------------------------------------------------------------------


def test_plan_revision_repository_supersession_and_continuation(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        plans = repo.plan_revisions()
        assert isinstance(plans, PlanRevisionRepository)

        head = plans.head(ids["goal_cid"])
        assert head is not None
        assert head.plan_cid == ids["plan_cid"]

        plans.append_revision(
            plan_cid=ids["plan_cid"],
            expected_revision=1,
            delta={"add_step": "verify"},
            body={"steps": ["seed", "migrate", "verify"]},
        )
        revisions = plans.list_revisions(ids["plan_cid"])
        assert len(revisions) >= 2

        plans.upsert(
            plan_cid="plan:cid:v2",
            goal_cid=ids["goal_cid"],
            plan_alias="plan-v2",
            status="active",
            body={"steps": ["seed", "migrate", "verify", "export"]},
            set_head=False,
        )
        plans.supersede(
            plan_cid=ids["plan_cid"],
            successor_plan_cid="plan:cid:v2",
            expected_revision=2,
            reason="steering",
        )
        head_after = plans.head(ids["goal_cid"])
        assert head_after is not None
        assert head_after.plan_cid == "plan:cid:v2"
        superseded = plans.get(ids["plan_cid"])
        assert superseded is not None
        assert superseded["status"] == "superseded"

        plans.continue_from(
            plan_cid="plan:cid:v2",
            continuation_plan_cid="plan:cid:v2-cont",
            expected_revision=1,
            body={"phase": "export"},
        )
        cont = plans.get("plan:cid:v2-cont")
        assert cont is not None
        assert cont["status"] == "active"
        assert cont["body"].get("continuation_of") == "plan:cid:v2"


# ---------------------------------------------------------------------------
# Goal reopen, blocks, attempts
# ---------------------------------------------------------------------------


def test_goal_reopen_blocks_and_attempts(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        repo.upsert_goal(
            goal_cid=ids["goal_cid"],
            goal_alias="DQP-G020",
            title="Intent authority",
            objective_id=ids["objective_id"],
            status="verified_complete",
            expected_revision=int(goal["revision"]),
        )
        closed = repo.get_goal(ids["goal_cid"])
        assert closed is not None
        reopen = repo.reopen_goal(
            goal_cid=ids["goal_cid"],
            expected_revision=int(closed["revision"]),
            reason="new evidence required",
        )
        assert reopen.event_type == IntentEventType.GOAL_REOPENED.value
        reopened = repo.get_goal(ids["goal_cid"])
        assert reopened is not None
        assert reopened["status"] == "reopened"

        block = repo.block_task(
            task_cid=ids["task_b"],
            blocker_kind="dependency",
            blocker_id=ids["task_a"],
            reason="waiting on A",
        )
        assert block.changed is True
        blocked = repo.get_task(ids["task_b"])
        assert blocked is not None
        assert blocked["status"] == "blocked"
        ready = repo.select_ready_tasks()
        assert all(item["task_cid"] != ids["task_b"] for item in ready)

        repo.unblock_task(task_cid=ids["task_b"])
        unblocked = repo.get_task(ids["task_b"])
        assert unblocked is not None
        assert unblocked["status"] == "ready"

        attempt = repo.record_attempt(task_cid=ids["task_a"], status="started")
        assert attempt.event_type == IntentEventType.ATTEMPT_RECORDED.value


# ---------------------------------------------------------------------------
# Event rebuild parity
# ---------------------------------------------------------------------------


def test_rebuild_from_admitted_events_matches_projections(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        repo.record_queue_backoff(
            task_cid=ids["task_b"], delay_ms=5_000, reason="retry later"
        )

        before = repo.snapshot()
        assert before.task_count == 2
        assert before.event_watermark > 0

        events = repo.list_events(limit=1000)
        assert any(
            item["event_type"] == IntentEventType.TASK_UPSERTED.value
            for item in events
        )
        assert any(
            item["event_type"] == IntentEventType.COMPLETION_RECORDED.value
            for item in events
        )

        after = repo.rebuild_projections_from_events()
        assert after.projection_cid == before.projection_cid
        assert after.task_count == before.task_count
        assert after.goal_count == before.goal_count
        assert after.plan_count == before.plan_count
        assert after.dependency_count == before.dependency_count

        rebuilt_task = repo.get_task(ids["task_a"])
        assert rebuilt_task is not None
        assert rebuilt_task["task_cid"] == "task:cid:001"
        assert rebuilt_task["status"] == "completed"
        rebuilt_dep = repo.get_task(ids["task_b"])
        assert rebuilt_dep is not None
        assert rebuilt_dep["dependencies"] == ("task:cid:001",)

        # Recovery is a pure database operation (no external files).
        recovery = repo.recover()
        assert recovery.event_type == IntentEventType.RECOVERY_APPLIED.value


# ---------------------------------------------------------------------------
# DatabaseTaskSource public API
# ---------------------------------------------------------------------------


def test_database_task_source_public_api_and_completion_gate(tmp_path: Path) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    try:
        receipt = source.materialize(
            {
                "repository_tree_id": "tree:dqp-012",
                "objectives": [
                    {
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "objective_id": "objective:dqp-012",
                        "title": "Move intent into the database",
                        "acceptance_criteria": ["projections rebuild"],
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "DQP-012",
                        "task_cid": "task:cid:012",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "acceptance_criteria": [
                            {
                                "criterion": "tests pass",
                                "required_digest": "sha256:" + ("11" * 32),
                                "evidence_kind": "validation",
                            }
                        ],
                        "validation_commands": [
                            "python -m pytest -q test/api/test_agent_supervisor_intent_repository.py"
                        ],
                        "effects": [
                            {
                                "path": "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
                                "effect": "create",
                            }
                        ],
                    },
                    {
                        "task_id": "DQP-012-B",
                        "task_cid": "task:cid:012b",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "depends_on": ["DQP-012"],
                        "acceptance_criteria": ["ready after A"],
                        "validation_commands": ["true"],
                    },
                ],
            }
        )
        assert receipt["task_count"] == 2
        assert receipt["plan_root_cid"]
        assert source.SCHEMA == DATABASE_TASK_SOURCE_SCHEMA

        snap = source.snapshot()
        assert snap.task_count == 2
        assert snap.goal_count >= 1
        assert snap.source_schema == DATABASE_TASK_SOURCE_SCHEMA

        task = source.get_task("DQP-012")
        assert task is not None
        assert task.task_cid == "task:cid:012"
        assert source.get_task("task:cid:012") is not None
        assert source.get_task("task:cid:012").task_cid == task.task_cid  # type: ignore[union-attr]

        page = source.list_tasks(limit=1)
        assert len(page.tasks) == 1
        second = source.list_tasks(cursor=page.next_cursor, limit=1)
        assert len(second.tasks) == 1
        assert {page.tasks[0].task_cid, second.tasks[0].task_cid} == {
            "task:cid:012",
            "task:cid:012b",
        }

        ready = source.ready_tasks()
        assert [item.task_cid for item in ready.tasks] == ["task:cid:012"]

        with pytest.raises(TaskSourceCompletionError):
            source.compare_and_set_status(
                task.task_cid,
                task.revision,
                "completed",
            )

        source.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest="sha256:" + ("11" * 32),
            argv=["pytest"],
        )
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "completed",
            {"validation": "passed"},
            evidence_digests=["sha256:" + ("11" * 32)],
        )
        assert result.changed is True
        assert result.task.task_cid == "task:cid:012"
        assert result.task.status == "completed"

        ready_after = source.ready_tasks()
        assert [item.task_cid for item in ready_after.tasks] == ["task:cid:012b"]

        # Objective / plan identity APIs.
        assert source.get_objective("objective:dqp-012") is not None
        assert source.get_goal("goal:cid:g20") is not None
        assert source.get_plan(str(receipt["plan_root_cid"])) is not None

        assert source.projection_matches_events() is True

        with pytest.raises(TaskSourceBoundsError):
            source.list_tasks(limit=MAX_QUERY_LIMIT + 1)
    finally:
        source.close()


def test_database_task_source_stale_cursor_and_cas_conflict(tmp_path: Path) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:x",
                "objectives": [
                    {
                        "goal_cid": "goal:1",
                        "goal_id": "G1",
                        "title": "One",
                    }
                ],
                "taskboard": [
                    {
                        "task_cid": "task:1",
                        "task_id": "T1",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                    {
                        "task_cid": "task:2",
                        "task_id": "T2",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                ],
            }
        )
        page = source.list_tasks(limit=1)
        assert page.next_cursor
        # Corrupt cursor revision.
        with pytest.raises(TaskSourceConflictError):
            source.list_tasks(cursor=page.next_cursor[:-1] + "x", limit=1)

        task = source.get_task("task:1")
        assert task is not None
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind="validation",
            digest="sha256:" + ("22" * 32),
        )
        source.compare_and_set_status(task.task_cid, task.revision, "in_progress")
        with pytest.raises(TaskSourceConflictError):
            source.compare_and_set_status(task.task_cid, task.revision, "blocked")


def test_single_transaction_emits_events_without_external_files(
    tmp_path: Path,
) -> None:
    """Mutations only touch the database path — no sidecar saga files."""

    db = tmp_path / "control.duckdb"
    with open_intent_repository(db) as repo:
        _seed_graph(repo)
        watermark = repo.event_watermark()
        assert watermark > 0
    # Only the duckdb file (and maybe wal) under tmp_path — no markdown/json saga.
    names = {path.name for path in tmp_path.iterdir()}
    assert "control.duckdb" in names
    unexpected = {
        name
        for name in names
        if name.endswith((".md", ".json", ".jsonl")) and "control" not in name
    }
    assert not unexpected
