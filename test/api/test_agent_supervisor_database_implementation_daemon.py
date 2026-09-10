"""Tests for DatabaseImplementationDaemon@1 cutover (DQP-018).

Evidence subset: ready selection, strict shards, lost response, provider
capacity, hard quota, timeout, cancellation, crash, restart, stale worker,
status parity.

Acceptance: Four daemon processes claim distinct work; no task status is
updated in Markdown under database authority; JSON queue/status/events/PID
projections can be absent; crash/restart resumes from committed phase and does
not duplicate provider/effect work.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as database_portal_bridge_module,
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.merge import (
    database_coordination as coordination_module,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    read_jsonl_events,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    connect_duckdb_with_policy,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError as DatabaseTaskSourceConflictError,
    TaskSourceUnknownOutcomeError as DatabaseTaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_PROVIDER,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_FENCED_PROVIDER_OUTER_SNAPSHOT_SCHEMA,
    DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA,
    DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_ID,
    DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_SCHEMA,
    DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_PINS,
    DATABASE_RETRY_BUDGET_BACKPRESSURE_SCHEMA,
    DATABASE_RETRY_BUDGET_SCHEMA,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DATABASE_TERMINAL_LANDED_COMPLETION_OPERATION,
    DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT,
    DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    _canonical_mapping_matches,
    _prepared_reconciliation_barrier_core_matches,
    _database_portal_historical_interrupted_state_transition_budget_matches,
    _database_portal_quiesced_stale_dispatch_release_budget_matches,
    _database_terminal_claim_ordinal_lower_bound,
    database_fenced_provider_inner_population_receipt_valid,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
    DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS,
    DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_REASON,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_AUTHORIZATION_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_FIELDS,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID,
    DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_FIELDS,
    DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_PIN,
    DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_NO_PROVIDER_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_QUIESCED_STALE_DISPATCH_RELEASE_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA,
    DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
    DatabasePortalTerminalQuiescentStateAdvanced,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_database_implementation_daemon_from_args,
    build_portal_implementation_daemon_from_args,
    resolve_database_implementation_paths,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database implementation daemon tests",
)


def _population(task_count: int = 4) -> dict[str, object]:
    tasks = []
    for index in range(1, task_count + 1):
        tasks.append(
            {
                "task_cid": f"task:cid:{index:03d}",
                "task_id": f"DQP-T{index:03d}",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": index,
                "title": f"Task {index}",
            }
        )
    return {
        "repository_tree_id": "tree:dqp-018",
        "objectives": [
            {
                "objective_id": "objective:dqp-018",
                "objective_alias": "DQP-O018",
                "title": "Daemon cutover",
                "goal_cid": "goal:cid:root",
                "goal_alias": "DQP-G030",
                "status": "open",
            }
        ],
        "tasks": tasks,
    }


def _open_daemon(
    tmp_path: Path,
    *,
    session: str = "",
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    markdown_path: Path | None = None,
    provider_fn: Callable[[DatabaseTaskAttempt], dict[str, object]] | None = None,
    effect_fn: Callable[[DatabaseTaskAttempt, object], dict[str, object]] | None = None,
    validation_fn: Callable[
        [DatabaseTaskAttempt, object], dict[str, object]
    ]
    | None = None,
    lease_ms: int = 60_000,
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    task_prefix: str = "",
    max_task_attempts: int = 0,
    callbacks_bound: bool = True,
) -> DatabaseImplementationDaemon:
    database_path = tmp_path / "control.duckdb"
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = tmp_path / "execution.duckdb"

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {"status": "ok", "task_cid": attempt.task_cid}

    def effect(
        attempt: DatabaseTaskAttempt, provider_result: dict[str, object]
    ) -> dict[str, object]:
        if effect_calls is not None:
            effect_calls.append(attempt.task_cid)
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    return DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=coordination_path,
        execution_path=execution_path,
        owner_session_id=session,
        authority_mode="embedded",
        task_source_kind="duckdb",
        markdown_path=markdown_path,
        # Projections intentionally absent.
        state_path=None,
        strategy_path=None,
        events_path=None,
        pid_path=None,
        queue_path=None,
        lease_ms=lease_ms,
        provider_fn=(provider_fn or default_provider) if callbacks_bound else None,
        effect_fn=(effect_fn or effect) if callbacks_bound else None,
        validation_fn=validation_fn if callbacks_bound else None,
        clock_ms=clock_ms,
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        task_prefix=task_prefix,
        max_task_attempts=max_task_attempts,
    )


def _fenced_provider_dispatch_test_inputs(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build an exact current outer snapshot for the real execution store."""

    canonical_provider_key = f"provider:{attempt.attempt_id}"
    if daemon._dispatch_journal_entry(
        attempt,
        dispatch_kind="provider",
        idempotency_key=canonical_provider_key,
    ) is None:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=canonical_provider_key,
        )

    evidence: dict[str, object] = {
        "schema": DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_SCHEMA,
        "attempt_id": attempt.attempt_id,
        "task_cid": attempt.task_cid,
        "evidence_id": "sha256:" + "1" * 64,
        "migration_manifest_id": "sha256:" + "2" * 64,
        "migration_credit_id": "sha256:" + "3" * 64,
    }
    unsigned_fence = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "fenced-provider-recovery-dispatch-fence@1"
        ),
        "attempt_id": attempt.attempt_id,
        "task_cid": attempt.task_cid,
        "evidence_id": evidence["evidence_id"],
        "migration_manifest_id": evidence["migration_manifest_id"],
        "migration_credit_id": evidence["migration_credit_id"],
        "fencing_token": int(attempt.fencing_token),
        "fence_epoch": int(attempt.fence_epoch),
    }
    snapshot: dict[str, object] = {
        "schema": daemon_module.DATABASE_FENCED_PROVIDER_OUTER_SNAPSHOT_SCHEMA,
        "attempt_record": attempt.to_dict(),
        "provider_dispatch": (
            dict(dispatch)
            if isinstance(
                dispatch := daemon._dispatch_journal_entry(
                    attempt,
                    dispatch_kind="provider",
                    idempotency_key=canonical_provider_key,
                ),
                Mapping,
            )
            else None
        ),
        "callback_population": dict(
            daemon._fenced_provider_callback_population(attempt)
        ),
        "phase_history": daemon.phase_history(attempt.attempt_id),
        "provider_invocation_absent": True,
        "effect_claim_absent": True,
        "effect_dispatch_absent": True,
        "recovery_dispatch_fence_id": (
            daemon._database_no_provider_rearm_digest(unsigned_fence)
        ),
    }
    snapshot["snapshot_id"] = daemon._database_no_provider_rearm_digest(
        snapshot
    )
    assert daemon._fenced_provider_outer_state_matches(
        attempt,
        snapshot,
        require_recovery_dispatch_fence=False,
    )
    return evidence, snapshot


def _second_execution_store_view(
    daemon: DatabaseImplementationDaemon,
) -> DatabaseImplementationDaemon:
    """Open a second DuckDB connection without creating a second writer."""

    view = object.__new__(DatabaseImplementationDaemon)
    view._lock = threading.RLock()
    view._terminal_close_failure = ""
    view._closed = False
    view._clock_ms = daemon._clock_ms
    import duckdb

    view._connection = DuckDBConnection.wrap(
        connect_duckdb_with_policy(duckdb, daemon.execution_path)
    )
    return view


def test_interface_identities() -> None:
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert DatabaseImplementationDaemon.INTERFACE == (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE
    assert is_database_authority_mode(authority_mode="embedded")
    assert is_database_authority_mode(task_source_kind="duckdb")
    assert not is_database_authority_mode(
        authority_mode="legacy_markdown", task_source_kind="legacy-markdown"
    )


def test_inner_population_receipt_is_cross_store_stable_and_replay_exact(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "1" * 64,
            "recovery_credit_id": "sha256:" + "2" * 64,
            "receipt_nonce": "inner-receipt:one",
            "receipt_epoch": 1,
        }
        first = dict(
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        )
        second = dict(
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        )
    finally:
        daemon.close()

    assert first == second
    assert first["schema"] == DATABASE_FENCED_PROVIDER_INNER_POPULATION_RECEIPT_SCHEMA
    assert first["transaction_boundary"] == "cross_store_stable_read"
    assert database_fenced_provider_inner_population_receipt_valid(first)
    assert first["groups"]["database_task_attempts"]["count"] == 1
    assert first["groups"]["provider_invocations"]["count"] == 0
    assert first["groups"]["effect_claims"]["count"] == 0
    assert first["authority"]["process_instance_record"]["state"] == "active"


def _pctdd005_quack_control_schema_inputs() -> tuple[
    DatabaseProgramConfig,
    dict[str, object],
    dict[str, object],
]:
    pin = DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="env://PCTDD005_QUACK_SCHEMA_TEST_TOKEN",
        quack_endpoint="quack://127.0.0.1:41307",
        store_id=str(pin["owner_store_id"]),
        store_generation=str(pin["control_store_generation"]),
        schema_revision="1",
    )
    owner_binding: dict[str, object] = {
        "server_id": "server:pctdd005-quack-schema-test",
        "store_id": program.store_id,
        "database_uuid": str(pin["owner_database_uuid"]),
        "schema_revision": int(pin["owner_schema_revision"]),
        "schema_fingerprint": str(pin["owner_schema_fingerprint"]),
        "generation": int(pin["owner_generation_floor"]),
        "process_birth_id": "birth:pctdd005-quack-schema-test",
        "listen_uri": program.quack_endpoint,
        "extension_fingerprint": "sha256:" + "f" * 64,
    }
    profile: dict[str, object] = {
        "profile_revision": (
            daemon_module.DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION
        ),
        "profile_id": (
            daemon_module.DATASETS_AUTHORITATIVE_CONTROL_SCHEMA_PROFILE_ID
        ),
        "control_store_id": program.store_id,
        "control_store_generation": program.store_generation,
        "transport_schema_revision": program.schema_revision,
        "storage_schema_fingerprint": str(pin["owner_schema_fingerprint"]),
        "owner_database_uuid": str(pin["owner_database_uuid"]),
        "owner_generation_floor": int(pin["owner_generation_floor"]),
    }
    return program, owner_binding, profile


def _quack_schema_evidence_daemon(
    tmp_path: Path,
    *,
    program: DatabaseProgramConfig,
    owner_binding: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> DatabaseImplementationDaemon:
    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_uri=program.quack_endpoint,
        control_store_id=program.store_id,
        control_store_generation=program.store_generation,
        authenticated_control_store_binding=owner_binding,
        expected_control_schema_profile=profile,
        install_schema=False,
    )


def test_quack_numeric_schema_revision_uses_authenticated_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, owner_binding, profile = _pctdd005_quack_control_schema_inputs()
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        program.schema_revision,
    )
    daemon = _quack_schema_evidence_daemon(
        tmp_path,
        program=program,
        owner_binding=owner_binding,
        profile=profile,
    )

    daemon._verify_control_schema_for_open()
    evidence = dict(daemon.control_schema_evidence)

    assert program.schema_revision == "1"
    assert program.environment()[
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
    ] == "1"
    assert daemon.state_schema_revision == "1"
    assert evidence == {
        "state_schema_revision": "1",
        "profile_id": (
            daemon_module.DATASETS_AUTHORITATIVE_CONTROL_SCHEMA_PROFILE_ID
        ),
        "schema_fingerprint": owner_binding["schema_fingerprint"],
        "verified": True,
    }


def test_quack_numeric_schema_revision_rejects_missing_or_crossed_profile_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, owner_binding, profile = _pctdd005_quack_control_schema_inputs()
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        program.schema_revision,
    )

    missing = _quack_schema_evidence_daemon(
        tmp_path / "missing",
        program=program,
        owner_binding=None,
        profile=None,
    )
    missing._verify_control_schema_for_open()
    assert dict(missing.control_schema_evidence) == {
        "state_schema_revision": "1",
        "profile_id": "",
        "schema_fingerprint": "",
        "verified": False,
    }

    crossed_owner = dict(owner_binding)
    crossed_owner["schema_fingerprint"] = "sha256:" + "0" * 64
    crossed = _quack_schema_evidence_daemon(
        tmp_path / "crossed",
        program=program,
        owner_binding=crossed_owner,
        profile=profile,
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="does not match the authenticated owner/storage binding",
    ):
        crossed._verify_control_schema_for_open()

    incomplete = _quack_schema_evidence_daemon(
        tmp_path / "incomplete",
        program=program,
        owner_binding=owner_binding,
        profile=None,
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="control-schema evidence is incomplete",
    ):
        incomplete._verify_control_schema_for_open()


def test_inner_population_receipt_accepts_only_bound_quack_profile_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path / "embedded")
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        receipt = dict(
            daemon.fenced_provider_inner_population_receipt(
                attempt,
                task_revision=task.revision,
                recovery_manifest_id="sha256:" + "7" * 64,
                recovery_credit_id="sha256:" + "8" * 64,
                receipt_nonce="inner-receipt:quack-profile",
                receipt_epoch=1,
            )
        )
    finally:
        daemon.close()

    program, owner_binding, profile = _pctdd005_quack_control_schema_inputs()
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        program.schema_revision,
    )
    evidence_daemon = _quack_schema_evidence_daemon(
        tmp_path / "quack-evidence",
        program=program,
        owner_binding=owner_binding,
        profile=profile,
    )
    evidence_daemon._verify_control_schema_for_open()

    def rehash(candidate: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = copy.deepcopy(dict(candidate))
        unsigned.pop("receipt_cid", None)
        unsigned["receipt_cid"] = "sha256:" + hashlib.sha256(
            canonical_json(unsigned).encode("utf-8")
        ).hexdigest()
        return unsigned

    admitted = copy.deepcopy(receipt)
    admitted["authority"]["authority_mode"] = "quack"
    admitted["authority"]["control_schema_evidence"] = dict(
        evidence_daemon.control_schema_evidence
    )
    admitted = rehash(admitted)
    assert database_fenced_provider_inner_population_receipt_valid(admitted)

    missing = copy.deepcopy(admitted)
    missing["authority"]["control_schema_evidence"]["profile_id"] = ""
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(missing)
    )

    conflated = copy.deepcopy(admitted)
    conflated["authority"]["control_schema_evidence"][
        "state_schema_revision"
    ] = daemon_module.DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(conflated)
    )

    mismatched = copy.deepcopy(admitted)
    mismatched["authority"]["control_schema_evidence"][
        "schema_fingerprint"
    ] = "schema-profile:not-authenticated"
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(mismatched)
    )


def test_inner_population_receipt_fails_without_writer_flock_and_on_tamper(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "3" * 64,
            "recovery_credit_id": "sha256:" + "4" * 64,
            "receipt_nonce": "inner-receipt:tamper",
            "receipt_epoch": 1,
        }
        receipt = dict(
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        )
        writer_handle = daemon._embedded_writer_lock_handle
        daemon._embedded_writer_lock_handle = None
        try:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="writer fence",
            ):
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        finally:
            daemon._embedded_writer_lock_handle = writer_handle

        assert writer_handle is not None
        fcntl.flock(writer_handle.fileno(), fcntl.LOCK_UN)
        try:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="writer fence",
            ):
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        finally:
            fcntl.flock(
                writer_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        replacement = daemon._embedded_writer_lock_path.open("r+b")
        daemon._embedded_writer_lock_handle = replacement
        try:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="writer fence",
            ):
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        finally:
            replacement.close()
            daemon._embedded_writer_lock_handle = writer_handle

        writer_handle.close()
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="writer fence",
        ):
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        daemon._embedded_writer_lock_handle = None
    finally:
        daemon.close()

    unknown = {**receipt, "unreviewed": True}
    assert not database_fenced_provider_inner_population_receipt_valid(unknown)
    tampered = dict(receipt)
    tampered["groups"] = dict(receipt["groups"])
    tampered["groups"]["provider_invocations"] = dict(
        receipt["groups"]["provider_invocations"]
    )
    tampered["groups"]["provider_invocations"]["count"] = 1
    assert not database_fenced_provider_inner_population_receipt_valid(tampered)

    def rehash(candidate: dict[str, object]) -> dict[str, object]:
        unsigned = copy.deepcopy(candidate)
        unsigned.pop("receipt_cid", None)
        unsigned["receipt_cid"] = "sha256:" + hashlib.sha256(
            canonical_json(unsigned).encode("utf-8")
        ).hexdigest()
        return unsigned

    legacy_quack = copy.deepcopy(receipt)
    legacy_quack["authority"]["authority_mode"] = "quack"
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(legacy_quack)
    )

    partial_legacy_profile = copy.deepcopy(receipt)
    partial_legacy_profile["authority"]["control_schema_evidence"][
        "profile_id"
    ] = "datasets-authoritative-operational-control-plane@1"
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(partial_legacy_profile)
    )

    subject_splice = copy.deepcopy(receipt)
    subject_splice["subject"]["task_cid"] = "task:forged"
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(subject_splice)
    )

    deep_unknown = copy.deepcopy(receipt)
    attempt_group = deep_unknown["groups"]["database_task_attempts"]
    attempt_group["rows"][0]["unreviewed"] = True
    attempt_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            attempt_group["rows"]
        )
    )
    deep_unknown["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            deep_unknown["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(deep_unknown)
    )

    oversized = copy.deepcopy(receipt)
    oversized_attempt_group = oversized["groups"]["database_task_attempts"]
    oversized_attempt_group["rows"][0]["body_json"][
        "canonical_byte_length"
    ] = daemon_module.DATABASE_FENCED_PROVIDER_INNER_MAX_FIELD_BYTES + 1
    oversized_attempt_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            oversized_attempt_group["rows"]
        )
    )
    oversized["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            oversized["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(oversized)
    )

    zero_length = copy.deepcopy(receipt)
    zero_group = zero_length["groups"]["database_task_attempts"]
    zero_group["rows"][0]["body_json"]["canonical_byte_length"] = 0
    zero_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            zero_group["rows"]
        )
    )
    zero_length["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            zero_length["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(zero_length)
    )

    out_of_range = copy.deepcopy(receipt)
    attempt_group = out_of_range["groups"]["database_task_attempts"]
    attempt_group["rows"][0]["started_at_ms"] = 2**63
    attempt_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            attempt_group["rows"]
        )
    )
    out_of_range["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            out_of_range["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(out_of_range)
    )

    duplicate_inner = copy.deepcopy(receipt)
    phase_group = duplicate_inner["groups"]["attempt_phases"]
    assert phase_group["rows"]
    phase_group["rows"].append(copy.deepcopy(phase_group["rows"][0]))
    phase_group["count"] = len(phase_group["rows"])
    phase_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            phase_group["rows"]
        )
    )
    duplicate_inner["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            duplicate_inner["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(duplicate_inner)
    )

    same_primary_key = copy.deepcopy(receipt)
    same_key_phase_group = same_primary_key["groups"]["attempt_phases"]
    assert same_key_phase_group["rows"]
    forged_phase = copy.deepcopy(same_key_phase_group["rows"][0])
    forged_phase["committed_at_ms"] += 1
    same_key_phase_group["rows"].append(forged_phase)
    same_key_phase_group["rows"].sort(
        key=lambda row: (row["committed_at_ms"], row["phase"])
    )
    same_key_phase_group["count"] = len(same_key_phase_group["rows"])
    same_key_phase_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            same_key_phase_group["rows"]
        )
    )
    same_primary_key["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            same_primary_key["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(same_primary_key)
    )

    cross_row_splice = copy.deepcopy(receipt)
    spliced_phase_group = cross_row_splice["groups"]["attempt_phases"]
    assert spliced_phase_group["rows"]
    spliced_phase_group["rows"][0]["fencing_token"] += 100
    spliced_phase_group["rows_digest"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            spliced_phase_group["rows"]
        )
    )
    cross_row_splice["execution_population_root"] = (
        DatabaseImplementationDaemon._database_canonical_digest(
            cross_row_splice["groups"]
        )
    )
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(cross_row_splice)
    )

    duplicate_nested = copy.deepcopy(receipt)
    nested_receipt = duplicate_nested["coordinator_receipt"]
    nested_token_group = nested_receipt["groups"]["token_history"]
    assert nested_token_group["rows"]
    nested_token_group["rows"].append(
        copy.deepcopy(nested_token_group["rows"][0])
    )
    nested_token_group["count"] = len(nested_token_group["rows"])
    nested_token_group["rows_digest"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(nested_token_group["rows"])
    )
    nested_receipt["task_population_root"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(nested_receipt["groups"])
    )
    nested_unsigned = copy.deepcopy(nested_receipt)
    nested_unsigned.pop("receipt_cid", None)
    nested_receipt["receipt_cid"] = coordination_module._sha256_hex(
        coordination_module.canonical_json_bytes(nested_unsigned)
    )
    duplicate_nested["coordinator_receipt_cid"] = nested_receipt["receipt_cid"]
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(duplicate_nested)
    )

    cross_splice = copy.deepcopy(receipt)
    nested = cross_splice["coordinator_receipt"]
    nested["receipt_nonce"] = "inner-receipt:different"
    nested_unsigned = copy.deepcopy(nested)
    nested_unsigned.pop("receipt_cid", None)
    nested["receipt_cid"] = "sha256:" + hashlib.sha256(
        coordination_module.canonical_json_bytes(nested_unsigned)
    ).hexdigest()
    cross_splice["coordinator_receipt_cid"] = nested["receipt_cid"]
    assert not database_fenced_provider_inner_population_receipt_valid(
        rehash(cross_splice)
    )

    assert receipt["privacy_boundary"] == (
        daemon_module.DATABASE_FENCED_PROVIDER_INNER_PRIVACY_BOUNDARY
    )
    assert receipt["nonclaims"] == list(
        daemon_module.DATABASE_FENCED_PROVIDER_INNER_NONCLAIMS
    )


def test_inner_population_receipt_requires_current_quiescent_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "5" * 64,
            "recovery_credit_id": "sha256:" + "6" * 64,
            "receipt_nonce": "inner-receipt:quiescent",
            "receipt_epoch": 1,
        }
        with daemon._lock:
            daemon._active_external_callbacks = 1
        try:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match="quiescent",
            ):
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        finally:
            with daemon._lock:
                daemon._active_external_callbacks = 0

        observed = daemon.process_birth
        monkeypatch.setattr(
            daemon_module,
            "current_process_birth",
            lambda: type(observed)(
                pid=observed.pid + 1,
                start_time_ticks=observed.start_time_ticks,
                boot_id=observed.boot_id,
                parent_pid=observed.parent_pid,
            ),
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="quiescent",
        ):
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
    finally:
        daemon.close()


def test_inner_population_receipt_retains_lock_through_callback_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    captured = threading.Event()
    release = threading.Event()
    mutation_done = threading.Event()
    errors: list[BaseException] = []
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "7" * 64,
            "recovery_credit_id": "sha256:" + "8" * 64,
            "receipt_nonce": "inner-receipt:retained-lock",
            "receipt_epoch": 1,
        }
        original = daemon._fenced_provider_execution_population_receipt

        def delayed(*args: object, **call_kwargs: object) -> Mapping[str, object]:
            result = original(*args, **call_kwargs)
            captured.set()
            if not release.wait(5):
                raise AssertionError("receipt callback release timed out")
            return result

        monkeypatch.setattr(
            daemon,
            "_fenced_provider_execution_population_receipt",
            delayed,
        )

        def issue() -> None:
            try:
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
            except BaseException as exc:
                errors.append(exc)

        def mutate() -> None:
            with daemon._lock:
                daemon._record_event(
                    "receipt-race",
                    attempt_id=attempt.attempt_id,
                    task_cid=attempt.task_cid,
                )
                daemon._require_connection().commit()
            mutation_done.set()

        issuer = threading.Thread(target=issue)
        issuer.start()
        assert captured.wait(5)
        writer = threading.Thread(target=mutate)
        writer.start()
        assert not mutation_done.wait(0.2)
        release.set()
        issuer.join(5)
        writer.join(5)
        assert not issuer.is_alive()
        assert not writer.is_alive()
        assert mutation_done.is_set()
        assert errors == []
    finally:
        release.set()
        daemon.close()


def test_inner_population_receipt_lock_contention_fails_without_deadlock(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    errors: list[BaseException] = []
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "9" * 64,
            "recovery_credit_id": "sha256:" + "a" * 64,
            "receipt_nonce": "inner-receipt:contended",
            "receipt_epoch": 1,
        }

        def issue() -> None:
            try:
                daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
            except BaseException as exc:
                errors.append(exc)

        daemon._lock.acquire()
        try:
            issuer = threading.Thread(target=issue)
            issuer.start()
            issuer.join(2)
            assert not issuer.is_alive()
        finally:
            daemon._lock.release()
        assert len(errors) == 1
        assert isinstance(errors[0], DatabaseImplementationConflictError)
        assert "contended" in str(errors[0])
    finally:
        daemon.close()


def test_inner_population_receipt_cancellation_rolls_back_both_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        kwargs = {
            "task_revision": task.revision,
            "recovery_manifest_id": "sha256:" + "b" * 64,
            "recovery_credit_id": "sha256:" + "c" * 64,
            "receipt_nonce": "inner-receipt:cancelled",
            "receipt_epoch": 1,
        }
        original = daemon_module._database_inner_population_preflight
        calls = 0

        def cancel_once(*args: object, **call_kwargs: object) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt("cancelled inner transaction")
            return original(*args, **call_kwargs)

        monkeypatch.setattr(
            daemon_module,
            "_database_inner_population_preflight",
            cancel_once,
        )
        with pytest.raises(KeyboardInterrupt, match="cancelled inner"):
            daemon.fenced_provider_inner_population_receipt(attempt, **kwargs)
        assert not daemon._require_connection().in_transaction
        assert not daemon.coordinator._connection.in_transaction
        receipt = daemon.fenced_provider_inner_population_receipt(
            attempt,
            **kwargs,
        )
        assert database_fenced_provider_inner_population_receipt_valid(receipt)
    finally:
        daemon.close()


def test_process_birth_sidecar_allows_default_owner_clean_reopen(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path)
    prior_process = first.process_instance_id
    stable_owner = first.owner_session_id
    first_birth = first.process_birth.to_dict()
    first.close()

    successor = _open_daemon(tmp_path)
    try:
        assert successor.owner_session_id == stable_owner
        assert successor.process_instance_id != prior_process
        prior = successor._database_process_instance_record(prior_process)
        assert prior is not None
        assert prior["owner_session_id"] == stable_owner
        assert prior["process_birth"] == first_birth
        assert prior["state"] == "closed"
        # A clean-close marker is not itself proof that the process birth is
        # dead; this test reopens in the same live Python process.
        assert successor._fenced_provider_predecessor_is_dead(
            task=SimpleNamespace(task_alias="FUTURE-T001", task_cid="task:future"),
            receipt={
                "process_instance_id": prior_process,
                "owner_session_id": stable_owner,
            },
            evidence={},
        ) is False
    finally:
        successor.close()


def test_arbitrary_successor_session_cannot_bypass_live_process_birth(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:predecessor")
    original_owner = daemon.owner_session_id
    try:
        daemon.owner_session_id = "session:spoofed-successor"
        assert daemon._fenced_provider_predecessor_is_dead(
            task=SimpleNamespace(task_alias="FUTURE-T001", task_cid="task:future"),
            receipt={
                "process_instance_id": daemon.process_instance_id,
                "owner_session_id": original_owner,
            },
            evidence={},
        ) is False
    finally:
        daemon.owner_session_id = original_owner
        daemon.close()


@pytest.mark.parametrize(
    ("liveness", "expected"),
    (
        (daemon_module.OwnerLiveness.ALIVE, False),
        (daemon_module.OwnerLiveness.UNKNOWN, False),
        (daemon_module.OwnerLiveness.DEAD, True),
    ),
)
def test_predecessor_gate_uses_exact_persisted_process_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    liveness: object,
    expected: bool,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:persisted-birth")
    process_instance_id = daemon.process_instance_id
    record = daemon._database_process_instance_record(process_instance_id)
    assert record is not None
    persisted_birth = daemon_module.ProcessBirthIdentity.from_dict(
        record["process_birth"]
    )
    observed: list[object] = []

    def classify(birth: object) -> object:
        observed.append(birth)
        return liveness

    monkeypatch.setattr(daemon_module, "owner_liveness", classify)
    try:
        assert daemon._fenced_provider_predecessor_is_dead(
            task=SimpleNamespace(
                task_alias="FUTURE-T001",
                task_cid="task:future",
            ),
            receipt={
                "process_instance_id": process_instance_id,
                "owner_session_id": daemon.owner_session_id,
            },
            evidence={},
        ) is expected
        assert observed == [persisted_birth]
    finally:
        daemon.close()


def test_close_response_loss_detaches_all_authority_before_writer_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    original_record = daemon._database_process_instance_record
    reads = 0

    def response_lost(process_instance_id: str) -> Mapping[str, Any] | None:
        nonlocal reads
        reads += 1
        if reads >= 2:
            raise RuntimeError("injected committed-close response loss")
        return original_record(process_instance_id)

    monkeypatch.setattr(
        daemon,
        "_database_process_instance_record",
        response_lost,
    )
    with pytest.raises(RuntimeError, match="committed-close response loss"):
        daemon.close()
    assert daemon._connection is None
    assert daemon._coordinator is None
    assert daemon._task_source is None
    assert daemon._embedded_writer_lock_handle is None
    assert daemon._closed is True
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="poisoned by an uncertain close",
    ):
        daemon.open()

    successor = _open_daemon(tmp_path)
    successor.close()


def test_close_refuses_while_external_callback_is_active(tmp_path: Path) -> None:
    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    entered = threading.Event()
    release = threading.Event()
    results: list[Mapping[str, Any]] = []
    errors: list[BaseException] = []

    def callback() -> Mapping[str, Any]:
        entered.set()
        assert release.wait(timeout=5)
        return {"status": "done"}

    def run() -> None:
        try:
            results.append(
                daemon._run_with_attempt_heartbeat(attempt, callback)
            )
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="external callback is active",
    ):
        daemon.close()
    assert daemon._connection is not None
    assert daemon._embedded_writer_lock_handle is not None
    assert daemon._database_process_instance_record(
        daemon.process_instance_id
    )["state"] == "active"

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert results == [{"status": "done"}]
    daemon.close()


@pytest.mark.parametrize("installer_connection", ("shared", "second"))
@pytest.mark.parametrize("dispatch_kind", ("provider", "effect"))
def test_real_dispatch_fence_wins_fresh_begin_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installer_connection: str,
    dispatch_kind: str,
) -> None:
    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = (
        daemon
        if installer_connection == "shared"
        else _second_execution_store_view(daemon)
    )
    checked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_dispatch_entry = daemon._dispatch_journal_entry

    def pause_after_prior_read(*args: object, **kwargs: object) -> object:
        result = real_dispatch_entry(*args, **kwargs)
        if threading.current_thread().name == "late-fresh-dispatch":
            checked.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(
        daemon,
        "_dispatch_journal_entry",
        pause_after_prior_read,
    )

    def late_begin() -> None:
        late_key = (
            f"provider:alternate:{attempt.attempt_id}"
            if dispatch_kind == "provider"
            else f"effect:{attempt.attempt_id}"
        )
        try:
            daemon._begin_callback_dispatch(
                attempt,
                dispatch_kind=dispatch_kind,
                idempotency_key=late_key,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=late_begin,
        name="late-fresh-dispatch",
        daemon=True,
    )
    thread.start()
    assert checked.wait(timeout=5)
    try:
        installed = installer._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
        assert installed["state"] == "sealed"
    finally:
        release.set()
        thread.join(timeout=5)
        if installer is not daemon:
            installer._connection.close()
            installer._connection = None
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    assert real_dispatch_entry(
        attempt,
        dispatch_kind=dispatch_kind,
        idempotency_key=(
            f"provider:alternate:{attempt.attempt_id}"
            if dispatch_kind == "provider"
            else f"effect:{attempt.attempt_id}"
        ),
    ) is None
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt)[
        "state"
    ] == "sealed"
    daemon.close()


def test_preexisting_alternate_provider_journal_blocks_dispatch_fence(
    tmp_path: Path,
) -> None:
    """A point-identical canonical row cannot hide a second provider row."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    attempt_before = attempt.to_dict()
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    alternate_key = f"provider:alternate:{attempt.attempt_id}"
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=alternate_key,
    )

    with pytest.raises(
        DatabaseImplementationConflictError,
        match="outer state advanced",
    ):
        daemon._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )

    assert daemon._fenced_provider_recovery_dispatch_fence(attempt) is None
    assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt_before
    count = daemon._require_connection().execute(
        "SELECT COUNT(*) FROM attempt_dispatch_journal WHERE attempt_id = ?",
        [attempt.attempt_id],
    ).fetchone()
    assert count is not None and int(count[0]) == 2
    daemon.close()


@pytest.mark.parametrize("field", ("dispatch_id", "started_at_ms"))
def test_canonical_provider_row_replacement_blocks_dispatch_fence(
    tmp_path: Path,
    field: str,
) -> None:
    """Every authority-bearing canonical journal column is snapshot-bound."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    if field == "dispatch_id":
        daemon._require_connection().execute(
            """
            UPDATE attempt_dispatch_journal
            SET dispatch_id = ?
            WHERE attempt_id = ? AND dispatch_kind = 'provider'
            """,
            ["dispatch:replacement", attempt.attempt_id],
        )
    else:
        daemon._require_connection().execute(
            """
            UPDATE attempt_dispatch_journal
            SET started_at_ms = started_at_ms + 1
            WHERE attempt_id = ? AND dispatch_kind = 'provider'
            """,
            [attempt.attempt_id],
        )

    with pytest.raises(
        DatabaseImplementationConflictError,
        match="outer state advanced",
    ):
        daemon._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt) is None
    daemon.close()


def test_callback_population_digest_tamper_blocks_dispatch_fence(
    tmp_path: Path,
) -> None:
    """A self-inconsistent population digest is rejected before mutation."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    tampered = json.loads(json.dumps(snapshot))
    tampered["callback_population"]["dispatch_journal"]["digest"] = (
        "sha256:" + "f" * 64
    )
    callback_unsigned = dict(tampered["callback_population"])
    callback_unsigned.pop("population_id")
    tampered["callback_population"]["population_id"] = (
        daemon._database_no_provider_rearm_digest(callback_unsigned)
    )
    snapshot_unsigned = dict(tampered)
    snapshot_unsigned.pop("snapshot_id")
    tampered["snapshot_id"] = daemon._database_no_provider_rearm_digest(
        snapshot_unsigned
    )

    with pytest.raises(
        DatabaseImplementationConflictError,
        match="input is invalid",
    ):
        daemon._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=tampered,
        )
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt) is None
    daemon.close()


def test_independent_alternate_provider_insert_wins_before_fence_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installer SQL rechecks the whole population after its precheck."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = _second_execution_store_view(daemon)
    prechecked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_match = installer._fenced_provider_outer_state_matches

    def pause_after_population_precheck(*args: object, **kwargs: object) -> bool:
        matched = real_match(*args, **kwargs)
        prechecked.set()
        assert release.wait(timeout=5)
        return matched

    monkeypatch.setattr(
        installer,
        "_fenced_provider_outer_state_matches",
        pause_after_population_precheck,
    )

    def install() -> None:
        try:
            installer._install_fenced_provider_recovery_dispatch_fence(
                attempt,
                evidence=evidence,
                snapshot=snapshot,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=install, daemon=True)
    thread.start()
    assert prechecked.wait(timeout=5)
    try:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:alternate:{attempt.attempt_id}",
        )
    finally:
        release.set()
        thread.join(timeout=5)
        installer._connection.close()
        installer._connection = None

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt) is None
    count = daemon._require_connection().execute(
        "SELECT COUNT(*) FROM attempt_dispatch_journal WHERE attempt_id = ?",
        [attempt.attempt_id],
    ).fetchone()
    assert count is not None and int(count[0]) == 2
    daemon.close()


def test_independent_phase_commit_wins_after_fence_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fence CAS also pins the attempt revision backing phase history."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = _second_execution_store_view(daemon)
    prechecked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_match = installer._fenced_provider_outer_state_matches

    def pause_after_outer_precheck(*args: object, **kwargs: object) -> bool:
        matched = real_match(*args, **kwargs)
        prechecked.set()
        assert release.wait(timeout=5)
        return matched

    monkeypatch.setattr(
        installer,
        "_fenced_provider_outer_state_matches",
        pause_after_outer_precheck,
    )

    def install() -> None:
        try:
            installer._install_fenced_provider_recovery_dispatch_fence(
                attempt,
                evidence=evidence,
                snapshot=snapshot,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=install, daemon=True)
    thread.start()
    assert prechecked.wait(timeout=5)
    try:
        daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
    finally:
        release.set()
        thread.join(timeout=5)
        installer._connection.close()
        installer._connection = None

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt) is None
    assert daemon.get_attempt(attempt.attempt_id).revision == attempt.revision + 1
    daemon.close()


@pytest.mark.parametrize("installer_connection", ("shared", "second"))
def test_real_dispatch_fence_wins_deferred_resume_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installer_connection: str,
) -> None:
    now = [1_000]
    daemon = _open_daemon(tmp_path, clock_ms=lambda: now[0])
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    key = f"provider:{attempt.attempt_id}"
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=key,
    )
    deferred_body = {
        "exception_type": "DatabasePortalProviderRouteDeferred",
        "backoff_seconds": 1,
        "retry_not_before_ms": 2_000,
    }
    daemon._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=key,
        outcome="deferred",
        body=deferred_body,
        updated_at_ms=1_000,
    )
    now[0] = 2_000
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = (
        daemon
        if installer_connection == "shared"
        else _second_execution_store_view(daemon)
    )
    checked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_dispatch_entry = daemon._dispatch_journal_entry

    def pause_after_deferred_read(*args: object, **kwargs: object) -> object:
        result = real_dispatch_entry(*args, **kwargs)
        if threading.current_thread().name == "late-deferred-resume":
            checked.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(
        daemon,
        "_dispatch_journal_entry",
        pause_after_deferred_read,
    )

    def late_resume() -> None:
        try:
            daemon._begin_callback_dispatch(
                attempt,
                dispatch_kind="provider",
                idempotency_key=key,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=late_resume,
        name="late-deferred-resume",
        daemon=True,
    )
    thread.start()
    assert checked.wait(timeout=5)
    try:
        installer._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
    finally:
        release.set()
        thread.join(timeout=5)
        if installer is not daemon:
            installer._connection.close()
            installer._connection = None
    assert not thread.is_alive()
    assert len(errors) == 1
    assert "single-resumer admission" in str(errors[0])
    assert dict(real_dispatch_entry(
        attempt,
        dispatch_kind="provider",
        idempotency_key=key,
    ))["body"] == deferred_body
    assert daemon._fenced_provider_recovery_dispatch_fence(attempt)[
        "state"
    ] == "sealed"
    daemon.close()


@pytest.mark.parametrize("installer_connection", ("shared", "second"))
def test_real_dispatch_fence_wins_outcome_update_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installer_connection: str,
) -> None:
    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    key = f"provider:{attempt.attempt_id}"
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=key,
    )
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = (
        daemon
        if installer_connection == "shared"
        else _second_execution_store_view(daemon)
    )
    checked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_fence = daemon._fenced_provider_recovery_dispatch_fence

    def pause_after_fence_precheck(selected: object) -> object:
        result = real_fence(selected)
        if threading.current_thread().name == "late-outcome":
            checked.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(
        daemon,
        "_fenced_provider_recovery_dispatch_fence",
        pause_after_fence_precheck,
    )

    def late_outcome() -> None:
        try:
            daemon._record_callback_dispatch_outcome(
                attempt,
                dispatch_kind="provider",
                idempotency_key=key,
                outcome="returned",
                body={"status": "late"},
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=late_outcome,
        name="late-outcome",
        daemon=True,
    )
    thread.start()
    assert checked.wait(timeout=5)
    try:
        installer._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
    finally:
        release.set()
        thread.join(timeout=5)
        if installer is not daemon:
            installer._connection.close()
            installer._connection = None
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    journal = daemon._dispatch_journal_entry(
        attempt,
        dispatch_kind="provider",
        idempotency_key=key,
    )
    assert journal is not None and journal["outcome"] == "started"
    assert real_fence(attempt)["state"] == "sealed"
    daemon.close()


@pytest.mark.parametrize("installer_connection", ("shared", "second"))
def test_real_dispatch_fence_wins_phase_commit_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installer_connection: str,
) -> None:
    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = (
        daemon
        if installer_connection == "shared"
        else _second_execution_store_view(daemon)
    )
    checked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_fence = daemon._fenced_provider_recovery_dispatch_fence

    def pause_after_fence_precheck(selected: object) -> object:
        result = real_fence(selected)
        if threading.current_thread().name == "late-phase":
            checked.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(
        daemon,
        "_fenced_provider_recovery_dispatch_fence",
        pause_after_fence_precheck,
    )

    def late_phase() -> None:
        try:
            daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=late_phase,
        name="late-phase",
        daemon=True,
    )
    thread.start()
    assert checked.wait(timeout=5)
    try:
        installer._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
    finally:
        release.set()
        thread.join(timeout=5)
        if installer is not daemon:
            installer._connection.close()
            installer._connection = None
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    assert daemon.phase_history(attempt.attempt_id) == snapshot["phase_history"]
    assert real_fence(attempt)["state"] == "sealed"
    daemon.close()


@pytest.mark.parametrize("installer_connection", ("shared", "second"))
@pytest.mark.parametrize("receipt_kind", ("provider", "effect"))
def test_real_dispatch_fence_wins_receipt_insert_interleaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installer_connection: str,
    receipt_kind: str,
) -> None:
    """The SQL guard closes precheck-to-receipt-INSERT on both connections."""

    daemon = _open_daemon(tmp_path)
    daemon.materialize_population(_population(1))
    attempt = daemon.claim_next()
    assert attempt is not None
    # The no-callback reference route reaches the same authoritative receipt
    # INSERT without introducing a dispatch-journal fact that would correctly
    # make fence installation lose before this interleaving starts.
    daemon._provider_fn = None
    daemon._effect_fn = None
    evidence, snapshot = _fenced_provider_dispatch_test_inputs(daemon, attempt)
    installer = (
        daemon
        if installer_connection == "shared"
        else _second_execution_store_view(daemon)
    )
    checked = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    real_fence = daemon._fenced_provider_recovery_dispatch_fence

    def pause_after_fence_precheck(selected: object) -> object:
        result = real_fence(selected)
        if threading.current_thread().name == "late-receipt-insert":
            checked.set()
            assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(
        daemon,
        "_fenced_provider_recovery_dispatch_fence",
        pause_after_fence_precheck,
    )

    def late_receipt() -> None:
        try:
            if receipt_kind == "provider":
                daemon.run_provider(attempt)
            else:
                daemon.run_effect(attempt, {"status": "reference"})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=late_receipt,
        name="late-receipt-insert",
        daemon=True,
    )
    thread.start()
    assert checked.wait(timeout=5)
    try:
        installer._install_fenced_provider_recovery_dispatch_fence(
            attempt,
            evidence=evidence,
            snapshot=snapshot,
        )
    finally:
        release.set()
        thread.join(timeout=5)
        if installer is not daemon:
            installer._connection.close()
            installer._connection = None
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DatabaseImplementationConflictError)
    assert daemon.provider_invocation_recorded(
        attempt.attempt_id,
        idempotency_key=f"provider:{attempt.attempt_id}",
    ) is None
    assert daemon.effect_claim_recorded(
        attempt.attempt_id,
        idempotency_key=f"effect:{attempt.attempt_id}",
    ) is None
    assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict()
    assert real_fence(attempt)["state"] == "sealed"
    daemon.close()


def test_historical_death_manifest_is_content_addressed() -> None:
    manifest = {
        "schema": DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_SCHEMA,
        "revision": "pctdd-provider-recovery-2026-09-02",
        "operator_owned": True,
        "historical_only": True,
        "occurrences": [
            dict(item) for item in DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_PINS
        ],
    }
    assert DatabaseImplementationDaemon._database_no_provider_rearm_digest(
        manifest
    ) == DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_ID


def test_historical_same_session_requires_exact_operator_pin_and_hash_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The synthetic operator pin supplies the association between the
    # historical database process ID and the managed PID.  The hash chain
    # independently proves only that the named managed PID later quiesced.
    path = tmp_path / "lane-0" / "test_supervisor_events.jsonl"
    source_before = "1" * 40
    source_after = "2" * 40
    append_jsonl_event(
        path,
        "supervisor_control_plane_update_detected",
        {
            "control_plane_source_revision": source_before,
            "control_plane_current_source_revision": source_after,
            "control_plane_reload_quiescence": {
                "daemon_fence": {
                    "fenced": True,
                    "reason": "managed_daemon_owned_process_fenced",
                    "safe_to_restart": True,
                },
                "markers_removed": True,
                "pid": 4242,
                "quiesced": True,
                "reason": "control_plane_source_changed",
                "remaining_pid": None,
                "supervised_child_alive": False,
                "supervised_child_pid": 4242,
                "terminated": True,
            },
        },
    )
    event = read_jsonl_events(path, repair=False)[0]
    pin = {
        "attempt_id": "attempt:historical",
        "claim_id": "claim:historical",
        "death_event_id": event["event_id"],
        "death_event_relative_path": "lane-0/test_supervisor_events.jsonl",
        "death_event_sequence": 1,
        "death_event_snapshot_id": event["snapshot_id"],
        "death_event_stream_id": event["stream_id"],
        "death_event_timestamp": event["timestamp"],
        "managed_daemon_pid": 4242,
        "migration_credit_id": "sha256:" + "3" * 64,
        "migration_manifest_id": "sha256:" + "4" * 64,
        "occurrence_anchor_event_id": "sha256:" + "5" * 64,
        "occurrence_started_at_ms": 1,
        "owner_session_id": "session:restart-stable",
        "process_instance_id": "process:historical",
        "source_revision_after": source_after,
        "source_revision_before": source_before,
        "task_alias": "PCTDD-006",
        "task_cid": "task:historical",
    }
    other_one = {
        **pin,
        "attempt_id": "attempt:other-one",
        "claim_id": "claim:other-one",
        "task_alias": "PCTDD-007",
        "task_cid": "task:other-one",
    }
    other_two = {
        **pin,
        "attempt_id": "attempt:other-two",
        "claim_id": "claim:other-two",
        "task_alias": "PCTDD-034",
        "task_cid": "task:other-two",
    }
    pins = (pin, other_one, other_two)
    manifest = {
        "schema": DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_SCHEMA,
        "revision": "pctdd-provider-recovery-2026-09-02",
        "operator_owned": True,
        "historical_only": True,
        "occurrences": [dict(item) for item in pins],
    }
    monkeypatch.setattr(
        daemon_module,
        "DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_PINS",
        pins,
    )
    monkeypatch.setattr(
        daemon_module,
        "DATABASE_FENCED_PROVIDER_PREDECESSOR_DEATH_MANIFEST_ID",
        DatabaseImplementationDaemon._database_no_provider_rearm_digest(
            manifest
        ),
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.supervisor_events_path = path
    task = SimpleNamespace(task_alias="PCTDD-006", task_cid="task:historical")
    receipt = {
        "attempt_id": pin["attempt_id"],
        "claim_id": pin["claim_id"],
        "owner_session_id": pin["owner_session_id"],
        "process_instance_id": pin["process_instance_id"],
    }
    evidence = {
        "migration_credit_id": pin["migration_credit_id"],
        "migration_manifest_id": pin["migration_manifest_id"],
        "implementation_started_event_id": pin["occurrence_anchor_event_id"],
    }

    assert daemon._historical_fenced_provider_predecessor_death_proven(
        task=task,
        receipt=receipt,
        evidence=evidence,
    ) is True

    forged = dict(evidence)
    forged["migration_credit_id"] = "sha256:" + "6" * 64
    assert daemon._historical_fenced_provider_predecessor_death_proven(
        task=task,
        receipt=receipt,
        evidence=forged,
    ) is False


def test_four_daemon_processes_claim_distinct_work(tmp_path: Path) -> None:
    markdown = tmp_path / "board.md"
    markdown.write_text(
        "# Board\n\n## DQP-T001 Sample\n\n- Status: todo\n",
        encoding="utf-8",
    )
    original_markdown = markdown.read_text(encoding="utf-8")

    seed = _open_daemon(tmp_path, session="session:seed", markdown_path=markdown)
    try:
        seed.materialize_population(_population(4))
    finally:
        seed.close()

    claimed: list[str] = []
    for index in range(1, 5):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:{index}",
            markdown_path=markdown,
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"session {index} failed to claim"
            claimed.append(attempt.task_cid)
            assert attempt.owner_session_id == f"session:{index}"
            assert attempt.committed_phase == "claimed"
        finally:
            daemon.close()

    assert len(claimed) == 4
    assert len(set(claimed)) == 4

    idle = _open_daemon(tmp_path, session="session:extra", markdown_path=markdown)
    try:
        assert idle.claim_next() is None
        assert idle.markdown_status_write_count == 0
    finally:
        idle.close()
    assert markdown.read_text(encoding="utf-8") == original_markdown


def test_retry_budget_caps_provider_across_fresh_database_portal_epochs(
    tmp_path: Path,
) -> None:
    provider_attempts: list[str] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:bounded-portals",
        provider_fn=failing_portal_provider,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))

        first = daemon.run_once()
        assert first["implementation_result"]["status"] == "failed"
        assert first["implementation_result"]["retry_exhausted"] is False
        first_task = daemon.task_source.get("task:cid:001")
        assert first_task is not None
        assert first_task.status == "retrying"
        assert first_task.body["completion_receipt"]["schema"] == (
            DATABASE_RETRY_BUDGET_SCHEMA
        )
        assert first_task.body["completion_receipt"]["attempts_used"] == 1

        second = daemon.run_once()
        assert second["attempt_id"] != first["attempt_id"]
        assert second["implementation_result"]["status"] == "retry_exhausted"
        assert second["implementation_result"]["retry_exhausted"] is True
        second_task = daemon.task_source.get("task:cid:001")
        assert second_task is not None
        assert second_task.status == "blocked"
        assert second_task.body["completion_receipt"]["attempts_used"] == 2

        backpressure = daemon.run_once()
        assert backpressure["implementation_result"] is None
        assert backpressure["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        assert backpressure["retry_exhausted_task_cids"] == ["task:cid:001"]
        assert backpressure["retry_budget_backpressure"]["schema"] == (
            DATABASE_RETRY_BUDGET_BACKPRESSURE_SCHEMA
        )
        assert backpressure["retry_budget_backpressure"]["tasks"][0][
            "attempts_used"
        ] == 2

        # Each failed database claim gets a fresh private Portal directory in
        # production.  The canonical database receipt, not that disposable
        # directory, owns the total provider budget.
        assert len(provider_attempts) == 2
        assert len(set(provider_attempts)) == 2
    finally:
        daemon.close()


def test_validation_spec_repair_opens_one_fresh_bounded_retry_epoch(
    tmp_path: Path,
) -> None:
    provider_attempts: list[str] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-repair",
        provider_fn=failing_portal_provider,
        max_task_attempts=1,
    )
    try:
        daemon.materialize_population(_population(1))
        exhausted = daemon.run_once()
        assert exhausted["implementation_result"]["retry_exhausted"] is True
        assert len(provider_attempts) == 1

        repaired = _population(1)
        repaired_task = repaired["tasks"][0]
        assert isinstance(repaired_task, dict)
        repaired_task["status"] = "retrying"
        repaired_task["validation_commands"] = [
            {"argv": ["python", "-m", "pytest", "fixed_validation.py"]}
        ]
        daemon.materialize_population(repaired)

        retried = daemon.run_once()
        assert retried["implementation_result"]["retry_exhausted"] is True
        assert len(provider_attempts) == 2
        assert provider_attempts[0] != provider_attempts[1]
    finally:
        daemon.close()


def test_retry_budget_is_global_across_lane_local_coordination_stores(
    tmp_path: Path,
) -> None:
    provider_attempts: list[tuple[str, str]] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append((attempt.owner_session_id, attempt.attempt_id))
        raise DatabasePortalBridgeError("declared_validation_failed")

    seed = _open_daemon(tmp_path / "seed", session="session:seed")
    lane_coordinators = []
    lanes: list[DatabaseImplementationDaemon] = []
    try:
        seed.materialize_population(_population(1))
        for lane_index in range(2):
            lane_root = tmp_path / f"lane-{lane_index}"
            coordinator = open_database_coordinator(
                lane_root / "coordination.duckdb"
            )
            lane_coordinators.append(coordinator)
            lane = DatabaseImplementationDaemon(
                database_path=seed.database_path,
                coordination_path=lane_root / "coordination.duckdb",
                execution_path=lane_root / "execution.duckdb",
                owner_session_id=f"session:lane-{lane_index}",
                authority_mode="embedded",
                task_source_kind="duckdb",
                task_source=seed.task_source,
                coordinator=coordinator,
                provider_fn=failing_portal_provider,
                max_task_attempts=2,
            )
            lanes.append(lane)

        first = lanes[0].run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        second = lanes[1].run_once()
        assert second["implementation_result"]["retry_exhausted"] is True

        for lane in lanes:
            idle = lane.run_once()
            assert idle["implementation_result"] is None
            assert idle["selection_idle_reason"] == (
                "all_selectable_ready_tasks_reached_max_task_attempts"
            )
        assert [owner for owner, _attempt_id in provider_attempts] == [
            "session:lane-0",
            "session:lane-1",
        ]
        assert len({attempt_id for _owner, attempt_id in provider_attempts}) == 2
    finally:
        for lane in lanes:
            lane.close()
        for coordinator in lane_coordinators:
            coordinator.close()
        seed.close()


@pytest.mark.parametrize("replacement_cap", [0, 1, 3])
def test_persisted_retry_policy_rejects_lane_cap_mismatch(
    tmp_path: Path,
    replacement_cap: int,
) -> None:
    first_calls: list[str] = []

    def fail_first(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        first_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    first = _open_daemon(
        tmp_path,
        session="session:policy-origin",
        provider_fn=fail_first,
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        result = first.run_once()
        assert result["implementation_result"]["retry_exhausted"] is False
        persisted = first.task_source.get("task:cid:001")
        assert persisted is not None
        assert persisted.body["completion_receipt"]["max_task_attempts"] == 2
    finally:
        first.close()

    replacement_calls: list[str] = []
    replacement = _open_daemon(
        tmp_path,
        session=f"session:policy-mismatch:{replacement_cap}",
        provider_calls=replacement_calls,
        max_task_attempts=replacement_cap,
    )
    try:
        claims_before = replacement.coordinator.coordination_registry_projection()[
            "task_claim_state_counts"
        ]
        blocked = replacement.run_once()
        assert blocked["implementation_result"] is None
        assert blocked["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = blocked["retry_budget_backpressure"]["tasks"][0]
        assert entry["policy_mismatch"] is True
        assert entry["max_task_attempts"] == 2
        assert entry["configured_max_task_attempts"] == replacement_cap
        assert replacement_calls == []
        assert len(first_calls) == 1
        projection = replacement.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        assert projection["task_claim_state_counts"] == claims_before
        repeated = replacement.run_once()
        assert repeated["unchanged"] is True
        assert repeated["write_count"] == 0
        assert (
            replacement.coordinator.coordination_registry_projection()[
                "task_claim_state_counts"
            ]
            == claims_before
        )
    finally:
        replacement.close()


def test_canonical_cross_lane_claim_cas_loss_is_benign_and_never_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_attempts: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:cas-loser",
        provider_calls=provider_attempts,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))

        def lose_canonical_cas(*_args: object, **_kwargs: object) -> object:
            raise DatabaseTaskSourceConflictError("simulated other-lane winner")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lose_canonical_cas,
        )
        assert daemon.claim_next() is None
        assert provider_attempts == []
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        assert projection["task_claim_state_counts"] == [
            {"state": "released", "count": 1}
        ]
        assert daemon.list_running_attempts() == []
    finally:
        daemon.close()


def test_identical_materializer_replay_cannot_rearm_exhausted_budget(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:materializer-replay",
        provider_fn=fail,
        max_task_attempts=1,
    )
    population = _population(1)
    try:
        daemon.materialize_population(population)
        daemon.run_once()
        assert len(calls) == 1
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "blocked"
        assert task.body["completion_receipt"]["attempts_used"] == 1
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert len(calls) == 1
    finally:
        daemon.close()


def test_unreviewed_nested_portal_rearm_evidence_is_claim_fenced() -> None:
    """An empty shortcut identity is invalid, never merely inapplicable."""

    shortcut = SimpleNamespace(
        task_cid="task:cid:pctdd-034",
        revision=9,
        status="retrying",
        body={
            "completion_receipt": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "nested-portal-rearm-evidence@1"
                ),
                "evidence_id": "",
            }
        },
    )

    assert (
        DatabaseImplementationDaemon._no_provider_rearm_fence_state(shortcut)
        == "invalid"
    )
    assert DatabaseImplementationDaemon._automatic_claim_forbidden(shortcut)

    nested = SimpleNamespace(
        **{
            **vars(shortcut),
            "body": {
                "completion_receipt": {
                    "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                    "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                    "no_provider_rearm_evidence_id": "",
                    "no_provider_rearm_evidence": {
                        "schema": "nested-portal-rearm-evidence@1",
                        "evidence_id": "",
                    },
                }
            },
        }
    )
    assert (
        DatabaseImplementationDaemon._no_provider_rearm_fence_state(nested)
        == "invalid"
    )
    assert DatabaseImplementationDaemon._automatic_claim_forbidden(nested)


def test_identical_materializer_replay_preserves_live_claim_revision(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:live-materializer-replay",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    population = _population(1)
    try:
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        before = daemon.task_source.get(attempt.task_cid)
        assert before is not None and before.status == "in_progress"
        before_receipt = dict(before.body["completion_receipt"])

        daemon.materialize_population(population)

        after = daemon.task_source.get(attempt.task_cid)
        assert after is not None and after.status == "in_progress"
        assert after.revision == before.revision
        assert after.body["completion_receipt"] == before_receipt
        result = daemon.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
    finally:
        daemon.close()


def test_materializer_revision_cas_cannot_overwrite_concurrent_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:materializer-claim-race",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        original_upsert = daemon.task_source._intent.upsert_task
        injected: dict[str, DatabaseTaskAttempt] = {}

        def claim_before_stale_upsert(**kwargs: object) -> object:
            if kwargs.get("task_cid") == "task:cid:001" and not injected:
                attempt = daemon.claim_next()
                assert attempt is not None
                injected["attempt"] = attempt
            return original_upsert(**kwargs)

        monkeypatch.setattr(
            daemon.task_source._intent,
            "upsert_task",
            claim_before_stale_upsert,
        )
        changed = _population(1)
        changed_task = changed["tasks"][0]
        assert isinstance(changed_task, dict)
        changed_task["title"] = "Changed during claim race"

        with pytest.raises(DatabaseTaskSourceConflictError):
            daemon.materialize_population(changed)

        attempt = injected["attempt"]
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None and current.status == "in_progress"
        assert current.body["completion_receipt"]["attempt_id"] == attempt.attempt_id
        assert current.body["completion_receipt"]["attempts_used"] == 1
    finally:
        daemon.close()


def test_identical_singular_validation_replay_preserves_exhausted_budget(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:singular-validation-replay",
        provider_fn=fail,
        max_task_attempts=1,
    )
    population = _population(1)
    population_task = population["tasks"][0]
    assert isinstance(population_task, dict)
    population_task["validation"] = "python -m pytest singular_validation.py"
    try:
        daemon.materialize_population(population)
        exhausted = daemon.run_once()
        assert exhausted["implementation_result"]["retry_exhausted"] is True
        before = daemon.task_source.get("task:cid:001")
        assert before is not None and before.status == "blocked"
        before_receipt = dict(before.body["completion_receipt"])

        daemon.materialize_population(population)

        after = daemon.task_source.get("task:cid:001")
        assert after is not None and after.status == "blocked"
        assert after.body["completion_receipt"] == before_receipt
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert calls == [before_receipt["attempt_id"]]
    finally:
        daemon.close()


def test_unexpected_provider_exception_never_replays_same_attempt(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def explode(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise RuntimeError("provider exploded")

    daemon = _open_daemon(
        tmp_path,
        session="session:provider-exception",
        provider_fn=explode,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        second = daemon.run_once()
        idle = daemon.run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(calls) == 2
        assert len(set(calls)) == 2
    finally:
        daemon.close()


def test_effect_exception_is_unknown_and_blocks_without_replay(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        raise RuntimeError("effect outcome unknown")

    daemon = _open_daemon(
        tmp_path,
        session="session:effect-exception",
        provider_calls=provider_calls,
        effect_fn=effect,
        max_task_attempts=3,
    )

    try:
        daemon.materialize_population(_population(1))
        failed = daemon.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        task = daemon.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["forced_block"] is True
        daemon.run_once()
        assert len(provider_calls) == 1
        assert len(effect_calls) == 1
        same_session = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert same_session == []
        assert daemon.task_source.get("task:cid:001").status == "blocked"
    finally:
        daemon.close()


def test_later_session_does_not_rearm_effect_unknown_outcome(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        raise RuntimeError("effect outcome unknown")

    blocker = _open_daemon(
        tmp_path,
        session="session:effect-exception-block",
        provider_calls=provider_calls,
        effect_fn=effect,
        max_task_attempts=3,
    )
    try:
        blocker.materialize_population(_population(1))
        failed = blocker.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        assert blocker.task_source.get("task:cid:001").status == "blocked"
    finally:
        blocker.close()

    successor_calls: list[str] = []
    successor = _open_daemon(
        tmp_path,
        session="session:effect-exception-rearm",
        provider_calls=successor_calls,
        max_task_attempts=3,
    )
    try:
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert rearms == []
        task = successor.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == "database_unknown_outcome_blocked"
        assert receipt["reason"] == "callback_authority_incomplete_blocked"
        assert receipt["retry_exhausted"] is True
        assert successor.reconcile_blocked_unknown_outcome_tasks() == []
        idle = successor.run_once()
        assert idle["implementation_result"] is None
        attempts = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            ["task:cid:001"],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert successor_calls == []
        assert len(provider_calls) == 1
        assert len(effect_calls) == 1
    finally:
        successor.close()


def test_crash_reconciler_preserves_dispatch_process_for_automatic_rearm(
    tmp_path: Path,
) -> None:
    first = _open_daemon(
        tmp_path,
        session="session:unknown-dispatch-crash",
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        dispatch_process = first.process_instance_id
        first._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        claimed = first.task_source.get(attempt.task_cid)
        assert claimed is not None and claimed.status == "in_progress"
        assert (
            claimed.body["completion_receipt"]["process_instance_id"]
            == dispatch_process
        )
    finally:
        first.close()

    successor = _open_daemon(
        tmp_path,
        session="session:unknown-dispatch-crash",
        max_task_attempts=2,
    )
    try:
        running = successor.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        assert successor.process_instance_id != dispatch_process

        failed, receipt = successor._finalize_failed_attempt(
            running,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )

        assert failed.status == "failed"
        failed_phase = next(
            phase
            for phase in successor.phase_history(failed.attempt_id)
            if phase["phase"] == "failed"
        )
        assert "terminal_reconciliation" not in failed_phase["body"]
        assert receipt["process_instance_id"] == dispatch_process
        assert receipt["reconciled_by_process_instance_id"] == (
            successor.process_instance_id
        )
        blocked = successor.task_source.get(attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        rearms = successor.reconcile_blocked_unknown_outcome_tasks()

        assert len(rearms) == 1
        assert rearms[0]["task_cid"] == attempt.task_cid
        rearmed = successor.task_source.get(attempt.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        assert rearmed.body["completion_receipt"]["operation"] == (
            DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION
        )
    finally:
        successor.close()


def test_exact_interrupted_implementation_refunds_attempt_two_without_widening_unknown_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match the live attempt-2/count-1 recovery without a generic rearm."""

    first = _open_daemon(
        tmp_path,
        session="session:interrupted-refund",
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        attempt_one = first.claim_next()
        assert attempt_one is not None
        first._begin_callback_dispatch(
            attempt_one,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_one.attempt_id}",
        )
        first._finalize_failed_attempt(
            attempt_one,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
    finally:
        first.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:interrupted-refund",
        max_task_attempts=2,
    )
    callback_calls: list[str] = []
    try:
        generic = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(generic) == 1
        assert generic[0]["unknown_outcome_rearm_count"] == 1
        attempt_two = daemon.claim_next()
        assert attempt_two is not None
        assert attempt_two.attempt_number == 2
        daemon._begin_callback_dispatch(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
            outcome="raised",
            body={
                "exception_type": "DatabasePortalBridgeError",
                "message": "interrupted nested implementation recovered",
            },
        )
        terminal_reconciliation = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            "attempt_id": attempt_two.attempt_id,
            "claim_id": attempt_two.claim_id,
            "task_cid": attempt_two.task_cid,
            "attempt_number": attempt_two.attempt_number,
            "owner_session_id": attempt_two.owner_session_id,
            "lease_id": attempt_two.lease_id,
            "fencing_token": attempt_two.fencing_token,
            "fence_epoch": attempt_two.fence_epoch,
            "binding_id": "sha256:" + "3" * 64,
            "nested_state_digest": "sha256:" + "a" * 64,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": "database_daemon_startup",
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": "sha256:" + "9" * 64,
            "commit_barrier_receipt_id": "sha256:" + "b" * 64,
        }
        terminal_reconciliation["evidence_id"] = content_identity(
            terminal_reconciliation
        )
        failed, receipt = daemon._finalize_failed_attempt(
            attempt_two,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
            reconciliation_evidence=terminal_reconciliation,
        )
        assert failed.status == "failed"
        assert receipt["attempt_number"] == 2
        assert receipt["attempts_used"] == 1
        assert receipt["unknown_outcome_rearm_count"] == 1

        task = daemon.task_source.get(attempt_two.task_cid)
        assert task is not None and task.status == "blocked"
        receipt = dict(task.body["completion_receipt"])
        assert receipt["terminal_reconciliation"] == terminal_reconciliation
        terminal_reconciliation_evidence_id = str(
            terminal_reconciliation["evidence_id"]
        )
        evidence = {
            "schema": (
                DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
            ),
            "attempt_id": attempt_two.attempt_id,
            "claim_id": attempt_two.claim_id,
            "task_cid": attempt_two.task_cid,
            "task_alias": attempt_two.task_alias,
            "attempt_number": attempt_two.attempt_number,
            "owner_session_id": attempt_two.owner_session_id,
            "lease_id": attempt_two.lease_id,
            "fencing_token": attempt_two.fencing_token,
            "fence_epoch": attempt_two.fence_epoch,
            "attempt_root_key": hashlib.sha256(
                attempt_two.attempt_id.encode("utf-8")
            ).hexdigest()[:24],
            "attempt_authority_root_digest": "sha256:" + "1" * 64,
            "attempt_root_digest": "sha256:" + "2" * 64,
            "binding_id": "sha256:" + "3" * 64,
            "binding_admission_id": content_identity(
                {"binding-admission": "current"}
            ),
            "binding_admission_digest": "sha256:" + "4" * 64,
            "projection_immutable_digest": "sha256:" + "5" * 64,
            "nested_task_cid": "nested:task:current",
            "nested_attempt": 1,
            "terminal_reconciliation_evidence_id": (
                terminal_reconciliation_evidence_id
            ),
            "first_clear_receipt_id": "sha256:" + "6" * 64,
            "interrupted_retry_evidence_id": "sha256:" + "7" * 64,
            "interrupted_retry_id": content_identity(
                {"interrupted-retry": "current"}
            ),
            "state_recovery_event_id": "sha256:" + "8" * 64,
            "claim_release_receipt_id": content_identity(
                {"claim-release": "current"}
            ),
            "prepared_reconciliation_receipt_id": "sha256:" + "9" * 64,
            "commit_barrier_receipt_id": "sha256:" + "b" * 64,
            "state_digest": "sha256:" + "a" * 64,
            "outer_block_receipt_digest": (
                daemon._database_no_provider_rearm_digest(receipt)
            ),
            "provider_dispatched": False,
            "implementation_dispatched": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "recovery_terminal": True,
            "retained_candidate_disposition": "preserved_unvalidated",
        }
        authorization = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-implementation-"
                "rearm-authorization@1"
            ),
            **{
                name: evidence[name]
                for name in (
                    "attempt_id",
                    "claim_id",
                    "task_cid",
                    "attempt_number",
                    "owner_session_id",
                    "lease_id",
                    "fencing_token",
                    "fence_epoch",
                    "binding_id",
                    "binding_admission_id",
                    "binding_admission_digest",
                    "projection_immutable_digest",
                    "nested_task_cid",
                    "nested_attempt",
                    "terminal_reconciliation_evidence_id",
                    "first_clear_receipt_id",
                    "interrupted_retry_evidence_id",
                    "interrupted_retry_id",
                    "state_recovery_event_id",
                    "claim_release_receipt_id",
                    "prepared_reconciliation_receipt_id",
                    "commit_barrier_receipt_id",
                    "state_digest",
                    "outer_block_receipt_digest",
                )
            },
        }
        evidence["rearm_authorization_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                authorization,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        evidence["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert daemon._valid_no_provider_rearm_evidence(
            evidence,
            task=task,
            original=receipt,
            expected_evidence_id=evidence["evidence_id"],
        )
        for content_id_field in (
            "binding_admission_id",
            "interrupted_retry_id",
            "claim_release_receipt_id",
        ):
            wrong_profile = {**evidence, content_id_field: "sha256:" + "e" * 64}
            wrong_authorization = {
                **authorization,
                content_id_field: wrong_profile[content_id_field],
            }
            wrong_profile["rearm_authorization_id"] = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        wrong_authorization,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            )
            wrong_unsigned = dict(wrong_profile)
            wrong_unsigned.pop("evidence_id")
            wrong_profile["evidence_id"] = "sha256:" + hashlib.sha256(
                json.dumps(
                    wrong_unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            assert not daemon._valid_no_provider_rearm_evidence(
                wrong_profile,
                task=task,
                original=receipt,
                expected_evidence_id=wrong_profile["evidence_id"],
            )
        tampered_authorization = {
            **evidence,
            "rearm_authorization_id": "sha256:" + "c" * 64,
        }
        tampered_unsigned = dict(tampered_authorization)
        tampered_unsigned.pop("evidence_id")
        tampered_authorization["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                tampered_unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert not daemon._valid_no_provider_rearm_evidence(
            tampered_authorization,
            task=task,
            original=receipt,
            expected_evidence_id=tampered_authorization["evidence_id"],
        )
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            lambda _task, _receipt: dict(evidence),
        )
        monkeypatch.setattr(
            daemon,
            "_resume_attempt_without_process_crash",
            lambda _attempt: callback_calls.append("callback"),
        )

        refunded = daemon.run_once()

        assert refunded["selection_idle_reason"] == (
            "database_unknown_outcomes_rearmed"
        )
        assert refunded["implementation_result"] is None
        assert callback_calls == []
        assert len(refunded["unknown_outcome_rearms"]) == 1
        outcome = refunded["unknown_outcome_rearms"][0]
        assert outcome["previous_attempt_id"] == attempt_two.attempt_id
        assert outcome["unknown_outcome_rearm_count"] == 1
        assert outcome["nested_event_head_id"] == (
            evidence["state_recovery_event_id"]
        )
        rearmed = daemon.task_source.get(attempt_two.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        rearm_receipt = rearmed.body["completion_receipt"]
        assert rearm_receipt["attempts_used"] == 0
        assert rearm_receipt["retry_exhausted"] is False
        # The exact no-provider refund is not a second generic allowance.
        assert rearm_receipt["unknown_outcome_rearm_count"] == 1
        assert rearm_receipt["no_provider_rearm_evidence_id"] == (
            evidence["evidence_id"]
        )
        for numeric_field in ("attempts_used", "attempt_number"):
            for malformed in (None, "not-an-integer", True, [], {}):
                malformed_receipt = dict(rearm_receipt)
                malformed_original = dict(
                    malformed_receipt["no_provider_rearm_original_block_receipt"]
                )
                malformed_original[numeric_field] = malformed
                malformed_receipt[
                    "no_provider_rearm_original_block_receipt"
                ] = malformed_original
                malformed_task = SimpleNamespace(
                    task_cid=rearmed.task_cid,
                    task_alias=rearmed.task_alias,
                    revision=rearmed.revision,
                    status=rearmed.status,
                    body={
                        **dict(rearmed.body),
                        "completion_receipt": malformed_receipt,
                    },
                )
                assert (
                    daemon._no_provider_rearm_fence_state(malformed_task)
                    == "invalid"
                )
                assert daemon._automatic_claim_forbidden(malformed_task)
        assert daemon.reconcile_blocked_unknown_outcome_tasks() == []
        assert callback_calls == []
    finally:
        daemon.close()


def test_interrupted_rearm_evidence_binds_terminal_barrier_and_recovery_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recomputed outer link cannot detach the immutable recovery barrier."""

    attempt = SimpleNamespace(
        attempt_id="attempt:interrupted:exact",
        claim_id="claim:interrupted:exact",
        task_cid="task:cid:pctdd-034",
        task_alias="PCTDD-034",
        attempt_number=2,
        owner_session_id="session:interrupted:exact",
        lease_id="lease:interrupted:exact",
        fencing_token=7,
        fence_epoch=3,
        status="failed",
        committed_phase="failed",
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths = bridge._paths(attempt)
    state_digest = "sha256:" + "1" * 64
    binding = {
        "task_alias": attempt.task_alias,
        "binding_id": "sha256:" + "2" * 64,
        "projection_immutable_digest": "sha256:" + "3" * 64,
    }
    exact_attempt = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "attempt_number": attempt.attempt_number,
        "owner_session_id": attempt.owner_session_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    durable_binding = {
        **exact_attempt,
        **binding,
        "stage": "portal_entered",
        "record_id": content_identity({"binding-admission": "exact"}),
    }
    released_attempt = {
        "released_from": 1,
        "released_to": 0,
        "event_id": "sha256:" + "5" * 64,
    }
    replay = {
        "reconciled": True,
        "blocked": False,
        "reason": "interrupted_implementation_recovered_for_retry",
        "task_id": attempt.task_alias,
        "canonical_task_cid": "nested:task:current",
        "attempt": 1,
        "task_claim_reconciliation": {
            "reconciled": True,
            "blocked": False,
            "reason": "quiesced_task_claim_released",
            "task_id": attempt.task_alias,
            "task_status": "todo",
            "released_unfinished_retry_id": content_identity(
                {"interrupted-retry": "exact"}
            ),
            "receipt_id": content_identity({"claim-release": "exact"}),
            "released_unfinished_attempt": released_attempt,
        },
        "provider_dispatched": False,
        "implementation_dispatched": False,
        "acceptance_inferred": False,
        "retained_candidate_disposition": "preserved_unvalidated",
        "stale_lock_cleared": False,
    }
    recovery = {
        **replay,
        "provider_forbidden_terminal_recovery": {
            "applicable": False,
            "blocked": False,
            "implementation_dispatched": False,
            "provider_dispatched": False,
            "reason": "provider_forbidden_terminal_recovery_not_applicable",
            "reconciled": False,
        },
    }
    prepared_id = "sha256:" + "8" * 64
    barrier_id = "sha256:" + "9" * 64
    trigger = "database_daemon_startup"
    prepared = {
        "receipt_id": prepared_id,
        "binding_id": binding["binding_id"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "trigger": trigger,
        "nested_state": {
            "present": True,
            "active": False,
            "active_task_id": "",
            "active_attempt": 0,
            "active_phase": "",
            "state_path": str(paths.state),
            "state_digest": state_digest,
        },
        "provider_runner_fence": {
            "safe_to_restart": True,
            "applicable": False,
            "fenced": False,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
        "portal_reconciliation": recovery,
    }
    barrier = {
        **prepared,
        "receipt_id": barrier_id,
        "prepared_reconciliation_receipt_id": prepared_id,
    }
    source_receipt = {
        "binding_id": binding["binding_id"],
        "receipt_id": "sha256:" + "a" * 64,
    }
    source = {
        "reconciliation_receipt": source_receipt,
        "evidence_id": "sha256:" + "b" * 64,
    }
    link = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-terminal-reconciliation-link@1"
        ),
        **exact_attempt,
        "binding_id": binding["binding_id"],
        "nested_state_digest": state_digest,
        "nested_reason": "nested_portal_attempt_reconciled",
        "nested_reconciled": True,
        "trigger": trigger,
        "intended_database_disposition": "blocked_unknown_outcome",
        "prepared_reconciliation_receipt_id": prepared_id,
        "commit_barrier_receipt_id": barrier_id,
    }
    link["evidence_id"] = content_identity(link)
    calls: list[str] = []
    replay_holder: dict[str, object] = {}

    class RetryOnlyPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, object]:
            return dict(replay_holder)

        def reconcile_provider_forbidden_terminal_result(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": (
                    "provider_forbidden_terminal_recovery_not_applicable"
                ),
                "reconciled": False,
            }

        def reconcile_interrupted_database_implementation_attempt(
            self,
            evidence: object,
        ) -> dict[str, object]:
            assert evidence == source
            calls.append("recovery")
            return dict(replay)

        def close_event_runtime(self) -> None:
            return None

    bridge.portal_factory = lambda _paths, _alias: RetryOnlyPortal()
    bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _value: None)
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    state = {
        "implementation_in_progress": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
    }
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: (dict(state), state_digest),
    )

    outer_receipt = {**exact_attempt, "terminal_reconciliation": link}
    evidence = bridge._interrupted_implementation_rearm_evidence(
        attempt,
        outer_receipt,
    )

    assert evidence is not None
    assert evidence["rearm_authorization_id"].startswith("sha256:")
    assert evidence["terminal_reconciliation_evidence_id"] == link["evidence_id"]
    assert evidence["commit_barrier_receipt_id"] == barrier_id
    assert calls == ["recovery"]
    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=attempt,
        original=outer_receipt,
        expected_evidence_id=evidence["evidence_id"],
    )

    for changed in (
        {"attempt_number": 2},
        {"attempt_number": 4},
        {"attempts_used": 2},
        {"rearm_count": 1},
    ):
        budget = {
            "task_alias": "PCTDD-005",
            "attempt_number": 3,
            "attempts_used": 1,
            "rearm_count": 0,
            **changed,
        }
        assert not (
            _database_portal_historical_interrupted_state_transition_budget_matches(
                **budget,
            )
        )

    for numeric_field in ("attempt_number", "fencing_token", "fence_epoch"):
        numeric_alias = dict(link)
        numeric_alias[numeric_field] = float(numeric_alias[numeric_field])
        assert bridge._interrupted_implementation_rearm_evidence(
            attempt,
            {**exact_attempt, "terminal_reconciliation": numeric_alias},
        ) is None
    assert calls == ["recovery"]

    malformed_nested = {
        **prepared,
        "nested_state": {
            **dict(prepared["nested_state"]),
            "active_attempt": False,
        },
    }
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(malformed_nested)
            if receipt_id == prepared_id
            else {
                **dict(malformed_nested),
                "receipt_id": barrier_id,
                "prepared_reconciliation_receipt_id": prepared_id,
            }
        ),
    )
    assert bridge._interrupted_implementation_rearm_evidence(
        attempt,
        outer_receipt,
    ) is None
    assert calls == ["recovery"]
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )

    tampered_link = {**link, "trigger": "tampered-trigger"}
    tampered_link.pop("evidence_id")
    tampered_link["evidence_id"] = content_identity(tampered_link)
    assert bridge._interrupted_implementation_rearm_evidence(
        attempt,
        {**exact_attempt, "terminal_reconciliation": tampered_link},
    ) is None
    assert calls == ["recovery"]

    persisted_bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "persisted-barrier-attempts",
        portal_factory=lambda _paths, _alias: RetryOnlyPortal(),
    )
    persisted_paths = persisted_bridge._paths(attempt)
    persisted_paths.reconciliation.mkdir(parents=True)
    persisted_nested = {
        "present": True,
        "active": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
        "state_path": str(persisted_paths.state),
        "state_digest": state_digest,
    }
    persisted_core = {
        "binding_id": binding["binding_id"],
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "nested_state": persisted_nested,
        "provider_runner_fence": prepared["provider_runner_fence"],
        "portal_reconciliation": recovery,
        "terminal_provider_evidence": False,
    }
    persisted_prepared = persisted_bridge.persist_reconciliation_receipt(
        attempt,
        {
            **persisted_core,
            "stage": "prepared",
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "reconciled_at": "2026-01-01T00:00:00+00:00",
        },
    )
    persisted_bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(
        persisted_bridge,
        "_read_binding",
        lambda _path: dict(binding),
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_verify_binding_identity",
        lambda _value: None,
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_strict_state_record",
        lambda _path: (dict(state), state_digest),
    )
    for changed_field, changed_value in (
        ("binding_id", "sha256:" + "f" * 64),
        ("reason", "contradictory-reason"),
        ("reconciled", False),
        ("blocked", True),
        (
            "nested_state",
            {**persisted_nested, "state_digest": "sha256:" + "e" * 64},
        ),
        (
            "provider_runner_fence",
            {**prepared["provider_runner_fence"], "safe_to_restart": False},
        ),
        (
            "portal_reconciliation",
            {**recovery, "reason": "contradictory-recovery"},
        ),
        ("reconciled_at", "2026-01-01T00:00:01+00:00"),
    ):
        contradictory_core = {
            **persisted_core,
            changed_field: changed_value,
        }
        contradictory_barrier = (
            persisted_bridge.persist_reconciliation_receipt(
                attempt,
                {
                    **contradictory_core,
                    "stage": "commit_barrier",
                    "trigger": trigger,
                    "intended_database_disposition": (
                        "blocked_unknown_outcome"
                    ),
                    "prepared_reconciliation_receipt_id": (
                        persisted_prepared["receipt_id"]
                    ),
                    "reconciled_at": (
                        changed_value
                        if changed_field == "reconciled_at"
                        else "2026-01-01T00:00:00+00:00"
                    ),
                },
            )
        )
        contradictory_link = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            **exact_attempt,
            "binding_id": binding["binding_id"],
            "nested_state_digest": state_digest,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": persisted_prepared[
                "receipt_id"
            ],
            "commit_barrier_receipt_id": contradictory_barrier["receipt_id"],
        }
        contradictory_link["evidence_id"] = content_identity(
            contradictory_link
        )
        assert persisted_bridge._interrupted_implementation_rearm_evidence(
            attempt,
            {
                **exact_attempt,
                "terminal_reconciliation": contradictory_link,
            },
        ) is None
    assert calls == ["recovery"]


def _historical_interrupted_state_transition_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build the exact immutable PCTDD-005 attempt-three migration tuple."""

    pin = DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_PIN
    attempt = SimpleNamespace(
        attempt_id=pin["attempt_id"],
        claim_id=pin["claim_id"],
        task_cid=pin["task_cid"],
        task_alias=pin["task_alias"],
        attempt_number=pin["attempt_number"],
        owner_session_id=pin["owner_session_id"],
        lease_id=pin["lease_id"],
        fencing_token=pin["fencing_token"],
        fence_epoch=pin["fence_epoch"],
        status="failed",
        committed_phase="failed",
    )

    def provider_must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("historical state-transition replay dispatched")

    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "historical-state-transition-attempts",
        portal_factory=provider_must_not_run,
    )
    paths = bridge._paths(attempt)
    nested_task_cid = pin["nested_task_cid"]
    interrupted_retry_id = "baguqeera" + "b" * 52
    claim_release_receipt_id = "baguqeera" + "c" * 52
    projection = "PCTDD-005 immutable projection\n"
    projection_digest = "sha256:" + hashlib.sha256(
        projection.encode("utf-8")
    ).hexdigest()
    binding = {
        "task_alias": "PCTDD-005",
        "task_revision": pin["task_revision"],
        "binding_id": pin["binding_id"],
        "projection_immutable_digest": projection_digest,
    }
    exact_attempt = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "attempt_number": attempt.attempt_number,
        "owner_session_id": attempt.owner_session_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    durable_binding = {
        **exact_attempt,
        "binding_id": binding["binding_id"],
        "projection_immutable_digest": projection_digest,
        "stage": "portal_entered",
        "record_id": content_identity(
            {"historical-state-transition-binding": "PCTDD-005"}
        ),
    }
    identity = {
        "task_id": "PCTDD-005",
        "canonical_task_key": "task-key:pctdd-005",
        "canonical_task_cid": nested_task_cid,
        "board_namespace": (
            "parallel-content-sealing-proof-carrying-tdd-v1"
        ),
    }
    state = {
        "implementation_in_progress": False,
        "active_task_id": "",
        "active_task_key": "",
        "active_task_cid": "",
        "active_task_title": "",
        "active_task_track": "",
        "active_task_started_at": "",
        "active_attempt": 0,
        "active_phase": "",
        "active_phase_started_at": "",
        "active_phase_detail": "",
        "active_log_path": "",
        "active_worktree_path": "",
        "active_branch": "",
        "active_provider_runner": {},
        "implementation_attempts": {},
        "implementation_attempts_by_cid": {},
    }
    reconstructed_pre_state = {
        **state,
        "implementation_attempts": {"PCTDD-005": 1},
        "implementation_attempts_by_cid": {nested_task_cid: 1},
    }
    pre_state_bytes = (
        json.dumps(reconstructed_pre_state, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    pre_state_digest = "sha256:" + hashlib.sha256(
        pre_state_bytes
    ).hexdigest()
    post_state_bytes = (
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    post_state_digest = "sha256:" + hashlib.sha256(
        post_state_bytes
    ).hexdigest()
    source_evidence_id = "sha256:" + "2" * 64
    source_receipt_id = "sha256:" + "3" * 64
    preparation_event_id = pin["retry_preparation_event_id"]
    state_recovery_event_id = pin["state_recovery_event_id"]
    claim_release_event_id = pin["claim_release_event_id"]
    released_attempt = {
        "released_from": 1,
        "released_to": 0,
        "event_id": state_recovery_event_id,
    }
    claim_release = {
        "reconciled": True,
        "blocked": False,
        "reason": "quiesced_task_claim_released",
        "task_id": "PCTDD-005",
        "canonical_task_key": identity["canonical_task_key"],
        "canonical_task_cid": nested_task_cid,
        "board_namespace": identity["board_namespace"],
        "task_status": "todo",
        "attempt": 1,
        "released_unfinished_retry_id": interrupted_retry_id,
        "receipt_id": claim_release_receipt_id,
        "released_unfinished_attempt": released_attempt,
        "claim_id": "nested-claim:pctdd-005",
        "claim_lease_id": "nested-lease:pctdd-005",
        "lifecycle_record_id": "lifecycle:pctdd-005",
        "lifecycle_fence": 3,
    }
    recovery = {
        "reconciled": True,
        "blocked": False,
        "reason": "interrupted_implementation_recovered_for_retry",
        "task_id": "PCTDD-005",
        "canonical_task_cid": nested_task_cid,
        "attempt": 1,
        "task_claim_reconciliation": claim_release,
        "provider_dispatched": False,
        "implementation_dispatched": False,
        "acceptance_inferred": False,
        "retained_candidate_disposition": "preserved_unvalidated",
        "stale_lock_cleared": False,
        "provider_forbidden_terminal_recovery": {
            "applicable": False,
            "blocked": False,
            "implementation_dispatched": False,
            "provider_dispatched": False,
            "reason": "provider_forbidden_terminal_recovery_not_applicable",
            "reconciled": False,
        },
    }
    source_nested = {
        "active_task_id": "PCTDD-005",
        "active_attempt": 1,
        "active_worktree_path": "/isolated/PCTDD-005",
        "active_branch": "agent/PCTDD-005",
    }
    source = {
        "evidence_id": source_evidence_id,
        "reconciliation_receipt": {
            "binding_id": binding["binding_id"],
            "receipt_id": source_receipt_id,
            "nested_state": source_nested,
            "portal_reconciliation": {
                "task_claim_reconciliation": claim_release,
            },
        },
    }
    current_nested = {
        "present": True,
        "active": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
        "state_path": str(paths.state),
        "state_digest": pre_state_digest,
    }
    prepared_id = pin["prepared_reconciliation_receipt_id"]
    barrier_id = pin["commit_barrier_receipt_id"]
    prepared = {
        "receipt_id": prepared_id,
        "binding_id": binding["binding_id"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "trigger": "database_daemon_startup",
        "nested_state": current_nested,
        "provider_runner_fence": {
            "safe_to_restart": True,
            "applicable": False,
            "fenced": False,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
        "portal_reconciliation": recovery,
    }
    barrier = {
        **prepared,
        "receipt_id": barrier_id,
        "prepared_reconciliation_receipt_id": prepared_id,
    }
    events = [
        {
            "type": "interrupted_implementation_retry_prepared",
            "sequence": 1,
            "event_id": preparation_event_id,
            "previous_event_id": "",
            "task_id": "PCTDD-005",
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": nested_task_cid,
            "board_namespace": identity["board_namespace"],
            "attempt": 1,
            "database_evidence_id": source_evidence_id,
            "database_receipt_id": source_receipt_id,
            "interrupted_retry_id": interrupted_retry_id,
            "workspace_path": source_nested["active_worktree_path"],
            "branch": source_nested["active_branch"],
            "claim_id": claim_release["claim_id"],
            "claim_lease_id": claim_release["claim_lease_id"],
            "lifecycle_record_id": claim_release["lifecycle_record_id"],
            "lifecycle_fence": claim_release["lifecycle_fence"],
        },
        {
            "type": "implementation_state_recovered",
            "sequence": 2,
            "event_id": state_recovery_event_id,
            "previous_event_id": preparation_event_id,
            "task_id": "PCTDD-005",
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": nested_task_cid,
            "board_namespace": identity["board_namespace"],
            "attempt": 1,
            "reason": "inflight_process_missing",
            "finished_attempt": False,
            "interrupted_retry_id": interrupted_retry_id,
            "attempt_recovery": {
                "attempt": 1,
                "canonical_task_cid": nested_task_cid,
                "consumed": False,
                "previous_cid_count": 1,
                "previous_display_count": 1,
                "released": True,
                "released_to": 0,
                "task_id": "PCTDD-005",
            },
        },
        {
            "type": "implementation_task_claim_released",
            "sequence": 3,
            "event_id": claim_release_event_id,
            "previous_event_id": state_recovery_event_id,
            "task_id": "PCTDD-005",
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": nested_task_cid,
            "board_namespace": identity["board_namespace"],
            "attempt": 1,
            "reason": "quiesced_task_claim_released",
            "reconciled": True,
            "blocked": False,
            "task_status": "todo",
            "released_unfinished_retry_id": interrupted_retry_id,
            "released_unfinished_attempt": released_attempt,
            "receipt_id": claim_release_receipt_id,
            "claim_id": claim_release["claim_id"],
            "claim_lease_id": claim_release["claim_lease_id"],
            "lifecycle_record_id": claim_release["lifecycle_record_id"],
            "lifecycle_fence": claim_release["lifecycle_fence"],
        },
    ]
    snapshot_holder = {
        "value": {
            "binding": binding,
            "projection": projection,
            "state": state,
            "state_digest": post_state_digest,
            "events": events,
            "manifest": {
                "stream_id": "event-log:sha256:" + "9" * 64,
                "snapshot_id": "event-log-snapshot:sha256:" + "a" * 64,
                "manifest_digest": "sha256:" + "b" * 64,
                "latest_sequence": 3,
                "last_event_id": claim_release_event_id,
            },
        }
    }
    link = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-terminal-reconciliation-link@1"
        ),
        **exact_attempt,
        "binding_id": binding["binding_id"],
        "nested_state_digest": pre_state_digest,
        "nested_reason": "nested_portal_attempt_reconciled",
        "nested_reconciled": True,
        "trigger": "database_daemon_startup",
        "intended_database_disposition": "blocked_unknown_outcome",
        "prepared_reconciliation_receipt_id": prepared_id,
        "commit_barrier_receipt_id": barrier_id,
    }
    link["evidence_id"] = content_identity(link)
    receipt = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        **exact_attempt,
        "operation": "database_unknown_outcome_blocked",
        "reason": "callback_authority_incomplete_blocked",
        "attempts_used": 1,
        "unknown_outcome_rearm_count": 0,
        "retry_exhausted": True,
        "forced_block": True,
        "authority_outcome": "unknown",
        "process_instance_id": "process:pctdd-005:attempt-3",
        "terminal_reconciliation": link,
    }

    bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _value: None)
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    monkeypatch.setattr(
        bridge,
        "_pinned_no_provider_snapshot",
        lambda _paths: snapshot_holder["value"],
    )
    monkeypatch.setattr(
        bridge,
        "_projection_task_identity",
        lambda _paths, _binding, _projection: dict(identity),
    )
    monkeypatch.setattr(
        bridge,
        "_verify_nested_state_identity",
        lambda *_args, **_kwargs: {
            "present": True,
            "active": False,
        },
    )
    return SimpleNamespace(
        attempt=attempt,
        binding=binding,
        bridge=bridge,
        receipt=receipt,
        snapshot_holder=snapshot_holder,
    )


def test_historical_interrupted_state_transition_rearm_admits_exact_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _historical_interrupted_state_transition_case(
        tmp_path,
        monkeypatch,
    )

    evidence = case.bridge._interrupted_implementation_rearm_evidence(
        case.attempt,
        case.receipt,
        expected_evidence_schema=(
            DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA
        ),
    )

    assert evidence is not None
    assert evidence["schema"] == (
        DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA
    )
    assert set(evidence) == set(
        DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_FIELDS
    )
    assert evidence["attempt_number"] == 3
    assert evidence["nested_attempt"] == 1
    assert evidence["pre_display_attempt_count"] == 1
    assert evidence["post_display_attempt_count"] == 0
    assert evidence["pre_cid_attempt_count"] == 1
    assert evidence["post_cid_attempt_count"] == 0
    assert evidence["attempt_consumed"] is False
    assert evidence["historical_transition_only"] is True
    assert evidence["nested_state_quiescent"] is True
    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=case.attempt,
        original=case.receipt,
        expected_evidence_id=evidence["evidence_id"],
    )

    routed_evidence = case.bridge.no_provider_dispatch_rearm_evidence(
        case.attempt,
        outer_block_receipt=case.receipt,
    )
    assert routed_evidence == evidence

    malformed_event_count = {**evidence, "event_count": "3"}
    assert not DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        malformed_event_count,
        task=case.attempt,
        original=case.receipt,
        expected_evidence_id=evidence["evidence_id"],
    )
    missing_rearm_count = dict(case.receipt)
    missing_rearm_count.pop("unknown_outcome_rearm_count")
    assert not DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=case.attempt,
        original=missing_rearm_count,
        expected_evidence_id=evidence["evidence_id"],
    )


def test_historical_interrupted_state_transition_rearm_rejects_suffix_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _historical_interrupted_state_transition_case(
        tmp_path,
        monkeypatch,
    )
    snapshot = case.snapshot_holder["value"]
    events = list(snapshot["events"])
    recovery_event = dict(events[-2])
    recovery_event["attempt_recovery"] = {
        **dict(recovery_event["attempt_recovery"]),
        "consumed": True,
    }
    events[-2] = recovery_event
    case.snapshot_holder["value"] = {**snapshot, "events": events}

    assert case.bridge._interrupted_implementation_rearm_evidence(
        case.attempt,
        case.receipt,
        expected_evidence_schema=(
            DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA
        ),
    ) is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("task_revision", 14),
        ("binding_id", "sha256:" + "f" * 64),
    ),
)
def test_historical_interrupted_state_transition_rearm_rejects_unsealed_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    case = _historical_interrupted_state_transition_case(
        tmp_path,
        monkeypatch,
    )
    case.binding[field] = replacement

    assert case.bridge._interrupted_implementation_rearm_evidence(
        case.attempt,
        case.receipt,
        expected_evidence_schema=(
            DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA
        ),
    ) is None


def test_terminal_quiescent_advance_contract_is_exported_and_fail_closed() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        database_portal_bridge,
    )

    expected_exports = {
        "DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_AUTHORIZATION_SCHEMA",
        "DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_FIELDS",
        "DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA",
        "DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_PIN",
        "DatabasePortalTerminalQuiescentStateAdvanced",
    }
    assert expected_exports <= set(database_portal_bridge.__all__)

    payload = {
        "implementation_in_progress": False,
        "active_task_id": "",
        "active_task_key": "",
        "active_task_cid": "",
        "active_task_title": "",
        "active_task_track": "",
        "active_task_started_at": "",
        "active_attempt": 0,
        "active_phase": "",
        "active_phase_started_at": "",
        "active_phase_detail": "",
        "active_log_path": "",
        "active_worktree_path": "",
        "active_branch": "",
        "active_provider_runner": {},
    }
    quiescent = {"present": True, "active": False}
    assert DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        payload,
        quiescent,
    )
    legacy_sparse_payload = {
        "implementation_in_progress": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
    }
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        legacy_sparse_payload,
        quiescent,
    )
    assert DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        legacy_sparse_payload,
        quiescent,
        allow_legacy_sparse=True,
    )
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        {**legacy_sparse_payload, "unexpected": "field"},
        quiescent,
        allow_legacy_sparse=True,
    )
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        {**legacy_sparse_payload, "active_attempt": False},
        quiescent,
        allow_legacy_sparse=True,
    )
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        {**legacy_sparse_payload, "active_attempt": 0.0},
        quiescent,
        allow_legacy_sparse=True,
    )
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        payload,
        {"present": False, "active": False},
    )
    assert not DatabasePortalExecutionBridge._nested_state_is_exactly_quiescent(
        {**payload, "implementation_in_progress": True},
        {"present": True, "active": True},
    )


def test_stale_dispatch_migration_rearm_uses_exact_policy_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical release suffix gets only its versioned refund proof."""

    attempt = SimpleNamespace(
        attempt_id="attempt:stale-dispatch-migration",
        claim_id="claim:stale-dispatch-migration",
        task_cid="task:cid:pctdd-034",
        task_alias="PCTDD-034",
        attempt_number=2,
        owner_session_id="session:stale-dispatch-migration",
        lease_id="lease:stale-dispatch-migration",
        fencing_token=11,
        fence_epoch=5,
        status="failed",
        committed_phase="failed",
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "migration-attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths = bridge._paths(attempt)
    pre_state_digest = "sha256:" + "1" * 64
    post_state_digest = "sha256:" + "2" * 64
    binding = {
        "task_alias": attempt.task_alias,
        "binding_id": "sha256:" + "3" * 64,
        "projection_immutable_digest": "sha256:" + "4" * 64,
    }
    exact_attempt = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "attempt_number": attempt.attempt_number,
        "owner_session_id": attempt.owner_session_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    durable_binding = {
        **exact_attempt,
        **binding,
        "stage": "portal_entered",
        "record_id": content_identity(
            {"binding-admission": "stale-dispatch-migration"}
        ),
    }
    first_clear_receipt_id = "sha256:" + "5" * 64
    migration_retry_evidence_id = "sha256:" + "6" * 64
    prepared_id = "sha256:" + "7" * 64
    barrier_id = "sha256:" + "8" * 64
    migration_preparation_event_id = "sha256:" + "9" * 64
    state_recovery_event_id = "sha256:" + "a" * 64
    migration_terminal_event_id = "sha256:" + "b" * 64
    legacy_release_event_id = "sha256:" + "c" * 64
    migration_id = content_identity(
        {"migration": "stale-dispatch-release"}
    )
    migration_receipt_id = content_identity(
        {"migration-receipt": migration_id}
    )
    legacy_release_receipt_id = content_identity(
        {"legacy-release": "stale-dispatch"}
    )
    nested_task_cid = content_identity(
        {"nested-task": attempt.task_alias}
    )
    source = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "stale-dispatch-release-migration-retry@1"
        ),
        "binding_id": binding["binding_id"],
        "reconciliation_receipt": {
            "binding_id": binding["binding_id"],
            "receipt_id": first_clear_receipt_id,
            "nested_state": {"active_attempt": 1},
            "portal_reconciliation": {
                "task_claim_reconciliation": {
                    "canonical_task_cid": nested_task_cid,
                }
            },
        },
        "evidence_id": migration_retry_evidence_id,
    }
    claim_release = {
        "reconciled": True,
        "blocked": False,
        "reason": "quiesced_task_claim_released",
        "task_id": attempt.task_alias,
        "canonical_task_cid": nested_task_cid,
        "attempt": 1,
        "task_status": "todo",
        "stale_dispatch_intent_released_for_retry": True,
        "receipt_id": legacy_release_receipt_id,
    }
    forbidden_terminal = {
        "applicable": False,
        "blocked": False,
        "implementation_dispatched": False,
        "provider_dispatched": False,
        "reason": "provider_forbidden_terminal_recovery_not_applicable",
        "reconciled": False,
    }
    recovery = {
        "reconciled": True,
        "blocked": False,
        "reason": "already_quiesced",
        "task_id": attempt.task_alias,
        "task_claim_reconciliation": claim_release,
        "provider_forbidden_terminal_recovery": forbidden_terminal,
    }
    trigger = "database_daemon_startup"
    prepared = {
        "receipt_id": prepared_id,
        "binding_id": binding["binding_id"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "trigger": trigger,
        "nested_state": {
            "present": True,
            "active": False,
            "active_task_id": "",
            "active_attempt": 0,
            "active_phase": "",
            "state_path": str(paths.state),
            "state_digest": pre_state_digest,
        },
        "provider_runner_fence": {
            "safe_to_restart": True,
            "applicable": False,
            "fenced": False,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
        "portal_reconciliation": recovery,
    }
    barrier = {
        **prepared,
        "receipt_id": barrier_id,
        "prepared_reconciliation_receipt_id": prepared_id,
    }
    link = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-terminal-reconciliation-link@1"
        ),
        **exact_attempt,
        "binding_id": binding["binding_id"],
        "nested_state_digest": pre_state_digest,
        "nested_reason": "nested_portal_attempt_reconciled",
        "nested_reconciled": True,
        "trigger": trigger,
        "intended_database_disposition": "blocked_unknown_outcome",
        "prepared_reconciliation_receipt_id": prepared_id,
        "commit_barrier_receipt_id": barrier_id,
    }
    link["evidence_id"] = content_identity(link)
    outer_receipt = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        **exact_attempt,
        "operation": "database_unknown_outcome_blocked",
        "reason": "provider_dispatch_outcome_unknown",
        "retry_exhausted": True,
        "forced_block": True,
        "authority_outcome": "unknown",
        "process_instance_id": "process:stale-dispatch-migration",
        "attempts_used": 1,
        "unknown_outcome_rearm_count": 1,
        "terminal_reconciliation": link,
    }
    outer_before = json.loads(json.dumps(outer_receipt))
    replay = {
        "reconciled": True,
        "blocked": False,
        "reason": "stale_dispatch_release_migrated_for_retry",
        "task_id": attempt.task_alias,
        "canonical_task_cid": nested_task_cid,
        "attempt": 1,
        "migration_id": migration_id,
        "preparation_event_id": migration_preparation_event_id,
        "state_recovery_event_id": state_recovery_event_id,
        "migration_terminal_event_id": migration_terminal_event_id,
        "migration_receipt_id": migration_receipt_id,
        "legacy_claim_release_receipt_id": legacy_release_receipt_id,
        "legacy_claim_release_event_id": legacy_release_event_id,
        "pre_state_digest": pre_state_digest,
        "post_state_digest": post_state_digest,
        "provider_dispatched": False,
        "implementation_dispatched": False,
        "acceptance_inferred": False,
        "retained_candidate_disposition": "preserved_unvalidated",
        "stale_lock_cleared": False,
        "stale_lock_clear_event_id": "",
    }
    migration_calls: list[tuple[object, str]] = []

    class MigrationOnlyPortal:
        def reconcile_interrupted_database_implementation_attempt(
            self,
            _evidence: object,
        ) -> dict[str, object]:
            raise AssertionError("legacy interrupted-retry adapter was selected")

        def reconcile_stale_dispatch_release_migration(
            self,
            evidence: object,
            *,
            expected_pre_state_digest: str,
        ) -> dict[str, object]:
            migration_calls.append((evidence, expected_pre_state_digest))
            return dict(replay)

        def close_event_runtime(self) -> None:
            return None

    bridge.portal_factory = lambda _paths, _alias: MigrationOnlyPortal()
    bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _value: None)
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: None,
    )
    monkeypatch.setattr(
        bridge,
        "_stale_dispatch_migration_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    state_snapshots = [
        (
            {
                "implementation_in_progress": False,
                "active_task_id": "",
                "active_attempt": 0,
                "active_phase": "",
                "implementation_attempts": {attempt.task_alias: 1},
            },
            pre_state_digest,
        ),
        (
            {
                "implementation_in_progress": False,
                "active_task_id": "",
                "active_attempt": 0,
                "active_phase": "",
                "implementation_attempts": {},
            },
            post_state_digest,
        ),
    ]

    def strict_state(_path: Path) -> tuple[dict[str, object], str]:
        assert state_snapshots
        state, digest = state_snapshots.pop(0)
        return dict(state), digest

    monkeypatch.setattr(bridge, "_strict_state_record", strict_state)
    original_rearm = bridge._interrupted_implementation_rearm_evidence
    rearm_calls: list[str] = []

    def observe_rearm(
        candidate_attempt: object,
        candidate_receipt: object,
        *,
        expected_evidence_schema: str | None = None,
    ) -> dict[str, object] | None:
        rearm_calls.append(str(getattr(candidate_attempt, "attempt_id", "")))
        assert isinstance(candidate_receipt, dict)
        return original_rearm(
            candidate_attempt,
            candidate_receipt,
            expected_evidence_schema=expected_evidence_schema,
        )

    monkeypatch.setattr(
        bridge,
        "_interrupted_implementation_rearm_evidence",
        observe_rearm,
    )

    evidence = bridge.no_provider_dispatch_rearm_evidence(
        attempt,
        outer_block_receipt=outer_receipt,
    )

    assert evidence is not None
    assert evidence["schema"] == (
        DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA
    )
    assert rearm_calls == [attempt.attempt_id]
    assert migration_calls == [(source, pre_state_digest)]
    assert state_snapshots == []
    assert outer_receipt == outer_before
    assert outer_receipt["attempt_number"] == (
        outer_receipt["attempts_used"]
        + outer_receipt["unknown_outcome_rearm_count"]
    )
    assert evidence["provider_dispatched"] is False
    assert evidence["implementation_dispatched"] is False
    assert evidence["validation_attempted"] is False
    assert evidence["commit_created"] is False
    assert evidence["merge_attempted"] is False
    assert evidence["acceptance_inferred"] is False
    assert evidence["stale_lock_cleared"] is False
    assert evidence["stale_lock_clear_event_id"] == ""
    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=attempt,
        original=outer_receipt,
        expected_evidence_id=evidence["evidence_id"],
    )

    wrong_schema = {
        **dict(evidence),
        "schema": DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA,
    }
    wrong_schema.pop("evidence_id")
    wrong_schema["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            wrong_schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    assert not DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        wrong_schema,
        task=attempt,
        original=outer_receipt,
        expected_evidence_id=wrong_schema["evidence_id"],
    )

    wrong_policy = {
        **outer_receipt,
        "operation": "unreviewed_stale_dispatch_release",
    }
    assert bridge.no_provider_dispatch_rearm_evidence(
        attempt,
        outer_block_receipt=wrong_policy,
    ) is None
    assert rearm_calls == [attempt.attempt_id]
    assert migration_calls == [(source, pre_state_digest)]

    valid_replay = dict(replay)
    replay_tampers = {
        "extra_key": {**valid_replay, "unreviewed_replay_field": True},
        "wrong_task_id": {
            **valid_replay,
            "task_id": "PCTDD-WRONG",
        },
        "wrong_canonical_task_cid": {
            **valid_replay,
            "canonical_task_cid": content_identity(
                {"nested-task": "wrong"}
            ),
        },
        "wrong_attempt": {**valid_replay, "attempt": 2},
        "false_lock_with_event": {
            **valid_replay,
            "stale_lock_cleared": False,
            "stale_lock_clear_event_id": "sha256:" + "d" * 64,
        },
        "cleared_lock_without_event": {
            **valid_replay,
            "stale_lock_cleared": True,
            "stale_lock_clear_event_id": "",
        },
    }
    for case, tampered_replay in replay_tampers.items():
        replay.clear()
        replay.update(tampered_replay)
        state_snapshots.extend(
            [
                (
                    {
                        "implementation_in_progress": False,
                        "active_task_id": "",
                        "active_attempt": 0,
                        "active_phase": "",
                        "implementation_attempts": {attempt.task_alias: 1},
                    },
                    pre_state_digest,
                ),
                (
                    {
                        "implementation_in_progress": False,
                        "active_task_id": "",
                        "active_attempt": 0,
                        "active_phase": "",
                        "implementation_attempts": {},
                    },
                    post_state_digest,
                ),
            ]
        )
        assert bridge.no_provider_dispatch_rearm_evidence(
            attempt,
            outer_block_receipt=outer_receipt,
        ) is None, case
        assert state_snapshots == [], case
    replay.clear()
    replay.update(valid_replay)


class _StaleDispatchSelectorHarness(DatabaseImplementationDaemon):
    """Side-effect-free harness for the daemon's evidence selector."""

    def open(self) -> "_StaleDispatchSelectorHarness":
        return self

    @property
    def task_source(self) -> object:
        return self._selector_task_source

    @property
    def coordinator(self) -> object:
        return self._selector_coordinator


def _historical_stale_dispatch_selector_case(
    *,
    task_alias: str,
    attempt_number: int,
    attempts_used: int = 1,
) -> SimpleNamespace:
    """Build the exact outer historical shape without opening a store."""

    daemon = object.__new__(_StaleDispatchSelectorHarness)
    task_cid = f"task:cid:{task_alias.lower()}"
    attempt_id = f"attempt:historical-{task_alias.lower()}"
    claim_id = f"claim:historical-{task_alias.lower()}"
    owner_session_id = f"session:historical-{task_alias.lower()}"
    lease_id = f"lease:historical-{task_alias.lower()}"
    validation_spec_cid = "sha256:" + "1" * 64
    execution_spec_cid = "sha256:" + "2" * 64
    task_revision = 4
    retry_budget = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "validation_spec_cid": validation_spec_cid,
        "attempts_used": attempts_used,
        "max_task_attempts": 1,
        "configured_max_task_attempts": 1,
        "policy_mismatch": False,
        "malformed": False,
        "retry_exhausted": True,
    }
    control_claim = {
        "task_cid": task_cid,
        "revision": task_revision - 1,
        "execution_spec_cid": execution_spec_cid,
        "validation_spec_cid": validation_spec_cid,
    }
    attempt = DatabaseTaskAttempt(
        attempt_id=attempt_id,
        claim_id=claim_id,
        task_cid=task_cid,
        task_alias=task_alias,
        attempt_number=attempt_number,
        owner_session_id=owner_session_id,
        lease_id=lease_id,
        fencing_token=attempt_number,
        fence_epoch=attempt_number,
        committed_phase="failed",
        status="failed",
        started_at_ms=1,
        finished_at_ms=2,
        body={
            "control_claim": control_claim,
            "retry_budget": retry_budget,
        },
    )
    task_record = {
        "task_cid": task_cid,
        "task_alias": task_alias,
        "status": "blocked",
        "revision": task_revision,
        "body": {},
    }
    task = SimpleNamespace(**task_record)
    task.to_dict = lambda: json.loads(json.dumps(task_record))
    exact_attempt = {
        "attempt_id": attempt_id,
        "claim_id": claim_id,
        "task_cid": task_cid,
        "attempt_number": attempt_number,
        "owner_session_id": owner_session_id,
        "lease_id": lease_id,
        "fencing_token": attempt_number,
        "fence_epoch": attempt_number,
    }
    link = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-terminal-reconciliation-link@1"
        ),
        **exact_attempt,
        "binding_id": "sha256:" + "3" * 64,
        "nested_state_digest": "sha256:" + "4" * 64,
        "nested_reason": "nested_portal_attempt_reconciled",
        "nested_reconciled": True,
        "trigger": "database_daemon_startup",
        "intended_database_disposition": "blocked_unknown_outcome",
        "prepared_reconciliation_receipt_id": "sha256:" + "5" * 64,
        "commit_barrier_receipt_id": "sha256:" + "6" * 64,
    }
    link["evidence_id"] = content_identity(link)
    receipt = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        **exact_attempt,
        "validation_spec_cid": validation_spec_cid,
        "attempts_used": attempts_used,
        "max_task_attempts": 1,
        "retry_exhausted": True,
        "process_instance_id": f"process:historical-{task_alias.lower()}",
        "operation": "database_unknown_outcome_blocked",
        "reason": "provider_dispatch_outcome_unknown",
        "forced_block": True,
        "authority_outcome": "unknown",
        # Both preserved P005/P034 receipts predate this optional counter.
        "terminal_reconciliation": link,
    }
    phase_body = {
        "database_disposition": "blocked_unknown_outcome",
        "reason": "provider_dispatch_outcome_unknown",
        "retry_exhausted": True,
        "unknown_authority": True,
        "terminal_reconciliation": link,
    }

    def sha256_record(value: object) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    evidence = {
        "schema": DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA,
        **exact_attempt,
        "task_alias": task_alias,
        "attempt_root_key": hashlib.sha256(
            attempt_id.encode("utf-8")
        ).hexdigest()[:24],
        "attempt_authority_root_digest": "sha256:" + "7" * 64,
        "attempt_root_digest": "sha256:" + "8" * 64,
        "binding_id": link["binding_id"],
        "binding_admission_id": content_identity(
            {"binding-admission": task_alias}
        ),
        "binding_admission_digest": "sha256:" + "9" * 64,
        "projection_immutable_digest": "sha256:" + "a" * 64,
        "nested_task_cid": content_identity({"nested-task": task_alias}),
        "nested_attempt": 1,
        "terminal_reconciliation_evidence_id": link["evidence_id"],
        "first_clear_receipt_id": "sha256:" + "b" * 64,
        "migration_retry_evidence_id": "sha256:" + "c" * 64,
        "migration_id": content_identity({"migration": task_alias}),
        "migration_preparation_event_id": "sha256:" + "d" * 64,
        "state_recovery_event_id": "sha256:" + "e" * 64,
        "migration_terminal_event_id": "sha256:" + "f" * 64,
        "migration_receipt_id": content_identity(
            {"migration-receipt": task_alias}
        ),
        "legacy_claim_release_receipt_id": content_identity(
            {"legacy-claim-release": task_alias}
        ),
        "legacy_claim_release_event_id": "sha256:" + "0" * 64,
        "stale_lock_cleared": False,
        "stale_lock_clear_event_id": "",
        "prepared_reconciliation_receipt_id": link[
            "prepared_reconciliation_receipt_id"
        ],
        "commit_barrier_receipt_id": link["commit_barrier_receipt_id"],
        "pre_state_digest": link["nested_state_digest"],
        "state_digest": "sha256:" + "1" * 64,
        "outer_block_receipt_digest": (
            DatabaseImplementationDaemon._database_no_provider_rearm_digest(
                receipt
            )
        ),
        "migration_manifest_id": (
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID
        ),
        "migration_credit_id": "sha256:" + "d" * 64,
        "migration_credit_ordinal": 1,
        "provider_dispatched": False,
        "implementation_dispatched": False,
        "validation_attempted": False,
        "commit_created": False,
        "merge_attempted": False,
        "acceptance_inferred": False,
        "recovery_terminal": True,
        "retained_candidate_disposition": "preserved_unvalidated",
    }
    authorization = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-stale-dispatch-migration-"
            "rearm-authorization@1"
        ),
        **{
            name: evidence[name]
            for name in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "lease_id",
                "fencing_token",
                "fence_epoch",
                "binding_id",
                "binding_admission_id",
                "binding_admission_digest",
                "projection_immutable_digest",
                "nested_task_cid",
                "nested_attempt",
                "terminal_reconciliation_evidence_id",
                "first_clear_receipt_id",
                "migration_retry_evidence_id",
                "migration_id",
                "migration_preparation_event_id",
                "state_recovery_event_id",
                "migration_terminal_event_id",
                "migration_receipt_id",
                "legacy_claim_release_receipt_id",
                "legacy_claim_release_event_id",
                "stale_lock_cleared",
                "stale_lock_clear_event_id",
                "prepared_reconciliation_receipt_id",
                "commit_barrier_receipt_id",
                "pre_state_digest",
                "state_digest",
                "outer_block_receipt_digest",
            )
        },
    }
    evidence["rearm_authorization_id"] = sha256_record(authorization)
    evidence["evidence_id"] = sha256_record(evidence)

    calls: dict[str, list[object]] = {
        "verifier": [],
        "journal": [],
        "provider": [],
        "effect": [],
    }

    def verifier(
        candidate_attempt: object,
        *,
        outer_block_receipt: object,
    ) -> dict[str, object]:
        calls["verifier"].append(candidate_attempt)
        assert outer_block_receipt is receipt
        return dict(evidence)

    claim_record = {
        "task_cid": task_cid,
        "claim_id": claim_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "owner_session_id": owner_session_id,
        "lease_id": lease_id,
        "fencing_token": attempt_number,
        "fence_epoch": attempt_number,
    }
    claim = SimpleNamespace(state=SimpleNamespace(value="released"))
    claim.to_dict = lambda: dict(claim_record)
    terminal_saga = {
        **exact_attempt,
        "intended_database_disposition": "blocked_unknown_outcome",
        "evidence_id": link["evidence_id"],
        "prepared_reconciliation_receipt_id": link[
            "prepared_reconciliation_receipt_id"
        ],
        "commit_barrier_receipt_id": link["commit_barrier_receipt_id"],
        "stage": "terminal",
        "receipt_id": "sha256:" + "2" * 64,
    }

    daemon._selector_task_source = SimpleNamespace(
        get=lambda candidate: task if candidate == task_cid else None
    )
    daemon._selector_coordinator = SimpleNamespace(
        get_prepared_task_completion=lambda _candidate: None,
        get_task_claim=lambda candidate: (
            claim if candidate == claim_id else None
        ),
    )
    daemon._database_portal_bridge = SimpleNamespace(
        no_provider_dispatch_rearm_evidence=verifier
    )
    daemon.get_attempt = lambda candidate: (
        attempt if candidate == attempt_id else None
    )
    daemon._retry_budget_state = lambda _task: {
        "max_task_attempts": 1,
        "malformed": False,
        "policy_mismatch": False,
    }
    daemon._task_execution_spec_cid = lambda _task: execution_spec_cid
    daemon._retry_budget_validation_spec_cid = (
        lambda _task: validation_spec_cid
    )
    daemon.phase_history = lambda _candidate: [
        {"phase": "claimed", "body": {}},
        {"phase": "context", "body": {}},
        {"phase": "failed", "body": phase_body},
    ]
    daemon._database_portal_terminal_reconciliation_saga = (
        lambda _attempt: terminal_saga
    )
    daemon.provider_invocation_recorded = lambda *args, **kwargs: (
        calls["provider"].append((args, kwargs)) and None
    )
    daemon.effect_claim_recorded = lambda *args, **kwargs: (
        calls["effect"].append((args, kwargs)) and None
    )
    started_body = (
        {
            "resumed_from": "deferred",
            "preentry_publication_retry_count": 0,
        }
        if task_alias == "PCTDD-005"
        else {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-callback-dispatch@1"
            ),
            "outcome": "unknown_until_callback_returns",
        }
    )

    def journal(
        _attempt: object,
        *,
        dispatch_kind: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        calls["journal"].append((dispatch_kind, idempotency_key))
        if dispatch_kind == "effect":
            return None
        return {
            "outcome": "started",
            "body": dict(started_body),
            "updated_at_ms": 1,
        }

    daemon._dispatch_journal_entry = journal
    return SimpleNamespace(
        daemon=daemon,
        task=task,
        attempt=attempt,
        receipt=receipt,
        phase_body=phase_body,
        evidence=evidence,
        claim_record=claim_record,
        terminal_saga=terminal_saga,
        calls=calls,
        started_body=started_body,
    )


def _quiesced_release_started_selector_case(
    *,
    task_alias: str,
    attempt_number: int,
    rearm_count: int,
    evidence_schema: str = (
        DATABASE_PORTAL_QUIESCED_STALE_DISPATCH_RELEASE_REARM_EVIDENCE_SCHEMA
    ),
) -> SimpleNamespace:
    """Build the exact outer gate for a sealed quiesced release occurrence."""

    case = _historical_stale_dispatch_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
    )
    case.receipt["reason"] = "callback_authority_incomplete_blocked"
    case.receipt["unknown_outcome_rearm_count"] = rearm_count
    case.phase_body["reason"] = "callback_authority_incomplete_blocked"
    case.evidence["schema"] = evidence_schema
    case.evidence["outer_block_receipt_digest"] = (
        DatabaseImplementationDaemon._database_no_provider_rearm_digest(
            case.receipt
        )
    )
    case.evidence.pop("evidence_id", None)
    case.evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            case.evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    # These tests isolate the daemon's pre-verifier journal/tuple gate and its
    # post-verifier schema gate.  The bridge's full occurrence validator is
    # covered independently by its artifact and live-history test matrix.
    case.daemon._valid_no_provider_rearm_evidence = (
        lambda *args, **kwargs: True
    )
    return case


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "rearm_count"),
    (
        ("PCTDD-005", 3, 0),
        ("PCTDD-006", 4, 1),
        ("PCTDD-007", 4, 1),
        ("PCTDD-034", 8, 0),
    ),
)
def test_quiesced_release_started_journal_admits_only_exact_current_tuple(
    task_alias: str,
    attempt_number: int,
    rearm_count: int,
) -> None:
    """The four sealed callback-authority occurrences reach their verifier."""

    case = _quiesced_release_started_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
        rearm_count=rearm_count,
    )

    admitted = case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    )

    assert admitted is not None
    assert admitted["schema"] == (
        DATABASE_PORTAL_QUIESCED_STALE_DISPATCH_RELEASE_REARM_EVIDENCE_SCHEMA
    )
    assert case.calls["verifier"] == [case.attempt]
    assert case.calls["provider"] == [
        (
            (case.attempt.attempt_id,),
            {
                "idempotency_key": (
                    f"provider:{case.attempt.attempt_id}"
                )
            },
        )
    ]


def test_fenced_provider_started_journal_requires_sealed_migration_pin() -> None:
    case = _quiesced_release_started_selector_case(
        task_alias="PCTDD-006",
        attempt_number=1,
        rearm_count=0,
    )
    case.evidence.clear()
    case.evidence.update(
        _fenced_provider_unpublished_evidence(
            case.attempt,
            case.receipt,
        )
    )

    admitted = case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    )

    assert admitted is None
    assert case.calls["verifier"] == []
    assert case.calls["effect"] == [
        (
            (case.attempt.attempt_id,),
            {"idempotency_key": f"effect:{case.attempt.attempt_id}"},
        )
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_verifier_calls"),
    (
        ("predecessor_tuple", 0),
        ("wrong_rearm_count", 0),
        ("missing_terminal_link", 0),
        ("wrong_callback_reason", 1),
        ("not_retry_exhausted", 0),
        ("open_started_body", 0),
    ),
)
def test_quiesced_release_started_journal_near_misses_fail_closed(
    mutation: str,
    expected_verifier_calls: int,
) -> None:
    case = _quiesced_release_started_selector_case(
        task_alias="PCTDD-034",
        attempt_number=6 if mutation == "predecessor_tuple" else 8,
        rearm_count=0,
    )
    if mutation == "wrong_rearm_count":
        case.receipt["unknown_outcome_rearm_count"] = 1
    elif mutation == "missing_terminal_link":
        case.receipt.pop("terminal_reconciliation")
        case.phase_body.pop("terminal_reconciliation")
    elif mutation == "wrong_callback_reason":
        case.receipt["reason"] = "provider_dispatch_outcome_unknown"
        case.phase_body["reason"] = "provider_dispatch_outcome_unknown"
    elif mutation == "not_retry_exhausted":
        case.receipt["retry_exhausted"] = False
    else:
        case.started_body["unreviewed_field"] = True

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert len(case.calls["verifier"]) == expected_verifier_calls


def test_quiesced_release_started_journal_requires_release_evidence_schema(
) -> None:
    """A different admitted proof schema cannot inherit this journal gate."""

    case = _quiesced_release_started_selector_case(
        task_alias="PCTDD-034",
        attempt_number=8,
        rearm_count=0,
        evidence_schema=(
            DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
        ),
    )

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert case.calls["verifier"] == [case.attempt]


@pytest.mark.parametrize(
    ("task_alias", "attempt_number"),
    [("PCTDD-005", 1), ("PCTDD-034", 5)],
)
def test_synthetic_stale_dispatch_started_journal_cannot_inherit_sealed_credit(
    task_alias: str,
    attempt_number: int,
) -> None:
    """Legacy tuple shape alone cannot inherit an operator-sealed credit."""

    case = _historical_stale_dispatch_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
    )
    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    admitted = case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    )

    assert admitted is None
    # Read-only candidate probes may run, but no synthetic tuple can mutate
    # the task/attempt/receipt or manufacture a migration credit.  Exact
    # operator-sealed occurrence coverage lives in the occurrence-recovery
    # suite rather than this legacy shape harness.
    assert case.calls["verifier"] == [case.attempt]
    assert case.calls["journal"] == [
        ("effect", f"effect:{case.attempt.attempt_id}"),
        ("provider", f"provider:{case.attempt.attempt_id}"),
    ]
    assert len(case.calls["provider"]) == 1
    assert len(case.calls["effect"]) == 1
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


@pytest.mark.parametrize(
    ("mutation", "expected_verifier_calls", "expected_probe_calls"),
    [
        ("standard_policy", 0, 1),
        ("missing_terminal_link", 0, 0),
        ("wrong_evidence_schema", 1, 1),
        ("boolean_resumed_count", 0, 1),
        ("float_resumed_count", 0, 1),
        ("extra_started_body_field", 0, 1),
    ],
)
def test_stale_dispatch_started_journal_fails_closed_without_exact_migration(
    mutation: str,
    expected_verifier_calls: int,
    expected_probe_calls: int,
) -> None:
    case = _historical_stale_dispatch_selector_case(
        task_alias="PCTDD-005",
        attempt_number=1,
    )
    if mutation == "standard_policy":
        case.receipt["reason"] = "callback_authority_incomplete_blocked"
        case.phase_body["reason"] = "callback_authority_incomplete_blocked"
    elif mutation == "missing_terminal_link":
        case.receipt["reason"] = "callback_authority_incomplete_blocked"
        case.phase_body["reason"] = "callback_authority_incomplete_blocked"
        case.receipt.pop("terminal_reconciliation")
        case.phase_body.pop("terminal_reconciliation")
    elif mutation == "wrong_evidence_schema":
        case.evidence["schema"] = (
            DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
        )
        case.evidence.pop("evidence_id")
        case.evidence["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                case.evidence,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        # Isolate the selector's schema gate from the schema-specific payload
        # validator: even an otherwise admitted interrupted proof cannot
        # authorize this historical started-journal exception.
        case.daemon._valid_no_provider_rearm_evidence = (
            lambda *args, **kwargs: True
        )
    elif mutation == "boolean_resumed_count":
        case.started_body["preentry_publication_retry_count"] = False
    elif mutation == "float_resumed_count":
        case.started_body["preentry_publication_retry_count"] = 0.0
    else:
        case.started_body["unreviewed_field"] = True

    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert len(case.calls["verifier"]) == expected_verifier_calls
    assert len(case.calls["provider"]) == expected_probe_calls
    assert len(case.calls["effect"]) == expected_probe_calls
    assert len(case.calls["journal"]) == expected_probe_calls * 2
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


def test_stale_dispatch_invalid_budget_rejects_before_nested_migration() -> None:
    """The one-shot nested verifier is not invoked for an underbound budget."""

    case = _historical_stale_dispatch_selector_case(
        task_alias="PCTDD-034",
        attempt_number=5,
        attempts_used=6,
    )
    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert case.calls["verifier"] == []
    assert case.calls["journal"] == []
    assert case.calls["provider"] == []
    assert case.calls["effect"] == []
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


def _terminal_linked_interrupted_selector_case(
    *,
    attempt_number: int = 7,
    attempts_used: int = 1,
    rearm_count: object = 0,
) -> SimpleNamespace:
    """Build an exact linked interrupted proof whose counters have a gap."""

    case = _historical_stale_dispatch_selector_case(
        task_alias="PCTDD-034",
        attempt_number=attempt_number,
        attempts_used=attempts_used,
    )
    case.receipt["reason"] = "callback_authority_incomplete_blocked"
    case.receipt["unknown_outcome_rearm_count"] = rearm_count
    case.phase_body["reason"] = "callback_authority_incomplete_blocked"
    case.evidence["schema"] = (
        DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
    )
    unsigned = dict(case.evidence)
    unsigned.pop("evidence_id", None)
    case.evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    def raised_journal(
        _attempt: object,
        *,
        dispatch_kind: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        case.calls["journal"].append((dispatch_kind, idempotency_key))
        if dispatch_kind == "effect":
            return None
        return {
            "outcome": "raised",
            "body": {"exception_type": "DatabasePortalBridgeError"},
            "updated_at_ms": 1,
        }

    case.daemon._dispatch_journal_entry = raised_journal
    case.daemon._valid_no_provider_rearm_evidence = (
        lambda evidence, *, task, original, expected_evidence_id: bool(
            task is case.task
            and original is case.receipt
            and evidence.get("schema")
            == DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
            and evidence.get("evidence_id") == expected_evidence_id
        )
    )
    return case


@pytest.mark.parametrize(
    ("attempt_number", "attempts_used", "rearm_count", "expected"),
    (
        (1, 1, 0, True),
        (10_000, 1, 0, True),
        (7, 1, 3, True),
        (3, 1, 3, False),
        (True, 1, 0, False),
        (1, True, 0, False),
        (1, 1, False, False),
        (1, 0, 0, False),
        (1, 1, -1, False),
        (5, 1, DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT + 1, False),
    ),
)
def test_terminal_claim_ordinal_is_only_a_typed_corruption_lower_bound(
    attempt_number: object,
    attempts_used: object,
    rearm_count: object,
    expected: bool,
) -> None:
    assert (
        _database_terminal_claim_ordinal_lower_bound(
            attempt_number=attempt_number,
            attempts_used=attempts_used,
            rearm_count=rearm_count,
        )
        is expected
    )


def test_terminal_linked_interrupted_selector_accepts_zero_with_higher_ordinal(
) -> None:
    """An ordinal gap coexists with, but never authorizes, exact evidence."""

    case = _terminal_linked_interrupted_selector_case()
    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    admitted = case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    )

    assert admitted is not None
    assert admitted["schema"] == (
        DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
    )
    assert case.attempt.attempt_number == 7
    assert case.receipt["attempts_used"] == 1
    assert case.receipt["unknown_outcome_rearm_count"] == 0
    assert case.attempt.attempt_number > (
        case.receipt["attempts_used"]
        + case.receipt["unknown_outcome_rearm_count"]
    )
    assert case.calls["verifier"] == [case.attempt]
    assert case.calls["journal"] == [
        ("effect", f"effect:{case.attempt.attempt_id}"),
        ("provider", f"provider:{case.attempt.attempt_id}"),
    ]
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


def _terminal_linked_timeout_selector_case(
    *,
    outcome: str = "raised",
    body: object = None,
    quiescent_evidence: bool = True,
) -> SimpleNamespace:
    """Build the exact live outer TimeoutError shape around closed evidence."""

    case = _terminal_linked_interrupted_selector_case()
    if quiescent_evidence:
        case.evidence.update(
            {
                "schema": (
                    DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA
                ),
                "route_deferred": True,
                "nested_state_quiescent": True,
                "task_never_selected": True,
                "implementation_dispatched": False,
                "attempt_consumed": False,
                "acceptance_inferred": False,
            }
        )
    unsigned = dict(case.evidence)
    unsigned.pop("evidence_id", None)
    case.evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    dispatch_body = (
        {"exception_type": "TimeoutError"} if body is None else body
    )

    def timeout_journal(
        _attempt: object,
        *,
        dispatch_kind: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        case.calls["journal"].append((dispatch_kind, idempotency_key))
        if dispatch_kind == "effect":
            return None
        return {
            "outcome": outcome,
            "body": dispatch_body,
            "updated_at_ms": 1,
        }

    case.daemon._dispatch_journal_entry = timeout_journal
    expected_schema = str(case.evidence["schema"])
    case.daemon._valid_no_provider_rearm_evidence = (
        lambda evidence, *, task, original, expected_evidence_id: bool(
            task is case.task
            and original is case.receipt
            and evidence.get("schema") == expected_schema
            and evidence.get("evidence_id") == expected_evidence_id
        )
    )
    return case


def test_terminal_linked_timeout_admits_quiescent_or_interrupted_closed_proof(
) -> None:
    """The live outer timeout never chooses the nested evidence class itself."""

    for quiescent_evidence in (True, False):
        case = _terminal_linked_timeout_selector_case(
            quiescent_evidence=quiescent_evidence,
        )

        admitted = case.daemon._database_portal_no_provider_rearm_evidence(
            case.task,
            case.receipt,
        )

        assert admitted is not None
        assert admitted["schema"] == (
            DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA
            if quiescent_evidence
            else DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
        )
        assert case.calls["verifier"] == [case.attempt]
        assert case.calls["journal"] == [
            ("effect", f"effect:{case.attempt.attempt_id}"),
            ("provider", f"provider:{case.attempt.attempt_id}"),
        ]


@pytest.mark.parametrize(
    ("outcome", "body"),
    (
        ("raised", {"exception_type": "RuntimeError"}),
        ("raised", {"exception_type": "TimeoutError", "message": "late"}),
        ("raised", {"exception_type": "TimeoutError", "retryable": True}),
        ("deferred", {"exception_type": "TimeoutError"}),
        ("raised", "TimeoutError"),
    ),
)
def test_terminal_linked_timeout_outer_journal_shape_is_closed(
    outcome: str,
    body: object,
) -> None:
    case = _terminal_linked_timeout_selector_case(
        outcome=outcome,
        body=body,
    )

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert case.calls["verifier"] == []


@pytest.mark.parametrize(
    ("mutation", "expected_verifier_calls", "expected_journal_calls"),
    (
        ("boolean_zero", 0, 0),
        ("underbound_budget", 0, 0),
        ("missing_terminal_link", 0, 0),
        ("wrong_claim", 0, 2),
        ("wrong_saga", 0, 0),
        ("wrong_evidence_root", 1, 2),
    ),
)
def test_terminal_linked_interrupted_selector_keeps_budget_edges_closed(
    mutation: str,
    expected_verifier_calls: int,
    expected_journal_calls: int,
) -> None:
    """Numeric aliases, underbinding, and link loss never reach recovery."""

    case = _terminal_linked_interrupted_selector_case(
        attempts_used=8 if mutation == "underbound_budget" else 1,
        rearm_count=False if mutation == "boolean_zero" else 0,
    )
    if mutation == "missing_terminal_link":
        case.receipt.pop("terminal_reconciliation")
        case.phase_body.pop("terminal_reconciliation")
    elif mutation == "wrong_claim":
        case.claim_record["lease_id"] = "lease:wrong-current-claim"
    elif mutation == "wrong_saga":
        case.terminal_saga["commit_barrier_receipt_id"] = "sha256:" + "f" * 64
    elif mutation == "wrong_evidence_root":
        case.evidence["state_digest"] = "sha256:" + "e" * 64
    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert len(case.calls["verifier"]) == expected_verifier_calls
    assert len(case.calls["journal"]) == expected_journal_calls
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


def test_interrupted_rearm_uses_production_run_once_link_and_terminal_saga(
    tmp_path: Path,
) -> None:
    """The real verifier admits one linked refund without widening budget."""

    seed = _open_daemon(
        tmp_path,
        session="session:production-interrupted-refund",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        attempt_one = seed.claim_next()
        assert attempt_one is not None
        seed._begin_callback_dispatch(
            attempt_one,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_one.attempt_id}",
        )
        seed._finalize_failed_attempt(
            attempt_one,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
    finally:
        seed.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:production-interrupted-refund",
        max_task_attempts=2,
        callbacks_bound=False,
    )
    replay_holder: dict[str, object] = {}
    recovery_calls: list[str] = []

    class RetryOnlyPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, object]:
            return dict(replay_holder)

        def reconcile_provider_forbidden_terminal_result(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": (
                    "provider_forbidden_terminal_recovery_not_applicable"
                ),
                "reconciled": False,
            }

        def reconcile_interrupted_database_implementation_attempt(
            self,
            evidence: object,
        ) -> dict[str, object]:
            assert isinstance(evidence, dict)
            assert evidence.get("schema") == (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-implementation-retry@1"
            )
            recovery_calls.append("recovery")
            return dict(replay_holder)

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=daemon.task_source,
        attempt_root=tmp_path / "portal-attempts",
        portal_factory=lambda _paths, _alias: RetryOnlyPortal(),
    )
    daemon.bind_execution_callbacks(
        provider_fn=bridge.run_provider,
        effect_fn=bridge.apply_effect,
        validation_fn=bridge.validate_effect,
    )
    daemon.bind_database_portal_bridge(bridge)
    try:
        generic = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(generic) == 1
        assert generic[0]["unknown_outcome_rearm_count"] == 1

        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 2
        attempt = daemon.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"context_cid": content_identity({"attempt": attempt.attempt_id})},
        )
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        record = daemon.task_source.get(attempt.task_cid)
        assert record is not None
        paths, binding = bridge._ensure_attempt_projection(
            attempt,
            record,
            admit_before_publish=True,
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="raised",
            body={
                "exception_type": "DatabasePortalBridgeError",
            },
        )
        assert bridge._binding_recorder is not None
        bridge._binding_recorder(attempt, binding, "portal_entered")

        quiescent_state = {
            "implementation_in_progress": False,
            "active_task_id": "",
            "active_attempt": 0,
            "active_phase": "",
        }
        state_bytes = json.dumps(
            quiescent_state,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        paths.state.write_bytes(state_bytes)
        paths.state.chmod(0o600)
        state_digest = "sha256:" + hashlib.sha256(state_bytes).hexdigest()
        nested_task_cid = "nested:task:current"
        nested_attempt = 1
        workspace = str(tmp_path / "retained-candidate")
        first_clear = bridge.persist_reconciliation_receipt(
            attempt,
            {
                "stage": "blocked",
                "trigger": "supervisor_signal_shutdown",
                "reconciled_at": "2026-01-01T00:00:00+00:00",
                "reconciled": False,
                "blocked": True,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "binding_id": binding["binding_id"],
                "task_alias": attempt.task_alias,
                "nested_state": {
                    "present": True,
                    "active": True,
                    "active_task_id": attempt.task_alias,
                    "active_attempt": nested_attempt,
                    "active_phase": "implementing",
                    "active_worktree_path": workspace,
                    "active_branch": "implementation/test-attempt-1",
                    "state_path": str(paths.state),
                    "state_digest": "sha256:" + "0" * 64,
                },
                "provider_runner_fence": {
                    "applicable": True,
                    "fenced": True,
                    "safe_to_restart": True,
                    "pid": 4242,
                    "reason": "ordinary_provider_runner_exact_birth_fenced",
                },
                "provider_runner_reconciliation_authority": (
                    "ordinary_provider_runner_fence"
                ),
                "portal_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "task_claim_reconciliation_blocked",
                    "reconciled_at": "2026-01-01T00:00:00+00:00",
                    "protected_path_reconciliation": {
                        "blocked": False,
                        "reason": "crash_reconciliation_unchanged",
                        "task_id": attempt.task_alias,
                        "attempt": nested_attempt,
                        "workspace_path": workspace,
                    },
                    "worktree_lifecycle_reconciliation": {
                        "blocked": False,
                        "reconciled": True,
                        "state": "terminal",
                        "task_id": attempt.task_alias,
                        "attempt": nested_attempt,
                        "workspace_path": workspace,
                        "record_id": content_identity({"lifecycle": "terminal"}),
                        "fence": 1,
                    },
                    "task_claim_reconciliation": {
                        "blocked": True,
                        "reconciled": False,
                        "reason": "canonical_task_not_terminal",
                        "observed_task_status": "todo",
                        "task_id": attempt.task_alias,
                        "canonical_task_cid": nested_task_cid,
                    },
                    "attempt_recovery": {
                        "consumed": False,
                        "attempt": nested_attempt,
                        "task_id": attempt.task_alias,
                        "canonical_task_cid": nested_task_cid,
                        "previous_display_count": nested_attempt,
                        "previous_cid_count": nested_attempt,
                    },
                },
                "terminal_provider_evidence": False,
            },
        )
        interrupted_retry_id = content_identity(
            {"interrupted-retry": attempt.attempt_id}
        )
        released_attempt = {
            "released_from": nested_attempt,
            "released_to": nested_attempt - 1,
            "event_id": "sha256:" + "2" * 64,
        }
        replay_holder.update(
            {
                "reconciled": True,
                "blocked": False,
                "reason": "interrupted_implementation_recovered_for_retry",
                "task_id": attempt.task_alias,
                "canonical_task_cid": nested_task_cid,
                "attempt": nested_attempt,
                "task_claim_reconciliation": {
                    "reconciled": True,
                    "blocked": False,
                    "reason": "quiesced_task_claim_released",
                    "task_id": attempt.task_alias,
                    "task_status": "todo",
                    "released_unfinished_retry_id": interrupted_retry_id,
                    "receipt_id": content_identity(
                        {"claim-release": attempt.attempt_id}
                    ),
                    "released_unfinished_attempt": released_attempt,
                },
                "provider_dispatched": False,
                "implementation_dispatched": False,
                "acceptance_inferred": False,
                "retained_candidate_disposition": "preserved_unvalidated",
                "stale_lock_cleared": False,
            }
        )
        recovery = {
            **replay_holder,
            "provider_forbidden_terminal_recovery": {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": "provider_forbidden_terminal_recovery_not_applicable",
                "reconciled": False,
            },
        }
        terminal_core = {
            "reconciled": True,
            "blocked": False,
            "reason": "nested_portal_attempt_reconciled",
            "binding_id": binding["binding_id"],
            "nested_state": {
                "present": True,
                "active": False,
                "active_task_id": "",
                "active_attempt": 0,
                "active_phase": "",
                "state_path": str(paths.state),
                "state_digest": state_digest,
            },
            "provider_runner_fence": {
                "safe_to_restart": True,
                "applicable": False,
                "fenced": False,
                "reason": "ordinary_provider_runner_receipt_absent",
            },
            "portal_reconciliation": recovery,
            "terminal_provider_evidence": False,
        }
        trigger = "database_daemon_startup"
        prepared = bridge.persist_reconciliation_receipt(
            attempt,
            {
                **terminal_core,
                "stage": "prepared",
                "trigger": trigger,
                "intended_database_disposition": "blocked_unknown_outcome",
                "reconciled_at": "2026-01-01T00:00:01+00:00",
            },
        )
        barrier = bridge.persist_reconciliation_receipt(
            attempt,
            {
                **terminal_core,
                "stage": "commit_barrier",
                "trigger": trigger,
                "intended_database_disposition": "blocked_unknown_outcome",
                "prepared_reconciliation_receipt_id": prepared["receipt_id"],
                "reconciled_at": "2026-01-01T00:00:01+00:00",
            },
        )
        link = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "task_cid": attempt.task_cid,
            "attempt_number": attempt.attempt_number,
            "owner_session_id": attempt.owner_session_id,
            "lease_id": attempt.lease_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "binding_id": binding["binding_id"],
            "nested_state_digest": state_digest,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": prepared["receipt_id"],
            "commit_barrier_receipt_id": barrier["receipt_id"],
        }
        link["evidence_id"] = content_identity(link)
        daemon._record_database_portal_terminal_reconciliation_barrier(
            attempt,
            link,
        )
        failed, receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
            reconciliation_evidence=link,
        )
        assert receipt["attempts_used"] == 1
        assert receipt["unknown_outcome_rearm_count"] == 1
        repaired = daemon._repair_database_portal_terminal_receipts(
            bridge=bridge,
            trigger="test_terminal_repair",
            exact_attempt=failed,
        )
        assert len(repaired) == 1 and repaired[0]["reconciled"] is True
        saga = daemon._database_portal_terminal_reconciliation_saga(failed)
        assert saga is not None and saga["stage"] == "terminal"

        failed_phase = daemon.phase_history(failed.attempt_id)[-1]["body"]
        mismatched_phase = dict(failed_phase)
        mismatched_phase["terminal_reconciliation"] = {
            **link,
            "trigger": "different-trigger",
        }
        assert daemon._database_interrupted_failed_phase_link(
            mismatched_phase,
            receipt,
        ) is None
        assert recovery_calls == []

        blocked_task = daemon.task_source.get(attempt.task_cid)
        assert blocked_task is not None
        assert daemon._database_interrupted_failed_phase_link(
            daemon.phase_history(failed.attempt_id)[-1]["body"],
            receipt,
        ) is not None
        assert [
            phase["phase"] for phase in daemon.phase_history(failed.attempt_id)
        ] == ["claimed", "context", "failed"]
        provider_dispatch = daemon._dispatch_journal_entry(
            failed,
            dispatch_kind="provider",
            idempotency_key=f"provider:{failed.attempt_id}",
        )
        assert provider_dispatch is not None
        assert provider_dispatch["outcome"] == "raised"
        assert provider_dispatch["body"]["exception_type"] == (
            "DatabasePortalBridgeError"
        )
        terminal_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert terminal_claim is not None
        assert str(terminal_claim.state.value) == "released"
        control_claim = dict(failed.body["control_claim"])
        assert blocked_task.revision == int(control_claim["revision"]) + 1
        assert control_claim["task_cid"] == failed.task_cid
        assert control_claim["execution_spec_cid"] == (
            daemon._task_execution_spec_cid(blocked_task)
        )
        assert control_claim["validation_spec_cid"] == (
            daemon._retry_budget_validation_spec_cid(blocked_task)
        )
        budget_state = daemon._retry_budget_state(blocked_task)
        assert budget_state["malformed"] is False
        assert budget_state["policy_mismatch"] is False
        assert receipt["max_task_attempts"] == budget_state["max_task_attempts"]

        result = daemon.run_once()

        assert result["selection_idle_reason"] == (
            "database_unknown_outcomes_rearmed"
        )
        assert len(result["unknown_outcome_rearms"]) == 1
        assert recovery_calls == ["recovery"]
        rearmed = daemon.task_source.get(attempt.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        rearm_receipt = rearmed.body["completion_receipt"]
        assert rearm_receipt["attempts_used"] == 0
        assert rearm_receipt["unknown_outcome_rearm_count"] == 1
        assert rearm_receipt["no_provider_rearm_evidence"][
            "first_clear_receipt_id"
        ] == first_clear["receipt_id"]
        assert daemon.reconcile_blocked_unknown_outcome_tasks() == []
        assert daemon._unresolved_database_no_provider_rearm_sagas() == ()
        assert recovery_calls == ["recovery"]
    finally:
        daemon.close()


def test_unknown_outcome_rearm_limit_survives_claim_and_block_revisions(
    tmp_path: Path,
) -> None:
    population = _population(1)
    seed = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
    finally:
        seed.close()

    for expected_count in range(1, DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT + 1):
        blocker = _open_daemon(
            tmp_path,
            session="session:durable-unknown-rearm-limit",
            max_task_attempts=2,
        )
        try:
            attempt = blocker.claim_next()
            assert attempt is not None
            blocker._begin_callback_dispatch(
                attempt,
                dispatch_kind="provider",
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            _failed, receipt = blocker._finalize_failed_attempt(
                attempt,
                reason="provider_dispatch_outcome_unknown",
                force_block=True,
                unknown_authority=True,
            )
            prior_count = expected_count - 1
            if prior_count:
                assert receipt["unknown_outcome_rearm_count"] == prior_count
        finally:
            blocker.close()

        successor = _open_daemon(
            tmp_path,
            session="session:durable-unknown-rearm-limit",
            max_task_attempts=2,
        )
        try:
            rearms = successor.reconcile_blocked_unknown_outcome_tasks()
            assert len(rearms) == 1
            task = successor.task_source.get("task:cid:001")
            assert task is not None and task.status == "retrying"
            assert task.body["completion_receipt"][
                "unknown_outcome_rearm_count"
            ] == expected_count
        finally:
            successor.close()

    final_blocker = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        final_attempt = final_blocker.claim_next()
        assert final_attempt is not None
        final_blocker._begin_callback_dispatch(
            final_attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{final_attempt.attempt_id}",
        )
        _failed, final_receipt = final_blocker._finalize_failed_attempt(
            final_attempt,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
        assert final_receipt["unknown_outcome_rearm_count"] == (
            DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        )
    finally:
        final_blocker.close()

    terminal = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        assert terminal.reconcile_blocked_unknown_outcome_tasks() == []
        task = terminal.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"][
            "unknown_outcome_rearm_count"
        ] == DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
    finally:
        terminal.close()


def test_later_process_rearms_exhausted_portal_provider_failure(
    tmp_path: Path,
) -> None:
    seed = _open_daemon(
        tmp_path,
        session="session:portal-exhausted-seed",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason="portal_provider_failed",
        )
        receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
        assert seed.reconcile_blocked_unknown_outcome_tasks() == []
        assert seed.task_source.get("task:cid:001").status == "blocked"
    finally:
        seed.close()

    successor_calls: list[str] = []
    successor = _open_daemon(
        tmp_path,
        session="session:portal-exhausted-rearm",
        provider_calls=successor_calls,
        max_task_attempts=2,
    )
    try:
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert len(rearms) == 1
        task = successor.task_source.get("task:cid:001")
        assert task is not None and task.status == "retrying"
        assert task.body["completion_receipt"]["attempts_used"] == 0
        claimed = successor.run_once()
        assert claimed["claimed_task_cid"] == "task:cid:001"
        assert successor_calls == ["task:cid:001"]
    finally:
        successor.close()


def test_malformed_terminal_candidate_does_not_starve_unrelated_rearm(
    tmp_path: Path,
) -> None:
    seed = _open_daemon(
        tmp_path,
        session="session:terminal-selector-liveness-seed",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(2))
        terminal_task = seed.task_source.get("task:cid:001")
        generic_task = seed.task_source.get("task:cid:002")
        assert terminal_task is not None and generic_task is not None
        terminal_receipt = seed._retry_budget_receipt(
            terminal_task,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        terminal_receipt.update(
            {
                "authority_outcome": "unknown",
                "forced_block": True,
                "terminal_reconciliation": "malformed",
            }
        )
        seed._cas_task_status_database(
            terminal_task.task_cid,
            expected_revision=int(terminal_task.revision),
            new_status="blocked",
            receipt=terminal_receipt,
        )
        generic_receipt = seed._retry_budget_receipt(
            generic_task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason="portal_provider_failed",
        )
        generic_receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            generic_task.task_cid,
            expected_revision=int(generic_task.revision),
            new_status="blocked",
            receipt=generic_receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:terminal-selector-liveness-successor",
        max_task_attempts=2,
    )
    try:
        # The malformed link is rejected before this sentinel can be used.
        successor._database_portal_bridge = object()
        outcomes = successor.reconcile_blocked_unknown_outcome_tasks()

        assert any(
            item.get("task_cid") == "task:cid:001"
            and item.get("reason") == "terminal_landed_candidate_link_invalid"
            and item.get("blocked") is True
            for item in outcomes
        )
        assert any(
            item.get("task_cid") == "task:cid:002"
            and item.get("operation") == "database_unknown_outcome_rearmed"
            for item in outcomes
        )
        terminal = successor.task_source.get("task:cid:001")
        generic = successor.task_source.get("task:cid:002")
        assert terminal is not None and terminal.status == "blocked"
        assert generic is not None and generic.status == "retrying"
    finally:
        successor.close()


def test_nested_state_changed_landed_recovery_falls_through_to_generic_rearm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-link nested digest change is retry authority, not a lane fence.

    Landed completion must not CAS-complete when nested state moved after the
    terminal link was written.  That exact miss is also not a failed landed
    candidate: generic unknown-outcome rearm must still see the task, and an
    unrelated exhausted provider failure in the same page must still rearm.
    """

    seed = _open_daemon(
        tmp_path,
        session="session:nested-state-changed-liveness-seed",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(2))
        terminal_task = seed.task_source.get("task:cid:001")
        generic_task = seed.task_source.get("task:cid:002")
        assert terminal_task is not None and generic_task is not None
        terminal_receipt = seed._retry_budget_receipt(
            terminal_task,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        terminal_receipt.update(
            {
                "authority_outcome": "unknown",
                "forced_block": True,
                "terminal_reconciliation": {"schema": "candidate"},
            }
        )
        seed._cas_task_status_database(
            terminal_task.task_cid,
            expected_revision=int(terminal_task.revision),
            new_status="blocked",
            receipt=terminal_receipt,
        )
        generic_receipt = seed._retry_budget_receipt(
            generic_task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason="portal_provider_failed",
        )
        generic_receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            generic_task.task_cid,
            expected_revision=int(generic_task.revision),
            new_status="blocked",
            receipt=generic_receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:nested-state-changed-liveness-successor",
        max_task_attempts=2,
    )
    generic_rearm_tasks: list[str] = []

    def raise_nested_state_changed(*, task, bridge):
        raise DatabasePortalTerminalQuiescentStateAdvanced(
            "blocked terminal landed recovery nested state changed"
        )

    def observe_generic_rearm(task, receipt):
        generic_rearm_tasks.append(str(task.task_cid))
        return None

    try:
        successor._database_portal_bridge = object()
        monkeypatch.setattr(
            successor,
            "_reconcile_one_blocked_terminal_landed_task",
            raise_nested_state_changed,
        )
        monkeypatch.setattr(
            successor,
            "_database_portal_no_provider_rearm_evidence",
            observe_generic_rearm,
        )
        outcomes = successor.reconcile_blocked_unknown_outcome_tasks()

        assert not any(
            item.get("reason") == "terminal_landed_candidate_recovery_blocked"
            for item in outcomes
        )
        assert "task:cid:001" in generic_rearm_tasks
        assert any(
            item.get("task_cid") == "task:cid:002"
            and item.get("operation") == "database_unknown_outcome_rearmed"
            for item in outcomes
        )
        terminal = successor.task_source.get("task:cid:001")
        generic = successor.task_source.get("task:cid:002")
        assert terminal is not None and terminal.status == "blocked"
        assert generic is not None and generic.status == "retrying"
    finally:
        successor.close()


def test_run_once_skip_defers_coordination_storage_repair_for_extra_gate() -> None:
    """Coordination ART rebuild must not crash extra-gate finalize/release.

    Lane-0 died on DatabaseCoordinationStorageRepairedError while
    releasing PCTDD-005 attempt:f0413133, throwing away the hashlib
    first pass. Official unstick is retry after reopen, never CAS.
    Extra-gate aliases still cannot bypass safe_to_restart=False.
    """

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationStorageRepairedError,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    Daemon = daemon_module.DatabaseImplementationDaemon
    opened: list[str] = []
    daemon = Daemon.__new__(Daemon)
    daemon._coordinator = SimpleNamespace(
        open=lambda: opened.append("open") or daemon._coordinator
    )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/coordination-storage-repair@1",
        "logical_projection_equal": True,
        "retry_required": True,
    }
    exc = DatabaseCoordinationStorageRepairedError(
        "DuckDB ART storage was rebuilt with equal logical projection; "
        "the interrupted fenced operation must be reconciled before retry",
        receipt=receipt,
    )

    def _boom() -> dict[str, Any]:
        raise exc

    daemon._run_once_database_authoritative = _boom  # type: ignore[method-assign]
    result = Daemon.run_once(daemon)

    assert opened == ["open", "open"]
    assert result["unchanged"] is True
    assert result["write_count"] == 0
    assert result["selection_idle_reason"] == (
        "database_coordination_storage_repaired_retry_required"
    )
    assert result["database_portal_reconciliation"]["blocked"] is False
    assert result["database_portal_reconciliation"]["reason"] == (
        "database_coordination_storage_repaired_retry_required"
    )
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_extra_gate_retrying_receipt_is_not_nonconsuming_refund_tamper() -> None:
    """Nested-state-changed extra-gate retry receipts stay claimable.

    Provider-route saga replay must not treat extra-gate audit pairs
    (previous_operation / authority_outcome) as a refund-authority tamper
    fail-close.  The current retrying receipt is retry authority.
    Extra-gate aliases still cannot bypass safe_to_restart=False.
    """

    Daemon = daemon_module.DatabaseImplementationDaemon
    receipt = {
        "operation": "database_retry_rearmed",
        "forced_block": False,
        "retry_exhausted": False,
        "previous_operation": "database_unknown_outcome_blocked",
        "previous_owner_session_id": "embedded-store:predecessor",
        "authority_outcome": "rearmed",
    }
    retrying = SimpleNamespace(
        status="retrying",
        task_alias="PCTDD-034",
        body={"completion_receipt": receipt},
    )
    assert Daemon._task_alias_is_extra_gate(retrying) is True
    assert Daemon._extra_gate_retry_receipt_is_claimable(retrying, receipt) is True
    in_progress = SimpleNamespace(
        status="in_progress",
        task_alias="PCTDD-034",
        body={"completion_receipt": receipt},
    )
    assert (
        Daemon._extra_gate_retry_receipt_is_claimable(in_progress, receipt)
        is False
    )


def test_extra_gate_foreign_process_running_attempt_is_not_live_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead extra-gate workers must not keep a 2h accepted lease.

    Resume-then-callback_failure never spawned grok. Official unstick is
    rearm, never CAS. Extra-gate aliases still cannot bypass
    safe_to_restart=False.
    """

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    Daemon = daemon_module.DatabaseImplementationDaemon
    daemon = SimpleNamespace(
        process_instance_id="process:live",
        task_source=SimpleNamespace(get=lambda _cid: None),
        _task_alias_is_extra_gate=Daemon._task_alias_is_extra_gate,
    )
    foreign = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        body={"process_instance_id": "process:dead"},
    )
    local = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        body={"process_instance_id": "process:live"},
    )
    other = SimpleNamespace(
        task_alias="PCTDD-008",
        task_cid="task:cid:008",
        body={"process_instance_id": "process:dead"},
    )
    empty = SimpleNamespace(
        task_alias="PCTDD-007",
        task_cid="task:cid:007",
        body={},
    )
    rewritten = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        body={},
    )
    daemon.task_source = SimpleNamespace(
        get=lambda _cid: SimpleNamespace(
            body={
                "completion_receipt": {
                    "process_instance_id": "process:live",
                }
            }
        )
    )
    cid_only = SimpleNamespace(
        task_alias="",
        task_cid="baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
        body={"process_instance_id": "process:live"},
    )
    assert (
        Daemon._extra_gate_running_attempt_is_live_local(daemon, foreign)
        is False
    )
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, local) is False
    assert Daemon._task_alias_is_extra_gate(cid_only) is True
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, cid_only) is False
    monkeypatch.setattr(
        daemon_module,
        "active_codex_exec_workers",
        lambda *_args, **_kwargs: [{"pid": 3112389}],
    )
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, local) is True
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, other) is True
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, empty) is False
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, rewritten) is False
    assert Daemon._extra_gate_running_attempt_is_live_local(daemon, cid_only) is True
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


def test_extra_gate_incomplete_projection_closes_when_grok_is_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead extra-gate grok must not keep a deferred incomplete projection.

    Resume renewed the 2h lease on ``Portal task projection is not complete``
    after PCTDD-005 grok 1592311 died, so rearm never dispatched. Official
    unstick is rearm, never CAS. Extra-gate aliases still cannot bypass
    ``safe_to_restart=False``.
    """

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    Daemon = daemon_module.DatabaseImplementationDaemon
    daemon = SimpleNamespace(
        process_instance_id="process:live",
        task_source=SimpleNamespace(get=lambda _cid: None),
        _task_alias_is_extra_gate=Daemon._task_alias_is_extra_gate,
        _extra_gate_running_attempt_is_live_local=lambda attempt: (
            Daemon._extra_gate_running_attempt_is_live_local(daemon, attempt)
        ),
        _extra_gate_recorded_runner_pid=lambda attempt: (
            Daemon._extra_gate_recorded_runner_pid(daemon, attempt)
        ),
        _extra_gate_recorded_runner_was_spawned_here=lambda attempt: (
            Daemon._extra_gate_recorded_runner_was_spawned_here(daemon, attempt)
        ),
        _extra_gate_attempt_belongs_to_this_process=lambda attempt: (
            Daemon._extra_gate_attempt_belongs_to_this_process(daemon, attempt)
        ),
        _extra_gate_claim_has_not_started_worker=lambda attempt: (
            Daemon._extra_gate_claim_has_not_started_worker(daemon, attempt)
        ),
        _extra_gate_attempt_is_within_launch_grace=lambda attempt: (
            Daemon._extra_gate_attempt_is_within_launch_grace(daemon, attempt)
        ),
        _now_ms=lambda: 1_000_000,
        _extra_gate_dead_runner_block_opens_generic_rearm=lambda task, receipt: (
            Daemon._extra_gate_dead_runner_block_opens_generic_rearm(
                daemon, task, receipt
            )
        ),
    )
    monkeypatch.setattr(
        daemon_module,
        "active_codex_exec_workers",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(daemon_module, "process_is_running", lambda _pid: False)
    dead = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        process_instance_id="process:dead",
        body={
            "process_instance_id": "process:dead",
            "active_provider_runner": {"pid": 1592311, "owner_pid": 1366231},
        },
    )
    this_dead = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        process_instance_id="process:live",
        started_at_ms=1,
        body={
            "process_instance_id": "process:live",
            "started_at_ms": 1,
            "active_provider_runner": {
                "pid": 1592311,
                "owner_pid": os.getpid(),
            },
        },
    )
    this_dead_fresh = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        process_instance_id="process:live",
        started_at_ms=999_000,
        body={
            "process_instance_id": "process:live",
            "started_at_ms": 999_000,
            "active_provider_runner": {
                "pid": 1592311,
                "owner_pid": os.getpid(),
            },
        },
    )
    predecessor = SimpleNamespace(
        task_alias="PCTDD-034",
        task_cid="task:cid:034",
        process_instance_id="process:live",
        started_at_ms=999_000,
        body={
            "process_instance_id": "process:live",
            "started_at_ms": 999_000,
            "active_provider_runner": {"pid": 2804560, "owner_pid": 2421577},
        },
    )
    live_runner = SimpleNamespace(
        task_alias="PCTDD-034",
        task_cid="task:cid:034",
        process_instance_id="process:other",
        body={"active_provider_runner": {"pid": 2804560}},
    )
    mid_launch = SimpleNamespace(
        task_alias="PCTDD-007",
        task_cid="task:cid:007",
        process_instance_id="process:live",
        body={"process_instance_id": "process:live"},
    )
    ordinary = SimpleNamespace(
        task_alias="PCTDD-008",
        task_cid="task:cid:008",
        body={},
    )
    fresh_copied = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="task:cid:005",
        process_instance_id="process:other",
        body={
            "process_instance_id": "process:other",
            "retry_budget": {
                "retained_recovery_consumption": {
                    "consumed_before_worker_start": True,
                }
            },
            "active_provider_runner": {"pid": 1592311, "owner_pid": 1366231},
        },
    )
    blocked_034 = SimpleNamespace(
        task_alias="PCTDD-034",
        status="blocked",
        body={},
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(daemon, dead)
        is False
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(daemon, this_dead)
        is False
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(
            daemon, this_dead_fresh
        )
        is True
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(
            daemon, predecessor
        )
        is True
    )
    monkeypatch.setattr(
        daemon_module, "process_is_running", lambda pid: int(pid) == 2804560
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(
            daemon, live_runner
        )
        is True
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(
            daemon, mid_launch
        )
        is True
    )
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(daemon, ordinary)
        is True
    )
    monkeypatch.setattr(daemon_module, "process_is_running", lambda _pid: False)
    assert (
        Daemon._extra_gate_incomplete_projection_is_in_flight(
            daemon, fresh_copied
        )
        is True
    )
    quack_exc = type("IOException", (Exception,), {})(
        "IO Error: Failed to send message: IO Error: Could not connect "
        "to server error for HTTP POST to 'http://127.0.0.1:27278/quack'"
    )
    assert Daemon._exception_is_transient_quack_transport(quack_exc) is True
    assert Daemon._exception_is_transient_quack_transport(RuntimeError("x")) is False
    assert (
        Daemon._extra_gate_dead_runner_block_opens_generic_rearm(
            daemon,
            blocked_034,
            {"reason": "extra_gate_incomplete_projection_dead_runner"},
        )
        is True
    )
    assert (
        Daemon._extra_gate_dead_runner_block_opens_generic_rearm(
            daemon,
            ordinary,
            {"reason": "extra_gate_incomplete_projection_dead_runner"},
        )
        is False
    )
    assert (
        Daemon._extra_gate_dead_runner_block_opens_generic_rearm(
            daemon,
            blocked_034,
            {
                "reason": (
                    "'DatabasePortalExecutionBridge' object has no "
                    "attribute '_extra_gate_incomplete_projection_needs_provider'"
                )
            },
        )
        is True
    )
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )



def test_extra_gate_pre_dispatch_provider_error_defers_without_force_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra-gate setup exceptions must not burn the one-shot as unknown.

    Live g9 claimed 005/007/034, then wrapped a non-Portal exception as
    ``provider callback raised without admissible return evidence``. Portal
    events stayed empty and grok never started. Official unstick is a
    bounded provider-route deferral of the exact claim, never CAS.
    Extra-gate aliases still cannot bypass ``safe_to_restart=False``.
    """

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationError,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeError,
        DatabasePortalProviderRouteDeferred,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    Daemon = daemon_module.DatabaseImplementationDaemon
    daemon = SimpleNamespace(
        process_instance_id="process:live",
        _task_alias_is_extra_gate=Daemon._task_alias_is_extra_gate,
    )
    extra_gate = SimpleNamespace(
        task_alias="PCTDD-005",
        task_cid="baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza",
    )
    ordinary = SimpleNamespace(
        task_alias="PCTDD-008",
        task_cid="task:cid:008",
    )
    boom = RuntimeError("embedded execution store already has an active database writer")
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, extra_gate, boom
        )
        is True
    )
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, ordinary, boom
        )
        is False
    )
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, extra_gate, DatabasePortalBridgeError("portal blocked")
        )
        is False
    )
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, extra_gate, DatabaseCoordinationError("not open")
        )
        is False
    )
    missing = FileNotFoundError(
        2,
        "No such file or directory",
        Path(
            "/home/barberb/lift_coding/.worktrees/pctdd-g9-orphan-recovery/"
            "data/agent_supervisor/parallel_content_sealing_proof_carrying_tdd_v1_g9/"
            "worktrees/workspace_b6c6c987ab22_678bd72ae733"
        ),
    )
    assert Daemon._exception_is_missing_managed_workspace(missing) is True
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, extra_gate, missing
        )
        is True
    )
    assert (
        Daemon._exception_is_missing_managed_workspace(
            RuntimeError("callback_authority_incomplete_blocked")
        )
        is False
    )
    monkeypatch.setattr(
        daemon_module,
        "active_codex_exec_workers",
        lambda *_args, **_kwargs: [{"pid": 3112389}],
    )
    assert (
        Daemon._extra_gate_provider_pre_dispatch_should_defer(
            daemon, extra_gate, boom
        )
        is False
    )
    deferred = DatabasePortalProviderRouteDeferred(
        "extra_gate_provider_pre_dispatch:RuntimeError",
        backoff_seconds=20,
    )
    assert isinstance(deferred, DatabasePortalBridgeError)
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_successor_session_preserves_unbound_extra_gate_in_progress(
    tmp_path: Path,
) -> None:
    """A different process/session without exact claim evidence stays held."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    population = _population(1)
    population["tasks"][0]["task_id"] = "PCTDD-006"
    seed = _open_daemon(
        tmp_path,
        session="embedded-store:predecessor-006",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_claim",
            reason="claimed",
        )
        receipt["owner_session_id"] = "embedded-store:predecessor-006"
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="in_progress",
            receipt=receipt,
        )
        stuck = seed.task_source.get("task:cid:001")
        assert stuck is not None and stuck.status == "in_progress"
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="embedded-store:successor-006",
        max_task_attempts=2,
    )
    try:
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert rearms == []
        recovered = successor.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "in_progress"
        assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
            {
                "safe_to_restart": False,
                "blocked": True,
                "quiesced": False,
                "reconciled": False,
                "reason": "database_portal_retained_reconciliation_blocked",
            }
        )
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_same_session_preserves_unbound_extra_gate_in_progress(
    tmp_path: Path,
) -> None:
    """A different process/session without exact claim evidence stays held."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    population = _population(1)
    population["tasks"][0]["task_id"] = "PCTDD-006"
    daemon = _open_daemon(
        tmp_path,
        session="embedded-store:same-session-006",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        receipt = daemon._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_claim",
            reason="claimed",
        )
        receipt["owner_session_id"] = "embedded-store:same-session-006"
        receipt["process_instance_id"] = "process:dead-claim-without-grok"
        daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="in_progress",
            receipt=receipt,
        )
        stuck = daemon.task_source.get("task:cid:001")
        assert stuck is not None and stuck.status == "in_progress"
        rearms = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert rearms == []
        recovered = daemon.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "in_progress"
        assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
            {
                "safe_to_restart": False,
                "blocked": True,
                "quiesced": False,
                "reconciled": False,
                "reason": "database_portal_retained_reconciliation_blocked",
            }
        )
    finally:
        daemon.close()


def test_extra_gate_claimable_retry_skips_invalid_provider_route_cooldown() -> None:
    """Extra-gate nested-state-changed receipts are not cooldown-invalid.

    Home daemons were idling on database_provider_route_cooldown_invalid
    after saga-replaying PCTDD-034 because authority_outcome audit fields
    fail the exact cooldown census.  Ordinary PCTDD-001 missing-attempt
    cooldowns stay typed-invalid.  Extra-gate aliases still cannot bypass
    safe_to_restart=False.
    """

    Daemon = daemon_module.DatabaseImplementationDaemon
    receipt = {
        "schema": daemon_module.DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": "database_retry_rearmed",
        "reason": "provider_route_deferred_rearmed",
        "forced_block": False,
        "retry_exhausted": False,
        "attempt_consumed": False,
        "backoff_seconds": 20,
        "retry_not_before_ms": 1,
        "authority_outcome": "rearmed",
        "previous_operation": "database_unknown_outcome_blocked",
        "attempt_id": "attempt:a25c7f1da71e4a2ca575e4dd2275b7d5",
    }

    class _Probe(Daemon):
        def get_attempt(self, attempt_id: str) -> None:  # type: ignore[override]
            return None

        def _now_ms(self) -> int:  # type: ignore[override]
            return 0

    probe = _Probe.__new__(_Probe)
    extra_gate = SimpleNamespace(
        status="retrying",
        task_alias="PCTDD-034",
        task_cid="baguqeerali4k6zayrolznqdh23y4xcpnznnowygnnx6vvhsdixztv7peiada",
        body={"completion_receipt": receipt},
    )
    ordinary = SimpleNamespace(
        status="retrying",
        task_alias="PCTDD-001",
        task_cid="task:cid:001",
        body={"completion_receipt": receipt},
    )
    assert probe._task_alias_is_extra_gate(extra_gate) is True
    assert probe._task_alias_is_extra_gate(ordinary) is False
    assert probe._extra_gate_retry_receipt_is_claimable(extra_gate, receipt) is True
    assert probe._provider_route_retry_cooldown_state(extra_gate) == "not_applicable"
    assert probe._provider_route_retry_cooldown_state(ordinary) == "invalid"


@pytest.mark.parametrize("alias,cid", [
    ("PCTDD-005", "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza"),
    ("PCTDD-006", "baguqeerah7muo423u3xf5gi32hazctify2i55cavbdugzzythfqdl4wyif6a"),
    ("PCTDD-007", "baguqeerazst6lunrikvyslwfqzfbqbpwiivb5hxjsdzwvd7jjsqnnfpadwuq"),
    ("PCTDD-034", "baguqeerali4k6zayrolznqdh23y4xcpnznnowygnnx6vvhsdixztv7peiada"),
])
@pytest.mark.parametrize("error_type,error", [
    ("DatabaseImplementationConflictError", "terminal phase changed its actual database disposition"),
    ("ContractValidationError", "canonical proof contracts cannot contain floats"),
])
def test_terminal_receipt_validation_failure_preserves_dispatch_fence(
    alias: str, cid: str, error_type: str, error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task identity cannot turn an invalid terminal receipt into retry authority."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:extra-gate-terminal-disposition",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    recon = {
        "reconciled": False,
        "blocked": True,
        "reason": "database_portal_terminal_repair_batch_blocked",
        "trigger": "database_daemon_startup",
        "owner_session_id": daemon.owner_session_id,
        "active_attempt_count": 0,
        "current_process_attempt_count_skipped": 0,
        "reconciled_attempt_count": 0,
        "repair_batch_pending": False,
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
        "attempts": [
            {
                "reconciled": False,
                "blocked": True,
                "reason": "terminal_reconciliation_receipt_repair_failed",
                "trigger": "database_daemon_startup",
                "attempt_id": "attempt:f49b4e967c5c4fa7a358b4c2837094d4",
                "claim_id": "claim:5bec03a98d4341498cbeb897e92d6a58",
                "task_cid": cid,
                "task_alias": alias,
                "error_type": error_type,
                "error": error,
            }
        ],
    }
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
    try:
        daemon.materialize_population(_population(1))
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = False
        monkeypatch.setattr(
            daemon,
            "reconcile_quiesced_database_portal_attempts",
            lambda **_kwargs: recon,
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_expired_running_attempts",
            lambda apply_selection=False: [],
        )
        monkeypatch.setattr(
            daemon,
            "_project_preserved_database_portal_callback_evidence",
            lambda _recon: [],
        )
        result = daemon.run_once()
        assert result.get("selection_idle_reason") == (
            "database_portal_reconciliation_blocked"
        )
        assert result.get("claimed_task_cid") in (None, "")
        assert provider_calls == []
    finally:
        daemon.close()


def test_invalid_terminal_receipt_never_enters_retry_mutation_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the terminal barrier before acquiring any retry authority."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:extra-gate-mutation-fence",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    recon = {
        "reconciled": False,
        "blocked": True,
        "reason": "database_portal_terminal_repair_batch_blocked",
        "trigger": "database_daemon_startup",
        "owner_session_id": daemon.owner_session_id,
        "active_attempt_count": 0,
        "repair_batch_pending": False,
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
        "attempts": [
            {
                "reconciled": False,
                "blocked": True,
                "reason": "terminal_reconciliation_receipt_repair_failed",
                "task_alias": "PCTDD-034",
                "task_cid": (
                    "baguqeerali4k6zayrolznqdh23y4xcpnznnowygnnx6vvhsdixztv7peiada"
                ),
                "error_type": "DatabaseImplementationConflictError",
                "error": "terminal phase changed its actual database disposition",
            }
        ],
    }
    assert daemon_module.DatabaseImplementationDaemon._mutation_fence_lock_timeout(
        TimeoutError(
            "timed out acquiring DuckDB process lock: "
            "/tmp/quack-owner/write-transaction.lock"
        )
    )
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
    try:
        daemon.materialize_population(_population(1))
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = False

        def raise_fence(**_kwargs: object) -> list[dict[str, object]]:
            raise TimeoutError(
                "timed out acquiring DuckDB process lock: "
                "/tmp/quack-owner/write-transaction.lock"
            )

        monkeypatch.setattr(
            daemon,
            "reconcile_quiesced_database_portal_attempts",
            lambda **_kwargs: recon,
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            raise_fence,
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_expired_running_attempts",
            lambda apply_selection=False: [],
        )
        monkeypatch.setattr(
            daemon,
            "_project_preserved_database_portal_callback_evidence",
            lambda _recon: [],
        )
        result = daemon.run_once()
        assert result.get("selection_idle_reason") == (
            "database_portal_reconciliation_blocked"
        )
        assert result.get("claimed_task_cid") in (None, "")
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.parametrize("alias", ["PCTDD-005", "PCTDD-006", "PCTDD-007", "PCTDD-034"])
@pytest.mark.parametrize("failure", ["missing_state", "launch_birth", "fenced_runner_open_claim"])
def test_unresolved_callback_never_opens_ordinary_dispatch(
    alias: str, failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent state or a dead provider is not closed callback authority."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:extra-gate-missing-nested-state",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    recon = {
        "reconciled": False,
        "blocked": True,
        "reason": "database_portal_attempt_reconciliation_blocked",
        "trigger": "database_daemon_startup",
        "owner_session_id": daemon.owner_session_id,
        "active_attempt_count": 1,
        "current_process_attempt_count_skipped": 0,
        "reconciled_attempt_count": 1,
        "repair_batch_pending": False,
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
        "attempts": [
            {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "attempt_id": "attempt:3b93e2f61cbe48eab1b0f8b923c7c7d6",
                "claim_id": "claim:873f9db643774acd9fa5913a4ae2e4c9",
                "task_cid": (
                    "baguqeerah7muo423u3xf5gi32hazctify2i55cavbdugzzythfqdl4wyif6a"
                ),
                "task_alias": "PCTDD-006",
                "nested_state": {
                    "present": False,
                    "state_path": str(tmp_path / "portal-task-state.json"),
                    "state_digest": "",
                    "active": False,
                },
                "provider_runner_fence": {
                    "applicable": False,
                    "safe_to_restart": True,
                    "fenced": False,
                    "reason": "ordinary_provider_runner_receipt_absent",
                },
                "portal_reconciliation": {
                    "reconciled": False,
                    "blocked": True,
                    "reason": "provider_forbidden_terminal_recovery_blocked",
                    "provider_forbidden_terminal_recovery": {
                        "reconciled": False,
                        "blocked": True,
                        "applicable": True,
                        "reason": (
                            "provider_forbidden_terminal_recovery_state_invalid"
                        ),
                        "provider_dispatched": False,
                        "implementation_dispatched": False,
                        "state_reason": "missing_state_file",
                    },
                },
            }
        ],
    }
    item = recon["attempts"][0]
    item["task_alias"] = alias
    if failure == "launch_birth":
        item["nested_state"] = {
            "active": True, "active_phase": "implementing",
            "active_phase_detail": "provider_launch_birth",
        }
    elif failure == "fenced_runner_open_claim":
        item["provider_runner_fence"] = {
            "applicable": True, "fenced": True, "safe_to_restart": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
        }
        item["portal_reconciliation"] = {
            "blocked": True, "reason": "task_claim_reconciliation_blocked",
        }
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
    try:
        daemon.materialize_population(_population(1))
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = False
        monkeypatch.setattr(
            daemon,
            "reconcile_quiesced_database_portal_attempts",
            lambda **_kwargs: recon,
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            lambda: [],
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_expired_running_attempts",
            lambda apply_selection=False: [],
        )
        monkeypatch.setattr(
            daemon,
            "_project_preserved_database_portal_callback_evidence",
            lambda _recon: [],
        )
        result = daemon.run_once()
        assert result.get("selection_idle_reason") == (
            "database_portal_reconciliation_blocked"
        )
        assert result.get("claimed_task_cid") in (None, "")
        assert provider_calls == []
    finally:
        daemon.close()


def test_extra_gate_may_claim_off_hash_home_shard() -> None:
    """Retrying extra-gate work is selectable off hash-home.

    PCTDD-005/006/007 hash to shard 0. Strict sharding left lanes 1-3 on
    no_ready_tasks while lane-0 was fail-closed. Extra-gate aliases still
    cannot bypass safe_to_restart=False.
    """

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    class _Task:
        task_alias = "PCTDD-006"
        task_cid = (
            "baguqeerah7muo423u3xf5gi32hazctify2i55cavbdugzzythfqdl4wyif6a"
        )

    class _Other:
        task_alias = "PCTDD-008"
        task_cid = "baguqeera-not-extra-gate"

    daemon = daemon_module.DatabaseImplementationDaemon.__new__(
        daemon_module.DatabaseImplementationDaemon
    )
    daemon.strict_task_sharding = True
    daemon.task_shard_count = 4
    daemon.task_shard_index = 2
    assert daemon._extra_gate_or_home_shard(_Task()) is True
    assert daemon._extra_gate_or_home_shard(_Other()) is False
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )



def test_terminal_retry_label_cannot_skip_reconciliation_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical retry labels cannot authorize same-pass rearm."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:extra-gate-continuation-rearm",
        max_task_attempts=2,
    )
    recon = {
        "reconciled": True,
        "blocked": False,
        "reason": "database_portal_post_cas_transitions_reconciled",
        "trigger": "database_daemon_startup",
        "owner_session_id": daemon.owner_session_id,
        "active_attempt_count": 1,
        "reconciled_attempt_count": 15,
        "continuation_required": True,
        "repair_batch_pending": False,
        "reconciliation_complete": False,
        "quiesced": False,
        "safe_to_restart": False,
        "attempts": [
            {
                "reconciled": True,
                "blocked": False,
                "reason": "terminal_reconciliation_extra_gate_retry_authority",
                "task_alias": "PCTDD-005",
                "task_cid": (
                    "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza"
                ),
                "error_type": "DatabaseImplementationConflictError",
                "error": "terminal phase changed its actual database disposition",
            }
        ],
    }
    assert not PortalImplementationSupervisor._retained_startup_allows_normal_launch(
        {
            "safe_to_restart": False,
            "blocked": True,
            "quiesced": False,
            "reconciled": False,
            "reason": "database_portal_retained_reconciliation_blocked",
        }
    )
    rearm_outcome = [
        {
            "task_cid": (
                "baguqeeralebfcpvwg72mkrku5nngr6kuda22x6bqx257fi4w3ztelab56iza"
            ),
            "task_alias": "PCTDD-005",
            "operation": "database_unknown_outcome_rearmed",
            "rearmed": True,
        }
    ]
    try:
        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = False
        monkeypatch.setattr(
            daemon,
            "reconcile_quiesced_database_portal_attempts",
            lambda **_kwargs: recon,
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            lambda: rearm_outcome,
        )
        monkeypatch.setattr(
            daemon,
            "_project_preserved_database_portal_callback_evidence",
            lambda _recon: [],
        )
        result = daemon.run_once()
        assert result.get("selection_idle_reason") == (
            "database_portal_reconciliation_completed"
        )
        assert not result.get("unknown_outcome_rearms")
        assert result.get("claimed_task_cid") in (None, "")
    finally:
        daemon.close()


def test_nested_state_changed_landed_recovery_does_not_fence_ready_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:nested-state-changed-ready-dispatch",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(2))
        landed = daemon.task_source.get("task:cid:001")
        assert landed is not None
        receipt = daemon._retry_budget_receipt(
            landed,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        receipt.update(
            {
                "authority_outcome": "unknown",
                "forced_block": True,
                "terminal_reconciliation": {"schema": "candidate"},
            }
        )
        daemon._cas_task_status_database(
            landed.task_cid,
            expected_revision=int(landed.revision),
            new_status="blocked",
            receipt=receipt,
        )

        def raise_nested_state_changed(*, task, bridge):
            raise DatabasePortalTerminalQuiescentStateAdvanced(
                "blocked terminal landed recovery nested state changed"
            )

        daemon._database_portal_bridge = object()
        daemon._database_portal_reconciliation_checked = True
        daemon._database_portal_reconciliation_result = {"blocked": False}
        monkeypatch.setattr(
            daemon,
            "_reconcile_one_blocked_terminal_landed_task",
            raise_nested_state_changed,
        )

        result = daemon.run_once()

        assert result["claimed_task_cid"] == "task:cid:002"
        assert provider_calls == ["task:cid:002"]
        assert result.get("selection_idle_reason") != (
            "database_no_provider_rearm_recovery_fenced"
        )
        assert not any(
            item.get("reason") == "terminal_landed_candidate_recovery_blocked"
            for item in result.get("unknown_outcome_rearms") or []
        )
        blocked = daemon.task_source.get("task:cid:001")
        assert blocked is not None and blocked.status == "blocked"
    finally:
        daemon.close()


def test_terminal_state_advance_message_without_typed_signal_stays_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the bridge-owned typed transition may enter generic rearm."""

    seed = _open_daemon(
        tmp_path,
        session="session:untyped-terminal-state-message",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        receipt.update(
            {
                "authority_outcome": "unknown",
                "forced_block": True,
                "terminal_reconciliation": {"schema": "candidate"},
            }
        )
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:untyped-terminal-state-message-successor",
        max_task_attempts=2,
    )

    def raise_unrelated_failure(*, task, bridge):
        raise DatabasePortalBridgeError(
            "blocked terminal landed recovery nested state changed"
        )

    try:
        successor._database_portal_bridge = object()
        monkeypatch.setattr(
            successor,
            "_reconcile_one_blocked_terminal_landed_task",
            raise_unrelated_failure,
        )

        outcomes = successor.reconcile_blocked_terminal_landed_tasks()

        assert len(outcomes) == 1
        assert outcomes[0]["task_cid"] == "task:cid:001"
        assert outcomes[0]["blocked"] is True
        assert outcomes[0]["reason"] == (
            "terminal_landed_candidate_recovery_blocked"
        )
        assert outcomes[0]["error_type"] == "DatabasePortalBridgeError"
    finally:
        successor.close()


def test_read_only_terminal_quarantine_does_not_starve_ready_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-quarantine-ready-dispatch",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        quarantine = {
            "task_cid": "task:cid:quarantined",
            "task_alias": "PCTDD-QUARANTINED",
            "operation": "database_terminal_landed_completion",
            "recovered": False,
            "rearmed": False,
            "blocked": True,
            "reason": "terminal_landed_candidate_policy_invalid",
        }
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            lambda: [quarantine],
        )

        result = daemon.run_once()

        assert result["claimed_task_cid"] == "task:cid:001"
        assert provider_calls == ["task:cid:001"]
        assert result["unknown_outcome_rearms"] == [quarantine]
    finally:
        daemon.close()


def test_terminal_recovery_blocker_still_fences_ready_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-recovery-blocks-ready-dispatch",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        blocked = {
            "task_cid": "task:cid:quarantined",
            "task_alias": "PCTDD-QUARANTINED",
            "operation": "database_terminal_landed_completion",
            "recovered": False,
            "rearmed": False,
            "blocked": True,
            "reason": "terminal_landed_candidate_recovery_blocked",
            "error_type": "DatabaseImplementationConflictError",
            "error": "partial barrier state requires another recovery pass",
        }
        monkeypatch.setattr(
            daemon,
            "reconcile_blocked_unknown_outcome_tasks",
            lambda: [blocked],
        )

        result = daemon.run_once()

        assert result["implementation_result"] is None
        assert result["selection_idle_reason"] == (
            "database_no_provider_rearm_recovery_fenced"
        )
        assert result["unknown_outcome_rearms"] == [blocked]
        assert provider_calls == []
    finally:
        daemon.close()


def test_terminal_candidate_quarantine_covers_full_bounded_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-selector-full-page",
        max_task_attempts=2,
    )
    terminal_tasks = []
    for index in range(128):
        task_cid = f"task:cid:terminal:{index:03d}"
        terminal_tasks.append(
            SimpleNamespace(
                task_cid=task_cid,
                task_alias=f"PCTDD-TERMINAL-{index:03d}",
                revision=1,
                status="blocked",
                body={
                    "completion_receipt": {
                        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                        "operation": "database_unknown_outcome_blocked",
                        "reason": "callback_authority_incomplete_blocked",
                        "forced_block": True,
                        "authority_outcome": "unknown",
                        "terminal_reconciliation": {"schema": "candidate"},
                    }
                },
            )
        )
    overflow = SimpleNamespace(
        task_cid="task:cid:terminal:overflow",
        task_alias="PCTDD-TERMINAL-OVERFLOW",
        revision=1,
        status="blocked",
        body={
            "completion_receipt": {
                "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                "operation": "database_retry_exhausted",
                "reason": "portal_provider_failed",
                "retry_exhausted": True,
                "terminal_reconciliation": {"schema": "malformed"},
            }
        },
    )
    tasks = (*terminal_tasks, overflow)
    cas_calls: list[str] = []
    try:
        daemon._database_portal_bridge = object()
        monkeypatch.setattr(
            daemon.task_source,
            "list_tasks",
            lambda **_kwargs: SimpleNamespace(tasks=tasks),
        )
        monkeypatch.setattr(
            daemon,
            "_automatic_claim_forbidden",
            lambda _task: False,
        )
        monkeypatch.setattr(
            daemon,
            "_reconcile_one_blocked_terminal_landed_task",
            lambda *, task, bridge: {
                "task_cid": str(task.task_cid),
                "task_alias": str(task.task_alias),
                "operation": "database_terminal_landed_completion",
                "recovered": False,
                "rearmed": False,
                "blocked": True,
                "reason": "terminal_landed_candidate_recovery_blocked",
            },
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lambda task_cid, **_kwargs: cas_calls.append(str(task_cid)),
        )

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert len(outcomes) == 129
        overflow_outcomes = [
            item
            for item in outcomes
            if item.get("task_cid") == overflow.task_cid
        ]
        assert len(overflow_outcomes) == 1
        assert overflow_outcomes[0]["reason"] == (
            "terminal_landed_candidate_policy_invalid"
        )
        assert overflow_outcomes[0]["rearmed"] is False
        assert overflow.task_cid not in cas_calls
    finally:
        daemon.close()


def test_post_effect_dispatch_journal_failure_blocks_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        return {"status": "applied", "effect_key": "external:one"}

    daemon = _open_daemon(
        tmp_path,
        session="session:effect-post-return",
        effect_fn=effect,
        max_task_attempts=3,
    )

    original_record = daemon._record_callback_dispatch_outcome

    def fail_after_effect(*args: object, **kwargs: object) -> None:
        if kwargs.get("dispatch_kind") == "effect" and kwargs.get("outcome") == "returned":
            raise RuntimeError("lost effect return journal")
        original_record(*args, **kwargs)

    monkeypatch.setattr(daemon, "_record_callback_dispatch_outcome", fail_after_effect)
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        monkeypatch.setattr(
            daemon, "_record_callback_dispatch_outcome", original_record
        )
        daemon.run_once()
        assert len(effect_calls) == 1
    finally:
        daemon.close()


@pytest.mark.parametrize("dispatch_kind", ["provider", "effect"])
@pytest.mark.parametrize("failure_site", ["phase_event", "committed_journal"])
def test_post_phase_callback_bookkeeping_failure_resumes_exact_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatch_kind: str,
    failure_site: str,
) -> None:
    provider_attempts: list[str] = []
    effect_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        return {"status": "ok", "attempt_id": attempt.attempt_id}

    def effect(
        attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        effect_attempts.append(attempt.attempt_id)
        return {
            "status": "applied",
            "effect_key": "external:post-phase",
            "attempt_id": attempt.attempt_id,
        }

    daemon = _open_daemon(
        tmp_path,
        session=f"session:post-phase:{dispatch_kind}:{failure_site}",
        provider_fn=provider,
        effect_fn=effect,
        max_task_attempts=2,
    )
    injected = {"done": False}
    if failure_site == "phase_event":
        original_event = daemon._record_event
        target_phase = (
            ATTEMPT_PHASE_PROVIDER
            if dispatch_kind == "provider"
            else ATTEMPT_PHASE_EFFECT
        )

        def fail_committed_phase_event(
            event_type: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            body = kwargs.get("body")
            if (
                not injected["done"]
                and event_type == "attempt_phase_committed"
                and isinstance(body, dict)
                and body.get("phase") == target_phase
            ):
                injected["done"] = True
                raise RuntimeError("lost committed phase event response")
            original_event(event_type, *args, **kwargs)

        monkeypatch.setattr(daemon, "_record_event", fail_committed_phase_event)
    else:
        original_dispatch = daemon._record_callback_dispatch_outcome

        def fail_committed_dispatch_journal(
            *args: object,
            **kwargs: object,
        ) -> None:
            if (
                not injected["done"]
                and kwargs.get("dispatch_kind") == dispatch_kind
                and kwargs.get("outcome") == "committed"
            ):
                injected["done"] = True
                raise RuntimeError("lost committed dispatch journal response")
            original_dispatch(*args, **kwargs)

        monkeypatch.setattr(
            daemon,
            "_record_callback_dispatch_outcome",
            fail_committed_dispatch_journal,
        )

    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        pending_result = pending["implementation_result"]
        assert pending_result["status"] == (
            "callback_commit_reconciliation_pending"
        )
        assert pending_result["dispatch_kind"] == dispatch_kind
        assert pending_result["retry_budget_consumed"] is False
        attempt_id = pending_result["attempt_id"]
        attempt = daemon.get_attempt(attempt_id)
        assert attempt is not None and attempt.status == "running"
        assert attempt.phase_committed(
            ATTEMPT_PHASE_PROVIDER
            if dispatch_kind == "provider"
            else ATTEMPT_PHASE_EFFECT
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        assert task.body["completion_receipt"]["attempts_used"] == 1

        recovered = daemon.run_once()
        recovered_result = recovered["implementation_result"]
        assert recovered_result["status"] == "succeeded"
        assert recovered_result["attempt"]["attempt_id"] == attempt_id
        assert provider_attempts == [attempt_id]
        assert effect_attempts == [attempt_id]
        assert injected["done"] is True
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("max_attempts", "expected_status"),
    [(2, "retrying"), (1, "blocked")],
)
def test_missing_claim_history_rearms_or_blocks_canonical_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_attempts: int,
    expected_status: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:missing-claim:{max_attempts}",
        max_task_attempts=max_attempts,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        monkeypatch.setattr(daemon.coordinator, "get_task_claim", lambda _claim: None)
        outcomes = daemon.reconcile_expired_running_attempts()
        assert outcomes[0]["authority_outcome"] == "unknown"
        assert outcomes[0]["status"] == expected_status
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == expected_status
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
    finally:
        daemon.close()


def test_failure_status_cas_outage_recovers_without_provider_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:failure-cas",
        provider_fn=fail,
        max_task_attempts=2,
    )
    original_cas = daemon._cas_task_status_database
    injected = {"done": False}

    def fail_retry_cas(*args: object, **kwargs: object) -> object:
        if kwargs.get("new_status") in {"retrying", "blocked"} and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("injected retry CAS outage")
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(daemon, "_cas_task_status_database", fail_retry_cas)
    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        assert pending["implementation_result"]["status"] == (
            "failure_reconciliation_pending"
        )
        assert len(calls) == 1
        recovered = daemon.run_once()
        assert recovered["implementation_result"]["status"] == "failed"
        assert len(calls) == 1
    finally:
        daemon.close()


def test_claim_insert_failure_is_compensated_before_any_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-insert",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_insert = daemon._insert_attempt_from_claim
    monkeypatch.setattr(
        daemon,
        "_insert_attempt_from_claim",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("insert outage")),
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["implementation_result"] is None
        task = daemon.task_source.get("task:cid:001")
        assert task is not None and task.status == "retrying"
        assert calls == []
        monkeypatch.setattr(daemon, "_insert_attempt_from_claim", original_insert)
        daemon.run_once()
        assert calls == ["task:cid:001"]
    finally:
        daemon.close()


def test_claim_insert_and_compensation_cas_failure_recovers_next_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-insert-compensation",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_insert = daemon._insert_attempt_from_claim
    original_cas = daemon._cas_task_status_database
    injected = {"insert": False, "compensation": False}

    def fail_first_insert(*args: object, **kwargs: object) -> object:
        if not injected["insert"]:
            injected["insert"] = True
            raise RuntimeError("insert outage")
        return original_insert(*args, **kwargs)

    def fail_first_compensation(*args: object, **kwargs: object) -> object:
        if (
            kwargs.get("new_status") in {"retrying", "blocked"}
            and not injected["compensation"]
        ):
            injected["compensation"] = True
            raise RuntimeError("compensation CAS outage")
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(daemon, "_insert_attempt_from_claim", fail_first_insert)
    monkeypatch.setattr(daemon, "_cas_task_status_database", fail_first_compensation)
    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        assert pending["implementation_result"] is None
        stranded = daemon.task_source.get("task:cid:001")
        assert stranded is not None and stranded.status == "in_progress"
        assert daemon.list_running_attempts() == []
        assert calls == []

        # The next pass consumes the durable database_claim marker, restores
        # retryable state, releases the old claim, and opens one fresh attempt.
        recovered = daemon.run_once()
        assert recovered["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:001"]
        assert recovered["write_count"] >= 2
    finally:
        daemon.close()


def test_nonpassing_validation_consumes_only_global_retry_budget(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    validation_calls: list[str] = []

    def reject(attempt: DatabaseTaskAttempt, _effect: object) -> dict[str, object]:
        validation_calls.append(attempt.attempt_id)
        return {"outcome": "failed", "evidence_digest": "sha256:rejected"}

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-nonpass",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=reject,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        second = daemon.run_once()
        idle = daemon.run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(provider_calls) == len(effect_calls) == len(validation_calls) == 2
        assert len(set(validation_calls)) == 2
    finally:
        daemon.close()


def test_malformed_retry_receipt_fails_closed_without_dispatch(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:malformed-retry",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        population = _population(1)
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        malformed = _population(1)
        malformed_task = malformed["tasks"][0]
        assert isinstance(malformed_task, dict)
        malformed_task["completion_receipt"] = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "task_cid": task.task_cid,
            "validation_spec_cid": daemon._retry_budget_validation_spec_cid(task),
            "attempts_used": "not-an-integer",
        }
        daemon.materialize_population(malformed)

        blocked = daemon.run_once()
        assert blocked["implementation_result"] is None
        assert blocked["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = blocked["retry_budget_backpressure"]["tasks"][0]
        assert entry["malformed"] is True
        assert provider_calls == []
    finally:
        daemon.close()


def test_orphan_recovery_preserves_malformed_retry_latch(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:malformed-orphan",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        daemon.sync_ready_tasks_into_coordination()
        claim = daemon.coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            lease_ms=daemon.lease_ms,
            now_ms=daemon._now_ms(),
        )
        assert claim is not None
        task = daemon.task_source.get(claim.task_cid)
        assert task is not None
        malformed_receipt = daemon._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_claim",
            attempt=claim,
        )
        malformed_receipt["attempts_used"] = "not-an-integer"
        daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=task.revision,
            new_status="in_progress",
            receipt=malformed_receipt,
        )

        outcomes = daemon.reconcile_orphaned_canonical_claims()
        assert outcomes[0]["status"] == "blocked"
        recovered = daemon.task_source.get(task.task_cid)
        assert recovered is not None and recovered.status == "blocked"
        assert recovered.body["completion_receipt"]["malformed"] is True
        assert recovered.body["completion_receipt"]["retry_exhausted"] is True

        idle = daemon.run_once()
        assert idle["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = idle["retry_budget_backpressure"]["tasks"][0]
        assert entry["task_cid"] == task.task_cid
        assert entry["malformed"] is True
    finally:
        daemon.close()


def test_retry_backpressure_distinguishes_some_from_all_selectable_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:partial-backpressure",
        task_prefix="DQP-T",
        max_task_attempts=1,
    )
    try:
        population = _population(2)
        daemon.materialize_population(population)
        first = daemon.task_source.get("task:cid:001")
        assert first is not None
        population_task = population["tasks"][0]
        assert isinstance(population_task, dict)
        population_task["completion_receipt"] = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "task_cid": first.task_cid,
            "validation_spec_cid": daemon._retry_budget_validation_spec_cid(first),
            "attempts_used": 1,
        }
        daemon.materialize_population(population)
        monkeypatch.setattr(daemon, "claim_next", lambda: None)

        result = daemon.run_once()
        assert result["selection_idle_reason"] == (
            "some_selectable_tasks_reached_max_task_attempts"
        )
        backpressure = result["retry_budget_backpressure"]
        assert backpressure["any_eligible_exhausted"] is True
        assert backpressure["all_eligible_exhausted"] is False
        assert backpressure["eligible_ready_task_cids"] == ["task:cid:002"]
        assert result["retry_exhausted_task_cids"] == ["task:cid:001"]
    finally:
        daemon.close()


def test_claim_cas_unknown_response_is_reconciled_without_release_or_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-unknown",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_cas = daemon._cas_task_status_database
    injected = {"done": False}

    def commit_then_unknown(*args: object, **kwargs: object) -> object:
        result = original_cas(*args, **kwargs)
        if kwargs.get("new_status") == "in_progress" and not injected["done"]:
            injected["done"] = True
            raise DatabaseTaskSourceUnknownOutcomeError("lost CAS response")
        return result

    monkeypatch.setattr(daemon, "_cas_task_status_database", commit_then_unknown)
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:001"]
    finally:
        daemon.close()


def test_replacement_task_body_revokes_old_claim_before_provider(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:task-replacement",
        provider_calls=calls,
        max_task_attempts=2,
    )
    try:
        original = _population(1)
        daemon.materialize_population(original)
        old = daemon.claim_next()
        assert old is not None
        replacement = _population(1)
        replacement_task = replacement["tasks"][0]
        assert isinstance(replacement_task, dict)
        replacement_task["title"] = "Replacement task body"
        replacement_task["validation_commands"] = ["python -m pytest replacement.py"]
        daemon.materialize_population(replacement)
        revoked = daemon.run_once()
        assert revoked["implementation_result"]["callback_failure"] is True
        assert calls == []
        current = daemon.task_source.get(old.task_cid)
        assert current is not None and current.status == "ready"
        daemon.run_once()
        assert calls == [old.task_cid]
    finally:
        daemon.close()


def test_attempt_control_claim_rejects_numeric_revision_aliases(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:control-claim-numeric-alias",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        claim = dict(attempt.body["control_claim"])
        assert daemon._database_attempt_has_exact_control(attempt, task)
        for numeric_alias in (True, float(claim["revision"])):
            tampered = SimpleNamespace(
                task_cid=attempt.task_cid,
                body={
                    **dict(attempt.body),
                    "control_claim": {
                        **claim,
                        "revision": numeric_alias,
                    },
                },
            )
            assert not daemon._database_attempt_has_exact_control(
                tampered,
                task,
            )
    finally:
        daemon.close()


@pytest.mark.parametrize("numeric_alias", (True, 1.0))
def test_rearm_snapshot_comparison_rejects_numeric_aliases(
    numeric_alias: object,
) -> None:
    assert not _canonical_mapping_matches(
        {"body": {"revision": 1}},
        {"body": {"revision": numeric_alias}},
    )


def test_prepared_barrier_core_matches_nested_observational_floats() -> None:
    nested = {"state_digest": "sha256:" + "d" * 64, "age_seconds": 1.5}
    prepared = {
        "stage": "prepared",
        "receipt_id": "sha256:" + "a" * 64,
        "reason": "nested_quiesced",
        "nested_state": nested,
    }
    barrier = {
        "stage": "commit_barrier",
        "receipt_id": "sha256:" + "b" * 64,
        "prepared_reconciliation_receipt_id": prepared["receipt_id"],
        "reason": "nested_quiesced",
        "nested_state": nested,
    }
    assert _prepared_reconciliation_barrier_core_matches(prepared, barrier) is True
    assert (
        _prepared_reconciliation_barrier_core_matches(
            prepared, {**barrier, "reason": "other"}
        )
        is False
    )


def test_old_validation_epoch_failure_cannot_charge_replacement_claim(
    tmp_path: Path,
) -> None:
    old_lane = _open_daemon(
        tmp_path / "seed",
        session="session:old-validation-epoch",
        max_task_attempts=2,
    )
    new_coordinator = None
    new_lane = None
    try:
        old_lane.materialize_population(_population(1))
        old_attempt = old_lane.claim_next()
        assert old_attempt is not None

        replacement = _population(1)
        replacement_task = replacement["tasks"][0]
        assert isinstance(replacement_task, dict)
        replacement_task["validation_commands"] = [
            {"argv": ["python", "-m", "pytest", "replacement.py"]}
        ]
        old_lane.materialize_population(replacement)

        lane_root = tmp_path / "replacement-lane"
        new_coordinator = open_database_coordinator(
            lane_root / "coordination.duckdb"
        )
        new_lane = DatabaseImplementationDaemon(
            database_path=old_lane.database_path,
            coordination_path=lane_root / "coordination.duckdb",
            execution_path=lane_root / "execution.duckdb",
            owner_session_id="session:new-validation-epoch",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_source=old_lane.task_source,
            coordinator=new_coordinator,
            max_task_attempts=2,
        )
        new_attempt = new_lane.claim_next()
        assert new_attempt is not None
        before = new_lane.task_source.get(new_attempt.task_cid)
        assert before is not None and before.status == "in_progress"
        new_receipt = dict(before.body["completion_receipt"])
        assert new_receipt["attempt_id"] == new_attempt.attempt_id
        assert new_receipt["attempts_used"] == 1

        _failed, stale_receipt = old_lane._finalize_failed_attempt(
            old_attempt,
            reason="old validation callback failed after replacement",
        )
        assert stale_receipt["operation"] == (
            "database_superseded_attempt_revoked"
        )
        after = new_lane.task_source.get(new_attempt.task_cid)
        assert after is not None and after.status == "in_progress"
        assert after.body["completion_receipt"] == new_receipt
        assert old_lane.get_attempt(old_attempt.attempt_id).status == "failed"
    finally:
        if new_lane is not None:
            new_lane.close()
        if new_coordinator is not None:
            new_coordinator.close()
        old_lane.close()


def test_strict_shards_claim_only_home_lane_tasks(tmp_path: Path) -> None:
    seed = _open_daemon(tmp_path, session="session:seed")
    try:
        seed.materialize_population(_population(8))
    finally:
        seed.close()

    claimed: dict[int, str] = {}
    for index in range(4):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:shard-{index}",
            task_shard_count=4,
            task_shard_index=index,
            strict_task_sharding=True,
            task_prefix="DQP-T",
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"shard {index} found no home-lane work"
            alias = str(attempt.task_alias or "")
            home = daemon._task_home_shard_index(alias)
            assert home == index, f"{alias} home={home} claimed by shard {index}"
            claimed[index] = alias
        finally:
            daemon.close()

    assert len(set(claimed.values())) == 4


def test_no_markdown_status_update_under_database_authority(tmp_path: Path) -> None:
    markdown = tmp_path / "tasks.md"
    markdown.write_text(
        "# Tasks\n\n## DQP-T001 Work\n\n- Status: todo\n",
        encoding="utf-8",
    )
    before = markdown.read_text(encoding="utf-8")
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:md",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        markdown_path=markdown,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["markdown_status_writes"] == 0
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"
        with pytest.raises(DatabaseImplementationAuthorityError, match="Markdown"):
            daemon.write_markdown_task_status("DQP-T001", "completed")
        assert markdown.read_text(encoding="utf-8") == before
        assert "- Status: completed" not in markdown.read_text(encoding="utf-8")
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon = open_database_implementation_daemon(
        tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:proj",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        assert daemon.projections_required() is False
        assert daemon.state_path is None
        assert daemon.strategy_path is None
        assert daemon.events_path is None
        assert daemon.pid_path is None
        assert daemon.queue_path is None
        # No projection files created by open/materialize/run.
        daemon.materialize_population(_population(1))
        daemon.run_once()
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


def test_datasets_authoritative_open_requires_preinstalled_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    control_path = tmp_path / "missing-control.duckdb"
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="preinstalled by the trusted materializer",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not control_path.exists()
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_full_control_plane_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "full-control.duckdb"
    install_control_plane_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    with open_duckdb_connection(control_path) as connection:
        names = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "proof_obligations" in names
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_tampered_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "tampered-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    with open_duckdb_connection(control_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'sha256:tampered'"
        )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_verifies_existing_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "operational-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=control_path,
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        evidence = dict(daemon.control_schema_evidence)
        assert evidence["state_schema_revision"] == (
            "datasets-authoritative-operational-v1"
        )
        assert evidence["verified"] is True
        assert evidence["profile_id"]
        assert evidence["schema_fingerprint"]
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()


def test_crash_restart_resumes_without_duplicating_provider_or_effect(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    first = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == ["task:cid:001"]
        assert attempt.committed_phase == ATTEMPT_PHASE_PROVIDER
        # Crash boundary: process dies after provider commits, before effect.
        assert effect_calls == []
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        running = second.list_running_attempts()
        assert len(running) == 1
        assert running[0].attempt_id == attempt_id
        assert running[0].committed_phase == ATTEMPT_PHASE_PROVIDER
        result = second.resume_attempt(running[0])
        assert result["resumed"] is True
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert result["status"] == "succeeded"
        task = second.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"

        # Second resume of a finished attempt is a no-op for provider/effect.
        finished = second.get_attempt(attempt_id)
        assert finished is not None
        again = second.resume_attempt(finished)
        assert again["resumed"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        second.close()


def test_implicit_embedded_owner_is_store_scoped_and_restart_stable(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first_owner = first.owner_session_id
        assert first_owner.startswith("embedded-store:")
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, _, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        assert second.owner_session_id == first_owner
        result = second.run_once()
        assert result["implementation_result"]["provider_duplicated"] is True
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        second.close()


def test_implicit_embedded_owner_is_distinct_for_different_stores(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path / "first")
    second = _open_daemon(tmp_path / "second")
    try:
        assert first.owner_session_id.startswith("embedded-store:")
        assert second.owner_session_id.startswith("embedded-store:")
        assert first.owner_session_id != second.owner_session_id
    finally:
        second.close()
        first.close()


def test_embedded_writer_lock_rejects_a_concurrent_same_store_opener(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path)
    try:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="active database writer",
        ):
            _open_daemon(tmp_path)
        first.materialize_population(_population(1))
        assert first.claim_next() is not None
    finally:
        first.close()

    replacement = _open_daemon(tmp_path)
    try:
        assert replacement.owner_session_id == first.owner_session_id
    finally:
        replacement.close()


def test_effect_phase_resume_skips_both_provider_and_effect(tmp_path: Path) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, effect_result, _ = first.run_effect(attempt, provider_result)
        assert attempt.committed_phase == ATTEMPT_PHASE_EFFECT
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        attempt = second.get_attempt(attempt_id)
        assert attempt is not None
        result = second.resume_attempt(attempt)
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is True
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["status"] == "succeeded"
    finally:
        second.close()


def test_provider_heartbeat_renews_exact_task_claim(tmp_path: Path) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    observed_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        initial = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert initial is not None
        observed_revisions.append(int(initial.revision))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert current is not None
            if int(current.revision) > int(initial.revision):
                observed_revisions.append(int(current.revision))
                break
            time.sleep(0.005)
        assert len(observed_revisions) == 2, "background lease renewal did not run"
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:heartbeat",
        provider_fn=provider,
        lease_ms=5_000,
    )
    holder["daemon"] = daemon
    daemon._lease_heartbeat_interval_seconds = 0.01
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        updated, _, duplicated = daemon.run_provider(attempt)
        assert duplicated is False
        assert updated.committed_phase == ATTEMPT_PHASE_PROVIDER
        assert observed_revisions[1] > observed_revisions[0]
    finally:
        daemon.close()


def test_provider_result_is_rejected_after_fenced_takeover(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    replacement_claim_ids: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        # Cross the renewed deadline and let another session claim the same
        # ready coordination task before this provider result is returned.
        now["ms"] = 7_000
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:replacement",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.task_cid == attempt.task_cid
        replacement_claim_ids.append(replacement.claim_id)
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:stale-provider",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert replacement_claim_ids
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            is None
        )
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "context"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_expired_attempt_cannot_commit_logical_completion(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:expired-completion",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)
        now["ms"] = 6_000
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        assert daemon.coordinator.claimability(attempt.task_cid)["claimable"] is True
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "validation"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_restart_retires_prepared_absent_expired_attempt_then_refences_retry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        old_attempt = first.claim_next()
        assert old_attempt is not None
        old_attempt = first.commit_phase(old_attempt, "context")
        old_owner = first.owner_session_id
    finally:
        first.close()

    # No intervening coordinator mutation performs an expiry sweep.
    now["ms"] = 7_000
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert replacement.owner_session_id == old_owner
        expiry_result = replacement.run_once()
        reconciliations = expiry_result["expired_attempt_reconciliations"]
        assert len(reconciliations) == 1
        assert reconciliations[0]["status"] == "expired"
        assert reconciliations[0]["provider_evidence_reused"] is False
        assert reconciliations[0]["effect_evidence_reused"] is False
        assert expiry_result["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        # Expiry and a freshly fenced retry are separate durable passes.
        result = replacement.run_once()
        assert result["attempt_id"] != old_attempt.attempt_id
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [old_attempt.task_cid]
        assert effect_calls == [old_attempt.task_cid]
        retired = replacement.get_attempt(old_attempt.attempt_id)
        assert retired is not None
        assert retired.status == "failed"
        assert retired.committed_phase == "failed"
        replacement_claim = replacement.coordinator.get_task_claim(
            result["claim_id"]
        )
        assert replacement_claim is not None
        assert replacement_claim.fencing_token > old_attempt.fencing_token
    finally:
        replacement.close()


@pytest.mark.parametrize("callback_kind", ("provider", "effect"))
@pytest.mark.parametrize("result_state", ("missing", "corrupt"))
def test_expired_committed_callback_with_invalid_result_never_redispatches(
    tmp_path: Path,
    callback_kind: str,
    result_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    predecessor = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        predecessor.materialize_population(_population(1))
        attempt = predecessor.claim_next()
        assert attempt is not None
        attempt = predecessor.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = predecessor.run_provider(attempt)
        assert duplicated is False
        table = "provider_invocations"
        if callback_kind == "effect":
            attempt, _effect_result, duplicated = predecessor.run_effect(
                attempt,
                provider_result,
            )
            assert duplicated is False
            table = "effect_claims"
        connection = predecessor._require_connection()
        if result_state == "missing":
            connection.execute(
                f"DELETE FROM {table} WHERE attempt_id = ?",
                [attempt.attempt_id],
            )
        else:
            connection.execute(
                f"UPDATE {table} SET result_json = ? WHERE attempt_id = ?",
                ["{not-strict-json", attempt.attempt_id],
            )
        owner_session_id = predecessor.owner_session_id
    finally:
        predecessor.close()

    now["ms"] = 7_000
    successor = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert successor.owner_session_id == owner_session_id
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        assert expired["callback_authority_incomplete"] is True
        for _pass in range(2):
            later = successor.run_once()
            assert later["implementation_result"] is None
        count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()
        assert count is not None and int(count[0]) == 1
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == (
            [attempt.task_cid] if callback_kind == "effect" else []
        )
    finally:
        successor.close()


def test_completed_control_cas_is_recovered_from_prepared_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry after control CAS cannot expose an uncoordinated completion."""

    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_complete = daemon.coordinator.complete_task_claim

        def expire_at_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            kwargs["now_ms"] = now["ms"]
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            expire_at_promotion,
        )
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "b" * 64,
                    "argv": ["focused-validation"],
                },
            )

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        assert task.revision == 3
        readiness = daemon.coordinator.claimability(attempt.task_cid)
        assert readiness["claimable"] is False
        assert readiness["completion_status"] == "prepared"
        prepared = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert prepared is not None
        assert prepared["attempt_id"] == attempt.attempt_id
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        # Restore the ordinary method.  The next pass proves the exact control
        # receipt, promotes and settles the expired preparation, and repairs
        # the execution projection without rerunning provider/effect work.
        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            original_complete,
        )
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = daemon.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        assert recovered.committed_phase == ATTEMPT_PHASE_COMPLETE
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
    finally:
        daemon.close()


def test_restart_recovers_prepared_control_completion_without_prior_expiry_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = first.commit_phase(attempt, phase)

        def crash_before_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            raise RuntimeError("simulated crash before coordination promotion")

        monkeypatch.setattr(
            first.coordinator,
            "complete_task_claim",
            crash_before_promotion,
        )
        with pytest.raises(RuntimeError, match="before coordination promotion"):
            first.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "e" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        unswept = first.coordinator.get_task_claim(attempt.claim_id)
        assert unswept is not None
        assert unswept.state.value == "accepted"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = replacement.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        claim = replacement.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
        assert provider_calls == []
        assert effect_calls == []
    finally:
        replacement.close()


def test_promoted_completion_replays_after_local_phase_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:promotion-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_commit_phase = daemon.commit_phase

        def lose_local_complete(
            current: DatabaseTaskAttempt | str,
            phase: str,
            **kwargs: object,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_COMPLETE:
                raise RuntimeError("simulated local COMPLETE outage")
            return original_commit_phase(current, phase, **kwargs)

        monkeypatch.setattr(daemon, "commit_phase", lose_local_complete)
        with pytest.raises(RuntimeError, match="simulated local COMPLETE outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "d" * 64,
                    "argv": ["focused-validation"],
                },
            )
        promoted = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "accepted"
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert result["implementation_result"] is None
        assert len(result["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt.attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        assert provider_calls == []
        assert effect_calls == []
        settled = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert settled is not None
        assert settled.state.value == "released"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "failure_window",
    ["promotion_response_loss", "local_complete_outage"],
)
def test_run_once_preserves_promoted_completion_for_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_window: str,
) -> None:
    """The non-crashing wrapper must not turn durable success into FAILED."""

    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session=f"session:wrapper-{failure_window}",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        if failure_window == "promotion_response_loss":
            original_complete_claim = daemon.coordinator.complete_task_claim

            def lose_promotion_response(
                *args: object,
                **kwargs: object,
            ) -> object:
                original_complete_claim(*args, **kwargs)
                raise RuntimeError("simulated promotion response loss")

            monkeypatch.setattr(
                daemon.coordinator,
                "complete_task_claim",
                lose_promotion_response,
            )
        else:
            original_commit_phase = daemon.commit_phase

            def lose_local_complete(
                current: DatabaseTaskAttempt | str,
                phase: str,
                **kwargs: object,
            ) -> DatabaseTaskAttempt:
                if phase == ATTEMPT_PHASE_COMPLETE:
                    raise RuntimeError("simulated local COMPLETE outage")
                return original_commit_phase(current, phase, **kwargs)

            monkeypatch.setattr(daemon, "commit_phase", lose_local_complete)

        first = daemon.run_once()
        pending = first["implementation_result"]
        assert pending["status"] == "completion_reconciliation_pending"
        assert pending["retry_budget_consumed"] is False
        attempt_id = str(first["attempt_id"])
        claim_id = str(first["claim_id"])
        stored = daemon.get_attempt(attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"
        promoted = daemon.coordinator.get_prepared_task_completion(
            stored.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"

        if failure_window == "promotion_response_loss":
            monkeypatch.setattr(
                daemon.coordinator,
                "complete_task_claim",
                original_complete_claim,
            )
        else:
            monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)

        second = daemon.run_once()
        assert len(second["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        settled = daemon.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value in {"completed", "released"}
        assert provider_calls == [stored.task_cid]
        assert effect_calls == [stored.task_cid]
    finally:
        daemon.close()


def test_expired_preparation_without_control_cas_is_permanently_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-abort",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_cas = daemon._cas_task_status_database

        def reject_control_completion(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated control CAS outage")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            reject_control_completion,
        )
        with pytest.raises(RuntimeError, match="simulated control CAS outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "c" * 64,
                    "argv": ["focused-validation"],
                },
            )
        assert daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"

        monkeypatch.setattr(daemon, "_cas_task_status_database", original_cas)
        now["ms"] = 7_000
        result = daemon.run_once()
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "aborted"
        assert result["completion_reconciliations"][0]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert result["completion_reconciliations"][0]["retry_required"] is False
        assert result["implementation_result"] is None
        old_attempt = daemon.get_attempt(attempt.attempt_id)
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        final_completion = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert final_completion is None
        blocked = daemon.task_source.get(attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"
        assert blocked.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()

        callback_calls: list[str] = []

        def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
            callback_calls.append("callback")
            raise AssertionError("post-validation barrier was redispatched")

        daemon = _open_daemon(
            tmp_path,
            session="session:prepared-abort",
            provider_fn=forbidden,
            effect_fn=forbidden,
            validation_fn=forbidden,
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
        )
        for _pass in range(2):
            restarted = daemon.run_once()
            assert restarted["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_calls == []
    finally:
        daemon.close()


def test_task_claim_settlement_authority_loss_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:settlement-loss")
    try:
        daemon.materialize_population(_population(1))

        def reject_settlement(*args: object, **kwargs: object) -> object:
            raise DatabaseCoordinationError("simulated settlement authority loss")

        monkeypatch.setattr(
            daemon.coordinator,
            "settle_task_claim",
            reject_settlement,
        )
        with pytest.raises(
            DatabaseCoordinationError,
            match="simulated settlement authority loss",
        ):
            daemon.run_once()
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("restart_ms", "expected_claim_state"),
    ((2_000, "released"), (7_000, "completed")),
)
def test_restart_settles_promoted_completion_after_local_complete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_ms: int,
    expected_claim_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_settlement(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated crash before claim settlement")

        monkeypatch.setattr(
            first.coordinator,
            "settle_task_claim",
            crash_before_settlement,
        )
        with pytest.raises(RuntimeError, match="before claim settlement"):
            first.run_once()
        row = first._require_connection().execute(
            """
            SELECT attempt_id, claim_id FROM database_task_attempts
            WHERE status = 'succeeded'
            """
        ).fetchone()
        assert row is not None
        attempt_id, claim_id = str(row[0]), str(row[1])
        unsettled = first.coordinator.get_task_claim(claim_id)
        assert unsettled is not None
        assert unsettled.state.value == "accepted"
    finally:
        first.close()

    now["ms"] = restart_ms
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "succeeded"
        settled = replacement.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value == expected_claim_state
        local = replacement.get_attempt(attempt_id)
        assert local is not None
        assert local.status == "succeeded"
        assert replacement.coordinator.list_unsettled_task_completions() == []
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        replacement.close()


def test_automatic_run_once_never_claims_manual_or_review_only_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        population = _population(2)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        tasks[1]["review_only"] = True
        daemon.materialize_population(population)
        result = daemon.run_once()
        assert result["unchanged"] is True
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert daemon.list_running_attempts() == []
        assert daemon.coordinator.get_task_claim("claim:missing") is None
        for task_cid in ("task:cid:001", "task:cid:002"):
            task = daemon.task_source.get(task_cid)
            assert task is not None
            assert task.status == "ready"

        # The coordinator still exposes the task to a separately authorized
        # trusted manual-seal path; only automatic daemon dispatch is excluded.
        direct = daemon.coordinator.claim_task(
            task_cid="task:cid:001",
            owner_session_id="session:trusted-manual-seal",
            now_ms=daemon._now_ms(),
        )
        assert direct.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_parse_args_accepts_database_authority_flags() -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            "/tmp/control.duckdb",
            "--owner-session-id",
            "session:cli",
            "--once",
        ]
    )
    assert args.task_source_kind == "duckdb"
    assert args.authority_mode == "embedded"
    assert Path(args.database_path) == Path("/tmp/control.duckdb")
    assert args.owner_session_id == "session:cli"
    paths = resolve_database_implementation_paths(args)
    assert paths["database_path"] == Path("/tmp/control.duckdb")


def test_quack_runner_resolves_lane_private_database_paths(tmp_path: Path) -> None:
    def lane_args(index: int):
        return parse_args(
            [
                "--task-source-kind",
                "duckdb",
                "--authority-mode",
                "quack",
                "--database-path",
                str(tmp_path / "shared-control.duckdb"),
                "--coordination-path",
                str(tmp_path / "shared-coordination.duckdb"),
                "--quack-endpoint",
                "quack:127.0.0.1:45671",
                "--state-dir",
                str(tmp_path / f"lane-{index}"),
                "--state-prefix",
                f"pctdd_lane_{index}",
                "--once",
            ]
        )

    first = resolve_database_implementation_paths(lane_args(0))
    second = resolve_database_implementation_paths(lane_args(1))
    assert first["database_path"] == (
        tmp_path / "lane-0" / "quack-lane-control.duckdb"
    )
    assert second["database_path"] == (
        tmp_path / "lane-1" / "quack-lane-control.duckdb"
    )
    assert first["database_path"] != second["database_path"]
    assert first["coordination_path"] == (
        tmp_path / "lane-0" / "quack-lane-coordination.duckdb"
    )
    assert second["coordination_path"] == (
        tmp_path / "lane-1" / "quack-lane-coordination.duckdb"
    )
    assert first["coordination_path"] != second["coordination_path"]

    daemons = [
        DatabaseImplementationDaemon(
            database_path=paths["database_path"],
            coordination_path=paths["coordination_path"],
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri="quack:127.0.0.1:45671",
            control_store_id="store:quack-lane-test",
            control_store_generation="generation:quack-lane-test",
            install_schema=False,
        )
        for paths in (first, second)
    ]
    try:
        assert daemons[0].execution_path != daemons[1].execution_path
        assert daemons[0].coordination_path != daemons[1].coordination_path
        daemons[0]._acquire_embedded_writer_lock()
        daemons[1]._acquire_embedded_writer_lock()
    finally:
        for daemon in daemons:
            daemon.close()


def test_quack_builder_ignores_shared_database_keyword_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    captured: dict[str, object] = {}

    def fake_daemon(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(daemon_module, "DatabaseImplementationDaemon", fake_daemon)
    state_dir = tmp_path / "lane-2"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--database-path",
            str(tmp_path / "shared-cli.duckdb"),
            "--coordination-path",
            str(tmp_path / "shared-coordination.duckdb"),
            "--quack-endpoint",
            "quack:127.0.0.1:45671",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "generation-test",
            "--state-schema-revision",
            "schema-test",
            "--state-dir",
            str(state_dir),
            "--once",
        ]
    )

    build_database_implementation_daemon_from_args(
        args,
        database_path=tmp_path / "shared-keyword.duckdb",
    )
    assert captured["database_path"] == state_dir / "quack-lane-control.duckdb"
    assert captured["coordination_path"] == (
        state_dir / "quack-lane-coordination.duckdb"
    )


def test_runner_builds_database_daemon_without_json_projections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon = build_database_implementation_daemon_from_args(
        args,
        owner_session_id="session:runner",
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.state_path is None
        assert daemon.events_path is None
        assert daemon.max_task_attempts == 3
        assert daemon.projections_required() is False
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["authority_mode"] == "embedded"
        assert result["markdown_status_writes"] == 0
    finally:
        daemon.close()


def test_runner_portal_builder_selects_database_daemon(tmp_path: Path) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--task-prefix",
            "DQP-",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.max_task_attempts == 3
        assert context.state_path.name.startswith("dqp_")
        daemon.materialize_population(_population(2))
        first = daemon.claim_next()
        second = daemon.claim_next()
        # Single session claims one at a time via claim_ready; second claim is
        # a different task while the first remains leased.
        assert first is not None
        assert second is not None
        assert first.task_cid != second.task_cid
    finally:
        daemon.close()


def _deferred_provider_rearm_evidence(
    daemon: DatabaseImplementationDaemon,
    candidate: DatabaseTaskAttempt,
    outer_receipt: dict[str, object],
) -> dict[str, object]:
    """Build the exact reviewed deferred-provider proof used by recovery tests."""

    event_head_id = "sha256:" + "9" * 64
    evidence: dict[str, object] = {
        "schema": DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA,
        "attempt_id": candidate.attempt_id,
        "claim_id": candidate.claim_id,
        "task_cid": candidate.task_cid,
        "task_alias": candidate.task_alias,
        "attempt_number": candidate.attempt_number,
        "owner_session_id": candidate.owner_session_id,
        "lease_id": candidate.lease_id,
        "fencing_token": candidate.fencing_token,
        "fence_epoch": candidate.fence_epoch,
        "attempt_root_key": hashlib.sha256(
            candidate.attempt_id.encode("utf-8")
        ).hexdigest()[:24],
        "attempt_authority_root_digest": "sha256:" + "1" * 64,
        "attempt_root_digest": "sha256:" + "2" * 64,
        "binding_id": "sha256:" + "3" * 64,
        "binding_admission_id": content_identity(
            {"deferred-provider-binding": candidate.attempt_id}
        ),
        "binding_admission_digest": "sha256:" + "4" * 64,
        "projection_immutable_digest": "sha256:" + "5" * 64,
        "nested_task_cid": content_identity(
            {"deferred-provider-nested-task": candidate.task_cid}
        ),
        "nested_attempt": 1,
        "event_stream_id": "event-log:sha256:" + "6" * 64,
        "event_snapshot_id": "event-log-snapshot:sha256:" + "7" * 64,
        "event_manifest_digest": "sha256:" + "8" * 64,
        "event_count": 4,
        "event_head_sequence": 4,
        "event_head_id": event_head_id,
        "task_selected_event_id": "sha256:" + "a" * 64,
        "retry_deferred_event_id": "sha256:" + "b" * 64,
        "daemon_pass_event_id": event_head_id,
        "diagnostic_event_count": 1,
        "diagnostic_event_ids_digest": "sha256:" + "c" * 64,
        "deferred_reason": DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_REASON,
        "deferred_backoff_seconds": (
            DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_BACKOFF_SECONDS
        ),
        "diagnostic_receipt_id": "",
        "state_digest": "sha256:" + "d" * 64,
        "outer_block_receipt_digest": (
            daemon._database_no_provider_rearm_digest(outer_receipt)
        ),
        "provider_dispatched": False,
        "attempt_consumed": False,
        "validation_attempted": False,
        "commit_created": False,
        "merge_attempted": False,
        "acceptance_inferred": False,
        "route_deferred": True,
        "nested_state_quiescent": True,
    }
    evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return evidence


def _terminal_no_effect_route_rearm_evidence(
    candidate: DatabaseTaskAttempt,
    outer_receipt: dict[str, object],
) -> dict[str, object]:
    """Build one exact outer admission candidate for the versioned bridge."""

    def sha(value: object) -> str:
        return "sha256:" + hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()

    route_plan = {
        "authorization": None,
        "fallback_implementer_identity": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_provider_id": "codex",
        "fallback_reasoning_effort": "medium",
        "fallback_trigger": "primary_quota_exhausted",
        "invocation_binding": None,
        "primary_model_id": "grok-4.6",
        "primary_provider_id": "grok_cli",
        "route_id": (
            "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
        ),
    }
    prelude_event_count = 9 if candidate.task_alias == "PCTDD-034" else 0
    diagnostic_event_count = (
        14 if candidate.task_alias in {"PCTDD-005", "PCTDD-034"} else 13
    )
    event_count = prelude_event_count + diagnostic_event_count + 8
    event_head_id = sha("terminal-no-effect-daemon-pass")
    evidence: dict[str, object] = {
        "schema": DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA,
        "attempt_id": candidate.attempt_id,
        "claim_id": candidate.claim_id,
        "task_cid": candidate.task_cid,
        "task_alias": candidate.task_alias,
        "attempt_number": candidate.attempt_number,
        "owner_session_id": candidate.owner_session_id,
        "lease_id": candidate.lease_id,
        "fencing_token": candidate.fencing_token,
        "fence_epoch": candidate.fence_epoch,
        "attempt_root_key": hashlib.sha256(
            candidate.attempt_id.encode("utf-8")
        ).hexdigest()[:24],
        "attempt_authority_root_digest": sha("attempt-authority-root"),
        "attempt_root_digest": sha("attempt-root"),
        "binding_id": sha("binding"),
        "binding_admission_id": content_identity(
            {"terminal-no-effect-binding": candidate.attempt_id}
        ),
        "binding_admission_digest": sha("binding-admission"),
        "projection_immutable_digest": sha("projection"),
        "task_revision": 7,
        "board_namespace": "parallel-content-sealing-proof-carrying-tdd-v1",
        "nested_task_cid": content_identity(
            {"terminal-no-effect-task": candidate.task_cid}
        ),
        "nested_attempt": 1,
        "event_stream_id": "event-log:" + sha("event-stream"),
        "event_snapshot_id": "event-log-snapshot:" + sha("event-snapshot"),
        "event_manifest_digest": sha("event-manifest"),
        "event_count": event_count,
        "event_head_sequence": event_count,
        "event_head_id": event_head_id,
        "prelude_event_count": prelude_event_count,
        "prelude_event_ids_digest": sha("prelude-events"),
        "task_selected_event_id": sha("task-selected"),
        "diagnostic_event_count": diagnostic_event_count,
        "diagnostic_event_ids_digest": sha("diagnostic-events"),
        "protected_snapshot_recorded_event_id": sha("snapshot-recorded"),
        "implementation_started_event_id": sha("implementation-started"),
        "pre_implementation_event_id": sha("pre-implementation"),
        "pre_implementation_receipt_cid": content_identity(
            {"pre-implementation": candidate.attempt_id}
        ),
        "protected_snapshot_cleared_event_id": sha("snapshot-cleared"),
        "worktree_release_event_id": sha("worktree-release"),
        "implementation_finished_event_id": sha("implementation-finished"),
        "daemon_pass_event_id": event_head_id,
        "state_digest": sha("state"),
        "outer_block_receipt_digest": (
            DatabaseImplementationDaemon._database_no_provider_rearm_digest(
                outer_receipt
            )
        ),
        "command_sha256": sha("command"),
        "route_plan_sha256": "sha256:" + hashlib.sha256(
            canonical_json(route_plan).encode("utf-8")
        ).hexdigest(),
        "route_id": (
            "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
        ),
        "primary_provider": "grok_cli",
        "primary_model": "grok-4.6",
        "fallback_provider": "codex",
        "fallback_model": "gpt-5.6-terra",
        "fallback_reasoning_effort": "medium",
        "log_relative_path": (
            f"implementation-logs/{candidate.task_alias.lower()}-attempt-1.log"
        ),
        "log_sha256": sha("implementation-log"),
        "log_size": 4096,
        "log_identity_digest": sha("implementation-log-identity"),
        "quota_probe_receipt_id": sha("quota-probe-receipt"),
        "quota_probe_receipt_digest": sha("quota-probe-receipt-bytes"),
        "route_outcome_id": sha("route-outcome"),
        "route_outcome_digest": sha("route-outcome-bytes"),
        "failure_class": "hard_quota_exhausted",
        "verifier_status": "not_run",
        "runner_returncode": 1,
        "provider_dispatched": False,
        "wrapper_process_dispatched": True,
        "quota_probe_dispatched": True,
        "primary_model_dispatched": False,
        "fallback_model_dispatched": False,
        "implementation_dispatched": False,
        "provider_effect_committed": False,
        "implementation_effect_committed": False,
        "legacy_nested_attempt_consumed": True,
        "rearm_attempt_consumed": False,
        "attempt_consumed": False,
        "validation_attempted": False,
        "commit_created": False,
        "merge_attempted": False,
        "acceptance_inferred": False,
        "protected_snapshot_unchanged": True,
        "workspace_unchanged": True,
        "cleanup_terminal": True,
        "route_denied": True,
        "historical_receipt_only": True,
        "fresh_fallback_authority": False,
        "nested_state_quiescent": True,
    }
    evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return evidence


def _rehash_terminal_no_effect_route_evidence(
    evidence: dict[str, object],
) -> None:
    unsigned = dict(evidence)
    unsigned.pop("evidence_id", None)
    evidence["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _fenced_provider_unpublished_evidence(
    candidate: DatabaseTaskAttempt,
    original: Mapping[str, object],
) -> dict[str, object]:
    sha = "sha256:" + "a" * 64
    cid = "baguqeera" + "a" * 52
    terminal_link = dict(original.get("terminal_reconciliation") or {})
    values: dict[str, object] = {
        "schema": DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_SCHEMA,
        "attempt_id": candidate.attempt_id,
        "claim_id": candidate.claim_id,
        "task_cid": candidate.task_cid,
        "task_alias": candidate.task_alias,
        "attempt_number": candidate.attempt_number,
        "owner_session_id": candidate.owner_session_id,
        "lease_id": candidate.lease_id,
        "fencing_token": candidate.fencing_token,
        "fence_epoch": candidate.fence_epoch,
        "attempt_root_key": hashlib.sha256(
            candidate.attempt_id.encode("utf-8")
        ).hexdigest()[:24],
        "attempt_authority_root_digest": sha,
        "attempt_root_digest": sha,
        "attempt_directory_names_digest": sha,
        "binding_id": sha,
        "binding_admission_id": cid,
        "binding_admission_digest": sha,
        "projection_immutable_digest": sha,
        "task_revision": 1,
        "board_namespace": "parallel-content-sealing-proof-carrying-tdd-v1",
        "nested_task_cid": cid,
        "nested_attempt": 1,
        "terminal_reconciliation_evidence_id": str(
            terminal_link.get("evidence_id") or cid
        ),
        "prepared_reconciliation_receipt_id": str(
            terminal_link.get("prepared_reconciliation_receipt_id") or sha
        ),
        "commit_barrier_receipt_id": str(
            terminal_link.get("commit_barrier_receipt_id") or sha
        ),
        "terminal_reconciliation_receipt_id": sha,
        "reconciliation_receipt_count": 5,
        "reconciliation_receipt_ids_digest": sha,
        "fenced_provider_receipt_count": 2,
        "fenced_provider_receipt_ids_digest": sha,
        "provider_fence_chronology_digest": sha,
        "container_removed_receipt_id": sha,
        "terminal_lifecycle_receipt_id": sha,
        "provider_runner_pid": 4242,
        "provider_runner_receipt_id": cid,
        "provider_container_fence_receipt_id": cid,
        "task_claim_release_receipt_id": cid,
        "task_claim_release_receipt_name": (
            "canonical-task-0123456789abcdef01234567-a1.json"
        ),
        "task_claim_release_event_id": sha,
        "implementation_started_event_id": sha,
        "terminal_lifecycle_event_id": sha,
        "event_stream_id": "event-log:sha256:" + "b" * 64,
        "event_snapshot_id": "event-log-snapshot:sha256:" + "b" * 64,
        "event_manifest_digest": sha,
        "event_count": 42,
        "event_head_sequence": 42,
        "event_head_id": sha,
        "workspace_path": "/tmp/destroyed-pctdd-worker",
        "branch": "implementation/pctdd-006-a1",
        "baseline_ref": "c" * 40,
        "branch_disposition": "absent",
        "branch_target": "",
        "workspace_absent": True,
        "prepared_state_digest": sha,
        "state_digest": sha,
        "outer_block_receipt_digest": (
            DatabaseImplementationDaemon._database_no_provider_rearm_digest(
                original
            )
        ),
        "provider_runner_started": True,
        "implementation_runner_started": True,
        "terminal_provider_outcome_admitted": False,
        "terminal_provider_evidence_present": False,
        "effect_admitted": False,
        "attempt_consumed": True,
        "retry_authorized_once": True,
        "authoritative_validation_admitted": False,
        "commit_admitted": False,
        "merge_admitted": False,
        "publication_admitted": False,
        "candidate_ref_delta": False,
        "acceptance_inferred": False,
        "recovery_terminal": True,
        "candidate_disposition": "destroyed_unaccepted",
        "nested_state_quiescent": True,
    }
    occurrence = {
        name: (1 if name == "credit_ordinal" else values.get(name))
        for name in (
            database_portal_bridge_module.
            _FENCED_PROVIDER_UNPUBLISHED_OCCURRENCE_FIELDS
        )
    }
    manifest = {
        "schema": (
            database_portal_bridge_module.
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_SCHEMA
        ),
        "revision": "pctdd-provider-recovery-2026-09-02",
        "operator_owned": True,
        "one_shot": True,
        "occurrences": [occurrence],
    }
    manifest_id = "sha256:" + hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    migration_credit = {
        "schema": (
            database_portal_bridge_module.
            DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_CREDIT_SCHEMA
        ),
        "manifest_id": manifest_id,
        "occurrence": occurrence,
    }
    values.update(
        {
            "migration_manifest_id": manifest_id,
            "migration_credit_id": "sha256:" + hashlib.sha256(
                json.dumps(
                    migration_credit,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "migration_credit_ordinal": 1,
        }
    )
    authorization_fields = (
        "binding_id",
        "binding_admission_id",
        "binding_admission_digest",
        "projection_immutable_digest",
        "task_revision",
        "board_namespace",
        "nested_task_cid",
        "nested_attempt",
        "terminal_reconciliation_evidence_id",
        "prepared_reconciliation_receipt_id",
        "commit_barrier_receipt_id",
        "terminal_reconciliation_receipt_id",
        "reconciliation_receipt_count",
        "reconciliation_receipt_ids_digest",
        "fenced_provider_receipt_count",
        "fenced_provider_receipt_ids_digest",
        "provider_fence_chronology_digest",
        "container_removed_receipt_id",
        "terminal_lifecycle_receipt_id",
        "provider_runner_pid",
        "provider_runner_receipt_id",
        "provider_container_fence_receipt_id",
        "task_claim_release_receipt_id",
        "task_claim_release_receipt_name",
        "task_claim_release_event_id",
        "implementation_started_event_id",
        "terminal_lifecycle_event_id",
        "event_stream_id",
        "event_snapshot_id",
        "event_manifest_digest",
        "event_count",
        "event_head_sequence",
        "event_head_id",
        "workspace_path",
        "branch",
        "baseline_ref",
        "branch_disposition",
        "branch_target",
        "workspace_absent",
        "prepared_state_digest",
        "state_digest",
        "outer_block_receipt_digest",
        "migration_manifest_id",
        "migration_credit_id",
        "migration_credit_ordinal",
    )
    authorization = {
        "schema": DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_AUTHORIZATION_SCHEMA,
        **{
            name: values[name]
            for name in (
                "attempt_id",
                "claim_id",
                "task_cid",
                "attempt_number",
                "owner_session_id",
                "lease_id",
                "fencing_token",
                "fence_epoch",
            )
        },
        **{name: values[name] for name in authorization_fields},
    }
    values["rearm_authorization_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            authorization,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    assert set(values) == set(
        DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_FIELDS
    ) - {"evidence_id"}
    values["evidence_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return values


def test_normal_fenced_provider_promotion_response_loss_recovers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final callback response loss preserves the irrevocable admitted fence."""

    daemon = _open_daemon(
        tmp_path,
        session="session:fenced-promotion-response-loss",
        max_task_attempts=1,
    )
    population = _population(1)
    population["tasks"][0]["task_id"] = "DQP-T006"
    daemon.materialize_population(population)
    attempt = daemon.claim_next()
    assert attempt is not None
    attempt = daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
    provider_key = f"provider:{attempt.attempt_id}"
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=provider_key,
    )
    daemon._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=provider_key,
        outcome="returned",
        body={"status": "unpublished"},
    )
    failed, blocked_receipt = daemon._finalize_failed_attempt(
        attempt,
        reason="callback_authority_incomplete_blocked",
        force_block=True,
        unknown_authority=True,
        reconciliation_evidence={"evidence_id": "sha256:" + "4" * 64},
    )
    evidence = _fenced_provider_unpublished_evidence(
        failed,
        blocked_receipt,
    )
    final_response_losses: list[str] = []

    class ExactFencedProviderBridge:
        def fenced_provider_unpublished_migration_available(
            self,
            selected: object,
            *,
            outer_block_receipt: object,
        ) -> bool:
            return (
                getattr(selected, "attempt_id", "") == failed.attempt_id
                and outer_block_receipt == blocked_receipt
            )

        def revalidate_fenced_provider_unpublished_rearm_evidence(
            self,
            selected: object,
            *,
            outer_block_receipt: object,
            expected_evidence: object,
        ) -> bool:
            return (
                getattr(selected, "attempt_id", "") == failed.attempt_id
                and outer_block_receipt == blocked_receipt
                and expected_evidence == evidence
            )

        def execute_with_revalidated_fenced_provider_unpublished(
            self,
            selected: object,
            *,
            outer_block_receipt: object,
            expected_evidence: object,
            callback: Callable[[], object],
        ) -> object:
            assert self.revalidate_fenced_provider_unpublished_rearm_evidence(
                selected,
                outer_block_receipt=outer_block_receipt,
                expected_evidence=expected_evidence,
            )
            result = callback()
            if (
                isinstance(result, Mapping)
                and result.get("state") == "admitted"
            ):
                final_response_losses.append("lost-after-admitted")
                raise DatabasePortalBridgeError(
                    "injected final promotion response loss"
                )
            return result

    daemon._database_portal_bridge = ExactFencedProviderBridge()
    monkeypatch.setattr(
        daemon,
        "_database_portal_no_provider_rearm_evidence",
        lambda task, receipt: (
            dict(evidence)
            if task.task_cid == failed.task_cid and receipt == blocked_receipt
            else None
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_fenced_provider_predecessor_is_dead",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        DatabaseImplementationDaemon,
        "_valid_no_provider_rearm_evidence",
        staticmethod(lambda *_args, **_kwargs: True),
    )
    try:
        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(outcomes) == 1
        assert outcomes[0]["rearmed"] is True
        assert final_response_losses == ["lost-after-admitted"]
        rearmed = daemon.task_source.get(failed.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        assert rearmed.body["completion_receipt"][
            "no_provider_rearm_fence"
        ]["state"] == "admitted"
        assert daemon._fenced_provider_recovery_dispatch_fence(failed)[
            "state"
        ] == "admitted"

        successor = daemon.claim_next()
        assert successor is not None
        assert successor.task_cid == failed.task_cid
        assert successor.attempt_id != failed.attempt_id
        assert daemon.claim_next() is None
        assert daemon._fenced_provider_recovery_dispatch_fence(failed)[
            "state"
        ] == "admitted"
    finally:
        daemon.close()


def test_fenced_provider_unpublished_evidence_validator_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = DatabaseTaskAttempt(
        attempt_id="attempt:fenced-provider",
        claim_id="claim:fenced-provider",
        task_cid="task:cid:fenced-provider",
        task_alias="PCTDD-006",
        attempt_number=3,
        owner_session_id="session:fenced-provider",
        lease_id="lease:fenced-provider",
        fencing_token=17,
        fence_epoch=9,
        committed_phase="failed",
        status="failed",
        started_at_ms=1,
        finished_at_ms=2,
        body={},
    )
    terminal_link = {
        "evidence_id": "baguqeera" + "a" * 52,
        "prepared_reconciliation_receipt_id": "sha256:" + "a" * 64,
        "commit_barrier_receipt_id": "sha256:" + "a" * 64,
    }
    original: dict[str, object] = {
        "attempt_id": candidate.attempt_id,
        "claim_id": candidate.claim_id,
        "task_cid": candidate.task_cid,
        "attempt_number": candidate.attempt_number,
        "owner_session_id": candidate.owner_session_id,
        "lease_id": candidate.lease_id,
        "fencing_token": candidate.fencing_token,
        "fence_epoch": candidate.fence_epoch,
        "attempts_used": 1,
        "terminal_reconciliation": terminal_link,
    }
    task = SimpleNamespace(
        task_cid=candidate.task_cid,
        task_alias=candidate.task_alias,
    )
    evidence = _fenced_provider_unpublished_evidence(candidate, original)
    occurrence = {
        name: (
            evidence["migration_credit_ordinal"]
            if name == "credit_ordinal"
            else evidence[name]
        )
        for name in (
            database_portal_bridge_module.
            _FENCED_PROVIDER_UNPUBLISHED_OCCURRENCE_FIELDS
        )
    }
    monkeypatch.setattr(
        database_portal_bridge_module,
        "DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_PINS",
        (occurrence,),
    )
    monkeypatch.setattr(
        database_portal_bridge_module,
        "DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_MIGRATION_MANIFEST_ID",
        evidence["migration_manifest_id"],
    )
    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=task,
        original=original,
        expected_evidence_id=str(evidence["evidence_id"]),
    )

    for field, value in (
        ("effect_admitted", True),
        ("candidate_ref_delta", True),
        ("workspace_absent", False),
        ("branch_disposition", "baseline"),
        ("attempt_consumed", False),
        ("provider_runner_started", False),
        ("terminal_provider_evidence_present", True),
        ("claim_id", "claim:spliced-attempt"),
        ("provider_fence_chronology_digest", "sha256:" + "d" * 64),
        ("unexpected_publication_authority", True),
    ):
        malformed = dict(evidence)
        malformed[field] = value
        unsigned = dict(malformed)
        unsigned.pop("evidence_id", None)
        malformed["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert not DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
            malformed,
            task=task,
            original=original,
            expected_evidence_id=str(malformed["evidence_id"]),
        ), field


def test_fenced_provider_rearm_preserves_consumed_attempt_and_grants_one_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = SimpleNamespace(
        task_cid="task:cid:fenced-consumed",
        task_alias="DQP-T006",
        validations=(),
        body={},
        revision=13,
        status="retrying",
    )
    validation_spec_cid = (
        DatabaseImplementationDaemon._retry_budget_validation_spec_cid(task)
    )
    terminal_evidence_id = "baguqeera" + "a" * 52
    original: dict[str, object] = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": "database_unknown_outcome_blocked",
        "task_cid": task.task_cid,
        "validation_spec_cid": validation_spec_cid,
        "attempts_used": 1,
        "max_task_attempts": 1,
        "retry_exhausted": True,
        "process_instance_id": "process:old",
        "owner_session_id": "session:old",
        "reason": "callback_authority_incomplete_blocked",
        "unknown_outcome_rearm_count": 0,
        "forced_block": True,
        "authority_outcome": "unknown",
        "attempt_id": "attempt:fenced-consumed",
        "claim_id": "claim:fenced-consumed",
        "lease_id": "lease:fenced-consumed",
        "attempt_number": 1,
        "fencing_token": 7,
        "fence_epoch": 3,
        "terminal_reconciliation": {"evidence_id": terminal_evidence_id},
    }
    evidence_id = "sha256:" + "e" * 64
    evidence = {
        "schema": DATABASE_PORTAL_FENCED_PROVIDER_UNPUBLISHED_REARM_EVIDENCE_SCHEMA,
        "terminal_reconciliation_evidence_id": terminal_evidence_id,
        "attempt_consumed": True,
        "retry_authorized_once": True,
        "evidence_id": evidence_id,
    }
    blocked_revision = 10
    retrying_revision = 11
    saga_nonce = "no-provider-rearm:" + "1" * 24
    blocked_digest = DatabaseImplementationDaemon._database_no_provider_rearm_digest(
        original
    )
    saga_id = DatabaseImplementationDaemon._database_no_provider_rearm_saga_id(
        saga_nonce=saga_nonce,
        task_cid=task.task_cid,
        attempt_id=str(original["attempt_id"]),
        claim_id=str(original["claim_id"]),
        evidence_id=evidence_id,
        blocked_receipt_digest=blocked_digest,
        blocked_revision=blocked_revision,
    )
    receipt: dict[str, object] = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
        "task_cid": task.task_cid,
        "validation_spec_cid": validation_spec_cid,
        "attempts_used": 1,
        "max_task_attempts": 1,
        "retry_exhausted": False,
        "process_instance_id": "process:new",
        "owner_session_id": "session:new",
        "reason": original["reason"],
        "unknown_outcome_rearm_count": 1,
        "forced_block": False,
        "authority_outcome": "rearmed",
        "previous_operation": original["operation"],
        "previous_owner_session_id": original["owner_session_id"],
        "previous_attempt_id": original["attempt_id"],
        "previous_claim_id": original["claim_id"],
        "no_provider_rearm_evidence": evidence,
        "no_provider_rearm_evidence_id": evidence_id,
        "previous_block_process_instance_id": original["process_instance_id"],
        "no_provider_rearm_saga_id": saga_id,
        "no_provider_rearm_original_block_receipt": original,
    }
    dispatch_rows = [
        {
            "dispatch_id": "dispatch:fenced-consumed",
            "attempt_id": original["attempt_id"],
            "task_cid": task.task_cid,
            "dispatch_kind": "provider",
            "idempotency_key": f"provider:{original['attempt_id']}",
            "owner_session_id": original["owner_session_id"],
            "fencing_token": original["fencing_token"],
            "fence_epoch": original["fence_epoch"],
            "started_at_ms": 100,
            "updated_at_ms": 100,
            "outcome": "started",
            "body": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "database-callback-dispatch@1"
                ),
                "outcome": "unknown_until_callback_returns",
            },
        }
    ]
    callback_population: dict[str, object] = {
        "schema": (
            daemon_module.DATABASE_FENCED_PROVIDER_CALLBACK_POPULATION_SCHEMA
        ),
        "attempt_id": original["attempt_id"],
        "dispatch_journal": {
            "count": 1,
            "rows": dispatch_rows,
            "digest": DatabaseImplementationDaemon._database_canonical_digest(
                dispatch_rows
            ),
        },
        "provider_invocations": {
            "count": 0,
            "rows": [],
            "digest": daemon_module.DATABASE_EMPTY_CANONICAL_LIST_DIGEST,
        },
        "effect_claims": {
            "count": 0,
            "rows": [],
            "digest": daemon_module.DATABASE_EMPTY_CANONICAL_LIST_DIGEST,
        },
    }
    callback_population["population_id"] = (
        DatabaseImplementationDaemon._database_no_provider_rearm_digest(
            callback_population
        )
    )
    outer_snapshot: dict[str, object] = {
        "schema": DATABASE_FENCED_PROVIDER_OUTER_SNAPSHOT_SCHEMA,
        "attempt_record": {
            "attempt_id": original["attempt_id"],
            "task_cid": task.task_cid,
            "owner_session_id": original["owner_session_id"],
            "fencing_token": original["fencing_token"],
            "fence_epoch": original["fence_epoch"],
            "status": "failed",
            "committed_phase": "failed",
        },
        "provider_dispatch": {
            "outcome": "started",
            "body": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "database-callback-dispatch@1"
                ),
                "outcome": "unknown_until_callback_returns",
            },
            "updated_at_ms": 100,
        },
        "callback_population": callback_population,
        "phase_history": [{"phase": "failed", "status": "failed"}],
        "provider_invocation_absent": True,
        "effect_claim_absent": True,
        "effect_dispatch_absent": True,
        "recovery_dispatch_fence_id": "sha256:" + "f" * 64,
    }
    outer_snapshot["snapshot_id"] = (
        DatabaseImplementationDaemon._database_no_provider_rearm_digest(
            outer_snapshot
        )
    )
    receipt["fenced_provider_outer_attempt_snapshot"] = outer_snapshot
    immutable_digest = (
        DatabaseImplementationDaemon._database_no_provider_rearm_digest(receipt)
    )
    receipt["no_provider_rearm_fence"] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-no-provider-rearm-fence@1"
        ),
        "saga_id": saga_id,
        "saga_nonce": saga_nonce,
        "state": "admitted",
        "evidence_id": evidence_id,
        "blocked_receipt_digest": blocked_digest,
        "blocked_revision": blocked_revision,
        "retrying_revision": retrying_revision,
        "admitted_revision": retrying_revision + 2,
        "immutable_receipt_digest": immutable_digest,
    }
    task.body = {"completion_receipt": receipt}
    monkeypatch.setattr(
        DatabaseImplementationDaemon,
        "_valid_no_provider_rearm_evidence",
        staticmethod(lambda *_args, **_kwargs: True),
    )
    assert DatabaseImplementationDaemon._no_provider_rearm_fence_state(task) == (
        "admitted"
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.max_task_attempts = 1
    fenced_attempt = SimpleNamespace(
        attempt_id=original["attempt_id"],
        task_cid=task.task_cid,
    )
    daemon.get_attempt = lambda attempt_id: (
        fenced_attempt
        if attempt_id == fenced_attempt.attempt_id
        else None
    )
    daemon._fenced_provider_recovery_dispatch_fence = (
        lambda attempt: {"state": "admitted"}
        if attempt is fenced_attempt
        else None
    )
    daemon._fenced_provider_outer_state_matches = (
        lambda attempt, snapshot: (
            attempt is fenced_attempt and snapshot == outer_snapshot
        )
    )
    retry_state = daemon._retry_budget_state(task)
    assert retry_state["attempts_used"] == 1
    assert retry_state["retry_exhausted"] is False

    refunded = dict(receipt)
    refunded["attempts_used"] = 0
    task.body = {"completion_receipt": refunded}
    assert DatabaseImplementationDaemon._no_provider_rearm_fence_state(task) == (
        "invalid"
    )


def test_terminal_no_effect_route_evidence_validator_is_closed_and_fenced() -> None:
    candidate = DatabaseTaskAttempt(
        attempt_id="attempt:terminal-no-effect",
        claim_id="claim:terminal-no-effect",
        task_cid="task:cid:terminal-no-effect",
        task_alias="PCTDD-034",
        attempt_number=6,
        owner_session_id="session:terminal-no-effect",
        lease_id="lease:terminal-no-effect",
        fencing_token=17,
        fence_epoch=9,
        committed_phase="failed",
        status="failed",
        started_at_ms=1,
        finished_at_ms=2,
        body={},
    )
    original: dict[str, object] = {
        "attempt_id": candidate.attempt_id,
        "claim_id": candidate.claim_id,
        "task_cid": candidate.task_cid,
        "attempt_number": candidate.attempt_number,
        "owner_session_id": candidate.owner_session_id,
        "lease_id": candidate.lease_id,
        "fencing_token": candidate.fencing_token,
        "fence_epoch": candidate.fence_epoch,
        "attempts_used": 1,
    }
    task = SimpleNamespace(
        task_cid=candidate.task_cid,
        task_alias=candidate.task_alias,
    )
    evidence = _terminal_no_effect_route_rearm_evidence(candidate, original)

    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=task,
        original=original,
        expected_evidence_id=str(evidence["evidence_id"]),
    )

    near_misses: tuple[tuple[str, object], ...] = (
        ("wrapper_process_dispatched", False),
        ("quota_probe_dispatched", False),
        ("primary_model_dispatched", True),
        ("fallback_model_dispatched", True),
        ("legacy_nested_attempt_consumed", False),
        ("rearm_attempt_consumed", True),
        ("attempt_consumed", True),
        ("fresh_fallback_authority", True),
        ("route_denied", False),
        ("historical_receipt_only", False),
        ("claim_id", "claim:other"),
        ("fencing_token", candidate.fencing_token + 1),
        ("board_namespace", "unreviewed-board"),
        ("nested_attempt", 2),
        ("prelude_event_count", 8),
        ("diagnostic_event_count", 13),
        ("event_count", 30),
        ("route_plan_sha256", "sha256:" + "0" * 64),
        ("route_id", "route:unreviewed"),
        ("log_relative_path", "../escaped.log"),
        ("log_relative_path", "implementation-logs/unreviewed.log"),
        ("log_relative_path", "implementation-logs/pctdd-034-attempt-2.log"),
        ("runner_returncode", 2),
        ("runner_returncode", True),
        ("unexpected_authority", True),
    )
    for field, value in near_misses:
        malformed = dict(evidence)
        malformed[field] = value
        _rehash_terminal_no_effect_route_evidence(malformed)
        assert not DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
            malformed,
            task=task,
            original=original,
            expected_evidence_id=str(malformed["evidence_id"]),
        ), field


def _terminal_no_effect_historical_selector_case(
    *,
    task_alias: str = "PCTDD-034",
    attempt_number: int = 6,
    attempts_used: int = 1,
    rearm_count: int | None = None,
) -> SimpleNamespace:
    case = _historical_stale_dispatch_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
        attempts_used=attempts_used,
    )
    case.receipt["reason"] = "callback_authority_incomplete_blocked"
    case.receipt.pop("terminal_reconciliation")
    if rearm_count is not None:
        case.receipt["unknown_outcome_rearm_count"] = rearm_count
    case.phase_body.clear()
    case.phase_body.update(
        {
            "database_disposition": "blocked_unknown_outcome",
            "reason": "callback_authority_incomplete_blocked",
            "retry_exhausted": True,
            "unknown_authority": True,
        }
    )
    evidence = _terminal_no_effect_route_rearm_evidence(
        case.attempt,
        case.receipt,
    )
    case.evidence.clear()
    case.evidence.update(evidence)
    case.daemon._database_portal_terminal_reconciliation_saga = (
        lambda _attempt: None
    )

    def journal(
        _attempt: object,
        *,
        dispatch_kind: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        case.calls["journal"].append((dispatch_kind, idempotency_key))
        if dispatch_kind == "effect":
            return None
        return {
            "outcome": "raised",
            "body": {"exception_type": "DatabasePortalBridgeError"},
            "updated_at_ms": 1,
        }

    case.daemon._dispatch_journal_entry = journal
    return case


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "rearm_count"),
    (
        ("PCTDD-005", 2, None),
        ("PCTDD-006", 3, 1),
        ("PCTDD-007", 3, 1),
        ("PCTDD-034", 6, None),
    ),
)
def test_terminal_no_effect_route_selector_admits_historical_attempt_suffix(
    task_alias: str,
    attempt_number: int,
    rearm_count: int | None,
) -> None:
    """Only the exact versioned legacy route budget is admitted."""

    case = _terminal_no_effect_historical_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
        rearm_count=rearm_count,
    )

    admitted = case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    )

    assert admitted is not None
    assert admitted["schema"] == (
        DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA
    )
    assert admitted["attempt_number"] == attempt_number
    assert admitted["legacy_nested_attempt_consumed"] is True
    assert admitted["rearm_attempt_consumed"] is False
    assert admitted["fresh_fallback_authority"] is False
    assert case.calls["verifier"] == [case.attempt]


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "attempts_used", "rearm_count"),
    (
        ("PCTDD-005", 3, 1, 0),
        ("PCTDD-005", 6, 1, 0),
        ("PCTDD-006", 4, 1, 1),
        ("PCTDD-007", 4, 1, 1),
        ("PCTDD-034", 8, 1, 0),
    ),
)
def test_quiesced_stale_release_budget_admits_only_current_outer_histories(
    task_alias: str,
    attempt_number: int,
    attempts_used: int,
    rearm_count: int,
) -> None:
    assert _database_portal_quiesced_stale_dispatch_release_budget_matches(
        task_alias=task_alias,
        attempt_number=attempt_number,
        attempts_used=attempts_used,
        rearm_count=rearm_count,
    )


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "attempts_used", "rearm_count"),
    (
        ("PCTDD-005", 2, 1, 0),
        ("PCTDD-006", 3, 1, 1),
        ("PCTDD-007", 3, 1, 1),
        ("PCTDD-034", 6, 1, 0),
        ("PCTDD-005", 3, 2, 0),
        ("PCTDD-006", 4, 1, 0),
        ("PCTDD-007", 4, 1, 2),
        ("PCTDD-034", 8, 1, 1),
    ),
)
def test_quiesced_stale_release_budget_rejects_predecessors_and_near_misses(
    task_alias: str,
    attempt_number: int,
    attempts_used: int,
    rearm_count: int,
) -> None:
    assert not _database_portal_quiesced_stale_dispatch_release_budget_matches(
        task_alias=task_alias,
        attempt_number=attempt_number,
        attempts_used=attempts_used,
        rearm_count=rearm_count,
    )


def _historical_quiesced_release_task_cid(task_alias: str) -> str:
    if task_alias == "PCTDD-005":
        return "task:cid:pctdd-005"
    matches = [
        str(pin["task_cid"])
        for pin in DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS
        if pin["task_alias"] == task_alias
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_extra_gate_predecessor_process_preserves_unknown_outcome_limit(
    tmp_path: Path,
) -> None:
    """A new process cannot refund a consumed retry budget without proof."""

    population = _population(1)
    population["tasks"][0]["task_id"] = "PCTDD-006"
    seed = _open_daemon(
        tmp_path,
        session="session:extra-gate-pred-rearm-limit",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_unknown_outcome_blocked",
            reason="provider_dispatch_outcome_unknown",
        )
        receipt["forced_block"] = True
        receipt["authority_outcome"] = "unknown"
        receipt["unknown_outcome_rearm_count"] = DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        receipt["process_instance_id"] = "process:predecessor-owner"
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
        assert seed._automatic_claim_forbidden_current(
            seed.task_source.get("task:cid:001")
        ) is True
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:extra-gate-pred-rearm-limit",
        max_task_attempts=2,
    )
    try:
        assert successor.process_instance_id != "process:predecessor-owner"
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert not any(item.get("rearmed") is True for item in rearms)
        recovered = successor.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "blocked"
    finally:
        successor.close()


_MISSING_WORKSPACE_REASON = (
    "[Errno 2] No such file or directory: PosixPath('"
    "/home/barberb/lift_coding/.worktrees/pctdd-g9-orphan-recovery/"
    "data/agent_supervisor/parallel_content_sealing_proof_carrying_tdd_v1_g9/"
    "worktrees/workspace_b6c6c987ab22_678bd72ae733')"
)


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_extra_gate_callback_authority_incomplete_without_evidence_stays_blocked(
    tmp_path: Path,
) -> None:
    """An alias and predecessor change cannot resolve an unknown callback."""

    population = _population(1)
    population["tasks"][0]["task_id"] = "PCTDD-006"
    seed = _open_daemon(
        tmp_path,
        session="session:extra-gate-callback-incomplete-rearm",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        receipt["forced_block"] = True
        receipt["authority_outcome"] = "unknown"
        receipt["retry_exhausted"] = True
        receipt["unknown_outcome_rearm_count"] = DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        receipt["process_instance_id"] = "process:predecessor-owner"
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:extra-gate-callback-incomplete-rearm",
        max_task_attempts=2,
    )
    cas_statuses: list[str] = []
    original_cas = successor._cas_task_status_database
    try:
        assert successor.process_instance_id != "process:predecessor-owner"

        def tracking_cas(task_cid: str, **kwargs: object) -> object:
            status = str(kwargs.get("new_status") or "")
            cas_statuses.append(status)
            assert status != "completed"
            return original_cas(task_cid, **kwargs)

        monkeypatch_cas = tracking_cas
        successor._cas_task_status_database = monkeypatch_cas  # type: ignore[method-assign]
        successor._database_portal_no_provider_rearm_evidence = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: None
        )
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert not any(item.get("rearmed") is True for item in rearms)
        recovered = successor.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "blocked"
        assert "completed" not in cas_statuses
        assert cas_statuses == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_extra_gate_stale_terminal_link_does_not_grant_generic_rearm(
    tmp_path: Path,
) -> None:
    """An unverified terminal candidate cannot resolve an unknown callback."""

    population = _population(1)
    population["tasks"][0]["task_id"] = "PCTDD-007"
    seed = _open_daemon(
        tmp_path,
        session="session:extra-gate-stale-landed-rearm",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        receipt["forced_block"] = True
        receipt["authority_outcome"] = "unknown"
        receipt["retry_exhausted"] = True
        receipt["unknown_outcome_rearm_count"] = DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        receipt["process_instance_id"] = "process:predecessor-owner"
        receipt["terminal_reconciliation"] = {"schema": "stale-candidate"}
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:extra-gate-stale-landed-rearm",
        max_task_attempts=2,
    )
    cas_statuses: list[str] = []
    original_cas = successor._cas_task_status_database
    landed_calls: list[str] = []
    try:
        def tracking_cas(task_cid: str, **kwargs: object) -> object:
            status = str(kwargs.get("new_status") or "")
            cas_statuses.append(status)
            assert status != "completed"
            return original_cas(task_cid, **kwargs)

        def landed(*, task: object, bridge: object) -> dict[str, object]:
            landed_calls.append(str(getattr(task, "task_alias", "")))
            raise DatabaseImplementationConflictError(
                "blocked landed recovery receipt is malformed or stale"
            )

        successor._cas_task_status_database = tracking_cas  # type: ignore[method-assign]
        successor._database_portal_no_provider_rearm_evidence = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: None
        )
        successor._reconcile_one_blocked_terminal_landed_task = landed  # type: ignore[method-assign]
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert landed_calls == []
        assert all(
            item.get("operation") != DATABASE_TERMINAL_LANDED_COMPLETION_OPERATION
            for item in rearms
        )
        assert not any(item.get("rearmed") is True for item in rearms)
        recovered = successor.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "blocked"
        assert "completed" not in cas_statuses
    finally:
        successor.close()


@pytest.mark.parametrize("task_alias", ("PCTDD-005", "PCTDD-007", "PCTDD-034"))
@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_extra_gate_missing_workspace_retry_exhausted_stays_blocked(
    tmp_path: Path,
    task_alias: str,
) -> None:
    """A missing path string is not proof of a provider-free predecessor."""

    population = _population(1)
    population["tasks"][0]["task_id"] = task_alias
    seed = _open_daemon(
        tmp_path,
        session=f"session:extra-gate-missing-ws-{task_alias.lower()}",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason=_MISSING_WORKSPACE_REASON,
        )
        receipt["retry_exhausted"] = True
        receipt["unknown_outcome_rearm_count"] = DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
        blocked = seed.task_source.get("task:cid:001")
        assert blocked is not None
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session=f"session:extra-gate-missing-ws-{task_alias.lower()}",
        max_task_attempts=2,
    )
    cas_statuses: list[str] = []
    original_cas = successor._cas_task_status_database
    try:
        def tracking_cas(task_cid: str, **kwargs: object) -> object:
            status = str(kwargs.get("new_status") or "")
            cas_statuses.append(status)
            assert status != "completed"
            return original_cas(task_cid, **kwargs)

        successor._cas_task_status_database = tracking_cas  # type: ignore[method-assign]
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert not any(item.get("rearmed") is True for item in rearms)
        recovered = successor.task_source.get("task:cid:001")
        assert recovered is not None and recovered.status == "blocked"
        assert "completed" not in cas_statuses
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_ordinary_missing_workspace_retry_exhausted_stays_blocked(
    tmp_path: Path,
) -> None:
    """Ordinary tasks cannot use extra-gate missing-workspace rearm."""

    seed = _open_daemon(
        tmp_path,
        session="session:ordinary-missing-ws-rearm",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason=_MISSING_WORKSPACE_REASON,
        )
        receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
        blocked = seed.task_source.get("task:cid:001")
        assert blocked is not None
        assert seed.reconcile_blocked_unknown_outcome_tasks() == []
        held = seed.task_source.get("task:cid:001")
        assert held is not None and held.status == "blocked"
    finally:
        seed.close()


@pytest.mark.parametrize("task_alias", ("PCTDD-006", "PCTDD-007", "PCTDD-034"))
def test_retained_recovery_reserved_alias_cannot_be_rebound(task_alias: str) -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    task = SimpleNamespace(
        task_cid=f"task:forged:{task_alias.lower()}",
        task_alias=task_alias,
        status="blocked",
        revision=1,
        body={},
    )
    assert daemon._retained_recovery_reserved_epoch_state(task) == (
        "invalid_reserved_identity"
    )
    assert daemon._automatic_claim_forbidden_current(task) is True


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "attempts_used", "rearm_count"),
    (
        ("PCTDD-005", 3, 1, 0),
        ("PCTDD-005", 6, 1, 0),
        ("PCTDD-006", 4, 1, 1),
        ("PCTDD-007", 4, 1, 1),
        ("PCTDD-034", 8, 1, 0),
    ),
)
def test_exact_quiesced_stale_release_bypasses_terminal_landed_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_alias: str,
    attempt_number: int,
    attempts_used: int,
    rearm_count: int,
) -> None:
    """The exact release profile reaches only the proof-backed rearm gate."""

    daemon = _open_daemon(
        tmp_path,
        session=f"session:release-selector:{task_alias.lower()}",
        max_task_attempts=2,
    )
    task = SimpleNamespace(
        task_cid=_historical_quiesced_release_task_cid(task_alias),
        task_alias=task_alias,
        status="blocked",
        revision=1,
        body={
            "completion_receipt": {
                "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                "operation": "database_unknown_outcome_blocked",
                "reason": "callback_authority_incomplete_blocked",
                "forced_block": True,
                "authority_outcome": "unknown",
                "retry_exhausted": True,
                "attempt_number": attempt_number,
                "attempts_used": attempts_used,
                "unknown_outcome_rearm_count": rearm_count,
                "terminal_reconciliation": {"schema": "exact-candidate"},
            }
        },
    )
    landed_calls: list[str] = []
    generic_calls: list[str] = []
    try:
        daemon._database_portal_bridge = object()
        monkeypatch.setattr(
            daemon.task_source,
            "list_tasks",
            lambda **_kwargs: SimpleNamespace(tasks=(task,)),
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])

        def landed(*, task: object, bridge: object) -> dict[str, object]:
            landed_calls.append(str(getattr(task, "task_alias", "")))
            raise AssertionError("exact release profile reached landed recovery")

        def generic(task: object, receipt: object) -> None:
            generic_calls.append(str(getattr(task, "task_alias", "")))
            return None

        monkeypatch.setattr(
            daemon,
            "_reconcile_one_blocked_terminal_landed_task",
            landed,
        )
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            generic,
        )
        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lambda task_cid, **kwargs: SimpleNamespace(
                task_cid=task_cid,
                status=str(kwargs.get("new_status") or ""),
                revision=int(getattr(task, "revision", 0) or 0) + 1,
                body={"completion_receipt": dict(kwargs.get("receipt") or {})},
            ),
        )

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert landed_calls == []
        assert generic_calls == [task_alias]
        assert all(
            item.get("operation") != DATABASE_TERMINAL_LANDED_COMPLETION_OPERATION
            for item in outcomes
        )
    finally:
        daemon.close()


def test_callback_authority_incomplete_without_evidence_stays_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown callback effects require exact recovery evidence before retry."""

    seed = _open_daemon(
        tmp_path,
        session="session:callback-authority-generic-rearm-seed",
        max_task_attempts=2,
    )
    try:
        population = _population(1)
        seed.materialize_population(population)
        attempt = seed.claim_next()
        assert attempt is not None
        seed._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        seed._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        _failed, blocked_receipt = seed._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        assert blocked_receipt["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        task_cid = str(attempt.task_cid)
    finally:
        seed.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:callback-authority-generic-rearm-successor",
        max_task_attempts=2,
    )
    cas_statuses: list[str] = []
    original_cas = daemon._cas_task_status_database
    try:
        blocked = daemon.task_source.get(task_cid)
        assert blocked is not None and blocked.status == "blocked"
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            lambda *_args, **_kwargs: None,
        )

        def tracking_cas(
            task_cid: str,
            **kwargs: object,
        ) -> object:
            status = str(kwargs.get("new_status") or "")
            cas_statuses.append(status)
            assert status != "completed"
            return original_cas(task_cid, **kwargs)

        monkeypatch.setattr(daemon, "_cas_task_status_database", tracking_cas)

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert outcomes == []
        held = daemon.task_source.get(task_cid)
        assert held is not None and held.status == "blocked"
        assert cas_statuses == []
    finally:
        daemon.close()


def test_consumed_reserved_callback_authority_incomplete_stays_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown callback effects require exact recovery evidence before retry."""

    seed = _open_daemon(
        tmp_path,
        session="session:consumed-callback-authority-generic-rearm-seed",
        max_task_attempts=2,
    )
    try:
        population = _population(1)
        seed.materialize_population(population)
        attempt = seed.claim_next()
        assert attempt is not None
        seed._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        seed._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        seed._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        task_cid = str(attempt.task_cid)
    finally:
        seed.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:consumed-callback-authority-generic-rearm-successor",
        max_task_attempts=2,
    )
    cas_statuses: list[str] = []
    original_cas = daemon._cas_task_status_database
    try:
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            daemon,
            "_automatic_claim_forbidden_current",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            daemon,
            "_retained_recovery_reserved_epoch_state",
            lambda *_args, **_kwargs: "consumed",
        )

        def tracking_cas(
            task_cid: str,
            **kwargs: object,
        ) -> object:
            status = str(kwargs.get("new_status") or "")
            cas_statuses.append(status)
            assert status != "completed"
            return original_cas(task_cid, **kwargs)

        monkeypatch.setattr(daemon, "_cas_task_status_database", tracking_cas)

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert outcomes == []
        held = daemon.task_source.get(task_cid)
        assert held is not None and held.status == "blocked"
        assert cas_statuses == []
    finally:
        daemon.close()


def test_generic_rearm_receipt_cannot_hide_malformed_retained_records(
    tmp_path: Path,
) -> None:
    """A generic retry marker cannot make malformed retained evidence valid."""

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-stale-retained-not-exhausted",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.list_tasks(limit=8).tasks[0]
        receipt = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
            "task_cid": str(task.task_cid),
            "validation_spec_cid": daemon._retry_budget_validation_spec_cid(
                task
            ),
            "attempts_used": 0,
            "max_task_attempts": 2,
            "retry_exhausted": False,
            "forced_block": False,
            "retained_recovery_admission": {"schema": "stale-one-shot"},
            "retained_recovery_consumption": {"schema": "stale-one-shot"},
        }
        daemon._cas_task_status_database(
            str(task.task_cid),
            expected_revision=int(task.revision),
            new_status="retrying",
            receipt=receipt,
        )
        updated = daemon.task_source.get(task.task_cid)
        assert updated is not None
        state = daemon._retry_budget_state(updated)
        assert state["retry_exhausted"] is True
        assert state["malformed"] is True
        assert daemon._automatic_claim_forbidden_current(updated) is True
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "attempts_used", "rearm_count"),
    (
        ("PCTDD-005", 2, 1, 0),
        ("PCTDD-006", 4, 1, 0),
        ("PCTDD-007", 4, 1, 2),
        ("PCTDD-034", 8, 2, 0),
    ),
)
def test_near_quiesced_release_remains_in_terminal_landed_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_alias: str,
    attempt_number: int,
    attempts_used: int,
    rearm_count: int,
) -> None:
    """A neighboring budget tuple cannot enter the new proof route."""

    daemon = _open_daemon(
        tmp_path,
        session=f"session:release-near-miss:{task_alias.lower()}",
        max_task_attempts=2,
    )
    task = SimpleNamespace(
        task_cid=_historical_quiesced_release_task_cid(task_alias),
        task_alias=task_alias,
        status="blocked",
        revision=1,
        body={
            "completion_receipt": {
                "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                "operation": "database_unknown_outcome_blocked",
                "reason": "callback_authority_incomplete_blocked",
                "forced_block": True,
                "authority_outcome": "unknown",
                "retry_exhausted": True,
                "attempt_number": attempt_number,
                "attempts_used": attempts_used,
                "unknown_outcome_rearm_count": rearm_count,
                "terminal_reconciliation": {"schema": "near-miss"},
            }
        },
    )
    generic_calls: list[str] = []
    cas_calls: list[str] = []
    try:
        daemon._database_portal_bridge = object()
        monkeypatch.setattr(
            daemon.task_source,
            "list_tasks",
            lambda **_kwargs: SimpleNamespace(tasks=(task,)),
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(
            daemon,
            "_reconcile_one_blocked_terminal_landed_task",
            lambda *, task, bridge: {
                "task_cid": str(task.task_cid),
                "task_alias": str(task.task_alias),
                "operation": DATABASE_TERMINAL_LANDED_COMPLETION_OPERATION,
                "recovered": False,
                "rearmed": False,
                "blocked": True,
                "reason": "terminal_landed_candidate_recovery_blocked",
            },
        )
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            lambda task, receipt: generic_calls.append(str(task.task_alias)),
        )
        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lambda task_cid, **_kwargs: cas_calls.append(str(task_cid)),
        )

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert len(outcomes) == 1
        assert outcomes[0]["task_cid"] == task.task_cid
        expected_reason = (
            "terminal_landed_candidate_manual_authority_required"
            if task_alias == "PCTDD-005"
            else "terminal_landed_candidate_recovery_blocked"
        )
        assert outcomes[0]["reason"] == expected_reason
        assert outcomes[0]["blocked"] is True
        assert generic_calls == []
        assert cas_calls == []
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("task_alias", "attempt_number", "attempts_used", "rearm_count"),
    (
        ("PCTDD-034", 5, 1, None),
        ("PCTDD-034", 7, 1, None),
        ("PCTDD-006", 3, 1, 0),
        ("PCTDD-006", 3, 1, 2),
        ("PCTDD-034", 6, 0, None),
        ("PCTDD-034", 6, 2, None),
    ),
)
def test_terminal_no_effect_route_selector_rejects_near_historical_budget(
    task_alias: str,
    attempt_number: int,
    attempts_used: int,
    rearm_count: int | None,
) -> None:
    """No neighboring ordinal or budget inherits migration authority."""

    case = _terminal_no_effect_historical_selector_case(
        task_alias=task_alias,
        attempt_number=attempt_number,
        attempts_used=attempts_used,
        rearm_count=rearm_count,
    )
    task_before = case.task.to_dict()
    attempt_before = case.attempt.to_dict()
    receipt_before = json.loads(json.dumps(case.receipt))

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    # The 3 == 1 + 2 case remains eligible for the generic exact-count
    # non-consuming verifier so the established deferred route is preserved;
    # its returned terminal-migration schema still fails the closed tuple gate.
    expected_verifier_calls = (
        [case.attempt]
        if (task_alias, attempt_number, attempts_used, rearm_count)
        == ("PCTDD-006", 3, 1, 2)
        else []
    )
    assert case.calls["verifier"] == expected_verifier_calls
    if expected_verifier_calls:
        assert case.calls["journal"] == [
            ("effect", f"effect:{case.attempt.attempt_id}"),
            ("provider", f"provider:{case.attempt.attempt_id}"),
        ]
        assert len(case.calls["provider"]) == 1
        assert len(case.calls["effect"]) == 1
    else:
        assert case.calls["journal"] == []
        assert case.calls["provider"] == []
        assert case.calls["effect"] == []
    assert case.task.to_dict() == task_before
    assert case.attempt.to_dict() == attempt_before
    assert case.receipt == receipt_before


def test_terminal_no_effect_route_selector_rejects_terminal_claim_fence_mismatch() -> None:
    case = _terminal_no_effect_historical_selector_case()
    claim = case.daemon._selector_coordinator.get_task_claim(
        case.attempt.claim_id
    )
    assert claim is not None
    exact = claim.to_dict()
    claim.to_dict = lambda: {
        **exact,
        "fencing_token": int(exact["fencing_token"]) + 1,
    }

    assert case.daemon._database_portal_no_provider_rearm_evidence(
        case.task,
        case.receipt,
    ) is None
    assert case.calls["verifier"] == []
    assert len(case.calls["provider"]) == 1
    assert len(case.calls["effect"]) == 1


def test_terminal_no_effect_route_rearm_uses_existing_saga_without_dispatch(
    tmp_path: Path,
) -> None:
    _count_zero_deferred_provider_rearm_task(
        tmp_path,
        task_alias="PCTDD-005",
    )
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-no-effect-rearm",
        provider_calls=provider_calls,
        max_task_attempts=1,
    )

    class ExactTerminalNoEffectBridge:
        def validate_active_attempt_roots(
            self,
            attempts: list[DatabaseTaskAttempt],
        ) -> dict[str, str]:
            assert attempts == []
            return {}

        def no_provider_dispatch_rearm_evidence(
            self,
            candidate: DatabaseTaskAttempt,
            *,
            outer_block_receipt: dict[str, object],
        ) -> dict[str, object]:
            return _terminal_no_effect_route_rearm_evidence(
                candidate,
                outer_block_receipt,
            )

    try:
        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 2
        attempt = daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        key = f"provider:{attempt.attempt_id}"
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=key,
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=key,
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        _failed, blocked_receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        assert blocked_receipt["unknown_outcome_rearm_count"] == 0
        daemon._database_portal_bridge = ExactTerminalNoEffectBridge()

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert len(outcomes) == 1
        assert outcomes[0]["rearmed"] is True
        assert outcomes[0]["unknown_outcome_rearm_count"] == 0
        assert outcomes[0]["provider_dispatched"] is False
        assert provider_calls == []
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "retrying"
        receipt = dict(task.body["completion_receipt"])
        assert receipt["attempts_used"] == 0
        assert receipt["retry_exhausted"] is False
        assert receipt["unknown_outcome_rearm_count"] == 0
        assert receipt["no_provider_rearm_evidence"]["schema"] == (
            DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA
        )
        assert receipt["no_provider_rearm_fence"]["state"] == "admitted"
        assert DatabaseImplementationDaemon._no_provider_rearm_fence_state(
            task
        ) == "admitted"

        for historical_attempt, historical_rearm_count in (
            (1, 0),
            (3, 0),
            (2, 1),
        ):
            malformed_receipt = json.loads(json.dumps(receipt))
            original = malformed_receipt[
                "no_provider_rearm_original_block_receipt"
            ]
            original["attempt_number"] = historical_attempt
            if historical_rearm_count:
                original["unknown_outcome_rearm_count"] = (
                    historical_rearm_count
                )
            else:
                original.pop("unknown_outcome_rearm_count", None)
            malformed_evidence = malformed_receipt[
                "no_provider_rearm_evidence"
            ]
            malformed_evidence["attempt_number"] = historical_attempt
            blocked_digest = (
                DatabaseImplementationDaemon._database_no_provider_rearm_digest(
                    original
                )
            )
            malformed_evidence["outer_block_receipt_digest"] = blocked_digest
            _rehash_terminal_no_effect_route_evidence(malformed_evidence)
            evidence_id = str(malformed_evidence["evidence_id"])
            malformed_receipt["no_provider_rearm_evidence_id"] = evidence_id
            malformed_receipt["unknown_outcome_rearm_count"] = (
                historical_rearm_count
            )
            malformed_fence = malformed_receipt["no_provider_rearm_fence"]
            malformed_fence["evidence_id"] = evidence_id
            malformed_fence["blocked_receipt_digest"] = blocked_digest
            malformed_saga_id = (
                DatabaseImplementationDaemon._database_no_provider_rearm_saga_id(
                    saga_nonce=str(malformed_fence["saga_nonce"]),
                    task_cid=str(task.task_cid),
                    attempt_id=str(original["attempt_id"]),
                    claim_id=str(original["claim_id"]),
                    evidence_id=evidence_id,
                    blocked_receipt_digest=blocked_digest,
                    blocked_revision=int(malformed_fence["blocked_revision"]),
                )
            )
            malformed_fence["saga_id"] = malformed_saga_id
            malformed_receipt["no_provider_rearm_saga_id"] = (
                malformed_saga_id
            )
            immutable = dict(malformed_receipt)
            immutable.pop("no_provider_rearm_fence")
            malformed_fence["immutable_receipt_digest"] = (
                DatabaseImplementationDaemon._database_no_provider_rearm_digest(
                    immutable
                )
            )
            malformed_task = SimpleNamespace(
                task_cid=task.task_cid,
                task_alias=task.task_alias,
                revision=task.revision,
                status=task.status,
                body={"completion_receipt": malformed_receipt},
            )
            assert DatabaseImplementationDaemon._no_provider_rearm_fence_state(
                malformed_task
            ) == "invalid"

        receipt_before = json.loads(json.dumps(receipt))
        assert daemon.reconcile_blocked_unknown_outcome_tasks() == []
        replayed = daemon.task_source.get(attempt.task_cid)
        assert replayed is not None
        assert replayed.body["completion_receipt"] == receipt_before
        assert provider_calls == []
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "failure_mode",
    ("none", "stale_head", "post_cas_artifact_replaced"),
)
def test_quiesced_stale_release_rearm_is_fenced_nonconsuming_and_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """The new occurrence proof may authorize one semantic rearm, never work."""

    # One count-zero predecessor followed by the exact terminal-no-effect
    # migration creates the current PCTDD-005 outer ordinal without consuming
    # its one-attempt budget.
    _count_zero_deferred_provider_rearm_task(
        tmp_path,
        task_alias="PCTDD-005",
    )
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    validation_calls: list[str] = []

    def validation(
        attempt: DatabaseTaskAttempt,
        _effect_result: object,
    ) -> dict[str, object]:
        validation_calls.append(attempt.attempt_id)
        return {"status": "passed"}

    daemon = _open_daemon(
        tmp_path,
        session=f"session:quiesced-release:{failure_mode}",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=validation,
        max_task_attempts=1,
    )
    try:
        predecessor = daemon.claim_next()
        assert predecessor is not None and predecessor.attempt_number == 2
        predecessor = daemon.commit_phase(predecessor, ATTEMPT_PHASE_CONTEXT)
        predecessor_key = f"provider:{predecessor.attempt_id}"
        daemon._begin_callback_dispatch(
            predecessor,
            dispatch_kind="provider",
            idempotency_key=predecessor_key,
        )
        daemon._record_callback_dispatch_outcome(
            predecessor,
            dispatch_kind="provider",
            idempotency_key=predecessor_key,
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        daemon._finalize_failed_attempt(
            predecessor,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )

        class ExactTerminalNoEffectPredecessorBridge:
            def validate_active_attempt_roots(
                self,
                attempts: list[DatabaseTaskAttempt],
            ) -> dict[str, str]:
                assert attempts == []
                return {}

            def no_provider_dispatch_rearm_evidence(
                self,
                candidate: DatabaseTaskAttempt,
                *,
                outer_block_receipt: dict[str, object],
            ) -> dict[str, object]:
                return _terminal_no_effect_route_rearm_evidence(
                    candidate,
                    outer_block_receipt,
                )

        daemon._database_portal_bridge = ExactTerminalNoEffectPredecessorBridge()
        predecessor_outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(predecessor_outcomes) == 1
        assert predecessor_outcomes[0]["rearmed"] is True

        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 3
        attempt = daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        provider_key = f"provider:{attempt.attempt_id}"
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=provider_key,
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=provider_key,
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        link: dict[str, object] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "task_cid": attempt.task_cid,
            "attempt_number": attempt.attempt_number,
            "owner_session_id": attempt.owner_session_id,
            "lease_id": attempt.lease_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "binding_id": "sha256:" + "3" * 64,
            "nested_state_digest": "sha256:" + "a" * 64,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": "database_daemon_startup",
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": "sha256:" + "9" * 64,
            "commit_barrier_receipt_id": "sha256:" + "b" * 64,
        }
        link["evidence_id"] = content_identity(link)
        failed, blocked_receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
            reconciliation_evidence=link,
        )
        assert failed.status == "failed"
        assert blocked_receipt["attempt_number"] == 3
        assert blocked_receipt["attempts_used"] == 1
        assert blocked_receipt.get("unknown_outcome_rearm_count", 0) == 0

        evidence: dict[str, object] = {
            "schema": (
                DATABASE_PORTAL_QUIESCED_STALE_DISPATCH_RELEASE_REARM_EVIDENCE_SCHEMA
            ),
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "task_cid": attempt.task_cid,
            "task_alias": attempt.task_alias,
            "attempt_number": attempt.attempt_number,
            "owner_session_id": attempt.owner_session_id,
            "lease_id": attempt.lease_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "terminal_reconciliation_evidence_id": link["evidence_id"],
            "event_head_id": "sha256:" + "c" * 64,
            "provider_dispatched": False,
            "implementation_dispatched": False,
            "attempt_consumed": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "recovery_terminal": True,
            "retained_candidate_disposition": "preserved_unvalidated",
        }
        evidence["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        nested_barrier_calls: list[str] = []

        class ExactQuiescedReleaseBridge:
            def no_provider_dispatch_rearm_evidence(
                self,
                candidate: DatabaseTaskAttempt,
                *,
                outer_block_receipt: dict[str, object],
            ) -> dict[str, object]:
                assert candidate.to_dict() == failed.to_dict()
                assert outer_block_receipt == blocked_receipt
                return dict(evidence)

            def execute_with_revalidated_quiesced_stale_dispatch_release(
                self,
                candidate: DatabaseTaskAttempt,
                *,
                outer_block_receipt: object,
                expected_evidence: object,
                callback: Callable[[], object],
            ) -> object:
                assert candidate.to_dict() == failed.to_dict()
                assert outer_block_receipt == blocked_receipt
                assert expected_evidence == evidence
                nested_barrier_calls.append(str(evidence["event_head_id"]))
                if failure_mode == "stale_head":
                    raise DatabasePortalBridgeError(
                        "nested event head advanced before control CAS"
                    )
                result = callback()
                if failure_mode == "post_cas_artifact_replaced":
                    raise DatabasePortalBridgeError(
                        "release receipt was atomically replaced after control CAS"
                    )
                return result

        daemon._database_portal_bridge = ExactQuiescedReleaseBridge()
        monkeypatch.setattr(
            daemon,
            "_database_portal_terminal_reconciliation_saga",
            lambda _candidate: {
                "attempt_id": attempt.attempt_id,
                "claim_id": attempt.claim_id,
                "task_cid": attempt.task_cid,
                "attempt_number": attempt.attempt_number,
                "owner_session_id": attempt.owner_session_id,
                "lease_id": attempt.lease_id,
                "fencing_token": attempt.fencing_token,
                "fence_epoch": attempt.fence_epoch,
                "intended_database_disposition": "blocked_unknown_outcome",
                "evidence_id": link["evidence_id"],
                "prepared_reconciliation_receipt_id": link[
                    "prepared_reconciliation_receipt_id"
                ],
                "commit_barrier_receipt_id": link["commit_barrier_receipt_id"],
                "stage": "terminal",
                "receipt_id": "sha256:" + "d" * 64,
            },
        )
        monkeypatch.setattr(
            daemon,
            "_valid_no_provider_rearm_evidence",
            lambda *_args, **_kwargs: True,
        )
        if failure_mode == "post_cas_artifact_replaced":
            # Compensation revalidates through the class-level static gate.
            # This integration fixture deliberately carries only the fields
            # needed to exercise the saga; schema closure is covered by the
            # dedicated evidence-validator matrix.
            monkeypatch.setattr(
                DatabaseImplementationDaemon,
                "_valid_no_provider_rearm_evidence",
                staticmethod(lambda *_args, **_kwargs: True),
            )

        terminal_barrier_calls: list[str] = []
        real_terminal_barrier = (
            daemon.coordinator.execute_with_terminal_task_claim_barrier
        )

        def terminal_barrier(
            claim: object,
            callback: Callable[[], object],
            *,
            lease: object,
        ) -> object:
            terminal_barrier_calls.append(str(getattr(claim, "claim_id", "")))
            return real_terminal_barrier(claim, callback, lease=lease)

        monkeypatch.setattr(
            daemon.coordinator,
            "execute_with_terminal_task_claim_barrier",
            terminal_barrier,
        )
        cas_calls: list[str] = []
        real_cas = daemon._cas_task_status_database

        def tracked_cas(*args: object, **kwargs: object) -> object:
            cas_calls.append(str(kwargs.get("new_status") or ""))
            return real_cas(*args, **kwargs)

        monkeypatch.setattr(daemon, "_cas_task_status_database", tracked_cas)
        attempt_before = failed.to_dict()

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert terminal_barrier_calls == [attempt.claim_id]
        assert nested_barrier_calls == [evidence["event_head_id"]]
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == attempt_before
        assert provider_calls == []
        assert effect_calls == []
        assert validation_calls == []
        assert daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=provider_key,
        ) is None
        assert daemon.effect_claim_recorded(
            attempt.attempt_id,
            idempotency_key=f"effect:{attempt.attempt_id}",
        ) is None
        assert daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        if failure_mode == "stale_head":
            assert cas_calls == []
            assert task.status == "blocked"
            assert not any(outcome.get("rearmed") is True for outcome in outcomes)
        elif failure_mode == "post_cas_artifact_replaced":
            # A non-cooperating producer can race the directory flock.  The
            # retained-FD postcheck raises after the first status CAS, and the
            # existing pending-fence saga restores blocked before dispatch.
            # Both the tentative rearm and its fail-closed compensation use
            # the instrumented daemon CAS facade.
            assert cas_calls == ["retrying", "blocked"]
            assert task.status == "blocked"
            assert not any(outcome.get("rearmed") is True for outcome in outcomes)
        else:
            # The established shared-fence saga uses three physical writes, but
            # exactly one terminal barrier and one nested proof authorize the
            # single semantic blocked -> retrying rearm.
            assert cas_calls == ["retrying", "blocked", "retrying"]
            assert len(outcomes) == 1 and outcomes[0]["rearmed"] is True
            assert task.status == "retrying"
            receipt = dict(task.body["completion_receipt"])
            assert receipt["attempts_used"] == 0
            assert receipt["unknown_outcome_rearm_count"] == 0
            assert receipt["no_provider_rearm_evidence"] == evidence
            assert receipt["no_provider_rearm_fence"]["state"] == "admitted"
    finally:
        daemon.close()


def _count_zero_deferred_provider_rearm_task(
    tmp_path: Path,
    *,
    task_alias: str = "DQP-T001",
) -> SimpleNamespace:
    """Create one exact admitted count-zero deferred-provider rearm."""

    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:count-zero-deferred-rearm",
        provider_calls=provider_calls,
        max_task_attempts=1,
    )

    class ExactDeferredProviderBridge:
        def validate_active_attempt_roots(
            self,
            attempts: list[DatabaseTaskAttempt],
        ) -> dict[str, str]:
            assert attempts == []
            return {}

        def no_provider_dispatch_rearm_evidence(
            self,
            candidate: DatabaseTaskAttempt,
            *,
            outer_block_receipt: dict[str, object],
        ) -> dict[str, object]:
            return _deferred_provider_rearm_evidence(
                daemon,
                candidate,
                outer_block_receipt,
            )

    try:
        population = _population(1)
        population["tasks"][0]["task_id"] = task_alias
        if task_alias == DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN["task_alias"]:
            # A reserved alias must retain its canonical CID even when this
            # fixture models the earlier, ordinary pre-migration epoch.
            population["tasks"][0]["task_cid"] = (
                DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN["task_cid"]
            )
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 1
        attempt = daemon.commit_phase(attempt, ATTEMPT_PHASE_CONTEXT)
        idempotency_key = f"provider:{attempt.attempt_id}"
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=idempotency_key,
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=idempotency_key,
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        _failed, blocked_receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        assert blocked_receipt["attempts_used"] == 1
        assert "unknown_outcome_rearm_count" not in blocked_receipt

        daemon._database_portal_bridge = ExactDeferredProviderBridge()
        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(outcomes) == 1
        assert outcomes[0]["unknown_outcome_rearm_count"] == 0
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "retrying"
        receipt = json.loads(
            json.dumps(task.body["completion_receipt"])
        )
        assert receipt["unknown_outcome_rearm_count"] == 0
        assert receipt["attempts_used"] == 0
        assert receipt["no_provider_rearm_evidence"]["schema"] == (
            DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA
        )
        assert receipt["no_provider_rearm_fence"]["state"] == "admitted"
        assert provider_calls == []
        return SimpleNamespace(
            task_cid=str(task.task_cid),
            task_alias=str(task.task_alias),
            revision=int(task.revision),
            status=str(task.status),
            body={"completion_receipt": receipt},
        )
    finally:
        daemon.close()


def test_attempt_two_deferred_provider_evidence_is_nonconsuming_and_shared_fenced(
    tmp_path: Path,
) -> None:
    """Admit the exact P006/P007 2=1+1 route without widening its budget."""

    provider_calls: list[str] = []
    seed = _open_daemon(
        tmp_path,
        session="session:deferred-provider-attempt-two",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        attempt_one = seed.claim_next()
        assert attempt_one is not None and attempt_one.attempt_number == 1
        seed._begin_callback_dispatch(
            attempt_one,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_one.attempt_id}",
        )
        seed._finalize_failed_attempt(
            attempt_one,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
    finally:
        seed.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:deferred-provider-attempt-two",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )

    verifier_calls: list[tuple[int, int, str]] = []

    class ExactDeferredProviderBridge:
        def validate_active_attempt_roots(
            self,
            attempts: list[DatabaseTaskAttempt],
        ) -> dict[str, str]:
            assert attempts == []
            return {}

        def no_provider_dispatch_rearm_evidence(
            self,
            candidate: DatabaseTaskAttempt,
            *,
            outer_block_receipt: dict[str, object],
        ) -> dict[str, object] | None:
            verifier_calls.append(
                (
                    int(outer_block_receipt.get("attempt_number") or 0),
                    int(
                        outer_block_receipt.get("unknown_outcome_rearm_count")
                        or 0
                    ),
                    str(outer_block_receipt.get("reason") or ""),
                )
            )
            if outer_block_receipt.get("reason") != (
                "callback_authority_incomplete_blocked"
            ):
                return None
            return _deferred_provider_rearm_evidence(
                daemon,
                candidate,
                outer_block_receipt,
            )

    try:
        first_rearm = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(first_rearm) == 1
        assert first_rearm[0]["unknown_outcome_rearm_count"] == 1
        attempt_two = daemon.claim_next()
        assert attempt_two is not None and attempt_two.attempt_number == 2
        attempt_two = daemon.commit_phase(
            attempt_two,
            ATTEMPT_PHASE_CONTEXT,
        )
        daemon._begin_callback_dispatch(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
            outcome="raised",
            body={"exception_type": "DatabasePortalBridgeError"},
        )
        _failed, blocked_receipt = daemon._finalize_failed_attempt(
            attempt_two,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        assert blocked_receipt["attempt_number"] == 2
        assert blocked_receipt["attempts_used"] == 1
        assert blocked_receipt["unknown_outcome_rearm_count"] == 1
        blocked_task = daemon.task_source.get(attempt_two.task_cid)
        assert blocked_task is not None and blocked_task.status == "blocked"

        bridge = ExactDeferredProviderBridge()
        daemon._database_portal_bridge = bridge
        for wrong_count in (0, 2):
            inexact_count = {
                **blocked_receipt,
                "unknown_outcome_rearm_count": wrong_count,
            }
            calls_before = len(verifier_calls)
            assert daemon._database_portal_no_provider_rearm_evidence(
                blocked_task,
                inexact_count,
            ) is None
            assert len(verifier_calls) == calls_before

        wrong_reason = {
            **blocked_receipt,
            "reason": "provider_dispatch_outcome_unknown",
        }
        calls_before = len(verifier_calls)
        assert daemon._database_portal_no_provider_rearm_evidence(
            blocked_task,
            wrong_reason,
        ) is None
        assert verifier_calls[calls_before:] == [
            (2, 1, "provider_dispatch_outcome_unknown")
        ]

        admitted_evidence = daemon._database_portal_no_provider_rearm_evidence(
            blocked_task,
            blocked_receipt,
        )
        assert admitted_evidence is not None
        assert admitted_evidence["schema"] == (
            DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA
        )
        assert admitted_evidence["attempt_consumed"] is False
        assert verifier_calls[-1] == (
            2,
            1,
            "callback_authority_incomplete_blocked",
        )

        result = daemon.run_once()

        assert result["selection_idle_reason"] == (
            "database_unknown_outcomes_rearmed"
        )
        assert result["implementation_result"] is None
        assert len(result["unknown_outcome_rearms"]) == 1
        outcome = result["unknown_outcome_rearms"][0]
        assert outcome["previous_attempt_id"] == attempt_two.attempt_id
        assert outcome["unknown_outcome_rearm_count"] == 1
        assert outcome["provider_dispatched"] is False
        rearmed = daemon.task_source.get(attempt_two.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        rearm_receipt = rearmed.body["completion_receipt"]
        assert rearm_receipt["attempts_used"] == 0
        assert rearm_receipt["unknown_outcome_rearm_count"] == 1
        assert rearm_receipt["no_provider_rearm_evidence"]["schema"] == (
            DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA
        )
        assert rearm_receipt["no_provider_rearm_fence"]["state"] == "admitted"
        assert daemon._no_provider_rearm_fence_state(rearmed) == "admitted"
        assert not daemon._automatic_claim_forbidden(rearmed)
        assert provider_calls == []

        for field, value in (
            ("unknown_outcome_rearm_count", 0),
            ("unknown_outcome_rearm_count", 2),
            ("reason", "provider_dispatch_outcome_unknown"),
        ):
            malformed_receipt = json.loads(json.dumps(rearm_receipt))
            malformed_original = dict(
                malformed_receipt["no_provider_rearm_original_block_receipt"]
            )
            malformed_original[field] = value
            malformed_receipt[
                "no_provider_rearm_original_block_receipt"
            ] = malformed_original
            malformed_task = SimpleNamespace(
                task_cid=rearmed.task_cid,
                task_alias=rearmed.task_alias,
                revision=rearmed.revision,
                status=rearmed.status,
                body={
                    **dict(rearmed.body),
                    "completion_receipt": malformed_receipt,
                },
            )
            assert daemon._no_provider_rearm_fence_state(malformed_task) == (
                "invalid"
            )
            assert daemon._automatic_claim_forbidden(malformed_task)
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "fence_state",
    ("pending", "admitting", "compensating"),
)
def test_count_zero_proof_backed_shared_fence_crash_states_compensate(
    tmp_path: Path,
    fence_state: str,
) -> None:
    """Recover every pre-admission crash using the exact count-zero proof."""

    admitted = _count_zero_deferred_provider_rearm_task(tmp_path)
    receipt = json.loads(json.dumps(admitted.body["completion_receipt"]))
    fence = dict(receipt["no_provider_rearm_fence"])
    retrying_revision = int(fence["retrying_revision"])
    admitted_revision = int(fence["admitted_revision"])
    if fence_state == "pending":
        fence["state"] = "pending"
        fence["admitted_revision"] = 0
        task_revision = retrying_revision
        task_status = "retrying"
        expected_cas_count = 1
    elif fence_state == "admitting":
        fence["state"] = "admitting"
        task_revision = retrying_revision + 1
        task_status = "blocked"
        expected_cas_count = 2
    else:
        fence["state"] = "compensating"
        task_revision = admitted_revision
        task_status = "retrying"
        expected_cas_count = 1
    receipt["no_provider_rearm_fence"] = fence
    crash_task = SimpleNamespace(
        task_cid=admitted.task_cid,
        task_alias=admitted.task_alias,
        revision=task_revision,
        status=task_status,
        body={"completion_receipt": receipt},
    )
    assert DatabaseImplementationDaemon._no_provider_rearm_fence_state(
        crash_task
    ) == fence_state

    class SharedTaskSource:
        def __init__(self, task: SimpleNamespace) -> None:
            self.current = task

        def list_tasks(self, *, limit: int) -> SimpleNamespace:
            assert limit > 0
            return SimpleNamespace(tasks=(self.current,))

        def get(self, task_cid: str) -> SimpleNamespace | None:
            if task_cid != self.current.task_cid:
                return None
            return self.current

    source = SharedTaskSource(crash_task)
    daemon = object.__new__(_StaleDispatchSelectorHarness)
    daemon._selector_task_source = source
    cas_calls: list[tuple[int, str]] = []

    def exact_cas(
        task_cid: str,
        *,
        expected_revision: int,
        new_status: str,
        receipt: dict[str, object],
    ) -> SimpleNamespace:
        current = source.current
        assert task_cid == current.task_cid
        assert expected_revision == current.revision
        cas_calls.append((expected_revision, new_status))
        updated = SimpleNamespace(
            task_cid=current.task_cid,
            task_alias=current.task_alias,
            revision=current.revision + 1,
            status=new_status,
            body={"completion_receipt": dict(receipt)},
        )
        source.current = updated
        return SimpleNamespace(task=updated)

    daemon._cas_task_status_database = exact_cas
    outcomes = daemon._reconcile_shared_no_provider_rearm_fences()

    assert len(outcomes) == 1
    assert outcomes[0]["prior_fence_state"] == fence_state
    assert outcomes[0]["control_compensated"] is True
    assert len(cas_calls) == expected_cas_count
    assert source.current.status == "blocked"
    assert source.current.revision == task_revision + expected_cas_count
    compensation_receipt = source.current.body["completion_receipt"]
    assert compensation_receipt["operation"] == (
        "database_unknown_outcome_blocked"
    )
    assert compensation_receipt["unknown_outcome_rearm_count"] == 0
    compensation = compensation_receipt["no_provider_rearm_compensation"]
    assert compensation["saga_id"] == fence["saga_id"]
    assert compensation["evidence_id"] == fence["evidence_id"]
    assert compensation["retrying_revision"] == (
        source.current.revision - 1
    )
    assert compensation["compensated_revision"] == source.current.revision


def test_count_zero_shared_fence_compensation_schema_policy_is_closed(
    tmp_path: Path,
) -> None:
    """Count zero is reserved for the exact proof-backed schema vocabulary."""

    admitted = _count_zero_deferred_provider_rearm_task(tmp_path)
    exact_nonconsuming_schemas = {
        DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_HISTORICAL_INTERRUPTED_IMPLEMENTATION_STATE_TRANSITION_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_QUIESCED_STALE_DISPATCH_RELEASE_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_STALE_DISPATCH_MIGRATION_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_TERMINAL_QUIESCENT_DEFERRED_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_TERMINAL_NO_EFFECT_ROUTE_REARM_EVIDENCE_SCHEMA,
    }
    for schema in exact_nonconsuming_schemas:
        receipt = json.loads(json.dumps(admitted.body["completion_receipt"]))
        receipt["no_provider_rearm_evidence"]["schema"] = schema
        task = SimpleNamespace(
            task_cid=admitted.task_cid,
            task_alias=admitted.task_alias,
            revision=admitted.revision,
            status=admitted.status,
            body={"completion_receipt": receipt},
        )
        compensation = (
            DatabaseImplementationDaemon._shared_no_provider_rearm_compensation_receipt(
                task,
                retrying_revision=task.revision,
            )
        )
        assert compensation is not None, schema
        assert compensation["unknown_outcome_rearm_count"] == 0

    for schema in (
        DATABASE_PORTAL_NO_PROVIDER_REARM_EVIDENCE_SCHEMA,
        DATABASE_PORTAL_DEFERRED_PROVIDER_REARM_EVIDENCE_SCHEMA + "-near-miss",
        "",
    ):
        receipt = json.loads(json.dumps(admitted.body["completion_receipt"]))
        receipt["no_provider_rearm_evidence"]["schema"] = schema
        task = SimpleNamespace(
            task_cid=admitted.task_cid,
            task_alias=admitted.task_alias,
            revision=admitted.revision,
            status=admitted.status,
            body={"completion_receipt": receipt},
        )
        assert (
            DatabaseImplementationDaemon._shared_no_provider_rearm_compensation_receipt(
                task,
                retrying_revision=task.revision,
            )
            is None
        ), schema
