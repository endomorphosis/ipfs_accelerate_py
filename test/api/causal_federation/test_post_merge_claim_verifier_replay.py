"""Exact generation-nine post-merge claim-verifier replay regressions."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskRecord,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    DATABASE_VIRGIN_TASK_TRANSFER_BINDING_SCHEMA,
    DATABASE_VIRGIN_TASK_TRANSFER_CURSOR_SCHEMA,
    DATABASE_VIRGIN_TASK_TRANSFER_MODE,
    database_task_alias_home_shard_index,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    DATABASE_POST_MERGE_COMPLETION_CLAIM_VERIFIER_FAILURE_REASON,
    DATABASE_POST_MERGE_COMPLETION_CLAIM_VERIFIER_REPLAY_OPERATION,
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerAuthorizationError,
    _process_birth_content_id,
    validated_post_merge_completion_claim_verifier_replay_lineage,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    _database_provider_callback_unknown_fingerprint,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability,
    _migrate,
)

_TASK_CID = "task:generation-nine-claim-verifier"
_TASK_ALIAS = "SPAR-002"
_SOURCE_REVISION = 10
_TERMINAL_REVISION = 14
_OLD_IDENTITY_FIELDS = (
    "attempt_id",
    "attempt_number",
    "claim_id",
    "lease_id",
    "owner_session_id",
    "fencing_token",
    "fence_epoch",
)
_EXECUTION_FIELDS = (
    "execution_phase",
    "execution_revision",
    "execution_finished_at_ms",
)


def _seed_id(seed: dict[str, Any]) -> str:
    body = dict(seed)
    body.pop("seed_id", None)
    return "sha256:" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _transfer_cursor(
    *,
    binding: dict[str, Any],
    identity: dict[str, Any],
    claimed_from_revision: int,
) -> dict[str, Any]:
    body = {
        "schema": DATABASE_VIRGIN_TASK_TRANSFER_CURSOR_SCHEMA,
        "binding_id": binding["binding_id"],
        "claim_id": identity["claim_id"],
        "attempt_id": identity["attempt_id"],
        "owner_session_id": identity["owner_session_id"],
        "lease_id": identity["lease_id"],
        "fencing_token": identity["fencing_token"],
        "fence_epoch": identity["fence_epoch"],
        "claimed_from_revision": claimed_from_revision,
    }
    return {**body, "cursor_id": content_identity(body)}


def _generation_nine_fixture(
    *,
    with_transfer: bool = False,
) -> dict[str, Any]:
    old_identity = {
        "attempt_id": "attempt:generation-nine-source",
        "attempt_number": 3,
        "claim_id": "claim:generation-nine-source",
        "lease_id": "lease:generation-nine-source",
        "owner_session_id": "session:generation-nine-source",
        "fencing_token": 3,
        "fence_epoch": 3,
    }
    old_execution = {
        "execution_phase": "failed",
        "execution_revision": 3,
        "execution_finished_at_ms": 10_003,
    }
    current_identity = {
        "attempt_id": "attempt:generation-nine-verifier",
        "attempt_number": 4,
        "claim_id": "claim:generation-nine-verifier",
        "lease_id": "lease:generation-nine-verifier",
        "owner_session_id": "session:generation-nine-verifier",
        "fencing_token": 4,
        "fence_epoch": 4,
    }
    if with_transfer:
        # Virgin-transfer retries retain the recipient lane's owner session
        # while advancing claim, attempt, lease, and fence identities.
        current_identity["owner_session_id"] = old_identity["owner_session_id"]
    current_execution = {
        "execution_phase": "failed",
        "execution_revision": 3,
        "execution_finished_at_ms": 14_003,
    }
    route = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/task-execution-route-binding@1"),
        "policy_id": "policy:generation-nine",
        "plan_root_cid": "plan:generation-nine",
        "repository_tree_id": "tree:generation-nine",
        "source_revision": 1,
        "task_cid": _TASK_CID,
        "task_alias": _TASK_ALIAS,
        "task_revision": 1,
        "task_contract_cid": "contract:generation-nine",
        "execution_mode": "deterministic-only",
    }
    route_lineage = {
        "execution_route_binding": route,
        "execution_route_policy_id": route["policy_id"],
        "execution_route_origin_revision": route["task_revision"],
    }
    source_transfer_lineage: dict[str, Any] = {}
    current_transfer_lineage: dict[str, Any] = {}
    transfer_recipient = 0
    if with_transfer:
        shard_count = 3
        home = database_task_alias_home_shard_index(_TASK_ALIAS, shard_count)
        transfer_recipient = (home + 1) % shard_count
        cohort = {
            "kind": "database-virgin-task-transfer-cohort",
            "task_prefix": "SPAR-",
            "task_shard_count": shard_count,
            "claim_policy_id": "",
            "store_generation": "",
        }
        binding_body = {
            "schema": DATABASE_VIRGIN_TASK_TRANSFER_BINDING_SCHEMA,
            "mode": DATABASE_VIRGIN_TASK_TRANSFER_MODE,
            "cohort_id": content_identity(cohort),
            "claim_policy_id": "",
            "store_generation": "",
            "task_cid": _TASK_CID,
            "task_alias": _TASK_ALIAS,
            "task_prefix": "SPAR-",
            "task_shard_count": shard_count,
            "home_shard_index": home,
            "recipient_shard_index": transfer_recipient,
            "source_task_revision": _SOURCE_REVISION - 1,
            "claim_id": old_identity["claim_id"],
            "attempt_id": old_identity["attempt_id"],
            "owner_session_id": old_identity["owner_session_id"],
            "lease_id": old_identity["lease_id"],
            "fencing_token": old_identity["fencing_token"],
            "fence_epoch": old_identity["fence_epoch"],
        }
        binding = {
            **binding_body,
            "binding_id": content_identity(binding_body),
        }
        source_transfer_lineage = {
            "virgin_task_transfer": binding,
            "virgin_task_transfer_claim_cursor": _transfer_cursor(
                binding=binding,
                identity=old_identity,
                claimed_from_revision=_SOURCE_REVISION - 1,
            ),
        }
        current_transfer_lineage = {
            "virgin_task_transfer": binding,
            "virgin_task_transfer_claim_cursor": _transfer_cursor(
                binding=binding,
                identity=current_identity,
                claimed_from_revision=_SOURCE_REVISION + 1,
            ),
        }
    source_terminal = {
        "operation": "database_portal_terminal_failure",
        **old_identity,
        **old_execution,
        "reason": "Portal callback reconciliation binding is invalid",
        "retryable": False,
        "coordination": {},
        "control_expected_status": "in_progress",
        "control_expected_revision": _SOURCE_REVISION - 1,
        **route_lineage,
        **source_transfer_lineage,
    }
    old_seed = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@1"
        ),
        "task_cid": _TASK_CID,
        "task_alias": _TASK_ALIAS,
        **old_identity,
        "source_task_revision": _SOURCE_REVISION,
        "request_id": "request:generation-nine",
        "candidate_commit": "1" * 40,
        "qualified_target_commit": "2" * 40,
        "qualification_kind": "callback_integration",
        "qualification_receipt_id": "receipt:generation-nine-old",
        "queue_source_attempt_id": old_identity["attempt_id"],
        "queue_source_claim_id": old_identity["claim_id"],
        "queue_source_lease_id": old_identity["lease_id"],
        "queue_source_fencing_token": old_identity["fencing_token"],
        "queue_source_fence_epoch": old_identity["fence_epoch"],
        "queue_source_binding_id": "sha256:" + "3" * 64,
        "queue_source_projection_immutable_digest": "sha256:" + "4" * 64,
        "recovery_evidence_id": "sha256:" + "5" * 64,
        "terminal_reason": source_terminal["reason"],
    }
    old_seed["seed_id"] = _seed_id(old_seed)
    old_queue_reason = (
        "database_post_merge_declared_outputs_callback_integration:"
        + old_seed["request_id"]
        + ":"
        + old_seed["qualification_receipt_id"]
    )
    recovery = {
        "operation": ("database_post_merge_declared_outputs_callback_integration_recovery"),
        **old_identity,
        **old_execution,
        "request_id": old_seed["request_id"],
        "candidate_commit": old_seed["candidate_commit"],
        "source_binding_id": old_seed["queue_source_binding_id"],
        "source_projection_immutable_digest": old_seed["queue_source_projection_immutable_digest"],
        "queue_reason": old_queue_reason,
        "queue_receipt": {"historical_queue_revision": 3},
        "coordination": {
            "attempt_id": old_identity["attempt_id"],
            "claim_id": old_identity["claim_id"],
            "attempt_number": old_identity["attempt_number"],
        },
        "control_expected_status": "blocked",
        "control_expected_revision": _SOURCE_REVISION,
        "source_integration_commit": "6" * 40,
        "source_train_receipt_id": "sha256:" + "7" * 64,
        "qualified_target_commit": old_seed["qualified_target_commit"],
        "callback_requalification_receipt_id": old_seed["qualification_receipt_id"],
        "callback_reconciliation_evidence_id": old_seed["recovery_evidence_id"],
        "post_merge_completion_recovery_seed": old_seed,
    }
    pid = 101
    start_time_ticks = 202
    boot_id = "boot:generation-nine"
    parent_pid = 100
    process_attestation = {
        "schema": TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
        "grant_id": "grant:generation-nine",
        "client_id": "client:generation-nine",
        "process_birth_id": _process_birth_content_id(pid, start_time_ticks, boot_id, parent_pid),
        "pid": pid,
        "uid": 1_000,
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id,
        "parent_pid": parent_pid,
    }
    seeded_claim = {
        "operation": "database_claim",
        **current_identity,
        "claimed_from_revision": _SOURCE_REVISION + 1,
        "task_shard_count": 3 if with_transfer else 1,
        "task_shard_index": transfer_recipient if with_transfer else 0,
        "strict_task_sharding": True,
        "idle_lane_work_stealing": (
            DATABASE_VIRGIN_TASK_TRANSFER_MODE if with_transfer else "disabled"
        ),
        "task_prefix": "SPAR-",
        "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
        "claim_process_attestation": process_attestation,
        "post_merge_completion_recovery_source_attempt_id": old_identity["attempt_id"],
        "post_merge_completion_recovery_seed": old_seed,
        **route_lineage,
        **current_transfer_lineage,
    }
    admitted_claim = {
        **seeded_claim,
        "operation": "database_attempt_admitted",
        "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
        "admitted_from_revision": _SOURCE_REVISION + 2,
        "attempt_execution_phase": "claimed",
        "attempt_execution_revision": 1,
    }
    verifier_terminal = {
        "operation": "database_portal_terminal_failure",
        **current_identity,
        **current_execution,
        "reason": DATABASE_POST_MERGE_COMPLETION_CLAIM_VERIFIER_FAILURE_REASON,
        "retryable": False,
        # Generation nine's verifier rejected the claim before a worker
        # transition could persist a coordination projection in the shared
        # task receipt.  The exact execution sidecar still owns the complete
        # claim/attempt/lease triple and is checked independently by the
        # daemon before this historical empty projection can be replayed.
        "coordination": {},
        "control_expected_status": "in_progress",
        "control_expected_revision": _TERMINAL_REVISION - 1,
        **route_lineage,
        **current_transfer_lineage,
    }
    semantic_body = {"title": "generation-nine fixture", "ordinal": 2}
    receipts = (
        source_terminal,
        recovery,
        seeded_claim,
        admitted_claim,
        verifier_terminal,
    )
    rows = [
        {
            "revision": revision,
            "status": status,
            "body": {**semantic_body, "completion_receipt": receipt},
        }
        for revision, status, receipt in zip(
            range(_SOURCE_REVISION, _TERMINAL_REVISION + 1),
            ("blocked", "retrying", "in_progress", "in_progress", "blocked"),
            receipts,
            strict=True,
        )
    ]

    # The successor seed preserves the old @1 source lineage.  Only the
    # control generation and freshly-qualified target evidence advance.
    successor_seed = copy.deepcopy(old_seed)
    successor_seed.update(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@2"
            ),
            "recovery_control_revision": _TERMINAL_REVISION,
            "qualified_target_commit": "8" * 40,
            "qualification_receipt_id": "receipt:generation-nine-fresh",
            "recovery_evidence_id": "sha256:" + "9" * 64,
        }
    )
    successor_seed["seed_id"] = _seed_id(successor_seed)
    successor_queue_reason = (
        "database_post_merge_declared_outputs_callback_integration:"
        + successor_seed["request_id"]
        + ":"
        + successor_seed["qualification_receipt_id"]
    )
    transition = {
        "operation": DATABASE_POST_MERGE_COMPLETION_CLAIM_VERIFIER_REPLAY_OPERATION,
        **current_identity,
        **current_execution,
        "request_id": successor_seed["request_id"],
        "candidate_commit": successor_seed["candidate_commit"],
        "source_binding_id": successor_seed["queue_source_binding_id"],
        "source_projection_immutable_digest": successor_seed[
            "queue_source_projection_immutable_digest"
        ],
        "queue_reason": successor_queue_reason,
        "queue_receipt": {},
        "coordination": {
            "attempt_id": current_identity["attempt_id"],
            "claim_id": current_identity["claim_id"],
            "attempt_number": current_identity["attempt_number"],
        },
        "control_expected_status": "blocked",
        "control_expected_revision": _TERMINAL_REVISION,
        "source_integration_commit": "a" * 40,
        "source_train_receipt_id": "sha256:" + "b" * 64,
        "qualified_target_commit": successor_seed["qualified_target_commit"],
        "callback_requalification_receipt_id": successor_seed["qualification_receipt_id"],
        "callback_reconciliation_evidence_id": successor_seed["recovery_evidence_id"],
        "post_merge_completion_recovery_seed": successor_seed,
        **route_lineage,
        **current_transfer_lineage,
    }
    return {
        "rows": rows,
        "terminal_body": rows[-1]["body"],
        "old_identity": old_identity,
        "current_identity": current_identity,
        "transition": transition,
        "queue_reason": successor_queue_reason,
    }


def _validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    return dict(
        validated_post_merge_completion_claim_verifier_replay_lineage(
            task_cid=_TASK_CID,
            task_alias=_TASK_ALIAS,
            task_status="blocked",
            task_revision=_TERMINAL_REVISION,
            task_body=fixture["terminal_body"],
            revisions=fixture["rows"],
        )
    )


def test_exact_generation_nine_five_row_suffix_is_accepted() -> None:
    fixture = _generation_nine_fixture()

    lineage = _validate_fixture(fixture)

    assert lineage["source_task_revision"] == _SOURCE_REVISION
    assert lineage["terminal_task_revision"] == _TERMINAL_REVISION
    assert lineage["source_seed"]["schema"].endswith("recovery-seed@1")
    assert lineage["lineage_id"]


def test_generation_nine_suffix_tamper_and_extra_successor_fail_closed() -> None:
    fixture = _generation_nine_fixture()
    tampered = copy.deepcopy(fixture["rows"])
    tampered[2]["body"]["completion_receipt"][
        "post_merge_completion_recovery_source_attempt_id"
    ] = "attempt:forged"

    with pytest.raises(
        TypedStateOwnerAuthorizationError,
        match="claim lineage differs",
    ):
        validated_post_merge_completion_claim_verifier_replay_lineage(
            task_cid=_TASK_CID,
            task_alias=_TASK_ALIAS,
            task_status="blocked",
            task_revision=_TERMINAL_REVISION,
            task_body=fixture["terminal_body"],
            revisions=tampered,
        )

    nonadvancing = copy.deepcopy(fixture["rows"])
    for row in nonadvancing[2:]:
        receipt = row["body"]["completion_receipt"]
        receipt["attempt_number"] = fixture["old_identity"]["attempt_number"]
        coordination = receipt.get("coordination")
        if isinstance(coordination, dict):
            coordination["attempt_number"] = fixture["old_identity"]["attempt_number"]
    with pytest.raises(
        TypedStateOwnerAuthorizationError,
        match="claim did not advance its source",
    ):
        validated_post_merge_completion_claim_verifier_replay_lineage(
            task_cid=_TASK_CID,
            task_alias=_TASK_ALIAS,
            task_status="blocked",
            task_revision=_TERMINAL_REVISION,
            task_body=nonadvancing[-1]["body"],
            revisions=nonadvancing,
        )

    advanced = [
        *fixture["rows"],
        {
            "revision": _TERMINAL_REVISION + 1,
            "status": "retrying",
            "body": fixture["terminal_body"],
        },
    ]
    with pytest.raises(
        TypedStateOwnerAuthorizationError,
        match="no exact five-row suffix",
    ):
        validated_post_merge_completion_claim_verifier_replay_lineage(
            task_cid=_TASK_CID,
            task_alias=_TASK_ALIAS,
            task_status="blocked",
            task_revision=_TERMINAL_REVISION,
            task_body=fixture["terminal_body"],
            revisions=advanced,
        )


class _MutationMustNotRun:
    def recover_post_merge_retry(self, **_kwargs: Any) -> Any:
        pytest.fail("idempotent response-loss replay attempted a second mutation")


def _response_loss_adapter(
    fixture: dict[str, Any],
) -> tuple[TypedDatabaseTaskSource, TaskRecord]:
    queue_receipt = {"expected_queue_revision": 7}
    stored_transition = {
        **fixture["transition"],
        "queue_receipt": queue_receipt,
    }
    retrying_body = {
        **fixture["terminal_body"],
        "completion_receipt": stored_transition,
    }
    retrying = TaskRecord(
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        goal_cid="goal:generation-nine",
        ordinal=2,
        status="retrying",
        revision=_TERMINAL_REVISION + 1,
        body=retrying_body,
    )
    history = [
        *copy.deepcopy(fixture["rows"]),
        {
            "revision": retrying.revision,
            "status": retrying.status,
            "body": retrying_body,
        },
    ]
    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._client = _MutationMustNotRun()
    adapter.get = lambda _task_cid: retrying
    adapter._retry_cooldown_row = lambda _task_cid: {
        "retry_not_before_ms": 20_000,
        "extension": {
            "task_cid": _TASK_CID,
            "expected_task_revision": _TERMINAL_REVISION,
            "delay_ms": 0,
            "selection_penalty": 0,
            "reason": fixture["queue_reason"],
            "retry_not_before_ms": 20_000,
            **{name: fixture["transition"][name] for name in _OLD_IDENTITY_FIELDS},
        },
    }
    adapter._validate_retrying_cooldown_binding = lambda *_args: None
    adapter.task_revision_history_projection = lambda _task_cid: {"revisions": history}
    adapter.snapshot = lambda: SimpleNamespace(event_cursor=19)
    return adapter, retrying


def test_response_loss_replay_is_idempotent_and_rejects_old_attempt() -> None:
    fixture = _generation_nine_fixture()
    adapter, retrying = _response_loss_adapter(fixture)
    terminal_receipt = fixture["terminal_body"]["completion_receipt"]

    for _ in range(2):
        result = adapter.recover_post_merge_retry(
            task_cid=_TASK_CID,
            expected_revision=_TERMINAL_REVISION,
            expected_control_receipt=terminal_receipt,
            status="retrying",
            receipt=fixture["transition"],
            delay_ms=0,
            reason=fixture["queue_reason"],
            exact_retry_not_before_ms=20_000,
        )
        assert result["cas_result"].task == retrying
        assert result["cas_result"].changed is False

    old_attempt_transition = copy.deepcopy(fixture["transition"])
    source_receipt = fixture["rows"][1]["body"]["completion_receipt"]
    for name in (*_OLD_IDENTITY_FIELDS, *_EXECUTION_FIELDS):
        old_attempt_transition[name] = source_receipt[name]
    old_attempt_transition["coordination"] = source_receipt["coordination"]
    with pytest.raises(
        TaskSourceConflictError,
        match="(?:authority is invalid|differs from durable queue authority)",
    ):
        adapter.recover_post_merge_retry(
            task_cid=_TASK_CID,
            expected_revision=_TERMINAL_REVISION,
            expected_control_receipt=terminal_receipt,
            status="retrying",
            receipt=old_attempt_transition,
            delay_ms=0,
            reason=fixture["queue_reason"],
            exact_retry_not_before_ms=20_000,
        )


def test_stale_blocked_snapshot_cannot_overwrite_n_plus_one_suffix() -> None:
    fixture = _generation_nine_fixture()
    stale = TaskRecord(
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        goal_cid="goal:generation-nine",
        ordinal=2,
        status="blocked",
        revision=_TERMINAL_REVISION,
        body=fixture["terminal_body"],
    )
    history = [
        {
            "revision": revision,
            "status": "ready",
            "body": {"fixture": revision},
        }
        for revision in range(1, _SOURCE_REVISION)
    ]
    history.extend(copy.deepcopy(fixture["rows"]))
    history.append(
        {
            "revision": _TERMINAL_REVISION + 1,
            "status": "retrying",
            "body": {
                **fixture["terminal_body"],
                "completion_receipt": fixture["transition"],
            },
        }
    )
    owner_calls: list[int] = []

    class _StaleOwner:
        def recover_post_merge_retry(self, **kwargs: Any) -> Any:
            owner_calls.append(int(kwargs["expected_task_revision"]))
            return SimpleNamespace(
                accepted=False,
                result={"error": "stale blocked head; canonical revision is N+1"},
            )

    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._client = _StaleOwner()
    adapter._clock_ms = lambda: 20_000
    adapter.get = lambda _task_cid: stale
    adapter.task_revision_history_projection = lambda _task_cid: {"revisions": history}

    with pytest.raises(TaskSourceConflictError, match=r"canonical revision is N\+1"):
        adapter.recover_post_merge_retry(
            task_cid=_TASK_CID,
            expected_revision=_TERMINAL_REVISION,
            expected_control_receipt=fixture["terminal_body"]["completion_receipt"],
            status="retrying",
            receipt=fixture["transition"],
            delay_ms=0,
            reason=fixture["queue_reason"],
        )
    assert owner_calls == [_TERMINAL_REVISION]


def test_pre_worker_replay_evidence_requires_exact_phases_and_zero_effects() -> None:
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:pre-worker-verifier",
        claim_id="claim:pre-worker-verifier",
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        attempt_number=4,
        owner_session_id="session:pre-worker-verifier",
        fencing_token=4,
        fence_epoch=4,
        lease_id="lease:pre-worker-verifier",
        committed_phase="failed",
        status="failed",
        started_at_ms=10_000,
        finished_at_ms=10_003,
        revision=3,
        body={},
    )
    failed_body = {
        "reason": DATABASE_POST_MERGE_COMPLETION_CLAIM_VERIFIER_FAILURE_REASON,
        "portal_retryable_failure": False,
        "portal_terminal_failure": True,
        "deferred": False,
        "attempt_consumed": "unknown",
        "provider_dispatched": "unknown",
        "typed_deferral_slot_consumed": "unknown",
        "backoff_seconds": 0,
    }
    phases = [
        {
            "phase": "claimed",
            "revision": 1,
            "body": {},
            "fencing_token": 4,
            "fence_epoch": 4,
            "committed_at_ms": 10_001,
        },
        {
            "phase": "context",
            "revision": 2,
            "body": {"resumed": True},
            "fencing_token": 4,
            "fence_epoch": 4,
            "committed_at_ms": 10_002,
        },
        {
            "phase": "failed",
            "revision": 3,
            "body": failed_body,
            "fencing_token": 4,
            "fence_epoch": 4,
            "committed_at_ms": 10_003,
        },
    ]
    provider = {
        "schema": DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA,
        "failure_kind": "provider_callback_outcome_unknown",
        "provider_effect_state": "unknown_may_have_started",
        "callback_state": "started_outcome_unknown",
        "idempotency_key": "provider:" + attempt.attempt_id,
        "task_cid": attempt.task_cid,
        "task_contract_digest": "sha256:" + "a" * 64,
        "repository_tree_id": "tree:pre-worker-verifier",
        "database_binding_id": "",
        "portal_failure_fingerprint": "",
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "owner_session_id": attempt.owner_session_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "callback_started_at_ms": 10_002,
    }
    provider["failure_fingerprint"] = _database_provider_callback_unknown_fingerprint(provider)

    class _CountResult:
        def __init__(self, count: int) -> None:
            self._count = count

        def fetchone(self) -> tuple[int]:
            return (self._count,)

    class _CountConnection:
        def __init__(self, count: int) -> None:
            self._count = count

        def execute(self, _sql: str, _parameters: Any) -> _CountResult:
            return _CountResult(self._count)

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.phase_history = lambda _attempt_id: copy.deepcopy(phases)
    daemon.provider_invocation_recorded = lambda _attempt_id, *, idempotency_key: copy.deepcopy(
        provider
    )
    daemon._uses_quack_command_gateway = lambda: False
    daemon._require_connection = lambda: _CountConnection(0)

    evidence = daemon._verified_post_merge_claim_verifier_pre_worker_failure(attempt)
    assert evidence["phase_revisions"] == [1, 2, 3]
    assert evidence["effect_count"] == 0
    assert evidence["evidence_id"]

    daemon.phase_history = lambda _attempt_id: [
        *copy.deepcopy(phases),
        {
            "phase": "effect",
            "revision": 4,
            "body": {},
            "fencing_token": 4,
            "fence_epoch": 4,
            "committed_at_ms": 10_004,
        },
    ]
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="did not fail before worker effects",
    ):
        daemon._verified_post_merge_claim_verifier_pre_worker_failure(attempt)

    daemon.phase_history = lambda _attempt_id: copy.deepcopy(phases)
    daemon._require_connection = lambda: _CountConnection(1)
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="observed a worker effect",
    ):
        daemon._verified_post_merge_claim_verifier_pre_worker_failure(attempt)


def test_empty_terminal_coordination_is_discovered_after_exact_sidecar_expiry() -> None:
    """Reproduce the generation-nine two-tick discovery path.

    The shared verifier terminal intentionally contains no coordination
    projection.  Discovery sees the exact sidecar claim after its deadline
    but before the coordinator has durably expired it.  The dedicated route
    may expire only that exact fence after proving the five-row lineage and
    pre-worker failure, then admits exactly this task.
    """

    fixture = _generation_nine_fixture()
    history_rows = [
        {
            "revision": revision,
            "status": "ready",
            "body": {"fixture_revision": revision},
        }
        for revision in range(1, _SOURCE_REVISION)
    ]
    history_rows.extend(copy.deepcopy(fixture["rows"]))
    projection_body = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1"),
        "task_cid": _TASK_CID,
        "revisions": history_rows,
    }
    history_projection = {
        **projection_body,
        "projection_cid": content_identity(projection_body),
    }
    task = TaskRecord(
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        goal_cid="goal:generation-nine",
        ordinal=2,
        status="blocked",
        revision=_TERMINAL_REVISION,
        body=fixture["terminal_body"],
    )
    identity = fixture["current_identity"]
    attempt = DatabaseTaskAttempt(
        attempt_id=identity["attempt_id"],
        claim_id=identity["claim_id"],
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        attempt_number=identity["attempt_number"],
        owner_session_id=identity["owner_session_id"],
        fencing_token=identity["fencing_token"],
        fence_epoch=identity["fence_epoch"],
        lease_id=identity["lease_id"],
        committed_phase="failed",
        status="failed",
        started_at_ms=10_000,
        finished_at_ms=fixture["rows"][-1]["body"]["completion_receipt"][
            "execution_finished_at_ms"
        ],
        revision=fixture["rows"][-1]["body"]["completion_receipt"]["execution_revision"],
        body={},
    )
    expires_at_ms = 15_000
    state = {
        "claim": "accepted",
        "attempt": "running",
        "lease": "accepted",
    }
    common = {
        "task_cid": _TASK_CID,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "owner_session_id": attempt.owner_session_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }

    class _Projection:
        def __init__(self, render: Any) -> None:
            self._render = render

        def to_dict(self) -> dict[str, Any]:
            return self._render()

        def __getattr__(self, name: str) -> Any:
            try:
                return self._render()[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    claim = _Projection(
        lambda: {
            **common,
            "claim_id": attempt.claim_id,
            "lease_id": attempt.lease_id,
            "state": state["claim"],
            "expires_at_ms": expires_at_ms,
            "revision": 2,
        }
    )
    coordination_attempt = _Projection(
        lambda: {
            **common,
            "status": state["attempt"],
            "revision": 2,
        }
    )
    lease = _Projection(
        lambda: {
            **common,
            "lease_id": attempt.lease_id,
            "lease_kind": "task",
            "scope_key": f"task:{_TASK_CID}",
            "scope": _TASK_CID,
            "mode": "exclusive",
            "claim_id": attempt.claim_id,
            "expires_at_ms": expires_at_ms,
            "state": state["lease"],
        }
    )

    def expire_claim(_claim: Any, *, now_ms: int) -> Any:
        assert now_ms > expires_at_ms
        state.update(claim="expired", attempt="expired", lease="expired")
        return lease

    coordinator = SimpleNamespace(
        get_task_claim=lambda _claim_id: claim,
        get_task_attempt=lambda _attempt_id: coordination_attempt,
        get_lease=lambda _lease_id: lease,
        get_prepared_task_completion=lambda _task_cid: None,
        get_task_claim_successor_projection=lambda _task_cid, **_kwargs: None,
        expire_task_claim=expire_claim,
    )
    task_source = SimpleNamespace(
        task_revision_history_projection=lambda _task_cid: copy.deepcopy(history_projection),
        list_tasks=lambda **_kwargs: SimpleNamespace(tasks=(task,)),
        get=lambda _task_cid: task,
    )
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._task_source = task_source
    daemon._coordinator = coordinator
    daemon.open = lambda: daemon
    daemon._automatic_claim_forbidden = lambda _task: False
    daemon.get_attempt = lambda _attempt_id: attempt
    daemon._latest_failed_attempts = lambda: []
    daemon._now_ms = lambda: expires_at_ms + 1
    daemon._failed_attempt_coordination_successor = lambda _attempt: None
    daemon._verified_post_merge_claim_verifier_pre_worker_failure = lambda _attempt: {
        "attempt_id": attempt.attempt_id,
        "phase_revisions": [1, 2, 3],
        "provider_failure_fingerprint": "sha256:" + "a" * 64,
        "effect_count": 0,
        "evidence_id": "sha256:" + "b" * 64,
    }

    # A lane that has no local execution attempt must skip the global-board
    # task cleanly instead of aborting every lane's recovery scan.
    daemon.get_attempt = lambda _attempt_id: None
    assert daemon._post_merge_completion_claim_verifier_replay_context(task) is None
    assert daemon.post_merge_completion_recovery_task_cids() == ()

    daemon.get_attempt = lambda _attempt_id: attempt
    context = daemon._post_merge_completion_claim_verifier_replay_context(task)
    assert context is not None
    assert context["current_receipt"]["coordination"] == {}
    assert context["pre_worker_evidence"]["effect_count"] == 0
    assert context["coordination_reconciliation"]["expired_now"] is True
    assert state == {
        "claim": "expired",
        "attempt": "expired",
        "lease": "expired",
    }
    assert daemon.post_merge_completion_recovery_task_cids() == (_TASK_CID,)


def test_replayed_retry_claim_advances_current_fence_and_carries_old_seed() -> None:
    """The post-recovery claim works even without either lane-local source row."""

    fixture = _generation_nine_fixture()
    replay = copy.deepcopy(fixture["transition"])
    for field in (
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    ):
        replay.pop(field, None)
    replay["queue_receipt"] = {"expected_queue_revision": 1}
    seed = replay["post_merge_completion_recovery_seed"]
    task = TaskRecord(
        task_cid=_TASK_CID,
        task_alias=_TASK_ALIAS,
        goal_cid="goal:generation-nine",
        ordinal=2,
        status="retrying",
        revision=_TERMINAL_REVISION + 1,
        body={"completion_receipt": replay},
    )
    target_identity = {
        "task_cid": _TASK_CID,
        "attempt_id": "attempt:generation-ten-consumer",
        "attempt_number": fixture["current_identity"]["attempt_number"] + 1,
        "owner_session_id": "session:generation-ten-consumer",
        "fencing_token": fixture["current_identity"]["fencing_token"] + 1,
        "fence_epoch": fixture["current_identity"]["fence_epoch"] + 1,
    }
    target_claim = {
        **target_identity,
        "claim_id": "claim:generation-ten-consumer",
        "lease_id": "lease:generation-ten-consumer",
    }
    observed: dict[str, Any] = {}

    class _Projection:
        def __init__(self, value: dict[str, Any]) -> None:
            self._value = value

        def to_dict(self) -> dict[str, Any]:
            return dict(self._value)

    class _Source:
        @staticmethod
        def get(_task_cid: str) -> TaskRecord:
            return task

        @staticmethod
        def compare_and_set_status(
            task_cid: str,
            *,
            expected_revision: int,
            status: str,
            receipt: dict[str, Any],
            evidence_digests: Any,
        ) -> dict[str, Any]:
            observed.update(
                task_cid=task_cid,
                expected_revision=expected_revision,
                status=status,
                receipt=dict(receipt),
                evidence_digests=evidence_digests,
            )
            return observed

    daemon = SimpleNamespace(
        task_source=_Source(),
        markdown_path=None,
        task_shard_count=1,
        task_shard_index=0,
        task_prefix="SPAR-",
        owner_session_id=target_identity["owner_session_id"],
        coordinator=SimpleNamespace(
            get_task_attempt=lambda _attempt_id: _Projection(target_identity),
            get_task_claim=lambda _claim_id: _Projection(target_claim),
        ),
        _database_virgin_transfer_lineage_for_transition=lambda _task: {},
        get_attempt=lambda _attempt_id: None,
    )
    daemon._retry_source_attempt_from_shared_seed = lambda **kwargs: (
        DatabaseImplementationDaemon._retry_source_attempt_from_shared_seed(
            daemon,
            **kwargs,
        )
    )

    def verify_recovery_state(source_attempt: DatabaseTaskAttempt, _task: Any) -> dict[str, Any]:
        assert source_attempt.attempt_id == fixture["current_identity"]["attempt_id"]
        assert source_attempt.attempt_number == fixture["current_identity"]["attempt_number"]
        return {"post_merge_completion_recovery_seed": seed}

    daemon._verified_post_merge_declared_output_recovery_state = verify_recovery_state
    claim_receipt = {
        "operation": "database_claim",
        "attempt_id": target_identity["attempt_id"],
        "claim_id": target_claim["claim_id"],
        "lease_id": target_claim["lease_id"],
        "owner_session_id": target_identity["owner_session_id"],
        "attempt_number": target_identity["attempt_number"],
        "fencing_token": target_identity["fencing_token"],
        "fence_epoch": target_identity["fence_epoch"],
    }

    result = DatabaseImplementationDaemon._cas_task_status_database(
        daemon,
        _TASK_CID,
        expected_revision=task.revision,
        new_status="in_progress",
        receipt=claim_receipt,
    )

    assert result is observed
    assert observed["status"] == "in_progress"
    assert observed["receipt"]["attempt_id"] == target_identity["attempt_id"]
    assert observed["receipt"]["attempt_number"] == target_identity["attempt_number"]
    assert (
        observed["receipt"]["post_merge_completion_recovery_source_attempt_id"]
        == seed["attempt_id"]
    )
    assert observed["receipt"]["post_merge_completion_recovery_seed"] == seed

    target_identity["attempt_number"] = fixture["current_identity"]["attempt_number"]
    target_claim["attempt_number"] = target_identity["attempt_number"]
    target_claim["fencing_token"] = fixture["current_identity"]["fencing_token"]
    target_claim["fence_epoch"] = fixture["current_identity"]["fence_epoch"] - 1
    with pytest.raises(
        DatabaseImplementationConflictError,
        match="replay target did not advance",
    ):
        DatabaseImplementationDaemon._cas_task_status_database(
            daemon,
            _TASK_CID,
            expected_revision=task.revision,
            new_status="in_progress",
            receipt=claim_receipt,
        )


@pytest.mark.parametrize(
    ("with_transfer", "queue_mode"),
    [
        (False, "valid"),
        (True, "valid"),
        (False, "missing"),
        (False, "foreign"),
        (False, "foreign_reason"),
        (False, "foreign_revision"),
    ],
)
def test_typed_owner_atomically_replays_exact_verifier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_transfer: bool,
    queue_mode: str,
) -> None:
    """The real owner advances the old cooldown and task head exactly once."""

    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        raising=False,
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        raising=False,
    )
    fixture = _generation_nine_fixture(with_transfer=with_transfer)
    database = tmp_path / "claim-verifier-replay.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:claim-verifier-replay",
            "plan_root_cid": "plan:claim-verifier-replay",
            "goals": [
                {
                    "goal_cid": "goal:claim-verifier-replay",
                    "goal_alias": "SPAR-G-CLAIM-VERIFIER-REPLAY",
                    "title": "Claim verifier replay",
                }
            ],
            "tasks": [
                {
                    "task_cid": _TASK_CID,
                    "task_id": _TASK_ALIAS,
                    "goal_cid": "goal:claim-verifier-replay",
                    "status": "ready",
                }
            ],
        }
    )
    source.close()

    # Reproduce the accepted generation-nine bytes while no state owner is
    # running. The compatibility repair itself is exercised only through the
    # typed owner below.
    old_recovery = fixture["rows"][1]["body"]["completion_receipt"]
    old_identity = fixture["old_identity"]
    queue_identity = dict(old_identity)
    if queue_mode == "foreign":
        queue_identity.update(
            {
                "attempt_id": "attempt:foreign-prior-cooldown",
                "attempt_number": 2,
                "claim_id": "claim:foreign-prior-cooldown",
                "lease_id": "lease:foreign-prior-cooldown",
                "owner_session_id": "session:foreign-prior-cooldown",
                "fencing_token": 2,
                "fence_epoch": 2,
            }
        )
    old_started_at_ms = 10_500
    old_extension = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": _TASK_CID,
        "expected_task_revision": _SOURCE_REVISION,
        **queue_identity,
        "delay_ms": 0,
        "started_at_ms": old_started_at_ms,
        "retry_not_before_ms": old_started_at_ms,
        "selection_penalty": 0,
        "consecutive_failures": queue_identity["attempt_number"],
        "reason": old_recovery["queue_reason"],
        "expected_queue_revision": -1,
        "expected_queue_attempt": 0,
    }
    if queue_mode == "foreign_reason":
        old_extension["reason"] = "database_post_merge_declared_outputs:foreign"
    if queue_mode == "foreign_revision":
        old_extension["expected_task_revision"] = _SOURCE_REVISION - 1
    old_resolution_cid = content_identity(
        {
            "typed_retry_cooldown": old_extension,
            "started_at_ms": old_started_at_ms,
        }
    )
    connection = open_duckdb_connection(database)
    try:
        connection.execute(
            "DELETE FROM task_revisions WHERE task_cid = ?",
            [_TASK_CID],
        )
        for revision in range(1, _SOURCE_REVISION):
            connection.execute(
                "INSERT INTO task_revisions("
                "task_cid, revision, status, body_json, recorded_at"
                ") VALUES (?, ?, ?, ?, ?)",
                [
                    _TASK_CID,
                    revision,
                    "ready",
                    canonical_json_bytes({"fixture_revision": revision}).decode("utf-8"),
                    "generation-nine-fixture",
                ],
            )
        for row in fixture["rows"]:
            connection.execute(
                "INSERT INTO task_revisions("
                "task_cid, revision, status, body_json, recorded_at"
                ") VALUES (?, ?, ?, ?, ?)",
                [
                    _TASK_CID,
                    row["revision"],
                    row["status"],
                    canonical_json_bytes(row["body"]).decode("utf-8"),
                    "generation-nine-fixture",
                ],
            )
        connection.execute(
            "UPDATE tasks SET status = ?, revision = ?, body_json = ?, "
            "updated_at = ? WHERE task_cid = ?",
            [
                "blocked",
                _TERMINAL_REVISION,
                canonical_json_bytes(fixture["terminal_body"]).decode("utf-8"),
                "generation-nine-fixture",
                _TASK_CID,
            ],
        )
        if queue_mode != "missing":
            connection.execute(
                "INSERT INTO leases("
                "task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision, extension_schema, "
                "extension_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    _TASK_CID,
                    queue_identity["claim_id"],
                    old_resolution_cid,
                    queue_identity["owner_session_id"],
                    queue_identity["fence_epoch"],
                    queue_identity["fencing_token"],
                    0,
                    queue_identity["attempt_number"],
                    "released",
                    old_started_at_ms,
                    old_extension["reason"],
                    old_started_at_ms,
                    queue_identity["owner_session_id"],
                    queue_identity["fence_epoch"],
                    1,
                    TYPED_RETRY_COOLDOWN_SCHEMA,
                    canonical_json_bytes(old_extension).decode("utf-8"),
                ],
            )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    server = build_server(
        database_path=database,
        state_dir=tmp_path / "claim-verifier-replay-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="claim-verifier-replay-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
    )
    owner = server.start()
    client_id = "database-implementation-daemon:claim-verifier-replay"
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=owner.process_birth_id,
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
        store_id=owner.store_id,
        process_birth_id=owner.process_birth_id,
    )
    adapter: TypedDatabaseTaskSource | None = None
    try:
        client.attach(owner.listen_uri, server_id=owner.server_id)
        adapter = TypedDatabaseTaskSource(
            client,
            clock_ms=lambda: 20_000,
        )
        blocked = adapter.get(_TASK_CID)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.revision == _TERMINAL_REVISION
        generation_before_denials = client.load_generation()
        cooldown_before_denials = adapter._retry_cooldown_row(_TASK_CID)
        if queue_mode == "missing":
            assert cooldown_before_denials is None
        else:
            assert cooldown_before_denials is not None
            assert cooldown_before_denials["attempt"] == queue_identity["attempt_number"]

        # No requested receipt shape may bypass the history-bound owner. In
        # particular, ``None`` preserves the prior terminal receipt and used
        # to evade a destination-receipt-only guard.
        for generic_receipt in (
            None,
            {"operation": "forged_generic_reopen"},
            fixture["transition"],
        ):
            with pytest.raises((TaskSourceConflictError, TransactionError)):
                adapter.compare_and_set_status(
                    blocked,
                    blocked.revision,
                    "retrying",
                    generic_receipt,
                )
        unchanged = adapter.get(_TASK_CID)
        assert unchanged is not None
        assert unchanged.status == "blocked"
        assert unchanged.revision == _TERMINAL_REVISION
        assert client.load_generation().revision == (generation_before_denials.revision)
        assert adapter._retry_cooldown_row(_TASK_CID) == (cooldown_before_denials)

        if queue_mode != "valid":
            with pytest.raises((TaskSourceConflictError, TransactionError)):
                adapter.recover_post_merge_retry(
                    task_cid=_TASK_CID,
                    expected_revision=_TERMINAL_REVISION,
                    expected_control_receipt=fixture["terminal_body"]["completion_receipt"],
                    status="retrying",
                    receipt=fixture["transition"],
                    delay_ms=0,
                    reason=fixture["queue_reason"],
                )
            denied = adapter.get(_TASK_CID)
            assert denied is not None
            assert denied.status == "blocked"
            assert denied.revision == _TERMINAL_REVISION
            assert client.load_generation().revision == (generation_before_denials.revision)
            return

        source_seed = fixture["rows"][1]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        for reused_field in (
            "qualified_target_commit",
            "qualification_receipt_id",
            "recovery_evidence_id",
        ):
            stale = copy.deepcopy(fixture["transition"])
            stale_seed = stale["post_merge_completion_recovery_seed"]
            stale_seed[reused_field] = source_seed[reused_field]
            stale_seed["seed_id"] = _seed_id(stale_seed)
            if reused_field == "qualified_target_commit":
                stale["qualified_target_commit"] = source_seed[reused_field]
            elif reused_field == "qualification_receipt_id":
                stale["callback_requalification_receipt_id"] = source_seed[reused_field]
                stale["queue_reason"] = (
                    "database_post_merge_declared_outputs_"
                    "callback_integration:"
                    + stale_seed["request_id"]
                    + ":"
                    + stale_seed["qualification_receipt_id"]
                )
            else:
                stale["callback_reconciliation_evidence_id"] = source_seed[reused_field]
            with pytest.raises((TaskSourceConflictError, TransactionError)):
                adapter.recover_post_merge_retry(
                    task_cid=_TASK_CID,
                    expected_revision=_TERMINAL_REVISION,
                    expected_control_receipt=fixture["terminal_body"]["completion_receipt"],
                    status="retrying",
                    receipt=stale,
                    delay_ms=0,
                    reason=stale["queue_reason"],
                )
            assert client.load_generation().revision == (generation_before_denials.revision)

        first = adapter.recover_post_merge_retry(
            task_cid=_TASK_CID,
            expected_revision=_TERMINAL_REVISION,
            expected_control_receipt=fixture["terminal_body"]["completion_receipt"],
            status="retrying",
            receipt=fixture["transition"],
            delay_ms=0,
            reason=fixture["queue_reason"],
        )
        assert first["cas_result"].changed is True
        assert first["queue_reused"] is True
        assert first["queue_receipt"]["expected_queue_revision"] == 1
        durable = adapter.get(_TASK_CID)
        assert durable is not None
        assert durable.status == "retrying"
        assert durable.revision == _TERMINAL_REVISION + 1
        assert {
            key: value for key, value in durable.body.items() if key != "completion_receipt"
        } == {
            key: value
            for key, value in fixture["terminal_body"].items()
            if key != "completion_receipt"
        }
        assert durable.body["completion_receipt"] == first["transition_receipt"]
        cooldown = adapter._retry_cooldown_row(_TASK_CID)
        assert cooldown is not None
        assert cooldown["attempt"] == fixture["current_identity"]["attempt_number"]
        assert cooldown["revision"] == 2
        generation = client.load_generation()

        replay = adapter.recover_post_merge_retry(
            task_cid=_TASK_CID,
            expected_revision=_TERMINAL_REVISION,
            expected_control_receipt=fixture["terminal_body"]["completion_receipt"],
            status="retrying",
            receipt=fixture["transition"],
            delay_ms=0,
            reason=fixture["queue_reason"],
        )
        assert replay["cas_result"].changed is False
        assert replay["transition_receipt"] == first["transition_receipt"]
        assert client.load_generation().revision == generation.revision
    finally:
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()
