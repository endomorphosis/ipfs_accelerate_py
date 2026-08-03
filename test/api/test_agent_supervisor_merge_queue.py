from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    LEGACY_POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA,
    POST_MERGE_CORRECTION_CONSUMPTION_SCHEMA,
    POST_MERGE_CORRECTION_FAILURE_SCHEMA,
    POST_MERGE_CORRECTION_LEGACY_FAILURE_ANCHOR_SCHEMA,
    POST_MERGE_CORRECTION_LEGACY_HIGH_WATER_ANCHOR_SCHEMA,
    POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA,
    POST_MERGE_REVIEW_DENIAL_CONSUMPTION_SCHEMA,
    POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA,
    MergeQueue,
    MergeQueueFenceError,
    MergeQueueFullError,
    MergeQueueIntegrityError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _enqueue(
    queue: MergeQueue,
    ordinal: int,
    *,
    priority: str = "P2",
    worktree_bytes: int = 0,
):
    metadata = {"worktree_bytes": worktree_bytes} if worktree_bytes else {}
    return queue.enqueue(
        branch_name=f"candidate/{ordinal}",
        task_id=f"TASK-{ordinal}",
        canonical_task_id=f"canonical-task-{ordinal}",
        commit_sha=f"{ordinal + 1:040x}",
        priority=priority,
        metadata=metadata,
    )


def _post_merge_denial_record(
    *,
    repository_id: str,
    target_branch: str,
) -> dict[str, object]:
    finding_material = {
        "source_ordinal": 1,
        "code": "missing-fence",
        "severity": "high",
        "summary": "Bind the correction to the exact reviewed candidate.",
    }
    material: dict[str, object] = {
        "schema": POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA,
        "target_repository_id": repository_id,
        "target_branch": target_branch,
        "task_id": "UIR-002",
        "canonical_task_key": "task/v1/example",
        "canonical_task_cid": "baguqeeraexample",
        "board_namespace": "uiir-v1",
        "task_binding_id": "baguqeerataskbinding",
        "review_attempt": 1,
        "implementation_attempt": 1,
        "target_implementation_attempt": 2,
        "implementation_commit": "1" * 40,
        "merge_commit": "2" * 40,
        "repository_tree_id": f"git-tree:{'3' * 40}",
        "review_receipt_id": "baguqeerareceipt",
        "review_request_id": "baguqeerarequest",
        "review_response_id": "baguqeeraresponse",
        "diff_binding_id": "baguqeeradiff",
        "implementer_provenance_id": "baguqeeraprovenance",
        "correction_origin_stream_id": "event-log:sha256:origin",
        "source_event_id": f"sha256:{'7' * 64}",
        "source_event_sequence": 41,
        "correction_authorized": True,
        "decision": "changes_required",
        "source_finding_count": 1,
        "included_finding_count": 1,
        "truncated": False,
        "findings": [
            {
                **finding_material,
                "finding_id": content_identity(finding_material),
            }
        ],
        "repository_write_authorized": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    terminal_material = {
        "target_repository_id": repository_id,
        "target_branch": target_branch,
        "task_id": material["task_id"],
        "canonical_task_key": material["canonical_task_key"],
        "canonical_task_cid": material["canonical_task_cid"],
        "task_binding_id": material["task_binding_id"],
        "implementation_commit": material["implementation_commit"],
    }
    material["terminal_key_id"] = content_identity(terminal_material)
    return {
        **material,
        "denial_id": content_identity(material),
    }


def _evolved_post_merge_denial_record(
    record: dict[str, object],
    *,
    marker: str,
    correction_authorized: bool,
) -> dict[str, object]:
    evolved = dict(record)
    digit = "4" if marker == "a" else "5"
    evolved.update(
        {
            "review_attempt": 2,
            "merge_commit": digit * 40,
            "repository_tree_id": f"git-tree:{digit * 40}",
            "review_receipt_id": f"baguqeerareceipt{marker}",
            "review_request_id": f"baguqeerarequest{marker}",
            "review_response_id": f"baguqeeraresponse{marker}",
            "diff_binding_id": f"baguqeeradiff{marker}",
            "implementer_provenance_id": (
                f"baguqeeraprovenance{marker}"
            ),
            "correction_origin_stream_id": (
                f"event-log:sha256:origin-{marker}"
            ),
            "correction_authorized": correction_authorized,
        }
    )
    evolved.pop("denial_id")
    evolved["denial_id"] = content_identity(evolved)
    return evolved


def _correction_common(
    denial: dict[str, object],
    *,
    attempt: int,
) -> dict[str, object]:
    return {
        "denial_id": denial["denial_id"],
        "target_repository_id": denial["target_repository_id"],
        "target_branch": denial["target_branch"],
        "task_id": denial["task_id"],
        "canonical_task_key": denial["canonical_task_key"],
        "canonical_task_cid": denial["canonical_task_cid"],
        "board_namespace": denial["board_namespace"],
        "task_binding_id": denial["task_binding_id"],
        "attempt": attempt,
        "origin_stream_id": denial["correction_origin_stream_id"],
    }


def _correction_consumption(
    denial: dict[str, object],
    *,
    attempt: int,
    authority_kind: str,
    authority_id: str,
    sequence: int,
) -> dict[str, object]:
    return {
        "schema": POST_MERGE_CORRECTION_CONSUMPTION_SCHEMA,
        **_correction_common(denial, attempt=attempt),
        "authority_kind": authority_kind,
        "authority_id": authority_id,
        "started_event_id": f"event-started-{sequence}",
        "started_event_sequence": sequence,
    }


def _correction_failure(
    denial: dict[str, object],
    *,
    attempt: int,
    authority_kind: str,
    authority_id: str,
    sequence: int,
    failure_kind: str = "implementation",
) -> dict[str, object]:
    return {
        "schema": POST_MERGE_CORRECTION_FAILURE_SCHEMA,
        **_correction_common(denial, attempt=attempt),
        "authority_kind": authority_kind,
        "authority_id": authority_id,
        "terminal_event_id": f"event-terminal-{sequence}",
        "terminal_event_sequence": sequence,
        "failure_kind": failure_kind,
    }


def _correction_grant(
    denial: dict[str, object],
    *,
    attempt: int,
    failure_record: dict[str, object],
    sequence: int,
    grant_id: str,
    recovery_seed: dict[str, str] | None = None,
) -> dict[str, object]:
    failure_detail = failure_record["detail"]
    assert isinstance(failure_detail, dict)
    return {
        "schema": POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA,
        **_correction_common(denial, attempt=attempt),
        "grant_id": grant_id,
        "grant_event_id": f"event-grant-{sequence}",
        "grant_event_sequence": sequence,
        "failure_record_id": failure_record["record_id"],
        "failure_event_id": failure_detail["terminal_event_id"],
        "failure_event_sequence": failure_detail[
            "terminal_event_sequence"
        ],
        "failure_kind": failure_detail["failure_kind"],
        "repair_task_id": f"REPAIR-{attempt}",
        "repair_task_binding_id": f"repair-task-binding-{attempt}",
        "repair_binding_id": f"repair-binding-{attempt}",
        **(
            recovery_seed
            or {
                "recovery_seed_ref": "",
                "recovery_seed_tree_id": "",
                "recovery_seed_submodule_path": "",
                "recovery_seed_submodule_commit": "",
            }
        ),
    }


def _legacy_correction_failure_anchor(
    denial: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": POST_MERGE_CORRECTION_LEGACY_FAILURE_ANCHOR_SCHEMA,
        **_correction_common(denial, attempt=3),
        "authority_kind": "review_denial",
        "authority_id": denial["denial_id"],
        "correction_attempt": 2,
        "correction_started_event_id": "legacy-start-attempt-2",
        "correction_started_event_sequence": 1993,
        "correction_terminal_event_id": "legacy-terminal-attempt-2",
        "correction_terminal_event_sequence": 2024,
        "superseding_started_event_id": "legacy-start-attempt-3",
        "superseding_started_event_sequence": 2078,
        "terminal_event_id": "legacy-terminal-attempt-3",
        "terminal_event_sequence": 2109,
        "failure_kind": "implementation",
        "migration_reason": "legacy_untyped_retry_lineage",
        "recovery_seed_ref": "",
        "recovery_seed_tree_id": "",
        "recovery_seed_submodule_path": "ipfs_datasets_py",
        "recovery_seed_submodule_commit": "8" * 40,
    }


def _legacy_correction_high_water_anchor(
    denial: dict[str, object],
    consumption: dict[str, object],
) -> dict[str, object]:
    attempt_events = [
        {
            "attempt": 2,
            "started_event_id": f"sha256:{'8' * 64}",
            "started_event_sequence": 50,
            "terminal_event_id": consumption["consuming_event_id"],
            "terminal_event_sequence": consumption[
                "consuming_event_sequence"
            ],
            "terminal_event_type": consumption["consuming_event_type"],
        },
        {
            "attempt": 3,
            "started_event_id": f"sha256:{'9' * 64}",
            "started_event_sequence": 110,
            "terminal_event_id": f"sha256:{'a' * 64}",
            "terminal_event_sequence": 120,
            "terminal_event_type": "implementation_finished",
        },
        {
            "attempt": 4,
            "started_event_id": f"sha256:{'b' * 64}",
            "started_event_sequence": 130,
            "terminal_event_id": f"sha256:{'c' * 64}",
            "terminal_event_sequence": 140,
            "terminal_event_type": "implementation_state_recovered",
        },
    ]
    return {
        "schema": (
            POST_MERGE_CORRECTION_LEGACY_HIGH_WATER_ANCHOR_SCHEMA
        ),
        **_correction_common(denial, attempt=4),
        "authority_kind": "review_denial",
        "authority_id": denial["denial_id"],
        "legacy_denial_consumption_id": consumption["consumption_id"],
        "first_correction_attempt": 2,
        "attempt_events": attempt_events,
        "terminal_event_id": attempt_events[-1]["terminal_event_id"],
        "terminal_event_sequence": attempt_events[-1][
            "terminal_event_sequence"
        ],
        "failure_kind": "implementation",
        "migration_reason": "legacy_untyped_retry_high_water",
        "recovery_seed_ref": "d" * 40,
        "recovery_seed_tree_id": f"git-tree:{'e' * 40}",
        "recovery_seed_submodule_path": "ipfs_datasets_py",
        "recovery_seed_submodule_commit": "f" * 40,
    }


def test_post_merge_denial_registry_is_permanent_idempotent_and_restart_safe(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'a' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    record = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )

    assert queue.record_post_merge_review_denial(record) == record
    assert queue.record_post_merge_review_denial(record) == record
    restarted = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )

    assert restarted.verified_post_merge_review_denials() == (record,)
    with restarted._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM post_merge_review_denials"
        ).fetchone()
    assert count is not None and int(count["count"]) == 1


def test_post_merge_denial_registry_coalesces_evolved_target_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'b' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    record = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(record)
    terminal_only = _evolved_post_merge_denial_record(
        record,
        marker="a",
        correction_authorized=False,
    )

    assert queue.record_post_merge_review_denial(terminal_only) == record

    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE post_merge_review_denials
               SET record_json='{"tampered":true}'
               WHERE terminal_key_id=?""",
            (record["terminal_key_id"],),
        )
        connection.commit()
    with pytest.raises(MergeQueueIntegrityError, match="schema fields"):
        queue.verified_post_merge_review_denials()


@pytest.mark.parametrize("origin_first", (False, True))
def test_post_merge_denial_registry_authorized_origin_wins_in_both_orders(
    tmp_path: Path,
    origin_first: bool,
) -> None:
    repository_id = f"repository:sha256:{'d' * 64}"
    queue = MergeQueue(
        tmp_path / f"queue-{int(origin_first)}",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    origin = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumer = _evolved_post_merge_denial_record(
        origin,
        marker="a",
        correction_authorized=False,
    )

    for candidate in (
        (origin, consumer)
        if origin_first
        else (consumer, origin)
    ):
        queue.record_post_merge_review_denial(candidate)

    assert queue.verified_post_merge_review_denials() == (origin,)


def test_post_merge_denial_correction_authority_promotes_monotonically(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'c' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    authorized = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    terminal_only = dict(authorized)
    terminal_only["correction_authorized"] = False
    terminal_only.pop("denial_id")
    terminal_only["denial_id"] = content_identity(terminal_only)

    assert queue.record_post_merge_review_denial(terminal_only) == terminal_only
    assert queue.record_post_merge_review_denial(authorized) == authorized
    assert queue.record_post_merge_review_denial(terminal_only) == authorized
    assert queue.verified_post_merge_review_denials() == (authorized,)


def test_post_merge_correction_registry_is_one_shot_and_restart_safe(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'1' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    authority = queue.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert authority["authority_available"] is True
    assert authority["authority_kind"] == "review_denial"
    assert authority["authorized_attempt"] == 2

    consumption = _correction_consumption(
        denial,
        attempt=2,
        authority_kind="review_denial",
        authority_id=str(denial["denial_id"]),
        sequence=10,
    )
    consumed = queue.record_post_merge_correction_consumption(
        consumption,
        expected_parent_record_id=str(denial["denial_id"]),
    )
    assert queue.record_post_merge_correction_consumption(
        consumption,
        expected_parent_record_id=str(denial["denial_id"]),
    ) == consumed
    conflicting = dict(consumption)
    conflicting["started_event_id"] = "different-start"
    with pytest.raises(MergeQueueFenceError, match="consumed"):
        queue.record_post_merge_correction_consumption(
            conflicting,
            expected_parent_record_id=str(denial["denial_id"]),
        )

    failure = queue.record_post_merge_correction_failure(
        _correction_failure(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=20,
        ),
        expected_parent_record_id=str(consumed["record_id"]),
    )
    grant = queue.record_post_merge_correction_repair_grant(
        _correction_grant(
            denial,
            attempt=3,
            failure_record=failure,
            sequence=30,
            grant_id="grant-attempt-3",
        ),
        expected_parent_record_id=str(failure["record_id"]),
    )

    restarted = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    assert restarted.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == (consumed, failure, grant)
    authority = restarted.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert authority["authority_available"] is True
    assert authority["authority_kind"] == "repair_grant"
    assert authority["authority_id"] == "grant-attempt-3"
    assert authority["authorized_attempt"] == 3


def test_post_merge_correction_failed_grant_can_receive_next_repair(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'2' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    denial_consumed = queue.record_post_merge_correction_consumption(
        _correction_consumption(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=10,
        ),
        expected_parent_record_id=str(denial["denial_id"]),
    )
    first_failure = queue.record_post_merge_correction_failure(
        _correction_failure(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=20,
        ),
        expected_parent_record_id=str(denial_consumed["record_id"]),
    )
    first_grant = queue.record_post_merge_correction_repair_grant(
        _correction_grant(
            denial,
            attempt=3,
            failure_record=first_failure,
            sequence=30,
            grant_id="grant-attempt-3",
        ),
        expected_parent_record_id=str(first_failure["record_id"]),
    )
    grant_consumed = queue.record_post_merge_correction_consumption(
        _correction_consumption(
            denial,
            attempt=3,
            authority_kind="repair_grant",
            authority_id="grant-attempt-3",
            sequence=40,
        ),
        expected_parent_record_id=str(first_grant["record_id"]),
    )
    second_failure = queue.record_post_merge_correction_failure(
        _correction_failure(
            denial,
            attempt=3,
            authority_kind="repair_grant",
            authority_id="grant-attempt-3",
            sequence=50,
        ),
        expected_parent_record_id=str(grant_consumed["record_id"]),
    )
    second_grant = queue.record_post_merge_correction_repair_grant(
        _correction_grant(
            denial,
            attempt=4,
            failure_record=second_failure,
            sequence=60,
            grant_id="grant-attempt-4",
        ),
        expected_parent_record_id=str(second_failure["record_id"]),
    )

    authority = queue.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert authority["authority_available"] is True
    assert authority["authority_id"] == "grant-attempt-4"
    assert authority["authorized_attempt"] == 4
    assert queue.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    )[-1] == second_grant


def test_post_merge_correction_registry_rejects_gaps_and_identity_drift(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'3' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    root_failure = _correction_failure(
        denial,
        attempt=2,
        authority_kind="review_denial",
        authority_id=str(denial["denial_id"]),
        sequence=20,
    )
    with pytest.raises(
        MergeQueueIntegrityError,
        match="root lacks one-shot authority",
    ):
        queue.record_post_merge_correction_failure(
            root_failure,
            expected_parent_record_id=str(denial["denial_id"]),
        )
    drifted = _correction_consumption(
        denial,
        attempt=2,
        authority_kind="review_denial",
        authority_id=str(denial["denial_id"]),
        sequence=10,
    )
    drifted["board_namespace"] = "foreign-board"
    with pytest.raises(MergeQueueFenceError, match="identity differs"):
        queue.record_post_merge_correction_consumption(
            drifted,
            expected_parent_record_id=str(denial["denial_id"]),
        )
    consumed = queue.record_post_merge_correction_consumption(
        _correction_consumption(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=10,
        ),
        expected_parent_record_id=str(denial["denial_id"]),
    )
    evolved = _evolved_post_merge_denial_record(
        denial,
        marker="a",
        correction_authorized=True,
    )
    evolved.pop("denial_id")
    evolved["denial_id"] = content_identity(evolved)
    with pytest.raises(
        MergeQueueIntegrityError,
        match="multiple authorized origin streams",
    ):
        queue.record_post_merge_review_denial(evolved)
    assert queue.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == (consumed,)


def test_post_merge_correction_cas_allows_only_one_concurrent_child(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'4' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    candidates = [
        _correction_consumption(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=sequence,
        )
        for sequence in (10, 11)
    ]

    def append(value: dict[str, object]) -> str:
        try:
            result = queue.record_post_merge_correction_consumption(
                value,
                expected_parent_record_id=str(denial["denial_id"]),
            )
            return str(result["record_id"])
        except MergeQueueFenceError:
            return "fenced"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(append, candidates))

    assert outcomes.count("fenced") == 1
    assert len(queue.verified_post_merge_correction_chain()) == 1


def test_post_merge_correction_registry_migrates_legacy_denial_head_once(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'5' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        # Model a genuine pre-registry database. A one-time migration may
        # create a pristine head only when the durable deployment marker is
        # absent.
        connection.execute(
            """DELETE FROM agent_supervisor_store_metadata
               WHERE key='post_merge_correction_registry:migrated'"""
        )
        connection.execute(
            """DELETE FROM post_merge_correction_chain_heads
               WHERE denial_id=?""",
            (denial["denial_id"],),
        )
        connection.commit()
    migrated = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    assert migrated.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )["authority_available"] is True


def test_post_merge_correction_registry_rejects_full_consumed_rollback(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'5' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)

    queue.record_post_merge_correction_consumption(
        _correction_consumption(
            denial,
            attempt=2,
            authority_kind="review_denial",
            authority_id=str(denial["denial_id"]),
            sequence=10,
        ),
        expected_parent_record_id=str(denial["denial_id"]),
    )
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """DELETE FROM post_merge_correction_chain_records
               WHERE denial_id=?""",
            (denial["denial_id"],),
        )
        connection.execute(
            """DELETE FROM post_merge_correction_chain_heads
               WHERE denial_id=?""",
            (denial["denial_id"],),
        )
        connection.commit()
    with pytest.raises(
        MergeQueueIntegrityError,
        match="head coverage changed",
    ):
        MergeQueue(
            queue_path,
            target_repository_id=repository_id,
            target_branch="agent/uiir",
            require_target_binding=True,
        )


def test_post_merge_correction_legacy_anchor_binds_only_seeded_next_attempt(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'6' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    anchor_value = _legacy_correction_failure_anchor(denial)
    anchor = queue.record_post_merge_correction_legacy_failure_anchor(
        anchor_value,
        expected_parent_record_id=str(denial["denial_id"]),
    )
    assert queue.record_post_merge_correction_legacy_failure_anchor(
        anchor_value,
        expected_parent_record_id=str(denial["denial_id"]),
    ) == anchor
    state = queue.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert state["state"] == "legacy_failure_anchored"
    assert state["authority_available"] is False
    assert state["recovery_seed_ref"] == ""

    without_seed = _correction_grant(
        denial,
        attempt=4,
        failure_record=anchor,
        sequence=2200,
        grant_id="legacy-grant-attempt-4",
    )
    with pytest.raises(
        MergeQueueIntegrityError,
        match="legacy recovery seed is required",
    ):
        queue.record_post_merge_correction_repair_grant(
            without_seed,
            expected_parent_record_id=str(anchor["record_id"]),
        )
    seed = {
        "recovery_seed_ref": "6" * 40,
        "recovery_seed_tree_id": f"git-tree:{'7' * 40}",
        "recovery_seed_submodule_path": str(
            anchor_value["recovery_seed_submodule_path"]
        ),
        "recovery_seed_submodule_commit": str(
            anchor_value["recovery_seed_submodule_commit"]
        ),
    }
    wrong_postimage = dict(seed)
    wrong_postimage["recovery_seed_submodule_commit"] = "9" * 40
    with pytest.raises(
        MergeQueueIntegrityError,
        match="crosses failure identity",
    ):
        queue.record_post_merge_correction_repair_grant(
            _correction_grant(
                denial,
                attempt=4,
                failure_record=anchor,
                sequence=2200,
                grant_id="wrong-postimage-grant",
                recovery_seed=wrong_postimage,
            ),
            expected_parent_record_id=str(anchor["record_id"]),
        )
    grant = queue.record_post_merge_correction_repair_grant(
        _correction_grant(
            denial,
            attempt=4,
            failure_record=anchor,
            sequence=2200,
            grant_id="legacy-grant-attempt-4",
            recovery_seed=seed,
        ),
        expected_parent_record_id=str(anchor["record_id"]),
    )
    state = queue.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert state["authority_available"] is True
    assert state["authorized_attempt"] == 4
    assert {
        name: state[name] for name in seed
    } == seed

    consumed = queue.record_post_merge_correction_consumption(
        _correction_consumption(
            denial,
            attempt=4,
            authority_kind="repair_grant",
            authority_id="legacy-grant-attempt-4",
            sequence=2300,
        ),
        expected_parent_record_id=str(grant["record_id"]),
    )
    assert consumed["attempt"] == 4
    consumed_state = queue.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert consumed_state["authority_available"] is False
    assert consumed_state["recovery_seed_ref"] == ""


def test_post_merge_correction_legacy_high_water_is_restart_safe_and_exact_next(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'a' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(
        denial,
        consuming_event_sequence=100,
    )
    queue.record_post_merge_review_denial(denial)
    queue.record_post_merge_review_denial_consumption(consumption)
    anchor_value = _legacy_correction_high_water_anchor(
        denial,
        consumption,
    )
    anchor = queue.record_post_merge_correction_legacy_high_water_anchor(
        anchor_value,
        expected_parent_record_id=str(denial["denial_id"]),
    )
    assert queue.record_post_merge_correction_legacy_high_water_anchor(
        anchor_value,
        expected_parent_record_id=str(denial["denial_id"]),
    ) == anchor
    assert anchor["detail"]["legacy_denial_consumption_id"] == (
        consumption["consumption_id"]
    )
    divergent = json.loads(json.dumps(anchor_value))
    divergent["attempt_events"][1]["terminal_event_id"] = (
        f"sha256:{'2' * 64}"
    )
    with pytest.raises(MergeQueueFenceError, match="was consumed"):
        queue.record_post_merge_correction_legacy_high_water_anchor(
            divergent,
            expected_parent_record_id=str(denial["denial_id"]),
        )

    restarted = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    assert restarted.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == (anchor,)
    state = restarted.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )
    assert state["state"] == "legacy_high_water_anchored"
    assert state["authority_available"] is False
    assert state["authorized_attempt"] == 4

    seed = {
        name: str(anchor_value[name])
        for name in (
            "recovery_seed_ref",
            "recovery_seed_tree_id",
            "recovery_seed_submodule_path",
            "recovery_seed_submodule_commit",
        )
    }
    with pytest.raises(
        MergeQueueIntegrityError,
        match="crosses failure identity",
    ):
        restarted.record_post_merge_correction_repair_grant(
            _correction_grant(
                denial,
                attempt=4,
                failure_record=anchor,
                sequence=150,
                grant_id="stale-high-water-grant",
                recovery_seed=seed,
            ),
            expected_parent_record_id=str(anchor["record_id"]),
        )
    grant = restarted.record_post_merge_correction_repair_grant(
        _correction_grant(
            denial,
            attempt=5,
            failure_record=anchor,
            sequence=150,
            grant_id="high-water-grant-attempt-5",
            recovery_seed=seed,
        ),
        expected_parent_record_id=str(anchor["record_id"]),
    )
    assert grant["attempt"] == 5
    restarted_again = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    assert restarted_again.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == (anchor, grant)
    assert restarted_again.verified_post_merge_correction_authority(
        str(denial["denial_id"])
    )["authorized_attempt"] == 5


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_consumption_id", "schema fields changed"),
        ("unknown_consumption_id", "witness is unavailable"),
        ("first_terminal_tamper", "does not match its denial lineage"),
        ("start_before_tombstone", "does not match its denial lineage"),
    ),
)
def test_post_merge_correction_legacy_high_water_requires_exact_consumption(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repository_id = f"repository:sha256:{'b' * 64}"
    queue = MergeQueue(
        tmp_path / mutation,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(
        denial,
        consuming_event_sequence=100,
    )
    queue.record_post_merge_review_denial(denial)
    queue.record_post_merge_review_denial_consumption(consumption)
    value = _legacy_correction_high_water_anchor(denial, consumption)
    if mutation == "missing_consumption_id":
        value.pop("legacy_denial_consumption_id")
    elif mutation == "unknown_consumption_id":
        value["legacy_denial_consumption_id"] = "baguqeeramissing"
    elif mutation == "first_terminal_tamper":
        value["attempt_events"][0]["terminal_event_id"] = (
            f"sha256:{'1' * 64}"
        )
    else:
        value["attempt_events"][0]["started_event_sequence"] = int(
            denial["source_event_sequence"]
        )

    with pytest.raises(MergeQueueIntegrityError, match=message):
        queue.record_post_merge_correction_legacy_high_water_anchor(
            value,
            expected_parent_record_id=str(denial["denial_id"]),
        )
    assert queue.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("gap", "attempt history is invalid"),
        ("event_reuse", "attempt entry is invalid"),
        ("malformed_event_id", "attempt entry is invalid"),
        ("unresolved_root", "recovery seed binding is invalid"),
    ),
)
def test_post_merge_correction_legacy_high_water_rejects_malformed_history(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repository_id = f"repository:sha256:{'c' * 64}"
    queue = MergeQueue(
        tmp_path / mutation,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(
        denial,
        consuming_event_sequence=100,
    )
    queue.record_post_merge_review_denial(denial)
    queue.record_post_merge_review_denial_consumption(consumption)
    value = _legacy_correction_high_water_anchor(denial, consumption)
    if mutation == "gap":
        value["attempt_events"].pop(1)
    elif mutation == "event_reuse":
        value["attempt_events"][1]["started_event_id"] = value[
            "attempt_events"
        ][0]["terminal_event_id"]
    elif mutation == "malformed_event_id":
        value["attempt_events"][1]["terminal_event_id"] = (
            "legacy-terminal-id"
        )
    else:
        value["recovery_seed_ref"] = ""
        value["recovery_seed_tree_id"] = ""

    with pytest.raises(MergeQueueIntegrityError, match=message):
        queue.record_post_merge_correction_legacy_high_water_anchor(
            value,
            expected_parent_record_id=str(denial["denial_id"]),
        )
    assert queue.verified_post_merge_correction_chain(
        str(denial["denial_id"])
    ) == ()


@pytest.mark.parametrize(
    ("terminal_type", "failure_kind", "accepted"),
    (
        ("implementation_state_recovered", "implementation", True),
        ("implementation_state_recovered", "validation", False),
        ("post_merge_correction_queue_reconciled", "merge", True),
        ("post_merge_correction_queue_reconciled", "implementation", False),
        ("implementation_finished", "implementation", True),
        ("implementation_finished", "validation", True),
        ("implementation_finished", "merge", False),
    ),
)
def test_post_merge_correction_legacy_high_water_terminal_kind_coherence(
    tmp_path: Path,
    terminal_type: str,
    failure_kind: str,
    accepted: bool,
) -> None:
    repository_id = f"repository:sha256:{'d' * 64}"
    queue = MergeQueue(
        tmp_path / f"{terminal_type}-{failure_kind}",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(
        denial,
        consuming_event_sequence=100,
    )
    queue.record_post_merge_review_denial(denial)
    queue.record_post_merge_review_denial_consumption(consumption)
    value = _legacy_correction_high_water_anchor(denial, consumption)
    value["attempt_events"][-1]["terminal_event_type"] = terminal_type
    value["failure_kind"] = failure_kind
    if accepted:
        anchor = queue.record_post_merge_correction_legacy_high_water_anchor(
            value,
            expected_parent_record_id=str(denial["denial_id"]),
        )
        assert anchor["detail"]["failure_kind"] == failure_kind
    else:
        with pytest.raises(
            MergeQueueIntegrityError,
            match="terminal and failure kind disagree",
        ):
            queue.record_post_merge_correction_legacy_high_water_anchor(
                value,
                expected_parent_record_id=str(denial["denial_id"]),
            )


@pytest.mark.parametrize("tamper", ("delete", "rewrite"))
def test_post_merge_correction_legacy_high_water_revalidates_consumption_on_restart(
    tmp_path: Path,
    tamper: str,
) -> None:
    repository_id = f"repository:sha256:{'e' * 64}"
    queue_path = tmp_path / tamper
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(
        denial,
        consuming_event_sequence=100,
    )
    queue.record_post_merge_review_denial(denial)
    queue.record_post_merge_review_denial_consumption(consumption)
    queue.record_post_merge_correction_legacy_high_water_anchor(
        _legacy_correction_high_water_anchor(denial, consumption),
        expected_parent_record_id=str(denial["denial_id"]),
    )
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if tamper == "delete":
            connection.execute(
                """DELETE FROM post_merge_review_denial_consumptions
                   WHERE consumption_id=?""",
                (consumption["consumption_id"],),
            )
        else:
            connection.execute(
                """UPDATE post_merge_review_denial_consumptions
                   SET record_json='{}'
                   WHERE consumption_id=?""",
                (consumption["consumption_id"],),
            )
        connection.commit()

    with pytest.raises(MergeQueueIntegrityError):
        queue.verified_post_merge_correction_chain(
            str(denial["denial_id"])
        )
    with pytest.raises(MergeQueueIntegrityError):
        MergeQueue(
            queue_path,
            target_repository_id=repository_id,
            target_branch="agent/uiir",
            require_target_binding=True,
        )


def test_post_merge_correction_history_mirror_is_atomic_and_prefix_safe(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'7' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    queue.record_post_merge_review_denial(denial)
    consumption = _correction_consumption(
        denial,
        attempt=2,
        authority_kind="review_denial",
        authority_id=str(denial["denial_id"]),
        sequence=10,
    )
    first = queue.mirror_post_merge_correction_history([consumption])
    failure_value = _correction_failure(
        denial,
        attempt=2,
        authority_kind="review_denial",
        authority_id=str(denial["denial_id"]),
        sequence=20,
    )
    full = queue.mirror_post_merge_correction_history(
        [consumption, failure_value]
    )
    assert full[:1] == first
    assert queue.mirror_post_merge_correction_history(
        [consumption]
    ) == full

    conflicting = dict(consumption)
    conflicting["started_event_id"] = "conflicting-ledger-event"
    with pytest.raises(MergeQueueFenceError, match="conflicts"):
        queue.mirror_post_merge_correction_history([conflicting])

    second_queue = MergeQueue(
        tmp_path / "atomic",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    second_queue.record_post_merge_review_denial(denial)
    invalid_grant = {
        "schema": POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA,
        **_correction_common(denial, attempt=3),
        "grant_id": "gap-grant",
        "grant_event_id": "gap-grant-event",
        "grant_event_sequence": 30,
        "failure_record_id": "missing-failure-record",
        "failure_event_id": "missing-failure-event",
        "failure_event_sequence": 20,
        "failure_kind": "implementation",
        "repair_task_id": "REPAIR-3",
        "repair_task_binding_id": "repair-task-binding-3",
        "repair_binding_id": "repair-binding-3",
        "recovery_seed_ref": "",
        "recovery_seed_tree_id": "",
        "recovery_seed_submodule_path": "",
        "recovery_seed_submodule_commit": "",
    }
    with pytest.raises(
        MergeQueueIntegrityError,
        match="state transition",
    ):
        second_queue.mirror_post_merge_correction_history(
            [consumption, invalid_grant]
        )
    assert second_queue.verified_post_merge_correction_chain() == ()


def test_batch_claims_have_a_deterministic_total_order_and_unique_fences(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(
        tmp_path / "queue",
        clock=lambda: 100.0,
        max_processing=8,
        priority_aging_seconds=0,
    )
    low = _enqueue(queue, 0, priority="P3")
    high_b = _enqueue(queue, 1, priority="P0")
    high_a = _enqueue(queue, 2, priority="P0")
    medium = _enqueue(queue, 3, priority="P1")

    claimed = queue.dequeue_many(8, consumer_id="merge-train:deterministic")

    same_priority = sorted((high_a.request_id, high_b.request_id))
    assert [request.request_id for request in claimed] == [
        *same_priority,
        medium.request_id,
        low.request_id,
    ]
    assert all(request.consumer_id == "merge-train:deterministic" for request in claimed)
    assert all(request.claim_token for request in claimed)
    assert all(request.claim_generation == 1 for request in claimed)
    assert len({request.claim_token for request in claimed}) == len(claimed)


@pytest.mark.parametrize(
    "stale_request",
    (
        lambda claimed: replace(claimed, consumer_id="merge-train:impostor"),
        lambda claimed: replace(claimed, claim_token="stale-token"),
        lambda claimed: replace(
            claimed, claim_generation=max(0, claimed.claim_generation - 1)
        ),
    ),
    ids=("wrong-owner", "wrong-token", "stale-generation"),
)
def test_completion_requires_the_exact_current_claim_fence(
    tmp_path: Path, stale_request
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:owner")
    assert claimed is not None

    with pytest.raises(MergeQueueFenceError):
        queue.complete(stale_request(claimed))

    stored = queue.get(pending.request_id)
    assert stored is not None
    assert stored.status == "processing"
    assert stored.consumer_id == claimed.consumer_id
    assert stored.claim_token == claimed.claim_token
    queue.complete(claimed, metadata={"validated": True})
    assert queue.get(pending.request_id).status == "completed"  # type: ignore[union-attr]


def test_recovered_claim_increments_generation_and_fences_crashed_worker(
    tmp_path: Path,
) -> None:
    now = [10.0]
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_age_seconds=5,
        max_attempts=3,
    )
    pending = _enqueue(queue, 0)
    crashed_claim = queue.dequeue(consumer_id="worker:crashed")
    assert crashed_claim is not None

    now[0] = 20.0
    restarted = MergeQueue(
        queue_path,
        clock=lambda: now[0],
        max_age_seconds=5,
        max_attempts=3,
    )
    replacement = restarted.dequeue(consumer_id="worker:replacement")
    assert replacement is not None
    assert replacement.request_id == pending.request_id
    assert replacement.claim_generation > crashed_claim.claim_generation
    assert replacement.claim_token != crashed_claim.claim_token

    with pytest.raises(MergeQueueFenceError):
        restarted.complete(crashed_claim)
    assert restarted.get(pending.request_id).status == "processing"  # type: ignore[union-attr]

    restarted.complete(replacement)
    durable = MergeQueue(queue_path).get(pending.request_id)
    assert durable is not None
    assert durable.status == "completed"
    assert durable.claim_generation == replacement.claim_generation + 1
    assert durable.claim_token == ""


def test_capacity_merge_debt_and_worktree_disk_admission_are_bounded(
    tmp_path: Path,
) -> None:
    observed_worktree_bytes = [0]
    queue = MergeQueue(
        tmp_path / "queue",
        max_queue_size=3,
        max_processing=2,
        max_worktree_bytes=10,
        worktree_usage=lambda: observed_worktree_bytes[0],
    )
    requests = [_enqueue(queue, ordinal, worktree_bytes=6) for ordinal in range(3)]
    with pytest.raises(MergeQueueFullError):
        _enqueue(queue, 3, worktree_bytes=1)

    first_batch = queue.dequeue_many(3, consumer_id="merge-train:first")
    assert len(first_batch) == 1
    assert queue.dequeue_many(1, consumer_id="merge-train:blocked") == ()
    observed_worktree_bytes[0] = 10
    status = queue.status()
    assert status["merge_debt"] == 1
    assert status["max_processing"] == 2
    assert status["reserved_worktree_bytes"] == 6
    assert status["max_worktree_bytes"] == 10
    assert status["disk_backpressure"] is True
    assert status["backpressure"] is True

    queue.complete(first_batch[0])
    observed_worktree_bytes[0] = 0
    second = queue.dequeue(consumer_id="merge-train:second")
    assert second is not None
    assert second.request_id in {
        requests[1].request_id,
        requests[2].request_id,
    }
    assert queue.status()["reserved_worktree_bytes"] == 6


def test_merge_debt_stops_additional_claims_until_a_slot_is_released(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    for ordinal in range(4):
        _enqueue(queue, ordinal)

    claimed = queue.dequeue_many(4, consumer_id="merge-train:batch")

    assert len(claimed) == 2
    assert queue.dequeue(consumer_id="merge-train:other") is None
    status = queue.status()
    assert status["merge_debt"] == status["max_processing"] == 2
    assert status["backpressure"] is True

    queue.complete(claimed[0])
    replacement = queue.dequeue(consumer_id="merge-train:other")
    assert replacement is not None
    assert replacement.request_id not in {
        request.request_id for request in claimed
    }
    assert queue.status()["merge_debt"] == 2


def test_failed_validation_is_quarantined_with_a_durable_receipt(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:validator")
    assert claimed is not None

    receipt_path = queue.fail(
        claimed,
        reason="post-merge validation failed",
        metadata={"validation_receipt_id": "sha256:failed"},
    )

    assert receipt_path is not None
    assert receipt_path.parent == queue.quarantine_dir
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["request_id"] == pending.request_id
    assert payload["status"] == "quarantined"
    assert payload["failure_reason"] == "post-merge validation failed"
    assert payload["receipt_type"] == "merge_quarantine"
    assert payload["metadata"]["quarantine"] == {
        "validation_receipt_id": "sha256:failed"
    }

    restarted = MergeQueue(queue_path)
    stored = restarted.get(pending.request_id)
    assert stored is not None
    assert stored.status == "quarantined"
    assert restarted.dequeue(consumer_id="merge-train:restart") is None
    assert restarted.status()["quarantined"] == 1
    duplicate = restarted.enqueue(
        branch_name="candidate/duplicate",
        task_id="TASK-ALIAS",
        canonical_task_id=pending.canonical_task_id,
        commit_sha=pending.commit_sha,
    )
    assert duplicate.request_id == pending.request_id


def test_cancelled_work_is_fenced_and_survives_restart(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue"
    queue = MergeQueue(queue_path)
    pending = _enqueue(queue, 0)
    claimed = queue.dequeue(consumer_id="merge-train:obsolete-base")
    assert claimed is not None

    cancelled = queue.cancel(
        claimed,
        reason="base advanced while preflight was running",
        metadata={"replacement_base": "b" * 40},
    )

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    with pytest.raises(MergeQueueFenceError):
        queue.complete(claimed)
    restarted = MergeQueue(queue_path)
    durable = restarted.get(pending.request_id)
    assert durable is not None
    assert durable.status == "cancelled"
    assert durable.failure_reason == "base advanced while preflight was running"
    assert restarted.status()["cancelled"] == 1
    assert restarted.dequeue(consumer_id="merge-train:restart") is None


def test_bound_main_consumer_cannot_claim_benchmark_or_legacy_request(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "shared-queue"
    repository_id = f"repository:sha256:{'a' * 64}"
    legacy = MergeQueue(queue_path).enqueue(
        branch_name="implementation/legacy",
        task_id="LEGACY-001",
        canonical_task_id="canonical-legacy",
        commit_sha="1" * 40,
        priority="P0",
    )
    benchmark_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="benchmark/semantic-roundtrip",
        require_target_binding=True,
    )
    benchmark = benchmark_queue.enqueue(
        branch_name="implementation/benchmark",
        task_id="SRT-014",
        canonical_task_id="canonical-srt-014",
        commit_sha="2" * 40,
        priority="P0",
    )
    main_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="main",
        require_target_binding=True,
    )
    main = main_queue.enqueue(
        branch_name="implementation/main",
        task_id="MAIN-001",
        canonical_task_id="canonical-main",
        commit_sha="3" * 40,
        priority="P1",
    )

    claimed_by_main = main_queue.dequeue_many(
        3,
        consumer_id="merge-train:main",
    )

    assert [request.request_id for request in claimed_by_main] == [
        main.request_id
    ]
    assert main_queue.pending_count() == 0
    assert main_queue.processing_count() == 1
    assert main_queue.active_canonical_task_ids() == {"canonical-main"}
    assert main_queue.status()["target_branch"] == "main"
    assert benchmark_queue.get(benchmark.request_id).status == "pending"  # type: ignore[union-attr]
    assert benchmark_queue.get(benchmark.request_id).consumer_id == ""  # type: ignore[union-attr]
    assert benchmark_queue.get(legacy.request_id).status == "pending"  # type: ignore[union-attr]
    with pytest.raises(MergeQueueFenceError, match="target differs"):
        main_queue.cancel(benchmark.request_id)
    claimed_by_benchmark = benchmark_queue.dequeue(
        consumer_id="merge-train:benchmark"
    )
    assert claimed_by_benchmark is not None
    assert claimed_by_benchmark.request_id == benchmark.request_id
    assert main_queue.owns_claim(claimed_by_benchmark) is False
    with pytest.raises(MergeQueueFenceError, match="target differs"):
        main_queue.complete(claimed_by_benchmark)

    assert main_queue.recover_abandoned_train_claims() == 1
    main_after_recovery = main_queue.get(main.request_id)
    assert main_after_recovery is not None
    assert main_after_recovery.status == "pending"
    assert main_after_recovery.attempt == 2
    benchmark_after_recovery = benchmark_queue.get(benchmark.request_id)
    assert benchmark_after_recovery is not None
    assert benchmark_after_recovery.status == "processing"
    assert benchmark_after_recovery.attempt == 1
    assert benchmark_after_recovery.consumer_id == "merge-train:benchmark"


def test_case_distinct_git_targets_have_distinct_deduplication_keys(
    tmp_path: Path,
) -> None:
    queue_path = tmp_path / "shared-queue"
    repository_id = f"repository:sha256:{'b' * 64}"
    upper_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="Feature",
        require_target_binding=True,
    )
    lower_queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="feature",
        require_target_binding=True,
    )
    enqueue_kwargs = {
        "branch_name": "implementation/case-sensitive",
        "task_id": "CASE-001",
        "canonical_task_id": "canonical-case-001",
        "commit_sha": "4" * 40,
    }

    upper = upper_queue.enqueue(**enqueue_kwargs)
    lower = lower_queue.enqueue(**enqueue_kwargs)

    assert upper.request_id != lower.request_id
    assert upper.target_branch == "Feature"
    assert lower.target_branch == "feature"
    assert upper_queue.pending_count() == 1
    assert lower_queue.pending_count() == 1


def _post_merge_denial_consumption_record(
    denial: dict[str, object],
    *,
    consuming_event_id: str = f"sha256:{'6' * 64}",
    consuming_event_sequence: int = 42,
    consuming_event_type: str = "implementation_finished",
) -> dict[str, object]:
    material: dict[str, object] = {
        "schema": POST_MERGE_REVIEW_DENIAL_CONSUMPTION_SCHEMA,
        "terminal_key_id": denial["terminal_key_id"],
        "denial_id": denial["denial_id"],
        "target_repository_id": denial["target_repository_id"],
        "target_branch": denial["target_branch"],
        "task_id": denial["task_id"],
        "canonical_task_key": denial["canonical_task_key"],
        "canonical_task_cid": denial["canonical_task_cid"],
        "board_namespace": denial["board_namespace"],
        "task_binding_id": denial["task_binding_id"],
        "implementation_commit": denial["implementation_commit"],
        "implementation_attempt": denial["implementation_attempt"],
        "target_implementation_attempt": denial[
            "target_implementation_attempt"
        ],
        "correction_origin_stream_id": denial[
            "correction_origin_stream_id"
        ],
        "consuming_event_type": consuming_event_type,
        "consuming_event_id": consuming_event_id,
        "consuming_event_sequence": consuming_event_sequence,
        "consuming_implementation_attempt": denial[
            "target_implementation_attempt"
        ],
        "attempt_consumed": True,
        "repository_write_authorized": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    return {
        **material,
        "consumption_id": content_identity(material),
    }


def _legacy_post_merge_denial_record(
    record: dict[str, object],
) -> dict[str, object]:
    legacy = dict(record)
    legacy["schema"] = (
        LEGACY_POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
    )
    legacy.pop("source_event_id")
    legacy.pop("source_event_sequence")
    legacy.pop("denial_id")
    legacy["denial_id"] = content_identity(legacy)
    return legacy


def _insert_stored_post_merge_denial(
    queue: MergeQueue,
    record: dict[str, object],
) -> None:
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    )
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO post_merge_review_denials (
                 terminal_key_id, denial_id,
                 target_repository_id, target_branch, task_id,
                 canonical_task_key, canonical_task_cid,
                 task_binding_id, implementation_commit,
                 record_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["terminal_key_id"],
                record["denial_id"],
                record["target_repository_id"],
                record["target_branch"],
                record["task_id"],
                record["canonical_task_key"],
                record["canonical_task_cid"],
                record["task_binding_id"],
                record["implementation_commit"],
                canonical,
                1.0,
            ),
        )
        connection.execute(
            """DELETE FROM agent_supervisor_store_metadata
               WHERE key='post_merge_correction_registry:migrated'"""
        )
        connection.commit()
    queue._initialize_post_merge_correction_registry()


def test_completed_projection_filters_candidates_existentially(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    first = queue.enqueue(
        branch_name="candidate/first",
        task_id="UIR-002",
        canonical_task_id="baguqeeraexample",
        canonical_task_key="task/v1/example",
        commit_sha="1" * 40,
    )
    second = queue.enqueue(
        branch_name="candidate/second",
        task_id="UIR-002",
        canonical_task_id="baguqeeraexample",
        canonical_task_key="task/v1/example",
        commit_sha="2" * 40,
    )
    for request in (first, second):
        claimed = queue.dequeue(
            consumer_id=f"merge-train:{request.request_id}"
        )
        assert claimed is not None
        queue.complete(claimed)

    assert queue.completed_canonical_task_ids() == {
        "baguqeeraexample"
    }
    assert queue.completed_canonical_task_ids(
        candidate_is_denied=lambda request: (
            request.commit_sha == first.commit_sha
        )
    ) == {"baguqeeraexample"}
    assert queue.completed_canonical_task_ids(
        candidate_is_denied=lambda _request: True
    ) == set()
    assert queue.completed_canonical_task_ids(
        candidate_is_eligible=lambda request: (
            request.commit_sha == second.commit_sha
        )
    ) == {"baguqeeraexample"}
    assert queue.completed_canonical_task_ids(
        candidate_is_eligible=lambda _request: False
    ) == set()
    with pytest.raises(RuntimeError, match="denial proof unavailable"):
        queue.completed_canonical_task_ids(
            candidate_is_denied=lambda _request: (_ for _ in ()).throw(
                RuntimeError("denial proof unavailable")
            )
        )


@pytest.mark.parametrize(
    ("predicate_name", "malformed_result"),
    (
        ("candidate_is_denied", None),
        ("candidate_is_denied", 0),
        ("candidate_is_denied", "false"),
        ("candidate_is_eligible", None),
        ("candidate_is_eligible", 1),
        ("candidate_is_eligible", {"error": "unavailable"}),
    ),
)
def test_completed_projection_rejects_non_boolean_predicate_results(
    tmp_path: Path,
    predicate_name: str,
    malformed_result: object,
) -> None:
    queue = MergeQueue(tmp_path / "queue")
    pending = queue.enqueue(
        branch_name="candidate/only",
        task_id="UIR-002",
        canonical_task_id="baguqeeraexample",
        canonical_task_key="task/v1/example",
        commit_sha="1" * 40,
    )
    claimed = queue.dequeue(
        consumer_id=f"merge-train:{pending.request_id}"
    )
    assert claimed is not None
    queue.complete(claimed)

    with pytest.raises(
        TypeError,
        match=rf"{predicate_name} must return an exact bool",
    ):
        queue.completed_canonical_task_ids(
            **{
                predicate_name: (
                    lambda _request: malformed_result
                ),
            }
        )


def test_post_merge_denial_consumption_is_permanent_and_tamper_evident(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'f' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    denial = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    consumption = _post_merge_denial_consumption_record(denial)
    queue.record_post_merge_review_denial(denial)

    assert (
        queue.record_post_merge_review_denial_consumption(
            consumption
        )
        == consumption
    )
    assert (
        queue.record_post_merge_review_denial_consumption(
            consumption
        )
        == consumption
    )
    restarted = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    assert (
        restarted.verified_post_merge_review_denial_consumptions()
        == (consumption,)
    )

    with restarted._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE post_merge_review_denial_consumptions
               SET record_json='{"tampered":true}'"""
        )
        connection.commit()
    with pytest.raises(
        MergeQueueIntegrityError,
        match="schema fields",
    ):
        restarted.verified_post_merge_review_denial_consumptions()


def test_post_merge_consumption_rejects_unauthorized_denial_before_promotion(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'9' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    authorized = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    unauthorized = _evolved_post_merge_denial_record(
        authorized,
        marker="a",
        correction_authorized=False,
    )
    queue.record_post_merge_review_denial(unauthorized)

    with pytest.raises(
        MergeQueueIntegrityError,
        match="does not match its permanent denial",
    ):
        queue.record_post_merge_review_denial_consumption(
            _post_merge_denial_consumption_record(unauthorized)
        )
    assert (
        queue.verified_post_merge_review_denial_consumptions()
        == ()
    )

    assert queue.record_post_merge_review_denial(authorized) == authorized
    consumption = _post_merge_denial_consumption_record(authorized)
    assert (
        queue.record_post_merge_review_denial_consumption(
            consumption
        )
        == consumption
    )
    assert (
        queue.verified_post_merge_review_denial_consumptions()
        == (consumption,)
    )


def test_post_merge_consumption_survives_authorized_representative_evolution(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'8' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    original = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    evolved = _evolved_post_merge_denial_record(
        original,
        marker="a",
        correction_authorized=True,
    )
    queue.record_post_merge_review_denial(evolved)
    consumption = _post_merge_denial_consumption_record(evolved)
    queue.record_post_merge_review_denial_consumption(consumption)

    selected = queue.record_post_merge_review_denial(original)

    assert selected == evolved
    assert selected["correction_origin_stream_id"] == (
        consumption["correction_origin_stream_id"]
    )
    assert (
        queue.verified_post_merge_review_denial_consumptions()
        == (consumption,)
    )


def test_post_merge_denial_registry_rejects_competing_authorized_origins(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'e' * 64}"
    seed = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    first = _evolved_post_merge_denial_record(
        seed,
        marker="a",
        correction_authorized=True,
    )
    second = _evolved_post_merge_denial_record(
        seed,
        marker="b",
        correction_authorized=True,
    )
    for index, order in enumerate(((first, second), (second, first))):
        queue = MergeQueue(
            tmp_path / f"authorized-{index}",
            target_repository_id=repository_id,
            target_branch="agent/uiir",
            require_target_binding=True,
        )
        queue.record_post_merge_review_denial(order[0])
        with pytest.raises(
            MergeQueueIntegrityError,
            match="multiple authorized origin streams",
        ):
            queue.record_post_merge_review_denial(order[1])
        assert queue.verified_post_merge_review_denials() == (
            order[0],
        )


def test_legacy_denial_is_suppression_only_until_source_bound_promotion(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'7' * 64}"
    queue_path = tmp_path / "queue"
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    current = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    legacy = _legacy_post_merge_denial_record(current)
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """DELETE FROM agent_supervisor_store_metadata
               WHERE key='post_merge_correction_registry:migrated'"""
        )
        connection.commit()
    _insert_stored_post_merge_denial(queue, legacy)
    queue = MergeQueue(
        queue_path,
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )

    assert queue.verified_post_merge_review_denials() == (legacy,)
    with pytest.raises(
        MergeQueueIntegrityError,
        match="does not match its permanent denial",
    ):
        queue.record_post_merge_review_denial_consumption(
            _post_merge_denial_consumption_record(legacy)
        )

    assert queue.record_post_merge_review_denial(current) == current
    assert queue.verified_post_merge_review_denials() == (current,)


def test_legacy_denial_consumption_marker_fails_closed(
    tmp_path: Path,
) -> None:
    repository_id = f"repository:sha256:{'6' * 64}"
    queue = MergeQueue(
        tmp_path / "queue",
        target_repository_id=repository_id,
        target_branch="agent/uiir",
        require_target_binding=True,
    )
    current = _post_merge_denial_record(
        repository_id=repository_id,
        target_branch="agent/uiir",
    )
    legacy = _legacy_post_merge_denial_record(current)
    consumption = _post_merge_denial_consumption_record(legacy)
    _insert_stored_post_merge_denial(queue, legacy)
    canonical = json.dumps(
        consumption,
        sort_keys=True,
        separators=(",", ":"),
    )
    with queue._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO post_merge_review_denial_consumptions (
                 terminal_key_id, consumption_id, denial_id,
                 target_repository_id, target_branch, task_id,
                 canonical_task_key, canonical_task_cid,
                 task_binding_id, implementation_commit,
                 record_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                consumption["terminal_key_id"],
                consumption["consumption_id"],
                consumption["denial_id"],
                consumption["target_repository_id"],
                consumption["target_branch"],
                consumption["task_id"],
                consumption["canonical_task_key"],
                consumption["canonical_task_cid"],
                consumption["task_binding_id"],
                consumption["implementation_commit"],
                canonical,
                2.0,
            ),
        )
        connection.commit()

    with pytest.raises(
        MergeQueueIntegrityError,
        match="consumption registry denial binding changed",
    ):
        queue.verified_post_merge_review_denial_consumptions()
    with pytest.raises(
        MergeQueueIntegrityError,
        match="consumed denial representative changed",
    ):
        queue.record_post_merge_review_denial(current)
