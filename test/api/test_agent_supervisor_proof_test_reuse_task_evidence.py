from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_cached_test_validation import (
    ProofCachedTestValidationReceipt,
    ProofCachedTestValidationResult,
    validation_command_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_task_evidence import (
    ProofTestReuseTaskEvidenceCollection,
    ProofTestReuseTaskEvidenceCollector,
    ProofTestReuseTaskEvidenceError,
    TaskCompletionProvenanceKind,
    TaskEvidenceGapKind,
)

NOW = 1_786_000_000.0
NOW_MS = int(NOW * 1_000)
COMMAND = (
    "IPFS_TEST_PROOF_REUSE_MODE=off "
    "python3 -m pytest test/proof/test_one.py -q"
)
REPOSITORY_ID = "repository:sha256:current"
STATE_CID = "baguqeera-state-current"
COMMIT = "f" * 40
TREE = "e" * 40
GITLINKS = "baguqeera-gitlinks-current"
FOREST = "baguqeera-forest-current"
OVERLAY = "baguqeera-overlay-clean"
POLICY = "baguqeera-policy-current"
CAPABILITY = "baguqeera-capability-current"
KEY = "baguqeera-key-current"
CIRCUIT = "baguqeera-circuit-current"
OBJECTIVE_REVISION = "baguqeera-objective-current"


def _task(task_id: str, *, goal_id: str | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal_id": goal_id or f"{task_id}-GOAL",
        "canonical_task_cid": f"baguqeera-task-{task_id.lower()}",
        "board_namespace": "proof-backed-test-reuse-v1",
        # Status is intentionally not required or consulted.
        "status": "todo",
        "validation": [COMMAND],
    }


def _board(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "valid": True,
        "board_namespace": "proof-backed-test-reuse-v1",
        "task_count": len(tasks),
        "todo_sha256": "a" * 64,
    }


def _seal(record: dict[str, Any], field: str) -> dict[str, Any]:
    return {**record, field: content_identity(record)}


def _validation(
    task: dict[str, Any],
    *,
    observed_at_ms: int = NOW_MS - 1_000,
    fresh_until_ms: int = NOW_MS + 30_000,
) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "goal_id": task["goal_id"],
        "task_cid": task["canonical_task_cid"],
        "validation_command": COMMAND,
        "validation_command_cid": validation_command_identity(COMMAND),
        "repository_id": REPOSITORY_ID,
        "repository_state_cid": STATE_CID,
        "git_commit_id": COMMIT,
        "git_tree_id": TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "proof_reuse_mode": "off",
        "disposition": "executed",
        "status": "passed",
        "passed": True,
        "exit_code": 0,
        "skipped_count": 0,
        "observed_at_ms": observed_at_ms,
        "fresh_until_ms": fresh_until_ms,
    }
    return _seal(record, "validation_receipt_cid")


def _queue(task: dict[str, Any], *, commit: str = "d" * 40) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "canonical_task_cid": task["canonical_task_cid"],
        "status": "completed",
        "commit_sha": commit,
    }


def _approval(
    task: dict[str, Any],
    *,
    kind: str = "reviewed_integration",
    commit: str = "d" * 40,
) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "task_cid": task["canonical_task_cid"],
        "kind": kind,
        "approved": True,
        "reviewer_id": "reviewer@example.invalid",
        "integrated_commit_id": commit,
        "integration_target_commit_id": COMMIT,
        "integration_receipt_cid": "baguqeera-reviewed-integration",
    }
    return _seal(record, "approval_cid")


def _retrospective_approval(
    task: dict[str, Any], *, commit: str = "c" * 40
) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "task_cid": task["canonical_task_cid"],
        "kind": "retrospective_review",
        "approved": True,
        "reviewer_id": "operator@example.invalid",
        "integrated_commit_id": commit,
        "approved_policy_cid": POLICY,
    }
    return _seal(record, "policy_approval_cid")


def _collector(**overrides: Any) -> ProofTestReuseTaskEvidenceCollector:
    values: dict[str, Any] = {
        "repository_id": REPOSITORY_ID,
        "repository_state_cid": STATE_CID,
        "git_commit_id": COMMIT,
        "git_tree_id": TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "objective_revision": OBJECTIVE_REVISION,
        "policy_cid": POLICY,
        "capability_cid": CAPABILITY,
        "verifying_key_cid": KEY,
        "circuit_cid": CIRCUIT,
        "ancestry_verifier": lambda ancestor, target: (
            bool(ancestor) and target == COMMIT
        ),
        "approval_verifier": lambda approval: True,
        "clock": lambda: NOW,
    }
    values.update(overrides)
    return ProofTestReuseTaskEvidenceCollector(**values)


def _gap_kinds(result: Any) -> set[TaskEvidenceGapKind]:
    return {gap.kind for gap in result.gaps}


def test_derives_population_from_validated_board_and_binds_current_tree() -> None:
    tasks = [_task("PTR-CUSTOM-2"), _task("PTR-CUSTOM-1")]
    result = _collector().collect(
        _board(tasks),
        task_records=tasks,
        merge_queue_records=[_queue(task) for task in tasks],
        validation_receipts=[_validation(task) for task in tasks],
    )

    assert result.authoritative
    assert result.required_task_ids == ("PTR-CUSTOM-1", "PTR-CUSTOM-2")
    assert not result.gaps
    for task in tasks:
        evidence = result.evidence_by_task[task["task_id"]]
        assert evidence.task_cid == task["canonical_task_cid"]
        assert evidence.repository_id == REPOSITORY_ID
        assert evidence.repository_state_cid == STATE_CID
        assert evidence.git_commit_id == COMMIT
        assert evidence.git_tree_id == TREE
        assert evidence.gitlink_state_cid == GITLINKS
        assert evidence.repository_forest_cid == FOREST
        assert evidence.dirty_overlay_cid == OVERLAY
        assert evidence.to_dict()["policy_cid"] == POLICY
        assert evidence.to_dict()["capability_cid"] == CAPABILITY
        assert evidence.to_dict()["verifying_key_cid"] == KEY
        assert evidence.to_dict()["circuit_cid"] == CIRCUIT
        assert evidence.validation.validation_command == COMMAND
        assert evidence.validation.validation_receipt_cid
        assert (
            evidence.task_provenance["kind"]
            == TaskCompletionProvenanceKind.MANAGED_MERGE.value
        )
        assert evidence.to_dict()["authority"] == "authoritative"


def test_task_evidence_packets_round_trip_and_reject_tampering() -> None:
    task = _task("PTR-009")
    result = _collector().collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[_validation(task)],
    )

    replayed = ProofTestReuseTaskEvidenceCollection.from_dict(result.to_record())
    assert replayed.content_id == result.content_id
    assert replayed.evidence[0].content_id == result.evidence[0].content_id
    assert (
        replayed.evidence[0].validation.provenance_cid
        == result.evidence[0].validation.provenance_cid
    )

    tampered = deepcopy(result.to_record())
    tampered["evidence"][0]["validation"]["receipt"]["git_tree_id"] = "forged"
    with pytest.raises(ProofTestReuseTaskEvidenceError):
        ProofTestReuseTaskEvidenceCollection.from_dict(tampered)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda receipt: receipt.update(
                {"fresh_until_ms": NOW_MS - 1, "validation_receipt_cid": ""}
            ),
            TaskEvidenceGapKind.VALIDATION_STALE,
        ),
        (
            lambda receipt: receipt.update(
                {
                    "observed_at_ms": NOW_MS - 301_000,
                    "fresh_until_ms": NOW_MS + 30_000,
                    "validation_receipt_cid": "",
                }
            ),
            TaskEvidenceGapKind.VALIDATION_STALE,
        ),
        (
            lambda receipt: receipt.update(
                {
                    "disposition": "ordinary_skip",
                    "skipped_count": 1,
                    "validation_receipt_cid": "",
                }
            ),
            TaskEvidenceGapKind.ORDINARY_SKIP,
        ),
        (
            lambda receipt: receipt.update(
                {"skipped": True, "validation_receipt_cid": ""}
            ),
            TaskEvidenceGapKind.ORDINARY_SKIP,
        ),
        (
            lambda receipt: receipt.update(
                {"skipped_count": "one", "validation_receipt_cid": ""}
            ),
            TaskEvidenceGapKind.VALIDATION_MALFORMED,
        ),
        (
            lambda receipt: receipt.update(
                {"repository_forest_cid": "wrong", "validation_receipt_cid": ""}
            ),
            TaskEvidenceGapKind.VALIDATION_BINDING_MISMATCH,
        ),
        (
            lambda receipt: receipt.update(
                {"proof_reuse_mode": "read", "validation_receipt_cid": ""}
            ),
            TaskEvidenceGapKind.PROOF_REUSE_NOT_OFF,
        ),
    ],
)
def test_validation_failures_are_typed_gaps(
    mutation: Any, expected: TaskEvidenceGapKind
) -> None:
    task = _task("PTR-002")
    receipt = _validation(task)
    mutation(receipt)
    # Reseal mutations where identity validation would otherwise mask the
    # semantically more specific rejection.
    receipt.pop("validation_receipt_cid", None)
    receipt = _seal(receipt, "validation_receipt_cid")

    result = _collector().collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[receipt],
    )

    assert not result.authoritative
    assert not result.evidence
    assert expected in _gap_kinds(result)
    assert all(not gap.authoritative and gap.authority == "none" for gap in result.gaps)


def test_malformed_or_missing_inputs_never_gain_authority() -> None:
    task = _task("PTR-003")
    malformed = _validation(task)
    malformed["passed"] = "yes"
    result = _collector().collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[malformed],
    )
    assert _gap_kinds(result) == {TaskEvidenceGapKind.VALIDATION_FAILED}

    missing = _collector().collect(_board([task]), task_records=[task])
    assert _gap_kinds(missing) == {TaskEvidenceGapKind.VALIDATION_MISSING}
    assert not missing.authoritative


def test_board_count_and_duplicate_or_contradictory_records_fail_closed() -> None:
    task = _task("PTR-010")
    bad_board = {**_board([task]), "task_count": 2}
    mismatch = _collector().collect(bad_board, task_records=[task])
    assert TaskEvidenceGapKind.BOARD_POPULATION_MISMATCH in _gap_kinds(mismatch)

    duplicate = _collector().collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task), _queue(task)],
        validation_receipts=[_validation(task)],
    )
    assert (
        TaskEvidenceGapKind.COMPLETION_PROVENANCE_CONTRADICTORY
        in _gap_kinds(duplicate)
    )

    contradiction = _collector().collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[_validation(task)],
        approval_records=[_approval(task)],
    )
    assert (
        TaskEvidenceGapKind.COMPLETION_PROVENANCE_CONTRADICTORY
        in _gap_kinds(contradiction)
    )


def test_queue_authority_requires_verified_ancestry() -> None:
    task = _task("PTR-012")
    inputs = {
        "task_records": [task],
        "merge_queue_records": [_queue(task)],
        "validation_receipts": [_validation(task)],
    }
    unavailable = _collector(ancestry_verifier=None).collect(_board([task]), **inputs)
    assert _gap_kinds(unavailable) == {TaskEvidenceGapKind.ANCESTRY_UNAVAILABLE}

    rejected = _collector(ancestry_verifier=lambda ancestor, target: False).collect(
        _board([task]), **inputs
    )
    assert _gap_kinds(rejected) == {TaskEvidenceGapKind.ANCESTRY_UNVERIFIED}


@pytest.mark.parametrize("task_id", ["PTR-000", "PTR-001", "PTR-011", "PTR-041"])
def test_historic_queue_gaps_require_genuine_approval(task_id: str) -> None:
    task = _task(task_id)
    result = _collector().collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[_validation(task)],
    )

    assert _gap_kinds(result) == {TaskEvidenceGapKind.APPROVAL_MISSING}
    assert not result.authoritative


def test_ptr_000_accepts_only_immutable_operator_planning_seal() -> None:
    task = _task("PTR-000")
    record = {
        "task_id": task["task_id"],
        "task_cid": task["canonical_task_cid"],
        "kind": "planning_seal",
        "approved": True,
        "operator_id": "operator@example.invalid",
        "planning_seal_cid": "baguqeera-plan-seal",
        "sealed_objective_revision": OBJECTIVE_REVISION,
    }
    approval = _seal(record, "operator_approval_cid")
    result = _collector().collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[_validation(task)],
        approval_records=[approval],
    )
    assert result.authoritative
    assert (
        result.evidence[0].task_provenance["kind"]
        == TaskCompletionProvenanceKind.OPERATOR_PLANNING_SEAL.value
    )

    forged = deepcopy(approval)
    forged["reviewer_note"] = "mutated after review"
    rejected = _collector().collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[_validation(task)],
        approval_records=[forged],
    )
    assert _gap_kinds(rejected) == {TaskEvidenceGapKind.APPROVAL_MALFORMED}

    unverifiable = _collector(approval_verifier=None).collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[_validation(task)],
        approval_records=[approval],
    )
    assert _gap_kinds(unverifiable) == {TaskEvidenceGapKind.APPROVAL_UNVERIFIED}


def test_retrospective_requires_ancestry_current_rerun_and_reviewed_approval() -> None:
    task = _task("PTR-041")
    integrated_commit = "c" * 40
    history = {
        "task_id": task["task_id"],
        "task_cid": task["canonical_task_cid"],
        "integrated_commit_id": integrated_commit,
        "source": "reviewed repository history",
    }
    result = _collector().collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[_validation(task)],
        retrospective_records=[history],
        approval_records=[
            _retrospective_approval(task, commit=integrated_commit)
        ],
    )

    assert result.authoritative
    provenance = result.evidence[0].task_provenance
    assert (
        provenance["kind"]
        == TaskCompletionProvenanceKind.RETROSPECTIVE_INTEGRATION_VERIFICATION.value
    )
    assert provenance["ancestry_verified"] is True
    assert provenance["current_tree_rerun_passed"] is True
    assert (
        provenance["current_tree_rerun_receipt_cid"]
        == result.evidence[0].validation_receipt_cid
    )
    assert provenance["policy_approved"] is True

    ordinary_skip = _validation(task)
    ordinary_skip["disposition"] = "ordinary_skip"
    ordinary_skip["skipped_count"] = 1
    ordinary_skip.pop("validation_receipt_cid")
    ordinary_skip = _seal(ordinary_skip, "validation_receipt_cid")
    rejected = _collector().collect(
        _board([task]),
        task_records=[task],
        validation_receipts=[ordinary_skip],
        retrospective_records=[history],
        approval_records=[
            _retrospective_approval(task, commit=integrated_commit)
        ],
    )
    assert _gap_kinds(rejected) == {TaskEvidenceGapKind.ORDINARY_SKIP}


def _proof_receipt(task: dict[str, Any]) -> ProofCachedTestValidationReceipt:
    return ProofCachedTestValidationReceipt(
        task_id=task["task_id"],
        goal_id=task["goal_id"],
        goal_revision=OBJECTIVE_REVISION,
        validation_command=COMMAND,
        validation_command_cid=validation_command_identity(COMMAND),
        repository_id=REPOSITORY_ID,
        repository_state_cid=STATE_CID,
        repository_forest_cid=FOREST,
        git_commit_id=COMMIT,
        git_tree_id=TREE,
        gitlink_state_cid=GITLINKS,
        gitlink_closure_complete=True,
        dirty=False,
        dirty_overlay_cid=OVERLAY,
        decision_cid="baguqeera-decision",
        execution_key_cid="baguqeera-execution",
        test_receipt_cid="baguqeera-test-receipt",
        certificate_cid="baguqeera-certificate",
        policy_cid=POLICY,
        statement_cid="baguqeera-statement",
        circuit_cid=CIRCUIT,
        verifying_key_cid=KEY,
        proof_system_id="groth16",
        certificate_epoch="epoch-2026-08",
        certificate_authority=CertificateAuthority.AUTHORITATIVE,
        verifier_id="local-verifier@1",
        verifier_result=ProofCachedTestValidationResult.VERIFIED,
        verifier_authority=CertificateAuthority.AUTHORITATIVE,
        verified_at_ms=NOW_MS - 1_000,
        fresh_until_ms=NOW_MS + 30_000,
        reason_codes=("proof_reverified",),
    )


def test_proof_backed_skip_is_accepted_only_after_local_verification() -> None:
    task = _task("PTR-060")
    proof_receipt = _proof_receipt(task)
    calls: list[str] = []

    def verify(receipt: ProofCachedTestValidationReceipt) -> bool:
        calls.append(receipt.validation_receipt_cid)
        return True

    accepted = _collector(proof_skip_verifier=verify).collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[proof_receipt.to_record()],
    )
    assert accepted.authoritative
    assert calls == [proof_receipt.validation_receipt_cid]
    assert accepted.evidence[0].validation.disposition == "proof_backed_skip"

    unavailable = _collector(proof_skip_verifier=None).collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[proof_receipt.to_record()],
    )
    assert _gap_kinds(unavailable) == {
        TaskEvidenceGapKind.PROOF_SKIP_VERIFIER_UNAVAILABLE
    }

    rejected = _collector(proof_skip_verifier=lambda receipt: False).collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[proof_receipt.to_record()],
    )
    assert _gap_kinds(rejected) == {TaskEvidenceGapKind.PROOF_SKIP_UNVERIFIED}

    stale_by_collector_policy = _collector(
        freshness_seconds=0.5,
        proof_skip_verifier=lambda receipt: True,
    ).collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[_queue(task)],
        validation_receipts=[proof_receipt.to_record()],
    )
    assert _gap_kinds(stale_by_collector_policy) == {
        TaskEvidenceGapKind.VALIDATION_STALE
    }
