"""Exact, content-bound post-merge validation evidence."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.authoritative_completion import (
    AuthoritativeCompletionMixin,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    build_post_merge_validation_evidence,
    verify_post_merge_validation_evidence,
)

TASK_ID = "PMV-001"
IMPLEMENTATION_COMMIT = "a" * 40
MERGE_COMMIT = "b" * 40
REPOSITORY_TREE_ID = f"git-tree:{'c' * 40}"


def _evidence() -> dict[str, object]:
    return build_post_merge_validation_evidence(
        task_id=TASK_ID,
        target_commit=MERGE_COMMIT,
        repository_tree_id=REPOSITORY_TREE_ID,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "selection": {"scope": "post_merge"},
            "results": [
                {
                    "validation_id": "declared:unit",
                    "returncode": 0,
                    "stage": "test",
                }
            ],
        },
    )


def test_builder_emits_exact_cid_bound_receipt() -> None:
    evidence = _evidence()

    assert evidence["schema"] == POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
    assert evidence["validation_scope"] == "post_merge"
    assert evidence["target_commit"] == MERGE_COMMIT
    assert evidence["validated_commit"] == MERGE_COMMIT
    assert evidence["repository_tree_id"] == REPOSITORY_TREE_ID
    assert str(evidence["validation_result_cid"]).startswith("b")
    assert str(evidence["validation_receipt_id"]).startswith("b")
    assert verify_post_merge_validation_evidence(
        evidence,
        expected_task_id=TASK_ID,
        expected_target_commit=MERGE_COMMIT,
        expected_repository_tree_id=REPOSITORY_TREE_ID,
    ) == (True, ())


def test_builder_rejects_keys_that_collide_after_bounding() -> None:
    shared_prefix = "x" * 4096
    with pytest.raises(
        ContractValidationError,
        match="keys collide after bounding",
    ):
        build_post_merge_validation_evidence(
            task_id=TASK_ID,
            target_commit=MERGE_COMMIT,
            repository_tree_id=REPOSITORY_TREE_ID,
            validation_result={
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "details": {
                    f"{shared_prefix}a": "first",
                    f"{shared_prefix}b": "second",
                },
            },
        )


def test_builder_redacts_raw_output_and_verifier_rejects_unbound_extra() -> None:
    evidence = build_post_merge_validation_evidence(
        task_id=TASK_ID,
        target_commit=MERGE_COMMIT,
        repository_tree_id=REPOSITORY_TREE_ID,
        validation_result={
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [{"returncode": 0, "stdout": "private output"}],
        },
    )
    results = evidence["validation_result"]["results"]  # type: ignore[index]
    assert results == [{"returncode": 0}]

    evidence["validation_result"]["stdout"] = "unbound"  # type: ignore[index]
    verified, reasons = verify_post_merge_validation_evidence(evidence)
    assert verified is False
    assert "post_merge_validation_result_not_canonical" in reasons


@pytest.mark.parametrize(
    ("path", "replacement", "reason"),
    [
        (("task_id",), "PMV-OTHER", "post_merge_validation_task_mismatch"),
        (
            ("target_commit",),
            "d" * 40,
            "post_merge_validation_target_mismatch",
        ),
        (
            ("repository_tree_id",),
            f"git-tree:{'e' * 40}",
            "post_merge_validation_tree_binding_mismatch",
        ),
        (
            ("validation_result", "results"),
            [],
            "post_merge_validation_result_cid_mismatch",
        ),
        (
            ("validation_receipt_id",),
            "arbitrary-nonempty-id",
            "post_merge_validation_receipt_id_mismatch",
        ),
    ],
)
def test_verifier_rejects_tampered_receipt(
    path: tuple[str, ...],
    replacement: object,
    reason: str,
) -> None:
    evidence = deepcopy(_evidence())
    target = evidence
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = replacement  # type: ignore[index]

    verified, reasons = verify_post_merge_validation_evidence(
        evidence,
        expected_task_id=TASK_ID,
        expected_target_commit=MERGE_COMMIT,
        expected_repository_tree_id=REPOSITORY_TREE_ID,
    )

    assert verified is False
    assert reason in reasons


class _AcceptanceHarness(AuthoritativeCompletionMixin):
    def _verified_acceptance_binding(
        self,
        implementation_commit: str,
        merge_commit: str,
        repository_tree_id: str,
    ) -> tuple[str, str, bool]:
        return merge_commit, repository_tree_id, bool(implementation_commit)

    @staticmethod
    def _task_has_proof_obligation(_task: object) -> bool:
        return False

    @staticmethod
    def _task_uses_typed_local_execution(_task: object) -> bool:
        return True

    @staticmethod
    def _task_declares_independent_codex_review(_task: object) -> bool:
        return False

    @staticmethod
    def _task_model_assisted_provider_roles(_task: object) -> tuple[str, ...]:
        return ()


def test_authoritative_completion_requires_verified_receipt_identity() -> None:
    harness = _AcceptanceHarness()
    task = SimpleNamespace(task_id=TASK_ID)
    valid = harness.build_task_implementation_receipt(
        task,
        implementation_commit=IMPLEMENTATION_COMMIT,
        merge_commit=MERGE_COMMIT,
        repository_tree_id=REPOSITORY_TREE_ID,
        merged=True,
        validation_result=_evidence(),
        model_invocation_observed=False,
    )
    forged = deepcopy(_evidence())
    forged["validation_receipt_id"] = "nonempty-but-not-content-bound"
    denied = harness.build_task_implementation_receipt(
        task,
        implementation_commit=IMPLEMENTATION_COMMIT,
        merge_commit=MERGE_COMMIT,
        repository_tree_id=REPOSITORY_TREE_ID,
        merged=True,
        validation_result=forged,
        model_invocation_observed=False,
    )

    assert valid.validation_passed is True
    assert "freshness" not in valid.pending_gates
    assert "semantic" not in valid.pending_gates
    assert denied.validation_passed is False
    assert "freshness" in denied.pending_gates
    assert "semantic" in denied.pending_gates
