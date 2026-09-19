"""DOEP-094 proof/test-based merge admission."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    admit_proof_test_merge,
    build_post_merge_validation_evidence,
)


def _tests(*, passed: bool, tree: str = "tree-1", commit: str = "abc") -> dict:
    return build_post_merge_validation_evidence(
        task_id="DOEP-094",
        target_commit=commit,
        repository_tree_id=tree,
        validation_result={"attempted": True, "passed": passed, "returncode": 0 if passed else 1},
        validated_commit=commit,
    )


def test_merge_requires_both_tests_and_proofs() -> None:
    tests = _tests(passed=True)
    proofs = {"passed": True, "repository_tree_id": "tree-1"}
    admitted = admit_proof_test_merge(
        test_evidence=tests,
        proof_evidence=proofs,
        expected_task_id="DOEP-094",
        expected_target_commit="abc",
        expected_repository_tree_id="tree-1",
    )
    assert admitted["admitted"] is True
    assert admitted["completion_authority"] is False


def test_missing_proofs_or_failed_tests_are_refused() -> None:
    tests = _tests(passed=False)
    refused = admit_proof_test_merge(
        test_evidence=tests,
        proof_evidence={"passed": True, "repository_tree_id": "tree-1"},
        expected_task_id="DOEP-094",
        expected_target_commit="abc",
        expected_repository_tree_id="tree-1",
    )
    assert refused["admitted"] is False
    missing_proof = admit_proof_test_merge(
        test_evidence=_tests(passed=True),
        proof_evidence=None,
        expected_task_id="DOEP-094",
        expected_target_commit="abc",
        expected_repository_tree_id="tree-1",
    )
    assert missing_proof["reason_codes"] == ("proofs_not_passed",)
    assert missing_proof["completion_authority"] is False
