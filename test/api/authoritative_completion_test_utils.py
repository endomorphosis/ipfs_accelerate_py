"""Test-only builders for exact repository-bound completion authority."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.authoritative_completion import (
    AuthoritativeCompletionGate,
    ImplementationReceipt,
    bound_gate_evidence,
    build_implementation_receipt,
    promote_authoritative_completion,
)


def _git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def real_git_authority(
    repo_root: Path,
    task_id: str,
    *,
    deterministic_only: bool = False,
) -> tuple[ImplementationReceipt, AuthoritativeCompletionGate]:
    """Build an admitted packet bound to the repository's exact current tree."""

    implementation_commit = _git_output(repo_root, "rev-parse", "HEAD")
    repository_tree_id = (
        f"git-tree:{_git_output(repo_root, 'rev-parse', 'HEAD^{tree}')}"
    )
    binding = {
        "task_id": task_id,
        "implementation_commit": implementation_commit,
        "merge_commit": implementation_commit,
        "repository_tree_id": repository_tree_id,
    }
    validation = {
        **binding,
        "satisfied": True,
        "passed": True,
        "stale": False,
        "validation_scope": "post_merge",
        "validation_receipt_id": f"validation:{task_id}",
    }
    evidence = {
        "merge": bound_gate_evidence(
            "merge",
            **binding,
            satisfied=True,
        ),
        "freshness": bound_gate_evidence("freshness", **validation),
        "semantic": bound_gate_evidence("semantic", **validation),
        "proof": bound_gate_evidence(
            "proof",
            **binding,
            satisfied=True,
            not_applicable=True,
            applicability_decision="no_declared_proof_obligation",
        ),
        "provider_review": bound_gate_evidence(
            "provider_review",
            **binding,
            satisfied=True,
            **(
                {
                    "not_applicable": True,
                    "route_kind": "deterministic_only",
                    "model_invocation_observed": False,
                }
                if deterministic_only
                else {
                    "review_presence": "independent",
                    "provider_result_admitted": True,
                    "review_receipt_id": f"review:{task_id}",
                }
            ),
        ),
        "deterministic_only": bound_gate_evidence(
            "deterministic_only",
            **binding,
            satisfied=True,
            not_applicable=not deterministic_only,
            policy=(
                "deterministic_only"
                if deterministic_only
                else "not_deterministic_only"
            ),
            model_invocation_observed=False,
        ),
    }
    receipt = build_implementation_receipt(
        task_id=task_id,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=repository_tree_id,
        merged=True,
        validation_passed=True,
        gate_evidence=evidence,
        deterministic_only=deterministic_only,
    )
    promoted, gate = promote_authoritative_completion(
        receipt,
        expected_task_id=task_id,
    )
    if not gate.admitted:
        raise AssertionError(f"test authority was not admitted: {gate.to_dict()}")
    return promoted, gate
