"""Composite post-merge binding contracts on current main surfaces.

The UIIR worktree's live ``post_merge_review`` module is a multi-kLOC parallel
evolution.  Main already admits post-merge acceptance through
``post_merge_validation`` + production provider-review attestation.  These
tests lock the binding that composite/submodule-style landings cannot skip
those gates, and that trust-boundary Git ignores graft/replace ancestry.
"""

from __future__ import annotations

import os
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.authoritative_completion import (
    ImplementationReceipt,
    bound_gate_evidence,
    evaluate_authoritative_completion_gate,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.git_environment import (
    sanitized_git_environment,
)


def test_sanitized_git_environment_disables_grafts_and_replace_objects(
    tmp_path: Path, monkeypatch
) -> None:
    graft = tmp_path / "grafts"
    graft.write_text("deadbeef " + ("cafebabe" * 5) + "\n", encoding="utf-8")
    monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/replace")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "evil.git"))
    monkeypatch.setenv("PATH", "/usr/bin")

    env = sanitized_git_environment()

    assert env["GIT_GRAFT_FILE"] == os.devnull
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_DIR" not in env
    assert "GIT_REPLACE_REF_BASE" not in env
    assert env["PATH"] == "/usr/bin"


def test_composite_style_receipt_still_requires_provider_review_and_post_merge() -> None:
    """A merge that lands child gitlinks is not completion-complete without both gates."""

    binding = {
        "task_id": "CMP-001",
        "implementation_commit": "a" * 40,
        "merge_commit": "b" * 40,
        "repository_tree_id": "c" * 40,
    }
    evidence = {
        "merge": bound_gate_evidence(
            "merge",
            **binding,
            satisfied=True,
            composite_gitlink_paths=("external/child",),
            landed_child_commits={"external/child": "d" * 40},
        ),
        "validation": bound_gate_evidence(
            "validation",
            **binding,
            satisfied=False,
            validation_scope="post_merge",
        ),
    }
    receipt = ImplementationReceipt(
        task_id=binding["task_id"],
        implementation_commit=binding["implementation_commit"],
        merge_commit=binding["merge_commit"],
        repository_tree_id=binding["repository_tree_id"],
        merged=True,
        validation_passed=False,
        gate_evidence=evidence,
        model_invocation_observed=True,
    )
    gate = evaluate_authoritative_completion_gate(receipt)
    assert gate.admitted is False


def test_composite_merge_with_satisfied_validation_still_needs_provider_review() -> None:
    binding = {
        "task_id": "CMP-002",
        "implementation_commit": "1" * 40,
        "merge_commit": "2" * 40,
        "repository_tree_id": "3" * 40,
    }
    evidence = {
        "merge": bound_gate_evidence(
            "merge",
            **binding,
            satisfied=True,
            composite_gitlink_paths=("ipfs_kit_py",),
        ),
        "validation": bound_gate_evidence(
            "validation",
            **binding,
            satisfied=True,
            validation_scope="post_merge",
            attempt_id="attempt-1",
            returncode=0,
        ),
    }
    receipt = ImplementationReceipt(
        task_id=binding["task_id"],
        implementation_commit=binding["implementation_commit"],
        merge_commit=binding["merge_commit"],
        repository_tree_id=binding["repository_tree_id"],
        merged=True,
        validation_passed=True,
        gate_evidence=evidence,
        model_invocation_observed=True,
    )
    gate = evaluate_authoritative_completion_gate(receipt)
    assert gate.admitted is False
    # Provider review must still be missing / unsatisfied for model-backed work.
    assert "provider_review" not in (receipt.gate_evidence or {}) or (
        receipt.gate_evidence.get("provider_review", {}).get("satisfied") is not True
    )
