"""DOEP-065 ContextPack/ContextCapsule invalidation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextIdentityError,
    ContextReference,
    invalidate_context_capsule,
)


def _capsule(*, tree_id: str = "tree-a") -> ContextCapsule:
    return ContextCapsule(
        repository_id="repo:doep",
        tree_id=tree_id,
        objective_id="DOEP-G070",
        objective_revision="1",
        policy_id="policy:doep",
        policy_revision="1",
        caller="doep-065",
        stage="planning",
        budget=ContextBudget(),
        goal={"outcome": "invalidate stale packs"},
        authority={"maximum": "proposal"},
        scope={"paths": ["ipfs_accelerate_py/agent_supervisor/context"]},
        acceptance={"criteria": ["stale tree invalidation"]},
        evidence=(
            ContextReference(
                reference_id="policy",
                kind="policy",
                content_id="bafy-policy",
                repository_id="repo:doep",
                tree_id=tree_id,
                path="docs/architecture/agent_supervisor_direct_objective_event_driven_planning.todo.md",
                summary="plan card",
                token_count=8,
            ),
        ),
        input_tokens=16,
    )


def test_same_tree_remains_valid() -> None:
    receipt = invalidate_context_capsule(_capsule(), tree_id="tree-a", reason="check")
    assert receipt["valid"] is True
    assert receipt["stale"] is False
    assert receipt["completion_authority"] is False


def test_tree_mismatch_invalidates_without_completion_authority() -> None:
    receipt = invalidate_context_capsule(_capsule(tree_id="tree-a"), tree_id="tree-b", reason="rebase")
    assert receipt["stale"] is True
    assert receipt["valid"] is False
    assert receipt["completion_authoritative"] is False
    assert receipt["bound_tree_id"] == "tree-a"
    assert receipt["current_tree_id"] == "tree-b"


def test_missing_tree_id_fails_closed() -> None:
    with pytest.raises(ContextIdentityError, match="tree_id"):
        invalidate_context_capsule(_capsule(), tree_id="", reason="missing")
