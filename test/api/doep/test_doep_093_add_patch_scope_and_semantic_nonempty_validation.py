"""DOEP-093 patch scope and semantic-nonempty validation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.scope_adjudication import (
    ScopeAdjudicationError,
    validate_semantic_nonempty_patch,
)


def test_in_scope_semantic_patch_is_accepted_without_completion() -> None:
    receipt = validate_semantic_nonempty_patch(
        changed_paths=["ipfs_accelerate_py/agent_supervisor/planning/foo.py"],
        declared_scope=["ipfs_accelerate_py/agent_supervisor"],
        semantic_delta={"added_symbols": ["Foo"]},
    )
    assert receipt["accepted"] is True
    assert receipt["completion_authority"] is False


def test_empty_or_out_of_scope_or_markdown_only_fails() -> None:
    with pytest.raises(ScopeAdjudicationError, match="empty"):
        validate_semantic_nonempty_patch(changed_paths=[], declared_scope=["**"])
    with pytest.raises(ScopeAdjudicationError, match="outside declared scope"):
        validate_semantic_nonempty_patch(
            changed_paths=["secrets/token"],
            declared_scope=["ipfs_accelerate_py/**"],
        )
    with pytest.raises(ScopeAdjudicationError, match="semantically empty"):
        validate_semantic_nonempty_patch(
            changed_paths=["docs/note.md"],
            declared_scope=["docs/**"],
            semantic_delta={},
        )
