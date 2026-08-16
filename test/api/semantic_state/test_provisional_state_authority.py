"""Focused provisional-versus-canonical authority checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.provisional_state import (
    ProvisionalStateError,
    authority_transitions,
    bind_provisional_root,
    publish_canonical,
)


def test_provisional_root_is_isolated() -> None:
    bound = bind_provisional_root(
        {
            "root_cid": "sha256:" + ("aa" * 32),
            "task_id": "LGSWF-022",
            "attempt_id": "attempt:1",
            "worktree_tree": "tree-a",
        }
    )
    assert bound["canonical"] is False
    assert "verification" in bound["usable_for"]
    with pytest.raises(ProvisionalStateError, match="missing"):
        bind_provisional_root({"root_cid": "sha256:" + ("aa" * 32)})


def test_worker_and_provisional_publish_rejected() -> None:
    with pytest.raises(ProvisionalStateError, match="cannot become canonical"):
        publish_canonical(
            {
                "source": "provisional",
                "root_cid": "sha256:" + ("aa" * 32),
                "worktree_tree": "tree-a",
                "accepted_merge_tree": "tree-a",
                "fresh_datasets_rescan": True,
                "delta_receipt": True,
                "publisher": "post-merge-supervisor-refresh",
            }
        )
    with pytest.raises(ProvisionalStateError, match="cannot become canonical"):
        publish_canonical({"source": "worker", "root_cid": "x"})
    with pytest.raises(ProvisionalStateError, match="stale attempt"):
        publish_canonical(
            {
                "source": "merge",
                "stale_attempt": True,
                "root_cid": "sha256:" + ("aa" * 32),
                "worktree_tree": "tree-a",
                "accepted_merge_tree": "tree-a",
                "fresh_datasets_rescan": True,
                "delta_receipt": True,
                "publisher": "post-merge-supervisor-refresh",
            }
        )
    with pytest.raises(ProvisionalStateError, match="wrong worktree"):
        publish_canonical(
            {
                "source": "merge",
                "root_cid": "sha256:" + ("aa" * 32),
                "worktree_tree": "tree-a",
                "accepted_merge_tree": "tree-b",
                "fresh_datasets_rescan": True,
                "delta_receipt": True,
                "publisher": "post-merge-supervisor-refresh",
            }
        )


def test_only_post_merge_refresh_publishes_canonical() -> None:
    accepted = publish_canonical(
        {
            "source": "merge",
            "root_cid": "sha256:" + ("bb" * 32),
            "worktree_tree": "tree-c",
            "accepted_merge_tree": "tree-c",
            "fresh_datasets_rescan": True,
            "delta_receipt": True,
            "publisher": "post-merge-supervisor-refresh",
        }
    )
    assert accepted["canonical"] is True
    transitions = authority_transitions()
    assert any(row["from"] == "provisional" and row["allowed"] == "no" for row in transitions)
