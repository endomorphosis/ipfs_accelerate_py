from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.negative_memory import (
    NEGATIVE_MEMORY_BLOCKS_REUSE,
    negative_memory_view,
)


def test_missing_negative_memory_does_not_block() -> None:
    view = negative_memory_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_explicit_negative_episode_blocks_reuse_and_does_not_complete() -> None:
    view = negative_memory_view({"negative_episode": True, "key_cid": "key:1"})
    assert view["claimed"] is True
    assert view["blocks_completion"] is True
    assert view["reason_code"] == NEGATIVE_MEMORY_BLOCKS_REUSE
    assert view["completes_task"] is False
    assert view["accepted_as_authority"] is False


def test_retained_negative_cid_blocks_matching_query() -> None:
    view = negative_memory_view(
        {
            "retained_negative_cids": ("key:bad", "key:other"),
            "key_cid": "key:bad",
        }
    )
    assert view["blocks_completion"] is True
    miss = negative_memory_view(
        {
            "retained_negative_cids": ("key:bad",),
            "key_cid": "key:ok",
        }
    )
    assert miss["claimed"] is True
    assert miss["blocks_completion"] is False


def test_accepted_transition_negative_episode_cannot_be_reused() -> None:
    view = negative_memory_view(
        {
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
                "negative_episode": True,
                "transition_cid": "tr:1",
            }
        }
    )
    assert view["blocks_completion"] is True
    assert view["completes_task"] is False
