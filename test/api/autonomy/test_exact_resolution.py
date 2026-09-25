from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.exact_resolution import (
    ACCEPTED_TRANSITION_REUSABLE,
    EXACT_RESOLUTION_SATISFIED,
    SIMILAR_TRANSITION_NOT_EXACT,
    SIMILARITY_NOT_RESOLUTION,
    TRANSITION_NOT_ACCEPTED,
    exact_resolution_view,
    similarity_blocks_completion,
)


def test_missing_world_payload_does_not_block_completion() -> None:
    view = exact_resolution_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["accepted_as_authority"] is False
    assert view["completes_task"] is False
    assert similarity_blocks_completion(None) is False


def test_similarity_candidates_without_exact_match_block_completion() -> None:
    view = exact_resolution_view(
        {"similarity_candidates": [{"score": 0.99, "nearest": True}]}
    )
    assert view["nominated"] is True
    assert view["ann_shaped"] is True
    assert view["exact_match"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == SIMILARITY_NOT_RESOLUTION
    assert view["completes_task"] is False


def test_exact_reuse_with_nomination_does_not_block() -> None:
    view = exact_resolution_view(
        {
            "reuse_decision": {
                "decision": "reuse",
                "exact_match": True,
                "reason_code": "exact_identity_match",
            },
            "similarity_candidates": [{"score": 0.91}],
        }
    )
    assert view["nominated"] is True
    assert view["exact_match"] is True
    assert view["blocks_completion"] is False
    assert view["reason_code"] == EXACT_RESOLUTION_SATISFIED


def test_accepted_pre_action_post_becomes_reusable_state() -> None:
    view = exact_resolution_view(
        {
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
            }
        }
    )
    assert view["transition_reusable"] is True
    assert view["exact_match"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == ACCEPTED_TRANSITION_REUSABLE


def test_unaccepted_transition_cannot_become_reusable_state() -> None:
    view = exact_resolution_view(
        {
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": False,
            }
        }
    )
    assert view["transition_reusable"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == TRANSITION_NOT_ACCEPTED


def test_similar_transition_cannot_substitute_for_exact_pre_action_post() -> None:
    view = exact_resolution_view(
        {
            "accepted_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:1",
                "accepted": True,
            },
            "query_transition": {
                "pre_cid": "pre:1",
                "action_cid": "act:1",
                "post_cid": "post:similar",
            },
            "similar_transition": True,
        }
    )
    assert view["nominated"] is True
    assert view["exact_match"] is False
    assert view["transition_reusable"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == SIMILAR_TRANSITION_NOT_EXACT
    assert view["completes_task"] is False


def test_reuse_reject_without_exact_match_blocks() -> None:
    assert (
        similarity_blocks_completion(
            {
                "reuse_decision": {
                    "verdict": "reject",
                    "exact_match": False,
                    "reason_code": "similarity_is_not_reuse",
                }
            }
        )
        is True
    )
