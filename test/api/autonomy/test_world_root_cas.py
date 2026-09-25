from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.world_root_cas import (
    PUBLICATION_REQUEST_ONLY,
    WORLD_ROOT_CAS_NOT_COMPLETION,
    world_root_cas_view,
)


def test_missing_world_root_payload_does_not_block() -> None:
    view = world_root_cas_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_publication_request_is_not_completion() -> None:
    view = world_root_cas_view(
        {
            "world_root_publication": {
                "semantic_world_root_cid": "root:1",
                "proposal_only": True,
                "cas_completed": False,
                "completion_authority": False,
                "changes_current_root": False,
            }
        }
    )
    assert view["claimed"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == PUBLICATION_REQUEST_ONLY
    assert view["accepted_as_authority"] is False


def test_cas_completed_on_world_root_cannot_complete() -> None:
    view = world_root_cas_view(
        {
            "world_root_publication": {
                "semantic_world_root_cid": "root:1",
                "cas_completed": True,
            }
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == WORLD_ROOT_CAS_NOT_COMPLETION


def test_vfs_outbox_cannot_claim_completion_authority() -> None:
    view = world_root_cas_view(
        {"vfs_outbox": True, "completion_authority": True}
    )
    assert view["blocks_completion"] is True
    assert view["completes_task"] is False


def test_publication_cannot_change_current_root() -> None:
    view = world_root_cas_view(
        {
            "publication_request": True,
            "semantic_world_root_cid": "root:1",
            "changes_current_root": True,
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == WORLD_ROOT_CAS_NOT_COMPLETION
