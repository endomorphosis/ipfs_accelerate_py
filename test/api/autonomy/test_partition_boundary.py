from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.partition_boundary import (
    CROSS_SCC_WITHOUT_CONTRACT,
    PARTIAL_SCC_FORBIDDEN,
    PARTITION_RESPECTED,
    partition_boundary_view,
)


def test_missing_partition_payload_does_not_block() -> None:
    view = partition_boundary_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False


def test_single_scc_wave_is_respected_and_does_not_complete() -> None:
    view = partition_boundary_view(
        {
            "refactor_wave": True,
            "write_sccs": ("scc:a",),
        }
    )
    assert view["respected"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["reason_code"] == PARTITION_RESPECTED


def test_cross_scc_without_contract_cannot_complete() -> None:
    view = partition_boundary_view(
        {
            "refactor_wave": True,
            "write_sccs": ("scc:a", "scc:b"),
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == CROSS_SCC_WITHOUT_CONTRACT
    contracted = partition_boundary_view(
        {
            "refactor_wave": True,
            "write_sccs": ("scc:a", "scc:b"),
            "boundary_contract_set_cid": "contract:ab",
        }
    )
    assert contracted["respected"] is True
    assert contracted["blocks_completion"] is False
    assert contracted["completes_task"] is False


def test_partial_scc_completion_is_forbidden() -> None:
    view = partition_boundary_view(
        {
            "extraction_wave": True,
            "scc_ids": ("scc:a",),
            "scc_member_count": 4,
            "completed_scc_members": 2,
        }
    )
    assert view["blocks_completion"] is True
    assert view["reason_code"] == PARTIAL_SCC_FORBIDDEN
    assert view["accepted_as_authority"] is False
