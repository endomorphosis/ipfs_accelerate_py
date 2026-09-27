from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
    clear_completion_blocks,
    completion_is_blocked,
    last_completion_blocks,
    publish_completion_blocks,
)


def test_empty_blocks_do_not_complete_and_do_not_block() -> None:
    clear_completion_blocks()
    blocks = last_completion_blocks()
    assert blocks["blocks_completion"] is False
    assert blocks["completion_authority"] is False
    assert blocks["accepted_as_authority"] is False
    assert completion_is_blocked() is False


def test_published_block_is_conjunctive_and_not_authority() -> None:
    clear_completion_blocks()
    publish_completion_blocks(rollout_not_required=True)
    blocks = last_completion_blocks()
    assert blocks["rollout_not_required"] is True
    assert blocks["blocks_completion"] is True
    assert blocks["reason"] == "rollout_not_required"
    assert blocks["completion_authority"] is False
    assert completion_is_blocked() is True
    assert completion_is_blocked({"runtime_settled": True, "merge_queue_empty": True}) is True
    clear_completion_blocks()
    assert completion_is_blocked() is False
    assert completion_is_blocked({"undeclared_cst_transform": True}) is True


def test_attach_completion_blocks_does_not_treat_unchanged_as_success() -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
        attach_completion_blocks,
    )

    clear_completion_blocks()
    publish_completion_blocks(cold_execution_required=True)
    stamped = attach_completion_blocks({"unchanged": True, "write_count": 0})
    assert stamped["unchanged"] is True
    assert stamped["blocked"] is True
    assert stamped["blocks_completion"] is True
    assert stamped["completion_authority"] is False
    assert stamped["reason"] == "cold_execution_required"
    assert stamped["completion_blocks"]["cold_execution_required"] is True
    lease = attach_completion_blocks(
        {"unchanged": True, "reason": "lease_held", "blocked": True}
    )
    assert lease["reason"] == "lease_held"
    assert lease["blocked"] is True


def test_admitted_proof_skip_runs_while_completion_is_blocked() -> None:
    from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
        RuntimeReuseAction,
        disposition_skip,
        gate_skip_disposition,
    )

    clear_completion_blocks()
    admitted = disposition_skip(
        certificate_cid="cid:cert:alpha",
        receipt_cid="cid:receipt:alpha",
    )
    kept = gate_skip_disposition(admitted)
    assert kept.action is RuntimeReuseAction.SKIP
    assert kept.is_skip is True

    publish_completion_blocks(cold_execution_required=True)
    refused = gate_skip_disposition(admitted)
    assert refused.action is RuntimeReuseAction.RUN
    assert refused.reason_code == "completion_blocked"
    assert refused.is_skip is False
    clear_completion_blocks()
