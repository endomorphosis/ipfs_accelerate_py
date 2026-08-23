"""EAAEF-032: exact-action approval and preserved denials."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.authority.approvals import (
    EFFECTS_REQUIRING_APPROVAL,
    ApprovalError,
    ApprovalLog,
)


def test_required_effects_need_authenticated_approval() -> None:
    log = ApprovalLog()
    for action in sorted(EFFECTS_REQUIRING_APPROVAL):
        with pytest.raises(ApprovalError, match="requires"):
            log.require(action=action, input_binding="cid:a", principal_id="did:key:operator")
        record = log.decide(
            principal_id="did:key:operator",
            action=action,
            input_binding="cid:a",
            decision="approved",
            reason_code="operator",
            created_at_ms=1,
        )
        assert log.require(action=action, input_binding="cid:a", principal_id="did:key:operator") is record


def test_denials_are_preserved() -> None:
    log = ApprovalLog()
    log.decide(
        principal_id="did:key:reviewer",
        action="merge",
        input_binding="patch:1",
        decision="denied",
        reason_code="unsafe",
        created_at_ms=1,
    )
    with pytest.raises(ApprovalError, match="denial"):
        log.decide(
            principal_id="did:key:reviewer",
            action="merge",
            input_binding="patch:1",
            decision="approved",
            reason_code="retry",
            created_at_ms=2,
        )
    with pytest.raises(ApprovalError, match="denial"):
        log.require(action="merge", input_binding="patch:1", principal_id="did:key:reviewer")
    assert log.is_denied(action="merge", input_binding="patch:1")


def test_worker_and_cid_cannot_approve() -> None:
    log = ApprovalLog()
    with pytest.raises(ApprovalError, match="cannot approve"):
        log.decide(
            principal_id="worker:lane-0",
            action="push",
            input_binding="cid:x",
            decision="approved",
            reason_code="self",
            created_at_ms=1,
        )
    with pytest.raises(ApprovalError, match="cannot approve"):
        log.decide(
            principal_id="sha256:" + ("a" * 64),
            action="network",
            input_binding="cid:x",
            decision="approved",
            reason_code="cid",
            created_at_ms=1,
        )
