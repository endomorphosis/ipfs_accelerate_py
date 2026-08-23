"""EAAEF-083: one accepted logical result per claim key."""

from __future__ import annotations

from threading import Thread

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_logical_claim import (
    LOGICAL_CLAIM_KEY_FIELDS,
    DuplicateLogicalAcceptanceError,
    LogicalClaim,
    LogicalClaimError,
    LogicalClaimLedger,
    UnregisteredAttemptError,
)


def _fields(**overrides: str) -> dict[str, str]:
    payload = {
        "task_id": "EAAEF-083",
        "plan_revision": "EAAEF-PLAN-R1",
        "base_tree": "4298f4b06fa753a60ff8f95ffead39be9a83092c",
        "semantic_root": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        "task_spec_cid": "sha256:3e652bf8b135871311c945c44e1a6bba35d52a94077745246eb8187ff5972139",
        "idempotency_key": "sha256:eaabfb1922628eedac657b2ad1784a8c078aeb1ac609532c67b07386ac22411c",
    }
    payload.update(overrides)
    return payload


def _claim(**overrides: str) -> LogicalClaim:
    return LogicalClaim.bind(**_fields(**overrides))


def test_logical_claim_key_binds_all_fields() -> None:
    claim = _claim()
    assert LOGICAL_CLAIM_KEY_FIELDS == (
        "task_id",
        "plan_revision",
        "base_tree",
        "semantic_root",
        "task_spec_cid",
        "idempotency_key",
    )
    assert claim.key == (
        claim.task_id,
        claim.plan_revision,
        claim.base_tree,
        claim.semantic_root,
        claim.task_spec_cid,
        claim.idempotency_key,
    )
    assert claim.content_id.startswith("b")
    clone = _claim()
    assert clone.key == claim.key
    assert clone.content_id == claim.content_id
    for field in LOGICAL_CLAIM_KEY_FIELDS:
        other = _claim(**{field: f"other-{field}"})
        assert other.key != claim.key
        assert other.content_id != claim.content_id


def test_missing_key_fields_fail_closed() -> None:
    for field in LOGICAL_CLAIM_KEY_FIELDS:
        with pytest.raises(LogicalClaimError, match=field):
            _claim(**{field: ""})
        with pytest.raises(LogicalClaimError, match=field):
            _claim(**{field: "   "})


def test_multiple_attempts_may_register() -> None:
    claim = _claim()
    first = claim.register("attempt-1")
    second = claim.register("attempt-2")
    again = claim.register("attempt-1")
    assert first["status"] == "registered"
    assert second["status"] == "registered"
    assert again["status"] == "registered"
    assert claim.attempts == ("attempt-1", "attempt-2")
    assert claim.accepted_attempt_id is None


def test_accept_succeeds_once_then_fails_closed() -> None:
    claim = _claim()
    claim.register("attempt-1")
    claim.register("attempt-2")
    receipt = claim.accept("attempt-1")
    assert receipt["status"] == "accepted"
    assert receipt["attempt_id"] == "attempt-1"
    assert receipt["accepted_attempt_id"] == "attempt-1"
    assert claim.accepted_attempt_id == "attempt-1"
    with pytest.raises(DuplicateLogicalAcceptanceError, match="duplicate"):
        claim.accept("attempt-2")
    with pytest.raises(DuplicateLogicalAcceptanceError, match="duplicate"):
        claim.accept("attempt-1")
    late = claim.register("attempt-3")
    assert late["status"] == "registered"
    with pytest.raises(DuplicateLogicalAcceptanceError, match="duplicate"):
        claim.accept("attempt-3")


def test_accept_unregistered_attempt_fails_closed() -> None:
    claim = _claim()
    claim.register("attempt-1")
    with pytest.raises(UnregisteredAttemptError, match="not registered"):
        claim.accept("attempt-missing")


def test_different_idempotency_keys_are_different_claims() -> None:
    first = _claim(idempotency_key="idem-a")
    second = _claim(idempotency_key="idem-b")
    assert first.key != second.key
    first.register("attempt-1")
    second.register("attempt-1")
    accepted_first = first.accept("attempt-1")
    accepted_second = second.accept("attempt-1")
    assert accepted_first["status"] == "accepted"
    assert accepted_second["status"] == "accepted"
    assert accepted_first["logical_claim_key"] != accepted_second["logical_claim_key"]


def test_ledger_shares_acceptance_across_same_logical_key() -> None:
    ledger = LogicalClaimLedger()
    first = ledger.bind(**_fields())
    second = ledger.bind(LogicalClaim.from_mapping(_fields()))
    assert first is second
    other = ledger.bind(**_fields(idempotency_key="idem-other"))
    assert other is not first
    ledger.register(first, "attempt-1")
    ledger.register({"task_id": first.task_id, **{k: getattr(first, k) for k in LOGICAL_CLAIM_KEY_FIELDS if k != "task_id"}}, "attempt-2")
    receipt = ledger.accept(second, "attempt-1")
    assert receipt["status"] == "accepted"
    with pytest.raises(DuplicateLogicalAcceptanceError, match="duplicate"):
        ledger.accept(first, "attempt-2")
    other.register("attempt-1")
    assert other.accept("attempt-1")["status"] == "accepted"


def test_concurrent_accept_admits_one_result() -> None:
    claim = _claim()
    claim.register("attempt-1")
    claim.register("attempt-2")
    outcomes: list[str] = []

    def worker(attempt_id: str) -> None:
        try:
            claim.accept(attempt_id)
            outcomes.append("accepted")
        except DuplicateLogicalAcceptanceError:
            outcomes.append("rejected")

    threads = (
        Thread(target=worker, args=("attempt-1",)),
        Thread(target=worker, args=("attempt-2",)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("accepted") == 1
    assert outcomes.count("rejected") == 1
    assert claim.accepted_attempt_id in {"attempt-1", "attempt-2"}
