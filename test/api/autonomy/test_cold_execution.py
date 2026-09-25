from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.cold_execution import (
    COLD_EXECUTION_REQUIRED,
    COLLECTION_SEED_NOT_PASS,
    DEFINITION_IS_NOT_CALL,
    MEMO_REUSE_PERMITTED,
    TEARDOWN_FAILED,
    TEARDOWN_REQUIRED,
    cold_execution_view,
    current_reuse_permitted,
    memo_blocks_completion,
)


def _permits(**overrides: object) -> dict[str, object]:
    payload = {
        "current_inputs": True,
        "current_effects": True,
        "current_evidence": True,
        "publication_checked": True,
    }
    payload.update(overrides)
    return payload


def test_missing_proof_reuse_payload_does_not_block() -> None:
    view = cold_execution_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["accepted_as_authority"] is False
    assert view["completes_task"] is False
    assert memo_blocks_completion(None) is False


def test_hash_memo_without_current_checks_requires_cold_execution() -> None:
    view = cold_execution_view(
        {
            "proof_reuse_decision": {
                "action": "SKIP",
                "reason_code": "proof_cache_hit",
            }
        }
    )
    assert view["memo_hit"] is True
    assert view["reuse_permitted"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == COLD_EXECUTION_REQUIRED
    assert view["completes_task"] is False


def test_memo_reuse_requires_all_four_current_checks() -> None:
    base = {
        "hash_memo": True,
        "current_inputs": True,
        "current_effects": True,
        "current_evidence": True,
        "publication_checked": False,
    }
    assert current_reuse_permitted(base) is False
    assert memo_blocks_completion(base) is True
    permitted = _permits(hash_memo=True)
    assert current_reuse_permitted(permitted) is True
    view = cold_execution_view(permitted)
    assert view["blocks_completion"] is False
    assert view["reuse_permitted"] is True
    assert view["reason_code"] == MEMO_REUSE_PERMITTED


def test_cold_execution_is_the_reference_and_does_not_complete() -> None:
    view = cold_execution_view({"cold_execution": True, "action": "RUN"})
    assert view["cold_reference"] is True
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert view["accepted_as_authority"] is False


def test_collection_seed_is_lookup_only_not_a_pass() -> None:
    view = cold_execution_view(
        _permits(
            hash_memo=True,
            pctdd_stage="collection_seed",
            proof_reuse_decision={"action": "SKIP", "reason_code": "proof_cache_hit"},
        )
    )
    assert view["reuse_permitted"] is False
    assert view["blocks_completion"] is True
    assert view["reason_code"] == COLLECTION_SEED_NOT_PASS
    assert view["completes_task"] is False


def test_fixture_definition_reuse_is_not_a_call_pass() -> None:
    view = cold_execution_view(
        _permits(
            hash_memo=True,
            reuse_kind="fixture_definition",
            whole_item_pass=True,
        )
    )
    assert view["reason_code"] == DEFINITION_IS_NOT_CALL
    assert view["reuse_permitted"] is False
    assert view["blocks_completion"] is True


def test_setup_requires_teardown_even_when_call_is_reusable() -> None:
    missing = cold_execution_view(
        _permits(hash_memo=True, setup_ran=True, teardown_ran=False)
    )
    assert missing["reason_code"] == TEARDOWN_REQUIRED
    assert missing["reuse_permitted"] is False
    failed = cold_execution_view(
        _permits(
            hash_memo=True,
            setup_ran=True,
            teardown_ran=True,
            teardown_failed=True,
        )
    )
    assert failed["reason_code"] == TEARDOWN_FAILED
    assert failed["completes_task"] is False
    ok = cold_execution_view(
        _permits(
            hash_memo=True,
            reuse_kind="call",
            setup_ran=True,
            teardown_ran=True,
        )
    )
    assert ok["reuse_permitted"] is True
    assert ok["reason_code"] == MEMO_REUSE_PERMITTED


def test_sawm_reuse_decision_is_not_a_proof_memo_hit() -> None:
    view = cold_execution_view(
        {
            "reuse_decision": {
                "decision": "reuse",
                "exact_match": True,
            }
        }
    )
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
