import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.portal_recovery_budget import portal_recovery_budget_consumed

SOURCE = {"source_head": "a" * 40, "source_tree": "b" * 40}
OTHER = {"source_head": "c" * 40, "source_tree": "d" * 40}


def test_legacy_budget_stays_closed_without_verified_source():
    assert portal_recovery_budget_consumed([{"settlement_id": "old"}], settlement_id="new")


def test_new_source_allows_one_distinct_settled_failure():
    history = [{"settlement_id": "old"}]
    assert not portal_recovery_budget_consumed(history, settlement_id="new", accepted_source=SOURCE)
    history.append({"settlement_id": "new", "accepted_recovery_source": SOURCE})
    assert portal_recovery_budget_consumed(history, settlement_id="later", accepted_source=SOURCE)
    assert not portal_recovery_budget_consumed(history, settlement_id="later", accepted_source=OTHER)


def test_same_settlement_never_rearms_again_even_with_new_source():
    assert portal_recovery_budget_consumed(
        [{"settlement_id": "same", "accepted_recovery_source": SOURCE}],
        settlement_id="same", accepted_source=OTHER,
    )


@pytest.mark.parametrize("source", [{}, {"source_head": "a" * 40}, {"source_head": True, "source_tree": "b" * 40}])
def test_invalid_source_does_not_reset_budget(source):
    assert portal_recovery_budget_consumed([{"settlement_id": "old"}], settlement_id="new", accepted_source=source)


@pytest.mark.parametrize("event", [{}, {"settlement_id": "old", "accepted_recovery_source": {}}, {"settlement_id": "old", "accepted_recovery_source": "bad"}])
def test_corrupt_history_fails_closed(event):
    assert portal_recovery_budget_consumed([event], settlement_id="new", accepted_source=SOURCE)
