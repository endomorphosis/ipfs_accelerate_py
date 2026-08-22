"""PCCE-015: assurance outcome bridge."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.proof_context.assurance_bridge import (
    AssuranceBridgeError,
    admit_campaign_outcome,
    open_campaign_api,
)


def test_open_campaign_api_is_lazy() -> None:
    api = open_campaign_api()
    assert api.interface_id


def test_critical_survivor_fails_acceptance() -> None:
    result = admit_campaign_outcome("succeeded", critical_survivor=True)
    assert result["accepted"] is False
    assert result["status"] == "critical_survivor"


def test_unavailable_and_timeout_are_not_success() -> None:
    for status in ("unavailable", "timeout", "infrastructure_failure"):
        result = admit_campaign_outcome(status)
        assert result["accepted"] is False


def test_simulated_and_self_approve_fail_closed() -> None:
    with pytest.raises(AssuranceBridgeError):
        admit_campaign_outcome("succeeded", provenance="simulated")
    with pytest.raises(AssuranceBridgeError):
        admit_campaign_outcome("succeeded", self_approved=True)
    with pytest.raises(AssuranceBridgeError):
        admit_campaign_outcome("succeeded", hidden_benchmark_exposed=True)


def test_typed_outcomes_are_closed() -> None:
    with pytest.raises(AssuranceBridgeError):
        admit_campaign_outcome("passed_anyway")
    assert admit_campaign_outcome("omission")["accepted"] is False
    assert admit_campaign_outcome("vacuity")["accepted"] is False
    assert admit_campaign_outcome("context_expansion")["accepted"] is False
