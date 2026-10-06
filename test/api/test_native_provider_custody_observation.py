"""Closed custody diagnostics cannot carry authority or private payloads."""
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_provider_custody_observation import (
    append_native_provider_custody_check as append_check,
    validate_native_provider_custody_observation as validate,
)


def _complete():
    value = None
    for stage, reason in (("issuance", "eligible"), ("native_exit", "verified"),
            ("bridge_join", "verified"), ("capability_take", "verified")):
        value = append_check(value, stage, reason)
    return value


def test_complete_observation_has_no_authority_and_is_a_defensive_copy():
    original = _complete()
    copy = validate(original)
    assert copy == original
    assert copy["observation_only"] is True
    assert all(copy[key] is False for key in (
        "completion_authority", "retry_authority", "settlement_authority"))
    copy["checks"][0]["reason_code"] = "private mutation"
    assert original["checks"][0]["reason_code"] == "eligible"


@pytest.mark.parametrize("mutation", ["authority", "body", "private-reason", "private-stage",
    "stage-order", "duplicate-stage", "too-many", "wrong-status", "missing-status",
    "extra-check-field", "unknown-schema", "bool-stage", "mapping-subclass"])
def test_malformed_or_unbounded_observation_is_rejected(mutation):
    value = _complete()
    if mutation == "authority":
        value["settlement_authority"] = True
    elif mutation == "body":
        value["raw_provider_body"] = "private payload"
    elif mutation == "private-reason":
        value["checks"][0]["reason_code"] = "private payload"
    elif mutation == "private-stage":
        value["checks"][0]["stage"] = "/private/path"
    elif mutation == "stage-order":
        value["checks"].reverse()
    elif mutation == "duplicate-stage":
        value["checks"][1] = deepcopy(value["checks"][0])
    elif mutation == "too-many":
        value["checks"].append(deepcopy(value["checks"][-1]))
    elif mutation == "wrong-status":
        value["checks"][0]["status"] = "denied"
    elif mutation == "missing-status":
        value["checks"][0].pop("status")
    elif mutation == "extra-check-field":
        value["checks"][0]["command"] = "private payload"
    elif mutation == "unknown-schema":
        value["schema"] = "foreign"
    elif mutation == "bool-stage":
        value["checks"][0]["stage"] = True
    else:
        value = type("UntrustedMapping", (dict,), {})(value)
    assert validate(value) is None


def test_append_does_not_mutate_or_repair_untrusted_history():
    first = append_check(None, "issuance", "lifecycle_not_finalized")
    before = deepcopy(first)
    second = append_check(first, "native_exit", "not_issued")
    assert first == before and len(second["checks"]) == 2
    assert second["checks"][1]["status"] == "denied"
    assert append_check(first, "issuance", "eligible") is None
    assert append_check(first, "native_exit", "private reason") is None
    first["extra"] = "private payload"
    assert append_check(first, "native_exit", "not_issued") is None
