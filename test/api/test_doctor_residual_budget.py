"""Doctor limits stay bound without claiming unimplemented coding enforcement."""
import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_budget as budget
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context as context
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_residual_context import retained  # noqa: F401


LIMITS = dict(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1)


def _capsule():
    return {"residual_budget": dict(LIMITS),
            "token_budget": {"residual_" + key: value for key, value in LIMITS.items()}}


def test_budget_observation_is_detached_and_describes_proposal_accounting():
    capsule = _capsule()
    observed = budget.residual_budget_observation(capsule)
    assert observed["token_accounting"] == "residual-proposal-response-tokens"
    assert observed["call_accounting"] == observed["round_accounting"] == "residual-proposal-admissions"
    assert observed["cost_unit_accounting"] == "caller-supplied-residual-proposal-cost-units"
    assert observed["limits"] == LIMITS
    assert set(observed["coding_route_enforcement"].values()) == {False}
    for key in ("hard_native_token_cap", "provider_cost_unit_conversion",
                "remaining_budget_known", "additional_coding_call_authorized"):
        assert observed[key] is False
    assert observed["observed_usage"] is None
    assert budget.validate_budget_observation(observed) == observed
    observed["limits"]["max_calls"] = 99
    observed["coding_route_enforcement"]["tokens"] = True
    capsule["residual_budget"]["max_tokens"] = 1
    assert budget.residual_budget_observation(_capsule())["limits"] == LIMITS
    assert budget.residual_budget_observation(_capsule())["coding_route_enforcement"]["tokens"] is False


@pytest.mark.parametrize("key", budget.LIMIT_FIELDS)
@pytest.mark.parametrize("value", [None, False, True, 0, -1, 1.0, "1", 1 << 63])
def test_declared_limit_refuses_nonpositive_coerced_or_unbounded_values(key, value):
    capsule = _capsule()
    capsule["residual_budget"][key] = value
    capsule["token_budget"]["residual_" + key] = value
    with pytest.raises(ValueError):
        budget.validate_residual_budget(capsule)


@pytest.mark.parametrize("change", ["missing", "extra", "missing_copy", "extra_copy",
                                    "different_copy", "boolean_copy", "nonstring_copy"])
def test_budget_requires_closed_limits_and_matching_duplicate_projection(change):
    capsule = _capsule()
    if change == "missing":
        del capsule["residual_budget"]["max_calls"]
    elif change == "extra":
        capsule["residual_budget"]["scope_expansion"] = 1
    elif change == "missing_copy":
        del capsule["token_budget"]["residual_max_calls"]
    elif change == "extra_copy":
        capsule["token_budget"]["residual_max_other"] = 1
    elif change == "different_copy":
        capsule["token_budget"]["residual_max_calls"] = 2
    elif change == "boolean_copy":
        capsule["token_budget"]["residual_max_calls"] = True
    else:
        capsule["token_budget"][1] = 1
    with pytest.raises(ValueError):
        budget.validate_residual_budget(capsule)


@pytest.mark.parametrize("change", ["missing", "extra", "token_semantics", "cost_conversion",
    "hard_cap", "remaining", "authorization", "observed_usage", "false_as_zero", "enforcement_zero",
    "enforcement_true", "extra_enforcement", "limit_boolean"])
def test_untrusted_budget_observation_cannot_claim_enforcement_or_usage(change):
    value = budget.residual_budget_observation(_capsule())
    if change == "missing":
        del value["remaining_budget_known"]
    elif change == "extra":
        value["admitted"] = True
    elif change == "token_semantics":
        value["token_accounting"] = "provider-total-tokens"
    elif change == "cost_conversion":
        value["provider_cost_unit_conversion"] = True
    elif change == "hard_cap":
        value["hard_native_token_cap"] = True
    elif change == "remaining":
        value["remaining_budget_known"] = True
    elif change == "authorization":
        value["additional_coding_call_authorized"] = True
    elif change == "observed_usage":
        value["observed_usage"] = {"tokens": 0}
    elif change == "false_as_zero":
        value["hard_native_token_cap"] = 0
    elif change == "enforcement_zero":
        value["coding_route_enforcement"]["tokens"] = 0
    elif change == "enforcement_true":
        value["coding_route_enforcement"]["tokens"] = True
    elif change == "extra_enforcement":
        value["coding_route_enforcement"]["other"] = False
    else:
        value["limits"]["max_calls"] = True
    with pytest.raises(ValueError):
        budget.validate_budget_observation(value)


def _rewrite(request, payload):
    payload["context_cid"] = content_identity({key: value for key, value in payload.items()
                                               if key != "context_cid"})
    raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    request["artifact"].chmod(0o600)
    request["artifact"].write_bytes(raw)
    request["artifact"].chmod(0o444)
    return {**request, "expected_sha256": hashlib.sha256(raw).hexdigest()}


def test_real_artifact_preserves_legacy_input_and_keeps_verbose_observation_out_of_model(retained):
    _, _, _, request = retained
    payload = json.loads(request["artifact"].read_bytes())
    assert payload["schema"] == context.SCHEMA
    assert payload["budget_observation"] == budget.residual_budget_observation(payload["capsule"])
    current_text, receipt = context.load_doctor_residual_advisory(**request)
    assert receipt["schema"] == "doctor-residual-router-context@2"
    assert receipt["residual_budget"] == LIMITS
    assert receipt["budget_observation"] == payload["budget_observation"]
    assert "coding_route_enforcement" not in current_text
    prefix, current_wire = current_text.rsplit("\n", 1)
    current_advisory = json.loads(current_wire)
    assert current_advisory.pop("declared_residual_proposal_budget") == LIMITS
    legacy = copy.deepcopy(payload)
    legacy["schema"] = context.LEGACY_SCHEMA
    del legacy["budget_observation"]
    legacy_request = _rewrite(request, legacy)
    legacy_text, legacy_receipt = context.load_doctor_residual_advisory(**legacy_request)
    current_advisory.update(schema="doctor-residual-coding-advisory@1", context_cid=legacy["context_cid"])
    assert legacy_text == prefix + "\n" + json.dumps(current_advisory, sort_keys=True, separators=(",", ":"))
    assert legacy_receipt["schema"] == "doctor-residual-router-context@1"
    assert "residual_budget" not in legacy_receipt and "budget_observation" not in legacy_receipt
    historical, historical_receipt = context.load_doctor_residual_advisory(
        **{**legacy_request, "workspace": None, "require_current_source": False})
    assert historical == legacy_text
    assert historical_receipt == {**legacy_receipt, "source_freshness_verified": False, "historical_replay": True}
    # The only model-input growth is one compact declared-budget object; all
    # enforcement/capability detail remains in the owner artifact and receipt.
    added_bytes = len(current_text.encode()) - len(legacy_text.encode())
    assert added_bytes == len(',"declared_residual_proposal_budget":' + json.dumps(LIMITS, sort_keys=True, separators=(",", ":")))
    assert added_bytes < 128


def test_native_loader_rejects_rehashed_budget_and_capability_tampering(retained):
    _, _, _, request = retained
    original = json.loads(request["artifact"].read_bytes())
    cases = []
    for key in budget.LIMIT_FIELDS:
        payload = copy.deepcopy(original)
        payload["capsule"]["residual_budget"][key] += 1
        cases.append(payload)
    for key in ("hard_native_token_cap", "provider_cost_unit_conversion", "remaining_budget_known",
                "additional_coding_call_authorized"):
        payload = copy.deepcopy(original)
        payload["budget_observation"][key] = True
        cases.append(payload)
    payload = copy.deepcopy(original)
    payload["budget_observation"]["coding_route_enforcement"]["tokens"] = 0
    cases.append(payload)
    payload = copy.deepcopy(original)
    del payload["budget_observation"]
    cases.append(payload)
    for payload in cases:
        rewritten = _rewrite(request, payload)
        with pytest.raises(ValueError):
            context.load_doctor_residual_advisory(**rewritten)
    context.load_doctor_residual_advisory(**_rewrite(request, original))
