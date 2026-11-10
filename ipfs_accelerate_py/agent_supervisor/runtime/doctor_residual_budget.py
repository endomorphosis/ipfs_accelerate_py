"""Validate declared Doctor budgets without inventing coding-route enforcement.

The shared residual-proposal admission path charges response tokens and
caller-supplied cost units. The workspace-capable coding CLI path does not run
that charging path. Its generation kwargs establish neither a native token cap
nor a conversion from provider dollars to those cost units.
"""
from __future__ import annotations

SCHEMA = "doctor-residual-budget-observation@1"
LIMIT_FIELDS = ("max_calls", "max_tokens", "max_rounds", "max_cost_units")


def _limits(value):
    if (type(value) is not dict or set(value) != set(LIMIT_FIELDS)
            or any(type(value[key]) is not int or not 1 <= value[key] < (1 << 63)
                   for key in LIMIT_FIELDS)):
        raise ValueError("closed positive integer Doctor residual budget required")
    return {key: value[key] for key in LIMIT_FIELDS}


def validate_residual_budget(capsule):
    """Return a detached budget after checking both native capsule projections."""
    if type(capsule) is not dict:
        raise ValueError("native Doctor residual capsule required")
    budget = _limits(capsule.get("residual_budget"))
    counters = capsule.get("token_budget")
    expected = {"residual_" + key: value for key, value in budget.items()}
    if (type(counters) is not dict or any(type(key) is not str for key in counters)
            or {key for key in counters if key.startswith("residual_")} != set(expected)
            or any(type(counters.get(key)) is not int or counters[key] != value
                   for key, value in expected.items())):
        raise ValueError("Doctor residual budget projections differ")
    return budget


def _observation(limits):
    return {
        "schema": SCHEMA,
        "limits": dict(limits),
        "token_accounting": "residual-proposal-response-tokens",
        "call_accounting": "residual-proposal-admissions",
        "round_accounting": "residual-proposal-admissions",
        "cost_unit_accounting": "caller-supplied-residual-proposal-cost-units",
        "coding_route_enforcement": {
            "calls": False, "tokens": False, "rounds": False, "cost_units": False,
        },
        "hard_native_token_cap": False,
        "provider_cost_unit_conversion": False,
        "remaining_budget_known": False,
        "additional_coding_call_authorized": False,
        "observed_usage": None,
    }


def residual_budget_observation(capsule):
    """Describe declared limits; this observation grants no further call."""
    return _observation(validate_residual_budget(capsule))


def validate_budget_observation(value):
    """Validate an untrusted receipt without interpreting it as admission."""
    if type(value) is not dict:
        raise ValueError("closed Doctor residual budget observation required")
    expected = _observation(_limits(value.get("limits")))
    # Equality alone would accept integer zero in place of False.
    if (set(value) != set(expected)
            or any(type(value[key]) is not type(item) or value[key] != item
                   for key, item in expected.items())
            or type(value["coding_route_enforcement"]) is not dict
            or any(item is not False for item in value["coding_route_enforcement"].values())):
        raise ValueError("Doctor residual budget observation differs")
    return expected
