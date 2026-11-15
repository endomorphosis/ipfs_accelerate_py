"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations

LIMIT_FIELDS = ('max_calls', 'max_tokens', 'max_rounds', 'max_cost_units')

def _limits(value):
    if type(value) is not dict or set(value) != set(LIMIT_FIELDS) or any((type(value[key]) is not int or not 1 <= value[key] < 1 << 63 for key in LIMIT_FIELDS)):
        raise ValueError('closed positive integer Doctor residual budget required')
    return {key: value[key] for key in LIMIT_FIELDS}

def validate_residual_budget(capsule):
    """Return a detached budget after checking both native capsule projections."""
    if type(capsule) is not dict:
        raise ValueError('native Doctor residual capsule required')
    budget = _limits(capsule.get('residual_budget'))
    counters = capsule.get('token_budget')
    expected = {'residual_' + key: value for key, value in budget.items()}
    if type(counters) is not dict or any((type(key) is not str for key in counters)) or {key for key in counters if key.startswith('residual_')} != set(expected) or any((type(counters.get(key)) is not int or counters[key] != value for key, value in expected.items())):
        raise ValueError('Doctor residual budget projections differ')
    return budget
