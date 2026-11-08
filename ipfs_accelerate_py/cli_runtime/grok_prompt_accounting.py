"""Bounded diagnostics for Grok's final prompt ledger, never a second charge.

The native headless reference includes the prompt and completed subagents in
this ledger and excludes compaction and other side-model calls. Model rows and
the final usage envelope overlap; neither model calls nor main rounds are HTTP
request counts. This projection does not establish a bill or invocation total.
"""
from __future__ import annotations

import hashlib
import json
from itertools import islice

SCHEMA = "native-grok-prompt-accounting@1"
MAX_COUNT = (1 << 63) - 1
MAX_MODEL_ROWS = 32
MAX_MODEL_ID_CHARS = 256
MAX_PROJECTION_BYTES = 65_536
TICKS_PER_USD = 10_000_000_000

_ALIASES = {
    "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
    "cached_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cached_input_tokens"),
    "cache_write_input_tokens": ("cache_creation_input_tokens", "cache_write_input_tokens",
                                "cacheWriteInputTokens", "cacheCreationInputTokens"),
    "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
    "reasoning_output_tokens": ("reasoning_tokens", "reasoningTokens", "reasoning_output_tokens"),
    "total_tokens": ("total_tokens", "totalTokens"),
    "model_calls": ("modelCalls", "model_calls"),
}
_COVERAGE = {
    "scope": "native_final_prompt_ledger",
    "includes_prompt": True,
    "includes_completed_subagents": True,
    "includes_compaction": False,
    "includes_side_model_calls": False,
    "model_roles": "unclassified",
    "num_turns_scope": "main_agent_rounds_with_usage",
    "model_calls_scope": "prompt_ledger_model_calls_including_subagents",
    "num_turns_is_http_request_count": False,
    "model_calls_are_http_request_count": False,
    "primary_http_requests": None,
    "auxiliary_http_requests": None,
    "primary_tokens": None,
    "auxiliary_tokens": None,
    "whole_invocation_tokens": None,
    "whole_invocation_complete": None,
    "billing_total_verified": False,
}
_CONSTANTS = {
    "schema": SCHEMA,
    "scope": "native_final_prompt_ledger",
    "observation_only": True,
    "raw_provider_data_exported": False,
    "billing_total_verified": False,
    "summation_permitted": False,
}
_KEYS = frozenset((*_CONSTANTS, "payload_container", "num_turns", "usage_is_incomplete",
                   "cost_is_partial", "cost", "model_usage", "coverage", "projection_sha256"))
_MODEL_KEYS = frozenset({"container", "observed_rows", "retained_rows", "omitted_rows",
    "truncated", "malformed_rows", "invalid_identity_rows", "invalid_counter_fields",
    "conflicting_counter_fields", "missing_counter_fields", "rows"})
_ROW_KEYS = frozenset({"ordinal", "model_id_sha256", "identity_status", "container", "counters"})
_SCALAR_STATUSES = frozenset({"missing", "valid", "invalid"})
_COUNTER_STATUSES = _SCALAR_STATUSES | {"conflict"}


def _need(condition):
    if not condition:
        raise ValueError("invalid Grok prompt accounting projection")


def _count(value):
    return type(value) is int and 0 <= value <= MAX_COUNT


def _mapping(value, keys):
    # Check cardinality before iterating untrusted mappings or their keys.
    return (type(value) is dict and len(value) == len(keys)
            and all(type(key) is str for key in value) and set(value) == keys)


def _enum(value, choices):
    return type(value) is str and len(value) <= 16 and value in choices


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _scalar(source, key, boolean=False):
    if key not in source:
        return {"status": "missing", "value": None}
    value = source[key]
    valid = type(value) is bool if boolean else _count(value)
    return {"status": "valid" if valid else "invalid", "value": value if valid else None}


def _counter(source, names):
    values = [source[name] for name in names if name in source]
    if not values:
        return {"status": "missing", "value": None}
    if not all(_count(value) for value in values):
        return {"status": "invalid", "value": None}
    if any(value != values[0] for value in values):
        return {"status": "conflict", "value": None}
    return {"status": "valid", "value": values[0]}


def _model_identity(value):
    if (type(value) is not str or not 1 <= len(value) <= MAX_MODEL_ID_CHARS
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        return None
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        return None
    return hashlib.sha256(encoded).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def grok_prompt_accounting_coverage() -> dict:
    """Return detached, fixed coverage facts; unknown side calls stay unknown."""
    return dict(_COVERAGE)


def project_grok_prompt_accounting(payload) -> dict:
    """Read selected fields and at most 32 rows; never walk arbitrary payloads."""
    data = payload if type(payload) is dict else {}
    turns = _scalar(data, "num_turns")
    incomplete = _scalar(data, "usage_is_incomplete", boolean=True)
    partial = _scalar(data, "cost_is_partial", boolean=True)
    ticks = _scalar(data, "total_cost_usd_ticks")
    usable = (ticks["value"] if ticks["status"] == "valid"
              and incomplete["status"] == partial["status"] == "valid"
              and incomplete["value"] is partial["value"] is False else None)
    native = data.get("modelUsage")
    container = ("missing" if "modelUsage" not in data else
                 "mapping" if type(native) is dict else "invalid")
    observed = len(native) if container == "mapping" else None
    rows = []
    # Dictionary length is constant-time. Neither omitted keys/values nor row
    # subtrees are traversed, stringified, serialized or hashed.
    if container == "mapping":
        for ordinal, (model, value) in enumerate(islice(native.items(), MAX_MODEL_ROWS), 1):
            identity = _model_identity(model)
            counters = value if type(value) is dict else {}
            rows.append({"ordinal": ordinal, "model_id_sha256": identity,
                "identity_status": "valid" if identity is not None else "invalid",
                "container": "mapping" if type(value) is dict else "invalid",
                "counters": {name: _counter(counters, aliases) for name, aliases in _ALIASES.items()}})
    model_usage = {"container": container, "observed_rows": observed,
        "retained_rows": len(rows), "omitted_rows": observed - len(rows) if observed is not None else None,
        "truncated": observed is not None and observed > len(rows),
        "malformed_rows": sum(row["container"] == "invalid" for row in rows),
        "invalid_identity_rows": sum(row["identity_status"] == "invalid" for row in rows),
        "invalid_counter_fields": sum(counter["status"] == "invalid" for row in rows for counter in row["counters"].values()),
        "conflicting_counter_fields": sum(counter["status"] == "conflict" for row in rows for counter in row["counters"].values()),
        "missing_counter_fields": sum(counter["status"] == "missing" for row in rows for counter in row["counters"].values()),
        "rows": rows}
    result = {**_CONSTANTS,
        "payload_container": "mapping" if type(payload) is dict else "missing" if payload is None else "invalid",
        "num_turns": turns, "usage_is_incomplete": incomplete, "cost_is_partial": partial,
        "cost": {"status": ticks["status"], "observed_usd_ticks": ticks["value"],
                 "usable_usd_ticks": usable, "ticks_per_usd": TICKS_PER_USD},
        "model_usage": model_usage, "coverage": grok_prompt_accounting_coverage()}
    encoded = _wire(result)
    _need(len(encoded) <= MAX_PROJECTION_BYTES)
    result["projection_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _validate_scalar(value, boolean=False, counter=False):
    _need(_mapping(value, frozenset({"status", "value"})))
    _need(_enum(value["status"], _COUNTER_STATUSES if counter else _SCALAR_STATUSES))
    if value["status"] == "valid":
        _need(type(value["value"]) is bool if boolean else _count(value["value"]))
    else:
        _need(value["value"] is None)


def validate_grok_prompt_accounting(value) -> dict:
    """Validate closed bounded structure and digest before returning a copy."""
    _need(_mapping(value, _KEYS))
    for key, expected in _CONSTANTS.items():
        _need(type(value[key]) is type(expected) and value[key] == expected)
    _need(_enum(value["payload_container"], {"missing", "mapping", "invalid"}))
    _need(_mapping(value["coverage"], frozenset(_COVERAGE)))
    for key, expected in _COVERAGE.items():
        _need(type(value["coverage"][key]) is type(expected) and value["coverage"][key] == expected)
    _validate_scalar(value["num_turns"])
    _validate_scalar(value["usage_is_incomplete"], boolean=True)
    _validate_scalar(value["cost_is_partial"], boolean=True)
    cost = value["cost"]
    _need(_mapping(cost, frozenset({"status", "observed_usd_ticks", "usable_usd_ticks", "ticks_per_usd"})))
    _need(_enum(cost["status"], _SCALAR_STATUSES))
    _need(type(cost["ticks_per_usd"]) is int and cost["ticks_per_usd"] == TICKS_PER_USD)
    _need(_count(cost["observed_usd_ticks"]) if cost["status"] == "valid" else cost["observed_usd_ticks"] is None)
    usable = (cost["observed_usd_ticks"] if cost["status"] == "valid"
              and value["usage_is_incomplete"]["status"] == value["cost_is_partial"]["status"] == "valid"
              and value["usage_is_incomplete"]["value"] is value["cost_is_partial"]["value"] is False else None)
    _need(type(cost["usable_usd_ticks"]) is type(usable) and cost["usable_usd_ticks"] == usable)
    models = value["model_usage"]
    _need(_mapping(models, _MODEL_KEYS))
    _need(_enum(models["container"], {"missing", "mapping", "invalid"}))
    _need(_count(models["retained_rows"]) and models["retained_rows"] <= MAX_MODEL_ROWS)
    rows = models["rows"]
    _need(type(rows) is list and len(rows) == models["retained_rows"])
    if models["container"] == "mapping":
        _need(_count(models["observed_rows"]) and _count(models["omitted_rows"]))
        _need(models["retained_rows"] == min(models["observed_rows"], MAX_MODEL_ROWS))
        _need(models["omitted_rows"] == models["observed_rows"] - models["retained_rows"])
    else:
        _need(models["observed_rows"] is models["omitted_rows"] is None and not rows)
    _need(type(models["truncated"]) is bool and models["truncated"] is (models["omitted_rows"] is not None and models["omitted_rows"] > 0))
    for ordinal, row in enumerate(rows, 1):
        _need(_mapping(row, _ROW_KEYS))
        _need(type(row["ordinal"]) is int and row["ordinal"] == ordinal)
        _need(_enum(row["identity_status"], {"valid", "invalid"}))
        _need(_sha(row["model_id_sha256"]) if row["identity_status"] == "valid" else row["model_id_sha256"] is None)
        _need(_enum(row["container"], {"mapping", "invalid"}))
        _need(_mapping(row["counters"], frozenset(_ALIASES)))
        for counter in row["counters"].values():
            _validate_scalar(counter, counter=True)
            if row["container"] == "invalid":
                _need(counter["status"] == "missing")
    expected_counts = {
        "malformed_rows": sum(row["container"] == "invalid" for row in rows),
        "invalid_identity_rows": sum(row["identity_status"] == "invalid" for row in rows),
        "invalid_counter_fields": sum(c["status"] == "invalid" for row in rows for c in row["counters"].values()),
        "conflicting_counter_fields": sum(c["status"] == "conflict" for row in rows for c in row["counters"].values()),
        "missing_counter_fields": sum(c["status"] == "missing" for row in rows for c in row["counters"].values()),
    }
    for key, expected in expected_counts.items():
        _need(_count(models[key]) and models[key] == expected)
    if value["payload_container"] != "mapping":
        _need(all(value[key]["status"] == "missing" for key in ("num_turns", "usage_is_incomplete", "cost_is_partial")))
        _need(cost["status"] == models["container"] == "missing")
    _need(_sha(value["projection_sha256"]))
    body = {key: item for key, item in value.items() if key != "projection_sha256"}
    encoded = _wire(body)
    _need(len(encoded) <= MAX_PROJECTION_BYTES)
    _need(hashlib.sha256(encoded).hexdigest() == value["projection_sha256"])
    return json.loads(_wire(value))


def grok_prompt_accounting_receipt(observation) -> dict | None:
    """Extract only a validated detached sibling; invalid evidence stays absent."""
    if type(observation) is not dict or "native_grok_prompt_accounting" not in observation:
        return None
    try:
        return validate_grok_prompt_accounting(observation["native_grok_prompt_accounting"])
    except (ValueError, TypeError, RecursionError, OverflowError):
        return None
