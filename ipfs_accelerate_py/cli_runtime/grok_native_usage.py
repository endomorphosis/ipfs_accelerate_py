"""Project only final native Grok JSON counters, never nested tool/model text."""
from __future__ import annotations

import hashlib
import json
import re

_FIELDS = {
    "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
    "cached_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cached_input_tokens"),
    "cache_write_input_tokens": ("cache_creation_input_tokens", "cache_write_input_tokens", "cacheWriteInputTokens"),
    "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
    "reasoning_output_tokens": ("reasoning_tokens", "reasoningTokens", "reasoning_output_tokens"),
    "total_tokens": ("total_tokens", "totalTokens"),
}

# Grok 1.0.46's shipped headless reference documents these ACP/Messages
# boundaries. Unknown values remain unknown; provider text is never a reason.
_STOP_REASONS = frozenset({"end_turn", "stop", "max_tokens", "max_turn_requests",
                          "refusal", "cancelled", "tool_use", "pause_turn", "stop_sequence"})
_ERROR_SUBTYPES = frozenset({"error_max_turns", "error_during_execution",
                            "error_max_structured_output_retries"})
_PROCESS_OUTCOMES = frozenset({"returned", "failed", "timeout", "unknown"})
_REASON_CODES = frozenset({"end_turn", "max_turns", "max_tokens", "refusal", "cancelled",
                          "structured_output_retries", "execution_error", "process_error",
                          "timeout", "other_stop", "unknown"})
_CLASSIFICATION_SOURCES = frozenset({"stop_reason", "error_subtype", "native_type",
                                    "message_marker", "process", "none"})


def _closed_alias(payload: dict, names: tuple[str, ...], allowed: frozenset) -> str:
    values = [payload[name] for name in names if name in payload]
    if values and all(type(value) is str and value in allowed for value in values) and len(set(values)) == 1:
        return values[0]
    return "unknown"


def _outcome_observation(payload: object, *, exit_code: int | None, timed_out: bool) -> dict:
    envelope = type(payload) is dict
    data = payload if envelope else {}
    stop = _closed_alias(data, ("stopReason", "stop_reason"), _STOP_REASONS)
    subtype = _closed_alias(data, ("subtype",), _ERROR_SUBTYPES)
    raw_type = data.get("type")
    native_type = raw_type.lower() if type(raw_type) is str and len(raw_type) <= 64 else ""
    native_error = (native_type == "error" or data.get("is_error") is True
                    or subtype != "unknown" or native_type == "max_turns_reached")
    # The native plain-JSON error format can omit stopReason and subtype. These
    # exact 1.0.46 binary literals identify its turn guard without retaining any
    # error body. A marker in response text or nested model output is ignored.
    max_turns_marker = native_error and any(
        type(data.get(key)) is str and len(data[key]) <= 256 and re.fullmatch(
            r"(?:max turns reached(?: \(limit: [0-9]{1,6}\))?|Reached the maximum number of turns)",
            data[key]) is not None for key in ("message", "error"))
    process = ("timeout" if timed_out is True else "returned" if type(exit_code) is int and exit_code == 0
               else "failed" if type(exit_code) is int else "unknown")
    if process == "timeout":
        reason, source = "timeout", "process"
    elif stop == "max_turn_requests":
        reason, source = "max_turns", "stop_reason"
    elif subtype == "error_max_turns":
        reason, source = "max_turns", "error_subtype"
    elif native_type == "max_turns_reached":
        reason, source = "max_turns", "native_type"
    elif max_turns_marker:
        reason, source = "max_turns", "message_marker"
    elif stop in {"max_tokens", "refusal", "cancelled"}:
        reason, source = stop, "stop_reason"
    elif subtype == "error_max_structured_output_retries":
        reason, source = "structured_output_retries", "error_subtype"
    elif native_error:
        reason, source = "execution_error", "error_subtype" if subtype != "unknown" else "native_type"
    elif process == "failed":
        reason, source = "process_error", "process"
    elif process == "returned" and stop in {"end_turn", "stop"}:
        reason, source = "end_turn", "stop_reason"
    elif stop != "unknown":
        reason, source = "other_stop", "stop_reason"
    else:
        reason, source = "unknown", "none"
    return {"native_grok_envelope_observed": envelope, "native_grok_process_outcome": process,
            "native_grok_stop_reason": stop, "native_grok_error_subtype": subtype,
            "native_grok_error_observed": native_error, "native_grok_reason_code": reason,
            "native_grok_classification_source": source}


def grok_outcome_receipt(observation: dict) -> dict:
    """Retain bounded outcome evidence even when no token usage was emitted."""
    def enum(name, choices):
        value = observation.get("native_grok_" + name)
        return value if type(value) is str and value in choices else "unknown"
    return {"schema": "native-grok-outcome@1",
            "envelope_observed": observation.get("native_grok_envelope_observed") is True,
            "process_outcome": enum("process_outcome", _PROCESS_OUTCOMES),
            "stop_reason": enum("stop_reason", _STOP_REASONS),
            "error_subtype": enum("error_subtype", _ERROR_SUBTYPES),
            "error_observed": observation.get("native_grok_error_observed") is True,
            "reason_code": enum("reason_code", _REASON_CODES),
            "classification_source": enum("classification_source", _CLASSIFICATION_SOURCES),
            "raw_provider_data_exported": False, "completion_authority": False}


def native_grok_observation(payload: object, *, exit_code: int | None, timed_out: bool = False) -> dict:
    """Return bounded scalar metadata without reconstructing absent categories."""
    outcome = _outcome_observation(payload, exit_code=exit_code, timed_out=timed_out)
    if type(payload) is not dict:
        return outcome
    usage = payload.get("usage")
    if type(usage) is not dict:
        return outcome
    result = {}
    for name, aliases in _FIELDS.items():
        values = [usage[key] for key in aliases if key in usage]
        if (values and all(type(value) is int and value >= 0 for value in values)
                and len(set(values)) == 1):
            result["native_grok_" + name] = values[0]
    if not result:
        return outcome
    result["native_grok_usage_observed"] = True
    # A successful process does not independently prove a complete billed total.
    result["native_grok_task_complete"] = outcome["native_grok_reason_code"] == "end_turn"
    if type(payload.get("usage_is_incomplete")) is bool:
        result["native_grok_usage_complete"] = not payload["usage_is_incomplete"]
    result["native_grok_usage_sha256"] = hashlib.sha256(
        json.dumps({k: result[k] for k in sorted(result)}, sort_keys=True).encode()).hexdigest()
    return {**result, **outcome}


def grok_usage_receipt(observation: dict) -> dict | None:
    if observation.get("native_grok_usage_observed") is not True:
        return None
    usage = {key: observation["native_grok_" + key] for key in _FIELDS
             if type(observation.get("native_grok_" + key)) is int
             and observation["native_grok_" + key] >= 0}
    if not usage:
        return None
    return {"schema": "native-grok-final-usage@1", "usage": usage, "usage_available": True,
            "usage_observation_sha256": observation.get("native_grok_usage_sha256"),
            "task_complete_observed": observation.get("native_grok_task_complete") is True,
            "usage_complete_observed": observation.get("native_grok_usage_complete"),
            "totals_scope": "native_final_envelope",
            "billing_total_verified": False, "cache_included_in_input": False,
            "dollar_cost": None, "observed_final_usage_records": 1,
            "raw_provider_data_exported": False}
