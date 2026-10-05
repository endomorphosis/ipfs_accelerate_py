"""Project only final native Grok JSON counters, never nested tool/model text."""
from __future__ import annotations

import hashlib
import json

_FIELDS = {
    "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
    "cached_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cached_input_tokens"),
    "cache_write_input_tokens": ("cache_creation_input_tokens", "cache_write_input_tokens", "cacheWriteInputTokens"),
    "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
    "reasoning_output_tokens": ("reasoning_tokens", "reasoningTokens", "reasoning_output_tokens"),
    "total_tokens": ("total_tokens", "totalTokens"),
}


def native_grok_observation(payload: object, *, exit_code: int | None, timed_out: bool = False) -> dict:
    """Return bounded scalar metadata without reconstructing absent categories."""
    if type(payload) is not dict:
        return {}
    usage = payload.get("usage")
    if type(usage) is not dict:
        return {}
    result = {}
    for name, aliases in _FIELDS.items():
        values = [usage[key] for key in aliases if key in usage]
        if (values and all(type(value) is int and value >= 0 for value in values)
                and len(set(values)) == 1):
            result["native_grok_" + name] = values[0]
    if not result:
        return {}
    result["native_grok_usage_observed"] = True
    # A successful process does not independently prove a complete billed total.
    stop_reason = payload.get("stopReason", payload.get("stop_reason"))
    result["native_grok_task_complete"] = (
        type(exit_code) is int and exit_code == 0 and not timed_out
        and payload.get("type") != "error"
        and type(stop_reason) is str and stop_reason in {"end_turn", "stop"})
    if type(payload.get("usage_is_incomplete")) is bool:
        result["native_grok_usage_complete"] = not payload["usage_is_incomplete"]
    result["native_grok_usage_sha256"] = hashlib.sha256(
        json.dumps({k: result[k] for k in sorted(result)}, sort_keys=True).encode()).hexdigest()
    return result


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
