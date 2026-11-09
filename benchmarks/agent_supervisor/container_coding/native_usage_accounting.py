"""Provider-scoped completeness for retained native usage, never billing proof."""
from __future__ import annotations


def _boolean(value):
    return value if type(value) is bool else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _completion(values):
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def native_usage_contract(raw: dict, native: dict) -> tuple[bool | None, str | None, bool | None]:
    """Return (usage complete, native count scope, cache included in input).

    Grok's sparse omitted completeness flag remains unknown: final transport
    success alone does not establish complete accounting. The final-envelope
    scope does not cover every side-model call or establish a complete bill.
    Codex's cumulative session contract retains its native completion signal.
    """
    raw = raw if isinstance(raw, dict) else {}
    native = native if isinstance(native, dict) else {}
    provider, schema = raw.get("provider"), native.get("schema")
    session_complete = _boolean(native.get("task_complete_observed"))
    if (provider in (None, "codex_cli") and schema == "native-codex-cumulative-usage@1"
            and native.get("cache_included_in_input", True) is True):
        return session_complete, "native_session_cumulative", True
    if provider in (None, "grok_cli") and schema == "native-grok-final-usage@1":
        usage = native.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        values = [_count(usage.get(key)) for key in
                  ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "total_tokens")]
        valid = (native.get("usage_available") is True
            and native.get("totals_scope") == "native_final_envelope"
            and native.get("cache_included_in_input") is False
            and all(value is not None for value in values) and sum(values[:4]) == values[4])
        complete = _completion([session_complete, _boolean(native.get("usage_complete_observed"))]) if valid else None
        return complete, "native_final_envelope", _boolean(native.get("cache_included_in_input"))
    return None, None, None


_COLLECTION_COUNTERS = ("entries_examined", "log_files_seen", "log_files_read", "bytes_read", "lines_examined",
    "provider_observations", "doctor_observations", "oversized_logs", "unreadable_logs", "unstable_logs",
    "nonregular_logs", "directory_errors", "malformed_json_candidates", "oversized_lines", "bound_exceeded", "collection_errors")
_COLLECTION_REASONS = frozenset({"entry_limit", "file_limit", "byte_limit", "line_limit", "observation_limit",
    "oversized_log", "unreadable_log", "unstable_log", "nonregular_log", "directory_error",
    "malformed_json_candidate", "oversized_line", "collection_error"})
_COLLECTION_FALSE_FIELDS = ("provider_usage_authenticity_verified", "provider_dispatch_count_verified",
    "token_savings_measured", "completion_authority", "execution_authority")
_COLLECTION_UNKNOWN_FIELDS = ("attempt_population_complete", "provider_call_count", "campaign_call_population_complete")
_COLLECTION_FIELDS = frozenset({"schema", "status", "complete", "counts", "reason_codes", "scope", "observation_only",
    *_COLLECTION_FALSE_FIELDS, *_COLLECTION_UNKNOWN_FIELDS})


def provider_receipt_collection_observation(value):
    """Project reported scan diagnostics; a clean scan proves no census."""
    if type(value) is not dict or set(value) != _COLLECTION_FIELDS:
        return None
    counts, reasons = value["counts"], value["reason_codes"]
    if (value["schema"] != "terminal-provider-receipt-collection@1"
            or type(value["status"]) is not str or value["status"] not in {"observed", "incomplete"}
            or type(value["complete"]) is not bool or (value["status"] == "observed") is not value["complete"]
            or value["scope"] != "retained_state_logs_only" or value["observation_only"] is not True
            or any(value[field] is not False for field in _COLLECTION_FALSE_FIELDS)
            or any(value[field] is not None for field in _COLLECTION_UNKNOWN_FIELDS)
            or type(counts) is not dict or set(counts) != set(_COLLECTION_COUNTERS)
            or type(reasons) is not list or len(reasons) > len(_COLLECTION_REASONS)
            or any(type(reason) is not str or reason not in _COLLECTION_REASONS for reason in reasons)
            or reasons != sorted(set(reasons))):
        return None
    maxima = dict(entries_examined=4097, log_files_seen=129, log_files_read=128, bytes_read=32_000_000,
        lines_examined=131073, provider_observations=512, doctor_observations=512)
    if (any(type(counts[field]) is not int or not 0 <= counts[field] <= maxima.get(field, 131073)
            for field in _COLLECTION_COUNTERS)
            or counts["log_files_read"] > counts["log_files_seen"]
            or counts["provider_observations"] + counts["doctor_observations"] > 512):
        return None
    if value["complete"]:
        if (reasons or any(counts[field] for field in _COLLECTION_COUNTERS[7:])
                or counts["log_files_seen"] != counts["log_files_read"]
                or counts["entries_examined"] > 4096 or counts["log_files_seen"] > 128
                or counts["lines_examined"] > 131072):
            return None
    elif not reasons:
        return None
    return {**value, "counts": dict(counts), "reason_codes": list(reasons)}
