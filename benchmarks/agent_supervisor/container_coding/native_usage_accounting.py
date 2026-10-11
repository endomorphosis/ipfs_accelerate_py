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
