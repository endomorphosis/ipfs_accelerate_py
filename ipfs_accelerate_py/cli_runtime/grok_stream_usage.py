"""Bounded numeric observations from Grok ``streaming-json`` usage events.

The shipped headless reference describes a usage event at each model-response
boundary. Event ordinals are not authenticated turn identifiers, and neither
counter additivity nor cache semantics is inferred here. Final usage accounting
continues to use the separate terminal-envelope projector.
"""
from __future__ import annotations

import math

MAX_USAGE_EVENTS = 128
MAX_COUNT = (1 << 63) - 1
USAGE_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens")
_STOP_REASONS = frozenset({"end_turn", "stop", "max_tokens", "max_turn_requests",
                         "refusal", "cancelled", "tool_use", "pause_turn", "stop_sequence", "unknown"})
_STATUSES = frozenset({"missing", "valid", "partial", "invalid"})
_CONSTANTS = {"schema": "native-grok-usage-events@1", "row_limit": MAX_USAGE_EVENTS,
              "selection": "first_usage_events", "counter_semantics": "unverified",
              "ordinals_are_turn_ids": False, "summation_permitted": False,
              "billing_total_verified": False, "raw_provider_data_exported": False}
_KEYS = frozenset((*_CONSTANTS, "observed_events", "retained_events", "omitted_events", "rows"))
_ROW_KEYS = frozenset({"event_ordinal", "frame_ordinal", "elapsed_seconds", "after_final",
                       "stop_reason", "counter_status", "usage_container", "invalid_counter_fields", "usage"})


def _count(value):
    return type(value) is int and 0 <= value <= MAX_COUNT


def validate_grok_usage_events(value: object, *, observed_events: int, frames: int,
                               duration_seconds: float | None, final_event_seconds: float | None,
                               events_after_final: int) -> dict:
    """Validate a closed projection and return an independent nested copy."""
    if type(value) is not dict or set(value) != _KEYS:
        raise ValueError("invalid Grok usage event fields")
    for key, expected in _CONSTANTS.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("invalid Grok usage event constants")
    if any(not _count(value[key]) for key in ("observed_events", "retained_events", "omitted_events")):
        raise ValueError("invalid Grok usage event counts")
    if (value["observed_events"] != observed_events
            or observed_events > frames
            or value["retained_events"] != min(observed_events, MAX_USAGE_EVENTS)
            or value["omitted_events"] != observed_events - value["retained_events"]):
        raise ValueError("inconsistent Grok usage event counts")
    rows = value["rows"]
    if type(rows) is not list or len(rows) != value["retained_events"]:
        raise ValueError("invalid Grok usage event rows")
    copied = []
    last_frame = 0
    after_final_rows = 0
    for ordinal, row in enumerate(rows, 1):
        if type(row) is not dict or set(row) != _ROW_KEYS:
            raise ValueError("invalid Grok usage event row fields")
        if (not _count(row["event_ordinal"]) or row["event_ordinal"] != ordinal
                or not _count(row["frame_ordinal"]) or not last_frame < row["frame_ordinal"] <= frames):
            raise ValueError("invalid Grok usage event ordinal")
        elapsed = row["elapsed_seconds"]
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or duration_seconds is None or not 0 <= elapsed <= duration_seconds):
            raise ValueError("invalid Grok usage event time")
        if type(row["after_final"]) is not bool:
            raise ValueError("invalid Grok usage event boundary")
        if row["after_final"]:
            after_final_rows += 1
            if final_event_seconds is None or elapsed < final_event_seconds:
                raise ValueError("inconsistent Grok usage event final boundary")
        elif after_final_rows or final_event_seconds is not None and elapsed > final_event_seconds:
            raise ValueError("inconsistent Grok usage event final ordering")
        if after_final_rows > events_after_final:
            raise ValueError("inconsistent Grok usage event final count")
        if type(row["stop_reason"]) is not str or row["stop_reason"] not in _STOP_REASONS:
            raise ValueError("invalid Grok usage event stop reason")
        if type(row["counter_status"]) is not str or row["counter_status"] not in _STATUSES:
            raise ValueError("invalid Grok usage event status")
        container = row["usage_container"]
        if type(container) is not str or container not in {"missing", "mapping", "invalid"}:
            raise ValueError("invalid Grok usage event container")
        invalid = row["invalid_counter_fields"]
        usage = row["usage"]
        if (type(usage) is not dict or set(usage) != set(USAGE_FIELDS)
                or any(item is not None and not _count(item) for item in usage.values())
                or not _count(invalid) or invalid > len(USAGE_FIELDS)):
            raise ValueError("invalid Grok usage event counters")
        present = sum(item is not None for item in usage.values())
        if present + invalid > len(USAGE_FIELDS):
            raise ValueError("inconsistent Grok usage event counters")
        if container != "mapping" and (present or invalid):
            raise ValueError("inconsistent Grok usage event container")
        expected = (("partial" if invalid else "valid") if present
                    else ("invalid" if invalid or container == "invalid" else "missing"))
        if row["counter_status"] != expected:
            raise ValueError("inconsistent Grok usage event status")
        copied.append({**row, "usage": dict(usage)})
        last_frame = row["frame_ordinal"]
    return {**value, "rows": copied}


class GrokUsageEventObserver:
    """Retain the first bounded event rows without storing provider identifiers."""

    def __init__(self):
        self.observed_events = 0
        self.rows = []

    def observe(self, event: dict, *, frame_ordinal: int, elapsed_seconds: float,
                after_final: bool):
        self.observed_events = min(MAX_COUNT, self.observed_events + 1)
        if len(self.rows) >= MAX_USAGE_EVENTS:
            return
        raw = event.get("usage")
        container = "mapping" if type(raw) is dict else "missing" if raw is None else "invalid"
        usage = dict.fromkeys(USAGE_FIELDS)
        invalid = 0
        if type(raw) is dict:
            for field in USAGE_FIELDS:
                if field in raw:
                    if _count(raw[field]):
                        usage[field] = raw[field]
                    else:
                        invalid += 1
        present = any(item is not None for item in usage.values())
        status = (("partial" if invalid else "valid") if present
                  else ("invalid" if invalid or container == "invalid" else "missing"))
        reason = event.get("stopReason")
        self.rows.append({"event_ordinal": self.observed_events, "frame_ordinal": frame_ordinal,
            "elapsed_seconds": elapsed_seconds, "after_final": after_final,
            "stop_reason": reason if type(reason) is str and reason in _STOP_REASONS else "unknown",
            "counter_status": status, "usage_container": container,
            "invalid_counter_fields": invalid, "usage": usage})

    def snapshot(self) -> dict:
        return {**_CONSTANTS, "observed_events": self.observed_events, "retained_events": len(self.rows),
                "omitted_events": self.observed_events - len(self.rows),
                "rows": [{**row, "usage": dict(row["usage"])} for row in self.rows]}
