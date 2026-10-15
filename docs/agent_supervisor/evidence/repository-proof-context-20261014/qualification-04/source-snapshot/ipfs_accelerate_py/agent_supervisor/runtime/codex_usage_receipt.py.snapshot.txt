"""Recover native cumulative usage for an exact local CLI thread and cwd."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .codex_interaction_telemetry import build_interaction_telemetry, validated_cumulative_usage
from .native_context_delivery import build_native_context_delivery, unavailable_native_context_delivery


_OUTPUT_INDICATORS = {
    "permission_denied": re.compile(r"permission denied", re.I),
    "read_only_filesystem": re.compile(r"read.only file system", re.I),
    "operation_not_permitted": re.compile(r"operation not permitted", re.I),
    "sandbox_failure": re.compile(
        r"(?:sandbox|bwrap|bubblewrap)[^\n]{0,160}(?:fail|denied|error)|"
        r"(?:fail|denied|error)[^\n]{0,160}(?:sandbox|bwrap|bubblewrap)", re.I),
    "tool_protocol_error": re.compile(
        r"unknown tool|tool[^\n]{0,80}not found|is not a function|"
        r"is not defined|SyntaxError|ReferenceError|TypeError", re.I),
    "tool_host_unavailable": re.compile(
        r"failed to spawn code-mode host|codex-code-mode-host[^\n]{0,256}"
        r"(?:no such file or directory|not found|cannot open)", re.I),
}
_ABSOLUTE_APP = re.compile(r"(?<![A-Za-z0-9_./-])/app(?=/|[\s'\"`),;}\]]|$)")
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
_USAGE_OPTIONAL = ("cache_write_input_tokens", "reasoning_output_tokens")
MAX_USAGE_GROWTH_RECORDS = 128
MAX_USAGE_GROWTH_BYTES = 32_768


def _usage_categories(value):
    """Closed numeric projection; missing and invalid categories stay unknown."""
    value = value if isinstance(value, dict) else {}
    selected = {key: value[key] if type(value.get(key)) is int
                and 0 <= value[key] < (1 << 63) else None
                for key in (*_USAGE_KEYS, *_USAGE_OPTIONAL)}
    valid = all(selected[key] is not None for key in _USAGE_KEYS)
    valid = valid and (selected["total_tokens"] == selected["input_tokens"] + selected["output_tokens"]
                       and selected["cached_input_tokens"] <= selected["input_tokens"])
    valid = valid and all(key not in value or selected[key] is not None for key in _USAGE_OPTIONAL)
    if selected["reasoning_output_tokens"] is not None and selected["output_tokens"] is not None:
        valid = valid and selected["reasoning_output_tokens"] <= selected["output_tokens"]
    return selected, valid


class _UsageGrowth:
    """Bounded consecutive-snapshot observations, never reconstructed requests."""

    def __init__(self):
        self.previous = None
        self.rows = []
        self.observed = 0
        self.row_bytes = 2  # JSON list brackets.
        self.status_counts = {}
        self.last_snapshot_valid = None
        self.unreadable_record_gaps = 0

    def record_gap(self):
        self.previous = None
        self.unreadable_record_gaps += 1

    def observe(self, info):
        self.observed += 1
        info = info if isinstance(info, dict) else {}
        current, valid = _usage_categories(info.get("total_token_usage"))
        last, last_valid = _usage_categories(info.get("last_token_usage"))
        delta = None
        if not valid:
            status = "missing_or_invalid_cumulative_usage"
        elif self.previous is None:
            status = "initial_baseline_unobserved" if self.observed == 1 else "previous_baseline_unavailable"
        elif any(current[k] < self.previous[k] for k in current
                 if current[k] is not None and self.previous[k] is not None):
            status = "nonmonotone_cumulative_usage"
        else:
            candidate = {k: current[k] - self.previous[k]
                         if current[k] is not None and self.previous[k] is not None else None
                         for k in current}
            # Cache reclassification can make an otherwise monotone difference
            # incoherent. Do not invent negative uncached input or a request.
            coherent = (candidate["cached_input_tokens"] <= candidate["input_tokens"]
                        and (candidate["reasoning_output_tokens"] is None
                             or candidate["reasoning_output_tokens"] <= candidate["output_tokens"]))
            if not coherent:
                status = "inconsistent_usage_delta"
            else:
                delta = candidate
                status = ("repeated_snapshot" if current == self.previous
                          else "observed_consecutive_delta")
        self.last_snapshot_valid = valid
        self.previous = current if valid and status not in {
            "nonmonotone_cumulative_usage", "inconsistent_usage_delta"} else None
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        row = {
            "token_event_index": self.observed, "status": status,
            "cumulative_usage": current, "reported_last_usage": last,
            "reported_last_usage_valid": last_valid, "delta_usage": delta,
            "cumulative_uncached_input_tokens": current["input_tokens"] - current["cached_input_tokens"] if valid else None,
            "delta_uncached_input_tokens": delta["input_tokens"] - delta["cached_input_tokens"] if delta is not None else None,
            "reported_last_matches_delta": (all(last[k] == delta[k] for k in last if last[k] is not None)
                                            if last_valid and delta is not None
                                            and all(delta[k] is not None for k in last if last[k] is not None) else None),
        }
        size = len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        if (self.observed == len(self.rows) + 1 and len(self.rows) < MAX_USAGE_GROWTH_RECORDS
                and self.row_bytes + size + bool(self.rows) <= MAX_USAGE_GROWTH_BYTES):
            self.row_bytes += size + bool(self.rows)
            self.rows.append(row)

    def receipt(self):
        return {
            "schema": "native-codex-usage-growth@1", "observed_token_events": self.observed,
            "retained_token_events": len(self.rows), "omitted_token_events": self.observed - len(self.rows),
            "truncated": self.observed != len(self.rows), "max_retained_records": MAX_USAGE_GROWTH_RECORDS,
            "max_retained_rows_bytes": MAX_USAGE_GROWTH_BYTES, "retained_rows_canonical_bytes": self.row_bytes,
            "status_counts": self.status_counts, "final_snapshot_valid": self.last_snapshot_valid,
            "cumulative_sequence_consistent": self.sequence_consistent(),
            "unreadable_record_gaps": self.unreadable_record_gaps,
            "records": self.rows, "cache_included_in_input": True,
            "scope": "Consecutive observed cumulative snapshots; events are not necessarily distinct model requests. Initial or interrupted baselines are not zero.",
            "raw_content_exported": False, "billing_total_verified": False,
        }

    def sequence_consistent(self):
        return not any(self.status_counts.get(key, 0) for key in (
            "nonmonotone_cumulative_usage", "inconsistent_usage_delta"))


def _tool_output_bytes(value):
    try:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, (dict, list)):
            return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except (UnicodeError, ValueError, TypeError, RecursionError):
        pass
    return None


def _tool_text(value) -> str | None:
    """Interpret only recorded tool payloads, without returning them in receipts."""
    if isinstance(value, str):
        # Function arguments/output may themselves be serialized JSON. Decode
        # escapes before classifying a canonical absolute-path reference.
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return value
        return json.dumps(decoded, ensure_ascii=False) if isinstance(decoded, (dict, list)) else value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return None


def _tool_diagnostics(records, *, completed: bool, malformed: int) -> dict:
    calls = outputs = app_calls = unknown_calls = unknown_outputs = 0
    output_bytes = unknown_output_bytes = 0
    observed = dict.fromkeys(_OUTPUT_INDICATORS, 0)
    for record in records:
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if not isinstance(kind, str):
            continue
        if kind in {"function_call", "custom_tool_call"}:
            calls += 1
            text = _tool_text(payload.get("arguments" if kind == "function_call" else "input"))
            if text is None:
                unknown_calls += 1
            elif _ABSOLUTE_APP.search(text):
                app_calls += 1
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            outputs += 1
            size = _tool_output_bytes(payload.get("output"))
            if size is None:
                unknown_output_bytes += 1
            else:
                output_bytes += size
            text = _tool_text(payload.get("output"))
            if text is None:
                unknown_outputs += 1
            else:
                for name, pattern in _OUTPUT_INDICATORS.items():
                    observed[name] += bool(pattern.search(text))
    calls_known = completed and malformed == 0 and unknown_calls == 0
    outputs_known = completed and malformed == 0 and unknown_outputs == 0 and outputs == calls
    return {
        "schema": "native-codex-tool-diagnostics@1",
        "observed_tool_call_events": calls, "observed_tool_output_events": outputs,
        "observed_calls_with_absolute_canonical_app_reference": app_calls,
        "absolute_canonical_app_reference_observed": True if app_calls else False if calls_known else None,
        "observed_output_indicator_counts": observed,
        "complete_output_indicator_counts": observed if outputs_known else None,
        "tool_event_counts_complete": calls_known and outputs_known,
        "unclassified_tool_calls": unknown_calls, "unclassified_tool_outputs": unknown_outputs,
        "observed_tool_output_utf8_bytes": output_bytes,
        "complete_tool_output_utf8_bytes": output_bytes if outputs_known and unknown_output_bytes == 0 else None,
        "outputs_without_byte_measurement": unknown_output_bytes,
        "tool_output_byte_scope": "Recorded UTF-8 strings or canonical JSON structured payloads; not tokenized or provider-input bytes.",
        "malformed_json_records": malformed,
        "classification_scope": "lexical indicators in native tool records; a path reference is not proof of a write",
        "root_cause_established": False, "raw_tool_data_exported": False,
    }


def _bounded_rollout(path: Path) -> bytes | None:
    """Read a bounded regular file; a substituted FIFO must not block cleanup."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 32_000_000:
                return None
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(32_000_001)
            if len(raw) > 32_000_000:
                return None
            return raw
        finally:
            os.close(fd)
    except OSError:
        return None


def recover_codex_usage(*, home: Path, thread_id: str, workspace: Path) -> dict | None:
    """Return recorded categories without counting cached input twice.

    Unknown usage stays unavailable. A final token_count before interruption is
    a numeric observation, never proof of a complete billed total.
    No prompts, model output or credentials are copied into the receipt.
    """
    if not re.fullmatch(r"[a-zA-Z0-9-]{16,80}", str(thread_id)):
        return None
    root = Path(home).resolve() / "sessions"
    paths = list(root.glob(f"*/*/*/*{thread_id}.jsonl")) if root.is_dir() else []
    matches = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        raw = _bounded_rollout(path)
        if raw is None:
            continue
        lines = raw.splitlines()
        if not lines:
            continue
        try:
            first = json.loads(lines[0])
        except (ValueError, RecursionError):
            continue
        if not isinstance(first, dict):
            continue
        meta = first.get("payload", {})
        if (first.get("type") != "session_meta" or not isinstance(meta, dict) or meta.get("id") != thread_id
                or Path(str(meta.get("cwd", ""))).resolve() != workspace.resolve()):
            continue
        usage = None
        completed = False
        completion_event_seen = False
        latest_usage_snapshot_valid = None
        counter_regression_observed = False
        previous_valid_usage = None
        count = 0
        records = []
        malformed = 0
        growth = _UsageGrowth()
        for line in lines[1:]:
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):
                malformed += 1
                growth.record_gap()
                continue
            if not isinstance(record, dict) or type(record.get("type")) is not str:
                malformed += 1
                growth.record_gap()
                continue
            # A second metadata boundary makes exact thread attribution
            # ambiguous, even when the first line matched the requested cwd.
            if record.get("type") == "session_meta":
                return None
            records.append(record)
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                malformed += 1
                growth.record_gap()
                continue
            if record.get("type") == "turn_context":
                completed = False
            if record.get("type") in ("event_msg", "response_item") and not isinstance(payload.get("type"), str):
                malformed += 1
                growth.record_gap()
                continue
            if record.get("type") == "response_item" and (
                    payload.get("type") in ("function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output")
                    or payload.get("type") == "message" and payload.get("role") == "user"):
                completed = False
            if record.get("type") != "event_msg" or not isinstance(payload, dict):
                continue
            if payload.get("type") == "task_complete":
                completed = completion_event_seen = True
            elif payload.get("type") in ("task_started", "turn_started", "turn_aborted", "task_aborted", "user_message"):
                completed = False
            if payload.get("type") == "token_count":
                completed = False
                info = payload.get("info")
                growth.observe(info)
                # Native quota updates may carry no cumulative snapshot.
                if info is None or (type(info) is dict and "total_token_usage" not in info):
                    continue
                selected, _ = validated_cumulative_usage(payload)
                latest_usage_snapshot_valid = selected is not None
                if selected is not None:
                    if previous_valid_usage is not None and any(
                            selected[key] < previous_valid_usage[key]
                            for key in selected.keys() & previous_valid_usage.keys()):
                        counter_regression_observed = True
                    previous_valid_usage = selected
                    usage = selected
                    count += 1
                elif type(info) is dict and type(info.get("total_token_usage")) is dict and usage is None:
                    # Preserve individually valid sparse failure observations,
                    # but never label an explicit partial snapshot complete.
                    usage = {key: value for key, value in info["total_token_usage"].items()
                             if key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                                        "output_tokens", "reasoning_output_tokens", "total_tokens")
                             and type(value) is int and 0 <= value < (1 << 63)} or None
        # Preserve useful failure observations even when a provider never
        # emits token_count. Absent categories stay absent (never zero).
        # Retain the previous numeric observation when the
        # terminal token event is malformed or missing its cumulative totals.
        # A later task_complete cannot promote those stale counts to complete.
        completed = (completed and malformed == 0
                     and (growth.observed == 0 or growth.last_snapshot_valid is True)
                     and growth.sequence_consistent())
        complete = (completed and malformed == 0 and latest_usage_snapshot_valid is not False
                    and not counter_regression_observed)
        telemetry = build_interaction_telemetry(records, completed=complete, malformed=malformed)
        try:
            delivery = build_native_context_delivery(records, telemetry=telemetry)
        except Exception:
            # Optional fingerprints must not suppress original native usage or
            # change its arithmetic, completion and isolation observations.
            delivery = unavailable_native_context_delivery(telemetry)
        matches.append({
                "schema": "native-codex-cumulative-usage@1", "thread_id": thread_id,
                "rollout_sha256": hashlib.sha256(raw).hexdigest(),
                "usage": usage or {}, "usage_available": usage is not None,
                "observed_token_count_records": count,
                "usage_growth": growth.receipt(),
                "latest_usage_snapshot_valid": latest_usage_snapshot_valid,
                "counter_regression_observed": counter_regression_observed,
                "tool_diagnostics": _tool_diagnostics(records, completed=complete, malformed=malformed),
                "interaction_telemetry": telemetry,
                "native_context_delivery": delivery,
                "task_complete_event_observed": completion_event_seen,
                "task_complete_observed": complete, "billing_total_verified": False,
                "cache_included_in_input": True, "dollar_cost": None,
            })
    return matches[0] if len(matches) == 1 else None
