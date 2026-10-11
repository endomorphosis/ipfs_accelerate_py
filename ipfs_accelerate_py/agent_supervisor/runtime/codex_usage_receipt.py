"""Recover native cumulative usage for an exact local CLI thread and cwd."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .codex_interaction_telemetry import build_interaction_telemetry, validated_cumulative_usage


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
    a lower bound on observed usage, never proof of a complete billed total.
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
        for line in lines[1:]:
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):
                malformed += 1
                continue
            if not isinstance(record, dict) or type(record.get("type")) is not str:
                malformed += 1
                continue
            # A second metadata boundary makes exact thread attribution
            # ambiguous, even when the first line matched the requested cwd.
            if record.get("type") == "session_meta":
                return None
            records.append(record)
            payload = record.get("payload", {})
            if record.get("type") == "turn_context":
                completed = False
            if record.get("type") in ("event_msg", "response_item") and (
                    not isinstance(payload, dict) or not isinstance(payload.get("type"), str)):
                malformed += 1
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
        complete = (completed and malformed == 0 and latest_usage_snapshot_valid is not False
                    and not counter_regression_observed)
        matches.append({
                "schema": "native-codex-cumulative-usage@1", "thread_id": thread_id,
                "rollout_sha256": hashlib.sha256(raw).hexdigest(),
                "usage": usage or {}, "usage_available": usage is not None,
                "observed_token_count_records": count,
                "latest_usage_snapshot_valid": latest_usage_snapshot_valid,
                "counter_regression_observed": counter_regression_observed,
                "tool_diagnostics": _tool_diagnostics(records, completed=complete, malformed=malformed),
                "interaction_telemetry": build_interaction_telemetry(
                    records, completed=complete, malformed=malformed),
                "task_complete_event_observed": completion_event_seen,
                "task_complete_observed": complete, "billing_total_verified": False,
                "cache_included_in_input": True, "dollar_cost": None,
            })
    return matches[0] if len(matches) == 1 else None
