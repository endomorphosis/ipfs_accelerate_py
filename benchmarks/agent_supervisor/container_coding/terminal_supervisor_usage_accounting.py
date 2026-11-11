"""Pure session-safe bridge from supervisor receipts to benchmark counters.

Only the supplied receipt population is reconciled. Native cumulative snapshots
are selected once per session. Missing custody is retained as an unqualified
observation, never converted into a fresh session or a zero-cost call.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core

SCHEMA = "terminal-supervisor-usage-accounting@1"
MAX_ROWS = core.MAX_INVOCATIONS
MAX_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = core.MAX_INPUT_BYTES
MAX_NODES = 131072
MAX_DEPTH = 48
COUNTERS = core.COUNTERS
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _wire(value, maximum=MAX_BYTES):
    pending = [(value, 0)]
    nodes = characters = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "usage_structure_bound")
        if type(item) is str:
            characters += len(item)
        elif type(item) is dict:
            _need(len(item) + len(pending) <= MAX_NODES and all(type(k) is str for k in item),
                  "usage_structure_bound")
            characters += sum(len(k) for k in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(len(item) + len(pending) <= MAX_NODES, "usage_structure_bound")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            _need(item.bit_length() <= 64, "usage_integer_bound")
        elif type(item) is float:
            _need(math.isfinite(item), "usage_nonfinite_input")
        else:
            _need(item is None or type(item) is bool, "usage_non_json_input")
        _need(characters <= maximum, "usage_byte_bound")
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("usage_non_json_input") from error
    _need(len(raw) <= maximum, "usage_byte_bound")
    return raw


def _digest(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _marker(report):
    if "unreceipted_provider_attempt" not in report:
        return False
    value = report["unreceipted_provider_attempt"]
    if value is None or type(value) is bool:
        return value
    if type(value) in (dict, list) and value:
        _wire(value, MAX_ROW_BYTES)
        return True
    raise ValueError("invalid_unreceipted_provider_attempt_marker")


def _combined_marker(markers):
    return True if any(value is True for value in markers) else None if any(
        value is None for value in markers) else False


def _counts(native):
    usage = native.get("usage") if type(native) is dict else None
    return {key: value if type(value) is int and 0 <= value <= core.MAX_COUNTER else None
            for key in COUNTERS
            for value in [usage.get(key) if type(usage) is dict else None]}


def _custody_reason(receipt):
    for key in ("provider", "model"):
        value = receipt.get(key)
        if type(value) is not str or not _IDENTITY.fullmatch(value):
            return "missing_or_invalid_" + key
    native = receipt.get("native_rollout_usage")
    if native is None:
        return "native_usage_unavailable"
    if type(native) is not dict:
        return "invalid_native_usage"
    schema = native.get("schema")
    if schema == "native-codex-cumulative-usage@1":
        thread = native.get("thread_id")
        if type(thread) is not str or not re.fullmatch(r"[A-Za-z0-9-]{16,80}", thread):
            return "native_session_identity_unavailable"
        pin, records = native.get("rollout_sha256"), native.get("observed_token_count_records")
        if type(pin) is not str or not _SHA.fullmatch(pin) or type(records) is not int or records <= 0:
            return "native_session_custody_incomplete"
    elif schema == "native-grok-final-usage@1":
        pin, records = native.get("usage_observation_sha256"), native.get("observed_final_usage_records")
        if type(pin) is not str or not _SHA.fullmatch(pin) or type(records) is not int or records != 1:
            return "native_final_envelope_custody_incomplete"
    else:
        return "native_usage_schema_unsupported"
    return None


def _sum(values):
    if not values:
        return None
    total = sum(values)
    _need(total <= core.MAX_COUNTER, "usage_aggregate_counter_bound")
    return total


def _make_summary(inventory, observations, marker):
    _need(len(inventory) <= MAX_ROWS and len(observations) <= MAX_ROWS, "usage_population_bound")
    identities = {}
    for entry in inventory:
        identity, pin = entry["invocation_id"], entry["receipt_sha256"]
        if identity in identities:
            _need(identities[identity] == pin, "conflicting_provider_invocation_receipts")
        identities[identity] = pin
    native = core.summarize_invocations(observations, unreceipted_provider_attempt=marker)
    unsupported = [dict(invocation_id=entry["invocation_id"], receipt_sha256=entry["receipt_sha256"],
                        reason=entry["unsupported_reason"],
                        unqualified_native_counters=entry["unqualified_native_counters"])
                   for entry in inventory if entry["unsupported_reason"] is not None]
    complete = native["complete_supplied_native_usage"] and not unsupported
    sessions = native["sessions"]
    selected = [native["observations"][session["selected_observation_index"]]
                for session in sessions if session["selected_observation_index"] is not None]
    reported = [native["observations"][session["selected_observation_index"]]
                for session in sessions if session["selected_observation_index"] is not None
                and session["reported_counts"]["total_tokens"] is not None]
    schemas = {row["native_usage"]["schema"] for row in selected}
    scopes = {row["native_usage"]["scope"] for row in selected}
    scope = next(iter(scopes)) if len(scopes) == 1 and not unsupported else None
    cache = True if schemas == {"native-codex-cumulative-usage@1"} else False if schemas == {
        "native-grok-final-usage@1"} else None
    if unsupported:
        cache = None
    # Raw input categories differ between providers. Never silently add a
    # Codex cache-inclusive input count to a Grok fresh-input count.
    subtotal = {}
    for key in COUNTERS:
        values = [row["native_usage"]["counts"][key] for row in reported
                  if row["native_usage"]["counts"][key] is not None
                  and row["native_usage"]["status"] != "invalid"]
        subtotal[key] = None if key == "input_tokens" and len(schemas) > 1 else _sum(values)
    tokens = {}
    for key in COUNTERS:
        values = [row["native_usage"]["counts"][key] for row in selected]
        tokens[key] = (_sum(values) if complete and values and all(value is not None for value in values)
                       and not (key == "input_tokens" and len(schemas) > 1) else None)
    normalized = dict(native["complete_native_counts"]) if complete else dict.fromkeys(core.NORMALIZED_COUNTERS)
    endpoint_complete = bool(inventory) and marker is False and all(
        entry["endpoint_task_complete_observed"] is True for entry in inventory)
    states = [session["complete"] for session in sessions]
    usage_complete = (False if any(value is False for value in states) else True if complete
                      else None)
    exportable = (complete and scope is not None and type(cache) is bool
                  and all(type(tokens[key]) is int for key in
                          ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")))
    source = ("observed_native_cumulative_totals" if cache is True else
              "observed_native_final_totals" if cache is False else
              "observed_native_mixed_or_unknown_totals")
    result = dict(schema=SCHEMA, **tokens, tokens=tokens,
        normalized_tokens=normalized, known_token_subtotals=subtotal,
        known_normalized_counter_subtotals=native["known_native_counter_subtotals"],
        provider_calls=len(identities), supplied_receipt_count=len(inventory),
        duplicate_receipt_count=len(inventory) - len(identities),
        unreceipted_provider_attempt=marker, all_invocations_receipted=bool(inventory) and marker is False,
        observed_complete_sessions=endpoint_complete, native_usage_complete_observed=usage_complete,
        native_scoped_usage_complete_observed=usage_complete,
        usage_complete_observed=usage_complete, cache_included_in_input=cache,
        token_count_scope=scope, source=source,
        observed_counts_are_lower_bounds=bool(selected) and not unsupported and all(
            row["native_usage"]["scope"] == "native_session_cumulative"
            and row["native_usage"]["status"] != "invalid" for row in selected),
        harbor_usage_exportable=exportable, usage_accounting_complete=exportable,
        complete_supplied_native_usage=complete, unsupported_invocations=unsupported,
        native_usage_accounting=native, receipt_inventory=inventory,
        coverage=dict(scope="supplied supervisor receipts only", supported_receipts=len(observations),
                      unsupported_receipts=len(unsupported), supplied_receipts=len(inventory),
                      campaign_call_population_complete=None, auxiliary_calls_complete=None),
        billing_total_verified=False, dollar_cost=None, reported_cost_usd=None,
        known_reported_cost_subtotal_usd=None, input_observation_authenticity_verified=False,
        provider_ingestion_verified=False, token_savings_established=False,
        cumulative_snapshots_are_additive=False, raw_content_exported=False)
    result["summary_sha256"] = _digest(result)
    return result


def summarize_supervisor_usage(report):
    """Return bounded content-free accounting for one supplied supervisor report.

    An absent legacy missing-attempt marker means no missing attempt was
    reported in this supplied report. An explicit null preserves uncertainty.
    Neither case establishes campaign or auxiliary-call census completeness.
    """
    _need(type(report) is dict, "invalid_supervisor_report")
    rows = report.get("provider_invocations", [])
    _need(type(rows) is list and len(rows) <= MAX_ROWS, "usage_population_bound")
    marker = _marker(report)
    inventory, observations = [], []
    total_bytes = 0
    for receipt in rows:
        _need(type(receipt) is dict, "invalid_provider_invocation_receipt")
        identity = receipt.get("invocation_id")
        _need(type(identity) is str and _IDENTITY.fullmatch(identity), "invalid_provider_invocation_identity")
        raw = _wire(receipt, MAX_ROW_BYTES)
        total_bytes += len(raw)
        _need(total_bytes <= MAX_BYTES, "usage_aggregate_byte_bound")
        # Project exactly the bounded bytes covered by the complete-row pin.
        receipt = json.loads(raw)
        reason = _custody_reason(receipt)
        observation = None
        if reason is None:
            try:
                observation = core.observe_invocation(receipt)
            except ValueError:
                reason = "invalid_native_or_invocation_observation"
        native = receipt.get("native_rollout_usage")
        endpoint = native.get("task_complete_observed") if type(native) is dict else None
        index = None
        if observation is not None:
            index = len(observations)
            observations.append(observation)
        inventory.append(dict(invocation_id=identity, receipt_sha256=hashlib.sha256(raw).hexdigest(),
            observation_index=index, unsupported_reason=reason,
            endpoint_task_complete_observed=endpoint if type(endpoint) is bool else None,
            unqualified_native_counters=_counts(native) if reason is not None else None))
    return _make_summary(inventory, observations, marker)


def summarize_supervisor_usage_reports(reports):
    """Flatten supplied report populations before reconciling native sessions."""
    _need(type(reports) is list and len(reports) <= MAX_ROWS, "usage_report_population_bound")
    return summarize_supervisor_usage_summaries([summarize_supervisor_usage(report) for report in reports])


def summarize_supervisor_usage_summaries(summaries):
    """Reconcile safe summaries across trials without adding trial totals.

    Self pins detect mutation; they do not authenticate supplied observations.
    Rebuilding each summary validates its full closed shape and core seals.
    """
    _need(type(summaries) is list and len(summaries) <= MAX_ROWS, "usage_summary_population_bound")
    inventory, observations, markers = [], [], []
    total_bytes = 0
    for summary in summaries:
        _need(type(summary) is dict and summary.get("schema") == SCHEMA, "invalid_usage_summary")
        raw = _wire(summary)
        total_bytes += len(raw)
        _need(total_bytes <= MAX_BYTES, "usage_aggregate_byte_bound")
        summary = json.loads(raw)
        _need(summary.get("summary_sha256") == _digest({key: value for key, value in summary.items()
                                                       if key != "summary_sha256"}), "usage_summary_pin_mismatch")
        entries, native = summary.get("receipt_inventory"), summary.get("native_usage_accounting")
        _need(type(entries) is list and type(native) is dict, "invalid_usage_summary_inventory")
        rows = native.get("observations")
        _need(type(rows) is list and len(entries) <= MAX_ROWS, "invalid_usage_summary_observations")
        for entry in entries:
            _need(type(entry) is dict and set(entry) == {"invocation_id", "receipt_sha256", "observation_index",
                  "unsupported_reason", "endpoint_task_complete_observed", "unqualified_native_counters"},
                  "invalid_usage_summary_entry")
            _need(type(entry["invocation_id"]) is str and _IDENTITY.fullmatch(entry["invocation_id"]),
                  "invalid_provider_invocation_identity")
            _need(type(entry["receipt_sha256"]) is str and _SHA.fullmatch(entry["receipt_sha256"]),
                  "invalid_receipt_pin")
            endpoint = entry["endpoint_task_complete_observed"]
            _need(endpoint is None or type(endpoint) is bool, "invalid_endpoint_completion")
            index, reason = entry["observation_index"], entry["unsupported_reason"]
            if reason is None:
                _need(type(index) is int and 0 <= index < len(rows)
                      and rows[index].get("invocation_id") == entry["invocation_id"]
                      and entry["unqualified_native_counters"] is None, "invalid_usage_observation_join")
            else:
                _need(type(reason) is str and re.fullmatch(r"[a-z_]{1,80}", reason)
                      and index is None, "invalid_unsupported_observation")
                counts = entry["unqualified_native_counters"]
                _need(type(counts) is dict and set(counts) == set(COUNTERS) and all(
                    value is None or type(value) is int and 0 <= value <= core.MAX_COUNTER
                    for value in counts.values()), "invalid_unqualified_counters")
        indices = [entry["observation_index"] for entry in entries if entry["observation_index"] is not None]
        _need(indices == list(range(len(rows))), "usage_observation_population_mismatch")
        marker = summary.get("unreceipted_provider_attempt")
        _need(marker is None or type(marker) is bool, "invalid_unreceipted_provider_attempt_marker")
        rebuilt = _make_summary(entries, rows, marker)
        _need(_wire(rebuilt) == raw, "usage_summary_regeneration_mismatch")
        offset = len(observations)
        observations.extend(rows)
        inventory.extend({**entry, "observation_index": None if entry["observation_index"] is None
                          else offset + entry["observation_index"]} for entry in entries)
        # A report with no retained receipt cannot establish that its trial
        # made no provider attempt. Keep that gap when combining trials; a
        # later complete snapshot may close an identified session, but cannot
        # identify an empty report's unknown calls.
        markers.append(None if marker is False and not entries else marker)
        _need(len(inventory) <= MAX_ROWS, "usage_population_bound")
    return _make_summary(inventory, observations, _combined_marker(markers))
