"""Join native session counters without treating optional context as usage.

This adapter uses the existing reconciler. It neither reads native transcripts
nor infers missing session identities. Original per-invocation observations
remain in the comparison even when complete accounting is unavailable.
"""
from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core


def reconcile(report):
    result = {"status": "unavailable", "reason": "invalid_native_receipt_population",
              "summary": None, "context_fields_used_for_native_accounting": False}
    raw = report.get("provider_invocations", [])
    if type(raw) is not list or len(raw) > 128 or any(type(row) is not dict for row in raw):
        return result
    observations = []
    try:
        for row in raw:
            # Prompt/ledger validation is independent of the native accounting
            # contract. A malformed optional context ledger must not discard
            # otherwise valid native usage, or qualify an invalid native receipt.
            projected = {key: row[key] for key in core.OBSERVE_INPUT_FIELDS if key in row
                         and key not in {"model_prompt_sha256", "model_prompt_bytes", "native_context_delivery"}}
            native = projected.get("native_rollout_usage")
            if type(native) is dict:
                projected["native_rollout_usage"] = {key: value for key, value in native.items()
                    if key not in {"native_context_delivery", "interaction_telemetry"}}
            observations.append(core.observe_invocation(projected))
        marker = report.get("unreceipted_provider_attempt")
        marker = True if type(marker) is dict else marker if type(marker) is bool else None
        summary = core.summarize_invocations(observations, unreceipted_provider_attempt=marker)
    except (ValueError, TypeError, KeyError):
        # Do not export exception text or silently fall back to additive totals.
        return result
    return {**result, "status": "observed", "reason": "supplied_native_population_reconciled",
            "summary": summary}


def selected_usage(reconciliation, invocations, *, token_fields):
    """Select provider-native categories; mixed-provider input stays unnormalised.

    Costs lack a cumulative/per-invocation scope declaration. When different
    invocations share a native session, its reported dollar subtotal stays null.
    Repeated copies of the same invocation are harmless.
    """
    summary = reconciliation.get("summary")
    if summary is None:
        return [], None
    by_id = {row["invocation_id"]: row for row in invocations}
    selected = []
    for session in summary["sessions"]:
        index = session["selected_observation_index"]
        if index is None:
            selected.append({"counts": dict.fromkeys(token_fields), "cost": None})
            continue
        observation = summary["observations"][index]
        native = observation["native_usage"]
        counts = native["counts"]
        # Retain reported partial counters, but never treat conflicts as known.
        counts = dict.fromkeys(token_fields) if native["status"] == "invalid" else {
            key: counts.get(key) for key in token_fields}
        identities = {summary["observations"][i]["invocation_id"] for i in session["observation_indices"]}
        row = by_id.get(observation["invocation_id"], {})
        cost = row.get("known_reported_cost_subtotal_usd") if len(identities) == 1 else None
        selected.append({"counts": counts, "cost": cost})
    marker = summary["unreceipted_provider_attempt"]
    states = [session["complete"] for session in summary["sessions"]]
    complete = (None if marker is not False or not states else False if any(s is False for s in states)
                else True if all(s is True for s in states) else None)
    return selected, complete
