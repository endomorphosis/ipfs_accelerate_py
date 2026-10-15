"""Shared retained-decision accounting; local prompt estimates are never charges.

The existing ordinary provider projection remains authoritative for its own
native scope. Decision coverage is currently qualified for Codex sessions only;
other provider observations stay visible without acquiring complete totals.
"""
from __future__ import annotations

from .native_usage_accounting import native_usage_contract

DECISION_REPORT_FIELDS = (
    "semantic_decision_dialogues", "semantic_decision_dialogue_expected",
    "semantic_decision_owner_bindings", "semantic_decision_owner_binding_expected",
    "semantic_decision_expected_task_id", "semantic_decision_nomination_sha256",
    "semantic_decision_typed_nomination_expected",
)
COUNTERS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")


def decision_usage_selected(report: dict) -> bool:
    return any(key in report for key in DECISION_REPORT_FIELDS)


def _accounting_inputs(report):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import (
        semantic_decision_accounting_inputs,
    )
    rows, coverage = semantic_decision_accounting_inputs(report)
    for key in ("semantic_decision_dialogue_expected", "semantic_decision_owner_binding_expected",
                "semantic_decision_typed_nomination_expected"):
        if key in report and type(report[key]) is not bool:
            coverage["authoritative_usage_coverage_complete"] = False
            coverage["all_attempts_receipted"] = False
            coverage["provider_call_count_complete"] = False
            coverage["issues"].append({"reason": "invalid_expected_decision_mode"})
    return rows, coverage


def measured_decision_usage(report: dict) -> dict:
    """Reconstruct coverage and cumulative sessions from original retained rows."""
    from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
        reconcile_router_cumulative_usage,
    )

    rows, coverage = _accounting_inputs(report)
    unique = {}
    for row in rows:
        identity = row.get("invocation_id") if type(row) is dict else None
        if type(identity) is not str or not identity:
            raise ValueError("provider invocation identity is unavailable")
        if identity in unique and unique[identity] != row:
            raise ValueError("conflicting provider receipts")
        unique[identity] = row
    reconciliation = reconcile_router_cumulative_usage(rows)
    unreceipted = "unreceipted_provider_attempt" in report
    contracts = [native_usage_contract(row, row.get("native_rollout_usage") or {})
                 for row in unique.values()]
    codex_only = bool(unique) and all(row.get("provider") == "codex_cli" for row in unique.values())
    native_scope_qualified = bool(contracts) and all(
        scope == "native_session_cumulative" and cache is True
        for _complete, scope, cache in contracts)
    complete = (reconciliation["supplied_receipt_usage_complete"] is True
        and not unreceipted and coverage["authoritative_usage_coverage_complete"] is True
        and codex_only and native_scope_qualified)
    result = {
        "provider_calls": (len(unique) + coverage["known_additional_provider_calls"]
                           if coverage["provider_call_count_complete"] else None),
        "observed_complete_sessions": complete,
        "cache_included_in_input": True if codex_only else None,
        "source": "observed_native_cumulative_totals" if codex_only else "observed_native_mixed_or_unknown_totals",
        "all_invocations_receipted": ((bool(rows) or coverage["reported_provider_dispatches"] > 0)
            and not unreceipted and coverage["all_attempts_receipted"]),
        "billing_total_verified": False, "dollar_cost": None,
        "accounting_scope": "supplied retained receipts only", "campaign_usage_complete": None,
        "known_counter_subtotals": reconciliation["known_counter_subtotals"],
        "known_provider_token_subtotal": reconciliation["known_provider_token_subtotal"],
        "reconciliation": reconciliation,
        "native_scoped_usage_complete_observed": complete,
        "token_count_scope": "native_session_cumulative" if codex_only else None,
        "harbor_usage_exportable": complete, "usage_accounting_complete": complete,
    }
    if decision_usage_selected(report):
        result["semantic_decision_coverage"] = coverage
    for key in COUNTERS:
        values = [session["known_counter_subtotals"][key] for session in reconciliation["sessions"]]
        result[key] = (reconciliation["provider_total_tokens"] if key == "total_tokens"
            else sum(values) if all(type(value) is int and value >= 0 for value in values) else None) if complete else None
    # These are separate observed native categories, never added to the Codex
    # subtotal or to wrapper/local prompt estimates. Sparse/unsupported provider
    # contracts cannot acquire authoritative decision coverage by being present.
    other = _other_native_observations(unique.values())
    if other:
        result["other_provider_native_observations"] = other
    return result


def _other_native_observations(rows):
    other = []
    for row in rows:
        if type(row) is not dict or row.get("provider") == "codex_cli":
            continue
        native = row.get("native_rollout_usage")
        native = native if type(native) is dict else {}
        usage = native.get("usage")
        usage = usage if type(usage) is dict else {}
        observed_complete, scope, cache = native_usage_contract(row, native)
        observed = {"provider": "grok_cli" if row.get("provider") == "grok_cli" else "other",
            "native_count_scope": scope, "cache_included_in_input": cache,
            "native_scope_complete_observed": observed_complete,
            "decision_usage_coverage_qualified": False}
        for key in (*COUNTERS, "cache_write_input_tokens"):
            value = usage.get(key)
            observed[key] = value if type(value) is int and value >= 0 else None
        other.append(observed)
    return other


def finalize_usage_accounting(report: dict) -> None:
    """Add a bounded accounting observation without changing original evidence."""
    value = {"schema": "terminal-provider-usage-accounting@1",
        "known_unreceipted_provider_attempt": "unreceipted_provider_attempt" in report,
        "report_provider_total_tokens": None, "campaign_usage_complete": None,
        "billing_total_verified": False, "production_token_savings_established": False,
        "execution_authority": False, "completion_authority": False}
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
            reconcile_router_cumulative_usage,
        )
        rows, coverage = _accounting_inputs(report)
        reconciliation = reconcile_router_cumulative_usage(rows)
        native_scope_qualified = bool(rows) and all(type(row) is dict
            and row.get("provider") == "codex_cli"
            and native_usage_contract(row, row.get("native_rollout_usage") or {})[1:]
                == ("native_session_cumulative", True) for row in rows)
    except Exception as error:
        value.update(status="unknown", error_type=type(error).__name__[:128])
    else:
        # Conflicting retained native rows must still produce reconciliation
        # evidence. The stricter Harbor projection refuses such aliases, while
        # this final record preserves their known charge and conflict reasons.
        value.update(status="observed", reconciliation=reconciliation,
            report_provider_total_tokens=(reconciliation["provider_total_tokens"]
                if native_scope_qualified and "unreceipted_provider_attempt" not in report
                and coverage["authoritative_usage_coverage_complete"] is True else None))
        if decision_usage_selected(report):
            value["semantic_decision_coverage"] = coverage
        other = _other_native_observations(rows)
        if other:
            value["other_provider_native_observations"] = other
    report["provider_usage_accounting"] = value
