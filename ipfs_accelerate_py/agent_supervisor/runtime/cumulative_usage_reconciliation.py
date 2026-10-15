"""Reconcile retained Codex session counters without executing or opening stores.

The caller supplies receipts in observation order. Allocation invocation IDs
are not native session IDs: several invocations can retain the same cumulative
session, while a fresh retry or expansion session has its own charge. The
provider denominator is input plus output. Cached input and reasoning output
are subset diagnostics; tool text is already represented in provider input.

This utility measures only the supplied receipts. It cannot establish that all
campaign calls were retained, provider billing, task correctness, or authority.
Historical native @1 completion flags alone do not prove final usage complete.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

SCHEMA = "supervisor-cumulative-usage-reconciliation@1"
INVOCATION_SCHEMA = "router-implementation-invocation@1"
NATIVE_SCHEMA = "native-codex-cumulative-usage@1"
COMPLETION_BASIS = "native_lifecycle_after_latest_usage@1"
MAX_RECEIPTS = 10_000
MAX_COUNTER = (1 << 63) - 1
MAX_EVIDENCE_PER_SESSION = 16
COUNTERS = (
    "input_tokens", "output_tokens", "cached_input_tokens",
    "cache_write_input_tokens", "reasoning_output_tokens", "total_tokens",
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_THREAD = re.compile(r"[A-Za-z0-9-]{16,80}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _counter(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_COUNTER


def _sum_known(values) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _workspace_hash(value: Any) -> str | None:
    if (type(value) is not str or not 1 <= len(value) <= 4096 or "\x00" in value
            or len(value.encode()) > 4096):
        return None
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _parse(receipt: Any, ordinal: int) -> tuple[dict | None, dict | None]:
    """Project only safe accounting fields; arbitrary receipt data never leaves."""
    def issue(reason: str):
        return None, {"observation_ordinal": ordinal, "reason": reason}

    if type(receipt) is not dict or receipt.get("schema") != INVOCATION_SCHEMA:
        return issue("unsupported_invocation_schema")
    invocation = receipt.get("invocation_id")
    if not _matches(invocation, _ID):
        return issue("missing_or_invalid_invocation_identity")
    native = receipt.get("native_rollout_usage")
    if native is None:
        return issue("native_usage_unavailable")
    if type(native) is not dict or native.get("schema") != NATIVE_SCHEMA:
        return issue("unsupported_native_usage_schema")
    thread = native.get("thread_id")
    if not _matches(thread, _THREAD):
        return issue("missing_or_invalid_native_thread")

    errors: set[str] = set()
    model = receipt.get("model")
    model = model if _matches(model, _MODEL) else None
    workspace = _workspace_hash(receipt.get("workspace"))
    if receipt.get("provider") != "codex_cli":
        errors.add("provider_schema_mismatch")
    if model is None or workspace is None:
        errors.add("model_or_workspace_binding_unavailable")
    outer_usage = receipt.get("usage")
    outer_usage = outer_usage if type(outer_usage) is dict else {}
    outer_thread, outer_session = outer_usage.get("thread_id"), outer_usage.get("session_id")
    if outer_thread is None and outer_session is None:
        errors.add("outer_session_binding_unavailable")
    elif ((outer_thread is not None and outer_thread != thread)
          or (outer_session is not None and outer_session != thread)):
        errors.add("outer_native_session_binding_conflict")
    if outer_usage.get("model_id") is not None and outer_usage["model_id"] != model:
        errors.add("requested_observed_model_conflict")

    raw_usage = native.get("usage")
    if type(raw_usage) is not dict:
        errors.add("invalid_native_counter_mapping")
        raw_usage = {}
    usage = {}
    for name in COUNTERS:
        value = raw_usage.get(name)
        usage[name] = value if _counter(value) else None
        if name in raw_usage and not _counter(value):
            errors.add("invalid_native_counter")
    available = native.get("usage_available")
    if type(available) is not bool:
        errors.add("invalid_usage_availability")
    elif available != any(value is not None for value in usage.values()):
        errors.add("usage_availability_counter_conflict")
    count = native.get("observed_token_count_records")
    if not _counter(count) or (available is True and count == 0):
        errors.add("invalid_cumulative_record_count")
        count = None
    digest = native.get("rollout_sha256")
    if not _matches(digest, _SHA):
        errors.add("invalid_rollout_digest")
        digest = None
    if native.get("cache_included_in_input") is not True:
        errors.add("unsupported_cache_accounting_convention")
    prompt_digests = {}
    for name in ("native_prompt_sha256", "prompt_sha256", "router_prompt_sha256",
                 "model_prompt_sha256", "workspace_advisory_sha256"):
        value = receipt.get(name)
        if value is not None:
            if _matches(value, _SHA):
                prompt_digests[name] = value
            else:
                errors.add("invalid_prompt_binding_digest")
    for subset, total in (("cached_input_tokens", "input_tokens"),
                          ("cache_write_input_tokens", "input_tokens"),
                          ("reasoning_output_tokens", "output_tokens")):
        if usage[subset] is not None and usage[total] is not None and usage[subset] > usage[total]:
            errors.add("subset_exceeds_provider_counter")
    if all(usage[name] is not None for name in ("input_tokens", "output_tokens", "total_tokens")):
        if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
            errors.add("reported_provider_total_conflict")

    declared = native.get("task_complete_observed")
    if declared is not None and type(declared) is not bool:
        errors.add("invalid_declared_completion")
        declared = None
    basis = native.get("completion_basis")
    malformed = native.get("lifecycle_malformed_records")
    lifecycle = native.get("usage_lifecycle_complete")
    strict = None
    reasons = set()
    if basis is None:
        reasons.add("legacy_final_usage_completeness_unknown")
    elif basis != COMPLETION_BASIS:
        reasons.add("unsupported_completion_basis")
    elif not _counter(malformed) or type(lifecycle) is not bool:
        errors.add("invalid_completion_metadata")
    elif lifecycle and (malformed != 0 or declared is not True or available is not True):
        errors.add("completion_metadata_conflict")
    else:
        strict = lifecycle
        if not strict:
            reasons.add("native_usage_lifecycle_incomplete")
    if usage["input_tokens"] is None or usage["output_tokens"] is None:
        reasons.add("provider_counter_categories_incomplete")
        if strict is True:
            strict = None
    # Local prompt counts, outer CLI usage and arbitrary tool-token fields are
    # not an additional native session charge and are intentionally omitted.
    row = {"observation_ordinal": ordinal, "invocation_id": invocation,
           "thread_id": thread, "model": model, "workspace_binding_sha256": workspace,
           "prompt_binding_sha256": prompt_digests,
           "invocation_status": receipt.get("status") if _matches(receipt.get("status"), _ID) else None,
           "invocation_phase": receipt.get("phase") if _matches(receipt.get("phase"), _ID) else None,
           "rollout_sha256": digest, "observed_token_count_records": count,
           "usage": usage, "producer_declared_complete": declared,
           "completion_basis": basis if _matches(basis, _MODEL) else None,
           "lifecycle_malformed_records": malformed if _counter(malformed) else None,
           "usage_lifecycle_complete": lifecycle if type(lifecycle) is bool else None,
           "strict_usage_complete": strict, "reasons": reasons, "errors": errors}
    return row, None


def reconcile_router_cumulative_usage(receipts: Sequence[dict]) -> dict:
    """Count each native session once using explicit caller observation order.

    Only the two declared Codex receipt schemas are supported. A count or
    counter regression, alias conflict, or changed model/workspace quarantines
    the entire session; no largest snapshot or partial-snapshot union is used.
    Complete totals describe supplied receipts only. Population completeness,
    billing and all forms of success/admission remain unestablished.
    """
    if not isinstance(receipts, (list, tuple)) or len(receipts) > MAX_RECEIPTS:
        raise ValueError("a bounded ordered list or tuple of retained receipts is required")
    sessions: dict[str, list[dict]] = {}
    invocations: dict[str, dict] = {}
    digests: dict[str, dict] = {}
    last_known: dict[str, dict[str, tuple[int, int]]] = {}
    issues, repeated = [], 0
    for ordinal, receipt in enumerate(receipts):
        row, issue = _parse(receipt, ordinal)
        if issue is not None:
            issues.append(issue)
            continue
        assert row is not None
        rows = sessions.setdefault(row["thread_id"], [])
        prior = invocations.get(row["invocation_id"])
        signature = {key: value for key, value in row.items()
                     if key not in {"observation_ordinal", "errors", "reasons"}}
        if prior is not None:
            prior_signature = {key: value for key, value in prior.items()
                               if key not in {"observation_ordinal", "errors", "reasons"}}
            if signature == prior_signature and row["errors"] == prior["errors"]:
                repeated += 1
                continue
            prior["errors"].add("invocation_identity_alias_conflict")
            row["errors"].add("invocation_identity_alias_conflict")
        else:
            invocations[row["invocation_id"]] = row
        if row["rollout_sha256"] is not None:
            previous_digest = digests.get(row["rollout_sha256"])
            if previous_digest is not None and any(
                row[key] != previous_digest[key]
                for key in ("thread_id", "model", "workspace_binding_sha256", "usage",
                            "observed_token_count_records", "producer_declared_complete",
                            "completion_basis", "lifecycle_malformed_records", "usage_lifecycle_complete")
            ):
                previous_digest["errors"].add("rollout_digest_alias_conflict")
                row["errors"].add("rollout_digest_alias_conflict")
            else:
                digests[row["rollout_sha256"]] = row
        if rows:
            previous = rows[-1]
            if any(row[key] != previous[key] for key in ("model", "workspace_binding_sha256")):
                row["errors"].add("native_session_binding_changed")
            old_count, new_count = previous["observed_token_count_records"], row["observed_token_count_records"]
            if old_count is not None and new_count is not None:
                if new_count < old_count:
                    row["errors"].add("cumulative_record_count_regression")
        remembered = last_known.setdefault(row["thread_id"], {})
        new_count = row["observed_token_count_records"]
        if new_count is not None:
            for key in COUNTERS:
                new = row["usage"][key]
                if new is None:
                    continue
                if key in remembered:
                    old_count, old = remembered[key]
                    if new_count == old_count and new != old:
                        row["errors"].add("same_record_count_counter_conflict")
                    elif new_count > old_count and new < old:
                        row["errors"].add("cumulative_counter_regression")
                remembered[key] = (new_count, new)
        rows.append(row)

    projected = []
    for thread, rows in sessions.items():
        if not rows:  # A repeated receipt cannot create a new native session.
            continue
        latest = rows[-1]
        errors = set().union(*(row["errors"] for row in rows))
        status = ("conflict" if errors else "complete" if latest["strict_usage_complete"] is True
                  else "unavailable" if all(value is None for value in latest["usage"].values())
                  else "partial")
        usage = {key: None if errors else value for key, value in latest["usage"].items()}
        subtotal = _sum_known((usage["input_tokens"], usage["output_tokens"]))
        projected.append({
            "provider": "codex_cli", "thread_id": thread, "status": status,
            "model": latest["model"], "workspace_binding_sha256": latest["workspace_binding_sha256"],
            "invocation_ids": sorted({row["invocation_id"] for row in rows}),
            "selected_observation_ordinal": latest["observation_ordinal"] if not errors else None,
            "selected_rollout_sha256": latest["rollout_sha256"] if not errors else None,
            "observed_token_count_records": latest["observed_token_count_records"] if not errors else None,
            "producer_declared_complete": latest["producer_declared_complete"],
            "strict_usage_complete": latest["strict_usage_complete"] if not errors else None,
            "reasons": sorted(errors | latest["reasons"]),
            "known_counter_subtotals": usage, "known_provider_token_subtotal": subtotal,
            "provider_total_tokens": subtotal if status == "complete" else None,
            "snapshot_evidence": [{
                key: row[key] for key in ("observation_ordinal", "invocation_id", "rollout_sha256",
                                          "observed_token_count_records", "prompt_binding_sha256", "usage")
            } for row in rows[-MAX_EVIDENCE_PER_SESSION:]],
            "snapshot_evidence_omitted_count": max(0, len(rows) - MAX_EVIDENCE_PER_SESSION),
        })
    known = {key: _sum_known(row["known_counter_subtotals"][key] for row in projected)
             for key in COUNTERS}
    subtotal = _sum_known(row["known_provider_token_subtotal"] for row in projected)
    complete = bool(projected) and not issues and all(row["status"] == "complete" for row in projected)
    return {"schema": SCHEMA, "observations_received": len(receipts),
            "unique_invocation_ids": len(invocations), "repeated_receipt_observations": repeated,
            "native_sessions_observed": len(projected), "sessions": projected, "issues": issues,
            "known_counter_subtotals": known, "known_provider_token_subtotal": subtotal,
            "provider_total_tokens": subtotal if complete else None,
            "supplied_receipt_usage_complete": complete, "campaign_usage_complete": None,
            "total_scope": "supplied retained receipts only",
            "observation_order_basis": "caller sequence and native cumulative record cardinality",
            "provider_token_definition": "input_tokens + output_tokens",
            "cached_and_reasoning_tokens_are_subsets": True, "tool_tokens_added": False,
            "billing_total_verified": False, "execution_authority": False,
            "completion_authority": False, "correctness_established": False}
