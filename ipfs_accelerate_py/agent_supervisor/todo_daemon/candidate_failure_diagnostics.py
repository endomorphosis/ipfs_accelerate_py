"""Closed diagnostic codes retained across candidate retries, without authority."""

from __future__ import annotations

import json
from typing import Any

from ..validation.implementation_failure_review import FailureReviewReason
from ..validation.proposal_validation import ProposalFindingCode

MAX_CODES = 16
MAX_SUMMARY_BYTES = 2048
KNOWN_REVIEW_CODES = frozenset(code.value for code in FailureReviewReason)
KNOWN_FINDING_CODES = frozenset(code.value for code in ProposalFindingCode) | {
    # Emitted by the native protected validation-channel proposal guard.
    "validation_channel_tampering_forbidden",
}


def _codes(value: Any, allowed: frozenset[str]) -> list[str]:
    if type(value) not in (list, tuple):
        return []
    return sorted({
        item for item in value[:MAX_CODES]
        if type(item) is str and len(item) <= 128 and item in allowed
    })


def normalize_candidate_failure_diagnostics(value: Any) -> dict[str, list[str]]:
    """Discard prose, paths, commands, unknown codes and candidate object hooks."""
    if type(value) is not dict:
        return {}
    result = {}
    for field, allowed in (
        ("reason_codes", KNOWN_REVIEW_CODES),
        ("finding_codes", KNOWN_FINDING_CODES),
    ):
        codes = _codes(value.get(field), allowed)
        if codes:
            result[field] = codes
    # At most 32 already bounded strings are considered; never persist a
    # larger optional phase body if the closed vocabulary grows in future.
    while len(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()) > MAX_SUMMARY_BYTES:
        field = max(result, key=lambda key: len(result[key]))
        result[field].pop()
        if not result[field]:
            del result[field]
    return result


def summarize_candidate_failure(implementation: Any) -> dict[str, list[str]]:
    """Extract only closed codes from the current Portal validation result."""
    if type(implementation) is not dict:
        return {}
    validation = implementation.get("validation_result")
    if type(validation) is not dict:
        return {}
    containers = [validation]
    for field in ("proposal_gate", "proposal_validation", "failure_review"):
        value = validation.get(field)
        if type(value) is dict:
            containers.append(value)
    reasons: set[str] = set()
    findings: set[str] = set()
    for value in containers:
        reason = value.get("reason")
        if type(reason) is str and len(reason) <= 128 and reason in KNOWN_REVIEW_CODES:
            reasons.add(reason)
        for field in ("reason_codes", "finding_codes"):
            codes = value.get(field)
            reasons.update(_codes(codes, KNOWN_REVIEW_CODES))
            findings.update(_codes(codes, KNOWN_FINDING_CODES))
    return normalize_candidate_failure_diagnostics({
        "reason_codes": sorted(reasons), "finding_codes": sorted(findings),
    })


def candidate_failure_codes_from_history(
    attempt: Any, history: Any, *, reason: str,
) -> dict[str, list[str]]:
    """Recover optional codes from this exact immutable failed phase only."""
    if (
        getattr(attempt, "status", None) != "failed"
        or getattr(attempt, "committed_phase", None) != "failed"
        or type(history) is not list or len(history) > 32
    ):
        return {}
    phases = [row for row in history if type(row) is dict and row.get("phase") == "failed"]
    if len(phases) != 1:
        return {}
    row = phases[0]
    for field, attribute in (
        ("revision", "revision"), ("fencing_token", "fencing_token"),
        ("fence_epoch", "fence_epoch"), ("committed_at_ms", "finished_at_ms"),
    ):
        observed, expected = row.get(field), getattr(attempt, attribute, None)
        if type(observed) is not int or type(expected) is not int or observed != expected:
            return {}
    body = row.get("body")
    if (
        type(body) is not dict or body.get("reason") != reason
        or body.get("attempt_consumed") is not True
        or body.get("provider_dispatched") is not True
    ):
        return {}
    return normalize_candidate_failure_diagnostics(body.get("candidate_failure_diagnostics"))
