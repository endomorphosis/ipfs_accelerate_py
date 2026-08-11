"""Fail-closed provider failure classification for supervisor routing.

Provider-failure findings come only from a fixed, isolated, no-tools probe
that runs before the task model is dispatched. Hard-quota findings still need
independent confirmation. Task stdout/stderr is never fallback authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from ...llm_router import (
    _AGENT_IMPLEMENTATION_FAILURE_RECEIPT_SCHEMA,
    _AGENT_IMPLEMENTATION_FAILURE_SOURCE,
    _AGENT_IMPLEMENTATION_PROBE_CONTRACT,
    _AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID,
    _AGENT_IMPLEMENTATION_PROBE_PROMPT,
    AGENT_IMPLEMENTATION_GROK_NOT_SIGNED_IN_GUIDANCE,
    build_agent_implementation_failure_receipt,
    classify_agent_implementation_failure,
    valid_agent_implementation_failure_receipt,
)

GROK_FAILURE_RECEIPT_SCHEMA = _AGENT_IMPLEMENTATION_FAILURE_RECEIPT_SCHEMA
GROK_FAILURE_RECEIPT_PREFIX = "IPFS_ACCELERATE_GROK_FAILURE_RECEIPT="
GROK_ROUTE_OUTCOME_SCHEMA = "ipfs_accelerate_py.agent_supervisor.grok-route-outcome@1"
GROK_ROUTE_OUTCOME_PREFIX = "IPFS_ACCELERATE_GROK_ROUTE_OUTCOME="
MAX_GROK_FAILURE_EVIDENCE_BYTES = 128 * 1024
GROK_QUOTA_PROBE_SOURCE = _AGENT_IMPLEMENTATION_FAILURE_SOURCE
GROK_QUOTA_PROBE_PROMPT = _AGENT_IMPLEMENTATION_PROBE_PROMPT
GROK_QUOTA_PROBE_TIMEOUT_SECONDS = 60
GROK_NOT_SIGNED_IN_GUIDANCE = (
    AGENT_IMPLEMENTATION_GROK_NOT_SIGNED_IN_GUIDANCE
)
GROK_QUOTA_PROBE_CONTRACT = _AGENT_IMPLEMENTATION_PROBE_CONTRACT
GROK_QUOTA_PROBE_CONTRACT_ID = _AGENT_IMPLEMENTATION_PROBE_CONTRACT_ID


def _reject_duplicate_record_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate runner record key")
        value[key] = item
    return value


def _bounded_record_text(value: str) -> str:
    """Bound record input without manufacturing a logical line start.

    A raw tail slice can begin in the middle of model-controlled output.  If
    that slice happens to start with the reserved receipt prefix, treating the
    slice boundary as a line boundary would grant control-plane meaning to
    ordinary child output.  Retain the first candidate line only when the
    byte immediately before the bounded window proves it was LF-delimited;
    otherwise discard the partial line.
    """

    encoded = str(value or "").encode("utf-8", errors="replace")
    start = max(0, len(encoded) - MAX_GROK_FAILURE_EVIDENCE_BYTES)
    if start and encoded[start - 1 : start] != b"\n":
        next_lf = encoded.find(b"\n", start)
        if next_lf < 0:
            return ""
        start = next_lf + 1
    return encoded[start:].decode("utf-8", errors="replace")


def classify_grok_stderr(stderr_text: str) -> dict[str, str]:
    """Compatibility alias for the canonical router-owned classifier."""

    return classify_agent_implementation_failure(
        stderr_text,
        max_evidence_bytes=MAX_GROK_FAILURE_EVIDENCE_BYTES,
    )


def build_grok_failure_receipt(
    *,
    probe_stderr_text: str,
    nonce: str,
    model: str,
    probe_returncode: int,
    primary_dispatched: bool = False,
    evidence_size: int | None = None,
    evidence_overflow: bool | None = None,
) -> dict[str, Any]:
    """Build a content-addressed receipt for the isolated quota preflight."""

    return build_agent_implementation_failure_receipt(
        probe_stderr_text=probe_stderr_text,
        nonce=nonce,
        model=model,
        probe_returncode=probe_returncode,
        primary_dispatched=primary_dispatched,
        evidence_size=evidence_size,
        evidence_overflow=evidence_overflow,
    )


def render_grok_failure_receipt(receipt: Mapping[str, Any]) -> str:
    return GROK_FAILURE_RECEIPT_PREFIX + json.dumps(
        dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
    )


def build_grok_route_outcome(
    *,
    receipt: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    decision: str,
    verifier_status: str,
    fallback_dispatched: bool,
    fallback_returncode: int | None,
    quota_evidence_id: str = "",
) -> dict[str, Any]:
    """Bind the runner's terminal route decision to one preflight receipt."""

    outcome = {
        "schema": GROK_ROUTE_OUTCOME_SCHEMA,
        "source": "grok_cli_runner",
        "nonce": str(receipt.get("nonce") or ""),
        "primary_model": str(receipt.get("primary_model") or ""),
        "probe_returncode": receipt.get("probe_returncode"),
        "preflight_receipt_id": str(receipt.get("receipt_id") or ""),
        "failure_class": str(receipt.get("failure_class") or ""),
        "route_plan": dict(route_plan),
        "quota_evidence_id": str(quota_evidence_id or ""),
        "decision": str(decision),
        "verifier_status": str(verifier_status),
        "fallback_dispatched": bool(fallback_dispatched),
        "fallback_returncode": fallback_returncode,
    }
    outcome["outcome_id"] = _grok_route_outcome_identity(outcome)
    return outcome


def _grok_route_outcome_identity(outcome: Mapping[str, Any]) -> str:
    body = dict(outcome)
    body.pop("outcome_id", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def render_grok_route_outcome(outcome: Mapping[str, Any]) -> str:
    return GROK_ROUTE_OUTCOME_PREFIX + json.dumps(
        dict(outcome),
        sort_keys=True,
        separators=(",", ":"),
    )


def extract_grok_route_outcomes(text: str) -> tuple[dict[str, Any], ...]:
    """Extract bounded terminal route records from a combined daemon log."""

    outcomes: list[dict[str, Any]] = []
    for line in _bounded_record_text(text).split("\n"):
        if not line.startswith(GROK_ROUTE_OUTCOME_PREFIX):
            continue
        raw = line[len(GROK_ROUTE_OUTCOME_PREFIX) :]
        if "\r" in raw or len(raw.encode("utf-8")) > 4096:
            continue
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_record_keys,
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            outcomes.append(value)
    return tuple(outcomes[-4:])


def valid_grok_route_outcome(
    outcome: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    runner_returncode: int,
) -> bool:
    """Validate a terminal decision and its exact preflight/exit binding."""

    expected_fields = {
        "schema",
        "source",
        "nonce",
        "primary_model",
        "probe_returncode",
        "preflight_receipt_id",
        "failure_class",
        "route_plan",
        "quota_evidence_id",
        "decision",
        "verifier_status",
        "fallback_dispatched",
        "fallback_returncode",
        "outcome_id",
    }
    decision = outcome.get("decision")
    verifier_status = outcome.get("verifier_status")
    fallback_returncode = outcome.get("fallback_returncode")
    common_valid = bool(
        set(outcome) == expected_fields
        and outcome.get("schema") == GROK_ROUTE_OUTCOME_SCHEMA
        and outcome.get("source") == "grok_cli_runner"
        and outcome.get("nonce") == receipt.get("nonce")
        and outcome.get("primary_model") == receipt.get("primary_model")
        and outcome.get("probe_returncode") == receipt.get("probe_returncode")
        and outcome.get("preflight_receipt_id") == receipt.get("receipt_id")
        and outcome.get("failure_class") == receipt.get("failure_class")
        and outcome.get("route_plan") == dict(route_plan)
        and (
            outcome.get("quota_evidence_id") == ""
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(outcome.get("quota_evidence_id") or ""),
            )
            is not None
        )
        and verifier_status
        in {
            "not_run",
            "not_required_exact_auth",
            "confirmed_quota",
            "not_confirmed",
        }
        and isinstance(outcome.get("fallback_dispatched"), bool)
        and outcome.get("outcome_id") == _grok_route_outcome_identity(outcome)
    )
    if not common_valid:
        return False
    if (
        verifier_status == "confirmed_quota"
        and not outcome.get("quota_evidence_id")
    ) or (
        verifier_status != "confirmed_quota"
        and outcome.get("quota_evidence_id")
    ):
        return False
    if decision == "denied":
        return bool(
            outcome.get("fallback_dispatched") is False
            and fallback_returncode is None
            and runner_returncode == receipt.get("probe_returncode")
        )
    if decision == "fallback_succeeded":
        return bool(
            outcome.get("fallback_dispatched") is True
            and fallback_returncode == runner_returncode == 0
        )
    if decision == "fallback_failed":
        return bool(
            isinstance(fallback_returncode, int)
            and not isinstance(fallback_returncode, bool)
            and fallback_returncode == runner_returncode != 0
        )
    return False


def extract_grok_failure_receipts(text: str) -> tuple[dict[str, Any], ...]:
    """Extract bounded runner receipt lines from a combined daemon log."""

    receipts: list[dict[str, Any]] = []
    # Runner records are canonical LF-delimited lines. Deliberately do not use
    # str.splitlines(): accepting bare CR or Unicode separators would disagree
    # with the child-output filter's framing and create a prefix-smuggling gap.
    for line in _bounded_record_text(text).split("\n"):
        if not line.startswith(GROK_FAILURE_RECEIPT_PREFIX):
            continue
        raw = line[len(GROK_FAILURE_RECEIPT_PREFIX) :]
        if "\r" in raw:
            continue
        if len(raw.encode("utf-8")) > 4096:
            continue
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_record_keys,
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            receipts.append(value)
    return tuple(receipts[-4:])


def valid_grok_hard_quota_receipt(
    receipt: Mapping[str, Any],
    *,
    nonce: str,
    model: str,
    returncode: int,
) -> bool:
    """Return whether a pre-dispatch probe grants hard-quota authority."""

    return bool(
        valid_grok_failure_receipt(
            receipt,
            nonce=nonce,
            model=model,
            returncode=returncode,
        )
        and receipt.get("failure_class") == "hard_quota_exhausted"
    )


def valid_grok_failure_receipt(
    receipt: Mapping[str, Any],
    *,
    nonce: str,
    model: str,
    returncode: int,
) -> bool:
    """Validate any isolated pre-dispatch probe outcome receipt."""

    return valid_agent_implementation_failure_receipt(
        receipt,
        nonce=nonce,
        model=model,
        probe_returncode=returncode,
    )


__all__ = [
    "GROK_FAILURE_RECEIPT_PREFIX",
    "GROK_FAILURE_RECEIPT_SCHEMA",
    "GROK_ROUTE_OUTCOME_PREFIX",
    "GROK_ROUTE_OUTCOME_SCHEMA",
    "GROK_QUOTA_PROBE_CONTRACT_ID",
    "GROK_QUOTA_PROBE_CONTRACT",
    "GROK_QUOTA_PROBE_PROMPT",
    "GROK_QUOTA_PROBE_SOURCE",
    "GROK_QUOTA_PROBE_TIMEOUT_SECONDS",
    "MAX_GROK_FAILURE_EVIDENCE_BYTES",
    "build_grok_failure_receipt",
    "build_grok_route_outcome",
    "classify_grok_stderr",
    "extract_grok_failure_receipts",
    "extract_grok_route_outcomes",
    "render_grok_failure_receipt",
    "render_grok_route_outcome",
    "valid_grok_failure_receipt",
    "valid_grok_hard_quota_receipt",
    "valid_grok_route_outcome",
]
