"""Fail-closed provider failure classification for supervisor routing.

Hard-quota authority comes only from a fixed, isolated, no-tools probe that
runs before the task model is dispatched. Task stdout/stderr is never fallback
authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

GROK_FAILURE_RECEIPT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.grok-failure-receipt@2"
GROK_FAILURE_RECEIPT_PREFIX = "IPFS_ACCELERATE_GROK_FAILURE_RECEIPT="
MAX_GROK_FAILURE_EVIDENCE_BYTES = 128 * 1024
GROK_QUOTA_PROBE_SOURCE = "isolated_no_tools_pre_dispatch_probe"
GROK_QUOTA_PROBE_PROMPT = "This is a provider-capacity preflight. Reply with exactly OK."
GROK_QUOTA_PROBE_TIMEOUT_SECONDS = 60
GROK_QUOTA_PROBE_CONTRACT = {
    "schema": "ipfs_accelerate_py.agent_supervisor.grok-quota-probe@1",
    "model": "grok-4.5",
    "mode": "chat",
    "max_turns": 1,
    "permission_mode": "dontAsk",
    "tools": "",
    "no_plan": True,
    "no_subagents": True,
    "disable_web_search": True,
    "no_memory": True,
    "isolated_workspace": True,
    "task_context": False,
    "prompt": GROK_QUOTA_PROBE_PROMPT,
    "timeout_seconds": GROK_QUOTA_PROBE_TIMEOUT_SECONDS,
}
GROK_QUOTA_PROBE_CONTRACT_ID = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            GROK_QUOTA_PROBE_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
)

_HARD_QUOTA_PATTERN = re.compile(
    r"(?:"
    r"(?:grok(?:\s+build)?|xai)[^\r\n]{0,200}(?:"
    r"\b402\b|insufficient[_ ]quota|quota[_ ]exceeded|quota exhausted|"
    r"balance exhausted|usage balance exhausted)|"
    r"(?:\b402\b|insufficient[_ ]quota|quota[_ ]exceeded|quota exhausted|"
    r"balance exhausted|usage balance exhausted)[^\r\n]{0,200}"
    r"(?:grok(?:\s+build)?|xai)|"
    r"status\s+402|out of credits|usage balance exhausted|"
    r"over (?:your )?spending limit"
    r")",
    re.IGNORECASE,
)
_RATE_LIMIT_PATTERN = re.compile(
    r"(?:\b429\b|rate[_ -]?limit(?:ed|s|_exceeded)?|too many requests|"
    r"resource[_ -]?exhausted|overloaded)",
    re.IGNORECASE,
)
_AUTH_PATTERN = re.compile(
    r"(?:\b401\b|\b403\b|not signed in|not authenticated|authentication "
    r"failed|invalid api key|unauthorized|forbidden)",
    re.IGNORECASE,
)
_INVALID_REQUEST_PATTERN = re.compile(
    r"(?:\b400\b|invalid model|model not found|bad request|invalid argument)",
    re.IGNORECASE,
)
_TRANSPORT_PATTERN = re.compile(
    r"(?:tls|certificate|connection (?:refused|reset)|dns|name resolution|"
    r"network unreachable|timed? out|timeout)",
    re.IGNORECASE,
)


def _bounded_text(value: str) -> str:
    encoded = str(value or "").encode("utf-8", errors="replace")
    return encoded[-MAX_GROK_FAILURE_EVIDENCE_BYTES:].decode(
        "utf-8",
        errors="replace",
    )


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
    """Classify bounded probe stderr without retaining raw provider output.

    Any authentication, request, rate-limit, or transport signal dominates a
    quota-looking fragment. Mixed or ambiguous failures therefore cannot grant
    cross-provider authority.
    """

    text = _bounded_text(stderr_text)
    classes = (
        ("authentication", _AUTH_PATTERN),
        ("invalid_request", _INVALID_REQUEST_PATTERN),
        ("rate_limited", _RATE_LIMIT_PATTERN),
        ("transport", _TRANSPORT_PATTERN),
        ("hard_quota_exhausted", _HARD_QUOTA_PATTERN),
    )
    for failure_class, pattern in classes:
        match = pattern.search(text)
        if match is None:
            continue
        normalized = " ".join(match.group(0).lower().split())
        return {
            "failure_class": failure_class,
            "evidence_sha256": ("sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()),
        }
    return {
        "failure_class": "unknown",
        "evidence_sha256": ("sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()),
    }


def build_grok_failure_receipt(
    *,
    probe_stderr_text: str,
    nonce: str,
    model: str,
    probe_returncode: int,
    primary_dispatched: bool = False,
) -> dict[str, Any]:
    """Build a content-addressed receipt for the isolated quota preflight."""

    classified = classify_grok_stderr(probe_stderr_text)
    receipt = {
        "schema": GROK_FAILURE_RECEIPT_SCHEMA,
        "source": GROK_QUOTA_PROBE_SOURCE,
        "probe_contract_id": GROK_QUOTA_PROBE_CONTRACT_ID,
        "nonce": str(nonce),
        "primary_provider": "grok",
        "primary_model": str(model),
        "primary_dispatched": bool(primary_dispatched),
        "probe_returncode": int(probe_returncode),
        **classified,
    }
    receipt["receipt_id"] = _grok_failure_receipt_identity(receipt)
    return receipt


def _grok_failure_receipt_identity(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_id", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def render_grok_failure_receipt(receipt: Mapping[str, Any]) -> str:
    return GROK_FAILURE_RECEIPT_PREFIX + json.dumps(
        dict(receipt),
        sort_keys=True,
        separators=(",", ":"),
    )


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
            value = json.loads(raw)
        except json.JSONDecodeError:
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

    expected_fields = {
        "schema",
        "source",
        "probe_contract_id",
        "nonce",
        "primary_provider",
        "primary_model",
        "primary_dispatched",
        "probe_returncode",
        "failure_class",
        "evidence_sha256",
        "receipt_id",
    }
    observed_returncode = receipt.get("probe_returncode")
    return bool(
        set(receipt) == expected_fields
        and receipt.get("schema") == GROK_FAILURE_RECEIPT_SCHEMA
        and receipt.get("source") == GROK_QUOTA_PROBE_SOURCE
        and receipt.get("probe_contract_id") == GROK_QUOTA_PROBE_CONTRACT_ID
        and re.fullmatch(r"[0-9a-f]{64}", str(nonce or ""))
        and receipt.get("nonce") == nonce
        and receipt.get("primary_provider") == "grok"
        and receipt.get("primary_model") == model == "grok-4.5"
        and receipt.get("primary_dispatched") is False
        and receipt.get("failure_class")
        in {
            "hard_quota_exhausted",
            "authentication",
            "invalid_request",
            "rate_limited",
            "transport",
            "unknown",
        }
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("evidence_sha256") or ""),
        )
        and isinstance(observed_returncode, int)
        and not isinstance(observed_returncode, bool)
        and observed_returncode == returncode != 0
        and receipt.get("receipt_id") == _grok_failure_receipt_identity(receipt)
    )


__all__ = [
    "GROK_FAILURE_RECEIPT_PREFIX",
    "GROK_FAILURE_RECEIPT_SCHEMA",
    "GROK_QUOTA_PROBE_CONTRACT_ID",
    "GROK_QUOTA_PROBE_CONTRACT",
    "GROK_QUOTA_PROBE_PROMPT",
    "GROK_QUOTA_PROBE_SOURCE",
    "GROK_QUOTA_PROBE_TIMEOUT_SECONDS",
    "MAX_GROK_FAILURE_EVIDENCE_BYTES",
    "build_grok_failure_receipt",
    "classify_grok_stderr",
    "extract_grok_failure_receipts",
    "render_grok_failure_receipt",
    "valid_grok_failure_receipt",
    "valid_grok_hard_quota_receipt",
]
