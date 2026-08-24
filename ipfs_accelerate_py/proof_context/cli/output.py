"""Stable, identity-bound machine output for the proof-context CLI (PCCE-043).

This module is deliberately a projection layer.  It never decides lifecycle
status, invents evidence, opens a runtime, or performs I/O.  Callers provide a
typed status and observed data; this module makes absence explicit, redacts
secrets, bounds untrusted text, and produces deterministic JSON.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "ipfs-accelerate.proof-context.v0.1/command-result@1"
CONTRACT_VERSION: Final[str] = "0.1"
MAX_TEXT_LENGTH: Final[int] = 4096
MAX_COLLECTION_ITEMS: Final[int] = 64
MAX_MAPPING_ITEMS: Final[int] = 64
MISSING_EVIDENCE: Final[str] = "missing_evidence"

IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "task_id",
    "run_id",
    "patch_id",
)
REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "status",
    "exit_code",
    "trace_id",
    "correlation_id",
    "identities",
    "artifact_cids",
)

# This mapping is intentionally closed: unknown status must fail, never succeed.
EXIT_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "succeeded": 0,
        "failed": 1,
        "timeout": 1,
        "cancelled": 1,
        "verification_failed": 1,
        "proof_failed": 1,
        "assurance_failed": 1,
        "context_insufficient": 1,
        "model_escalation_required": 1,
        "human_review_required": 1,
        "infrastructure_failure": 1,
        "partial_effect": 1,
        "repair_required": 1,
        "invalid": 2,
        "rejected": 3,
        "simulated": 4,
        "unavailable": 5,
        "stale": 6,
    }
)

_SECRET_KEY = re.compile(r"(?:authorization|credential|password|secret|token|api[_-]?key)", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+|api[_-]?key\s*[=:]\s*|token\s*[=:]\s*)[^\s,;]+"
)


def exit_code_for(status: str, *, provenance: str | None = None) -> int:
    """Return the stable non-zero outcome for non-live or unknown results."""

    if status == "succeeded" and provenance not in (None, "live"):
        return EXIT_CODES["simulated"] if provenance == "simulated" else EXIT_CODES["failed"]
    return EXIT_CODES.get(status, EXIT_CODES["failed"])


def _bounded_text(value: Any) -> str:
    text = _SECRET_VALUE.sub("[REDACTED]", str(value))
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH] + "…[truncated]"
    return text


def redact_and_bound(value: Any, *, depth: int = 0) -> Any:
    """Return JSON-safe data without secrets or unbounded command diagnostics."""

    if depth >= 8:
        return "[truncated: maximum nesting]"
    if isinstance(value, Mapping):
        rendered: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_MAPPING_ITEMS:
                rendered["truncated_items"] = len(value) - MAX_MAPPING_ITEMS
                break
            name = _bounded_text(key)
            rendered[name] = "[REDACTED]" if _SECRET_KEY.search(name) else redact_and_bound(item, depth=depth + 1)
        return rendered
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [redact_and_bound(item, depth=depth + 1) for item in value[:MAX_COLLECTION_ITEMS]]
        if len(value) > MAX_COLLECTION_ITEMS:
            items.append("[truncated: additional items omitted]")
        return items
    if isinstance(value, bytes):
        return "[binary content omitted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value)


def normalized_identities(identities: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Preserve all identity slots while representing unobserved values honestly."""

    source = identities or {}
    return {
        field: (_bounded_text(source[field]) if source.get(field) is not None else None)
        for field in IDENTITY_FIELDS
    }


def command_result(
    *,
    status: str,
    correlation_id: str,
    identities: Mapping[str, Any] | None,
    artifact_cids: Sequence[str] | None = None,
    command: str | None = None,
    trace_id: str | None = None,
    provenance: str | None = "live",
    details: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a complete machine envelope without claiming missing evidence.

    ``trace_id`` defaults to the supplied correlation ID because an invocation
    has one trace/correlation identity even when the runtime has not emitted a
    distinct trace ID.  Repository, task, run, and patch identities remain
    ``null`` until observed; they are never synthesized from local paths.
    """

    if not isinstance(correlation_id, str) or not correlation_id.strip():
        raise ValueError("correlation_id is required")
    normalized_status = _bounded_text(status)
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "command": _bounded_text(command) if command else None,
        "status": normalized_status,
        "exit_code": exit_code_for(normalized_status, provenance=provenance),
        "trace_id": _bounded_text(trace_id or correlation_id),
        "correlation_id": _bounded_text(correlation_id),
        "identities": normalized_identities(identities),
        "artifact_cids": [
            _bounded_text(cid) for cid in (artifact_cids or ()) if isinstance(cid, str) and cid
        ][:MAX_COLLECTION_ITEMS],
        "provenance": _bounded_text(provenance) if provenance else None,
        "details": redact_and_bound(details or {}),
        "error": redact_and_bound(error) if error is not None else None,
    }
    return result


def serialize(result: Mapping[str, Any]) -> str:
    """Serialize a result deterministically after enforcing required fields."""

    missing = [field for field in REQUIRED_FIELDS if field not in result]
    if missing:
        raise ValueError("command result missing required fields: " + ", ".join(missing))
    return json.dumps(redact_and_bound(result), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


__all__ = [
    "CONTRACT_VERSION",
    "EXIT_CODES",
    "IDENTITY_FIELDS",
    "MAX_COLLECTION_ITEMS",
    "MAX_TEXT_LENGTH",
    "MISSING_EVIDENCE",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "command_result",
    "exit_code_for",
    "normalized_identities",
    "redact_and_bound",
    "serialize",
]
