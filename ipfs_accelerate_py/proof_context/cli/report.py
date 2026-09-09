"""Bounded, truthful human patch reports for PCCE CLI outcomes (PCCE-043)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ipfs_accelerate_py.proof_context.cli.output import (
    MISSING_EVIDENCE,
    command_result,
    redact_and_bound,
)

REPORT_SCHEMA_VERSION: Final[str] = "ipfs-accelerate.proof-context.v0.1/human-patch-report@1"
REPORT_SECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("Task", "task"),
    ("Revision", "revision"),
    ("Routing", "routing"),
    ("Context", "context"),
    ("Changes", "changes"),
    ("Verification / proof reuse", "verification_proof_reuse"),
    ("Assurance", "assurance"),
    ("Escalation", "escalation"),
    ("Costs", "costs"),
    ("Review", "review"),
    ("Receipts", "receipts"),
    ("Seal", "seal"),
)


def _display(value: Any) -> str:
    """Render values visibly, retaining an explicit missing-evidence label."""

    value = redact_and_bound(value)
    if value is None or value == "" or value == {} or value == []:
        return MISSING_EVIDENCE
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_display(value[key])}" for key in sorted(value, key=str))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(_display(item) for item in value)
    return str(value)


def _cost_display(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _display(value)
    amount = value.get("amount", value.get("value"))
    basis = value.get("basis")
    label = value.get("label")
    if label not in {"observed", "estimated", "baseline", MISSING_EVIDENCE}:
        label = MISSING_EVIDENCE
    return f"{label}: {_display(amount)}; basis: {_display(basis)}"


def render_human_patch_report(
    result: Mapping[str, Any] | None = None,
    *,
    status: str | None = None,
    correlation_id: str | None = None,
    identities: Mapping[str, Any] | None = None,
    artifact_cids: Sequence[str] | None = None,
    fields: Mapping[str, Any] | None = None,
) -> str:
    """Render every required patch-report section for pass and non-pass states.

    The caller may pass an existing machine result or raw result arguments.
    Costs are never presented as observed unless the supplied label says so;
    absent sections and identities remain ``missing_evidence``.
    """

    if result is None:
        if status is None or correlation_id is None:
            raise ValueError("status and correlation_id are required without a result")
        result = command_result(
            status=status,
            correlation_id=correlation_id,
            identities=identities,
            artifact_cids=artifact_cids,
            details=fields,
        )
    safe = redact_and_bound(result)
    details = safe.get("details", {}) if isinstance(safe, Mapping) else {}
    if not isinstance(details, Mapping):
        details = {}
    identities_value = safe.get("identities", {}) if isinstance(safe, Mapping) else {}
    lines = [
        f"Proof-carrying context patch report ({REPORT_SCHEMA_VERSION})",
        f"Command: {_display(safe.get('command'))}",
        f"Status: {_display(safe.get('status'))}",
        f"Exit code: {_display(safe.get('exit_code'))}",
        f"Provenance: {_display(safe.get('provenance'))}",
        f"Trace / correlation: {_display(safe.get('trace_id'))} / {_display(safe.get('correlation_id'))}",
        f"Repository identity: {_display(identities_value.get('repository_id') if isinstance(identities_value, Mapping) else None)}",
        f"Run identity: {_display(identities_value.get('run_id') if isinstance(identities_value, Mapping) else None)}",
        f"Patch identity: {_display(identities_value.get('patch_id') if isinstance(identities_value, Mapping) else None)}",
        f"Artifact CIDs: {_display(safe.get('artifact_cids'))}",
    ]
    for title, key in REPORT_SECTIONS:
        value = details.get(key)
        if key == "task" and value in (None, "", {}, []):
            value = (
                identities_value.get("task_id") if isinstance(identities_value, Mapping) else None
            )
        rendered = _cost_display(value) if key == "costs" else _display(value)
        lines.append(f"{title}: {rendered}")
    return "\n".join(lines) + "\n"


__all__ = ["REPORT_SCHEMA_VERSION", "REPORT_SECTIONS", "render_human_patch_report"]
