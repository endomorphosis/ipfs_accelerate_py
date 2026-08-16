"""SemanticBindingAdmission@1 — typed admission from records, not Markdown."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

SCHEMA = "lgswf/semantic-binding-admission@1"
SENTINELS = {
    "ACCEPTED_LGSWF-006_SOURCE_HEAD",
    "REBIND_REQUIRED_BY_LGSWF-005",
}


class BindingAdmissionError(ValueError):
    """Admission refused a missing, stale, or sentinel semantic field."""


def project_task_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("treat_markdown_as_authority"):
        raise BindingAdmissionError("Markdown prose is not semantic authority")
    projected = {
        "entity_id": record.get("task_id") or record.get("entity_id"),
        "accepted_plan_cid": record.get("accepted_plan_cid"),
        "semantic_state_root_cid": record.get("semantic_state_root_cid"),
        "completion_rule": record.get("completion_rule"),
        "binding_cid": record.get("binding_cid"),
    }
    reasons = []
    for key, value in projected.items():
        if not value:
            reasons.append(f"missing:{key}")
        elif value in SENTINELS:
            reasons.append(f"sentinel:{key}")
    if record.get("claimed_history_mutation"):
        raise BindingAdmissionError("claimed/running specifications are immutable")
    ready = not reasons
    return MappingProxyType(
        {
            "schema": SCHEMA,
            "ready": ready,
            "reasons": tuple(reasons),
            "projected": MappingProxyType({k: v for k, v in projected.items() if v}),
        }
    )


def admit_executable(record: Mapping[str, Any]) -> Mapping[str, Any]:
    projection = project_task_record(record)
    if not projection["ready"]:
        return projection
    if record.get("stale_binding"):
        return MappingProxyType(
            {
                "schema": SCHEMA,
                "ready": False,
                "reasons": ("stale-binding",),
                "projected": projection["projected"],
            }
        )
    return MappingProxyType(
        {
            "schema": SCHEMA,
            "ready": True,
            "reasons": (),
            "projected": projection["projected"],
        }
    )
