"""SemanticWorkBinding@1 — reference-only goal/subgoal/task/attempt binding."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA = "lgswf/semantic-work-binding@1"
INTERFACE = "SemanticWorkBinding@1"

REQUIRED_FIELDS = (
    "entity_kind",
    "entity_id",
    "accepted_plan_cid",
    "repository_tree_cid",
    "semantic_state_root_cid",
    "target_symbol_cids",
    "target_artifact_cids",
    "target_capsule_cids",
    "raw_source_requirements",
    "environment_binding_cids",
    "preconditions",
    "postconditions",
    "exceptional_conditions",
    "allowed_effects",
    "prohibited_effects",
    "effect_scope",
    "test_cids",
    "proof_cids",
    "assumptions",
    "limitations",
    "counterexamples",
    "invalidation",
    "completion_rule",
    "required_authority",
    "human_review",
)

ENTITY_KINDS = frozenset({"goal", "subgoal", "task", "attempt"})
FORBIDDEN = frozenset(
    {
        "capsule_body",
        "contract_body",
        "proof_body",
        "raw_source",
        "prompt",
        "credential",
        "provider_payload",
    }
)


class WorkBindingError(ValueError):
    """SemanticWorkBinding@1 rejected the payload."""


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_cid(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def field_ownership_table() -> tuple[dict[str, str], ...]:
    rows = []
    for field in REQUIRED_FIELDS:
        authority = (
            "datasets-semantic"
            if field
            in {
                "semantic_state_root_cid",
                "target_symbol_cids",
                "target_artifact_cids",
                "target_capsule_cids",
                "environment_binding_cids",
            }
            else "operational"
        )
        rows.append(
            {
                "field": field,
                "owner": (
                    "ipfs_datasets_py"
                    if authority == "datasets-semantic"
                    else "ipfs_accelerate_py"
                ),
                "authority": authority,
            }
        )
    return tuple(rows)


def parse_work_binding(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise WorkBindingError("binding must be a mapping")
    if set(payload) & FORBIDDEN:
        raise WorkBindingError("binding must not copy semantic or provider payloads")
    unknown = set(payload) - set(REQUIRED_FIELDS) - {"schema", "interface", "binding_cid"}
    if unknown:
        raise WorkBindingError(f"unknown fields: {sorted(unknown)}")
    if payload.get("schema", SCHEMA) != SCHEMA:
        raise WorkBindingError("unsupported schema")
    if payload.get("entity_kind") not in ENTITY_KINDS:
        raise WorkBindingError("entity_kind must be goal/subgoal/task/attempt")
    entity_id = payload.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id:
        raise WorkBindingError("entity_id is required")
    for cid_field in (
        "accepted_plan_cid",
        "repository_tree_cid",
        "semantic_state_root_cid",
    ):
        value = payload.get(cid_field)
        if value in {
            "ACCEPTED_LGSWF-006_SOURCE_HEAD",
            "REBIND_REQUIRED_BY_LGSWF-005",
        }:
            raise WorkBindingError(f"stale sentinel on {cid_field}")
        if not _is_cid(value):
            raise WorkBindingError(f"malformed CID on {cid_field}")
    lists = (
        "target_symbol_cids",
        "target_artifact_cids",
        "target_capsule_cids",
        "environment_binding_cids",
        "test_cids",
        "proof_cids",
    )
    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "entity_kind": payload["entity_kind"],
        "entity_id": entity_id,
        "accepted_plan_cid": payload["accepted_plan_cid"],
        "repository_tree_cid": payload["repository_tree_cid"],
        "semantic_state_root_cid": payload["semantic_state_root_cid"],
    }
    for field in lists:
        values = payload.get(field, ())
        if not isinstance(values, (list, tuple)):
            raise WorkBindingError(f"{field} must be a list of CIDs")
        if any(not _is_cid(item) for item in values):
            raise WorkBindingError(f"forged or malformed reference in {field}")
        normalized[field] = tuple(values)
    for field in (
        "raw_source_requirements",
        "preconditions",
        "postconditions",
        "exceptional_conditions",
        "allowed_effects",
        "prohibited_effects",
        "assumptions",
        "limitations",
        "counterexamples",
        "invalidation",
    ):
        values = payload.get(field, ())
        if values is None:
            raise WorkBindingError(f"{field} may not be fabricated; use typed hold")
        if not isinstance(values, (list, tuple, str)):
            raise WorkBindingError(f"{field} has unsupported type")
        normalized[field] = values if isinstance(values, str) else tuple(values)
    for field in ("effect_scope", "completion_rule", "required_authority", "human_review"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise WorkBindingError(f"{field} requires a typed value, not a default")
        normalized[field] = value
    if normalized["required_authority"] != "supervisor":
        raise WorkBindingError("required_authority must remain supervisor")
    identity = {key: normalized[key] for key in sorted(normalized)}
    normalized["binding_cid"] = _sha256_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    )
    provided = payload.get("binding_cid")
    if provided and provided != normalized["binding_cid"]:
        raise WorkBindingError("binding_cid does not match canonical identity")
    return MappingProxyType(normalized)


def example_binding(*, entity_id: str = "LGSWF-020") -> Mapping[str, Any]:
    cid = _sha256_text
    return parse_work_binding(
        {
            "schema": SCHEMA,
            "entity_kind": "task",
            "entity_id": entity_id,
            "accepted_plan_cid": cid("plan"),
            "repository_tree_cid": cid("tree"),
            "semantic_state_root_cid": cid("semantic"),
            "target_symbol_cids": (cid("symbol"),),
            "target_artifact_cids": (),
            "target_capsule_cids": (cid("capsule"),),
            "raw_source_requirements": ("exact-r2-fallback",),
            "environment_binding_cids": (cid("env"),),
            "preconditions": ("current-binding",),
            "postconditions": ("validated",),
            "exceptional_conditions": ("stale-root",),
            "allowed_effects": ("write-owned-paths",),
            "prohibited_effects": ("copy-capsule-body",),
            "effect_scope": "owned-paths",
            "test_cids": (cid("test"),),
            "proof_cids": (cid("proof"),),
            "assumptions": ("r2-admitted",),
            "limitations": ("no-payload-copy",),
            "counterexamples": (),
            "invalidation": ("root-change",),
            "completion_rule": "supervisor-acceptance",
            "required_authority": "supervisor",
            "human_review": "not-required",
        }
    )
