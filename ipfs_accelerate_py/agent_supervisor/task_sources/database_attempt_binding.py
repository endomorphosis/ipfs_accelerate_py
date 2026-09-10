"""Closed, self-hashed database Portal attempt-binding validation.

Version 2 retains all version-1 claim fields and binds the claim-time control
projection. Validation grants no execution or completion authority; callers
must still compare the binding with their authoritative receipt and fences.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_PREFIX = "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@"
_BASE_FIELDS = frozenset({
    "schema", "interface", "attempt_id", "claim_id", "task_cid", "task_alias",
    "goal_cid", "plan_cid", "task_revision", "fencing_token", "fence_epoch",
    "lease_id", "task_body_digest", "projection_seed_digest",
    "projection_immutable_digest", "authoritative_task_store",
    "projection_authority", "binding_id",
})
_CONTROL_FIELDS = frozenset({
    "control_binding_id", "control_task_projection_cid",
    "control_expected_revision", "control_portal_binding_basis_cid",
})


def validate_database_attempt_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy, rejecting unknown schemas, fields and coercions."""
    if not isinstance(value, Mapping):
        raise ValueError("database Portal attempt binding must be an object")
    value = dict(value)
    schema = value.get("schema")
    if schema not in (_PREFIX + "1", _PREFIX + "2"):
        raise ValueError("database Portal attempt binding schema is invalid")
    expected = _BASE_FIELDS | (_CONTROL_FIELDS if schema == _PREFIX + "2" else frozenset())
    if set(value) != expected:
        raise ValueError("database Portal attempt binding fields are invalid")
    for field in ("task_revision", "fencing_token", "fence_epoch"):
        if type(value[field]) is not int or value[field] < 1:
            raise ValueError("database Portal attempt binding integer is invalid")
    for field in ("attempt_id", "claim_id", "task_cid", "task_alias", "lease_id",
                  "task_body_digest", "projection_seed_digest", "projection_immutable_digest"):
        if type(value[field]) is not str or not value[field]:
            raise ValueError("database Portal attempt binding identity is invalid")
    if (type(value["goal_cid"]) is not str or type(value["plan_cid"]) is not str
            or value["interface"] != "DatabasePortalExecutionBridge@1"
            or value["authoritative_task_store"] != "duckdb"
            or value["projection_authority"] is not False):
        raise ValueError("database Portal attempt binding authority is invalid")
    if schema == _PREFIX + "2":
        if (type(value["control_expected_revision"]) is not int
                or value["control_expected_revision"] != value["task_revision"]):
            raise ValueError("database Portal control revision is invalid")
        for field in _CONTROL_FIELDS - {"control_expected_revision"}:
            if type(value[field]) is not str or not value[field]:
                raise ValueError("database Portal control identity is invalid")
    normalized = dict(value)
    binding_id = normalized.pop("binding_id")
    digest = "sha256:" + hashlib.sha256(json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if type(binding_id) is not str or binding_id != digest:
        raise ValueError("database Portal attempt binding digest is invalid")
    return value
