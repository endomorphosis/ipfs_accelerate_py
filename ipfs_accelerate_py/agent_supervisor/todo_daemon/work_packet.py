"""DaemonWorkPacket@current-version-extension."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA = "lgswf/daemon-work-packet@1"
REQUIRED = (
    "goal_cid",
    "plan_cid",
    "tree_cid",
    "semantic_state_root_cid",
    "context_cid",
    "scope",
    "effects",
    "resource_vector",
    "provider",
    "model",
    "validation",
    "proof",
    "completion_rule",
    "lease_id",
    "fence_epoch",
    "attempt_id",
    "idempotency_key",
    "checkpoint_cid",
    "cancellation",
    "output_cid",
    "control_plane_profile",
    "state_owner_identity",
    "state_owner_epoch",
    "repository_capabilities",
)
FORBIDDEN = frozenset(
    {"quack_endpoint", "duckdb_path", "credential", "prompt", "provider_payload", "secret"}
)
SENTINELS = {"REBIND_REQUIRED_BY_LGSWF-005", "ACCEPTED_LGSWF-006_SOURCE_HEAD"}


class WorkPacketError(ValueError):
    """Daemon work packet rejected."""


def parse_work_packet(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(payload) & FORBIDDEN:
        raise WorkPacketError("forbidden embedded endpoint/path/secret")
    unknown = set(payload) - set(REQUIRED) - {"schema", "mode", "packet_cid"}
    if unknown:
        raise WorkPacketError(f"unknown fields: {sorted(unknown)}")
    for field in REQUIRED:
        if field not in payload:
            raise WorkPacketError(f"missing {field}")
        value = payload[field]
        if isinstance(value, str) and value in SENTINELS:
            raise WorkPacketError(f"stale sentinel on {field}")
    mode = payload.get("mode") or "embedded-one-writer"
    if mode not in {"embedded-one-writer", "quack-exact-1.5.5"}:
        raise WorkPacketError("generic 1.5 profile cannot grant multi-process authority")
    if payload.get("direct_multiprocess_duckdb"):
        raise WorkPacketError("direct multi-process DuckDB path rejected")
    body = {field: payload[field] for field in REQUIRED}
    body["schema"] = SCHEMA
    body["mode"] = mode
    body["packet_cid"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return MappingProxyType(body)
