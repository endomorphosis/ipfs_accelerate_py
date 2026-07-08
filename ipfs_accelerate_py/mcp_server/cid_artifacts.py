"""Legacy Profile B artifact surface adapted to canonical MCP++ artifacts."""

from __future__ import annotations

from typing import Any, Dict

from .mcplusplus.artifacts import (
    ArtifactStore,
    build_decision,
    build_event,
    build_intent,
    build_receipt,
    canonicalize_artifact,
    compute_artifact_cid,
    envelope_from_payloads,
)


def artifact_cid(obj: Dict[str, Any]) -> str:
    """Source-compatible artifact CID helper."""
    return compute_artifact_cid(dict(obj or {}))


__all__ = [
    "ArtifactStore",
    "artifact_cid",
    "build_intent",
    "build_decision",
    "build_receipt",
    "build_event",
    "canonicalize_artifact",
    "compute_artifact_cid",
    "envelope_from_payloads",
]
