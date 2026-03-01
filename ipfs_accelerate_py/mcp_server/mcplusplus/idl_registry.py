"""MCP-IDL registry primitives for unified MCP++ runtime.

This module provides deterministic descriptor canonicalization and a small
in-memory interface repository used by Profile A (`mcp-idl`) tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional


def canonicalize_descriptor(descriptor: Dict[str, Any]) -> bytes:
    """Return deterministic canonical bytes for an interface descriptor."""
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_interface_cid(descriptor: Dict[str, Any]) -> str:
    """Compute stable content-addressed identifier for descriptor bytes.

    The implementation uses a deterministic SHA-256 digest representation as a
    lightweight CID placeholder for current migration phases.
    """
    digest = hashlib.sha256(canonicalize_descriptor(descriptor)).hexdigest()
    return f"cidv1-sha256-{digest}"


def build_descriptor(
    *,
    name: str,
    namespace: str,
    version: str,
    methods: List[Dict[str, Any]],
    errors: Optional[List[Dict[str, Any]]] = None,
    requires: Optional[List[str]] = None,
    compatibility: Optional[Dict[str, List[str]]] = None,
    semantic_tags: Optional[List[str]] = None,
    observability: Optional[Dict[str, Any]] = None,
    interaction_patterns: Optional[Dict[str, Any]] = None,
    resource_cost_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a descriptor with required and optional MCP-IDL fields."""
    return {
        "name": name,
        "namespace": namespace,
        "version": version,
        "methods": methods,
        "errors": errors or [],
        "requires": requires or [],
        "compatibility": compatibility or {"compatible_with": [], "supersedes": []},
        "semantic_tags": semantic_tags or [],
        "observability": observability or {"trace": True, "provenance": True},
        "interaction_patterns": interaction_patterns or {"request_response": True, "event_streams": False},
        "resource_cost_hints": resource_cost_hints or {},
    }


@dataclass(frozen=True)
class CompatibilityVerdict:
    """Compatibility result shape for `interfaces/compat`."""

    compatible: bool
    reasons: List[str]
    requires_missing: List[str]
    suggested_alternatives: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "reasons": list(self.reasons),
            "requires_missing": list(self.requires_missing),
            "suggested_alternatives": list(self.suggested_alternatives),
        }


class InterfaceDescriptorRegistry:
    """In-memory MCP-IDL descriptor repository."""

    def __init__(self, supported_capabilities: Optional[Iterable[str]] = None) -> None:
        self._by_cid: Dict[str, Dict[str, Any]] = {}
        self._supported_capabilities = set(supported_capabilities or [])

    @property
    def supported_capabilities(self) -> set[str]:
        """Return supported capabilities used for compatibility checks."""
        return set(self._supported_capabilities)

    def set_supported_capabilities(self, capabilities: Iterable[str]) -> None:
        """Replace capability set used by compatibility checks."""
        self._supported_capabilities = set(capabilities)

    def register_descriptor(self, descriptor: Dict[str, Any]) -> str:
        """Register a descriptor and return its computed `interface_cid`."""
        interface_cid = compute_interface_cid(descriptor)
        payload = dict(descriptor)
        payload["interface_cid"] = interface_cid
        self._by_cid[interface_cid] = payload
        return interface_cid

    def list_interfaces(self) -> List[str]:
        """List registered interface CIDs in deterministic order."""
        return sorted(self._by_cid.keys())

    def get_descriptor(self, interface_cid: str) -> Optional[Dict[str, Any]]:
        """Get descriptor payload for CID, if present."""
        payload = self._by_cid.get(interface_cid)
        return dict(payload) if payload is not None else None

    def compat(self, interface_cid: str) -> CompatibilityVerdict:
        """Evaluate compatibility against local supported capabilities."""
        descriptor = self._by_cid.get(interface_cid)
        if descriptor is None:
            return CompatibilityVerdict(
                compatible=False,
                reasons=["interface_not_found"],
                requires_missing=[],
                suggested_alternatives=[],
            )

        requires = [str(x) for x in descriptor.get("requires", [])]
        missing = sorted([req for req in requires if req not in self._supported_capabilities])
        if missing:
            return CompatibilityVerdict(
                compatible=False,
                reasons=["missing_required_capabilities"],
                requires_missing=missing,
                suggested_alternatives=[
                    cid
                    for cid, payload in self._by_cid.items()
                    if cid != interface_cid and not set(payload.get("requires", [])).difference(self._supported_capabilities)
                ],
            )

        return CompatibilityVerdict(
            compatible=True,
            reasons=[],
            requires_missing=[],
            suggested_alternatives=[],
        )

    def select(self, task_hint_cid: str = "", budget: int = 20) -> List[str]:
        """Return a deterministic, budgeted subset of compatible interfaces."""
        _ = task_hint_cid  # Reserved for future ranking heuristics.
        compatible = [cid for cid in self.list_interfaces() if self.compat(cid).compatible]
        return compatible[: max(0, int(budget))]


__all__ = [
    "CompatibilityVerdict",
    "InterfaceDescriptorRegistry",
    "build_descriptor",
    "canonicalize_descriptor",
    "compute_interface_cid",
]
