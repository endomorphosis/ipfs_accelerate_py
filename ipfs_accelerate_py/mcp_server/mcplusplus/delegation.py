"""UCAN-style delegation validation for MCP++ Profile C.

This module provides deterministic, dependency-light validation of delegation
chains for execution-time authorization checks in unified dispatch paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class UcanCapability:
    """Capability tuple for resource/ability checks."""

    resource: str
    ability: str

    def matches(self, *, resource: str, ability: str) -> bool:
        resource_ok = self.resource in {"*", resource}
        ability_ok = self.ability in {"*", ability}
        return resource_ok and ability_ok


@dataclass(frozen=True)
class UcanDelegation:
    """Delegation edge from issuer -> audience with capability set."""

    issuer: str
    audience: str
    capabilities: Tuple[UcanCapability, ...]
    expiry: float | None = None
    revoked: bool = False

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.expiry is None:
            return False
        t = float(now if now is not None else time.time())
        return t > float(self.expiry)


@dataclass(frozen=True)
class UcanValidationResult:
    """Structured result for delegation chain validation."""

    allowed: bool
    reason: str
    chain_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "chain_length": self.chain_length,
        }


def _parse_capabilities(raw: Iterable[Dict[str, Any]]) -> Tuple[UcanCapability, ...]:
    caps: List[UcanCapability] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resource = str(item.get("resource", "") or "").strip() or "*"
        ability = str(item.get("ability", "") or "").strip() or "*"
        caps.append(UcanCapability(resource=resource, ability=ability))
    return tuple(caps)


def parse_delegation_chain(raw_chain: Iterable[Dict[str, Any]]) -> List[UcanDelegation]:
    """Parse raw dict chain into typed delegation entries."""
    chain: List[UcanDelegation] = []
    for item in raw_chain:
        if not isinstance(item, dict):
            continue
        chain.append(
            UcanDelegation(
                issuer=str(item.get("issuer", "") or "").strip(),
                audience=str(item.get("audience", "") or "").strip(),
                capabilities=_parse_capabilities(item.get("capabilities", []) or []),
                expiry=float(item["expiry"]) if item.get("expiry") is not None else None,
                revoked=bool(item.get("revoked", False)),
            )
        )
    return chain


def _covers(parent: UcanDelegation, child: UcanDelegation) -> bool:
    """Return True if child's capabilities are attenuated by parent's set."""
    if not child.capabilities:
        return False
    for child_cap in child.capabilities:
        ok = any(
            parent_cap.matches(resource=child_cap.resource, ability=child_cap.ability)
            for parent_cap in parent.capabilities
        )
        if not ok:
            return False
    return True


def validate_delegation_chain(
    *,
    chain: Iterable[UcanDelegation],
    resource: str,
    ability: str,
    actor: str = "",
    now: float | None = None,
) -> UcanValidationResult:
    """Validate a root->leaf delegation chain for execution authorization."""
    parsed = list(chain)
    if not parsed:
        return UcanValidationResult(False, "missing_delegation_chain", 0)

    for idx, d in enumerate(parsed):
        if not d.issuer or not d.audience:
            return UcanValidationResult(False, f"invalid_principal_at_hop_{idx}", len(parsed))
        if d.revoked:
            return UcanValidationResult(False, f"revoked_at_hop_{idx}", len(parsed))
        if d.is_expired(now=now):
            return UcanValidationResult(False, f"expired_at_hop_{idx}", len(parsed))

    # issuer/audience continuity: audience(i) == issuer(i+1)
    for idx in range(len(parsed) - 1):
        if parsed[idx].audience != parsed[idx + 1].issuer:
            return UcanValidationResult(False, f"broken_chain_at_hop_{idx}", len(parsed))

    # attenuation: each child must be subset of parent capabilities
    for idx in range(len(parsed) - 1):
        if not _covers(parsed[idx], parsed[idx + 1]):
            return UcanValidationResult(False, f"capability_escalation_at_hop_{idx+1}", len(parsed))

    leaf = parsed[-1]
    if actor and leaf.audience != actor:
        return UcanValidationResult(False, "actor_mismatch", len(parsed))

    granted = any(cap.matches(resource=resource, ability=ability) for cap in leaf.capabilities)
    if not granted:
        return UcanValidationResult(False, "capability_not_granted", len(parsed))

    return UcanValidationResult(True, "allowed", len(parsed))


def validate_raw_delegation_chain(
    *,
    raw_chain: Iterable[Dict[str, Any]],
    resource: str,
    ability: str,
    actor: str = "",
    now: float | None = None,
) -> UcanValidationResult:
    """Convenience validator for raw dict chain payloads."""
    chain = parse_delegation_chain(raw_chain)
    return validate_delegation_chain(
        chain=chain,
        resource=resource,
        ability=ability,
        actor=actor,
        now=now,
    )


__all__ = [
    "UcanCapability",
    "UcanDelegation",
    "UcanValidationResult",
    "parse_delegation_chain",
    "validate_delegation_chain",
    "validate_raw_delegation_chain",
]
