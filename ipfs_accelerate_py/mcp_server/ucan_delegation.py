"""Legacy UCAN delegation surface adapted to canonical MCP++ delegation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .mcplusplus.delegation import (
    UcanCapability,
    UcanDelegation,
    compute_delegation_proof_cid,
    parse_delegation_chain,
    validate_delegation_chain,
)


@dataclass
class Capability:
    """Legacy compatibility capability surface."""

    resource: str
    ability: str

    def matches(self, resource: str, ability: str) -> bool:
        return UcanCapability(resource=self.resource, ability=self.ability).matches(
            resource=resource,
            ability=ability,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"resource": self.resource, "ability": self.ability}


@dataclass
class Delegation:
    """Legacy compatibility delegation surface."""

    cid: str
    issuer: str
    audience: str
    capabilities: List[Capability]
    expiry: Optional[float] = None
    proof_cid: Optional[str] = None
    signature: Optional[str] = None
    revoked: bool = False
    caveats: List[Dict[str, Any]] = field(default_factory=list)

    def to_ucan(self) -> UcanDelegation:
        return UcanDelegation(
            issuer=self.issuer,
            audience=self.audience,
            capabilities=tuple(
                UcanCapability(resource=c.resource, ability=c.ability) for c in (self.capabilities or [])
            ),
            expiry=self.expiry,
            revoked=bool(self.revoked),
            proof_cid=str(self.proof_cid or ""),
            signature=str(self.signature or ""),
            caveats=tuple(dict(c) for c in (self.caveats or [])),
        )

    @classmethod
    def from_ucan(cls, delegation: UcanDelegation, *, cid: str = "") -> "Delegation":
        resolved_cid = cid or compute_delegation_proof_cid(delegation)
        return cls(
            cid=resolved_cid,
            issuer=delegation.issuer,
            audience=delegation.audience,
            capabilities=[Capability(resource=c.resource, ability=c.ability) for c in delegation.capabilities],
            expiry=delegation.expiry,
            proof_cid=delegation.proof_cid or None,
            signature=delegation.signature or None,
            revoked=delegation.revoked,
            caveats=[dict(c) for c in delegation.caveats],
        )

    def is_expired(self, now: Optional[float] = None) -> bool:
        return self.to_ucan().is_expired(now=now)

    def has_capability(self, resource: str, ability: str) -> bool:
        return any(c.matches(resource, ability) for c in self.capabilities)


class DelegationEvaluator:
    """Legacy evaluator facade backed by canonical validation helpers."""

    def __init__(self, max_chain_depth: int = 0) -> None:
        self._store: Dict[str, Delegation] = {}
        self._max_chain_depth = int(max_chain_depth)

    def add(self, delegation: Delegation) -> None:
        self._store[str(delegation.cid)] = delegation

    def get(self, cid: str) -> Optional[Delegation]:
        return self._store.get(str(cid))

    def remove(self, cid: str) -> bool:
        key = str(cid)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def list_cids(self) -> List[str]:
        return list(self._store.keys())

    def build_chain(self, leaf_cid: str) -> List[Delegation]:
        chain: List[Delegation] = []
        current_cid = str(leaf_cid)
        seen: set[str] = set()

        while current_cid:
            if current_cid in seen:
                raise ValueError(f"Cycle detected in delegation chain at CID '{current_cid}'")
            seen.add(current_cid)

            current = self._store.get(current_cid)
            if current is None:
                if not chain:
                    return []
                raise KeyError(f"Delegation '{current_cid}' not found in store")

            chain.append(current)
            current_cid = str(current.proof_cid or "")

        chain.reverse()
        if self._max_chain_depth > 0 and len(chain) > self._max_chain_depth:
            raise ValueError(
                f"Delegation chain length {len(chain)} exceeds max_chain_depth {self._max_chain_depth}"
            )
        return chain

    def can_invoke(
        self,
        leaf_cid: str,
        resource: str,
        ability: str,
        actor: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if str(leaf_cid) not in self._store:
            return False, f"Delegation '{leaf_cid}' not found"

        chain = self.build_chain(str(leaf_cid))
        if not chain:
            return False, "Empty delegation chain"

        if actor and chain[-1].audience != actor:
            return False, "Leaf audience does not match actor"

        ucan_chain = [d.to_ucan() for d in chain]
        result = validate_delegation_chain(
            chain=ucan_chain,
            actor=actor or chain[-1].audience,
            resource=resource,
            ability=ability,
            now=now,
        )
        return bool(result.allowed), str(result.reason)


_default_evaluator = DelegationEvaluator()


def get_delegation_evaluator() -> DelegationEvaluator:
    return _default_evaluator


def add_delegation(delegation: Delegation) -> None:
    _default_evaluator.add(delegation)


def get_delegation(cid: str) -> Optional[Delegation]:
    return _default_evaluator.get(cid)


@dataclass
class InvocationContext:
    """Legacy invocation-context helper for execution proof metadata."""

    actor: str
    resource: str
    ability: str
    proof_cids: List[str] = field(default_factory=list)


def _normalize_cid(raw: str) -> str:
    return str(raw or "").strip()


def parse_raw_chain(raw_chain: Iterable[Dict[str, Any]]) -> List[Delegation]:
    """Parse legacy raw chain payload into compatibility delegations."""
    chain = parse_delegation_chain(raw_chain)
    out: List[Delegation] = []
    previous_cid = ""
    for item in chain:
        cid = compute_delegation_proof_cid(item)
        if previous_cid and not item.proof_cid:
            proof_cid = previous_cid
        else:
            proof_cid = _normalize_cid(item.proof_cid)
        out.append(Delegation.from_ucan(item, cid=cid if cid else "",))
        if out:
            out[-1].proof_cid = proof_cid or None
        previous_cid = cid
    return out


__all__ = [
    "Capability",
    "Delegation",
    "DelegationEvaluator",
    "InvocationContext",
    "add_delegation",
    "get_delegation",
    "get_delegation_evaluator",
    "parse_raw_chain",
]
