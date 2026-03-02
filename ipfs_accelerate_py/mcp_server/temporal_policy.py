"""Legacy temporal policy surface adapted to canonical MCP++ policy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .mcplusplus.policy_engine import (
    PolicyClause as CanonicalPolicyClause,
    evaluate_policy,
    parse_policy_clauses as _parse_policy_clauses,
)


@dataclass
class PolicyClause:
    """Compatibility clause model for source temporal policy surface."""

    clause_type: str
    actor: str = "*"
    action: str = "*"
    resource: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    obligation_deadline: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_type": self.clause_type,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "obligation_deadline": self.obligation_deadline,
            "metadata": dict(self.metadata),
        }


@dataclass
class PolicyObject:
    """Compatibility policy container for source temporal policy surface."""

    clauses: List[PolicyClause] = field(default_factory=list)
    version: str = "v1"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clauses": [c.to_dict() for c in self.clauses],
            "version": self.version,
            "description": self.description,
        }


def parse_policy_clauses(raw_clauses: Iterable[Dict[str, Any]]) -> List[PolicyClause]:
    """Parse raw clauses through canonical parser and return compatibility wrappers."""
    parsed = _parse_policy_clauses(raw_clauses)
    return [
        PolicyClause(
            clause_type=c.clause_type,
            actor=c.actor,
            action=c.action,
            resource=c.resource,
            valid_from=c.valid_from,
            valid_until=c.valid_until,
            obligation_deadline=c.obligation_deadline,
            metadata=dict(c.metadata or {}),
        )
        for c in parsed
    ]


def make_simple_permission_policy(
    actor: str,
    action: str,
    *,
    resource: Optional[str] = None,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    description: str = "",
) -> PolicyObject:
    """Build source-compatible single-permission policy object."""
    return PolicyObject(
        clauses=[
            PolicyClause(
                clause_type="permission",
                actor=actor,
                action=action,
                resource=resource,
                valid_from=valid_from,
                valid_until=valid_until,
            )
        ],
        description=description or f"Allow {actor} to call {action}",
    )


class PolicyEvaluator:
    """Source-compatible evaluator delegating to canonical policy engine."""

    def evaluate(
        self,
        intent: Any,
        policy: PolicyObject,
        *,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        now: Optional[datetime] = None,
        proofs_checked: Optional[List[str]] = None,
        evaluator_did: Optional[str] = None,
    ) -> Dict[str, Any]:
        tool = str(getattr(intent, "tool", "") or "")
        intent_cid = str(getattr(intent, "intent_cid", "") or "")

        clauses: List[CanonicalPolicyClause] = []
        for item in policy.clauses:
            clauses.append(
                CanonicalPolicyClause(
                    clause_type=item.clause_type,
                    actor=item.actor,
                    action=item.action,
                    resource=item.resource,
                    valid_from=item.valid_from,
                    valid_until=item.valid_until,
                    obligation_deadline=item.obligation_deadline,
                    metadata=dict(item.metadata or {}),
                )
            )

        decision = evaluate_policy(
            clauses=clauses,
            actor=str(actor or "*"),
            action=tool,
            resource=resource,
            now=now,
        )

        return {
            "decision": decision.decision,
            "intent_cid": intent_cid,
            "policy_cid": "",
            "proofs_checked": list(proofs_checked or []),
            "justification": decision.justification,
            "obligations": list(decision.obligations),
            "evaluator_dids": [evaluator_did] if evaluator_did else [],
        }


def get_policy_registry() -> None:
    """Compatibility placeholder retained while NL policy registry remains deferred."""
    return None


__all__ = [
    "PolicyClause",
    "PolicyObject",
    "PolicyEvaluator",
    "parse_policy_clauses",
    "make_simple_permission_policy",
    "get_policy_registry",
]
