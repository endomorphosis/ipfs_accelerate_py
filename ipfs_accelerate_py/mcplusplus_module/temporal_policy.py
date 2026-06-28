"""Profile D: Temporal Deontic Policy Evaluation for ipfs_accelerate_py.

Implements the MCP++ Profile D specification:
- Permissions — what is allowed
- Prohibitions — what is forbidden
- Obligations — what must be done (with deadlines)
- Temporal constraints — validity windows, deadlines

Integrates with the CID-native execution pipeline (cid_ucan.py) to
enforce policies at invocation time.

Module: ipfs_accelerate_py.mcplusplus_module.temporal_policy
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from .cid_ucan import compute_cid


# ---------------------------------------------------------------------------
# Policy representation
# ---------------------------------------------------------------------------

@dataclass
class PolicyClause:
    """A single deontic clause: permission, prohibition, or obligation."""

    clause_type: str  # "permission" | "prohibition" | "obligation"
    actor: str = "*"
    action: str = "*"
    resource: Optional[str] = None
    valid_from: Optional[float] = None  # Unix timestamp
    valid_until: Optional[float] = None  # Unix timestamp
    obligation_deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_temporally_valid(self, now: Optional[float] = None) -> bool:
        """Return whether this clause is active at the given time."""
        t = now or time.time()
        if self.valid_from is not None and t < self.valid_from:
            return False
        if self.valid_until is not None and t > self.valid_until:
            return False
        return True

    def matches(self, actor: str, action: str, resource: Optional[str] = None,
                now: Optional[float] = None) -> bool:
        """Return True if this clause applies to the given context."""
        if not self.is_temporally_valid(now):
            return False
        if self.actor != "*" and self.actor != actor:
            return False
        if self.action != "*" and self.action != action:
            return False
        if self.resource is not None and resource is not None:
            if self.resource != "*" and self.resource != resource:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_type": self.clause_type,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "obligation_deadline": self.obligation_deadline,
            "metadata": self.metadata,
        }


@dataclass
class PolicyObject:
    """A complete temporal deontic policy with CID addressing."""

    name: str
    clauses: List[PolicyClause] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "policy",
                "name": self.name,
                "clauses": [c.to_dict() for c in self.clauses],
                "version": self.version,
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "clauses": [c.to_dict() for c in self.clauses],
        }


@dataclass
class PolicyDecision:
    """Result of evaluating an intent against a policy."""

    verdict: str  # "allow" | "deny" | "allow_with_obligations"
    policy_cid: str
    justification: str = ""
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        return self.verdict in ("allow", "allow_with_obligations")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "policy_cid": self.policy_cid,
            "justification": self.justification,
            "obligations": self.obligations,
            "allowed": self.allowed,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Policy Evaluator
# ---------------------------------------------------------------------------

class PolicyEvaluator:
    """Runtime policy evaluator for Profile D.

    Evaluates an intent (method + params + actor) against a PolicyObject:
    - If any matching prohibition exists → deny
    - If a matching permission exists → allow (with obligations if any)
    - Otherwise → deny (closed-world assumption)
    """

    def __init__(self):
        self._policies: Dict[str, PolicyObject] = {}

    def register(self, policy: PolicyObject) -> str:
        """Register a policy. Returns its CID."""
        self._policies[policy.cid] = policy
        return policy.cid

    def get(self, policy_cid: str) -> Optional[PolicyObject]:
        return self._policies.get(policy_cid)

    def evaluate(self, method: str, actor: str = "*",
                 resource: Optional[str] = None,
                 policy_cid: Optional[str] = None,
                 now: Optional[float] = None) -> PolicyDecision:
        """Evaluate whether actor can invoke method under the given policy.

        If policy_cid is None, evaluates against ALL registered policies
        (most restrictive wins — any deny = deny).
        """
        t = now or time.time()

        if policy_cid:
            policy = self._policies.get(policy_cid)
            if not policy:
                return PolicyDecision(
                    verdict="deny", policy_cid=policy_cid,
                    justification=f"Unknown policy: {policy_cid}",
                )
            return self._evaluate_single(method, actor, resource, policy, t)

        # Evaluate all policies — aggregate
        if not self._policies:
            # No policies registered = open access
            return PolicyDecision(
                verdict="allow", policy_cid="",
                justification="No policies registered (open access)",
            )

        all_decisions = []
        for pid, policy in self._policies.items():
            all_decisions.append(self._evaluate_single(method, actor, resource, policy, t))

        # Any deny = deny (most restrictive)
        denials = [d for d in all_decisions if d.verdict == "deny"]
        if denials:
            return denials[0]

        # Merge obligations
        obligations = []
        for d in all_decisions:
            obligations.extend(d.obligations)

        if obligations:
            return PolicyDecision(
                verdict="allow_with_obligations",
                policy_cid=all_decisions[0].policy_cid if all_decisions else "",
                justification=f"Allowed with {len(obligations)} obligation(s)",
                obligations=obligations,
            )

        if all_decisions:
            return all_decisions[0]

        return PolicyDecision(
            verdict="deny", policy_cid="",
            justification="No matching policy found",
        )

    def _evaluate_single(self, method: str, actor: str,
                         resource: Optional[str], policy: PolicyObject,
                         now: float) -> PolicyDecision:
        """Evaluate one policy against the intent."""
        has_permission = False
        obligations: List[Dict[str, Any]] = []
        denial_reasons: List[str] = []

        for clause in policy.clauses:
            if not clause.matches(actor, method, resource, now):
                continue

            if clause.clause_type == "prohibition":
                denial_reasons.append(
                    f"Prohibited by policy '{policy.name}': {actor} cannot {method}"
                )
            elif clause.clause_type == "permission":
                has_permission = True
            elif clause.clause_type == "obligation":
                obligations.append({
                    "action": clause.action,
                    "deadline": clause.obligation_deadline,
                    "metadata": clause.metadata,
                })

        if denial_reasons:
            return PolicyDecision(
                verdict="deny", policy_cid=policy.cid,
                justification="; ".join(denial_reasons),
            )
        elif has_permission and obligations:
            return PolicyDecision(
                verdict="allow_with_obligations", policy_cid=policy.cid,
                justification=f"Permitted with {len(obligations)} obligation(s)",
                obligations=obligations,
            )
        elif has_permission:
            return PolicyDecision(
                verdict="allow", policy_cid=policy.cid,
                justification=f"Explicit permission for {actor} to {method}",
            )
        else:
            return PolicyDecision(
                verdict="deny", policy_cid=policy.cid,
                justification=f"No matching permission in policy '{policy.name}' for {actor}/{method}",
            )


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_permission_policy(
    name: str,
    actor: str,
    actions: List[str],
    valid_hours: float = 24.0,
    resource: Optional[str] = None,
) -> PolicyObject:
    """Create a simple time-bounded permission policy.

    Args:
        name: Human-readable policy name
        actor: DID or identifier of the permitted actor
        actions: List of method/tool names to permit
        valid_hours: How long the policy is valid (from now)
        resource: Optional resource scope
    """
    now = time.time()
    clauses = [
        PolicyClause(
            clause_type="permission",
            actor=actor,
            action=action,
            resource=resource,
            valid_from=now,
            valid_until=now + (valid_hours * 3600),
        )
        for action in actions
    ]
    return PolicyObject(name=name, clauses=clauses)


def make_prohibition_policy(
    name: str,
    actor: str = "*",
    actions: List[str] = None,
    resource: Optional[str] = None,
) -> PolicyObject:
    """Create a prohibition policy blocking specific actions."""
    clauses = [
        PolicyClause(
            clause_type="prohibition",
            actor=actor,
            action=action or "*",
            resource=resource,
        )
        for action in (actions or ["*"])
    ]
    return PolicyObject(name=name, clauses=clauses)


# ---------------------------------------------------------------------------
# Global singleton (thread-safe)
# ---------------------------------------------------------------------------

_EVALUATOR: Optional[PolicyEvaluator] = None
_POLICY_LOCK = threading.Lock()


def get_policy_evaluator() -> PolicyEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        with _POLICY_LOCK:
            if _EVALUATOR is None:
                _EVALUATOR = PolicyEvaluator()
    return _EVALUATOR
