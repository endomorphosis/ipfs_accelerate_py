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

import logging

from .cid_ucan import compute_cid

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.temporal_policy")


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

    def save_policies(self, path: str) -> int:
        """Persist all registered policies to disk (atomic write)."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {"policies": {cid: p.to_dict() for cid, p in self._policies.items()}}
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return len(self._policies)

    def load_policies(self, path: str) -> int:
        """Load policies from disk. Returns count loaded."""
        import os
        if not os.path.isfile(path):
            return 0
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for cid, p_dict in data.get("policies", {}).items():
                clauses = [
                    PolicyClause(
                        clause_type=c.get("clause_type", "permission"),
                        actor=c.get("actor", "*"),
                        action=c.get("action", "*"),
                        resource=c.get("resource"),
                        valid_from=c.get("valid_from"),
                        valid_until=c.get("valid_until"),
                        obligation_deadline=c.get("obligation_deadline"),
                        metadata=c.get("metadata", {}),
                    )
                    for c in p_dict.get("clauses", [])
                ]
                policy = PolicyObject(
                    name=p_dict.get("name", ""),
                    clauses=clauses,
                    description=p_dict.get("description", ""),
                )
                self._policies[policy.cid] = policy
            return len(data.get("policies", {}))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load policies from %s: %s", path, e)
            return 0

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
# Obligation Tracker
# ---------------------------------------------------------------------------

import logging

_obligation_logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.obligations")


@dataclass
class TrackedObligation:
    """An obligation that must be fulfilled, with tracking metadata."""
    obligation_id: str
    action: str
    deadline: float  # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    fulfilled: bool = False
    fulfilled_at: Optional[float] = None
    execution_cid: str = ""  # CID of the execution that created this obligation
    created_at: float = field(default_factory=time.time)

    @property
    def is_overdue(self) -> bool:
        """Check if this obligation has passed its deadline without fulfillment."""
        return not self.fulfilled and self.deadline > 0 and time.time() > self.deadline


class ObligationTracker:
    """Tracks and enforces obligations from temporal deontic policies.

    Obligations are actions that MUST be performed (with optional deadlines).
    This tracker records pending obligations, allows marking them as fulfilled,
    and can report overdue obligations for alerting/enforcement.
    """

    MAX_OBLIGATIONS = 10000  # Prevent unbounded memory growth

    def __init__(self):
        self._obligations: Dict[str, TrackedObligation] = {}
        self._lock = threading.Lock()

    def record(self, obligations: List[Dict[str, Any]], execution_cid: str = "") -> List[str]:
        """Record obligations from a policy decision.

        Args:
            obligations: List of obligation dicts from PolicyDecision
            execution_cid: CID of the execution envelope that triggered these

        Returns:
            List of obligation IDs created.
        """
        ids = []
        with self._lock:
            # Evict stale fulfilled obligations if approaching capacity
            if len(self._obligations) >= self.MAX_OBLIGATIONS:
                self._evict_fulfilled()

            for ob in obligations:
                ob_id = compute_cid({
                    "type": "obligation",
                    "action": ob.get("action", ""),
                    "deadline": ob.get("deadline", 0),
                    "execution_cid": execution_cid,
                    "created_at": time.time(),
                })
                tracked = TrackedObligation(
                    obligation_id=ob_id,
                    action=ob.get("action", ""),
                    deadline=ob.get("deadline", 0),
                    metadata=ob.get("metadata", {}),
                    execution_cid=execution_cid,
                )
                self._obligations[ob_id] = tracked
                ids.append(ob_id)
                _obligation_logger.info(
                    "Obligation recorded: id=%s action=%s deadline=%s",
                    ob_id[:16], tracked.action, tracked.deadline or "none",
                )
        return ids

    def _evict_fulfilled(self) -> int:
        """Remove fulfilled obligations older than 1 hour. Must hold self._lock."""
        now = time.time()
        to_delete = [
            ob_id for ob_id, ob in self._obligations.items()
            if ob.fulfilled and (now - (ob.fulfilled_at or 0) > 3600)
        ]
        if not to_delete:
            # If still at capacity, evict oldest fulfilled regardless of age
            fulfilled = [(ob_id, ob.fulfilled_at or 0) for ob_id, ob in self._obligations.items() if ob.fulfilled]
            fulfilled.sort(key=lambda x: x[1])
            to_delete = [ob_id for ob_id, _ in fulfilled[:len(fulfilled) // 2]]
        for ob_id in to_delete:
            del self._obligations[ob_id]
        if to_delete:
            _obligation_logger.info("Evicted %d fulfilled obligations (capacity management)", len(to_delete))
        return len(to_delete)

    def fulfill(self, obligation_id: str) -> bool:
        """Mark an obligation as fulfilled.

        Returns True if the obligation was found and marked, False otherwise.
        """
        with self._lock:
            ob = self._obligations.get(obligation_id)
            if ob and not ob.fulfilled:
                ob.fulfilled = True
                ob.fulfilled_at = time.time()
                _obligation_logger.info("Obligation fulfilled: id=%s action=%s", obligation_id[:16], ob.action)
                return True
        return False

    def get_overdue(self) -> List[TrackedObligation]:
        """Get all obligations that have passed their deadline without fulfillment."""
        with self._lock:
            return [ob for ob in self._obligations.values() if ob.is_overdue]

    def get_pending(self) -> List[TrackedObligation]:
        """Get all unfulfilled obligations."""
        with self._lock:
            return [ob for ob in self._obligations.values() if not ob.fulfilled]

    def summary(self) -> Dict[str, int]:
        """Get a summary count of obligations by status."""
        with self._lock:
            total = len(self._obligations)
            fulfilled = sum(1 for ob in self._obligations.values() if ob.fulfilled)
            overdue = sum(1 for ob in self._obligations.values() if ob.is_overdue)
            pending = total - fulfilled
            return {"total": total, "fulfilled": fulfilled, "pending": pending, "overdue": overdue}


# Global obligation tracker singleton
_OBLIGATION_TRACKER: Optional[ObligationTracker] = None


def get_obligation_tracker() -> ObligationTracker:
    global _OBLIGATION_TRACKER
    if _OBLIGATION_TRACKER is None:
        with _POLICY_LOCK:
            if _OBLIGATION_TRACKER is None:
                _OBLIGATION_TRACKER = ObligationTracker()
    return _OBLIGATION_TRACKER


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
                # Auto-load persisted policies
                import os
                policy_path = os.path.join(
                    os.environ.get("MCPPP_STORAGE_DIR",
                                   os.path.expanduser("~/.ipfs_accelerate/state")),
                    "policies.json",
                )
                loaded = _EVALUATOR.load_policies(policy_path)
                if loaded:
                    logger.info("Loaded %d policies from %s", loaded, policy_path)
    return _EVALUATOR
