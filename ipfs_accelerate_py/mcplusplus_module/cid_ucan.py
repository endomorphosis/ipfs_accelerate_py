"""CID-native execution artifacts and UCAN delegation for ipfs_accelerate_py.

Implements MCP++ Profile B (CID-Native Execution) and Profile C (UCAN Delegation)
for the accelerate server. Uses Trio-compatible patterns throughout.

Profile B provides:
- IntentObject: Declared intention with method, params, constraints
- DecisionObject: Authorization decision referencing an intent
- ReceiptObject: Execution result with CID provenance
- ExecutionEnvelope: Combines all three into a CID-addressable unit

Profile C provides:
- Delegation: UCAN capability delegation with resource/ability pairs
- DelegationEvaluator: Validates delegation chains with signature verification
- DIDKeyManager: Ed25519 key management for signing/verifying

Module: ipfs_accelerate_py.mcplusplus_module.cid_ucan
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.cid_ucan")


# ---------------------------------------------------------------------------
# CID utilities
# ---------------------------------------------------------------------------

def compute_cid(data: Any) -> str:
    """Compute a CID-like hash for any JSON-serializable data."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return f"bafy{digest[:52]}"


# ---------------------------------------------------------------------------
# Profile B: CID-Native Execution
# ---------------------------------------------------------------------------

@dataclass
class IntentObject:
    """Declared intention to invoke a method with specific parameters."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    requester: str = ""
    timestamp: float = field(default_factory=time.time)
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "intent",
                "method": self.method,
                "params": self.params,
                "constraints": self.constraints,
                "requester": self.requester,
                "timestamp": self.timestamp,
            })


@dataclass
class DecisionObject:
    """Authorization decision for an intent."""
    intent_cid: str
    authorized: bool
    reason: str = ""
    delegator: str = ""
    policy_cid: str = ""
    timestamp: float = field(default_factory=time.time)
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "decision",
                "intent_cid": self.intent_cid,
                "authorized": self.authorized,
                "reason": self.reason,
                "timestamp": self.timestamp,
            })


@dataclass
class ReceiptObject:
    """Execution receipt with result and provenance."""
    intent_cid: str
    decision_cid: str
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    executor: str = ""
    timestamp: float = field(default_factory=time.time)
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "receipt",
                "intent_cid": self.intent_cid,
                "decision_cid": self.decision_cid,
                "result": self.result,
                "error": self.error,
                "timestamp": self.timestamp,
            })

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ExecutionEnvelope:
    """Complete execution unit combining intent, decision, and receipt."""
    intent: IntentObject
    decision: DecisionObject
    receipt: ReceiptObject
    cid: str = ""

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "envelope",
                "intent_cid": self.intent.cid,
                "decision_cid": self.decision.cid,
                "receipt_cid": self.receipt.cid,
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "intent": {"cid": self.intent.cid, "method": self.intent.method, "params": self.intent.params},
            "decision": {"cid": self.decision.cid, "authorized": self.decision.authorized, "reason": self.decision.reason},
            "receipt": {"cid": self.receipt.cid, "result": self.receipt.result, "error": self.receipt.error, "duration_ms": self.receipt.duration_ms},
        }


# ---------------------------------------------------------------------------
# Profile C: UCAN Delegation
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    """A resource/ability pair."""
    resource: str
    ability: str

    def covers(self, resource: str, ability: str) -> bool:
        """Check if this capability covers the requested resource/ability."""
        # Wildcard matching
        res_match = self.resource == "*" or self.resource == resource or resource.startswith(self.resource.rstrip("*"))
        abil_match = self.ability == "*" or self.ability == ability
        return res_match and abil_match


@dataclass
class Delegation:
    """UCAN delegation token."""
    issuer: str
    audience: str
    capabilities: List[Capability] = field(default_factory=list)
    expiry: float = 0.0  # 0 = no expiry
    not_before: float = 0.0
    proof_cids: List[str] = field(default_factory=list)
    cid: str = ""
    signature: Optional[str] = None

    def __post_init__(self):
        if not self.cid:
            self.cid = compute_cid({
                "type": "delegation",
                "issuer": self.issuer,
                "audience": self.audience,
                "capabilities": [{"resource": c.resource, "ability": c.ability} for c in self.capabilities],
                "expiry": self.expiry,
                "not_before": self.not_before,
            })

    def is_expired(self, now: Optional[float] = None) -> bool:
        t = now or time.time()
        if self.expiry > 0 and t > self.expiry:
            return True
        if self.not_before > 0 and t < self.not_before:
            return True
        return False

    def has_capability(self, resource: str, ability: str) -> bool:
        return any(cap.covers(resource, ability) for cap in self.capabilities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "issuer": self.issuer,
            "audience": self.audience,
            "capabilities": [{"resource": c.resource, "ability": c.ability} for c in self.capabilities],
            "expiry": self.expiry,
            "not_before": self.not_before,
            "proof_cids": self.proof_cids,
            "signed": self.signature is not None,
        }


class DelegationEvaluator:
    """Evaluates UCAN delegation chains with signature verification."""

    MAX_CHAIN_DEPTH = 10  # Prevent DDoS via deep delegation chains

    def __init__(self):
        self._store: Dict[str, Delegation] = {}
        self._revoked: set = set()

    def add(self, delegation: Delegation) -> None:
        self._store[delegation.cid] = delegation

    def revoke(self, cid: str) -> None:
        self._revoked.add(cid)

    def build_chain(self, leaf_cid: str) -> List[Delegation]:
        """Build the delegation chain from leaf to root.
        
        Raises ValueError if chain exceeds MAX_CHAIN_DEPTH.
        """
        chain = []
        visited = set()
        current_cid = leaf_cid

        while current_cid and current_cid not in visited:
            if len(chain) >= self.MAX_CHAIN_DEPTH:
                raise ValueError(
                    f"Delegation chain exceeds maximum depth ({self.MAX_CHAIN_DEPTH})"
                )
            visited.add(current_cid)
            delegation = self._store.get(current_cid)
            if not delegation:
                break
            chain.append(delegation)
            # Follow proof chain (first proof CID)
            if delegation.proof_cids:
                current_cid = delegation.proof_cids[0]
            else:
                break

        return list(reversed(chain))  # root-first order

    def can_invoke(self, leaf_cid: str, resource: str, ability: str,
                   actor: Optional[str] = None, now: Optional[float] = None) -> Tuple[bool, str]:
        """Check if the delegation chain authorizes resource/ability for actor."""
        if leaf_cid in self._revoked:
            return False, f"Delegation '{leaf_cid}' has been revoked"

        if leaf_cid not in self._store:
            return False, f"Unknown delegation: {leaf_cid}"

        chain = self.build_chain(leaf_cid)
        if not chain:
            return False, "Empty delegation chain"

        # Actor check
        leaf = chain[-1]
        if actor is not None and leaf.audience != actor:
            return False, f"Actor '{actor}' does not match leaf audience '{leaf.audience}'"

        # Expiry check
        t = now or time.time()
        for d in chain:
            if d.is_expired(now=t):
                return False, f"Delegation '{d.cid}' has expired"

        # Revocation check
        for d in chain:
            if d.cid in self._revoked:
                return False, f"Delegation '{d.cid}' in chain has been revoked"

        # Signature verification (when signatures present)
        for d in chain:
            if d.signature is not None:
                if not self._verify_signature(d):
                    return False, f"Delegation '{d.cid}' has invalid signature"

        # Capability check
        for d in chain:
            if d.has_capability(resource, ability):
                return True, "authorized"

        return False, f"No delegation in chain grants '{ability}' on '{resource}'"

    def _verify_signature(self, delegation: Delegation) -> bool:
        """Verify delegation signature using Ed25519.
        
        Fail-closed: returns False if crypto libs aren't available and a
        signature is present (delegations without signatures skip verification).
        """
        if not delegation.signature:
            return True  # No signature present — unsigned delegation (open access)

        if not delegation.issuer.startswith("did:key:"):
            # Non-DID issuer with a signature — we can't verify it, so deny
            logger.warning("Cannot verify signature for non-DID issuer: %s", delegation.issuer)
            return False

        try:
            from nacl.signing import VerifyKey
            from nacl.encoding import HexEncoder

            # Extract public key from DID (multicodec ed25519-pub prefix: 0xed01)
            key_material = delegation.issuer.replace("did:key:z", "")
            verify_key = VerifyKey(key_material.encode(), encoder=HexEncoder)

            # Verify signature over canonical payload
            payload = json.dumps({
                "issuer": delegation.issuer,
                "audience": delegation.audience,
                "capabilities": [{"resource": c.resource, "ability": c.ability} for c in delegation.capabilities],
            }, sort_keys=True).encode()

            verify_key.verify(payload, bytes.fromhex(delegation.signature))
            return True
        except ImportError:
            # Fail-closed: deny signed delegations when we can't verify
            logger.warning("PyNaCl not available — cannot verify signed delegation, denying")
            return False
        except Exception as e:
            logger.warning(f"Signature verification failed for {delegation.cid}: {e}")
            return False


# ---------------------------------------------------------------------------
# Event DAG (Profile B provenance tracking)
# ---------------------------------------------------------------------------

@dataclass
class DAGEvent:
    """A node in the append-only provenance DAG."""
    cid: str
    event_type: str  # "intent" | "decision" | "receipt" | "envelope"
    parent_cids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventDAG:
    """Append-only provenance graph for MCP++ execution tracking."""

    def __init__(self):
        self._events: Dict[str, DAGEvent] = {}
        self._children: Dict[str, List[str]] = {}  # parent -> children
        self._lock = threading.Lock()

    def append(self, event: DAGEvent) -> str:
        """Add an event to the DAG. Returns its CID."""
        with self._lock:
            self._events[event.cid] = event
            for parent in event.parent_cids:
                self._children.setdefault(parent, []).append(event.cid)
        return event.cid

    def frontier(self) -> List[DAGEvent]:
        """Return leaf events (events with no children)."""
        with self._lock:
            all_parents = set()
            for event in self._events.values():
                all_parents.update(event.parent_cids)
            return [e for e in self._events.values() if e.cid not in self._children]

    def history(self, limit: int = 50) -> List[DAGEvent]:
        """Return recent events, newest first."""
        with self._lock:
            events = sorted(self._events.values(), key=lambda e: e.timestamp, reverse=True)
            return events[:limit]

    def provenance(self, cid: str) -> List[DAGEvent]:
        """Trace provenance chain from a CID back to roots."""
        with self._lock:
            chain = []
            visited = set()
            queue = [cid]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                event = self._events.get(current)
                if event:
                    chain.append(event)
                    queue.extend(event.parent_cids)
            return chain

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_events": len(self._events),
                "frontier_size": len([e for e in self._events.values() if e.cid not in self._children]),
                "events": {cid: {"type": e.event_type, "parents": e.parent_cids, "timestamp": e.timestamp}
                           for cid, e in self._events.items()},
            }


# ---------------------------------------------------------------------------
# Global singletons (thread-safe)
# ---------------------------------------------------------------------------

_EVALUATOR: Optional[DelegationEvaluator] = None
_EVENT_DAG: Optional[EventDAG] = None
_SINGLETON_LOCK = threading.Lock()


def get_evaluator() -> DelegationEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        with _SINGLETON_LOCK:
            if _EVALUATOR is None:
                _EVALUATOR = DelegationEvaluator()
    return _EVALUATOR


def get_event_dag() -> EventDAG:
    global _EVENT_DAG
    if _EVENT_DAG is None:
        with _SINGLETON_LOCK:
            if _EVENT_DAG is None:
                _EVENT_DAG = EventDAG()
    return _EVENT_DAG


# ---------------------------------------------------------------------------
# High-level execution helper
# ---------------------------------------------------------------------------

async def execute_with_envelope(
    method: str,
    params: Dict[str, Any],
    requester: str = "",
    delegation_cid: Optional[str] = None,
    executor_fn=None,
) -> ExecutionEnvelope:
    """Execute a method call with full CID-native envelope tracking.
    
    Creates Intent → Decision → Receipt → Envelope, recording everything
    in the Event DAG for provenance.
    
    Args:
        method: Method name (e.g. "run_model", "infer")
        params: Method parameters
        requester: DID or identifier of the requester
        delegation_cid: Optional UCAN delegation CID for authorization
        executor_fn: Async callable that performs the actual execution
        
    Returns:
        ExecutionEnvelope with full provenance chain
    """
    dag = get_event_dag()

    # Create Intent
    intent = IntentObject(method=method, params=params, requester=requester)
    dag.append(DAGEvent(cid=intent.cid, event_type="intent", payload={"method": method}))

    # Create Decision
    if delegation_cid:
        evaluator = get_evaluator()
        authorized, reason = evaluator.can_invoke(
            delegation_cid, f"mcp://tool/{method}", "invoke", actor=requester
        )
    else:
        # No delegation required (open access)
        authorized, reason = True, "open_access"

    decision = DecisionObject(
        intent_cid=intent.cid,
        authorized=authorized,
        reason=reason,
        policy_cid=delegation_cid or "",
    )
    dag.append(DAGEvent(
        cid=decision.cid, event_type="decision",
        parent_cids=[intent.cid], payload={"authorized": authorized},
    ))

    # Execute (if authorized)
    start_time = time.time()
    result = None
    error = None

    if authorized and executor_fn:
        try:
            result = await executor_fn(method, params)
        except Exception as e:
            error = str(e)
    elif not authorized:
        error = f"Unauthorized: {reason}"

    duration_ms = (time.time() - start_time) * 1000

    # Create Receipt
    receipt = ReceiptObject(
        intent_cid=intent.cid,
        decision_cid=decision.cid,
        result=result,
        error=error,
        duration_ms=duration_ms,
        executor="ipfs_accelerate_py",
    )
    dag.append(DAGEvent(
        cid=receipt.cid, event_type="receipt",
        parent_cids=[decision.cid], payload={"success": receipt.success},
    ))

    # Create Envelope
    envelope = ExecutionEnvelope(intent=intent, decision=decision, receipt=receipt)
    dag.append(DAGEvent(
        cid=envelope.cid, event_type="envelope",
        parent_cids=[intent.cid, decision.cid, receipt.cid],
    ))

    return envelope
