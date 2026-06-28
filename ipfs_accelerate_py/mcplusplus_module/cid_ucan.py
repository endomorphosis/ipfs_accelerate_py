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

    def save_revocations(self, path: str) -> None:
        """Persist revocation list to disk."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"revoked": list(self._revoked)}, f)

    def load_revocations(self, path: str) -> int:
        """Load revocation list from disk. Returns count loaded."""
        import os
        if not os.path.isfile(path):
            return 0
        with open(path, "r") as f:
            data = json.load(f)
        cids = data.get("revoked", [])
        self._revoked.update(cids)
        return len(cids)

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
    """Append-only provenance graph for MCP++ execution tracking.

    Supports tiered storage with ZK proof compaction:
    - Hot tier: Recent events in memory (up to HOT_TIER_MAX)
    - Cold tier: Older epochs persisted to disk
    - Compacted tier: ZK proofs summarizing cold epochs

    When hot tier exceeds threshold, oldest epoch is compacted via
    Merkle tree + simulated Groth16 proof, moved to cold storage,
    and removed from memory. Provenance queries transparently load
    cold events when traversal crosses epoch boundaries.
    """

    MAX_EVENTS = 10000  # Hard cap (prevents unbounded growth even without compaction)

    def __init__(self, storage_dir: str = ""):
        self._events: Dict[str, DAGEvent] = {}
        self._children: Dict[str, List[str]] = {}  # parent -> children
        self._lock = threading.Lock()
        self._compactor: Optional[Any] = None
        self._storage_dir = storage_dir

    def _get_compactor(self):
        """Lazy-init the DAG compactor."""
        if self._compactor is None:
            try:
                from .dag_compaction import DAGCompactor, COLD_TIER_DIR
                storage = self._storage_dir or COLD_TIER_DIR
                self._compactor = DAGCompactor(storage_dir=storage)
            except ImportError:
                logger.warning("dag_compaction module not available; compaction disabled")
        return self._compactor

    def append(self, event: DAGEvent) -> str:
        """Add an event to the DAG. Returns its CID.

        Triggers ZK compaction when hot tier exceeds threshold.
        """
        with self._lock:
            # Hard cap eviction (fallback if compaction is disabled)
            if len(self._events) >= self.MAX_EVENTS:
                oldest = sorted(self._events.values(), key=lambda e: e.timestamp)[:100]
                for old in oldest:
                    del self._events[old.cid]
                    self._children.pop(old.cid, None)
            self._events[event.cid] = event
            for parent in event.parent_cids:
                self._children.setdefault(parent, []).append(event.cid)

        # Check if ZK compaction should run (outside lock to avoid blocking)
        self._maybe_compact()
        return event.cid

    def _maybe_compact(self) -> None:
        """Trigger epoch compaction if hot tier is too large."""
        compactor = self._get_compactor()
        if compactor is None:
            return
        if not compactor.should_compact(len(self._events)):
            return

        with self._lock:
            # Serialize events for compactor
            events_dict = {
                cid: {
                    "cid": cid,
                    "event_type": e.event_type,
                    "parent_cids": e.parent_cids,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                }
                for cid, e in self._events.items()
            }
            children_dict = dict(self._children)

        # Compact outside the lock (disk I/O)
        result = compactor.compact_epoch(events_dict, children_dict)
        if result:
            # Remove compacted events from hot tier
            with self._lock:
                for cid in result.compacted_cids:
                    self._events.pop(cid, None)
                    self._children.pop(cid, None)
                # Also clean up children references to removed parents
                for parent_cid in list(self._children.keys()):
                    self._children[parent_cid] = [
                        c for c in self._children[parent_cid]
                        if c not in set(result.compacted_cids)
                    ]
                    if not self._children[parent_cid]:
                        del self._children[parent_cid]

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
        """Trace provenance chain from a CID back to roots.

        Transparently loads cold-tier events when traversal crosses
        epoch boundaries (parent CID not in hot tier).
        """
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
                else:
                    # Try loading from cold tier
                    cold_event = self._load_cold_event(current)
                    if cold_event:
                        chain.append(cold_event)
                        queue.extend(cold_event.parent_cids)
            return chain

    def _load_cold_event(self, cid: str) -> Optional[DAGEvent]:
        """Load a single event from cold storage by CID."""
        compactor = self._get_compactor()
        if compactor is None:
            return None

        epoch_id = compactor.find_epoch_for_cid(cid)
        if epoch_id is None:
            return None

        events = compactor.load_cold_epoch(epoch_id)
        for e in events:
            if e.get("cid") == cid:
                return DAGEvent(
                    cid=cid,
                    event_type=e.get("event_type", "unknown"),
                    parent_cids=e.get("parent_cids", []),
                    payload=e.get("payload", {}),
                    timestamp=e.get("timestamp", 0),
                )
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG state including compaction summary."""
        compactor = self._get_compactor()
        with self._lock:
            result = {
                "hot_events": len(self._events),
                "frontier_size": len([e for e in self._events.values() if e.cid not in self._children]),
                "events": {cid: {"type": e.event_type, "parents": e.parent_cids, "timestamp": e.timestamp}
                           for cid, e in self._events.items()},
            }
        if compactor:
            result["compaction"] = compactor.summary()
            result["total_events"] = len(self._events) + compactor.total_compacted_events
        else:
            result["total_events"] = len(self._events)
        return result


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
    timeout_ms: int = 30000,
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
        timeout_ms: Maximum execution time in milliseconds (default: 30s)
        
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

    # Execute (if authorized) with timeout
    start_time = time.time()
    result = None
    error = None

    if authorized and executor_fn:
        try:
            # Use trio timeout if available in current context
            try:
                import trio
                with trio.move_on_after(timeout_ms / 1000.0) as cancel_scope:
                    result = await executor_fn(method, params)
                if cancel_scope.cancelled_caught:
                    error = f"Execution timeout after {timeout_ms}ms"
            except (RuntimeError, AttributeError):
                # Not in a trio context — execute without timeout
                result = await executor_fn(method, params)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
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
