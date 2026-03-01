"""CID-native execution artifact helpers for MCP++ Profile B.

The module provides deterministic artifact object builders and content-addressed
identifiers for intent/decision/receipt/event nodes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, Iterable, Optional


def canonicalize_artifact(payload: Dict[str, Any]) -> bytes:
    """Return deterministic bytes for artifact content."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_artifact_cid(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 content address for an artifact."""
    digest = hashlib.sha256(canonicalize_artifact(payload)).hexdigest()
    return f"cidv1-sha256-{digest}"


def build_intent(
    *,
    interface_cid: str,
    tool: str,
    input_cid: str,
    expected_output_schema_cid: str = "",
    constraints_policy_cid: str = "",
    correlation_id: str = "",
    declared_side_effects: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build canonical intent payload."""
    return {
        "interface_cid": interface_cid,
        "tool": tool,
        "input_cid": input_cid,
        "expected_output_schema_cid": expected_output_schema_cid,
        "constraints_policy_cid": constraints_policy_cid,
        "correlation_id": correlation_id,
        "declared_side_effects": list(declared_side_effects or []),
    }


def build_decision(
    *,
    decision: str,
    intent_cid: str,
    policy_cid: str = "",
    proofs_checked: Optional[Iterable[str]] = None,
    evaluation_witness_cid: str = "",
    justification: str = "",
    obligations: Optional[Iterable[Dict[str, Any]]] = None,
    policy_version: str = "",
    evaluator_dids: Optional[Iterable[str]] = None,
    signatures: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build canonical decision payload."""
    return {
        "decision": decision,
        "intent_cid": intent_cid,
        "policy_cid": policy_cid,
        "proofs_checked": list(proofs_checked or []),
        "evaluation_witness_cid": evaluation_witness_cid,
        "justification": justification,
        "obligations": list(obligations or []),
        "policy_version": policy_version,
        "evaluator_dids": list(evaluator_dids or []),
        "signatures": list(signatures or []),
    }


def build_receipt(
    *,
    intent_cid: str,
    output_cid: str,
    decision_cid: str,
    observed_side_effects: Optional[Iterable[str]] = None,
    proofs_checked: Optional[Iterable[str]] = None,
    correlation_id: str = "",
    time_observed: str = "",
    signatures: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build canonical receipt payload."""
    return {
        "intent_cid": intent_cid,
        "output_cid": output_cid,
        "observed_side_effects": list(observed_side_effects or []),
        "proofs_checked": list(proofs_checked or []),
        "decision_cid": decision_cid,
        "correlation_id": correlation_id,
        "time_observed": time_observed,
        "signatures": list(signatures or []),
    }


def build_event(
    *,
    parents: Optional[Iterable[str]] = None,
    interface_cid: str,
    intent_cid: str,
    proof_cid: str,
    decision_cid: str,
    output_cid: str,
    receipt_cid: str,
    peer_did: str = "",
    timestamps: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build canonical event node payload."""
    return {
        "parents": list(parents or []),
        "interface_cid": interface_cid,
        "intent_cid": intent_cid,
        "proof_cid": proof_cid,
        "decision_cid": decision_cid,
        "output_cid": output_cid,
        "receipt_cid": receipt_cid,
        "peer_did": peer_did,
        "timestamps": dict(timestamps or {}),
    }


def envelope_from_payloads(
    *,
    interface_cid: str,
    input_payload: Dict[str, Any],
    tool: str,
    output_payload: Dict[str, Any],
    decision: str = "allow",
    decision_justification: str = "",
    decision_obligations: Optional[Iterable[Dict[str, Any]]] = None,
    proof_cid: str = "",
    policy_cid: str = "",
    correlation_id: str = "",
    parent_event_cids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a full immutable artifact envelope and return all payloads/CIDs."""
    input_cid = compute_artifact_cid(input_payload)
    output_cid = compute_artifact_cid(output_payload)

    intent = build_intent(
        interface_cid=interface_cid,
        tool=tool,
        input_cid=input_cid,
        correlation_id=correlation_id,
        constraints_policy_cid=policy_cid,
    )
    intent_cid = compute_artifact_cid(intent)

    decision_payload = build_decision(
        decision=decision,
        intent_cid=intent_cid,
        policy_cid=policy_cid,
        proofs_checked=[proof_cid] if proof_cid else [],
        justification=str(decision_justification or ""),
        obligations=list(decision_obligations or []),
    )
    decision_cid = compute_artifact_cid(decision_payload)

    receipt = build_receipt(
        intent_cid=intent_cid,
        output_cid=output_cid,
        decision_cid=decision_cid,
        proofs_checked=[proof_cid] if proof_cid else [],
        correlation_id=correlation_id,
    )
    receipt_cid = compute_artifact_cid(receipt)

    event = build_event(
        parents=parent_event_cids,
        interface_cid=interface_cid,
        intent_cid=intent_cid,
        proof_cid=proof_cid,
        decision_cid=decision_cid,
        output_cid=output_cid,
        receipt_cid=receipt_cid,
    )
    event_cid = compute_artifact_cid(event)

    return {
        "input": input_payload,
        "input_cid": input_cid,
        "intent": intent,
        "intent_cid": intent_cid,
        "decision": decision_payload,
        "decision_cid": decision_cid,
        "output": output_payload,
        "output_cid": output_cid,
        "receipt": receipt,
        "receipt_cid": receipt_cid,
        "event": event,
        "event_cid": event_cid,
    }


class ArtifactStore:
    """Thread-safe in-memory artifact store keyed by CID.

    The store is intentionally simple for deterministic testability while the
    broader persistence backend strategy is finalized.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_cid: dict[str, dict[str, Any]] = {}

    def put(self, cid: str, payload: dict[str, Any]) -> None:
        """Persist one artifact payload by content-addressed identifier."""
        key = str(cid or "").strip()
        if not key:
            return
        with self._lock:
            self._by_cid[key] = dict(payload)

    def put_many(self, records: dict[str, dict[str, Any]]) -> int:
        """Persist a mapping of `cid -> payload` and return number written."""
        count = 0
        with self._lock:
            for cid, payload in (records or {}).items():
                key = str(cid or "").strip()
                if not key or not isinstance(payload, dict):
                    continue
                self._by_cid[key] = dict(payload)
                count += 1
        return count

    def get(self, cid: str) -> Optional[dict[str, Any]]:
        """Return a stored artifact payload by CID, if present."""
        key = str(cid or "").strip()
        with self._lock:
            payload = self._by_cid.get(key)
            return dict(payload) if isinstance(payload, dict) else None

    def stats(self) -> dict[str, int]:
        """Return deterministic store statistics."""
        with self._lock:
            return {"artifact_count": int(len(self._by_cid))}


__all__ = [
    "ArtifactStore",
    "build_decision",
    "build_event",
    "build_intent",
    "build_receipt",
    "canonicalize_artifact",
    "compute_artifact_cid",
    "envelope_from_payloads",
]
