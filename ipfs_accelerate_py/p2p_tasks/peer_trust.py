"""Peer trust-level resolution for TaskQueue P2P access control.

Trust tiers control which tasks a remote peer may claim from the local queue.

Levels (highest to lowest):
- TRUSTED:  peer provided a matching shared token or a valid UCAN delegation.
- ELEVATED: peer DID appears in the local event DAG as having performed work.
- BASELINE: peer is whitelisted (allowed) but no trust evidence was found.

Environment:
- IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS / IPFS_DATASETS_PY_TASK_P2P_TRUST_TIERS
    Set to "1"/"true" to enable tiered access (default: 0 = binary allow/deny).
- IPFS_ACCELERATE_PY_TASK_P2P_BASELINE_MAX_PRIORITY / IPFS_DATASETS_PY_TASK_P2P_BASELINE_MAX_PRIORITY
    Maximum task priority (1-10) that BASELINE peers may claim (default: 5).
    ELEVATED and TRUSTED peers are never subject to this cap.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional


class PeerTrustLevel(str, Enum):
    """Trust level assigned to a remote peer for queue access decisions."""

    TRUSTED = "trusted"
    ELEVATED = "elevated"
    BASELINE = "baseline"


def trust_tiers_enabled() -> bool:
    """Return True when tiered trust access control is active."""
    for name in (
        "IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS",
        "IPFS_DATASETS_PY_TASK_P2P_TRUST_TIERS",
    ):
        raw = os.environ.get(name, "").strip().lower()
        if raw:
            return raw in {"1", "true", "yes", "on"}
    return False


def baseline_max_claim_priority() -> int:
    """Return the maximum task priority a BASELINE peer may claim (1-10)."""
    for name in (
        "IPFS_ACCELERATE_PY_TASK_P2P_BASELINE_MAX_PRIORITY",
        "IPFS_DATASETS_PY_TASK_P2P_BASELINE_MAX_PRIORITY",
    ):
        raw = os.environ.get(name, "").strip()
        if raw:
            try:
                val = int(raw)
                return max(1, min(10, val))
            except ValueError:
                pass
    return 5


def _shared_token() -> Optional[str]:
    for name in (
        "IPFS_ACCELERATE_PY_TASK_P2P_TOKEN",
        "IPFS_DATASETS_PY_TASK_P2P_TOKEN",
    ):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def _token_matches(msg: Dict[str, Any]) -> bool:
    """Return True if the message carries a token matching the configured shared token."""
    expected = _shared_token()
    if not expected:
        return False
    provided = msg.get("token") or msg.get("ucan_token") or ""
    return isinstance(provided, str) and provided == expected


def _ucan_token_present(msg: Dict[str, Any]) -> bool:
    """Return True if the message carries any UCAN/JWT token field."""
    for key in ("ucan", "ucan_token", "jwt", "delegation_cid", "proof_cid"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _peer_did_in_event_dag(peer_did: str, event_dag: Any) -> bool:
    """Return True if ``peer_did`` appears as a worker in the event DAG."""
    if not peer_did or event_dag is None:
        return False

    try:
        # EventDAG wraps an EventDAGStore; use export_snapshot for full scan.
        store = getattr(event_dag, "_store", None)
        if store is not None:
            snapshot = store.export_snapshot()
        else:
            snapshot = event_dag.export_snapshot()  # type: ignore[attr-defined]
    except Exception:
        return False

    if not isinstance(snapshot, dict):
        return False

    for item in snapshot.get("events", []):
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item, dict) else item
        if not isinstance(payload, dict):
            payload = item
        did = str(payload.get("peer_did", "") or "").strip()
        if did and did == peer_did:
            return True
    return False


def resolve_peer_trust_level(
    msg: Dict[str, Any],
    *,
    event_dag: Any = None,
    peer_did: Optional[str] = None,
) -> PeerTrustLevel:
    """Derive a PeerTrustLevel from an inbound P2P request message.

    Args:
        msg: The inbound request dict (Protocol V1 or JSON-RPC envelope).
        event_dag: Optional EventDAG instance to check peer work history.
        peer_did: Optional explicit peer DID from transport metadata.
            Defaults to ``msg.get("peer_did")``.

    Returns:
        - ``TRUSTED`` if the message includes the correct shared token, or when
          no shared token is configured and a UCAN token field is present.
        - ``ELEVATED`` if the peer's DID appears in the local event DAG as
          having performed work.
        - ``BASELINE`` otherwise (peer is still whitelisted / allowed).
    """
    # 1. Shared-token match → TRUSTED unconditionally.
    if _token_matches(msg):
        return PeerTrustLevel.TRUSTED

    # 2. No shared token is configured: UCAN presence signals a cooperating peer.
    if not _shared_token() and _ucan_token_present(msg):
        return PeerTrustLevel.TRUSTED

    # 3. Check event DAG work history for ELEVATED trust.
    did = peer_did or str(msg.get("peer_did", "") or "").strip()
    if not did:
        # Fall back to worker_id / peer_id fields as DID hint.
        for field in ("worker_id", "peer_id", "peer", "actor"):
            val = msg.get(field)
            if isinstance(val, str) and val.strip():
                did = val.strip()
                break

    if did and _peer_did_in_event_dag(did, event_dag):
        return PeerTrustLevel.ELEVATED

    return PeerTrustLevel.BASELINE


__all__ = [
    "PeerTrustLevel",
    "baseline_max_claim_priority",
    "resolve_peer_trust_level",
    "trust_tiers_enabled",
]
