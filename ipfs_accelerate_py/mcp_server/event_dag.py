"""Legacy Event DAG surface adapted to canonical MCP++ event DAG store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .mcplusplus.event_dag import EventDAGStore


@dataclass
class EventNode:
    """Source-compatible event node used by EventDAG adapter."""

    parents: List[str] = field(default_factory=list)
    interface_cid: str = ""
    intent_cid: str = ""
    decision_cid: str = ""
    output_cid: str = ""
    receipt_cid: str = ""
    proof_cid: Optional[str] = None
    peer_did: Optional[str] = None
    timestamp_created: str = ""
    timestamp_observed: str = ""

    @property
    def event_cid(self) -> str:
        from .cid_artifacts import artifact_cid

        return artifact_cid(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parents": list(self.parents),
            "interface_cid": self.interface_cid,
            "intent_cid": self.intent_cid,
            "decision_cid": self.decision_cid,
            "output_cid": self.output_cid,
            "receipt_cid": self.receipt_cid,
            "proof_cid": self.proof_cid,
            "peer_did": self.peer_did,
            "timestamps": {
                "created": self.timestamp_created,
                "observed": self.timestamp_observed,
            },
        }


class EventDAG:
    """Source-compatible Event DAG facade backed by canonical EventDAGStore."""

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = bool(strict)
        self._store = EventDAGStore()

    def append(self, node: EventNode) -> str:
        if not self.strict:
            snapshot = self._store.export_snapshot()
            known = {e.get("event_cid") for e in snapshot.get("events", []) if isinstance(e, dict)}
            payload = node.to_dict()
            parents = [p for p in payload.get("parents", []) if p in known]
            payload["parents"] = parents
            event_cid = node.event_cid
            self._store.add_event(event_cid, payload)
            return event_cid

        event_cid = node.event_cid
        self._store.add_event(event_cid, node.to_dict())
        return event_cid

    def get(self, event_cid: str) -> Optional[EventNode]:
        payload = self._store.get_event(event_cid)
        if payload is None:
            return None
        return EventNode(
            parents=list(payload.get("parents", []) or []),
            interface_cid=str(payload.get("interface_cid", "") or ""),
            intent_cid=str(payload.get("intent_cid", "") or ""),
            decision_cid=str(payload.get("decision_cid", "") or ""),
            output_cid=str(payload.get("output_cid", "") or ""),
            receipt_cid=str(payload.get("receipt_cid", "") or ""),
            proof_cid=str(payload.get("proof_cid", "") or "") or None,
            peer_did=str(payload.get("peer_did", "") or "") or None,
            timestamp_created=str((payload.get("timestamps") or {}).get("created", "") or ""),
            timestamp_observed=str((payload.get("timestamps") or {}).get("observed", "") or ""),
        )

    def frontier(self) -> List[str]:
        snapshot = self._store.export_snapshot()
        events = snapshot.get("events", []) if isinstance(snapshot, dict) else []
        all_cids = [e.get("event_cid") for e in events if isinstance(e, dict) and e.get("event_cid")]
        parents = []
        for item in events:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                parents.extend(payload.get("parents", []) or [])
        parent_set = set(str(x) for x in parents)
        return [cid for cid in all_cids if cid not in parent_set]

    def walk(self, event_cid: str) -> List[str]:
        return self._store.rollback_path(event_cid)

    def descendants(self, event_cid: str) -> List[str]:
        roots = self._store.replay_from_root(event_cid)
        return roots[1:] if roots else []

    def rollback_to(self, event_cid: str) -> List[str]:
        return self.descendants(event_cid)


def build_linear_dag(nodes: Iterable[EventNode]) -> EventDAG:
    dag = EventDAG(strict=False)
    prev_cid = ""
    for node in nodes:
        if prev_cid and prev_cid not in node.parents:
            node.parents = [prev_cid] + list(node.parents)
        prev_cid = dag.append(node)
    return dag


__all__ = ["EventNode", "EventDAG", "build_linear_dag"]
