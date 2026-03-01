"""Event DAG primitives for MCP++ provenance and partial ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class EventNode:
    """Immutable-style event node keyed by event CID."""

    event_cid: str
    payload: Dict[str, Any]
    parents: List[str] = field(default_factory=list)


class EventDAGStore:
    """In-memory event DAG with parent integrity checks and lineage traversal."""

    def __init__(self) -> None:
        self._events: Dict[str, EventNode] = {}
        self._children: Dict[str, Set[str]] = {}

    def has_event(self, event_cid: str) -> bool:
        """Return True if event exists."""
        return event_cid in self._events

    def add_event(self, event_cid: str, payload: Dict[str, Any]) -> EventNode:
        """Insert event after validating parent references.

        Raises:
            ValueError: if any declared parent is missing.
        """
        parents = list(payload.get("parents", []) or [])
        for parent in parents:
            if parent not in self._events:
                raise ValueError(f"missing_parent:{parent}")

        node = EventNode(event_cid=event_cid, payload=dict(payload), parents=parents)
        self._events[event_cid] = node
        for parent in parents:
            self._children.setdefault(parent, set()).add(event_cid)
        return node

    def get_event(self, event_cid: str) -> Dict[str, Any] | None:
        """Return event payload by CID, if present."""
        node = self._events.get(event_cid)
        if node is None:
            return None
        out = dict(node.payload)
        out.setdefault("parents", list(node.parents))
        out["event_cid"] = event_cid
        return out

    def get_lineage(self, event_cid: str) -> List[str]:
        """Return root-to-leaf lineage for one event CID.

        For branching histories, returns one deterministic path using lexical
        parent order at each step.
        """
        if event_cid not in self._events:
            return []

        lineage = [event_cid]
        current = event_cid
        while True:
            node = self._events[current]
            if not node.parents:
                break
            parent = sorted(node.parents)[0]
            lineage.append(parent)
            current = parent
        lineage.reverse()
        return lineage

    def list_roots(self) -> List[str]:
        """List root event CIDs (events with no parents)."""
        roots = [cid for cid, node in self._events.items() if not node.parents]
        return sorted(roots)

    def stats(self) -> Dict[str, Any]:
        """Return basic DAG metrics for diagnostics."""
        return {
            "event_count": len(self._events),
            "root_count": len(self.list_roots()),
            "edge_count": sum(len(node.parents) for node in self._events.values()),
        }

    def export_snapshot(self) -> Dict[str, Any]:
        """Export a deterministic JSON-serializable snapshot of the DAG."""
        events: List[Dict[str, Any]] = []
        for event_cid in sorted(self._events.keys()):
            node = self._events[event_cid]
            payload = dict(node.payload)
            payload.setdefault("parents", list(node.parents))
            events.append(
                {
                    "event_cid": event_cid,
                    "payload": payload,
                }
            )

        return {
            "version": 1,
            "events": events,
            "stats": self.stats(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "EventDAGStore":
        """Rebuild a store from a deterministic snapshot export."""
        store = cls()
        items = (snapshot or {}).get("events", [])
        if not isinstance(items, list):
            return store

        pending: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            event_cid = str(item.get("event_cid") or "").strip()
            payload = item.get("payload")
            if not event_cid or not isinstance(payload, dict):
                continue
            pending.append({"event_cid": event_cid, "payload": dict(payload)})

        # Deterministically add events after their parents using simple fixed-point passes.
        remaining = list(sorted(pending, key=lambda x: x["event_cid"]))
        progress = True
        while remaining and progress:
            progress = False
            next_round: List[Dict[str, Any]] = []
            for item in remaining:
                event_cid = item["event_cid"]
                payload = item["payload"]
                parents = list(payload.get("parents", []) or [])
                if any(parent not in store._events for parent in parents):
                    next_round.append(item)
                    continue
                store.add_event(event_cid, payload)
                progress = True
            remaining = next_round

        if remaining:
            unresolved = ",".join(x["event_cid"] for x in remaining)
            raise ValueError(f"snapshot_unresolved_parents:{unresolved}")

        return store

    def replay_from_root(self, root_event_cid: str) -> List[str]:
        """Return deterministic replay order from one root through descendants."""
        root = str(root_event_cid or "").strip()
        if root not in self._events:
            return []

        order: List[str] = []
        queue: List[str] = [root]
        seen: Set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            children = sorted(self._children.get(current, set()))
            queue.extend(children)
        return order

    def rollback_path(self, event_cid: str) -> List[str]:
        """Return deterministic rollback order from leaf to root lineage."""
        lineage = self.get_lineage(event_cid)
        lineage.reverse()
        return lineage


__all__ = [
    "EventDAGStore",
    "EventNode",
]
