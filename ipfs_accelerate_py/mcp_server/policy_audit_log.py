"""Deterministic in-memory policy audit log for unified MCP runtime.

This module provides a lightweight audit trail for authorization/policy outcomes
recorded during unified `tools_dispatch` execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import threading
import time
from typing import Any, Dict, List, Optional


@dataclass
class PolicyAuditEntry:
    """Single audit entry for an authorization/policy decision."""

    timestamp: float
    decision: str
    tool: str
    actor: str
    intent_cid: str = ""
    policy_cid: str = ""
    justification: str = ""
    obligations: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class PolicyAuditLog:
    """Thread-safe in-memory audit log with bounded retention."""

    def __init__(self, *, enabled: bool = False, max_entries: int = 10_000) -> None:
        self._enabled = bool(enabled)
        self._max_entries = int(max_entries)
        self._entries: List[PolicyAuditEntry] = []
        self._lock = threading.Lock()
        self._total_recorded = 0
        self._counts: Dict[str, int] = {
            "allow": 0,
            "deny": 0,
            "allow_with_obligations": 0,
            "authorization_denied": 0,
            "policy_denied": 0,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def record(
        self,
        *,
        decision: str,
        tool: str,
        actor: str,
        intent_cid: str = "",
        policy_cid: str = "",
        justification: str = "",
        obligations: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[PolicyAuditEntry]:
        """Record a decision entry, returning it when enabled."""
        if not self._enabled:
            return None

        entry = PolicyAuditEntry(
            timestamp=float(time.time() if timestamp is None else timestamp),
            decision=str(decision or "").strip() or "unknown",
            tool=str(tool or ""),
            actor=str(actor or ""),
            intent_cid=str(intent_cid or ""),
            policy_cid=str(policy_cid or ""),
            justification=str(justification or ""),
            obligations=[str(x) for x in (obligations or [])],
            extra=dict(extra or {}),
        )

        with self._lock:
            if self._max_entries > 0 and len(self._entries) >= self._max_entries:
                self._entries.pop(0)
            self._entries.append(entry)
            self._total_recorded += 1
            self._counts[entry.decision] = self._counts.get(entry.decision, 0) + 1

        return entry

    def recent(self, n: int = 20) -> List[PolicyAuditEntry]:
        with self._lock:
            return list(self._entries[-max(0, int(n)):])

    def clear(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            total = int(self._total_recorded)
            in_memory = len(self._entries)
        allow = int(counts.get("allow", 0) + counts.get("allow_with_obligations", 0))
        deny = int(counts.get("deny", 0) + counts.get("authorization_denied", 0) + counts.get("policy_denied", 0))
        return {
            "enabled": self._enabled,
            "total_recorded": total,
            "in_memory": in_memory,
            "allow_count": allow,
            "deny_count": deny,
            "by_decision": counts,
            "max_entries": self._max_entries,
        }


__all__ = [
    "PolicyAuditEntry",
    "PolicyAuditLog",
]
