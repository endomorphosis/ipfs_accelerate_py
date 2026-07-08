"""Risk scoring and frontier scheduling helpers for MCP++ Profile layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RiskRecord:
    """Per-actor risk bookkeeping."""

    actor: str
    total_invocations: int = 0
    denied_count: int = 0
    obligation_count: int = 0
    disputed_count: int = 0
    last_event_cid: str = ""

    @property
    def denial_ratio(self) -> float:
        if self.total_invocations <= 0:
            return 0.0
        return float(self.denied_count) / float(self.total_invocations)

    @property
    def score(self) -> float:
        # Weighted deterministic risk formula (0..1 clipped).
        base = (self.denial_ratio * 0.60) + (min(10, self.obligation_count) * 0.03) + (min(10, self.disputed_count) * 0.07)
        return max(0.0, min(1.0, base))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "total_invocations": self.total_invocations,
            "denied_count": self.denied_count,
            "obligation_count": self.obligation_count,
            "disputed_count": self.disputed_count,
            "denial_ratio": round(self.denial_ratio, 5),
            "score": round(self.score, 5),
            "last_event_cid": self.last_event_cid,
        }


@dataclass(order=True)
class FrontierItem:
    """Prioritization frontier item for risk-adjusted scheduling."""

    priority: float
    sequence: int
    event_cid: str = field(compare=False)
    actor: str = field(compare=False)
    expected_value: float = field(compare=False, default=0.5)
    dependency_ready: bool = field(compare=False, default=True)
    retry_count: int = field(compare=False, default=0)
    consensus_signal: Dict[str, Any] = field(compare=False, default_factory=dict)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class RiskScheduler:
    """Tracks risk records and computes risk-adjusted frontier priority."""

    def __init__(self) -> None:
        self._records: Dict[str, RiskRecord] = {}
        self._frontier: List[FrontierItem] = []
        self._sequence: int = 0

    def _get_record(self, actor: str) -> RiskRecord:
        key = actor or "*"
        record = self._records.get(key)
        if record is None:
            record = RiskRecord(actor=key)
            self._records[key] = record
        return record

    def record_outcome(
        self,
        *,
        actor: str,
        allowed: bool,
        obligations: int = 0,
        event_cid: str = "",
        disputed: bool = False,
    ) -> RiskRecord:
        """Update actor risk state from one invocation outcome."""
        record = self._get_record(actor)
        record.total_invocations += 1
        if not allowed:
            record.denied_count += 1
        record.obligation_count += max(0, int(obligations))
        if disputed:
            record.disputed_count += 1
        if event_cid:
            record.last_event_cid = event_cid
        return record

    def get_actor_risk(self, actor: str) -> Dict[str, Any]:
        """Return risk view for one actor."""
        return self._get_record(actor).to_dict()

    def list_risks(self) -> List[Dict[str, Any]]:
        """Return deterministic list of all actor risk records."""
        return [self._records[key].to_dict() for key in sorted(self._records.keys())]

    def enqueue_frontier(
        self,
        *,
        event_cid: str,
        actor: str,
        expected_value: float,
        dependency_ready: bool,
        retry_count: int = 0,
        consensus_signal: Optional[Dict[str, Any]] = None,
        enable_consensus_signal: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FrontierItem:
        """Push event onto risk-adjusted scheduling frontier.

        Lower numeric priority means sooner scheduling.
        """
        record = self._get_record(actor)
        risk_penalty = record.score
        readiness_penalty = 0.0 if dependency_ready else 1.0
        value_bonus = max(0.0, min(1.0, expected_value))
        retry_penalty = min(0.50, max(0, int(retry_count)) * 0.05)

        consensus_adjustment = 0.0
        signal = dict(consensus_signal or {})
        if enable_consensus_signal and signal:
            confidence_raw = signal.get("confidence", 0.0)
            try:
                confidence = max(0.0, min(1.0, float(confidence_raw)))
            except (TypeError, ValueError):
                confidence = 0.0
            # Higher consensus confidence should slightly prioritize execution.
            consensus_adjustment -= 0.20 * confidence
            # Disputed consensus should be non-breaking and simply penalize priority.
            if bool(signal.get("disputed", False)):
                consensus_adjustment += 0.20

        priority = risk_penalty + readiness_penalty + (1.0 - value_bonus) + retry_penalty + consensus_adjustment

        self._sequence += 1

        item = FrontierItem(
            priority=priority,
            sequence=self._sequence,
            event_cid=event_cid,
            actor=actor,
            expected_value=value_bonus,
            dependency_ready=dependency_ready,
            retry_count=max(0, int(retry_count)),
            consensus_signal=signal,
            metadata=dict(metadata or {}),
        )
        heapq.heappush(self._frontier, item)
        return item

    def pop_next(self) -> FrontierItem | None:
        """Pop next event from frontier by computed priority."""
        if not self._frontier:
            return None
        return heapq.heappop(self._frontier)

    def frontier_size(self) -> int:
        """Return frontier queue size."""
        return len(self._frontier)

    def stats(self) -> Dict[str, Any]:
        """Return aggregate scheduler/risk metrics."""
        avg_risk = 0.0
        if self._records:
            avg_risk = sum(record.score for record in self._records.values()) / len(self._records)
        return {
            "actor_count": len(self._records),
            "frontier_size": self.frontier_size(),
            "average_risk": round(avg_risk, 5),
        }


__all__ = [
    "FrontierItem",
    "RiskRecord",
    "RiskScheduler",
]
