"""Temporal deontic policy evaluation for MCP++ Profile D.

The engine intentionally starts with a deterministic, dependency-light model
that supports permission/prohibition/obligation clauses with temporal bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class PolicyClause:
    """One temporal deontic clause."""

    clause_type: str  # permission | prohibition | obligation
    actor: str = "*"
    action: str = "*"
    resource: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    obligation_deadline: str | None = None
    metadata: Dict[str, Any] | None = None

    def applies(self, *, actor: str, action: str, resource: str | None, now: datetime) -> bool:
        if self.actor not in {"*", actor}:
            return False
        if self.action not in {"*", action}:
            return False
        if self.resource is not None and resource is not None and self.resource != resource:
            return False

        start = _parse_iso8601(self.valid_from)
        if start is not None and now < start:
            return False
        end = _parse_iso8601(self.valid_until)
        if end is not None and now > end:
            return False
        return True


@dataclass(frozen=True)
class PolicyDecision:
    """Evaluation output for one intent/action check."""

    decision: str  # allow | deny | allow_with_obligations
    justification: str
    obligations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "justification": self.justification,
            "obligations": [dict(x) for x in self.obligations],
        }


def parse_policy_clauses(raw_clauses: Iterable[Dict[str, Any]]) -> List[PolicyClause]:
    """Parse raw policy clause payloads into typed rules."""
    clauses: List[PolicyClause] = []
    for item in raw_clauses:
        if not isinstance(item, dict):
            continue
        clauses.append(
            PolicyClause(
                clause_type=str(item.get("clause_type", "") or "").strip().lower(),
                actor=str(item.get("actor", "*") or "*").strip() or "*",
                action=str(item.get("action", "*") or "*").strip() or "*",
                resource=(str(item.get("resource", "")).strip() or None) if item.get("resource") is not None else None,
                valid_from=str(item.get("valid_from", "")).strip() or None,
                valid_until=str(item.get("valid_until", "")).strip() or None,
                obligation_deadline=str(item.get("obligation_deadline", "")).strip() or None,
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return clauses


def evaluate_policy(
    *,
    clauses: Iterable[PolicyClause],
    actor: str,
    action: str,
    resource: str | None = None,
    now: datetime | None = None,
) -> PolicyDecision:
    """Evaluate temporal deontic rules for one action invocation."""
    eval_time = now or _utc_now()
    denial_reasons: List[str] = []
    has_permission = False
    obligations: List[Dict[str, Any]] = []

    for clause in clauses:
        if not clause.applies(actor=actor, action=action, resource=resource, now=eval_time):
            continue

        if clause.clause_type == "prohibition":
            denial_reasons.append(f"prohibition(actor={actor}, action={action})")
        elif clause.clause_type == "permission":
            has_permission = True
        elif clause.clause_type == "obligation":
            obligations.append(
                {
                    "type": "obligation",
                    "action": clause.action,
                    "deadline": clause.obligation_deadline or "",
                    "metadata": dict(clause.metadata or {}),
                }
            )

    if denial_reasons:
        return PolicyDecision(
            decision="deny",
            justification="; ".join(denial_reasons),
            obligations=[],
        )

    if has_permission and obligations:
        return PolicyDecision(
            decision="allow_with_obligations",
            justification=f"allowed with {len(obligations)} obligation(s)",
            obligations=obligations,
        )

    if has_permission:
        return PolicyDecision(
            decision="allow",
            justification="permission matched",
            obligations=[],
        )

    return PolicyDecision(
        decision="deny",
        justification="no matching permission",
        obligations=[],
    )


def evaluate_raw_policy(
    *,
    raw_clauses: Iterable[Dict[str, Any]],
    actor: str,
    action: str,
    resource: str | None = None,
    now: datetime | None = None,
) -> PolicyDecision:
    """Convenience evaluator for raw dict clause payloads."""
    return evaluate_policy(
        clauses=parse_policy_clauses(raw_clauses),
        actor=actor,
        action=action,
        resource=resource,
        now=now,
    )


__all__ = [
    "PolicyClause",
    "PolicyDecision",
    "evaluate_policy",
    "evaluate_raw_policy",
    "parse_policy_clauses",
]
