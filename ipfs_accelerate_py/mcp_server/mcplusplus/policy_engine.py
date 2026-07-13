"""Temporal deontic policy evaluation for MCP++ Profile D.

The engine intentionally starts with a deterministic, dependency-light model
that supports permission/prohibition/obligation clauses with temporal bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "fulfilled"}
    return False


def _obligation_is_fulfilled(metadata: Dict[str, Any], now: datetime) -> bool:
    fulfilled_at = _parse_iso8601(str(metadata.get("fulfilled_at", "") or ""))
    if fulfilled_at is not None and fulfilled_at <= now:
        return True
    if fulfilled_at is not None and fulfilled_at > now:
        return False
    return _coerce_bool(metadata.get("fulfilled", False))


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
        if not _matches_pattern(self.actor, actor):
            return False
        if not _matches_pattern(self.action, action):
            return False
        if self.resource is not None and (resource is None or not _matches_pattern(self.resource, resource)):
            return False

        start = _parse_iso8601(self.valid_from)
        if start is not None and now < start:
            return False
        end = _parse_iso8601(self.valid_until)
        if end is not None and now > end:
            return False
        return True


def _matches_pattern(pattern: str, value: str) -> bool:
    return pattern == "*" or pattern == value or (pattern.endswith("/*") and value.startswith(pattern[:-1]))


@dataclass(frozen=True)
class PolicyDecision:
    """Evaluation output for one intent/action check."""

    decision: str  # allow | deny | allow_with_obligations
    justification: str
    obligations: List[Dict[str, Any]]
    evidence: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "decision": self.decision,
            "justification": self.justification,
            "obligations": [dict(x) for x in self.obligations],
        }
        if self.evidence is not None:
            result["evidence"] = dict(self.evidence)
        return result


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
    fulfilled_obligations = 0

    for clause in clauses:
        if not clause.applies(actor=actor, action=action, resource=resource, now=eval_time):
            continue

        if clause.clause_type == "prohibition":
            denial_reasons.append(f"prohibition(actor={actor}, action={action})")
        elif clause.clause_type == "permission":
            has_permission = True
        elif clause.clause_type == "obligation":
            metadata = dict(clause.metadata or {})
            if _obligation_is_fulfilled(metadata, eval_time):
                fulfilled_obligations += 1
                continue

            deadline = clause.obligation_deadline or ""
            deadline_dt = _parse_iso8601(deadline)
            obligations.append(
                {
                    "type": "obligation",
                    "action": clause.action,
                    "deadline": deadline,
                    "status": "overdue" if deadline_dt is not None and eval_time > deadline_dt else "pending",
                    "metadata": metadata,
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
        if fulfilled_obligations:
            return PolicyDecision(
                decision="allow",
                justification=f"permission matched; {fulfilled_obligations} obligation(s) already fulfilled",
                obligations=[],
            )
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


def evaluate_with_ipfs_datasets_policy(
    *,
    raw_clauses: Iterable[Dict[str, Any]],
    actor: str,
    action: str,
    resource: str | None = None,
    policy_text: str | List[str] | None = None,
    request_zkp_certificate: bool = True,
) -> PolicyDecision:
    """Evaluate Profile D through the canonical ``ipfs_datasets_py`` export.

    The local evaluator remains a dependency-light fallback for installations
    that intentionally omit datasets.  The canonical path carries policy and
    decision CIDs, formal logic, and a ZKP-ready statement into accelerator
    receipts without claiming that the statement is already a proof.
    """
    try:
        evaluate_execution_policy = _load_datasets_profile_d_evaluator()

        result = evaluate_execution_policy(
            actor=actor,
            action=action,
            resource=resource,
            policy={"clauses": list(raw_clauses)} if policy_text is None else None,
            policy_text=policy_text,
            request_zkp_certificate=request_zkp_certificate,
        )
        return PolicyDecision(
            decision=str(result["decision"]),
            justification=str(result.get("justification") or ""),
            obligations=[dict(item) for item in result.get("obligations", []) if isinstance(item, dict)],
            evidence={
                key: result[key]
                for key in ("policy_cid", "decision_cid", "formal_logic", "formal_logic_cid", "zkp_certificate")
                if key in result
            },
        )
    except Exception:
        return evaluate_raw_policy(
            raw_clauses=raw_clauses,
            actor=actor,
            action=action,
            resource=resource,
        )


def evaluate_profile_d_execution_policy(
    *,
    actor: str,
    action: str,
    resource: str | None = None,
    policy: Dict[str, Any] | None = None,
    policy_text: str | List[str] | None = None,
    evaluated_at: str | None = None,
    intent_cid: str | None = None,
    request_zkp_certificate: bool = False,
) -> Dict[str, Any]:
    """Run the interoperable Profile D RPC contract through datasets.

    Network-facing endpoints cannot silently fall back to the older local
    evaluator because it drops formal-logic provenance and ZKP statements.
    Internal execution keeps the dependency-light fallback above; this public
    transport function fails closed when the canonical dependency rejects a
    request or is unavailable.
    """
    evaluate_execution_policy = _load_datasets_profile_d_evaluator()
    return evaluate_execution_policy(
        actor=actor,
        action=action,
        resource=resource,
        policy=policy,
        policy_text=policy_text,
        evaluated_at=evaluated_at,
        intent_cid=intent_cid,
        request_zkp_certificate=request_zkp_certificate,
    )


def _load_datasets_profile_d_evaluator():
    """Load the canonical dependency despite the legacy vendored namespace.

    This repository still contains ``external/ipfs_accelerate/ipfs_datasets_py``
    for compatibility.  It shadows the separately installed package in source
    checkouts, so prefer the external canonical package when the legacy copy
    lacks Profile D.  Normal installations take the first import path.
    """
    try:
        module = importlib.import_module("ipfs_datasets_py.logic.profile_d_policy")
        return module.evaluate_execution_policy
    except ModuleNotFoundError:
        canonical_root = Path(__file__).resolve().parents[4] / "ipfs_datasets"
        if not (canonical_root / "ipfs_datasets_py" / "logic" / "profile_d_policy.py").is_file():
            raise
        for name in list(sys.modules):
            if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
                del sys.modules[name]
        sys.path.insert(0, str(canonical_root))
        module = importlib.import_module("ipfs_datasets_py.logic.profile_d_policy")
        return module.evaluate_execution_policy


__all__ = [
    "PolicyClause",
    "PolicyDecision",
    "evaluate_policy",
    "evaluate_profile_d_execution_policy",
    "evaluate_with_ipfs_datasets_policy",
    "evaluate_raw_policy",
    "parse_policy_clauses",
]
