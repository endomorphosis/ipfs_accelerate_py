"""Instruction and trust-domain separation for untrusted handoff context (EAAEF-120).

Source comments, model outputs, imported history, and attachments cannot change
policy, authority, secrets, proof keys, tests, Quack ownership, or promotion
criteria.  Untrusted text is classified and stripped of imperative policy
overrides before it may enter a ContextPack.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


UNTRUSTED_CONTEXT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/untrusted-context@1"
)
PROTECTED_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "policy",
        "secrets",
        "proof_keys",
        "tests",
        "quack_ownership",
        "promotion_criteria",
        "merge_admission",
    }
)
_OVERRIDE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bignore (all )?(previous|prior) (instructions|policy)\b", re.I),
    re.compile(r"\byou are now (the )?(root|owner|admin|quack owner)\b", re.I),
    re.compile(r"\b(set|change|override|grant|promote) (the )?(policy|authority|secret|proof key|quack owner)\b", re.I),
    re.compile(r"\bskip (the )?(tests|proofs|verification|approval)\b", re.I),
    re.compile(r"\bself[- ]approve\b", re.I),
    re.compile(r"\bexport (the )?(private key|secret|token)\b", re.I),
    re.compile(r"\bmark (this )?(task|merge) (completed|accepted|admitted)\b", re.I),
)


class TrustDomain(str, Enum):
    """Where a fragment of context originated."""

    OPERATOR_POLICY = "operator_policy"
    SOURCE_COMMENT = "source_comment"
    MODEL_OUTPUT = "model_output"
    IMPORTED_HISTORY = "imported_history"
    ATTACHMENT = "attachment"
    TOOL_OUTPUT = "tool_output"


class UntrustedContextError(ValueError):
    """Untrusted context attempted a protected-domain override."""


class TrustDomainError(UntrustedContextError):
    """A trust domain is not allowed to issue the requested instruction."""


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def contains_policy_override(text: str) -> bool:
    blob = _normalize(text)
    return any(pattern.search(blob) for pattern in _OVERRIDE_PATTERNS)


def classify_fragment(text: str, *, domain: TrustDomain) -> Mapping[str, Any]:
    """Return a public classification; never execute the fragment."""

    override = contains_policy_override(text)
    return MappingProxyType(
        {
            "schema": UNTRUSTED_CONTEXT_SCHEMA,
            "domain": domain.value,
            "trusted": domain is TrustDomain.OPERATOR_POLICY,
            "policy_override_detected": override,
            "admitted_to_protected_domains": (),
        }
    )


@dataclass(frozen=True)
class ContextAdmission:
    """Decision for one untrusted fragment against protected domains."""

    domain: TrustDomain
    admitted: bool
    stripped_text: str
    rejected_reason: str = ""
    blocked_domains: tuple[str, ...] = ()

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": UNTRUSTED_CONTEXT_SCHEMA,
                "domain": self.domain.value,
                "admitted": self.admitted,
                "stripped_text": self.stripped_text,
                "rejected_reason": self.rejected_reason,
                "blocked_domains": list(self.blocked_domains),
            }
        )


def admit_untrusted_text(
    text: str,
    *,
    domain: TrustDomain,
    requested_domains: Sequence[str] = (),
) -> ContextAdmission:
    """Admit untrusted text only as data, never as policy.

    Operator policy may address protected domains.  Every other domain is
    stripped of override language and cannot enlarge authority.
    """

    requested = tuple(str(item) for item in requested_domains if str(item))
    unknown = [item for item in requested if item not in PROTECTED_DOMAINS]
    if unknown:
        raise UntrustedContextError(f"unknown protected domain: {unknown[0]}")
    if domain is TrustDomain.OPERATOR_POLICY:
        return ContextAdmission(
            domain=domain,
            admitted=True,
            stripped_text=str(text),
            blocked_domains=(),
        )
    if requested:
        raise TrustDomainError(
            f"{domain.value} cannot change protected domains {list(requested)}"
        )
    stripped = str(text)
    reason = ""
    if contains_policy_override(text):
        stripped = ""
        reason = "policy_override_stripped"
    return ContextAdmission(
        domain=domain,
        admitted=True,
        stripped_text=stripped,
        rejected_reason=reason,
        blocked_domains=tuple(PROTECTED_DOMAINS),
    )


def assert_no_authority_enlargement(
    fragments: Sequence[Mapping[str, Any] | ContextAdmission],
) -> None:
    """Fail closed if any non-operator fragment claimed a protected domain."""

    for fragment in fragments:
        payload = fragment.to_dict() if isinstance(fragment, ContextAdmission) else fragment
        domain = str(payload.get("domain") or "")
        if domain == TrustDomain.OPERATOR_POLICY.value:
            continue
        admitted_domains = payload.get("admitted_to_protected_domains") or ()
        if admitted_domains:
            raise TrustDomainError(
                "untrusted context cannot admit protected domains"
            )
        if payload.get("trusted") is True:
            raise TrustDomainError("untrusted context cannot be marked trusted")
