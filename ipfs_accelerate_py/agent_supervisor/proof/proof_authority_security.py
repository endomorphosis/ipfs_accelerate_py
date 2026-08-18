"""Fail-closed Q2 proof-authority gate.

Proposal, model, tactician, hammer, and prompt roles produce candidates.
They never become proof authority.  A timeout is observational and is not
a falsehood label.  Forged receipts, prompt-selected authority, and
self-attestation fail closed.

This module is an authority boundary, not a prover.  An admit decision
only means the claim is eligible for an independent checker; it does not
satisfy a proof obligation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .formal_verification_contracts import content_identity


PROOF_AUTHORITY_SECURITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-authority-security@1"
)
PROOF_AUTHORITY_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-authority-security-request@1"
)
PROOF_AUTHORITY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-authority-security-receipt@1"
)

PROPOSAL_ROLES: Final[frozenset[str]] = frozenset(
    {
        "candidate",
        "evaluator",
        "hammer",
        "llm",
        "model",
        "model_draft",
        "prompt",
        "proposal",
        "provider",
        "self",
        "tactician",
    }
)
INDEPENDENT_CHECKER_ROLES: Final[frozenset[str]] = frozenset(
    {
        "independent_checker",
        "kernel",
        "kernel_checker",
        "native_reconstruction",
    }
)
AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "has_proof_authority",
        "proof_authority",
        "proof_authority_granted",
        "raises_proof_authority",
    }
)
PROMPT_INJECTION_MARKERS: Final[tuple[str, ...]] = (
    "ignore previous",
    "ignore all previous",
    "disregard policy",
    "override policy",
    "you are now",
    "system prompt",
    "hidden_label",
    "hidden labels",
    "grant proof authority",
    "self-promote",
    "trust_remote_code",
    "skip checker",
    "skip the checker",
    "treat timeout as false",
    "treat timeout as falsehood",
)
REMOTE_CODE_MARKERS: Final[tuple[str, ...]] = (
    "__reduce__",
    "eval(",
    "exec(",
    "os.system",
    "pickle.loads",
    "subprocess.",
    "trust_remote_code",
)


class ProofAuthorityError(ValueError):
    """Malformed proof-authority security request."""


class ProofAuthorityDecision(str, Enum):
    ADMIT = "admit"
    REJECT = "reject"


class ProofAuthorityReason(str, Enum):
    PROPOSAL_PROOF_AUTHORITY = "proposal_proof_authority"
    FORGED_PROOF = "forged_proof"
    POLICY_PROMPT_INJECTION = "policy_prompt_injection"
    TIMEOUT_AS_FALSEHOOD = "timeout_as_falsehood"
    MISSING_INDEPENDENT_CHECKER = "missing_independent_checker"
    MODEL_SELF_ATTESTATION = "model_self_attestation"
    REMOTE_CODE_PAYLOAD = "remote_code_payload"
    HIDDEN_LABEL_PROOF = "hidden_label_proof"


CLOSED_PROOF_AUTHORITY_REJECTIONS: Final[tuple[str, ...]] = tuple(
    item.value for item in ProofAuthorityReason
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ProofAuthorityError(f"{name} must be a string")
    else:
        text = value.strip()
    if "\x00" in text:
        raise ProofAuthorityError(f"{name} must not contain NUL")
    if required and not text:
        raise ProofAuthorityError(f"{name} must be a non-empty string")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProofAuthorityError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProofAuthorityError(f"{name} must be a mapping")
    return value


def _role(value: Any) -> str:
    return _text(value, "producer_role").casefold().replace("-", "_")


def _walk_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk_strings(item))
        return tuple(found)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        found = []
        for item in value:
            found.extend(_walk_strings(item))
        return tuple(found)
    return ()


def _contains_marker(value: Any, markers: Sequence[str]) -> bool:
    haystacks = tuple(item.casefold() for item in _walk_strings(value))
    return any(
        marker in haystack for haystack in haystacks for marker in markers
    )


_INJECTION_RE = re.compile(
    "|".join(re.escape(marker) for marker in PROMPT_INJECTION_MARKERS),
    re.IGNORECASE,
)


def detect_policy_prompt_injection(value: Any) -> bool:
    """Return True when policy or prompt text tries to rewrite authority."""

    return any(_INJECTION_RE.search(item) for item in _walk_strings(value))


def claims_proof_authority(value: Any) -> bool:
    """Return True when a nested payload asserts proof authority."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name).strip().casefold().replace("-", "_")
            if name in AUTHORITY_CLAIM_KEYS and item not in (False, None, 0, ""):
                return True
            if claims_proof_authority(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(claims_proof_authority(item) for item in value)
    return False


@dataclass(frozen=True)
class ProofAuthorityRequest:
    """One candidate proof claim.  Not a checker verdict."""

    claim_id: str
    producer_role: str
    proof_identity: str = ""
    checker_identity: str = ""
    checker_role: str = ""
    independently_checked: bool = False
    timeout_occurred: bool = False
    timeout_labeled_falsehood: bool = False
    hidden_labels_used: bool = False
    claimed_verified: bool = False
    payload: Mapping[str, Any] = MappingProxyType({})
    policy_text: str = ""
    schema: str = PROOF_AUTHORITY_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "producer_role", _role(self.producer_role))
        object.__setattr__(
            self,
            "proof_identity",
            _text(self.proof_identity, "proof_identity", required=False),
        )
        object.__setattr__(
            self,
            "checker_identity",
            _text(self.checker_identity, "checker_identity", required=False),
        )
        object.__setattr__(
            self,
            "checker_role",
            _text(self.checker_role, "checker_role", required=False).casefold().replace("-", "_"),
        )
        object.__setattr__(
            self,
            "independently_checked",
            _bool(self.independently_checked, "independently_checked"),
        )
        object.__setattr__(
            self, "timeout_occurred", _bool(self.timeout_occurred, "timeout_occurred")
        )
        object.__setattr__(
            self,
            "timeout_labeled_falsehood",
            _bool(self.timeout_labeled_falsehood, "timeout_labeled_falsehood"),
        )
        object.__setattr__(
            self, "hidden_labels_used", _bool(self.hidden_labels_used, "hidden_labels_used")
        )
        object.__setattr__(
            self, "claimed_verified", _bool(self.claimed_verified, "claimed_verified")
        )
        payload = _mapping(self.payload, "payload")
        object.__setattr__(
            self,
            "payload",
            MappingProxyType({str(key): payload[key] for key in payload}),
        )
        object.__setattr__(
            self, "policy_text", _text(self.policy_text, "policy_text", required=False)
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROOF_AUTHORITY_REQUEST_SCHEMA:
            raise ProofAuthorityError("unsupported proof-authority request schema")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofAuthorityRequest":
        if not isinstance(payload, Mapping):
            raise ProofAuthorityError("proof-authority request must be a mapping")
        return cls(
            claim_id=payload.get("claim_id", ""),
            producer_role=payload.get("producer_role", payload.get("role", "")),
            proof_identity=payload.get("proof_identity", ""),
            checker_identity=payload.get("checker_identity", ""),
            checker_role=payload.get("checker_role", ""),
            independently_checked=bool(payload.get("independently_checked", False)),
            timeout_occurred=bool(payload.get("timeout_occurred", False)),
            timeout_labeled_falsehood=bool(
                payload.get("timeout_labeled_falsehood", False)
            ),
            hidden_labels_used=bool(payload.get("hidden_labels_used", False)),
            claimed_verified=bool(payload.get("claimed_verified", False)),
            payload=payload.get("payload", {}),
            policy_text=str(payload.get("policy_text", "") or ""),
            schema=str(payload.get("schema", PROOF_AUTHORITY_REQUEST_SCHEMA) or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checker_identity": self.checker_identity,
            "checker_role": self.checker_role,
            "claim_id": self.claim_id,
            "claimed_verified": self.claimed_verified,
            "hidden_labels_used": self.hidden_labels_used,
            "independently_checked": self.independently_checked,
            "payload": dict(self.payload),
            "policy_text": self.policy_text,
            "producer_role": self.producer_role,
            "proof_identity": self.proof_identity,
            "schema": self.schema,
            "timeout_labeled_falsehood": self.timeout_labeled_falsehood,
            "timeout_occurred": self.timeout_occurred,
        }


@dataclass(frozen=True)
class ProofAuthorityReceipt:
    """Tamper-evident Q2 decision.  Not proof satisfaction."""

    decision: ProofAuthorityDecision
    claim_id: str
    producer_role: str
    reasons: tuple[str, ...]
    independently_checked: bool
    schema: str = PROOF_AUTHORITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        decision = (
            self.decision
            if isinstance(self.decision, ProofAuthorityDecision)
            else ProofAuthorityDecision(str(self.decision))
        )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "producer_role", _role(self.producer_role))
        object.__setattr__(
            self,
            "reasons",
            tuple(_text(item, "reason") for item in self.reasons),
        )
        object.__setattr__(
            self,
            "independently_checked",
            _bool(self.independently_checked, "independently_checked"),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROOF_AUTHORITY_RECEIPT_SCHEMA:
            raise ProofAuthorityError("unsupported proof-authority receipt schema")
        unknown = sorted(set(self.reasons) - set(CLOSED_PROOF_AUTHORITY_REJECTIONS))
        if unknown:
            raise ProofAuthorityError(
                "unknown proof-authority reasons: " + ", ".join(unknown)
            )
        if self.decision is ProofAuthorityDecision.ADMIT and self.reasons:
            raise ProofAuthorityError("admitted proof-authority receipts cannot carry reasons")
        if self.decision is ProofAuthorityDecision.REJECT and not self.reasons:
            raise ProofAuthorityError("rejected proof-authority receipts require reasons")

    @property
    def admitted(self) -> bool:
        return self.decision is ProofAuthorityDecision.ADMIT

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "decision": self.decision.value,
            "independently_checked": self.independently_checked,
            "producer_role": self.producer_role,
            "reasons": list(self.reasons),
            "schema": self.schema,
        }


def evaluate_proof_authority(
    request: ProofAuthorityRequest | Mapping[str, Any],
) -> ProofAuthorityReceipt:
    """Admit only independently checked, non-forged, non-injected claims."""

    selected = (
        request
        if isinstance(request, ProofAuthorityRequest)
        else ProofAuthorityRequest.from_dict(request)
    )
    reasons: list[str] = []
    if claims_proof_authority(selected.payload):
        reasons.append(ProofAuthorityReason.PROPOSAL_PROOF_AUTHORITY.value)
    if (
        selected.producer_role in PROPOSAL_ROLES
        and selected.claimed_verified
        and not selected.independently_checked
    ):
        reasons.append(ProofAuthorityReason.PROPOSAL_PROOF_AUTHORITY.value)
    if selected.claimed_verified and not selected.independently_checked:
        reasons.append(ProofAuthorityReason.FORGED_PROOF.value)
    if selected.claimed_verified and not selected.proof_identity:
        reasons.append(ProofAuthorityReason.FORGED_PROOF.value)
    if selected.independently_checked:
        if not selected.checker_identity:
            reasons.append(ProofAuthorityReason.MISSING_INDEPENDENT_CHECKER.value)
        if selected.checker_role not in INDEPENDENT_CHECKER_ROLES:
            reasons.append(ProofAuthorityReason.MISSING_INDEPENDENT_CHECKER.value)
        if (
            selected.producer_role == selected.checker_role
            or selected.checker_role in PROPOSAL_ROLES
        ):
            reasons.append(ProofAuthorityReason.MODEL_SELF_ATTESTATION.value)
    if selected.timeout_labeled_falsehood or (
        selected.timeout_occurred and selected.claimed_verified
    ):
        reasons.append(ProofAuthorityReason.TIMEOUT_AS_FALSEHOOD.value)
    if selected.hidden_labels_used:
        reasons.append(ProofAuthorityReason.HIDDEN_LABEL_PROOF.value)
    if detect_policy_prompt_injection(selected.policy_text) or detect_policy_prompt_injection(
        selected.payload
    ):
        reasons.append(ProofAuthorityReason.POLICY_PROMPT_INJECTION.value)
    if _contains_marker(selected.payload, REMOTE_CODE_MARKERS):
        reasons.append(ProofAuthorityReason.REMOTE_CODE_PAYLOAD.value)
    unique = tuple(dict.fromkeys(reasons))
    return ProofAuthorityReceipt(
        decision=(
            ProofAuthorityDecision.REJECT
            if unique
            else ProofAuthorityDecision.ADMIT
        ),
        claim_id=selected.claim_id,
        producer_role=selected.producer_role,
        reasons=unique,
        independently_checked=selected.independently_checked,
    )


__all__ = (
    "CLOSED_PROOF_AUTHORITY_REJECTIONS",
    "INDEPENDENT_CHECKER_ROLES",
    "PROOF_AUTHORITY_RECEIPT_SCHEMA",
    "PROOF_AUTHORITY_REQUEST_SCHEMA",
    "PROOF_AUTHORITY_SECURITY_SCHEMA",
    "PROPOSAL_ROLES",
    "ProofAuthorityDecision",
    "ProofAuthorityError",
    "ProofAuthorityReason",
    "ProofAuthorityReceipt",
    "ProofAuthorityRequest",
    "claims_proof_authority",
    "detect_policy_prompt_injection",
    "evaluate_proof_authority",
)
