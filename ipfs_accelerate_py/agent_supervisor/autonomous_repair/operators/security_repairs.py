"""Authorization, effect, and policy repair operators (DCR-046).

Interface: ``SecurityRepairOperators@1``

Evidence: ``dcr/safety-repair@1``

Implements the structural preview/inverse bodies for
:attr:`OperatorKind.REPAIR_AUTHORIZATION_GUARD`.  The library restores
reviewed SecurityIR authority/effect linkage and fail-closed policy gates
under MCP++ Profiles C (UCAN delegation) and D (temporal deontic policy).

Normative rules (fail-closed)
-----------------------------
* Policy outage or a missing decision **denies**.
* Stale, revoked, or wrong-audience grants **fail**.
* No server-supplied authorization assertion is trusted; execution-time
  checks re-evaluate principal, audience, capability, revocation, obligations,
  temporal validity, and declared effects from reviewed SecurityIR.
* Operators may restore reviewed bindings but cannot invent authority, policy
  semantics, UCAN grants, or effect classifications — those **abstain** for
  review.
* Operators remain proposal-only: they never grant write, proof, or semantic
  authority and never mutate production trees.

Predicted symbols: :class:`AuthorizationBindingOperator`,
:class:`EffectAnnotationOperator`, :class:`PolicyGateOperator`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

SECURITY_REPAIR_OPERATORS_INTERFACE: Final[str] = "SecurityRepairOperators@1"
SECURITY_REPAIR_EVIDENCE: Final[str] = "dcr/safety-repair@1"
SECURITY_IR_INTERFACE: Final[str] = "SecurityIR"
SECURITY_REPAIR_VERSION: Final[int] = 1

# MCP++ profile bindings for Profiles C (UCAN) and D (policy).
# Identifiers stay within the closed token alphabet (no '+'/ '@' glyphs).
MCP_PROFILE_C: Final[str] = "mcpplusplus/profile-c-ucan/v1"
MCP_PROFILE_D: Final[str] = "mcpplusplus/profile-d-policy/v1"
MCP_PROFILES_C_D: Final[tuple[str, str]] = (MCP_PROFILE_C, MCP_PROFILE_D)

SECURITY_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-repair@1"
)
SECURITY_IR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-ir@1"
)
SECURITY_GRANT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-grant@1"
)
SECURITY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-authorization-binding@1"
)
SECURITY_EFFECT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-effect-annotation@1"
)
SECURITY_POLICY_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-policy-gate@1"
)
SECURITY_AUTHZ_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-authorization-request@1"
)
SECURITY_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-repair-request@1"
)
SECURITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-repair-receipt@1"
)
SECURITY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-decision@1"
)
SECURITY_OPERATOR_VECTORS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-operator-vectors@1"
)

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_COLLECTION: Final[int] = 256
MAX_REASON_CODES: Final[int] = 32
MAX_OBLIGATIONS: Final[int] = 64

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
)
_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:bafy|bagu|bafk|sha256:)[A-Za-z0-9:_-]{8,200}$"
)

# Fields that would smuggle authority, generation, or dynamic code.
_FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
        "handler_body",
        "private_key",
        "secret",
        "password",
    }
)

# Server-supplied claim keys that must never establish a permit.
_SERVER_ASSERTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authorized",
        "is_authorized",
        "authorization_assertion",
        "server_authorization",
        "server_asserted_permit",
        "trusted_decision",
        "pre_authorized",
        "bypass_auth",
        "admin_override",
        "force_allow",
        "already_checked",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SecurityRepairError(ContractValidationError):
    """Malformed security repair input or closed-boundary violation."""


class SecurityRepairAbstention(SecurityRepairError):
    """Operator cannot proceed without inventing authority or semantics."""


class RepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed outcomes for one security repair attempt."""

    PREVIEW_READY = "preview_ready"
    ALREADY_ALIGNED = "already_aligned"
    ABSTAIN = "abstain"
    REJECTED = "rejected"
    DENIED = "denied"


class OperatorRole(str, Enum):  # noqa: UP042
    """Closed operator roles implementing REPAIR_AUTHORIZATION_GUARD."""

    AUTHORIZATION_BINDING = "authorization_binding"
    EFFECT_ANNOTATION = "effect_annotation"
    POLICY_GATE = "policy_gate"


class SecurityVerdict(str, Enum):  # noqa: UP042
    """Closed security decision verdicts.  Unknown is not safe."""

    PERMIT = "permit"
    DENY = "deny"


class DenialReasonCode(str, Enum):  # noqa: UP042
    """Closed denial / failure reason codes for execution-time checks."""

    ALLOWED = "allowed"
    POLICY_OUTAGE = "policy_outage"
    MISSING_DECISION = "missing_decision"
    STALE_GRANT = "stale_grant"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REVOKED = "revoked"
    WRONG_AUDIENCE = "wrong_audience"
    SERVER_ASSERTION_UNTRUSTED = "server_assertion_untrusted"
    NO_APPLICABLE_GRANT = "no_applicable_grant"
    CAPABILITY_MISMATCH = "capability_mismatch"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    EFFECT_MISMATCH = "effect_mismatch"
    UNDECLARED_EFFECT = "undeclared_effect"
    OBLIGATION_UNMET = "obligation_unmet"
    MISSING_SECURITY_IR = "missing_security_ir"
    INVENTED_AUTHORITY = "invented_authority"
    INVENTED_POLICY = "invented_policy"
    INVENTED_GRANT = "invented_grant"
    INVENTED_EFFECT = "invented_effect"
    MALFORMED_REQUEST = "malformed_request"


class GrantStatus(str, Enum):  # noqa: UP042
    """Closed grant lifecycle statuses."""

    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"
    REVOKED = "revoked"
    WRONG_AUDIENCE = "wrong_audience"
    NOT_YET_VALID = "not_yet_valid"


class PolicyAvailability(str, Enum):  # noqa: UP042
    """Whether the policy evaluation surface is available."""

    AVAILABLE = "available"
    OUTAGE = "outage"
    MISSING = "missing"
    UNKNOWN = "unknown"


class EffectClass(str, Enum):  # noqa: UP042
    """Closed effect classifications that may be annotated, never invented."""

    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"
    DISPATCH = "dispatch"
    DELEGATE = "delegate"
    PROVE = "prove"
    MERGE = "merge"
    NETWORK = "network"
    SIDE_EFFECT = "side_effect"
    NO_EFFECT = "no_effect"


class AuthoritySource(str, Enum):  # noqa: UP042
    """Authority retained on SecurityIR and repair artifacts.

    Only reviewed / production SecurityIR may authorize a repair preview.
    Invented, prose-inferred, or server-asserted sources abstain.
    """

    REVIEWED = "reviewed"
    PRODUCTION = "production"
    FIXTURE = "fixture"
    SERVER_ASSERTED = "server_asserted"
    PROSE_INFERRED = "prose_inferred"
    INVENTED = "invented"
    MISSING = "missing"

    @property
    def authorizes_security_source(self) -> bool:
        return self in {
            AuthoritySource.REVIEWED,
            AuthoritySource.PRODUCTION,
            AuthoritySource.FIXTURE,
        }

    @property
    def is_abstaining_source(self) -> bool:
        return self in {
            AuthoritySource.SERVER_ASSERTED,
            AuthoritySource.PROSE_INFERRED,
            AuthoritySource.INVENTED,
            AuthoritySource.MISSING,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        raise SecurityRepairError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise SecurityRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise SecurityRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise SecurityRepairError(f"{name} exceeds its byte bound")
    if identifier and result and not _IDENTIFIER_RE.fullmatch(result):
        raise SecurityRepairError(f"{name} must be a closed identifier")
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SecurityRepairError(f"{name} must be a boolean")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SecurityRepairError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative(value, name)


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise SecurityRepairError(f"unsupported {name}: {value!r}") from exc


def _cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=MAX_ID_BYTES)
    if text and not _CID_RE.fullmatch(text):
        # Allow content_identity-shaped digests that may not match the short
        # multiformats prefix set when fixtures supply sha256: digests.
        if not text.startswith("sha256:") and not text.startswith("b"):
            raise SecurityRepairError(f"{name} must be a content identity")
    return text


def _string_tuple(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
    ordered: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise SecurityRepairError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise SecurityRepairError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, f"{name}[{index}]", identifier=True)
        if text in seen:
            raise SecurityRepairError(f"{name} must not contain duplicates")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise SecurityRepairError(f"{name} must not be empty")
    if ordered:
        return tuple(result)
    return tuple(sorted(result))


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    for key in payload:
        lowered = str(key).strip().lower()
        if lowered in _FORBIDDEN_PAYLOAD_KEYS:
            raise SecurityRepairError(
                f"{label} contains forbidden field {lowered!r}"
            )


def _tuple_of(
    values: Any,
    name: str,
    factory,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
) -> tuple[Any, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        items = values
    else:
        raise SecurityRepairError(f"{name} must be a sequence")
    if len(items) > maximum:
        raise SecurityRepairError(f"{name} exceeds its item bound")
    result = tuple(factory(item, f"{name}[{index}]") for index, item in enumerate(items))
    if required and not result:
        raise SecurityRepairError(f"{name} must not be empty")
    return result


# ---------------------------------------------------------------------------
# SecurityIR contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityGrant(CanonicalContract):
    """One reviewed UCAN-style capability grant (MCP++ Profile C).

    Grants bind issuer → audience with a capability, resource, temporal window,
    and optional revocation marker.  Execution-time checks re-validate every
    field; server assertions never short-circuit this contract.
    """

    SCHEMA: ClassVar[str] = SECURITY_GRANT_SCHEMA

    grant_id: str
    issuer: str
    audience: str
    capability: str
    resource: str
    not_before_ms: int = 0
    expires_at_ms: int | None = None
    revoked: bool = False
    revocation_id: str = ""
    proof_cid: str = ""
    obligations: tuple[str, ...] = ()
    effect_ids: tuple[str, ...] = ()
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grant_id", _text(self.grant_id, "grant_id", identifier=True)
        )
        object.__setattr__(
            self, "issuer", _text(self.issuer, "issuer", identifier=True)
        )
        object.__setattr__(
            self, "audience", _text(self.audience, "audience", identifier=True)
        )
        object.__setattr__(
            self, "capability", _text(self.capability, "capability", identifier=True)
        )
        object.__setattr__(
            self, "resource", _text(self.resource, "resource", identifier=True)
        )
        not_before = _nonnegative(self.not_before_ms, "not_before_ms")
        expires = _optional_nonnegative(self.expires_at_ms, "expires_at_ms")
        if expires is not None and expires <= not_before:
            raise SecurityRepairError(
                "expires_at_ms must be greater than not_before_ms"
            )
        object.__setattr__(self, "not_before_ms", not_before)
        object.__setattr__(self, "expires_at_ms", expires)
        object.__setattr__(self, "revoked", _bool(self.revoked, "revoked"))
        object.__setattr__(
            self,
            "revocation_id",
            _text(self.revocation_id, "revocation_id", required=False, identifier=True),
        )
        object.__setattr__(
            self,
            "proof_cid",
            _cid(self.proof_cid, "proof_cid", required=False)
            if self.proof_cid
            else "",
        )
        object.__setattr__(
            self,
            "obligations",
            _string_tuple(self.obligations, "obligations", ordered=True),
        )
        object.__setattr__(
            self,
            "effect_ids",
            _string_tuple(self.effect_ids, "effect_ids", ordered=True),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        if self.authority.is_abstaining_source:
            raise SecurityRepairError(
                "SecurityGrant cannot carry invented/server-asserted authority"
            )

    def status_at(self, evaluated_at_ms: int, *, expected_audience: str) -> GrantStatus:
        """Return the grant lifecycle status at an evaluation instant."""

        if self.revoked or self.revocation_id:
            return GrantStatus.REVOKED
        if expected_audience and self.audience != expected_audience:
            return GrantStatus.WRONG_AUDIENCE
        if evaluated_at_ms < self.not_before_ms:
            return GrantStatus.NOT_YET_VALID
        if self.expires_at_ms is not None and evaluated_at_ms >= self.expires_at_ms:
            return GrantStatus.EXPIRED
        # Stale: expired-or-equal window already handled; treat near-miss as
        # expired above.  A grant is "stale" when it was active but the policy
        # revision that issued it is no longer current — callers may set
        # revoked/revocation_id for that; otherwise ACTIVE.
        return GrantStatus.ACTIVE

    def _payload(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "issuer": self.issuer,
            "audience": self.audience,
            "capability": self.capability,
            "resource": self.resource,
            "not_before_ms": self.not_before_ms,
            "expires_at_ms": self.expires_at_ms,
            "revoked": self.revoked,
            "revocation_id": self.revocation_id,
            "proof_cid": self.proof_cid,
            "obligations": list(self.obligations),
            "effect_ids": list(self.effect_ids),
            "authority": self.authority.value,
            "profile": MCP_PROFILE_C,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityGrant":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("security grant must be an object")
        _reject_forbidden_fields(payload, label="security grant")
        return cls(
            grant_id=payload.get("grant_id", ""),
            issuer=payload.get("issuer", ""),
            audience=payload.get("audience", ""),
            capability=payload.get("capability", ""),
            resource=payload.get("resource", ""),
            not_before_ms=payload.get("not_before_ms", 0),
            expires_at_ms=payload.get("expires_at_ms"),
            revoked=payload.get("revoked", False),
            revocation_id=payload.get("revocation_id", ""),
            proof_cid=payload.get("proof_cid", ""),
            obligations=payload.get("obligations") or (),
            effect_ids=payload.get("effect_ids") or (),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class AuthorizationBinding(CanonicalContract):
    """Reviewed principal ↔ audience ↔ capability binding restored by operators."""

    SCHEMA: ClassVar[str] = SECURITY_BINDING_SCHEMA

    binding_id: str
    principal: str
    audience: str
    capability: str
    resource: str
    grant_id: str
    effect_ids: tuple[str, ...] = ()
    authority: AuthoritySource = AuthoritySource.REVIEWED
    profile: str = MCP_PROFILE_C

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _text(self.binding_id, "binding_id", identifier=True)
        )
        object.__setattr__(
            self, "principal", _text(self.principal, "principal", identifier=True)
        )
        object.__setattr__(
            self, "audience", _text(self.audience, "audience", identifier=True)
        )
        object.__setattr__(
            self, "capability", _text(self.capability, "capability", identifier=True)
        )
        object.__setattr__(
            self, "resource", _text(self.resource, "resource", identifier=True)
        )
        object.__setattr__(
            self, "grant_id", _text(self.grant_id, "grant_id", identifier=True)
        )
        object.__setattr__(
            self,
            "effect_ids",
            _string_tuple(self.effect_ids, "effect_ids", ordered=True),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        object.__setattr__(
            self, "profile", _text(self.profile, "profile", identifier=True)
        )
        if self.authority.is_abstaining_source:
            raise SecurityRepairError(
                "AuthorizationBinding cannot invent or server-assert authority"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "principal": self.principal,
            "audience": self.audience,
            "capability": self.capability,
            "resource": self.resource,
            "grant_id": self.grant_id,
            "effect_ids": list(self.effect_ids),
            "authority": self.authority.value,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationBinding":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("authorization binding must be an object")
        _reject_forbidden_fields(payload, label="authorization binding")
        return cls(
            binding_id=payload.get("binding_id", ""),
            principal=payload.get("principal", ""),
            audience=payload.get("audience", ""),
            capability=payload.get("capability", ""),
            resource=payload.get("resource", ""),
            grant_id=payload.get("grant_id", ""),
            effect_ids=payload.get("effect_ids") or (),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
            profile=payload.get("profile", MCP_PROFILE_C),
        )


@dataclass(frozen=True)
class EffectAnnotation(CanonicalContract):
    """Reviewed effect classification for one action/resource pair."""

    SCHEMA: ClassVar[str] = SECURITY_EFFECT_SCHEMA

    effect_id: str
    action: str
    resource: str
    effect_class: EffectClass
    declared: bool = True
    authority: AuthoritySource = AuthoritySource.REVIEWED
    obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "effect_id", _text(self.effect_id, "effect_id", identifier=True)
        )
        object.__setattr__(
            self, "action", _text(self.action, "action", identifier=True)
        )
        object.__setattr__(
            self, "resource", _text(self.resource, "resource", identifier=True)
        )
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, EffectClass, "effect_class"),
        )
        object.__setattr__(self, "declared", _bool(self.declared, "declared"))
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        object.__setattr__(
            self,
            "obligations",
            _string_tuple(self.obligations, "obligations", ordered=True),
        )
        if self.authority.is_abstaining_source:
            raise SecurityRepairError(
                "EffectAnnotation cannot invent or server-assert effect classes"
            )
        if not self.declared:
            raise SecurityRepairError(
                "EffectAnnotation must be declared; undeclared effects fail closed"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "action": self.action,
            "resource": self.resource,
            "effect_class": self.effect_class.value,
            "declared": True,
            "authority": self.authority.value,
            "obligations": list(self.obligations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectAnnotation":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("effect annotation must be an object")
        _reject_forbidden_fields(payload, label="effect annotation")
        return cls(
            effect_id=payload.get("effect_id", ""),
            action=payload.get("action", ""),
            resource=payload.get("resource", ""),
            effect_class=payload.get("effect_class", EffectClass.NO_EFFECT),
            declared=payload.get("declared", True),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
            obligations=payload.get("obligations") or (),
        )


@dataclass(frozen=True)
class PolicyGate(CanonicalContract):
    """MCP++ Profile D policy gate state for one action evaluation surface."""

    SCHEMA: ClassVar[str] = SECURITY_POLICY_GATE_SCHEMA

    gate_id: str
    policy_id: str
    availability: PolicyAvailability
    decision: str = ""  # allow | deny | allow_with_obligations | "" when missing
    obligations: tuple[str, ...] = ()
    justification: str = ""
    authority: AuthoritySource = AuthoritySource.REVIEWED
    profile: str = MCP_PROFILE_D

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gate_id", _text(self.gate_id, "gate_id", identifier=True)
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", identifier=True)
        )
        object.__setattr__(
            self,
            "availability",
            _enum(self.availability, PolicyAvailability, "availability"),
        )
        decision = _text(self.decision, "decision", required=False, identifier=True)
        if decision and decision not in {
            "allow",
            "deny",
            "allow_with_obligations",
        }:
            raise SecurityRepairError(
                "decision must be allow, deny, or allow_with_obligations"
            )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(
            self,
            "obligations",
            _string_tuple(self.obligations, "obligations", ordered=True),
        )
        object.__setattr__(
            self,
            "justification",
            _text(self.justification, "justification", required=False),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        object.__setattr__(
            self, "profile", _text(self.profile, "profile", identifier=True)
        )

    @property
    def is_available(self) -> bool:
        return self.availability is PolicyAvailability.AVAILABLE

    @property
    def has_decision(self) -> bool:
        return bool(self.decision)

    def _payload(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "policy_id": self.policy_id,
            "availability": self.availability.value,
            "decision": self.decision,
            "obligations": list(self.obligations),
            "justification": self.justification,
            "authority": self.authority.value,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyGate":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("policy gate must be an object")
        _reject_forbidden_fields(payload, label="policy gate")
        return cls(
            gate_id=payload.get("gate_id", ""),
            policy_id=payload.get("policy_id", ""),
            availability=payload.get("availability", PolicyAvailability.MISSING),
            decision=payload.get("decision", ""),
            obligations=payload.get("obligations") or (),
            justification=payload.get("justification", ""),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
            profile=payload.get("profile", MCP_PROFILE_D),
        )


@dataclass(frozen=True)
class SecurityIR(CanonicalContract):
    """Closed SecurityIR document binding principals, grants, effects, and gates.

    Interface: ``SecurityIR``.  Only reviewed/production/fixture authority may
    authorize repairs; operators never invent grants, effects, or policy.
    """

    SCHEMA: ClassVar[str] = SECURITY_IR_SCHEMA
    INTERFACE: ClassVar[str] = SECURITY_IR_INTERFACE

    document_id: str
    trusted_issuers: tuple[str, ...]
    grants: tuple[SecurityGrant, ...]
    bindings: tuple[AuthorizationBinding, ...]
    effects: tuple[EffectAnnotation, ...]
    policy_gates: tuple[PolicyGate, ...] = ()
    revoked_proof_cids: tuple[str, ...] = ()
    authority: AuthoritySource = AuthoritySource.REVIEWED
    profiles: tuple[str, ...] = MCP_PROFILES_C_D
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "document_id", identifier=True),
        )
        object.__setattr__(
            self,
            "trusted_issuers",
            _string_tuple(self.trusted_issuers, "trusted_issuers", required=True),
        )
        grants = _tuple_of(
            self.grants,
            "grants",
            lambda item, label: (
                item if isinstance(item, SecurityGrant) else SecurityGrant.from_dict(item)
            ),
            required=False,
        )
        grant_ids = [item.grant_id for item in grants]
        if len(grant_ids) != len(set(grant_ids)):
            raise SecurityRepairError("grant_id values must be unique")
        object.__setattr__(self, "grants", grants)

        bindings = _tuple_of(
            self.bindings,
            "bindings",
            lambda item, label: (
                item
                if isinstance(item, AuthorizationBinding)
                else AuthorizationBinding.from_dict(item)
            ),
            required=False,
        )
        binding_ids = [item.binding_id for item in bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise SecurityRepairError("binding_id values must be unique")
        object.__setattr__(self, "bindings", bindings)

        effects = _tuple_of(
            self.effects,
            "effects",
            lambda item, label: (
                item
                if isinstance(item, EffectAnnotation)
                else EffectAnnotation.from_dict(item)
            ),
            required=False,
        )
        effect_ids = [item.effect_id for item in effects]
        if len(effect_ids) != len(set(effect_ids)):
            raise SecurityRepairError("effect_id values must be unique")
        object.__setattr__(self, "effects", effects)

        gates = _tuple_of(
            self.policy_gates,
            "policy_gates",
            lambda item, label: (
                item if isinstance(item, PolicyGate) else PolicyGate.from_dict(item)
            ),
            required=False,
        )
        object.__setattr__(self, "policy_gates", gates)
        object.__setattr__(
            self,
            "revoked_proof_cids",
            _string_tuple(self.revoked_proof_cids, "revoked_proof_cids"),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        profiles = _string_tuple(self.profiles, "profiles", required=True, ordered=True)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self,
            "source_refs",
            _string_tuple(self.source_refs, "source_refs", ordered=True),
        )

    @property
    def semantic_digest(self) -> str:
        return _digest(self.to_dict())

    def grant_by_id(self, grant_id: str) -> SecurityGrant | None:
        for grant in self.grants:
            if grant.grant_id == grant_id:
                return grant
        return None

    def binding_by_id(self, binding_id: str) -> AuthorizationBinding | None:
        for binding in self.bindings:
            if binding.binding_id == binding_id:
                return binding
        return None

    def effect_by_id(self, effect_id: str) -> EffectAnnotation | None:
        for effect in self.effects:
            if effect.effect_id == effect_id:
                return effect
        return None

    def primary_policy_gate(self) -> PolicyGate | None:
        return self.policy_gates[0] if self.policy_gates else None

    def with_binding(self, binding: AuthorizationBinding) -> "SecurityIR":
        others = tuple(b for b in self.bindings if b.binding_id != binding.binding_id)
        return SecurityIR(
            document_id=self.document_id,
            trusted_issuers=self.trusted_issuers,
            grants=self.grants,
            bindings=(*others, binding),
            effects=self.effects,
            policy_gates=self.policy_gates,
            revoked_proof_cids=self.revoked_proof_cids,
            authority=self.authority,
            profiles=self.profiles,
            source_refs=self.source_refs,
        )

    def with_effect(self, effect: EffectAnnotation) -> "SecurityIR":
        others = tuple(e for e in self.effects if e.effect_id != effect.effect_id)
        return SecurityIR(
            document_id=self.document_id,
            trusted_issuers=self.trusted_issuers,
            grants=self.grants,
            bindings=self.bindings,
            effects=(*others, effect),
            policy_gates=self.policy_gates,
            revoked_proof_cids=self.revoked_proof_cids,
            authority=self.authority,
            profiles=self.profiles,
            source_refs=self.source_refs,
        )

    def with_policy_gate(self, gate: PolicyGate) -> "SecurityIR":
        others = tuple(g for g in self.policy_gates if g.gate_id != gate.gate_id)
        return SecurityIR(
            document_id=self.document_id,
            trusted_issuers=self.trusted_issuers,
            grants=self.grants,
            bindings=self.bindings,
            effects=self.effects,
            policy_gates=(*others, gate),
            revoked_proof_cids=self.revoked_proof_cids,
            authority=self.authority,
            profiles=self.profiles,
            source_refs=self.source_refs,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "document_id": self.document_id,
            "trusted_issuers": list(self.trusted_issuers),
            "grants": [item.to_dict() for item in self.grants],
            "bindings": [item.to_dict() for item in self.bindings],
            "effects": [item.to_dict() for item in self.effects],
            "policy_gates": [item.to_dict() for item in self.policy_gates],
            "revoked_proof_cids": list(self.revoked_proof_cids),
            "authority": self.authority.value,
            "profiles": list(self.profiles),
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityIR":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("SecurityIR must be an object")
        _reject_forbidden_fields(payload, label="SecurityIR")
        return cls(
            document_id=payload.get("document_id", ""),
            trusted_issuers=payload.get("trusted_issuers") or (),
            grants=payload.get("grants") or (),
            bindings=payload.get("bindings") or (),
            effects=payload.get("effects") or (),
            policy_gates=payload.get("policy_gates") or (),
            revoked_proof_cids=payload.get("revoked_proof_cids") or (),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
            profiles=payload.get("profiles") or MCP_PROFILES_C_D,
            source_refs=payload.get("source_refs") or (),
        )


# ---------------------------------------------------------------------------
# Execution-time authorization request / decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityAuthorizationRequest(CanonicalContract):
    """Exact principal/action/resource/effect request for execution-time check."""

    SCHEMA: ClassVar[str] = SECURITY_AUTHZ_REQUEST_SCHEMA

    principal: str
    audience: str
    capability: str
    resource: str
    action: str
    expected_effects: tuple[str, ...]
    evaluated_at_ms: int
    fulfilled_obligations: tuple[str, ...] = ()
    # Server-supplied claims are accepted as opaque telemetry only and NEVER
    # establish a permit.  Presence of assertion keys is recorded and ignored.
    server_assertions: Mapping[str, Any] = MappingProxyType({})
    grant_id: str = ""
    policy_gate_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal", _text(self.principal, "principal", identifier=True)
        )
        object.__setattr__(
            self, "audience", _text(self.audience, "audience", identifier=True)
        )
        object.__setattr__(
            self, "capability", _text(self.capability, "capability", identifier=True)
        )
        object.__setattr__(
            self, "resource", _text(self.resource, "resource", identifier=True)
        )
        object.__setattr__(
            self, "action", _text(self.action, "action", identifier=True)
        )
        object.__setattr__(
            self,
            "expected_effects",
            _string_tuple(
                self.expected_effects, "expected_effects", required=True, ordered=True
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _nonnegative(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        object.__setattr__(
            self,
            "fulfilled_obligations",
            _string_tuple(
                self.fulfilled_obligations, "fulfilled_obligations", ordered=True
            ),
        )
        if self.server_assertions is None:
            object.__setattr__(self, "server_assertions", MappingProxyType({}))
        elif not isinstance(self.server_assertions, Mapping):
            raise SecurityRepairError("server_assertions must be an object")
        else:
            _reject_forbidden_fields(
                self.server_assertions, label="server_assertions"
            )
            object.__setattr__(
                self,
                "server_assertions",
                MappingProxyType(dict(self.server_assertions)),
            )
        object.__setattr__(
            self,
            "grant_id",
            _text(self.grant_id, "grant_id", required=False, identifier=True),
        )
        object.__setattr__(
            self,
            "policy_gate_id",
            _text(
                self.policy_gate_id, "policy_gate_id", required=False, identifier=True
            ),
        )

    @property
    def carries_server_authorization_assertion(self) -> bool:
        """True when the payload tries to smuggle a permit via server claims."""

        for key in self.server_assertions:
            if str(key).strip().lower() in _SERVER_ASSERTION_KEYS:
                return True
            value = self.server_assertions[key]
            if isinstance(value, bool) and value is True and str(key).lower() in {
                "ok",
                "success",
                "allowed",
                "permit",
            }:
                return True
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "audience": self.audience,
            "capability": self.capability,
            "resource": self.resource,
            "action": self.action,
            "expected_effects": list(self.expected_effects),
            "evaluated_at_ms": self.evaluated_at_ms,
            "fulfilled_obligations": list(self.fulfilled_obligations),
            "server_assertions": dict(self.server_assertions),
            "grant_id": self.grant_id,
            "policy_gate_id": self.policy_gate_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityAuthorizationRequest":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("security authorization request must be an object")
        _reject_forbidden_fields(payload, label="security authorization request")
        return cls(
            principal=payload.get("principal", ""),
            audience=payload.get("audience", ""),
            capability=payload.get("capability", ""),
            resource=payload.get("resource", ""),
            action=payload.get("action", ""),
            expected_effects=payload.get("expected_effects") or (),
            evaluated_at_ms=payload.get("evaluated_at_ms", -1),
            fulfilled_obligations=payload.get("fulfilled_obligations") or (),
            server_assertions=payload.get("server_assertions") or {},
            grant_id=payload.get("grant_id", ""),
            policy_gate_id=payload.get("policy_gate_id", ""),
        )


@dataclass(frozen=True)
class SecurityDecision(CanonicalContract):
    """Execution-time security decision.  Authorization never proves code correctness."""

    SCHEMA: ClassVar[str] = SECURITY_DECISION_SCHEMA

    verdict: SecurityVerdict
    reason: DenialReasonCode
    security_ir_identity: str
    request_identity: str
    evaluated_at_ms: int
    matched_grant_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    ignored_server_assertions: bool = False
    establishes_generated_code_correctness: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum(self.verdict, SecurityVerdict, "verdict")
        )
        object.__setattr__(
            self, "reason", _enum(self.reason, DenialReasonCode, "reason")
        )
        object.__setattr__(
            self,
            "security_ir_identity",
            _text(self.security_ir_identity, "security_ir_identity"),
        )
        object.__setattr__(
            self,
            "request_identity",
            _text(self.request_identity, "request_identity"),
        )
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _nonnegative(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        object.__setattr__(
            self,
            "matched_grant_ids",
            _string_tuple(self.matched_grant_ids, "matched_grant_ids", ordered=True),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes", ordered=True),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise SecurityRepairError("reason_codes exceeds its item bound")
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        object.__setattr__(
            self,
            "ignored_server_assertions",
            _bool(self.ignored_server_assertions, "ignored_server_assertions"),
        )
        # Authority seal: authorization never establishes code correctness.
        if self.establishes_generated_code_correctness is not False:
            raise SecurityRepairError(
                "authorization cannot establish generated-code correctness"
            )
        object.__setattr__(self, "establishes_generated_code_correctness", False)
        if self.verdict is SecurityVerdict.PERMIT:
            if self.reason is not DenialReasonCode.ALLOWED or not self.matched_grant_ids:
                raise SecurityRepairError(
                    "permit decision requires allowed reason and matched grants"
                )
        elif self.reason is DenialReasonCode.ALLOWED:
            raise SecurityRepairError("deny decision cannot carry allowed reason")

    @property
    def permitted(self) -> bool:
        return self.verdict is SecurityVerdict.PERMIT

    def _payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason.value,
            "security_ir_identity": self.security_ir_identity,
            "request_identity": self.request_identity,
            "evaluated_at_ms": self.evaluated_at_ms,
            "matched_grant_ids": list(self.matched_grant_ids),
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "ignored_server_assertions": self.ignored_server_assertions,
            "permits_action": self.permitted,
            "establishes_generated_code_correctness": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityDecision":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("security decision must be an object")
        _reject_forbidden_fields(payload, label="security decision")
        if payload.get("establishes_generated_code_correctness") not in (None, False):
            raise SecurityRepairError(
                "authorization cannot establish generated-code correctness"
            )
        return cls(
            verdict=payload.get("verdict", SecurityVerdict.DENY),
            reason=payload.get("reason", DenialReasonCode.MALFORMED_REQUEST),
            security_ir_identity=payload.get("security_ir_identity", ""),
            request_identity=payload.get("request_identity", ""),
            evaluated_at_ms=payload.get("evaluated_at_ms", 0),
            matched_grant_ids=payload.get("matched_grant_ids") or (),
            reason_codes=payload.get("reason_codes") or (),
            detail=payload.get("detail", ""),
            ignored_server_assertions=payload.get("ignored_server_assertions", False),
            establishes_generated_code_correctness=False,
        )


def _deny(
    *,
    security_ir: SecurityIR | None,
    request: SecurityAuthorizationRequest,
    reason: DenialReasonCode,
    extra_codes: Sequence[str] = (),
    detail: str = "",
    ignored_server: bool = False,
) -> SecurityDecision:
    codes: list[str] = []
    for code in (reason.value, *extra_codes):
        if code and code not in codes:
            codes.append(code)
    return SecurityDecision(
        verdict=SecurityVerdict.DENY,
        reason=reason,
        security_ir_identity=(
            security_ir.content_id
            if security_ir is not None
            else content_identity({"missing_security_ir": True})
        ),
        request_identity=request.content_id,
        evaluated_at_ms=request.evaluated_at_ms,
        matched_grant_ids=(),
        reason_codes=tuple(codes[:MAX_REASON_CODES]),
        detail=detail,
        ignored_server_assertions=ignored_server,
    )


def evaluate_security_authorization(
    security_ir: SecurityIR | None,
    request: SecurityAuthorizationRequest,
) -> SecurityDecision:
    """Execution-time fail-closed authorization against reviewed SecurityIR.

    Normative acceptance rules:

    * Policy outage / missing decision → **deny**
    * Stale / revoked / wrong-audience grants → **fail (deny)**
    * Server-supplied authorization assertions are **never trusted**
    """

    if not isinstance(request, SecurityAuthorizationRequest):
        if isinstance(request, Mapping):
            request = SecurityAuthorizationRequest.from_dict(request)
        else:
            raise SecurityRepairError("request must be a SecurityAuthorizationRequest")

    ignored_server = request.carries_server_authorization_assertion

    if security_ir is None:
        return _deny(
            security_ir=None,
            request=request,
            reason=DenialReasonCode.MISSING_SECURITY_IR,
            extra_codes=("fail_closed",),
            detail="no SecurityIR available",
            ignored_server=ignored_server,
        )
    if not isinstance(security_ir, SecurityIR):
        if isinstance(security_ir, Mapping):
            security_ir = SecurityIR.from_dict(security_ir)
        else:
            raise SecurityRepairError("security_ir must be a SecurityIR document")

    if not security_ir.authority.authorizes_security_source:
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.INVENTED_AUTHORITY,
            extra_codes=(f"authority:{security_ir.authority.value}",),
            detail="SecurityIR authority does not authorize evaluation",
            ignored_server=ignored_server,
        )

    # --- Profile D: policy gate (outage / missing decision deny) ---
    gate = None
    if request.policy_gate_id:
        for candidate in security_ir.policy_gates:
            if candidate.gate_id == request.policy_gate_id:
                gate = candidate
                break
        if gate is None:
            return _deny(
                security_ir=security_ir,
                request=request,
                reason=DenialReasonCode.MISSING_DECISION,
                extra_codes=("policy_gate_not_found",),
                detail=f"policy gate {request.policy_gate_id} not found",
                ignored_server=ignored_server,
            )
    else:
        gate = security_ir.primary_policy_gate()

    if gate is None:
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.MISSING_DECISION,
            extra_codes=("no_policy_gate",),
            detail="policy decision is missing",
            ignored_server=ignored_server,
        )

    if gate.availability is PolicyAvailability.OUTAGE:
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.POLICY_OUTAGE,
            extra_codes=("policy_surface_unavailable",),
            detail=gate.justification or "policy evaluation surface outage",
            ignored_server=ignored_server,
        )
    if gate.availability in {PolicyAvailability.MISSING, PolicyAvailability.UNKNOWN}:
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.MISSING_DECISION,
            extra_codes=(f"policy_availability:{gate.availability.value}",),
            detail="policy decision unavailable",
            ignored_server=ignored_server,
        )
    if not gate.has_decision:
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.MISSING_DECISION,
            extra_codes=("empty_policy_decision",),
            detail="policy gate has no decision",
            ignored_server=ignored_server,
        )
    if gate.decision == "deny":
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.NO_APPLICABLE_GRANT,
            extra_codes=("policy_denied",),
            detail=gate.justification or "policy denied",
            ignored_server=ignored_server,
        )

    # Obligations required by the policy gate must be fulfilled.
    unmet = [
        obligation
        for obligation in gate.obligations
        if obligation not in request.fulfilled_obligations
    ]
    if unmet and gate.decision == "allow_with_obligations":
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=DenialReasonCode.OBLIGATION_UNMET,
            extra_codes=tuple(f"unmet:{item}" for item in unmet[:8]),
            detail="policy obligations not fulfilled",
            ignored_server=ignored_server,
        )

    # --- Profile C: grant / audience / temporal / revocation checks ---
    candidates: list[SecurityGrant] = []
    if request.grant_id:
        grant = security_ir.grant_by_id(request.grant_id)
        if grant is None:
            return _deny(
                security_ir=security_ir,
                request=request,
                reason=DenialReasonCode.NO_APPLICABLE_GRANT,
                extra_codes=("grant_not_found",),
                detail=f"grant {request.grant_id} not found",
                ignored_server=ignored_server,
            )
        candidates = [grant]
    else:
        candidates = [
            grant
            for grant in security_ir.grants
            if grant.audience == request.audience
            and grant.capability == request.capability
            and grant.resource == request.resource
        ]

    if not candidates:
        # Server assertion cannot mint a grant.
        extra = ("server_assertion_ignored",) if ignored_server else ()
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=(
                DenialReasonCode.SERVER_ASSERTION_UNTRUSTED
                if ignored_server
                else DenialReasonCode.NO_APPLICABLE_GRANT
            ),
            extra_codes=extra,
            detail="no applicable grant; server assertions are not trusted",
            ignored_server=ignored_server,
        )

    failures: list[DenialReasonCode] = []
    matched: list[str] = []
    for grant in candidates:
        if grant.issuer not in security_ir.trusted_issuers:
            failures.append(DenialReasonCode.INVENTED_GRANT)
            continue
        if grant.proof_cid and grant.proof_cid in security_ir.revoked_proof_cids:
            failures.append(DenialReasonCode.REVOKED)
            continue
        status = grant.status_at(
            request.evaluated_at_ms, expected_audience=request.audience
        )
        if status is GrantStatus.REVOKED:
            failures.append(DenialReasonCode.REVOKED)
            continue
        if status is GrantStatus.WRONG_AUDIENCE:
            failures.append(DenialReasonCode.WRONG_AUDIENCE)
            continue
        if status is GrantStatus.EXPIRED:
            failures.append(DenialReasonCode.EXPIRED)
            continue
        if status is GrantStatus.NOT_YET_VALID:
            failures.append(DenialReasonCode.NOT_YET_VALID)
            continue
        if status is GrantStatus.STALE:
            failures.append(DenialReasonCode.STALE_GRANT)
            continue
        if grant.capability != request.capability:
            failures.append(DenialReasonCode.CAPABILITY_MISMATCH)
            continue
        # Profile C actor binding: the executing principal must be the leaf audience.
        if request.principal != grant.audience:
            failures.append(DenialReasonCode.PRINCIPAL_MISMATCH)
            continue
        # Effects declared on the grant must cover expected effects.
        if grant.effect_ids:
            missing_effects = [
                effect
                for effect in request.expected_effects
                if effect not in grant.effect_ids
            ]
            if missing_effects:
                failures.append(DenialReasonCode.EFFECT_MISMATCH)
                continue
        # Reviewed effect annotations must declare every expected effect.
        declared_ids = {effect.effect_id for effect in security_ir.effects}
        undeclared = [
            effect
            for effect in request.expected_effects
            if effect not in declared_ids
        ]
        if undeclared:
            failures.append(DenialReasonCode.UNDECLARED_EFFECT)
            continue
        # Grant obligations.
        unmet_grant = [
            obligation
            for obligation in grant.obligations
            if obligation not in request.fulfilled_obligations
        ]
        if unmet_grant:
            failures.append(DenialReasonCode.OBLIGATION_UNMET)
            continue
        matched.append(grant.grant_id)

    if not matched:
        # Prefer the most severe failure reason in a stable order.
        priority = (
            DenialReasonCode.REVOKED,
            DenialReasonCode.WRONG_AUDIENCE,
            DenialReasonCode.STALE_GRANT,
            DenialReasonCode.EXPIRED,
            DenialReasonCode.NOT_YET_VALID,
            DenialReasonCode.INVENTED_GRANT,
            DenialReasonCode.EFFECT_MISMATCH,
            DenialReasonCode.UNDECLARED_EFFECT,
            DenialReasonCode.OBLIGATION_UNMET,
            DenialReasonCode.CAPABILITY_MISMATCH,
            DenialReasonCode.PRINCIPAL_MISMATCH,
            DenialReasonCode.NO_APPLICABLE_GRANT,
        )
        reason = DenialReasonCode.NO_APPLICABLE_GRANT
        for candidate_reason in priority:
            if candidate_reason in failures:
                reason = candidate_reason
                break
        # Stale is an alias surface for expired/not-yet-valid in acceptance text.
        if reason in {DenialReasonCode.EXPIRED, DenialReasonCode.NOT_YET_VALID}:
            extra = (reason.value,)
            reason = DenialReasonCode.STALE_GRANT
        else:
            # Primary reason is already recorded by _deny; avoid duplicates.
            extra = ()
        if ignored_server:
            extra = (*extra, DenialReasonCode.SERVER_ASSERTION_UNTRUSTED.value)
        return _deny(
            security_ir=security_ir,
            request=request,
            reason=reason,
            extra_codes=extra,
            detail="grant validation failed",
            ignored_server=ignored_server,
        )

    # Even on permit, record that server assertions were ignored if present.
    reason_codes = [DenialReasonCode.ALLOWED.value, "execution_time_check"]
    if ignored_server:
        reason_codes.append(DenialReasonCode.SERVER_ASSERTION_UNTRUSTED.value)
        reason_codes.append("server_assertion_ignored")
    reason_codes.append("policy_gate_ok")
    reason_codes.append("grant_active")

    return SecurityDecision(
        verdict=SecurityVerdict.PERMIT,
        reason=DenialReasonCode.ALLOWED,
        security_ir_identity=security_ir.content_id,
        request_identity=request.content_id,
        evaluated_at_ms=request.evaluated_at_ms,
        matched_grant_ids=tuple(matched),
        reason_codes=tuple(reason_codes[:MAX_REASON_CODES]),
        detail="permitted by reviewed SecurityIR grant and policy gate",
        ignored_server_assertions=ignored_server,
    )


# ---------------------------------------------------------------------------
# Repair request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecurityRepairRequest(CanonicalContract):
    """Closed input for one security repair operator application."""

    SCHEMA: ClassVar[str] = SECURITY_REQUEST_SCHEMA

    security_ir: SecurityIR
    role: OperatorRole
    # Reviewed target artifacts the operator may restore (never invent).
    reviewed_binding: AuthorizationBinding | None = None
    reviewed_effect: EffectAnnotation | None = None
    reviewed_policy_gate: PolicyGate | None = None
    # Current (possibly drifted / missing) state under repair.
    current_binding: AuthorizationBinding | None = None
    current_effect: EffectAnnotation | None = None
    current_policy_gate: PolicyGate | None = None
    # Optional execution-time request used to prove the repaired gate.
    authorization_request: SecurityAuthorizationRequest | None = None
    require_execution_check: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.security_ir, SecurityIR):
            if isinstance(self.security_ir, Mapping):
                object.__setattr__(
                    self, "security_ir", SecurityIR.from_dict(self.security_ir)
                )
            else:
                raise SecurityRepairError("security_ir must be a SecurityIR document")
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))

        def _opt_binding(value: Any, name: str) -> AuthorizationBinding | None:
            if value is None:
                return None
            if isinstance(value, AuthorizationBinding):
                return value
            if isinstance(value, Mapping):
                return AuthorizationBinding.from_dict(value)
            raise SecurityRepairError(f"{name} must be an AuthorizationBinding or null")

        def _opt_effect(value: Any, name: str) -> EffectAnnotation | None:
            if value is None:
                return None
            if isinstance(value, EffectAnnotation):
                return value
            if isinstance(value, Mapping):
                return EffectAnnotation.from_dict(value)
            raise SecurityRepairError(f"{name} must be an EffectAnnotation or null")

        def _opt_gate(value: Any, name: str) -> PolicyGate | None:
            if value is None:
                return None
            if isinstance(value, PolicyGate):
                return value
            if isinstance(value, Mapping):
                return PolicyGate.from_dict(value)
            raise SecurityRepairError(f"{name} must be a PolicyGate or null")

        object.__setattr__(
            self, "reviewed_binding", _opt_binding(self.reviewed_binding, "reviewed_binding")
        )
        object.__setattr__(
            self, "reviewed_effect", _opt_effect(self.reviewed_effect, "reviewed_effect")
        )
        object.__setattr__(
            self,
            "reviewed_policy_gate",
            _opt_gate(self.reviewed_policy_gate, "reviewed_policy_gate"),
        )
        object.__setattr__(
            self, "current_binding", _opt_binding(self.current_binding, "current_binding")
        )
        object.__setattr__(
            self, "current_effect", _opt_effect(self.current_effect, "current_effect")
        )
        object.__setattr__(
            self,
            "current_policy_gate",
            _opt_gate(self.current_policy_gate, "current_policy_gate"),
        )
        if self.authorization_request is not None and not isinstance(
            self.authorization_request, SecurityAuthorizationRequest
        ):
            if isinstance(self.authorization_request, Mapping):
                object.__setattr__(
                    self,
                    "authorization_request",
                    SecurityAuthorizationRequest.from_dict(self.authorization_request),
                )
            else:
                raise SecurityRepairError(
                    "authorization_request must be a SecurityAuthorizationRequest or null"
                )
        object.__setattr__(
            self,
            "require_execution_check",
            _bool(self.require_execution_check, "require_execution_check"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "security_ir": self.security_ir.to_dict(),
            "role": self.role.value,
            "reviewed_binding": (
                None if self.reviewed_binding is None else self.reviewed_binding.to_dict()
            ),
            "reviewed_effect": (
                None if self.reviewed_effect is None else self.reviewed_effect.to_dict()
            ),
            "reviewed_policy_gate": (
                None
                if self.reviewed_policy_gate is None
                else self.reviewed_policy_gate.to_dict()
            ),
            "current_binding": (
                None if self.current_binding is None else self.current_binding.to_dict()
            ),
            "current_effect": (
                None if self.current_effect is None else self.current_effect.to_dict()
            ),
            "current_policy_gate": (
                None
                if self.current_policy_gate is None
                else self.current_policy_gate.to_dict()
            ),
            "authorization_request": (
                None
                if self.authorization_request is None
                else self.authorization_request.to_dict()
            ),
            "require_execution_check": self.require_execution_check,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityRepairRequest":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("security repair request must be an object")
        _reject_forbidden_fields(payload, label="security repair request")
        return cls(
            security_ir=payload.get("security_ir") or {},
            role=payload.get("role", OperatorRole.POLICY_GATE),
            reviewed_binding=payload.get("reviewed_binding"),
            reviewed_effect=payload.get("reviewed_effect"),
            reviewed_policy_gate=payload.get("reviewed_policy_gate"),
            current_binding=payload.get("current_binding"),
            current_effect=payload.get("current_effect"),
            current_policy_gate=payload.get("current_policy_gate"),
            authorization_request=payload.get("authorization_request"),
            require_execution_check=payload.get("require_execution_check", True),
        )


@dataclass(frozen=True)
class SecurityRepairReceipt(CanonicalContract):
    """Non-authoritative preview/inverse receipt for one security repair."""

    SCHEMA: ClassVar[str] = SECURITY_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = SECURITY_REPAIR_OPERATORS_INTERFACE

    disposition: RepairDisposition
    role: OperatorRole
    operator_kind: str
    reason_codes: tuple[str, ...]
    security_ir_identity: str
    preview_security_ir: SecurityIR | None = None
    inverse_security_ir: SecurityIR | None = None
    preview_binding: AuthorizationBinding | None = None
    preview_effect: EffectAnnotation | None = None
    preview_policy_gate: PolicyGate | None = None
    inverse_binding: AuthorizationBinding | None = None
    inverse_effect: EffectAnnotation | None = None
    inverse_policy_gate: PolicyGate | None = None
    execution_decision: SecurityDecision | None = None
    execution_check_ok: bool = False
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    evidence_id: str = SECURITY_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairDisposition, "disposition"),
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes", required=True, ordered=True),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise SecurityRepairError("reason_codes exceeds its item bound")
        object.__setattr__(
            self,
            "security_ir_identity",
            _text(self.security_ir_identity, "security_ir_identity"),
        )
        object.__setattr__(
            self, "execution_check_ok", _bool(self.execution_check_ok, "execution_check_ok")
        )
        for flag in (
            "proposal_only",
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
        ):
            current = getattr(self, flag)
            if flag == "proposal_only":
                if current is not True:
                    raise SecurityRepairError("receipts must remain proposal-only")
                object.__setattr__(self, flag, True)
            else:
                if current is not False:
                    raise SecurityRepairError(
                        f"{flag} cannot be true on a repair receipt"
                    )
                object.__setattr__(self, flag, False)
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        if self.evidence_id != SECURITY_REPAIR_EVIDENCE:
            raise SecurityRepairError(
                f"evidence_id must be exactly {SECURITY_REPAIR_EVIDENCE}"
            )

    @property
    def is_editable(self) -> bool:
        return self.disposition is RepairDisposition.PREVIEW_READY

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "role": self.role.value,
            "operator_kind": self.operator_kind,
            "reason_codes": list(self.reason_codes),
            "security_ir_identity": self.security_ir_identity,
            "preview_security_ir": (
                None
                if self.preview_security_ir is None
                else self.preview_security_ir.to_dict()
            ),
            "inverse_security_ir": (
                None
                if self.inverse_security_ir is None
                else self.inverse_security_ir.to_dict()
            ),
            "preview_binding": (
                None if self.preview_binding is None else self.preview_binding.to_dict()
            ),
            "preview_effect": (
                None if self.preview_effect is None else self.preview_effect.to_dict()
            ),
            "preview_policy_gate": (
                None
                if self.preview_policy_gate is None
                else self.preview_policy_gate.to_dict()
            ),
            "inverse_binding": (
                None if self.inverse_binding is None else self.inverse_binding.to_dict()
            ),
            "inverse_effect": (
                None if self.inverse_effect is None else self.inverse_effect.to_dict()
            ),
            "inverse_policy_gate": (
                None
                if self.inverse_policy_gate is None
                else self.inverse_policy_gate.to_dict()
            ),
            "execution_decision": (
                None
                if self.execution_decision is None
                else self.execution_decision.to_dict()
            ),
            "execution_check_ok": self.execution_check_ok,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": SECURITY_REPAIR_VERSION,
            "profiles": list(MCP_PROFILES_C_D),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityRepairReceipt":
        if not isinstance(payload, Mapping):
            raise SecurityRepairError("security repair receipt must be an object")
        _reject_forbidden_fields(payload, label="security repair receipt")

        def _opt(mapping_key: str, factory):
            value = payload.get(mapping_key)
            if value is None:
                return None
            return factory(value)

        return cls(
            disposition=payload.get("disposition", RepairDisposition.REJECTED),
            role=payload.get("role", OperatorRole.POLICY_GATE),
            operator_kind=payload.get(
                "operator_kind", OperatorKind.REPAIR_AUTHORIZATION_GUARD.value
            ),
            reason_codes=payload.get("reason_codes") or ("rejected",),
            security_ir_identity=payload.get("security_ir_identity", ""),
            preview_security_ir=_opt("preview_security_ir", SecurityIR.from_dict),
            inverse_security_ir=_opt("inverse_security_ir", SecurityIR.from_dict),
            preview_binding=_opt("preview_binding", AuthorizationBinding.from_dict),
            preview_effect=_opt("preview_effect", EffectAnnotation.from_dict),
            preview_policy_gate=_opt("preview_policy_gate", PolicyGate.from_dict),
            inverse_binding=_opt("inverse_binding", AuthorizationBinding.from_dict),
            inverse_effect=_opt("inverse_effect", EffectAnnotation.from_dict),
            inverse_policy_gate=_opt("inverse_policy_gate", PolicyGate.from_dict),
            execution_decision=_opt("execution_decision", SecurityDecision.from_dict),
            execution_check_ok=payload.get("execution_check_ok", False),
            proposal_only=payload.get("proposal_only", True),
            grants_write_authority=payload.get("grants_write_authority", False),
            grants_proof_authority=payload.get("grants_proof_authority", False),
            semantic_authority=payload.get("semantic_authority", False),
            evidence_id=payload.get("evidence_id", SECURITY_REPAIR_EVIDENCE),
        )


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------


def _registry_descriptor():
    registry = build_default_operator_registry()
    return registry.require_known(OperatorKind.REPAIR_AUTHORIZATION_GUARD)


def _base_receipt(
    request: SecurityRepairRequest,
    *,
    disposition: RepairDisposition,
    role: OperatorRole,
    reasons: Sequence[str],
    preview_ir: SecurityIR | None = None,
    preview_binding: AuthorizationBinding | None = None,
    preview_effect: EffectAnnotation | None = None,
    preview_gate: PolicyGate | None = None,
    inverse_binding: AuthorizationBinding | None = None,
    inverse_effect: EffectAnnotation | None = None,
    inverse_gate: PolicyGate | None = None,
    decision: SecurityDecision | None = None,
    execution_ok: bool = False,
) -> SecurityRepairReceipt:
    return SecurityRepairReceipt(
        disposition=disposition,
        role=role,
        operator_kind=OperatorKind.REPAIR_AUTHORIZATION_GUARD.value,
        reason_codes=tuple(reasons) or (disposition.value,),
        security_ir_identity=request.security_ir.content_id,
        preview_security_ir=preview_ir,
        inverse_security_ir=request.security_ir,
        preview_binding=preview_binding,
        preview_effect=preview_effect,
        preview_policy_gate=preview_gate,
        inverse_binding=inverse_binding,
        inverse_effect=inverse_effect,
        inverse_policy_gate=inverse_gate,
        execution_decision=decision,
        execution_check_ok=execution_ok,
    )


def _guard_registry(role: OperatorRole, request: SecurityRepairRequest) -> SecurityRepairReceipt | None:
    descriptor = _registry_descriptor()
    if descriptor.kind is not OperatorKind.REPAIR_AUTHORIZATION_GUARD:
        return _base_receipt(
            request,
            disposition=RepairDisposition.REJECTED,
            role=role,
            reasons=("registry_kind_mismatch",),
        )
    if descriptor.family is not OperatorFamily.SECURITY:
        return _base_receipt(
            request,
            disposition=RepairDisposition.REJECTED,
            role=role,
            reasons=("registry_family_mismatch",),
        )
    if descriptor.proposal_only is not True or descriptor.grants_write_authority:
        return _base_receipt(
            request,
            disposition=RepairDisposition.REJECTED,
            role=role,
            reasons=("descriptor_authority_violation",),
        )
    if not request.security_ir.authority.authorizes_security_source:
        return _base_receipt(
            request,
            disposition=RepairDisposition.ABSTAIN,
            role=role,
            reasons=(
                "security_source_not_reviewed",
                f"authority:{request.security_ir.authority.value}",
                "conflict_policy_abstain",
            ),
        )
    return None


def _run_execution_check(
    security_ir: SecurityIR,
    request: SecurityRepairRequest,
) -> tuple[bool, SecurityDecision | None, tuple[str, ...]]:
    if not request.require_execution_check:
        return True, None, ("execution_check_skipped",)
    if request.authorization_request is None:
        return False, None, ("execution_check_request_required",)
    decision = evaluate_security_authorization(
        security_ir, request.authorization_request
    )
    if not decision.permitted:
        return False, decision, ("execution_check_denied", *decision.reason_codes)
    return True, decision, ("execution_check_ok", *decision.reason_codes)


class AuthorizationBindingOperator:
    """Restore reviewed principal/audience/capability bindings (Profile C).

    Cannot invent UCAN grants or authority; missing reviewed bindings abstain.
    """

    ROLE: ClassVar[OperatorRole] = OperatorRole.AUTHORIZATION_BINDING
    INTERFACE: ClassVar[str] = SECURITY_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor()

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        if not isinstance(request, SecurityRepairRequest):
            raise SecurityRepairError("request must be a SecurityRepairRequest")
        if request.role is not self.ROLE:
            request = SecurityRepairRequest(
                security_ir=request.security_ir,
                role=self.ROLE,
                reviewed_binding=request.reviewed_binding,
                reviewed_effect=request.reviewed_effect,
                reviewed_policy_gate=request.reviewed_policy_gate,
                current_binding=request.current_binding,
                current_effect=request.current_effect,
                current_policy_gate=request.current_policy_gate,
                authorization_request=request.authorization_request,
                require_execution_check=request.require_execution_check,
            )
        blocked = _guard_registry(self.ROLE, request)
        if blocked is not None:
            return blocked

        reviewed = request.reviewed_binding
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                reasons=(
                    "missing_reviewed_binding",
                    DenialReasonCode.INVENTED_AUTHORITY.value,
                    "conflict_policy_abstain",
                ),
            )
        # Binding must reference an existing reviewed grant — never invent one.
        grant = request.security_ir.grant_by_id(reviewed.grant_id)
        if grant is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                reasons=(
                    "binding_grant_missing",
                    DenialReasonCode.INVENTED_GRANT.value,
                    "conflict_policy_abstain",
                ),
            )
        if grant.issuer not in request.security_ir.trusted_issuers:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("grant_issuer_not_trusted",),
            )
        if (
            reviewed.audience != grant.audience
            or reviewed.capability != grant.capability
            or reviewed.resource != grant.resource
        ):
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("binding_grant_mismatch",),
            )

        current = request.current_binding
        if current is not None and current.content_id == reviewed.content_id:
            preview_ir = request.security_ir
            ok, decision, codes = _run_execution_check(preview_ir, request)
            if request.require_execution_check and not ok:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.DENIED,
                    role=self.ROLE,
                    reasons=("already_aligned_but_execution_denied", *codes),
                    preview_ir=preview_ir,
                    preview_binding=reviewed,
                    inverse_binding=current,
                    decision=decision,
                    execution_ok=False,
                )
            return _base_receipt(
                request,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                role=self.ROLE,
                reasons=("already_aligned", *codes),
                preview_ir=preview_ir,
                preview_binding=reviewed,
                inverse_binding=current,
                decision=decision,
                execution_ok=ok,
            )

        preview_ir = request.security_ir.with_binding(reviewed)
        ok, decision, codes = _run_execution_check(preview_ir, request)
        if request.require_execution_check and not ok:
            return _base_receipt(
                request,
                disposition=RepairDisposition.DENIED,
                role=self.ROLE,
                reasons=("execution_check_failed", *codes),
                preview_ir=None,
                preview_binding=None,
                inverse_binding=current,
                decision=decision,
                execution_ok=False,
            )
        return _base_receipt(
            request,
            disposition=RepairDisposition.PREVIEW_READY,
            role=self.ROLE,
            reasons=("preview_ready", "binding_restored", *codes),
            preview_ir=preview_ir,
            preview_binding=reviewed,
            inverse_binding=current,
            decision=decision,
            execution_ok=ok,
        )

    def preview(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: SecurityRepairReceipt) -> AuthorizationBinding | None:
        if not isinstance(receipt, SecurityRepairReceipt):
            raise SecurityRepairError("receipt must be a SecurityRepairReceipt")
        return receipt.inverse_binding


class EffectAnnotationOperator:
    """Restore reviewed effect classifications without inventing effect classes."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.EFFECT_ANNOTATION
    INTERFACE: ClassVar[str] = SECURITY_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor()

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        if not isinstance(request, SecurityRepairRequest):
            raise SecurityRepairError("request must be a SecurityRepairRequest")
        if request.role is not self.ROLE:
            request = SecurityRepairRequest(
                security_ir=request.security_ir,
                role=self.ROLE,
                reviewed_binding=request.reviewed_binding,
                reviewed_effect=request.reviewed_effect,
                reviewed_policy_gate=request.reviewed_policy_gate,
                current_binding=request.current_binding,
                current_effect=request.current_effect,
                current_policy_gate=request.current_policy_gate,
                authorization_request=request.authorization_request,
                require_execution_check=request.require_execution_check,
            )
        blocked = _guard_registry(self.ROLE, request)
        if blocked is not None:
            return blocked

        reviewed = request.reviewed_effect
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                reasons=(
                    "missing_reviewed_effect",
                    DenialReasonCode.INVENTED_EFFECT.value,
                    "conflict_policy_abstain",
                ),
            )
        if not reviewed.declared:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=("effect_not_declared", DenialReasonCode.UNDECLARED_EFFECT.value),
            )

        current = request.current_effect
        if current is not None and current.content_id == reviewed.content_id:
            preview_ir = request.security_ir
            ok, decision, codes = _run_execution_check(preview_ir, request)
            if request.require_execution_check and not ok:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.DENIED,
                    role=self.ROLE,
                    reasons=("already_aligned_but_execution_denied", *codes),
                    preview_ir=preview_ir,
                    preview_effect=reviewed,
                    inverse_effect=current,
                    decision=decision,
                    execution_ok=False,
                )
            return _base_receipt(
                request,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                role=self.ROLE,
                reasons=("already_aligned", *codes),
                preview_ir=preview_ir,
                preview_effect=reviewed,
                inverse_effect=current,
                decision=decision,
                execution_ok=ok,
            )

        preview_ir = request.security_ir.with_effect(reviewed)
        ok, decision, codes = _run_execution_check(preview_ir, request)
        if request.require_execution_check and not ok:
            return _base_receipt(
                request,
                disposition=RepairDisposition.DENIED,
                role=self.ROLE,
                reasons=("execution_check_failed", *codes),
                decision=decision,
                inverse_effect=current,
                execution_ok=False,
            )
        return _base_receipt(
            request,
            disposition=RepairDisposition.PREVIEW_READY,
            role=self.ROLE,
            reasons=("preview_ready", "effect_restored", *codes),
            preview_ir=preview_ir,
            preview_effect=reviewed,
            inverse_effect=current,
            decision=decision,
            execution_ok=ok,
        )

    def preview(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: SecurityRepairReceipt) -> EffectAnnotation | None:
        if not isinstance(receipt, SecurityRepairReceipt):
            raise SecurityRepairError("receipt must be a SecurityRepairReceipt")
        return receipt.inverse_effect


class PolicyGateOperator:
    """Restore fail-closed Profile D policy gates.

    Policy outage and missing decisions deny at evaluation time; this operator
    only restores reviewed gate bindings and never invents policy semantics.
    """

    ROLE: ClassVar[OperatorRole] = OperatorRole.POLICY_GATE
    INTERFACE: ClassVar[str] = SECURITY_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor()

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        if not isinstance(request, SecurityRepairRequest):
            raise SecurityRepairError("request must be a SecurityRepairRequest")
        if request.role is not self.ROLE:
            request = SecurityRepairRequest(
                security_ir=request.security_ir,
                role=self.ROLE,
                reviewed_binding=request.reviewed_binding,
                reviewed_effect=request.reviewed_effect,
                reviewed_policy_gate=request.reviewed_policy_gate,
                current_binding=request.current_binding,
                current_effect=request.current_effect,
                current_policy_gate=request.current_policy_gate,
                authorization_request=request.authorization_request,
                require_execution_check=request.require_execution_check,
            )
        blocked = _guard_registry(self.ROLE, request)
        if blocked is not None:
            return blocked

        reviewed = request.reviewed_policy_gate
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                reasons=(
                    "missing_reviewed_policy_gate",
                    DenialReasonCode.INVENTED_POLICY.value,
                    "conflict_policy_abstain",
                ),
            )
        # Operators restore reviewed gates but will not "fix" an outage into
        # an allow without a reviewed available decision.
        if reviewed.availability is PolicyAvailability.OUTAGE:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=(
                    DenialReasonCode.POLICY_OUTAGE.value,
                    "cannot_restore_outage_as_allow",
                ),
            )
        if reviewed.availability is not PolicyAvailability.AVAILABLE or not reviewed.has_decision:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                reasons=(
                    DenialReasonCode.MISSING_DECISION.value,
                    "reviewed_gate_missing_decision",
                ),
            )
        if reviewed.decision == "deny":
            # Restoring an explicit deny gate is valid (fail-closed enforcement).
            pass

        current = request.current_policy_gate
        if current is not None and current.content_id == reviewed.content_id:
            preview_ir = request.security_ir
            ok, decision, codes = _run_execution_check(preview_ir, request)
            # For deny gates, execution check is expected to deny — that is OK
            # for already-aligned fail-closed restoration.
            if (
                request.require_execution_check
                and not ok
                and reviewed.decision != "deny"
            ):
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.DENIED,
                    role=self.ROLE,
                    reasons=("already_aligned_but_execution_denied", *codes),
                    preview_ir=preview_ir,
                    preview_gate=reviewed,
                    inverse_gate=current,
                    decision=decision,
                    execution_ok=False,
                )
            aligned_ok = ok or reviewed.decision == "deny"
            return _base_receipt(
                request,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                role=self.ROLE,
                reasons=("already_aligned", *codes),
                preview_ir=preview_ir,
                preview_gate=reviewed,
                inverse_gate=current,
                decision=decision,
                execution_ok=aligned_ok,
            )

        preview_ir = request.security_ir.with_policy_gate(reviewed)
        ok, decision, codes = _run_execution_check(preview_ir, request)
        if (
            request.require_execution_check
            and not ok
            and reviewed.decision != "deny"
        ):
            return _base_receipt(
                request,
                disposition=RepairDisposition.DENIED,
                role=self.ROLE,
                reasons=("execution_check_failed", *codes),
                inverse_gate=current,
                decision=decision,
                execution_ok=False,
            )
        return _base_receipt(
            request,
            disposition=RepairDisposition.PREVIEW_READY,
            role=self.ROLE,
            reasons=("preview_ready", "policy_gate_restored", *codes),
            preview_ir=preview_ir,
            preview_gate=reviewed,
            inverse_gate=current,
            decision=decision,
            execution_ok=ok or reviewed.decision == "deny",
        )

    def preview(self, request: SecurityRepairRequest) -> SecurityRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: SecurityRepairReceipt) -> PolicyGate | None:
        if not isinstance(receipt, SecurityRepairReceipt):
            raise SecurityRepairError("receipt must be a SecurityRepairReceipt")
        return receipt.inverse_policy_gate


class SecurityRepairOperators:
    """Facade bundling the three DCR-046 security repair operators."""

    INTERFACE: ClassVar[str] = SECURITY_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = SECURITY_REPAIR_EVIDENCE

    def __init__(
        self,
        *,
        authorization_binding: AuthorizationBindingOperator | None = None,
        effect_annotation: EffectAnnotationOperator | None = None,
        policy_gate: PolicyGateOperator | None = None,
    ) -> None:
        self.authorization_binding = (
            authorization_binding or AuthorizationBindingOperator()
        )
        self.effect_annotation = effect_annotation or EffectAnnotationOperator()
        self.policy_gate = policy_gate or PolicyGateOperator()


def build_security_repair_operators() -> SecurityRepairOperators:
    """Construct the sealed DCR-046 security operator set."""

    return SecurityRepairOperators()


def materialize_security_operator_vectors(
    security_ir: SecurityIR,
    request: SecurityAuthorizationRequest,
) -> dict[str, Any]:
    """Emit compact deterministic vectors for acceptance evidence."""

    decision = evaluate_security_authorization(security_ir, request)
    return {
        "schema": SECURITY_OPERATOR_VECTORS_SCHEMA,
        "interface": SECURITY_REPAIR_OPERATORS_INTERFACE,
        "evidence_id": SECURITY_REPAIR_EVIDENCE,
        "security_ir_interface": SECURITY_IR_INTERFACE,
        "profiles": list(MCP_PROFILES_C_D),
        "security_ir_identity": security_ir.content_id,
        "request_identity": request.content_id,
        "decision": decision.to_dict(),
        "principal": request.principal,
        "audience": request.audience,
        "capability": request.capability,
        "revocation": any(g.revoked or g.revocation_id for g in security_ir.grants),
        "obligations": list(request.fulfilled_obligations),
        "temporal_validity": {
            "evaluated_at_ms": request.evaluated_at_ms,
            "grant_windows": [
                {
                    "grant_id": grant.grant_id,
                    "not_before_ms": grant.not_before_ms,
                    "expires_at_ms": grant.expires_at_ms,
                    "status": grant.status_at(
                        request.evaluated_at_ms, expected_audience=request.audience
                    ).value,
                }
                for grant in security_ir.grants
            ],
        },
        "execution_time_check": decision.reason_codes,
        "effects": list(request.expected_effects),
        "server_assertions_trusted": False,
        "ignored_server_assertions": decision.ignored_server_assertions,
    }


__all__ = (
    "SECURITY_REPAIR_OPERATORS_INTERFACE",
    "SECURITY_REPAIR_EVIDENCE",
    "SECURITY_IR_INTERFACE",
    "MCP_PROFILE_C",
    "MCP_PROFILE_D",
    "MCP_PROFILES_C_D",
    "SecurityRepairError",
    "SecurityRepairAbstention",
    "RepairDisposition",
    "OperatorRole",
    "SecurityVerdict",
    "DenialReasonCode",
    "GrantStatus",
    "PolicyAvailability",
    "EffectClass",
    "AuthoritySource",
    "SecurityGrant",
    "AuthorizationBinding",
    "EffectAnnotation",
    "PolicyGate",
    "SecurityIR",
    "SecurityAuthorizationRequest",
    "SecurityDecision",
    "SecurityRepairRequest",
    "SecurityRepairReceipt",
    "AuthorizationBindingOperator",
    "EffectAnnotationOperator",
    "PolicyGateOperator",
    "SecurityRepairOperators",
    "build_security_repair_operators",
    "evaluate_security_authorization",
    "materialize_security_operator_vectors",
)
