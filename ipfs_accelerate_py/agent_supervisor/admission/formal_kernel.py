"""FACP-039 — restricted Effect Admission Kernel (EAK) runtime.

Provider-free host kernel that:

* derives AdmissionToken obligations mechanically from OperationSpec@1 fields
  (FACP-038 §5);
* mints argument-bound tokens **only** as ``effect_admission_kernel``;
* verifies every declared obligation plus exact ``argument_cid`` before unlock;
* fails closed on expiry, revocation, nonce replay, and changed arguments; and
* compiles rich/unknown source policy conservatively to a decidable runtime IR
  whose unknown constructs may yield only denial, obligation, or typed
  indeterminate — never a permissive allow.

This module wraps existing authorization / permit vocabulary conceptually
(expiry, revocation, lease, nonce replay) without rewriting those primitives
and without invoking effect handlers or accepting browser/model/prompt tokens.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = "facp/admission-kernel@1"
TOKEN_SCHEMA: Final[str] = "facp/admission-token@1"
POLICY_IR_SCHEMA: Final[str] = "facp/admission-policy-ir@1"
DECISION_SCHEMA: Final[str] = "facp/admission-decision@1"
TASK_ID: Final[str] = "FACP-039"
GOAL_ID: Final[str] = "FACP-G320"
BUNDLE: Final[str] = "facp/admission/kernel"
KERNEL_ISSUER: Final[str] = "effect_admission_kernel"
KERNEL_VERSION: Final[str] = "effect-admission-kernel/v1"

EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "actor",
    "device",
    "tenant",
    "resource",
    "operation",
    "argument",
    "contract",
    "delegation",
    "policy",
    "confirmation",
    "lease",
    "expiry",
    "nonce",
    "signature",
    "revocation",
)

FORBIDDEN_TOKEN_ISSUERS: Final[frozenset[str]] = frozenset(
    {
        "browser",
        "browser_consent",
        "prompt",
        "model",
        "peer",
        "payment",
        "caller",
        "tenant",
        "ui",
        "consent",
        "dry_run",
        "allow",
    }
)

FREE_FORM_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "authorization",
        "allowed",
        "consent",
        "dry_run",
        "permission",
        "grant",
        "success",
        "outcome",
    }
)

EFFECT_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "pure",
        "read",
        "write",
        "process",
        "credential",
        "install",
        "repository",
        "publish",
        "payment",
        "private",
        "legal",
        "irreversible",
    }
)

AUTHORITY_OBLIGATIONS: Final[frozenset[str]] = frozenset(
    {"none", "actor_authenticated", "capability_verified"}
)
POLICY_OBLIGATIONS: Final[frozenset[str]] = frozenset(
    {"none", "host_policy_required", "host_policy_with_obligations"}
)
CONFIRMATION_OBLIGATIONS: Final[frozenset[str]] = frozenset(
    {"none", "one_use_confirmation_required"}
)
LEASE_OBLIGATIONS: Final[frozenset[str]] = frozenset({"none", "lease_required"})
OBSERVATION_OBLIGATIONS: Final[frozenset[str]] = frozenset(
    {
        "none",
        "independent_observation_required",
        "delegated_observation_allowed",
    }
)

UNIVERSAL_TOKEN_OBLIGATIONS: Final[frozenset[str]] = frozenset(
    {
        "kernel_issued",
        "operation_bound",
        "effect_class_bound",
        "argument_bound",
        "nonce_bound",
        "expiry_bound",
    }
)

TOKEN_OBLIGATION_CONSTRUCTORS: Final[frozenset[str]] = frozenset(
    {
        *UNIVERSAL_TOKEN_OBLIGATIONS,
        "actor_bound",
        "capability_bound",
        "delegation_bound",
        "policy_bound",
        "policy_obligations_bound",
        "confirmation_bound",
        "lease_bound",
        "observation_bound",
    }
)

HANDLER_UNLOCK_TYPESTATES: Final[frozenset[str]] = frozenset({"Reserved", "Started"})

TYPESTATE_HAPPY_PATH: Final[tuple[str, ...]] = (
    "Proposed",
    "ContractResolved",
    "ActorAuthenticated",
    "CapabilityVerified",
    "PolicyEvaluated",
    "ObligationsSatisfied",
    "ConfirmationSatisfied",
    "LeaseHeld",
    "Reserved",
    "Started",
    "Observed",
    "ReceiptSealed",
)

TYPESTATE_EXCEPTIONAL: Final[frozenset[str]] = frozenset(
    {
        "Rejected",
        "Unavailable",
        "Failed",
        "Unknown",
        "CompensationRequired",
        "Compensated",
        "Aborted",
    }
)

# Known, decidable source-policy clause kinds the conservative compiler accepts.
_KNOWN_CLAUSE_KINDS: Final[frozenset[str]] = frozenset(
    {"permission", "prohibition", "obligation"}
)
_KNOWN_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "policy_id",
        "clauses",
        "name",
        "version",
        "description",
    }
)
_KNOWN_CLAUSE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "clause_type",
        "kind",
        "actor",
        "action",
        "resource",
        "valid_from",
        "valid_until",
        "obligation_deadline",
        "deadline",
        "effect",
        "metadata",
    }
)


class AdmissionErrorCode(str, Enum):
    """Stable fail-closed codes for the restricted admission kernel."""

    UNKNOWN_ENUM = "UNKNOWN_ENUM"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_TYPE = "INVALID_TYPE"
    FORBIDDEN_FLOAT = "FORBIDDEN_FLOAT"
    FREE_FORM_AUTHORITY = "FREE_FORM_AUTHORITY"
    TOKEN_OBLIGATION_MISMATCH = "TOKEN_OBLIGATION_MISMATCH"
    HANDLER_NOT_UNLOCKED = "HANDLER_NOT_UNLOCKED"
    NON_KERNEL_TOKEN_ISSUER = "NON_KERNEL_TOKEN_ISSUER"
    PURE_TOKEN_FORBIDDEN = "PURE_TOKEN_FORBIDDEN"
    EXPIRED_TOKEN = "EXPIRED_TOKEN"
    REVOKED_TOKEN = "REVOKED_TOKEN"
    REPLAYED_TOKEN = "REPLAYED_TOKEN"
    ARGUMENT_MISMATCH = "ARGUMENT_MISMATCH"
    OPERATION_MISMATCH = "OPERATION_MISMATCH"
    BINDING_MISMATCH = "BINDING_MISMATCH"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    ILLEGAL_TYPESTATE_TRANSITION = "ILLEGAL_TYPESTATE_TRANSITION"
    BLIND_UNKNOWN_REPLAY = "BLIND_UNKNOWN_REPLAY"
    POLICY_INDETERMINATE = "POLICY_INDETERMINATE"
    POLICY_DENIED = "POLICY_DENIED"
    UNSATISFIED_OBLIGATION = "UNSATISFIED_OBLIGATION"
    FORBIDDEN_ISSUER = "FORBIDDEN_ISSUER"
    NOT_YET_VALID = "NOT_YET_VALID"
    INVALID_TOKEN = "INVALID_TOKEN"


class PolicyIRVerdict(str, Enum):
    """Closed runtime IR verdicts for compiled source policy."""

    ALLOW = "allow"
    DENY = "deny"
    OBLIGATION = "obligation"
    INDETERMINATE = "indeterminate"


class AdmissionVerdict(str, Enum):
    ADMIT = "admit"
    DENY = "deny"
    INDETERMINATE = "indeterminate"


class AdmissionError(ValueError):
    """Malformed or fail-closed admission input."""

    def __init__(self, code: AdmissionErrorCode | str, message: str) -> None:
        self.code = AdmissionErrorCode(code)
        super().__init__(message)


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise AdmissionError(
            AdmissionErrorCode.FORBIDDEN_FLOAT,
            f"floats are forbidden at {path}",
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_floats(item, f"{path}[{index}]")


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise AdmissionError(
            AdmissionErrorCode.INVALID_TYPE, f"{name} must be a string"
        )
    if "\x00" in value or value != value.strip():
        raise AdmissionError(
            AdmissionErrorCode.INVALID_TYPE,
            f"{name} must not contain NUL or surrounding whitespace",
        )
    if required and not value:
        raise AdmissionError(
            AdmissionErrorCode.MISSING_FIELD, f"{name} is required"
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdmissionError(
            AdmissionErrorCode.INVALID_TYPE,
            f"{name} must be a non-negative integer",
        )
    return value


def _closed_enum(value: Any, allowed: frozenset[str], name: str) -> str:
    text = _text(value, name)
    if text not in allowed:
        raise AdmissionError(
            AdmissionErrorCode.UNKNOWN_ENUM,
            f"unknown {name}={text!r}",
        )
    return text


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(sorted(ordered))


# ---------------------------------------------------------------------------
# OperationSpec projection + mechanical obligation derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationSpecView:
    """Closed OperationSpec@1 fields required by the admission kernel."""

    operation_id: str
    effect_class: str
    authority_obligation: str = "none"
    policy_obligation: str = "none"
    confirmation_obligation: str = "none"
    lease_obligation: str = "none"
    observation_obligation: str = "none"
    idempotency_class: str = "at_most_once"
    reversibility_class: str = "reversible"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self,
            "effect_class",
            _closed_enum(self.effect_class, EFFECT_CLASSES, "effect_class"),
        )
        object.__setattr__(
            self,
            "authority_obligation",
            _closed_enum(
                self.authority_obligation,
                AUTHORITY_OBLIGATIONS,
                "authority_obligation",
            ),
        )
        object.__setattr__(
            self,
            "policy_obligation",
            _closed_enum(
                self.policy_obligation, POLICY_OBLIGATIONS, "policy_obligation"
            ),
        )
        object.__setattr__(
            self,
            "confirmation_obligation",
            _closed_enum(
                self.confirmation_obligation,
                CONFIRMATION_OBLIGATIONS,
                "confirmation_obligation",
            ),
        )
        object.__setattr__(
            self,
            "lease_obligation",
            _closed_enum(
                self.lease_obligation, LEASE_OBLIGATIONS, "lease_obligation"
            ),
        )
        object.__setattr__(
            self,
            "observation_obligation",
            _closed_enum(
                self.observation_obligation,
                OBSERVATION_OBLIGATIONS,
                "observation_obligation",
            ),
        )
        object.__setattr__(
            self,
            "idempotency_class",
            _text(self.idempotency_class, "idempotency_class"),
        )
        object.__setattr__(
            self,
            "reversibility_class",
            _text(self.reversibility_class, "reversibility_class"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "effect_class": self.effect_class,
            "authority_obligation": self.authority_obligation,
            "policy_obligation": self.policy_obligation,
            "confirmation_obligation": self.confirmation_obligation,
            "lease_obligation": self.lease_obligation,
            "observation_obligation": self.observation_obligation,
            "idempotency_class": self.idempotency_class,
            "reversibility_class": self.reversibility_class,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OperationSpecView":
        _reject_floats(raw)
        unknown = set(raw) - {
            "schema",
            "schema_version",
            "operation_id",
            "effect_class",
            "authority_obligation",
            "policy_obligation",
            "confirmation_obligation",
            "lease_obligation",
            "observation_obligation",
            "idempotency_class",
            "reversibility_class",
            "namespace",
            "name",
            "version",
            "input_schema_cid",
            "output_schema_cid",
            "error_codes",
            "evidence_class",
            "allowed_outcomes",
            "resource_bounds",
        }
        free = unknown & FREE_FORM_AUTHORITY_KEYS
        if free:
            raise AdmissionError(
                AdmissionErrorCode.FREE_FORM_AUTHORITY,
                f"free-form authority keys forbidden: {sorted(free)}",
            )
        if unknown:
            raise AdmissionError(
                AdmissionErrorCode.UNKNOWN_FIELD,
                f"unknown OperationSpec fields: {sorted(unknown)}",
            )
        return cls(
            operation_id=str(raw["operation_id"]),
            effect_class=str(raw["effect_class"]),
            authority_obligation=str(raw.get("authority_obligation", "none")),
            policy_obligation=str(raw.get("policy_obligation", "none")),
            confirmation_obligation=str(
                raw.get("confirmation_obligation", "none")
            ),
            lease_obligation=str(raw.get("lease_obligation", "none")),
            observation_obligation=str(
                raw.get("observation_obligation", "none")
            ),
            idempotency_class=str(raw.get("idempotency_class", "at_most_once")),
            reversibility_class=str(
                raw.get("reversibility_class", "reversible")
            ),
        )


def derive_token_obligations(
    spec: OperationSpecView | Mapping[str, Any],
) -> frozenset[str]:
    """Pure mechanical derivation of AdmissionToken obligations (FACP-038 §5)."""
    view = (
        spec
        if isinstance(spec, OperationSpecView)
        else OperationSpecView.from_mapping(spec)
    )
    if view.effect_class == "pure":
        return frozenset()

    out: set[str] = set(UNIVERSAL_TOKEN_OBLIGATIONS)
    if view.authority_obligation == "actor_authenticated":
        out.add("actor_bound")
    elif view.authority_obligation == "capability_verified":
        out.update({"actor_bound", "capability_bound", "delegation_bound"})

    if view.policy_obligation == "host_policy_required":
        out.add("policy_bound")
    elif view.policy_obligation == "host_policy_with_obligations":
        out.update({"policy_bound", "policy_obligations_bound"})

    if view.confirmation_obligation == "one_use_confirmation_required":
        out.add("confirmation_bound")
    if view.lease_obligation == "lease_required":
        out.add("lease_bound")
    if view.observation_obligation != "none":
        out.add("observation_bound")
    return frozenset(out)


# ---------------------------------------------------------------------------
# Evidence bindings (FACP-039 evidence subset)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionBindings:
    """Opaque content identities required by declared token obligations."""

    actor_cid: str = ""
    device_cid: str = ""
    tenant_cid: str = ""
    resource_cid: str = ""
    operation_id: str = ""
    argument_cid: str = ""
    contract_cid: str = ""
    delegation_cid: str = ""
    policy_cid: str = ""
    confirmation_cid: str = ""
    lease_id: str = ""
    not_before: int = 0
    not_after: int = 0
    nonce: str = ""
    signature_cid: str = ""
    revocation_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "actor_cid",
            "device_cid",
            "tenant_cid",
            "resource_cid",
            "operation_id",
            "argument_cid",
            "contract_cid",
            "delegation_cid",
            "policy_cid",
            "confirmation_cid",
            "lease_id",
            "nonce",
            "signature_cid",
            "revocation_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "not_before", _nonneg_int(self.not_before, "not_before"))
        object.__setattr__(self, "not_after", _nonneg_int(self.not_after, "not_after"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_cid": self.actor_cid,
            "device_cid": self.device_cid,
            "tenant_cid": self.tenant_cid,
            "resource_cid": self.resource_cid,
            "operation_id": self.operation_id,
            "argument_cid": self.argument_cid,
            "contract_cid": self.contract_cid,
            "delegation_cid": self.delegation_cid,
            "policy_cid": self.policy_cid,
            "confirmation_cid": self.confirmation_cid,
            "lease_id": self.lease_id,
            "not_before": self.not_before,
            "not_after": self.not_after,
            "nonce": self.nonce,
            "signature_cid": self.signature_cid,
            "revocation_id": self.revocation_id,
        }

    def evidence_present(self) -> frozenset[str]:
        present: set[str] = set()
        mapping = {
            "actor": self.actor_cid,
            "device": self.device_cid,
            "tenant": self.tenant_cid,
            "resource": self.resource_cid,
            "operation": self.operation_id,
            "argument": self.argument_cid,
            "contract": self.contract_cid,
            "delegation": self.delegation_cid,
            "policy": self.policy_cid,
            "confirmation": self.confirmation_cid,
            "lease": self.lease_id,
            "expiry": self.not_after > 0,
            "nonce": self.nonce,
            "signature": self.signature_cid,
            "revocation": self.revocation_id,
        }
        for key, value in mapping.items():
            if value:
                present.add(key)
        return frozenset(present)


def _required_bindings_for(obligations: frozenset[str]) -> frozenset[str]:
    """Map token obligations to evidence-subset binding names that must be set."""
    required: set[str] = set()
    if "operation_bound" in obligations:
        required.add("operation")
    if "argument_bound" in obligations:
        required.add("argument")
    if "nonce_bound" in obligations:
        required.add("nonce")
    if "expiry_bound" in obligations:
        required.add("expiry")
    if "actor_bound" in obligations:
        required.add("actor")
    if "capability_bound" in obligations or "delegation_bound" in obligations:
        required.update({"delegation", "actor"})
    if "policy_bound" in obligations or "policy_obligations_bound" in obligations:
        required.add("policy")
    if "confirmation_bound" in obligations:
        required.add("confirmation")
    if "lease_bound" in obligations:
        required.add("lease")
    # Universal host token always carries contract + signature material when
    # effectful; device/tenant/resource are recorded when supplied.
    if obligations:
        required.update({"contract", "signature"})
    return frozenset(required)


# ---------------------------------------------------------------------------
# Admission token
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionToken:
    """Kernel-issued, argument-bound admission token."""

    operation_id: str
    effect_class: str
    argument_cid: str
    actor_cid: str
    nonce: str
    not_after: int
    satisfied_obligations: tuple[str, ...]
    issuer: str = KERNEL_ISSUER
    schema: str = TOKEN_SCHEMA
    schema_version: int = 1
    not_before: int = 0
    token_id: str = ""
    bindings: Mapping[str, Any] = field(default_factory=dict)
    derived_obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self,
            "effect_class",
            _closed_enum(self.effect_class, EFFECT_CLASSES, "effect_class"),
        )
        object.__setattr__(
            self, "argument_cid", _text(self.argument_cid, "argument_cid")
        )
        object.__setattr__(self, "actor_cid", _text(self.actor_cid, "actor_cid"))
        object.__setattr__(self, "nonce", _text(self.nonce, "nonce"))
        object.__setattr__(self, "not_after", _nonneg_int(self.not_after, "not_after"))
        object.__setattr__(
            self, "not_before", _nonneg_int(self.not_before, "not_before")
        )
        object.__setattr__(self, "issuer", _text(self.issuer, "issuer"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise AdmissionError(
                AdmissionErrorCode.INVALID_TYPE, "schema_version must be int"
            )
        obligations = _sorted_unique(
            _closed_enum(item, TOKEN_OBLIGATION_CONSTRUCTORS, "obligation")
            for item in self.satisfied_obligations
        )
        object.__setattr__(self, "satisfied_obligations", obligations)
        derived = _sorted_unique(
            _closed_enum(item, TOKEN_OBLIGATION_CONSTRUCTORS, "derived_obligation")
            for item in self.derived_obligations
        )
        object.__setattr__(self, "derived_obligations", derived)
        object.__setattr__(
            self, "bindings", MappingProxyType(dict(self.bindings))
        )
        if not self.token_id:
            object.__setattr__(self, "token_id", self.content_id)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "issuer": self.issuer,
            "operation_id": self.operation_id,
            "effect_class": self.effect_class,
            "argument_cid": self.argument_cid,
            "actor_cid": self.actor_cid,
            "nonce": self.nonce,
            "not_before": self.not_before,
            "not_after": self.not_after,
            "satisfied_obligations": list(self.satisfied_obligations),
            "derived_obligations": list(self.derived_obligations),
            "bindings": dict(self.bindings),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["token_id"] = self.token_id or self.content_id
        return payload

    def to_canonical_token(self) -> dict[str, Any]:
        """Projection to the closed AdmissionToken@1 CCC field set."""
        return {
            "schema": TOKEN_SCHEMA,
            "schema_version": 1,
            "operation_id": self.operation_id,
            "actor_cid": self.actor_cid,
            "argument_cid": self.argument_cid,
            "nonce": self.nonce,
            "not_after": self.not_after,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AdmissionToken":
        _reject_floats(raw)
        return cls(
            operation_id=str(raw["operation_id"]),
            effect_class=str(raw.get("effect_class", "read")),
            argument_cid=str(raw["argument_cid"]),
            actor_cid=str(raw["actor_cid"]),
            nonce=str(raw["nonce"]),
            not_after=int(raw["not_after"]),
            satisfied_obligations=tuple(raw.get("satisfied_obligations") or ()),
            issuer=str(raw.get("issuer", KERNEL_ISSUER)),
            schema=str(raw.get("schema", TOKEN_SCHEMA)),
            schema_version=int(raw.get("schema_version", 1)),
            not_before=int(raw.get("not_before", 0)),
            token_id=str(raw.get("token_id") or ""),
            bindings=dict(raw.get("bindings") or {}),
            derived_obligations=tuple(raw.get("derived_obligations") or ()),
        )


@dataclass(frozen=True)
class AdmissionDecision:
    """Result of verify / unlock evaluation."""

    verdict: AdmissionVerdict
    code: AdmissionErrorCode | None
    message: str
    token_id: str = ""
    unlocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "schema_version": 1,
            "verdict": self.verdict.value,
            "code": None if self.code is None else self.code.value,
            "message": self.message,
            "token_id": self.token_id,
            "unlocked": self.unlocked,
        }


# ---------------------------------------------------------------------------
# Conservative source-policy compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledPolicyClause:
    kind: str
    actor: str = "*"
    action: str = "*"
    resource: str = "*"
    valid_from: int | None = None
    valid_until: int | None = None
    obligation_deadline: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "obligation_deadline": self.obligation_deadline,
        }


@dataclass(frozen=True)
class RuntimePolicyIR:
    """Decidable, default-deny runtime IR for host policy evaluation."""

    verdict: PolicyIRVerdict
    policy_cid: str
    clauses: tuple[CompiledPolicyClause, ...] = ()
    obligations: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    unknown_constructs: tuple[str, ...] = ()
    fully_translated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_IR_SCHEMA,
            "schema_version": 1,
            "verdict": self.verdict.value,
            "policy_cid": self.policy_cid,
            "clauses": [clause.to_dict() for clause in self.clauses],
            "obligations": list(self.obligations),
            "reasons": list(self.reasons),
            "unknown_constructs": list(self.unknown_constructs),
            "fully_translated": self.fully_translated,
        }


def _policy_cid(source: Mapping[str, Any] | None) -> str:
    if source is None:
        return content_identity({"policy": "empty"})
    claimed = source.get("policy_cid") or source.get("cid")
    if isinstance(claimed, str) and claimed.strip():
        return claimed.strip()
    return content_identity(dict(source))


def compile_source_policy(
    source: Mapping[str, Any] | None,
    *,
    actor: str = "*",
    action: str = "*",
    resource: str = "*",
    now_ms: int = 0,
) -> RuntimePolicyIR:
    """Conservatively compile rich/unknown source policy to runtime IR.

    Unknown or untranslatable constructs never become ``allow``. They compile
    only to ``deny``, ``obligation``, or typed ``indeterminate``.
    """
    if source is None:
        return RuntimePolicyIR(
            verdict=PolicyIRVerdict.DENY,
            policy_cid=_policy_cid(None),
            reasons=("missing_source_policy",),
            fully_translated=True,
        )

    if not isinstance(source, Mapping):
        raise AdmissionError(
            AdmissionErrorCode.INVALID_TYPE, "source policy must be a mapping"
        )
    _reject_floats(source)

    unknown_top = sorted(set(source) - _KNOWN_SOURCE_KEYS - {"policy_cid", "cid"})
    free_form = sorted(set(source) & FREE_FORM_AUTHORITY_KEYS)
    unknown_constructs: list[str] = []
    reasons: list[str] = []

    if free_form:
        unknown_constructs.extend(f"free_form:{key}" for key in free_form)
        reasons.append("free_form_authority")
    if unknown_top:
        unknown_constructs.extend(f"unknown_field:{key}" for key in unknown_top)
        reasons.append("unknown_source_fields")

    raw_clauses = source.get("clauses", ())
    if raw_clauses is None:
        raw_clauses = ()
    if not isinstance(raw_clauses, Sequence) or isinstance(
        raw_clauses, (str, bytes, bytearray)
    ):
        return RuntimePolicyIR(
            verdict=PolicyIRVerdict.INDETERMINATE,
            policy_cid=_policy_cid(source),
            reasons=("clauses_not_sequence",),
            unknown_constructs=tuple(unknown_constructs + ["clauses_type"]),
            fully_translated=False,
        )

    compiled: list[CompiledPolicyClause] = []
    obligations: list[str] = []
    matched_permission = False
    matched_prohibition = False
    translation_failed = bool(unknown_constructs)

    for index, clause in enumerate(raw_clauses):
        if not isinstance(clause, Mapping):
            translation_failed = True
            unknown_constructs.append(f"clause[{index}]:non_object")
            continue
        clause_unknown = sorted(set(clause) - _KNOWN_CLAUSE_KEYS)
        if clause_unknown:
            translation_failed = True
            unknown_constructs.extend(
                f"clause[{index}].{key}" for key in clause_unknown
            )
        kind = str(clause.get("clause_type") or clause.get("kind") or "")
        if kind not in _KNOWN_CLAUSE_KINDS:
            translation_failed = True
            unknown_constructs.append(f"clause[{index}]:unknown_kind:{kind or 'missing'}")
            continue

        def _opt_int(field_name: str) -> int | None:
            value = clause.get(field_name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdmissionError(
                    AdmissionErrorCode.INVALID_TYPE,
                    f"clause {field_name} must be int or null",
                )
            return value

        try:
            valid_from = _opt_int("valid_from")
            valid_until = _opt_int("valid_until")
            deadline = clause.get("obligation_deadline", clause.get("deadline"))
            if deadline is None:
                obligation_deadline = None
            elif isinstance(deadline, bool) or not isinstance(deadline, int):
                raise AdmissionError(
                    AdmissionErrorCode.INVALID_TYPE,
                    "obligation_deadline must be int or null",
                )
            else:
                obligation_deadline = deadline
        except AdmissionError:
            translation_failed = True
            unknown_constructs.append(f"clause[{index}]:invalid_temporal")
            continue

        actor_pat = str(clause.get("actor", "*") or "*")
        action_pat = str(clause.get("action", "*") or "*")
        resource_pat = str(clause.get("resource", "*") or "*")
        compiled.append(
            CompiledPolicyClause(
                kind=kind,
                actor=actor_pat,
                action=action_pat,
                resource=resource_pat,
                valid_from=valid_from,
                valid_until=valid_until,
                obligation_deadline=obligation_deadline,
            )
        )

        def _matches() -> bool:
            if valid_from is not None and now_ms < valid_from:
                return False
            if valid_until is not None and now_ms > valid_until:
                return False
            if actor_pat not in ("*", actor):
                return False
            if action_pat not in ("*", action):
                return False
            if resource_pat not in ("*", resource):
                return False
            return True

        if not _matches():
            continue
        if kind == "prohibition":
            matched_prohibition = True
        elif kind == "permission":
            matched_permission = True
        elif kind == "obligation":
            obligations.append(
                f"obligation:{action_pat}:{obligation_deadline or 0}"
            )

    # Unknown constructs: never allow.
    if translation_failed:
        if obligations and not matched_prohibition:
            verdict = PolicyIRVerdict.OBLIGATION
            reasons.append("unknown_constructs_with_obligation")
        elif matched_prohibition or not matched_permission:
            verdict = PolicyIRVerdict.DENY
            reasons.append("unknown_constructs_default_deny")
        else:
            verdict = PolicyIRVerdict.INDETERMINATE
            reasons.append("unknown_constructs_typed_indeterminate")
        return RuntimePolicyIR(
            verdict=verdict,
            policy_cid=_policy_cid(source),
            clauses=tuple(compiled),
            obligations=_sorted_unique(obligations),
            reasons=_sorted_unique(reasons),
            unknown_constructs=_sorted_unique(unknown_constructs),
            fully_translated=False,
        )

    if matched_prohibition:
        return RuntimePolicyIR(
            verdict=PolicyIRVerdict.DENY,
            policy_cid=_policy_cid(source),
            clauses=tuple(compiled),
            obligations=_sorted_unique(obligations),
            reasons=("prohibition_matched",),
            fully_translated=True,
        )
    if obligations and matched_permission:
        return RuntimePolicyIR(
            verdict=PolicyIRVerdict.OBLIGATION,
            policy_cid=_policy_cid(source),
            clauses=tuple(compiled),
            obligations=_sorted_unique(obligations),
            reasons=("permission_with_obligations",),
            fully_translated=True,
        )
    if matched_permission:
        return RuntimePolicyIR(
            verdict=PolicyIRVerdict.ALLOW,
            policy_cid=_policy_cid(source),
            clauses=tuple(compiled),
            reasons=("permission_matched",),
            fully_translated=True,
        )
    return RuntimePolicyIR(
        verdict=PolicyIRVerdict.DENY,
        policy_cid=_policy_cid(source),
        clauses=tuple(compiled),
        obligations=_sorted_unique(obligations),
        reasons=("default_deny_no_permission",),
        fully_translated=True,
    )


# ---------------------------------------------------------------------------
# Restricted Effect Admission Kernel
# ---------------------------------------------------------------------------


@dataclass
class EffectAdmissionKernel:
    """Host-only mint/verify/unlock boundary for effectful operations."""

    clock_ms: Callable[[], int] = field(default_factory=lambda: (lambda: 0))
    _used_nonces: set[str] = field(default_factory=set, init=False, repr=False)
    _revoked_token_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _issued_tokens: dict[str, AdmissionToken] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.clock_ms):
            raise AdmissionError(
                AdmissionErrorCode.INVALID_TYPE, "clock_ms must be callable"
            )

    def now_ms(self) -> int:
        value = self.clock_ms()
        return _nonneg_int(value, "clock_ms()")

    def derive_obligations(
        self, spec: OperationSpecView | Mapping[str, Any]
    ) -> frozenset[str]:
        return derive_token_obligations(spec)

    def compile_policy(
        self,
        source: Mapping[str, Any] | None,
        *,
        actor: str = "*",
        action: str = "*",
        resource: str = "*",
        now_ms: int | None = None,
    ) -> RuntimePolicyIR:
        return compile_source_policy(
            source,
            actor=actor,
            action=action,
            resource=resource,
            now_ms=self.now_ms() if now_ms is None else now_ms,
        )

    def mint_token(
        self,
        spec: OperationSpecView | Mapping[str, Any],
        bindings: AdmissionBindings | Mapping[str, Any],
        *,
        issuer: str = KERNEL_ISSUER,
        satisfied_obligations: Sequence[str] | None = None,
        source_policy: Mapping[str, Any] | None = None,
        consume_nonce: bool = False,
    ) -> AdmissionToken:
        """Mint a kernel AdmissionToken after all derived obligations are met.

        Non-kernel issuers (browser/model/prompt/peer/payment/...) are rejected.
        Pure operations never receive a host unlock token.
        """
        view = (
            spec
            if isinstance(spec, OperationSpecView)
            else OperationSpecView.from_mapping(spec)
        )
        if isinstance(bindings, Mapping):
            bindings = AdmissionBindings(**{
                key: bindings[key]
                for key in AdmissionBindings.__dataclass_fields__
                if key in bindings
            })
        if not isinstance(bindings, AdmissionBindings):
            raise AdmissionError(
                AdmissionErrorCode.INVALID_TYPE,
                "bindings must be AdmissionBindings or mapping",
            )

        issuer_text = _text(issuer, "issuer")
        if issuer_text in FORBIDDEN_TOKEN_ISSUERS or issuer_text != KERNEL_ISSUER:
            raise AdmissionError(
                AdmissionErrorCode.FORBIDDEN_ISSUER
                if issuer_text in FORBIDDEN_TOKEN_ISSUERS
                else AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER,
                f"only {KERNEL_ISSUER} may mint admission tokens; got {issuer_text!r}",
            )

        if view.effect_class == "pure":
            raise AdmissionError(
                AdmissionErrorCode.PURE_TOKEN_FORBIDDEN,
                "pure effect_class must not mint host AdmissionTokens",
            )

        derived = derive_token_obligations(view)
        if satisfied_obligations is None:
            have = set(derived)
        else:
            have = {
                _closed_enum(item, TOKEN_OBLIGATION_CONSTRUCTORS, "obligation")
                for item in satisfied_obligations
            }
        if not derived.issubset(have):
            missing = sorted(derived - have)
            raise AdmissionError(
                AdmissionErrorCode.TOKEN_OBLIGATION_MISMATCH,
                f"missing satisfied obligations: {missing}",
            )

        required = _required_bindings_for(derived)
        present = bindings.evidence_present()
        missing_evidence = sorted(required - present)
        if missing_evidence:
            raise AdmissionError(
                AdmissionErrorCode.MISSING_EVIDENCE,
                f"missing evidence bindings: {missing_evidence}",
            )

        if bindings.operation_id and bindings.operation_id != view.operation_id:
            raise AdmissionError(
                AdmissionErrorCode.OPERATION_MISMATCH,
                "bindings.operation_id must match OperationSpec.operation_id",
            )
        if not bindings.argument_cid:
            raise AdmissionError(
                AdmissionErrorCode.MISSING_FIELD, "argument_cid is required"
            )
        if bindings.not_after <= 0:
            raise AdmissionError(
                AdmissionErrorCode.MISSING_FIELD, "not_after expiry is required"
            )
        now = self.now_ms()
        if bindings.not_after <= now:
            raise AdmissionError(
                AdmissionErrorCode.EXPIRED_TOKEN,
                "cannot mint an already-expired token",
            )
        if bindings.not_before and bindings.not_before > now:
            raise AdmissionError(
                AdmissionErrorCode.NOT_YET_VALID,
                "token not_before is in the future",
            )

        # Policy obligation path: compile source policy fail-closed.
        if "policy_bound" in derived or "policy_obligations_bound" in derived:
            ir = self.compile_policy(
                source_policy,
                actor=bindings.actor_cid or "*",
                action=view.operation_id,
                resource=bindings.resource_cid or "*",
            )
            if ir.verdict is PolicyIRVerdict.DENY:
                raise AdmissionError(
                    AdmissionErrorCode.POLICY_DENIED,
                    f"source policy denied: {list(ir.reasons)}",
                )
            if ir.verdict is PolicyIRVerdict.INDETERMINATE:
                raise AdmissionError(
                    AdmissionErrorCode.POLICY_INDETERMINATE,
                    f"source policy indeterminate: {list(ir.unknown_constructs)}",
                )
            if (
                "policy_obligations_bound" in derived
                and ir.verdict is PolicyIRVerdict.ALLOW
                and not ir.obligations
            ):
                # host_policy_with_obligations requires an obligation-bearing IR.
                raise AdmissionError(
                    AdmissionErrorCode.UNSATISFIED_OBLIGATION,
                    "policy_obligations_bound requires obligation-bearing policy IR",
                )
            if ir.policy_cid and bindings.policy_cid and ir.policy_cid != bindings.policy_cid:
                raise AdmissionError(
                    AdmissionErrorCode.BINDING_MISMATCH,
                    "compiled policy_cid does not match bindings.policy_cid",
                )

        with self._lock:
            if bindings.nonce in self._used_nonces:
                raise AdmissionError(
                    AdmissionErrorCode.REPLAYED_TOKEN,
                    "nonce has already been consumed",
                )
            token = AdmissionToken(
                operation_id=view.operation_id,
                effect_class=view.effect_class,
                argument_cid=bindings.argument_cid,
                actor_cid=bindings.actor_cid,
                nonce=bindings.nonce,
                not_after=bindings.not_after,
                not_before=bindings.not_before,
                satisfied_obligations=_sorted_unique(have),
                derived_obligations=_sorted_unique(derived),
                issuer=KERNEL_ISSUER,
                bindings=bindings.to_dict(),
            )
            self._issued_tokens[token.token_id] = token
            if consume_nonce:
                self._used_nonces.add(bindings.nonce)
            return token

    def revoke(self, token: AdmissionToken | str) -> None:
        token_id = token.token_id if isinstance(token, AdmissionToken) else _text(
            token, "token_id"
        )
        with self._lock:
            self._revoked_token_ids.add(token_id)

    def consume(self, token: AdmissionToken) -> None:
        """Mark the token nonce as used (one-use / replay protection)."""
        if not isinstance(token, AdmissionToken):
            raise AdmissionError(
                AdmissionErrorCode.INVALID_TYPE, "token must be AdmissionToken"
            )
        with self._lock:
            if token.nonce in self._used_nonces:
                raise AdmissionError(
                    AdmissionErrorCode.REPLAYED_TOKEN,
                    "token nonce already consumed",
                )
            if token.token_id in self._revoked_token_ids:
                raise AdmissionError(
                    AdmissionErrorCode.REVOKED_TOKEN, "token is revoked"
                )
            self._used_nonces.add(token.nonce)

    def verify_token(
        self,
        token: AdmissionToken | Mapping[str, Any],
        *,
        operation_id: str,
        argument_cid: str,
        required_obligations: Iterable[str] | None = None,
        now_ms: int | None = None,
        consume: bool = False,
    ) -> AdmissionDecision:
        """Verify issuer, obligations, exact argument CID, expiry, revoke, replay."""
        try:
            parsed = (
                token
                if isinstance(token, AdmissionToken)
                else AdmissionToken.from_dict(token)
            )
        except AdmissionError as exc:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=exc.code,
                message=str(exc),
            )

        now = self.now_ms() if now_ms is None else _nonneg_int(now_ms, "now_ms")
        op = _text(operation_id, "operation_id")
        arg = _text(argument_cid, "argument_cid")

        if parsed.issuer != KERNEL_ISSUER:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER,
                message=f"issuer {parsed.issuer!r} is not the effect admission kernel",
                token_id=parsed.token_id,
            )
        if parsed.operation_id != op:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.OPERATION_MISMATCH,
                message="operation_id does not match call",
                token_id=parsed.token_id,
            )
        if parsed.argument_cid != arg:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.ARGUMENT_MISMATCH,
                message="argument_cid does not match call",
                token_id=parsed.token_id,
            )
        if parsed.not_before and now < parsed.not_before:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.NOT_YET_VALID,
                message="token is not yet valid",
                token_id=parsed.token_id,
            )
        if now >= parsed.not_after:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.EXPIRED_TOKEN,
                message="token expired",
                token_id=parsed.token_id,
            )

        with self._lock:
            if parsed.token_id in self._revoked_token_ids:
                return AdmissionDecision(
                    verdict=AdmissionVerdict.DENY,
                    code=AdmissionErrorCode.REVOKED_TOKEN,
                    message="token revoked",
                    token_id=parsed.token_id,
                )
            if parsed.nonce in self._used_nonces:
                return AdmissionDecision(
                    verdict=AdmissionVerdict.DENY,
                    code=AdmissionErrorCode.REPLAYED_TOKEN,
                    message="token nonce replayed",
                    token_id=parsed.token_id,
                )

        required = frozenset(
            _closed_enum(item, TOKEN_OBLIGATION_CONSTRUCTORS, "obligation")
            for item in (
                required_obligations
                if required_obligations is not None
                else parsed.derived_obligations or parsed.satisfied_obligations
            )
        )
        have = frozenset(parsed.satisfied_obligations)
        if not required.issubset(have):
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.TOKEN_OBLIGATION_MISMATCH,
                message=f"token missing obligations: {sorted(required - have)}",
                token_id=parsed.token_id,
            )

        if consume:
            try:
                self.consume(parsed)
            except AdmissionError as exc:
                return AdmissionDecision(
                    verdict=AdmissionVerdict.DENY,
                    code=exc.code,
                    message=str(exc),
                    token_id=parsed.token_id,
                )

        return AdmissionDecision(
            verdict=AdmissionVerdict.ADMIT,
            code=None,
            message="token admitted",
            token_id=parsed.token_id,
            unlocked=False,
        )

    def unlock_handler(
        self,
        *,
        spec: OperationSpecView | Mapping[str, Any],
        typestate: str,
        token: AdmissionToken | Mapping[str, Any] | None,
        argument_cid: str,
        terminal: str | None = None,
        now_ms: int | None = None,
        consume: bool = True,
    ) -> AdmissionDecision:
        """Apply the FACP-038 handler unlock rule; never invokes the handler."""
        view = (
            spec
            if isinstance(spec, OperationSpecView)
            else OperationSpecView.from_mapping(spec)
        )
        state = _text(typestate, "typestate")
        if view.effect_class == "pure":
            if token is not None:
                return AdmissionDecision(
                    verdict=AdmissionVerdict.DENY,
                    code=AdmissionErrorCode.PURE_TOKEN_FORBIDDEN,
                    message="pure handlers must not carry AdmissionTokens",
                )
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.HANDLER_NOT_UNLOCKED,
                message="pure handlers do not unlock via AdmissionToken",
            )

        if state not in HANDLER_UNLOCK_TYPESTATES:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.HANDLER_NOT_UNLOCKED,
                message=f"typestate {state!r} does not unlock handlers",
            )
        if terminal is not None:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.HANDLER_NOT_UNLOCKED,
                message="terminal typestate marker forbids unlock",
            )
        if token is None:
            return AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=AdmissionErrorCode.HANDLER_NOT_UNLOCKED,
                message="effectful handler requires a kernel-issued token",
            )

        derived = derive_token_obligations(view)
        decision = self.verify_token(
            token,
            operation_id=view.operation_id,
            argument_cid=argument_cid,
            required_obligations=derived,
            now_ms=now_ms,
            consume=consume,
        )
        if decision.verdict is not AdmissionVerdict.ADMIT:
            return decision
        return AdmissionDecision(
            verdict=AdmissionVerdict.ADMIT,
            code=None,
            message="handler unlocked",
            token_id=decision.token_id,
            unlocked=True,
        )


def fresh_nonce(*, size: int = 16) -> str:
    """Return a new opaque hex nonce for token minting."""
    if size < 8:
        raise AdmissionError(
            AdmissionErrorCode.INVALID_TYPE, "nonce size must be >= 8"
        )
    return secrets.token_hex(size)


def binding_cid(label: str, material: Any) -> str:
    """Deterministic opaque CID helper for tests and hermetic hosts."""
    return content_identity({"label": label, "material": material})


def default_kernel(*, now_ms: int = 0) -> EffectAdmissionKernel:
    """Construct a kernel with a fixed clock (tests / hermetic hosts)."""
    return EffectAdmissionKernel(clock_ms=lambda: now_ms)


__all__ = (
    "SCHEMA",
    "TOKEN_SCHEMA",
    "POLICY_IR_SCHEMA",
    "DECISION_SCHEMA",
    "TASK_ID",
    "GOAL_ID",
    "BUNDLE",
    "KERNEL_ISSUER",
    "KERNEL_VERSION",
    "EVIDENCE_SUBSET",
    "AdmissionBindings",
    "AdmissionDecision",
    "AdmissionError",
    "AdmissionErrorCode",
    "AdmissionToken",
    "AdmissionVerdict",
    "CompiledPolicyClause",
    "EffectAdmissionKernel",
    "OperationSpecView",
    "PolicyIRVerdict",
    "RuntimePolicyIR",
    "binding_cid",
    "compile_source_policy",
    "default_kernel",
    "derive_token_obligations",
    "fresh_nonce",
)
