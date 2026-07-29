"""Compile pinned SecurityIR declarations into exact authorization decisions.

The shared IR adapter verifies and normalizes SecurityIR.  This module is the
next authority boundary: it compiles only reviewed declarations, binds the
result to the pinned SecurityIR root, and evaluates an exact request before
delegating the principal/grant check to :mod:`authorization_logic`.

Security claims, assumptions, formal views, legal permissions, intent,
retrieval results, and model output are never grants.  They may constrain or
block a decision, but only a matching ``allow`` policy statement accepted by
the deterministic authorization evaluator can produce a permit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .authorization_logic import (
    AuthorizationDecision,
    AuthorizationGrant,
    AuthorizationPolicy,
    AuthorizationRequest,
    AuthorizationVerdict,
    Capability,
    evaluate_authorization,
)
from .ir_adapters import (
    IRAdapterResult,
    IRAdapterStatus,
    IRNodeKind,
    NormalizedIRArtifact,
    NormalizedIRNode,
    NormalizedResultAuthority,
)
from .ir_registry import IRFamily


SECURITY_CONSTRAINT_ADAPTER_VERSION: Final[int] = 1
SECURITY_AUTHORIZATION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-authorization-request@1"
)
SECURITY_SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-source-binding@1"
)
SECURITY_DECLARATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-declaration@1"
)
SECURITY_POLICY_RULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-policy-rule@1"
)
SECURITY_STATE_MACHINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-state-machine@1"
)
SECURITY_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-obligation@1"
)
SECURITY_POLICY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-policy-receipt@1"
)
SECURITY_DECISION_CHECK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-decision-check@1"
)
SECURITY_DECISION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/security-decision-receipt@1"
)

_MAX_ITEMS: Final[int] = 4096
_MAX_TEXT_BYTES: Final[int] = 8192
_EXACT_FIELDS: Final[tuple[str, ...]] = (
    "principal",
    "action",
    "tool",
    "target",
    "data_flow",
    "expected_effect",
    "requested_authority",
)
_OPTIONAL_FLOW_FIELDS: Final[tuple[str, ...]] = (
    "source_zone",
    "channel",
    "target_zone",
)
_WILDCARDS: Final[frozenset[str]] = frozenset({"*", "any", "all"})
_NON_GRANT_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "intent",
        "intentir",
        "intent_ir",
        "legal",
        "legalir",
        "legal_ir",
        "legal_permission",
        "model",
        "model_output",
        "llm",
        "retrieval",
        "retrieval_rank",
        "semantic_rank",
        "graphrag",
        "rag",
    }
)
_DECLARATION_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "principal": "principal",
        "principals": "principal",
        "security_principal": "principal",
        "actor": "principal",
        "identity": "principal",
        "asset": "asset",
        "assets": "asset",
        "security_asset": "asset",
        "resource": "resource",
        "resources": "resource",
        "security_resource": "resource",
        "tool": "resource",
        "target": "resource",
        "zone": "zone",
        "zones": "zone",
        "trust_zone": "zone",
        "security_zone": "zone",
        "channel": "channel",
        "channels": "channel",
        "communication_channel": "channel",
        "security_channel": "channel",
        "policy": "policy",
        "policies": "policy",
        "authorization_policy": "policy",
        "access_control_policy": "policy",
        "security_policy": "policy",
        "rule": "policy",
        "state_machine": "state_machine",
        "statemachine": "state_machine",
        "security_state_machine": "state_machine",
    }
)


class SecurityConstraintError(ValueError):
    """A SecurityIR compilation or authorization contract is malformed."""


class SecurityCompilationStatus(str, Enum):
    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class SecurityRuleEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class SecurityDecisionOutcome(str, Enum):
    PERMIT = "permit"
    DENY = "deny"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class SecurityCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise SecurityConstraintError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise SecurityConstraintError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise SecurityConstraintError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > maximum:
        raise SecurityConstraintError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _integer(value: Any, name: str, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SecurityConstraintError(f"{name} must be a non-negative integer")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecurityConstraintError("value is not canonical JSON") from exc


def _content_id(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _ids(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise SecurityConstraintError(f"{name} must be a string or sequence")
    if len(values) > _MAX_ITEMS:
        raise SecurityConstraintError(f"{name} exceeds its count bound")
    result = tuple(sorted({_text(item, name) for item in values}))
    return result


def _contains_wildcard(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _WILDCARDS
    if isinstance(value, Mapping):
        return any(_contains_wildcard(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_wildcard(item) for item in value)
    return False


def _marker(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _exact_value(value: Any, name: str) -> Any:
    if value is None:
        raise SecurityConstraintError(f"{name} must be explicit")
    if isinstance(value, str):
        result: Any = _text(value, name)
    elif isinstance(value, bool) or isinstance(value, int):
        result = value
    elif isinstance(value, Mapping):
        if not value:
            raise SecurityConstraintError(f"{name} must not be empty")
        result = {
            _text(key, f"{name} key"): _exact_value(item, f"{name}.{key}")
            for key, item in sorted(value.items())
        }
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not value:
            raise SecurityConstraintError(f"{name} must not be empty")
        result = [_exact_value(item, name) for item in value]
    else:
        raise SecurityConstraintError(f"{name} has an unsupported value")
    if _contains_wildcard(result):
        raise SecurityConstraintError(f"{name} cannot contain a wildcard")
    return _freeze(result)


def _optional_exact(value: Any, name: str) -> Any | None:
    return None if value in (None, "") else _exact_value(value, name)


def _kind(node: NormalizedIRNode) -> str | None:
    normalized = (
        node.declaration_kind.strip().lower().replace("-", "_").replace(" ", "_")
    )
    return _DECLARATION_ALIASES.get(normalized)


def _node_values(node: NormalizedIRNode) -> Mapping[str, Any]:
    result = dict(node.attributes)
    for nested_name in ("scope", "authorization", "constraints"):
        nested = node.attributes.get(nested_name)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise SecurityConstraintError(f"{nested_name} must be an object")
        result.update(nested)
    return result


def _binding(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode
) -> "SecuritySourceBinding":
    return SecuritySourceBinding(
        node_id=node.node_id,
        security_root_artifact_id=artifact.root_artifact_id,
        security_root_cid_v1=artifact.root_cid_v1,
        security_root_supervisor_digest=artifact.root_supervisor_digest,
        source_references=node.source_references,
        provenance_references=node.provenance_references,
        grounded=node.grounded,
        result_authority=node.result_authority,
    )


@dataclass(frozen=True)
class SecurityAuthorizationRequest:
    """Exact facts for one SecurityIR authorization decision."""

    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    principal: str
    action: str
    tool: str
    target: str
    data_flow: Any
    expected_effect: Any
    current_state: Any
    requested_authority: str
    evaluated_at_ms: int
    state_version: Any = None
    state_revision: Any = None
    source_zone: str | None = None
    channel: str | None = None
    target_zone: str | None = None
    satisfied_assumption_ids: tuple[str, ...] = ()
    accepted_claim_result_ids: tuple[str, ...] = ()
    asserted_grant_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "security_root_artifact_id",
            "security_root_cid_v1",
            "security_root_supervisor_digest",
            "principal",
            "action",
            "tool",
            "target",
            "requested_authority",
        ):
            value = _text(getattr(self, name), name)
            if _contains_wildcard(value):
                raise SecurityConstraintError(f"{name} cannot be a wildcard")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "data_flow", _exact_value(self.data_flow, "data_flow")
        )
        object.__setattr__(
            self, "expected_effect", _exact_value(self.expected_effect, "expected_effect")
        )
        object.__setattr__(
            self, "current_state", _exact_value(self.current_state, "current_state")
        )
        if self.state_version is not None and self.state_revision is not None:
            if _plain(self.state_version) != _plain(self.state_revision):
                raise SecurityConstraintError(
                    "state_version and state_revision must identify the same snapshot"
                )
        selected_version = (
            self.state_version
            if self.state_version is not None
            else self.state_revision
        )
        selected_version = _optional_exact(selected_version, "state_version")
        object.__setattr__(self, "state_version", selected_version)
        object.__setattr__(self, "state_revision", selected_version)
        flow = self.data_flow if isinstance(self.data_flow, Mapping) else {}
        flow_aliases = {
            "source_zone": ("source_zone", "from_zone"),
            "channel": ("channel", "channel_id"),
            "target_zone": ("target_zone", "to_zone"),
        }
        for name in _OPTIONAL_FLOW_FIELDS:
            value = getattr(self, name)
            if value in (None, ""):
                value = next(
                    (flow[item] for item in flow_aliases[name] if item in flow),
                    None,
                )
            if value not in (None, ""):
                value = _text(value, name)
                if _contains_wildcard(value):
                    raise SecurityConstraintError(f"{name} cannot be a wildcard")
            else:
                value = None
            object.__setattr__(self, name, value)
        moment = _integer(self.evaluated_at_ms, "evaluated_at_ms")
        assert moment is not None
        object.__setattr__(self, "evaluated_at_ms", moment)
        for name in (
            "satisfied_assumption_ids",
            "accepted_claim_result_ids",
            "asserted_grant_sources",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))

    @property
    def exact_inputs(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "principal": self.principal,
            "action": self.action,
            "tool": self.tool,
            "target": self.target,
            "data_flow": _plain(self.data_flow),
            "expected_effect": _plain(self.expected_effect),
            "requested_authority": self.requested_authority,
        }
        for name in _OPTIONAL_FLOW_FIELDS:
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return MappingProxyType(result)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_AUTHORIZATION_REQUEST_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            **dict(self.exact_inputs),
            "current_state": _plain(self.current_state),
            "state_version": _plain(self.state_version),
            "state_revision": _plain(self.state_revision),
            "evaluated_at_ms": self.evaluated_at_ms,
            "satisfied_assumption_ids": list(self.satisfied_assumption_ids),
            "accepted_claim_result_ids": list(self.accepted_claim_result_ids),
            "asserted_grant_sources": list(self.asserted_grant_sources),
            "intent_is_authority": False,
            "legal_permission_is_authority": False,
            "model_output_is_authority": False,
            "retrieval_rank_is_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = _content_id("security-authorization-request", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecurityAuthorizationRequest":
        if not isinstance(payload, Mapping):
            raise SecurityConstraintError("security authorization request must be an object")
        result = cls(
            security_root_artifact_id=payload.get("security_root_artifact_id", ""),
            security_root_cid_v1=payload.get("security_root_cid_v1", ""),
            security_root_supervisor_digest=payload.get(
                "security_root_supervisor_digest", ""
            ),
            principal=payload.get("principal", ""),
            action=payload.get("action", ""),
            tool=payload.get("tool", ""),
            target=payload.get("target", ""),
            data_flow=payload.get("data_flow"),
            expected_effect=payload.get("expected_effect"),
            current_state=payload.get("current_state"),
            requested_authority=payload.get("requested_authority", ""),
            evaluated_at_ms=payload.get("evaluated_at_ms", -1),
            state_version=payload.get("state_version"),
            state_revision=payload.get("state_revision"),
            source_zone=payload.get("source_zone"),
            channel=payload.get("channel"),
            target_zone=payload.get("target_zone"),
            satisfied_assumption_ids=tuple(
                payload.get("satisfied_assumption_ids") or ()
            ),
            accepted_claim_result_ids=tuple(
                payload.get("accepted_claim_result_ids") or ()
            ),
            asserted_grant_sources=tuple(payload.get("asserted_grant_sources") or ()),
        )
        if payload.get("content_id") not in (None, "", result.content_id):
            raise SecurityConstraintError("security request identity mismatch")
        return result


SecurityAuthorizationQuery = SecurityAuthorizationRequest


@dataclass(frozen=True)
class SecuritySourceBinding:
    node_id: str
    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    source_references: tuple[Mapping[str, Any], ...]
    provenance_references: tuple[Mapping[str, Any], ...]
    grounded: bool
    result_authority: NormalizedResultAuthority

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_SOURCE_BINDING_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "node_id": self.node_id,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            "source_references": [_plain(item) for item in self.source_references],
            "provenance_references": [
                _plain(item) for item in self.provenance_references
            ],
            "grounded": self.grounded,
            "result_authority": self.result_authority.value,
            "grants_execution_authority": False,
        }


@dataclass(frozen=True)
class SecurityDeclaration:
    declaration_id: str
    kind: str
    attributes: Mapping[str, Any]
    source_binding: SecuritySourceBinding

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_DECLARATION_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "declaration_id": self.declaration_id,
            "kind": self.kind,
            "attributes": _plain(self.attributes),
            "source_binding": self.source_binding.to_dict(),
            "is_authorization_grant": False,
        }


@dataclass(frozen=True)
class SecurityPolicyRule:
    policy_id: str
    effect: SecurityRuleEffect
    exact_scope: Mapping[str, Any]
    universal_fields: tuple[str, ...]
    priority: int
    assumption_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    state_machine_id: str | None
    from_state: str | None
    to_state: str | None
    conflicts_with: tuple[str, ...]
    source_binding: SecuritySourceBinding

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", SecurityRuleEffect(self.effect))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_POLICY_RULE_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "policy_id": self.policy_id,
            "effect": self.effect.value,
            "exact_scope": _plain(self.exact_scope),
            "universal_fields": list(self.universal_fields),
            "priority": self.priority,
            "assumption_ids": list(self.assumption_ids),
            "claim_ids": list(self.claim_ids),
            "obligation_ids": list(self.obligation_ids),
            "state_machine_id": self.state_machine_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "conflicts_with": list(self.conflicts_with),
            "source_binding": self.source_binding.to_dict(),
        }

    def matches(self, request: SecurityAuthorizationRequest) -> bool:
        inputs = request.exact_inputs
        for name in (*_EXACT_FIELDS, *_OPTIONAL_FLOW_FIELDS):
            if (
                name in _OPTIONAL_FLOW_FIELDS
                and name not in inputs
                and name not in self.exact_scope
            ):
                continue
            if name in self.universal_fields:
                continue
            if name not in self.exact_scope:
                return False
            if name not in inputs or _plain(self.exact_scope[name]) != _plain(inputs[name]):
                return False
        return True

    def matches_except_effect(self, request: SecurityAuthorizationRequest) -> bool:
        inputs = request.exact_inputs
        for name in (*_EXACT_FIELDS, *_OPTIONAL_FLOW_FIELDS):
            if name == "expected_effect" or name in self.universal_fields:
                continue
            if (
                name in _OPTIONAL_FLOW_FIELDS
                and name not in inputs
                and name not in self.exact_scope
            ):
                continue
            if name not in self.exact_scope:
                return False
            if name not in inputs or _plain(self.exact_scope[name]) != _plain(inputs[name]):
                return False
        return True


@dataclass(frozen=True)
class SecurityTransition:
    transition_id: str
    from_state: str
    to_state: str
    action: str
    tool: str | None
    target: str | None
    expected_effect: Any | None
    guard_assumption_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action": self.action,
            "tool": self.tool,
            "target": self.target,
            "expected_effect": _plain(self.expected_effect),
            "guard_assumption_ids": list(self.guard_assumption_ids),
        }


@dataclass(frozen=True)
class CompiledSecurityStateMachine:
    state_machine_id: str
    resource_id: str
    states: tuple[str, ...]
    current_state: str
    state_version: Any
    transitions: tuple[SecurityTransition, ...]
    source_binding: SecuritySourceBinding

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_STATE_MACHINE_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "state_machine_id": self.state_machine_id,
            "resource_id": self.resource_id,
            "states": list(self.states),
            "current_state": self.current_state,
            "state_version": _plain(self.state_version),
            "transitions": [item.to_dict() for item in self.transitions],
            "source_binding": self.source_binding.to_dict(),
        }


@dataclass(frozen=True)
class CompiledSecurityObligation:
    obligation_id: str
    required: bool
    discharged: bool
    assumption_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    source_binding: SecuritySourceBinding

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_OBLIGATION_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "obligation_id": self.obligation_id,
            "required": self.required,
            "discharged": self.discharged,
            "assumption_ids": list(self.assumption_ids),
            "claim_ids": list(self.claim_ids),
            "source_binding": self.source_binding.to_dict(),
        }


@dataclass(frozen=True)
class SecurityPolicyReceipt:
    status: SecurityCompilationStatus
    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    principals: tuple[SecurityDeclaration, ...] = ()
    assets: tuple[SecurityDeclaration, ...] = ()
    resources: tuple[SecurityDeclaration, ...] = ()
    zones: tuple[SecurityDeclaration, ...] = ()
    channels: tuple[SecurityDeclaration, ...] = ()
    policies: tuple[SecurityPolicyRule, ...] = ()
    state_machines: tuple[CompiledSecurityStateMachine, ...] = ()
    threat_assumptions: tuple[SecurityDeclaration, ...] = ()
    claims: tuple[SecurityDeclaration, ...] = ()
    formal_obligations: tuple[CompiledSecurityObligation, ...] = ()
    result_authorities: tuple[SecurityDeclaration, ...] = ()
    authorization_policy: AuthorizationPolicy | None = None
    reason_codes: tuple[str, ...] = ()
    authoritative_scan_complete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SecurityCompilationStatus(self.status))
        object.__setattr__(
            self, "reason_codes", tuple(sorted(set(self.reason_codes)))
        )

    @property
    def successful(self) -> bool:
        return (
            self.status is SecurityCompilationStatus.COMPILED
            and self.authoritative_scan_complete
            and self.authorization_policy is not None
        )

    @property
    def fail_closed(self) -> bool:
        return not self.successful

    @property
    def policy_identity(self) -> str:
        return self.content_id

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def compiled(self) -> bool:
        return self.successful

    @property
    def unsupported(self) -> bool:
        return self.status is SecurityCompilationStatus.UNSUPPORTED

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_POLICY_RECEIPT_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "status": self.status.value,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            "principals": [item.to_dict() for item in self.principals],
            "assets": [item.to_dict() for item in self.assets],
            "resources": [item.to_dict() for item in self.resources],
            "zones": [item.to_dict() for item in self.zones],
            "channels": [item.to_dict() for item in self.channels],
            "policies": [item.to_dict() for item in self.policies],
            "state_machines": [item.to_dict() for item in self.state_machines],
            "threat_assumptions": [
                item.to_dict() for item in self.threat_assumptions
            ],
            "claims": [item.to_dict() for item in self.claims],
            "formal_obligations": [
                item.to_dict() for item in self.formal_obligations
            ],
            "result_authorities": [
                item.to_dict() for item in self.result_authorities
            ],
            "authorization_policy": (
                self.authorization_policy.to_dict()
                if self.authorization_policy is not None
                else None
            ),
            "reason_codes": list(self.reason_codes),
            "authoritative_scan_complete": self.authoritative_scan_complete,
            "deny_overrides": True,
            "wildcard_broadening_allowed": False,
            "claims_are_grants": False,
            "assumptions_are_grants": False,
            "formalizations_are_grants": False,
            "grants_execution_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = _content_id("security-policy-receipt", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


CompiledSecurityPolicy = SecurityPolicyReceipt
SecurityCompilationResult = SecurityPolicyReceipt


@dataclass(frozen=True)
class SecurityDecisionCheck:
    check_id: str
    status: SecurityCheckStatus
    reason_code: str
    policy_ids: tuple[str, ...] = ()
    declaration_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", _text(self.check_id, "check_id"))
        object.__setattr__(self, "status", SecurityCheckStatus(self.status))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))
        object.__setattr__(self, "policy_ids", _ids(self.policy_ids, "policy_ids"))
        object.__setattr__(
            self, "declaration_ids", _ids(self.declaration_ids, "declaration_ids")
        )

    @property
    def passed(self) -> bool:
        return self.status is SecurityCheckStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_DECISION_CHECK_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "check_id": self.check_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "policy_ids": list(self.policy_ids),
            "declaration_ids": list(self.declaration_ids),
        }


@dataclass(frozen=True)
class SecurityDecisionReceipt:
    outcome: SecurityDecisionOutcome
    policy_receipt_id: str
    request_id: str
    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    evaluated_at_ms: int
    checks: tuple[SecurityDecisionCheck, ...]
    matched_policy_ids: tuple[str, ...] = ()
    authorization_decision: AuthorizationDecision | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", SecurityDecisionOutcome(self.outcome))
        object.__setattr__(
            self,
            "checks",
            tuple(sorted(self.checks, key=lambda item: item.check_id)),
        )
        object.__setattr__(
            self, "matched_policy_ids", _ids(self.matched_policy_ids, "matched_policy_ids")
        )
        object.__setattr__(
            self, "reason_codes", tuple(sorted(set(self.reason_codes)))
        )
        if self.outcome is SecurityDecisionOutcome.PERMIT:
            if (
                self.authorization_decision is None
                or not self.authorization_decision.permitted
                or not all(item.passed for item in self.checks)
                or not self.matched_policy_ids
            ):
                raise SecurityConstraintError(
                    "permit requires exact checks and a reference authorization permit"
                )

    @property
    def permitted(self) -> bool:
        return self.outcome is SecurityDecisionOutcome.PERMIT

    @property
    def permits_action(self) -> bool:
        return self.permitted

    @property
    def verdict(self) -> SecurityDecisionOutcome:
        return self.outcome

    @property
    def allowed(self) -> bool:
        return self.permitted

    @property
    def accepted(self) -> bool:
        return self.permitted

    @property
    def fail_closed(self) -> bool:
        return not self.permitted

    @property
    def establishes_generated_code_correctness(self) -> bool:
        return False

    @property
    def grants_execution_authority(self) -> bool:
        # This is an authorization input to a later exact execution permit.
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SECURITY_DECISION_RECEIPT_SCHEMA,
            "adapter_version": SECURITY_CONSTRAINT_ADAPTER_VERSION,
            "outcome": self.outcome.value,
            "policy_receipt_id": self.policy_receipt_id,
            "request_id": self.request_id,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            "evaluated_at_ms": self.evaluated_at_ms,
            "checks": [item.to_dict() for item in self.checks],
            "matched_policy_ids": list(self.matched_policy_ids),
            "authorization_decision": (
                self.authorization_decision.to_dict()
                if self.authorization_decision is not None
                else None
            ),
            "reason_codes": list(self.reason_codes),
            "permits_action": self.permitted,
            "deny_overrides": True,
            "establishes_generated_code_correctness": False,
            "grants_execution_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = _content_id("security-decision-receipt", payload)
        return payload

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


SecurityAuthorizationDecision = SecurityDecisionReceipt
SecurityAuthorizationOutcome = SecurityDecisionOutcome
SecurityPolicyCompilationResult = SecurityPolicyReceipt


def _declaration(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode, kind: str
) -> SecurityDeclaration:
    return SecurityDeclaration(
        declaration_id=node.node_id,
        kind=kind,
        attributes=_freeze(dict(node.attributes)),
        source_binding=_binding(artifact, node),
    )


def _trusted_node(
    node: NormalizedIRNode,
    *authorities: NormalizedResultAuthority,
    require_grounded: bool = True,
) -> bool:
    return (
        (node.grounded or not require_grounded)
        and node.review_state.accepted
        and node.trust_state.accepted
        and node.result_authority in authorities
        and bool(node.source_references)
        and bool(node.provenance_references)
    )


def _rule(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode
) -> SecurityPolicyRule:
    values = _node_values(node)
    raw_effect = values.get(
        "decision", values.get("verdict", values.get("mode", values.get("effect")))
    )
    if not isinstance(raw_effect, str):
        raise SecurityConstraintError("policy effect must be explicit")
    effect_aliases = {
        "permit": SecurityRuleEffect.ALLOW,
        "permitted": SecurityRuleEffect.ALLOW,
        "allow": SecurityRuleEffect.ALLOW,
        "forbid": SecurityRuleEffect.DENY,
        "forbidden": SecurityRuleEffect.DENY,
        "prohibit": SecurityRuleEffect.DENY,
        "deny": SecurityRuleEffect.DENY,
        "unknown": SecurityRuleEffect.UNKNOWN,
        "conflict": SecurityRuleEffect.CONFLICT,
        "conflicting": SecurityRuleEffect.CONFLICT,
    }
    effect = effect_aliases.get(raw_effect.strip().lower())
    if effect is None:
        raise SecurityConstraintError("unsupported policy effect")
    raw_grant_sources = values.get(
        "grant_sources",
        values.get(
            "grant_source",
            values.get("authority_source", values.get("authorization_source")),
        ),
    )
    grant_sources = _ids(raw_grant_sources, "grant_sources")
    if any(
        _marker(item) in _NON_GRANT_SOURCES
        for item in grant_sources
    ):
        raise SecurityConstraintError(
            "intent, legal, model, and retrieval inputs cannot be policy grants"
        )
    for marker in (
        "intent_is_authority",
        "intent_authorized",
        "legal_permission_is_authority",
        "legal_permission_authorized",
        "model_output_is_authority",
        "model_authorized",
        "retrieval_rank_is_authority",
        "retrieval_authorized",
    ):
        if values.get(marker) not in (None, False):
            raise SecurityConstraintError(f"{marker} cannot grant authorization")

    universal = _ids(
        values.get("universal_fields", values.get("applies_to_all", ())),
        "universal_fields",
    )
    supported_fields = frozenset((*_EXACT_FIELDS, *_OPTIONAL_FLOW_FIELDS))
    if not set(universal).issubset(supported_fields):
        raise SecurityConstraintError("universal_fields contains an unsupported field")
    if effect is SecurityRuleEffect.ALLOW and universal:
        raise SecurityConstraintError(
            "allow policy cannot broaden exact scope with universal_fields"
        )
    exact: dict[str, Any] = {}
    aliases: Mapping[str, tuple[str, ...]] = {
        "principal": ("principal", "subject"),
        "action": ("action", "operation"),
        "tool": ("tool", "tool_id"),
        "target": ("target", "resource", "resource_id"),
        "data_flow": ("data_flow", "flow"),
        "expected_effect": (
            "expected_effect",
            "result_effect",
            "effect_manifest",
            "effect",
        ),
        "requested_authority": ("requested_authority", "authority", "authority_class"),
        "source_zone": ("source_zone", "from_zone"),
        "channel": ("channel", "channel_id"),
        "target_zone": ("target_zone", "to_zone"),
    }
    for name, candidates in aliases.items():
        raw = next((values[item] for item in candidates if item in values), None)
        if raw is not None:
            exact[name] = _exact_value(raw, name)
        elif name in _EXACT_FIELDS and name not in universal:
            raise SecurityConstraintError(f"policy lacks exact {name}")
    if isinstance(exact.get("data_flow"), Mapping):
        flow = exact["data_flow"]
        for name, candidates in {
            "source_zone": ("source_zone", "from_zone"),
            "channel": ("channel", "channel_id"),
            "target_zone": ("target_zone", "to_zone"),
        }.items():
            if name not in exact:
                nested_value = next(
                    (flow[item] for item in candidates if item in flow),
                    None,
                )
                if nested_value is not None:
                    exact[name] = _exact_value(nested_value, name)
    priority = _integer(values.get("priority", values.get("precedence")), "priority", default=0)
    assert priority is not None
    state_machine_id = values.get("state_machine_id")
    from_state = values.get("from_state", values.get("required_state"))
    to_state = values.get("to_state", values.get("next_state"))
    for name, value in (
        ("state_machine_id", state_machine_id),
        ("from_state", from_state),
        ("to_state", to_state),
    ):
        if value not in (None, ""):
            if not isinstance(value, str):
                raise SecurityConstraintError(f"{name} must be a string")
            if _contains_wildcard(value):
                raise SecurityConstraintError(f"{name} cannot contain a wildcard")
    return SecurityPolicyRule(
        policy_id=node.node_id,
        effect=effect,
        exact_scope=_freeze(exact),
        universal_fields=universal,
        priority=priority,
        assumption_ids=_ids(
            values.get("assumption_ids", values.get("required_assumptions")),
            "assumption_ids",
        ),
        claim_ids=_ids(
            values.get("claim_ids", values.get("required_claims")), "claim_ids"
        ),
        obligation_ids=_ids(
            values.get("obligation_ids", values.get("proof_obligation_ids")),
            "obligation_ids",
        ),
        state_machine_id=(
            None if state_machine_id in (None, "") else _text(state_machine_id, "state_machine_id")
        ),
        from_state=None if from_state in (None, "") else _text(from_state, "from_state"),
        to_state=None if to_state in (None, "") else _text(to_state, "to_state"),
        conflicts_with=_ids(values.get("conflicts_with"), "conflicts_with"),
        source_binding=_binding(artifact, node),
    )


def _transition(value: Any, machine_id: str, index: int) -> SecurityTransition:
    if not isinstance(value, Mapping):
        raise SecurityConstraintError("state transition must be an object")
    transition_id = value.get("id", value.get("transition_id"))
    if transition_id in (None, ""):
        transition_id = f"{machine_id}:transition:{index}"
    expected_effect = value.get("expected_effect", value.get("effect"))
    return SecurityTransition(
        transition_id=_text(transition_id, "transition_id"),
        from_state=_text(value.get("from", value.get("from_state", "")), "from_state"),
        to_state=_text(value.get("to", value.get("to_state", "")), "to_state"),
        action=_text(value.get("action", value.get("event", "")), "transition action"),
        tool=(
            None
            if value.get("tool", value.get("tool_id")) in (None, "")
            else _text(value.get("tool", value.get("tool_id")), "transition tool")
        ),
        target=(
            None
            if value.get("target", value.get("resource_id")) in (None, "")
            else _text(
                value.get("target", value.get("resource_id")), "transition target"
            )
        ),
        expected_effect=(
            None
            if expected_effect is None
            else _exact_value(expected_effect, "transition expected_effect")
        ),
        guard_assumption_ids=_ids(
            value.get("guard_assumption_ids", value.get("assumption_ids")),
            "guard_assumption_ids",
        ),
    )


def _state_machine(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode
) -> CompiledSecurityStateMachine:
    values = _node_values(node)
    states = _ids(values.get("states"), "states")
    if not states:
        raise SecurityConstraintError("state machine states must not be empty")
    current = _text(
        values.get("current_state", values.get("current", "")), "current_state"
    )
    if current not in states:
        raise SecurityConstraintError("current_state is not a declared state")
    resource_id = _text(
        values.get("resource_id", values.get("target", "")), "state resource_id"
    )
    state_version = _exact_value(
        values.get("state_version", values.get("revision")), "state_version"
    )
    raw_transitions = values.get("transitions")
    if isinstance(raw_transitions, (str, bytes)) or not isinstance(
        raw_transitions, Sequence
    ):
        raise SecurityConstraintError("state transitions must be a sequence")
    transitions = tuple(
        _transition(item, node.node_id, index)
        for index, item in enumerate(raw_transitions)
    )
    if len({item.transition_id for item in transitions}) != len(transitions):
        raise SecurityConstraintError("state transition IDs must be unique")
    if any(
        item.from_state not in states or item.to_state not in states
        for item in transitions
    ):
        raise SecurityConstraintError("transition references an unknown state")
    return CompiledSecurityStateMachine(
        state_machine_id=node.node_id,
        resource_id=resource_id,
        states=states,
        current_state=current,
        state_version=state_version,
        transitions=tuple(sorted(transitions, key=lambda item: item.transition_id)),
        source_binding=_binding(artifact, node),
    )


def _obligation(
    artifact: NormalizedIRArtifact, node: NormalizedIRNode
) -> CompiledSecurityObligation:
    values = _node_values(node)
    required = values.get("required", True)
    discharged = values.get("discharged", values.get("satisfied", False))
    if not isinstance(required, bool) or not isinstance(discharged, bool):
        raise SecurityConstraintError("obligation flags must be booleans")
    return CompiledSecurityObligation(
        obligation_id=node.node_id,
        required=required,
        discharged=discharged,
        assumption_ids=_ids(values.get("assumption_ids"), "assumption_ids"),
        claim_ids=_ids(values.get("claim_ids"), "claim_ids"),
        source_binding=_binding(artifact, node),
    )


def _failed_policy(
    *,
    reason: str,
    status: SecurityCompilationStatus = SecurityCompilationStatus.UNSUPPORTED,
    artifact: NormalizedIRArtifact | None = None,
) -> SecurityPolicyReceipt:
    return SecurityPolicyReceipt(
        status=status,
        security_root_artifact_id=artifact.root_artifact_id if artifact else "",
        security_root_cid_v1=artifact.root_cid_v1 if artifact else "",
        security_root_supervisor_digest=(
            artifact.root_supervisor_digest if artifact else ""
        ),
        reason_codes=(reason,),
        authoritative_scan_complete=False,
    )


class SecurityConstraintAdapter:
    """Provider-free SecurityIR compiler and exact authorization evaluator."""

    adapter_id: Final[str] = "supervisor-security-constraint-adapter@1"

    def compile(
        self, artifact: NormalizedIRArtifact | IRAdapterResult | None
    ) -> SecurityPolicyReceipt:
        if artifact is None:
            return _failed_policy(reason="missing_trusted_security_source")
        if isinstance(artifact, IRAdapterResult):
            if artifact.status is not IRAdapterStatus.NORMALIZED:
                assert artifact.failure is not None
                return _failed_policy(
                    reason=f"security_ir_{artifact.failure.code.value}"
                )
            artifact = artifact.require_artifact()
        if not isinstance(artifact, NormalizedIRArtifact):
            raise SecurityConstraintError(
                "artifact must be NormalizedIRArtifact, IRAdapterResult, or None"
            )
        if artifact.family is not IRFamily.SECURITY:
            return _failed_policy(reason="unsupported_ir_family", artifact=artifact)
        if (
            not artifact.review_state.accepted
            or not artifact.trust_state.accepted
            or artifact.declared_authority.value not in {"authoritative", "verified"}
        ):
            return _failed_policy(
                reason="security_root_requires_review", artifact=artifact
            )

        buckets: dict[str, list[SecurityDeclaration]] = {
            "principal": [],
            "asset": [],
            "resource": [],
            "zone": [],
            "channel": [],
        }
        rules: list[SecurityPolicyRule] = []
        machines: list[CompiledSecurityStateMachine] = []
        reasons: set[str] = set()
        seen_ids: set[str] = set()
        try:
            for node in artifact.declarations:
                if node.node_id in seen_ids:
                    raise SecurityConstraintError("duplicate security declaration")
                seen_ids.add(node.node_id)
                kind = _kind(node)
                if kind is None:
                    reasons.add("unsupported_declaration_kind")
                    continue
                if not _trusted_node(node, NormalizedResultAuthority.POLICY_INPUT):
                    reasons.add("untrusted_or_unreviewed_declaration")
                    continue
                if kind in buckets:
                    buckets[kind].append(_declaration(artifact, node, kind))
                elif kind == "policy":
                    rules.append(_rule(artifact, node))
                elif kind == "state_machine":
                    machines.append(_state_machine(artifact, node))

            assumptions = tuple(
                _declaration(artifact, node, "threat_assumption")
                for node in artifact.assumptions
                if _trusted_node(
                    node,
                    NormalizedResultAuthority.CONTEXT_ONLY,
                    require_grounded=False,
                )
            )
            claims = tuple(
                _declaration(artifact, node, "claim")
                for node in artifact.claims
                if _trusted_node(node, NormalizedResultAuthority.POLICY_INPUT)
            )
            obligations = tuple(
                _obligation(artifact, node)
                for node in artifact.obligations
                if _trusted_node(node, NormalizedResultAuthority.POLICY_INPUT)
            )
            result_authorities = tuple(
                _declaration(artifact, node, "result_authority")
                for node in artifact.result_authority
                if node.grounded
                and node.review_state.accepted
                and node.trust_state.accepted
                and node.result_authority
                is NormalizedResultAuthority.VERIFIED_INPUT
            )
        except (SecurityConstraintError, TypeError, ValueError):
            return _failed_policy(
                reason="malformed_security_declaration", artifact=artifact
            )

        principals = tuple(
            sorted(buckets["principal"], key=lambda item: item.declaration_id)
        )
        assets = tuple(sorted(buckets["asset"], key=lambda item: item.declaration_id))
        resources = tuple(
            sorted(buckets["resource"], key=lambda item: item.declaration_id)
        )
        zones = tuple(sorted(buckets["zone"], key=lambda item: item.declaration_id))
        channels = tuple(
            sorted(buckets["channel"], key=lambda item: item.declaration_id)
        )
        rules = sorted(rules, key=lambda item: item.policy_id)
        machines = sorted(machines, key=lambda item: item.state_machine_id)

        principal_ids = {item.declaration_id for item in principals}
        asset_ids = {item.declaration_id for item in assets}
        resource_ids = {item.declaration_id for item in resources}
        zone_ids = {item.declaration_id for item in zones}
        channel_ids = {item.declaration_id for item in channels}
        assumption_ids = {item.declaration_id for item in assumptions}
        claim_ids = {item.declaration_id for item in claims}
        obligation_ids = {item.obligation_id for item in obligations}
        result_authority_ids = {
            item.declaration_id for item in result_authorities
        }
        machine_ids = {item.state_machine_id for item in machines}
        rule_ids = {item.policy_id for item in rules}
        for rule in rules:
            scope = rule.exact_scope
            if "principal" in scope and scope["principal"] not in principal_ids:
                reasons.add("policy_references_unknown_principal")
            if "tool" in scope and scope["tool"] not in resource_ids:
                reasons.add("policy_references_unknown_resource")
            if (
                "target" in scope
                and scope["target"] not in resource_ids
                and scope["target"] not in asset_ids
            ):
                reasons.add("policy_references_unknown_resource")
            if "source_zone" in scope and scope["source_zone"] not in zone_ids:
                reasons.add("policy_references_unknown_zone")
            if "target_zone" in scope and scope["target_zone"] not in zone_ids:
                reasons.add("policy_references_unknown_zone")
            if "channel" in scope and scope["channel"] not in channel_ids:
                reasons.add("policy_references_unknown_channel")
            if set(rule.assumption_ids) - assumption_ids:
                reasons.add("policy_references_unknown_assumption")
            if set(rule.claim_ids) - claim_ids:
                reasons.add("policy_references_unknown_claim")
            if set(rule.obligation_ids) - obligation_ids:
                reasons.add("policy_references_unknown_obligation")
            if rule.state_machine_id and rule.state_machine_id not in machine_ids:
                reasons.add("policy_references_unknown_state_machine")
            if set(rule.conflicts_with) - rule_ids:
                reasons.add("policy_references_unknown_conflict")
        if any(machine.resource_id not in resource_ids for machine in machines):
            reasons.add("state_machine_references_unknown_resource")
        for channel in channels:
            source_zone = channel.attributes.get(
                "source_zone", channel.attributes.get("from_zone")
            )
            target_zone = channel.attributes.get(
                "target_zone", channel.attributes.get("to_zone")
            )
            if source_zone is not None and source_zone not in zone_ids:
                reasons.add("channel_references_unknown_zone")
            if target_zone is not None and target_zone not in zone_ids:
                reasons.add("channel_references_unknown_zone")
        for claim in claims:
            result_id = claim.attributes.get(
                "result_authority_id",
                claim.attributes.get("result_id"),
            )
            if result_id is not None and (
                not isinstance(result_id, str)
                or result_id not in result_authority_ids
            ):
                reasons.add("claim_references_unknown_result_authority")
        if not principals:
            reasons.add("no_security_principals")
        if not resources:
            reasons.add("no_security_resources")
        if not rules:
            reasons.add("no_authoritative_security_policies")

        blocking = {
            reason
            for reason in reasons
            if reason
            not in {
                # Unsupported, unreferenced declarations are still explicit
                # and make the compilation unsupported rather than disappearing.
            }
        }
        grants: list[AuthorizationGrant] = []
        trusted_root = f"security-root:{artifact.root_supervisor_digest}"
        if not blocking:
            for rule in rules:
                if (
                    rule.effect is SecurityRuleEffect.ALLOW
                    and isinstance(rule.exact_scope.get("principal"), str)
                ):
                    grants.append(
                        AuthorizationGrant(
                            statement_id=rule.policy_id,
                            issuer=trusted_root,
                            subject=rule.exact_scope["principal"],
                            capability=Capability.CLAIM_TASK,
                            task_scope=(rule.policy_id,),
                            lease_scope=(rule.policy_id,),
                            worktree_scope=(rule.policy_id,),
                            path_scope=(rule.policy_id,),
                        )
                    )
        authorization_policy = (
            AuthorizationPolicy(
                policy_id=f"security-ir:{artifact.root_artifact_id}",
                version=artifact.root_supervisor_digest,
                trusted_roots=(trusted_root,),
                grants=tuple(grants),
            )
            if not blocking
            else None
        )
        return SecurityPolicyReceipt(
            status=(
                SecurityCompilationStatus.COMPILED
                if not blocking
                else SecurityCompilationStatus.UNSUPPORTED
            ),
            security_root_artifact_id=artifact.root_artifact_id,
            security_root_cid_v1=artifact.root_cid_v1,
            security_root_supervisor_digest=artifact.root_supervisor_digest,
            principals=principals,
            assets=assets,
            resources=resources,
            zones=zones,
            channels=channels,
            policies=tuple(rules),
            state_machines=tuple(machines),
            threat_assumptions=assumptions,
            claims=claims,
            formal_obligations=obligations,
            result_authorities=result_authorities,
            authorization_policy=authorization_policy,
            reason_codes=tuple(sorted(reasons)),
            authoritative_scan_complete=not blocking,
        )

    def evaluate(
        self,
        policy: SecurityPolicyReceipt,
        request: SecurityAuthorizationRequest,
    ) -> SecurityDecisionReceipt:
        if not isinstance(policy, SecurityPolicyReceipt):
            raise SecurityConstraintError("policy must be a SecurityPolicyReceipt")
        if not isinstance(request, SecurityAuthorizationRequest):
            raise SecurityConstraintError(
                "request must be a SecurityAuthorizationRequest"
            )
        checks: list[SecurityDecisionCheck] = []
        reasons: set[str] = set()

        def add(
            check_id: str,
            status: SecurityCheckStatus,
            reason: str,
            *,
            policy_ids: tuple[str, ...] = (),
            declaration_ids: tuple[str, ...] = (),
        ) -> None:
            checks.append(
                SecurityDecisionCheck(
                    check_id,
                    status,
                    reason,
                    policy_ids,
                    declaration_ids,
                )
            )
            if status is not SecurityCheckStatus.PASS:
                reasons.add(reason)

        root_matches = (
            request.security_root_artifact_id == policy.security_root_artifact_id
            and request.security_root_cid_v1 == policy.security_root_cid_v1
            and request.security_root_supervisor_digest
            == policy.security_root_supervisor_digest
        )
        add(
            "security_ir_root",
            SecurityCheckStatus.PASS if root_matches else SecurityCheckStatus.FAIL,
            "security_root_bound" if root_matches else "changed_security_root",
        )
        add(
            "compiled_policy",
            SecurityCheckStatus.PASS
            if policy.successful
            else SecurityCheckStatus.UNKNOWN,
            "policy_compiled" if policy.successful else "unsupported_policy",
        )
        poisoned = tuple(
            item
            for item in request.asserted_grant_sources
            if _marker(item) in _NON_GRANT_SOURCES
        )
        add(
            "grant_authority_source",
            SecurityCheckStatus.FAIL if poisoned else SecurityCheckStatus.PASS,
            "non_security_input_used_as_grant"
            if poisoned
            else "security_policy_is_only_grant_source",
            declaration_ids=poisoned,
        )

        principals = {item.declaration_id for item in policy.principals}
        resources = {item.declaration_id for item in policy.resources}
        assets = {item.declaration_id for item in policy.assets}
        zones = {item.declaration_id for item in policy.zones}
        channels = {item.declaration_id for item in policy.channels}
        resource_missing = tuple(
            item
            for item, known in (
                (request.tool, resources),
                (request.target, resources | assets),
            )
            if item not in known
        )
        add(
            "principal",
            SecurityCheckStatus.PASS
            if request.principal in principals
            else SecurityCheckStatus.UNKNOWN,
            "known_principal"
            if request.principal in principals
            else "unknown_principal",
            declaration_ids=(request.principal,),
        )
        add(
            "resources",
            SecurityCheckStatus.PASS
            if not resource_missing
            else SecurityCheckStatus.UNKNOWN,
            "known_resources" if not resource_missing else "unknown_resource",
            declaration_ids=resource_missing,
        )
        flow_missing: list[str] = []
        if request.source_zone is not None and request.source_zone not in zones:
            flow_missing.append(request.source_zone)
        if request.target_zone is not None and request.target_zone not in zones:
            flow_missing.append(request.target_zone)
        if request.channel is not None and request.channel not in channels:
            flow_missing.append(request.channel)
        add(
            "trust_zones_and_channel",
            SecurityCheckStatus.PASS
            if not flow_missing
            else SecurityCheckStatus.UNKNOWN,
            "known_flow_boundary"
            if not flow_missing
            else "unknown_zone_or_channel",
            declaration_ids=tuple(flow_missing),
        )

        matching = tuple(rule for rule in policy.policies if rule.matches(request))
        near_effect = tuple(
            rule
            for rule in policy.policies
            if rule.matches_except_effect(request)
            and not rule.matches(request)
        )
        explicit_conflicts = tuple(
            rule
            for rule in matching
            if rule.effect is SecurityRuleEffect.CONFLICT
            or set(rule.conflicts_with).intersection(
                item.policy_id for item in matching
            )
        )
        unknown_rules = tuple(
            rule for rule in matching if rule.effect is SecurityRuleEffect.UNKNOWN
        )
        deny_rules = tuple(
            rule for rule in matching if rule.effect is SecurityRuleEffect.DENY
        )
        allow_rules = tuple(
            rule for rule in matching if rule.effect is SecurityRuleEffect.ALLOW
        )
        if explicit_conflicts:
            policy_status = SecurityCheckStatus.CONFLICT
            policy_reason = "conflicting_policy"
        elif deny_rules:
            policy_status = SecurityCheckStatus.FAIL
            policy_reason = "deny_override"
        elif unknown_rules:
            policy_status = SecurityCheckStatus.UNKNOWN
            policy_reason = "explicit_unknown_policy"
        elif not matching and near_effect:
            policy_status = SecurityCheckStatus.FAIL
            policy_reason = "changed_expected_effect"
        elif not allow_rules:
            policy_status = SecurityCheckStatus.UNKNOWN
            policy_reason = "no_exact_applicable_policy"
        else:
            policy_status = SecurityCheckStatus.PASS
            policy_reason = "exact_allow_policy"
        add(
            "exact_policy_scope",
            policy_status,
            policy_reason,
            policy_ids=tuple(item.policy_id for item in matching or near_effect),
        )

        assumption_ids = {item.declaration_id for item in policy.threat_assumptions}
        claims = {item.declaration_id: item for item in policy.claims}
        claim_ids = set(claims)
        result_authority_ids = {
            item.declaration_id for item in policy.result_authorities
        }
        obligations = {
            item.obligation_id: item for item in policy.formal_obligations
        }
        required_assumptions = {
            dependency
            for rule in allow_rules
            for dependency in rule.assumption_ids
        }
        required_claims = {
            dependency for rule in allow_rules for dependency in rule.claim_ids
        }
        required_obligations = {
            dependency for rule in allow_rules for dependency in rule.obligation_ids
        }
        for claim_id in required_claims:
            claim = claims.get(claim_id)
            if claim is not None:
                required_assumptions.update(
                    _ids(claim.attributes.get("assumption_ids"), "assumption_ids")
                )
        for obligation_id in required_obligations:
            obligation = obligations.get(obligation_id)
            if obligation is not None:
                required_assumptions.update(obligation.assumption_ids)
        missing_assumptions = tuple(
            sorted(
                required_assumptions
                - set(request.satisfied_assumption_ids)
                | (required_assumptions - assumption_ids)
            )
        )
        missing_claim_results: set[str] = set()
        accepted_results = set(request.accepted_claim_result_ids)
        accepted_statuses = {
            "accepted",
            "authoritative",
            "discharged",
            "proved",
            "proven",
            "satisfied",
            "valid",
            "verified",
        }
        for claim_id in required_claims:
            claim = claims.get(claim_id)
            if claim is None:
                missing_claim_results.add(claim_id)
                continue
            result_id = claim.attributes.get(
                "result_authority_id",
                claim.attributes.get("result_id"),
            )
            expected_receipt_id = result_id if isinstance(result_id, str) else claim_id
            if expected_receipt_id not in accepted_results:
                missing_claim_results.add(expected_receipt_id)
            if isinstance(result_id, str) and result_id not in result_authority_ids:
                missing_claim_results.add(result_id)
            status = claim.attributes.get(
                "result_status", claim.attributes.get("status")
            )
            if status is not None and (
                not isinstance(status, str)
                or status.strip().lower() not in accepted_statuses
            ):
                missing_claim_results.add(claim_id)
        missing_claims = tuple(sorted(missing_claim_results))
        undischarged = tuple(
            sorted(
                item
                for item in required_obligations
                if item not in obligations
                or (
                    obligations[item].required
                    and not obligations[item].discharged
                )
            )
        )
        add(
            "threat_assumptions",
            SecurityCheckStatus.PASS
            if not missing_assumptions
            else SecurityCheckStatus.UNKNOWN,
            "assumption_dependencies_satisfied"
            if not missing_assumptions
            else "unsatisfied_threat_assumption",
            declaration_ids=missing_assumptions,
        )
        add(
            "claim_result_authority",
            SecurityCheckStatus.PASS
            if not missing_claims
            else SecurityCheckStatus.UNKNOWN,
            "claim_dependencies_accepted"
            if not missing_claims
            else "claim_result_authority_missing",
            declaration_ids=missing_claims,
        )
        add(
            "formal_obligations",
            SecurityCheckStatus.PASS
            if not undischarged
            else SecurityCheckStatus.FAIL,
            "formal_obligations_discharged"
            if not undischarged
            else "formal_obligation_undischarged",
            declaration_ids=undischarged,
        )

        machine_by_id = {
            item.state_machine_id: item for item in policy.state_machines
        }
        state_rules = tuple(rule for rule in allow_rules if rule.state_machine_id)
        stale: list[str] = []
        guard_missing: set[str] = set()
        transition_missing: list[str] = []
        for rule in state_rules:
            assert rule.state_machine_id is not None
            machine = machine_by_id.get(rule.state_machine_id)
            if machine is None:
                transition_missing.append(rule.state_machine_id)
                continue
            if isinstance(request.current_state, Mapping):
                supplied_state = request.current_state.get(machine.state_machine_id)
                if supplied_state is None:
                    supplied_state = request.current_state.get(machine.resource_id)
            else:
                supplied_state = request.current_state
            if isinstance(request.state_version, Mapping):
                supplied_version = request.state_version.get(machine.state_machine_id)
                if supplied_version is None:
                    supplied_version = request.state_version.get(machine.resource_id)
            else:
                supplied_version = request.state_version
            expected_state = rule.from_state or machine.current_state
            if (
                supplied_state != machine.current_state
                or supplied_state != expected_state
                or _plain(supplied_version) != _plain(machine.state_version)
            ):
                stale.append(machine.state_machine_id)
                continue
            candidates = tuple(
                transition
                for transition in machine.transitions
                if transition.from_state == supplied_state
                and transition.action == request.action
                and transition.tool in (None, request.tool)
                and transition.target in (None, request.target)
                and (
                    transition.expected_effect is None
                    or _plain(transition.expected_effect)
                    == _plain(request.expected_effect)
                )
                and (
                    rule.to_state is None or transition.to_state == rule.to_state
                )
            )
            if not candidates:
                transition_missing.append(machine.state_machine_id)
            for transition in candidates:
                guard_missing.update(
                    set(transition.guard_assumption_ids)
                    - set(request.satisfied_assumption_ids)
                )
        if stale:
            state_status = SecurityCheckStatus.FAIL
            state_reason = "stale_state"
            state_ids = tuple(stale)
        elif transition_missing:
            state_status = SecurityCheckStatus.FAIL
            state_reason = "state_transition_not_allowed"
            state_ids = tuple(transition_missing)
        elif guard_missing:
            state_status = SecurityCheckStatus.UNKNOWN
            state_reason = "state_guard_unsatisfied"
            state_ids = tuple(sorted(guard_missing))
        else:
            state_status = SecurityCheckStatus.PASS
            state_reason = "state_guards_and_transitions_satisfied"
            state_ids = ()
        add(
            "state_machine",
            state_status,
            state_reason,
            declaration_ids=state_ids,
        )

        reference_decision: AuthorizationDecision | None = None
        if (
            policy.authorization_policy is not None
            and bool(allow_rules)
            and all(item.status is SecurityCheckStatus.PASS for item in checks)
        ):
            selected_allow = sorted(
                allow_rules, key=lambda item: (-item.priority, item.policy_id)
            )[0]
            reference_decision = evaluate_authorization(
                policy.authorization_policy,
                AuthorizationRequest(
                    principal=request.principal,
                    capability=Capability.CLAIM_TASK,
                    task_id=selected_allow.policy_id,
                    evaluated_at_ms=request.evaluated_at_ms,
                ),
            )
        if reference_decision is None:
            if any(item.status is SecurityCheckStatus.CONFLICT for item in checks):
                reference_status = SecurityCheckStatus.CONFLICT
                reference_reason = "reference_authorization_blocked_by_conflict"
            elif any(item.status is SecurityCheckStatus.UNKNOWN for item in checks) and not any(
                item.status is SecurityCheckStatus.FAIL for item in checks
            ):
                reference_status = SecurityCheckStatus.UNKNOWN
                reference_reason = "reference_authorization_blocked_by_unknown"
            else:
                reference_status = SecurityCheckStatus.FAIL
                reference_reason = "reference_authorization_not_reached"
        elif reference_decision.verdict is AuthorizationVerdict.PERMIT:
            reference_status = SecurityCheckStatus.PASS
            reference_reason = "reference_authorization_permit"
        else:
            reference_status = SecurityCheckStatus.FAIL
            reference_reason = f"reference_{reference_decision.reason.value}"
        add(
            "reference_authorization",
            reference_status,
            reference_reason,
            policy_ids=tuple(item.policy_id for item in allow_rules),
        )

        if any(item.status is SecurityCheckStatus.CONFLICT for item in checks):
            outcome = SecurityDecisionOutcome.CONFLICT
        elif any(item.status is SecurityCheckStatus.FAIL for item in checks):
            outcome = SecurityDecisionOutcome.DENY
        elif all(item.status is SecurityCheckStatus.PASS for item in checks):
            outcome = SecurityDecisionOutcome.PERMIT
        elif any(item.status is SecurityCheckStatus.UNKNOWN for item in checks):
            outcome = SecurityDecisionOutcome.UNKNOWN
        else:
            outcome = SecurityDecisionOutcome.DENY
        return SecurityDecisionReceipt(
            outcome=outcome,
            policy_receipt_id=policy.content_id,
            request_id=request.content_id,
            security_root_artifact_id=policy.security_root_artifact_id,
            security_root_cid_v1=policy.security_root_cid_v1,
            security_root_supervisor_digest=policy.security_root_supervisor_digest,
            evaluated_at_ms=request.evaluated_at_ms,
            checks=tuple(checks),
            matched_policy_ids=tuple(item.policy_id for item in matching),
            authorization_decision=reference_decision,
            reason_codes=tuple(sorted(reasons)),
        )

    authorize = evaluate
    check = evaluate


def compile_security_constraints(
    artifact: NormalizedIRArtifact | IRAdapterResult | None,
) -> SecurityPolicyReceipt:
    """Compile a pinned, normalized SecurityIR artifact."""

    return SecurityConstraintAdapter().compile(artifact)


def evaluate_security_authorization(
    policy: SecurityPolicyReceipt,
    request: SecurityAuthorizationRequest,
) -> SecurityDecisionReceipt:
    """Evaluate one exact request against a compiled SecurityIR policy."""

    return SecurityConstraintAdapter().evaluate(policy, request)


def revalidate_security_authorization(
    policy: SecurityPolicyReceipt,
    request: SecurityAuthorizationRequest,
    receipt: SecurityDecisionReceipt,
) -> SecurityDecisionReceipt:
    """Re-evaluate and authenticate a previously emitted decision receipt.

    Security decisions are inputs to later admission and permit boundaries,
    not bearer grants.  Consumers at those boundaries must not trust a caller
    supplied ``PERMIT`` projection: they replay the deterministic evaluator
    against the current policy and exact request and require the complete
    receipt identity to remain unchanged.
    """

    if not isinstance(receipt, SecurityDecisionReceipt):
        raise SecurityConstraintError(
            "receipt must be a SecurityDecisionReceipt"
        )
    current = evaluate_security_authorization(policy, request)
    if current != receipt or current.content_id != receipt.content_id:
        raise SecurityConstraintError(
            "security decision receipt is stale, forged, or detached"
        )
    return current


compile_security_policy = compile_security_constraints
check_security_authorization = evaluate_security_authorization
verify_security_authorization = revalidate_security_authorization


def authorize_security_action(
    artifact: NormalizedIRArtifact | IRAdapterResult | None,
    request: SecurityAuthorizationRequest,
) -> SecurityDecisionReceipt:
    """Compile and evaluate without weakening either receipt boundary."""

    return evaluate_security_authorization(
        compile_security_constraints(artifact), request
    )


__all__ = [
    "CompiledSecurityObligation",
    "CompiledSecurityPolicy",
    "CompiledSecurityStateMachine",
    "SECURITY_AUTHORIZATION_REQUEST_SCHEMA",
    "SECURITY_CONSTRAINT_ADAPTER_VERSION",
    "SECURITY_DECISION_CHECK_SCHEMA",
    "SECURITY_DECISION_RECEIPT_SCHEMA",
    "SECURITY_DECLARATION_SCHEMA",
    "SECURITY_OBLIGATION_SCHEMA",
    "SECURITY_POLICY_RECEIPT_SCHEMA",
    "SECURITY_POLICY_RULE_SCHEMA",
    "SECURITY_SOURCE_BINDING_SCHEMA",
    "SECURITY_STATE_MACHINE_SCHEMA",
    "SecurityAuthorizationDecision",
    "SecurityAuthorizationQuery",
    "SecurityAuthorizationRequest",
    "SecurityAuthorizationOutcome",
    "SecurityCheckStatus",
    "SecurityCompilationResult",
    "SecurityCompilationStatus",
    "SecurityConstraintAdapter",
    "SecurityConstraintError",
    "SecurityDecisionCheck",
    "SecurityDecisionOutcome",
    "SecurityDecisionReceipt",
    "SecurityDeclaration",
    "SecurityPolicyReceipt",
    "SecurityPolicyCompilationResult",
    "SecurityPolicyRule",
    "SecurityRuleEffect",
    "SecuritySourceBinding",
    "SecurityTransition",
    "authorize_security_action",
    "check_security_authorization",
    "compile_security_policy",
    "compile_security_constraints",
    "evaluate_security_authorization",
    "revalidate_security_authorization",
    "verify_security_authorization",
]
