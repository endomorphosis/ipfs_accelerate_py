"""Exact IntentIR/generated-code authorization mapping and correlation.

This module sits between the non-authoritative intent/code observation
contracts and :mod:`security_constraint_adapter`.  It has three jobs:

* project each pinned intent action and each added-code fact group into an
  exact :class:`SecurityAuthorizationRequest`;
* retain incomplete or ambiguous projections as explicit ``unknown`` records;
* evaluate and correlate the two streams without allowing a successful intent
  decision to mask undeclared, broadened, contradictory, or rejected code.

No fuzzy retrieval, string similarity, or model output participates in this
boundary.  Runtime bindings are explicit, all requests carry the same pinned
Security IR root, and only the existing security adapter evaluates policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .code_security_facts import (
    CodeSecurityDelta,
    CodeSecurityExtractionStatus,
    CodeSecurityFact,
    CodeSecurityFactKind,
    CodeSecurityFactSet,
)
from .intent_constraint_adapter import (
    IntentCompilationStatus,
    IntentConstraintCompilationResult,
    IntentConstraintKind,
    IntentConstraintSet,
)
from .security_constraint_adapter import (
    SecurityAuthorizationRequest,
    SecurityDecisionOutcome,
    SecurityDecisionReceipt,
    SecurityPolicyReceipt,
    evaluate_security_authorization,
)


CVE_SECURITY_GATE_VERSION: Final[int] = 1
CVE_SECURITY_REQUEST_CONTEXT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-request-context@1"
)
CVE_SECURITY_REQUEST_MAPPING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-request-mapping@1"
)
CVE_SECURITY_CORRELATION_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-correlation-finding@1"
)
CVE_SECURITY_GATE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-gate-result@1"
)

_MAX_TEXT_BYTES: Final[int] = 8_192
_MAX_MAPPINGS: Final[int] = 100_000
_MISSING: Final[object] = object()


class CVESecurityGateError(ValueError):
    """A gate input or canonical contract is malformed."""


class SecurityFactStream(str, Enum):
    """The independent observation stream that produced a request."""

    INTENT = "intent"
    CODE = "code"


class SecurityRequestMappingStatus(str, Enum):
    """Whether an observation has one lossless authorization projection."""

    EXACT = "exact"
    UNKNOWN = "unknown"


class CVESecurityGateOutcome(str, Enum):
    """Aggregate outcome; this record itself never grants execution."""

    PASS = "pass"
    REJECT = "reject"
    UNKNOWN = "unknown"


class CVESecurityGateFindingCode(str, Enum):
    """Stable fail-closed mapping, correlation, and evaluation reasons."""

    EMPTY_INTENT_STREAM = "empty_intent_stream"
    EMPTY_CODE_STREAM = "empty_code_stream"
    INCOMPLETE_INTENT_MAPPING = "incomplete_intent_mapping"
    AMBIGUOUS_INTENT_MAPPING = "ambiguous_intent_mapping"
    INCOMPLETE_CODE_MAPPING = "incomplete_code_mapping"
    AMBIGUOUS_CODE_MAPPING = "ambiguous_code_mapping"
    UNSUPPORTED_INTENT = "unsupported_intent"
    UNSUPPORTED_CODE = "unsupported_code"
    UNDECLARED_CODE_EFFECT = "undeclared_code_effect"
    BROADENED_CODE_EFFECT = "broadened_code_effect"
    CONTRADICTORY_CODE_EFFECT = "contradictory_code_effect"
    INTENT_SECURITY_REJECTED = "intent_security_rejected"
    CODE_SECURITY_REJECTED = "code_security_rejected"
    SECURITY_DECISION_UNKNOWN = "security_decision_unknown"


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CVESecurityGateError("value is not canonical JSON") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(_freeze(item) for item in value)
    return value


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise CVESecurityGateError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise CVESecurityGateError(
            f"{name} must be non-empty without surrounding whitespace or NUL"
        )
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise CVESecurityGateError(f"{name} exceeds its UTF-8 byte bound")
    return value


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    return tuple(sorted({_text(item, name) for item in values}))


def _same(left: Any, right: Any) -> bool:
    return _canonical_bytes(_plain(left)) == _canonical_bytes(_plain(right))


def _select(
    record: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    fallback: Any = _MISSING,
) -> tuple[Any, bool]:
    """Return one alias value and whether contradictory aliases were present."""

    values = [record[name] for name in aliases if record.get(name) not in (None, "")]
    if not values:
        return fallback, False
    first = values[0]
    return first, any(not _same(first, item) for item in values[1:])


@dataclass(frozen=True)
class SecurityRequestContext:
    """Explicit root and runtime facts shared by both observation streams."""

    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    principal: str
    tool: str
    current_state: Any
    requested_authority: str
    evaluated_at_ms: int
    state_version: Any = None
    source_zone: str | None = None
    channel: str | None = None
    target_zone: str | None = None
    satisfied_assumption_ids: tuple[str, ...] = ()
    accepted_claim_result_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "security_root_artifact_id",
            "security_root_cid_v1",
            "security_root_supervisor_digest",
            "principal",
            "tool",
            "requested_authority",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        # Reuse the canonical request contract to validate all nested state,
        # timestamps, zones, and identifiers without maintaining a second
        # interpretation of exact Security IR inputs.
        probe = SecurityAuthorizationRequest(
            security_root_artifact_id=self.security_root_artifact_id,
            security_root_cid_v1=self.security_root_cid_v1,
            security_root_supervisor_digest=self.security_root_supervisor_digest,
            principal=self.principal,
            action="cve-security-context-validation",
            tool=self.tool,
            target="cve-security-context-validation",
            data_flow={"binding": "context-validation"},
            expected_effect={"binding": "context-validation"},
            current_state=self.current_state,
            state_version=self.state_version,
            requested_authority=self.requested_authority,
            evaluated_at_ms=self.evaluated_at_ms,
            source_zone=self.source_zone,
            channel=self.channel,
            target_zone=self.target_zone,
            satisfied_assumption_ids=self.satisfied_assumption_ids,
            accepted_claim_result_ids=self.accepted_claim_result_ids,
        )
        for name in (
            "current_state",
            "state_version",
            "source_zone",
            "channel",
            "target_zone",
            "satisfied_assumption_ids",
            "accepted_claim_result_ids",
            "evaluated_at_ms",
        ):
            object.__setattr__(self, name, getattr(probe, name))

    @classmethod
    def from_policy(
        cls,
        policy: SecurityPolicyReceipt,
        *,
        principal: str,
        tool: str,
        current_state: Any,
        requested_authority: str,
        evaluated_at_ms: int,
        state_version: Any = None,
        source_zone: str | None = None,
        channel: str | None = None,
        target_zone: str | None = None,
        satisfied_assumption_ids: Sequence[str] = (),
        accepted_claim_result_ids: Sequence[str] = (),
    ) -> "SecurityRequestContext":
        if not isinstance(policy, SecurityPolicyReceipt):
            raise CVESecurityGateError("policy must be a SecurityPolicyReceipt")
        return cls(
            security_root_artifact_id=policy.security_root_artifact_id,
            security_root_cid_v1=policy.security_root_cid_v1,
            security_root_supervisor_digest=policy.security_root_supervisor_digest,
            principal=principal,
            tool=tool,
            current_state=current_state,
            state_version=state_version,
            requested_authority=requested_authority,
            evaluated_at_ms=evaluated_at_ms,
            source_zone=source_zone,
            channel=channel,
            target_zone=target_zone,
            satisfied_assumption_ids=tuple(satisfied_assumption_ids),
            accepted_claim_result_ids=tuple(accepted_claim_result_ids),
        )

    def request(
        self,
        *,
        action: str,
        target: str,
        data_flow: Any,
        expected_effect: Any,
        principal: str | None = None,
        tool: str | None = None,
        current_state: Any = _MISSING,
        requested_authority: str | None = None,
    ) -> SecurityAuthorizationRequest:
        return SecurityAuthorizationRequest(
            security_root_artifact_id=self.security_root_artifact_id,
            security_root_cid_v1=self.security_root_cid_v1,
            security_root_supervisor_digest=self.security_root_supervisor_digest,
            principal=principal or self.principal,
            action=action,
            tool=tool or self.tool,
            target=target,
            data_flow=data_flow,
            expected_effect=expected_effect,
            current_state=(
                self.current_state if current_state is _MISSING else current_state
            ),
            state_version=self.state_version,
            requested_authority=requested_authority or self.requested_authority,
            evaluated_at_ms=self.evaluated_at_ms,
            source_zone=self.source_zone,
            channel=self.channel,
            target_zone=self.target_zone,
            satisfied_assumption_ids=self.satisfied_assumption_ids,
            accepted_claim_result_ids=self.accepted_claim_result_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_REQUEST_CONTEXT_SCHEMA,
            "gate_version": CVE_SECURITY_GATE_VERSION,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            "principal": self.principal,
            "tool": self.tool,
            "current_state": _plain(self.current_state),
            "state_version": _plain(self.state_version),
            "requested_authority": self.requested_authority,
            "evaluated_at_ms": self.evaluated_at_ms,
            "source_zone": self.source_zone,
            "channel": self.channel,
            "target_zone": self.target_zone,
            "satisfied_assumption_ids": list(self.satisfied_assumption_ids),
            "accepted_claim_result_ids": list(self.accepted_claim_result_ids),
        }


@dataclass(frozen=True)
class SecurityRequestMapping:
    """One exact request or one explicit, evidence-bound unknown projection."""

    stream: SecurityFactStream
    source_id: str
    status: SecurityRequestMappingStatus
    request: SecurityAuthorizationRequest | None = None
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[CVESecurityGateFindingCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream", SecurityFactStream(self.stream))
        object.__setattr__(
            self, "status", SecurityRequestMappingStatus(self.status)
        )
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(
            self, "evidence_ids", _strings(self.evidence_ids, "evidence_id")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                sorted(
                    {CVESecurityGateFindingCode(item) for item in self.reason_codes},
                    key=lambda item: item.value,
                )
            ),
        )
        if self.status is SecurityRequestMappingStatus.EXACT:
            if not isinstance(self.request, SecurityAuthorizationRequest):
                raise CVESecurityGateError("exact mapping requires a canonical request")
            if self.reason_codes:
                raise CVESecurityGateError("exact mapping cannot carry failure reasons")
        elif self.request is not None:
            raise CVESecurityGateError("unknown mapping cannot carry a request")
        elif not self.reason_codes:
            raise CVESecurityGateError("unknown mapping requires a reason")

    @property
    def exact(self) -> bool:
        return self.status is SecurityRequestMappingStatus.EXACT

    @property
    def mapping_id(self) -> str:
        return _identity("cve-security-request-mapping", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_REQUEST_MAPPING_SCHEMA,
            "gate_version": CVE_SECURITY_GATE_VERSION,
            "stream": self.stream.value,
            "source_id": self.source_id,
            "status": self.status.value,
            "request": self.request.to_dict() if self.request else None,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": [item.value for item in self.reason_codes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"mapping_id": self.mapping_id, **self._payload()}


@dataclass(frozen=True)
class SecurityCorrelationFinding:
    """A deterministic relationship between intent and generated-code facts."""

    code: CVESecurityGateFindingCode
    code_mapping_ids: tuple[str, ...] = ()
    intent_mapping_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", CVESecurityGateFindingCode(self.code))
        object.__setattr__(
            self,
            "code_mapping_ids",
            _strings(self.code_mapping_ids, "code_mapping_id"),
        )
        object.__setattr__(
            self,
            "intent_mapping_ids",
            _strings(self.intent_mapping_ids, "intent_mapping_id"),
        )
        if not isinstance(self.details, Mapping):
            raise CVESecurityGateError("finding details must be an object")
        object.__setattr__(self, "details", _freeze(dict(self.details)))

    @property
    def finding_id(self) -> str:
        return _identity("cve-security-correlation-finding", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_CORRELATION_FINDING_SCHEMA,
            "gate_version": CVE_SECURITY_GATE_VERSION,
            "code": self.code.value,
            "code_mapping_ids": list(self.code_mapping_ids),
            "intent_mapping_ids": list(self.intent_mapping_ids),
            "details": _plain(self.details),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, **self._payload()}


@dataclass(frozen=True)
class SecurityMappedDecision:
    """Existing-adapter decision tied back to its observation mapping."""

    mapping_id: str
    stream: SecurityFactStream
    decision: SecurityDecisionReceipt

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping_id", _text(self.mapping_id, "mapping_id"))
        object.__setattr__(self, "stream", SecurityFactStream(self.stream))
        if not isinstance(self.decision, SecurityDecisionReceipt):
            raise CVESecurityGateError("decision must be a SecurityDecisionReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping_id": self.mapping_id,
            "stream": self.stream.value,
            "decision": self.decision.to_dict(),
        }


def _unknown(
    stream: SecurityFactStream,
    source_id: str,
    code: CVESecurityGateFindingCode,
    evidence_ids: Sequence[str] = (),
) -> SecurityRequestMapping:
    return SecurityRequestMapping(
        stream=stream,
        source_id=source_id,
        status=SecurityRequestMappingStatus.UNKNOWN,
        evidence_ids=tuple(evidence_ids),
        reason_codes=(code,),
    )


def _intent_constraints(
    value: IntentConstraintSet | IntentConstraintCompilationResult,
) -> tuple[IntentConstraintSet | None, CVESecurityGateFindingCode | None]:
    if isinstance(value, IntentConstraintSet):
        constraints = value
    elif isinstance(value, IntentConstraintCompilationResult):
        if (
            value.status is not IntentCompilationStatus.COMPILED
            or value.constraint_set is None
        ):
            return None, CVESecurityGateFindingCode.UNSUPPORTED_INTENT
        constraints = value.constraint_set
    else:
        raise CVESecurityGateError(
            "intent must be an IntentConstraintSet or compilation result"
        )
    if (
        constraints.unsupported_node_ids
        or constraints.contradictory_effect_groups
        or constraints.graph_truncated
    ):
        return None, CVESecurityGateFindingCode.UNSUPPORTED_INTENT
    return constraints, None


def _effect_for_action(
    constraints: IntentConstraintSet, action_id: str
) -> tuple[Any, tuple[str, ...], bool]:
    effects = [
        item
        for item in constraints.effects
        if action_id in item.action_ids and item.required and not item.context_only
    ]
    if not effects:
        return _MISSING, (), False
    values = []
    for item in effects:
        expression = dict(item.expression)
        explicit, explicit_conflict = _select(
            expression,
            ("expected_effect", "result_effect", "effect_manifest"),
            fallback=_MISSING,
        )
        if explicit_conflict:
            return _MISSING, tuple(item.constraint_id for item in effects), True
        if explicit is not _MISSING:
            values.append(explicit)
            continue
        semantic = {
            key: expression[key]
            for key in ("operation", "fluent_id", "event_id", "target", "value")
            if key in expression
        }
        values.append(semantic if semantic else _MISSING)
    if any(item is _MISSING for item in values):
        return _MISSING, tuple(item.constraint_id for item in effects), False
    ambiguous = any(not _same(values[0], item) for item in values[1:])
    return values[0], tuple(item.constraint_id for item in effects), ambiguous


def map_intent_security_requests(
    intent: IntentConstraintSet | IntentConstraintCompilationResult,
    context: SecurityRequestContext,
) -> tuple[SecurityRequestMapping, ...]:
    """Map compiled, pinned IntentIR action facts to exact security requests."""

    if not isinstance(context, SecurityRequestContext):
        raise CVESecurityGateError("context must be a SecurityRequestContext")
    constraints, failure = _intent_constraints(intent)
    if constraints is None:
        source_id = (
            intent.compilation_id
            if isinstance(intent, IntentConstraintCompilationResult)
            else intent.constraint_set_id
        )
        return (_unknown(SecurityFactStream.INTENT, source_id, failure),)  # type: ignore[arg-type]

    mappings: list[SecurityRequestMapping] = []
    for action_constraint in constraints.actions:
        expression = dict(action_constraint.expression)
        action, action_conflict = _select(
            expression, ("action", "operation"), fallback=_MISSING
        )
        if action is _MISSING and len(action_constraint.action_ids) == 1:
            action = action_constraint.action_ids[0]
        target, target_conflict = _select(
            expression, ("target", "resource", "resource_id"), fallback=_MISSING
        )
        data_flow, flow_conflict = _select(
            expression, ("data_flow", "flow"), fallback=_MISSING
        )
        expected_effect, effect_conflict = _select(
            expression,
            ("expected_effect", "result_effect", "effect_manifest"),
            fallback=_MISSING,
        )
        linked_ids: tuple[str, ...] = ()
        if expected_effect is _MISSING:
            expected_effect, linked_ids, linked_conflict = _effect_for_action(
                constraints,
                action_constraint.action_ids[0]
                if len(action_constraint.action_ids) == 1
                else "",
            )
            effect_conflict = effect_conflict or linked_conflict
        principal, principal_conflict = _select(
            expression, ("principal", "subject"), fallback=context.principal
        )
        tool, tool_conflict = _select(
            expression, ("tool", "tool_id"), fallback=context.tool
        )
        current_state, state_conflict = _select(
            expression,
            ("current_state", "state"),
            fallback=context.current_state,
        )
        authority, authority_conflict = _select(
            expression,
            ("requested_authority", "authority", "authority_class"),
            fallback=context.requested_authority,
        )
        evidence_ids = (
            action_constraint.constraint_id,
            *action_constraint.source_binding_ids,
            *linked_ids,
        )
        if (
            len(action_constraint.action_ids) != 1
            or action_conflict
            or target_conflict
            or flow_conflict
            or effect_conflict
            or principal_conflict
            or tool_conflict
            or state_conflict
            or authority_conflict
        ):
            mappings.append(
                _unknown(
                    SecurityFactStream.INTENT,
                    action_constraint.constraint_id,
                    CVESecurityGateFindingCode.AMBIGUOUS_INTENT_MAPPING,
                    evidence_ids,
                )
            )
            continue
        if any(
            item is _MISSING
            for item in (action, target, data_flow, expected_effect)
        ):
            mappings.append(
                _unknown(
                    SecurityFactStream.INTENT,
                    action_constraint.constraint_id,
                    CVESecurityGateFindingCode.INCOMPLETE_INTENT_MAPPING,
                    evidence_ids,
                )
            )
            continue
        try:
            request = context.request(
                principal=principal,
                action=action,
                tool=tool,
                target=target,
                data_flow=data_flow,
                expected_effect=expected_effect,
                current_state=current_state,
                requested_authority=authority,
            )
        except (TypeError, ValueError):
            mappings.append(
                _unknown(
                    SecurityFactStream.INTENT,
                    action_constraint.constraint_id,
                    CVESecurityGateFindingCode.INCOMPLETE_INTENT_MAPPING,
                    evidence_ids,
                )
            )
            continue
        mappings.append(
            SecurityRequestMapping(
                stream=SecurityFactStream.INTENT,
                source_id=action_constraint.constraint_id,
                status=SecurityRequestMappingStatus.EXACT,
                request=request,
                evidence_ids=evidence_ids,
            )
        )
    if not mappings:
        mappings.append(
            _unknown(
                SecurityFactStream.INTENT,
                constraints.constraint_set_id,
                CVESecurityGateFindingCode.EMPTY_INTENT_STREAM,
            )
        )
    if len(mappings) > _MAX_MAPPINGS:
        raise CVESecurityGateError("intent mappings exceed the hard count bound")
    return tuple(sorted(mappings, key=lambda item: item.mapping_id))


def _code_group_key(fact: CodeSecurityFact) -> tuple[str, str]:
    return fact.binding.binding_id, fact.source_scope.scope_id


def map_code_security_requests(
    facts: CodeSecurityFactSet,
    context: SecurityRequestContext,
) -> tuple[SecurityRequestMapping, ...]:
    """Map added-code fact groups to exact security requests.

    Facts are correlated only when they share the exact source binding and AST
    scope emitted by the extractor.  Removed-side observations do not describe
    candidate execution and are therefore retained outside the request stream.
    """

    if not isinstance(facts, CodeSecurityFactSet):
        raise CVESecurityGateError("facts must be a CodeSecurityFactSet")
    if not isinstance(context, SecurityRequestContext):
        raise CVESecurityGateError("context must be a SecurityRequestContext")

    mappings: list[SecurityRequestMapping] = []
    if facts.status is not CodeSecurityExtractionStatus.EXTRACTED:
        code = (
            CVESecurityGateFindingCode.AMBIGUOUS_CODE_MAPPING
            if facts.ambiguous
            else CVESecurityGateFindingCode.UNSUPPORTED_CODE
        )
        mappings.append(
            _unknown(
                SecurityFactStream.CODE,
                facts.fact_set_id,
                code,
                tuple(item.diagnostic_id for item in facts.diagnostics),
            )
        )

    groups: dict[tuple[str, str], list[CodeSecurityFact]] = defaultdict(list)
    for fact in facts.facts:
        if fact.source_scope.delta is CodeSecurityDelta.ADDED:
            groups[_code_group_key(fact)].append(fact)

    dimensions = {
        "action": CodeSecurityFactKind.ACTION,
        "target": CodeSecurityFactKind.TARGET,
        "data_flow": CodeSecurityFactKind.DATA_FLOW,
        "expected_effect": CodeSecurityFactKind.EFFECT,
    }
    for (binding_id, scope_id), group in sorted(groups.items()):
        values = {
            name: tuple(
                sorted({item.value for item in group if item.kind is kind})
            )
            for name, kind in dimensions.items()
        }
        # LANGUAGE, SOURCE_SCOPE, GUARD, and CAPABILITY facts qualify an
        # action observation but do not independently describe an action.
        # Extractors intentionally emit some of them on enclosing AST scopes.
        if not any(values.values()):
            continue
        evidence_ids = tuple(item.fact_id for item in group)
        source_id = _identity(
            "code-security-request-source",
            {"binding_id": binding_id, "scope_id": scope_id},
        )
        if any(len(items) > 1 for items in values.values()):
            mappings.append(
                _unknown(
                    SecurityFactStream.CODE,
                    source_id,
                    CVESecurityGateFindingCode.AMBIGUOUS_CODE_MAPPING,
                    evidence_ids,
                )
            )
            continue
        if any(len(items) != 1 for items in values.values()):
            mappings.append(
                _unknown(
                    SecurityFactStream.CODE,
                    source_id,
                    CVESecurityGateFindingCode.INCOMPLETE_CODE_MAPPING,
                    evidence_ids,
                )
            )
            continue
        request = context.request(
            action=values["action"][0],
            target=values["target"][0],
            data_flow=values["data_flow"][0],
            expected_effect=values["expected_effect"][0],
        )
        mappings.append(
            SecurityRequestMapping(
                stream=SecurityFactStream.CODE,
                source_id=source_id,
                status=SecurityRequestMappingStatus.EXACT,
                request=request,
                evidence_ids=evidence_ids,
            )
        )

    if not mappings:
        mappings.append(
            _unknown(
                SecurityFactStream.CODE,
                facts.fact_set_id,
                CVESecurityGateFindingCode.EMPTY_CODE_STREAM,
            )
        )
    if len(mappings) > _MAX_MAPPINGS:
        raise CVESecurityGateError("code mappings exceed the hard count bound")
    return tuple(sorted(mappings, key=lambda item: item.mapping_id))


def _request_dimensions(
    mapping: SecurityRequestMapping,
) -> tuple[str, str, bytes, bytes]:
    assert mapping.request is not None
    request = mapping.request
    return (
        request.action,
        request.target,
        _canonical_bytes(_plain(request.data_flow)),
        _canonical_bytes(_plain(request.expected_effect)),
    )


def correlate_security_requests(
    intent_mappings: Sequence[SecurityRequestMapping],
    code_mappings: Sequence[SecurityRequestMapping],
) -> tuple[SecurityCorrelationFinding, ...]:
    """Correlate exact code effects against exact intent declarations."""

    intent = tuple(intent_mappings)
    code = tuple(code_mappings)
    if any(item.stream is not SecurityFactStream.INTENT for item in intent):
        raise CVESecurityGateError("intent mappings contain a non-intent stream")
    if any(item.stream is not SecurityFactStream.CODE for item in code):
        raise CVESecurityGateError("code mappings contain a non-code stream")

    findings: list[SecurityCorrelationFinding] = []
    exact_intent = tuple(item for item in intent if item.exact)
    for mapping in (*intent, *code):
        if mapping.exact:
            continue
        findings.append(
            SecurityCorrelationFinding(
                code=mapping.reason_codes[0],
                code_mapping_ids=(
                    (mapping.mapping_id,)
                    if mapping.stream is SecurityFactStream.CODE
                    else ()
                ),
                intent_mapping_ids=(
                    (mapping.mapping_id,)
                    if mapping.stream is SecurityFactStream.INTENT
                    else ()
                ),
            )
        )

    intent_dimensions = {
        item.mapping_id: _request_dimensions(item) for item in exact_intent
    }
    for code_mapping in (item for item in code if item.exact):
        code_action, code_target, code_flow, code_effect = _request_dimensions(
            code_mapping
        )
        exact_matches = [
            item_id
            for item_id, dimensions in intent_dimensions.items()
            if dimensions
            == (code_action, code_target, code_flow, code_effect)
        ]
        if exact_matches:
            continue
        same_action = [
            item_id
            for item_id, dimensions in intent_dimensions.items()
            if dimensions[0] == code_action
        ]
        same_scope = [
            item_id
            for item_id in same_action
            if intent_dimensions[item_id][1:3] == (code_target, code_flow)
        ]
        if same_scope:
            finding_code = CVESecurityGateFindingCode.CONTRADICTORY_CODE_EFFECT
            related = same_scope
        elif same_action:
            finding_code = CVESecurityGateFindingCode.BROADENED_CODE_EFFECT
            related = same_action
        else:
            finding_code = CVESecurityGateFindingCode.UNDECLARED_CODE_EFFECT
            related = []
        findings.append(
            SecurityCorrelationFinding(
                code=finding_code,
                code_mapping_ids=(code_mapping.mapping_id,),
                intent_mapping_ids=tuple(related),
                details={
                    "action": code_action,
                    "target": code_target,
                    "data_flow": _plain(code_mapping.request.data_flow),  # type: ignore[union-attr]
                    "expected_effect": _plain(
                        code_mapping.request.expected_effect  # type: ignore[union-attr]
                    ),
                },
            )
        )
    unique = {item.finding_id: item for item in findings}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class CVESecurityGateResult:
    """Bound aggregate of mappings, correlations, and existing decisions."""

    outcome: CVESecurityGateOutcome
    policy_receipt_id: str
    context: SecurityRequestContext
    intent_mappings: tuple[SecurityRequestMapping, ...]
    code_mappings: tuple[SecurityRequestMapping, ...]
    decisions: tuple[SecurityMappedDecision, ...]
    findings: tuple[SecurityCorrelationFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", CVESecurityGateOutcome(self.outcome))
        object.__setattr__(
            self, "policy_receipt_id", _text(self.policy_receipt_id, "policy_receipt_id")
        )
        if not isinstance(self.context, SecurityRequestContext):
            raise CVESecurityGateError("context must be canonical")
        for name, values, stream in (
            ("intent_mappings", self.intent_mappings, SecurityFactStream.INTENT),
            ("code_mappings", self.code_mappings, SecurityFactStream.CODE),
        ):
            if any(
                not isinstance(item, SecurityRequestMapping)
                or item.stream is not stream
                for item in values
            ):
                raise CVESecurityGateError(f"{name} contains an invalid mapping")
            object.__setattr__(
                self, name, tuple(sorted(values, key=lambda item: item.mapping_id))
            )
        if any(not isinstance(item, SecurityMappedDecision) for item in self.decisions):
            raise CVESecurityGateError("decisions contain an invalid record")
        if any(
            not isinstance(item, SecurityCorrelationFinding)
            for item in self.findings
        ):
            raise CVESecurityGateError("findings contain an invalid record")
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(self.decisions, key=lambda item: item.mapping_id)),
        )
        object.__setattr__(
            self,
            "findings",
            tuple(sorted(self.findings, key=lambda item: item.finding_id)),
        )

    @property
    def passed(self) -> bool:
        return self.outcome is CVESecurityGateOutcome.PASS

    @property
    def fail_closed(self) -> bool:
        return not self.passed

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def gate_id(self) -> str:
        return _identity("cve-security-gate-result", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_GATE_RESULT_SCHEMA,
            "gate_version": CVE_SECURITY_GATE_VERSION,
            "outcome": self.outcome.value,
            "policy_receipt_id": self.policy_receipt_id,
            "context": self.context.to_dict(),
            "intent_mappings": [item.to_dict() for item in self.intent_mappings],
            "code_mappings": [item.to_dict() for item in self.code_mappings],
            "decisions": [item.to_dict() for item in self.decisions],
            "findings": [item.to_dict() for item in self.findings],
            "intent_pass_cannot_mask_code_fail": True,
            "grants_execution_authority": False,
            "authorizes_completion": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"gate_id": self.gate_id, **self._payload()}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


def evaluate_cve_security_gate(
    policy: SecurityPolicyReceipt,
    intent: IntentConstraintSet | IntentConstraintCompilationResult,
    code_facts: CodeSecurityFactSet,
    context: SecurityRequestContext,
) -> CVESecurityGateResult:
    """Map, correlate, and independently authorize both security fact streams."""

    if not isinstance(policy, SecurityPolicyReceipt):
        raise CVESecurityGateError("policy must be a SecurityPolicyReceipt")
    if not isinstance(context, SecurityRequestContext):
        raise CVESecurityGateError("context must be a SecurityRequestContext")
    policy_root = (
        policy.security_root_artifact_id,
        policy.security_root_cid_v1,
        policy.security_root_supervisor_digest,
    )
    context_root = (
        context.security_root_artifact_id,
        context.security_root_cid_v1,
        context.security_root_supervisor_digest,
    )
    if policy_root != context_root:
        raise CVESecurityGateError(
            "request context must bind the evaluated Security IR root"
        )

    intent_mappings = map_intent_security_requests(intent, context)
    code_mappings = map_code_security_requests(code_facts, context)
    findings = list(correlate_security_requests(intent_mappings, code_mappings))
    decisions: list[SecurityMappedDecision] = []
    rejected = False
    unknown = any(not item.exact for item in (*intent_mappings, *code_mappings))

    for mapping in (*intent_mappings, *code_mappings):
        if not mapping.exact:
            continue
        assert mapping.request is not None
        decision = evaluate_security_authorization(policy, mapping.request)
        decisions.append(
            SecurityMappedDecision(mapping.mapping_id, mapping.stream, decision)
        )
        if decision.outcome in {
            SecurityDecisionOutcome.DENY,
            SecurityDecisionOutcome.CONFLICT,
        }:
            rejected = True
            findings.append(
                SecurityCorrelationFinding(
                    code=(
                        CVESecurityGateFindingCode.INTENT_SECURITY_REJECTED
                        if mapping.stream is SecurityFactStream.INTENT
                        else CVESecurityGateFindingCode.CODE_SECURITY_REJECTED
                    ),
                    intent_mapping_ids=(
                        (mapping.mapping_id,)
                        if mapping.stream is SecurityFactStream.INTENT
                        else ()
                    ),
                    code_mapping_ids=(
                        (mapping.mapping_id,)
                        if mapping.stream is SecurityFactStream.CODE
                        else ()
                    ),
                    details={"decision_outcome": decision.outcome.value},
                )
            )
        elif decision.outcome is SecurityDecisionOutcome.UNKNOWN:
            unknown = True
            findings.append(
                SecurityCorrelationFinding(
                    code=CVESecurityGateFindingCode.SECURITY_DECISION_UNKNOWN,
                    intent_mapping_ids=(
                        (mapping.mapping_id,)
                        if mapping.stream is SecurityFactStream.INTENT
                        else ()
                    ),
                    code_mapping_ids=(
                        (mapping.mapping_id,)
                        if mapping.stream is SecurityFactStream.CODE
                        else ()
                    ),
                )
            )

    correlation_rejections = {
        CVESecurityGateFindingCode.UNDECLARED_CODE_EFFECT,
        CVESecurityGateFindingCode.BROADENED_CODE_EFFECT,
        CVESecurityGateFindingCode.CONTRADICTORY_CODE_EFFECT,
    }
    rejected = rejected or any(item.code in correlation_rejections for item in findings)
    outcome = (
        CVESecurityGateOutcome.REJECT
        if rejected
        else CVESecurityGateOutcome.UNKNOWN
        if unknown or not decisions
        else CVESecurityGateOutcome.PASS
    )
    unique_findings = {item.finding_id: item for item in findings}
    return CVESecurityGateResult(
        outcome=outcome,
        policy_receipt_id=policy.content_id,
        context=context,
        intent_mappings=intent_mappings,
        code_mappings=code_mappings,
        decisions=tuple(decisions),
        findings=tuple(unique_findings.values()),
    )


# Compatibility spellings for enforcement integration.
CVESecurityRequestContext = SecurityRequestContext
CVESecurityRequestMapping = SecurityRequestMapping
map_intent_to_security_requests = map_intent_security_requests
map_code_facts_to_security_requests = map_code_security_requests
correlate_intent_and_code_security = correlate_security_requests
run_cve_security_gate = evaluate_cve_security_gate


__all__ = [
    "CVE_SECURITY_CORRELATION_FINDING_SCHEMA",
    "CVE_SECURITY_GATE_RESULT_SCHEMA",
    "CVE_SECURITY_GATE_VERSION",
    "CVE_SECURITY_REQUEST_CONTEXT_SCHEMA",
    "CVE_SECURITY_REQUEST_MAPPING_SCHEMA",
    "CVESecurityGateError",
    "CVESecurityGateFindingCode",
    "CVESecurityGateOutcome",
    "CVESecurityGateResult",
    "CVESecurityRequestContext",
    "CVESecurityRequestMapping",
    "SecurityCorrelationFinding",
    "SecurityFactStream",
    "SecurityMappedDecision",
    "SecurityRequestContext",
    "SecurityRequestMapping",
    "SecurityRequestMappingStatus",
    "correlate_intent_and_code_security",
    "correlate_security_requests",
    "evaluate_cve_security_gate",
    "map_code_facts_to_security_requests",
    "map_code_security_requests",
    "map_intent_security_requests",
    "map_intent_to_security_requests",
    "run_cve_security_gate",
]
