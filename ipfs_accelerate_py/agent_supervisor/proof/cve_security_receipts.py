"""Bounded, redacted observability receipts for the CVE security gate.

The enforcement contracts retain complete, replayable gate evidence. Complete
evidence is intentionally too rich for logs and cache indexes: request values
and correlation details can contain generated code, data-flow values, or
operator-supplied text. This module projects that evidence into a small
content-addressed record containing identifiers and stable reason codes only.

An observability receipt is evidence about a decision, never the authority to
execute it. Execution authority continues to come exclusively from the
short-lived execution-permit boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .cve_security_gate import (
    CVESecurityGateOutcome,
    CVESecurityGateResult,
    SecurityFactStream,
)
from .ir_constraint_compiler import (
    CVESecurityEnforcementEvidence,
    CVESecurityEnforcementStage,
)


CVE_SECURITY_OBSERVABILITY_VERSION: Final[int] = 1
CVE_SECURITY_OBSERVABILITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "cve-security-decision-observability-receipt@1"
)
CVE_SECURITY_STREAM_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-stream-evidence@1"
)
CVE_SECURITY_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-counterexample-summary@1"
)

MAX_IDENTIFIERS_PER_FIELD: Final[int] = 256
MAX_COUNTEREXAMPLES: Final[int] = 256
MAX_IDENTIFIER_UTF8_BYTES: Final[int] = 1_024
MAX_RECEIPT_UTF8_BYTES: Final[int] = 64 * 1_024

_SECURITY_ROOT_KIND: Final[str] = "security_ir"
_REDACTED_FIELDS: Final[tuple[str, ...]] = (
    "authorization_decision",
    "code_body",
    "correlation_finding.details",
    "request.current_state",
    "request.data_flow",
    "request.expected_effect",
    "secret",
)
_SENSITIVE_VALUE = re.compile(
    r"""(?ix)
    -----BEGIN[ ]+(?:RSA[ ]+|EC[ ]+|OPENSSH[ ]+)?PRIVATE[ ]+KEY-----
    | \b(?:api[_-]?key|authorization|bearer|passwd|password|secret)
      \s*[:=]\s*\S+
    | \bsk-[a-z0-9_-]{16,}
    | \bgh[pousr]_[a-z0-9]{16,}
    | ://[^/\s:@]+:[^/\s@]+@
    """
)


class CVESecurityReceiptError(ValueError):
    """A receipt input is malformed, unsafe, inconsistent, or over its bound."""


class SecurityReceiptRole(str, Enum):
    """The semantic role of this record."""

    EVIDENCE = "evidence"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise CVESecurityReceiptError(
            "floating point values are not canonical receipt data"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CVESecurityReceiptError("receipt mapping keys must be strings")
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _plain(converter())
    raise CVESecurityReceiptError(
        f"unsupported receipt value: {type(value).__name__}"
    )


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
        raise CVESecurityReceiptError("receipt is not canonical JSON") from exc


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise CVESecurityReceiptError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value or "\r" in value or "\n" in value:
        raise CVESecurityReceiptError(
            f"{name} must not contain whitespace framing, NUL, or line breaks"
        )
    if required and not value:
        raise CVESecurityReceiptError(f"{name} is required")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_UTF8_BYTES:
        raise CVESecurityReceiptError(
            f"{name} exceeds {MAX_IDENTIFIER_UTF8_BYTES} UTF-8 bytes"
        )
    if value and _SENSITIVE_VALUE.search(value):
        raise CVESecurityReceiptError(
            f"{name} appears to contain a credential or secret"
        )
    return value


def _identifiers(
    values: Sequence[str] | str | None,
    name: str,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence):
        raise CVESecurityReceiptError(f"{name} must be a sequence")
    if len(values) > MAX_IDENTIFIERS_PER_FIELD:
        raise CVESecurityReceiptError(
            f"{name} exceeds its {MAX_IDENTIFIERS_PER_FIELD}-item input bound"
        )
    result = tuple(sorted({_identifier(item, name) for item in values}))
    if len(result) > MAX_IDENTIFIERS_PER_FIELD:
        raise CVESecurityReceiptError(
            f"{name} exceeds its {MAX_IDENTIFIERS_PER_FIELD}-item bound"
        )
    return result


def _integer(value: Any, name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CVESecurityReceiptError(f"{name} must be a non-negative integer")
    return value


def _root_token(artifact_id: str, cid_v1: str, supervisor_digest: str) -> str:
    return ":".join((artifact_id, cid_v1, supervisor_digest))


@dataclass(frozen=True)
class SecurityReceiptStreamEvidence:
    """Identifier-only links to one independently evaluated fact stream."""

    stream: SecurityFactStream
    mapping_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    decision_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream", SecurityFactStream(self.stream))
        for name in (
            "mapping_ids",
            "source_ids",
            "evidence_ids",
            "request_ids",
            "decision_ids",
        ):
            object.__setattr__(
                self, name, _identifiers(getattr(self, name), name)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_STREAM_EVIDENCE_SCHEMA,
            "stream": self.stream.value,
            "mapping_ids": list(self.mapping_ids),
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "request_ids": list(self.request_ids),
            "decision_ids": list(self.decision_ids),
            "contains_raw_facts": False,
            "grants_execution_authority": False,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SecurityReceiptStreamEvidence":
        if not isinstance(value, Mapping):
            raise CVESecurityReceiptError("stream evidence must be an object")
        if value.get("schema") not in (None, CVE_SECURITY_STREAM_EVIDENCE_SCHEMA):
            raise CVESecurityReceiptError("stream evidence schema mismatch")
        if value.get("contains_raw_facts") not in (None, False):
            raise CVESecurityReceiptError("stream evidence cannot contain raw facts")
        if value.get("grants_execution_authority") not in (None, False):
            raise CVESecurityReceiptError(
                "stream evidence cannot grant execution authority"
            )
        return cls(
            stream=value.get("stream", ""),
            mapping_ids=tuple(value.get("mapping_ids") or ()),
            source_ids=tuple(value.get("source_ids") or ()),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            request_ids=tuple(value.get("request_ids") or ()),
            decision_ids=tuple(value.get("decision_ids") or ()),
        )


@dataclass(frozen=True)
class SecurityReceiptCounterexample:
    """A safe finding summary that deliberately excludes arbitrary details."""

    finding_id: str
    reason_code: str
    intent_mapping_ids: tuple[str, ...] = ()
    code_mapping_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self, "reason_code", _identifier(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "intent_mapping_ids",
            _identifiers(self.intent_mapping_ids, "intent_mapping_ids"),
        )
        object.__setattr__(
            self,
            "code_mapping_ids",
            _identifiers(self.code_mapping_ids, "code_mapping_ids"),
        )

    @property
    def counterexample_id(self) -> str:
        return _identity("cve-security-counterexample", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_COUNTEREXAMPLE_SCHEMA,
            "finding_id": self.finding_id,
            "reason_code": self.reason_code,
            "intent_mapping_ids": list(self.intent_mapping_ids),
            "code_mapping_ids": list(self.code_mapping_ids),
            "details_redacted": True,
            "grants_execution_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"counterexample_id": self.counterexample_id, **self._payload()}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SecurityReceiptCounterexample":
        if not isinstance(value, Mapping):
            raise CVESecurityReceiptError("counterexample must be an object")
        if value.get("schema") not in (None, CVE_SECURITY_COUNTEREXAMPLE_SCHEMA):
            raise CVESecurityReceiptError("counterexample schema mismatch")
        if value.get("details_redacted") not in (None, True):
            raise CVESecurityReceiptError(
                "counterexample details must remain redacted"
            )
        if value.get("grants_execution_authority") not in (None, False):
            raise CVESecurityReceiptError(
                "counterexample cannot grant execution authority"
            )
        result = cls(
            finding_id=value.get("finding_id", ""),
            reason_code=value.get("reason_code", ""),
            intent_mapping_ids=tuple(value.get("intent_mapping_ids") or ()),
            code_mapping_ids=tuple(value.get("code_mapping_ids") or ()),
        )
        claimed = str(value.get("counterexample_id") or "")
        if claimed and claimed != result.counterexample_id:
            raise CVESecurityReceiptError("counterexample identity mismatch")
        return result


@dataclass(frozen=True)
class BoundedSecurityDecisionReceipt:
    """Canonical, stage-bound and safe-to-log CVE security decision evidence."""

    stage: CVESecurityEnforcementStage
    repository_tree_id: str
    outcome: CVESecurityGateOutcome
    security_root_artifact_id: str
    security_root_cid_v1: str
    security_root_supervisor_digest: str
    policy_receipt_id: str
    gate_result_id: str
    enforcement_evidence_id: str
    evidence_authority: str
    evaluated_at_ms: int
    parent_evidence_id: str = ""
    expires_at_ms: int | None = None
    intent_evidence: SecurityReceiptStreamEvidence = field(
        default_factory=lambda: SecurityReceiptStreamEvidence(
            SecurityFactStream.INTENT
        )
    )
    code_evidence: SecurityReceiptStreamEvidence = field(
        default_factory=lambda: SecurityReceiptStreamEvidence(
            SecurityFactStream.CODE
        )
    )
    matched_policy_ids: tuple[str, ...] = ()
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    source_cids: tuple[str, ...] = ()
    semantic_roots: Mapping[str, str] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    counterexamples: tuple[SecurityReceiptCounterexample, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", CVESecurityEnforcementStage(self.stage)
        )
        object.__setattr__(self, "outcome", CVESecurityGateOutcome(self.outcome))
        for name in (
            "repository_tree_id",
            "security_root_artifact_id",
            "security_root_cid_v1",
            "security_root_supervisor_digest",
            "policy_receipt_id",
            "gate_result_id",
            "enforcement_evidence_id",
            "evidence_authority",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "parent_evidence_id",
            _identifier(
                self.parent_evidence_id, "parent_evidence_id", required=False
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _integer(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _integer(self.expires_at_ms, "expires_at_ms", optional=True),
        )
        if (
            self.expires_at_ms is not None
            and self.expires_at_ms < self.evaluated_at_ms
        ):
            raise CVESecurityReceiptError(
                "expires_at_ms cannot precede evaluated_at_ms"
            )
        if not isinstance(
            self.intent_evidence, SecurityReceiptStreamEvidence
        ) or self.intent_evidence.stream is not SecurityFactStream.INTENT:
            raise CVESecurityReceiptError(
                "intent_evidence must bind the intent stream"
            )
        if not isinstance(
            self.code_evidence, SecurityReceiptStreamEvidence
        ) or self.code_evidence.stream is not SecurityFactStream.CODE:
            raise CVESecurityReceiptError(
                "code_evidence must bind the code stream"
            )
        for name in (
            "matched_policy_ids",
            "cve_ids",
            "cwe_ids",
            "source_cids",
            "reason_codes",
        ):
            object.__setattr__(
                self, name, _identifiers(getattr(self, name), name)
            )
        if not isinstance(self.semantic_roots, Mapping):
            raise CVESecurityReceiptError("semantic_roots must be an object")
        if len(self.semantic_roots) > MAX_IDENTIFIERS_PER_FIELD:
            raise CVESecurityReceiptError("semantic_roots exceeds its item bound")
        roots = {
            _identifier(key, "semantic root kind"): _identifier(
                value, "semantic root"
            )
            for key, value in self.semantic_roots.items()
        }
        expected_security_root = _root_token(
            self.security_root_artifact_id,
            self.security_root_cid_v1,
            self.security_root_supervisor_digest,
        )
        if roots.get(_SECURITY_ROOT_KIND) != expected_security_root:
            raise CVESecurityReceiptError(
                "semantic_roots must contain the exact security_ir root"
            )
        object.__setattr__(
            self,
            "semantic_roots",
            MappingProxyType(dict(sorted(roots.items()))),
        )
        examples: list[SecurityReceiptCounterexample] = []
        if len(self.counterexamples) > MAX_COUNTEREXAMPLES:
            raise CVESecurityReceiptError("counterexamples exceeds its item bound")
        for item in self.counterexamples:
            if isinstance(item, Mapping):
                item = SecurityReceiptCounterexample.from_dict(item)
            if not isinstance(item, SecurityReceiptCounterexample):
                raise CVESecurityReceiptError(
                    "counterexamples contains an invalid record"
                )
            examples.append(item)
        examples = sorted(
            {item.counterexample_id: item for item in examples}.values(),
            key=lambda item: item.counterexample_id,
        )
        object.__setattr__(self, "counterexamples", tuple(examples))
        bounded_payload = {
            "receipt_id": (
                "cve-security-observability-receipt:sha256:" + ("0" * 64)
            ),
            **self._payload(),
        }
        if len(_canonical_bytes(bounded_payload)) > MAX_RECEIPT_UTF8_BYTES:
            raise CVESecurityReceiptError(
                f"receipt exceeds {MAX_RECEIPT_UTF8_BYTES} UTF-8 bytes"
            )

    @property
    def record_role(self) -> SecurityReceiptRole:
        return SecurityReceiptRole.EVIDENCE

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def declared_dependencies(self) -> Mapping[str, Any]:
        """Every external input whose change must invalidate a cached result."""

        return MappingProxyType(
            {
                "stage": self.stage.value,
                "repository_tree_id": self.repository_tree_id,
                "outcome": self.outcome.value,
                "security_roots": dict(self.semantic_roots),
                "policy_receipt_id": self.policy_receipt_id,
                "gate_result_id": self.gate_result_id,
                "enforcement_evidence_id": self.enforcement_evidence_id,
                "parent_evidence_id": self.parent_evidence_id,
                "evidence_authority": self.evidence_authority,
                "evaluated_at_ms": self.evaluated_at_ms,
                "expires_at_ms": self.expires_at_ms,
                "intent_evidence": self.intent_evidence.to_dict(),
                "code_evidence": self.code_evidence.to_dict(),
                "matched_policy_ids": list(self.matched_policy_ids),
                "cve_ids": list(self.cve_ids),
                "cwe_ids": list(self.cwe_ids),
                "source_cids": list(self.source_cids),
                "reason_codes": list(self.reason_codes),
                "counterexample_ids": [
                    item.counterexample_id for item in self.counterexamples
                ],
            }
        )

    @property
    def cache_key(self) -> str:
        return _identity(
            "cve-security-observability-cache",
            {
                "schema": CVE_SECURITY_OBSERVABILITY_RECEIPT_SCHEMA,
                "version": CVE_SECURITY_OBSERVABILITY_VERSION,
                "dependencies": self.declared_dependencies,
            },
        )

    def _payload(self, *, include_cache_key: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CVE_SECURITY_OBSERVABILITY_RECEIPT_SCHEMA,
            "receipt_version": CVE_SECURITY_OBSERVABILITY_VERSION,
            "record_role": self.record_role.value,
            "stage": self.stage.value,
            "repository_tree_id": self.repository_tree_id,
            "outcome": self.outcome.value,
            "security_root_artifact_id": self.security_root_artifact_id,
            "security_root_cid_v1": self.security_root_cid_v1,
            "security_root_supervisor_digest": self.security_root_supervisor_digest,
            "policy_receipt_id": self.policy_receipt_id,
            "gate_result_id": self.gate_result_id,
            "enforcement_evidence_id": self.enforcement_evidence_id,
            "parent_evidence_id": self.parent_evidence_id,
            "evidence_authority": self.evidence_authority,
            "evaluated_at_ms": self.evaluated_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "intent_evidence": self.intent_evidence.to_dict(),
            "code_evidence": self.code_evidence.to_dict(),
            "matched_policy_ids": list(self.matched_policy_ids),
            "cve_ids": list(self.cve_ids),
            "cwe_ids": list(self.cwe_ids),
            "source_cids": list(self.source_cids),
            "semantic_roots": dict(self.semantic_roots),
            "reason_codes": list(self.reason_codes),
            "counterexamples": [
                item.to_dict() for item in self.counterexamples
            ],
            "declared_dependencies": _plain(self.declared_dependencies),
            "redacted": True,
            "redacted_fields": list(_REDACTED_FIELDS),
            "contains_code_body": False,
            "contains_secrets": False,
            "evidence_is_authority": False,
            "grants_execution_authority": False,
            "authorizes_completion": False,
        }
        if include_cache_key:
            payload["cache_key"] = self.cache_key
        return payload

    @property
    def receipt_id(self) -> str:
        return _identity(
            "cve-security-observability-receipt", self._payload()
        )

    def to_dict(self) -> dict[str, Any]:
        return {"receipt_id": self.receipt_id, **self._payload()}

    def to_event_fields(self) -> dict[str, Any]:
        """Return the complete bounded payload suitable for structured logs."""

        return self.to_dict()

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "BoundedSecurityDecisionReceipt":
        if not isinstance(value, Mapping):
            raise CVESecurityReceiptError("receipt must be an object")
        fixed_fields = {
            "schema": CVE_SECURITY_OBSERVABILITY_RECEIPT_SCHEMA,
            "receipt_version": CVE_SECURITY_OBSERVABILITY_VERSION,
            "record_role": SecurityReceiptRole.EVIDENCE.value,
            "redacted": True,
            "contains_code_body": False,
            "contains_secrets": False,
            "evidence_is_authority": False,
            "grants_execution_authority": False,
            "authorizes_completion": False,
        }
        for name, expected in fixed_fields.items():
            if value.get(name, expected) != expected:
                raise CVESecurityReceiptError(f"receipt {name} mismatch")
        supplied_redactions = value.get("redacted_fields")
        if supplied_redactions is not None:
            if (
                isinstance(supplied_redactions, (str, bytes))
                or not isinstance(supplied_redactions, Sequence)
                or tuple(supplied_redactions) != _REDACTED_FIELDS
            ):
                raise CVESecurityReceiptError(
                    "receipt redacted_fields mismatch"
                )
        result = cls(
            stage=value.get("stage", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            outcome=value.get("outcome", ""),
            security_root_artifact_id=value.get("security_root_artifact_id", ""),
            security_root_cid_v1=value.get("security_root_cid_v1", ""),
            security_root_supervisor_digest=value.get(
                "security_root_supervisor_digest", ""
            ),
            policy_receipt_id=value.get("policy_receipt_id", ""),
            gate_result_id=value.get("gate_result_id", ""),
            enforcement_evidence_id=value.get("enforcement_evidence_id", ""),
            parent_evidence_id=value.get("parent_evidence_id", ""),
            evidence_authority=value.get("evidence_authority", ""),
            evaluated_at_ms=value.get("evaluated_at_ms", -1),
            expires_at_ms=value.get("expires_at_ms"),
            intent_evidence=SecurityReceiptStreamEvidence.from_dict(
                value.get("intent_evidence") or {}
            ),
            code_evidence=SecurityReceiptStreamEvidence.from_dict(
                value.get("code_evidence") or {}
            ),
            matched_policy_ids=tuple(value.get("matched_policy_ids") or ()),
            cve_ids=tuple(value.get("cve_ids") or ()),
            cwe_ids=tuple(value.get("cwe_ids") or ()),
            source_cids=tuple(value.get("source_cids") or ()),
            semantic_roots=value.get("semantic_roots") or {},
            reason_codes=tuple(value.get("reason_codes") or ()),
            counterexamples=tuple(value.get("counterexamples") or ()),
        )
        if value.get("cache_key") not in (None, "", result.cache_key):
            raise CVESecurityReceiptError("receipt cache key mismatch")
        supplied_dependencies = value.get("declared_dependencies")
        if supplied_dependencies is not None and _canonical_bytes(
            supplied_dependencies
        ) != _canonical_bytes(result.declared_dependencies):
            raise CVESecurityReceiptError(
                "receipt dependency manifest mismatch"
            )
        if value.get("receipt_id") not in (None, "", result.receipt_id):
            raise CVESecurityReceiptError("receipt identity mismatch")
        if len(result.canonical_bytes) > MAX_RECEIPT_UTF8_BYTES:
            raise CVESecurityReceiptError(
                f"receipt exceeds {MAX_RECEIPT_UTF8_BYTES} UTF-8 bytes"
            )
        return result


def _stream_evidence(
    gate: CVESecurityGateResult,
    stream: SecurityFactStream,
) -> SecurityReceiptStreamEvidence:
    mappings = (
        gate.intent_mappings
        if stream is SecurityFactStream.INTENT
        else gate.code_mappings
    )
    mapping_ids = {item.mapping_id for item in mappings}
    decisions = tuple(
        item
        for item in gate.decisions
        if item.stream is stream and item.mapping_id in mapping_ids
    )
    return SecurityReceiptStreamEvidence(
        stream=stream,
        mapping_ids=tuple(item.mapping_id for item in mappings),
        source_ids=tuple(item.source_id for item in mappings),
        evidence_ids=tuple(
            evidence_id
            for item in mappings
            for evidence_id in item.evidence_ids
        ),
        request_ids=tuple(
            item.request.content_id
            for item in mappings
            if item.request is not None
        ),
        decision_ids=tuple(item.decision.content_id for item in decisions),
    )


def emit_cve_security_decision_receipt(
    evidence: CVESecurityEnforcementEvidence,
    *,
    cve_ids: Sequence[str] = (),
    cwe_ids: Sequence[str] = (),
    source_cids: Sequence[str] = (),
    semantic_roots: Mapping[str, str] | None = None,
) -> BoundedSecurityDecisionReceipt:
    """Project full enforcement evidence to a bounded observability receipt.

    Arbitrary finding details and exact request values are intentionally not
    accepted by this API. The caller may add only opaque CVE/CWE/source
    identities and additional semantic roots.
    """

    if not isinstance(evidence, CVESecurityEnforcementEvidence):
        raise CVESecurityReceiptError(
            "evidence must be CVESecurityEnforcementEvidence"
        )
    gate = evidence.gate_result
    context = gate.context
    root = (
        context.security_root_artifact_id,
        context.security_root_cid_v1,
        context.security_root_supervisor_digest,
    )
    mapping_ids = {
        item.mapping_id
        for item in (*gate.intent_mappings, *gate.code_mappings)
    }
    for item in gate.decisions:
        decision = item.decision
        if item.mapping_id not in mapping_ids:
            raise CVESecurityReceiptError(
                "decision references a mapping outside the gate result"
            )
        if decision.policy_receipt_id != gate.policy_receipt_id:
            raise CVESecurityReceiptError(
                "decision and gate policy receipt identities differ"
            )
        if (
            decision.security_root_artifact_id,
            decision.security_root_cid_v1,
            decision.security_root_supervisor_digest,
        ) != root:
            raise CVESecurityReceiptError(
                "decision is detached from the gate Security IR root"
            )
        if decision.evaluated_at_ms != context.evaluated_at_ms:
            raise CVESecurityReceiptError(
                "decision timestamp is detached from the gate context"
            )

    roots = dict(semantic_roots or {})
    security_root = _root_token(*root)
    if roots.get(_SECURITY_ROOT_KIND, security_root) != security_root:
        raise CVESecurityReceiptError(
            "supplied security_ir root differs from gate evidence"
        )
    roots[_SECURITY_ROOT_KIND] = security_root

    matched_policy_ids = tuple(
        policy_id
        for item in gate.decisions
        for policy_id in item.decision.matched_policy_ids
    )
    reasons = {
        code.value
        for mapping in (*gate.intent_mappings, *gate.code_mappings)
        for code in mapping.reason_codes
    }
    reasons.update(item.code.value for item in gate.findings)
    for item in gate.decisions:
        reasons.update(item.decision.reason_codes)
        reasons.update(check.reason_code for check in item.decision.checks)
    counterexamples = tuple(
        SecurityReceiptCounterexample(
            finding_id=item.finding_id,
            reason_code=item.code.value,
            intent_mapping_ids=item.intent_mapping_ids,
            code_mapping_ids=item.code_mapping_ids,
        )
        for item in gate.findings
    )

    return BoundedSecurityDecisionReceipt(
        stage=evidence.stage,
        repository_tree_id=evidence.repository_tree_id,
        outcome=gate.outcome,
        security_root_artifact_id=root[0],
        security_root_cid_v1=root[1],
        security_root_supervisor_digest=root[2],
        policy_receipt_id=gate.policy_receipt_id,
        gate_result_id=gate.gate_id,
        enforcement_evidence_id=evidence.evidence_id,
        parent_evidence_id=evidence.parent_evidence_id,
        evidence_authority=evidence.authority,
        evaluated_at_ms=context.evaluated_at_ms,
        expires_at_ms=evidence.expires_at_ms,
        intent_evidence=_stream_evidence(gate, SecurityFactStream.INTENT),
        code_evidence=_stream_evidence(gate, SecurityFactStream.CODE),
        matched_policy_ids=matched_policy_ids,
        cve_ids=tuple(cve_ids),
        cwe_ids=tuple(cwe_ids),
        source_cids=tuple(source_cids),
        semantic_roots=roots,
        reason_codes=tuple(reasons),
        counterexamples=counterexamples,
    )


# Concise and compatibility-oriented spellings for callers.
CVESecurityDecisionReceipt = BoundedSecurityDecisionReceipt
CVESecurityReceipt = BoundedSecurityDecisionReceipt
build_cve_security_decision_receipt = emit_cve_security_decision_receipt
build_cve_security_receipt = emit_cve_security_decision_receipt


__all__ = [
    "BoundedSecurityDecisionReceipt",
    "CVESecurityDecisionReceipt",
    "CVESecurityReceipt",
    "CVESecurityReceiptError",
    "CVE_SECURITY_COUNTEREXAMPLE_SCHEMA",
    "CVE_SECURITY_OBSERVABILITY_RECEIPT_SCHEMA",
    "CVE_SECURITY_OBSERVABILITY_VERSION",
    "CVE_SECURITY_STREAM_EVIDENCE_SCHEMA",
    "MAX_COUNTEREXAMPLES",
    "MAX_IDENTIFIERS_PER_FIELD",
    "MAX_IDENTIFIER_UTF8_BYTES",
    "MAX_RECEIPT_UTF8_BYTES",
    "SecurityReceiptCounterexample",
    "SecurityReceiptRole",
    "SecurityReceiptStreamEvidence",
    "build_cve_security_decision_receipt",
    "build_cve_security_receipt",
    "emit_cve_security_decision_receipt",
]
