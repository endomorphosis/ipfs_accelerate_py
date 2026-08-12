"""Body-free inference explanation for target resolution receipts (ASE-011).

``render_target_resolution`` turns a
:class:`~ipfs_accelerate_py.agent_supervisor.entrypoints.contracts.TargetResolutionReceipt`
into a bounded, deterministic provenance document that operators and automation
can replay without starting work.

Design rules enforced here:

- every selected, defaulted, ambiguous, unavailable, or denied field gets an
  evidence-backed explanation (source, evidence CID, reasons, alternatives);
- prompt bodies, credentials, bearer tokens, and raw source material never
  enter explanation text, JSON, or error messages;
- rendering is pure and read-only over the receipt;
- human and JSON projections are derived from one sealed payload so identities
  stay stable under identical receipts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Iterable

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

from . import contracts as _contracts
from .contracts import (
    REQUIRED_TARGET_DECISION_FIELDS,
    EntrypointContractError,
    ResolutionDisposition,
    ResolutionSource,
    SecretBearingRecordError,
    TargetCandidate,
    TargetInferenceDecision,
    TargetResolutionReceipt,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
INFERENCE_EXPLANATION_SCHEMA: Final = f"{SCHEMA_PREFIX}/inference-explanation@1"
FIELD_EXPLANATION_SCHEMA: Final = f"{SCHEMA_PREFIX}/field-explanation@1"
ALTERNATIVE_SUMMARY_SCHEMA: Final = f"{SCHEMA_PREFIX}/alternative-summary@1"

# Evidence requirement shared with plan_lint (ASE-G035 / ASE-011).
INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID: Final = (
    "inference_explain.INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID"
)
INFERENCE_EXPLAIN_REQUIREMENT_ID: Final = (
    "requirement:agent-supervisor.entrypoints.inference-explain@1"
)

MAX_REASON_TEXT_BYTES: Final = 512
MAX_FIELD_SUMMARY_BYTES: Final = 1_024
MAX_HUMAN_TEXT_BYTES: Final = 256 * 1024
MAX_ALTERNATIVES_RENDERED: Final = 64

# Reuse the closed secret-scanner inventory from contracts so this module does
# not re-introduce PEM header / credential marker literals that the proposal
# gate treats as secret-bearing content when added as new files.
_JWT_RE = _contracts._JWT_RE
_SECRET_ASSIGNMENT_RE = _contracts._SECRET_ASSIGNMENT_RE
_KNOWN_SECRET_TOKEN_RE = _contracts._KNOWN_SECRET_TOKEN_RE
_SECRET_TEXT_MARKERS = _contracts._SECRET_TEXT_MARKERS

_DISPOSITION_VERB: Final[Mapping[ResolutionDisposition, str]] = {
    ResolutionDisposition.UNIQUE: "selected",
    ResolutionDisposition.DEFAULTED: "defaulted",
    ResolutionDisposition.AMBIGUOUS: "left ambiguous",
    ResolutionDisposition.UNAVAILABLE: "marked unavailable",
    ResolutionDisposition.DENIED: "denied",
}


class InferenceExplainError(EntrypointContractError):
    """Raised when explanation input is malformed or would leak bodies."""


class ExplanationFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    BOTH = "both"


def _safe_error(message: str) -> InferenceExplainError:
    """Build an error whose message is free of secret-like material."""

    cleaned = _redact_sensitive_text(str(message or "explanation failed"))
    return InferenceExplainError(cleaned[:MAX_REASON_TEXT_BYTES])


def _contains_secret_material(value: str) -> bool:
    if not value:
        return False
    lowered = value.casefold()
    if any(marker in lowered for marker in _SECRET_TEXT_MARKERS):
        return True
    if _JWT_RE.search(value):
        return True
    if _SECRET_ASSIGNMENT_RE.search(value):
        return True
    if _KNOWN_SECRET_TOKEN_RE.search(value):
        return True
    return False


def _redact_sensitive_text(value: str) -> str:
    """Remove secret-shaped substrings from error paths (never return bodies)."""

    text = str(value or "")
    text = _JWT_RE.sub("[redacted-jwt]", text)
    text = _KNOWN_SECRET_TOKEN_RE.sub("[redacted-token]", text)
    text = _SECRET_ASSIGNMENT_RE.sub("[redacted-assignment]", text)
    for marker in _SECRET_TEXT_MARKERS:
        if marker in text.casefold():
            text = re.sub(re.escape(marker), "[redacted-marker]", text, flags=re.I)
    return text


def _reject_secret_or_body(
    value: str,
    *,
    name: str,
    prompt_body: str | bytes | None = None,
) -> str:
    text = str(value or "")
    if _contains_secret_material(text):
        raise _safe_error(f"{name} contains secret-bearing material")
    if prompt_body is not None:
        try:
            body = (
                prompt_body.decode("utf-8")
                if isinstance(prompt_body, (bytes, bytearray))
                else str(prompt_body)
            )
        except UnicodeDecodeError:
            body = ""
        if body and body in text:
            raise _safe_error(f"{name} must not embed prompt body material")
    return text


def _bounded_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise _safe_error(f"{name} must be text")
    if "\x00" in value:
        raise _safe_error(f"{name} contains a NUL byte")
    encoded = value.encode("utf-8")
    if len(encoded) > maximum:
        raise _safe_error(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _reason_codes(codes: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in codes:
        text = str(item or "").strip()
        if not text:
            continue
        if _contains_secret_material(text):
            raise _safe_error("reason_codes contain secret-bearing material")
        cleaned.append(text)
    return tuple(sorted(set(cleaned)))


def _source_label(source: ResolutionSource | str) -> str:
    if isinstance(source, ResolutionSource):
        return source.value
    return str(source)


def _disposition_label(disposition: ResolutionDisposition | str) -> str:
    if isinstance(disposition, ResolutionDisposition):
        return disposition.value
    return str(disposition)


def _explain_reasons(
    decision: TargetInferenceDecision,
) -> tuple[str, ...]:
    """Return typed reason codes that justify the disposition."""

    codes = list(decision.reason_codes)
    if decision.disposition is ResolutionDisposition.UNIQUE:
        if decision.override_accepted:
            codes.append("explicit_override_accepted")
        if not codes:
            codes.append("unique_candidate_selected")
    elif decision.disposition is ResolutionDisposition.DEFAULTED:
        if not codes:
            codes.append("builtin_or_lower_precedence_default")
    elif decision.disposition is ResolutionDisposition.AMBIGUOUS:
        if not codes:
            codes.append("multiple_equal_rank_candidates")
    elif decision.disposition is ResolutionDisposition.UNAVAILABLE:
        if not codes:
            codes.append("no_viable_candidate")
    elif decision.disposition is ResolutionDisposition.DENIED:
        if not codes:
            codes.append("policy_or_authority_denied")
    return _reason_codes(codes)


def _reason_sentence(decision: TargetInferenceDecision) -> str:
    verb = _DISPOSITION_VERB.get(decision.disposition, "resolved")
    reasons = _explain_reasons(decision)
    reason_text = ", ".join(reasons) if reasons else "no_typed_reason"
    if decision.disposition in {
        ResolutionDisposition.UNIQUE,
        ResolutionDisposition.DEFAULTED,
    }:
        source = _source_label(decision.selected_source)
        return (
            f"Field {decision.field_name!s} was {verb} from source {source} "
            f"with evidence {decision.evidence_cid} "
            f"(reasons: {reason_text})."
        )
    return (
        f"Field {decision.field_name!s} was {verb} "
        f"with evidence {decision.evidence_cid} "
        f"(reasons: {reason_text})."
    )


@dataclass(frozen=True)
class AlternativeSummary:
    """Body-free summary of one non-selected or competing candidate."""

    SCHEMA: ClassVar[str] = ALTERNATIVE_SUMMARY_SCHEMA

    value: str
    source: str
    source_precedence: int
    evidence_cid: str
    confidence_ppm: int
    rejection_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            _reject_secret_or_body(
                _bounded_text(self.value, "value", maximum=MAX_FIELD_SUMMARY_BYTES),
                name="value",
            ),
        )
        object.__setattr__(
            self,
            "source",
            _bounded_text(self.source, "source", maximum=128),
        )
        if not isinstance(self.source_precedence, int) or isinstance(
            self.source_precedence, bool
        ):
            raise _safe_error("source_precedence must be an integer")
        if not isinstance(self.confidence_ppm, int) or isinstance(
            self.confidence_ppm, bool
        ):
            raise _safe_error("confidence_ppm must be an integer")
        object.__setattr__(
            self,
            "evidence_cid",
            _bounded_text(self.evidence_cid, "evidence_cid", maximum=256),
        )
        object.__setattr__(
            self,
            "rejection_reason",
            _reject_secret_or_body(
                _bounded_text(
                    self.rejection_reason,
                    "rejection_reason",
                    maximum=MAX_REASON_TEXT_BYTES,
                ),
                name="rejection_reason",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "value": self.value,
            "source": self.source,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "confidence_ppm": self.confidence_ppm,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_candidate(cls, candidate: TargetCandidate) -> AlternativeSummary:
        return cls(
            value=candidate.value,
            source=_source_label(candidate.source),
            source_precedence=int(candidate.source_precedence),
            evidence_cid=candidate.evidence_cid,
            confidence_ppm=int(candidate.confidence_ppm),
            rejection_reason=candidate.rejection_reason,
        )


@dataclass(frozen=True)
class FieldExplanation:
    """Evidence-backed explanation for one target-inference decision field."""

    SCHEMA: ClassVar[str] = FIELD_EXPLANATION_SCHEMA

    field_name: str
    disposition: str
    selected_value: str
    selected_source: str
    source_precedence: int
    evidence_cid: str
    reason_codes: tuple[str, ...]
    reason: str
    effect: str
    override_accepted: bool
    unresolved: bool
    alternatives: tuple[AlternativeSummary, ...]
    decision_cid: str
    revalidation_rule: str
    fresh_until_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_name",
            _bounded_text(self.field_name, "field_name", maximum=128),
        )
        object.__setattr__(
            self,
            "disposition",
            _bounded_text(self.disposition, "disposition", maximum=64),
        )
        object.__setattr__(
            self,
            "selected_value",
            _reject_secret_or_body(
                _bounded_text(
                    self.selected_value,
                    "selected_value",
                    maximum=MAX_FIELD_SUMMARY_BYTES,
                ),
                name="selected_value",
            ),
        )
        object.__setattr__(
            self,
            "selected_source",
            _bounded_text(self.selected_source, "selected_source", maximum=128),
        )
        if not isinstance(self.source_precedence, int) or isinstance(
            self.source_precedence, bool
        ):
            raise _safe_error("source_precedence must be an integer")
        object.__setattr__(
            self,
            "evidence_cid",
            _bounded_text(self.evidence_cid, "evidence_cid", maximum=256),
        )
        codes = _reason_codes(self.reason_codes)
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(
            self,
            "reason",
            _reject_secret_or_body(
                _bounded_text(self.reason, "reason", maximum=MAX_REASON_TEXT_BYTES),
                name="reason",
            ),
        )
        object.__setattr__(
            self,
            "effect",
            _bounded_text(self.effect, "effect", maximum=64),
        )
        if not isinstance(self.override_accepted, bool):
            raise _safe_error("override_accepted must be a boolean")
        if not isinstance(self.unresolved, bool):
            raise _safe_error("unresolved must be a boolean")
        alts = tuple(self.alternatives)
        if any(not isinstance(item, AlternativeSummary) for item in alts):
            raise _safe_error("alternatives must contain AlternativeSummary values")
        if len(alts) > MAX_ALTERNATIVES_RENDERED:
            raise _safe_error(
                f"alternatives exceeds {MAX_ALTERNATIVES_RENDERED} items"
            )
        object.__setattr__(self, "alternatives", alts)
        object.__setattr__(
            self,
            "decision_cid",
            _bounded_text(self.decision_cid, "decision_cid", maximum=256),
        )
        object.__setattr__(
            self,
            "revalidation_rule",
            _bounded_text(
                self.revalidation_rule, "revalidation_rule", maximum=64
            ),
        )
        if not isinstance(self.fresh_until_ms, int) or isinstance(
            self.fresh_until_ms, bool
        ):
            raise _safe_error("fresh_until_ms must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "field_name": self.field_name,
            "disposition": self.disposition,
            "selected_value": self.selected_value,
            "selected_source": self.selected_source,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "reason_codes": list(self.reason_codes),
            "reason": self.reason,
            "effect": self.effect,
            "override_accepted": self.override_accepted,
            "unresolved": self.unresolved,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "decision_cid": self.decision_cid,
            "revalidation_rule": self.revalidation_rule,
            "fresh_until_ms": self.fresh_until_ms,
        }

    @classmethod
    def from_decision(cls, decision: TargetInferenceDecision) -> FieldExplanation:
        if not isinstance(decision, TargetInferenceDecision):
            raise _safe_error("decision must be a TargetInferenceDecision")
        selected_matches = [
            item
            for item in decision.candidates
            if item.value == decision.selected_value
            and item.source is decision.selected_source
        ]
        alternatives: list[AlternativeSummary] = []
        for candidate in decision.candidates:
            if selected_matches and candidate in selected_matches:
                continue
            # For unresolved fields every candidate is an alternative.
            if decision.unresolved or candidate.rejection_reason:
                alternatives.append(AlternativeSummary.from_candidate(candidate))
            elif not selected_matches:
                alternatives.append(AlternativeSummary.from_candidate(candidate))
        alternatives.sort(
            key=lambda item: (
                item.source_precedence,
                item.source,
                item.value,
                item.evidence_cid,
            )
        )
        return cls(
            field_name=decision.field_name,
            disposition=_disposition_label(decision.disposition),
            selected_value=decision.selected_value,
            selected_source=_source_label(decision.selected_source),
            source_precedence=int(decision.source_precedence),
            evidence_cid=decision.evidence_cid,
            reason_codes=_explain_reasons(decision),
            reason=_reason_sentence(decision),
            effect=decision.effect.value,
            override_accepted=bool(decision.override_accepted),
            unresolved=bool(decision.unresolved),
            alternatives=tuple(alternatives[:MAX_ALTERNATIVES_RENDERED]),
            decision_cid=decision.content_id,
            revalidation_rule=decision.revalidation_rule.value,
            fresh_until_ms=int(decision.fresh_until_ms),
        )


@dataclass(frozen=True)
class InferenceExplanation:
    """Complete body-free explanation of one target resolution receipt."""

    SCHEMA: ClassVar[str] = INFERENCE_EXPLANATION_SCHEMA

    requirement_id: str
    receipt_cid: str
    invocation_cid: str
    prompt_cid: str
    configuration_root_cid: str
    unresolved_fields: tuple[str, ...]
    fields: tuple[FieldExplanation, ...]
    summary: str
    human_text: str
    resolved_at_ms: int
    fresh_until_ms: int
    effects_blocked: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _bounded_text(self.requirement_id, "requirement_id", maximum=256),
        )
        for name in (
            "receipt_cid",
            "invocation_cid",
            "prompt_cid",
            "configuration_root_cid",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), name, maximum=256),
            )
        unresolved = tuple(
            sorted({str(item) for item in self.unresolved_fields if str(item)})
        )
        object.__setattr__(self, "unresolved_fields", unresolved)
        fields = tuple(self.fields)
        if any(not isinstance(item, FieldExplanation) for item in fields):
            raise _safe_error("fields must contain FieldExplanation values")
        names = [item.field_name for item in fields]
        if len(names) != len(set(names)):
            raise _safe_error("fields contain duplicate field names")
        fields = tuple(sorted(fields, key=lambda item: item.field_name))
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "summary",
            _reject_secret_or_body(
                _bounded_text(self.summary, "summary", maximum=MAX_REASON_TEXT_BYTES),
                name="summary",
            ),
        )
        object.__setattr__(
            self,
            "human_text",
            _reject_secret_or_body(
                _bounded_text(
                    self.human_text, "human_text", maximum=MAX_HUMAN_TEXT_BYTES
                ),
                name="human_text",
            ),
        )
        if not isinstance(self.resolved_at_ms, int) or isinstance(
            self.resolved_at_ms, bool
        ):
            raise _safe_error("resolved_at_ms must be an integer")
        if not isinstance(self.fresh_until_ms, int) or isinstance(
            self.fresh_until_ms, bool
        ):
            raise _safe_error("fresh_until_ms must be an integer")
        if not isinstance(self.effects_blocked, bool):
            raise _safe_error("effects_blocked must be a boolean")
        # Seal identity by evaluating content_id.
        _ = self.content_id

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict(include_identity=False))

    @property
    def explanation_cid(self) -> str:
        return self.content_id

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "requirement_id": self.requirement_id,
            "receipt_cid": self.receipt_cid,
            "invocation_cid": self.invocation_cid,
            "prompt_cid": self.prompt_cid,
            "configuration_root_cid": self.configuration_root_cid,
            "unresolved_fields": list(self.unresolved_fields),
            "fields": [item.to_dict() for item in self.fields],
            "summary": self.summary,
            "human_text": self.human_text,
            "resolved_at_ms": self.resolved_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "effects_blocked": self.effects_blocked,
        }
        if include_identity:
            payload["explanation_cid"] = self.content_id
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        text = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=True,
        )
        return _reject_secret_or_body(text, name="json_projection")

    def render(self, fmt: ExplanationFormat | str = ExplanationFormat.BOTH) -> str:
        format_value = (
            fmt if isinstance(fmt, ExplanationFormat) else ExplanationFormat(str(fmt))
        )
        if format_value is ExplanationFormat.JSON:
            return self.to_json()
        if format_value is ExplanationFormat.TEXT:
            return self.human_text
        return f"{self.human_text.rstrip()}\n\n---\n\n{self.to_json()}"


def _scan_payload_for_bodies(
    payload: Mapping[str, Any] | Sequence[Any] | str,
    *,
    prompt_body: str | bytes | None,
    path: str = "explanation",
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _scan_payload_for_bodies(
                value, prompt_body=prompt_body, path=f"{path}.{key}"
            )
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            _scan_payload_for_bodies(
                value, prompt_body=prompt_body, path=f"{path}[{index}]"
            )
        return
    if isinstance(payload, (bytes, bytearray)):
        raise _safe_error(f"{path} must not carry raw bytes")
    if isinstance(payload, str):
        _reject_secret_or_body(payload, name=path, prompt_body=prompt_body)


def _build_human_text(
    *,
    receipt: TargetResolutionReceipt,
    fields: Sequence[FieldExplanation],
    summary: str,
) -> str:
    lines: list[str] = [
        "Target resolution explanation (body-free)",
        f"requirement: {INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID}",
        f"receipt_cid: {receipt.receipt_cid}",
        f"invocation_cid: {receipt.invocation_cid}",
        f"prompt_cid: {receipt.prompt_cid}",
        f"summary: {summary}",
        "",
        "Fields:",
    ]
    for item in fields:
        lines.append(
            f"- {item.field_name}: disposition={item.disposition} "
            f"source={item.selected_source or '-'} "
            f"value={item.selected_value or '-'} "
            f"evidence={item.evidence_cid}"
        )
        lines.append(f"  reason: {item.reason}")
        if item.reason_codes:
            lines.append(f"  reason_codes: {', '.join(item.reason_codes)}")
        if item.alternatives:
            lines.append(f"  alternatives: {len(item.alternatives)}")
            for alt in item.alternatives:
                reject = alt.rejection_reason or "competing"
                lines.append(
                    f"    - value={alt.value} source={alt.source} "
                    f"evidence={alt.evidence_cid} rejection={reject}"
                )
    if receipt.unresolved_fields:
        lines.append("")
        lines.append(
            "Unresolved fields: " + ", ".join(receipt.unresolved_fields)
        )
    text = "\n".join(lines) + "\n"
    return _reject_secret_or_body(
        _bounded_text(text, "human_text", maximum=MAX_HUMAN_TEXT_BYTES),
        name="human_text",
    )


def _summary_for(receipt: TargetResolutionReceipt) -> str:
    by_disposition: dict[str, int] = {}
    for decision in receipt.decisions:
        key = decision.disposition.value
        by_disposition[key] = by_disposition.get(key, 0) + 1
    parts = [
        f"{name}={count}"
        for name, count in sorted(by_disposition.items())
    ]
    unresolved = len(receipt.unresolved_fields)
    return (
        f"receipt {receipt.receipt_cid} explains {len(receipt.decisions)} fields "
        f"({', '.join(parts)}; unresolved={unresolved})"
    )


def explain_field(decision: TargetInferenceDecision) -> FieldExplanation:
    """Public helper: explain one decision without a full receipt."""

    return FieldExplanation.from_decision(decision)


def render_target_resolution(
    receipt: TargetResolutionReceipt | Mapping[str, Any],
    *,
    format: ExplanationFormat | str = ExplanationFormat.BOTH,
    prompt_body: str | bytes | None = None,
    requirement_id: str = INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID,
) -> InferenceExplanation:
    """Render a body-free, evidence-backed explanation of *receipt*.

    Parameters
    ----------
    receipt:
        A :class:`TargetResolutionReceipt` or its closed mapping form.
    format:
        Preferred projection for :meth:`InferenceExplanation.render`.  The
        returned object always carries both human and JSON projections.
    prompt_body:
        Optional non-durable prompt body used only as a *negative* leak oracle.
        It is never copied into the explanation.  When provided, any embedding
        of the body into decision values or explanation text fails closed.
    requirement_id:
        Stable evidence identifier for ASE-G035 / ASE-011.
    """

    try:
        if isinstance(receipt, TargetResolutionReceipt):
            resolved = receipt
        elif isinstance(receipt, Mapping):
            resolved = TargetResolutionReceipt.from_dict(receipt)
        else:
            raise _safe_error(
                "receipt must be a TargetResolutionReceipt or mapping"
            )
    except SecretBearingRecordError as exc:
        raise _safe_error("receipt rejected as secret-bearing") from exc
    except EntrypointContractError as exc:
        raise _safe_error("receipt is not a valid target resolution receipt") from exc
    except Exception as exc:  # noqa: BLE001 - fail closed without leaking bodies
        raise _safe_error("receipt could not be loaded") from exc

    # Validate format early (value unused beyond validation; caller may re-render).
    if not isinstance(format, ExplanationFormat):
        try:
            ExplanationFormat(str(format))
        except ValueError as exc:
            raise _safe_error("unsupported explanation format") from exc

    decision_names = {item.field_name for item in resolved.decisions}
    required = set(REQUIRED_TARGET_DECISION_FIELDS)
    if decision_names != required:
        missing = sorted(required.difference(decision_names))
        extra = sorted(decision_names.difference(required))
        raise _safe_error(
            f"receipt decisions incomplete missing={missing} extra={extra}"
        )

    fields = tuple(
        FieldExplanation.from_decision(decision)
        for decision in sorted(
            resolved.decisions, key=lambda item: item.field_name
        )
    )
    for field in fields:
        if not field.evidence_cid:
            raise _safe_error(
                f"field {field.field_name} lacks evidence_cid"
            )
        if not field.reason_codes:
            raise _safe_error(
                f"field {field.field_name} lacks evidence-backed reason codes"
            )
        if field.disposition in {
            ResolutionDisposition.UNIQUE.value,
            ResolutionDisposition.DEFAULTED.value,
        }:
            if not field.selected_value or not field.selected_source:
                raise _safe_error(
                    f"field {field.field_name} selected without value/source"
                )
        if field.disposition in {
            ResolutionDisposition.AMBIGUOUS.value,
            ResolutionDisposition.UNAVAILABLE.value,
            ResolutionDisposition.DENIED.value,
        }:
            if field.selected_value:
                raise _safe_error(
                    f"field {field.field_name} unresolved but carries a value"
                )

    summary = _summary_for(resolved)
    human_text = _build_human_text(
        receipt=resolved, fields=fields, summary=summary
    )
    explanation = InferenceExplanation(
        requirement_id=requirement_id,
        receipt_cid=resolved.receipt_cid,
        invocation_cid=resolved.invocation_cid,
        prompt_cid=resolved.prompt_cid,
        configuration_root_cid=resolved.configuration_root_cid,
        unresolved_fields=tuple(resolved.unresolved_fields),
        fields=fields,
        summary=summary,
        human_text=human_text,
        resolved_at_ms=int(resolved.resolved_at_ms),
        fresh_until_ms=int(resolved.fresh_until_ms),
        effects_blocked=bool(resolved.unresolved_fields),
    )
    _scan_payload_for_bodies(
        explanation.to_dict(), prompt_body=prompt_body, path="explanation"
    )
    _scan_payload_for_bodies(
        explanation.human_text, prompt_body=prompt_body, path="human_text"
    )
    # Force JSON path through the same scan.
    _ = explanation.to_json(indent=None)
    return explanation


__all__ = (
    "ALTERNATIVE_SUMMARY_SCHEMA",
    "AlternativeSummary",
    "ExplanationFormat",
    "FIELD_EXPLANATION_SCHEMA",
    "FieldExplanation",
    "INFERENCE_EXPLANATION_SCHEMA",
    "INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID",
    "INFERENCE_EXPLAIN_REQUIREMENT_ID",
    "InferenceExplainError",
    "InferenceExplanation",
    "explain_field",
    "render_target_resolution",
)
