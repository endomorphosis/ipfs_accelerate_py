"""Minimal compiler for irreducible human-escalation packets.

``HumanEscalationCompiler@1`` emits at most one precise bounded question.  It
does not deliver the packet, bind an answer, admit an effect, or grant
authority.  Control-service delivery remains APMC-017.

A packet is compiled only when a named unresolved decision is irreducible:
operator-only authority, irreversible/legal/financial effects, irresolvable
contradiction, policy-required privacy or budget choice, or residual ambiguity
after every admitted non-human route has been exhausted.  Required human
decisions cannot be suppressed.  Non-irreducible questions return a
deterministic non-escalation reason instead of a packet.

Packets never ask for full-history review.  Evidence is referenced by content
identity only.  Equivalent irreducible questions collapse to one packet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final

from .contracts import (
    AuthorityClass,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    DecisionQuestion,
    DecisionQuestionType,
    HumanEscalationPacket,
    MetaAction,
    MetaDecision,
    MetaDecisionDisposition,
    PrivacyClass,
    QuestionDisposition,
    RiskClass,
)

HUMAN_ESCALATION_COMPILER_INTERFACE: Final[str] = "HumanEscalationCompiler@1"
HUMAN_ESCALATION_COMPILER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/human-escalation-compiler@1"
)
MAX_COMPILER_QUESTIONS: Final[int] = 1_024
MAX_ID_BYTES: Final[int] = 512
MAX_QUESTION_BYTES: Final[int] = 2_048
MAX_OPTION_BYTES: Final[int] = 512
MIN_PACKET_OPTIONS: Final[int] = 2
MAX_PACKET_OPTIONS: Final[int] = 4
DEFAULT_TTL_MS: Final[int] = 3_600_000
SYNTHETIC_SAFE_OPTION: Final[str] = "keep_current_safest"
SYNTHETIC_REVIEW_OPTION: Final[str] = "request_authorized_review"

_NON_RESOLVING_ACTIONS = frozenset({MetaAction.NO_OP, MetaAction.QUARANTINE_TASK})
_NON_BYPASSABLE_CODES = frozenset(
    {
        "operator_only_authority",
        "irreversible_or_legal_effect",
        "privacy_policy_choice",
        "policy_required_budget_choice",
    }
)
_FULL_HISTORY_MARKERS: Final[tuple[str, ...]] = (
    "all context",
    "chain of thought",
    "complete history",
    "complete transcript",
    "dump logs",
    "dump the log",
    "entire graph",
    "entire history",
    "full history",
    "full-history",
    "full_history",
    "paste the repository",
    "raw prompt",
    "review all history",
    "review everything",
    "show all evidence",
    "source body",
    "unbounded context",
    "whole history",
)
_SAFEST_TOKENS: Final[tuple[str, ...]] = (
    "abort",
    "block",
    "defer",
    "deny",
    "hold",
    "keep",
    "no_change",
    "non_promote",
    "observe",
    "pause",
    "quarantine",
    "reject",
    "remain",
    "safest",
    "shadow",
    "wait",
)
_LOW_RISK = frozenset(
    {
        RiskClass.R0_PURE,
        RiskClass.R1_READ_ONLY,
        RiskClass.R2_REVERSIBLE_LOCAL,
    }
)


class HumanEscalationError(ValueError):
    """Raised when compiler inputs themselves are malformed."""


class HumanEscalationDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PACKET_COMPILED = "packet_compiled"
    NOT_ESCALATED = "not_escalated"


def _compact_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise HumanEscalationError(f"{name} must be a string")
    result = value.strip()
    if (
        not result
        or len(result.encode("utf-8")) > MAX_ID_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise HumanEscalationError(f"{name} must be a compact bounded identifier")
    return result


def _bounded_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (1 << 63) - 1:
        raise HumanEscalationError(f"{name} must be a bounded non-negative integer")
    return value


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise HumanEscalationError(f"{name} is unsupported") from exc


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def requests_full_history_review(value: object) -> bool:
    """Return True when text asks a human to review unbounded history."""

    if not isinstance(value, str) or not value.strip():
        return False
    normalized = _normalize_text(value)
    collapsed = normalized.replace(" ", "")
    if "fullhistory" in collapsed:
        return True
    return any(marker in normalized for marker in _FULL_HISTORY_MARKERS)


@dataclass(frozen=True)
class HumanEscalationContext:
    """Non-authoritative facts used to prove irreducibility."""

    policy_id: str = "policy:human-escalation"
    policy: AutonomyPolicy | None = None
    envelope: AutonomyEnvelope | None = None
    meta_decision: MetaDecision | None = None
    admitted_non_human_actions: frozenset[MetaAction] = field(default_factory=frozenset)
    no_admitted_non_human_route: bool = False
    required_authority_class: AuthorityClass = AuthorityClass.DERIVED
    privacy_class: PrivacyClass = PrivacyClass.LOCAL_ONLY
    privacy_choice_required: bool = False
    budget_choice_required: bool = False
    suppress_unnecessary: bool = True
    human_budget_remaining: int = 1
    model_budget_remaining: int = 0
    now_ms: int = 0
    default_ttl_ms: int = DEFAULT_TTL_MS

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _compact_identifier(self.policy_id, "policy_id"))
        if self.policy is not None and not isinstance(self.policy, AutonomyPolicy):
            raise HumanEscalationError("policy must be an AutonomyPolicy or None")
        if self.envelope is not None and not isinstance(self.envelope, AutonomyEnvelope):
            raise HumanEscalationError("envelope must be an AutonomyEnvelope or None")
        if self.meta_decision is not None and not isinstance(self.meta_decision, MetaDecision):
            raise HumanEscalationError("meta_decision must be a MetaDecision or None")
        raw_actions = self.admitted_non_human_actions
        if isinstance(raw_actions, str) or not isinstance(
            raw_actions, (set, frozenset, tuple, list)
        ):
            raise HumanEscalationError("admitted_non_human_actions must be a MetaAction set")
        normalized_actions: set[MetaAction] = set()
        for item in raw_actions:
            action = _enum(item, MetaAction, "admitted_non_human_actions")
            if action in _NON_RESOLVING_ACTIONS or action is MetaAction.REQUEST_HUMAN_DECISION:
                raise HumanEscalationError(
                    "admitted_non_human_actions cannot include human or non-resolving actions"
                )
            normalized_actions.add(action)
        object.__setattr__(self, "admitted_non_human_actions", frozenset(normalized_actions))
        for name in (
            "no_admitted_non_human_route",
            "privacy_choice_required",
            "budget_choice_required",
            "suppress_unnecessary",
        ):
            if not isinstance(getattr(self, name), bool):
                raise HumanEscalationError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "required_authority_class",
            _enum(self.required_authority_class, AuthorityClass, "required_authority_class"),
        )
        object.__setattr__(
            self, "privacy_class", _enum(self.privacy_class, PrivacyClass, "privacy_class")
        )
        for name in (
            "human_budget_remaining",
            "model_budget_remaining",
            "now_ms",
            "default_ttl_ms",
        ):
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name))
        if self.default_ttl_ms == 0:
            raise HumanEscalationError("default_ttl_ms must be positive")
        if self.envelope is not None and self.envelope.policy_id and self.policy is not None:
            if self.envelope.policy_id != self.policy.policy_id:
                raise HumanEscalationError("envelope policy_id does not match policy")
        if self.policy is not None and self.policy_id == "policy:human-escalation":
            object.__setattr__(self, "policy_id", self.policy.policy_id)
        elif self.envelope is not None and self.policy_id == "policy:human-escalation":
            object.__setattr__(self, "policy_id", self.envelope.policy_id)


@dataclass(frozen=True)
class HumanEscalationMetrics:
    """Integer-only outcome counters for one compile invocation."""

    questions_considered: int
    packets_emitted: int
    questions_batched: int
    questions_suppressed: int
    mandatory_decisions_preserved: int
    options_emitted: int
    full_history_requests_rejected: int
    non_escalations: int

    def __post_init__(self) -> None:
        for name in (
            "questions_considered",
            "packets_emitted",
            "questions_batched",
            "questions_suppressed",
            "mandatory_decisions_preserved",
            "options_emitted",
            "full_history_requests_rejected",
            "non_escalations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HumanEscalationError(f"{name} must be a non-negative integer")
        if self.packets_emitted > 1:
            raise HumanEscalationError("compiler may emit at most one packet")
        if self.options_emitted > MAX_PACKET_OPTIONS:
            raise HumanEscalationError("metrics cannot claim more than four options")

    def to_dict(self) -> Mapping[str, int]:
        payload = {
            "questions_considered": int(self.questions_considered),
            "packets_emitted": int(self.packets_emitted),
            "questions_batched": int(self.questions_batched),
            "questions_suppressed": int(self.questions_suppressed),
            "mandatory_decisions_preserved": int(self.mandatory_decisions_preserved),
            "options_emitted": int(self.options_emitted),
            "full_history_requests_rejected": int(self.full_history_requests_rejected),
            "non_escalations": int(self.non_escalations),
        }
        return MappingProxyType(payload)


@dataclass(frozen=True)
class HumanEscalationResult:
    """Closed compile outcome: one packet or a deterministic non-escalation."""

    disposition: HumanEscalationDisposition
    reason_codes: tuple[str, ...]
    metrics: HumanEscalationMetrics
    packet: HumanEscalationPacket | None = None
    question_ids: tuple[str, ...] = ()
    batched_question_ids: tuple[str, ...] = ()
    suppressed_question_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, HumanEscalationDisposition, "disposition"),
        )
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise HumanEscalationError("reason_codes must be a nonempty tuple")
        if any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_ID_BYTES
            or any(char.isspace() for char in item)
            for item in self.reason_codes
        ):
            raise HumanEscalationError("reason_codes must be compact identifiers")
        if not isinstance(self.metrics, HumanEscalationMetrics):
            raise HumanEscalationError("metrics must be HumanEscalationMetrics")
        for name in ("question_ids", "batched_question_ids", "suppressed_question_ids"):
            raw = getattr(self, name)
            if not isinstance(raw, tuple) or len(raw) > MAX_COMPILER_QUESTIONS:
                raise HumanEscalationError(f"{name} must be a bounded identity tuple")
            object.__setattr__(self, name, tuple(_compact_identifier(item, name) for item in raw))
        if self.disposition is HumanEscalationDisposition.PACKET_COMPILED:
            if not isinstance(self.packet, HumanEscalationPacket):
                raise HumanEscalationError("compiled disposition requires a packet")
            if self.metrics.packets_emitted != 1 or self.metrics.non_escalations != 0:
                raise HumanEscalationError("compiled metrics must record one packet")
        else:
            if self.packet is not None:
                raise HumanEscalationError("non-escalation cannot carry a packet")
            if self.metrics.packets_emitted != 0 or self.metrics.non_escalations != 1:
                raise HumanEscalationError("non-escalation metrics must record no packet")
        self._reject_full_history_packet()

    def _reject_full_history_packet(self) -> None:
        packet = self.packet
        if packet is None:
            return
        surfaces: list[object] = [packet.question, *packet.options, packet.recommended_option]
        surfaces.extend(packet.predicted_consequences.values())
        surfaces.extend(packet.continuation_by_option.values())
        for value in surfaces:
            if isinstance(value, str) and requests_full_history_review(value):
                raise HumanEscalationError("compiled packet asks for full-history review")
            if isinstance(value, Mapping):
                for nested in value.values():
                    if isinstance(nested, str) and requests_full_history_review(nested):
                        raise HumanEscalationError("compiled packet asks for full-history review")

    @property
    def escalated(self) -> bool:
        return self.disposition is HumanEscalationDisposition.PACKET_COMPILED


def _questions_from(
    question: DecisionQuestion | None,
    questions: tuple[DecisionQuestion, ...] | Sequence[DecisionQuestion],
) -> tuple[DecisionQuestion, ...]:
    collected: list[DecisionQuestion] = []
    if question is not None:
        collected.append(question)
    if isinstance(questions, str) or not isinstance(questions, Sequence):
        raise HumanEscalationError("questions must be a bounded sequence")
    if len(questions) > MAX_COMPILER_QUESTIONS:
        raise HumanEscalationError("questions exceed the compiler bound")
    collected.extend(questions)
    if any(not isinstance(item, DecisionQuestion) for item in collected):
        raise HumanEscalationError("questions must contain DecisionQuestion values")
    unique: list[DecisionQuestion] = []
    seen: set[str] = set()
    for item in collected:
        if item.question_id in seen:
            continue
        seen.add(item.question_id)
        unique.append(item)
    return tuple(unique)


def _max_risk(question: DecisionQuestion, context: HumanEscalationContext) -> RiskClass:
    candidates = [question.risk_if_incorrect, question.risk_if_left_unresolved]
    if context.envelope is not None:
        candidates.append(context.envelope.risk_assessment.risk_class)
    return max(candidates, key=lambda item: item.rank)


def _admissibly_terminal(question: DecisionQuestion) -> bool:
    return (
        question.disposition is QuestionDisposition.RESOLVED
        and bool(question.terminal_answer)
        and question.terminal_answer in question.current_alternatives
        and question.residual_uncertainty_bp == 0
        and not question.contradictory_evidence_ids
        and set(question.required_evidence_ids).issubset(question.known_evidence_ids)
    )


def _non_human_route_admitted(context: HumanEscalationContext) -> bool:
    if context.no_admitted_non_human_route:
        return False
    if context.admitted_non_human_actions:
        return True
    decision = context.meta_decision
    if decision is None:
        return False
    if decision.disposition is not MetaDecisionDisposition.SELECTED:
        return False
    return decision.selected_action not in _NON_RESOLVING_ACTIONS.union(
        {MetaAction.REQUEST_HUMAN_DECISION}
    )


def _scheduler_selected_human(context: HumanEscalationContext) -> bool:
    decision = context.meta_decision
    if decision is None:
        return False
    return (
        decision.selected_action is MetaAction.REQUEST_HUMAN_DECISION
        and decision.disposition
        in {MetaDecisionDisposition.SELECTED, MetaDecisionDisposition.ESCALATE}
    )


def _operator_required(question: DecisionQuestion, context: HumanEscalationContext) -> bool:
    if context.required_authority_class is AuthorityClass.OPERATOR_REQUIRED:
        return True
    policy = context.policy
    risk = _max_risk(question, context)
    if policy is not None and not policy.allows(AutonomyLevel.EXECUTE_REVERSIBLE, risk):
        return risk.rank >= RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.rank
    envelope = context.envelope
    if envelope is not None and envelope.autonomy_level.rank <= AutonomyLevel.RECOMMEND.rank:
        return risk.rank >= RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL.rank
    return False


def _question_text_blob(question: DecisionQuestion) -> str:
    parts = (
        question.terminal_decision_rule,
        question.question_type.value.replace("_", " "),
        *question.current_alternatives,
    )
    return " ".join(_normalize_text(part) for part in parts)


def _mentions_any(question: DecisionQuestion, *needles: str) -> bool:
    blob = _question_text_blob(question)
    return any(needle in blob for needle in needles)


def _privacy_choice_required(
    question: DecisionQuestion, context: HumanEscalationContext
) -> bool:
    if context.privacy_choice_required:
        return True
    if context.privacy_class not in {
        PrivacyClass.SENSITIVE,
        PrivacyClass.FORBIDDEN_EXTERNAL,
    }:
        return False
    if not (context.no_admitted_non_human_route or _scheduler_selected_human(context)):
        return False
    return _mentions_any(question, "privacy", "disclose", "disclosure", "sensitive", "secret")


def _budget_choice_required(
    question: DecisionQuestion, context: HumanEscalationContext
) -> bool:
    if (
        context.suppress_unnecessary
        and not question.mandatory
        and _max_risk(question, context) in _LOW_RISK
        and not question.contradictory_evidence_ids
    ):
        return False
    if context.budget_choice_required:
        return True
    if context.model_budget_remaining > 0 or context.human_budget_remaining == 0:
        return False
    if not (context.no_admitted_non_human_route or _scheduler_selected_human(context)):
        return False
    return _mentions_any(question, "budget", "human question")


def _irreducibility_codes(
    question: DecisionQuestion, context: HumanEscalationContext
) -> tuple[str, ...]:
    codes: list[str] = []
    risk = _max_risk(question, context)
    envelope = context.envelope
    irreversible = risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL or (
        envelope is not None
        and (
            envelope.risk_assessment.irreversible_external_effect
            or envelope.risk_assessment.legal_or_financial_effect
            or envelope.risk_assessment.risk_class
            is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL
        )
    )
    if irreversible:
        codes.append("irreversible_or_legal_effect")
    if _operator_required(question, context):
        codes.append("operator_only_authority")
    if question.contradictory_evidence_ids and question.residual_uncertainty_bp > 0:
        codes.append("irresolvable_contradiction")
    if _privacy_choice_required(question, context):
        codes.append("privacy_policy_choice")
    if _budget_choice_required(question, context):
        codes.append("policy_required_budget_choice")
    named_human = (
        question.question_type is DecisionQuestionType.WHETHER_HUMAN_CHOICE_IS_IRREDUCIBLE
    )
    if named_human and (
        context.no_admitted_non_human_route
        or _scheduler_selected_human(context)
        or irreversible
        or _operator_required(question, context)
        or question.residual_uncertainty_bp > 0
    ):
        codes.append("irreducible_ambiguity")
    if question.mandatory and codes:
        codes.append("mandatory_human_decision")
    # Preserve insertion order while dropping duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def _non_escalation_reason(
    question: DecisionQuestion, context: HumanEscalationContext
) -> str:
    if question.disposition is QuestionDisposition.INVALIDATED:
        return "question_invalidated"
    if question.disposition is QuestionDisposition.RESOLVED:
        return (
            "question_already_terminal"
            if _admissibly_terminal(question)
            else "inadmissible_terminal_claim"
        )
    if _non_human_route_admitted(context):
        return "admitted_non_human_route"
    if (
        context.suppress_unnecessary
        and not question.mandatory
        and _max_risk(question, context) in _LOW_RISK
        and not question.contradictory_evidence_ids
    ):
        return "unnecessary_question_suppressed"
    sanitized = _sanitize_options(question.current_alternatives)
    if (
        not question.mandatory
        and len(sanitized) < MIN_PACKET_OPTIONS
        and (
            any(requests_full_history_review(item) for item in question.current_alternatives)
            or requests_full_history_review(question.terminal_decision_rule)
        )
    ):
        return "full_history_review_forbidden"
    if not context.no_admitted_non_human_route and not _scheduler_selected_human(context):
        return "non_human_routes_not_exhausted"
    return "question_not_irreducible"


def classify_irreducibility(
    question: DecisionQuestion, context: HumanEscalationContext
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a question is irreducible and the closed reason codes."""

    if not isinstance(question, DecisionQuestion):
        raise HumanEscalationError("question must be a DecisionQuestion")
    if not isinstance(context, HumanEscalationContext):
        raise HumanEscalationError("context must be a HumanEscalationContext")
    if question.disposition is QuestionDisposition.INVALIDATED:
        return False, ("question_invalidated",)
    if question.disposition is QuestionDisposition.RESOLVED:
        reason = (
            "question_already_terminal"
            if _admissibly_terminal(question)
            else "inadmissible_terminal_claim"
        )
        return False, (reason,)
    codes = _irreducibility_codes(question, context)
    if _non_human_route_admitted(context):
        if codes and set(codes).intersection(_NON_BYPASSABLE_CODES):
            return True, codes
        return False, ("admitted_non_human_route",)
    if (
        context.suppress_unnecessary
        and not question.mandatory
        and _max_risk(question, context) in _LOW_RISK
        and not question.contradictory_evidence_ids
    ):
        return False, ("unnecessary_question_suppressed",)
    if codes:
        return True, codes
    return False, (_non_escalation_reason(question, context),)


def _sanitize_options(options: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for option in options:
        if not isinstance(option, str):
            continue
        text = " ".join(option.split())
        if not text or requests_full_history_review(text):
            continue
        if len(text.encode("utf-8")) > MAX_OPTION_BYTES:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _safest_rank(option: str) -> tuple[int, int, str]:
    normalized = _normalize_text(option)
    score = sum(-1 for token in _SAFEST_TOKENS if token.replace("_", " ") in normalized)
    return (score, len(option), option)


def _bound_options(options: Sequence[str], *, mandatory: bool) -> tuple[str, ...]:
    sanitized = _sanitize_options(options)
    if len(sanitized) < MIN_PACKET_OPTIONS:
        if not mandatory:
            return sanitized
        filled = list(sanitized)
        for synthetic in (SYNTHETIC_SAFE_OPTION, SYNTHETIC_REVIEW_OPTION):
            if synthetic not in filled:
                filled.append(synthetic)
            if len(filled) >= MIN_PACKET_OPTIONS:
                break
        sanitized = tuple(filled[:MAX_PACKET_OPTIONS])
    ranked = tuple(sorted(sanitized, key=_safest_rank))
    return ranked[:MAX_PACKET_OPTIONS]


def _batch_key(question: DecisionQuestion) -> tuple[str, str, str, tuple[str, ...]]:
    options = _sanitize_options(question.current_alternatives)
    if not options:
        options = tuple(sorted(question.current_alternatives))
    return (
        question.objective_id,
        question.question_type.value,
        question.terminal_decision_rule,
        tuple(sorted(options)),
    )


def _question_text(question: DecisionQuestion, options: Sequence[str]) -> str:
    type_label = question.question_type.value.replace("_", " ")
    rule = " ".join(question.terminal_decision_rule.split())
    if requests_full_history_review(rule):
        rule = "choose one bounded option"
    option_list = ", ".join(options)
    text = (
        f"Select one bounded option for {type_label} on {question.objective_id}: "
        f"{option_list}. Rule: {rule}."
    )
    if requests_full_history_review(text) or len(text.encode("utf-8")) > MAX_QUESTION_BYTES:
        text = (
            f"Select one bounded option for {type_label} on {question.objective_id}: "
            f"{option_list}."
        )
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_QUESTION_BYTES:
        text = encoded[:MAX_QUESTION_BYTES].decode("utf-8", errors="ignore").rstrip()
    return text


def _continuation(option: str) -> str:
    normalized = _normalize_text(option).replace(" ", "_")
    if any(
        token in normalized
        for token in ("keep", "defer", "hold", "wait", "shadow", "observe", "no_change", "safest")
    ):
        return "record_non_promotion"
    if any(token in normalized for token in ("reject", "deny", "abort", "quarantine", "block")):
        return "quarantine_and_stop"
    if any(
        token in normalized
        for token in ("review", "approve", "authorize", "release", "promote", "proceed")
    ):
        return "await_authority"
    return "await_scoped_decision"


def _consequence(option: str, *, risk: RiskClass, mandatory: bool) -> str:
    continuation = _continuation(option)
    if continuation == "record_non_promotion":
        return "No live routing or authority change; the current safest state is retained."
    if continuation == "quarantine_and_stop":
        return "The blocked criterion stays unresolved and the task remains quarantined."
    if continuation == "await_authority":
        if risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
            return "An authorized operator decision is queued before any irreversible effect."
        return "An authorized review is queued before the selected continuation proceeds."
    if mandatory:
        return "The selected bounded continuation is recorded against the blocked criterion."
    return "The selected bounded continuation is recorded without expanding authority."


def _cost_and_risk(
    option: str, *, risk: RiskClass, authority: AuthorityClass
) -> Mapping[str, object]:
    continuation = _continuation(option)
    reversible = (
        continuation != "await_authority"
        or risk.rank < RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL.rank
    )
    return MappingProxyType(
        {
            "risk_class": risk.value,
            "authority_class": authority.value,
            "human_cost_units": 1,
            "reversible": reversible,
            "continuation": continuation,
        }
    )


def _expiry_ms(questions: Sequence[DecisionQuestion], context: HumanEscalationContext) -> int:
    candidates = [context.now_ms + context.default_ttl_ms]
    for question in questions:
        if question.decision_deadline_ms > 0:
            candidates.append(question.decision_deadline_ms)
    if context.envelope is not None and context.envelope.expiry_ms > 0:
        candidates.append(context.envelope.expiry_ms)
    return min(candidates)


def _evidence_ids(questions: Sequence[DecisionQuestion]) -> tuple[str, ...]:
    evidence: list[str] = []
    for question in questions:
        for item in question.known_evidence_ids + question.required_evidence_ids:
            if item not in evidence:
                evidence.append(item)
    if evidence:
        return tuple(sorted(set(evidence)))
    return tuple(sorted({question.question_id for question in questions}))


def _blocked_criteria(questions: Sequence[DecisionQuestion]) -> tuple[str, ...]:
    criteria: list[str] = []
    for question in questions:
        for item in question.acceptance_criterion_ids:
            if item not in criteria:
                criteria.append(item)
    return tuple(sorted(set(criteria)))


def _compile_packet(
    questions: Sequence[DecisionQuestion], context: HumanEscalationContext
) -> HumanEscalationPacket:
    representative = min(questions, key=lambda item: item.question_id)
    option_pool: list[str] = []
    for question in questions:
        for option in question.current_alternatives:
            if option not in option_pool:
                option_pool.append(option)
    risk = max((_max_risk(item, context) for item in questions), key=lambda item: item.rank)
    mandatory = any(item.mandatory for item in questions) or risk is (
        RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL
    )
    options = _bound_options(option_pool, mandatory=True)
    if len(options) < MIN_PACKET_OPTIONS:
        raise HumanEscalationError("irreducible question lacks two bounded options")
    authority = (
        AuthorityClass.OPERATOR_REQUIRED
        if _operator_required(representative, context)
        or risk is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL
        else context.required_authority_class
    )
    recommended = options[0]
    return HumanEscalationPacket(
        objective_id=representative.objective_id,
        blocked_criterion_ids=_blocked_criteria(questions),
        question=_question_text(representative, options),
        options=options,
        recommended_option=recommended,
        predicted_consequences={
            option: _consequence(option, risk=risk, mandatory=mandatory) for option in options
        },
        cost_and_risk={
            option: dict(_cost_and_risk(option, risk=risk, authority=authority))
            for option in options
        },
        evidence_ids=_evidence_ids(questions),
        continuation_by_option={option: _continuation(option) for option in options},
        expires_at_ms=_expiry_ms(questions, context),
    )


def _metrics(
    *,
    considered: int,
    packet: HumanEscalationPacket | None,
    batched: Sequence[str],
    suppressed: Sequence[str],
    mandatory_preserved: int,
    full_history_rejected: int,
) -> HumanEscalationMetrics:
    emitted = 1 if packet is not None else 0
    return HumanEscalationMetrics(
        questions_considered=considered,
        packets_emitted=emitted,
        questions_batched=len(batched),
        questions_suppressed=len(suppressed),
        mandatory_decisions_preserved=mandatory_preserved,
        options_emitted=0 if packet is None else len(packet.options),
        full_history_requests_rejected=full_history_rejected,
        non_escalations=0 if packet is not None else 1,
    )


class HumanEscalationCompiler:
    """Pure compiler for :class:`HumanEscalationPacket` values."""

    interface = HUMAN_ESCALATION_COMPILER_INTERFACE

    def compile(
        self,
        *,
        question: DecisionQuestion | None = None,
        questions: tuple[DecisionQuestion, ...] = (),
        context: HumanEscalationContext | None = None,
    ) -> HumanEscalationResult:
        if context is None:
            context = HumanEscalationContext()
        if not isinstance(context, HumanEscalationContext):
            raise HumanEscalationError("context must be a HumanEscalationContext")
        collected = _questions_from(question, questions)
        if not collected:
            return HumanEscalationResult(
                disposition=HumanEscalationDisposition.NOT_ESCALATED,
                reason_codes=("no_named_unresolved_question",),
                metrics=_metrics(
                    considered=0,
                    packet=None,
                    batched=(),
                    suppressed=(),
                    mandatory_preserved=0,
                    full_history_rejected=0,
                ),
            )
        objectives = {item.objective_id for item in collected}
        if len(objectives) != 1:
            raise HumanEscalationError("questions must share one objective_id")

        classifications = tuple(
            (item, *classify_irreducibility(item, context)) for item in collected
        )
        full_history_rejected = sum(
            1
            for item, _, _ in classifications
            if any(requests_full_history_review(option) for option in item.current_alternatives)
            or requests_full_history_review(item.terminal_decision_rule)
        )
        irreducible = tuple(item for item, flag, _ in classifications if flag)
        suppressed_ids = tuple(
            sorted(
                item.question_id
                for item, flag, reasons in classifications
                if not flag and "unnecessary_question_suppressed" in reasons
            )
        )
        if not irreducible:
            reasons = tuple(
                sorted({code for _, flag, codes in classifications if not flag for code in codes})
            )
            primary = min(classifications, key=lambda item: item[0].question_id)
            reason_codes = (primary[2][0],) + tuple(
                code for code in reasons if code != primary[2][0]
            )
            return HumanEscalationResult(
                disposition=HumanEscalationDisposition.NOT_ESCALATED,
                reason_codes=reason_codes,
                metrics=_metrics(
                    considered=len(collected),
                    packet=None,
                    batched=(),
                    suppressed=suppressed_ids,
                    mandatory_preserved=0,
                    full_history_rejected=full_history_rejected,
                ),
                question_ids=tuple(item.question_id for item in collected),
                suppressed_question_ids=suppressed_ids,
            )

        groups: dict[tuple[str, str, str, tuple[str, ...]], list[DecisionQuestion]] = {}
        for item in irreducible:
            groups.setdefault(_batch_key(item), []).append(item)

        def _group_rank(
            key: tuple[str, str, str, tuple[str, ...]],
        ) -> tuple[int, int, int, str, tuple[str, str, str, tuple[str, ...]]]:
            members = groups[key]
            risk = max((_max_risk(item, context).rank for item in members), default=0)
            mandatory = 1 if any(item.mandatory for item in members) else 0
            return (-risk, -mandatory, -len(members), key[0], key)

        selected_key = sorted(groups, key=_group_rank)[0]
        selected = tuple(sorted(groups[selected_key], key=lambda item: item.question_id))
        packet = _compile_packet(selected, context)
        reason_codes: list[str] = ["packet_compiled"]
        for item in selected:
            _, codes = classify_irreducibility(item, context)
            for code in codes:
                if code not in reason_codes:
                    reason_codes.append(code)
        if len(selected) > 1:
            reason_codes.append("batched_equivalent_questions")
        mandatory_preserved = 1 if any(item.mandatory for item in selected) else 0
        batched_ids = tuple(item.question_id for item in selected)
        return HumanEscalationResult(
            disposition=HumanEscalationDisposition.PACKET_COMPILED,
            reason_codes=tuple(reason_codes),
            metrics=_metrics(
                considered=len(collected),
                packet=packet,
                batched=batched_ids if len(batched_ids) > 1 else (),
                suppressed=suppressed_ids,
                mandatory_preserved=mandatory_preserved,
                full_history_rejected=full_history_rejected,
            ),
            packet=packet,
            question_ids=tuple(item.question_id for item in collected),
            batched_question_ids=batched_ids if len(batched_ids) > 1 else (),
            suppressed_question_ids=suppressed_ids,
        )


__all__ = [
    "DEFAULT_TTL_MS",
    "HUMAN_ESCALATION_COMPILER_INTERFACE",
    "HUMAN_ESCALATION_COMPILER_SCHEMA",
    "HumanEscalationCompiler",
    "HumanEscalationContext",
    "HumanEscalationDisposition",
    "HumanEscalationError",
    "HumanEscalationMetrics",
    "HumanEscalationResult",
    "classify_irreducibility",
    "requests_full_history_review",
]
