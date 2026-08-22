"""Deterministic software-first scheduling for named decision questions.

The scheduler ranks already-declared :class:`ResolutionCandidate` values.  It
does not execute them, call a provider, reserve resources, admit an effect, or
grant authority.  Callers must reserve the selected cost through the objective
budget ledger and must still pass every effect through ``DecisionRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Final

from .contracts import (
    AuthorityClass,
    BudgetLedger,
    BudgetReservationStatus,
    DecisionQuestion,
    MetaAction,
    MetaDecision,
    MetaDecisionDisposition,
    PrivacyClass,
    QuestionDisposition,
    ResolutionCandidate,
)

NO_UNRESOLVED_QUESTION_ID: Final[str] = "apmc-no-unresolved-question"

_MODEL_ACTIONS = frozenset(
    {
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)
_REMOTE_MODEL_ACTIONS = frozenset(
    {
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)
_NON_RESOLVING_ACTIONS = frozenset({MetaAction.NO_OP, MetaAction.QUARANTINE_TASK})
_VALIDATION_ACTIONS = frozenset(
    {
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        MetaAction.RUN_SCHEMA_VALIDATION,
        MetaAction.RUN_TYPE_CHECK,
        MetaAction.RUN_SELECTED_TEST,
        MetaAction.RUN_FULL_VALIDATION,
    }
)

# Hard route classes implement the closed default precedence.  Utility is used
# only within a class, so a high advisory-model score cannot outrank current
# authoritative evidence or deterministic software.
_ROUTE_CLASS: Final[dict[MetaAction, int]] = {
    MetaAction.NO_OP: 0,
    MetaAction.QUARANTINE_TASK: 8,
    MetaAction.READ_CACHED_RECEIPT: 0,
    MetaAction.RUN_LOCAL_STATIC_ANALYSIS: 1,
    MetaAction.RUN_INCREMENTAL_INDEX_QUERY: 1,
    MetaAction.RUN_GRAPH_RETRIEVAL: 1,
    MetaAction.RUN_SCHEMA_VALIDATION: 1,
    MetaAction.RUN_TYPE_CHECK: 1,
    MetaAction.REPLAN_AFFECTED_SUFFIX: 1,
    MetaAction.EXPAND_CONTEXT_REFERENCE: 2,
    MetaAction.RUN_SELECTED_TEST: 3,
    MetaAction.RUN_FULL_VALIDATION: 3,
    MetaAction.RUN_SMT_OR_PROVER: 3,
    MetaAction.GENERATE_BOUNDED_REPAIR: 3,
    MetaAction.CALL_LOCAL_SMALL_MODEL: 4,
    MetaAction.CALL_REMOTE_STANDARD_MODEL: 5,
    MetaAction.CALL_REMOTE_STRONG_MODEL: 6,
    MetaAction.REQUEST_HUMAN_DECISION: 7,
}


def _route_class(candidate: ResolutionCandidate) -> int:
    """Return the closed precedence class for one admitted candidate.

    A cached receipt is first only when it already carries the current
    authoritative evidence class.  Derived or merely verified cached
    analysis remains the third route, after deterministic software.  This
    prevents cache locality from silently upgrading evidence authority.
    """

    action = candidate.resolution_action
    if action.action is MetaAction.READ_CACHED_RECEIPT:
        return 0 if action.authority_class is AuthorityClass.AUTHORITATIVE else 2
    return _ROUTE_CLASS[action.action]


class CognitiveSchedulingError(ValueError):
    """Raised when scheduler inputs themselves are malformed."""


@dataclass(frozen=True)
class CognitiveSchedulingContext:
    """Current non-authoritative facts used by the deterministic selector."""

    policy_id: str
    satisfied_precondition_ids: frozenset[str] = field(default_factory=frozenset)
    current_result_action_ids: frozenset[str] = field(default_factory=frozenset)
    repeated_failure_action_ids: frozenset[str] = field(default_factory=frozenset)
    local_small_model_available: bool = False
    remote_standard_model_available: bool = False
    remote_strong_model_available: bool = False
    remote_disclosure_permitted: bool = False
    human_request_permitted: bool = True
    new_evidence_since_failure: bool = False
    protected_validation_input_tokens: int = 0
    required_authority_class: AuthorityClass = AuthorityClass.DERIVED

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_id, str)
            or not self.policy_id
            or len(self.policy_id.encode("utf-8")) > 512
            or any(char.isspace() for char in self.policy_id)
        ):
            raise CognitiveSchedulingError("policy_id must be a bounded nonempty identifier")
        if (
            isinstance(self.protected_validation_input_tokens, bool)
            or not isinstance(self.protected_validation_input_tokens, int)
            or not 0 <= self.protected_validation_input_tokens <= (1 << 63) - 1
        ):
            raise CognitiveSchedulingError("protected validation tokens cannot be negative")
        for name in (
            "local_small_model_available",
            "remote_standard_model_available",
            "remote_strong_model_available",
            "remote_disclosure_permitted",
            "human_request_permitted",
            "new_evidence_since_failure",
        ):
            if not isinstance(getattr(self, name), bool):
                raise CognitiveSchedulingError(f"{name} must be a boolean")
        for name in (
            "satisfied_precondition_ids",
            "current_result_action_ids",
            "repeated_failure_action_ids",
        ):
            raw = getattr(self, name)
            if isinstance(raw, str) or not isinstance(raw, (set, frozenset, tuple, list)):
                raise CognitiveSchedulingError(f"{name} must be a bounded identifier set")
            if len(raw) > 4096 or any(
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > 512
                or any(char.isspace() for char in value)
                for value in raw
            ):
                raise CognitiveSchedulingError(f"{name} is unbounded or malformed")
            object.__setattr__(self, name, frozenset(raw))
        try:
            authority = (
                self.required_authority_class
                if isinstance(self.required_authority_class, AuthorityClass)
                else AuthorityClass(str(self.required_authority_class))
            )
        except (TypeError, ValueError) as exc:
            raise CognitiveSchedulingError("required_authority_class is unsupported") from exc
        object.__setattr__(self, "required_authority_class", authority)


@dataclass(frozen=True)
class _Rejected:
    candidate: ResolutionCandidate
    reason: str


def _authority_rank(authority: AuthorityClass) -> int:
    return tuple(AuthorityClass).index(authority)


def _active_reserved(ledger: BudgetLedger, field: str) -> int:
    return sum(
        getattr(item, field)
        for item in ledger.reservations
        if item.status is BudgetReservationStatus.RESERVED
    )


def _remaining(ledger: BudgetLedger, maximum: str) -> int:
    suffix = maximum.removeprefix("max_")
    committed = getattr(ledger, f"committed_{suffix}")
    reserved = _active_reserved(ledger, maximum)
    return getattr(ledger.budget, maximum) - committed - reserved


def _budget_rejection(
    candidate: ResolutionCandidate,
    *,
    ledger: BudgetLedger,
    context: CognitiveSchedulingContext,
) -> str:
    action = candidate.resolution_action
    kind = action.action
    if kind is MetaAction.NO_OP:
        return "answer_cannot_change_decision"
    is_model = kind in _MODEL_ACTIONS
    is_strong = kind is MetaAction.CALL_REMOTE_STRONG_MODEL
    if is_model and _remaining(ledger, "max_total_model_calls") < 1:
        return "total_model_call_budget_exhausted"
    if is_strong and _remaining(ledger, "max_strong_model_calls") < 1:
        return "strong_model_call_budget_exhausted"
    if _remaining(ledger, "max_input_tokens") < action.token_cost:
        return "input_token_budget_exhausted"
    if is_model and (
        _remaining(ledger, "max_input_tokens") - action.token_cost
        < context.protected_validation_input_tokens
    ):
        return "protected_validation_token_reserve"
    # ResolutionAction deliberately exposes one conservative token bound
    # rather than provider-specific input/output estimates.  A model call must
    # fit that bound independently in both ledgers; unused capacity is released
    # when authoritative measurements are reconciled.
    if is_model and _remaining(ledger, "max_output_tokens") < action.token_cost:
        return "output_token_budget_exhausted"
    if _remaining(ledger, "max_provider_spend_micros") < action.provider_cost_micros:
        return "provider_spend_budget_exhausted"
    if _remaining(ledger, "max_wall_time_ms") < action.latency_cost_ms:
        return "wall_time_budget_exhausted"
    if (
        kind is MetaAction.RUN_SMT_OR_PROVER
        and _remaining(ledger, "max_proof_time_ms") < action.latency_cost_ms
    ):
        return "proof_time_budget_exhausted"
    if kind in _VALIDATION_ACTIONS:
        if _remaining(ledger, "max_validation_time_ms") < action.latency_cost_ms:
            return "validation_time_budget_exhausted"
    elif _remaining(ledger, "max_validation_time_ms") < ledger.budget.validation_reserve_ms:
        return "protected_validation_time_reserve"
    if kind is MetaAction.REQUEST_HUMAN_DECISION and _remaining(ledger, "max_human_questions") < 1:
        return "human_question_budget_exhausted"
    if kind is MetaAction.GENERATE_BOUNDED_REPAIR and _remaining(ledger, "max_repair_rounds") < 1:
        return "repair_round_budget_exhausted"
    if kind is MetaAction.REPLAN_AFFECTED_SUFFIX and _remaining(ledger, "max_plan_branches") < 1:
        return "plan_branch_budget_exhausted"
    if (
        kind is MetaAction.EXPAND_CONTEXT_REFERENCE
        and _remaining(ledger, "max_context_expansions") < 1
    ):
        return "context_expansion_budget_exhausted"
    return ""


def _hard_rejection(
    candidate: ResolutionCandidate,
    *,
    question: DecisionQuestion,
    ledger: BudgetLedger,
    context: CognitiveSchedulingContext,
    deterministic_authority_exists: bool,
) -> str:
    action = candidate.resolution_action
    kind = action.action
    if candidate.question_id != question.question_id:
        return "candidate_question_mismatch"
    if candidate.policy_id != context.policy_id:
        return "candidate_policy_mismatch"
    if not candidate.admissible:
        return "candidate_not_admissible"
    if (
        question.possible_resolution_action_ids
        and action.action_id not in question.possible_resolution_action_ids
    ):
        return "action_not_declared_for_question"
    if not set(action.precondition_ids).issubset(context.satisfied_precondition_ids):
        return "preconditions_unsatisfied"
    if candidate.expected_decision_value <= 0 or not action.can_change_decision:
        return "answer_cannot_change_decision"
    if action.action_id in context.current_result_action_ids:
        return "current_identical_result_exists"
    if (
        action.action_id in context.repeated_failure_action_ids
        and not context.new_evidence_since_failure
    ):
        return "repeated_failure_without_new_evidence"
    if kind in _MODEL_ACTIONS and deterministic_authority_exists:
        return "deterministic_method_is_authoritative"
    if kind is MetaAction.CALL_LOCAL_SMALL_MODEL and not context.local_small_model_available:
        return "local_model_unavailable"
    if (
        kind is MetaAction.CALL_REMOTE_STANDARD_MODEL
        and not context.remote_standard_model_available
    ):
        return "remote_standard_model_unavailable"
    if kind is MetaAction.CALL_REMOTE_STRONG_MODEL and not context.remote_strong_model_available:
        return "remote_strong_model_unavailable"
    if kind in _REMOTE_MODEL_ACTIONS and not context.remote_disclosure_permitted:
        return "privacy_policy_forbids_disclosure"
    if kind in _REMOTE_MODEL_ACTIONS and action.privacy_class in {
        PrivacyClass.LOCAL_ONLY,
        PrivacyClass.FORBIDDEN_EXTERNAL,
    }:
        return "privacy_class_forbids_remote_route"
    if kind is MetaAction.REQUEST_HUMAN_DECISION and not context.human_request_permitted:
        return "human_request_not_permitted"
    if (
        _authority_rank(action.authority_class) < _authority_rank(context.required_authority_class)
        or not action.accepted_as_authority
    ):
        return "result_not_accepted_as_authority"
    return _budget_rejection(candidate, ledger=ledger, context=context)


def _cost(candidate: ResolutionCandidate) -> int:
    action = candidate.resolution_action
    return max(
        1,
        action.token_cost
        + action.latency_cost_ms
        + action.provider_cost_micros
        + action.resource_cost_units
        + action.invalidation_cost_units
        + action.privacy_cost_units,
    )


def _compare(left: ResolutionCandidate, right: ResolutionCandidate) -> int:
    left_route = _route_class(left)
    right_route = _route_class(right)
    if left_route != right_route:
        return -1 if left_route < right_route else 1
    # Compare value/cost without floats.  Higher utility sorts first.
    cross_left = left.expected_decision_value * _cost(right)
    cross_right = right.expected_decision_value * _cost(left)
    if cross_left != cross_right:
        return -1 if cross_left > cross_right else 1
    return (
        -1
        if left.candidate_id < right.candidate_id
        else (1 if left.candidate_id > right.candidate_id else 0)
    )


class CognitiveScheduler:
    """Pure selector implementing hard constraints and closed route precedence."""

    def select(
        self,
        *,
        question: DecisionQuestion | None,
        candidates: tuple[ResolutionCandidate, ...],
        budget_ledger: BudgetLedger,
        context: CognitiveSchedulingContext,
    ) -> MetaDecision:
        if not isinstance(budget_ledger, BudgetLedger):
            raise CognitiveSchedulingError("budget_ledger must be a BudgetLedger")
        if not isinstance(context, CognitiveSchedulingContext):
            raise CognitiveSchedulingError("context must be a CognitiveSchedulingContext")
        if not isinstance(candidates, tuple) or len(candidates) > 1_024:
            raise CognitiveSchedulingError("candidates must be a bounded tuple")
        if any(not isinstance(candidate, ResolutionCandidate) for candidate in candidates):
            raise CognitiveSchedulingError("candidates must contain ResolutionCandidate values")
        if question is not None and not isinstance(question, DecisionQuestion):
            raise CognitiveSchedulingError("question must be a DecisionQuestion or None")
        if question is None:
            return MetaDecision(
                question_id=NO_UNRESOLVED_QUESTION_ID,
                selected_candidate_id="",
                selected_action=MetaAction.NO_OP,
                considered_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                rejected_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                evidence_ids=(),
                reservation_id="",
                policy_id=context.policy_id,
                disposition=MetaDecisionDisposition.NO_OP,
                reason_codes=("no_named_unresolved_question",),
            )
        if question.disposition is QuestionDisposition.RESOLVED:
            admissibly_terminal = (
                bool(question.terminal_answer)
                and question.terminal_answer in question.current_alternatives
                and question.residual_uncertainty_bp == 0
                and not question.contradictory_evidence_ids
                and set(question.required_evidence_ids).issubset(question.known_evidence_ids)
            )
            disposition = (
                MetaDecisionDisposition.NO_OP
                if admissibly_terminal
                else MetaDecisionDisposition.BLOCKED
            )
            return MetaDecision(
                question_id=question.question_id,
                selected_candidate_id="",
                selected_action=MetaAction.NO_OP,
                considered_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                rejected_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                evidence_ids=question.known_evidence_ids,
                reservation_id="",
                policy_id=context.policy_id,
                disposition=disposition,
                reason_codes=(
                    "question_already_terminal"
                    if admissibly_terminal
                    else "inadmissible_terminal_claim",
                ),
            )
        if question.disposition is QuestionDisposition.BLOCKED:
            return MetaDecision(
                question_id=question.question_id,
                selected_candidate_id="",
                selected_action=MetaAction.NO_OP,
                considered_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                rejected_candidate_ids=tuple(sorted(item.candidate_id for item in candidates)),
                evidence_ids=question.known_evidence_ids,
                reservation_id="",
                policy_id=context.policy_id,
                disposition=MetaDecisionDisposition.BLOCKED,
                reason_codes=("question_disposition_blocked",),
            )

        # A deterministic route suppresses a model only when that route would
        # itself pass every current hard constraint.  A stale-policy,
        # repeated-failure, or over-budget software candidate is not an
        # authoritative method available *now*.
        deterministic_authority_exists = any(
            candidate.resolution_action.action not in _MODEL_ACTIONS
            and candidate.resolution_action.action not in _NON_RESOLVING_ACTIONS
            and not _hard_rejection(
                candidate,
                question=question,
                ledger=budget_ledger,
                context=context,
                deterministic_authority_exists=False,
            )
            for candidate in candidates
        )
        accepted: list[ResolutionCandidate] = []
        rejected: list[_Rejected] = []
        for candidate in candidates:
            reason = _hard_rejection(
                candidate,
                question=question,
                ledger=budget_ledger,
                context=context,
                deterministic_authority_exists=deterministic_authority_exists,
            )
            if reason:
                rejected.append(_Rejected(candidate, reason))
            else:
                accepted.append(candidate)

        considered = tuple(sorted(item.candidate_id for item in candidates))
        rejected_ids = tuple(sorted(item.candidate.candidate_id for item in rejected))
        if not accepted:
            reasons = tuple(sorted({item.reason for item in rejected})) or (
                "no_resolution_candidate",
            )
            disposition = (
                MetaDecisionDisposition.QUARANTINE
                if any(
                    "repeated_failure" in reason or "budget_exhausted" in reason
                    for reason in reasons
                )
                else MetaDecisionDisposition.BLOCKED
            )
            return MetaDecision(
                question_id=question.question_id,
                selected_candidate_id="",
                selected_action=(
                    MetaAction.QUARANTINE_TASK
                    if disposition is MetaDecisionDisposition.QUARANTINE
                    else MetaAction.NO_OP
                ),
                considered_candidate_ids=considered,
                rejected_candidate_ids=rejected_ids,
                evidence_ids=question.known_evidence_ids,
                reservation_id="",
                policy_id=context.policy_id,
                disposition=disposition,
                reason_codes=reasons,
            )

        selected = sorted(accepted, key=cmp_to_key(_compare))[0]
        return MetaDecision(
            question_id=question.question_id,
            selected_candidate_id=selected.candidate_id,
            selected_action=selected.resolution_action.action,
            considered_candidate_ids=considered,
            rejected_candidate_ids=rejected_ids,
            evidence_ids=tuple(
                sorted(set(question.known_evidence_ids).union(selected.evidence_ids))
            ),
            reservation_id="",
            policy_id=context.policy_id,
            disposition=MetaDecisionDisposition.SELECTED,
            reason_codes=(
                "hard_constraints_passed",
                "closed_route_precedence",
                "highest_integer_utility",
            ),
        )


__all__ = [
    "CognitiveScheduler",
    "CognitiveSchedulingContext",
    "CognitiveSchedulingError",
    "NO_UNRESOLVED_QUESTION_ID",
]
