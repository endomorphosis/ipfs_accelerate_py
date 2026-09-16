"""Apply TypeSafe nominations onto a decision graph without resolving it.

TypeSafe evidence is ``model_advice``. It cannot zero residual uncertainty
or mark a question admissibly terminal. The returned ``next_action`` is the
authoritative follow-up (SMT, strong model, static analysis, or human).

``AutonomousMetaController.step`` stays provider-free. Advise first with
:func:`prepare_step_candidates`, then pass the rebound candidates into
``step``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping, Optional

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    DecisionQuestionAdvice,
    advise_decision_question,
    typesafe_permitted,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    DecisionQuestion,
    ResolutionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.decision_graph import (
    DecisionGraphController,
    question_is_admissibly_terminal,
)


def advise_unresolved_question(
    question: DecisionQuestion,
    *,
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> DecisionQuestionAdvice:
    """Nominate an allowlisted answer for one unresolved decision question."""

    question_type = getattr(question.question_type, "value", question.question_type)
    return advise_decision_question(
        question_id=question.question_id,
        question_type=str(question_type),
        alternatives=question.current_alternatives,
        state=dict(state),
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )


def apply_typesafe_question_advice(
    controller: DecisionGraphController,
    question: DecisionQuestion,
    *,
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> DecisionQuestionAdvice:
    """Record advisory evidence. Never calls ``resolve``."""

    advice = advise_unresolved_question(
        question,
        state=state,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )
    controller.record_evidence(
        question.question_id,
        evidence_ids=(advice.evidence_id,),
        contradictory=False,
    )
    updated = next(
        item
        for item in controller.graph.questions
        if item.question_id == question.question_id
        or advice.evidence_id in item.known_evidence_ids
    )
    if question_is_admissibly_terminal(updated):
        raise RuntimeError("typesafe advice must not terminate a decision question")
    if updated.residual_uncertainty_bp <= 0:
        raise RuntimeError("typesafe advice must leave residual uncertainty")
    return advice


def _action_name(candidate: ResolutionCandidate) -> str:
    action = candidate.resolution_action.action
    return str(getattr(action, "value", action))


def rebind_candidates(
    candidates: Sequence[ResolutionCandidate],
    *,
    previous_question_id: str,
    question: DecisionQuestion,
    evidence_id: str = "",
) -> tuple[ResolutionCandidate, ...]:
    """Point candidates at the post-advice question identity."""

    rebound: list[ResolutionCandidate] = []
    extra = (evidence_id,) if evidence_id else ()
    for candidate in candidates:
        if candidate.question_id != previous_question_id:
            rebound.append(candidate)
            continue
        evidence = tuple(dict.fromkeys((*candidate.evidence_ids, *extra)))
        rebound.append(
            ResolutionCandidate(
                question_id=question.question_id,
                resolution_action=candidate.resolution_action,
                expected_decision_value=candidate.expected_decision_value,
                admissible=candidate.admissible,
                policy_id=candidate.policy_id,
                reason_codes=candidate.reason_codes,
                evidence_ids=evidence,
            )
        )
    return tuple(rebound)


def prefer_escalation_candidates(
    candidates: Sequence[ResolutionCandidate],
    advice: DecisionQuestionAdvice,
) -> tuple[ResolutionCandidate, ...]:
    """Keep matching ``next_action`` candidates; fail open if none match."""

    wanted = str(advice.next_action or "")
    matching = tuple(item for item in candidates if _action_name(item) == wanted)
    return matching if matching else tuple(candidates)


def prepare_step_candidates(
    controller: DecisionGraphController,
    question: DecisionQuestion,
    candidates: Sequence[ResolutionCandidate],
    *,
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
    call_typesafe: Optional[bool] = None,
) -> tuple[tuple[ResolutionCandidate, ...], DecisionQuestion, Optional[DecisionQuestionAdvice]]:
    """Advise one unresolved WHETHER/WHICH question, then filter candidates.

    Does not resolve the graph. Pass the returned candidates into
    ``AutonomousMetaController.step``.
    """

    question_type = str(getattr(question.question_type, "value", question.question_type))
    should = question_type.startswith("whether_") or question_type.startswith("which_")
    permitted = typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    )
    if call_typesafe is None:
        call_typesafe = should and permitted
    if not should or not call_typesafe:
        return tuple(candidates), question, None
    previous_id = question.question_id
    advice = apply_typesafe_question_advice(
        controller,
        question,
        state=state,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )
    updated = next(
        item
        for item in controller.graph.questions
        if advice.evidence_id in item.known_evidence_ids
    )
    rebound = rebind_candidates(
        candidates,
        previous_question_id=previous_id,
        question=updated,
        evidence_id=advice.evidence_id,
    )
    return prefer_escalation_candidates(rebound, advice), updated, advice


def admit_after_typesafe_advice(
    meta: Any,
    *,
    candidates: Sequence[ResolutionCandidate],
    context: Any,
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    timeout: float = 30.0,
    meaningful_change: bool = True,
) -> tuple[Any, Optional[DecisionQuestionAdvice]]:
    """Advise, then run the provider-free ``step`` on the filtered set."""

    question = meta.next_unresolved_question()
    advice = None
    prepared: Sequence[ResolutionCandidate] = candidates
    if question is not None and meaningful_change:
        remote = bool(getattr(context, "remote_disclosure_permitted", False))
        prepared, _updated, advice = prepare_step_candidates(
            meta.decision_graph,
            question,
            candidates,
            state=state,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote,
            timeout=timeout,
        )
    step = meta.step(
        candidates=tuple(prepared),
        context=context,
        meaningful_change=meaningful_change,
    )
    return step, advice


__all__ = [
    "admit_after_typesafe_advice",
    "advise_unresolved_question",
    "apply_typesafe_question_advice",
    "prefer_escalation_candidates",
    "prepare_step_candidates",
    "rebind_candidates",
]
