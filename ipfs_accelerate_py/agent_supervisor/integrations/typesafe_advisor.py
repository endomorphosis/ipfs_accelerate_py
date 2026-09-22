"""Advisory TypeSafe System One adapter for the agent supervisor.

TypeSafe may nominate, filter, and order. It cannot be accepted as
authority, cannot replace kernel/z3, and cannot synthesize overlays.
Records never store API keys, prompts, or proof text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import (
    Choice,
    Noul,
    Score,
    typesafe_configured,
)

ADVISOR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/typesafe-advisor-receipt@1"
AUTHORITY_CLASS = "advisory"
PROVIDER_NAME = "typesafe"

HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.60
NOUL_UNCERTAIN_LOW = 0.30
NOUL_UNCERTAIN_HIGH = 0.70

REMOTE_BLOCKED_PRIVACY = frozenset({"local_only", "forbidden_external"})
TRAP_SMT_MARKERS = (
    "FloatingPoint",
    "BitVec",
    "fp.add",
    "fp.eq",
    "bvslt",
    "bvmul",
    "QF_FP",
    "QF_BV",
)
WHETHER_PREFIX = "whether_"
WHICH_PREFIX = "which_"


class KernelSpend(str, Enum):
    SPEND = "spend_kernel"
    SKIP = "skip_kernel"
    UNAVAILABLE = "typesafe_unavailable"


class SmtTriageAction(str, Enum):
    RUN_Z3 = "run_z3"
    SKIP_Z3 = "skip_z3"
    UNAVAILABLE = "typesafe_unavailable"


@dataclass(frozen=True)
class AdvisoryReceipt:
    schema: str = ADVISOR_SCHEMA
    authority_class: str = AUTHORITY_CLASS
    accepted_as_authority: bool = False
    provider: str = PROVIDER_NAME
    action: str = ""
    disposition: str = ""
    claim_status: str = ""
    choice: str = ""
    noul: float = 0.0
    score: float = 0.0
    confidence: float = 0.0
    question_id: str = ""
    reason_codes: tuple[str, ...] = ()
    privacy_blocked: bool = False
    trap_family: bool = False
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_as_authority", False)
        object.__setattr__(self, "authority_class", AUTHORITY_CLASS)
        object.__setattr__(self, "schema", ADVISOR_SCHEMA)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_class": AUTHORITY_CLASS,
            "accepted_as_authority": False,
            "provider": PROVIDER_NAME,
            "action": self.action,
            "disposition": self.disposition,
            "claim_status": self.claim_status,
            "choice": self.choice,
            "noul": round(float(self.noul), 4),
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "question_id": self.question_id,
            "reason_codes": list(self.reason_codes),
            "privacy_blocked": self.privacy_blocked,
            "trap_family": self.trap_family,
            "usage": dict(self.usage),
        }


def typesafe_permitted(
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    privacy = str(privacy_class or "").strip().casefold()
    if privacy in REMOTE_BLOCKED_PRIVACY:
        return False
    if not remote_disclosure_permitted:
        return False
    return typesafe_configured(environ=environ)


def is_trap_family(
    *,
    smtlib: str = "",
    case_id: str = "",
    complexity: str = "",
) -> bool:
    if str(complexity or "").casefold() == "trap":
        return True
    ident = str(case_id or "").casefold()
    if ident.startswith("float") or ident.startswith("bv") or "uninterpreted" in ident:
        return True
    blob = str(smtlib or "")
    return any(marker in blob for marker in TRAP_SMT_MARKERS)


def _usage() -> dict[str, int]:
    try:
        from ipfs_accelerate_py.typesafe_inference import get_last_typesafe_observation

        obs = get_last_typesafe_observation()
    except Exception:
        return {}
    return {
        "input_tokens": int(obs.get("input_tokens") or 0),
        "output_tokens": int(obs.get("output_tokens") or 0),
    }


def noul_uncertain(*values: float) -> bool:
    """True when some noul sits in ``[0.30, 0.70]`` and none fire above 0.70."""

    fired = any(float(value) > NOUL_UNCERTAIN_HIGH for value in values)
    in_band = any(
        NOUL_UNCERTAIN_LOW <= float(value) <= NOUL_UNCERTAIN_HIGH for value in values
    )
    return bool(in_band and not fired)


def _confidence_from_result(result: Any) -> float:
    values: list[float] = []
    for answer in getattr(result, "choices", {}) or {}:
        item = result.choices[answer]
        values.append(float(getattr(item, "confidence", 0.0) or 0.0))
    for answer in getattr(result, "scores", {}) or {}:
        item = result.scores[answer]
        values.append(float(getattr(item, "confidence", 0.0) or 0.0))
    return min(values) if values else 0.0


def advise_proof_draft(
    *,
    goal_id: str,
    declaration: str,
    draft_text: str,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> AdvisoryReceipt:
    """Decide whether a Leanstral draft is worth a kernel check."""

    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return AdvisoryReceipt(
            action=KernelSpend.UNAVAILABLE.value,
            reason_codes=("privacy_or_unconfigured",),
            privacy_blocked=str(privacy_class or "").casefold() in REMOTE_BLOCKED_PRIVACY
            or not remote_disclosure_permitted,
        )
    from ipfs_accelerate_py.leanstral_typesafe import LeanstralGoal, solve_with_typesafe

    try:
        verdict = solve_with_typesafe(
            LeanstralGoal(goal_id=goal_id, declaration=declaration),
            draft_text,
            timeout=timeout,
        )
    except Exception:
        return AdvisoryReceipt(
            action=KernelSpend.SPEND.value,
            reason_codes=("typesafe_error_fail_open_to_kernel",),
        )
    from ipfs_accelerate_py.leanstral_typesafe import compose_draft_quality

    quality = compose_draft_quality(
        is_proof_body=verdict.is_proof_body,
        well_formed=verdict.well_formed,
        uses_forbidden=verdict.uses_forbidden,
    )
    skip = (
        verdict.disposition in {"reject", "abstain"}
        and verdict.confidence >= HIGH_CONFIDENCE
        and verdict.parsed.kind in {"abstain", "malformed", "incomplete"}
        and quality < 0.35
    )
    action = KernelSpend.SKIP.value if skip else KernelSpend.SPEND.value
    reasons = []
    if skip:
        reasons.append("advisory_skip_kernel")
    else:
        reasons.append("advisory_spend_kernel")
    if verdict.confidence < LOW_CONFIDENCE:
        reasons.append("low_confidence")
    reasons.append("composed_draft_quality")
    return AdvisoryReceipt(
        action=action,
        disposition=verdict.disposition,
        claim_status=verdict.claim_status,
        confidence=verdict.confidence,
        noul=quality,
        score=verdict.candidate_quality,
        reason_codes=tuple(reasons),
        usage=_usage(),
    )


def triage_smt(
    *,
    english: str,
    smtlib: str,
    case_id: str = "",
    complexity: str = "",
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> AdvisoryReceipt:
    """Advise whether to spend z3. Trap families and low confidence always run z3."""

    trap = is_trap_family(smtlib=smtlib, case_id=case_id, complexity=complexity)
    if trap:
        return AdvisoryReceipt(
            action=SmtTriageAction.RUN_Z3.value,
            reason_codes=("trap_family_force_z3", "skip_typesafe_http"),
            trap_family=True,
        )
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return AdvisoryReceipt(
            action=SmtTriageAction.RUN_Z3.value,
            reason_codes=("privacy_or_unconfigured",),
            privacy_blocked=str(privacy_class or "").casefold() in REMOTE_BLOCKED_PRIVACY
            or not remote_disclosure_permitted,
            trap_family=False,
        )
    from ipfs_accelerate_py.typesafe_z3_benchmark import (
        SmtCase,
        run_typesafe,
    )

    case = SmtCase(
        case_id=case_id or "anonymous",
        english=english,
        smtlib=smtlib,
        expected="",
        complexity=complexity or "unknown",
    )
    timing = run_typesafe(case, timeout=timeout)
    status = str(timing.status or "unknown")
    conf = float(timing.confidence or 0.0)
    reasons = []
    if status in {"error", "unknown"} or conf < LOW_CONFIDENCE:
        reasons.append("low_confidence_or_unknown")
        action = SmtTriageAction.RUN_Z3.value
    elif conf >= HIGH_CONFIDENCE and status in {"sat", "unsat"}:
        reasons.append("high_confidence_skip_z3")
        action = SmtTriageAction.SKIP_Z3.value
    else:
        reasons.append("default_run_z3")
        action = SmtTriageAction.RUN_Z3.value
    if action == SmtTriageAction.SKIP_Z3.value:
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
                should_trust_skip,
                spot_check_due,
            )

            if not should_trust_skip(
                trap_family=False,
                confidence=conf,
                family=case_id,
                smtlib=smtlib,
                case_id=case_id,
            ):
                action = SmtTriageAction.RUN_Z3.value
                reasons.append("calibration_force_z3")
            elif spot_check_due():
                action = SmtTriageAction.RUN_Z3.value
                reasons.append("typesafe_spot_check_z3")
        except Exception:
            pass
    return AdvisoryReceipt(
        action=action,
        claim_status=status,
        confidence=conf,
        reason_codes=tuple(reasons),
        trap_family=False,
        usage=dict(timing.usage or {}),
    )


def _noul_answer(result: Any, name: str) -> float:
    nouls = getattr(result, "nouls", None) or {}
    item = nouls.get(name) if isinstance(nouls, Mapping) else None
    if item is None:
        return 0.0
    try:
        return float(getattr(item, "noul", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _choice_answer(result: Any, name: str) -> str:
    choices = getattr(result, "choices", None) or {}
    item = choices.get(name) if isinstance(choices, Mapping) else None
    if item is None:
        return ""
    return str(getattr(item, "choice", "") or "")


def _contrastive_choice(kind: str, allowed: Sequence[str]) -> Choice:
    criteria = {
        item: {
            "what": item,
            "not_for": "any other listed option",
            "examples": [item],
        }
        for item in allowed
    }
    return Choice(
        instructions={
            "question": f"Which allowlisted option applies for {kind.replace('_', ' ')}?",
            "focus": "Select exactly one listed alternative. Do not invent an option.",
        },
        criteria=criteria,
    )


def planning_atomic_questions(
    question_type: str,
    alternatives: Sequence[str] = (),
) -> dict[str, Any]:
    """Narrow parallel questions. Code composes the nomination."""

    kind = str(question_type or "").strip().casefold()
    allowed = tuple(str(item).strip() for item in alternatives if str(item).strip())
    questions: dict[str, Any] = {}
    if kind == "whether_replan_is_required":
        questions = {
            "mandatory_check_failed": Noul(
                instructions={
                    "question": "Did a mandatory check fail?",
                    "inspect": "`failure`",
                    "focus": "Failed tests, typecheck, or kernel — not warnings.",
                },
                criteria={
                    "true": {"what": "A required check failed"},
                    "false": {"what": "No required check failed", "not_for": "warnings only"},
                },
            ),
            "stale_evidence": Noul(
                instructions={
                    "question": "Is the failure explained by stale evidence?",
                    "inspect": "`failure`",
                },
            ),
            "suffix_still_matches_tree": Noul(
                instructions={
                    "question": "Does the remaining plan suffix still match the tree?",
                    "inspect": "`plan`",
                },
            ),
        }
    elif kind == "whether_patch_is_semantically_nonempty":
        questions = {
            "edits_tracked_files": Noul(
                instructions={
                    "question": "Does the patch edit tracked files?",
                    "inspect": "`patch`",
                },
            ),
            "changes_behavior": Noul(
                instructions={
                    "question": "Does the patch change runtime or proof behavior?",
                    "inspect": "`patch`",
                },
            ),
            "only_comments_or_whitespace": Noul(
                instructions={
                    "question": "Are the edits comments or whitespace only?",
                    "inspect": "`patch`",
                },
            ),
        }
    elif kind == "which_proof_obligation_applies":
        for item in allowed[:8]:
            key = "applies_" + "".join(ch if ch.isalnum() else "_" for ch in item)[:48]
            questions[key] = Noul(
                instructions={
                    "question": "Does this allowlisted obligation apply?",
                    "inspect": "`obligation_ids`",
                    "focus": item,
                },
            )
    if allowed:
        questions["answer"] = _contrastive_choice(kind, allowed)
    elif kind.startswith(WHETHER_PREFIX) and not questions:
        questions["answer"] = Noul(
            instructions={"question": f"Decide {kind.replace('_', ' ')} for this state."},
        )
    elif kind.startswith(WHETHER_PREFIX) and "answer" not in questions:
        questions["answer"] = Noul(
            instructions={"question": f"Decide {kind.replace('_', ' ')} for this state."},
        )
    elif kind.startswith(WHICH_PREFIX) and not allowed:
        return {}
    elif not questions and kind.startswith(WHICH_PREFIX):
        questions["answer"] = _contrastive_choice(kind, allowed)
    return questions


def compose_planning_nomination(
    question_type: str,
    result: Any,
    alternatives: Sequence[str] = (),
) -> tuple[str, tuple[str, ...], float]:
    """Combine atomic answers in code. Never invents an alternative."""

    kind = str(question_type or "").strip().casefold()
    allowed = tuple(str(item).strip() for item in alternatives if str(item).strip())
    reasons: list[str] = ["composed_in_code"]
    model_choice = _choice_answer(result, "answer")
    nominated = model_choice if (not allowed or model_choice in allowed) else ""
    if allowed and model_choice and model_choice not in allowed:
        reasons.append("choice_not_in_allowlist")

    if kind == "whether_replan_is_required":
        failed = _noul_answer(result, "mandatory_check_failed")
        stale = _noul_answer(result, "stale_evidence")
        matches = _noul_answer(result, "suffix_still_matches_tree")
        if stale > NOUL_UNCERTAIN_HIGH:
            for name in ("preserve", "not_required", "reuse"):
                if name in allowed:
                    nominated = name
                    reasons.append("stale_evidence_prefer_preserve")
                    break
        elif failed > NOUL_UNCERTAIN_HIGH and matches < NOUL_UNCERTAIN_LOW:
            for name in ("replan_suffix", "selected", "recompute"):
                if name in allowed:
                    nominated = name
                    reasons.append("failed_and_suffix_mismatch")
                    break
        elif noul_uncertain(stale, failed, matches):
            reasons.append("uncertain_band")
        if not allowed:
            answer_noul = _noul_answer(result, "answer")
            signal = max(failed, answer_noul)
            if signal > NOUL_UNCERTAIN_HIGH:
                nominated = "yes"
                reasons.append("noul_above_uncertain_band")
            elif signal < NOUL_UNCERTAIN_LOW and (failed or answer_noul):
                nominated = "no"
                reasons.append("noul_below_uncertain_band")
            elif noul_uncertain(signal):
                nominated = ""
                reasons.append("uncertain_band")
    elif kind == "whether_patch_is_semantically_nonempty":
        edits = _noul_answer(result, "edits_tracked_files")
        behavior = _noul_answer(result, "changes_behavior")
        comments = _noul_answer(result, "only_comments_or_whitespace")
        nonempty = (0.5 * edits + 0.5 * behavior) * (1.0 - comments)
        if noul_uncertain(nonempty, edits, behavior, comments):
            reasons.append("uncertain_band")
        elif allowed:
            if nonempty > NOUL_UNCERTAIN_HIGH:
                for name in ("nonempty", "selected", "yes"):
                    if name in allowed:
                        nominated = name
                        reasons.append("composite_nonempty")
                        break
            elif nonempty < NOUL_UNCERTAIN_LOW:
                for name in ("empty", "not_required", "no"):
                    if name in allowed:
                        nominated = name
                        reasons.append("composite_empty")
                        break
        else:
            if nonempty > NOUL_UNCERTAIN_HIGH:
                nominated = "yes"
                reasons.append("composite_nonempty_noul")
            elif nonempty < NOUL_UNCERTAIN_LOW:
                nominated = "no"
                reasons.append("composite_empty_noul")
            else:
                nominated = ""
                reasons.append("uncertain_band")
    elif kind == "which_proof_obligation_applies" and allowed:
        scored = []
        for item in allowed[:8]:
            key = "applies_" + "".join(ch if ch.isalnum() else "_" for ch in item)[:48]
            scored.append((_noul_answer(result, key), item))
        scored.sort(reverse=True)
        if scored and scored[0][0] > NOUL_UNCERTAIN_HIGH:
            nominated = scored[0][1]
            reasons.append("highest_applies_noul")
        elif scored and noul_uncertain(scored[0][0]):
            reasons.append("uncertain_band")
            if model_choice in allowed:
                nominated = model_choice
                reasons.append("allowlist_choice")
        elif model_choice in allowed:
            nominated = model_choice
            reasons.append("allowlist_choice")

    if nominated and allowed and nominated not in allowed:
        nominated = ""
        reasons.append("composed_choice_not_in_allowlist")
    confidence = _confidence_from_result(result)
    return nominated, tuple(reasons), confidence


def _mirror_advisory(receipt: AdvisoryReceipt) -> AdvisoryReceipt:
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(receipt.question_id or receipt.action or "typesafe-advisory")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="typesafe_advisory_receipt",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return receipt


def evaluate_closed_question(
    *,
    question_id: str,
    question_type: str,
    alternatives: Sequence[str] = (),
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> AdvisoryReceipt:
    """Answer one DecisionQuestion-shaped closed query. Allowlists only."""

    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return _mirror_advisory(
            AdvisoryReceipt(
                action="abstain",
                question_id=question_id,
                reason_codes=("privacy_or_unconfigured",),
                privacy_blocked=str(privacy_class or "").casefold() in REMOTE_BLOCKED_PRIVACY
                or not remote_disclosure_permitted,
            )
        )
    kind = str(question_type or "").strip().casefold()
    allowed = tuple(str(item).strip() for item in alternatives if str(item).strip())
    if kind.startswith(WHICH_PREFIX) and not allowed:
        return _mirror_advisory(
            AdvisoryReceipt(
                action="abstain",
                question_id=question_id,
                reason_codes=("empty_allowlist",),
            )
        )
    questions = planning_atomic_questions(kind, allowed)
    if not questions:
        return _mirror_advisory(
            AdvisoryReceipt(
                action="abstain",
                question_id=question_id,
                reason_codes=("unsupported_question_type",),
            )
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    redacted_state = {
        "question": {"id": question_id, "type": kind, "alternatives": list(allowed)},
        "facts": dict(state),
    }
    try:
        result = system_one(redacted_state, questions, timeout=timeout)
    except Exception:
        return _mirror_advisory(
            AdvisoryReceipt(
                action="abstain",
                question_id=question_id,
                reason_codes=("typesafe_error",),
            )
        )
    nominated, reasons, confidence = compose_planning_nomination(kind, result, allowed)
    if not nominated:
        return _mirror_advisory(
            AdvisoryReceipt(
                action="abstain",
                question_id=question_id,
                choice=_choice_answer(result, "answer"),
                confidence=confidence,
                reason_codes=reasons or ("choice_not_in_allowlist",),
                usage=_usage(),
            )
        )
    return _mirror_advisory(
        AdvisoryReceipt(
            action="answered",
            question_id=question_id,
            choice=nominated,
            noul=_noul_answer(result, "answer"),
            confidence=confidence,
            reason_codes=reasons,
            usage=_usage(),
        )
    )


PROOF_QUESTION_TYPES = frozenset(
    {
        "which_proof_obligation_applies",
        "whether_patch_is_semantically_nonempty",
    }
)
HUMAN_QUESTION_TYPE = "whether_human_choice_is_irreducible"


def residual_uncertainty_bp(confidence: float) -> int:
    """TypeSafe can never drive residual uncertainty to 0."""

    conf = max(0.0, min(1.0, float(confidence)))
    return max(1, min(10_000, int(round((1.0 - conf) * 10_000))))


def escalation_meta_action(
    question_type: str,
    *,
    confidence: float = 0.0,
    answered: bool = False,
) -> str:
    """Authoritative next action after an advisory TypeSafe nomination."""

    kind = str(question_type or "").strip().casefold()
    if kind == HUMAN_QUESTION_TYPE:
        return "REQUEST_HUMAN_DECISION"
    if kind in PROOF_QUESTION_TYPES:
        return "RUN_SMT_OR_PROVER"
    if not answered or float(confidence) < LOW_CONFIDENCE:
        return "CALL_REMOTE_STRONG_MODEL"
    if kind.startswith(WHICH_PREFIX):
        return "RUN_LOCAL_STATIC_ANALYSIS"
    return "CALL_REMOTE_STRONG_MODEL"


def _evidence_id(receipt: AdvisoryReceipt) -> str:
    import hashlib
    import json

    payload = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"))
    return "typesafe-advice-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class DecisionQuestionAdvice:
    receipt: AdvisoryReceipt
    nominated_answer: str
    next_action: str
    residual_uncertainty_bp: int
    evidence_id: str
    can_resolve: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "can_resolve", False)
        object.__setattr__(
            self,
            "residual_uncertainty_bp",
            max(1, int(self.residual_uncertainty_bp or 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "nominated_answer": self.nominated_answer,
            "next_action": self.next_action,
            "residual_uncertainty_bp": self.residual_uncertainty_bp,
            "evidence_id": self.evidence_id,
            "can_resolve": False,
            "accepted_as_authority": False,
        }


def advise_decision_question(
    *,
    question_id: str,
    question_type: str,
    alternatives: Sequence[str] = (),
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> DecisionQuestionAdvice:
    """Nominate an allowlisted answer. Never marks the question terminal."""

    receipt = evaluate_closed_question(
        question_id=question_id,
        question_type=question_type,
        alternatives=alternatives,
        state=state,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )
    answered = receipt.action == "answered" and bool(receipt.choice)
    nominated = receipt.choice if answered else ""
    if nominated and alternatives and nominated not in {str(item) for item in alternatives}:
        nominated = ""
        answered = False
    next_action = escalation_meta_action(
        question_type,
        confidence=receipt.confidence,
        answered=answered,
    )
    return DecisionQuestionAdvice(
        receipt=receipt,
        nominated_answer=nominated,
        next_action=next_action,
        residual_uncertainty_bp=residual_uncertainty_bp(receipt.confidence),
        evidence_id=_evidence_id(receipt),
    )


def score_synthesis_candidate(
    *,
    candidate_id: str,
    allowlisted_ids: Sequence[str],
    state: Mapping[str, Any],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 30.0,
) -> AdvisoryReceipt:
    """Rerank one already-admitted synthesis candidate. Never invents IDs."""

    allowed = tuple(str(item).strip() for item in allowlisted_ids if str(item).strip())
    if candidate_id not in allowed:
        return AdvisoryReceipt(
            action="abstain",
            question_id=candidate_id,
            reason_codes=("candidate_not_allowlisted",),
        )
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return AdvisoryReceipt(
            action="abstain",
            question_id=candidate_id,
            reason_codes=("privacy_or_unconfigured",),
            privacy_blocked=True,
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    questions = {
        "quality": Score(
            instructions="How close is this already-admitted candidate to kernel-ready?",
            criteria=["unusable", "partial", "kernel-ready"],
        ),
        "unique": Noul(
            instructions="Is this the unique admitted candidate for the required behavior?",
        ),
    }
    try:
        result = system_one(
            {"candidate_id": candidate_id, "allowlisted_ids": list(allowed), "state": dict(state)},
            questions,
            timeout=timeout,
        )
    except Exception:
        return AdvisoryReceipt(
            action="abstain",
            question_id=candidate_id,
            reason_codes=("typesafe_error",),
        )
    quality = float(result.scores["quality"].score) if "quality" in result.scores else 0.0
    unique = float(result.nouls["unique"].noul) if "unique" in result.nouls else 0.0
    return AdvisoryReceipt(
        action="scored",
        question_id=candidate_id,
        choice=candidate_id,
        score=quality,
        noul=unique,
        confidence=_confidence_from_result(result),
        reason_codes=("advisory_rerank_only",),
        usage=_usage(),
    )


def score_synthesis_candidates_fanout(
    items: Sequence[tuple[str, str]],
    *,
    allowlisted_ids: Sequence[str],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, AdvisoryReceipt]:
    """Score up to eight allowlisted candidates in one System One call."""

    allowed = {str(item).strip() for item in allowlisted_ids if str(item).strip()}
    rows = [
        (str(ident).strip(), str(summary or "")[:240])
        for ident, summary in items
        if str(ident).strip() in allowed
    ][:8]
    if not rows or not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return {}
    from ipfs_accelerate_py.typesafe_inference import system_one

    state = {
        "candidates": {ident: {"id": ident, "summary": summary} for ident, summary in rows}
    }
    questions: dict[str, Any] = {}
    for ident, _summary in rows:
        questions[f"quality_{ident}"] = Score(
            instructions={
                "question": f"How close is `candidates.{ident}.summary` to kernel-ready?",
                "inspect": f"`candidates.{ident}.summary`",
            },
            criteria=["unusable", "partial", "kernel-ready"],
        )
        questions[f"unique_{ident}"] = Noul(
            instructions={
                "question": f"Is `candidates.{ident}.id` the unique admitted candidate?",
                "inspect": f"`candidates.{ident}.id`",
            },
        )
    try:
        result = system_one(state, questions, timeout=timeout)
    except Exception:
        return {}
    scored: dict[str, AdvisoryReceipt] = {}
    scores = getattr(result, "scores", None) or {}
    nouls = getattr(result, "nouls", None) or {}
    for ident, _summary in rows:
        quality_item = scores.get(f"quality_{ident}")
        unique_item = nouls.get(f"unique_{ident}")
        quality = float(getattr(quality_item, "score", 0.0) or 0.0)
        unique = float(getattr(unique_item, "noul", 0.0) or 0.0)
        conf = float(getattr(quality_item, "confidence", 0.0) or 0.0)
        scored[ident] = AdvisoryReceipt(
            action="scored",
            question_id=ident,
            choice=ident,
            score=quality,
            noul=unique,
            confidence=conf,
            reason_codes=("advisory_rerank_only", "fanout"),
        )
    return scored


def maybe_verify_leanstral_draft(
    draft: Any,
    theorem: Any,
    *,
    typesafe_precheck: bool = False,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    **kernel_kwargs: Any,
) -> Any:
    """Opt-in TypeSafe precheck, then the existing kernel gate.

    Skipping the kernel never produces KERNEL_VERIFIED. The kernel path is
    unchanged when the precheck is off or TypeSafe is unavailable.
    """

    from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
        verify_leanstral_draft,
    )

    def _mirror_typesafe(result: Any) -> Any:
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_work_record,
            )

            artifact = getattr(result, "model_artifact", None)
            admission = getattr(result, "admission", None)
            record_ref = str(
                getattr(artifact, "artifact_id", "")
                or getattr(admission, "theorem_id", "")
                or getattr(result, "status", "")
                or "typesafe-leanstral"
            )
            mirror_work_record(
                catalog_kind="proof_cache",
                record_kind="typesafe_leanstral_draft_gate",
                record_ref=record_ref,
                subject_kind="record_cid",
                subject_ref=record_ref,
            )
        except Exception:
            pass
        return result

    if not typesafe_precheck:
        return _mirror_typesafe(verify_leanstral_draft(draft, theorem, **kernel_kwargs))
    model = draft if not isinstance(draft, Mapping) else draft
    text = str(getattr(model, "draft_text", None) or (model.get("draft_text") if isinstance(model, Mapping) else "") or "")
    declaration = ""
    goal_id = ""
    if not isinstance(theorem, Mapping):
        declaration = str(getattr(theorem, "expected_statement", "") or getattr(theorem, "theorem_id", "") or "")
        goal_id = str(getattr(theorem, "obligation_id", "") or getattr(theorem, "theorem_id", "") or "")
    else:
        declaration = str(theorem.get("expected_statement") or theorem.get("theorem_id") or "")
        goal_id = str(theorem.get("obligation_id") or theorem.get("theorem_id") or "")
    try:
        advice = advise_proof_draft(
            goal_id=goal_id or "unknown",
            declaration=declaration or "unknown",
            draft_text=text,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
        )
    except Exception:
        return _mirror_typesafe(verify_leanstral_draft(draft, theorem, **kernel_kwargs))
    if advice.action == KernelSpend.SKIP.value:
        raise TypesafeKernelSkip(advice)
    result = verify_leanstral_draft(draft, theorem, **kernel_kwargs)
    if advice.action != KernelSpend.UNAVAILABLE.value and advice.claim_status:
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
                record_sample,
            )

            kernel = getattr(result, "kernel_verification", None)
            accepted = bool(getattr(kernel, "accepted", False))
            record_sample(
                family=str(goal_id or "proof")[:64],
                predicted=str(advice.claim_status),
                actual="unsat" if accepted else "sat",
                confidence=float(advice.confidence or 0.0),
            )
        except Exception:
            pass
    return _mirror_typesafe(result)


class TypesafeKernelSkip(RuntimeError):
    """Advisory skip of kernel spend. Not a verified rejection."""

    def __init__(self, receipt: AdvisoryReceipt) -> None:
        super().__init__("typesafe advisory skip_kernel")
        self.receipt = receipt


__all__ = [
    "ADVISOR_SCHEMA",
    "AUTHORITY_CLASS",
    "AdvisoryReceipt",
    "DecisionQuestionAdvice",
    "HIGH_CONFIDENCE",
    "HUMAN_QUESTION_TYPE",
    "KernelSpend",
    "LOW_CONFIDENCE",
    "NOUL_UNCERTAIN_HIGH",
    "NOUL_UNCERTAIN_LOW",
    "PROOF_QUESTION_TYPES",
    "SmtTriageAction",
    "TypesafeKernelSkip",
    "advise_decision_question",
    "advise_proof_draft",
    "compose_planning_nomination",
    "escalation_meta_action",
    "evaluate_closed_question",
    "planning_atomic_questions",
    "is_trap_family",
    "maybe_verify_leanstral_draft",
    "noul_uncertain",
    "residual_uncertainty_bp",
    "score_synthesis_candidate",
    "score_synthesis_candidates_fanout",
    "triage_smt",
    "typesafe_permitted",
]
