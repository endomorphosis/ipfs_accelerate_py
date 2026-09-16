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
    skip = (
        verdict.disposition in {"reject", "abstain"}
        and verdict.confidence >= HIGH_CONFIDENCE
        and verdict.parsed.kind in {"abstain", "malformed", "incomplete"}
    )
    action = KernelSpend.SKIP.value if skip else KernelSpend.SPEND.value
    reasons = []
    if skip:
        reasons.append("advisory_skip_kernel")
    else:
        reasons.append("advisory_spend_kernel")
    if verdict.confidence < LOW_CONFIDENCE:
        reasons.append("low_confidence")
    return AdvisoryReceipt(
        action=action,
        disposition=verdict.disposition,
        claim_status=verdict.claim_status,
        confidence=verdict.confidence,
        noul=verdict.is_proof_body,
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
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return AdvisoryReceipt(
            action=SmtTriageAction.RUN_Z3.value,
            reason_codes=("privacy_or_unconfigured",),
            privacy_blocked=str(privacy_class or "").casefold() in REMOTE_BLOCKED_PRIVACY
            or not remote_disclosure_permitted,
            trap_family=trap,
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
    if trap:
        reasons.append("trap_family_force_z3")
        action = SmtTriageAction.RUN_Z3.value
    elif status in {"error", "unknown"} or conf < LOW_CONFIDENCE:
        reasons.append("low_confidence_or_unknown")
        action = SmtTriageAction.RUN_Z3.value
    elif conf >= HIGH_CONFIDENCE and status in {"sat", "unsat"}:
        reasons.append("high_confidence_skip_z3")
        action = SmtTriageAction.SKIP_Z3.value
    else:
        reasons.append("default_run_z3")
        action = SmtTriageAction.RUN_Z3.value
    return AdvisoryReceipt(
        action=action,
        claim_status=status,
        confidence=conf,
        reason_codes=tuple(reasons),
        trap_family=trap,
        usage=dict(timing.usage or {}),
    )


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
        return AdvisoryReceipt(
            action="abstain",
            question_id=question_id,
            reason_codes=("privacy_or_unconfigured",),
            privacy_blocked=str(privacy_class or "").casefold() in REMOTE_BLOCKED_PRIVACY
            or not remote_disclosure_permitted,
        )
    kind = str(question_type or "").strip().casefold()
    allowed = tuple(str(item).strip() for item in alternatives if str(item).strip())
    if kind.startswith(WHICH_PREFIX) and not allowed:
        return AdvisoryReceipt(
            action="abstain",
            question_id=question_id,
            reason_codes=("empty_allowlist",),
        )
    if kind.startswith(WHETHER_PREFIX):
        questions: dict[str, Any] = {
            "answer": Noul(instructions=f"Decide {kind.replace('_', ' ')} for this state."),
        }
    elif kind.startswith(WHICH_PREFIX):
        questions = {
            "answer": Choice(
                instructions=f"Select one allowlisted option for {kind.replace('_', ' ')}.",
                criteria={item: None for item in allowed},
            ),
        }
    else:
        return AdvisoryReceipt(
            action="abstain",
            question_id=question_id,
            reason_codes=("unsupported_question_type",),
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    redacted_state = {
        "question_id": question_id,
        "question_type": kind,
        "alternatives": list(allowed),
        "state": dict(state),
    }
    try:
        result = system_one(redacted_state, questions, timeout=timeout)
    except Exception:
        return AdvisoryReceipt(
            action="abstain",
            question_id=question_id,
            reason_codes=("typesafe_error",),
        )
    if kind.startswith(WHETHER_PREFIX):
        noul = float(result.nouls["answer"].noul) if "answer" in result.nouls else 0.0
        choice = "yes" if noul >= 0.5 else "no"
        return AdvisoryReceipt(
            action="answered",
            question_id=question_id,
            choice=choice,
            noul=noul,
            confidence=_confidence_from_result(result),
            reason_codes=("noul_threshold_0_5",),
            usage=_usage(),
        )
    chosen = str(result.choices["answer"].choice) if "answer" in result.choices else ""
    if chosen not in allowed:
        return AdvisoryReceipt(
            action="abstain",
            question_id=question_id,
            choice=chosen,
            confidence=_confidence_from_result(result),
            reason_codes=("choice_not_in_allowlist",),
            usage=_usage(),
        )
    return AdvisoryReceipt(
        action="answered",
        question_id=question_id,
        choice=chosen,
        confidence=_confidence_from_result(result),
        reason_codes=("allowlist_choice",),
        usage=_usage(),
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

    if not typesafe_precheck:
        return verify_leanstral_draft(draft, theorem, **kernel_kwargs)
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
    advice = advise_proof_draft(
        goal_id=goal_id or "unknown",
        declaration=declaration or "unknown",
        draft_text=text,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    )
    if advice.action == KernelSpend.SKIP.value:
        raise TypesafeKernelSkip(advice)
    return verify_leanstral_draft(draft, theorem, **kernel_kwargs)


class TypesafeKernelSkip(RuntimeError):
    """Advisory skip of kernel spend. Not a verified rejection."""

    def __init__(self, receipt: AdvisoryReceipt) -> None:
        super().__init__("typesafe advisory skip_kernel")
        self.receipt = receipt


__all__ = [
    "ADVISOR_SCHEMA",
    "AUTHORITY_CLASS",
    "AdvisoryReceipt",
    "HIGH_CONFIDENCE",
    "KernelSpend",
    "LOW_CONFIDENCE",
    "SmtTriageAction",
    "TypesafeKernelSkip",
    "advise_proof_draft",
    "evaluate_closed_question",
    "is_trap_family",
    "maybe_verify_leanstral_draft",
    "score_synthesis_candidate",
    "triage_smt",
    "typesafe_permitted",
]
