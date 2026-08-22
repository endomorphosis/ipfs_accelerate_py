"""Offline distillation of local classification and ranking experts.

Fitting is forbidden until an admitted TrainingCorpusAdmission binds rights,
privacy, splits, tokenizer/compiler, labels, and environment.  The selected
form is the smallest class that is reliable on current held-out and
adversarial evaluation; a larger class needs a routing-changing quality delta.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
)
from .expert_specs import (
    MIN_ROUTING_CHANGING_DELTA_PPM,
    ExpertClass,
    expert_class_rank,
    parse_expert_class,
)
from .local_experts import (
    MAX_LOCAL_CHECKPOINTS,
    MAX_LOCAL_EXAMPLES,
    MAX_LOCAL_GPU_SECONDS,
    MAX_LOCAL_STEPS,
    MAX_LOCAL_WALL_SECONDS,
    REASON_TRAINING_UNAVAILABLE,
    ExpertEvaluation,
    ExpertEvaluationCase,
    LocalClassificationExpert,
    LocalRankingExpert,
    admit_local_expert_class,
    require_training_admission,
)
from .residual_ir import ResidualTaskInput
from .rights import TrainingCorpusAdmission
from .splits import SplitPartition
from .task_families import (
    SEMANTIC_KIND_CLASSIFICATION,
    SEMANTIC_KIND_RANKING,
    family_spec_for,
)

DISTILLATION_BUDGET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-distillation-budget@1"
)
DISTILLATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-distillation-result@1"
)
REASON_HELD_OUT_QUALITY: Final = "held_out_quality_not_admitted"
REASON_NO_CRITICAL_FALSE_ACCEPT: Final = "critical_false_accept"
REASON_GPU_FORBIDDEN: Final = "local_experts_gpu_seconds_must_be_zero"


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class DistillationBudget:
    """Hard caps for one local-expert distillation run."""

    examples: int = MAX_LOCAL_EXAMPLES
    steps: int = MAX_LOCAL_STEPS
    wall_seconds: int = MAX_LOCAL_WALL_SECONDS
    gpu_seconds: int = MAX_LOCAL_GPU_SECONDS
    checkpoints: int = MAX_LOCAL_CHECKPOINTS
    schema: str = DISTILLATION_BUDGET_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "budget_id",
            "examples",
            "steps",
            "wall_seconds",
            "gpu_seconds",
            "checkpoints",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != DISTILLATION_BUDGET_SCHEMA:
            raise ResidualIntelligenceError("unsupported distillation budget schema")
        object.__setattr__(
            self,
            "examples",
            bounded_int(self.examples, "examples", minimum=1, maximum=MAX_LOCAL_EXAMPLES),
        )
        object.__setattr__(
            self,
            "steps",
            bounded_int(self.steps, "steps", minimum=1, maximum=MAX_LOCAL_STEPS),
        )
        object.__setattr__(
            self,
            "wall_seconds",
            bounded_int(
                self.wall_seconds, "wall_seconds", minimum=0, maximum=MAX_LOCAL_WALL_SECONDS
            ),
        )
        object.__setattr__(
            self,
            "gpu_seconds",
            bounded_int(
                self.gpu_seconds, "gpu_seconds", minimum=0, maximum=MAX_LOCAL_GPU_SECONDS
            ),
        )
        if self.gpu_seconds != 0:
            raise ResidualIntelligenceError(REASON_GPU_FORBIDDEN)
        object.__setattr__(
            self,
            "checkpoints",
            bounded_int(
                self.checkpoints, "checkpoints", minimum=1, maximum=MAX_LOCAL_CHECKPOINTS
            ),
        )

    @property
    def budget_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "examples": self.examples,
            "steps": self.steps,
            "wall_seconds": self.wall_seconds,
            "gpu_seconds": 0,
            "checkpoints": self.checkpoints,
        }
        if include_id:
            result["budget_id"] = self.budget_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationBudget:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"budget_id"},
            noun="distillation budget",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            examples=payload.get("examples"),
            steps=payload.get("steps"),
            wall_seconds=payload.get("wall_seconds"),
            gpu_seconds=payload.get("gpu_seconds"),
            checkpoints=payload.get("checkpoints"),
        )
        claimed = str(payload.get("budget_id") or "")
        if claimed and claimed != result.budget_id:
            raise ResidualIntelligenceError("distillation budget identity mismatch")
        return result


DEFAULT_DISTILLATION_BUDGET: Final[DistillationBudget] = DistillationBudget()


@dataclass(frozen=True)
class DistillationResult:
    """Candidate-only distillation receipt for one family and calibration group."""

    task_family: ResidualTaskFamily
    selected_class: ExpertClass
    evaluation: ExpertEvaluation
    quality_delta_ppm: int
    routing_changing: bool
    evidence_current: bool
    training_unavailable: bool
    admission_id: str
    classification_expert: LocalClassificationExpert | None = None
    ranking_expert: LocalRankingExpert | None = None
    candidate_only: bool = True
    schema: str = DISTILLATION_RESULT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "result_id",
            "task_family",
            "selected_class",
            "evaluation",
            "quality_delta_ppm",
            "routing_changing",
            "evidence_current",
            "training_unavailable",
            "admission_id",
            "classification_expert",
            "ranking_expert",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != DISTILLATION_RESULT_SCHEMA:
            raise ResidualIntelligenceError("unsupported distillation result schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "selected_class", parse_expert_class(self.selected_class))
        if not isinstance(self.evaluation, ExpertEvaluation):
            raise ResidualIntelligenceError("distillation result requires ExpertEvaluation")
        object.__setattr__(
            self,
            "quality_delta_ppm",
            bounded_int(
                self.quality_delta_ppm,
                "quality_delta_ppm",
                minimum=0,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self, "routing_changing", _require_bool(self.routing_changing, "routing_changing")
        )
        object.__setattr__(
            self, "evidence_current", _require_bool(self.evidence_current, "evidence_current")
        )
        object.__setattr__(
            self,
            "training_unavailable",
            _require_bool(self.training_unavailable, "training_unavailable"),
        )
        object.__setattr__(
            self,
            "admission_id",
            ""
            if self.admission_id in (None, "")
            else required_text(self.admission_id, "admission_id"),
        )
        if self.classification_expert is not None and not isinstance(
            self.classification_expert, LocalClassificationExpert
        ):
            raise ResidualIntelligenceError(
                "classification_expert must be LocalClassificationExpert"
            )
        if self.ranking_expert is not None and not isinstance(
            self.ranking_expert, LocalRankingExpert
        ):
            raise ResidualIntelligenceError("ranking_expert must be LocalRankingExpert")
        if (self.classification_expert is None) == (self.ranking_expert is None):
            raise ResidualIntelligenceError(
                "distillation result must bind exactly one local expert"
            )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("distillation results must remain candidate_only=true")
        if self.training_unavailable:
            raise ResidualIntelligenceError(REASON_TRAINING_UNAVAILABLE)
        if self.classification_expert is not None:
            if self.classification_expert.expert_class is not self.selected_class:
                raise ResidualIntelligenceError("selected classification class mismatch")
            if self.classification_expert.candidate_only is not True:
                raise ResidualIntelligenceError(
                    "distillation results must remain candidate_only=true"
                )
        if self.ranking_expert is not None:
            if self.ranking_expert.expert_class is not self.selected_class:
                raise ResidualIntelligenceError("selected ranking class mismatch")

    @property
    def result_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def expert(self) -> LocalClassificationExpert | LocalRankingExpert:
        if self.classification_expert is not None:
            return self.classification_expert
        assert self.ranking_expert is not None
        return self.ranking_expert

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "selected_class": self.selected_class.value,
            "evaluation": self.evaluation.to_dict(),
            "quality_delta_ppm": self.quality_delta_ppm,
            "routing_changing": self.routing_changing,
            "evidence_current": self.evidence_current,
            "training_unavailable": False,
            "admission_id": self.admission_id,
            "classification_expert": (
                None
                if self.classification_expert is None
                else self.classification_expert.to_dict()
            ),
            "ranking_expert": (
                None if self.ranking_expert is None else self.ranking_expert.to_dict()
            ),
            "candidate_only": True,
        }
        if include_id:
            result["result_id"] = self.result_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistillationResult:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"result_id", "classification_expert", "ranking_expert"},
            noun="distillation result",
        )
        classification_payload = payload.get("classification_expert")
        ranking_payload = payload.get("ranking_expert")
        result = cls(
            schema=str(payload.get("schema") or ""),
            task_family=ResidualTaskFamily(str(payload.get("task_family") or "")),
            selected_class=parse_expert_class(str(payload.get("selected_class") or "")),
            evaluation=ExpertEvaluation.from_dict(payload.get("evaluation") or {}),
            quality_delta_ppm=payload.get("quality_delta_ppm"),
            routing_changing=payload.get("routing_changing"),
            evidence_current=payload.get("evidence_current"),
            training_unavailable=payload.get("training_unavailable"),
            admission_id=str(payload.get("admission_id") or ""),
            classification_expert=(
                None
                if classification_payload in (None, {})
                else LocalClassificationExpert.from_dict(classification_payload)
            ),
            ranking_expert=(
                None
                if ranking_payload in (None, {})
                else LocalRankingExpert.from_dict(ranking_payload)
            ),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("result_id") or "")
        if claimed and claimed != result.result_id:
            raise ResidualIntelligenceError("distillation result identity mismatch")
        return result


def _split_cases(
    cases: Sequence[ExpertEvaluationCase],
) -> tuple[tuple[ExpertEvaluationCase, ...], tuple[ExpertEvaluationCase, ...]]:
    train: list[ExpertEvaluationCase] = []
    held: list[ExpertEvaluationCase] = []
    for case in cases:
        if not isinstance(case, ExpertEvaluationCase):
            raise ResidualIntelligenceError("distillation cases must be ExpertEvaluationCase")
        if not isinstance(case.task_input, ResidualTaskInput):
            raise ResidualIntelligenceError("distillation cases require ResidualTaskInput")
        if case.partition in {SplitPartition.TRAIN, SplitPartition.DEVELOPMENT}:
            train.append(case)
        else:
            held.append(case)
    if not held:
        raise ResidualIntelligenceError("distillation requires current held-out evaluation")
    if not train:
        raise ResidualIntelligenceError("distillation requires a training partition")
    return tuple(train), tuple(held)


def _quality_score(evaluation: ExpertEvaluation) -> int:
    if not evaluation.quality_admitted:
        return 0
    produced = (
        evaluation.example_count
        - evaluation.cascade_abstain_count
        - evaluation.reject_form_count
    )
    return (produced * 1_000_000) // evaluation.example_count


def select_smallest_reliable_form(
    evaluations: Mapping[ExpertClass, ExpertEvaluation],
    *,
    smallest: ExpertClass,
    requested: ExpertClass,
    admission: TrainingCorpusAdmission,
    risk: RiskClass,
    family: ResidualTaskFamily,
) -> tuple[ExpertClass, ExpertEvaluation, int, bool]:
    if requested not in evaluations or smallest not in evaluations:
        raise ResidualIntelligenceError("smallest-reliable-form requires evaluations for A and the request")
    baseline = evaluations[smallest]
    chosen = smallest
    chosen_eval = baseline
    delta = 0
    routing_changing = False
    current = smallest
    while expert_class_rank(current) < expert_class_rank(requested):
        nxt_index = expert_class_rank(current) + 1
        nxt = tuple(ExpertClass)[nxt_index]
        if nxt not in evaluations:
            break
        candidate = evaluations[nxt]
        gain = max(0, _quality_score(candidate) - _quality_score(chosen_eval))
        if not candidate.quality_admitted:
            break
        if gain < MIN_ROUTING_CHANGING_DELTA_PPM:
            break
        admit_local_expert_class(
            family,
            nxt,
            risk=risk,
            quality_delta_ppm=gain,
            routing_changing=True,
            evidence_current=True,
            compared_class=current,
            admission=admission,
        )
        chosen = nxt
        chosen_eval = candidate
        delta = gain
        routing_changing = True
        current = nxt
    return chosen, chosen_eval, delta, routing_changing


def _ensure_budget(
    cases: Sequence[ExpertEvaluationCase],
    budget: DistillationBudget,
    *,
    steps: int,
    wall_seconds: int,
    gpu_seconds: int,
) -> None:
    if len(cases) > budget.examples:
        raise ResidualIntelligenceError(f"distillation exceeds {budget.examples} examples")
    bounded_int(steps, "steps", minimum=1, maximum=budget.steps)
    bounded_int(wall_seconds, "wall_seconds", minimum=0, maximum=budget.wall_seconds)
    gpu = bounded_int(gpu_seconds, "gpu_seconds", minimum=0, maximum=budget.gpu_seconds)
    if gpu != 0:
        raise ResidualIntelligenceError(REASON_GPU_FORBIDDEN)


def distill_classification_expert(
    seed: LocalClassificationExpert,
    *,
    admission: TrainingCorpusAdmission,
    cases: Sequence[ExpertEvaluationCase],
    budget: DistillationBudget | None = None,
    requested_class: ExpertClass | str | None = None,
    steps: int = 1,
    wall_seconds: int = 0,
    gpu_seconds: int = 0,
) -> DistillationResult:
    """Fit exact then linear candidates and keep the smallest reliable form."""

    require_training_admission(admission)
    limits = budget or DEFAULT_DISTILLATION_BUDGET
    _ensure_budget(
        cases, limits, steps=steps, wall_seconds=wall_seconds, gpu_seconds=gpu_seconds
    )
    family = seed.task_family
    spec = family_spec_for(family)
    if spec.semantic_kind != SEMANTIC_KIND_CLASSIFICATION:
        raise ResidualIntelligenceError("distill_classification_expert requires a classification family")
    train, held = _split_cases(cases)
    fitted = seed.fit(
        admission=admission,
        cases=train,
        steps=steps,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
    )
    smallest = parse_expert_class(spec.smallest_expert_class)
    wanted = parse_expert_class(requested_class or fitted.expert_class)
    class_a = LocalClassificationExpert(
        task_family=fitted.task_family,
        expert_class=ExpertClass.A,
        calibration_group=fitted.calibration_group,
        feature_names=fitted.feature_names,
        lookup=fitted.lookup,
        rules=(),
        class_labels=fitted.class_labels,
        linear_form=fitted.linear_form,
        coefficients=(),
        intercepts=(),
        linear_threshold_ppm=fitted.linear_threshold_ppm,
        selective_policy=fitted.selective_policy,
        ood_reference=fitted.ood_reference,
        ood_boundary=fitted.ood_boundary,
        policy_admits_ood=fitted.policy_admits_ood,
        ood_schema=fitted.ood_schema,
        ood_operation=fitted.ood_operation,
        ood_effects=fitted.ood_effects,
        ood_capabilities=fitted.ood_capabilities,
        ood_context_fields=fitted.ood_context_fields,
        fitted=True,
        admission_id=admission.admission_id,
        checkpoint_count=fitted.checkpoint_count,
    )
    evaluations = {ExpertClass.A: class_a.evaluate(held)}
    class_b = None
    if ExpertClass.B in {parse_expert_class(item) for item in spec.eligible_expert_classes}:
        class_b = LocalClassificationExpert(
            task_family=fitted.task_family,
            expert_class=ExpertClass.B,
            calibration_group=fitted.calibration_group,
            feature_names=fitted.feature_names,
            lookup=fitted.lookup,
            rules=fitted.rules,
            class_labels=fitted.class_labels,
            linear_form=fitted.linear_form,
            coefficients=(),
            intercepts=(),
            linear_threshold_ppm=fitted.linear_threshold_ppm,
            selective_policy=fitted.selective_policy,
            ood_reference=fitted.ood_reference,
            ood_boundary=fitted.ood_boundary,
            policy_admits_ood=fitted.policy_admits_ood,
            ood_schema=fitted.ood_schema,
            ood_operation=fitted.ood_operation,
            ood_effects=fitted.ood_effects,
            ood_capabilities=fitted.ood_capabilities,
            ood_context_fields=fitted.ood_context_fields,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=fitted.checkpoint_count,
        )
        evaluations[ExpertClass.B] = class_b.evaluate(held)
    class_c = None
    if ExpertClass.C in {parse_expert_class(item) for item in spec.eligible_expert_classes}:
        class_c = LocalClassificationExpert(
            task_family=fitted.task_family,
            expert_class=ExpertClass.C,
            calibration_group=fitted.calibration_group,
            feature_names=fitted.feature_names,
            lookup=fitted.lookup,
            rules=fitted.rules,
            class_labels=fitted.class_labels,
            linear_form=fitted.linear_form,
            coefficients=fitted.coefficients,
            intercepts=fitted.intercepts,
            linear_threshold_ppm=fitted.linear_threshold_ppm,
            selective_policy=fitted.selective_policy,
            ood_reference=fitted.ood_reference,
            ood_boundary=fitted.ood_boundary,
            policy_admits_ood=fitted.policy_admits_ood,
            ood_schema=fitted.ood_schema,
            ood_operation=fitted.ood_operation,
            ood_effects=fitted.ood_effects,
            ood_capabilities=fitted.ood_capabilities,
            ood_context_fields=fitted.ood_context_fields,
            fitted=True,
            admission_id=admission.admission_id,
            checkpoint_count=fitted.checkpoint_count,
        )
        evaluations[ExpertClass.C] = class_c.evaluate(held)
    selected, evaluation, delta, routing = select_smallest_reliable_form(
        evaluations,
        smallest=smallest,
        requested=wanted,
        admission=admission,
        risk=seed.calibration_group.risk,
        family=family,
    )
    expert = {ExpertClass.A: class_a, ExpertClass.B: class_b, ExpertClass.C: class_c}[selected]
    if expert is None:
        raise ResidualIntelligenceError("selected classification form is unavailable")
    if not evaluation.quality_admitted and selected is not smallest:
        raise ResidualIntelligenceError(REASON_HELD_OUT_QUALITY)
    if evaluation.critical_false_accept_count:
        raise ResidualIntelligenceError(REASON_NO_CRITICAL_FALSE_ACCEPT)
    return DistillationResult(
        task_family=family,
        selected_class=selected,
        evaluation=evaluation,
        quality_delta_ppm=delta,
        routing_changing=routing,
        evidence_current=True,
        training_unavailable=False,
        admission_id=admission.admission_id,
        classification_expert=expert,
        candidate_only=True,
    )


def distill_ranking_expert(
    seed: LocalRankingExpert,
    *,
    admission: TrainingCorpusAdmission,
    cases: Sequence[ExpertEvaluationCase],
    budget: DistillationBudget | None = None,
    requested_class: ExpertClass | str | None = None,
    steps: int = 1,
    wall_seconds: int = 0,
    gpu_seconds: int = 0,
) -> DistillationResult:
    """Fit exact, deterministic ranking, then a small ranker; keep the smallest reliable form."""

    require_training_admission(admission)
    limits = budget or DEFAULT_DISTILLATION_BUDGET
    _ensure_budget(
        cases, limits, steps=steps, wall_seconds=wall_seconds, gpu_seconds=gpu_seconds
    )
    family = seed.task_family
    spec = family_spec_for(family)
    if spec.semantic_kind != SEMANTIC_KIND_RANKING:
        raise ResidualIntelligenceError("distill_ranking_expert requires a ranking family")
    train, held = _split_cases(cases)
    fitted = seed.fit(
        admission=admission,
        cases=train,
        steps=steps,
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
    )
    smallest = parse_expert_class(spec.smallest_expert_class)
    wanted = parse_expert_class(requested_class or fitted.expert_class)
    class_a = LocalRankingExpert(
        task_family=fitted.task_family,
        expert_class=ExpertClass.A,
        calibration_group=fitted.calibration_group,
        feature_names=fitted.feature_names,
        lookup=fitted.lookup,
        ranking_weights=(),
        selective_policy=fitted.selective_policy,
        ood_reference=fitted.ood_reference,
        ood_boundary=fitted.ood_boundary,
        policy_admits_ood=fitted.policy_admits_ood,
        ood_schema=fitted.ood_schema,
        ood_operation=fitted.ood_operation,
        ood_effects=fitted.ood_effects,
        ood_capabilities=fitted.ood_capabilities,
        ood_context_fields=fitted.ood_context_fields,
        fitted=True,
        admission_id=admission.admission_id,
        checkpoint_count=fitted.checkpoint_count,
    )
    evaluations = {ExpertClass.A: class_a.evaluate(held)}
    class_b = LocalRankingExpert(
        task_family=fitted.task_family,
        expert_class=ExpertClass.B,
        calibration_group=fitted.calibration_group,
        feature_names=fitted.feature_names,
        lookup=fitted.lookup,
        ranking_weights=fitted.ranking_weights,
        selective_policy=fitted.selective_policy,
        ood_reference=fitted.ood_reference,
        ood_boundary=fitted.ood_boundary,
        policy_admits_ood=fitted.policy_admits_ood,
        ood_schema=fitted.ood_schema,
        ood_operation=fitted.ood_operation,
        ood_effects=fitted.ood_effects,
        ood_capabilities=fitted.ood_capabilities,
        ood_context_fields=fitted.ood_context_fields,
        fitted=True,
        admission_id=admission.admission_id,
        checkpoint_count=fitted.checkpoint_count,
    )
    evaluations[ExpertClass.B] = class_b.evaluate(held)
    class_d = LocalRankingExpert(
        task_family=fitted.task_family,
        expert_class=ExpertClass.D,
        calibration_group=fitted.calibration_group,
        feature_names=fitted.feature_names,
        lookup=fitted.lookup,
        ranking_weights=fitted.ranking_weights,
        small_ranker=fitted.small_ranker,
        selective_policy=fitted.selective_policy,
        ood_reference=fitted.ood_reference,
        ood_boundary=fitted.ood_boundary,
        policy_admits_ood=fitted.policy_admits_ood,
        ood_schema=fitted.ood_schema,
        ood_operation=fitted.ood_operation,
        ood_effects=fitted.ood_effects,
        ood_capabilities=fitted.ood_capabilities,
        ood_context_fields=fitted.ood_context_fields,
        fitted=True,
        admission_id=admission.admission_id,
        checkpoint_count=fitted.checkpoint_count,
    )
    # Class C is linear_logistic; ranking families may skip it only with evidence.
    class_c = LocalRankingExpert(
        task_family=fitted.task_family,
        expert_class=ExpertClass.C,
        calibration_group=fitted.calibration_group,
        feature_names=fitted.feature_names,
        lookup=fitted.lookup,
        ranking_weights=fitted.ranking_weights,
        selective_policy=fitted.selective_policy,
        ood_reference=fitted.ood_reference,
        ood_boundary=fitted.ood_boundary,
        policy_admits_ood=fitted.policy_admits_ood,
        ood_schema=fitted.ood_schema,
        ood_operation=fitted.ood_operation,
        ood_effects=fitted.ood_effects,
        ood_capabilities=fitted.ood_capabilities,
        ood_context_fields=fitted.ood_context_fields,
        fitted=True,
        admission_id=admission.admission_id,
        checkpoint_count=fitted.checkpoint_count,
    )
    evaluations[ExpertClass.C] = class_c.evaluate(held)
    evaluations[ExpertClass.D] = class_d.evaluate(held)
    selected, evaluation, delta, routing = select_smallest_reliable_form(
        evaluations,
        smallest=smallest,
        requested=wanted,
        admission=admission,
        risk=seed.calibration_group.risk,
        family=family,
    )
    expert = {
        ExpertClass.A: class_a,
        ExpertClass.B: class_b,
        ExpertClass.C: class_c,
        ExpertClass.D: class_d,
    }[selected]
    if not evaluation.quality_admitted and selected is not smallest:
        raise ResidualIntelligenceError(REASON_HELD_OUT_QUALITY)
    if evaluation.critical_false_accept_count:
        raise ResidualIntelligenceError(REASON_NO_CRITICAL_FALSE_ACCEPT)
    return DistillationResult(
        task_family=family,
        selected_class=selected,
        evaluation=evaluation,
        quality_delta_ppm=delta,
        routing_changing=routing,
        evidence_current=True,
        training_unavailable=False,
        admission_id=admission.admission_id,
        ranking_expert=expert,
        candidate_only=True,
    )


__all__ = (
    "DEFAULT_DISTILLATION_BUDGET",
    "DISTILLATION_BUDGET_SCHEMA",
    "DISTILLATION_RESULT_SCHEMA",
    "DistillationBudget",
    "DistillationResult",
    "distill_classification_expert",
    "distill_ranking_expert",
    "select_smallest_reliable_form",
)
