"""Versioned feature-gated change-propagation pipeline (RPR-044).

Orchestrates the intentional-contract-change path from bound stage artifacts
through atomic plan admission, analytical-or-model step emission, pre-provider
revalidation, checkpointed mutation, and fixed-point completion.

This module enforces *order* and *mandatory gates*.  Stage analyzers remain
in their dedicated modules; hermetic callers supply the already-bound
artifacts (change set, delta, closure, obligations, proofs, transforms) just
as the proof-gated @2 repair route does.  Live extractors may be wired later
without changing the orchestration contract.

Hard invariants:

* feature flag defaults off (legacy repair flow unchanged);
* all mutations invoke :class:`ChangePropagationTransaction`;
* completion invokes :class:`ChangePropagationValidator` and requires a
  current :class:`PropagationCompletionReceipt`;
* analytical success never calls a provider;
* model steps use the canonical bounded
  :class:`ChangePropagationProviderRouter`;
* task/writer scopes equal admitted paths;
* failures preserve abstention and rollback;
* optional heavy modules are imported lazily (cold import path).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from .change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    ChangeSetKind,
    ConsumerMigrationObligation,
    ImpactClosureReceipt,
    PlanDisposition,
    ProgramChangeSet,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    PropagationTransaction,
    TransactionState,
)


CHANGE_PROPAGATION_PIPELINE_INTERFACE: Final[str] = "ChangePropagationPipeline@1"
CHANGE_PROPAGATION_PIPELINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-pipeline@1"
)
CHANGE_PROPAGATION_PIPELINE_VERSION: Final[int] = 1
PRODUCER_ID: Final[str] = "change-propagation-pipeline@1"

PIPELINE_STAGE_ORDER: Final[tuple[str, ...]] = (
    "change_set",
    "delta",
    "graph_index",
    "closure_frontier",
    "consumer_inventory",
    "value_behavior_proof",
    "plan_admission",
    "packet_materialize",
    "pre_provider_gate",
    "transaction",
    "fixed_point_validation",
)


class ChangePropagationPipelineError(ValueError):
    """A typed fail-closed pipeline orchestration error."""


class PipelineDisposition(str, Enum):
    """Closed outcomes for one pipeline run."""

    DISABLED = "disabled"
    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


class PipelineStage(str, Enum):
    """Named pipeline stages in mandatory order."""

    CHANGE_SET = "change_set"
    DELTA = "delta"
    GRAPH_INDEX = "graph_index"
    CLOSURE_FRONTIER = "closure_frontier"
    CONSUMER_INVENTORY = "consumer_inventory"
    VALUE_BEHAVIOR_PROOF = "value_behavior_proof"
    PLAN_ADMISSION = "plan_admission"
    PACKET_MATERIALIZE = "packet_materialize"
    PRE_PROVIDER_GATE = "pre_provider_gate"
    TRANSACTION = "transaction"
    FIXED_POINT_VALIDATION = "fixed_point_validation"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ChangePropagationPipelinePolicy:
    """Feature gate and optional-provider controls for the propagation route."""

    enable_change_propagation: bool = False
    require_fixed_point_completion: bool = True
    allow_provider_for_model_steps: bool = True
    # When true, pure analytical packets never invoke a provider (default).
    analytical_skips_provider: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enable_change_propagation",
            "require_fixed_point_completion",
            "allow_provider_for_model_steps",
            "analytical_skips_provider",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ChangePropagationPipelineError(f"{name} must be a boolean")

    @classmethod
    def from_value(
        cls,
        value: "ChangePropagationPipelinePolicy | Mapping[str, Any] | None",
    ) -> "ChangePropagationPipelinePolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("change propagation pipeline policy must be a mapping")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown change propagation pipeline policy fields: "
                + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_change_propagation": self.enable_change_propagation,
            "require_fixed_point_completion": self.require_fixed_point_completion,
            "allow_provider_for_model_steps": self.allow_provider_for_model_steps,
            "analytical_skips_provider": self.analytical_skips_provider,
        }


@dataclass(frozen=True)
class ChangePropagationPipelineRequest:
    """Bound stage artifacts for one feature-gated propagation run.

    Stages are supplied as already-bound local artifacts so this route can
    enforce order (change set → … → fixed-point validation) without granting
    an optional provider any role in plan admission.
    """

    roots: PropagationAuthorityRoots
    # Stage artifacts (callers may pass evidence_bundle alone when complete).
    change_set: ProgramChangeSet | None = None
    delta: ProgramContractDelta | None = None
    impact_closure: ImpactClosureReceipt | None = None
    obligations: Sequence[ConsumerMigrationObligation] = ()
    analytical_transforms: Sequence[AnalyticalTransform] = ()
    value_mapping_proofs: Sequence[Any] = ()
    placement_decisions: Sequence[Any] = ()
    evidence_bundle: Any = None  # PlanEvidenceBundle when provided
    # Graph / index / inventory receipts (identity-only for hermetic tests).
    graph_id: str = ""
    index_id: str = ""
    consumer_inventory_id: str = ""
    # Optional pre-provider gate inputs (required when model steps exist).
    snapshot: Any = None  # RepositorySnapshot
    capability_report: Any = None
    provider_identity: Any = None
    writer_lease: Any = None
    now: int = 0
    # Mutation / completion (transactional path).
    execute_mutation: bool = False
    transaction_lease: Any = None  # TransactionLease
    path_before_hashes: Sequence[Any] = ()
    step_applicator: Callable[..., Any] | None = None
    restore_adapter: Callable[..., Any] | None = None
    hash_probe: Callable[[str], str] | None = None
    candidate_evidence: Any = None  # CandidatePropagationEvidence
    fixed_point_bound: int | None = None
    # Model-step routing (only when execute_model_steps is true).
    execute_model_steps: bool = False
    analytical_non_success_reason: str = "unsupported_shape"
    model_step_ids: Sequence[str] = ()
    grok_provider: Any = None
    codex_provider: Any = None
    deterministic_provider: Any = None
    current_snapshot_id: str = ""
    task_id: str = ""
    apply_model_writes: bool = False
    # Optional projected task scope fence (must equal admitted writes when set).
    task_write_paths: Sequence[str] | None = None
    writer_write_paths: Sequence[str] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ChangePropagationPipelineRequest":
        if not isinstance(value, Mapping):
            raise TypeError("change propagation pipeline request must be a mapping")
        roots = value["roots"]
        if isinstance(roots, Mapping):
            roots = PropagationAuthorityRoots.from_dict(roots)
        return cls(
            roots=roots,
            change_set=value.get("change_set"),
            delta=value.get("delta"),
            impact_closure=value.get("impact_closure"),
            obligations=tuple(value.get("obligations") or ()),
            analytical_transforms=tuple(value.get("analytical_transforms") or ()),
            value_mapping_proofs=tuple(value.get("value_mapping_proofs") or ()),
            placement_decisions=tuple(value.get("placement_decisions") or ()),
            evidence_bundle=value.get("evidence_bundle"),
            graph_id=str(value.get("graph_id") or ""),
            index_id=str(value.get("index_id") or ""),
            consumer_inventory_id=str(value.get("consumer_inventory_id") or ""),
            snapshot=value.get("snapshot"),
            capability_report=value.get("capability_report"),
            provider_identity=value.get("provider_identity"),
            writer_lease=value.get("writer_lease"),
            now=int(value.get("now") or 0),
            execute_mutation=bool(value.get("execute_mutation") or False),
            transaction_lease=value.get("transaction_lease"),
            path_before_hashes=tuple(value.get("path_before_hashes") or ()),
            step_applicator=value.get("step_applicator"),
            restore_adapter=value.get("restore_adapter"),
            hash_probe=value.get("hash_probe"),
            candidate_evidence=value.get("candidate_evidence"),
            fixed_point_bound=value.get("fixed_point_bound"),
            execute_model_steps=bool(value.get("execute_model_steps") or False),
            analytical_non_success_reason=str(
                value.get("analytical_non_success_reason") or "unsupported_shape"
            ),
            model_step_ids=tuple(value.get("model_step_ids") or ()),
            grok_provider=value.get("grok_provider"),
            codex_provider=value.get("codex_provider"),
            deterministic_provider=value.get("deterministic_provider"),
            current_snapshot_id=str(value.get("current_snapshot_id") or ""),
            task_id=str(value.get("task_id") or ""),
            apply_model_writes=bool(value.get("apply_model_writes") or False),
            task_write_paths=(
                None
                if value.get("task_write_paths") is None
                else tuple(value.get("task_write_paths") or ())
            ),
            writer_write_paths=(
                None
                if value.get("writer_write_paths") is None
                else tuple(value.get("writer_write_paths") or ())
            ),
        )


@dataclass(frozen=True)
class ChangePropagationPipelineResult:
    """Outcome of the feature-gated propagation route."""

    enabled: bool
    stage: str
    disposition: str
    detail: str = ""
    provider_invoked: bool = False
    stages_completed: tuple[str, ...] = ()
    change_set_id: str = ""
    delta_id: str = ""
    graph_id: str = ""
    index_id: str = ""
    impact_closure_id: str = ""
    consumer_inventory_id: str = ""
    plan_id: str = ""
    plan: AtomicPropagationPlan | None = None
    admission: Any = None
    packet: Any = None
    write_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    analytical_step_ids: tuple[str, ...] = ()
    model_required_step_ids: tuple[str, ...] = ()
    gate_receipt: Any = None
    transaction_report: Any = None
    transaction: PropagationTransaction | None = None
    completion: PropagationCompletionReceipt | None = None
    validation_outcome: Any = None
    model_route_results: tuple[Any, ...] = ()
    task_projection: Any = None
    reason_codes: tuple[str, ...] = ()
    rolled_back: bool = False

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            PipelineDisposition.ADMITTED.value,
            PipelineDisposition.COMPLETE.value,
        } and self.packet is not None

    @property
    def complete(self) -> bool:
        return (
            self.disposition == PipelineDisposition.COMPLETE.value
            and self.completion is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANGE_PROPAGATION_PIPELINE_SCHEMA,
            "interface": CHANGE_PROPAGATION_PIPELINE_INTERFACE,
            "pipeline_version": CHANGE_PROPAGATION_PIPELINE_VERSION,
            "enabled": self.enabled,
            "stage": self.stage,
            "disposition": self.disposition,
            "detail": self.detail,
            "provider_invoked": self.provider_invoked,
            "stages_completed": list(self.stages_completed),
            "change_set_id": self.change_set_id,
            "delta_id": self.delta_id,
            "graph_id": self.graph_id,
            "index_id": self.index_id,
            "impact_closure_id": self.impact_closure_id,
            "consumer_inventory_id": self.consumer_inventory_id,
            "plan_id": self.plan_id,
            "write_paths": list(self.write_paths),
            "read_paths": list(self.read_paths),
            "analytical_step_ids": list(self.analytical_step_ids),
            "model_required_step_ids": list(self.model_required_step_ids),
            "reason_codes": list(self.reason_codes),
            "rolled_back": self.rolled_back,
            "complete": self.complete,
            "admitted": self.admitted,
            "provider_success_is_not_completion": True,
            "partial_merge_allowed": False,
        }


def _disabled_result(detail: str) -> ChangePropagationPipelineResult:
    return ChangePropagationPipelineResult(
        enabled=False,
        stage=PipelineStage.DISABLED.value,
        disposition=PipelineDisposition.DISABLED.value,
        detail=detail,
    )


def _fail(
    *,
    stage: str,
    disposition: str,
    detail: str,
    completed: Sequence[str] = (),
    reason_codes: Sequence[str] = (),
    provider_invoked: bool = False,
    rolled_back: bool = False,
    **kwargs: Any,
) -> ChangePropagationPipelineResult:
    return ChangePropagationPipelineResult(
        enabled=True,
        stage=stage,
        disposition=disposition,
        detail=detail,
        provider_invoked=provider_invoked,
        stages_completed=tuple(completed),
        reason_codes=tuple(reason_codes),
        rolled_back=rolled_back,
        **kwargs,
    )


def _scope_equal(
    admitted: Sequence[str],
    declared: Sequence[str] | None,
    *,
    label: str,
) -> str | None:
    if declared is None:
        return None
    if tuple(sorted(admitted)) != tuple(sorted(declared)):
        return (
            f"{label} paths {tuple(sorted(declared))} != admitted "
            f"{tuple(sorted(admitted))}"
        )
    return None


@dataclass
class ChangePropagationPipeline:
    """Feature-gated orchestrator for transactional change propagation.

    Import of transaction, validator, planner, packet, gate, and provider
    modules is deferred until :meth:`run` so cold imports of this module stay
    light.
    """

    INTERFACE: Final[str] = CHANGE_PROPAGATION_PIPELINE_INTERFACE
    VERSION: Final[int] = CHANGE_PROPAGATION_PIPELINE_VERSION

    policy: ChangePropagationPipelinePolicy = field(
        default_factory=ChangePropagationPipelinePolicy
    )

    def __post_init__(self) -> None:
        self.policy = ChangePropagationPipelinePolicy.from_value(self.policy)

    def run(
        self,
        request: ChangePropagationPipelineRequest | Mapping[str, Any],
    ) -> ChangePropagationPipelineResult:
        """Execute the ordered, feature-gated propagation flow (fail-closed)."""

        if not isinstance(request, ChangePropagationPipelineRequest):
            if not isinstance(request, Mapping):
                raise TypeError(
                    "change propagation pipeline request must be a mapping"
                )
            request = ChangePropagationPipelineRequest.from_mapping(request)

        if not self.policy.enable_change_propagation:
            return _disabled_result("enable_change_propagation is false")

        if not isinstance(request.roots, PropagationAuthorityRoots):
            return _fail(
                stage=PipelineStage.CHANGE_SET.value,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="roots must be PropagationAuthorityRoots",
                reason_codes=("malformed_roots",),
            )

        completed: list[str] = []
        provider_invoked = False

        # ------------------------------------------------------------------
        # 1. Capture change set
        # ------------------------------------------------------------------
        stage = PipelineStage.CHANGE_SET.value
        change_set = request.change_set
        evidence = request.evidence_bundle
        if change_set is None and evidence is not None:
            # Derive a minimal change-set identity from evidence + roots.
            try:
                write_paths = tuple(
                    getattr(span, "path", "")
                    for span in getattr(evidence, "write_spans", ()) or ()
                )
                write_paths = tuple(p for p in write_paths if p)
                if not write_paths:
                    write_paths = tuple(
                        getattr(span, "path", "")
                        for span in getattr(evidence, "read_spans", ()) or ()
                    )
                    write_paths = tuple(p for p in write_paths if p)
                if write_paths:
                    change_set = ProgramChangeSet(
                        roots=request.roots,
                        kind=ChangeSetKind.REVIEWED_BASE_CANDIDATE,
                        producer_id=PRODUCER_ID,
                        changed_paths=write_paths,
                        evidence_refs=(
                            str(getattr(evidence, "change_set_id", "") or "evidence:bundle"),
                        ),
                    )
            except Exception as exc:  # fail-closed
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail=f"change set capture failed: {exc}",
                    reason_codes=("change_set_failed",),
                )
        if change_set is None:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="change set capture produced no ProgramChangeSet",
                reason_codes=("missing_change_set",),
            )
        if not isinstance(change_set, ProgramChangeSet):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="change_set must be ProgramChangeSet@1",
                reason_codes=("malformed_change_set",),
            )
        if change_set.roots != request.roots:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="change set roots drifted from request roots",
                reason_codes=("root_drift",),
            )
        change_set_id = change_set.content_id
        if evidence is not None:
            evidence_cs = str(getattr(evidence, "change_set_id", "") or "")
            if evidence_cs:
                change_set_id = evidence_cs
        completed.append(stage)

        # ------------------------------------------------------------------
        # 2. Extract delta
        # ------------------------------------------------------------------
        stage = PipelineStage.DELTA.value
        delta = request.delta
        delta_id = ""
        if delta is not None:
            if not isinstance(delta, ProgramContractDelta):
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.MALFORMED.value,
                    detail="delta must be ProgramContractDelta@1",
                    completed=completed,
                    reason_codes=("malformed_delta",),
                    change_set_id=change_set_id,
                )
            if delta.roots != request.roots:
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail="delta roots drifted",
                    completed=completed,
                    reason_codes=("root_drift",),
                    change_set_id=change_set_id,
                )
            delta_id = delta.content_id
        elif evidence is not None:
            delta_id = str(getattr(evidence, "delta_id", "") or "")
        if not delta_id:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="delta extraction produced no identity",
                completed=completed,
                reason_codes=("missing_delta",),
                change_set_id=change_set_id,
            )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 3. Build graph / index (identity binding)
        # ------------------------------------------------------------------
        stage = PipelineStage.GRAPH_INDEX.value
        graph_id = request.graph_id or request.roots.graph_id
        index_id = request.index_id or request.roots.index_id
        if not graph_id or not index_id:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="graph/index identity missing",
                completed=completed,
                reason_codes=("missing_graph_index",),
                change_set_id=change_set_id,
                delta_id=delta_id,
            )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 4. Compute closure / frontier
        # ------------------------------------------------------------------
        stage = PipelineStage.CLOSURE_FRONTIER.value
        closure = request.impact_closure
        if closure is None and evidence is not None:
            closure = getattr(evidence, "impact_closure", None)
        if closure is None or not isinstance(closure, ImpactClosureReceipt):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="impact closure/frontier missing",
                completed=completed,
                reason_codes=("missing_closure",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
            )
        if closure.roots != request.roots:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="impact closure roots drifted",
                completed=completed,
                reason_codes=("root_drift",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
            )
        impact_closure_id = closure.content_id
        completed.append(stage)

        # ------------------------------------------------------------------
        # 5. Inventory consumers
        # ------------------------------------------------------------------
        stage = PipelineStage.CONSUMER_INVENTORY.value
        consumer_ids = tuple(item.consumer_id for item in closure.consumers)
        if not consumer_ids:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.ABSTAINED.value,
                detail="consumer inventory empty; no migration obligations",
                completed=completed,
                reason_codes=("empty_consumers",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
            )
        consumer_inventory_id = (
            request.consumer_inventory_id
            or f"inventory:{impact_closure_id[:24]}"
        )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 6. Retrieve / prove values and behavior (evidence presence)
        # ------------------------------------------------------------------
        stage = PipelineStage.VALUE_BEHAVIOR_PROOF.value
        obligations = tuple(request.obligations)
        if not obligations and evidence is not None:
            obligations = tuple(getattr(evidence, "obligations", ()) or ())
        transforms = tuple(request.analytical_transforms)
        if not transforms and evidence is not None:
            transforms = tuple(getattr(evidence, "analytical_transforms", ()) or ())
        proofs = tuple(request.value_mapping_proofs)
        if not proofs and evidence is not None:
            proofs = tuple(getattr(evidence, "value_mapping_proofs", ()) or ())
        placements = tuple(request.placement_decisions)
        if not placements and evidence is not None:
            placements = tuple(getattr(evidence, "placement_decisions", ()) or ())
        if not obligations:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.ABSTAINED.value,
                detail="value/behavior proof stage has no obligations",
                completed=completed,
                reason_codes=("missing_obligations",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
            )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 7. Admit atomic plan
        # ------------------------------------------------------------------
        stage = PipelineStage.PLAN_ADMISSION.value
        # Lazy import of planner (cold path).
        from ..planning.change_propagation_plan import (
            ChangePropagationPlanner,
            PlanEvidenceBundle,
            PropagationPlanAdmission,
        )

        if evidence is None:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="plan admission requires a PlanEvidenceBundle",
                completed=completed,
                reason_codes=("missing_evidence_bundle",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
            )
        if not isinstance(evidence, PlanEvidenceBundle):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="evidence_bundle must be PlanEvidenceBundle",
                completed=completed,
                reason_codes=("malformed_evidence",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
            )
        try:
            admission = ChangePropagationPlanner().admit(evidence)
        except Exception as exc:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail=f"plan admission failed: {exc}",
                completed=completed,
                reason_codes=("admission_error",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
            )
        if not isinstance(admission, PropagationPlanAdmission):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="admission result malformed",
                completed=completed,
                reason_codes=("malformed_admission",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
            )
        plan = admission.plan
        if (
            not admission.admitted
            or admission.disposition is not PlanDisposition.ADMITTED
            or not isinstance(plan, AtomicPropagationPlan)
            or plan.disposition is not PlanDisposition.ADMITTED
        ):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.ABSTAINED.value,
                detail="atomic plan abstained; no packet materialization",
                completed=completed,
                reason_codes=tuple(
                    str(r) for r in (admission.reason_codes or ())
                ) or ("plan_abstained",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=getattr(plan, "plan_id", "") or "",
                plan=plan if isinstance(plan, AtomicPropagationPlan) else None,
                admission=admission,
            )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 8. Emit analytical or model steps (materialize packet)
        # ------------------------------------------------------------------
        stage = PipelineStage.PACKET_MATERIALIZE.value
        from ..proof.change_propagation_edit_packet import (
            ChangePropagationEditPacket,
            ChangePropagationEditPacketError,
            materialize_change_propagation_edit_packet,
        )

        try:
            packet = materialize_change_propagation_edit_packet(
                admission,
                roots=request.roots,
                evidence=evidence,
            )
        except ChangePropagationEditPacketError as exc:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail=f"packet materialization failed: {exc}",
                completed=completed,
                reason_codes=("packet_materialize_failed",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
            )
        if not isinstance(packet, ChangePropagationEditPacket):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="materialize did not return ChangePropagationEditPacket@1",
                completed=completed,
                reason_codes=("malformed_packet",),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
            )
        write_paths = tuple(packet.permitted_write_paths)
        read_paths = tuple(packet.permitted_read_paths)
        analytical_ids = tuple(packet.analytical_step_ids)
        model_ids = tuple(packet.model_required_step_ids)

        # Task / writer scopes must equal admitted paths when declared.
        for label, declared in (
            ("task", request.task_write_paths),
            ("writer", request.writer_write_paths),
        ):
            mismatch = _scope_equal(write_paths, declared, label=label)
            if mismatch:
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail=mismatch,
                    completed=completed,
                    reason_codes=("scope_mismatch",),
                    change_set_id=change_set_id,
                    delta_id=delta_id,
                    graph_id=graph_id,
                    index_id=index_id,
                    impact_closure_id=impact_closure_id,
                    consumer_inventory_id=consumer_inventory_id,
                    plan_id=plan.plan_id,
                    plan=plan,
                    admission=admission,
                    packet=packet,
                    write_paths=write_paths,
                    read_paths=read_paths,
                    analytical_step_ids=analytical_ids,
                    model_required_step_ids=model_ids,
                )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 9. Recheck pre-provider gate (always when model steps; optional always)
        # ------------------------------------------------------------------
        stage = PipelineStage.PRE_PROVIDER_GATE.value
        gate_receipt = None
        needs_gate = bool(model_ids) or request.execute_model_steps
        if needs_gate:
            if (
                request.snapshot is None
                or request.capability_report is None
            ):
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail=(
                        "pre-provider gate requires snapshot and capability "
                        "report when model steps exist"
                    ),
                    completed=completed,
                    reason_codes=("gate_inputs_missing",),
                    change_set_id=change_set_id,
                    delta_id=delta_id,
                    graph_id=graph_id,
                    index_id=index_id,
                    impact_closure_id=impact_closure_id,
                    consumer_inventory_id=consumer_inventory_id,
                    plan_id=plan.plan_id,
                    plan=plan,
                    admission=admission,
                    packet=packet,
                    write_paths=write_paths,
                    read_paths=read_paths,
                    analytical_step_ids=analytical_ids,
                    model_required_step_ids=model_ids,
                )
            from ..validation.change_propagation_pre_provider_gate import (
                ChangePropagationPreProviderGate,
                ChangePropagationPreProviderGateError,
            )

            gate = ChangePropagationPreProviderGate()
            step_for_gate = model_ids[0] if model_ids else None
            try:
                reasons = gate.validate(
                    packet,
                    admission,
                    request.snapshot,
                    current_roots=request.roots,
                    capability_report=request.capability_report,
                    now=request.now,
                    step_id=step_for_gate,
                    provider_identity=request.provider_identity,
                    writer_lease=request.writer_lease,
                    impact_closure=closure,
                )
                if reasons:
                    return _fail(
                        stage=stage,
                        disposition=PipelineDisposition.REJECTED.value,
                        detail=(
                            "pre-provider gate blocked: "
                            + ", ".join(r.value for r in reasons)
                        ),
                        completed=completed,
                        reason_codes=tuple(r.value for r in reasons),
                        change_set_id=change_set_id,
                        delta_id=delta_id,
                        graph_id=graph_id,
                        index_id=index_id,
                        impact_closure_id=impact_closure_id,
                        consumer_inventory_id=consumer_inventory_id,
                        plan_id=plan.plan_id,
                        plan=plan,
                        admission=admission,
                        packet=packet,
                        write_paths=write_paths,
                        read_paths=read_paths,
                        analytical_step_ids=analytical_ids,
                        model_required_step_ids=model_ids,
                    )
                gate_receipt = gate.require_valid(
                    packet,
                    admission,
                    request.snapshot,
                    current_roots=request.roots,
                    capability_report=request.capability_report,
                    now=request.now,
                    step_id=step_for_gate,
                    provider_identity=request.provider_identity,
                    writer_lease=request.writer_lease,
                    impact_closure=closure,
                )
            except (
                ChangePropagationPreProviderGateError,
                TypeError,
                ValueError,
            ) as exc:
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail=f"pre-provider gate failed: {exc}",
                    completed=completed,
                    reason_codes=("gate_failed",),
                    change_set_id=change_set_id,
                    delta_id=delta_id,
                    graph_id=graph_id,
                    index_id=index_id,
                    impact_closure_id=impact_closure_id,
                    consumer_inventory_id=consumer_inventory_id,
                    plan_id=plan.plan_id,
                    plan=plan,
                    admission=admission,
                    packet=packet,
                    write_paths=write_paths,
                    read_paths=read_paths,
                    analytical_step_ids=analytical_ids,
                    model_required_step_ids=model_ids,
                )
        completed.append(stage)

        # ------------------------------------------------------------------
        # 9b. Model steps → canonical bounded provider; analytical → no call
        # ------------------------------------------------------------------
        model_route_results: list[Any] = []
        pure_analytical = bool(analytical_ids) and not model_ids
        if pure_analytical and self.policy.analytical_skips_provider:
            # Hard invariant: analytical success makes no provider call.
            provider_invoked = False
        elif request.execute_model_steps and model_ids:
            if not self.policy.allow_provider_for_model_steps:
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail="model steps present but provider routing disabled",
                    completed=completed,
                    reason_codes=("provider_disabled",),
                    change_set_id=change_set_id,
                    delta_id=delta_id,
                    graph_id=graph_id,
                    index_id=index_id,
                    impact_closure_id=impact_closure_id,
                    consumer_inventory_id=consumer_inventory_id,
                    plan_id=plan.plan_id,
                    plan=plan,
                    admission=admission,
                    packet=packet,
                    write_paths=write_paths,
                    read_paths=read_paths,
                    analytical_step_ids=analytical_ids,
                    model_required_step_ids=model_ids,
                    gate_receipt=gate_receipt,
                )
            if request.provider_identity is None:
                return _fail(
                    stage=stage,
                    disposition=PipelineDisposition.REJECTED.value,
                    detail="model steps require provider_identity",
                    completed=completed,
                    reason_codes=("missing_provider_identity",),
                    change_set_id=change_set_id,
                    delta_id=delta_id,
                    graph_id=graph_id,
                    index_id=index_id,
                    impact_closure_id=impact_closure_id,
                    consumer_inventory_id=consumer_inventory_id,
                    plan_id=plan.plan_id,
                    plan=plan,
                    admission=admission,
                    packet=packet,
                    write_paths=write_paths,
                    read_paths=read_paths,
                    analytical_step_ids=analytical_ids,
                    model_required_step_ids=model_ids,
                    gate_receipt=gate_receipt,
                )
            from ..todo_daemon.change_propagation_provider_router import (
                ChangePropagationProviderRouter,
            )

            router = ChangePropagationProviderRouter(
                identity=request.provider_identity,
                grok_provider=request.grok_provider,
                codex_provider=request.codex_provider,
                deterministic_provider=request.deterministic_provider,
            )
            target_steps = (
                tuple(request.model_step_ids)
                if request.model_step_ids
                else model_ids
            )
            for step_id in target_steps:
                # Model writes are never applied via partial daemon paths here;
                # apply_model_writes only proposes under lease for the router.
                # Durable mutation still requires the transaction stage below.
                provider_invoked = True
                route_result = router.route_step(
                    packet,
                    step_id=step_id,
                    analytical_non_success_reason=request.analytical_non_success_reason,
                    current_snapshot_id=request.current_snapshot_id,
                    task_id=request.task_id,
                    apply=bool(request.apply_model_writes),
                    writer_lease=request.writer_lease,
                )
                model_route_results.append(route_result)

        # ------------------------------------------------------------------
        # Early return when mutation is not requested (plan + packet only)
        # ------------------------------------------------------------------
        if not request.execute_mutation:
            return ChangePropagationPipelineResult(
                enabled=True,
                stage=stage,
                disposition=PipelineDisposition.ADMITTED.value,
                detail=(
                    "admitted atomic plan materialized; mutation not requested"
                ),
                provider_invoked=provider_invoked,
                stages_completed=tuple(completed),
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                model_route_results=tuple(model_route_results),
            )

        # ------------------------------------------------------------------
        # 10. Transactional mutation (mandatory for all writes)
        # ------------------------------------------------------------------
        stage = PipelineStage.TRANSACTION.value
        from ..planning.change_propagation_transaction import (
            ChangePropagationTransaction,
            TransactionLease,
        )

        if request.transaction_lease is None:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.REJECTED.value,
                detail="mutation requires a TransactionLease",
                completed=completed,
                reason_codes=("missing_transaction_lease",),
                provider_invoked=provider_invoked,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                model_route_results=tuple(model_route_results),
            )
        if not isinstance(request.transaction_lease, TransactionLease):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="transaction_lease must be TransactionLease",
                completed=completed,
                reason_codes=("malformed_lease",),
                provider_invoked=provider_invoked,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                model_route_results=tuple(model_route_results),
            )

        txn_kwargs: dict[str, Any] = {}
        if request.step_applicator is not None:
            txn_kwargs["step_applicator"] = request.step_applicator
        if request.restore_adapter is not None:
            txn_kwargs["restore_adapter"] = request.restore_adapter
        if request.hash_probe is not None:
            txn_kwargs["hash_probe"] = request.hash_probe
        transaction = ChangePropagationTransaction(**txn_kwargs)
        hashes = tuple(request.path_before_hashes) or tuple(packet.before_hashes)
        report = transaction.execute(
            plan,
            lease=request.transaction_lease,
            path_before_hashes=hashes,
            packet=packet,
        )
        completed.append(stage)

        if not report.committed:
            rolled_back = report.rollback is not None or (
                report.transaction.state is TransactionState.ROLLED_BACK
            )
            return _fail(
                stage=stage,
                disposition=(
                    PipelineDisposition.ROLLED_BACK.value
                    if rolled_back
                    else PipelineDisposition.REJECTED.value
                ),
                detail=(
                    "transaction failed or rolled back: "
                    + ", ".join(report.reason_codes)
                ),
                completed=completed,
                reason_codes=tuple(report.reason_codes),
                provider_invoked=provider_invoked,
                rolled_back=rolled_back,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                transaction_report=report,
                transaction=report.transaction,
                model_route_results=tuple(model_route_results),
            )

        # ------------------------------------------------------------------
        # 11. Fixed-point completion (mandatory)
        # ------------------------------------------------------------------
        stage = PipelineStage.FIXED_POINT_VALIDATION.value
        if not self.policy.require_fixed_point_completion:
            # Policy may only skip the *requirement* for incomplete results in
            # diagnostics; the integrated path still invokes the validator.
            pass

        if request.candidate_evidence is None:
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.INCOMPLETE.value,
                detail=(
                    "fixed-point validation requires CandidatePropagationEvidence"
                ),
                completed=completed,
                reason_codes=("missing_candidate_evidence",),
                provider_invoked=provider_invoked,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                transaction_report=report,
                transaction=report.transaction,
                model_route_results=tuple(model_route_results),
            )

        from ..validation.change_propagation_validation import (
            ChangePropagationValidator,
        )

        validator = ChangePropagationValidator()
        outcome = validator.validate(
            plan,
            report.transaction,
            evidence=request.candidate_evidence,
            packet=packet,
            execution_report=report,
            fixed_point_bound=request.fixed_point_bound,
        )
        completed.append(stage)

        if not outcome.complete or outcome.completion is None:
            reasons = tuple(outcome.report.reason_codes)
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.INCOMPLETE.value,
                detail=(
                    "fixed-point validation incomplete: "
                    + (", ".join(reasons) or "incomplete")
                ),
                completed=completed,
                reason_codes=reasons or ("fixed_point_incomplete",),
                provider_invoked=provider_invoked,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                transaction_report=report,
                transaction=report.transaction,
                completion=outcome.completion,
                validation_outcome=outcome,
                model_route_results=tuple(model_route_results),
            )

        if not isinstance(outcome.completion, PropagationCompletionReceipt):
            return _fail(
                stage=stage,
                disposition=PipelineDisposition.MALFORMED.value,
                detail="validator did not return PropagationCompletionReceipt@1",
                completed=completed,
                reason_codes=("malformed_completion",),
                provider_invoked=provider_invoked,
                change_set_id=change_set_id,
                delta_id=delta_id,
                graph_id=graph_id,
                index_id=index_id,
                impact_closure_id=impact_closure_id,
                consumer_inventory_id=consumer_inventory_id,
                plan_id=plan.plan_id,
                plan=plan,
                admission=admission,
                packet=packet,
                write_paths=write_paths,
                read_paths=read_paths,
                analytical_step_ids=analytical_ids,
                model_required_step_ids=model_ids,
                gate_receipt=gate_receipt,
                transaction_report=report,
                transaction=report.transaction,
                validation_outcome=outcome,
                model_route_results=tuple(model_route_results),
            )

        return ChangePropagationPipelineResult(
            enabled=True,
            stage=stage,
            disposition=PipelineDisposition.COMPLETE.value,
            detail="transactional mutation closed under fixed-point receipt",
            provider_invoked=provider_invoked,
            stages_completed=tuple(completed),
            change_set_id=change_set_id,
            delta_id=delta_id,
            graph_id=graph_id,
            index_id=index_id,
            impact_closure_id=impact_closure_id,
            consumer_inventory_id=consumer_inventory_id,
            plan_id=plan.plan_id,
            plan=plan,
            admission=admission,
            packet=packet,
            write_paths=write_paths,
            read_paths=read_paths,
            analytical_step_ids=analytical_ids,
            model_required_step_ids=model_ids,
            gate_receipt=gate_receipt,
            transaction_report=report,
            transaction=report.transaction,
            completion=outcome.completion,
            validation_outcome=outcome,
            model_route_results=tuple(model_route_results),
        )


def run_change_propagation_pipeline(
    request: ChangePropagationPipelineRequest | Mapping[str, Any],
    *,
    policy: ChangePropagationPipelinePolicy | Mapping[str, Any] | None = None,
) -> ChangePropagationPipelineResult:
    """Module entry point matching :meth:`ChangePropagationPipeline.run`."""

    return ChangePropagationPipeline(
        policy=ChangePropagationPipelinePolicy.from_value(policy)
    ).run(request)


__all__ = [
    "CHANGE_PROPAGATION_PIPELINE_INTERFACE",
    "CHANGE_PROPAGATION_PIPELINE_SCHEMA",
    "CHANGE_PROPAGATION_PIPELINE_VERSION",
    "PIPELINE_STAGE_ORDER",
    "PRODUCER_ID",
    "ChangePropagationPipeline",
    "ChangePropagationPipelineError",
    "ChangePropagationPipelinePolicy",
    "ChangePropagationPipelineRequest",
    "ChangePropagationPipelineResult",
    "PipelineDisposition",
    "PipelineStage",
    "run_change_propagation_pipeline",
]
