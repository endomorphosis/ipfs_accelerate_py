"""Conformance tests for cache-first isolated Hammer/CEGIS doctor verification (LPR-035)."""

from __future__ import annotations

import inspect

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    identify_strict_artifact,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding,
    DoctorAuthorityRoots,
    DoctorEvidenceRole,
    DoctorEvidenceSnapshot,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    HypothesisDisposition,
    LogicHypothesis,
    NativeGoalDisposition,
    ProgramLogicNativeGoalBinding,
    ProofStatus,
    SemanticRoundTripReceipt,
    SourceAuthorityClass,
    SourceRouteKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    HAMMER_IMPORT_ISOLATION,
    HAMMER_IMPORT_ISOLATION_HARDENED,
    get_isolated_hammer_loader,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician import (
    DeterministicDoctorTactician,
    DoctorGoalCompilation,
    DoctorTacticianPlanDisposition,
    DoctorTacticianPlanReceipt,
)
from ipfs_accelerate_py.agent_supervisor.planning.logic_prediction_admission import (
    AutomaticConsequenceKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DETERMINISTIC_DOCTOR_HAMMER_INTERFACE,
    DoctorHammerBounds,
    DoctorHammerDisposition,
    DoctorHammerReasonCode,
    DoctorHammerSafetyError,
    DoctorRepairCandidate,
    DoctorRepairObligationCompiler,
    DoctorRepairProofReceipt,
    DeterministicDoctorHammer,
    NativeReconstructionDisposition,
    NativeReconstructionReceipt,
    create_deterministic_doctor_hammer,
    isolation_is_adequate,
    verify_doctor_repair,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorCacheDisposition,
    DoctorIdentityBinding,
    DoctorProofCacheGate,
    DoctorProofCacheKey,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.tactician_hammer_coordinator import (
    CoordinationConclusiveness,
    HammerCoordinationOutcome,
    HammerCoordinationReceipt,
    PremiseSelectorMode,
)
from ipfs_accelerate_py.agent_supervisor.validation.hammer_native_execution_gate import (
    NativeExecutionLane,
    NativeExecutionOperation,
    NativeExecutionPermit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _doctor_roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _snapshot(roots: DoctorAuthorityRoots | None = None) -> DoctorEvidenceSnapshot:
    roots = roots or _doctor_roots()
    return DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:fixture",
        file_blob_cids=("blob:a", "blob:b"),
        completeness="complete",
        invalidation_refs=(roots.tree_id,),
        clean_rebuild_equivalence_receipt_id="rebuild:eq:1",
    )


def _finding(roots: DoctorAuthorityRoots | None = None, **overrides: object) -> DeterministicDoctorFinding:
    roots = roots or _doctor_roots()
    values: dict[str, object] = {
        "roots": roots,
        "finding_id": "finding:one",
        "snapshot_id": "snapshot:fixture",
        "disposition": DoctorRepairDisposition.SUPPORTED,
        "observed_fact_refs": ("fact:signature-mismatch",),
        "expected_behavior_refs": ("contract:reviewed:accept-input",),
        "evidence_role": DoctorEvidenceRole.OBSERVED_FACT,
        "affected_symbol_refs": ("symbol:process",),
        "consumer_refs": ("consumer:caller",),
        "invalidation_refs": (roots.tree_id,),
    }
    values.update(overrides)
    return DeterministicDoctorFinding(**values)  # type: ignore[arg-type]


def _planned(
    finding: DeterministicDoctorFinding | None = None,
) -> tuple[DoctorTacticianPlanReceipt, DoctorGoalCompilation]:
    finding = finding or _finding()
    tact = DeterministicDoctorTactician()
    compilation = tact.compile_finding(finding, snapshot=_snapshot(finding.roots))
    plan = tact.plan_finding(finding, snapshot=_snapshot(finding.roots), compilation=compilation)
    assert plan.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert compilation.disposition.value == "complete"
    return plan, compilation


def _permit(**overrides: object) -> NativeExecutionPermit:
    values: dict[str, object] = {
        "permit_id": "permit:doctor-hammer",
        "operations": (
            NativeExecutionOperation.PORTFOLIO,
            NativeExecutionOperation.SOLVER,
            NativeExecutionOperation.RECONSTRUCTION,
            NativeExecutionOperation.KERNEL,
            NativeExecutionOperation.COUNTERMODEL_REPLAY,
        ),
        "environment_lock_id": "environment:fixture",
        "lane": NativeExecutionLane.SUPERVISED,
        "allowed_solvers": ("z3",),
    }
    values.update(overrides)
    return NativeExecutionPermit(**values)  # type: ignore[arg-type]


def _hypothesis(
    plan: DoctorTacticianPlanReceipt,
    compilation: DoctorGoalCompilation,
    *,
    hypothesis_id: str = "hyp:unique",
    consequence: str = "consequence:unique-repair",
    value_ref: str = "value:unique-a",
    disposition: HypothesisDisposition = HypothesisDisposition.PROVED,
    proof_status: ProofStatus = ProofStatus.KERNEL_VERIFIED,
) -> LogicHypothesis:
    goal = compilation.goals[0]
    return LogicHypothesis(
        roots=plan.roots,
        hypothesis_id=hypothesis_id,
        target_goal_id=goal.goal_id,
        disposition=disposition,
        claimed_consequence_ref=consequence,
        value_ref=value_ref,
        selected_premise_ids=tuple(compilation.selected_observation_ids)
        or tuple(plan.selected_premise_ids),
        evidence_route_kinds=(SourceRouteKind.LOCAL_STATIC, SourceRouteKind.REVIEWED_CONTRACT),
        source_authority=SourceAuthorityClass.AUTHORITATIVE,
        proof_status=proof_status,
        completeness=True,
        invalidation_refs=(plan.roots.tree_id, plan.roots.corpus_id),
    )


def _native_binding(
    plan: DoctorTacticianPlanReceipt,
    hyp: LogicHypothesis,
    *,
    binding_id: str = "binding:one",
    obligation_id: str = "obligation:logic-ir",
) -> ProgramLogicNativeGoalBinding:
    return ProgramLogicNativeGoalBinding(
        roots=plan.roots,
        binding_id=binding_id,
        logic_ir_obligation_id=obligation_id,
        premise_ids=hyp.selected_premise_ids,
        native_itp_id="itp:lean",
        goal_snapshot_id=f"goal-snapshot:{hyp.hypothesis_id}",
        native_theorem_source_id=f"native-src:{hyp.hypothesis_id}",
        proof_hole_id=f"hole:{hyp.hypothesis_id}",
        kernel_id="kernel:lean4",
        semantic_round_trip=SemanticRoundTripReceipt(
            receipt_id=f"srt:{hyp.hypothesis_id}",
            logic_ir_claim_id=obligation_id,
            native_statement_id=f"native-stmt:{hyp.hypothesis_id}",
            equivalence_method="statement_equivalence",
            disposition=NativeGoalDisposition.ROUND_TRIP_OK,
        ),
        disposition=NativeGoalDisposition.ROUND_TRIP_OK,
        import_ids=("import:Init",),
        invalidation_refs=(plan.roots.tree_id, plan.roots.toolchain_id),
    )


def _verified_hammer(
    plan: DoctorTacticianPlanReceipt,
    hyp: LogicHypothesis,
    *,
    outcome: HammerCoordinationOutcome = HammerCoordinationOutcome.VERIFIED,
    kernel_checked: bool = True,
    proof_success: bool = True,
    binding_id: str = "binding:one",
    obligation_id: str = "obligation:logic-ir",
    translation_map_id: str = "translation-map:exact",
    reconstruction_id: str = "",
) -> HammerCoordinationReceipt:
    conclusive = (
        CoordinationConclusiveness.CONCLUSIVE_PROOF
        if outcome is HammerCoordinationOutcome.VERIFIED and kernel_checked and proof_success
        else CoordinationConclusiveness.NON_CONCLUSIVE
    )
    if outcome is HammerCoordinationOutcome.COUNTEREXAMPLE:
        conclusive = CoordinationConclusiveness.DIAGNOSTIC
    recon = reconstruction_id or f"reconstruction:{hyp.hypothesis_id}"
    return HammerCoordinationReceipt(
        receipt_id=f"hammer:{hyp.hypothesis_id}:{outcome.value}",
        outcome=outcome,
        conclusiveness=conclusive,
        gate_decision={"authorized": True, "permit_id": "permit:doctor-hammer"},
        policy_intersection={"policy_id": plan.roots.policy_id},
        resource_enforcement={"strength": "posix_rlimit"},
        selector_mode=PremiseSelectorMode.DETERMINISTIC,
        translation_map_id=translation_map_id,
        environment_lock_id=plan.roots.environment_id,
        obligation_id=obligation_id,
        request_id=f"request:{hyp.hypothesis_id}",
        native_goal_binding_id=binding_id,
        receipt_binding={
            "reconstruction_id": recon,
            "native_goal_binding_id": binding_id,
        },
        proof_success=proof_success,
        kernel_checked=kernel_checked,
        import_isolation=HAMMER_IMPORT_ISOLATION_HARDENED,
        reason_codes=("ok",) if outcome is HammerCoordinationOutcome.VERIFIED else (),
        metadata={"tree_id": plan.roots.tree_id, "corpus_revision": plan.roots.corpus_id},
    )


def _reconstruction(
    plan: DoctorTacticianPlanReceipt,
    hyp: LogicHypothesis,
    binding: ProgramLogicNativeGoalBinding,
    *,
    matching: bool = True,
) -> NativeReconstructionReceipt:
    disposition = (
        NativeReconstructionDisposition.RECONSTRUCTED
        if matching
        else NativeReconstructionDisposition.MISMATCH
    )
    return NativeReconstructionReceipt(
        receipt_id=f"recon:{hyp.hypothesis_id}",
        disposition=disposition,
        roots=plan.roots,
        obligation_id=binding.logic_ir_obligation_id,
        native_goal_binding_id=binding.binding_id,
        kernel_id=binding.kernel_id,
        toolchain_id=plan.roots.toolchain_id,
        environment_id=plan.roots.environment_id,
        translation_map_id="translation-map:exact",
        theorem_source_id=binding.native_theorem_source_id,
        reconstruction_id=f"reconstruction:{hyp.hypothesis_id}",
        kernel_receipt_id=f"kernel-receipt:{hyp.hypothesis_id}",
        matching_theorem=matching,
        reason_codes=("ok",) if matching else ("reconstruction_failed",),
        invalidation_refs=(plan.roots.tree_id, plan.roots.toolchain_id, binding.kernel_id),
    )


def _identity(name: str, logical_id: str | None = None) -> DoctorIdentityBinding:
    identity = identify_strict_artifact({"component": name, "version": 1})
    return DoctorIdentityBinding.from_identity(
        identity, logical_id=logical_id or f"{name}-1"
    )


def _cache_key(plan: DoctorTacticianPlanReceipt, *, goal_logical: str = "goal-1") -> DoctorProofCacheKey:
    tree = _identity("tree", plan.roots.tree_id)
    return DoctorProofCacheKey(
        forest=_identity("forest", plan.roots.forest_id),
        tree=tree,
        overlay=_identity("overlay", plan.roots.overlay_id),
        ast=_identity("ast", "ast-1"),
        graph=_identity("graph", plan.roots.graph_id),
        corpus=_identity("corpus", plan.roots.corpus_id),
        goal=_identity("goal", goal_logical),
        premises=(
            _identity("premise-a", "premise-a"),
            _identity("premise-b", "premise-b"),
        ),
        translation=_identity("translation", plan.roots.translator_id),
        solver=_identity("solver", "solver-1"),
        kernel=_identity("kernel", "kernel:lean4"),
        toolchain=_identity("toolchain", plan.roots.toolchain_id),
        registry=_identity("registry", "registry-1"),
        policy=_identity("policy", plan.roots.policy_id),
        budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=8_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=2,
            max_premises=4,
            network_allowed=False,
        ),
        sandbox=_identity("sandbox", "sandbox-1"),
        environment=_identity("environment", plan.roots.environment_id),
        candidate_tree=tree,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )


def _kernel_receipt(key: DoctorProofCacheKey) -> ProofReceipt:
    return ProofReceipt(
        obligation_id=key.goal.logical_id,
        plan_id="plan:doctor",
        attempt_id="attempt:doctor",
        repository_id="repository:fixture",
        repository_tree_id=key.tree.logical_id,
        ast_scope_ids=("scope-1",),
        premise_ids=tuple(item.logical_id for item in key.premises),
        translator_id=key.translation.logical_id,
        solver_id=key.solver.logical_id,
        kernel_id=key.kernel.logical_id,
        toolchain_id=key.toolchain.logical_id,
        theorem_registry_id=key.registry.logical_id,
        policy_id=key.policy.logical_id,
        resource_budget=key.budget,
        verdict=ProofVerdict.PROVED,
        evidence=(
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id="kernel-artifact-1",
                subject_id=key.goal.logical_id,
                verifier_id=key.kernel.logical_id,
                independent=True,
            ),
        ),
        freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id="kernel-receipt:goal-1",
        metadata={},
    )


# ---------------------------------------------------------------------------
# Closed vocabulary / safety bounds
# ---------------------------------------------------------------------------


def test_interface_and_bounds_forbid_llm_and_writes() -> None:
    assert DETERMINISTIC_DOCTOR_HAMMER_INTERFACE == "DeterministicDoctorHammer@1"
    bounds = DoctorHammerBounds()
    assert bounds.allow_llm_route is False
    assert bounds.semantic_authority is False
    assert bounds.source_writes_allowed is False
    assert bounds.require_native_permit is True
    assert bounds.require_isolation is True
    assert bounds.require_unique_consequence is True
    with pytest.raises(DoctorHammerSafetyError):
        DoctorHammerBounds(allow_llm_route=True)
    with pytest.raises(DoctorHammerSafetyError):
        DoctorHammerBounds(semantic_authority=True)
    with pytest.raises(DoctorHammerSafetyError):
        DoctorHammerBounds(source_writes_allowed=True)


def test_isolation_probe_requires_hardened_loader() -> None:
    loader = get_isolated_hammer_loader()
    report = loader.isolation_report()
    assert report["import_isolation"] == HAMMER_IMPORT_ISOLATION_HARDENED
    assert isolation_is_adequate(report) is True
    assert isolation_is_adequate({"import_isolation": "unsafe", "concurrency_safe": True}) is False
    assert isolation_is_adequate(
        {
            "import_isolation": HAMMER_IMPORT_ISOLATION_HARDENED,
            "concurrency_safe": False,
        }
    ) is False
    assert isolation_is_adequate(
        {
            "import_isolation": HAMMER_IMPORT_ISOLATION_HARDENED,
            "mutates_home": True,
        }
    ) is False


def test_module_never_imports_llm_router() -> None:
    import ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer as mod

    source = inspect.getsource(mod)
    # No import / call of model providers — denylisted symbols only in closed
    # marker frozensets used for fail-closed rejection, never as live routes.
    assert "import llm_router" not in source
    assert "from llm_router" not in source
    assert "chat_completion(" not in source
    assert "import openai" not in source
    assert "import anthropic" not in source
    # Bounds permanently disable LLM routes.
    assert DoctorHammerBounds().allow_llm_route is False


# ---------------------------------------------------------------------------
# Obligation compiler
# ---------------------------------------------------------------------------


def test_obligation_compiler_binds_identities_for_admitted_plan() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    compiler = DoctorRepairObligationCompiler()
    lowered = compiler.compile(plan, compilation, hypotheses=(hyp,), kernel_id="kernel:lean4")
    assert lowered.disposition.value in {"lowered", "partial"}
    assert lowered.obligation_ids
    assert lowered.native_binding_ids
    assert lowered.translator_id == plan.roots.translator_id
    assert lowered.toolchain_id == plan.roots.toolchain_id
    assert lowered.policy_id == plan.roots.policy_id
    assert lowered.environment_id == plan.roots.environment_id
    assert lowered.kernel_id == "kernel:lean4"
    assert plan.roots.tree_id in lowered.invalidation_refs


def test_obligation_compiler_rejects_unplanned_receipt() -> None:
    plan, compilation = _planned()
    # Force a non-planned disposition via a rejected re-plan style shell.
    bad = DoctorTacticianPlanReceipt(
        roots=plan.roots,
        receipt_id="receipt:bad",
        finding_id=plan.finding_id,
        snapshot_id=plan.snapshot_id,
        compilation_id=plan.compilation_id,
        disposition=DoctorTacticianPlanDisposition.ABSTAINED,
        reason_codes=("finding_abstain",),
        invalidation_refs=plan.invalidation_refs,
    )
    lowered = DoctorRepairObligationCompiler().compile(bad, compilation)
    assert lowered.disposition.value == "rejected"
    assert DoctorHammerReasonCode.PLAN_NOT_ADMITTED.value in lowered.reason_codes


# ---------------------------------------------------------------------------
# Fail-closed abstentions
# ---------------------------------------------------------------------------


def test_abstain_without_native_permit_zero_writes() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    hammer = DeterministicDoctorHammer()
    receipt = hammer.verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=None,
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.PERMIT_MISSING.value in receipt.reason_codes
    assert receipt.source_write_count == 0
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.write_authority is False
    assert receipt.semantic_authority is False


def test_abstain_disabled_permit() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=NativeExecutionPermit.disabled(),
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.PERMIT_DENIED.value in receipt.reason_codes
    assert receipt.source_write_count == 0


def test_abstain_stale_roots() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
        ProgramLogicAuthorityRoots,
    )

    stale_roots = ProgramLogicAuthorityRoots(
        repository_id=plan.roots.repository_id,
        objective_id=plan.roots.objective_id,
        trace_id=plan.roots.trace_id,
        change_id=plan.roots.change_id,
        consumer_id=plan.roots.consumer_id,
        forest_id=plan.roots.forest_id,
        tree_id="tree:stale-other",
        overlay_id=plan.roots.overlay_id,
        graph_id=plan.roots.graph_id,
        index_id=plan.roots.index_id,
        corpus_id=plan.roots.corpus_id,
        model_id=plan.roots.model_id,
        translator_id=plan.roots.translator_id,
        toolchain_id=plan.roots.toolchain_id,
        policy_id=plan.roots.policy_id,
        environment_id=plan.roots.environment_id,
    )

    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        current_roots=stale_roots,
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.STALE_ROOTS.value in receipt.reason_codes
    assert receipt.source_write_count == 0


def test_abstain_zero_candidates() -> None:
    plan, compilation = _planned()
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        candidates=(),
        hypotheses=(),
        permit=_permit(),
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.NO_CANDIDATES.value in receipt.reason_codes


def test_abstain_unavailable_provider() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    # No coordinator, no prebuilt hammer → UNAVAILABLE → abstain.
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.PROVIDER_UNAVAILABLE.value in receipt.reason_codes
    assert receipt.source_write_count == 0
    assert receipt.llm_invocation_count == 0


def test_abstain_timeout_outcome() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    timed = _verified_hammer(plan, hyp, outcome=HammerCoordinationOutcome.TIMEOUT)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": timed},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.TIMEOUT.value in receipt.reason_codes


def test_abstain_inconsistency() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        consistency_disposition=ConsistencyDisposition.STRUCTURAL_CONFLICT,
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.INCONSISTENCY.value in receipt.reason_codes


def test_abstain_ambiguous_multiple_eligible_consequences() -> None:
    plan, compilation = _planned()
    hyp_a = _hypothesis(
        plan,
        compilation,
        hypothesis_id="hyp:a",
        consequence="consequence:a",
        value_ref="value:a",
    )
    hyp_b = _hypothesis(
        plan,
        compilation,
        hypothesis_id="hyp:b",
        consequence="consequence:b",
        value_ref="value:b",
    )
    binding_a = _native_binding(plan, hyp_a, binding_id="binding:a", obligation_id="obligation:a")
    binding_b = _native_binding(plan, hyp_b, binding_id="binding:b", obligation_id="obligation:b")
    recon_a = _reconstruction(plan, hyp_a, binding_a)
    recon_b = _reconstruction(plan, hyp_b, binding_b)
    hammer_a = _verified_hammer(plan, hyp_a)
    hammer_b = _verified_hammer(plan, hyp_b)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp_a, hyp_b),
        permit=_permit(),
        prebuilt_hammer_receipts={
            f"candidate:{hyp_a.hypothesis_id}": hammer_a,
            f"candidate:{hyp_b.hypothesis_id}": hammer_b,
        },
        prebuilt_native_bindings={
            f"candidate:{hyp_a.hypothesis_id}": binding_a,
            f"candidate:{hyp_b.hypothesis_id}": binding_b,
        },
        prebuilt_reconstructions={
            f"candidate:{hyp_a.hypothesis_id}": recon_a,
            f"candidate:{hyp_b.hypothesis_id}": recon_b,
        },
        automatic_kind=AutomaticConsequenceKind.VALUE,
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.MULTIPLE_ELIGIBLE.value in receipt.reason_codes
    assert set(receipt.eligible_consequence_refs) == {"consequence:a", "consequence:b"}
    assert receipt.uniqueness_satisfied is False
    assert receipt.source_write_count == 0


def test_isolation_unavailable_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)

    class _BadLoader:
        def isolation_report(self) -> dict[str, object]:
            return {
                "import_isolation": "import_isolation_unsafe",
                "concurrency_safe": False,
                "mutates_home": True,
                "mutates_sys_prefix": False,
                "process_global": True,
            }

    hammer = DeterministicDoctorHammer(loader=_BadLoader())  # type: ignore[arg-type]
    receipt = hammer.verify(plan, compilation, hypotheses=(hyp,), permit=_permit())
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.ISOLATION_UNAVAILABLE.value in receipt.reason_codes
    assert receipt.isolation_adequate is False
    assert receipt.source_write_count == 0


# ---------------------------------------------------------------------------
# Cache revalidation
# ---------------------------------------------------------------------------


def test_cache_bindings_revalidated_before_use(tmp_path) -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    key = _cache_key(plan)
    gate = DoctorProofCacheGate(path=tmp_path / "doctor-cache.sqlite")
    stored = gate.put(key, _kernel_receipt(key))
    assert stored.stored is True

    hammer_receipt = _verified_hammer(plan, hyp)
    engine = DeterministicDoctorHammer(cache_gate=gate)
    receipt = engine.verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        cache_key=key,
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": hammer_receipt},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert receipt.cache_revalidated is True
    assert receipt.cache_audits
    stages = {item.get("stage") for item in receipt.cache_audits}
    assert "lookup" in stages or "render" in stages or "commit" in stages
    # Successful unique path should admit.
    assert receipt.disposition is DoctorHammerDisposition.ADMITTED
    assert receipt.source_write_count == 0
    assert receipt.llm_invocation_count == 0


# ---------------------------------------------------------------------------
# Countermodel replay before refutation
# ---------------------------------------------------------------------------


def test_raw_countermodel_does_not_refute_without_replay() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    raw_cm = _verified_hammer(
        plan, hyp, outcome=HammerCoordinationOutcome.COUNTEREXAMPLE
    )
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": raw_cm},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
        # No replay / no proof of negation → raw CM is non-authoritative.
    )
    # Zero eligible after raw CM → abstain (not refuted).
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.ZERO_ELIGIBLE.value in receipt.reason_codes
    assert receipt.source_write_count == 0


def test_independently_replayed_countermodel_refutes() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    cm_hammer = _verified_hammer(
        plan, hyp, outcome=HammerCoordinationOutcome.COUNTEREXAMPLE
    )
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": cm_hammer},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
        countermodel_replays={
            f"candidate:{hyp.hypothesis_id}": {
                "status": "validated",
                "replay_method": "deterministic_logic_ir_replay",
                "evidence_id": "replay:evidence:1",
            }
        },
    )
    assert receipt.disposition is DoctorHammerDisposition.REFUTED
    assert receipt.countermodel_validation_ids
    assert receipt.source_write_count == 0
    assert receipt.llm_invocation_count == 0


def test_proof_of_negation_refutes() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    cm_hammer = _verified_hammer(
        plan, hyp, outcome=HammerCoordinationOutcome.COUNTEREXAMPLE
    )
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": cm_hammer},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
        proof_of_negation_ids={
            f"candidate:{hyp.hypothesis_id}": "proof-of-negation:exact"
        },
    )
    assert receipt.disposition is DoctorHammerDisposition.REFUTED
    assert receipt.source_write_count == 0


# ---------------------------------------------------------------------------
# Happy path: unique reconstructed consequence
# ---------------------------------------------------------------------------


def test_unique_reconstructed_consequence_admits() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    hammer_receipt = _verified_hammer(plan, hyp)
    engine = create_deterministic_doctor_hammer()
    receipt = engine.verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": hammer_receipt},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
        automatic_kind=AutomaticConsequenceKind.VALUE,
    )
    assert receipt.disposition is DoctorHammerDisposition.ADMITTED
    assert receipt.is_admitted
    assert receipt.uniqueness_satisfied is True
    assert receipt.selected_consequence_ref == hyp.claimed_consequence_ref
    assert receipt.selected_hypothesis_id == hyp.hypothesis_id
    assert receipt.native_reconstruction is not None
    assert receipt.native_reconstruction.is_reconstructed
    assert receipt.native_reconstruction.matching_theorem is True
    assert receipt.kernel_id
    assert receipt.translator_id == plan.roots.translator_id
    assert receipt.toolchain_id == plan.roots.toolchain_id
    assert receipt.policy_id == plan.roots.policy_id
    assert receipt.environment_id == plan.roots.environment_id
    assert receipt.isolation_adequate is True
    assert receipt.import_isolation in {
        HAMMER_IMPORT_ISOLATION,
        HAMMER_IMPORT_ISOLATION_HARDENED,
    }
    assert receipt.permit_id == "permit:doctor-hammer"
    assert receipt.source_write_count == 0
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.write_authority is False
    assert receipt.semantic_authority is False
    # CEGIS ran (finite, bounded) even with empty evidence stream.
    assert receipt.cegis_disposition in {
        "fixed_point",
        "refined",
        "inconclusive",
        "bound_exhausted",
        "",
    }
    # Round-trip receipt.
    restored = DoctorRepairProofReceipt.from_dict(receipt.to_dict())
    assert restored.disposition is DoctorHammerDisposition.ADMITTED
    assert restored.selected_consequence_ref == receipt.selected_consequence_ref


def test_non_matching_reconstruction_excludes_candidate() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding, matching=False)
    hammer_receipt = _verified_hammer(plan, hyp)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": hammer_receipt},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.ZERO_ELIGIBLE.value in receipt.reason_codes


def test_solver_only_verified_without_kernel_is_not_admitted() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    # Marked verified but without kernel_checked → non-conclusive.
    soft = _verified_hammer(plan, hyp, kernel_checked=False, proof_success=True)
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": soft},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert receipt.disposition is DoctorHammerDisposition.ABSTAINED
    assert DoctorHammerReasonCode.ZERO_ELIGIBLE.value in receipt.reason_codes


def test_cegis_is_finite_and_repetition_bounded() -> None:
    bounds = DoctorHammerBounds(max_cegis_rounds=3, max_repeated_states=2)
    cegis_bounds = bounds.to_cegis_bounds()
    assert cegis_bounds.max_rounds == 3
    assert cegis_bounds.max_repeated_states == 2
    assert cegis_bounds.max_rounds <= 64


def test_verify_doctor_repair_convenience() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    binding = _native_binding(plan, hyp)
    recon = _reconstruction(plan, hyp, binding)
    receipt = verify_doctor_repair(
        plan,
        compilation,
        hypotheses=(hyp,),
        permit=_permit(),
        prebuilt_hammer_receipts={f"candidate:{hyp.hypothesis_id}": _verified_hammer(plan, hyp)},
        prebuilt_native_bindings={f"candidate:{hyp.hypothesis_id}": binding},
        prebuilt_reconstructions={f"candidate:{hyp.hypothesis_id}": recon},
    )
    assert isinstance(receipt, DoctorRepairProofReceipt)
    assert receipt.disposition is DoctorHammerDisposition.ADMITTED


def test_candidate_from_hypothesis_and_body_free() -> None:
    plan, compilation = _planned()
    hyp = _hypothesis(plan, compilation)
    cand = DoctorRepairCandidate.from_hypothesis(hyp, operator_kind="exact_rename")
    assert cand.candidate_id == f"candidate:{hyp.hypothesis_id}"
    assert cand.consequence_ref == hyp.claimed_consequence_ref
    assert cand.semantic_authority is False
    with pytest.raises(Exception):
        DoctorRepairCandidate(
            candidate_id="c:1",
            consequence_ref="cons:1",
            semantic_authority=True,
        )


def test_two_candidates_same_consequence_still_unique() -> None:
    """Multiple candidates sharing one consequence remain unique."""
    plan, compilation = _planned()
    hyp_a = _hypothesis(
        plan,
        compilation,
        hypothesis_id="hyp:a",
        consequence="consequence:shared",
        value_ref="value:shared",
    )
    hyp_b = _hypothesis(
        plan,
        compilation,
        hypothesis_id="hyp:b",
        consequence="consequence:shared",
        value_ref="value:shared",
    )
    binding_a = _native_binding(plan, hyp_a, binding_id="binding:a", obligation_id="obligation:a")
    binding_b = _native_binding(plan, hyp_b, binding_id="binding:b", obligation_id="obligation:b")
    receipt = DeterministicDoctorHammer().verify(
        plan,
        compilation,
        hypotheses=(hyp_a, hyp_b),
        permit=_permit(),
        prebuilt_hammer_receipts={
            f"candidate:{hyp_a.hypothesis_id}": _verified_hammer(plan, hyp_a),
            f"candidate:{hyp_b.hypothesis_id}": _verified_hammer(plan, hyp_b),
        },
        prebuilt_native_bindings={
            f"candidate:{hyp_a.hypothesis_id}": binding_a,
            f"candidate:{hyp_b.hypothesis_id}": binding_b,
        },
        prebuilt_reconstructions={
            f"candidate:{hyp_a.hypothesis_id}": _reconstruction(plan, hyp_a, binding_a),
            f"candidate:{hyp_b.hypothesis_id}": _reconstruction(plan, hyp_b, binding_b),
        },
    )
    assert receipt.disposition is DoctorHammerDisposition.ADMITTED
    assert receipt.selected_consequence_ref == "consequence:shared"
    assert receipt.uniqueness_satisfied is True
    # Deterministic pick among same consequence.
    assert receipt.selected_candidate_id == f"candidate:{hyp_a.hypothesis_id}"
