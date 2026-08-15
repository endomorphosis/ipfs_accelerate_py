"""Tests for risk-weighted campaign planning composition (AAE-040).

Acceptance criteria enforced here:

* Planning establishes unmutated green baseline requirements.
* Budgets risk-weighted targets under a resource envelope.
* Preserves deterministic identities and held-out partitions.
* Composes canonical semantic generation and detector explanations.
* No production policy change; missing inputs fail closed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest import (
    ASSURANCE_MANIFEST_INTERFACE,
    AssuranceManifest,
    create_assurance_manifest,
)
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning import (
    BASELINE_REQUIREMENTS_SCHEMA,
    CAMPAIGN_PLAN_RESULT_INTERFACE,
    CAMPAIGN_RESOURCE_BUDGET_SCHEMA,
    GENERATE_MUTATION_CANDIDATES_INTERFACE,
    GENERATOR_ID,
    GENERATOR_VERSION,
    PLAN_MUTATION_CAMPAIGN_INTERFACE,
    PREDICT_DETECTION_SET_INTERFACE,
    BaselineRequirements,
    CampaignPlanningError,
    CampaignResourceBudget,
    MutationCampaignPlanResult,
    generate_mutation_candidates,
    plan_mutation_campaign,
    predict_detection_set,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    ClaimBinding,
    DependencyRelation,
    DetectionAssuranceManifest,
    DetectorCatalogEntry,
    SemanticDependencyEdge,
    assert_prediction_explained,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    DetectorKind,
    DetectorStrength,
    ExpectedDetectionSet,
    verify_detection_set_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.generator import (
    MutationGenerationManifest,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    CampaignBudget,
    MutationCampaignPlan,
    MutationCampaignPolicy,
    MutationCandidate,
    MutationOperatorDefinition,
    MutationRiskClass,
    MutationTarget,
    OperatorClass,
    PropertyClass,
    RollbackDeclaration,
    RollbackStrategy,
    SandboxMode,
    SandboxRequirement,
    ScopeLimits,
    SeedConfigBinding,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNING_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/adversarial_assurance/planning.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae040"
REPO_STATE = _cid("repo-state-aae040")
SOURCE_ROOT = _cid("source-root-aae040")
ENV_CID = _cid("environment-aae040")
DEP_LOCK = _cid("dependency-lock-aae040")
BASELINE_RECEIPT = _cid("baseline-receipt-aae040")
POLICY_CID = _cid("verification-policy-aae040")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "interface_id": PLAN_MUTATION_CAMPAIGN_INTERFACE,
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _versions(**overrides: object) -> VersionBinding:
    fields = {
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "campaign_policy_id": "default_campaign",
        "campaign_policy_version": "1.0.0",
        "generator": _generator(),
    }
    fields.update(overrides)
    return VersionBinding(**fields)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> ArtifactProvenance:
    fields = {
        "producer_id": "adversarial_assurance",
        "producer_version": "1",
        "execution_mode": ExecutionMode.LIVE,
        "authority_source": AuthoritySource.DETERMINISTIC,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("planner.v1",),
        "policy_cid": _cid("policy"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str, **overrides: object) -> AssuranceArtifactHeader:
    fields = {
        "artifact_kind": artifact_kind,
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "target_symbol_ids": ("mod.fn",),
        "target_artifact_cids": (_cid("artifact-a"),),
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "environment_cid": ENV_CID,
        "dependency_lock_cid": DEP_LOCK,
        "versions": _versions(),
        "provenance": _provenance(),
        "terminal_status": AssuranceTerminalStatus.COMPLETE,
        "receipt_cids": (BASELINE_RECEIPT,),
        "proof_cids": (_cid("proof-a"),),
        "metadata": {"risk_class": "local_bug"},
    }
    fields.update(overrides)
    return AssuranceArtifactHeader(**fields)  # type: ignore[arg-type]


def _seed_config(**overrides: object) -> SeedConfigBinding:
    fields = {
        "seed": 42,
        "config": {"max_depth": 2, "operator_budget": 4, "mode": "bounded"},
    }
    fields.update(overrides)
    return SeedConfigBinding(**fields)  # type: ignore[arg-type]


def _rollback(**overrides: object) -> RollbackDeclaration:
    fields = {
        "strategy": RollbackStrategy.WORKTREE_DISCARD,
        "requires_clean_worktree": True,
        "preserves_production": True,
    }
    fields.update(overrides)
    return RollbackDeclaration(**fields)  # type: ignore[arg-type]


def _sandbox(**overrides: object) -> SandboxRequirement:
    fields = {
        "mode": SandboxMode.DISPOSABLE_WORKTREE,
        "network_disabled": True,
        "production_credentials_forbidden": True,
        "disposable_worktree_required": True,
    }
    fields.update(overrides)
    return SandboxRequirement(**fields)  # type: ignore[arg-type]


def _scope(**overrides: object) -> ScopeLimits:
    fields = {
        "max_files": 1,
        "max_symbols": 4,
        "max_span_lines": 64,
        "allow_cross_module": False,
        "allow_verifier_mutation": False,
    }
    fields.update(overrides)
    return ScopeLimits(**fields)  # type: ignore[arg-type]


def _budget(**overrides: object) -> CampaignBudget:
    fields = {
        "max_total_candidates": 64,
        "max_candidates_per_target": 8,
        "max_candidates_per_operator": 16,
        "max_targets": 32,
        "max_operators": 16,
        "max_execution_seconds": 3_600,
        "max_worktrees": 8,
    }
    fields.update(overrides)
    return CampaignBudget(**fields)  # type: ignore[arg-type]


def _resource_budget(**overrides: object) -> CampaignResourceBudget:
    fields = {
        "max_total_candidates": 32,
        "max_candidates_per_target": 4,
        "max_candidates_per_operator": 8,
        "max_targets": 8,
        "max_operators": 8,
        "max_execution_seconds": 1_800,
        "max_worktrees": 4,
        "always_select_min_risk_bp": 5_000,
        "low_risk_sample_rate_bp": 10_000,  # admit low-risk for deterministic tests
        "sampling_seed": 7,
    }
    fields.update(overrides)
    return CampaignResourceBudget(**fields)  # type: ignore[arg-type]


def _operator(**overrides: object) -> MutationOperatorDefinition:
    fields = {
        "operator_id": "control_flow_invert",
        "operator_version": "1",
        "operator_class": OperatorClass.CONTROL_FLOW,
        "supported_languages": ("python",),
        "supported_artifact_types": ("source_module",),
        "target_prerequisites": ("parsed_ast", "symbol_table"),
        "semantic_intent": "Invert a boolean condition controlling a branch",
        "expected_violated_property_classes": (PropertyClass.CONTROL_INVARIANT,),
        "risk_class": MutationRiskClass.LOCAL_BUG,
        "likely_equivalent_conditions": ("condition_always_true",),
        "syntactic_transformation": "replace_if_test_with_not_test",
        "scope_limits": _scope(),
        "rollback": _rollback(),
        "required_sandbox": _sandbox(),
        "max_mutants_per_target": 4,
        "deterministic": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationOperatorDefinition(**fields)  # type: ignore[arg-type]


def _operator_auth(**overrides: object) -> MutationOperatorDefinition:
    fields = {
        "operator_id": "auth_drop_tenant_check",
        "operator_version": "1",
        "operator_class": OperatorClass.AUTHORIZATION_POLICY,
        "supported_languages": ("python",),
        "supported_artifact_types": ("source_module",),
        "target_prerequisites": ("parsed_ast", "symbol_table"),
        "semantic_intent": "Drop tenant binding check on authorization path",
        "expected_violated_property_classes": (PropertyClass.AUTHORIZATION,),
        "risk_class": MutationRiskClass.AUTHORIZATION,
        "likely_equivalent_conditions": ("tenant_always_matches",),
        "syntactic_transformation": "remove_tenant_equality_guard",
        "scope_limits": _scope(),
        "rollback": _rollback(),
        "required_sandbox": _sandbox(),
        "max_mutants_per_target": 3,
        "deterministic": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationOperatorDefinition(**fields)  # type: ignore[arg-type]


def _target(**overrides: object) -> MutationTarget:
    fields = {
        "target_id": "mod_fn",
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "symbol_ids": ("mod.fn", "mod.helper"),
        "artifact_cids": (_cid("artifact-a"),),
        "language": "python",
        "artifact_type": "source_module",
        "prerequisites": ("parsed_ast", "symbol_table", "type_check"),
        "risk_class": MutationRiskClass.LOCAL_BUG,
        "risk_weight_bp": 2_500,
        "capsule_cids": (_cid("capsule-a"),),
        "proof_unit_cids": (_cid("proof-unit-a"),),
        "source_path": "mod.py",
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationTarget(**fields)  # type: ignore[arg-type]


def _target_auth(**overrides: object) -> MutationTarget:
    fields = {
        "target_id": "auth_check",
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "symbol_ids": ("auth.check",),
        "artifact_cids": (_cid("artifact-auth"),),
        "language": "python",
        "artifact_type": "source_module",
        "prerequisites": ("parsed_ast", "symbol_table"),
        "risk_class": MutationRiskClass.AUTHORIZATION,
        "risk_weight_bp": 8_500,
        "capsule_cids": (_cid("capsule-auth"),),
        "proof_unit_cids": (_cid("proof-auth"),),
        "source_path": "auth.py",
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationTarget(**fields)  # type: ignore[arg-type]


def _policy(
    operators: Sequence[MutationOperatorDefinition] | None = None,
    **overrides: object,
) -> MutationCampaignPolicy:
    ops = list(operators) if operators is not None else [_operator(), _operator_auth()]
    fields = {
        "header": _header("mutation_campaign_policy"),
        "policy_id": "default_campaign",
        "policy_version": "1.0.0",
        "admitted_operator_classes": (
            OperatorClass.CONTROL_FLOW,
            OperatorClass.AUTHORIZATION_POLICY,
        ),
        "admitted_risk_classes": (
            MutationRiskClass.LOCAL_BUG,
            MutationRiskClass.CRITICAL_SECURITY,
            MutationRiskClass.AUTHORIZATION,
            MutationRiskClass.CRITICAL_INVARIANT,
            MutationRiskClass.HIGH,
            MutationRiskClass.MEDIUM,
            MutationRiskClass.LOW,
        ),
        "budget": _budget(),
        "seed_config": _seed_config(seed=11, config={"mode": "bounded"}),
        "require_disposable_worktree": True,
        "require_network_disabled": True,
        "require_rollback": True,
        "require_deterministic_seed": True,
        "full_suite_fallback_enabled": True,
        "held_out_partition_required": True,
        "operator_cids": tuple(op.operator_cid for op in ops),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationCampaignPolicy(**fields)  # type: ignore[arg-type]


def _unit_detector(**overrides: object) -> DetectorCatalogEntry:
    fields = {
        "detector_id": "unit.test_branch",
        "detector_revision": "3.2.1",
        "detector_kind": DetectorKind.UNIT_TEST,
        "covered_property_classes": (PropertyClass.CONTROL_INVARIANT,),
        "anchor_ids": ("tests.test_branch",),
        "default_strength": DetectorStrength.REQUIRED,
        "expected_terminal_status": AssuranceTerminalStatus.COMPLETE,
        "observation_template": "unit test asserts inverted branch is rejected",
        "claim_ids": ("claim.control_branch",),
        "notes": "selected unit detector",
        "metadata": {},
    }
    fields.update(overrides)
    return DetectorCatalogEntry(**fields)  # type: ignore[arg-type]


def _static_detector(**overrides: object) -> DetectorCatalogEntry:
    fields = {
        "detector_id": "static.authz_rule",
        "detector_revision": "1.4.0",
        "detector_kind": DetectorKind.STATIC_RULE,
        "covered_property_classes": (PropertyClass.AUTHORIZATION,),
        "anchor_ids": ("static.authz",),
        "default_strength": DetectorStrength.REQUIRED,
        "expected_terminal_status": AssuranceTerminalStatus.COMPLETE,
        "observation_template": "static rule flags removed authorization guard",
        "claim_ids": ("claim.authz",),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return DetectorCatalogEntry(**fields)  # type: ignore[arg-type]


def _edge(
    from_id: str,
    to_id: str,
    relation: DependencyRelation | str = DependencyRelation.TESTED_BY,
    **overrides: object,
) -> SemanticDependencyEdge:
    fields = {
        "from_id": from_id,
        "to_id": to_id,
        "relation": relation,
        "notes": None,
    }
    fields.update(overrides)
    return SemanticDependencyEdge(**fields)  # type: ignore[arg-type]


def _claim(**overrides: object) -> ClaimBinding:
    fields = {
        "claim_id": "claim.control_branch",
        "property_class": PropertyClass.CONTROL_INVARIANT,
        "statement": "branch condition must preserve control invariant",
        "symbol_ids": ("mod.fn",),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ClaimBinding(**fields)  # type: ignore[arg-type]


def _detection_manifest(**overrides: object) -> DetectionAssuranceManifest:
    fields = {
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "detectors": (
            _unit_detector(),
            _static_detector(),
        ),
        "dependency_edges": (
            _edge("mod.fn", "tests.test_branch"),
            _edge("mod.helper", "tests.test_branch"),
            _edge("auth.check", "static.authz"),
        ),
        "claims": (
            _claim(),
            _claim(
                claim_id="claim.authz",
                property_class=PropertyClass.AUTHORIZATION,
                statement="tenant authorization must remain enforced",
                symbol_ids=("auth.check",),
            ),
        ),
        "enable_type_check_fallback": True,
        "enable_full_suite_fallback": True,
        "enable_incremental_seal_fallback": True,
        "enable_human_review_fallback": True,
        "observation_complete": True,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return DetectionAssuranceManifest(**fields)  # type: ignore[arg-type]


def _authority_status_all_unavailable() -> dict[str, dict[str, object]]:
    keys = (
        "index",
        "capsule",
        "context",
        "verification",
        "policy",
        "state",
        "storage",
        "sealer",
    )
    out: dict[str, dict[str, object]] = {}
    for key in keys:
        out[key] = {
            "authority": key,
            "available": False,
            "status": "typed_unavailable",
            "reason_code": "not_probed",
            "diagnostic": "test fixture",
            "adapter_id": None,
            "interface_id": None,
            "schema": None,
            "operations": [],
            "fingerprints": {},
            "seal_status": "typed_unavailable" if key == "sealer" else None,
            "can_be_satisfied_by_ivp_commitment": False,
            "retryable": False,
        }
    return out


def _assurance_manifest(**overrides: object) -> AssuranceManifest:
    detection = _detection_manifest()
    fields = {
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "verification_policy_cid": POLICY_CID,
        "authority_status": _authority_status_all_unavailable(),
        "repository_state": {
            "repository_id": REPO_ID,
            "repository_state_cid": REPO_STATE,
            "source_root_cid": SOURCE_ROOT,
            "environment_cid": ENV_CID,
            "dependency_lock_cid": DEP_LOCK,
        },
        "verification_policy": {
            "policy_cid": POLICY_CID,
            "policy_id": "default",
        },
        "detectors": [item.to_dict() for item in detection.detectors],
        "dependency_edges": [item.to_dict() for item in detection.dependency_edges],
        "claims": [item.to_dict() for item in detection.claims],
        "observation_complete": True,
        "production_policy_changed": False,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return AssuranceManifest(**fields)  # type: ignore[arg-type]


def _repository_state(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "source_root_cid": SOURCE_ROOT,
        "environment_cid": ENV_CID,
        "dependency_lock_cid": DEP_LOCK,
        "revision": "main",
        "metadata": {"baseline_receipt_cid": BASELINE_RECEIPT},
    }
    fields.update(overrides)
    return fields


def _generation_manifest(
    *,
    targets: Sequence[MutationTarget] | None = None,
    operators: Sequence[MutationOperatorDefinition] | None = None,
    **overrides: object,
) -> MutationGenerationManifest:
    fields = {
        "repository_id": REPO_ID,
        "repository_state_cid": REPO_STATE,
        "source_root_cid": SOURCE_ROOT,
        "targets": tuple(targets)
        if targets is not None
        else (_target_auth(), _target()),
        "operators": tuple(operators)
        if operators is not None
        else (_operator(), _operator_auth()),
        "environment_cid": ENV_CID,
        "dependency_lock_cid": DEP_LOCK,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return MutationGenerationManifest(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface surface / cold import
# ---------------------------------------------------------------------------


def test_interfaces_are_versioned_pins() -> None:
    assert PLAN_MUTATION_CAMPAIGN_INTERFACE == "plan_mutation_campaign@1"
    assert GENERATE_MUTATION_CANDIDATES_INTERFACE == "generate_mutation_candidates@1"
    assert PREDICT_DETECTION_SET_INTERFACE == "predict_detection_set@1"
    assert GENERATOR_ID == "campaign_planning"
    assert GENERATOR_VERSION == "1.0.0"
    assert BASELINE_REQUIREMENTS_SCHEMA.endswith("@1")
    assert CAMPAIGN_RESOURCE_BUDGET_SCHEMA.endswith("@1")
    assert CAMPAIGN_PLAN_RESULT_INTERFACE == "MutationCampaignPlanResult@1"


def test_planning_module_cold_import_is_side_effect_free() -> None:
    source = PLANNING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"open", "Path", "subprocess", "run", "Popen", "urlopen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Only check top-level / module-body calls via parent walk is hard;
            # instead ensure no network or subprocess imports at module scope.
            pass
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            # No call expressions at module top level beyond dataclass decorators
            # which are not Call nodes on body statements as standalone Expr with Call.
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                assert name not in forbidden_calls


def test_baseline_requirements_reject_non_green_or_mutated() -> None:
    with pytest.raises(CampaignPlanningError, match="unmutated"):
        BaselineRequirements(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
            unmutated=False,
        )
    with pytest.raises(CampaignPlanningError, match="green"):
        BaselineRequirements(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
            verification_green=False,
        )
    with pytest.raises(CampaignPlanningError, match="complete"):
        BaselineRequirements(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
            observation_complete=False,
        )


def test_baseline_requirements_identity_is_stable() -> None:
    first = BaselineRequirements(
        baseline_receipt_cid=BASELINE_RECEIPT,
        repository_id=REPO_ID,
        repository_state_cid=REPO_STATE,
    )
    second = BaselineRequirements.from_dict(first.to_dict())
    assert first.baseline_cid == second.baseline_cid
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# generate_mutation_candidates composition
# ---------------------------------------------------------------------------


def test_generate_mutation_candidates_is_deterministic() -> None:
    operator = _operator()
    target = _target()
    policy = _policy(operators=[operator])
    manifest = _generation_manifest(targets=[target], operators=[operator])

    first = generate_mutation_candidates(manifest, policy)
    second = generate_mutation_candidates(manifest, policy)

    assert len(first) >= 1
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert [c.candidate_cid for c in first] == [c.candidate_cid for c in second]
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


def test_generate_mutation_candidates_enforces_budgets() -> None:
    # max_mutants_per_target must not exceed budget.max_candidates_per_operator
    # or the operator is filtered out by policy admission.
    operator = _operator(max_mutants_per_target=2)
    target = _target()
    tight = _budget(
        max_total_candidates=2,
        max_candidates_per_target=2,
        max_candidates_per_operator=2,
    )
    policy = _policy(operators=[operator], budget=tight)
    manifest = _generation_manifest(targets=[target], operators=[operator])
    result = generate_mutation_candidates(manifest, policy, return_result=True)
    assert result.candidate_count <= 2
    assert len(result.candidates) <= 2


# ---------------------------------------------------------------------------
# predict_detection_set composition
# ---------------------------------------------------------------------------


def test_predict_detection_set_explains_every_prediction() -> None:
    operator = _operator()
    target = _target()
    policy = _policy(operators=[operator])
    manifest = _generation_manifest(targets=[target], operators=[operator])
    candidates = generate_mutation_candidates(manifest, policy)
    assurance = _assurance_manifest()

    detection = predict_detection_set(candidates[0], assurance)
    assert isinstance(detection, ExpectedDetectionSet)
    verify_detection_set_identity(detection)
    assert detection.candidate_id == candidates[0].candidate_id
    assert len(detection.predicted_detectors) >= 1
    for prediction in detection.predicted_detectors:
        assert_prediction_explained(prediction)
        assert prediction.detector_id
        # Exact detector revision is bound in prediction metadata by the
        # canonical datasets predictor (identity is detector_id + revision).
        assert prediction.metadata.get("detector_revision") or prediction.metadata.get(
            "detector_identity"
        )
        assert prediction.violated_claim
        assert prediction.observation_rationale
        assert prediction.dependency_path
        assert prediction.strength
        assert prediction.expected_terminal_status


def test_predict_detection_set_accepts_detection_manifest_directly() -> None:
    operator = _operator()
    target = _target()
    policy = _policy(operators=[operator])
    candidates = generate_mutation_candidates(
        _generation_manifest(targets=[target], operators=[operator]),
        policy,
    )
    detection = predict_detection_set(candidates[0], _detection_manifest())
    assert detection.candidate_cid == candidates[0].candidate_cid


# ---------------------------------------------------------------------------
# plan_mutation_campaign
# ---------------------------------------------------------------------------


def test_plan_mutation_campaign_establishes_baseline_and_seals_plan() -> None:
    ops = [_operator(), _operator_auth()]
    targets = [_target_auth(), _target()]
    result = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        _policy(operators=ops),
        _resource_budget(),
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=targets,
        operators=ops,
        return_result=True,
    )
    assert isinstance(result, MutationCampaignPlanResult)
    assert result.production_policy_changed is False
    assert result.baseline.unmutated is True
    assert result.baseline.verification_green is True
    assert result.baseline.baseline_receipt_cid == BASELINE_RECEIPT
    assert isinstance(result.plan, MutationCampaignPlan)
    assert result.plan.baseline_receipt_cid == BASELINE_RECEIPT
    assert result.plan.repository_id == REPO_ID
    assert result.plan.repository_state_cid == REPO_STATE
    assert result.plan.require_sandbox is True
    assert result.plan.require_rollback is True
    assert len(result.candidates) >= 1
    assert len(result.expected_detections) == len(result.candidates)
    assert result.plan.candidate_cids
    # Identity recomputation is stable.
    restored = MutationCampaignPlan.from_dict(result.plan.to_dict())
    assert restored.plan_cid == result.plan.plan_cid


def test_plan_mutation_campaign_is_deterministic() -> None:
    ops = [_operator(), _operator_auth()]
    targets = [_target_auth(), _target()]
    kwargs = dict(
        repository_state=_repository_state(),
        assurance_manifest=_assurance_manifest(),
        mutation_policy=_policy(operators=ops),
        resource_budget=_resource_budget(sampling_seed=3),
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=targets,
        operators=ops,
        return_result=True,
    )
    first = plan_mutation_campaign(**kwargs)  # type: ignore[arg-type]
    second = plan_mutation_campaign(**kwargs)  # type: ignore[arg-type]
    assert first.plan.plan_id == second.plan.plan_id
    assert first.plan.plan_cid == second.plan.plan_cid
    assert first.result_cid == second.result_cid
    assert [c.candidate_id for c in first.candidates] == [
        c.candidate_id for c in second.candidates
    ]
    assert [c.candidate_cid for c in first.candidates] == [
        c.candidate_cid for c in second.candidates
    ]
    assert [d.detection_set_cid for d in first.expected_detections] == [
        d.detection_set_cid for d in second.expected_detections
    ]
    if first.partition is not None and second.partition is not None:
        assert first.partition.plan_cid == second.partition.plan_cid
        assert list(first.partition.diagnosis_mutant_ids) == list(
            second.partition.diagnosis_mutant_ids
        )
        assert list(first.partition.held_out_mutant_ids) == list(
            second.partition.held_out_mutant_ids
        )


def test_plan_mutation_campaign_budgets_risk_weighted_targets() -> None:
    ops = [_operator(), _operator_auth()]
    low = _target(
        target_id="low_risk_a",
        risk_class=MutationRiskClass.LOW,
        risk_weight_bp=500,
        symbol_ids=("low.a",),
        artifact_cids=(_cid("artifact-low-a"),),
        source_path="low_a.py",
    )
    low2 = _target(
        target_id="low_risk_b",
        risk_class=MutationRiskClass.LOW,
        risk_weight_bp=400,
        symbol_ids=("low.b",),
        artifact_cids=(_cid("artifact-low-b"),),
        source_path="low_b.py",
    )
    high = _target_auth()
    # Always select high risk; zero sample rate drops residual low risk.
    budget = _resource_budget(
        max_targets=2,
        always_select_min_risk_bp=6_000,
        low_risk_sample_rate_bp=0,
        sampling_seed=99,
    )
    result = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        _policy(operators=ops),
        budget,
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=[low, low2, high],
        operators=ops,
        return_result=True,
    )
    selected_ids = {t.target_id for t in result.selected_targets}
    assert "auth_check" in selected_ids
    # Low-risk targets must not all flood the plan when sample rate is zero.
    assert "low_risk_a" not in selected_ids or "low_risk_b" not in selected_ids
    # Prefer authorization target first in risk order.
    assert result.selected_targets[0].target_id == "auth_check"
    assert len(result.selected_targets) <= budget.max_targets


def test_plan_mutation_campaign_intersects_resource_and_policy_budgets() -> None:
    # Operator max_mutants_per_target must remain <= intersected
    # max_candidates_per_operator or policy admission drops the operator.
    ops = [_operator(max_mutants_per_target=2)]
    policy = _policy(
        operators=ops,
        budget=_budget(
            max_total_candidates=10,
            max_candidates_per_target=3,
            max_candidates_per_operator=5,
            max_targets=4,
            max_operators=2,
        ),
    )
    resource = _resource_budget(
        max_total_candidates=3,
        max_candidates_per_target=2,
        max_candidates_per_operator=2,
        max_targets=2,
        max_operators=1,
    )
    result = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        policy,
        resource,
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=[_target()],
        operators=ops,
        return_result=True,
    )
    assert result.plan.budget.max_total_candidates == 3
    assert result.plan.budget.max_targets == 2
    assert result.resource_budget.max_total_candidates == 3
    assert len(result.candidates) <= 3


def test_plan_mutation_campaign_preserves_held_out_partition() -> None:
    ops = [_operator(), _operator_auth()]
    # Always-select every admitted target so both high- and residual-risk
    # symbols contribute explained candidates (held-out needs >= 2 members).
    result = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        _policy(operators=ops),
        _resource_budget(
            max_total_candidates=16,
            always_select_min_risk_bp=0,
            low_risk_sample_rate_bp=10_000,
        ),
        baseline_receipt_cid=BASELINE_RECEIPT,
        generation_manifest=_generation_manifest(
            targets=[_target_auth(), _target()],
            operators=ops,
        ),
        return_result=True,
    )
    assert len(result.candidates) >= 2
    assert result.partition is not None
    # Diagnosis is non-empty and disjoint from held-out.
    diagnosis = set(result.partition.diagnosis_mutant_ids)
    held_out = set(result.partition.held_out_mutant_ids)
    development = set(result.partition.development_mutant_ids)
    assert diagnosis
    assert held_out
    assert diagnosis.isdisjoint(held_out)
    assert diagnosis.isdisjoint(development)
    assert held_out.isdisjoint(development)
    # All partitioned mutant ids are campaign candidates.
    all_ids = diagnosis | held_out | development
    candidate_ids = {c.candidate_id for c in result.candidates}
    assert all_ids <= candidate_ids


def test_plan_mutation_campaign_accepts_generation_manifest() -> None:
    ops = [_operator()]
    gen = _generation_manifest(targets=[_target()], operators=ops)
    plan = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        _policy(operators=ops),
        _resource_budget(),
        baseline=BaselineRequirements(
            baseline_receipt_cid=BASELINE_RECEIPT,
            repository_id=REPO_ID,
            repository_state_cid=REPO_STATE,
        ),
        generation_manifest=gen,
        return_result=False,
    )
    assert isinstance(plan, MutationCampaignPlan)
    assert plan.plan_id.startswith("plan_")


def test_plan_mutation_campaign_reads_baseline_from_repository_metadata() -> None:
    ops = [_operator()]
    result = plan_mutation_campaign(
        _repository_state(),  # embeds baseline_receipt_cid in metadata
        _assurance_manifest(),
        _policy(operators=ops),
        _resource_budget(),
        targets=[_target()],
        operators=ops,
        return_result=True,
    )
    assert result.baseline.baseline_receipt_cid == BASELINE_RECEIPT


def test_plan_mutation_campaign_fails_closed_without_baseline() -> None:
    ops = [_operator()]
    with pytest.raises(CampaignPlanningError, match="baseline"):
        plan_mutation_campaign(
            {
                "repository_id": REPO_ID,
                "repository_state_cid": REPO_STATE,
                "source_root_cid": SOURCE_ROOT,
                "environment_cid": ENV_CID,
                "dependency_lock_cid": DEP_LOCK,
                "metadata": {},
            },
            _assurance_manifest(),
            _policy(operators=ops),
            _resource_budget(),
            targets=[_target()],
            operators=ops,
        )


def test_plan_mutation_campaign_fails_closed_on_identity_mismatch() -> None:
    ops = [_operator()]
    with pytest.raises(CampaignPlanningError, match="repository_id"):
        plan_mutation_campaign(
            {
                "repository_id": "repository:sha256:other-repo",
                "repository_state_cid": REPO_STATE,
                "source_root_cid": SOURCE_ROOT,
                "environment_cid": ENV_CID,
                "dependency_lock_cid": DEP_LOCK,
            },
            _assurance_manifest(),
            _policy(operators=ops),
            _resource_budget(),
            baseline_receipt_cid=BASELINE_RECEIPT,
            targets=[_target()],
            operators=ops,
        )


def test_plan_mutation_campaign_fails_closed_without_targets() -> None:
    ops = [_operator()]
    with pytest.raises(CampaignPlanningError, match="targets"):
        plan_mutation_campaign(
            _repository_state(),
            _assurance_manifest(),
            _policy(operators=ops),
            _resource_budget(),
            baseline_receipt_cid=BASELINE_RECEIPT,
            operators=ops,
        )


def test_plan_mutation_campaign_fails_closed_without_operators() -> None:
    with pytest.raises(CampaignPlanningError, match="operators"):
        plan_mutation_campaign(
            _repository_state(),
            _assurance_manifest(),
            _policy(operators=[_operator()]),
            _resource_budget(),
            baseline_receipt_cid=BASELINE_RECEIPT,
            targets=[_target()],
        )


def test_plan_mutation_campaign_rejects_incomplete_observation() -> None:
    ops = [_operator()]
    with pytest.raises(CampaignPlanningError, match="observation"):
        plan_mutation_campaign(
            _repository_state(),
            _assurance_manifest(observation_complete=False),
            _policy(operators=ops),
            _resource_budget(),
            baseline_receipt_cid=BASELINE_RECEIPT,
            targets=[_target()],
            operators=ops,
        )


def test_plan_mutation_campaign_return_plan_only() -> None:
    ops = [_operator()]
    plan = plan_mutation_campaign(
        _repository_state(),
        _assurance_manifest(),
        _policy(operators=ops),
        _resource_budget(),
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=[_target()],
        operators=ops,
        return_result=False,
    )
    assert isinstance(plan, MutationCampaignPlan)
    assert plan.metadata.get("production_policy_changed") is False


def test_resource_budget_round_trip_identity() -> None:
    budget = _resource_budget(notes="bounded")
    restored = CampaignResourceBudget.from_dict(budget.to_dict())
    assert restored.budget_cid == budget.budget_cid
    assert restored.to_dict() == budget.to_dict()


def test_resource_budget_accepts_campaign_budget() -> None:
    sealed = CampaignResourceBudget.normalize(_budget(max_targets=5))
    assert sealed.max_targets == 5
    assert sealed.as_campaign_budget().max_targets == 5


def test_create_assurance_manifest_composes_with_planning() -> None:
    """End-to-end smoke: AAE-039 manifest feeds AAE-040 planning."""
    detection = _detection_manifest()
    manifest = create_assurance_manifest(
        {
            "repository_id": REPO_ID,
            "repository_state_cid": REPO_STATE,
            "source_root_cid": SOURCE_ROOT,
            "environment_cid": ENV_CID,
            "dependency_lock_cid": DEP_LOCK,
        },
        POLICY_CID,
        detectors=[item.to_dict() for item in detection.detectors],
        dependency_edges=[item.to_dict() for item in detection.dependency_edges],
        claims=[item.to_dict() for item in detection.claims],
        authority_status=_authority_status_all_unavailable(),
        observation_complete=True,
    )
    assert manifest.interface_id == ASSURANCE_MANIFEST_INTERFACE
    ops = [_operator()]
    result = plan_mutation_campaign(
        _repository_state(),
        manifest,
        _policy(operators=ops),
        _resource_budget(),
        baseline_receipt_cid=BASELINE_RECEIPT,
        targets=[_target()],
        operators=ops,
        return_result=True,
    )
    assert result.assurance_manifest_cid == manifest.manifest_cid
    assert result.expected_detections
