"""Source-contract tests only; all owners/reviews/proofs below are unit fixtures.

No fixture establishes production review, Lean admission or execution authority.
Existing prompt-admission fixtures supply the complete native join structure so
these tests can attack provenance, replay and scope without inventing another
planning/IR compiler in the implementation under test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis import planning_admission_materials as api
from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import (
    PlanningAdmissionRequestFactory,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission_service import (
    PlanAdmissionServiceRequest,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import (
    PromptPlanAdmissionPolicy,
    PromptPlanAdmissionRequest,
    admit_prompt_plan,
)
from test.api.test_agent_supervisor_prompt_plan_admission import (
    _graph_fixture,
    _ir_request,
)


class FixtureOwnerVerifier:
    """Explicit unit-test provenance stand-in, never a production verifier."""

    def __init__(self, *, accepted=True):
        self.accepted = accepted
        self.calls = []

    def verify_owner_materials(self, context, envelope):
        self.calls.append((context.context_id, envelope.envelope_id))
        return api.OwnerMaterialVerification(
            accepted=self.accepted,
            context_id=context.context_id,
            envelope_id=envelope.envelope_id,
            owner_id=envelope.owner_id,
            verifier_id="unit-only:independent-verifier",
            verified_reference_ids=envelope.provenance_reference_ids,
        )


@pytest.fixture
def owner_case():
    workflow, scan, graph = _graph_fixture()
    policy = PromptPlanAdmissionPolicy()
    run = api.PlanningAdmissionRunBindings(
        "unit-only:execution-plan", tuple(sorted({policy.policy_id, workflow.policy_root, *graph.policy_roots})),
    )
    captured = []

    def resolver(context):
        native = _ir_request(graph, workflow, scan.dirty_worktree_root)
        materials = PlanAdmissionServiceRequest(
            candidate_plan=native.candidate_plan,
            repository_tree_id=native.repository_tree_id,
            formal_plan_id=context.formal.plan_id,
            formal_source_identity=context.formal.source_identity,
            evidence_bundle_id=context.evidence_bundle_id,
            execution_plan_id=run.execution_plan_id,
            policy_ids=run.policy_ids,
            intent_request=native.intent_request,
            legal_results=native.legal_results,
            security_policy=native.security_policy,
            security_requests=native.security_requests,
            action_bindings=native.action_bindings,
            authority=native.authority,
            root_bindings=native.root_bindings,
            program_dependencies=native.program_dependencies,
            assumptions=native.assumptions,
            proof_results=native.proof_results,
            validation_requirements=native.validation_requirements,
            validation_results=native.validation_results,
            generated_formula_ids=native.generated_formula_ids,
            mandatory_closure=native.mandatory_closure,
            graph_complete=native.graph_complete,
            intent_effects=tuple(dict(request.expected_effect) for request in native.security_requests),
            code_effects=tuple(dict(request.expected_effect) for request in native.security_requests),
        )
        envelope = api.OwnerAdmissionMaterialEnvelope(
            materials, "unit-only:owner", ("unit-only:owner-custody", "unit-only:source-and-grant-review"),
        )
        captured.append((context, envelope))
        return envelope

    return workflow, scan, graph, policy, run, resolver, captured


def _build(case, *, resolver=None, verifier=None, run=None, policy=None):
    workflow, scan, graph, original_policy, original_run, original_resolver, _ = case
    builder = api.OwnerBackedPlanningAdmissionBuilder(
        material_resolver=resolver or original_resolver,
        owner_verifier=verifier or FixtureOwnerVerifier(),
        run_bindings=run or original_run,
        policy=policy or original_policy,
    )
    return builder(request=workflow, scan=scan, graph=graph,
                   repository_tree_id=scan.dirty_worktree_root)


def _mutating_resolver(case, **changes):
    def resolver(context):
        envelope = case[5](context)
        return replace(envelope, materials=replace(envelope.materials, **changes))
    return resolver


def _codes(error):
    return {issue.code for issue in error.value.issues}


def test_explicit_owner_interfaces_required(owner_case):
    with pytest.raises(TypeError, match="owner_verifier"):
        api.OwnerBackedPlanningAdmissionBuilder(
            material_resolver=owner_case[5], owner_verifier=None, run_bindings=owner_case[4],
        )
    with pytest.raises(TypeError, match="material_resolver"):
        api.OwnerBackedPlanningAdmissionBuilder(
            material_resolver=None, owner_verifier=FixtureOwnerVerifier(), run_bindings=owner_case[4],
        )


def test_unit_owner_materials_preserve_exact_request_and_use_native_service(owner_case, monkeypatch):
    admitted = []
    native_admit = api.PlanAdmissionService.admit
    def observed_admission(service, materials):
        receipt = native_admit(service, materials)
        admitted.append(receipt)
        return receipt
    monkeypatch.setattr(api.PlanAdmissionService, "admit", observed_admission)
    verifier = FixtureOwnerVerifier()
    result = _build(owner_case, verifier=verifier)
    context, envelope = owner_case[6][0]
    assert isinstance(result, PromptPlanAdmissionRequest)
    assert result.workflow_request is owner_case[0]
    assert result.scan_receipt is owner_case[1]
    assert result.graph is owner_case[2]
    assert result.policy is owner_case[3]
    assert verifier.calls == [(context.context_id, envelope.envelope_id)]
    assert result.ir_request.proof_results == envelope.materials.proof_results
    assert result.ir_request.validation_results == envelope.materials.validation_results
    assert result.ir_request.authority == envelope.materials.authority
    assert len(admitted) == 1 and admitted[0].request_id == envelope.materials.request_id
    assert not hasattr(result, "admitted")


def test_full_native_service_receipt_remains_bound_to_this_call(owner_case):
    workflow, scan, graph, policy, run, resolver, _ = owner_case
    builder = api.OwnerBackedPlanningAdmissionBuilder(
        material_resolver=resolver, owner_verifier=FixtureOwnerVerifier(), run_bindings=run, policy=policy,
    )
    result = builder.build_with_receipt(
        request=workflow, scan=scan, graph=graph, repository_tree_id=scan.dirty_worktree_root,
    )
    context, envelope = owner_case[6][0]
    assert result.context_id == context.context_id
    assert result.envelope_id == envelope.envelope_id
    assert result.service_receipt.ir_request_id == result.request.ir_request.request_id
    assert result.service_receipt.evidence_bundle_id == context.evidence_bundle_id
    assert result.service_receipt.execution_plan_id == run.execution_plan_id
    assert result.service_receipt.policy_ids == run.policy_ids
    assert result.service_receipt.admitted  # Native planning fixture only, not Lean.


def test_missing_materials_report_all_independent_domains(owner_case):
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=lambda context: None)
    assert error.value.reason_code == "owner_admission_materials_unavailable"
    assert len(error.value.issues) == 7
    assert {issue.path for issue in error.value.issues} >= {
        "$.materials.intent_request", "$.materials.legal_results", "$.materials.authority",
    }


def test_owner_refusal_preserves_specific_issues(owner_case):
    class Refusal(FixtureOwnerVerifier):
        def verify_owner_materials(self, context, envelope):
            return replace(super().verify_owner_materials(context, envelope),
                           accepted=False, issues=(api.AdmissionMaterialIssue(
                               "missing_authenticated_review", "$.owner.review", "external reviewer custody is unavailable",
                           ),))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, verifier=Refusal())
    assert _codes(error) == {"missing_authenticated_review"}


@pytest.mark.parametrize("claimed", [True, {"accepted": True, "reviewed": True}])
def test_boolean_or_self_asserted_review_is_not_a_verifier_receipt(owner_case, claimed):
    class SelfAssertion:
        def verify_owner_materials(self, context, envelope):
            return claimed
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, verifier=SelfAssertion())
    assert _codes(error) == {"invalid_owner_verification"}


@pytest.mark.parametrize("field,value", [
    ("context_id", "unit-only:other-context"),
    ("envelope_id", "unit-only:other-envelope"),
    ("owner_id", "unit-only:other-owner"),
    ("verified_reference_ids", ("unit-only:one-reference",)),
])
def test_verifier_replay_and_incomplete_reference_coverage_rejected(owner_case, field, value):
    class Replayed(FixtureOwnerVerifier):
        def verify_owner_materials(self, context, envelope):
            return replace(super().verify_owner_materials(context, envelope), **{field: value})
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, verifier=Replayed())
    assert _codes(error) == {"owner_verification_replay"}


def test_owner_material_mutation_during_verification_rejected(owner_case):
    class Mutating(FixtureOwnerVerifier):
        def verify_owner_materials(self, context, envelope):
            decision = super().verify_owner_materials(context, envelope)
            object.__setattr__(envelope, "materials", replace(envelope.materials, execution_plan_id="unit-only:mutated"))
            return decision
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, verifier=Mutating())
    assert _codes(error) == {"materials_changed_during_verification"}


@pytest.mark.parametrize("field,value", [
    ("repository_tree_id", "unit-only:stale-tree"),
    ("formal_plan_id", "unit-only:other-plan"),
    ("formal_source_identity", "unit-only:other-source"),
    ("evidence_bundle_id", "unit-only:other-evidence"),
    ("execution_plan_id", "unit-only:other-execution"),
    ("policy_ids", ("unit-only:other-policy",)),
])
def test_owner_material_identity_mismatches_rejected(owner_case, field, value):
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=_mutating_resolver(owner_case, **{field: value}))
    assert "material_binding_mismatch" in _codes(error)
    assert f"$.materials.{field}" in {issue.path for issue in error.value.issues}


def test_current_tree_rechecked_before_resolving(owner_case):
    workflow, scan, graph, policy, run, _, _ = owner_case
    def should_not_resolve(context):
        pytest.fail("stale source must stop before owner resolution")
    builder = api.OwnerBackedPlanningAdmissionBuilder(
        material_resolver=should_not_resolve, owner_verifier=FixtureOwnerVerifier(), run_bindings=run, policy=policy,
    )
    with pytest.raises(api.AdmissionMaterialsError) as error:
        builder(request=workflow, scan=scan, graph=graph, repository_tree_id="unit-only:stale")
    assert _codes(error) == {"stale_repository_tree"}


def test_stale_observed_root_rejected_even_with_current_expected_root(owner_case):
    def resolver(context):
        envelope = owner_case[5](context)
        roots = tuple(replace(root, observed="unit-only:stale-root") if root.kind == "legal" else root
                      for root in envelope.materials.root_bindings)
        return replace(envelope, materials=replace(envelope.materials, root_bindings=roots))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=resolver)
    assert _codes(error) == {"stale_or_untrusted_root"}


@pytest.mark.parametrize("field,value", [
    ("action_bindings", ()), ("program_dependencies", ()),
    ("proof_results", ()), ("validation_results", ()),
    ("validation_requirements", ()),
    ("generated_formula_ids", ("unit-only:other-formula",)),
])
def test_native_action_proof_and_validation_joins_retained(owner_case, field, value):
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=_mutating_resolver(owner_case, **{field: value}))
    assert "ir_binding_mismatch" in _codes(error) or "missing_proof" in _codes(error)


def test_exact_security_effect_coverage_not_weakened(owner_case):
    def resolver(context):
        envelope = owner_case[5](context)
        requests = envelope.materials.security_requests
        altered = replace(requests[0], expected_effect={"operation": "delete", "target": "unit-only:foreign"})
        return replace(envelope, materials=replace(envelope.materials, security_requests=(altered, *requests[1:])))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=resolver)
    assert "ir_binding_mismatch" in _codes(error)


def test_missing_grants_are_not_minted_and_native_service_refuses(owner_case):
    def resolver(context):
        envelope = owner_case[5](context)
        authority = replace(envelope.materials.authority, grant_source_ids=())
        return replace(envelope, materials=replace(envelope.materials, authority=authority))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=resolver)
    assert "unknown_mandatory_authority" in _codes(error)
    assert error.value.service_receipt is not None
    assert not error.value.service_receipt.admitted
    assert owner_case[6][-1][1].materials.authority.grant_source_ids != ()


@pytest.mark.parametrize("stream", ["intent_effects", "code_effects"])
def test_extra_intent_and_generated_code_effect_streams_cannot_skip_security(owner_case, stream):
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=_mutating_resolver(owner_case, **{
            stream: ({"operation": "delete", "target": "unit-only:uncovered-effect"},),
        }))
    assert "security_stream_gap" in _codes(error)
    assert error.value.service_receipt is not None
    assert not error.value.service_receipt.admitted


@pytest.mark.parametrize("stream", ["intent_effects", "code_effects"])
def test_missing_independent_effect_stream_does_not_fall_back_to_candidate(owner_case, stream):
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=_mutating_resolver(owner_case, **{stream: ()}))
    assert "missing_independent_effect_stream" in _codes(error)


def test_failed_validation_state_still_rejected_by_native_service(owner_case):
    from ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler import ValidationStatus
    def resolver(context):
        envelope = owner_case[5](context)
        results = tuple(replace(result, status=ValidationStatus.FAILED)
                        for result in envelope.materials.validation_results)
        return replace(envelope, materials=replace(envelope.materials, validation_results=results))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=resolver)
    assert "validation_failed" in _codes(error)


def test_normal_prompt_gate_is_still_required_after_material_service(owner_case):
    result = _build(owner_case)
    strict = replace(result.policy, protected_paths=("pkg",))
    outcome = admit_prompt_plan(replace(result, policy=strict))
    assert not outcome.admitted
    assert outcome.admitted_graph is None


def test_existing_factory_hook_preserves_specific_material_error(owner_case):
    class Analysis:
        last_view = None
    builder = api.OwnerBackedPlanningAdmissionBuilder(
        material_resolver=lambda context: None, owner_verifier=FixtureOwnerVerifier(),
        run_bindings=owner_case[4], policy=owner_case[3],
    )
    factory = PlanningAdmissionRequestFactory(Analysis(), ir_request_builder=builder)
    with pytest.raises(api.AdmissionMaterialsError) as error:
        factory.build(*owner_case[:3])
    assert error.value.reason_code == "owner_admission_materials_unavailable"
    assert "missing_owner_material" in _codes(error)


def test_refusal_can_be_passed_from_resolver_without_losing_scope(owner_case):
    def resolver(context):
        raise api.AdmissionMaterialsError((api.AdmissionMaterialIssue(
            "missing_native_security_owner", "$.materials.security_policy", "no current owner artifact",
        ),))
    with pytest.raises(api.AdmissionMaterialsError) as error:
        _build(owner_case, resolver=resolver)
    assert _codes(error) == {"missing_native_security_owner"}


def test_material_envelope_and_issue_limits_are_closed(owner_case):
    with pytest.raises(ValueError, match="duplicates"):
        api.PlanningAdmissionRunBindings("unit-only:execution", ("unit-only:policy", "unit-only:policy"))
    with pytest.raises(ValueError, match="bounded"):
        api.AdmissionMaterialIssue("code", "$.path", "x" * 1025)
    with pytest.raises(ValueError, match="bounded"):
        api.AdmissionMaterialsError((api.AdmissionMaterialIssue("code", "$.path", "message"),) * 257)
    with pytest.raises(ValueError, match="blockers"):
        api.OwnerMaterialVerification(
            True, "unit-only:context", "unit-only:envelope", "unit-only:owner", "unit-only:verifier",
            ("unit-only:reference",), (api.AdmissionMaterialIssue("code", "$.path", "message"),),
        )
