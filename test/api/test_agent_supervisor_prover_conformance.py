from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    DCEC,
    ReviewedPredicate,
    TDFOL,
    TermSort,
    atom,
    variable,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
)
from ipfs_accelerate_py.agent_supervisor.logic_translation_validation import (
    ApproximationDirection,
    LogicForm,
    SemanticDimension,
    SemanticInventory,
    TranslationArtifact,
    TranslationClass,
    TranslationContract,
    TranslationIssueCode,
    inventory_from_reviewed_formula,
    validate_translation,
)
from ipfs_accelerate_py.agent_supervisor.prover_conformance import (
    DEFAULT_CONFORMANCE_FIXTURE_SET,
    DEFAULT_CONFORMANCE_FIXTURE_SET_ID,
    DEFAULT_QUARANTINE_RULES,
    LEGACY_CEC_DEONTIC_API,
    LEGACY_CEC_DCEC_WRAPPER,
    LEGACY_CEC_PROOF_CACHE,
    LEGACY_DCEC_TO_TDFOL_TRANSLATOR,
    LEGACY_TDFOL_PROOF_CACHE,
    LEGACY_TDFOL_TO_FOL_TRANSLATOR,
    REQUIRED_CONFORMANCE_FORMS,
    REQUIRED_CONFORMANCE_KINDS,
    ConformanceObservation,
    ConformanceRunConfig,
    ConformanceStatus,
    ConformanceTestKind,
    ProverConformanceRunner,
    ProverQuarantineRegistry,
    QuarantineReason,
    RouteHealth,
    gate_prover_path,
)


def _contract(
    *,
    source_form: LogicForm = LogicForm.DCEC,
    translation_class: TranslationClass = TranslationClass.EXACT,
    fixture_set_id: str = DEFAULT_CONFORMANCE_FIXTURE_SET_ID,
    abstracted_dimensions: tuple[SemanticDimension, ...] = (),
    required_bounds: tuple[str, ...] = (),
    permitted_assurance: AssuranceLevel | None = None,
    approximation_direction: ApproximationDirection = ApproximationDirection.NONE,
) -> TranslationContract:
    return TranslationContract(
        contract_id=f"{source_form.value}-to-smtlib@1",
        source_identity="baguqeera-source",
        source_form=source_form,
        target_form=LogicForm.SMT_LIB,
        translator_id="supervisor-test-translator",
        translator_version="1",
        translator_identity="sha256:translator",
        semantic_profile_id=f"supervisor-{source_form.value}",
        semantic_profile_version="1",
        translation_class=translation_class,
        fixture_set_id=fixture_set_id,
        abstracted_dimensions=abstracted_dimensions,
        required_bounds=required_bounds,
        permitted_assurance=permitted_assurance,
        approximation_direction=approximation_direction,
    )


def _inventory() -> SemanticInventory:
    return SemanticInventory(
        actors=("agent:worker",),
        times=("time:7",),
        quantifiers=("forall:task",),
        modal_operators=("obligation",),
        bounds=("upper:9",),
        premises=("premise:authorized",),
        predicates=("completed",),
        polarities=("completed:positive",),
        variables=("task:task",),
    )


def _artifact(
    contract: TranslationContract,
    *,
    source: SemanticInventory | None = None,
    target: SemanticInventory | None = None,
    source_identity: str | None = None,
    finite_bounds: dict[str, int] | None = None,
    abstraction_log: tuple[str, ...] = (),
) -> TranslationArtifact:
    inventory = source or _inventory()
    return TranslationArtifact(
        contract_identity=contract.content_id,
        source_identity=source_identity or contract.source_identity,
        target_text="(assert translated_formula)",
        source_inventory=inventory,
        target_inventory=target or inventory,
        fixture_set_id=contract.fixture_set_id,
        finite_bounds=finite_bounds or {},
        abstraction_log=abstraction_log,
    )


@pytest.mark.parametrize(
    ("translation_class", "maximum", "results"),
    [
        (
            TranslationClass.EXACT,
            AssuranceLevel.SOLVER_CHECKED,
            {"proved", "disproved", "satisfiable", "unsatisfiable"},
        ),
        (
            TranslationClass.EQUISATISFIABLE,
            AssuranceLevel.SOLVER_CHECKED,
            {"satisfiable", "unsatisfiable"},
        ),
        (
            TranslationClass.BOUNDED_ABSTRACTION,
            AssuranceLevel.SOLVER_CHECKED,
            {
                "bounded_proved",
                "bounded_disproved",
                "bounded_satisfiable",
                "bounded_unsatisfiable",
            },
        ),
        (
            TranslationClass.CONSERVATIVE_APPROXIMATION,
            AssuranceLevel.CANDIDATE,
            {"candidate", "counterexample"},
        ),
        (
            TranslationClass.HEURISTIC,
            AssuranceLevel.UNVERIFIED,
            {"proposal"},
        ),
    ],
)
def test_translation_classes_define_exact_assurance_and_result_contracts(
    translation_class: TranslationClass,
    maximum: AssuranceLevel,
    results: set[str],
) -> None:
    kwargs = {}
    if translation_class is TranslationClass.BOUNDED_ABSTRACTION:
        kwargs = {
            "required_bounds": ("max_trace_steps",),
            "abstracted_dimensions": (SemanticDimension.TIMES,),
        }
    if translation_class is TranslationClass.CONSERVATIVE_APPROXIMATION:
        kwargs = {
            "abstracted_dimensions": (SemanticDimension.MODAL_OPERATORS,),
            "approximation_direction": ApproximationDirection.OVER,
        }

    contract = _contract(translation_class=translation_class, **kwargs)

    assert contract.maximum_assurance is maximum
    assert set(contract.permitted_results) == results
    assert contract.permits(
        maximum,
        result_class=next(iter(results)),
        bounded=translation_class is TranslationClass.BOUNDED_ABSTRACTION,
    )
    assert not contract.permits(AssuranceLevel.KERNEL_VERIFIED)
    if translation_class is TranslationClass.BOUNDED_ABSTRACTION:
        assert not contract.permits(
            AssuranceLevel.SOLVER_CHECKED,
            result_class="bounded_proved",
            bounded=False,
        )


def test_invalid_translation_contracts_cannot_overclaim() -> None:
    with pytest.raises(ContractValidationError, match="cannot permit"):
        _contract(
            translation_class=TranslationClass.HEURISTIC,
            permitted_assurance=AssuranceLevel.SOLVER_CHECKED,
        )
    with pytest.raises(ContractValidationError, match="finite bounds"):
        _contract(translation_class=TranslationClass.BOUNDED_ABSTRACTION)
    with pytest.raises(ContractValidationError, match="approximation direction"):
        _contract(
            translation_class=TranslationClass.CONSERVATIVE_APPROXIMATION,
            abstracted_dimensions=(SemanticDimension.MODAL_OPERATORS,),
        )
    with pytest.raises(ContractValidationError, match="cannot declare"):
        _contract(
            translation_class=TranslationClass.EXACT,
            abstracted_dimensions=(SemanticDimension.TIMES,),
        )


def test_reviewed_ast_inventory_captures_agents_times_modalities_bounds_variables_and_premises() -> None:
    ready = atom(
        ReviewedPredicate.TASK_READY,
        variable(TermSort.TASK, "task"),
    )
    formula = DCEC.obligation("worker-7", ready, 3)
    bounded = TDFOL.liveness(formula, upper_bound=9, lower_bound=2)

    inventory = inventory_from_reviewed_formula(
        bounded, premise_ids=("premise:authority", "premise:lease")
    )

    assert any("worker-7" in value for value in inventory.actors)
    assert any("=3" in value for value in inventory.times)
    assert any("obligation" in value for value in inventory.modal_operators)
    assert any("liveness" in value for value in inventory.modal_operators)
    assert any("upper=9" in value for value in inventory.bounds)
    assert any("task" in value for value in inventory.variables)
    assert any("finite:task:task" in value for value in inventory.quantifiers)
    assert inventory.premises == ("premise:authority", "premise:lease")
    assert inventory.metadata["formula_id"] == bounded.formula_id


def test_exact_translation_with_identical_inventories_is_conformant_and_content_addressed() -> None:
    contract = _contract()
    artifact = _artifact(contract)

    result = validate_translation(contract, artifact)

    assert result.conformant
    assert not result.quarantine_required
    assert result.maximum_assurance is AssuranceLevel.SOLVER_CHECKED
    assert result.permits(AssuranceLevel.SOLVER_CHECKED)
    assert json.loads(result.to_json()) == result.to_dict()
    assert result.content_id == validate_translation(contract, artifact).content_id
    assert artifact.target_identity.startswith("sha256:")
    assert TranslationContract.from_dict(contract.to_record()) == contract
    assert TranslationArtifact.from_dict(artifact.to_record()) == artifact
    assert type(result).from_dict(result.to_record()) == result


@pytest.mark.parametrize(
    ("dimension", "issue_code"),
    [
        (SemanticDimension.ACTORS, TranslationIssueCode.DROPPED_ACTOR),
        (SemanticDimension.TIMES, TranslationIssueCode.DROPPED_TIME),
        (SemanticDimension.QUANTIFIERS, TranslationIssueCode.DROPPED_QUANTIFIER),
        (
            SemanticDimension.MODAL_OPERATORS,
            TranslationIssueCode.DROPPED_MODAL_OPERATOR,
        ),
        (SemanticDimension.BOUNDS, TranslationIssueCode.DROPPED_BOUND),
        (SemanticDimension.PREMISES, TranslationIssueCode.DROPPED_PREMISE),
        (SemanticDimension.PREDICATES, TranslationIssueCode.DROPPED_PREDICATE),
        (SemanticDimension.POLARITIES, TranslationIssueCode.CHANGED_POLARITY),
        (SemanticDimension.VARIABLES, TranslationIssueCode.DROPPED_VARIABLE),
    ],
)
def test_each_semantic_loss_quarantines_and_prevents_silent_promotion(
    dimension: SemanticDimension,
    issue_code: TranslationIssueCode,
) -> None:
    contract = _contract()
    target_payload = _inventory().to_dict()
    target_payload[dimension.value] = []
    target = SemanticInventory.from_dict(target_payload)

    result = validate_translation(contract, _artifact(contract, target=target))

    assert not result.conformant
    assert result.quarantine_required
    assert result.maximum_assurance is AssuranceLevel.UNVERIFIED
    assert not result.promotion_allowed
    assert not result.permits(AssuranceLevel.CANDIDATE)
    assert [issue.code for issue in result.issues] == [issue_code]
    assert result.issues[0].missing == _inventory().values(dimension)


def test_bounded_declared_abstraction_is_visible_and_requires_log_and_bounds() -> None:
    contract = _contract(
        translation_class=TranslationClass.BOUNDED_ABSTRACTION,
        abstracted_dimensions=(SemanticDimension.TIMES,),
        required_bounds=("max_trace_steps", "quantifier_domain"),
    )
    target_payload = _inventory().to_dict()
    target_payload["times"] = []
    target = SemanticInventory.from_dict(target_payload)

    valid = validate_translation(
        contract,
        _artifact(
            contract,
            target=target,
            finite_bounds={"max_trace_steps": 9, "quantifier_domain": 3},
            abstraction_log=("times mapped to finite trace steps 0..9",),
        ),
    )
    missing_log = validate_translation(
        contract,
        _artifact(
            contract,
            target=target,
            finite_bounds={"max_trace_steps": 9, "quantifier_domain": 3},
        ),
    )
    missing_bound = validate_translation(
        contract,
        _artifact(
            contract,
            target=target,
            finite_bounds={"max_trace_steps": 9},
            abstraction_log=("times mapped to finite trace steps 0..9",),
        ),
    )

    assert valid.conformant and valid.bounded
    assert valid.permits(AssuranceLevel.SOLVER_CHECKED, bounded=True)
    assert not valid.permits(AssuranceLevel.SOLVER_CHECKED, bounded=False)
    assert not missing_log.conformant
    assert TranslationIssueCode.DROPPED_TIME in {
        issue.code for issue in missing_log.issues
    }
    assert not missing_bound.conformant
    assert TranslationIssueCode.MISSING_FINITE_BOUNDS in {
        issue.code for issue in missing_bound.issues
    }


def test_contract_fixture_and_target_identity_mismatches_fail_closed() -> None:
    contract = _contract()
    other = replace(_contract(), translator_version="2")
    wrong_contract = replace(
        _artifact(contract), contract_identity=other.content_id
    )
    wrong_fixture = replace(_artifact(contract), fixture_set_id="baguqeera-stale")

    assert not validate_translation(contract, wrong_contract).conformant
    assert not validate_translation(contract, wrong_fixture).conformant
    payload = _artifact(contract).to_dict()
    payload["target_identity"] = "sha256:forged"
    with pytest.raises(ContractValidationError, match="target identity"):
        TranslationArtifact.from_dict(payload)


def test_source_substitution_and_introduced_hidden_premise_are_detected() -> None:
    contract = _contract()
    substituted = validate_translation(
        contract,
        _artifact(contract, source_identity="baguqeera-substituted-theorem"),
    )
    introduced_inventory = replace(
        _inventory(),
        premises=("premise:authorized", "premise:hidden-axiom"),
    )
    introduced = validate_translation(
        contract,
        _artifact(contract, target=introduced_inventory),
    )

    assert TranslationIssueCode.SOURCE_IDENTITY_MISMATCH in {
        issue.code for issue in substituted.issues
    }
    semantic_changes = [
        issue
        for issue in introduced.issues
        if issue.code is TranslationIssueCode.SEMANTIC_CHANGE
    ]
    assert semantic_changes
    assert semantic_changes[0].dimension is SemanticDimension.PREMISES
    assert semantic_changes[0].missing == ("premise:hidden-axiom",)
    assert not introduced.promotion_allowed


def test_default_fixture_set_has_full_cross_product_coverage() -> None:
    fixture_set = DEFAULT_CONFORMANCE_FIXTURE_SET

    assert len(fixture_set.fixtures) == (
        len(REQUIRED_CONFORMANCE_FORMS) * len(REQUIRED_CONFORMANCE_KINDS)
    )
    assert fixture_set.covers(
        REQUIRED_CONFORMANCE_FORMS, REQUIRED_CONFORMANCE_KINDS
    )
    for form in REQUIRED_CONFORMANCE_FORMS:
        assert {fixture.kind for fixture in fixture_set.for_form(form)} == set(
            ConformanceTestKind
        )
    assert {
        fixture.source_form for fixture in fixture_set.fixtures
    } == set(LogicForm)
    assert type(fixture_set).from_dict(fixture_set.to_record()) == fixture_set
    forged = fixture_set.to_record()
    forged["name"] = "substituted"
    with pytest.raises(ContractValidationError, match="identity"):
        type(fixture_set).from_dict(forged)


def _passing_runner_observation(fixture, contract) -> ConformanceObservation:
    artifact = TranslationArtifact(
        contract_identity=contract.content_id,
        source_identity=fixture.content_id,
        target_text=f"(assert {fixture.fixture_id})",
        source_inventory=fixture.source_inventory,
        target_inventory=fixture.source_inventory,
        fixture_set_id=contract.fixture_set_id,
    )
    return ConformanceObservation(
        artifact=artifact,
        round_trip_inventory=fixture.source_inventory,
        candidate_outcome=fixture.expected_outcome,
        oracle_outcome=fixture.expected_outcome,
        relation_preserved=True,
        mutation_detected=True,
        rejected=True,
        metadata={"deterministic": True},
    )


def _passing_dcec_report(path_id: str = LEGACY_CEC_DCEC_WRAPPER):
    contract = _contract(source_form=LogicForm.DCEC)
    return ProverConformanceRunner(DEFAULT_CONFORMANCE_FIXTURE_SET).run(
        prover_id="dcec",
        path_id=path_id,
        contract=contract,
        fixture_runner=_passing_runner_observation,
    )


def test_runner_executes_all_five_methods_and_emits_bound_passing_receipt() -> None:
    report = _passing_dcec_report()

    assert report.passed
    assert report.complete
    assert report.fixture_set_id == DEFAULT_CONFORMANCE_FIXTURE_SET_ID
    assert report.permitted_assurance is AssuranceLevel.SOLVER_CHECKED
    assert len(report.cases) == len(ConformanceTestKind)
    assert {case.kind for case in report.cases} == set(ConformanceTestKind)
    assert set(report.statuses) == {status.value for status in ConformanceStatus}
    assert report.statuses["passed"] == 5
    assert report.covers((LogicForm.DCEC,), tuple(ConformanceTestKind))
    assert report.report_id.startswith("baguq")
    assert type(report).from_dict(report.to_record()) == report


def test_runner_fails_mutation_that_does_not_change_semantics() -> None:
    contract = _contract()

    def runner(fixture, contract):
        observation = _passing_runner_observation(fixture, contract)
        if fixture.kind is ConformanceTestKind.MUTATION:
            return replace(observation, mutation_detected=False)
        return observation

    report = ProverConformanceRunner(DEFAULT_CONFORMANCE_FIXTURE_SET).run(
        prover_id="dcec",
        path_id="test-path",
        contract=contract,
        fixture_runner=runner,
    )

    assert not report.passed
    assert report.complete
    assert report.permitted_assurance is AssuranceLevel.UNVERIFIED
    mutation = next(
        case for case in report.cases if case.kind is ConformanceTestKind.MUTATION
    )
    assert mutation.status is ConformanceStatus.FAILED
    assert "escaped" in mutation.reason


def test_runner_rejects_source_substitution_and_incomplete_case_budget() -> None:
    contract = _contract()

    def substituted(fixture, contract):
        result = _passing_runner_observation(fixture, contract)
        return replace(
            result,
            artifact=replace(result.artifact, source_identity="baguqeera-other"),
        )

    substituted_report = ProverConformanceRunner(
        DEFAULT_CONFORMANCE_FIXTURE_SET
    ).run(
        prover_id="dcec",
        path_id="test-path",
        contract=contract,
        fixture_runner=substituted,
    )
    bounded_report = ProverConformanceRunner(
        DEFAULT_CONFORMANCE_FIXTURE_SET,
        config=ConformanceRunConfig(max_cases=2),
    ).run(
        prover_id="dcec",
        path_id="test-path",
        contract=contract,
        fixture_runner=_passing_runner_observation,
    )

    assert not substituted_report.passed
    assert all(case.status is ConformanceStatus.FAILED for case in substituted_report.cases)
    assert all("different source" in case.reason for case in substituted_report.cases)
    assert not bounded_report.complete
    assert len(bounded_report.cases) == 2
    assert bounded_report.permitted_assurance is AssuranceLevel.UNVERIFIED


def test_runner_errors_and_stale_fixture_set_never_promote() -> None:
    stale_contract = _contract(fixture_set_id="baguqeera-stale")

    def broken(_fixture, _contract):
        raise RuntimeError("adapter crashed")

    error_report = ProverConformanceRunner(
        DEFAULT_CONFORMANCE_FIXTURE_SET
    ).run(
        prover_id="dcec",
        path_id="test-path",
        contract=_contract(),
        fixture_runner=broken,
    )
    stale_report = ProverConformanceRunner(
        DEFAULT_CONFORMANCE_FIXTURE_SET
    ).run(
        prover_id="dcec",
        path_id="test-path",
        contract=stale_contract,
        fixture_runner=_passing_runner_observation,
    )

    assert not error_report.passed
    assert all(case.status is ConformanceStatus.ERROR for case in error_report.cases)
    assert error_report.permitted_assurance is AssuranceLevel.UNVERIFIED
    assert not stale_report.complete
    assert stale_report.permitted_assurance is AssuranceLevel.UNVERIFIED


def test_in_process_fixture_callback_is_deadline_bounded() -> None:
    def hanging(_fixture, _contract):
        time.sleep(0.05)
        return {}

    report = ProverConformanceRunner(
        DEFAULT_CONFORMANCE_FIXTURE_SET,
        config=ConformanceRunConfig(timeout_seconds=0.01),
    ).run(
        prover_id="dcec",
        path_id=LEGACY_CEC_DCEC_WRAPPER,
        contract=_contract(),
        fixture_runner=hanging,
    )

    assert not report.complete
    assert not report.passed
    assert report.statuses["timed_out"] == len(ConformanceTestKind)
    decision = gate_prover_path(LEGACY_CEC_DCEC_WRAPPER, report)
    assert decision.health is RouteHealth.QUARANTINED
    assert QuarantineReason.TIMEOUT in decision.reasons


def test_known_cec_api_and_timing_sensitive_cache_paths_start_degraded() -> None:
    registry = ProverQuarantineRegistry()
    expected = {
        LEGACY_CEC_DCEC_WRAPPER: QuarantineReason.API_DRIFT,
        LEGACY_CEC_DEONTIC_API: QuarantineReason.API_DRIFT,
        LEGACY_CEC_PROOF_CACHE: QuarantineReason.TIMING_SENSITIVE_CACHE,
        LEGACY_TDFOL_PROOF_CACHE: QuarantineReason.TIMING_SENSITIVE_CACHE,
        LEGACY_DCEC_TO_TDFOL_TRANSLATOR: QuarantineReason.API_DRIFT,
        LEGACY_TDFOL_TO_FOL_TRANSLATOR: (
            QuarantineReason.TRANSLATION_NONCONFORMANCE
        ),
    }

    assert len(DEFAULT_QUARANTINE_RULES) == len(expected)
    for path_id, reason in expected.items():
        decision = registry.assess(
            path_id, authoritative_for=("temporal_deontic_plan",)
        )
        assert decision.health is RouteHealth.DEGRADED
        assert not decision.promotion_allowed
        assert decision.maximum_assurance is AssuranceLevel.UNVERIFIED
        assert decision.reasons == (reason,)
        assert decision.retained_authorities == ()


def test_only_complete_semantic_receipt_releases_legacy_path() -> None:
    registry = ProverQuarantineRegistry()
    passed = _passing_dcec_report()

    released = registry.assess(
        LEGACY_CEC_DCEC_WRAPPER,
        passed,
        authoritative_for=("temporal_deontic_plan",),
    )

    assert released.health is RouteHealth.CONFORMANT
    assert released.promotion_allowed
    assert released.maximum_assurance is AssuranceLevel.SOLVER_CHECKED
    assert released.retained_authorities == ("temporal_deontic_plan",)
    assert released.report_id == passed.report_id
    assert type(released).from_dict(released.to_record()) == released

    def bad_runner(fixture, contract):
        result = _passing_runner_observation(fixture, contract)
        return (
            replace(result, mutation_detected=False)
            if fixture.kind is ConformanceTestKind.MUTATION
            else result
        )

    failed = ProverConformanceRunner(DEFAULT_CONFORMANCE_FIXTURE_SET).run(
        prover_id="dcec",
        path_id=LEGACY_CEC_DCEC_WRAPPER,
        contract=_contract(),
        fixture_runner=bad_runner,
    )
    quarantined = registry.assess(
        LEGACY_CEC_DCEC_WRAPPER,
        failed,
        authoritative_for=("temporal_deontic_plan",),
    )
    assert quarantined.health is RouteHealth.QUARANTINED
    assert not quarantined.promotion_allowed
    assert quarantined.retained_authorities == ()
    assert QuarantineReason.TRANSLATION_NONCONFORMANCE in quarantined.reasons


def test_receipt_for_wrong_path_or_wrong_logic_form_cannot_release_rule() -> None:
    report = _passing_dcec_report(path_id="another-path")

    wrong_path = gate_prover_path(LEGACY_CEC_DCEC_WRAPPER, report)

    assert wrong_path.health is RouteHealth.QUARANTINED
    assert wrong_path.reasons == (QuarantineReason.MALFORMED_RESULT,)

    # A complete DCEC report does not satisfy the TDFOL cache rule.
    same_path_report = replace(report, path_id=LEGACY_TDFOL_PROOF_CACHE)
    wrong_form = gate_prover_path(LEGACY_TDFOL_PROOF_CACHE, same_path_report)
    assert wrong_form.health is RouteHealth.QUARANTINED
    assert not wrong_form.promotion_allowed


def test_unassessed_new_path_and_nonconformant_artifact_fail_closed() -> None:
    unknown = gate_prover_path(
        "third.party.unreviewed",
        authoritative_for=("all_properties",),
    )

    assert unknown.health is RouteHealth.UNASSESSED
    assert not unknown.promotion_allowed
    assert unknown.maximum_assurance is AssuranceLevel.UNVERIFIED
    assert unknown.retained_authorities == ()

    contract = _contract()
    source = _inventory()
    target = replace(source, premises=())
    validation = validate_translation(
        contract, _artifact(contract, source=source, target=target)
    )
    assert not validation.conformant
    assert TranslationIssueCode.DROPPED_PREMISE in {
        item.code for item in validation.issues
    }
