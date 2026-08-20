from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ProcedureContractError,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.transition_model import (
    CalibrationAdmissionError,
    CalibrationDimension,
    CalibrationDisposition,
    CalibrationPolicy,
    ConfidenceClass,
    ObservationClass,
    PredictionCalibration,
    PredictionUse,
    TransitionClass,
    TransitionMeasurements,
    TransitionModel,
    TransitionModelError,
    TransitionModelState,
    TransitionObservation,
    TransitionPrediction,
    TransitionTerminalStatus,
    calibrate_prediction,
    calibrate_transition_model,
    prediction_may_discharge,
)


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo",
        repository_commit="commit-1",
        tree_id="tree-1",
        objective_id="PCPC-G000",
        task_id="PCPC-008",
        contract_revision="contract-1",
        policy_revision="policy-1",
        environment_id="environment-1",
    )


def _measurements(**changes: object) -> TransitionMeasurements:
    values = {
        "changed_files": 1,
        "changed_symbols": 1,
        "effects": 1,
        "tests": 2,
        "proofs": 1,
        "duration_ms": 200,
        "tokens": 50,
        "provider_cost_micros": 20,
        "merge_conflicts": 0,
        "terminal_status": TransitionTerminalStatus.SUCCEEDED,
    }
    values.update(changes)
    return TransitionMeasurements(**values)


def _prediction(**changes: object) -> TransitionPrediction:
    values = {
        "bindings": _bindings(),
        "model_id": "model-source-edit",
        "model_revision": 0,
        "transition_class": TransitionClass.SOURCE_EDIT,
        "confidence_class": ConfidenceClass.EXACT,
        "source_state_id": "world-before",
        "predicted_state_id": "world-after",
        "predicted_delta_id": "delta-predicted",
        "measurements": _measurements(),
        "changed_file_ids": ("src/a.py",),
        "changed_symbol_ids": ("src.a:run",),
        "effect_ids": ("effect-write-a",),
        "test_ids": ("test-a", "test-b"),
        "proof_ids": ("proof-a",),
        "merge_conflict_ids": (),
        "validation_dependency_ids": ("validation-policy-1",),
    }
    values.update(changes)
    return TransitionPrediction(**values)


def _observation(**changes: object) -> TransitionObservation:
    values = {
        "bindings": _bindings(),
        "transition_class": TransitionClass.SOURCE_EDIT,
        "source_state_id": "world-before",
        "observed_state_id": "world-after",
        "world_state_delta_id": "delta-observed",
        "measurements": _measurements(),
        "producer_id": "independent-transition-observer",
        "observation_class": ObservationClass.ADMITTED_EXTERNAL,
        "admission_receipt_id": "observation-admission-1",
        "changed_file_ids": ("src/a.py",),
        "changed_symbol_ids": ("src.a:run",),
        "effect_ids": ("effect-write-a",),
        "test_ids": ("test-a", "test-b"),
        "proof_ids": ("proof-a",),
        "merge_conflict_ids": (),
        "evidence_ids": ("test-receipt", "proof-receipt"),
    }
    values.update(changes)
    return TransitionObservation(**values)


def _model(**changes: object) -> TransitionModel:
    values = {
        "bindings": _bindings(),
        "model_id": "model-source-edit",
        "revision": 0,
        "transition_class": TransitionClass.SOURCE_EDIT,
        "confidence_class": ConfidenceClass.EXACT,
        "source_episode_ids": ("episode-1", "episode-2"),
        "operation_catalog_revision": "operations-1",
        "effect_policy_revision": "effects-1",
        "verification_policy_revision": "verification-1",
    }
    values.update(changes)
    return TransitionModel(**values)


def _walk_has_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_walk_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_walk_has_float(item) for item in value)
    return False


def test_contract_round_trips_and_unknown_fields_are_rejected() -> None:
    prediction = _prediction()
    observation = _observation()
    model = _model()

    assert TransitionPrediction.from_dict(prediction.to_dict()) == prediction
    assert TransitionObservation.from_dict(observation.to_dict()) == observation
    assert TransitionModel.from_dict(model.to_dict()) == model

    with pytest.raises(ProcedureContractError):
        TransitionPrediction.from_dict({**prediction.to_dict(), "callback": "bad"})


def test_exact_and_admitted_conservative_discharge_only_planning() -> None:
    exact = _prediction()
    assert prediction_may_discharge(exact, PredictionUse.DETERMINISTIC_PLANNING)

    for forbidden in (
        PredictionUse.AUTHORITY,
        PredictionUse.POSTCONDITION,
        PredictionUse.PROOF,
        PredictionUse.COMPLETION,
        PredictionUse.VALIDATION_SUPPRESSION,
        PredictionUse.HUMAN_REVIEW_SUPPRESSION,
    ):
        assert not prediction_may_discharge(exact, forbidden)

    conservative = _prediction(
        confidence_class=ConfidenceClass.CONSERVATIVE,
        conservative_evidence_id="conservative-admission-1",
    )
    assert not prediction_may_discharge(conservative, PredictionUse.DETERMINISTIC_PLANNING)
    assert prediction_may_discharge(
        conservative,
        PredictionUse.DETERMINISTIC_PLANNING,
        admitted_conservative_evidence_ids=("conservative-admission-1",),
    )


@pytest.mark.parametrize("confidence", (ConfidenceClass.EMPIRICAL, ConfidenceClass.HEURISTIC))
def test_empirical_and_heuristic_only_influence_cost_and_priority(
    confidence: ConfidenceClass,
) -> None:
    prediction = _prediction(confidence_class=confidence)
    assert prediction_may_discharge(prediction, PredictionUse.COST)
    assert prediction_may_discharge(prediction, PredictionUse.PRIORITY)
    assert not prediction_may_discharge(prediction, PredictionUse.DETERMINISTIC_PLANNING)
    assert not prediction_may_discharge(prediction, PredictionUse.COMPLETION)


def test_calibration_compares_all_required_dimensions_as_integers() -> None:
    calibration = calibrate_prediction(
        _prediction(),
        _observation(),
        admitted_observation_receipt_ids=("observation-admission-1",),
    )

    assert {item.dimension for item in calibration.comparisons} == set(CalibrationDimension)
    assert len(calibration.comparisons) == 10
    assert all(type(item.predicted) is int for item in calibration.comparisons)
    assert all(type(item.observed) is int for item in calibration.comparisons)
    assert calibration.disposition is CalibrationDisposition.MATCHED
    assert calibration.total_absolute_error == 0
    assert not _walk_has_float(calibration.to_dict())
    assert PredictionCalibration.from_dict(calibration.to_dict()) == calibration


def test_unadmitted_or_simulated_observation_cannot_update_model() -> None:
    model = _model()
    candidate = _observation(
        observation_class=ObservationClass.CANDIDATE,
        admission_receipt_id="",
    )
    unchanged, calibration = calibrate_transition_model(
        model,
        _prediction(),
        candidate,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    assert unchanged is model
    assert calibration.disposition is CalibrationDisposition.UNADMITTED

    simulated = _observation(
        observation_class=ObservationClass.SIMULATED,
        admission_receipt_id="",
    )
    unchanged, simulated_calibration = calibrate_transition_model(
        model,
        _prediction(),
        simulated,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    assert unchanged is model
    assert simulated_calibration.observation_admitted is False


def test_exact_structural_drift_automatically_invalidates_model() -> None:
    observation = _observation(
        measurements=_measurements(changed_files=2),
        changed_file_ids=("src/a.py", "src/escaped.py"),
    )
    updated, calibration = calibrate_transition_model(
        _model(),
        _prediction(),
        observation,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )

    assert calibration.drift_detected
    assert CalibrationDimension.FILES in calibration.critical_dimensions
    assert updated.state is TransitionModelState.INVALIDATED
    assert updated.confidence_class is ConfidenceClass.UNKNOWN
    assert updated.drift_count == 1


def test_noncritical_drift_demotes_then_repeated_drift_invalidates() -> None:
    policy = CalibrationPolicy(
        duration_tolerance_ms=0,
        invalidate_after_consecutive_drift=2,
        invalidate_after_total_drift=3,
    )
    model = _model(confidence_class=ConfidenceClass.CONSERVATIVE)
    first_prediction = _prediction(confidence_class=ConfidenceClass.CONSERVATIVE)
    slow = _observation(measurements=_measurements(duration_ms=201))
    demoted, first_calibration = calibrate_transition_model(
        model,
        first_prediction,
        slow,
        policy=policy,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )

    assert first_calibration.drift_dimensions == (CalibrationDimension.DURATION,)
    assert not first_calibration.critical_drift
    assert demoted.state is TransitionModelState.DEMOTED
    assert demoted.confidence_class is ConfidenceClass.EMPIRICAL

    second_prediction = replace(
        first_prediction,
        model_revision=demoted.revision,
        confidence_class=demoted.confidence_class,
    )
    invalidated, _ = calibrate_transition_model(
        demoted,
        second_prediction,
        slow,
        policy=policy,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    assert invalidated.state is TransitionModelState.INVALIDATED
    assert invalidated.confidence_class is ConfidenceClass.UNKNOWN


def test_same_counts_but_different_identities_are_drift() -> None:
    observation = _observation(changed_file_ids=("src/b.py",))
    calibration = calibrate_prediction(
        _prediction(),
        observation,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    assert calibration.comparison_for(CalibrationDimension.FILES).absolute_error == 0
    assert CalibrationDimension.FILES in calibration.identity_mismatch_dimensions
    assert calibration.critical_drift


def test_terminal_status_is_a_closed_integer_comparison() -> None:
    observation = _observation(
        measurements=_measurements(terminal_status=TransitionTerminalStatus.FAILED)
    )
    calibration = calibrate_prediction(
        _prediction(),
        observation,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    terminal = calibration.comparison_for(CalibrationDimension.TERMINAL)
    assert terminal.predicted == 0
    assert terminal.observed == 1
    assert terminal.absolute_error == 1
    assert calibration.critical_drift


def test_deserialized_calibration_still_requires_external_admission_to_apply() -> None:
    model = _model()
    calibration = calibrate_prediction(
        _prediction(),
        _observation(),
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    replayed = PredictionCalibration.from_dict(calibration.to_dict())

    with pytest.raises(CalibrationAdmissionError):
        model.apply_calibration(replayed)
    updated = model.apply_calibration(replayed, admitted_calibration_ids=(replayed.content_id,))
    assert updated.revision == 1
    assert (
        updated.apply_calibration(replayed, admitted_calibration_ids=(replayed.content_id,))
        is updated
    )


def test_transition_class_or_binding_mismatch_is_rejected_or_invalidated() -> None:
    with pytest.raises(TransitionModelError):
        calibrate_prediction(
            _prediction(),
            _observation(bindings=replace(_bindings(), tree_id="other-tree")),
        )

    mismatched_class = _observation(transition_class=TransitionClass.MERGE)
    updated, calibration = calibrate_transition_model(
        _model(),
        _prediction(),
        mismatched_class,
        admitted_observation_receipt_ids=("observation-admission-1",),
    )
    assert calibration.transition_class_match is False
    assert updated.state is TransitionModelState.INVALIDATED


def test_measurements_reject_floats_nonfinite_values_and_unknown_terminal() -> None:
    with pytest.raises(ProcedureContractError):
        _measurements(duration_ms=1.5)
    with pytest.raises(ProcedureContractError):
        _measurements(provider_cost_micros=float("inf"))
    with pytest.raises(ProcedureContractError):
        _measurements(terminal_status="claimed_complete")
