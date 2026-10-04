"""Exact constant prerequisites; no refit or optimizer convergence claim."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import math

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_ranker_step_certificate as api
from test.api.test_terminal_codebase_requirement_grounding import public_inputs


def _arguments(envelope):
    return {"corpus_receipt": envelope["corpus"], "original_inputs": envelope["original_inputs"],
        "expected_corpus_sha256": envelope["corpus"]["corpus_sha256"]}


def _fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def test_native_step_parameters_replay_without_fitting(public_inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("step certification must not fit/update a head")
    monkeypatch.setattr(api.ranker, "train_terminal_codebase_intent_ranker", forbidden)
    result = api.build_terminal_ranker_step_certificate(**_arguments(public_inputs))
    assert result["dimension"] == 80
    assert len(result["native_train_pairs"]) == 4
    assert len(result["pair_squared_norms_exact"]) == 4
    mu = _fraction(result["regularization_exact"])
    eta = _fraction(result["step_exact"])
    smooth = _fraction(result["conditional_smoothness_exact"])
    q = _fraction(result["conditional_contraction_exact"])
    assert mu == Fraction.from_float(0.01)  # actual float, not a silent replacement by 1/100
    assert mu != Fraction(1, 100)
    assert eta == Fraction.from_float(result["native_step_size"])
    assert eta > 0 and eta * smooth <= 1
    assert q == 1 - mu * eta and 0 <= q < 1
    assert result["lean_source_sha256"] == hashlib.sha256(result["lean_source"].encode()).hexdigest()
    assert result["training_calls"] == result["optimizer_updates"] == result["checker_calls"] == 0
    assert result["lean_checker_status"] == "not_run"
    assert result["logistic_hessian_bound_proved"] is False
    assert result["binary64_error_bound_proved"] is False
    assert result["asymptotic_optimizer_convergence_proved"] is False
    assert result["autoencoder_convergence_proved"] is False
    assert result["full_task_satisfaction"] == "unknown"
    assert api.validate_terminal_ranker_step_certificate(result, **_arguments(public_inputs)) == result


def test_foreign_corpus_pin_refused(public_inputs):
    arguments = _arguments(public_inputs)
    arguments["expected_corpus_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(api.RankerStepCertificateError):
        api.build_terminal_ranker_step_certificate(**arguments)


def test_foreign_convergence_authority_refused(public_inputs):
    result = api.build_terminal_ranker_step_certificate(**_arguments(public_inputs))
    foreign = deepcopy(result)
    foreign["asymptotic_optimizer_convergence_proved"] = True
    with pytest.raises(api.RankerStepCertificateError):
        api.validate_terminal_ranker_step_certificate(foreign, **_arguments(public_inputs))


@pytest.mark.parametrize("step", [0.0, -1.0, 101.0, math.inf, math.nan, True])
def test_invalid_or_oversized_step_refused(step):
    with pytest.raises(api.RankerStepCertificateError):
        api.certify_ranker_step_parameters(differences=[[0.0] * 80],
            regularization=0.01, step_size=step)


@pytest.mark.parametrize("rows", [[], [[0.0] * 79], [[True] * 80], [[math.inf] * 80]])
def test_invalid_difference_domain_refused(rows):
    with pytest.raises(api.RankerStepCertificateError):
        api.certify_ranker_step_parameters(differences=rows, regularization=0.01, step_size=1.0)


def test_feature_norms_and_float_embeddings_are_exact():
    row = [0.1, 0.2] + [0.0] * 78
    result = api.certify_ranker_step_parameters(differences=[row], regularization=0.01, step_size=0.5)
    norm = Fraction.from_float(0.1) ** 2 + Fraction.from_float(0.2) ** 2
    assert _fraction(result["pair_squared_norms_exact"][0]) == norm
    assert _fraction(result["conditional_smoothness_exact"]) == Fraction.from_float(0.01) + norm / 4
