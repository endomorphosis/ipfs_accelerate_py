"""Finite numerical evidence cannot silently become optimizer convergence."""
from copy import deepcopy
import pytest

from benchmarks.agent_supervisor.container_coding.terminal_codebase_decoder_experiment import assess_decoder_training


def _training():
    return {"epochs": 5, "training_losses": [0.002, 0.0019, 0.0018, 0.0017, 0.0016],
            "gradient_norms": [0.1] * 5, "native_kernel_calls": 5,
            "initial_head_sha256": "a"*64, "final_head_sha256": "b"*64}


def test_fixed_tail_uses_native_pre_update_trace_and_separate_checkpoint_evaluation():
    result = assess_decoder_training(_training(), {"native_production_cross_entropy": 0.001})
    assert result["finite_window_stable"] is True
    assert len(result["tail_decimal_rational_changes"]) == 4
    assert result["last_native_trace_loss"] == 0.0016
    assert result["final_checkpoint_loss"] == 0.001
    assert result["asymptotic_optimizer_convergence_proved"] is False


def test_exact_decimal_tolerance_boundary_does_not_inherit_float_subtraction_error():
    training = _training()
    training["training_losses"] = [0.0011, 0.0009, 0.0007, 0.0005, 0.0003]
    result = assess_decoder_training(training, {"native_production_cross_entropy": 0.0002})
    assert result["finite_window_stable"] is True
    assert result["tail_decimal_rational_changes"] == [{"numerator": 1, "denominator": 5000}] * 4


@pytest.mark.parametrize("kind", ["short", "nan", "bool", "negative", "missing_gradient", "no_native_calls"])
def test_incomplete_or_invalid_numeric_evidence_rejects(kind):
    training = deepcopy(_training())
    if kind == "short":
        training["training_losses"].pop()
    elif kind == "nan":
        training["training_losses"][0] = float("nan")
    elif kind == "bool":
        training["training_losses"][0] = True
    elif kind == "negative":
        training["gradient_norms"][0] = -1
    elif kind == "missing_gradient":
        training["gradient_norms"].pop()
    else:
        training["native_kernel_calls"] = 0
    with pytest.raises(ValueError, match="complete finite native"):
        assess_decoder_training(training, {"native_production_cross_entropy": 0.001})


def test_failed_stability_and_parameter_changes_remain_explicit_outcomes():
    training = _training()
    training["training_losses"] = [0.4, 0.3, 0.2, 0.1, 0.08]
    training["final_head_sha256"] = training["initial_head_sha256"]
    result = assess_decoder_training(training, {"native_production_cross_entropy": 0.09})
    assert result["finite_window_stable"] is False
    assert result["loss_decreased"] is True
    assert result["weights_changed"] is False
