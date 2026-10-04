"""Training evidence must preserve metric and qualification boundaries."""
from copy import deepcopy
import hashlib

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_codebase_ir_experiment import (
    assess_training, verify_instruction_binding,
)


def metrics():
    return {"before_reconstruction_loss": 0.02, "after_reconstruction_loss": 0.002,
        "native_objective_losses": [0.013, 0.0129, 0.0128, 0.0127, 0.0126],
        "native_gradient_norms": [0.1, 0.05, 0.04, 0.03, 0.02], "epochs": 5,
        "initial_weights_sha256": "a" * 64, "final_weights_sha256": "b" * 64,
        "holdout_evaluated": False, "training_scope": "transductive_admitted_code_functions"}


def test_stabilization_uses_one_objective_instead_of_appending_endpoint_mse():
    report = assess_training(metrics())
    assert report["finite_window_stable"]
    assert len(report["tail_absolute_changes"]) == 4
    assert max(report["tail_absolute_changes"]) < 0.0002
    assert report["loss_decreased"]
    assert not report["asymptotic_optimizer_convergence_proved"]
    assert not report["held_out_accuracy_measured"]


@pytest.mark.parametrize("kind", ["nonfinite", "incomplete", "missing_gradients", "negative"])
def test_missing_or_invalid_training_observations_cannot_qualify(kind):
    value = metrics()
    if kind == "nonfinite":
        value["after_reconstruction_loss"] = float("nan")
    elif kind == "incomplete":
        value["native_objective_losses"].pop()
    elif kind == "missing_gradients":
        value["native_gradient_norms"] = []
    else:
        value["native_objective_losses"][0] = -0.1
    with pytest.raises(ValueError, match="finite"):
        assess_training(value)


def test_increasing_loss_and_no_parameter_update_remain_failed_observations():
    value = deepcopy(metrics())
    value["after_reconstruction_loss"] = value["before_reconstruction_loss"] * 2
    value["final_weights_sha256"] = value["initial_weights_sha256"]
    value["native_gradient_norms"] = [0.0] * 5
    report = assess_training(value)
    assert not report["loss_decreased"]
    assert not report["weights_changed"]
    assert not report["nonzero_gradient_observed"]


@pytest.mark.parametrize("drift", ["original", "captured", "both"])
def test_instruction_cannot_change_after_training_into_a_new_snapshot_digest(tmp_path, drift):
    original = tmp_path / "instruction.md"
    repository = tmp_path / "repository"
    repository.mkdir()
    captured = repository / ".supervisor-instruction.md"
    body = b"Fix the public source and retain the exact requirement.\n"
    original.write_bytes(body)
    captured.write_bytes(body)
    fixture = {"repository": str(repository),
        "instruction_provenance": {"sha256": hashlib.sha256(body).hexdigest()}}
    assert verify_instruction_binding(fixture, original) == fixture["instruction_provenance"]["sha256"]
    if drift in ("original", "both"):
        original.write_bytes(body + b"Additional task.\n")
    if drift in ("captured", "both"):
        captured.write_bytes(body + b"Additional task.\n")
    with pytest.raises(ValueError, match="captured fixture provenance"):
        verify_instruction_binding(fixture, original)
