"""Training evidence must preserve metric and qualification boundaries."""
from copy import deepcopy
import hashlib
import json

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


def _explicit_context(text):
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_ir_experiment as experiment
    return {"schema": experiment.INTENT_MATCHING_CONTEXT_SCHEMA,
        "source_text": text, "source_identity": {"content_sha256": hashlib.sha256(text.encode()).hexdigest()},
        # Native document/query semantics are checked by the separate join
        # builder. These inert objects exercise only acquisition/custody here.
        "intent_document": {"authored_acquisition_control": True}, "query": {"focus": "explicit"}}


@pytest.mark.parametrize("mutation", ["none", "changed_instruction", "duplicate", "nonfinite", "missing", "symlink"])
def test_explicit_context_preserves_complete_instruction_and_refuses_bad_acquisition(tmp_path, mutation):
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_ir_experiment as experiment
    text = "Repair bottle.\n\nOpaque clause: preserve μ and every original span.\n"
    value = _explicit_context(text)
    path = tmp_path / "context.json"
    if mutation == "changed_instruction":
        value["source_text"] = "Repair bottle."
    if mutation == "nonfinite":
        value["query"]["value"] = float("nan")
    path.write_text(json.dumps(value))
    if mutation == "duplicate":
        path.write_text(path.read_text().replace('{', '{"schema":"duplicate",', 1))
    if mutation == "missing":
        path.unlink()
    if mutation == "symlink":
        original = tmp_path / "original-context.json"
        path.rename(original)
        path.symlink_to(original)
    if mutation != "none":
        with pytest.raises((ValueError, OSError)):
            experiment._load_intent_matching_context(path, instruction_bytes=text.encode())
    else:
        assert experiment._load_intent_matching_context(path, instruction_bytes=text.encode()) == value
        copied = experiment._load_intent_matching_context(value, instruction_bytes=text.encode())
        copied["intent_document"]["authored_acquisition_control"] = False
        assert value["intent_document"]["authored_acquisition_control"] is True
        assert copied["source_text"].encode() == text.encode()


def test_skipped_retraining_retains_real_zero_inference_and_fixed_rank_controls(tmp_path, monkeypatch):
    """Real tensor inference over authored inputs; no training/source authority claim."""
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_ir_experiment as experiment
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae

    learning = tmp_path / "learning"
    learning.mkdir()
    checkpoint = {"weights": [[[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0],
                              [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]]}
    features = {"rows": [
        {"row_id": "first", "path": "bottle.py", "symbol": "first", "line": 1, "features": [1.0, 0.0]},
        {"row_id": "second", "path": "bottle.py", "symbol": "second", "line": 3, "features": [0.0, 2.0]}]}
    (learning / "checkpoint.json").write_text(json.dumps(checkpoint))
    (learning / "features.json").write_text(json.dumps(features))
    learner = {"repository": str(tmp_path), "output": str(learning), "source_hashes": {"bottle.py": "a" * 64},
        "checkpoint_sha256": "b" * 64, "epochs_completed": 3,
        "metrics": {"seed": 1729, "after_reconstruction_loss": 0.02}}
    def forbidden(**kwargs):
        pytest.fail("explicit repeat opt-out must not start another fit")
    monkeypatch.setattr(ae, "train_codebase_autoencoder", forbidden)
    original = (learning / "checkpoint.json").read_bytes()
    controls = experiment._training_controls({"learner": learner}, tmp_path / "controls",
        repeat=False, retain_inference=True)
    assert len(controls) == 1 and controls[0]["name"] == "zero_weights_inference"
    zero = controls[0]
    assert {row["row_id"]: row["reconstruction_error"] for row in zero["inference"]["ranks"]} == {
        "first": 0.5, "second": 2.0}
    assert all(row["latent"] == [0.0, 0.0] for row in zero["inference"]["ranks"])
    assert (learning / "checkpoint.json").read_bytes() == original
    trained = {"ranks": [{**row, "latent": [0.25, -0.25], "reconstruction_error": error}
        for row, error in zip(features["rows"], (0.1, 0.2))]}
    rows = experiment._nomination_model_controls(learner=learner,
        training_receipt={"features_sha256": "c" * 64}, learned_index=trained, zero_control=zero)
    by_name = {row["name"]: row for row in rows}
    assert by_name["model_off"]["ranking"] == []
    assert by_name["trained"]["ranking"] == trained["ranks"]
    assert by_name["zero_heads"]["ranking"] == zero["inference"]["ranks"]
    assert by_name["shuffled_order"]["ranking"] == list(reversed(trained["ranks"]))
    assert all(row["training_executed"] is False and row["source_hashes"] == learner["source_hashes"]
               and row["checkpoint_sha256"] == learner["checkpoint_sha256"] for row in rows)
    rows[0]["ranking"][0]["latent"][0] = 99
    assert trained["ranks"][0]["latent"] == [0.25, -0.25]


def test_nomination_family_preserves_whole_envelope_after_chunking_and_native_restart(tmp_path):
    """Authored envelope tests storage custody, not producer/model qualification."""
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_supervisor_fixture as fixture_api
    from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata

    text = "Opaque original μ clause.\n" * 18000
    receipt = {"schema": "authored-storage-custody-control@1",
        "original_match": {"intent_source": {"text": text},
            "residual_requirements": [{"status": "unresolved_software_behavior", "original_text": text}]},
        "fixed_candidates": {"lexical": [{"row_id": "current-source"}],
            "kg": [{"subject": "current-source", "relation": "calls", "object": "unknown"}]},
        "model_controls": [{"name": "model_off", "ranking": [], "training_executed": False}],
        "checkpoint_sha256": "b" * 64, "proof_authority": False,
        "execution_authority": False, "semantic_alignment_verified": False}
    family = "intent_codebase_training_nominations"
    # Current experiment has 27 ordinary families; one added family and the
    # existing overflow chunks remain below the unchanged 32-family bound.
    records = {f"baseline_{index}": [] for index in range(23)}
    records.update({name: [] for name in metadata.DEFAULT_FAMILIES})
    records[family] = [receipt]
    stored = fixture_api.bound_terminal_codebase_metadata_records(records)
    assert len(stored) == 29 and len(stored) <= metadata.LIMITS["families"] == 32
    assert fixture_api.reconstruct_terminal_codebase_metadata_records(stored) == records
    output = tmp_path / "metadata"
    report = metadata.hydrate_codebase_ir_metadata(records=stored, output=output,
        source_snapshot={"schema": "authored-storage-source-control@1", "source_sha256": "a" * 64})
    assert report["fresh_process_readback"]["verified"] is True
    assert metadata.validate_codebase_ir_metadata(output=output, expected=report, fresh_process=True) == report
    recovered = {name: [json.loads(line)["payload"]
        for line in (output / export["relative_path"]).read_text().splitlines()]
        for name, export in report["exports"].items()}
    assert fixture_api.reconstruct_terminal_codebase_metadata_records(recovered) == records
    assert recovered[family][0]["schema"] == fixture_api.ARTIFACT_SCHEMA


def test_detached_nomination_refuses_resealed_source_substitution_before_semantic_replay(monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_ir_experiment as experiment
    from benchmarks.agent_supervisor.container_coding import terminal_codebase_intent_training_join as join
    def forbidden(*args, **kwargs):
        pytest.fail("unbound recovered inputs must refuse before the semantic validator")
    monkeypatch.setattr(join, "validate_terminal_intent_training_join", forbidden)
    recovered = {"sources": [{"path": "bottle.py", "source_sha256": "a" * 64}],
        "training": [{"learner": {}, "receipt": {}, "checkpoint": {}}],
        experiment.INTENT_NOMINATION_FAMILY: [{"schema": experiment.INTENT_NOMINATION_METADATA_SCHEMA,
            "receipt": {"resealed": True}, "replay_inputs": {
                "source_records": [{"path": "bottle.py", "source_sha256": "b" * 64}]}}]}
    with pytest.raises(ValueError, match="independently reconstructed source/training"):
        experiment._validate_recovered_intent_nomination(recovered)
