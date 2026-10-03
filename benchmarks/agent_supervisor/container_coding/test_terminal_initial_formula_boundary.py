"""Frozen formula selection crosses the real initial-context wrapper boundary."""
import json
from pathlib import Path

import pytest

from tests.unit.logic.formalization.autoencoder.test_security_formula_decoder import formula_checkpoint  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_frozen_security import checkpoint, teacher, fork, joint_inputs  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _security_runtime_inputs
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formula_decoder


def test_frozen_formula_runtime_selection_reaches_actual_inference_and_replay(
        original, checkpoint, formula_checkpoint, monkeypatch, tmp_path):
    root, instruction, state = original
    # Authored boundary fixture, not a benchmark task or held-out quality claim.
    (root / "bottle.py").write_text("def application(value):\n    return (value + 7) * (value - 4)\n")
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    descriptor = tmp_path / "formula-selection.json"
    descriptor.write_text(json.dumps(formula_checkpoint))
    protocol = {"review_ref": "authored:initial-formula-boundary", "callback_parameter": "start_response"}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    runtime = _security_runtime_inputs(security_checkpoint=Path(checkpoint["output"]),
        security_checkpoint_manifest_sha256=checkpoint["manifest_sha256"], security_initializer=None,
        canonical_cve_export=None, canonical_cve_manifest_sha256=None,
        formula_decoder_descriptor=descriptor, header_protocol_descriptor=protocol_path)
    monkeypatch.setattr(security_formula_decoder, "train_security_formula_decoder",
                        lambda **_: pytest.fail("frozen initial inference retrained"))
    receipt = prep.initial_context(state=state, **runtime)
    advice = receipt["security_autoencoder_advice"]
    assert advice["formula_decoder"] == formula_checkpoint
    assert advice["header_protocol"] == protocol
    assert advice["formula_registration"]["checkpoint"] == formula_checkpoint
    assert advice["formalization"]["summary"]["learned_formula_count"] >= 1
    assert advice["training_steps"] == advice["provider_calls"] == advice["download_calls"] == 0
    assert advice["proof_authority"] is False
    assert advice["metadata_ducklake"]["status"] == advice["world_ducklake"]["status"] == "projected"
    selection = json.loads((state / "security-checkpoint-selection.json").read_text())
    assert selection["formula_decoder"] == formula_checkpoint and selection["header_protocol"] == protocol
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    summary = next(row for row in loaded["summaries"] if row["schema"] == "frozen-security-planning-summary@1")
    assert summary["formal_formula_heads_present"] is True
    assert not (state / "planner-invoked.json").exists()


@pytest.mark.parametrize("selection,diagnostic", [
    ({"formula_decoder": {}}, "frozen security context profile"),
    ({"header_protocol": {}}, "selected formula decoder"),
])
def test_wrapper_preserves_native_selection_rejection_before_index_writes(original, selection, diagnostic):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    with pytest.raises(ValueError, match=diagnostic):
        prep.initial_context(state=state, **selection)
    assert not (root / ".runtime/terminal-vectors").exists()
    assert not (state / "security-checkpoint-selection.json").exists()
