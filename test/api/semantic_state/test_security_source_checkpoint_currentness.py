"""Real checkpoint-path drift controls with explicitly simulated inference."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import security_source_program_advisor_384 as subject
from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384 as shared


@pytest.mark.parametrize("decoder", ["structured", "sequence_v2"])
@pytest.mark.parametrize("phase", ["unchanged", "inference", "lake", "lake_failure"])
@pytest.mark.parametrize("state_profile", [False, True])
def test_selected_checkpoint_stays_current_through_optional_native_work(tmp_path, monkeypatch, decoder, phase, state_profile):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b'{"fixture":"numerical loader is simulated"}')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    text = "def identity(value: int) -> int:\n    return value\n"
    source_digest = hashlib.sha256(text.encode()).hexdigest()
    report = dict(domain_id="security_ir", checkpoint_sha256=digest,
        rows=[dict(id="input-0", source_sha256=source_digest,
            candidate_ir={"fixture": "authored prediction"}, source_contract={"status": "qualified"})])

    def change():
        checkpoint.write_bytes(b'{"fixture":"checkpoint successor"}')

    def infer(texts, **options):
        assert texts == [text]
        if phase == "inference":
            change()
        return deepcopy(report)

    def load(path, **options):
        assert path == str(checkpoint)
        assert options == dict(expected_sha256=digest, decoder=decoder)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        return SimpleNamespace(describe=lambda: {"domain_id": "security_ir"}, infer_texts=infer)

    def lake(*args, **kwargs):
        if phase in {"lake", "lake_failure"}:
            change()
        if phase == "lake_failure":
            raise FileNotFoundError("authored unavailable Lake control")
        return SimpleNamespace(to_dict=lambda: {"status": "passed", "backend_executed": True})

    monkeypatch.setattr(shared, "load_source_program_decoder_384", load)
    monkeypatch.setattr(shared, "build_decoded_source_program_lake", lake)
    selected = dict(schema=subject.CONFIG_SCHEMA, checkpoint_path=str(checkpoint),
        checkpoint_sha256=digest, decoder=decoder, embedding_snapshot_path=None,
        lake={"executable": "/authored/lake", "timeout_seconds": 1})
    if state_profile:
        from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384_v2 as compatible
        from ipfs_accelerate_py.agent_supervisor.runtime import security_source_state_advisor_384 as state
        selected.update(schema=subject.STATE_CONFIG_SCHEMA,
            finite_state_domains={"module.py": {"value": {"lower": 0, "upper": 1}}})
        monkeypatch.setattr(compatible, "load_source_program_decoder_384_v2",
            lambda path, input_view, **options: load(path, **options))
        monkeypatch.setattr(subject, "_checkpoint_compatibility", lambda *args: {"fixture": "simulated compatibility"})
        monkeypatch.setattr(state, "consume_source_state_advice",
            lambda **options: {"status": "state_candidate_advice", "fixture": "simulated native state"})
    result = subject.prepare_security_source_program_advice(config=selected,
        source_rows=[dict(id="module.py", source_text=text, source_sha256=source_digest)])
    assert result["continue_planning"] is True
    assert result["provider_calls"] == result["training_steps"] == result["download_calls"] == 0
    if phase == "unchanged":
        assert result["status"] == "source_candidate_advice"
        assert result["inference"] == report
    else:
        assert result["status"] == "fail_open_unavailable"
        assert result["failure_stage"] == "checkpoint_currentness"
        assert result["inference"] is result["lake"] is None
        assert result.get("source_state") is None
        assert "qualified_candidate_count" not in result


def test_actual_local_security_weights_survive_currentness_fence():
    """CPU qualification supplies offline assets; all numerical owners are real."""
    checkpoint, snapshot = (os.environ.get(name) for name in (
        "CODEBASE384_CHECKPOINT", "CODEBASE384_EMBEDDING_SNAPSHOT"))
    if not checkpoint or not snapshot:
        pytest.skip("explicit offline Security384 checkpoint and embedding snapshot required")
    before = Path(checkpoint).read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    text = "def derive(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"
    result = subject.prepare_security_source_program_advice(config=dict(schema=subject.STATE_CONFIG_SCHEMA,
        checkpoint_path=checkpoint, checkpoint_sha256=digest, decoder="structured",
        embedding_snapshot_path=snapshot, lake=None, finite_state_domains={"module.py": {
            "capacity": {"lower": -1, "upper": 1}, "threshold": {"lower": 0, "upper": 1}}}),
        source_rows=[dict(id="module.py", source_text=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())])
    assert result["status"] in {"source_candidate_advice", "fail_open_no_qualified_candidates"}, result
    assert result["checkpoint_compatibility"]["artifact_sha256"] == digest
    assert result["inference"]["checkpoint_sha256"] == digest
    assert result["inference"]["rows"][0]["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert Path(checkpoint).read_bytes() == before
    assert result["continue_planning"] and all(result[key] is False for key in subject.FALSE)
    assert result["provider_calls"] == result["training_steps"] == result["download_calls"] == 0
