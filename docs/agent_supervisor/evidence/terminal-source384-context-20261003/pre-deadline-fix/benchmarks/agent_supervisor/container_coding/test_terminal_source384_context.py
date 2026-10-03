"""Actual pinned inference, signed source scope and planning/dispatch replays.

The provider response is authored. These tests do not claim a live benchmark
result or interpret source alignment as a security proof.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner


@pytest.fixture
def selected_config(tmp_path):
    checkpoint = os.environ.get("CODEBASE384_CHECKPOINT")
    snapshot = os.environ.get("CODEBASE384_EMBEDDING_SNAPSHOT")
    if not checkpoint or not snapshot:
        pytest.skip("explicit pinned checkpoint and cached GTE snapshot required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    copied = tmp_path / "selected-checkpoint.json"
    shutil.copyfile(checkpoint, copied)
    config = dict(schema="terminal-source384-config@1", mode="pinned_parent",
        checkpoint_path=str(copied), checkpoint_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
        embedding_snapshot=snapshot, embedding_revision=embedding.PINNED_REVISION,
        embedding_assets=embedding._snapshot_assets(snapshot)[1], training_steps=0, download_calls=0)
    path = tmp_path / "source384-config.json"
    path.write_text(json.dumps(config, sort_keys=True))
    return path


def test_real_pinned_inference_reaches_planner_and_dispatch_without_replay(original, selected_config, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as native
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_nomination
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state, source384_config=selected_config)
    receipt = result["source384_context"]
    inference = json.loads((Path(receipt["output"]) / "inference.json").read_bytes())
    assert inference["native_worker_executed"] is True
    assert inference["inference_executed"] is True
    assert inference["report"]["output"]["model_loads"] == 1
    assert receipt["summary"]["coverage"] == inference["report"]["coverage"]
    samples = receipt["summary"]["candidate_samples"]
    assert len(samples) == 1
    assert json.loads(samples[0]["candidate_ir_json"]) == inference["report"]["output"]["rows"][0]["candidate"]["candidate_ir"]
    assert "projected_embedding" not in json.dumps(samples)
    assert samples[0]["source_validation_status"] == "unsupported"
    assert samples[0]["source_semantics_verified"] is False
    assert receipt["summary"]["repository_retention"] == "not_evaluated"
    assert receipt["training_steps"] == 0
    assert set(receipt["source_hashes"]) == set(prepared["manifest"]["payload"]["sources"])
    assert receipt["checkpoint_sha256"] == json.loads(selected_config.read_text())["checkpoint_sha256"]
    for row in inference["report"]["output"]["rows"]:
        assert row["candidate"]["proof_authority"] is False
        assert row["candidate"].get("source_contract", {}).get("status") != "qualified"
    monkeypatch.setattr(native, "_worker", lambda *a, **k: pytest.fail("neural inference repeated during observation"))
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert receipt["summary"] in loaded["summaries"]
    calls = []
    def router(prompt, **kwargs):
        calls.append(prompt)
        assert receipt["checkpoint_sha256"] in prompt
        assert receipt["inference_sha256"] in prompt
        assert "terminal-source384-planning-summary@1" in prompt
        assert samples[0]["candidate_sha256"] in prompt
        assert json.dumps(samples[0]["candidate_ir_json"]) in prompt
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    planned = prep.plan(state=state, provider_callable=router)
    assert planned["qualified"] is True and len(calls) == 1
    assert planned["initial_indexed_context"]["supplied_to_router"] is True
    context = prep.context(state=state)
    bundle = context["context_bundle"]
    payload = json.loads((root / bundle["artifact"]).read_text())
    task = payload["tasks"][0]
    assert task["source384_context"] == receipt
    options = dict(repository=root, artifact=bundle["artifact"], expected_sha256=bundle["sha256"],
                   task_id=task["task_id"], task_cid=task["task_cid"])
    assert load_task_context_nomination(**options)
    # Real source/model/receipt changes must fail at the canonical worker seam.
    checkpoint = Path(json.loads(selected_config.read_text())["checkpoint_path"])
    for path in (root / "bottle.py", checkpoint, selected_config,
                 Path(receipt["output"]) / "inference.json"):
        before = path.read_bytes()
        try:
            path.write_bytes(before + b" ")
            with pytest.raises(ValueError):
                load_task_context_nomination(**options)
        finally:
            path.write_bytes(before)
    old_pins = owner._pins()
    monkeypatch.setattr(owner, "_pins", lambda: {**old_pins, "consumer": "0" * 64})
    with pytest.raises(ValueError, match="producer"):
        load_task_context_nomination(**options)


def test_source384_missing_selection_cannot_silently_use_legacy_profile(original, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    prep.initial_context(state=state)
    (state / "source384-selection.json").write_text("{}")
    with pytest.raises(ValueError, match="selected Source384 context is missing"):
        initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)


def test_checkpoint_change_during_router_is_rejected_before_admission(original, selected_config, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    prep.initial_context(state=state, source384_config=selected_config)
    checkpoint = Path(json.loads(selected_config.read_text())["checkpoint_path"])
    calls = []
    def router(prompt, **kwargs):
        calls.append(prompt)
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    planned = prep.plan(state=state, provider_callable=router)
    assert planned["qualified"] is False
    assert "checkpoint bytes differ" in planned["failure"]["message"]
    assert len(calls) == 1
    assert not (state / "admission.json").exists()


def test_source384_rejects_legacy_training_before_writes(original):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    with pytest.raises(ValueError, match="mutually exclusive"):
        initial.prepare_initial_context(state=state, prepared=prepared, model_snapshot=None,
            model_revision="", required_raw_paths=[prep.INSTRUCTION, prep.SMOKE],
            source384_config=state / "absent.json", train_autoencoder=True)
    assert not (state / "source384-selection.json").exists()


def test_native_inventory_rejects_undeclared_captured_source(tmp_path):
    from types import SimpleNamespace
    manifest = SimpleNamespace(snapshot=SimpleNamespace(entries=[SimpleNamespace(path="oracle.py")]))
    index = SimpleNamespace(load=lambda cid: manifest)
    with pytest.raises(ValueError, match="undeclared source"):
        owner._inventory(index, SimpleNamespace(manifest_cid="test"), {"code.py": "0" * 64})


@pytest.mark.parametrize("paths,functions,selected", [(["a.py"],1024,128),
    (["a.py","b.py"],512,128), (["a.py","b.py"],1024,64)])
def test_native_inference_cannot_shrink_declared_population(paths, functions, selected):
    inventory = [dict(path=name, disposition="captured") for name in ("a.py", "b.py")]
    inference = dict(report=dict(preparation=dict(paths=paths, max_functions=functions,
                                                 max_selected_units=selected)))
    with pytest.raises(ValueError, match="complete declared Python population"):
        owner._validate_population(inference, inventory)


def test_observation_deadline_includes_read_and_asset_validation_time(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    now = [100.]
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", lambda **kwargs: nullcontext(object()))
    def consume_budget(**kwargs):
        assert kwargs["deadline"] == 110.
        now[0] = 111.
        return {}
    monkeypatch.setattr(owner, "_validate_source384_context", consume_budget)
    with pytest.raises(ValueError, match="deadline expired"):
        owner.validate_source384_context(repository="unused", expected_receipt={}, timeout_seconds=10.)
