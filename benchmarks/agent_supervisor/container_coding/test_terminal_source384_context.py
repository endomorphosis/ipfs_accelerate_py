"""Actual pinned inference, signed source scope and planning/dispatch replays.

The provider response is authored. These tests do not claim a live benchmark
result or interpret source alignment as a security proof.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner
from test.api.test_semantic_router_integration import provider  # noqa: F401


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


def test_real_pinned_inference_reaches_planner_and_dispatch_without_replay(original, selected_config, monkeypatch, provider):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as native
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_nomination
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state, source384_config=selected_config,
                                  numerical_formalization_profile="python-numerical-structure@1")
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
    assert payload["schema"] == "supervisor-task-context-nominations@4"
    options = dict(repository=root, artifact=bundle["artifact"], expected_sha256=bundle["sha256"],
                   task_id=task["task_id"], task_cid=task["task_cid"])
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import read_task_context_historical_selection
    assert read_task_context_historical_selection(**options)["source384_context"] == receipt
    assert load_task_context_nomination(**options)
    # Observe the actual compiled worker capsule through the shared router and
    # reversible codec. Only the remote provider is authored; the model-backed
    # Source384 receipt, numerical projections, lakes and both validators are real.
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as worker_router
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    daemon = TodoImplementationDaemon(todo_path=root / ".runtime/worker.todo.md", repo_root=root,
        state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
        events_path=state / "worker/events.jsonl", task_header_prefix="## TB-")
    daemon._task_context_nomination_bundle = bundle
    portal = PortalTask(task_id=task["task_id"], canonical_task_cid=task["task_cid"],
        title="Retained public task", status="ready", completion="manual", priority="P2",
        track="implementation", outputs=["bottle.py", "report.jsonl"], validation=[],
        acceptance="Current source observations", metadata={"database task cid": task["task_cid"]})
    try:
        prompt = daemon._build_implementation_prompt(portal, attempt=1)
        wire, _ = json.JSONDecoder().raw_decode(prompt)
        refs = [row for row in wire["evidence"] if row["kind"] == "source-formalization-context"]
        formal = json.loads("".join(row["summary"] for row in refs))
        assert len(formal["contexts"]) == 2
        assert all(row["metadata"]["required"] for row in refs)
        assert formal["contexts"][0]["summary"]["candidate_samples"] == samples
        assert formal["contexts"][1]["summary"]["source_unit_links"]
        workspace = state / "allocated-worker"
        subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
        for name in receipt["source_hashes"]:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("CODEX_HOME", str(state / "empty-worker-home"))
        output, invocation = worker_router.run(prompt=prompt, provider="codex_cli", model="pinned",
            timeout=1, max_output_tokens=128, semantic_repository=root)
        _, observed = provider
        assert output == "literal summary" and len(observed) == 1
        encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
        model_prompt, advisory = worker_router.render_model_prompt(prompt=encoded.provider_prompt,
            purpose="coding", workspace=workspace, semantic_transport=True)
        assert model_prompt == observed[0][0]
        assert samples[0]["candidate_sha256"] in model_prompt
        for link in formal["contexts"][1]["summary"]["source_unit_links"]:
            assert link["unit_id"] in model_prompt and link["link_report_sha256"] in model_prompt
        assert codec.restore_semantic_router_prompt(provider_prompt=model_prompt[len(advisory):],
            table=encoded.table, repository=root) == prompt
        observation = invocation["source_formalization_context"]
        assert observation["workspace_source_current"] is True
        assert observation["context_kinds"] == ["source384-candidate-context", "numerical-formalization-context"]
        assert observation["neural_inference_replayed"] is False
        assert observation["proof_authority"] is False
        assert invocation["model_prompt_sha256"] == hashlib.sha256(model_prompt.encode()).hexdigest()
        assert invocation["usage"]["prompt_tokens"] == 123  # Authored provider accounting only.
    finally:
        daemon.close_event_runtime()
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


def _authored_source_alignment_case(statuses=("unsupported", "unsupported", "qualified", "qualified")):
    """Authored decoder predictions with real source qualification; no weights loaded."""
    from types import SimpleNamespace
    from ipfs_datasets_py.logic.formalization.autoencoder.source_function_units import extract_function_units
    from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384 import SourceProgramDecoder384
    from ipfs_datasets_py.logic.software_verification.program import ProgramExpression

    source = "\n".join(
        f"def function_{index}(capacity{': int' if status != 'unsupported' else ''}, "
        f"threshold{': int' if status != 'unsupported' else ''}) -> bool:\n    return capacity < threshold\n"
        for index, status in enumerate(statuses))
    raw = source.encode()
    inventory = extract_function_units(source_bytes=raw, source_sha256=hashlib.sha256(raw).hexdigest(),
                                       source_path="bottle.py")
    predicted = []
    inputs = []
    for unit, status in zip(inventory["units"], statuses):
        operands = ("expr:capacity", "expr:threshold")
        candidate = dict(kind="program_expression", document=ProgramExpression(
            "expr:result", "binary", "boolean", operand_ids=operands, evaluation_order=operands,
            operator="<=" if status == "mismatch" else "<", source_ref_ids=("source",)).to_dict())
        predicted.append(dict(id=unit["unit_id"], source_sha256=unit["normalized_body_sha256"],
                              candidate_ir=candidate, status="unqualified_candidate"))
        inputs.append(dict(id=unit["unit_id"], source_text=unit["normalized_source_text"], embedding=[.125] * 384))
    runtime = SourceProgramDecoder384(SimpleNamespace(
        describe=lambda: dict(domain_id="security_ir"),
        infer=lambda rows: dict(domain_id="security_ir", rows=predicted)), checkpoint_sha256="c" * 64)
    checked = runtime.infer(inputs)["rows"]
    assert [row["source_contract"]["status"] for row in checked] == list(statuses)
    inference = dict(report=dict(key=dict(source_head={"authored": True}, version_id="authored-version"),
        preparation=dict(files=[dict(path="bottle.py", extraction=inventory)]), coverage={},
        output=dict(rows=[dict(id=row["id"], candidate=row) for row in checked])))
    return source, inventory, inference


def _authored_summary(inference):
    return owner._summary(inference, inventory=[dict(path="bottle.py", disposition="captured")],
                          checkpoint_sha256="c" * 64, inference_sha256=hashlib.sha256(owner._raw(inference)).hexdigest())


def test_source_aligned_summary_candidates_are_not_hidden_by_earlier_unsupported_units():
    _, inventory, inference = _authored_source_alignment_case()
    before = owner._raw(inference)
    summary = _authored_summary(inference)
    assert [r["unit_id"] for r in summary["candidate_samples"]] == [u["unit_id"] for u in inventory["units"][2:]]
    assert summary["candidate_count"] == 4 and summary["omitted_candidates"] == 2
    assert summary["source_aligned_candidate_count"] == summary["sampled_source_aligned_candidates"] == 2
    assert summary["omitted_source_aligned_candidates"] == 0
    assert summary["candidate_origin"] == "learned_checkpoint_inference"
    assert summary["candidate_selection"] == "source_alignment_then_path_byte_order@1"
    assert summary["source_validation_scope"] == "guarded_scalar_source_candidate_alignment"
    assert summary["source_semantics_verified"] is summary["whole_program_semantics_verified"] is False
    assert all(row["source_validation_status"] == "qualified" and row["status"] == "unqualified_candidate"
               and row["proof_authority"] is False and row["source_semantics_verified"] is False
               for row in summary["candidate_samples"])
    assert owner._raw(inference) == before and len(owner._raw(summary)) <= 8192


def test_source_aligned_summary_preserves_path_byte_order_within_ranks():
    _, inventory, inference = _authored_source_alignment_case(("unsupported", "mismatch", "qualified"))
    inference["report"]["output"]["rows"].reverse()
    summary = _authored_summary(inference)
    assert [row["unit_id"] for row in summary["candidate_samples"]] == [
        inventory["units"][2]["unit_id"], inventory["units"][0]["unit_id"]]
    assert summary["source_aligned_candidate_count"] == summary["sampled_source_aligned_candidates"] == 1
    assert summary["omitted_candidates"] == 1


def test_source_aligned_summary_oversize_omissions_do_not_consume_sample_slots():
    _, inventory, inference = _authored_source_alignment_case(("unsupported", "qualified", "qualified"))
    # A long qualified name can exceed the existing per-sample bound.
    inference["report"]["preparation"]["files"][0]["extraction"]["units"][1]["qualified_name"] = "x" * 2048
    summary = _authored_summary(inference)
    assert [row["unit_id"] for row in summary["candidate_samples"]] == [
        inventory["units"][2]["unit_id"], inventory["units"][0]["unit_id"]]
    assert summary["source_aligned_candidate_count"] == 2
    assert summary["sampled_source_aligned_candidates"] == summary["omitted_source_aligned_candidates"] == 1
    assert summary["omitted_candidates"] == 1
    assert all(len(owner._raw(row)) <= 2048 for row in summary["candidate_samples"])


def test_source_aligned_summary_reaches_actual_planner_before_goals(original, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

    root, instruction, state = original
    source, inventory, inference = _authored_source_alignment_case()
    (root / "bottle.py").write_text(source)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state)
    summary = _authored_summary(inference)
    config = state / "authored-source384-config.json"
    config.write_bytes(b'{"scope":"authored-decoder-no-model-execution"}')
    output = state / "source384-context"
    output.mkdir()
    receipt = dict(schema=owner.SCHEMA, repository=str(root), output=str(output),
        config_path=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        source_hashes={name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()},
        summary=summary, **owner.AUTHORITY)
    receipt_raw = owner._raw(receipt)
    (output / "receipt.json").write_bytes(receipt_raw)
    prep._write(state / "source384-selection.json", dict(config_path=str(config), config_sha256=receipt["config_sha256"]))
    descriptor_path = root / result["descriptor"]["artifact"]
    descriptor = json.loads(descriptor_path.read_bytes())
    descriptor["source384_context"] = receipt
    descriptor_path.write_bytes(owner._raw(descriptor))
    result.update(source384_context=receipt, descriptor=initial._reference(root, descriptor_path))
    prep._write(state / "initial-context-result.json", result)
    checks = []
    def authored_currentness(*, repository, expected_receipt):
        assert repository == root and expected_receipt == receipt
        assert (output / "receipt.json").read_bytes() == receipt_raw
        assert hashlib.sha256(config.read_bytes()).hexdigest() == receipt["config_sha256"]
        assert (root / "bottle.py").read_text() == source
        checks.append(True)
        return expected_receipt
    monkeypatch.setattr(owner, "validate_source384_context", authored_currentness)
    calls = []
    def router(prompt, **kwargs):
        with IntentRepository(state / "intent.duckdb") as intent:
            assert intent.plan_projection()["tasks"] == []
        assert checks == [True]
        for unit in inventory["units"][2:]:
            assert unit["unit_id"] in prompt
        for candidate in summary["candidate_samples"]:
            assert candidate["candidate_sha256"] in prompt
            assert json.dumps(candidate["candidate_ir_json"]) in prompt
        assert "learned_checkpoint_inference" in prompt
        assert "guarded_scalar_source_candidate_alignment" in prompt
        assert '"source_semantics_verified":false' in prompt.replace(" ", "")
        calls.append(True)
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)
    _version(monkeypatch)
    planned = prep.plan(state=state, provider_callable=router)
    assert planned["qualified"] is True and calls == [True] and checks == [True, True]
    assert planned["initial_indexed_context"]["supplied_to_router"] is True


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


def test_complete_create_manifest_source_population(tmp_path, monkeypatch):
    sources = {}
    for index in range(256):
        name = f"source_{index}.py"
        raw = f"value = {index}\n".encode()
        (tmp_path / name).write_bytes(raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    assert owner._sources(tmp_path, sources) == sources
    (tmp_path / "source_255.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="differs from signed population"):
        owner._sources(tmp_path, sources)
    monkeypatch.setattr(owner, "_read", lambda *a: pytest.fail("over-bound manifest read source bytes"))
    with pytest.raises(ValueError, match="bounded signed Source384 source population"):
        owner._sources(tmp_path, {**sources, "source_256.py": "0" * 64})


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


@pytest.mark.parametrize("preflight", ["_output", "_sources", "_pins"])
def test_preparation_deadline_includes_preflight_before_output_or_admission(tmp_path, monkeypatch, preflight):
    from types import SimpleNamespace
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    root = tmp_path / "repository"
    root.mkdir()
    source = root / "module.py"
    source.write_text("def value(): return 1\n")
    output = tmp_path / "private-context"
    now = [100.]
    original = getattr(owner, preflight)
    def slow_preflight(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] = 111.
        return result
    monkeypatch.setattr(owner, preflight, slow_preflight)
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources",
        lambda **kwargs: pytest.fail("expired preflight cannot enter native admission"))
    with pytest.raises(ValueError, match="preparation deadline expired"):
        owner.prepare_source384_context(repository=root,
            source_hashes={source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
            output=output, config_path=tmp_path / "not-read.json", timeout_seconds=10.)
    assert not output.exists()
