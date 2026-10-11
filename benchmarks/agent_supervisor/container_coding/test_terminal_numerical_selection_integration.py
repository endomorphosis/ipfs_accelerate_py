"""Actual experimental numerical weights reach advisory supervisor context.

The checkpoint, GTE encoder, native owner, lakes, planner admission, worker
compiler and historical audit are real. Planner/provider replies are authored;
these tests do not establish benchmark performance or prove NumPy semantics.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as worker_router
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
from test.api.test_semantic_router_integration import provider  # noqa: F401


# Exact independently authored release holdout texts, never training inputs.
MATCHING_SOURCE = """import numpy as array_api

def choose_index(scores, buffer, options, token):
    return array_api.argmax(scores)
"""
MISMATCHING_SOURCE = """import numpy as np

def select_offset(values, other, context, marker):
    return np.argmax(values)
"""
UNSUPPORTED_SOURCE = """import numpy as np

def choose(values):
    return np.argmax(np.abs(values))
"""
RELEASE_SHA256 = "26ccc7cf8420e0ed6cbbe22603e09be0e3572fb32a1e489e877fc5081f506e03"
HEAD_SHA256 = "d22acf5a95d2ceb9340e6ad741928c536b4d52eb13ce846c040a4778f62b33d8"
PROJECTION_SHA256 = "4e90ec5a3ff3978d3f6d40fb366ea01f2a056fe6c59b3fddcf7c50f66366afd7"


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


@pytest.fixture
def numerical_config(tmp_path):
    checkpoint = os.environ.get("NUMERICAL_SELECTION384_CHECKPOINT")
    snapshot = os.environ.get("CODEBASE384_EMBEDDING_SNAPSHOT")
    if not checkpoint or not snapshot:
        pytest.skip("explicit numerical development checkpoint and cached GTE snapshot required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding

    raw = Path(checkpoint).read_bytes()
    assert _sha(raw) == RELEASE_SHA256
    copied = tmp_path / "numerical-selection-checkpoint.json"
    copied.write_bytes(raw)
    config = dict(schema="supervisor-numerical-selection-config@1",
        profile="numpy-unary-selection-384@1", allow_experimental=True,
        checkpoint_path=str(copied), checkpoint_sha256=RELEASE_SHA256,
        embedding_snapshot=snapshot, embedding_revision=embedding.PINNED_REVISION,
        embedding_assets=embedding._snapshot_assets(snapshot)[1], training_steps=0, download_calls=0)
    path = tmp_path / "numerical-selection-config.json"
    path.write_text(json.dumps(config, sort_keys=True))
    return path


def _forbid_new_inference(monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native

    monkeypatch.setattr(native, "_worker",
        lambda *a, **k: pytest.fail("retained context validation repeated neural inference"))


def test_unselected_numerical_checkpoint_leaves_default_initial_context_unchanged(original, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_context as owner
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_config as configuration

    def forbidden(*args, **kwargs):
        pytest.fail("default preparation implicitly selected experimental numerical weights")

    monkeypatch.setattr(owner, "prepare_numerical_selection_context", forbidden)
    monkeypatch.setattr(configuration, "load_numerical_selection_config", forbidden)
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state)
    assert "numerical_selection_384" not in result
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert all(row["schema"] != "supervisor-learned-numerical-selection-summary@1" for row in loaded["summaries"])


def test_real_numerical_consumer_preserves_mismatch_abstention_and_cached_report(
        tmp_path, numerical_config, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_context as owner

    root = tmp_path / "source-repository"
    root.mkdir()
    sources = {"matching.py": MATCHING_SOURCE, "mismatching.py": MISMATCHING_SOURCE,
               "unsupported.py": UNSUPPORTED_SOURCE}
    for name, source in sources.items():
        (root / name).write_text(source)
    for args in (("init", "-q"), ("add", "."),
                 ("-c", "user.name=Qualification", "-c", "user.email=qualification@example.invalid",
                  "commit", "-qm", "independently authored inference fixtures")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    receipt = owner.prepare_numerical_selection_context(repository=root,
        source_hashes={name: _sha(source) for name, source in sources.items()},
        output=tmp_path / "numerical-owner", config_path=numerical_config,
        expected_config_sha256=_sha(numerical_config.read_bytes()))
    inference_path = Path(receipt["output"]) / "inference.json"
    inference = json.loads(inference_path.read_bytes())
    assert inference["native_worker_executed"] is inference["inference_executed"] is True
    report = inference["report"]
    output = report["output"]
    assert output["model_loads"] == 1
    assert report["key"]["head_sha256"] == HEAD_SHA256
    assert report["key"]["projection_sha256"] == PROJECTION_SHA256
    by_path = {row["path"]: row for row in output["rows"]}
    assert set(by_path) == {"matching.py", "mismatching.py"}
    matched = by_path["matching.py"]["candidate"]
    mismatched = by_path["mismatching.py"]["candidate"]
    assert matched["source_candidate_match"] is True
    assert matched["prediction"]["candidate_ir"]["document"]["operator"] == "numpy.argmax"
    assert matched["prediction"]["candidate_ir"]["document"]["operand_ids"][-1] == "expr:parameter:0"
    assert mismatched["source_candidate_match"] is False
    assert mismatched["status"] == "fail_open_source_mismatch"
    assert mismatched["candidate_replaced"] is False
    assert mismatched["continue_planning"] is True
    assert mismatched["matched_projection_refs"] is None
    # The original, disagreed-with prediction remains available as advice.
    assert mismatched["prediction"]["candidate_ir"] is not None
    families = {row["family"] for row in matched["matched_projection_refs"]["artifacts"]}
    assert {"program", "first_order", "higher_order"} <= families
    assert matched["matched_projection_refs"]["pending_obligation_ids"]
    unsupported = next(row for row in report["preparation"]["files"] if row["path"] == "unsupported.py")
    assert unsupported["disposition"] == "unsupported_profile"
    assert unsupported["reason"]
    assert report["coverage"]["source_aligned_fragments"] == 1
    assert report["coverage"]["source_mismatches"] == 1
    assert report["coverage"]["selected_units"] == 2
    assert output["embedding_receipt"]
    for candidate in (matched, mismatched):
        assert candidate["model_inference_executed"] is True
        assert candidate["proof_authority"] is candidate["source_semantics_verified"] is False
        assert candidate["source_obligations_discharged"] == 0
    assert receipt["training_steps"] == receipt["provider_calls"] == 0
    samples = receipt["summary"]["candidates"]
    aligned_sample = next(row for row in samples if row["path"] == "matching.py")
    assert json.loads(aligned_sample["candidate_ir_json"]) == matched["prediction"]["candidate_ir"]
    mismatch_sample = next(row for row in samples if row["path"] == "mismatching.py")
    assert json.loads(mismatch_sample["candidate_ir_json"]) == mismatched["prediction"]["candidate_ir"]
    assert receipt["summary"]["omitted_candidates"] == 0
    before = inference_path.read_bytes()
    _forbid_new_inference(monkeypatch)
    owner.validate_numerical_selection_context(repository=root, expected_receipt=receipt)
    owner.validate_numerical_selection_context(repository=root, expected_receipt=receipt)
    assert inference_path.read_bytes() == before
    # Hash-validating the retained artifacts cannot silently accept their loss.
    inference_path.write_bytes(before + b" ")
    with pytest.raises(ValueError):
        owner.validate_numerical_selection_context(repository=root, expected_receipt=receipt)


def test_cached_numerical_checkpoint_reaches_planning_worker_and_historical_audit(
        original, numerical_config, provider, monkeypatch):
    root, instruction, state = original
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(state / "orchestration"))
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    (root / "bottle.py").write_text(MATCHING_SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    indexed = prep.initial_context(state=state, numerical_selection_config=numerical_config,
        numerical_formalization_profile="python-numerical-structure@1")
    learned = indexed["numerical_selection_384"]
    assert learned["checkpoint_file_sha256"] == RELEASE_SHA256
    assert learned["summary"]["head_sha256"] == HEAD_SHA256
    assert learned["summary"]["projection_sha256"] == PROJECTION_SHA256
    assert learned["training_steps"] == learned["provider_calls"] == 0
    assert learned["source_hashes"] == {
        name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()
    }
    assert learned["hydration"]["ducklake_verified"] is True
    samples = learned["summary"]["candidates"]
    assert len(samples) == 1 and samples[0]["status"] == "source_aligned_fragment"
    candidate = json.loads(samples[0]["candidate_ir_json"])
    assert candidate["document"]["operator"] == "numpy.argmax"
    assert candidate["document"]["operand_ids"][-1] == "expr:parameter:0"
    _forbid_new_inference(monkeypatch)
    planning_calls = []

    def planner(prompt, **kwargs):
        planning_calls.append(prompt)
        assert RELEASE_SHA256 in prompt
        assert json.dumps(samples[0]["candidate_ir_json"]) in prompt
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)

    _version(monkeypatch)
    assert prep.plan(state=state, provider_callable=planner)["qualified"] is True
    assert len(planning_calls) == 1
    context = prep.context(state=state)
    assert context["numerical_selection_384"] == learned
    bundle = context["context_bundle"]
    task = json.loads((root / bundle["artifact"]).read_bytes())["tasks"][0]
    daemon = TodoImplementationDaemon(todo_path=root / ".runtime/numerical.todo.md", repo_root=root,
        state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
        events_path=state / "worker/events.jsonl", task_header_prefix="## TB-",
        implementation_log_dir=state / "launch/implementation-logs")
    daemon._task_context_nomination_bundle = bundle
    portal = PortalTask(task_id=task["task_id"], canonical_task_cid=task["task_cid"],
        title="Numerical advisory delivery", status="ready", completion="manual", priority="P2",
        track="implementation", outputs=["bottle.py", "report.jsonl"], validation=[],
        acceptance="Observe the exact task-bound numerical candidate.",
        metadata={"database task cid": task["task_cid"]})
    try:
        prompt = daemon._build_implementation_prompt(portal, attempt=1)
        daemon._persist_implementation_context_receipt(portal, attempt=1)
        wire, _ = json.JSONDecoder().raw_decode(prompt)
        refs = [row for row in wire["evidence"] if row["kind"] == "source-formalization-context"]
        assert refs and all(row["metadata"]["required"] for row in refs)
        evidence = json.loads("".join(row["summary"] for row in refs))
        selected = [row for row in evidence["contexts"] if row["summary"] == learned["summary"]]
        assert len(selected) == 1
        assert selected[0]["kind"] == "learned-numerical-selection-context"
        assert evidence["proof_authority"] is evidence["execution_authority"] is False
        workspace_root = state / "allocated"
        workspace_root.mkdir()
        workspace = workspace_root / "worker"
        subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)],
            check=True, capture_output=True)
        for name in learned["source_hashes"]:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        monkeypatch.chdir(workspace)
        provider_state, observed = provider
        checkpoint = Path(json.loads(numerical_config.read_bytes())["checkpoint_path"])
        # Every negative precedes the first provider call and restores bytes.
        for path in (root / "bottle.py", workspace / "bottle.py", checkpoint, numerical_config,
                     Path(learned["output"]) / "receipt.json"):
            before = path.read_bytes()
            try:
                path.write_bytes(before + b" ")
                with pytest.raises(ValueError):
                    worker_router.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
                        max_output_tokens=128, semantic_repository=root)
                assert observed == [], str(path)
            finally:
                path.write_bytes(before)
        provider_state["after_generate"] = lambda: (workspace / "bottle.py").write_text(
            MATCHING_SOURCE + "\n# authored worker edit\n")
        output, receipt = worker_router.run(prompt=prompt, provider="codex_cli", model="pinned",
            timeout=1, max_output_tokens=128, semantic_repository=root)
        receipt = {"phase": "coding", **receipt}
        assert output == "literal summary" and len(observed) == 1
        dispatch = receipt["source_formalization_context"]
        assert dispatch["validation_phase"] == "before_provider"
        assert selected[0]["kind"] in dispatch["context_kinds"]
        assert dispatch["neural_inference_replayed"] is dispatch["proof_authority"] is False
        assert receipt["model_prompt_sha256"] == _sha(observed[0][0])
        # Translation's context remains exact even though the provider edits
        # the allocated source after all pre-dispatch checks have completed.
        encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
        _, advisory = worker_router.render_model_prompt(prompt=encoded.provider_prompt,
            purpose="coding", workspace=workspace, semantic_transport=True)
        assert codec.restore_semantic_router_prompt(provider_prompt=observed[0][0][len(advisory):],
            table=encoded.table, repository=root) == prompt
        (root / "bottle.py").write_text(MATCHING_SOURCE + "\n# later canonical edit\n")
        with pytest.raises(ValueError):
            worker_router.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
                max_output_tokens=128, semantic_repository=root)
        historical = audit.collect_terminal_context_audit(state=state, receipts=[receipt],
            workspace_root=workspace_root, timeout_seconds=10)
        assert historical["all_observed_coding_inputs_verified"] is True, historical
        assert historical["provider_calls"] == 0
        assert historical["proof_authority"] is historical["completion_authority"] is False
        changed = deepcopy(receipt)
        del changed["source_formalization_context"]
        rejected = audit.audit_terminal_context(state=state, receipts=[changed], workspace_root=workspace_root)
        assert rejected["all_observed_coding_inputs_verified"] is not True
        assert len(observed) == 1
    finally:
        daemon.close_event_runtime()
