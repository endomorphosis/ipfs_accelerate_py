"""Cached inference reaches the worker and survives historical input auditing.

Source384 CPU/GTE inference, numerical projections, DuckLake persistence,
signed planning admission, context compilation, the shared router, and the
historical auditor are real. Planning/provider replies are authored fixtures;
this is neither a benchmark score nor a numerical correctness proof.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_context import selected_config  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as worker_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon
from test.api.test_semantic_router_integration import provider  # noqa: F401


SOURCE = """import numpy as np

def choose(values):
    return np.argmax(values)

def compare(left: int, right: int) -> bool:
    return left < right
"""


def test_real_cached_formalization_dispatch_has_historical_audit_after_source_edits(
        original, selected_config, provider, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as native

    root, instruction, state = original
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(state / "orchestration"))
    monkeypatch.delenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", raising=False)
    (root / "bottle.py").write_text(SOURCE)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    indexed = prep.initial_context(state=state, source384_config=selected_config,
        numerical_formalization_profile="python-numerical-structure@1")
    learned = indexed["source384_context"]
    inference = json.loads((Path(learned["output"]) / "inference.json").read_bytes())
    assert inference["native_worker_executed"] is inference["inference_executed"] is True
    assert inference["report"]["output"]["model_loads"] == 1
    assert learned["training_steps"] == 0
    monkeypatch.setattr(native, "_worker", lambda *a, **k: pytest.fail("audit replayed neural inference"))
    planning_calls = []
    def planner(prompt, **kwargs):
        planning_calls.append(prompt)
        assert learned["checkpoint_sha256"] in prompt
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)
    _version(monkeypatch)
    assert prep.plan(state=state, provider_callable=planner)["qualified"] is True
    assert len(planning_calls) == 1
    context = prep.context(state=state)
    bundle = context["context_bundle"]
    task = json.loads((root / bundle["artifact"]).read_text())["tasks"][0]
    formal = context["numerical_formalization"]
    assert formal["hydration"]["ducklake_verified"] is True
    assert formal["summary"]["source_unit_links"]
    daemon = TodoImplementationDaemon(todo_path=root / ".runtime/historical.todo.md", repo_root=root,
        state_path=state / "worker/tasks.json", strategy_path=state / "worker/strategy.json",
        events_path=state / "worker/events.jsonl", task_header_prefix="## TB-",
        implementation_log_dir=state / "launch/implementation-logs")
    daemon._task_context_nomination_bundle = bundle
    portal = PortalTask(task_id=task["task_id"], canonical_task_cid=task["task_cid"],
        title="Historical dispatch qualification", status="ready", completion="manual", priority="P2",
        track="implementation", outputs=["bottle.py", "report.jsonl"], validation=[],
        acceptance="Observe the exact task-bound candidate context.", metadata={"database task cid": task["task_cid"]})
    try:
        prompt = daemon._build_implementation_prompt(portal, attempt=1)
        daemon._persist_implementation_context_receipt(portal, attempt=1)
        wire, _ = json.JSONDecoder().raw_decode(prompt)
        rows = [row for row in wire["evidence"] if row["kind"] == "source-formalization-context"]
        envelope = json.loads("".join(row["summary"] for row in rows))
        assert [row["kind"] for row in envelope["contexts"]] == [
            "source384-candidate-context", "numerical-formalization-context"]
        workspace_root = state / "allocated"
        workspace_root.mkdir()
        workspace = workspace_root / "worker"
        subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
        for name in learned["source_hashes"]:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("CODEX_HOME", str(state / "empty-worker-home"))
        provider_state, observed = provider
        # Edits occur in the provider callback, after pre-dispatch validation.
        provider_state["after_generate"] = lambda: (workspace / "bottle.py").write_text(SOURCE + "\n# worker edit\n")
        output, receipt = worker_router.run(prompt=prompt, provider="codex_cli", model="pinned",
            timeout=1, max_output_tokens=128, semantic_repository=root)
        # The supervisor's retained receipt collector adds its observed phase.
        receipt = {"phase": "coding", **receipt}
        assert output == "literal summary" and len(observed) == 1
        assert receipt["source_formalization_context"]["validation_phase"] == "before_provider"
        assert receipt["source_formalization_context"]["context_kinds"] == [
            "source384-candidate-context", "numerical-formalization-context"]
        assert receipt["model_prompt_sha256"] == hashlib.sha256(observed[0][0].encode()).hexdigest()
        (root / "bottle.py").write_text(SOURCE + "\n# later canonical publication\n")
        # A historical audit joins retained artifacts. A new live dispatch must
        # still refuse the same stale source evidence before another provider.
        with pytest.raises(ValueError):
            worker_router.run(prompt=prompt, provider="codex_cli", model="pinned", timeout=1,
                max_output_tokens=128, semantic_repository=root)
        assert len(observed) == 1

        def observe(candidate=receipt):
            return audit.audit_terminal_context(state=state, receipts=[candidate], workspace_root=workspace_root)

        historical = observe()
        assert historical["all_observed_coding_inputs_verified"] is True, historical
        assert historical["provider_calls"] == 0
        assert historical["completion_authority"] is historical["task_correctness_established"] is False
        exported = json.dumps(historical)
        assert "candidate_ir_json" not in exported and "return np.argmax" not in exported
        assert str(workspace) not in exported
        # Exercise the real bounded collector: optional dispatch observations
        # must survive its independent receipt projection into the child.
        child = audit.collect_terminal_context_audit(state=state, receipts=[receipt],
            workspace_root=workspace_root, timeout_seconds=10)
        assert child["all_observed_coding_inputs_verified"] is True, child
        retained = state / "retained-final-audit.json"
        retained.write_text(json.dumps(child, sort_keys=True))
        from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _audit_observation
        compact = _audit_observation({"context_input_audit": json.loads(retained.read_text())})
        delivered = compact["source_formalization"]
        assert len(delivered) == 1 and delivered[0]["reported_model_input_verified"] is True
        assert delivered[0]["matched_invocation_ids"] == [receipt["invocation_id"]]
        assert delivered[0]["worker_evidence_sha256"] == receipt["source_formalization_context"]["worker_evidence_sha256"]
        assert delivered[0]["current_sources_checked"] is delivered[0]["proof_authority"] is False

        for mutation in ("missing_observation", "wrong_evidence", "wrong_task", "promoted_authority"):
            changed = deepcopy(receipt)
            if mutation == "missing_observation":
                del changed["source_formalization_context"]
            elif mutation == "wrong_evidence":
                changed["source_formalization_context"]["worker_evidence_sha256"] = "0" * 64
            elif mutation == "wrong_task":
                changed["source_formalization_context"]["task_cid"] = "task:foreign"
            else:
                changed["source_formalization_context"]["proof_authority"] = True
            assert observe(changed)["all_observed_coding_inputs_verified"] is not True, mutation

        # A formerly valid provider receipt cannot hide retained selection or
        # owner receipt damage. Restore fixture bytes between each control.
        for path in (root / bundle["artifact"], Path(learned["output"]) / "receipt.json",
                     Path(formal["output"]) / "receipt.json"):
            before = path.read_bytes()
            try:
                path.write_bytes(before + b" ")
                assert observe()["all_observed_coding_inputs_verified"] is not True, path
                path.unlink()
                assert observe()["all_observed_coding_inputs_verified"] is not True, path
            finally:
                path.write_bytes(before)
        assert observe()["all_observed_coding_inputs_verified"] is True
        assert len(observed) == 1
    finally:
        daemon.close_event_runtime()
