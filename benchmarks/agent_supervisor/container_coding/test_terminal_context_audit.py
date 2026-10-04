"""Native compilation/round-trip audit tests; no provider or admission claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def compiled(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    folder = state / "launch/implementation-logs"
    folder.mkdir(parents=True)
    query = "private source query must not be exported: αβ 雪\n " * 140
    digest = sha("original bottle source")
    expected = {"task_cid": "cid:task", "semantic_root_cid": "cid:semantic",
                "world_snapshot_cid": "cid:world", "index_id": "cid:vectors"}
    prepared = {"query": query, "spec": {"task_key": "TB-CODE-TASK"},
                "worker_inputs": ["bottle.py"], "manifest": {"payload": {
                    "sources": {"bottle.py": {"sha256": digest}}}}}
    payloads = {
        "semantic-context": {"task_id": "TB-CODE-TASK", "semantic_root_cid": "cid:semantic",
            "manifest": {"bottle.py": {"sha256": digest}}, "completion_authority": False},
        "code-retrieval-context": {"task_id": "TB-CODE-TASK", "index_id": "cid:vectors",
            "result_id": "cid:result", "query_id": "cid:query", "query_text": query,
            "source_sha256": {"bottle.py": digest}, "status": "current", "nomination_only": True,
            "semantic_authority": False, "execution_authority": False, "completion_authority": False},
        "intent-world-context": {"semantic_root_cid": "cid:semantic", "world_snapshot_cid": "cid:world",
            "plan_projection_cid": "cid:projection", "intent_freshness_checked": False,
            "tasks": [{"task_cid": "cid:task", "body": {"local_planning_contract_cid": "cid:contract"}}],
            "execution_authority": False, "completion_authority": False},
    }
    refs = []
    for i, (kind, payload) in enumerate(payloads.items()):
        refs.extend(build_text_context_references(json.dumps(payload, ensure_ascii=False),
            reference_prefix=f"context-{i}", kind=kind, chunk_bytes=613,
            required=True))
    compiled = ContextCompiler(ContextBudget(max_input_tokens=20000, max_items=64,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:fixture", tree_id="tree:fixture", objective_id="TB-CODE-TASK",
        objective_revision="sha256:task", policy_id="policy:fixture", policy_revision="sha256:policy",
        caller="supervisor:fixture", stage="implementation", goal={"id": "TB-CODE-TASK"},
        authority={"mode": "proposal"}, scope={"paths": ["bottle.py"]},
        acceptance={"criteria": ["pending public check"]}, evidence=refs)
    capsule = folder / "tb-code-task-base-context-capsule.json"
    capsule.write_text(compiled.capsule.to_json())
    (state / "context-result.json").write_text(json.dumps(expected))
    (state / "prepared.json").write_text(json.dumps(prepared))
    native = render_context_capsule(compiled.capsule)
    workspace_root = tmp_path / "removed-worktrees"
    # Finalization deliberately does not need a surviving worker/worktree.
    workspace = workspace_root / "workspace-fixture"
    model, header = render_model_prompt(prompt=native, purpose="coding", workspace=workspace)
    receipt = {"schema": "router-implementation-invocation@1", "phase": "coding",
        "invocation_id": "fixture-invocation", "purpose": "coding", "workspace": str(workspace),
        "prompt_sha256": sha(native), "prompt_bytes": len(native.encode()),
        "native_prompt_sha256": sha(native), "native_prompt_bytes": len(native.encode()),
        "model_prompt_sha256": sha(model), "model_prompt_bytes": len(model.encode()),
        "workspace_advisory_sha256": sha(header), "workspace_advisory_bytes": len(header.encode())}
    return {"state": state, "capsule": capsule, "receipts": [receipt],
            "workspace_root": workspace_root, "query": query}


def observe(fixture):
    return audit.audit_terminal_context(state=fixture["state"], receipts=fixture["receipts"],
                                        workspace_root=fixture["workspace_root"])


def test_exact_native_and_model_bytes_survive_worker_exit(compiled):
    result = observe(compiled)
    assert result["status"] == "verified_all_observed_model_inputs"
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["invocations"][0]["model_input_verified"] is True
    assert all(result["capsules"][0]["checks"].values())
    assert result["capsules"][0]["context_chunks"]["code-retrieval-context"]["count"] > 1
    assert not compiled["workspace_root"].exists()
    assert result["completion_authority"] is result["task_correctness_established"] is False
    assert result["capsules"][0]["intent_freshness_checked"] is False
    exported = json.dumps(result)
    assert compiled["query"] not in exported
    assert "private source query" not in exported
    assert compiled["receipts"][0]["workspace"] not in exported


def test_actual_bounded_child_process_roundtrip(compiled):
    result = audit.collect_terminal_context_audit(state=compiled["state"],
        receipts=compiled["receipts"], workspace_root=compiled["workspace_root"], timeout_seconds=10)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["provider_calls"] == 0


def test_owner_child_does_not_import_from_task_working_directory(compiled, tmp_path, monkeypatch):
    untrusted = tmp_path / "task-checkout"
    untrusted.mkdir()
    marker = untrusted / "unsafe-import-executed"
    (untrusted / "json.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('wrong import boundary')\n"
        "raise RuntimeError('task module imported by owner')\n")
    monkeypatch.chdir(untrusted)
    result = audit.collect_terminal_context_audit(state=compiled["state"],
        receipts=compiled["receipts"], workspace_root=compiled["workspace_root"], timeout_seconds=10)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert not marker.exists()


def test_wrong_receipt_workspace_cannot_match_model_input(compiled):
    compiled["receipts"][0]["workspace"] += "-foreign"
    result = observe(compiled)
    assert result["status"] == "verified_native_input_only"
    assert result["invocations"][0]["native_input_verified"] is True
    assert result["invocations"][0]["model_input_verified"] is False
    assert result["all_observed_coding_inputs_verified"] is False


def test_unmatched_retry_is_not_promoted_by_one_exact_match(compiled):
    retry = {**compiled["receipts"][0], "invocation_id": "retry-fixture",
        "native_prompt_sha256": "f" * 64, "prompt_sha256": "f" * 64}
    compiled["receipts"].append(retry)
    result = observe(compiled)
    assert result["status"] == "verified_some_observed_model_inputs"
    assert result["all_observed_coding_inputs_verified"] is False
    assert [row["model_input_verified"] for row in result["invocations"]] == [True, False]
    assert result["invocations"][1]["reason"] == "no_exact_stored_capsule_match"


def test_persisted_capsule_tampering_rejected(compiled):
    payload = json.loads(compiled["capsule"].read_text())
    payload["evidence"][0]["summary"] += " "
    compiled["capsule"].write_text(json.dumps(payload))
    result = observe(compiled)
    assert result["any_model_input_verified"] is False
    assert not result["matches"] and result["read_errors"]


@pytest.mark.parametrize("field", ["query", "source"])
def test_same_native_bytes_do_not_override_prepared_input_binding(compiled, field):
    path = compiled["state"] / "prepared.json"
    value = json.loads(path.read_text())
    if field == "query":
        value["query"] += "changed"
    else:
        value["manifest"]["payload"]["sources"]["bottle.py"]["sha256"] = "f" * 64
    path.write_text(json.dumps(value))
    result = observe(compiled)
    assert result["capsules"][0]["all_context_checks_passed"] is False
    assert result["any_model_input_verified"] is False


def test_missing_capsule_stays_unknown(compiled):
    compiled["capsule"].unlink()
    result = observe(compiled)
    assert result["reason"] == "native_capsule_unavailable"
    assert result["any_model_input_verified"] is None
    assert result["invocations"][0]["model_input_verified"] is None


@pytest.mark.parametrize("replacement", ["symlink", "fifo", "oversized"])
def test_nonregular_or_oversized_capsule_is_bounded_and_rejected(compiled, tmp_path, replacement):
    path = compiled["capsule"]
    raw = path.read_bytes()
    path.unlink()
    if replacement == "symlink":
        outside = tmp_path / "foreign.json"
        outside.write_bytes(raw)
        path.symlink_to(outside)
    elif replacement == "fifo":
        os.mkfifo(path)
    else:
        with path.open("wb") as stream:
            stream.truncate(audit.MAX_FILE_BYTES + 1)
    before = time.monotonic()
    result = observe(compiled)
    assert time.monotonic() - before < 2
    assert result["any_model_input_verified"] is False
    assert result["read_errors"] and not result["matches"]


def test_directory_replaced_after_inventory_is_not_followed(compiled, tmp_path, monkeypatch):
    original = audit._capsule_paths
    parent = compiled["capsule"].parent
    outside = tmp_path / "foreign-directory"
    def swapped(fd):
        paths = original(fd)
        parent.rename(outside)
        parent.symlink_to(outside, target_is_directory=True)
        return paths
    monkeypatch.setattr(audit, "_capsule_paths", swapped)
    result = observe(compiled)
    assert result["any_model_input_verified"] is False
    assert result["read_errors"] and not result["matches"]


def test_state_parent_symlink_refused(compiled, tmp_path):
    linked = tmp_path / "state-link"
    linked.symlink_to(compiled["state"], target_is_directory=True)
    result = audit.collect_terminal_context_audit(state=linked, receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], timeout_seconds=10)
    assert result["status"] == "unknown"
    assert result["reason"] == "audit_unavailable"


def test_inventory_bound_abstains(compiled, monkeypatch):
    monkeypatch.setattr(audit, "MAX_INVENTORY_ENTRIES", 1)
    with pytest.raises(ValueError, match="inventory"):
        observe(compiled)


def test_timeout_is_unknown_and_does_not_modify_receipts(compiled, monkeypatch):
    before = copy.deepcopy(compiled["receipts"])
    def timeout(*args, **kwargs):
        assert args[0][:3] == [sys.executable, "-B", "-P"]
        assert kwargs["timeout"] == .5
        raise subprocess.TimeoutExpired("audit", .5)
    monkeypatch.setattr(audit.subprocess, "run", timeout)
    result = audit.collect_terminal_context_audit(state=compiled["state"],
        receipts=compiled["receipts"], workspace_root=compiled["workspace_root"], timeout_seconds=.5)
    assert result["status"] == "unknown" and result["reason"] == "audit_timeout"
    assert compiled["receipts"] == before


def test_finalization_keeps_usage_and_completion_on_audit_failure(compiled, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _final_context_audit
    report = {"arm": "full", "task_completed": True, "stop": {"status": "succeeded"},
              "provider_invocations": [{**compiled["receipts"][0], "usage": {"input_tokens": 123}}]}
    prior = copy.deepcopy(report)
    def failure(**kwargs):
        assert 0 < kwargs["timeout_seconds"] <= 1
        raise OSError("private detail must not escape")
    monkeypatch.setattr(audit, "collect_terminal_context_audit", failure)
    _final_context_audit(report, state=compiled["state"], deadline=time.monotonic() + 2)
    assert {key: report[key] for key in prior} == prior
    assert report["context_input_audit"]["status"] == "unknown"
    assert "private detail" not in json.dumps(report)


def test_finalization_without_remaining_budget_does_not_spawn(compiled, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _final_context_audit
    report = {"arm": "full", "provider_invocations": compiled["receipts"]}
    def forbidden(*args, **kwargs):
        raise AssertionError("child started after deadline")
    monkeypatch.setattr(audit.subprocess, "run", forbidden)
    _final_context_audit(report, state=compiled["state"], deadline=time.monotonic())
    assert report["context_input_audit"]["reason"] == "audit_budget_unavailable"
